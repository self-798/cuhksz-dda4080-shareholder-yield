import pandas as pd, numpy as np, warnings, json, os
warnings.filterwarnings('ignore')
os.makedirs('pict', exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

print("Loading data...")
price_cols = pd.read_csv('data/hk_price.csv', usecols=['date', 'sid', 'close', 'AdjClose'])
price_cols['date'] = pd.to_datetime(price_cols['date'])
price_cols['date_m'] = price_cols['date'].dt.to_period('M')
monthly_price = price_cols.sort_values('date').groupby(['date_m', 'sid']).last().reset_index()
monthly_price_pivot = monthly_price.pivot(index='date_m', columns='sid', values='AdjClose')
monthly_ret = monthly_price_pivot.pct_change(fill_method=None).shift(-1)

with pd.HDFStore('data/hk_shares.h5') as store:
    shares = store.get(store.keys()[0]).reset_index()
shares['sid'] = shares['order_book_id'].str[1:5] + '.HK'
shares['date_m'] = pd.to_datetime(shares['date']).dt.to_period('M')
shares_monthly = shares.sort_values('date').groupby(['date_m', 'sid']).last().reset_index()
mcap_df = pd.merge(monthly_price[['date_m', 'sid', 'close']],
                   shares_monthly[['date_m', 'sid', 'total']],
                   on=['date_m', 'sid'], how='inner')
mcap_df['mcap'] = mcap_df['close'] * mcap_df['total']
mcap_pivot = mcap_df.pivot(index='date_m', columns='sid', values='mcap')

price = pd.read_csv('data/hk_price.csv', usecols=['date', 'sid', 'close', 'amount'])
price['date'] = pd.to_datetime(price['date'])
price_pivot_full = price.pivot(index='date', columns='sid', values='close').sort_index()
ret_pivot = price_pivot_full.pct_change()
vol_pivot = ret_pivot.rolling(window=252, min_periods=100).std() * np.sqrt(252)
vol_monthly = vol_pivot.resample('ME').last()
vol_stacked = vol_monthly.stack().reset_index()
vol_stacked.columns = ['date', 'sid', 'volatility']
vol_stacked['date_m'] = vol_stacked['date'].dt.to_period('M')

amount_pivot = price.pivot(index='date', columns='sid', values='amount').sort_index()
adv_pivot = amount_pivot.rolling(window=63, min_periods=21).mean()
adv_monthly = adv_pivot.resample('ME').last()
adv_stacked = adv_monthly.stack().reset_index()
adv_stacked.columns = ['date', 'sid', 'adtv_3m']
adv_stacked['date_m'] = adv_stacked['date'].dt.to_period('M')

dy = pd.read_hdf('data/hk_dividendyield.h5')
if isinstance(dy, pd.DataFrame): dy = dy.iloc[:, 0]
dy = dy.dropna().reset_index()
dy.columns = ['date', 'sid', 'dy'] if dy.shape[1] == 3 else ['date', 'dy']
dy['date'] = pd.to_datetime(dy['date'])
dy['date_m'] = dy['date'].dt.to_period('M')
dy_pivot = dy.pivot(index='date', columns='sid', values='dy').sort_index()

hsci = pd.read_csv('data/HSCI.csv')
hsci['date'] = pd.to_datetime(hsci['date'])
hsci['is_hsci'] = 1
hsci = hsci.drop_duplicates(subset=['date', 'sid'])
hsci['date_m'] = hsci['date'].dt.to_period('M')

buyback = pd.read_csv('data/em_buyback_filtered.csv')
buyback['date'] = pd.to_datetime(buyback['日期'])
buyback['sid'] = buyback['股票代码'].astype(str).str.zfill(5).str[1:5] + '.HK'
buyback['date_m'] = buyback['date'].dt.to_period('M')
monthly_buyback = buyback.groupby(['date_m', 'sid'])['回购总额'].sum().reset_index()
monthly_buyback_pivot = monthly_buyback.pivot(index='date_m', columns='sid', values='回购总额').fillna(0).sort_index()
buyback_36m_pivot = monthly_buyback_pivot.rolling(window=36, min_periods=12).sum()
buyback_36m_stacked = buyback_36m_pivot.stack().reset_index()
buyback_36m_stacked.columns = ['date_m', 'sid', 'buyback_36m']

df_factors = hsci[['date_m', 'sid', 'is_hsci']].merge(
    mcap_df[['date_m', 'sid', 'mcap']], on=['date_m', 'sid'], how='inner'
).merge(vol_stacked[['date_m', 'sid', 'volatility']], on=['date_m', 'sid'], how='inner'
).merge(adv_stacked[['date_m', 'sid', 'adtv_3m']], on=['date_m', 'sid'], how='inner'
).merge(dy[['date_m', 'sid', 'dy']], on=['date_m', 'sid'], how='left'
).merge(buyback_36m_stacked, on=['date_m', 'sid'], how='left')
df_factors['buyback_36m'] = df_factors['buyback_36m'].fillna(0)
df_factors = df_factors.drop_duplicates(subset=['date_m', 'sid'])

# HSI benchmark
hsi = pd.read_csv('data/HSI_index.csv')
hsi['date'] = pd.to_datetime(hsi['date'])
hsi['date_m'] = hsi['date'].dt.to_period('M')
hsi_monthly = hsi.sort_values('date').groupby('date_m').last().reset_index()
hsi_monthly = hsi_monthly.set_index('date_m')['close']
hsi_ret = hsi_monthly.pct_change().shift(-1).dropna()
hsi_ret.index = hsi_ret.index + 1
BACKTEST_START = pd.Period('2012-01', 'M')
hsi_ret = hsi_ret[hsi_ret.index >= BACKTEST_START + 1]

# HKDC benchmark (real)
hkdc = pd.read_csv('data/hk_stock_connect_high_div.csv')
hkdc['date'] = pd.to_datetime(hkdc['date'])
hkdc['date_m'] = hkdc['date'].dt.to_period('M')
hkdc_monthly = hkdc.sort_values('date').groupby('date_m').last().reset_index()
hkdc_monthly = hkdc_monthly.set_index('date_m')['close']
hkdc_ret = hkdc_monthly.pct_change().shift(-1).dropna()
hkdc_ret.index = hkdc_ret.index + 1
hkdc_ret = hkdc_ret[hkdc_ret.index >= BACKTEST_START + 1]

print("Data loaded.")

# Industry mapping
def get_industry(sid):
    code = int(sid.replace('.HK', ''))
    if 1 <= code <= 99: return 'Financials_Utilities'
    elif 100 <= code <= 999: return 'Property_Industrial'
    elif 1000 <= code <= 1999: return 'Consumer_Retail'
    elif 2000 <= code <= 2999: return 'Diversified'
    elif 3000 <= code <= 3999: return 'Energy_Materials'
    elif 6000 <= code <= 6999: return 'Banking'
    elif 7000 <= code <= 7999: return 'Technology'
    else: return 'Other_Misc'

def select_stocks_neutral(group, do_neutralize=False, use_lowvol=False):
    current_m = group.name
    if isinstance(current_m, tuple): current_m = current_m[0]
    end_date = current_m.to_timestamp(how='end')
    dy_36m = dy_pivot.loc[:end_date].tail(36)
    price_36m = price_pivot_full.reindex(dy_36m.index, method='ffill')
    has_div = dy_36m.columns[dy_36m.gt(0).any(axis=0)] if len(dy_36m) > 0 else pd.Index([])
    pool = group[(group['sid'].isin(has_div)) | (group['buyback_36m'] > 0)]
    adtv_th = group['adtv_3m'].quantile(0.2)
    mcap_th = group['mcap'].quantile(0.2)
    pool = pool[(pool['adtv_3m'] >= adtv_th) & (pool['mcap'] >= mcap_th)]
    if len(pool) == 0: return []
    pool = pool.copy()
    dps_36m = price_36m * dy_36m
    current_prices = price_36m.iloc[-1] if len(price_36m) > 0 else pd.Series(dtype=float)
    if len(dps_36m) > 0 and len(current_prices) > 0:
        pool['div_yield'] = pool['sid'].map(dps_36m.mean() / current_prices).fillna(0)
    else:
        pool['div_yield'] = 0.0
    pool['buyback_yield'] = ((pool['buyback_36m'] / 3) / pool['mcap']) * 100
    pool['buyback_yield'] = pool['buyback_yield'].replace([np.inf, -np.inf, np.nan], 0)
    pool['score_raw'] = pool['div_yield'] + pool['buyback_yield']
    pool = pool.dropna(subset=['score_raw'])
    if len(pool) == 0: return []
    cutoff = pool['score_raw'].quantile(0.9)
    pool = pool[pool['score_raw'] <= cutoff]
    if len(pool) == 0: return []
    if do_neutralize:
        pool['industry'] = pool['sid'].apply(get_industry)
        pool['score'] = pool.groupby('industry')['score_raw'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0)
    else:
        pool['score'] = pool['score_raw']
    pool = pool.dropna(subset=['score'])
    n_top = max(10, int(len(pool) * 0.20))
    pool = pool.nlargest(n_top, 'score')
    n_final = min(20, len(pool))
    if use_lowvol:
        return pool.nsmallest(n_final, 'volatility')['sid'].tolist()
    else:
        return pool.nlargest(n_final, 'score')['sid'].tolist()

print("Running strategies...")
sigs_B_raw = df_factors.groupby('date_m', group_keys=True).apply(lambda g: select_stocks_neutral(g, False, False))
sigs_B_neu = df_factors.groupby('date_m', group_keys=True).apply(lambda g: select_stocks_neutral(g, True, False))
sigs_C_raw = df_factors.groupby('date_m', group_keys=True).apply(lambda g: select_stocks_neutral(g, False, True))
sigs_C_neu = df_factors.groupby('date_m', group_keys=True).apply(lambda g: select_stocks_neutral(g, True, True))

def run_bt(signals):
    dates = signals.index[signals.index >= BACKTEST_START]
    rets = []
    current = []
    for t_idx, date_m in enumerate(dates):
        if date_m not in monthly_ret.index: continue
        if t_idx % 3 == 0:
            new_stocks = signals.loc[date_m]
            if len(new_stocks) > 0: current = new_stocks
        if len(current) == 0: continue
        nxt = monthly_ret.loc[date_m].reindex(current).fillna(0)
        rets.append((date_m + 1, nxt.mean()))
    return pd.Series([r for _, r in rets], index=[d for d, _ in rets])

bt_B_raw = run_bt(sigs_B_raw)
bt_B_neu = run_bt(sigs_B_neu)
bt_C_raw = run_bt(sigs_C_raw)
bt_C_neu = run_bt(sigs_C_neu)

def calc_metrics(returns):
    ann_ret = returns.mean() * 12
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else np.nan
    cum_ret = (1 + returns).cumprod()
    max_dd = (cum_ret / cum_ret.cummax().clip(lower=1.0) - 1).min()
    return ann_ret, ann_vol, sharpe, max_dd

# ====== ABSOLUTE PERFORMANCE ======
print("\n" + "=" * 90)
print("STRATEGY PERFORMANCE: RAW vs INDUSTRY-NEUTRALIZED")
print("=" * 90)
print(f"{'Strategy':<22} {'Ann Ret':<10} {'Ann Vol':<10} {'Sharpe':<8} {'Max DD':<10} {'FF3 Alpha':<12}")
print("-" * 75)
for name, bt in [('B 原始 (Raw)', bt_B_raw), ('B 中性化 (Neutral)', bt_B_neu),
                 ('C 原始 (Raw)', bt_C_raw), ('C 中性化 (Neutral)', bt_C_neu)]:
    ar, av, sh, md = calc_metrics(bt)
    print(f"{name:<22} {ar:>8.2%}   {av:>8.2%}   {sh:>6.2f}   {md:>8.2%}")

# ====== EXCESS OVER HSI ======
print("\n\n" + "=" * 90)
print("EXCESS RETURN ANALYSIS (vs HSI) -- 指数增强核心指标")
print("=" * 90)
print(f"{'Strategy':<22} {'Ann Excess':<12} {'Tracking Err':<14} {'Info Ratio':<12} {'Hit Rate':<10} {'Exc MaxDD':<10} {'Cum Exc':<8}")
print("-" * 90)

for name, bt in [('B 原始', bt_B_raw), ('B 中性化', bt_B_neu), ('C 原始', bt_C_raw), ('C 中性化', bt_C_neu)]:
    common = bt.index.intersection(hsi_ret.index)
    excess = bt.reindex(common) - hsi_ret.reindex(common)
    ann_excess = excess.mean() * 12
    ann_te = excess.std() * np.sqrt(12)
    ir = ann_excess / ann_te if ann_te > 0 else 0
    hit_rate = (excess > 0).mean()
    cum_excess = 1 + excess.cumsum()
    max_dd_excess = (cum_excess / cum_excess.cummax().clip(lower=1.0) - 1).min()
    print(f"{name:<22} {ann_excess:>+8.2%}     {ann_te:>9.2%}      {ir:>7.2f}     {hit_rate:>7.1%}    {max_dd_excess:>8.2%}    {cum_excess.iloc[-1]-1:>+6.1%}")

# ====== EXCESS OVER HKDC ======
print("\n\n" + "=" * 90)
print("EXCESS RETURN ANALYSIS (vs 港股通高股息指数)")
print("=" * 90)
print(f"{'Strategy':<22} {'Ann Excess':<12} {'Tracking Err':<14} {'Info Ratio':<12} {'Hit Rate':<10}")
print("-" * 75)
for name, bt in [('B 原始', bt_B_raw), ('B 中性化', bt_B_neu), ('C 原始', bt_C_raw), ('C 中性化', bt_C_neu)]:
    common = bt.index.intersection(hkdc_ret.index)
    excess = bt.reindex(common) - hkdc_ret.reindex(common)
    ann_excess = excess.mean() * 12
    ann_te = excess.std() * np.sqrt(12)
    ir = ann_excess / ann_te if ann_te > 0 else 0
    hit_rate = (excess > 0).mean()
    print(f"{name:<22} {ann_excess:>+8.2%}     {ann_te:>9.2%}      {ir:>7.2f}     {hit_rate:>7.1%}")

# ====== ROLLING EXCESS ======
print("\n\nPlotting rolling 12-month excess returns...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, (strat_pair, title) in zip(axes, [
    (('B 原始', bt_B_raw, 'B 中性化', bt_B_neu), 'Strategy B: Raw vs Neutral'),
    (('C 原始', bt_C_raw, 'C 中性化', bt_C_neu), 'Strategy C: Raw vs Neutral')
]):
    n1, bt1, n2, bt2 = strat_pair
    for name, bt, color, ls in [(n1, bt1, 'blue', '-'), (n2, bt2, 'red', '--')]:
        common = bt.index.intersection(hsi_ret.index)
        excess = bt.reindex(common) - hsi_ret.reindex(common)
        roll = excess.rolling(12).mean() * 12
        ax.plot(roll.index.to_timestamp(), roll.values, label=name, color=color, linestyle=ls, linewidth=1.5)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title(title + '\n12-Month Rolling Excess Return over HSI', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rolling 12M Excess Return', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

plt.tight_layout()
plt.savefig('pict/v4_plot_neutral_excess_rolling.png', dpi=150, bbox_inches='tight')
plt.close()

# ====== ANNUAL EXCESS BAR CHART ======
fig, ax = plt.subplots(figsize=(14, 6))
years = sorted(set(hsi_ret.index.year) & set(bt_B_raw.index.year) & set(bt_B_neu.index.year))
x = np.arange(len(years))
width = 0.2
for i, (name, bt, color) in enumerate([
    ('B Raw', bt_B_raw, 'steelblue'), ('B Neutral', bt_B_neu, 'coral'),
    ('C Raw', bt_C_raw, 'darkgreen'), ('C Neutral', bt_C_neu, 'orange')
]):
    excess_annual = []
    for y in years:
        mask = bt.index.year == y
        common_hsi = bt.index[mask].intersection(hsi_ret.index)
        if len(common_hsi) > 0:
            exc = bt.reindex(common_hsi) - hsi_ret.reindex(common_hsi)
            excess_annual.append((1+exc).prod()-1)
        else:
            excess_annual.append(np.nan)
    ax.bar(x + i*width, [e*100 for e in excess_annual], width, label=name, color=color, alpha=0.85)

ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xticks(x + width*1.5)
ax.set_xticklabels([str(y) for y in years], rotation=45)
ax.set_title('Annual Excess Return over HSI: Raw vs Industry-Neutralized', fontsize=13, fontweight='bold')
ax.set_ylabel('Annual Excess Return (%)', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('pict/v4_plot_neutral_excess_annual.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: pict/v4_plot_neutral_excess_rolling.png")
print("Saved: pict/v4_plot_neutral_excess_annual.png")

# ====== CONCLUSION ======
print("\n\n" + "=" * 90)
print("CONCLUSION: 行业中性化对超额收益的影响")
print("=" * 90)
for name, bt_raw, bt_neu in [('B', bt_B_raw, bt_B_neu), ('C', bt_C_raw, bt_C_neu)]:
    common_r = bt_raw.index.intersection(hsi_ret.index)
    common_n = bt_neu.index.intersection(hsi_ret.index)
    exc_raw = bt_raw.reindex(common_r) - hsi_ret.reindex(common_r)
    exc_neu = bt_neu.reindex(common_n) - hsi_ret.reindex(common_n)
    ir_raw = (exc_raw.mean()*12) / (exc_raw.std()*np.sqrt(12))
    ir_neu = (exc_neu.mean()*12) / (exc_neu.std()*np.sqrt(12))
    hit_raw = (exc_raw > 0).mean()
    hit_neu = (exc_neu > 0).mean()
    print(f"\n{name}策略: IR {ir_raw:.2f}->{ir_neu:.2f} ({ir_neu-ir_raw:+.2f}), "
          f"胜率 {hit_raw:.1%}->{hit_neu:.1%} ({hit_neu-hit_raw:+.1%}), "
          f"年化超额 {(exc_raw.mean()*12):.2%}->{(exc_neu.mean()*12):.2%}")

print("\nDone!")
