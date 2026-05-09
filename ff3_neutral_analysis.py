"""
FF3因子中性化分析: 对Shareholder Yield得分分别做市值、BM、市值+BM的中性化
"""
import pandas as pd, numpy as np, warnings, os
warnings.filterwarnings('ignore')
os.makedirs('pict', exist_ok=True)
import matplotlib
matplotlib.use('Agg')

# Chinese font support
import matplotlib.font_manager as fm
for f in fm.fontManager.ttflist:
    if f.name == 'Microsoft YaHei':
        fm.fontManager.addfont(f.fname)
        break
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import statsmodels.api as sm

print("=" * 80)
print("FF3因子中性化分析: Size / Value / Size+Value")
print("=" * 80)

# ====== DATA LOADING (same as v4 notebook) ======
print("\n[1/5] Loading data...")
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
                   shares_monthly[['date_m', 'sid', 'total']], on=['date_m', 'sid'], how='inner')
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

# BM data
bm = pd.read_csv('data/hk_bm_monthly.csv')
bm['date_m'] = pd.to_datetime(bm['date_m']).dt.to_period('M')
bm['sid'] = bm['sid'].astype(str).str.zfill(4) + '.HK'
bm = bm[['date_m', 'sid', 'BM']]

df_factors = hsci[['date_m', 'sid', 'is_hsci']].merge(
    mcap_df[['date_m', 'sid', 'mcap']], on=['date_m', 'sid'], how='inner'
).merge(vol_stacked[['date_m', 'sid', 'volatility']], on=['date_m', 'sid'], how='inner'
).merge(adv_stacked[['date_m', 'sid', 'adtv_3m']], on=['date_m', 'sid'], how='inner'
).merge(dy[['date_m', 'sid', 'dy']], on=['date_m', 'sid'], how='left'
).merge(buyback_36m_stacked, on=['date_m', 'sid'], how='left'
).merge(bm, on=['date_m', 'sid'], how='left')
df_factors['buyback_36m'] = df_factors['buyback_36m'].fillna(0)
df_factors = df_factors.drop_duplicates(subset=['date_m', 'sid'])

# HSI benchmark
hsi = pd.read_csv('data/HSI_index.csv')
hsi['date'] = pd.to_datetime(hsi['date'])
hsi['date_m'] = hsi['date'].dt.to_period('M')
hsi_monthly = hsi.sort_values('date').groupby('date_m').last().reset_index().set_index('date_m')['close']
hsi_ret = hsi_monthly.pct_change().shift(-1).dropna()
hsi_ret.index = hsi_ret.index + 1
BACKTEST_START = pd.Period('2012-01', 'M')
hsi_ret = hsi_ret[hsi_ret.index >= BACKTEST_START + 1]

print(f"   Factors shape: {df_factors.shape}")
print(f"   BM coverage: {df_factors['BM'].notna().mean()*100:.1f}%")

# ====== STRATEGY DEFINITION ======
print("\n[2/5] Defining neutralization methods...")

def compute_sy_score(pool, dy_pivot, price_pivot_full):
    """Compute raw SY score for a pool of stocks"""
    pool = pool.copy()
    dps_36m = price_pivot_full.reindex(dy_pivot.index, method='ffill') * dy_pivot
    current_prices = price_pivot_full.reindex(dy_pivot.index, method='ffill').iloc[-1] if len(price_pivot_full) > 0 else pd.Series(dtype=float)
    if len(dps_36m) > 0 and len(current_prices) > 0:
        pool['div_yield'] = pool['sid'].map(dps_36m.mean() / current_prices).fillna(0)
    else:
        pool['div_yield'] = 0.0
    pool['buyback_yield'] = ((pool['buyback_36m'] / 3) / pool['mcap']) * 100
    pool['buyback_yield'] = pool['buyback_yield'].replace([np.inf, -np.inf, np.nan], 0)
    pool['score_raw'] = pool['div_yield'] + pool['buyback_yield']
    return pool.dropna(subset=['score_raw'])

def neutralize_scores(pool, method):
    """
    Neutralize SY scores cross-sectionally.
    method: 'size' (log mcap), 'value' (BM), 'size_value' (log mcap + BM), 'none' (raw)
    Returns pool with 'score' column = neutralized score
    """
    pool = pool.copy()
    pool['score'] = pool['score_raw']  # default: raw

    if method == 'none':
        return pool

    pool['log_mcap'] = np.log(pool['mcap'])

    if method == 'size':
        # Regress score_raw ~ log_mcap, take residuals
        valid = pool['log_mcap'].notna()
        if valid.sum() >= 10:
            y = pool.loc[valid, 'score_raw'].values
            X = sm.add_constant(pool.loc[valid, 'log_mcap'].values)
            model = sm.OLS(y, X).fit()
            pool.loc[valid, 'score'] = model.resid
        return pool

    elif method == 'value':
        valid = pool['BM'].notna() & (pool['BM'] > 0) & np.isfinite(pool['BM'])
        if valid.sum() >= 10:
            y = pool.loc[valid, 'score_raw'].values
            X = sm.add_constant(pool.loc[valid, 'BM'].values)
            model = sm.OLS(y, X).fit()
            pool.loc[valid, 'score'] = model.resid
        return pool

    elif method == 'size_value':
        valid = pool['log_mcap'].notna() & pool['BM'].notna() & (pool['BM'] > 0) & np.isfinite(pool['BM'])
        if valid.sum() >= 10:
            y = pool.loc[valid, 'score_raw'].values
            X = np.column_stack([
                np.ones(valid.sum()),
                pool.loc[valid, 'log_mcap'].values,
                pool.loc[valid, 'BM'].values
            ])
            model = sm.OLS(y, X).fit()
            pool.loc[valid, 'score'] = model.resid
        return pool

    return pool

def select_stocks(group, neutralize_method='none', use_lowvol=False):
    """Main strategy function with configurable neutralization"""
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
    if len(pool) < 10: return []

    # Compute raw SY scores
    pool = compute_sy_score(pool, dy_36m, price_36m)
    if len(pool) < 10: return []

    # Extreme trim
    cutoff = pool['score_raw'].quantile(0.9)
    pool = pool[pool['score_raw'] <= cutoff]
    if len(pool) < 10: return []

    # Apply neutralization
    pool = neutralize_scores(pool, neutralize_method)
    pool = pool.dropna(subset=['score'])
    if len(pool) < 10: return []

    # Select top 20% by (potentially neutralized) score
    n_top = max(10, int(len(pool) * 0.20))
    pool = pool.nlargest(n_top, 'score')

    n_final = min(20, len(pool))
    if use_lowvol:
        return pool.nsmallest(n_final, 'volatility')['sid'].tolist()
    else:
        return pool.nlargest(n_final, 'score')['sid'].tolist()

# ====== RUN ALL VARIANTS ======
print("\n[3/5] Running 8 strategy variants...")

variants = [
    ('B_Raw', 'none', False),
    ('B_SizeNeut', 'size', False),
    ('B_ValueNeut', 'value', False),
    ('B_SizeValueNeut', 'size_value', False),
    ('C_Raw', 'none', True),
    ('C_SizeNeut', 'size', True),
    ('C_ValueNeut', 'value', True),
    ('C_SizeValueNeut', 'size_value', True),
]

signals_dict = {}
for name, method, lowvol in variants:
    print(f"  {name} (method={method}, lowvol={lowvol})...", end=' ', flush=True)
    sigs = df_factors.groupby('date_m', group_keys=True).apply(
        lambda g: select_stocks(g, neutralize_method=method, use_lowvol=lowvol))
    signals_dict[name] = sigs
    print(f"OK ({len(sigs)} periods, avg {sigs.apply(len).mean():.1f} stocks)")

# ====== BACKTEST ======
print("\n[4/5] Running backtests...")

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

bt_dict = {}
for name in signals_dict:
    bt_dict[name] = run_bt(signals_dict[name])

def calc_metrics(returns):
    ann_ret = returns.mean() * 12
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else np.nan
    cum_ret = 1 + returns.cumsum()
    max_dd = (cum_ret / cum_ret.cummax().clip(lower=1.0) - 1).min()
    return ann_ret, ann_vol, sharpe, max_dd, cum_ret.iloc[-1]

# ====== FF3 ALPHA ======
ff3 = pd.read_csv('data/hk_ff3_factors.csv')
ff3['date_m'] = pd.to_datetime(ff3['date_m']).dt.to_period('M')
ff3 = ff3.set_index('date_m')

def compute_ff3_alpha(returns):
    common = returns.index.intersection(ff3.index)
    if len(common) < 24: return np.nan, np.nan, np.nan, np.nan, np.nan
    y = returns.reindex(common)
    X = sm.add_constant(ff3.loc[common, ['MKT', 'SMB', 'HML']])
    model = sm.OLS(y.values, X.values).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
    return (model.params[0]*100, model.tvalues[0], model.pvalues[0],
            model.params[1], model.rsquared)

# ====== RESULTS ======
print("\n[5/5] Computing results...\n")

# Excess return analysis
print("=" * 100)
print("FF3因子中性化: 完整结果对比")
print("=" * 100)

for strategy_group, strat_names in [
    ('STRATEGY B (纯SY + 极端值剔除)', ['B_Raw', 'B_SizeNeut', 'B_ValueNeut', 'B_SizeValueNeut']),
    ('STRATEGY C (SY + 极端值剔除 + 低波)', ['C_Raw', 'C_SizeNeut', 'C_ValueNeut', 'C_SizeValueNeut']),
]:
    print(f"\n{'='*80}")
    print(f"  {strategy_group}")
    print(f"{'='*80}")
    print(f"{'Variant':<22} {'Ann Ret':<10} {'Ann Vol':<10} {'Sharpe':<8} {'Max DD':<10} {'FF3 a/mo':<10} {'t(a)':<8} {'MKT b':<8} {'Total':<8}")
    print("-" * 95)

    for name in strat_names:
        bt = bt_dict[name]
        ar, av, sh, md, tr = calc_metrics(bt)
        alpha, ta, pa, mkt_beta, rsq = compute_ff3_alpha(bt)
        stars = '***' if pa < 0.01 else ('**' if pa < 0.05 else ('*' if pa < 0.1 else ''))
        label = name.replace('B_', '').replace('C_', '')
        print(f"{label:<22} {ar:>8.2%}   {av:>8.2%}   {sh:>6.2f}   {md:>8.2%}   {alpha:>7.3f}{stars:<3}  {ta:>6.2f}   {mkt_beta:>6.3f}   {tr:>5.2f}x")

# Excess over HSI analysis
print(f"\n\n{'='*100}")
print("超额收益分析 (相对HSI基准) -- 指数增强核心指标")
print(f"{'='*100}")
print(f"{'Variant':<22} {'Ann Exc':<10} {'TrackErr':<10} {'IR':<8} {'HitRate':<10} {'Exc MaxDD':<10} {'Cum Exc':<8}")
print("-" * 85)

for name in ['B_Raw', 'B_SizeNeut', 'B_ValueNeut', 'B_SizeValueNeut',
             'C_Raw', 'C_SizeNeut', 'C_ValueNeut', 'C_SizeValueNeut']:
    bt = bt_dict[name]
    common = bt.index.intersection(hsi_ret.index)
    excess = bt.reindex(common) - hsi_ret.reindex(common)
    ann_exc = excess.mean() * 12
    ann_te = excess.std() * np.sqrt(12)
    ir = ann_exc / ann_te if ann_te > 0 else 0
    hit = (excess > 0).mean()
    cum_exc = 1 + excess.cumsum()
    mdd_exc = (cum_exc / cum_exc.cummax().clip(lower=1.0) - 1).min()
    label = name.replace('B_', '').replace('C_', '').replace('Neut', '中性')
    print(f"{label:<22} {ann_exc:>+7.2%}     {ann_te:>8.2%}    {ir:>6.2f}   {hit:>7.1%}    {mdd_exc:>8.2%}    {cum_exc.iloc[-1]-1:>+6.1%}")

# ====== IMPROVEMENT SUMMARY ======
print(f"\n\n{'='*100}")
print("中性化效果总结: 相对原始策略的变化")
print(f"{'='*100}")
for raw_name, neut_name, label in [
    ('B_Raw', 'B_SizeNeut', '市值中性化'),
    ('B_Raw', 'B_ValueNeut', 'BM中性化'),
    ('B_Raw', 'B_SizeValueNeut', '市值+BM中性化'),
    ('C_Raw', 'C_SizeNeut', '市值中性化'),
    ('C_Raw', 'C_ValueNeut', 'BM中性化'),
    ('C_Raw', 'C_SizeValueNeut', '市值+BM中性化'),
]:
    bt_raw = bt_dict[raw_name]
    bt_neu = bt_dict[neut_name]
    common_r = bt_raw.index.intersection(hsi_ret.index)
    common_n = bt_neu.index.intersection(hsi_ret.index)
    exc_raw = bt_raw.reindex(common_r) - hsi_ret.reindex(common_r)
    exc_neu = bt_neu.reindex(common_n) - hsi_ret.reindex(common_n)

    ir_raw = (exc_raw.mean()*12) / (exc_raw.std()*np.sqrt(12))
    ir_neu = (exc_neu.mean()*12) / (exc_neu.std()*np.sqrt(12))
    hit_raw = (exc_raw > 0).mean()
    hit_neu = (exc_neu > 0).mean()

    alpha_raw, ta_raw, _, _, _ = compute_ff3_alpha(bt_raw)
    alpha_neu, ta_neu, _, _, _ = compute_ff3_alpha(bt_neu)

    ir_delta = ir_neu - ir_raw
    print(f"\n  [{raw_name.replace('B_','B ').replace('C_','C ')}] {label}:")
    print(f"    IR:      {ir_raw:.2f} -> {ir_neu:.2f} ({ir_delta:+.2f})")
    print(f"    HitRate: {hit_raw:.1%} -> {hit_neu:.1%} ({hit_neu-hit_raw:+.1%})")
    print(f"    FF3 a/m: {alpha_raw:.3f}% -> {alpha_neu:.3f}% ({alpha_neu-alpha_raw:+.3f}%)")

# ====== PLOTS ======
print("\n\nGenerating plots...")

# Cumulative return plot
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
colors = {'Raw': 'blue', 'SizeNeut': 'red', 'ValueNeut': 'green', 'SizeValueNeut': 'purple'}
linestyles = {'Raw': '-', 'SizeNeut': '--', 'ValueNeut': '-.', 'SizeValueNeut': ':'}

for ax, prefix, title in zip(axes, ['B_', 'C_'], ['Strategy B (Trim)', 'Strategy C (Trim+LowVol)']):
    # HSI
    cum_hsi = 1 + hsi_ret.cumsum()
    common_idx = bt_dict[f'{prefix}Raw'].index.intersection(hsi_ret.index)
    ax.plot(common_idx.to_timestamp(), cum_hsi.reindex(common_idx), label='HSI', color='gray', linewidth=1, alpha=0.6)

    for suffix, label, color in [('Raw', 'Raw (原始)', 'blue'), ('SizeNeut', 'Size-Neutral', 'red'),
                                   ('ValueNeut', 'Value-Neutral', 'green'), ('SizeValueNeut', 'Size+Value-Neutral', 'purple')]:
        bt = bt_dict[f'{prefix}{suffix}']
        cum = 1 + bt.cumsum()
        ax.plot(cum.index.to_timestamp(), cum.values, label=label, color=color,
                linestyle=linestyles[suffix], linewidth=1.5)

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Net Value', fontsize=11)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig('pict/v4_plot_ff3_neutral_cumulative.png', dpi=150, bbox_inches='tight')
plt.close()

# Excess return plot
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for ax, prefix, title in zip(axes, ['B_', 'C_'], ['Strategy B: Excess over HSI', 'Strategy C: Excess over HSI']):
    for suffix, label, color in [('Raw', 'Raw', 'blue'), ('SizeNeut', 'Size-Neutral', 'red'),
                                   ('ValueNeut', 'Value-Neutral', 'green'), ('SizeValueNeut', 'Size+Value', 'purple')]:
        bt = bt_dict[f'{prefix}{suffix}']
        common = bt.index.intersection(hsi_ret.index)
        excess = bt.reindex(common) - hsi_ret.reindex(common)
        cum_exc = 1 + excess.cumsum()
        ax.plot(cum_exc.index.to_timestamp(), cum_exc.values, label=label, color=color,
                linestyle=linestyles[suffix], linewidth=1.5)

    ax.axhline(y=1, color='black', linewidth=0.5)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Excess over HSI', fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pict/v4_plot_ff3_neutral_excess.png', dpi=150, bbox_inches='tight')
plt.close()

# Rolling IR plot
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
for ax, prefix, title in zip(axes, ['B_', 'C_'], ['Strategy B: Rolling 12M Info Ratio', 'Strategy C: Rolling 12M Info Ratio']):
    for suffix, label, color in [('Raw', 'Raw', 'blue'), ('SizeNeut', 'Size-Neutral', 'red'),
                                   ('ValueNeut', 'Value-Neutral', 'green'), ('SizeValueNeut', 'Size+Value', 'purple')]:
        bt = bt_dict[f'{prefix}{suffix}']
        common = bt.index.intersection(hsi_ret.index)
        excess = bt.reindex(common) - hsi_ret.reindex(common)
        roll_ir = (excess.rolling(12).mean() / excess.rolling(12).std()) * np.sqrt(12)
        ax.plot(roll_ir.index.to_timestamp(), roll_ir.values, label=label, color=color,
                linestyle=linestyles[suffix], linewidth=1.5)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Rolling 12M Information Ratio', fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pict/v4_plot_ff3_neutral_rolling_ir.png', dpi=150, bbox_inches='tight')
plt.close()

# Bar chart: IR comparison
fig, ax = plt.subplots(figsize=(12, 6))
strategies_b = ['B Raw', 'B SizeNeut', 'B ValueNeut', 'B SizeValue']
strategies_c = ['C Raw', 'C SizeNeut', 'C ValueNeut', 'C SizeValue']
all_strats = strategies_b + strategies_c
ir_values = []
hit_values = []

for name in ['B_Raw', 'B_SizeNeut', 'B_ValueNeut', 'B_SizeValueNeut',
             'C_Raw', 'C_SizeNeut', 'C_ValueNeut', 'C_SizeValueNeut']:
    bt = bt_dict[name]
    common = bt.index.intersection(hsi_ret.index)
    excess = bt.reindex(common) - hsi_ret.reindex(common)
    ann_exc = excess.mean() * 12
    ann_te = excess.std() * np.sqrt(12)
    ir_values.append(ann_exc / ann_te if ann_te > 0 else 0)
    hit_values.append((excess > 0).mean() * 100)

x = np.arange(len(all_strats))
width = 0.35
bars1 = ax.bar(x - width/2, ir_values, width, label='Information Ratio', color='steelblue', alpha=0.85)
ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, hit_values, width, label='Hit Rate (%)', color='coral', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([s.replace('SizeValueNeut','Size+Value').replace('Neut','Neut') for s in all_strats], rotation=45, ha='right')
ax.set_ylabel('Information Ratio', fontsize=11)
ax2.set_ylabel('Monthly Hit Rate (%)', fontsize=11)
ax.set_title('FF3 Factor Neutralization: IR & Hit Rate Comparison', fontsize=13, fontweight='bold')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('pict/v4_plot_ff3_neutral_ir_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("Saved: pict/v4_plot_ff3_neutral_cumulative.png")
print("Saved: pict/v4_plot_ff3_neutral_excess.png")
print("Saved: pict/v4_plot_ff3_neutral_rolling_ir.png")
print("Saved: pict/v4_plot_ff3_neutral_ir_comparison.png")
print("\nDone!")
