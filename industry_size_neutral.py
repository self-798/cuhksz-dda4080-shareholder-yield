"""
行业+市值中性化分析: 使用准确的HSICS行业分类 + FF3 Size中性化
"""
import pandas as pd, numpy as np, warnings, os, json, sys
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
print("行业+市值中性化: 准确HSICS行业分类 + Size因子")
print("=" * 80)

# ====== LOAD INDUSTRY DATA ======
cache_file = 'data/hk_industry_map_v2.json'
with open(cache_file, encoding='utf-8') as f:
    industry_raw = json.load(f)

# Filter valid industries
industry_map = {}
for code, ind in industry_raw.items():
    if ind != 'Unknown' and not ind.startswith('ERR'):
        industry_map[code] = ind

print(f"Industry data: {len(industry_map)} stocks with valid industry")
from collections import Counter
for ind, cnt in Counter(industry_map.values()).most_common(20):
    print(f"  {ind}: {cnt}")

# Standardize industry names (merge similar ones)
def standardize_industry(name):
    m = {
        '银行':'银行', '地产':'地产', '公用事业':'公用事业', '电讯':'电讯',
        '软件服务':'科技', '综合企业':'综合企业',
        '保险':'金融', '其他金融':'金融', '证券':'金融',
        '食物饮品':'消费', '纺织及服饰':'消费', '汽车':'消费', '家电':'消费',
        '医疗保健':'医疗', '药品':'医疗', '生物科技':'医疗',
        '建筑':'工业', '工业工程':'工业', '运输':'工业', '工用运输':'工业',
        '原材料':'材料', '一般金属及矿石':'材料', '化工':'材料',
        '媒体':'传媒', '旅游':'消费', '酒店':'消费', '餐饮':'消费',
        '石油及天然气':'能源', '煤炭':'能源', '新能源':'能源',
        '半导体':'科技', '资讯科技器材':'科技',
        '农业':'其他', '其他':'其他',
    }
    return m.get(name, name)

industry_map_std = {k: standardize_industry(v) for k, v in industry_map.items()}
print(f"\nAfter standardization: {len(set(industry_map_std.values()))} industries")
for ind, cnt in Counter(industry_map_std.values()).most_common(30):
    print(f"  {ind}: {cnt}")

# ====== DATA LOADING ======
print("\n[1/4] Loading data...")
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

# Add industry column
df_factors['industry'] = df_factors['sid'].apply(
    lambda s: industry_map_std.get(s.replace('.HK', ''), 'Unknown'))

# HSI benchmark
hsi = pd.read_csv('data/HSI_index.csv')
hsi['date'] = pd.to_datetime(hsi['date'])
hsi['date_m'] = hsi['date'].dt.to_period('M')
hsi_monthly = hsi.sort_values('date').groupby('date_m').last().reset_index().set_index('date_m')['close']
hsi_ret = hsi_monthly.pct_change().shift(-1).dropna()
hsi_ret.index = hsi_ret.index + 1
BACKTEST_START = pd.Period('2012-01', 'M')
hsi_ret = hsi_ret[hsi_ret.index >= BACKTEST_START + 1]

print(f"   Industry coverage: {df_factors['industry'].notna().mean()*100:.1f}%")
print(f"   Non-Unknown industry: {(df_factors['industry']!='Unknown').mean()*100:.1f}%")

# ====== NEUTRALIZATION FUNCTIONS ======
print("\n[2/4] Defining neutralization methods...")

def compute_sy(pool, dy_pivot, price_pivot_full):
    pool = pool.copy()
    dps_36m = price_pivot_full.reindex(dy_pivot.index, method='ffill') * dy_pivot
    cp = price_pivot_full.reindex(dy_pivot.index, method='ffill').iloc[-1]
    if len(cp) > 0:
        pool['div_yield'] = pool['sid'].map(dps_36m.mean() / cp).fillna(0)
    else:
        pool['div_yield'] = 0.0
    pool['buyback_yield'] = ((pool['buyback_36m'] / 3) / pool['mcap']) * 100
    pool['buyback_yield'] = pool['buyback_yield'].replace([np.inf, -np.inf, np.nan], 0)
    pool['score_raw'] = pool['div_yield'] + pool['buyback_yield']
    return pool.dropna(subset=['score_raw'])

def neutralize(pool, method):
    """method: 'none', 'industry', 'industry_size'"""
    pool = pool.copy()
    pool['score'] = pool['score_raw']
    pool['log_mcap'] = np.log(pool['mcap'])

    if method == 'none':
        return pool

    if method == 'industry':
        # Within-industry z-score
        for ind in pool['industry'].unique():
            mask = pool['industry'] == ind
            if mask.sum() >= 3:
                g = pool.loc[mask, 'score_raw']
                pool.loc[mask, 'score'] = (g - g.mean()) / g.std() if g.std() > 0 else 0.0
        return pool

    if method == 'industry_size':
        # Within-industry, regress score_raw ~ log_mcap, take residuals
        for ind in pool['industry'].unique():
            mask = (pool['industry'] == ind) & pool['log_mcap'].notna()
            if mask.sum() >= 5:
                y = pool.loc[mask, 'score_raw'].values
                X = sm.add_constant(pool.loc[mask, 'log_mcap'].values)
                try:
                    model = sm.OLS(y, X).fit()
                    pool.loc[mask, 'score'] = model.resid
                except:
                    pass
        return pool

    return pool

def select_stocks(group, method='none', use_lowvol=False):
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
    pool = compute_sy(pool, dy_36m, price_36m)
    if len(pool) < 10: return []
    cutoff = pool['score_raw'].quantile(0.9)
    pool = pool[pool['score_raw'] <= cutoff]
    if len(pool) < 10: return []
    pool = neutralize(pool, method)
    pool = pool.dropna(subset=['score'])
    if len(pool) < 10: return []
    n_top = max(10, int(len(pool) * 0.20))
    pool = pool.nlargest(n_top, 'score')
    n_final = min(20, len(pool))
    if use_lowvol:
        return pool.nsmallest(n_final, 'volatility')['sid'].tolist()
    else:
        return pool.nlargest(n_final, 'score')['sid'].tolist()

# ====== RUN VARIANTS ======
print("\n[3/4] Running 6 variants...")
variants = [
    ('B_Raw', 'none', False),
    ('B_IndNeut', 'industry', False),
    ('B_IndSizeNeut', 'industry_size', False),
    ('C_Raw', 'none', True),
    ('C_IndNeut', 'industry', True),
    ('C_IndSizeNeut', 'industry_size', True),
]

signals_dict = {}
for name, method, lowvol in variants:
    print(f"  {name}...", end=' ', flush=True)
    sigs = df_factors.groupby('date_m', group_keys=True).apply(
        lambda g: select_stocks(g, method=method, use_lowvol=lowvol))
    signals_dict[name] = sigs
    print(f"OK ({len(sigs)} periods)")

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

# FF3 Alpha
ff3 = pd.read_csv('data/hk_ff3_factors.csv')
ff3['date_m'] = pd.to_datetime(ff3['date_m']).dt.to_period('M')
ff3 = ff3.set_index('date_m')

def compute_ff3_alpha(returns):
    common = returns.index.intersection(ff3.index)
    if len(common) < 24: return np.nan, np.nan, np.nan
    y = returns.reindex(common)
    X = sm.add_constant(ff3.loc[common, ['MKT', 'SMB', 'HML']])
    model = sm.OLS(y.values, X.values).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
    return (model.params[0]*100, model.tvalues[0], model.pvalues[0])

# ====== RESULTS ======
print("\n[4/4] Computing results...\n")

# Main comparison table
print("=" * 110)
print("行业+市值中性化: 完整结果")
print("=" * 110)

for grp_name, names in [
    ('STRATEGY B (纯SY+极端值剔除)', ['B_Raw', 'B_IndNeut', 'B_IndSizeNeut']),
    ('STRATEGY C (SY+极端值剔除+低波)', ['C_Raw', 'C_IndNeut', 'C_IndSizeNeut']),
]:
    print(f"\n{'='*80}")
    print(f"  {grp_name}")
    print(f"{'='*80}")
    print(f"{'Variant':<20} {'Ann Ret':<10} {'Ann Vol':<10} {'Sharpe':<8} {'Max DD':<10} {'FF3 a/mo':<10} {'t(a)':<8} {'Total':<8}")
    print("-" * 90)
    for name in names:
        bt = bt_dict[name]
        ar, av, sh, md, tr = calc_metrics(bt)
        alpha, ta, pa = compute_ff3_alpha(bt)
        stars = '***' if pa < 0.01 else ('**' if pa < 0.05 else '*')
        label = name.replace('B_', '').replace('C_', '')
        print(f"{label:<20} {ar:>8.2%}   {av:>8.2%}   {sh:>6.2f}   {md:>8.2%}   {alpha:>7.3f}{stars:<3}  {ta:>6.2f}   {tr:>5.2f}x")

# Excess over HSI
print(f"\n\n{'='*110}")
print("超额收益分析 (相对HSI)")
print(f"{'='*110}")
print(f"{'Variant':<20} {'Ann Exc':<10} {'TrackErr':<10} {'IR':<8} {'HitRate':<10} {'Exc MaxDD':<10} {'Cum Exc':<8}")
print("-" * 85)
for name in ['B_Raw', 'B_IndNeut', 'B_IndSizeNeut', 'C_Raw', 'C_IndNeut', 'C_IndSizeNeut']:
    bt = bt_dict[name]
    common = bt.index.intersection(hsi_ret.index)
    excess = bt.reindex(common) - hsi_ret.reindex(common)
    ann_exc = excess.mean() * 12
    ann_te = excess.std() * np.sqrt(12)
    ir = ann_exc / ann_te if ann_te > 0 else 0
    hit = (excess > 0).mean()
    cum_exc = 1 + excess.cumsum()
    mdd_exc = (cum_exc / cum_exc.cummax().clip(lower=1.0) - 1).min()
    label = name.replace('B_', '').replace('C_', '')
    print(f"{label:<20} {ann_exc:>+7.2%}     {ann_te:>8.2%}    {ir:>6.2f}   {hit:>7.1%}    {mdd_exc:>8.2%}    {cum_exc.iloc[-1]-1:>+6.1%}")

# Comparison summary
print(f"\n\n{'='*110}")
print("中性化效果总结")
print(f"{'='*110}")
for raw_name, neut_name, label in [
    ('B_Raw', 'B_IndNeut', '行业中性化'),
    ('B_Raw', 'B_IndSizeNeut', '行业+市值中性化'),
    ('C_Raw', 'C_IndNeut', '行业中性化'),
    ('C_Raw', 'C_IndSizeNeut', '行业+市值中性化'),
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
    alpha_raw, _, _ = compute_ff3_alpha(bt_raw)
    alpha_neu, _, _ = compute_ff3_alpha(bt_neu)
    print(f"\n  [{raw_name.split('_')[0]}] {label}:")
    print(f"    IR:      {ir_raw:.2f} -> {ir_neu:.2f} ({ir_neu-ir_raw:+.2f})")
    print(f"    HitRate: {hit_raw:.1%} -> {hit_neu:.1%} ({hit_neu-hit_raw:+.1%})")
    print(f"    FF3 a/m: {alpha_raw:.3f}% -> {alpha_neu:.3f}% ({alpha_neu-alpha_raw:+.3f}%)")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, prefix, title in zip(axes, ['B_', 'C_'], ['Strategy B (Trim)', 'Strategy C (Trim+LowVol)']):
    cum_hsi = 1 + hsi_ret.cumsum()
    common_idx = bt_dict[f'{prefix}Raw'].index.intersection(hsi_ret.index)
    ax.plot(common_idx.to_timestamp(), cum_hsi.reindex(common_idx), label='HSI', color='gray', alpha=0.5)
    for suffix, label, color, ls in [
        ('Raw', 'Raw', 'blue', '-'), ('IndNeut', 'Industry-Neutral', 'red', '--'),
        ('IndSizeNeut', 'Industry+Size-Neutral', 'green', ':')
    ]:
        bt = bt_dict[f'{prefix}{suffix}']
        cum = 1 + bt.cumsum()
        ax.plot(cum.index.to_timestamp(), cum.values, label=label, color=color, linestyle=ls, linewidth=1.5)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pict/v4_plot_indsize_neutral_cumulative.png', dpi=150, bbox_inches='tight')
plt.close()

# Excess plot
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, prefix, title in zip(axes, ['B_', 'C_'], ['B: Excess over HSI', 'C: Excess over HSI']):
    for suffix, label, color, ls in [
        ('Raw', 'Raw', 'blue', '-'), ('IndNeut', 'Industry-Neutral', 'red', '--'),
        ('IndSizeNeut', 'Industry+Size', 'green', ':')
    ]:
        bt = bt_dict[f'{prefix}{suffix}']
        common = bt.index.intersection(hsi_ret.index)
        excess = bt.reindex(common) - hsi_ret.reindex(common)
        cum_exc = 1 + excess.cumsum()
        ax.plot(cum_exc.index.to_timestamp(), cum_exc.values, label=label, color=color, linestyle=ls, linewidth=1.5)
    ax.axhline(y=1, color='black', linewidth=0.5)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pict/v4_plot_indsize_neutral_excess.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: pict/v4_plot_indsize_neutral_cumulative.png")
print("Saved: pict/v4_plot_indsize_neutral_excess.png")
print("\nDone!")
