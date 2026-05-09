"""HSICS industry + size neutralization - complete analysis"""
import pandas as pd, numpy as np, json, os, warnings
warnings.filterwarnings('ignore')
os.makedirs('pict', exist_ok=True)
import matplotlib; matplotlib.use('Agg')

# Chinese font support
import matplotlib.font_manager as fm
for f in fm.fontManager.ttflist:
    if f.name == 'Microsoft YaHei':
        fm.fontManager.addfont(f.fname)
        break
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt, matplotlib.ticker as mtick
import statsmodels.api as sm

print("="*80)
print("HSICS行业+市值中性化分析 (基于恒生行业分类系统)")
print("="*80)

# ====== HSICS MAPPING ======
HSICS_MAP = {
    '油气生产商':'00_能源业','石油及天然气':'00_能源业','煤炭':'00_能源业',
    '黄金及贵金属':'05_原材料业','一般金属及矿石':'05_原材料业','钢铁':'05_原材料业',
    '铜':'05_原材料业','铝':'05_原材料业','其他金属及矿物':'05_原材料业',
    '化肥及农用化合物':'05_原材料业','林业及木材':'05_原材料业','纸及纸制品':'05_原材料业',
    '特殊化工用品':'05_原材料业','原材料':'05_原材料业',
    '工业工程':'10_工业','商用运输工具':'10_工业','工业零件及器材':'10_工业',
    '电子零件':'10_工业','环保工程':'10_工业','重型机械':'10_工业',
    '新能源物料':'10_工业','航空航天':'10_工业','建筑':'10_工业','运输':'10_工业','工用运输':'10_工业',
    '汽车':'23_非必需性消费','家电':'23_非必需性消费','纺织及服饰':'23_非必需性消费',
    '媒体及娱乐':'23_非必需性消费','广告':'23_非必需性消费','广播':'23_非必需性消费',
    '影视娱乐':'23_非必需性消费','出版':'23_非必需性消费','教育':'23_非必需性消费',
    '旅游':'23_非必需性消费','酒店':'23_非必需性消费','餐饮':'23_非必需性消费',
    '赌场及博彩':'23_非必需性消费','消闲':'23_非必需性消费','零售':'23_非必需性消费',
    '航空服务':'23_非必需性消费','其他支援服务':'23_非必需性消费',
    '食物饮品':'25_必需性消费','农业':'25_必需性消费','超市':'25_必需性消费',
    '个人护理':'25_必需性消费','包装食品':'25_必需性消费','饮料':'25_必需性消费','农产品':'25_必需性消费',
    '药品':'28_医疗保健业','生物科技':'28_医疗保健业','医疗':'28_医疗保健业',
    '医疗保健':'28_医疗保健业','中医药':'28_医疗保健业','医药':'28_医疗保健业',
    '电讯':'35_电讯业','卫星及无线通讯':'35_电讯业','电讯服务':'35_电讯业',
    '公用事业':'40_公用事业','电力':'40_公用事业','燃气':'40_公用事业','水务':'40_公用事业',
    '银行':'50_金融业','保险':'50_金融业','证券':'50_金融业','其他金融':'50_金融业',
    '投资及资产管理':'50_金融业','信贷':'50_金融业','支付服务':'50_金融业',
    '地产':'60_地产建筑业','物业':'60_地产建筑业','建筑材料':'60_地产建筑业',
    '房地产':'60_地产建筑业','地产代理':'60_地产建筑业','地产发展商':'60_地产建筑业',
    '楼宇建造':'60_地产建筑业','重型基建':'60_地产建筑业',
    '软件服务':'70_资讯科技业','半导体':'70_资讯科技业','资讯科技器材':'70_资讯科技业',
    '互联网':'70_资讯科技业','游戏':'70_资讯科技业','应用软件':'70_资讯科技业',
    '电讯设备':'70_资讯科技业','电脑及周边':'70_资讯科技业',
    '综合企业':'80_综合企业',
}

def code_range_to_hsics(sid):
    try:
        code = int(sid.replace('.HK', ''))
    except ValueError:
        return '80_综合企业'
    if 1 <= code <= 99: return '80_综合企业'
    elif 100 <= code <= 499: return '60_地产建筑业'
    elif 500 <= code <= 799: return '10_工业'
    elif 800 <= code <= 999: return '23_非必需性消费'
    elif 1000 <= code <= 1299: return '23_非必需性消费'
    elif 1300 <= code <= 1399: return '50_金融业'
    elif 1400 <= code <= 1899: return '10_工业'
    elif 1900 <= code <= 1999: return '50_金融业'
    elif 2000 <= code <= 2199: return '25_必需性消费'
    elif 2200 <= code <= 2399: return '23_非必需性消费'
    elif 2400 <= code <= 2499: return '70_资讯科技业'
    elif 2500 <= code <= 2899: return '23_非必需性消费'
    elif 2900 <= code <= 2999: return '25_必需性消费'
    elif 3000 <= code <= 3099: return '70_资讯科技业'
    elif 3100 <= code <= 3699: return '10_工业'
    elif 3700 <= code <= 3799: return '23_非必需性消费'
    elif 3800 <= code <= 3899: return '05_原材料业'
    elif 3900 <= code <= 3999: return '60_地产建筑业'
    elif 4000 <= code <= 5999: return '10_工业'
    elif 6000 <= code <= 6199: return '50_金融业'
    elif 6200 <= code <= 6299: return '23_非必需性消费'
    elif 6300 <= code <= 6399: return '10_工业'
    elif 6400 <= code <= 6599: return '70_资讯科技业'
    elif 6600 <= code <= 6699: return '25_必需性消费'
    elif 6700 <= code <= 6999: return '60_地产建筑业'
    elif 7000 <= code <= 7999: return '70_资讯科技业'
    elif 8000 <= code <= 8999: return '23_非必需性消费'
    elif 9000 <= code <= 9999: return '10_工业'
    else: return '80_综合企业'

# Load verified akshare data
cache_file = 'data/hk_industry_map_v2.json'
akshare_data = {}
if os.path.exists(cache_file):
    with open(cache_file, encoding='utf-8') as f:
        raw = json.load(f)
    for code, ind in raw.items():
        if ind != 'Unknown' and not ind.startswith('ERR') and ind in HSICS_MAP:
            akshare_data[code] = HSICS_MAP[ind]

print(f"Verified akshare industries: {len(akshare_data)} stocks")

# Build complete mapping
hsci = pd.read_csv('data/HSCI.csv')
industry_map = {}
for sid in hsci['sid'].unique():
    code = sid.replace('.HK', '')
    industry_map[sid] = akshare_data.get(code, code_range_to_hsics(sid))

from collections import Counter
print(f"Industry distribution ({len(industry_map)} stocks):")
for ind, cnt in Counter(industry_map.values()).most_common(15):
    print(f"  {ind}: {cnt}")

# ====== DATA LOADING ======
print("\n[1/3] Loading market data...")
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
mcap_df = pd.merge(monthly_price[['date_m', 'sid', 'close']], shares_monthly[['date_m', 'sid', 'total']], on=['date_m', 'sid'], how='inner')
mcap_df['mcap'] = mcap_df['close'] * mcap_df['total']

price = pd.read_csv('data/hk_price.csv', usecols=['date', 'sid', 'close', 'amount'])
price['date'] = pd.to_datetime(price['date'])
price_pivot_full = price.pivot(index='date', columns='sid', values='close').sort_index()
ret_pvt = price_pivot_full.pct_change()
vol_pvt = ret_pvt.rolling(window=252, min_periods=100).std() * np.sqrt(252)
vol_m = vol_pvt.resample('ME').last().stack().reset_index()
vol_m.columns = ['date', 'sid', 'volatility']
vol_m['date_m'] = vol_m['date'].dt.to_period('M')

amt_pvt = price.pivot(index='date', columns='sid', values='amount').sort_index()
adv_pvt = amt_pvt.rolling(window=63, min_periods=21).mean()
adv_m = adv_pvt.resample('ME').last().stack().reset_index()
adv_m.columns = ['date', 'sid', 'adtv_3m']
adv_m['date_m'] = adv_m['date'].dt.to_period('M')

dy = pd.read_hdf('data/hk_dividendyield.h5')
if isinstance(dy, pd.DataFrame): dy = dy.iloc[:, 0]
dy = dy.dropna().reset_index()
dy.columns = ['date', 'sid', 'dy'] if dy.shape[1] == 3 else ['date', 'dy']
dy['date'] = pd.to_datetime(dy['date']); dy['date_m'] = dy['date'].dt.to_period('M')
dy_pivot = dy.pivot(index='date', columns='sid', values='dy').sort_index()

hsci_df = pd.read_csv('data/HSCI.csv'); hsci_df['date'] = pd.to_datetime(hsci_df['date'])
hsci_df['is_hsci'] = 1; hsci_df = hsci_df.drop_duplicates(subset=['date', 'sid'])
hsci_df['date_m'] = hsci_df['date'].dt.to_period('M')

buyback = pd.read_csv('data/em_buyback_filtered.csv')
buyback['date'] = pd.to_datetime(buyback['日期'])
buyback['sid'] = buyback['股票代码'].astype(str).str.zfill(5).str[1:5] + '.HK'
buyback['date_m'] = buyback['date'].dt.to_period('M')
mbb = buyback.groupby(['date_m', 'sid'])['回购总额'].sum().reset_index()
mbb_pvt = mbb.pivot(index='date_m', columns='sid', values='回购总额').fillna(0).sort_index()
bb36 = mbb_pvt.rolling(window=36, min_periods=12).sum().stack().reset_index()
bb36.columns = ['date_m', 'sid', 'buyback_36m']

df_factors = hsci_df[['date_m', 'sid', 'is_hsci']].merge(
    mcap_df[['date_m', 'sid', 'mcap']], on=['date_m', 'sid'], how='inner'
).merge(vol_m[['date_m', 'sid', 'volatility']], on=['date_m', 'sid'], how='inner'
).merge(adv_m[['date_m', 'sid', 'adtv_3m']], on=['date_m', 'sid'], how='inner'
).merge(dy[['date_m', 'sid', 'dy']], on=['date_m', 'sid'], how='left'
).merge(bb36, on=['date_m', 'sid'], how='left')
df_factors['buyback_36m'] = df_factors['buyback_36m'].fillna(0)
df_factors = df_factors.drop_duplicates(subset=['date_m', 'sid'])
df_factors['hsics'] = df_factors['sid'].map(industry_map)

hsi = pd.read_csv('data/HSI_index.csv'); hsi['date'] = pd.to_datetime(hsi['date'])
hsi['date_m'] = hsi['date'].dt.to_period('M')
hsi_m = hsi.sort_values('date').groupby('date_m').last().reset_index().set_index('date_m')['close']
hsi_ret = hsi_m.pct_change().shift(-1).dropna()
hsi_ret.index = hsi_ret.index + 1
BACKTEST_START = pd.Period('2012-01', 'M')
hsi_ret = hsi_ret[hsi_ret.index >= BACKTEST_START + 1]
print(f"   Factors: {df_factors.shape}")

# ====== STRATEGY ======
print("\n[2/3] Running strategies...")
def compute_sy(pool, d36, p36):
    pool = pool.copy()
    dps = p36 * d36; cp = p36.iloc[-1] if len(p36) > 0 else pd.Series()
    if len(cp) > 0: pool['div_yield'] = pool['sid'].map(dps.mean() / cp).fillna(0)
    else: pool['div_yield'] = 0.0
    pool['buyback_yield'] = ((pool['buyback_36m']/3)/pool['mcap'])*100
    pool['buyback_yield'] = pool['buyback_yield'].replace([np.inf, -np.inf, np.nan], 0)
    pool['score_raw'] = pool['div_yield'] + pool['buyback_yield']
    return pool.dropna(subset=['score_raw'])

def neutralize(pool, method):
    pool = pool.copy(); pool['score'] = pool['score_raw']
    if method == 'none': return pool
    pool['log_mcap'] = np.log(pool['mcap'])
    if method == 'industry':
        for ind in pool['hsics'].dropna().unique():
            m = pool['hsics'] == ind
            if m.sum() >= 3:
                g = pool.loc[m, 'score_raw']
                pool.loc[m, 'score'] = (g - g.mean()) / g.std() if g.std() > 0 else 0.0
    elif method == 'industry_size':
        for ind in pool['hsics'].dropna().unique():
            m = (pool['hsics'] == ind) & pool['log_mcap'].notna()
            if m.sum() >= 5:
                y = pool.loc[m, 'score_raw'].values
                X = sm.add_constant(pool.loc[m, 'log_mcap'].values)
                try: pool.loc[m, 'score'] = sm.OLS(y, X).fit().resid
                except: pass
    return pool

def select(g, method='none', lowvol=False):
    cm = g.name
    if isinstance(cm, tuple): cm = cm[0]
    ed = cm.to_timestamp(how='end')
    d36 = dy_pivot.loc[:ed].tail(36); p36 = price_pivot_full.reindex(d36.index, method='ffill')
    has_div = d36.columns[d36.gt(0).any(axis=0)] if len(d36) > 0 else pd.Index([])
    pool = g[(g['sid'].isin(has_div)) | (g['buyback_36m'] > 0)]
    adtv_th = g['adtv_3m'].quantile(0.2); mcap_th = g['mcap'].quantile(0.2)
    pool = pool[(pool['adtv_3m'] >= adtv_th) & (pool['mcap'] >= mcap_th)]
    if len(pool) < 10: return []
    pool = compute_sy(pool, d36, p36)
    if len(pool) < 10: return []
    cutoff = pool['score_raw'].quantile(0.9); pool = pool[pool['score_raw'] <= cutoff]
    if len(pool) < 10: return []
    pool = neutralize(pool, method); pool = pool.dropna(subset=['score'])
    if len(pool) < 10: return []
    n_top = max(10, int(len(pool) * 0.20)); pool = pool.nlargest(n_top, 'score')
    n_final = min(20, len(pool))
    return pool.nsmallest(n_final, 'volatility')['sid'].tolist() if lowvol else pool.nlargest(n_final, 'score')['sid'].tolist()

variants = [
    ('B_Raw', 'none', False), ('B_HSICS_Ind', 'industry', False),
    ('B_HSICS_IndSize', 'industry_size', False),
    ('C_Raw', 'none', True), ('C_HSICS_Ind', 'industry', True),
    ('C_HSICS_IndSize', 'industry_size', True),
]
signals = {}
for name, method, lowvol in variants:
    print(f"  {name}...", end=' ', flush=True)
    signals[name] = df_factors.groupby('date_m', group_keys=True).apply(lambda g: select(g, method, lowvol))
    print(f"OK")

def run_bt(sig):
    dates = sig.index[sig.index >= BACKTEST_START]; rets = []; current = []
    for ti, dm in enumerate(dates):
        if dm not in monthly_ret.index: continue
        if ti % 3 == 0:
            ns = sig.loc[dm]
            if len(ns) > 0: current = ns
        if len(current) == 0: continue
        nxt = monthly_ret.loc[dm].reindex(current).fillna(0)
        rets.append((dm + 1, nxt.mean()))
    return pd.Series([r for _, r in rets], index=[d for d, _ in rets])

bt = {n: run_bt(signals[n]) for n in signals}

def metrics(r):
    ar = r.mean() * 12; av = r.std() * np.sqrt(12); sh = ar / av if av != 0 else np.nan
    cr = (1 + r).cumprod(); md = (cr / cr.cummax().clip(lower=1.0) - 1).min()
    return ar, av, sh, md, cr.iloc[-1]

ff3 = pd.read_csv('data/hk_ff3_factors.csv')
ff3['date_m'] = pd.to_datetime(ff3['date_m']).dt.to_period('M'); ff3 = ff3.set_index('date_m')

def ff3_alpha(r):
    common = r.index.intersection(ff3.index)
    if len(common) < 24: return np.nan, np.nan, np.nan
    y = r.reindex(common); X = sm.add_constant(ff3.loc[common, ['MKT', 'SMB', 'HML']])
    m = sm.OLS(y.values, X.values).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
    return m.params[0] * 100, m.tvalues[0], m.pvalues[0]

# ====== RESULTS ======
print("\n[3/3] Results:\n")
print("=" * 110)
print("HSICS行业+市值中性化 — 完整结果对比")
print("=" * 110)

for grp, names in [
    ('STRATEGY B (纯SY+极端值剔除)', ['B_Raw', 'B_HSICS_Ind', 'B_HSICS_IndSize']),
    ('STRATEGY C (SY+极端值剔除+低波)', ['C_Raw', 'C_HSICS_Ind', 'C_HSICS_IndSize']),
]:
    print(f"\n{'='*80}\n  {grp}\n{'='*80}")
    print(f"{'Variant':<26} {'Ann Ret':<10} {'Ann Vol':<10} {'Sharpe':<8} {'Max DD':<10} {'FF3 a/mo':<10} {'t(a)':<8} {'Total':<8}")
    print("-" * 95)
    for n in names:
        ar, av, sh, md, tr = metrics(bt[n]); al, ta, pa = ff3_alpha(bt[n])
        s = '***' if pa < 0.01 else ('**' if pa < 0.05 else '*')
        print(f"{n:<26} {ar:>8.2%}   {av:>8.2%}   {sh:>6.2f}   {md:>8.2f}   {al:>7.3f}{s:<3}  {ta:>6.2f}   {tr:>5.2f}x")

print(f"\n\n{'='*110}")
print("超额收益分析 (相对HSI基准)")
print(f"{'='*110}")
print(f"{'Variant':<26} {'Ann Exc':<10} {'TrackErr':<10} {'IR':<8} {'HitRate':<10} {'Exc MaxDD':<10}")
print("-" * 90)
all_names = ['B_Raw', 'B_HSICS_Ind', 'B_HSICS_IndSize', 'C_Raw', 'C_HSICS_Ind', 'C_HSICS_IndSize']
for n in all_names:
    common = bt[n].index.intersection(hsi_ret.index)
    exc = bt[n].reindex(common) - hsi_ret.reindex(common)
    ae = exc.mean() * 12; at = exc.std() * np.sqrt(12); ir = ae / at if at > 0 else 0
    hit = (exc > 0).mean(); ce = 1 + exc.cumsum()
    mde = (ce / ce.cummax().clip(lower=1.0) - 1).min()
    print(f"{n:<26} {ae:>+7.2%}     {at:>8.2%}    {ir:>6.2f}   {hit:>7.1%}    {mde:>8.2%}")

print(f"\n\n{'='*110}")
print("中性化效果总结")
print(f"{'='*110}")
for raw_n, neut_n, label in [
    ('B_Raw', 'B_HSICS_Ind', 'HSICS行业中性化'),
    ('B_Raw', 'B_HSICS_IndSize', 'HSICS行业+市值中性化'),
    ('C_Raw', 'C_HSICS_Ind', 'HSICS行业中性化'),
    ('C_Raw', 'C_HSICS_IndSize', 'HSICS行业+市值中性化'),
]:
    cr = bt[raw_n].index.intersection(hsi_ret.index)
    cn = bt[neut_n].index.intersection(hsi_ret.index)
    er = bt[raw_n].reindex(cr) - hsi_ret.reindex(cr)
    en = bt[neut_n].reindex(cn) - hsi_ret.reindex(cn)
    ir_r = (er.mean()*12) / (er.std()*np.sqrt(12))
    ir_n = (en.mean()*12) / (en.std()*np.sqrt(12))
    hr_r = (er > 0).mean(); hr_n = (en > 0).mean()
    al_r, _, _ = ff3_alpha(bt[raw_n]); al_n, _, _ = ff3_alpha(bt[neut_n])
    print(f"\n  [{raw_n.split('_')[0]}] {label}:")
    print(f"    IR:      {ir_r:.2f} -> {ir_n:.2f} ({ir_n-ir_r:+.2f})")
    print(f"    HitRate: {hr_r:.1%} -> {hr_n:.1%} ({hr_n-hr_r:+.1%})")
    print(f"    FF3 a/m: {al_r:.3f}% -> {al_n:.3f}% ({al_n-al_r:+.3f}%)")

# Plots
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, pref, title in zip(axes, ['B_', 'C_'], ['Strategy B (Trim)', 'Strategy C (Trim+LowVol)']):
    cum_hsi = 1 + hsi_ret.cumsum()
    cidx = bt[f'{pref}Raw'].index.intersection(hsi_ret.index)
    ax.plot(cidx.to_timestamp(), cum_hsi.reindex(cidx), label='HSI基准', color='gray', alpha=0.5)
    for suf, label, color, ls in [
        ('Raw', '原始(Raw)', 'blue', '-'),
        ('HSICS_Ind', 'HSICS行业中性', 'red', '--'),
        ('HSICS_IndSize', 'HSICS行业+市值中性', 'green', ':')
    ]:
        b = bt[f'{pref}{suf}']; cum = 1 + b.cumsum()
        ax.plot(cum.index.to_timestamp(), cum.values, label=label, color=color, linestyle=ls, linewidth=1.5)
    ax.set_title(title, fontsize=13, fontweight='bold'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('pict/v4_plot_hsics_cumulative.png', dpi=150, bbox_inches='tight'); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, pref, title in zip(axes, ['B_', 'C_'], ['B: 超额收益(相对HSI)', 'C: 超额收益(相对HSI)']):
    for suf, label, color, ls in [
        ('Raw', '原始(Raw)', 'blue', '-'),
        ('HSICS_Ind', 'HSICS行业中性', 'red', '--'),
        ('HSICS_IndSize', 'HSICS行业+市值中性', 'green', ':')
    ]:
        b = bt[f'{pref}{suf}']; common = b.index.intersection(hsi_ret.index)
        exc = b.reindex(common) - hsi_ret.reindex(common); ce = 1 + exc.cumsum()
        ax.plot(ce.index.to_timestamp(), ce.values, label=label, color=color, linestyle=ls, linewidth=1.5)
    ax.axhline(y=1, color='black', linewidth=0.5)
    ax.set_title(title, fontsize=13, fontweight='bold'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('pict/v4_plot_hsics_excess.png', dpi=150, bbox_inches='tight'); plt.close()

print("\nSaved: pict/v4_plot_hsics_cumulative.png, pict/v4_plot_hsics_excess.png")
print("Done!")
