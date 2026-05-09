"""Master script: load data once, regenerate ALL plots with Chinese fonts + additive cumulative"""
import pandas as pd, numpy as np, warnings, os, json
warnings.filterwarnings('ignore')
os.makedirs('pict', exist_ok=True)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.ticker as mtick
import matplotlib.font_manager as fm

# Chinese font
for f in fm.fontManager.ttflist:
    if f.name == 'Microsoft YaHei':
        fm.fontManager.addfont(f.fname); break
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

import statsmodels.api as sm

print("="*60)
print("MASTER PLOT REGENERATION — Chinese Fonts + Additive Cumulative")
print("="*60)

# ====== ALL DATA (load once) ======
print("\n[1/4] Loading all data...")
price_cols = pd.read_csv('data/hk_price.csv', usecols=['date', 'sid', 'close', 'AdjClose'])
price_cols['date'] = pd.to_datetime(price_cols['date']); price_cols['date_m'] = price_cols['date'].dt.to_period('M')
monthly_price = price_cols.sort_values('date').groupby(['date_m', 'sid']).last().reset_index()
mp_pivot = monthly_price.pivot(index='date_m', columns='sid', values='AdjClose')
monthly_ret = mp_pivot.pct_change(fill_method=None).shift(-1)

with pd.HDFStore('data/hk_shares.h5') as s:
    shares = s.get(s.keys()[0]).reset_index()
shares['sid'] = shares['order_book_id'].str[1:5] + '.HK'
shares['date_m'] = pd.to_datetime(shares['date']).dt.to_period('M')
shares_m = shares.sort_values('date').groupby(['date_m', 'sid']).last().reset_index()
mcap_df = pd.merge(monthly_price[['date_m','sid','close']], shares_m[['date_m','sid','total']], on=['date_m','sid'], how='inner')
mcap_df['mcap'] = mcap_df['close'] * mcap_df['total']
mcap_pvt = mcap_df.pivot(index='date_m', columns='sid', values='mcap')

price = pd.read_csv('data/hk_price.csv', usecols=['date', 'sid', 'close', 'amount'])
price['date'] = pd.to_datetime(price['date'])
pp = price.pivot(index='date', columns='sid', values='close').sort_index()
rp = pp.pct_change()
vp = rp.rolling(window=252, min_periods=100).std() * np.sqrt(252)
vm = vp.resample('ME').last().stack().reset_index(); vm.columns = ['date','sid','volatility']; vm['date_m'] = vm['date'].dt.to_period('M')
ap = price.pivot(index='date', columns='sid', values='amount').sort_index()
adp = ap.rolling(window=63, min_periods=21).mean()
adm = adp.resample('ME').last().stack().reset_index(); adm.columns = ['date','sid','adtv_3m']; adm['date_m'] = adm['date'].dt.to_period('M')

dy = pd.read_hdf('data/hk_dividendyield.h5')
if isinstance(dy, pd.DataFrame): dy = dy.iloc[:, 0]
dy = dy.dropna().reset_index(); dy.columns = ['date','sid','dy'] if dy.shape[1]==3 else ['date','dy']
dy['date'] = pd.to_datetime(dy['date']); dy['date_m'] = dy['date'].dt.to_period('M')
dy_pvt = dy.pivot(index='date', columns='sid', values='dy').sort_index()

hsci = pd.read_csv('data/HSCI.csv'); hsci['date'] = pd.to_datetime(hsci['date'])
hsci['is_hsci'] = 1; hsci = hsci.drop_duplicates(subset=['date','sid']); hsci['date_m'] = hsci['date'].dt.to_period('M')

bb = pd.read_csv('data/em_buyback_filtered.csv'); bb['date'] = pd.to_datetime(bb['日期'])
bb['sid'] = bb['股票代码'].astype(str).str.zfill(5).str[1:5] + '.HK'; bb['date_m'] = bb['date'].dt.to_period('M')
mbb = bb.groupby(['date_m','sid'])['回购总额'].sum().reset_index()
mbbp = mbb.pivot(index='date_m', columns='sid', values='回购总额').fillna(0).sort_index()
bb36 = mbbp.rolling(window=36, min_periods=12).sum().stack().reset_index(); bb36.columns = ['date_m','sid','buyback_36m']

df = hsci[['date_m','sid','is_hsci']].merge(mcap_df[['date_m','sid','mcap']], on=['date_m','sid'], how='inner'
).merge(vm[['date_m','sid','volatility']], on=['date_m','sid'], how='inner'
).merge(adm[['date_m','sid','adtv_3m']], on=['date_m','sid'], how='inner'
).merge(dy[['date_m','sid','dy']], on=['date_m','sid'], how='left'
).merge(bb36, on=['date_m','sid'], how='left')
df['buyback_36m'] = df['buyback_36m'].fillna(0); df = df.drop_duplicates(subset=['date_m','sid'])

hsi = pd.read_csv('data/HSI_index.csv'); hsi['date'] = pd.to_datetime(hsi['date']); hsi['date_m'] = hsi['date'].dt.to_period('M')
hsi_m = hsi.sort_values('date').groupby('date_m').last().reset_index().set_index('date_m')['close']
hsi_r = hsi_m.pct_change().shift(-1).dropna(); hsi_r.index = hsi_r.index + 1
BT = pd.Period('2012-01', 'M'); hsi_r = hsi_r[hsi_r.index >= BT + 1]

hkdc = pd.read_csv('data/hk_stock_connect_high_div.csv'); hkdc['date'] = pd.to_datetime(hkdc['date']); hkdc['date_m'] = hkdc['date'].dt.to_period('M')
hkdcm = hkdc.sort_values('date').groupby('date_m').last().reset_index().set_index('date_m')['close']
hkdc_r = hkdcm.pct_change().shift(-1).dropna(); hkdc_r.index = hkdc_r.index + 1; hkdc_r = hkdc_r[hkdc_r.index >= BT + 1]

ff3 = pd.read_csv('data/hk_ff3_factors.csv'); ff3['date_m'] = pd.to_datetime(ff3['date_m']).dt.to_period('M'); ff3 = ff3.set_index('date_m')
print("   Done.")

# ====== STRATEGIES ======
print("\n[2/4] Running strategies...")
def compute_sy(pool, d36, p36):
    pool = pool.copy()
    dps = p36 * d36; cp = p36.iloc[-1] if len(p36) > 0 else pd.Series()
    if len(cp) > 0: pool['div_yield'] = pool['sid'].map(dps.mean()/cp).fillna(0)
    else: pool['div_yield'] = 0.0
    pool['buyback_yield'] = ((pool['buyback_36m']/3)/pool['mcap'])*100
    pool['buyback_yield'] = pool['buyback_yield'].replace([np.inf,-np.inf,np.nan],0)
    pool['score_raw'] = pool['div_yield'] + pool['buyback_yield']
    return pool.dropna(subset=['score_raw'])

def select_raw(group, trim=True, lowvol=False):
    cm = group.name
    if isinstance(cm, tuple): cm = cm[0]
    ed = cm.to_timestamp(how='end'); d36 = dy_pvt.loc[:ed].tail(36); p36 = pp.reindex(d36.index, method='ffill')
    has_div = d36.columns[d36.gt(0).any(axis=0)] if len(d36)>0 else pd.Index([])
    pool = group[(group['sid'].isin(has_div))|(group['buyback_36m']>0)]
    adtv_th = group['adtv_3m'].quantile(0.2); mcap_th = group['mcap'].quantile(0.2)
    pool = pool[(pool['adtv_3m']>=adtv_th)&(pool['mcap']>=mcap_th)]
    if len(pool)<10: return []
    pool = compute_sy(pool, d36, p36)
    if len(pool)<10: return []
    if trim:
        cutoff = pool['score_raw'].quantile(0.9); pool = pool[pool['score_raw']<=cutoff]
        if len(pool)<10: return []
    n_top = max(10, int(len(pool)*0.20)); pool = pool.nlargest(n_top, 'score_raw')
    nn = min(20, len(pool))
    return pool.nsmallest(nn, 'volatility')['sid'].tolist() if lowvol else pool.nlargest(nn, 'score_raw')['sid'].tolist()

for lv, nm in [(False,'B'), (True,'C')]:
    print(f"  {nm}...", end=' ', flush=True)
    globals()[f'sig_{nm}'] = df.groupby('date_m', group_keys=True).apply(lambda g: select_raw(g, True, lv))
    print("OK")
sig_A = df.groupby('date_m', group_keys=True).apply(lambda g: select_raw(g, False, False)); print("  A OK")

def run_bt(sig):
    dates = sig.index[sig.index>=BT]; rets = []; current = []
    for ti, dm in enumerate(dates):
        if dm not in monthly_ret.index: continue
        if ti%3==0:
            ns = sig.loc[dm]
            if len(ns)>0: current = ns
        if len(current)==0: continue
        rets.append((dm+1, monthly_ret.loc[dm].reindex(current).fillna(0).mean()))
    return pd.Series([r for _,r in rets], index=[d for d,_ in rets])

bt = {n: run_bt(globals()[f'sig_{n}']) for n in ['A','B','C']}
print("   Backtests done.")

# ====== GENERATE ALL PLOTS ======
print("\n[3/4] Generating plots...")

# Plot 1: Main cumulative
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
for name, b, color, ls in [('A: 无剔除极端', bt['A'], 'red', '-'), ('B: 剔除极端(10%)', bt['B'], 'blue', '-'),
    ('C: 剔除+低波', bt['C'], 'green', '-'), ('恒生指数(HSI)', hsi_r, 'gray', '--'), ('港股通高股息', hkdc_r, 'orange', '--')]:
    common = b.index.intersection(hsi_r.index)
    cum = 1 + b.reindex(common).cumsum()
    ax1.plot(cum.index.to_timestamp(), cum.values, label=name, color=color, linestyle=ls, linewidth=1.5)
ax1.set_title('综合收益率策略累计净值（算术累加法）', fontsize=14); ax1.set_ylabel('累计净值'); ax1.legend(fontsize=9, loc='upper left'); ax1.grid(True, alpha=0.3)
for name, b, color in [('A: 无剔除', bt['A'], 'red'), ('B: 剔除极端', bt['B'], 'blue'), ('C: 剔除+低波', bt['C'], 'green'), ('HSI', hsi_r, 'gray')]:
    cum = 1 + b.cumsum(); dd = cum/cum.cummax().clip(lower=1.0) - 1
    ax2.fill_between(dd.index.to_timestamp(), dd.values, 0, alpha=0.3, label=name, color=color)
ax2.set_ylabel('回撤'); ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0)); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('pict/v4_plot_01.png', dpi=150, bbox_inches='tight'); plt.close(); print("  01")

# Plot 2: Excess over HSI
fig, ax = plt.subplots(figsize=(14, 6))
for name, b, color in [('B: 剔除极端', bt['B'], 'blue'), ('C: 剔除+低波', bt['C'], 'green')]:
    common = b.index.intersection(hsi_r.index); exc = b.reindex(common) - hsi_r.reindex(common)
    ax.plot(exc.index.to_timestamp(), (1+exc.cumsum()).values, label=name, color=color, linewidth=1.5)
ax.axhline(y=1, color='black', linewidth=0.5)
ax.set_title('累计超额收益（相对恒生指数，算术累加法）', fontsize=14); ax.set_ylabel('累计超额'); ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('pict/v4_plot_02.png', dpi=150, bbox_inches='tight'); plt.close(); print("  02")

# Plot 3: Excess over HKDIV
fig, ax = plt.subplots(figsize=(14, 6))
for name, b, color in [('B: 剔除极端', bt['B'], 'blue'), ('C: 剔除+低波', bt['C'], 'green')]:
    common = b.index.intersection(hkdc_r.index); exc = b.reindex(common) - hkdc_r.reindex(common)
    ax.plot(exc.index.to_timestamp(), (1+exc.cumsum()).values, label=name, color=color, linewidth=1.5)
ax.axhline(y=1, color='black', linewidth=0.5)
ax.set_title('累计超额收益（相对港股通高股息指数，算术累加法）', fontsize=14); ax.set_ylabel('累计超额'); ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('pict/v4_plot_03.png', dpi=150, bbox_inches='tight'); plt.close(); print("  03")

# Plot 4: Drawdown comparison
fig, ax = plt.subplots(figsize=(14, 6))
for name, b, color, ls in [('A: 无剔除', bt['A'], 'red', '-'), ('B: 剔除极端', bt['B'], 'blue', '-'),
    ('C: 剔除+低波', bt['C'], 'green', '-'), ('恒生指数(HSI)', hsi_r, 'gray', '--'), ('港股通高股息', hkdc_r, 'orange', '--')]:
    cum = 1 + b.cumsum(); dd = cum/cum.cummax().clip(lower=1.0) - 1
    ax.plot(dd.index.to_timestamp(), dd.values, label=name, color=color, linestyle=ls, linewidth=1.5)
ax.set_title('策略回撤对比', fontsize=14); ax.set_ylabel('回撤'); ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0)); ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('pict/v4_plot_08.png', dpi=150, bbox_inches='tight'); plt.close(); print("  08")

# FF3 Alpha bar chart
print("\n[4/4] FF3 and TE plots...")
alphas = {'A(无剔除)': -0.048, 'B(剔除极端)': 0.783, 'C(剔除+低波)': 0.714, 'HSI': -0.050, '港股通高股息': 0.345}
tstats = {'A(无剔除)': -0.18, 'B(剔除极端)': 4.42, 'C(剔除+低波)': 6.06, 'HSI': -0.45, '港股通高股息': 2.30}
fig, ax = plt.subplots(figsize=(12, 5)); names = list(alphas.keys()); vals = list(alphas.values())
colors_f = ['red' if v < 0 else 'steelblue' for v in vals]
bars = ax.bar(names, vals, color=colors_f, alpha=0.85)
for bar, v, t in zip(bars, vals, [tstats[n] for n in names]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02 if v>=0 else bar.get_height()-0.08,
            f'{v:+.3f}% (t={t:.1f})', ha='center', fontsize=9, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.5); ax.set_title('Fama-French 三因子Alpha对比（月度）', fontsize=14)
ax.set_ylabel('月度Alpha (%)'); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout(); plt.savefig('pict/v4_plot_ff3_neutral_ir_comparison.png', dpi=150, bbox_inches='tight'); plt.close(); print("  FF3 alpha")

# Performance summary table as plot
fig, ax = plt.subplots(figsize=(14, 6)); ax.axis('off')
table_data = [
    ['策略', '年化收益', '年化波动', '夏普比率', '最大回撤', 'FF3 α/月', '超额(HSI)', '信息比率', '月胜率'],
    ['A: 无剔除', '5.78%', '30.25%', '0.19', '-74.49%', '-0.048%', '+2.29%', '0.17', '51.5%'],
    ['B: 剔除极端', '14.06%', '22.26%', '0.63', '-35.19%', '0.783%***', '+11.99%', '1.11', '64.4%'],
    ['C: 剔除+低波', '11.40%', '16.24%', '0.70', '-27.69%', '0.714%***', '+7.35%', '0.66', '54.8%'],
    ['HSI基准', '3.49%', '20.05%', '0.17', '-55.34%', '-0.050%', '—', '—', '—'],
    ['港股通高股息', '9.05%', '24.76%', '0.37', '-50.94%', '0.345%**', '+5.56%', '0.34', '60.0%'],
]
tbl = ax.table(cellText=table_data, cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
for i in range(len(table_data[0])): tbl[0, i].set_facecolor('#4472C4'); tbl[0, i].set_text_props(color='white', fontweight='bold')
for j in [2]:  # Highlight B row
    for i in range(len(table_data[0])): tbl[j, i].set_facecolor('#D6E4F0')
ax.set_title('策略绩效总览表', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout(); plt.savefig('pict/v4_plot_05.png', dpi=150, bbox_inches='tight'); plt.close(); print("  05")

# Neutralization summary
fig, ax = plt.subplots(figsize=(14, 6)); ax.axis('off')
neut_data = [
    ['中性化方法', 'B: ΔIR', 'B: Δ月胜率', 'C: ΔIR', 'C: Δ月胜率', 'B: Δ跟踪误差', 'C: Δ跟踪误差'],
    ['粗略行业(代码区间)', '-0.21', '-4.1%', '+0.03', '+2.7%', '+0.19%', '+0.43%'],
    ['HSICS行业(精确)', '-0.17', '-6.8%', '+0.01', '+2.1%', '+0.39%', '+0.62%'],
    ['FF3市值中性化', '-0.04', '+0.7%', '-0.05', '+1.4%', '+0.25%', '+0.03%'],
    ['FF3价值(BM)中性化', '-0.27', '-3.4%', '-0.21', '-1.4%', '-0.34%', '+0.04%'],
    ['FF3市值+BM中性化', '-0.26', '-5.5%', '-0.26', '-5.5%', '+0.22%', '+0.35%'],
    ['HSICS行业+市值中性化', '-0.20', '-4.1%', '-0.04', '+2.7%', '+0.71%', '+0.84%'],
]
tbl2 = ax.table(cellText=neut_data, cellLoc='center', loc='center')
tbl2.auto_set_font_size(False); tbl2.set_fontsize(9)
for i in range(len(neut_data[0])): tbl2[0, i].set_facecolor('#4472C4'); tbl2[0, i].set_text_props(color='white', fontweight='bold')
ax.set_title('所有中性化方法效果汇总', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout(); plt.savefig('pict/v4_plot_06.png', dpi=150, bbox_inches='tight'); plt.close(); print("  06")

# Extreme value characterization
fig, ax = plt.subplots(figsize=(14, 8)); ax.axis('off')
ext_data = [
    ['指标', '数值', '含义'],
    ['被剔除股票平均市值分位数', '29.7%', '小于70%的股票'],
    ['处于市值后20%的比例', '44.8%', '近半是小微盘'],
    ['处于市值后50%的比例', '78.7%', '近八成在市值下半段'],
    ['被剔除股票月均收益', '0.44%', '低于正常组(0.60%)'],
    ['年化收益差距', '-1.96%', '被剔除=价值陷阱'],
    ['被剔除股票月波动率', '11.27%', '高于正常组(10.21%)'],
    ['跑赢正常组的月份', '44.8%', '不到一半'],
    ['SY得分中位数(极端)', '11.97', '正常组仅3.04，差3.9x'],
    ['极端SY中小盘占比(2025-09)', '56%', '右尾被小盘股主导'],
]
tbl3 = ax.table(cellText=ext_data, cellLoc='center', loc='center')
tbl3.auto_set_font_size(False); tbl3.set_fontsize(10)
for i in range(len(ext_data[0])): tbl3[0, i].set_facecolor('#C00000'); tbl3[0, i].set_text_props(color='white', fontweight='bold')
ax.set_title('被剔除极端值股票特征总览', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout(); plt.savefig('pict/v4_plot_07.png', dpi=150, bbox_inches='tight'); plt.close(); print("  07")

# Stock count sensitivity
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# B data
b_data = {10:(0.87,12.74,11.13), 15:(1.02,11.70,11.89), 20:(1.11,10.83,11.99), 30:(1.05,10.19,10.71), 40:(1.08,9.68,10.43), 50:(0.97,9.32,9.03)}
c_data = {10:(0.68,12.36,8.42), 15:(0.65,11.33,7.41), 20:(0.66,11.18,7.35), 30:(0.80,10.39,8.29), 40:(0.87,9.64,8.43), 50:(0.95,9.07,8.59)}
for ax, dat, title in zip(axes, [b_data, c_data], ['策略B: 纯SY+极端剔除', '策略C: SY+极端剔除+低波']):
    ns = sorted(dat.keys())
    ax.plot(ns, [dat[n][1] for n in ns], 'o-', color='steelblue', linewidth=2, markersize=8, label='跟踪误差(%)')
    ax.set_xlabel('持股数量'); ax.set_ylabel('跟踪误差(%)', color='steelblue'); ax.tick_params(axis='y', labelcolor='steelblue')
    ax2b = ax.twinx()
    ax2b.plot(ns, [dat[n][0] for n in ns], 's--', color='coral', linewidth=2, markersize=8, label='信息比率')
    ax2b.set_ylabel('信息比率', color='coral'); ax2b.tick_params(axis='y', labelcolor='coral')
    l1,lb1=ax.get_legend_handles_labels(); l2,lb2=ax2b.get_legend_handles_labels()
    ax.legend(l1+l2,lb1+lb2,loc='center right'); ax.set_title(title, fontsize=13); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('pict/v4_plot_te_ir_vs_n.png', dpi=150, bbox_inches='tight'); plt.close(); print("  TE")

print("\n=== ALL PLOTS REGENERATED WITH CHINESE FONTS ===")
print("v4_plot_01,02,03,05,06,07,08 + ff3/te plots")
