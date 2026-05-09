"""Regenerate all plots with additive cumulative returns (cumsum, not cumprod)"""
import pandas as pd, numpy as np, warnings, os
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

print("="*60)
print("REGENERATING ALL PLOTS WITH ADDITIVE CUMULATIVE RETURNS")
print("="*60)

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
mcap_df = pd.merge(monthly_price[['date_m', 'sid', 'close']], shares_monthly[['date_m', 'sid', 'total']], on=['date_m', 'sid'], how='inner')
mcap_df['mcap'] = mcap_df['close'] * mcap_df['total']
mcap_pivot = mcap_df.pivot(index='date_m', columns='sid', values='mcap')

price = pd.read_csv('data/hk_price.csv', usecols=['date', 'sid', 'close', 'amount'])
price['date'] = pd.to_datetime(price['date'])
price_pivot_full = price.pivot(index='date', columns='sid', values='close').sort_index()
ret_pvt = price_pivot_full.pct_change()
vol_pvt = ret_pvt.rolling(window=252, min_periods=100).std() * np.sqrt(252)
vol_m = vol_pvt.resample('ME').last().stack().reset_index()
vol_m.columns = ['date', 'sid', 'volatility']; vol_m['date_m'] = vol_m['date'].dt.to_period('M')

amt_pvt = price.pivot(index='date', columns='sid', values='amount').sort_index()
adv_pvt = amt_pvt.rolling(window=63, min_periods=21).mean()
adv_m = adv_pvt.resample('ME').last().stack().reset_index()
adv_m.columns = ['date', 'sid', 'adtv_3m']; adv_m['date_m'] = adv_m['date'].dt.to_period('M')

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

hsi = pd.read_csv('data/HSI_index.csv'); hsi['date'] = pd.to_datetime(hsi['date'])
hsi['date_m'] = hsi['date'].dt.to_period('M')
hsi_m = hsi.sort_values('date').groupby('date_m').last().reset_index().set_index('date_m')['close']
hsi_ret = hsi_m.pct_change().shift(-1).dropna(); hsi_ret.index = hsi_ret.index + 1
BACKTEST_START = pd.Period('2012-01', 'M')
hsi_ret = hsi_ret[hsi_ret.index >= BACKTEST_START + 1]

hkdc = pd.read_csv('data/hk_stock_connect_high_div.csv'); hkdc['date'] = pd.to_datetime(hkdc['date'])
hkdc['date_m'] = hkdc['date'].dt.to_period('M')
hkdc_m = hkdc.sort_values('date').groupby('date_m').last().reset_index().set_index('date_m')['close']
hkdc_ret = hkdc_m.pct_change().shift(-1).dropna(); hkdc_ret.index = hkdc_ret.index + 1
hkdc_ret = hkdc_ret[hkdc_ret.index >= BACKTEST_START + 1]

# ====== STRATEGY ======
def compute_sy(pool, d36, p36):
    pool = pool.copy()
    dps = p36 * d36; cp = p36.iloc[-1] if len(p36) > 0 else pd.Series()
    if len(cp) > 0: pool['div_yield'] = pool['sid'].map(dps.mean() / cp).fillna(0)
    else: pool['div_yield'] = 0.0
    pool['buyback_yield'] = ((pool['buyback_36m']/3)/pool['mcap'])*100
    pool['buyback_yield'] = pool['buyback_yield'].replace([np.inf,-np.inf,np.nan],0)
    pool['score_raw'] = pool['div_yield'] + pool['buyback_yield']
    return pool.dropna(subset=['score_raw'])

def select(g, use_lowvol=False):
    cm = g.name
    if isinstance(cm, tuple): cm = cm[0]
    ed = cm.to_timestamp(how='end')
    d36 = dy_pivot.loc[:ed].tail(36); p36 = price_pivot_full.reindex(d36.index, method='ffill')
    has_div = d36.columns[d36.gt(0).any(axis=0)] if len(d36)>0 else pd.Index([])
    pool = g[(g['sid'].isin(has_div)) | (g['buyback_36m']>0)]
    adtv_th=g['adtv_3m'].quantile(0.2); mcap_th=g['mcap'].quantile(0.2)
    pool=pool[(pool['adtv_3m']>=adtv_th)&(pool['mcap']>=mcap_th)]
    if len(pool)<10: return None, None, None
    pool=compute_sy(pool,d36,p36)
    if len(pool)<10: return None, None, None
    return pool, d36, p36

print("\n[2/4] Running strategies A, B, C, D...")

def select_A(g):
    pool, d36, p36 = select(g)
    if pool is None or len(pool)==0: return []
    # No extreme trim for A
    n_top=max(10,int(len(pool)*0.20)); pool=pool.nlargest(n_top,'score_raw')
    return pool.nlargest(min(20,len(pool)),'score_raw')['sid'].tolist()

def select_B(g):
    pool, d36, p36 = select(g)
    if pool is None or len(pool)==0: return []
    cutoff=pool['score_raw'].quantile(0.9); pool=pool[pool['score_raw']<=cutoff]
    if len(pool)<10: return []
    n_top=max(10,int(len(pool)*0.20)); pool=pool.nlargest(n_top,'score_raw')
    return pool.nlargest(min(20,len(pool)),'score_raw')['sid'].tolist()

def select_C(g):
    pool, d36, p36 = select(g)
    if pool is None or len(pool)==0: return []
    cutoff=pool['score_raw'].quantile(0.9); pool=pool[pool['score_raw']<=cutoff]
    if len(pool)<10: return []
    n_top=max(10,int(len(pool)*0.20)); pool=pool.nlargest(n_top,'score_raw')
    return pool.nsmallest(min(20,len(pool)),'volatility')['sid'].tolist()

def select_D(g):
    pool, d36, p36 = select(g)
    if pool is None or len(pool)==0: return []
    cutoff=pool['score_raw'].quantile(0.9); pool=pool[pool['score_raw']<=cutoff]
    if len(pool)<10: return []
    n_top=max(10,int(len(pool)*0.20)); pool=pool.nlargest(n_top,'score_raw')
    pool=pool[pool['fcff']>0] if 'fcff' in pool.columns else pool
    if len(pool)<10: return []
    return pool.nlargest(min(20,len(pool)),'score_raw')['sid'].tolist()

sigs_A = df_factors.groupby('date_m', group_keys=True).apply(select_A)
sigs_B = df_factors.groupby('date_m', group_keys=True).apply(select_B)
sigs_C = df_factors.groupby('date_m', group_keys=True).apply(select_C)

# For D, need fcff column
fcff_data = pd.read_csv('data/hk_fcff.csv')
fcff_data['sid'] = fcff_data['windcode']
fcff_data['report_year'] = pd.to_datetime(fcff_data['report_period'], format='%Y%m%d').dt.year
# Simple: latest fcff per stock
fcff_latest = fcff_data.groupby('sid')['fcff_simple_hkd'].last()
df_factors['fcff'] = df_factors['sid'].map(fcff_latest).fillna(0)
sigs_D = df_factors.groupby('date_m', group_keys=True).apply(select_D)

def run_bt(sig):
    dates=sig.index[sig.index>=BACKTEST_START]; rets=[]; current=[]
    for ti,dm in enumerate(dates):
        if dm not in monthly_ret.index: continue
        if ti%3==0:
            ns=sig.loc[dm]
            if len(ns)>0: current=ns
        if len(current)==0: continue
        nxt=monthly_ret.loc[dm].reindex(current).fillna(0)
        rets.append((dm+1,nxt.mean()))
    return pd.Series([r for _,r in rets],index=[d for d,_ in rets])

bt_A = run_bt(sigs_A); bt_B = run_bt(sigs_B); bt_C = run_bt(sigs_C); bt_D = run_bt(sigs_D)

# ====== PLOT 1: Cumulative returns (ADDITIVE) ======
print("\n[3/4] Generating additive cumulative return plots...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

# All cumulative returns as 1 + cumsum
for name, bt, color, ls in [
    ('A: 无剔除', bt_A, 'red', '-'), ('B: 剔除极端', bt_B, 'blue', '-'),
    ('C: 剔除+低波', bt_C, 'green', '-'), ('D: 剔除+FCFF', bt_D, 'purple', '-'),
    ('HSI', hsi_ret, 'gray', '--'), ('港股通高股息', hkdc_ret, 'orange', '--')
]:
    common = bt.index.intersection(hsi_ret.index)
    cum = 1 + bt.reindex(common).cumsum()
    ax1.plot(cum.index.to_timestamp(), cum.values, label=name, color=color, linestyle=ls, linewidth=1.5)
ax1.set_title('策略净值曲线（算术累加法）', fontsize=14, fontweight='bold')
ax1.set_ylabel('累计净值 (1 + 累计收益率)', fontsize=12)
ax1.legend(fontsize=10, loc='upper left'); ax1.grid(True, alpha=0.3); ax1.axhline(y=1, color='black', linewidth=0.5)

# Drawdowns
for name, bt, color in [('A', bt_A, 'red'), ('B', bt_B, 'blue'), ('C', bt_C, 'green'), ('HSI', hsi_ret, 'gray')]:
    cum = 1 + bt.cumsum()
    dd = cum / cum.cummax().clip(lower=1.0) - 1
    ax2.fill_between(dd.index.to_timestamp(), dd.values, 0, alpha=0.3, label=name, color=color)
ax2.set_ylabel('回撤', fontsize=12); ax2.set_xlabel('日期', fontsize=12)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax2.legend(fontsize=9, loc='lower left'); ax2.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('pict/v4_plot_01.png', dpi=150, bbox_inches='tight'); plt.close()

# ====== PLOT 2: Excess over HSI ======
fig, ax = plt.subplots(figsize=(14, 6))
for name, bt, color in [('B', bt_B, 'blue'), ('C', bt_C, 'green'), ('D', bt_D, 'purple')]:
    common = bt.index.intersection(hsi_ret.index)
    exc = bt.reindex(common) - hsi_ret.reindex(common)
    cum_exc = 1 + exc.cumsum()
    ax.plot(cum_exc.index.to_timestamp(), cum_exc.values, label=name, color=color, linewidth=1.5)
ax.axhline(y=1, color='black', linewidth=0.5)
ax.set_title('累计超额收益（相对HSI，算术累加法）', fontsize=14, fontweight='bold')
ax.set_ylabel('累计超额 (1 + 累计超额收益率)', fontsize=12)
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('pict/v4_plot_02.png', dpi=150, bbox_inches='tight'); plt.close()

# ====== PLOT 3: Excess over HKDIV ======
fig, ax = plt.subplots(figsize=(14, 6))
for name, bt, color in [('B', bt_B, 'blue'), ('C', bt_C, 'green')]:
    common = bt.index.intersection(hkdc_ret.index)
    exc = bt.reindex(common) - hkdc_ret.reindex(common)
    cum_exc = 1 + exc.cumsum()
    ax.plot(cum_exc.index.to_timestamp(), cum_exc.values, label=name, color=color, linewidth=1.5)
ax.axhline(y=1, color='black', linewidth=0.5)
ax.set_title('累计超额收益（相对港股通高股息，算术累加法）', fontsize=14, fontweight='bold')
ax.set_ylabel('累计超额 (1 + 累计超额收益率)', fontsize=12)
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('pict/v4_plot_03.png', dpi=150, bbox_inches='tight'); plt.close()

# ====== PLOT 4: Market cap group cumulative ======
print("[4/4] Market cap & double sort plots...")

# Simple mcap group plot
mcap_grp_returns = {'Small': [], 'Mid': [], 'Large': [], 'dates': []}
for date_m, group in df_factors.groupby('date_m'):
    if date_m < BACKTEST_START: continue
    if date_m not in monthly_ret.index: continue
    if len(group) < 20: continue
    adtv_th = group['adtv_3m'].quantile(0.2); mcap_th = group['mcap'].quantile(0.2)
    pool = group[(group['adtv_3m']>=adtv_th)&(group['mcap']>=mcap_th)].copy()
    if len(pool) < 20: continue
    ed = date_m.to_timestamp(how='end')
    d36 = dy_pivot.loc[:ed].tail(36); p36 = price_pivot_full.reindex(d36.index, method='ffill')
    has_div = d36.columns[d36.gt(0).any(axis=0)] if len(d36)>0 else pd.Index([])
    pool = pool[(pool['sid'].isin(has_div))|(pool['buyback_36m']>0)]
    if len(pool) < 20: continue
    pool = compute_sy(pool, d36, p36)
    if len(pool) < 20: continue
    cutoff = pool['score_raw'].quantile(0.9); pool = pool[pool['score_raw']<=cutoff]
    if len(pool) < 20: continue
    n_top = int(len(pool)*0.35); pool = pool.nlargest(n_top, 'score_raw')
    if len(pool) < 10: continue
    mcap_lo = pool['mcap'].quantile(0.333); mcap_hi = pool['mcap'].quantile(0.667)
    nxt = monthly_ret.loc[date_m]
    for label, sids in [('Small', pool[pool['mcap']<=mcap_lo]['sid']),
                         ('Mid', pool[(pool['mcap']>mcap_lo)&(pool['mcap']<=mcap_hi)]['sid']),
                         ('Large', pool[pool['mcap']>mcap_hi]['sid'])]:
        r = nxt.reindex(sids).fillna(0).mean() if len(sids)>0 else 0.0
        mcap_grp_returns[label].append(r)
    mcap_grp_returns['dates'].append(date_m+1)

fig, ax = plt.subplots(figsize=(12, 5))
for grp, color in [('Small','red'),('Mid','blue'),('Large','green')]:
    r_series = pd.Series(mcap_grp_returns[grp], index=mcap_grp_returns['dates'])
    cum = 1 + r_series.cumsum()
    ax.plot(cum.index.to_timestamp(), cum.values, label=grp, color=color, linewidth=1.5)
ax.set_title('市值分组累计收益（Top 35% SY，算术累加法）', fontsize=13, fontweight='bold')
ax.set_ylabel('累计净值', fontsize=11); ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('pict/v4_plot_04.png', dpi=150, bbox_inches='tight'); plt.close()

# Drawdown comparison
fig, ax = plt.subplots(figsize=(14, 6))
for name, bt, color in [('A: 无剔除', bt_A, 'red'), ('B: 剔除极端', bt_B, 'blue'),
                         ('C: 剔除+低波', bt_C, 'green'), ('D: 剔除+FCFF', bt_D, 'purple'),
                         ('HSI', hsi_ret, 'gray')]:
    cum = 1 + bt.cumsum()
    dd = cum / cum.cummax().clip(lower=1.0) - 1
    ax.plot(dd.index.to_timestamp(), dd.values, label=name, color=color, linewidth=1.5, alpha=0.8)
ax.set_title('策略回撤对比', fontsize=14, fontweight='bold')
ax.set_ylabel('回撤', fontsize=12); ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('pict/v4_plot_08.png', dpi=150, bbox_inches='tight'); plt.close()

print("\nSaved: v4_plot_01, 02, 03, 04, 08")
print("All additive cumulative plots regenerated!")
