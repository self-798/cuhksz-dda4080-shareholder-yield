"""Tracking error attribution + stock count sensitivity"""
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

print("="*80)
print("跟踪误差归因 + 持股数量敏感性分析")
print("="*80)

# ====== DATA ======
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
print("   Done.")

# ====== STRATEGY ======
print("\n[2/4] Running strategies with n_final = 10, 15, 20, 30, 40, 50...")

def compute_sy(pool, d36, p36):
    pool = pool.copy()
    dps = p36 * d36; cp = p36.iloc[-1] if len(p36) > 0 else pd.Series()
    if len(cp) > 0: pool['div_yield'] = pool['sid'].map(dps.mean() / cp).fillna(0)
    else: pool['div_yield'] = 0.0
    pool['buyback_yield'] = ((pool['buyback_36m']/3)/pool['mcap'])*100
    pool['buyback_yield'] = pool['buyback_yield'].replace([np.inf,-np.inf,np.nan],0)
    pool['score_raw'] = pool['div_yield'] + pool['buyback_yield']
    return pool.dropna(subset=['score_raw'])

def select(g, n_final=20, use_lowvol=False):
    cm = g.name
    if isinstance(cm, tuple): cm = cm[0]
    ed = cm.to_timestamp(how='end')
    d36 = dy_pivot.loc[:ed].tail(36); p36 = price_pivot_full.reindex(d36.index, method='ffill')
    has_div = d36.columns[d36.gt(0).any(axis=0)] if len(d36)>0 else pd.Index([])
    pool = g[(g['sid'].isin(has_div)) | (g['buyback_36m']>0)]
    adtv_th=g['adtv_3m'].quantile(0.2); mcap_th=g['mcap'].quantile(0.2)
    pool=pool[(pool['adtv_3m']>=adtv_th)&(pool['mcap']>=mcap_th)]
    if len(pool)<10: return []
    pool=compute_sy(pool,d36,p36)
    if len(pool)<10: return []
    cutoff=pool['score_raw'].quantile(0.9); pool=pool[pool['score_raw']<=cutoff]
    if len(pool)<10: return []
    n_top=max(n_final,int(len(pool)*0.20)); pool=pool.nlargest(n_top,'score_raw')
    nn=min(n_final,len(pool))
    return pool.nsmallest(nn,'volatility')['sid'].tolist() if use_lowvol else pool.nlargest(nn,'score_raw')['sid'].tolist()

sizes = [10, 15, 20, 30, 40, 50]
all_signals = {}
for s in sizes:
    for lv, label in [(False, 'B'), (True, 'C')]:
        name = f'{label}_n{s}'
        print(f"  {name}...", end=' ', flush=True)
        all_signals[name] = df_factors.groupby('date_m', group_keys=True).apply(
            lambda g, s=s, lv=lv: select(g, n_final=s, use_lowvol=lv))
        print("OK")

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

bt = {n: run_bt(all_signals[n]) for n in all_signals}

# ====== RESULTS ======
print("\n[3/4] Computing results...")
print("\n" + "="*100)
print("持股数量对跟踪误差和IR的影响")
print("="*100)

for strat_name, strat_label in [('B', 'STRATEGY B (纯SY+极端剔除)'), ('C', 'STRATEGY C (SY+极端剔除+低波)')]:
    print(f"\n{'='*80}")
    print(f"  {strat_label}")
    print(f"{'='*80}")
    print(f"{'n':<8} {'Ann Ret':<10} {'Ann Vol':<10} {'Sharpe':<8} {'Exc(HSI)':<10} {'TrackErr':<10} {'IR':<8} {'HitRate':<8}")
    print("-"*75)
    for s in sizes:
        name = f'{strat_name}_n{s}'
        r = bt[name]
        ar=r.mean()*12; av=r.std()*np.sqrt(12); sh=ar/av if av>0 else 0
        common=r.index.intersection(hsi_ret.index)
        exc=r.reindex(common)-hsi_ret.reindex(common)
        ae=exc.mean()*12; at=exc.std()*np.sqrt(12); ir=ae/at if at>0 else 0
        hit=(exc>0).mean()
        print(f"{s:<8} {ar:>8.2%}   {av:>8.2%}   {sh:>6.2f}   {ae:>+7.2%}     {at:>8.2%}    {ir:>6.2f}   {hit:>7.1%}")

# ====== TRACKING ERROR DECOMPOSITION ======
print("\n\n[4/4] Tracking error decomposition (n=20 vs n=50)...")

for strat_name in ['B', 'C']:
    for s in [20, 50]:
        name = f'{strat_name}_n{s}'
        r = bt[name]
        common = r.index.intersection(hsi_ret.index)
        r_aligned = r.reindex(common)
        hsi_aligned = hsi_ret.reindex(common)
        corr = np.corrcoef(r_aligned, hsi_aligned)[0,1]

        X = sm.add_constant(hsi_aligned.values)
        model = sm.OLS(r_aligned.values, X).fit()
        beta = model.params[1]

        exc = r_aligned - hsi_aligned
        te2_total = exc.var() * 12
        market_var = hsi_aligned.var() * 12
        systematic_te2 = (beta - 1)**2 * market_var
        idio_te2 = max(0, te2_total - systematic_te2)

        te = np.sqrt(te2_total)
        sys_pct = systematic_te2 / te2_total * 100 if te2_total > 0 else 0
        idio_pct = idio_te2 / te2_total * 100 if te2_total > 0 else 0

        print(f"\n  {name}:")
        print(f"    TE={te:.2%}, Corr(HSI)={corr:.3f}, Beta={beta:.3f}")
        print(f"    系统性TE (Beta≠1): {np.sqrt(systematic_te2):.2%} ({sys_pct:.0f}%)")
        print(f"    特异性TE (选股差异): {np.sqrt(idio_te2):.2%} ({idio_pct:.0f}%)")

# ====== PLOT ======
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, strat_name, title in zip(axes, ['B', 'C'], ['Strategy B (Trim)', 'Strategy C (Trim+LowVol)']):
    te_vals = []; ir_vals = []; n_vals = []
    for s in sizes:
        name = f'{strat_name}_n{s}'
        r = bt[name]
        common = r.index.intersection(hsi_ret.index)
        exc = r.reindex(common) - hsi_ret.reindex(common)
        ae = exc.mean() * 12; at = exc.std() * np.sqrt(12)
        te_vals.append(at * 100); ir_vals.append(ae/at if at>0 else 0)
        n_vals.append(s)

    ax.plot(n_vals, te_vals, 'o-', color='steelblue', linewidth=2, markersize=8, label='Tracking Error (%)')
    ax.set_xlabel('Number of Holdings', fontsize=11); ax.set_ylabel('Tracking Error (%)', fontsize=11, color='steelblue')
    ax.tick_params(axis='y', labelcolor='steelblue')

    ax2 = ax.twinx()
    ax2.plot(n_vals, ir_vals, 's--', color='coral', linewidth=2, markersize=8, label='Information Ratio')
    ax2.set_ylabel('Information Ratio (IR)', fontsize=11, color='coral')
    ax2.tick_params(axis='y', labelcolor='coral')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    ax.set_title(f'{title}\nHoldings vs Tracking Error & IR', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pict/v4_plot_te_nstocks.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: pict/v4_plot_te_nstocks.png")
print("Done!")
