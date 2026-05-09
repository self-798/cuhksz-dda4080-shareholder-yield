"""Tracking error attribution + stock count sensitivity — plots for report"""
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

# ====== DATA ======
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

# Collect all metrics
results = {'B': [], 'C': []}
for s in sizes:
    for sn in ['B', 'C']:
        name = f'{sn}_n{s}'; r = bt[name]
        ar=r.mean()*12; av=r.std()*np.sqrt(12); sh=ar/av if av>0 else 0
        common=r.index.intersection(hsi_ret.index)
        r_aligned=r.reindex(common); hsi_aligned=hsi_ret.reindex(common)
        exc=r_aligned-hsi_aligned; ae=exc.mean()*12; at=exc.std()*np.sqrt(12)
        ir=ae/at if at>0 else 0; hit=(exc>0).mean()
        corr=np.corrcoef(r_aligned,hsi_aligned)[0,1]
        X=sm.add_constant(hsi_aligned.values); model=sm.OLS(r_aligned.values,X).fit()
        beta=model.params[1]
        te2_total=exc.var()*12; market_var=hsi_aligned.var()*12
        systematic_te2=(beta-1)**2*market_var; idio_te2=max(0,te2_total-systematic_te2)
        sys_pct=systematic_te2/te2_total*100 if te2_total>0 else 0
        results[sn].append({'n':s,'ar':ar,'av':av,'sh':sh,'ae':ae,'te':at,'ir':ir,
                           'hit':hit,'beta':beta,'corr':corr,'sys_te':np.sqrt(systematic_te2),
                           'idio_te':np.sqrt(idio_te2),'sys_pct':sys_pct})

# ====== PLOT 1: IR & TE vs n ======
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, sn, title in zip(axes, ['B', 'C'], ['Strategy B (纯SY+极端剔除)', 'Strategy C (SY+极端剔除+低波)']):
    d = results[sn]; ns=[r['n'] for r in d]
    ax.plot(ns,[r['te']*100 for r in d],'o-',color='steelblue',linewidth=2,markersize=8,label='Tracking Error (%)')
    ax.set_xlabel('持股数量',fontsize=11); ax.set_ylabel('跟踪误差 (%)',fontsize=11,color='steelblue')
    ax.tick_params(axis='y',labelcolor='steelblue')
    ax2=ax.twinx()
    ax2.plot(ns,[r['ir'] for r in d],'s--',color='coral',linewidth=2,markersize=8,label='Information Ratio')
    ax2.set_ylabel('信息比率 (IR)',fontsize=11,color='coral')
    ax2.tick_params(axis='y',labelcolor='coral')
    l1,lbl1=ax.get_legend_handles_labels(); l2,lbl2=ax2.get_legend_handles_labels()
    ax.legend(l1+l2,lbl1+lbl2,loc='center right'); ax.set_title(title,fontsize=13,fontweight='bold')
    ax.grid(True,alpha=0.3)
    # Mark optimal
    best_n = ns[np.argmax([r['ir'] for r in d])]
    ax.axvline(x=best_n,color='red',linestyle=':',alpha=0.5)
    ax.annotate(f'IR最优: n={best_n}',xy=(best_n,ax.get_ylim()[1]*0.9),fontsize=9,color='red')
plt.tight_layout(); plt.savefig('pict/v4_plot_te_ir_vs_n.png',dpi=150,bbox_inches='tight'); plt.close()

# ====== PLOT 2: Beta & TE decomposition ======
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, sn, title in zip(axes, ['B', 'C'], ['Strategy B Beta & TE分解', 'Strategy C Beta & TE分解']):
    d = results[sn]; ns=[r['n'] for r in d]
    betas=[r['beta'] for r in d]
    ax.plot(ns,betas,'o-',color='darkgreen',linewidth=2,markersize=8,label='Beta (相对HSI)')
    ax.axhline(y=1,color='black',linestyle='--',linewidth=0.8,alpha=0.5)
    ax.fill_between(ns,[r['sys_te']*100 for r in d],alpha=0.3,color='orange',label='系统性TE (%)')
    ax.fill_between(ns,[r['sys_te']*100 for r in d],[r['sys_te']*100+r['idio_te']*100 for r in d],
                    alpha=0.3,color='steelblue',label='特质性TE (%)')
    ax.set_xlabel('持股数量',fontsize=11); ax.set_ylabel('Beta / 跟踪误差 (%)',fontsize=11)
    ax.set_title(title,fontsize=13,fontweight='bold'); ax.legend(fontsize=9); ax.grid(True,alpha=0.3)
plt.tight_layout(); plt.savefig('pict/v4_plot_te_decomposition.png',dpi=150,bbox_inches='tight'); plt.close()

# ====== PLOT 3: Excess return & Hit rate ======
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, sn, title in zip(axes, ['B', 'C'], ['Strategy B 超额与胜率', 'Strategy C 超额与胜率']):
    d = results[sn]; ns=[r['n'] for r in d]
    ax.plot(ns,[r['ae']*100 for r in d],'o-',color='darkred',linewidth=2,markersize=8,label='年化超额(%)')
    ax.set_xlabel('持股数量',fontsize=11); ax.set_ylabel('年化超额 (%)',fontsize=11,color='darkred')
    ax.tick_params(axis='y',labelcolor='darkred')
    ax2=ax.twinx()
    ax2.plot(ns,[r['hit']*100 for r in d],'s--',color='purple',linewidth=2,markersize=8,label='月胜率(%)')
    ax2.set_ylabel('月度胜率 (%)',fontsize=11,color='purple')
    ax2.tick_params(axis='y',labelcolor='purple')
    l1,lbl1=ax.get_legend_handles_labels(); l2,lbl2=ax2.get_legend_handles_labels()
    ax.legend(l1+l2,lbl1+lbl2,loc='center right'); ax.set_title(title,fontsize=13,fontweight='bold')
    ax.grid(True,alpha=0.3)
plt.tight_layout(); plt.savefig('pict/v4_plot_te_excess_hit.png',dpi=150,bbox_inches='tight'); plt.close()

# ====== PLOT 4: TE pie chart for n=20 vs n=50 ======
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for row, sn in enumerate(['B', 'C']):
    for col, s in enumerate([20, 50]):
        ax = axes[row, col]
        d = [r for r in results[sn] if r['n']==s][0]
        sizes_pie = [d['sys_pct'], 100-d['sys_pct']]
        colors_pie = ['orange', 'steelblue']
        labels_pie = [f'系统性 ({d["sys_pct"]:.0f}%)', f'特质性 ({100-d["sys_pct"]:.0f}%)']
        if d['sys_pct'] < 1:  # Too small, combine
            sizes_pie = [100]; colors_pie = ['steelblue']; labels_pie = [f'特质性 (≈100%)']
        wedges, texts, _ = ax.pie(sizes_pie, labels=labels_pie, colors=colors_pie,
                                   autopct='', startangle=90)
        ax.set_title(f'{sn}策略 n={s}: TE={d["te"]*100:.1f}%  Beta={d["beta"]:.3f}',fontsize=11,fontweight='bold')
plt.tight_layout(); plt.savefig('pict/v4_plot_te_pie.png',dpi=150,bbox_inches='tight'); plt.close()

print("\nAll 4 plots saved:")
print("  pict/v4_plot_te_ir_vs_n.png")
print("  pict/v4_plot_te_decomposition.png")
print("  pict/v4_plot_te_excess_hit.png")
print("  pict/v4_plot_te_pie.png")
print("Done!")
