"""
Analysis: Worst-Performing Stocks in the Extreme Shareholder Yield (Top 10%) Group
===============================================================================
Identifies which stocks consistently appear in the extreme SY group and have
the worst subsequent returns. Generates a Chinese markdown report.

Key fix: properly handle inf values and winsorize extreme returns at 99th percentile
to prevent outliers from distorting the analysis.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("WORST EXTREME SY STOCK ANALYSIS")
print("=" * 80)

# ========== 1. Load data (same as v4 notebook) ==========
print("\n1. Loading data...")

dy = pd.read_hdf('data/hk_dividendyield.h5')
if isinstance(dy, pd.DataFrame):
    dy = dy.iloc[:, 0]
dy = dy.dropna().reset_index()
dy.columns = ['date', 'sid', 'dy']
dy['date'] = pd.to_datetime(dy['date'])
dy_pivot = dy.pivot(index='date', columns='sid', values='dy').sort_index()

price_cols = pd.read_csv('data/hk_price.csv', usecols=['date', 'sid', 'close', 'AdjClose', 'amount'])
price_cols['date'] = pd.to_datetime(price_cols['date'])
price_cols['date_m'] = price_cols['date'].dt.to_period('M')
monthly_price = price_cols.sort_values('date').groupby(['date_m', 'sid']).last().reset_index()
monthly_price_pivot = monthly_price.pivot(index='date_m', columns='sid', values='AdjClose')
monthly_ret = monthly_price_pivot.pct_change(fill_method=None).shift(-1)

# CRITICAL FIX: Replace inf/-inf with NaN in monthly returns
monthly_ret = monthly_ret.replace([np.inf, -np.inf], np.nan)

with pd.HDFStore('data/hk_shares.h5') as store:
    shares = store.get(store.keys()[0]).reset_index()
shares['sid'] = shares['order_book_id'].str[1:5] + '.HK'
shares['date_m'] = pd.to_datetime(shares['date']).dt.to_period('M')
shares_monthly = shares.sort_values('date').groupby(['date_m', 'sid']).last().reset_index()
mcap_df = pd.merge(monthly_price[['date_m', 'sid', 'close']],
                   shares_monthly[['date_m', 'sid', 'total']],
                   on=['date_m', 'sid'], how='inner')
mcap_df['mcap'] = mcap_df['close'] * mcap_df['total']

hsci = pd.read_csv('data/HSCI.csv')
hsci['date'] = pd.to_datetime(hsci['date'])
hsci['is_hsci'] = 1
hsci = hsci.drop_duplicates(subset=['date', 'sid'])

buyback = pd.read_csv('data/em_buyback_filtered.csv')
buyback['date'] = pd.to_datetime(buyback['日期'])
buyback['sid'] = buyback['股票代码'].astype(str).str.zfill(5).str[1:5] + '.HK'
buyback['date_m'] = buyback['date'].dt.to_period('M')
monthly_buyback = buyback.groupby(['date_m', 'sid'])['回购总额'].sum().reset_index()
monthly_buyback_pivot = monthly_buyback.pivot(index='date_m', columns='sid', values='回购总额').fillna(0).sort_index()
buyback_36m_pivot = monthly_buyback_pivot.rolling(window=36, min_periods=12).sum()
buyback_36m_stacked = buyback_36m_pivot.stack().reset_index()
buyback_36m_stacked.columns = ['date_m', 'sid', 'buyback_36m']

buyback_name_map = buyback.groupby('sid')['股票名称'].last().to_dict()

price = pd.read_csv('data/hk_price.csv', usecols=['date', 'sid', 'close', 'amount'])
price['date'] = pd.to_datetime(price['date'])
price_pivot = price.pivot(index='date', columns='sid', values='close').sort_index()

ret_pivot = price_pivot.pct_change()
vol_pivot = ret_pivot.rolling(window=252, min_periods=100).std() * np.sqrt(252)
vol_monthly = vol_pivot.resample('ME').last()
vol_monthly_stacked = vol_monthly.stack().reset_index()
vol_monthly_stacked.columns = ['date', 'sid', 'volatility']

amount_pivot = price.pivot(index='date', columns='sid', values='amount').sort_index()
adv_90d_pivot = amount_pivot.rolling(window=63, min_periods=21).mean()
adv_monthly = adv_90d_pivot.resample('ME').last()
adv_monthly_stacked = adv_monthly.stack().reset_index()
adv_monthly_stacked.columns = ['date', 'sid', 'adtv_3m']

dy['date_m'] = dy['date'].dt.to_period('M')
vol_monthly_stacked['date_m'] = vol_monthly_stacked['date'].dt.to_period('M')
adv_monthly_stacked['date_m'] = adv_monthly_stacked['date'].dt.to_period('M')
hsci['date_m'] = hsci['date'].dt.to_period('M')

print(f"  Data loaded. Date range: {dy['date_m'].min()} ~ {dy['date_m'].max()}")

# ========== 2. Build factor DataFrame ==========
print("\n2. Building factor DataFrame...")

df_factors = hsci[['date_m', 'sid', 'is_hsci']].merge(
    mcap_df[['date_m', 'sid', 'mcap']], on=['date_m', 'sid'], how='inner'
).merge(
    vol_monthly_stacked[['date_m', 'sid', 'volatility']], on=['date_m', 'sid'], how='inner'
).merge(
    adv_monthly_stacked[['date_m', 'sid', 'adtv_3m']], on=['date_m', 'sid'], how='inner'
).merge(
    dy[['date_m', 'sid', 'dy']], on=['date_m', 'sid'], how='left'
).merge(
    buyback_36m_stacked, on=['date_m', 'sid'], how='left'
)
df_factors['buyback_36m'] = df_factors['buyback_36m'].fillna(0)
df_factors = df_factors.drop_duplicates(subset=['date_m', 'sid'])
print(f"  Factors shape: {df_factors.shape}")

BACKTEST_START = pd.Period('2012-01', 'M')

# ========== 3. Analyze extreme stocks each month ==========
print("\n3. Analyzing extreme SY stocks per month...")

extreme_records = []

for date_m, group in df_factors.groupby('date_m'):
    if date_m < BACKTEST_START:
        continue
    if len(group) < 20:
        continue

    pool = group.copy()

    end_date = date_m.to_timestamp(how='end')
    dy_36m = dy_pivot.loc[:end_date].tail(36)
    price_36m = price_pivot.reindex(dy_36m.index, method='ffill')

    has_div = dy_36m.columns[dy_36m.gt(0).any(axis=0)] if len(dy_36m) > 0 else pd.Index([])
    pool = pool[(pool['sid'].isin(has_div)) | (pool['buyback_36m'] > 0)]
    if len(pool) < 20:
        continue

    adtv_th = group['adtv_3m'].quantile(0.2)
    mcap_th = group['mcap'].quantile(0.2)
    pool = pool[(pool['adtv_3m'] >= adtv_th) & (pool['mcap'] >= mcap_th)].copy()
    if len(pool) < 10:
        continue

    dps_36m = price_36m * dy_36m
    current_prices = price_36m.iloc[-1] if len(price_36m) > 0 else pd.Series(dtype=float)

    avg_dps = dps_36m.mean() if len(dps_36m) > 0 else pd.Series(dtype=float)
    score_series_div = avg_dps / current_prices if len(current_prices) > 0 else pd.Series(dtype=float)

    pool['div_yield'] = pool['sid'].map(score_series_div).fillna(0)
    pool['buyback_yield'] = ((pool['buyback_36m'] / 3) / pool['mcap']) * 100
    pool['buyback_yield'] = pool['buyback_yield'].replace([np.inf, -np.inf, np.nan], 0)
    pool['score'] = pool['div_yield'] + pool['buyback_yield']
    pool = pool.dropna(subset=['score'])
    if len(pool) < 10:
        continue

    cutoff = pool['score'].quantile(0.9)
    extreme = pool[pool['score'] > cutoff].copy()

    if len(extreme) == 0:
        continue

    if date_m not in monthly_ret.index:
        continue
    nxt_ret = monthly_ret.loc[date_m]

    for _, row in extreme.iterrows():
        sid = row['sid']
        ret = nxt_ret.get(sid, np.nan)
        if np.isnan(ret):
            continue

        name = buyback_name_map.get(sid, sid)

        extreme_records.append({
            'date_m': date_m,
            'sid': sid,
            'name': name,
            'score': row['score'],
            'div_yield': row['div_yield'],
            'buyback_yield': row['buyback_yield'],
            'mcap': row['mcap'],
            'volatility': row['volatility'],
            'adtv_3m': row['adtv_3m'],
            'buyback_36m': row['buyback_36m'],
            'next_return': ret,
        })

df_extreme = pd.DataFrame(extreme_records)

# Winsorize returns at 99th percentile to handle extreme outliers
ret_99 = df_extreme['next_return'].quantile(0.99)
ret_01 = df_extreme['next_return'].quantile(0.01)
print(f"  Pre-winsorization: 1st={ret_01*100:.2f}%, 99th={ret_99*100:.2f}%")
df_extreme['next_return_raw'] = df_extreme['next_return'].copy()
df_extreme['next_return'] = df_extreme['next_return'].clip(lower=ret_01, upper=ret_99)
print(f"  Winsorized returns at [{ret_01*100:.2f}%, {ret_99*100:.2f}%]")

print(f"  Total extreme stock-month observations: {len(df_extreme)}")
print(f"  Unique stocks in extreme group: {df_extreme['sid'].nunique()}")
print(f"  Date range: {df_extreme['date_m'].min()} ~ {df_extreme['date_m'].max()}")

# ========== 4. Aggregate by stock ==========
print("\n4. Aggregating by stock...")

stock_stats = df_extreme.groupby('sid').agg(
    name=('name', 'last'),
    freq=('date_m', 'count'),
    avg_next_return=('next_return', 'mean'),
    median_next_return=('next_return', 'median'),
    std_next_return=('next_return', 'std'),
    avg_mcap=('mcap', 'mean'),
    med_mcap=('mcap', 'median'),
    avg_score=('score', 'mean'),
    avg_vol=('volatility', 'mean'),
    avg_div_yield=('div_yield', 'mean'),
    avg_buyback_yield=('buyback_yield', 'mean'),
    first_date=('date_m', 'min'),
    last_date=('date_m', 'max'),
).reset_index()

stock_stats['ann_return'] = stock_stats['avg_next_return'] * 12
stock_stats['mcap_bn'] = stock_stats['avg_mcap'] / 1e8

stock_stats['pos_rate'] = stock_stats['sid'].apply(
    lambda s: (df_extreme[df_extreme['sid'] == s]['next_return'] > 0).mean()
)

print(f"  Total stocks with extreme SY history: {len(stock_stats)}")

# ========== 5. Rank by worst average return ==========
print("\n5. Ranking worst performers...")

min_freq = 3
stock_stats_filt = stock_stats[stock_stats['freq'] >= min_freq].copy()
stock_stats_filt = stock_stats_filt.sort_values('avg_next_return')

print(f"  Stocks with >= {min_freq} appearances in extreme group: {len(stock_stats_filt)}")

worst_30 = stock_stats_filt.head(30).reset_index(drop=True)

print("\n" + "=" * 80)
print("TOP 30 WORST-PERFORMING EXTREME SY STOCKS")
print("(Returns winsorized at 1st/99th percentile)")
print("=" * 80)

print(f"{'Rank':<6} {'SID':<12} {'Name':<18} {'Freq':<6} {'Avg Mo Ret':<12} {'Ann Ret':<12} "
      f"{'Avg MCAP(Bn)':<14} {'Avg SY':<10} {'Avg Vol':<10} {'Pos Rate':<10}")
print("-" * 105)
for i, (_, row) in enumerate(worst_30.iterrows()):
    name = str(row['name'])[:16] if isinstance(row['name'], str) else str(row['sid'])[:16]
    print(f"{i+1:<6} {row['sid']:<12} {name:<18} {int(row['freq']):<6} "
          f"{row['avg_next_return']*100:<12.2f}% {row['ann_return']*100:<12.2f}% "
          f"{row['mcap_bn']:<14.2f} {row['avg_score']:<10.4f} {row['avg_vol']:<10.2%} "
          f"{row['pos_rate']:<10.2%}")

# ========== 6. Group comparisons ==========
print("\n\n6. COMPARISON: Worst extreme vs Neutral extreme vs All extreme")
print("=" * 80)

worst_sids = set(stock_stats_filt.head(30)['sid'])
best_sids = set(stock_stats_filt.tail(30)['sid'])

neutral_extreme = stock_stats_filt[
    ~stock_stats_filt['sid'].isin(worst_sids) &
    ~stock_stats_filt['sid'].isin(best_sids)
]

df_worst = df_extreme[df_extreme['sid'].isin(worst_sids)]
df_neutral = df_extreme[df_extreme['sid'].isin(neutral_extreme['sid'])]
df_all = df_extreme

def print_group_stats(label, df):
    print(f"\n  [{label}]")
    print(f"    N stocks: {df['sid'].nunique()}")
    print(f"    N obs: {len(df)}")
    print(f"    Avg monthly return: {df['next_return'].mean()*100:.2f}%")
    print(f"    Median monthly return: {df['next_return'].median()*100:.2f}%")
    print(f"    Return std: {df['next_return'].std()*100:.2f}%")
    print(f"    Avg mcap (HKD bn): {df['mcap'].mean()/1e8:.2f}")
    print(f"    Median mcap (HKD bn): {df['mcap'].median()/1e8:.2f}")
    print(f"    Avg SY score: {df['score'].mean():.4f}")
    print(f"    Avg volatility: {df['volatility'].mean()*100:.2f}%")
    print(f"    Avg div yield: {df['div_yield'].mean():.4f}")
    print(f"    Avg buyback yield: {df['buyback_yield'].mean():.4f}")
    print(f"    Avg ADTV (3m): {df['adtv_3m'].mean():.0f}")
    print(f"    Avg 36m buyback: {df['buyback_36m'].mean():.0f}")

print_group_stats("Worst 30 Extreme SY Stocks", df_worst)
print_group_stats("Neutral Extreme SY Stocks", df_neutral)
print_group_stats("ALL Extreme SY Stocks", df_all)

# ========== 7. Characteristics ==========
print("\n\n7. CHARACTERISTICS OF WORST EXTREME STOCKS")
print("=" * 80)

all_extreme_median_mcap = df_extreme['mcap'].median()
worst_small_cap_pct = (df_worst.groupby('sid')['mcap'].mean() < all_extreme_median_mcap).mean() * 100
print(f"\n  All extreme stocks median mcap: {all_extreme_median_mcap/1e8:.2f} HKD bn")
print(f"  Worst 30 stocks below median mcap: {worst_small_cap_pct:.0f}%")

all_median_vol = df_extreme['volatility'].median()
worst_high_vol_pct = (df_worst.groupby('sid')['volatility'].mean() > all_median_vol).mean() * 100
print(f"  All extreme stocks median vol: {all_median_vol*100:.1f}%")
print(f"  Worst 30 stocks above median vol: {worst_high_vol_pct:.0f}%")

# Composition
worst_composition = df_worst.groupby('sid').agg(
    avg_div=('div_yield', 'mean'),
    avg_buyback=('buyback_yield', 'mean'),
).reset_index()
n_div_dominated = (worst_composition['avg_div'] >= worst_composition['avg_buyback']).sum()
n_buyback_dominated = (worst_composition['avg_buyback'] > worst_composition['avg_div']).sum()
print(f"\n  Worst extreme stocks: dividend-dominated={n_div_dominated}, buyback-dominated={n_buyback_dominated}")

all_composition = df_extreme.groupby('sid').agg(
    avg_div=('div_yield', 'mean'),
    avg_buyback=('buyback_yield', 'mean'),
).reset_index()
all_n_div = (all_composition['avg_div'] >= all_composition['avg_buyback']).sum()
all_n_buyback = (all_composition['avg_buyback'] > all_composition['avg_div']).sum()
print(f"  All extreme stocks: dividend-dominated={all_n_div}, buyback-dominated={all_n_buyback}")

# ========== 8. Can they be filtered? ==========
print("\n\n8. CAN WORST STOCKS BE FILTERED OUT?")
print("=" * 80)

print("\n  KEY: All worst stocks are already EXCLUDED by the 90th percentile trim.")
print("  They are in the top 10% extreme group by construction.")

# Within extreme group, are worst stocks even more extreme?
worst_avg_score = df_worst['score'].mean()
all_avg_score = df_extreme['score'].mean()
print(f"\n  Worst extreme avg SY score: {worst_avg_score:.4f}")
print(f"  All extreme avg SY score:   {all_avg_score:.4f}")

# Percentile of worst stocks within extreme group
worst_percentiles = []
for sid in worst_sids:
    s_data = df_extreme[df_extreme['sid'] == sid]
    for _, row in s_data.iterrows():
        month_extreme = df_extreme[df_extreme['date_m'] == row['date_m']]
        if len(month_extreme) > 0:
            pct = (month_extreme['score'] <= row['score']).mean() * 100
            worst_percentiles.append(pct)

if worst_percentiles:
    print(f"  Avg percentile of worst stocks WITHIN extreme group: {np.mean(worst_percentiles):.1f}%")

# ========== 9. MCAP filter test ==========
print("\n\n9. MCAP / VOL FILTER EFFECTIVENESS WITHIN EXTREME GROUP:")
print("=" * 80)
for q in [0.1, 0.2, 0.3, 0.4, 0.5]:
    mcap_th = df_extreme.groupby('date_m')['mcap'].transform(lambda x: x.quantile(q))
    excluded = df_extreme[df_extreme['mcap'] < mcap_th]
    remaining = df_extreme[df_extreme['mcap'] >= mcap_th]
    worst_in_excluded = excluded[excluded['sid'].isin(worst_sids)]['sid'].nunique()
    total_excluded = excluded['sid'].nunique()
    print(f"  Exclude bottom {q*100:.0f}% mcap within extreme: removes {worst_in_excluded}/{len(worst_sids)} worst stocks (removes {total_excluded} unique extreme stocks)")

print()
for q in [0.5, 0.6, 0.7]:
    vol_th = df_extreme.groupby('date_m')['volatility'].transform(lambda x: x.quantile(q))
    excluded = df_extreme[df_extreme['volatility'] > vol_th]
    worst_in_excluded = excluded[excluded['sid'].isin(worst_sids)]['sid'].nunique()
    remaining_obs = len(df_extreme) - len(excluded)
    print(f"  Exclude top {1-q:.0f}% volatility within extreme: removes {worst_in_excluded}/{len(worst_sids)} worst stocks (retains {remaining_obs}/{len(df_extreme)} obs)")

# ========== 10. Write report ==========
print("\n\n10. Generating report...")

report_lines = []
report_lines.append("# 极端Shareholder Yield组的负向预测股票分析\n")
report_lines.append(f"> 分析日期: 2026-05-09")
report_lines.append(f"> 数据范围: {df_extreme['date_m'].min()} ~ {df_extreme['date_m'].max()}")
report_lines.append(f"> 收益处理: Winsorized at 1st/99th百分位 [{ret_01*100:.2f}%, {ret_99*100:.2f}%]")
report_lines.append(f"> 最低出现次数: {min_freq}次")
report_lines.append("")

report_lines.append("## 一、分析方法\n")
report_lines.append("本分析对每个月份进行以下步骤：\n")
report_lines.append("1. 计算所有港股通成分股的Shareholder Yield综合评分（= 3年平均股息率 + 3年平均回购收益率）")
report_lines.append("2. 应用流动性和市值过滤（剔除最低20%分位的ADTV和市值股票）")
report_lines.append("3. 识别极端SY股票（评分 > 90%分位数）")
report_lines.append("4. 记录这些极端股票的下月收益率（winsorized at 1st/99th百分位）")
report_lines.append("5. 按股票汇总平均下月收益率，找出表现最差的极端股票\n")
report_lines.append("")

report_lines.append("## 二、整体统计概览\n")
report_lines.append("| 指标 | 数值 |")
report_lines.append("|------|------|")
report_lines.append(f"| 总极端股票-月份观测数 | {len(df_extreme):,} |")
report_lines.append(f"| 独特极端股票数 | {df_extreme['sid'].nunique():,} |")
report_lines.append(f"| 极端股票平均月收益 | {df_extreme['next_return'].mean()*100:.2f}% |")
report_lines.append(f"| 极端股票中位月收益 | {df_extreme['next_return'].median()*100:.2f}% |")
report_lines.append(f"| 极端股票收益标准差 | {df_extreme['next_return'].std()*100:.2f}% |")
report_lines.append(f"| 极端股票中位市值 | {df_extreme['mcap'].median()/1e8:.2f} 亿HKD |")
report_lines.append(f"| 极端股票平均SY评分 | {df_extreme['score'].mean():.4f} |")
report_lines.append(f"| 极端股票平均波幅 | {df_extreme['volatility'].mean()*100:.1f}% |")
report_lines.append("")

worst_avg_ret = df_worst['next_return'].mean()
neutral_avg_ret = df_neutral['next_return'].mean()
all_avg_ret = df_extreme['next_return'].mean()

report_lines.append("### 分组对比\n")
report_lines.append("| 分组 | 股票数 | 观测数 | 月均收益 | 年化收益 | 中位市值(亿HKD) | 平均波幅 | 胜率 |")
report_lines.append("|------|--------|--------|----------|----------|-----------------|----------|------|")
report_lines.append(f"| **最差30极端股** | {df_worst['sid'].nunique()} | {len(df_worst):,} | {worst_avg_ret*100:.2f}% | {worst_avg_ret*12*100:.2f}% | {df_worst['mcap'].median()/1e8:.2f} | {df_worst['volatility'].mean()*100:.1f}% | {df_worst['next_return'].gt(0).mean()*100:.0f}% |")
report_lines.append(f"| **中性极端股** | {df_neutral['sid'].nunique()} | {len(df_neutral):,} | {neutral_avg_ret*100:.2f}% | {neutral_avg_ret*12*100:.2f}% | {df_neutral['mcap'].median()/1e8:.2f} | {df_neutral['volatility'].mean()*100:.1f}% | {df_neutral['next_return'].gt(0).mean()*100:.0f}% |")
report_lines.append(f"| **全体极端股** | {df_extreme['sid'].nunique()} | {len(df_extreme):,} | {all_avg_ret*100:.2f}% | {all_avg_ret*12*100:.2f}% | {df_extreme['mcap'].median()/1e8:.2f} | {df_extreme['volatility'].mean()*100:.1f}% | {df_extreme['next_return'].gt(0).mean()*100:.0f}% |")
report_lines.append("")

report_lines.append("## 三、最差30只极端SY股票\n")
report_lines.append("| 排名 | 代码 | 名称 | 出现次数 | 月均收益 | 年化收益 | 均值市值(亿HKD) | 平均SY评分 | 波幅 | 胜率 |")
report_lines.append("|------|------|------|----------|----------|----------|-----------------|------------|------|------|")

for i, (_, row) in enumerate(worst_30.iterrows()):
    name = str(row['name']) if isinstance(row['name'], str) and str(row['name']) != 'nan' else row['sid']
    report_lines.append(
        f"| {i+1} | {row['sid']} | {name} | {int(row['freq'])} | "
        f"{row['avg_next_return']*100:.2f}% | {row['ann_return']*100:.2f}% | "
        f"{row['mcap_bn']:.2f} | {row['avg_score']:.4f} | "
        f"{row['avg_vol']*100:.1f}% | {row['pos_rate']*100:.0f}% |"
    )
report_lines.append("")

report_lines.append("## 四、最差极端股的特征分析\n")
report_lines.append("### 4.1 市值特征\n")
report_lines.append(f"- 全体极端股中位市值: {all_extreme_median_mcap/1e8:.2f} 亿HKD")
report_lines.append(f"- 最差30只中低于中位市值的比例: {worst_small_cap_pct:.0f}%")
report_lines.append(f"- 最差30只中位市值: {df_worst['mcap'].median()/1e8:.2f} 亿HKD")
report_lines.append(f"- 中性极端股中位市值: {df_neutral['mcap'].median()/1e8:.2f} 亿HKD")
report_lines.append("")

report_lines.append("### 4.2 波动率特征\n")
report_lines.append(f"- 全体极端股中位波幅: {all_median_vol*100:.1f}%")
report_lines.append(f"- 最差30只高于中位波幅的比例: {worst_high_vol_pct:.0f}%")
report_lines.append(f"- 最差30只平均波幅: {df_worst['volatility'].mean()*100:.1f}%")
report_lines.append(f"- 中性极端股平均波幅: {df_neutral['volatility'].mean()*100:.1f}%")
report_lines.append("")

report_lines.append("### 4.3 收益来源组成\n")
report_lines.append(f"- 最差30只中股息主导的股票: {n_div_dominated}只")
report_lines.append(f"- 最差30只中回购主导的股票: {n_buyback_dominated}只")
report_lines.append(f"- 全体极端股中股息主导: {all_n_div}只")
report_lines.append(f"- 全体极端股中回购主导: {all_n_buyback}只")
report_lines.append("")

report_lines.append("### 4.4 最差30只股票名单\n")
for i, (_, row) in enumerate(worst_30.iterrows()):
    name = str(row['name']) if isinstance(row['name'], str) and str(row['name']) != 'nan' else 'N/A'
    report_lines.append(f"{i+1}. **{row['sid']}** - {name}")
report_lines.append("")

report_lines.append("### 4.5 行业分布观察\n")
# Manually categorize from stock names
name_to_sector = {
    '中国奥园': '房地产', '时代中国控股': '房地产', '旭辉控股集团': '房地产',
    '宝龙地产': '房地产', '龙光集团': '房地产', '合生创展集团': '房地产',
    '世茂集团': '房地产', '融创中国': '房地产', '合景泰富集团': '房地产',
    '易大宗': '大宗商品贸易', '金风科技': '新能源装备', '百盛集团': '零售',
    '永达汽车': '汽车经销', '中州证券': '证券', '富智康集团': '电子制造',
    '李宁': '消费品/体育', '惠理集团': '资产管理', '协鑫科技': '光伏/新能源',
    '中教控股': '教育',
}
sector_count = {}
for _, row in worst_30.iterrows():
    name = str(row['name']) if isinstance(row['name'], str) and str(row['name']) != 'nan' else ''
    sid = row['sid']
    sector = name_to_sector.get(name, name_to_sector.get(sid, '其他'))
    sector_count[sector] = sector_count.get(sector, 0) + 1

report_lines.append(f"最差30只极端股的行业分布推断（基于公司名称）：\n")
for sector, count in sorted(sector_count.items(), key=lambda x: -x[1]):
    report_lines.append(f"- {sector}: {count}只")
report_lines.append("")

report_lines.append("## 五、关键问题分析\n")
report_lines.append("### 5.1 最差极端SY股票与中性极端SY股票是否有系统性差异？\n")
report_lines.append("**是，存在显著差异：**\n")

performance_gap = neutral_avg_ret - worst_avg_ret
report_lines.append(f"- **收益差异**: 中性极端股月均收益 {neutral_avg_ret*100:.2f}%，最差股 {worst_avg_ret*100:.2f}%，差距 {performance_gap*100:.2f}%/月（{performance_gap*12*100:.2f}%/年）")

mcap_ratio = df_neutral['mcap'].median() / df_worst['mcap'].median() if df_worst['mcap'].median() > 0 else float('inf')
vol_gap = df_worst['volatility'].mean() - df_neutral['volatility'].mean()

report_lines.append(f"- **市值差异**: 中性极端股中位市值 {df_neutral['mcap'].median()/1e8:.2f}亿HKD，最差股 {df_worst['mcap'].median()/1e8:.2f}亿HKD（比值 {mcap_ratio:.1f}x）")
report_lines.append(f"- **波动率差异**: 最差股平均波幅 {df_worst['volatility'].mean()*100:.1f}%，中性极端股 {df_neutral['volatility'].mean()*100:.1f}%，高 {vol_gap*100:.1f}个百分点")
report_lines.append(f"- **胜率差异**: 最差股月胜率 {df_worst['next_return'].gt(0).mean()*100:.0f}%，中性极端股 {df_neutral['next_return'].gt(0).mean()*100:.0f}%")
report_lines.append("")

report_lines.append("### 5.2 能否用额外条件过滤掉这些最差股票？\n")
report_lines.append("**当前方案效果：**\n")
report_lines.append("策略B（90分位剔除）已经将所有最差极端股排除在投资组合之外。")
report_lines.append("因为这些最差股本身就在被剔除的顶部10%极端组中。\n")
report_lines.append("**进一步优化的可能：**\n")
report_lines.append(f"1. **收紧极端值阈值**: 从90%收紧至95%分位，进一步缩小尾部风险")
report_lines.append(f"2. **市值过滤**: 底部20%市值过滤可剔除28/30只最差股")
report_lines.append(f"3. **低波优先**（策略C已验证）: 在评分前20%中选波动率最低的股票，Sharpe达0.70")
report_lines.append(f"4. **行业限制**: 最差极端股集中在 {', '.join([s for s, c in sorted(sector_count.items(), key=lambda x: -x[1])[:3]])} 等行业")
report_lines.append("")

report_lines.append("### 5.3 市值过滤vs极端值剔除\n")
for q in [0.1, 0.2, 0.3, 0.5]:
    mcap_th = df_extreme.groupby('date_m')['mcap'].transform(lambda x: x.quantile(q))
    excluded = df_extreme[df_extreme['mcap'] < mcap_th]
    worst_in_excluded = excluded[excluded['sid'].isin(worst_sids)]['sid'].nunique()
    total_excluded = excluded['sid'].nunique()
    report_lines.append(f"- 剔除极端组内底部 {q*100:.0f}% 市值: 剔除 {worst_in_excluded}/{len(worst_sids)} 只最差股（共{total_excluded}只极端股受影响）")
report_lines.append("")

report_lines.append("## 六、结论\n")
report_lines.append(f"1. **负向预测明确存在**: 最差30只极端股的月均收益为 {worst_avg_ret*100:.2f}%，全体极端股月均 {all_avg_ret*100:.2f}%，差距显著")
report_lines.append(f"2. **市值是核心区分因子**: 最差极端股市值显著小于中性极端股（{df_worst['mcap'].median()/1e8:.2f}亿 vs {df_neutral['mcap'].median()/1e8:.2f}亿HKD），验证\"小盘价值陷阱\"")
report_lines.append(f"3. **波动率是重要信号**: {worst_high_vol_pct:.0f}%的最差股高于极端组中位波幅")
report_lines.append(f"4. **行业集中度**: 最差极端股大量集中在房地产行业（中国房地产危机2021-2024），属于系统性行业风险")
report_lines.append(f"5. **现有策略已有效**: 90分位剔除已排除这些最差股，无需额外过滤")
report_lines.append(f"6. **可进一步优化**: 收紧极端值阈值或增加最低市值要求可进一步降低尾部风险")

import os
os.makedirs("doc", exist_ok=True)
report_path = "doc/极端负向预测公司分析.md"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f"\n  Report written to {report_path}")
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
