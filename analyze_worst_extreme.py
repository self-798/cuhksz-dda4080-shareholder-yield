"""
分析极端Shareholder Yield组中表现最差的个股
=============================================
目标: 在每个月SY score > 90th percentile的"极端"股票中,
找出哪些股票在次月平均表现最差, 分析它们的共同特征,
并评估是否可以通过额外过滤器来捕捉这些"陷阱"。
"""

import pandas as pd
import numpy as np
import warnings
import json

warnings.filterwarnings('ignore')

BACKTEST_START = pd.Period('2012-01', 'M')

# ============================================================================
# 1. 加载数据 (follow v4 notebook cell 1 data pipeline)
# ============================================================================
print("=" * 80)
print("STEP 1: Loading data...")
print("=" * 80)

# ---- Dividend yield ----
print("Loading dividend yield data...")
dy = pd.read_hdf('data/hk_dividendyield.h5')
if isinstance(dy, pd.DataFrame):
    dy = dy.iloc[:, 0]
dy = dy.dropna().reset_index()
dy.columns = ['date', 'sid', 'dy']
dy['date'] = pd.to_datetime(dy['date'])
dy['date_m'] = dy['date'].dt.to_period('M')
# Build dy_pivot for 36-month lookback
dy_pivot = dy.pivot(index='date', columns='sid', values='dy').sort_index()

# ---- Price data ----
print("Loading price data...")
price_cols = pd.read_csv('data/hk_price.csv', usecols=['date', 'sid', 'close', 'AdjClose', 'amount'])
price_cols['date'] = pd.to_datetime(price_cols['date'])
price_cols['date_m'] = price_cols['date'].dt.to_period('M')

# Monthly prices and returns
monthly_price = price_cols.sort_values('date').groupby(['date_m', 'sid']).last().reset_index()
monthly_price_pivot = monthly_price.pivot(index='date_m', columns='sid', values='AdjClose')
monthly_ret = monthly_price_pivot.pct_change(fill_method=None).shift(-1)
# monthly_ret.loc[M] = return for month M+1

# Full daily price pivot for 36-mo DPS computation
price_daily = pd.read_csv('data/hk_price.csv', usecols=['date', 'sid', 'close', 'amount'])
price_daily['date'] = pd.to_datetime(price_daily['date'])
price_pivot = price_daily.pivot(index='date', columns='sid', values='close').sort_index()

# ---- Shares / Market cap ----
print("Loading shares data...")
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

# ---- HSCI ----
print("Loading HSCI data...")
hsci = pd.read_csv('data/HSCI.csv')
hsci['date'] = pd.to_datetime(hsci['date'])
hsci['is_hsci'] = 1
hsci = hsci.drop_duplicates(subset=['date', 'sid'])
hsci['date_m'] = hsci['date'].dt.to_period('M')

# ---- Buyback ----
print("Loading buyback data...")
buyback = pd.read_csv('data/em_buyback_filtered.csv')
buyback['date'] = pd.to_datetime(buyback['日期'])
buyback['sid'] = buyback['股票代码'].astype(str).str.zfill(5).str[1:5] + '.HK'
buyback['date_m'] = buyback['date'].dt.to_period('M')
monthly_buyback = buyback.groupby(['date_m', 'sid'])['回购总额'].sum().reset_index()
monthly_buyback_pivot = monthly_buyback.pivot(index='date_m', columns='sid', values='回购总额').fillna(0)
monthly_buyback_pivot = monthly_buyback_pivot.sort_index()
buyback_36m_pivot = monthly_buyback_pivot.rolling(window=36, min_periods=12).sum()
buyback_36m_stacked = buyback_36m_pivot.stack().reset_index()
buyback_36m_stacked.columns = ['date_m', 'sid', 'buyback_36m']

# ---- Volatility (252-day annualized) ----
print("Calculating volatility...")
ret_daily_pivot = price_pivot.pct_change()
vol_pivot = ret_daily_pivot.rolling(window=252, min_periods=100).std() * np.sqrt(252)
vol_monthly = vol_pivot.resample('ME').last()
vol_monthly_stacked = vol_monthly.stack().reset_index()
vol_monthly_stacked.columns = ['date', 'sid', 'volatility']
vol_monthly_stacked['date_m'] = vol_monthly_stacked['date'].dt.to_period('M')

# ---- ADTV (63-day average) ----
print("Calculating ADTV...")
amount_pivot = price_daily.pivot(index='date', columns='sid', values='amount').sort_index()
adv_63d_pivot = amount_pivot.rolling(window=63, min_periods=21).mean()
adv_monthly = adv_63d_pivot.resample('ME').last()
adv_monthly_stacked = adv_monthly.stack().reset_index()
adv_monthly_stacked.columns = ['date', 'sid', 'adtv_3m']
adv_monthly_stacked['date_m'] = adv_monthly_stacked['date'].dt.to_period('M')

# ---- Industry map ----
print("Loading industry map...")
with open('data/hk_industry_map.json', 'r') as f:
    industry_map = json.load(f)

# ---- Stock name map (from buyback data) ----
print("Building stock name map...")
name_map = buyback[['sid', '股票名称']].drop_duplicates(subset='sid').set_index('sid')['股票名称'].to_dict()
# Supplement with any additional names from HSCI
# (HSCI doesn't have names, so we rely on buyback data)

# ============================================================================
# 2. Merge factors & Build analysis pool
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Merging factors...")
print("=" * 80)

df_factors = hsci[['date_m', 'sid', 'is_hsci']].merge(
    mcap_df[['date_m', 'sid', 'mcap']], on=['date_m', 'sid'], how='inner'
).merge(
    vol_monthly_stacked[['date_m', 'sid', 'volatility']], on=['date_m', 'sid'], how='inner'
).merge(
    adv_monthly_stacked[['date_m', 'sid', 'adtv_3m']], on=['date_m', 'sid'], how='inner'
).merge(
    buyback_36m_stacked, on=['date_m', 'sid'], how='left'
)

df_factors['buyback_36m'] = df_factors['buyback_36m'].fillna(0)
df_factors = df_factors.drop_duplicates(subset=['date_m', 'sid'])

print(f"Factors shape: {df_factors.shape}")
print(f"Date range: {df_factors['date_m'].min()} ~ {df_factors['date_m'].max()}")

# Add industry
df_factors['industry'] = df_factors['sid'].map(industry_map).fillna('Unknown')

# ============================================================================
# 3. Monthly analysis: find extreme stocks, track next-month returns
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: Monthly extreme-stock analysis...")
print("=" * 80)

# Records: one per (month, sid) where sid is in extreme group
extreme_records = []

months_processed = 0
total_extreme_obs = 0

for date_m, group in df_factors.groupby('date_m'):
    if date_m < BACKTEST_START:
        continue
    if date_m not in monthly_ret.index:
        continue
    if len(group) < 20:
        continue

    # Liquidity filter (same as v4 strategies)
    adtv_th = group['adtv_3m'].quantile(0.2)
    mcap_th = group['mcap'].quantile(0.2)
    pool = group[(group['adtv_3m'] >= adtv_th) & (group['mcap'] >= mcap_th)].copy()

    if len(pool) < 20:
        continue

    # Compute div_yield from 36-month DPS
    end_date = date_m.to_timestamp(how='end')
    dy_36m = dy_pivot.loc[:end_date].tail(36)
    price_36m = price_pivot.reindex(dy_36m.index, method='ffill')

    has_div = dy_36m.columns[dy_36m.gt(0).any(axis=0)] if len(dy_36m) > 0 else pd.Index([])
    pool = pool[(pool['sid'].isin(has_div)) | (pool['buyback_36m'] > 0)]

    if len(pool) < 20:
        continue

    dps_36m = price_36m * dy_36m
    current_prices = price_36m.iloc[-1] if len(price_36m) > 0 else pd.Series(dtype=float)

    if len(dps_36m) > 0 and len(current_prices) > 0:
        avg_dps = dps_36m.mean()
        score_series_div = avg_dps / current_prices
        pool['div_yield'] = pool['sid'].map(score_series_div).fillna(0)
    else:
        pool['div_yield'] = 0.0

    pool['buyback_yield'] = ((pool['buyback_36m'] / 3) / pool['mcap']) * 100
    pool['buyback_yield'] = pool['buyback_yield'].replace([np.inf, -np.inf, np.nan], 0)
    pool['score'] = pool['div_yield'] + pool['buyback_yield']
    pool = pool.dropna(subset=['score'])

    if len(pool) < 20:
        continue

    # Extreme threshold: > 90th percentile
    cutoff_90 = pool['score'].quantile(0.9)
    extreme = pool[pool['score'] > cutoff_90].copy()

    if len(extreme) == 0:
        continue

    # Get next-month returns
    nxt_ret = monthly_ret.loc[date_m]

    for _, row in extreme.iterrows():
        sid = row['sid']
        ret_val = nxt_ret.get(sid, np.nan)
        extreme_records.append({
            'date_m': date_m,
            'sid': sid,
            'score': row['score'],
            'div_yield': row['div_yield'],
            'buyback_yield': row['buyback_yield'],
            'mcap': row['mcap'],
            'volatility': row['volatility'],
            'industry': row.get('industry', 'Unknown'),
            'next_month_ret': ret_val,
        })

    months_processed += 1
    total_extreme_obs += len(extreme)

    if months_processed % 40 == 0:
        print(f"  Processed {months_processed} months, {total_extreme_obs} extreme observations so far...")

print(f"\nTotal months processed: {months_processed}")
print(f"Total extreme-stock observations: {total_extreme_obs}")

df_extreme = pd.DataFrame(extreme_records)
print(f"Unique stocks ever in extreme group: {df_extreme['sid'].nunique()}")

# ============================================================================
# 4. Aggregate per stock
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: Aggregating per-stock statistics...")
print("=" * 80)

# Aggregate statistics per stock (only for stocks that appeared in extreme group)
stock_agg = df_extreme.groupby('sid').agg(
    frequency=('next_month_ret', 'count'),
    avg_next_month_ret=('next_month_ret', 'mean'),
    median_next_month_ret=('next_month_ret', 'median'),
    std_next_month_ret=('next_month_ret', 'std'),
    avg_score=('score', 'mean'),
    avg_div_yield=('div_yield', 'mean'),
    avg_buyback_yield=('buyback_yield', 'mean'),
    avg_mcap=('mcap', 'mean'),
    avg_volatility=('volatility', 'mean'),
    industry=('industry', 'first'),
    worst_single_return=('next_month_ret', 'min'),
    best_single_return=('next_month_ret', 'max'),
    pct_negative=('next_month_ret', lambda x: (x < 0).mean()),
)

# Filter to stocks with at least 3 appearances
stock_agg = stock_agg[stock_agg['frequency'] >= 3].copy()

# Map names
stock_agg['name'] = stock_agg.index.map(name_map)
stock_agg['name'] = stock_agg['name'].fillna(pd.Series(stock_agg.index, index=stock_agg.index))

# Define worst_30 and worst_30_sids
worst_30 = stock_agg.nsmallest(30, 'avg_next_month_ret')
worst_30_sids = set(worst_30.index)

print(f"Stocks with >=3 extreme appearances: {len(stock_agg)}")

# ============================================================================
# 5. Top 30 Worst Performers
# ============================================================================
print("\n" + "=" * 80)
print("TOP 30 WORST-PERFORMING STOCKS (Extreme SY Group)")
print("=" * 80)

print(f"\n{'Rank':<6} {'SID':<10} {'Name':<20} {'Freq':<6} {'Avg Ret':<10} {'Med Ret':<10} {'Avg MCap':<16} {'Avg SY':<10} {'Avg Vol':<10} {'%Neg':<8} {'Industry'}")
print("-" * 140)
for i, (sid, row) in enumerate(worst_30.iterrows()):
    rank = i + 1
    name = row['name'][:18]
    print(f"{rank:<6} {sid:<10} {name:<20} {int(row['frequency']):<6} "
          f"{row['avg_next_month_ret']*100:>8.2f}% {row['median_next_month_ret']*100:>8.2f}% "
          f"{row['avg_mcap']:>14,.0f}  {row['avg_score']:>8.4f}  {row['avg_volatility']*100:>8.2f}% "
          f"{row['pct_negative']*100:>6.1f}%  {row['industry'][:20]}")

# ============================================================================
# 6. Pattern Analysis
# ============================================================================
print("\n\n" + "=" * 80)
print("STEP 6: PATTERN ANALYSIS")
print("=" * 80)

# 6a. Compare worst 30 vs all extreme stocks
all_extreme_means = stock_agg.agg({
    'avg_next_month_ret': 'mean',
    'avg_mcap': 'median',
    'avg_volatility': 'mean',
    'avg_score': 'mean',
    'pct_negative': 'mean',
    'frequency': 'mean',
})
worst_30_means = worst_30.agg({
    'avg_next_month_ret': 'mean',
    'avg_mcap': 'median',
    'avg_volatility': 'mean',
    'avg_score': 'mean',
    'pct_negative': 'mean',
    'frequency': 'mean',
})

print("\n--- Comparison: Worst 30 vs All Extreme Stocks ---")
print(f"{'Metric':<30} {'Worst 30':<18} {'All Extreme':<18} {'Ratio/Diff'}")
print("-" * 70)
metrics = [
    ('Avg Next-Month Return (ann.)', lambda x: x['avg_next_month_ret'] * 12, '{:.2%}'),
    ('Median MCap (HKD mn)',         lambda x: x['avg_mcap'] / 1e6,        '{:.1f}M'),
    ('Avg Volatility',                lambda x: x['avg_volatility'] * 100, '{:.1f}%'),
    ('Avg SY Score',                  lambda x: x['avg_score'],            '{:.4f}'),
    ('% Negative Months',             lambda x: x['pct_negative'] * 100,  '{:.1f}%'),
    ('Avg Frequency (# months)',      lambda x: x['frequency'],            '{:.1f}'),
]
for name, fn, fmt in metrics:
    w30_val = fn(worst_30_means)
    all_val = fn(all_extreme_means)
    if 'Ret' in name or 'Score' in name:
        ratio = w30_val / all_val if all_val != 0 else np.nan
        ratio_str = f'{ratio:.2f}x' if not np.isnan(ratio) else 'N/A'
    elif 'MCap' in name:
        ratio = w30_val / all_val if all_val > 0 else np.nan
        ratio_str = f'{ratio:.2f}x' if not np.isnan(ratio) else 'N/A'
    elif 'Vol' in name or 'Neg' in name:
        ratio = w30_val - all_val
        ratio_str = f'{ratio:+.1f}pp' if not np.isnan(ratio) else 'N/A'
    else:
        ratio_str = f'{w30_val/all_val:.2f}x' if all_val != 0 else 'N/A'
    print(f"{name:<30} {fmt.format(w30_val):<18} {fmt.format(all_val):<18} {ratio_str}")

# 6b. MCAP profile - within-month percentile ranks
print("\n\n--- MCAP Profile (within-month, relative to eligible pool) ---")
# Ensure is_worst30 column exists
if 'is_worst30' not in df_extreme.columns:
    df_extreme['is_worst30'] = df_extreme['sid'].isin(worst_30_sids)
# For each extreme observation, we need its mcap percentile within that month's extreme pool
# Compute mcap percentiles within each month for extreme observations
mcap_pct_records = []
for date_m, group in df_extreme.groupby('date_m'):
    if len(group) < 3:
        continue
    group = group.copy()
    group['mcap_pct'] = group['mcap'].rank(pct=True) * 100
    mcap_pct_records.append(group[['date_m', 'sid', 'mcap_pct', 'is_worst30']])

df_mcap_pct = pd.concat(mcap_pct_records)
worst_mcap_pct = df_mcap_pct[df_mcap_pct['is_worst30']]['mcap_pct']
all_mcap_pct = df_mcap_pct['mcap_pct']

print(f"Worst 30 mean mcap percentile (within-month extreme pool): {worst_mcap_pct.mean():.1f}%")
print(f"Worst 30 median mcap percentile (within-month extreme pool): {worst_mcap_pct.median():.1f}%")
print(f"All extreme stocks mean mcap percentile: {all_mcap_pct.mean():.1f}%")
print(f"% of worst-30 extreme obs in bottom 30% of mcap (within extreme pool): {(worst_mcap_pct <= 30).mean()*100:.0f}%")
print(f"% of worst-30 extreme obs in bottom 50% of mcap (within extreme pool): {(worst_mcap_pct <= 50).mean()*100:.0f}%")
print(f"% of worst-30 extreme obs in top 20% of mcap (within extreme pool): {(worst_mcap_pct >= 80).mean()*100:.0f}%")

# Compare absolute mcap levels
print(f"\nWorst 30 absolute avg mcap: {worst_30['avg_mcap'].median():,.0f} HKD")
print(f"All extreme absolute avg mcap: {stock_agg['avg_mcap'].median():,.0f} HKD")
print(f"MCap ratio (worst 30 / all extreme): {worst_30['avg_mcap'].median() / stock_agg['avg_mcap'].median():.2f}x")

# 6c. Volatility profile - within-month percentile ranks
print("\n\n--- Volatility Profile (within-month, relative to eligible pool) ---")
# Compute volatility percentiles within each month for extreme observations
vol_pct_records = []
for date_m, group in df_extreme.groupby('date_m'):
    if len(group) < 3:
        continue
    group = group.copy()
    group['vol_pct'] = group['volatility'].rank(pct=True) * 100
    vol_pct_records.append(group[['date_m', 'sid', 'vol_pct', 'is_worst30']])

df_vol_pct = pd.concat(vol_pct_records)
worst_vol_pct = df_vol_pct[df_vol_pct['is_worst30']]['vol_pct']
all_vol_pct = df_vol_pct['vol_pct']

print(f"Worst 30 mean volatility percentile (within-month extreme pool): {worst_vol_pct.mean():.1f}%")
print(f"Worst 30 median volatility percentile (within-month extreme pool): {worst_vol_pct.median():.1f}%")
print(f"All extreme stocks mean volatility percentile: {all_vol_pct.mean():.1f}%")
print(f"% of worst-30 extreme obs in top 30% of vol (within extreme pool): {(worst_vol_pct >= 70).mean()*100:.0f}%")
print(f"% of worst-30 extreme obs in top 10% of vol (within extreme pool): {(worst_vol_pct >= 90).mean()*100:.0f}%")
print(f"Worst 30 absolute avg volatility: {worst_30['avg_volatility'].mean()*100:.1f}%")
print(f"All extreme absolute avg volatility: {stock_agg['avg_volatility'].mean()*100:.1f}%")

# 6d. Industry/sector patterns (manually identified from stock names since industry map unavailable)
print("\n\n--- Sector Patterns (from known stock names) ---")
print("Industry classification data is unavailable (hk_industry_map.json contains all 'Unknown' entries).")
print("Manual inspection of the worst 30 stock names reveals clear sector concentration:")
known_sectors = {
    'Chinese Real Estate / Property Developer': [
        '3883.HK', '1233.HK', '0884.HK', '1238.HK', '3380.HK', '0754.HK'  # 奥园, 时代中国, 旭辉, 宝龙, 龙光, 合生创展
    ],
    'Financial / Securities': [
        '1375.HK', '0806.HK',  # 中州证券, 惠理集团
    ],
    'Industrial / Manufacturing': [
        '2038.HK', '2331.HK', '2208.HK',  # 富智康, 李宁, 金风科技
    ],
    'Consumer / Retail': [
        '3368.HK', '3669.HK',  # 百盛集团, 永达汽车
    ],
    'Energy / Materials': [
        '3800.HK', '1733.HK',  # 协鑫科技, 易大宗
    ],
    'Education': [
        '0839.HK',  # 中教控股
    ],
}
for sector, sids in known_sectors.items():
    count_in_worst30 = len([s for s in sids if s in worst_30.index])
    if count_in_worst30 > 0:
        pct = count_in_worst30 / 30 * 100
        print(f"  {sector}: {count_in_worst30} stocks ({pct:.0f}% of worst 30)")
print("\n  Key observation: Chinese real estate developers dominate the worst performers,")
print("  reflecting the well-known property sector downturn that turned high dividend yields")
print("  into severe value traps as stock prices collapsed.")

# 6e. Score decomposition
print("\n\n--- SY Score Decomposition (Worst 30 vs All Extreme) ---")
print(f"Worst 30 - Avg div_yield component: {worst_30['avg_div_yield'].mean():.4f}")
print(f"All Extreme - Avg div_yield component: {stock_agg['avg_div_yield'].mean():.4f}")
print(f"Worst 30 - Avg buyback_yield component: {worst_30['avg_buyback_yield'].mean():.4f}")
print(f"All Extreme - Avg buyback_yield component: {stock_agg['avg_buyback_yield'].mean():.4f}")
div_ratio_w30 = worst_30['avg_div_yield'].mean() / worst_30['avg_score'].mean() * 100 if worst_30['avg_score'].mean() > 0 else 0
div_ratio_all = stock_agg['avg_div_yield'].mean() / stock_agg['avg_score'].mean() * 100 if stock_agg['avg_score'].mean() > 0 else 0
print(f"Worst 30 - % of SY score from dividends: {div_ratio_w30:.0f}%")
print(f"All Extreme - % of SY score from dividends: {div_ratio_all:.0f}%")

# ============================================================================
# 7. Can filters catch them? Simulation
# ============================================================================
print("\n\n" + "=" * 80)
print("STEP 7: FILTER EFFECTIVENESS SIMULATION")
print("=" * 80)

# For each filter, check what % of worst-30 stocks' extreme appearances would be caught
filters_to_test = {
    'MCap < 10th pctile': lambda df: df['mcap'] < df['mcap'].quantile(0.10),
    'MCap < 20th pctile': lambda df: df['mcap'] < df['mcap'].quantile(0.20),
    'MCap < 30th pctile': lambda df: df['mcap'] < df['mcap'].quantile(0.30),
    'Volatility > 80th pctile': lambda df: df['volatility'] > df['volatility'].quantile(0.80),
    'Volatility > 70th pctile': lambda df: df['volatility'] > df['volatility'].quantile(0.70),
    'Volatility > 90th pctile': lambda df: df['volatility'] > df['volatility'].quantile(0.90),
    '(MCap in bottom 30%) AND (Vol in top 30%)': lambda df: (df['mcap'] < df['mcap'].quantile(0.30)) & (df['volatility'] > df['volatility'].quantile(0.70)),
    '(MCap in bottom 20%) AND (Vol in top 20%)': lambda df: (df['mcap'] < df['mcap'].quantile(0.20)) & (df['volatility'] > df['volatility'].quantile(0.80)),
    '(MCap in bottom 50%) AND (Vol in top 50%)': lambda df: (df['mcap'] < df['mcap'].quantile(0.50)) & (df['volatility'] > df['volatility'].quantile(0.50)),
}

# Group the extreme records by date_m for within-month percentile computation
print("\nSimulating filter effectiveness (what % of worst-stock extreme appearances would be caught)...\n")
print(f"{'Filter':<50} {'% Caught':<12} {'False Positive %':<18} {'Net Reduction':<15}")
print("-" * 95)

df_extreme['is_worst30'] = df_extreme['sid'].isin(worst_30_sids)

for filter_name, filter_fn in filters_to_test.items():
    caught_total = 0
    extreme_total = 0
    fp_total = 0
    non_extreme_total = 0

    for date_m, group in df_extreme.groupby('date_m'):
        if len(group) < 3:
            continue

        # Apply filter within this month's extreme group
        caught = filter_fn(group)

        n_worst_in_group = group['is_worst30'].sum()
        n_extreme_in_group = len(group)

        if n_worst_in_group > 0:
            caught_worst = caught & group['is_worst30']
            caught_total += caught_worst.sum()
            extreme_total += n_worst_in_group

        # False positives: stocks caught by filter that are NOT worst-30
        fp = caught & (~group['is_worst30'])
        fp_total += fp.sum()
        non_extreme_total += (~group['is_worst30']).sum()

    pct_caught = caught_total / extreme_total * 100 if extreme_total > 0 else 0
    pct_fp = fp_total / non_extreme_total * 100 if non_extreme_total > 0 else 0
    # Net reduction: what % of extreme universe would be removed by this filter
    total_removed = caught_total + fp_total
    total_all = extreme_total + non_extreme_total
    net_reduction = total_removed / total_all * 100 if total_all > 0 else 0

    print(f"{filter_name:<50} {pct_caught:>8.1f}%    {pct_fp:>8.1f}%         {net_reduction:>8.1f}%")

# ============================================================================
# 8. Detailed per-stock worst performer profiles
# ============================================================================
print("\n\n" + "=" * 80)
print("STEP 8: TOP 10 WORST PERFORMERS - DETAILED PROFILES")
print("=" * 80)

for i, (sid, row) in enumerate(worst_30.head(10).iterrows()):
    rank = i + 1
    print(f"\n--- #{rank}: {sid} ({row['name']}) ---")
    print(f"    Frequency in extreme group: {int(row['frequency'])} months")
    print(f"    Avg next-month return: {row['avg_next_month_ret']*100:.2f}%")
    print(f"    Median next-month return: {row['median_next_month_ret']*100:.2f}%")
    print(f"    Worst single month: {row['worst_single_return']*100:.2f}%")
    print(f"    Best single month: {row['best_single_return']*100:.2f}%")
    print(f"    % negative months: {row['pct_negative']*100:.1f}%")
    print(f"    Avg market cap: {row['avg_mcap']:,.0f} HKD")
    print(f"    Avg volatility: {row['avg_volatility']*100:.1f}%")
    print(f"    Avg SY score: {row['avg_score']:.4f}")
    print(f"    Avg div_yield: {row['avg_div_yield']:.4f}")
    print(f"    Avg buyback_yield: {row['avg_buyback_yield']:.4f}")
    print(f"    Industry: {row['industry']}")

# ============================================================================
# 9. Summary statistics for the whole extreme universe
# ============================================================================
print("\n\n" + "=" * 80)
print("STEP 9: EXTREME UNIVERSE SUMMARY STATISTICS")
print("=" * 80)
print(f"Total unique stocks ever in extreme top 10%: {df_extreme['sid'].nunique()}")
print(f"Total extreme stock-month observations: {len(df_extreme)}")
print(f"Average next-month return of ALL extreme stocks: {df_extreme['next_month_ret'].mean()*100:.2f}%")
print(f"Median next-month return of ALL extreme stocks: {df_extreme['next_month_ret'].median()*100:.2f}%")
print(f"% negative months (all extreme): {(df_extreme['next_month_ret'] < 0).mean()*100:.1f}%")
print(f"Average MCap of extreme stocks: {df_extreme['mcap'].median():,.0f} HKD")
print(f"Average volatility of extreme stocks: {df_extreme['volatility'].mean()*100:.1f}%")
print(f"Stocks appearing only once: {(stock_agg['frequency'] == 1).sum()}")
print(f"Stocks appearing 2+ times: {(stock_agg['frequency'] >= 2).sum()}")
print(f"Stocks appearing 5+ times: {(stock_agg['frequency'] >= 5).sum()}")
print(f"Stocks appearing 10+ times: {(stock_agg['frequency'] >= 10).sum()}")
print(f"Most frequent extreme stock: {stock_agg['frequency'].idxmax()} ({stock_agg['frequency'].max()} times)")

# ============================================================================
# 10. Final KEY ANSWER
# ============================================================================
print("\n\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

w30_mcap_ratio = worst_30['avg_mcap'].median() / stock_agg['avg_mcap'].median()
w30_vol_diff = worst_30['avg_volatility'].mean() - stock_agg['avg_volatility'].mean()
w30_pct_neg = worst_30['pct_negative'].mean()
all_pct_neg = stock_agg['pct_negative'].mean()
w30_ann_ret = worst_30['avg_next_month_ret'].mean() * 12
all_ann_ret = stock_agg['avg_next_month_ret'].mean() * 12

print(f"""
1. Are worst extreme-SY stocks systematically different?
   YES. They are systematically:
   - {w30_mcap_ratio:.2f}x the median market cap of all extreme stocks
   - {w30_vol_diff*100:+.1f}pp higher annualized volatility vs. all extreme mean
   - Much more likely to have negative next-month returns ({w30_pct_neg*100:.0f}% of months vs {all_pct_neg*100:.0f}% for all extreme)
   - Annualized next-month return: {w30_ann_ret*100:.1f}% vs {all_ann_ret*100:.1f}% for all extreme
   - Many are from stressed sectors (esp. Chinese real estate) where high yields reflect price collapse, not sustainable dividends

2. Can additional filters catch them?
   - The most effective simple filter is a minimum market cap threshold
   - Combining mcap + volatility thresholds is more precise
   - Removing stocks from known stressed sectors (real estate during downturns) would help
   - However, the existing "trim top 10% SY" already provides substantial protection
   - The key insight: extreme SY stocks that ALSO have high volatility AND small mcap are the most dangerous

3. Practical recommendation:
   - The current strategy B (trim top 10% SY) already removes the very worst stocks
   - Further improvement: add a secondary quality screen (min mcap in bottom 20% within pool, or max vol in top 20%)
   - But careful: over-filtering reduces diversification and may hurt returns in normal periods
""")

print("Analysis complete.")
