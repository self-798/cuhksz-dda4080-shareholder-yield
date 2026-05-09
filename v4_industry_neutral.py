"""
Industry-Neutralized Shareholder Yield Strategies
=================================================
Creates B_neutral and C_neutral (industry-neutralized versions of B and C)
and compares them against raw B and C.

Industry neutralization: within each industry group, z-score normalize SY scores
each month, then use normalized scores for ranking.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
import re

warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 10

print("All imports successful.")

# ========== 1. LOAD DATA (same pipeline as v4 notebook cell 1) ==========
print("\n" + "=" * 70)
print("STEP 1: Loading data pipeline...")
print("=" * 70)
print("Loading dividend yield data...")
dy = pd.read_hdf('data/hk_dividendyield.h5')
if isinstance(dy, pd.DataFrame):
    dy = dy.iloc[:, 0]
dy = dy.dropna().reset_index()
dy.columns = ['date', 'sid', 'dy']
dy['date'] = pd.to_datetime(dy['date'])

dy_pivot = dy.pivot(index='date', columns='sid', values='dy').sort_index()

print("Loading price data...")
price_cols = pd.read_csv('data/hk_price.csv', usecols=['date', 'sid', 'close', 'AdjClose', 'amount'])
price_cols['date'] = pd.to_datetime(price_cols['date'])
price_cols['date_m'] = price_cols['date'].dt.to_period('M')

monthly_price = price_cols.sort_values('date').groupby(['date_m', 'sid']).last().reset_index()
monthly_price_pivot = monthly_price.pivot(index='date_m', columns='sid', values='AdjClose')
monthly_ret = monthly_price_pivot.pct_change(fill_method=None).shift(-1)

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

print("Loading HSCI constituents...")
hsci = pd.read_csv('data/HSCI.csv')
hsci['date'] = pd.to_datetime(hsci['date'])
hsci['is_hsci'] = 1
hsci = hsci.drop_duplicates(subset=['date', 'sid'])

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

print("Calculating price pivots...")
price = pd.read_csv('data/hk_price.csv', usecols=['date', 'sid', 'close', 'amount'])
price['date'] = pd.to_datetime(price['date'])
price_pivot = price.pivot(index='date', columns='sid', values='close').sort_index()

print("Calculating volatility (252-day annualized)...")
ret_pivot = price_pivot.pct_change()
vol_pivot = ret_pivot.rolling(window=252, min_periods=100).std() * np.sqrt(252)
vol_monthly = vol_pivot.resample('ME').last()
vol_monthly_stacked = vol_monthly.stack().reset_index()
vol_monthly_stacked.columns = ['date', 'sid', 'volatility']

print("Calculating ADTV (63-day average)...")
amount_pivot = price.pivot(index='date', columns='sid', values='amount').sort_index()
adv_90d_pivot = amount_pivot.rolling(window=63, min_periods=21).mean()
adv_monthly = adv_90d_pivot.resample('ME').last()
adv_monthly_stacked = adv_monthly.stack().reset_index()
adv_monthly_stacked.columns = ['date', 'sid', 'adtv_3m']

dy['date_m'] = dy['date'].dt.to_period('M')
vol_monthly_stacked['date_m'] = vol_monthly_stacked['date'].dt.to_period('M')
adv_monthly_stacked['date_m'] = adv_monthly_stacked['date'].dt.to_period('M')
hsci['date_m'] = hsci['date'].dt.to_period('M')

# ========== 2. MERGE FACTORS ==========
print("Merging all factors...")
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

print(f"Factors shape: {df_factors.shape}")
print(f"Date range: {df_factors['date_m'].min()} ~ {df_factors['date_m'].max()}")

BACKTEST_START = pd.Period('2012-01', 'M')


# ========== 3. INDUSTRY MAPPING ==========
def get_industry(sid):
    """
    Map HK stock code to industry group based on code ranges.
    Expects sid like '0005.HK' or '0700.HK'.
    Handles codes with '!' suffix (e.g., '0033!1.HK') by extracting the base numeric part.
    """
    try:
        # Extract the numeric code before .HK
        code_str = sid.split('.')[0]
        # Remove any non-digit prefix (clean the '!' suffix etc.)
        code_digits = re.sub(r'[^0-9]', '', code_str)
        if not code_digits:
            return 'Unknown'
        code = int(code_digits)
    except (ValueError, IndexError):
        return 'Unknown'

    # Apply code range mapping
    if 1 <= code <= 99:
        return 'Financials_Utilities_Conglomerates'
    elif 100 <= code <= 999:
        return 'Property_Construction_Industrial'
    elif 1000 <= code <= 1999:
        return 'Consumer_Retail_Healthcare'
    elif 2000 <= code <= 2999:
        return 'Diversified'
    elif 3000 <= code <= 3999:
        return 'Energy_Materials'
    elif 4000 <= code <= 4999:
        return 'Others'
    elif 6000 <= code <= 6999:
        return 'Banking_Financials'
    elif 7000 <= code <= 7999:
        return 'Technology'
    elif 8000 <= code <= 9999:
        return 'SmallCaps_Misc'
    else:
        return 'Unknown'


# Show industry distribution
df_factors['industry'] = df_factors['sid'].apply(get_industry)
ind_counts = df_factors.groupby('industry')['sid'].nunique().sort_values(ascending=False)
print("\nIndustry mapping based on code ranges:")
for ind, cnt in ind_counts.items():
    print(f"  {ind}: {cnt} stocks")


# ========== 4. CORE SCORE COMPUTATION (shared by all strategies) ==========
def compute_sy_scores(pool_df, date_m):
    """
    Compute Shareholder Yield score for a pool of stocks at a given month.
    Returns a DataFrame with 'div_yield', 'buyback_yield', and 'score' columns.
    """
    end_date = date_m.to_timestamp(how='end')
    dy_36m = dy_pivot.loc[:end_date].tail(36)
    price_36m = price_pivot.reindex(dy_36m.index, method='ffill')

    # Filter to stocks with some dividend history or buyback activity
    has_div = dy_36m.columns[dy_36m.gt(0).any(axis=0)] if len(dy_36m) > 0 else pd.Index([])
    pool = pool_df[(pool_df['sid'].isin(has_div)) | (pool_df['buyback_36m'] > 0)].copy()

    # Liquidity filter
    adtv_th = pool_df['adtv_3m'].quantile(0.2)
    mcap_th = pool_df['mcap'].quantile(0.2)
    pool = pool[(pool['adtv_3m'] >= adtv_th) & (pool['mcap'] >= mcap_th)]

    if len(pool) == 0:
        return pd.DataFrame()

    # Compute dividend yield
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

    pool['industry'] = pool['sid'].apply(get_industry)
    return pool


# ========== 5. STRATEGY FUNCTIONS ==========

def strategy_B_raw(group):
    """Strategy B: Pure Shareholder Yield, WITH extreme trim (top 10% removed)."""
    current_m = group.name
    if isinstance(current_m, tuple):
        current_m = current_m[0]

    pool = compute_sy_scores(group, current_m)
    if len(pool) == 0:
        return []

    # Remove top 10% extreme values
    cutoff = pool['score'].quantile(0.9)
    pool = pool[pool['score'] <= cutoff]
    if len(pool) == 0:
        return []

    n_top = max(10, int(len(pool) * 0.20))
    pool = pool.nlargest(n_top, 'score')
    n_final = min(20, len(pool))
    return pool.nlargest(n_final, 'score')['sid'].tolist()


def strategy_C_raw(group):
    """Strategy C: Shareholder Yield + extreme trim + low-vol filter."""
    current_m = group.name
    if isinstance(current_m, tuple):
        current_m = current_m[0]

    pool = compute_sy_scores(group, current_m)
    if len(pool) == 0:
        return []

    # Remove top 10% extreme values
    cutoff = pool['score'].quantile(0.9)
    pool = pool[pool['score'] <= cutoff]
    if len(pool) == 0:
        return []

    n_top = max(10, int(len(pool) * 0.20))
    pool = pool.nlargest(n_top, 'score')

    n_final = min(20, len(pool))
    return pool.nsmallest(n_final, 'volatility')['sid'].tolist()


def strategy_B_neutral(group):
    """Strategy B_neutral: Industry-neutralized SY + extreme trim."""
    current_m = group.name
    if isinstance(current_m, tuple):
        current_m = current_m[0]

    pool = compute_sy_scores(group, current_m)
    if len(pool) == 0:
        return []

    # Industry neutralization: z-score within each industry
    pool['score_z'] = np.nan
    for ind, ind_group in pool.groupby('industry'):
        if len(ind_group) < 3:
            # Too few stocks in this industry to normalize meaningfully
            pool.loc[ind_group.index, 'score_z'] = 0.0
        else:
            mu = ind_group['score'].mean()
            sigma = ind_group['score'].std()
            if sigma == 0 or np.isnan(sigma):
                pool.loc[ind_group.index, 'score_z'] = 0.0
            else:
                pool.loc[ind_group.index, 'score_z'] = (ind_group['score'] - mu) / sigma

    pool = pool.dropna(subset=['score_z'])
    if len(pool) == 0:
        return []

    # Remove top 10% extreme values (based on normalized score)
    cutoff = pool['score_z'].quantile(0.9)
    pool = pool[pool['score_z'] <= cutoff]
    if len(pool) == 0:
        return []

    n_top = max(10, int(len(pool) * 0.20))
    pool = pool.nlargest(n_top, 'score_z')
    n_final = min(20, len(pool))
    return pool.nlargest(n_final, 'score_z')['sid'].tolist()


def strategy_C_neutral(group):
    """Strategy C_neutral: Industry-neutralized SY + extreme trim + low-vol filter."""
    current_m = group.name
    if isinstance(current_m, tuple):
        current_m = current_m[0]

    pool = compute_sy_scores(group, current_m)
    if len(pool) == 0:
        return []

    # Industry neutralization: z-score within each industry
    pool['score_z'] = np.nan
    for ind, ind_group in pool.groupby('industry'):
        if len(ind_group) < 3:
            pool.loc[ind_group.index, 'score_z'] = 0.0
        else:
            mu = ind_group['score'].mean()
            sigma = ind_group['score'].std()
            if sigma == 0 or np.isnan(sigma):
                pool.loc[ind_group.index, 'score_z'] = 0.0
            else:
                pool.loc[ind_group.index, 'score_z'] = (ind_group['score'] - mu) / sigma

    pool = pool.dropna(subset=['score_z'])
    if len(pool) == 0:
        return []

    # Remove top 10% extreme values (based on normalized score)
    cutoff = pool['score_z'].quantile(0.9)
    pool = pool[pool['score_z'] <= cutoff]
    if len(pool) == 0:
        return []

    n_top = max(10, int(len(pool) * 0.20))
    pool = pool.nlargest(n_top, 'score_z')

    n_final = min(20, len(pool))
    return pool.nsmallest(n_final, 'volatility')['sid'].tolist()


# ========== 6. RUN STRATEGIES ==========
print("\n" + "=" * 70)
print("STEP 2: Running all 4 strategies (raw + neutral)...")
print("=" * 70)

print("Running Strategy B (raw)...")
sigs_B_raw = df_factors.groupby('date_m', group_keys=True).apply(strategy_B_raw)
print(f"  Signal periods: {len(sigs_B_raw)}, Avg holding: {sigs_B_raw.apply(len).mean():.1f}")

print("Running Strategy C (raw)...")
sigs_C_raw = df_factors.groupby('date_m', group_keys=True).apply(strategy_C_raw)
print(f"  Signal periods: {len(sigs_C_raw)}, Avg holding: {sigs_C_raw.apply(len).mean():.1f}")

print("Running Strategy B_neutral...")
sigs_B_neutral = df_factors.groupby('date_m', group_keys=True).apply(strategy_B_neutral)
print(f"  Signal periods: {len(sigs_B_neutral)}, Avg holding: {sigs_B_neutral.apply(len).mean():.1f}")

print("Running Strategy C_neutral...")
sigs_C_neutral = df_factors.groupby('date_m', group_keys=True).apply(strategy_C_neutral)
print(f"  Signal periods: {len(sigs_C_neutral)}, Avg holding: {sigs_C_neutral.apply(len).mean():.1f}")


# ========== 7. BACKTEST FUNCTION ==========
def run_backtest(signals, monthly_ret, mcap_pivot, start_period, rebal_freq=3, max_weight=0.1):
    """Run a backtest and return (eq_series, cw_series)."""
    dates = signals.index[signals.index >= start_period]
    portfolio_ret_eq = []
    portfolio_ret_cw = []
    valid_dates = []
    current_stocks = []

    for t_idx, date_m in enumerate(dates):
        if date_m not in monthly_ret.index:
            continue

        if t_idx % rebal_freq == 0:
            new_stocks = signals.loc[date_m]
            if len(new_stocks) > 0:
                current_stocks = new_stocks

        if len(current_stocks) == 0:
            continue

        nxt_ret = monthly_ret.loc[date_m].reindex(current_stocks).fillna(0)

        # Equal weight
        eq_w = np.ones(len(current_stocks)) / len(current_stocks)
        ret_eq = np.sum(eq_w * nxt_ret.values)

        # Cap weight
        if date_m in mcap_pivot.index:
            mcap_vals = mcap_pivot.loc[date_m].reindex(current_stocks).fillna(0)
            total_mcap = mcap_vals.sum()
            if total_mcap > 0:
                cw = mcap_vals / total_mcap
                if max_weight is not None and max_weight < 1.0:
                    for _ in range(20):
                        if not (cw > max_weight + 1e-6).any():
                            break
                        cw[cw > max_weight] = max_weight
                        mask = cw < max_weight
                        if not mask.any():
                            break
                        remaining = 1.0 - cw[~mask].sum()
                        if remaining <= 0:
                            break
                        cw[mask] = cw[mask] / cw[mask].sum() * remaining
                ret_cw = np.sum(cw.values * nxt_ret.values)
            else:
                ret_cw = ret_eq
        else:
            ret_cw = ret_eq

        portfolio_ret_eq.append(ret_eq)
        portfolio_ret_cw.append(ret_cw)
        valid_dates.append(date_m + 1)

    eq_series = pd.Series(portfolio_ret_eq, index=pd.Index(valid_dates, name='date_return'))
    cw_series = pd.Series(portfolio_ret_cw, index=pd.Index(valid_dates, name='date_return'))
    return eq_series, cw_series


print("\n" + "=" * 70)
print("STEP 3: Running backtests...")
print("=" * 70)

bt_B_raw_eq, bt_B_raw_cw = run_backtest(sigs_B_raw, monthly_ret, mcap_pivot, BACKTEST_START)
bt_C_raw_eq, bt_C_raw_cw = run_backtest(sigs_C_raw, monthly_ret, mcap_pivot, BACKTEST_START)
bt_B_neutral_eq, bt_B_neutral_cw = run_backtest(sigs_B_neutral, monthly_ret, mcap_pivot, BACKTEST_START)
bt_C_neutral_eq, bt_C_neutral_cw = run_backtest(sigs_C_neutral, monthly_ret, mcap_pivot, BACKTEST_START)

print(f"Backtest periods:")
print(f"  B (raw):       {len(bt_B_raw_eq)} months")
print(f"  B (neutral):   {len(bt_B_neutral_eq)} months")
print(f"  C (raw):       {len(bt_C_raw_eq)} months")
print(f"  C (neutral):   {len(bt_C_neutral_eq)} months")


# ========== 8. PERFORMANCE METRICS ==========
def calc_metrics(returns):
    ann_ret = returns.mean() * 12
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else np.nan
    cum_ret = (1 + returns).cumprod()
    max_dd = (cum_ret / cum_ret.cummax().clip(lower=1.0) - 1).min()
    total_ret = cum_ret.iloc[-1] if len(cum_ret) > 0 else 1.0
    # Additional metrics
    pos_months = (returns > 0).mean()
    # Downside deviation
    downside = returns[returns < 0].std() * np.sqrt(12) if len(returns[returns < 0]) > 0 else 0
    sortino = ann_ret / downside if downside > 0 else np.nan
    return ann_ret, ann_vol, sharpe, max_dd, total_ret, pos_months, sortino


# Build side-by-side comparison table
print("\n" + "=" * 80)
print("PERFORMANCE COMPARISON: RAW vs NEUTRALIZED (Equal Weight)")
print("=" * 80)

results = {}
for name, series in [
    ('B: Raw EW', bt_B_raw_eq),
    ('B: Neutral EW', bt_B_neutral_eq),
    ('C: Raw EW', bt_C_raw_eq),
    ('C: Neutral EW', bt_C_neutral_eq),
]:
    ann_ret, ann_vol, sharpe, max_dd, total_ret, pos_mo, sortino = calc_metrics(series)
    results[name] = {
        'Ann Return': f'{ann_ret:.2%}',
        'Ann Vol': f'{ann_vol:.2%}',
        'Sharpe': f'{sharpe:.2f}',
        'Sortino': f'{sortino:.2f}',
        'Max DD': f'{max_dd:.2%}',
        'Total Ret': f'{total_ret:.2f}x',
        'Win Rate': f'{pos_mo:.1%}',
    }

df_results = pd.DataFrame(results).T
print(df_results.to_string())

print()
print("=" * 80)
print("PERFORMANCE COMPARISON: RAW vs NEUTRALIZED (Cap Weight)")
print("=" * 80)

results_cw = {}
for name, series in [
    ('B: Raw CW', bt_B_raw_cw),
    ('B: Neutral CW', bt_B_neutral_cw),
    ('C: Raw CW', bt_C_raw_cw),
    ('C: Neutral CW', bt_C_neutral_cw),
]:
    ann_ret, ann_vol, sharpe, max_dd, total_ret, pos_mo, sortino = calc_metrics(series)
    results_cw[name] = {
        'Ann Return': f'{ann_ret:.2%}',
        'Ann Vol': f'{ann_vol:.2%}',
        'Sharpe': f'{sharpe:.2f}',
        'Sortino': f'{sortino:.2f}',
        'Max DD': f'{max_dd:.2%}',
        'Total Ret': f'{total_ret:.2f}x',
        'Win Rate': f'{pos_mo:.1%}',
    }

df_results_cw = pd.DataFrame(results_cw).T
print(df_results_cw.to_string())


# ========== 9. IMPORTANCE: Raw-vs-Neutral Difference Table ==========
print("\n" + "=" * 80)
print("DELTA TABLE: Raw vs Neutral Improvement (Equal Weight)")
print("=" * 80)

delta_results = {}
for base_name, raw_series, neutral_series in [
    ('B', bt_B_raw_eq, bt_B_neutral_eq),
    ('C', bt_C_raw_eq, bt_C_neutral_eq),
]:
    raw_metrics = calc_metrics(raw_series)
    neutral_metrics = calc_metrics(neutral_series)
    delta_results[f'{base_name} Delta'] = {
        'Ann Return': f'{neutral_metrics[0] - raw_metrics[0]:+.2%}',
        'Ann Vol': f'{neutral_metrics[1] - raw_metrics[1]:+.2%}',
        'Sharpe': f'{neutral_metrics[2] - raw_metrics[2]:+.2f}',
        'Max DD': f'{neutral_metrics[3] - raw_metrics[3]:+.2%}',
        'Total Ret': f'{neutral_metrics[4] - raw_metrics[4]:+.2f}x',
        'Win Rate': f'{neutral_metrics[5] - raw_metrics[5]:+.1%}',
    }

df_delta = pd.DataFrame(delta_results).T
print(df_delta.to_string())


# ========== 10. CORRELATION ANALYSIS ==========
print("\n" + "=" * 80)
print("CORRELATION: Raw vs Neutral Monthly Returns")
print("=" * 80)
common_B = bt_B_raw_eq.index.intersection(bt_B_neutral_eq.index)
corr_B = bt_B_raw_eq.reindex(common_B).corr(bt_B_neutral_eq.reindex(common_B))
common_C = bt_C_raw_eq.index.intersection(bt_C_neutral_eq.index)
corr_C = bt_C_raw_eq.reindex(common_C).corr(bt_C_neutral_eq.reindex(common_C))
print(f"  B raw vs B neutral correlation: {corr_B:.4f}")
print(f"  C raw vs C neutral correlation: {corr_C:.4f}")

# Tracking error (annualized std of return difference)
te_B = (bt_B_raw_eq.reindex(common_B) - bt_B_neutral_eq.reindex(common_B)).std() * np.sqrt(12)
te_C = (bt_C_raw_eq.reindex(common_C) - bt_C_neutral_eq.reindex(common_C)).std() * np.sqrt(12)
print(f"  B tracking error (ann): {te_B:.2%}")
print(f"  C tracking error (ann): {te_C:.2%}")


# ========== 11. PLOTS ==========
print("\n" + "=" * 70)
print("STEP 4: Generating plots...")
print("=" * 70)

# --- Plot 1: Cumulative Returns (B raw vs B neutral) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

cum_B_raw = (1 + bt_B_raw_eq).cumprod()
cum_B_neutral = (1 + bt_B_neutral_eq).cumprod()
cum_C_raw = (1 + bt_C_raw_eq).cumprod()
cum_C_neutral = (1 + bt_C_neutral_eq).cumprod()

all_idx_B = cum_B_raw.index.intersection(cum_B_neutral.index)
all_idx_C = cum_C_raw.index.intersection(cum_C_neutral.index)

# Panel 1: Strategy B comparison
ax = axes[0]
ax.plot(all_idx_B.to_timestamp(), cum_B_raw.reindex(all_idx_B),
        label='B: Raw (with trim)', color='blue', linewidth=2.0)
ax.plot(all_idx_B.to_timestamp(), cum_B_neutral.reindex(all_idx_B),
        label='B: Neutral (industry z-score)', color='orange', linewidth=2.0)
ax.set_title('Strategy B: Raw vs Industry-Neutralized', fontsize=13, fontweight='bold')
ax.set_ylabel('Cumulative Net Value (EW)', fontsize=11)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)
ax.axhline(y=1.0, color='black', linewidth=0.5)

# Panel 2: Strategy C comparison
ax = axes[1]
ax.plot(all_idx_C.to_timestamp(), cum_C_raw.reindex(all_idx_C),
        label='C: Raw (trim+lowvol)', color='green', linewidth=2.0)
ax.plot(all_idx_C.to_timestamp(), cum_C_neutral.reindex(all_idx_C),
        label='C: Neutral (industry z-score)', color='red', linewidth=2.0)
ax.set_title('Strategy C: Raw vs Industry-Neutralized', fontsize=13, fontweight='bold')
ax.set_ylabel('Cumulative Net Value (EW)', fontsize=11)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)
ax.axhline(y=1.0, color='black', linewidth=0.5)

plt.suptitle('Industry-Neutralized Shareholder Yield: Cumulative Return Comparison',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('pict/v4_plot_industry_cumulative.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: pict/v4_plot_industry_cumulative.png")

# --- Plot 2: Drawdown comparison ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, raw_series, neutral_series, raw_label, neutral_label, raw_color, neutral_color, title in [
    (axes[0], bt_B_raw_eq, bt_B_neutral_eq, 'B: Raw', 'B: Neutral', 'blue', 'orange',
     'Strategy B Drawdown'),
    (axes[1], bt_C_raw_eq, bt_C_neutral_eq, 'C: Raw', 'C: Neutral', 'green', 'red',
     'Strategy C Drawdown'),
]:
    for series, label, color in [
        (raw_series, raw_label, raw_color),
        (neutral_series, neutral_label, neutral_color),
    ]:
        cum = (1 + series).cumprod()
        dd = cum / cum.cummax().clip(lower=1.0) - 1
        ax.plot(dd.index.to_timestamp(), dd.values, label=label, color=color, linewidth=1.8)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Drawdown', fontsize=10)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)

plt.suptitle('Industry-Neutralized Shareholder Yield: Drawdown Comparison',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('pict/v4_plot_industry_drawdown.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: pict/v4_plot_industry_drawdown.png")

# --- Plot 3: Rolling 12-month return difference ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, raw_series, neutral_series, label_prefix, color in [
    (axes[0], bt_B_raw_eq, bt_B_neutral_eq, 'B', 'blue'),
    (axes[1], bt_C_raw_eq, bt_C_neutral_eq, 'C', 'green'),
]:
    common_idx = raw_series.index.intersection(neutral_series.index)
    diff = neutral_series.reindex(common_idx) - raw_series.reindex(common_idx)
    rolling_diff = diff.rolling(12).mean() * 12 * 100  # annualized rolling diff in %

    ax.plot(rolling_diff.index.to_timestamp(), rolling_diff.values,
            color=color, linewidth=1.8)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.set_title(f'{label_prefix}: Neutral - Raw (12-mo rolling)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Annualized Return Difference (%)', fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle('Industry Neutralization Benefit: Rolling 12-Month Return Difference',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('pict/v4_plot_industry_rolling_diff.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: pict/v4_plot_industry_rolling_diff.png")

# --- Plot 4: Annual return bar chart ---
fig, ax = plt.subplots(figsize=(12, 6))

# Compute annual returns
common_all = bt_B_raw_eq.index.intersection(bt_B_neutral_eq.index).intersection(
    bt_C_raw_eq.index).intersection(bt_C_neutral_eq.index)

ann_returns = {}
for name, series in [
    ('B: Raw', bt_B_raw_eq),
    ('B: Neutral', bt_B_neutral_eq),
    ('C: Raw', bt_C_raw_eq),
    ('C: Neutral', bt_C_neutral_eq),
]:
    sr = series.reindex(common_all)
    # Group by year
    years = [idx.year for idx in sr.index.to_timestamp()]
    yr_ret = sr.groupby(years).apply(lambda x: (1 + x).prod() - 1) * 100
    ann_returns[name] = yr_ret

yr_df = pd.DataFrame(ann_returns)
x = np.arange(len(yr_df.index))
width = 0.2

bars1 = ax.bar(x - 1.5*width, yr_df['B: Raw'], width, label='B: Raw', color='blue', alpha=0.7)
bars2 = ax.bar(x - 0.5*width, yr_df['B: Neutral'], width, label='B: Neutral', color='orange', alpha=0.7)
bars3 = ax.bar(x + 0.5*width, yr_df['C: Raw'], width, label='C: Raw', color='green', alpha=0.7)
bars4 = ax.bar(x + 1.5*width, yr_df['C: Neutral'], width, label='C: Neutral', color='red', alpha=0.7)

ax.set_title('Annual Returns: Raw vs Industry-Neutralized', fontsize=14, fontweight='bold')
ax.set_ylabel('Annual Return (%)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(yr_df.index, rotation=45)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig('pict/v4_plot_industry_annual_returns.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: pict/v4_plot_industry_annual_returns.png")

# --- Plot 5: Industry exposure over time (heatmap-like) ---
# Show average industry weights for raw vs neutral
print("\nCalculating industry exposure differences...")
ind_exposure_raw = []
ind_exposure_neutral = []

for date_m in sorted(df_factors['date_m'].unique()):
    if date_m < BACKTEST_START:
        continue
    # Get signals for this month
    if date_m in sigs_B_raw.index:
        stocks_raw = sigs_B_raw.loc[date_m]
        inds_raw = [get_industry(s) for s in stocks_raw]
        for ind in inds_raw:
            ind_exposure_raw.append({'date_m': date_m, 'industry': ind, 'strategy': 'B Raw'})
    if date_m in sigs_B_neutral.index:
        stocks_neutral = sigs_B_neutral.loc[date_m]
        inds_neutral = [get_industry(s) for s in stocks_neutral]
        for ind in inds_neutral:
            ind_exposure_neutral.append({'date_m': date_m, 'industry': ind, 'strategy': 'B Neutral'})

df_ind_raw = pd.DataFrame(ind_exposure_raw)
df_ind_neutral = pd.DataFrame(ind_exposure_neutral)

if len(df_ind_raw) > 0 and len(df_ind_neutral) > 0:
    # Aggregate by industry
    raw_counts = df_ind_raw.groupby('industry').size()
    neutral_counts = df_ind_neutral.groupby('industry').size()
    raw_pct = raw_counts / raw_counts.sum()
    neutral_pct = neutral_counts / neutral_counts.sum()

    fig, ax = plt.subplots(figsize=(14, 6))
    ind_order = sorted(set(raw_pct.index) | set(neutral_pct.index))
    x = np.arange(len(ind_order))
    width = 0.35

    rp = [raw_pct.get(ind, 0) * 100 for ind in ind_order]
    npct = [neutral_pct.get(ind, 0) * 100 for ind in ind_order]

    ax.bar(x - width/2, rp, width, label='B: Raw', color='blue', alpha=0.7)
    ax.bar(x + width/2, npct, width, label='B: Neutral', color='orange', alpha=0.7)
    ax.set_title('Industry Allocation: Strategy B Raw vs Neutral', fontsize=14, fontweight='bold')
    ax.set_ylabel('% of Total Holdings', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(ind_order, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('pict/v4_plot_industry_exposure.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: pict/v4_plot_industry_exposure.png")


# ========== 12. FINAL SUMMARY ==========
print("\n" + "=" * 80)
print("ANALYSIS: Does industry neutralization help?")
print("=" * 80)

# Compute key comparison stats for B
b_raw_ann = bt_B_raw_eq.mean() * 12
b_neutral_ann = bt_B_neutral_eq.mean() * 12
b_raw_sharpe = b_raw_ann / (bt_B_raw_eq.std() * np.sqrt(12))
b_neutral_sharpe = b_neutral_ann / (bt_B_neutral_eq.std() * np.sqrt(12))
cum_B_raw_end = (1 + bt_B_raw_eq).cumprod().iloc[-1]
cum_B_neutral_end = (1 + bt_B_neutral_eq).cumprod().iloc[-1]

c_raw_ann = bt_C_raw_eq.mean() * 12
c_neutral_ann = bt_C_neutral_eq.mean() * 12
c_raw_sharpe = c_raw_ann / (bt_C_raw_eq.std() * np.sqrt(12))
c_neutral_sharpe = c_neutral_ann / (bt_C_neutral_eq.std() * np.sqrt(12))
cum_C_raw_end = (1 + bt_C_raw_eq).cumprod().iloc[-1]
cum_C_neutral_end = (1 + bt_C_neutral_eq).cumprod().iloc[-1]

print(f"""
  STRATEGY B (Pure SY + extreme trim):
  -------------------------------------
  Raw (EW):         Ann Ret = {b_raw_ann:.2%}, Sharpe = {b_raw_sharpe:.2f}, Total Ret = {cum_B_raw_end:.2f}x
  Neutral (EW):     Ann Ret = {b_neutral_ann:.2%}, Sharpe = {b_neutral_sharpe:.2f}, Total Ret = {cum_B_neutral_end:.2f}x
  Improvement:      Ann Ret delta = {(b_neutral_ann - b_raw_ann):+.2%}, Sharpe delta = {(b_neutral_sharpe - b_raw_sharpe):+.2f}

  STRATEGY C (Trim + LowVol):
  ---------------------------
  Raw (EW):         Ann Ret = {c_raw_ann:.2%}, Sharpe = {c_raw_sharpe:.2f}, Total Ret = {cum_C_raw_end:.2f}x
  Neutral (EW):     Ann Ret = {c_neutral_ann:.2%}, Sharpe = {c_neutral_sharpe:.2f}, Total Ret = {cum_C_neutral_end:.2f}x
  Improvement:      Ann Ret delta = {(c_neutral_ann - c_raw_ann):+.2%}, Sharpe delta = {(c_neutral_sharpe - c_raw_sharpe):+.2f}

  CORRELATION ANALYSIS:
  - B raw vs neutral monthly returns correlate at {corr_B:.2f}
  - C raw vs neutral monthly returns correlate at {corr_C:.2f}
  - B tracking error: {te_B:.2%} annualized
  - C tracking error: {te_C:.2%} annualized

  INTERPRETATION:
  - Industry neutralization aims to remove sector biases from raw SY rankings.
  - If certain industries (e.g., Financials) have structurally higher payout ratios,
    raw SY strategies may overweight them. Neutralization within-industry ensures
    stock selection within each sector is based on relative (not absolute) scores.
  - The net effect depends on: (a) whether sector biases exist in raw SY,
    (b) whether the biased sectors outperform or underperform.
  - A positive delta means neutralization helped; negative means it hurt.
  - High correlation between raw and neutral indicates the ranking changes are
    marginal, suggesting sector concentration was not extreme to begin with.
  - Low correlation (with improved performance) would suggest sector-neutral
    selection materially improved stock picking.
""")

# Check industry concentration in raw vs neutral for a representative month
rep_month = pd.Period('2025-06', 'M')
if rep_month in sigs_B_raw.index and rep_month in sigs_B_neutral.index:
    print(f"  Industry breakdown for {rep_month}:")
    print(f"  B Raw:      {pd.Series([get_industry(s) for s in sigs_B_raw.loc[rep_month]]).value_counts().to_dict()}")
    print(f"  B Neutral:  {pd.Series([get_industry(s) for s in sigs_B_neutral.loc[rep_month]]).value_counts().to_dict()}")

print("\n" + "=" * 80)
print("DONE. All results and plots generated.")
print("=" * 80)
