# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quantitative finance research project for DDA4080 (Financial Engineering, CUHK-Shenzhen). Develops and backtests multi-factor stock selection strategies on the Hong Kong stock market (HSCI constituents), culminating in a **Shareholder Yield strategy** that combines dividend yield, buyback yield, and low-volatility filtering.

## Repository Structure

- **Root `.ipynb` files** — Working/draft notebooks for each strategy version (v1 buyback, v2 dividend+low-vol, v3 combined shareholder yield)
- **`Final/code/`** — Clean, final versions of all notebooks and data pipeline
- **`data/`** — Market data files (prices, dividends, buybacks, Fama-French factors, HSCI constituents)
- **`doc/`** — Research report (LaTeX source, PDF, Markdown)
- **`pict/`** — Generated charts (cumulative returns, drawdowns, pool compositions)
- **`fin4210hw6/`** — Unrelated FIN4210 homework (Black-Scholes options pricing)

## Key Notebooks (in `Final/code/`)

| Notebook | Purpose |
|----------|---------|
| `data_1.ipynb` through `data_4.ipynb` | Data pipeline: load, clean, merge price/dividend/buyback/HSCI data |
| `factor_1.ipynb`, `factor_2.ipynb` | Factor construction and Fama-French 3-factor regression analysis |
| `factor_回购.ipynb` | Pure buyback factor strategy (v1) |
| `v3_综合收益率.ipynb` | Final Shareholder Yield strategy (v3): dividend + buyback + low-vol |

## Data Files

All data is sourced from **RiceQuant API** (see `rqAPI.txt` for credentials):

| File | Content |
|------|---------|
| `hk_price.csv` (~600MB) | Daily OHLCV + adjusted close for all HK stocks |
| `hk_dividendyield.h5` | Monthly dividend yield per stock |
| `hk_shares.h5` | Monthly shares outstanding (for market cap calculation) |
| `HSCI.csv` | Monthly HSCI index constituent list |
| `em_buyback_filtered.csv` | Stock buyback records (date, stock code, amount) |
| `hk_ff3_factors.csv` | Hong Kong Fama-French 3 factors (MKT, SMB, HML) |
| `hk_risk_free_rate.csv` | Risk-free rate |
| `hk_bm_monthly.csv` | Book-to-market ratios |
| `hk_fcff.csv` | Free cash flow to firm |

## Strategy Architecture (v3 — Shareholder Yield)

The core strategy pipeline has 4 stages repeated each rebalance month:

1. **Universe filter**: HSCI constituents with any dividend or buyback activity in past 36 months; ADTV and market cap both in top 80% (exclude bottom 20% tail)
2. **Factor scoring**: `ShareholderYield = DivYield + BuybackYield` where:
   - `DivYield = (3-year average monthly DPS) / current price`
   - `BuybackYield = (total 3-year buyback / 3) / current market cap`
3. **Outlier trim + top selection**: Remove top 1% extreme scores; keep top 20% ranked by score
4. **Low-vol filter**: From remaining pool, select 10-20 stocks with lowest 1-year annualized daily volatility

Configuration: quarterly rebalance, equal-weight or capped market-cap weight (10-15% individual cap), backtest from 2012-01.

## Statistical Validation Pattern

All strategies are validated through:
- **Quintile portfolio sorts** (Q0=no activity, Q1-Q5 by factor score), equal-weight and cap-weight
- **Fama-French 3-factor regression** (`statsmodels.OLS` with `cov_type='HAC', cov_kwds={'maxlags': 3}` — Newey-West HAC standard errors)
- **Information Coefficient (IC)** analysis: Rank IC (Spearman) and Raw IC (Pearson) on rolling 12-month windows
- **Information Ratio (IR)**: rolling mean IC / rolling std IC

## Tech Stack

- Python 3 with pandas, numpy, matplotlib, scipy, statsmodels
- Jupyter notebooks as the primary execution environment
- Data in HDF5 (`.h5`) and CSV formats
- `pd.Period('M')` for monthly date alignment; `pd.to_period('M')` / `dt.to_period('M')` throughout

## Common Patterns

- Stock identifiers use the format `XXXX.HK` (e.g., `0005.HK` for HSBC)
- Buyback stock codes are converted from 5-digit strings: `.str.zfill(5).str[1:5] + '.HK'`
- Price data pivoted to `(date, sid)` matrices for factor calculations
- Monthly returns computed as `pct_change().shift(-1)` on end-of-month adjusted close prices
- `groupby('date_m', group_keys=True).apply(...)` with `group.name` for cross-sectional selection each month
- Pandas 3.0 compatibility: group keys no longer included as columns in apply results; use `group.name` instead
