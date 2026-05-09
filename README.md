# CUHK-Shenzhen DDA4080: Shareholder Yield Strategy

Multi-factor stock selection strategy research on the Hong Kong stock market (HSCI constituents), developed for DDA4080 Financial Engineering at CUHK-Shenzhen. The strategy combines **dividend yield**, **buyback yield**, and **low-volatility filtering** to construct a shareholder yield portfolio, with extensive robustness analysis including industry/size neutrality, Fama-French 3-factor regression, and tracking error decomposition.

## Strategy Overview

### Core Idea

The **Shareholder Yield** factor captures total cash returned to shareholders through both dividends and share buybacks:

$$\text{ShareholderYield} = \underbrace{\frac{\text{3-Year Avg Monthly DPS}}{\text{Current Price}}}_{\text{Dividend Yield}} + \underbrace{\frac{\text{Total 3-Year Buyback} / 3}{\text{Current Market Cap}}}_{\text{Buyback Yield}}$$

### Pipeline (4 stages per rebalance)

1. **Universe filter** — HSCI constituents with dividend/buyback activity in past 36 months; ADTV & market cap both in top 80%
2. **Factor scoring** — Compute Shareholder Yield = DivYield + BuybackYield
3. **Outlier trim + selection** — Remove top 1% extreme scores; keep top 20% by score
4. **Low-vol filter** — Select 10–20 stocks with lowest 1-year annualized volatility

**Configuration**: Quarterly rebalance, equal-weight or capped market-cap weight (10% individual cap), backtest from 2012-01.

## Key Results

### Strategy Performance (Equal-Weight, 2012–2025)

| Strategy | Ann. Return | Ann. Vol | Sharpe | Max DD |
|----------|------------|----------|--------|--------|
| **C: Shareholder Yield + Trim + Low Vol** | 11.40% | 16.24% | **0.70** | **-27.69%** |
| B: Shareholder Yield + Trim only | 14.06% | 22.26% | 0.63 | -35.19% |
| D: Shareholder Yield + Trim + FCFF | 13.45% | 21.64% | 0.62 | -36.26% |
| A: Raw SY (no trim) | 5.78% | 30.25% | 0.19 | -74.49% |
| HSI Benchmark | 3.49% | 20.05% | 0.17 | -55.34% |
| HK Connect High Div | 9.05% | 24.76% | 0.37 | -50.94% |

> **Key finding**: Extreme value trimming is essential — without it, the strategy collapses to 0.19 Sharpe and -74% max drawdown as "yield traps" (small caps with inflated yields from price crashes) dominate the portfolio.

### Robustness Analyses (v4)

- **Industry neutrality** — HSICS sector-level neutralization; strategy alpha survives industry adjustment
- **Size neutrality** — Combined industry + size neutralization
- **Fama-French 3-factor** — Newey-West HAC regressions on MKT/SMB/HML
- **Tracking error** — Decomposition into systematic vs. idiosyncratic components; N-stock sensitivity

## Repository Structure

```
├── data/                          # Market data (small CSVs only; large files excluded)
│   ├── HSCI.csv                   # HSCI index constituents (monthly)
│   ├── hk_ff3_factors.csv         # HK Fama-French 3 factors
│   ├── hk_risk_free_rate.csv      # Risk-free rate
│   ├── hk_bm_monthly.csv          # Book-to-market ratios
│   ├── em_buyback_filtered.csv    # Buyback records
│   ├── hk_industry_map_v2.json    # HSICS industry mapping
│   └── ...
├── Final/code/                    # Clean, final notebooks
│   ├── data_1.ipynb → data_4.ipynb   # Data pipeline
│   ├── factor_1.ipynb, factor_2.ipynb # Factor construction & FF3 regression
│   ├── factor_回购.ipynb              # v1: Pure buyback factor
│   └── v3_综合收益率.ipynb            # v3: Final shareholder yield strategy
├── doc/                           # Research reports (LaTeX, PDF, Markdown)
├── pict/                          # Generated charts
├── v2_红利低波.ipynb               # v2: Dividend + low-vol strategy
├── v3_综合收益率.ipynb             # v3: Working notebook
├── v4_极端值论证_基线对比.ipynb     # v4: Extreme value justification & baselines
├── *_neutral*.py                  # Neutrality analysis scripts
├── te_analysis*.py                 # Tracking error analysis
└── analyze_worst_extreme*.py      # Extreme stock diagnosis
```

## Validation Methodology

- **Quintile portfolio sorts** (Q0=no activity, Q1–Q5 by factor score), equal-weight and cap-weight
- **Fama-French 3-factor regression** with Newey-West HAC standard errors (`statsmodels.OLS`, `maxlags=3`)
- **Information Coefficient (IC)** — Rank IC (Spearman) and Raw IC (Pearson) on rolling 12-month windows
- **Information Ratio (IR)** — Rolling mean IC / rolling std IC

## Setup

```bash
pip install pandas numpy matplotlib scipy statsmodels jupyter
```

### Data

Data is sourced from the **RiceQuant API**. Large data files are excluded from this repository due to GitHub size limits (available upon request or via API regeneration):

| File | Size | Content |
|------|------|---------|
| `hk_price.csv` | ~600 MB | Daily OHLCV + adjusted close |
| `hk_dividendyield.h5` | ~170 MB | Monthly dividend yield |
| `hk_shares.h5` | ~170 MB | Monthly shares outstanding |

datalink:https://cuhko365-my.sharepoint.com/:f:/g/personal/123090233_link_cuhk_edu_cn/IgBHRwntAO7cRIBZtWRhcMurAdhq4vZs-Cs88Jf0TGcWclk?e=wixvCV
## Tech Stack

Python 3 · pandas · numpy · matplotlib · scipy · statsmodels · Jupyter notebooks

## Course Info

- **Course**: DDA4080 — Financial Engineering
- **Institution**: CUHK-Shenzhen (The Chinese University of Hong Kong, Shenzhen)
