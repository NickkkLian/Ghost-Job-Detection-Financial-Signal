# Ghost Job Detection: A Labor Market Mispricing Signal

**Nilabh Agrawal, Carter Jaquette, and Zixi Lian**  
Sauder School of Business, University of British Columbia  
COMM 486I — Applied AI in Finance

---

## Table of Contents

1. [Overview](#overview)
2. [Key Results](#key-results)
3. [Methodology](#methodology)
4. [Repository Structure](#repository-structure)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Scripts Reference](#scripts-reference)
8. [Data](#data)
9. [Citation](#citation)

---

## Overview

Ghost job postings — listings kept active by firms with no genuine intent to hire — are
a measurable but under-exploited signal in public equity markets. We construct a
**Ghost Job Intensity Score** from 310,821 Revelio Labs job postings spanning 2,919 US
public firms over 2009–2024 and show that cross-sectional variation in ghosting behaviour
predicts meaningful return dispersion.

**Pipeline summary:**

1. **Predict fill rates** — an XGBoost regressor trained on 14 behavioural and company
   features predicts each firm-quarter's next-quarter hiring inflow rate
   (Train R² = 0.947 · Val R² = 0.549 · Test R² = 0.486 · MAE = 3.1 pp).
2. **Flag ghosts** — firms with a predicted fill rate below the 21st-percentile threshold
   (0.2898, following Ng 2024) are classified as ghost posters; 43 of 204 out-of-sample
   firms (21.1%) are flagged.
3. **Score and rank** — a cross-sectional ghost intensity score is derived as
   `Ghost Score = 1 − Rank_t[PredFillRate(i,t)] / N_t`, so higher = more ghosting.
4. **Cluster** — a 7-component GMM (BIC-optimal) segments firms; Cluster 0 shows 1.6%
   ghost intensity (genuine hirers) while Clusters 3 and 4 reach 44–50%.
5. **Backtest** — a walk-forward backtest (2019–2026, 20 bps transaction costs) trades
   long the high-ghost and long the low-ghost legs simultaneously (Ghost Barbell).

---

## Key Results

### Strategy Performance (2019–2026, 20 bps TC)

| Strategy | Total Return | CAGR | Sharpe | Max Drawdown |
|---|---|---|---|---|
| **Ghost Barbell** | **+216.9%** | **+17.1%** | **0.84** | **−37.4%** |
| Long High Ghost | ~220% | — | — | — |
| Long Low Ghost | ~120% | — | — | — |
| SPY (benchmark) | +180.1% | +15.2% | 0.82 | −33.7% |
| Universe EW | +139.4% | — | — | — |
| Long / Short | −22.1% | — | −0.32 | −43.4% |

### Model Performance

| Model | Test R² | MAE |
|---|---|---|
| XGBoost (best iter = 185) | 0.486 | 3.1 pp |
| Ridge baseline | 0.279 | — |

### SHAP Feature Importance (top 3)

| Feature | Mean |SHAP| |
|---|---|
| `headcount` | 4.451 |
| `log_headcount` | 2.152 |
| `headcount_growth_4q` | 0.670 |

### GMM Clusters (k = 7, BIC = −1,508)

| Cluster | Ghost Rate |
|---|---|
| 0 | 1.6% (genuine hirers) |
| 3 | 44.2% |
| 4 | 50.0% |

---

## Methodology

### Data

- **Source:** Revelio Labs job postings (proprietary — see [Data](#data))
- **Universe:** 310,821 postings · 2,919 US public firms · 2009–2024
- **Qualifying firms:** 1,017 firms with ≥ 3 postings per quarter
- **Out-of-sample test set:** 204 firms

### Features (14 total)

| Group | Features |
|---|---|
| Posting behaviour (10) | Fill rate, posting volume, repost rate, and related metrics derived from job posting lifecycle |
| Company (4) | `headcount`, `log_headcount`, `headcount_growth_4q`, `naics_int` (industry code) |

### Ghost Threshold

Following Ng (2024), firms are classified as ghost posters when their predicted fill rate
falls below **0.2898** (21st percentile of the predicted distribution). This threshold
balances Type I / Type II error and aligns with the academic ghost-posting literature.

### Backtest Design

- **Period:** 2019-01-01 to 2026-01-01 (walk-forward, no lookahead)
- **Rebalancing:** Quarterly, using scores available ≥ 45 calendar days after quarter-end
- **Transaction costs:** 20 bps round-trip
- **Concentration:** Top / bottom 10% of scored universe per leg
- **Data-quality filters:** Daily return winsorisation at ±25%; volatility filter > 150%
  annualised; liquidity filter < $1M average daily volume; ticker identity cross-check
  against Yahoo Finance `longName`
- **Lookahead guarantee:** At every rebalancing date T, only scores with
  `available_date = quarter_end + lag_days ≤ T` are used

### Strategies Tested

| Strategy | Description |
|---|---|
| Ghost Barbell | Long high-ghost + long low-ghost simultaneously |
| Long High Ghost | Long top-decile ghost-intensity firms only |
| Long Low Ghost | Long bottom-decile ghost-intensity firms only |
| Universe EW | Equal-weight all scored firms |
| Long / Short | Long high-ghost, short low-ghost |
| SPY | Benchmark (buy-and-hold) |

---

## Repository Structure

```
ghost-job-detection/
│
├── README.md                          ← you are here
├── .gitignore
├── requirements.txt                   ← pip-installable dependencies
│
├── Ghost_Job_Report.pdf               ← final written report (5 body pp + appendix)
├── Ghost_Job_Presentation.pptx        ← slide deck (25 slides + 2 appendix)
│
├── notebooks/
│   └── ghost_job_detection.ipynb      ← end-to-end pipeline (EDA → model → GMM → scores)
│
├── scripts/
│   ├── ghost_backtest_v2.py           ← walk-forward backtest engine
│   └── ghost_concentration_compare_3.py  ← concentration sensitivity sweep
│
└── data/
    └── README.md                      ← data access instructions (data not included)
```

---

## Installation

Requires **Python 3.10+**.

```bash
git clone https://github.com/<your-username>/ghost-job-detection.git
cd ghost-job-detection
pip install -r requirements.txt
```

> **Colab users:** The notebook contains a `from google.colab import drive` cell for
> mounting Google Drive. This is pre-installed in Colab and requires no pip installation.
> If running locally, comment out that cell and set your data path directly.

---

## Usage

### Notebook

Open `notebooks/ghost_job_detection.ipynb` in Jupyter or Google Colab and run all cells
in order. The notebook covers:

1. Data loading and cleaning (Revelio Labs parquet files)
2. Exploratory data analysis and feature engineering
3. XGBoost model training, validation, and SHAP analysis
4. Ghost score construction and threshold classification
5. GMM clustering
6. Ghost score export (`ghost_scores_fq.parquet`) for use by the backtest scripts

### Backtest Script

```bash
python scripts/ghost_backtest_v2.py \
    --ghost_file  ghost_scores_fq.parquet \
    --start_date  2019-01-01 \
    --end_date    2026-01-01 \
    --lag_days    45 \
    --concentration 0.10 \
    --output_dir  backtest_output
```

Or import directly from another notebook:

```python
from scripts.ghost_backtest_v2 import main
nav, metrics = main(ghost_file='ghost_scores_fq.parquet', start_date='2019-01-01')
```

**Output files** (written to `--output_dir`):

| File | Contents |
|---|---|
| `backtest_nav.csv` | Daily NAV for each strategy |
| `backtest_metrics.csv` | Performance summary (return, CAGR, Sharpe, max DD) |
| `backtest_plot.png` | Combined performance chart |

### Concentration Sensitivity Script

Sweeps multiple concentration levels in a single run and produces a side-by-side
comparison chart:

```bash
python scripts/ghost_concentration_compare_3.py \
    --ghost_file     ghost_scores_fq.parquet \
    --start_date     2019-01-01 \
    --concentrations 0.05 0.10 0.25 \
    --tc_bps         20 \
    --output_dir     backtest_output
```

---

## Scripts Reference

### `ghost_backtest_v2.py`

Walk-forward backtest engine with hardened data-quality controls.

| Argument | Default | Description |
|---|---|---|
| `--ghost_file` | `ghost_scores_fq.parquet` | Input parquet with `ticker`, `quarter`, `ghost_score` columns |
| `--start_date` | `2019-01-01` | Backtest start date |
| `--end_date` | `2026-01-01` | Backtest end date |
| `--lag_days` | `45` | Publication lag — days after quarter-end before scores are usable |
| `--concentration` | `0.10` | Fraction of universe in each long/short leg |
| `--tc_bps` | `20` | Round-trip transaction cost in basis points |
| `--vol_filter` | `1.50` | Drop tickers with annualised volatility above this threshold |
| `--min_adv_usd` | `1000000` | Drop tickers with average daily volume below this ($USD) |
| `--output_dir` | `backtest_output` | Directory for output files and charts |

### `ghost_concentration_compare_3.py`

Sensitivity analysis across multiple concentration parameters.

| Argument | Default | Description |
|---|---|---|
| `--ghost_file` | `ghost_scores_fq.parquet` | Same input format as backtest script |
| `--concentrations` | `0.05 0.10 0.25` | Space-separated list of concentration levels to sweep |
| `--rebal_freq` | `Q` | Rebalancing frequency (`Q` = quarterly, `M` = monthly) |
| `--tc_bps` | `20` | Transaction cost in basis points |
| `--benchmark` | `SPY` | Benchmark ticker for comparison |
| `--no_identity_check` | flag | Skip Yahoo Finance ticker identity cross-check |
| `--output_dir` | `backtest_output` | Directory for output files and charts |

---

## Data

The Revelio Labs job postings dataset is **proprietary** and is not included in this
repository. See `data/README.md` for information on how to request access through
Revelio Labs' academic data programme.

The pipeline expects the following parquet files in your data directory:

| File | Description |
|---|---|
| `job_postings.parquet` | Raw posting-level data (ticker, date, fill rate, etc.) |
| `company_measures.parquet` | Firm-quarter panel (headcount, NAICS, etc.) |
| `ghost_scores_fq.parquet` | Output of the notebook — firm-quarter ghost scores (generated locally) |

---

## Citation

If you reference this work, please cite:

```
Agrawal, N., Jaquette, C., & Lian, Z. (2025). Ghost job detection: A labor market
mispricing signal. Unpublished manuscript, Sauder School of Business,
University of British Columbia.
```

---

*COMM 486I — Applied AI in Finance · UBC Sauder School of Business · Group 6*
