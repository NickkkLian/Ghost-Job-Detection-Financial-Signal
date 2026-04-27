# Ghost Job Detection: A Labor Market Mispricing Signal

**Carter Jaquette, Nilabh Agrawal, and Zixi Lian**  
Sauder School of Business, University of British Columbia  
COMM 486I — Applied AI in Finance

---

## Overview
We construct a ghost job intensity signal from 310,821 Revelio Labs 
job postings across 2,919 US public firms (2009–2024). An XGBoost 
regressor predicts next-quarter hiring inflows; the predicted fill 
rate is converted into a cross-sectional ghost intensity score. 
A walk-forward backtest (2019–2026) shows the Ghost Barbell strategy 
returns +216.9% (Sharpe 0.84) vs. SPY +180.1% (Sharpe 0.82).

## Key Results
| Strategy | Total Return | Sharpe | Max DD |
|---|---|---|---|
| Ghost Barbell | +216.9% | 0.84 | −37.4% |
| Long High Ghost | ~220% | — | — |
| SPY (benchmark) | +180.1% | 0.82 | −33.7% |
| Long / Short | −22.1% | −0.32 | −43.4% |

## Repository Structure
- `notebooks/` — main analysis notebook
- `report/` — final written report (PDF)
- `presentation/` — slide deck
- `data/` — data access instructions (data not included, proprietary)
- `outputs/figures/` — all generated charts

## Requirements
```bash
pip install -r requirements.txt
```

## Data
Proprietary — see `data/README.md` for access instructions.
