# WorldQuant IQC Alpha Research

Alpha research toolkit for WorldQuant's [International Quant Championship (IQC)](https://www.worldquant.com/brain/iqc/).

## Overview

This repo provides a systematic workflow for discovering, testing, and submitting alpha factors to WorldQuant Brain. The full pipeline:

```
Explore data catalog  →  Discover signal  →  Local sanity check  →  Brain simulation  →  Analyse results
   (notebook 01)          (notebook 06)         (notebook 07)       (notebooks 02–05)    (notebook 08)
```

**Target metric:** `fitness = sqrt(abs(returns) / max(turnover, 0.125)) * sharpe`, with Sharpe ≥ 1.25.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
# Edit .env with your WorldQuant Brain username and password
# Get credentials at: https://platform.worldquantbrain.com
```

### 3. Verify your setup

Open and run `notebooks/00_setup.ipynb` — it checks your `.env` and confirms API access.

---

## Notebook Guide

| Notebook | Purpose | Run order |
|----------|---------|-----------|
| `00_setup.ipynb` | Auth check — run this first | 1st |
| `01_datafields.ipynb` | Download + explore 120k+ Brain data fields | 2nd |
| `02_first_alpha.ipynb` | Submit your first alpha, understand result metrics | 3rd |
| `03_momentum.ipynb` | Systematic momentum sweep (ts_rank, ts_zscore) | Research |
| `04_reversion.ipynb` | Mean-reversion sweep (-ts_zscore, ts_mean - close) | Research |
| `05_fundamental.ipynb` | Value + quality alphas from accounting data | Research |
| `06_signal_discovery.ipynb` | Statistical + ML-driven signal discovery | Research |
| `07_local_backtest.ipynb` | Fast sanity check with yfinance before Brain submission | Before submitting |
| `08_results.ipynb` | Results dashboard — rank, visualise, export | After simulations |

---

## Project Structure

```
wq/
  client.py       — Brain API: auth, fetch data fields, submit/poll simulations
  alpha.py        — AlphaConfig dataclass, fitness formula, alpha templates
  utils.py        — Results I/O, catalog search, local signal test

notebooks/        — Research notebooks (see table above)
alphas/           — YAML alpha libraries (momentum, reversion, fundamental)
data/             — Cache dir: datafields.parquet, results.csv (gitignored)
```

---

## Key Concepts

### Fitness formula

```python
fitness = sqrt(abs(returns) / max(turnover, 0.125)) * sharpe
```

- High Sharpe + moderate returns + low turnover → high fitness
- Sharpe ≥ 1.25 is the minimum bar for submission

### Common Brain operators

| Operator | What it does |
|----------|-------------|
| `ts_rank(x, n)` | Percentile rank of x within the past n days (0–1) |
| `ts_zscore(x, n)` | z-score of x over past n days |
| `ts_corr(x, y, n)` | Rolling n-day correlation |
| `ts_mean(x, n)` | Rolling n-day mean |
| `ts_delta(x, n)` | x today minus x n days ago |
| `rank(x)` | Cross-sectional percentile rank across all stocks |
| `group_rank(x, g)` | Cross-sectional rank within group (industry/subindustry) |

### Alpha settings that matter most

- **`neutralization`**: `Subindustry` removes sector bets — always use for fundamental alphas
- **`decay`**: smooths the signal over time; 0 for technical, 10–20 for fundamental
- **`universe`**: `TOP3000` gives more stocks → higher fitness; `TOP500` for large-cap focus
- **`delay`**: always use `1` to avoid look-ahead bias

---

## Quick start: submit an alpha

```python
from wq.client import get_client, simulate
from wq.alpha import AlphaConfig

client = get_client()

config = AlphaConfig(
    expression="ts_rank(close, 20)",
    universe="TOP3000",
    neutralization="Subindustry",
    tags=["momentum"],
    notes="20-day price momentum baseline",
)

result = simulate(client, config)
print(result)
```

---

## Local sanity check before submitting

```python
from wq.utils import local_signal_test
import yfinance as yf

# Compute your signal as a DataFrame (dates × stocks)
data = yf.download([...], ...)
signal = data['Close'].rolling(20).apply(lambda x: x.rank().iloc[-1] / len(x), raw=False)
returns = data['Close'].pct_change()

result = local_signal_test(signal, returns, forward_days=5)
print(result)
# → {'mean_ic': 0.025, 'ic_tstat': 3.2, 'ls_sharpe': 0.91, 'verdict': 'SUBMIT'}
```

If `verdict == 'SUBMIT'` → proceed to Brain simulation.
