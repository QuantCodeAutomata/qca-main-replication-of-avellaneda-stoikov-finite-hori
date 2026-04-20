# Avellaneda-Stoikov Market Making Replication

Replication of the core finite-horizon market-making model from:

> Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book.*
> Quantitative Finance, 8(3), 217–224.

## Overview

This repository implements two experiments:

### Experiment 1: Theory-Faithful Time-Varying Spread
Uses the full spread formula from the paper's approximate finite-horizon solution:
```
spread(t) = gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/k)
```

### Experiment 2: Liquidity-Only (Table-Matching) Spread
Resolves the paper's spread ambiguity by using only the liquidity component:
```
spread = (2/gamma) * ln(1 + gamma/k)
```
This constant spread matches the paper's reported table values (~1.33, ~1.29, ~1.15 for gamma = 0.01, 0.1, 0.5).

## Model Parameters

| Parameter | Value |
|-----------|-------|
| S0 | 100 |
| T | 1 |
| dt | 0.005 (200 steps) |
| sigma | 2 |
| A | 140 |
| k | 1.5 |
| gamma | {0.01, 0.1, 0.5} |
| n_paths | 1000 per strategy per gamma |

## Key Formulas

- **Reservation price:** `r(s,q,t) = s - q * gamma * sigma^2 * (T-t)`
- **Inventory-aware quotes:** `p_a = r + 0.5*spread`, `p_b = r - 0.5*spread`
- **Symmetric benchmark:** `p_a = S + 0.5*spread`, `p_b = S - 0.5*spread`
- **Execution intensity:** `lambda(delta) = A * exp(-k * delta)`
- **Fill probability:** `p_fill = min(lambda * dt, 1.0)` (clipped)
- **Terminal P&L:** `Pi_T = X_T + q_T * S_T`

## Repository Structure

```
├── src/
│   ├── simulation.py        # Core AS model formulas and Monte Carlo engine
│   ├── exp1_theory_spread.py # Experiment 1 implementation
│   ├── exp2_liquidity_spread.py # Experiment 2 implementation
│   └── visualization.py     # Plots and tables
├── tests/
│   ├── test_simulation.py   # Unit tests for simulation engine
│   └── test_experiments.py  # Integration tests for experiments
├── results/                 # Generated figures and RESULTS.md
├── main.py                  # Main runner
└── requirements.txt
```

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# Run all experiments
python main.py

# Run tests
pytest tests/ -v
```

## Results

All results are saved to `results/`:
- `RESULTS.md` — Summary tables, hypothesis tests, and metrics
- `hist_profit_*.png` — Terminal profit histograms
- `path_*.png` — Illustrative path plots
- `gamma_comparison_*.png` — Risk metrics by gamma
- `summary_*.csv` / `summary_*.md` — Summary statistics tables

## Implementation Notes

- **Price update:** Binomial `±sigma*sqrt(dt)` (not Gaussian), as specified in the paper.
- **Fill model:** Independent Bernoulli draws for bid and ask sides; both can fill in the same step.
- **Probability capping:** `p_fill = lambda*dt` is clipped to `[0,1]` when delta is small/negative.
- **Seed policy:** Fixed global seed (42); separate RNG per strategy run; no common random numbers across strategies.
- **No market microstructure:** No tick size, fees, rebates, or inventory limits.
