# Repository Knowledge — Avellaneda-Stoikov Market Making Replication

## Project Overview
Replication of Avellaneda & Stoikov (2008) finite-horizon market-making model.
Two experiments: theory-faithful spread (Exp1) and liquidity-only spread (Exp2).

## Key Formulas
- Reservation price: `r = S - q * gamma * sigma^2 * tau`
- Theory-faithful spread: `gamma*sigma^2*tau + (2/gamma)*ln(1 + gamma/k)`
- Liquidity-only spread: `(2/gamma)*ln(1 + gamma/k)` (constant, matches paper table)
- Execution intensity: `lambda(delta) = A * exp(-k * delta)`
- Fill probability: `min(lambda * dt, 1.0)` — clipped at 1.0

## Parameters
- S0=100, T=1, dt=0.005, sigma=2, A=140, k=1.5
- gamma in {0.01, 0.1, 0.5}, n_paths=1000, seed=42

## Repository Structure
- `src/simulation.py` — Core engine: ASParams, formulas, simulate_path, run_monte_carlo
- `src/exp1_theory_spread.py` — Experiment 1 runner and hypothesis checks
- `src/exp2_liquidity_spread.py` — Experiment 2 runner and hypothesis checks
- `src/visualization.py` — Matplotlib plots (histograms, path plots, bar charts)
- `tests/test_simulation.py` — Unit tests for simulation engine
- `tests/test_experiments.py` — Integration tests for both experiments
- `main.py` — Runs both experiments, saves results/RESULTS.md
- `results/` — All output figures and tables

## Implementation Choices
- Binomial price update (not Gaussian) per paper specification
- Independent Bernoulli fills for bid/ask (both can fill same step)
- Probability capping at 1.0 is documented as implementation necessity
- No common random numbers across strategies (separate seeded RNG per run)
- All formulas implemented from scratch (no library equivalent found via Context7)

## Testing
- Run: `pytest tests/ -v`
- Tests cover: formula correctness, path mechanics, MC reproducibility, edge cases
- Hypothesis tests embedded in exp1/exp2 modules

## Expected Spread Values (Exp2)
- gamma=0.01: ~1.33
- gamma=0.1: ~1.29
- gamma=0.5: ~1.15
