# Avellaneda-Stoikov Market Making Replication — Results

## Parameter Sheet

| Parameter | Value |
|-----------|-------|
| S0 | 100 |
| T | 1 |
| dt | 0.005 (200 steps) |
| sigma | 2 |
| A | 140 |
| k | 1.5 |
| q0 | 0 |
| X0 | 0 |
| gamma values | {0.01, 0.1, 0.5} |
| n_paths | 1000 per strategy per gamma |
| Price update | Binomial ±sigma*sqrt(dt) |
| Fill probability | lambda*dt clipped to [0,1] |
| Fill model | Independent Bernoulli (both sides can fill same step) |
| Random seed | 42 (fixed global, separate RNG per strategy run) |
| Common random numbers | Not used across strategies |

---

## Experiment 1: Theory-Faithful Time-Varying Spread

**Spread formula:** `spread(t) = gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/k)`

| Strategy        | Spread Convention              |   Quoted Spread |   Mean Profit |   Std Profit |   Mean Final q |   Std Final q |   gamma |
|:----------------|:-------------------------------|----------------:|--------------:|-------------:|---------------:|--------------:|--------:|
| Inventory-Aware | Theory-Faithful (time-varying) |          1.3689 |       68.451  |       8.9404 |          0.029 |        5.0723 |    0.01 |
| Symmetric       | Theory-Faithful (time-varying) |          1.3689 |       69.1129 |      13.48   |         -0.047 |        8.6125 |    0.01 |
| Inventory-Aware | Theory-Faithful (time-varying) |          1.6908 |       64.7883 |       6.6507 |          0.096 |        2.9092 |    0.1  |
| Symmetric       | Theory-Faithful (time-varying) |          1.6908 |       68.2471 |      13.3808 |         -0.025 |        8.2873 |    0.1  |
| Inventory-Aware | Theory-Faithful (time-varying) |          3.1507 |       48.3941 |       5.8447 |          0.107 |        1.9394 |    0.5  |
| Symmetric       | Theory-Faithful (time-varying) |          3.1507 |       58.336  |      11.6023 |          0.043 |        7.1627 |    0.5  |

### Hypothesis Tests — Experiment 1

| Hypothesis | Result |
|------------|--------|
| H1_std_profit_lower_gamma0.01 | PASS ✓ |
| H2_std_q_lower_gamma0.01 | PASS ✓ |
| H1_std_profit_lower_gamma0.1 | PASS ✓ |
| H2_std_q_lower_gamma0.1 | PASS ✓ |
| H1_std_profit_lower_gamma0.5 | PASS ✓ |
| H2_std_q_lower_gamma0.5 | PASS ✓ |
| H3_convergence_gamma_0.01 | PASS ✓ |
| H4_higher_gamma_lower_profit | PASS ✓ |
| H5_higher_gamma_lower_std_profit | PASS ✓ |

**Interpretation:**
- Inventory-aware strategy has lower std(Profit) than symmetric: YES
- Inventory-aware strategy has lower std(Final q) than symmetric: YES
- Strategies converge for gamma=0.01: YES
- Higher gamma reduces mean profit: YES

---

## Experiment 2: Liquidity-Only (Table-Matching) Spread

**Spread formula:** `spread = (2/gamma)*ln(1 + gamma/k)` (constant in time)

### Spread Value Verification vs Paper

| gamma | Computed Spread | Paper Value | Abs Diff | Match |
|-------|----------------|-------------|----------|-------|
| 0.01 | 1.3289 | 1.33 | 0.0011 | ✓ |
| 0.1 | 1.2908 | 1.29 | 0.0008 | ✓ |
| 0.5 | 1.1507 | 1.15 | 0.0007 | ✓ |

| Strategy        | Spread Convention         |   Quoted Spread |   Mean Profit |   Std Profit |   Mean Final q |   Std Final q |   gamma |
|:----------------|:--------------------------|----------------:|--------------:|-------------:|---------------:|--------------:|--------:|
| Inventory-Aware | Liquidity-Only (constant) |          1.3289 |       68.4679 |       8.8872 |          0.072 |        5.0871 |    0.01 |
| Symmetric       | Liquidity-Only (constant) |          1.3289 |       69.0492 |      13.6577 |         -0.111 |        8.6532 |    0.01 |
| Inventory-Aware | Liquidity-Only (constant) |          1.2908 |       64.2324 |       6.0996 |          0.098 |        2.9187 |    0.1  |
| Symmetric       | Liquidity-Only (constant) |          1.2908 |       68.881  |      13.6351 |         -0.018 |        8.709  |    0.1  |
| Inventory-Aware | Liquidity-Only (constant) |          1.1507 |       23.9319 |       5.052  |          0.07  |        1.93   |    0.5  |
| Symmetric       | Liquidity-Only (constant) |          1.1507 |       68.289  |      13.8164 |          0.124 |        9.1672 |    0.5  |

### Comparison to Paper's Reported Values

| gamma | Strategy | Simulated Profit | Paper Profit | Diff |
|-------|----------|-----------------|--------------|------|
| 0.01 | Inventory-Aware | 68.47 | 66.78 | +1.69 |
| 0.01 | Symmetric | 69.05 | 67.36 | +1.69 |
| 0.1 | Inventory-Aware | 64.23 | 62.94 | +1.29 |
| 0.1 | Symmetric | 68.88 | 67.21 | +1.67 |
| 0.5 | Inventory-Aware | 23.93 | 33.92 | -9.99 |
| 0.5 | Symmetric | 68.29 | 66.2 | +2.09 |

### Hypothesis Tests — Experiment 2

| Hypothesis | Result |
|------------|--------|
| H1_spread_matches_paper_gamma0.01 | PASS ✓ |
| H1_spread_matches_paper_gamma0.1 | PASS ✓ |
| H1_spread_matches_paper_gamma0.5 | PASS ✓ |
| H2_std_profit_lower_gamma0.01 | PASS ✓ |
| H3_std_q_lower_gamma0.01 | PASS ✓ |
| H2_std_profit_lower_gamma0.1 | PASS ✓ |
| H3_std_q_lower_gamma0.1 | PASS ✓ |
| H2_std_profit_lower_gamma0.5 | PASS ✓ |
| H3_std_q_lower_gamma0.5 | PASS ✓ |
| H4_convergence_gamma_0.01 | PASS ✓ |
| H5_higher_gamma_lower_profit | PASS ✓ |
| H6_gamma01_profit_near_paper | PASS ✓ |

---

## Generated Figures

### Experiment 1 (Theory-Faithful Spread)
- `hist_profit_exp1_theory_spread_gamma0.01.png` — Profit histograms
- `path_exp1_theory_spread_gamma0.01.png` — Illustrative path
- `hist_profit_exp1_theory_spread_gamma0.1.png` — Profit histograms
- `path_exp1_theory_spread_gamma0.1.png` — Illustrative path
- `hist_profit_exp1_theory_spread_gamma0.5.png` — Profit histograms
- `path_exp1_theory_spread_gamma0.5.png` — Illustrative path
- `gamma_comparison_exp1_theory_spread.png` — Risk metrics by gamma

### Experiment 2 (Liquidity-Only Spread)
- `hist_profit_exp2_liquidity_spread_gamma0.01.png` — Profit histograms
- `path_exp2_liquidity_spread_gamma0.01.png` — Illustrative path
- `hist_profit_exp2_liquidity_spread_gamma0.1.png` — Profit histograms
- `path_exp2_liquidity_spread_gamma0.1.png` — Illustrative path
- `hist_profit_exp2_liquidity_spread_gamma0.5.png` — Profit histograms
- `path_exp2_liquidity_spread_gamma0.5.png` — Illustrative path
- `gamma_comparison_exp2_liquidity_spread.png` — Risk metrics by gamma
