"""
Experiment 2: Liquidity-Only (Table-Matching) Spread Replication.

Resolves the paper's spread ambiguity by using only the liquidity component:

    spread = (2/gamma) * ln(1 + gamma/k)

This constant-in-time spread matches the quoted spread values reported in the
paper's results table:
  - gamma=0.01 → spread ≈ 1.33
  - gamma=0.1  → spread ≈ 1.29
  - gamma=0.5  → spread ≈ 1.15

The reservation-price logic and execution model are identical to Experiment 1;
only the spread formula changes.

Research objectives:
  - Verify spread values match paper's reported values.
  - Retest main qualitative hypotheses under this alternative convention.
  - Assess whether this convention better reproduces the paper's profit/inventory
    statistics (inventory profit ≈ 62.94 vs symmetric ≈ 67.21 for gamma=0.1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.simulation import (
    ASParams,
    run_monte_carlo,
    compute_summary,
    spread_liquidity_only,
    validate_reservation_price_properties,
)
from src.visualization import (
    plot_profit_histograms,
    plot_illustrative_path,
    save_summary_table,
    plot_gamma_comparison,
)

# Experiment identifier
EXPERIMENT_LABEL = "exp2_liquidity_spread"

# Paper parameters (same as Exp1)
BASE_PARAMS = dict(
    S0=100.0,
    T=1.0,
    dt=0.005,
    sigma=2.0,
    A=140.0,
    k=1.5,
    n_paths=1000,
    X0=0.0,
    q0=0,
    seed=42,
)

GAMMA_VALUES = [0.01, 0.1, 0.5]

# Paper-reported spread values for verification
PAPER_SPREAD_VALUES = {0.01: 1.33, 0.1: 1.29, 0.5: 1.15}

# Paper-reported profit values for comparison
PAPER_PROFITS = {
    0.01: {"inventory": 66.78, "symmetric": 67.36},
    0.1:  {"inventory": 62.94, "symmetric": 67.21},
    0.5:  {"inventory": 33.92, "symmetric": 66.20},
}


def verify_spread_values(gamma_values: list[float], k: float) -> dict[float, dict]:
    """
    Analytically verify that the liquidity-only spread matches the paper's
    reported values.

    Parameters
    ----------
    gamma_values : list[float]
        List of gamma values to check.
    k : float
        Intensity sensitivity parameter.

    Returns
    -------
    dict[float, dict]
        For each gamma: computed spread, paper value, and absolute difference.
    """
    verification: dict[float, dict] = {}
    for gamma in gamma_values:
        computed = spread_liquidity_only(gamma, k)
        paper_val = PAPER_SPREAD_VALUES.get(gamma, float("nan"))
        diff = abs(computed - paper_val)
        verification[gamma] = {
            "computed_spread": round(computed, 4),
            "paper_spread": paper_val,
            "abs_diff": round(diff, 4),
            "matches": diff < 0.05,  # within 5 basis points
        }
    return verification


def run_experiment_2(output_dir: Path) -> tuple[list[dict], dict[str, Any]]:
    """
    Run Experiment 2: liquidity-only (table-matching) spread.

    For each gamma in {0.01, 0.1, 0.5}, simulate 1000 paths for both the
    inventory-aware and symmetric strategies using the constant liquidity-only
    spread formula.

    Parameters
    ----------
    output_dir : Path
        Directory where results (figures, tables) will be saved.

    Returns
    -------
    tuple[list[dict], dict[str, Any]]
        (summary_rows, all_mc_results).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    all_mc_results: dict[str, Any] = {}

    # Verify spread values match paper
    spread_verification = verify_spread_values(GAMMA_VALUES, k=BASE_PARAMS["k"])
    print("\n[Exp2] Spread verification against paper values:")
    for gamma, info in spread_verification.items():
        match_str = "✓ MATCH" if info["matches"] else "✗ MISMATCH"
        print(
            f"  gamma={gamma}: computed={info['computed_spread']:.4f}  "
            f"paper≈{info['paper_spread']}  diff={info['abs_diff']:.4f}  {match_str}"
        )

    for gamma in GAMMA_VALUES:
        params = ASParams(**{**BASE_PARAMS, "gamma": gamma})

        # Validate theoretical properties
        validate_reservation_price_properties(params)

        spread_val = spread_liquidity_only(gamma, params.k)
        print(f"\n[Exp2] gamma={gamma}  spread(constant)={spread_val:.4f}")

        for strategy in ("inventory_aware", "symmetric"):
            label = "Inventory-Aware" if strategy == "inventory_aware" else "Symmetric"
            print(f"  Running {label} strategy ({params.n_paths} paths)...")

            mc = run_monte_carlo(params, "liquidity_only", strategy)
            all_mc_results[(gamma, strategy)] = mc

            row = compute_summary(
                mc,
                strategy_label=label,
                spread_label="Liquidity-Only (constant)",
                spread_value=spread_val,
            )
            row["gamma"] = gamma
            summary_rows.append(row)

            # Compare to paper values
            paper_profit = PAPER_PROFITS.get(gamma, {}).get(
                "inventory" if strategy == "inventory_aware" else "symmetric", None
            )
            paper_str = f"  paper≈{paper_profit}" if paper_profit else ""
            print(
                f"    Mean Profit={row['Mean Profit']:.2f}{paper_str}  "
                f"Std Profit={row['Std Profit']:.2f}  "
                f"Mean q={row['Mean Final q']:.3f}  "
                f"Std q={row['Std Final q']:.3f}"
            )

        # Profit histograms
        plot_profit_histograms(
            all_mc_results[(gamma, "inventory_aware")],
            all_mc_results[(gamma, "symmetric")],
            gamma=gamma,
            experiment_label=EXPERIMENT_LABEL,
            output_dir=output_dir,
        )

        # Illustrative path
        plot_illustrative_path(
            all_mc_results[(gamma, "inventory_aware")]["illustrative_path"],
            gamma=gamma,
            experiment_label=EXPERIMENT_LABEL,
            output_dir=output_dir,
        )

    # Summary table
    save_summary_table(summary_rows, EXPERIMENT_LABEL, output_dir)

    # Gamma comparison bar chart
    plot_gamma_comparison(summary_rows, EXPERIMENT_LABEL, output_dir)

    return summary_rows, all_mc_results


def check_hypotheses_exp2(
    summary_rows: list[dict],
    spread_verification: dict[float, dict],
) -> dict[str, bool]:
    """
    Evaluate Experiment 2 hypotheses.

    Hypotheses:
    H1: Spread values match paper's reported values (within tolerance).
    H2: For each gamma, std(Profit) of inventory-aware < std(Profit) of symmetric.
    H3: For each gamma, std(Final q) of inventory-aware < std(Final q) of symmetric.
    H4: For gamma=0.01, strategies are relatively similar.
    H5: Higher gamma → lower mean profit for inventory-aware.
    H6: Mean profit for gamma=0.1 inventory-aware is in reasonable range of paper's 62.94.

    Parameters
    ----------
    summary_rows : list[dict]
        Summary statistics from run_experiment_2.
    spread_verification : dict[float, dict]
        Spread verification results from verify_spread_values.

    Returns
    -------
    dict[str, bool]
        Hypothesis test results.
    """
    df = pd.DataFrame(summary_rows)
    results: dict[str, bool] = {}

    # H1: spread values match paper
    for gamma in GAMMA_VALUES:
        results[f"H1_spread_matches_paper_gamma{gamma}"] = bool(spread_verification[gamma]["matches"])

    # H2 and H3: per-gamma comparisons
    for gamma in GAMMA_VALUES:
        inv = df[(df["gamma"] == gamma) & (df["Strategy"] == "Inventory-Aware")].iloc[0]
        sym = df[(df["gamma"] == gamma) & (df["Strategy"] == "Symmetric")].iloc[0]

        results[f"H2_std_profit_lower_gamma{gamma}"] = bool(inv["Std Profit"] < sym["Std Profit"])
        results[f"H3_std_q_lower_gamma{gamma}"] = bool(inv["Std Final q"] < sym["Std Final q"])

    # H4: convergence for small gamma
    # The strategies converge as gamma→0: ratio std(inv)/std(sym) should be
    # closer to 1 for gamma=0.01 than for gamma=0.5 (higher ratio = more similar).
    inv_001 = df[(df["gamma"] == 0.01) & (df["Strategy"] == "Inventory-Aware")].iloc[0]
    sym_001 = df[(df["gamma"] == 0.01) & (df["Strategy"] == "Symmetric")].iloc[0]
    inv_05 = df[(df["gamma"] == 0.5) & (df["Strategy"] == "Inventory-Aware")].iloc[0]
    sym_05 = df[(df["gamma"] == 0.5) & (df["Strategy"] == "Symmetric")].iloc[0]
    ratio_001 = inv_001["Std Profit"] / sym_001["Std Profit"] if sym_001["Std Profit"] > 0 else np.inf
    ratio_05 = inv_05["Std Profit"] / sym_05["Std Profit"] if sym_05["Std Profit"] > 0 else np.inf
    results["H4_convergence_gamma_0.01"] = bool(ratio_001 > ratio_05)

    # H5: higher gamma → lower mean profit for inventory-aware
    inv_profits = {
        g: df[(df["gamma"] == g) & (df["Strategy"] == "Inventory-Aware")].iloc[0]["Mean Profit"]
        for g in GAMMA_VALUES
    }
    results["H5_higher_gamma_lower_profit"] = bool(inv_profits[0.01] > inv_profits[0.5])

    # H6: gamma=0.1 inventory profit in reasonable range of paper's 62.94
    inv_01_profit = inv_profits[0.1]
    results["H6_gamma01_profit_near_paper"] = bool(abs(inv_01_profit - 62.94) < 20.0)

    return results
