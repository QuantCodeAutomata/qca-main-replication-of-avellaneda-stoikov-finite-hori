"""
Experiment 1: Theory-Faithful Time-Varying Spread Replication.

Replicates the Avellaneda-Stoikov (2008) finite-horizon market-making model
using the full time-varying spread:

    spread(t) = gamma * sigma^2 * tau + (2/gamma) * ln(1 + gamma/k)

where tau = T - t.

Compares the inventory-aware reservation-price strategy against the symmetric
benchmark centered on the mid-price.

Research objectives:
  - Lower std(Profit) for inventory-aware vs symmetric for each gamma.
  - Lower std(Final q) for inventory-aware vs symmetric for each gamma.
  - Convergence between strategies as gamma → 0.
  - Higher gamma → lower mean profit but tighter risk control.
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
    spread_theory_faithful,
    validate_reservation_price_properties,
)
from src.visualization import (
    plot_profit_histograms,
    plot_illustrative_path,
    save_summary_table,
    plot_gamma_comparison,
)

# Experiment identifier
EXPERIMENT_LABEL = "exp1_theory_spread"

# Paper parameters (fixed across gamma values)
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


def run_experiment_1(output_dir: Path) -> tuple[list[dict], dict[str, Any]]:
    """
    Run Experiment 1: theory-faithful time-varying spread.

    For each gamma in {0.01, 0.1, 0.5}, simulate 1000 paths for both the
    inventory-aware and symmetric strategies using the full spread formula.

    Parameters
    ----------
    output_dir : Path
        Directory where results (figures, tables) will be saved.

    Returns
    -------
    tuple[list[dict], dict[str, Any]]
        (summary_rows, all_mc_results) where summary_rows is a list of
        summary statistics dicts and all_mc_results stores raw MC outputs
        keyed by (gamma, strategy).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    all_mc_results: dict[str, Any] = {}

    for gamma in GAMMA_VALUES:
        params = ASParams(**{**BASE_PARAMS, "gamma": gamma})

        # Validate theoretical properties for this gamma
        validate_reservation_price_properties(params)

        # Representative spread at t=0 (tau=T)
        spread_at_t0 = spread_theory_faithful(gamma, params.sigma, params.k, tau=params.T)

        print(f"\n[Exp1] gamma={gamma}  spread(t=0)={spread_at_t0:.4f}")

        for strategy in ("inventory_aware", "symmetric"):
            label = "Inventory-Aware" if strategy == "inventory_aware" else "Symmetric"
            print(f"  Running {label} strategy ({params.n_paths} paths)...")

            mc = run_monte_carlo(params, "theory_faithful", strategy)
            all_mc_results[(gamma, strategy)] = mc

            row = compute_summary(
                mc,
                strategy_label=label,
                spread_label="Theory-Faithful (time-varying)",
                spread_value=spread_at_t0,
            )
            row["gamma"] = gamma
            summary_rows.append(row)

            print(
                f"    Mean Profit={row['Mean Profit']:.2f}  "
                f"Std Profit={row['Std Profit']:.2f}  "
                f"Mean q={row['Mean Final q']:.3f}  "
                f"Std q={row['Std Final q']:.3f}"
            )

        # Profit histograms for this gamma
        plot_profit_histograms(
            all_mc_results[(gamma, "inventory_aware")],
            all_mc_results[(gamma, "symmetric")],
            gamma=gamma,
            experiment_label=EXPERIMENT_LABEL,
            output_dir=output_dir,
        )

        # Illustrative path for inventory-aware strategy
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


def check_hypotheses_exp1(summary_rows: list[dict]) -> dict[str, bool]:
    """
    Evaluate the experiment's research hypotheses against the simulation results.

    Hypotheses:
    H1: For each gamma, std(Profit) of inventory-aware < std(Profit) of symmetric.
    H2: For each gamma, std(Final q) of inventory-aware < std(Final q) of symmetric.
    H3: For gamma=0.01, the two strategies are relatively similar
        (std(Profit) ratio close to 1).
    H4: Higher gamma → lower mean profit for inventory-aware strategy.
    H5: Higher gamma → lower std(Profit) for inventory-aware strategy.

    Parameters
    ----------
    summary_rows : list[dict]
        Summary statistics from run_experiment_1.

    Returns
    -------
    dict[str, bool]
        Hypothesis test results.
    """
    df = pd.DataFrame(summary_rows)
    results: dict[str, bool] = {}

    # H1 and H2: per-gamma comparisons
    for gamma in GAMMA_VALUES:
        inv = df[(df["gamma"] == gamma) & (df["Strategy"] == "Inventory-Aware")].iloc[0]
        sym = df[(df["gamma"] == gamma) & (df["Strategy"] == "Symmetric")].iloc[0]

        h1 = bool(inv["Std Profit"] < sym["Std Profit"])
        h2 = bool(inv["Std Final q"] < sym["Std Final q"])
        results[f"H1_std_profit_lower_gamma{gamma}"] = h1
        results[f"H2_std_q_lower_gamma{gamma}"] = h2

    # H3: convergence for small gamma
    # The strategies converge as gamma→0: the ratio std(inv)/std(sym) should be
    # closer to 1 for gamma=0.01 than for gamma=0.5 (higher ratio = more similar).
    inv_001 = df[(df["gamma"] == 0.01) & (df["Strategy"] == "Inventory-Aware")].iloc[0]
    sym_001 = df[(df["gamma"] == 0.01) & (df["Strategy"] == "Symmetric")].iloc[0]
    inv_05 = df[(df["gamma"] == 0.5) & (df["Strategy"] == "Inventory-Aware")].iloc[0]
    sym_05 = df[(df["gamma"] == 0.5) & (df["Strategy"] == "Symmetric")].iloc[0]
    ratio_001 = inv_001["Std Profit"] / sym_001["Std Profit"] if sym_001["Std Profit"] > 0 else np.inf
    ratio_05 = inv_05["Std Profit"] / sym_05["Std Profit"] if sym_05["Std Profit"] > 0 else np.inf
    # gamma=0.01 should be more similar (ratio closer to 1) than gamma=0.5
    results["H3_convergence_gamma_0.01"] = bool(ratio_001 > ratio_05)

    # H4: higher gamma → lower mean profit for inventory-aware
    inv_profits = {
        g: df[(df["gamma"] == g) & (df["Strategy"] == "Inventory-Aware")].iloc[0]["Mean Profit"]
        for g in GAMMA_VALUES
    }
    results["H4_higher_gamma_lower_profit"] = bool(inv_profits[0.01] > inv_profits[0.5])

    # H5: higher gamma → lower std(Profit) for inventory-aware
    inv_std = {
        g: df[(df["gamma"] == g) & (df["Strategy"] == "Inventory-Aware")].iloc[0]["Std Profit"]
        for g in GAMMA_VALUES
    }
    results["H5_higher_gamma_lower_std_profit"] = bool(inv_std[0.01] > inv_std[0.5])

    return results
