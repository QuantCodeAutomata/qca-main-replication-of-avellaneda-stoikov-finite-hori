"""
Main entry point for the Avellaneda-Stoikov Market Making Replication.

Runs both experiments and saves all results to the results/ directory.

Experiment 1: Theory-faithful time-varying spread
    spread(t) = gamma*sigma^2*tau + (2/gamma)*ln(1 + gamma/k)

Experiment 2: Liquidity-only (table-matching) spread
    spread = (2/gamma)*ln(1 + gamma/k)

Parameter sheet:
  S0=100, T=1, dt=0.005, sigma=2, A=140, k=1.5
  q0=0, X0=0, gamma in {0.01, 0.1, 0.5}
  n_paths=1000 per strategy per gamma
  Price update: binomial ±sigma*sqrt(dt)
  Fill probability: lambda*dt clipped to [0,1]
  Both bid and ask can fill in the same step (independent Bernoulli draws)
  Seed: 42 (fixed global seed, separate RNG per strategy run)
  Common random numbers: NOT used across strategies
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.exp1_theory_spread import (
    run_experiment_1,
    check_hypotheses_exp1,
    GAMMA_VALUES as GAMMA_VALUES_EXP1,
)
from src.exp2_liquidity_spread import (
    run_experiment_2,
    check_hypotheses_exp2,
    verify_spread_values,
    GAMMA_VALUES as GAMMA_VALUES_EXP2,
    BASE_PARAMS as BASE_PARAMS_EXP2,
)

RESULTS_DIR = Path("results")


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def format_hypothesis_results(results: dict[str, bool]) -> str:
    """Format hypothesis test results as a readable string."""
    lines = []
    for key, passed in results.items():
        status = "PASS ✓" if passed else "FAIL ✗"
        lines.append(f"  {status}  {key}")
    return "\n".join(lines)


def build_results_markdown(
    exp1_summary: list[dict],
    exp2_summary: list[dict],
    exp1_hypotheses: dict[str, bool],
    exp2_hypotheses: dict[str, bool],
    spread_verification: dict,
) -> str:
    """
    Build the RESULTS.md content with all experiment metrics.

    Parameters
    ----------
    exp1_summary : list[dict]
        Summary rows from Experiment 1.
    exp2_summary : list[dict]
        Summary rows from Experiment 2.
    exp1_hypotheses : dict[str, bool]
        Hypothesis test results for Experiment 1.
    exp2_hypotheses : dict[str, bool]
        Hypothesis test results for Experiment 2.
    spread_verification : dict
        Spread verification results for Experiment 2.

    Returns
    -------
    str
        Full Markdown content for RESULTS.md.
    """
    lines = [
        "# Avellaneda-Stoikov Market Making Replication — Results",
        "",
        "## Parameter Sheet",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| S0 | 100 |",
        "| T | 1 |",
        "| dt | 0.005 (200 steps) |",
        "| sigma | 2 |",
        "| A | 140 |",
        "| k | 1.5 |",
        "| q0 | 0 |",
        "| X0 | 0 |",
        "| gamma values | {0.01, 0.1, 0.5} |",
        "| n_paths | 1000 per strategy per gamma |",
        "| Price update | Binomial ±sigma*sqrt(dt) |",
        "| Fill probability | lambda*dt clipped to [0,1] |",
        "| Fill model | Independent Bernoulli (both sides can fill same step) |",
        "| Random seed | 42 (fixed global, separate RNG per strategy run) |",
        "| Common random numbers | Not used across strategies |",
        "",
        "---",
        "",
        "## Experiment 1: Theory-Faithful Time-Varying Spread",
        "",
        "**Spread formula:** `spread(t) = gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/k)`",
        "",
    ]

    # Exp1 summary table
    df1 = pd.DataFrame(exp1_summary)
    lines.append(df1.to_markdown(index=False))
    lines.append("")

    # Exp1 hypotheses
    lines.append("### Hypothesis Tests — Experiment 1")
    lines.append("")
    lines.append("| Hypothesis | Result |")
    lines.append("|------------|--------|")
    for key, passed in exp1_hypotheses.items():
        status = "PASS ✓" if passed else "FAIL ✗"
        lines.append(f"| {key} | {status} |")
    lines.append("")

    # Exp1 interpretation
    all_h1 = all(v for k, v in exp1_hypotheses.items() if k.startswith("H1"))
    all_h2 = all(v for k, v in exp1_hypotheses.items() if k.startswith("H2"))
    lines.append("**Interpretation:**")
    lines.append(
        f"- Inventory-aware strategy has lower std(Profit) than symmetric: "
        f"{'YES' if all_h1 else 'NO (partial)'}"
    )
    lines.append(
        f"- Inventory-aware strategy has lower std(Final q) than symmetric: "
        f"{'YES' if all_h2 else 'NO (partial)'}"
    )
    lines.append(
        f"- Strategies converge for gamma=0.01: "
        f"{'YES' if exp1_hypotheses.get('H3_convergence_gamma_0.01') else 'NO'}"
    )
    lines.append(
        f"- Higher gamma reduces mean profit: "
        f"{'YES' if exp1_hypotheses.get('H4_higher_gamma_lower_profit') else 'NO'}"
    )
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Experiment 2: Liquidity-Only (Table-Matching) Spread")
    lines.append("")
    lines.append("**Spread formula:** `spread = (2/gamma)*ln(1 + gamma/k)` (constant in time)")
    lines.append("")

    # Spread verification
    lines.append("### Spread Value Verification vs Paper")
    lines.append("")
    lines.append("| gamma | Computed Spread | Paper Value | Abs Diff | Match |")
    lines.append("|-------|----------------|-------------|----------|-------|")
    for gamma, info in spread_verification.items():
        match_str = "✓" if info["matches"] else "✗"
        lines.append(
            f"| {gamma} | {info['computed_spread']:.4f} | "
            f"{info['paper_spread']} | {info['abs_diff']:.4f} | {match_str} |"
        )
    lines.append("")

    # Exp2 summary table
    df2 = pd.DataFrame(exp2_summary)
    lines.append(df2.to_markdown(index=False))
    lines.append("")

    # Paper comparison
    lines.append("### Comparison to Paper's Reported Values")
    lines.append("")
    lines.append("| gamma | Strategy | Simulated Profit | Paper Profit | Diff |")
    lines.append("|-------|----------|-----------------|--------------|------|")
    paper_profits = {
        0.01: {"Inventory-Aware": 66.78, "Symmetric": 67.36},
        0.1:  {"Inventory-Aware": 62.94, "Symmetric": 67.21},
        0.5:  {"Inventory-Aware": 33.92, "Symmetric": 66.20},
    }
    for _, row in df2.iterrows():
        g = row["gamma"]
        strat = row["Strategy"]
        sim_profit = row["Mean Profit"]
        paper_val = paper_profits.get(g, {}).get(strat, float("nan"))
        diff = sim_profit - paper_val if not np.isnan(paper_val) else float("nan")
        lines.append(
            f"| {g} | {strat} | {sim_profit:.2f} | {paper_val} | {diff:+.2f} |"
        )
    lines.append("")

    # Exp2 hypotheses
    lines.append("### Hypothesis Tests — Experiment 2")
    lines.append("")
    lines.append("| Hypothesis | Result |")
    lines.append("|------------|--------|")
    for key, passed in exp2_hypotheses.items():
        status = "PASS ✓" if passed else "FAIL ✗"
        lines.append(f"| {key} | {status} |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Generated Figures")
    lines.append("")
    lines.append("### Experiment 1 (Theory-Faithful Spread)")
    for gamma in [0.01, 0.1, 0.5]:
        lines.append(f"- `hist_profit_exp1_theory_spread_gamma{gamma}.png` — Profit histograms")
        lines.append(f"- `path_exp1_theory_spread_gamma{gamma}.png` — Illustrative path")
    lines.append("- `gamma_comparison_exp1_theory_spread.png` — Risk metrics by gamma")
    lines.append("")
    lines.append("### Experiment 2 (Liquidity-Only Spread)")
    for gamma in [0.01, 0.1, 0.5]:
        lines.append(f"- `hist_profit_exp2_liquidity_spread_gamma{gamma}.png` — Profit histograms")
        lines.append(f"- `path_exp2_liquidity_spread_gamma{gamma}.png` — Illustrative path")
    lines.append("- `gamma_comparison_exp2_liquidity_spread.png` — Risk metrics by gamma")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Run both experiments and save all results."""
    print_section("Avellaneda-Stoikov Market Making Replication")
    print("Parameter sheet:")
    print("  S0=100, T=1, dt=0.005, sigma=2, A=140, k=1.5")
    print("  gamma in {0.01, 0.1, 0.5}, n_paths=1000, seed=42")
    print("  Price update: binomial ±sigma*sqrt(dt)")
    print("  Fill probability: lambda*dt clipped to [0,1]")
    print("  Independent Bernoulli fills (both sides can fill same step)")

    RESULTS_DIR.mkdir(exist_ok=True)

    # -----------------------------------------------------------------------
    # Experiment 1
    # -----------------------------------------------------------------------
    print_section("Experiment 1: Theory-Faithful Time-Varying Spread")
    exp1_summary, exp1_mc = run_experiment_1(RESULTS_DIR)
    exp1_hypotheses = check_hypotheses_exp1(exp1_summary)

    print("\n[Exp1] Hypothesis test results:")
    print(format_hypothesis_results(exp1_hypotheses))

    # -----------------------------------------------------------------------
    # Experiment 2
    # -----------------------------------------------------------------------
    print_section("Experiment 2: Liquidity-Only (Table-Matching) Spread")
    exp2_summary, exp2_mc = run_experiment_2(RESULTS_DIR)

    spread_verification = verify_spread_values(GAMMA_VALUES_EXP2, k=BASE_PARAMS_EXP2["k"])
    exp2_hypotheses = check_hypotheses_exp2(exp2_summary, spread_verification)

    print("\n[Exp2] Hypothesis test results:")
    print(format_hypothesis_results(exp2_hypotheses))

    # -----------------------------------------------------------------------
    # Save RESULTS.md
    # -----------------------------------------------------------------------
    print_section("Saving Results")
    md_content = build_results_markdown(
        exp1_summary,
        exp2_summary,
        exp1_hypotheses,
        exp2_hypotheses,
        spread_verification,
    )

    results_md_path = RESULTS_DIR / "RESULTS.md"
    with open(results_md_path, "w") as f:
        f.write(md_content)
    print(f"  Saved: {results_md_path}")

    # List all saved files
    saved_files = sorted(RESULTS_DIR.iterdir())
    print(f"\n  Total files in results/: {len(saved_files)}")
    for f in saved_files:
        print(f"    {f.name}")

    # Overall pass/fail summary
    all_hypotheses = {**exp1_hypotheses, **exp2_hypotheses}
    n_pass = sum(all_hypotheses.values())
    n_total = len(all_hypotheses)
    print(f"\n  Hypothesis tests: {n_pass}/{n_total} passed")

    if n_pass == n_total:
        print("\n  ✓ All hypotheses confirmed — replication successful!")
    else:
        failed = [k for k, v in all_hypotheses.items() if not v]
        print(f"\n  ✗ {n_total - n_pass} hypothesis/hypotheses failed:")
        for f in failed:
            print(f"    - {f}")


if __name__ == "__main__":
    main()
