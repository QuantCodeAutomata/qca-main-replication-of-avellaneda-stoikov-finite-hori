"""
Tests for Experiment 1 and Experiment 2 implementations.

Validates:
  - Experiment-level hypothesis checks
  - Spread convention differences between experiments
  - Qualitative properties of simulation results
  - Spread value verification for Experiment 2
  - Hypothesis check functions
"""

from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation import ASParams, run_monte_carlo, spread_liquidity_only, spread_theory_faithful
from src.exp1_theory_spread import (
    check_hypotheses_exp1,
    GAMMA_VALUES as GAMMA_EXP1,
    BASE_PARAMS as BASE_PARAMS_EXP1,
)
from src.exp2_liquidity_spread import (
    check_hypotheses_exp2,
    verify_spread_values,
    GAMMA_VALUES as GAMMA_EXP2,
    BASE_PARAMS as BASE_PARAMS_EXP2,
    PAPER_SPREAD_VALUES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fast_params_dict() -> dict:
    """Fast parameters for experiment-level tests (fewer paths)."""
    return {
        "S0": 100.0, "T": 1.0, "dt": 0.005, "sigma": 2.0,
        "A": 140.0, "k": 1.5, "n_paths": 200, "X0": 0.0, "q0": 0, "seed": 42,
    }


@pytest.fixture(scope="module")
def exp1_summary_rows(fast_params_dict: dict) -> list[dict]:
    """Run a fast version of Exp1 and return summary rows."""
    from src.simulation import compute_summary, spread_theory_faithful
    rows = []
    for gamma in GAMMA_EXP1:
        params = ASParams(**{**fast_params_dict, "gamma": gamma})
        spread_val = spread_theory_faithful(gamma, params.sigma, params.k, tau=params.T)
        for strategy in ("inventory_aware", "symmetric"):
            label = "Inventory-Aware" if strategy == "inventory_aware" else "Symmetric"
            mc = run_monte_carlo(params, "theory_faithful", strategy)
            row = compute_summary(mc, label, "Theory-Faithful", spread_val)
            row["gamma"] = gamma
            rows.append(row)
    return rows


@pytest.fixture(scope="module")
def exp2_summary_rows(fast_params_dict: dict) -> list[dict]:
    """Run a fast version of Exp2 and return summary rows."""
    from src.simulation import compute_summary, spread_liquidity_only
    rows = []
    for gamma in GAMMA_EXP2:
        params = ASParams(**{**fast_params_dict, "gamma": gamma})
        spread_val = spread_liquidity_only(gamma, params.k)
        for strategy in ("inventory_aware", "symmetric"):
            label = "Inventory-Aware" if strategy == "inventory_aware" else "Symmetric"
            mc = run_monte_carlo(params, "liquidity_only", strategy)
            row = compute_summary(mc, label, "Liquidity-Only", spread_val)
            row["gamma"] = gamma
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Tests: Experiment 1 — Theory-Faithful Spread
# ---------------------------------------------------------------------------

class TestExp1TheorySpread:
    def test_gamma_values_correct(self) -> None:
        """Experiment 1 should use gamma in {0.01, 0.1, 0.5}."""
        assert set(GAMMA_EXP1) == {0.01, 0.1, 0.5}

    def test_base_params_correct(self) -> None:
        """Base parameters should match paper specification."""
        assert BASE_PARAMS_EXP1["S0"] == 100.0
        assert BASE_PARAMS_EXP1["T"] == 1.0
        assert BASE_PARAMS_EXP1["dt"] == 0.005
        assert BASE_PARAMS_EXP1["sigma"] == 2.0
        assert BASE_PARAMS_EXP1["A"] == 140.0
        assert BASE_PARAMS_EXP1["k"] == 1.5

    def test_theory_spread_larger_than_liquidity_at_t0(self) -> None:
        """Theory-faithful spread at t=0 should exceed liquidity-only spread."""
        for gamma in GAMMA_EXP1:
            s_theory = spread_theory_faithful(gamma, sigma=2.0, k=1.5, tau=1.0)
            s_liq = spread_liquidity_only(gamma, k=1.5)
            assert s_theory > s_liq, (
                f"Theory spread should exceed liquidity spread at t=0 for gamma={gamma}"
            )

    def test_inventory_aware_lower_std_profit(
        self, exp1_summary_rows: list[dict]
    ) -> None:
        """Inventory-aware should have lower std(Profit) than symmetric for each gamma."""
        import pandas as pd
        df = pd.DataFrame(exp1_summary_rows)
        for gamma in GAMMA_EXP1:
            inv = df[(df["gamma"] == gamma) & (df["Strategy"] == "Inventory-Aware")].iloc[0]
            sym = df[(df["gamma"] == gamma) & (df["Strategy"] == "Symmetric")].iloc[0]
            assert inv["Std Profit"] < sym["Std Profit"], (
                f"gamma={gamma}: inv std={inv['Std Profit']:.2f} >= sym std={sym['Std Profit']:.2f}"
            )

    def test_inventory_aware_lower_std_q(
        self, exp1_summary_rows: list[dict]
    ) -> None:
        """Inventory-aware should have lower std(Final q) than symmetric for each gamma."""
        import pandas as pd
        df = pd.DataFrame(exp1_summary_rows)
        for gamma in GAMMA_EXP1:
            inv = df[(df["gamma"] == gamma) & (df["Strategy"] == "Inventory-Aware")].iloc[0]
            sym = df[(df["gamma"] == gamma) & (df["Strategy"] == "Symmetric")].iloc[0]
            assert inv["Std Final q"] < sym["Std Final q"], (
                f"gamma={gamma}: inv std_q={inv['Std Final q']:.2f} >= sym std_q={sym['Std Final q']:.2f}"
            )

    def test_higher_gamma_lower_profit_inventory_aware(
        self, exp1_summary_rows: list[dict]
    ) -> None:
        """Higher gamma should yield lower mean profit for inventory-aware strategy."""
        import pandas as pd
        df = pd.DataFrame(exp1_summary_rows)
        profits = {
            g: df[(df["gamma"] == g) & (df["Strategy"] == "Inventory-Aware")].iloc[0]["Mean Profit"]
            for g in GAMMA_EXP1
        }
        assert profits[0.01] > profits[0.5], (
            f"gamma=0.01 profit ({profits[0.01]:.2f}) should exceed gamma=0.5 profit ({profits[0.5]:.2f})"
        )

    def test_hypothesis_check_function_returns_dict(
        self, exp1_summary_rows: list[dict]
    ) -> None:
        """check_hypotheses_exp1 should return a dict of bool values."""
        results = check_hypotheses_exp1(exp1_summary_rows)
        assert isinstance(results, dict)
        assert all(isinstance(v, bool) for v in results.values()), (
            f"Non-bool values: {[(k, type(v)) for k, v in results.items() if not isinstance(v, bool)]}"
        )

    def test_hypothesis_keys_present(
        self, exp1_summary_rows: list[dict]
    ) -> None:
        """All expected hypothesis keys should be present."""
        results = check_hypotheses_exp1(exp1_summary_rows)
        for gamma in GAMMA_EXP1:
            assert f"H1_std_profit_lower_gamma{gamma}" in results
            assert f"H2_std_q_lower_gamma{gamma}" in results
        assert "H3_convergence_gamma_0.01" in results
        assert "H4_higher_gamma_lower_profit" in results

    def test_convergence_hypothesis_passes(
        self, exp1_summary_rows: list[dict]
    ) -> None:
        """
        Convergence: ratio std(inv)/std(sym) should be higher for gamma=0.01
        than for gamma=0.5 (strategies more similar for small gamma).
        """
        results = check_hypotheses_exp1(exp1_summary_rows)
        assert results["H3_convergence_gamma_0.01"], (
            "Convergence hypothesis failed: gamma=0.01 ratio should exceed gamma=0.5 ratio"
        )


# ---------------------------------------------------------------------------
# Tests: Experiment 2 — Liquidity-Only Spread
# ---------------------------------------------------------------------------

class TestExp2LiquiditySpread:
    def test_gamma_values_correct(self) -> None:
        """Experiment 2 should use gamma in {0.01, 0.1, 0.5}."""
        assert set(GAMMA_EXP2) == {0.01, 0.1, 0.5}

    def test_base_params_same_as_exp1(self) -> None:
        """Experiment 2 base params should match Experiment 1."""
        for key in ["S0", "T", "dt", "sigma", "A", "k"]:
            assert BASE_PARAMS_EXP2[key] == BASE_PARAMS_EXP1[key], (
                f"Parameter {key} differs between experiments"
            )

    def test_spread_values_match_paper(self) -> None:
        """Liquidity-only spread should match paper's reported values."""
        verification = verify_spread_values(GAMMA_EXP2, k=1.5)
        for gamma in GAMMA_EXP2:
            assert verification[gamma]["matches"], (
                f"Spread mismatch for gamma={gamma}: "
                f"computed={verification[gamma]['computed_spread']}, "
                f"paper={verification[gamma]['paper_spread']}"
            )

    def test_spread_gamma_001_approx_133(self) -> None:
        """Spread for gamma=0.01 should be approximately 1.33."""
        s = spread_liquidity_only(0.01, k=1.5)
        assert abs(s - 1.33) < 0.05

    def test_spread_gamma_01_approx_129(self) -> None:
        """Spread for gamma=0.1 should be approximately 1.29."""
        s = spread_liquidity_only(0.1, k=1.5)
        assert abs(s - 1.29) < 0.05

    def test_spread_gamma_05_approx_115(self) -> None:
        """Spread for gamma=0.5 should be approximately 1.15."""
        s = spread_liquidity_only(0.5, k=1.5)
        assert abs(s - 1.15) < 0.05

    def test_inventory_aware_lower_std_profit(
        self, exp2_summary_rows: list[dict]
    ) -> None:
        """Inventory-aware should have lower std(Profit) than symmetric for each gamma."""
        import pandas as pd
        df = pd.DataFrame(exp2_summary_rows)
        for gamma in GAMMA_EXP2:
            inv = df[(df["gamma"] == gamma) & (df["Strategy"] == "Inventory-Aware")].iloc[0]
            sym = df[(df["gamma"] == gamma) & (df["Strategy"] == "Symmetric")].iloc[0]
            assert inv["Std Profit"] < sym["Std Profit"], (
                f"gamma={gamma}: inv std={inv['Std Profit']:.2f} >= sym std={sym['Std Profit']:.2f}"
            )

    def test_inventory_aware_lower_std_q(
        self, exp2_summary_rows: list[dict]
    ) -> None:
        """Inventory-aware should have lower std(Final q) than symmetric for each gamma."""
        import pandas as pd
        df = pd.DataFrame(exp2_summary_rows)
        for gamma in GAMMA_EXP2:
            inv = df[(df["gamma"] == gamma) & (df["Strategy"] == "Inventory-Aware")].iloc[0]
            sym = df[(df["gamma"] == gamma) & (df["Strategy"] == "Symmetric")].iloc[0]
            assert inv["Std Final q"] < sym["Std Final q"], (
                f"gamma={gamma}: inv std_q={inv['Std Final q']:.2f} >= sym std_q={sym['Std Final q']:.2f}"
            )

    def test_higher_gamma_lower_profit_inventory_aware(
        self, exp2_summary_rows: list[dict]
    ) -> None:
        """Higher gamma should yield lower mean profit for inventory-aware strategy."""
        import pandas as pd
        df = pd.DataFrame(exp2_summary_rows)
        profits = {
            g: df[(df["gamma"] == g) & (df["Strategy"] == "Inventory-Aware")].iloc[0]["Mean Profit"]
            for g in GAMMA_EXP2
        }
        assert profits[0.01] > profits[0.5]

    def test_verify_spread_values_returns_correct_structure(self) -> None:
        """verify_spread_values should return dict with correct keys."""
        verification = verify_spread_values([0.1], k=1.5)
        assert 0.1 in verification
        for key in ["computed_spread", "paper_spread", "abs_diff", "matches"]:
            assert key in verification[0.1]

    def test_hypothesis_check_function_returns_dict(
        self, exp2_summary_rows: list[dict]
    ) -> None:
        """check_hypotheses_exp2 should return a dict of bool values."""
        verification = verify_spread_values(GAMMA_EXP2, k=1.5)
        results = check_hypotheses_exp2(exp2_summary_rows, verification)
        assert isinstance(results, dict)
        assert all(isinstance(v, bool) for v in results.values()), (
            f"Non-bool values: {[(k, type(v)) for k, v in results.items() if not isinstance(v, bool)]}"
        )

    def test_convergence_hypothesis_passes(
        self, exp2_summary_rows: list[dict]
    ) -> None:
        """
        Convergence: ratio std(inv)/std(sym) should be higher for gamma=0.01
        than for gamma=0.5 (strategies more similar for small gamma).
        """
        verification = verify_spread_values(GAMMA_EXP2, k=1.5)
        results = check_hypotheses_exp2(exp2_summary_rows, verification)
        assert results["H4_convergence_gamma_0.01"], (
            "Convergence hypothesis failed: gamma=0.01 ratio should exceed gamma=0.5 ratio"
        )

    def test_liquidity_spread_constant_in_time(self) -> None:
        """Liquidity-only spread should not depend on tau."""
        gamma, k = 0.1, 1.5
        s1 = spread_liquidity_only(gamma, k)
        s2 = spread_liquidity_only(gamma, k)
        assert s1 == s2  # deterministic, no tau argument


# ---------------------------------------------------------------------------
# Tests: Cross-experiment comparisons
# ---------------------------------------------------------------------------

class TestCrossExperimentComparisons:
    def test_exp1_spread_larger_than_exp2_at_t0(self) -> None:
        """
        At t=0 (tau=T=1), theory-faithful spread should exceed liquidity-only spread
        because it includes the inventory-risk term.
        """
        for gamma in [0.01, 0.1, 0.5]:
            s1 = spread_theory_faithful(gamma, sigma=2.0, k=1.5, tau=1.0)
            s2 = spread_liquidity_only(gamma, k=1.5)
            assert s1 > s2, f"gamma={gamma}: theory spread {s1:.4f} <= liquidity spread {s2:.4f}"

    def test_exp1_exp2_same_params(self) -> None:
        """Both experiments should use identical base parameters."""
        for key in ["S0", "T", "dt", "sigma", "A", "k"]:
            assert BASE_PARAMS_EXP1[key] == BASE_PARAMS_EXP2[key]

    def test_exp2_spread_matches_paper_better(self) -> None:
        """
        Exp2 spread values should match paper's reported values within tolerance.
        This is the key claim of Exp2.
        """
        for gamma in [0.01, 0.1, 0.5]:
            s = spread_liquidity_only(gamma, k=1.5)
            paper_val = PAPER_SPREAD_VALUES[gamma]
            assert abs(s - paper_val) < 0.05, (
                f"gamma={gamma}: computed={s:.4f}, paper={paper_val}"
            )

    def test_both_experiments_produce_positive_mean_profit(
        self,
        exp1_summary_rows: list[dict],
        exp2_summary_rows: list[dict],
    ) -> None:
        """Both experiments should produce positive mean profit for gamma=0.01."""
        import pandas as pd
        for rows, label in [(exp1_summary_rows, "Exp1"), (exp2_summary_rows, "Exp2")]:
            df = pd.DataFrame(rows)
            inv = df[(df["gamma"] == 0.01) & (df["Strategy"] == "Inventory-Aware")].iloc[0]
            assert inv["Mean Profit"] > 0, f"{label}: mean profit should be positive"
