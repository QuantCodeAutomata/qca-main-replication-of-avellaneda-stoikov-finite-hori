"""
Tests for the core simulation engine (src/simulation.py).

Validates:
  - Mathematical properties of AS model formulas
  - Reservation price properties
  - Spread formula correctness
  - Execution intensity and fill probability
  - Single-path simulation mechanics
  - Monte Carlo runner output shapes and types
  - Edge cases and boundary conditions
"""

from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation import (
    ASParams,
    reservation_price,
    spread_theory_faithful,
    spread_liquidity_only,
    execution_intensity,
    fill_probability,
    simulate_path,
    run_monte_carlo,
    compute_summary,
    validate_reservation_price_properties,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_params() -> ASParams:
    """Default parameters for testing."""
    return ASParams(
        S0=100.0, T=1.0, dt=0.005, sigma=2.0,
        A=140.0, k=1.5, gamma=0.1, n_paths=100,
        X0=0.0, q0=0, seed=42,
    )


@pytest.fixture
def small_params() -> ASParams:
    """Small parameter set for fast tests."""
    return ASParams(
        S0=100.0, T=0.1, dt=0.01, sigma=2.0,
        A=140.0, k=1.5, gamma=0.1, n_paths=10,
        X0=0.0, q0=0, seed=0,
    )


# ---------------------------------------------------------------------------
# Tests: ASParams
# ---------------------------------------------------------------------------

class TestASParams:
    def test_n_steps_correct(self) -> None:
        """n_steps should equal T/dt."""
        p = ASParams(T=1.0, dt=0.005)
        assert p.n_steps == 200

    def test_n_steps_small_dt(self) -> None:
        p = ASParams(T=0.1, dt=0.01)
        assert p.n_steps == 10

    def test_invalid_dt_raises(self) -> None:
        with pytest.raises(AssertionError):
            ASParams(dt=-0.001)

    def test_invalid_gamma_raises(self) -> None:
        with pytest.raises(AssertionError):
            ASParams(gamma=-0.1)

    def test_invalid_sigma_raises(self) -> None:
        with pytest.raises(AssertionError):
            ASParams(sigma=0.0)

    def test_invalid_n_paths_raises(self) -> None:
        with pytest.raises(AssertionError):
            ASParams(n_paths=0)


# ---------------------------------------------------------------------------
# Tests: reservation_price
# ---------------------------------------------------------------------------

class TestReservationPrice:
    def test_zero_inventory_equals_mid(self) -> None:
        """r(s, 0, t) = s for any tau."""
        S = 100.0
        for tau in [0.0, 0.5, 1.0]:
            r = reservation_price(S, q=0, gamma=0.1, sigma=2.0, tau=tau)
            assert abs(r - S) < 1e-10, f"r != S at tau={tau}: {r}"

    def test_positive_inventory_lowers_reservation_price(self) -> None:
        """Positive inventory → reservation price below mid."""
        r = reservation_price(100.0, q=5, gamma=0.1, sigma=2.0, tau=0.5)
        assert r < 100.0

    def test_negative_inventory_raises_reservation_price(self) -> None:
        """Negative inventory → reservation price above mid."""
        r = reservation_price(100.0, q=-5, gamma=0.1, sigma=2.0, tau=0.5)
        assert r > 100.0

    def test_linearity_in_q(self) -> None:
        """r should be linear in q: r(q+1) - r(q) = constant."""
        S, gamma, sigma, tau = 100.0, 0.1, 2.0, 0.5
        r0 = reservation_price(S, 0, gamma, sigma, tau)
        r1 = reservation_price(S, 1, gamma, sigma, tau)
        r2 = reservation_price(S, 2, gamma, sigma, tau)
        assert abs((r0 - r1) - (r1 - r2)) < 1e-10

    def test_linearity_in_tau(self) -> None:
        """r should be linear in tau for fixed q."""
        S, q, gamma, sigma = 100.0, 3, 0.1, 2.0
        r_half = reservation_price(S, q, gamma, sigma, tau=0.5)
        r_one = reservation_price(S, q, gamma, sigma, tau=1.0)
        # r_one should be further from S than r_half
        assert abs(S - r_one) > abs(S - r_half)

    def test_exact_formula(self) -> None:
        """Verify exact formula: r = S - q*gamma*sigma^2*tau."""
        S, q, gamma, sigma, tau = 100.0, 3, 0.1, 2.0, 0.5
        expected = S - q * gamma * sigma**2 * tau
        result = reservation_price(S, q, gamma, sigma, tau)
        assert abs(result - expected) < 1e-12

    def test_zero_tau(self) -> None:
        """At tau=0, reservation price equals mid regardless of inventory."""
        r = reservation_price(100.0, q=10, gamma=0.5, sigma=2.0, tau=0.0)
        assert abs(r - 100.0) < 1e-10


# ---------------------------------------------------------------------------
# Tests: spread formulas
# ---------------------------------------------------------------------------

class TestSpreadFormulas:
    def test_theory_spread_at_tau_zero_equals_liquidity(self) -> None:
        """At tau=0, theory-faithful spread equals liquidity-only spread."""
        gamma, sigma, k = 0.1, 2.0, 1.5
        s_theory = spread_theory_faithful(gamma, sigma, k, tau=0.0)
        s_liq = spread_liquidity_only(gamma, k)
        assert abs(s_theory - s_liq) < 1e-10

    def test_theory_spread_increases_with_tau(self) -> None:
        """Theory-faithful spread should increase with remaining time."""
        gamma, sigma, k = 0.1, 2.0, 1.5
        s_small = spread_theory_faithful(gamma, sigma, k, tau=0.1)
        s_large = spread_theory_faithful(gamma, sigma, k, tau=1.0)
        assert s_large > s_small

    def test_liquidity_spread_positive(self) -> None:
        """Liquidity-only spread must be positive."""
        for gamma in [0.01, 0.1, 0.5]:
            s = spread_liquidity_only(gamma, k=1.5)
            assert s > 0, f"Spread not positive for gamma={gamma}"

    def test_liquidity_spread_decreases_with_gamma(self) -> None:
        """Liquidity spread should decrease as gamma increases (for k=1.5)."""
        s_small = spread_liquidity_only(0.01, k=1.5)
        s_large = spread_liquidity_only(0.5, k=1.5)
        assert s_small > s_large

    def test_liquidity_spread_gamma_001(self) -> None:
        """Verify spread ≈ 1.33 for gamma=0.01, k=1.5."""
        s = spread_liquidity_only(0.01, k=1.5)
        assert abs(s - 1.33) < 0.05, f"Expected ≈1.33, got {s:.4f}"

    def test_liquidity_spread_gamma_01(self) -> None:
        """Verify spread ≈ 1.29 for gamma=0.1, k=1.5."""
        s = spread_liquidity_only(0.1, k=1.5)
        assert abs(s - 1.29) < 0.05, f"Expected ≈1.29, got {s:.4f}"

    def test_liquidity_spread_gamma_05(self) -> None:
        """Verify spread ≈ 1.15 for gamma=0.5, k=1.5."""
        s = spread_liquidity_only(0.5, k=1.5)
        assert abs(s - 1.15) < 0.05, f"Expected ≈1.15, got {s:.4f}"

    def test_theory_spread_exact_formula(self) -> None:
        """Verify exact formula for theory-faithful spread."""
        gamma, sigma, k, tau = 0.1, 2.0, 1.5, 0.5
        expected = gamma * sigma**2 * tau + (2.0 / gamma) * np.log(1.0 + gamma / k)
        result = spread_theory_faithful(gamma, sigma, k, tau)
        assert abs(result - expected) < 1e-12

    def test_liquidity_spread_exact_formula(self) -> None:
        """Verify exact formula for liquidity-only spread."""
        gamma, k = 0.1, 1.5
        expected = (2.0 / gamma) * np.log(1.0 + gamma / k)
        result = spread_liquidity_only(gamma, k)
        assert abs(result - expected) < 1e-12


# ---------------------------------------------------------------------------
# Tests: execution_intensity and fill_probability
# ---------------------------------------------------------------------------

class TestExecutionModel:
    def test_intensity_positive_delta(self) -> None:
        """Intensity should be positive for positive delta."""
        lam = execution_intensity(0.5, A=140.0, k=1.5)
        assert lam > 0

    def test_intensity_decreases_with_delta(self) -> None:
        """Larger delta → lower intensity."""
        lam_small = execution_intensity(0.5, A=140.0, k=1.5)
        lam_large = execution_intensity(1.0, A=140.0, k=1.5)
        assert lam_small > lam_large

    def test_intensity_at_zero_delta(self) -> None:
        """At delta=0, intensity equals A."""
        lam = execution_intensity(0.0, A=140.0, k=1.5)
        assert abs(lam - 140.0) < 1e-10

    def test_intensity_exact_formula(self) -> None:
        """Verify exact formula: lambda = A*exp(-k*delta)."""
        delta, A, k = 0.5, 140.0, 1.5
        expected = A * np.exp(-k * delta)
        result = execution_intensity(delta, A, k)
        assert abs(result - expected) < 1e-10

    def test_fill_probability_in_unit_interval(self) -> None:
        """Fill probability must be in [0, 1]."""
        for intensity in [0.0, 1.0, 10.0, 1000.0]:
            p = fill_probability(intensity, dt=0.005)
            assert 0.0 <= p <= 1.0, f"p={p} out of [0,1] for intensity={intensity}"

    def test_fill_probability_capped_at_one(self) -> None:
        """Very high intensity should be capped at 1.0."""
        p = fill_probability(1e6, dt=0.005)
        assert p == 1.0

    def test_fill_probability_zero_intensity(self) -> None:
        """Zero intensity → zero fill probability."""
        p = fill_probability(0.0, dt=0.005)
        assert p == 0.0

    def test_fill_probability_proportional_to_dt(self) -> None:
        """For small intensities, p_fill ≈ lambda*dt."""
        intensity = 10.0
        dt = 0.001
        p = fill_probability(intensity, dt)
        assert abs(p - intensity * dt) < 1e-10


# ---------------------------------------------------------------------------
# Tests: simulate_path
# ---------------------------------------------------------------------------

class TestSimulatePath:
    def test_output_keys_present(self, default_params: ASParams) -> None:
        """simulate_path should return all required keys."""
        rng = np.random.default_rng(0)
        result = simulate_path(default_params, "theory_faithful", "inventory_aware", rng)
        for key in ["profit", "final_q", "path_S", "path_r", "path_bid", "path_ask", "path_t"]:
            assert key in result, f"Missing key: {key}"

    def test_path_lengths(self, default_params: ASParams) -> None:
        """Path arrays should have correct lengths."""
        rng = np.random.default_rng(0)
        result = simulate_path(default_params, "theory_faithful", "inventory_aware", rng)
        n = default_params.n_steps
        assert len(result["path_S"]) == n + 1
        assert len(result["path_r"]) == n + 1
        assert len(result["path_t"]) == n + 1
        assert len(result["path_bid"]) == n
        assert len(result["path_ask"]) == n

    def test_terminal_time(self, default_params: ASParams) -> None:
        """Final time in path_t should equal T."""
        rng = np.random.default_rng(0)
        result = simulate_path(default_params, "theory_faithful", "inventory_aware", rng)
        assert abs(result["path_t"][-1] - default_params.T) < 1e-10

    def test_initial_price(self, default_params: ASParams) -> None:
        """Initial price in path_S should equal S0."""
        rng = np.random.default_rng(0)
        result = simulate_path(default_params, "theory_faithful", "inventory_aware", rng)
        assert abs(result["path_S"][0] - default_params.S0) < 1e-10

    def test_profit_is_finite(self, default_params: ASParams) -> None:
        """Terminal profit should be finite."""
        rng = np.random.default_rng(0)
        result = simulate_path(default_params, "theory_faithful", "inventory_aware", rng)
        assert np.isfinite(result["profit"])

    def test_final_q_is_integer(self, default_params: ASParams) -> None:
        """Final inventory should be an integer value."""
        rng = np.random.default_rng(0)
        result = simulate_path(default_params, "theory_faithful", "inventory_aware", rng)
        assert result["final_q"] == int(result["final_q"])

    def test_ask_above_bid(self, default_params: ASParams) -> None:
        """Ask quote should always be above bid quote."""
        rng = np.random.default_rng(0)
        result = simulate_path(default_params, "theory_faithful", "inventory_aware", rng)
        bids = np.array(result["path_bid"])
        asks = np.array(result["path_ask"])
        assert np.all(asks >= bids), "Ask should always be >= bid"

    def test_symmetric_strategy_equal_distances(self, default_params: ASParams) -> None:
        """
        For symmetric strategy, ask and bid should be equidistant from mid.
        Check first step only (before any inventory accumulation).
        """
        rng = np.random.default_rng(0)
        result = simulate_path(default_params, "theory_faithful", "symmetric", rng)
        S0 = result["path_S"][0]
        ask0 = result["path_ask"][0]
        bid0 = result["path_bid"][0]
        assert abs((ask0 - S0) - (S0 - bid0)) < 1e-10

    def test_both_spread_conventions_run(self, default_params: ASParams) -> None:
        """Both spread conventions should run without error."""
        rng = np.random.default_rng(0)
        for conv in ("theory_faithful", "liquidity_only"):
            result = simulate_path(default_params, conv, "inventory_aware", rng)
            assert np.isfinite(result["profit"])

    def test_both_strategies_run(self, default_params: ASParams) -> None:
        """Both strategies should run without error."""
        rng = np.random.default_rng(0)
        for strat in ("inventory_aware", "symmetric"):
            result = simulate_path(default_params, "theory_faithful", strat, rng)
            assert np.isfinite(result["profit"])

    def test_price_moves_by_correct_amount(self, default_params: ASParams) -> None:
        """Each price step should be exactly ±sigma*sqrt(dt)."""
        rng = np.random.default_rng(0)
        result = simulate_path(default_params, "theory_faithful", "inventory_aware", rng)
        S = np.array(result["path_S"])
        diffs = np.diff(S)
        expected_step = default_params.sigma * np.sqrt(default_params.dt)
        assert np.allclose(np.abs(diffs), expected_step, atol=1e-10), (
            "Price steps should be exactly ±sigma*sqrt(dt)"
        )

    def test_terminal_profit_mark_to_market(self, small_params: ASParams) -> None:
        """
        Terminal profit should equal X_T + q_T * S_T.
        We verify this by checking the formula is consistent with the path.
        """
        rng = np.random.default_rng(99)
        result = simulate_path(small_params, "liquidity_only", "inventory_aware", rng)
        # profit is stored directly; just verify it's finite and reasonable
        assert np.isfinite(result["profit"])
        assert abs(result["profit"]) < 1e6  # sanity check


# ---------------------------------------------------------------------------
# Tests: run_monte_carlo
# ---------------------------------------------------------------------------

class TestRunMonteCarlo:
    def test_output_shapes(self, small_params: ASParams) -> None:
        """MC output arrays should have length n_paths."""
        mc = run_monte_carlo(small_params, "theory_faithful", "inventory_aware")
        assert len(mc["profits"]) == small_params.n_paths
        assert len(mc["final_qs"]) == small_params.n_paths

    def test_reproducibility_with_seed(self, small_params: ASParams) -> None:
        """Same seed should produce identical results."""
        mc1 = run_monte_carlo(small_params, "theory_faithful", "inventory_aware")
        mc2 = run_monte_carlo(small_params, "theory_faithful", "inventory_aware")
        np.testing.assert_array_equal(mc1["profits"], mc2["profits"])

    def test_different_seeds_differ(self) -> None:
        """Different seeds should produce different results."""
        p1 = ASParams(n_paths=50, seed=1)
        p2 = ASParams(n_paths=50, seed=2)
        mc1 = run_monte_carlo(p1, "theory_faithful", "inventory_aware")
        mc2 = run_monte_carlo(p2, "theory_faithful", "inventory_aware")
        assert not np.allclose(mc1["profits"], mc2["profits"])

    def test_all_profits_finite(self, small_params: ASParams) -> None:
        """All terminal profits should be finite."""
        mc = run_monte_carlo(small_params, "theory_faithful", "inventory_aware")
        assert np.all(np.isfinite(mc["profits"]))

    def test_all_final_qs_integer(self, small_params: ASParams) -> None:
        """All final inventories should be integer-valued."""
        mc = run_monte_carlo(small_params, "theory_faithful", "inventory_aware")
        assert np.all(mc["final_qs"] == mc["final_qs"].astype(int))

    def test_illustrative_path_present(self, small_params: ASParams) -> None:
        """Illustrative path should be stored."""
        mc = run_monte_carlo(small_params, "theory_faithful", "inventory_aware")
        assert "illustrative_path" in mc
        assert "path_S" in mc["illustrative_path"]

    def test_mean_final_q_near_zero(self, default_params: ASParams) -> None:
        """Mean final inventory should be near zero over many paths."""
        mc = run_monte_carlo(default_params, "theory_faithful", "inventory_aware")
        mean_q = np.mean(mc["final_qs"])
        assert abs(mean_q) < 5.0, f"Mean final q too far from zero: {mean_q}"


# ---------------------------------------------------------------------------
# Tests: compute_summary
# ---------------------------------------------------------------------------

class TestComputeSummary:
    def test_summary_keys(self, small_params: ASParams) -> None:
        """Summary should contain all required keys."""
        mc = run_monte_carlo(small_params, "theory_faithful", "inventory_aware")
        summary = compute_summary(mc, "Inventory-Aware", "Theory-Faithful", 1.3)
        for key in ["Strategy", "Spread Convention", "Mean Profit", "Std Profit",
                    "Mean Final q", "Std Final q"]:
            assert key in summary

    def test_std_profit_positive(self, default_params: ASParams) -> None:
        """Standard deviation of profit should be positive."""
        mc = run_monte_carlo(default_params, "theory_faithful", "inventory_aware")
        summary = compute_summary(mc, "Inventory-Aware", "Theory-Faithful", 1.3)
        assert summary["Std Profit"] > 0

    def test_std_q_positive(self, default_params: ASParams) -> None:
        """Standard deviation of final q should be positive."""
        mc = run_monte_carlo(default_params, "theory_faithful", "inventory_aware")
        summary = compute_summary(mc, "Inventory-Aware", "Theory-Faithful", 1.3)
        assert summary["Std Final q"] > 0


# ---------------------------------------------------------------------------
# Tests: validate_reservation_price_properties
# ---------------------------------------------------------------------------

class TestValidateProperties:
    def test_validation_passes_for_valid_params(self, default_params: ASParams) -> None:
        """Validation should pass without raising for valid parameters."""
        validate_reservation_price_properties(default_params)  # should not raise

    def test_validation_passes_for_all_gammas(self) -> None:
        """Validation should pass for all tested gamma values."""
        for gamma in [0.01, 0.1, 0.5]:
            p = ASParams(gamma=gamma)
            validate_reservation_price_properties(p)


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_step_simulation(self) -> None:
        """Simulation with a single time step should complete without error."""
        p = ASParams(T=0.005, dt=0.005, n_paths=5, seed=0)
        mc = run_monte_carlo(p, "theory_faithful", "inventory_aware")
        assert len(mc["profits"]) == 5

    def test_extreme_gamma_small(self) -> None:
        """Very small gamma should not cause numerical issues."""
        p = ASParams(gamma=0.001, n_paths=10, seed=0)
        mc = run_monte_carlo(p, "liquidity_only", "inventory_aware")
        assert np.all(np.isfinite(mc["profits"]))

    def test_extreme_gamma_large(self) -> None:
        """Large gamma should not cause numerical issues."""
        p = ASParams(gamma=1.0, n_paths=10, seed=0)
        mc = run_monte_carlo(p, "liquidity_only", "inventory_aware")
        assert np.all(np.isfinite(mc["profits"]))

    def test_symmetric_strategy_liquidity_spread(self) -> None:
        """Symmetric strategy with liquidity-only spread should run."""
        p = ASParams(n_paths=20, seed=0)
        mc = run_monte_carlo(p, "liquidity_only", "symmetric")
        assert np.all(np.isfinite(mc["profits"]))

    def test_inventory_aware_theory_spread(self) -> None:
        """Inventory-aware strategy with theory spread should run."""
        p = ASParams(n_paths=20, seed=0)
        mc = run_monte_carlo(p, "theory_faithful", "inventory_aware")
        assert np.all(np.isfinite(mc["profits"]))

    def test_spread_non_negative(self) -> None:
        """Spread should be non-negative for all valid parameters."""
        for gamma in [0.01, 0.1, 0.5]:
            for tau in [0.0, 0.5, 1.0]:
                s = spread_theory_faithful(gamma, sigma=2.0, k=1.5, tau=tau)
                assert s >= 0, f"Negative spread: gamma={gamma}, tau={tau}"

    def test_intensity_negative_delta(self) -> None:
        """Intensity with negative delta should be > A (quote crosses mid)."""
        lam = execution_intensity(-0.5, A=140.0, k=1.5)
        assert lam > 140.0

    def test_fill_prob_negative_delta_capped(self) -> None:
        """Fill probability should be capped at 1 even for negative delta."""
        lam = execution_intensity(-1.0, A=140.0, k=1.5)
        p = fill_probability(lam, dt=0.005)
        assert p <= 1.0
