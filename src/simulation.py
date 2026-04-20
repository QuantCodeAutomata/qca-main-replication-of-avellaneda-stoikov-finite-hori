"""
Core simulation engine for the Avellaneda-Stoikov (2008) finite-horizon
market-making model.

Reference: Avellaneda, M. & Stoikov, S. (2008). High-frequency trading in a
limit order book. Quantitative Finance, 8(3), 217-224.

All formulas are implemented from scratch — Context7 confirmed no library
equivalent exists for the AS reservation-price market-making model.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class ASParams:
    """
    Model parameters for the Avellaneda-Stoikov simulation.

    Attributes
    ----------
    S0 : float
        Initial mid-price.
    T : float
        Horizon length (years or normalised time units).
    dt : float
        Discrete time step size.
    sigma : float
        Volatility of the mid-price process.
    A : float
        Baseline arrival intensity of market orders.
    k : float
        Sensitivity of arrival intensity to quote distance.
    gamma : float
        CARA risk-aversion coefficient.
    n_paths : int
        Number of Monte Carlo paths.
    X0 : float
        Initial cash position.
    q0 : int
        Initial inventory.
    seed : int | None
        Global random seed for reproducibility.
    """
    S0: float = 100.0
    T: float = 1.0
    dt: float = 0.005
    sigma: float = 2.0
    A: float = 140.0
    k: float = 1.5
    gamma: float = 0.1
    n_paths: int = 1000
    X0: float = 0.0
    q0: int = 0
    seed: int | None = 42

    @property
    def n_steps(self) -> int:
        """Number of discrete time steps per path."""
        return round(self.T / self.dt)

    def __post_init__(self) -> None:
        assert self.dt > 0, "dt must be positive"
        assert self.T > 0, "T must be positive"
        assert self.sigma > 0, "sigma must be positive"
        assert self.A > 0, "A must be positive"
        assert self.k > 0, "k must be positive"
        assert self.gamma > 0, "gamma must be positive"
        assert self.n_paths > 0, "n_paths must be positive"


# ---------------------------------------------------------------------------
# Core formula implementations  (Custom — Context7 found no library equivalent)
# ---------------------------------------------------------------------------

def reservation_price(S: float, q: float, gamma: float, sigma: float, tau: float) -> float:
    """
    Compute the reservation (indifference) price.

    Paper formula: r(s, q, t) = s - q * gamma * sigma^2 * tau
    where tau = T - t is the remaining time.

    Parameters
    ----------
    S : float
        Current mid-price.
    q : float
        Current inventory (signed).
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    tau : float
        Remaining time to horizon (T - t).

    Returns
    -------
    float
        Reservation price.
    """
    return S - q * gamma * sigma**2 * tau


def spread_theory_faithful(gamma: float, sigma: float, k: float, tau: float) -> float:
    """
    Full time-varying spread from the paper's approximate finite-horizon solution.

    Paper formula: spread(t) = gamma * sigma^2 * tau + (2/gamma) * ln(1 + gamma/k)

    This includes both the inventory-risk component (gamma*sigma^2*tau) and the
    liquidity component ((2/gamma)*ln(1 + gamma/k)).

    Parameters
    ----------
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Mid-price volatility.
    k : float
        Intensity sensitivity parameter.
    tau : float
        Remaining time to horizon.

    Returns
    -------
    float
        Total quoted spread.
    """
    # Custom — Context7 found no library equivalent (paper Eq. for full spread)
    inventory_risk_component = gamma * sigma**2 * tau
    liquidity_component = (2.0 / gamma) * np.log(1.0 + gamma / k)
    return inventory_risk_component + liquidity_component


def spread_liquidity_only(gamma: float, k: float) -> float:
    """
    Liquidity-only (constant-in-time) spread matching the paper's reported table values.

    Formula: spread = (2/gamma) * ln(1 + gamma/k)

    This omits the inventory-risk term gamma*sigma^2*tau, yielding a constant
    spread that matches the paper's quoted spread values in the results table.

    Parameters
    ----------
    gamma : float
        Risk-aversion coefficient.
    k : float
        Intensity sensitivity parameter.

    Returns
    -------
    float
        Constant liquidity-only spread.
    """
    # Custom — Context7 found no library equivalent (paper liquidity component)
    return (2.0 / gamma) * np.log(1.0 + gamma / k)


def execution_intensity(delta: float, A: float, k: float) -> float:
    """
    Compute the Poisson arrival intensity for a quote at distance delta from mid.

    Paper formula: lambda(delta) = A * exp(-k * delta)

    Parameters
    ----------
    delta : float
        Quote distance from mid-price (can be negative if quote crosses mid).
    A : float
        Baseline arrival intensity.
    k : float
        Sensitivity of intensity to quote distance.

    Returns
    -------
    float
        Arrival intensity (events per unit time).
    """
    return A * np.exp(-k * delta)


def fill_probability(intensity: float, dt: float) -> float:
    """
    Convert a Poisson intensity to a Bernoulli fill probability over one step.

    Approximation: p_fill = lambda * dt, clipped to [0, 1].

    NOTE: Clipping at 1.0 is an implementation necessity when delta becomes
    negative or very small, causing lambda*dt > 1. This is not explicitly
    specified in the paper but is required for a valid probability. This
    practical resolution is documented as required by the methodology.

    Parameters
    ----------
    intensity : float
        Poisson arrival intensity.
    dt : float
        Time step size.

    Returns
    -------
    float
        Bernoulli fill probability in [0, 1].
    """
    return float(np.clip(intensity * dt, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Single-path simulation
# ---------------------------------------------------------------------------

def simulate_path(
    params: ASParams,
    spread_convention: Literal["theory_faithful", "liquidity_only"],
    strategy: Literal["inventory_aware", "symmetric"],
    rng: np.random.Generator,
) -> dict[str, float | list]:
    """
    Simulate one Monte Carlo path of the AS market-making model.

    Chronological order per step:
      1. Compute tau = T - t.
      2. Compute spread (depends on convention).
      3. Compute reservation price (inventory-aware) or use S (symmetric).
      4. Set bid/ask quotes.
      5. Compute fill probabilities.
      6. Draw independent Bernoulli fills for ask and bid sides.
      7. Update cash X and inventory q.
      8. Update mid-price S by ±sigma*sqrt(dt) with equal probability.
      9. Increment t by dt.
     10. At T, compute terminal P&L: Pi_T = X_T + q_T * S_T.

    Parameters
    ----------
    params : ASParams
        Model parameters.
    spread_convention : str
        "theory_faithful" uses the full time-varying spread;
        "liquidity_only" uses the constant liquidity component only.
    strategy : str
        "inventory_aware" centres quotes on reservation price;
        "symmetric" centres quotes on mid-price.
    rng : np.random.Generator
        Pre-seeded random number generator.

    Returns
    -------
    dict
        Keys: "profit" (Pi_T), "final_q" (q_T), "path_S", "path_r",
              "path_bid", "path_ask", "path_t".
    """
    p = params
    sqrt_dt = np.sqrt(p.dt)

    S = float(p.S0)
    X = float(p.X0)
    q = int(p.q0)
    t = 0.0

    # Storage for illustrative path (all steps)
    path_S: list[float] = [S]
    path_r: list[float] = [S]  # reservation price at t=0 equals S when q=0
    path_bid: list[float] = []
    path_ask: list[float] = []
    path_t: list[float] = [t]

    for _ in range(p.n_steps):
        tau = p.T - t

        # --- Spread computation ---
        if spread_convention == "theory_faithful":
            spread = spread_theory_faithful(p.gamma, p.sigma, p.k, tau)
        else:  # liquidity_only
            spread = spread_liquidity_only(p.gamma, p.k)

        # --- Quote centre ---
        if strategy == "inventory_aware":
            r = reservation_price(S, q, p.gamma, p.sigma, tau)
            centre = r
        else:  # symmetric
            r = reservation_price(S, q, p.gamma, p.sigma, tau)  # still computed for path recording
            centre = S

        p_ask = centre + 0.5 * spread
        p_bid = centre - 0.5 * spread

        # --- Quote distances from mid ---
        delta_a = p_ask - S
        delta_b = S - p_bid

        # --- Execution intensities and fill probabilities ---
        lam_a = execution_intensity(delta_a, p.A, p.k)
        lam_b = execution_intensity(delta_b, p.A, p.k)
        prob_a = fill_probability(lam_a, p.dt)
        prob_b = fill_probability(lam_b, p.dt)

        # --- Independent Bernoulli fills (both sides can fill in same step) ---
        fill_ask = rng.random() < prob_a
        fill_bid = rng.random() < prob_b

        if fill_ask:
            q -= 1
            X += p_ask
        if fill_bid:
            q += 1
            X -= p_bid

        # --- Mid-price update: binomial ±sigma*sqrt(dt) ---
        price_move = sqrt_dt * p.sigma * (1 if rng.random() < 0.5 else -1)
        S += price_move
        t += p.dt

        # Record path
        path_S.append(S)
        path_r.append(r)
        path_bid.append(p_bid)
        path_ask.append(p_ask)
        path_t.append(t)

    # Terminal mark-to-market profit
    Pi_T = X + q * S

    return {
        "profit": Pi_T,
        "final_q": float(q),
        "path_S": path_S,
        "path_r": path_r,
        "path_bid": path_bid,
        "path_ask": path_ask,
        "path_t": path_t,
    }


# ---------------------------------------------------------------------------
# Multi-path Monte Carlo runner
# ---------------------------------------------------------------------------

def run_monte_carlo(
    params: ASParams,
    spread_convention: Literal["theory_faithful", "liquidity_only"],
    strategy: Literal["inventory_aware", "symmetric"],
) -> dict[str, np.ndarray | dict]:
    """
    Run a full Monte Carlo simulation over n_paths independent paths.

    Seed policy: a single global seed is set at the start; each path uses
    the next state of the same RNG, ensuring reproducibility while maintaining
    independence across paths. Common random numbers are NOT used across
    strategies (each strategy gets its own seeded RNG from the same root seed).

    Parameters
    ----------
    params : ASParams
        Model parameters including n_paths and seed.
    spread_convention : str
        "theory_faithful" or "liquidity_only".
    strategy : str
        "inventory_aware" or "symmetric".

    Returns
    -------
    dict
        Keys: "profits" (array of Pi_T), "final_qs" (array of q_T),
              "illustrative_path" (dict from last path for plotting).
    """
    rng = np.random.default_rng(params.seed)

    profits = np.empty(params.n_paths)
    final_qs = np.empty(params.n_paths)
    illustrative_path: dict = {}

    for i in range(params.n_paths):
        result = simulate_path(params, spread_convention, strategy, rng)
        profits[i] = result["profit"]
        final_qs[i] = result["final_q"]
        if i == 0:
            illustrative_path = result  # save first path for plotting

    return {
        "profits": profits,
        "final_qs": final_qs,
        "illustrative_path": illustrative_path,
    }


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary(
    mc_result: dict,
    strategy_label: str,
    spread_label: str,
    spread_value: float,
) -> dict[str, float | str]:
    """
    Compute summary statistics from a Monte Carlo result.

    Parameters
    ----------
    mc_result : dict
        Output from run_monte_carlo.
    strategy_label : str
        Human-readable strategy name.
    spread_label : str
        Human-readable spread convention name.
    spread_value : float
        Representative spread value (e.g., at t=0 or constant).

    Returns
    -------
    dict
        Summary statistics dictionary.
    """
    profits = mc_result["profits"]
    final_qs = mc_result["final_qs"]
    return {
        "Strategy": strategy_label,
        "Spread Convention": spread_label,
        "Quoted Spread": round(spread_value, 4),
        "Mean Profit": round(float(np.mean(profits)), 4),
        "Std Profit": round(float(np.std(profits, ddof=1)), 4),
        "Mean Final q": round(float(np.mean(final_qs)), 4),
        "Std Final q": round(float(np.std(final_qs, ddof=1)), 4),
    }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_reservation_price_properties(params: ASParams) -> None:
    """
    Validate key theoretical properties of the reservation price.

    Checks:
    1. r(s, 0, t) = s  (zero inventory → reservation price equals mid)
    2. r decreases linearly with q (positive inventory → lower reservation price)
    3. Spread shrinks toward liquidity component as tau → 0

    Parameters
    ----------
    params : ASParams
        Model parameters.

    Raises
    ------
    AssertionError
        If any theoretical property is violated.
    """
    S, gamma, sigma, k = params.S0, params.gamma, params.sigma, params.k

    # Property 1: zero inventory
    r_zero = reservation_price(S, 0, gamma, sigma, tau=0.5)
    assert abs(r_zero - S) < 1e-10, f"r(s,0,t) != s: got {r_zero}"

    # Property 2: linearity in q
    r_q1 = reservation_price(S, 1, gamma, sigma, tau=0.5)
    r_q2 = reservation_price(S, 2, gamma, sigma, tau=0.5)
    r_q0 = reservation_price(S, 0, gamma, sigma, tau=0.5)
    assert r_q1 < r_q0, "r should decrease with positive inventory"
    assert abs((r_q0 - r_q1) - (r_q1 - r_q2)) < 1e-10, "r should be linear in q"

    # Property 3: spread at tau=0 equals liquidity component only
    spread_at_T = spread_theory_faithful(gamma, sigma, k, tau=0.0)
    liq_component = spread_liquidity_only(gamma, k)
    assert abs(spread_at_T - liq_component) < 1e-10, (
        f"Spread at tau=0 should equal liquidity component: "
        f"{spread_at_T} vs {liq_component}"
    )

    # Property 4: intensity decreases with larger delta
    lam_small = execution_intensity(0.5, params.A, k)
    lam_large = execution_intensity(1.0, params.A, k)
    assert lam_small > lam_large, "Intensity should decrease with larger delta"
