"""
Visualization utilities for the Avellaneda-Stoikov replication experiments.

Produces:
  - Profit histograms comparing inventory-aware vs symmetric strategies.
  - Illustrative path plots showing S_t, reservation price, bid, and ask.
  - Summary tables saved as CSV and Markdown.

Uses matplotlib for all plots.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Histogram plots
# ---------------------------------------------------------------------------

def plot_profit_histograms(
    results_inv: dict[str, Any],
    results_sym: dict[str, Any],
    gamma: float,
    experiment_label: str,
    output_dir: Path,
) -> Path:
    """
    Plot overlapping histograms of terminal profit for inventory-aware and
    symmetric strategies.

    Parameters
    ----------
    results_inv : dict
        Monte Carlo results for the inventory-aware strategy.
    results_sym : dict
        Monte Carlo results for the symmetric strategy.
    gamma : float
        Risk-aversion coefficient (used in title and filename).
    experiment_label : str
        Short label for the experiment (e.g., "exp1_theory" or "exp2_liquidity").
    output_dir : Path
        Directory where the figure will be saved.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    profits_inv = results_inv["profits"]
    profits_sym = results_sym["profits"]

    fig, ax = plt.subplots(figsize=(9, 5))

    bins = np.linspace(
        min(profits_inv.min(), profits_sym.min()),
        max(profits_inv.max(), profits_sym.max()),
        50,
    )

    ax.hist(
        profits_inv,
        bins=bins,
        alpha=0.6,
        color="steelblue",
        label=f"Inventory-Aware  μ={np.mean(profits_inv):.2f}  σ={np.std(profits_inv, ddof=1):.2f}",
        density=True,
    )
    ax.hist(
        profits_sym,
        bins=bins,
        alpha=0.6,
        color="tomato",
        label=f"Symmetric        μ={np.mean(profits_sym):.2f}  σ={np.std(profits_sym, ddof=1):.2f}",
        density=True,
    )

    ax.set_xlabel("Terminal Profit  Π_T", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"Terminal Profit Distribution — γ = {gamma}\n({experiment_label})",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = output_dir / f"hist_profit_{experiment_label}_gamma{gamma}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Illustrative path plot
# ---------------------------------------------------------------------------

def plot_illustrative_path(
    path_data: dict[str, list],
    gamma: float,
    experiment_label: str,
    output_dir: Path,
) -> Path:
    """
    Plot one illustrative simulation path showing mid-price, reservation price,
    bid quote, and ask quote over time.

    Parameters
    ----------
    path_data : dict
        Output from simulate_path containing path_t, path_S, path_r,
        path_bid, path_ask.
    gamma : float
        Risk-aversion coefficient.
    experiment_label : str
        Short label for the experiment.
    output_dir : Path
        Directory where the figure will be saved.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    t = path_data["path_t"]
    S = path_data["path_S"]
    r = path_data["path_r"]
    bid = path_data["path_bid"]
    ask = path_data["path_ask"]

    # bid/ask have n_steps entries (computed before each price update)
    # S and r have n_steps+1 entries (include initial value)
    # Align: use t[:-1] for bid/ask, t for S and r
    t_quotes = t[:-1]  # time at which quotes were set

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(t, S, color="black", linewidth=1.2, label="Mid-price $S_t$", zorder=3)
    ax.plot(t, r, color="purple", linewidth=1.0, linestyle="--",
            label="Reservation price $r_t$", zorder=3)
    ax.plot(t_quotes, ask, color="green", linewidth=0.8, linestyle="-.",
            label="Ask quote $p^a_t$", alpha=0.85)
    ax.plot(t_quotes, bid, color="red", linewidth=0.8, linestyle="-.",
            label="Bid quote $p^b_t$", alpha=0.85)

    ax.set_xlabel("Time $t$", fontsize=12)
    ax.set_ylabel("Price", fontsize=12)
    ax.set_title(
        f"Illustrative Path — Inventory-Aware Strategy  γ = {gamma}\n({experiment_label})",
        fontsize=13,
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = output_dir / f"path_{experiment_label}_gamma{gamma}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def save_summary_table(
    rows: list[dict],
    experiment_label: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    """
    Save a summary statistics table as both CSV and Markdown.

    Parameters
    ----------
    rows : list[dict]
        List of summary dictionaries (one per strategy-gamma combination).
    experiment_label : str
        Short label for the experiment.
    output_dir : Path
        Directory where files will be saved.

    Returns
    -------
    tuple[Path, Path]
        Paths to the CSV and Markdown files.
    """
    df = pd.DataFrame(rows)

    csv_path = output_dir / f"summary_{experiment_label}.csv"
    md_path = output_dir / f"summary_{experiment_label}.md"

    df.to_csv(csv_path, index=False)

    with open(md_path, "w") as f:
        f.write(f"## Summary Table — {experiment_label}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")

    return csv_path, md_path


# ---------------------------------------------------------------------------
# Multi-gamma comparison plot
# ---------------------------------------------------------------------------

def plot_gamma_comparison(
    summary_rows: list[dict],
    experiment_label: str,
    output_dir: Path,
) -> Path:
    """
    Bar chart comparing std(Profit) and std(Final q) across gamma values
    for both strategies.

    Parameters
    ----------
    summary_rows : list[dict]
        Summary statistics rows.
    experiment_label : str
        Short label for the experiment.
    output_dir : Path
        Directory where the figure will be saved.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    df = pd.DataFrame(summary_rows)
    gammas = sorted(df["gamma"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(gammas))
    width = 0.35

    for ax, metric, ylabel in zip(
        axes,
        ["Std Profit", "Std Final q"],
        ["std(Terminal Profit)", "std(Final Inventory)"],
    ):
        inv_vals = [
            df[(df["gamma"] == g) & (df["Strategy"] == "Inventory-Aware")][metric].values[0]
            for g in gammas
        ]
        sym_vals = [
            df[(df["gamma"] == g) & (df["Strategy"] == "Symmetric")][metric].values[0]
            for g in gammas
        ]

        ax.bar(x - width / 2, inv_vals, width, label="Inventory-Aware", color="steelblue", alpha=0.8)
        ax.bar(x + width / 2, sym_vals, width, label="Symmetric", color="tomato", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"γ={g}" for g in gammas])
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{ylabel} by γ", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Risk Metrics Comparison — {experiment_label}", fontsize=13)
    fig.tight_layout()
    fname = output_dir / f"gamma_comparison_{experiment_label}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname
