"""Strategy-comparison plot for RQ2-extended (random vs targeted)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

def plot_degradation_curves(
    curve_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    out_path: str | Path,
    chance_acc: float = 0.1,
    title: str = "Degradation Curves (Tipping Point)",
) -> Path:
    """Plot accuracy vs damage ratio per layer, with std bands and tipping marks.

    Args:
        curve_df: per-(layer, ratio) mean/std accuracy.
        summary_df: per-layer tipping characterization (for vertical markers).
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    tipping_by_layer = dict(
        zip(summary_df["target_layer"], summary_df["tipping_ratio"])
    )
    type_by_layer = dict(zip(summary_df["target_layer"], summary_df["curve_type"]))

    for layer, grp in curve_df.groupby("target_layer"):
        grp = grp.sort_values("damage_ratio")
        x = grp["damage_ratio"].to_numpy()
        mean = grp["accuracy_mean"].to_numpy()
        std = grp["accuracy_std"].to_numpy()

        ctype = type_by_layer.get(layer, "?")
        line, = ax.plot(x, mean, marker="o", ms=3, label=f"{layer} ({ctype})")
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=line.get_color())

        # mark the tipping point if it exists.
        tip = tipping_by_layer.get(layer)
        if tip is not None:
            ax.axvline(tip, color=line.get_color(), ls=":", lw=1, alpha=0.7)

    ax.axhline(chance_acc, color="gray", ls="--", lw=1, label=f"chance ({chance_acc:.2f})")
    ax.set_xlabel("Damage ratio")
    ax.set_ylabel("Accuracy (mean ± std)")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def plot_strategy_comparison(
    curve_df: pd.DataFrame,
    out_path: str | Path,
    chance_acc: float = 0.1,
    title: str = "Damage Modality Comparison (random vs targeted)",
) -> Path:
    """One subplot per layer; one line per strategy, with std bands.

    Makes the core RQ2-extended contrast obvious: at the same damage ratio,
    'targeted-critical' should collapse first, 'targeted-redundant' last.
    """
    layers = sorted(curve_df["target_layer"].unique())
    n = len(layers)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows),
        squeeze=False, sharey=True,
    )

    for i, layer in enumerate(layers):
        ax = axes[i // ncols][i % ncols]
        sub = curve_df[curve_df["target_layer"] == layer]
        for strat, grp in sub.groupby("strategy"):
            grp = grp.sort_values("damage_ratio")
            x = grp["damage_ratio"].to_numpy()
            mean = grp["accuracy_mean"].to_numpy()
            std = grp["accuracy_std"].to_numpy()
            line, = ax.plot(x, mean, marker="o", ms=3, label=strat)
            ax.fill_between(x, mean - std, mean + std, alpha=0.15,
                            color=line.get_color())
        ax.axhline(chance_acc, color="gray", ls="--", lw=1)
        ax.set_title(layer, fontsize=10)
        ax.set_xlabel("Damage ratio")
        ax.set_ylim(0, 1)
        if i % ncols == 0:
            ax.set_ylabel("Accuracy")
        ax.legend(fontsize=8)

    # Hide any unused axes.
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
