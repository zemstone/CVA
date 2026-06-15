"""Visualization for Experiment 1: the layer-criticality heatmap."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_criticality_heatmap(
    df: pd.DataFrame, out_path: str | Path, title: str = "Layer Criticality Map"
) -> Path:
    """Heatmap of mean accuracy drop over (layer x damage_ratio).

    Args:
        df: Tidy results from run_layer_sensitivity.
        out_path: Where to save the figure (PNG).
    """
    pivot = (
        df.groupby(["target_layer", "damage_ratio"])["accuracy_drop"]
        .mean()
        .reset_index()
        .pivot(index="target_layer", columns="damage_ratio", values="accuracy_drop")
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="Reds",
        cbar_kws={"label": "Mean accuracy drop"}, ax=ax,
    )
    ax.set_xlabel("Damage ratio")
    ax.set_ylabel("Target layer")
    ax.set_title(title)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
