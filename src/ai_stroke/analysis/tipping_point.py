"""Tipping-point detection for degradation curves (RQ2).

Given accuracy as a function of damage ratio, we quantify *how* a model fails:

- ``tipping_ratio``  : smallest damage ratio where accuracy first drops below a
  'collapse' threshold (e.g. half-way between healthy and chance). This is the
  ANN analogue of exhausting 'cognitive reserve'.
- ``max_drop_ratio`` : damage ratio of the steepest single-step accuracy drop
  (the 'cliff' location).
- ``curve_type``     : qualitative shape -> 'graceful' | 'linear' | 'cliff'.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TippingPointResult:
    """Summary of a single degradation curve."""

    tipping_ratio: Optional[float]   # None if it never collapses
    max_drop_ratio: float            # where the steepest drop occurs
    max_drop_value: float            # magnitude of that steepest drop
    curve_type: str                  # 'graceful' | 'linear' | 'cliff'
    auc: float                       # area under accuracy curve (robustness)


def _collapse_threshold(healthy_acc: float, chance_acc: float) -> float:
    """Accuracy level that counts as 'collapsed': midpoint healthy<->chance."""
    return chance_acc + 0.5 * (healthy_acc - chance_acc)


def analyze_degradation_curve(
    ratios: np.ndarray,
    accuracies: np.ndarray,
    chance_acc: float = 0.1,
    cliff_factor: float = 2.5,
) -> TippingPointResult:
    """Characterize one degradation curve.

    Args:
        ratios: Monotonically increasing damage ratios (e.g. 0.0 .. 1.0).
        accuracies: Mean accuracy at each ratio (same length as ``ratios``).
        chance_acc: Random-guess accuracy (1/num_classes -> 0.1 for CIFAR-10).
        cliff_factor: If the steepest drop is >= cliff_factor * the average
            per-step drop, the curve is labelled a 'cliff' (sudden collapse).

    Returns:
        TippingPointResult with the tipping ratio, cliff location, and shape.
    """
    ratios = np.asarray(ratios, dtype=float)
    accuracies = np.asarray(accuracies, dtype=float)
    if ratios.shape != accuracies.shape:
        raise ValueError("ratios and accuracies must share the same shape.")
    if not np.all(np.diff(ratios) > 0):
        raise ValueError("ratios must be strictly increasing.")

    healthy_acc = float(accuracies[0])
    threshold = _collapse_threshold(healthy_acc, chance_acc)

    # 1) tipping point: first ratio where accuracy dips below the threshold.
    below = np.where(accuracies < threshold)[0]
    tipping_ratio = float(ratios[below[0]]) if below.size > 0 else None

    # 2) steepest single-step drop (the 'cliff').
    step_drops = -np.diff(accuracies)            # positive = accuracy fell
    max_idx = int(np.argmax(step_drops))
    max_drop_value = float(step_drops[max_idx])
    # cliff is located at the *right* edge of the steepest segment.
    max_drop_ratio = float(ratios[max_idx + 1])

    # 3) curve shape classification.
    mean_drop = float(np.mean(np.clip(step_drops, 0, None))) or 1e-9
    total_drop = healthy_acc - chance_acc
    if total_drop <= 1e-6:
        curve_type = "graceful"
    elif max_drop_value >= cliff_factor * mean_drop:
        curve_type = "cliff"           # one dominant sudden collapse
    elif tipping_ratio is not None and tipping_ratio < 0.3:
        curve_type = "linear"          # degrades early & steadily
    else:
        curve_type = "graceful"        # holds up, then declines gently

    # 4) robustness as normalized area under the accuracy curve.
    auc = float(np.trapz(accuracies, ratios) / (ratios[-1] - ratios[0]))

    return TippingPointResult(
        tipping_ratio=tipping_ratio,
        max_drop_ratio=max_drop_ratio,
        max_drop_value=max_drop_value,
        curve_type=curve_type,
        auc=auc,
    )
