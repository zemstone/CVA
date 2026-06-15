"""Tests for tipping-point curve analysis using synthetic curves with known shapes."""
from __future__ import annotations

import numpy as np

from ai_stroke.analysis.tipping_point import analyze_degradation_curve


def test_cliff_curve_is_detected() -> None:
    """A curve that holds high then suddenly crashes -> 'cliff'."""
    ratios = np.linspace(0, 1, 11)
    acc = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
    res = analyze_degradation_curve(ratios, acc, chance_acc=0.1)
    assert res.curve_type == "cliff"
    assert np.isclose(res.max_drop_ratio, 0.6)  # crash happens at 0.6


def test_linear_curve_is_detected() -> None:
    """A steadily declining curve -> 'linear'."""
    ratios = np.linspace(0, 1, 11)
    acc = np.linspace(0.9, 0.1, 11)
    res = analyze_degradation_curve(ratios, acc, chance_acc=0.1)
    assert res.curve_type == "linear"


def test_graceful_curve_holds_long() -> None:
    """A curve that stays high until very late -> 'graceful' (high AUC)."""
    ratios = np.linspace(0, 1, 11)
    acc = np.array([0.9, 0.9, 0.89, 0.88, 0.87, 0.86, 0.85, 0.83, 0.8, 0.6, 0.2])
    res = analyze_degradation_curve(ratios, acc, chance_acc=0.1)
    assert res.curve_type == "graceful"
    assert res.auc > 0.7  # robust: large area under accuracy curve


def test_no_collapse_returns_none_tipping() -> None:
    """A curve that never drops below threshold has no tipping ratio."""
    ratios = np.linspace(0, 1, 11)
    acc = np.full(11, 0.9)
    res = analyze_degradation_curve(ratios, acc, chance_acc=0.1)
    assert res.tipping_ratio is None
