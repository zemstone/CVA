"""Tests for targeted (importance-ranked) ablation."""
from __future__ import annotations

import torch
import torch.nn as nn

from ai_stroke.lesion.targeted_ablation import TargetedAblation, _channel_importance


def _conv_with_known_importance() -> nn.Conv2d:
    """Conv where output channel k has weight magnitude proportional to k."""
    conv = nn.Conv2d(4, 5, kernel_size=3)
    with torch.no_grad():
        for k in range(5):
            conv.weight[k] = float(k)   # channel 0 = least, channel 4 = most
        conv.bias.zero_()
    return conv


def test_importance_ranking_is_correct() -> None:
    conv = _conv_with_known_importance()
    imp = _channel_importance(conv).tolist()
    assert imp == sorted(imp)  # strictly increasing by construction


def test_most_important_kills_top_channels() -> None:
    conv = _conv_with_known_importance()
    strat = TargetedAblation(mode="most_important")
    killed = set(strat.select_channels(conv, n_kill=2, seed=0).tolist())
    assert killed == {3, 4}   # the two highest-magnitude channels


def test_least_important_kills_bottom_channels() -> None:
    conv = _conv_with_known_importance()
    strat = TargetedAblation(mode="least_important")
    killed = set(strat.select_channels(conv, n_kill=2, seed=0).tolist())
    assert killed == {0, 1}   # the two lowest-magnitude channels


def test_targeted_is_deterministic_across_seeds() -> None:
    conv = _conv_with_known_importance()
    strat = TargetedAblation(mode="most_important")
    a = strat.select_channels(conv, n_kill=3, seed=0)
    b = strat.select_channels(conv, n_kill=3, seed=999)
    assert set(a.tolist()) == set(b.tolist())  # seed-independent
