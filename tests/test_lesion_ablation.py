"""Unit tests for the ablation lesion. Damage logic MUST be tested because
every experiment depends on its correctness and reversibility.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ai_stroke.lesion.ablation import RandomNodeAblation


def _tiny_model() -> nn.Module:
    return nn.Sequential(nn.Conv2d(3, 10, 3, padding=1))  # name: "0"


def test_ablation_kills_correct_number_of_units() -> None:
    model = _tiny_model()
    strategy = RandomNodeAblation()
    with strategy.lesion(model, "0", damage_ratio=0.3, seed=0) as info:
        dead = (model[0].weight.view(10, -1).abs().sum(dim=1) == 0).sum().item()
    assert info.num_units_killed == 3
    assert dead == 3


def test_lesion_is_reversible() -> None:
    model = _tiny_model()
    original = model[0].weight.detach().clone()
    strategy = RandomNodeAblation()
    with strategy.lesion(model, "0", damage_ratio=0.5, seed=1):
        pass  # damage applied inside
    # After context exit, weights must be fully restored.
    assert torch.allclose(model[0].weight, original)


def test_same_seed_kills_same_units() -> None:
    model = _tiny_model()
    strategy = RandomNodeAblation()
    with strategy.lesion(model, "0", 0.4, seed=7):
        dead_a = (model[0].weight.view(10, -1).abs().sum(dim=1) == 0).nonzero().flatten()
    with strategy.lesion(model, "0", 0.4, seed=7):
        dead_b = (model[0].weight.view(10, -1).abs().sum(dim=1) == 0).nonzero().flatten()
    assert torch.equal(dead_a, dead_b)
