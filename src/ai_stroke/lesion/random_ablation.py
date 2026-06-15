"""Random node ablation (diffuse damage baseline)."""
from __future__ import annotations

import numpy as np
import torch.nn as nn

from ai_stroke.lesion.base import LesionStrategy
from ai_stroke.lesion.registry import LESION_REGISTRY


@LESION_REGISTRY.register("random_node_ablation")
class RandomNodeAblation(LesionStrategy):
    """Zero out a random subset of output units (uniform, diffuse damage)."""

    name = "random_node_ablation"

    def select_channels(
        self, module: nn.Module, n_kill: int, seed: int
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n_units = module.weight.shape[0]  # output dim is dim 0 for Conv/Linear
        return rng.choice(n_units, size=n_kill, replace=False)
