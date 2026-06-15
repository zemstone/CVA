"""Targeted ablation: remove channels ranked by importance (focal damage).

Two modes simulate opposite 'stroke' profiles:

- ``most_important`` : kill the highest-importance channels first. This is the
  ANN analogue of a stroke striking a functionally critical region -> expected
  rapid, cliff-like collapse.
- ``least_important``: kill the lowest-importance channels first -> probes how
  much redundancy / 'reserve' the layer carries (expected graceful decline).

Importance is the L-norm of each output channel's weight tensor (a standard,
training-free filter-importance proxy used in network pruning).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ai_stroke.lesion.base import LesionStrategy
from ai_stroke.lesion.registry import LESION_REGISTRY


def _channel_importance(module: nn.Module, p: float = 1.0) -> torch.Tensor:
    """Per-output-channel importance = L_p norm of its weight slice.

    For Conv2d weight [out, in, kh, kw] and Linear weight [out, in], we flatten
    everything but dim 0 and take the norm -> one scalar per output unit.
    """
    w = module.weight.detach()
    return w.reshape(w.shape[0], -1).norm(p=p, dim=1)


@LESION_REGISTRY.register("targeted_ablation")
class TargetedAblation(LesionStrategy):
    """Importance-ranked ablation (focal / functionally-targeted damage)."""

    name = "targeted_ablation"

    def __init__(self, mode: str = "most_important", p: float = 1.0) -> None:
        """Args:
            mode: 'most_important' (kill critical channels first) or
                  'least_important' (kill redundant channels first).
            p: Order of the L_p norm used as the importance score.
        """
        if mode not in ("most_important", "least_important"):
            raise ValueError(f"Unknown mode: {mode!r}")
        self.mode = mode
        self.p = p
        # Make the registry key informative when logged.
        self.name = f"targeted_ablation::{mode}"

    def select_channels(
        self, module: nn.Module, n_kill: int, seed: int
    ) -> np.ndarray:
        """Deterministic: rank by importance, take the top/bottom ``n_kill``.

        ``seed`` is unused (selection is deterministic), but kept in the
        signature so the strategy is drop-in compatible with the sweep engine.
        """
        importance = _channel_importance(module, p=self.p).cpu().numpy()
        order = np.argsort(importance)  # ascending: least -> most important
        if self.mode == "most_important":
            return order[-n_kill:]      # top-importance channels
        return order[:n_kill]           # bottom-importance channels
