"""Abstract interface for all 'stroke' (lesion) strategies.

Design rule: a lesion strategy NEVER retrains. It only damages weights and is
fully reversible via a context manager, so experiment 1 measures the pure
effect of damage with no recovery contamination.
"""
from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterator

import torch
import torch.nn as nn


@dataclass
class LesionResult:
    """Bookkeeping for an applied lesion (for logging / reproducibility)."""

    target_layer: str
    damage_ratio: float
    num_units_total: int
    num_units_killed: int
    seed: int
    extra: Dict[str, float] = field(default_factory=dict)


class LesionStrategy(ABC):
    """Base class for a reversible weight-damage strategy.

    Subclasses implement ``_apply`` (zero-out selected units) and must store
    enough state to restore the original weights afterwards.
    """

    def __init__(self) -> None:
        self._backup: Dict[str, torch.Tensor] = {}

    @abstractmethod
    def _apply(
        self, model: nn.Module, target_layer: str, damage_ratio: float, seed: int
    ) -> LesionResult:
        """Damage ``target_layer`` in-place and return metadata."""

    def _backup_layer(self, layer: nn.Module, key: str) -> None:
        """Clone original weights so the lesion can be undone."""
        self._backup[key] = layer.weight.detach().clone()

    def _restore(self, model: nn.Module) -> None:
        """Restore all backed-up weights (heal the brain)."""
        modules = dict(model.named_modules())
        for key, weight in self._backup.items():
            with torch.no_grad():
                modules[key].weight.copy_(weight)
        self._backup.clear()

    @contextlib.contextmanager
    def lesion(
        self, model: nn.Module, target_layer: str, damage_ratio: float, seed: int
    ) -> Iterator[LesionResult]:
        """Context manager: apply lesion, yield metadata, then auto-restore.

        Example:
            with strategy.lesion(model, "features.0", 0.3, seed=0) as info:
                acc = evaluate(model)   # measured WITH damage
            # weights are automatically healed here
        """
        result = self._apply(model, target_layer, damage_ratio, seed)
        try:
            yield result
        finally:
            self._restore(model)
