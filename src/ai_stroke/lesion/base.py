"""Base class for reversible lesion strategies.

A lesion zeroes out a fraction of a layer's output units (Conv2d output
channels or Linear output features), then restores the original weights on
exit -- guaranteeing no cross-contamination between experimental runs.

Subclasses customize ONLY *which* channels to remove via ``select_channels``.
All weight backup / zero-out / restore plumbing lives here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn


def _get_module(model: nn.Module, name: str) -> nn.Module:
    """Resolve a dotted module path (e.g. 'features.0') to the module object."""
    module = model
    for part in name.split("."):
        module = getattr(module, part)
    return module


def _num_output_units(module: nn.Module) -> int:
    """Number of output channels (Conv2d) or output features (Linear)."""
    if isinstance(module, nn.Conv2d):
        return module.out_channels
    if isinstance(module, nn.Linear):
        return module.out_features
    raise TypeError(f"Unsupported layer type for lesion: {type(module).__name__}")


class LesionStrategy(ABC):
    """Reversible lesion applied to one layer's output units."""

    name: str = "base"

    @abstractmethod
    def select_channels(
        self, module: nn.Module, n_kill: int, seed: int
    ) -> np.ndarray:
        """Return indices of output units to zero out.

        Args:
            module: The target Conv2d/Linear module.
            n_kill: Number of output units to ablate.
            seed: RNG seed for reproducibility.

        Returns:
            1-D array of int indices into the output dimension, length ``n_kill``.
        """
        raise NotImplementedError

    @contextmanager
    def lesion(
        self,
        model: nn.Module,
        layer_name: str,
        ratio: float,
        seed: int = 0,
    ) -> Iterator[None]:
        """Temporarily ablate ``ratio`` of a layer's output units.

        Backs up weights (and bias), zeroes the selected output units, yields,
        then restores the original parameters -- even if an exception occurs.
        ``ratio == 0`` is a valid no-op (still safely backed up/restored).
        """
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"ratio must be in [0, 1], got {ratio}.")

        module = _get_module(model, layer_name)
        n_units = _num_output_units(module)
        n_kill = int(round(ratio * n_units))

        # Backup on CPU clones so we always restore exactly.
        weight_backup = module.weight.detach().clone()
        bias_backup = (
            module.bias.detach().clone() if module.bias is not None else None
        )

        try:
            if n_kill > 0:
                idx = self.select_channels(module, n_kill, seed)
                with torch.no_grad():
                    module.weight[idx] = 0.0
                    if module.bias is not None:
                        module.bias[idx] = 0.0
            yield
        finally:
            # Always heal -- guarantees a clean baseline for the next run.
            with torch.no_grad():
                module.weight.copy_(weight_backup)
                if bias_backup is not None:
                    module.bias.copy_(bias_backup)
