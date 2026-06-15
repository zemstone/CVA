"""Node-ablation lesion: randomly kill whole channels (Conv) or units (FC).

This mimics neuron death in a stroke. Killing is done by zeroing the entire
output channel/unit (weight row), matching the 'neuron death' analogy rather
than scattering individual weight edits.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ai_stroke.lesion.base import LesionResult, LesionStrategy
from ai_stroke.lesion.registry import LESION_REGISTRY


@LESION_REGISTRY.register("random_node_ablation")
class RandomNodeAblation(LesionStrategy):
    """Zero out a random fraction of output units in a target layer.

    - Conv2d: an output 'unit' is a filter -> weight shape [out_ch, in, kh, kw];
      we zero whole rows along dim 0 (a dead filter).
    - Linear: an output 'unit' is a neuron -> weight shape [out, in];
      we zero whole rows along dim 0 (a dead neuron).
    """

    def _apply(
        self, model: nn.Module, target_layer: str, damage_ratio: float, seed: int
    ) -> LesionResult:
        if not 0.0 <= damage_ratio <= 1.0:
            raise ValueError(f"damage_ratio must be in [0, 1], got {damage_ratio}.")

        modules = dict(model.named_modules())
        if target_layer not in modules:
            raise KeyError(
                f"Layer '{target_layer}' not in model. "
                f"Available: {[n for n, m in modules.items() if hasattr(m, 'weight')]}"
            )

        layer = modules[target_layer]
        if not isinstance(layer, (nn.Conv2d, nn.Linear)):
            raise TypeError(
                f"Ablation supports Conv2d/Linear only, got {type(layer).__name__}."
            )

        self._backup_layer(layer, target_layer)

        num_units = layer.weight.shape[0]
        num_kill = int(round(num_units * damage_ratio))

        # Deterministic unit selection given (seed) for reproducibility.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        kill_idx = torch.randperm(num_units, generator=generator)[:num_kill]

        with torch.no_grad():
            layer.weight[kill_idx] = 0.0
            if layer.bias is not None:
                layer.bias[kill_idx] = 0.0

        return LesionResult(
            target_layer=target_layer,
            damage_ratio=damage_ratio,
            num_units_total=num_units,
            num_units_killed=num_kill,
            seed=seed,
        )
