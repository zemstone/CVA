"""Model introspection helpers.

Targeting layers by hand (e.g. 'features.34') is error-prone because Conv,
BatchNorm, ReLU and MaxPool are interleaved in a Sequential. These helpers
resolve *real* Conv/Linear layers automatically so experiment configs can
refer to them by semantic depth ('first conv', 'last conv') instead of a
fragile integer index.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import torch.nn as nn

_TargetType = Literal["conv", "linear"]


@dataclass(frozen=True)
class LayerInfo:
    """A single targetable layer discovered in a model."""

    name: str          # dotted module path, e.g. "features.40"
    kind: str          # "Conv2d" or "Linear"
    out_units: int     # output channels (Conv) or neurons (Linear)
    depth_index: int   # 0-based order among same-kind layers (0 = earliest)


def list_targetable_layers(
    model: nn.Module, kinds: tuple[type[nn.Module], ...] = (nn.Conv2d, nn.Linear)
) -> List[LayerInfo]:
    """Return every Conv2d/Linear layer in forward order with metadata.

    The ``depth_index`` is computed per-kind, so the first Conv has
    depth_index=0, the second Conv depth_index=1, etc. (Linear counted
    separately).
    """
    conv_counter = 0
    linear_counter = 0
    infos: List[LayerInfo] = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            infos.append(
                LayerInfo(name, "Conv2d", module.out_channels, conv_counter)
            )
            conv_counter += 1
        elif isinstance(module, nn.Linear):
            infos.append(
                LayerInfo(name, "Linear", module.out_features, linear_counter)
            )
            linear_counter += 1

    return infos


def resolve_layer_name(
    model: nn.Module, kind: _TargetType, depth_index: int
) -> str:
    """Translate ('conv', 2) -> the real module name of the 3rd Conv layer.

    Raises:
        IndexError: If ``depth_index`` exceeds the number of layers of ``kind``.
    """
    target_cls = nn.Conv2d if kind == "conv" else nn.Linear
    matches = [
        info for info in list_targetable_layers(model, (target_cls,))
    ]
    if depth_index >= len(matches):
        raise IndexError(
            f"Requested {kind} depth_index={depth_index}, but model only has "
            f"{len(matches)} {kind} layers."
        )
    return matches[depth_index].name


def pick_layers_by_depth(
    model: nn.Module, fractions: tuple[float, ...] = (0.0, 0.33, 0.66, 1.0)
) -> List[str]:
    """Pick Conv layer names spread across the network depth.

    Useful for Experiment 1: sample 'early / mid / late' Conv layers
    automatically regardless of the exact architecture, e.g. fractions
    (0.0, 0.5, 1.0) -> first, middle, last Conv layer names.
    """
    convs = list_targetable_layers(model, (nn.Conv2d,))
    if not convs:
        raise ValueError("Model has no Conv2d layers to target.")

    last = len(convs) - 1
    chosen: List[str] = []
    for frac in fractions:
        idx = round(frac * last)
        name = convs[idx].name
        if name not in chosen:  # dedupe when fractions collide on small nets
            chosen.append(name)
    return chosen
