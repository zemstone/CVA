"""Tests for model introspection helpers (guard against fragile indices)."""
from __future__ import annotations

import torch.nn as nn

from ai_stroke.models.inspect import (
    list_targetable_layers,
    pick_layers_by_depth,
    resolve_layer_name,
)
from ai_stroke.models.vgg import VGG16Cifar


def test_lists_all_conv_and_linear_layers() -> None:
    model = VGG16Cifar(num_classes=10)
    infos = list_targetable_layers(model)
    convs = [i for i in infos if i.kind == "Conv2d"]
    linears = [i for i in infos if i.kind == "Linear"]
    assert len(convs) == 13   # VGG-16 has 13 conv layers
    assert len(linears) == 2  # our CIFAR head has 2 FC layers


def test_resolve_layer_name_points_to_real_conv() -> None:
    model = VGG16Cifar(num_classes=10)
    name = resolve_layer_name(model, "conv", 0)
    module = dict(model.named_modules())[name]
    assert isinstance(module, nn.Conv2d)  # never lands on BN/ReLU


def test_pick_layers_by_depth_spans_network() -> None:
    model = VGG16Cifar(num_classes=10)
    names = pick_layers_by_depth(model, fractions=(0.0, 0.5, 1.0))
    modules = dict(model.named_modules())
    assert all(isinstance(modules[n], nn.Conv2d) for n in names)
    assert len(names) == 3
