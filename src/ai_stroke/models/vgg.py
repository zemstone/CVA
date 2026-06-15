"""VGG-16 adapted for CIFAR-10 (32x32), trained from scratch.

Layers are named so lesion experiments can target them via config, e.g.
'features.0' (first conv block) ... 'classifier.0' (first FC).
"""
from __future__ import annotations

from typing import List, Union

import torch
import torch.nn as nn

from ai_stroke.models.registry import MODEL_REGISTRY

# Standard VGG-16 layout: int = conv out-channels, "M" = max-pool.
_VGG16_CFG: List[Union[int, str]] = [
    64, 64, "M",
    128, 128, "M",
    256, 256, 256, "M",
    512, 512, 512, "M",
    512, 512, 512, "M",
]


def _make_features(cfg: List[Union[int, str]]) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_ch = 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_ch, v, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(v))
            layers.append(nn.ReLU(inplace=True))
            in_ch = v
    return nn.Sequential(*layers)


@MODEL_REGISTRY.register("vgg16_cifar")
class VGG16Cifar(nn.Module):
    """VGG-16 with BatchNorm, sized for 32x32 CIFAR inputs."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = _make_features(_VGG16_CFG)
        # After 5 max-pools on 32x32 -> 1x1 spatial, 512 channels.
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
