"""Evaluation logic. For lesion experiments, the model MUST be in eval mode so
BatchNorm running stats are frozen (pure-damage measurement, no adaptation).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> float:
    """Top-1 accuracy over the loader. Caller is responsible for model.eval()."""
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return correct / total if total > 0 else 0.0
