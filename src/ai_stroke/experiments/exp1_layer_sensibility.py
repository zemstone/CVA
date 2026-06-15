"""Experiment 1: Layer Sensitivity ('which layer is most critical?').

For each (target_layer x damage_ratio), apply random node ablation N times
with different seeds and record accuracy drop (mean +/- std). The lesion is
reversible, so a single healthy checkpoint is reused across all conditions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ai_stroke.engine.evaluator import evaluate_accuracy
from ai_stroke.lesion.registry import LESION_REGISTRY
from ai_stroke.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Exp1Config:
    target_layers: List[str]      # e.g. ["features.0", "features.10", "classifier.0"]
    damage_ratios: List[float]    # e.g. [0.3, 0.5, 0.7]
    n_repeats: int                # e.g. 10
    lesion_name: str              # e.g. "random_node_ablation"


def run_layer_sensitivity(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Exp1Config,
) -> pd.DataFrame:
    """Run the full sweep and return a tidy DataFrame of results.

    Columns: target_layer, damage_ratio, seed, accuracy, accuracy_drop.
    """
    model.eval()  # freeze BatchNorm -> pure damage effect (no adaptation)
    strategy = LESION_REGISTRY.get(cfg.lesion_name)()

    baseline_acc = evaluate_accuracy(model, loader, device)
    logger.info("Healthy baseline accuracy: %.4f", baseline_acc)

    records: List[Dict[str, float | str]] = []
    for layer in cfg.target_layers:
        for ratio in cfg.damage_ratios:
            for seed in range(cfg.n_repeats):
                with strategy.lesion(model, layer, ratio, seed) as info:
                    acc = evaluate_accuracy(model, loader, device)
                # weights auto-restored here
                records.append({
                    "target_layer": layer,
                    "damage_ratio": ratio,
                    "seed": seed,
                    "accuracy": acc,
                    "accuracy_drop": baseline_acc - acc,
                    "units_killed": info.num_units_killed,
                })
            logger.info(
                "layer=%s ratio=%.2f done (%d repeats).", layer, ratio, cfg.n_repeats
            )

    return pd.DataFrame.from_records(records)
