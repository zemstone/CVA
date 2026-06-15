"""Experiment 2: Tipping Point with multi-strategy comparison (RQ2 extended).

Compares damage *modalities* (random vs targeted) on the same layers, mapping
how the choice of *which* units fail changes the tipping point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ai_stroke.analysis.tipping_point import analyze_degradation_curve
from ai_stroke.engine.evaluator import evaluate_accuracy
from ai_stroke.lesion.base import LesionStrategy
from ai_stroke.lesion.registry import LESION_REGISTRY
from ai_stroke.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class StrategySpec:
    """How to instantiate one lesion strategy + a human-readable label."""

    key: str                                # registry key
    label: str                              # short label for plots/tables
    kwargs: Dict[str, Any] = field(default_factory=dict)
    deterministic: bool = False             # if True, force n_repeats = 1

    def build(self) -> LesionStrategy:
        return LESION_REGISTRY.get(self.key)(**self.kwargs)


def default_strategy_specs() -> List[StrategySpec]:
    """Random vs the two targeted extremes (the core RQ2-extended comparison)."""
    return [
        StrategySpec("random_node_ablation", "random"),
        StrategySpec(
            "targeted_ablation", "targeted-critical",
            kwargs={"mode": "most_important"}, deterministic=True,
        ),
        StrategySpec(
            "targeted_ablation", "targeted-redundant",
            kwargs={"mode": "least_important"}, deterministic=True,
        ),
    ]


@dataclass
class Exp2Config:
    target_layers: List[str]
    strategies: List[StrategySpec] = field(default_factory=default_strategy_specs)
    n_repeats: int = 10
    n_ratio_points: int = 21
    ratio_min: float = 0.0
    ratio_max: float = 1.0
    chance_acc: float = 0.1
    ratios: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.ratios = np.linspace(self.ratio_min, self.ratio_max, self.n_ratio_points)


def run_tipping_point(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Exp2Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep damage across (strategy x layer x ratio x seed).

    Returns:
        (curve_df, summary_df)
        - curve_df : per-(strategy, layer, ratio) mean/std accuracy.
        - summary_df: per-(strategy, layer) tipping-point characterization.
    """
    model.eval()  # BN frozen -> pure damage effect
    curve_records: List[Dict[str, Any]] = []
    summary_records: List[Dict[str, Any]] = []

    for spec in cfg.strategies:
        strategy = spec.build()
        n_repeats = 1 if spec.deterministic else cfg.n_repeats

        for layer in cfg.target_layers:
            mean_accs: List[float] = []
            for ratio in cfg.ratios:
                seed_accs = [
                    _eval_under_lesion(model, strategy, layer, ratio, seed, loader, device)
                    for seed in range(n_repeats)
                ]
                mean_acc = float(np.mean(seed_accs))
                mean_accs.append(mean_acc)
                curve_records.append({
                    "strategy": spec.label,
                    "target_layer": layer,
                    "damage_ratio": float(ratio),
                    "accuracy_mean": mean_acc,
                    "accuracy_std": float(np.std(seed_accs)),
                })

            result = analyze_degradation_curve(
                cfg.ratios, np.asarray(mean_accs), chance_acc=cfg.chance_acc
            )
            summary_records.append({
                "strategy": spec.label,
                "target_layer": layer,
                "tipping_ratio": result.tipping_ratio,
                "cliff_ratio": result.max_drop_ratio,
                "cliff_drop": result.max_drop_value,
                "curve_type": result.curve_type,
                "robustness_auc": result.auc,
            })
            logger.info(
                "[%s] layer=%s | type=%s | tipping=%s | AUC=%.3f",
                spec.label, layer, result.curve_type,
                result.tipping_ratio, result.auc,
            )

    return pd.DataFrame(curve_records), pd.DataFrame(summary_records)


def _eval_under_lesion(
    model: nn.Module,
    strategy: LesionStrategy,
    layer: str,
    ratio: float,
    seed: int,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Apply a reversible lesion and measure accuracy (model healed on exit)."""
    with strategy.lesion(model, layer, float(ratio), seed):
        return evaluate_accuracy(model, loader, device)
