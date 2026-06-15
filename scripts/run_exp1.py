"""CLI entry point for Experiment 1 (layer sensitivity).

Usage:
    python scripts/run_exp1.py --checkpoint artifacts/checkpoints/vgg16_cifar.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ai_stroke.data.datamodule import get_dataloaders
from ai_stroke.experiments.exp1_layer_sensitivity import Exp1Config, run_layer_sensitivity
from ai_stroke.models.registry import MODEL_REGISTRY
from ai_stroke.models.inspect import pick_layers_by_depth, resolve_layer_name
from ai_stroke.utils.logging import get_logger
from ai_stroke.utils.seed import set_seed
from ai_stroke.viz.heatmap import plot_criticality_heatmap

logger = get_logger("run_exp1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Experiment 1: layer sensitivity.")
    p.add_argument("--checkpoint", type=str, required=True, help="Healthy model weights.")
    p.add_argument("--data-dir", type=str, default="artifacts/data")
    p.add_argument("--results-dir", type=str, default="artifacts/results")
    p.add_argument("--figures-dir", type=str, default="artifacts/figures")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # 1) Load healthy baseline model.
    model = MODEL_REGISTRY.get("vgg16_cifar")(num_classes=10).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    logger.info("Loaded healthy checkpoint: %s", args.checkpoint)

    # 2) Data (test split only is needed for evaluation).
    _, test_loader = get_dataloaders(args.data_dir)

    # 3) Auto-pick early/mid/late Conv layers + the first FC, by semantic depth.
    #    No fragile integer indices: names are resolved from the real model.
    conv_targets = pick_layers_by_depth(model, fractions=(0.0, 0.33, 0.66, 1.0))
    fc_target = resolve_layer_name(model, "linear", 0)  # first FC (decision/PFC)

    cfg = Exp1Config(
        target_layers=conv_targets + [fc_target],
        damage_ratios=[0.3, 0.5, 0.7],
        n_repeats=10,
        lesion_name="random_node_ablation",
    )
    logger.info("Resolved target layers: %s", cfg.target_layers)

    # 4) Run and persist results.
    df = run_layer_sensitivity(model, test_loader, device, cfg)

    results_path = Path(args.results_dir) / "exp1_layer_sensitivity.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)
    logger.info("Saved results: %s", results_path)

    fig_path = plot_criticality_heatmap(
        df, Path(args.figures_dir) / "exp1_criticality_heatmap.png"
    )
    logger.info("Saved figure: %s", fig_path)

    # 5) Quick console summary.
    summary = (
        df.groupby(["target_layer", "damage_ratio"])["accuracy_drop"]
        .agg(["mean", "std"]).round(4)
    )
    logger.info("Summary (accuracy drop):\n%s", summary)


if __name__ == "__main__":
    main()
