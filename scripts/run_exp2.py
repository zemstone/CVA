"""CLI entry point for Experiment 2 (tipping point / RQ2).

Usage:
    python scripts/run_exp2.py --checkpoint artifacts/checkpoints/vgg16_cifar.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ai_stroke.data.datamodule import get_dataloaders
from ai_stroke.experiments.exp2_tipping_point import Exp2Config, run_tipping_point
from ai_stroke.models.inspect import pick_layers_by_depth, resolve_layer_name
from ai_stroke.models.registry import MODEL_REGISTRY
from ai_stroke.utils.logging import get_logger
from ai_stroke.utils.seed import set_seed
from ai_stroke.viz.degradation import plot_degradation_curves, plot_strategy_comparison

logger = get_logger("run_exp2")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Experiment 2: tipping point.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-dir", type=str, default="artifacts/data")
    p.add_argument("--results-dir", type=str, default="artifacts/results")
    p.add_argument("--figures-dir", type=str, default="artifacts/figures")
    p.add_argument("--n-repeats", type=int, default=10)
    p.add_argument("--n-points", type=int, default=21)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model = MODEL_REGISTRY.get("vgg16_cifar")(num_classes=10).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    logger.info("Loaded healthy checkpoint: %s", args.checkpoint)

    _, test_loader = get_dataloaders(args.data_dir)

    # Same semantic layers as RQ1 so curves are directly comparable.
    conv_targets = pick_layers_by_depth(model, fractions=(0.0, 0.33, 0.66, 1.0))
    fc_target = resolve_layer_name(model, "linear", 0)

    cfg = Exp2Config(
        target_layers=conv_targets + [fc_target],
        n_repeats=args.n_repeats,
        n_ratio_points=args.n_points,
        chance_acc=0.1,
    )
    logger.info("Sweeping layers: %s", cfg.target_layers)

    curve_df, summary_df = run_tipping_point(model, test_loader, device, cfg)

    plot_strategy_comparison(
        curve_df,
        Path(arg.figures_dir) / "exp2_strategy_comparison.png",
        chance_acc=cfg.chance_acc
    )

    # Persist results.
    res_dir = Path(args.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    curve_df.to_csv(res_dir / "exp2_degradation_curves.csv", index=False)
    summary_df.to_csv(res_dir / "exp2_tipping_summary.csv", index=False)
    logger.info("Saved curve + summary CSVs to %s", res_dir)

    fig = plot_degradation_curves(
        curve_df, summary_df,
        Path(args.figures_dir) / "exp2_degradation_curves.png",
        chance_acc=cfg.chance_acc,
    )
    logger.info("Saved figure: %s", fig)
    logger.info("Tipping-point summary:\n%s", summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
