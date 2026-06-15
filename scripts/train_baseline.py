"""CLI: train the healthy VGG-16 baseline on CIFAR-10 from scratch.

Usage:
    python scripts/train_baseline.py --epochs 100 \
        --checkpoint artifacts/checkpoints/vgg16_cifar.pt
"""
from __future__ import annotations

import argparse

import torch

from ai_stroke.data.datamodule import get_dataloaders
from ai_stroke.engine.trainer import TrainConfig, train_baseline
from ai_stroke.models.inspect import list_targetable_layers
from ai_stroke.models.registry import MODEL_REGISTRY
from ai_stroke.utils.logging import get_logger
from ai_stroke.utils.seed import set_seed

logger = get_logger("train_baseline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train healthy VGG-16 baseline.")
    p.add_argument("--checkpoint", type=str, default="artifacts/checkpoints/vgg16_cifar.pt")
    p.add_argument("--data-dir", type=str, default="artifacts/data")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    train_loader, test_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size
    )

    model = MODEL_REGISTRY.get("vgg16_cifar")(num_classes=10)

    # Print the *real* targetable layers so the user can copy correct names
    # straight into the Experiment 1 config (no manual index counting).
    logger.info("Targetable layers (use these names in exp configs):")
    for info in list_targetable_layers(model):
        logger.info(
            "  %-16s | %-7s | out_units=%4d | %s#%d",
            info.name, info.kind, info.out_units, info.kind.lower(), info.depth_index,
        )

    cfg = TrainConfig(epochs=args.epochs, lr=args.lr)
    best_acc = train_baseline(
        model, train_loader, test_loader, device, cfg, args.checkpoint
    )
    logger.info("Baseline ready. Best acc=%.4f -> %s", best_acc, args.checkpoint)


if __name__ == "__main__":
    main()
