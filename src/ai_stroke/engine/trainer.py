"""Training engine for the healthy baseline model.

Kept separate from the CLI script so it can be unit-tested and reused by
future experiments (e.g. RQ3 recovery / fine-tuning).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ai_stroke.engine.evaluator import evaluate_accuracy
from ai_stroke.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    label_smoothing: float = 0.0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch; return mean loss."""
    model.train()
    running_loss = 0.0
    n_batches = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        n_batches += 1
    return running_loss / max(n_batches, 1)


def train_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    checkpoint_path: str | Path,
) -> float:
    """Train the 'healthy brain' from scratch and save the best checkpoint.

    Returns:
        Best test accuracy achieved.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.lr,
        momentum=cfg.momentum, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        model.eval()
        acc = evaluate_accuracy(model, test_loader, device)
        logger.info(
            "epoch %3d/%d | loss=%.4f | test_acc=%.4f | lr=%.5f",
            epoch, cfg.epochs, loss, acc, scheduler.get_last_lr()[0],
        )

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("New best (%.4f) saved -> %s", best_acc, checkpoint_path)

    logger.info("Training done. Best test accuracy: %.4f", best_acc)
    return best_acc
