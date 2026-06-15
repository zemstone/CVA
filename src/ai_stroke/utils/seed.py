"""Reproducibility utilities: fix all random sources for deterministic experiments."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Fix every relevant RNG so lesion experiments are reproducible.

    Args:
        seed: Global random seed.
        deterministic: If True, force cuDNN into deterministic mode
            (slower but required for trustworthy lesion comparisons).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
