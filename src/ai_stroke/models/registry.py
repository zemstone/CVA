"""Registry instance for models."""
from __future__ import annotations

import torch.nn as nn

from ai_stroke.utils.registry import Registry

MODEL_REGISTRY: Registry[type[nn.Module]] = Registry("model")
