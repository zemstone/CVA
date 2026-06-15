"""Registry instance for lesion strategies."""
from __future__ import annotations

from ai_stroke.lesion.base import LesionStrategy
from ai_stroke.utils.registry import Registry

LESION_REGISTRY: Registry[type[LesionStrategy]] = Registry("lesion")
