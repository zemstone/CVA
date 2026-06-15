"""YAML config loading with light validation and dict-to-attribute access."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class Config(dict):
    """Dict subclass allowing attribute access (cfg.model.name) and nesting."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_config(path: str | Path) -> Config:
    """Load a YAML file into a Config object.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the YAML root is not a mapping.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw)}.")
    return Config(raw)
