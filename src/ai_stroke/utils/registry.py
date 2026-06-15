"""A generic string -> object registry, the backbone of our extensibility.

Models and lesion strategies register themselves here so experiments can be
fully driven by YAML config (no hardcoded class names).
"""
from __future__ import annotations

from typing import Callable, Dict, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps a string key to a factory/class. Raises on duplicate or missing keys."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._store: Dict[str, T] = {}

    def register(self, key: str) -> Callable[[T], T]:
        """Decorator: register a class/function under ``key``."""

        def _wrap(obj: T) -> T:
            if key in self._store:
                raise KeyError(f"'{key}' already registered in '{self._name}' registry.")
            self._store[key] = obj
            return obj

        return _wrap

    def get(self, key: str) -> T:
        if key not in self._store:
            raise KeyError(
                f"'{key}' not found in '{self._name}' registry. "
                f"Available keys: {sorted(self._store.keys())}"
            )
        return self._store[key]

    def keys(self) -> list[str]:
        return sorted(self._store.keys())
