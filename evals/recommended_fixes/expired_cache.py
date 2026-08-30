"""Frozen defect used to evaluate actionable recommended fixes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CacheEntry:
    value: str
    expires_at: float


class Cache:
    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._entries: dict[str, CacheEntry] = {}

    def get(self, key: str) -> str | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.expires_at <= self._clock():
            self._entries.pop(key, None)
        return entry.value
