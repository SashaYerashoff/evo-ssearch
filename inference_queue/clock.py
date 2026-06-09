from __future__ import annotations

from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Protocol, Union


class Clock(Protocol):
    def now(self) -> datetime:
        """Return an aware timestamp."""


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class ManualClock:
    """Thread-safe clock for deterministic scheduling tests and simulations."""

    def __init__(self, current: datetime) -> None:
        self._lock = RLock()
        self._current = _as_utc(current)

    def now(self) -> datetime:
        with self._lock:
            return self._current

    def advance(self, amount: Union[timedelta, float, int]) -> datetime:
        delta = amount if isinstance(amount, timedelta) else timedelta(seconds=amount)
        if delta < timedelta(0):
            raise ValueError("clock cannot advance by a negative duration")
        with self._lock:
            self._current += delta
            return self._current

    def set(self, current: datetime) -> None:
        value = _as_utc(current)
        with self._lock:
            if value < self._current:
                raise ValueError("clock cannot move backwards")
            self._current = value


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)
