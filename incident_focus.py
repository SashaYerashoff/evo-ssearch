"""Bounded, process-local focus leases for active incidents.

The manager deliberately owns no database, HTTP, model, or capture concerns.  A
caller may temporarily raise attention for one or more channels while an
incident is being followed.  Expiry is lazy and deterministic: every public
read or mutation removes expired leases under the same lock.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable


class FocusLevel(str, Enum):
    FOLLOW = "follow"
    CRITICAL = "critical"


class FocusLeaseCapacityError(RuntimeError):
    """The bounded lease table is full of still-active incidents."""


@dataclass(frozen=True)
class FocusLease:
    incident_id: str
    level: FocusLevel
    channel_ids: tuple[int, ...]
    created_at_ms: int
    updated_at_ms: int
    expires_at_ms: int
    context: str = ""

    def remaining_ms(self, now_ms: int) -> int:
        return max(0, int(self.expires_at_ms) - int(now_ms))


@dataclass(frozen=True)
class FocusDirective:
    level: FocusLevel
    incident_ids: tuple[str, ...]
    expires_at_ms: int
    contexts: tuple[str, ...] = ()


class IncidentFocusLeaseManager:
    """Thread-safe, bounded focus leases keyed by incident id."""

    def __init__(
        self,
        *,
        max_leases: int = 64,
        max_channels_per_lease: int = 64,
        max_ttl_seconds: float = 8 * 60 * 60,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        if int(max_leases) <= 0:
            raise ValueError("max_leases must be positive")
        if int(max_channels_per_lease) <= 0:
            raise ValueError("max_channels_per_lease must be positive")
        if float(max_ttl_seconds) <= 0:
            raise ValueError("max_ttl_seconds must be positive")
        self.max_leases = int(max_leases)
        self.max_channels_per_lease = int(max_channels_per_lease)
        self.max_ttl_ms = int(float(max_ttl_seconds) * 1000.0)
        self._clock_ms = clock_ms or (lambda: int(time.monotonic() * 1000.0))
        self._leases: dict[str, FocusLease] = {}
        self._lock = threading.RLock()

    def start(
        self,
        incident_id: str,
        channel_ids: Iterable[int],
        *,
        level: FocusLevel | str = FocusLevel.FOLLOW,
        ttl_seconds: float = 5 * 60,
        context: str | None = None,
    ) -> FocusLease:
        normalized_id = self._incident_id(incident_id)
        normalized_channels = self._channel_ids(channel_ids)
        normalized_level = self._level(level)
        ttl_ms = self._ttl_ms(ttl_seconds)
        with self._lock:
            now_ms = self._now_ms()
            self._expire_locked(now_ms)
            previous = self._leases.get(normalized_id)
            if previous is None and len(self._leases) >= self.max_leases:
                raise FocusLeaseCapacityError(
                    f"incident focus capacity reached ({self.max_leases})"
                )
            lease = FocusLease(
                incident_id=normalized_id,
                level=normalized_level,
                channel_ids=normalized_channels,
                created_at_ms=(
                    previous.created_at_ms if previous is not None else now_ms
                ),
                updated_at_ms=now_ms,
                expires_at_ms=now_ms + ttl_ms,
                context=self._context(
                    previous.context if context is None and previous is not None else context
                ),
            )
            self._leases[normalized_id] = lease
            return lease

    def stop(self, incident_id: str) -> bool:
        normalized_id = self._incident_id(incident_id)
        with self._lock:
            self._expire_locked(self._now_ms())
            return self._leases.pop(normalized_id, None) is not None

    def expire(self) -> int:
        with self._lock:
            return self._expire_locked(self._now_ms())

    def get(self, incident_id: str) -> FocusLease | None:
        normalized_id = self._incident_id(incident_id)
        with self._lock:
            self._expire_locked(self._now_ms())
            return self._leases.get(normalized_id)

    def directive_for_channel(self, channel_id: int) -> FocusDirective | None:
        normalized_channel = self._channel_id(channel_id)
        with self._lock:
            self._expire_locked(self._now_ms())
            matches = [
                lease
                for lease in self._leases.values()
                if normalized_channel in lease.channel_ids
            ]
            if not matches:
                return None
            strongest = (
                FocusLevel.CRITICAL
                if any(lease.level is FocusLevel.CRITICAL for lease in matches)
                else FocusLevel.FOLLOW
            )
            relevant = sorted(
                (lease for lease in matches if lease.level is strongest),
                key=lambda lease: (lease.expires_at_ms, lease.incident_id),
            )
            return FocusDirective(
                level=strongest,
                incident_ids=tuple(lease.incident_id for lease in relevant),
                expires_at_ms=max(lease.expires_at_ms for lease in relevant),
                contexts=tuple(
                    lease.context
                    for lease in relevant
                    if lease.context
                )[:4],
            )

    def compact_digest(self, *, incident_limit: int = 16) -> dict[str, object]:
        limit = max(0, min(self.max_leases, int(incident_limit)))
        with self._lock:
            now_ms = self._now_ms()
            self._expire_locked(now_ms)
            leases = sorted(
                self._leases.values(),
                key=lambda lease: (
                    0 if lease.level is FocusLevel.CRITICAL else 1,
                    lease.expires_at_ms,
                    lease.incident_id,
                ),
            )
            visible = leases[:limit]
            return {
                "active": len(leases),
                "critical": sum(
                    lease.level is FocusLevel.CRITICAL for lease in leases
                ),
                "follow": sum(
                    lease.level is FocusLevel.FOLLOW for lease in leases
                ),
                "incidents": [
                    {
                        "id": lease.incident_id,
                        "level": lease.level.value,
                        "channels": list(lease.channel_ids),
                        "expires_in_ms": lease.remaining_ms(now_ms),
                        "context_attached": bool(lease.context),
                    }
                    for lease in visible
                ],
                "omitted": max(0, len(leases) - len(visible)),
            }

    def _expire_locked(self, now_ms: int) -> int:
        expired = [
            incident_id
            for incident_id, lease in self._leases.items()
            if lease.expires_at_ms <= now_ms
        ]
        for incident_id in expired:
            self._leases.pop(incident_id, None)
        return len(expired)

    def _now_ms(self) -> int:
        value = int(self._clock_ms())
        if value < 0:
            raise ValueError("clock_ms must not return a negative value")
        return value

    @staticmethod
    def _incident_id(value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("incident_id must not be empty")
        if len(normalized) > 160:
            raise ValueError("incident_id must not exceed 160 characters")
        return normalized

    @staticmethod
    def _channel_id(value: int) -> int:
        try:
            normalized = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("channel ids must be positive integers") from exc
        if normalized <= 0:
            raise ValueError("channel ids must be positive integers")
        return normalized

    def _channel_ids(self, values: Iterable[int]) -> tuple[int, ...]:
        normalized = tuple(sorted({self._channel_id(value) for value in values}))
        if not normalized:
            raise ValueError("channel_ids must not be empty")
        if len(normalized) > self.max_channels_per_lease:
            raise ValueError(
                "channel_ids exceeds max_channels_per_lease "
                f"({self.max_channels_per_lease})"
            )
        return normalized

    @staticmethod
    def _level(value: FocusLevel | str) -> FocusLevel:
        try:
            return value if isinstance(value, FocusLevel) else FocusLevel(str(value))
        except ValueError as exc:
            raise ValueError("level must be follow or critical") from exc

    def _ttl_ms(self, value: float) -> int:
        try:
            ttl_ms = int(float(value) * 1000.0)
        except (TypeError, ValueError) as exc:
            raise ValueError("ttl_seconds must be a positive number") from exc
        if ttl_ms <= 0:
            raise ValueError("ttl_seconds must be a positive number")
        if ttl_ms > self.max_ttl_ms:
            raise ValueError(
                f"ttl_seconds exceeds maximum ({self.max_ttl_ms / 1000.0:g})"
            )
        return ttl_ms

    @staticmethod
    def _context(value: str | None) -> str:
        # Context is inert, bounded prior evidence carried to the live VLM.
        # It never changes lease routing or attention authority.
        return " ".join(str(value or "").split())[:2400]


__all__ = [
    "FocusDirective",
    "FocusLease",
    "FocusLeaseCapacityError",
    "FocusLevel",
    "IncidentFocusLeaseManager",
]
