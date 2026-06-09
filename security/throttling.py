from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class LoginThrottlePolicy:
    max_attempts: int = 5
    window_seconds: float = 300.0
    lockout_seconds: float = 900.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if self.lockout_seconds <= 0:
            raise ValueError("lockout_seconds must be positive")


@dataclass(frozen=True, slots=True)
class LoginThrottleRecord:
    failed_attempts: int
    window_started_at: float
    locked_until: float | None = None


@dataclass(frozen=True, slots=True)
class LoginThrottleDecision:
    allowed: bool
    attempts_remaining: int
    retry_after_seconds: int = 0


@runtime_checkable
class LoginThrottleRepository(Protocol):
    def get(self, key: str) -> LoginThrottleRecord | None:
        ...

    def save(self, key: str, record: LoginThrottleRecord) -> None:
        ...

    def delete(self, key: str) -> None:
        ...


class InMemoryLoginThrottleRepository:
    """Test/development repository; production adapters should be shared."""

    def __init__(self) -> None:
        self._records: dict[str, LoginThrottleRecord] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> LoginThrottleRecord | None:
        with self._lock:
            return self._records.get(key)

    def save(self, key: str, record: LoginThrottleRecord) -> None:
        with self._lock:
            self._records[key] = record

    def delete(self, key: str) -> None:
        with self._lock:
            self._records.pop(key, None)


class LoginThrottleService:
    def __init__(
        self,
        repository: LoginThrottleRepository,
        *,
        policy: LoginThrottlePolicy | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._repository = repository
        self._policy = policy or LoginThrottlePolicy()
        self._clock = clock

    def check(self, key: str) -> LoginThrottleDecision:
        key = self._validate_key(key)
        now = self._clock()
        record = self._active_record(key, now)
        if record is None:
            return LoginThrottleDecision(
                allowed=True,
                attempts_remaining=self._policy.max_attempts,
            )
        if record.locked_until is not None and record.locked_until > now:
            return self._locked_decision(record, now)
        return LoginThrottleDecision(
            allowed=True,
            attempts_remaining=max(
                0, self._policy.max_attempts - record.failed_attempts
            ),
        )

    def record_failure(self, key: str) -> LoginThrottleDecision:
        key = self._validate_key(key)
        now = self._clock()
        record = self._active_record(key, now)
        if (
            record is not None
            and record.locked_until is not None
            and record.locked_until > now
        ):
            return self._locked_decision(record, now)

        failed_attempts = 1 if record is None else record.failed_attempts + 1
        window_started_at = now if record is None else record.window_started_at
        locked_until = None
        if failed_attempts >= self._policy.max_attempts:
            locked_until = now + self._policy.lockout_seconds

        updated = LoginThrottleRecord(
            failed_attempts=failed_attempts,
            window_started_at=window_started_at,
            locked_until=locked_until,
        )
        self._repository.save(key, updated)
        if locked_until is not None:
            return self._locked_decision(updated, now)
        return LoginThrottleDecision(
            allowed=True,
            attempts_remaining=self._policy.max_attempts - failed_attempts,
        )

    def record_success(self, key: str) -> None:
        self._repository.delete(self._validate_key(key))

    check_attempt = check
    register_failure = record_failure
    register_success = record_success

    def _active_record(
        self,
        key: str,
        now: float,
    ) -> LoginThrottleRecord | None:
        record = self._repository.get(key)
        if record is None:
            return None
        if record.locked_until is not None:
            if record.locked_until > now:
                return record
            self._repository.delete(key)
            return None
        if now - record.window_started_at >= self._policy.window_seconds:
            self._repository.delete(key)
            return None
        return record

    def _locked_decision(
        self,
        record: LoginThrottleRecord,
        now: float,
    ) -> LoginThrottleDecision:
        assert record.locked_until is not None
        return LoginThrottleDecision(
            allowed=False,
            attempts_remaining=0,
            retry_after_seconds=max(1, math.ceil(record.locked_until - now)),
        )

    @staticmethod
    def _validate_key(key: str) -> str:
        if not key or not key.strip():
            raise ValueError("throttle key is required")
        return key
