"""PostgreSQL-backed login throttle repository."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from eva_db import PsycopgPool, TransactionContext
from security.throttling import LoginThrottleRecord


NIL_UUID = uuid.UUID(int=0)


class PostgresLoginThrottleRepository:
    """Durable throttle state shared by every EVA API process."""

    def __init__(self, pool: PsycopgPool, tenant_id: str | uuid.UUID) -> None:
        self._pool = pool
        self._tenant_id = _require_uuid(tenant_id, "tenant_id")

    def get(self, key: str) -> LoginThrottleRecord | None:
        throttle_key = _normalize_key(key)
        with self._pool.transaction(
            _transaction_context(self._tenant_id),
            readonly=True,
        ) as connection:
            row = connection.execute(
                """
                SELECT failed_attempts, window_started_at, locked_until
                FROM iam.login_attempts
                WHERE tenant_id = %s AND throttle_key = %s
                """,
                (self._tenant_id, throttle_key),
            ).fetchone()
        if row is None:
            return None
        return LoginThrottleRecord(
            failed_attempts=int(row[0]),
            window_started_at=_timestamp(row[1], "window_started_at"),
            locked_until=(
                None
                if row[2] is None
                else _timestamp(row[2], "locked_until")
            ),
        )

    def save(self, key: str, record: LoginThrottleRecord) -> None:
        throttle_key = _normalize_key(key)
        if record.failed_attempts < 0:
            raise ValueError("failed_attempts cannot be negative")
        window_started_at = _datetime(record.window_started_at, "window_started_at")
        locked_until = (
            None
            if record.locked_until is None
            else _datetime(record.locked_until, "locked_until")
        )
        with self._pool.transaction(
            _transaction_context(self._tenant_id),
        ) as connection:
            connection.execute(
                """
                INSERT INTO iam.login_attempts (
                    tenant_id,
                    throttle_key,
                    failed_attempts,
                    window_started_at,
                    locked_until
                )
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (tenant_id, throttle_key) DO UPDATE
                SET failed_attempts = EXCLUDED.failed_attempts,
                    window_started_at = EXCLUDED.window_started_at,
                    locked_until = EXCLUDED.locked_until,
                    updated_at = clock_timestamp()
                """,
                (
                    self._tenant_id,
                    throttle_key,
                    int(record.failed_attempts),
                    window_started_at,
                    locked_until,
                ),
            )

    def delete(self, key: str) -> None:
        throttle_key = _normalize_key(key)
        with self._pool.transaction(
            _transaction_context(self._tenant_id),
        ) as connection:
            connection.execute(
                """
                DELETE FROM iam.login_attempts
                WHERE tenant_id = %s AND throttle_key = %s
                """,
                (self._tenant_id, throttle_key),
            )


def _transaction_context(tenant_id: uuid.UUID) -> TransactionContext:
    return TransactionContext(tenant_id=tenant_id, actor_id=NIL_UUID)


def _require_uuid(value: str | uuid.UUID, field_name: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a UUID") from exc


def _normalize_key(key: str) -> str:
    normalized = str(key or "").strip()
    if not normalized or "\x00" in normalized or len(normalized) > 512:
        raise ValueError("throttle key must contain 1 to 512 safe characters")
    return normalized


def _datetime(timestamp: float, field_name: str) -> datetime:
    try:
        value = datetime.fromtimestamp(float(timestamp), timezone.utc)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a valid epoch timestamp") from exc
    return value


def _timestamp(value: Any, field_name: str) -> float:
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a datetime")
    normalized = value
    if normalized.tzinfo is None or normalized.utcoffset() is None:
        normalized = normalized.replace(tzinfo=timezone.utc)
    return normalized.timestamp()
