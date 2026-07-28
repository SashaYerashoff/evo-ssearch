"""Durable PostgreSQL sink for security audit events."""

from __future__ import annotations

import hashlib
import json
import uuid
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from typing import Any, Protocol

from security.audit import AuditEvent, redact_audit_details


_RESULT_ALIASES = {
    "allow": "success",
    "allowed": "success",
    "complete": "success",
    "completed": "success",
    "ok": "success",
    "result": "success",
    "succeed": "success",
    "succeeded": "success",
    "success": "success",
    "error": "failure",
    "errored": "failure",
    "exception": "failure",
    "fail": "failure",
    "failed": "failure",
    "failure": "failure",
    "block": "denied",
    "blocked": "denied",
    "denied": "denied",
    "deny": "denied",
    "forbidden": "denied",
    "reject": "denied",
    "rejected": "denied",
}


class _Connection(Protocol):
    def execute(self, query: str, params: Any = ...) -> Any: ...


class _TransactionPool(Protocol):
    def transaction(
        self,
        context: Any = None,
        *,
        readonly: bool = False,
    ) -> AbstractContextManager[_Connection]: ...


def _required_uuid(value: str | None, field_name: str) -> uuid.UUID:
    if value is None or not str(value).strip():
        raise ValueError(f"{field_name} is required for durable audit events")
    try:
        return uuid.UUID(str(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a UUID") from exc


def _optional_uuid(value: str | None, field_name: str) -> uuid.UUID | None:
    if value is None:
        return None
    try:
        return uuid.UUID(str(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a UUID") from exc


def _channel_id(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("channel_id must be a positive integer")
    return value


def _result(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("audit result must be a string")
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return _RESULT_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported audit result: {value!r}") from exc


def _safe_details_json(event: AuditEvent) -> str:
    safe_details = redact_audit_details(event.details)
    return json.dumps(
        safe_details,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("audit timestamps must be timezone-aware")
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _canonical_event_bytes(
    *,
    event_id: uuid.UUID,
    tenant_id: uuid.UUID,
    event: AuditEvent,
    actor_user_id: uuid.UUID | None,
    channel_id: int | None,
    result: str,
    safe_details_json: str,
) -> bytes:
    payload = {
        "version": 1,
        "id": str(event_id),
        "tenant_id": str(tenant_id),
        "occurred_at": _canonical_timestamp(event.timestamp),
        "request_id": event.request_id,
        "actor_user_id": (
            None if actor_user_id is None else str(actor_user_id)
        ),
        "actor_roles": list(event.actor_roles),
        "source_ip": event.source_ip,
        "action": event.action,
        "target_type": event.target_type,
        "target_id": event.target_id,
        "channel_id": channel_id,
        "result": result,
        "safe_details": json.loads(safe_details_json),
    }
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _event_hash(previous_event_hash: bytes | None, event_bytes: bytes) -> bytes:
    previous = previous_event_hash or b""
    return hashlib.sha256(
        b"eva-audit-chain-v1\0" + previous + b"\0" + event_bytes
    ).digest()


def _database_hash(value: object) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if not isinstance(value, bytes) or len(value) != 32:
        raise RuntimeError("previous audit event hash is invalid")
    return value


class PostgresAuditWriter:
    """Append security events to PostgreSQL and propagate every write failure."""

    def __init__(self, pool: _TransactionPool) -> None:
        self._pool = pool

    def __call__(self, event: AuditEvent) -> uuid.UUID:
        return self.write(event)

    def write(self, event: AuditEvent) -> uuid.UUID:
        if not isinstance(event, AuditEvent):
            raise TypeError("event must be an AuditEvent")

        event_id = uuid.uuid4()
        tenant_id = _required_uuid(event.tenant_id, "tenant_id")
        actor_user_id = _optional_uuid(event.actor_user_id, "actor_user_id")
        channel_id = _channel_id(event.channel_id)
        result = _result(event.result)
        safe_details_json = _safe_details_json(event)
        event_bytes = _canonical_event_bytes(
            event_id=event_id,
            tenant_id=tenant_id,
            event=event,
            actor_user_id=actor_user_id,
            channel_id=channel_id,
            result=result,
            safe_details_json=safe_details_json,
        )

        with self._pool.transaction() as connection:
            connection.execute(
                """
                SELECT
                    set_config('eva.tenant_id', %s, true),
                    set_config('eva.actor_id', %s, true),
                    set_config('eva.request_id', %s, true)
                """,
                (
                    str(tenant_id),
                    "" if actor_user_id is None else str(actor_user_id),
                    event.request_id or "",
                ),
            )
            # Serialize only this tenant's chain head. A hash collision here
            # merely serializes two unrelated tenants; it cannot mix their
            # rows because the subsequent lookup remains tenant-scoped by RLS.
            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (str(tenant_id),),
            )
            previous_row = connection.execute(
                """
                SELECT event_hash
                FROM audit.events
                WHERE tenant_id = %s
                ORDER BY sequence_number DESC
                LIMIT 1
                """,
                (tenant_id,),
            ).fetchone()
            previous_event_hash = _database_hash(
                previous_row[0] if previous_row is not None else None
            )
            event_hash = _event_hash(previous_event_hash, event_bytes)
            row = connection.execute(
                """
                INSERT INTO audit.events (
                    id,
                    tenant_id,
                    occurred_at,
                    request_id,
                    actor_user_id,
                    actor_roles,
                    source_ip,
                    action,
                    target_type,
                    target_id,
                    channel_id,
                    result,
                    safe_details,
                    previous_event_hash,
                    event_hash
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s::jsonb, %s, %s
                )
                RETURNING id
                """,
                (
                    event_id,
                    tenant_id,
                    event.timestamp,
                    event.request_id,
                    actor_user_id,
                    list(event.actor_roles),
                    event.source_ip,
                    event.action,
                    event.target_type,
                    event.target_id,
                    channel_id,
                    result,
                    safe_details_json,
                    previous_event_hash,
                    event_hash,
                ),
            ).fetchone()
            if row is None:
                raise RuntimeError("audit insert returned no event id")

        return uuid.UUID(str(row[0]))
