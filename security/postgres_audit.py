"""Durable PostgreSQL sink for security audit events."""

from __future__ import annotations

import json
import uuid
from contextlib import AbstractContextManager
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
                    safe_details
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s::jsonb
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
                ),
            ).fetchone()
            if row is None:
                raise RuntimeError("audit insert returned no event id")

        return uuid.UUID(str(row[0]))
