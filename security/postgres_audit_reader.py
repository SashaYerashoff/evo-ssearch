"""Read-only PostgreSQL access for security audit events."""

from __future__ import annotations

import base64
import json
import uuid
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from eva_db import TransactionContext
from security.audit import redact_audit_details
from security.context import AuthContext


_MAX_LIMIT = 100
_DEFAULT_LIMIT = 50
_RESULTS = frozenset({"success", "failure", "denied"})


class _Cursor(Protocol):
    def fetchall(self) -> list[Any]: ...


class _Connection(Protocol):
    def execute(self, query: str, params: Any = ...) -> _Cursor: ...


class _TransactionPool(Protocol):
    def transaction(
        self,
        context: Any = None,
        *,
        readonly: bool = False,
    ) -> AbstractContextManager[_Connection]: ...


@dataclass(frozen=True, slots=True)
class AuditEventRecord:
    id: str
    sequence_number: int
    tenant_id: str
    occurred_at: datetime
    request_id: str | None
    actor_user_id: str | None
    actor_roles: tuple[str, ...]
    source_ip: str | None
    action: str
    target_type: str | None
    target_id: str | None
    channel_id: int | None
    result: str
    details: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sequenceNumber": self.sequence_number,
            "tenantId": self.tenant_id,
            "occurredAt": self.occurred_at.isoformat(),
            "requestId": self.request_id,
            "actorUserId": self.actor_user_id,
            "actorRoles": list(self.actor_roles),
            "sourceIp": self.source_ip,
            "action": self.action,
            "targetType": self.target_type,
            "targetId": self.target_id,
            "channelId": self.channel_id,
            "result": self.result,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class AuditEventPage:
    events: tuple[AuditEventRecord, ...]
    next_cursor: str | None


def _required_uuid(value: object, field_name: str) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{field_name} must be a UUID") from exc


def _optional_uuid(value: object, field_name: str) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    return _required_uuid(value, field_name)


def _bounded_limit(value: object) -> int:
    if value is None or str(value).strip() == "":
        return _DEFAULT_LIMIT
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("limit must be an integer") from exc
    if parsed < 1 or parsed > _MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {_MAX_LIMIT}")
    return parsed


def _positive_int(value: object, field_name: str) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0 or str(parsed) != str(value).strip():
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def _text_filter(value: object, field_name: str, *, max_length: int = 256) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > max_length or "\x00" in text:
        raise ValueError(f"{field_name} is too long or unsafe")
    return text


def _result_filter(value: object) -> str | None:
    text = _text_filter(value, "result", max_length=32)
    if text is None:
        return None
    normalized = text.lower()
    if normalized not in _RESULTS:
        raise ValueError("result must be success, failure, or denied")
    return normalized


def _time_filter(value: object, field_name: str) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        raw = str(value).strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include timezone")
    return parsed.astimezone(timezone.utc)


def _encode_cursor(record: AuditEventRecord) -> str:
    payload = {
        "occurredAt": record.occurred_at.isoformat(),
        "sequenceNumber": record.sequence_number,
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_cursor(value: object) -> tuple[datetime, int] | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        padded = text + ("=" * (-len(text) % 4))
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
    except Exception as exc:
        raise ValueError("cursor is invalid") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("cursor is invalid")
    occurred_at = _time_filter(payload.get("occurredAt"), "cursor.occurredAt")
    sequence_number = _positive_int(payload.get("sequenceNumber"), "cursor.sequenceNumber")
    if occurred_at is None or sequence_number is None:
        raise ValueError("cursor is invalid")
    return occurred_at, sequence_number


def _details(value: object) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("audit details must be JSON object") from exc
    if not isinstance(value, Mapping):
        raise ValueError("audit details must be JSON object")
    return redact_audit_details(value)


def _record_from_row(row: Any) -> AuditEventRecord:
    occurred_at = _time_filter(row[3], "occurred_at")
    if occurred_at is None:
        raise ValueError("occurred_at is required")
    return AuditEventRecord(
        id=str(row[0]),
        sequence_number=int(row[1]),
        tenant_id=str(row[2]),
        occurred_at=occurred_at,
        request_id=None if row[4] is None else str(row[4]),
        actor_user_id=None if row[5] is None else str(row[5]),
        actor_roles=tuple(str(role) for role in (row[6] or ())),
        source_ip=None if row[7] is None else str(row[7]),
        action=str(row[8]),
        target_type=None if row[9] is None else str(row[9]),
        target_id=None if row[10] is None else str(row[10]),
        channel_id=None if row[11] is None else int(row[11]),
        result=str(row[12]),
        details=_details(row[13]),
    )


class PostgresAuditReader:
    """Tenant-scoped, read-only audit event reader."""

    def __init__(self, pool: _TransactionPool) -> None:
        self._pool = pool

    def list_events(
        self,
        context: AuthContext,
        *,
        limit: object = None,
        cursor: object = None,
        since: object = None,
        until: object = None,
        actor_user_id: object = None,
        action: object = None,
        target_type: object = None,
        target_id: object = None,
        channel_id: object = None,
        result: object = None,
        request_id: object = None,
    ) -> AuditEventPage:
        tenant_id = _required_uuid(context.tenant_id, "tenant_id")
        actor_id = _required_uuid(context.user_id, "actor_id")
        page_limit = _bounded_limit(limit)
        cursor_value = _decode_cursor(cursor)
        since_dt = _time_filter(since, "since")
        until_dt = _time_filter(until, "until")
        if since_dt is not None and until_dt is not None and since_dt > until_dt:
            raise ValueError("since must be before until")
        actor_filter = _optional_uuid(actor_user_id, "actorUserId")
        channel_filter = _positive_int(channel_id, "channelId")
        result_filter = _result_filter(result)
        text_filters = {
            "action": _text_filter(action, "action"),
            "target_type": _text_filter(target_type, "targetType"),
            "target_id": _text_filter(target_id, "targetId"),
            "request_id": _text_filter(request_id, "requestId", max_length=128),
        }

        where = ["tenant_id = %s"]
        params: list[Any] = [tenant_id]
        if since_dt is not None:
            where.append("occurred_at >= %s")
            params.append(since_dt)
        if until_dt is not None:
            where.append("occurred_at <= %s")
            params.append(until_dt)
        if actor_filter is not None:
            where.append("actor_user_id = %s")
            params.append(actor_filter)
        if channel_filter is not None:
            where.append("channel_id = %s")
            params.append(channel_filter)
        if result_filter is not None:
            where.append("result = %s")
            params.append(result_filter)
        for column, value in text_filters.items():
            if value is None:
                continue
            where.append(f"{column} = %s")
            params.append(value)
        if cursor_value is not None:
            where.append("(occurred_at, sequence_number) < (%s, %s)")
            params.extend(cursor_value)

        query = f"""
            SELECT
                id,
                sequence_number,
                tenant_id,
                occurred_at,
                request_id,
                actor_user_id,
                actor_roles,
                source_ip::text,
                action,
                target_type,
                target_id,
                channel_id,
                result,
                safe_details
            FROM audit.events
            WHERE {' AND '.join(where)}
            ORDER BY occurred_at DESC, sequence_number DESC
            LIMIT %s
        """
        params.append(page_limit + 1)

        tx_context = TransactionContext(
            tenant_id=tenant_id,
            actor_id=actor_id,
            request_id=context.request_id,
        )
        with self._pool.transaction(tx_context, readonly=True) as connection:
            rows = connection.execute(query, tuple(params)).fetchall()

        records = tuple(_record_from_row(row) for row in rows[:page_limit])
        next_cursor = _encode_cursor(records[-1]) if len(rows) > page_limit and records else None
        return AuditEventPage(events=records, next_cursor=next_cursor)
