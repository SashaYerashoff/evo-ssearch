"""Tenant-scoped durable storage for operator-review incidents.

Incidents contain only bounded structured text and references to evidence that
already lives in the archive/attention stores.  Image payloads and thumbnails
are deliberately rejected at the repository boundary.
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from eva_db import PsycopgPool, TransactionContext


INCIDENT_STORAGE_REVISION = "20260801_0011"
NIL_UUID = uuid.UUID(int=0)
INCIDENT_STATES = frozenset(
    {"candidate", "draft", "following", "ended", "reported", "closed"}
)
_JSON_LIMIT_BYTES = 262_144
_MAX_CHANNELS = 32
_MAX_REF_ITEMS = 512
_DISALLOWED_JSON_KEYS = frozenset(
    {
        "frame_bytes",
        "image",
        "image_bytes",
        "image_b64",
        "image_url",
        "frame_b64",
        "jpeg_b64",
        "png_b64",
        "thumbnail",
        "thumbnail_b64",
    }
)
_JSON_OBJECT_FIELDS = frozenset(
    {"anchor_ref", "coverage", "report", "follow_policy"}
)
_JSON_ARRAY_FIELDS = frozenset(
    {"timeline_refs", "evidence_refs", "qualia_refs", "uncertainties"}
)
_MUTABLE_FIELDS = frozenset(
    {
        "state",
        "title",
        "channel_ids",
        "possible_start_ms",
        "observed_start_ms",
        "observed_end_ms",
        "possible_end_ms",
        *_JSON_OBJECT_FIELDS,
        *_JSON_ARRAY_FIELDS,
    }
)


class IncidentStoreNotReady(RuntimeError):
    """Raised when the incident migration has not been applied."""


class IncidentRevisionConflict(RuntimeError):
    """Raised when an update uses a stale optimistic revision."""

    def __init__(self, incident_id: str, expected: int, actual: int) -> None:
        self.incident_id = incident_id
        self.expected_revision = expected
        self.actual_revision = actual
        super().__init__(
            f"incident {incident_id} revision is {actual}, expected {expected}"
        )


def _uuid_text(value: Any, field_name: str) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"{field_name} must be a UUID") from exc


def _timestamp_ms(value: Any, field_name: str, *, optional: bool = False) -> int | None:
    if value is None and optional:
        return None
    try:
        timestamp = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if timestamp < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return timestamp


def _title(value: Any) -> str:
    text = str(value or "").strip()
    if not text or "\x00" in text or len(text) > 200:
        raise ValueError("title must contain 1 to 200 safe characters")
    return text


def _state(value: Any) -> str:
    state = str(value or "candidate").strip().lower()
    if state not in INCIDENT_STATES:
        raise ValueError(
            "state must be one of: " + ", ".join(sorted(INCIDENT_STATES))
        )
    return state


def _channel_ids(value: Any) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise ValueError("channel_ids must be a list")
    channels: list[int] = []
    for raw in value:
        try:
            channel_id = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("channel_ids must contain integers") from exc
        if channel_id <= 0:
            raise ValueError("channel_ids must contain positive integers")
        if channel_id not in channels:
            channels.append(channel_id)
    if not 1 <= len(channels) <= _MAX_CHANNELS:
        raise ValueError(f"channel_ids must contain 1 to {_MAX_CHANNELS} channels")
    return channels


def _json_value(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and value.lstrip().lower().startswith("data:image/"):
            raise ValueError(f"{path} must reference images instead of embedding them")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise ValueError(f"{path} object keys must be strings")
            key = raw_key.strip()
            if not key:
                raise ValueError(f"{path} object keys must not be empty")
            if key.lower() in _DISALLOWED_JSON_KEYS:
                raise ValueError(f"{path}.{key} must be stored as a reference")
            normalized[key] = _json_value(item, f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_REF_ITEMS:
            raise ValueError(f"{path} must contain at most {_MAX_REF_ITEMS} items")
        return [_json_value(item, f"{path}[]") for item in value]
    raise ValueError(f"{path} must contain JSON-compatible values only")


def _bounded_json(value: Any, field_name: str, expected: type) -> Any:
    normalized = _json_value(value, field_name)
    if not isinstance(normalized, expected):
        kind = "object" if expected is dict else "array"
        raise ValueError(f"{field_name} must be a JSON {kind}")
    encoded = json.dumps(
        normalized,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > _JSON_LIMIT_BYTES:
        raise ValueError(f"{field_name} exceeds {_JSON_LIMIT_BYTES} bytes")
    return normalized


def _json_object(value: Any, field_name: str) -> dict[str, Any]:
    return _bounded_json(value if value is not None else {}, field_name, dict)


def _json_array(value: Any, field_name: str) -> list[Any]:
    return _bounded_json(value if value is not None else [], field_name, list)


def _jsonb(value: Any) -> Any:
    from psycopg.types.json import Jsonb

    return Jsonb(value)


def _decode_json(value: Any, expected: type) -> Any:
    if isinstance(value, expected):
        return value.copy() if isinstance(value, dict) else list(value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, expected):
            return decoded
    return {} if expected is dict else []


def _normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    possible_start = _timestamp_ms(record.get("possible_start_ms"), "possible_start_ms")
    assert possible_start is not None
    observed_start = _timestamp_ms(
        record.get("observed_start_ms"), "observed_start_ms", optional=True
    )
    observed_end = _timestamp_ms(
        record.get("observed_end_ms"), "observed_end_ms", optional=True
    )
    possible_end = _timestamp_ms(
        record.get("possible_end_ms"), "possible_end_ms", optional=True
    )
    if observed_start is not None and observed_start < possible_start:
        raise ValueError("observed_start_ms must not precede possible_start_ms")
    start_floor = observed_start if observed_start is not None else possible_start
    if observed_end is not None and observed_end < start_floor:
        raise ValueError("observed_end_ms must not precede incident start")
    end_floor = (
        observed_end
        if observed_end is not None
        else observed_start
        if observed_start is not None
        else possible_start
    )
    if possible_end is not None and possible_end < end_floor:
        raise ValueError("possible_end_ms must not precede observed incident time")

    return {
        "id": _uuid_text(record.get("id") or uuid.uuid4(), "id"),
        "revision": 1,
        "state": _state(record.get("state")),
        "title": _title(record.get("title")),
        "channel_ids": _channel_ids(record.get("channel_ids")),
        "possible_start_ms": possible_start,
        "observed_start_ms": observed_start,
        "observed_end_ms": observed_end,
        "possible_end_ms": possible_end,
        "anchor_ref": _json_object(record.get("anchor_ref"), "anchor_ref"),
        "timeline_refs": _json_array(record.get("timeline_refs"), "timeline_refs"),
        "evidence_refs": _json_array(record.get("evidence_refs"), "evidence_refs"),
        "qualia_refs": _json_array(record.get("qualia_refs"), "qualia_refs"),
        "coverage": _json_object(record.get("coverage"), "coverage"),
        "uncertainties": _json_array(record.get("uncertainties"), "uncertainties"),
        "report": _json_object(record.get("report"), "report"),
        "follow_policy": _json_object(record.get("follow_policy"), "follow_policy"),
    }


def _is_missing_incident_relation(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return (
        "undefinedtable" in name
        or "undefined_table" in text
        or "does not exist" in text
    ) and "archive.incidents" in text


class PostgresIncidentStore:
    """Validated PostgreSQL repository with optimistic incident revisions."""

    backend = "postgres"
    _SELECT_COLUMNS = """
        id::text, revision, state, title, channel_ids,
        possible_start_ms, observed_start_ms, observed_end_ms, possible_end_ms,
        anchor_ref, timeline_refs, evidence_refs, qualia_refs, coverage_json,
        uncertainties_json, report_json, follow_policy_json,
        created_by::text, updated_by::text, created_at, updated_at
    """

    def __init__(
        self,
        pool: PsycopgPool,
        tenant_id: str | uuid.UUID,
        *,
        actor_id: str | uuid.UUID | None = None,
    ) -> None:
        self.pool = pool
        self.tenant_id = _uuid_text(tenant_id, "tenant_id")
        self.actor_id = _uuid_text(actor_id or NIL_UUID, "actor_id")

    def _context(self, actor_id: str | uuid.UUID | None = None) -> TransactionContext:
        return TransactionContext(
            tenant_id=self.tenant_id,
            actor_id=_uuid_text(actor_id or self.actor_id, "actor_id"),
        )

    @classmethod
    def _row_to_dict(cls, row: Sequence[Any]) -> dict[str, Any]:
        return {
            "id": str(row[0]),
            "revision": int(row[1]),
            "state": str(row[2]),
            "title": str(row[3]),
            "channel_ids": [int(item) for item in (row[4] or [])],
            "possible_start_ms": int(row[5]),
            "observed_start_ms": int(row[6]) if row[6] is not None else None,
            "observed_end_ms": int(row[7]) if row[7] is not None else None,
            "possible_end_ms": int(row[8]) if row[8] is not None else None,
            "anchor_ref": _decode_json(row[9], dict),
            "timeline_refs": _decode_json(row[10], list),
            "evidence_refs": _decode_json(row[11], list),
            "qualia_refs": _decode_json(row[12], list),
            "coverage": _decode_json(row[13], dict),
            "uncertainties": _decode_json(row[14], list),
            "report": _decode_json(row[15], dict),
            "follow_policy": _decode_json(row[16], dict),
            "created_by": str(row[17]),
            "updated_by": str(row[18]),
            "created_at": row[19].isoformat() if hasattr(row[19], "isoformat") else str(row[19]),
            "updated_at": row[20].isoformat() if hasattr(row[20], "isoformat") else str(row[20]),
        }

    def health(self) -> dict[str, Any]:
        try:
            with self.pool.transaction(self._context(), readonly=True) as connection:
                connection.execute("SELECT 1 FROM archive.incidents LIMIT 1")
        except Exception as exc:
            return {
                "ok": False,
                "backend": self.backend,
                "status": "not_migrated" if _is_missing_incident_relation(exc) else "unavailable",
                "required_revision": INCIDENT_STORAGE_REVISION,
                "error": type(exc).__name__,
            }
        return {
            "ok": True,
            "backend": self.backend,
            "status": "reachable",
            "tenant_id": self.tenant_id,
        }

    def create_incident(
        self,
        record: Mapping[str, Any],
        *,
        actor_id: str | uuid.UUID | None = None,
    ) -> dict[str, Any]:
        normalized = _normalize_record(record)
        actor = _uuid_text(actor_id or self.actor_id, "actor_id")
        try:
            with self.pool.transaction(self._context(actor)) as connection:
                row = connection.execute(
                    f"""
                    INSERT INTO archive.incidents (
                        tenant_id, id, revision, state, title, channel_ids,
                        possible_start_ms, observed_start_ms, observed_end_ms,
                        possible_end_ms, anchor_ref, timeline_refs, evidence_refs,
                        qualia_refs, coverage_json, uncertainties_json, report_json,
                        follow_policy_json, created_by, updated_by
                    )
                    VALUES (
                        %s, %s, 1, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s
                    )
                    RETURNING {self._SELECT_COLUMNS}
                    """,
                    (
                        self.tenant_id,
                        normalized["id"],
                        normalized["state"],
                        normalized["title"],
                        normalized["channel_ids"],
                        normalized["possible_start_ms"],
                        normalized["observed_start_ms"],
                        normalized["observed_end_ms"],
                        normalized["possible_end_ms"],
                        _jsonb(normalized["anchor_ref"]),
                        _jsonb(normalized["timeline_refs"]),
                        _jsonb(normalized["evidence_refs"]),
                        _jsonb(normalized["qualia_refs"]),
                        _jsonb(normalized["coverage"]),
                        _jsonb(normalized["uncertainties"]),
                        _jsonb(normalized["report"]),
                        _jsonb(normalized["follow_policy"]),
                        actor,
                        actor,
                    ),
                ).fetchone()
        except Exception as exc:
            if _is_missing_incident_relation(exc):
                raise IncidentStoreNotReady(
                    f"Incident storage requires migration {INCIDENT_STORAGE_REVISION}."
                ) from exc
            raise
        if row is None:
            raise RuntimeError("incident insert returned no row")
        return self._row_to_dict(row)

    def get_incident(self, incident_id: str | uuid.UUID) -> dict[str, Any] | None:
        normalized_id = _uuid_text(incident_id, "incident_id")
        try:
            with self.pool.transaction(self._context(), readonly=True) as connection:
                row = connection.execute(
                    f"""
                    SELECT {self._SELECT_COLUMNS}
                    FROM archive.incidents
                    WHERE tenant_id = %s AND id = %s
                    """,
                    (self.tenant_id, normalized_id),
                ).fetchone()
        except Exception as exc:
            if _is_missing_incident_relation(exc):
                raise IncidentStoreNotReady(
                    f"Incident storage requires migration {INCIDENT_STORAGE_REVISION}."
                ) from exc
            raise
        return self._row_to_dict(row) if row is not None else None

    def update_incident(
        self,
        incident_id: str | uuid.UUID,
        *,
        expected_revision: int,
        changes: Mapping[str, Any],
        actor_id: str | uuid.UUID | None = None,
    ) -> dict[str, Any]:
        normalized_id = _uuid_text(incident_id, "incident_id")
        revision = int(expected_revision)
        if revision <= 0:
            raise ValueError("expected_revision must be positive")
        unknown = set(map(str, changes)).difference(_MUTABLE_FIELDS)
        if unknown:
            raise ValueError("unknown incident fields: " + ", ".join(sorted(unknown)))
        if not changes:
            raise ValueError("changes must not be empty")
        actor = _uuid_text(actor_id or self.actor_id, "actor_id")
        try:
            with self.pool.transaction(self._context(actor)) as connection:
                current_row = connection.execute(
                    f"""
                    SELECT {self._SELECT_COLUMNS}
                    FROM archive.incidents
                    WHERE tenant_id = %s AND id = %s
                    FOR UPDATE
                    """,
                    (self.tenant_id, normalized_id),
                ).fetchone()
                if current_row is None:
                    raise LookupError("incident not found")
                current = self._row_to_dict(current_row)
                actual_revision = int(current["revision"])
                if actual_revision != revision:
                    raise IncidentRevisionConflict(
                        normalized_id, revision, actual_revision
                    )
                merged = {key: current.get(key) for key in _MUTABLE_FIELDS}
                merged.update(dict(changes))
                merged["id"] = normalized_id
                normalized = _normalize_record(merged)
                row = connection.execute(
                    f"""
                    UPDATE archive.incidents
                    SET revision = revision + 1,
                        state = %s,
                        title = %s,
                        channel_ids = %s,
                        possible_start_ms = %s,
                        observed_start_ms = %s,
                        observed_end_ms = %s,
                        possible_end_ms = %s,
                        anchor_ref = %s,
                        timeline_refs = %s,
                        evidence_refs = %s,
                        qualia_refs = %s,
                        coverage_json = %s,
                        uncertainties_json = %s,
                        report_json = %s,
                        follow_policy_json = %s,
                        updated_by = %s,
                        updated_at = clock_timestamp()
                    WHERE tenant_id = %s AND id = %s AND revision = %s
                    RETURNING {self._SELECT_COLUMNS}
                    """,
                    (
                        normalized["state"],
                        normalized["title"],
                        normalized["channel_ids"],
                        normalized["possible_start_ms"],
                        normalized["observed_start_ms"],
                        normalized["observed_end_ms"],
                        normalized["possible_end_ms"],
                        _jsonb(normalized["anchor_ref"]),
                        _jsonb(normalized["timeline_refs"]),
                        _jsonb(normalized["evidence_refs"]),
                        _jsonb(normalized["qualia_refs"]),
                        _jsonb(normalized["coverage"]),
                        _jsonb(normalized["uncertainties"]),
                        _jsonb(normalized["report"]),
                        _jsonb(normalized["follow_policy"]),
                        actor,
                        self.tenant_id,
                        normalized_id,
                        revision,
                    ),
                ).fetchone()
                if row is None:
                    raise IncidentRevisionConflict(
                        normalized_id, revision, revision + 1
                    )
        except (LookupError, IncidentRevisionConflict):
            raise
        except Exception as exc:
            if _is_missing_incident_relation(exc):
                raise IncidentStoreNotReady(
                    f"Incident storage requires migration {INCIDENT_STORAGE_REVISION}."
                ) from exc
            raise
        return self._row_to_dict(row)

    def list_incidents(
        self,
        *,
        channel_ids: Sequence[int] | None = None,
        states: Sequence[str] | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        clauses = ["tenant_id = %s"]
        params: list[Any] = [self.tenant_id]
        if channel_ids is not None:
            channels = _channel_ids(channel_ids)
            clauses.append("channel_ids && %s::bigint[]")
            params.append(channels)
        if states is not None:
            normalized_states = sorted({_state(item) for item in states})
            if not normalized_states:
                raise ValueError("states must not be empty")
            clauses.append("state = ANY(%s)")
            params.append(normalized_states)
        start = _timestamp_ms(since_ms, "since_ms", optional=True)
        end = _timestamp_ms(until_ms, "until_ms", optional=True)
        if start is not None and end is not None and start > end:
            raise ValueError("since_ms must not be later than until_ms")
        if start is not None:
            clauses.append(
                "COALESCE(possible_end_ms, observed_end_ms, observed_start_ms, possible_start_ms) >= %s"
            )
            params.append(start)
        if end is not None:
            clauses.append("possible_start_ms <= %s")
            params.append(end)
        bounded_limit = max(1, min(500, int(limit or 100)))
        bounded_offset = max(0, int(offset or 0))
        where_sql = "WHERE " + " AND ".join(clauses)
        try:
            with self.pool.transaction(self._context(), readonly=True) as connection:
                total_row = connection.execute(
                    f"SELECT COUNT(*) FROM archive.incidents {where_sql}",
                    tuple(params),
                ).fetchone()
                rows = connection.execute(
                    f"""
                    SELECT {self._SELECT_COLUMNS}
                    FROM archive.incidents
                    {where_sql}
                    ORDER BY possible_start_ms DESC, updated_at DESC, id ASC
                    LIMIT %s OFFSET %s
                    """,
                    tuple(params + [bounded_limit, bounded_offset]),
                ).fetchall()
        except Exception as exc:
            if _is_missing_incident_relation(exc):
                raise IncidentStoreNotReady(
                    f"Incident storage requires migration {INCIDENT_STORAGE_REVISION}."
                ) from exc
            raise
        total = int(total_row[0] or 0) if total_row else 0
        return [self._row_to_dict(row) for row in rows], total


__all__ = [
    "INCIDENT_STATES",
    "INCIDENT_STORAGE_REVISION",
    "IncidentRevisionConflict",
    "IncidentStoreNotReady",
    "PostgresIncidentStore",
]
