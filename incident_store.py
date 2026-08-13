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


INCIDENT_STORAGE_REVISION = "20260805_0012"
NIL_UUID = uuid.UUID(int=0)
INCIDENT_STATES = frozenset(
    {"candidate", "draft", "following", "ended", "reported", "closed"}
)
INCIDENT_PERCEPTION_STATES = frozenset(
    {"unknown", "observed", "not_observed", "ended"}
)
INCIDENT_RISK_STATES = frozenset(
    {"unknown", "active", "contained", "resolved", "occurred"}
)
INCIDENT_CASE_STATES = frozenset(
    {"unknown", "candidate", "open", "closed", "dismissed", "false_positive"}
)
INCIDENT_ATTENTION_STATES = frozenset(
    {"unknown", "inactive", "follow", "critical"}
)
INCIDENT_RELATION_TYPES = frozenset(
    {
        "series_member",
        "caused_by",
        "concurrent_with",
        "possible_same_as",
        "merged_into",
        "split_from",
        "supersedes",
    }
)
INCIDENT_RELATION_STATES = frozenset({"candidate", "confirmed", "rejected"})
INCIDENT_RELATION_CONFIDENCE = frozenset({"unknown", "low", "medium", "high"})
INCIDENT_TRANSITION_STATES = {
    "perception": INCIDENT_PERCEPTION_STATES,
    "risk": INCIDENT_RISK_STATES,
    "case": INCIDENT_CASE_STATES,
    "attention": INCIDENT_ATTENTION_STATES,
    "legacy": INCIDENT_STATES,
}
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
        "perception_state",
        "risk_state",
        "case_state",
        "attention_state",
        "identity_key",
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


class IncidentIdempotencyConflict(RuntimeError):
    """Raised when an idempotency key is replayed with different content."""

    def __init__(self, idempotency_key: str) -> None:
        self.idempotency_key = idempotency_key
        super().__init__(
            f"idempotency key {idempotency_key!r} already identifies different content"
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


def _enum_state(
    value: Any,
    field_name: str,
    allowed: frozenset[str],
) -> str:
    state = str(value or "unknown").strip().lower()
    if state not in allowed:
        raise ValueError(
            f"{field_name} must be one of: " + ", ".join(sorted(allowed))
        )
    return state


def _optional_key(value: Any, field_name: str, *, maximum: int = 200) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or "\x00" in text or len(text) > maximum:
        raise ValueError(
            f"{field_name} must contain 1 to {maximum} safe characters when set"
        )
    return text


def _required_key(value: Any, field_name: str, *, maximum: int = 200) -> str:
    text = _optional_key(value, field_name, maximum=maximum)
    if text is None:
        raise ValueError(f"{field_name} is required")
    return text


def _optional_channel_id(value: Any) -> int | None:
    if value is None:
        return None
    try:
        channel_id = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("channel_id must be a positive integer") from exc
    if channel_id <= 0:
        raise ValueError("channel_id must be a positive integer")
    return channel_id


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
        "perception_state": _enum_state(
            record.get("perception_state"),
            "perception_state",
            INCIDENT_PERCEPTION_STATES,
        ),
        "risk_state": _enum_state(
            record.get("risk_state"),
            "risk_state",
            INCIDENT_RISK_STATES,
        ),
        "case_state": _enum_state(
            record.get("case_state"),
            "case_state",
            INCIDENT_CASE_STATES,
        ),
        "attention_state": _enum_state(
            record.get("attention_state"),
            "attention_state",
            INCIDENT_ATTENTION_STATES,
        ),
        "identity_key": _optional_key(record.get("identity_key"), "identity_key"),
        "idempotency_key": _optional_key(
            record.get("idempotency_key"), "idempotency_key"
        ),
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


def _normalize_observation(record: Mapping[str, Any]) -> dict[str, Any]:
    observed_at_ms = _timestamp_ms(record.get("observed_at_ms"), "observed_at_ms")
    assert observed_at_ms is not None
    return {
        "id": _uuid_text(record.get("id") or uuid.uuid4(), "id"),
        "incident_id": _uuid_text(record.get("incident_id"), "incident_id"),
        "idempotency_key": _required_key(
            record.get("idempotency_key"), "idempotency_key"
        ),
        "source_kind": _required_key(
            record.get("source_kind"), "source_kind", maximum=80
        ),
        "observed_at_ms": observed_at_ms,
        "channel_id": _optional_channel_id(record.get("channel_id")),
        "perception_state": _enum_state(
            record.get("perception_state"),
            "perception_state",
            INCIDENT_PERCEPTION_STATES,
        ),
        "source_ref": _json_object(record.get("source_ref"), "source_ref"),
        "payload": _json_object(record.get("payload"), "payload"),
    }


def _normalize_episode(record: Mapping[str, Any]) -> dict[str, Any]:
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
        raise ValueError("observed_end_ms must not precede episode start")
    end_floor = observed_end if observed_end is not None else start_floor
    if possible_end is not None and possible_end < end_floor:
        raise ValueError("possible_end_ms must not precede observed episode time")
    return {
        "id": _uuid_text(record.get("id") or uuid.uuid4(), "id"),
        "incident_id": _uuid_text(record.get("incident_id"), "incident_id"),
        "idempotency_key": _required_key(
            record.get("idempotency_key"), "idempotency_key"
        ),
        "episode_key": _required_key(record.get("episode_key"), "episode_key"),
        "perception_state": _enum_state(
            record.get("perception_state"),
            "perception_state",
            INCIDENT_PERCEPTION_STATES,
        ),
        "semantic_key": _optional_key(
            record.get("semantic_key"), "semantic_key", maximum=160
        ),
        "entity_key": _optional_key(
            record.get("entity_key"), "entity_key", maximum=160
        ),
        "zone_key": _optional_key(
            record.get("zone_key"), "zone_key", maximum=160
        ),
        "possible_start_ms": possible_start,
        "observed_start_ms": observed_start,
        "observed_end_ms": observed_end,
        "possible_end_ms": possible_end,
        "routine_before_ref": _json_object(
            record.get("routine_before_ref"), "routine_before_ref"
        ),
        "routine_after_ref": _json_object(
            record.get("routine_after_ref"), "routine_after_ref"
        ),
        "evidence_refs": _json_array(record.get("evidence_refs"), "evidence_refs"),
        "coverage": _json_object(record.get("coverage"), "coverage"),
    }


def _normalize_relation(record: Mapping[str, Any]) -> dict[str, Any]:
    subject = _uuid_text(record.get("subject_incident_id"), "subject_incident_id")
    object_id = _uuid_text(record.get("object_incident_id"), "object_incident_id")
    if subject == object_id:
        raise ValueError("incident relation endpoints must be distinct")
    relation_type = str(record.get("relation_type") or "").strip().lower()
    if relation_type not in INCIDENT_RELATION_TYPES:
        raise ValueError(
            "relation_type must be one of: "
            + ", ".join(sorted(INCIDENT_RELATION_TYPES))
        )
    relation_state = str(record.get("relation_state") or "candidate").strip().lower()
    if relation_state not in INCIDENT_RELATION_STATES:
        raise ValueError(
            "relation_state must be one of: "
            + ", ".join(sorted(INCIDENT_RELATION_STATES))
        )
    confidence = str(record.get("confidence") or "unknown").strip().lower()
    if confidence not in INCIDENT_RELATION_CONFIDENCE:
        raise ValueError(
            "confidence must be one of: "
            + ", ".join(sorted(INCIDENT_RELATION_CONFIDENCE))
        )
    rationale = str(record.get("rationale") or "").strip()
    if "\x00" in rationale or len(rationale) > 2_000:
        raise ValueError("rationale must contain at most 2000 safe characters")
    return {
        "id": _uuid_text(record.get("id") or uuid.uuid4(), "id"),
        "subject_incident_id": subject,
        "object_incident_id": object_id,
        "idempotency_key": _required_key(
            record.get("idempotency_key"), "idempotency_key"
        ),
        "relation_type": relation_type,
        "relation_state": relation_state,
        "confidence": confidence,
        "rationale": rationale,
        "payload": _json_object(record.get("payload"), "payload"),
    }


def _normalize_transition(record: Mapping[str, Any]) -> dict[str, Any]:
    axis = str(record.get("axis") or "").strip().lower()
    allowed = INCIDENT_TRANSITION_STATES.get(axis)
    if allowed is None:
        raise ValueError(
            "axis must be one of: "
            + ", ".join(sorted(INCIDENT_TRANSITION_STATES))
        )
    from_state = record.get("from_state")
    if from_state is not None:
        from_state = _enum_state(from_state, "from_state", allowed)
    to_state = _enum_state(record.get("to_state"), "to_state", allowed)
    try:
        incident_revision = int(record.get("incident_revision"))
    except (TypeError, ValueError) as exc:
        raise ValueError("incident_revision must be a positive integer") from exc
    if incident_revision <= 0:
        raise ValueError("incident_revision must be a positive integer")
    transitioned_at_ms = _timestamp_ms(
        record.get("transitioned_at_ms"), "transitioned_at_ms"
    )
    assert transitioned_at_ms is not None
    reason = str(record.get("reason") or "").strip()
    if "\x00" in reason or len(reason) > 2_000:
        raise ValueError("reason must contain at most 2000 safe characters")
    return {
        "id": _uuid_text(record.get("id") or uuid.uuid4(), "id"),
        "incident_id": _uuid_text(record.get("incident_id"), "incident_id"),
        "idempotency_key": _required_key(
            record.get("idempotency_key"), "idempotency_key"
        ),
        "axis": axis,
        "from_state": from_state,
        "to_state": to_state,
        "incident_revision": incident_revision,
        "transitioned_at_ms": transitioned_at_ms,
        "reason": reason,
        "source_kind": _required_key(
            record.get("source_kind"), "source_kind", maximum=80
        ),
        "source_ref": _json_object(record.get("source_ref"), "source_ref"),
        "payload": _json_object(record.get("payload"), "payload"),
    }


_INCIDENT_CREATE_COMPARE_FIELDS = (
    "state",
    "perception_state",
    "risk_state",
    "case_state",
    "attention_state",
    "identity_key",
    "idempotency_key",
    "title",
    "channel_ids",
    "possible_start_ms",
    "observed_start_ms",
    "observed_end_ms",
    "possible_end_ms",
    "anchor_ref",
    "timeline_refs",
    "evidence_refs",
    "qualia_refs",
    "coverage",
    "uncertainties",
    "report",
    "follow_policy",
)


def _same_incident_create(
    existing: Mapping[str, Any],
    normalized: Mapping[str, Any],
) -> bool:
    return all(
        existing.get(field) == normalized.get(field)
        for field in _INCIDENT_CREATE_COMPARE_FIELDS
    )


def _same_observation(
    existing: Mapping[str, Any],
    normalized: Mapping[str, Any],
) -> bool:
    return all(
        existing.get(field) == normalized.get(field)
        for field in (
            "incident_id",
            "idempotency_key",
            "source_kind",
            "observed_at_ms",
            "channel_id",
            "perception_state",
            "source_ref",
            "payload",
        )
    )


def _same_episode(existing: Mapping[str, Any], normalized: Mapping[str, Any]) -> bool:
    return all(
        existing.get(field) == normalized.get(field)
        for field in (
            "incident_id",
            "idempotency_key",
            "episode_key",
            "perception_state",
            "semantic_key",
            "entity_key",
            "zone_key",
            "possible_start_ms",
            "observed_start_ms",
            "observed_end_ms",
            "possible_end_ms",
            "routine_before_ref",
            "routine_after_ref",
            "evidence_refs",
            "coverage",
        )
    )


def _same_relation(existing: Mapping[str, Any], normalized: Mapping[str, Any]) -> bool:
    return all(
        existing.get(field) == normalized.get(field)
        for field in (
            "subject_incident_id",
            "object_incident_id",
            "idempotency_key",
            "relation_type",
            "relation_state",
            "confidence",
            "rationale",
            "payload",
        )
    )


def _same_transition(
    existing: Mapping[str, Any], normalized: Mapping[str, Any]
) -> bool:
    return all(
        existing.get(field) == normalized.get(field)
        for field in (
            "incident_id",
            "idempotency_key",
            "axis",
            "from_state",
            "to_state",
            "incident_revision",
            "transitioned_at_ms",
            "reason",
            "source_kind",
            "source_ref",
            "payload",
        )
    )


def _is_incident_schema_not_ready(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    missing_relation = (
        "undefinedtable" in name
        or "undefined_table" in text
        or "does not exist" in text
    ) and "archive.incident" in text
    missing_lifecycle_column = (
        "undefinedcolumn" in name or "undefined_column" in text
    ) and any(
        column in text
        for column in (
            "perception_state",
            "risk_state",
            "case_state",
            "attention_state",
            "identity_key",
            "idempotency_key",
        )
    )
    return missing_relation or missing_lifecycle_column


class PostgresIncidentStore:
    """Validated PostgreSQL repository with optimistic incident revisions."""

    backend = "postgres"
    _SELECT_COLUMNS = """
        id::text, revision, state, title, channel_ids,
        possible_start_ms, observed_start_ms, observed_end_ms, possible_end_ms,
        anchor_ref, timeline_refs, evidence_refs, qualia_refs, coverage_json,
        uncertainties_json, report_json, follow_policy_json,
        perception_state, risk_state, case_state, attention_state,
        identity_key, idempotency_key,
        created_by::text, updated_by::text, created_at, updated_at
    """
    _OBSERVATION_SELECT_COLUMNS = """
        id::text, incident_id::text, idempotency_key, source_kind,
        observed_at_ms, channel_id, perception_state, source_ref, payload_json,
        created_by::text, created_at
    """
    _EPISODE_SELECT_COLUMNS = """
        id::text, incident_id::text, idempotency_key, episode_key,
        perception_state, semantic_key, entity_key, zone_key,
        possible_start_ms, observed_start_ms, observed_end_ms, possible_end_ms,
        routine_before_ref, routine_after_ref, evidence_refs, coverage_json,
        created_by::text, created_at
    """
    _RELATION_SELECT_COLUMNS = """
        id::text, subject_incident_id::text, object_incident_id::text,
        idempotency_key, relation_type, relation_state, confidence, rationale,
        payload_json, created_by::text, created_at
    """
    _TRANSITION_SELECT_COLUMNS = """
        id::text, incident_id::text, idempotency_key, axis, from_state,
        to_state, incident_revision, transitioned_at_ms, reason, source_kind,
        source_ref, payload_json, created_by::text, created_at
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
            "perception_state": str(row[17]),
            "risk_state": str(row[18]),
            "case_state": str(row[19]),
            "attention_state": str(row[20]),
            "identity_key": str(row[21]) if row[21] is not None else None,
            "idempotency_key": str(row[22]) if row[22] is not None else None,
            "created_by": str(row[23]),
            "updated_by": str(row[24]),
            "created_at": row[25].isoformat() if hasattr(row[25], "isoformat") else str(row[25]),
            "updated_at": row[26].isoformat() if hasattr(row[26], "isoformat") else str(row[26]),
        }

    @classmethod
    def _observation_row_to_dict(cls, row: Sequence[Any]) -> dict[str, Any]:
        return {
            "id": str(row[0]),
            "incident_id": str(row[1]),
            "idempotency_key": str(row[2]),
            "source_kind": str(row[3]),
            "observed_at_ms": int(row[4]),
            "channel_id": int(row[5]) if row[5] is not None else None,
            "perception_state": str(row[6]),
            "source_ref": _decode_json(row[7], dict),
            "payload": _decode_json(row[8], dict),
            "created_by": str(row[9]),
            "created_at": row[10].isoformat() if hasattr(row[10], "isoformat") else str(row[10]),
        }

    @classmethod
    def _episode_row_to_dict(cls, row: Sequence[Any]) -> dict[str, Any]:
        return {
            "id": str(row[0]),
            "incident_id": str(row[1]),
            "idempotency_key": str(row[2]),
            "episode_key": str(row[3]),
            "perception_state": str(row[4]),
            "semantic_key": str(row[5]) if row[5] is not None else None,
            "entity_key": str(row[6]) if row[6] is not None else None,
            "zone_key": str(row[7]) if row[7] is not None else None,
            "possible_start_ms": int(row[8]),
            "observed_start_ms": int(row[9]) if row[9] is not None else None,
            "observed_end_ms": int(row[10]) if row[10] is not None else None,
            "possible_end_ms": int(row[11]) if row[11] is not None else None,
            "routine_before_ref": _decode_json(row[12], dict),
            "routine_after_ref": _decode_json(row[13], dict),
            "evidence_refs": _decode_json(row[14], list),
            "coverage": _decode_json(row[15], dict),
            "created_by": str(row[16]),
            "created_at": row[17].isoformat() if hasattr(row[17], "isoformat") else str(row[17]),
        }

    @classmethod
    def _relation_row_to_dict(cls, row: Sequence[Any]) -> dict[str, Any]:
        return {
            "id": str(row[0]),
            "subject_incident_id": str(row[1]),
            "object_incident_id": str(row[2]),
            "idempotency_key": str(row[3]),
            "relation_type": str(row[4]),
            "relation_state": str(row[5]),
            "confidence": str(row[6]),
            "rationale": str(row[7]),
            "payload": _decode_json(row[8], dict),
            "created_by": str(row[9]),
            "created_at": row[10].isoformat() if hasattr(row[10], "isoformat") else str(row[10]),
        }

    @classmethod
    def _transition_row_to_dict(cls, row: Sequence[Any]) -> dict[str, Any]:
        return {
            "id": str(row[0]),
            "incident_id": str(row[1]),
            "idempotency_key": str(row[2]),
            "axis": str(row[3]),
            "from_state": str(row[4]) if row[4] is not None else None,
            "to_state": str(row[5]),
            "incident_revision": int(row[6]),
            "transitioned_at_ms": int(row[7]),
            "reason": str(row[8]),
            "source_kind": str(row[9]),
            "source_ref": _decode_json(row[10], dict),
            "payload": _decode_json(row[11], dict),
            "created_by": str(row[12]),
            "created_at": row[13].isoformat() if hasattr(row[13], "isoformat") else str(row[13]),
        }

    def health(self) -> dict[str, Any]:
        try:
            with self.pool.transaction(self._context(), readonly=True) as connection:
                connection.execute(
                    "SELECT perception_state, risk_state, case_state, attention_state "
                    "FROM archive.incidents LIMIT 1"
                )
                connection.execute(
                    "SELECT 1 FROM archive.incident_observations LIMIT 1"
                )
                connection.execute("SELECT 1 FROM archive.incident_episodes LIMIT 1")
                connection.execute("SELECT 1 FROM archive.incident_relations LIMIT 1")
                connection.execute("SELECT 1 FROM archive.incident_transitions LIMIT 1")
        except Exception as exc:
            return {
                "ok": False,
                "backend": self.backend,
                "status": "not_migrated" if _is_incident_schema_not_ready(exc) else "unavailable",
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
                        follow_policy_json, perception_state, risk_state,
                        case_state, attention_state, identity_key, idempotency_key,
                        created_by, updated_by
                    )
                    VALUES (
                        %s, %s, 1, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT DO NOTHING
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
                        normalized["perception_state"],
                        normalized["risk_state"],
                        normalized["case_state"],
                        normalized["attention_state"],
                        normalized["identity_key"],
                        normalized["idempotency_key"],
                        actor,
                        actor,
                    ),
                ).fetchone()
                if row is None and (
                    normalized["idempotency_key"] is not None
                    or normalized["identity_key"] is not None
                ):
                    row = connection.execute(
                        f"""
                        SELECT {self._SELECT_COLUMNS}
                        FROM archive.incidents
                        WHERE tenant_id = %s AND (
                            (%s::text IS NOT NULL AND idempotency_key = %s)
                            OR (%s::text IS NOT NULL AND identity_key = %s)
                        )
                        ORDER BY created_at ASC, id ASC
                        LIMIT 1
                        """,
                        (
                            self.tenant_id,
                            normalized["idempotency_key"],
                            normalized["idempotency_key"],
                            normalized["identity_key"],
                            normalized["identity_key"],
                        ),
                    ).fetchone()
                    if row is not None:
                        existing = self._row_to_dict(row)
                        if not _same_incident_create(existing, normalized):
                            raise IncidentIdempotencyConflict(
                                normalized["idempotency_key"]
                                or normalized["identity_key"]
                                or "unknown"
                            )
        except Exception as exc:
            if _is_incident_schema_not_ready(exc):
                raise IncidentStoreNotReady(
                    f"Incident storage requires migration {INCIDENT_STORAGE_REVISION}."
                ) from exc
            raise
        if row is None:
            raise RuntimeError(
                "incident insert conflicted without a matching identity/idempotency key"
            )
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
            if _is_incident_schema_not_ready(exc):
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
        transition: Mapping[str, Any] | None = None,
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
                if "identity_key" in changes:
                    requested_identity = _optional_key(
                        changes.get("identity_key"), "identity_key"
                    )
                    current_identity = current.get("identity_key")
                    if (
                        current_identity is not None
                        and requested_identity != current_identity
                    ):
                        raise ValueError(
                            "identity_key is immutable once assigned"
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
                        perception_state = %s,
                        risk_state = %s,
                        case_state = %s,
                        attention_state = %s,
                        identity_key = %s,
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
                        normalized["perception_state"],
                        normalized["risk_state"],
                        normalized["case_state"],
                        normalized["attention_state"],
                        normalized["identity_key"],
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
                updated = self._row_to_dict(row)
                if transition is not None:
                    source_kind = _required_key(
                        transition.get("source_kind"),
                        "transition.source_kind",
                        maximum=80,
                    )
                    transitioned_at_ms = _timestamp_ms(
                        transition.get("transitioned_at_ms"),
                        "transition.transitioned_at_ms",
                    )
                    assert transitioned_at_ms is not None
                    reason = str(transition.get("reason") or "").strip()
                    source_ref = _json_object(
                        transition.get("source_ref"), "transition.source_ref"
                    )
                    payload = _json_object(
                        transition.get("payload"), "transition.payload"
                    )
                    for axis, field in (
                        ("legacy", "state"),
                        ("perception", "perception_state"),
                        ("risk", "risk_state"),
                        ("case", "case_state"),
                        ("attention", "attention_state"),
                    ):
                        if current.get(field) == updated.get(field):
                            continue
                        normalized_transition = _normalize_transition(
                            {
                                "incident_id": normalized_id,
                                "idempotency_key": (
                                    f"revision:{updated['revision']}:{axis}"
                                ),
                                "axis": axis,
                                "from_state": current.get(field),
                                "to_state": updated.get(field),
                                "incident_revision": updated["revision"],
                                "transitioned_at_ms": transitioned_at_ms,
                                "reason": reason,
                                "source_kind": source_kind,
                                "source_ref": source_ref,
                                "payload": payload,
                            }
                        )
                        self._insert_transition(
                            connection, normalized_transition, actor
                        )
        except (LookupError, IncidentRevisionConflict):
            raise
        except Exception as exc:
            if _is_incident_schema_not_ready(exc):
                raise IncidentStoreNotReady(
                    f"Incident storage requires migration {INCIDENT_STORAGE_REVISION}."
                ) from exc
            raise
        return updated

    def _insert_transition(
        self,
        connection: Any,
        normalized: Mapping[str, Any],
        actor: str,
    ) -> dict[str, Any]:
        row = connection.execute(
            f"""
            INSERT INTO archive.incident_transitions (
                tenant_id, id, incident_id, idempotency_key, axis,
                from_state, to_state, incident_revision,
                transitioned_at_ms, reason, source_kind, source_ref,
                payload_json, created_by
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (tenant_id, incident_id, idempotency_key)
            DO NOTHING
            RETURNING {self._TRANSITION_SELECT_COLUMNS}
            """,
            (
                self.tenant_id,
                normalized["id"],
                normalized["incident_id"],
                normalized["idempotency_key"],
                normalized["axis"],
                normalized["from_state"],
                normalized["to_state"],
                normalized["incident_revision"],
                normalized["transitioned_at_ms"],
                normalized["reason"],
                normalized["source_kind"],
                _jsonb(normalized["source_ref"]),
                _jsonb(normalized["payload"]),
                actor,
            ),
        ).fetchone()
        if row is None:
            row = connection.execute(
                f"""
                SELECT {self._TRANSITION_SELECT_COLUMNS}
                FROM archive.incident_transitions
                WHERE tenant_id = %s AND incident_id = %s
                  AND idempotency_key = %s
                """,
                (
                    self.tenant_id,
                    normalized["incident_id"],
                    normalized["idempotency_key"],
                ),
            ).fetchone()
            if row is not None:
                existing = self._transition_row_to_dict(row)
                if not _same_transition(existing, normalized):
                    raise IncidentIdempotencyConflict(
                        str(normalized["idempotency_key"])
                    )
        if row is None:
            raise RuntimeError("incident transition insert returned no row")
        return self._transition_row_to_dict(row)

    def append_observation(
        self,
        observation: Mapping[str, Any],
        *,
        actor_id: str | uuid.UUID | None = None,
    ) -> dict[str, Any]:
        """Append one immutable observation, replay-safe by idempotency key."""

        normalized = _normalize_observation(observation)
        actor = _uuid_text(actor_id or self.actor_id, "actor_id")
        try:
            with self.pool.transaction(self._context(actor)) as connection:
                row = connection.execute(
                    f"""
                    INSERT INTO archive.incident_observations (
                        tenant_id, id, incident_id, idempotency_key, source_kind,
                        observed_at_ms, channel_id, perception_state, source_ref,
                        payload_json, created_by
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (tenant_id, incident_id, idempotency_key)
                    DO NOTHING
                    RETURNING {self._OBSERVATION_SELECT_COLUMNS}
                    """,
                    (
                        self.tenant_id,
                        normalized["id"],
                        normalized["incident_id"],
                        normalized["idempotency_key"],
                        normalized["source_kind"],
                        normalized["observed_at_ms"],
                        normalized["channel_id"],
                        normalized["perception_state"],
                        _jsonb(normalized["source_ref"]),
                        _jsonb(normalized["payload"]),
                        actor,
                    ),
                ).fetchone()
                if row is None:
                    row = connection.execute(
                        f"""
                        SELECT {self._OBSERVATION_SELECT_COLUMNS}
                        FROM archive.incident_observations
                        WHERE tenant_id = %s
                          AND incident_id = %s
                          AND idempotency_key = %s
                        """,
                        (
                            self.tenant_id,
                            normalized["incident_id"],
                            normalized["idempotency_key"],
                        ),
                    ).fetchone()
                    if row is not None:
                        existing = self._observation_row_to_dict(row)
                        if not _same_observation(existing, normalized):
                            raise IncidentIdempotencyConflict(
                                normalized["idempotency_key"]
                            )
        except Exception as exc:
            if _is_incident_schema_not_ready(exc):
                raise IncidentStoreNotReady(
                    f"Incident storage requires migration {INCIDENT_STORAGE_REVISION}."
                ) from exc
            raise
        if row is None:
            raise RuntimeError("observation insert returned no row")
        return self._observation_row_to_dict(row)

    def list_observations(
        self,
        incident_id: str | uuid.UUID,
        *,
        since_ms: int | None = None,
        until_ms: int | None = None,
        source_kind: str | None = None,
        limit: int = 500,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        normalized_id = _uuid_text(incident_id, "incident_id")
        start = _timestamp_ms(since_ms, "since_ms", optional=True)
        end = _timestamp_ms(until_ms, "until_ms", optional=True)
        if start is not None and end is not None and start > end:
            raise ValueError("since_ms must not be later than until_ms")
        clauses = ["tenant_id = %s", "incident_id = %s"]
        params: list[Any] = [self.tenant_id, normalized_id]
        if start is not None:
            clauses.append("observed_at_ms >= %s")
            params.append(start)
        if end is not None:
            clauses.append("observed_at_ms <= %s")
            params.append(end)
        if source_kind is not None:
            clauses.append("source_kind = %s")
            params.append(
                _required_key(source_kind, "source_kind", maximum=80)
            )
        bounded_limit = max(1, min(2_000, int(limit or 500)))
        bounded_offset = max(0, int(offset or 0))
        where_sql = "WHERE " + " AND ".join(clauses)
        try:
            with self.pool.transaction(self._context(), readonly=True) as connection:
                total_row = connection.execute(
                    f"SELECT COUNT(*) FROM archive.incident_observations {where_sql}",
                    tuple(params),
                ).fetchone()
                rows = connection.execute(
                    f"""
                    SELECT {self._OBSERVATION_SELECT_COLUMNS}
                    FROM archive.incident_observations
                    {where_sql}
                    ORDER BY observed_at_ms ASC, id ASC
                    LIMIT %s OFFSET %s
                    """,
                    tuple(params + [bounded_limit, bounded_offset]),
                ).fetchall()
        except Exception as exc:
            if _is_incident_schema_not_ready(exc):
                raise IncidentStoreNotReady(
                    f"Incident storage requires migration {INCIDENT_STORAGE_REVISION}."
                ) from exc
            raise
        total = int(total_row[0] or 0) if total_row else 0
        return [self._observation_row_to_dict(row) for row in rows], total

    def append_episode(
        self,
        episode: Mapping[str, Any],
        *,
        actor_id: str | uuid.UUID | None = None,
    ) -> dict[str, Any]:
        """Append one replay-safe perceptual episode for an incident."""

        normalized = _normalize_episode(episode)
        actor = _uuid_text(actor_id or self.actor_id, "actor_id")
        try:
            with self.pool.transaction(self._context(actor)) as connection:
                row = connection.execute(
                    f"""
                    INSERT INTO archive.incident_episodes (
                        tenant_id, id, incident_id, idempotency_key, episode_key,
                        perception_state, semantic_key, entity_key, zone_key,
                        possible_start_ms, observed_start_ms, observed_end_ms,
                        possible_end_ms, routine_before_ref, routine_after_ref,
                        evidence_refs, coverage_json, created_by
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT DO NOTHING
                    RETURNING {self._EPISODE_SELECT_COLUMNS}
                    """,
                    (
                        self.tenant_id,
                        normalized["id"],
                        normalized["incident_id"],
                        normalized["idempotency_key"],
                        normalized["episode_key"],
                        normalized["perception_state"],
                        normalized["semantic_key"],
                        normalized["entity_key"],
                        normalized["zone_key"],
                        normalized["possible_start_ms"],
                        normalized["observed_start_ms"],
                        normalized["observed_end_ms"],
                        normalized["possible_end_ms"],
                        _jsonb(normalized["routine_before_ref"]),
                        _jsonb(normalized["routine_after_ref"]),
                        _jsonb(normalized["evidence_refs"]),
                        _jsonb(normalized["coverage"]),
                        actor,
                    ),
                ).fetchone()
                if row is None:
                    row = connection.execute(
                        f"""
                        SELECT {self._EPISODE_SELECT_COLUMNS}
                        FROM archive.incident_episodes
                        WHERE tenant_id = %s AND incident_id = %s
                          AND (idempotency_key = %s OR episode_key = %s)
                        ORDER BY created_at ASC, id ASC
                        LIMIT 1
                        """,
                        (
                            self.tenant_id,
                            normalized["incident_id"],
                            normalized["idempotency_key"],
                            normalized["episode_key"],
                        ),
                    ).fetchone()
                    if row is not None:
                        existing = self._episode_row_to_dict(row)
                        if not _same_episode(existing, normalized):
                            raise IncidentIdempotencyConflict(
                                normalized["idempotency_key"]
                            )
        except Exception as exc:
            if _is_incident_schema_not_ready(exc):
                raise IncidentStoreNotReady(
                    f"Incident storage requires migration {INCIDENT_STORAGE_REVISION}."
                ) from exc
            raise
        if row is None:
            raise RuntimeError("episode insert returned no row")
        return self._episode_row_to_dict(row)

    def list_episodes(
        self,
        incident_id: str | uuid.UUID,
        *,
        limit: int = 500,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        normalized_id = _uuid_text(incident_id, "incident_id")
        bounded_limit = max(1, min(2_000, int(limit or 500)))
        bounded_offset = max(0, int(offset or 0))
        try:
            with self.pool.transaction(self._context(), readonly=True) as connection:
                total_row = connection.execute(
                    "SELECT COUNT(*) FROM archive.incident_episodes "
                    "WHERE tenant_id = %s AND incident_id = %s",
                    (self.tenant_id, normalized_id),
                ).fetchone()
                rows = connection.execute(
                    f"""
                    SELECT {self._EPISODE_SELECT_COLUMNS}
                    FROM archive.incident_episodes
                    WHERE tenant_id = %s AND incident_id = %s
                    ORDER BY possible_start_ms ASC, id ASC
                    LIMIT %s OFFSET %s
                    """,
                    (self.tenant_id, normalized_id, bounded_limit, bounded_offset),
                ).fetchall()
        except Exception as exc:
            if _is_incident_schema_not_ready(exc):
                raise IncidentStoreNotReady(
                    f"Incident storage requires migration {INCIDENT_STORAGE_REVISION}."
                ) from exc
            raise
        total = int(total_row[0] or 0) if total_row else 0
        return [self._episode_row_to_dict(row) for row in rows], total

    def append_relation(
        self,
        relation: Mapping[str, Any],
        *,
        actor_id: str | uuid.UUID | None = None,
    ) -> dict[str, Any]:
        """Append a candidate/reviewed relationship without merging incidents."""

        normalized = _normalize_relation(relation)
        actor = _uuid_text(actor_id or self.actor_id, "actor_id")
        try:
            with self.pool.transaction(self._context(actor)) as connection:
                row = connection.execute(
                    f"""
                    INSERT INTO archive.incident_relations (
                        tenant_id, id, subject_incident_id, object_incident_id,
                        idempotency_key, relation_type, relation_state,
                        confidence, rationale, payload_json, created_by
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (tenant_id, subject_incident_id, idempotency_key)
                    DO NOTHING
                    RETURNING {self._RELATION_SELECT_COLUMNS}
                    """,
                    (
                        self.tenant_id,
                        normalized["id"],
                        normalized["subject_incident_id"],
                        normalized["object_incident_id"],
                        normalized["idempotency_key"],
                        normalized["relation_type"],
                        normalized["relation_state"],
                        normalized["confidence"],
                        normalized["rationale"],
                        _jsonb(normalized["payload"]),
                        actor,
                    ),
                ).fetchone()
                if row is None:
                    row = connection.execute(
                        f"""
                        SELECT {self._RELATION_SELECT_COLUMNS}
                        FROM archive.incident_relations
                        WHERE tenant_id = %s AND subject_incident_id = %s
                          AND idempotency_key = %s
                        """,
                        (
                            self.tenant_id,
                            normalized["subject_incident_id"],
                            normalized["idempotency_key"],
                        ),
                    ).fetchone()
                    if row is not None:
                        existing = self._relation_row_to_dict(row)
                        if not _same_relation(existing, normalized):
                            raise IncidentIdempotencyConflict(
                                normalized["idempotency_key"]
                            )
        except Exception as exc:
            if _is_incident_schema_not_ready(exc):
                raise IncidentStoreNotReady(
                    f"Incident storage requires migration {INCIDENT_STORAGE_REVISION}."
                ) from exc
            raise
        if row is None:
            raise RuntimeError("incident relation insert returned no row")
        return self._relation_row_to_dict(row)

    def list_relations(
        self,
        incident_id: str | uuid.UUID,
        *,
        limit: int = 500,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        normalized_id = _uuid_text(incident_id, "incident_id")
        bounded_limit = max(1, min(2_000, int(limit or 500)))
        bounded_offset = max(0, int(offset or 0))
        where_sql = (
            "WHERE tenant_id = %s AND "
            "(subject_incident_id = %s OR object_incident_id = %s)"
        )
        params = (self.tenant_id, normalized_id, normalized_id)
        try:
            with self.pool.transaction(self._context(), readonly=True) as connection:
                total_row = connection.execute(
                    f"SELECT COUNT(*) FROM archive.incident_relations {where_sql}",
                    params,
                ).fetchone()
                rows = connection.execute(
                    f"""
                    SELECT {self._RELATION_SELECT_COLUMNS}
                    FROM archive.incident_relations
                    {where_sql}
                    ORDER BY created_at ASC, id ASC
                    LIMIT %s OFFSET %s
                    """,
                    (*params, bounded_limit, bounded_offset),
                ).fetchall()
        except Exception as exc:
            if _is_incident_schema_not_ready(exc):
                raise IncidentStoreNotReady(
                    f"Incident storage requires migration {INCIDENT_STORAGE_REVISION}."
                ) from exc
            raise
        total = int(total_row[0] or 0) if total_row else 0
        return [self._relation_row_to_dict(row) for row in rows], total

    def append_transition(
        self,
        transition: Mapping[str, Any],
        *,
        actor_id: str | uuid.UUID | None = None,
    ) -> dict[str, Any]:
        """Append one immutable lifecycle transition, replay-safe by revision/axis."""

        normalized = _normalize_transition(transition)
        actor = _uuid_text(actor_id or self.actor_id, "actor_id")
        try:
            with self.pool.transaction(self._context(actor)) as connection:
                stored = self._insert_transition(connection, normalized, actor)
        except Exception as exc:
            if _is_incident_schema_not_ready(exc):
                raise IncidentStoreNotReady(
                    f"Incident storage requires migration {INCIDENT_STORAGE_REVISION}."
                ) from exc
            raise
        return stored

    def list_transitions(
        self,
        incident_id: str | uuid.UUID,
        *,
        limit: int = 500,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        normalized_id = _uuid_text(incident_id, "incident_id")
        bounded_limit = max(1, min(2_000, int(limit or 500)))
        bounded_offset = max(0, int(offset or 0))
        try:
            with self.pool.transaction(self._context(), readonly=True) as connection:
                total_row = connection.execute(
                    "SELECT COUNT(*) FROM archive.incident_transitions "
                    "WHERE tenant_id = %s AND incident_id = %s",
                    (self.tenant_id, normalized_id),
                ).fetchone()
                rows = connection.execute(
                    f"""
                    SELECT {self._TRANSITION_SELECT_COLUMNS}
                    FROM archive.incident_transitions
                    WHERE tenant_id = %s AND incident_id = %s
                    ORDER BY transitioned_at_ms ASC, id ASC
                    LIMIT %s OFFSET %s
                    """,
                    (self.tenant_id, normalized_id, bounded_limit, bounded_offset),
                ).fetchall()
        except Exception as exc:
            if _is_incident_schema_not_ready(exc):
                raise IncidentStoreNotReady(
                    f"Incident storage requires migration {INCIDENT_STORAGE_REVISION}."
                ) from exc
            raise
        total = int(total_row[0] or 0) if total_row else 0
        return [self._transition_row_to_dict(row) for row in rows], total

    def list_incidents(
        self,
        *,
        channel_ids: Sequence[int] | None = None,
        states: Sequence[str] | None = None,
        perception_states: Sequence[str] | None = None,
        risk_states: Sequence[str] | None = None,
        case_states: Sequence[str] | None = None,
        attention_states: Sequence[str] | None = None,
        since_ms: int | None = None,
        until_ms: int | None = None,
        top_level_only: bool = False,
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
        for field_name, values, allowed in (
            ("perception_state", perception_states, INCIDENT_PERCEPTION_STATES),
            ("risk_state", risk_states, INCIDENT_RISK_STATES),
            ("case_state", case_states, INCIDENT_CASE_STATES),
            ("attention_state", attention_states, INCIDENT_ATTENTION_STATES),
        ):
            if values is None:
                continue
            normalized_values = sorted(
                {_enum_state(value, field_name, allowed) for value in values}
            )
            if not normalized_values:
                raise ValueError(f"{field_name}s must not be empty")
            clauses.append(f"{field_name} = ANY(%s)")
            params.append(normalized_values)
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
        if top_level_only:
            # Nested incidents remain first-class rows addressable by id, but
            # they must not consume review-board pagination or totals until an
            # operator or an independently grounded attention signal promotes
            # them. Older rows may already carry the grounded priority while
            # retaining a stale nested marker; include those defensively so a
            # configured alert cannot disappear from review. Keep the predicate
            # in PostgreSQL rather than filtering a bounded page in Python.
            clauses.append(
                "NOT ("
                "COALESCE(report_json #>> '{presentation,scope}', '') = 'nested' "
                "AND COALESCE("
                "report_json #>> '{presentation,parent_incident_id}', ''"
                ") <> '' "
                "AND COALESCE(report_json ->> 'priority', 'context') "
                "NOT IN ('operator_criterion', 'safety')"
                ")"
            )
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
            if _is_incident_schema_not_ready(exc):
                raise IncidentStoreNotReady(
                    f"Incident storage requires migration {INCIDENT_STORAGE_REVISION}."
                ) from exc
            raise
        total = int(total_row[0] or 0) if total_row else 0
        return [self._row_to_dict(row) for row in rows], total


__all__ = [
    "INCIDENT_ATTENTION_STATES",
    "INCIDENT_CASE_STATES",
    "INCIDENT_PERCEPTION_STATES",
    "INCIDENT_RELATION_CONFIDENCE",
    "INCIDENT_RELATION_STATES",
    "INCIDENT_RELATION_TYPES",
    "INCIDENT_RISK_STATES",
    "INCIDENT_STATES",
    "INCIDENT_STORAGE_REVISION",
    "IncidentIdempotencyConflict",
    "IncidentRevisionConflict",
    "IncidentStoreNotReady",
    "PostgresIncidentStore",
]
