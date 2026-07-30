"""Durable, non-blocking storage primitives for EVA attention telemetry.

The capture loop emits dense CV measurements, but this module deliberately stores
only compact motion/quiet intervals and references to the sparse frames that were
embedded or promoted to a VLM apex.  Raw frames and image payloads do not belong in
this data plane.

``PostgresAttentionStore`` owns transactional persistence and bounded time-range
queries. ``BufferedAttentionWriter`` keeps database latency and outages off the
capture hot path by accepting immutable batches into a bounded in-memory queue.
"""

from __future__ import annotations

import json
import math
import threading
import time
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Protocol, Tuple

from eva_db import PsycopgPool, TransactionContext


ATTENTION_STORAGE_REVISION = "20260726_0008"
NIL_UUID = uuid.UUID(int=0)
INTERVAL_STATES = frozenset({"quiet", "motion", "mixed", "unknown", "degraded"})
LINK_KINDS = frozenset({"embedding", "vlm_apex"})
LINK_ROLES = frozenset(
    {"support", "control", "pre", "onset", "apex", "post", "companion"}
)
PROBE_LIFECYCLE_STATES = frozenset(
    {"created", "active", "expired", "retired", "promoted", "rejected"}
)
PROBE_THRESHOLD_STATES = frozenset(
    {
        "hit",
        "below_pos",
        "below_margin",
        "below_both",
        "not_evaluated",
        "suppressed",
    }
)
_DISALLOWED_JSON_KEYS = frozenset(
    {
        "frame_bytes",
        "image_bytes",
        "image_b64",
        "frame_b64",
        "jpeg_b64",
        "png_b64",
        "thumbnail_b64",
    }
)


class AttentionStoreNotReady(RuntimeError):
    """Raised when the attention tables have not been migrated yet."""


def _uuid_text(value: str | uuid.UUID, field_name: str) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"{field_name} must be a UUID") from exc


def _positive_channel(value: int) -> int:
    channel_id = int(value)
    if channel_id <= 0:
        raise ValueError("channel_id must be positive")
    return channel_id


def _timestamp_ms(value: int, field_name: str) -> int:
    timestamp = int(value)
    if timestamp < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return timestamp


def _nonempty_text(value: str, field_name: str, *, maximum: int = 512) -> str:
    text = str(value or "").strip()
    if not text or "\x00" in text or len(text) > maximum:
        raise ValueError(
            f"{field_name} must contain 1 to {maximum} safe characters"
        )
    return text


def _optional_text(
    value: Optional[str],
    field_name: str,
    *,
    maximum: int = 512,
) -> Optional[str]:
    if value is None:
        return None
    return _nonempty_text(value, field_name, maximum=maximum)


def _finite_float(value: float, field_name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _fraction(value: float, field_name: str) -> float:
    number = _finite_float(value, field_name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return number


def _json_value(value: Any, path: str = "record") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
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
        return [_json_value(item, f"{path}[]") for item in value]
    raise ValueError(f"{path} must contain JSON-compatible values only")


def _json_object(value: Mapping[str, Any], field_name: str) -> Dict[str, Any]:
    normalized = _json_value(value, field_name)
    if not isinstance(normalized, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    # The explicit round trip guards against accidental custom scalar types.
    json.dumps(normalized, ensure_ascii=True, sort_keys=True, allow_nan=False)
    return normalized


def canonical_json(value: Mapping[str, Any]) -> str:
    """Render an attention record deterministically for replay/audit logs."""

    normalized = _json_object(value, "record")
    return json.dumps(
        normalized,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _jsonb(value: Mapping[str, Any]) -> Any:
    # Optional PostgreSQL dependencies remain lazy, like the rest of eva_db.
    from psycopg.types.json import Jsonb

    return Jsonb(_json_object(value, "record"))


@dataclass(frozen=True)
class EmbeddingSnapshotRef:
    """A sparse saved-frame reference produced at the embedding cadence."""

    id: str
    channel_id: int
    captured_at_ms: int
    embedding_ref: str
    embedding_model: str
    frame_ref: Optional[str] = None
    cadence_ms: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _uuid_text(self.id, "id"))
        object.__setattr__(self, "channel_id", _positive_channel(self.channel_id))
        object.__setattr__(
            self,
            "captured_at_ms",
            _timestamp_ms(self.captured_at_ms, "captured_at_ms"),
        )
        object.__setattr__(
            self,
            "embedding_ref",
            _nonempty_text(self.embedding_ref, "embedding_ref", maximum=1024),
        )
        object.__setattr__(
            self,
            "embedding_model",
            _nonempty_text(self.embedding_model, "embedding_model", maximum=160),
        )
        object.__setattr__(
            self,
            "frame_ref",
            _optional_text(self.frame_ref, "frame_ref", maximum=1024),
        )
        if self.cadence_ms is not None:
            cadence_ms = int(self.cadence_ms)
            if cadence_ms <= 0 or cadence_ms > 3_600_000:
                raise ValueError("cadence_ms must be between 1 and 3600000")
            object.__setattr__(self, "cadence_ms", cadence_ms)


@dataclass(frozen=True)
class ProbeScoreRecord:
    """P/N/M produced for one probe against one saved embedding snapshot."""

    id: str
    embedding_snapshot_id: str
    scored_at_ms: int
    probe_id: str
    probe_version: str
    pos_score: float
    neg_score: float
    margin: float
    threshold_state: str
    pos_floor: Optional[float] = None
    margin_threshold: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _uuid_text(self.id, "id"))
        object.__setattr__(
            self,
            "embedding_snapshot_id",
            _uuid_text(self.embedding_snapshot_id, "embedding_snapshot_id"),
        )
        object.__setattr__(
            self,
            "scored_at_ms",
            _timestamp_ms(self.scored_at_ms, "scored_at_ms"),
        )
        object.__setattr__(
            self, "probe_id", _nonempty_text(self.probe_id, "probe_id", maximum=160)
        )
        object.__setattr__(
            self,
            "probe_version",
            _nonempty_text(self.probe_version, "probe_version", maximum=160),
        )
        for field_name in ("pos_score", "neg_score", "margin"):
            object.__setattr__(
                self,
                field_name,
                _finite_float(getattr(self, field_name), field_name),
            )
        if not -1.0 <= self.pos_score <= 1.0:
            raise ValueError("pos_score must be between -1 and 1")
        if not -1.0 <= self.neg_score <= 1.0:
            raise ValueError("neg_score must be between -1 and 1")
        if not -2.0 <= self.margin <= 2.0:
            raise ValueError("margin must be between -2 and 2")
        state = str(self.threshold_state or "").strip().lower()
        if state not in PROBE_THRESHOLD_STATES:
            raise ValueError(
                "threshold_state must be one of "
                f"{sorted(PROBE_THRESHOLD_STATES)}"
            )
        object.__setattr__(self, "threshold_state", state)
        for field_name in ("pos_floor", "margin_threshold"):
            value = getattr(self, field_name)
            if value is None:
                continue
            threshold = _finite_float(value, field_name)
            if field_name == "pos_floor" and not 0.0 <= threshold <= 1.0:
                raise ValueError("pos_floor must be between 0 and 1")
            if field_name == "margin_threshold" and not 0.0 <= threshold <= 2.0:
                raise ValueError("margin_threshold must be between 0 and 2")
            object.__setattr__(self, field_name, threshold)
        if state != "not_evaluated" and (
            self.pos_floor is None or self.margin_threshold is None
        ):
            raise ValueError(
                "evaluated threshold_state requires pos_floor and margin_threshold"
            )


@dataclass(frozen=True)
class MotionInterval:
    """Numeric aggregate over dense CV samples; never contains image data."""

    id: str
    channel_id: int
    started_at_ms: int
    ended_at_ms: int
    state: str
    sample_count: int
    motion_mean: float
    motion_max: float
    motion_p95: float
    motion_integral: float
    moving_fraction: float
    quiet_fraction: float
    activity_x_max: float
    peak_at_ms: Optional[int] = None
    expected_sample_count: Optional[int] = None
    baseline_ref: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _uuid_text(self.id, "id"))
        object.__setattr__(self, "channel_id", _positive_channel(self.channel_id))
        started = _timestamp_ms(self.started_at_ms, "started_at_ms")
        ended = _timestamp_ms(self.ended_at_ms, "ended_at_ms")
        if ended < started:
            raise ValueError("ended_at_ms must not precede started_at_ms")
        object.__setattr__(self, "started_at_ms", started)
        object.__setattr__(self, "ended_at_ms", ended)
        state = str(self.state or "").strip().lower()
        if state not in INTERVAL_STATES:
            raise ValueError(f"state must be one of {sorted(INTERVAL_STATES)}")
        object.__setattr__(self, "state", state)
        samples = int(self.sample_count)
        if samples <= 0:
            raise ValueError("sample_count must be positive")
        object.__setattr__(self, "sample_count", samples)
        for field_name in (
            "motion_mean",
            "motion_max",
            "motion_p95",
            "motion_integral",
            "activity_x_max",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_float(getattr(self, field_name), field_name),
            )
        if min(self.motion_mean, self.motion_max, self.motion_p95) < 0:
            raise ValueError("motion aggregates must be non-negative")
        if self.motion_integral < 0 or self.activity_x_max < 0:
            raise ValueError("motion_integral and activity_x_max must be non-negative")
        object.__setattr__(
            self,
            "moving_fraction",
            _fraction(self.moving_fraction, "moving_fraction"),
        )
        object.__setattr__(
            self,
            "quiet_fraction",
            _fraction(self.quiet_fraction, "quiet_fraction"),
        )
        if self.moving_fraction + self.quiet_fraction > 1.000001:
            raise ValueError("moving_fraction + quiet_fraction must not exceed 1")
        if self.peak_at_ms is not None:
            peak = _timestamp_ms(self.peak_at_ms, "peak_at_ms")
            if not started <= peak <= ended:
                raise ValueError("peak_at_ms must fall inside the interval")
            object.__setattr__(self, "peak_at_ms", peak)
        if self.expected_sample_count is not None:
            expected = int(self.expected_sample_count)
            if expected <= 0:
                raise ValueError("expected_sample_count must be positive")
            object.__setattr__(self, "expected_sample_count", expected)
        object.__setattr__(
            self,
            "baseline_ref",
            _optional_text(self.baseline_ref, "baseline_ref", maximum=256),
        )


@dataclass(frozen=True)
class IntervalEvidenceLink:
    """Connect a CV interval to an embedding snapshot or external VLM apex."""

    id: str
    interval_id: str
    occurred_at_ms: int
    kind: str
    role: str
    embedding_snapshot_id: Optional[str] = None
    apex_ref: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _uuid_text(self.id, "id"))
        object.__setattr__(
            self, "interval_id", _uuid_text(self.interval_id, "interval_id")
        )
        object.__setattr__(
            self,
            "occurred_at_ms",
            _timestamp_ms(self.occurred_at_ms, "occurred_at_ms"),
        )
        kind = str(self.kind or "").strip().lower()
        if kind not in LINK_KINDS:
            raise ValueError(f"kind must be one of {sorted(LINK_KINDS)}")
        object.__setattr__(self, "kind", kind)
        role = str(self.role or "").strip().lower()
        if role not in LINK_ROLES:
            raise ValueError(f"role must be one of {sorted(LINK_ROLES)}")
        object.__setattr__(self, "role", role)
        snapshot_id = (
            _uuid_text(self.embedding_snapshot_id, "embedding_snapshot_id")
            if self.embedding_snapshot_id is not None
            else None
        )
        apex_ref = _optional_text(self.apex_ref, "apex_ref", maximum=1024)
        if kind == "embedding" and (snapshot_id is None or apex_ref is not None):
            raise ValueError(
                "embedding links require embedding_snapshot_id and no apex_ref"
            )
        if kind == "vlm_apex" and (apex_ref is None or snapshot_id is not None):
            raise ValueError("vlm_apex links require apex_ref and no snapshot id")
        object.__setattr__(self, "embedding_snapshot_id", snapshot_id)
        object.__setattr__(self, "apex_ref", apex_ref)


@dataclass(frozen=True)
class AttentionEpisodeRecord:
    id: str
    channel_id: int
    started_at_ms: int
    ended_at_ms: int
    trigger: str
    status: str
    record: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _uuid_text(self.id, "id"))
        object.__setattr__(self, "channel_id", _positive_channel(self.channel_id))
        started = _timestamp_ms(self.started_at_ms, "started_at_ms")
        ended = _timestamp_ms(self.ended_at_ms, "ended_at_ms")
        if ended < started:
            raise ValueError("ended_at_ms must not precede started_at_ms")
        object.__setattr__(self, "started_at_ms", started)
        object.__setattr__(self, "ended_at_ms", ended)
        object.__setattr__(
            self, "trigger", _nonempty_text(self.trigger, "trigger", maximum=80)
        )
        object.__setattr__(
            self, "status", _nonempty_text(self.status, "status", maximum=40)
        )
        object.__setattr__(self, "record", _json_object(self.record, "record"))

    @property
    def printable_json(self) -> str:
        return canonical_json(self.record)


@dataclass(frozen=True)
class SchedulerDecisionRecord:
    id: str
    decided_at_ms: int
    action: str
    record: Mapping[str, Any]
    channel_id: Optional[int] = None
    episode_id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _uuid_text(self.id, "id"))
        object.__setattr__(
            self,
            "decided_at_ms",
            _timestamp_ms(self.decided_at_ms, "decided_at_ms"),
        )
        object.__setattr__(
            self, "action", _nonempty_text(self.action, "action", maximum=80)
        )
        if self.channel_id is not None:
            object.__setattr__(
                self, "channel_id", _positive_channel(self.channel_id)
            )
        if self.episode_id is not None:
            object.__setattr__(
                self, "episode_id", _uuid_text(self.episode_id, "episode_id")
            )
        object.__setattr__(self, "record", _json_object(self.record, "record"))

    @property
    def printable_json(self) -> str:
        return canonical_json(self.record)


@dataclass(frozen=True)
class ProbeLineageRecord:
    id: str
    probe_id: str
    channel_id: int
    created_at_ms: int
    lifecycle_state: str
    record: Mapping[str, Any]
    parent_alert_ref: Optional[str] = None
    parent_probe_id: Optional[str] = None
    expires_at_ms: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _uuid_text(self.id, "id"))
        object.__setattr__(
            self, "probe_id", _nonempty_text(self.probe_id, "probe_id", maximum=160)
        )
        object.__setattr__(self, "channel_id", _positive_channel(self.channel_id))
        created = _timestamp_ms(self.created_at_ms, "created_at_ms")
        object.__setattr__(self, "created_at_ms", created)
        state = str(self.lifecycle_state or "").strip().lower()
        if state not in PROBE_LIFECYCLE_STATES:
            raise ValueError(
                "lifecycle_state must be one of "
                f"{sorted(PROBE_LIFECYCLE_STATES)}"
            )
        object.__setattr__(self, "lifecycle_state", state)
        object.__setattr__(
            self,
            "parent_alert_ref",
            _optional_text(
                self.parent_alert_ref, "parent_alert_ref", maximum=1024
            ),
        )
        object.__setattr__(
            self,
            "parent_probe_id",
            _optional_text(self.parent_probe_id, "parent_probe_id", maximum=160),
        )
        if self.expires_at_ms is not None:
            expires = _timestamp_ms(self.expires_at_ms, "expires_at_ms")
            if expires < created:
                raise ValueError("expires_at_ms must not precede created_at_ms")
            object.__setattr__(self, "expires_at_ms", expires)
        object.__setattr__(self, "record", _json_object(self.record, "record"))

    @property
    def printable_json(self) -> str:
        return canonical_json(self.record)


@dataclass(frozen=True)
class AttentionBatch:
    snapshots: Tuple[EmbeddingSnapshotRef, ...] = ()
    probe_scores: Tuple[ProbeScoreRecord, ...] = ()
    intervals: Tuple[MotionInterval, ...] = ()
    links: Tuple[IntervalEvidenceLink, ...] = ()
    episodes: Tuple[AttentionEpisodeRecord, ...] = ()
    decisions: Tuple[SchedulerDecisionRecord, ...] = ()
    probe_lineage: Tuple[ProbeLineageRecord, ...] = ()

    def __post_init__(self) -> None:
        for field_name, expected_type in (
            ("snapshots", EmbeddingSnapshotRef),
            ("probe_scores", ProbeScoreRecord),
            ("intervals", MotionInterval),
            ("links", IntervalEvidenceLink),
            ("episodes", AttentionEpisodeRecord),
            ("decisions", SchedulerDecisionRecord),
            ("probe_lineage", ProbeLineageRecord),
        ):
            values = tuple(getattr(self, field_name))
            if not all(isinstance(value, expected_type) for value in values):
                raise TypeError(f"{field_name} contains an invalid record")
            object.__setattr__(self, field_name, values)

    @property
    def record_count(self) -> int:
        return sum(
            len(values)
            for values in (
                self.snapshots,
                self.probe_scores,
                self.intervals,
                self.links,
                self.episodes,
                self.decisions,
                self.probe_lineage,
            )
        )

    @property
    def empty(self) -> bool:
        return self.record_count == 0

    @classmethod
    def merge(cls, batches: Sequence["AttentionBatch"]) -> "AttentionBatch":
        return cls(
            snapshots=tuple(
                item for batch in batches for item in batch.snapshots
            ),
            probe_scores=tuple(
                item for batch in batches for item in batch.probe_scores
            ),
            intervals=tuple(
                item for batch in batches for item in batch.intervals
            ),
            links=tuple(item for batch in batches for item in batch.links),
            episodes=tuple(item for batch in batches for item in batch.episodes),
            decisions=tuple(
                item for batch in batches for item in batch.decisions
            ),
            probe_lineage=tuple(
                item for batch in batches for item in batch.probe_lineage
            ),
        )


@dataclass(frozen=True)
class AttentionWriteResult:
    ok: bool
    accepted_records: int
    inserted_records: int
    error: Optional[str] = None


class AttentionBatchStore(Protocol):
    def write_batch(self, batch: AttentionBatch) -> AttentionWriteResult:
        ...


class MemoryAttentionStore:
    """Bounded process-local adapter for tests and DB-disabled deployments.

    This is intentionally an explicit fallback rather than an implicit mirror:
    callers can inspect or drain its immutable batches, while production wiring
    can keep PostgreSQL failures visible instead of pretending they were durable.
    """

    backend = "memory"

    def __init__(
        self,
        *,
        max_batches: int = 256,
        max_records: int = 8192,
    ) -> None:
        self.max_batches = max(1, int(max_batches))
        self.max_records = max(1, int(max_records))
        self._batches: Deque[AttentionBatch] = deque()
        self._record_count = 0
        self._dropped_batches = 0
        self._dropped_records = 0
        self._lock = threading.Lock()

    def write_batch(self, batch: AttentionBatch) -> AttentionWriteResult:
        if not isinstance(batch, AttentionBatch):
            raise TypeError("batch must be AttentionBatch")
        if batch.empty:
            return AttentionWriteResult(True, 0, 0)
        with self._lock:
            if batch.record_count > self.max_records:
                self._dropped_batches += 1
                self._dropped_records += batch.record_count
                return AttentionWriteResult(
                    ok=False,
                    accepted_records=batch.record_count,
                    inserted_records=0,
                    error="batch_too_large",
                )
            while self._batches and (
                len(self._batches) >= self.max_batches
                or self._record_count + batch.record_count > self.max_records
            ):
                dropped = self._batches.popleft()
                self._record_count -= dropped.record_count
                self._dropped_batches += 1
                self._dropped_records += dropped.record_count
            self._batches.append(batch)
            self._record_count += batch.record_count
        return AttentionWriteResult(
            ok=True,
            accepted_records=batch.record_count,
            inserted_records=batch.record_count,
        )

    def drain_batches(self) -> Tuple[AttentionBatch, ...]:
        """Atomically return and clear retained batches for replay/export."""

        with self._lock:
            batches = tuple(self._batches)
            self._batches.clear()
            self._record_count = 0
            return batches

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "queued_batches": len(self._batches),
                "queued_records": self._record_count,
                "dropped_batches": self._dropped_batches,
                "dropped_records": self._dropped_records,
            }


class PostgresAttentionStore:
    """Tenant-isolated PostgreSQL repository for attention telemetry."""

    backend = "postgres"

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
        self._context = TransactionContext(
            tenant_id=self.tenant_id,
            actor_id=self.actor_id,
        )

    def health(self) -> Dict[str, Any]:
        try:
            with self.pool.transaction(self._context, readonly=True) as connection:
                connection.execute(
                    "SELECT 1 FROM archive.attention_intervals LIMIT 1"
                )
        except Exception as exc:
            return {
                "ok": False,
                "backend": self.backend,
                "status": (
                    "not_migrated"
                    if _is_missing_attention_relation(exc)
                    else "unavailable"
                ),
                "required_revision": ATTENTION_STORAGE_REVISION,
                "error": type(exc).__name__,
            }
        return {
            "ok": True,
            "backend": self.backend,
            "status": "reachable",
            "tenant_id": self.tenant_id,
        }

    def write_batch(self, batch: AttentionBatch) -> AttentionWriteResult:
        if not isinstance(batch, AttentionBatch):
            raise TypeError("batch must be AttentionBatch")
        if batch.empty:
            return AttentionWriteResult(True, 0, 0)
        inserted = 0
        try:
            with self.pool.transaction(self._context) as connection:
                for item in batch.snapshots:
                    inserted += _rowcount(
                        connection.execute(
                            """
                            INSERT INTO archive.attention_embedding_snapshots (
                                tenant_id, id, channel_id, captured_at_ms,
                                embedding_ref, embedding_model, frame_ref, cadence_ms
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT DO NOTHING
                            """,
                            (
                                self.tenant_id,
                                item.id,
                                item.channel_id,
                                item.captured_at_ms,
                                item.embedding_ref,
                                item.embedding_model,
                                item.frame_ref,
                                item.cadence_ms,
                            ),
                        )
                    )
                for item in batch.probe_scores:
                    inserted += _rowcount(
                        connection.execute(
                            """
                            INSERT INTO archive.attention_probe_scores (
                                tenant_id, id, embedding_snapshot_id,
                                scored_at_ms, probe_id, probe_version,
                                pos_score, neg_score, margin, pos_floor,
                                margin_threshold, threshold_state
                            )
                            VALUES (
                                %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s
                            )
                            ON CONFLICT DO NOTHING
                            """,
                            (
                                self.tenant_id,
                                item.id,
                                item.embedding_snapshot_id,
                                item.scored_at_ms,
                                item.probe_id,
                                item.probe_version,
                                item.pos_score,
                                item.neg_score,
                                item.margin,
                                item.pos_floor,
                                item.margin_threshold,
                                item.threshold_state,
                            ),
                        )
                    )
                for item in batch.intervals:
                    inserted += _rowcount(
                        connection.execute(
                            """
                            INSERT INTO archive.attention_intervals (
                                tenant_id, id, channel_id, started_at_ms,
                                ended_at_ms, state, sample_count,
                                expected_sample_count, motion_mean, motion_max,
                                motion_p95, motion_integral, moving_fraction,
                                quiet_fraction, activity_x_max, peak_at_ms,
                                baseline_ref
                            )
                            VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s, %s
                            )
                            ON CONFLICT (tenant_id, id) DO NOTHING
                            """,
                            (
                                self.tenant_id,
                                item.id,
                                item.channel_id,
                                item.started_at_ms,
                                item.ended_at_ms,
                                item.state,
                                item.sample_count,
                                item.expected_sample_count,
                                item.motion_mean,
                                item.motion_max,
                                item.motion_p95,
                                item.motion_integral,
                                item.moving_fraction,
                                item.quiet_fraction,
                                item.activity_x_max,
                                item.peak_at_ms,
                                item.baseline_ref,
                            ),
                        )
                    )
                for item in batch.links:
                    inserted += _rowcount(
                        connection.execute(
                            """
                            INSERT INTO archive.attention_interval_links (
                                tenant_id, id, interval_id, occurred_at_ms,
                                kind, role, embedding_snapshot_id, apex_ref
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (tenant_id, id) DO NOTHING
                            """,
                            (
                                self.tenant_id,
                                item.id,
                                item.interval_id,
                                item.occurred_at_ms,
                                item.kind,
                                item.role,
                                item.embedding_snapshot_id,
                                item.apex_ref,
                            ),
                        )
                    )
                for item in batch.episodes:
                    inserted += _rowcount(
                        connection.execute(
                            """
                            INSERT INTO archive.attention_episodes (
                                tenant_id, id, channel_id, started_at_ms,
                                ended_at_ms, trigger, status, record_json,
                                canonical_json
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (tenant_id, id) DO NOTHING
                            """,
                            (
                                self.tenant_id,
                                item.id,
                                item.channel_id,
                                item.started_at_ms,
                                item.ended_at_ms,
                                item.trigger,
                                item.status,
                                _jsonb(item.record),
                                item.printable_json,
                            ),
                        )
                    )
                for item in batch.probe_lineage:
                    inserted += _rowcount(
                        connection.execute(
                            """
                            INSERT INTO archive.attention_probe_lineage (
                                tenant_id, id, probe_id, channel_id,
                                created_at_ms, expires_at_ms, lifecycle_state,
                                parent_alert_ref, parent_probe_id, record_json,
                                canonical_json
                            )
                            VALUES (
                                %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s
                            )
                            ON CONFLICT (tenant_id, id) DO NOTHING
                            """,
                            (
                                self.tenant_id,
                                item.id,
                                item.probe_id,
                                item.channel_id,
                                item.created_at_ms,
                                item.expires_at_ms,
                                item.lifecycle_state,
                                item.parent_alert_ref,
                                item.parent_probe_id,
                                _jsonb(item.record),
                                item.printable_json,
                            ),
                        )
                    )
                for item in batch.decisions:
                    inserted += _rowcount(
                        connection.execute(
                            """
                            INSERT INTO archive.attention_scheduler_decisions (
                                tenant_id, id, channel_id, episode_id,
                                decided_at_ms, action, record_json,
                                canonical_json
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (tenant_id, id) DO NOTHING
                            """,
                            (
                                self.tenant_id,
                                item.id,
                                item.channel_id,
                                item.episode_id,
                                item.decided_at_ms,
                                item.action,
                                _jsonb(item.record),
                                item.printable_json,
                            ),
                        )
                    )
        except Exception as exc:
            if _is_missing_attention_relation(exc):
                error = "not_migrated"
            else:
                error = type(exc).__name__
            return AttentionWriteResult(
                ok=False,
                accepted_records=batch.record_count,
                inserted_records=0,
                error=error,
            )
        return AttentionWriteResult(
            ok=True,
            accepted_records=batch.record_count,
            inserted_records=inserted,
        )

    def query_intervals(
        self,
        *,
        channel_id: int,
        start_ms: int,
        end_ms: int,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        channel, started, ended, bounded = _query_bounds(
            channel_id, start_ms, end_ms, limit
        )
        with self.pool.transaction(self._context, readonly=True) as connection:
            rows = connection.execute(
                """
                SELECT
                    id::text, channel_id, started_at_ms, ended_at_ms, state,
                    sample_count, expected_sample_count, motion_mean,
                    motion_max, motion_p95, motion_integral, moving_fraction,
                    quiet_fraction, activity_x_max, peak_at_ms, baseline_ref
                FROM archive.attention_intervals
                WHERE tenant_id = %s
                  AND channel_id = %s
                  AND ended_at_ms >= %s
                  AND started_at_ms <= %s
                ORDER BY started_at_ms ASC, id ASC
                LIMIT %s
                """,
                (self.tenant_id, channel, started, ended, bounded),
            ).fetchall()
        keys = (
            "id",
            "channel_id",
            "started_at_ms",
            "ended_at_ms",
            "state",
            "sample_count",
            "expected_sample_count",
            "motion_mean",
            "motion_max",
            "motion_p95",
            "motion_integral",
            "moving_fraction",
            "quiet_fraction",
            "activity_x_max",
            "peak_at_ms",
            "baseline_ref",
        )
        return [dict(zip(keys, row)) for row in rows]

    def query_snapshots(
        self,
        *,
        channel_id: int,
        start_ms: int,
        end_ms: int,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        channel, started, ended, bounded = _query_bounds(
            channel_id, start_ms, end_ms, limit
        )
        with self.pool.transaction(self._context, readonly=True) as connection:
            rows = connection.execute(
                """
                SELECT
                    id::text, channel_id, captured_at_ms, embedding_ref,
                    embedding_model, frame_ref, cadence_ms
                FROM archive.attention_embedding_snapshots
                WHERE tenant_id = %s
                  AND channel_id = %s
                  AND captured_at_ms BETWEEN %s AND %s
                ORDER BY captured_at_ms ASC, id ASC
                LIMIT %s
                """,
                (self.tenant_id, channel, started, ended, bounded),
            ).fetchall()
        keys = (
            "id",
            "channel_id",
            "captured_at_ms",
            "embedding_ref",
            "embedding_model",
            "frame_ref",
            "cadence_ms",
        )
        return [dict(zip(keys, row)) for row in rows]

    def query_evidence_links(
        self,
        *,
        channel_id: int,
        start_ms: int,
        end_ms: int,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        channel, started, ended, bounded = _query_bounds(
            channel_id, start_ms, end_ms, limit
        )
        with self.pool.transaction(self._context, readonly=True) as connection:
            rows = connection.execute(
                """
                SELECT
                    link.id::text, link.interval_id::text,
                    link.occurred_at_ms, link.kind, link.role,
                    link.embedding_snapshot_id::text,
                    snapshot.embedding_ref, link.apex_ref
                FROM archive.attention_interval_links AS link
                JOIN archive.attention_intervals AS interval
                  ON interval.tenant_id = link.tenant_id
                 AND interval.id = link.interval_id
                LEFT JOIN archive.attention_embedding_snapshots AS snapshot
                  ON snapshot.tenant_id = link.tenant_id
                 AND snapshot.id = link.embedding_snapshot_id
                WHERE link.tenant_id = %s
                  AND interval.channel_id = %s
                  AND link.occurred_at_ms BETWEEN %s AND %s
                ORDER BY link.occurred_at_ms ASC, link.id ASC
                LIMIT %s
                """,
                (self.tenant_id, channel, started, ended, bounded),
            ).fetchall()
        keys = (
            "id",
            "interval_id",
            "occurred_at_ms",
            "kind",
            "role",
            "embedding_snapshot_id",
            "embedding_ref",
            "apex_ref",
        )
        return [dict(zip(keys, row)) for row in rows]

    def query_probe_scores(
        self,
        *,
        channel_id: int,
        start_ms: int,
        end_ms: int,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        channel, started, ended, bounded = _query_bounds(
            channel_id, start_ms, end_ms, limit
        )
        with self.pool.transaction(self._context, readonly=True) as connection:
            rows = connection.execute(
                """
                SELECT
                    score.id::text, score.embedding_snapshot_id::text,
                    snapshot.captured_at_ms, score.scored_at_ms,
                    score.probe_id, score.probe_version,
                    score.pos_score, score.neg_score, score.margin,
                    score.pos_floor, score.margin_threshold,
                    score.threshold_state
                FROM archive.attention_probe_scores AS score
                JOIN archive.attention_embedding_snapshots AS snapshot
                  ON snapshot.tenant_id = score.tenant_id
                 AND snapshot.id = score.embedding_snapshot_id
                WHERE score.tenant_id = %s
                  AND snapshot.channel_id = %s
                  AND snapshot.captured_at_ms BETWEEN %s AND %s
                ORDER BY
                    snapshot.captured_at_ms ASC,
                    score.probe_id ASC,
                    score.id ASC
                LIMIT %s
                """,
                (self.tenant_id, channel, started, ended, bounded),
            ).fetchall()
        keys = (
            "id",
            "embedding_snapshot_id",
            "captured_at_ms",
            "scored_at_ms",
            "probe_id",
            "probe_version",
            "pos_score",
            "neg_score",
            "margin",
            "pos_floor",
            "margin_threshold",
            "threshold_state",
        )
        return [dict(zip(keys, row)) for row in rows]

    def query_episodes(
        self,
        *,
        channel_id: int,
        start_ms: int,
        end_ms: int,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        channel, started, ended, bounded = _query_bounds(
            channel_id, start_ms, end_ms, limit
        )
        with self.pool.transaction(self._context, readonly=True) as connection:
            rows = connection.execute(
                """
                SELECT
                    id::text, channel_id, started_at_ms, ended_at_ms,
                    trigger, status, record_json, canonical_json
                FROM archive.attention_episodes
                WHERE tenant_id = %s
                  AND channel_id = %s
                  AND ended_at_ms >= %s
                  AND started_at_ms <= %s
                ORDER BY started_at_ms ASC, id ASC
                LIMIT %s
                """,
                (self.tenant_id, channel, started, ended, bounded),
            ).fetchall()
        return [
            {
                "id": str(row[0]),
                "channel_id": int(row[1]),
                "started_at_ms": int(row[2]),
                "ended_at_ms": int(row[3]),
                "trigger": str(row[4]),
                "status": str(row[5]),
                "record": _decode_json(row[6]),
                "canonical_json": str(row[7]),
            }
            for row in rows
        ]

    def query_decisions(
        self,
        *,
        channel_id: int,
        start_ms: int,
        end_ms: int,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        channel, started, ended, bounded = _query_bounds(
            channel_id, start_ms, end_ms, limit
        )
        with self.pool.transaction(self._context, readonly=True) as connection:
            rows = connection.execute(
                """
                SELECT
                    id::text, channel_id, episode_id::text, decided_at_ms,
                    action, record_json, canonical_json
                FROM archive.attention_scheduler_decisions
                WHERE tenant_id = %s
                  AND channel_id = %s
                  AND decided_at_ms BETWEEN %s AND %s
                ORDER BY decided_at_ms ASC, id ASC
                LIMIT %s
                """,
                (self.tenant_id, channel, started, ended, bounded),
            ).fetchall()
        return [
            {
                "id": str(row[0]),
                "channel_id": int(row[1]) if row[1] is not None else None,
                "episode_id": str(row[2]) if row[2] is not None else None,
                "decided_at_ms": int(row[3]),
                "action": str(row[4]),
                "record": _decode_json(row[5]),
                "canonical_json": str(row[6]),
            }
            for row in rows
        ]

    def query_probe_lineage(
        self,
        *,
        channel_id: int,
        start_ms: int,
        end_ms: int,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        channel, started, ended, bounded = _query_bounds(
            channel_id, start_ms, end_ms, limit
        )
        with self.pool.transaction(self._context, readonly=True) as connection:
            rows = connection.execute(
                """
                SELECT
                    id::text, probe_id, channel_id, created_at_ms,
                    expires_at_ms, lifecycle_state, parent_alert_ref,
                    parent_probe_id, record_json, canonical_json
                FROM archive.attention_probe_lineage
                WHERE tenant_id = %s
                  AND channel_id = %s
                  AND created_at_ms BETWEEN %s AND %s
                ORDER BY created_at_ms ASC, id ASC
                LIMIT %s
                """,
                (self.tenant_id, channel, started, ended, bounded),
            ).fetchall()
        return [
            {
                "id": str(row[0]),
                "probe_id": str(row[1]),
                "channel_id": int(row[2]),
                "created_at_ms": int(row[3]),
                "expires_at_ms": (
                    int(row[4]) if row[4] is not None else None
                ),
                "lifecycle_state": str(row[5]),
                "parent_alert_ref": row[6],
                "parent_probe_id": row[7],
                "record": _decode_json(row[8]),
                "canonical_json": str(row[9]),
            }
            for row in rows
        ]

    def apply_retention(
        self,
        *,
        before_ms: int,
        batch_size: int = 5000,
    ) -> Dict[str, int]:
        cutoff = _timestamp_ms(before_ms, "before_ms")
        bounded = max(1, min(50_000, int(batch_size)))
        specs = (
            ("attention_scheduler_decisions", "decided_at_ms"),
            ("attention_probe_lineage", "created_at_ms"),
            ("attention_episodes", "ended_at_ms"),
            ("attention_intervals", "ended_at_ms"),
            ("attention_embedding_snapshots", "captured_at_ms"),
        )
        deleted: Dict[str, int] = {}
        with self.pool.transaction(self._context) as connection:
            for table, timestamp_column in specs:
                cursor = connection.execute(
                    f"""
                    DELETE FROM archive.{table}
                    WHERE tenant_id = %s
                      AND id IN (
                          SELECT id
                          FROM archive.{table}
                          WHERE tenant_id = %s
                            AND {timestamp_column} < %s
                          ORDER BY {timestamp_column} ASC, id ASC
                          LIMIT %s
                      )
                    """,
                    (self.tenant_id, self.tenant_id, cutoff, bounded),
                )
                deleted[table] = _rowcount(cursor)
        return deleted


@dataclass(frozen=True)
class BufferSubmitResult:
    accepted: bool
    queued_batches: int
    queued_records: int
    dropped_batches: int
    dropped_records: int
    reason: Optional[str] = None


class BufferedAttentionWriter:
    """Bounded background writer that never performs DB I/O in ``submit``."""

    def __init__(
        self,
        store: AttentionBatchStore,
        *,
        max_batches: int = 256,
        max_records: int = 8192,
        write_batch_records: int = 512,
        retry_initial_seconds: float = 0.25,
        retry_max_seconds: float = 10.0,
        autostart: bool = True,
        thread_name: str = "eva-attention-writer",
    ) -> None:
        self.store = store
        self.max_batches = max(1, int(max_batches))
        self.max_records = max(1, int(max_records))
        self.write_batch_records = max(1, int(write_batch_records))
        self.retry_initial_seconds = max(0.01, float(retry_initial_seconds))
        self.retry_max_seconds = max(
            self.retry_initial_seconds, float(retry_max_seconds)
        )
        self._queue: Deque[AttentionBatch] = deque()
        self._queued_records = 0
        self._dropped_batches = 0
        self._dropped_records = 0
        self._written_batches = 0
        self._written_records = 0
        self._write_failures = 0
        self._last_error: Optional[str] = None
        self._inflight = False
        self._accepting = True
        self._stopping = False
        self._condition = threading.Condition()
        self._thread = threading.Thread(
            target=self._run,
            name=thread_name,
            daemon=True,
        )
        if autostart:
            self.start()

    def start(self) -> None:
        with self._condition:
            if self._thread.is_alive():
                return
            if self._stopping:
                raise RuntimeError("writer cannot be restarted after close")
            self._thread.start()

    def submit(self, batch: AttentionBatch) -> BufferSubmitResult:
        if not isinstance(batch, AttentionBatch):
            raise TypeError("batch must be AttentionBatch")
        with self._condition:
            if not self._accepting:
                return self._submit_result(False, "closed")
            if batch.empty:
                return self._submit_result(True, "empty")
            if batch.record_count > self.max_records:
                self._dropped_batches += 1
                self._dropped_records += batch.record_count
                return self._submit_result(False, "batch_too_large")
            while self._queue and (
                len(self._queue) >= self.max_batches
                or self._queued_records + batch.record_count > self.max_records
            ):
                dropped = self._queue.popleft()
                self._queued_records -= dropped.record_count
                self._dropped_batches += 1
                self._dropped_records += dropped.record_count
            self._queue.append(batch)
            self._queued_records += batch.record_count
            self._condition.notify()
            return self._submit_result(True, None)

    def stats(self) -> Dict[str, Any]:
        with self._condition:
            return {
                "queued_batches": len(self._queue),
                "queued_records": self._queued_records,
                "inflight": self._inflight,
                "written_batches": self._written_batches,
                "written_records": self._written_records,
                "write_failures": self._write_failures,
                "dropped_batches": self._dropped_batches,
                "dropped_records": self._dropped_records,
                "last_error": self._last_error,
                "accepting": self._accepting,
            }

    def drain(self, timeout_seconds: float = 5.0) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        with self._condition:
            while self._queue or self._inflight:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            return True

    def close(self, *, flush_timeout_seconds: float = 2.0) -> bool:
        with self._condition:
            self._accepting = False
        drained = self.drain(flush_timeout_seconds)
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=max(0.1, flush_timeout_seconds))
        return drained

    def flush_once(self) -> AttentionWriteResult:
        """Synchronously flush one coalesced write, primarily for service hooks/tests."""

        with self._condition:
            batch = self._take_batch_locked()
            if batch is None:
                return AttentionWriteResult(True, 0, 0)
            self._inflight = True
        result = self._write(batch)
        with self._condition:
            self._inflight = False
            self._finish_write_locked(batch, result)
            self._condition.notify_all()
        return result

    def _run(self) -> None:
        retry_delay = self.retry_initial_seconds
        while True:
            with self._condition:
                while not self._queue and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                batch = self._take_batch_locked()
                if batch is None:
                    continue
                self._inflight = True
            result = self._write(batch)
            with self._condition:
                self._inflight = False
                self._finish_write_locked(batch, result)
                self._condition.notify_all()
            if result.ok:
                retry_delay = self.retry_initial_seconds
                continue
            # Only the background worker sleeps, never the capture caller.
            retry_deadline = time.monotonic() + retry_delay
            with self._condition:
                while not self._stopping:
                    remaining = retry_deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(timeout=remaining)
                if self._stopping:
                    return
            retry_delay = min(self.retry_max_seconds, retry_delay * 2.0)

    def _take_batch_locked(self) -> Optional[AttentionBatch]:
        if not self._queue:
            return None
        batches: List[AttentionBatch] = []
        count = 0
        while self._queue:
            next_batch = self._queue[0]
            if batches and count + next_batch.record_count > self.write_batch_records:
                break
            self._queue.popleft()
            self._queued_records -= next_batch.record_count
            batches.append(next_batch)
            count += next_batch.record_count
            if count >= self.write_batch_records:
                break
        return AttentionBatch.merge(batches)

    def _write(self, batch: AttentionBatch) -> AttentionWriteResult:
        try:
            result = self.store.write_batch(batch)
        except Exception as exc:
            return AttentionWriteResult(
                ok=False,
                accepted_records=batch.record_count,
                inserted_records=0,
                error=type(exc).__name__,
            )
        if not isinstance(result, AttentionWriteResult):
            return AttentionWriteResult(
                ok=False,
                accepted_records=batch.record_count,
                inserted_records=0,
                error="invalid_store_result",
            )
        return result

    def _finish_write_locked(
        self,
        batch: AttentionBatch,
        result: AttentionWriteResult,
    ) -> None:
        if result.ok:
            self._written_batches += 1
            self._written_records += result.inserted_records
            self._last_error = None
            return
        self._write_failures += 1
        self._last_error = result.error or "write_failed"
        # Retry when capacity permits. If producers filled the bounded queue while
        # the DB was down, prefer recent telemetry and account for the old loss.
        if (
            len(self._queue) < self.max_batches
            and self._queued_records + batch.record_count <= self.max_records
        ):
            self._queue.appendleft(batch)
            self._queued_records += batch.record_count
        else:
            self._dropped_batches += 1
            self._dropped_records += batch.record_count

    def _submit_result(
        self, accepted: bool, reason: Optional[str]
    ) -> BufferSubmitResult:
        return BufferSubmitResult(
            accepted=accepted,
            queued_batches=len(self._queue),
            queued_records=self._queued_records,
            dropped_batches=self._dropped_batches,
            dropped_records=self._dropped_records,
            reason=reason,
        )


def _query_bounds(
    channel_id: int,
    start_ms: int,
    end_ms: int,
    limit: int,
) -> Tuple[int, int, int, int]:
    channel = _positive_channel(channel_id)
    started = _timestamp_ms(start_ms, "start_ms")
    ended = _timestamp_ms(end_ms, "end_ms")
    if ended < started:
        raise ValueError("end_ms must not precede start_ms")
    bounded = max(1, min(1000, int(limit)))
    return channel, started, ended, bounded


def _rowcount(cursor: Any) -> int:
    return max(0, int(getattr(cursor, "rowcount", 0) or 0))


def _decode_json(value: Any) -> Dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return dict(decoded) if isinstance(decoded, Mapping) else {}
    return {}


def _is_missing_attention_relation(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    missing = (
        "undefinedtable" in name
        or "undefined_table" in text
        or "does not exist" in text
    )
    return missing and "archive.attention_" in text


__all__ = [
    "ATTENTION_STORAGE_REVISION",
    "AttentionBatch",
    "AttentionBatchStore",
    "AttentionEpisodeRecord",
    "AttentionStoreNotReady",
    "AttentionWriteResult",
    "BufferSubmitResult",
    "BufferedAttentionWriter",
    "EmbeddingSnapshotRef",
    "IntervalEvidenceLink",
    "MemoryAttentionStore",
    "MotionInterval",
    "PostgresAttentionStore",
    "ProbeLineageRecord",
    "ProbeScoreRecord",
    "SchedulerDecisionRecord",
    "canonical_json",
]
