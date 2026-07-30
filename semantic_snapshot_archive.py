"""Durable asynchronous archive for already-computed semantic snapshots.

The writer is intentionally downstream of CLIP.  It accepts the normalized
embedding and thumbnail produced by ``ProbeManager.add_frame`` and never
re-opens an image or invokes an embedder.
"""

from __future__ import annotations

import json
import math
import queue
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence, Tuple


class DetectionsStoreLike(Protocol):
    def add_detections(self, records: Sequence[Dict[str, Any]]) -> int: ...


class SnapshotSubmitStatus(str, Enum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"
    DROPPED = "dropped"


@dataclass(frozen=True, slots=True)
class SnapshotSubmitResult:
    status: SnapshotSubmitStatus
    dedupe_key: Optional[str]
    reason: str

    @property
    def accepted(self) -> bool:
        return self.status is SnapshotSubmitStatus.ACCEPTED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "status": self.status.value,
            "dedupe_key": self.dedupe_key,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class _SnapshotEnvelope:
    dedupe_key: str
    channel_id: int
    timestamp_ms: int
    cadence_slot: int
    record: Dict[str, Any]
    queued_at: float


class _ShortWriteError(RuntimeError):
    pass


def _plain_json_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        raise ValueError("provenance nesting is too deep")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("provenance contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _plain_json_value(item, depth=depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _plain_json_value(item, depth=depth + 1)
            for item in value
        ]
    scalar = getattr(value, "item", None)
    if callable(scalar):
        converted = scalar()
        if converted is not value:
            return _plain_json_value(converted, depth=depth + 1)
    raise ValueError(f"provenance value {type(value).__name__} is not JSON-safe")


class SemanticSnapshotArchiveWriter:
    """Bounded batch writer for one semantic snapshot per cadence slot/channel."""

    SOURCE = "semantic_snapshot"
    FORMAT_VERSION = 1
    _COUNTER_KEYS = (
        "accepted_total",
        "duplicate_total",
        "rejected_total",
        "dropped_total",
        "write_attempt_total",
        "retry_total",
        "short_write_total",
        "store_reported_inserted_total",
        "batch_persisted_total",
        "persisted_total",
        "batch_failed_total",
        "failure_total",
        "gap_total",
        "wall_cadence_gap_total",
        "source_cadence_gap_total",
    )

    def __init__(
        self,
        detections_store: DetectionsStoreLike,
        *,
        cadence_ms: int = 1000,
        max_queue: int = 512,
        batch_size: int = 32,
        flush_interval_seconds: float = 0.2,
        max_attempts: int = 3,
        initial_backoff_seconds: float = 0.05,
        max_backoff_seconds: float = 1.0,
        embedding_norm_tolerance: float = 0.02,
        dedupe_capacity: int = 100_000,
        max_thumbnail_chars: int = 8_000_000,
        max_provenance_chars: int = 32_000,
        embedding_space_fn: Optional[Callable[[], Mapping[str, Any]]] = None,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        autostart: bool = True,
    ) -> None:
        if cadence_ms < 1:
            raise ValueError("cadence_ms must be positive")
        if max_queue < 1:
            raise ValueError("max_queue must be positive")
        if batch_size < 1 or batch_size > max_queue:
            raise ValueError("batch_size must be between 1 and max_queue")
        if flush_interval_seconds <= 0:
            raise ValueError("flush_interval_seconds must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if initial_backoff_seconds < 0:
            raise ValueError("initial_backoff_seconds cannot be negative")
        if max_backoff_seconds < initial_backoff_seconds:
            raise ValueError("max_backoff_seconds cannot be less than initial backoff")
        if not 0 < embedding_norm_tolerance < 1:
            raise ValueError("embedding_norm_tolerance must be between 0 and 1")
        if dedupe_capacity < max_queue:
            raise ValueError("dedupe_capacity must be at least max_queue")

        self.detections_store = detections_store
        self.cadence_ms = int(cadence_ms)
        self.max_queue = int(max_queue)
        self.batch_size = int(batch_size)
        self.flush_interval_seconds = float(flush_interval_seconds)
        self.max_attempts = int(max_attempts)
        self.initial_backoff_seconds = float(initial_backoff_seconds)
        self.max_backoff_seconds = float(max_backoff_seconds)
        self.embedding_norm_tolerance = float(embedding_norm_tolerance)
        self.dedupe_capacity = int(dedupe_capacity)
        self.max_thumbnail_chars = max(1, int(max_thumbnail_chars))
        self.max_provenance_chars = max(1, int(max_provenance_chars))
        self.embedding_space_fn = embedding_space_fn
        self._clock = clock
        self._monotonic = monotonic

        self._queue: queue.Queue[_SnapshotEnvelope] = queue.Queue(
            maxsize=self.max_queue
        )
        self._condition = threading.Condition(threading.RLock())
        self._known: OrderedDict[str, str] = OrderedDict()
        self._counters: Counter[str] = Counter()
        self._gap_reasons: Counter[str] = Counter()
        self._accepted_by_channel: Counter[int] = Counter()
        self._persisted_by_channel: Counter[int] = Counter()
        self._failed_by_channel: Counter[int] = Counter()
        self._first_accepted_wall_by_channel: Dict[int, float] = {}
        self._last_accepted_wall_by_channel: Dict[int, float] = {}
        self._last_source_slot_by_channel: Dict[int, int] = {}
        self._last_source_timestamp_by_channel: Dict[int, int] = {}
        self._wall_gaps_by_channel: Counter[int] = Counter()
        self._source_gaps_by_channel: Counter[int] = Counter()
        self._pending_count = 0
        self._in_flight = 0
        self._accepting = True
        self._started = False
        self._stopped = False
        self._last_error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        if autostart:
            self.start()

    @staticmethod
    def _positive_channel_id(value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError("channel_id must be a positive integer")
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("channel_id must be a positive integer") from exc
        if result <= 0 or str(result) != str(value).strip():
            raise ValueError("channel_id must be a positive integer")
        return result

    @staticmethod
    def _positive_timestamp_ms(value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError("timestamp_ms must be a positive integer")
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("timestamp_ms must be a positive integer") from exc
        if result <= 0 or str(result) != str(value).strip():
            raise ValueError("timestamp_ms must be a positive integer")
        return result

    def _normalized_embedding(self, value: Any) -> Tuple[float, ...]:
        raw = value.tolist() if callable(getattr(value, "tolist", None)) else value
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise ValueError("embedding must be a one-dimensional sequence")
        if not raw or len(raw) > 16_384:
            raise ValueError("embedding dimension is invalid")
        vector = []
        for item in raw:
            if isinstance(item, (list, tuple)):
                raise ValueError("embedding must be one-dimensional")
            try:
                number = float(item)
            except (TypeError, ValueError) as exc:
                raise ValueError("embedding entries must be numeric") from exc
            if not math.isfinite(number):
                raise ValueError("embedding contains a non-finite number")
            vector.append(number)
        norm = math.sqrt(math.fsum(number * number for number in vector))
        if abs(norm - 1.0) > self.embedding_norm_tolerance:
            raise ValueError(
                f"embedding must already be L2-normalized (norm={norm:.6f})"
            )
        return tuple(vector)

    def _provenance(self, value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("provenance must be a mapping")
        result = _plain_json_value(value)
        encoded = json.dumps(
            result,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        if len(encoded) > self.max_provenance_chars:
            raise ValueError("provenance is too large")
        return result

    def _dedupe_key(self, channel_id: int, cadence_slot: int) -> str:
        return (
            f"semantic_snapshot:v{self.FORMAT_VERSION}:"
            f"ch{channel_id}:cadence{self.cadence_ms}:slot{cadence_slot}"
        )

    def _record_gap_locked(self, reason: str, count: int = 1) -> None:
        normalized = str(reason or "unknown").strip()[:120] or "unknown"
        self._counters["gap_total"] += max(0, int(count))
        self._gap_reasons[normalized] += max(0, int(count))

    def _reject(
        self,
        *,
        status: SnapshotSubmitStatus,
        reason: str,
        dedupe_key: Optional[str] = None,
        gap: bool = True,
    ) -> SnapshotSubmitResult:
        with self._condition:
            self._counters["rejected_total"] += 1
            if status is SnapshotSubmitStatus.DROPPED:
                self._counters["dropped_total"] += 1
            if gap:
                self._record_gap_locked(reason)
        return SnapshotSubmitResult(status, dedupe_key, reason)

    def start(self) -> None:
        with self._condition:
            if self._started:
                return
            if self._stopped or self._stop_event.is_set():
                raise RuntimeError("semantic snapshot writer cannot be restarted")
            self._thread = threading.Thread(
                target=self._run,
                name="eva-semantic-snapshot-archive",
                daemon=True,
            )
            self._started = True
            self._thread.start()

    def submit(
        self,
        *,
        channel_id: int,
        timestamp_ms: int,
        embedding: Any,
        thumbnail: str,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> SnapshotSubmitResult:
        """Queue an already-computed embedding without blocking on the database."""

        try:
            channel = self._positive_channel_id(channel_id)
            timestamp = self._positive_timestamp_ms(timestamp_ms)
            vector = self._normalized_embedding(embedding)
            if not isinstance(thumbnail, str) or not thumbnail.strip():
                raise ValueError("thumbnail must be a non-empty base64 string")
            thumbnail_value = thumbnail.strip()
            if len(thumbnail_value) > self.max_thumbnail_chars:
                raise ValueError("thumbnail is too large")
            provenance_value = self._provenance(provenance)
        except ValueError as exc:
            return self._reject(
                status=SnapshotSubmitStatus.REJECTED,
                reason=f"validation:{exc}",
            )

        cadence_slot = timestamp // self.cadence_ms
        dedupe_key = self._dedupe_key(channel, cadence_slot)
        recorded_at_ms = int(float(self._clock()) * 1000.0)
        # Continuous embeddings use compact hourly shards.  At the eight-channel
        # port profile this keeps an exact operator search bounded to roughly
        # 3,600 vectors per channel/shard instead of loading a whole day at once.
        shard_hour = time.strftime(
            "%Y%m%d%H",
            time.localtime(float(timestamp) / 1000.0),
        )
        embedding_space: Dict[str, Any] = {}
        if callable(self.embedding_space_fn):
            try:
                raw_embedding_space = self.embedding_space_fn()
                if isinstance(raw_embedding_space, Mapping):
                    embedding_space = {
                        str(key): _plain_json_value(value)
                        for key, value in raw_embedding_space.items()
                        if str(key) in {"backend", "model", "dimension"}
                        and value is not None
                    }
            except Exception:
                # The vector remains searchable. Calibration treats missing
                # identity conservatively for non-legacy embedding spaces.
                embedding_space = {}
        embedding_space.setdefault("dimension", int(len(vector)))
        record = {
            "dedupe_key": dedupe_key,
            "timestamp_ms": timestamp,
            "recorded_at_ms": recorded_at_ms,
            "probe_id": f"semantic-snapshot:ch{channel}",
            "probe_name": "Semantic snapshot",
            "channel_id": channel,
            "severity": "info",
            "bookmark_enabled": False,
            "bookmark_sent": False,
            "pos_score": 0.0,
            "neg_score": 0.0,
            "margin": 0.0,
            "thumbnail_b64": thumbnail_value,
            "source": self.SOURCE,
            "shard_key": f"semantic:ch{channel}:{shard_hour}",
            "clip_vec": vector,
            "payload": {
                "kind": "semantic_snapshot_v1",
                "version": self.FORMAT_VERSION,
                "cadence_ms": self.cadence_ms,
                "cadence_slot": cadence_slot,
                "independent_of_alert_or_probe_hit": True,
                "selection_policy": "one_embedding_per_channel_cadence_slot",
                "embedding_space": embedding_space,
                "provenance": provenance_value,
            },
        }
        envelope = _SnapshotEnvelope(
            dedupe_key=dedupe_key,
            channel_id=channel,
            timestamp_ms=timestamp,
            cadence_slot=cadence_slot,
            record=record,
            queued_at=self._monotonic(),
        )

        with self._condition:
            if not self._accepting or self._stopped:
                return self._reject(
                    status=SnapshotSubmitStatus.DROPPED,
                    reason="writer_stopped",
                    dedupe_key=dedupe_key,
                )
            if dedupe_key in self._known:
                self._counters["duplicate_total"] += 1
                return SnapshotSubmitResult(
                    SnapshotSubmitStatus.DUPLICATE,
                    dedupe_key,
                    f"already_{self._known[dedupe_key]}",
                )
            try:
                self._queue.put_nowait(envelope)
            except queue.Full:
                return self._reject(
                    status=SnapshotSubmitStatus.DROPPED,
                    reason="backpressure_queue_full",
                    dedupe_key=dedupe_key,
                )
            self._known[dedupe_key] = "queued"
            self._pending_count += 1
            self._counters["accepted_total"] += 1
            self._accepted_by_channel[channel] += 1
            accepted_at = envelope.queued_at
            previous_wall = self._last_accepted_wall_by_channel.get(channel)
            if previous_wall is not None:
                elapsed_slots = int(
                    max(0.0, accepted_at - previous_wall)
                    * 1000.0
                    // self.cadence_ms
                )
                wall_gap = max(0, elapsed_slots - 1)
                if wall_gap:
                    self._counters["wall_cadence_gap_total"] += wall_gap
                    self._wall_gaps_by_channel[channel] += wall_gap
            previous_source_slot = self._last_source_slot_by_channel.get(
                channel
            )
            if (
                previous_source_slot is not None
                and cadence_slot > previous_source_slot + 1
            ):
                source_gap = cadence_slot - previous_source_slot - 1
                self._counters["source_cadence_gap_total"] += source_gap
                self._source_gaps_by_channel[channel] += source_gap
            self._first_accepted_wall_by_channel.setdefault(
                channel,
                accepted_at,
            )
            self._last_accepted_wall_by_channel[channel] = accepted_at
            self._last_source_slot_by_channel[channel] = cadence_slot
            self._last_source_timestamp_by_channel[channel] = timestamp
            self._prune_known_locked()
            self._condition.notify_all()
        return SnapshotSubmitResult(
            SnapshotSubmitStatus.ACCEPTED,
            dedupe_key,
            "queued",
        )

    def submit_probe_frame(
        self,
        frame_result: Mapping[str, Any],
        *,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> SnapshotSubmitResult:
        """Adapter for the result returned by ``ProbeManager.add_frame``."""

        if not isinstance(frame_result, Mapping):
            return self._reject(
                status=SnapshotSubmitStatus.REJECTED,
                reason="validation:frame_result must be a mapping",
            )
        merged_provenance = dict(provenance or {})
        for key in ("embedding_ref", "frame_uid"):
            if frame_result.get(key) is not None:
                merged_provenance[key] = frame_result.get(key)
        return self.submit(
            channel_id=frame_result.get("channel_id"),
            timestamp_ms=frame_result.get("timestamp_ms"),
            embedding=frame_result.get("embedding"),
            thumbnail=frame_result.get("thumbnail"),
            provenance=merged_provenance,
        )

    def _run(self) -> None:
        while True:
            if self._stop_event.is_set() and self._queue.empty():
                break
            try:
                first = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            batch = [first]
            flush_deadline = self._monotonic() + self.flush_interval_seconds
            while len(batch) < self.batch_size:
                remaining = flush_deadline - self._monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._queue.get(timeout=remaining))
                except queue.Empty:
                    break

            with self._condition:
                self._in_flight += len(batch)
                for envelope in batch:
                    self._known[envelope.dedupe_key] = "writing"
            try:
                self._write_batch(batch)
            finally:
                with self._condition:
                    self._in_flight -= len(batch)
                    self._pending_count -= len(batch)
                    for _envelope in batch:
                        self._queue.task_done()
                    self._prune_known_locked()
                    self._condition.notify_all()

        with self._condition:
            self._condition.notify_all()

    def _write_batch(self, batch: Sequence[_SnapshotEnvelope]) -> None:
        error: Optional[Exception] = None
        for attempt in range(1, self.max_attempts + 1):
            with self._condition:
                self._counters["write_attempt_total"] += 1
            try:
                records = [dict(envelope.record) for envelope in batch]
                idempotent_writer = getattr(
                    self.detections_store,
                    "ensure_detections",
                    None,
                )
                if callable(idempotent_writer):
                    inserted = int(idempotent_writer(records))
                    with self._condition:
                        self._counters[
                            "idempotent_write_attempt_total"
                        ] += 1
                else:
                    inserted = int(
                        self.detections_store.add_detections(records)
                    )
                with self._condition:
                    self._counters["store_reported_inserted_total"] += max(
                        0,
                        inserted,
                    )
                if inserted != len(batch):
                    with self._condition:
                        self._counters["short_write_total"] += 1
                    raise _ShortWriteError(
                        f"archive accepted {inserted} of {len(batch)} snapshots"
                    )
                with self._condition:
                    self._counters["batch_persisted_total"] += 1
                    self._counters["persisted_total"] += len(batch)
                    self._last_error = None
                    for envelope in batch:
                        self._persisted_by_channel[envelope.channel_id] += 1
                        self._known[envelope.dedupe_key] = "persisted"
                return
            except Exception as exc:
                error = exc
                with self._condition:
                    self._last_error = f"{type(exc).__name__}: {exc}"[:500]
                if attempt >= self.max_attempts:
                    break
                with self._condition:
                    self._counters["retry_total"] += 1
                delay = min(
                    self.max_backoff_seconds,
                    self.initial_backoff_seconds * (2 ** (attempt - 1)),
                )
                if self._stop_event.wait(delay):
                    break

        message = (
            f"{type(error).__name__}: {error}"
            if error is not None
            else "unknown archive failure"
        )
        with self._condition:
            self._counters["batch_failed_total"] += 1
            self._counters["failure_total"] += len(batch)
            self._record_gap_locked("archive_write_failed", len(batch))
            self._last_error = message[:500]
            for envelope in batch:
                self._failed_by_channel[envelope.channel_id] += 1
                self._known[envelope.dedupe_key] = "failed"

    def _prune_known_locked(self) -> None:
        excess = len(self._known) - self.dedupe_capacity
        if excess <= 0:
            return
        removable = [
            key
            for key, state in self._known.items()
            if state in {"persisted", "failed", "dropped"}
        ]
        for key in removable[:excess]:
            self._known.pop(key, None)

    def drain(self, timeout: Optional[float] = None) -> bool:
        """Wait until every accepted snapshot is persisted or explicitly failed."""

        deadline = (
            None
            if timeout is None
            else self._monotonic() + max(0.0, float(timeout))
        )
        with self._condition:
            if self._pending_count and not self._started:
                return False
            while self._pending_count > 0:
                remaining = (
                    None
                    if deadline is None
                    else max(0.0, deadline - self._monotonic())
                )
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(
                    timeout=0.1 if remaining is None else min(0.1, remaining)
                )
            return True

    def _drop_queued(self, reason: str) -> int:
        dropped = []
        while True:
            try:
                dropped.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if not dropped:
            return 0
        with self._condition:
            for envelope in dropped:
                self._known[envelope.dedupe_key] = "dropped"
                self._queue.task_done()
            self._pending_count -= len(dropped)
            self._counters["rejected_total"] += len(dropped)
            self._counters["dropped_total"] += len(dropped)
            self._record_gap_locked(reason, len(dropped))
            self._condition.notify_all()
        return len(dropped)

    def stop(
        self,
        *,
        drain: bool = True,
        timeout: float = 5.0,
    ) -> bool:
        """Stop accepting work and return whether all accepted work was drained."""

        timeout_value = max(0.0, float(timeout))
        started_at = self._monotonic()
        with self._condition:
            if self._stopped:
                return self._pending_count == 0
            self._accepting = False
            needs_start = drain and self._pending_count > 0 and not self._started
        if needs_start:
            self.start()

        drained = self.drain(timeout=timeout_value) if drain else False
        if not drained:
            self._drop_queued(
                "stop_without_drain" if not drain else "stop_drain_timeout"
            )
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            elapsed = max(0.0, self._monotonic() - started_at)
            thread.join(timeout=max(0.0, timeout_value - elapsed))
        with self._condition:
            alive = bool(thread is not None and thread.is_alive())
            self._stopped = not alive
            self._condition.notify_all()
            return bool(drained and not alive)

    def status(self) -> Dict[str, Any]:
        with self._condition:
            oldest_age = 0.0
            with self._queue.mutex:
                queued_items = list(self._queue.queue)
            if queued_items:
                oldest_age = max(
                    0.0,
                    self._monotonic() - queued_items[0].queued_at,
                )
            counters = {
                key: int(self._counters.get(key, 0))
                for key in self._COUNTER_KEYS
            }
            counters.update(
                {
                    key: int(value)
                    for key, value in self._counters.items()
                    if key not in counters
                }
            )
            now_monotonic = self._monotonic()
            channel_cadence: Dict[str, Dict[str, Any]] = {}
            for channel_id, accepted_count in sorted(
                self._accepted_by_channel.items()
            ):
                first_at = self._first_accepted_wall_by_channel.get(channel_id)
                last_at = self._last_accepted_wall_by_channel.get(channel_id)
                observed_seconds = (
                    max(0.0, last_at - first_at)
                    if first_at is not None and last_at is not None
                    else 0.0
                )
                observed_hz = (
                    max(0, int(accepted_count) - 1) / observed_seconds
                    if observed_seconds > 0
                    else 0.0
                )
                channel_cadence[str(channel_id)] = {
                    "accepted": int(accepted_count),
                    "persisted": int(
                        self._persisted_by_channel.get(channel_id, 0)
                    ),
                    "failed": int(
                        self._failed_by_channel.get(channel_id, 0)
                    ),
                    "observed_hz": round(observed_hz, 3),
                    "target_hz": round(1000.0 / self.cadence_ms, 3),
                    "staleness_seconds": round(
                        max(0.0, now_monotonic - last_at)
                        if last_at is not None
                        else 0.0,
                        3,
                    ),
                    "last_source_timestamp_ms": (
                        self._last_source_timestamp_by_channel.get(channel_id)
                    ),
                    "wall_gap_slots": int(
                        self._wall_gaps_by_channel.get(channel_id, 0)
                    ),
                    "source_gap_slots": int(
                        self._source_gaps_by_channel.get(channel_id, 0)
                    ),
                }
            return {
                "source": self.SOURCE,
                "format_version": self.FORMAT_VERSION,
                "cadence_ms": self.cadence_ms,
                "started": self._started,
                "accepting": self._accepting,
                "stopped": self._stopped,
                "worker_alive": bool(
                    self._thread is not None and self._thread.is_alive()
                ),
                "queue_depth": self._queue.qsize(),
                "max_queue": self.max_queue,
                "batch_size": self.batch_size,
                "pending": self._pending_count,
                "in_flight": self._in_flight,
                "oldest_queue_age_seconds": round(oldest_age, 3),
                "dedupe_entries": len(self._known),
                "counters": dict(sorted(counters.items())),
                "accepted_by_channel": {
                    str(channel_id): int(count)
                    for channel_id, count in sorted(
                        self._accepted_by_channel.items()
                    )
                },
                "persisted_by_channel": {
                    str(channel_id): int(count)
                    for channel_id, count in sorted(
                        self._persisted_by_channel.items()
                    )
                },
                "failed_by_channel": {
                    str(channel_id): int(count)
                    for channel_id, count in sorted(
                        self._failed_by_channel.items()
                    )
                },
                "channel_cadence": channel_cadence,
                "gap_reasons": dict(sorted(self._gap_reasons.items())),
                "last_error": self._last_error,
            }


__all__ = [
    "SemanticSnapshotArchiveWriter",
    "SnapshotSubmitResult",
    "SnapshotSubmitStatus",
]
