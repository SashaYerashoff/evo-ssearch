"""Grounded incident reconstruction over EVA's existing visual memory.

The service deliberately builds a bounded, evidence-linked draft before any
language model is asked to narrate it.  It treats VLM text, state transitions,
attention telemetry and probe P/N/M as separate sources and never upgrades an
attention signal into visual proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import statistics
from typing import Any, Mapping, Optional, Protocol, Sequence
import xml.etree.ElementTree as ET


INCIDENT_MAX_WINDOW_MS = 6 * 60 * 60 * 1000
INCIDENT_DEFAULT_RADIUS_MS = 15 * 60 * 1000
INCIDENT_CONTINUITY_GAP_MS = 2 * 60 * 1000
INCIDENT_SEMANTIC_ROLE_WINDOW_MS = 30 * 1000
INCIDENT_SEMANTIC_QUERY_LIMIT = 64
INCIDENT_MAX_SEMANTIC_REFS = 5
INCIDENT_MAX_SCORE_REFS = 256


class DetectionMemory(Protocol):
    def fetch_detections_by_ids(
        self,
        ids: Sequence[int],
        include_vectors: bool = True,
        include_thumbnail: bool = True,
    ) -> list[dict[str, Any]]: ...

    def list_detections(self, **kwargs: Any) -> tuple[list[dict[str, Any]], int]: ...

    def list_vlm_summary_batches(self, **kwargs: Any) -> tuple[list[dict[str, Any]], int]: ...


class AttentionMemory(Protocol):
    def query_intervals(self, **kwargs: Any) -> list[dict[str, Any]]: ...

    def query_probe_scores(self, **kwargs: Any) -> list[dict[str, Any]]: ...

    def query_evidence_links(self, **kwargs: Any) -> list[dict[str, Any]]: ...


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return (
        value
        if isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        else ()
    )


def _text(value: Any, maximum: int = 4000) -> str:
    return str(value or "").strip()[:maximum]


def _timestamp_label(timestamp_ms: Any) -> str:
    value = _int(timestamp_ms)
    if value <= 0:
        return ""
    return datetime.fromtimestamp(value / 1000.0, timezone.utc).isoformat()


def _payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(row.get("payload"))


def _summary_timestamp_ms(row: Mapping[str, Any], key: str) -> int:
    direct = _int(row.get(key))
    if direct > 0:
        return direct
    return int(max(0.0, _float(row.get("created_at"))) * 1000.0)


def _event_timestamp(raw: Mapping[str, Any], fallback: int) -> int:
    for key in (
        "timestamp_ms",
        "occurred_at_ms",
        "started_at_ms",
        "start_ms",
        "transition_at_ms",
    ):
        parsed = _int(raw.get(key))
        if parsed > 0:
            return parsed
    return int(fallback)


def _event_label(raw: Mapping[str, Any], fallback: str) -> str:
    for key in ("title", "label", "event", "event_type", "summary", "evidence"):
        candidate = _text(raw.get(key), 600)
        if candidate:
            return candidate
    return fallback


def _event_key(raw: Mapping[str, Any], fallback: str) -> str:
    for key in ("key", "event_id", "event_type", "state_key", "type"):
        candidate = _text(raw.get(key), 120)
        if candidate:
            return candidate
    return fallback


def _event_severity(raw: Mapping[str, Any], fallback: str = "info") -> str:
    value = _text(raw.get("severity") or fallback, 24).lower()
    return value if value in {"info", "low", "normal", "high", "critical"} else fallback


def _summary_is_candidate(row: Mapping[str, Any]) -> bool:
    payload = _payload(row)
    if _int(row.get("alert_total") or payload.get("alert_total")) > 0:
        return True
    if _int(
        row.get("state_transition_total")
        or payload.get("state_transition_total")
    ) > 0:
        return True
    return any(
        bool(_sequence(payload.get(key)))
        for key in (
            "alert_events",
            "state_transition_events",
            "events",
        )
    )


def _summary_row_key(row: Mapping[str, Any]) -> str:
    payload = _payload(row)
    batch_id = _text(row.get("batch_id") or payload.get("batch_id"), 160)
    if batch_id:
        return f"batch:{batch_id}"
    return (
        f"window:{_summary_timestamp_ms(row, 'batch_start_ms')}:"
        f"{_summary_timestamp_ms(row, 'batch_end_ms')}:"
        f"{_int(row.get('archive_id') or row.get('id'))}"
    )


def _connected_candidate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    anchor_ms: int,
    max_gap_ms: int,
) -> list[Mapping[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda item: (
            _summary_timestamp_ms(item, "batch_start_ms"),
            _int(item.get("archive_id")),
        ),
    )
    candidates = [row for row in ordered if _summary_is_candidate(row)]
    if not candidates:
        if not ordered:
            return []
        return [
            min(
                ordered,
                key=lambda item: abs(
                    _summary_timestamp_ms(item, "batch_end_ms") - anchor_ms
                ),
            )
        ]
    seed = min(
        range(len(candidates)),
        key=lambda index: abs(
            _int(candidates[index].get("batch_end_ms")) - anchor_ms
        ),
    )
    if (
        abs(_summary_timestamp_ms(candidates[seed], "batch_end_ms") - anchor_ms)
        > max_gap_ms
    ):
        # A distant alert is not evidence that the operator-selected quiet
        # batch belongs to that incident.  Keep the anchor local and explicit.
        return [
            min(
                ordered,
                key=lambda item: abs(
                    _summary_timestamp_ms(item, "batch_end_ms") - anchor_ms
                ),
            )
        ]
    selected = [candidates[seed]]
    cursor_start = _int(candidates[seed].get("batch_start_ms"))
    cursor_end = _int(candidates[seed].get("batch_end_ms"), cursor_start)
    index = seed - 1
    while index >= 0:
        row_end = _int(candidates[index].get("batch_end_ms"))
        if cursor_start - row_end > max_gap_ms:
            break
        selected.insert(0, candidates[index])
        cursor_start = _int(candidates[index].get("batch_start_ms"), row_end)
        index -= 1
    index = seed + 1
    while index < len(candidates):
        row_start = _int(candidates[index].get("batch_start_ms"))
        if row_start - cursor_end > max_gap_ms:
            break
        selected.append(candidates[index])
        cursor_end = _int(candidates[index].get("batch_end_ms"), row_start)
        index += 1
    return selected


@dataclass(frozen=True)
class IncidentDraftRequest:
    channel_id: int
    anchor_detection_id: Optional[int] = None
    since_ms: Optional[int] = None
    until_ms: Optional[int] = None
    now_ms: Optional[int] = None

    def __post_init__(self) -> None:
        if int(self.channel_id) <= 0:
            raise ValueError("channel_id must be positive")
        if self.anchor_detection_id is not None and int(self.anchor_detection_id) <= 0:
            raise ValueError("anchor_detection_id must be positive")
        if self.since_ms is not None and int(self.since_ms) < 0:
            raise ValueError("since_ms must be non-negative")
        if self.until_ms is not None and int(self.until_ms) < 0:
            raise ValueError("until_ms must be non-negative")
        if (
            self.since_ms is not None
            and self.until_ms is not None
            and int(self.since_ms) > int(self.until_ms)
        ):
            raise ValueError("since_ms must not be later than until_ms")


class IncidentDraftAssembler:
    """Build a bounded incident draft from existing memory and telemetry."""

    def __init__(
        self,
        detections: DetectionMemory,
        attention: Optional[AttentionMemory] = None,
    ) -> None:
        self.detections = detections
        self.attention = attention

    def assemble(self, request: IncidentDraftRequest) -> dict[str, Any]:
        channel_id = int(request.channel_id)
        anchor_row: Optional[Mapping[str, Any]] = None
        anchor_ms = int(request.now_ms or 0)
        if request.anchor_detection_id is not None:
            rows = self.detections.fetch_detections_by_ids(
                [int(request.anchor_detection_id)],
                include_vectors=False,
                include_thumbnail=False,
            )
            if not rows:
                raise LookupError("anchor detection not found")
            anchor_row = rows[0]
            if _int(anchor_row.get("channel_id")) != channel_id:
                raise ValueError("anchor detection does not belong to channel")
            anchor_ms = _int(anchor_row.get("timestamp_ms"))
        if anchor_ms <= 0:
            anchor_ms = int(request.until_ms or request.since_ms or 0)
        if anchor_ms <= 0:
            raise ValueError("an anchor detection or explicit time window is required")

        since_ms = int(
            request.since_ms
            if request.since_ms is not None
            else max(0, anchor_ms - INCIDENT_DEFAULT_RADIUS_MS)
        )
        until_ms = int(
            request.until_ms
            if request.until_ms is not None
            else anchor_ms + INCIDENT_DEFAULT_RADIUS_MS
        )
        if until_ms - since_ms > INCIDENT_MAX_WINDOW_MS:
            raise ValueError("incident draft window must not exceed 6 hours")

        summaries, summary_total = self.detections.list_vlm_summary_batches(
            channel_id=channel_id,
            since_ms=since_ms,
            until_ms=until_ms,
            limit=500,
            offset=0,
        )
        raw_window_summaries, raw_summary_total = self.detections.list_detections(
            channel_id=channel_id,
            source="vlm_summary",
            since_ms=since_ms,
            until_ms=until_ms,
            limit=500,
            offset=0,
            include_thumbnail=False,
        )
        structured_window = self._dedupe_summary_detections(
            raw_window_summaries,
            (),
        )
        summaries = self._augment_summary_batches(
            summaries,
            structured_window,
        )
        connected = _connected_candidate_rows(
            summaries,
            anchor_ms=anchor_ms,
            max_gap_ms=INCIDENT_CONTINUITY_GAP_MS,
        )
        context_summaries, context_roles = self._summary_context(
            summaries,
            connected,
        )
        observed_start = min(
            (_int(row.get("batch_start_ms"), anchor_ms) for row in connected),
            default=anchor_ms,
        )
        candidate_end = max(
            (_int(row.get("batch_end_ms"), anchor_ms) for row in connected),
            default=anchor_ms,
        )
        drill_since = max(since_ms, observed_start - INCIDENT_CONTINUITY_GAP_MS)
        drill_until = min(until_ms, candidate_end + INCIDENT_CONTINUITY_GAP_MS)

        alerts, alert_total = self.detections.list_detections(
            channel_id=channel_id,
            source="vlm_alert",
            since_ms=drill_since,
            until_ms=drill_until,
            limit=500,
            offset=0,
            include_thumbnail=False,
        )
        structured_summaries = self._dedupe_summary_detections(
            structured_window,
            context_summaries,
        )
        telemetry = self._attention_rows(channel_id, drill_since, drill_until)
        timeline = self._timeline(
            structured_summaries,
            alerts,
            telemetry["intervals"],
        )
        apex_ms = self._apex_timestamp(timeline, anchor_ms)
        resolution_ms = self._resolution_timestamp(timeline)
        has_post_control = any(
            role == "context_after"
            and not _summary_is_candidate(row)
            for row in context_summaries
            for role in (context_roles.get(_summary_row_key(row)),)
        )
        observed_end: Optional[int] = (
            int(resolution_ms)
            if resolution_ms is not None
            else int(candidate_end)
            if has_post_control
            else None
        )
        semantic_refs, semantic_status = self._semantic_snapshot_refs(
            channel_id=channel_id,
            drill_since=drill_since,
            drill_until=drill_until,
            observed_start=observed_start,
            apex_ms=apex_ms,
            candidate_end=candidate_end,
            context_summaries=context_summaries,
            context_roles=context_roles,
        )
        evidence = self._evidence(
            context_summaries,
            alerts,
            telemetry["links"],
            summary_roles=context_roles,
            semantic_refs=semantic_refs,
        )
        qualia = self._qualia_digest(
            telemetry["scores"],
            telemetry["intervals"],
            attention_status=telemetry["status"],
        )
        coverage = self._coverage(
            summaries,
            since_ms=drill_since,
            until_ms=drill_until,
            summary_total=summary_total,
            attention_status=telemetry["status"],
            semantic_status=semantic_status,
        )
        severity = self._severity(timeline)
        title = self._title(timeline, anchor_row)
        uncertainties: list[str] = []
        if not connected:
            uncertainties.append("No L0 batch was available around the anchor.")
        if coverage["status"] != "covered":
            uncertainties.append("The reconstructed period contains incomplete video-description coverage.")
        if not timeline:
            uncertainties.append("No structured event transition was found; the draft is anchored to the selected evidence only.")
        if request.since_ms is None or request.until_ms is None:
            uncertainties.append("Time bounds were proposed automatically and require operator review.")

        return {
            "state": "draft",
            "title": title,
            "severity": severity,
            "primary_channel_id": channel_id,
            "channel_ids": [channel_id],
            "anchor": {
                "type": "detection" if anchor_row is not None else "window",
                "detection_id": int(request.anchor_detection_id)
                if request.anchor_detection_id is not None
                else None,
                "timestamp_ms": anchor_ms,
            },
            "time_bounds": {
                "possible_start_ms": drill_since,
                "observed_start_ms": observed_start,
                "apex_ms": apex_ms,
                "observed_end_ms": observed_end,
                "possible_end_ms": drill_until,
                "start_status": "observed" if connected else "possible",
                "end_status": (
                    "resolved"
                    if resolution_ms is not None
                    else "post_control"
                    if has_post_control
                    else "open"
                ),
            },
            "summary_context": self._summary_context_digest(
                context_summaries,
                context_roles,
            ),
            "timeline": timeline,
            "evidence": evidence,
            "qualia_digest": qualia,
            "coverage": coverage,
            "uncertainties": uncertainties,
            "focus_lease": None,
            "provenance": {
                "summary_rows_considered": len(summaries),
                "connected_summary_rows": len(connected),
                "context_summary_rows": len(context_summaries),
                "structured_summary_rows": len(structured_summaries),
                "structured_summary_rows_available": int(raw_summary_total),
                "alert_rows": len(alerts),
                "alert_rows_available": int(alert_total),
                "attention_store": bool(self.attention is not None),
            },
        }

    def refresh(
        self,
        incident: Mapping[str, Any],
        *,
        until_ms: int,
    ) -> dict[str, Any]:
        """Rebuild an existing envelope through a later point in time.

        The method is side-effect free. Callers decide whether and how to
        persist the refreshed record, which keeps follow scheduling and
        authorization outside this reconstruction module.
        """

        channel_ids = [
            _int(value)
            for value in _sequence(incident.get("channel_ids"))
            if _int(value) > 0
        ]
        channel_id = _int(incident.get("primary_channel_id")) or (
            channel_ids[0] if channel_ids else 0
        )
        if channel_id <= 0:
            raise ValueError("incident has no positive channel id")
        bounds = _mapping(incident.get("time_bounds"))
        since_ms = _int(
            bounds.get("possible_start_ms")
            or incident.get("possible_start_ms")
        )
        if since_ms <= 0:
            raise ValueError("incident has no possible start")
        anchor = _mapping(incident.get("anchor") or incident.get("anchor_ref"))
        anchor_detection_id = _int(anchor.get("detection_id")) or None
        anchor_ms = _int(anchor.get("timestamp_ms")) or since_ms
        refreshed = self.assemble(
            IncidentDraftRequest(
                channel_id=channel_id,
                anchor_detection_id=anchor_detection_id,
                since_ms=since_ms,
                until_ms=int(until_ms),
                now_ms=anchor_ms,
            )
        )
        for key in ("id", "revision", "state", "focus_lease"):
            if key in incident:
                refreshed[key] = incident[key]
        return refreshed

    @staticmethod
    def _augment_summary_batches(
        summaries: Sequence[Mapping[str, Any]],
        structured: Sequence[Mapping[str, Any]],
    ) -> list[Mapping[str, Any]]:
        structured_by_batch: dict[str, Mapping[str, Any]] = {}
        for row in structured:
            payload = _payload(row)
            batch_id = _text(payload.get("batch_id"), 160)
            if batch_id:
                structured_by_batch[batch_id] = row
        augmented: list[Mapping[str, Any]] = []
        for raw in summaries:
            row = dict(raw)
            batch_id = _text(row.get("batch_id"), 160)
            structured_row = structured_by_batch.get(batch_id)
            if structured_row is not None:
                structured_payload = dict(_payload(structured_row))
                existing_payload = dict(_payload(row))
                existing_payload.update(structured_payload)
                row["payload"] = existing_payload
                transitions = _sequence(
                    structured_payload.get("state_transition_events")
                )
                if transitions:
                    row["state_transition_total"] = len(transitions)
            augmented.append(row)
        return augmented

    @staticmethod
    def _summary_context(
        summaries: Sequence[Mapping[str, Any]],
        connected: Sequence[Mapping[str, Any]],
    ) -> tuple[list[Mapping[str, Any]], dict[str, str]]:
        ordered = sorted(
            summaries,
            key=lambda row: (
                _summary_timestamp_ms(row, "batch_start_ms"),
                _summary_timestamp_ms(row, "batch_end_ms"),
                _summary_row_key(row),
            ),
        )
        connected_keys = {_summary_row_key(row) for row in connected}
        roles = {key: "candidate" for key in connected_keys}
        indexes = [
            index
            for index, row in enumerate(ordered)
            if _summary_row_key(row) in connected_keys
        ]
        if not indexes:
            return list(connected), roles
        first_index = min(indexes)
        last_index = max(indexes)
        before = next(
            (
                ordered[index]
                for index in range(first_index - 1, -1, -1)
                if not _summary_is_candidate(ordered[index])
            ),
            None,
        )
        after = next(
            (
                ordered[index]
                for index in range(last_index + 1, len(ordered))
                if not _summary_is_candidate(ordered[index])
            ),
            None,
        )
        selected = list(connected)
        if before is not None:
            selected.append(before)
            roles[_summary_row_key(before)] = "context_before"
        if after is not None:
            selected.append(after)
            roles[_summary_row_key(after)] = "context_after"
        selected.sort(
            key=lambda row: (
                _summary_timestamp_ms(row, "batch_start_ms"),
                _summary_row_key(row),
            )
        )
        return selected, roles

    @staticmethod
    def _summary_context_digest(
        rows: Sequence[Mapping[str, Any]],
        roles: Mapping[str, str],
    ) -> list[dict[str, Any]]:
        return [
            {
                "batch_id": _text(row.get("batch_id"), 160),
                "batch_start_ms": _summary_timestamp_ms(row, "batch_start_ms"),
                "batch_end_ms": _summary_timestamp_ms(row, "batch_end_ms"),
                "role": roles.get(_summary_row_key(row), "candidate"),
                "candidate": _summary_is_candidate(row),
            }
            for row in rows[:502]
        ]

    @staticmethod
    def _dedupe_summary_detections(
        rows: Sequence[Mapping[str, Any]],
        connected: Sequence[Mapping[str, Any]],
    ) -> list[Mapping[str, Any]]:
        connected_batch_ids = {
            _text(row.get("batch_id"), 160)
            for row in connected
            if _text(row.get("batch_id"), 160)
        }
        selected: dict[str, Mapping[str, Any]] = {}
        for row in sorted(rows, key=lambda item: _int(item.get("timestamp_ms"))):
            payload = _payload(row)
            batch_id = _text(payload.get("batch_id"), 160)
            if connected_batch_ids and batch_id not in connected_batch_ids:
                continue
            key = batch_id or (
                f"{_int(payload.get('batch_start_ms') or row.get('timestamp_ms'))}:"
                f"{_int(payload.get('batch_end_ms') or row.get('timestamp_ms'))}"
            )
            current = selected.get(key)
            if current is None:
                selected[key] = row
                continue
            current_payload = _payload(current)
            current_signal_count = sum(
                len(_sequence(current_payload.get(field)))
                for field in ("alert_events", "state_transition_events", "events")
            )
            candidate_signal_count = sum(
                len(_sequence(payload.get(field)))
                for field in ("alert_events", "state_transition_events", "events")
            )
            if candidate_signal_count > current_signal_count:
                selected[key] = row
        return sorted(
            selected.values(),
            key=lambda item: _int(item.get("timestamp_ms")),
        )

    def _attention_rows(
        self,
        channel_id: int,
        since_ms: int,
        until_ms: int,
    ) -> dict[str, Any]:
        specs = {
            "intervals": ("query_intervals", 2000),
            "scores": ("query_probe_scores", 5000),
            "links": ("query_evidence_links", 2000),
        }
        rows: dict[str, Any] = {key: [] for key in specs}
        query_status: dict[str, dict[str, Any]] = {}
        if self.attention is None:
            for key in specs:
                query_status[key] = {
                    "status": "unavailable",
                    "row_count": 0,
                    "reason": "attention_store_unavailable",
                }
            rows["status"] = {
                "overall": "unavailable",
                "queries": query_status,
            }
            return rows
        successful = 0
        for key, (method_name, limit) in specs.items():
            method = getattr(self.attention, method_name, None)
            if not callable(method):
                query_status[key] = {
                    "status": "unavailable",
                    "row_count": 0,
                    "reason": "query_not_supported",
                }
                continue
            try:
                values = method(
                    channel_id=channel_id,
                    start_ms=since_ms,
                    end_ms=until_ms,
                    limit=limit,
                )
                rows[key] = [
                    dict(value)
                    for value in values
                    if isinstance(value, Mapping)
                ]
                successful += 1
                query_status[key] = {
                    "status": "ok",
                    "row_count": len(rows[key]),
                }
            except Exception as exc:
                query_status[key] = {
                    "status": "unavailable",
                    "row_count": 0,
                    "reason": type(exc).__name__,
                }
        rows["status"] = {
            "overall": (
                "ok"
                if successful == len(specs)
                else "partial"
                if successful > 0
                else "unavailable"
            ),
            "queries": query_status,
        }
        return rows

    def _semantic_snapshot_refs(
        self,
        *,
        channel_id: int,
        drill_since: int,
        drill_until: int,
        observed_start: int,
        apex_ms: int,
        candidate_end: int,
        context_summaries: Sequence[Mapping[str, Any]],
        context_roles: Mapping[str, str],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        before_row = next(
            (
                row
                for row in context_summaries
                if context_roles.get(_summary_row_key(row)) == "context_before"
            ),
            None,
        )
        after_row = next(
            (
                row
                for row in context_summaries
                if context_roles.get(_summary_row_key(row)) == "context_after"
            ),
            None,
        )
        targets: list[tuple[str, int, str]] = [
            (
                "control_before",
                (
                    _summary_timestamp_ms(before_row, "batch_start_ms")
                    + _summary_timestamp_ms(before_row, "batch_end_ms")
                )
                // 2
                if before_row is not None
                else observed_start,
                "before",
            ),
            ("onset", observed_start, "near"),
            ("apex", apex_ms, "near"),
            ("post", candidate_end, "near"),
        ]
        if after_row is not None:
            targets.append(
                (
                    "control_after",
                    (
                        _summary_timestamp_ms(after_row, "batch_start_ms")
                        + _summary_timestamp_ms(after_row, "batch_end_ms")
                    )
                    // 2,
                    "after",
                )
            )
        refs: list[dict[str, Any]] = []
        query_status: dict[str, dict[str, Any]] = {}
        for role, target_ms, direction in targets[:INCIDENT_MAX_SEMANTIC_REFS]:
            if direction == "before":
                query_since = max(
                    drill_since,
                    target_ms - INCIDENT_SEMANTIC_ROLE_WINDOW_MS,
                )
                query_until = min(drill_until, target_ms)
            elif direction == "after":
                query_since = max(drill_since, target_ms)
                query_until = min(
                    drill_until,
                    target_ms + INCIDENT_SEMANTIC_ROLE_WINDOW_MS,
                )
            else:
                query_since = max(
                    drill_since,
                    target_ms - INCIDENT_SEMANTIC_ROLE_WINDOW_MS,
                )
                query_until = min(
                    drill_until,
                    target_ms + INCIDENT_SEMANTIC_ROLE_WINDOW_MS,
                )
            if query_until < query_since:
                query_status[role] = {
                    "status": "empty",
                    "row_count": 0,
                }
                continue
            try:
                values, total = self.detections.list_detections(
                    channel_id=channel_id,
                    source="semantic_snapshot",
                    since_ms=query_since,
                    until_ms=query_until,
                    limit=INCIDENT_SEMANTIC_QUERY_LIMIT,
                    offset=0,
                    include_thumbnail=False,
                )
            except Exception as exc:
                query_status[role] = {
                    "status": "unavailable",
                    "row_count": 0,
                    "reason": type(exc).__name__,
                }
                continue
            candidates = [
                value
                for value in values
                if isinstance(value, Mapping)
                and _text(value.get("source"), 40) == "semantic_snapshot"
            ]
            query_status[role] = {
                "status": "ok" if candidates else "empty",
                "row_count": len(candidates),
                "rows_available": int(total),
            }
            if not candidates:
                continue
            selected = min(
                candidates,
                key=lambda value: (
                    abs(_int(value.get("timestamp_ms")) - target_ms),
                    _int(value.get("timestamp_ms")),
                    _int(value.get("id")),
                ),
            )
            payload = _payload(selected)
            refs.append(
                {
                    "kind": "semantic_snapshot",
                    "role": role,
                    "detection_id": _int(selected.get("id")) or None,
                    "timestamp_ms": _int(selected.get("timestamp_ms")),
                    "cadence_ms": _int(payload.get("cadence_ms")) or None,
                    "shard_key": _text(selected.get("shard_key"), 160) or None,
                }
            )
        statuses = {
            str(value.get("status") or "unavailable")
            for value in query_status.values()
        }
        overall = (
            "unavailable"
            if query_status and statuses == {"unavailable"}
            else "partial"
            if "unavailable" in statuses or "empty" in statuses
            else "ok"
        )
        return refs, {
            "overall": overall,
            "references": len(refs),
            "queries": query_status,
        }

    @staticmethod
    def _timeline(
        summaries: Sequence[Mapping[str, Any]],
        alerts: Sequence[Mapping[str, Any]],
        intervals: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for row in summaries:
            payload = _payload(row)
            fallback_ts = _int(
                payload.get("batch_end_ms") or row.get("timestamp_ms")
            )
            for field, source, fallback_key in (
                ("alert_events", "vlm_structured_alert", "alert"),
                ("state_transition_events", "state_transition", "transition"),
                ("events", "batch_state", "event"),
            ):
                for raw in _sequence(payload.get(field)):
                    if not isinstance(raw, Mapping):
                        continue
                    item = {
                        "timestamp_ms": _event_timestamp(raw, fallback_ts),
                        "semantic_key": _event_key(raw, fallback_key),
                        "label": _event_label(raw, fallback_key.replace("_", " ").title()),
                        "severity": _event_severity(raw),
                        "confidence": _text(raw.get("confidence"), 24) or "unknown",
                        "source": source,
                        "summary_detection_id": _int(row.get("id")) or None,
                    }
                    for key in (
                        "state",
                        "event_type",
                        "from_state",
                        "to_state",
                    ):
                        value = _text(raw.get(key), 40).lower()
                        if value:
                            item[key] = value
                    snapshot_indices = [
                        _int(value)
                        for value in _sequence(raw.get("snapshot_indices"))
                        if _int(value) > 0
                    ][:16]
                    if snapshot_indices:
                        item["snapshot_indices"] = snapshot_indices
                    items.append(item)
        for row in alerts:
            payload = _payload(row)
            item = {
                "timestamp_ms": _int(row.get("timestamp_ms")),
                "semantic_key": _event_key(payload, "vlm_alert"),
                "label": _event_label(
                    payload,
                    _text(row.get("probe_name"), 600) or "VLM alert",
                ),
                "severity": _event_severity(payload, _text(row.get("severity"), 24) or "normal"),
                "confidence": _text(payload.get("confidence"), 24) or "unknown",
                "source": "vlm_alert",
                "detection_id": _int(row.get("id")) or None,
            }
            state = _text(payload.get("state"), 40).lower()
            if state:
                item["state"] = state
            items.append(item)
        for interval in intervals:
            if _text(interval.get("state"), 24) not in {"motion", "mixed"}:
                continue
            timestamp_ms = _int(interval.get("peak_at_ms") or interval.get("started_at_ms"))
            items.append(
                {
                    "timestamp_ms": timestamp_ms,
                    "semantic_key": "motion_peak",
                    "label": (
                        "Homeostatic motion peak "
                        f"(p95={_float(interval.get('motion_p95')):.3f}, "
                        f"activity={_float(interval.get('activity_x_max')):.3f})"
                    ),
                    "severity": "info",
                    "confidence": "signal_only",
                    "source": "cv_motion_interval",
                    "attention_interval_id": _text(interval.get("id"), 80) or None,
                }
            )
        items.sort(key=lambda item: (_int(item.get("timestamp_ms")), _text(item.get("source"))))
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[int, str, str]] = set()
        for item in items:
            key = (
                _int(item.get("timestamp_ms")),
                _text(item.get("semantic_key"), 120),
                _text(item.get("label"), 300),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped[:500]

    @staticmethod
    def _resolution_timestamp(
        timeline: Sequence[Mapping[str, Any]],
    ) -> Optional[int]:
        resolved_states = {
            "resolved",
            "finished",
            "ended",
            "closed",
            "absent",
        }
        timestamps: list[int] = []
        for item in timeline:
            state = _text(item.get("state"), 40).lower()
            to_state = _text(item.get("to_state"), 40).lower()
            if (
                state in resolved_states
                or to_state == "absent"
            ):
                timestamp_ms = _int(item.get("timestamp_ms"))
                if timestamp_ms > 0:
                    timestamps.append(timestamp_ms)
        return max(timestamps) if timestamps else None

    @staticmethod
    def _evidence(
        summaries: Sequence[Mapping[str, Any]],
        alerts: Sequence[Mapping[str, Any]],
        links: Sequence[Mapping[str, Any]],
        *,
        summary_roles: Optional[Mapping[str, str]] = None,
        semantic_refs: Sequence[Mapping[str, Any]] = (),
    ) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for row in summaries:
            detection_id = _int(row.get("archive_id"))
            if detection_id <= 0:
                continue
            key = ("detection", str(detection_id))
            if key in seen:
                continue
            seen.add(key)
            evidence.append(
                {
                    "kind": "detection",
                    "role": (
                        (summary_roles or {}).get(
                            _summary_row_key(row),
                            "summary_anchor",
                        )
                    ),
                    "detection_id": detection_id,
                    "timestamp_ms": _int(row.get("batch_end_ms")),
                    "batch_id": _text(row.get("batch_id"), 160),
                }
            )
        for row in alerts:
            detection_id = _int(row.get("id"))
            if detection_id <= 0:
                continue
            key = ("detection", str(detection_id))
            if key in seen:
                continue
            seen.add(key)
            evidence.append(
                {
                    "kind": "detection",
                    "role": "alert",
                    "detection_id": detection_id,
                    "timestamp_ms": _int(row.get("timestamp_ms")),
                }
            )
        for row in links:
            reference = _text(
                row.get("embedding_snapshot_id") or row.get("apex_ref") or row.get("id"),
                1024,
            )
            if not reference:
                continue
            kind = _text(row.get("kind"), 24) or "attention"
            key = (kind, reference)
            if key in seen:
                continue
            seen.add(key)
            evidence.append(
                {
                    "kind": kind,
                    "role": _text(row.get("role"), 24) or "support",
                    "reference": reference,
                    "timestamp_ms": _int(row.get("occurred_at_ms")),
                }
            )
        for row in semantic_refs:
            detection_id = _int(row.get("detection_id"))
            role = _text(row.get("role"), 32)
            if detection_id <= 0 or not role:
                continue
            evidence.append(
                {
                    "kind": "semantic_snapshot",
                    "role": role,
                    "detection_id": detection_id,
                    "timestamp_ms": _int(row.get("timestamp_ms")),
                    "cadence_ms": _int(row.get("cadence_ms")) or None,
                    "shard_key": _text(row.get("shard_key"), 160) or None,
                }
            )
        evidence.sort(key=lambda item: _int(item.get("timestamp_ms")))
        return evidence[:500]

    @staticmethod
    def _qualia_digest(
        scores: Sequence[Mapping[str, Any]],
        intervals: Sequence[Mapping[str, Any]],
        *,
        attention_status: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        by_probe: dict[str, dict[str, Any]] = {}
        score_refs: list[dict[str, Any]] = []
        for row in scores:
            probe_id = _text(row.get("probe_id"), 160)
            if not probe_id:
                continue
            current = by_probe.setdefault(
                probe_id,
                {
                    "probe_id": probe_id,
                    "samples": 0,
                    "hits": 0,
                    "max_positive": -1.0,
                    "max_negative": -1.0,
                    "max_margin": -2.0,
                },
            )
            current["samples"] += 1
            if _text(row.get("threshold_state"), 32) == "hit":
                current["hits"] += 1
            current["max_positive"] = max(current["max_positive"], _float(row.get("pos_score"), -1.0))
            current["max_negative"] = max(current["max_negative"], _float(row.get("neg_score"), -1.0))
            current["max_margin"] = max(current["max_margin"], _float(row.get("margin"), -2.0))
            if len(score_refs) < INCIDENT_MAX_SCORE_REFS:
                score_refs.append(
                    {
                        "probe_id": probe_id,
                        "probe_version": _text(row.get("probe_version"), 120),
                        "embedding_snapshot_id": _text(
                            row.get("embedding_snapshot_id"),
                            160,
                        )
                        or None,
                        "captured_at_ms": _int(row.get("captured_at_ms")) or None,
                        "scored_at_ms": _int(row.get("scored_at_ms")) or None,
                        "positive": _float(row.get("pos_score")),
                        "negative": _float(row.get("neg_score")),
                        "margin": _float(row.get("margin")),
                        "threshold_state": _text(
                            row.get("threshold_state"),
                            32,
                        ),
                    }
                )
        motion_values = [
            _float(row.get("motion_p95"))
            for row in intervals
            if _text(row.get("state"), 24) in {"motion", "mixed"}
        ]
        active_intervals = [
            row
            for row in intervals
            if _text(row.get("state"), 24) in {"motion", "mixed"}
        ]
        activity_values = [
            _float(row.get("activity_x_max"))
            for row in active_intervals
        ]
        elevated = sorted(
            (
                _int(row.get("started_at_ms") or row.get("peak_at_ms")),
                _int(row.get("ended_at_ms") or row.get("peak_at_ms") or row.get("started_at_ms")),
                _float(row.get("activity_x_max")),
                _int(row.get("peak_at_ms") or row.get("started_at_ms")),
            )
            for row in active_intervals
            if _float(row.get("activity_x_max")) >= 3.0
        )
        burst_count = 0
        previous_end = -1
        for started_at_ms, ended_at_ms, _activity, _peak_at_ms in elevated:
            if burst_count == 0 or started_at_ms - previous_end > 1_500:
                burst_count += 1
            previous_end = max(previous_end, ended_at_ms)
        apex_index = max(
            range(len(active_intervals)),
            key=lambda index: _float(active_intervals[index].get("activity_x_max")),
            default=-1,
        )
        apex_at_ms = (
            _int(
                active_intervals[apex_index].get("peak_at_ms")
                or active_intervals[apex_index].get("started_at_ms")
            )
            if apex_index >= 0
            else None
        )
        final_elevated_end_ms = max((item[1] for item in elevated), default=0)
        return {
            "ground_truth": False,
            "interpretation": "attention signals only",
            "probe_count": len(by_probe),
            "probes": sorted(by_probe.values(), key=lambda item: (-item["hits"], item["probe_id"]))[:64],
            "score_refs": score_refs,
            "motion_interval_count": len(motion_values),
            "motion_p95_max": max(motion_values, default=0.0),
            "motion_p95_mean": statistics.fmean(motion_values) if motion_values else 0.0,
            "motion_profile": {
                "activity_x_max": max(activity_values, default=0.0),
                "activity_x_mean": statistics.fmean(activity_values) if activity_values else 0.0,
                "apex_at_ms": apex_at_ms,
                "elevated_duration_ms": sum(
                    max(0, ended_at_ms - started_at_ms)
                    for started_at_ms, ended_at_ms, _activity, _peak_at_ms in elevated
                ),
                "burst_count": burst_count,
                "settling_ms": (
                    max(0, final_elevated_end_ms - apex_at_ms)
                    if apex_at_ms is not None and final_elevated_end_ms > 0
                    else 0
                ),
            },
            "attention_status": dict(attention_status or {}),
        }

    @staticmethod
    def _coverage(
        summaries: Sequence[Mapping[str, Any]],
        *,
        since_ms: int,
        until_ms: int,
        summary_total: int,
        attention_status: Optional[Mapping[str, Any]] = None,
        semantic_status: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        spans = sorted(
            (
                _int(row.get("batch_start_ms")),
                _int(row.get("batch_end_ms")),
            )
            for row in summaries
            if _int(row.get("batch_end_ms")) >= _int(row.get("batch_start_ms")) > 0
        )
        durations = [end - start for start, end in spans if end > start]
        gap_threshold_ms = max(
            5_000,
            min(
                60_000,
                int(statistics.median(durations) / 2) if durations else 30_000,
            ),
        )
        gaps: list[dict[str, Any]] = []
        cursor = int(since_ms)
        covered_ms = 0
        for start, end in spans:
            clipped_start = max(since_ms, start)
            clipped_end = min(until_ms, end)
            if clipped_end < clipped_start:
                continue
            if clipped_start - cursor > gap_threshold_ms:
                gaps.append(
                    {
                        "kind": "inferred_l0_gap",
                        "start_ms": cursor,
                        "end_ms": clipped_start,
                        "duration_ms": clipped_start - cursor,
                    }
                )
            if clipped_end > cursor:
                covered_ms += max(0, clipped_end - max(cursor, clipped_start))
                cursor = clipped_end
        if until_ms - cursor > gap_threshold_ms:
            gaps.append(
                {
                    "kind": "inferred_l0_gap",
                    "start_ms": cursor,
                    "end_ms": until_ms,
                    "duration_ms": until_ms - cursor,
                }
            )
        window_ms = max(1, until_ms - since_ms)
        coverage_status = "no_data" if not spans else "partial" if gaps else "covered"
        covered_fraction = round(min(1.0, covered_ms / window_ms), 4)
        l0_ledger = {
            "status": coverage_status,
            "gap_threshold_ms": gap_threshold_ms,
            "span_count": len(spans),
            "inferred_gap_count": len(gaps),
            "covered_fraction_estimate": covered_fraction,
            "gaps": gaps[:100],
        }
        return {
            "status": coverage_status,
            "window_start_ms": since_ms,
            "window_end_ms": until_ms,
            "summary_rows": len(spans),
            "summary_rows_available": int(summary_total),
            "covered_fraction_estimate": covered_fraction,
            "gaps": gaps[:100],
            "must_state_coverage": True,
            "ledger": {
                "l0": l0_ledger,
                "attention": dict(attention_status or {"overall": "unavailable"}),
                "semantic_snapshots": dict(
                    semantic_status or {"overall": "unavailable"}
                ),
            },
        }

    @staticmethod
    def _severity(timeline: Sequence[Mapping[str, Any]]) -> str:
        rank = {"info": 0, "low": 1, "normal": 2, "high": 3, "critical": 4}
        return max(
            (_text(item.get("severity"), 24) for item in timeline),
            key=lambda value: rank.get(value, 0),
            default="info",
        )

    @staticmethod
    def _title(
        timeline: Sequence[Mapping[str, Any]],
        anchor: Optional[Mapping[str, Any]],
    ) -> str:
        meaningful = [
            item
            for item in timeline
            if _text(item.get("source"), 32) != "cv_motion_interval"
        ]
        if meaningful:
            return _text(meaningful[0].get("label"), 200) or "Incident draft"
        if anchor is not None:
            return _event_label(_payload(anchor), "Incident draft")[:200]
        return "Incident draft"

    @staticmethod
    def _apex_timestamp(timeline: Sequence[Mapping[str, Any]], fallback: int) -> int:
        severity = {"info": 0, "low": 1, "normal": 2, "high": 3, "critical": 4}
        if not timeline:
            return int(fallback)
        apex = max(
            timeline,
            key=lambda item: (
                severity.get(_text(item.get("severity"), 24), 0),
                _int(item.get("timestamp_ms")),
            ),
        )
        return _int(apex.get("timestamp_ms"), fallback)


def incident_report_markdown(incident: Mapping[str, Any]) -> str:
    bounds = _mapping(incident.get("time_bounds"))
    synopsis = _mapping(incident.get("synopsis"))
    summary = _text(
        incident.get("summary") or synopsis.get("description"),
        4000,
    )
    homeostasis = _mapping(incident.get("homeostasis") or synopsis.get("homeostasis"))
    follow_result = _mapping(incident.get("follow_result") or synopsis.get("follow_result"))
    temporal = _mapping(incident.get("temporal_memory"))
    lines = [
        f"# EVA incident report — {_text(incident.get('title'), 300) or 'Untitled incident'}",
        "",
        f"- Incident ID: `{_text(incident.get('id'), 120) or 'draft'}`",
        f"- State: `{_text(incident.get('state'), 40) or 'draft'}`",
        f"- Perception: `{_text(incident.get('perception_state'), 40) or 'unknown'}`",
        f"- Risk: `{_text(incident.get('risk_state'), 40) or 'unknown'}`",
        f"- Case: `{_text(incident.get('case_state'), 40) or 'unknown'}`",
        f"- Attention: `{_text(incident.get('attention_state'), 40) or 'unknown'}`",
        f"- Severity: `{_text(incident.get('severity'), 40) or 'info'}`",
        f"- Channels: {', '.join(str(item) for item in _sequence(incident.get('channel_ids')))}",
        f"- Observed start: {_timestamp_label(bounds.get('observed_start_ms')) or 'unknown'}",
        f"- Apex: {_timestamp_label(bounds.get('apex_ms')) or 'unknown'}",
        f"- Observed end: {_timestamp_label(bounds.get('observed_end_ms')) or 'open/unknown'}",
        "",
        "## Operator synopsis",
        "",
        summary or "No grounded visual synopsis is available; operator review is required.",
        "",
        "## Key moments",
        "",
        "| Time (UTC) | Key | Observation | Source | Confidence |",
        "|---|---|---|---|---|",
    ]
    key_moments = _sequence(incident.get("key_moments"))
    source_moments = key_moments or _sequence(incident.get("timeline"))
    for raw in source_moments[:12]:
        if not isinstance(raw, Mapping):
            continue
        cell = lambda value: _text(value, 600).replace("|", "\\|").replace("\n", " ")
        lines.append(
            "| "
            + " | ".join(
                (
                    cell(_timestamp_label(raw.get("timestamp_ms"))),
                    cell(raw.get("semantic_key")),
                    cell(raw.get("label")),
                    cell(raw.get("source")),
                    cell(raw.get("confidence")),
                )
            )
            + " |"
        )
    if not any(isinstance(raw, Mapping) for raw in source_moments[:12]):
        lines.append("| n/a | n/a | No grounded semantic milestone recovered | n/a | low |")

    lines.extend(["", "## Homeostatic attention", ""])
    lines.append(
        "- Motion: "
        f"mean p95={_float(homeostasis.get('motion_p95_mean')):.4f}, "
        f"max p95={_float(homeostasis.get('motion_p95_max')):.4f}; "
        f"bursts={_int(homeostasis.get('burst_count'))}."
    )
    lines.append(
        "- Activity: "
        f"mean={_float(homeostasis.get('activity_x_mean')):.2f}x, "
        f"max={_float(homeostasis.get('activity_x_max')):.2f}x; "
        f"elevated={_int(homeostasis.get('elevated_duration_ms'))} ms."
    )
    lines.append(
        "- Probes: "
        f"{_int(homeostasis.get('probe_hits'))} hits / "
        f"{_int(homeostasis.get('probe_samples'))} samples across "
        f"{_int(homeostasis.get('probe_count'))} probes."
    )
    if follow_result:
        lines.extend(
            [
                "",
                "## Follow outcome",
                "",
                f"- Outcome: `{_text(follow_result.get('outcome'), 80) or 'inconclusive'}`",
                f"- Observations: {_int(follow_result.get('observation_count'))}",
                f"- Result: {_text(follow_result.get('description'), 1000) or 'No grounded conclusion.'}",
            ]
        )

    episodes = [item for item in _sequence(temporal.get("episodes")) if isinstance(item, Mapping)]
    series_links = [item for item in _sequence(temporal.get("series_links")) if isinstance(item, Mapping)]
    lifecycle = [item for item in _sequence(temporal.get("lifecycle_history")) if isinstance(item, Mapping)]
    lines.extend(["", "## Temporal memory", ""])
    lines.append(
        f"Episodes: {int(_int(temporal.get('episode_total'), len(episodes)))}; "
        f"series links: {len(series_links)}; "
        f"lifecycle transitions: {int(_int(temporal.get('transition_total'), len(lifecycle)))}."
    )
    for episode in episodes[:16]:
        start = _timestamp_label(episode.get("observed_start_ms") or episode.get("possible_start_ms")) or "unknown"
        end = _timestamp_label(episode.get("observed_end_ms") or episode.get("possible_end_ms")) or "open"
        lines.append(
            f"- Episode `{_text(episode.get('id'), 120)}`: {start} → {end}; "
            f"track `{_text(episode.get('semantic_key'), 160) or 'unverified'}`; "
            f"disposition `{_text(episode.get('scale_disposition'), 80) or 'unclassified_keep'}`."
        )
    for relation in series_links[:16]:
        lines.append(
            f"- Series `{_text(relation.get('relation_state'), 40) or 'candidate'}`: "
            f"related incident `{_text(relation.get('related_incident_id'), 120)}`; "
            f"track `{_text(relation.get('semantic_key'), 160)}`; "
            f"gap {_int(relation.get('gap_ms'))} ms."
        )
    if lifecycle:
        lines.extend(["", "### Lifecycle history", ""])
        for transition in lifecycle[-24:]:
            lines.append(
                f"- {_timestamp_label(transition.get('transitioned_at_ms')) or 'time unknown'} | "
                f"`{_text(transition.get('axis'), 40)}`: "
                f"`{_text(transition.get('from_state'), 40) or 'unset'}` → "
                f"`{_text(transition.get('to_state'), 40) or 'unknown'}` | "
                f"{_text(transition.get('reason'), 500) or 'no reason recorded'}"
            )
    lines.extend(["", "## Coverage and uncertainty", ""])
    coverage = _mapping(incident.get("coverage"))
    lines.append(
        f"Coverage: **{_text(coverage.get('status'), 40) or 'unknown'}**; "
        f"summary rows: {_int(coverage.get('summary_rows'))}."
    )
    for item in _sequence(incident.get("uncertainties")):
        lines.append(f"- {_text(item, 1000)}")
    lines.extend(
        [
            "",
            "_P/N/M, probes and motion values in this report are attention signals, not visual ground truth._",
            "",
        ]
    )
    return "\n".join(lines)


def incident_report_xml(incident: Mapping[str, Any]) -> bytes:
    root = ET.Element("evaIncidentReport")
    root.set("groundTruthStatus", "operator_review_required")
    for key in (
        "id",
        "state",
        "perception_state",
        "risk_state",
        "case_state",
        "attention_state",
        "title",
        "severity",
    ):
        node = ET.SubElement(root, key)
        node.text = _text(incident.get(key), 2000)
    channels = ET.SubElement(root, "channels")
    for channel_id in _sequence(incident.get("channel_ids")):
        node = ET.SubElement(channels, "channel")
        node.set("id", str(_int(channel_id)))
    bounds_node = ET.SubElement(root, "timeBounds")
    for key, value in _mapping(incident.get("time_bounds")).items():
        node = ET.SubElement(bounds_node, str(key))
        node.text = str(value or "")
    synopsis_node = ET.SubElement(root, "operatorSynopsis")
    synopsis_node.text = _text(
        incident.get("summary") or _mapping(incident.get("synopsis")).get("description"),
        8000,
    )
    homeostasis_node = ET.SubElement(root, "homeostaticAttention")
    for key, value in _mapping(incident.get("homeostasis")).items():
        node = ET.SubElement(homeostasis_node, str(key))
        node.text = str(value if value is not None else "")
    follow = _mapping(incident.get("follow_result"))
    if follow:
        follow_node = ET.SubElement(root, "followOutcome")
        for key in ("outcome", "description", "observation_count", "started_at_ms", "ended_at_ms"):
            node = ET.SubElement(follow_node, key)
            node.text = str(follow.get(key) if follow.get(key) is not None else "")
    timeline_node = ET.SubElement(root, "timeline")
    for raw in _sequence(incident.get("timeline")):
        if not isinstance(raw, Mapping):
            continue
        item = ET.SubElement(timeline_node, "event")
        for key in (
            "timestamp_ms",
            "semantic_key",
            "label",
            "severity",
            "confidence",
            "source",
            "detection_id",
        ):
            node = ET.SubElement(item, key)
            node.text = str(raw.get(key) or "")
    evidence_node = ET.SubElement(root, "evidence")
    for raw in _sequence(incident.get("evidence")):
        if not isinstance(raw, Mapping):
            continue
        item = ET.SubElement(evidence_node, "reference")
        for key, value in raw.items():
            item.set(str(key), str(value or ""))
    coverage_node = ET.SubElement(root, "coverage")
    coverage_node.text = str(dict(_mapping(incident.get("coverage"))))
    uncertainties = ET.SubElement(root, "uncertainties")
    for value in _sequence(incident.get("uncertainties")):
        node = ET.SubElement(uncertainties, "uncertainty")
        node.text = _text(value, 2000)
    temporal = _mapping(incident.get("temporal_memory"))
    temporal_node = ET.SubElement(root, "temporalMemory")
    temporal_node.set("episodeTotal", str(_int(temporal.get("episode_total"))))
    temporal_node.set("transitionTotal", str(_int(temporal.get("transition_total"))))
    episodes_node = ET.SubElement(temporal_node, "episodes")
    for raw in _sequence(temporal.get("episodes"))[:16]:
        if not isinstance(raw, Mapping):
            continue
        item = ET.SubElement(episodes_node, "episode")
        for key in (
            "id", "perception_state", "semantic_key", "possible_start_ms",
            "observed_start_ms", "observed_end_ms", "possible_end_ms",
            "scale_disposition", "evidence_count",
        ):
            node = ET.SubElement(item, key)
            node.text = str(raw.get(key) if raw.get(key) is not None else "")
    series_node = ET.SubElement(temporal_node, "seriesLinks")
    for raw in _sequence(temporal.get("series_links"))[:16]:
        if not isinstance(raw, Mapping):
            continue
        item = ET.SubElement(series_node, "seriesLink")
        for key in (
            "relation_id", "relation_state", "related_incident_id", "semantic_key",
            "series_key", "gap_ms", "confidence",
        ):
            node = ET.SubElement(item, key)
            node.text = str(raw.get(key) if raw.get(key) is not None else "")
    lifecycle_node = ET.SubElement(temporal_node, "lifecycleHistory")
    for raw in _sequence(temporal.get("lifecycle_history"))[-24:]:
        if not isinstance(raw, Mapping):
            continue
        item = ET.SubElement(lifecycle_node, "transition")
        for key in (
            "id", "axis", "from_state", "to_state", "incident_revision",
            "transitioned_at_ms", "reason", "source_kind",
        ):
            node = ET.SubElement(item, key)
            node.text = str(raw.get(key) if raw.get(key) is not None else "")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


__all__ = [
    "IncidentDraftAssembler",
    "IncidentDraftRequest",
    "incident_report_markdown",
    "incident_report_xml",
]
