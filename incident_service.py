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
        connected = _connected_candidate_rows(
            summaries,
            anchor_ms=anchor_ms,
            max_gap_ms=INCIDENT_CONTINUITY_GAP_MS,
        )
        observed_start = min(
            (_int(row.get("batch_start_ms"), anchor_ms) for row in connected),
            default=anchor_ms,
        )
        observed_end = max(
            (_int(row.get("batch_end_ms"), anchor_ms) for row in connected),
            default=anchor_ms,
        )
        drill_since = max(since_ms, observed_start - INCIDENT_CONTINUITY_GAP_MS)
        drill_until = min(until_ms, observed_end + INCIDENT_CONTINUITY_GAP_MS)

        alerts, alert_total = self.detections.list_detections(
            channel_id=channel_id,
            source="vlm_alert",
            since_ms=drill_since,
            until_ms=drill_until,
            limit=500,
            offset=0,
            include_thumbnail=False,
        )
        raw_summaries, _raw_summary_total = self.detections.list_detections(
            channel_id=channel_id,
            source="vlm_summary",
            since_ms=drill_since,
            until_ms=drill_until,
            limit=500,
            offset=0,
            include_thumbnail=False,
        )
        structured_summaries = self._dedupe_summary_detections(
            raw_summaries,
            connected,
        )
        telemetry = self._attention_rows(channel_id, drill_since, drill_until)
        timeline = self._timeline(
            structured_summaries,
            alerts,
            telemetry["intervals"],
        )
        evidence = self._evidence(connected, alerts, telemetry["links"])
        qualia = self._qualia_digest(telemetry["scores"], telemetry["intervals"])
        coverage = self._coverage(
            summaries,
            since_ms=drill_since,
            until_ms=drill_until,
            summary_total=summary_total,
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
                "apex_ms": self._apex_timestamp(timeline, anchor_ms),
                "observed_end_ms": observed_end,
                "possible_end_ms": drill_until,
            },
            "timeline": timeline,
            "evidence": evidence,
            "qualia_digest": qualia,
            "coverage": coverage,
            "uncertainties": uncertainties,
            "focus_lease": None,
            "provenance": {
                "summary_rows_considered": len(summaries),
                "connected_summary_rows": len(connected),
                "structured_summary_rows": len(structured_summaries),
                "alert_rows": len(alerts),
                "alert_rows_available": int(alert_total),
                "attention_store": bool(self.attention is not None),
            },
        }

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

    def _attention_rows(self, channel_id: int, since_ms: int, until_ms: int) -> dict[str, list[dict[str, Any]]]:
        if self.attention is None:
            return {"intervals": [], "scores": [], "links": []}
        try:
            return {
                "intervals": self.attention.query_intervals(
                    channel_id=channel_id,
                    start_ms=since_ms,
                    end_ms=until_ms,
                    limit=2000,
                ),
                "scores": self.attention.query_probe_scores(
                    channel_id=channel_id,
                    start_ms=since_ms,
                    end_ms=until_ms,
                    limit=5000,
                ),
                "links": self.attention.query_evidence_links(
                    channel_id=channel_id,
                    start_ms=since_ms,
                    end_ms=until_ms,
                    limit=2000,
                ),
            }
        except Exception:
            return {"intervals": [], "scores": [], "links": []}

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
                    items.append(
                        {
                            "timestamp_ms": _event_timestamp(raw, fallback_ts),
                            "semantic_key": _event_key(raw, fallback_key),
                            "label": _event_label(raw, fallback_key.replace("_", " ").title()),
                            "severity": _event_severity(raw),
                            "confidence": _text(raw.get("confidence"), 24) or "unknown",
                            "source": source,
                            "summary_detection_id": _int(row.get("id")) or None,
                        }
                    )
        for row in alerts:
            payload = _payload(row)
            items.append(
                {
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
            )
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
    def _evidence(
        summaries: Sequence[Mapping[str, Any]],
        alerts: Sequence[Mapping[str, Any]],
        links: Sequence[Mapping[str, Any]],
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
                    "role": "summary_anchor",
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
        evidence.sort(key=lambda item: _int(item.get("timestamp_ms")))
        return evidence[:500]

    @staticmethod
    def _qualia_digest(
        scores: Sequence[Mapping[str, Any]],
        intervals: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        by_probe: dict[str, dict[str, Any]] = {}
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
        motion_values = [
            _float(row.get("motion_p95"))
            for row in intervals
            if _text(row.get("state"), 24) in {"motion", "mixed"}
        ]
        return {
            "ground_truth": False,
            "interpretation": "attention signals only",
            "probe_count": len(by_probe),
            "probes": sorted(by_probe.values(), key=lambda item: (-item["hits"], item["probe_id"]))[:64],
            "motion_interval_count": len(motion_values),
            "motion_p95_max": max(motion_values, default=0.0),
            "motion_p95_mean": statistics.fmean(motion_values) if motion_values else 0.0,
        }

    @staticmethod
    def _coverage(
        summaries: Sequence[Mapping[str, Any]],
        *,
        since_ms: int,
        until_ms: int,
        summary_total: int,
    ) -> dict[str, Any]:
        spans = sorted(
            (
                _int(row.get("batch_start_ms")),
                _int(row.get("batch_end_ms")),
            )
            for row in summaries
            if _int(row.get("batch_end_ms")) >= _int(row.get("batch_start_ms")) > 0
        )
        gaps: list[dict[str, int]] = []
        cursor = int(since_ms)
        covered_ms = 0
        for start, end in spans:
            clipped_start = max(since_ms, start)
            clipped_end = min(until_ms, end)
            if clipped_end < clipped_start:
                continue
            if clipped_start - cursor > INCIDENT_CONTINUITY_GAP_MS:
                gaps.append({"start_ms": cursor, "end_ms": clipped_start})
            if clipped_end > cursor:
                covered_ms += max(0, clipped_end - max(cursor, clipped_start))
                cursor = clipped_end
        if until_ms - cursor > INCIDENT_CONTINUITY_GAP_MS:
            gaps.append({"start_ms": cursor, "end_ms": until_ms})
        window_ms = max(1, until_ms - since_ms)
        return {
            "status": "no_data" if not spans else "partial" if gaps else "covered",
            "window_start_ms": since_ms,
            "window_end_ms": until_ms,
            "summary_rows": len(spans),
            "summary_rows_available": int(summary_total),
            "covered_fraction_estimate": round(min(1.0, covered_ms / window_ms), 4),
            "gaps": gaps[:100],
            "must_state_coverage": True,
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
    lines = [
        f"# EVA incident report — {_text(incident.get('title'), 300) or 'Untitled incident'}",
        "",
        f"- Incident ID: `{_text(incident.get('id'), 120) or 'draft'}`",
        f"- State: `{_text(incident.get('state'), 40) or 'draft'}`",
        f"- Severity: `{_text(incident.get('severity'), 40) or 'info'}`",
        f"- Channels: {', '.join(str(item) for item in _sequence(incident.get('channel_ids')))}",
        f"- Observed start: {_timestamp_label(bounds.get('observed_start_ms')) or 'unknown'}",
        f"- Apex: {_timestamp_label(bounds.get('apex_ms')) or 'unknown'}",
        f"- Observed end: {_timestamp_label(bounds.get('observed_end_ms')) or 'open/unknown'}",
        "",
        "## Timeline",
        "",
        "| Time (UTC) | Key | Observation | Source | Confidence |",
        "|---|---|---|---|---|",
    ]
    for raw in _sequence(incident.get("timeline")):
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
    for key in ("id", "state", "title", "severity"):
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
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


__all__ = [
    "IncidentDraftAssembler",
    "IncidentDraftRequest",
    "incident_report_markdown",
    "incident_report_xml",
]
