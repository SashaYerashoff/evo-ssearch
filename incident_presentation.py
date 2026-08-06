"""Bounded, human-facing incident synthesis and Follow classification.

The incident ledger keeps every grounded reference.  This module deliberately
does not: it turns repeated CV intervals and L0 heartbeats into an operator
synopsis without treating attention signals as visual proof.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


_SIGNAL_SOURCES = {"cv_motion_interval", "homeostasis", "probe_signal"}
_GENERIC_KEYS = {"", "event", "transition", "alert", "vlm_alert", "motion_peak"}
_RESOLVED_STATES = {"resolved", "finished", "ended", "closed", "absent", "returned"}
_ACTION_WORDS = (
    "enters", "entered", "exits", "exited", "leaves", "sits", "sat",
    "stands", "stood", "falls", "fell", "crosses", "crossed", "approaches",
    "approached", "stops", "stopped", "collides", "collided", "drifts",
    "drifted", "appears", "appeared", "disappears", "disappeared", "moves",
    "moved", "parks", "parked", "returns", "returned", "waves", "waved",
)
_SUBJECTS = (
    "small craft", "sailing boat", "motor boat", "watercraft", "person",
    "vehicle", "vessel", "ship", "boat", "yacht", "cat", "animal", "object",
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else ()


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _plain_text(value: Any, limit: int = 1000) -> str:
    text = str(value or "").replace("\x00", " ")
    text = re.sub(r"`{1,3}", "", text)
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n-–—:;,.#")
    return text[:limit]


def _episode_text(value: Any) -> str:
    raw = str(value or "")
    match = re.search(
        r"(?:#{1,4}\s*)?episode\s+update\s*[:\-]?\s*(.*?)"
        r"(?=(?:#{1,4}\s*)?(?:routine(?:\s+and\s+deviations)?|worth\s+to\s+remember|batch_state_json)\b|$)",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return _plain_text(match.group(1) if match else raw, 800)


def _timeline(incident: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = incident.get("timeline_refs") or incident.get("timeline") or incident.get("events") or []
    return [item for item in _sequence(raw) if isinstance(item, Mapping)]


def semantic_timeline(incident: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return deduplicated operator milestones, excluding signal-only rows."""

    ranked: list[tuple[int, int, dict[str, Any]]] = []
    source_rank = {"vlm_alert": 0, "vlm_structured_alert": 1, "state_transition": 2, "batch_state": 3}
    seen: set[tuple[str, str]] = set()
    for raw in _timeline(incident):
        source = str(raw.get("source") or "").strip().lower()
        confidence = str(raw.get("confidence") or "").strip().lower()
        if source in _SIGNAL_SOURCES or confidence == "signal_only":
            continue
        label = _episode_text(raw.get("label") or raw.get("summary") or raw.get("description"))
        if not label:
            continue
        lowered = label.lower()
        if lowered.startswith(("no event", "no new event", "the scene is static", "static scene")):
            continue
        key = str(raw.get("semantic_key") or raw.get("key") or "").strip().lower()
        dedupe = (key, lowered[:240])
        if dedupe in seen:
            continue
        seen.add(dedupe)
        timestamp_ms = _integer(raw.get("timestamp_ms") or raw.get("occurred_at_ms")) or 0
        ranked.append(
            (
                source_rank.get(source, 4),
                timestamp_ms,
                {
                    "timestamp_ms": timestamp_ms or None,
                    "semantic_key": key,
                    "label": label[:360],
                    "state": str(raw.get("state") or raw.get("to_state") or "").strip().lower(),
                    "severity": str(raw.get("severity") or "info").strip().lower(),
                    "source": source or "semantic_event",
                },
            )
        )
    # Preserve chronology after using source rank only to choose among repeated
    # prose emitted for the same batch.
    ranked.sort(key=lambda item: (item[1], item[0], item[2]["label"]))
    return [item[2] for item in ranked[:8]]


def _title_from_text(text: str) -> str:
    cleaned = _episode_text(text)
    lowered = cleaned.lower()
    subject = next((value for value in _SUBJECTS if re.search(rf"\b{re.escape(value)}\b", lowered)), "")
    actions = [
        (match.start(), word)
        for word in _ACTION_WORDS
        for match in [re.search(rf"\b{re.escape(word)}\b", lowered)]
        if match is not None
    ]
    actions.sort()
    distinct_actions: list[str] = []
    for _, action in actions:
        if action not in distinct_actions:
            distinct_actions.append(action)
    if subject and len(distinct_actions) >= 2:
        return f"{subject.title()} {distinct_actions[0]} and {distinct_actions[-1]}"[:80]
    words = re.findall(r"[\w'-]+", cleaned, flags=re.UNICODE)
    if words and words[0].lower() in {"a", "an", "the"}:
        words = words[1:]
    if subject and distinct_actions:
        subject_words = subject.title().split()
        action = distinct_actions[0]
        tail: list[str] = []
        action_index = next((index for index, word in enumerate(words) if word.lower() == action), -1)
        if action_index >= 0:
            tail = [word for word in words[action_index + 1:] if word.lower() not in {"the", "a", "an"}][:2]
        candidate = [*subject_words, action, *tail]
    else:
        candidate = words[:5]
    candidate = candidate[:5]
    if len(candidate) == 2:
        candidate.extend(["monitored", "area"])
    elif len(candidate) < 3:
        candidate.extend(["requires", "review"][: 3 - len(candidate)])
    return " ".join(candidate).strip().capitalize()[:80] or "Incident requires review"


def human_incident_title(incident: Mapping[str, Any]) -> str:
    milestones = semantic_timeline(incident)
    if milestones:
        return _title_from_text(str(milestones[0].get("label") or ""))
    stored = _plain_text(incident.get("title"), 400)
    if stored and not re.search(r"scene\s+description|episode\s+update", stored, re.IGNORECASE):
        return _title_from_text(stored)
    return "Unverified motion episode"


def homeostasis_digest(incident: Mapping[str, Any]) -> dict[str, Any]:
    refs = [item for item in _sequence(incident.get("qualia_refs")) if isinstance(item, Mapping)]
    raw = dict(refs[0]) if refs else dict(_mapping(incident.get("qualia_digest")))
    probes = [item for item in _sequence(raw.get("probes")) if isinstance(item, Mapping)]
    profile = dict(_mapping(raw.get("motion_profile")))
    observed_start_ms = _integer(incident.get("observed_start_ms"))
    observed_end_ms = _integer(incident.get("observed_end_ms"))
    legacy_motion: list[tuple[int, float, float]] = []
    for item in _timeline(incident):
        if str(item.get("source") or "").strip().lower() != "cv_motion_interval":
            continue
        timestamp_ms = _integer(item.get("timestamp_ms")) or 0
        if observed_start_ms is not None and timestamp_ms < observed_start_ms:
            continue
        if observed_end_ms is not None and timestamp_ms > observed_end_ms:
            continue
        label = str(item.get("label") or "")
        activity_match = re.search(r"activity\s*=\s*(-?\d+(?:\.\d+)?)", label, re.IGNORECASE)
        motion_match = re.search(r"p95\s*=\s*(-?\d+(?:\.\d+)?)", label, re.IGNORECASE)
        if activity_match or motion_match:
            legacy_motion.append(
                (
                    timestamp_ms,
                    float(activity_match.group(1)) if activity_match else 0.0,
                    float(motion_match.group(1)) if motion_match else 0.0,
                )
            )
    legacy_activity = [item[1] for item in legacy_motion]
    legacy_p95 = [item[2] for item in legacy_motion]
    legacy_elevated = [item for item in legacy_motion if item[1] >= 3.0]
    legacy_bursts = 0
    previous_timestamp = -1
    for timestamp_ms, _activity, _motion in legacy_elevated:
        if legacy_bursts == 0 or timestamp_ms - previous_timestamp > 1_500:
            legacy_bursts += 1
        previous_timestamp = timestamp_ms
    legacy_apex = max(legacy_motion, key=lambda item: item[1], default=(0, 0.0, 0.0))
    probe_hits = sum(max(0, _integer(item.get("hits")) or 0) for item in probes)
    probe_samples = sum(max(0, _integer(item.get("samples")) or 0) for item in probes)
    return {
        "interpretation": "attention signals, not visual proof",
        "motion_interval_count": max(0, _integer(raw.get("motion_interval_count")) or 0),
        "motion_p95_mean": round(
            _number(raw.get("motion_p95_mean"))
            or (sum(legacy_p95) / len(legacy_p95) if legacy_p95 else 0.0),
            4,
        ),
        "motion_p95_max": round(
            _number(raw.get("motion_p95_max")) or max(legacy_p95, default=0.0),
            4,
        ),
        "activity_x_mean": round(
            _number(profile.get("activity_x_mean"))
            or (sum(legacy_activity) / len(legacy_activity) if legacy_activity else 0.0),
            2,
        ),
        "activity_x_max": round(
            _number(profile.get("activity_x_max")) or legacy_apex[1],
            2,
        ),
        "apex_at_ms": _integer(profile.get("apex_at_ms")) or legacy_apex[0] or None,
        "elevated_duration_ms": max(
            0,
            _integer(profile.get("elevated_duration_ms"))
            or len(legacy_elevated) * 1_000,
        ),
        "settling_ms": max(
            0,
            _integer(profile.get("settling_ms"))
            or (
                max((item[0] for item in legacy_elevated), default=legacy_apex[0]) - legacy_apex[0]
                if legacy_apex[0] else 0
            ),
        ),
        "burst_count": max(0, _integer(profile.get("burst_count")) or legacy_bursts),
        "probe_count": max(0, _integer(raw.get("probe_count")) or len(probes)),
        "probe_hits": probe_hits,
        "probe_samples": probe_samples,
    }


def build_incident_synopsis(incident: Mapping[str, Any]) -> dict[str, Any]:
    milestones = semantic_timeline(incident)
    report = _mapping(incident.get("report"))
    follow_result = _mapping(report.get("follow_result"))
    if milestones:
        sentences: list[str] = []
        for item in milestones[:3]:
            text = _plain_text(item.get("label"), 360)
            if text and text not in sentences:
                sentences.append(text.rstrip(".") + ".")
        description = " ".join(sentences)[:900]
    else:
        description = (
            "Attention signals changed during the selected interval, but EVA did not recover "
            "a grounded visual event. Operator review is required."
        )
    coverage = _mapping(incident.get("coverage"))
    coverage_status = str(coverage.get("status") or "unknown")
    uncertainty_count = len(_sequence(incident.get("uncertainties")))
    confidence = "high" if milestones and coverage_status == "covered" and not uncertainty_count else "medium" if milestones else "low"
    outcome = str(follow_result.get("outcome") or "").strip() or (
        "resolved" if str(incident.get("risk_state") or "") == "resolved" else
        "continuing" if str(incident.get("attention_state") or "") in {"follow", "critical"} else
        "awaiting_review"
    )
    return {
        "title": human_incident_title(incident),
        "description": description,
        "outcome": outcome,
        "confidence": confidence,
        "key_moments": milestones[:5],
        "homeostasis": homeostasis_digest(incident),
        "follow_result": dict(follow_result),
    }


def _semantic_keys(incident: Mapping[str, Any]) -> set[str]:
    keys = {
        str(item.get("semantic_key") or "").strip().lower()
        for item in semantic_timeline(incident)
    }
    return {key for key in keys if key not in _GENERIC_KEYS}


def classify_follow_heartbeat(incident: Mapping[str, Any], heartbeat: Mapping[str, Any]) -> dict[str, Any]:
    """Classify one L0 batch without inferring absence from omission."""

    if bool(heartbeat.get("coverage_gap")):
        return {"association": "coverage_gap", "perception_state": "not_observed", "matched_keys": []}
    batch_state = _mapping(heartbeat.get("batch_state"))
    raw_events = [
        item
        for field in ("events", "alerts", "observed_states")
        for item in _sequence(batch_state.get(field))
        if isinstance(item, Mapping)
    ]
    raw_events.extend(item for item in _sequence(heartbeat.get("state_transition_events")) if isinstance(item, Mapping))
    incident_keys = _semantic_keys(incident)
    event_keys = {
        str(item.get("semantic_key") or item.get("event_id") or item.get("key") or "").strip().lower()
        for item in raw_events
    }
    matched = sorted((incident_keys & event_keys) - _GENERIC_KEYS)
    routines = [item for item in _sequence(batch_state.get("routines")) if isinstance(item, Mapping)]
    routine_matches: set[str] = set()
    for routine in routines:
        if str(routine.get("state") or "").strip().lower() not in {"returned", "resolved", "ended"}:
            continue
        applies = {
            str(value or "").strip().lower()
            for value in _sequence(routine.get("applies_to_event_keys"))
        }
        routine_matches.update(incident_keys & applies)
    if routine_matches:
        return {"association": "resolved", "perception_state": "ended", "matched_keys": sorted(routine_matches)}
    if matched:
        matched_events = [
            item for item in raw_events
            if str(item.get("semantic_key") or item.get("event_id") or item.get("key") or "").strip().lower() in matched
        ]
        if any(str(item.get("state") or item.get("to_state") or "").strip().lower() in _RESOLVED_STATES for item in matched_events):
            return {"association": "resolved", "perception_state": "ended", "matched_keys": matched}
        return {"association": "supports", "perception_state": "observed", "matched_keys": matched}
    return {
        "association": "unrelated" if raw_events else "neutral",
        "perception_state": "unknown",
        "matched_keys": [],
    }


def build_follow_result(
    incident: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    *,
    ended_at_ms: int,
    stop_reason: str,
) -> dict[str, Any]:
    follow = _mapping(incident.get("follow_policy"))
    run_id = str(follow.get("run_id") or "").strip()
    relevant = [
        item for item in observations
        if not run_id or str(_mapping(item.get("source_ref")).get("follow_run_id") or "").strip() in {"", run_id}
    ]
    counts = Counter(
        str(_mapping(item.get("payload")).get("association") or "neutral").strip().lower()
        for item in relevant
    )
    homeostasis_rows = [
        _mapping(_mapping(item.get("payload")).get("homeostasis"))
        for item in relevant
    ]
    sample_count = sum(max(0, _integer(item.get("sample_count")) or 0) for item in homeostasis_rows)
    weighted_activity = sum(
        (_number(item.get("activity_x_mean")) or 0.0)
        * max(0, _integer(item.get("sample_count")) or 0)
        for item in homeostasis_rows
    )
    follow_homeostasis = {
        "sample_count": sample_count,
        "activity_x_max": round(
            max((_number(item.get("activity_x_max")) or 0.0 for item in homeostasis_rows), default=0.0),
            2,
        ),
        "activity_x_mean": round(weighted_activity / sample_count, 2) if sample_count else 0.0,
        "burst_count": sum(max(0, _integer(item.get("burst_count")) or 0) for item in homeostasis_rows),
    }
    if counts["resolved"]:
        outcome = "resolved"
        description = "A grounded return or resolution was observed before Follow ended."
    elif counts["supports"]:
        outcome = "continuing"
        description = "Follow confirmed continuation, but did not observe a grounded resolution."
    elif counts["coverage_gap"]:
        outcome = "inconclusive_coverage"
        description = "Follow ended without a conclusion because part of the scene was not observable."
    elif str(follow.get("relationship") or "") == "recurrence_watch":
        outcome = "recurrence_not_confirmed"
        description = "No grounded recurrence was confirmed during this Follow window; absence was not inferred."
    elif not relevant:
        outcome = "no_observations"
        description = "Follow ended before EVA received a usable L0 observation."
    else:
        outcome = "inconclusive"
        description = "Follow received L0 observations, but none established continuation or resolution."
    first_ms = min((_integer(item.get("observed_at_ms")) or ended_at_ms for item in relevant), default=ended_at_ms)
    last_ms = max((_integer(item.get("observed_at_ms")) or first_ms for item in relevant), default=first_ms)
    return {
        "run_id": run_id or None,
        "relationship": str(follow.get("relationship") or "continuation"),
        "started_at_ms": _integer(follow.get("started_at_ms")),
        "ended_at_ms": int(ended_at_ms),
        "stop_reason": str(stop_reason or "operator")[:80],
        "outcome": outcome,
        "description": description,
        "observation_count": len(relevant),
        "association_counts": dict(sorted(counts.items())),
        "homeostasis": follow_homeostasis,
        "first_observation_ms": first_ms if relevant else None,
        "last_observation_ms": last_ms if relevant else None,
    }


__all__ = [
    "build_follow_result",
    "build_incident_synopsis",
    "classify_follow_heartbeat",
    "homeostasis_digest",
    "human_incident_title",
    "semantic_timeline",
]
