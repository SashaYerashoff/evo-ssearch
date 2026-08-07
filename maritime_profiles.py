"""Operator-reviewable maritime templates for Protocol Deploy.

The templates narrow attention; they are not vessel registries, collision
detectors, or proof of intent.  All starter probes are installed in shadow
mode and must earn calibrated authority from independent SigLIP2 snapshots.
"""

from __future__ import annotations

import copy
from typing import Any


MARITIME_CHANNEL_ROLES = frozenset(
    {"maritime_gate", "maritime_coast", "maritime_mixed_ptz"}
)

MARITIME_L0_PROMPT = """You are the visual aggregation core of EVA AI, an intelligent security system. Your function is to turn the current snapshots and bounded homeostatic signals into grounded episode memory and alert candidates.

For this maritime channel, first establish camera coverage: distinguish a steady view from pan, tilt, zoom, preset cut, settling, obstruction, or signal loss. Camera-global motion is not vessel motion. Treat VECTOR_SIGNALS_JSON.camera_scene as routing metadata, never as visual proof. While a PTZ view is moving or unconfirmed, do not assert that a vessel, zone, or shoreline object is absent; report that it was not observed because the relevant view was unavailable.

When coverage is usable, describe only visible facts. Use coarse operational classes when supported: large commercial vessel, passenger vessel, fishing/work boat, sailing vessel, small motor craft, personal watercraft, or unknown vessel. Preserve uncertainty about class, scale, distance, identity, intent, and collision risk. Track episode continuity across batches, including gate passages, stopping or loitering in a navigational area, converging paths, small craft near a large vessel, shore approaches, distress-like visible states, and unexpected coastline activity. Alert Criteria are supplied separately and own notification severity.

Choose a cover that best explains the episode or coverage change. Prefer a clear apex or transition frame with useful before/after context; a maximum-motion or PTZ-blurred frame is not automatically a good cover. The backend appends the unified BATCH_STATE_JSON contract last."""

MARITIME_ROLLUP_PROMPTS = {
    "L1": """Aggregate grounded maritime L0 batches into short episodes. Preserve vessel passages, changes of course or relative proximity, shore approaches, distress-like observations, camera/preset coverage, and unresolved continuity. Never convert a PTZ coverage gap into a negative observation. Separate observed event time from camera-motion or unavailable time.""",
    "L2": """Aggregate maritime L1 episodes into an hourly operational account. Distinguish recurring traffic/routine from deviations, retain coarse vessel class and direction only when grounded, and preserve close-approach, gate obstruction, shore contact, distress, and camera-health episodes. Report uncovered presets/zones as not observed, not quiet.""",
    "L3": """Produce an eight-hour maritime consolidation useful to an operator asking what materially happened. Prefer a chronological account of meaningful vessel passages, interactions, coastline events, alert outcomes, operator feedback, and coverage limitations over a generic routine summary. Audit false positives and probe drift as proposals only. When a preset or zone lacked coverage, say not observed; a vessel or event can be absent from the report only when relevant coverage was actually available.""",
}


_ROLE_TEMPLATES: dict[str, dict[str, Any]] = {
    "maritime_gate": {
        "label": "Port gate / fairway",
        "expected_routine": "Vessels may enter or leave through the visible port gate and fairway; traffic frequency and permitted routes must be confirmed by the operator.",
        "unexpected_severity": "normal",
        "novelty_sensitivity": "high",
        "alerts": [
            {
                "name": "Vessel gate passage",
                "description": "A vessel visibly crosses the configured port-gate line; count direction separately when the view supports it.",
                "severity": "info",
                "positive_query": "vessel crossing the visible port entrance or fairway gate",
                "contrast_query": "open water and clear port entrance",
                "counter_mode": "count_transitions",
                "positive_label": "crossing",
                "negative_label": "clear_gate",
                "count_transition": "negative_to_positive",
            },
            {
                "name": "Fairway obstruction",
                "description": "A vessel is stopped or lingering in the visible gate or fairway instead of making ordinary passage.",
                "severity": "high",
                "positive_query": "vessel stopped or lingering in the port gate fairway",
                "contrast_query": "vessel making ordinary passage through open fairway",
                "counter_mode": "measure_duration",
                "positive_label": "obstructing",
                "negative_label": "passing",
                "duration_state": "positive",
                "alert_after_sec": 45,
            },
            {
                "name": "Close approach",
                "description": "Two vessels are visibly on converging or unusually close paths in the gate/fairway; VLM must verify geometry before alerting.",
                "severity": "critical",
                "positive_query": "two vessels very close together on converging paths",
                "contrast_query": "vessels separated on ordinary parallel or diverging paths",
                "counter_mode": "count_transitions",
                "positive_label": "close_approach",
                "negative_label": "separated",
            },
            {
                "name": "Small craft near large vessel",
                "description": "A small craft, sailing vessel, or personal watercraft is visibly operating unusually close to a large commercial vessel.",
                "severity": "high",
                "positive_query": "small boat or personal watercraft very close to a large ship",
                "contrast_query": "large ship with small craft safely separated",
            },
        ],
    },
    "maritime_coast": {
        "label": "Coastline / beach",
        "expected_routine": "Open coastline, beach, nearshore water, and routine distant vessel traffic; permitted shore use must be confirmed by the operator.",
        "unexpected_severity": "normal",
        "novelty_sensitivity": "high",
        "alerts": [
            {
                "name": "Small craft shore approach",
                "description": "A small craft visibly approaches, lands on, or departs from an unprepared shoreline or beach.",
                "severity": "high",
                "positive_query": "small boat landing on or departing from a beach shoreline",
                "contrast_query": "open beach and nearshore water with distant vessels",
                "counter_mode": "count_and_duration",
                "positive_label": "shore_contact",
                "negative_label": "clear_shore",
            },
            {
                "name": "Nearshore loitering",
                "description": "A small craft remains close to the visible shoreline rather than making ordinary transit.",
                "severity": "normal",
                "positive_query": "small boat lingering close to shoreline",
                "contrast_query": "small boat travelling steadily along or away from coast",
                "counter_mode": "measure_duration",
                "positive_label": "loitering",
                "negative_label": "transiting",
                "duration_state": "positive",
                "alert_after_sec": 90,
            },
            {
                "name": "Visible distress at sea",
                "description": "Visible capsizing, person in water, fire, heavy smoke, emergency signalling, or vessel in obvious distress.",
                "severity": "critical",
                "positive_query": "maritime distress with capsized boat person in water fire smoke or emergency signal",
                "contrast_query": "upright vessel making ordinary safe passage",
            },
            {
                "name": "Unexpected coastline activity",
                "description": "A concrete visible shoreline activity materially outside the operator-confirmed baseline; novelty is reviewed by VLM before alerting.",
                "severity": "normal",
                "positive_query": "unusual active gathering vehicle or cargo transfer at shoreline",
                "contrast_query": "ordinary open shoreline with routine sparse activity",
            },
        ],
    },
    "maritime_mixed_ptz": {
        "label": "Mixed PTZ maritime tour",
        "expected_routine": "The camera may move through operator-confirmed port, sea-gate, fairway, and coastline presets; each preset has separate coverage and baseline.",
        "unexpected_severity": "normal",
        "novelty_sensitivity": "high",
        "alerts": [
            {
                "name": "Vessel interaction",
                "description": "Two or more vessels show a visible close, converging, crossing, or erratic interaction in the current confirmed view.",
                "severity": "high",
                "positive_query": "multiple vessels on close converging or crossing paths",
                "contrast_query": "vessels widely separated on ordinary paths",
            },
            {
                "name": "Navigational area occupied",
                "description": "A vessel is visibly stopped or lingering in a port gate, fairway, or other operator-confirmed navigation zone.",
                "severity": "high",
                "positive_query": "vessel stopped or lingering in marked navigation area",
                "contrast_query": "clear navigation area or vessel making ordinary passage",
                "counter_mode": "measure_duration",
                "positive_label": "occupied",
                "negative_label": "clear",
                "duration_state": "positive",
                "alert_after_sec": 45,
            },
            {
                "name": "Shore approach",
                "description": "A small craft visibly approaches or contacts a beach or unprepared coastline in the current confirmed preset.",
                "severity": "high",
                "positive_query": "small craft approaching or contacting beach shoreline",
                "contrast_query": "small craft offshore making ordinary passage",
                "counter_mode": "count_transitions",
                "positive_label": "approach",
                "negative_label": "offshore",
            },
            {
                "name": "Visible maritime distress",
                "description": "Visible capsizing, person in water, fire, heavy smoke, emergency signalling, or an obviously disabled vessel.",
                "severity": "critical",
                "positive_query": "maritime distress capsized vessel person in water fire smoke or emergency signal",
                "contrast_query": "upright vessel making ordinary safe passage",
            },
        ],
    },
}


def maritime_requirement(role: str, channel_id: int) -> dict[str, Any]:
    normalized = str(role or "").strip().lower()
    if normalized not in MARITIME_CHANNEL_ROLES:
        raise ValueError(f"unsupported maritime channel role: {role}")
    template = copy.deepcopy(_ROLE_TEMPLATES[normalized])
    return {
        "name": template.pop("label"),
        "channel_ids": [int(channel_id)],
        **template,
    }


def maritime_role_label(role: str) -> str:
    template = _ROLE_TEMPLATES.get(str(role or "").strip().lower())
    return str((template or {}).get("label") or role)
