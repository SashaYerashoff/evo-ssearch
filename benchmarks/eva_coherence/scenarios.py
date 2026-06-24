"""Deterministic fixtures for the dev-only EVA coherence benchmark harness.

The data is intentionally small and script-like: runners can import the plain
dictionaries directly, or pass them to the fakes in ``fakes.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


ZENBOOK_CHANNEL_ID = 112
DOG_CHANNEL_ID = 305

WINDOW_START_TS = 1_782_194_400.0  # 2026-06-23T06:00:00Z
WINDOW_END_TS = 1_782_201_600.0  # 2026-06-23T08:00:00Z
WINDOW_START_MS = int(WINDOW_START_TS * 1000)
WINDOW_END_MS = int(WINDOW_END_TS * 1000)

FIXTURE_RUN_ID = "eva-coherence-20260623T060000Z"
FAKE_THUMBNAIL = "RkFLRV9KUEVHX1RIVU1C"


@dataclass(frozen=True)
class ScenarioFixture:
    """Importable benchmark scenario descriptor."""

    slug: str
    title: str
    channel_ids: tuple[int, ...]
    from_ts: float
    to_ts: float
    prompt: str
    expected_behavior: str
    metadata: Mapping[str, Any]


def ts(minutes_from_start: int) -> float:
    return WINDOW_START_TS + float(minutes_from_start * 60)


def ms(minutes_from_start: int) -> int:
    return int(ts(minutes_from_start) * 1000)


CHANNELS: list[dict[str, Any]] = [
    {
        "id": ZENBOOK_CHANNEL_ID,
        "title": "Zenbook webcam",
        "name": "Zenbook webcam",
        "enabled": True,
        "status": "online",
        "site": "lab",
    },
    {
        "id": DOG_CHANNEL_ID,
        "title": "Clinic side door",
        "name": "Clinic side door",
        "enabled": True,
        "status": "online",
        "site": "clinic",
    },
]

_SWEEP_CHANNEL_TITLES = {
    201: "North gate",
    202: "Loading bay",
    203: "Reception",
    204: "Side yard",
    205: "Canteen",
    206: "Server hall",
    207: "Parking aisle",
    208: "Warehouse door",
    209: "Workshop",
}

CHANNELS.extend(
    {
        "id": channel_id,
        "title": title,
        "name": title,
        "enabled": True,
        "status": "online",
        "site": "multi-channel-sweep",
    }
    for channel_id, title in _SWEEP_CHANNEL_TITLES.items()
)

CHANNELS.append(
    {
        "id": 210,
        "title": "Quiet archive channel",
        "name": "Quiet archive channel",
        "enabled": True,
        "status": "online",
        "site": "multi-channel-sweep",
    }
)


ZENBOOK_ROLLUPS: dict[str, list[dict[str, Any]]] = {
    "L0": [
        {
            "level": "L0",
            "channel_id": ZENBOOK_CHANNEL_ID,
            "run_id": FIXTURE_RUN_ID,
            "created_at": ts(5),
            "window_start": ts(0),
            "window_end": ts(10),
            "summary": "Desk and empty chair visible on the Zenbook webcam; Orlandina is not visible.",
            "frame_count": 6,
            "alert_total": 0,
            "alert_counts": {"normal": 0},
        },
        {
            "level": "L0",
            "channel_id": ZENBOOK_CHANNEL_ID,
            "run_id": FIXTURE_RUN_ID,
            "created_at": ts(25),
            "window_start": ts(20),
            "window_end": ts(30),
            "summary": "Laptop lid, chair, and window reflection are stable; Orlandina remains absent.",
            "frame_count": 6,
            "alert_total": 0,
            "alert_counts": {"normal": 0},
        },
        {
            "level": "L0",
            "channel_id": ZENBOOK_CHANNEL_ID,
            "run_id": FIXTURE_RUN_ID,
            "created_at": ts(45),
            "window_start": ts(40),
            "window_end": ts(50),
            "summary": "A passerby crosses the background but no view supports Orlandina being present.",
            "frame_count": 5,
            "alert_total": 0,
            "alert_counts": {"normal": 0},
        },
        {
            "level": "L0",
            "channel_id": ZENBOOK_CHANNEL_ID,
            "run_id": FIXTURE_RUN_ID,
            "created_at": ts(66),
            "window_start": ts(62),
            "window_end": ts(70),
            "summary": "Orlandina enters the Zenbook webcam view and sits at the desk.",
            "frame_count": 8,
            "alert_total": 1,
            "alert_counts": {"normal": 1},
            "alert_severities": ["normal"],
        },
        {
            "level": "L0",
            "channel_id": ZENBOOK_CHANNEL_ID,
            "run_id": FIXTURE_RUN_ID,
            "created_at": ts(84),
            "window_start": ts(80),
            "window_end": ts(90),
            "summary": "Orlandina is visible at the desk, facing the Zenbook webcam while typing.",
            "frame_count": 6,
            "alert_total": 1,
            "alert_counts": {"normal": 1},
            "alert_severities": ["normal"],
        },
        {
            "level": "L0",
            "channel_id": ZENBOOK_CHANNEL_ID,
            "run_id": FIXTURE_RUN_ID,
            "created_at": ts(114),
            "window_start": ts(110),
            "window_end": ts(120),
            "summary": "The chair is empty again after Orlandina leaves the frame.",
            "frame_count": 5,
            "alert_total": 0,
            "alert_counts": {"normal": 0},
        },
    ],
    "L1": [
        {
            "level": "L1",
            "channel_id": ZENBOOK_CHANNEL_ID,
            "run_id": FIXTURE_RUN_ID,
            "window_start": ts(0),
            "window_end": ts(30),
            "summary": "First half-hour: the Zenbook webcam shows the desk area with Orlandina absent.",
            "frame_count": 12,
            "item_count": 2,
            "alert_total": 0,
            "alert_counts": {"normal": 0},
        },
        {
            "level": "L1",
            "channel_id": ZENBOOK_CHANNEL_ID,
            "run_id": FIXTURE_RUN_ID,
            "window_start": ts(30),
            "window_end": ts(60),
            "summary": "Second half-hour: background motion occurs, but Orlandina is still not present.",
            "frame_count": 5,
            "item_count": 1,
            "alert_total": 0,
            "alert_counts": {"normal": 0},
        },
        {
            "level": "L1",
            "channel_id": ZENBOOK_CHANNEL_ID,
            "run_id": FIXTURE_RUN_ID,
            "window_start": ts(60),
            "window_end": ts(90),
            "summary": "Transition window: Orlandina appears around 07:05 UTC and remains visible at the desk.",
            "frame_count": 14,
            "item_count": 2,
            "alert_total": 2,
            "alert_counts": {"normal": 2},
            "alert_severities": ["normal"],
        },
        {
            "level": "L1",
            "channel_id": ZENBOOK_CHANNEL_ID,
            "run_id": FIXTURE_RUN_ID,
            "window_start": ts(90),
            "window_end": ts(120),
            "summary": "Final half-hour: Orlandina is visible briefly, then the chair becomes empty again.",
            "frame_count": 5,
            "item_count": 1,
            "alert_total": 0,
            "alert_counts": {"normal": 0},
        },
    ],
    "L2": [
        {
            "level": "L2",
            "channel_id": ZENBOOK_CHANNEL_ID,
            "run_id": FIXTURE_RUN_ID,
            "window_start": ts(0),
            "window_end": ts(60),
            "summary": "06:00-07:00 UTC: Orlandina is absent from the Zenbook webcam view.",
            "frame_count": 17,
            "item_count": 3,
            "alert_total": 0,
            "alert_counts": {"normal": 0},
        },
        {
            "level": "L2",
            "channel_id": ZENBOOK_CHANNEL_ID,
            "run_id": FIXTURE_RUN_ID,
            "window_start": ts(60),
            "window_end": ts(120),
            "summary": "07:00-08:00 UTC: Orlandina transitions from absent to present, visible at the desk after about 07:05 UTC.",
            "frame_count": 19,
            "item_count": 3,
            "alert_total": 2,
            "alert_counts": {"normal": 2},
            "alert_severities": ["normal"],
        },
    ],
    "L3": [],
}


DOG_ROLLUPS: dict[str, list[dict[str, Any]]] = {
    "L0": [
        {
            "level": "L0",
            "channel_id": DOG_CHANNEL_ID,
            "run_id": FIXTURE_RUN_ID,
            "created_at": ts(74),
            "window_start": ts(72),
            "window_end": ts(78),
            "summary": "A dog is visible near the clinic side door; no ear tag is visible from this angle.",
            "frame_count": 4,
            "alert_total": 1,
            "alert_counts": {"notice": 1},
            "alert_severities": ["notice"],
        }
    ],
    "L1": [
        {
            "level": "L1",
            "channel_id": DOG_CHANNEL_ID,
            "run_id": FIXTURE_RUN_ID,
            "window_start": ts(60),
            "window_end": ts(90),
            "summary": "One dog sighting. The visual evidence supports 'no visible ear tag' only; vaccination status is not observable.",
            "frame_count": 4,
            "item_count": 1,
            "alert_total": 1,
            "alert_counts": {"notice": 1},
            "alert_severities": ["notice"],
        }
    ],
    "L2": [
        {
            "level": "L2",
            "channel_id": DOG_CHANNEL_ID,
            "run_id": FIXTURE_RUN_ID,
            "window_start": ts(0),
            "window_end": ts(120),
            "summary": "Across the window, the dog case should be worded as no visible ear tag, not as an unvaccinated dog.",
            "frame_count": 4,
            "item_count": 1,
            "alert_total": 1,
            "alert_counts": {"notice": 1},
            "alert_severities": ["notice"],
        }
    ],
    "L3": [],
}


_SWEEP_EVENTS = {
    201: "person pauses at the north gate",
    202: "forklift crosses the loading bay",
    203: "visitor waits at reception",
    204: "door opens toward the side yard",
    205: "staff member cleans a table",
    206: "technician enters the server hall",
    207: "car stops in the parking aisle",
    208: "cart is moved through the warehouse door",
    209: "sparks are tested at the workshop bench",
}


def _sweep_rollups(channel_id: int, index: int, title: str, event: str) -> dict[str, list[dict[str, Any]]]:
    start_minute = 12 + index * 5
    return {
        "L0": [
            {
                "level": "L0",
                "channel_id": channel_id,
                "run_id": FIXTURE_RUN_ID,
                "created_at": ts(start_minute + 2),
                "window_start": ts(start_minute),
                "window_end": ts(start_minute + 5),
                "summary": f"{title}: {event}.",
                "frame_count": 3,
                "alert_total": 1,
                "alert_counts": {"normal": 1},
                "alert_severities": ["normal"],
            }
        ],
        "L1": [
            {
                "level": "L1",
                "channel_id": channel_id,
                "run_id": FIXTURE_RUN_ID,
                "window_start": ts(start_minute),
                "window_end": ts(start_minute + 15),
                "summary": f"{title} has activity in the sweep window: {event}.",
                "frame_count": 3,
                "item_count": 1,
                "alert_total": 1,
                "alert_counts": {"normal": 1},
                "alert_severities": ["normal"],
            }
        ],
        "L2": [],
        "L3": [],
    }


SUMMARY_ROLLUPS_BY_CHANNEL: dict[int, dict[str, list[dict[str, Any]]]] = {
    ZENBOOK_CHANNEL_ID: ZENBOOK_ROLLUPS,
    DOG_CHANNEL_ID: DOG_ROLLUPS,
}

for _idx, (_channel_id, _event) in enumerate(_SWEEP_EVENTS.items()):
    SUMMARY_ROLLUPS_BY_CHANNEL[_channel_id] = _sweep_rollups(
        _channel_id,
        _idx,
        _SWEEP_CHANNEL_TITLES[_channel_id],
        _event,
    )

SUMMARY_ROLLUPS_BY_CHANNEL[210] = {"L0": [], "L1": [], "L2": [], "L3": []}


SUMMARY_LOGS_BY_CHANNEL: dict[int, list[dict[str, Any]]] = {
    channel_id: [dict(node) for node in levels.get("L0", [])]
    for channel_id, levels in SUMMARY_ROLLUPS_BY_CHANNEL.items()
}


DETECTION_ROWS: list[dict[str, Any]] = [
    {
        "id": 112001,
        "detection_id": 112001,
        "channel_id": ZENBOOK_CHANNEL_ID,
        "source": "vlm_summary",
        "event_timestamp_ms": ms(8),
        "thumbnail": FAKE_THUMBNAIL,
        "image_path": "/fixtures/eva/zenbook/112_0608_absent.jpg",
        "probe_id": "video-summary",
        "probe_name": "VLM summary frame",
        "severity": "normal",
        "similarity": 0.44,
        "payload": {
            "run_id": FIXTURE_RUN_ID,
            "frame_timestamp_ms": ms(8),
            "frame_index": 8,
            "anchor_role": "pre_transition",
            "summary": "Zenbook webcam frame shows an empty chair and desk; Orlandina is absent.",
        },
    },
    {
        "id": 112002,
        "detection_id": 112002,
        "channel_id": ZENBOOK_CHANNEL_ID,
        "source": "vlm_summary",
        "timestamp_ms": ms(39),
        "thumbnail": FAKE_THUMBNAIL,
        "image_path": "/fixtures/eva/zenbook/112_0639_absent.jpg",
        "probe_id": "video-summary",
        "probe_name": "VLM summary frame",
        "severity": "normal",
        "similarity": 0.48,
        "payload": {
            "run_id": FIXTURE_RUN_ID,
            "frame_timestamp_ms": ms(39),
            "frame_index": 39,
            "anchor_role": "absent_context",
            "summary": "Desk area remains visible; no frame evidence of Orlandina being present.",
        },
    },
    {
        "id": 112003,
        "detection_id": 112003,
        "channel_id": ZENBOOK_CHANNEL_ID,
        "source": "vlm_alert",
        "recorded_at_ms": ms(65),
        "thumbnail": FAKE_THUMBNAIL,
        "image_path": "/fixtures/eva/zenbook/112_0705_orlandina_enters.jpg",
        "probe_id": "orlandina-presence",
        "probe_name": "Orlandina presence",
        "severity": "normal",
        "pos_score": 0.91,
        "neg_score": 0.12,
        "margin": 0.79,
        "similarity": 0.91,
        "payload": {
            "run_id": FIXTURE_RUN_ID,
            "batch_start_ms": ms(62),
            "batch_end_ms": ms(70),
            "frame_timestamp_ms": ms(65),
            "frame_index": 65,
            "anchor_role": "transition_evidence",
            "severity": "normal",
            "alert_total": 1,
            "alert_counts": {"normal": 1},
            "summary": "Orlandina enters the Zenbook webcam view and sits at the desk.",
        },
    },
    {
        "id": 112004,
        "detection_id": 112004,
        "channel_id": ZENBOOK_CHANNEL_ID,
        "source": "vlm_summary",
        "timestamp_ms": ms(82),
        "event_timestamp_ms": ms(82),
        "thumbnail": FAKE_THUMBNAIL,
        "image_path": "/fixtures/eva/zenbook/112_0722_orlandina_present.jpg",
        "probe_id": "video-summary",
        "probe_name": "VLM summary frame",
        "severity": "normal",
        "similarity": 0.87,
        "payload": {
            "run_id": FIXTURE_RUN_ID,
            "frame_timestamp_ms": ms(82),
            "frame_index": 82,
            "anchor_role": "present_context",
            "summary": "Orlandina is visible at the desk, facing the Zenbook webcam while typing.",
        },
    },
    {
        "id": 305001,
        "detection_id": 305001,
        "channel_id": DOG_CHANNEL_ID,
        "source": "vlm_summary",
        "timestamp_ms": ms(73),
        "thumbnail": FAKE_THUMBNAIL,
        "image_path": "/fixtures/eva/dog/305_0713_no_visible_ear_tag.jpg",
        "probe_id": "video-summary",
        "probe_name": "VLM summary frame",
        "severity": "notice",
        "similarity": 0.88,
        "payload": {
            "run_id": FIXTURE_RUN_ID,
            "frame_timestamp_ms": ms(73),
            "frame_index": 73,
            "anchor_role": "sensitive_wording_positive",
            "summary": "A dog is visible near the clinic side door; no ear tag is visible from this angle.",
        },
    },
    {
        "id": 305002,
        "detection_id": 305002,
        "channel_id": DOG_CHANNEL_ID,
        "source": "vlm_alert",
        "event_timestamp_ms": ms(74),
        "thumbnail": FAKE_THUMBNAIL,
        "image_path": "/fixtures/eva/dog/305_0714_dog_alert.jpg",
        "probe_id": "dog-no-visible-ear-tag",
        "probe_name": "Dog without visible ear tag",
        "severity": "notice",
        "pos_score": 0.86,
        "neg_score": 0.18,
        "margin": 0.68,
        "similarity": 0.86,
        "payload": {
            "run_id": FIXTURE_RUN_ID,
            "batch_start_ms": ms(72),
            "batch_end_ms": ms(78),
            "frame_timestamp_ms": ms(74),
            "frame_index": 74,
            "anchor_role": "sensitive_wording_evidence",
            "severity": "notice",
            "alert_total": 1,
            "alert_counts": {"notice": 1},
            "summary": "Dog at clinic side door. No visible ear tag; vaccination status is not observable.",
        },
    },
    {
        "id": 305003,
        "detection_id": 305003,
        "channel_id": DOG_CHANNEL_ID,
        "source": "probe",
        "recorded_at_ms": ms(75),
        "thumbnail": FAKE_THUMBNAIL,
        "image_path": "/fixtures/eva/dog/305_0715_probe_no_visible_tag.jpg",
        "probe_id": "dog-no-visible-ear-tag",
        "probe_name": "Dog without visible ear tag",
        "severity": "notice",
        "pos_score": 0.83,
        "neg_score": 0.22,
        "margin": 0.61,
        "similarity": 0.83,
        "payload": {
            "run_id": FIXTURE_RUN_ID,
            "frame_timestamp_ms": ms(75),
            "frame_index": 75,
            "anchor_role": "sensitive_wording_probe",
            "summary": "Probe fired for dog without visible ear tag. Do not state the dog is unvaccinated.",
        },
    },
]


SCENARIOS: dict[str, ScenarioFixture] = {
    "zenbook_orlandina_transition": ScenarioFixture(
        slug="zenbook_orlandina_transition",
        title="Zenbook webcam Orlandina absent-to-present transition",
        channel_ids=(ZENBOOK_CHANNEL_ID,),
        from_ts=WINDOW_START_TS,
        to_ts=WINDOW_END_TS,
        prompt="Summarize Zenbook webcam channel 112 over the last two hours and check whether Orlandina appears.",
        expected_behavior=(
            "Use L2/L1/L0 summaries and VLM evidence frames; describe Orlandina as absent before "
            "07:05 UTC and present afterward, without claiming more certainty than the frame evidence supports."
        ),
        metadata={
            "depths": ("L2", "L1", "L0"),
            "transition_ts": ts(65),
            "evidence_sources": ("vlm_summary", "vlm_alert"),
        },
    ),
    "multi_channel_event_sweep": ScenarioFixture(
        slug="multi_channel_event_sweep",
        title="Multi-channel event sweep exceeds per-turn channel limit",
        channel_ids=tuple(_SWEEP_EVENTS.keys()),
        from_ts=WINDOW_START_TS,
        to_ts=WINDOW_END_TS,
        prompt="Review all active video-summary channels for notable events in this two-hour window.",
        expected_behavior=(
            "Detect that active channels exceed the per-turn limit, present candidate channels, and ask "
            "for confirmation or chunk the work before reading every channel."
        ),
        metadata={
            "active_channel_count": len(_SWEEP_EVENTS) + 2,
            "agent_per_turn_limit": 8,
            "sweep_channels": tuple(_SWEEP_EVENTS.keys()),
        },
    ),
    "dog_no_visible_ear_tag": ScenarioFixture(
        slug="dog_no_visible_ear_tag",
        title="Sensitive wording: no visible ear tag is not an unvaccinated claim",
        channel_ids=(DOG_CHANNEL_ID,),
        from_ts=WINDOW_START_TS,
        to_ts=WINDOW_END_TS,
        prompt="Was there an unvaccinated dog at the clinic side door?",
        expected_behavior=(
            "Say the visual evidence shows a dog without a visible ear tag and that vaccination status "
            "cannot be determined from the image."
        ),
        metadata={
            "allowed_wording": "dog without a visible ear tag",
            "avoid_claim": "unvaccinated dog",
            "evidence_sources": ("vlm_summary", "vlm_alert", "probe"),
        },
    ),
}


def get_scenario(slug: str) -> ScenarioFixture:
    return SCENARIOS[slug]
