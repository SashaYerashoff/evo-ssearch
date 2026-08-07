from incident_presentation import (
    build_follow_result,
    build_incident_synopsis,
    classify_follow_heartbeat,
)


def _incident():
    return {
        "title": "### Scene description A cluttered indoor room",
        "timeline_refs": [
            *[
                {
                    "timestamp_ms": 1_000 + index * 1_000,
                    "semantic_key": "motion_peak",
                    "label": f"Homeostatic motion peak activity={index}",
                    "source": "cv_motion_interval",
                    "confidence": "signal_only",
                }
                for index in range(30)
            ],
            {
                "timestamp_ms": 35_000,
                "semantic_key": "person_entry",
                "source": "batch_state",
                "label": (
                    "### Scene description A cluttered room. "
                    "### Episode update A person enters from the left, approaches the chair, and sits down. "
                    "### Routine and deviations No further deviation."
                ),
            },
        ],
        "qualia_refs": [
            {
                "motion_interval_count": 60,
                "motion_p95_max": 0.071,
                "motion_p95_mean": 0.012,
                "motion_profile": {
                    "activity_x_max": 28.525,
                    "activity_x_mean": 6.2,
                    "apex_at_ms": 35_000,
                    "elevated_duration_ms": 8_000,
                    "settling_ms": 14_000,
                    "burst_count": 1,
                },
                "probes": [{"probe_id": "chair", "samples": 12, "hits": 3}],
            }
        ],
        "coverage": {"status": "covered"},
        "uncertainties": [],
        "perception_state": "observed",
        "risk_state": "unknown",
        "case_state": "open",
        "attention_state": "inactive",
        "report": {},
    }


def test_synopsis_hides_signal_rows_and_builds_short_semantic_title():
    synopsis = build_incident_synopsis(_incident())

    assert synopsis["title"] == "Person enters and sits"
    assert synopsis["description"].startswith("A person enters from the left")
    assert "Homeostatic motion peak" not in synopsis["description"]
    assert len(synopsis["key_moments"]) == 1
    assert synopsis["homeostasis"]["activity_x_max"] == 28.52
    assert synopsis["homeostasis"]["probe_hits"] == 3


def test_heartbeat_association_requires_matching_semantic_track():
    incident = _incident()
    neutral = classify_follow_heartbeat(incident, {"batch_state": {"events": []}})
    unrelated = classify_follow_heartbeat(
        incident,
        {"batch_state": {"events": [{"event_id": "cat_jump", "state": "new"}]}},
    )
    supporting = classify_follow_heartbeat(
        incident,
        {"batch_state": {"events": [{"event_id": "person_entry", "state": "continuing"}]}},
    )
    resolved = classify_follow_heartbeat(
        incident,
        {
            "batch_state": {
                "routines": [
                    {"state": "returned", "applies_to_event_keys": ["person_entry"]},
                ]
            }
        },
    )

    assert neutral["association"] == "neutral"
    assert neutral["perception_state"] == "unknown"
    assert unrelated["association"] == "unrelated"
    assert supporting["association"] == "supports"
    assert resolved["association"] == "resolved"


def test_recurrence_watch_never_turns_silence_into_absence():
    incident = {
        **_incident(),
        "follow_policy": {
            "run_id": "watch-1",
            "relationship": "recurrence_watch",
            "started_at_ms": 10_000,
        },
    }
    result = build_follow_result(
        incident,
        [
            {
                "observed_at_ms": 20_000,
                "source_ref": {"follow_run_id": "watch-1"},
                "payload": {"association": "neutral"},
            }
        ],
        ended_at_ms=30_000,
        stop_reason="ttl_expired",
    )

    assert result["outcome"] == "recurrence_not_confirmed"
    assert "absence was not inferred" in result["description"]


def test_legacy_motion_labels_backfill_human_homeostasis_without_migration():
    incident = {
        **_incident(),
        "observed_start_ms": 30_000,
        "observed_end_ms": 40_000,
        "qualia_refs": [{"motion_interval_count": 3}],
        "timeline_refs": [
            {
                "timestamp_ms": 34_000,
                "semantic_key": "motion_peak",
                "source": "cv_motion_interval",
                "confidence": "signal_only",
                "label": "Homeostatic motion peak (p95=0.035, activity=14.365)",
            },
            {
                "timestamp_ms": 35_000,
                "semantic_key": "motion_peak",
                "source": "cv_motion_interval",
                "confidence": "signal_only",
                "label": "Homeostatic motion peak (p95=0.071, activity=28.525)",
            },
            {
                "timestamp_ms": 36_000,
                "semantic_key": "motion_peak",
                "source": "cv_motion_interval",
                "confidence": "signal_only",
                "label": "Homeostatic motion peak (p95=0.006, activity=2.288)",
            },
        ],
    }

    digest = build_incident_synopsis(incident)["homeostasis"]

    assert digest["activity_x_max"] == 28.52
    assert digest["apex_at_ms"] == 35_000
    assert digest["elevated_duration_ms"] == 2_000
    assert digest["burst_count"] == 1
