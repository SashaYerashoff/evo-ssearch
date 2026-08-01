from __future__ import annotations

from incident_service import (
    IncidentDraftAssembler,
    IncidentDraftRequest,
    _connected_candidate_rows,
    incident_report_markdown,
    incident_report_xml,
)


class _Detections:
    def __init__(self) -> None:
        self.anchor = {
            "id": 41,
            "channel_id": 112,
            "timestamp_ms": 1_700_000_120_000,
            "probe_name": "Possible crossing",
            "payload": {"title": "Small craft crosses the gate"},
        }

    def fetch_detections_by_ids(self, ids, **_kwargs):
        return [self.anchor] if 41 in ids else []

    def list_vlm_summary_batches(self, **_kwargs):
        return (
            [
                {
                    "archive_id": 10,
                    "batch_id": "b-1",
                    "batch_start_ms": 1_700_000_060_000,
                    "batch_end_ms": 1_700_000_100_000,
                    "alert_total": 1,
                    "payload": {
                        "alert_events": [
                            {
                                "key": "gate_crossing",
                                "label": "Small craft enters the gate",
                                "severity": "normal",
                                "confidence": "medium",
                                "timestamp_ms": 1_700_000_090_000,
                            }
                        ]
                    },
                },
                {
                    "archive_id": 11,
                    "batch_id": "b-2",
                    "batch_start_ms": 1_700_000_100_000,
                    "batch_end_ms": 1_700_000_160_000,
                    "alert_total": 1,
                    "payload": {
                        "state_transition_events": [
                            {
                                "state_key": "risk",
                                "label": "Observed tracks begin to separate",
                                "severity": "low",
                                "timestamp_ms": 1_700_000_150_000,
                            }
                        ]
                    },
                },
            ],
            2,
        )

    def list_detections(self, **kwargs):
        if kwargs.get("source") == "vlm_summary":
            return (
                [
                    {
                        "id": 10,
                        "channel_id": 112,
                        "timestamp_ms": 1_700_000_100_000,
                        "payload": {
                            "batch_id": "b-1",
                            "batch_start_ms": 1_700_000_060_000,
                            "batch_end_ms": 1_700_000_100_000,
                            "alert_events": [
                                {
                                    "key": "gate_crossing",
                                    "label": "Small craft enters the gate",
                                    "severity": "normal",
                                    "confidence": "medium",
                                    "timestamp_ms": 1_700_000_090_000,
                                }
                            ],
                        },
                    },
                    {
                        "id": 11,
                        "channel_id": 112,
                        "timestamp_ms": 1_700_000_160_000,
                        "payload": {
                            "batch_id": "b-2",
                            "batch_start_ms": 1_700_000_100_000,
                            "batch_end_ms": 1_700_000_160_000,
                            "state_transition_events": [
                                {
                                    "state_key": "risk",
                                    "label": "Observed tracks begin to separate",
                                    "severity": "low",
                                    "timestamp_ms": 1_700_000_150_000,
                                }
                            ],
                        },
                    },
                ],
                2,
            )
        return (
            [
                {
                    "id": 41,
                    "channel_id": 112,
                    "timestamp_ms": 1_700_000_120_000,
                    "severity": "high",
                    "probe_name": "Possible crossing",
                    "payload": {
                        "key": "possible_convergence",
                        "title": "Possible convergence requires review",
                        "confidence": "medium",
                    },
                }
            ],
            1,
        )


class _Attention:
    def query_intervals(self, **_kwargs):
        return [
            {
                "id": "interval-1",
                "state": "motion",
                "started_at_ms": 1_700_000_080_000,
                "ended_at_ms": 1_700_000_110_000,
                "peak_at_ms": 1_700_000_100_000,
                "motion_p95": 0.8,
                "activity_x_max": 1.4,
            }
        ]

    def query_probe_scores(self, **_kwargs):
        return [
            {
                "probe_id": "small-craft",
                "threshold_state": "hit",
                "pos_score": 0.81,
                "neg_score": 0.22,
                "margin": 0.59,
            }
        ]

    def query_evidence_links(self, **_kwargs):
        return [
            {
                "id": "link-1",
                "kind": "embedding",
                "role": "apex",
                "embedding_snapshot_id": "snapshot-1",
                "occurred_at_ms": 1_700_000_100_000,
            }
        ]


def test_incident_draft_is_bounded_grounded_and_coverage_aware():
    draft = IncidentDraftAssembler(_Detections(), _Attention()).assemble(
        IncidentDraftRequest(channel_id=112, anchor_detection_id=41)
    )

    assert draft["state"] == "draft"
    assert draft["channel_ids"] == [112]
    assert draft["severity"] == "high"
    assert draft["time_bounds"]["observed_start_ms"] == 1_700_000_060_000
    assert draft["time_bounds"]["observed_end_ms"] == 1_700_000_160_000
    assert {item["semantic_key"] for item in draft["timeline"]} >= {
        "gate_crossing",
        "possible_convergence",
        "risk",
        "motion_peak",
    }
    assert draft["qualia_digest"]["ground_truth"] is False
    assert draft["qualia_digest"]["probes"][0]["hits"] == 1
    assert any(item.get("reference") == "snapshot-1" for item in draft["evidence"])
    assert draft["coverage"]["must_state_coverage"] is True


def test_incident_draft_rejects_cross_channel_anchor():
    try:
        IncidentDraftAssembler(_Detections()).assemble(
            IncidentDraftRequest(channel_id=118, anchor_detection_id=41)
        )
    except ValueError as exc:
        assert "does not belong" in str(exc)
    else:
        raise AssertionError("cross-channel anchor must be rejected")


def test_incident_exports_keep_timeline_and_grounding_warning():
    draft = IncidentDraftAssembler(_Detections(), _Attention()).assemble(
        IncidentDraftRequest(channel_id=112, anchor_detection_id=41)
    )
    draft["id"] = "incident-1"

    markdown = incident_report_markdown(draft)
    xml = incident_report_xml(draft).decode("utf-8")

    assert "Possible convergence requires review" in markdown
    assert "attention signals, not visual ground truth" in markdown
    assert 'groundTruthStatus="operator_review_required"' in xml
    assert "gate_crossing" in xml


def test_distant_alert_does_not_steal_a_quiet_selected_anchor():
    rows = [
        {
            "archive_id": 1,
            "batch_start_ms": 1_000,
            "batch_end_ms": 2_000,
            "alert_total": 0,
        },
        {
            "archive_id": 2,
            "batch_start_ms": 600_000,
            "batch_end_ms": 660_000,
            "alert_total": 1,
        },
    ]

    selected = _connected_candidate_rows(rows, anchor_ms=1_500, max_gap_ms=120_000)

    assert [row["archive_id"] for row in selected] == [1]


def test_state_transition_count_makes_summary_an_incident_candidate():
    rows = [
        {
            "archive_id": 1,
            "batch_start_ms": 1_000,
            "batch_end_ms": 2_000,
            "state_transition_total": 1,
        },
        {
            "archive_id": 2,
            "batch_start_ms": 2_000,
            "batch_end_ms": 3_000,
            "state_transition_total": 1,
        },
    ]

    selected = _connected_candidate_rows(rows, anchor_ms=2_500, max_gap_ms=120_000)

    assert [row["archive_id"] for row in selected] == [1, 2]
