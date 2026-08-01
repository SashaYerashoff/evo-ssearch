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
                    "archive_id": 9,
                    "batch_id": "b-0",
                    "batch_start_ms": 1_700_000_000_000,
                    "batch_end_ms": 1_700_000_060_000,
                    "alert_total": 0,
                    "payload": {},
                },
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
                {
                    "archive_id": 12,
                    "batch_id": "b-3",
                    "batch_start_ms": 1_700_000_160_000,
                    "batch_end_ms": 1_700_000_220_000,
                    "alert_total": 0,
                    "payload": {},
                },
            ],
            4,
        )

    def list_detections(self, **kwargs):
        if kwargs.get("source") == "vlm_summary":
            return (
                [
                    {
                        "id": 9,
                        "channel_id": 112,
                        "timestamp_ms": 1_700_000_060_000,
                        "payload": {
                            "batch_id": "b-0",
                            "batch_start_ms": 1_700_000_000_000,
                            "batch_end_ms": 1_700_000_060_000,
                        },
                    },
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
                    {
                        "id": 12,
                        "channel_id": 112,
                        "timestamp_ms": 1_700_000_220_000,
                        "payload": {
                            "batch_id": "b-3",
                            "batch_start_ms": 1_700_000_160_000,
                            "batch_end_ms": 1_700_000_220_000,
                        },
                    },
                ],
                4,
            )
        if kwargs.get("source") == "semantic_snapshot":
            timestamps = (
                1_700_000_030_000,
                1_700_000_060_000,
                1_700_000_120_000,
                1_700_000_160_000,
                1_700_000_190_000,
            )
            rows = [
                {
                    "id": 100 + index,
                    "channel_id": 112,
                    "timestamp_ms": timestamp_ms,
                    "source": "semantic_snapshot",
                    "shard_key": f"snapshot-{index}",
                    "payload": {"cadence_ms": 1000},
                }
                for index, timestamp_ms in enumerate(timestamps)
                if kwargs.get("since_ms", 0) <= timestamp_ms <= kwargs.get("until_ms", timestamp_ms)
            ]
            return rows, len(rows)
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
                "probe_version": "v1",
                "embedding_snapshot_id": "snapshot-1",
                "captured_at_ms": 1_700_000_100_000,
                "scored_at_ms": 1_700_000_100_100,
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


class _OpenDetections(_Detections):
    def list_vlm_summary_batches(self, **kwargs):
        rows, _total = super().list_vlm_summary_batches(**kwargs)
        selected = [row for row in rows if row.get("batch_id") != "b-3"]
        return selected, len(selected)

    def list_detections(self, **kwargs):
        rows, total = super().list_detections(**kwargs)
        if kwargs.get("source") != "vlm_summary":
            return rows, total
        selected = [
            row
            for row in rows
            if (row.get("payload") or {}).get("batch_id") != "b-3"
        ]
        return selected, len(selected)


class _PartialAttention(_Attention):
    def query_probe_scores(self, **_kwargs):
        raise RuntimeError("probe score store unavailable")


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
    assert draft["qualia_digest"]["score_refs"][0]["embedding_snapshot_id"] == "snapshot-1"
    assert draft["qualia_digest"]["score_refs"][0]["captured_at_ms"] == 1_700_000_100_000
    assert any(item.get("reference") == "snapshot-1" for item in draft["evidence"])
    assert draft["coverage"]["must_state_coverage"] is True


def test_incident_envelope_keeps_context_and_semantic_control_roles():
    draft = IncidentDraftAssembler(_Detections(), _Attention()).assemble(
        IncidentDraftRequest(channel_id=112, anchor_detection_id=41)
    )

    context_roles = [item["role"] for item in draft["summary_context"]]
    assert context_roles.count("context_before") == 1
    assert context_roles.count("candidate") == 2
    assert context_roles.count("context_after") == 1
    semantic_roles = {
        item["role"]
        for item in draft["evidence"]
        if item.get("kind") == "semantic_snapshot"
    }
    assert semantic_roles == {
        "control_before",
        "onset",
        "apex",
        "post",
        "control_after",
    }
    assert draft["time_bounds"]["end_status"] == "post_control"
    assert draft["coverage"]["ledger"]["semantic_snapshots"]["overall"] == "ok"
    assert draft["coverage"]["ledger"]["attention"]["overall"] == "ok"


def test_ongoing_incident_keeps_observed_end_open_without_post_control():
    draft = IncidentDraftAssembler(_OpenDetections(), _Attention()).assemble(
        IncidentDraftRequest(channel_id=112, anchor_detection_id=41)
    )

    assert draft["time_bounds"]["observed_end_ms"] is None
    assert draft["time_bounds"]["end_status"] == "open"


def test_attention_query_failure_is_partial_not_silently_empty():
    draft = IncidentDraftAssembler(_Detections(), _PartialAttention()).assemble(
        IncidentDraftRequest(channel_id=112, anchor_detection_id=41)
    )

    attention = draft["coverage"]["ledger"]["attention"]
    assert attention["overall"] == "partial"
    assert attention["queries"]["scores"]["status"] == "unavailable"
    assert attention["queries"]["intervals"]["status"] == "ok"
    assert any(item["semantic_key"] == "motion_peak" for item in draft["timeline"])


def test_one_minute_missing_l0_window_is_reported_as_a_gap():
    coverage = IncidentDraftAssembler._coverage(
        [
            {"batch_start_ms": 1_000, "batch_end_ms": 61_000},
            {"batch_start_ms": 121_000, "batch_end_ms": 181_000},
        ],
        since_ms=1_000,
        until_ms=181_000,
        summary_total=2,
    )

    assert coverage["status"] == "partial"
    assert coverage["ledger"]["l0"]["inferred_gap_count"] == 1
    assert coverage["gaps"][0]["duration_ms"] == 60_000


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
