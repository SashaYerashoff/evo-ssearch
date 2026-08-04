import unittest
from unittest.mock import patch

from agent import (
    AGENT_VIDEO_SUMMARY_CHANNELS_PER_TURN,
    AgentTools,
    _aggregate_vlm_alert_episodes,
    _apply_turn_tool_context,
    _compact_prompt_settings_for_model,
    _compact_tool_result_for_model,
    _compact_vector_signal_for_model,
    _archive_research_response_needs_recovery,
    _format_epoch_minute,
    _format_archive_research_fallback,
    _format_turn_signal_ledger_message,
    _new_turn_signal_ledger,
    _record_turn_signal_ledger,
    _remember_turn_tool_result,
    _select_archive_vision_candidates,
    _seed_turn_tool_context,
    _safe_detection,
    _strip_thumbnails,
    _summary_node_alert_score,
    _tool_result_for_ui,
    build_system_prompt,
)


class _SummaryManager:
    def __init__(self):
        self.summary_rollup_requests = []
        self.channels = [
            {"id": 7, "title": "Kitchen"},
            {"id": 8, "title": "Door"},
            {"id": 9, "title": "Quiet"},
        ]
        self.logs_by_channel = {
            7: [
                {"created_at": 100.0, "summary": "before window", "frame_count": 2},
                {"created_at": 150.0, "summary": "person enters", "frame_count": 3, "alert_counts": {"normal": 1}},
                {"created_at": 400.0, "summary": "after window", "frame_count": 4},
            ],
            8: [
                {"created_at": 175.0, "summary": "door light changed", "frame_count": 2},
            ],
            9: [],
        }

    def get_channels(self, force=False):
        return list(self.channels)

    def streams_status(self):
        return {
            "video_streams": [
                {
                    "channel_id": 7,
                    "running": True,
                    "model": "vlm-a1",
                    "pending_frames": 3,
                    "summary_queue_depth": 2,
                    "summary_queue_frame_count": 16,
                    "summary_inflight": True,
                    "summary_worker_alive": True,
                    "dropped_frames": 1,
                    "queue_dropped_batches": 0,
                    "recent_frame_count": 2,
                    "frozen_signal": True,
                    "frozen_signal_age_sec": 22.5,
                    "frozen_frame_count": 4,
                    "log_count": 12,
                    "last_alert_counts": {"low": 1},
                },
            ],
            "analytics_streams": [
                {
                    "channel_id": 9,
                    "stream_type": "analytics",
                    "running": True,
                    "last_snapshot_at": 1.0,
                    "recent_frame_count": 1,
                },
            ],
            "desired_video_channels": [7, 8],
            "desired_video_missing": [
                {"channel_id": 8, "last_restore_error": "snapshot timeout"},
            ],
            "video_history_channels": [7, 8],
            "channel_status_digest": [
                {
                    "channel_id": 7,
                    "summary_count": 12,
                    "last_summary_ts": 150.0,
                    "alert_total": 1,
                    "alert_counts_by_severity": {"low": 1},
                    "recent_alerts": [
                        {
                            "title": "Doorway activity",
                            "severity": "low",
                            "delivery_status": "sent",
                            "timestamp_ms": 150000,
                        }
                    ],
                    "alert_delivery_breakdown": {"sent": 1},
                    "alert_parser_breakdown": {"json_alert_count": 1},
                    "state_transition_total": 0,
                    "current_observed_state": [],
                },
            ],
        }

    @staticmethod
    def _log_bounds(row):
        created = float(row["created_at"])
        start_ms = row.get("batch_start_ms")
        end_ms = row.get("batch_end_ms")
        if start_ms is None and end_ms is None:
            return created, created
        if start_ms is None:
            start_ms = end_ms
        if end_ms is None:
            end_ms = start_ms
        start = float(start_ms) / 1000.0
        end = float(end_ms) / 1000.0
        return (end, start) if end < start else (start, end)

    def session_status(self, channel_id, run_selector=None, start_ts=None, end_ts=None, limit=None):
        logs = []
        for row in self.logs_by_channel.get(int(channel_id), []):
            row_start, row_end = self._log_bounds(row)
            if start_ts is not None and row_end < float(start_ts):
                continue
            if end_ts is not None and row_start > float(end_ts):
                continue
            logs.append(dict(row))
        return {"running": False, "channel_id": channel_id, "logs": logs, "selected_run": None}

    def summary_rollups(
        self,
        channel_id,
        run_selector=None,
        start_ts=None,
        end_ts=None,
        level_limit=None,
        target_level=None,
        synthesize=True,
    ):
        self.summary_rollup_requests.append(
            {
                "channel_id": channel_id,
                "target_level": target_level,
                "synthesize": synthesize,
            }
        )
        # Intentionally include out-of-window nodes to verify AgentTools performs strict post-filtering.
        nodes = [
            {
                "level": "L1",
                "window_start": 90.0,
                "window_end": 99.0,
                "summary": "before",
                "frame_count": 1,
            },
            {
                "level": "L1",
                "window_start": 140.0,
                "window_end": 160.0,
                "summary": "inside",
                "frame_count": 3,
                "alert_counts": {"normal": 1},
                "alert_total": 1,
            },
            {
                "level": "L1",
                "window_start": 301.0,
                "window_end": 330.0,
                "summary": "after",
                "frame_count": 1,
            },
        ]
        return {
            "running": False,
            "selected_run": None,
            "run_filter_id": None,
            "levels": {"L0": [], "L1": nodes, "L2": [], "L3": []},
        }


class _DetectionStore:
    def __init__(self, rows=None):
        self.rows = rows or []

    @staticmethod
    def _row_ts(row):
        return int(row.get("timestamp_ms") or row.get("event_timestamp_ms") or row.get("recorded_at_ms") or 0)

    def list_detections(
        self,
        probe_id=None,
        channel_id=None,
        source=None,
        since_ms=None,
        until_ms=None,
        limit=50,
        offset=0,
    ):
        rows = []
        for row in self.rows:
            if probe_id is not None and row.get("probe_id") != probe_id:
                continue
            if channel_id is not None and int(row.get("channel_id") or 0) != int(channel_id):
                continue
            if source is not None and row.get("source") != source:
                continue
            ts = self._row_ts(row)
            if since_ms is not None and ts < int(since_ms):
                continue
            if until_ms is not None and ts > int(until_ms):
                continue
            rows.append(dict(row))
        rows.sort(key=lambda item: (self._row_ts(item), int(item.get("id") or 0)), reverse=True)
        total = len(rows)
        return rows[offset: offset + limit], total

    def summarize_by_probe(self, *args, **kwargs):
        return []

    def list_vector_candidates(
        self,
        *,
        channel_id=None,
        source=None,
        since_ms=None,
        until_ms=None,
        limit=50,
        offset=0,
        only_with_clip=True,
        include_vectors=False,
        include_thumbnail=True,
        embedding_space=None,
        allow_legacy_embedding_space=False,
    ):
        rows, _total = self.list_detections(
            probe_id=None,
            channel_id=channel_id,
            source=source,
            since_ms=since_ms,
            until_ms=until_ms,
            limit=limit,
            offset=offset,
        )
        if only_with_clip:
            rows = [row for row in rows if row.get("clip_vec") is not None]
        prepared = []
        for row in rows:
            item = dict(row)
            if not include_vectors:
                item.pop("clip_vec", None)
                item.pop("dino_vec", None)
            if not include_thumbnail:
                item.pop("thumbnail", None)
            prepared.append(item)
        return prepared

    def count_vector_candidates(
        self,
        *,
        channel_id=None,
        source=None,
        since_ms=None,
        until_ms=None,
        only_with_clip=True,
        embedding_space=None,
        allow_legacy_embedding_space=False,
    ):
        rows, total = self.list_detections(
            probe_id=None,
            channel_id=channel_id,
            source=source,
            since_ms=since_ms,
            until_ms=until_ms,
            limit=1_000_000,
            offset=0,
        )
        if only_with_clip:
            return sum(1 for row in rows if row.get("clip_vec") is not None)
        return total

    def fetch_detections_by_ids(self, ids, include_vectors=False, include_thumbnail=True):
        wanted = {int(item) for item in ids}
        prepared = []
        for row in self.rows:
            if int(row.get("id") or row.get("detection_id") or 0) not in wanted:
                continue
            item = dict(row)
            if not include_vectors:
                item.pop("clip_vec", None)
                item.pop("dino_vec", None)
            if not include_thumbnail:
                item.pop("thumbnail", None)
            prepared.append(item)
        return prepared


class _ProbeStore:
    def __init__(self, probes=None):
        self.probes = probes or []

    def list_probes(self):
        return [dict(probe) for probe in self.probes]


class _ChannelsMethodOnlyManager:
    def get_channels(self, force=False):
        return [
            {"id": 1353, "title": "Office lobby"},
            {"id": 1463, "title": "Street loop"},
        ]

    def streams_status(self):
        return {"video_streams": [], "channel_status_digest": []}


def _tools(
    manager=None,
    search_detections_fn=None,
    detections_store=None,
    call_lm_fn=None,
    embed_text_fn=None,
    probes_store=None,
    embedding_metadata_fn=None,
):
    return AgentTools(
        detections_store=detections_store or _DetectionStore(),
        probes_store=probes_store or _ProbeStore(),
        luxriot_manager=manager or _SummaryManager(),
        embed_text_fn=embed_text_fn or (lambda _text: None),
        embed_image_fn=lambda _image: None,
        call_lm_fn=call_lm_fn or (lambda *_args, **_kwargs: ""),
        encode_jpeg_fn=lambda *_args, **_kwargs: "",
        search_indexed_folder_fn=lambda **_kwargs: [],
        search_detections_fn=search_detections_fn or (lambda **_kwargs: []),
        embedding_metadata_fn=embedding_metadata_fn,
    )


class TurnSignalLedgerTests(unittest.TestCase):
    def test_lookup_help_ledger_keeps_citations_without_snippets(self):
        ledger = _new_turn_signal_ledger("how do I backup the database?")
        _record_turn_signal_ledger(
            ledger,
            "lookup_help",
            {
                "results": [
                    {
                        "doc": "docs/operator/operator_guide.md",
                        "section": "Video description status",
                        "score": 3.4,
                        "snippet": "OPERATOR_SNIPPET_SHOULD_NOT_BE_DUPLICATED",
                    }
                ],
                "best_match_restricted": True,
                "best_restricted_section": "Backup and recovery",
                "best_required_permission": "settings:manage",
                "restricted_matches": [
                    {
                        "doc": "docs/admin/backup_recovery.md",
                        "section": "Backup and recovery",
                        "required_permission": "settings:manage",
                        "score": 9.1,
                        "snippet": "RESTRICTED_STEPS_SHOULD_NOT_LEAK",
                    }
                ],
            },
        )

        message = _format_turn_signal_ledger_message(ledger)

        self.assertIsNotNone(message)
        self.assertIn("Documentation/help signals", message)
        self.assertIn("Restricted-help signals", message)
        self.assertIn("settings:manage", message)
        self.assertNotIn("OPERATOR_SNIPPET_SHOULD_NOT_BE_DUPLICATED", message)
        self.assertNotIn("RESTRICTED_STEPS_SHOULD_NOT_LEAK", message)

    def test_visual_state_ledger_labels_clip_as_candidate_signal(self):
        ledger = _new_turn_signal_ledger("count appearances")
        _record_turn_signal_ledger(
            ledger,
            "track_visual_state_transitions",
            {
                "channel_id": 112,
                "score_semantics": "clip_pnm_state_machine_not_ground_truth",
                "counts": {"transition": 2, "appearance": 1, "disappearance": 1},
                "frame_count": 120,
                "coverage": {"status": "partial"},
                "boundary_frames": [
                    {"detection_id": 77, "image_url": "/detections/thumbnail/77"}
                ],
            },
        )

        message = _format_turn_signal_ledger_message(ledger)

        self.assertIsNotNone(message)
        self.assertIn("Semantic/CLIP/count signals", message)
        self.assertIn("candidate signals, not proof", message)
        self.assertIn("Evidence/frame signals", message)
        self.assertIn("coverage/truncation/errors", message)


class AgentVideoSummaryToolTests(unittest.TestCase):
    def test_repeated_vlm_alerts_are_grouped_as_candidate_episodes(self):
        result = _aggregate_vlm_alert_episodes(
            [
                {
                    "title": "Vehicle drifting with tire smoke",
                    "severity": "high",
                    "delivery_status": "sent",
                    "timestamp_ms": 1_000,
                },
                {
                    "title": "Drifting vehicle creates tire smoke",
                    "severity": "high",
                    "delivery_status": "dedup_suppressed",
                    "timestamp_ms": 31_000,
                },
                {
                    "title": "Person entered the room",
                    "severity": "low",
                    "delivery_status": "sent",
                    "timestamp_ms": 40_000,
                },
                {
                    "title": "Vehicle drifting again",
                    "severity": "high",
                    "delivery_status": "sent",
                    "timestamp_ms": 901_000,
                },
            ],
            raw_alert_count=7,
            severity_counts={"high": 6, "low": 1},
            delivery_breakdown={"sent": 3, "dedup_suppressed": 4},
            episode_gap_sec=600,
        )

        self.assertEqual(result["raw_alert_count"], 7)
        self.assertEqual(result["structured_alert_count"], 4)
        self.assertEqual(result["candidate_episode_count"], 3)
        self.assertEqual(result["repeated_structured_alert_count"], 1)
        self.assertEqual(result["unclustered_alert_count"], 3)
        self.assertEqual(result["severity_counts"]["high"], 6)
        self.assertIn("not proof", result["semantics"])

    def test_ui_detection_rows_do_not_invent_image_url_without_thumbnail(self):
        missing = {
            "id": 117031,
            "detection_id": 117031,
            "source": "probe",
        }
        safe = _safe_detection(missing)
        self.assertFalse(safe["has_thumbnail"])
        self.assertNotIn("image_url", safe)

        stripped = _strip_thumbnails([{**missing, "is_detection": True}])
        self.assertFalse(stripped[0]["has_thumbnail"])
        self.assertNotIn("image_url", stripped[0])

        present = _safe_detection({**missing, "thumbnail": "abcd"})
        self.assertTrue(present["has_thumbnail"])
        self.assertEqual(present["image_url"], "/detections/thumbnail/117031")

    def test_compact_prompt_settings_preserves_layer_semantics_and_migration(self):
        compact = _compact_prompt_settings_for_model(
            {
                "channel_id": 7,
                "stream_system_prompt": "Describe public space activity.",
                "alert_policy_prompt": "Flag people fighting.",
                "json_alert_prompt": "ALERTS_JSON contract",
                "prompt_layers": {
                    "stream": {
                        "notes": ["Live L0 summaries use the editable stream prompt."],
                        "warnings": [],
                    },
                    "alerts": {
                        "notes": ["Use this layer for channel-specific alert criteria."],
                        "warnings": ["legacy criteria found"],
                    },
                    "json": {
                        "notes": ["This JSON layer is appended last."],
                    },
                    "rollups": {
                        "L1": {
                            "notes": ["Alert Ledger must preserve alert counts."]
                        }
                    },
                },
                "prompt_health": {
                    "needs_migration": True,
                    "warnings": ["move alert criteria"],
                    "candidate_alert_policy_lines": ["Flag people fighting"],
                    "suggested_stream_system_prompt": "Describe public space activity.",
                    "suggested_alert_policy_prompt": "Flag people fighting.",
                },
                "memory_metabolism": {
                    "status": "active",
                    "semantics": "L1/L2 memory returns to later L0 as prior context.",
                    "current_state": {
                        "present": True,
                        "source_level": "L2",
                        "active_watchlist_count": 2,
                    },
                    "stages": [
                        {
                            "level": "L1",
                            "cadence": "15 minutes",
                            "applies_to_live_memory": True,
                        },
                        {
                            "level": "L3",
                            "cadence": "8 hours",
                            "applies_to_live_memory": False,
                        },
                    ],
                },
            }
        )

        self.assertTrue(compact["prompt_health"]["needs_migration"])
        self.assertEqual(compact["prompt_health"]["suggested_alert_policy_prompt"], "Flag people fighting.")
        self.assertIn("L0 live-description role/style", compact["prompt_layers"]["stream"]["semantics"])
        self.assertIn("Operator watch/alert criteria", compact["prompt_layers"]["alerts"]["semantics"])
        self.assertIn("BATCH_STATE_JSON", compact["prompt_layers"]["json"]["semantics"])
        self.assertIn("compressed memory maps", compact["prompt_layers"]["rollups"]["semantics"])
        self.assertEqual(compact["memory_metabolism"]["status"], "active")
        self.assertEqual(compact["memory_metabolism"]["current_state"]["source_level"], "L2")
        self.assertTrue(compact["memory_metabolism"]["stages"][0]["applies_to_live_memory"])
        self.assertFalse(compact["memory_metabolism"]["stages"][1]["applies_to_live_memory"])

    def test_system_prompt_reframes_sensitive_visible_evidence_instead_of_refusing(self):
        class _ProbeStore:
            def list_probes(self):
                return [{"id": "probe-1", "name": "legacy probe", "channel_id": 7}]

        prompt = build_system_prompt(
            _ProbeStore(),
            _DetectionStore(),
            _SummaryManager(),
        )

        self.assertIn("Video-description runtime:", prompt)
        self.assertIn("CH 7: running, video_lm=vlm-a1", prompt)
        self.assertIn(
            "volatile period counters omitted; call list_video_summary_channels",
            prompt,
        )
        self.assertNotIn("recent_alerts=Doorway activity", prompt)
        self.assertIn("CH 8: desired but not running", prompt)
        self.assertIn("Configured semantic probes (1 total; secondary/internal", prompt)
        self.assertLess(prompt.index("Video-description runtime:"), prompt.index("Configured semantic probes"))
        self.assertIn("Default reports and status answers must be video-description-first", prompt)
        self.assertIn("double-check video-description alerts with probes", prompt)
        self.assertIn("turn VLM alerts into a secondary CLIP attention layer", prompt)
        self.assertIn("create one preview probe per event/channel", prompt)
        self.assertIn("remove private names and abstract labels", prompt)
        self.assertIn("Fight alert", prompt)
        self.assertIn("two people fighting", prompt)
        self.assertIn("Vehicle drift alert", prompt)
        self.assertIn("car doing a burnout or drift", prompt)
        self.assertIn("For probe negative prompts, never use literal absence/negation", prompt)
        self.assertIn("clear roadway with normal traffic", prompt)
        self.assertIn("empty public entrance", prompt)
        self.assertIn("use calibrate_probe_from_archive when archive frames exist", prompt)
        self.assertIn("For broad calibration across many channels", prompt)
        self.assertIn("server-side job_id", prompt)
        self.assertIn("remaining_items", prompt)
        self.assertIn("recommended_probe_args", prompt)
        self.assertIn("update_existing=true", prompt)
        self.assertIn("first-party operator/admin documentation through lookup_help", prompt)
        self.assertIn("Never answer that you cannot access the operator/admin docs", prompt)
        self.assertIn("L0/L1/L2/L3 prompts or settings", prompt)
        self.assertIn("adapted summary/translation of the cited sections", prompt)
        self.assertIn("source=probe means CLIP probe hits, not sensors", prompt)
        self.assertIn("machine translation draft", prompt)
        self.assertIn("Rank video-summary signals by provenance", prompt)
        self.assertIn("Routine memory/baseline is prior context, not current evidence", prompt)
        self.assertIn("unconfirmed prose-only evidence", prompt)
        self.assertIn("parser/delivery diagnostics as pipeline health, not incident counts", prompt)
        self.assertIn("do not refuse when the request can be reframed", prompt)
        self.assertIn("smoking weed/pipe/joint", prompt)
        self.assertIn("person holding a small cylindrical object", prompt)

    def test_system_prompt_reads_channels_through_get_channels(self):
        prompt = build_system_prompt(
            _ProbeStore(),
            _DetectionStore(),
            _ChannelsMethodOnlyManager(),
        )

        self.assertIn("Available channels: 1353 (Office lobby), 1463 (Street loop)", prompt)
        self.assertNotIn("Luxriot not connected", prompt)

    def test_live_turn_uses_small_route_specific_video_prompt(self):
        legacy = build_system_prompt(
            _ProbeStore(),
            _DetectionStore(),
            _SummaryManager(),
        )
        scoped = build_system_prompt(
            _ProbeStore(),
            _DetectionStore(),
            _SummaryManager(),
            tool_intents=["video_research", "channel_inventory"],
            secure_tool_mode=True,
        )

        self.assertLess(len(scoped), len(legacy) // 2)
        self.assertIn("Normalize a relative window once", scoped)
        self.assertIn("No coverage means unknown, not calm", scoped)
        self.assertIn("Chat write tools are preview-only", scoped)
        self.assertNotIn("For broad calibration across many channels", scoped)

    def test_channel_ref_resolution_reads_channels_through_get_channels(self):
        tools = _tools(manager=_ChannelsMethodOnlyManager())

        channel_id = tools._resolve_channel_id(
            {"channel_ref": "office lobby"},
            required=True,
        )

        self.assertEqual(channel_id, 1353)

    def test_describe_frame_prefers_detection_thumbnail_over_live_channel(self):
        class SnapshotFailManager(_SummaryManager):
            def get_snapshot_base64(self, channel_id):
                raise AssertionError("live snapshot should not be requested")

        store = _DetectionStore(
            [
                {
                    "id": 501,
                    "channel_id": 7,
                    "thumbnail": "ZmFrZS1qcGVn",
                }
            ]
        )
        tools = _tools(
            manager=SnapshotFailManager(),
            detections_store=store,
        )

        result = tools._describe_frame({"detection_id": 501, "channel_id": 7})

        self.assertEqual(result["source"], "thumbnail")
        self.assertEqual(result["snapshot_b64"], "ZmFrZS1qcGVn")

    def test_describe_frame_strips_data_url_prefix_from_thumbnail(self):
        captured = {}

        def call_lm(messages):
            captured["messages"] = messages
            return "ok"

        store = _DetectionStore(
            [
                {
                    "id": 501,
                    "channel_id": 7,
                    "thumbnail": "data:image/jpeg;base64,ZmFrZS1qcGVn",
                }
            ]
        )
        tools = _tools(detections_store=store, call_lm_fn=call_lm)

        result = tools._describe_frame({"detection_id": 501})

        self.assertEqual(result["source"], "thumbnail")
        self.assertEqual(result["snapshot_b64"], "ZmFrZS1qcGVn")
        self.assertEqual(result["image_url"], "/detections/thumbnail/501")
        image_part = captured["messages"][1]["content"][1]["image_url"]["url"]
        self.assertEqual(image_part, "data:image/jpeg;base64,ZmFrZS1qcGVn")

    def test_describe_frame_ui_result_uses_image_url_without_inline_base64(self):
        result = {
            "description": "ok",
            "source": "thumbnail",
            "detection_id": 501,
            "image_url": "/detections/thumbnail/501",
            "snapshot_b64": "ZmFrZS1qcGVn",
        }

        ui_result = _tool_result_for_ui("describe_frame", result)

        self.assertEqual(ui_result["image_url"], "/detections/thumbnail/501")
        self.assertNotIn("snapshot_b64", ui_result)
        self.assertIn("snapshot_b64", result)

    def test_describe_frame_batches_ranked_candidates_in_one_vision_call(self):
        captured = []

        def call_lm(messages):
            captured.append(messages)
            return (
                '{"verdicts": ['
                '{"snapshot_index": 1, "verdict": "match", '
                '"visible_evidence": "A hairless cat is visible on the shelf."},'
                '{"snapshot_index": 2, "verdict": "no_match", '
                '"visible_evidence": "Only an empty shelf is visible."}'
                ']}'
            )

        store = _DetectionStore(
            [
                {
                    "id": 501,
                    "channel_id": 7,
                    "timestamp_ms": 1_000,
                    "source": "vlm_summary",
                    "thumbnail": "ZmFrZS0x",
                },
                {
                    "id": 502,
                    "channel_id": 7,
                    "timestamp_ms": 2_000,
                    "source": "semantic_snapshot",
                    "thumbnail": "ZmFrZS0y",
                },
            ]
        )
        tools = _tools(detections_store=store, call_lm_fn=call_lm)

        result = tools._describe_frame(
            {
                "detection_ids": [501, 502],
                "prompt": "sphynx cat",
            }
        )

        self.assertEqual(len(captured), 1)
        image_parts = [
            part
            for part in captured[0][1]["content"]
            if part.get("type") == "image_url"
        ]
        self.assertEqual(len(image_parts), 2)
        self.assertEqual(result["source"], "archive_candidate_batch")
        self.assertEqual(result["parse_status"], "parsed")
        self.assertEqual(result["matched_detection_ids"], [501])
        self.assertEqual(result["no_match_detection_ids"], [502])
        self.assertEqual(result["uncertain_count"], 0)

    def test_describe_frame_batch_treats_unparsed_output_as_uncertain(self):
        store = _DetectionStore(
            [
                {"id": 501, "channel_id": 7, "thumbnail": "ZmFrZS0x"},
                {"id": 502, "channel_id": 7, "thumbnail": "ZmFrZS0y"},
            ]
        )
        tools = _tools(
            detections_store=store,
            call_lm_fn=lambda _messages: "The first image probably contains a cat.",
        )

        result = tools._describe_frame(
            {"detection_ids": [501, 502], "prompt": "sphynx cat"}
        )

        self.assertEqual(result["parse_status"], "unparsed")
        self.assertEqual(result["match_count"], 0)
        self.assertEqual(result["uncertain_detection_ids"], [501, 502])

    def test_archive_vision_candidates_dedupe_one_frame_and_prefer_summary(self):
        rows = [
            {
                "detection_id": 101,
                "channel_id": 7,
                "timestamp_ms": 1_000,
                "source": "vlm_alert",
            },
            {
                "detection_id": 102,
                "channel_id": 7,
                "timestamp_ms": 1_000,
                "source": "vlm_summary",
            },
            {
                "detection_id": 103,
                "channel_id": 7,
                "timestamp_ms": 2_000,
                "source": "semantic_snapshot",
            },
        ]

        selected = _select_archive_vision_candidates(rows, limit=8)

        self.assertEqual([row["detection_id"] for row in selected], [102, 103])

    def test_archive_search_compaction_preserves_text_and_candidate_limits(self):
        result = {
            "scope": "detections",
            "query": "sphynx cat",
            "count": 12,
            "match_semantics": "ranked_candidates_not_binary_matches",
            "time_window": {"duration_sec": 86_400},
            "lexical_match_count_in_returned": 1,
            "vision_candidate_ids": [101, 102],
            "vision_candidate_count": 2,
            "vision_verification_required": True,
            "coverage": {"scanned_candidates": 20_000, "truncated": True},
            "results": [
                {
                    "detection_id": index,
                    "channel_id": 7,
                    "timestamp_ms": index * 1_000,
                    "source": "vlm_summary",
                    "score": 0.8,
                    "image_url": f"/detections/thumbnail/{index}",
                    "text_evidence_excerpt": (
                        "A Sphynx cat is perched atop the fridge."
                        if index == 101
                        else "Person at desk"
                    ),
                    "lexical_match": index == 101,
                    "lexical_match_kind": "exact_phrase" if index == 101 else "none",
                }
                for index in range(101, 113)
            ],
        }

        compact = _compact_tool_result_for_model("search_archive", result)

        self.assertEqual(compact["results_returned_to_model"], 8)
        self.assertEqual(compact["results_omitted_from_model"], 4)
        self.assertEqual(compact["vision_candidate_ids"], [101, 102])
        self.assertIn("Sphynx cat", compact["results"][0]["text_evidence_excerpt"])
        self.assertEqual(
            compact["results"][0]["score_semantics"],
            "semantic_retrieval_ranking_not_probability",
        )

    def test_archive_negative_claim_requires_parsed_vision_and_no_match(self):
        context = {
            "tool_intents": ["archive_research"],
            "archive_search_completed": True,
            "archive_vision_required": True,
            "archive_vision_completed": False,
        }
        claim = 'No "sphynx cat" detected in the archive.'

        self.assertTrue(_archive_research_response_needs_recovery(claim, context))

        context.update(
            {
                "archive_vision_completed": True,
                "archive_vision_parse_status": "parsed",
                "archive_vision_match_count": 1,
                "archive_vision_no_match_count": 0,
                "archive_vision_uncertain_count": 0,
            }
        )
        self.assertTrue(_archive_research_response_needs_recovery(claim, context))

        context["archive_vision_match_count"] = 0
        context["archive_vision_no_match_count"] = 8
        self.assertFalse(_archive_research_response_needs_recovery(claim, context))

        context["archive_vision_uncertain_count"] = 1
        self.assertTrue(_archive_research_response_needs_recovery(claim, context))

        positive = "Visual evidence was found for a sphynx cat."
        context["archive_vision_uncertain_count"] = 0
        self.assertTrue(_archive_research_response_needs_recovery(positive, context))
        context["archive_vision_match_count"] = 1
        self.assertFalse(_archive_research_response_needs_recovery(positive, context))

    def test_archive_fallback_states_bounded_vision_scope(self):
        search = {
            "query": "sphynx cat",
            "count": 24,
            "results_returned_to_model": 8,
            "lexical_match_count_in_returned": 1,
            "time_window": {
                "from_local": "2026-08-02T20:00:00+03:00",
                "to_local": "2026-08-03T20:00:00+03:00",
                "duration_sec": 86_400,
            },
            "coverage": {"scanned_candidates": 20_000, "total_candidates": 20_953},
        }
        vision = {
            "source": "archive_candidate_batch",
            "candidate_count": 8,
            "match_count": 1,
            "no_match_count": 6,
            "uncertain_count": 1,
            "parse_status": "parsed",
            "verdicts": [
                {
                    "detection_id": 501,
                    "verdict": "match",
                    "visible_evidence": "A hairless cat is visible on the shelf.",
                }
            ],
        }
        tool_messages = [
            {"role": "tool", "name": "search_archive", "content": __import__("json").dumps(search)},
            {"role": "tool", "name": "describe_frame", "content": __import__("json").dumps(vision)},
        ]

        text = _format_archive_research_fallback(
            {"user_query": "find sphynx cat"},
            tool_messages=tool_messages,
        )

        self.assertIn("Vision batch: reviewed 8", text)
        self.assertIn("#501 — match", text)
        self.assertIn("not proof of absence across the whole archive", text)

    def test_turn_context_carries_time_channel_and_vlm_evidence_defaults(self):
        context = _seed_turn_tool_context(
            "Check video descriptions for Zenbook webcam for the last two hours and confirm with snaps."
        )
        _remember_turn_tool_result(
            "normalize_time_window",
            {
                "from_ts": 100.0,
                "to_ts": 200.0,
                "since_ms": 100_000,
                "until_ms": 200_000,
            },
            context,
        )
        _remember_turn_tool_result(
            "list_video_summary_channels",
            {
                "time_window": {
                    "from_ts": 100.0,
                    "to_ts": 200.0,
                    "since_ms": 100_000,
                    "until_ms": 200_000,
                },
                "candidate_channels": [{"channel_id": 7, "title": "Kitchen"}],
            },
            context,
        )

        summary_args = _apply_turn_tool_context("get_video_summaries", {}, context)
        self.assertEqual(summary_args["channel_id"], 7)
        self.assertEqual(summary_args["from_ts"], 100.0)
        self.assertEqual(summary_args["to_ts"], 200.0)
        self.assertTrue(summary_args["include_evidence_frames"])

        count_args = _apply_turn_tool_context("count_video_summary_events", {"entity_query": "cat"}, context)
        self.assertEqual(count_args["channel_id"], 7)
        self.assertEqual(count_args["from_ts"], 100.0)
        self.assertEqual(count_args["to_ts"], 200.0)

        state_args = _apply_turn_tool_context(
            "track_visual_state_transitions",
            {
                "positive_state_query": "sphynx cat on top of computer tower",
                "negative_state_query": "empty computer tower with no cat",
            },
            context,
        )
        self.assertEqual(state_args["channel_id"], 7)
        self.assertEqual(state_args["from_ts"], 100.0)
        self.assertEqual(state_args["to_ts"], 200.0)

        detection_args = _apply_turn_tool_context("get_detections", {}, context)
        self.assertEqual(detection_args["channel_id"], 7)
        self.assertEqual(detection_args["since_ms"], 100_000)
        self.assertEqual(detection_args["until_ms"], 200_000)
        self.assertEqual(detection_args["source"], "vlm_summary")

    def test_operator_relative_period_overrides_model_invented_calendar_dates(self):
        context = _seed_turn_tool_context(
            "Show channel 112 L1 and L2 summaries for the last 3 days."
        )

        normalize_args = _apply_turn_tool_context(
            "normalize_time_window",
            {
                "date": "2026-03-15",
                "start_time": "05:20",
                "end_time": "08:00",
            },
            context,
        )
        summary_args = _apply_turn_tool_context(
            "get_video_summaries",
            {
                "channel_id": 112,
                "depth": "L1",
                "from_ts": 1_773_550_800,
                "to_ts": 1_773_733_200,
            },
            context,
        )
        report_args = _apply_turn_tool_context(
            "generate_report",
            {
                "channel_id": 112,
                "from_ts": 1,
                "to_ts": 2,
            },
            context,
        )

        self.assertNotIn("date", normalize_args)
        self.assertNotIn("start_time", normalize_args)
        self.assertIn("last 3 days", normalize_args["relative_range"])
        self.assertIn("last 3 days", summary_args["relative_range"])
        self.assertIn("last 3 days", report_args["relative_range"])

        ru_context = _seed_turn_tool_context(
            "Сделай отчёт по каналу 112 за последние 3 дня."
        )
        ru_report_args = _apply_turn_tool_context(
            "generate_report",
            {"channel_id": 112, "from_ts": 1, "to_ts": 2},
            ru_context,
        )
        self.assertIn("последние 3 дня", ru_report_args["relative_range"])

    def test_operator_explicit_calendar_range_is_not_replaced_by_relative_words(self):
        context = _seed_turn_tool_context(
            "Compare 2026-03-15 through 2026-03-17 with the last 3 days."
        )

        self.assertNotIn("operator_relative_range", context)

    def test_russian_runtime_status_uses_current_runtime_fast_path(self):
        context = _seed_turn_tool_context(
            "Покажи активные стримы, модели, очереди, потери и последние ошибки."
        )

        self.assertTrue(context["runtime_status_only"])
        prepared = _apply_turn_tool_context(
            "list_video_summary_channels",
            {"since_hours": 6},
            context,
        )
        self.assertTrue(prepared["runtime_only"])

    def test_turn_context_does_not_override_explicit_source_or_time(self):
        context = _seed_turn_tool_context("Check video descriptions and confirm with images.")
        context["channel_id"] = 7
        context["time_window"] = {
            "from_ts": 100.0,
            "to_ts": 200.0,
            "since_ms": 100_000,
            "until_ms": 200_000,
        }

        args = _apply_turn_tool_context(
            "get_detections",
            {
                "channel_id": 8,
                "source": "probe",
                "since_hours": 24,
            },
            context,
        )

        self.assertEqual(args["channel_id"], 8)
        self.assertEqual(args["source"], "probe")
        self.assertEqual(args["since_hours"], 24)
        self.assertNotIn("since_ms", args)

    def test_turn_context_fills_only_missing_exact_window_bound(self):
        context = _seed_turn_tool_context("continue the same period")
        context["time_window"] = {
            "from_ts": 100.0,
            "to_ts": 200.0,
            "since_ms": 100_000,
            "until_ms": 200_000,
        }

        summary_args = _apply_turn_tool_context(
            "get_video_summaries",
            {"from_ts": 125.0},
            context,
        )
        archive_args = _apply_turn_tool_context(
            "search_archive",
            {"until_ms": 175_000},
            context,
        )
        report_args = _apply_turn_tool_context(
            "generate_report",
            {"to_ts": 180.0},
            context,
        )

        self.assertEqual(summary_args, {"from_ts": 125.0, "to_ts": 200.0})
        self.assertEqual(archive_args["since_ms"], 100_000)
        self.assertEqual(archive_args["until_ms"], 175_000)
        self.assertEqual(report_args, {"from_ts": 100.0, "to_ts": 180.0})

    def test_turn_context_does_not_convert_detection_describe_frame_to_live_snapshot(self):
        context = _seed_turn_tool_context("Confirm this frame with images.")
        context["channel_id"] = 7

        args = _apply_turn_tool_context(
            "describe_frame",
            {"detection_id": 501},
            context,
        )

        self.assertEqual(args["detection_id"], 501)
        self.assertNotIn("channel_id", args)

    def test_archive_search_passes_source_and_labels_vlm_summary_results(self):
        captured = {}

        def search_detections(**kwargs):
            captured.update(kwargs)
            return [
                {
                    "detection_id": 101,
                    "timestamp_ms": 1781389900000,
                    "source": "vlm_summary",
                    "probe_name": "VLM summary frame",
                    "channel_id": 7,
                    "score": 0.81,
                }
            ]

        result = _tools(search_detections_fn=search_detections).execute(
            "search_archive",
            {
                "query": "person at desk",
                "scope": "detections",
                "source": "vlm_summary",
                "channel_id": 7,
            },
        )

        self.assertEqual(captured["source"], "vlm_summary")
        self.assertEqual(result["source"], "vlm_summary")
        self.assertEqual(result["source_label"], "Video-description frame")
        self.assertEqual(result["results"][0]["archive_item_type"], "video_description_frame")

    def test_archive_search_compaction_preserves_coverage_and_similarity_alias(self):
        def search_detections(**kwargs):
            return {
                "coverage": {
                    "status": "partial",
                    "candidate_count": 20_000,
                    "truncated": True,
                },
                "results": [
                    {
                        "detection_id": 101,
                        "timestamp_ms": 1781389900000,
                        "source": "vlm_summary",
                        "channel_id": 7,
                        "similarity": 0.81,
                    }
                ],
            }

        result = _tools(search_detections_fn=search_detections).execute(
            "search_archive",
            {
                "query": "vehicle drift",
                "scope": "detections",
                "source": "vlm_summary",
                "channel_id": 7,
            },
        )
        compact = _compact_tool_result_for_model("search_archive", result)

        self.assertEqual(compact["coverage"]["status"], "partial")
        self.assertTrue(compact["coverage"]["truncated"])
        self.assertAlmostEqual(compact["results"][0]["score"], 0.81)
        self.assertAlmostEqual(compact["results"][0]["similarity"], 0.81)
        ledger = _new_turn_signal_ledger("find drift")
        _record_turn_signal_ledger(ledger, "search_archive", compact)
        self.assertEqual(ledger["evidence"][0]["coverage"]["status"], "partial")
        self.assertAlmostEqual(ledger["evidence"][0]["best_similarity"], 0.81)

    def test_visual_window_signals_returns_pnm_attention_signal(self):
        calls = []

        def search_detections(**kwargs):
            calls.append(dict(kwargs))
            query = kwargs.get("query")
            source = kwargs.get("source")
            if source == "vlm_alert" and query == "cat on top of black PC tower":
                return [
                    {
                        "id": 601,
                        "detection_id": 601,
                        "timestamp_ms": 1781390100000,
                        "source": "vlm_alert",
                        "channel_id": 7,
                        "similarity": 0.34,
                        "thumbnail": "abc123",
                    }
                ]
            if source == "vlm_alert" and query == "empty PC tower with no cat":
                return [
                    {
                        "id": 601,
                        "detection_id": 601,
                        "timestamp_ms": 1781390100000,
                        "source": "vlm_alert",
                        "channel_id": 7,
                        "similarity": 0.12,
                        "thumbnail": "abc123",
                    }
                ]
            if source == "vlm_summary" and query == "cat on top of black PC tower":
                return [
                    {
                        "id": 701,
                        "detection_id": 701,
                        "timestamp_ms": 1781390200000,
                        "source": "vlm_summary",
                        "channel_id": 7,
                        "similarity": 0.28,
                        "thumbnail": "def456",
                    }
                ]
            if source == "vlm_summary" and query == "empty PC tower with no cat":
                return [
                    {
                        "id": 702,
                        "detection_id": 702,
                        "timestamp_ms": 1781390300000,
                        "source": "vlm_summary",
                        "channel_id": 7,
                        "similarity": 0.30,
                        "thumbnail": "ghi789",
                    }
                ]
            return []

        result = _tools(search_detections_fn=search_detections).execute(
            "get_visual_window_signals",
            {
                "channel_id": 7,
                "positive_query": "cat on top of black PC tower",
                "negative_query": "empty PC tower with no cat",
                "since_ms": 1781389800000,
                "until_ms": 1781390400000,
            },
        )

        self.assertEqual(result["sources"], ["vlm_alert", "vlm_summary"])
        self.assertEqual(len(calls), 4)
        self.assertEqual(result["score_semantics"], "clip_retrieval_signal_not_proof")
        self.assertAlmostEqual(result["pnm"]["p"], 0.34)
        self.assertAlmostEqual(result["pnm"]["n"], 0.30)
        self.assertAlmostEqual(result["pnm"]["m"], 0.04)
        self.assertEqual(result["candidate_frames"][0]["detection_id"], 601)
        self.assertAlmostEqual(result["candidate_frames"][0]["positive_score"], 0.34)
        self.assertAlmostEqual(result["candidate_frames"][0]["negative_score"], 0.12)
        self.assertAlmostEqual(result["candidate_frames"][0]["margin"], 0.22)
        self.assertTrue(result["candidate_frames"][0]["needs_describe_frame"])

        compact = _compact_tool_result_for_model("get_visual_window_signals", result)
        self.assertEqual(compact["pnm"]["score_semantics"], "clip_retrieval_signal_not_proof")
        self.assertEqual(compact["candidate_frames"][0]["detection_id"], 601)
        self.assertIn("image_url", compact["candidate_frames"][0])

    def test_normalize_time_window_last_night_returns_seconds_and_milliseconds(self):
        tools = _tools()
        result = tools.execute(
            "normalize_time_window",
            {
                "date": "2026-06-14",
                "start_time": "01:30",
                "end_time": "08:30",
                "timezone": "Europe/Riga",
            },
        )

        self.assertEqual(result["from_ts"], 1781389800)
        self.assertEqual(result["to_ts"], 1781415000)
        self.assertEqual(result["since_ms"], 1781389800000)
        self.assertEqual(result["until_ms"], 1781415000000)

    def test_configured_timezone_is_default_and_formats_operator_timestamps_consistently(self):
        with patch("agent.AGENT_SITE_TIMEZONE", "Etc/GMT-4"):
            result = _tools().execute(
                "normalize_time_window",
                {
                    "date": "2026-06-14",
                    "start_time": "01:30",
                    "end_time": "08:30",
                },
            )

            self.assertEqual(result["timezone"], "Etc/GMT-4")
            self.assertEqual(result["from_local"], "2026-06-14T01:30:00+04:00")
            self.assertEqual(_format_epoch_minute(result["from_ts"]), "2026-06-14 01:30")
            self.assertEqual(_format_epoch_minute(result["to_ts"]), "2026-06-14 08:30")

    def test_normalize_time_window_accepts_relative_last_two_hours(self):
        tools = _tools()
        result = tools.execute(
            "normalize_time_window",
            {
                "relative_range": "last two hours",
                "timezone": "Europe/Riga",
            },
        )

        self.assertEqual(result["duration_sec"], 7200)
        self.assertEqual(result["day_hint"], "relative")
        self.assertEqual(result["to_ts"] - result["from_ts"], 7200)
        self.assertEqual(result["until_ms"] - result["since_ms"], 7_200_000)

    def test_normalize_time_window_accepts_relative_last_day_as_rolling_24h(self):
        tools = _tools()
        result = tools.execute(
            "normalize_time_window",
            {
                "relative_range": "last day",
                "timezone": "Europe/Riga",
            },
        )

        self.assertEqual(result["duration_sec"], 86_400)
        self.assertEqual(result["day_hint"], "relative")
        self.assertEqual(result["relative_range"], "last day")
        self.assertEqual(result["to_ts"] - result["from_ts"], 86_400)
        self.assertEqual(result["until_ms"] - result["since_ms"], 86_400_000)

    def test_normalize_time_window_accepts_relative_last_week_as_rolling_7_days(self):
        tools = _tools()
        result = tools.execute(
            "normalize_time_window",
            {
                "relative_range": "last week",
                "timezone": "Europe/Riga",
            },
        )

        self.assertEqual(result["duration_sec"], 604_800)
        self.assertEqual(result["day_hint"], "relative")
        self.assertEqual(result["relative_range"], "last week")
        self.assertEqual(result["to_ts"] - result["from_ts"], 604_800)
        self.assertEqual(result["until_ms"] - result["since_ms"], 604_800_000)

    def test_get_video_summaries_resolves_last_three_days_on_server_clock(self):
        tools = _tools()
        fixed_now = 1_783_900_000.0

        with patch("agent.time.time", return_value=fixed_now):
            result = tools.execute(
                "get_video_summaries",
                {
                    "channel_id": 7,
                    "depth": "L1",
                    "relative_range": "last 3 days",
                },
            )

        self.assertEqual(result["time_window"]["to_ts"], fixed_now)
        self.assertEqual(result["time_window"]["from_ts"], fixed_now - 259_200)
        self.assertEqual(result["time_window"]["duration_sec"], 259_200)
        self.assertEqual(
            result["time_window"]["window_source"],
            "operator_relative_range",
        )
        self.assertEqual(
            result["coverage"]["available"]["requested_span_sec"],
            259_200,
        )

    def test_summary_restore_defaults_to_durable_l2_l3_preview_for_two_weeks(self):
        manager = _SummaryManager()
        captured = {}

        def plan_rollup_backfill(**kwargs):
            captured.update(kwargs)
            return {
                "status": "preview",
                "channel_ids": [7, 8],
                "channel_count": 2,
                "levels": list(kwargs["levels"]),
                "totals": {
                    "source_windows": 30,
                    "already_ready": 4,
                    "missing_semantic": 26,
                },
                "estimated_hours": 1.25,
                "estimated_hours_range": [0.75, 2.25],
                "load_policy": "single background worker",
            }

        manager.plan_rollup_backfill = plan_rollup_backfill
        fixed_now = 1_783_900_000.0
        with patch("agent.time.time", return_value=fixed_now):
            result = _tools(manager).execute(
                "restore_video_summary_history",
                {
                    "channel_ids": [7, 8],
                    "relative_range": "last two weeks",
                    "preview": True,
                },
            )

        self.assertEqual(captured["levels"], ["L2", "L3"])
        self.assertEqual(captured["start_ts"], fixed_now - 14 * 86400)
        self.assertEqual(captured["end_ts"], fixed_now)
        self.assertEqual(result["time_window"]["duration_sec"], 14 * 86400)
        self.assertTrue(result["preview"])
        self.assertIn("without another command", result["operator_action"])

        compact = _compact_tool_result_for_model("restore_video_summary_history", result)
        self.assertEqual(compact["restoration_scope"]["queueable_windows"], 26)
        self.assertIn("not queued work", compact["restoration_scope"]["queue_contract"])

    def test_summary_restore_status_reads_durable_worker_state(self):
        manager = _SummaryManager()
        manager.rollup_backfill_status = lambda: {
            "status": "running",
            "job_id": "rollup-backfill-1",
            "progress_percent": 42.5,
            "eta_hours": 3.2,
            "durable": True,
        }

        result = _tools(manager).execute("get_video_summary_restore_status", {})

        self.assertEqual(result["status"], "running")
        self.assertEqual(result["job_id"], "rollup-backfill-1")
        self.assertEqual(result["eta_hours"], 3.2)
        self.assertTrue(result["durable"])

    def test_normalize_time_window_accepts_date_without_clock_as_calendar_day(self):
        tools = _tools()
        result = tools.execute(
            "normalize_time_window",
            {
                "date": "2026-06-23",
                "timezone": "Europe/Riga",
            },
        )

        self.assertEqual(result["from_local"], "2026-06-23T00:00:00+03:00")
        self.assertEqual(result["to_local"], "2026-06-24T00:00:00+03:00")
        self.assertEqual(result["duration_sec"], 86_400)

    def test_normalize_time_window_accepts_relative_phrase_in_start_time(self):
        tools = _tools()
        result = tools.execute(
            "normalize_time_window",
            {
                "start_time": "last two hours",
                "end_time": "now",
                "timezone": "Europe/Riga",
            },
        )

        self.assertEqual(result["duration_sec"], 7200)
        self.assertEqual(result["relative_range"], "last two hours")

    def test_normalize_time_window_without_period_is_non_error_status(self):
        tools = _tools()
        result = tools.execute("normalize_time_window", {"timezone": "Europe/Riga"})

        self.assertEqual(result["status"], "not_specified")
        self.assertFalse(result["has_time_window"])
        self.assertEqual(result["timezone"], "Europe/Riga")

    def test_get_video_summaries_accepts_milliseconds_and_filters_window(self):
        manager = _SummaryManager()
        captured = {}

        def epoch_rollups(channel_id, run_selector=None, start_ts=None, end_ts=None, level_limit=None):
            captured["run_selector"] = run_selector
            return {
                "running": False,
                "selected_run": run_selector,
                "run_filter_id": None,
                "levels": {
                    "L0": [],
                    "L1": [
                        {
                            "level": "L1",
                            "window_start": 1781389700.0,
                            "window_end": 1781389799.0,
                            "summary": "before",
                        },
                        {
                            "level": "L1",
                            "window_start": 1781389900.0,
                            "window_end": 1781390100.0,
                            "summary": "inside",
                            "frame_count": 3,
                        },
                        {
                            "level": "L1",
                            "window_start": 1781392000.0,
                            "window_end": 1781392100.0,
                            "summary": "after",
                        },
                    ],
                    "L2": [],
                    "L3": [],
                },
            }

        manager.summary_rollups = epoch_rollups
        result = _tools(manager).execute(
            "get_video_summaries",
            {
                "channel_id": 7,
                "depth": "L1",
                "from_ts": 1_781_389_800_000,
                "to_ts": 1_781_391_000_000,
                "limit": 10,
            },
        )

        self.assertEqual(result["time_window"]["normalized_input_units"]["from_ts"], "milliseconds")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["total_in_window"], 1)
        self.assertEqual(result["entries"][0]["summary"], "inside")
        self.assertEqual(captured["run_selector"], "all")
        self.assertEqual(result["selected_run"], "all")
        self.assertEqual(result["coverage"]["status"], "partial")
        self.assertEqual(result["coverage"]["returned"]["first_ts"], 1781389900.0)
        self.assertTrue(result["coverage"]["must_state_coverage"])

    def test_summary_tools_request_only_the_selected_rollup_depth(self):
        manager = _SummaryManager()
        tools = _tools(manager)

        tools.execute(
            "get_video_summaries",
            {
                "channel_id": 7,
                "depth": "L1",
                "from_ts": 100.0,
                "to_ts": 300.0,
            },
        )
        tools.execute(
            "count_video_summary_events",
            {
                "channel_id": 7,
                "depth": "L1",
                "entity_query": "person",
                "from_ts": 100.0,
                "to_ts": 300.0,
            },
        )

        self.assertEqual(len(manager.summary_rollup_requests), 2)
        self.assertTrue(all(row["target_level"] == "L1" for row in manager.summary_rollup_requests))
        self.assertTrue(all(row["synthesize"] is False for row in manager.summary_rollup_requests))

    def test_get_video_summaries_samples_across_period_when_truncated(self):
        manager = _SummaryManager()

        def long_rollups(channel_id, run_selector=None, start_ts=None, end_ts=None, level_limit=None):
            nodes = []
            for index in range(10):
                node = {
                    "level": "L1",
                    "window_start": 100.0 + index * 60.0,
                    "window_end": 130.0 + index * 60.0,
                    "summary": f"window-{index}",
                    "frame_count": 12,
                }
                if index == 5:
                    node["summary"] = "window-5 drifting alert"
                    node["alert_counts"] = {"high": 1}
                    node["alert_total"] = 1
                nodes.append(node)
            return {
                "running": False,
                "selected_run": None,
                "run_filter_id": None,
                "levels": {"L0": [], "L1": nodes, "L2": [], "L3": []},
            }

        manager.summary_rollups = long_rollups
        result = _tools(manager).execute(
            "get_video_summaries",
            {
                "channel_id": 7,
                "depth": "L1",
                "from_ts": 100.0,
                "to_ts": 700.0,
                "limit": 4,
            },
        )

        self.assertEqual(result["selection_strategy"], "period_sample_alert_priority")
        self.assertTrue(result["truncated"])
        self.assertEqual(result["coverage"]["selection_strategy"], "period_sample_alert_priority")
        self.assertEqual(
            [entry["summary"] for entry in result["entries"]],
            ["window-0", "window-3", "window-5 drifting alert", "window-9"],
        )
        self.assertEqual(result["coverage"]["returned"]["first_ts"], 100.0)
        self.assertEqual(result["coverage"]["returned"]["last_ts"], 670.0)

    def test_get_video_summaries_keeps_completed_semantics_in_period_sample(self):
        manager = _SummaryManager()

        def rollups(channel_id, run_selector=None, start_ts=None, end_ts=None, level_limit=None):
            nodes = [
                {
                    "level": "L1",
                    "window_start": 100.0 + index * 60.0,
                    "window_end": 130.0 + index * 60.0,
                    "summary": f"queued-{index}",
                    "summary_kind": "queued",
                    "generation_status": "queued",
                    "frame_count": 12,
                }
                for index in range(10)
            ]
            nodes[4].update(
                {
                    "summary": "full semantic behavior narrative",
                    "summary_kind": "llm",
                    "generation_status": "ready",
                }
            )
            return {
                "running": False,
                "selected_run": None,
                "run_filter_id": None,
                "levels": {"L0": [], "L1": nodes, "L2": [], "L3": []},
            }

        manager.summary_rollups = rollups
        result = _tools(manager).execute(
            "get_video_summaries",
            {
                "channel_id": 7,
                "depth": "L1",
                "from_ts": 100.0,
                "to_ts": 700.0,
                "limit": 3,
            },
        )

        self.assertEqual(
            result["selection_strategy"],
            "period_sample_semantic_alert_priority",
        )
        semantic = [
            row for row in result["entries"]
            if row.get("summary_kind") == "llm"
        ]
        self.assertEqual(len(semantic), 1)
        self.assertEqual(semantic[0]["summary"], "full semantic behavior narrative")
        self.assertEqual(semantic[0]["generation_status"], "ready")

    def test_get_video_summaries_uses_high_bounded_scan_limit_by_default(self):
        manager = _SummaryManager()
        captured = {}

        def rollups(channel_id, run_selector=None, start_ts=None, end_ts=None, level_limit=None):
            captured["level_limit"] = level_limit
            nodes = [
                {
                    "level": "L1",
                    "window_start": 100.0 + index * 10.0,
                    "window_end": 105.0 + index * 10.0,
                    "summary": f"node-{index}",
                }
                for index in range(5)
            ]
            return {
                "running": False,
                "selected_run": None,
                "run_filter_id": None,
                "level_limit": level_limit,
                "source_counts": {"L0": 0, "L1": level_limit, "L2": 0, "L3": 0},
                "levels": {"L0": [], "L1": nodes, "L2": [], "L3": []},
            }

        manager.summary_rollups = rollups
        result = _tools(manager).execute(
            "get_video_summaries",
            {
                "channel_id": 7,
                "depth": "L1",
                "from_ts": 100.0,
                "to_ts": 200.0,
                "limit": 2,
            },
        )

        self.assertEqual(captured["level_limit"], 500)
        self.assertEqual(result["display_limit"], 2)
        self.assertEqual(result["level_limit_applied"], 500)
        self.assertEqual(result["source_counts"]["L1"], 500)
        self.assertTrue(result["backend_truncated"])
        self.assertTrue(result["truncated"])

        compact = _compact_tool_result_for_model("get_video_summaries", result)
        self.assertEqual(compact["level_limit_applied"], 500)
        self.assertEqual(compact["source_counts"]["L1"], 500)
        self.assertTrue(compact["backend_truncated"])

    def test_get_video_summaries_compacts_provenance_bundle(self):
        manager = _SummaryManager()

        def rollups(channel_id, run_selector=None, start_ts=None, end_ts=None, level_limit=None):
            return {
                "running": False,
                "selected_run": None,
                "run_filter_id": None,
                "levels": {
                    "L0": [
                        {
                            "level": "L0",
                            "window_start": 100.0,
                            "window_end": 112.0,
                            "summary": "Prose says a public safety event happened.",
                            "frame_count": 12,
                            "alert_counts": {"high": 1},
                            "alert_total": 1,
                            "alert_parser_breakdown": {
                                "parser_alert_count": 2,
                                "json_alert_count": 1,
                                "prose_alert_count": 2,
                                "prose_only_signal_count": 1,
                            },
                            "alert_delivery_breakdown": {
                                "cooldown_skipped": 1,
                                "total": 1,
                            },
                            "alert_events": [
                                {
                                    "title": "Person down",
                                    "description": "Person lying on ground near entrance.",
                                    "severity": "high",
                                    "delivery_status": "cooldown_skipped",
                                    "timestamp_ms": 108000,
                                }
                            ],
                            "state_observations": [
                                {
                                    "key": "person near entrance",
                                    "label": "Person near entrance",
                                    "state": "present",
                                    "evidence": "visible near entrance",
                                }
                            ],
                            "state_transition_events": [
                                {
                                    "key": "person near entrance",
                                    "label": "Person near entrance",
                                    "event_type": "appearance",
                                    "from_state": "absent",
                                    "to_state": "present",
                                    "timestamp_ms": 108000,
                                    "evidence": "current observed state",
                                }
                            ],
                            "state_transition_total": 1,
                        }
                    ],
                    "L1": [],
                    "L2": [],
                    "L3": [],
                },
            }

        manager.summary_rollups = rollups
        result = _tools(manager).execute(
            "get_video_summaries",
            {
                "channel_id": 7,
                "depth": "L0",
                "from_ts": 90.0,
                "to_ts": 120.0,
                "limit": 5,
            },
        )

        self.assertEqual(result["provenance_totals"]["unconfirmed_prose_signal_count"], 1)
        entry = result["entries"][0]
        self.assertEqual(entry["alert_events"][0]["delivery_status"], "cooldown_skipped")
        self.assertEqual(entry["state_observations"][0]["state"], "present")
        self.assertEqual(entry["state_transition_events"][0]["event_type"], "appearance")
        self.assertEqual(entry["unconfirmed_prose_signal_count"], 1)

        compact = _compact_tool_result_for_model("get_video_summaries", result)
        compact_entry = compact["entries"][0]
        self.assertEqual(compact["provenance_totals"]["alert_delivery_breakdown"]["cooldown_skipped"], 1)
        self.assertEqual(compact_entry["alert_parser_breakdown"]["prose_only_signal_count"], 1)
        self.assertEqual(compact_entry["alert_events"][0]["delivery_status"], "cooldown_skipped")
        self.assertEqual(compact_entry["state_transition_events"][0]["event_type"], "appearance")

    def test_get_video_summaries_returns_evidence_frames_from_vlm_archive(self):
        rows = [
            {
                "id": 501,
                "timestamp_ms": 150_000,
                "recorded_at_ms": 150_000,
                "probe_id": "vlm_summary:7",
                "probe_name": "VLM summary ch 7",
                "channel_id": 7,
                "severity": "info",
                "bookmark_enabled": False,
                "bookmark_sent": False,
                "pos_score": 0.0,
                "neg_score": 0.0,
                "margin": 0.0,
                "thumbnail": "abc123",
                "source": "vlm_summary",
                "payload": {
                    "batch_start_ms": 140_000,
                    "batch_end_ms": 160_000,
                    "frame_timestamp_ms": 150_000,
                    "frame_index": 1,
                    "anchor_role": "sample",
                    "summary": "cat returned to the PC tower",
                },
            },
            {
                "id": 502,
                "timestamp_ms": 151_000,
                "recorded_at_ms": 151_000,
                "probe_id": "vlm_alert:7",
                "probe_name": "VLM alert ch 7",
                "channel_id": 7,
                "severity": "normal",
                "bookmark_enabled": False,
                "bookmark_sent": False,
                "pos_score": 0.0,
                "neg_score": 0.0,
                "margin": 0.0,
                "thumbnail": "def456",
                "source": "vlm_alert",
                "payload": {"batch_start_ms": 140_000, "batch_end_ms": 160_000, "alert_total": 1},
            },
            {
                "id": 503,
                "timestamp_ms": 152_000,
                "recorded_at_ms": 152_000,
                "probe_id": "vlm_summary:7",
                "probe_name": "VLM summary ch 7",
                "channel_id": 7,
                "severity": "info",
                "pos_score": 0.0,
                "neg_score": 0.0,
                "margin": 0.0,
                "thumbnail": "ghi789",
                "source": "vlm_summary",
            },
            {
                "id": 504,
                "timestamp_ms": 153_000,
                "recorded_at_ms": 153_000,
                "probe_id": "vlm_summary:7",
                "probe_name": "VLM summary ch 7",
                "channel_id": 7,
                "severity": "info",
                "pos_score": 0.0,
                "neg_score": 0.0,
                "margin": 0.0,
                "thumbnail": "jkl012",
                "source": "vlm_summary",
            },
        ]
        result = _tools(detections_store=_DetectionStore(rows)).execute(
            "get_video_summaries",
            {
                "channel_id": 7,
                "depth": "L1",
                "from_ts": 100.0,
                "to_ts": 300.0,
                "include_evidence_frames": True,
                "evidence_frame_limit": 2,
            },
        )

        self.assertEqual(result["evidence_frame_query"]["sources"], ["vlm_alert", "vlm_summary"])
        self.assertEqual(
            [query["source"] for query in result["evidence_frame_queries"]],
            ["vlm_alert", "vlm_summary"],
        )
        self.assertEqual(result["attempted_sources"], ["vlm_alert", "vlm_summary"])
        self.assertEqual(result["evidence_frame_totals"]["vlm_summary"], 3)
        self.assertEqual(result["evidence_frame_totals"]["vlm_alert"], 1)
        self.assertEqual(result["totals"]["vlm_alert"], 1)
        self.assertEqual(len(result["evidence_frames"]), 2)
        self.assertEqual(
            {row["source"] for row in result["evidence_frames"]},
            {"vlm_summary", "vlm_alert"},
        )
        self.assertTrue(result["evidence_frames"][0]["image_url"].startswith("/detections/thumbnail/"))

        compact = _compact_tool_result_for_model("get_video_summaries", result)
        self.assertEqual(compact["coverage"]["status"], "partial")
        self.assertEqual(compact["evidence_frames"][0]["payload"]["batch_start_ms"], 140_000)
        self.assertEqual(compact["evidence_frames"][0]["detection_id"], 501)
        self.assertEqual(compact["evidence_frames"][0]["score_semantics"], "not_applicable")
        self.assertIn("image_url", compact["evidence_frames"][0])

    def test_get_video_summaries_samples_period_anchors_and_alert_window(self):
        manager = _SummaryManager()

        def rollups(channel_id, run_selector=None, start_ts=None, end_ts=None, level_limit=None):
            return {
                "running": False,
                "levels": {
                    "L0": [],
                    "L1": [
                        {"level": "L1", "window_start": 100.0, "window_end": 110.0, "summary": "routine traffic"},
                        {
                            "level": "L1",
                            "window_start": 490.0,
                            "window_end": 510.0,
                            "summary": "deviation: vehicle drift",
                            "alert_total": 1,
                        },
                        {"level": "L1", "window_start": 890.0, "window_end": 900.0, "summary": "routine traffic"},
                    ],
                    "L2": [],
                    "L3": [],
                },
            }

        manager.summary_rollups = rollups
        rows = [
            {
                "id": index,
                "detection_id": index,
                "timestamp_ms": (100 + index * 100) * 1000,
                "source": "vlm_summary",
                "channel_id": 7,
            }
            for index in range(9)
        ]
        rows.append({
            "id": 50,
            "detection_id": 50,
            "timestamp_ms": 500_000,
            "source": "vlm_alert",
            "channel_id": 7,
        })

        result = _tools(manager, detections_store=_DetectionStore(rows)).execute(
            "get_video_summaries",
            {
                "channel_id": 7,
                "depth": "L1",
                "from_ts": 100.0,
                "to_ts": 1_000.0,
                "include_evidence_frames": True,
                "evidence_frame_limit": 3,
            },
        )

        self.assertEqual(result["evidence_selection_strategy"], "period_span_alert_priority")
        self.assertEqual(result["evidence_priority_windows"][0]["since_ms"], 490_000)
        self.assertEqual(
            [row["detection_id"] for row in result["evidence_frames"]],
            [0, 50, 8],
        )
        compact = _compact_tool_result_for_model("get_video_summaries", result)
        self.assertEqual(compact["evidence_selection_strategy"], "period_span_alert_priority")
        self.assertEqual(compact["evidence_priority_windows"][0]["until_ms"], 510_000)

    def test_count_video_summary_events_counts_presence_transitions_with_coverage(self):
        manager = _SummaryManager()
        captured = {}

        def rollups(channel_id, run_selector=None, start_ts=None, end_ts=None, level_limit=None):
            captured["run_selector"] = run_selector
            nodes = [
                {
                    "level": "L1",
                    "window_start": 100.0,
                    "window_end": 200.0,
                    "summary": "The white van remains absent from the loading bay. No vehicles are visible.",
                },
                {
                    "level": "L1",
                    "window_start": 200.0,
                    "window_end": 300.0,
                    "summary": "The white van is stationary inside the loading bay.",
                },
                {
                    "level": "L1",
                    "window_start": 300.0,
                    "window_end": 400.0,
                    "summary": "The white van leaves the loading bay and exits the frame to the right.",
                },
                {
                    "level": "L1",
                    "window_start": 400.0,
                    "window_end": 500.0,
                    "summary": "The loading bay is static. The white van remains absent from the area.",
                },
            ]
            return {
                "running": False,
                "selected_run": None,
                "run_filter_id": None,
                "levels": {"L0": [], "L1": nodes, "L2": [], "L3": []},
            }

        manager.summary_rollups = rollups
        result = _tools(manager).execute(
            "count_video_summary_events",
            {
                "channel_id": 7,
                "entity_query": "white van",
                "anchor_query": "loading bay",
                "depth": "L1",
                "from_ts": 100.0,
                "to_ts": 500.0,
            },
        )

        self.assertEqual(result["total_in_window"], 4)
        self.assertEqual(result["coverage"]["status"], "covered")
        self.assertEqual(result["counts"]["appearance_count"], 1)
        self.assertEqual(result["counts"]["disappearance_count"], 1)
        self.assertEqual(result["counts"]["inferred_appearance_count"], 1)
        self.assertEqual(result["counts"]["explicit_disappearance_count"], 1)
        self.assertEqual(captured["run_selector"], "all")
        self.assertEqual(
            [(row["type"], row["basis"]) for row in result["transition_events"]],
            [
                ("appearance", "inferred_adjacent_summary_state_change"),
                ("disappearance", "explicit_summary_mention"),
            ],
        )
        self.assertEqual([row["state"] for row in result["timeline"]], ["absent", "present", "absent", "absent"])

        compact = _compact_tool_result_for_model("count_video_summary_events", result)
        self.assertEqual(compact["counts"]["appearance_count"], 1)
        self.assertEqual(compact["counts"]["disappearance_count"], 1)
        self.assertEqual(compact["coverage"]["status"], "covered")
        self.assertEqual(compact["transition_events"][0]["type"], "appearance")
        self.assertEqual(compact["timeline_total"], 4)

    def test_count_video_summary_events_does_not_treat_visible_state_as_appearance(self):
        manager = _SummaryManager()

        def rollups(channel_id, run_selector=None, start_ts=None, end_ts=None, level_limit=None):
            return {
                "running": False,
                "selected_run": run_selector,
                "run_filter_id": None,
                "levels": {
                    "L0": [],
                    "L1": [
                        {
                            "level": "L1",
                            "window_start": 100.0,
                            "window_end": 200.0,
                            "summary": (
                                "The Sphynx cat Orlandina appears calm and remains stationary "
                                "atop the PC tower for the whole window."
                            ),
                        }
                    ],
                    "L2": [],
                    "L3": [],
                },
            }

        manager.summary_rollups = rollups
        result = _tools(manager).execute(
            "count_video_summary_events",
            {
                "channel_id": 7,
                "entity_query": "sphynx cat",
                "anchor_query": "computer tower",
                "depth": "L1",
                "from_ts": 100.0,
                "to_ts": 200.0,
            },
        )

        self.assertEqual(result["counts"]["appearance_count"], 0)
        self.assertEqual(result["counts"]["disappearance_count"], 0)
        self.assertEqual(result["timeline"][0]["state"], "present")

    def test_track_visual_state_transitions_counts_clip_state_boundaries(self):
        store = _DetectionStore(
            [
                {
                    "id": 1,
                    "channel_id": 7,
                    "source": "vlm_summary",
                    "event_timestamp_ms": 100_000,
                    "clip_vec": [0.0, 1.0],
                    "thumbnail": "unused-negative-1",
                },
                {
                    "id": 2,
                    "channel_id": 7,
                    "source": "vlm_summary",
                    "event_timestamp_ms": 101_000,
                    "clip_vec": [0.0, 1.0],
                    "thumbnail": "unused-negative-2",
                },
                {
                    "id": 3,
                    "channel_id": 7,
                    "source": "vlm_summary",
                    "event_timestamp_ms": 102_000,
                    "clip_vec": [1.0, 0.0],
                    "thumbnail": "unused-positive-1",
                },
                {
                    "id": 4,
                    "channel_id": 7,
                    "source": "vlm_summary",
                    "event_timestamp_ms": 103_000,
                    "clip_vec": [1.0, 0.0],
                    "thumbnail": "unused-positive-2",
                },
                {
                    "id": 5,
                    "channel_id": 7,
                    "source": "vlm_summary",
                    "event_timestamp_ms": 104_000,
                    "clip_vec": [0.0, 1.0],
                    "thumbnail": "unused-negative-3",
                },
                {
                    "id": 6,
                    "channel_id": 7,
                    "source": "vlm_summary",
                    "event_timestamp_ms": 105_000,
                    "clip_vec": [0.0, 1.0],
                    "thumbnail": "unused-negative-4",
                },
            ]
        )

        def embed_text(text):
            value = str(text).lower()
            if "empty" in value or "no cat" in value:
                return [0.0, 1.0]
            return [1.0, 0.0]

        result = _tools(detections_store=store, embed_text_fn=embed_text).execute(
            "track_visual_state_transitions",
            {
                "channel_id": 7,
                "subject_query": "Orlandina",
                "positive_state_query": "sphynx cat on top of computer tower",
                "negative_state_query": "empty computer tower with no cat",
                "positive_label": "present_on_tower",
                "negative_label": "absent_from_tower",
                "sources": ["vlm_summary"],
                "from_ts": 100.0,
                "to_ts": 106.0,
                "min_state_samples": 2,
            },
        )

        self.assertEqual(result["frame_count"], 6)
        self.assertEqual(result["negative_state_query"], "empty computer tower with no cat")
        self.assertEqual(result["negative_state_query_effective"], "empty computer tower")
        self.assertTrue(any("negated target terms" in warning for warning in result["warnings"]))
        self.assertEqual(result["coverage"]["status"], "covered")
        self.assertEqual(result["source_totals"], {"vlm_summary": 6})
        self.assertEqual(result["counts"]["appearance_count"], 1)
        self.assertEqual(result["counts"]["disappearance_count"], 1)
        self.assertEqual(
            [(row["type"], row["from_state"], row["to_state"]) for row in result["transitions"]],
            [
                ("appearance", "absent_from_tower", "present_on_tower"),
                ("disappearance", "present_on_tower", "absent_from_tower"),
            ],
        )
        self.assertEqual([row["state"] for row in result["segments"]], ["absent_from_tower", "present_on_tower", "absent_from_tower"])
        self.assertTrue(all(frame.get("image_url", "").startswith("/detections/thumbnail/") for frame in result["boundary_frames"]))
        self.assertTrue(all("thumbnail" not in frame for frame in result["boundary_frames"]))
        self.assertEqual(result["candidate_frames"][0]["state"], "present_on_tower")
        self.assertEqual(result["candidate_frames"][0]["positive_score"], 1.0)

        compact = _compact_tool_result_for_model("track_visual_state_transitions", result)
        self.assertEqual(compact["counts"]["appearance_count"], 1)
        self.assertEqual(compact["coverage"]["status"], "covered")
        self.assertEqual(compact["negative_state_query_effective"], "empty computer tower")
        self.assertEqual(compact["boundary_frames"][0]["image_url"], "/detections/thumbnail/2")
        self.assertEqual(compact["candidate_frames"][0]["image_url"], "/detections/thumbnail/4")
        self.assertEqual(compact["transitions"][0]["after_frame"]["image_url"], "/detections/thumbnail/3")

    def test_calibrate_probe_from_archive_batches_channels_and_suggests_thresholds(self):
        rows = []
        for idx, vec in enumerate(
            (
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.8, 0.2],
                [0.2, 0.8],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ),
            start=1,
        ):
            rows.append(
                {
                    "id": idx,
                    "channel_id": 7,
                    "source": "vlm_summary",
                    "event_timestamp_ms": 100_000 + idx * 1_000,
                    "clip_vec": vec,
                    "thumbnail": f"thumb-{idx}",
                }
            )
        rows.append(
            {
                "id": 20,
                "channel_id": 8,
                "source": "vlm_summary",
                "event_timestamp_ms": 101_000,
                "clip_vec": [0.0, 1.0],
                "thumbnail": "other-channel",
            }
        )

        def embed_text(text):
            value = str(text).lower()
            if "walking normally" in value or "clear sidewalk" in value:
                return [0.0, 1.0]
            return [1.0, 0.0]

        result = _tools(detections_store=_DetectionStore(rows), embed_text_fn=embed_text).execute(
            "calibrate_probe_from_archive",
            {
                "event_query": "two people fighting",
                "contrast_query": "people walking normally on clear sidewalk",
                "channel_ids": [7, 8, 9],
                "sources": ["vlm_summary"],
                "from_ts": 100.0,
                "to_ts": 120.0,
                "max_channels_per_call": 2,
                "candidate_limit": 100,
                "evidence_limit": 6,
            },
        )

        self.assertEqual(result["processed_channel_ids"], [7, 8])
        self.assertEqual(result["deferred_channel_ids"], [9])
        self.assertTrue(result["requires_continue"])
        self.assertIn("Continue calibration", result["next_batch_hint"])
        self.assertEqual(result["score_semantics"], "clip_pnm_archive_calibration_not_ground_truth")
        channel7 = result["channels"][0]
        self.assertEqual(channel7["channel_id"], 7)
        self.assertEqual(channel7["frame_count"], 8)
        self.assertEqual(channel7["coverage"]["status"], "covered")
        self.assertGreater(channel7["distributions"]["margin"]["max"], 0.9)
        self.assertGreaterEqual(channel7["suggested_thresholds"]["pos_floor"], 0.05)
        self.assertIn(channel7["suggested_thresholds"]["confidence"], {"medium", "high"})
        self.assertTrue(channel7["representative_frames"]["top_margin"][0]["image_url"].startswith("/detections/thumbnail/"))
        self.assertNotIn("thumbnail", channel7["representative_frames"]["top_margin"][0])

        compact = _compact_tool_result_for_model("calibrate_probe_from_archive", result)
        self.assertEqual(compact["processed_channel_ids"], [7, 8])
        self.assertEqual(compact["deferred_channel_ids"], [9])
        self.assertEqual(compact["channels"][0]["suggested_thresholds"]["confidence"], channel7["suggested_thresholds"]["confidence"])
        self.assertTrue(compact["channels"][0]["representative_frames"]["top_margin"][0]["image_url"].startswith("/detections/thumbnail/"))

    def test_siglip2_calibration_rejects_legacy_or_mismatched_archive_vectors(self):
        siglip_space = {
            "backend": "siglip2",
            "model": "google/siglip2-base-patch16-224",
            "revision": "pinned-rev-1",
            "fingerprint": "space-a1b2",
            "dimension": 2,
        }
        rows = [
            {
                "id": 1,
                "channel_id": 7,
                "source": "semantic_snapshot",
                "event_timestamp_ms": 101_000,
                "clip_vec": [1.0, 0.0],
                "payload": {"embedding_space": siglip_space},
            },
            {
                "id": 2,
                "channel_id": 7,
                "source": "semantic_snapshot",
                "event_timestamp_ms": 102_000,
                "clip_vec": [1.0, 0.0],
                "payload": {},
            },
            {
                "id": 3,
                "channel_id": 7,
                "source": "semantic_snapshot",
                "event_timestamp_ms": 103_000,
                "clip_vec": [1.0, 0.0],
                "payload": {
                    "embedding_space": {
                        "backend": "openai_clip",
                        "model": "ViT-B/32",
                        "dimension": 2,
                    }
                },
            },
            {
                "id": 4,
                "channel_id": 7,
                "source": "semantic_snapshot",
                "event_timestamp_ms": 104_000,
                "clip_vec": [1.0, 0.0],
                "payload": {
                    "embedding_space": {
                        **siglip_space,
                        "fingerprint": "different-space",
                    }
                },
            },
        ]

        result = _tools(
            detections_store=_DetectionStore(rows),
            embed_text_fn=lambda text: (
                [0.0, 1.0]
                if "empty" in str(text).lower()
                else [1.0, 0.0]
            ),
            embedding_metadata_fn=lambda: siglip_space,
        ).execute(
            "calibrate_probe_from_archive",
            {
                "event_query": "person near window",
                "contrast_query": "empty window",
                "channel_id": 7,
                "sources": ["semantic_snapshot"],
                "from_ts": 100.0,
                "to_ts": 110.0,
                "min_frames": 1,
            },
        )

        channel = result["channels"][0]
        self.assertEqual(channel["frame_count"], 1)
        self.assertEqual(channel["embedding_space_rejected"], 3)
        self.assertEqual(result["embedding_space"]["revision"], "pinned-rev-1")
        self.assertEqual(result["embedding_space"]["fingerprint"], "space-a1b2")
        self.assertTrue(
            any("different embedding space" in item for item in channel["warnings"])
        )

    def test_calibrate_probe_flags_over_firing_as_unsafe(self):
        rows = [
            {
                "id": idx,
                "channel_id": 7,
                "source": "vlm_summary",
                "event_timestamp_ms": 100_000 + idx * 1_000,
                "clip_vec": [1.0, 0.0],
            }
            for idx in range(1, 11)
        ]

        def embed_text(text):
            return [0.0, 1.0] if "normal traffic" in str(text).lower() else [1.0, 0.0]

        result = _tools(detections_store=_DetectionStore(rows), embed_text_fn=embed_text).execute(
            "calibrate_probe_from_archive",
            {
                "event_query": "vehicle doing burnout",
                "contrast_query": "normal traffic",
                "channel_id": 7,
                "sources": ["vlm_summary"],
                "from_ts": 100.0,
                "to_ts": 120.0,
            },
        )

        thresholds = result["channels"][0]["suggested_thresholds"]
        self.assertEqual(thresholds["calibration_status"], "over_firing")
        self.assertFalse(thresholds["safe_to_apply"])
        self.assertIn("over_firing_positive_like_ratio", thresholds["warnings"])
        self.assertIn("not clean separation", thresholds["prevalence"]["interpretation"])

    def test_calibrate_probe_flags_target_absent_as_unsafe(self):
        rows = [
            {
                "id": idx,
                "channel_id": 7,
                "source": "vlm_summary",
                "event_timestamp_ms": 100_000 + idx * 1_000,
                "clip_vec": [0.0, 1.0],
            }
            for idx in range(1, 11)
        ]

        def embed_text(text):
            return [0.0, 1.0] if "people walking normally" in str(text).lower() else [1.0, 0.0]

        result = _tools(detections_store=_DetectionStore(rows), embed_text_fn=embed_text).execute(
            "calibrate_probe_from_archive",
            {
                "event_query": "person lying on ground",
                "contrast_query": "people walking normally",
                "channel_id": 7,
                "sources": ["vlm_summary"],
                "from_ts": 100.0,
                "to_ts": 120.0,
            },
        )

        thresholds = result["channels"][0]["suggested_thresholds"]
        self.assertEqual(thresholds["calibration_status"], "target_absent")
        self.assertEqual(thresholds["recommended_action"], "do_not_apply_rephrase_or_collect_examples")
        self.assertTrue(thresholds["needs_manual_frame_review"])
        self.assertFalse(thresholds["safe_to_apply"])

    def test_calibrate_probe_flags_weak_margin_as_unsafe(self):
        rows = []
        for idx, vec in enumerate((
            [0.51, 0.49],
            [0.51, 0.49],
            [0.51, 0.49],
            [0.51, 0.49],
            [0.49, 0.51],
            [0.49, 0.51],
            [0.49, 0.51],
            [0.49, 0.51],
        ), start=1):
            rows.append({
                "id": idx,
                "channel_id": 7,
                "source": "vlm_summary",
                "event_timestamp_ms": 100_000 + idx * 1_000,
                "clip_vec": vec,
            })

        def embed_text(text):
            return [0.0, 1.0] if "clear roadway" in str(text).lower() else [1.0, 0.0]

        result = _tools(detections_store=_DetectionStore(rows), embed_text_fn=embed_text).execute(
            "calibrate_probe_from_archive",
            {
                "event_query": "smoke visible",
                "contrast_query": "clear roadway",
                "channel_id": 7,
                "sources": ["vlm_summary"],
                "from_ts": 100.0,
                "to_ts": 120.0,
            },
        )

        thresholds = result["channels"][0]["suggested_thresholds"]
        self.assertEqual(thresholds["calibration_status"], "weak_separation")
        self.assertEqual(thresholds["recommended_action"], "rephrase_positive_or_contrast")
        self.assertFalse(thresholds["safe_to_apply"])

    def test_noisy_scene_calibration_requires_reviewed_refine_and_held_out_shadow(self):
        rows = []
        positive_ids = {1, 2, 9, 10}
        for idx in range(1, 17):
            rows.append(
                {
                    "id": idx,
                    "channel_id": 7,
                    "source": "semantic_snapshot",
                    "event_timestamp_ms": 100_000 + idx * 1_000,
                    "clip_vec": [1.0, 0.0] if idx in positive_ids else [0.0, 1.0],
                }
            )

        tools = _tools(
            detections_store=_DetectionStore(rows),
            embed_text_fn=lambda text: (
                [0.0, 1.0]
                if "ordinary room" in str(text).lower()
                else [1.0, 0.0]
            ),
        )
        common = {
            "event_query": "person leaves the desk",
            "contrast_query": "ordinary room activity at the occupied desk",
            "channel_id": 7,
            "sources": ["semantic_snapshot"],
            "from_ts": 100.0,
            "to_ts": 120.0,
            "min_frames": 4,
        }

        discovery = tools.execute(
            "calibrate_probe_from_archive",
            {**common, "calibration_stage": "discovery"},
        )["channels"][0]
        self.assertFalse(discovery["suggested_thresholds"]["safe_to_apply"])
        self.assertEqual(discovery["calibration_stages"]["next_stage"], "refine")

        refined = tools.execute(
            "calibrate_probe_from_archive",
            {
                **common,
                "calibration_stage": "refine",
                "reviewed_positive_detection_ids": [1, 2],
                "reviewed_negative_detection_ids": [3, 4, 5, 6],
            },
        )["channels"][0]
        candidate = refined["suggested_thresholds"]
        self.assertEqual(candidate["calibration_status"], "reviewed_candidate")
        self.assertFalse(candidate["safe_to_apply"])
        self.assertEqual(refined["calibration_stages"]["next_stage"], "shadow")

        shadow = tools.execute(
            "calibrate_probe_from_archive",
            {
                **common,
                "calibration_stage": "shadow",
                "candidate_pos_floor": candidate["pos_floor"],
                "candidate_margin_thr": candidate["margin_thr"],
                "shadow_from_ms": 109_000,
                "reviewed_positive_detection_ids": [9, 10],
                "reviewed_negative_detection_ids": [11, 12, 13, 14],
            },
        )["channels"][0]
        self.assertEqual(
            shadow["suggested_thresholds"]["calibration_status"],
            "shadow_validated",
        )
        self.assertTrue(shadow["suggested_thresholds"]["safe_to_apply"])
        self.assertTrue(shadow["calibration_stages"]["promotion_ready"])

    def test_prepare_probe_calibration_batch_keeps_server_side_job_state(self):
        rows = [
            {
                "id": 1,
                "channel_id": 7,
                "source": "vlm_summary",
                "event_timestamp_ms": 100_000,
                "clip_vec": [1.0, 0.0],
            },
            {
                "id": 11,
                "channel_id": 7,
                "source": "vlm_summary",
                "event_timestamp_ms": 101_000,
                "clip_vec": [0.0, 1.0],
            },
            {
                "id": 2,
                "channel_id": 8,
                "source": "vlm_summary",
                "event_timestamp_ms": 100_000,
                "clip_vec": [1.0, 0.0],
            },
            {
                "id": 12,
                "channel_id": 8,
                "source": "vlm_summary",
                "event_timestamp_ms": 101_000,
                "clip_vec": [0.0, 1.0],
            },
        ]

        def embed_text(text):
            value = str(text).lower()
            if "walking" in value or "normal" in value:
                return [0.0, 1.0]
            return [1.0, 0.0]

        tools = _tools(
            detections_store=_DetectionStore(rows),
            embed_text_fn=embed_text,
        )

        first = tools.execute(
            "prepare_probe_calibration_batch",
            {
                "items": [
                    {
                        "name": "person down",
                        "event_query": "person lying on ground",
                        "contrast_query": "people walking normally",
                        "channel_id": 7,
                    },
                    {
                        "name": "person down",
                        "event_query": "person lying on ground",
                        "contrast_query": "people walking normally",
                        "channel_id": 8,
                    },
                ],
                "items_per_call": 1,
                "sources": ["vlm_summary"],
                "from_ts": 99.0,
                "to_ts": 101.0,
                "min_frames": 1,
            },
        )

        self.assertEqual(first["processed_this_call"], 1)
        self.assertEqual(first["remaining_count"], 1)
        self.assertTrue(first["requires_continue"])
        item = first["processed_items"][0]
        self.assertEqual(item["recommended_probe_args"]["tool"], "create_probe")
        self.assertTrue(item["recommended_probe_args"]["args"]["preview"])
        self.assertEqual(item["recommended_probe_args"]["args"]["positives"], ["person lying on ground"])

        context = _seed_turn_tool_context("continue")
        _remember_turn_tool_result("prepare_probe_calibration_batch", first, context)
        next_args = _apply_turn_tool_context("prepare_probe_calibration_batch", {}, context)
        self.assertEqual(next_args["job_id"], first["job_id"])

        calibration_calls = []
        original_calibrate = tools._calibrate_probe_from_archive

        def record_calibration(call_args):
            calibration_calls.append(dict(call_args))
            return original_calibrate(call_args)

        tools._calibrate_probe_from_archive = record_calibration
        second = tools.execute("prepare_probe_calibration_batch", {"job_id": first["job_id"]})
        self.assertEqual(second["status"], "complete")
        self.assertEqual(second["processed_total"], 2)
        self.assertEqual(second["remaining_count"], 0)
        self.assertEqual(calibration_calls[-1]["sources"], ["vlm_summary"])
        self.assertEqual(calibration_calls[-1]["from_ts"], 99.0)
        self.assertEqual(calibration_calls[-1]["to_ts"], 101.0)

        compact = _compact_tool_result_for_model("prepare_probe_calibration_batch", first)
        self.assertEqual(compact["job_id"], first["job_id"])
        self.assertTrue(compact["output_contract"]["recommended_probe_args_are_pass_through"])
        self.assertEqual(compact["processed_items"][0]["recommended_probe_args"]["tool"], "create_probe")

    def test_prepare_probe_calibration_batch_suppresses_unsafe_recommendations(self):
        rows = [
            {
                "id": idx,
                "channel_id": 7,
                "source": "vlm_summary",
                "event_timestamp_ms": 100_000 + idx * 1_000,
                "clip_vec": [1.0, 0.0],
            }
            for idx in range(1, 10)
        ]

        def embed_text(text):
            return [0.0, 1.0] if "normal traffic" in str(text).lower() else [1.0, 0.0]

        result = _tools(detections_store=_DetectionStore(rows), embed_text_fn=embed_text).execute(
            "prepare_probe_calibration_batch",
            {
                "items": [
                    {
                        "name": "vehicle burnout",
                        "event_query": "vehicle doing burnout",
                        "contrast_query": "normal traffic",
                        "channel_id": 7,
                    },
                ],
                "items_per_call": 1,
                "sources": ["vlm_summary"],
                "from_ts": 99.0,
                "to_ts": 120.0,
            },
        )

        item = result["processed_items"][0]
        self.assertEqual(item["suggested_thresholds"]["calibration_status"], "over_firing")
        self.assertIsNone(item["recommended_probe_args"])
        self.assertEqual(item["next_action"], "tighten_or_rephrase_contrast")

    def test_prepare_probe_calibration_batch_suppresses_negated_contrast_recommendations(self):
        rows = [
            {
                "id": idx,
                "channel_id": 7,
                "source": "vlm_summary",
                "event_timestamp_ms": 100_000 + idx * 1_000,
                "clip_vec": [1.0, 0.0] if idx <= 4 else [0.0, 1.0],
            }
            for idx in range(1, 9)
        ]

        def embed_text(text):
            value = str(text).lower()
            if "no person" in value:
                return [0.0, 1.0]
            return [1.0, 0.0]

        result = _tools(detections_store=_DetectionStore(rows), embed_text_fn=embed_text).execute(
            "prepare_probe_calibration_batch",
            {
                "items": [
                    {
                        "name": "person down",
                        "event_query": "person lying on ground",
                        "contrast_query": "no person",
                        "channel_id": 7,
                    },
                ],
                "items_per_call": 1,
                "sources": ["vlm_summary"],
                "from_ts": 99.0,
                "to_ts": 120.0,
            },
        )

        item = result["processed_items"][0]
        self.assertEqual(item["suggested_thresholds"]["calibration_status"], "bad_contrast")
        self.assertFalse(item["suggested_thresholds"]["safe_to_apply"])
        self.assertTrue(any("negation" in warning for warning in item["warnings"]))
        self.assertIsNone(item["recommended_probe_args"])
        compact = _compact_tool_result_for_model("prepare_probe_calibration_batch", result)
        self.assertIsNone(compact["processed_items"][0]["recommended_probe_args"])

    def test_probe_negative_prompts_reject_literal_negation(self):
        tools = _tools(
            probes_store=_ProbeStore([
                {
                    "id": "probe-1",
                    "name": "thumbs",
                    "channel_id": 7,
                    "positives": ["thumbs up gesture"],
                    "negatives": ["person with hand lowered"],
                    "pos_floor": 0.2,
                    "margin": 0.05,
                }
            ])
        )

        with self.assertRaisesRegex(Exception, "literal negation"):
            tools.execute(
                "create_probe",
                {
                    "name": "unsafe negative",
                    "channel_id": 7,
                    "positives": ["thumbs up gesture"],
                    "negatives": ["person with hand raised but not thumbs up"],
                    "preview": True,
                },
            )

        with self.assertRaisesRegex(Exception, "literal negation"):
            tools.execute(
                "update_probe",
                {
                    "probe_id": "probe-1",
                    "changes": {"negatives": ["no smoke visible"]},
                    "preview": True,
                },
            )

    def test_track_visual_state_transition_negative_embedding_warning_is_specific(self):
        store = _DetectionStore(
            [
                {
                    "id": 1,
                    "channel_id": 7,
                    "source": "vlm_summary",
                    "event_timestamp_ms": 100_000,
                    "clip_vec": [1.0, 0.0],
                    "thumbnail": "unused-positive",
                },
            ]
        )

        def embed_text(text):
            value = str(text).lower()
            if "empty" in value:
                return None
            return [1.0, 0.0]

        result = _tools(detections_store=store, embed_text_fn=embed_text).execute(
            "track_visual_state_transitions",
            {
                "channel_id": 7,
                "subject_query": "object",
                "positive_state_query": "object on table",
                "negative_state_query": "empty table",
                "sources": ["vlm_summary"],
                "from_ts": 100.0,
                "to_ts": 101.0,
            },
        )

        self.assertIn(
            "negative_state_query could not be embedded; unknown/positive separation is weaker.",
            result["warnings"],
        )
        self.assertNotIn(
            "negative_state_query was not provided; unknown/positive separation is weaker.",
            result["warnings"],
        )

    def test_get_detections_sorts_by_timestamp_fallbacks_and_compacts_vlm_semantics(self):
        rows = [
            {
                "id": 1,
                "recorded_at_ms": 300_000,
                "probe_id": "vlm_summary:7",
                "channel_id": 7,
                "source": "vlm_summary",
            },
            {
                "id": 2,
                "event_timestamp_ms": 100_000,
                "probe_id": "vlm_summary:7",
                "channel_id": 7,
                "source": "vlm_summary",
            },
            {
                "id": 3,
                "timestamp_ms": 200_000,
                "probe_id": "vlm_summary:7",
                "channel_id": 7,
                "source": "vlm_summary",
            },
        ]
        result = _tools(detections_store=_DetectionStore(rows)).execute(
            "get_detections",
            {
                "channel_id": 7,
                "source": "vlm_summary",
                "since_ms": 0,
                "until_ms": 400_000,
                "sort_by": "oldest",
                "limit": 10,
            },
        )

        self.assertEqual([row["id"] for row in result["detections"]], [2, 3, 1])

        compact = _compact_tool_result_for_model("get_detections", result)
        self.assertEqual([row["timestamp_ms"] for row in compact["detections"]], [100_000, 200_000, 300_000])
        self.assertEqual([row["detection_id"] for row in compact["detections"]], [2, 3, 1])
        self.assertTrue(
            all(row["score_semantics"] == "not_applicable" for row in compact["detections"])
        )

    def test_detection_window_oldest_reaches_start_of_large_period(self):
        store = _DetectionStore(
            [
                {
                    "id": index,
                    "source": "vlm_alert",
                    "channel_id": 7,
                    "timestamp_ms": 100_000 + index * 1_000,
                }
                for index in range(20)
            ]
        )
        tools = _tools(detections_store=store)

        rows, total = tools._list_detection_window(
            probe_id=None,
            channel_id=7,
            source="vlm_alert",
            since_ms=100_000,
            until_ms=119_000,
            limit=3,
            offset=0,
            sort_by="oldest",
            max_scan=3,
        )

        self.assertEqual(total, 20)
        self.assertEqual([row["id"] for row in rows], [0, 1, 2])

        tail_rows, tail_total = tools._list_detection_window(
            probe_id=None,
            channel_id=7,
            source="vlm_alert",
            since_ms=100_000,
            until_ms=119_000,
            limit=3,
            offset=19,
            sort_by="oldest",
            max_scan=3,
        )

        self.assertEqual(tail_total, 20)
        self.assertEqual([row["id"] for row in tail_rows], [19])

    def test_list_video_summary_channels_returns_active_candidates_and_confirmation_flag(self):
        result = _tools().execute(
            "list_video_summary_channels",
            {
                "from_ts": 100.0,
                "to_ts": 300.0,
                "limit": 10,
            },
        )

        self.assertEqual(result["active_count"], 2)
        self.assertEqual(result["inactive_count"], 1)
        self.assertEqual(result["inactive_channel_ids"], [9])
        self.assertEqual(result["candidate_channel_ids"], [7, 8])
        self.assertFalse(result["requires_confirmation"])
        self.assertEqual(
            [row["channel_id"] for row in result["candidate_channels"]],
            [7, 8],
        )
        row7 = result["candidate_channels"][0]
        self.assertEqual(row7["recent_alerts"][0]["title"], "Doorway activity")
        self.assertEqual(row7["status_digest"]["alert_delivery_breakdown"]["sent"], 1)
        self.assertEqual(row7["live_signal_status"], "frozen")
        self.assertTrue(row7["frozen_signal"])
        self.assertEqual(row7["frozen_frame_count"], 4)
        self.assertGreaterEqual(result["runtime_problem_count"], 1)
        self.assertEqual(result["runtime_problem_channels"][0]["channel_id"], 7)
        self.assertEqual(result["runtime_problem_channels"][0]["live_signal_status"], "frozen")
        stale_row = next(row for row in result["runtime_problem_channels"] if row["channel_id"] == 9)
        self.assertEqual(stale_row["live_signal_status"], "stale")
        self.assertTrue(stale_row["stale_signal"])
        compact = _compact_tool_result_for_model("list_video_summary_channels", result)
        self.assertEqual(compact["candidate_channels"][0]["recent_alerts"][0]["title"], "Doorway activity")
        self.assertEqual(compact["inactive_channel_ids"], [9])
        self.assertEqual(compact["candidate_channel_ids"], [7, 8])
        self.assertEqual(compact["candidate_channels"][0]["live_signal_status"], "frozen")
        self.assertTrue(compact["candidate_channels"][0]["frozen_signal"])
        self.assertEqual(compact["runtime_problem_channels"][0]["live_signal_status"], "frozen")
        self.assertTrue(any(row["live_signal_status"] == "stale" for row in compact["runtime_problem_channels"]))
        active_runtime = compact["active_runtime_streams"][0]
        self.assertEqual(active_runtime["channel_id"], 7)
        self.assertEqual(active_runtime["buffered_frames"], 3)
        self.assertEqual(active_runtime["summary_queue_depth"], 2)
        self.assertEqual(active_runtime["summary_queue_frames"], 16)
        runtime_only = _tools().execute(
            "list_video_summary_channels",
            {"from_ts": 100.0, "to_ts": 300.0, "runtime_only": True},
        )
        self.assertTrue(runtime_only["runtime_only"])
        self.assertEqual(runtime_only["active_count"], 1)
        self.assertEqual(runtime_only["inactive_count"], 0)
        self.assertEqual(runtime_only["candidate_channels"], [])

        manager = _SummaryManager()
        manager.session_status = lambda *args, **kwargs: self.fail(
            "runtime-only status must not scan historical summaries"
        )
        scoped_runtime = _tools(manager).execute(
            "list_video_summary_channels",
            {
                "channel_ids": [7],
                "from_ts": 100.0,
                "to_ts": 300.0,
                "runtime_only": True,
            },
        )
        self.assertEqual(scoped_runtime["running_video_channels"], [7])
        self.assertEqual(scoped_runtime["desired_video_channels"], [7])
        self.assertEqual(
            [row["channel_id"] for row in scoped_runtime["active_runtime_streams"]],
            [7],
        )
        self.assertNotIn(8, scoped_runtime["scope"]["active_channel_ids"])

        runtime_context = _seed_turn_tool_context(
            "List active streams, models, queues, dropped frames, and last errors"
        )
        prepared = _apply_turn_tool_context(
            "list_video_summary_channels", {"since_hours": 6}, runtime_context
        )
        self.assertTrue(prepared["runtime_only"])

    def test_get_video_summaries_distinguishes_pending_semantics_from_no_source_data(self):
        manager = _SummaryManager()

        def pending_rollups(channel_id, run_selector=None, start_ts=None, end_ts=None, level_limit=None):
            return {
                "running": True,
                "levels": {
                    "L0": [],
                    "L1": [
                        {
                            "level": "L1",
                            "window_start": 100.0,
                            "window_end": 200.0,
                            "summary": "Aggregation in progress.",
                            "summary_kind": "pending_context",
                            "generation_status": "pending",
                        }
                    ],
                    "L2": [],
                    "L3": [],
                },
            }

        manager.summary_rollups = pending_rollups
        result = _tools(manager).execute(
            "get_video_summaries",
            {"channel_id": 7, "depth": "L1", "from_ts": 100.0, "to_ts": 200.0},
        )

        self.assertEqual(result["total_in_window"], 1)
        self.assertEqual(result["semantic_available_count"], 0)
        self.assertEqual(result["semantic_pending_count"], 1)
        self.assertEqual(result["semantic_status"], "pending")
        self.assertEqual(result["source_coverage"]["status"], "covered")
        self.assertEqual(result["count"], 1)

    def test_list_video_summary_channels_uses_batch_bounds_for_activity_window(self):
        manager = _SummaryManager()
        captured = {}
        manager.logs_by_channel = {
            7: [
                {
                    "created_at": 2000.0,
                    "batch_start_ms": 100_000,
                    "batch_end_ms": 130_000,
                    "summary": "cat returns",
                    "frame_count": 12,
                }
            ],
            8: [],
            9: [],
        }
        original_session_status = manager.session_status

        def session_status(channel_id, run_selector=None, start_ts=None, end_ts=None, limit=None):
            captured["run_selector"] = run_selector
            return original_session_status(
                channel_id,
                run_selector=run_selector,
                start_ts=start_ts,
                end_ts=end_ts,
                limit=limit,
            )

        manager.session_status = session_status

        result = _tools(manager).execute(
            "list_video_summary_channels",
            {
                "channel_ids": [7],
                "from_ts": 90.0,
                "to_ts": 140.0,
                "limit": 10,
            },
        )

        self.assertEqual(result["active_count"], 1)
        self.assertEqual(captured["run_selector"], "all")
        row = result["candidate_channels"][0]
        self.assertEqual(row["first_ts"], 100.0)
        self.assertEqual(row["latest_ts"], 130.0)

    def test_list_video_summary_channels_accepts_millisecond_time_window(self):
        result = _tools().execute(
            "list_video_summary_channels",
            {
                "since_ms": 100_000,
                "until_ms": 300_000,
                "limit": 10,
            },
        )

        self.assertEqual(result["from_ts"], 100.0)
        self.assertEqual(result["to_ts"], 300.0)
        self.assertEqual(result["time_window"]["since_ms"], 100_000)
        self.assertEqual(result["time_window"]["until_ms"], 300_000)

    def test_channel_title_resolution_is_unicode_safe(self):
        manager = _SummaryManager()
        manager.channels = [{"id": 112, "title": "თბილისის ქუჩა №1"}]

        result = _tools(manager).execute(
            "get_video_summaries",
            {
                "channel_ref": "თბილისის ქუჩა",
                "from_ts": 100.0,
                "to_ts": 300.0,
            },
        )

        self.assertEqual(result["channel_id"], 112)

    def test_list_video_summary_channels_falls_back_to_local_history_when_channel_inventory_fails(self):
        manager = _SummaryManager()

        def get_channels(force=False):
            raise RuntimeError("No route to host")

        manager.get_channels = get_channels
        manager.logs_by_channel = {
            7: [
                {
                    "created_at": 150.0,
                    "summary": "local archive event",
                    "frame_count": 3,
                    "alert_counts": {"normal": 1},
                    "alert_total": 1,
                }
            ],
        }

        result = _tools(manager).execute(
            "list_video_summary_channels",
            {
                "from_ts": 100.0,
                "to_ts": 300.0,
                "limit": 10,
            },
        )
        compact = _compact_tool_result_for_model("list_video_summary_channels", result)

        self.assertEqual(result["channel_inventory_status"], "archive_fallback")
        self.assertIn("No route to host", result["channel_inventory_error"])
        self.assertEqual(result["active_count"], 1)
        self.assertEqual(result["candidate_channels"][0]["channel_id"], 7)
        self.assertEqual(result["candidate_channels"][0]["alert_total"], 1)
        self.assertEqual(result["error_count"], 1)
        self.assertEqual(result["inactive_count"], 1)
        self.assertEqual(result["total_channels_checked"], 2)
        self.assertEqual(compact["channel_inventory_status"], "archive_fallback")

    def test_list_video_summary_channels_augments_partial_live_inventory_with_provenance(self):
        manager = _SummaryManager()
        manager.channels = [{"id": 7, "title": "Live channel"}]
        manager.logs_by_channel = {
            8: [{"created_at": 150.0, "summary": "history", "frame_count": 1}],
        }

        def streams_status():
            return {
                "video_streams": [{"channel_id": 9, "running": True, "title": "Runtime channel"}],
                "channel_status_digest": [{"channel_id": 10, "title": "Digest channel"}],
                "desired_video_channels": [11],
                "desired_video_missing": [],
            }

        def session_status(channel_id, run_selector=None, start_ts=None, end_ts=None, limit=None):
            return {
                "running": int(channel_id) == 9,
                "logs": [{"created_at": 150.0, "summary": f"channel {channel_id}", "frame_count": 1}],
            }

        manager.streams_status = streams_status
        manager.session_status = session_status
        result = _tools(manager).execute(
            "list_video_summary_channels",
            {
                "channel_ids": [7, 8, 9, 10, 11, 99],
                "from_ts": 100.0,
                "to_ts": 300.0,
                "limit": 10,
            },
        )

        self.assertEqual(result["channel_inventory_status"], "live_augmented")
        self.assertEqual(result["live_inventory_count"], 1)
        self.assertEqual(result["checked_channel_ids"], [7, 8, 9, 10, 11])
        self.assertEqual(result["unchecked_channel_ids"], [99])
        provenance = {row["channel_id"]: row["sources"] for row in result["inventory_provenance"]}
        self.assertIn("live_inventory", provenance[7])
        self.assertIn("logs_by_channel", provenance[8])
        self.assertIn("runtime", provenance[9])
        self.assertIn("status_digest", provenance[10])
        self.assertIn("desired", provenance[11])
        self.assertEqual(provenance[99], ["requested"])
        compact = _compact_tool_result_for_model("list_video_summary_channels", result)
        self.assertEqual(compact["scope"]["checked_channel_ids"], [7, 8, 9, 10, 11])
        self.assertEqual(compact["scope"]["unchecked_channel_ids"], [99])

    def test_list_video_summary_channels_marks_stale_cached_inventory(self):
        manager = _SummaryManager()
        manager.channels = [{"id": 7, "title": "Cached channel"}]
        manager.channel_inventory_status = lambda: {
            "cached": True,
            "count": 1,
            "stale": True,
            "cache_age_sec": 44.0,
            "last_error": "temporary upstream timeout",
            "stream": {"completion": "settled"},
        }

        result = _tools(manager).execute(
            "list_video_summary_channels",
            {"from_ts": 100.0, "to_ts": 300.0, "limit": 10},
        )
        compact = _compact_tool_result_for_model("list_video_summary_channels", result)

        self.assertEqual(result["channel_inventory_status"], "stale_cache_augmented")
        self.assertIn("temporary upstream timeout", result["channel_inventory_error"])
        self.assertTrue(result["channel_inventory_cache"]["stale"])
        self.assertEqual(result["candidate_channels"][0]["channel_id"], 7)
        self.assertEqual(compact["channel_inventory_status"], "stale_cache_augmented")
        self.assertTrue(compact["channel_inventory_cache"]["stale"])

    def test_list_video_summary_channels_caps_candidates_when_confirmation_required(self):
        manager = _SummaryManager()
        manager.channels = [
            {"id": channel_id, "title": f"Channel {channel_id}"}
            for channel_id in range(1, AGENT_VIDEO_SUMMARY_CHANNELS_PER_TURN + 4)
        ]
        manager.logs_by_channel = {
            channel["id"]: [
                {
                    "created_at": 150.0 + channel["id"],
                    "summary": f"event on {channel['id']}",
                    "frame_count": 12,
                }
            ]
            for channel in manager.channels
        }

        result = _tools(manager).execute(
            "list_video_summary_channels",
            {
                "from_ts": 100.0,
                "to_ts": 300.0,
                "limit": AGENT_VIDEO_SUMMARY_CHANNELS_PER_TURN + 3,
            },
        )
        compact = _compact_tool_result_for_model("list_video_summary_channels", result)

        self.assertTrue(result["requires_confirmation"])
        self.assertEqual(result["active_count"], AGENT_VIDEO_SUMMARY_CHANNELS_PER_TURN + 3)
        self.assertEqual(result["returned"], AGENT_VIDEO_SUMMARY_CHANNELS_PER_TURN)
        self.assertEqual(len(result["candidate_channels"]), AGENT_VIDEO_SUMMARY_CHANNELS_PER_TURN)
        self.assertEqual(result["deferred_count"], 3)
        self.assertEqual(len(result["deferred_channel_ids"]), 3)
        self.assertEqual(compact["returned"], AGENT_VIDEO_SUMMARY_CHANNELS_PER_TURN)
        self.assertEqual(compact["deferred_count"], 3)
        self.assertEqual(compact["scope"]["deferred_count"], 3)
        self.assertEqual(compact["scope"]["deferred_channel_ids"], result["deferred_channel_ids"])

    def test_list_video_summary_channel_scope_id_lists_are_bounded(self):
        manager = _SummaryManager()
        manager.channels = [
            {"id": channel_id, "title": f"Channel {channel_id}"}
            for channel_id in range(1, 131)
        ]
        manager.logs_by_channel = {
            channel["id"]: [
                {
                    "created_at": 150.0,
                    "summary": f"event on {channel['id']}",
                    "frame_count": 1,
                }
            ]
            for channel in manager.channels
        }

        result = _tools(manager).execute(
            "list_video_summary_channels",
            {"from_ts": 100.0, "to_ts": 300.0, "limit": 100},
        )
        compact = _compact_tool_result_for_model("list_video_summary_channels", result)

        self.assertEqual(result["total_channels_checked"], 130)
        self.assertEqual(result["scope"]["checked_count"], 130)
        self.assertTrue(result["scope"]["id_lists_truncated"])
        self.assertEqual(result["scope"]["id_list_limit"], 100)
        self.assertEqual(len(result["checked_channel_ids"]), 100)
        self.assertEqual(result["deferred_count"], 122)
        self.assertEqual(len(result["deferred_channel_ids"]), 100)
        self.assertEqual(result["candidate_channel_ids"], list(range(1, 9)))
        self.assertEqual(compact["scope"], result["scope"])

    def test_compact_list_video_summary_channels_preserves_errors_and_unchecked_counts(self):
        manager = _SummaryManager()
        manager.channels = [
            {"id": 7, "title": "Kitchen"},
            {"id": 8, "title": "Door"},
        ]

        def session_status(channel_id, run_selector=None, start_ts=None, end_ts=None, limit=None):
            if int(channel_id) == 8:
                raise RuntimeError("camera status unavailable")
            return {
                "running": False,
                "channel_id": channel_id,
                "logs": [{"created_at": 150.0, "summary": "person enters", "frame_count": 3}],
                "selected_run": None,
            }

        manager.session_status = session_status
        result = _tools(manager).execute(
            "list_video_summary_channels",
            {
                "channel_ids": [7, 8, 99],
                "from_ts": 100.0,
                "to_ts": 300.0,
                "limit": 10,
            },
        )
        compact = _compact_tool_result_for_model("list_video_summary_channels", result)

        self.assertEqual(result["requested_count"], 3)
        self.assertEqual(result["unchecked_count"], 1)
        self.assertEqual(result["unchecked_channel_ids"], [99])
        self.assertEqual(result["error_count"], 1)
        self.assertEqual(compact["requested_count"], 3)
        self.assertEqual(compact["unchecked_count"], 1)
        self.assertEqual(compact["error_count"], 1)
        self.assertEqual(compact["errors"][0]["channel_id"], 8)
        self.assertEqual(compact["scope"]["checked_channel_ids"], [7, 8])
        self.assertEqual(compact["scope"]["unchecked_channel_ids"], [99])
        self.assertEqual(compact["scope"]["error_channel_ids"], [8])

    def test_generate_report_defaults_to_video_descriptions_and_avoids_probe_summary(self):
        class VideoReportStore(_DetectionStore):
            def summarize_by_probe(self, *args, **kwargs):
                raise AssertionError("default video report should not query probe summaries")

        class PipelineHealthManager(_SummaryManager):
            def __init__(self):
                super().__init__()
                self.logs_by_channel[7] = [
                    {
                        "created_at": 150.0,
                        "summary": "inside",
                        "frame_count": 3,
                        "alert_counts": {"normal": 1},
                        "parser_alert_count": 2,
                        "json_alert_count": 1,
                        "prose_alert_count": 2,
                        "alert_events": [
                            {
                                "title": "Visible event",
                                "delivery_status": "sent",
                            }
                        ],
                        "state_transition_total": 1,
                    }
                ]

        store = VideoReportStore(
            [
                {
                    "id": 501,
                    "detection_id": 501,
                    "source": "vlm_alert",
                    "channel_id": 7,
                    "timestamp_ms": 150_000,
                    "thumbnail": "inline-should-not-leak",
                }
            ]
        )
        result = _tools(manager=PipelineHealthManager(), detections_store=store).execute(
            "generate_report",
            {
                "from_ts": 100.0,
                "to_ts": 300.0,
                "channel_ids": [7, 8],
            },
        )

        self.assertEqual(result["report_type"], "video_descriptions")
        self.assertIn("Video-description report", result["report"])
        self.assertEqual(result["summary"]["alert_total"], 1)
        self.assertEqual(result["summary"]["desired_missing_count"], 1)
        self.assertEqual(result["coverage"]["status"], "partial")
        self.assertIn("Detection pipeline health", result["report"])
        self.assertEqual(result["pipeline_health"]["alert_parser_breakdown"]["prose_only_signal_count"], 1)
        self.assertEqual(result["pipeline_health"]["alert_delivery_breakdown"]["sent"], 1)
        self.assertEqual(result["pipeline_health"]["state_transition_total"], 1)
        self.assertEqual(result["vlm_alert_frames"][0]["image_url"], "/detections/thumbnail/501")
        self.assertNotIn("thumbnail", result["vlm_alert_frames"][0])

        compact = _compact_tool_result_for_model("generate_report", result)
        self.assertEqual(compact["report_type"], "video_descriptions")
        self.assertIn("report", compact)
        self.assertEqual(compact["pipeline_health"]["alert_delivery_breakdown"]["sent"], 1)
        self.assertEqual(compact["vlm_alert_frames"][0]["image_url"], "/detections/thumbnail/501")

    def test_generate_report_samples_vlm_alert_frames_across_period(self):
        class SingleChannelManager(_SummaryManager):
            channels = [{"id": 7, "title": "Lobby"}]

            def session_status(self, channel_id, run_selector=None, start_ts=None, end_ts=None, limit=None):
                return {
                    "running": False,
                    "logs": [
                        {
                            "created_at": 100.0 + index * 100.0,
                            "batch_start_ms": 100_000 + index * 100_000,
                            "batch_end_ms": 110_000 + index * 100_000,
                            "summary": f"summary {index}",
                            "frame_count": 12,
                            "alert_counts": {"normal": 1},
                            "alert_total": 1,
                        }
                        for index in range(9)
                    ],
                }

        store = _DetectionStore(
            [
                {
                    "id": index,
                    "detection_id": index,
                    "source": "vlm_alert",
                    "channel_id": 7,
                    "timestamp_ms": 100_000 + index * 100_000,
                }
                for index in range(9)
            ]
        )

        result = _tools(manager=SingleChannelManager(), detections_store=store).execute(
            "generate_report",
            {
                "from_ts": 100.0,
                "to_ts": 1_000.0,
                "channel_id": 7,
                "top_events": 3,
            },
        )

        self.assertEqual(
            [row["detection_id"] for row in result["vlm_alert_frames"]],
            [0, 4, 8],
        )

    def test_generate_report_accepts_millisecond_time_window_and_inventory_fallback(self):
        class OfflineInventoryManager(_SummaryManager):
            def get_channels(self, force=False):
                raise RuntimeError("No route to host")

        result = _tools(manager=OfflineInventoryManager()).execute(
            "generate_report",
            {
                "report_type": "video_descriptions",
                "since_ms": 100_000,
                "until_ms": 300_000,
                "top_events": 2,
            },
        )

        self.assertEqual(result["period"]["from_ts"], 100.0)
        self.assertEqual(result["period"]["to_ts"], 300.0)
        self.assertEqual(result["coverage"]["channel_inventory_status"], "archive_fallback")
        self.assertIn("Live channel inventory unavailable", result["report"])
        self.assertGreaterEqual(result["summary"]["returned_channels"], 1)

    def test_generate_report_probe_type_keeps_legacy_probe_shape(self):
        class ProbeReportStore(_DetectionStore):
            def summarize_by_probe(self, *args, **kwargs):
                self.summary_kwargs = kwargs
                return [
                    {
                        "probe_id": "probe-1",
                        "probe_name": "door",
                        "channel_id": 7,
                        "hit_count": 2,
                        "latest_timestamp_ms": 200_000,
                    }
                ]

        store = ProbeReportStore(
            [
                {
                    "id": 1,
                    "probe_id": "probe-1",
                    "probe_name": "door",
                    "source": "probe",
                    "channel_id": 7,
                    "timestamp_ms": 200_000,
                    "margin": 0.2,
                }
            ]
        )
        result = _tools(detections_store=store).execute(
            "generate_report",
            {
                "report_type": "probes",
                "since_hours": 24,
                "channel_id": 7,
            },
        )

        self.assertEqual(result["report_type"], "probes")
        self.assertEqual(result["total_detections"], 2)
        self.assertEqual(result["probe_count"], 1)
        self.assertEqual(result["probes"][0]["probe_name"], "door")
        self.assertEqual(store.summary_kwargs["source"], "probe")

    def test_generate_report_false_positives_uses_operator_annotations(self):
        class FeedbackReportStore(_DetectionStore):
            def generate_false_positive_report(self, **kwargs):
                self.feedback_kwargs = kwargs
                return {
                    "report_type": "false_positives",
                    "period": {
                        "since_ms": kwargs["since_ms"],
                        "until_ms": kwargs["until_ms"],
                    },
                    "coverage": {
                        "status": "covered",
                        "annotation_count": 2,
                        "ground_truth_status": "operator_annotation_only",
                    },
                    "summary": {
                        "annotation_count": 2,
                        "distinct_alert_count": 2,
                        "reviewer_count": 1,
                        "channel_count": 1,
                    },
                    "reason_counts": [
                        {
                            "reason_code": "benign_activity",
                            "reason_label": "Benign activity",
                            "count": 2,
                        }
                    ],
                    "channel_counts": [{"channel_id": 7, "count": 2}],
                    "feedback": [
                        {
                            "detection_id": 501,
                            "channel_id": 7,
                            "alert_timestamp_ms": 150_000,
                            "actor_id": "private-reviewer-id",
                            "reason_code": "benign_activity",
                            "reason_label": "Benign activity",
                            "alert_title": "Person near door",
                            "note": "Maintenance worker.",
                        }
                    ],
                    "report": "# False-positive operator feedback report",
                }

        store = FeedbackReportStore()
        result = _tools(detections_store=store).execute(
            "generate_report",
            {
                "report_type": "false_positives",
                "from_ts": 100.0,
                "to_ts": 300.0,
                "channel_ids": [7],
                "top_events": 5,
            },
        )
        compact = _compact_tool_result_for_model("generate_report", result)

        self.assertEqual(result["report_type"], "false_positives")
        self.assertEqual(store.feedback_kwargs["since_ms"], 100_000)
        self.assertEqual(store.feedback_kwargs["until_ms"], 300_000)
        self.assertEqual(store.feedback_kwargs["channel_ids"], [7])
        self.assertEqual(store.feedback_kwargs["item_limit"], 5)
        self.assertEqual(
            result["coverage"]["ground_truth_status"],
            "operator_annotation_only",
        )
        self.assertEqual(compact["reason_counts"][0]["count"], 2)
        self.assertEqual(compact["feedback"][0]["detection_id"], 501)
        self.assertNotIn("actor_id", compact["feedback"][0])

    def test_generate_probe_report_exposes_representative_events_to_model(self):
        class ProbeReportStore(_DetectionStore):
            def summarize_by_probe(self, *args, **kwargs):
                return [
                    {
                        "probe_id": "probe-1",
                        "probe_name": "door",
                        "channel_id": 7,
                        "hit_count": 9,
                        "latest_timestamp_ms": 900_000,
                    }
                ]

        store = ProbeReportStore(
            [
                {
                    "id": index,
                    "probe_id": "probe-1",
                    "probe_name": "door",
                    "source": "probe",
                    "channel_id": 7,
                    "timestamp_ms": 100_000 + index * 100_000,
                    "margin": 0.2,
                }
                for index in range(9)
            ]
        )

        result = _tools(detections_store=store).execute(
            "generate_report",
            {
                "report_type": "probes",
                "since_hours": 1_000_000,
                "channel_id": 7,
                "top_events": 3,
            },
        )
        compact = _compact_tool_result_for_model("generate_report", result)

        self.assertEqual(
            [row["detection_id"] for row in result["probes"][0]["representative_events"]],
            [0, 4, 8],
        )
        self.assertEqual(
            [row["detection_id"] for row in compact["probes"][0]["representative_events"]],
            [0, 4, 8],
        )

    def test_list_attention_bursts_ranks_windows_and_reports_gaps(self):
        class _BurstSummaryManager(_SummaryManager):
            def summary_rollups(self, channel_id, run_selector=None, start_ts=None, end_ts=None, level_limit=None, target_level=None, synthesize=True):
                nodes = [
                    {
                        "level": "L0",
                        "window_start": 100.0,
                        "window_end": 112.0,
                        "batch_start_ms": 100_000,
                        "batch_end_ms": 112_000,
                        "summary": "car drifting across the lot with smoke",
                        "vector_signal": {
                            "capture_attention": {
                                "baseline": {"level": 0.001, "warmup": False},
                                "seconds": [
                                    {"snapshot": 3, "mode": "burst", "activity_x": 11.1, "sharper_companion": True},
                                    {"snapshot": 5, "mode": "normal", "activity_x": 3.0},
                                ],
                            }
                        },
                    },
                    {
                        "level": "L0",
                        "window_start": 112.0,
                        "window_end": 124.0,
                        "batch_start_ms": 112_000,
                        "batch_end_ms": 124_000,
                        "summary": "quiet lot",
                    },
                    {
                        "level": "L0",
                        "window_start": 124.0,
                        "window_end": 136.0,
                        "batch_start_ms": 124_000,
                        "batch_end_ms": 136_000,
                        "summary": "[coverage gap] dropped",
                        "coverage_gap": True,
                        "gap_reason": "lm_backpressure_dropped_batch",
                    },
                    {
                        "level": "L0",
                        "window_start": 136.0,
                        "window_end": 148.0,
                        "batch_start_ms": 136_000,
                        "batch_end_ms": 148_000,
                        "summary": "person runs through",
                        "vector_signal": {
                            "capture_attention": {
                                "seconds": [
                                    {"snapshot": 1, "mode": "burst", "activity_x": 5.5},
                                ]
                            }
                        },
                    },
                ]
                return {"levels": {"L0": nodes}, "source_counts": {"L0": len(nodes)}}

        tools = _tools(manager=_BurstSummaryManager())
        result = tools.execute(
            "list_attention_bursts",
            {"channel_id": 7, "from_ts": 90.0, "to_ts": 200.0, "min_activity_x": 2.0},
        )

        self.assertEqual(result["burst_count"], 2)
        self.assertEqual(result["bursts"][0]["activity_x"], 11.1)
        self.assertEqual(result["bursts"][0]["snapshot"], 3)
        self.assertTrue(result["bursts"][0]["sharper_companion"])
        self.assertEqual(result["bursts"][0]["baseline_level"], 0.001)
        self.assertIn("drifting", result["bursts"][0]["summary_excerpt"])
        self.assertEqual(result["bursts"][1]["activity_x"], 5.5)
        self.assertNotIn(5, [row["snapshot"] for row in result["bursts"]])
        self.assertEqual(result["backpressure_gap_count"], 1)
        self.assertIn("unknowable, not absent", result["backpressure_note"])
        self.assertIn("not semantic proof", result["semantics"])

        strict = tools.execute(
            "list_attention_bursts",
            {"channel_id": 7, "from_ts": 90.0, "to_ts": 200.0, "min_activity_x": 6.0},
        )
        self.assertEqual(strict["burst_count"], 1)
        self.assertEqual(strict["bursts"][0]["activity_x"], 11.1)

    def test_system_prompt_routes_spike_questions_to_the_burst_tool(self):
        prompt = build_system_prompt(
            _ProbeStore(),
            _DetectionStore(),
            _SummaryManager(),
        )
        self.assertIn("call list_attention_bursts FIRST", prompt)
        self.assertIn("unknown intervals, never as calm", prompt)

    def test_capture_attention_survives_model_compaction(self):
        compact = _compact_vector_signal_for_model(
            {
                "capture_attention": {
                    "policy": "capture_per_second_cv_apex_v2",
                    "baseline": {"level": 0.0012, "warmup": False},
                    "seconds": [
                        {"snapshot": 3, "mode": "burst", "activity_x": 11.12, "sharper_companion": True},
                        {"snapshot": 5, "mode": "normal", "activity_x": 3.4},
                        {"snapshot": 6, "mode": "quiet"},
                        {"snapshot": None, "mode": "burst"},
                    ],
                }
            }
        )

        attention = compact["capture_attention"]
        self.assertEqual(attention["baseline"], {"level": 0.0012, "warmup": False})
        self.assertEqual(
            attention["seconds"],
            [
                {"snapshot": 3, "mode": "burst", "activity_x": 11.12, "sharper_companion": True},
                {"snapshot": 5, "mode": "normal", "activity_x": 3.4},
            ],
        )

    def test_burst_windows_outrank_plain_summaries_for_evidence(self):
        quiet_node = {"summary": "corridor is calm", "alert_total": 0}
        burst_node = {
            "summary": "corridor is calm",
            "alert_total": 0,
            "vector_signal": {
                "capture_attention": {
                    "seconds": [
                        {"snapshot": 2, "mode": "burst", "activity_x": 9.5},
                        {"snapshot": 4, "mode": "burst", "activity_x": 4.1},
                    ]
                }
            },
        }

        self.assertEqual(_summary_node_alert_score(quiet_node), 0)
        self.assertEqual(_summary_node_alert_score(burst_node), 4)
        self.assertGreater(
            _summary_node_alert_score(burst_node),
            _summary_node_alert_score(quiet_node),
        )

    def test_system_prompt_explains_capture_attention_semantics(self):
        prompt = build_system_prompt(
            _ProbeStore(),
            _DetectionStore(),
            _SummaryManager(),
        )

        self.assertIn("vector_signal.capture_attention", prompt)
        self.assertIn("mode=burst", prompt)
        self.assertIn("Motion blur on burst frames is expected physics", prompt)
        self.assertIn("anchor_role=burst_companion", prompt)
        self.assertIn("statistical attention, not semantic proof", prompt)

    def test_turn_context_applies_normalized_time_window_to_generate_report(self):
        context = _seed_turn_tool_context("Generate a video report for the selected channel.")
        context["channel_id"] = 7
        context["time_window"] = {
            "from_ts": 100.0,
            "to_ts": 200.0,
            "since_ms": 100_000,
            "until_ms": 200_000,
        }

        args = _apply_turn_tool_context("generate_report", {}, context)

        self.assertEqual(args["channel_id"], 7)
        self.assertEqual(args["from_ts"], 100.0)
        self.assertEqual(args["to_ts"], 200.0)


if __name__ == "__main__":
    unittest.main()
