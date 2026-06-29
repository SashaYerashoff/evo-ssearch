import unittest

from agent import (
    AGENT_VIDEO_SUMMARY_CHANNELS_PER_TURN,
    AgentTools,
    _apply_turn_tool_context,
    _compact_prompt_settings_for_model,
    _compact_tool_result_for_model,
    _format_turn_signal_ledger_message,
    _new_turn_signal_ledger,
    _record_turn_signal_ledger,
    _remember_turn_tool_result,
    _seed_turn_tool_context,
    _safe_detection,
    _strip_thumbnails,
    _tool_result_for_ui,
    build_system_prompt,
)


class _SummaryManager:
    def __init__(self):
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
                    "dropped_frames": 1,
                    "queue_dropped_batches": 0,
                    "log_count": 12,
                    "last_alert_counts": {"low": 1},
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

    def summary_rollups(self, channel_id, run_selector=None, start_ts=None, end_ts=None, level_limit=None):
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


def _tools(manager=None, search_detections_fn=None, detections_store=None, call_lm_fn=None, embed_text_fn=None, probes_store=None):
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
            }
        )

        self.assertTrue(compact["prompt_health"]["needs_migration"])
        self.assertEqual(compact["prompt_health"]["suggested_alert_policy_prompt"], "Flag people fighting.")
        self.assertIn("L0 live-description role/style", compact["prompt_layers"]["stream"]["semantics"])
        self.assertIn("Operator watch/alert criteria", compact["prompt_layers"]["alerts"]["semantics"])
        self.assertIn("Machine-readable ALERTS_JSON", compact["prompt_layers"]["json"]["semantics"])
        self.assertIn("compressed memory maps", compact["prompt_layers"]["rollups"]["semantics"])

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
        self.assertIn("recent_alerts=Doorway activity", prompt)
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
        self.assertFalse(result["requires_confirmation"])
        self.assertEqual(
            [row["channel_id"] for row in result["candidate_channels"]],
            [7, 8],
        )
        row7 = result["candidate_channels"][0]
        self.assertEqual(row7["recent_alerts"][0]["title"], "Doorway activity")
        self.assertEqual(row7["status_digest"]["alert_delivery_breakdown"]["sent"], 1)
        compact = _compact_tool_result_for_model("list_video_summary_channels", result)
        self.assertEqual(compact["candidate_channels"][0]["recent_alerts"][0]["title"], "Doorway activity")

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
