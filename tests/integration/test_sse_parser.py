"""Deterministic unit tests for the integration harness (run in the normal suite).

These verify the SSE parser, the Transcript accessors, and the scenario checker
without any live service, so the harness logic is regression-covered even though
the live agent smoke is opt-in.
"""
import json
import unittest

from tests.integration.eva_client import (
    SseDeadlineExceeded,
    Transcript,
    combine_transcripts,
    parse_sse_events,
)
from tests.integration.scenarios import (
    SCENARIOS,
    AnyResultCheck,
    AnyToolCheck,
    Scenario,
    ToolOrderCheck,
    ToolCheck,
    ResultCheck,
    UiEffectCheck,
    generation_quality,
    run_scenario,
    tool_efficiency,
)


def _sse(*objs) -> list:
    # mimic requests.iter_lines(): bytes, one per SSE 'data:' line
    return [f"data: {json.dumps(o)}".encode("utf-8") for o in objs]


SAMPLE = _sse(
    {"type": "session", "session_id": "s1"},
    {"type": "tool_call", "name": "calibrate_probe_from_archive", "args": {"channel_id": 112}},
    {"type": "tool_result", "name": "calibrate_probe_from_archive",
     "result": {"items": [{"safe_to_apply": False, "recommended_probe_args": None}]},
     "ui_effects": [{"target": "probes", "action": "show_preview",
                     "source": {"tool": "calibrate_probe_from_archive"}}]},
    {"type": "tool_call", "name": "update_probe", "args": {"preview": False, "probe_id": "p1"}},
    {"type": "tool_result", "name": "update_probe",
     "result": {"status": "preview", "approval": {"plan_id": "plan-1"}}},
    {"type": "text", "content": "Use the "},
    {"type": "text", "content": "UI Apply button."},
    {"type": "done", "session_id": "s1"},
)


class SseParserTest(unittest.TestCase):
    def setUp(self) -> None:
        self.t = Transcript(events=parse_sse_events(iter(SAMPLE)))

    def test_parser_skips_noise_and_decodes_json(self) -> None:
        noisy = [b": heartbeat", b"", b"data: not-json", b'data: {"type":"text","content":"x"}']
        events = parse_sse_events(iter(noisy))
        self.assertEqual(events, [{"type": "text", "content": "x"}])

    def test_parser_decodes_multiline_data_frame(self) -> None:
        multiline = [
            b"event: message",
            b'data: {"type":"text",',
            b'data: "content":"split"}',
            b"",
        ]
        events = parse_sse_events(iter(multiline))
        self.assertEqual(events, [{"type": "text", "content": "split"}])

    def test_parser_can_attach_client_receive_timing(self) -> None:
        values = iter((0.125, 0.5))
        events = parse_sse_events(
            iter(_sse(
                {"type": "tool_call", "call_id": "c1", "name": "list_probes", "args": {}},
                {"type": "tool_result", "call_id": "c1", "name": "list_probes", "result": {}},
            )),
            elapsed_fn=lambda: next(values),
        )

        self.assertEqual(events[0]["_received_at_sec"], 0.125)
        self.assertEqual(events[1]["_received_at_sec"], 0.5)

    def test_parser_enforces_wall_clock_deadline_even_with_heartbeats(self) -> None:
        values = iter((0.0, 2.0, 2.0))
        with self.assertRaises(SseDeadlineExceeded) as raised:
            parse_sse_events(
                iter([
                    b'data: {"type":"tool_call","name":"slow_tool","args":{}}',
                    b": heartbeat",
                ]),
                elapsed_fn=lambda: next(values),
                deadline_sec=1.0,
            )

        self.assertEqual(raised.exception.deadline_sec, 1.0)
        self.assertEqual(raised.exception.events[0]["name"], "slow_tool")

    def test_tool_calls_and_results(self) -> None:
        self.assertTrue(self.t.called("calibrate_probe_from_archive"))
        self.assertEqual(self.t.calls_of("update_probe"), [{"preview": False, "probe_id": "p1"}])
        self.assertEqual(self.t.result_of("calibrate_probe_from_archive")["items"][0]["safe_to_apply"], False)

    def test_text_session_and_approval(self) -> None:
        self.assertEqual(self.t.text, "Use the UI Apply button.")
        self.assertEqual(self.t.session_id, "s1")
        self.assertEqual(self.t.approval_plan_ids(), ["plan-1"])
        self.assertEqual(self.t.approval_plan_ids_for("update_probe"), ["plan-1"])
        self.assertTrue(self.t.prose_has(r"apply button"))
        self.assertFalse(self.t.errored)
        self.assertTrue(self.t.finished)
        self.assertEqual(self.t.tool_call_count, 2)
        self.assertEqual(self.t.dangling_tool_calls, [])
        self.assertEqual(self.t.budget_stops, [])
        self.assertEqual(self.t.ui_effects[0]["target"], "probes")

    def test_action_plan_compact_shape_is_also_discoverable(self) -> None:
        transcript = Transcript(events=[
            {
                "type": "tool_result",
                "name": "create_probe",
                "result": {"status": "preview", "action_plan": {"plan_id": "plan-2"}},
            },
        ])
        self.assertEqual(transcript.approval_plan_ids(), ["plan-2"])
        self.assertEqual(transcript.approval_plan_ids_for("create_probe"), ["plan-2"])

    def test_context_metrics_and_compact_tool_trace_are_reported(self) -> None:
        transcript = Transcript(events=parse_sse_events(iter(_sse(
            {
                "type": "context_metrics",
                "phase": "post_tool_batch",
                "estimated_tokens": 1234,
            },
            {
                "type": "tool_call",
                "call_id": "trace-1",
                "name": "get_video_summaries",
                "args": {"channel_id": 112},
            },
            {
                "type": "tool_result",
                "call_id": "trace-1",
                "name": "get_video_summaries",
                "result": {"count": 2},
            },
        ))))

        self.assertEqual(transcript.context_metrics[0]["estimated_tokens"], 1234)
        self.assertEqual(transcript.tool_trace[0]["name"], "get_video_summaries")
        self.assertGreater(transcript.tool_trace[0]["result_chars"], 0)

    def test_performance_metrics_pair_tools_and_sample_admission_queue(self) -> None:
        transcript = Transcript(
            events=[
                {"type": "tool_call", "call_id": "c1", "name": "list_probes", "args": {}, "_received_at_sec": 0.2},
                {"type": "tool_result", "call_id": "c1", "name": "list_probes", "result": {}, "_received_at_sec": 0.7},
                {"type": "text", "content": "Done.", "_received_at_sec": 1.0},
                {"type": "done", "_received_at_sec": 1.1},
            ],
            elapsed_seconds=1.2,
            telemetry_samples=[
                {
                    "path": "/lm/admission",
                    "at_sec": 0.0,
                    "payload": {"resources": [{
                        "resource": "http://127.0.0.1:1235/v1",
                        "active": 0,
                        "queued": 0,
                        "oldest_queue_age_sec": 0,
                        "average_wait_ms": 10,
                        "counters": {"admitted_total": 5, "admitted_agent": 5},
                    }]},
                },
                {
                    "path": "/lm/admission",
                    "at_sec": 1.2,
                    "payload": {"resources": [{
                        "resource": "http://127.0.0.1:1235/v1",
                        "active": 1,
                        "queued": 1,
                        "oldest_queue_age_sec": 0.2,
                        "average_wait_ms": 20,
                        "counters": {
                            "admitted_total": 7,
                            "admitted_agent": 7,
                            "queued_agent": 2,
                            "completed_agent": 1,
                        },
                    }]},
                },
            ],
        )

        metrics = transcript.performance_metrics

        self.assertEqual(metrics["tool_timings"][0]["duration_ms"], 500.0)
        self.assertEqual(metrics["lm_admission"]["agent_admissions"], 2)
        self.assertEqual(metrics["lm_admission"]["max_queued"], 1)
        self.assertEqual(metrics["lm_admission"]["max_oldest_queue_age_sec"], 0.2)

    def test_combined_workflow_keeps_setup_tools_but_only_final_prose(self) -> None:
        setup = Transcript(events=parse_sse_events(iter(_sse(
            {"type": "tool_call", "call_id": "c1", "name": "normalize_time_window", "args": {}},
            {"type": "tool_result", "call_id": "c1", "name": "normalize_time_window", "result": {}},
            {"type": "text", "content": "Please confirm."},
            {"type": "done", "session_id": "s1"},
        ))), elapsed_seconds=1.5)
        final = Transcript(events=parse_sse_events(iter(_sse(
            {"type": "tool_call", "call_id": "c2", "name": "get_video_summaries", "args": {}},
            {"type": "tool_result", "call_id": "c2", "name": "get_video_summaries", "result": {}},
            {"type": "text", "content": "Research complete."},
            {"type": "done", "session_id": "s1"},
        ))), elapsed_seconds=2.5)

        combined = combine_transcripts([setup, final])

        self.assertEqual([name for name, _args in combined.tool_calls], ["normalize_time_window", "get_video_summaries"])
        self.assertEqual(combined.text, "Research complete.")
        self.assertEqual(combined.elapsed_seconds, 4.0)
        self.assertTrue(combined.finished)

    def test_scenario_checker_hard_and_soft(self) -> None:
        scenario = Scenario(
            name="x",
            message="...",
            tool_checks=(
                ToolCheck("calibrate_probe_from_archive"),
                ToolCheck("create_probe", must_call=False),  # not called -> ok
            ),
            result_checks=(
                ResultCheck("calibrate_probe_from_archive",
                            lambda r: r["items"][0]["recommended_probe_args"] is None,
                            "unsafe -> no apply args"),
            ),
            prose_must=(r"apply",),
            prose_must_not=(r"stored approval",),
        )
        hard, soft = run_scenario(self.t, scenario)
        self.assertEqual(hard, [])
        self.assertEqual(soft, [])

    def test_scenario_checker_flags_forbidden_preview_false(self) -> None:
        scenario = Scenario(
            name="no-chat-apply",
            message="apply directly",
            tool_checks=(
                ToolCheck("update_probe", must_call=False,
                          arg_predicate=lambda a: a.get("preview") is False),
            ),
        )
        hard, _soft = run_scenario(self.t, scenario)
        self.assertTrue(any("forbidden" in f for f in hard))

    def test_scenario_checker_accepts_tool_alternatives(self) -> None:
        scenario = Scenario(
            name="calibrate-one-of",
            message="...",
            tool_checks=(AnyToolCheck(("prepare_probe_calibration_batch", "calibrate_probe_from_archive")),),
            result_checks=(
                AnyResultCheck(
                    ("prepare_probe_calibration_batch", "calibrate_probe_from_archive"),
                    lambda r: r["items"][0]["safe_to_apply"] is False,
                    "unsafe calibration seen",
                ),
            ),
        )
        hard, _soft = run_scenario(self.t, scenario)
        self.assertEqual(hard, [])

    def test_scenario_checker_enforces_order_ui_completion_and_budgets(self) -> None:
        scenario = Scenario(
            name="bounded",
            message="...",
            tool_order_checks=(
                ToolOrderCheck(("calibrate_probe_from_archive",), ("update_probe",), "calibrate first"),
            ),
            ui_effect_checks=(
                UiEffectCheck("probes", "show_preview", "calibrate_probe_from_archive"),
            ),
            max_tool_calls=2,
        )
        hard, _soft = run_scenario(self.t, scenario)
        self.assertEqual(hard, [])

        broken = Transcript(events=[
            {"type": "tool_call", "call_id": "dangling", "name": "search_archive", "args": {}},
            {"type": "tool_budget", "status": "exhausted"},
        ])
        hard, _soft = run_scenario(broken, Scenario(name="broken", message="...", max_tool_calls=0))
        self.assertTrue(any("without a done" in item for item in hard))
        self.assertTrue(any("without result" in item for item in hard))
        self.assertTrue(any("budget" in item for item in hard))
        self.assertTrue(any("exceeds" in item for item in hard))

    def test_ui_effect_payload_contract_is_checked(self) -> None:
        scenario = Scenario(
            name="payload",
            message="...",
            ui_effect_checks=(
                UiEffectCheck(
                    "probes",
                    "show_preview",
                    "calibrate_probe_from_archive",
                    payload_predicate=lambda payload: payload.get("name") == "expected",
                ),
            ),
        )

        hard, _soft = run_scenario(self.t, scenario)

        self.assertTrue(any("missing UI effect" in item for item in hard))

    def test_quality_and_efficiency_are_deterministic(self) -> None:
        transcript = Transcript(events=[
            {"type": "tool_call", "call_id": "c1", "name": "search_archive", "args": {"query": "person"}},
            {"type": "tool_result", "call_id": "c1", "name": "search_archive", "result": {}},
            {"type": "tool_call", "call_id": "c2", "name": "search_archive", "args": {"query": "person"}},
            {"type": "tool_result", "call_id": "c2", "name": "search_archive", "result": {}},
            {"type": "text", "content": "Coverage was checked for this channel. Coverage was checked for this channel."},
            {"type": "done"},
        ])
        scenario = Scenario(
            name="scored",
            message="...",
            prose_must=(r"coverage",),
            prose_must_not=(r"applied",),
            optimal_tool_calls=1,
        )

        quality = generation_quality(transcript, scenario)
        efficiency = tool_efficiency(transcript, scenario)

        self.assertLess(quality["score"], 100)
        self.assertEqual(quality["repeated_sentences"], 1)
        self.assertLess(efficiency["score"], 100)
        self.assertEqual(len(efficiency["exact_duplicate_calls"]), 1)

    def test_known_announce_without_execute_trace_is_rejected(self) -> None:
        scenario = next(
            item for item in SCENARIOS
            if item.name == "overnight_summary_completes_without_confirmation_loop"
        )
        announced_only = Transcript(events=parse_sse_events(iter(_sse(
            {"type": "text", "content": "Ready to proceed with time normalization."},
            {"type": "done", "session_id": "s1"},
        ))))

        hard, soft = run_scenario(announced_only, scenario)

        self.assertTrue(any("normalize_time_window" in item for item in hard))
        self.assertTrue(any("get_video_summaries" in item for item in hard))
        self.assertTrue(any("ready to proceed" in item for item in soft))

    def test_catalog_regex_fields_are_tuples(self) -> None:
        for scenario in SCENARIOS:
            for field_name in ("setup_messages", "prose_must", "prose_must_not", "requires"):
                value = getattr(scenario, field_name)
                self.assertIsInstance(value, tuple, f"{scenario.name}.{field_name}")
                self.assertTrue(
                    all(isinstance(item, str) for item in value),
                    f"{scenario.name}.{field_name}",
                )


if __name__ == "__main__":
    unittest.main()
