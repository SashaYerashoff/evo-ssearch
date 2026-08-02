"""Deterministic unit tests for the integration harness (run in the normal suite).

These verify the SSE parser, the Transcript accessors, and the scenario checker
without any live service, so the harness logic is regression-covered even though
the live agent smoke is opt-in.
"""
import json
import unittest

from tests.integration.eva_client import Transcript, combine_transcripts, parse_sse_events
from tests.integration.scenarios import (
    SCENARIOS,
    AnyResultCheck,
    AnyToolCheck,
    Scenario,
    ToolOrderCheck,
    ToolCheck,
    ResultCheck,
    UiEffectCheck,
    run_scenario,
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

    def test_tool_calls_and_results(self) -> None:
        self.assertTrue(self.t.called("calibrate_probe_from_archive"))
        self.assertEqual(self.t.calls_of("update_probe"), [{"preview": False, "probe_id": "p1"}])
        self.assertEqual(self.t.result_of("calibrate_probe_from_archive")["items"][0]["safe_to_apply"], False)

    def test_text_session_and_approval(self) -> None:
        self.assertEqual(self.t.text, "Use the UI Apply button.")
        self.assertEqual(self.t.session_id, "s1")
        self.assertEqual(self.t.approval_plan_ids(), ["plan-1"])
        self.assertTrue(self.t.prose_has(r"apply button"))
        self.assertFalse(self.t.errored)
        self.assertTrue(self.t.finished)
        self.assertEqual(self.t.tool_call_count, 2)
        self.assertEqual(self.t.dangling_tool_calls, [])
        self.assertEqual(self.t.budget_stops, [])
        self.assertEqual(self.t.ui_effects[0]["target"], "probes")

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
