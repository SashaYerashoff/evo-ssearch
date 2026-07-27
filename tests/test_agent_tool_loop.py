import json
import time
import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

import agent
from agent import (
    AgentRunner,
    TRUSTED_ACTION_RECEIPT_PREFIX,
    _LMResponse,
    _ToolCall,
    _compact_tool_result_for_model,
    _compact_tool_messages_for_context_budget,
    _coalesce_system_messages,
    _context_budget_snapshot,
    _filter_streamed_tool_markup,
    _parse_text_tool_calls,
    _select_relevant_tool_schemas,
    _seed_turn_tool_context,
    _apply_turn_tool_context,
    _AgentLMClient,
)
from agent_security import ToolExecutionContext


class _FakeStore:
    def __init__(self, history=None, research_state=None) -> None:
        self.messages = []
        self.history = history
        self.research_state = research_state

    def session_exists(self, _session_id, **_owner):
        return True

    def create_session(self, **_owner):
        return "session-1"

    def touch_session(self, _session_id, title=None, **_owner):
        return None

    def add_message(self, session_id, **message):
        self.messages.append({"session_id": session_id, **message})

    def load_history(self, _session_id, **_owner):
        if self.history is not None:
            return list(self.history)
        return [{"role": "user", "content": "test"}]

    def load_research_state(self, _session_id, **_owner):
        return self.research_state

    def save_research_state(self, _session_id, state, **_owner):
        self.research_state = dict(state)


class _FakeLMClient:
    def __init__(
        self,
        tool_rounds: int,
        tool_name: str = "list_channels",
        *,
        distinct_args: bool = False,
    ) -> None:
        self.remaining = tool_rounds
        self.tool_name = tool_name
        self.distinct_args = distinct_args
        self.tools = None
        self.final_messages = None
        self.tool_call_messages = []

    def call_with_tools(self, _messages, tools=None, cancel_event=None):
        self.tools = tools
        self.tool_call_messages.append(list(_messages))
        if self.remaining > 0:
            round_number = self.remaining
            self.remaining -= 1
            return _LMResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    _ToolCall(
                        id=f"call-{round_number}",
                        name=self.tool_name,
                        args={"round": round_number} if self.distinct_args else {},
                    )
                ],
            )
        return _LMResponse(content="done", finish_reason="stop", tool_calls=[])

    def stream_text(self, messages, cancel_event=None):
        self.final_messages = list(messages)
        yield "done"


class _FakeTools:
    def __init__(self, result=None) -> None:
        self.calls = 0
        self.call_args = []
        self.result = result

    def execute(self, name, args, progress_cb=None):
        self.calls += 1
        self.call_args.append((name, dict(args)))
        if self.result is not None:
            return self.result
        return {"name": name, "args": args, "count": self.calls}


class _FakeSecureTools:
    def __init__(self) -> None:
        self.calls = []

    def available_tool_schemas(self, _context):
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_channels",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    def execute(self, name, args, context, progress_cb=None):
        self.calls.append((name, args, context))
        return {"name": name, "args": args}

    def visible_probes(self, _context):
        return []


class _FakeApprovalTools:
    def __init__(self, result=None) -> None:
        self.calls = []
        self.result = result or {
            "status": "applied",
            "action_receipt": {
                "type": "agent_action_applied",
                "plan_id": "plan-1",
                "tool": "update_probe",
                "status": "applied",
                "result_status": "applied",
                "probe_id": "probe-7",
                "probe_name": "Door",
                "channel_id": 7,
            },
        }

    def approve_and_execute(self, plan_id, context):
        self.calls.append((plan_id, context))
        return self.result


class AgentToolLoopTests(unittest.TestCase):
    def test_system_messages_are_coalesced_at_front_for_strict_chat_templates(self):
        messages = [
            {"role": "system", "content": "base rules"},
            {"role": "user", "content": "inspect"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "list_channels", "arguments": "{}"},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "list_channels",
                "content": "{}",
            },
            {"role": "system", "content": "trusted turn ledger"},
        ]

        normalized = _coalesce_system_messages(messages)

        self.assertEqual(normalized[0], {
            "role": "system",
            "content": "base rules\n\ntrusted turn ledger",
        })
        self.assertEqual([item["role"] for item in normalized], [
            "system", "user", "assistant", "tool",
        ])
        self.assertEqual(messages[-1]["role"], "system")

    def test_tool_schemas_are_routed_by_operator_intent(self):
        def names(query, *, inventory_complete=False):
            context = _seed_turn_tool_context(query)
            if inventory_complete:
                context["video_inventory_completed"] = True
            return {
                row["function"]["name"]
                for row in _select_relevant_tool_schemas(agent._TOOL_SCHEMAS, context)
            }

        self.assertEqual(names("Hello, introduce yourself"), set())
        self.assertEqual(
            names("Show current active streams, models, queues and dropped frames"),
            {"list_video_summary_channels"},
        )
        self.assertEqual(
            names("Show recent VLM alerts and notable video-summary events for the last hour"),
            {"normalize_time_window", "list_video_summary_channels"},
        )
        detailed = names(
            "Show recent VLM alerts and notable video-summary events for the last hour",
            inventory_complete=True,
        )
        self.assertIn("get_video_summaries", detailed)
        self.assertIn("describe_frame", detailed)
        self.assertNotIn("create_probe", detailed)
        self.assertNotIn("update_prompt_settings", detailed)
        self.assertEqual(
            names("How do I open the archive review?"),
            {"lookup_help"},
        )
        probe_tools = names("Create a CLIP probe on channel #112 for a visible lighter flame")
        self.assertIn("create_probe", probe_tools)
        self.assertIn("calibrate_probe_from_archive", probe_tools)
        self.assertNotIn("get_video_summaries", probe_tools)

    def test_routing_repairs_common_operator_typos_and_inherits_followup_intent(self):
        initial = _seed_turn_tool_context(
            "Hi! Tell me about what happend this night"
        )
        self.assertIn("video_research", initial["tool_intents"])

        followup = _seed_turn_tool_context("2. the las 24 hours.")
        agent._inherit_followup_tool_context(
            followup,
            "2. the las 24 hours.",
            [
                {
                    "role": "user",
                    "content": "Hi! Tell me about what happend this night",
                },
                {"role": "assistant", "content": "Which period?"},
            ],
        )
        self.assertIn("video_research", followup["tool_intents"])
        self.assertEqual(followup["operator_relative_range"], "last 24 hours")
        self.assertTrue(followup["inherited_operator_intent"])

        continued = _seed_turn_tool_context("continue")
        agent._inherit_followup_tool_context(
            continued,
            "continue",
            [
                {
                    "role": "user",
                    "content": "Hi! Tell me about what happend this night",
                },
                {"role": "assistant", "content": "Which period?"},
                {"role": "user", "content": "2. the las 24 hours."},
                {"role": "assistant", "content": "I will normalize it."},
            ],
        )
        self.assertIn("video_research", continued["tool_intents"])
        self.assertEqual(continued["operator_relative_range"], "last 24 hours")

    def test_video_period_research_executes_required_reads_before_model_narrative(self):
        class ResearchTools(_FakeTools):
            def execute(self, name, args, progress_cb=None):
                self.calls += 1
                self.call_args.append((name, dict(args)))
                window = {
                    "from_ts": 100.0,
                    "to_ts": 86_500.0,
                    "since_ms": 100_000,
                    "until_ms": 86_500_000,
                    "duration_sec": 86_400,
                    "relative_range": "last 24 hours",
                }
                if name == "normalize_time_window":
                    return dict(window)
                if name == "list_video_summary_channels":
                    return {
                        "time_window": dict(window),
                        "requested_channel_ids": [112, 118],
                        "checked_channel_ids": [112, 118],
                        "candidate_channels": [
                            {"channel_id": 112, "summary_count": 10},
                            {"channel_id": 118, "summary_count": 20},
                        ],
                        "inactive_channel_ids": [],
                        "deferred_channel_ids": [],
                        "unchecked_channel_ids": [],
                        "errors": [],
                        "active_count": 2,
                        "inactive_count": 0,
                        "error_count": 0,
                    }
                if name == "get_video_summaries":
                    return {
                        "channel_id": args["channel_id"],
                        "depth": args["depth"],
                        "time_window": dict(window),
                        "count": 1,
                        "total_in_window": 1,
                        "coverage": {"status": "covered"},
                        "entries": [{"summary": "observed activity"}],
                    }
                raise AssertionError(name)

        history = [
            {
                "role": "user",
                "content": "Hi! Tell me about what happend this night",
            },
            {"role": "assistant", "content": "Which period?"},
            {"role": "user", "content": "2. the las 24 hours."},
        ]
        runner = AgentRunner.__new__(AgentRunner)
        runner.store = _FakeStore(history=history)
        runner._lm_client = _FakeLMClient(tool_rounds=0)
        runner._tools = ResearchTools()
        runner._ps = object()
        runner._ds = object()
        runner._lxm = object()

        events = [
            json.loads(item.removeprefix("data: ").strip())
            for item in runner.stream_chat(
                "session-1",
                "2. the las 24 hours.",
            )
            if item.startswith("data: ")
        ]

        self.assertEqual(
            [name for name, _args in runner._tools.call_args],
            [
                "normalize_time_window",
                "list_video_summary_channels",
                "get_video_summaries",
                "get_video_summaries",
            ],
        )
        self.assertEqual(runner._tools.call_args[0][1]["relative_range"], "last 24 hours")
        self.assertEqual(runner._tools.call_args[2][1]["depth"], "L2")
        self.assertEqual(runner._tools.call_args[3][1]["channel_id"], 118)
        self.assertEqual(
            sum(event.get("type") == "tool_call" for event in events),
            4,
        )
        self.assertTrue(
            any(event.get("type") == "research_plan_complete" for event in events)
        )
        self.assertEqual(runner._lm_client.tool_call_messages, [])

    def test_duplicate_video_read_is_suppressed_and_stops_tool_loop(self):
        runner = AgentRunner.__new__(AgentRunner)
        runner.store = _FakeStore()
        runner._lm_client = _FakeLMClient(
            tool_rounds=10,
            tool_name="get_video_summaries",
        )
        runner._tools = _FakeTools(
            result={
                "channel_id": 112,
                "depth": "L1",
                "count": 1,
                "total_in_window": 1,
                "coverage": {"status": "covered"},
                "entries": [{"summary": "stable result"}],
            }
        )
        runner._ps = object()
        runner._ds = object()
        runner._lxm = object()

        events = [
            json.loads(item.removeprefix("data: ").strip())
            for item in runner.stream_chat(
                "session-1",
                "Inspect video coverage for channel #112",
            )
            if item.startswith("data: ")
        ]

        self.assertEqual(runner._tools.calls, 1)
        self.assertEqual(
            sum(event.get("type") == "tool_call" for event in events),
            2,
        )
        guard = next(
            event for event in events if event.get("type") == "tool_loop_guard"
        )
        self.assertEqual(guard["reason"], "duplicate_read")
        duplicate_result = [
            event
            for event in events
            if event.get("type") == "tool_result"
        ][-1]
        self.assertTrue(duplicate_result["result"]["duplicate_suppressed"])

    def test_activated_runbook_tools_pass_the_intent_gate(self):
        query = "проверь канал 115, был ли почтальон вчера вечером?"
        context = _seed_turn_tool_context(query)
        # Without the runbook the RU phrasing matches no intent group.
        bare = _select_relevant_tool_schemas(agent._TOOL_SCHEMAS, context)
        self.assertEqual(bare, [])

        slugs = agent._extract_requested_skill_slugs(query)
        self.assertIn("video_event_check", slugs)
        skill_tools = agent._skill_tool_names(slugs)
        self.assertIn("get_video_summaries", skill_tools)
        self.assertIn("describe_frame", skill_tools)

        context["skill_tool_names"] = sorted(skill_tools)
        exposed = {
            row["function"]["name"]
            for row in _select_relevant_tool_schemas(agent._TOOL_SCHEMAS, context)
        }
        self.assertTrue(skill_tools.issubset(exposed))

    def test_skill_tool_names_ignores_unknown_and_stays_in_schema_envelope(self):
        self.assertEqual(agent._skill_tool_names([]), set())
        self.assertEqual(agent._skill_tool_names(["no_such_runbook"]), set())
        # Names never leave the provided (permission-filtered) schema list.
        context = _seed_turn_tool_context("привет! как дела?")
        context["skill_tool_names"] = ["get_video_summaries", "not_a_real_tool"]
        permitted = [
            row
            for row in agent._TOOL_SCHEMAS
            if row["function"]["name"] == "lookup_help"
        ]
        exposed = {
            row["function"]["name"]
            for row in _select_relevant_tool_schemas(permitted, context)
        }
        self.assertEqual(exposed, set())

    def test_write_tool_compaction_returns_stable_preview_envelope(self):
        raw = {
            "status": "preview",
            "action": "create",
            "exists": False,
            "proposed": {"name": "visible fire or smoke", "channel_id": 7},
            "conflicts": [{"id": f"p{i}", "name": f"probe {i}"} for i in range(12)],
            "approval": {
                "plan_id": "plan-123",
                "action": "create_probe",
                "expires_at": "2026-07-19T20:00:00+00:00",
                "required_permission": "probes:manage",
            },
        }
        compact = _compact_tool_result_for_model("create_probe", raw)
        self.assertEqual(compact["status"], "preview")
        self.assertEqual(compact["channel_id"], 7)
        self.assertEqual(len(compact["conflicts"]), 8)
        self.assertNotIn("approval", compact)
        self.assertEqual(compact["action_plan"]["plan_id"], "plan-123")
        self.assertEqual(compact["action_plan"]["status"], "awaiting_ui_apply")

        deleted = _compact_tool_result_for_model(
            "delete_probes",
            {"status": "applied", "delete_all": False, "deleted": 2,
             "targets": [{"id": "p1"}, {"id": "p2"}]},
        )
        self.assertEqual(deleted["deleted"], 2)
        self.assertEqual(len(deleted["targets"]), 2)
        self.assertNotIn("action_plan", deleted)

    def test_plain_chat_omits_empty_tools_payload(self):
        client = _AgentLMClient("http://agent.local/v1", "qwen3.5-9b-mtp", "", 120)

        class Admission:
            def admission(self, *_args, **_kwargs):
                return nullcontext()

        client.admission_controller = Admission()
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"finish_reason": "stop", "message": {"content": "hello"}}]
        }
        with patch.object(agent.requests, "post", return_value=response) as post:
            client.call_with_tools([{"role": "user", "content": "hello"}], tools=[])
        payload = post.call_args.kwargs["json"]
        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)

    def test_agent_disables_model_thinking_for_tool_and_final_calls(self):
        client = _AgentLMClient("http://agent.local/v1", "qwen3.5-9b-mtp", "", 120)

        class Admission:
            def admission(self, *_args, **_kwargs):
                return nullcontext()

            def status(self):
                return {"resources": []}

        client.admission_controller = Admission()
        tool_response = Mock()
        tool_response.raise_for_status.return_value = None
        tool_response.json.return_value = {
            "choices": [{"finish_reason": "stop", "message": {"content": "done"}}]
        }
        with patch.object(agent.requests, "post", return_value=tool_response) as post:
            client.call_with_tools([{"role": "user", "content": "inspect"}], tools=[])
        self.assertEqual(
            post.call_args.kwargs["json"]["chat_template_kwargs"],
            {"enable_thinking": False},
        )

        class StreamResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def raise_for_status(self):
                return None

            def iter_lines(self):
                yield b'data: {"choices":[{"delta":{"content":"done"}}]}'
                yield b'data: [DONE]'

        with patch.object(agent.requests, "post", return_value=StreamResponse()) as post:
            self.assertEqual("".join(client.stream_text([{"role": "user", "content": "finish"}])), "done")
        self.assertEqual(
            post.call_args.kwargs["json"]["chat_template_kwargs"],
            {"enable_thinking": False},
        )

    def test_recovers_allowed_xml_tool_call_and_strips_protocol_markup(self):
        content = """I will prepare the preview.\n<tool_call>
<function=update_prompt_settings>
<parameter=channel_id>112</parameter>
<parameter=changes>{"alert_policy_prompt":"Alert when a visible lighter flame appears in a person's hand."}</parameter>
<parameter=preview>True</parameter>
</function>
</tool_call>"""
        cleaned, calls, saw_markup = _parse_text_tool_calls(
            content,
            allowed_names={"get_prompt_settings", "update_prompt_settings"},
        )

        self.assertTrue(saw_markup)
        self.assertEqual(cleaned, "I will prepare the preview.")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "update_prompt_settings")
        self.assertEqual(calls[0].args["channel_id"], 112)
        self.assertTrue(calls[0].args["preview"])

        cleaned, calls, _ = _parse_text_tool_calls(
            content.replace("update_prompt_settings", "create_probe"),
            allowed_names={"get_prompt_settings", "update_prompt_settings"},
        )
        self.assertEqual(calls, [])
        self.assertNotIn("<tool_call>", cleaned or "")

    def test_stream_filter_removes_split_tool_protocol(self):
        chunks = ["Answer before <tool_", "call><function=create_probe>", "secret", "</tool_call> after"]
        self.assertEqual("".join(_filter_streamed_tool_markup(chunks)), "Answer before  after")

    def test_vlm_alert_request_routes_to_prompt_policy_without_disabling_explicit_probes(self):
        self.assertFalse(
            _seed_turn_tool_context("Show the latest VLM alert for channel #112")["vlm_alert_policy_request"]
        )
        self.assertFalse(
            _seed_turn_tool_context("Покажи последние VLM-алерты канала #112")["vlm_alert_policy_request"]
        )
        context = _seed_turn_tool_context(
            "set a new alert for the vlm channel #112 - if person fires a lighter"
        )
        self.assertTrue(context["vlm_alert_policy_request"])
        self.assertEqual(context["channel_id"], 112)
        context["prompt_settings_current"] = {"alert_policy_prompt": "Existing criterion."}
        prepared = _apply_turn_tool_context(
            "update_prompt_settings",
            {"changes": {
                "alert_policy_prompt": "model wording",
                "rollup_prompts": {"L1": "must not leak"},
                "bookmark_enabled": True,
                "migrate_legacy_alert_policy": True,
            }},
            context,
        )
        self.assertEqual(prepared["channel_id"], 112)
        self.assertTrue(prepared["preview"])
        self.assertEqual(
            prepared["changes"]["alert_policy_prompt"],
            "Existing criterion.\nAlert when a person ignites a lighter and a visible small flame appears in or near the person's hand.",
        )
        self.assertEqual(set(prepared["changes"]), {"alert_policy_prompt"})
        explicit_probe = _seed_turn_tool_context(
            "create a CLIP probe on VLM channel #112 for a visible lighter flame"
        )
        self.assertFalse(explicit_probe["vlm_alert_policy_request"])

    def test_context_budget_counts_tool_schemas_and_emergency_compacts_results(self):
        messages = [
            {"role": "system", "content": "rules"},
            {"role": "tool", "name": "get_video_summaries", "tool_call_id": "c1", "content": json.dumps({
                "channel_id": 7,
                "depth": "L1",
                "count": 10,
                "entries": [{"time": str(index), "summary": "x" * 3000} for index in range(10)],
            })},
        ]
        schemas = [{"type": "function", "function": {"name": "wide", "description": "y" * 3000}}]
        with_tools = _context_budget_snapshot(messages, tool_schemas=schemas)
        without_tools = _context_budget_snapshot(messages)
        compacted, status = _compact_tool_messages_for_context_budget(messages, token_budget=1000)

        self.assertGreater(with_tools["estimated_tokens"], without_tools["estimated_tokens"])
        self.assertGreater(with_tools["tool_schema_estimated_tokens"], 0)
        self.assertEqual(status["compacted_tool_messages"], 1)
        self.assertLess(len(compacted[1]["content"]), len(messages[1]["content"]))

    def test_video_summary_model_compaction_keeps_signal_without_verbose_contract(self):
        raw = {
            "channel_id": 112,
            "depth": "L0",
            "count": 25,
            "total_in_window": 154,
            "semantic_available_count": 154,
            "semantic_status": "ready",
            "truncated": True,
            "coverage": {
                "status": "truncated",
                "truncated": True,
                "selection_strategy": "period_sample_alert_priority",
                "note": "n" * 2000,
                "available": {
                    "entry_count": 154,
                    "status": "covered",
                    "first_time": "2026-07-13 09:27",
                    "last_time": "2026-07-13 10:27",
                    "large_internal_gaps": [{"detail": "x" * 2000}],
                },
                "returned": {"entry_count": 25, "status": "partial"},
            },
            "source_coverage": {"status": "covered", "available": {"entry_count": 154}},
            "evidence_frames": [
                {"id": 296781, "channel_id": 112, "source": "vlm_summary", "image_url": "/detections/thumbnail/296781"}
            ],
            "entries": [
                {
                    "time": f"10:{index:02d}",
                    "summary": "semantic narrative " + ("x" * 1500),
                    "alert_events": [{"title": "event"}] * 10,
                }
                for index in range(25)
            ],
        }

        compact = _compact_tool_result_for_model("get_video_summaries", raw)

        self.assertEqual(len(compact["entries"]), 5)
        self.assertEqual(len(compact["entries"][0]["summary"]), 700)
        self.assertEqual(len(compact["entries"][0]["alert_events"]), 4)
        self.assertEqual(compact["evidence_frames"][0]["image_url"], "/detections/thumbnail/296781")
        self.assertNotIn("large_internal_gaps", json.dumps(compact["coverage"]))
        self.assertLess(len(json.dumps(compact)), 9_000)

    def test_incomplete_final_response_uses_completed_tool_ledger(self):
        class IncompleteLM(_FakeLMClient):
            def stream_text(self, messages, cancel_event=None):
                self.final_messages = list(messages)
                yield "Let me fetch the remaining frames."

        runner = AgentRunner.__new__(AgentRunner)
        runner.store = _FakeStore()
        runner._lm_client = IncompleteLM(tool_rounds=1, tool_name="get_video_summaries")
        runner._tools = _FakeTools(result={
            "channel_id": 112,
            "depth": "L0",
            "count": 25,
            "total_in_window": 154,
            "semantic_status": "ready",
            "coverage": {"status": "truncated", "truncated": True},
            "evidence_frames": [{"id": 296781, "image_url": "/detections/thumbnail/296781"}],
            "evidence_frame_totals": {"vlm_summary": 154},
            "entries": [],
        })
        runner._ps = object()
        runner._ds = object()
        runner._lxm = object()

        events = [
            json.loads(item.removeprefix("data: ").strip())
            for item in runner.stream_chat("session-1", "Show VLM alerts for channel #112")
            if item.startswith("data: ")
        ]

        self.assertTrue(any(item.get("type") == "completion_recovery" for item in events))
        final_text = "".join(item.get("content", "") for item in events if item.get("type") == "text")
        self.assertIn("CH 112", final_text)
        self.assertNotIn("Let me fetch", final_text)

    def test_archive_tool_compaction_preserves_source_semantics(self):
        compact = _compact_tool_result_for_model(
            "search_archive",
            {
                "scope": "detections",
                "source": "vlm_summary",
                "source_label": "Video-description frame",
                "count": 1,
                "results": [
                    {
                        "detection_id": 42,
                        "image_path": "/tmp/frame.jpg",
                        "score": 0.78,
                        "timestamp_ms": 1_781_389_900_000,
                        "source": "vlm_summary",
                        "source_label": "Video-description frame",
                        "archive_item_type": "video_description_frame",
                        "probe_name": "VLM summary frame",
                        "channel_id": 7,
                    }
                ],
            },
        )

        self.assertEqual(compact["source"], "vlm_summary")
        self.assertEqual(compact["source_label"], "Video-description frame")
        self.assertEqual(compact["results"][0]["source"], "vlm_summary")
        self.assertEqual(compact["results"][0]["source_label"], "Video-description frame")
        self.assertEqual(
            compact["results"][0]["archive_item_type"],
            "video_description_frame",
        )

    def test_tool_loop_can_exceed_eight_rounds(self):
        runner = AgentRunner.__new__(AgentRunner)
        runner.store = _FakeStore()
        runner._lm_client = _FakeLMClient(tool_rounds=12, distinct_args=True)
        runner._tools = _FakeTools()
        runner._ps = object()
        runner._ds = object()
        runner._lxm = object()

        events = list(runner.stream_chat("session-1", "test"))
        payloads = [
            json.loads(event.removeprefix("data: ").strip())
            for event in events
            if event.startswith("data: ")
        ]

        self.assertEqual(runner._tools.calls, 12)
        self.assertEqual(sum(item.get("type") == "tool_call" for item in payloads), 12)
        self.assertFalse(any("exceeded" in item.get("message", "") for item in payloads))
        self.assertEqual(payloads[-1]["type"], "done")

    def test_authorized_loop_filters_schemas_and_uses_secure_dispatch(self):
        runner = AgentRunner.__new__(AgentRunner)
        runner.store = _FakeStore()
        runner._lm_client = _FakeLMClient(tool_rounds=1)
        runner._tools = _FakeTools()
        runner._secure_tools = _FakeSecureTools()
        runner._ps = object()
        runner._ds = object()
        runner._lxm = object()
        context = ToolExecutionContext(
            actor_id="361fe45f-f277-42f8-ae35-eaa0fc81cf38",
            tenant_id="59da6ca3-51b7-4d91-9190-aae06b76d846",
            roles={"operator"},
            permissions={"agent:use", "streams:view"},
            allowed_channel_ids={"7"},
            request_id="request-1",
        )

        list(runner.stream_chat("session-1", "list channels", tool_context=context))

        self.assertEqual(
            runner._lm_client.tools[0]["function"]["name"],
            "list_channels",
        )
        self.assertEqual(runner._tools.calls, 0)
        self.assertEqual(len(runner._secure_tools.calls), 1)
        self.assertEqual(
            runner._secure_tools.calls[0][2].session_id,
            "session-1",
        )

    def test_tool_loop_has_high_but_finite_budget(self):
        runner = AgentRunner.__new__(AgentRunner)
        runner.store = _FakeStore()
        runner._lm_client = _FakeLMClient(tool_rounds=100, distinct_args=True)
        runner._tools = _FakeTools()
        runner._ps = object()
        runner._ds = object()
        runner._lxm = object()

        events = list(runner.stream_chat("session-1", "test"))
        payloads = [
            json.loads(event.removeprefix("data: ").strip())
            for event in events
            if event.startswith("data: ")
        ]

        self.assertEqual(runner._tools.calls, 64)
        self.assertTrue(any(item.get("type") == "tool_budget" for item in payloads))
        self.assertEqual(payloads[-1]["type"], "done")

    def test_video_research_has_a_smaller_distinct_tool_budget(self):
        class DistinctVideoLM(_FakeLMClient):
            def __init__(self):
                super().__init__(tool_rounds=100, tool_name="get_video_summaries")
                self.round_number = 0

            def call_with_tools(self, _messages, tools=None, cancel_event=None):
                self.tools = tools
                self.tool_call_messages.append(list(_messages))
                self.round_number += 1
                return _LMResponse(
                    content="",
                    finish_reason="tool_calls",
                    tool_calls=[
                        _ToolCall(
                            id=f"video-{self.round_number}",
                            name="get_video_summaries",
                            args={"limit": self.round_number},
                        )
                    ],
                )

        runner = AgentRunner.__new__(AgentRunner)
        runner.store = _FakeStore()
        runner._lm_client = DistinctVideoLM()
        runner._tools = _FakeTools(
            result={
                "channel_id": 112,
                "depth": "L1",
                "count": 1,
                "total_in_window": 1,
                "coverage": {"status": "covered"},
                "entries": [{"summary": "result"}],
            }
        )
        runner._ps = object()
        runner._ds = object()
        runner._lxm = object()

        events = [
            json.loads(item.removeprefix("data: ").strip())
            for item in runner.stream_chat(
                "session-1",
                "Inspect video coverage for channel #112",
            )
            if item.startswith("data: ")
        ]

        self.assertEqual(
            runner._tools.calls,
            agent.AGENT_VIDEO_RESEARCH_MAX_TOOL_CALLS,
        )
        budget = next(item for item in events if item.get("type") == "tool_budget")
        self.assertEqual(
            budget["max_tool_calls"],
            agent.AGENT_VIDEO_RESEARCH_MAX_TOOL_CALLS,
        )

    def test_continue_uses_persisted_channel_ids_and_frozen_window(self):
        research_state = {
            "version": 1,
            "kind": "video_summary_inventory",
            "status": "pending",
            "frozen_window": {"from_ts": 100.0, "to_ts": 200.0},
            "requested_channel_ids": [1, 2, 3, 4],
            "completed_channel_ids": [1, 2],
            "remaining_channel_ids": [3, 4],
            "updated_at": time.time(),
        }
        result = {
            "time_window": {
                "from_ts": 100.0,
                "to_ts": 200.0,
                "since_ms": 100_000,
                "until_ms": 200_000,
            },
            "requested_channel_ids": [3, 4],
            "checked_channel_ids": [3, 4],
            "candidate_channels": [{"channel_id": 3}],
            "deferred_channel_ids": [4],
            "inactive_channel_ids": [],
            "unchecked_channel_ids": [],
            "errors": [],
        }
        runner = AgentRunner.__new__(AgentRunner)
        runner.store = _FakeStore(research_state=research_state)
        runner._lm_client = _FakeLMClient(
            tool_rounds=1,
            tool_name="list_video_summary_channels",
        )
        runner._tools = _FakeTools(result=result)
        runner._ps = object()
        runner._ds = object()
        runner._lxm = object()

        events = list(runner.stream_chat("session-1", "Continue with the remaining channels"))
        payloads = [
            json.loads(event.removeprefix("data: ").strip())
            for event in events
            if event.startswith("data: ")
        ]

        tool_name, tool_args = runner._tools.call_args[0]
        self.assertEqual(tool_name, "list_video_summary_channels")
        self.assertEqual(tool_args["channel_ids"], [3, 4])
        self.assertEqual(tool_args["from_ts"], 100.0)
        self.assertEqual(tool_args["to_ts"], 200.0)
        self.assertEqual(runner.store.research_state["completed_channel_ids"], [1, 2, 3])
        self.assertEqual(runner.store.research_state["remaining_channel_ids"], [4])
        state_event = next(item for item in payloads if item.get("type") == "research_state")
        self.assertTrue(state_event["persisted"])
        self.assertEqual(state_event["remaining_channel_ids"], [4])
        first_lm_prompt = runner._lm_client.tool_call_messages[0]
        self.assertTrue(
            any(
                "Trusted server research continuation ledger" in str(message.get("content") or "")
                for message in first_lm_prompt
            )
        )

    def test_unrelated_turn_does_not_apply_stale_research_defaults(self):
        research_state = {
            "version": 1,
            "kind": "video_summary_inventory",
            "status": "pending",
            "frozen_window": {"from_ts": 100.0, "to_ts": 200.0},
            "requested_channel_ids": [3, 4],
            "completed_channel_ids": [3],
            "remaining_channel_ids": [4],
            "updated_at": time.time(),
        }
        runner = AgentRunner.__new__(AgentRunner)
        runner.store = _FakeStore(research_state=research_state)
        runner._lm_client = _FakeLMClient(
            tool_rounds=1,
            tool_name="list_video_summary_channels",
        )
        runner._tools = _FakeTools(
            result={
                "time_window": {"from_ts": 900.0, "to_ts": 1_000.0},
                "requested_channel_ids": [],
                "checked_channel_ids": [],
                "candidate_channels": [],
                "inactive_channel_ids": [],
                "deferred_channel_ids": [],
                "unchecked_channel_ids": [],
                "errors": [],
            }
        )
        runner._ps = object()
        runner._ds = object()
        runner._lxm = object()

        list(runner.stream_chat("session-1", "Show current stream status"))

        _tool_name, tool_args = runner._tools.call_args[0]
        self.assertNotIn("channel_ids", tool_args)
        self.assertNotIn("from_ts", tool_args)
        self.assertNotIn("to_ts", tool_args)

    def test_signal_ledger_is_final_prompt_only_and_uses_compacted_results(self):
        raw_result = {
            "scope": "detections",
            "source": "vlm_summary",
            "source_label": "Video-description frame",
            "count": 1,
            "results": [
                {
                    "id": 42,
                    "detection_id": 42,
                    "timestamp_ms": 1_781_389_900_000,
                    "source": "vlm_summary",
                    "channel_id": 7,
                    "image_url": "/detections/thumbnail/42",
                    "thumbnail": "RAW_THUMBNAIL_SHOULD_NOT_LEAK",
                    "clip_vec": [0.1, 0.2, 0.3],
                    "private_note": "RAW_PRIVATE_NOTE_SHOULD_NOT_LEAK",
                }
            ],
        }
        runner = AgentRunner.__new__(AgentRunner)
        runner.store = _FakeStore()
        runner._lm_client = _FakeLMClient(tool_rounds=1, tool_name="search_archive")
        runner._tools = _FakeTools(result=raw_result)
        runner._ps = object()
        runner._ds = object()
        runner._lxm = object()

        list(runner.stream_chat("session-1", "find evidence"))

        final_messages = runner._lm_client.final_messages or []
        final_prompt = "\n".join(
            str(message.get("content") or "")
            for message in final_messages
            if isinstance(message, dict)
        )
        stored_text = "\n".join(str(message) for message in runner.store.messages)

        self.assertIn("Internal per-turn signal ledger", final_prompt)
        self.assertIn("Evidence/frame signals", final_prompt)
        self.assertIn("/detections/thumbnail/42", final_prompt)
        self.assertNotIn("RAW_THUMBNAIL_SHOULD_NOT_LEAK", final_prompt)
        self.assertNotIn("RAW_PRIVATE_NOTE_SHOULD_NOT_LEAK", final_prompt)
        self.assertNotIn("clip_vec", final_prompt)
        self.assertNotIn("Internal per-turn signal ledger", stored_text)

    def test_action_plan_apply_records_trusted_receipt(self):
        runner = AgentRunner.__new__(AgentRunner)
        runner.store = _FakeStore()
        runner._secure_tools = _FakeApprovalTools()
        context = ToolExecutionContext(
            actor_id="361fe45f-f277-42f8-ae35-eaa0fc81cf38",
            tenant_id="59da6ca3-51b7-4d91-9190-aae06b76d846",
            roles={"engineer"},
            permissions={"agent:use", "probes:manage"},
            allowed_channel_ids={"7"},
            agent_session_id="session-1",
            request_id="request-1",
        )

        result = runner.approve_action_plan("plan-1", context)

        self.assertEqual(result["status"], "applied")
        receipts = [
            message
            for message in runner.store.messages
            if message.get("role") == "system"
        ]
        self.assertEqual(len(receipts), 1)
        self.assertTrue(receipts[0]["content"].startswith(TRUSTED_ACTION_RECEIPT_PREFIX))
        self.assertIn('"tool": "update_probe"', receipts[0]["content"])
        self.assertIn('"status": "applied"', receipts[0]["content"])

    def test_preview_compaction_tells_model_to_wait_for_ui_apply(self):
        compact = _compact_tool_result_for_model(
            "create_probe",
            {
                "status": "preview",
                "action": "create_new",
                "exists": False,
                "proposed": {
                    "name": "person lying on ground",
                    "channel_id": 7,
                },
                "approval": {
                    "plan_id": "plan-1",
                    "action": "create_probe",
                    "expires_at": "2026-06-27T12:00:00+00:00",
                    "required_permission": "probes:manage",
                },
            },
        )

        self.assertEqual(compact["status"], "preview")
        self.assertEqual(compact["action_plan"]["status"], "awaiting_ui_apply")
        self.assertIn("UI Apply", compact["action_plan"]["next_step_hint"])
        self.assertNotIn("approval", compact)

    def test_history_is_trimmed_by_context_budget_without_persisting_notice(self):
        history = []
        for idx in range(12):
            history.append({"role": "user", "content": f"user {idx} " + ("x" * 800)})
            history.append({"role": "assistant", "content": f"assistant {idx} " + ("y" * 800)})
        history.append({"role": "user", "content": "current stored user"})
        runner = AgentRunner.__new__(AgentRunner)
        runner.store = _FakeStore(history=history)
        runner._lm_client = _FakeLMClient(tool_rounds=0)
        runner._tools = _FakeTools()
        runner._ps = object()
        runner._ds = object()
        runner._lxm = object()

        original_budget = agent.AGENT_CONTEXT_HISTORY_BUDGET_TOKENS
        try:
            agent.AGENT_CONTEXT_HISTORY_BUDGET_TOKENS = 900
            list(runner.stream_chat("session-1", "live current user"))
        finally:
            agent.AGENT_CONTEXT_HISTORY_BUDGET_TOKENS = original_budget

        final_messages = runner._lm_client.final_messages or []
        prompt_text = "\n".join(str(message.get("content") or "") for message in final_messages)
        stored_text = "\n".join(str(message) for message in runner.store.messages)

        self.assertIn("live current user", prompt_text)
        self.assertIn("history_trim", prompt_text)
        self.assertNotIn("user 0", prompt_text)
        self.assertNotIn("history_trim", stored_text)
        self.assertNotIn("Context budget signals", stored_text)

    def test_context_hard_stop_prevents_more_tools_without_persisting_notice(self):
        history = [{"role": "user", "content": "previous " + ("x" * 1200)}]
        runner = AgentRunner.__new__(AgentRunner)
        runner.store = _FakeStore(history=history)
        runner._lm_client = _FakeLMClient(tool_rounds=1)
        runner._tools = _FakeTools()
        runner._ps = object()
        runner._ds = object()
        runner._lxm = object()

        original_hard = agent.AGENT_CONTEXT_HARD_TOKENS
        original_warning = agent.AGENT_CONTEXT_WARNING_TOKENS
        try:
            agent.AGENT_CONTEXT_HARD_TOKENS = 200
            agent.AGENT_CONTEXT_WARNING_TOKENS = 100
            events = list(runner.stream_chat("session-1", "current " + ("z" * 1200)))
        finally:
            agent.AGENT_CONTEXT_HARD_TOKENS = original_hard
            agent.AGENT_CONTEXT_WARNING_TOKENS = original_warning

        payloads = [
            json.loads(event.removeprefix("data: ").strip())
            for event in events
            if event.startswith("data: ")
        ]
        stored_text = "\n".join(str(message) for message in runner.store.messages)

        self.assertEqual(runner._tools.calls, 0)
        self.assertTrue(
            any(item.get("type") == "context_budget" and item.get("status") == "hard_stop" for item in payloads)
        )
        self.assertNotIn("Agent context budget is near", stored_text)


if __name__ == "__main__":
    unittest.main()
