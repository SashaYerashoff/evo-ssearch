import json
import unittest

import agent
from agent import (
    AgentRunner,
    TRUSTED_ACTION_RECEIPT_PREFIX,
    _LMResponse,
    _ToolCall,
    _compact_tool_result_for_model,
)
from agent_security import ToolExecutionContext


class _FakeStore:
    def __init__(self, history=None) -> None:
        self.messages = []
        self.history = history

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


class _FakeLMClient:
    def __init__(self, tool_rounds: int, tool_name: str = "list_channels") -> None:
        self.remaining = tool_rounds
        self.tool_name = tool_name
        self.tools = None
        self.final_messages = None
        self.tool_call_messages = []

    def call_with_tools(self, _messages, tools=None):
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
                        args={},
                    )
                ],
            )
        return _LMResponse(content="done", finish_reason="stop", tool_calls=[])

    def stream_text(self, messages):
        self.final_messages = list(messages)
        yield "done"


class _FakeTools:
    def __init__(self, result=None) -> None:
        self.calls = 0
        self.result = result

    def execute(self, name, args, progress_cb=None):
        self.calls += 1
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
        runner._lm_client = _FakeLMClient(tool_rounds=12)
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

        list(runner.stream_chat("session-1", "test", tool_context=context))

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
        runner._lm_client = _FakeLMClient(tool_rounds=100)
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
