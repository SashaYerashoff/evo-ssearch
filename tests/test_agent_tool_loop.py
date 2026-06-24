import json
import unittest

from agent import AgentRunner, _LMResponse, _ToolCall, _compact_tool_result_for_model
from agent_security import ToolExecutionContext


class _FakeStore:
    def __init__(self) -> None:
        self.messages = []

    def session_exists(self, _session_id, **_owner):
        return True

    def create_session(self, **_owner):
        return "session-1"

    def touch_session(self, _session_id, title=None, **_owner):
        return None

    def add_message(self, session_id, **message):
        self.messages.append({"session_id": session_id, **message})

    def load_history(self, _session_id, **_owner):
        return [{"role": "user", "content": "test"}]


class _FakeLMClient:
    def __init__(self, tool_rounds: int) -> None:
        self.remaining = tool_rounds
        self.tools = None

    def call_with_tools(self, _messages, tools=None):
        self.tools = tools
        if self.remaining > 0:
            round_number = self.remaining
            self.remaining -= 1
            return _LMResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    _ToolCall(
                        id=f"call-{round_number}",
                        name="list_channels",
                        args={},
                    )
                ],
            )
        return _LMResponse(content="done", finish_reason="stop", tool_calls=[])

    def stream_text(self, _messages):
        yield "done"


class _FakeTools:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, name, args, progress_cb=None):
        self.calls += 1
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


if __name__ == "__main__":
    unittest.main()
