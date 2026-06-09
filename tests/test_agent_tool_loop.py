import json
import unittest

from agent import AgentRunner, _LMResponse, _ToolCall


class _FakeStore:
    def __init__(self) -> None:
        self.messages = []

    def session_exists(self, _session_id):
        return True

    def create_session(self):
        return "session-1"

    def touch_session(self, _session_id, title=None):
        return None

    def add_message(self, session_id, **message):
        self.messages.append({"session_id": session_id, **message})

    def load_history(self, _session_id):
        return [{"role": "user", "content": "test"}]


class _FakeLMClient:
    def __init__(self, tool_rounds: int) -> None:
        self.remaining = tool_rounds

    def call_with_tools(self, _messages):
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


class AgentToolLoopTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
