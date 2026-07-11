import queue
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.live_attention_soak import SoakResult, _record_transcript
from tests.integration.eva_client import Transcript


class LiveAttentionSoakUnitTests(unittest.TestCase):
    def test_cli_help_runs_outside_repository_working_directory(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "live_attention_soak.py"
        with tempfile.TemporaryDirectory() as working_directory:
            completed = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=working_directory,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--duration", completed.stdout)

    def test_transcript_records_channel_and_window_drift(self):
        transcript = Transcript(
            events=[
                {
                    "type": "tool_call",
                    "name": "get_video_summaries",
                    "args": {"channel_id": 120, "from_ts": 90.0, "to_ts": 200.0},
                },
                {"type": "done", "session_id": "session-1"},
            ]
        )
        events = queue.Queue()

        _record_transcript(
            transcript,
            expected_channel=118,
            from_ts=100.0,
            to_ts=200.0,
            events=events,
        )

        event = events.get_nowait()
        self.assertEqual(event["kind"], "turn")
        self.assertEqual(event["session_id"], "session-1")
        self.assertEqual(len(event["invariants"]), 2)
        self.assertIn("drifted to channel 120", event["invariants"][0])
        self.assertIn("changed from_ts", event["invariants"][1])

    def test_report_ok_requires_no_silent_errors(self):
        result = SoakResult(started_at=10.0, finished_at=20.0, seed=83, worker_count=1)
        self.assertTrue(result.as_dict()["ok"])

        result.invariant_errors.append("desired set changed")
        self.assertFalse(result.as_dict()["ok"])


if __name__ == "__main__":
    unittest.main()
