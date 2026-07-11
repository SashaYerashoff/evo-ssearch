"""Opt-in, read-only acceptance of EVA's channel/time attention conservation."""

from __future__ import annotations

import os
import unittest

from tests.integration.eva_client import EvaSession
from tests.integration.real_data_manifest import build_frozen_manifest


_BASE = os.getenv("EVA_LIVE_BASE_URL", "").strip()
_USER = os.getenv("EVA_LIVE_USER", "").strip()
_PASSWORD = os.getenv("EVA_LIVE_PASSWORD", "")
_FROM_RAW = os.getenv("EVA_LIVE_FROM_TS", "").strip()
_TO_RAW = os.getenv("EVA_LIVE_TO_TS", "").strip()
_CHANNEL_IDS = [
    int(item.strip())
    for item in os.getenv("EVA_LIVE_CHANNEL_IDS", "112,118,120").split(",")
    if item.strip().isdigit() and int(item.strip()) > 0
]
_VERIFY_TLS = os.getenv("EVA_LIVE_VERIFY_TLS", "").strip().lower() in {
    "1", "true", "yes", "on",
}


def _ids(rows, key="channel_id"):
    return {
        int(row[key])
        for row in rows or []
        if isinstance(row, dict) and row.get(key) is not None
    }


@unittest.skipUnless(
    _BASE and _USER and _PASSWORD and _FROM_RAW and _TO_RAW and _CHANNEL_IDS,
    "set EVA_LIVE_BASE_URL/USER/PASSWORD/FROM_TS/TO_TS/CHANNEL_IDS",
)
class LiveAttentionContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.from_ts = float(_FROM_RAW)
        cls.to_ts = float(_TO_RAW)
        cls.session = EvaSession(_BASE, verify_tls=_VERIFY_TLS, timeout=900)
        cls.session.login(_USER, _PASSWORD)
        cls.manifest = build_frozen_manifest(
            cls.session,
            from_ts=cls.from_ts,
            to_ts=cls.to_ts,
            channel_ids=_CHANNEL_IDS,
        )

    def test_inventory_conserves_every_requested_channel(self):
        requested = set(_CHANNEL_IDS)
        query = (
            "Use list_video_summary_channels to inventory exactly channel IDs "
            f"{sorted(requested)} between from_ts={self.from_ts} and to_ts={self.to_ts}. "
            "Do not replace the absolute window. Return explicit active, inactive, error, "
            "unchecked, and deferred channel IDs."
        )
        transcript = self.session.ask(query)
        result = transcript.result_of("list_video_summary_channels")

        self.assertIsInstance(result, dict, transcript.text)
        self.assertEqual(transcript.errors_of("list_video_summary_channels"), [])
        self.assertEqual(float(result["time_window"]["from_ts"]), self.from_ts)
        self.assertEqual(float(result["time_window"]["to_ts"]), self.to_ts)
        accounted = (
            set(map(int, result.get("checked_channel_ids") or []))
            | set(map(int, result.get("inactive_channel_ids") or []))
            | set(map(int, result.get("deferred_channel_ids") or []))
            | set(map(int, result.get("unchecked_channel_ids") or []))
            | _ids(result.get("errors") or [])
            | _ids(result.get("candidate_channels") or [])
        )
        self.assertEqual(
            accounted,
            requested,
            f"silent channel loss: requested={requested}, accounted={accounted}, result={result}",
        )
        self.assertEqual(
            int(result.get("total_channels_checked") or 0)
            + int(result.get("unchecked_count") or 0),
            len(requested),
        )

    def test_each_channel_preserves_frozen_window_and_evidence_bounds(self):
        for channel_id in _CHANNEL_IDS:
            with self.subTest(channel_id=channel_id):
                transcript = self.session.ask(
                    "Use get_video_summaries for numeric "
                    f"channel_id={channel_id}, depth=L0, from_ts={self.from_ts}, "
                    f"to_ts={self.to_ts}; include eight evidence frames. "
                    "Do not widen or replace this frozen window."
                )
                result = transcript.result_of("get_video_summaries")
                self.assertIsInstance(result, dict, transcript.text)
                self.assertEqual(int(result.get("channel_id")), channel_id)
                window = result.get("time_window") or {}
                self.assertEqual(float(window.get("from_ts")), self.from_ts)
                self.assertEqual(float(window.get("to_ts")), self.to_ts)
                self.assertIn((result.get("coverage") or {}).get("status"), {
                    "covered", "partial", "truncated", "no_data",
                })
                for frame in result.get("evidence_frames") or []:
                    timestamp_ms = int(frame.get("timestamp_ms") or 0)
                    self.assertGreaterEqual(timestamp_ms, int(self.from_ts * 1000))
                    self.assertLessEqual(timestamp_ms, int(self.to_ts * 1000))
                    self.assertEqual(int(frame.get("channel_id")), channel_id)

    def test_continue_uses_deferred_ids_without_repeating_first_chunk(self):
        inventory_ids = self.manifest.get("requested_channel_ids") or []
        if len(inventory_ids) <= 8:
            self.skipTest("continuation contract needs more than eight requested channels")
        first = self.session.ask(
            "Inventory all these channels with list_video_summary_channels inside the exact "
            f"window {self.from_ts}..{self.to_ts}: {inventory_ids}. Stop after the server chunk."
        )
        first_result = first.result_of("list_video_summary_channels") or {}
        deferred = set(map(int, first_result.get("deferred_channel_ids") or []))
        self.assertTrue(deferred, first_result)

        second = self.session.ask("Continue the same research with the remaining channels.", session_id=first.session_id)
        second_result = second.result_of("list_video_summary_channels") or {}
        second_requested = set(map(int, second_result.get("requested_channel_ids") or []))
        second_candidates = _ids(second_result.get("candidate_channels") or [])
        first_candidates = _ids(first_result.get("candidate_channels") or [])
        self.assertEqual(second_requested or deferred, deferred)
        self.assertTrue(second_candidates.issubset(deferred))
        self.assertFalse(second_candidates & first_candidates)
        self.assertEqual(
            float((second_result.get("time_window") or {})["from_ts"]),
            self.from_ts,
        )
        self.assertEqual(
            float((second_result.get("time_window") or {})["to_ts"]),
            self.to_ts,
        )


if __name__ == "__main__":
    unittest.main()
