import unittest
import threading
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from luxriot_connector import LuxriotCaptureSession, LuxriotManager


class CompactAttentionSignalTests(unittest.TestCase):
    def test_embedding_cadence_is_shared_by_all_sessions_of_a_channel(self):
        manager = object.__new__(LuxriotManager)
        manager._probe_embedding_cadence_lock = threading.Lock()
        manager._probe_embedding_cadence_slots = {}

        self.assertTrue(manager.claim_probe_embedding_slot(7, 10_100, 1_000))
        self.assertFalse(manager.claim_probe_embedding_slot(7, 10_900, 1_000))
        self.assertTrue(manager.claim_probe_embedding_slot(7, 11_000, 1_000))
        self.assertFalse(manager.claim_probe_embedding_slot(7, 10_500, 1_000))
        self.assertTrue(manager.claim_probe_embedding_slot(8, 10_900, 1_000))

        manager.release_probe_embedding_slot(7, 11_000, 1_000)
        self.assertTrue(manager.claim_probe_embedding_slot(7, 11_500, 1_000))

    def test_dense_motion_interval_is_a_first_class_model_signal(self):
        compact = LuxriotManager._compact_vector_signal(
            {
                "channel_id": 7,
                "motion_intervals": [
                    {
                        "started_at_ms": 1_000,
                        "ended_at_ms": 1_900,
                        "state": "quiet",
                        "sample_count": 4,
                        "motion_mean": 0.01,
                        "motion_max": 0.02,
                        "motion_p95": 0.02,
                        "motion_integral": 0.009,
                        "moving_fraction": 0.0,
                        "quiet_fraction": 1.0,
                        "activity_x_max": 0.4,
                    }
                ],
            }
        )

        self.assertEqual(compact["motion_intervals"][0]["sample_count"], 4)
        self.assertEqual(compact["motion_intervals"][0]["state"], "quiet")

    def test_per_frame_pnm_is_not_dropped_without_a_threshold_hit(self):
        compact = LuxriotManager._compact_vector_signal(
            {
                "channel_id": 7,
                "clip_frame_scores": [
                    {
                        "source_frame_index": 1,
                        "timestamp_ms": 1_500,
                        "snapshot_id": "snapshot-ref",
                        "probe_id": "probe-a",
                        "probe_version": "v1",
                        "p": 0.18,
                        "n": 0.12,
                        "m": 0.06,
                        "pos_floor": 0.32,
                        "margin_threshold": 0.08,
                        "threshold_state": "below_both",
                    }
                ],
            }
        )

        score = compact["clip_frame_scores"][0]
        self.assertEqual(score["threshold_state"], "below_both")
        self.assertAlmostEqual(score["p"], 0.18)
        self.assertAlmostEqual(score["n"], 0.12)
        self.assertAlmostEqual(score["m"], 0.06)

    def test_embedding_dispatch_obeys_configured_cadence(self):
        calls = []

        class ProbeManager:
            def add_frame(self, channel_id, _image, timestamp_ms, provenance=None):
                calls.append((channel_id, timestamp_ms, dict(provenance or {})))
                return {
                    "embedding_ref": f"probe-buffer:{channel_id}:{len(calls)}",
                }

        class Manager:
            probe_manager = ProbeManager()
            config = SimpleNamespace(
                LUXRIOT_ATTENTION_EMBEDDING_CADENCE_MS=1000,
            )

            @staticmethod
            def should_dispatch_probe_frame(_channel_id, *, capture_kind):
                return capture_kind == "video"

        session = object.__new__(LuxriotCaptureSession)
        session.manager = Manager()
        session.channel_id = 7
        session.capture_kind = "video"
        session.interval = 5.0
        session.lock = threading.Lock()
        session.capture_apex_probe_dispatch_count = 0
        session.capture_apex_probe_failure_count = 0
        session.capture_apex_probe_skipped_count = 0
        session._last_probe_embedding_timestamp_ms = None
        session._last_probe_embedding_slot = None
        session.probe_last_error = None
        session.capture_last_error = None
        session.summary_last_error = None
        session.last_error = None
        image = Image.new("RGB", (4, 4))

        first = session._add_selected_probe_frame(image, 10_000, {})
        skipped = session._add_selected_probe_frame(image, 10_999, {})
        second = session._add_selected_probe_frame(image, 11_000, {})

        self.assertIsNotNone(first)
        self.assertIsNone(skipped)
        self.assertIsNotNone(second)
        self.assertEqual([timestamp for _channel, timestamp, _meta in calls], [10_000, 11_000])
        self.assertEqual(session.capture_apex_probe_skipped_count, 1)


class L0BatchDeliveryContractTests(unittest.TestCase):
    @staticmethod
    def _manager(directory: Path, **overrides):
        config = SimpleNamespace(
            LUXRIOT_SNAPSHOT_INTERVAL=1,
            LUXRIOT_SNAPSHOT_MAX_EDGE=800,
            LUXRIOT_CAPTURE_SOURCE="snapshot",
            LUXRIOT_MAX_BUFFER_FRAMES=180,
            LUXRIOT_SUMMARY_MAX_BATCH_FRAMES=16,
            LUXRIOT_SUMMARY_MAX_WINDOW_SEC=60.0,
            LUXRIOT_SUMMARY_QUIET_CADENCE_SEC=5.0,
            LUXRIOT_SUMMARY_NORMAL_CADENCE_SEC=2.0,
            LUXRIOT_SUMMARY_BURST_CADENCE_SEC=1.0,
            LUXRIOT_ATTENTION_SCHEDULER_ENABLED=True,
            LUXRIOT_ATTENTION_EPISODE_DISPATCH_ENABLED=False,
            LUXRIOT_ATTENTION_EMBED_ALL_CHANNELS=True,
            LUXRIOT_SUMMARY_STATE_FILE=str(directory / "summaries.json"),
            LUXRIOT_ROLLUP_CACHE_FILE=str(directory / "rollups.json"),
            LUXRIOT_SYSTEM_PROMPT_DEFAULT="Describe.",
            LUXRIOT_ALERTS_JSON_PROMPT="",
            LUXRIOT_SUMMARY_HISTORY_LIMIT=20,
            LUXRIOT_SUMMARY_RETENTION_DAYS=0,
            LUXRIOT_AUTO_BOOKMARKS=False,
            LUXRIOT_ROLLUP_L1_LLM_ENABLED=False,
            LUXRIOT_ROLLUP_LLM_LEVELS="",
            LUXRIOT_BASE_URL="http://camera.invalid",
            LUXRIOT_USERNAME="",
            LUXRIOT_PASSWORD="",
        )
        for key, value in overrides.items():
            setattr(config, key, value)
        return LuxriotManager(
            config=config,
            lm_callback=lambda _messages, _model: "summary",
            message_builder=lambda *_args, **_kwargs: [],
            jpeg_encoder=lambda *_args, **_kwargs: "jpeg",
        )

    @staticmethod
    def _frame(timestamp_sec: float, mode: str):
        return {
            "captured_at": float(timestamp_sec),
            "time_sec": float(timestamp_sec),
            "thumbnail": "jpeg",
            "capture_selection": {"selection_mode": mode},
        }

    def test_batch_is_capped_at_16_and_dispatches_when_full_with_attention_enabled(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = self._manager(Path(temp))
            session = LuxriotCaptureSession(
                manager,
                channel_id=7,
                batch_size=32,
                prompt="Describe.",
                run_id="run-7",
            )
            batches = []
            session._dispatch_summary_frames = (
                lambda frames, **_kwargs: batches.append(list(frames)) is None or True
            )

            for offset in range(16):
                with session.lock:
                    self.assertTrue(
                        session._admit_summary_frame_locked(
                            self._frame(100.0 + offset, "burst")
                        )
                    )
                session._summarize_if_ready()

            self.assertEqual(session.batch_size, 16)
            self.assertEqual([len(batch) for batch in batches], [16])
            self.assertEqual(session.status()["pending_frames"], 0)

    def test_quiet_frames_are_sampled_every_ten_seconds_and_flush_at_target(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = self._manager(Path(temp))
            session = LuxriotCaptureSession(
                manager,
                channel_id=7,
                batch_size=16,
                prompt="Describe.",
                run_id="run-7",
            )
            batches = []
            session._dispatch_summary_frames = (
                lambda frames, **_kwargs: batches.append(list(frames)) is None or True
            )

            for offset in range(71):
                with session.lock:
                    session._admit_summary_frame_locked(
                        self._frame(100.0 + offset, "quiet")
                    )
                session._summarize_if_ready()

            self.assertEqual([len(batch) for batch in batches], [8])
            self.assertEqual(
                [frame["captured_at"] for frame in batches[0]],
                [100.0 + offset for offset in range(0, 71, 10)],
            )

    def test_attention_telemetry_does_not_start_sparse_vlm_dispatch_by_default(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = self._manager(Path(temp))

            decision = manager.observe_attention_cv(
                channel_id=7,
                timestamp_ms=1_000,
                motion_score=0.0,
                activity_x=0.0,
                mode="quiet",
            )

            self.assertIsNotNone(decision)
            status = manager.attention_status()
            self.assertTrue(status["enabled"])
            self.assertFalse(status["dispatch_enabled"])
            self.assertFalse(status["scheduler_alive"])

    def test_quiet_deadline_excludes_a_frame_beyond_the_120_second_window(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = self._manager(Path(temp))
            session = LuxriotCaptureSession(
                manager,
                channel_id=7,
                batch_size=16,
                prompt="Describe.",
                run_id="run-7",
            )
            batches = []
            session._dispatch_summary_frames = (
                lambda frames, **_kwargs: batches.append(list(frames)) is None or True
            )
            with session.lock:
                session.frames = [
                    self._frame(100.0, "quiet"),
                    self._frame(215.0, "quiet"),
                    self._frame(221.0, "quiet"),
                ]
                session._summary_batch_opened_monotonic = 0.0

            session._summarize_if_ready()

            self.assertEqual(
                [[frame["captured_at"] for frame in batch] for batch in batches],
                [[100.0, 215.0]],
            )
            self.assertEqual(
                [frame["captured_at"] for frame in session.frames],
                [221.0],
            )

    def test_attention_mode_retains_frozen_frames_for_quiet_heartbeat(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = self._manager(
                Path(temp),
                LUXRIOT_FROZEN_FRAME_MAX_SEC=10.0,
                LUXRIOT_FROZEN_FRAME_MIN_COUNT=3,
            )
            session = LuxriotCaptureSession(
                manager,
                channel_id=7,
                batch_size=16,
                prompt="Describe.",
                run_id="run-7",
            )
            session._add_selected_probe_frame = (
                lambda _image, timestamp_ms, _provenance: {
                    "embedding_ref": f"embedding:{timestamp_ms}",
                }
            )
            image = Image.new("RGB", (12, 8), color=(64, 64, 64))

            for observed_at, source_ms in (
                (100.0, 100_000),
                (105.0, 105_000),
                (112.0, 112_000),
            ):
                with patch("luxriot_connector.time.time", return_value=observed_at):
                    session._accept_captured_frame(
                        image,
                        source_ms,
                        summarize=True,
                    )

            status = session.status()
            self.assertTrue(status["frozen_signal"])
            self.assertEqual(status["frozen_frame_retained_count"], 1)
            self.assertEqual(status["frozen_frame_dropped_count"], 0)
            self.assertEqual(len(session.recent_frame_items()), 3)

    def test_local_mjpeg_budget_is_time_bounded_instead_of_network_capped(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = self._manager(
                Path(temp),
                LUXRIOT_LIVE_SEGMENT_SECONDS=60.0,
                LUXRIOT_LIVE_SEGMENT_FPS=4.0,
                LUXRIOT_LIVE_SEGMENT_MB=8.0,
            )
            manager.is_local_channel = lambda channel_id: int(channel_id) == 900001
            session = LuxriotCaptureSession(
                manager,
                channel_id=900001,
                batch_size=12,
                prompt="Describe.",
                run_id="run-usb",
            )

            budget = session._live_segment_capture_budget()

            self.assertEqual(budget["stream_seconds"], 60.0)
            self.assertEqual(
                budget["byte_budget"],
                60 * 32 * 1024 * 1024,
            )
            self.assertGreater(budget["byte_budget"], 1024 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
