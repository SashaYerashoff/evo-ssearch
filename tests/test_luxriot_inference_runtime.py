import ast
import base64
import json
import os
import stat
import tempfile
import threading
import time
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from unittest.mock import patch
from uuid import uuid4

import numpy as np
from PIL import Image

from inference_queue import (
    InMemoryInferenceQueueRepository,
    LuxriotInferenceQueueRuntime,
)
from luxriot_connector import LuxriotCaptureSession, LuxriotManager


def build_manager(
    directory: Path,
    lm_callback=None,
    alert_parser=None,
    summary_archive_callback=None,
    runtime_state_store=None,
    config_overrides=None,
) -> LuxriotManager:
    config = SimpleNamespace(
        LUXRIOT_SYSTEM_PROMPT_DEFAULT="Describe the stream.",
        LUXRIOT_ALERTS_JSON_PROMPT="",
        LUXRIOT_SUMMARY_HISTORY_LIMIT=100,
        LUXRIOT_SUMMARY_RETENTION_DAYS=0,
        LUXRIOT_AUTO_BOOKMARKS=False,
        LUXRIOT_BOOKMARK_COOLDOWN_SEC=60.0,
        LUXRIOT_ALERTS_MAX_PER_BATCH=8,
        LUXRIOT_SUMMARY_STATE_FILE=str(directory / "summaries.json"),
        LUXRIOT_ROLLUP_CACHE_FILE=str(directory / "rollups.json"),
        LUXRIOT_ROLLUP_L1_LLM_ENABLED=False,
        LUXRIOT_ROLLUP_LLM_LEVELS="",
        LUXRIOT_ROLLUP_MIN_SOURCE_TOKENS=8000,
        LUXRIOT_ROLLUP_LLM_CHAR_BUDGET=12000,
        LUXRIOT_ROLLUP_LLM_MAX_NEW_PER_CALL=1,
        LUXRIOT_ROLLUP_SUMMARY_CACHE_LIMIT=100,
        LUXRIOT_ROLLUP_TIME_ONLY=True,
        LUXRIOT_SNAPSHOT_INTERVAL=5,
        LUXRIOT_SNAPSHOT_MAX_EDGE=800,
        LUXRIOT_CAPTURE_SOURCE="auto",
        LUXRIOT_LIVE_SEGMENT_SECONDS=2.0,
        LUXRIOT_LIVE_SEGMENT_MB=1.0,
        LUXRIOT_LIVE_SEGMENT_EVERY_N=1,
        LUXRIOT_MAX_BUFFER_FRAMES=180,
        LUXRIOT_SUMMARY_ARCHIVE_FRAMES_PER_BATCH=4,
        LUXRIOT_BASE_URL="http://luxriot.invalid",
        LUXRIOT_USERNAME="",
        LUXRIOT_PASSWORD="",
    )
    if config_overrides:
        for key, value in dict(config_overrides).items():
            setattr(config, key, value)

    def message_builder(_channel, frames, prompt, system_prompt):
        return [
            {
                "frame_count": len(frames),
                "prompt": prompt,
                "system_prompt": system_prompt,
            }
        ]

    return LuxriotManager(
        config=config,
        lm_callback=lm_callback or (lambda _messages, _model: "summary"),
        message_builder=message_builder,
        jpeg_encoder=lambda _image, **_kwargs: "jpeg",
        alert_parser=alert_parser,
        summary_archive_callback=summary_archive_callback,
        runtime_state_store=runtime_state_store,
    )


class MemoryRuntimeStateStore:
    def __init__(self):
        self.payloads = {}

    def load_state(self, key):
        return self.payloads.get(key)

    def save_state(self, key, payload):
        self.payloads[key] = payload


def sample_frames(start: float = 100.0):
    return [
        {
            "thumbnail": "base64-frame-one",
            "captured_at": start,
            "time_sec": start,
            "width": 1280,
            "height": 720,
        },
        {
            "thumbnail": "base64-frame-two",
            "captured_at": start + 5.0,
            "time_sec": start + 5.0,
            "width": 1280,
            "height": 720,
        },
    ]


def _jpeg_b64(frame: np.ndarray) -> str:
    image = Image.fromarray(frame.astype(np.uint8), "RGB")
    out = BytesIO()
    image.save(out, format="JPEG", quality=85)
    return base64.b64encode(out.getvalue()).decode("ascii")


def _road_frame(x: int, *, width: int = 120, height: int = 80) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[46:58, x : x + 20, :] = 255
    return frame


def road_sample_frames(xs: Sequence[int], start: float = 100.0, step: float = 1.0):
    return [
        {
            "thumbnail": _jpeg_b64(_road_frame(x)),
            "captured_at": start + idx * step,
            "time_sec": start + idx * step,
            "width": 120,
            "height": 80,
        }
        for idx, x in enumerate(xs)
    ]


class FakeVectorProbeManager:
    def add_frame(self, channel_id, pil_image, timestamp_ms):
        return None

    def query(
        self,
        channel_id,
        positives,
        negatives,
        pos_floor,
        margin_thr,
        top_k,
        window_sec=None,
        image_probe=None,
        roi_norm=None,
        roi_padding=0.05,
    ):
        return {
            "results": [
                {
                    "timestamp_ms": 105000,
                    "channel_id": int(channel_id),
                    "pos_score": 0.42,
                    "neg_score": 0.18,
                    "margin": 0.24,
                    "thumbnail": "should-not-enter-vector-signal",
                }
            ],
            "frames_indexed": 12,
            "status": {"frames": 12},
        }


class FakeVectorProbeStore:
    def list_probes(self):
        return [
            {
                "id": "probe-drift",
                "name": "vehicle drift candidate",
                "channel_id": 7,
                "enabled": True,
                "positives": ["vehicle drifting or burnout"],
                "negatives": ["normal traffic flow"],
                "pos_floor": 0.25,
                "margin": 0.05,
                "severity": "high",
            }
        ]


def load_lm_alert_parser():
    source = Path(__file__).resolve().parent.parent.joinpath("oldapp.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    namespace = {
        "time": time,
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Sequence": Sequence,
        "Set": Set,
        "Tuple": Tuple,
    }
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_parse_lm_alerts":
            exec(compile(ast.Module([node], []), "oldapp.py", "exec"), namespace)
            return namespace["_parse_lm_alerts"]
    raise AssertionError("_parse_lm_alerts not found")


class LuxriotCaptureDispatchTests(unittest.TestCase):
    def test_summary_flush_preserves_recent_preview_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.set_summary_dispatcher(lambda _batch, _workload: {"queued": False})
            session = LuxriotCaptureSession(
                manager,
                channel_id=7,
                batch_size=2,
                prompt="Describe activity.",
                run_id="run-7",
            )
            image = SimpleNamespace(width=1280, height=720)

            session._accept_captured_frame(image, 1_000, summarize=True)
            session._accept_captured_frame(image, 2_000, summarize=True)

            self.assertEqual(len(session.frames), 0)
            self.assertEqual(len(session.recent_frame_items()), 2)
            self.assertEqual(session.nearest_frame_thumbnail(), "jpeg")
            status = session.status()
            self.assertEqual(status["pending_frames"], 0)
            self.assertEqual(status["recent_frame_count"], 2)

    def test_repeated_exact_frames_mark_source_frozen_and_stop_buffering(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                config_overrides={
                    "LUXRIOT_FROZEN_FRAME_MAX_SEC": 10.0,
                    "LUXRIOT_FROZEN_FRAME_MIN_COUNT": 3,
                },
            )
            session = LuxriotCaptureSession(
                manager,
                channel_id=7,
                batch_size=12,
                prompt="Describe activity.",
                run_id="run-7",
            )
            image = SimpleNamespace(width=1280, height=720)

            with patch("luxriot_connector.time.time", return_value=100.0):
                session._accept_captured_frame(image, 100_000, summarize=True)
            with patch("luxriot_connector.time.time", return_value=105.0):
                session._accept_captured_frame(image, 105_000, summarize=True)
            with patch("luxriot_connector.time.time", return_value=112.0):
                session._accept_captured_frame(image, 112_000, summarize=True)

            with patch("luxriot_connector.time.time", return_value=113.0):
                status = session.status()
            self.assertTrue(status["frozen_signal"])
            self.assertEqual(status["frozen_frame_count"], 3)
            self.assertEqual(status["frozen_frame_dropped_count"], 1)
            self.assertEqual(len(session.recent_frame_items()), 2)

            manager.jpeg_encoder = lambda _image, **_kwargs: "jpeg-new"
            with patch("luxriot_connector.time.time", return_value=114.0):
                session._accept_captured_frame(image, 114_000, summarize=True)
                status = session.status()
            self.assertFalse(status["frozen_signal"])
            self.assertEqual(len(session.recent_frame_items()), 3)

    def test_dispatch_failure_restores_detached_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))

            def unavailable(_batch, _workload):
                raise RuntimeError("database unavailable")

            manager.set_summary_dispatcher(unavailable)
            session = LuxriotCaptureSession(
                manager,
                channel_id=7,
                batch_size=2,
                prompt="Describe activity.",
                run_id="run-7",
            )
            session.frames = sample_frames()

            session._summarize_batch()

            self.assertEqual(len(session.frames), 2)
            self.assertIn("database unavailable", session.last_error)
            self.assertEqual(session.queue_dropped_batches, 1)

    def test_capture_loop_keeps_summary_failure_visible(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))

            def unavailable(_batch, _workload):
                raise RuntimeError("vlm unavailable")

            manager.set_summary_dispatcher(unavailable)
            session = LuxriotCaptureSession(
                manager,
                channel_id=7,
                batch_size=1,
                prompt="Describe activity.",
                run_id="run-7",
            )
            session.client = SimpleNamespace(
                get_snapshot=lambda _channel_id, **_kwargs: SimpleNamespace(width=1280, height=720)
            )

            def stop_after_one_wait(_interval):
                session.stop_event.set()
                return False

            session.stop_event.wait = stop_after_one_wait
            session._run()

            self.assertEqual(len(session.frames), 1)
            self.assertIn("vlm unavailable", session.last_error)
            self.assertEqual(session.queue_dropped_batches, 1)

    def test_async_summary_worker_does_not_block_recent_frame_updates(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.jpeg_encoder = lambda image, **_kwargs: f"jpeg-{getattr(image, 'marker', 'x')}"
            started = threading.Event()
            release = threading.Event()

            def slow_dispatcher(_batch, _workload):
                started.set()
                release.wait(timeout=2.0)
                return {"queued": False, "accepted": True}

            manager.set_summary_dispatcher(slow_dispatcher)
            session = LuxriotCaptureSession(
                manager,
                channel_id=7,
                batch_size=2,
                prompt="Describe activity.",
                run_id="run-7",
            )
            session.summary_worker_thread.start()
            try:
                session._accept_captured_frame(SimpleNamespace(width=1280, height=720, marker=1), 1_000, summarize=True)
                session._accept_captured_frame(SimpleNamespace(width=1280, height=720, marker=2), 2_000, summarize=True)
                self.assertTrue(started.wait(timeout=1.0))

                session._accept_captured_frame(SimpleNamespace(width=1280, height=720, marker=3), 3_000, summarize=True)

                self.assertEqual(len(session.recent_frame_items()), 3)
                self.assertEqual(session.nearest_frame_thumbnail(), "jpeg-3")
                status = session.status()
                self.assertTrue(status["summary_inflight"])
                self.assertEqual(status["pending_frames"], 1)
            finally:
                release.set()
                session.stop()

    def test_async_summary_queue_is_bounded_and_latest_wins(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                config_overrides={"LUXRIOT_SUMMARY_QUEUE_MAX_BATCHES": 1},
            )
            manager.jpeg_encoder = lambda image, **_kwargs: f"jpeg-{getattr(image, 'marker', 'x')}"
            started = threading.Event()
            release = threading.Event()

            def slow_dispatcher(_batch, _workload):
                started.set()
                release.wait(timeout=2.0)
                return {"queued": False, "accepted": True}

            manager.set_summary_dispatcher(slow_dispatcher)
            session = LuxriotCaptureSession(
                manager,
                channel_id=7,
                batch_size=1,
                prompt="Describe activity.",
                run_id="run-7",
            )
            session.summary_worker_thread.start()
            try:
                session._accept_captured_frame(SimpleNamespace(width=1280, height=720, marker=1), 1_000, summarize=True)
                self.assertTrue(started.wait(timeout=1.0))
                session._accept_captured_frame(SimpleNamespace(width=1280, height=720, marker=2), 2_000, summarize=True)
                session._accept_captured_frame(SimpleNamespace(width=1280, height=720, marker=3), 3_000, summarize=True)

                status = session.status()
                self.assertEqual(status["summary_queue_depth"], 1)
                self.assertEqual(status["summary_queue_frame_count"], 1)
                self.assertEqual(session.queue_dropped_batches, 1)
                self.assertEqual(session.nearest_frame_thumbnail(), "jpeg-3")
            finally:
                release.set()
                session.stop()

    def test_async_summary_worker_completion_records_history(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp), lm_callback=lambda _messages, _model: "async summary")
            manager.jpeg_encoder = lambda image, **_kwargs: f"jpeg-{getattr(image, 'marker', 'x')}"
            session = LuxriotCaptureSession(
                manager,
                channel_id=7,
                batch_size=2,
                prompt="Describe activity.",
                run_id="run-7",
            )
            manager.sessions[7] = session
            session.summary_worker_thread.start()
            try:
                session._accept_captured_frame(SimpleNamespace(width=1280, height=720, marker=1), 1_000, summarize=True)
                session._accept_captured_frame(SimpleNamespace(width=1280, height=720, marker=2), 2_000, summarize=True)
                for _ in range(40):
                    if session.status()["logs"]:
                        break
                    time.sleep(0.025)
                self.assertEqual(session.status()["logs"][0]["summary"], "async summary")
                self.assertEqual(manager.summary_history[7][0]["summary"], "async summary")
            finally:
                session.stop()

    def test_capture_loop_uses_live_segment_after_slow_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                config_overrides={
                    "LUXRIOT_CAPTURE_SOURCE": "auto",
                    "LUXRIOT_LIVE_SEGMENT_SECONDS": 2.0,
                    "LUXRIOT_LIVE_SEGMENT_MB": 1.0,
                    "LUXRIOT_LIVE_SEGMENT_EVERY_N": 1,
                },
            )
            session = LuxriotCaptureSession(
                manager,
                channel_id=7,
                batch_size=4,
                prompt="Describe activity.",
                run_id="run-7",
            )
            session.slow_snapshot_count = 1
            decoded_frames = [
                SimpleNamespace(
                    timestamp_ms=1_000,
                    image=np.zeros((4, 6, 3), dtype=np.uint8),
                ),
                SimpleNamespace(
                    timestamp_ms=2_000,
                    image=np.full((4, 6, 3), 128, dtype=np.uint8),
                ),
            ]

            with (
                patch.object(session, "_run_ffmpeg_live_segment_once", return_value=None),
                patch("luxriot_connector.iter_luxriot_live_segment_frames", return_value=iter(decoded_frames)) as live_iter,
            ):
                handled = session._run_live_segment_once()

            self.assertTrue(handled)
            live_iter.assert_called_once()
            self.assertEqual(len(session.frames), 2)
            self.assertEqual(session.active_capture_source, "live_segment")
            self.assertEqual(session.live_segment_count, 1)
            self.assertEqual(session.live_segment_frame_count, 2)
            self.assertEqual(session.last_live_segment_frames, 2)
            self.assertIsNone(session.last_live_segment_error)

    def test_auto_capture_falls_back_to_live_segment_after_snapshot_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                config_overrides={
                    "LUXRIOT_CAPTURE_SOURCE": "auto",
                    "LUXRIOT_CAPTURE_REQUEST_TIMEOUT_SEC": 1.0,
                },
            )
            session = LuxriotCaptureSession(
                manager,
                channel_id=120,
                batch_size=4,
                prompt="Describe activity.",
                run_id="run-120",
            )
            session.client = SimpleNamespace(
                get_snapshot=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("snapshot 404"))
            )

            def stop_after_cycle(_timeout: float) -> bool:
                session.stop_event.set()
                return True

            with (
                patch.object(session, "_run_live_segment_once", return_value=True) as live_segment,
                patch.object(session.stop_event, "wait", side_effect=stop_after_cycle),
            ):
                session._run()

            live_segment.assert_called_once()
            self.assertEqual(session.snapshot_failed_count, 1)
            self.assertIsNone(session.last_error)
            self.assertTrue(session._should_use_live_segment())

    def test_auto_capture_does_not_retry_unavailable_snapshot_after_live_segment_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                config_overrides={
                    "LUXRIOT_CAPTURE_SOURCE": "auto",
                    "LUXRIOT_CAPTURE_REQUEST_TIMEOUT_SEC": 1.0,
                },
            )
            session = LuxriotCaptureSession(
                manager,
                channel_id=120,
                batch_size=4,
                prompt="Describe activity.",
                run_id="run-120",
            )
            snapshot_calls = 0

            def fail_snapshot(*_args, **_kwargs):
                nonlocal snapshot_calls
                snapshot_calls += 1
                raise RuntimeError("snapshot 404")

            session.client = SimpleNamespace(get_snapshot=fail_snapshot)
            session.snapshot_failed_count = 1

            def stop_after_cycle(_timeout: float) -> bool:
                session.stop_event.set()
                return True

            with (
                patch.object(session, "_run_live_segment_once", return_value=False) as live_segment,
                patch.object(session.stop_event, "wait", side_effect=stop_after_cycle),
            ):
                session._run()

            live_segment.assert_called_once()
            self.assertEqual(snapshot_calls, 0)
            self.assertEqual(session.snapshot_failed_count, 1)

    def test_live_segment_failure_backoff_skips_immediate_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp), config_overrides={"LUXRIOT_CAPTURE_SOURCE": "auto"})
            session = LuxriotCaptureSession(
                manager,
                channel_id=120,
                batch_size=4,
                prompt="Describe activity.",
                run_id="run-120",
            )
            session._set_live_segment_backoff(failed=True)
            with patch.object(session, "_run_ffmpeg_live_segment_once", return_value=True) as ffmpeg_live:
                self.assertFalse(session._run_live_segment_once())
            ffmpeg_live.assert_not_called()

            session._set_live_segment_backoff(failed=False)
            with patch.object(session, "_run_ffmpeg_live_segment_once", return_value=True) as ffmpeg_live:
                self.assertTrue(session._run_live_segment_once())
            ffmpeg_live.assert_called_once()

    def test_sync_fallback_records_summary_without_dispatcher(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            batch = manager.create_summary_batch(
                channel_id=7,
                run_id="run-7",
                batch_size=2,
                prompt="Describe activity.",
                model_hint="model-a",
                interval_sec=5.0,
                frames=sample_frames(),
            )

            outcome = manager.dispatch_summary_batch(batch)

            self.assertFalse(outcome["queued"])
            self.assertEqual(
                manager.summary_history[7][0]["summary"],
                "summary",
            )

    def test_summary_batch_records_llm_input_stats_and_warnings(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                config_overrides={
                    "LM_VIDEO_INPUT_WARNING_CHARS": 120,
                    "LM_VIDEO_IMAGE_PAYLOAD_WARNING_CHARS": 80,
                },
            )
            manager.message_builder = lambda _channel, frames, prompt, system_prompt: [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *[
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{frame['thumbnail']}",
                                    "detail": "high",
                                },
                            }
                            for frame in frames
                        ],
                    ],
                },
            ]
            frames = [
                {
                    "thumbnail": "a" * 90,
                    "captured_at": 100.0,
                    "time_sec": 100.0,
                    "width": 1280,
                    "height": 720,
                },
                {
                    "thumbnail": "b" * 70,
                    "captured_at": 105.0,
                    "time_sec": 105.0,
                    "width": 1280,
                    "height": 720,
                },
            ]
            batch = manager.create_summary_batch(
                channel_id=7,
                run_id="run-7",
                batch_size=2,
                prompt="Describe activity with enough text to trip the warning.",
                model_hint="model-a",
                interval_sec=5.0,
                frames=frames,
            )

            entry = manager.run_summary_batch(batch)
            accepted = manager.accept_summary_entry(entry)

            stats = accepted["llm_input_stats"]
            self.assertEqual(stats["frame_count"], 2)
            self.assertEqual(stats["image_parts"], 2)
            self.assertEqual(stats["high_detail_images"], 2)
            self.assertGreaterEqual(stats["total_image_base64_chars"], 160)
            self.assertIn("warnings", stats)
            self.assertIn("llm_input_stats", manager.summary_history[7][0])

    def test_summary_batch_injects_vector_signal_bundle_into_l0_prompt_and_status(self):
        captured_messages = []

        def lm_callback(messages, _model):
            captured_messages.extend(messages)
            return (
                "Current observed state:\n"
                "Vehicle drift candidate: uncertain; vector cue should be visually checked.\n"
                "ALERTS_JSON:{\"alerts\":[]}"
            )

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp), lm_callback=lm_callback)
            manager.probe_manager = FakeVectorProbeManager()
            manager.probes_store = FakeVectorProbeStore()

            batch = manager.create_summary_batch(
                channel_id=7,
                run_id="run-7",
                batch_size=2,
                prompt="Describe activity.",
                model_hint="model-a",
                interval_sec=5.0,
                frames=sample_frames(),
            )

            vector_signal = batch["vector_signal"]
            json.dumps(vector_signal, allow_nan=False)
            self.assertEqual(vector_signal["clip_probe_signals"][0]["name"], "vehicle drift candidate")
            self.assertEqual(vector_signal["clip_probe_signals"][0]["apex_frame"], 2)
            self.assertEqual(vector_signal["clip_probe_signals"][0]["p"], 0.42)
            self.assertEqual(vector_signal["road_episodes"][0]["event_type"], "drift_burnout_candidate")
            self.assertIn("clip_probe", vector_signal["road_episodes"][0]["sources"])
            self.assertNotIn("thumbnail", json.dumps(vector_signal))

            entry = manager.run_summary_batch(batch)
            accepted = manager.accept_summary_entry(entry)

            prompt_text = json.dumps(captured_messages)
            self.assertIn("VECTOR_SIGNALS_JSON", prompt_text)
            self.assertIn("vector_homeostasis_attention_signal_not_visual_proof", prompt_text)
            self.assertIn("not visual proof", prompt_text)
            self.assertNotIn("should-not-enter-vector-signal", prompt_text)
            self.assertIn("vector_signal", accepted)
            self.assertEqual(manager.summary_history[7][0]["vector_signal"]["clip_probe_signals"][0]["m"], 0.24)
            l0 = manager.summary_rollups(channel_id=7, synthesize=False)["levels"]["L0"][0]
            self.assertEqual(l0["vector_signal"]["clip_probe_signals"][0]["probe_id"], "probe-drift")
            digest = manager.system_status_digest(channel_ids=[7])["channels"][0]
            self.assertEqual(digest["vector_signal_total"], 2)
            self.assertEqual(digest["recent_vector_signals"][0]["top_clip_probe"]["name"], "vehicle drift candidate")
            self.assertEqual(digest["recent_vector_signals"][0]["top_road_episode"]["event_type"], "drift_burnout_candidate")

    def test_road_cv_directional_cues_wait_for_frozen_high_confidence_scene(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                config_overrides={
                    "LUXRIOT_ROAD_SCENE_CALIBRATION_SAMPLES": 4,
                    "LUXRIOT_VECTOR_SIGNAL_PROBE_LIMIT": 0,
                },
            )

            forward = [12, 20, 28, 36, 44, 52, 60, 68]
            first = manager._build_vector_signal_bundle(
                7,
                road_sample_frames(forward, start=100.0),
                batch_start_ms=100000,
                batch_end_ms=107000,
            )

            self.assertEqual(first["road_cv_scene"]["status"], "uncalibrated")
            self.assertFalse(first["road_cv_scene"]["directional_enabled"])
            self.assertNotIn(
                "opposing_flow_candidate",
                {cue["cue_type"] for cue in first.get("road_cv_cues", [])},
            )

            for offset in (120.0, 140.0, 160.0):
                manager._build_vector_signal_bundle(
                    7,
                    road_sample_frames(forward, start=offset),
                    batch_start_ms=int(offset * 1000),
                    batch_end_ms=int((offset + 7) * 1000),
                )

            reverse = [68, 60, 52, 44, 36, 28, 20, 12]
            checked = manager._build_vector_signal_bundle(
                7,
                road_sample_frames(reverse, start=180.0),
                batch_start_ms=180000,
                batch_end_ms=187000,
            )

            self.assertEqual(checked["road_cv_scene"]["status"], "calibrated")
            self.assertTrue(checked["road_cv_scene"]["directional_enabled"])
            self.assertIn(
                "opposing_flow_candidate",
                {cue["cue_type"] for cue in checked.get("road_cv_cues", [])},
            )
            self.assertEqual(checked["health"]["road_cv_scene_status"], "calibrated")

    def test_low_fps_suppresses_road_cv_motion_but_keeps_clip_vector_signal(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.probe_manager = FakeVectorProbeManager()
            manager.probes_store = FakeVectorProbeStore()

            vector_signal = manager._build_vector_signal_bundle(
                7,
                road_sample_frames([12, 44, 76, 44], start=100.0, step=6.0),
                batch_start_ms=100000,
                batch_end_ms=118000,
            )

            self.assertEqual(vector_signal["clip_probe_signals"][0]["name"], "vehicle drift candidate")
            self.assertEqual(vector_signal["health"]["road_cv_low_fps_suppressed_frames"], 3)
            self.assertNotIn("road_cv_cues", vector_signal)
            self.assertEqual(vector_signal["road_episodes"][0]["event_type"], "drift_burnout_candidate")

    def test_sync_fallback_archives_batch_frame_anchors(self):
        archived = []

        def archive_callback(entry):
            archived.append(dict(entry))
            return {
                "attempted": len(entry.get("archive_frames") or []),
                "inserted": len(entry.get("archive_frames") or []),
                "summary_frames": len(entry.get("archive_frames") or []),
                "alert_frames": 0,
            }

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp), summary_archive_callback=archive_callback)
            batch = manager.create_summary_batch(
                channel_id=7,
                run_id="run-7",
                batch_size=2,
                prompt="Describe activity.",
                model_hint="model-a",
                interval_sec=5.0,
                frames=sample_frames(),
            )

            outcome = manager.dispatch_summary_batch(batch)

            self.assertFalse(outcome["queued"])
            self.assertEqual(len(archived), 1)
            frames = archived[0]["archive_frames"]
            self.assertEqual([frame["anchor_role"] for frame in frames], ["first", "last"])
            self.assertEqual([frame["timestamp_ms"] for frame in frames], [100000, 105000])
            self.assertNotIn("archive_frames", manager.summary_history[7][0])

    def test_sync_fallback_archives_period_spread_frame_anchors(self):
        archived = []

        def archive_callback(entry):
            archived.append(dict(entry))
            return {
                "attempted": len(entry.get("archive_frames") or []),
                "inserted": len(entry.get("archive_frames") or []),
                "summary_frames": len(entry.get("archive_frames") or []),
                "alert_frames": 0,
            }

        frames = [
            {
                "thumbnail": f"base64-frame-{index}",
                "captured_at": 100.0 + index,
                "time_sec": 100.0 + index,
                "width": 1280,
                "height": 720,
            }
            for index in range(5)
        ]

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp), summary_archive_callback=archive_callback)
            batch = manager.create_summary_batch(
                channel_id=7,
                run_id="run-7",
                batch_size=5,
                prompt="Describe activity.",
                model_hint="model-a",
                interval_sec=1.0,
                frames=frames,
            )

            outcome = manager.dispatch_summary_batch(batch)

            self.assertFalse(outcome["queued"])
            saved = archived[0]["archive_frames"]
            self.assertEqual([frame["frame_index"] for frame in saved], [0, 1, 3, 4])
            self.assertEqual([frame["anchor_role"] for frame in saved], ["first", "sample", "sample", "last"])
            self.assertEqual([frame["timestamp_ms"] for frame in saved], [100000, 101000, 103000, 104000])

    def test_summary_alert_counts_roll_up_by_severity(self):
        with tempfile.TemporaryDirectory() as temp:
            def parse_alerts(text, _channel_id, _default_ts_ms=None):
                if "critical-event" in text:
                    return [{"severity": "critical"}, {"severity": "normal"}]
                if "low-event" in text:
                    return [{"severity": "low"}]
                return []

            manager = build_manager(Path(temp), alert_parser=parse_alerts)
            now = time.time()
            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": "critical-event\nALERTS_JSON:\n{\"alerts\":[]}",
                    "frame_count": 2,
                    "created_at": now,
                },
            )
            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": "low-event\nALERTS_JSON:\n{\"alerts\":[]}",
                    "frame_count": 2,
                    "created_at": now + 1.0,
                },
            )

            logs = manager.session_status(7, run_selector="all")["logs"]
            self.assertEqual(logs[0]["alert_counts"], {"critical": 1, "normal": 1})
            self.assertEqual(logs[1]["alert_counts"], {"low": 1})

            rollups = manager.summary_rollups(7, run_selector="all", level_limit=10)
            self.assertEqual(rollups["levels"]["L1"][0]["alert_counts"], {"critical": 1, "normal": 1, "low": 1})
            self.assertEqual(rollups["levels"]["L1"][0]["alert_total"], 3)

    def test_summary_history_merge_normalizes_only_new_log(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            base = 1_781_700_000.0
            existing = []
            for index in range(5):
                normalized = manager._normalize_summary_log_entry(
                    {
                        "channel_id": 7,
                        "run_id": "run-7",
                        "summary": f"existing summary {index}",
                        "frame_count": 12,
                        "created_at": base + index,
                    }
                )
                self.assertIsNotNone(normalized)
                existing.append(normalized)
            manager.summary_history[7] = existing

            with patch.object(
                manager,
                "_normalize_summary_log_entry",
                wraps=manager._normalize_summary_log_entry,
            ) as normalize:
                manager.record_summary_log(
                    7,
                    {
                        "channel_id": 7,
                        "run_id": "run-7",
                        "summary": "new summary",
                        "frame_count": 12,
                        "created_at": base + 10.0,
                    },
                )

            self.assertEqual(normalize.call_count, 1)
            self.assertEqual(len(manager.summary_history[7]), 6)
            self.assertEqual(manager.summary_history[7][-1]["summary"], "new summary")

    def test_summary_history_duplicate_preserves_existing_alert_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            entry = {
                "channel_id": 7,
                "run_id": "run-7",
                "summary": "door opens",
                "frame_count": 12,
                "created_at": 1_781_700_000.0,
            }
            existing = manager._normalize_summary_log_entry(entry)
            incoming = manager._normalize_summary_log_entry(entry)
            self.assertIsNotNone(existing)
            self.assertIsNotNone(incoming)
            existing["alert_counts"] = {"high": 1}
            existing["alert_total"] = 1
            existing["alert_severities"] = ["high"]
            existing["signal_digest"] = {"alerts": {"high": 1}}
            incoming["alert_counts"] = {}
            incoming["alert_total"] = 0
            incoming["alert_severities"] = []
            incoming.pop("signal_digest", None)
            manager.summary_history[7] = [existing]

            manager._merge_summary_history_locked(7, [incoming])

            merged = manager.summary_history[7][0]
            self.assertEqual(merged["alert_counts"], {"high": 1})
            self.assertEqual(merged["alert_total"], 1)
            self.assertEqual(merged["alert_severities"], ["high"])
            self.assertEqual(merged["signal_digest"], {"alerts": {"high": 1}})

    def test_stream_status_compact_exposes_latest_alert_metadata_without_logs(self):
        compact = LuxriotManager._compact_stream_status(
            "video",
            {
                "channel_id": 7,
                "running": True,
                "snapshot_count": 8,
                "snapshot_failed_count": 1,
                "slow_snapshot_count": 3,
                "snapshot_slow_threshold_sec": 2.0,
                "last_snapshot_latency_sec": 7.25,
                "avg_snapshot_latency_sec": 3.5,
                "max_snapshot_latency_sec": 10.0,
                "last_snapshot_at": 101.5,
                "logs": [
                    {"created_at": 100.0, "summary": "routine"},
                    {
                        "created_at": 101.0,
                        "batch_end_ms": 101_000,
                        "summary": "alert",
                        "alert_total": 2,
                        "alert_counts": {"low": 1, "normal": 1},
                        "alert_severities": ["low", "normal"],
                        "bookmark_failed_count": 1,
                        "bookmark_last_error": "bookmark rejected",
                    },
                ],
            },
        )

        self.assertNotIn("logs", compact)
        self.assertEqual(compact["log_count"], 2)
        self.assertEqual(compact["last_summary_at"], 101.0)
        self.assertEqual(compact["last_summary_batch_end_ms"], 101_000)
        self.assertEqual(compact["last_alert_total"], 2)
        self.assertEqual(compact["last_alert_counts"], {"low": 1, "normal": 1})
        self.assertEqual(compact["last_bookmark_failed_count"], 1)
        self.assertEqual(compact["last_bookmark_last_error"], "bookmark rejected")
        self.assertEqual(compact["snapshot_count"], 8)
        self.assertEqual(compact["snapshot_failed_count"], 1)
        self.assertEqual(compact["slow_snapshot_count"], 3)
        self.assertEqual(compact["snapshot_slow_threshold_sec"], 2.0)
        self.assertEqual(compact["last_snapshot_latency_sec"], 7.25)
        self.assertEqual(compact["avg_snapshot_latency_sec"], 3.5)
        self.assertEqual(compact["max_snapshot_latency_sec"], 10.0)
        self.assertEqual(compact["last_snapshot_at"], 101.5)

    def test_probe_capture_shares_active_video_session(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            calls = {"thumbnail": 0}

            def video_status():
                return {
                    "channel_id": 7,
                    "running": True,
                    "capture_kind": "video",
                    "summarization_enabled": True,
                    "pending_frames": 4,
                    "interval_sec": 1.0,
                    "max_buffer_frames": 120,
                    "snapshot_count": 9,
                    "last_snapshot_latency_sec": 6.25,
                    "avg_snapshot_latency_sec": 4.0,
                    "slow_snapshot_count": 5,
                    "logs": [{"summary": "hidden from compact status"}],
                }

            video_session = SimpleNamespace(
                status=video_status,
                nearest_frame_thumbnail=lambda _ts=None: calls.__setitem__("thumbnail", calls["thumbnail"] + 1) or "thumb-b64",
            )
            manager.sessions[7] = video_session

            state = manager.start_probe_capture(7, fps=3.0)

            self.assertTrue(state["running"])
            self.assertTrue(state["shared_capture"])
            self.assertEqual(state["stream_type"], "analytics")
            self.assertEqual(state["capture_kind"], "analytics")
            self.assertFalse(state["summarization_enabled"])
            self.assertEqual(state["pending_frames"], 4)
            self.assertEqual(state["last_snapshot_latency_sec"], 6.25)
            self.assertEqual(state["requested_fps"], 3.0)
            self.assertNotIn("logs", state)
            self.assertNotIn(7, manager.probe_sessions)
            self.assertIn(7, manager.shared_probe_channels)

            streams = manager.streams_status()
            analytics = [item for item in streams["analytics_streams"] if item["channel_id"] == 7]
            self.assertEqual(len(analytics), 1)
            self.assertTrue(analytics[0]["shared_capture"])
            self.assertEqual(analytics[0]["last_snapshot_latency_sec"], 6.25)
            self.assertEqual(streams["running_total"], 2)
            self.assertEqual(streams["capture_thread_total"], 1)
            self.assertEqual(streams["shared_analytics_count"], 1)
            self.assertEqual(manager.probe_frame_thumbnail(7), "thumb-b64")
            self.assertEqual(calls["thumbnail"], 1)

            stopped = manager.stop_probe_capture(7, pause=True)
            self.assertTrue(stopped["shared_capture"])
            self.assertTrue(stopped["paused"])
            self.assertNotIn(7, manager.shared_probe_channels)
            self.assertIn(7, manager.paused_probe_channels)

    def test_probe_capture_prefers_shared_video_over_existing_analytics_session(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            stopped = {"analytics": 0}
            analytics_session = SimpleNamespace(
                status=lambda: {"channel_id": 7, "running": True, "capture_kind": "analytics"},
                stop=lambda: stopped.__setitem__("analytics", stopped["analytics"] + 1),
            )
            video_session = SimpleNamespace(
                status=lambda: {
                    "channel_id": 7,
                    "running": True,
                    "capture_kind": "video",
                    "summarization_enabled": True,
                    "pending_frames": 2,
                    "interval_sec": 1.0,
                    "logs": [],
                },
                nearest_frame_thumbnail=lambda _ts=None: None,
            )
            manager.probe_sessions[7] = analytics_session
            manager.sessions[7] = video_session

            state = manager.start_probe_capture(7)

            self.assertTrue(state["shared_capture"])
            self.assertEqual(stopped["analytics"], 1)
            self.assertNotIn(7, manager.probe_sessions)
            self.assertIn(7, manager.shared_probe_channels)

    def test_summary_filters_and_l0_rollups_use_batch_bounds(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            created_at = time.time()
            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": "cat returns\nALERTS_JSON:\n{\"alerts\":[]}",
                    "frame_count": 12,
                    "created_at": created_at,
                    "batch_start_ms": 100_000,
                    "batch_end_ms": 130_000,
                },
            )

            in_window = manager.session_status(7, run_selector="all", start_ts=90.0, end_ts=140.0)
            created_window = manager.session_status(
                7,
                run_selector="all",
                start_ts=created_at - 5.0,
                end_ts=created_at + 5.0,
            )

            self.assertEqual(len(in_window["logs"]), 1)
            self.assertEqual(len(created_window["logs"]), 0)

            rollups = manager.summary_rollups(7, run_selector="all", start_ts=90.0, end_ts=140.0)
            l0 = rollups["levels"]["L0"][0]
            self.assertEqual(l0["window_start"], 100.0)
            self.assertEqual(l0["window_end"], 130.0)
            self.assertEqual(l0["window_sec"], 30)

    def test_alerts_json_prompt_is_decoupled_from_bookmark_side_effects(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                alert_parser=lambda *_args, **_kwargs: [
                    {"title": "test", "description": "visible event", "severity": "normal"}
                ],
            )
            manager.default_bookmark_enabled = False

            prompt = manager.get_effective_stream_system_prompt(7)
            self.assertIn("ALERTS_JSON", prompt)

            batch = manager.create_summary_batch(
                channel_id=7,
                run_id="run-7",
                batch_size=1,
                prompt="Describe activity.",
                model_hint=None,
                interval_sec=1.0,
                frames=[{"captured_at": 100.0, "image_b64": "ZmFrZQ=="}],
            )
            self.assertIn("ALERTS_JSON", batch["system_prompt"])
            result = manager.process_summary_alerts(
                7,
                'ALERTS_JSON:\n{"alerts":[{"title":"test","severity":"normal"}]}',
                default_ts_ms=100_000,
            )
            self.assertEqual(result, 0)
            self.assertEqual(result.parsed, 1)
            self.assertEqual(result.alert_events[0]["delivery_status"], "bookmark_disabled")

    def test_process_summary_alerts_sends_more_than_three_distinct_alerts(self):
        with tempfile.TemporaryDirectory() as temp:
            def parse_alerts(_text, _channel_id, _default_ts_ms=None):
                return [
                    {"title": "Gate wave", "description": "Person waves near the gate", "severity": "info"},
                    {"title": "Restricted lane entry", "description": "Vehicle enters a restricted lane", "severity": "low"},
                    {"title": "Fight", "description": "Two people fighting", "severity": "high"},
                    {"title": "Drifting", "description": "Vehicle drifting", "severity": "normal"},
                    {"title": "Fire", "description": "Trash bin fire", "severity": "critical"},
                ]

            manager = build_manager(Path(temp), alert_parser=parse_alerts)
            manager.default_bookmark_enabled = True
            manager.default_bookmark_cooldown_sec = 0.0
            sent = []

            def fake_bookmark(**kwargs):
                sent.append(kwargs)
                return {"success": True}

            with patch.object(manager, "send_bookmark_event", side_effect=fake_bookmark):
                count = manager.process_summary_alerts(
                    7,
                    "Batch summary\nALERTS_JSON:\n{\"alerts\":[]}",
                    default_ts_ms=1_781_700_000_000,
                )

            self.assertEqual(count, 5)
            self.assertEqual(len(sent), 5)
            self.assertEqual(
                [item["severity"] for item in sent],
                ["info", "low", "high", "normal", "critical"],
            )

    def test_process_summary_alerts_reports_bookmark_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                alert_parser=lambda *_args, **_kwargs: [
                    {"title": "Fire", "description": "Visible flame", "severity": "high"}
                ],
            )
            manager.default_bookmark_enabled = True
            manager.default_bookmark_cooldown_sec = 0.0

            with (
                self.assertLogs("luxriot_connector", level="WARNING") as logs,
                patch.object(manager, "send_bookmark_event", side_effect=RuntimeError("evo down")),
            ):
                result = manager.process_summary_alerts(
                    7,
                    "Batch summary\nALERTS_JSON:\n{\"alerts\":[]}",
                    default_ts_ms=1_781_700_000_000,
                )

            self.assertEqual(result, 0)
            self.assertEqual(result.parsed, 1)
            self.assertEqual(result.failed, 1)
            self.assertIn("evo down", result.last_error)
            self.assertIn("Luxriot bookmark send failed", "\n".join(logs.output))

    def test_high_severity_bookmarks_bypass_info_cooldown(self):
        with tempfile.TemporaryDirectory() as temp:
            current = {"severity": "info"}

            def parse_alerts(_text, _channel_id, _default_ts_ms=None):
                return [
                    {
                        "title": "Repeated event",
                        "description": "Same visual event",
                        "severity": current["severity"],
                    }
                ]

            manager = build_manager(Path(temp), alert_parser=parse_alerts)
            manager.default_bookmark_enabled = True
            manager.default_bookmark_cooldown_sec = 60.0
            sent = []

            with patch.object(manager, "send_bookmark_event", side_effect=lambda **kwargs: sent.append(kwargs) or {"success": True}):
                first = manager.process_summary_alerts(
                    7,
                    "Batch summary\nALERTS_JSON:\n{\"alerts\":[]}",
                    default_ts_ms=1_781_700_000_000,
                )
                second = manager.process_summary_alerts(
                    7,
                    "Batch summary\nALERTS_JSON:\n{\"alerts\":[]}",
                    default_ts_ms=1_781_700_001_000,
                )
                current["severity"] = "high"
                third = manager.process_summary_alerts(
                    7,
                    "Batch summary\nALERTS_JSON:\n{\"alerts\":[]}",
                    default_ts_ms=1_781_700_002_000,
                )
                fourth = manager.process_summary_alerts(
                    7,
                    "Batch summary\nALERTS_JSON:\n{\"alerts\":[]}",
                    default_ts_ms=1_781_700_003_000,
                )

            self.assertEqual(first, 1)
            self.assertEqual(second, 0)
            self.assertEqual(second.skipped_duplicate, 1)
            self.assertEqual(third, 1)
            self.assertEqual(fourth, 1)
            self.assertEqual([item["severity"] for item in sent], ["info", "high", "high"])

    def test_rollup_source_selection_preserves_salient_children_under_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            base = 1_781_700_000.0
            children = []
            for index in range(24):
                child = {
                    "window_start": base + index * 30.0,
                    "summary": f"Routine quiet corridor segment {index} with no notable changes. " * 4,
                    "alert_counts": {},
                    "signal_digest": {},
                }
                if index == 11:
                    child["summary"] = "A vehicle performs drifting turns near the gate while traffic remains otherwise routine."
                    child["alert_counts"] = {"high": 1}
                    child["signal_digest"] = {
                        "alert_events": ["high: vehicle drifting near gate"],
                        "deviations": ["vehicle drifting near gate"],
                    }
                if index == 17:
                    child["summary"] = "Camera view is partially obstructed and several frames are missing."
                    child["signal_digest"] = {"missing_data": ["partial obstruction and missing frames"]}
                children.append(child)

            lines = manager._select_rollup_source_lines(children, char_budget=1100)
            joined = "\n".join(lines)

            self.assertLess(len(lines), len(children))
            self.assertIn("SOURCE_ALERTS high=1", joined)
            self.assertIn("vehicle performs drifting", joined)
            self.assertIn("partially obstructed", joined)

    def test_prose_alert_section_can_drive_alert_metadata_and_bookmarks(self):
        with tempfile.TemporaryDirectory() as temp:
            def parse_alerts(text, channel_id, default_ts_ms=None):
                alerts = []
                if "Info Level:" in text:
                    alerts.append(
                        {
                            "title": "Person waves at the gate",
                            "description": "Person waves at the gate (visible in Snapshots 6-8).",
                            "severity": "info",
                            "channel_id": channel_id,
                            "timestamp_ms": default_ts_ms,
                        }
                    )
                if "Warning Level:" in text:
                    alerts.append(
                        {
                            "title": "Vehicle enters restricted lane",
                            "description": "Vehicle enters the restricted lane near the gate.",
                            "severity": "low",
                            "channel_id": channel_id,
                            "timestamp_ms": default_ts_ms,
                        }
                    )
                return alerts

            manager = build_manager(Path(temp), alert_parser=parse_alerts)
            summary = (
                "Summary of Activity:\n"
                "Two distinct review events are visible near the gate.\n\n"
                "Alerts:\n\n"
                "Info Level: Person waves at the gate (visible in Snapshots 6-8).\n"
                "Warning Level: Vehicle enters the restricted lane near the gate (visible in Snapshots 10-12).\n"
            )

            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": summary,
                    "frame_count": 12,
                    "created_at": time.time(),
                },
            )
            logs = manager.session_status(7, run_selector="all")["logs"]
            self.assertEqual(logs[0]["alert_counts"], {"low": 1, "info": 1})
            self.assertEqual(logs[0]["signal_digest"]["alerts"], {"low": 1, "info": 1})
            self.assertIn("Person waves", " ".join(logs[0]["signal_digest"]["alert_events"]))

            manager.default_bookmark_enabled = True
            manager.default_bookmark_cooldown_sec = 0.0
            sent = []
            with patch.object(manager, "send_bookmark_event", side_effect=lambda **kwargs: sent.append(kwargs) or {"success": True}):
                count = manager.process_summary_alerts(
                    7,
                    summary,
                    default_ts_ms=1_781_700_000_000,
                )
            self.assertEqual(count, 2)
            self.assertEqual([item["severity"] for item in sent], ["info", "low"])

    def test_lm_alert_parser_handles_prose_and_warning_severity(self):
        parser = load_lm_alert_parser()
        summary = (
            "Activity Summary:\n"
            "A person waves while a vehicle enters a restricted lane.\n\n"
            "Alerts:\n\n"
            "Warning Level: Vehicle enters the restricted lane near the gate.\n"
            "Info: Person waves at the gate.\n"
            "ALERTS_JSON:\n"
            "{\n"
            "  \"alerts\": [\n"
            "    {\n"
            "      \"title\": \"Restricted Lane Entry\",\n"
            "      \"description\": \"Vehicle enters a restricted lane near the gate.\",\n"
            "      \"severity\": \"warning\",\n"
            "      \"state\": \"new\",\n"
            "      \"channel_id\": 112,\n"
            "      \"timestamp_ms\": 0\n"
            "    }\n"
            "  ]\n"
            "}\n"
        )

        alerts = parser(summary, 112, 1750734618000)

        self.assertEqual(len(alerts), 2)
        self.assertEqual([alert["severity"] for alert in alerts], ["low", "info"])

    def test_current_summary_log_does_not_drop_history_alert_metadata(self):
        history = {
            "channel_id": 7,
            "run_id": "run-7",
            "summary": "Alerts:\nInfo: Person waves.",
            "frame_count": 12,
            "created_at": 100.0,
            "alert_counts": {"info": 1},
            "alert_total": 1,
            "alert_severities": ["info"],
        }
        current = {
            "channel_id": 7,
            "run_id": "run-7",
            "summary": "Alerts:\nInfo: Person waves.",
            "frame_count": 12,
            "created_at": 100.0,
        }

        logs = LuxriotManager._combine_summary_logs([history], [current])

        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["alert_counts"], {"info": 1})
        self.assertEqual(logs[0]["alert_total"], 1)

    def test_memory_update_json_feeds_next_l0_system_prompt(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager._update_channel_routine_context(
                channel_id=7,
                rollup_id="l2-memory",
                window_end=1234.0,
                level="L2",
                summary_text=(
                    "### Window Snapshot\n"
                    "Routine lot with one deviation.\n\n"
                    "MEMORY_UPDATE_JSON:\n"
                    "{\n"
                    "  \"routine_baseline\": \"parking lot is usually empty overnight\",\n"
                    "  \"active_watchlist\": [\"watch the east gate\"],\n"
                    "  \"preserved_deviations\": [\n"
                    "    {\"time\": \"02:10-02:12\", \"severity\": \"high\", \"event\": \"vehicle drifting\", \"evidence\": \"repeated sliding turns\"}\n"
                    "  ],\n"
                    "  \"alert_tuning_notes\": [\"drifting should stay visible even when the lot is otherwise routine\"],\n"
                    "  \"ignore_as_routine\": [\"parked maintenance vehicles\"]\n"
                    "}"
                ),
            )

            batch = manager.create_summary_batch(
                channel_id=7,
                run_id="run-7",
                batch_size=2,
                prompt="Describe activity.",
                model_hint=None,
                interval_sec=5.0,
                frames=sample_frames(),
            )

            system_prompt = batch["system_prompt"]
            self.assertIn("Active Channel Memory", system_prompt)
            self.assertIn("parking lot is usually empty overnight", system_prompt)
            self.assertIn("vehicle drifting", system_prompt)
            self.assertIn("Do not let routine baseline suppress", system_prompt)

    def test_memory_update_json_accepts_qwen_key_variants(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager._update_channel_routine_context(
                channel_id=7,
                rollup_id="l1-memory",
                window_end=1234.0,
                level="L1",
                summary_text=(
                    "### Window Snapshot\n"
                    "Routine lot with one deviation.\n\n"
                    "MEMORY_UPDATE_JSON:\n"
                    "{"
                    "\"routineBaseline\":\"quiet lot overnight\","
                    "\"activeWatchlist\":[\"east gate\"],"
                    "\"preservedDeviations\":[{\"time\":\"02:10\",\"severity\":\"high\",\"event\":\"vehicle drifting\",\"evidence\":\"sliding turns\"}],"
                    "\"alerttuningnotes\":[\"keep drifting visible\"],"
                    "\"ignoreasroutine\":[\"parked maintenance vehicles\"]"
                    "}"
                ),
            )

            routine_text = manager.channel_routine_context[7]["routine"]
            self.assertIn("quiet lot overnight", routine_text)
            self.assertIn("east gate", routine_text)
            self.assertIn("vehicle drifting", routine_text)
            self.assertIn("keep drifting visible", routine_text)
            self.assertIn("parked maintenance vehicles", routine_text)

    def test_current_observed_state_parser_extracts_present_absent_unknown(self):
        summary = (
            "### Current observed state\n"
            "- Person near entrance: present in snapshots 1-12.\n"
            "- Vehicle in restricted lane: not visible on the roadway.\n"
            "- Smoke near waste bin: uncertain; haze may be lighting glare.\n\n"
            "ALERTS_JSON:\n{\"alerts\":[]}"
        )

        observations = LuxriotManager._extract_current_observed_states(summary)

        self.assertEqual(
            [(item["key"], item["state"]) for item in observations],
            [
                ("person near entrance", "present"),
                ("vehicle in restricted lane", "absent"),
                ("smoke near waste bin", "unknown"),
            ],
        )

        self.assertEqual(
            LuxriotManager._extract_current_observed_states("Person near entrance: present."),
            [],
        )

    def test_golden_state_transition_disappears_across_batches_after_confirmation(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                config_overrides={
                    "LUXRIOT_STATE_TRANSITIONS_ENABLED": True,
                    "LUXRIOT_STATE_TRANSITION_CONFIRM_BATCHES": 2,
                    "LUXRIOT_STATE_TRANSITION_ALERT_EVENTS": True,
                },
            )

            first = manager.accept_summary_entry(
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": (
                        "### Current observed state\n"
                        "- Person near entrance: present in snapshots 1-12.\n\n"
                        "ALERTS_JSON:\n{\"alerts\":[]}"
                    ),
                    "frame_count": 12,
                    "created_at": 100.0,
                    "batch_start_ms": 100000,
                    "batch_end_ms": 112000,
                }
            )
            second = manager.accept_summary_entry(
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": (
                        "### Current observed state\n"
                        "- Person near entrance: absent; entrance area is clear.\n\n"
                        "ALERTS_JSON:\n{\"alerts\":[]}"
                    ),
                    "frame_count": 12,
                    "created_at": 113.0,
                    "batch_start_ms": 113000,
                    "batch_end_ms": 125000,
                }
            )
            third = manager.accept_summary_entry(
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": (
                        "### Current observed state\n"
                        "- Person near entrance: absent; entrance area remains clear.\n\n"
                        "ALERTS_JSON:\n{\"alerts\":[]}"
                    ),
                    "frame_count": 12,
                    "created_at": 126.0,
                    "batch_start_ms": 126000,
                    "batch_end_ms": 138000,
                }
            )

            self.assertEqual(first["state_transition_total"], 0)
            self.assertEqual(second["state_transition_total"], 0)
            self.assertEqual(third["state_transition_total"], 1)
            transition = third["state_transition_events"][0]
            self.assertEqual(transition["key"], "person near entrance")
            self.assertEqual(transition["event_type"], "disappearance")
            self.assertEqual(transition["from_state"], "present")
            self.assertEqual(transition["to_state"], "absent")
            self.assertEqual(third["alert_counts"], {"info": 1})
            self.assertEqual(third["alert_events"][0]["delivery_status"], "state_tracker")

    def test_summary_rollups_expose_l0_provenance_and_rollup_aggregates(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": "Person fell near entrance. Warning Level: person down.",
                    "frame_count": 12,
                    "created_at": 100.0,
                    "batch_start_ms": 100000,
                    "batch_end_ms": 112000,
                    "alert_counts": {"high": 1},
                    "alert_total": 1,
                    "parser_alert_count": 2,
                    "json_alert_count": 1,
                    "prose_alert_count": 2,
                    "alert_events": [
                        {
                            "title": "Person down",
                            "description": "Person fell near entrance.",
                            "severity": "high",
                            "state": "new",
                            "channel_id": 7,
                            "timestamp_ms": 108000,
                            "delivery_status": "cooldown_skipped",
                        }
                    ],
                    "state_observations": [
                        {
                            "key": "person near entrance",
                            "label": "Person near entrance",
                            "state": "present",
                            "evidence": "visible on the floor near entrance",
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
                            "evidence": "confirmed by current observed state",
                        }
                    ],
                    "state_transition_total": 1,
                },
            )

            rollups = manager.summary_rollups(7, run_selector="all", level_limit=10, synthesize=False)
            l0 = rollups["levels"]["L0"][0]
            self.assertEqual(l0["alert_parser_breakdown"]["prose_only_signal_count"], 1)
            self.assertEqual(l0["alert_delivery_breakdown"]["cooldown_skipped"], 1)
            self.assertEqual(l0["alert_events"][0]["delivery_status"], "cooldown_skipped")
            self.assertEqual(l0["state_observations"][0]["state"], "present")
            self.assertEqual(l0["state_transition_events"][0]["event_type"], "appearance")

            l1 = rollups["levels"]["L1"][0]
            self.assertEqual(l1["alert_parser_breakdown"]["prose_only_signal_count"], 1)
            self.assertEqual(l1["alert_delivery_breakdown"]["cooldown_skipped"], 1)
            self.assertEqual(l1["state_transition_total"], 1)
            self.assertNotIn("alert_events", l1)
            self.assertNotIn("state_observations", l1)
            self.assertNotIn("state_transition_events", l1)

    def test_channel_status_digest_tracks_alert_titles_health_and_runtime_overlay(self):
        def parser(_summary, channel_id, timestamp_ms):
            return [
                {
                    "title": "Person down",
                    "description": "Person appears to need help near entrance.",
                    "severity": "high",
                    "state": "new",
                    "channel_id": channel_id,
                    "timestamp_ms": timestamp_ms,
                }
            ]

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp), alert_parser=parser)
            manager.accept_summary_entry(
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": (
                        "Person appears to need help near entrance.\n"
                        "### Current observed state\n"
                        "- Person near entrance: present; visible near doorway.\n"
                        "ALERTS_JSON:\n"
                        "{\"alerts\":[{\"title\":\"Person down\",\"severity\":\"high\",\"state\":\"new\"}]}"
                    ),
                    "frame_count": 12,
                    "created_at": 100.0,
                    "batch_start_ms": 100000,
                    "batch_end_ms": 112000,
                }
            )
            session = LuxriotCaptureSession(
                manager,
                channel_id=7,
                batch_size=12,
                prompt="Describe.",
                run_id="run-7",
                model_hint="vlm-a1",
            )
            session.frames = [{"thumbnail": "a"}, {"thumbnail": "b"}]
            session.dropped_frames = 3
            session.queue_dropped_batches = 2
            session.last_error = "snapshot timeout"
            manager.sessions[7] = session

            digest = manager.system_status_digest(channel_ids=[7])["channels"][0]
            self.assertEqual(digest["channel_id"], 7)
            self.assertFalse(digest["running"])
            self.assertEqual(digest["video_lm"], "vlm-a1")
            self.assertEqual(digest["pending_frames"], 2)
            self.assertEqual(digest["dropped_frames"], 3)
            self.assertEqual(digest["dropped_batches"], 2)
            self.assertEqual(digest["last_error"], "snapshot timeout")
            self.assertEqual(digest["recent_alerts"][0]["title"], "Person down")
            self.assertEqual(digest["recent_alerts"][0]["delivery_status"], "bookmark_disabled")
            self.assertEqual(digest["alert_counts_by_severity"]["high"], 1)
            self.assertEqual(digest["alert_delivery_breakdown"]["bookmark_disabled"], 1)
            self.assertEqual(digest["alert_parser_breakdown"]["json_alert_count"], 1)
            self.assertEqual(digest["current_observed_state"][0]["state"], "present")

    def test_channel_status_digest_rebuilds_from_persisted_summary_history(self):
        def parser(_summary, channel_id, timestamp_ms):
            return [
                {
                    "title": "Smoke visible",
                    "description": "Smoke is visible near roadway.",
                    "severity": "normal",
                    "state": "new",
                    "channel_id": channel_id,
                    "timestamp_ms": timestamp_ms,
                }
            ]

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            manager = build_manager(path, alert_parser=parser)
            manager.accept_summary_entry(
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": (
                        "Smoke visible near roadway.\n"
                        "ALERTS_JSON:\n"
                        "{\"alerts\":[{\"title\":\"Smoke visible\",\"severity\":\"normal\",\"state\":\"new\"}]}"
                    ),
                    "frame_count": 6,
                    "created_at": 200.0,
                    "batch_start_ms": 200000,
                    "batch_end_ms": 206000,
                }
            )
            manager.persist_summary_state()

            restored = build_manager(path, alert_parser=parser)
            digest = restored.system_status_digest(channel_ids=[7])["channels"][0]
            self.assertEqual(digest["summary_count"], 1)
            self.assertEqual(digest["recent_alerts"][0]["title"], "Smoke visible")
            self.assertEqual(digest["alert_counts_by_severity"]["normal"], 1)
            self.assertTrue(digest["rebuilt_from_history"])

    def test_stream_status_ignores_stale_digest_alert_titles_for_newer_session_log(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            with manager.cache_lock:
                manager.channel_status_digest[7] = {
                    "channel_id": 7,
                    "last_summary_ts": 100.0,
                    "recent_alerts": [{"title": "Old alert", "severity": "high"}],
                    "alert_counts_by_severity": {"high": 1},
                    "alert_delivery_breakdown": {"sent": 1},
                    "alert_parser_breakdown": {"json_alert_count": 1},
                }
            session = LuxriotCaptureSession(
                manager,
                channel_id=7,
                batch_size=12,
                prompt="Describe.",
                run_id="run-7",
                model_hint="vlm-a1",
            )
            with session.lock:
                session.logs.append({"created_at": 120.0, "summary": "newer log", "frame_count": 1})
            manager.sessions[7] = session

            digest = manager.streams_status()["channel_status_digest"][0]
            self.assertTrue(digest["stale_digest"])
            self.assertNotIn("recent_alerts", digest)
            self.assertNotIn("alert_counts_by_severity", digest)

    def test_state_transition_unknown_observation_does_not_confirm_change(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                config_overrides={
                    "LUXRIOT_STATE_TRANSITIONS_ENABLED": True,
                    "LUXRIOT_STATE_TRANSITION_CONFIRM_BATCHES": 1,
                    "LUXRIOT_STATE_TRANSITION_ALERT_EVENTS": True,
                },
            )
            manager.accept_summary_entry(
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": "### Current observed state\n- Person near entrance: present.\n",
                    "frame_count": 12,
                    "created_at": 100.0,
                    "batch_start_ms": 100000,
                    "batch_end_ms": 112000,
                }
            )
            unknown = manager.accept_summary_entry(
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": "### Current observed state\n- Person near entrance: uncertain; partly out of frame.\n",
                    "frame_count": 12,
                    "created_at": 113.0,
                    "batch_start_ms": 113000,
                    "batch_end_ms": 125000,
                }
            )

            self.assertEqual(unknown["state_transition_total"], 0)

    def test_rollup_prompt_layers_and_source_alerts_are_visible(self):
        with tempfile.TemporaryDirectory() as temp:
            captured_user_texts = []

            def lm_callback(messages, _model):
                captured_user_texts.append(messages[1]["content"][0]["text"])
                return (
                    "### Window Snapshot\n"
                    "Window with source alerts.\n\n"
                    "### Alert Ledger\n"
                    "- normal=1, low=1 preserved from source alerts.\n\n"
                    "MEMORY_UPDATE_JSON:\n"
                    "{\"routine_baseline\":\"quiet test scene\",\"alert_tuning_notes\":[\"preserve source alerts\"]}"
                )

            manager = build_manager(Path(temp), lm_callback=lm_callback)
            manager.rollup_llm_levels = {"L1"}
            manager.rollup_llm_max_new_per_call = 10
            base = 1_781_700_000.0
            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": "Person entered the frame.",
                    "frame_count": 12,
                    "created_at": base,
                    "alert_counts": {"normal": 1},
                },
            )
            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": "Doorway was briefly obstructed.",
                    "frame_count": 12,
                    "created_at": base + 30.0,
                    "alert_counts": {"low": 1},
                },
            )

            settings = manager.get_prompt_settings(channel_id=7)
            self.assertIn("prompt_layers", settings)
            self.assertIn("Alert Ledger must mention every source alert", settings["prompt_layers"]["rollups"]["L1"]["backend_instructions"])

            rollups = manager.summary_rollups(7, run_selector="all", level_limit=10)
            self.assertTrue(captured_user_texts)
            user_text = captured_user_texts[0]
            self.assertIn("Source alert totals: normal=1, low=1", user_text)
            self.assertIn("Window signal digest (compact continuity map):", user_text)
            self.assertIn("Alerts: normal=1, low=1", user_text)
            self.assertIn("[SOURCE_ALERTS normal=1]", user_text)
            self.assertIn("[SOURCE_ALERTS low=1]", user_text)
            stats = rollups["levels"]["L1"][0]["llm_input_stats"]
            self.assertEqual(stats["phase"], "rollup_request_built")
            self.assertEqual(stats["level"], "L1")
            self.assertEqual(stats["source_level"], "L0")
            self.assertGreaterEqual(stats["source_lines_selected"], 2)
            self.assertGreater(stats["text_chars"], 0)

    def test_summary_rollups_readonly_mode_does_not_synthesize_with_llm(self):
        with tempfile.TemporaryDirectory() as temp:
            calls = []

            def lm_callback(messages, _model):
                calls.append(messages)
                return "LLM rollup"

            manager = build_manager(Path(temp), lm_callback=lm_callback)
            manager.rollup_llm_levels = {"L1", "L2", "L3"}
            manager.rollup_llm_max_new_per_call = 10
            base = 1_781_700_000.0
            for offset in (0.0, 30.0, 60.0):
                manager.record_summary_log(
                    7,
                    {
                        "channel_id": 7,
                        "run_id": "run-7",
                        "summary": f"Routine loading bay activity {offset}.",
                        "frame_count": 12,
                        "created_at": base + offset,
                    },
                )

            rollups = manager.summary_rollups(7, run_selector="all", level_limit=10, synthesize=False)

            self.assertEqual(calls, [])
            self.assertIn("L1 rollup from L0", rollups["levels"]["L1"][0]["summary"])

    def test_rollup_cache_signature_changes_when_child_metadata_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            calls = []

            def lm_callback(messages, _model):
                calls.append(messages[1]["content"][0]["text"])
                return (
                    "### Window Snapshot\n"
                    f"L2 cache pass {len(calls)}.\n\n"
                    "MEMORY_UPDATE_JSON:\n"
                    f"{{\"routine_baseline\":\"cache pass {len(calls)}\"}}"
                )

            manager = build_manager(Path(temp), lm_callback=lm_callback)
            manager.rollup_llm_levels = {"L2"}
            manager.rollup_llm_max_new_per_call = 10
            base = 1_781_700_000.0
            for offset, summary in (
                (0.0, "Routine door activity."),
                (30.0, "Person crosses the doorway."),
            ):
                manager.record_summary_log(
                    7,
                    {
                        "channel_id": 7,
                        "run_id": "run-7",
                        "summary": summary,
                        "frame_count": 2,
                        "created_at": base + offset,
                        "alert_counts": {"info": 1},
                    },
                )

            first = manager.summary_rollups(7, run_selector="all", level_limit=10)
            self.assertEqual(len(calls), 1)
            self.assertIn("cache pass 1", first["levels"]["L2"][0]["summary"])

            manager.summary_history[7][0]["alert_counts"] = {"critical": 1}

            second = manager.summary_rollups(7, run_selector="all", level_limit=10)
            self.assertEqual(len(calls), 2)
            self.assertIn("cache pass 2", second["levels"]["L2"][0]["summary"])

    def test_fallback_rollup_summary_preserves_alerts_deviations_and_signal_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                config_overrides={
                    "LUXRIOT_ROLLUP_LLM_LEVELS": "none",
                    "LUXRIOT_ROLLUP_L1_LLM_ENABLED": True,
                },
            )
            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": (
                        "Quiet parking lot with one deviation.\n"
                        "MEMORY_UPDATE_JSON:\n"
                        "{\"preserved_deviations\":[{\"time\":\"02:10\",\"severity\":\"high\","
                        "\"event\":\"vehicle drifting\",\"evidence\":\"sliding turns\"}],"
                        "\"active_watchlist\":[\"east gate vehicle\"],"
                        "\"alert_tuning_notes\":[\"keep drifting visible\"]}"
                    ),
                    "frame_count": 2,
                    "created_at": 1_781_700_000.0,
                    "alert_counts": {"high": 1},
                },
            )

            rollups = manager.summary_rollups(7, run_selector="all", level_limit=10)
            summary = rollups["levels"]["L1"][0]["summary"]

            self.assertIn("Alert counts: high=1", summary)
            self.assertIn("Preserved deviations:", summary)
            self.assertIn("vehicle drifting", summary)
            self.assertIn("Signal digest:", summary)
            self.assertIn("Alerts: high=1", summary)

    def test_summary_rollups_preserve_deviation_memory_across_levels(self):
        with tempfile.TemporaryDirectory() as temp:
            def lm_callback(messages, _model):
                user_text = messages[1]["content"][0]["text"]
                if "Target level: L3" in user_text:
                    return (
                        "### Window Snapshot\n"
                        "Longer period mostly routine.\n\n"
                        "### Routine Baseline\n"
                        "Quiet exterior road.\n\n"
                        "### Preserved Deviations\n"
                        "- 02:10 vehicle drifting near the gate.\n\n"
                        "### Alert Ledger\n"
                        "- high | 02:10 | vehicle drifting | sliding turns visible.\n\n"
                        "MEMORY_UPDATE_JSON:\n"
                        "{\"routine_baseline\":\"quiet exterior road\","
                        "\"preserved_deviations\":[{\"time\":\"02:10\",\"severity\":\"high\",\"event\":\"vehicle drifting\",\"evidence\":\"sliding turns visible\"}],"
                        "\"alert_tuning_notes\":[\"do not collapse drifting into routine traffic\"],"
                        "\"ignore_as_routine\":[\"normal parked cars\"]}"
                    )
                if "Target level: L2" in user_text:
                    return (
                        "### Window Snapshot\n"
                        "Hour mostly routine with one security event.\n\n"
                        "### Routine Baseline\n"
                        "Low traffic near the gate.\n\n"
                        "### Preserved Deviations\n"
                        "- 02:10 vehicle drifting.\n\n"
                        "MEMORY_UPDATE_JSON:\n"
                        "{\"routine_baseline\":\"low traffic near the gate\","
                        "\"preserved_deviations\":[{\"time\":\"02:10\",\"severity\":\"high\",\"event\":\"vehicle drifting\",\"evidence\":\"repeated sharp turns\"}]}"
                    )
                return (
                    "### Window Snapshot\n"
                    "Short window with a drifting event.\n\n"
                    "### Preserved Deviations\n"
                    "- 02:10 vehicle drifting.\n\n"
                    "MEMORY_UPDATE_JSON:\n"
                    "{\"active_watchlist\":[\"east gate vehicle\"],"
                    "\"preserved_deviations\":[{\"time\":\"02:10\",\"severity\":\"high\",\"event\":\"vehicle drifting\",\"evidence\":\"sliding turns\"}]}"
                )

            manager = build_manager(Path(temp), lm_callback=lm_callback)
            manager.rollup_llm_levels = {"L1", "L2", "L3"}
            manager.rollup_llm_max_new_per_call = 10
            base = 1_781_700_000.0
            for offset, summary in (
                (0.0, "Routine traffic."),
                (30.0, "A vehicle performs repeated sharp turns near the gate."),
            ):
                manager.record_summary_log(
                    7,
                    {
                        "channel_id": 7,
                        "run_id": "run-7",
                        "summary": summary,
                        "frame_count": 2,
                        "created_at": base + offset,
                    },
                )

            rollups = manager.summary_rollups(7, run_selector="all", level_limit=10)
            routine_text = rollups["routine_context"]["routine"]
            self.assertIn("quiet exterior road", routine_text)
            self.assertIn("vehicle drifting", routine_text)
            self.assertIn("do not collapse drifting into routine traffic", routine_text)

            batch = manager.create_summary_batch(
                channel_id=7,
                run_id="run-7",
                batch_size=2,
                prompt="Describe activity.",
                model_hint=None,
                interval_sec=5.0,
                frames=sample_frames(),
            )
            self.assertIn("vehicle drifting", batch["system_prompt"])

    def test_older_memory_update_merges_items_without_replacing_newer_baseline(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager._update_channel_routine_context(
                channel_id=7,
                rollup_id="l2-new",
                window_end=200.0,
                level="L2",
                summary_text=(
                    "MEMORY_UPDATE_JSON:\n"
                    "{\"routine_baseline\":\"new quiet lobby baseline\","
                    "\"active_watchlist\":[\"north door\"]}"
                ),
            )
            manager._update_channel_routine_context(
                channel_id=7,
                rollup_id="l1-old",
                window_end=100.0,
                level="L1",
                summary_text=(
                    "MEMORY_UPDATE_JSON:\n"
                    "{\"routine_baseline\":\"old busy lobby baseline\","
                    "\"active_watchlist\":[\"east gate vehicle\"],"
                    "\"preserved_deviations\":[{\"time\":\"02:10\",\"severity\":\"high\","
                    "\"event\":\"vehicle drifting\",\"evidence\":\"sliding turns\"}],"
                    "\"alert_tuning_notes\":[\"keep drifting visible\"]}"
                ),
            )

            context = manager.channel_routine_context[7]
            memory = context["memory"]
            routine_text = context["routine"]

            self.assertEqual(context["rollup_id"], "l2-new")
            self.assertEqual(context["window_end"], 200.0)
            self.assertEqual(memory["routine_baseline"], "new quiet lobby baseline")
            self.assertIn("north door", routine_text)
            self.assertIn("east gate vehicle", routine_text)
            self.assertIn("vehicle drifting", routine_text)
            self.assertIn("keep drifting visible", routine_text)
            self.assertNotIn("old busy lobby baseline", routine_text)

    def test_rollup_llm_levels_none_and_off_disable_all_rollup_llm(self):
        for raw_value in ("none", "off"):
            with tempfile.TemporaryDirectory() as temp:
                manager = build_manager(
                    Path(temp),
                    config_overrides={
                        "LUXRIOT_ROLLUP_LLM_LEVELS": raw_value,
                        "LUXRIOT_ROLLUP_L1_LLM_ENABLED": True,
                    },
                )

                self.assertEqual(manager.rollup_llm_levels, set())

    def test_start_session_persists_channel_interval_override(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))

            with patch.object(LuxriotCaptureSession, "start", return_value=None):
                status = manager.start_session(
                    7,
                    batch_size=12,
                    prompt="Describe activity.",
                    interval_sec=4.5,
                )

            self.assertEqual(status["interval_sec"], 4.5)
            self.assertEqual(manager.channel_prompt_overrides[7]["capture_interval_sec"], 4.5)
            manager.stop_session(7)

            with patch.object(LuxriotCaptureSession, "start", return_value=None):
                status = manager.start_session(
                    7,
                    batch_size=12,
                    prompt="Describe activity.",
                )

            self.assertEqual(status["interval_sec"], 4.5)
            manager.stop_session(7)

    def test_start_and_stop_session_update_desired_state(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime_store = MemoryRuntimeStateStore()
            manager = build_manager(Path(temp), runtime_state_store=runtime_store)

            with patch.object(LuxriotCaptureSession, "start", return_value=None):
                manager.start_session(
                    7,
                    batch_size=12,
                    prompt="Describe activity.",
                    model_hint="vlm-a1",
                    interval_sec=4.5,
                )

            state = runtime_store.load_state(manager.DESIRED_LIVE_SESSIONS_KEY)
            self.assertTrue(state["sessions"]["7"]["enabled"])
            self.assertEqual(state["sessions"]["7"]["batch_size"], 12)
            self.assertEqual(state["sessions"]["7"]["prompt"], "Describe activity.")
            self.assertEqual(state["sessions"]["7"]["model_hint"], "vlm-a1")
            self.assertEqual(state["sessions"]["7"]["interval_sec"], 4.5)

            manager.stop_session(7)

            state = runtime_store.load_state(manager.DESIRED_LIVE_SESSIONS_KEY)
            self.assertFalse(state["sessions"]["7"]["enabled"])

    def test_restore_desired_live_sessions_starts_enabled_channels(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime_store = MemoryRuntimeStateStore()
            runtime_store.save_state(
                LuxriotManager.DESIRED_LIVE_SESSIONS_KEY,
                {
                    "version": 1,
                    "sessions": {
                        "7": {
                            "enabled": True,
                            "batch_size": 12,
                            "prompt": "Describe activity.",
                            "model_hint": "vlm-a1",
                            "interval_sec": 4.5,
                        },
                        "8": {"enabled": False},
                    },
                },
            )
            manager = build_manager(Path(temp), runtime_state_store=runtime_store)

            with patch.object(LuxriotCaptureSession, "start", return_value=None):
                result = manager.restore_desired_live_sessions()

            self.assertTrue(result["ok"])
            self.assertEqual(result["restored_count"], 1)
            self.assertIn(7, manager.sessions)
            self.assertEqual(manager.sessions[7].model_hint, "vlm-a1")
            self.assertEqual(manager.sessions[7].interval, 4.5)


class LuxriotInferenceQueueRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.repository = InMemoryInferenceQueueRepository()
        self.manager = build_manager(
            self.directory,
            lm_callback=lambda _messages, _model: "queued summary",
        )
        self.runtime = LuxriotInferenceQueueRuntime(
            manager=self.manager,
            enqueue_repository=self.repository,
            worker_repository=self.repository,
            tenant_id=str(uuid4()),
            capacity=1,
            spool_directory=self.directory / "spool",
            default_model="qwen35-9b-q4_k_m",
            worker_count=1,
            poll_interval_seconds=0.01,
            lease_seconds=10.0,
        )
        self.manager.set_summary_dispatcher(self.runtime.enqueue_summary)

    def tearDown(self):
        self.runtime.stop()
        self.temp.cleanup()

    def batch(self, *, channel_id=7, start=100.0, run_id="run-7"):
        return self.manager.create_summary_batch(
            channel_id=channel_id,
            run_id=run_id,
            batch_size=2,
            prompt="Describe activity.",
            model_hint=None,
            interval_sec=5.0,
            frames=sample_frames(start),
        )

    def wait_for(self, predicate, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        self.fail("condition was not reached before timeout")

    def test_worker_applies_durable_result_and_removes_spool(self):
        outcome = self.runtime.enqueue_summary(self.batch())
        self.assertTrue(outcome["accepted"])

        spool_files = list((self.directory / "spool").glob("*.json"))
        self.assertEqual(len(spool_files), 1)
        mode = stat.S_IMODE(os.stat(spool_files[0]).st_mode)
        self.assertEqual(mode, 0o600)

        self.runtime.start()
        self.wait_for(lambda: bool(self.manager.summary_history.get(7)))

        self.assertEqual(
            self.manager.summary_history[7][0]["summary"],
            "queued summary",
        )
        self.wait_for(lambda: not list((self.directory / "spool").glob("*.json")))
        self.assertEqual(list((self.directory / "spool").glob("*.json")), [])
        self.wait_for(lambda: self.runtime.status()["completed_count"] == 1)
        self.assertEqual(self.runtime.status()["completed_count"], 1)

    def test_worker_archives_batch_frame_anchors(self):
        archived = []
        self.manager.set_summary_archive_callback(
            lambda entry: archived.append(dict(entry)) or {
                "attempted": len(entry.get("archive_frames") or []),
                "inserted": len(entry.get("archive_frames") or []),
                "summary_frames": len(entry.get("archive_frames") or []),
                "alert_frames": 0,
            }
        )

        outcome = self.runtime.enqueue_summary(self.batch())
        self.assertTrue(outcome["accepted"])

        self.runtime.start()
        self.wait_for(lambda: bool(archived))

        frames = archived[0]["archive_frames"]
        self.assertEqual([frame["anchor_role"] for frame in frames], ["first", "last"])
        self.assertEqual([frame["timestamp_ms"] for frame in frames], [100000, 105000])
        self.assertNotIn("archive_frames", self.manager.summary_history[7][0])

    def test_worker_restores_queued_default_model_hint(self):
        seen_models = []

        def lm_callback(_messages, model):
            seen_models.append(model)
            return "queued summary"

        manager = build_manager(self.directory, lm_callback=lm_callback)
        repository = InMemoryInferenceQueueRepository()
        runtime = LuxriotInferenceQueueRuntime(
            manager=manager,
            enqueue_repository=repository,
            worker_repository=repository,
            tenant_id=str(uuid4()),
            capacity=1,
            spool_directory=self.directory / "spool-model",
            default_model="vlm-a",
            worker_count=1,
            poll_interval_seconds=0.01,
            lease_seconds=10.0,
        )
        try:
            manager.set_summary_dispatcher(runtime.enqueue_summary)
            outcome = runtime.enqueue_summary(self.batch())
            self.assertTrue(outcome["accepted"])
            runtime.start()
            self.wait_for(lambda: bool(seen_models))
        finally:
            runtime.stop()

        self.assertEqual(seen_models, ["vlm-a"])

    def test_queued_heartbeat_coalescing_removes_superseded_spool(self):
        self.runtime.stop()
        first = self.runtime.enqueue_summary(self.batch(start=100.0))
        second = self.runtime.enqueue_summary(self.batch(start=200.0))

        self.assertEqual(first["job_id"], second["job_id"])
        self.assertEqual(second["status"], "coalesced")
        spool_files = list((self.directory / "spool").glob("*.json"))
        self.assertEqual(len(spool_files), 1)
        job = self.repository.get_job(second["job_id"])
        self.assertEqual(spool_files[0].name, job.payload["spool_file"])

    def test_manual_work_evicts_heartbeat_and_removes_its_spool(self):
        heartbeat = self.runtime.enqueue_summary(self.batch(channel_id=7))
        manual = self.runtime.enqueue_summary(
            self.batch(channel_id=8, run_id="run-8"),
            workload_class="manual",
        )

        self.assertTrue(manual["accepted"])
        self.assertEqual(manual["evicted_job_id"], heartbeat["job_id"])
        spool_files = list((self.directory / "spool").glob("*.json"))
        self.assertEqual(len(spool_files), 1)
        manual_job = self.repository.get_job(manual["job_id"])
        self.assertEqual(spool_files[0].name, manual_job.payload["spool_file"])


if __name__ == "__main__":
    unittest.main()
