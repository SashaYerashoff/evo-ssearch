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

import random

import numpy as np
from PIL import Image, ImageFilter

from inference_queue import (
    InMemoryInferenceQueueRepository,
    LuxriotInferenceQueueRuntime,
)
from archive_store import PostgresRuntimeStateStore
from attention_policy import AttentionMode, CostBudgetState
from incident_attention import PromptBudgetError
from luxriot_connector import (
    DEFAULT_ALERTS_JSON_PROMPT,
    PREVIOUS_ENGLISH_BATCH_STATE_JSON_PROMPT,
    PREVIOUS_PRE_ENGLISH_BATCH_STATE_JSON_PROMPT,
    PREVIOUS_VERBOSE_BATCH_STATE_JSON_PROMPT,
    PREVIOUS_COMPACT_BATCH_STATE_JSON_PROMPT,
    LuxriotCaptureSession,
    LuxriotManager,
    _intel_hwaccel_for_config,
)
from temporal_memory import make_observation as make_temporal_observation


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
        LUXRIOT_ALERT_DEDUPE_WINDOW_SEC=600.0,
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


def test_intel_hwaccel_auto_falls_through_from_qsv_to_vaapi() -> None:
    config = SimpleNamespace(LUXRIOT_FFMPEG_HWACCEL="auto", LUXRIOT_FFMPEG_INTEL_DEVICE="")
    with (
        patch("luxriot_connector._discover_intel_render_device", return_value="/dev/dri/renderD128"),
        patch("luxriot_connector._probe_qsv_runtime", return_value=(False, "MFX session failed")),
        patch("luxriot_connector._probe_vaapi_runtime", return_value=(True, None)),
    ):
        assert _intel_hwaccel_for_config(config) == ("vaapi", "/dev/dri/renderD128", None)


class DurableRollupMemoryStateStore(MemoryRuntimeStateStore):
    def __init__(self):
        super().__init__()
        self.rollups = {}

    def save_rollup(self, payload):
        self.rollups[str(payload["rollup_id"])] = dict(payload)

    def save_rollups(self, payloads):
        written = 0
        for payload in payloads:
            key = str(payload["rollup_id"])
            if key in self.rollups:
                continue
            self.save_rollup(payload)
            written += 1
        return written

    def load_rollup(self, rollup_id):
        payload = self.rollups.get(str(rollup_id))
        return dict(payload) if payload else None

    def list_rollups(self, *, channel_id, start_ts=None, end_ts=None, levels=None, limit=10000):
        allowed = {str(level).upper() for level in (levels or ("L1", "L2", "L3"))}
        rows = []
        for payload in self.rollups.values():
            if int(payload.get("channel_id") or 0) != int(channel_id):
                continue
            if str(payload.get("level") or "").upper() not in allowed:
                continue
            if start_ts is not None and float(payload.get("window_end") or 0) < float(start_ts):
                continue
            if end_ts is not None and float(payload.get("window_start") or 0) > float(end_ts):
                continue
            rows.append(dict(payload))
        rows.sort(key=lambda row: float(row.get("window_start") or 0))
        return rows[:limit]

    def prune_rollups(self, cutoff_ts):
        before = len(self.rollups)
        self.rollups = {
            key: payload
            for key, payload in self.rollups.items()
            if float(payload.get("window_end") or 0) >= float(cutoff_ts)
        }
        return before - len(self.rollups)


class BlockingPostgresRuntimeStateStore(MemoryRuntimeStateStore):
    backend = "postgres"

    def __init__(self):
        super().__init__()
        self.save_started = threading.Event()
        self.release_save = threading.Event()

    def save_state(self, key, payload):
        self.save_started.set()
        if not self.release_save.wait(timeout=3.0):
            raise RuntimeError("test persistence release timed out")
        super().save_state(key, payload)


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


def operator_rollup_response(
    overview: str,
    *,
    routine: str = "No stable routine established.",
    observations: str = "No distinct exception recorded.",
    alerts: str = "No structured alerts recorded.",
    coverage: str = "No coverage interruption recorded.",
    takeaway: str = "No operator action suggested.",
    memory: Optional[Dict[str, Any]] = None,
) -> str:
    return (
        f"### Period Overview\n{overview}\n\n"
        f"### Routine and Behavior\n{routine}\n\n"
        f"### Notable Observations and Exceptions\n{observations}\n\n"
        f"### Alerts and Meaning\n{alerts}\n\n"
        f"### Coverage and Interruptions\n{coverage}\n\n"
        f"### Operator Takeaway\n{takeaway}\n\n"
        "MEMORY_UPDATE_JSON:\n"
        + json.dumps(memory or {}, ensure_ascii=False)
    )


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


def _deterministic_noise_frame(
    seed: int,
    blur_radius: float = 0.0,
    size: Tuple[int, int] = (320, 180),
) -> Image.Image:
    rnd = random.Random(seed)
    raw = bytes(rnd.randrange(256) for _ in range(size[0] * size[1]))
    frame = Image.frombytes("L", size, raw).convert("RGB")
    if blur_radius > 0:
        frame = frame.filter(ImageFilter.GaussianBlur(blur_radius))
    return frame


class LuxriotCaptureApexDeciderTests(unittest.TestCase):
    def _make_session(self, temp: str) -> Tuple[LuxriotManager, LuxriotCaptureSession]:
        manager = build_manager(Path(temp))
        session = LuxriotCaptureSession(
            manager,
            channel_id=7,
            batch_size=12,
            prompt="Describe activity.",
            run_id="run-7",
            interval_override=0.2,
        )
        return manager, session

    def test_normal_second_prefers_sharp_frame_of_the_same_action(self):
        with tempfile.TemporaryDirectory() as temp:
            _manager, session = self._make_session(temp)
            session._accept_captured_frame(_deterministic_noise_frame(1), 1_000, summarize=False)
            blurry_peak = _deterministic_noise_frame(2, blur_radius=6.0)
            # The sharp frame carries slightly less motion than the blurred
            # peak but stays inside the active band; v1 would ship the smear.
            sharp_active = Image.blend(blurry_peak, _deterministic_noise_frame(3), 0.75)
            session._accept_captured_frame(blurry_peak, 2_100, summarize=False)
            session._accept_captured_frame(sharp_active, 2_500, summarize=False)
            session._flush_capture_apex_bucket()

            selection = session.recent_frame_items()[-1]["capture_selection"]
            self.assertEqual(selection["policy"], "capture_per_second_cv_apex_v2")
            self.assertEqual(selection["selection_mode"], "normal")
            self.assertEqual(selection["selection_source"], "capture_cv_sharp_active")
            self.assertEqual(selection["selected_timestamp_ms"], 2_500)
            self.assertTrue(selection["apex_available"])
            self.assertTrue(selection["baseline"]["warmup"])
            self.assertEqual(selection["score_source"], "find_edges_variance")

    def test_burst_second_keeps_motion_peak_and_attaches_sharper_companion(self):
        with tempfile.TemporaryDirectory() as temp:
            _manager, session = self._make_session(temp)
            session.capture_activity_baseline_level = 0.001
            session.capture_activity_baseline_dev = 0.0002
            session.capture_activity_baseline_buckets = 500
            session._accept_captured_frame(_deterministic_noise_frame(1), 1_000, summarize=False)
            blurry_peak = _deterministic_noise_frame(2, blur_radius=6.0)
            sharp_companion = Image.blend(blurry_peak, _deterministic_noise_frame(3), 0.6)
            session._accept_captured_frame(blurry_peak, 2_100, summarize=False)
            session._accept_captured_frame(sharp_companion, 2_400, summarize=False)
            session._flush_capture_apex_bucket()

            frame = session.recent_frame_items()[-1]
            selection = frame["capture_selection"]
            self.assertEqual(selection["selection_mode"], "burst")
            self.assertEqual(selection["selection_source"], "capture_cv_frame_delta")
            self.assertEqual(selection["selected_timestamp_ms"], 2_100)
            self.assertGreater(selection.get("activity_x") or 0.0, 10.0)
            self.assertFalse(selection["baseline"]["warmup"])
            companion = frame.get("burst_companion")
            self.assertIsNotNone(companion)
            self.assertEqual(companion["timestamp_ms"], 2_400)
            self.assertEqual(companion["thumbnail"], "jpeg")
            self.assertEqual(companion["role"], "burst_sharp_companion")
            self.assertEqual(selection["companion"]["timestamp_ms"], 2_400)
            status = session.status()
            self.assertEqual(status["capture_apex_companion_count"], 1)
            self.assertEqual(status["capture_apex_mode_counts"].get("burst"), 1)

    def test_clarity_bias_prefers_sharpest_regardless_of_motion(self):
        with tempfile.TemporaryDirectory() as temp:
            manager, session = self._make_session(temp)
            manager.update_prompt_settings(channel_id=7, capture_selector_bias="clarity")
            session._accept_captured_frame(_deterministic_noise_frame(1), 1_000, summarize=False)
            blurry_peak = _deterministic_noise_frame(2, blur_radius=6.0)
            sharp_active = Image.blend(blurry_peak, _deterministic_noise_frame(3), 0.75)
            session._accept_captured_frame(blurry_peak, 2_100, summarize=False)
            session._accept_captured_frame(sharp_active, 2_500, summarize=False)
            session._flush_capture_apex_bucket()

            selection = session.recent_frame_items()[-1]["capture_selection"]
            self.assertEqual(selection["selection_mode"], "quiet")
            self.assertEqual(selection["selection_source"], "capture_cv_sharpest")
            self.assertEqual(selection["selected_timestamp_ms"], 2_500)
            self.assertEqual(selection["selector_bias"], "clarity")

    def test_disabled_selector_uses_midpoint_and_freezes_homeostasis(self):
        with tempfile.TemporaryDirectory() as temp:
            manager, session = self._make_session(temp)
            manager.update_prompt_settings(channel_id=7, capture_selector_enabled=False)
            session.capture_activity_baseline_level = 0.0125
            session.capture_activity_baseline_dev = 0.003
            session.capture_activity_baseline_buckets = 640

            session._accept_captured_frame(_deterministic_noise_frame(1), 2_100, summarize=False)
            session._accept_captured_frame(_deterministic_noise_frame(2), 2_500, summarize=False)
            session._accept_captured_frame(_deterministic_noise_frame(3), 2_900, summarize=False)
            session._flush_capture_apex_bucket()

            selection = session.recent_frame_items()[-1]["capture_selection"]
            self.assertFalse(selection["selector_enabled"])
            self.assertEqual(selection["selection_mode"], "disabled")
            self.assertEqual(selection["selection_source"], "selector_disabled_temporal_midpoint")
            self.assertEqual(selection["selected_timestamp_ms"], 2_500)
            self.assertEqual(selection["fallback_reason"], "capture_cv_selector_disabled")
            self.assertEqual(session.capture_activity_baseline_buckets, 640)
            self.assertEqual(session.status()["capture_selector_enabled"], False)

    def test_bucket_mode_is_relative_to_channel_baseline(self):
        with tempfile.TemporaryDirectory() as temp:
            _manager, session = self._make_session(temp)
            statue = {"level": 0.0005, "dev": 0.0001, "buckets": 500, "warmup": False}
            intersection = {"level": 0.05, "dev": 0.02, "buckets": 500, "warmup": False}
            warmup = {"level": 0.0005, "dev": 0.0001, "buckets": 10, "warmup": True}
            self.assertEqual(session._classify_capture_bucket_mode(0.02, statue, "auto"), "burst")
            self.assertEqual(session._classify_capture_bucket_mode(0.02, intersection, "auto"), "normal")
            self.assertEqual(session._classify_capture_bucket_mode(0.003, statue, "auto"), "quiet")
            self.assertEqual(session._classify_capture_bucket_mode(0.5, warmup, "auto"), "normal")
            self.assertEqual(session._classify_capture_bucket_mode(0.02, intersection, "action"), "burst")
            self.assertEqual(session._classify_capture_bucket_mode(0.5, intersection, "clarity"), "quiet")

    def test_activity_baseline_resists_burst_contamination(self):
        with tempfile.TemporaryDirectory() as temp:
            _manager, session = self._make_session(temp)
            for _ in range(200):
                session._update_capture_activity_baseline_locked(0.01)
            level_before = float(session.capture_activity_baseline_level)
            session._update_capture_activity_baseline_locked(0.9)
            self.assertLess(float(session.capture_activity_baseline_level), level_before * 1.5)
            self.assertEqual(session.capture_activity_baseline_buckets, 201)

    def test_channel_baseline_persists_and_seeds_new_sessions(self):
        with tempfile.TemporaryDirectory() as temp:
            store = MemoryRuntimeStateStore()
            manager = build_manager(Path(temp), runtime_state_store=store)
            manager.note_capture_baseline(7, {"level": 0.0125, "dev": 0.003, "buckets": 640})
            self.assertTrue(manager.persist_summary_state())

            reloaded = build_manager(Path(temp), runtime_state_store=store)
            restored = reloaded.get_persisted_capture_baseline(7)
            self.assertIsNotNone(restored)
            self.assertAlmostEqual(restored["level"], 0.0125)
            self.assertEqual(restored["buckets"], 640)

            session = LuxriotCaptureSession(
                reloaded,
                channel_id=7,
                batch_size=12,
                prompt="Describe activity.",
                run_id="run-7",
                interval_override=0.2,
            )
            self.assertAlmostEqual(float(session.capture_activity_baseline_level), 0.0125)
            self.assertEqual(session.capture_activity_baseline_buckets, 640)
            self.assertFalse(session.status()["capture_activity_baseline"]["warmup"])

    def test_postgres_history_persistence_does_not_hold_runtime_cache_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            store = BlockingPostgresRuntimeStateStore()
            manager = build_manager(Path(temp), runtime_state_store=store)
            started = time.monotonic()
            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": "Vehicle moved through the junction.",
                    "frame_count": 2,
                    "created_at": 1_781_700_000.0,
                    "frame_selection": {
                        "groups": [
                            {
                                "selected_source_frame_index": 2,
                                "selection_source": "road_cv_cue",
                                "apex_available": True,
                                "source_frame_indices": list(range(1, 25)),
                                "source_timestamps_ms": list(range(100_000, 124_000, 1000)),
                                "source_frame_hashes": ["a" * 40] * 24,
                            }
                        ]
                    },
                    "vector_signal": {
                        "channel_id": 7,
                        "road_cv_cues": [{"cue_type": "motion", "score": 0.8}],
                        "road_cv_frame_scores": [
                            {"source_frame_index": index, "timestamp_ms": 100_000 + index * 1000, "attention_score": 0.5}
                            for index in range(1, 25)
                        ],
                    },
                },
            )
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertTrue(store.save_started.wait(timeout=1.0))

            settings_started = time.monotonic()
            settings = manager.get_prompt_settings(channel_id=7)
            self.assertEqual(settings["channel_id"], 7)
            self.assertLess(time.monotonic() - settings_started, 0.2)

            stored = manager.summary_history[7][0]
            self.assertNotIn("road_cv_frame_scores", stored["vector_signal"])
            self.assertNotIn("source_frame_indices", stored["frame_selection"]["groups"][0])
            self.assertEqual(stored["frame_selection"]["groups"][0]["selection_source"], "road_cv_cue")

            store.release_save.set()
            deadline = time.monotonic() + 2.0
            while manager.summary_state_revision < 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(manager.summary_state_revision, 1)

    def test_new_prompt_revision_cannot_be_overwritten_by_older_async_history(self):
        class ReorderingStore(MemoryRuntimeStateStore):
            backend = "postgres"

            def __init__(self):
                super().__init__()
                self.first_started = threading.Event()
                self.release_first = threading.Event()
                self.calls = 0

            def save_state(self, key, payload):
                self.calls += 1
                if self.calls == 1:
                    self.first_started.set()
                    self.assert_released = self.release_first.wait(timeout=2.0)
                super().save_state(key, payload)

        with tempfile.TemporaryDirectory() as temp:
            store = ReorderingStore()
            manager = build_manager(Path(temp), runtime_state_store=store)
            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "batch_id": "batch-old",
                    "summary": "Routine room.",
                    "frame_count": 1,
                    "created_at": 1_781_700_000.0,
                },
            )
            self.assertTrue(store.first_started.wait(timeout=1.0))

            completed = threading.Event()

            def save_prompt():
                manager.update_prompt_settings(
                    channel_id=7,
                    alert_policy_prompt="Alert when a person falls.",
                )
                completed.set()

            writer = threading.Thread(target=save_prompt)
            writer.start()
            # Operator control-plane state no longer queues behind a running
            # L0 history write.
            self.assertTrue(completed.wait(timeout=1.0))
            store.release_first.set()
            writer.join(timeout=2.0)
            self.assertTrue(completed.is_set())
            deadline = time.monotonic() + 1.0
            while not getattr(store, "assert_released", False) and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(getattr(store, "assert_released", False))
            persisted = store.load_state(manager.PROMPT_SETTINGS_STATE_KEY)
            self.assertEqual(
                persisted["prompt_settings"]["channel_overrides"]["7"][
                    "alert_policy_prompt"
                ],
                "Alert when a person falls.",
            )

    def test_l0_frame_budget_keeps_endpoints_apex_and_temporal_coverage(self):
        frames = [
            {
                "timestamp_ms": 100_000 + index * 1_000,
                "selection_score": 9.0 if index == 5 else 0.1,
            }
            for index in range(8)
        ]
        selection = {
            "version": 1,
            "policy": "per_second_attention_apex_v1",
            "source_frame_count": 8,
            "selected_frame_count": 8,
            "groups": [
                {
                    "selected_timestamp_ms": frame["timestamp_ms"],
                    "selected_source_frame_index": index + 1,
                    "selection_source": "capture_cv_apex",
                    "apex_available": True,
                    "source_frame_indices": [index + 1],
                }
                for index, frame in enumerate(frames)
            ],
        }

        limited, metadata = LuxriotManager._limit_attention_frames(
            frames,
            selection,
            4,
        )

        timestamps = [frame["timestamp_ms"] for frame in limited]
        self.assertEqual(len(limited), 4)
        self.assertEqual(timestamps[0], 100_000)
        self.assertEqual(timestamps[-1], 107_000)
        self.assertIn(105_000, timestamps)
        self.assertEqual(metadata["pre_budget_selected_frame_count"], 8)
        self.assertEqual(metadata["selected_frame_count"], 4)
        self.assertEqual(metadata["omitted_selected_frames"], 4)

    def test_postgres_history_persistence_writes_only_new_l0_items(self):
        class CapturingPostgresRuntimeStateStore(MemoryRuntimeStateStore):
            backend = "postgres"

            def __init__(self):
                super().__init__()
                self.saved = []

            def save_state(self, key, payload):
                self.saved.append((key, dict(payload)))
                super().save_state(key, payload)

        with tempfile.TemporaryDirectory() as temp:
            store = CapturingPostgresRuntimeStateStore()
            manager = build_manager(Path(temp), runtime_state_store=store)
            manager.summary_history[7] = [
                {
                    "channel_id": 7,
                    "run_id": "old-run",
                    "batch_id": f"old-{index}",
                    "summary": f"Historical summary {index}",
                    "frame_count": 1,
                    "created_at": 1_781_600_000.0 + index,
                }
                for index in range(50)
            ]

            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "batch_id": "new-1",
                    "summary": "First new summary.",
                    "frame_count": 2,
                    "created_at": 1_781_700_000.0,
                },
            )
            deadline = time.monotonic() + 2.0
            while manager.summary_state_revision < 1 and time.monotonic() < deadline:
                time.sleep(0.01)

            self.assertEqual(len(manager.summary_history[7]), 51)
            first_payload = store.saved[-1][1]
            self.assertEqual(first_payload["summary_history_storage"], "upsert_items")
            self.assertEqual(len(first_payload["summary_history"]["7"]), 1)
            self.assertEqual(
                first_payload["summary_history"]["7"][0]["batch_id"],
                "new-1",
            )

            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "batch_id": "new-2",
                    "summary": "Second new summary.",
                    "frame_count": 2,
                    "created_at": 1_781_700_001.0,
                },
            )
            self.assertTrue(manager.persist_summary_state())
            second_payload = store.saved[-1][1]
            self.assertEqual(len(second_payload["summary_history"]["7"]), 1)
            self.assertEqual(
                second_payload["summary_history"]["7"][0]["batch_id"],
                "new-2",
            )

    def test_manager_cache_lock_allows_layered_runtime_reads(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            completed = threading.Event()

            def layered_read():
                with manager.cache_lock:
                    manager.get_prompt_settings(channel_id=7)
                completed.set()

            worker = threading.Thread(target=layered_read, daemon=True)
            worker.start()
            self.assertTrue(completed.wait(timeout=0.5))

    def test_compact_summary_feed_omits_internal_frame_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": "Sudden movement near the gate.",
                    "frame_count": 12,
                    "created_at": 1_781_700_000.0,
                    "thumbnail_detection_id": 44,
                    "thumbnail_role": "burst_apex",
                    "thumbnail_frame_index": 2,
                    "thumbnail_selection_source": "capture_cv_frame_delta",
                    "thumbnail_is_cover": True,
                    "thumbnail_snapshot_index": 3,
                    "cover_kind": "event",
                    "cover_reason": "Burst apex represents the episode.",
                    "cover_confidence": "high",
                    "batch_id": "batch-7",
                    "state_transition_total": 2,
                    "prompt": "large internal prompt",
                    "frame_selection": {
                        "groups": [
                            {
                                "selected_source_frame_index": 2,
                                "selection_source": "capture_cv_apex",
                                "apex_available": True,
                            }
                        ]
                    },
                    "vector_signal": {
                        "channel_id": 7,
                        "road_cv_frame_scores": [
                            {"source_frame_index": 2, "timestamp_ms": 1_781_700_000_000, "attention_score": 0.8}
                        ],
                        "capture_attention": {
                            "policy": "per_second_cv_apex_v2",
                            "seconds": [{"snapshot": 2, "mode": "burst", "activity_x": 8.0}],
                        },
                    },
                },
            )

            full_log = manager.session_status(7, run_selector="all")["logs"][0]
            feed_log = manager.session_status(7, run_selector="all", compact_feed=True)["logs"][0]
            self.assertIn("frame_selection", full_log)
            self.assertNotIn("frame_selection", feed_log)
            self.assertNotIn("prompt", feed_log)
            self.assertEqual(feed_log["thumbnail_detection_id"], 44)
            self.assertEqual(feed_log["thumbnail_role"], "burst_apex")
            self.assertEqual(feed_log["thumbnail_frame_index"], 2)
            self.assertTrue(feed_log["thumbnail_is_cover"])
            self.assertEqual(feed_log["thumbnail_snapshot_index"], 3)
            self.assertEqual(feed_log["cover_kind"], "event")
            self.assertEqual(feed_log["cover_reason"], "Burst apex represents the episode.")
            self.assertEqual(feed_log["cover_confidence"], "high")
            self.assertEqual(feed_log["batch_id"], "batch-7")
            self.assertEqual(feed_log["state_transition_total"], 2)
            self.assertEqual(
                feed_log["thumbnail_selection_source"],
                "capture_cv_frame_delta",
            )
            self.assertEqual(
                feed_log["vector_signal"]["capture_attention"]["seconds"][0]["mode"],
                "burst",
            )

    def test_scheduled_rollup_scans_bounded_recent_history(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.rollup_scheduler_backfill_windows = 3
            calls = []
            with patch.object(
                manager,
                "summary_rollups",
                side_effect=lambda **kwargs: calls.append(kwargs) or {"levels": {}},
            ):
                manager._run_scheduled_rollup(7, "L1", 3_701.0)

            self.assertEqual(calls[0]["channel_id"], 7)
            self.assertEqual(calls[0]["target_level"], "L1")
            self.assertEqual(calls[0]["start_ts"], 0.0)
            self.assertEqual(calls[0]["end_ts"], 3_599.999)
            self.assertIsNone(calls[0]["level_limit"])
            self.assertEqual(calls[0]["synthesize_levels"], {"L1"})
            self.assertEqual(calls[0]["max_new_per_level"], 3)

    def test_scheduled_rollup_lookback_is_bounded_on_long_running_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.rollup_windows["L1"] = 900
            manager.rollup_scheduler_backfill_windows = 2
            calls = []
            with patch.object(
                manager,
                "summary_rollups",
                side_effect=lambda **kwargs: calls.append(kwargs) or {"levels": {}},
            ):
                manager._run_scheduled_rollup(7, "L1", 100_001.0)

            closed_end = int(100_001.0 // 900) * 900
            self.assertEqual(calls[0]["start_ts"], float(closed_end - 2_700))
            self.assertEqual(calls[0]["end_ts"], float(closed_end) - 0.001)

    def test_scheduled_rollup_reopens_recent_failed_semantic_window(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.rollup_windows["L1"] = 900
            manager.rollup_scheduler_backfill_windows = 2
            manager._put_cached_rollup_summary(
                "l1-ch7-w900-90000",
                "Semantic L1 aggregation failed.",
                channel_id=7,
                level="L1",
                source_level="L0",
                window_start=90_000.0,
                window_end=90_900.0,
                window_sec=900,
                source_signature="failed-source",
                summary_kind="degraded",
                generation_status="failed",
                generation_error="endpoint unavailable",
                format_version=2,
            )
            calls = []
            with patch.object(
                manager,
                "summary_rollups",
                side_effect=lambda **kwargs: calls.append(kwargs) or {"levels": {}},
            ):
                manager._run_scheduled_rollup(7, "L1", 100_001.0)

            self.assertEqual(calls[0]["start_ts"], 90_000.0)

    def test_l0_backpressure_ignores_normal_inflight_and_requires_saturated_queue(self):
        class StatusSession:
            def __init__(self, status):
                self._status = status

            def status(self):
                return dict(self._status)

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.sessions[7] = StatusSession(
                {
                    "summary_inflight": True,
                    "summary_queue_depth": 1,
                    "summary_queue_max_batches": 2,
                }
            )
            self.assertFalse(manager._l0_backpressure_active(7))

            manager.sessions[7] = StatusSession(
                {
                    "summary_inflight": True,
                    "summary_queue_depth": 2,
                    "summary_queue_max_batches": 2,
                }
            )
            self.assertTrue(manager._l0_backpressure_active(7))

    def test_global_l0_backpressure_detects_another_saturated_live_channel(self):
        class StatusSession:
            def __init__(self, status):
                self._status = status

            def status(self):
                return dict(self._status)

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.sessions[7] = StatusSession(
                {
                    "summary_queue_depth": 0,
                    "summary_queue_max_batches": 2,
                }
            )
            manager.sessions[8] = StatusSession(
                {
                    "summary_queue_depth": 2,
                    "summary_queue_max_batches": 2,
                }
            )

            self.assertFalse(manager._l0_backpressure_active(7))
            self.assertTrue(manager._l0_backpressure_active())

    def test_l0_bootstrap_gate_waits_for_first_live_summary(self):
        class StatusSession:
            def __init__(self, status):
                self._status = status

            def status(self):
                return dict(self._status)

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.sessions[7] = StatusSession(
                {
                    "running": True,
                    "summarization_enabled": True,
                    "run_started_at": 1_000.0,
                    "summary_max_window_sec": 60.0,
                    "summary_last_success_at": None,
                }
            )

            self.assertTrue(manager._l0_bootstrap_pending(now=1_080.0))
            manager.sessions[7]._status["summary_last_success_at"] = 1_081.0
            self.assertFalse(manager._l0_bootstrap_pending(now=1_082.0))

    def test_l0_bootstrap_gate_expires_for_broken_stream(self):
        class StatusSession:
            def status(self):
                return {
                    "running": True,
                    "summarization_enabled": True,
                    "run_started_at": 1_000.0,
                    "summary_max_window_sec": 60.0,
                    "summary_last_success_at": None,
                }

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.sessions[7] = StatusSession()

            self.assertFalse(manager._l0_bootstrap_pending(now=1_120.0))

    def test_l0_backpressure_only_blocks_rollups_on_the_same_lm_resource(self):
        class StatusSession:
            def status(self):
                return {
                    "model": "vlm",
                    "summary_queue_depth": 2,
                    "summary_queue_max_batches": 2,
                }

        def lm_callback(_messages, _model):
            return operator_rollup_response("Routine window.")

        lm_callback.eva_resource_key = lambda selector: (
            "http://agent.local/v1"
            if str(selector or "").strip() == "agent"
            else "http://vlm.local/v1"
        )

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp), lm_callback=lm_callback)
            manager.sessions[7] = StatusSession()

            self.assertTrue(manager._l0_backpressure_active(model_hint="vlm"))
            self.assertFalse(manager._l0_backpressure_active(model_hint="agent"))

    def test_rollup_backpressure_deferral_has_a_hard_window_ceiling(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.rollup_windows["L1"] = 900
            manager.rollup_scheduler_max_deferral_windows = 2.0
            manager.rollup_scheduler_max_deferral_sec = 180.0
            manager._rollup_scheduler_deferred_since[(7, "L1")] = 1_000.0

            self.assertFalse(manager._rollup_deferral_exhausted((7, "L1"), "L1", 1_179.9))
            self.assertTrue(manager._rollup_deferral_exhausted((7, "L1"), "L1", 1_180.0))

            manager.rollup_scheduler_max_deferral_sec = 0.0
            self.assertFalse(manager._rollup_deferral_exhausted((7, "L1"), "L1", 2_799.9))
            self.assertTrue(manager._rollup_deferral_exhausted((7, "L1"), "L1", 2_800.0))
            manager.rollup_scheduler_max_deferral_windows = 0.0
            self.assertTrue(manager._rollup_deferral_exhausted((8, "L1"), "L1", 1_000.0))

    def test_l1_source_watermark_waits_for_an_accepted_batch_past_boundary(self):
        class StatusSession:
            def __init__(self, logs):
                self.logs = logs

            def status(self):
                return {
                    "running": True,
                    "summarization_enabled": True,
                    "logs": list(self.logs),
                }

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.rollup_windows["L1"] = 900
            session = StatusSession(
                [{"batch_end_ms": 1_799_500}]
            )
            manager.sessions[7] = session

            ready, watermark_ms, boundary = manager._rollup_l0_source_watermark_ready(
                7,
                "L1",
                now=1_831.0,
            )

            self.assertFalse(ready)
            self.assertEqual(watermark_ms, 1_799_500)
            self.assertEqual(boundary, 1_800)

            session.logs.append({"batch_end_ms": 1_801_000})
            ready, watermark_ms, boundary = manager._rollup_l0_source_watermark_ready(
                7,
                "L1",
                now=1_841.0,
            )

            self.assertTrue(ready)
            self.assertEqual(watermark_ms, 1_801_000)
            self.assertEqual(boundary, 1_800)

    def test_l1_source_watermark_does_not_block_stopped_or_higher_level_work(self):
        class StoppedSession:
            def status(self):
                return {
                    "running": False,
                    "summarization_enabled": True,
                    "logs": [],
                }

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.sessions[7] = StoppedSession()

            self.assertTrue(
                manager._rollup_l0_source_watermark_ready(
                    7,
                    "L1",
                    now=1_831.0,
                )[0]
            )

    def test_l1_source_watermark_closes_a_stale_partial_window(self):
        class StaleSession:
            def status(self):
                return {
                    "running": True,
                    "summarization_enabled": True,
                    "summary_inflight": False,
                    "summary_queue_depth": 0,
                    "pending_frames": 0,
                    "summary_max_window_sec": 60.0,
                    "last_snapshot_at": 1_700.0,
                    "logs": [{"batch_end_ms": 1_790_000}],
                }

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.rollup_windows["L1"] = 900
            manager.sessions[7] = StaleSession()

            ready, watermark_ms, boundary = manager._rollup_l0_source_watermark_ready(
                7,
                "L1",
                now=1_831.0,
            )

            self.assertTrue(ready)
            self.assertEqual(watermark_ms, 1_790_000)
            self.assertEqual(boundary, 1_800)
            self.assertTrue(
                manager._rollup_l0_source_watermark_ready(
                    7,
                    "L2",
                    now=7_321.0,
                )[0]
            )

    def test_rollup_scheduler_staggers_channels_deterministically(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.rollup_scheduler_initial_delay_sec = 30.0
            manager.rollup_scheduler_spacing_sec = 5.0
            due = [
                manager._rollup_initial_due(channel_id, "L1", 1_000.0, 50)
                for channel_id in range(1, 51)
            ]

            self.assertGreater(len(set(due)), 20)
            self.assertGreaterEqual(min(due), 1_830.0)
            self.assertLess(max(due), 2_080.0)

    def test_rollup_scheduler_initial_due_uses_next_level_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.rollup_scheduler_initial_delay_sec = 30.0
            manager.rollup_scheduler_spacing_sec = 5.0

            l1_due = manager._rollup_initial_due(7, "L1", 10_000.0, 3)
            l2_due = manager._rollup_initial_due(7, "L2", 10_000.0, 3)

            self.assertGreaterEqual(l1_due, 10_830.0)
            self.assertLess(l1_due, 10_890.0)
            self.assertGreaterEqual(l2_due, 10_920.0)
            self.assertLess(l2_due, 10_980.0)

    def test_rollup_scheduler_ignores_history_only_channels(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.summary_history[41] = [{"summary": "old stopped stream"}]
            manager.sessions[42] = object()
            with patch.object(
                manager,
                "_load_desired_live_sessions",
                return_value={
                    43: {"enabled": True},
                    44: {"enabled": False},
                },
            ):
                channels = manager._rollup_scheduler_channels()

            self.assertEqual(channels, [42, 43])

    def test_rollup_scheduler_recurring_due_aligns_to_window_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.rollup_windows["L1"] = 900
            manager.rollup_scheduler_spacing_sec = 5.0

            due = manager._rollup_next_due(
                7,
                "L1",
                started_at=10_000.0,
                completed_at=10_050.0,
                channel_count=3,
            )

            self.assertGreaterEqual(due, 10_830.0)
            self.assertLess(due, 10_890.0)

    def test_rollup_scheduler_catches_boundary_crossed_during_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.rollup_windows["L1"] = 900
            manager.rollup_scheduler_spacing_sec = 5.0

            due = manager._rollup_next_due(
                7,
                "L1",
                started_at=10_790.0,
                completed_at=10_850.0,
                channel_count=3,
            )

            self.assertGreaterEqual(due, 10_850.0)
            self.assertLess(due, 10_890.0)

    def test_rollup_scheduler_prioritizes_imminent_l1_over_ready_l2(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.rollup_scheduler_spacing_sec = 5.0

            selected = manager._select_rollup_scheduler_due_item(
                [
                    (1_000.0, (7, "L2")),
                    (1_021.0, (8, "L1")),
                    (1_005.0, (9, "L3")),
                ],
                now=1_010.0,
            )

            self.assertEqual(selected, (1_021.0, (8, "L1")))

    def test_rollup_scheduler_prioritizes_ready_levels_without_starving_future_l1(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.rollup_scheduler_spacing_sec = 5.0

            selected = manager._select_rollup_scheduler_due_item(
                [
                    (1_000.0, (7, "L2")),
                    (1_080.0, (8, "L1")),
                    (1_005.0, (9, "L3")),
                ],
                now=1_010.0,
            )

            self.assertEqual(selected, (1_000.0, (7, "L2")))

    def test_selector_bias_is_a_channel_setting_with_reset(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            settings = manager.update_prompt_settings(channel_id=7, capture_selector_bias="action")
            self.assertEqual(settings["capture_selector_bias"], "action")
            self.assertIn("capture_selector_bias", settings["override_fields"])
            self.assertEqual(settings["setting_sources"]["capture_selector_bias"], "channel_override")
            self.assertEqual(manager.get_capture_selector_bias(7), "action")
            self.assertEqual(manager.get_capture_selector_bias(8), "auto")

            reset = manager.update_prompt_settings(
                channel_id=7,
                clear_override_fields=["capture_selector_bias"],
            )
            self.assertEqual(reset["capture_selector_bias"], "auto")
            self.assertNotIn("capture_selector_bias", reset["override_fields"])

            with self.assertRaises(ValueError):
                manager.update_prompt_settings(channel_id=7, capture_selector_bias="fastest")

    def test_selector_enabled_is_a_persisted_channel_setting_with_reset(self):
        with tempfile.TemporaryDirectory() as temp:
            store = MemoryRuntimeStateStore()
            manager = build_manager(Path(temp), runtime_state_store=store)
            settings = manager.update_prompt_settings(
                channel_id=7,
                capture_selector_enabled=False,
            )
            self.assertFalse(settings["capture_selector_enabled"])
            self.assertIn("capture_selector_enabled", settings["override_fields"])
            self.assertEqual(
                settings["setting_sources"]["capture_selector_enabled"],
                "channel_override",
            )
            self.assertFalse(manager.get_capture_selector_enabled(7))
            self.assertTrue(manager.get_capture_selector_enabled(8))

            reloaded = build_manager(Path(temp), runtime_state_store=store)
            self.assertFalse(reloaded.get_capture_selector_enabled(7))
            reset = reloaded.update_prompt_settings(
                channel_id=7,
                clear_override_fields=["capture_selector_enabled"],
            )
            self.assertTrue(reset["capture_selector_enabled"])
            self.assertNotIn("capture_selector_enabled", reset["override_fields"])


def _burst_batch_frame(
    *,
    timestamp_ms: int,
    thumbnail: str = "frame-b64",
    mode: str = "burst",
    activity_x: float = 12.4,
    with_companion: bool = True,
) -> Dict[str, Any]:
    selection: Dict[str, Any] = {
        "policy": "capture_per_second_cv_apex_v2",
        "selection_mode": mode,
        "activity_x": activity_x,
        "activity_peak": 0.31,
        "baseline": {"level": 0.002, "dev": 0.0004, "buckets": 500, "warmup": False},
        "selected_timestamp_ms": int(timestamp_ms),
        "selected_source_frame_index": 1,
        "selection_source": "capture_cv_frame_delta",
        "apex_available": True,
    }
    frame: Dict[str, Any] = {
        "thumbnail": thumbnail,
        "captured_at": timestamp_ms / 1000.0,
        "time_sec": timestamp_ms / 1000.0,
        "timestamp_ms": int(timestamp_ms),
        "width": 320,
        "height": 180,
        "frame_hash": f"hash-{timestamp_ms}",
        "capture_selection": selection,
    }
    if with_companion:
        companion = {
            "role": "burst_sharp_companion",
            "thumbnail": f"companion-{timestamp_ms}",
            "timestamp_ms": int(timestamp_ms) + 400,
            "source_frame_index": 3,
            "frame_hash": f"companion-hash-{timestamp_ms}",
            "sharpness": 912.5,
            "activity": 0.21,
        }
        frame["burst_companion"] = companion
        selection["companion"] = {
            key: companion[key]
            for key in ("timestamp_ms", "source_frame_index", "frame_hash", "sharpness", "activity")
        }
    return frame


class LuxriotSummaryBackpressureTests(unittest.TestCase):
    def _make_session(self, temp: str, queue_max: int) -> Tuple[LuxriotManager, LuxriotCaptureSession]:
        manager = build_manager(
            Path(temp),
            config_overrides={"LUXRIOT_SUMMARY_QUEUE_MAX_BATCHES": queue_max},
        )
        session = LuxriotCaptureSession(
            manager,
            channel_id=7,
            batch_size=4,
            prompt="Describe activity.",
            run_id="run-7",
            interval_override=0.2,
        )
        return manager, session

    @staticmethod
    def _frames(start_sec: float, count: int, *, mode: str = "normal", activity_x: float = 1.0):
        return [
            {
                "captured_at": start_sec + index,
                "time_sec": start_sec + index,
                "thumbnail": f"jpeg-{start_sec + index:g}",
                "frame_hash": f"hash-{start_sec + index:g}",
                "capture_selection": {
                    "selection_mode": mode,
                    "activity_x": activity_x,
                },
            }
            for index in range(count)
        ]

    def test_backpressure_coalesces_windows_instead_of_dropping(self):
        with tempfile.TemporaryDirectory() as temp:
            _manager, session = self._make_session(temp, queue_max=2)
            started = threading.Event()
            release = threading.Event()

            def slow_dispatcher(_batch, _workload):
                started.set()
                release.wait(timeout=5.0)
                return {"queued": False, "accepted": True}

            session.manager.set_summary_dispatcher(slow_dispatcher)
            session.summary_worker_thread.start()
            try:
                # First batch goes inflight into the blocked dispatcher. Every
                # later window is folded into one fresh representative batch
                # instead of forming a stale FIFO behind it.
                session.frames = self._frames(100.0, 4)
                self.assertTrue(session._enqueue_summary_batch())
                self.assertTrue(started.wait(timeout=2.0))
                session.frames = self._frames(112.0, 4)
                self.assertTrue(session._enqueue_summary_batch())
                session.frames = self._frames(124.0, 4)
                self.assertTrue(session._enqueue_summary_batch())
                session.frames = self._frames(136.0, 4, mode="burst", activity_x=9.0)
                self.assertTrue(session._enqueue_summary_batch())

                with session.lock:
                    queue_snapshot = [
                        (list(frames), dict(meta))
                        for frames, _workload, meta in session.summary_queue
                    ]
                self.assertEqual(len(queue_snapshot), 1)
                self.assertEqual(session.summary_coalesced_batches, 2)
                self.assertEqual(session.queue_dropped_batches, 0)
                merged_frames, merged_meta = queue_snapshot[0]
                self.assertEqual(merged_meta["coalesced"]["batches"], 3)
                self.assertEqual(merged_meta["coalesced"]["omitted_frames"], 8)
                self.assertEqual(len(merged_frames), 4)
                spans = [frame["captured_at"] for frame in merged_frames]
                self.assertLess(min(spans), 116.0)
                self.assertGreater(max(spans), 135.0)
                self.assertEqual(session.status()["summary_coalesced_batches"], 2)
            finally:
                release.set()
                session.stop_event.set()
                with session.summary_condition:
                    session.summary_condition.notify_all()

    def test_coalescing_preserves_burst_frames(self):
        combined = (
            self._frames(100.0, 6, mode="quiet", activity_x=0.0)
            + self._frames(110.0, 3, mode="burst", activity_x=8.0)
            + self._frames(120.0, 6, mode="normal", activity_x=2.0)
        )
        kept, omitted = LuxriotCaptureSession._subsample_coalesced_frames(combined, 6)
        self.assertEqual(len(kept), 6)
        self.assertEqual(omitted, 9)
        kept_modes = [frame["capture_selection"]["selection_mode"] for frame in kept]
        self.assertEqual(kept_modes.count("burst"), 3)
        stamps = [frame["captured_at"] for frame in kept]
        self.assertEqual(stamps, sorted(stamps))
        self.assertEqual(stamps[0], 100.0)
        self.assertEqual(stamps[-1], 125.0)

    def test_single_frame_coalescing_keeps_the_newest_observation(self):
        kept, omitted = LuxriotCaptureSession._subsample_coalesced_frames(
            self._frames(100.0, 8),
            1,
        )

        self.assertEqual(omitted, 7)
        self.assertEqual([frame["captured_at"] for frame in kept], [107.0])

    def test_coalescing_preserves_the_strongest_workload_lane(self):
        with tempfile.TemporaryDirectory() as temp:
            _manager, session = self._make_session(temp, queue_max=2)
            for first, second in (("heartbeat", "event"), ("alert", "heartbeat")):
                session.summary_queue = [
                    (self._frames(100.0, 4), first, {"batch_size": 4}),
                    (self._frames(112.0, 4), second, {"batch_size": 4}),
                ]
                self.assertTrue(session._coalesce_oldest_queued_batches_locked())
                self.assertEqual(
                    session.summary_queue[0][1],
                    max(
                        (first, second),
                        key=session._summary_workload_priority,
                    ),
                )

    def test_exhausted_coalescing_leaves_an_explicit_coverage_gap(self):
        with tempfile.TemporaryDirectory() as temp:
            manager, session = self._make_session(temp, queue_max=2)
            started = threading.Event()
            release = threading.Event()

            def slow_dispatcher(_batch, _workload):
                started.set()
                release.wait(timeout=5.0)
                return {"queued": False, "accepted": True}

            manager.set_summary_dispatcher(slow_dispatcher)
            session.summary_worker_thread.start()
            try:
                # Batch 1 goes inflight; batches 2 and 3 become one pending
                # representative batch.
                session.frames = self._frames(100.0, 4)
                self.assertTrue(session._enqueue_summary_batch())
                self.assertTrue(started.wait(timeout=2.0))
                session.frames = self._frames(112.0, 4)
                self.assertTrue(session._enqueue_summary_batch())
                session.frames = self._frames(124.0, 4)
                self.assertTrue(session._enqueue_summary_batch())
                # Poison the pending batch as already fully coalesced so the
                # next window replaces it and records an explicit gap.
                with session.lock:
                    poisoned = []
                    for frames, workload, meta in session.summary_queue:
                        meta = dict(meta)
                        meta["coalesced"] = {"batches": _capture_max_coalesce(), "omitted_frames": 0}
                        poisoned.append((frames, workload, meta))
                    session.summary_queue[:] = poisoned
                session.frames = self._frames(136.0, 4)
                self.assertTrue(session._enqueue_summary_batch())

                self.assertEqual(session.queue_dropped_batches, 1)
                logs = manager.summary_history.get(7) or []
                gap_logs = [log for log in logs if log.get("coverage_gap")]
                self.assertEqual(len(gap_logs), 1)
                gap = gap_logs[0]
                self.assertEqual(gap["gap_reason"], "lm_backpressure_dropped_batch")
                self.assertIn("coverage gap", gap["summary"])
                self.assertEqual(gap["batch_start_ms"], 112_000)
                self.assertEqual(gap["batch_end_ms"], 127_000)
                self.assertEqual(gap["frame_count"], 8)
            finally:
                release.set()
                session.stop_event.set()
                with session.summary_condition:
                    session.summary_condition.notify_all()

    def test_coalesced_info_reaches_the_summary_entry(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp), lm_callback=lambda _messages, _hint: "All calm.")
            batch = manager.create_summary_batch(
                channel_id=7,
                run_id="run-7",
                batch_size=4,
                prompt="Describe.",
                model_hint=None,
                interval_sec=1.0,
                frames=self._frames(100.0, 4),
            )
            batch["coalesced"] = {"batches": 2, "omitted_frames": 4}
            entry = manager.run_summary_batch(batch)
            self.assertEqual(entry["coalesced"], {"batches": 2, "omitted_frames": 4})
            normalized = manager._normalize_summary_log_entry(entry)
            self.assertEqual(normalized["coalesced"], {"batches": 2, "omitted_frames": 4})

    def test_lm_response_timing_and_usage_reach_persisted_summary(self):
        class TimedSummary(str):
            def __new__(cls, value):
                result = super().__new__(cls, value)
                result.eva_response_meta = {
                    "attempt_count": 1,
                    "retried": False,
                    "finish_reason": "stop",
                    "admission_wait_ms": 2.5,
                    "http_ms": 91.25,
                    "prompt_tokens": 2300,
                    "completion_tokens": 128,
                    "total_tokens": 2428,
                    "endpoint": "must-not-persist",
                }
                return result

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                lm_callback=lambda _messages, _hint: TimedSummary("All calm."),
            )
            batch = manager.create_summary_batch(
                channel_id=7,
                run_id="run-7",
                batch_size=4,
                prompt="Describe.",
                model_hint=None,
                interval_sec=1.0,
                frames=self._frames(100.0, 4),
            )

            normalized = manager._normalize_summary_log_entry(
                manager.run_summary_batch(batch)
            )

        stats = normalized["lm_response_stats"]
        self.assertEqual(stats["completion_tokens"], 128)
        self.assertEqual(stats["http_ms"], 91.25)
        self.assertNotIn("endpoint", stats)

    def test_truncated_markerless_l0_is_persisted_as_closed_schema_only_state(self):
        class TruncatedSummary(str):
            def __new__(cls, value):
                result = super().__new__(cls, value)
                result.eva_response_meta = {
                    "attempt_count": 1,
                    "finish_reason": "length",
                    "completion_tokens": 320,
                }
                return result

        response = TruncatedSummary(
            "### Scene description\n"
            "A person gives a thumbs-up. Coverage is complete.\n\n"
            "### Episode update\nThumbs-up remains visible.\n\n"
            '{"version":2,"alerts":[{"title":"Thumbs up",'
            '"description":"Visible thumbs-up gesture","severity":"info",'
            '"snapshot_indices":[2]}],"events":[],"observed_states":[],'
            '"cover":{"snapshot_index":2,"kind":"event"},'
            '"scene":{"status":"matched","summary":"Person at desk"},'
            '"routines":[],"memory_pass":[],"batch_end_ms":101000,'
            '"clip_probe_legend":['
        )
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                lm_callback=lambda _messages, _hint: response,
            )
            batch = manager.create_summary_batch(
                channel_id=7,
                run_id="run-7",
                batch_size=2,
                prompt="Describe.",
                model_hint=None,
                interval_sec=1.0,
                frames=self._frames(100.0, 2),
            )
            entry = manager.run_summary_batch(batch)

        self.assertNotIn("clip_probe_legend", entry["summary"])
        self.assertNotIn("batch_end_ms", entry["summary"])
        self.assertNotIn("Coverage is complete", entry["summary"])
        payload = json.loads(entry["summary"].split("BATCH_STATE_JSON:\n", 1)[1])
        self.assertEqual(
            list(payload),
            [
                "version",
                "alerts",
                "events",
                "observed_states",
                "cover",
                "scene",
                "routines",
                "memory_pass",
            ],
        )
        self.assertEqual(payload["alerts"][0]["title"], "Thumbs up")
        stats = entry["lm_response_stats"]
        self.assertEqual(
            stats["batch_state_input_contract_status"],
            "partial_prefix_markerless",
        )
        self.assertEqual(
            stats["batch_state_output_contract_status"],
            "parsed_canonical",
        )
        self.assertTrue(stats["batch_state_canonicalized"])

    def test_l0_retries_cjk_output_once_and_keeps_english_result(self):
        calls = []
        responses = [
            (
                "### Scene description\n室内场景，有人在电脑前。\n\n"
                "### Episode update\n人保持坐姿。\n\n"
                "### Routine and deviations\n常规工作。\n\n"
                "### Worth to remember\nNone\n\n"
                "BATCH_STATE_JSON:\n"
                '{"version":2,"alerts":[],"events":[],"observed_states":[],'
                '"cover":{"snapshot_index":1,"kind":"routine","reason":"室内场景",'
                '"confidence":"medium"},"scene":{"status":"matched","summary":"有人工作"},'
                '"routines":[],"memory_pass":[]}'
            ),
            (
                "### Scene description\nA person is seated at a computer.\n\n"
                "### Episode update\nThe person remains seated.\n\n"
                "### Routine and deviations\nRoutine desk work.\n\n"
                "### Worth to remember\nNone\n\n"
                "BATCH_STATE_JSON:\n"
                '{"version":2,"alerts":[],"events":[],"observed_states":[],'
                '"cover":{"snapshot_index":1,"kind":"routine","reason":"Representative desk scene",'
                '"confidence":"medium"},"scene":{"status":"matched","summary":"Person at desk"},'
                '"routines":[],"memory_pass":[]}'
            ),
        ]

        def lm_callback(messages, _hint):
            calls.append(list(messages))
            return responses[len(calls) - 1]

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp), lm_callback=lm_callback)
            batch = manager.create_summary_batch(
                channel_id=7,
                run_id="run-7",
                batch_size=2,
                prompt="Describe.",
                model_hint=None,
                interval_sec=1.0,
                frames=sample_frames(),
            )
            entry = manager.run_summary_batch(batch)

        self.assertIn("concise English only", batch["system_prompt"])
        self.assertEqual(len(calls), 2)
        self.assertIn("violated the mandatory output-language contract", calls[1][-1]["content"])
        self.assertIn("室内场景", calls[1][-2]["content"])
        self.assertEqual(manager._east_asian_script_char_count(entry["summary"]), 0)
        self.assertIn("A person is seated at a computer", entry["summary"])
        self.assertEqual(entry["lm_response_stats"]["language_contract_status"], "recovered")
        self.assertEqual(entry["lm_response_stats"]["language_retry_count"], 1)
        self.assertEqual(entry["lm_response_stats"]["attempt_count"], 2)

    def test_l0_quarantines_persistent_cjk_but_preserves_structured_alert(self):
        calls = []
        cjk_response = (
            "### Scene description\n室内场景，有人在电脑前。\n\n"
            "### Episode update\n人举起拇指。\n\n"
            "### Routine and deviations\n常规工作。\n\n"
            "### Worth to remember\nNone\n\n"
            "BATCH_STATE_JSON:\n"
            '{"version":2,"alerts":[{"title":"拇指手势","description":"当前画面可见手势",'
            '"severity":"low","snapshot_indices":[1]}],"events":[],"observed_states":[],'
            '"cover":{"snapshot_index":1,"kind":"event","reason":"可见手势",'
            '"confidence":"medium"},"scene":{"status":"matched","summary":"有人工作"},'
            '"routines":[{"key":"日常工作","label":"日常工作","state":"continuing",'
            '"snapshot_indices":[1],"applies_to_event_keys":["先前事件"]}],'
            '"memory_pass":["记住这个场景"]}'
        )

        def lm_callback(messages, _hint):
            calls.append(list(messages))
            return cjk_response

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp), lm_callback=lm_callback)
            batch = manager.create_summary_batch(
                channel_id=7,
                run_id="run-7",
                batch_size=2,
                prompt="Describe.",
                model_hint=None,
                interval_sec=1.0,
                frames=sample_frames(),
            )
            entry = manager.run_summary_batch(batch)

        self.assertEqual(len(calls), 2)
        self.assertEqual(manager._east_asian_script_char_count(entry["summary"]), 0)
        self.assertIn("reliable English scene wording was unavailable", entry["summary"])
        self.assertEqual(entry["batch_state"]["contract_status"], "language_contract_fallback")
        self.assertEqual(len(entry["batch_state"]["alerts"]), 1)
        self.assertEqual(entry["batch_state"]["alerts"][0]["title"], "Operator review signal")
        self.assertEqual(entry["batch_state"]["alerts"][0]["snapshot_indices"], [1])
        self.assertEqual(entry["lm_response_stats"]["language_contract_status"], "fallback")
        self.assertEqual(entry["lm_response_stats"]["final_east_asian_chars"], 0)

    def test_l0_generation_budget_is_shorter_for_heartbeat_than_event(self):
        calls = []

        def lm_callback(_messages, _hint, **kwargs):
            calls.append(dict(kwargs))
            return "All calm."

        lm_callback.eva_workload_class = True
        lm_callback.eva_max_tokens_override = True
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp), lm_callback=lm_callback)
            heartbeat = manager.create_summary_batch(
                channel_id=7,
                run_id="run-7",
                batch_size=4,
                prompt="Describe.",
                model_hint=None,
                interval_sec=1.0,
                frames=self._frames(100.0, 4),
            )
            heartbeat["lm_workload_class"] = "vlm"
            heartbeat_entry = manager.run_summary_batch(heartbeat)

            event = manager.create_summary_batch(
                channel_id=7,
                run_id="run-7",
                batch_size=4,
                prompt="Describe.",
                model_hint=None,
                interval_sec=1.0,
                frames=self._frames(110.0, 4),
            )
            event["lm_workload_class"] = "alert"
            event_entry = manager.run_summary_batch(event)

        self.assertEqual(calls[0]["max_tokens_override"], 320)
        self.assertEqual(calls[1]["max_tokens_override"], 384)
        self.assertEqual(
            heartbeat_entry["llm_input_stats"]["generation_output_mode"],
            "heartbeat",
        )
        self.assertEqual(
            event_entry["llm_input_stats"]["generation_output_mode"],
            "event",
        )


def _capture_max_coalesce() -> int:
    from luxriot_connector import _SUMMARY_COALESCE_MAX_BATCHES

    return _SUMMARY_COALESCE_MAX_BATCHES


class LuxriotIncidentFocusRuntimeTests(unittest.TestCase):
    def test_focus_raises_l0_profile_and_degraded_source_stays_degraded(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.start_incident_focus(
                "incident-7",
                [7],
                level="follow",
                ttl_seconds=30,
                context='{"title":"Desk departure","timeline":[{"label":"Person left the desk"}]}',
            )
            mode, priority, digest = manager.apply_incident_focus(
                7,
                "quiet",
                0.1,
            )
            self.assertEqual(mode.value, "active")
            self.assertGreaterEqual(priority, 0.65)
            self.assertEqual(digest["incident_ids"], ["incident-7"])
            self.assertTrue(digest["context_attached"])
            live_prompt = manager.compose_live_system_prompt(
                7,
                "Describe the current snapshots.",
            )
            self.assertIn("INCIDENT_FOCUS_JSON", live_prompt)
            self.assertIn("Person left the desk", live_prompt)
            self.assertIn("not current visual evidence", live_prompt)

            session = LuxriotCaptureSession(
                manager,
                channel_id=7,
                batch_size=12,
                prompt="Describe.",
            )
            focused_frame = {
                "captured_at": 1.0,
                "attention_mode": mode.value,
                "incident_focus": digest,
            }
            self.assertEqual(
                session._summary_cadence_for_frame(focused_frame),
                2.5,
            )
            session.frames = [
                {
                    **focused_frame,
                    "captured_at": float(index),
                }
                for index in range(1, 13)
            ]
            self.assertEqual(session._summary_profile_locked().mode.value, "active")
            enqueued = []

            def enqueue(**kwargs):
                enqueued.append(kwargs)
                session.frames.clear()
                return True

            with patch.object(session, "_enqueue_summary_batch", side_effect=enqueue):
                session._summarize_if_ready()
            self.assertEqual(enqueued[0]["workload_class"], "event")
            self.assertEqual(enqueued[0]["frame_limit"], 12)

            degraded, degraded_priority, degraded_digest = (
                manager.apply_incident_focus(7, "degraded", 0.0)
            )
            self.assertEqual(degraded.value, "degraded")
            self.assertEqual(degraded_priority, 0.0)
            self.assertEqual(
                degraded_digest["suppressed"],
                "degraded_source",
            )

    def test_critical_focus_uses_burst_floor_and_stop_restores_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.start_incident_focus(
                "incident-critical",
                [7],
                level="critical",
                ttl_seconds=30,
            )
            mode, priority, _digest = manager.apply_incident_focus(7, "watch", 0.2)
            self.assertEqual(mode.value, "burst")
            self.assertGreaterEqual(priority, 1.1)
            self.assertTrue(manager.stop_incident_focus("incident-critical"))
            mode, priority, digest = manager.apply_incident_focus(7, "watch", 0.2)
            self.assertEqual(mode.value, "watch")
            self.assertEqual(priority, 0.2)
            self.assertIsNone(digest)

    def test_critical_focus_does_not_jump_an_older_coverage_waiter(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                config_overrides={
                    "LUXRIOT_ATTENTION_SCHEDULER_ENABLED": True,
                },
            )
            now_ms = int(time.time() * 1000.0)
            config = manager._l0_cost_budget_config
            manager._l0_cost_budget_state = CostBudgetState(
                timestamp_ms=now_ms,
                available_tokens=-config.token_borrow_limit,
                available_slot_seconds=-config.slot_seconds_borrow_limit,
            )
            self.assertFalse(
                manager.reserve_l0_cost_budget(
                    channel_id=8,
                    mode=AttentionMode.QUIET,
                    frame_count=8,
                    coverage_overdue=True,
                )
            )
            manager.start_incident_focus(
                "incident-critical",
                [7],
                level="critical",
                ttl_seconds=30,
            )
            focused_mode, _priority, _digest = manager.apply_incident_focus(
                7,
                AttentionMode.QUIET,
                0.0,
            )
            self.assertFalse(
                manager.reserve_l0_cost_budget(
                    channel_id=7,
                    mode=focused_mode,
                    frame_count=8,
                )
            )
            self.assertEqual(
                manager.l0_cost_budget_status()["waiting_channel_ids"],
                [7, 8],
            )

    def test_runtime_keeps_eight_hot_but_only_four_in_the_vlm_prompt(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            for index in range(10):
                manager.start_incident_focus(
                    f"incident-{index}",
                    [7],
                    level="follow",
                    ttl_seconds=30,
                    context=json.dumps({"title": f"Incident {index}"}),
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

            focus_json = batch["system_prompt"].split(
                "INCIDENT_FOCUS_JSON:\n", 1
            )[1].splitlines()[0]
            focus = json.loads(focus_json)
            self.assertEqual(len(focus["incident_ids"]), 4)
            self.assertEqual(focus["hot_unresolved_count"], 8)
            self.assertEqual(focus["omitted_incident_count"], 6)
            telemetry = batch["llm_input_stats"]["incident_context_budget"]
            self.assertEqual(telemetry["prompt_incident_count"], 4)
            self.assertEqual(telemetry["hot_count"], 8)
            status = manager.incident_focus_status()
            self.assertEqual(status["attention_policy"]["hot_unresolved_limit"], 8)
            self.assertEqual(
                status["prompt_budget"]["context_window_tokens"], 16384
            )

    def test_vision_budget_fails_before_sending_an_oversized_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                config_overrides={
                    "LUXRIOT_L0_VISION_BUDGET_TOKENS": 512,
                    "LUXRIOT_L0_VISION_TOKENS_PER_IMAGE_ESTIMATE": 300,
                },
            )
            with self.assertRaisesRegex(PromptBudgetError, "vision-token budget"):
                manager.create_summary_batch(
                    channel_id=7,
                    run_id="run-7",
                    batch_size=2,
                    prompt="Describe activity.",
                    model_hint=None,
                    interval_sec=5.0,
                    frames=sample_frames(),
                )


class LuxriotCaptureAttentionSignalTests(unittest.TestCase):
    def test_burst_seconds_reach_vector_signal_and_prompt_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.note_capture_baseline(7, {"level": 0.002, "dev": 0.0004, "buckets": 600})
            frames = [
                _burst_batch_frame(timestamp_ms=100_000),
                _burst_batch_frame(
                    timestamp_ms=101_000,
                    mode="normal",
                    activity_x=1.1,
                    with_companion=False,
                ),
            ]
            batch = manager.create_summary_batch(
                channel_id=7,
                run_id="run-7",
                batch_size=12,
                prompt="Describe.",
                model_hint=None,
                interval_sec=1.0,
                frames=frames,
            )
            attention = batch["vector_signal"].get("capture_attention")
            self.assertIsNotNone(attention)
            seconds = attention["seconds"]
            self.assertEqual(len(seconds), 1)
            self.assertEqual(seconds[0]["snapshot"], 1)
            self.assertEqual(seconds[0]["mode"], "burst")
            self.assertEqual(seconds[0]["blur"], "expected_motion")
            self.assertTrue(seconds[0]["sharper_companion"])
            self.assertAlmostEqual(seconds[0]["activity_x"], 12.4)
            self.assertFalse(attention["baseline"]["warmup"])

            system_prompt = batch["system_prompt"]
            self.assertIn("VECTOR_SIGNALS_JSON", system_prompt)
            self.assertIn("capture_attention marks snapshots", system_prompt)
            self.assertIn("Measured motion homeostasis", system_prompt)
            self.assertIn("typical per-second motion on this channel is low", system_prompt)

    def test_observation_contract_forbids_intent_and_safety_conclusions(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            prompt = manager.compose_live_system_prompt(7, "Describe the stream.")
            self.assertIn("Never assert intent or skill", prompt)
            self.assertIn("Never declare 'no safety hazard'", prompt)
            self.assertIn("visually unconfirmed", prompt)

    def test_quiet_batches_do_not_spend_prompt_tokens_on_attention(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            frames = [
                _burst_batch_frame(
                    timestamp_ms=100_000,
                    mode="quiet",
                    activity_x=0.4,
                    with_companion=False,
                )
            ]
            batch = manager.create_summary_batch(
                channel_id=7,
                run_id="run-7",
                batch_size=12,
                prompt="Describe.",
                model_hint=None,
                interval_sec=1.0,
                frames=frames,
            )
            self.assertNotIn("capture_attention", batch["vector_signal"] or {})

    def test_archive_sampler_keeps_burst_apex_and_companion(self):
        frames = [
            _burst_batch_frame(timestamp_ms=100_000, mode="normal", activity_x=1.0, with_companion=False),
            _burst_batch_frame(timestamp_ms=101_000),
            _burst_batch_frame(timestamp_ms=102_000, mode="normal", activity_x=1.0, with_companion=False),
        ]
        records = LuxriotManager._summary_archive_frames(
            frames,
            batch_start_ms=100_000,
            batch_end_ms=102_000,
            sample_count=2,
        )
        roles = [record["anchor_role"] for record in records]
        self.assertIn("burst_apex", roles)
        self.assertIn("burst_companion", roles)
        companion_record = next(record for record in records if record["anchor_role"] == "burst_companion")
        self.assertEqual(companion_record["thumbnail"], "companion-101000")
        self.assertEqual(companion_record["timestamp_ms"], 101_400)
        self.assertEqual(companion_record["companion_of_timestamp_ms"], 101_000)
        apex_record = next(record for record in records if record["anchor_role"] == "burst_apex")
        self.assertEqual(apex_record["timestamp_ms"], 101_000)
        self.assertEqual(companion_record["embedding_status"], "unavailable")

    def test_archive_sampler_reuses_independent_semantic_embedding(self):
        frames = [
            {
                "thumbnail": "frame-1",
                "timestamp_ms": 100_000,
                "clip_embedding": [0.25, 0.75],
                "embedding_space": {
                    "backend": "siglip2",
                    "model": "google/siglip2-base-patch16-224",
                    "dimension": 2,
                },
                "embedding_ref": "probe-buffer:7:41",
                "embedding_status": "ready",
            }
        ]

        records = LuxriotManager._summary_archive_frames(
            frames,
            batch_start_ms=100_000,
            batch_end_ms=100_000,
            sample_count=4,
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["clip_embedding"], [0.25, 0.75])
        self.assertEqual(records[0]["embedding_ref"], "probe-buffer:7:41")
        self.assertEqual(records[0]["embedding_status"], "ready")
        self.assertEqual(records[0]["embedding_space"]["backend"], "siglip2")

    def test_message_builder_appends_single_burst_companion_frame(self):
        import oldapp

        frames = [
            _burst_batch_frame(timestamp_ms=100_000, activity_x=5.0),
            _burst_batch_frame(timestamp_ms=101_000, activity_x=20.0),
            _burst_batch_frame(timestamp_ms=102_000, mode="normal", activity_x=1.0, with_companion=False),
        ]
        messages = oldapp._build_luxriot_messages("#7", frames, "Describe.", "System.")
        user_content = messages[1]["content"]
        image_parts = [part for part in user_content if part.get("type") == "image_url"]
        self.assertEqual(len(image_parts), 4)
        companion_notes = [
            part["text"]
            for part in user_content
            if part.get("type") == "text" and "sharper companion" in str(part.get("text") or "")
        ]
        self.assertEqual(len(companion_notes), 1)
        self.assertIn("Snapshot 4 - sharper companion of burst Snapshot 2", companion_notes[0])
        self.assertIn("companion-101000", image_parts[-1]["image_url"]["url"])

    def test_context_token_estimate_warns_before_model_truncation(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            stats = manager._estimate_message_payload_chars(
                [
                    {
                        "role": "user",
                        "content": (
                            [{"type": "text", "text": "x" * 8_000}]
                            + [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "data:image/jpeg;base64,abc", "detail": "high"},
                                }
                            ]
                            * 20
                        ),
                    }
                ]
            )
            self.assertGreaterEqual(stats["estimated_context_tokens"], 8_000)
            warnings = manager._summary_input_warnings(stats)
            self.assertTrue(any("estimated_context_tokens" in warning for warning in warnings))


class LuxriotCaptureDispatchTests(unittest.TestCase):
    def test_async_clip_dispatch_keeps_only_latest_pending_frame_per_channel(self):
        started = threading.Event()
        release = threading.Event()
        executions = []
        callbacks = []

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                config_overrides={
                    "LUXRIOT_CLIP_ASYNC_ENABLED": True,
                    "LUXRIOT_CLIP_ASYNC_WORKERS": 2,
                    "LUXRIOT_CLIP_ASYNC_QUEUE_CAPACITY": 8,
                },
            )

            def work(label):
                def run():
                    executions.append(label)
                    if label == "first":
                        started.set()
                        if not release.wait(timeout=3.0):
                            raise RuntimeError("test embedding release timed out")
                    return {"label": label}

                return run

            def callback(label):
                def complete(result, error):
                    callbacks.append((label, result, error))

                return complete

            self.assertTrue(
                manager.submit_probe_embedding(
                    7,
                    work("first"),
                    callback("first"),
                )
            )
            self.assertTrue(started.wait(timeout=1.0))
            self.assertTrue(
                manager.submit_probe_embedding(
                    7,
                    work("second"),
                    callback("second"),
                )
            )
            self.assertTrue(
                manager.submit_probe_embedding(
                    7,
                    work("third"),
                    callback("third"),
                )
            )
            self.assertEqual(
                callbacks,
                [("second", None, None)],
            )

            release.set()
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                status = manager.probe_embedding_dispatch_status()
                if (
                    status["completed_total"] == 2
                    and status["active_channels"] == 0
                ):
                    break
                time.sleep(0.01)

            manager.stop_probe_embedding_executor(wait=True)
            self.assertEqual(executions, ["first", "third"])
            self.assertEqual(
                [(label, result) for label, result, _error in callbacks],
                [
                    ("second", None),
                    ("first", {"label": "first"}),
                    ("third", {"label": "third"}),
                ],
            )
            self.assertTrue(all(error is None for _label, _result, error in callbacks))
            status = manager.probe_embedding_dispatch_status()
            self.assertEqual(status["submitted_total"], 3)
            self.assertEqual(status["completed_total"], 2)
            self.assertEqual(status["coalesced_total"], 1)
            self.assertEqual(status["pending"], 0)
            self.assertEqual(status["in_flight"], 0)
            self.assertEqual(status["active_channels"], 0)
            self.assertEqual(status["pending_latest_channels"], 0)
            self.assertGreaterEqual(status["last_queue_wait_ms"], 0.0)
            self.assertGreaterEqual(status["last_work_ms"], 0.0)
            self.assertGreaterEqual(status["last_callback_ms"], 0.0)
            self.assertGreaterEqual(status["average_work_ms"], 0.0)

    def test_async_clip_dispatch_does_not_block_capture_bucket_rollover(self):
        started = threading.Event()
        release = threading.Event()

        class BlockingProbeManager:
            def add_frame(
                self,
                channel_id,
                _image,
                timestamp_ms,
                provenance=None,
            ):
                started.set()
                if not release.wait(timeout=3.0):
                    raise RuntimeError("test embedding release timed out")
                return {
                    "channel_id": channel_id,
                    "timestamp_ms": timestamp_ms,
                    "embedding_ref": f"probe-buffer:{channel_id}:1",
                    "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
                    "thumbnail": "jpeg",
                }

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                config_overrides={
                    "LUXRIOT_CLIP_ASYNC_ENABLED": True,
                    "LUXRIOT_CLIP_ASYNC_WORKERS": 2,
                    "LUXRIOT_CLIP_ASYNC_QUEUE_CAPACITY": 8,
                    "LUXRIOT_ATTENTION_SCHEDULER_ENABLED": True,
                },
            )
            manager.probe_manager = BlockingProbeManager()
            manager.shared_probe_channels.add(7)
            session = LuxriotCaptureSession(
                manager,
                channel_id=7,
                batch_size=8,
                prompt="Describe.",
                run_id="run-7",
            )
            first = Image.new("RGB", (24, 16), color=(0, 0, 0))
            second = Image.new("RGB", (24, 16), color=(32, 32, 32))

            session._accept_captured_frame(
                first,
                100_100,
                summarize=False,
            )
            started_at = time.monotonic()
            session._accept_captured_frame(
                second,
                101_100,
                summarize=False,
            )
            elapsed = time.monotonic() - started_at

            self.assertTrue(started.wait(timeout=1.0))
            self.assertLess(elapsed, 0.5)
            self.assertEqual(len(session.frames), 1)
            self.assertEqual(session.frames[0]["embedding_status"], "pending")
            release.set()
            deadline = time.monotonic() + 3.0
            while (
                time.monotonic() < deadline
                and not session.frames[0].get("embedding_ref")
            ):
                time.sleep(0.01)
            self.assertEqual(len(session.frames), 1)
            self.assertTrue(session.frames[0].get("embedding_ref"))
            self.assertEqual(session.frames[0]["embedding_status"], "ready")
            status = manager.probe_embedding_dispatch_status()
            self.assertEqual(status["submitted_total"], 1)
            self.assertEqual(status["completed_total"], 1)
            self.assertEqual(status["failed_total"], 0)
            manager.stop_probe_embedding_executor(wait=True)

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
            self.assertEqual(session.queue_dropped_batches, 0)
            self.assertEqual(session.summary_coverage_gap_batches, 0)

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
            self.assertEqual(session.queue_dropped_batches, 0)
            self.assertEqual(session.summary_coverage_gap_batches, 0)

    def test_normal_capture_does_not_clear_summary_failure_before_success(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.set_summary_dispatcher(
                lambda _batch, _workload: (_ for _ in ()).throw(RuntimeError("vlm unavailable"))
            )
            session = LuxriotCaptureSession(
                manager,
                channel_id=7,
                batch_size=2,
                prompt="Describe activity.",
                run_id="run-7",
            )
            manager.sessions[7] = session

            dispatched = session._dispatch_summary_frames(
                sample_frames(),
                restore_on_failure=False,
            )

            self.assertFalse(dispatched)
            self.assertIn("vlm unavailable", session.summary_last_error)
            self.assertEqual(session.summary_failed_batches, 1)
            self.assertEqual(session.queue_dropped_batches, 1)
            self.assertEqual(session.dropped_frames, 2)
            gaps = [
                row
                for row in manager.summary_history.get(7, [])
                if row.get("coverage_gap")
            ]
            self.assertEqual(len(gaps), 1)
            self.assertEqual(gaps[0]["gap_reason"], "lm_dispatch_exception")
            self.assertEqual(gaps[0]["gap_error_type"], "RuntimeError")
            self.assertEqual(gaps[0]["gap_error"], "vlm unavailable")

            session._accept_captured_frame(
                SimpleNamespace(width=1280, height=720),
                110_000,
                summarize=True,
            )

            self.assertEqual(len(session.frames), 1)
            self.assertIn("vlm unavailable", session.summary_last_error)
            self.assertIn("vlm unavailable", session.last_error)

            accepted = manager.accept_summary_entry(
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": "successful summary",
                    "frame_count": 2,
                    "created_at": 110.0,
                    "batch_start_ms": 100_000,
                    "batch_end_ms": 110_000,
                }
            )

            self.assertTrue(accepted["accepted"])
            self.assertIsNone(session.summary_last_error)
            self.assertIsNone(session.last_error)
            self.assertIsNotNone(session.summary_last_success_at)

    def test_accepted_l0_batch_emits_one_replay_safe_incident_heartbeat(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            received = []
            manager.set_incident_observation_callback(
                lambda channel_id, payload: received.append(
                    (channel_id, dict(payload))
                )
                or {"inserted": 1}
            )
            entry = {
                "channel_id": 7,
                "run_id": "run-7",
                "batch_id": "vlm-stable-batch",
                "summary": "Person remains at the gate.",
                "frame_count": 2,
                "created_at": 110.0,
                "batch_start_ms": 100_000,
                "batch_end_ms": 110_000,
                "batch_state": {
                    "version": 2,
                    "events": [],
                    "routines": [],
                    "observed_states": [],
                    "alerts": [],
                },
            }

            accepted = manager.accept_summary_entry(entry)

            self.assertEqual(len(received), 1)
            self.assertEqual(received[0][0], 7)
            self.assertEqual(received[0][1]["batch_id"], "vlm-stable-batch")
            self.assertEqual(received[0][1]["batch_end_ms"], 110_000)
            self.assertEqual(
                accepted["incident_observation_admission"], {"inserted": 1}
            )

    def test_channel_false_and_zero_overrides_persist_without_inheriting_defaults(self):
        with tempfile.TemporaryDirectory() as temp:
            state_store = MemoryRuntimeStateStore()
            manager = build_manager(
                Path(temp),
                runtime_state_store=state_store,
                config_overrides={
                    "LUXRIOT_AUTO_BOOKMARKS": True,
                    "LUXRIOT_BOOKMARK_COOLDOWN_SEC": 60.0,
                },
            )

            settings = manager.update_prompt_settings(
                channel_id=7,
                bookmark_enabled=False,
                bookmark_cooldown_sec=0.0,
            )

            self.assertFalse(settings["bookmark_enabled"])
            self.assertEqual(settings["bookmark_cooldown_sec"], 0.0)
            self.assertIn("bookmark_enabled", manager.channel_prompt_overrides[7])
            self.assertIs(manager.channel_prompt_overrides[7]["bookmark_enabled"], False)
            self.assertIn("bookmark_cooldown_sec", manager.channel_prompt_overrides[7])
            self.assertEqual(manager.channel_prompt_overrides[7]["bookmark_cooldown_sec"], 0.0)
            self.assertEqual(settings["setting_sources"]["bookmark_enabled"], "channel_override")
            self.assertIn("bookmark_enabled", settings["override_fields"])
            self.assertTrue(settings["persistence"]["persisted"])
            self.assertEqual(settings["persistence"]["revision"], 1)

            restored = build_manager(
                Path(temp),
                runtime_state_store=state_store,
                config_overrides={
                    "LUXRIOT_AUTO_BOOKMARKS": True,
                    "LUXRIOT_BOOKMARK_COOLDOWN_SEC": 60.0,
                },
            )
            restored_settings = restored.get_prompt_settings(channel_id=7)
            self.assertFalse(restored_settings["bookmark_enabled"])
            self.assertEqual(restored_settings["bookmark_cooldown_sec"], 0.0)
            self.assertEqual(
                restored_settings["setting_sources"]["stream_system_prompt"],
                "persisted_runtime_default",
            )

    def test_channel_overrides_can_be_explicitly_cleared_back_to_inherited_defaults(self):
        with tempfile.TemporaryDirectory() as temp:
            state_store = MemoryRuntimeStateStore()
            manager = build_manager(
                Path(temp),
                runtime_state_store=state_store,
                config_overrides={
                    "LUXRIOT_AUTO_BOOKMARKS": True,
                    "LUXRIOT_BOOKMARK_COOLDOWN_SEC": 60.0,
                },
            )

            overridden = manager.update_prompt_settings(
                channel_id=7,
                stream_system_prompt="",
                alert_policy_prompt="Temporary watch condition",
                rollup_prompts={"L1": "Channel-specific L1"},
                bookmark_enabled=False,
                bookmark_cooldown_sec=0.0,
            )

            self.assertEqual(overridden["stream_system_prompt"], "")
            self.assertEqual(overridden["setting_sources"]["stream_system_prompt"], "channel_override")
            self.assertIn("rollup_prompts.L1", overridden["override_fields"])
            self.assertEqual(overridden["persistence"]["revision"], 1)

            inherited = manager.update_prompt_settings(
                channel_id=7,
                clear_override_fields=overridden["override_fields"],
            )

            self.assertEqual(inherited["override_fields"], [])
            self.assertFalse(inherited["has_channel_override"])
            self.assertNotIn(7, manager.channel_prompt_overrides)
            self.assertEqual(inherited["stream_system_prompt"], "Describe the stream.")
            self.assertTrue(inherited["bookmark_enabled"])
            self.assertEqual(inherited["bookmark_cooldown_sec"], 60.0)
            self.assertEqual(inherited["setting_sources"]["stream_system_prompt"], "persisted_runtime_default")
            self.assertEqual(inherited["persistence"]["revision"], 2)

            restored = build_manager(
                Path(temp),
                runtime_state_store=state_store,
                config_overrides={
                    "LUXRIOT_AUTO_BOOKMARKS": True,
                    "LUXRIOT_BOOKMARK_COOLDOWN_SEC": 60.0,
                },
            )
            self.assertEqual(restored.get_prompt_settings(channel_id=7)["override_fields"], [])

    def test_manual_alert_criteria_survive_runtime_manager_reload(self):
        with tempfile.TemporaryDirectory() as temp:
            state_store = MemoryRuntimeStateStore()
            manager = build_manager(Path(temp), runtime_state_store=state_store)
            criteria = (
                "Alert when a person leaves the workstation. "
                "Do not alert for the resident cat."
            )

            saved = manager.update_prompt_settings(
                channel_id=7,
                alert_policy_prompt=criteria,
            )

            self.assertEqual(saved["alert_policy_prompt"], criteria)
            self.assertEqual(
                saved["setting_sources"]["alert_policy_prompt"],
                "channel_override",
            )
            self.assertIn("alert_policy_prompt", saved["override_fields"])
            self.assertTrue(saved["persistence"]["persisted"])
            self.assertEqual(
                saved["persistence"]["authority"],
                manager.PROMPT_SETTINGS_STATE_KEY,
            )
            dedicated = state_store.load_state(manager.PROMPT_SETTINGS_STATE_KEY)
            self.assertEqual(
                dedicated["prompt_settings"]["channel_overrides"]["7"][
                    "alert_policy_prompt"
                ],
                criteria,
            )

            restored = build_manager(
                Path(temp),
                runtime_state_store=state_store,
            )
            restored_settings = restored.get_prompt_settings(channel_id=7)
            self.assertEqual(restored_settings["alert_policy_prompt"], criteria)
            self.assertEqual(
                restored_settings["setting_sources"]["alert_policy_prompt"],
                "channel_override",
            )
            self.assertIn("alert_policy_prompt", restored_settings["override_fields"])

            batch = restored.create_summary_batch(
                channel_id=7,
                run_id="prompt-audit",
                batch_size=2,
                prompt="",
                model_hint=None,
                interval_sec=1.0,
                frames=sample_frames(),
            )
            input_stats = batch["llm_input_stats"]
            self.assertEqual(
                input_stats["operator_alert_policy_chars"], len(criteria)
            )
            self.assertEqual(
                len(input_stats["operator_alert_policy_fingerprint"]), 16
            )
            self.assertIn(criteria, batch["system_prompt"])
            compacted_stats = restored._compact_llm_input_stats(input_stats)
            self.assertEqual(
                compacted_stats["operator_alert_policy_chars"], len(criteria)
            )
            self.assertEqual(
                compacted_stats["operator_alert_policy_fingerprint"],
                input_stats["operator_alert_policy_fingerprint"],
            )

    def test_dedicated_prompt_state_migrates_verbose_contract_without_losing_alerts(self):
        with tempfile.TemporaryDirectory() as temp:
            state_store = MemoryRuntimeStateStore()
            criteria = "Alert when a person raises a thumb."
            state_store.save_state(
                LuxriotManager.PROMPT_SETTINGS_STATE_KEY,
                {
                    "version": 1,
                    "revision": 7,
                    "updated_at": 1_786_633_000.0,
                    "prompt_settings": {
                        "stream_system_prompt": "Describe the stream.",
                        "alert_policy_prompt": "",
                        "rollup_prompts": {},
                        "bookmark_enabled": False,
                        "bookmark_cooldown_sec": 60.0,
                        "capture_selector_enabled": True,
                        "capture_selector_bias": "auto",
                        "json_alert_prompt": PREVIOUS_VERBOSE_BATCH_STATE_JSON_PROMPT,
                        "channel_overrides": {
                            "7": {"alert_policy_prompt": criteria}
                        },
                    },
                },
            )

            restored = build_manager(Path(temp), runtime_state_store=state_store)
            settings = restored.get_prompt_settings(channel_id=7)
            self.assertEqual(settings["alert_policy_prompt"], criteria)
            self.assertEqual(settings["json_alert_prompt"], DEFAULT_ALERTS_JSON_PROMPT)
            migrated = state_store.load_state(restored.PROMPT_SETTINGS_STATE_KEY)
            self.assertEqual(migrated["revision"], 8)
            self.assertEqual(
                migrated["prompt_settings"]["channel_overrides"]["7"][
                    "alert_policy_prompt"
                ],
                criteria,
            )
            self.assertEqual(
                migrated["prompt_settings"]["json_alert_prompt"],
                DEFAULT_ALERTS_JSON_PROMPT,
            )

    def test_compact_shipped_contract_migrates_without_overriding_custom_contracts(self):
        self.assertEqual(
            LuxriotManager._normalize_json_alert_prompt(
                PREVIOUS_COMPACT_BATCH_STATE_JSON_PROMPT
            ),
            DEFAULT_ALERTS_JSON_PROMPT,
        )
        custom = (
            "Custom current evidence contract.\n"
            "BATCH_STATE_JSON:\n"
            '{"version":2,"alerts":[],"events":[],"observed_states":[],'
            '"cover":{},"scene":{},"routines":[],"memory_pass":[]}'
        )
        self.assertEqual(
            LuxriotManager._normalize_json_alert_prompt(custom),
            custom,
        )

    def test_previous_english_contract_migrates_to_bounded_output_contract(self):
        self.assertEqual(
            LuxriotManager._normalize_json_alert_prompt(
                PREVIOUS_ENGLISH_BATCH_STATE_JSON_PROMPT
            ),
            DEFAULT_ALERTS_JSON_PROMPT,
        )

    def test_pre_english_contract_migrates_to_bounded_output_contract(self):
        self.assertNotIn(
            "concise English only",
            PREVIOUS_PRE_ENGLISH_BATCH_STATE_JSON_PROMPT,
        )
        self.assertEqual(
            LuxriotManager._normalize_json_alert_prompt(
                PREVIOUS_PRE_ENGLISH_BATCH_STATE_JSON_PROMPT
            ),
            DEFAULT_ALERTS_JSON_PROMPT,
        )

    def test_dedicated_prompt_state_survives_stale_high_revision_summary_write(self):
        with tempfile.TemporaryDirectory() as temp:
            state_store = MemoryRuntimeStateStore()
            manager = build_manager(Path(temp), runtime_state_store=state_store)
            criteria = "Alert when a person raises a thumb."
            manager.update_prompt_settings(
                channel_id=7,
                alert_policy_prompt=criteria,
            )

            # Simulate an overlapping old Gunicorn worker.  Its summary
            # revision may be much higher, but its embedded prompt snapshot is
            # stale and must no longer be authoritative.
            stale_summary = dict(state_store.load_state("luxriot_summary_state") or {})
            stale_prompt = dict(stale_summary.get("prompt_settings") or {})
            stale_overrides = dict(stale_prompt.get("channel_overrides") or {})
            stale_overrides["7"] = {"capture_interval_sec": 1.0}
            stale_prompt["channel_overrides"] = stale_overrides
            stale_summary["prompt_settings"] = stale_prompt
            stale_summary["revision"] = 10_000
            state_store.save_state("luxriot_summary_state", stale_summary)

            restored = build_manager(Path(temp), runtime_state_store=state_store)
            restored_settings = restored.get_prompt_settings(channel_id=7)
            self.assertEqual(restored_settings["alert_policy_prompt"], criteria)
            self.assertIn("alert_policy_prompt", restored_settings["override_fields"])
            self.assertEqual(
                restored_settings["persistence"]["authority"],
                manager.PROMPT_SETTINGS_STATE_KEY,
            )

    def test_full_form_echo_does_not_pin_inherited_channel_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            state_store = MemoryRuntimeStateStore()
            manager = build_manager(Path(temp), runtime_state_store=state_store)
            inherited = manager.get_prompt_settings(channel_id=7)

            saved = manager.update_prompt_settings(
                channel_id=7,
                stream_system_prompt=inherited["stream_system_prompt"],
                alert_policy_prompt="Alert when a person falls.",
                rollup_prompts=inherited["rollup_prompts"],
                json_alert_prompt=inherited["json_alert_prompt"],
                bookmark_enabled=inherited["bookmark_enabled"],
                bookmark_cooldown_sec=inherited["bookmark_cooldown_sec"],
                capture_selector_enabled=inherited["capture_selector_enabled"],
                capture_selector_bias=inherited["capture_selector_bias"],
            )

            self.assertEqual(saved["override_fields"], ["alert_policy_prompt"])
            self.assertEqual(
                manager.channel_prompt_overrides[7],
                {"alert_policy_prompt": "Alert when a person falls."},
            )

    def test_prompt_update_rejects_setting_and_clearing_the_same_override(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp), runtime_state_store=MemoryRuntimeStateStore())

            with self.assertRaisesRegex(ValueError, "updated and reset"):
                manager.update_prompt_settings(
                    channel_id=7,
                    bookmark_enabled=False,
                    clear_override_fields=["bookmark_enabled"],
                )

            self.assertNotIn(7, manager.channel_prompt_overrides)

    def test_prompt_settings_persistence_failure_rolls_back_and_is_visible(self):
        class FailingRuntimeStateStore(MemoryRuntimeStateStore):
            def save_state(self, key, payload):
                raise RuntimeError("database write unavailable")

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                runtime_state_store=FailingRuntimeStateStore(),
                config_overrides={"LUXRIOT_AUTO_BOOKMARKS": True},
            )

            with self.assertRaisesRegex(RuntimeError, "in-memory changes were rolled back"):
                manager.update_prompt_settings(
                    channel_id=7,
                    bookmark_enabled=False,
                )

            self.assertTrue(manager.get_prompt_settings(channel_id=7)["bookmark_enabled"])
            self.assertNotIn(7, manager.channel_prompt_overrides)
            persistence = manager.get_prompt_settings(channel_id=7)["persistence"]
            self.assertFalse(persistence["persisted"])
            self.assertTrue(persistence["dirty"])
            self.assertIn("database write unavailable", persistence["last_error"])

    def test_restart_save_failure_leaves_existing_session_running_on_current_generation(self):
        class FailingRuntimeStateStore(MemoryRuntimeStateStore):
            def save_state(self, key, payload):
                raise RuntimeError("state backend unavailable")

        class ExistingSession:
            def __init__(self):
                self.stopped = False

            def stop(self):
                self.stopped = True

            def status(self):
                return {"run_id": "old-run", "logs": [], "running": True}

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                runtime_state_store=FailingRuntimeStateStore(),
            )
            existing = ExistingSession()
            manager.sessions[7] = existing

            with self.assertRaisesRegex(RuntimeError, "channel settings could not be persisted"):
                manager.start_session(channel_id=7, system_prompt="new prompt")

            self.assertIs(manager.sessions[7], existing)
            self.assertFalse(existing.stopped)
            self.assertIsNone(manager._current_session_generation(7))
            self.assertNotIn(7, manager.channel_prompt_overrides)

    def test_restart_run_persistence_failure_does_not_stop_existing_session(self):
        class FailingRuntimeStateStore(MemoryRuntimeStateStore):
            def save_state(self, key, payload):
                raise RuntimeError("run store unavailable")

        class ExistingSession:
            def __init__(self):
                self.stopped = False

            def stop(self):
                self.stopped = True

            def status(self):
                return {"run_id": "old-run", "logs": [], "running": True}

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                runtime_state_store=FailingRuntimeStateStore(),
            )
            existing = ExistingSession()
            manager.sessions[7] = existing

            with self.assertRaisesRegex(RuntimeError, "run state could not be persisted"):
                manager.start_session(channel_id=7, update_desired=False)

            self.assertIs(manager.sessions[7], existing)
            self.assertFalse(existing.stopped)
            self.assertIsNone(manager._current_session_generation(7))

    def test_restart_stops_existing_session_without_holding_manager_cache_lock(self):
        class ExistingSession:
            def __init__(self, manager):
                self.manager = manager
                self.stopped = False
                self.cache_lock_was_free = False

            def stop(self):
                self.cache_lock_was_free = self.manager.cache_lock.acquire(blocking=False)
                if self.cache_lock_was_free:
                    self.manager.cache_lock.release()
                self.stopped = True

            def status(self):
                return {"run_id": "old-run", "logs": [], "running": not self.stopped}

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                runtime_state_store=MemoryRuntimeStateStore(),
            )
            existing = ExistingSession(manager)
            manager.sessions[7] = existing

            with patch.object(LuxriotCaptureSession, "start", return_value=None):
                status = manager.start_session(channel_id=7, update_desired=True)

            self.assertTrue(existing.stopped)
            self.assertTrue(existing.cache_lock_was_free)
            self.assertEqual(status["channel_id"], 7)
            self.assertTrue(status["session_generation"])
            self.assertIsNot(manager.sessions[7], existing)

    def test_desired_live_session_updates_are_serialized_across_channels(self):
        class ObservedRuntimeStateStore(MemoryRuntimeStateStore):
            def __init__(self):
                super().__init__()
                self.guard = threading.Lock()
                self.active = 0
                self.max_active = 0

            def _enter(self):
                with self.guard:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.01)

            def _leave(self):
                with self.guard:
                    self.active -= 1

            def load_state(self, key):
                self._enter()
                try:
                    return super().load_state(key)
                finally:
                    self._leave()

            def save_state(self, key, payload):
                self._enter()
                try:
                    return super().save_state(key, payload)
                finally:
                    self._leave()

        with tempfile.TemporaryDirectory() as temp:
            state_store = ObservedRuntimeStateStore()
            manager = build_manager(Path(temp), runtime_state_store=state_store)
            state_store.max_active = 0
            workers = [
                threading.Thread(
                    target=manager._set_desired_live_session,
                    kwargs={"channel_id": channel_id, "enabled": True},
                )
                for channel_id in (7, 8)
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=2.0)

            desired = manager._load_desired_live_sessions()
            self.assertEqual(set(desired), {7, 8})
            self.assertEqual(state_store.max_active, 1)

    def test_desired_live_session_status_reads_use_the_runtime_cache(self):
        class CountingRuntimeStateStore(MemoryRuntimeStateStore):
            def __init__(self):
                super().__init__()
                self.load_count = 0

            def load_state(self, key):
                self.load_count += 1
                return super().load_state(key)

        with tempfile.TemporaryDirectory() as temp:
            state_store = CountingRuntimeStateStore()
            state_store.save_state(
                LuxriotManager.DESIRED_LIVE_SESSIONS_KEY,
                {
                    "version": 1,
                    "sessions": {"7": {"enabled": True}},
                },
            )
            manager = build_manager(
                Path(temp),
                runtime_state_store=state_store,
            )
            initial_load_count = state_store.load_count

            first = manager._load_desired_live_sessions()
            first[7]["enabled"] = False
            second = manager._load_desired_live_sessions()
            manager.streams_status()

            self.assertEqual(
                state_store.load_count,
                initial_load_count + 1,
            )
            self.assertTrue(second[7]["enabled"])

    def test_channel_model_hint_is_restored_from_summary_state(self):
        with tempfile.TemporaryDirectory() as temp:
            state_store = MemoryRuntimeStateStore()
            manager = build_manager(Path(temp), runtime_state_store=state_store)
            manager.channel_prompt_overrides[7] = {"model_hint": "vlm-a"}
            self.assertTrue(manager.persist_summary_state())

            restored = build_manager(Path(temp), runtime_state_store=state_store)

            self.assertEqual(restored.channel_prompt_overrides[7]["model_hint"], "vlm-a")

    def test_postgres_split_runtime_state_keeps_revision_and_road_calibration(self):
        class Cursor:
            def fetchall(self):
                return [
                    (
                        "luxriot_summary_state:meta",
                        {
                            "version": 2,
                            "revision": 7,
                            "updated_at": 123.0,
                            "road_scene_calibrations": {"7": {"confidence": "high"}},
                            "capture_baselines": {"7": {"level": 0.012, "buckets": 600}},
                            "prompt_settings": {"bookmark_enabled": False},
                        },
                    ),
                    (
                        "luxriot_summary_state:history:7",
                        {"logs": [{"batch_id": "legacy", "summary": "Legacy"}]},
                    ),
                    (
                        "luxriot_summary_state:history_item:7:abc",
                        {
                            "channel_id": "7",
                            "log": {"batch_id": "item", "summary": "New item"},
                        },
                    ),
                ]

        class Connection:
            def execute(self, *_args, **_kwargs):
                return Cursor()

        class Transaction:
            def __enter__(self):
                return Connection()

            def __exit__(self, *_args):
                return False

        class Pool:
            def transaction(self, *_args, **_kwargs):
                return Transaction()

        store = PostgresRuntimeStateStore(Pool(), uuid4())
        payload = store._load_split_summary_state()

        self.assertEqual(payload["revision"], 7)
        self.assertEqual(
            payload["road_scene_calibrations"]["7"]["confidence"],
            "high",
        )
        self.assertEqual(payload["capture_baselines"]["7"]["buckets"], 600)
        self.assertEqual(
            [entry["batch_id"] for entry in payload["summary_history"]["7"]],
            ["legacy", "item"],
        )

    def test_postgres_incremental_summary_history_does_not_rewrite_legacy_row(self):
        calls = []

        class Connection:
            def execute(self, query, params=None):
                calls.append((query, params))
                return SimpleNamespace(rowcount=1)

        class Transaction:
            def __enter__(self):
                return Connection()

            def __exit__(self, *_args):
                return False

        class Pool:
            def transaction(self, *_args, **_kwargs):
                return Transaction()

        store = PostgresRuntimeStateStore(Pool(), uuid4())
        store.save_state(
            "luxriot_summary_state",
            {
                "version": 2,
                "revision": 8,
                "updated_at": 123.0,
                "summary_history_storage": "upsert_items",
                "summary_history": {
                    "7": [
                        {
                            "channel_id": 7,
                            "run_id": "run-7",
                            "batch_id": "new-1",
                            "created_at": 123.0,
                            "frame_count": 2,
                            "summary": "New summary",
                        }
                    ]
                },
                "summary_runs": {},
            },
        )

        inserted_keys = [
            str(params[1])
            for query, params in calls
            if params and "INSERT INTO archive.runtime_state" in query
        ]
        self.assertIn("luxriot_summary_state:meta", inserted_keys)
        self.assertTrue(
            any(
                key.startswith("luxriot_summary_state:history_item:7:")
                for key in inserted_keys
            )
        )
        self.assertNotIn("luxriot_summary_state:history:7", inserted_keys)

    def test_postgres_runtime_state_bulk_promotes_rollups_with_queryable_keys(self):
        calls = []

        class Connection:
            def execute(self, query, params=None):
                calls.append((query, params))
                return SimpleNamespace(rowcount=1)

        class Transaction:
            def __enter__(self):
                return Connection()

            def __exit__(self, *_args):
                return False

        class Pool:
            def transaction(self, *_args, **_kwargs):
                return Transaction()

        store = PostgresRuntimeStateStore(Pool(), uuid4())
        written = store.save_rollups(
            [
                {
                    "rollup_id": "l1-ch112-w900-1783880100",
                    "channel_id": 112,
                    "level": "L1",
                    "window_start": 1_783_880_100.0,
                    "window_end": 1_783_881_000.0,
                    "summary": "semantic L1",
                },
                {
                    "rollup_id": "l2-ch112-w3600-1783879200",
                    "channel_id": 112,
                    "level": "L2",
                    "window_start": 1_783_879_200.0,
                    "window_end": 1_783_882_800.0,
                    "summary": "semantic L2",
                },
            ]
        )

        self.assertEqual(written, 2)
        keys = [str(params[1]) for _query, params in calls]
        self.assertEqual(
            keys,
            [
                "luxriot_rollup:112:l1:1783880100",
                "luxriot_rollup:112:l2:1783879200",
            ],
        )
        self.assertTrue(all("DO NOTHING" in query for query, _params in calls))
        self.assertTrue(all("DO UPDATE" not in query for query, _params in calls))

    def test_postgres_runtime_state_rollup_range_read_does_not_wait_for_writer_lock(self):
        entered_transaction = threading.Event()
        errors = []

        class Cursor:
            def fetchall(self):
                return []

        class Connection:
            def execute(self, *_args, **_kwargs):
                return Cursor()

        class Transaction:
            def __enter__(self):
                entered_transaction.set()
                return Connection()

            def __exit__(self, *_args):
                return False

        class Pool:
            def transaction(self, *_args, **_kwargs):
                return Transaction()

        store = PostgresRuntimeStateStore(Pool(), uuid4())

        def read_rollups():
            try:
                store.list_rollups(channel_id=112)
            except Exception as exc:  # pragma: no cover - assertion reports it
                errors.append(exc)

        store.lock.acquire()
        try:
            reader = threading.Thread(target=read_rollups)
            reader.start()
            reader.join(timeout=0.5)
            self.assertFalse(reader.is_alive())
            self.assertTrue(entered_transaction.is_set())
        finally:
            store.lock.release()
        reader.join(timeout=1.0)
        self.assertEqual(errors, [])

    def test_channel_inventory_refresh_failure_retains_and_marks_stale_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))

            class SuccessfulClient:
                channel_inventory_meta = {
                    "complete": True,
                    "completion": "explicit",
                    "payload_count": 2,
                }

                def get_channels(self):
                    return [{"id": 7, "title": "North"}]

            class FailingClient:
                channel_inventory_meta = {
                    "complete": False,
                    "completion": "error",
                    "error": "http://viewer:sample-password@camera.invalid/?token=sample-token",
                }

                def get_channels(self):
                    raise RuntimeError(
                        "refresh http://viewer:sample-password@camera.invalid/?token=sample-token"
                    )

            with patch.object(
                manager,
                "build_client",
                side_effect=[SuccessfulClient(), FailingClient()],
            ):
                fresh = manager.get_channels(force=True)
                with self.assertLogs("luxriot_connector", level="WARNING"):
                    stale = manager.get_channels(force=True)

            self.assertEqual(fresh, [{"id": 7, "title": "North"}])
            self.assertEqual(stale, fresh)
            stale[0]["title"] = "mutated by caller"
            self.assertEqual(manager.get_channels()[0]["title"], "North")
            status = manager.channel_inventory_status()
            self.assertTrue(status["stale"])
            self.assertEqual(status["count"], 1)
            self.assertNotIn("sample-password", status["last_error"])
            self.assertNotIn("sample-token", status["last_error"])
            self.assertNotIn("sample-password", status["stream"]["error"])
            self.assertNotIn("sample-token", status["stream"]["error"])

    def test_complete_inventory_stops_missing_and_disabled_live_work(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))

            class Client:
                channel_inventory_meta = {
                    "complete": True,
                    "completion": "explicit",
                }

                def get_channels(self):
                    return [
                        {
                            "id": 7,
                            "title": "Disabled",
                            "enabled": False,
                            "availability": "disabled",
                        },
                        {
                            "id": 9,
                            "title": "Available",
                            "enabled": True,
                            "availability": "available",
                        },
                    ]

            manager.sessions = {7: object(), 8: object()}
            manager.probe_sessions = {7: object()}
            with (
                patch.object(manager, "build_client", return_value=Client()),
                patch.object(manager, "stop_session") as stop_session,
                patch.object(manager, "stop_probe_capture") as stop_probe,
            ):
                channels = manager.get_channels(force=True)

            self.assertEqual([row["id"] for row in channels], [7, 9])
            self.assertEqual(
                {call.args[0] for call in stop_session.call_args_list},
                {7, 8},
            )
            stop_probe.assert_called_once_with(7, pause=False)
            reconciliation = manager.channel_inventory_status()["stream"][
                "reconciliation"
            ]
            self.assertEqual(
                reconciliation["stopped_channel_ids"],
                [7, 8],
            )
            self.assertEqual(
                reconciliation["unavailable_channel_ids"],
                [7],
            )
            self.assertEqual(
                reconciliation["missing_channel_ids"],
                [8],
            )

    def test_superseded_session_completion_has_no_archive_bookmark_state_or_history_side_effects(self):
        archived = []

        def summary_text(_messages, _model):
            return (
                "Current observed state:\n- Gate: Present. Vehicle at gate.\n"
                'ALERTS_JSON:\n{"alerts":[{"title":"Gate event","severity":"normal"}]}'
            )

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                lm_callback=summary_text,
                alert_parser=lambda *_args, **_kwargs: [
                    {"title": "Gate event", "description": "Vehicle at gate", "severity": "normal"}
                ],
                summary_archive_callback=lambda entry: archived.append(dict(entry)) or {"inserted": 1},
            )
            manager.default_bookmark_enabled = True
            manager.default_bookmark_cooldown_sec = 0.0

            with (
                patch.object(LuxriotCaptureSession, "start"),
                patch.object(LuxriotCaptureSession, "stop"),
            ):
                manager.start_session(7, batch_size=12, prompt="Describe.")
                old_session = manager.sessions[7]
                batch = manager.create_summary_batch(
                    channel_id=7,
                    run_id=old_session.run_id,
                    batch_size=2,
                    prompt="Describe.",
                    model_hint=None,
                    interval_sec=1.0,
                    frames=sample_frames(),
                    session_generation=old_session.session_generation,
                )
                completed_entry = manager.run_summary_batch(batch)
                manager.start_session(7, batch_size=12, prompt="Describe newer run.")
                new_session = manager.sessions[7]

            sent = []
            with patch.object(
                manager,
                "send_bookmark_event",
                side_effect=lambda **kwargs: sent.append(kwargs) or {"success": True},
            ):
                accepted = manager.accept_summary_entry(completed_entry)

            self.assertFalse(accepted["accepted"])
            self.assertTrue(accepted["stale_session"])
            self.assertTrue(accepted["side_effects_skipped"])
            self.assertNotEqual(old_session.session_generation, new_session.session_generation)
            self.assertEqual(sent, [])
            self.assertEqual(archived, [])
            self.assertEqual(manager.summary_history.get(7, []), [])
            self.assertEqual(manager.channel_observed_state_tracker.get(7, {}), {})
            self.assertEqual(new_session.logs, [])

    def test_generation_is_rechecked_after_lm_admission_before_expensive_call(self):
        admitted = threading.Event()
        release = threading.Event()
        expensive_calls = []
        outcomes = []

        def admission_aware_lm(_messages, _model, *, preflight=None):
            admitted.set()
            release.wait(timeout=2.0)
            if preflight is not None:
                preflight()
            expensive_calls.append("network")
            return "summary"

        admission_aware_lm.eva_generation_preflight = True

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp), lm_callback=admission_aware_lm)
            with (
                patch.object(LuxriotCaptureSession, "start", return_value=None),
                patch.object(LuxriotCaptureSession, "stop", return_value=None),
            ):
                manager.start_session(7, batch_size=2, prompt="Old generation.")
                old_session = manager.sessions[7]
                batch = manager.create_summary_batch(
                    channel_id=7,
                    run_id=old_session.run_id,
                    batch_size=2,
                    prompt="Describe.",
                    model_hint=None,
                    interval_sec=1.0,
                    frames=sample_frames(),
                    session_generation=old_session.session_generation,
                )

                def run_batch():
                    outcomes.append(manager.dispatch_summary_batch(batch))

                worker = threading.Thread(target=run_batch, daemon=True)
                worker.start()
                self.assertTrue(admitted.wait(timeout=1.0))
                manager.start_session(7, batch_size=2, prompt="Current generation.")

            release.set()
            worker.join(timeout=2.0)

            self.assertEqual(expensive_calls, [])
            self.assertEqual(len(outcomes), 1)
            self.assertFalse(outcomes[0]["accepted"])
            self.assertEqual(outcomes[0]["status"], "superseded")

    def test_ffmpeg_error_status_redacts_url_credentials(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            session = LuxriotCaptureSession(
                manager,
                channel_id=7,
                batch_size=2,
                prompt="Describe.",
                run_id="run-7",
            )
            response = SimpleNamespace(
                headers={"Content-Type": "video/mp4"},
                iter_content=lambda _chunk_size: iter([b"bounded-video-segment"]),
                close=lambda: None,
            )
            session.client = SimpleNamespace(
                base_url="http://camera.invalid",
                username="viewer",
                password="sample-password",
                open_live_stream=lambda *_args, **_kwargs: response,
            )
            class FakePipe:
                def __init__(self):
                    self.closed = False

                def write(self, data):
                    return len(data)

                def close(self):
                    self.closed = True

            class FakeProcess:
                def __init__(self, _command, *, stderr, **_kwargs):
                    self.stdin = FakePipe()
                    self.stdout = SimpleNamespace(read=lambda _size: b"")
                    self.returncode = None
                    stderr.write(
                        b"unable to open http://viewer:sample-password@camera.invalid/live"
                        b"?access_token=sample-token"
                    )
                    stderr.flush()

                def poll(self):
                    return self.returncode

                def wait(self, timeout=None):
                    self.returncode = 1
                    return self.returncode

                def kill(self):
                    self.returncode = -9

            with patch("luxriot_connector.subprocess.Popen", side_effect=FakeProcess) as ffmpeg_open:
                handled = session._run_ffmpeg_live_segment_once()

            self.assertFalse(handled)
            self.assertNotIn("sample-password", session.last_live_segment_error)
            self.assertNotIn("sample-token", session.last_live_segment_error)
            self.assertIn("<redacted>", session.last_live_segment_error)
            command = [str(arg) for arg in ffmpeg_open.call_args.args[0]]
            command_text = " ".join(command)
            self.assertNotIn("viewer", command_text)
            self.assertNotIn("sample-password", command_text)
            self.assertNotIn("camera.invalid", command_text)
            self.assertNotIn("Authorization", command_text)
            self.assertIn("pipe:0", command_text)
            self.assertNotIn("segment.mp4", command_text)

    def test_live_segment_deadline_starts_again_when_authenticated_media_opens(self):
        started_at = 100.0
        timeout_sec = 67.0

        self.assertEqual(
            LuxriotCaptureSession._live_segment_process_deadline(
                started_at,
                timeout_sec,
            ),
            167.0,
        )
        self.assertEqual(
            LuxriotCaptureSession._live_segment_process_deadline(
                started_at,
                timeout_sec,
                108.0,
            ),
            175.0,
        )
        self.assertEqual(
            LuxriotCaptureSession._live_segment_process_deadline(
                started_at,
                timeout_sec,
                "invalid",
            ),
            167.0,
        )

    def test_live_segment_timeouts_allow_backpressure_but_bound_stalls(self):
        hard_timeout, idle_timeout = (
            LuxriotCaptureSession._live_segment_process_timeouts(
                segment_seconds=60.0,
                frame_limit=120,
                fps=2.0,
                read_timeout_sec=5.0,
            )
        )

        self.assertEqual(hard_timeout, 127.0)
        self.assertEqual(idle_timeout, 15.0)

    def test_ffmpeg_uses_intel_qsv_or_vaapi_decode_when_probe_passes(self):
        commands = []

        class FakePipe:
            def write(self, data):
                return len(data)

            def close(self):
                return None

        encoded = BytesIO()
        Image.new("RGB", (24, 16), color=(50, 80, 120)).save(encoded, format="JPEG")
        jpeg = encoded.getvalue()

        class FakeStdout:
            def __init__(self):
                self.sent = False

            def read(self, _size):
                if self.sent:
                    return b""
                self.sent = True
                return jpeg

        class FakeProcess:
            def __init__(self, command, **_kwargs):
                commands.append([str(item) for item in command])
                self.stdin = FakePipe()
                self.stdout = FakeStdout()
                self.returncode = None

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                self.returncode = 0
                return 0

            def kill(self):
                self.returncode = -9

        class Response:
            headers = {}

            def iter_content(self, _chunk_size):
                yield b"video-bytes"

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp), config_overrides={"LUXRIOT_LIVE_SEGMENT_FPS": 1.0})
            session = LuxriotCaptureSession(manager, channel_id=7, batch_size=2, prompt="Describe.", run_id="run-7")
            session.client = SimpleNamespace(open_live_stream=lambda *_args, **_kwargs: Response())
            with (
                patch("luxriot_connector._intel_hwaccel_for_config", return_value=("qsv", "/dev/dri/renderD128", None)),
                patch("luxriot_connector.subprocess.Popen", side_effect=FakeProcess),
            ):
                handled = session._run_ffmpeg_live_segment_once()

            vaapi_session = LuxriotCaptureSession(manager, channel_id=8, batch_size=2, prompt="Describe.", run_id="run-8")
            vaapi_session.client = SimpleNamespace(open_live_stream=lambda *_args, **_kwargs: Response())
            with (
                patch("luxriot_connector._intel_hwaccel_for_config", return_value=("vaapi", "/dev/dri/renderD128", None)),
                patch("luxriot_connector.subprocess.Popen", side_effect=FakeProcess),
            ):
                vaapi_handled = vaapi_session._run_ffmpeg_live_segment_once()

        self.assertTrue(handled)
        self.assertTrue(vaapi_handled)
        self.assertEqual(len(commands), 2)
        command = commands[0]
        self.assertIn("qsv=eva_qsv:/dev/dri/renderD128", command)
        self.assertEqual(command[command.index("-hwaccel") + 1], "qsv")
        self.assertIn("vpp_qsv=framerate=1/1:format=nv12", command[command.index("-vf") + 1])
        self.assertEqual(session.status()["live_segment_decoder"], "intel_qsv")
        self.assertTrue(session.status()["live_segment_hwaccel_available"])
        vaapi_command = commands[1]
        self.assertIn("vaapi=eva_va:/dev/dri/renderD128", vaapi_command)
        self.assertEqual(vaapi_command[vaapi_command.index("-hwaccel") + 1], "vaapi")
        self.assertIn("scale_vaapi=w=800:h=800", vaapi_command[vaapi_command.index("-vf") + 1])
        self.assertEqual(vaapi_session.status()["live_segment_decoder"], "intel_vaapi")

    def test_ffmpeg_qsv_failure_retries_once_in_software(self):
        commands = []

        class FakePipe:
            def write(self, data):
                return len(data)

            def close(self):
                return None

        encoded = BytesIO()
        Image.new("RGB", (24, 16), color=(30, 60, 90)).save(encoded, format="JPEG")
        jpeg = encoded.getvalue()

        class FakeProcess:
            def __init__(self, command, *, stderr, **_kwargs):
                self.command = [str(item) for item in command]
                commands.append(self.command)
                self.stdin = FakePipe()
                self.returncode = None
                if "-hwaccel" in self.command:
                    stderr.write(b"QSV device initialization failed")
                    stderr.flush()
                    self.stdout = SimpleNamespace(read=lambda _size: b"")
                    self.exit_code = 1
                else:
                    sent = {"value": False}

                    def read(_size):
                        if sent["value"]:
                            return b""
                        sent["value"] = True
                        return jpeg

                    self.stdout = SimpleNamespace(read=read)
                    self.exit_code = 0

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                self.returncode = self.exit_code
                return self.returncode

            def kill(self):
                self.returncode = -9

        class Response:
            headers = {}

            def iter_content(self, _chunk_size):
                yield b"video-bytes"

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp), config_overrides={"LUXRIOT_LIVE_SEGMENT_FPS": 1.0})
            session = LuxriotCaptureSession(manager, channel_id=7, batch_size=2, prompt="Describe.", run_id="run-7")
            session.client = SimpleNamespace(open_live_stream=lambda *_args, **_kwargs: Response())
            with (
                patch("luxriot_connector._intel_hwaccel_for_config", return_value=("qsv", "/dev/dri/renderD128", None)),
                patch("luxriot_connector.subprocess.Popen", side_effect=FakeProcess),
            ):
                handled = session._run_ffmpeg_live_segment_once()

        self.assertTrue(handled)
        self.assertEqual(len(commands), 2)
        self.assertIn("-hwaccel", commands[0])
        self.assertNotIn("-hwaccel", commands[1])
        status = session.status()
        self.assertEqual(status["live_segment_decoder"], "software")
        self.assertEqual(status["live_segment_hwaccel_fallback_count"], 1)
        self.assertIn("QSV device initialization failed", status["live_segment_hwaccel_probe_error"])

    def test_ffmpeg_dense_budget_represents_full_batch_window_and_only_apex_reaches_sinks(self):
        probe_calls = []
        vlm_frames = []
        archive_entries = []
        commands = []
        inflight_snapshots = []
        first_summary = threading.Event()
        summary_observations = []
        process_holder = {}
        source_anchor_ms = 1_700_000_000_000

        class RecordingProbeManager:
            def add_frame(self, channel_id, image, timestamp_ms, provenance=None):
                level = int(round(float(np.asarray(image, dtype=np.float32)[..., 0].mean())))
                probe_calls.append(
                    {
                        "channel_id": channel_id,
                        "level": level,
                        "timestamp_ms": timestamp_ms,
                        "provenance": dict(provenance or {}),
                    }
                )

        class FakePipe:
            def __init__(self):
                self.closed = False
                self.bytes_written = 0

            def write(self, data):
                count = len(data)
                self.bytes_written += count
                return count

            def close(self):
                self.closed = True

        class FakeProcess:
            def __init__(self, command, **_kwargs):
                commands.append([str(item) for item in command])
                inflight_snapshots.append(session.status())
                self.stdin = FakePipe()
                self.returncode = None
                frame_limit = int(command[command.index("-frames:v") + 1])
                jpeg_frames = []
                for index in range(frame_limit):
                    second = index // 3
                    phase = index % 3
                    levels = (20, 240, 230) if second % 2 == 0 else (220, 0, 10)
                    encoded = BytesIO()
                    Image.new("RGB", (24, 16), color=(levels[phase],) * 3).save(
                        encoded,
                        format="JPEG",
                        quality=100,
                        subsampling=0,
                    )
                    jpeg_frames.append(encoded.getvalue())

                class FakeStdout:
                    def __init__(self, frames):
                        self.frames = list(frames)
                        self.index = 0

                    def read(self, _size):
                        if self.index >= len(self.frames):
                            return b""
                        # Keep both stdout and the authenticated response open
                        # until the first 12-apex summary has been dispatched.
                        if self.index >= 40:
                            first_summary.wait(timeout=2.0)
                        value = self.frames[self.index]
                        self.index += 1
                        return value

                self.stdout = FakeStdout(jpeg_frames)
                process_holder["process"] = self

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                self.returncode = 0
                return self.returncode

            def kill(self):
                self.returncode = -9

        class StreamResponse:
            def __init__(self):
                self.closed = False
                self.headers = {"X-Stream-Start-Time": str(source_anchor_ms)}

            def iter_content(self, chunk_size):
                self.chunk_size = chunk_size
                for index in range(45):
                    if index >= 40:
                        first_summary.wait(timeout=2.0)
                    yield b"authenticated-video-bytes" * 8

            def close(self):
                self.closed = True

        def fake_lm(_messages, _model):
            return "Deterministic local summary."

        def archive_callback(entry):
            archived = dict(entry)
            archive_entries.append(archived)
            frames = list(archived.get("archive_frames") or [])
            return {
                "attempted": len(frames),
                "inserted": len(frames),
                "summary_frames": len(frames),
                "alert_frames": 0,
            }

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                lm_callback=fake_lm,
                summary_archive_callback=archive_callback,
                config_overrides={
                    "LUXRIOT_VECTOR_SIGNALS_ENABLED": False,
                    "LUXRIOT_LIVE_SEGMENT_FPS": 3.0,
                    "LUXRIOT_LIVE_SEGMENT_SECONDS": 15.0,
                    "LUXRIOT_LIVE_SEGMENT_MB": 8.0,
                },
            )
            manager.probe_manager = RecordingProbeManager()
            manager.shared_probe_channels.add(7)
            manager.jpeg_encoder = lambda image, **_kwargs: (
                f"level-{int(round(float(np.asarray(image, dtype=np.float32)[..., 0].mean())))}"
            )

            def message_builder(_channel, frames, _prompt, _system_prompt):
                vlm_frames.extend(dict(frame) for frame in frames)
                process = process_holder["process"]
                summary_observations.append(
                    {
                        "process_running": process.poll() is None,
                        "response_closed": response.closed,
                        "probe_count": len(probe_calls),
                        "inflight_frames": session.status()["live_segment_inflight_frames"],
                        "inflight_represented_seconds": session.status()["live_segment_inflight_represented_seconds"],
                    }
                )
                first_summary.set()
                return [{"frame_count": len(frames)}]

            manager.message_builder = message_builder
            response = StreamResponse()
            session = LuxriotCaptureSession(
                manager,
                channel_id=7,
                batch_size=12,
                prompt="Describe.",
                run_id="run-7",
                interval_override=1.0,
            )
            session.client = SimpleNamespace(
                base_url="http://camera.invalid",
                username="viewer",
                password="never-in-argv",
                open_live_stream=lambda *_args, **_kwargs: response,
            )
            session.last_live_segment_represented_seconds = 9.0
            manager.sessions[7] = session

            with patch("luxriot_connector.subprocess.Popen", side_effect=FakeProcess):
                handled = session._run_ffmpeg_live_segment_once()

            self.assertTrue(handled)
            self.assertTrue(response.closed)
            self.assertEqual(len(commands), 1)
            command = commands[0]
            self.assertEqual(command[command.index("-frames:v") + 1], "45")
            self.assertEqual(command[command.index("-i") + 1], "pipe:0")
            self.assertEqual(command[-5:], ["-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"])
            video_filter = command[command.index("-vf") + 1]
            self.assertIn("fps=3", video_filter)
            self.assertIn("scale=800:800:force_original_aspect_ratio=decrease", video_filter)
            command_text = " ".join(command)
            self.assertNotIn("viewer", command_text)
            self.assertNotIn("never-in-argv", command_text)
            self.assertNotIn("camera.invalid", command_text)
            self.assertTrue(inflight_snapshots[0]["live_segment_inflight"])
            self.assertEqual(inflight_snapshots[0]["last_live_segment_represented_seconds"], 9.0)
            self.assertEqual(inflight_snapshots[0]["live_segment_inflight_raw_frame_budget"], 45)
            self.assertEqual(inflight_snapshots[0]["live_segment_inflight_frames"], 0)
            self.assertEqual(inflight_snapshots[0]["live_segment_inflight_represented_seconds"], 0.0)
            self.assertEqual(summary_observations, [
                {
                    "process_running": True,
                    "response_closed": False,
                    "probe_count": 12,
                    "inflight_frames": 37,
                    "inflight_represented_seconds": 12.333,
                }
            ])

            status = session.status()
            self.assertFalse(status["live_segment_inflight"])
            self.assertEqual(status["live_segment_inflight_frames"], 0)
            self.assertEqual(status["live_segment_inflight_represented_seconds"], 0.0)
            self.assertEqual(status["last_live_segment_raw_frame_budget"], 45)
            self.assertEqual(status["last_live_segment_target_seconds"], 15.0)
            self.assertEqual(status["last_live_segment_summary_target_seconds"], 12.0)
            self.assertEqual(status["last_live_segment_represented_seconds"], 15.0)
            self.assertGreaterEqual(status["last_live_segment_byte_budget"], 60 * 1024 * 1024)
            self.assertEqual(status["live_segment_frame_count"], 45)
            self.assertEqual(status["capture_apex_raw_frame_count"], 45)
            self.assertEqual(status["capture_apex_selected_count"], 15)
            self.assertEqual(status["capture_apex_probe_dispatch_count"], 15)
            self.assertEqual(status["last_live_segment_source_start_timestamp_ms"], source_anchor_ms)
            self.assertEqual(
                status["last_live_segment_last_source_timestamp_ms"],
                source_anchor_ms + 14_667,
            )
            self.assertEqual(status["last_live_segment_timestamp_source"], "evo_x_stream_start_time")
            self.assertEqual(len(probe_calls), 15)
            self.assertTrue(all(len(call["provenance"]["source_frame_indices"]) == 3 for call in probe_calls))
            self.assertEqual(
                [call["timestamp_ms"] for call in probe_calls],
                [source_anchor_ms + 333 + second * 1000 for second in range(15)],
            )
            self.assertEqual(len(vlm_frames), 4)
            self.assertTrue(all(frame["thumbnail"] in {"level-0", "level-240"} for frame in vlm_frames))
            self.assertEqual(len(archive_entries), 1)
            self.assertEqual(archive_entries[0]["source_frame_count"], 12)
            self.assertEqual(archive_entries[0]["selected_frame_count"], 4)
            self.assertTrue(
                all(
                    frame["thumbnail"] in {"level-0", "level-240"}
                    for frame in archive_entries[0]["archive_frames"]
                )
            )
            self.assertEqual(status["pending_frames"], 3)
            continued_timestamp = session._next_live_source_timestamp_ms(
                source_anchor_ms=source_anchor_ms - 60_000,
                frame_index=0,
                fps=3.0,
            )
            self.assertGreater(continued_timestamp, source_anchor_ms + 14_667)

    def test_live_segment_without_source_clock_uses_decode_observation_time(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            session = LuxriotCaptureSession(
                manager,
                channel_id=7,
                batch_size=2,
                prompt="Describe.",
                run_id="run-7",
            )

            # Several decoder-startup frames can arrive in one wall-clock
            # tick. Keep them ordered without spreading that backlog into the
            # future or inheriting the media-open delay.
            self.assertEqual(
                session._next_live_observed_timestamp_ms(20_000),
                20_000,
            )
            self.assertEqual(
                session._next_live_observed_timestamp_ms(20_000),
                20_001,
            )
            self.assertEqual(
                session._next_live_observed_timestamp_ms(20_500),
                20_500,
            )

    def test_ffmpeg_incremental_window_stop_actively_breaks_feeder_and_stdout(self):
        response_started = threading.Event()
        process_killed = threading.Event()
        outcomes = []

        class FakePipe:
            def __init__(self):
                self.closed = False

            def write(self, data):
                if self.closed:
                    raise BrokenPipeError("closed")
                return len(data)

            def close(self):
                self.closed = True

        class BlockingStdout:
            def read(self, _size):
                process_killed.wait(timeout=2.0)
                return b""

        class BlockingProcess:
            def __init__(self, _command, **_kwargs):
                self.stdin = FakePipe()
                self.stdout = BlockingStdout()
                self.returncode = None

            def poll(self):
                return self.returncode

            def wait(self, timeout=None):
                process_killed.wait(timeout=timeout or 2.0)
                if self.returncode is None:
                    raise TimeoutError("still running")
                return self.returncode

            def kill(self):
                self.returncode = -9
                process_killed.set()

        class BlockingResponse:
            def __init__(self):
                self.closed = False
                self.headers = {}

            def iter_content(self, _chunk_size):
                response_started.set()
                while not self.closed:
                    yield b"stream-bytes" * 32
                    time.sleep(0.005)

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                config_overrides={
                    "LUXRIOT_LIVE_SEGMENT_FPS": 2.0,
                    "LUXRIOT_LIVE_SEGMENT_SECONDS": 60.0,
                },
            )
            session = LuxriotCaptureSession(
                manager,
                channel_id=7,
                batch_size=12,
                prompt="Describe.",
                run_id="run-7",
                interval_override=1.0,
            )
            response = BlockingResponse()
            session.client = SimpleNamespace(
                open_live_stream=lambda *_args, **_kwargs: response,
            )

            with patch("luxriot_connector.subprocess.Popen", side_effect=BlockingProcess):
                worker = threading.Thread(
                    target=lambda: outcomes.append(session._run_ffmpeg_live_segment_once()),
                    daemon=True,
                )
                worker.start()
                self.assertTrue(response_started.wait(timeout=1.0))
                stopped_at = time.monotonic()
                session.stop_event.set()
                worker.join(timeout=1.5)

            self.assertFalse(worker.is_alive())
            self.assertLess(time.monotonic() - stopped_at, 1.5)
            self.assertEqual(outcomes, [False])
            self.assertTrue(response.closed)
            self.assertTrue(process_killed.is_set())
            status = session.status()
            self.assertFalse(status["live_segment_inflight"])
            self.assertIsNone(status["last_live_segment_error"])
            self.assertEqual(status["live_segment_failed_count"], 0)

    def test_incremental_summary_slicing_dispatches_exact_batches_and_keeps_remainder(self):
        dispatched = []
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                config_overrides={"LUXRIOT_VECTOR_SIGNALS_ENABLED": False},
            )
            manager.set_summary_dispatcher(
                lambda batch, workload: dispatched.append(
                    {
                        "workload": workload,
                        "thumbnails": [frame["thumbnail"] for frame in batch["frames"]],
                        "source_frame_count": batch["source_frame_count"],
                        "selected_frame_count": batch["selected_frame_count"],
                    }
                )
                or {"queued": False, "accepted": True}
            )
            session = LuxriotCaptureSession(
                manager,
                channel_id=7,
                batch_size=12,
                prompt="Describe.",
                run_id="run-7",
                interval_override=1.0,
            )
            session.frames = [
                {
                    "thumbnail": f"frame-{index}",
                    "captured_at": 100.0 + index,
                    "time_sec": 100.0 + index,
                    "width": 24,
                    "height": 16,
                }
                for index in range(25)
            ]

            session._summarize_if_ready()

            self.assertEqual([len(item["thumbnails"]) for item in dispatched], [4, 4])
            self.assertEqual(
                dispatched[0]["thumbnails"],
                ["frame-0", "frame-1", "frame-6", "frame-11"],
            )
            self.assertEqual(
                dispatched[1]["thumbnails"],
                ["frame-12", "frame-13", "frame-18", "frame-23"],
            )
            self.assertEqual(
                [item["source_frame_count"] for item in dispatched],
                [12, 12],
            )
            self.assertEqual(
                [item["selected_frame_count"] for item in dispatched],
                [4, 4],
            )
            self.assertEqual([frame["thumbnail"] for frame in session.frames], ["frame-24"])

    def test_summary_deadline_dispatches_without_a_followup_capture_frame(self):
        dispatched = []
        completed = threading.Event()
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                config_overrides={"LUXRIOT_VECTOR_SIGNALS_ENABLED": False},
            )

            def dispatch(batch, workload):
                dispatched.append((batch, workload))
                completed.set()
                return {"queued": False, "accepted": True}

            manager.set_summary_dispatcher(dispatch)
            session = LuxriotCaptureSession(
                manager,
                channel_id=7,
                batch_size=12,
                prompt="Describe.",
                run_id="run-deadline",
                interval_override=1.0,
            )
            session.summary_max_window_sec = 0.05
            session.summary_worker_thread.start()
            try:
                with session.lock:
                    session._admit_summary_frame_locked(
                        {
                            "thumbnail": "only-frame",
                            "captured_at": 100.0,
                            "time_sec": 100.0,
                            "width": 24,
                            "height": 16,
                        }
                    )

                self.assertTrue(completed.wait(timeout=1.5))
                self.assertEqual(len(dispatched), 1)
                self.assertEqual(
                    [frame["thumbnail"] for frame in dispatched[0][0]["frames"]],
                    ["only-frame"],
                )
                self.assertEqual(session.frames, [])
            finally:
                session.stop_event.set()
                with session.summary_condition:
                    session.summary_condition.notify_all()
                session.summary_worker_thread.join(timeout=1.0)

    def test_capture_cv_apex_is_the_same_frame_sent_to_clip_vlm_and_archive(self):
        probe_calls = []
        vlm_frames = []
        encoded_markers = []

        class RecordingProbeManager:
            def add_frame(self, channel_id, image, timestamp_ms, provenance=None):
                probe_calls.append(
                    {
                        "channel_id": channel_id,
                        "marker": image.info.get("marker"),
                        "timestamp_ms": timestamp_ms,
                        "provenance": dict(provenance or {}),
                    }
                )

        def image(level, marker):
            frame = Image.new("RGB", (24, 16), color=(level, level, level))
            frame.info["marker"] = marker
            return frame

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                config_overrides={
                    "LUXRIOT_VECTOR_SIGNALS_ENABLED": False,
                    # This test exercises two consecutive embedding snapshots;
                    # production now honours the configured snapshot cadence.
                    "LUXRIOT_SNAPSHOT_INTERVAL": 1,
                },
            )
            def encode_selected(frame, **_kwargs):
                encoded_markers.append(frame.info.get("marker"))
                return f"hash-{frame.info.get('marker')}"

            manager.jpeg_encoder = encode_selected
            manager.probe_manager = RecordingProbeManager()
            manager.shared_probe_channels.add(7)

            def message_builder(_channel, frames, _prompt, _system_prompt):
                vlm_frames.extend(dict(frame) for frame in frames)
                return [{"frame_count": len(frames)}]

            manager.message_builder = message_builder
            session = LuxriotCaptureSession(
                manager,
                channel_id=7,
                batch_size=2,
                prompt="Describe.",
                run_id="run-7",
            )

            session._accept_captured_frame(image(0, "early"), 100_100, summarize=False)
            session._accept_captured_frame(image(255, "apex"), 100_500, summarize=False)
            session._accept_captured_frame(image(255, "late"), 100_800, summarize=False)
            self.assertEqual(probe_calls, [])
            session._accept_captured_frame(image(0, "next"), 101_100, summarize=False)

            self.assertEqual([call["marker"] for call in probe_calls], ["apex"])
            self.assertEqual(encoded_markers, ["apex"])
            self.assertEqual(probe_calls[0]["timestamp_ms"], 100_500)
            self.assertEqual(probe_calls[0]["provenance"]["selected_frame_hash"], "664fb8e9440850ff")
            self.assertEqual(
                probe_calls[0]["provenance"]["frame_hash_source"],
                "normalized_grayscale_sha1",
            )
            self.assertEqual(probe_calls[0]["provenance"]["selected_source_frame_index"], 2)
            self.assertEqual(probe_calls[0]["provenance"]["selection_source"], "capture_cv_frame_delta")
            self.assertEqual(probe_calls[0]["provenance"]["source_frame_indices"], [1, 2, 3])

            session._flush_capture_apex_bucket()
            self.assertEqual([call["marker"] for call in probe_calls], ["apex", "next"])
            self.assertEqual(encoded_markers, ["apex", "next"])
            batch = manager.create_summary_batch(
                channel_id=7,
                run_id="run-7",
                batch_size=2,
                prompt="Describe.",
                model_hint=None,
                interval_sec=1.0,
                frames=list(session.frames),
            )
            entry = manager.run_summary_batch(batch)

            self.assertEqual([frame["thumbnail"] for frame in vlm_frames], ["hash-apex", "hash-next"])
            self.assertEqual(batch["frame_selection"]["groups"][0]["selection_source"], "capture_cv_frame_delta")
            self.assertEqual(batch["frame_selection"]["groups"][0]["selected_source_frame_index"], 2)
            self.assertEqual(entry["archive_frames"][0]["frame_hash"], "664fb8e9440850ff")
            self.assertEqual(entry["archive_frames"][0]["selection_source"], "capture_cv_frame_delta")
            self.assertEqual(entry["archive_frames"][0]["source_frame_index"], 2)

    def test_browser_media_close_and_channel_switch_do_not_interrupt_analytics_apex_fanout(self):
        import oldapp

        probe_calls = []
        vlm_frames = []
        archive_entries = []
        lm_calls = []

        class RecordingProbeManager:
            def add_frame(self, channel_id, image, timestamp_ms, provenance=None):
                probe_calls.append(
                    {
                        "channel_id": channel_id,
                        "marker": image.info.get("marker"),
                        "timestamp_ms": timestamp_ms,
                        "provenance": dict(provenance or {}),
                    }
                )

        class AnalyticsClient:
            def __init__(self):
                self.closed = False

        class MediaUpstream:
            def __init__(self, marker):
                self.marker = marker
                self.status_code = 200
                self.headers = {"Content-Type": "video/mp4"}
                self.closed = False

            def iter_content(self, chunk_size=None):
                self.chunk_size = chunk_size
                yield b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isomiso2"
                yield f"tail-{self.marker}".encode("ascii")

            def close(self):
                self.closed = True

        class MediaClient:
            def __init__(self, upstream):
                self.upstream = upstream
                self.calls = []

            def _request(self, method, path, **kwargs):
                self.calls.append((method, path, kwargs))
                return self.upstream

        def frame(level, marker):
            image = Image.new("RGB", (24, 16), color=(level, level, level))
            image.info["marker"] = marker
            return image

        def fake_lm(messages, model_hint):
            lm_calls.append({"messages": list(messages), "model_hint": model_hint})
            return "No alert-worthy change."

        def archive_callback(entry):
            archived = dict(entry)
            archive_entries.append(archived)
            archive_frames = list(archived.get("archive_frames") or [])
            return {
                "attempted": len(archive_frames),
                "inserted": len(archive_frames),
                "summary_frames": len(archive_frames),
                "alert_frames": 0,
            }

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                lm_callback=fake_lm,
                summary_archive_callback=archive_callback,
                config_overrides={"LUXRIOT_VECTOR_SIGNALS_ENABLED": False},
            )
            manager.jpeg_encoder = lambda image, **_kwargs: f"jpeg-{image.info.get('marker')}"
            manager.probe_manager = RecordingProbeManager()
            manager.shared_probe_channels.add(7)

            def message_builder(_channel, frames, _prompt, _system_prompt):
                vlm_frames.extend(dict(item) for item in frames)
                return [{"frame_count": len(frames)}]

            manager.message_builder = message_builder
            analytics_client = AnalyticsClient()
            dead_media_upstream = MediaUpstream("closed-browser")
            switched_media_upstream = MediaUpstream("switched-channel")
            media_clients = iter(
                [
                    analytics_client,
                    MediaClient(dead_media_upstream),
                    MediaClient(switched_media_upstream),
                ]
            )
            manager.build_client = lambda: next(media_clients)  # type: ignore[method-assign]
            session = LuxriotCaptureSession(
                manager,
                channel_id=7,
                batch_size=2,
                prompt="Describe.",
                run_id="run-7",
                interval_override=0.2,
            )
            manager.sessions[7] = session

            session._accept_captured_frame(frame(0, "before-base"), 100_100, summarize=False)
            session._accept_captured_frame(frame(200, "before-apex"), 100_500, summarize=False)

            with (
                patch.object(oldapp, "luxriot_manager", manager),
                patch.object(oldapp.config, "AUTH_ENABLED", False),
                oldapp.app.test_client() as client,
            ):
                browser_response = client.get(
                    "/luxriot/media/live/7?stream=mainStream",
                    buffered=False,
                )
                self.assertEqual(browser_response.status_code, 200)
                self.assertTrue(next(iter(browser_response.response)).startswith(b"\x00\x00\x00\x18ftyp"))
                browser_response.close()

                session._accept_captured_frame(frame(210, "before-late"), 100_800, summarize=False)
                session._accept_captured_frame(frame(220, "after-base"), 101_100, summarize=False)

                switched_response = client.get(
                    "/luxriot/media/live/8?stream=mainStream",
                    buffered=False,
                )
                self.assertEqual(switched_response.status_code, 200)
                self.assertTrue(next(iter(switched_response.response)).startswith(b"\x00\x00\x00\x18ftyp"))
                switched_response.close()

            session._accept_captured_frame(frame(0, "after-apex"), 101_500, summarize=False)
            session._accept_captured_frame(frame(10, "after-late"), 101_800, summarize=False)
            session._accept_captured_frame(frame(20, "next-bucket"), 102_100, summarize=False)

            self.assertIs(session.client, analytics_client)
            self.assertIs(manager.sessions[7], session)
            self.assertFalse(session.stop_event.is_set())
            self.assertFalse(analytics_client.closed)
            self.assertTrue(dead_media_upstream.closed)
            self.assertTrue(switched_media_upstream.closed)
            self.assertEqual(
                [call["marker"] for call in probe_calls],
                ["before-apex", "after-apex"],
            )
            self.assertEqual(
                [call["provenance"]["bucket_start_ms"] for call in probe_calls],
                [100_000, 101_000],
            )
            self.assertEqual(session.status()["capture_apex_selected_count"], 2)
            self.assertEqual(session.status()["capture_apex_pending_frames"], 1)
            self.assertEqual(session.status()["capture_apex_probe_dispatch_count"], 2)
            self.assertEqual(session.status()["capture_apex_probe_failure_count"], 0)

            batch = manager.create_summary_batch(
                channel_id=7,
                run_id="run-7",
                batch_size=2,
                prompt="Describe.",
                model_hint=None,
                interval_sec=0.2,
                frames=list(session.frames),
            )
            accepted = manager.accept_summary_entry(manager.run_summary_batch(batch))

            self.assertTrue(accepted["accepted"])
            self.assertEqual(len(lm_calls), 1)
            self.assertEqual(
                [item["thumbnail"] for item in vlm_frames],
                ["jpeg-before-apex", "jpeg-after-apex"],
            )
            self.assertEqual(len(archive_entries), 1)
            self.assertEqual(
                [item["thumbnail"] for item in archive_entries[0]["archive_frames"]],
                ["jpeg-before-apex", "jpeg-after-apex"],
            )
            stream = manager.streams_status()["video_streams"][0]
            self.assertEqual(stream["capture_apex_selected_count"], 2)
            self.assertEqual(stream["last_source_frame_count"], 6)
            self.assertEqual(stream["last_selected_frame_count"], 2)
            self.assertEqual(stream["capture_apex_probe_dispatch_count"], 2)
            self.assertEqual(stream["last_archive_attempted"], 2)
            self.assertEqual(stream["last_archive_inserted"], 2)
            self.assertEqual(stream["last_frame_selection"]["policy"], "per_second_attention_apex_v1")
            self.assertEqual(manager.summary_history[7][0]["archive_inserted"], 2)

    def test_video_channel_without_configured_probes_skips_clip_embedding_but_keeps_apex(self):
        probe_calls = []

        class RecordingProbeManager:
            def add_frame(self, *_args, **_kwargs):
                probe_calls.append(True)

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.probe_manager = RecordingProbeManager()
            session = LuxriotCaptureSession(manager, 112, 12, "Describe.", run_id="run-112")
            first = Image.new("RGB", (20, 12), color=(0, 0, 0))
            second = Image.new("RGB", (20, 12), color=(255, 255, 255))

            session._accept_captured_frame(first, 100_100, summarize=False)
            session._accept_captured_frame(second, 101_100, summarize=False)

            self.assertEqual(probe_calls, [])
            self.assertEqual(session.status()["capture_apex_selected_count"], 1)
            self.assertEqual(session.status()["capture_apex_probe_skipped_count"], 1)
            self.assertEqual(len(session.frames), 1)
            self.assertEqual(session.frames[0]["embedding_status"], "unavailable")

    def test_capture_cv_buckets_are_channel_local_and_never_index_raw_non_apex_frames(self):
        probe_calls = []

        class RecordingProbeManager:
            def add_frame(self, channel_id, image, timestamp_ms, provenance=None):
                probe_calls.append((channel_id, image.info.get("marker"), timestamp_ms, dict(provenance or {})))

        def image(level, marker):
            frame = Image.new("RGB", (20, 12), color=(level, level, level))
            frame.info["marker"] = marker
            return frame

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.jpeg_encoder = lambda frame, **_kwargs: f"jpeg-{frame.info.get('marker')}"
            manager.probe_manager = RecordingProbeManager()
            manager.shared_probe_channels.update({7, 8})
            channel_7 = LuxriotCaptureSession(manager, 7, 12, "Describe.", run_id="run-7")
            channel_8 = LuxriotCaptureSession(manager, 8, 12, "Describe.", run_id="run-8")

            channel_7._accept_captured_frame(image(0, "7-raw-a"), 200_100, summarize=False)
            channel_8._accept_captured_frame(image(0, "8-raw-a"), 200_120, summarize=False)
            channel_7._accept_captured_frame(image(255, "7-apex"), 200_500, summarize=False)
            channel_8._accept_captured_frame(image(180, "8-apex"), 200_600, summarize=False)
            channel_7._accept_captured_frame(image(255, "7-raw-c"), 200_800, summarize=False)
            channel_8._accept_captured_frame(image(180, "8-raw-c"), 200_850, summarize=False)
            channel_8._accept_captured_frame(image(0, "8-next"), 201_100, summarize=False)
            channel_7._accept_captured_frame(image(0, "7-next"), 201_100, summarize=False)

            self.assertEqual(
                [(channel_id, marker) for channel_id, marker, _timestamp, _provenance in probe_calls],
                [(8, "8-apex"), (7, "7-apex")],
            )
            self.assertNotIn(
                "raw",
                " ".join(marker for _channel_id, marker, _timestamp, _provenance in probe_calls),
            )
            self.assertEqual(channel_7.frames[0]["capture_selection"]["selected_source_frame_index"], 2)
            self.assertEqual(channel_8.frames[0]["capture_selection"]["selected_source_frame_index"], 2)

    def test_capture_cv_stop_flushes_final_single_frame_once(self):
        probe_calls = []

        class RecordingProbeManager:
            def add_frame(self, channel_id, image, timestamp_ms, provenance=None):
                probe_calls.append((channel_id, image.info.get("marker"), timestamp_ms, dict(provenance or {})))

        frame = Image.new("RGB", (20, 12), color=(64, 64, 64))
        frame.info["marker"] = "only"
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.jpeg_encoder = lambda image, **_kwargs: f"jpeg-{image.info.get('marker')}"
            manager.probe_manager = RecordingProbeManager()
            manager.shared_probe_channels.add(7)
            session = LuxriotCaptureSession(manager, 7, 12, "Describe.", run_id="run-7")
            session._accept_captured_frame(frame, 300_100, summarize=False)

            session.stop()
            session.stop()

            self.assertEqual(len(probe_calls), 1)
            self.assertEqual(probe_calls[0][1], "only")
            self.assertEqual(probe_calls[0][3]["selection_source"], "single_frame")
            self.assertEqual(
                probe_calls[0][3]["fallback_reason"],
                "single_frame_only_no_intra_second_choice",
            )
            self.assertEqual(len(session.frames), 1)

    def test_one_fps_capture_is_indexed_and_buffered_without_reduction(self):
        probe_calls = []

        class RecordingProbeManager:
            def add_frame(self, channel_id, image, timestamp_ms, provenance=None):
                probe_calls.append((channel_id, image.info.get("marker"), timestamp_ms, dict(provenance or {})))

        def image(marker):
            frame = Image.new("RGB", (20, 12), color=(64, 64, 64))
            frame.info["marker"] = marker
            return frame

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.jpeg_encoder = lambda frame, **_kwargs: f"jpeg-{frame.info.get('marker')}"
            manager.probe_manager = RecordingProbeManager()
            manager.shared_probe_channels.add(7)
            session = LuxriotCaptureSession(
                manager,
                7,
                12,
                "Describe.",
                run_id="run-7",
                interval_override=1.0,
            )

            session._accept_captured_frame(image("first"), 400_100, summarize=True)
            session._accept_captured_frame(image("second"), 401_100, summarize=True)

            self.assertEqual([call[1] for call in probe_calls], ["first", "second"])
            self.assertEqual([frame["thumbnail"] for frame in session.frames], ["jpeg-first", "jpeg-second"])
            self.assertEqual(
                [call[3]["selection_source"] for call in probe_calls],
                ["single_frame", "single_frame"],
            )

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
                self.assertEqual(session.summary_failed_batches, 1)
                self.assertIn("queue overflow", session.summary_last_error)
                self.assertEqual(session.dropped_frames, 1)
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
            session.snapshot_count = 2
            session.snapshot_slow_streak = 2
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
            self.assertEqual(session.snapshot_slow_streak, 2)
            self.assertTrue(session._should_use_live_segment())

    def test_auto_capture_requires_a_slow_streak_and_stays_on_dense_live_source(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                config_overrides={"LUXRIOT_CAPTURE_SOURCE": "auto"},
            )
            session = LuxriotCaptureSession(
                manager,
                channel_id=7,
                batch_size=4,
                prompt="Describe activity.",
                run_id="run-7",
            )
            threshold = session.snapshot_slow_threshold_sec

            session._record_snapshot_result(threshold + 0.1, success=True)
            self.assertEqual(session.snapshot_slow_streak, 1)
            self.assertFalse(session._should_use_live_segment())

            session._record_snapshot_result(0.1, success=True)
            self.assertEqual(session.snapshot_slow_streak, 0)
            self.assertFalse(session._should_use_live_segment())

            session._record_snapshot_result(threshold + 0.1, success=True)
            session._record_snapshot_result(threshold + 0.1, success=True)
            self.assertEqual(session.snapshot_slow_streak, 2)
            self.assertTrue(session._should_use_live_segment())

            # A successful dense segment must not re-arm a blocking snapshot
            # request.  Source selection is sticky for this session; live
            # failure and session restart remain the recovery paths.
            self.assertEqual(session.snapshot_slow_streak, 2)
            self.assertTrue(session._should_use_live_segment())

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

    def test_summary_batch_selects_attention_apex_per_second_and_propagates_provenance(self):
        frames = [
            {
                "thumbnail": thumbnail,
                "captured_at": timestamp,
                "time_sec": timestamp,
                "width": 1280,
                "height": 720,
            }
            for thumbnail, timestamp in zip(
                ("frame-a", "frame-b", "frame-c", "frame-d", "frame-e", "frame-f"),
                (100.05, 100.35, 100.75, 101.10, 101.50, 102.00),
            )
        ]
        vector_signal = {
            "version": 1,
            "channel_id": 7,
            "road_cv_cues": [
                {
                    "cue_type": "motion_candidate",
                    "score": 0.81,
                    "timestamp_ms": 100350,
                    "frame_index": 2,
                    "apex_frame": 2,
                }
            ],
            "clip_probe_signals": [
                {
                    "name": "vehicle drift candidate",
                    "m": 0.24,
                    "timestamp_ms": 101500,
                    "apex_frame": 5,
                }
            ],
        }
        vlm_frames = []

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))

            def message_builder(_channel, selected_frames, _prompt, _system_prompt):
                vlm_frames.extend(dict(frame) for frame in selected_frames)
                return [{"frame_count": len(selected_frames)}]

            manager.message_builder = message_builder
            with patch.object(manager, "_build_vector_signal_bundle", return_value=vector_signal):
                batch = manager.create_summary_batch(
                    channel_id=7,
                    run_id="run-7",
                    batch_size=6,
                    prompt="Describe activity.",
                    model_hint="model-a",
                    interval_sec=1.0,
                    frames=frames,
                )

            self.assertEqual([frame["thumbnail"] for frame in batch["frames"]], ["frame-b", "frame-e", "frame-f"])
            self.assertEqual([frame["source_frame_index"] for frame in batch["frames"]], [2, 5, 6])
            selection = batch["frame_selection"]
            self.assertEqual(selection["source_frame_count"], 6)
            self.assertEqual(selection["selected_frame_count"], 3)
            self.assertEqual(
                [group["selection_source"] for group in selection["groups"]],
                ["road_cv_cue", "clip_probe", "single_frame"],
            )
            self.assertEqual(selection["groups"][0]["source_frame_indices"], [1, 2, 3])
            self.assertEqual(selection["groups"][0]["source_timestamps_ms"], [100050, 100350, 100750])
            self.assertEqual(selection["groups"][0]["selected_source_frame_index"], 2)
            self.assertEqual(selection["groups"][1]["selected_timestamp_ms"], 101500)
            self.assertFalse(selection["groups"][2]["apex_available"])
            self.assertEqual(
                selection["groups"][2]["fallback_reason"],
                "single_frame_only_no_intra_second_choice",
            )

            entry = manager.run_summary_batch(batch)

            self.assertEqual([frame["thumbnail"] for frame in vlm_frames], ["frame-b", "frame-e", "frame-f"])
            self.assertEqual(entry["source_frame_count"], 6)
            self.assertEqual(entry["selected_frame_count"], 3)
            self.assertEqual(entry["frame_selection"]["groups"], selection["groups"])
            self.assertEqual(
                [frame["source_frame_index"] for frame in entry["archive_frames"]],
                [2, 5, 6],
            )
            self.assertEqual(entry["archive_frames"][0]["selection_source"], "road_cv_cue")
            self.assertEqual(entry["archive_frames"][2]["fallback_reason"], "single_frame_only_no_intra_second_choice")

            accepted = manager.accept_summary_entry(entry)
            stored = manager.summary_history[7][0]
            self.assertEqual(stored["frame_selection"]["selection_sources"]["road_cv_cue"], 1)
            self.assertEqual(stored["source_frame_count"], 6)
            l0 = manager.summary_rollups(channel_id=7, synthesize=False)["levels"]["L0"][0]
            self.assertEqual(l0["frame_selection"]["groups"][1]["selection_source"], "clip_probe")
            digest = manager.system_status_digest(channel_ids=[7])["channels"][0]
            self.assertEqual(digest["source_frame_count"], 6)
            self.assertEqual(digest["selected_frame_count"], 3)
            self.assertEqual(digest["selection_fallback_count"], 1)
            self.assertEqual(digest["last_frame_selection"]["groups"][0]["selected_source_frame_index"], 2)
            compact_runtime = manager._compact_stream_status("video", {"channel_id": 7, "logs": [accepted]})
            self.assertEqual(compact_runtime["last_source_frame_count"], 6)
            self.assertEqual(compact_runtime["last_frame_selection"]["groups"][2]["fallback_reason"], "single_frame_only_no_intra_second_choice")

    def test_summary_batch_uses_deterministic_midpoint_when_apex_unavailable(self):
        frames = [
            {
                "thumbnail": thumbnail,
                "captured_at": timestamp,
                "time_sec": timestamp,
            }
            for thumbnail, timestamp in (
                ("frame-a", 200.05),
                ("frame-b", 200.40),
                ("frame-c", 200.90),
            )
        ]

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            with patch.object(manager, "_build_vector_signal_bundle", return_value={}):
                first = manager.create_summary_batch(
                    channel_id=7,
                    run_id="run-7",
                    batch_size=3,
                    prompt="Describe activity.",
                    model_hint="model-a",
                    interval_sec=1.0,
                    frames=frames,
                )
                second = manager.create_summary_batch(
                    channel_id=7,
                    run_id="run-7b",
                    batch_size=3,
                    prompt="Describe activity.",
                    model_hint="model-a",
                    interval_sec=1.0,
                    frames=frames,
                )

            self.assertEqual([frame["thumbnail"] for frame in first["frames"]], ["frame-b"])
            self.assertEqual([frame["thumbnail"] for frame in second["frames"]], ["frame-b"])
            group = first["frame_selection"]["groups"][0]
            self.assertEqual(group["selection_source"], "deterministic_temporal_midpoint")
            self.assertFalse(group["apex_available"])
            self.assertEqual(group["fallback_reason"], "no_frame_level_attention_signal")
            self.assertEqual(group["selected_source_frame_index"], 2)
            self.assertEqual(group["source_timestamps_ms"], [200050, 200400, 200900])

    def test_one_fps_frames_are_preserved_with_single_frame_provenance(self):
        frames = [
            {
                "thumbnail": f"frame-{index}",
                "captured_at": 300.0 + index,
                "time_sec": 300.0 + index,
            }
            for index in range(3)
        ]

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            with patch.object(manager, "_build_vector_signal_bundle", return_value={}):
                batch = manager.create_summary_batch(
                    channel_id=7,
                    run_id="run-7",
                    batch_size=3,
                    prompt="Describe activity.",
                    model_hint="model-a",
                    interval_sec=1.0,
                    frames=frames,
                )

            self.assertEqual([frame["thumbnail"] for frame in batch["frames"]], ["frame-0", "frame-1", "frame-2"])
            self.assertEqual([frame["source_frame_index"] for frame in batch["frames"]], [1, 2, 3])
            selection = batch["frame_selection"]
            self.assertEqual(selection["source_frame_count"], 3)
            self.assertEqual(selection["selected_frame_count"], 3)
            self.assertEqual(selection["single_frame_count"], 3)
            self.assertEqual(selection["fallback_count"], 3)
            self.assertEqual(
                [group["selection_source"] for group in selection["groups"]],
                ["single_frame", "single_frame", "single_frame"],
            )

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
            frame_scores = first["road_cv_frame_scores"]
            self.assertEqual([row["source_frame_index"] for row in frame_scores], list(range(1, 9)))
            self.assertEqual([row["timestamp_ms"] for row in frame_scores], list(range(100000, 108000, 1000)))
            self.assertTrue(
                any(
                    isinstance(row.get("attention_score"), (int, float))
                    and row["attention_score"] > 0
                    for row in frame_scores
                )
            )
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

    def test_quiet_low_confidence_channel_idles_expensive_road_cv_ray(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                config_overrides={"LUXRIOT_ROAD_SCENE_CALIBRATION_SAMPLES": 4},
            )
            manager.road_scene_calibrations[7] = {
                "confidence": "medium",
                "sample_count": 4,
            }
            frames = [
                {
                    "thumbnail": f"frame-{index}",
                    "captured_at": 100.0 + index,
                    "capture_selection": {
                        "selection_mode": "quiet",
                        "activity_x": 0.4,
                    },
                }
                for index in range(4)
            ]

            with patch.object(
                manager,
                "_decode_frame_thumbnail_to_rgb_array",
            ) as decode:
                road_cues, frame_scores, road_scene, camera_scene, health = (
                    manager._road_cv_vector_signals(
                        7,
                        frames,
                    )
                )

            decode.assert_not_called()
            self.assertEqual(road_cues, [])
            self.assertEqual(frame_scores, [])
            self.assertEqual(road_scene, {})
            self.assertEqual(camera_scene, {})
            self.assertEqual(
                health["road_cv_status"],
                "homeostatic_idle",
            )
            self.assertEqual(
                health["road_cv_calibration_sample_count"],
                4,
            )

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
            self.assertEqual([frame["frame_index"] for frame in saved], [0, 1, 2, 3])
            self.assertEqual(
                [frame["source_frame_index"] for frame in saved],
                [1, 2, 3, 5],
            )
            self.assertEqual(
                [frame["anchor_role"] for frame in saved],
                ["first", "sample", "sample", "last"],
            )
            self.assertEqual(
                [frame["timestamp_ms"] for frame in saved],
                [100000, 101000, 102000, 104000],
            )
            self.assertTrue(saved[1]["is_cover"])
            self.assertEqual(saved[1]["cover_source"], "backend_fallback")

    def test_summary_alert_counts_roll_up_by_severity(self):
        with tempfile.TemporaryDirectory() as temp:
            def parse_alerts(text, _channel_id, _default_ts_ms=None):
                payload = json.loads(text.split("BATCH_STATE_JSON:", 1)[1])
                return list(payload.get("alerts") or [])

            manager = build_manager(Path(temp), alert_parser=parse_alerts)
            now = time.time()
            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": (
                        "critical-event\nALERTS_JSON:\n"
                        "{\"alerts\":[{\"severity\":\"critical\"},{\"severity\":\"normal\"}]}"
                    ),
                    "frame_count": 2,
                    "created_at": now,
                },
            )
            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": (
                        "low-event\nALERTS_JSON:\n"
                        "{\"alerts\":[{\"severity\":\"low\"}]}"
                    ),
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

    def test_hot_l0_history_is_bounded_for_hierarchical_rollups(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                config_overrides={
                    "LUXRIOT_SUMMARY_HISTORY_LIMIT": 5000,
                    "LUXRIOT_SUMMARY_STATE_HOT_LIMIT": 240,
                },
            )
            base = 1_781_700_000.0
            for index in range(260):
                manager.record_summary_log(
                    7,
                    {
                        "channel_id": 7,
                        "run_id": "run-7",
                        "summary": f"summary {index}",
                        "frame_count": 12,
                        "created_at": base + index,
                    },
                )

            self.assertEqual(len(manager.summary_history[7]), 240)
            self.assertEqual(manager.summary_history[7][0]["summary"], "summary 20")

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
                        "latency_trace": {
                            "summary_enqueued_at_ms": 90_000,
                            "summary_dispatch_started_at_ms": 90_125,
                            "inference_ms": 9_500,
                            "archive_processing_ms": 84,
                        },
                        "llm_input_stats": {
                            "prepare_ms": 42,
                            "generation_output_tokens": 384,
                        },
                        "lm_response_stats": {
                            "prompt_tokens": 3_900,
                            "completion_tokens": 311,
                            "finish_reason": "stop",
                        },
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
        self.assertEqual(compact["last_latency_trace"]["inference_ms"], 9_500)
        self.assertEqual(compact["last_latency_trace"]["archive_processing_ms"], 84)
        self.assertEqual(compact["last_llm_input_stats"]["prepare_ms"], 42)
        self.assertEqual(
            compact["last_llm_input_stats"]["generation_output_tokens"],
            384,
        )
        self.assertEqual(compact["last_lm_response_stats"]["prompt_tokens"], 3_900)
        self.assertEqual(compact["last_lm_response_stats"]["completion_tokens"], 311)
        self.assertEqual(compact["last_lm_response_stats"]["finish_reason"], "stop")
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

    def test_batch_state_prompt_is_decoupled_from_bookmark_side_effects(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                alert_parser=lambda *_args, **_kwargs: [
                    {"title": "test", "description": "visible event", "severity": "normal"}
                ],
            )
            manager.default_bookmark_enabled = False

            prompt = manager.get_effective_stream_system_prompt(7)
            self.assertIn("BATCH_STATE_JSON", prompt)

            batch = manager.create_summary_batch(
                channel_id=7,
                run_id="run-7",
                batch_size=1,
                prompt="Describe activity.",
                model_hint=None,
                interval_sec=1.0,
                frames=[{"captured_at": 100.0, "image_b64": "ZmFrZQ=="}],
            )
            self.assertIn("BATCH_STATE_JSON", batch["system_prompt"])
            result = manager.process_summary_alerts(
                7,
                'BATCH_STATE_JSON:\n{"version":1,"cover":{"snapshot_index":1},'
                '"events":[],"observed_states":[],"routines":[],"memory_pass":[],'
                '"alerts":[{"title":"test","severity":"normal","snapshot_indices":[1]}]}',
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

    def test_process_summary_alerts_records_exact_bookmark_delivery_timing(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                alert_parser=lambda *_args, **_kwargs: [
                    {
                        "title": "Collision risk",
                        "description": "Two vessels are converging",
                        "severity": "high",
                    }
                ],
            )
            manager.default_bookmark_enabled = True

            with patch.object(manager, "send_bookmark_event", return_value={"success": True}):
                result = manager.process_summary_alerts(
                    7,
                    "Batch summary\nALERTS_JSON:\n{\"alerts\":[]}",
                    default_ts_ms=int(time.time() * 1000.0) - 50,
                )

            self.assertEqual(int(result), 1)
            event = result.alert_events[0]
            self.assertGreaterEqual(event["bookmark_ack_at_ms"], event["bookmark_attempted_at_ms"])
            self.assertEqual(
                event["bookmark_delivery_ms"],
                event["bookmark_ack_at_ms"] - event["bookmark_attempted_at_ms"],
            )
            self.assertGreaterEqual(event["event_to_bookmark_ack_ms"], 0)
            result_payload = result.as_dict()
            self.assertEqual(
                result_payload["bookmark_first_ack_at_ms"],
                event["bookmark_ack_at_ms"],
            )

    def test_process_summary_alerts_anchors_bookmark_to_first_evidence_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                alert_parser=load_lm_alert_parser(),
            )
            manager.default_bookmark_enabled = True
            sent: List[Dict[str, Any]] = []
            summary = (
                "Two vessel tracks visibly converge.\n"
                "BATCH_STATE_JSON:\n"
                '{"version":1,"cover":{"snapshot_index":3},"events":[],'
                '"observed_states":[],"routines":[],"memory_pass":[],'
                '"alerts":[{"title":"Vessel convergence",'
                '"description":"Two vessel tracks visibly converge",'
                '"severity":"high","timestamp_ms":1781700060000,'
                '"snapshot_indices":[2,3]}]}'
            )

            with patch.object(
                manager,
                "send_bookmark_event",
                side_effect=lambda **kwargs: sent.append(kwargs) or {"success": True},
            ):
                result = manager.process_summary_alerts(
                    7,
                    summary,
                    default_ts_ms=1_781_700_060_000,
                    min_ts_ms=1_781_700_000_000,
                    max_ts_ms=1_781_700_060_000,
                    snapshot_timestamps_ms={
                        1: 1_781_700_005_000,
                        2: 1_781_700_018_000,
                        3: 1_781_700_031_000,
                    },
                )

            self.assertEqual(sent[0]["timestamp_ms"], 1_781_700_018_000)
            self.assertEqual(result.alert_events[0]["timestamp_ms"], 1_781_700_018_000)
            self.assertEqual(result.alert_events[0]["anchor_snapshot"], 2)
            self.assertEqual(result.alert_events[0]["timestamp_source"], "snapshot_index")

    def test_summary_batch_preserves_capture_queue_and_inference_latency_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                lm_callback=lambda _messages, _model: "summary",
            )
            batch = manager.create_summary_batch(
                channel_id=7,
                run_id="run-7",
                batch_size=2,
                prompt="Describe activity.",
                model_hint=None,
                interval_sec=1.0,
                frames=[
                    {"captured_at": 100.0, "thumbnail": "ZmFrZQ=="},
                    {"captured_at": 101.0, "thumbnail": "ZmFrZQ=="},
                ],
            )
            batch["latency_trace"] = {
                "batch_sealed_at_ms": 101_100,
                "summary_enqueued_at_ms": 101_200,
                "summary_dispatch_started_at_ms": 101_300,
            }

            entry = manager.run_summary_batch(batch)
            normalized = manager._normalize_summary_log_entry(entry)

            self.assertIsNotNone(normalized)
            trace = normalized["latency_trace"]
            self.assertEqual(trace["batch_first_frame_at_ms"], 100_000)
            self.assertEqual(trace["batch_last_frame_at_ms"], 101_000)
            self.assertEqual(trace["batch_sealed_at_ms"], 101_100)
            self.assertEqual(trace["summary_enqueued_at_ms"], 101_200)
            self.assertEqual(trace["summary_dispatch_started_at_ms"], 101_300)
            self.assertGreaterEqual(trace["inference_completed_at_ms"], trace["inference_started_at_ms"])
            self.assertEqual(
                trace["inference_ms"],
                trace["inference_completed_at_ms"] - trace["inference_started_at_ms"],
            )

    def test_bookmark_delivery_deduplicates_normalized_title_and_severity(self):
        with tempfile.TemporaryDirectory() as temp:
            current = {"title": "Vehicle   burnout", "severity": "WARNING"}

            def parse_alerts(_text, _channel_id, _default_ts_ms=None):
                return [{"title": current["title"], "description": "Looped clip", "severity": current["severity"]}]

            manager = build_manager(
                Path(temp),
                alert_parser=parse_alerts,
                config_overrides={
                    "LUXRIOT_BOOKMARK_COOLDOWN_SEC": 60.0,
                    "LUXRIOT_ALERT_DEDUPE_WINDOW_SEC": 600.0,
                },
            )
            manager.default_bookmark_enabled = True
            sent = []
            with patch.object(manager, "send_bookmark_event", side_effect=lambda **kwargs: sent.append(kwargs) or {"success": True}):
                first = manager.process_summary_alerts(120, "ALERTS_JSON: {}")
                current["title"] = "  vehicle burnout  "
                current["severity"] = "low"
                second = manager.process_summary_alerts(120, "ALERTS_JSON: {}")

            self.assertEqual(first, 1)
            self.assertEqual(second, 0)
            self.assertEqual(len(sent), 1)
            self.assertEqual(second.parsed, 1)
            self.assertEqual(second.alert_events[0]["delivery_status"], "deduplicated")
            self.assertEqual(second.skipped_duplicate, 1)

    def test_operator_cooldown_caps_content_dedupe_window(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                alert_parser=lambda *_args, **_kwargs: [
                    {
                        "title": "British flag mug",
                        "description": "Person drinks from the mug",
                        "severity": "warning",
                    }
                ],
                config_overrides={
                    "LUXRIOT_BOOKMARK_COOLDOWN_SEC": 10.0,
                    "LUXRIOT_ALERT_DEDUPE_WINDOW_SEC": 600.0,
                },
            )
            manager.default_bookmark_enabled = True
            sent = []
            now = {"value": 1_785_480_000.0}

            with (
                patch(
                    "luxriot_connector.time.time",
                    side_effect=lambda: now["value"],
                ),
                patch.object(
                    manager,
                    "send_bookmark_event",
                    side_effect=lambda **kwargs: sent.append(kwargs)
                    or {"success": True},
                ),
            ):
                first = manager.process_summary_alerts(112, "ALERTS_JSON: {}")
                now["value"] += 5.0
                second = manager.process_summary_alerts(112, "ALERTS_JSON: {}")
                now["value"] += 6.0
                third = manager.process_summary_alerts(112, "ALERTS_JSON: {}")

            self.assertEqual((first, second, third), (1, 0, 1))
            self.assertEqual(len(sent), 2)
            self.assertEqual(
                second.alert_events[0]["delivery_status"],
                "deduplicated",
            )

    def test_bookmark_content_dedupe_keeps_title_and_severity_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            current = {"title": "Burnout", "severity": "normal"}

            def parse_alerts(_text, _channel_id, _default_ts_ms=None):
                return [{"title": current["title"], "description": "Visible event", "severity": current["severity"]}]

            manager = build_manager(
                Path(temp),
                alert_parser=parse_alerts,
                config_overrides={
                    "LUXRIOT_BOOKMARK_COOLDOWN_SEC": 0.0,
                    "LUXRIOT_ALERT_DEDUPE_WINDOW_SEC": 600.0,
                },
            )
            manager.default_bookmark_enabled = True
            sent = []
            with patch.object(manager, "send_bookmark_event", side_effect=lambda **kwargs: sent.append(kwargs) or {"success": True}):
                first = manager.process_summary_alerts(120, "ALERTS_JSON: {}")
                current["title"] = "Drifting"
                second = manager.process_summary_alerts(120, "ALERTS_JSON: {}")
                current["severity"] = "high"
                third = manager.process_summary_alerts(120, "ALERTS_JSON: {}")

            self.assertEqual((first, second, third), (1, 1, 1))
            self.assertEqual(len(sent), 3)

    def test_bookmark_content_dedupe_window_zero_disables_it(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                alert_parser=lambda *_args, **_kwargs: [
                    {"title": "Burnout", "description": "Looped clip", "severity": "high"}
                ],
                config_overrides={
                    "LUXRIOT_BOOKMARK_COOLDOWN_SEC": 0.0,
                    "LUXRIOT_ALERT_DEDUPE_WINDOW_SEC": 0.0,
                },
            )
            manager.default_bookmark_enabled = True
            sent = []
            with patch.object(manager, "send_bookmark_event", side_effect=lambda **kwargs: sent.append(kwargs) or {"success": True}):
                first = manager.process_summary_alerts(120, "ALERTS_JSON: {}")
                second = manager.process_summary_alerts(120, "ALERTS_JSON: {}")

            self.assertEqual((first, second), (1, 1))
            self.assertEqual(len(sent), 2)

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

    def test_full_l0_suppresses_only_identical_recent_fast_phase_alert(self):
        with tempfile.TemporaryDirectory() as temp:
            current = {"title": "Collision risk"}

            def parse_alerts(_text, _channel_id, _default_ts_ms=None):
                return [
                    {
                        "title": current["title"],
                        "description": "Visible converging trajectories",
                        "severity": "high",
                    }
                ]

            manager = build_manager(
                Path(temp),
                alert_parser=parse_alerts,
                config_overrides={"VLM_FAST_ALERT_DEDUPE_WINDOW_SEC": 12.0},
            )
            manager.default_bookmark_enabled = True
            sent = []
            with patch.object(
                manager,
                "send_bookmark_event",
                side_effect=lambda **kwargs: sent.append(kwargs) or {"success": True},
            ):
                fast = manager.process_summary_alerts(
                    7,
                    "Batch summary\nALERTS_JSON:\n{\"alerts\":[]}",
                    default_ts_ms=1_781_700_000_000,
                    delivery_lane="fast_alert",
                )
                repeated_full = manager.process_summary_alerts(
                    7,
                    "Batch summary\nALERTS_JSON:\n{\"alerts\":[]}",
                    default_ts_ms=1_781_700_001_000,
                )
                current["title"] = "Person collapsed"
                distinct_full = manager.process_summary_alerts(
                    7,
                    "Batch summary\nALERTS_JSON:\n{\"alerts\":[]}",
                    default_ts_ms=1_781_700_002_000,
                )

            self.assertEqual(int(fast), 1)
            self.assertEqual(int(repeated_full), 0)
            self.assertEqual(
                repeated_full.alert_events[0]["delivery_status"],
                "fast_phase_duplicate",
            )
            self.assertEqual(int(distinct_full), 1)
            self.assertEqual(len(sent), 2)

    def test_fast_full_dedupe_uses_operator_criterion_and_source_time(self):
        with tempfile.TemporaryDirectory() as temp:
            current = {
                "title": "Thumb Up Gesture Detected",
                "timestamp_ms": 1_781_700_000_000,
            }

            def parse_alerts(_text, _channel_id, _default_ts_ms=None):
                return [
                    {
                        "title": current["title"],
                        "description": "A person visibly gives a thumbs-up gesture.",
                        "severity": "high",
                        "timestamp_ms": current["timestamp_ms"],
                    }
                ]

            manager = build_manager(
                Path(temp),
                alert_parser=parse_alerts,
                config_overrides={"VLM_FAST_ALERT_DEDUPE_WINDOW_SEC": 12.0},
            )
            manager.default_bookmark_enabled = True
            manager.channel_prompt_overrides[7] = {
                "alert_policy_prompt": "Alert when see thumbs up gesture, priority low"
            }
            sent = []
            now = {"value": 1_785_480_000.0}
            with (
                patch(
                    "luxriot_connector.time.time",
                    side_effect=lambda: now["value"],
                ),
                patch.object(
                    manager,
                    "send_bookmark_event",
                    side_effect=lambda **kwargs: sent.append(kwargs)
                    or {"success": True},
                ),
            ):
                fast = manager.process_summary_alerts(
                    7,
                    "ALERTS_JSON: {}",
                    default_ts_ms=current["timestamp_ms"],
                    delivery_lane="fast_alert",
                )
                # The full batch finishes much later and uses different model
                # wording, but points at the same source-time episode.
                now["value"] += 40.0
                current["title"] = "Thumbs up gesture"
                repeated_full = manager.process_summary_alerts(
                    7,
                    "ALERTS_JSON: {}",
                    default_ts_ms=current["timestamp_ms"],
                )
                # A later source event outside the bounded episode window is
                # distinct even though the configured criterion is the same.
                now["value"] += 1.0
                current["timestamp_ms"] += 20_000
                distinct_full = manager.process_summary_alerts(
                    7,
                    "ALERTS_JSON: {}",
                    default_ts_ms=current["timestamp_ms"],
                )

            criteria = manager._operator_alert_policy_criteria(
                manager.channel_prompt_overrides[7]["alert_policy_prompt"]
            )
            self.assertEqual(
                manager._alert_policy_match_tokens(criteria[0]),
                {"thumb", "up", "gesture"},
            )
            self.assertEqual(
                manager._operator_alert_policy_severity(
                    manager.channel_prompt_overrides[7]["alert_policy_prompt"],
                    criteria[0],
                ),
                "low",
            )
            self.assertEqual(
                manager._bookmark_content_key(
                    {"title": "Thumb Up Gesture Detected", "severity": "high"},
                    criteria,
                ),
                manager._bookmark_content_key(
                    {"title": "Thumbs up gesture", "severity": "high"},
                    criteria,
                ),
            )
            self.assertEqual(int(fast), 1)
            self.assertEqual(int(repeated_full), 0)
            self.assertEqual(
                repeated_full.alert_events[0]["delivery_status"],
                "fast_phase_duplicate",
            )
            self.assertEqual(int(distinct_full), 1)
            self.assertEqual(len(sent), 2)

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

    def test_prose_alert_section_is_diagnostic_only_and_cannot_drive_bookmarks(self):
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
            self.assertEqual(logs[0]["alert_counts"], {})
            self.assertNotIn("alerts", logs[0]["signal_digest"])

            manager.default_bookmark_enabled = True
            manager.default_bookmark_cooldown_sec = 0.0
            sent = []
            with patch.object(manager, "send_bookmark_event", side_effect=lambda **kwargs: sent.append(kwargs) or {"success": True}):
                count = manager.process_summary_alerts(
                    7,
                    summary,
                    default_ts_ms=1_781_700_000_000,
                )
            self.assertEqual(count, 0)
            self.assertEqual(sent, [])
            self.assertEqual(
                count.as_dict()["alert_parser_error"],
                "structured_alert_contract_missing",
            )

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
            now = time.time()
            manager._update_channel_routine_context(
                channel_id=7,
                rollup_id="l1-memory",
                window_end=now - 1.0,
                level="L1",
                memory_update={"active_watchlist": ["watch the east gate"]},
                summary_text="",
            )
            manager._update_channel_routine_context(
                channel_id=7,
                rollup_id="l2-memory",
                window_end=now,
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
            self.assertIn("watch the east gate", system_prompt)
            self.assertNotIn("parking lot is usually empty overnight", system_prompt)
            self.assertNotIn("vehicle drifting", system_prompt)
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
            self.assertIn("east gate", routine_text)
            self.assertIn("vehicle drifting", routine_text)
            self.assertNotIn("quiet lot overnight", routine_text)
            self.assertNotIn("keep drifting visible", routine_text)
            self.assertNotIn("parked maintenance vehicles", routine_text)

    def test_cold_probe_text_prewarm_does_not_block_l0_scoring(self):
        class ProbeStore:
            @staticmethod
            def list_probes():
                return [
                    {
                        "id": "cold-probe",
                        "name": "person near window",
                        "channel_id": 7,
                        "positives": ["person near window"],
                        "negatives": ["empty window"],
                        "enabled": True,
                    }
                ]

        class ProbeManager:
            scheduled = []

            @staticmethod
            def texts_cached(texts):
                return False

            @classmethod
            def prewarm_texts_async(cls, texts):
                cls.scheduled.append(list(texts))
                return len(texts)

            @staticmethod
            def score_frames(*_args, **_kwargs):
                raise AssertionError("cold text vectors must not block L0")

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.probe_manager = ProbeManager()
            manager.probes_store = ProbeStore()

            started = time.monotonic()
            signals, health = manager._clip_probe_vector_signals(
                7,
                [{"captured_at": 1.0}],
                batch_start_ms=900,
                batch_end_ms=1_100,
            )

        self.assertEqual(signals, [])
        self.assertEqual(health["clip_probe_status"], "text_prewarm_pending")
        self.assertEqual(health["clip_probe_prewarm_scheduled"], 2)
        self.assertEqual(
            ProbeManager.scheduled,
            [["person near window", "empty window"]],
        )
        self.assertLess(time.monotonic() - started, 0.1)

    def test_auto_probe_scores_are_shadow_but_operator_scores_regulate_attention(self):
        class ProbeStore:
            def __init__(self, origin):
                self.origin = origin

            def list_probes(self):
                return [
                    {
                        "id": f"{self.origin}-probe",
                        "name": "person near window",
                        "channel_id": 7,
                        "positives": ["person near window"],
                        "negatives": ["empty window"],
                        "pos_floor": 0.2,
                        "margin": 0.05,
                        "enabled": True,
                        "origin": self.origin,
                        "temporary": self.origin == "auto",
                        "attention_only": self.origin == "auto",
                    }
                ]

        class ProbeManager:
            @staticmethod
            def score_frames(*_args, **_kwargs):
                return {
                    "results": [
                        {
                            "timestamp_ms": 1_000,
                            "pos_score": 0.42,
                            "neg_score": 0.18,
                            "margin": 0.24,
                        }
                    ]
                }

            @staticmethod
            def query(*_args, **_kwargs):
                return {
                    "frames_indexed": 1,
                    "results": [
                        {
                            "timestamp_ms": 1_000,
                            "pos_score": 0.42,
                            "neg_score": 0.18,
                            "margin": 0.24,
                        }
                    ],
                }

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                config_overrides={
                    "LUXRIOT_ATTENTION_SCHEDULER_ENABLED": True,
                },
            )
            manager.probe_manager = ProbeManager()
            manager.link_attention_snapshot(
                channel_id=7,
                timestamp_ms=1_000,
                snapshot_id="snapshot-1",
                embedding_ref="embedding://7/snapshot-1",
                frame_hash="hash-1",
            )
            emitted = []
            manager.set_attention_event_callback(
                lambda event_type, payload: emitted.append(
                    (event_type, payload)
                )
            )
            frames = [
                {
                    "captured_at": 1.0,
                    "attention_snapshot_id": "snapshot-1",
                }
            ]

            manager.probes_store = ProbeStore("auto")
            auto_signals, _health = manager._clip_probe_vector_signals(
                7,
                frames,
                batch_start_ms=900,
                batch_end_ms=1_100,
            )
            self.assertEqual(
                auto_signals[0]["attention_authority"],
                "shadow",
            )
            self.assertFalse(
                any(event_type == "probe_scores" for event_type, _ in emitted)
            )
            shadow_decision = manager.observe_attention_cv(
                channel_id=7,
                timestamp_ms=1_100,
                motion_score=0.0,
                activity_x=0.0,
                mode="quiet",
            )
            self.assertEqual(
                shadow_decision.state.vector.probe_positive,
                0.0,
            )

            emitted.clear()
            frames[0].pop("attention_probe_scores", None)
            manager.probes_store = ProbeStore("operator")
            operator_signals, _health = manager._clip_probe_vector_signals(
                7,
                frames,
                batch_start_ms=900,
                batch_end_ms=1_100,
            )
            self.assertEqual(
                operator_signals[0]["attention_authority"],
                "regulatory",
            )
            self.assertTrue(
                any(event_type == "probe_scores" for event_type, _ in emitted)
            )
            regulatory_decision = manager.observe_attention_cv(
                channel_id=7,
                timestamp_ms=1_200,
                motion_score=0.0,
                activity_x=0.0,
                mode="quiet",
            )
            self.assertEqual(
                regulatory_decision.state.vector.probe_positive,
                0.42,
            )
            self.assertEqual(
                regulatory_decision.state.vector.probe_margin,
                0.24,
            )

    def test_siglip2_probe_stays_shadow_until_embedding_space_is_stamped(self):
        class ProbeStore:
            def __init__(self, embedding_space):
                self.embedding_space = embedding_space

            def list_probes(self):
                return [
                    {
                        "id": "operator-probe",
                        "name": "person near window",
                        "channel_id": 7,
                        "positives": ["person near window"],
                        "negatives": ["empty window"],
                        "pos_floor": 0.2,
                        "margin": 0.05,
                        "enabled": True,
                        "origin": "operator",
                        "embedding_space": self.embedding_space,
                    }
                ]

        class ProbeManager:
            @staticmethod
            def score_frames(*_args, **_kwargs):
                return {
                    "results": [
                        {
                            "timestamp_ms": 1_000,
                            "pos_score": 0.42,
                            "neg_score": 0.18,
                            "margin": 0.24,
                        }
                    ]
                }

            @staticmethod
            def query(*_args, **_kwargs):
                return {
                    "frames_indexed": 1,
                    "results": [
                        {
                            "timestamp_ms": 1_000,
                            "pos_score": 0.42,
                            "neg_score": 0.18,
                            "margin": 0.24,
                        }
                    ],
                }

        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.config.CLIP_MODEL = "google/siglip2-base-patch16-224"
            manager.config.CLIP_MODEL_REVISION = "test-siglip-revision"
            manager.probe_manager = ProbeManager()
            frames = [
                {
                    "captured_at": 1.0,
                    "attention_snapshot_id": "snapshot-1",
                }
            ]

            manager.probes_store = ProbeStore(None)
            unverified, health = manager._clip_probe_vector_signals(
                7,
                frames,
                batch_start_ms=900,
                batch_end_ms=1_100,
            )
            self.assertEqual(unverified[0]["attention_authority"], "shadow")
            self.assertEqual(
                unverified[0]["authority_reason"],
                "embedding_space_unverified",
            )
            self.assertEqual(
                health["clip_probe_embedding_space_mismatch"],
                1,
            )

            manager.probes_store = ProbeStore(
                {
                    "backend": "siglip2",
                    "model": "google/siglip2-base-patch16-224",
                    "revision": "test-siglip-revision",
                    "dimension": 768,
                }
            )
            verified, health = manager._clip_probe_vector_signals(
                7,
                frames,
                batch_start_ms=900,
                batch_end_ms=1_100,
            )
            self.assertEqual(verified[0]["attention_authority"], "regulatory")
            self.assertEqual(health["clip_probe_regulatory"], 1)

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
                    "batch_state": {
                        "version": 1,
                        "contract_status": "parsed",
                        "observed_states": [
                            {
                                "key": "person near entrance",
                                "label": "Person near entrance",
                                "state": "present",
                                "snapshot_indices": [1, 12],
                                "evidence": "visible near doorway",
                            }
                        ],
                        "alerts": [],
                    },
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
                    "batch_state": {
                        "version": 1,
                        "contract_status": "parsed",
                        "observed_states": [
                            {
                                "key": "person near entrance",
                                "label": "Person near entrance",
                                "state": "absent",
                                "snapshot_indices": [1, 12],
                                "evidence": "entrance area is clear",
                            }
                        ],
                        "alerts": [],
                    },
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
                    "batch_state": {
                        "version": 1,
                        "contract_status": "parsed",
                        "observed_states": [
                            {
                                "key": "person near entrance",
                                "label": "Person near entrance",
                                "state": "absent",
                                "snapshot_indices": [1, 12],
                                "evidence": "entrance area remains clear",
                            }
                        ],
                        "alerts": [],
                    },
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
            self.assertEqual(l1["alert_events"][0]["title"], "Person down")
            self.assertEqual(l1["alert_events"][0]["delivery_status"], "cooldown_skipped")
            self.assertNotIn("state_observations", l1)
            self.assertNotIn("state_transition_events", l1)

    def test_summary_rollups_target_level_does_not_compute_hidden_higher_levels(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": "A vehicle crossed the intersection.",
                    "frame_count": 12,
                    "created_at": 100.0,
                    "batch_start_ms": 100000,
                    "batch_end_ms": 112000,
                },
            )
            built_levels = []
            original_build = manager._build_rollup_level

            def recording_build(*args, **kwargs):
                built_levels.append(str(kwargs.get("level") or ""))
                return original_build(*args, **kwargs)

            with patch.object(manager, "_build_rollup_level", side_effect=recording_build):
                rollups = manager.summary_rollups(
                    7,
                    run_selector="all",
                    level_limit=10,
                    synthesize=False,
                    target_level="L1",
                )

            self.assertEqual(built_levels, ["L1"])
            self.assertEqual(rollups["computed_levels"], ["L0", "L1"])
            self.assertEqual(rollups["not_requested_levels"], ["L2", "L3"])
            self.assertEqual(rollups["levels"]["L2"], [])
            self.assertEqual(rollups["levels"]["L3"], [])
            self.assertEqual(rollups["aggregation"]["status"], "ready")
            self.assertGreaterEqual(rollups["aggregation"]["elapsed_sec"], 0.0)

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
                    "batch_state": {
                        "version": 1,
                        "contract_status": "parsed",
                        "observed_states": [
                            {
                                "key": "person near entrance",
                                "label": "Person near entrance",
                                "state": "present",
                                "snapshot_indices": [1],
                                "evidence": "visible near doorway",
                            }
                        ],
                        "alerts": [
                            {
                                "title": "Person down",
                                "severity": "high",
                                "state": "new",
                            }
                        ],
                    },
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
                return operator_rollup_response(
                    "Window with source alerts.",
                    alerts="Normal and low review alerts were preserved with their observable meaning.",
                    memory={
                        "routine_baseline": "quiet test scene",
                        "alert_tuning_notes": ["preserve source alerts"],
                    },
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
            backend_instructions = settings["prompt_layers"]["rollups"]["L1"]["backend_instructions"]
            self.assertIn("Explain each alert's observable meaning", backend_instructions)
            self.assertIn("never ask the operator to confirm intent", backend_instructions)
            self.assertIn("Sampled snapshots cannot prove complete scene coverage", backend_instructions)
            self.assertIn("no interruption recorded in metadata", backend_instructions)

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

    def test_rollup_model_uses_dedicated_text_profile_before_channel_vlm(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                config_overrides={"LUXRIOT_ROLLUP_LLM_MODEL": "agent"},
            )
            manager.channel_prompt_overrides[7] = {"model_hint": "vlm"}
            manager.summary_runs[7] = [{"model": "qwen/qwen3-vl-4b"}]

            with manager.cache_lock:
                selected = manager._get_rollup_model_hint_locked(7)

            self.assertEqual(selected, "agent")

    def test_l3_rollup_includes_operator_feedback_and_invalidates_cache_signature(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                config_overrides={"LUXRIOT_ROLLUP_L3_WINDOW_SEC": 28800},
            )
            feedback_count = {"value": 1}
            loader_calls = []

            def feedback_loader(channel_id, start_ts, end_ts, limit):
                loader_calls.append((channel_id, start_ts, end_ts, limit))
                return {
                    "summary": {
                        "annotation_count": feedback_count["value"],
                        "distinct_alert_count": feedback_count["value"],
                        "reviewer_count": 1,
                        "channel_count": 1,
                    },
                    "coverage": {
                        "status": "covered",
                        "annotation_count": feedback_count["value"],
                        "returned_count": feedback_count["value"],
                        "truncated": False,
                    },
                    "reason_counts": [
                        {
                            "reason_code": "benign_activity",
                            "reason_label": "Benign activity",
                            "count": feedback_count["value"],
                        }
                    ],
                    "feedback": [
                        {
                            "detection_id": 501,
                            "alert_timestamp_ms": 1_781_700_000_000,
                            "alert_title": "Person near door",
                            "reason_code": "benign_activity",
                            "reason_label": "Benign activity",
                            "note": "Maintenance worker.",
                        }
                    ],
                    "period": {
                        "last_updated_at_ms": feedback_count["value"] * 1000,
                    },
                }

            manager.set_operator_feedback_report_loader(feedback_loader)
            child = {
                "rollup_id": "l2-ch7-window",
                "channel_id": 7,
                "level": "L2",
                "window_start": 1_781_700_000.0,
                "window_end": 1_781_703_600.0,
                "item_count": 4,
                "frame_count": 48,
                "run_ids": ["run-7"],
                "summary": "Routine doorway activity with one alert.",
                "alert_counts": {"normal": 1},
                "source_ids": ["l1-ch7-window"],
            }
            l1_child = {
                "rollup_id": "l1-ch7-window",
                "source_ids": ["l0-ch7-interesting"],
                "item_count": 4,
                "frame_count": 48,
                "alert_counts": {"normal": 1},
            }
            l0_child = {
                "rollup_id": "l0-ch7-interesting",
                "window_start": 1_781_700_100.0,
                "window_end": 1_781_700_130.0,
                "summary": "A doorway alert fired while a maintenance worker crossed.",
                "item_count": 1,
                "frame_count": 12,
                "alert_counts": {"normal": 1},
                "vector_signal_total": 2,
            }
            manager._attach_l2_hierarchy_audits(
                [child],
                [l1_child],
                [l0_child],
            )

            first = manager._build_rollup_level(
                7,
                "L3",
                "L2",
                28800,
                [child],
                synthesize=False,
            )[0]
            messages = manager._build_rollup_messages(
                7,
                "L3",
                "L2",
                first,
                [child],
            )
            feedback_count["value"] = 2
            second = manager._build_rollup_level(
                7,
                "L3",
                "L2",
                28800,
                [child],
                synthesize=False,
            )[0]

            user_text = messages[1]["content"][0]["text"]
            self.assertEqual(first["window_sec"], 28800)
            self.assertEqual(
                first["operator_feedback"]["coverage"]["ground_truth_status"],
                "operator_annotation_only",
            )
            self.assertIn("Operator false-positive annotations", user_text)
            self.assertIn("Maintenance worker.", user_text)
            self.assertIn("not independent ground truth", user_text)
            self.assertEqual(first["hierarchy_audit"]["status"], "consistent")
            self.assertEqual(len(first["l0_drill_samples"]), 1)
            self.assertIn("Deterministic L1-to-L2 hierarchy audit", user_text)
            self.assertIn("maximum 4", user_text)
            self.assertNotEqual(
                first["source_signature"],
                second["source_signature"],
            )
            self.assertEqual(loader_calls[0][0], 7)
            self.assertEqual(loader_calls[0][3], 16)

    def test_open_rollup_window_is_pending_and_never_sent_to_llm(self):
        with tempfile.TemporaryDirectory() as temp:
            calls = []

            def lm_callback(_messages, _model):
                calls.append(True)
                return operator_rollup_response("This should not be generated yet.")

            manager = build_manager(Path(temp), lm_callback=lm_callback)
            manager.rollup_llm_levels = {"L1"}
            child = {
                "rollup_id": "l0-ch7-open",
                "window_start": 10_050.0,
                "window_end": 10_060.0,
                "item_count": 1,
                "frame_count": 12,
                "run_ids": ["run-open"],
                "summary": "A currently open observation.",
            }

            with patch("luxriot_connector.time.time", return_value=10_100.0):
                row = manager._build_rollup_level(
                    7,
                    "L1",
                    "L0",
                    900,
                    [child],
                    synthesize=True,
                )[0]

            self.assertEqual(row["summary_kind"], "pending_context")
            self.assertEqual(row["generation_status"], "pending")
            self.assertIn("still collecting context", row["summary"])
            self.assertEqual(calls, [])

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

            with patch.object(
                manager,
                "_refresh_channel_memory_from_rollups",
            ) as memory_refresh:
                rollups = manager.summary_rollups(
                    7,
                    run_selector="all",
                    level_limit=10,
                    synthesize=False,
                )

            self.assertEqual(calls, [])
            memory_refresh.assert_not_called()
            row = rollups["levels"]["L1"][0]
            self.assertEqual(row["summary_kind"], "queued")
            self.assertIn("Semantic L1 aggregation is queued", row["summary"])
            self.assertNotIn("Highlights:", row["summary"])

    def test_rollup_cache_signature_changes_when_child_metadata_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            calls = []

            def lm_callback(messages, _model):
                calls.append(messages[1]["content"][0]["text"])
                return operator_rollup_response(
                    f"L2 cache pass {len(calls)}.",
                    memory={"routine_baseline": f"cache pass {len(calls)}"},
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

    def test_higher_rollup_uses_cached_semantic_children_before_synthesis(self):
        with tempfile.TemporaryDirectory() as temp:
            l2_inputs = []

            def lm_callback(messages, _model):
                user_text = messages[1]["content"][0]["text"]
                if "Target level: L2" in user_text:
                    l2_inputs.append(user_text)
                    return operator_rollup_response("Hour narrative built from semantic L1 context.")
                return operator_rollup_response(
                    "Distinctive semantic L1 narrative: a person left the desk and later returned."
                )

            manager = build_manager(Path(temp), lm_callback=lm_callback)
            manager.rollup_llm_levels = {"L1", "L2"}
            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": "Person leaves and returns.",
                    "frame_count": 12,
                    "created_at": 1_781_700_000.0,
                },
            )

            manager.summary_rollups(
                7,
                run_selector="all",
                target_level="L1",
                synthesize_levels={"L1"},
            )
            result = manager.summary_rollups(
                7,
                run_selector="all",
                target_level="L2",
                synthesize_levels={"L2"},
            )

            self.assertTrue(l2_inputs)
            self.assertIn("Distinctive semantic L1 narrative", l2_inputs[0])
            self.assertNotIn("Semantic L1 aggregation is queued", l2_inputs[0])
            self.assertEqual(result["levels"]["L2"][0]["summary_kind"], "llm")

    def test_semantic_rollup_survives_hot_cache_eviction_and_remains_queryable(self):
        with tempfile.TemporaryDirectory() as temp:
            state_store = DurableRollupMemoryStateStore()
            manager = build_manager(Path(temp), runtime_state_store=state_store)
            window_start = 1_781_700_000.0
            summary = operator_rollup_response(
                "A person worked at the desk and briefly left before returning."
            )

            manager._put_cached_rollup_summary(
                "l1-ch7-w900-1781700000",
                summary,
                channel_id=7,
                level="L1",
                source_level="L0",
                window_start=window_start,
                window_end=window_start + 900.0,
                window_sec=900,
                item_count=12,
                frame_count=144,
                source_tokens=1200,
                run_ids=["run-7"],
                source_ids=["l0-a"],
                source_signature="sig-a",
                summary_kind="llm",
                generation_status="ready",
                format_version=2,
            )
            manager.rollup_summary_cache.clear()

            result = manager.summary_rollups(
                7,
                run_selector="all",
                start_ts=window_start,
                end_ts=window_start + 900.0,
                target_level="L1",
                synthesize=False,
            )

            self.assertEqual(result["stored_rollups_count"], 1)
            self.assertEqual(len(result["levels"]["L1"]), 1)
            self.assertIn("briefly left", result["levels"]["L1"][0]["summary"])
            self.assertEqual(result["levels"]["L1"][0]["generation_status"], "ready")

    def test_successful_rollup_persists_temporal_incident_memory(self):
        with tempfile.TemporaryDirectory() as temp:
            state_store = DurableRollupMemoryStateStore()
            manager = build_manager(
                Path(temp),
                runtime_state_store=state_store,
                lm_callback=lambda _messages, _model: operator_rollup_response(
                    "A vehicle collision was observed in the sampled window.",
                    observations="A collision remained the grounded exception.",
                    alerts="One critical safety alert was preserved.",
                    coverage="The statement is limited to sampled frames.",
                ),
            )
            manager.rollup_llm_levels = {"L1"}
            manager.rollup_llm_max_new_per_call = 10
            base = 1_781_700_000.0
            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "batch_id": "batch-collision",
                    "summary": "Two vehicles collide in the sampled frames.",
                    "frame_count": 12,
                    "created_at": base,
                    "batch_start_ms": int(base * 1000),
                    "batch_end_ms": int((base + 30.0) * 1000),
                    "batch_state": {
                        "events": [],
                        "alerts": [
                            {
                                "title": "Vehicle collision",
                                "description": "Two vehicles visibly collide.",
                                "severity": "critical",
                                "state": "new",
                                "snapshot_indices": [4, 5],
                            }
                        ],
                    },
                    "alert_counts": {"critical": 1},
                },
            )

            row = manager.summary_rollups(
                7,
                run_selector="all",
                target_level="L1",
                level_limit=10,
            )["levels"]["L1"][0]
            stored = state_store.load_rollup(row["rollup_id"])

            self.assertIsNotNone(stored)
            self.assertTrue(stored["temporal_observations"])
            self.assertTrue(stored["incident_ledger"])
            self.assertTrue(stored["incident_dispositions"])
            self.assertEqual(
                stored["incident_ledger"][0]["priority"],
                "safety",
            )

    def test_cached_rollup_backfills_new_l2_composition_without_llm_call(self):
        with tempfile.TemporaryDirectory() as temp:
            state_store = DurableRollupMemoryStateStore()
            lm_calls = []
            manager = build_manager(
                Path(temp),
                runtime_state_store=state_store,
                lm_callback=lambda _messages, _model: lm_calls.append(True) or "unexpected",
            )
            manager.rollup_llm_levels = {"L2"}
            rollup_id = "l2-ch7-w3600-1781700000"
            summary = operator_rollup_response(
                "A collision and its nearby response were observed."
            )
            manager._put_cached_rollup_summary(
                rollup_id,
                summary,
                channel_id=7,
                level="L2",
                source_level="L1",
                window_start=1_781_700_000.0,
                window_end=1_781_703_600.0,
                window_sec=3600,
                source_tokens=1200,
                source_ids=["l1-a"],
                source_signature="same-source",
                summary_kind="llm",
                generation_status="ready",
                format_version=2,
            )
            temporal = make_temporal_observation(
                channel_id=7,
                source_batch_id="batch-collision",
                ordinal=0,
                kind="event",
                state="new",
                semantic_key="vehicle collision",
                label="Vehicle collision",
                start_ms=10_000,
                end_ms=12_000,
            ).to_dict()
            temporal.update(trigger_kind="safety_alert", severity="critical")
            node = {
                "rollup_id": rollup_id,
                "channel_id": 7,
                "level": "L2",
                "source_level": "L1",
                "window_start": 1_781_700_000.0,
                "window_end": 1_781_703_600.0,
                "window_sec": 3600,
                "source_tokens": 1200,
                "source_ids": ["l1-a"],
                "source_signature": "same-source",
                "summary": summary,
                "routine_ledger": [{"routine_id": "routine-a"}],
                "temporal_observations": [temporal],
                "incident_ledger": [{"episode_id": "episode-a"}],
                "incident_dispositions": [
                    {"observation_id": temporal["observation_id"]}
                ],
                "incident_compositions": [{"composition_id": "composition-a"}],
            }

            manager._apply_rollup_llm_summaries(
                7,
                "L2",
                "L1",
                [(node, [])],
                max_new=1,
            )
            stored = state_store.load_rollup(rollup_id)

            self.assertEqual(lm_calls, [])
            self.assertEqual(stored["routine_ledger"][0]["routine_id"], "routine-a")
            self.assertEqual(
                stored["temporal_observations"][0]["observation_id"],
                temporal["observation_id"],
            )
            self.assertEqual(stored["incident_ledger"][0]["episode_id"], "episode-a")
            self.assertEqual(
                stored["incident_compositions"][0]["composition_id"],
                "composition-a",
            )

    def test_durable_rollup_range_read_skips_duplicate_and_off_channel_hot_normalization(self):
        with tempfile.TemporaryDirectory() as temp:
            state_store = DurableRollupMemoryStateStore()
            manager = build_manager(Path(temp), runtime_state_store=state_store)
            window_start = 1_781_700_000.0
            requested = {
                "rollup_id": "l1-ch7-w900-1781700000",
                "channel_id": 7,
                "level": "L1",
                "source_level": "L0",
                "window_start": window_start,
                "window_end": window_start + 900.0,
                "window_sec": 900,
                "summary": operator_rollup_response("A person worked at the desk."),
                "summary_kind": "llm",
                "generation_status": "ready",
                "format_version": 2,
                "created_at": window_start + 900.0,
            }
            state_store.save_rollup(requested)
            manager.rollup_summary_cache[requested["rollup_id"]] = dict(requested)
            for index in range(200):
                other = dict(requested)
                other["rollup_id"] = f"l1-ch112-w900-{1781700000 + index * 900}"
                other["channel_id"] = 112
                other["window_start"] = window_start + index * 900.0
                other["window_end"] = other["window_start"] + 900.0
                manager.rollup_summary_cache[other["rollup_id"]] = other

            original_normalize = manager._normalize_cached_rollup_entry
            with patch.object(
                manager,
                "_normalize_cached_rollup_entry",
                wraps=original_normalize,
            ) as normalize:
                rows = manager._list_cached_rollups(
                    channel_id=7,
                    start_ts=window_start,
                    end_ts=window_start + 900.0,
                )

            self.assertEqual(len(rows), 1)
            self.assertEqual(normalize.call_count, 1)

    def test_per_rollup_store_skips_rewriting_legacy_monolithic_cache(self):
        class TrackingStore(DurableRollupMemoryStateStore):
            def __init__(self):
                super().__init__()
                self.saved_state_keys = []

            def save_state(self, key, payload):
                self.saved_state_keys.append(str(key))
                super().save_state(key, payload)

        with tempfile.TemporaryDirectory() as temp:
            state_store = TrackingStore()
            manager = build_manager(Path(temp), runtime_state_store=state_store)
            window_start = 1_781_700_000.0

            manager._put_cached_rollup_summary(
                "l1-ch7-w900-1781700000",
                operator_rollup_response("A person worked at the desk."),
                channel_id=7,
                level="L1",
                source_level="L0",
                window_start=window_start,
                window_end=window_start + 900.0,
                window_sec=900,
                item_count=1,
                frame_count=12,
                source_tokens=120,
                source_ids=["l0-a"],
                source_signature="source-v2",
                summary_kind="llm",
                generation_status="ready",
                format_version=2,
            )

            self.assertIn("l1-ch7-w900-1781700000", state_store.rollups)
            self.assertNotIn("luxriot_rollup_cache", state_store.saved_state_keys)
            manager.persist_rollup_cache()
            self.assertNotIn("luxriot_rollup_cache", state_store.saved_state_keys)

    def test_durable_rollup_overlays_stale_legacy_snapshot_on_startup(self):
        with tempfile.TemporaryDirectory() as temp:
            state_store = DurableRollupMemoryStateStore()
            window_start = 1_781_700_000.0
            rollup_id = "l1-ch7-w900-1781700000"
            stale = {
                "rollup_id": rollup_id,
                "channel_id": 7,
                "level": "L1",
                "source_level": "L0",
                "window_start": window_start,
                "window_end": window_start + 900.0,
                "window_sec": 900,
                "summary": operator_rollup_response("Stale legacy summary."),
                "summary_kind": "llm",
                "generation_status": "ready",
                "format_version": 2,
                "created_at": window_start + 900.0,
            }
            durable = {
                **stale,
                "summary": operator_rollup_response("New durable summary."),
                "source_signature": "new-source",
            }
            state_store.payloads["luxriot_rollup_cache"] = {
                "version": 1,
                "entries": [stale],
            }
            state_store.save_rollup(durable)

            manager = build_manager(
                Path(temp),
                runtime_state_store=state_store,
                config_overrides={"LUXRIOT_ROLLUP_RETENTION_DAYS": 90},
            )

            cached = manager._get_cached_rollup_record(rollup_id)
            self.assertIsNotNone(cached)
            self.assertIn("New durable summary", cached["summary"])
            self.assertEqual(state_store.rollups[rollup_id]["source_signature"], "new-source")

    def test_081_semantic_cache_is_adopted_without_regeneration(self):
        with tempfile.TemporaryDirectory() as temp:
            state_store = DurableRollupMemoryStateStore()
            base = float(int((time.time() - 3600.0) // 900) * 900)
            legacy_summary = (
                "### Window Snapshot\n"
                "A person worked at the desk, left briefly, and returned.\n\n"
                "### Routine Baseline\nDesk work remained the routine.\n\n"
                "### Preserved Deviations\nOne short absence was observed.\n\n"
                "### Alert Ledger\nNo structured alert was recorded.\n\n"
                "### Operator Notes\nThe observed sequence completed within the window."
            )
            rollup_id = f"l1-ch7-w900-{int(base)}"
            state_store.payloads["luxriot_rollup_cache"] = {
                "version": 1,
                "entries": [
                    {
                        "rollup_id": rollup_id,
                        "channel_id": 7,
                        "level": "L1",
                        "source_level": "L0",
                        "window_start": base,
                        "window_end": base + 900.0,
                        "window_sec": 900,
                        "source_ids": ["l0-old-a"],
                        "source_signature": "0.8.1-source",
                        "summary": legacy_summary,
                        "summary_kind": "llm",
                        "created_at": base + 900.0,
                    }
                ],
            }
            lm_calls = []
            manager = build_manager(
                Path(temp),
                lm_callback=lambda _messages, _model: lm_calls.append(True) or "unexpected",
                runtime_state_store=state_store,
                config_overrides={"LUXRIOT_ROLLUP_RETENTION_DAYS": 90},
            )
            manager.set_summary_archive_readers(
                lambda _channel, _start, _end: ([], 0),
                lambda _channel, _start, _end, _bucket: [
                    {"window_start": base, "window_end": base + 900.0}
                ],
            )

            adopted = state_store.load_rollup(rollup_id)
            plan = manager.plan_rollup_backfill(
                channel_ids=[7],
                start_ts=base,
                end_ts=base + 900.0,
                levels=["L1"],
            )
            restored = manager._restore_rollup_window(
                {
                    "channel_id": 7,
                    "level": "L1",
                    "window_start": base,
                    "window_end": base + 900.0,
                    "rollup_id": rollup_id,
                }
            )
            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "new-runtime-run",
                    "summary": "A newly loaded L0 observation for the imported window.",
                    "frame_count": 12,
                    "created_at": base + 60.0,
                    "batch_start_ms": int((base + 30.0) * 1000),
                    "batch_end_ms": int((base + 90.0) * 1000),
                },
            )
            rendered = manager.summary_rollups(
                7,
                run_selector="all",
                start_ts=base,
                end_ts=base + 899.0,
                target_level="L1",
                synthesize=True,
            )

            self.assertIsNotNone(adopted)
            self.assertEqual(adopted["summary_kind"], "legacy_cached")
            self.assertEqual(adopted["generation_status"], "legacy_ready")
            self.assertTrue(manager._rollup_semantic_ready(adopted))
            self.assertEqual(manager._rollup_scheduler_status["rollup_cache_entries_loaded"], 1)
            self.assertEqual(manager._rollup_scheduler_status["legacy_rollups_adopted"], 1)
            self.assertEqual(
                manager._rollup_scheduler_status["legacy_rollups_adopted_by_level"],
                {"L1": 1},
            )
            self.assertEqual(plan["totals"]["already_ready"], 1)
            self.assertEqual(plan["totals"]["missing_semantic"], 0)
            self.assertEqual(restored["status"], "already_ready")
            self.assertEqual(rendered["levels"]["L1"][0]["summary_kind"], "legacy_cached")
            self.assertIn("worked at the desk", rendered["levels"]["L1"][0]["summary"])
            self.assertEqual(lm_calls, [])

    def test_081_mechanical_fallback_is_not_adopted_as_semantic(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            normalized = manager._normalize_cached_rollup_entry(
                {
                    "rollup_id": "l1-ch7-w900-1781700000",
                    "channel_id": 7,
                    "level": "L1",
                    "source_level": "L0",
                    "window_start": 1_781_700_000.0,
                    "window_end": 1_781_700_900.0,
                    "window_sec": 900,
                    "summary": "L1 rollup from L0: repeated batch text",
                    "summary_kind": "degraded",
                    "generation_status": "semantic_guard_rejected",
                    "format_version": 1,
                }
            )

            self.assertIsNotNone(normalized)
            self.assertEqual(normalized["summary_kind"], "degraded")
            self.assertFalse(manager._rollup_semantic_ready(normalized))

    def test_081_semantic_overclaim_is_sanitized_without_lm_rewrite(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            normalized = manager._normalize_cached_rollup_entry(
                {
                    "rollup_id": "l1-ch7-w900-1781700000",
                    "channel_id": 7,
                    "level": "L1",
                    "source_level": "L0",
                    "window_start": 1_781_700_000.0,
                    "window_end": 1_781_700_900.0,
                    "window_sec": 900,
                    "summary": (
                        "### Window Snapshot\nRoutine sampled window.\n\n"
                        "### Operator Notes\nNo blind spots or missing coverage were found."
                    ),
                    "summary_kind": "degraded",
                    "generation_status": "semantic_guard_rejected",
                    "format_version": 1,
                }
            )

            self.assertIsNotNone(normalized)
            self.assertEqual(normalized["summary_kind"], "legacy_cached")
            self.assertEqual(normalized["generation_status"], "legacy_sanitized")
            self.assertTrue(normalized["legacy_sanitized"])
            self.assertNotIn("No blind spots", normalized["summary"])
            self.assertIn("sampled frames are partial evidence", normalized["summary"])
            self.assertTrue(manager._rollup_semantic_ready(normalized))

    def test_readonly_rollup_keeps_last_semantic_narrative_while_refresh_is_pending(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            base = {
                "rollup_id": "l1-ch7-w900-1781700000",
                "channel_id": 7,
                "level": "L1",
                "source_level": "L0",
                "window_start": 1_781_700_000.0,
                "window_end": 1_781_700_900.0,
                "window_sec": 900,
                "format_version": 2,
            }
            stored = {
                **base,
                "summary": operator_rollup_response(
                    "A person worked at the desk and briefly left."
                ),
                "summary_kind": "llm",
                "generation_status": "ready",
                "source_signature": "old-source",
            }
            generated = {
                **base,
                "summary": "Semantic L1 aggregation is queued.",
                "summary_kind": "queued",
                "generation_status": "queued",
                "source_signature": "expanded-source",
            }

            merged = manager._merge_rollup_rows([generated], [stored])[0]

            self.assertIn("briefly left", merged["summary"])
            self.assertEqual(merged["summary_kind"], "llm_cached")
            self.assertEqual(merged["generation_status"], "refresh_pending")
            self.assertTrue(merged["semantic_refresh_pending"])

    def test_synthesis_preserves_durable_rollup_against_partial_restart_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            lm_calls = []

            def lm_callback(_messages, _model):
                lm_calls.append(True)
                return operator_rollup_response("Regenerated from a partial tail.")

            manager = build_manager(Path(temp), lm_callback=lm_callback)
            manager.rollup_llm_levels = {"L1"}
            manager.rollup_time_only = True
            rollup_id = "l1-ch7-w900-1781700000"
            cached = manager._normalize_cached_rollup_entry(
                {
                    "rollup_id": rollup_id,
                    "channel_id": 7,
                    "level": "L1",
                    "source_level": "L0",
                    "window_start": 1_781_700_000.0,
                    "window_end": 1_781_700_900.0,
                    "window_sec": 900,
                    "source_ids": ["l0-old-a", "l0-old-b", "l0-restart-tail"],
                    "source_signature": "complete-source",
                    "item_count": 35,
                    "frame_count": 140,
                    "source_tokens": 9000,
                    "summary": operator_rollup_response(
                        "The complete durable window contains 35 observations."
                    ),
                    "summary_kind": "llm",
                    "generation_status": "ready",
                    "format_version": 2,
                }
            )
            self.assertIsNotNone(cached)
            manager.rollup_summary_cache[rollup_id] = cached
            candidate = {
                "rollup_id": rollup_id,
                "channel_id": 7,
                "level": "L1",
                "source_level": "L0",
                "window_start": 1_781_700_000.0,
                "window_end": 1_781_700_900.0,
                "window_sec": 900,
                "source_ids": ["l0-restart-tail"],
                "source_signature": "partial-source",
                "item_count": 1,
                "frame_count": 4,
                "source_tokens": 400,
                "summary": operator_rollup_response("Only the restart tail is loaded."),
                "summary_kind": "queued",
                "generation_status": "queued",
                "format_version": 2,
            }

            manager._apply_rollup_llm_summaries(
                channel_id=7,
                level="L1",
                source_level="L0",
                node_children_pairs=[(candidate, [])],
                max_new=1,
            )

            self.assertEqual(lm_calls, [])
            self.assertEqual(candidate["item_count"], 35)
            self.assertEqual(candidate["frame_count"], 140)
            self.assertEqual(
                candidate["source_ids"],
                ["l0-old-a", "l0-old-b", "l0-restart-tail"],
            )
            self.assertIn("complete durable window", candidate["summary"])
            self.assertTrue(candidate["source_regression_deferred"])
            self.assertEqual(
                manager._rollup_scheduler_status["source_regressions_preserved"],
                1,
            )

    def test_durable_store_rejects_partial_rollup_when_hot_cache_races(self):
        with tempfile.TemporaryDirectory() as temp:
            state_store = DurableRollupMemoryStateStore()
            manager = build_manager(
                Path(temp),
                runtime_state_store=state_store,
            )
            rollup_id = "l1-ch7-w900-1781700000"
            complete = manager._normalize_cached_rollup_entry(
                {
                    "rollup_id": rollup_id,
                    "channel_id": 7,
                    "level": "L1",
                    "source_level": "L0",
                    "window_start": 1_781_700_000.0,
                    "window_end": 1_781_700_900.0,
                    "window_sec": 900,
                    "source_ids": ["l0-old-a", "l0-restart-tail"],
                    "source_signature": "complete-source",
                    "item_count": 35,
                    "frame_count": 140,
                    "summary": operator_rollup_response("Complete durable window."),
                    "summary_kind": "llm",
                    "generation_status": "ready",
                    "format_version": 2,
                }
            )
            self.assertIsNotNone(complete)
            state_store.save_rollup(complete)
            manager.rollup_summary_cache[rollup_id] = {
                **complete,
                "source_ids": ["l0-restart-tail"],
                "source_signature": "racing-hot-tail",
                "item_count": 1,
                "frame_count": 4,
            }

            manager._put_cached_rollup_summary(
                rollup_id,
                operator_rollup_response("Partial restart tail."),
                channel_id=7,
                level="L1",
                source_level="L0",
                window_start=1_781_700_000.0,
                window_end=1_781_700_900.0,
                window_sec=900,
                source_ids=["l0-restart-tail"],
                source_signature="partial-source",
                item_count=1,
                frame_count=4,
                summary_kind="llm",
                generation_status="ready",
                format_version=2,
            )

            durable = state_store.load_rollup(rollup_id)
            self.assertEqual(durable["item_count"], 35)
            self.assertEqual(durable["frame_count"], 140)
            self.assertIn("Complete durable window", durable["summary"])
            self.assertEqual(
                manager.rollup_summary_cache[rollup_id]["item_count"],
                35,
            )
            self.assertEqual(
                manager._rollup_scheduler_status["source_regressions_preserved"],
                1,
            )

            merged = manager._merge_rollup_rows(
                [
                    {
                        **complete,
                        "source_ids": ["l0-restart-tail"],
                        "source_signature": "partial-source",
                        "item_count": 1,
                        "frame_count": 4,
                        "summary": operator_rollup_response("Partial restart tail."),
                    }
                ],
                [complete],
            )[0]
            self.assertEqual(merged["item_count"], 35)
            self.assertEqual(merged["frame_count"], 140)
            self.assertTrue(merged["source_regression_deferred"])

    def test_readonly_rollup_skips_sources_for_durable_target_buckets(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            window_sec = int(manager.rollup_windows["L1"])
            bucket_start = float(
                manager._bucket_start(1_781_700_000.0, window_sec)
            )
            stored = {
                "rollup_id": manager._canonical_rollup_id(
                    "L1", 7, bucket_start, window_sec
                ),
                "channel_id": 7,
                "level": "L1",
                "window_start": bucket_start,
                "window_end": bucket_start + window_sec,
                "window_sec": window_sec,
                "summary": operator_rollup_response("Durable narrative."),
                "summary_kind": "llm",
                "generation_status": "ready",
            }
            sources = [
                {
                    "rollup_id": "l0-inside-durable",
                    "window_start": bucket_start + 10.0,
                    "window_end": bucket_start + 20.0,
                    "summary": "Already represented.",
                },
                {
                    "rollup_id": "l0-missing-next-window",
                    "window_start": bucket_start + window_sec + 10.0,
                    "window_end": bucket_start + window_sec + 20.0,
                    "summary": "Needs a read-only bucket.",
                },
            ]

            selected = manager._rollup_sources_for_unstored_windows(
                channel_id=7,
                level="L1",
                window_sec=window_sec,
                source_nodes=sources,
                stored_rows=[stored],
            )

            self.assertEqual(
                [row["rollup_id"] for row in selected],
                ["l0-missing-next-window"],
            )

    def test_readonly_summary_uses_durable_bucket_without_refresh_rebuild(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            window_sec = int(manager.rollup_windows["L1"])
            bucket_start = float(
                manager._bucket_start(1_781_700_000.0, window_sec)
            )
            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": "A late L0 observation expanded the source set.",
                    "frame_count": 3,
                    "created_at": bucket_start + 10.0,
                },
            )
            rollup_id = manager._canonical_rollup_id(
                "L1", 7, bucket_start, window_sec
            )
            manager.rollup_summary_cache[rollup_id] = {
                "rollup_id": rollup_id,
                "channel_id": 7,
                "level": "L1",
                "source_level": "L0",
                "window_start": bucket_start,
                "window_end": bucket_start + window_sec,
                "window_sec": window_sec,
                "summary": operator_rollup_response("Durable narrative."),
                "operator_summary": operator_rollup_response(
                    "Durable narrative."
                ),
                "summary_kind": "llm",
                "generation_status": "ready",
                "source_signature": "stored-source-signature",
                "format_version": 2,
                "created_at": bucket_start + window_sec,
            }

            result = manager.summary_rollups(
                7,
                run_selector="all",
                start_ts=bucket_start,
                end_ts=bucket_start + window_sec,
                level_limit=10,
                target_level="L1",
                synthesize=False,
            )

            row = next(
                item
                for item in result["levels"]["L1"]
                if item["rollup_id"] == rollup_id
            )
            self.assertEqual(row["summary_kind"], "llm")
            self.assertEqual(row["generation_status"], "ready")
            self.assertNotIn("semantic_refresh_pending", row)

    def test_targeted_read_materializes_l0_only_for_missing_l1_buckets(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            window_sec = int(manager.rollup_windows["L1"])
            bucket_start = float(
                manager._bucket_start(1_781_700_000.0, window_sec)
            )
            for offset, summary in (
                (10.0, "Observation represented by durable L1."),
                (window_sec + 10.0, "Observation in a missing L1 bucket."),
            ):
                manager.record_summary_log(
                    7,
                    {
                        "channel_id": 7,
                        "run_id": "run-7",
                        "summary": summary,
                        "frame_count": 3,
                        "created_at": bucket_start + offset,
                    },
                )
            l1_id = manager._canonical_rollup_id(
                "L1", 7, bucket_start, window_sec
            )
            manager.rollup_summary_cache[l1_id] = {
                "rollup_id": l1_id,
                "channel_id": 7,
                "level": "L1",
                "source_level": "L0",
                "window_start": bucket_start,
                "window_end": bucket_start + window_sec,
                "window_sec": window_sec,
                "summary": operator_rollup_response("Durable L1 narrative."),
                "summary_kind": "llm",
                "generation_status": "ready",
                "format_version": 2,
            }
            original_l0_builder = manager._l0_nodes_from_logs
            materialized_logs = []

            def recording_l0_builder(channel_id, logs):
                materialized_logs.extend(logs)
                return original_l0_builder(channel_id, logs)

            with patch.object(
                manager,
                "_l0_nodes_from_logs",
                side_effect=recording_l0_builder,
            ):
                result = manager.summary_rollups(
                    7,
                    run_selector="all",
                    start_ts=bucket_start,
                    end_ts=bucket_start + 2 * window_sec,
                    target_level="L2",
                    synthesize=False,
                )

            self.assertEqual(len(materialized_logs), 1)
            self.assertIn("missing L1 bucket", materialized_logs[0]["summary"])
            self.assertEqual(result["source_counts"]["L0"], 2)
            self.assertEqual(len(result["levels"]["L0"]), 1)
            self.assertEqual(
                [row["rollup_id"] for row in result["levels"]["L1"]],
                [
                    l1_id,
                    manager._canonical_rollup_id(
                        "L1", 7, bucket_start + window_sec, window_sec
                    ),
                ],
            )

    def test_targeted_l2_synthesis_reuses_durable_l1_without_rebuilding_it(self):
        with tempfile.TemporaryDirectory() as temp:
            calls = []

            def lm_callback(messages, _model):
                calls.append(messages)
                return operator_rollup_response(
                    "The hour continued the durable desk-work pattern."
                )

            manager = build_manager(Path(temp), lm_callback=lm_callback)
            manager.rollup_llm_levels = {"L1", "L2"}
            bucket_start = float(
                manager._bucket_start(
                    1_781_700_000.0,
                    int(manager.rollup_windows["L1"]),
                )
            )
            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": "A late L0 observation in the stored L1 window.",
                    "frame_count": 3,
                    "created_at": bucket_start + 10.0,
                },
            )
            l1_id = manager._canonical_rollup_id(
                "L1", 7, bucket_start, int(manager.rollup_windows["L1"])
            )
            manager.rollup_summary_cache[l1_id] = {
                "rollup_id": l1_id,
                "channel_id": 7,
                "level": "L1",
                "source_level": "L0",
                "window_start": bucket_start,
                "window_end": bucket_start + manager.rollup_windows["L1"],
                "window_sec": manager.rollup_windows["L1"],
                "summary": operator_rollup_response(
                    "Durable L1 desk-work narrative."
                ),
                "operator_summary": operator_rollup_response(
                    "Durable L1 desk-work narrative."
                ),
                "summary_kind": "llm",
                "generation_status": "ready",
                "source_signature": "durable-l1-source",
                "format_version": 2,
            }
            build_calls = []
            original_build = manager._build_rollup_level

            def recording_build(*args, **kwargs):
                build_calls.append(
                    (
                        kwargs.get("level"),
                        len(kwargs.get("source_nodes") or []),
                        kwargs.get("synthesize"),
                    )
                )
                return original_build(*args, **kwargs)

            with patch.object(
                manager,
                "_build_rollup_level",
                side_effect=recording_build,
            ):
                result = manager.summary_rollups(
                    7,
                    run_selector="all",
                    start_ts=bucket_start,
                    end_ts=bucket_start + manager.rollup_windows["L1"],
                    target_level="L2",
                    synthesize=True,
                    synthesize_levels={"L2"},
                )

            self.assertEqual(build_calls[0], ("L1", 0, False))
            self.assertEqual(build_calls[1][0], "L2")
            self.assertGreater(build_calls[1][1], 0)
            self.assertTrue(build_calls[1][2])
            self.assertEqual(len(calls), 1)
            self.assertEqual(result["levels"]["L1"][0]["summary_kind"], "llm")
            self.assertEqual(result["levels"]["L2"][0]["summary_kind"], "llm")

    def test_durable_rollup_prevents_regeneration_after_hot_cache_eviction(self):
        with tempfile.TemporaryDirectory() as temp:
            calls = []

            def lm_callback(_messages, _model):
                calls.append(True)
                return operator_rollup_response("Generated only once.")

            state_store = DurableRollupMemoryStateStore()
            manager = build_manager(
                Path(temp),
                lm_callback=lm_callback,
                runtime_state_store=state_store,
            )
            manager.rollup_llm_levels = {"L1"}
            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": "Person remains at the desk.",
                    "frame_count": 12,
                    "created_at": 1_781_700_000.0,
                },
            )

            manager.summary_rollups(7, target_level="L1")
            self.assertEqual(len(calls), 1)
            manager.rollup_summary_cache.clear()
            manager.summary_rollups(7, target_level="L1")

            self.assertEqual(len(calls), 1)
            self.assertIn("l1-ch7-w900-1781699400", manager.rollup_summary_cache)

    def test_rollup_retention_can_outlive_hot_l0_history(self):
        with tempfile.TemporaryDirectory() as temp:
            state_store = DurableRollupMemoryStateStore()
            manager = build_manager(
                Path(temp),
                runtime_state_store=state_store,
                config_overrides={
                    "LUXRIOT_SUMMARY_RETENTION_DAYS": 7,
                    "LUXRIOT_ROLLUP_RETENTION_DAYS": 90,
                },
            )
            old_window = time.time() - 14 * 86400.0
            manager._put_cached_rollup_summary(
                manager._canonical_rollup_id("L2", 7, old_window, 3600),
                operator_rollup_response("Two-week-old behavior remains queryable."),
                channel_id=7,
                level="L2",
                source_level="L1",
                window_start=old_window,
                window_end=old_window + 3600.0,
                window_sec=3600,
                source_signature="two-week-source",
                summary_kind="llm",
                generation_status="ready",
                format_version=2,
            )

            rows = manager._list_cached_rollups(
                7,
                old_window - 1.0,
                old_window + 3601.0,
            )

            self.assertEqual(len(rows), 1)
            self.assertIn("Two-week-old behavior", rows[0]["summary"])

    def test_durable_rollup_backfill_restores_l2_from_archived_l0_text(self):
        calls = []

        def lm_callback(_messages, _model, **kwargs):
            calls.append(kwargs.get("workload_class"))
            return operator_rollup_response(
                "Across the hour, routine desk work was briefly interrupted by one departure."
            )

        lm_callback.eva_workload_class = True

        with tempfile.TemporaryDirectory() as temp:
            state_store = DurableRollupMemoryStateStore()
            manager = build_manager(
                Path(temp),
                lm_callback=lm_callback,
                runtime_state_store=state_store,
                config_overrides={
                    "LUXRIOT_SUMMARY_RETENTION_DAYS": 7,
                    "LUXRIOT_ROLLUP_RETENTION_DAYS": 90,
                },
            )
            manager.rollup_llm_levels = {"L2", "L3"}
            manager.rollup_backfill_spacing_sec = 0.01
            base = float(int((time.time() - 7200.0) // 3600) * 3600)

            def bucket_loader(channel_id, start_ts, end_ts, bucket_sec):
                self.assertEqual(channel_id, 7)
                self.assertEqual(bucket_sec, 900)
                return [
                    {
                        "window_start": base + offset,
                        "window_end": base + offset + 900.0,
                        "batch_count": 2,
                    }
                    for offset in (0.0, 900.0, 1800.0, 2700.0)
                ]

            def history_loader(channel_id, start_ts, end_ts):
                self.assertEqual(channel_id, 7)
                logs = [
                    {
                        "channel_id": 7,
                        "run_id": "archive-run",
                        "summary": f"Archived observation {index}: routine desk activity.",
                        "frame_count": 12,
                        "created_at": base + index * 900.0 + 30.0,
                        "batch_start_ms": int((base + index * 900.0 + 10.0) * 1000),
                        "batch_end_ms": int((base + index * 900.0 + 40.0) * 1000),
                    }
                    for index in range(4)
                ]
                return logs, len(logs)

            manager.set_summary_archive_readers(history_loader, bucket_loader)
            plan = manager.plan_rollup_backfill(
                channel_ids=[7],
                start_ts=base,
                end_ts=base + 3600.0,
                levels=["L2"],
            )
            self.assertEqual(plan["totals"]["missing_semantic"], 1)
            self.assertEqual(plan["restoration_scope"]["queueable_windows"], 1)
            self.assertEqual(plan["restoration_scope"]["not_restorable_no_archived_source"], 0)
            self.assertIn("Only queueable_windows", plan["restoration_scope"]["queue_contract"])
            started = manager.start_rollup_backfill(
                channel_ids=[7],
                start_ts=base,
                end_ts=base + 3600.0,
                levels=["L2"],
            )
            job_id = started["job_id"]
            deadline = time.monotonic() + 3.0
            status = manager.rollup_backfill_status()
            while status["status"] not in {"completed", "completed_with_gaps", "failed"} and time.monotonic() < deadline:
                time.sleep(0.02)
                status = manager.rollup_backfill_status()

            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["job_id"], job_id)
            self.assertEqual(status["progress"]["restored"], 1)
            self.assertEqual(status["progress_percent"], 100.0)
            rollup_id = manager._canonical_rollup_id("L2", 7, base, 3600)
            self.assertTrue(manager._rollup_semantic_ready(state_store.load_rollup(rollup_id)))
            self.assertEqual(calls, ["background"])

    def test_rollup_backfill_reports_source_gap_without_retry_loop(self):
        with tempfile.TemporaryDirectory() as temp:
            state_store = DurableRollupMemoryStateStore()
            manager = build_manager(
                Path(temp),
                runtime_state_store=state_store,
                config_overrides={"LUXRIOT_ROLLUP_RETENTION_DAYS": 90},
            )
            manager.rollup_llm_levels = {"L2"}
            manager.rollup_backfill_spacing_sec = 0.01
            base = float(int((time.time() - 7200.0) // 3600) * 3600)
            manager.set_summary_archive_readers(
                lambda _channel, _start, _end: ([], 0),
                lambda _channel, _start, _end, _bucket: [
                    {"window_start": base, "window_end": base + 900.0}
                ],
            )

            manager.start_rollup_backfill(
                channel_ids=[7],
                start_ts=base,
                end_ts=base + 3600.0,
                levels=["L2"],
            )
            deadline = time.monotonic() + 2.0
            status = manager.rollup_backfill_status()
            while status["status"] not in {"completed_with_gaps", "failed"} and time.monotonic() < deadline:
                time.sleep(0.02)
                status = manager.rollup_backfill_status()

            self.assertEqual(status["status"], "completed_with_gaps")
            self.assertEqual(status["progress"]["source_missing"], 1)
            self.assertEqual(status["progress"]["retries"], 0)

    def test_completed_backfill_with_transient_failures_can_be_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                runtime_state_store=DurableRollupMemoryStateStore(),
            )
            previous_job_id = "rollup-backfill-previous"
            request_key = "same-request"
            with manager._rollup_backfill_condition:
                manager._rollup_backfill_state = {
                    "version": 1,
                    "job_id": previous_job_id,
                    "request_key": request_key,
                    "status": "completed_with_gaps",
                    "progress": {
                        "processed": 1,
                        "restored": 0,
                        "already_ready": 0,
                        "source_missing": 0,
                        "failed": 1,
                        "retries": 3,
                    },
                    "plan": {"totals": {"missing_semantic": 1}},
                }
            plan = {
                "request_key": request_key,
                "channel_ids": [7],
                "levels": ["L2"],
                "from_ts": 1_000.0,
                "to_ts": 4_600.0,
                "totals": {"missing_semantic": 1},
                "estimated_seconds": 45.0,
                "estimated_hours": 0.01,
                "estimated_hours_range": [0.01, 0.02],
            }

            with patch.object(manager, "plan_rollup_backfill", return_value=plan):
                result = manager.start_rollup_backfill(
                    channel_ids=[7],
                    start_ts=1_000.0,
                    end_ts=4_600.0,
                    levels=["L2"],
                )

            self.assertNotEqual(result["job_id"], previous_job_id)
            self.assertFalse(result.get("idempotent_existing_job", False))
            self.assertEqual(result["progress"]["failed"], 0)

    def test_rollup_backfill_resumes_durable_cursor_after_process_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            state_store = DurableRollupMemoryStateStore()
            base = float(int((time.time() - 7200.0) // 3600) * 3600)
            state_store.save_state(
                LuxriotManager.ROLLUP_BACKFILL_STATE_KEY,
                {
                    "version": 1,
                    "job_id": "rollup-backfill-resume",
                    "request_key": "resume-key",
                    "status": "running",
                    "created_at": time.time() - 60.0,
                    "updated_at": time.time() - 30.0,
                    "started_at": time.time() - 60.0,
                    "completed_at": None,
                    "from_ts": base,
                    "to_ts": base + 3600.0,
                    "channel_ids": [7],
                    "levels": ["L2"],
                    "plan": {"totals": {"missing_semantic": 1}},
                    "cursor": {
                        "level_index": 0,
                        "channel_index": 0,
                        "after_window_start": None,
                        "attempt": 0,
                    },
                    "progress": {
                        "processed": 0,
                        "restored": 0,
                        "already_ready": 0,
                        "source_missing": 0,
                        "failed": 0,
                        "retries": 0,
                    },
                },
            )

            manager = build_manager(
                Path(temp),
                lm_callback=lambda _messages, _model: operator_rollup_response(
                    "Restarted worker restored the hour narrative."
                ),
                runtime_state_store=state_store,
                config_overrides={"LUXRIOT_ROLLUP_RETENTION_DAYS": 90},
            )
            manager.rollup_llm_levels = {"L2"}
            manager.rollup_backfill_spacing_sec = 0.01
            manager.set_summary_archive_readers(
                lambda _channel, _start, _end: (
                    [
                        {
                            "channel_id": 7,
                            "run_id": "archive-run",
                            "summary": "Archived routine observation.",
                            "frame_count": 12,
                            "created_at": base + 30.0,
                            "batch_start_ms": int((base + 10.0) * 1000),
                            "batch_end_ms": int((base + 40.0) * 1000),
                        }
                    ],
                    1,
                ),
                lambda _channel, _start, _end, _bucket: [
                    {"window_start": base, "window_end": base + 900.0}
                ],
            )
            manager.start_rollup_workers()
            deadline = time.monotonic() + 3.0
            status = manager.rollup_backfill_status()
            while status["status"] not in {"completed", "failed"} and time.monotonic() < deadline:
                time.sleep(0.02)
                status = manager.rollup_backfill_status()

            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["job_id"], "rollup-backfill-resume")
            self.assertEqual(status["progress"]["restored"], 1)

    def test_rollup_operator_summary_and_memory_are_stored_separately(self):
        with tempfile.TemporaryDirectory() as temp:
            def lm_callback(_messages, _model):
                return operator_rollup_response(
                    "A person remained at the desk through the window.",
                    routine="Desk work remained the stable behavior.",
                    observations="The person briefly left and returned.",
                    memory={
                        "routine_baseline": "desk work",
                        "active_watchlist": ["brief absence"],
                    },
                )

            manager = build_manager(Path(temp), lm_callback=lm_callback)
            manager.rollup_llm_levels = {"L1"}
            manager.rollup_llm_max_new_per_call = 10
            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": "Person works at desk, briefly leaves, then returns.",
                    "frame_count": 12,
                    "created_at": 1_781_700_000.0,
                },
            )

            row = manager.summary_rollups(7, run_selector="all", level_limit=10)["levels"]["L1"][0]

            self.assertEqual(row["summary_kind"], "llm")
            self.assertEqual(row["format_version"], 2)
            self.assertNotIn("MEMORY_UPDATE_JSON", row["summary"])
            self.assertNotIn("active_watchlist", row["summary"])
            self.assertEqual(row["memory_update"]["routine_baseline"], "desk work")
            self.assertNotIn("desk work", manager.channel_routine_context[7]["routine"])
            self.assertIn("brief absence", manager.channel_routine_context[7]["routine"])

    def test_invalid_rollup_operator_contract_is_degraded_and_diagnostic(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp), lm_callback=lambda _messages, _model: "raw internal list ...")
            manager.rollup_llm_levels = {"L1"}
            manager.rollup_llm_max_new_per_call = 10
            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": "Routine corridor activity.",
                    "frame_count": 12,
                    "created_at": 1_781_700_000.0,
                },
            )

            row = manager.summary_rollups(7, run_selector="all", level_limit=10)["levels"]["L1"][0]

            self.assertEqual(row["summary_kind"], "degraded")
            self.assertEqual(row["generation_status"], "failed")
            self.assertEqual(row["generation_error"], "invalid_operator_contract")
            self.assertNotIn("raw internal list", row["summary"])
            self.assertEqual(manager._rollup_scheduler_status["invalid_operator_contract"], 1)
            self.assertEqual(manager._rollup_scheduler_status["corrective_retries"], 1)
            cached = manager._get_cached_rollup_record(row["rollup_id"])
            self.assertIsNotNone(cached)
            self.assertEqual(cached["generation_status"], "failed")
            self.assertEqual(cached["generation_error"], "invalid_operator_contract")

    def test_successful_rollup_retry_clears_stale_generation_error(self):
        with tempfile.TemporaryDirectory() as temp:
            state_store = DurableRollupMemoryStateStore()
            manager = build_manager(
                Path(temp),
                lm_callback=lambda _messages, _model: "invalid response",
                runtime_state_store=state_store,
            )
            manager.rollup_llm_levels = {"L1"}
            manager.rollup_llm_max_new_per_call = 10
            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": "A person remained near the desk.",
                    "frame_count": 12,
                    "created_at": 1_781_700_000.0,
                },
            )

            failed = manager.summary_rollups(
                7,
                run_selector="all",
                level_limit=10,
            )["levels"]["L1"][0]
            self.assertEqual(failed["generation_status"], "failed")
            self.assertIn("generation_error", failed)

            manager.lm_callback = lambda _messages, _model: operator_rollup_response(
                "A person remained near the desk throughout the sampled window."
            )
            recovered = manager.summary_rollups(
                7,
                run_selector="all",
                level_limit=10,
            )["levels"]["L1"][0]

            self.assertEqual(recovered["summary_kind"], "llm")
            self.assertEqual(recovered["generation_status"], "ready")
            self.assertNotIn("generation_error", recovered)
            cached = manager._get_cached_rollup_record(recovered["rollup_id"])
            self.assertIsNotNone(cached)
            self.assertNotIn("generation_error", cached)

    def test_ready_cached_rollup_normalization_drops_stale_generation_error(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            normalized = manager._normalize_cached_rollup_entry(
                {
                    "rollup_id": "l1-ch7-w900-1781700000",
                    "channel_id": 7,
                    "level": "L1",
                    "source_level": "L0",
                    "window_start": 1_781_700_000.0,
                    "window_end": 1_781_700_900.0,
                    "window_sec": 900,
                    "summary": operator_rollup_response(
                        "A recovered semantic summary."
                    ),
                    "summary_kind": "llm",
                    "generation_status": "ready",
                    "generation_error": "old endpoint unavailable",
                    "format_version": 2,
                }
            )

            self.assertIsNotNone(normalized)
            self.assertNotIn("generation_error", normalized)

    def test_rollup_contract_accepts_harmless_heading_drift_without_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            calls = []

            def lm_callback(_messages, _model):
                calls.append(1)
                return (
                    "## 1. Period Overview:\nRoutine window.\n\n"
                    "## Routine & Behaviour\nDesk work continued.\n\n"
                    "### Notable Observation & Exception\nA brief absence.\n\n"
                    "## Alerts & Meaning\nNo alerts.\n\n"
                    "### Coverage & Interruption\nCoverage complete.\n\n"
                    "## Operator Takeaway\nNo action.\n\n"
                    "MEMORY_UPDATE_JSON:\n{}"
                )

            manager = build_manager(Path(temp), lm_callback=lm_callback)
            manager.rollup_llm_levels = {"L1"}
            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": "Desk work with a brief absence.",
                    "frame_count": 12,
                    "created_at": 1_781_700_000.0,
                },
            )

            row = manager.summary_rollups(7, run_selector="all")["levels"]["L1"][0]

            self.assertEqual(len(calls), 1)
            self.assertEqual(row["summary_kind"], "llm")
            self.assertIn("### Routine and Behavior", row["summary"])
            self.assertIn("### Coverage and Interruptions", row["summary"])

    def test_rollup_restores_channel_metadata_promoted_to_first_heading(self):
        with tempfile.TemporaryDirectory() as temp:
            calls = []

            def lm_callback(_messages, _model, **kwargs):
                calls.append(dict(kwargs))
                return (
                    "### Channel 7 — 10:00-10:15, 12 frames, 1 items.\n"
                    "Routine room observations.\n\n"
                    "### Routine and Behavior\nDesk work continued.\n\n"
                    "### Notable Observations and Exceptions\nA brief departure was sampled.\n\n"
                    "### Alerts and Meaning\nNo structured alerts were recorded.\n\n"
                    "### Coverage and Interruptions\nNo interruption was recorded in metadata.\n\n"
                    "### Operator Takeaway\nReview the sampled departure if relevant.\n\n"
                    "MEMORY_UPDATE_JSON:\n{}"
                )

            lm_callback.eva_workload_class = True
            lm_callback.eva_max_tokens_override = True
            manager = build_manager(
                Path(temp),
                lm_callback=lm_callback,
                config_overrides={"LUXRIOT_ROLLUP_L1_MAX_TOKENS": 777},
            )
            manager.rollup_llm_levels = {"L1"}
            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": "Desk work with a brief departure.",
                    "frame_count": 12,
                    "created_at": 1_781_700_000.0,
                },
            )

            row = manager.summary_rollups(7, run_selector="all")["levels"]["L1"][0]

            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["workload_class"], "rollup")
            self.assertEqual(calls[0]["max_tokens_override"], 777)
            self.assertEqual(row["summary_kind"], "llm")
            self.assertTrue(row["summary"].startswith("### Period Overview\nChannel 7"))

    def test_rollup_normalizes_inline_small_model_sections(self):
        flattened = (
            "### Period Overview Channel 7 — 10:00-10:15, 12 frames, 1 items.  "
            "### Routine and Behavior Desk work continued.  "
            "### Notable Observations and Exceptions A brief departure.  "
            "### Alerts and Meaning One alert remains unreviewed.  "
            "### Coverage and Interruptions No interruption recorded in metadata.  "
            "### Operator Takeaway Review the sampled departure."
        )

        normalized = LuxriotManager._normalize_rollup_operator_headings(flattened)

        self.assertTrue(LuxriotManager._rollup_operator_contract_valid(normalized))
        self.assertIn("### Period Overview\nChannel 7", normalized)
        self.assertIn("### Alerts and Meaning\nOne alert", normalized)

    def test_unreviewed_alert_dismissal_is_guarded_and_sanitized(self):
        summary = operator_rollup_response(
            "A person remained in the foreground.",
            alerts="The repeated alerts are benign and non-actionable.",
        )
        node = {"alert_total": 3, "operator_feedback": {}}

        issues = LuxriotManager._rollup_grounding_guard_issues(summary, node)
        sanitized = LuxriotManager._sanitize_rollup_operator_overclaims(summary, node)

        self.assertIn("unreviewed_alert_dismissal", issues)
        self.assertNotIn("unreviewed_alert_dismissal", LuxriotManager._rollup_grounding_guard_issues(sanitized, node))
        self.assertIn("3 alert signal emissions remain unreviewed", sanitized)

        memory = LuxriotManager._sanitize_rollup_memory_update(
            {
                "routine_baseline": "domestic activity",
                "ignore_as_routine": [
                    "repeated benign foreground alert",
                    "cat sleeping on the shelf",
                ],
            },
            node,
        )
        self.assertEqual(memory["routine_baseline"], "domestic activity")
        self.assertEqual(memory["ignore_as_routine"], ["cat sleeping on the shelf"])

    def test_unreviewed_alerts_cannot_be_closed_by_operator_takeaway(self):
        summary = operator_rollup_response(
            "A person and cat were visible in sampled frames.",
            alerts=(
                "No operator feedback was supplied for these alerts; they remain "
                "unreviewed and unclassified."
            ),
            takeaway=(
                "The activity appears routine with no unresolved safety or security "
                "incidents. No immediate follow-up is required."
            ),
        )
        node = {"alert_total": 5, "operator_feedback": {}}

        issues = LuxriotManager._rollup_grounding_guard_issues(summary, node)
        sanitized = LuxriotManager._sanitize_rollup_operator_overclaims(
            summary,
            node,
        )

        self.assertIn("unreviewed_alert_dismissal", issues)
        self.assertNotIn("no unresolved safety", sanitized.lower())
        self.assertNotIn("no immediate follow-up", sanitized.lower())
        self.assertIn("5 alert signal emissions remain unreviewed", sanitized)
        self.assertNotIn(
            "unreviewed_alert_dismissal",
            LuxriotManager._rollup_grounding_guard_issues(sanitized, node),
        )

    def test_repeated_alert_total_is_presented_as_sampled_emissions(self):
        summary = operator_rollup_response(
            "Vehicle drifting remained visible in sampled frames.",
            alerts=(
                'The system flagged 53 normal-severity alerts for "Dangerous vehicle behavior".'
            ),
        )
        node = {
            "alert_total": 53,
            "alert_delivery_breakdown": {
                "total": 53,
                "sent": 14,
                "deduplicated": 39,
            },
        }

        issues = LuxriotManager._rollup_grounding_guard_issues(summary, node)
        sanitized = LuxriotManager._sanitize_rollup_operator_overclaims(
            summary,
            node,
        )

        self.assertIn("alert_emission_count_unqualified", issues)
        self.assertIn("53 alert signal emissions", sanitized)
        self.assertIn("14 sent", sanitized)
        self.assertIn("39 deduplicated", sanitized)
        self.assertIn("not distinct real-world incidents", sanitized)
        self.assertNotIn(
            "alert_emission_count_unqualified",
            LuxriotManager._rollup_grounding_guard_issues(sanitized, node),
        )

    def test_qualified_sampled_alert_emission_count_passes_guard(self):
        summary = operator_rollup_response(
            "Vehicle drifting remained visible in sampled frames.",
            alerts=(
                "EVA recorded 53 sampled alert signal emissions; repeated emissions "
                "are not distinct real-world incidents."
            ),
        )
        node = {"alert_total": 53}

        self.assertNotIn(
            "alert_emission_count_unqualified",
            LuxriotManager._rollup_grounding_guard_issues(summary, node),
        )

    def test_temporal_episode_open_status_is_not_operator_case_state(self):
        summary = operator_rollup_response(
            "A person briefly moved near the monitor.",
            takeaway="The incident remains open and requires visual verification.",
        )
        context_node = {
            "incident_ledger": [
                {
                    "episode_id": "episode-context",
                    "status": "open",
                    "priority": "context",
                }
            ]
        }

        issues = LuxriotManager._rollup_grounding_guard_issues(
            summary,
            context_node,
        )
        sanitized = LuxriotManager._sanitize_rollup_operator_overclaims(
            summary,
            context_node,
        )

        self.assertIn("temporal_episode_as_case_state", issues)
        self.assertIn("unsupported_incident_escalation", issues)
        self.assertNotIn(
            "temporal_episode_as_case_state",
            LuxriotManager._rollup_grounding_guard_issues(
                sanitized,
                context_node,
            ),
        )
        self.assertNotIn(
            "unsupported_incident_escalation",
            LuxriotManager._rollup_grounding_guard_issues(
                sanitized,
                context_node,
            ),
        )
        self.assertIn("carries no operator-case state", sanitized)

        operator_node = {
            "incident_ledger": [
                {
                    "episode_id": "episode-thumb",
                    "status": "open",
                    "priority": "operator_criterion",
                }
            ]
        }
        operator_issues = LuxriotManager._rollup_grounding_guard_issues(
            summary,
            operator_node,
        )
        operator_sanitized = LuxriotManager._sanitize_rollup_operator_overclaims(
            summary,
            operator_node,
        )

        self.assertIn("temporal_episode_as_case_state", operator_issues)
        self.assertNotIn("unsupported_incident_escalation", operator_issues)
        self.assertIn("review its grounded alert", operator_sanitized)
        self.assertEqual(
            LuxriotManager._rollup_grounding_guard_issues(
                operator_sanitized,
                operator_node,
            ),
            [],
        )

        open_episode_summary = operator_rollup_response(
            "A brief head turn was sampled.",
            takeaway=(
                "The open episode requires continued observation."
            ),
        )
        episode_issues = LuxriotManager._rollup_grounding_guard_issues(
            open_episode_summary,
            context_node,
        )
        episode_sanitized = LuxriotManager._sanitize_rollup_operator_overclaims(
            open_episode_summary,
            context_node,
        )
        self.assertIn("temporal_episode_as_case_state", episode_issues)
        self.assertIn("unsupported_incident_escalation", episode_issues)
        self.assertEqual(
            LuxriotManager._rollup_grounding_guard_issues(
                episode_sanitized,
                context_node,
            ),
            [],
        )

        speculative_summary = operator_rollup_response(
            "A brief head turn was sampled.",
            takeaway=(
                "The head-turn event remains open and requires monitoring "
                "for potential exit intent."
            ),
        )
        speculative_issues = LuxriotManager._rollup_grounding_guard_issues(
            speculative_summary,
            context_node,
        )
        speculative_sanitized = (
            LuxriotManager._sanitize_rollup_operator_overclaims(
                speculative_summary,
                context_node,
            )
        )
        self.assertIn("temporal_episode_as_case_state", speculative_issues)
        self.assertIn("unsupported_incident_escalation", speculative_issues)
        self.assertIn("intent_speculation", speculative_issues)
        self.assertEqual(
            LuxriotManager._rollup_grounding_guard_issues(
                speculative_sanitized,
                context_node,
            ),
            [],
        )

        passive_case_state_summary = operator_rollup_response(
            "A repeating drift sequence was sampled.",
            takeaway=(
                "The continuous episode was marked as an open safety priority "
                "in the server ledger."
            ),
        )
        passive_case_state_issues = LuxriotManager._rollup_grounding_guard_issues(
            passive_case_state_summary,
            operator_node,
        )
        passive_case_state_sanitized = (
            LuxriotManager._sanitize_rollup_operator_overclaims(
                passive_case_state_summary,
                operator_node,
            )
        )
        self.assertIn(
            "temporal_episode_as_case_state",
            passive_case_state_issues,
        )
        self.assertNotIn("open safety priority", passive_case_state_sanitized)
        self.assertIn("review its grounded alert", passive_case_state_sanitized)
        self.assertEqual(
            LuxriotManager._rollup_grounding_guard_issues(
                passive_case_state_sanitized,
                operator_node,
            ),
            [],
        )

        past_tense_case_state_summary = operator_rollup_response(
            "Drift was visible in the sampled frames.",
            alerts=(
                "Several emissions resolved, but a major episode remained "
                "open at the window close."
            ),
        )
        past_tense_case_state_issues = (
            LuxriotManager._rollup_grounding_guard_issues(
                past_tense_case_state_summary,
                operator_node,
            )
        )
        past_tense_case_state_sanitized = (
            LuxriotManager._sanitize_rollup_operator_overclaims(
                past_tense_case_state_summary,
                operator_node,
            )
        )
        self.assertIn(
            "temporal_episode_as_case_state",
            past_tense_case_state_issues,
        )
        self.assertNotIn("remained open", past_tense_case_state_sanitized)
        self.assertEqual(
            LuxriotManager._rollup_grounding_guard_issues(
                past_tense_case_state_sanitized,
                operator_node,
            ),
            [],
        )

        for claim in (
            "One sub-episode remaining open at the end of the window.",
            "This was an open safety episode requiring attention.",
            "The drifting vehicle remains an unresolved safety priority.",
            "A second track involving smoke and vehicle maneuvering remains open.",
            (
                "No immediate resolution was observed for the active incident, "
                "suggesting a need for continued surveillance."
            ),
        ):
            variant = operator_rollup_response(
                "Drift was visible in the sampled frames.",
                takeaway=claim,
            )
            variant_issues = LuxriotManager._rollup_grounding_guard_issues(
                variant,
                operator_node,
            )
            variant_sanitized = (
                LuxriotManager._sanitize_rollup_operator_overclaims(
                    variant,
                    operator_node,
                )
            )
            self.assertIn(
                "temporal_episode_as_case_state",
                variant_issues,
                claim,
            )
            self.assertEqual(
                LuxriotManager._rollup_grounding_guard_issues(
                    variant_sanitized,
                    operator_node,
                ),
                [],
                claim,
            )
            self.assertIn("review its grounded alert", variant_sanitized)

        for grounded_negative in (
            "There was no active incident in the supplied samples.",
            "There was no evidence of an active incident in the supplied samples.",
            "The sequence does not represent an active incident.",
        ):
            self.assertNotIn(
                "temporal_episode_as_case_state",
                LuxriotManager._rollup_grounding_guard_issues(
                    operator_rollup_response(
                        "Drift was visible in the sampled frames.",
                        takeaway=grounded_negative,
                    ),
                    operator_node,
                ),
                grounded_negative,
            )

        active_watchlist_summary = operator_rollup_response(
            "A configured watch remains pending.",
            takeaway=(
                "The active watchlist item remains open for future verification."
            ),
        )
        self.assertNotIn(
            "temporal_episode_as_case_state",
            LuxriotManager._rollup_grounding_guard_issues(
                active_watchlist_summary,
                context_node,
            ),
        )
        self.assertEqual(
            LuxriotManager._rollup_grounding_guard_issues(
                operator_rollup_response(
                    "The observed episode has no sampled boundary.",
                    takeaway="No operator follow-up is warranted from that fact alone.",
                ),
                context_node,
            ),
            [],
        )

    def test_cached_rollup_repairs_temporal_case_overclaim_without_lm_call(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            normalized = manager._normalize_cached_rollup_entry(
                {
                    "rollup_id": "l1-ch7-w900-1781700000",
                    "channel_id": 7,
                    "level": "L1",
                    "source_level": "L0",
                    "window_start": 1_781_700_000.0,
                    "window_end": 1_781_700_900.0,
                    "window_sec": 900,
                    "summary": operator_rollup_response(
                        "A person briefly moved near the monitor.",
                        takeaway=(
                            "The incident remains open and requires visual verification."
                        ),
                    ),
                    "summary_kind": "llm",
                    "generation_status": "ready",
                    "format_version": 2,
                    "incident_ledger": [
                        {
                            "episode_id": "episode-context",
                            "status": "open",
                            "priority": "context",
                        }
                    ],
                }
            )

        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["summary_kind"], "llm")
        self.assertEqual(
            normalized["generation_status"],
            "semantic_guard_sanitized",
        )
        self.assertTrue(normalized["semantic_guard_sanitized"])
        self.assertIn("carries no operator-case state", normalized["summary"])
        self.assertTrue(manager._rollup_semantic_ready(normalized))

    def test_rollup_prompt_distinguishes_temporal_episodes_from_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            messages = manager._build_rollup_messages(
                7,
                "L1",
                "L0",
                {
                    "window_start": 1_781_700_000.0,
                    "window_end": 1_781_700_900.0,
                    "incident_ledger": [
                        {
                            "episode_id": "episode-context",
                            "status": "open",
                            "priority": "context",
                        }
                    ],
                },
                [],
            )

        system_text = messages[0]["content"][0]["text"]
        user_text = messages[1]["content"][0]["text"]
        self.assertIn("status=open means only that no episode boundary", system_text)
        self.assertIn("not operator case state", user_text)
        self.assertIn(
            "does not establish whether an operator case is active or resolved",
            user_text,
        )

    def test_rollup_temporal_scope_is_guarded_and_sanitized(self):
        short_node = {
            "window_start": 1_000.0,
            "window_end": 1_900.0,
        }
        short_summary = operator_rollup_response(
            "The room was static for the first 23 minutes before a person entered."
        )

        short_issues = LuxriotManager._rollup_grounding_guard_issues(
            short_summary,
            short_node,
        )
        short_sanitized = LuxriotManager._sanitize_rollup_operator_overclaims(
            short_summary,
            short_node,
        )

        self.assertIn("temporal_duration_overrun", short_issues)
        self.assertNotIn("first 23 minutes", short_sanitized)
        self.assertIn("during the early part of the window", short_sanitized)
        self.assertNotIn(
            "temporal_duration_overrun",
            LuxriotManager._rollup_grounding_guard_issues(
                short_sanitized,
                short_node,
            ),
        )

        long_node = {
            "window_start": 10_000.0,
            "window_end": 38_800.0,
        }
        long_summary = operator_rollup_response(
            "A person remained seated for the majority of the hour."
        )
        long_sanitized = LuxriotManager._sanitize_rollup_operator_overclaims(
            long_summary,
            long_node,
        )

        self.assertIn(
            "temporal_window_unit_mismatch",
            LuxriotManager._rollup_grounding_guard_issues(
                long_summary,
                long_node,
            ),
        )
        self.assertIn("for most of the window", long_sanitized)
        self.assertNotIn(
            "temporal_window_unit_mismatch",
            LuxriotManager._rollup_grounding_guard_issues(
                long_sanitized,
                long_node,
            ),
        )

    def test_rollup_contract_gets_one_corrective_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            calls = []

            def lm_callback(_messages, _model):
                calls.append(1)
                if len(calls) == 1:
                    return "Period report without the required sections."
                return operator_rollup_response("Routine window after format correction.")

            manager = build_manager(Path(temp), lm_callback=lm_callback)
            manager.rollup_llm_levels = {"L1"}
            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": "Routine window.",
                    "frame_count": 12,
                    "created_at": 1_781_700_000.0,
                },
            )

            row = manager.summary_rollups(7, run_selector="all")["levels"]["L1"][0]

            self.assertEqual(len(calls), 2)
            self.assertEqual(row["summary_kind"], "llm")
            self.assertEqual(manager._rollup_scheduler_status["corrective_retries"], 1)
            self.assertEqual(manager._rollup_scheduler_status["corrective_retry_successes"], 1)

    def test_rollup_prompts_bound_operator_and_memory_output_by_level(self):
        l1 = LuxriotManager._rollup_backend_instruction_text("L1")
        l2 = LuxriotManager._rollup_backend_instruction_text("L2")
        l3 = LuxriotManager._rollup_backend_instruction_text("L3")

        self.assertIn("within 180 words", l1)
        self.assertIn("at most 3 items", l1)
        self.assertIn("within 260 words", l2)
        self.assertIn("at most 5 items", l2)
        self.assertIn("within 420 words", l3)
        self.assertIn("at most 7 items", l3)
        self.assertIn("sampled alert-signal emissions", l1)
        self.assertIn("not distinct incidents", l3)
        self.assertIn("need no follow-up/action", l3)

    def test_rollup_semantic_guard_sanitizes_supported_rewrites_without_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            calls = []

            def lm_callback(_messages, _model):
                calls.append(1)
                return operator_rollup_response(
                    "Routine sampled window.",
                    coverage="No blind spots or missing coverage were found.",
                    takeaway="No safety or security concerns require operator review.",
                )

            manager = build_manager(Path(temp), lm_callback=lm_callback)
            manager.rollup_llm_levels = {"L1"}
            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": "Routine sampled window.",
                    "frame_count": 12,
                    "created_at": 1_781_700_000.0,
                },
            )

            row = manager.summary_rollups(7, run_selector="all")["levels"]["L1"][0]

            self.assertEqual(len(calls), 1)
            self.assertEqual(row["summary_kind"], "llm")
            self.assertNotIn("No blind spots", row["summary"])
            self.assertIn("sampled frames are partial evidence", row["summary"])
            self.assertEqual(manager._rollup_scheduler_status["semantic_guard_retries"], 0)
            self.assertEqual(manager._rollup_scheduler_status["semantic_guard_sanitized"], 1)

    def test_cached_rollup_with_supported_overclaim_is_sanitized_without_regeneration(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            unsafe_summary = operator_rollup_response(
                "Routine sampled window.",
                coverage="No blind spots or missing coverage were found.",
            )

            manager._put_cached_rollup_summary(
                "l1-ch7-w900-1781700000",
                unsafe_summary,
                channel_id=7,
                level="L1",
                source_level="L0",
                window_start=1_781_700_000.0,
                window_end=1_781_700_900.0,
                window_sec=900,
                source_signature="source-v1",
                summary_kind="llm",
                generation_status="ready",
                format_version=2,
            )

            cached = manager._get_cached_rollup_record("l1-ch7-w900-1781700000")
            self.assertIsNotNone(cached)
            self.assertEqual(cached["summary_kind"], "llm")
            self.assertEqual(cached["generation_status"], "semantic_guard_sanitized")
            self.assertTrue(cached["semantic_guard_sanitized"])
            self.assertNotIn("No blind spots", cached["summary"])
            self.assertIn("sampled frames are partial evidence", cached["summary"])
            self.assertTrue(manager._rollup_semantic_ready(cached))

    def test_synthesis_pass_persists_cached_semantic_guard_repair(self):
        with tempfile.TemporaryDirectory() as temp:
            calls = []
            state_store = DurableRollupMemoryStateStore()
            manager = build_manager(
                Path(temp),
                lm_callback=lambda _messages, _model: calls.append(True) or "unused",
                runtime_state_store=state_store,
            )
            manager.rollup_llm_levels = {"L1"}
            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": "A person remained near the desk.",
                    "frame_count": 12,
                    "created_at": 1_781_700_000.0,
                },
            )
            candidate = manager.summary_rollups(
                7,
                run_selector="all",
                target_level="L1",
                synthesize=False,
            )["levels"]["L1"][0]
            rollup_id = candidate["rollup_id"]
            unsafe = dict(candidate)
            unsafe_summary = operator_rollup_response(
                "A person remained near the desk.",
                coverage="No blind spots or missing coverage were found.",
            )
            unsafe.update(
                {
                    "summary": unsafe_summary,
                    "operator_summary": unsafe_summary,
                    "summary_kind": "llm",
                    "generation_status": "ready",
                    "format_version": 2,
                }
            )
            state_store.save_rollup(unsafe)
            manager.rollup_summary_cache.clear()

            row = manager.summary_rollups(
                7,
                run_selector="all",
                target_level="L1",
                synthesize=True,
            )["levels"]["L1"][0]

            self.assertEqual(calls, [])
            self.assertNotIn("No blind spots", row["summary"])
            durable = state_store.load_rollup(rollup_id)
            self.assertIsNotNone(durable)
            self.assertNotIn("No blind spots", durable["summary"])
            self.assertIn("sampled frames are partial evidence", durable["summary"])

    def test_rollup_semantic_guard_sanitizes_persistent_overclaim_without_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            calls = []

            def lm_callback(_messages, _model):
                calls.append(1)
                return operator_rollup_response(
                    "Routine sampled window.",
                    coverage="- No blind spots or missing coverage were found.",
                    takeaway="No safety or security concerns require operator review to confirm intent.",
                )

            manager = build_manager(Path(temp), lm_callback=lm_callback)
            manager.rollup_llm_levels = {"L1"}
            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": "Routine sampled window.",
                    "frame_count": 12,
                    "created_at": 1_781_700_000.0,
                },
            )

            row = manager.summary_rollups(7, run_selector="all")["levels"]["L1"][0]

            self.assertEqual(len(calls), 1)
            self.assertEqual(row["summary_kind"], "llm")
            self.assertNotIn("No blind spots", row["summary"])
            self.assertNotIn("confirm intent", row["summary"])
            self.assertIn("sampled frames are partial evidence", row["summary"])
            self.assertEqual(manager._rollup_scheduler_status["semantic_guard_sanitized"], 1)

    def test_rollup_uses_background_admission_workload_when_callback_supports_it(self):
        with tempfile.TemporaryDirectory() as temp:
            workloads = []

            def lm_callback(_messages, _model, *, workload_class=None):
                workloads.append(workload_class)
                return operator_rollup_response("Routine window.")

            lm_callback.eva_workload_class = True
            manager = build_manager(Path(temp), lm_callback=lm_callback)
            manager.rollup_llm_levels = {"L1"}
            manager.record_summary_log(
                7,
                {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": "Routine window.",
                    "frame_count": 12,
                    "created_at": 1_781_700_000.0,
                },
            )

            row = manager.summary_rollups(7, run_selector="all")["levels"]["L1"][0]

            self.assertEqual(row["summary_kind"], "llm")
            self.assertEqual(workloads, ["rollup"])

    def test_queued_rollup_is_honest_and_does_not_leak_homeostasis(self):
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
            row = rollups["levels"]["L1"][0]
            summary = row["summary"]

            self.assertEqual(row["summary_kind"], "queued")
            self.assertIn("source contains high=1", summary)
            self.assertIn("Semantic L1 aggregation is queued", summary)
            self.assertNotIn("vehicle drifting", summary)
            self.assertNotIn("Signal digest", summary)
            self.assertNotIn("alert tuning", summary.lower())

    def test_summary_rollups_preserve_deviation_memory_across_levels(self):
        with tempfile.TemporaryDirectory() as temp:
            def lm_callback(messages, _model):
                user_text = messages[1]["content"][0]["text"]
                if "Target level: L3" in user_text:
                    return operator_rollup_response(
                        "Longer period mostly routine.",
                        routine="Quiet exterior road.",
                        observations="At 02:10 a vehicle drifted near the gate.",
                        alerts="High review alert at 02:10 for visible sliding turns.",
                        memory={
                            "routine_baseline": "quiet exterior road",
                            "preserved_deviations": [{"time": "02:10", "severity": "high", "event": "vehicle drifting", "evidence": "sliding turns visible"}],
                            "alert_tuning_notes": ["do not collapse drifting into routine traffic"],
                            "ignore_as_routine": ["normal parked cars"],
                        },
                    )
                if "Target level: L2" in user_text:
                    return operator_rollup_response(
                        "Hour mostly routine with one security event.",
                        routine="Low traffic near the gate.",
                        observations="At 02:10 a vehicle drifted.",
                        memory={
                            "routine_baseline": "low traffic near the gate",
                            "preserved_deviations": [{"time": "02:10", "severity": "high", "event": "vehicle drifting", "evidence": "repeated sharp turns"}],
                        },
                    )
                return operator_rollup_response(
                    "Short window with a drifting event.",
                    observations="At 02:10 a vehicle drifted.",
                    memory={
                        "active_watchlist": ["east gate vehicle"],
                        "preserved_deviations": [{"time": "02:10", "severity": "high", "event": "vehicle drifting", "evidence": "sliding turns"}],
                    },
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
            self.assertIn("low traffic near the gate", routine_text)
            self.assertIn("vehicle drifting", routine_text)
            self.assertNotIn("do not collapse drifting into routine traffic", routine_text)

            batch = manager.create_summary_batch(
                channel_id=7,
                run_id="run-7",
                batch_size=2,
                prompt="Describe activity.",
                model_hint=None,
                interval_sec=5.0,
                frames=sample_frames(),
            )
            self.assertNotIn("Active Channel Memory", batch["system_prompt"])
            self.assertNotIn("east gate vehicle", batch["system_prompt"])
            self.assertNotIn("vehicle drifting", batch["system_prompt"])

    def test_structured_events_and_watched_states_reach_l3_without_prose_loss(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            base_ms = 1_781_683_200_000
            logs = []
            for offset_ms, event_id, label in (
                (0, "cat-race", "Two cats chase each other across the room"),
                (
                    30 * 60 * 1000,
                    "window-crawl",
                    "A person crawls past the exterior window",
                ),
            ):
                timestamp_ms = base_ms + offset_ms
                logs.append(
                    {
                        "channel_id": 7,
                        "run_id": "run-7",
                        # Deliberately bland prose: the structured state must
                        # survive even when the first summary sentence does not.
                        "summary": "Static room with ordinary furniture.",
                        "frame_count": 8,
                        "created_at": timestamp_ms / 1000.0,
                        "batch_start_ms": timestamp_ms,
                        "batch_end_ms": timestamp_ms + 50_000,
                        "batch_state": {
                            "version": 1,
                            "events": [
                                {
                                    "event_id": event_id,
                                    "label": label,
                                    "state": "new",
                                    "snapshot_indices": [2, 4],
                                    "summary": label,
                                    "novelty": "novel",
                                    "pass_up": True,
                                }
                            ],
                            "observed_states": [
                                {
                                    "key": "mail_carrier_visit",
                                    "label": "Mail carrier visit",
                                    "state": "absent",
                                    "snapshot_indices": [1, 8],
                                    "evidence": (
                                        "Entrance area is clear and visible in "
                                        "the sampled snapshots"
                                    ),
                                }
                            ],
                            "alerts": [],
                        },
                    }
                )

            l0 = manager._l0_nodes_from_logs(7, logs)
            l1 = manager._build_rollup_level(
                7,
                "L1",
                "L0",
                900,
                l0,
                synthesize=False,
            )
            l2 = manager._build_rollup_level(
                7,
                "L2",
                "L1",
                3600,
                l1,
                synthesize=False,
            )
            l3 = manager._build_rollup_level(
                7,
                "L3",
                "L2",
                28800,
                l2,
                synthesize=False,
            )

            self.assertEqual(len(l3), 1)
            labels = [item["label"] for item in l3[0]["event_ledger"]]
            self.assertIn("Two cats chase each other across the room", labels)
            self.assertIn("A person crawls past the exterior window", labels)
            watched = {
                item["key"]: item for item in l3[0]["state_ledger"]
            }["mail_carrier_visit"]
            self.assertEqual(watched["present_count"], 0)
            self.assertEqual(watched["absent_count"], 2)

            prompt = manager._build_rollup_messages(
                7,
                "L3",
                "L2",
                l3[0],
                l2,
            )[1]["content"][0]["text"]
            self.assertIn("Two cats chase each other", prompt)
            self.assertIn("person crawls past the exterior window", prompt)
            self.assertIn('"present_count":0', prompt)
            self.assertIn(
                "event_ledger_erased",
                manager._rollup_grounding_guard_issues(
                    operator_rollup_response(
                        "Nothing happened during the period.",
                        observations="No notable events occurred.",
                    ),
                    l3[0],
                ),
            )

    def test_routine_boundaries_split_parallel_incidents_and_coverage_does_not(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            base_ms = 1_781_683_200_000

            def log(
                offset_ms,
                *,
                events=(),
                routines=(),
                coverage_gap=False,
            ):
                timestamp_ms = base_ms + offset_ms
                return {
                    "channel_id": 7,
                    "run_id": "run-7",
                    "summary": "Structured temporal evidence.",
                    "frame_count": 0 if coverage_gap else 8,
                    "created_at": timestamp_ms / 1000.0,
                    "batch_start_ms": timestamp_ms,
                    "batch_end_ms": timestamp_ms + 10_000,
                    "coverage_gap": coverage_gap,
                    "gap_reason": "camera_transition" if coverage_gap else "",
                    "batch_state": {
                        "version": 2,
                        "events": list(events),
                        "routines": list(routines),
                        "observed_states": [],
                        "alerts": [],
                    },
                }

            logs = [
                log(
                    0,
                    events=(
                        {
                            "event_id": "fire_response",
                            "label": "Fire response remains active",
                            "state": "new",
                            "snapshot_indices": [2],
                            "summary": "Smoke response begins.",
                            "novelty": "novel",
                            "pass_up": True,
                        },
                        {
                            "event_id": "bag_theft",
                            "label": "Brief bag theft",
                            "state": "new",
                            "snapshot_indices": [4],
                            "summary": "A bag is taken during the response.",
                            "novelty": "novel",
                            "pass_up": True,
                        },
                    ),
                ),
                log(
                    60_000,
                    events=(
                        {
                            "event_id": "fire_response",
                            "label": "Fire response remains active",
                            "state": "continuing",
                            "snapshot_indices": [3],
                            "summary": "Response continues.",
                            "novelty": "continuing",
                            "pass_up": True,
                        },
                    ),
                    routines=(
                        {
                            "key": "ordinary_bag_area",
                            "label": "Bag area returned to routine",
                            "state": "returned",
                            "snapshot_indices": [8],
                            "applies_to_event_keys": ["bag_theft"],
                        },
                    ),
                ),
                log(120_000, coverage_gap=True),
                log(
                    180_000,
                    events=(
                        {
                            "event_id": "fire_response",
                            "label": "Fire response remains active",
                            "state": "continuing",
                            "snapshot_indices": [5],
                            "summary": "Response is still visible after coverage returns.",
                            "novelty": "continuing",
                            "pass_up": True,
                        },
                    ),
                ),
                log(
                    240_000,
                    routines=(
                        {
                            "key": "ordinary_port_state",
                            "label": "Scene returned to ordinary operation",
                            "state": "returned",
                            "snapshot_indices": [7, 8],
                            "applies_to_event_keys": ["fire_response"],
                        },
                    ),
                ),
            ]

            l0 = manager._l0_nodes_from_logs(7, logs)
            l1 = manager._build_rollup_level(
                7, "L1", "L0", 900, l0, synthesize=False
            )

            self.assertEqual(len(l1), 1)
            episodes = {
                item["semantic_key"]: item for item in l1[0]["incident_ledger"]
            }
            self.assertEqual(set(episodes), {"scene fire_smoke", "object take_object"})
            self.assertEqual(
                episodes["object take_object"]["status"], "ended_by_routine"
            )
            self.assertEqual(
                episodes["scene fire_smoke"]["status"], "ended_by_routine"
            )
            self.assertEqual(
                len(episodes["scene fire_smoke"]["coverage_gap_ids"]), 1
            )
            self.assertEqual(episodes["object take_object"]["coverage_gap_ids"], [])
            self.assertEqual(
                len(l1[0]["incident_dispositions"]),
                len(l1[0]["temporal_observations"]),
            )
            self.assertTrue(
                all(
                    item["scale_disposition"] == "routine_at_this_scale"
                    for item in episodes.values()
                )
            )

            l2 = manager._build_rollup_level(
                7, "L2", "L1", 3600, l1, synthesize=False
            )
            l3 = manager._build_rollup_level(
                7, "L3", "L2", 28800, l2, synthesize=False
            )
            self.assertEqual(len(l2), 1)
            self.assertEqual(len(l3), 1)
            for node in (l2[0], l3[0]):
                higher = {
                    item["semantic_key"]: item
                    for item in node["incident_ledger"]
                }
                self.assertEqual(
                    set(higher),
                    {"scene fire_smoke", "object take_object"},
                )
                self.assertEqual(
                    higher["object take_object"]["status"],
                    "ended_by_routine",
                )
                self.assertEqual(
                    len(higher["object take_object"]["observation_ids"]),
                    1,
                )
                self.assertEqual(
                    len(higher["scene fire_smoke"]["coverage_gap_ids"]),
                    1,
                )
                self.assertEqual(
                    len(node["incident_dispositions"]),
                    len(node["temporal_observations"]),
                )

    def test_two_grounded_routine_windows_close_unmarked_posture_episode(self):
        event_rows = LuxriotManager._l0_temporal_observations(
            channel_id=112,
            source_batch_id="l0-slump",
            batch_state={
                "events": [
                    {
                        "event_id": "person_posture_change",
                        "label": "Person leaning forward, head resting on desk",
                        "summary": "Person changes from typing to head on desk.",
                        "state": "new",
                        "novelty": "novel",
                        "pass_up": True,
                        "snapshot_indices": [4, 5, 6],
                    }
                ]
            },
            batch_start_ms=500,
            batch_end_ms=1_000,
        )

        def routine_child(
            rollup_id: str,
            start_sec: float,
            label: str,
            *,
            coverage_gap: bool = False,
        ) -> Dict[str, Any]:
            return {
                "rollup_id": rollup_id,
                "channel_id": 112,
                "window_start": start_sec,
                "window_end": start_sec + 0.5,
                "coverage_gap": coverage_gap,
                "temporal_observations": [],
                "routine_ledger": [
                    {
                        "semantic_key": "person routine",
                        "label": label,
                        "scope_level": "L0",
                        "state": "continuing",
                        "support_windows": 1,
                        "covered_windows": 1,
                        "unknown_windows": 0,
                        "returned_count": 0,
                        "source_ids": [rollup_id],
                    }
                ],
            }

        children = [
            {
                "rollup_id": "l0-slump",
                "channel_id": 112,
                "window_start": 0.5,
                "window_end": 1.0,
                "coverage_gap": False,
                "temporal_observations": event_rows,
                "routine_ledger": [],
            },
            # Persisting abnormal posture must not count as recovery.
            routine_child("l0-still-slumped", 1.5, "Person slumped over desk"),
            routine_child("l0-typing-1", 2.5, "Person seated and typing"),
            routine_child("l0-typing-2", 3.5, "Person continues typing"),
        ]

        memory = LuxriotManager._aggregate_temporal_memory(children, level="L1")

        self.assertEqual(len(memory["incident_ledger"]), 1)
        episode = memory["incident_ledger"][0]
        self.assertEqual(episode["semantic_key"], "person fall")
        self.assertEqual(episode["status"], "ended_by_routine")
        inferred = [
            row
            for row in memory["temporal_observations"]
            if row.get("boundary_inference") == "two_covered_l0_routines"
        ]
        self.assertEqual(len(inferred), 1)
        self.assertEqual(inferred[0]["applies_to"], ["person fall"])
        self.assertEqual(
            inferred[0]["evidence_refs"],
            ["l0-typing-1", "l0-typing-2"],
        )

    def test_coverage_gap_resets_inferred_routine_confirmation(self):
        event_rows = LuxriotManager._l0_temporal_observations(
            channel_id=112,
            source_batch_id="l0-phone",
            batch_state={
                "events": [
                    {
                        "event_id": "person_phone_use",
                        "label": "Person speaking on a phone",
                        "state": "new",
                        "novelty": "novel",
                        "pass_up": True,
                        "snapshot_indices": [2, 3],
                    }
                ]
            },
            batch_start_ms=500,
            batch_end_ms=1_000,
        )
        routine = {
            "semantic_key": "person routine",
            "label": "Person seated and typing",
            "scope_level": "L0",
            "state": "continuing",
            "support_windows": 1,
            "covered_windows": 1,
            "unknown_windows": 0,
            "returned_count": 0,
            "legacy_unstructured": False,
        }
        children = [
            {
                "rollup_id": "l0-phone",
                "window_start": 0.5,
                "window_end": 1.0,
                "temporal_observations": event_rows,
                "routine_ledger": [],
            },
            {
                "rollup_id": "l0-routine-before-gap",
                "window_start": 1.5,
                "window_end": 2.0,
                "temporal_observations": [],
                "routine_ledger": [routine],
            },
            {
                "rollup_id": "l0-gap",
                "window_start": 2.5,
                "window_end": 3.0,
                "coverage_gap": True,
                "temporal_observations": [],
                "routine_ledger": [],
            },
            {
                "rollup_id": "l0-routine-after-gap",
                "window_start": 3.5,
                "window_end": 4.0,
                "temporal_observations": [],
                "routine_ledger": [routine],
            },
        ]

        memory = LuxriotManager._aggregate_temporal_memory(children, level="L1")

        self.assertEqual(memory["incident_ledger"][0]["status"], "open")
        self.assertFalse(
            any(
                row.get("boundary_inference")
                for row in memory["temporal_observations"]
            )
        )

    def test_isolated_lower_scale_routine_becomes_long_incident_candidate(self):
        children = [
            {
                "rollup_id": f"l1-{index}",
                "coverage_gap": False,
                "routine_ledger": (
                    [
                        {
                            "semantic_key": "person_away_from_desk",
                            "label": "Person remains away from desk",
                            "support_windows": 1,
                            "unknown_windows": 0,
                            "returned_count": 0,
                            "source_ids": ["l0-source"],
                        }
                    ]
                    if index == 0
                    else []
                ),
            }
            for index in range(4)
        ]

        ledger = LuxriotManager._aggregate_routine_ledger(children, level="L2")

        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["state"], "isolated")
        self.assertEqual(
            ledger[0]["scale_disposition"], "long_incident_candidate"
        )

    def test_temporal_memory_rejects_routine_micro_motion_incidents(self):
        observations = LuxriotManager._l0_temporal_observations(
            channel_id=112,
            source_batch_id="vlm-room-1",
            batch_state={
                "events": [
                    {
                        "event_id": "person_head_movement_112_000001",
                        "label": "Person adjusts gaze and turns head slightly",
                        "state": "new",
                        "novelty": "novel",
                        "pass_up": True,
                        "snapshot_indices": [3],
                    },
                    {
                        "event_id": "person_entry_112_20260805",
                        "label": "Person enters from the right doorway",
                        "state": "new",
                        "novelty": "novel",
                        "pass_up": True,
                        "snapshot_indices": [5, 6],
                    },
                ],
                "routines": [],
            },
            batch_start_ms=1_000,
            batch_end_ms=2_000,
        )

        events = [row for row in observations if row["kind"] == "event"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["label"], "Person enters from the right doorway")
        self.assertEqual(events[0]["semantic_key"], "person enter")
        self.assertEqual(events[0]["trigger_kind"], "episode_event")

    def test_temporal_memory_rejects_generic_person_movement(self):
        observations = LuxriotManager._l0_temporal_observations(
            channel_id=112,
            source_batch_id="vlm-room-2",
            batch_state={
                "events": [
                    {
                        "event_id": "person_movement_212108",
                        "label": "person moving",
                        "state": "new",
                        "novelty": "novel",
                        "pass_up": True,
                        "snapshot_indices": [1, 2, 3],
                    }
                ]
            },
            batch_start_ms=1_000,
            batch_end_ms=2_000,
        )

        self.assertEqual(observations, [])

    def test_temporal_memory_keeps_phone_call_as_distinct_context_episode(self):
        observations = LuxriotManager._l0_temporal_observations(
            channel_id=112,
            source_batch_id="vlm-room-phone",
            batch_state={
                "events": [
                    {
                        "event_id": "person_phone_use",
                        "label": "Person speaking on a phone",
                        "summary": "Person holds a phone to the ear and speaks.",
                        "state": "new",
                        "novelty": "novel",
                        "pass_up": True,
                        "snapshot_indices": [3, 4, 5],
                    }
                ]
            },
            batch_start_ms=1_000,
            batch_end_ms=2_000,
        )

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["semantic_key"], "person phone_call")
        self.assertEqual(observations[0]["trigger_kind"], "episode_event")

    def test_temporal_memory_treats_transition_to_head_on_desk_as_safety(self):
        observations = LuxriotManager._l0_temporal_observations(
            channel_id=112,
            source_batch_id="vlm-room-slumped",
            batch_state={
                "events": [
                    {
                        "event_id": "person_posture_change",
                        "label": "Person leaning forward, head resting on desk",
                        "summary": "Person transitions from typing to resting head on desk.",
                        "state": "new",
                        "novelty": "novel",
                        "pass_up": True,
                        "snapshot_indices": [4, 5, 6, 7, 8],
                    }
                ]
            },
            batch_start_ms=1_000,
            batch_end_ms=2_000,
        )

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["semantic_key"], "person fall")
        self.assertEqual(observations[0]["trigger_kind"], "safety_event")

    def test_temporal_memory_preserves_structured_alert_without_event_duplicate(self):
        observations = LuxriotManager._l0_temporal_observations(
            channel_id=112,
            source_batch_id="vlm-room-alert-1",
            batch_state={
                "events": [],
                "alerts": [
                    {
                        "title": "Person collapse near desk",
                        "description": "Person visibly falls from the chair.",
                        "severity": "high",
                        "state": "new",
                        "snapshot_indices": [3, 4],
                    }
                ],
            },
            batch_start_ms=1_000,
            batch_end_ms=2_000,
        )

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["semantic_key"], "person fall")
        self.assertEqual(observations[0]["trigger_kind"], "safety_alert")
        self.assertEqual(
            observations[0]["evidence_refs"],
            ["vlm-room-alert-1:snapshot:3", "vlm-room-alert-1:snapshot:4"],
        )

    def test_temporal_memory_admits_configured_gesture_once_and_marks_priority(self):
        observations = LuxriotManager._l0_temporal_observations(
            channel_id=112,
            source_batch_id="vlm-room-alert-gesture",
            operator_alert_policy=(
                "Alert if you spot a thumbs-up gesture, severity - info\n"
                "Alert if you spot a victory gesture, severity - normal"
            ),
            batch_state={
                "events": [
                    {
                        "event_id": "thumb_up_gesture",
                        "label": "Thumb-up gesture",
                        "summary": "Novel gesture observed in snapshot 3.",
                        "state": "new",
                        "novelty": "novel",
                        "pass_up": True,
                        "snapshot_indices": [3],
                    }
                ],
                "alerts": [
                    {
                        "title": "Thumb-up gesture detected",
                        "description": "Person raised thumb in snapshot 3.",
                        "severity": "info",
                        "state": "new",
                        "snapshot_indices": [3],
                    }
                ],
            },
            batch_start_ms=1_000,
            batch_end_ms=2_000,
        )

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["semantic_key"], "person thumbs_up")
        self.assertEqual(observations[0]["trigger_kind"], "operator_alert")
        self.assertEqual(observations[0]["severity"], "info")
        self.assertIn("thumbs-up gesture", observations[0]["operator_criterion"])

    def test_temporal_memory_rejects_unconfigured_model_alert_micro_motion(self):
        observations = LuxriotManager._l0_temporal_observations(
            channel_id=112,
            source_batch_id="vlm-room-alert-noise",
            operator_alert_policy="Alert if you spot a thumbs-up gesture, severity - info",
            batch_state={
                "events": [
                    {
                        "event_id": "person_gesture",
                        "label": "Person raises hand to face",
                        "summary": "Ordinary brief hand movement.",
                        "state": "new",
                        "novelty": "novel",
                        "pass_up": True,
                        "snapshot_indices": [6],
                    }
                ],
                "alerts": [
                    {
                        "title": "Person raising hand to face",
                        "description": "Unusual gesture observed in snapshot 6.",
                        "severity": "info",
                        "state": "new",
                        "snapshot_indices": [6],
                    }
                ],
            },
            batch_start_ms=1_000,
            batch_end_ms=2_000,
        )

        self.assertEqual(observations, [])

    def test_temporal_memory_matches_exit_wording_to_configured_leave_criterion(self):
        observations = LuxriotManager._l0_temporal_observations(
            channel_id=112,
            source_batch_id="vlm-room-alert-exit",
            operator_alert_policy=(
                "Alert if person entering or leaving scene, severity - info"
            ),
            batch_state={
                "alerts": [
                    {
                        "title": "Person exiting frame",
                        "description": "Person exits through the right doorway.",
                        "severity": "info",
                        "state": "new",
                        "snapshot_indices": [5],
                    }
                ],
            },
            batch_start_ms=1_000,
            batch_end_ms=2_000,
        )

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["semantic_key"], "person exit")
        self.assertEqual(observations[0]["trigger_kind"], "operator_alert")

    def test_operator_cat_policy_does_not_match_person_using_cat_statue(self):
        criterion = "cat entering or leaving scene, severity - high"

        self.assertIsNone(
            LuxriotManager._operator_policy_event_match(
                criterion,
                {
                    "label": "Person interacts with cat statue",
                    "summary": "Person enters, touches the statue, then exits.",
                    "state": "new",
                    "snapshot_indices": [3, 4, 5],
                },
            )
        )
        self.assertIsNotNone(
            LuxriotManager._operator_policy_event_match(
                criterion,
                {
                    "label": "Cat enters the scene",
                    "summary": "Cat walks into view and later leaves.",
                    "state": "new",
                    "snapshot_indices": [2, 6],
                },
            )
        )

    def test_temporal_alert_merge_preserves_highest_severity(self):
        observations = LuxriotManager._l0_temporal_observations(
            channel_id=112,
            source_batch_id="vlm-room-duplicate-alert",
            operator_alert_policy=(
                "Alert if person entering or leaving scene, severity - high"
            ),
            batch_state={
                "events": [
                    {
                        "event_id": "person_exit",
                        "label": "Person exits the scene",
                        "summary": "Person leaves the camera view.",
                        "state": "new",
                        "novelty": "novel",
                        "pass_up": True,
                        "snapshot_indices": [5],
                    }
                ],
                "alerts": [
                    {
                        "title": "Person exits the scene",
                        "description": "Person leaves the camera view.",
                        "severity": "high",
                        "state": "new",
                        "snapshot_indices": [5],
                    },
                    {
                        "title": "Person exits the scene",
                        "description": "Person leaves the camera view.",
                        "severity": "info",
                        "state": "new",
                        "snapshot_indices": [5],
                    },
                ],
            },
            batch_start_ms=1_000,
            batch_end_ms=2_000,
        )

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["severity"], "high")

    def test_operator_leave_policy_does_not_treat_left_direction_as_egress(self):
        criterion = "person entering or leaving scene, severity - info"

        self.assertIsNone(
            LuxriotManager._operator_policy_event_match(
                criterion,
                {
                    "label": "Person turning head",
                    "summary": "Person turns head slightly, then further left.",
                    "state": "new",
                    "snapshot_indices": [7, 8],
                },
            )
        )
        self.assertIsNotNone(
            LuxriotManager._operator_policy_event_match(
                criterion,
                {
                    "label": "Person left the scene",
                    "summary": "The person left the camera view through the doorway.",
                    "state": "new",
                    "snapshot_indices": [3, 4],
                },
            )
        )

    def test_temporal_memory_rejects_false_leave_alert_from_left_direction(self):
        observations = LuxriotManager._l0_temporal_observations(
            channel_id=112,
            source_batch_id="vlm-room-head-left",
            operator_alert_policy=(
                "Alert if person entering or leaving scene, severity - info"
            ),
            batch_state={
                "events": [
                    {
                        "event_id": "person_head_movement",
                        "label": "Person turning head",
                        "summary": "Person turns head slightly, then further left.",
                        "state": "new",
                        "novelty": "novel",
                        "pass_up": True,
                        "snapshot_indices": [7, 8],
                    }
                ],
                "alerts": [
                    {
                        "title": "Person turning head — operator criterion",
                        "description": "Person turns head slightly, then further left.",
                        "severity": "info",
                        "state": "new",
                        "snapshot_indices": [7, 8],
                    }
                ],
            },
            batch_start_ms=1_000,
            batch_end_ms=2_000,
        )

        self.assertEqual(observations, [])

    def test_l2_temporal_composition_is_rooted_in_safety_and_keeps_context_nested(self):
        collision = make_temporal_observation(
            channel_id=112,
            source_batch_id="batch-collision",
            ordinal=0,
            kind="event",
            state="new",
            semantic_key="vehicle collision",
            label="Two vehicles collided",
            start_ms=10_000,
            end_ms=12_000,
            evidence_refs=["batch-collision:snapshot:3"],
        ).to_dict()
        collision.update(trigger_kind="safety_alert", severity="critical")
        phone = make_temporal_observation(
            channel_id=112,
            source_batch_id="batch-phone",
            ordinal=0,
            kind="event",
            state="new",
            semantic_key="person phone_call",
            label="Bystander calls by phone",
            start_ms=30_000,
            end_ms=32_000,
            evidence_refs=["batch-phone:snapshot:4"],
        ).to_dict()
        phone.update(trigger_kind="episode_event", severity="info")
        memory = LuxriotManager._aggregate_temporal_memory(
            [
                {"temporal_observations": [collision]},
                {"temporal_observations": [phone]},
            ],
            level="L2",
        )

        self.assertEqual(len(memory["incident_compositions"]), 1)
        composition = memory["incident_compositions"][0]
        episodes = {
            item["semantic_key"]: item for item in memory["incident_ledger"]
        }
        self.assertEqual(episodes["vehicle collision"]["priority"], "safety")
        self.assertEqual(episodes["person phone_call"]["priority"], "context")
        self.assertEqual(
            composition["parent_episode_id"],
            episodes["vehicle collision"]["episode_id"],
        )
        self.assertEqual(
            composition["parent_observation_ids"],
            [collision["observation_id"]],
        )
        self.assertIn(
            episodes["person phone_call"]["episode_id"],
            composition["nested_episode_ids"],
        )
        self.assertFalse(composition["automatic_merge"])
        self.assertTrue(composition["operator_review_required"])

    def test_l2_temporal_composition_does_not_promote_info_operator_signal(self):
        thumb = make_temporal_observation(
            channel_id=112,
            source_batch_id="batch-thumb",
            ordinal=0,
            kind="event",
            state="new",
            semantic_key="person thumbs_up",
            label="Thumb-up gesture",
            start_ms=10_000,
            end_ms=11_000,
        ).to_dict()
        thumb.update(trigger_kind="operator_alert", severity="info")
        head = make_temporal_observation(
            channel_id=112,
            source_batch_id="batch-head",
            ordinal=0,
            kind="event",
            state="new",
            semantic_key="person head_turn",
            label="Person turns head",
            start_ms=12_000,
            end_ms=13_000,
        ).to_dict()
        head.update(trigger_kind="episode_event", severity="info")
        memory = LuxriotManager._aggregate_temporal_memory(
            [{"temporal_observations": [thumb, head]}],
            level="L2",
        )

        self.assertNotIn("incident_compositions", memory)

    def test_l3_temporal_composition_requires_distinct_l2_source_windows(self):
        collision = make_temporal_observation(
            channel_id=112,
            source_batch_id="batch-cross-window-collision",
            ordinal=0,
            kind="event",
            state="continuing",
            semantic_key="vehicle collision",
            label="Two vehicles remain at the collision scene",
            start_ms=3_590_000,
            end_ms=3_599_000,
        ).to_dict()
        collision.update(trigger_kind="safety_alert", severity="critical")
        phone = make_temporal_observation(
            channel_id=112,
            source_batch_id="batch-cross-window-phone",
            ordinal=0,
            kind="event",
            state="new",
            semantic_key="person phone_call",
            label="Bystander calls by phone",
            start_ms=3_601_000,
            end_ms=3_605_000,
        ).to_dict()
        phone.update(trigger_kind="episode_event", severity="info")

        cross_window = LuxriotManager._aggregate_temporal_memory(
            [
                {
                    "rollup_id": "l2-ch112-w3600-0",
                    "temporal_observations": [collision],
                },
                {
                    "rollup_id": "l2-ch112-w3600-3600",
                    "temporal_observations": [phone],
                },
            ],
            level="L3",
        )
        one_window = LuxriotManager._aggregate_temporal_memory(
            [
                {
                    "rollup_id": "l2-ch112-w3600-0",
                    "temporal_observations": [collision, phone],
                }
            ],
            level="L3",
        )

        self.assertEqual(len(cross_window["incident_compositions"]), 1)
        composition = cross_window["incident_compositions"][0]
        self.assertEqual(composition["level"], "L3")
        self.assertTrue(composition["cross_window"])
        self.assertEqual(composition["source_window_count"], 2)
        self.assertEqual(
            composition["source_rollup_ids"],
            ["l2-ch112-w3600-0", "l2-ch112-w3600-3600"],
        )
        self.assertNotIn("incident_compositions", one_window)

    def test_high_operator_composition_keeps_same_entity_and_rejects_chain_noise(self):
        parent = {
            "episode_id": "episode-cat-enter",
            "channel_id": 112,
            "semantic_key": "cat enter",
            "priority": "operator_criterion",
            "severity": "high",
            "status": "open",
            "start_ms": 10_000,
            "last_observed_ms": 12_000,
            "observation_ids": ["obs-cat-enter"],
            "evidence_refs": [],
        }
        cat_motion = {
            "episode_id": "episode-cat-motion",
            "channel_id": 112,
            "semantic_key": "cat motion",
            "priority": "context",
            "status": "open",
            "start_ms": 20_000,
            "last_observed_ms": 21_000,
            "observation_ids": ["obs-cat-motion"],
            "evidence_refs": [],
        }
        person_exit = {
            "episode_id": "episode-person-exit",
            "channel_id": 112,
            "semantic_key": "person exit",
            "priority": "context",
            "status": "open",
            "start_ms": 22_000,
            "last_observed_ms": 23_000,
            "observation_ids": ["obs-person-exit"],
            "evidence_refs": [],
        }
        late_cat = {
            "episode_id": "episode-late-cat",
            "channel_id": 112,
            "semantic_key": "cat jump_climb",
            "priority": "context",
            "status": "open",
            "start_ms": 10_000 + 6 * 60 * 1000,
            "last_observed_ms": 10_000 + 6 * 60 * 1000 + 1_000,
            "observation_ids": ["obs-late-cat"],
            "evidence_refs": [],
        }

        compositions = LuxriotManager._compose_temporal_incident_candidates(
            [parent, cat_motion, person_exit, late_cat],
            level="L2",
        )

        self.assertEqual(len(compositions), 1)
        self.assertEqual(
            compositions[0]["semantic_keys"],
            ["cat enter", "cat motion"],
        )
        self.assertNotIn("episode-person-exit", compositions[0]["episode_ids"])
        self.assertNotIn("episode-late-cat", compositions[0]["episode_ids"])

    def test_observed_gap_boundary_does_not_extend_episode_for_composition(self):
        old_alert = {
            "episode_id": "episode-old-cat",
            "channel_id": 112,
            "semantic_key": "cat enter",
            "priority": "operator_criterion",
            "severity": "high",
            "status": "ended_by_observed_gap",
            "start_ms": 10_000,
            "last_observed_ms": 12_000,
            "boundary_at_ms": 60 * 60 * 1000,
            "observation_ids": ["obs-old-cat"],
            "evidence_refs": [],
        }
        near_late_boundary = {
            "episode_id": "episode-late-motion",
            "channel_id": 112,
            "semantic_key": "cat movement",
            "priority": "context",
            "status": "open",
            "start_ms": 60 * 60 * 1000 + 1_000,
            "last_observed_ms": 60 * 60 * 1000 + 2_000,
            "observation_ids": ["obs-late-motion"],
            "evidence_refs": [],
        }

        compositions = LuxriotManager._compose_temporal_incident_candidates(
            [old_alert, near_late_boundary],
            level="L2",
        )

        self.assertEqual(compositions, [])

    def test_rollup_incident_dispatch_retries_failure_then_coalesces_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            calls = []

            def callback(channel_id, node):
                calls.append((channel_id, node["rollup_id"]))
                if len(calls) == 1:
                    raise RuntimeError("incident store warming")
                return {"attached": 1}

            manager.set_rollup_incident_callback(callback)
            node = {
                "rollup_id": "l2-ch112-w3600-0",
                "source_signature": "source-a",
                "incident_compositions": [{"composition_id": "composition-a"}],
            }

            manager._dispatch_rollup_incident_compositions(112, [node])
            self.assertIn("incident_composition_error", node)
            manager._dispatch_rollup_incident_compositions(112, [node])
            manager._dispatch_rollup_incident_compositions(112, [node])

            self.assertEqual(calls, [(112, node["rollup_id"])] * 2)
            self.assertEqual(node["incident_composition_result"], {"attached": 1})

    def test_closed_l3_build_dispatches_only_cross_window_composition(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager.rollup_llm_levels = set()
            calls = []
            manager.set_rollup_incident_callback(
                lambda channel_id, node: calls.append(
                    (channel_id, node["rollup_id"], node["incident_compositions"])
                )
                or {"attached": 1}
            )
            collision = make_temporal_observation(
                channel_id=112,
                source_batch_id="batch-l3-dispatch-collision",
                ordinal=0,
                kind="event",
                state="continuing",
                semantic_key="vehicle collision",
                label="Vehicles remain at a collision scene",
                start_ms=3_590_000,
                end_ms=3_599_000,
            ).to_dict()
            collision.update(trigger_kind="safety_alert", severity="critical")
            phone = make_temporal_observation(
                channel_id=112,
                source_batch_id="batch-l3-dispatch-phone",
                ordinal=0,
                kind="event",
                state="new",
                semantic_key="person phone_call",
                label="Bystander calls by phone",
                start_ms=3_601_000,
                end_ms=3_605_000,
            ).to_dict()
            phone.update(trigger_kind="episode_event", severity="info")
            sources = [
                {
                    "rollup_id": "l2-ch112-w3600-0",
                    "channel_id": 112,
                    "window_start": 0.0,
                    "window_end": 3_600.0,
                    "item_count": 1,
                    "frame_count": 1,
                    "source_tokens": 10,
                    "run_ids": ["run-a"],
                    "summary": "Collision remains visible.",
                    "temporal_observations": [collision],
                },
                {
                    "rollup_id": "l2-ch112-w3600-3600",
                    "channel_id": 112,
                    "window_start": 3_600.0,
                    "window_end": 7_200.0,
                    "item_count": 1,
                    "frame_count": 1,
                    "source_tokens": 10,
                    "run_ids": ["run-a"],
                    "summary": "A bystander uses a phone.",
                    "temporal_observations": [phone],
                },
            ]

            nodes = manager._build_rollup_level(
                112,
                "L3",
                "L2",
                28_800,
                sources,
                synthesize=True,
            )

            self.assertEqual(len(nodes), 1)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], 112)
            self.assertTrue(calls[0][2][0]["cross_window"])
            self.assertEqual(
                nodes[0]["incident_composition_result"],
                {"attached": 1},
            )

    def test_temporal_keys_ignore_generated_ids_and_preserve_parallel_events(self):
        first = LuxriotManager._l0_temporal_observations(
            channel_id=112,
            source_batch_id="vlm-room-1",
            batch_state={
                "events": [
                    {
                        "event_id": "cat_leap_112_000001",
                        "label": "Sphynx cat leaps from the shelf",
                        "state": "new",
                        "novelty": "novel",
                        "pass_up": True,
                    },
                    {
                        "event_id": "person_entry_112_20260805_210000",
                        "label": "Person enters the room",
                        "state": "new",
                        "novelty": "novel",
                        "pass_up": True,
                    },
                ]
            },
            batch_start_ms=1_000,
            batch_end_ms=2_000,
        )
        second = LuxriotManager._l0_temporal_observations(
            channel_id=112,
            source_batch_id="vlm-room-2",
            batch_state={
                "events": [
                    {
                        "event_id": "cat_jumping_frame_8",
                        "label": "The cat jumps down from the shelf",
                        "state": "continuing",
                        "novelty": "continuing",
                        "pass_up": True,
                    }
                ]
            },
            batch_start_ms=3_000,
            batch_end_ms=4_000,
        )

        self.assertEqual(
            {row["semantic_key"] for row in first},
            {"cat jump_climb", "person enter"},
        )
        self.assertEqual(second[0]["semantic_key"], "cat jump_climb")
        memory = LuxriotManager._aggregate_temporal_memory(
            [
                {"temporal_observations": first},
                {"temporal_observations": second},
            ],
            level="L1",
        )
        episodes = {
            row["semantic_key"]: row for row in memory["incident_ledger"]
        }
        self.assertEqual(set(episodes), {"cat jump_climb", "person enter"})
        self.assertEqual(
            len(episodes["cat jump_climb"]["observation_ids"]),
            2,
        )

    def test_temporal_keys_merge_reworded_vehicle_drift(self):
        keys = {
            LuxriotManager._temporal_event_semantic_key(
                {
                    "event_id": "drift_smoke_vehicle",
                    "label": "vehicle drifting with smoke",
                }
            ),
            LuxriotManager._temporal_event_semantic_key(
                {
                    "event_id": "drift_and_pedestrian_presence",
                    "label": "Orange car drifting with smoke, pedestrians standing",
                }
            ),
        }

        self.assertEqual(keys, {"vehicle maneuver"})

    def test_temporal_memory_merges_plural_vehicle_event_and_alert(self):
        observations = LuxriotManager._l0_temporal_observations(
            channel_id=118,
            source_batch_id="vlm-traffic-drift",
            batch_state={
                "events": [
                    {
                        "event_id": "drift_smoke",
                        "label": "Drift with smoke",
                        "summary": "Orange and red cars drift with smoke",
                        "state": "new",
                        "novelty": "novel",
                        "pass_up": True,
                        "snapshot_indices": [5, 6, 7, 8],
                    }
                ],
                "alerts": [
                    {
                        "title": "Dangerous vehicle behavior",
                        "description": "Orange and red cars drift with smoke",
                        "severity": "normal",
                        "state": "new",
                        "snapshot_indices": [5, 6, 7, 8],
                    }
                ],
            },
            batch_start_ms=1_000,
            batch_end_ms=2_000,
        )

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["semantic_key"], "vehicle maneuver")
        self.assertEqual(observations[0]["trigger_kind"], "safety_alert")
        self.assertEqual(
            observations[0]["evidence_refs"],
            [
                "vlm-traffic-drift:snapshot:5",
                "vlm-traffic-drift:snapshot:6",
                "vlm-traffic-drift:snapshot:7",
                "vlm-traffic-drift:snapshot:8",
            ],
        )

    def test_rollup_event_ledger_drops_static_pass_up_noise(self):
        ledger = LuxriotManager._l0_event_ledger(
            rollup_id="l0-ch7-static",
            batch_state={
                "events": [
                    {
                        "event_id": "static_scene",
                        "label": "Person resting on couch",
                        "state": "continuing",
                        "summary": "Person remains stationary on couch across all snapshots.",
                        "novelty": "routine",
                        # A small VLM may set this incorrectly. The chronology
                        # must not treat it as authority.
                        "pass_up": True,
                    },
                    {
                        "event_id": "cat_chase",
                        "label": "Two cats chase each other across the room",
                        "state": "new",
                        "summary": "The cats run from the shelf toward the door.",
                        "novelty": "novel",
                        "pass_up": True,
                    },
                    {
                        "event_id": "no_events",
                        "label": "No observable events",
                        "state": "new",
                        "summary": "No new or resolved events are visible.",
                        "novelty": "novel",
                        "pass_up": True,
                    },
                ]
            },
            alert_events=[],
            transition_events=[],
            default_timestamp_ms=1_781_683_200_000,
        )

        self.assertEqual(
            [item["label"] for item in ledger],
            ["Two cats chase each other across the room"],
        )

    def test_rollup_event_ledger_groups_repeated_observations_without_claiming_occurrences(self):
        children = []
        base_ms = 1_781_683_200_000
        for index, state in enumerate(("new", "continuing", "continuing")):
            timestamp_ms = base_ms + index * 60_000
            children.append(
                {
                    "event_ledger": [
                        {
                            "id": f"l0-{index}:event:0",
                            "event_key": "person_leaves_desk",
                            "kind": "event",
                            "timestamp_ms": timestamp_ms,
                            "first_observed_ms": timestamp_ms,
                            "last_observed_ms": timestamp_ms,
                            "observation_count": 1,
                            "new_observation_count": 1 if state == "new" else 0,
                            "label": "Person leaves the desk",
                            "state": state,
                            "novelty": "deviation",
                            "pass_up": True,
                            "source_id": f"l0-{index}",
                        }
                    ]
                }
            )

        ledger = LuxriotManager._aggregate_event_ledger(children)

        self.assertEqual(len(ledger), 1)
        episode = ledger[0]
        self.assertEqual(episode["observation_count"], 3)
        self.assertEqual(episode["new_observation_count"], 1)
        self.assertEqual(episode["first_observed_ms"], base_ms)
        self.assertEqual(episode["last_observed_ms"], base_ms + 120_000)
        self.assertNotIn("occurrence_count", episode)
        self.assertEqual(episode["source_ids"], ["l0-0", "l0-1", "l0-2"])

    def test_rollup_event_ledger_groups_repeated_alert_emissions_as_one_episode(self):
        children = [
            {
                "event_ledger": [
                    {
                        "id": f"l0-{index}:alert:0",
                        "kind": "alert",
                        "timestamp_ms": 1_781_683_200_000 + index * 10_000,
                        "label": "Person in white shirt seated in front of camera",
                        "severity": "low",
                        "pass_up": True,
                        "source_id": f"l0-{index}",
                    }
                ]
            }
            for index in range(8)
        ]

        ledger = LuxriotManager._aggregate_event_ledger(children)

        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["kind"], "alert")
        self.assertEqual(ledger[0]["observation_count"], 8)
        self.assertEqual(ledger[0]["first_observed_ms"], 1_781_683_200_000)
        self.assertEqual(ledger[0]["last_observed_ms"], 1_781_683_270_000)

    def test_older_memory_update_preserves_deviation_without_resurrecting_watchlist(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(Path(temp))
            manager._update_channel_routine_context(
                channel_id=7,
                rollup_id="l1-new",
                window_end=200.0,
                level="L1",
                memory_update={"active_watchlist": ["north door"]},
                summary_text="",
            )
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
            self.assertNotIn("east gate vehicle", routine_text)
            self.assertIn("vehicle drifting", routine_text)
            self.assertNotIn("keep drifting visible", routine_text)
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

    def test_start_session_applies_supported_small_batch_without_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                config_overrides={
                    "LUXRIOT_BATCH_SIZES": (4, 8, 12, 16),
                    "LUXRIOT_DEFAULT_BATCH_SIZE": 12,
                },
            )

            with patch.object(LuxriotCaptureSession, "start", return_value=None):
                status = manager.start_session(7, batch_size=8)

            self.assertEqual(status["batch_size"], 8)
            manager.stop_session(7)

    def test_start_session_rejects_unsupported_batch_instead_of_substituting(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = build_manager(
                Path(temp),
                config_overrides={
                    "LUXRIOT_BATCH_SIZES": (4, 8, 12, 16),
                    "LUXRIOT_DEFAULT_BATCH_SIZE": 12,
                },
            )

            with self.assertRaisesRegex(ValueError, "Unsupported batch_size 24"):
                manager.start_session(7, batch_size=24)

            self.assertNotIn(7, manager.sessions)

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

    def test_superseded_queued_generation_never_starts_lm_or_marks_current_session_failed(self):
        lm_started = threading.Event()
        release_lm = threading.Event()
        lm_calls = []

        def blocking_lm(_messages, _model):
            lm_calls.append("called")
            lm_started.set()
            release_lm.wait(timeout=3.0)
            return "old generation summary"

        self.manager.lm_callback = blocking_lm
        with (
            patch.object(LuxriotCaptureSession, "start", return_value=None),
            patch.object(LuxriotCaptureSession, "stop", return_value=None),
        ):
            self.manager.start_session(7, batch_size=2, prompt="Describe old generation.")
            old_session = self.manager.sessions[7]
            first_batch = self.manager.create_summary_batch(
                channel_id=7,
                run_id=old_session.run_id,
                batch_size=2,
                prompt="First old batch.",
                model_hint=None,
                interval_sec=1.0,
                frames=sample_frames(100.0),
                session_generation=old_session.session_generation,
            )
            queued_batch = self.manager.create_summary_batch(
                channel_id=7,
                run_id=old_session.run_id,
                batch_size=2,
                prompt="Queued old batch.",
                model_hint=None,
                interval_sec=1.0,
                frames=sample_frames(200.0),
                session_generation=old_session.session_generation,
            )
            self.assertTrue(self.runtime.enqueue_summary(first_batch, workload_class="manual")["accepted"])
            self.runtime.start()
            self.assertTrue(lm_started.wait(timeout=2.0))
            self.assertTrue(self.runtime.enqueue_summary(queued_batch, workload_class="manual")["accepted"])

            self.manager.start_session(7, batch_size=2, prompt="Describe current generation.")
            current_session = self.manager.sessions[7]

        release_lm.set()
        self.wait_for(lambda: self.runtime.status()["superseded_count"] == 1)

        self.assertEqual(lm_calls, ["called"])
        self.assertEqual(self.runtime.status()["failed_count"], 0)
        self.assertEqual(self.runtime.status()["superseded_count"], 1)
        self.assertEqual(current_session.summary_failed_batches, 0)
        self.assertIsNone(current_session.summary_last_error)

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
        # The archive callback runs before the accepted entry is published to
        # summary_history. Wait for the worker-visible completion boundary so
        # this assertion cannot race the next line in accept_summary_entry().
        self.wait_for(lambda: bool(self.manager.summary_history.get(7)))

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

    def test_l0_heartbeat_windows_remain_distinct_in_durable_queue(self):
        self.runtime.stop()
        repository = InMemoryInferenceQueueRepository()
        runtime = LuxriotInferenceQueueRuntime(
            manager=self.manager,
            enqueue_repository=repository,
            worker_repository=repository,
            tenant_id=str(uuid4()),
            capacity=2,
            spool_directory=self.directory / "spool-distinct",
            default_model="qwen35-9b-q4_k_m",
            worker_count=0,
        )
        first = runtime.enqueue_summary(self.batch(start=100.0))
        second = runtime.enqueue_summary(self.batch(start=200.0))

        self.assertNotEqual(first["job_id"], second["job_id"])
        self.assertEqual(first["status"], "enqueued")
        self.assertEqual(second["status"], "enqueued")
        self.assertEqual(repository.metrics_snapshot().coalesced_count, 0)
        self.assertEqual(
            len(list((self.directory / "spool-distinct").glob("*.json"))),
            2,
        )
        jobs = repository.list_jobs()
        self.assertTrue(all(job.payload["coalesce_heartbeat"] is False for job in jobs))

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
        gaps = [
            row
            for row in self.manager.summary_history.get(7, [])
            if row.get("coverage_gap")
        ]
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["gap_reason"], "inference_queue_evicted")


if __name__ == "__main__":
    unittest.main()
