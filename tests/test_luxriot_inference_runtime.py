import os
import stat
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

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
) -> LuxriotManager:
    config = SimpleNamespace(
        LUXRIOT_SYSTEM_PROMPT_DEFAULT="Describe the stream.",
        LUXRIOT_ALERTS_JSON_PROMPT="",
        LUXRIOT_SUMMARY_HISTORY_LIMIT=100,
        LUXRIOT_AUTO_BOOKMARKS=False,
        LUXRIOT_BOOKMARK_COOLDOWN_SEC=60.0,
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
        LUXRIOT_MAX_BUFFER_FRAMES=180,
        LUXRIOT_BASE_URL="http://luxriot.invalid",
        LUXRIOT_USERNAME="",
        LUXRIOT_PASSWORD="",
    )

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
    )


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


class LuxriotCaptureDispatchTests(unittest.TestCase):
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
                get_snapshot=lambda _channel_id: SimpleNamespace(width=1280, height=720)
            )

            def stop_after_one_wait(_interval):
                session.stop_event.set()
                return False

            session.stop_event.wait = stop_after_one_wait
            session._run()

            self.assertEqual(len(session.frames), 1)
            self.assertIn("vlm unavailable", session.last_error)
            self.assertEqual(session.queue_dropped_batches, 1)

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
        self.assertEqual(list((self.directory / "spool").glob("*.json")), [])
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
