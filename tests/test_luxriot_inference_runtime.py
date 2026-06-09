import os
import stat
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from inference_queue import (
    InMemoryInferenceQueueRepository,
    LuxriotInferenceQueueRuntime,
)
from luxriot_connector import LuxriotCaptureSession, LuxriotManager


def build_manager(directory: Path, lm_callback=None) -> LuxriotManager:
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
