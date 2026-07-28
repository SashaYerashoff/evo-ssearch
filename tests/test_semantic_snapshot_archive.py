import math
import threading
import unittest

from semantic_snapshot_archive import (
    SemanticSnapshotArchiveWriter,
    SnapshotSubmitStatus,
)


class RecordingDetectionsStore:
    def __init__(self):
        self.calls = []
        self.records = []
        self.lock = threading.Lock()

    def add_detections(self, records):
        with self.lock:
            batch = [dict(record) for record in records]
            self.calls.append(batch)
            self.records.extend(batch)
            return len(batch)


class FlakyDetectionsStore(RecordingDetectionsStore):
    def __init__(self, failures):
        super().__init__()
        self.failures = failures
        self.attempts = 0

    def add_detections(self, records):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise RuntimeError("database unavailable")
        return super().add_detections(records)


class IdempotentDetectionsStore(RecordingDetectionsStore):
    def ensure_detections(self, records):
        with self.lock:
            batch = [dict(record) for record in records]
            self.calls.append(batch)
            # Simulate ON CONFLICT: the row already exists, but the complete
            # deterministic batch is durably satisfied.
            return len(batch)


def _embedding():
    value = 1.0 / math.sqrt(2.0)
    return [value, value]


class SemanticSnapshotArchiveWriterTests(unittest.TestCase):
    def setUp(self):
        self.writers = []

    def tearDown(self):
        for writer in self.writers:
            writer.stop(drain=True, timeout=2)

    def _writer(self, store=None, **overrides):
        values = {
            "autostart": False,
            "max_queue": 16,
            "batch_size": 8,
            "flush_interval_seconds": 0.01,
            "initial_backoff_seconds": 0.001,
            "max_backoff_seconds": 0.002,
        }
        values.update(overrides)
        writer = SemanticSnapshotArchiveWriter(
            store or RecordingDetectionsStore(),
            **values,
        )
        self.writers.append(writer)
        return writer

    def test_exactly_one_durable_record_per_second_per_channel(self):
        store = RecordingDetectionsStore()
        writer = self._writer(store)

        first = writer.submit(
            channel_id=7,
            timestamp_ms=1_000,
            embedding=_embedding(),
            thumbnail="thumb-7-first",
            provenance={"selection_source": "capture_cv_frame_delta"},
        )
        duplicate = writer.submit(
            channel_id=7,
            timestamp_ms=1_999,
            embedding=_embedding(),
            thumbnail="thumb-7-duplicate",
        )
        other_second = writer.submit(
            channel_id=7,
            timestamp_ms=2_000,
            embedding=_embedding(),
            thumbnail="thumb-7-second",
        )
        other_channel = writer.submit(
            channel_id=8,
            timestamp_ms=1_500,
            embedding=_embedding(),
            thumbnail="thumb-8-first",
        )

        self.assertTrue(first.accepted)
        self.assertEqual(duplicate.status, SnapshotSubmitStatus.DUPLICATE)
        self.assertTrue(other_second.accepted)
        self.assertTrue(other_channel.accepted)

        writer.start()
        self.assertTrue(writer.drain(timeout=2))
        self.assertEqual(len(store.records), 3)
        self.assertEqual(
            {(row["channel_id"], row["payload"]["cadence_slot"]) for row in store.records},
            {(7, 1), (7, 2), (8, 1)},
        )
        self.assertEqual(
            {row["source"] for row in store.records},
            {"semantic_snapshot"},
        )
        self.assertTrue(
            all(
                row["shard_key"].startswith(
                    f"semantic:ch{row['channel_id']}:"
                )
                for row in store.records
            )
        )
        self.assertTrue(
            all(row["payload"]["independent_of_alert_or_probe_hit"] for row in store.records)
        )
        status = writer.status()
        self.assertEqual(status["counters"]["accepted_total"], 3)
        self.assertEqual(status["counters"]["duplicate_total"], 1)
        self.assertEqual(status["counters"]["persisted_total"], 3)
        self.assertEqual(status["counters"].get("gap_total", 0), 0)

    def test_probe_frame_adapter_reuses_embedding_and_provenance(self):
        store = RecordingDetectionsStore()
        writer = self._writer(store)
        result = writer.submit_probe_frame(
            {
                "channel_id": 12,
                "timestamp_ms": 10_500,
                "embedding": _embedding(),
                "thumbnail": "jpeg-base64",
                "embedding_ref": "probe-buffer:12:42",
                "frame_uid": 42,
            },
            provenance={"selection_source": "capture_apex"},
        )

        self.assertTrue(result.accepted)
        writer.start()
        self.assertTrue(writer.drain(timeout=2))
        row = store.records[0]
        self.assertEqual(row["clip_vec"], tuple(_embedding()))
        self.assertEqual(row["thumbnail_b64"], "jpeg-base64")
        self.assertEqual(
            row["payload"]["provenance"],
            {
                "selection_source": "capture_apex",
                "embedding_ref": "probe-buffer:12:42",
                "frame_uid": 42,
            },
        )

    def test_batches_snapshots(self):
        store = RecordingDetectionsStore()
        writer = self._writer(store, batch_size=3)
        for second in range(1, 6):
            self.assertTrue(
                writer.submit(
                    channel_id=7,
                    timestamp_ms=second * 1_000,
                    embedding=_embedding(),
                    thumbnail=f"thumb-{second}",
                ).accepted
            )

        writer.start()
        self.assertTrue(writer.drain(timeout=2))
        self.assertEqual([len(batch) for batch in store.calls], [3, 2])
        self.assertEqual(writer.status()["counters"]["batch_persisted_total"], 2)

    def test_retries_transient_store_failure_without_gap(self):
        store = FlakyDetectionsStore(failures=1)
        writer = self._writer(store, max_attempts=3)
        for channel in (7, 8):
            writer.submit(
                channel_id=channel,
                timestamp_ms=1_000,
                embedding=_embedding(),
                thumbnail=f"thumb-{channel}",
            )

        writer.start()
        self.assertTrue(writer.drain(timeout=2))
        status = writer.status()
        self.assertEqual(store.attempts, 2)
        self.assertEqual(status["counters"]["retry_total"], 1)
        self.assertEqual(status["counters"]["persisted_total"], 2)
        self.assertEqual(status["counters"].get("gap_total", 0), 0)
        self.assertIsNone(status["last_error"])

    def test_idempotent_store_conflict_is_durable_not_a_gap(self):
        store = IdempotentDetectionsStore()
        writer = self._writer(store)
        writer.submit(
            channel_id=7,
            timestamp_ms=1_000,
            embedding=_embedding(),
            thumbnail="already-durable",
        )

        writer.start()
        self.assertTrue(writer.drain(timeout=2))
        status = writer.status()
        self.assertEqual(status["counters"]["persisted_total"], 1)
        self.assertEqual(
            status["counters"]["idempotent_write_attempt_total"],
            1,
        )
        self.assertEqual(status["counters"]["gap_total"], 0)

    def test_status_exposes_per_channel_cadence_gaps_and_staleness(self):
        now = [10.0]
        writer = self._writer(
            RecordingDetectionsStore(),
            monotonic=lambda: now[0],
        )
        writer.submit(
            channel_id=7,
            timestamp_ms=1_000,
            embedding=_embedding(),
            thumbnail="first",
        )
        now[0] = 13.0
        writer.submit(
            channel_id=7,
            timestamp_ms=4_000,
            embedding=_embedding(),
            thumbnail="fourth",
        )

        now[0] = 14.5
        status = writer.status()
        channel = status["channel_cadence"]["7"]
        self.assertEqual(channel["wall_gap_slots"], 2)
        self.assertEqual(channel["source_gap_slots"], 2)
        self.assertEqual(channel["staleness_seconds"], 1.5)
        self.assertEqual(
            status["counters"]["wall_cadence_gap_total"],
            2,
        )
        self.assertEqual(
            status["counters"]["source_cadence_gap_total"],
            2,
        )

    def test_permanent_failure_is_an_explicit_gap(self):
        store = FlakyDetectionsStore(failures=10)
        writer = self._writer(store, max_attempts=2)
        writer.submit(
            channel_id=7,
            timestamp_ms=1_000,
            embedding=_embedding(),
            thumbnail="thumb",
        )

        writer.start()
        self.assertTrue(writer.drain(timeout=2))
        status = writer.status()
        self.assertEqual(status["counters"]["retry_total"], 1)
        self.assertEqual(status["counters"]["failure_total"], 1)
        self.assertEqual(status["counters"]["gap_total"], 1)
        self.assertEqual(status["gap_reasons"], {"archive_write_failed": 1})
        self.assertIn("database unavailable", status["last_error"])

    def test_bounded_queue_reports_backpressure_drop(self):
        store = RecordingDetectionsStore()
        writer = self._writer(
            store,
            max_queue=2,
            batch_size=2,
            dedupe_capacity=2,
        )
        first = writer.submit(
            channel_id=7,
            timestamp_ms=1_000,
            embedding=_embedding(),
            thumbnail="one",
        )
        second = writer.submit(
            channel_id=7,
            timestamp_ms=2_000,
            embedding=_embedding(),
            thumbnail="two",
        )
        dropped = writer.submit(
            channel_id=7,
            timestamp_ms=3_000,
            embedding=_embedding(),
            thumbnail="three",
        )

        self.assertTrue(first.accepted)
        self.assertTrue(second.accepted)
        self.assertEqual(dropped.status, SnapshotSubmitStatus.DROPPED)
        self.assertEqual(dropped.reason, "backpressure_queue_full")
        status = writer.status()
        self.assertEqual(status["counters"]["dropped_total"], 1)
        self.assertEqual(status["counters"]["rejected_total"], 1)
        self.assertEqual(status["counters"]["gap_total"], 1)
        self.assertEqual(
            status["gap_reasons"],
            {"backpressure_queue_full": 1},
        )

        writer.start()
        self.assertTrue(writer.drain(timeout=2))
        self.assertEqual(len(store.records), 2)

    def test_invalid_embedding_is_rejected_without_clip_recomputation(self):
        store = RecordingDetectionsStore()
        writer = self._writer(store)
        rejected = writer.submit(
            channel_id=7,
            timestamp_ms=1_000,
            embedding=[3.0, 4.0],
            thumbnail="thumb",
        )

        self.assertEqual(rejected.status, SnapshotSubmitStatus.REJECTED)
        self.assertIn("L2-normalized", rejected.reason)
        status = writer.status()
        self.assertEqual(status["counters"]["rejected_total"], 1)
        self.assertEqual(status["counters"]["gap_total"], 1)
        self.assertEqual(store.records, [])

    def test_stop_drains_and_rejects_late_submission(self):
        store = RecordingDetectionsStore()
        writer = self._writer(store)
        writer.submit(
            channel_id=7,
            timestamp_ms=1_000,
            embedding=_embedding(),
            thumbnail="thumb",
        )

        self.assertTrue(writer.stop(drain=True, timeout=2))
        self.assertEqual(len(store.records), 1)
        late = writer.submit(
            channel_id=7,
            timestamp_ms=2_000,
            embedding=_embedding(),
            thumbnail="late",
        )
        self.assertEqual(late.status, SnapshotSubmitStatus.DROPPED)
        self.assertEqual(late.reason, "writer_stopped")
        self.assertEqual(writer.status()["counters"]["gap_total"], 1)


if __name__ == "__main__":
    unittest.main()
