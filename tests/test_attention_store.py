import json
import threading
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from attention_store import (
    AttentionBatch,
    AttentionEpisodeRecord,
    AttentionWriteResult,
    BufferedAttentionWriter,
    EmbeddingSnapshotRef,
    IntervalEvidenceLink,
    MemoryAttentionStore,
    MotionInterval,
    PostgresAttentionStore,
    ProbeLineageRecord,
    ProbeScoreRecord,
    SchedulerDecisionRecord,
    canonical_json,
)


ROOT = Path(__file__).resolve().parent.parent
MIGRATION = (
    ROOT / "migrations" / "versions" / "20260726_0008_attention_storage.py"
)
TENANT_ID = "f3c3533e-bf17-46a1-a543-696d95b8cf6f"


def uid() -> str:
    return str(uuid.uuid4())


def snapshot(**changes: Any) -> EmbeddingSnapshotRef:
    values = {
        "id": uid(),
        "channel_id": 112,
        "captured_at_ms": 1000,
        "embedding_ref": "archive.detections:91:clip",
        "embedding_model": "openclip-vit-b-32",
        "frame_ref": "luxriot:112:1000",
        "cadence_ms": 1000,
    }
    values.update(changes)
    return EmbeddingSnapshotRef(**values)


def interval(**changes: Any) -> MotionInterval:
    values = {
        "id": uid(),
        "channel_id": 112,
        "started_at_ms": 900,
        "ended_at_ms": 1900,
        "state": "motion",
        "sample_count": 6,
        "expected_sample_count": 6,
        "motion_mean": 0.22,
        "motion_max": 0.71,
        "motion_p95": 0.65,
        "motion_integral": 0.24,
        "moving_fraction": 0.66,
        "quiet_fraction": 0.34,
        "activity_x_max": 3.1,
        "peak_at_ms": 1300,
        "baseline_ref": "baseline:112:r7",
    }
    values.update(changes)
    return MotionInterval(**values)


def probe_score(
    embedding_snapshot_id: str,
    **changes: Any,
) -> ProbeScoreRecord:
    values = {
        "id": uid(),
        "embedding_snapshot_id": embedding_snapshot_id,
        "scored_at_ms": 1010,
        "probe_id": "person-near-gate",
        "probe_version": "sha256:abc",
        "pos_score": 0.61,
        "neg_score": 0.22,
        "margin": 0.39,
        "pos_floor": 0.40,
        "margin_threshold": 0.10,
        "threshold_state": "hit",
    }
    values.update(changes)
    return ProbeScoreRecord(**values)


class AttentionRecordTests(unittest.TestCase):
    def test_interval_contains_only_numeric_aggregate_and_validates_bounds(self):
        item = interval()
        self.assertEqual(item.sample_count, 6)
        self.assertEqual(item.state, "motion")
        with self.assertRaisesRegex(ValueError, "peak_at_ms"):
            interval(peak_at_ms=2000)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            interval(moving_fraction=0.8, quiet_fraction=0.4)
        with self.assertRaisesRegex(ValueError, "finite"):
            interval(motion_mean=float("nan"))

    def test_links_require_exactly_one_typed_reference(self):
        snap = snapshot()
        cv = interval()
        link = IntervalEvidenceLink(
            id=uid(),
            interval_id=cv.id,
            occurred_at_ms=1000,
            kind="embedding",
            role="support",
            embedding_snapshot_id=snap.id,
        )
        self.assertEqual(link.embedding_snapshot_id, snap.id)
        with self.assertRaisesRegex(ValueError, "require"):
            IntervalEvidenceLink(
                id=uid(),
                interval_id=cv.id,
                occurred_at_ms=1300,
                kind="vlm_apex",
                role="apex",
            )

    def test_probe_score_persists_pnm_thresholds_and_version(self):
        snap = snapshot()
        score = probe_score(snap.id)
        self.assertEqual(score.embedding_snapshot_id, snap.id)
        self.assertEqual(score.threshold_state, "hit")
        self.assertAlmostEqual(score.margin, 0.39)
        with self.assertRaisesRegex(ValueError, "requires"):
            probe_score(
                snap.id,
                threshold_state="below_pos",
                pos_floor=None,
                margin_threshold=None,
            )
        with self.assertRaisesRegex(ValueError, "threshold_state"):
            probe_score(snap.id, threshold_state="maybe")

    def test_printable_records_are_canonical_and_reject_image_payloads(self):
        record = {"weights": {"motion": 0.5}, "action": "enqueue", "frame_ref": "x:1"}
        decision = SchedulerDecisionRecord(
            id=uid(),
            decided_at_ms=1200,
            channel_id=112,
            action="enqueue_vlm",
            record=record,
        )
        self.assertEqual(json.loads(decision.printable_json), record)
        self.assertEqual(decision.printable_json, canonical_json(record))
        with self.assertRaisesRegex(ValueError, "reference"):
            SchedulerDecisionRecord(
                id=uid(),
                decided_at_ms=1200,
                channel_id=112,
                action="enqueue_vlm",
                record={"image_b64": "not-allowed"},
            )
        with self.assertRaisesRegex(ValueError, "JSON-compatible"):
            AttentionEpisodeRecord(
                id=uid(),
                channel_id=112,
                started_at_ms=1000,
                ended_at_ms=2000,
                trigger="motion",
                status="open",
                record={"raw_frame": b"bytes"},
            )

    def test_batch_keeps_sparse_refs_and_audit_records_together(self):
        snap = snapshot()
        cv = interval()
        batch = AttentionBatch(
            snapshots=(snap,),
            probe_scores=(probe_score(snap.id),),
            intervals=(cv,),
            links=(
                IntervalEvidenceLink(
                    id=uid(),
                    interval_id=cv.id,
                    occurred_at_ms=snap.captured_at_ms,
                    kind="embedding",
                    role="support",
                    embedding_snapshot_id=snap.id,
                ),
                IntervalEvidenceLink(
                    id=uid(),
                    interval_id=cv.id,
                    occurred_at_ms=1300,
                    kind="vlm_apex",
                    role="apex",
                    apex_ref="vlm-job:abc:frame:3",
                ),
            ),
            probe_lineage=(
                ProbeLineageRecord(
                    id=uid(),
                    probe_id="child-alert-probe",
                    channel_id=112,
                    created_at_ms=1400,
                    expires_at_ms=61_400,
                    lifecycle_state="created",
                    parent_alert_ref="detection:88",
                    record={"ttl_ms": 60_000, "proposal_rank": 2},
                ),
            ),
        )
        self.assertEqual(batch.record_count, 6)


class RecordingStore:
    def __init__(self, outcomes=None):
        self.batches = []
        self.outcomes = list(outcomes or [])
        self.lock = threading.Lock()

    def write_batch(self, batch):
        with self.lock:
            self.batches.append(batch)
            if self.outcomes:
                outcome = self.outcomes.pop(0)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome
        return AttentionWriteResult(
            ok=True,
            accepted_records=batch.record_count,
            inserted_records=batch.record_count,
        )


class BufferedWriterTests(unittest.TestCase):
    def test_submit_is_bounded_and_drops_oldest_telemetry(self):
        store = RecordingStore()
        writer = BufferedAttentionWriter(
            store,
            max_batches=2,
            max_records=2,
            autostart=False,
        )
        first = AttentionBatch(snapshots=(snapshot(captured_at_ms=1000),))
        second = AttentionBatch(snapshots=(snapshot(captured_at_ms=2000),))
        third = AttentionBatch(snapshots=(snapshot(captured_at_ms=3000),))

        writer.submit(first)
        writer.submit(second)
        result = writer.submit(third)

        self.assertTrue(result.accepted)
        self.assertEqual(result.queued_records, 2)
        self.assertEqual(result.dropped_batches, 1)
        writer.flush_once()
        written_times = [
            item.captured_at_ms
            for batch in store.batches
            for item in batch.snapshots
        ]
        self.assertEqual(written_times, [2000, 3000])

    def test_database_failure_is_contained_and_retried(self):
        failed = AttentionWriteResult(False, 1, 0, "unavailable")
        store = RecordingStore(outcomes=[failed])
        writer = BufferedAttentionWriter(store, autostart=False)
        writer.submit(AttentionBatch(snapshots=(snapshot(),)))

        result = writer.flush_once()
        self.assertFalse(result.ok)
        self.assertEqual(writer.stats()["queued_records"], 1)
        self.assertEqual(writer.stats()["write_failures"], 1)

        result = writer.flush_once()
        self.assertTrue(result.ok)
        self.assertEqual(writer.stats()["queued_records"], 0)
        self.assertEqual(writer.stats()["written_records"], 1)

    def test_memory_adapter_is_bounded_and_drainable_for_db_disabled_mode(self):
        store = MemoryAttentionStore(max_batches=2, max_records=2)
        store.write_batch(
            AttentionBatch(snapshots=(snapshot(captured_at_ms=1000),))
        )
        store.write_batch(
            AttentionBatch(snapshots=(snapshot(captured_at_ms=2000),))
        )
        store.write_batch(
            AttentionBatch(snapshots=(snapshot(captured_at_ms=3000),))
        )

        self.assertEqual(store.stats()["queued_records"], 2)
        self.assertEqual(store.stats()["dropped_records"], 1)
        retained = store.drain_batches()
        retained_times = [
            item.captured_at_ms
            for batch in retained
            for item in batch.snapshots
        ]
        self.assertEqual(retained_times, [2000, 3000])
        self.assertEqual(store.stats()["queued_records"], 0)

    def test_background_writer_drains_without_capture_thread_db_io(self):
        store = RecordingStore()
        writer = BufferedAttentionWriter(
            store,
            retry_initial_seconds=0.01,
            retry_max_seconds=0.02,
        )
        result = writer.submit(AttentionBatch(snapshots=(snapshot(),)))
        self.assertTrue(result.accepted)
        self.assertTrue(writer.drain(1.0))
        self.assertTrue(writer.close())
        self.assertEqual(writer.stats()["written_records"], 1)


class FakeResult:
    def __init__(self, *, rows=None, rowcount=1):
        self.rows = list(rows or [])
        self.rowcount = rowcount

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, rows=None):
        self.calls = []
        self.rows = list(rows or [])

    def execute(self, sql, params=None):
        self.calls.append((" ".join(sql.split()), params))
        rows, self.rows = self.rows, []
        return FakeResult(rows=rows)


class FakePool:
    def __init__(self, connection):
        self.connection = connection
        self.contexts = []

    @contextmanager
    def transaction(self, context, readonly=False):
        self.contexts.append((context, readonly))
        yield self.connection


class PostgresStoreTests(unittest.TestCase):
    def test_child_writes_are_guarded_when_parent_telemetry_was_dropped(self):
        connection = FakeConnection()
        store = PostgresAttentionStore(FakePool(connection), TENANT_ID)
        snap = snapshot()
        cv = interval()
        missing_episode_id = uid()
        result = store.write_batch(
            AttentionBatch(
                probe_scores=(probe_score(snap.id),),
                links=(
                    IntervalEvidenceLink(
                        id=uid(),
                        interval_id=cv.id,
                        occurred_at_ms=1000,
                        kind="embedding",
                        role="support",
                        embedding_snapshot_id=snap.id,
                    ),
                ),
                decisions=(
                    SchedulerDecisionRecord(
                        id=uid(),
                        channel_id=112,
                        episode_id=missing_episode_id,
                        decided_at_ms=2000,
                        action="fast_vlm_no_alert",
                        record={"frame_count": 6},
                    ),
                ),
            )
        )

        self.assertTrue(result.ok)
        sql = "\n".join(statement for statement, _params in connection.calls)
        self.assertIn("FROM archive.attention_embedding_snapshots", sql)
        self.assertIn("FROM archive.attention_intervals", sql)
        self.assertIn("FROM archive.attention_episodes", sql)
        self.assertIn("ELSE NULL", sql)

    def test_query_is_tenant_channel_time_scoped_and_limit_is_bounded(self):
        connection = FakeConnection()
        store = PostgresAttentionStore(FakePool(connection), TENANT_ID)

        self.assertEqual(
            store.query_snapshots(
                channel_id=112,
                start_ms=1000,
                end_ms=5000,
                limit=9000,
            ),
            [],
        )

        sql, params = connection.calls[-1]
        self.assertIn("tenant_id = %s", sql)
        self.assertIn("channel_id = %s", sql)
        self.assertIn("captured_at_ms BETWEEN %s AND %s", sql)
        self.assertEqual(params, (TENANT_ID, 112, 1000, 5000, 1000))

    def test_evidence_query_resolves_embedding_refs_through_intervals(self):
        connection = FakeConnection()
        store = PostgresAttentionStore(FakePool(connection), TENANT_ID)

        self.assertEqual(
            store.query_evidence_links(
                channel_id=112,
                start_ms=1000,
                end_ms=5000,
            ),
            [],
        )

        sql, params = connection.calls[-1]
        self.assertIn("JOIN archive.attention_intervals", sql)
        self.assertIn("LEFT JOIN archive.attention_embedding_snapshots", sql)
        self.assertIn("interval.channel_id = %s", sql)
        self.assertEqual(params, (TENANT_ID, 112, 1000, 5000, 500))

    def test_probe_score_query_is_snapshot_channel_and_time_scoped(self):
        connection = FakeConnection()
        store = PostgresAttentionStore(FakePool(connection), TENANT_ID)

        self.assertEqual(
            store.query_probe_scores(
                channel_id=112,
                start_ms=1000,
                end_ms=5000,
                limit=5000,
            ),
            [],
        )

        sql, params = connection.calls[-1]
        self.assertIn("JOIN archive.attention_embedding_snapshots", sql)
        self.assertIn("snapshot.channel_id = %s", sql)
        self.assertIn("snapshot.captured_at_ms BETWEEN %s AND %s", sql)
        self.assertEqual(params, (TENANT_ID, 112, 1000, 5000, 1000))

    def test_retention_is_cutoff_and_batch_bounded_for_every_table(self):
        connection = FakeConnection()
        store = PostgresAttentionStore(FakePool(connection), TENANT_ID)

        result = store.apply_retention(before_ms=50_000, batch_size=99_999)

        self.assertEqual(len(result), 5)
        self.assertEqual(len(connection.calls), 5)
        for sql, params in connection.calls:
            self.assertIn("tenant_id = %s", sql)
            self.assertIn("LIMIT %s", sql)
            self.assertEqual(params, (TENANT_ID, TENANT_ID, 50_000, 50_000))

    def test_postgres_failure_returns_safe_result(self):
        class MissingConnection:
            def execute(self, sql, params=None):
                raise RuntimeError(
                    'relation "archive.attention_intervals" does not exist'
                )

        store = PostgresAttentionStore(FakePool(MissingConnection()), TENANT_ID)
        result = store.write_batch(
            AttentionBatch(intervals=(interval(),))
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "not_migrated")


class AttentionMigrationTests(unittest.TestCase):
    def test_migration_has_compact_tables_rls_links_and_bounded_indexes(self):
        source = MIGRATION.read_text(encoding="utf-8")
        self.assertIn('revision: str = "20260726_0008"', source)
        self.assertIn(
            'down_revision: str | None = "20260725_0007"', source
        )
        for table in (
            "attention_embedding_snapshots",
            "attention_probe_scores",
            "attention_intervals",
            "attention_interval_links",
            "attention_episodes",
            "attention_scheduler_decisions",
            "attention_probe_lineage",
        ):
            self.assertIn(f"CREATE TABLE archive.{table}", source)
        self.assertIn("FOREIGN KEY (tenant_id, interval_id)", source)
        self.assertIn("FOREIGN KEY (tenant_id, embedding_snapshot_id)", source)
        self.assertIn("pos_score double precision", source)
        self.assertIn("neg_score double precision", source)
        self.assertIn("margin_threshold double precision", source)
        self.assertIn("ENABLE ROW LEVEL SECURITY", source)
        self.assertIn("FORCE ROW LEVEL SECURITY", source)
        self.assertIn("current_setting('eva.tenant_id', true)", source)
        self.assertNotIn("thumbnail_b64", source)
        self.assertNotIn("image_bytes", source)


if __name__ == "__main__":
    unittest.main()
