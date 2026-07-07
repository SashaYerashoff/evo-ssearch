import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from eva_db import DatabaseSettings, PsycopgPool, TransactionContext
from inference_queue import (
    EnqueueStatus,
    InferenceEnqueueService,
    InferenceJob,
    InferenceQueueRepository,
    JobResult,
    JobState,
    ManualClock,
    QueueFullError,
    RetryPolicy,
    WorkloadClass,
)
from inference_queue.postgres import PostgresInferenceQueueRepository


UTC = timezone.utc


class NoDatabasePool:
    def __init__(self):
        self.transaction_calls = 0

    @contextmanager
    def transaction(self, *_args, **_kwargs):
        self.transaction_calls += 1
        raise AssertionError("validation should fail before opening a transaction")
        yield

    def __repr__(self):
        return "NoDatabasePool()"


class PostgresRepositoryValidationTests(unittest.TestCase):
    def setUp(self):
        self.tenant_id = str(uuid4())
        self.pool = NoDatabasePool()
        self.clock = ManualClock(datetime(2026, 6, 9, 12, 0, tzinfo=UTC))
        self.repository = PostgresInferenceQueueRepository(
            self.pool,
            self.tenant_id,
            clock=self.clock,
        )

    def job(self, **overrides):
        values = {
            "id": str(uuid4()),
            "tenant_id": self.tenant_id,
            "channel_id": "7",
            "model": "qwen35-9b-q4_k_m",
            "prompt": "Describe the current frame.",
            "workload_class": WorkloadClass.EVENT,
            "priority": 200,
            "created_at": self.clock.now(),
            "available_at": self.clock.now(),
            "payload": {"frame_id": "frame-1"},
        }
        values.update(overrides)
        return InferenceJob(**values)

    def test_is_repository_protocol_and_repr_has_no_payload(self):
        self.assertIsInstance(self.repository, InferenceQueueRepository)
        representation = repr(self.repository)
        self.assertIn(self.tenant_id, representation)
        self.assertNotIn("Describe the current frame", representation)

    def test_constructor_requires_uuid_tenant_and_actor(self):
        with self.assertRaisesRegex(ValueError, "tenant_id"):
            PostgresInferenceQueueRepository(self.pool, "tenant-a")
        with self.assertRaisesRegex(ValueError, "actor_id"):
            PostgresInferenceQueueRepository(
                self.pool,
                self.tenant_id,
                actor_id="worker-service",
            )

    def test_enqueue_requires_uuid_ids_and_matching_tenant(self):
        with self.assertRaisesRegex(ValueError, "job.id"):
            self.repository.enqueue(self.job(id="job-1"), capacity=10)
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.repository.enqueue(
                self.job(tenant_id=str(uuid4())),
                capacity=10,
            )
        self.assertEqual(self.pool.transaction_calls, 0)

    def test_enqueue_requires_positive_canonical_integer_channel(self):
        for channel_id in ("camera-7", "0", "-1", "07"):
            with self.subTest(channel_id=channel_id):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    self.repository.enqueue(
                        self.job(channel_id=channel_id),
                        capacity=10,
                    )
        self.assertEqual(self.pool.transaction_calls, 0)

    def test_enqueue_rejects_schema_incompatible_values(self):
        with self.assertRaisesRegex(ValueError, "capacity"):
            self.repository.enqueue(self.job(), capacity=0)
        with self.assertRaisesRegex(ValueError, "priority"):
            self.repository.enqueue(self.job(priority=1001), capacity=10)
        with self.assertRaisesRegex(ValueError, "reserved"):
            self.repository.enqueue(
                self.job(payload={"_eva_queue": {"prompt": "forged"}}),
                capacity=10,
            )
        with self.assertRaisesRegex(ValueError, "JSON serializable"):
            self.repository.enqueue(
                self.job(payload={"frame": object()}),
                capacity=10,
            )
        terminal = self.job(
            state=JobState.DROPPED,
            finished_at=self.clock.now(),
        )
        with self.assertRaisesRegex(Exception, "only queued"):
            self.repository.enqueue(terminal, capacity=10)
        self.assertEqual(self.pool.transaction_calls, 0)

    def test_claim_validation_and_empty_filter_do_not_touch_database(self):
        with self.assertRaisesRegex(ValueError, "worker_id"):
            self.repository.claim("", timedelta(seconds=10))
        with self.assertRaisesRegex(ValueError, "lease_duration"):
            self.repository.claim("worker", timedelta(0))
        self.assertIsNone(
            self.repository.claim(
                "worker",
                timedelta(seconds=10),
                workload_classes=[],
            )
        )
        self.assertEqual(self.pool.transaction_calls, 0)

    def test_result_must_be_json_serializable_before_database_access(self):
        job_id = str(uuid4())
        result = JobResult(
            job_id=job_id,
            output={"timestamp": self.clock.now()},
            completed_at=self.clock.now(),
            worker_id="worker",
        )
        with self.assertRaisesRegex(ValueError, "result must be JSON serializable"):
            self.repository.complete(job_id, "worker", result)
        self.assertEqual(self.pool.transaction_calls, 0)

    def test_public_lookup_requires_uuid_job_id(self):
        with self.assertRaisesRegex(ValueError, "job_id"):
            self.repository.get_job("job-1")
        with self.assertRaisesRegex(ValueError, "job_id"):
            self.repository.get_result("job-1")
        self.assertEqual(self.pool.transaction_calls, 0)


@unittest.skipUnless(
    os.getenv("EVA_TEST_DATABASE_DSN"),
    "set EVA_TEST_DATABASE_DSN for live PostgreSQL queue tests",
)
class PostgresRepositoryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = PsycopgPool(
            DatabaseSettings(
                dsn=os.environ["EVA_TEST_DATABASE_DSN"],
                pool_min_size=0,
                pool_max_size=8,
            )
        )
        cls.pool.open()

    @classmethod
    def tearDownClass(cls):
        cls.pool.close()

    def setUp(self):
        self.tenant_id = str(uuid4())
        self.actor_id = str(uuid4())
        self.clock = ManualClock(datetime(2026, 6, 9, 12, 0, tzinfo=UTC))
        self.repository = PostgresInferenceQueueRepository(
            self.pool,
            self.tenant_id,
            actor_id=self.actor_id,
            clock=self.clock,
        )
        self.retry_policy = RetryPolicy(
            max_attempts=3,
            initial_backoff=timedelta(seconds=5),
            multiplier=2,
            max_backoff=timedelta(seconds=30),
        )
        self.service = self.service_with_capacity(20)

    def tearDown(self):
        context = TransactionContext(
            tenant_id=self.tenant_id,
            actor_id=self.actor_id,
        )
        with self.pool.transaction(context) as connection:
            connection.execute(
                "DELETE FROM jobs.inference_jobs WHERE tenant_id = %s",
                (self.tenant_id,),
            )

    def service_with_capacity(self, capacity, ids=None):
        factory = (
            (lambda: str(uuid4()))
            if ids is None
            else iter(ids).__next__
        )
        return InferenceEnqueueService(
            self.repository,
            capacity=capacity,
            clock=self.clock,
            retry_policy=self.retry_policy,
            id_factory=factory,
        )

    def enqueue(self, workload=WorkloadClass.EVENT, **overrides):
        values = {
            "tenant_id": self.tenant_id,
            "channel_id": "1",
            "model": "qwen35-9b-q4_k_m",
            "prompt": "Describe the current frame.",
            "payload": {"frame": "frame-1"},
        }
        values.update(overrides)
        return self.service.enqueue(workload_class=workload, **values)

    def test_idempotency_and_heartbeat_coalescing_keep_newest_payload(self):
        first = self.enqueue(
            WorkloadClass.HEARTBEAT,
            payload={"frame": "old"},
        )
        self.clock.advance(1)
        latest_deadline = self.clock.now() + timedelta(seconds=30)
        latest = self.enqueue(
            WorkloadClass.HEARTBEAT,
            payload={"frame": "new", "batch": [2, 3]},
            priority=150,
            deadline=latest_deadline,
            idempotency_key="capture-2",
        )
        repeated = self.enqueue(
            WorkloadClass.HEARTBEAT,
            payload={"frame": "must-not-win"},
            idempotency_key="capture-2",
        )

        self.assertEqual(first.status, EnqueueStatus.ENQUEUED)
        self.assertEqual(latest.status, EnqueueStatus.COALESCED)
        self.assertEqual(latest.job.id, first.job.id)
        self.assertEqual(latest.job.payload["frame"], "new")
        self.assertEqual(latest.job.payload["batch"], (2, 3))
        self.assertEqual(latest.job.priority, 150)
        self.assertEqual(latest.job.deadline, latest_deadline)
        self.assertEqual(repeated.status, EnqueueStatus.IDEMPOTENT)
        self.assertEqual(repeated.job.payload["frame"], "new")
        self.assertNotIn("_eva_queue", repeated.job.payload)

        with self.pool.transaction(self.repository._context) as connection:
            stored = connection.execute(
                """
                SELECT payload
                FROM jobs.inference_jobs
                WHERE tenant_id = %s AND id = %s
                """,
                (self.tenant_id, first.job.id),
            ).fetchone()[0]
        self.assertEqual(
            stored["_eva_queue"]["prompt"],
            "Describe the current frame.",
        )
        self.assertEqual(
            self.repository.metrics().coalesced_count,
            1,
        )

    def test_leased_heartbeat_has_one_refreshable_successor(self):
        first = self.enqueue(
            WorkloadClass.HEARTBEAT,
            payload={"frame": "leased"},
        ).job
        leased = self.repository.claim("worker-a", timedelta(seconds=10))
        self.assertEqual(leased.id, first.id)

        successor = self.enqueue(
            WorkloadClass.HEARTBEAT,
            payload={"frame": "successor-1"},
        )
        newest = self.enqueue(
            WorkloadClass.HEARTBEAT,
            payload={"frame": "successor-2"},
        )

        self.assertEqual(successor.status, EnqueueStatus.ENQUEUED)
        self.assertEqual(newest.status, EnqueueStatus.COALESCED)
        self.assertEqual(newest.job.id, successor.job.id)
        self.assertEqual(newest.job.payload["frame"], "successor-2")
        self.assertEqual(self.repository.metrics().queue_depth, 1)

        completed_at = self.clock.now()
        self.repository.complete(
            first.id,
            "worker-a",
            JobResult(
                job_id=first.id,
                output={"summary": "old frame"},
                completed_at=completed_at,
                worker_id="worker-a",
            ),
        )
        claimed = self.repository.claim("worker-a", timedelta(seconds=10))
        self.assertEqual(claimed.id, successor.job.id)
        self.assertEqual(claimed.payload["frame"], "successor-2")

    def test_claim_orders_priority_deadline_fifo_and_uses_skip_locked(self):
        late = self.clock.now() + timedelta(minutes=5)
        early = self.clock.now() + timedelta(minutes=1)
        event_late = self.enqueue(
            channel_id="10",
            priority=250,
            deadline=late,
        ).job
        event_early = self.enqueue(
            channel_id="11",
            priority=250,
            deadline=early,
        ).job
        manual = self.enqueue(
            WorkloadClass.MANUAL,
            channel_id="12",
        ).job

        first = self.repository.claim("worker-1", timedelta(seconds=30))
        self.assertEqual(first.id, manual.id)
        self.repository.complete(
            first.id,
            "worker-1",
            JobResult(
                job_id=first.id,
                output="done",
                completed_at=self.clock.now(),
                worker_id="worker-1",
            ),
        )
        second = self.repository.claim("worker-1", timedelta(seconds=30))
        self.assertEqual(second.id, event_early.id)
        self.repository.complete(
            second.id,
            "worker-1",
            JobResult(
                job_id=second.id,
                output="done",
                completed_at=self.clock.now(),
                worker_id="worker-1",
            ),
        )
        third = self.repository.claim("worker-1", timedelta(seconds=30))
        self.assertEqual(third.id, event_late.id)
        self.repository.complete(
            third.id,
            "worker-1",
            JobResult(
                job_id=third.id,
                output="done",
                completed_at=self.clock.now(),
                worker_id="worker-1",
            ),
        )

        fifo_first = self.enqueue(channel_id="13", priority=200).job
        self.clock.advance(1)
        fifo_second = self.enqueue(channel_id="14", priority=200).job
        claimed_fifo = self.repository.claim(
            "worker-1", timedelta(seconds=30)
        )
        self.assertEqual(claimed_fifo.id, fifo_first.id)

        parallel_one = self.enqueue(channel_id="15", priority=250).job
        parallel_two = self.enqueue(channel_id="16", priority=250).job
        with ThreadPoolExecutor(max_workers=2) as executor:
            claimed = list(
                executor.map(
                    lambda worker: self.repository.claim(
                        worker, timedelta(seconds=30)
                    ),
                    ("worker-2", "worker-3"),
                )
            )
        self.assertEqual(
            {job.id for job in claimed},
            {parallel_one.id, parallel_two.id},
        )
        self.assertNotEqual(fifo_first.id, fifo_second.id)

    def test_lease_complete_retry_stale_recovery_and_attempt_history(self):
        completed = self.enqueue(channel_id="20").job
        claimed = self.repository.claim("worker-a", timedelta(seconds=10))
        renewed = self.repository.renew_lease(
            claimed.id,
            "worker-a",
            timedelta(seconds=20),
        )
        self.assertGreater(renewed.lease_expires_at, claimed.lease_expires_at)
        result = JobResult(
            job_id=completed.id,
            output={"labels": ["person"]},
            completed_at=self.clock.now(),
            worker_id="worker-a",
            metadata={"model": "qwen"},
        )
        self.repository.complete(completed.id, "worker-a", result)
        self.assertEqual(self.repository.get_result(completed.id), result)

        retrying = self.enqueue(channel_id="21").job
        self.repository.claim("worker-b", timedelta(seconds=10))
        queued = self.repository.fail(
            retrying.id,
            "worker-b",
            "temporary",
            self.retry_policy,
        )
        self.assertEqual(queued.state, JobState.QUEUED)
        self.assertEqual(
            queued.available_at,
            self.clock.now() + timedelta(seconds=5),
        )
        self.clock.advance(5)
        leased_again = self.repository.claim(
            "worker-b", timedelta(seconds=10)
        )
        self.assertEqual(leased_again.id, retrying.id)

        self.clock.advance(10)
        recovered_count = self.repository.recover_stale_leases(
            self.retry_policy
        )
        self.assertEqual(recovered_count, 1)
        recovered = self.repository.get_job(retrying.id)
        self.assertEqual(recovered.state, JobState.QUEUED)
        self.assertEqual(recovered.last_error, "lease expired")

        with self.pool.transaction(self.repository._context) as connection:
            states = connection.execute(
                """
                SELECT attempt_number, state
                FROM jobs.job_attempts
                WHERE tenant_id = %s AND job_id = %s
                ORDER BY attempt_number
                """,
                (self.tenant_id, retrying.id),
            ).fetchall()
        self.assertEqual(states, [(1, "failed"), (2, "abandoned")])
        self.assertEqual(self.repository.metrics().retried_count, 2)

    def test_capacity_dead_letter_and_dropped_metrics(self):
        ids = [str(uuid4()) for _ in range(5)]
        service = self.service_with_capacity(2, ids)
        heartbeat = service.enqueue_heartbeat(
            tenant_id=self.tenant_id,
            channel_id="30",
            model="vlm",
            prompt="heartbeat",
        )
        service.enqueue_event(
            tenant_id=self.tenant_id,
            channel_id="31",
            model="vlm",
            prompt="event",
        )
        manual = service.enqueue_manual(
            tenant_id=self.tenant_id,
            channel_id="32",
            model="vlm",
            prompt="manual",
        )
        dropped = service.enqueue_heartbeat(
            tenant_id=self.tenant_id,
            channel_id="33",
            model="vlm",
            prompt="heartbeat",
        )

        self.assertEqual(manual.evicted_job_id, heartbeat.job.id)
        self.assertEqual(dropped.status, EnqueueStatus.DROPPED)
        with self.assertRaises(QueueFullError):
            service.enqueue_event(
                tenant_id=self.tenant_id,
                channel_id="34",
                model="vlm",
                prompt="event",
            )

        leased = self.repository.claim("worker", timedelta(seconds=10))
        self.repository.fail(
            leased.id,
            "worker",
            "permanent",
            RetryPolicy(max_attempts=1),
        )
        metrics = self.repository.metrics_snapshot()
        self.assertEqual(metrics.queue_depth, 1)
        self.assertEqual(metrics.dropped_count, 2)
        self.assertEqual(metrics.dead_letter_count, 1)

    def test_concurrent_enqueue_is_atomic_for_coalescing_and_idempotency(self):
        heartbeat_service = self.service_with_capacity(100)

        def enqueue_heartbeat(index):
            return heartbeat_service.enqueue_heartbeat(
                tenant_id=self.tenant_id,
                channel_id="40",
                model="vlm",
                prompt="heartbeat",
                payload={"frame": index},
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            outcomes = list(executor.map(enqueue_heartbeat, range(24)))
        self.assertEqual(
            sum(item.status is EnqueueStatus.ENQUEUED for item in outcomes),
            1,
        )
        self.assertEqual(
            sum(item.status is EnqueueStatus.COALESCED for item in outcomes),
            23,
        )

        event_service = self.service_with_capacity(100)

        def enqueue_event(_index):
            return event_service.enqueue_event(
                tenant_id=self.tenant_id,
                channel_id="41",
                model="vlm",
                prompt="event",
                idempotency_key="event-atomic",
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            events = list(executor.map(enqueue_event, range(24)))
        self.assertEqual(
            sum(item.status is EnqueueStatus.ENQUEUED for item in events),
            1,
        )
        self.assertEqual(
            sum(item.status is EnqueueStatus.IDEMPOTENT for item in events),
            23,
        )
        self.assertEqual(len({item.job.id for item in events}), 1)


if __name__ == "__main__":
    unittest.main()
