import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from itertools import count

from inference_queue import (
    EnqueueStatus,
    InferenceEnqueueService,
    InferenceWorker,
    InMemoryInferenceQueueRepository,
    JobState,
    LeaseLostError,
    ManualClock,
    QueueFullError,
    RetryPolicy,
    WorkloadClass,
)


class InferenceQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = ManualClock(datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc))
        self.repository = InMemoryInferenceQueueRepository(self.clock)
        ids = count(1)
        self.retry_policy = RetryPolicy(
            max_attempts=3,
            initial_backoff=timedelta(seconds=5),
            multiplier=2,
            max_backoff=timedelta(seconds=30),
        )
        self.service = InferenceEnqueueService(
            self.repository,
            capacity=3,
            clock=self.clock,
            retry_policy=self.retry_policy,
            id_factory=lambda: f"job-{next(ids)}",
        )

    def enqueue(self, workload_class=WorkloadClass.EVENT, **overrides):
        values = {
            "tenant_id": "tenant-a",
            "channel_id": "channel-1",
            "model": "vlm-a",
            "prompt": "describe activity",
            "payload": {"frame": "frame-1", "regions": [1, 2]},
        }
        values.update(overrides)
        return self.service.enqueue(
            workload_class=workload_class,
            **values,
        )

    def worker(self, worker_id="worker-1", lease_seconds=10):
        return InferenceWorker(
            worker_id,
            self.repository,
            lease_duration=timedelta(seconds=lease_seconds),
            retry_policy=self.retry_policy,
            clock=self.clock,
        )

    def test_models_are_immutable(self):
        job = self.enqueue().job

        with self.assertRaises(FrozenInstanceError):
            job.priority = 1
        with self.assertRaises(TypeError):
            job.payload["frame"] = "changed"
        with self.assertRaises(TypeError):
            job.payload["regions"][0] = 9

        claimed = self.worker().claim()
        result = self.worker().complete(claimed.id, {"labels": ["person"]})
        with self.assertRaises(TypeError):
            result.output["labels"][0] = "vehicle"

    def test_heartbeat_coalesces_by_tenant_channel_model_and_prompt(self):
        first = self.enqueue(WorkloadClass.HEARTBEAT)
        duplicate = self.enqueue(
            WorkloadClass.HEARTBEAT,
            payload={"frame": "newer-frame"},
        )
        other_channel = self.enqueue(
            WorkloadClass.HEARTBEAT,
            channel_id="channel-2",
        )

        self.assertEqual(duplicate.status, EnqueueStatus.COALESCED)
        self.assertEqual(duplicate.job.id, first.job.id)
        self.assertNotEqual(other_channel.job.id, first.job.id)
        metrics = self.repository.metrics_snapshot()
        self.assertEqual(metrics.queue_depth, 2)
        self.assertEqual(metrics.coalesced_count, 1)

    def test_evidence_heartbeat_can_opt_out_of_replacement(self):
        first = self.enqueue(
            WorkloadClass.HEARTBEAT,
            payload={"frame": "window-1", "coalesce_heartbeat": False},
        )
        second = self.enqueue(
            WorkloadClass.HEARTBEAT,
            payload={"frame": "window-2", "coalesce_heartbeat": False},
        )

        self.assertEqual(first.status, EnqueueStatus.ENQUEUED)
        self.assertEqual(second.status, EnqueueStatus.ENQUEUED)
        self.assertNotEqual(first.job.id, second.job.id)
        metrics = self.repository.metrics_snapshot()
        self.assertEqual(metrics.queue_depth, 2)
        self.assertEqual(metrics.coalesced_count, 0)

    def test_queued_heartbeat_coalescing_refreshes_newest_content(self):
        original_deadline = self.clock.now() + timedelta(minutes=5)
        first = self.enqueue(
            WorkloadClass.HEARTBEAT,
            payload={"frame": "old-frame", "batch": [1]},
            priority=90,
            deadline=original_deadline,
        )
        marker = self.enqueue(
            WorkloadClass.EVENT,
            channel_id="channel-2",
            payload={"frame": "event-frame"},
        )
        self.clock.advance(3)
        newest_deadline = self.clock.now() + timedelta(minutes=1)
        refreshed = self.enqueue(
            WorkloadClass.HEARTBEAT,
            payload={"frame": "new-frame", "batch": [2, 3]},
            priority=140,
            deadline=newest_deadline,
        )

        self.assertEqual(refreshed.status, EnqueueStatus.COALESCED)
        self.assertEqual(refreshed.job.id, first.job.id)
        self.assertEqual(refreshed.job.payload["frame"], "new-frame")
        self.assertEqual(refreshed.job.payload["batch"], (2, 3))
        self.assertEqual(refreshed.job.priority, 140)
        self.assertEqual(refreshed.job.deadline, newest_deadline)
        self.assertEqual(refreshed.job.created_at, self.clock.now())
        self.assertEqual(
            [job.id for job in self.repository.list_jobs()],
            [first.job.id, marker.job.id],
        )

    def test_leased_heartbeat_gets_one_refreshable_queued_successor(self):
        first = self.enqueue(
            WorkloadClass.HEARTBEAT,
            payload={"frame": "leased-frame"},
        )
        worker = self.worker()
        leased = worker.claim()
        self.assertEqual(leased.id, first.job.id)

        successor = self.enqueue(
            WorkloadClass.HEARTBEAT,
            payload={"frame": "successor-1"},
        )
        newest = self.enqueue(
            WorkloadClass.HEARTBEAT,
            payload={"frame": "successor-2"},
        )

        self.assertEqual(successor.status, EnqueueStatus.ENQUEUED)
        self.assertNotEqual(successor.job.id, leased.id)
        self.assertEqual(newest.status, EnqueueStatus.COALESCED)
        self.assertEqual(newest.job.id, successor.job.id)
        self.assertEqual(newest.job.payload["frame"], "successor-2")
        metrics = self.repository.metrics_snapshot()
        self.assertEqual(metrics.queue_depth, 1)
        self.assertEqual(metrics.leased_count, 1)

        worker.complete(leased.id, "old frame complete")
        claimed_successor = worker.claim()
        self.assertEqual(claimed_successor.id, successor.job.id)
        self.assertEqual(claimed_successor.payload["frame"], "successor-2")

    def test_coalesced_heartbeat_keeps_incoming_idempotency_key(self):
        first = self.enqueue(WorkloadClass.HEARTBEAT)
        coalesced = self.enqueue(
            WorkloadClass.HEARTBEAT,
            idempotency_key="capture-2",
        )
        self.assertEqual(coalesced.status, EnqueueStatus.COALESCED)

        worker = self.worker()
        worker.claim()
        worker.complete(first.job.id, "done")
        repeated = self.enqueue(
            WorkloadClass.HEARTBEAT,
            idempotency_key="capture-2",
        )

        self.assertEqual(repeated.status, EnqueueStatus.IDEMPOTENT)
        self.assertEqual(repeated.job.id, first.job.id)
        self.assertEqual(repeated.job.state, JobState.SUCCEEDED)

    def test_claim_prefers_priority_then_deadline_then_fifo(self):
        late_deadline = self.clock.now() + timedelta(minutes=5)
        early_deadline = self.clock.now() + timedelta(minutes=1)
        first_event = self.enqueue(
            WorkloadClass.EVENT,
            channel_id="event-late",
            priority=250,
            deadline=late_deadline,
        ).job
        urgent_event = self.enqueue(
            WorkloadClass.EVENT,
            channel_id="event-early",
            priority=250,
            deadline=early_deadline,
        ).job
        manual = self.enqueue(
            WorkloadClass.MANUAL,
            channel_id="manual",
        ).job
        worker = self.worker()

        self.assertEqual(worker.claim().id, manual.id)
        worker.complete(manual.id, "done")
        self.assertEqual(worker.claim().id, urgent_event.id)
        worker.complete(urgent_event.id, "done")
        self.assertEqual(worker.claim().id, first_event.id)

    def test_capacity_drops_heartbeat_and_event_evicts_queued_heartbeat(self):
        service = InferenceEnqueueService(
            self.repository,
            capacity=2,
            clock=self.clock,
            retry_policy=self.retry_policy,
            id_factory=iter(["hb-1", "hb-2", "event-1", "hb-3"]).__next__,
        )
        heartbeat_one = service.enqueue_heartbeat(
            tenant_id="tenant-a",
            channel_id="channel-1",
            model="vlm",
            prompt="heartbeat",
        )
        service.enqueue_heartbeat(
            tenant_id="tenant-a",
            channel_id="channel-2",
            model="vlm",
            prompt="heartbeat",
        )

        event = service.enqueue_event(
            tenant_id="tenant-a",
            channel_id="channel-3",
            model="vlm",
            prompt="alarm",
        )
        dropped = service.enqueue_heartbeat(
            tenant_id="tenant-a",
            channel_id="channel-4",
            model="vlm",
            prompt="heartbeat",
        )

        self.assertEqual(event.status, EnqueueStatus.ENQUEUED)
        self.assertEqual(event.evicted_job_id, heartbeat_one.job.id)
        self.assertEqual(
            self.repository.get_job(heartbeat_one.job.id).state,
            JobState.DROPPED,
        )
        self.assertEqual(dropped.status, EnqueueStatus.DROPPED)
        self.assertFalse(dropped.accepted)
        metrics = self.repository.metrics_snapshot()
        self.assertEqual(metrics.active_count, 2)
        self.assertEqual(metrics.dropped_count, 2)

    def test_capacity_is_explicit_when_only_event_and_manual_work_remains(self):
        service = InferenceEnqueueService(
            self.repository,
            capacity=1,
            clock=self.clock,
            id_factory=iter(["event", "manual"]).__next__,
        )
        service.enqueue_event(
            tenant_id="tenant-a",
            channel_id="channel-1",
            model="vlm",
            prompt="alarm",
        )

        with self.assertRaises(QueueFullError):
            service.enqueue_manual(
                tenant_id="tenant-a",
                channel_id="channel-1",
                model="vlm",
                prompt="operator request",
            )

        self.assertEqual(self.repository.metrics_snapshot().queue_depth, 1)

    def test_idempotency_is_tenant_scoped_and_survives_completion(self):
        first = self.enqueue(idempotency_key="event-42")
        duplicate = self.enqueue(
            channel_id="another-channel",
            prompt="different prompt",
            idempotency_key="event-42",
        )
        other_tenant = self.enqueue(
            tenant_id="tenant-b",
            idempotency_key="event-42",
        )

        self.assertEqual(duplicate.status, EnqueueStatus.IDEMPOTENT)
        self.assertEqual(duplicate.job.id, first.job.id)
        self.assertNotEqual(other_tenant.job.id, first.job.id)

        worker = self.worker()
        claimed = worker.claim()
        worker.complete(claimed.id, {"summary": "complete"})
        repeated = self.enqueue(idempotency_key="event-42")
        self.assertEqual(repeated.status, EnqueueStatus.IDEMPOTENT)
        self.assertEqual(repeated.job.state, JobState.SUCCEEDED)

    def test_lease_renewal_completion_and_owner_checks(self):
        job = self.enqueue().job
        owner = self.worker("owner", lease_seconds=10)
        other = self.worker("other", lease_seconds=10)

        claimed = owner.claim()
        original_expiry = claimed.lease_expires_at
        self.clock.advance(5)
        renewed = owner.renew(job.id)
        self.assertGreater(renewed.lease_expires_at, original_expiry)
        with self.assertRaises(LeaseLostError):
            other.complete(job.id, "not allowed")

        result = owner.complete(job.id, {"answer": "ok"})
        self.assertEqual(self.repository.get_result(job.id), result)
        self.assertEqual(self.repository.get_job(job.id).state, JobState.SUCCEEDED)

    def test_stale_lease_is_recovered_for_another_worker(self):
        job = self.enqueue().job
        first_worker = self.worker("worker-1", lease_seconds=10)
        second_worker = self.worker("worker-2", lease_seconds=10)
        first_worker.claim()

        self.clock.advance(10)
        recovered = second_worker.claim()

        self.assertEqual(recovered.id, job.id)
        self.assertEqual(recovered.attempt, 2)
        self.assertEqual(recovered.lease_owner, "worker-2")
        with self.assertRaises(LeaseLostError):
            first_worker.complete(job.id, "late result")
        self.assertEqual(self.repository.metrics_snapshot().retried_count, 1)

    def test_failures_back_off_then_reach_dead_letter(self):
        job = self.enqueue().job
        worker = self.worker()

        worker.claim()
        first_failure = worker.fail(job.id, "temporary")
        self.assertEqual(first_failure.state, JobState.QUEUED)
        self.assertEqual(
            first_failure.available_at,
            self.clock.now() + timedelta(seconds=5),
        )
        self.assertIsNone(worker.claim())

        self.clock.advance(5)
        worker.claim()
        second_failure = worker.fail(job.id, "temporary again")
        self.assertEqual(
            second_failure.available_at,
            self.clock.now() + timedelta(seconds=10),
        )

        self.clock.advance(10)
        worker.claim()
        final_failure = worker.fail(job.id, "permanent")
        self.assertEqual(final_failure.state, JobState.DEAD_LETTER)
        self.assertEqual(final_failure.last_error, "permanent")
        metrics = self.repository.metrics_snapshot()
        self.assertEqual(metrics.retried_count, 2)
        self.assertEqual(metrics.dead_letter_count, 1)
        self.assertEqual(metrics.active_count, 0)

    def test_metrics_report_depth_oldest_age_and_counters(self):
        self.enqueue(WorkloadClass.HEARTBEAT)
        self.enqueue(WorkloadClass.HEARTBEAT)
        self.clock.advance(12)
        self.enqueue(
            WorkloadClass.EVENT,
            channel_id="channel-2",
            idempotency_key="event-id",
        )
        snapshot = self.repository.metrics_snapshot()

        self.assertEqual(snapshot.queue_depth, 2)
        self.assertEqual(snapshot.active_count, 2)
        self.assertEqual(snapshot.oldest_age_seconds, 12)
        self.assertEqual(snapshot.coalesced_count, 1)
        self.assertEqual(snapshot.dropped_count, 0)
        self.assertEqual(snapshot.retried_count, 0)
        self.assertEqual(snapshot.dead_letter_count, 0)

    def test_concurrent_heartbeat_enqueues_are_coalesced_atomically(self):
        service = InferenceEnqueueService(
            self.repository,
            capacity=10,
            clock=self.clock,
        )

        def enqueue_heartbeat(_index):
            return service.enqueue_heartbeat(
                tenant_id="tenant-a",
                channel_id="channel-1",
                model="vlm",
                prompt="heartbeat",
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            outcomes = list(executor.map(enqueue_heartbeat, range(50)))

        self.assertEqual(
            sum(outcome.status is EnqueueStatus.ENQUEUED for outcome in outcomes),
            1,
        )
        self.assertEqual(
            sum(outcome.status is EnqueueStatus.COALESCED for outcome in outcomes),
            49,
        )
        snapshot = self.repository.metrics()
        self.assertEqual(snapshot.queue_depth, 1)
        self.assertEqual(snapshot.coalesced_count, 49)


if __name__ == "__main__":
    unittest.main()
