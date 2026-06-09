from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from threading import RLock
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from .clock import Clock, SystemClock
from .errors import (
    InvalidJobStateError,
    JobNotFoundError,
    LeaseLostError,
    QueueFullError,
)
from .models import (
    EnqueueResult,
    EnqueueStatus,
    InferenceJob,
    JobResult,
    JobState,
    MetricsSnapshot,
    RetryPolicy,
    WorkloadClass,
)


_HeartbeatKey = Tuple[str, str, str, str]
_IdempotencyKey = Tuple[str, str]


class InMemoryInferenceQueueRepository:
    """Thread-safe reference implementation of the repository protocol."""

    def __init__(self, clock: Optional[Clock] = None) -> None:
        self._clock = clock or SystemClock()
        self._lock = RLock()
        self._jobs: Dict[str, InferenceJob] = {}
        self._results: Dict[str, JobResult] = {}
        self._sequence: Dict[str, int] = {}
        self._next_sequence = 0
        self._idempotency: Dict[_IdempotencyKey, str] = {}
        self._heartbeats: Dict[_HeartbeatKey, str] = {}
        self._coalesced_count = 0
        self._dropped_count = 0
        self._retried_count = 0
        self._dead_letter_count = 0

    def enqueue(self, job: InferenceJob, capacity: int) -> EnqueueResult:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        if job.state is not JobState.QUEUED:
            raise InvalidJobStateError("only queued jobs can be enqueued")

        with self._lock:
            if job.id in self._jobs:
                raise ValueError(f"job id already exists: {job.id}")

            duplicate = self._idempotent_job(job)
            if duplicate is not None:
                return EnqueueResult(duplicate, EnqueueStatus.IDEMPOTENT)

            coalesced = self._coalesced_heartbeat(job)
            if coalesced is not None:
                refreshed, replaced_payload = coalesced
                if job.idempotency_key is not None:
                    self._idempotency[
                        (job.tenant_id, job.idempotency_key)
                    ] = refreshed.id
                self._coalesced_count += 1
                return EnqueueResult(
                    refreshed,
                    EnqueueStatus.COALESCED,
                    replaced_payload=replaced_payload,
                )

            evicted_job_id = None
            evicted_payload = None
            if self._queue_depth() >= capacity:
                if job.workload_class is WorkloadClass.HEARTBEAT:
                    dropped = replace(
                        job,
                        state=JobState.DROPPED,
                        last_error="queue capacity reached",
                        finished_at=self._clock.now(),
                    )
                    self._store_new(dropped)
                    self._dropped_count += 1
                    return EnqueueResult(dropped, EnqueueStatus.DROPPED)

                evicted = self._evictable_heartbeat()
                if evicted is None:
                    raise QueueFullError(capacity, job.workload_class)
                evicted_payload = evicted.payload
                self._drop_existing(evicted, "evicted for event/manual workload")
                evicted_job_id = evicted.id

            self._store_new(job)
            self._activate(job)
            return EnqueueResult(
                job,
                EnqueueStatus.ENQUEUED,
                evicted_job_id=evicted_job_id,
                evicted_payload=evicted_payload,
            )

    def claim(
        self,
        worker_id: str,
        lease_duration: timedelta,
        workload_classes: Optional[Iterable[WorkloadClass]] = None,
    ) -> Optional[InferenceJob]:
        _validate_worker(worker_id)
        _validate_duration(lease_duration)
        allowed = (
            {WorkloadClass(item) for item in workload_classes}
            if workload_classes is not None
            else None
        )

        with self._lock:
            now = self._clock.now()
            self._expire_queued_deadlines(now)
            candidates = [
                job
                for job in self._jobs.values()
                if job.state is JobState.QUEUED
                and job.available_at <= now
                and (allowed is None or job.workload_class in allowed)
            ]
            if not candidates:
                return None
            job = min(candidates, key=self._claim_order)
            leased = replace(
                job,
                state=JobState.LEASED,
                attempt=job.attempt + 1,
                lease_owner=worker_id,
                lease_expires_at=now + lease_duration,
            )
            self._jobs[job.id] = leased
            self._deactivate(job)
            return leased

    def renew_lease(
        self, job_id: str, worker_id: str, lease_duration: timedelta
    ) -> InferenceJob:
        _validate_worker(worker_id)
        _validate_duration(lease_duration)
        with self._lock:
            now = self._clock.now()
            job = self._leased_job(job_id, worker_id, now)
            renewed = replace(job, lease_expires_at=now + lease_duration)
            self._jobs[job_id] = renewed
            return renewed

    def complete(
        self, job_id: str, worker_id: str, result: JobResult
    ) -> JobResult:
        _validate_worker(worker_id)
        if result.job_id != job_id:
            raise ValueError("result job_id does not match completed job")
        if result.worker_id != worker_id:
            raise ValueError("result worker_id does not match lease owner")
        with self._lock:
            now = self._clock.now()
            job = self._leased_job(job_id, worker_id, now)
            completed = replace(
                job,
                state=JobState.SUCCEEDED,
                lease_owner=None,
                lease_expires_at=None,
                finished_at=result.completed_at,
            )
            self._jobs[job_id] = completed
            self._results[job_id] = result
            self._deactivate(job)
            return result

    def fail(
        self,
        job_id: str,
        worker_id: str,
        error: str,
        retry_policy: RetryPolicy,
    ) -> InferenceJob:
        _validate_worker(worker_id)
        if not error:
            raise ValueError("error cannot be empty")
        with self._lock:
            now = self._clock.now()
            job = self._leased_job(job_id, worker_id, now)
            max_attempts = min(job.max_attempts, retry_policy.max_attempts)
            if self._queued_heartbeat(job) is not None:
                return self._drop_leased(
                    job, "superseded by newer queued heartbeat", now
                )
            if job.attempt >= max_attempts:
                return self._dead_letter(job, error, now)

            available_at = now + retry_policy.delay_for(job.attempt)
            if job.deadline is not None and available_at >= job.deadline:
                return self._dead_letter(job, error, now)

            retried = replace(
                job,
                state=JobState.QUEUED,
                available_at=available_at,
                lease_owner=None,
                lease_expires_at=None,
                last_error=error,
            )
            self._jobs[job_id] = retried
            self._activate(retried)
            self._retried_count += 1
            return retried

    def recover_stale_leases(self, retry_policy: RetryPolicy) -> int:
        with self._lock:
            now = self._clock.now()
            stale = [
                job
                for job in self._jobs.values()
                if job.state is JobState.LEASED
                and job.lease_expires_at is not None
                and job.lease_expires_at <= now
            ]
            for job in stale:
                if self._queued_heartbeat(job) is not None:
                    self._drop_leased(
                        job, "superseded by newer queued heartbeat", now
                    )
                    continue
                max_attempts = min(job.max_attempts, retry_policy.max_attempts)
                if (
                    job.attempt >= max_attempts
                    or (job.deadline is not None and job.deadline <= now)
                ):
                    self._dead_letter(job, "lease expired", now)
                    continue
                recovered = replace(
                    job,
                    state=JobState.QUEUED,
                    available_at=now,
                    lease_owner=None,
                    lease_expires_at=None,
                    last_error="lease expired",
                )
                self._jobs[job.id] = recovered
                self._activate(recovered)
                self._retried_count += 1
            return len(stale)

    def metrics_snapshot(self) -> MetricsSnapshot:
        with self._lock:
            now = self._clock.now()
            queued = [
                job for job in self._jobs.values() if job.state is JobState.QUEUED
            ]
            leased_count = sum(
                job.state is JobState.LEASED for job in self._jobs.values()
            )
            oldest_age = (
                max((now - job.created_at).total_seconds() for job in queued)
                if queued
                else 0.0
            )
            return MetricsSnapshot(
                queue_depth=len(queued),
                active_count=len(queued) + leased_count,
                leased_count=leased_count,
                oldest_age_seconds=max(0.0, oldest_age),
                coalesced_count=self._coalesced_count,
                dropped_count=self._dropped_count,
                retried_count=self._retried_count,
                dead_letter_count=self._dead_letter_count,
            )

    def metrics(self) -> MetricsSnapshot:
        return self.metrics_snapshot()

    def get_job(self, job_id: str) -> Optional[InferenceJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def get_result(self, job_id: str) -> Optional[JobResult]:
        with self._lock:
            return self._results.get(job_id)

    def list_jobs(self) -> List[InferenceJob]:
        with self._lock:
            return sorted(
                self._jobs.values(), key=lambda job: self._sequence[job.id]
            )

    def _store_new(self, job: InferenceJob) -> None:
        self._jobs[job.id] = job
        self._sequence[job.id] = self._next_sequence
        self._next_sequence += 1
        if job.idempotency_key is not None:
            self._idempotency[(job.tenant_id, job.idempotency_key)] = job.id

    def _activate(self, job: InferenceJob) -> None:
        if job.workload_class is WorkloadClass.HEARTBEAT:
            self._heartbeats[self._heartbeat_key(job)] = job.id

    def _deactivate(self, job: InferenceJob) -> None:
        if job.workload_class is WorkloadClass.HEARTBEAT:
            key = self._heartbeat_key(job)
            if self._heartbeats.get(key) == job.id:
                self._heartbeats.pop(key, None)

    def _idempotent_job(self, job: InferenceJob) -> Optional[InferenceJob]:
        if job.idempotency_key is None:
            return None
        existing_id = self._idempotency.get(
            (job.tenant_id, job.idempotency_key)
        )
        return self._jobs.get(existing_id) if existing_id is not None else None

    def _coalesced_heartbeat(
        self, job: InferenceJob
    ) -> Optional[Tuple[InferenceJob, Mapping[str, object]]]:
        if job.workload_class is not WorkloadClass.HEARTBEAT:
            return None
        existing_id = self._heartbeats.get(self._heartbeat_key(job))
        if existing_id is None:
            return None
        existing = self._jobs.get(existing_id)
        if existing is None or existing.state not in {
            JobState.QUEUED,
            JobState.LEASED,
        }:
            self._heartbeats.pop(self._heartbeat_key(job), None)
            return None
        if existing.state is JobState.LEASED:
            self._heartbeats.pop(self._heartbeat_key(job), None)
            return None
        refreshed = replace(
            existing,
            priority=job.priority,
            created_at=job.created_at,
            available_at=job.available_at,
            payload=job.payload,
            deadline=job.deadline,
            idempotency_key=existing.idempotency_key or job.idempotency_key,
            attempt=0,
            max_attempts=job.max_attempts,
            last_error=None,
        )
        self._jobs[existing.id] = refreshed
        return refreshed, existing.payload

    def _queue_depth(self) -> int:
        return sum(
            job.state is JobState.QUEUED
            for job in self._jobs.values()
        )

    def _queued_heartbeat(self, job: InferenceJob) -> Optional[InferenceJob]:
        if job.workload_class is not WorkloadClass.HEARTBEAT:
            return None
        queued_id = self._heartbeats.get(self._heartbeat_key(job))
        if queued_id is None or queued_id == job.id:
            return None
        queued = self._jobs.get(queued_id)
        return queued if queued is not None and queued.state is JobState.QUEUED else None

    def _evictable_heartbeat(self) -> Optional[InferenceJob]:
        candidates = [
            job
            for job in self._jobs.values()
            if job.state is JobState.QUEUED
            and job.workload_class is WorkloadClass.HEARTBEAT
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda job: (
                job.priority,
                job.created_at,
                self._sequence[job.id],
            ),
        )

    def _drop_existing(self, job: InferenceJob, reason: str) -> None:
        dropped = replace(
            job,
            state=JobState.DROPPED,
            last_error=reason,
            finished_at=self._clock.now(),
        )
        self._jobs[job.id] = dropped
        self._deactivate(job)
        self._dropped_count += 1

    def _drop_leased(
        self, job: InferenceJob, reason: str, now: datetime
    ) -> InferenceJob:
        dropped = replace(
            job,
            state=JobState.DROPPED,
            lease_owner=None,
            lease_expires_at=None,
            last_error=reason,
            finished_at=now,
        )
        self._jobs[job.id] = dropped
        self._dropped_count += 1
        return dropped

    def _dead_letter(
        self, job: InferenceJob, error: str, now: datetime
    ) -> InferenceJob:
        dead = replace(
            job,
            state=JobState.DEAD_LETTER,
            lease_owner=None,
            lease_expires_at=None,
            last_error=error,
            finished_at=now,
        )
        self._jobs[job.id] = dead
        self._deactivate(job)
        self._dead_letter_count += 1
        return dead

    def _expire_queued_deadlines(self, now: datetime) -> None:
        expired = [
            job
            for job in self._jobs.values()
            if job.state is JobState.QUEUED
            and job.deadline is not None
            and job.deadline <= now
        ]
        for job in expired:
            self._dead_letter(job, "deadline exceeded", now)

    def _leased_job(
        self, job_id: str, worker_id: str, now: datetime
    ) -> InferenceJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        if job.state is not JobState.LEASED:
            raise InvalidJobStateError(
                f"job {job_id} is {job.state.value}, not leased"
            )
        if job.lease_owner != worker_id:
            raise LeaseLostError(f"job {job_id} is leased by another worker")
        if job.lease_expires_at is None or job.lease_expires_at <= now:
            raise LeaseLostError(f"lease for job {job_id} has expired")
        return job

    def _claim_order(self, job: InferenceJob) -> Tuple[float, float, datetime, int]:
        deadline = job.deadline.timestamp() if job.deadline is not None else float("inf")
        return (-float(job.priority), deadline, job.created_at, self._sequence[job.id])

    @staticmethod
    def _heartbeat_key(job: InferenceJob) -> _HeartbeatKey:
        return (job.tenant_id, job.channel_id, job.model, job.prompt)


def _validate_worker(worker_id: str) -> None:
    if not worker_id:
        raise ValueError("worker_id cannot be empty")


def _validate_duration(duration: timedelta) -> None:
    if duration <= timedelta(0):
        raise ValueError("lease_duration must be positive")
