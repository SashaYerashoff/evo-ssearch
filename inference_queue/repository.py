from __future__ import annotations

from datetime import timedelta
from typing import Iterable, Optional, Protocol, runtime_checkable

from .models import (
    EnqueueResult,
    InferenceJob,
    JobResult,
    MetricsSnapshot,
    RetryPolicy,
    WorkloadClass,
)


@runtime_checkable
class InferenceQueueRepository(Protocol):
    """Atomic persistence boundary for a future database adapter."""

    def enqueue(self, job: InferenceJob, capacity: int) -> EnqueueResult:
        ...

    def claim(
        self,
        worker_id: str,
        lease_duration: timedelta,
        workload_classes: Optional[Iterable[WorkloadClass]] = None,
    ) -> Optional[InferenceJob]:
        ...

    def renew_lease(
        self, job_id: str, worker_id: str, lease_duration: timedelta
    ) -> InferenceJob:
        ...

    def complete(
        self, job_id: str, worker_id: str, result: JobResult
    ) -> JobResult:
        ...

    def fail(
        self,
        job_id: str,
        worker_id: str,
        error: str,
        retry_policy: RetryPolicy,
    ) -> InferenceJob:
        ...

    def recover_stale_leases(self, retry_policy: RetryPolicy) -> int:
        ...

    def metrics_snapshot(self) -> MetricsSnapshot:
        ...

    def metrics(self) -> MetricsSnapshot:
        ...

    def get_job(self, job_id: str) -> Optional[InferenceJob]:
        ...

    def get_result(self, job_id: str) -> Optional[JobResult]:
        ...
