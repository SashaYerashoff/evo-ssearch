from __future__ import annotations

from datetime import timedelta
from typing import Any, Iterable, Mapping, Optional

from .clock import Clock, SystemClock
from .models import InferenceJob, JobResult, RetryPolicy, WorkloadClass
from .repository import InferenceQueueRepository


class InferenceWorker:
    def __init__(
        self,
        worker_id: str,
        repository: InferenceQueueRepository,
        *,
        lease_duration: timedelta = timedelta(seconds=30),
        retry_policy: Optional[RetryPolicy] = None,
        clock: Optional[Clock] = None,
    ) -> None:
        if not worker_id:
            raise ValueError("worker_id cannot be empty")
        if lease_duration <= timedelta(0):
            raise ValueError("lease_duration must be positive")
        self.worker_id = worker_id
        self._repository = repository
        self._lease_duration = lease_duration
        self._retry_policy = retry_policy or RetryPolicy()
        self._clock = clock or SystemClock()

    def claim(
        self,
        workload_classes: Optional[Iterable[WorkloadClass]] = None,
    ) -> Optional[InferenceJob]:
        self.recover_stale_leases()
        return self._repository.claim(
            self.worker_id, self._lease_duration, workload_classes
        )

    def renew(
        self, job_id: str, lease_duration: Optional[timedelta] = None
    ) -> InferenceJob:
        return self._repository.renew_lease(
            job_id, self.worker_id, lease_duration or self._lease_duration
        )

    def complete(
        self,
        job_id: str,
        output: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> JobResult:
        result = JobResult(
            job_id=job_id,
            output=output,
            completed_at=self._clock.now(),
            worker_id=self.worker_id,
            metadata=metadata or {},
        )
        return self._repository.complete(job_id, self.worker_id, result)

    def fail(self, job_id: str, error: str) -> InferenceJob:
        return self._repository.fail(
            job_id, self.worker_id, error, self._retry_policy
        )

    def recover_stale_leases(self) -> int:
        return self._repository.recover_stale_leases(self._retry_policy)
