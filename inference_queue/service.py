from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Mapping, Optional
from uuid import uuid4

from .clock import Clock, SystemClock
from .models import (
    DEFAULT_PRIORITIES,
    EnqueueResult,
    InferenceJob,
    RetryPolicy,
    WorkloadClass,
)
from .repository import InferenceQueueRepository


class InferenceEnqueueService:
    """Non-blocking admission service; it never performs or waits for inference."""

    def __init__(
        self,
        repository: InferenceQueueRepository,
        *,
        capacity: int,
        clock: Optional[Clock] = None,
        retry_policy: Optional[RetryPolicy] = None,
        id_factory: Optional[Callable[[], str]] = None,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._repository = repository
        self._capacity = capacity
        self._clock = clock or SystemClock()
        self._retry_policy = retry_policy or RetryPolicy()
        self._id_factory = id_factory or (lambda: str(uuid4()))

    def enqueue(
        self,
        *,
        tenant_id: str,
        channel_id: str,
        model: str,
        prompt: str,
        workload_class: WorkloadClass,
        payload: Optional[Mapping[str, Any]] = None,
        priority: Optional[int] = None,
        deadline: Optional[datetime] = None,
        idempotency_key: Optional[str] = None,
    ) -> EnqueueResult:
        workload = WorkloadClass(workload_class)
        now = self._clock.now()
        job = InferenceJob(
            id=self._id_factory(),
            tenant_id=tenant_id,
            channel_id=str(channel_id),
            model=model,
            prompt=prompt,
            workload_class=workload,
            priority=DEFAULT_PRIORITIES[workload] if priority is None else priority,
            payload=payload or {},
            created_at=now,
            available_at=now,
            deadline=deadline,
            idempotency_key=idempotency_key,
            max_attempts=self._retry_policy.max_attempts,
        )
        return self._repository.enqueue(job, self._capacity)

    def enqueue_heartbeat(self, **kwargs: Any) -> EnqueueResult:
        return self.enqueue(workload_class=WorkloadClass.HEARTBEAT, **kwargs)

    def enqueue_event(self, **kwargs: Any) -> EnqueueResult:
        return self.enqueue(workload_class=WorkloadClass.EVENT, **kwargs)

    def enqueue_manual(self, **kwargs: Any) -> EnqueueResult:
        return self.enqueue(workload_class=WorkloadClass.MANUAL, **kwargs)
