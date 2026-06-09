from __future__ import annotations

from dataclasses import dataclass

from .models import WorkloadClass


class InferenceQueueError(Exception):
    """Base error for inference queue domain operations."""


@dataclass(frozen=True)
class QueueFullError(InferenceQueueError):
    capacity: int
    workload_class: WorkloadClass

    def __str__(self) -> str:
        return (
            f"inference queue capacity {self.capacity} reached; "
            f"cannot admit {self.workload_class.value} work"
        )


class JobNotFoundError(InferenceQueueError):
    pass


class InvalidJobStateError(InferenceQueueError):
    pass


class LeaseLostError(InferenceQueueError):
    pass
