from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Optional


class WorkloadClass(str, Enum):
    HEARTBEAT = "heartbeat"
    EVENT = "event"
    MANUAL = "manual"


DEFAULT_PRIORITIES = {
    WorkloadClass.HEARTBEAT: 100,
    WorkloadClass.EVENT: 200,
    WorkloadClass.MANUAL: 300,
}


class JobState(str, Enum):
    QUEUED = "queued"
    LEASED = "leased"
    SUCCEEDED = "succeeded"
    DEAD_LETTER = "dead_letter"
    DROPPED = "dropped"


class EnqueueStatus(str, Enum):
    ENQUEUED = "enqueued"
    COALESCED = "coalesced"
    IDEMPOTENT = "idempotent"
    DROPPED = "dropped"


def heartbeat_coalescing_enabled(job: "InferenceJob") -> bool:
    """Whether a heartbeat may replace an older queued heartbeat.

    Generic health heartbeats are safely replaceable with their newest value.
    Evidence-bearing heartbeats, including L0 VLM windows, are not.  Those
    producers set ``coalesce_heartbeat`` to ``False`` in the job payload.
    """

    if job.workload_class is not WorkloadClass.HEARTBEAT:
        return False
    value = job.payload.get("coalesce_heartbeat", True)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


class FrozenMapping(Mapping[str, Any]):
    """Small immutable mapping used to keep frozen models deeply immutable."""

    __slots__ = ("_data",)

    def __init__(self, values: Optional[Mapping[str, Any]] = None) -> None:
        frozen = {str(key): _freeze(value) for key, value in (values or {}).items()}
        object.__setattr__(self, "_data", MappingProxyType(frozen))

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return repr(dict(self._data))

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError("FrozenMapping is immutable")


def _freeze(value: Any) -> Any:
    if isinstance(value, FrozenMapping):
        return value
    if isinstance(value, Mapping):
        return FrozenMapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def _aware_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    initial_backoff: timedelta = timedelta(seconds=1)
    multiplier: float = 2.0
    max_backoff: timedelta = timedelta(minutes=1)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.initial_backoff < timedelta(0):
            raise ValueError("initial_backoff cannot be negative")
        if self.multiplier < 1:
            raise ValueError("multiplier must be at least 1")
        if self.max_backoff < self.initial_backoff:
            raise ValueError("max_backoff cannot be less than initial_backoff")

    def delay_for(self, failed_attempt: int) -> timedelta:
        if failed_attempt < 1:
            raise ValueError("failed_attempt must be at least 1")
        seconds = self.initial_backoff.total_seconds() * (
            self.multiplier ** (failed_attempt - 1)
        )
        return min(timedelta(seconds=seconds), self.max_backoff)


@dataclass(frozen=True, slots=True)
class InferenceJob:
    id: str
    tenant_id: str
    channel_id: str
    model: str
    prompt: str
    workload_class: WorkloadClass
    priority: int
    created_at: datetime
    available_at: datetime
    payload: Mapping[str, Any] = field(default_factory=FrozenMapping)
    deadline: Optional[datetime] = None
    idempotency_key: Optional[str] = None
    attempt: int = 0
    max_attempts: int = 3
    state: JobState = JobState.QUEUED
    lease_owner: Optional[str] = None
    lease_expires_at: Optional[datetime] = None
    last_error: Optional[str] = None
    finished_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        for field_name in ("id", "tenant_id", "channel_id", "model", "prompt"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} cannot be empty")
        object.__setattr__(self, "state", JobState(self.state))
        object.__setattr__(
            self, "workload_class", WorkloadClass(self.workload_class)
        )
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.attempt < 0 or self.attempt > self.max_attempts:
            raise ValueError("attempt must be between zero and max_attempts")
        if self.idempotency_key == "":
            raise ValueError("idempotency_key cannot be empty")
        object.__setattr__(self, "created_at", _aware_utc(self.created_at, "created_at"))
        object.__setattr__(
            self, "available_at", _aware_utc(self.available_at, "available_at")
        )
        if self.deadline is not None:
            deadline = _aware_utc(self.deadline, "deadline")
            if deadline <= self.created_at:
                raise ValueError("deadline must be after created_at")
            object.__setattr__(self, "deadline", deadline)
        if self.lease_expires_at is not None:
            object.__setattr__(
                self,
                "lease_expires_at",
                _aware_utc(self.lease_expires_at, "lease_expires_at"),
            )
        if self.finished_at is not None:
            object.__setattr__(
                self, "finished_at", _aware_utc(self.finished_at, "finished_at")
            )
        if self.state is JobState.LEASED:
            if self.lease_owner is None or self.lease_expires_at is None:
                raise ValueError("leased jobs require an owner and expiry")
            if self.finished_at is not None:
                raise ValueError("leased jobs cannot be finished")
        elif self.state is JobState.QUEUED:
            if self.lease_owner is not None or self.lease_expires_at is not None:
                raise ValueError("queued jobs cannot hold a lease")
            if self.finished_at is not None:
                raise ValueError("queued jobs cannot be finished")
        else:
            if self.lease_owner is not None or self.lease_expires_at is not None:
                raise ValueError("terminal jobs cannot hold a lease")
            if self.finished_at is None:
                raise ValueError("terminal jobs require finished_at")
        object.__setattr__(self, "payload", FrozenMapping(self.payload))

    @property
    def job_id(self) -> str:
        return self.id

    @property
    def workload(self) -> WorkloadClass:
        return self.workload_class


@dataclass(frozen=True, slots=True)
class JobResult:
    job_id: str
    output: Any
    completed_at: datetime
    worker_id: str
    metadata: Mapping[str, Any] = field(default_factory=FrozenMapping)

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ValueError("job_id cannot be empty")
        if not self.worker_id:
            raise ValueError("worker_id cannot be empty")
        object.__setattr__(
            self, "completed_at", _aware_utc(self.completed_at, "completed_at")
        )
        object.__setattr__(self, "output", _freeze(self.output))
        object.__setattr__(self, "metadata", FrozenMapping(self.metadata))


@dataclass(frozen=True, slots=True)
class EnqueueResult:
    job: InferenceJob
    status: EnqueueStatus
    evicted_job_id: Optional[str] = None
    replaced_payload: Optional[Mapping[str, Any]] = None
    evicted_payload: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if self.replaced_payload is not None:
            object.__setattr__(
                self,
                "replaced_payload",
                FrozenMapping(self.replaced_payload),
            )
        if self.evicted_payload is not None:
            object.__setattr__(
                self,
                "evicted_payload",
                FrozenMapping(self.evicted_payload),
            )

    @property
    def accepted(self) -> bool:
        return (
            self.status is not EnqueueStatus.DROPPED
            and self.job.state is not JobState.DROPPED
        )


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    queue_depth: int
    active_count: int
    leased_count: int
    oldest_age_seconds: float
    coalesced_count: int
    dropped_count: int
    retried_count: int
    dead_letter_count: int

    @property
    def oldest_age(self) -> timedelta:
        return timedelta(seconds=self.oldest_age_seconds)

    @property
    def coalesced(self) -> int:
        return self.coalesced_count

    @property
    def dropped(self) -> int:
        return self.dropped_count

    @property
    def retried(self) -> int:
        return self.retried_count

    @property
    def dead_letter(self) -> int:
        return self.dead_letter_count
