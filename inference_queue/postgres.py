"""PostgreSQL-backed inference queue repository.

The repository is tenant-scoped so every transaction can apply the RLS context
required by the jobs schema. Prompt text and repository bookkeeping live under
an internal payload key and are never emitted to logs by this module.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from typing import Any, Optional

from eva_db import PsycopgPool, TransactionContext

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


_INTERNAL_PAYLOAD_KEY = "_eva_queue"
_USER_IDEMPOTENCY_PREFIX = "u:"
_JOB_IDEMPOTENCY_PREFIX = "j:"
_ACTOR_NAMESPACE = uuid.UUID("f36449aa-20c5-4a42-97cb-d9999720e614")

_JOB_COLUMNS = """
    id::text,
    tenant_id::text,
    channel_id,
    model_id,
    workload_class,
    priority,
    created_at,
    available_at,
    payload,
    deadline_at,
    idempotency_key,
    attempt_count,
    max_attempts,
    state,
    lease_owner,
    lease_expires_at,
    error_metadata,
    finished_at,
    result_metadata
"""


class PostgresInferenceQueueRepository:
    """Atomic PostgreSQL implementation of ``InferenceQueueRepository``."""

    def __init__(
        self,
        pool: PsycopgPool,
        tenant_id: str | uuid.UUID,
        *,
        actor_id: str | uuid.UUID | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._pool = pool
        self._tenant_id = _uuid_text(tenant_id, "tenant_id")
        effective_actor = actor_id or uuid.uuid5(
            _ACTOR_NAMESPACE, f"inference-queue:{self._tenant_id}"
        )
        self._context = TransactionContext(
            tenant_id=self._tenant_id,
            actor_id=_uuid_text(effective_actor, "actor_id"),
        )
        self._clock = clock or SystemClock()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(tenant_id={self._tenant_id!r}, "
            f"pool={self._pool!r})"
        )

    def enqueue(self, job: InferenceJob, capacity: int) -> EnqueueResult:
        channel_id = self._validate_enqueue(job, capacity)
        now = self._clock.now()
        stored_idempotency = _stored_idempotency_key(job)
        packed_payload = _pack_payload(job)

        try:
            with self._pool.transaction(self._context) as connection:
                self._lock_admission(connection)
                if self._job_id_exists(connection, job.id):
                    raise ValueError(f"job id already exists: {job.id}")

                if job.idempotency_key is not None:
                    duplicate = self._find_idempotent(
                        connection, _user_idempotency_key(job.idempotency_key)
                    )
                    if duplicate is not None:
                        return EnqueueResult(
                            duplicate, EnqueueStatus.IDEMPOTENT
                        )

                if job.workload_class is WorkloadClass.HEARTBEAT:
                    queued = self._find_queued_heartbeat(
                        connection,
                        channel_id=channel_id,
                        model=job.model,
                        prompt=job.prompt,
                    )
                    if queued is not None:
                        refreshed = self._refresh_heartbeat(
                            connection,
                            queued,
                            incoming=job,
                            packed_payload=packed_payload,
                            now=now,
                        )
                        return EnqueueResult(
                            refreshed,
                            EnqueueStatus.COALESCED,
                            replaced_payload=queued.payload,
                        )

                queue_depth = connection.execute(
                    """
                    SELECT count(*)
                    FROM jobs.inference_jobs
                    WHERE tenant_id = %s AND state = 'queued'
                    """,
                    (self._tenant_id,),
                ).fetchone()[0]

                evicted_job_id: str | None = None
                evicted_payload: Mapping[str, Any] | None = None
                if queue_depth >= capacity:
                    if job.workload_class is WorkloadClass.HEARTBEAT:
                        dropped = self._insert_job(
                            connection,
                            job,
                            channel_id=channel_id,
                            stored_idempotency=stored_idempotency,
                            packed_payload=packed_payload,
                            state=JobState.DROPPED,
                            error="queue capacity reached",
                            finished_at=now,
                        )
                        return EnqueueResult(
                            dropped, EnqueueStatus.DROPPED
                        )

                    evicted = connection.execute(
                        f"""
                        SELECT {_JOB_COLUMNS}
                        FROM jobs.inference_jobs
                        WHERE tenant_id = %s
                          AND state = 'queued'
                          AND workload_class = 'heartbeat'
                        ORDER BY priority ASC, created_at ASC, id ASC
                        FOR UPDATE
                        LIMIT 1
                        """,
                        (self._tenant_id,),
                    ).fetchone()
                    if evicted is None:
                        raise _QueueFullSignal
                    evicted_job = _row_to_job(evicted)
                    evicted_job_id = evicted_job.id
                    evicted_payload = evicted_job.payload
                    self._mark_terminal(
                        connection,
                        evicted_job_id,
                        JobState.DROPPED,
                        "evicted for event/manual workload",
                        now,
                    )

                inserted = self._insert_job(
                    connection,
                    job,
                    channel_id=channel_id,
                    stored_idempotency=stored_idempotency,
                    packed_payload=packed_payload,
                )
                return EnqueueResult(
                    inserted,
                    EnqueueStatus.ENQUEUED,
                    evicted_job_id=evicted_job_id,
                    evicted_payload=evicted_payload,
                )
        except _QueueFullSignal:
            raise QueueFullError(
                capacity, job.workload_class
            ) from None

    def claim(
        self,
        worker_id: str,
        lease_duration: timedelta,
        workload_classes: Optional[Iterable[WorkloadClass]] = None,
    ) -> Optional[InferenceJob]:
        _validate_worker(worker_id)
        _validate_duration(lease_duration)
        allowed = (
            tuple(WorkloadClass(item).value for item in workload_classes)
            if workload_classes is not None
            else None
        )
        if allowed == ():
            return None

        now = self._clock.now()
        lease_expires_at = now + lease_duration
        with self._pool.transaction(self._context) as connection:
            self._expire_queued_deadlines(connection, now)
            workload_clause = ""
            parameters: list[Any] = [self._tenant_id, now]
            if allowed is not None:
                workload_clause = "AND workload_class = ANY(%s)"
                parameters.append(list(allowed))
            candidate = connection.execute(
                f"""
                SELECT {_JOB_COLUMNS}
                FROM jobs.inference_jobs
                WHERE tenant_id = %s
                  AND state = 'queued'
                  AND available_at <= %s
                  {workload_clause}
                ORDER BY
                    priority DESC,
                    deadline_at ASC NULLS LAST,
                    created_at ASC,
                    id ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
                """,
                tuple(parameters),
            ).fetchone()
            if candidate is None:
                return None

            job_id = str(candidate[0])
            attempt = int(candidate[11]) + 1
            row = connection.execute(
                f"""
                UPDATE jobs.inference_jobs
                SET state = 'leased',
                    attempt_count = %s,
                    lease_owner = %s,
                    lease_expires_at = %s,
                    started_at = COALESCE(started_at, %s),
                    updated_at = %s
                WHERE tenant_id = %s AND id = %s
                RETURNING {_JOB_COLUMNS}
                """,
                (
                    attempt,
                    worker_id,
                    lease_expires_at,
                    now,
                    now,
                    self._tenant_id,
                    job_id,
                ),
            ).fetchone()
            connection.execute(
                """
                INSERT INTO jobs.job_attempts (
                    id,
                    tenant_id,
                    job_id,
                    attempt_number,
                    worker_id,
                    state,
                    started_at
                )
                VALUES (%s, %s, %s, %s, %s, 'started', %s)
                """,
                (
                    str(uuid.uuid4()),
                    self._tenant_id,
                    job_id,
                    attempt,
                    worker_id,
                    now,
                ),
            )
            return _row_to_job(row)

    def renew_lease(
        self, job_id: str, worker_id: str, lease_duration: timedelta
    ) -> InferenceJob:
        job_id = _uuid_text(job_id, "job_id")
        _validate_worker(worker_id)
        _validate_duration(lease_duration)
        now = self._clock.now()
        with self._pool.transaction(self._context) as connection:
            job = self._locked_job(connection, job_id)
            self._require_live_lease(job, worker_id, now)
            row = connection.execute(
                f"""
                UPDATE jobs.inference_jobs
                SET lease_expires_at = %s, updated_at = %s
                WHERE tenant_id = %s AND id = %s
                RETURNING {_JOB_COLUMNS}
                """,
                (now + lease_duration, now, self._tenant_id, job_id),
            ).fetchone()
            return _row_to_job(row)

    def complete(
        self, job_id: str, worker_id: str, result: JobResult
    ) -> JobResult:
        job_id = _uuid_text(job_id, "job_id")
        _validate_worker(worker_id)
        if result.job_id != job_id:
            raise ValueError("result job_id does not match completed job")
        if result.worker_id != worker_id:
            raise ValueError("result worker_id does not match lease owner")
        result_envelope = _result_envelope(result)
        now = self._clock.now()

        with self._pool.transaction(self._context) as connection:
            job = self._locked_job(connection, job_id)
            self._require_live_lease(job, worker_id, now)
            connection.execute(
                """
                UPDATE jobs.inference_jobs
                SET state = 'succeeded',
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    result_metadata = %s,
                    error_metadata = NULL,
                    finished_at = %s,
                    updated_at = %s
                WHERE tenant_id = %s AND id = %s
                """,
                (
                    _jsonb(result_envelope),
                    result.completed_at,
                    now,
                    self._tenant_id,
                    job_id,
                ),
            )
            self._finish_attempt(
                connection,
                job_id,
                job.attempt,
                state="succeeded",
                finished_at=result.completed_at,
            )
            return result

    def fail(
        self,
        job_id: str,
        worker_id: str,
        error: str,
        retry_policy: RetryPolicy,
    ) -> InferenceJob:
        job_id = _uuid_text(job_id, "job_id")
        _validate_worker(worker_id)
        if not error:
            raise ValueError("error cannot be empty")
        now = self._clock.now()

        with self._pool.transaction(self._context) as connection:
            job = self._locked_job(connection, job_id)
            self._require_live_lease(job, worker_id, now)
            self._finish_attempt(
                connection,
                job_id,
                job.attempt,
                state="failed",
                finished_at=now,
                error=error,
            )

            if self._queued_successor(connection, job) is not None:
                return self._mark_terminal(
                    connection,
                    job_id,
                    JobState.DROPPED,
                    "superseded by newer queued heartbeat",
                    now,
                )

            max_attempts = min(job.max_attempts, retry_policy.max_attempts)
            if job.attempt >= max_attempts:
                return self._mark_terminal(
                    connection, job_id, JobState.DEAD_LETTER, error, now
                )

            available_at = now + retry_policy.delay_for(job.attempt)
            if job.deadline is not None and available_at >= job.deadline:
                return self._mark_terminal(
                    connection, job_id, JobState.DEAD_LETTER, error, now
                )
            return self._requeue(
                connection,
                job,
                available_at=available_at,
                error=error,
                now=now,
            )

    def recover_stale_leases(self, retry_policy: RetryPolicy) -> int:
        now = self._clock.now()
        with self._pool.transaction(self._context) as connection:
            rows = connection.execute(
                f"""
                SELECT {_JOB_COLUMNS}
                FROM jobs.inference_jobs
                WHERE tenant_id = %s
                  AND state = 'leased'
                  AND lease_expires_at <= %s
                ORDER BY lease_expires_at ASC, id ASC
                FOR UPDATE SKIP LOCKED
                """,
                (self._tenant_id, now),
            ).fetchall()
            for row in rows:
                job = _row_to_job(row)
                self._finish_attempt(
                    connection,
                    job.id,
                    job.attempt,
                    state="abandoned",
                    finished_at=now,
                    error="lease expired",
                )
                if self._queued_successor(connection, job) is not None:
                    self._mark_terminal(
                        connection,
                        job.id,
                        JobState.DROPPED,
                        "superseded by newer queued heartbeat",
                        now,
                    )
                    continue

                max_attempts = min(job.max_attempts, retry_policy.max_attempts)
                if (
                    job.attempt >= max_attempts
                    or (job.deadline is not None and job.deadline <= now)
                ):
                    self._mark_terminal(
                        connection,
                        job.id,
                        JobState.DEAD_LETTER,
                        "lease expired",
                        now,
                    )
                    continue
                self._requeue(
                    connection,
                    job,
                    available_at=now,
                    error="lease expired",
                    now=now,
                )
            return len(rows)

    def metrics_snapshot(self) -> MetricsSnapshot:
        now = self._clock.now()
        with self._pool.transaction(self._context, readonly=True) as connection:
            row = connection.execute(
                """
                SELECT
                    count(*) FILTER (WHERE state = 'queued'),
                    count(*) FILTER (WHERE state IN ('queued', 'leased')),
                    count(*) FILTER (WHERE state = 'leased'),
                    min(created_at) FILTER (WHERE state = 'queued'),
                    COALESCE(sum(
                        CASE
                            WHEN COALESCE(
                                payload #>> '{_eva_queue,coalesced_count}', ''
                            ) ~ '^[0-9]+$'
                            THEN (payload #>> '{_eva_queue,coalesced_count}')::bigint
                            ELSE 0
                        END
                    ), 0),
                    count(*) FILTER (WHERE state = 'dropped'),
                    COALESCE(sum(
                        CASE
                            WHEN COALESCE(
                                payload #>> '{_eva_queue,retry_count}', ''
                            ) ~ '^[0-9]+$'
                            THEN (payload #>> '{_eva_queue,retry_count}')::bigint
                            ELSE 0
                        END
                    ), 0),
                    count(*) FILTER (WHERE state = 'dead_letter')
                FROM jobs.inference_jobs
                WHERE tenant_id = %s
                """,
                (self._tenant_id,),
            ).fetchone()
        oldest = row[3]
        oldest_age = (
            max(0.0, (now - oldest).total_seconds()) if oldest is not None else 0.0
        )
        return MetricsSnapshot(
            queue_depth=int(row[0]),
            active_count=int(row[1]),
            leased_count=int(row[2]),
            oldest_age_seconds=oldest_age,
            coalesced_count=int(row[4]),
            dropped_count=int(row[5]),
            retried_count=int(row[6]),
            dead_letter_count=int(row[7]),
        )

    def metrics(self) -> MetricsSnapshot:
        return self.metrics_snapshot()

    def get_job(self, job_id: str) -> Optional[InferenceJob]:
        job_id = _uuid_text(job_id, "job_id")
        with self._pool.transaction(self._context, readonly=True) as connection:
            row = connection.execute(
                f"""
                SELECT {_JOB_COLUMNS}
                FROM jobs.inference_jobs
                WHERE tenant_id = %s AND id = %s
                """,
                (self._tenant_id, job_id),
            ).fetchone()
        return None if row is None else _row_to_job(row)

    def get_result(self, job_id: str) -> Optional[JobResult]:
        job = self.get_job(job_id)
        if (
            job is None
            or job.state is not JobState.SUCCEEDED
            or job.finished_at is None
        ):
            return None
        with self._pool.transaction(self._context, readonly=True) as connection:
            row = connection.execute(
                """
                SELECT result_metadata
                FROM jobs.inference_jobs
                WHERE tenant_id = %s AND id = %s
                """,
                (self._tenant_id, job.id),
            ).fetchone()
        if row is None or not isinstance(row[0], Mapping):
            return None
        envelope = row[0]
        return JobResult(
            job_id=job.id,
            output=envelope.get("output"),
            completed_at=job.finished_at,
            worker_id=str(envelope.get("worker_id") or ""),
            metadata=envelope.get("metadata") or {},
        )

    def list_jobs(self) -> list[InferenceJob]:
        with self._pool.transaction(self._context, readonly=True) as connection:
            rows = connection.execute(
                f"""
                SELECT {_JOB_COLUMNS}
                FROM jobs.inference_jobs
                WHERE tenant_id = %s
                ORDER BY created_at ASC, id ASC
                """,
                (self._tenant_id,),
            ).fetchall()
        return [_row_to_job(row) for row in rows]

    def _validate_enqueue(self, job: InferenceJob, capacity: int) -> int:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        if job.state is not JobState.QUEUED:
            raise InvalidJobStateError("only queued jobs can be enqueued")
        if _uuid_text(job.id, "job.id") != job.id:
            raise ValueError("job.id must use canonical UUID text")
        if _uuid_text(job.tenant_id, "job.tenant_id") != self._tenant_id:
            raise ValueError("job tenant_id does not match repository tenant")
        try:
            channel_id = int(job.channel_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("channel_id must be a positive integer") from exc
        if channel_id < 1 or str(channel_id) != str(job.channel_id):
            raise ValueError("channel_id must be a positive integer")
        if not 0 <= job.priority <= 1000:
            raise ValueError("priority must be between 0 and 1000")
        if _INTERNAL_PAYLOAD_KEY in job.payload:
            raise ValueError(
                f"payload key {_INTERNAL_PAYLOAD_KEY!r} is reserved"
            )
        _validate_json(job.payload, "payload")
        return channel_id

    def _lock_admission(self, connection: Any) -> None:
        connection.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (f"eva-inference-enqueue:{self._tenant_id}",),
        )

    def _job_id_exists(self, connection: Any, job_id: str) -> bool:
        return (
            connection.execute(
                """
                SELECT 1
                FROM jobs.inference_jobs
                WHERE tenant_id = %s AND id = %s
                """,
                (self._tenant_id, job_id),
            ).fetchone()
            is not None
        )

    def _find_idempotent(
        self, connection: Any, stored_key: str
    ) -> InferenceJob | None:
        row = connection.execute(
            f"""
            SELECT {_JOB_COLUMNS}
            FROM jobs.inference_jobs
            WHERE tenant_id = %s
              AND (
                  idempotency_key = %s
                  OR COALESCE(
                      payload #> '{{{_INTERNAL_PAYLOAD_KEY},idempotency_aliases}}',
                      '[]'::jsonb
                  ) ? %s
              )
            ORDER BY created_at ASC, id ASC
            LIMIT 1
            """,
            (self._tenant_id, stored_key, stored_key),
        ).fetchone()
        return None if row is None else _row_to_job(row)

    def _find_queued_heartbeat(
        self,
        connection: Any,
        *,
        channel_id: int,
        model: str,
        prompt: str,
        exclude_job_id: str | None = None,
    ) -> InferenceJob | None:
        exclusion = ""
        parameters: list[Any] = [
            self._tenant_id,
            channel_id,
            model,
            prompt,
        ]
        if exclude_job_id is not None:
            exclusion = "AND id <> %s"
            parameters.append(exclude_job_id)
        row = connection.execute(
            f"""
            SELECT {_JOB_COLUMNS}
            FROM jobs.inference_jobs
            WHERE tenant_id = %s
              AND channel_id = %s
              AND model_id = %s
              AND workload_class = 'heartbeat'
              AND state = 'queued'
              AND payload #>> '{{{_INTERNAL_PAYLOAD_KEY},prompt}}' = %s
              {exclusion}
            ORDER BY created_at ASC, id ASC
            FOR UPDATE
            LIMIT 1
            """,
            tuple(parameters),
        ).fetchone()
        return None if row is None else _row_to_job(row)

    def _refresh_heartbeat(
        self,
        connection: Any,
        existing: InferenceJob,
        *,
        incoming: InferenceJob,
        packed_payload: dict[str, Any],
        now: datetime,
    ) -> InferenceJob:
        current_row = connection.execute(
            """
            SELECT payload, idempotency_key
            FROM jobs.inference_jobs
            WHERE tenant_id = %s AND id = %s
            FOR UPDATE
            """,
            (self._tenant_id, existing.id),
        ).fetchone()
        internal = dict((current_row[0] or {}).get(_INTERNAL_PAYLOAD_KEY) or {})
        aliases = list(internal.get("idempotency_aliases") or [])
        canonical = str(current_row[1])
        if incoming.idempotency_key is not None:
            incoming_key = _user_idempotency_key(incoming.idempotency_key)
            if canonical.startswith(_JOB_IDEMPOTENCY_PREFIX):
                canonical = incoming_key
            elif incoming_key != canonical and incoming_key not in aliases:
                aliases.append(incoming_key)

        new_internal = dict(packed_payload[_INTERNAL_PAYLOAD_KEY])
        new_internal["idempotency_aliases"] = aliases
        new_internal["coalesced_count"] = int(
            internal.get("coalesced_count") or 0
        ) + 1
        new_internal["retry_count"] = int(internal.get("retry_count") or 0)
        refreshed_payload = _plain_mapping(packed_payload)
        refreshed_payload[_INTERNAL_PAYLOAD_KEY] = new_internal

        row = connection.execute(
            f"""
            UPDATE jobs.inference_jobs
            SET priority = %s,
                created_at = %s,
                available_at = %s,
                payload = %s,
                deadline_at = %s,
                idempotency_key = %s,
                attempt_count = 0,
                max_attempts = %s,
                error_metadata = NULL,
                updated_at = %s
            WHERE tenant_id = %s AND id = %s AND state = 'queued'
            RETURNING {_JOB_COLUMNS}
            """,
            (
                incoming.priority,
                incoming.created_at,
                incoming.available_at,
                _jsonb(refreshed_payload),
                incoming.deadline,
                canonical,
                incoming.max_attempts,
                now,
                self._tenant_id,
                existing.id,
            ),
        ).fetchone()
        return _row_to_job(row)

    def _insert_job(
        self,
        connection: Any,
        job: InferenceJob,
        *,
        channel_id: int,
        stored_idempotency: str,
        packed_payload: Mapping[str, Any],
        state: JobState = JobState.QUEUED,
        error: str | None = None,
        finished_at: datetime | None = None,
    ) -> InferenceJob:
        row = connection.execute(
            f"""
            INSERT INTO jobs.inference_jobs (
                id,
                tenant_id,
                channel_id,
                workload_class,
                model_id,
                priority,
                deadline_at,
                state,
                attempt_count,
                max_attempts,
                idempotency_key,
                payload,
                error_metadata,
                created_at,
                available_at,
                finished_at,
                updated_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, 0, %s, %s, %s, %s,
                %s, %s, %s, %s
            )
            RETURNING {_JOB_COLUMNS}
            """,
            (
                job.id,
                self._tenant_id,
                channel_id,
                job.workload_class.value,
                job.model,
                job.priority,
                job.deadline,
                state.value,
                job.max_attempts,
                stored_idempotency,
                _jsonb(packed_payload),
                None if error is None else _jsonb({"message": error}),
                job.created_at,
                job.available_at,
                finished_at,
                self._clock.now(),
            ),
        ).fetchone()
        return _row_to_job(row)

    def _locked_job(self, connection: Any, job_id: str) -> InferenceJob:
        row = connection.execute(
            f"""
            SELECT {_JOB_COLUMNS}
            FROM jobs.inference_jobs
            WHERE tenant_id = %s AND id = %s
            FOR UPDATE
            """,
            (self._tenant_id, job_id),
        ).fetchone()
        if row is None:
            raise JobNotFoundError(job_id)
        return _row_to_job(row)

    @staticmethod
    def _require_live_lease(
        job: InferenceJob, worker_id: str, now: datetime
    ) -> None:
        if job.state is not JobState.LEASED:
            raise InvalidJobStateError(
                f"job {job.id} is {job.state.value}, not leased"
            )
        if job.lease_owner != worker_id:
            raise LeaseLostError(f"job {job.id} is leased by another worker")
        if job.lease_expires_at is None or job.lease_expires_at <= now:
            raise LeaseLostError(f"lease for job {job.id} has expired")

    def _queued_successor(
        self, connection: Any, job: InferenceJob
    ) -> InferenceJob | None:
        if job.workload_class is not WorkloadClass.HEARTBEAT:
            return None
        return self._find_queued_heartbeat(
            connection,
            channel_id=int(job.channel_id),
            model=job.model,
            prompt=job.prompt,
            exclude_job_id=job.id,
        )

    def _mark_terminal(
        self,
        connection: Any,
        job_id: str,
        state: JobState,
        error: str,
        now: datetime,
    ) -> InferenceJob:
        if state not in {JobState.DEAD_LETTER, JobState.DROPPED}:
            raise ValueError("terminal state must be dead_letter or dropped")
        row = connection.execute(
            f"""
            UPDATE jobs.inference_jobs
            SET state = %s,
                lease_owner = NULL,
                lease_expires_at = NULL,
                error_metadata = %s,
                finished_at = %s,
                updated_at = %s
            WHERE tenant_id = %s AND id = %s
            RETURNING {_JOB_COLUMNS}
            """,
            (
                state.value,
                _jsonb({"message": error}),
                now,
                now,
                self._tenant_id,
                job_id,
            ),
        ).fetchone()
        return _row_to_job(row)

    def _requeue(
        self,
        connection: Any,
        job: InferenceJob,
        *,
        available_at: datetime,
        error: str,
        now: datetime,
    ) -> InferenceJob:
        payload_row = connection.execute(
            """
            SELECT payload
            FROM jobs.inference_jobs
            WHERE tenant_id = %s AND id = %s
            FOR UPDATE
            """,
            (self._tenant_id, job.id),
        ).fetchone()
        payload = dict(payload_row[0] or {})
        internal = dict(payload.get(_INTERNAL_PAYLOAD_KEY) or {})
        internal["retry_count"] = int(internal.get("retry_count") or 0) + 1
        payload[_INTERNAL_PAYLOAD_KEY] = internal
        row = connection.execute(
            f"""
            UPDATE jobs.inference_jobs
            SET state = 'queued',
                available_at = %s,
                lease_owner = NULL,
                lease_expires_at = NULL,
                payload = %s,
                error_metadata = %s,
                finished_at = NULL,
                updated_at = %s
            WHERE tenant_id = %s AND id = %s
            RETURNING {_JOB_COLUMNS}
            """,
            (
                available_at,
                _jsonb(payload),
                _jsonb({"message": error}),
                now,
                self._tenant_id,
                job.id,
            ),
        ).fetchone()
        return _row_to_job(row)

    def _finish_attempt(
        self,
        connection: Any,
        job_id: str,
        attempt: int,
        *,
        state: str,
        finished_at: datetime,
        error: str | None = None,
    ) -> None:
        connection.execute(
            """
            UPDATE jobs.job_attempts
            SET state = %s,
                finished_at = %s,
                error_class = %s,
                safe_error_metadata = %s
            WHERE tenant_id = %s
              AND job_id = %s
              AND attempt_number = %s
              AND state = 'started'
            """,
            (
                state,
                finished_at,
                None if error is None else "InferenceError",
                None if error is None else _jsonb({"message": error}),
                self._tenant_id,
                job_id,
                attempt,
            ),
        )

    def _expire_queued_deadlines(
        self, connection: Any, now: datetime
    ) -> None:
        connection.execute(
            """
            UPDATE jobs.inference_jobs
            SET state = 'dead_letter',
                error_metadata = %s,
                finished_at = %s,
                updated_at = %s
            WHERE tenant_id = %s
              AND state = 'queued'
              AND deadline_at IS NOT NULL
              AND deadline_at <= %s
            """,
            (
                _jsonb({"message": "deadline exceeded"}),
                now,
                now,
                self._tenant_id,
                now,
            ),
        )


PostgreSQLInferenceQueueRepository = PostgresInferenceQueueRepository


class _QueueFullSignal(Exception):
    pass


def _row_to_job(row: Any) -> InferenceJob:
    raw_payload = dict(row[8] or {})
    internal = dict(raw_payload.pop(_INTERNAL_PAYLOAD_KEY, {}) or {})
    prompt = internal.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("stored inference job has no prompt metadata")
    error_metadata = row[16] if isinstance(row[16], Mapping) else {}
    return InferenceJob(
        id=str(row[0]),
        tenant_id=str(row[1]),
        channel_id=str(row[2]),
        model=str(row[3]),
        prompt=prompt,
        workload_class=WorkloadClass(row[4]),
        priority=int(row[5]),
        created_at=row[6],
        available_at=row[7],
        payload=raw_payload,
        deadline=row[9],
        idempotency_key=_public_idempotency_key(str(row[10])),
        attempt=int(row[11]),
        max_attempts=int(row[12]),
        state=JobState(row[13]),
        lease_owner=row[14],
        lease_expires_at=row[15],
        last_error=error_metadata.get("message"),
        finished_at=row[17],
    )


def _pack_payload(job: InferenceJob) -> dict[str, Any]:
    payload = _plain_mapping(job.payload)
    payload[_INTERNAL_PAYLOAD_KEY] = {
        "prompt": job.prompt,
        "idempotency_aliases": [],
        "coalesced_count": 0,
        "retry_count": 0,
    }
    return payload


def _result_envelope(result: JobResult) -> dict[str, Any]:
    envelope = {
        "output": _plain_value(result.output),
        "metadata": _plain_mapping(result.metadata),
        "worker_id": result.worker_id,
        "completed_at": result.completed_at.isoformat(),
    }
    _validate_json(envelope, "result")
    return envelope


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _plain_value(item) for key, item in value.items()}


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _plain_mapping(value)
    if isinstance(value, (tuple, list)):
        return [_plain_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_plain_value(item) for item in sorted(value, key=repr)]
    return value


def _validate_json(value: Any, field_name: str) -> None:
    try:
        json.dumps(_plain_value(value), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON serializable") from exc


def _jsonb(value: Any) -> Any:
    from psycopg.types.json import Jsonb

    return Jsonb(_plain_value(value))


def _stored_idempotency_key(job: InferenceJob) -> str:
    if job.idempotency_key is None:
        return f"{_JOB_IDEMPOTENCY_PREFIX}{job.id}"
    return _user_idempotency_key(job.idempotency_key)


def _user_idempotency_key(value: str) -> str:
    return f"{_USER_IDEMPOTENCY_PREFIX}{value}"


def _public_idempotency_key(value: str) -> str | None:
    if value.startswith(_USER_IDEMPOTENCY_PREFIX):
        return value[len(_USER_IDEMPOTENCY_PREFIX) :]
    return None


def _uuid_text(value: str | uuid.UUID, field_name: str) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"{field_name} must be a UUID") from exc


def _validate_worker(worker_id: str) -> None:
    if not worker_id:
        raise ValueError("worker_id cannot be empty")


def _validate_duration(duration: timedelta) -> None:
    if duration <= timedelta(0):
        raise ValueError("lease_duration must be positive")
