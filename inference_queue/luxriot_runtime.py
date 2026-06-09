"""Durable Luxriot summary dispatch backed by the inference queue."""

from __future__ import annotations

import json
import os
import re
import socket
import threading
import time
import uuid
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any

from .errors import InferenceQueueError
from .models import EnqueueStatus, JobState, WorkloadClass
from .repository import InferenceQueueRepository
from .service import InferenceEnqueueService
from .worker import InferenceWorker


_SPOOL_KIND = "luxriot_summary_v1"
_SPOOL_NAME = re.compile(r"^[0-9a-f]{32}\.json$")


class LuxriotInferenceQueueRuntime:
    """Queues frame batches and optionally consumes them in local worker threads."""

    def __init__(
        self,
        *,
        manager: Any,
        enqueue_repository: InferenceQueueRepository,
        tenant_id: str,
        capacity: int,
        spool_directory: str | Path,
        default_model: str,
        worker_repository: InferenceQueueRepository | None = None,
        worker_count: int = 0,
        poll_interval_seconds: float = 0.25,
        lease_seconds: float = 180.0,
        spool_retention_hours: float = 24.0,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        if worker_count < 0:
            raise ValueError("worker_count cannot be negative")
        if worker_count and worker_repository is None:
            raise ValueError("worker_repository is required when workers are enabled")
        self.manager = manager
        self.enqueue_repository = enqueue_repository
        self.worker_repository = worker_repository
        self.tenant_id = str(uuid.UUID(str(tenant_id)))
        self.capacity = int(capacity)
        self.default_model = str(default_model or "").strip()
        if not self.default_model:
            raise ValueError("default_model cannot be empty")
        self.worker_count = int(worker_count)
        self.poll_interval_seconds = max(0.05, float(poll_interval_seconds))
        self.lease_seconds = max(10.0, float(lease_seconds))
        self.spool_retention_seconds = max(
            3600.0,
            float(spool_retention_hours) * 3600.0,
        )
        self.spool_directory = Path(spool_directory).expanduser().resolve()
        self.spool_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            self.spool_directory.chmod(0o700)
        except OSError:
            pass

        self._enqueue_service = InferenceEnqueueService(
            enqueue_repository,
            capacity=self.capacity,
        )
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._apply_lock = threading.Lock()
        self._spool_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._started = False
        self._last_error: str | None = None
        self._last_gc_at = 0.0
        self._completed_count = 0
        self._failed_count = 0

    def start(self) -> None:
        with self._state_lock:
            if self._started:
                return
            self._started = True
        self._stop_event.clear()
        self._reconcile_completed()
        for index in range(self.worker_count):
            thread = threading.Thread(
                target=self._worker_loop,
                args=(index,),
                name=f"eva-inference-{index + 1}",
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        deadline = time.monotonic() + max(0.0, timeout)
        for thread in list(self._threads):
            remaining = max(0.0, deadline - time.monotonic())
            thread.join(timeout=remaining)
        self._threads.clear()
        with self._state_lock:
            self._started = False

    def enqueue_summary(
        self,
        batch: Mapping[str, Any],
        workload_class: str = "heartbeat",
    ) -> dict[str, Any]:
        workload = WorkloadClass(str(workload_class))
        channel_id = self._positive_channel_id(batch.get("channel_id"))
        model = str(batch.get("model_hint") or self.default_model).strip()
        prompt = str(batch.get("prompt") or "Describe notable activity.")
        run_id = str(batch.get("run_id") or "").strip()
        batch_start_ms = int(batch.get("batch_start_ms") or 0)
        batch_end_ms = int(batch.get("batch_end_ms") or batch_start_ms)
        idempotency_key = (
            f"luxriot-summary:{run_id or 'no-run'}:{channel_id}:"
            f"{batch_start_ms}:{batch_end_ms}:{workload.value}"
        )
        payload = {
            "kind": _SPOOL_KIND,
            "run_id": run_id,
            "frame_count": int(batch.get("frame_count") or 0),
            "batch_start_ms": batch_start_ms,
            "batch_end_ms": batch_end_ms,
        }
        with self._spool_lock:
            spool_name = self._write_spool(batch)
            payload["spool_file"] = spool_name
            try:
                result = self._enqueue_service.enqueue(
                    tenant_id=self.tenant_id,
                    channel_id=str(channel_id),
                    model=model,
                    prompt=prompt,
                    workload_class=workload,
                    payload=payload,
                    idempotency_key=idempotency_key,
                )
            except Exception:
                self._delete_spool(spool_name)
                raise

            if result.status is EnqueueStatus.IDEMPOTENT or not result.accepted:
                self._delete_spool(spool_name)
            self._delete_payload_spool(result.replaced_payload)
            self._delete_payload_spool(result.evicted_payload)
            self._maybe_collect_spool()
        return {
            "queued": True,
            "accepted": result.accepted,
            "status": result.status.value,
            "job_id": result.job.id,
            "evicted_job_id": result.evicted_job_id,
        }

    def status(self) -> dict[str, Any]:
        metrics = self.enqueue_repository.metrics_snapshot()
        with self._state_lock:
            started = self._started
            last_error = self._last_error
            completed = self._completed_count
            failed = self._failed_count
        return {
            "enabled": True,
            "started": started,
            "worker_count": self.worker_count,
            "workers_alive": sum(thread.is_alive() for thread in self._threads),
            "queue_depth": metrics.queue_depth,
            "active_count": metrics.active_count,
            "leased_count": metrics.leased_count,
            "oldest_age_seconds": metrics.oldest_age_seconds,
            "coalesced_count": metrics.coalesced_count,
            "dropped_count": metrics.dropped_count,
            "retried_count": metrics.retried_count,
            "dead_letter_count": metrics.dead_letter_count,
            "completed_count": completed,
            "failed_count": failed,
            "last_error": last_error,
        }

    def _worker_loop(self, index: int) -> None:
        assert self.worker_repository is not None
        worker_id = (
            f"{socket.gethostname()}:{os.getpid()}:{index + 1}:"
            f"{uuid.uuid4().hex[:8]}"
        )
        worker = InferenceWorker(
            worker_id,
            self.worker_repository,
            lease_duration=timedelta(seconds=self.lease_seconds),
        )
        next_reconcile = 0.0
        while not self._stop_event.is_set():
            now = time.monotonic()
            if index == 0 and now >= next_reconcile:
                self._reconcile_completed()
                next_reconcile = now + 60.0
            try:
                job = worker.claim()
            except Exception as exc:
                self._set_error(exc)
                self._stop_event.wait(self.poll_interval_seconds)
                continue
            if job is None:
                self._stop_event.wait(self.poll_interval_seconds)
                continue
            self._execute_job(worker, job)

    def _execute_job(self, worker: InferenceWorker, job: Any) -> None:
        renew_stop = threading.Event()
        renew_error: list[Exception] = []

        def renew_lease() -> None:
            interval = max(1.0, self.lease_seconds / 3.0)
            while not renew_stop.wait(interval):
                try:
                    worker.renew(job.id)
                except Exception as exc:
                    renew_error.append(exc)
                    return

        renew_thread = threading.Thread(
            target=renew_lease,
            name=f"eva-lease-{job.id}",
            daemon=True,
        )
        renew_thread.start()
        try:
            batch = self._load_job_batch(job)
            entry = self.manager.run_summary_batch(batch)
            if renew_error:
                raise renew_error[0]
            result = worker.complete(
                job.id,
                dict(entry),
                metadata={"kind": _SPOOL_KIND},
            )
        except Exception as exc:
            renew_stop.set()
            renew_thread.join(timeout=1.0)
            self._fail_job(worker, job, exc)
            return
        finally:
            renew_stop.set()
        renew_thread.join(timeout=1.0)

        try:
            with self._apply_lock:
                self.manager.accept_summary_entry(result.output)
                self._delete_spool(str(job.payload.get("spool_file") or ""))
            with self._state_lock:
                self._completed_count += 1
                self._last_error = None
        except Exception as exc:
            # The durable result remains in PostgreSQL and is retried by reconciliation.
            self._set_error(exc)

    def _fail_job(self, worker: InferenceWorker, job: Any, exc: Exception) -> None:
        error = self._safe_error(exc)
        try:
            failed = worker.fail(job.id, error)
            if failed.state in {JobState.DEAD_LETTER, JobState.DROPPED}:
                self._delete_spool(str(job.payload.get("spool_file") or ""))
        except InferenceQueueError:
            pass
        except Exception as fail_exc:
            self._set_error(fail_exc)
        with self._state_lock:
            self._failed_count += 1
            self._last_error = error

    def _reconcile_completed(self) -> None:
        repository = self.worker_repository
        if repository is None:
            return
        list_jobs = getattr(repository, "list_jobs", None)
        if not callable(list_jobs):
            return
        try:
            jobs = list_jobs()
        except Exception as exc:
            self._set_error(exc)
            return
        for job in jobs:
            if job.state is not JobState.SUCCEEDED:
                continue
            if str(job.payload.get("kind") or "") != _SPOOL_KIND:
                continue
            spool_name = str(job.payload.get("spool_file") or "")
            if not self._spool_path(spool_name).exists():
                continue
            try:
                result = repository.get_result(job.id)
                if result is None:
                    continue
                with self._apply_lock:
                    self.manager.accept_summary_entry(result.output)
                    self._delete_spool(spool_name)
            except Exception as exc:
                self._set_error(exc)

    def _load_job_batch(self, job: Any) -> dict[str, Any]:
        if str(job.payload.get("kind") or "") != _SPOOL_KIND:
            raise ValueError("unsupported inference job payload")
        batch = self._read_spool(str(job.payload.get("spool_file") or ""))
        channel_id = self._positive_channel_id(batch.get("channel_id"))
        if str(channel_id) != str(job.channel_id):
            raise ValueError("spooled channel does not match inference job")
        return batch

    def _write_spool(self, batch: Mapping[str, Any]) -> str:
        spool_name = f"{uuid.uuid4().hex}.json"
        target = self._spool_path(spool_name)
        temporary = target.with_suffix(".tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(temporary, flags, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    self._plain_value(batch),
                    handle,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except Exception:
            try:
                temporary.unlink()
            except OSError:
                pass
            raise
        return spool_name

    def _read_spool(self, spool_name: str) -> dict[str, Any]:
        path = self._spool_path(spool_name)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("invalid inference spool payload")
        return payload

    def _delete_spool(self, spool_name: str) -> None:
        if not spool_name:
            return
        try:
            self._spool_path(spool_name).unlink()
        except FileNotFoundError:
            pass

    def _delete_payload_spool(
        self,
        payload: Mapping[str, Any] | None,
    ) -> None:
        if payload is None:
            return
        spool_name = str(payload.get("spool_file") or "")
        if spool_name:
            self._delete_spool(spool_name)

    def _spool_path(self, spool_name: str) -> Path:
        if not _SPOOL_NAME.fullmatch(str(spool_name or "")):
            raise ValueError("invalid inference spool file")
        path = (self.spool_directory / spool_name).resolve()
        if path.parent != self.spool_directory:
            raise ValueError("inference spool path escapes configured directory")
        return path

    def _maybe_collect_spool(self) -> None:
        now = time.monotonic()
        if now - self._last_gc_at < 60.0:
            return
        self._last_gc_at = now
        referenced: set[str] = set()
        list_jobs = getattr(self.enqueue_repository, "list_jobs", None)
        if callable(list_jobs):
            try:
                for job in list_jobs():
                    if job.state in {
                        JobState.QUEUED,
                        JobState.LEASED,
                        JobState.SUCCEEDED,
                    }:
                        name = str(job.payload.get("spool_file") or "")
                        if _SPOOL_NAME.fullmatch(name):
                            referenced.add(name)
            except Exception:
                return
        cutoff = time.time() - self.spool_retention_seconds
        for path in self.spool_directory.glob("*.json"):
            try:
                if path.name not in referenced and path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                continue

    def _set_error(self, exc: Exception) -> None:
        with self._state_lock:
            self._last_error = self._safe_error(exc)

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        message = " ".join(str(exc).split())
        return f"{type(exc).__name__}: {message}"[:500]

    @staticmethod
    def _positive_channel_id(value: Any) -> int:
        try:
            channel_id = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("channel_id must be a positive integer") from exc
        if channel_id < 1:
            raise ValueError("channel_id must be a positive integer")
        return channel_id

    @classmethod
    def _plain_value(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): cls._plain_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._plain_value(item) for item in value]
        return value
