"""Bounded cross-channel microbatching for live image embeddings.

The archive contract is one embedding-backed snapshot per configured cadence
for every enabled channel.  Batching is therefore an execution optimization,
not a sampling policy: every submitted image receives exactly one result or an
explicit error.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Mapping, Optional, Sequence

import numpy as np


class EmbeddingBatchError(RuntimeError):
    """Base class for explicit live embedding failures."""


class EmbeddingBatchRejected(EmbeddingBatchError):
    """Raised when the bounded queue cannot accept another snapshot."""


class EmbeddingBatchTimeout(EmbeddingBatchError):
    """Raised when a caller cannot obtain its embedding before the deadline."""


@dataclass(frozen=True)
class EmbeddingBatchOutput:
    """One batch plus immutable metadata from the same encoder generation."""

    embeddings: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class _EmbeddingRequest:
    image: Any
    submitted_at: float
    done: threading.Event = field(default_factory=threading.Event)
    result: Optional[np.ndarray] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    error: Optional[BaseException] = None
    cancelled: bool = False


class ImageEmbeddingBatcher:
    """Combine concurrent one-image calls into bounded ``embed_many`` calls."""

    def __init__(
        self,
        embed_many: Callable[[Sequence[Any]], np.ndarray],
        *,
        max_batch_size: int = 8,
        max_wait_ms: float = 75.0,
        queue_capacity: int = 128,
        request_timeout_sec: float = 15.0,
        autostart: bool = True,
        name: str = "eva-clip-batcher",
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms cannot be negative")
        if queue_capacity < max_batch_size:
            raise ValueError("queue_capacity must be at least max_batch_size")
        if request_timeout_sec <= 0:
            raise ValueError("request_timeout_sec must be positive")
        self.embed_many = embed_many
        self.max_batch_size = int(max_batch_size)
        self.max_wait_seconds = float(max_wait_ms) / 1000.0
        self.queue_capacity = int(queue_capacity)
        self.request_timeout_sec = float(request_timeout_sec)
        self.name = str(name or "eva-clip-batcher")

        self._condition = threading.Condition(threading.RLock())
        self._queue: Deque[_EmbeddingRequest] = deque()
        self._thread: Optional[threading.Thread] = None
        self._stopping = False
        self._inflight = 0
        self._counters: Counter[str] = Counter()
        self._batch_size_total = 0
        self._batch_size_max = 0
        self._last_error: Optional[str] = None
        if autostart:
            self.start()

    def start(self) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            if self._stopping:
                raise RuntimeError("embedding batcher has been stopped")
            self._thread = threading.Thread(
                target=self._worker,
                name=self.name,
                daemon=True,
            )
            self._thread.start()

    def embed_one(
        self,
        image: Any,
        *,
        timeout_sec: Optional[float] = None,
    ) -> np.ndarray:
        """Return exactly one embedding or raise an explicit bounded error."""

        result, _metadata = self.embed_one_with_metadata(
            image,
            timeout_sec=timeout_sec,
        )
        return result

    def embed_one_with_metadata(
        self,
        image: Any,
        *,
        timeout_sec: Optional[float] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Return one embedding and its batch encoder-generation metadata."""

        request = _EmbeddingRequest(image=image, submitted_at=time.monotonic())
        with self._condition:
            if self._stopping:
                raise EmbeddingBatchRejected("embedding batcher is stopping")
            if self._thread is None or not self._thread.is_alive():
                self.start()
            if len(self._queue) >= self.queue_capacity:
                self._counters["rejected_total"] += 1
                raise EmbeddingBatchRejected("embedding batch queue is full")
            self._queue.append(request)
            self._counters["submitted_total"] += 1
            self._condition.notify_all()

        timeout = (
            self.request_timeout_sec
            if timeout_sec is None
            else max(0.001, float(timeout_sec))
        )
        if not request.done.wait(timeout):
            with self._condition:
                request.cancelled = True
                self._counters["timed_out_total"] += 1
                try:
                    self._queue.remove(request)
                except ValueError:
                    pass
                self._condition.notify_all()
            raise EmbeddingBatchTimeout("embedding batch request timed out")
        if request.error is not None:
            if isinstance(request.error, EmbeddingBatchError):
                raise request.error
            raise EmbeddingBatchError(str(request.error)) from request.error
        if request.result is None:
            raise EmbeddingBatchError("embedding batch produced no result")
        return request.result, dict(request.metadata)

    def drain(self, timeout_sec: float = 15.0) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        with self._condition:
            while self._queue or self._inflight:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=min(0.1, remaining))
            return True

    def stop(self, timeout_sec: float = 5.0) -> bool:
        with self._condition:
            self._stopping = True
            pending = list(self._queue)
            self._queue.clear()
            for request in pending:
                request.error = EmbeddingBatchRejected(
                    "embedding batcher stopped before processing"
                )
                request.done.set()
            self._condition.notify_all()
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout_sec)))
        return bool(thread is None or not thread.is_alive())

    def status(self) -> dict[str, Any]:
        with self._condition:
            completed_batches = int(self._counters.get("batches_total", 0))
            return {
                "started": bool(self._thread is not None and self._thread.is_alive()),
                "stopping": bool(self._stopping),
                "queue_depth": len(self._queue),
                "queue_capacity": self.queue_capacity,
                "inflight": self._inflight,
                "max_batch_size": self.max_batch_size,
                "max_wait_ms": round(self.max_wait_seconds * 1000.0, 3),
                "average_batch_size": (
                    round(self._batch_size_total / completed_batches, 3)
                    if completed_batches
                    else 0.0
                ),
                "largest_batch_size": self._batch_size_max,
                "last_error": self._last_error,
                "counters": dict(sorted(self._counters.items())),
            }

    def _worker(self) -> None:
        while True:
            batch: list[_EmbeddingRequest] = []
            with self._condition:
                while not self._queue and not self._stopping:
                    self._condition.wait(timeout=0.5)
                if self._stopping and not self._queue:
                    return
                first = self._queue.popleft()
                if first.cancelled:
                    continue
                batch.append(first)
                deadline = first.submitted_at + self.max_wait_seconds
                while len(batch) < self.max_batch_size:
                    while self._queue and len(batch) < self.max_batch_size:
                        candidate = self._queue.popleft()
                        if not candidate.cancelled:
                            batch.append(candidate)
                    if len(batch) >= self.max_batch_size:
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or self._stopping:
                        break
                    self._condition.wait(timeout=remaining)
                self._inflight += len(batch)

            try:
                raw = self.embed_many([request.image for request in batch])
                metadata: Mapping[str, Any] = {}
                if isinstance(raw, EmbeddingBatchOutput):
                    metadata = dict(raw.metadata)
                    raw = raw.embeddings
                matrix = np.asarray(raw, dtype=np.float32)
                if matrix.ndim == 1 and len(batch) == 1:
                    matrix = matrix.reshape(1, -1)
                if matrix.ndim != 2 or matrix.shape[0] != len(batch):
                    raise EmbeddingBatchError(
                        "embed_many returned an invalid batch shape "
                        f"{tuple(matrix.shape)} for {len(batch)} requests"
                    )
                for index, request in enumerate(batch):
                    request.result = np.asarray(
                        matrix[index],
                        dtype=np.float32,
                    ).reshape(-1)
                    request.metadata = dict(metadata)
                with self._condition:
                    self._counters["completed_total"] += len(batch)
                    self._counters["batches_total"] += 1
                    self._batch_size_total += len(batch)
                    self._batch_size_max = max(self._batch_size_max, len(batch))
                    self._last_error = None
            except BaseException as exc:
                for request in batch:
                    request.error = exc
                with self._condition:
                    self._counters["failed_total"] += len(batch)
                    self._counters["failed_batches_total"] += 1
                    self._last_error = f"{type(exc).__name__}: {exc}"[:500]
            finally:
                with self._condition:
                    self._inflight = max(0, self._inflight - len(batch))
                    for request in batch:
                        request.done.set()
                    self._condition.notify_all()
