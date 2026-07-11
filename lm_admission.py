"""In-process admission control for shared OpenAI-compatible LM endpoints.

EVA deliberately runs one application worker.  This controller makes contention
between interactive agent turns, live VLM batches, and rollups explicit instead
of letting every thread attack a capacity-one llama.cpp server independently.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import urlsplit, urlunsplit


class LMAdmissionTimeout(TimeoutError):
    pass


class LMAdmissionCancelled(RuntimeError):
    pass


def normalize_lm_resource(base_url: str, model: str) -> str:
    """Return a credential-free key for one serving endpoint.

    The model is intentionally not part of the key: LM Studio/llama.cpp may expose
    several model IDs while all of them still contend for the same process/GPU.
    Deployments that need independent concurrency should use separate endpoints.
    """

    raw = str(base_url or "").strip().rstrip("/")
    parsed = urlsplit(raw)
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port is not None else ""
    netloc = f"{host}{port}"
    clean_url = urlunsplit((parsed.scheme.lower(), netloc, parsed.path.rstrip("/"), "", ""))
    return clean_url


def configured_lm_capacity(profile_id: Optional[str] = None, default: int = 1) -> int:
    normalized = "".join(
        char if char.isalnum() else "_"
        for char in str(profile_id or "").strip().upper()
    ).strip("_")
    names = []
    if normalized:
        names.append(f"EVOSSEARCH_LM_PROFILE_{normalized}_MAX_INFLIGHT")
    names.append("EVOSSEARCH_LM_MAX_INFLIGHT")
    raw: Any = None
    for name in names:
        if str(os.getenv(name) or "").strip():
            raw = os.getenv(name)
            break
    try:
        value = int(raw) if raw is not None else int(default)
    except (TypeError, ValueError):
        value = int(default)
    return max(1, min(64, value))


_PRIORITY = {
    "interactive": 0,
    "agent": 0,
    "alert": 1,
    "describe": 1,
    "video": 2,
    "vlm": 2,
    "heartbeat": 2,
    "rollup": 3,
    "background": 3,
}


@dataclass
class _Waiter:
    ticket_id: str
    workload: str
    queued_at: float
    sequence: int
    admitted_at: Optional[float] = None


class _ResourceState:
    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self.active: Dict[str, _Waiter] = {}
        self.waiting: deque[_Waiter] = deque()
        self.sequence = 0
        self.counters: Counter[str] = Counter()
        self.wait_ms_total = 0.0
        self.wait_ms_max = 0.0


class LMAdmissionController:
    def __init__(self, *, aging_seconds: float = 30.0) -> None:
        self.aging_seconds = max(1.0, float(aging_seconds))
        self._condition = threading.Condition(threading.RLock())
        self._resources: Dict[str, _ResourceState] = {}

    @staticmethod
    def _workload(value: str) -> str:
        normalized = str(value or "background").strip().lower()
        return normalized or "background"

    def _state(self, resource: str, capacity: int) -> _ResourceState:
        state = self._resources.get(resource)
        if state is None:
            state = _ResourceState(capacity)
            self._resources[resource] = state
        else:
            # A resource shared by differently named profiles must obey the most
            # conservative configured capacity.
            state.capacity = min(state.capacity, max(1, int(capacity)))
        return state

    def _effective_priority(self, waiter: _Waiter, now: float) -> tuple[int, int]:
        base = int(_PRIORITY.get(waiter.workload, 2))
        aged_steps = int(max(0.0, now - waiter.queued_at) / self.aging_seconds)
        return max(0, base - aged_steps), waiter.sequence

    def _next_waiter(self, state: _ResourceState, now: float) -> Optional[_Waiter]:
        if not state.waiting:
            return None
        return min(
            state.waiting,
            key=lambda waiter: self._effective_priority(waiter, now),
        )

    def acquire(
        self,
        resource: str,
        *,
        workload: str,
        capacity: int = 1,
        timeout: Optional[float] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        if not str(resource or "").strip():
            raise ValueError("LM resource key is required")
        workload = self._workload(workload)
        started = time.monotonic()
        deadline = None if timeout is None else started + max(0.0, float(timeout))
        with self._condition:
            state = self._state(resource, capacity)
            waiter = _Waiter(
                ticket_id=str(uuid.uuid4()),
                workload=workload,
                queued_at=started,
                sequence=state.sequence,
            )
            state.sequence += 1
            state.waiting.append(waiter)
            state.counters["queued_total"] += 1
            state.counters[f"queued_{workload}"] += 1

            while True:
                now = time.monotonic()
                if cancel_event is not None and cancel_event.is_set():
                    state.waiting.remove(waiter)
                    state.counters["cancelled_total"] += 1
                    self._condition.notify_all()
                    raise LMAdmissionCancelled("LM request was cancelled while queued")
                if deadline is not None and now >= deadline:
                    state.waiting.remove(waiter)
                    state.counters["timed_out_total"] += 1
                    self._condition.notify_all()
                    raise LMAdmissionTimeout("LM admission queue timeout")

                next_waiter = self._next_waiter(state, now)
                if len(state.active) < state.capacity and next_waiter is waiter:
                    state.waiting.remove(waiter)
                    waiter.admitted_at = now
                    state.active[waiter.ticket_id] = waiter
                    wait_ms = max(0.0, now - waiter.queued_at) * 1000.0
                    state.wait_ms_total += wait_ms
                    state.wait_ms_max = max(state.wait_ms_max, wait_ms)
                    state.counters["admitted_total"] += 1
                    state.counters[f"admitted_{workload}"] += 1
                    return waiter.ticket_id

                remaining = None if deadline is None else max(0.0, deadline - now)
                wait_for = 0.25 if remaining is None else min(0.25, remaining)
                self._condition.wait(timeout=wait_for)

    def release(self, resource: str, ticket_id: str, *, outcome: str = "completed") -> None:
        with self._condition:
            state = self._resources.get(resource)
            if state is None or ticket_id not in state.active:
                return
            waiter = state.active.pop(ticket_id)
            normalized = str(outcome or "completed").strip().lower() or "completed"
            state.counters[f"{normalized}_total"] += 1
            state.counters[f"{normalized}_{waiter.workload}"] += 1
            self._condition.notify_all()

    @contextmanager
    def admission(
        self,
        resource: str,
        *,
        workload: str,
        capacity: int = 1,
        timeout: Optional[float] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        ticket = self.acquire(
            resource,
            workload=workload,
            capacity=capacity,
            timeout=timeout,
            cancel_event=cancel_event,
        )
        outcome = "completed"
        try:
            yield ticket
        except Exception:
            outcome = "failed"
            raise
        finally:
            self.release(resource, ticket, outcome=outcome)

    def status(self) -> Dict[str, Any]:
        now = time.monotonic()
        with self._condition:
            resources: List[Dict[str, Any]] = []
            for resource, state in sorted(self._resources.items()):
                queued_by_workload = Counter(waiter.workload for waiter in state.waiting)
                active_by_workload = Counter(waiter.workload for waiter in state.active.values())
                oldest_age = max(
                    (max(0.0, now - waiter.queued_at) for waiter in state.waiting),
                    default=0.0,
                )
                admitted = int(state.counters.get("admitted_total") or 0)
                resources.append(
                    {
                        "resource": resource,
                        "capacity": state.capacity,
                        "active": len(state.active),
                        "queued": len(state.waiting),
                        "active_by_workload": dict(sorted(active_by_workload.items())),
                        "queued_by_workload": dict(sorted(queued_by_workload.items())),
                        "oldest_queue_age_sec": round(oldest_age, 3),
                        "average_wait_ms": round(state.wait_ms_total / admitted, 3) if admitted else 0.0,
                        "max_wait_ms": round(state.wait_ms_max, 3),
                        "counters": dict(sorted(state.counters.items())),
                    }
                )
            return {
                "resource_count": len(resources),
                "active": sum(row["active"] for row in resources),
                "queued": sum(row["queued"] for row in resources),
                "resources": resources,
            }


_DEFAULT_CONTROLLER = LMAdmissionController()


def get_lm_admission_controller() -> LMAdmissionController:
    return _DEFAULT_CONTROLLER
