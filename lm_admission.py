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


_ADMISSION_CLASS = {
    "interactive": "agent",
    "agent": "agent",
    "alert": "alert",
    "describe": "alert",
    "rollup": "rollup",
    "background": "rollup",
    "video": "live_l0",
    "vlm": "live_l0",
    "heartbeat": "live_l0",
}

# Classes are deliberately strict at admission time. Interactive work and
# urgent alerts own the protected lane. Full L0 is allowed to pre-empt
# consolidation; an L1-L3 rollup must never delay current visual attention when
# both profiles resolve to the same endpoint.
_CLASS_PRIORITY = {
    "agent": 0,
    "alert": 1,
    "live_l0": 2,
    "rollup": 3,
}
_PROTECTED_CLASSES = frozenset({"agent", "alert"})


@dataclass
class _Waiter:
    ticket_id: str
    workload: str
    admission_class: str
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
        self.wait_ms_total_by_class: Counter[str] = Counter()
        self.wait_ms_max_by_class: Dict[str, float] = {}
        self.reservation_debt = 0
        self.reservation_debt_max = 0
        self.reservation_debt_updated_at = time.monotonic()
        self.reservation_debt_ms_total = 0.0


class LMAdmissionController:
    def __init__(
        self,
        *,
        aging_seconds: float = 30.0,
        protected_slots: int = 1,
    ) -> None:
        self.aging_seconds = max(1.0, float(aging_seconds))
        if protected_slots < 0:
            raise ValueError("protected_slots cannot be negative")
        self.protected_slots = int(protected_slots)
        self._condition = threading.Condition(threading.RLock())
        self._resources: Dict[str, _ResourceState] = {}

    @staticmethod
    def _workload(value: str) -> str:
        normalized = str(value or "background").strip().lower()
        return normalized or "background"

    @staticmethod
    def _admission_class(workload: str) -> str:
        return _ADMISSION_CLASS.get(workload, "live_l0")

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

    def _effective_priority(
        self,
        waiter: _Waiter,
        now: float,
    ) -> tuple[int, int, int]:
        # Class order is strict (agent > alert > live L0 > rollup). Aging is
        # retained as a same-class diagnostic/tiebreaker without allowing a
        # background job to jump an interactive operator turn.
        base = int(_CLASS_PRIORITY.get(waiter.admission_class, 3))
        aged_steps = int(max(0.0, now - waiter.queued_at) / self.aging_seconds)
        return base, -aged_steps, waiter.sequence

    def _next_waiter(self, state: _ResourceState, now: float) -> Optional[_Waiter]:
        if not state.waiting:
            return None
        protected = [
            waiter
            for waiter in state.waiting
            if waiter.admission_class in _PROTECTED_CLASSES
        ]
        return min(
            protected or state.waiting,
            key=lambda waiter: self._effective_priority(waiter, now),
        )

    def _live_limit(self, state: _ResourceState) -> int:
        # Capacity-one endpoints cannot reserve a slot without deadlocking all
        # visual work. At capacity >= 2, keep the configured protected lane
        # physically free instead of merely moving alerts to the front of a
        # queue behind already-running long L0 requests.
        protected = min(self.protected_slots, max(0, state.capacity - 1))
        return max(1, state.capacity - protected)

    @staticmethod
    def _active_live_count(state: _ResourceState) -> int:
        return sum(
            waiter.admission_class == "live_l0"
            for waiter in state.active.values()
        )

    @staticmethod
    def _active_unprotected_count(state: _ResourceState) -> int:
        return sum(
            waiter.admission_class not in _PROTECTED_CLASSES
            for waiter in state.active.values()
        )

    @staticmethod
    def _protected_waiting_count(state: _ResourceState) -> int:
        return sum(
            waiter.admission_class in _PROTECTED_CLASSES
            for waiter in state.waiting
        )

    def _update_reservation_debt(
        self,
        state: _ResourceState,
        now: float,
    ) -> None:
        effective_now = max(now, state.reservation_debt_updated_at)
        elapsed_ms = (
            effective_now - state.reservation_debt_updated_at
        ) * 1000.0
        if state.reservation_debt:
            state.reservation_debt_ms_total += (
                elapsed_ms * state.reservation_debt
            )
        state.reservation_debt_updated_at = effective_now

        protected_waiting = self._protected_waiting_count(state)
        active_live = self._active_live_count(state)
        debt = (
            max(0, active_live - self._live_limit(state))
            if protected_waiting
            else 0
        )
        if debt > 0 and state.reservation_debt == 0:
            state.counters["reservation_debt_events_total"] += 1
        state.reservation_debt = debt
        state.reservation_debt_max = max(state.reservation_debt_max, debt)

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
                admission_class=self._admission_class(workload),
                queued_at=started,
                sequence=state.sequence,
            )
            state.sequence += 1
            state.waiting.append(waiter)
            state.counters["queued_total"] += 1
            state.counters[f"queued_{workload}"] += 1
            state.counters[f"queued_class_{waiter.admission_class}"] += 1
            self._update_reservation_debt(state, started)

            while True:
                now = time.monotonic()
                self._update_reservation_debt(state, now)
                if cancel_event is not None and cancel_event.is_set():
                    state.waiting.remove(waiter)
                    state.counters["cancelled_total"] += 1
                    self._update_reservation_debt(state, now)
                    self._condition.notify_all()
                    raise LMAdmissionCancelled("LM request was cancelled while queued")
                if deadline is not None and now >= deadline:
                    state.waiting.remove(waiter)
                    state.counters["timed_out_total"] += 1
                    self._update_reservation_debt(state, now)
                    self._condition.notify_all()
                    raise LMAdmissionTimeout("LM admission queue timeout")

                next_waiter = self._next_waiter(state, now)
                active_unprotected = self._active_unprotected_count(state)
                unprotected_limit = self._live_limit(state)
                protected_lane_available = (
                    waiter.admission_class in _PROTECTED_CLASSES
                    or active_unprotected < unprotected_limit
                )
                if (
                    len(state.active) < state.capacity
                    and next_waiter is waiter
                    and protected_lane_available
                ):
                    active_live = self._active_live_count(state)
                    live_limit = self._live_limit(state)
                    older_live_waiting = any(
                        other is not waiter
                        and other.admission_class == "live_l0"
                        and other.sequence < waiter.sequence
                        for other in state.waiting
                    )
                    if (
                        waiter.admission_class in _PROTECTED_CLASSES
                        and older_live_waiting
                    ):
                        state.counters["preemptions_total"] += 1
                        state.counters[
                            f"preemptions_{waiter.admission_class}_over_l0_total"
                        ] += 1
                    if (
                        waiter.admission_class in _PROTECTED_CLASSES
                        and active_live >= live_limit
                        and self.protected_slots > 0
                    ):
                        state.counters["reserved_slot_admissions_total"] += 1
                        state.counters[
                            f"reserved_slot_admissions_{waiter.admission_class}"
                        ] += 1
                    state.waiting.remove(waiter)
                    waiter.admitted_at = now
                    state.active[waiter.ticket_id] = waiter
                    wait_ms = max(0.0, now - waiter.queued_at) * 1000.0
                    state.wait_ms_total += wait_ms
                    state.wait_ms_max = max(state.wait_ms_max, wait_ms)
                    state.wait_ms_total_by_class[
                        waiter.admission_class
                    ] += wait_ms
                    state.wait_ms_max_by_class[waiter.admission_class] = max(
                        state.wait_ms_max_by_class.get(waiter.admission_class, 0.0),
                        wait_ms,
                    )
                    state.counters["admitted_total"] += 1
                    state.counters[f"admitted_{workload}"] += 1
                    state.counters[
                        f"admitted_class_{waiter.admission_class}"
                    ] += 1
                    self._update_reservation_debt(state, now)
                    self._condition.notify_all()
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
            self._update_reservation_debt(state, time.monotonic())
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
                self._update_reservation_debt(state, now)
                queued_by_workload = Counter(waiter.workload for waiter in state.waiting)
                active_by_workload = Counter(waiter.workload for waiter in state.active.values())
                queued_by_class = Counter(
                    waiter.admission_class for waiter in state.waiting
                )
                active_by_class = Counter(
                    waiter.admission_class for waiter in state.active.values()
                )
                oldest_age = max(
                    (max(0.0, now - waiter.queued_at) for waiter in state.waiting),
                    default=0.0,
                )
                admitted = int(state.counters.get("admitted_total") or 0)
                admitted_by_class = {
                    admission_class: int(
                        state.counters.get(
                            f"admitted_class_{admission_class}",
                            0,
                        )
                    )
                    for admission_class in _CLASS_PRIORITY
                }
                average_wait_by_class = {
                    admission_class: round(
                        state.wait_ms_total_by_class.get(
                            admission_class,
                            0.0,
                        )
                        / admitted_by_class[admission_class],
                        3,
                    )
                    if admitted_by_class[admission_class]
                    else 0.0
                    for admission_class in _CLASS_PRIORITY
                }
                live_limit = self._live_limit(state)
                active_live = self._active_live_count(state)
                protected_waiting = self._protected_waiting_count(state)
                resources.append(
                    {
                        "resource": resource,
                        "capacity": state.capacity,
                        "protected_slots": min(
                            self.protected_slots,
                            max(0, state.capacity - 1),
                        ),
                        "live_l0_limit_while_protected_waiting": live_limit,
                        "active": len(state.active),
                        "queued": len(state.waiting),
                        "active_by_workload": dict(sorted(active_by_workload.items())),
                        "queued_by_workload": dict(sorted(queued_by_workload.items())),
                        "active_by_class": dict(sorted(active_by_class.items())),
                        "queued_by_class": dict(sorted(queued_by_class.items())),
                        "oldest_queue_age_sec": round(oldest_age, 3),
                        "average_wait_ms": round(state.wait_ms_total / admitted, 3) if admitted else 0.0,
                        "max_wait_ms": round(state.wait_ms_max, 3),
                        "average_wait_ms_by_class": average_wait_by_class,
                        "max_wait_ms_by_class": {
                            admission_class: round(
                                state.wait_ms_max_by_class.get(
                                    admission_class,
                                    0.0,
                                ),
                                3,
                            )
                            for admission_class in _CLASS_PRIORITY
                        },
                        "reservation": {
                            "protected_waiting": protected_waiting,
                            "live_l0_active": active_live,
                            "borrowed_slots_active": (
                                max(
                                    0,
                                    self._active_unprotected_count(state)
                                    - live_limit,
                                )
                            ),
                            "debt_current": state.reservation_debt,
                            "debt_max": state.reservation_debt_max,
                            "debt_ms_total": round(
                                state.reservation_debt_ms_total,
                                3,
                            ),
                            "reserved_slot_admissions_total": int(
                                state.counters.get(
                                    "reserved_slot_admissions_total",
                                    0,
                                )
                            ),
                            "borrowed_slot_admissions_total": int(
                                state.counters.get(
                                    "borrowed_slot_admissions_total",
                                    0,
                                )
                            ),
                            "preemptions_total": int(
                                state.counters.get(
                                    "preemptions_total",
                                    0,
                                )
                            ),
                        },
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
