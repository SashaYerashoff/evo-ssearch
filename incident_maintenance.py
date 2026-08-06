"""Bounded background reconciliation for durable incident attention state.

The worker does not infer perception or risk.  It only asks the shared command
service to materialize Follow TTL expiry that is already encoded in the
incident row.  ``run_once`` is deliberately public and dependency-injected so
restart/replay behavior can be tested without starting threads or Flask.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import Any


class IncidentMaintenanceWorker:
    def __init__(
        self,
        incident_store: Any,
        service_factory: Callable[[], Any],
        *,
        interval_sec: float = 15.0,
        batch_size: int = 128,
    ) -> None:
        self.incident_store = incident_store
        self.service_factory = service_factory
        self.interval_sec = max(1.0, min(300.0, float(interval_sec)))
        self.batch_size = max(1, min(500, int(batch_size)))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._projection_offset = 0
        self._status: dict[str, Any] = {
            "running": False,
            "passes": 0,
            "records_scanned": 0,
            "records_finalized": 0,
            "projections_scanned": 0,
            "episodes_materialized": 0,
            "series_candidates_materialized": 0,
            "series_candidates_rejected": 0,
            "revision_conflicts": 0,
            "errors": 0,
            "last_error": None,
        }

    def run_once(self) -> dict[str, Any]:
        """Reconcile one bounded page of legacy ``following`` projections."""

        scanned = 0
        finalized = 0
        conflicts = 0
        errors = 0
        last_error: str | None = None
        projection_scanned = 0
        episodes_materialized = 0
        series_materialized = 0
        series_rejected = 0
        try:
            records, _total = self.incident_store.list_incidents(
                states=["following"],
                limit=self.batch_size,
                offset=0,
            )
            service = self.service_factory()
            for raw in records:
                if not isinstance(raw, Mapping):
                    continue
                scanned += 1
                try:
                    reconciled = service.reconcile_expired_follow(raw)
                    before = raw.get("follow_policy") if isinstance(raw.get("follow_policy"), Mapping) else {}
                    after = (
                        reconciled.get("follow_policy")
                        if isinstance(reconciled.get("follow_policy"), Mapping)
                        else {}
                    )
                    if before.get("active") is True and after.get("active") is False:
                        finalized += 1
                except Exception as exc:  # one incident must not block the page
                    if exc.__class__.__name__ == "IncidentRevisionConflict":
                        conflicts += 1
                    else:
                        errors += 1
                        last_error = exc.__class__.__name__

            projector = getattr(service, "ensure_temporal_projection", None)
            if callable(projector):
                with self._lock:
                    projection_offset = int(self._projection_offset)
                projection_records, projection_total = self.incident_store.list_incidents(
                    limit=self.batch_size,
                    offset=projection_offset,
                )
                for raw in projection_records:
                    if not isinstance(raw, Mapping):
                        continue
                    projection_scanned += 1
                    try:
                        result = projector(raw)
                        episodes_materialized += int(bool(result.get("episode_created")))
                        series_materialized += int(bool(result.get("relation_created")))
                        series_rejected += max(0, int(result.get("relations_rejected") or 0))
                    except Exception as exc:
                        if exc.__class__.__name__ == "IncidentRevisionConflict":
                            conflicts += 1
                        else:
                            errors += 1
                            last_error = exc.__class__.__name__
                next_offset = projection_offset + len(projection_records)
                with self._lock:
                    self._projection_offset = (
                        0 if next_offset >= int(projection_total or 0) else next_offset
                    )
        except Exception as exc:
            errors += 1
            last_error = exc.__class__.__name__

        with self._lock:
            self._status["passes"] = int(self._status["passes"]) + 1
            self._status["records_scanned"] = int(self._status["records_scanned"]) + scanned
            self._status["records_finalized"] = int(self._status["records_finalized"]) + finalized
            self._status["projections_scanned"] = int(self._status["projections_scanned"]) + projection_scanned
            self._status["episodes_materialized"] = int(self._status["episodes_materialized"]) + episodes_materialized
            self._status["series_candidates_materialized"] = int(
                self._status["series_candidates_materialized"]
            ) + series_materialized
            self._status["series_candidates_rejected"] = int(
                self._status["series_candidates_rejected"]
            ) + series_rejected
            self._status["revision_conflicts"] = int(self._status["revision_conflicts"]) + conflicts
            self._status["errors"] = int(self._status["errors"]) + errors
            self._status["last_error"] = last_error
            return dict(self._status)

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop,
                daemon=True,
                name="eva-incident-maintenance",
            )
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout)))
        with self._lock:
            self._thread = None
            self._status["running"] = False

    def status(self) -> dict[str, Any]:
        with self._lock:
            state = dict(self._status)
            state["alive"] = bool(self._thread is not None and self._thread.is_alive())
            state["interval_sec"] = self.interval_sec
            state["batch_size"] = self.batch_size
            return state

    def _loop(self) -> None:
        with self._lock:
            self._status["running"] = True
        try:
            # Reconcile immediately on process start.  This closes expired
            # process-local leases after a restart without waiting for video.
            while not self._stop.is_set():
                self.run_once()
                if self._stop.wait(self.interval_sec):
                    break
        finally:
            with self._lock:
                self._status["running"] = False


__all__ = ["IncidentMaintenanceWorker"]
