"""Gunicorn lifecycle hooks for EVA AI runtime durability."""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading


LOGGER = logging.getLogger(__name__)


def post_worker_init(worker) -> None:
    """Run EVA cleanup when Gunicorn retires a worker during graceful HUP."""

    previous = signal.getsignal(signal.SIGTERM)
    if getattr(previous, "_eva_runtime_cleanup", False):
        return

    def handle_sigterm(signum, frame) -> None:
        if callable(previous):
            previous(signum, frame)
        _flush_oldapp_runtime("worker_sigterm")
        completed = _bounded_shutdown_oldapp_runtime(
            "worker_sigterm",
            timeout_seconds=8.0,
        )
        if not completed:
            LOGGER.warning(
                "Gunicorn worker runtime cleanup exceeded its retirement deadline; "
                "forcing the superseded worker to exit"
            )
        # ThreadPoolExecutor registers an interpreter-exit join before EVA's
        # atexit hook.  Even after Gunicorn marks this worker dead, one running
        # SigLIP future can otherwise keep the superseded process alive and let
        # two complete EVA runtimes own the same streams.  State was flushed
        # above and cleanup received a bounded chance; retire this worker
        # without entering Python's unbounded thread-join path.
        os._exit(0)

    handle_sigterm._eva_runtime_cleanup = True  # type: ignore[attr-defined]
    signal.signal(signal.SIGTERM, handle_sigterm)


def _flush_oldapp_runtime(reason: str) -> None:
    module = sys.modules.get("oldapp")
    manager = getattr(module, "luxriot_manager", None) if module is not None else None
    if manager is None:
        return
    for method_name in ("persist_summary_state", "persist_rollup_cache"):
        method = getattr(manager, method_name, None)
        if not callable(method):
            continue
        try:
            method()
        except Exception:
            LOGGER.exception("Gunicorn %s hook failed during %s", reason, method_name)


def _shutdown_oldapp_runtime(reason: str) -> None:
    """Stop non-daemon capture/executor threads before Python waits for them.

    Gunicorn's graceful HUP asks the worker to exit, but Python will not finish
    while ThreadPoolExecutor workers are still serving Luxriot/SigLIP jobs.
    ``atexit`` runs too late to break that dependency. Invoke the same
    idempotent cleanup explicitly from Gunicorn's in-worker lifecycle hooks.
    """

    module = sys.modules.get("oldapp")
    shutdown = (
        getattr(module, "_shutdown_background_workers", None)
        if module is not None
        else None
    )
    if not callable(shutdown):
        return
    try:
        shutdown()
    except Exception:
        LOGGER.exception("Gunicorn %s hook failed during runtime shutdown", reason)


def _bounded_shutdown_oldapp_runtime(
    reason: str,
    *,
    timeout_seconds: float,
) -> bool:
    """Give EVA cleanup a bounded window during worker retirement."""

    completed = threading.Event()

    def run() -> None:
        try:
            _shutdown_oldapp_runtime(reason)
        finally:
            completed.set()

    thread = threading.Thread(
        target=run,
        daemon=True,
        name="eva-gunicorn-worker-shutdown",
    )
    thread.start()
    return completed.wait(timeout=max(0.0, float(timeout_seconds)))


def worker_exit(server, worker) -> None:
    _flush_oldapp_runtime("worker_exit")
    _shutdown_oldapp_runtime("worker_exit")


def worker_int(worker) -> None:
    _flush_oldapp_runtime("worker_int")
    _shutdown_oldapp_runtime("worker_int")


def worker_abort(worker) -> None:
    _flush_oldapp_runtime("worker_abort")
    _shutdown_oldapp_runtime("worker_abort")
