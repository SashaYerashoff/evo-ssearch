"""Gunicorn lifecycle hooks for EVA AI runtime durability."""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time


LOGGER = logging.getLogger(__name__)
_EVA_RELOAD_HANDOVER_ACTIVE = False
_EVA_RELOAD_BASELINE_WORKER_AGE = 0


def on_reload(server) -> None:
    """Keep the serving worker until the replacement finishes cold startup.

    EVA production runs one worker because capture schedulers and in-process
    queues are deliberately single-owner.  Gunicorn normally spawns the new
    worker and immediately SIGTERMs the old one, long before transformers and
    SigLIP have loaded.  Temporarily target two workers; the replacement asks
    the master to retire the old process from ``post_worker_init`` after its
    non-capture runtime is ready.
    """

    configured_workers = max(1, int(getattr(server.cfg, "workers", 1) or 1))
    if configured_workers != 1:
        LOGGER.warning(
            "EVA readiness handover is only enabled for the supported "
            "single-worker runtime; using Gunicorn's normal reload"
        )
        return
    global _EVA_RELOAD_HANDOVER_ACTIVE, _EVA_RELOAD_BASELINE_WORKER_AGE
    _EVA_RELOAD_HANDOVER_ACTIVE = True
    _EVA_RELOAD_BASELINE_WORKER_AGE = int(
        getattr(server, "worker_age", 0) or 0
    )
    server.num_workers = 2
    LOGGER.info(
        "EVA reload handover retaining the serving worker while the "
        "replacement warms"
    )


def child_exit(server, worker) -> None:
    """Close the temporary handover target after either worker exits.

    A failed warmed candidate exits before it can request TTOU. Without this
    master-side correction Gunicorn would keep target=2 and immediately start
    another 137-second cold candidate beside the healthy serving worker.
    """

    global _EVA_RELOAD_HANDOVER_ACTIVE
    if not _EVA_RELOAD_HANDOVER_ACTIVE:
        return
    worker_age = int(getattr(worker, "age", 0) or 0)
    if worker_age > _EVA_RELOAD_BASELINE_WORKER_AGE:
        # The replacement died. Keep the old worker and stop retrying this HUP.
        server.num_workers = 1
        LOGGER.error(
            "EVA replacement worker exited before handover; retaining the "
            "previous serving worker"
        )
    elif int(getattr(server, "num_workers", 1) or 1) > 1:
        # The serving worker disappeared before the replacement was ready.
        # Avoid spawning a third process; the candidate will notice that it is
        # now alone and acquire deferred runtime ownership after warm-up.
        server.num_workers = 1
        LOGGER.warning(
            "EVA serving worker exited during replacement warm-up; the "
            "replacement will restore runtime ownership when ready"
        )
    elif int(getattr(server, "num_workers", 1) or 1) <= 1:
        # The expected old worker retired after the warmed candidate's TTOU.
        LOGGER.info("EVA previous worker retired after successful handover")
    _EVA_RELOAD_HANDOVER_ACTIVE = False


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
    _start_ready_handover(worker)


def _worker_sibling_pids(worker) -> list[int]:
    parent_pid = int(getattr(worker, "ppid", 0) or os.getppid())
    own_pid = int(getattr(worker, "pid", 0) or os.getpid())
    try:
        raw = open(
            f"/proc/{parent_pid}/task/{parent_pid}/children",
            "r",
            encoding="utf-8",
        ).read()
    except OSError:
        return []
    siblings = []
    for token in raw.split():
        try:
            pid = int(token)
        except ValueError:
            continue
        if pid > 0 and pid != own_pid:
            siblings.append(pid)
    return siblings


def _start_ready_handover(worker) -> None:
    """Retire the previous worker only after this worker's cold warmup.

    ``oldapp`` deliberately defers camera restore when it sees a sibling under
    the same Gunicorn master.  The handover thread lets this worker enter the
    HTTP run loop, reduces the temporary worker count, waits until the old
    capture owner is gone, and then restores cameras in this process.
    """

    module = sys.modules.get("oldapp")
    siblings = _worker_sibling_pids(worker)
    if not siblings:
        pending = (
            getattr(module, "runtime_handover_pending", None)
            if module is not None
            else None
        )
        if callable(pending) and bool(pending()):
            complete = getattr(module, "complete_runtime_handover", None)
            if not callable(complete):
                LOGGER.error("EVA runtime handover completion hook is unavailable")
                return
            try:
                result = complete()
            except Exception:
                LOGGER.exception("EVA orphaned replacement ownership restore failed")
                return
            LOGGER.warning(
                "EVA replacement acquired runtime after the previous worker "
                "exited during warm-up status=%s",
                (result or {}).get("status") if isinstance(result, dict) else result,
            )
        return
    pending = (
        getattr(module, "runtime_handover_pending", None)
        if module is not None
        else None
    )
    candidate_ready = getattr(module, "runtime_handover_candidate_ready", None)
    if (
        not callable(pending)
        or not bool(pending())
        or not callable(candidate_ready)
        or not bool(candidate_ready())
    ):
        LOGGER.error(
            "Replacement worker failed cold startup; preserving the serving "
            "worker and rejecting this replacement"
        )
        os._exit(1)
        return

    parent_pid = int(getattr(worker, "ppid", 0) or os.getppid())

    def handover() -> None:
        # Return from post_worker_init first so this warmed worker can accept
        # HTTP while the previous capture owner performs bounded cleanup.
        time.sleep(0.25)
        siblings = _worker_sibling_pids(worker)
        if not siblings:
            LOGGER.warning(
                "EVA handover found no serving sibling; restoring runtime "
                "without worker-count transition"
            )
        else:
            LOGGER.info(
                "EVA replacement warmed; retiring previous worker pid=%s",
                ",".join(str(pid) for pid in siblings),
            )
            try:
                os.kill(parent_pid, signal.SIGTTOU)
            except OSError:
                LOGGER.exception("Failed to request Gunicorn worker handover")
                return

        deadline = time.monotonic() + 30.0
        while _worker_sibling_pids(worker) and time.monotonic() < deadline:
            time.sleep(0.1)
        if _worker_sibling_pids(worker):
            LOGGER.error(
                "Previous EVA worker did not retire; capture restore remains "
                "blocked to prevent duplicate ownership"
            )
            return
        complete = getattr(module, "complete_runtime_handover", None)
        if not callable(complete):
            LOGGER.error("EVA runtime handover completion hook is unavailable")
            return
        try:
            result = complete()
        except Exception:
            LOGGER.exception("EVA runtime handover completion failed")
            return
        LOGGER.info(
            "EVA runtime handover completed status=%s",
            (result or {}).get("status") if isinstance(result, dict) else result,
        )

    threading.Thread(
        target=handover,
        daemon=True,
        name="eva-gunicorn-ready-handover",
    ).start()


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
