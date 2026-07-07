"""Gunicorn lifecycle hooks for EVA AI runtime durability."""

from __future__ import annotations

import logging
import sys


LOGGER = logging.getLogger(__name__)


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


def worker_exit(server, worker) -> None:
    _flush_oldapp_runtime("worker_exit")


def worker_int(worker) -> None:
    _flush_oldapp_runtime("worker_int")


def worker_abort(worker) -> None:
    _flush_oldapp_runtime("worker_abort")

