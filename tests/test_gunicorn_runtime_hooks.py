import sys
import signal
import threading
import types
import unittest
from unittest.mock import patch

import gunicorn_conf


class GunicornRuntimeHookTests(unittest.TestCase):
    def test_post_worker_init_wraps_sigterm_with_runtime_cleanup(self) -> None:
        calls = []
        previous_handler = lambda _signum, _frame: calls.append("gunicorn")
        module = types.SimpleNamespace(
            luxriot_manager=types.SimpleNamespace(
                persist_summary_state=lambda: calls.append("summary"),
                persist_rollup_cache=lambda: calls.append("rollup"),
            ),
            _shutdown_background_workers=lambda: calls.append("shutdown"),
        )
        previous_module = sys.modules.get("oldapp")
        sys.modules["oldapp"] = module
        try:
            with patch(
                "gunicorn_conf.signal.getsignal",
                return_value=previous_handler,
            ), patch("gunicorn_conf.signal.signal") as install, patch(
                "gunicorn_conf.os._exit"
            ) as force_exit:
                gunicorn_conf.post_worker_init(None)
                handler = install.call_args.args[1]
                handler(signal.SIGTERM, None)
        finally:
            if previous_module is None:
                sys.modules.pop("oldapp", None)
            else:
                sys.modules["oldapp"] = previous_module

        self.assertEqual(
            calls,
            ["gunicorn", "summary", "rollup", "shutdown"],
        )
        force_exit.assert_called_once_with(0)

    def test_bounded_shutdown_reports_a_stuck_runtime(self) -> None:
        blocker = threading.Event()
        module = types.SimpleNamespace(
            _shutdown_background_workers=lambda: blocker.wait(1.0),
        )
        previous = sys.modules.get("oldapp")
        sys.modules["oldapp"] = module
        try:
            completed = gunicorn_conf._bounded_shutdown_oldapp_runtime(
                "test",
                timeout_seconds=0.01,
            )
        finally:
            blocker.set()
            if previous is None:
                sys.modules.pop("oldapp", None)
            else:
                sys.modules["oldapp"] = previous

        self.assertFalse(completed)

    def test_worker_exit_flushes_loaded_oldapp_runtime(self) -> None:
        calls = []
        manager = types.SimpleNamespace(
            persist_summary_state=lambda: calls.append("summary"),
            persist_rollup_cache=lambda: calls.append("rollup"),
        )
        previous = sys.modules.get("oldapp")
        sys.modules["oldapp"] = types.SimpleNamespace(
            luxriot_manager=manager,
            _shutdown_background_workers=lambda: calls.append("shutdown"),
        )
        try:
            gunicorn_conf.worker_exit(None, None)
        finally:
            if previous is None:
                sys.modules.pop("oldapp", None)
            else:
                sys.modules["oldapp"] = previous

        self.assertEqual(calls, ["summary", "rollup", "shutdown"])

    def test_worker_exit_ignores_unloaded_oldapp_runtime(self) -> None:
        previous = sys.modules.pop("oldapp", None)
        try:
            gunicorn_conf.worker_exit(None, None)
        finally:
            if previous is not None:
                sys.modules["oldapp"] = previous


if __name__ == "__main__":
    unittest.main()
