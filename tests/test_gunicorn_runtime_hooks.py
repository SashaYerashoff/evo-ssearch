import sys
import signal
import threading
import types
import unittest
from unittest.mock import patch

import gunicorn_conf


class GunicornRuntimeHookTests(unittest.TestCase):
    def test_reload_temporarily_keeps_old_and_new_single_workers(self) -> None:
        server = types.SimpleNamespace(
            cfg=types.SimpleNamespace(workers=1),
            num_workers=1,
            worker_age=9,
        )
        original_active = gunicorn_conf._EVA_RELOAD_HANDOVER_ACTIVE
        original_age = gunicorn_conf._EVA_RELOAD_BASELINE_WORKER_AGE
        try:
            gunicorn_conf.on_reload(server)

            self.assertEqual(server.num_workers, 2)
            self.assertTrue(gunicorn_conf._EVA_RELOAD_HANDOVER_ACTIVE)
            self.assertEqual(
                gunicorn_conf._EVA_RELOAD_BASELINE_WORKER_AGE,
                9,
            )
        finally:
            gunicorn_conf._EVA_RELOAD_HANDOVER_ACTIVE = original_active
            gunicorn_conf._EVA_RELOAD_BASELINE_WORKER_AGE = original_age

    def test_failed_replacement_resets_temporary_worker_target(self) -> None:
        server = types.SimpleNamespace(num_workers=2)
        worker = types.SimpleNamespace(age=10)
        original_active = gunicorn_conf._EVA_RELOAD_HANDOVER_ACTIVE
        original_age = gunicorn_conf._EVA_RELOAD_BASELINE_WORKER_AGE
        try:
            gunicorn_conf._EVA_RELOAD_HANDOVER_ACTIVE = True
            gunicorn_conf._EVA_RELOAD_BASELINE_WORKER_AGE = 9

            gunicorn_conf.child_exit(server, worker)

            self.assertEqual(server.num_workers, 1)
            self.assertFalse(gunicorn_conf._EVA_RELOAD_HANDOVER_ACTIVE)
        finally:
            gunicorn_conf._EVA_RELOAD_HANDOVER_ACTIVE = original_active
            gunicorn_conf._EVA_RELOAD_BASELINE_WORKER_AGE = original_age

    def test_serving_worker_exit_during_warmup_keeps_only_candidate(self) -> None:
        server = types.SimpleNamespace(num_workers=2)
        worker = types.SimpleNamespace(age=9)
        original_active = gunicorn_conf._EVA_RELOAD_HANDOVER_ACTIVE
        original_age = gunicorn_conf._EVA_RELOAD_BASELINE_WORKER_AGE
        try:
            gunicorn_conf._EVA_RELOAD_HANDOVER_ACTIVE = True
            gunicorn_conf._EVA_RELOAD_BASELINE_WORKER_AGE = 9

            gunicorn_conf.child_exit(server, worker)

            self.assertEqual(server.num_workers, 1)
            self.assertFalse(gunicorn_conf._EVA_RELOAD_HANDOVER_ACTIVE)
        finally:
            gunicorn_conf._EVA_RELOAD_HANDOVER_ACTIVE = original_active
            gunicorn_conf._EVA_RELOAD_BASELINE_WORKER_AGE = original_age

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

    def test_warmed_replacement_retires_old_worker_then_restores_runtime(self) -> None:
        calls = []
        module = types.SimpleNamespace(
            runtime_handover_pending=lambda: True,
            runtime_handover_candidate_ready=lambda: True,
            complete_runtime_handover=lambda: calls.append("restore")
            or {"status": "restored"},
        )
        worker = types.SimpleNamespace(ppid=100, pid=200)
        sibling_reads = iter(([150], [150], [], []))

        class ImmediateThread:
            def __init__(self, *, target, **_kwargs):
                self.target = target

            def start(self):
                self.target()

        previous = sys.modules.get("oldapp")
        sys.modules["oldapp"] = module
        try:
            with (
                patch(
                    "gunicorn_conf._worker_sibling_pids",
                    side_effect=lambda _worker: list(next(sibling_reads)),
                ),
                patch("gunicorn_conf.threading.Thread", ImmediateThread),
                patch("gunicorn_conf.time.sleep"),
                patch("gunicorn_conf.os.kill") as kill,
            ):
                gunicorn_conf._start_ready_handover(worker)
        finally:
            if previous is None:
                sys.modules.pop("oldapp", None)
            else:
                sys.modules["oldapp"] = previous

        kill.assert_called_once_with(100, signal.SIGTTOU)
        self.assertEqual(calls, ["restore"])

    def test_orphaned_warmed_replacement_acquires_runtime_without_signal(self) -> None:
        calls = []
        module = types.SimpleNamespace(
            runtime_handover_pending=lambda: True,
            complete_runtime_handover=lambda: calls.append("restore")
            or {"status": "restored"},
        )
        worker = types.SimpleNamespace(ppid=100, pid=200)
        previous = sys.modules.get("oldapp")
        sys.modules["oldapp"] = module
        try:
            with (
                patch("gunicorn_conf._worker_sibling_pids", return_value=[]),
                patch("gunicorn_conf.os.kill") as kill,
            ):
                gunicorn_conf._start_ready_handover(worker)
        finally:
            if previous is None:
                sys.modules.pop("oldapp", None)
            else:
                sys.modules["oldapp"] = previous

        self.assertEqual(calls, ["restore"])
        kill.assert_not_called()

    def test_failed_replacement_never_retires_healthy_worker(self) -> None:
        module = types.SimpleNamespace(
            runtime_handover_pending=lambda: False,
            runtime_handover_candidate_ready=lambda: False,
        )
        worker = types.SimpleNamespace(ppid=100, pid=200)
        previous = sys.modules.get("oldapp")
        sys.modules["oldapp"] = module
        try:
            with (
                patch("gunicorn_conf._worker_sibling_pids", return_value=[150]),
                patch("gunicorn_conf.os._exit") as force_exit,
                patch("gunicorn_conf.os.kill") as kill,
            ):
                gunicorn_conf._start_ready_handover(worker)
        finally:
            if previous is None:
                sys.modules.pop("oldapp", None)
            else:
                sys.modules["oldapp"] = previous

        force_exit.assert_called_once_with(1)
        kill.assert_not_called()

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
