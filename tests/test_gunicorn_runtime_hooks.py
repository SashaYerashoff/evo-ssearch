import sys
import types
import unittest

import gunicorn_conf


class GunicornRuntimeHookTests(unittest.TestCase):
    def test_worker_exit_flushes_loaded_oldapp_runtime(self) -> None:
        calls = []
        manager = types.SimpleNamespace(
            persist_summary_state=lambda: calls.append("summary"),
            persist_rollup_cache=lambda: calls.append("rollup"),
        )
        previous = sys.modules.get("oldapp")
        sys.modules["oldapp"] = types.SimpleNamespace(luxriot_manager=manager)
        try:
            gunicorn_conf.worker_exit(None, None)
        finally:
            if previous is None:
                sys.modules.pop("oldapp", None)
            else:
                sys.modules["oldapp"] = previous

        self.assertEqual(calls, ["summary", "rollup"])

    def test_worker_exit_ignores_unloaded_oldapp_runtime(self) -> None:
        previous = sys.modules.pop("oldapp", None)
        try:
            gunicorn_conf.worker_exit(None, None)
        finally:
            if previous is not None:
                sys.modules["oldapp"] = previous


if __name__ == "__main__":
    unittest.main()
