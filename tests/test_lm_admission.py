import threading
import time
import unittest

from lm_admission import (
    LMAdmissionController,
    LMAdmissionTimeout,
    normalize_lm_resource,
)


class LMAdmissionTests(unittest.TestCase):
    def _wait_for_queued(self, controller, expected, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if controller.status()["queued"] == expected:
                return
            time.sleep(0.01)
        self.fail(f"queue did not reach {expected}: {controller.status()}")

    def test_resource_key_removes_credentials_query_and_fragment(self):
        resource = normalize_lm_resource(
            "http://user:secret@127.0.0.1:1234/v1/?token=nope#fragment",
            "qwen",
        )

        self.assertEqual(resource, "http://127.0.0.1:1234/v1")
        self.assertNotIn("secret", resource)
        self.assertNotIn("token", resource)

    def test_capacity_one_serializes_requests_and_exposes_queue(self):
        controller = LMAdmissionController()
        resource = "lm|model"
        first = controller.acquire(resource, workload="vlm", capacity=1)
        admitted = threading.Event()

        def worker():
            ticket = controller.acquire(resource, workload="agent", capacity=1, timeout=2)
            admitted.set()
            controller.release(resource, ticket)

        thread = threading.Thread(target=worker)
        thread.start()
        self._wait_for_queued(controller, 1)
        self.assertFalse(admitted.is_set())

        controller.release(resource, first)
        thread.join(timeout=2)

        self.assertTrue(admitted.is_set())
        self.assertFalse(thread.is_alive())
        status = controller.status()["resources"][0]
        self.assertEqual(status["active"], 0)
        self.assertEqual(status["queued"], 0)
        self.assertEqual(status["counters"]["admitted_total"], 2)

    def test_interactive_request_precedes_newer_background_waiter(self):
        controller = LMAdmissionController(aging_seconds=60)
        resource = "lm|model"
        first = controller.acquire(resource, workload="vlm", capacity=1)
        order = []

        def worker(workload):
            ticket = controller.acquire(resource, workload=workload, capacity=1, timeout=2)
            order.append(workload)
            time.sleep(0.02)
            controller.release(resource, ticket)

        background = threading.Thread(target=worker, args=("background",))
        interactive = threading.Thread(target=worker, args=("agent",))
        background.start()
        self._wait_for_queued(controller, 1)
        interactive.start()
        self._wait_for_queued(controller, 2)

        controller.release(resource, first)
        background.join(timeout=2)
        interactive.join(timeout=2)

        self.assertEqual(order, ["agent", "background"])

    def test_queue_timeout_is_accounted_and_does_not_leak_waiter(self):
        controller = LMAdmissionController()
        resource = "lm|model"
        first = controller.acquire(resource, workload="vlm", capacity=1)

        with self.assertRaises(LMAdmissionTimeout):
            controller.acquire(
                resource,
                workload="agent",
                capacity=1,
                timeout=0.03,
            )

        controller.release(resource, first)
        status = controller.status()["resources"][0]
        self.assertEqual(status["queued"], 0)
        self.assertEqual(status["counters"]["timed_out_total"], 1)


if __name__ == "__main__":
    unittest.main()
