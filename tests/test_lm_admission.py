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

    def test_live_l0_keeps_fourth_slot_reserved_for_protected_work(self):
        controller = LMAdmissionController(protected_slots=1)
        resource = "shared-gpu"
        tickets = [
            controller.acquire(
                resource,
                workload="vlm",
                capacity=4,
                timeout=0.2,
            )
            for _index in range(3)
        ]
        fourth_admitted = threading.Event()
        release_fourth = threading.Event()

        def fourth_live():
            ticket = controller.acquire(resource, workload="vlm", capacity=4, timeout=2)
            fourth_admitted.set()
            release_fourth.wait(timeout=2)
            controller.release(resource, ticket)

        thread = threading.Thread(target=fourth_live)
        thread.start()
        self._wait_for_queued(controller, 1)
        try:
            status = controller.status()["resources"][0]
            self.assertEqual(status["active"], 3)
            self.assertEqual(status["active_by_class"], {"live_l0": 3})
            self.assertEqual(
                status["live_l0_limit_while_protected_waiting"],
                3,
            )
            self.assertEqual(status["reservation"]["borrowed_slots_active"], 0)
            self.assertFalse(fourth_admitted.is_set())
            alert = controller.acquire(resource, workload="alert", capacity=4, timeout=1)
            controller.release(resource, alert)
            self.assertFalse(fourth_admitted.is_set())
            controller.release(resource, tickets.pop())
            self.assertTrue(fourth_admitted.wait(timeout=1))
            self.assertEqual(status["reservation"]["debt_current"], 0)
        finally:
            release_fourth.set()
            thread.join(timeout=2)
            for ticket in tickets:
                controller.release(resource, ticket)

    def test_interactive_agent_uses_reserved_slot_before_older_rollup(self):
        controller = LMAdmissionController(protected_slots=1)
        resource = "shared-gpu"
        live_tickets = [
            controller.acquire(resource, workload="heartbeat", capacity=4)
            for _index in range(3)
        ]
        order = []
        agent_admitted = threading.Event()
        rollup_admitted = threading.Event()
        release_agent = threading.Event()
        release_rollup = threading.Event()

        def contender(workload, admitted, release_gate):
            ticket = controller.acquire(
                resource,
                workload=workload,
                capacity=4,
                timeout=2,
            )
            order.append(workload)
            admitted.set()
            release_gate.wait(timeout=2)
            controller.release(resource, ticket)

        rollup_thread = threading.Thread(
            target=contender,
            args=("rollup", rollup_admitted, release_rollup),
        )
        agent_thread = threading.Thread(
            target=contender,
            args=("agent", agent_admitted, release_agent),
        )
        rollup_thread.start()
        self._wait_for_queued(controller, 1)
        agent_thread.start()

        try:
            self.assertTrue(agent_admitted.wait(timeout=1))
            self.assertFalse(rollup_admitted.is_set())
            self.assertEqual(order, ["agent"])

            release_agent.set()
            time.sleep(0.02)
            self.assertFalse(rollup_admitted.is_set())
            controller.release(resource, live_tickets.pop())
            self.assertTrue(rollup_admitted.wait(timeout=1))
            self.assertEqual(order, ["agent", "rollup"])

            status = controller.status()["resources"][0]
            self.assertGreater(
                status["average_wait_ms_by_class"]["agent"],
                0.0,
            )
            self.assertGreater(
                status["average_wait_ms_by_class"]["rollup"],
                0.0,
            )
            self.assertEqual(
                status["reservation"]["reserved_slot_admissions_total"],
                1,
            )
            self.assertEqual(
                status["counters"].get("reserved_slot_admissions_rollup", 0),
                0,
            )
        finally:
            release_agent.set()
            release_rollup.set()
            agent_thread.join(timeout=2)
            rollup_thread.join(timeout=2)
            for ticket in live_tickets:
                controller.release(resource, ticket)
        self.assertFalse(agent_thread.is_alive())
        self.assertFalse(rollup_thread.is_alive())

    def test_alert_preempts_older_ordinary_l0_waiter(self):
        controller = LMAdmissionController(protected_slots=1)
        resource = "shared-gpu"
        blocker = controller.acquire(
            resource,
            workload="vlm",
            capacity=1,
        )
        order = []
        alert_admitted = threading.Event()
        l0_admitted = threading.Event()
        release_alert = threading.Event()

        def ordinary_l0():
            ticket = controller.acquire(
                resource,
                workload="heartbeat",
                capacity=1,
                timeout=2,
            )
            order.append("l0")
            l0_admitted.set()
            controller.release(resource, ticket)

        def alert():
            ticket = controller.acquire(
                resource,
                workload="alert",
                capacity=1,
                timeout=2,
            )
            order.append("alert")
            alert_admitted.set()
            release_alert.wait(timeout=2)
            controller.release(resource, ticket)

        l0_thread = threading.Thread(target=ordinary_l0)
        alert_thread = threading.Thread(target=alert)
        l0_thread.start()
        self._wait_for_queued(controller, 1)
        alert_thread.start()
        self._wait_for_queued(controller, 2)

        try:
            controller.release(resource, blocker)
            self.assertTrue(alert_admitted.wait(timeout=1))
            self.assertFalse(l0_admitted.is_set())
            self.assertEqual(order, ["alert"])
            status = controller.status()["resources"][0]
            self.assertEqual(
                status["counters"]["preemptions_alert_over_l0_total"],
                1,
            )
            self.assertEqual(
                status["reservation"]["preemptions_total"],
                1,
            )
        finally:
            release_alert.set()
            alert_thread.join(timeout=2)
            l0_thread.join(timeout=2)
        self.assertEqual(order, ["alert", "l0"])
        self.assertFalse(alert_thread.is_alive())
        self.assertFalse(l0_thread.is_alive())

    def test_strict_reservation_admits_agent_without_debt(self):
        controller = LMAdmissionController(protected_slots=1)
        resource = "shared-gpu"
        live_tickets = [
            controller.acquire(resource, workload="vlm", capacity=4)
            for _index in range(3)
        ]
        agent_admitted = threading.Event()
        release_agent = threading.Event()

        def agent():
            ticket = controller.acquire(
                resource,
                workload="interactive",
                capacity=4,
                timeout=2,
            )
            agent_admitted.set()
            release_agent.wait(timeout=2)
            controller.release(resource, ticket)

        thread = threading.Thread(target=agent)
        thread.start()
        self.assertTrue(agent_admitted.wait(timeout=1))
        admitted_status = controller.status()["resources"][0]
        self.assertEqual(
            admitted_status["active_by_class"],
            {"agent": 1, "live_l0": 3},
        )
        self.assertEqual(admitted_status["reservation"]["debt_current"], 0)
        self.assertEqual(admitted_status["reservation"]["debt_max"], 0)

        try:
            pass
        finally:
            release_agent.set()
            thread.join(timeout=2)
            for ticket in live_tickets:
                controller.release(resource, ticket)
        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
