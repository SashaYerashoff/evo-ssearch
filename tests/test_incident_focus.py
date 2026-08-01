import threading
import unittest

from incident_focus import (
    FocusLeaseCapacityError,
    FocusLevel,
    IncidentFocusLeaseManager,
)


class IncidentFocusLeaseManagerTests(unittest.TestCase):
    def test_strongest_channel_directive_expiry_refresh_and_stop(self):
        now = [1_000]
        manager = IncidentFocusLeaseManager(clock_ms=lambda: now[0])

        first = manager.start(
            "incident-follow",
            [7, 8, 7],
            level="follow",
            ttl_seconds=5,
            context='{"summary":"person left the desk"}',
        )
        manager.start(
            "incident-critical",
            [8],
            level="critical",
            ttl_seconds=2,
        )

        self.assertEqual(first.channel_ids, (7, 8))
        self.assertIn("person left the desk", first.context)
        self.assertEqual(
            manager.directive_for_channel(8).level,
            FocusLevel.CRITICAL,
        )
        now[0] = 3_001
        self.assertEqual(
            manager.directive_for_channel(8).level,
            FocusLevel.FOLLOW,
        )

        refreshed = manager.start(
            "incident-follow",
            [9],
            level=FocusLevel.FOLLOW,
            ttl_seconds=4,
        )
        self.assertEqual(refreshed.created_at_ms, 1_000)
        self.assertEqual(refreshed.context, first.context)
        self.assertIsNone(manager.directive_for_channel(7))
        self.assertTrue(manager.stop("incident-follow"))
        self.assertFalse(manager.stop("incident-follow"))
        self.assertEqual(manager.compact_digest()["active"], 0)

    def test_capacity_is_bounded_and_expired_slot_is_reused(self):
        now = [0]
        manager = IncidentFocusLeaseManager(
            max_leases=1,
            clock_ms=lambda: now[0],
        )
        manager.start("one", [1], ttl_seconds=1)
        with self.assertRaises(FocusLeaseCapacityError):
            manager.start("two", [2], ttl_seconds=1)

        now[0] = 1_000
        manager.start("two", [2], ttl_seconds=1)
        self.assertEqual(manager.compact_digest()["active"], 1)

    def test_concurrent_refreshes_keep_one_incident_record(self):
        manager = IncidentFocusLeaseManager(max_leases=4)
        failures = []

        def refresh(channel_id):
            try:
                manager.start("shared", [channel_id], ttl_seconds=30)
            except Exception as exc:  # pragma: no cover - asserted below
                failures.append(exc)

        threads = [threading.Thread(target=refresh, args=(item,)) for item in range(1, 9)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(failures, [])
        self.assertEqual(manager.compact_digest()["active"], 1)


if __name__ == "__main__":
    unittest.main()
