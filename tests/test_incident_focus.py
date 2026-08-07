import json
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
            context='{"summary":"fire alarm still active"}',
        )

        self.assertEqual(first.channel_ids, (7, 8))
        self.assertIn("person left the desk", first.context)
        directive = manager.directive_for_channel(8)
        self.assertEqual(directive.level, FocusLevel.CRITICAL)
        self.assertEqual(
            directive.incident_ids,
            ("incident-critical", "incident-follow"),
        )
        self.assertEqual(directive.foreground_incident_ids, directive.incident_ids)
        self.assertEqual(len(directive.contexts), 2)
        self.assertTrue(any("fire alarm" in item for item in directive.contexts))
        self.assertTrue(any("left the desk" in item for item in directive.contexts))
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

    def test_attention_overflow_is_parked_without_resolving_leases(self):
        manager = IncidentFocusLeaseManager(clock_ms=lambda: 1_000)
        for index in range(10):
            manager.start(
                f"incident-{index:02d}",
                [1],
                ttl_seconds=30,
                context=f'{{"summary":"incident {index}"}}',
            )

        directive = manager.directive_for_channel(1)

        self.assertEqual(len(directive.incident_ids), 10)
        self.assertEqual(len(directive.foreground_incident_ids), 4)
        self.assertEqual(len(directive.hot_incident_ids), 8)
        self.assertEqual(len(directive.parked_incident_ids), 2)
        self.assertEqual(len(directive.contexts), 10)
        for incident_id in directive.parked_incident_ids:
            lease = manager.get(incident_id)
            self.assertIsNotNone(lease)
            self.assertTrue(lease.unresolved)
        parked_decisions = [
            decision
            for decision in directive.ranking
            if decision.incident_id in directive.parked_incident_ids
        ]
        self.assertTrue(parked_decisions)
        self.assertTrue(
            all(not decision.resolution_inferred for decision in parked_decisions)
        )

    def test_oversized_or_broken_structured_context_is_never_sliced(self):
        manager = IncidentFocusLeaseManager(
            max_context_tokens=80,
            clock_ms=lambda: 1_000,
        )

        lease = manager.start(
            "broken-context",
            [1],
            ttl_seconds=30,
            context='{"summary":"' + ("evidence " * 500),
        )

        payload = json.loads(lease.context)
        self.assertEqual(payload["incident_id"], "lease-context")
        self.assertIn(payload["context_compaction"], {"digest", "parked"})
        self.assertFalse(payload["resolution_inferred"])

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
