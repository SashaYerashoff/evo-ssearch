import json
import unittest

from alert_probe_lifecycle import AlertProbeLifecycle, AlertProbeValidationError


def _spec(label, positive, negative):
    return {
        "label": label,
        "positives": [positive],
        "negatives": [negative],
    }


class AlertProbeLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.now = 1000.0
        self.lifecycle = AlertProbeLifecycle(
            per_channel_cap=4,
            global_cap=8,
            default_ttl_seconds=10.0,
            min_ttl_seconds=1.0,
            max_ttl_seconds=60.0,
            cooldown_seconds=5.0,
            clock=lambda: self.now,
        )

    def test_admits_two_to_four_probes_with_parent_lineage_and_deterministic_ids(self):
        specs = [
            _spec("person", "person crossing the gate", "empty gate"),
            _spec("vehicle", "moving vehicle at the gate", "parked vehicle"),
        ]

        first = self.lifecycle.admit_alert(
            parent_alert_id="alert-1",
            channel_id=7,
            specs=specs,
        )

        self.assertTrue(first.accepted)
        self.assertEqual(first.reason, "admitted")
        self.assertEqual(len(first.probes), 2)
        self.assertEqual({probe.parent_alert_id for probe in first.probes}, {"alert-1"})
        self.assertEqual({probe.generation for probe in first.probes}, {0})
        self.assertEqual({probe.origin for probe in first.probes}, {"vlm_alert"})
        self.assertEqual(first.probes[0].created_at, 1000.0)
        self.assertEqual(first.probes[0].expires_at, 1010.0)
        self.assertEqual(first.probes[0].cooldown_until, 1015.0)

        same_policy = AlertProbeLifecycle(
            per_channel_cap=4,
            global_cap=8,
            default_ttl_seconds=10.0,
            min_ttl_seconds=1.0,
            max_ttl_seconds=60.0,
            cooldown_seconds=5.0,
            clock=lambda: self.now,
        )
        repeated = same_policy.admit_alert(
            parent_alert_id="alert-1",
            channel_id=7,
            specs=specs,
        )
        self.assertEqual(
            [probe.probe_id for probe in first.probes],
            [probe.probe_id for probe in repeated.probes],
        )

    def test_requires_two_to_four_probes_and_positive_negative_contrast(self):
        with self.assertRaisesRegex(AlertProbeValidationError, "2-4"):
            self.lifecycle.admit_alert(
                parent_alert_id="alert-one",
                channel_id=7,
                specs=[_spec("one", "person", "empty")],
            )
        with self.assertRaisesRegex(AlertProbeValidationError, "2-4"):
            self.lifecycle.admit_alert(
                parent_alert_id="alert-five",
                channel_id=7,
                specs=[
                    _spec(str(index), f"positive {index}", f"negative {index}")
                    for index in range(5)
                ],
            )
        with self.assertRaisesRegex(AlertProbeValidationError, "negatives"):
            self.lifecycle.admit_alert(
                parent_alert_id="alert-no-negative",
                channel_id=7,
                specs=[
                    {"positives": ["person"], "negatives": []},
                    _spec("vehicle", "moving vehicle", "parked vehicle"),
                ],
            )
        with self.assertRaisesRegex(AlertProbeValidationError, "must not overlap"):
            self.lifecycle.admit_alert(
                parent_alert_id="alert-overlap",
                channel_id=7,
                specs=[
                    _spec("person", "Person at Gate", "person at gate"),
                    _spec("vehicle", "moving vehicle", "parked vehicle"),
                ],
            )

    def test_rejects_recursive_probe_derivation(self):
        result = self.lifecycle.admit_alert(
            parent_alert_id="alert-from-probe",
            channel_id=7,
            specs=[
                _spec("one", "person at gate", "empty gate"),
                _spec("two", "moving vehicle", "parked vehicle"),
            ],
            origin="probe_hit",
            generation=1,
            source_probe_id="alert-probe-existing",
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "recursive_probe_derivation_forbidden")
        self.assertEqual(self.lifecycle.active_probes(), ())

        event_result = self.lifecycle.admit_alert_event(
            {
                "title": "Probe hit",
                "description": "A temporary probe crossed its threshold.",
                "severity": "low",
                "channel_id": 7,
                "timestamp_ms": 1000000,
                "source": "probe_hit",
                "generation": 1,
                "source_probe_id": "alert-probe-existing",
            },
            allow_generated_fallback=True,
        )
        self.assertFalse(event_result.accepted)
        self.assertEqual(
            event_result.reason,
            "recursive_probe_derivation_forbidden",
        )
        self.assertEqual(self.lifecycle.active_probes(), ())

    def test_semantic_dedupe_is_polarity_aware_and_honors_cooldown(self):
        original = self.lifecycle.admit_alert(
            parent_alert_id="alert-original",
            channel_id=7,
            specs=[
                _spec("person", "red coat person near gate", "empty gate"),
                _spec("vehicle", "moving delivery vehicle", "parked vehicle"),
            ],
        )
        self.assertTrue(original.accepted)

        duplicate = self.lifecycle.admit_alert(
            parent_alert_id="alert-duplicate",
            channel_id=7,
            specs=[
                _spec("person copy", "person near gate in red coat", "gate empty"),
                _spec("different", "door held open", "closed door"),
            ],
        )
        self.assertFalse(duplicate.accepted)
        self.assertEqual(duplicate.reason, "semantically_duplicate_active_probe")

        self.now = 1010.0
        self.lifecycle.expire()
        cooling = self.lifecycle.admit_alert(
            parent_alert_id="alert-cooling",
            channel_id=7,
            specs=[
                _spec("person copy", "person near gate in red coat", "gate empty"),
                _spec("different", "door held open", "closed door"),
            ],
        )
        self.assertFalse(cooling.accepted)
        self.assertEqual(cooling.reason, "semantically_duplicate_probe_in_cooldown")

        self.now = 1015.1
        admitted = self.lifecycle.admit_alert(
            parent_alert_id="alert-after-cooldown",
            channel_id=7,
            specs=[
                _spec("person copy", "person near gate in red coat", "gate empty"),
                _spec("different", "door held open", "closed door"),
            ],
        )
        self.assertTrue(admitted.accepted)

    def test_duplicate_specs_inside_one_alert_are_transactionally_rejected(self):
        result = self.lifecycle.admit_alert(
            parent_alert_id="alert-duplicates",
            channel_id=7,
            specs=[
                _spec("a", "person near entrance", "empty entrance"),
                _spec("b", "entrance person near", "entrance empty"),
            ],
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "duplicate_probe_specs_in_alert")
        self.assertEqual(self.lifecycle.active_probes(), ())

    def test_channel_and_global_caps_reject_whole_alert(self):
        first = self.lifecycle.admit_alert(
            parent_alert_id="alert-1",
            channel_id=1,
            specs=[
                _spec("a", "person at west gate", "empty west gate"),
                _spec("b", "moving car at west gate", "parked car at west gate"),
                _spec("c", "open west door", "closed west door"),
            ],
        )
        self.assertTrue(first.accepted)

        channel_capped = self.lifecycle.admit_alert(
            parent_alert_id="alert-2",
            channel_id=1,
            specs=[
                _spec("d", "smoke by west gate", "clear air at west gate"),
                _spec("e", "fallen person west", "standing person west"),
            ],
        )
        self.assertFalse(channel_capped.accepted)
        self.assertEqual(channel_capped.reason, "per_channel_cap_exceeded")
        self.assertEqual(len(self.lifecycle.active_probes(channel_id=1)), 3)

        second_channel = self.lifecycle.admit_alert(
            parent_alert_id="alert-3",
            channel_id=2,
            specs=[
                _spec("f", "person at east gate", "empty east gate"),
                _spec("g", "moving car at east gate", "parked car at east gate"),
                _spec("h", "open east door", "closed east door"),
                _spec("i", "smoke east", "clear air east"),
            ],
        )
        self.assertTrue(second_channel.accepted)

        globally_capped = self.lifecycle.admit_alert(
            parent_alert_id="alert-4",
            channel_id=3,
            specs=[
                _spec("j", "person at north gate", "empty north gate"),
                _spec("k", "moving car north", "parked car north"),
            ],
        )
        self.assertFalse(globally_capped.accepted)
        self.assertEqual(globally_capped.reason, "global_cap_exceeded")
        self.assertEqual(len(self.lifecycle.active_probes()), 7)

    def test_expiry_retirement_and_runtime_status_are_serializable(self):
        admitted = self.lifecycle.admit_alert(
            parent_alert_id="alert-status",
            channel_id=7,
            specs=[
                _spec("person", "person at loading bay", "empty loading bay"),
                _spec("vehicle", "moving truck", "parked truck"),
            ],
        )
        retired = self.lifecycle.retire_probe(
            admitted.probes[0].probe_id,
            reason="operator dismissed",
            now=1002.0,
        )
        self.assertEqual(retired.status, "retired")
        self.assertEqual(retired.end_reason, "operator dismissed")

        self.now = 1011.0
        expired = self.lifecycle.expire()
        self.assertEqual(len(expired), 1)
        self.assertEqual(expired[0].status, "expired")
        self.assertEqual(expired[0].ended_at, 1010.0)

        payload = self.lifecycle.status(now=1011.0)
        self.assertEqual(payload["counts"], {"active": 0, "expired": 1, "retired": 1})
        serialized = self.lifecycle.dumps(now=1011.0)
        decoded = json.loads(serialized)
        self.assertEqual(decoded, payload)
        self.assertEqual(serialized, self.lifecycle.dumps(now=1011.0))

    def test_same_parent_alert_is_idempotently_not_reprocessed(self):
        specs = [
            _spec("person", "person at gate", "empty gate"),
            _spec("car", "moving car", "parked car"),
        ]
        first = self.lifecycle.admit_alert(
            parent_alert_id="same-alert",
            channel_id=7,
            specs=specs,
        )
        second = self.lifecycle.admit_alert(
            parent_alert_id="same-alert",
            channel_id=7,
            specs=specs,
        )

        self.assertTrue(first.accepted)
        self.assertFalse(second.accepted)
        self.assertEqual(second.reason, "parent_alert_already_processed")
        self.assertEqual(
            [probe.probe_id for probe in first.probes],
            [probe.probe_id for probe in second.probes],
        )

    def test_custom_similarity_can_use_runtime_semantic_embeddings(self):
        lifecycle = AlertProbeLifecycle(
            per_channel_cap=4,
            global_cap=8,
            min_ttl_seconds=1.0,
            similarity_fn=lambda left, right: (
                0.95 if "automobile" in left.positives[0] and "car" in right.positives[0] else 0.0
            ),
            clock=lambda: self.now,
        )
        first = lifecycle.admit_alert(
            parent_alert_id="car-alert",
            channel_id=4,
            specs=[
                _spec("car", "moving car", "parked car"),
                _spec("person", "walking person", "empty walkway"),
            ],
        )
        self.assertTrue(first.accepted)

        duplicate = lifecycle.admit_alert(
            parent_alert_id="automobile-alert",
            channel_id=4,
            specs=[
                _spec("auto", "moving automobile", "parked automobile"),
                _spec("door", "open door", "closed door"),
            ],
        )
        self.assertFalse(duplicate.accepted)
        self.assertEqual(duplicate.reason, "semantically_duplicate_active_probe")

    def test_ingests_eva_alert_event_and_returns_probe_store_payloads(self):
        event = {
            "title": "Vehicle drifting near gate",
            "description": "A moving vehicle deviates toward the closed east gate.",
            "severity": "high",
            "channel_id": 118,
            "timestamp_ms": 1000500,
            "delivery_status": "bookmark_disabled",
            "probe_specs": [
                _spec(
                    "drift persistence",
                    "vehicle drifting toward closed east gate",
                    "vehicle travelling normally through roadway",
                ),
                _spec(
                    "gate proximity",
                    "vehicle unusually close to east gate",
                    "empty gate area with vehicle away from gate",
                ),
            ],
        }

        admitted = self.lifecycle.admit_alert_event(event)

        self.assertTrue(admitted.accepted)
        payloads = admitted.store_payloads()
        self.assertEqual(len(payloads), 2)
        first = payloads[0]
        self.assertEqual(first["id"], admitted.probes[0].probe_id)
        self.assertEqual(first["channel_id"], 118)
        self.assertEqual(first["severity"], "high")
        self.assertTrue(first["temporary"])
        self.assertTrue(first["enabled"])
        self.assertFalse(first["bookmark"])
        self.assertFalse(first["bookmark_authorized"])
        self.assertEqual(first["source"], "vlm_alert")
        self.assertEqual(first["generation"], 0)
        self.assertEqual(first["parent_alert_title"], event["title"])
        self.assertEqual(first["parent_alert_timestamp_ms"], event["timestamp_ms"])
        self.assertEqual(first["expires_at_ms"], 1010000)
        self.assertEqual(first["runtime_status"], "active")
        self.assertEqual(first["lifecycle"]["status"], "active")
        self.assertEqual(first["pos_floor"], 0.05)
        self.assertEqual(first["margin"], 0.02)
        json.dumps(first, allow_nan=False)

        same_event = AlertProbeLifecycle(
            per_channel_cap=4,
            global_cap=8,
            default_ttl_seconds=10.0,
            min_ttl_seconds=1.0,
            max_ttl_seconds=60.0,
            cooldown_seconds=5.0,
            clock=lambda: self.now,
        ).admit_alert_event(event)
        self.assertEqual(admitted.parent_alert_id, same_event.parent_alert_id)
        self.assertTrue(admitted.parent_alert_id.startswith("vlm-alert-"))

    def test_alert_event_requires_explicit_contrast_specs(self):
        with self.assertRaisesRegex(AlertProbeValidationError, "explicit"):
            self.lifecycle.admit_alert_event(
                {
                    "title": "Person down",
                    "description": "A person appears prone.",
                    "severity": "critical",
                    "channel_id": 7,
                    "timestamp_ms": 1000000,
                }
            )

    def test_opt_in_fallback_builds_two_positive_only_low_confidence_payloads(self):
        event = {
            "title": "Person down",
            "description": "A person appears prone beside the loading bay.",
            "severity": "critical",
            "channel_id": 7,
            "timestamp_ms": 1000000,
        }

        admitted = self.lifecycle.admit_alert_event(
            event,
            allow_generated_fallback=True,
        )

        self.assertTrue(admitted.accepted)
        self.assertEqual(len(admitted.probes), 2)
        self.assertTrue(all(probe.generated_fallback for probe in admitted.probes))
        self.assertEqual({probe.confidence for probe in admitted.probes}, {"low"})
        self.assertEqual([probe.spec.negatives for probe in admitted.probes], [(), ()])
        self.assertEqual(admitted.probes[0].spec.positives, ("Person down",))
        self.assertIn(
            "security camera view of Person down",
            admitted.probes[1].spec.positives[0],
        )

        payloads = admitted.store_payloads()
        self.assertTrue(all(payload["generated_fallback"] for payload in payloads))
        self.assertEqual({payload["confidence"] for payload in payloads}, {"low"})
        self.assertEqual([payload["negatives"] for payload in payloads], [[], []])
        self.assertTrue(all(payload["temporary"] for payload in payloads))

    def test_fallback_variants_are_admitted_even_for_a_long_related_title(self):
        admitted = self.lifecycle.admit_alert_event(
            {
                "title": (
                    "Person wearing a dark jacket and carrying a large bag "
                    "walks slowly beside the closed loading entrance"
                ),
                "description": "",
                "severity": "low",
                "channel_id": 8,
                "timestamp_ms": 1000001,
            },
            allow_generated_fallback=True,
        )

        self.assertTrue(admitted.accepted)
        self.assertEqual(len(admitted.probes), 2)


if __name__ == "__main__":
    unittest.main()
