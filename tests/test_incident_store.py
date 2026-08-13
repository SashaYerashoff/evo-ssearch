from __future__ import annotations

import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from incident_store import (
    INCIDENT_ATTENTION_STATES,
    INCIDENT_CASE_STATES,
    INCIDENT_PERCEPTION_STATES,
    INCIDENT_RISK_STATES,
    INCIDENT_STORAGE_REVISION,
    IncidentIdempotencyConflict,
    IncidentRevisionConflict,
    PostgresIncidentStore,
)


ROOT = Path(__file__).resolve().parent.parent
LEGACY_MIGRATION = ROOT / "migrations" / "versions" / "20260801_0011_incidents.py"
MIGRATION = ROOT / "migrations" / "versions" / (
    "20260805_0012_incident_temporal_memory.py"
)
TENANT_ID = "f3c3533e-bf17-46a1-a543-696d95b8cf6f"
ACTOR_ID = "85f620ba-fc37-4f67-ad1a-8fcf3d983461"
INCIDENT_ID = "384be4e4-8c4e-4aa8-941a-17807168cb8c"
OBSERVATION_ID = "384be4e4-8c4e-4aa8-941a-17807168cb8d"
EPISODE_ID = "384be4e4-8c4e-4aa8-941a-17807168cb8e"
RELATION_ID = "384be4e4-8c4e-4aa8-941a-17807168cb8f"
OTHER_INCIDENT_ID = "384be4e4-8c4e-4aa8-941a-17807168cb90"
TRANSITION_ID = "384be4e4-8c4e-4aa8-941a-17807168cb91"


def incident_row(
    *,
    revision: int = 1,
    title: str = "Craft near port gate",
    perception_state: str = "unknown",
    risk_state: str = "unknown",
    case_state: str = "unknown",
    attention_state: str = "unknown",
    identity_key: str | None = None,
    idempotency_key: str | None = None,
):
    now = datetime(2026, 8, 1, 9, 30, tzinfo=timezone.utc)
    return (
        INCIDENT_ID,
        revision,
        "draft",
        title,
        [112, 118],
        1_000,
        1_100,
        1_900,
        2_000,
        {"detection_id": 41},
        [{"event_ref": "vlm-summary:10"}],
        [{"detection_ref": "archive.detections:41"}],
        [{"interval_ref": "attention.interval:9"}],
        {"status": "covered"},
        ["Vessel type requires operator review."],
        {},
        {"mode": "elevated"},
        perception_state,
        risk_state,
        case_state,
        attention_state,
        identity_key,
        idempotency_key,
        ACTOR_ID,
        ACTOR_ID,
        now,
        now,
    )


def observation_row(*, payload=None, idempotency_key: str = "batch:b-7:event:1"):
    now = datetime(2026, 8, 1, 9, 31, tzinfo=timezone.utc)
    return (
        OBSERVATION_ID,
        INCIDENT_ID,
        idempotency_key,
        "l0_event",
        1_500,
        112,
        "observed",
        {"summary_id": 77},
        payload if payload is not None else {"semantic_key": "gate_crossing"},
        ACTOR_ID,
        now,
    )


def episode_row(*, idempotency_key: str = "primary:episode-1"):
    now = datetime(2026, 8, 1, 9, 31, tzinfo=timezone.utc)
    return (
        EPISODE_ID,
        INCIDENT_ID,
        idempotency_key,
        "episode-1",
        "observed",
        "craft gate_crossing",
        None,
        "port_gate",
        1_000,
        1_100,
        1_900,
        2_000,
        {},
        {"routine_id": "routine-2"},
        [{"detection_id": 41}],
        {"scale_disposition": "unclassified_keep"},
        ACTOR_ID,
        now,
    )


def relation_row(*, idempotency_key: str = "series:pair-1"):
    now = datetime(2026, 8, 1, 9, 32, tzinfo=timezone.utc)
    return (
        RELATION_ID,
        INCIDENT_ID,
        OTHER_INCIDENT_ID,
        idempotency_key,
        "series_member",
        "candidate",
        "medium",
        "Same semantic track in a separate window.",
        {"automatic_merge": False},
        ACTOR_ID,
        now,
    )


def transition_row(*, idempotency_key: str = "revision:2:case"):
    now = datetime(2026, 8, 1, 9, 33, tzinfo=timezone.utc)
    return (
        TRANSITION_ID,
        INCIDENT_ID,
        idempotency_key,
        "case",
        "candidate",
        "open",
        2,
        2_100,
        "operator confirmed incident",
        "operator_review",
        {"action": "confirm"},
        {"ground_truth": "operator_review"},
        ACTOR_ID,
        now,
    )


def incident_input(**changes):
    record = {
        "id": INCIDENT_ID,
        "state": "draft",
        "title": "Craft near port gate",
        "channel_ids": [112, 118],
        "possible_start_ms": 1_000,
        "observed_start_ms": 1_100,
        "observed_end_ms": 1_900,
        "possible_end_ms": 2_000,
        "anchor_ref": {"detection_id": 41},
        "timeline_refs": [{"event_ref": "vlm-summary:10"}],
        "evidence_refs": [{"detection_ref": "archive.detections:41"}],
        "qualia_refs": [{"interval_ref": "attention.interval:9"}],
        "coverage": {"status": "covered"},
        "uncertainties": ["Vessel type requires operator review."],
        "report": {},
        "follow_policy": {"mode": "elevated"},
    }
    record.update(changes)
    return record


class FakeResult:
    def __init__(self, rows=None):
        self.rows = list(rows or [])

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, results=None):
        self.calls = []
        self.results = list(results or [])

    def execute(self, sql, params=None):
        self.calls.append((" ".join(sql.split()), params))
        rows = self.results.pop(0) if self.results else []
        return FakeResult(rows)


class FakePool:
    def __init__(self, connection):
        self.connection = connection
        self.contexts = []

    @contextmanager
    def transaction(self, context, readonly=False):
        self.contexts.append((context, readonly))
        yield self.connection


class IncidentValidationTests(unittest.TestCase):
    def test_create_rejects_inline_images_and_invalid_time_bounds(self):
        store = PostgresIncidentStore(FakePool(FakeConnection()), TENANT_ID)
        with self.assertRaisesRegex(ValueError, "reference"):
            store.create_incident(
                incident_input(evidence_refs=[{"thumbnail_b64": "pixels"}])
            )
        with self.assertRaisesRegex(ValueError, "observed_start_ms"):
            store.create_incident(incident_input(observed_start_ms=999))

    def test_create_validates_channels_and_json_shapes(self):
        store = PostgresIncidentStore(FakePool(FakeConnection()), TENANT_ID)
        with self.assertRaisesRegex(ValueError, "channel_ids"):
            store.create_incident(incident_input(channel_ids=[]))
        with self.assertRaisesRegex(ValueError, "JSON array"):
            store.create_incident(incident_input(timeline_refs={"id": 1}))

    def test_lifecycle_axes_are_independent_and_unknown_safe(self):
        connection = FakeConnection(results=[[incident_row()]])
        store = PostgresIncidentStore(FakePool(connection), TENANT_ID)

        created = store.create_incident(incident_input())

        self.assertEqual(created["state"], "draft")
        self.assertEqual(created["perception_state"], "unknown")
        self.assertEqual(created["risk_state"], "unknown")
        self.assertEqual(created["case_state"], "unknown")
        self.assertEqual(created["attention_state"], "unknown")
        self.assertIn("unknown", INCIDENT_PERCEPTION_STATES)
        self.assertIn("unknown", INCIDENT_RISK_STATES)
        self.assertIn("unknown", INCIDENT_CASE_STATES)
        self.assertIn("unknown", INCIDENT_ATTENTION_STATES)

        with self.assertRaisesRegex(ValueError, "risk_state"):
            store.create_incident(incident_input(risk_state="draft"))


class IncidentPostgresStoreTests(unittest.TestCase):
    def test_create_is_tenant_scoped_and_returns_durable_record(self):
        connection = FakeConnection(results=[[incident_row()]])
        pool = FakePool(connection)
        store = PostgresIncidentStore(pool, TENANT_ID, actor_id=ACTOR_ID)

        created = store.create_incident(incident_input())

        self.assertEqual(created["id"], INCIDENT_ID)
        self.assertEqual(created["channel_ids"], [112, 118])
        sql, params = connection.calls[0]
        self.assertIn("INSERT INTO archive.incidents", sql)
        self.assertEqual(params[0], TENANT_ID)
        self.assertEqual(params[-2:], (ACTOR_ID, ACTOR_ID))
        self.assertEqual(pool.contexts[0][0].tenant_id, TENANT_ID)
        self.assertFalse(pool.contexts[0][1])

    def test_create_replay_is_idempotent_and_uses_typed_optional_keys(self):
        identity_key = "channel:112:craft:gate"
        idempotency_key = "incident:create:batch-7:1"
        connection = FakeConnection(
            results=[
                [],
                [
                    incident_row(
                        identity_key=identity_key,
                        idempotency_key=idempotency_key,
                    )
                ],
            ]
        )
        store = PostgresIncidentStore(FakePool(connection), TENANT_ID)

        replayed = store.create_incident(
            incident_input(
                identity_key=identity_key,
                idempotency_key=idempotency_key,
            )
        )

        self.assertEqual(replayed["id"], INCIDENT_ID)
        replay_sql = connection.calls[1][0]
        self.assertIn("%s::text IS NOT NULL", replay_sql)

        conflicting_connection = FakeConnection(
            results=[
                [],
                [
                    incident_row(
                        identity_key=identity_key,
                        idempotency_key=idempotency_key,
                    )
                ],
            ]
        )
        conflicting_store = PostgresIncidentStore(
            FakePool(conflicting_connection), TENANT_ID
        )
        with self.assertRaises(IncidentIdempotencyConflict):
            conflicting_store.create_incident(
                incident_input(
                    title="Different incident",
                    identity_key=identity_key,
                    idempotency_key=idempotency_key,
                )
            )

    def test_get_is_readonly_and_tenant_scoped(self):
        connection = FakeConnection(results=[[incident_row()]])
        pool = FakePool(connection)
        store = PostgresIncidentStore(pool, TENANT_ID)

        fetched = store.get_incident(INCIDENT_ID)

        self.assertEqual(fetched["revision"], 1)
        sql, params = connection.calls[0]
        self.assertIn("tenant_id = %s AND id = %s", sql)
        self.assertEqual(params, (TENANT_ID, INCIDENT_ID))
        self.assertTrue(pool.contexts[0][1])

    def test_update_locks_row_and_increments_matching_revision(self):
        connection = FakeConnection(
            results=[[incident_row()], [incident_row(revision=2, title="Confirmed craft")]]
        )
        store = PostgresIncidentStore(FakePool(connection), TENANT_ID)

        updated = store.update_incident(
            INCIDENT_ID,
            expected_revision=1,
            changes={"title": "Confirmed craft"},
        )

        self.assertEqual(updated["revision"], 2)
        self.assertEqual(updated["title"], "Confirmed craft")
        self.assertIn("FOR UPDATE", connection.calls[0][0])
        self.assertIn("AND revision = %s", connection.calls[1][0])
        self.assertEqual(connection.calls[1][1][-1], 1)

    def test_update_sets_lifecycle_axes_without_reinterpreting_legacy_state(self):
        connection = FakeConnection(
            results=[
                [incident_row()],
                [
                    incident_row(
                        revision=2,
                        perception_state="ended",
                        risk_state="occurred",
                        case_state="open",
                        attention_state="inactive",
                        identity_key="channel:112:theft:bag-7",
                    )
                ],
            ]
        )
        store = PostgresIncidentStore(FakePool(connection), TENANT_ID)

        updated = store.update_incident(
            INCIDENT_ID,
            expected_revision=1,
            changes={
                "perception_state": "ended",
                "risk_state": "occurred",
                "case_state": "open",
                "attention_state": "inactive",
                "identity_key": "channel:112:theft:bag-7",
            },
        )

        self.assertEqual(updated["state"], "draft")
        self.assertEqual(updated["perception_state"], "ended")
        self.assertEqual(updated["risk_state"], "occurred")
        self.assertEqual(updated["case_state"], "open")
        self.assertEqual(updated["identity_key"], "channel:112:theft:bag-7")
        self.assertIn("risk_state = %s", connection.calls[1][0])

    def test_update_and_transition_ledger_commit_in_one_transaction(self):
        connection = FakeConnection(
            results=[
                [incident_row(case_state="candidate")],
                [incident_row(revision=2, case_state="open")],
                [transition_row()],
            ]
        )
        pool = FakePool(connection)
        store = PostgresIncidentStore(pool, TENANT_ID)

        updated = store.update_incident(
            INCIDENT_ID,
            expected_revision=1,
            changes={"case_state": "open"},
            transition={
                "transitioned_at_ms": 2_100,
                "reason": "operator confirmed incident",
                "source_kind": "operator_review",
                "source_ref": {"action": "confirm"},
                "payload": {"ground_truth": "operator_review"},
            },
        )

        self.assertEqual(updated["case_state"], "open")
        self.assertEqual(len(pool.contexts), 1)
        self.assertFalse(pool.contexts[0][1])
        self.assertIn("UPDATE archive.incidents", connection.calls[1][0])
        self.assertIn("INSERT INTO archive.incident_transitions", connection.calls[2][0])
        self.assertEqual(connection.calls[2][1][3], "revision:2:case")

    def test_identity_key_is_assign_once(self):
        connection = FakeConnection(
            results=[[incident_row(identity_key="channel:112:fire:zone-a")]]
        )
        store = PostgresIncidentStore(FakePool(connection), TENANT_ID)

        with self.assertRaisesRegex(ValueError, "immutable"):
            store.update_incident(
                INCIDENT_ID,
                expected_revision=1,
                changes={"identity_key": "channel:112:theft:bag-7"},
            )

        self.assertEqual(len(connection.calls), 1)

    def test_update_rejects_stale_revision_before_write(self):
        connection = FakeConnection(results=[[incident_row(revision=3)]])
        store = PostgresIncidentStore(FakePool(connection), TENANT_ID)

        with self.assertRaises(IncidentRevisionConflict) as raised:
            store.update_incident(
                INCIDENT_ID,
                expected_revision=2,
                changes={"title": "Stale edit"},
            )

        self.assertEqual(raised.exception.actual_revision, 3)
        self.assertEqual(len(connection.calls), 1)

    def test_list_filters_by_channel_state_and_overlapping_window(self):
        connection = FakeConnection(results=[[(1,)], [incident_row()]])
        pool = FakePool(connection)
        store = PostgresIncidentStore(pool, TENANT_ID)

        rows, total = store.list_incidents(
            channel_ids=[112],
            states=["draft"],
            since_ms=900,
            until_ms=2_100,
            limit=900,
            offset=-10,
        )

        self.assertEqual(total, 1)
        self.assertEqual(rows[0]["id"], INCIDENT_ID)
        sql, params = connection.calls[1]
        self.assertIn("channel_ids && %s::bigint[]", sql)
        self.assertIn("state = ANY(%s)", sql)
        self.assertIn("COALESCE(possible_end_ms", sql)
        self.assertEqual(params[-2:], (500, 0))
        self.assertTrue(pool.contexts[0][1])

    def test_list_filters_independent_lifecycle_axes(self):
        connection = FakeConnection(results=[[(1,)], [incident_row()]])
        store = PostgresIncidentStore(FakePool(connection), TENANT_ID)

        store.list_incidents(
            perception_states=["observed"],
            risk_states=["active"],
            case_states=["open"],
            attention_states=["critical"],
        )

        sql, params = connection.calls[1]
        self.assertIn("perception_state = ANY(%s)", sql)
        self.assertIn("risk_state = ANY(%s)", sql)
        self.assertIn("case_state = ANY(%s)", sql)
        self.assertIn("attention_state = ANY(%s)", sql)
        self.assertEqual(
            params[1:5],
            (["observed"], ["active"], ["open"], ["critical"]),
        )

    def test_review_list_filters_nested_incidents_before_count_and_paging(self):
        connection = FakeConnection(results=[[(1,)], [incident_row()]])
        store = PostgresIncidentStore(FakePool(connection), TENANT_ID)

        store.list_incidents(top_level_only=True, limit=50, offset=10)

        count_sql, _count_params = connection.calls[0]
        page_sql, page_params = connection.calls[1]
        for sql in (count_sql, page_sql):
            self.assertIn("report_json #>> '{presentation,scope}'", sql)
            self.assertIn(
                "report_json #>> '{presentation,parent_incident_id}'",
                sql,
            )
            self.assertIn("report_json ->> 'priority'", sql)
            self.assertIn("NOT IN ('operator_criterion', 'safety')", sql)
        self.assertEqual(page_params[-2:], (50, 10))

    def test_review_list_hides_only_untouched_legacy_temporal_candidates(self):
        connection = FakeConnection(results=[[(1,)], [incident_row()]])
        store = PostgresIncidentStore(FakePool(connection), TENANT_ID)

        store.list_incidents(operator_review_only=True, limit=50, offset=10)

        count_sql, _count_params = connection.calls[0]
        page_sql, page_params = connection.calls[1]
        for sql in (count_sql, page_sql):
            self.assertIn("report_json ->> 'source'", sql)
            self.assertIn("= 'vlm_l0_temporal'", sql)
            self.assertIn("report_json ->> 'priority'", sql)
            self.assertIn("case_state = 'candidate'", sql)
            self.assertIn("attention_state IN ('unknown', 'inactive')", sql)
            self.assertIn("risk_state = 'unknown'", sql)
        self.assertEqual(page_params[-2:], (50, 10))


class IncidentObservationStoreTests(unittest.TestCase):
    def test_append_observation_is_tenant_scoped_and_returns_immutable_record(self):
        connection = FakeConnection(results=[[observation_row()]])
        pool = FakePool(connection)
        store = PostgresIncidentStore(pool, TENANT_ID, actor_id=ACTOR_ID)

        created = store.append_observation(
            {
                "incident_id": INCIDENT_ID,
                "idempotency_key": "batch:b-7:event:1",
                "source_kind": "l0_event",
                "observed_at_ms": 1_500,
                "channel_id": 112,
                "perception_state": "observed",
                "source_ref": {"summary_id": 77},
                "payload": {"semantic_key": "gate_crossing"},
            }
        )

        self.assertEqual(created["id"], OBSERVATION_ID)
        self.assertEqual(created["incident_id"], INCIDENT_ID)
        self.assertEqual(created["perception_state"], "observed")
        sql, params = connection.calls[0]
        self.assertIn("INSERT INTO archive.incident_observations", sql)
        self.assertIn("ON CONFLICT (tenant_id, incident_id, idempotency_key)", sql)
        self.assertEqual(params[0], TENANT_ID)
        self.assertEqual(params[-1], ACTOR_ID)
        self.assertFalse(pool.contexts[0][1])

    def test_append_observation_replay_is_idempotent_but_payload_change_conflicts(self):
        replay_connection = FakeConnection(results=[[], [observation_row()]])
        store = PostgresIncidentStore(FakePool(replay_connection), TENANT_ID)
        payload = {
            "incident_id": INCIDENT_ID,
            "idempotency_key": "batch:b-7:event:1",
            "source_kind": "l0_event",
            "observed_at_ms": 1_500,
            "channel_id": 112,
            "perception_state": "observed",
            "source_ref": {"summary_id": 77},
            "payload": {"semantic_key": "gate_crossing"},
        }

        replayed = store.append_observation(payload)
        self.assertEqual(replayed["id"], OBSERVATION_ID)
        self.assertEqual(len(replay_connection.calls), 2)

        conflict_connection = FakeConnection(results=[[], [observation_row()]])
        conflicting = PostgresIncidentStore(
            FakePool(conflict_connection), TENANT_ID
        )
        with self.assertRaises(IncidentIdempotencyConflict):
            conflicting.append_observation(
                {**payload, "payload": {"semantic_key": "different"}}
            )

    def test_list_observations_is_bounded_chronological_and_tenant_scoped(self):
        connection = FakeConnection(results=[[(1,)], [observation_row()]])
        pool = FakePool(connection)
        store = PostgresIncidentStore(pool, TENANT_ID)

        rows, total = store.list_observations(
            INCIDENT_ID,
            since_ms=1_000,
            until_ms=2_000,
            source_kind="l0_event",
            limit=20_000,
            offset=-1,
        )

        self.assertEqual(total, 1)
        self.assertEqual(rows[0]["id"], OBSERVATION_ID)
        sql, params = connection.calls[1]
        self.assertIn("ORDER BY observed_at_ms ASC, id ASC", sql)
        self.assertEqual(params[:2], (TENANT_ID, INCIDENT_ID))
        self.assertEqual(params[-2:], (2_000, 0))
        self.assertTrue(all(readonly for _context, readonly in pool.contexts))


class IncidentTemporalLedgerStoreTests(unittest.TestCase):
    def test_append_and_list_episode_are_replay_safe(self):
        connection = FakeConnection(results=[[episode_row()]])
        store = PostgresIncidentStore(FakePool(connection), TENANT_ID)
        payload = {
            "id": EPISODE_ID,
            "incident_id": INCIDENT_ID,
            "idempotency_key": "primary:episode-1",
            "episode_key": "episode-1",
            "perception_state": "observed",
            "semantic_key": "craft gate_crossing",
            "zone_key": "port_gate",
            "possible_start_ms": 1_000,
            "observed_start_ms": 1_100,
            "observed_end_ms": 1_900,
            "possible_end_ms": 2_000,
            "routine_after_ref": {"routine_id": "routine-2"},
            "evidence_refs": [{"detection_id": 41}],
            "coverage": {"scale_disposition": "unclassified_keep"},
        }

        stored = store.append_episode(payload)

        self.assertEqual(stored["episode_key"], "episode-1")
        self.assertEqual(stored["semantic_key"], "craft gate_crossing")
        self.assertIn("ON CONFLICT DO NOTHING", connection.calls[0][0])

        replay_connection = FakeConnection(results=[[], [episode_row()]])
        replayed = PostgresIncidentStore(
            FakePool(replay_connection), TENANT_ID
        ).append_episode(payload)
        self.assertEqual(replayed["id"], EPISODE_ID)

        list_connection = FakeConnection(results=[[(1,)], [episode_row()]])
        rows, total = PostgresIncidentStore(
            FakePool(list_connection), TENANT_ID
        ).list_episodes(INCIDENT_ID, limit=5)
        self.assertEqual(total, 1)
        self.assertEqual(rows[0]["coverage"]["scale_disposition"], "unclassified_keep")
        self.assertIn("ORDER BY possible_start_ms ASC", list_connection.calls[1][0])

    def test_series_relation_is_candidate_and_never_merges_automatically(self):
        connection = FakeConnection(results=[[relation_row()]])
        store = PostgresIncidentStore(FakePool(connection), TENANT_ID)
        payload = {
            "id": RELATION_ID,
            "subject_incident_id": INCIDENT_ID,
            "object_incident_id": OTHER_INCIDENT_ID,
            "idempotency_key": "series:pair-1",
            "relation_type": "series_member",
            "relation_state": "candidate",
            "confidence": "medium",
            "rationale": "Same semantic track in a separate window.",
            "payload": {"automatic_merge": False},
        }

        relation = store.append_relation(payload)

        self.assertEqual(relation["relation_state"], "candidate")
        self.assertFalse(relation["payload"]["automatic_merge"])
        self.assertNotIn("UPDATE archive.incidents", connection.calls[0][0])

        list_connection = FakeConnection(results=[[(1,)], [relation_row()]])
        rows, total = PostgresIncidentStore(
            FakePool(list_connection), TENANT_ID
        ).list_relations(INCIDENT_ID)
        self.assertEqual(total, 1)
        self.assertEqual(rows[0]["relation_type"], "series_member")
        self.assertIn("subject_incident_id = %s OR object_incident_id = %s", list_connection.calls[0][0])

    def test_temporal_ledgers_reject_self_relation_and_invalid_episode_time(self):
        store = PostgresIncidentStore(FakePool(FakeConnection()), TENANT_ID)
        with self.assertRaisesRegex(ValueError, "distinct"):
            store.append_relation(
                {
                    "subject_incident_id": INCIDENT_ID,
                    "object_incident_id": INCIDENT_ID,
                    "idempotency_key": "self",
                    "relation_type": "series_member",
                }
            )
        with self.assertRaisesRegex(ValueError, "observed_end_ms"):
            store.append_episode(
                {
                    "incident_id": INCIDENT_ID,
                    "idempotency_key": "bad-time",
                    "episode_key": "bad-time",
                    "possible_start_ms": 2_000,
                    "observed_start_ms": 2_100,
                    "observed_end_ms": 2_050,
                }
            )

    def test_append_and_list_transition_are_replay_safe_and_chronological(self):
        payload = {
            "id": TRANSITION_ID,
            "incident_id": INCIDENT_ID,
            "idempotency_key": "revision:2:case",
            "axis": "case",
            "from_state": "candidate",
            "to_state": "open",
            "incident_revision": 2,
            "transitioned_at_ms": 2_100,
            "reason": "operator confirmed incident",
            "source_kind": "operator_review",
            "source_ref": {"action": "confirm"},
            "payload": {"ground_truth": "operator_review"},
        }
        connection = FakeConnection(results=[[transition_row()]])
        stored = PostgresIncidentStore(
            FakePool(connection), TENANT_ID
        ).append_transition(payload)

        self.assertEqual(stored["axis"], "case")
        self.assertEqual(stored["from_state"], "candidate")
        self.assertEqual(stored["to_state"], "open")
        self.assertIn("INSERT INTO archive.incident_transitions", connection.calls[0][0])

        replay_connection = FakeConnection(results=[[], [transition_row()]])
        replayed = PostgresIncidentStore(
            FakePool(replay_connection), TENANT_ID
        ).append_transition(payload)
        self.assertEqual(replayed["id"], TRANSITION_ID)

        list_connection = FakeConnection(results=[[(1,)], [transition_row()]])
        rows, total = PostgresIncidentStore(
            FakePool(list_connection), TENANT_ID
        ).list_transitions(INCIDENT_ID, limit=5)
        self.assertEqual(total, 1)
        self.assertEqual(rows[0]["incident_revision"], 2)
        self.assertIn(
            "ORDER BY transitioned_at_ms ASC, id ASC",
            list_connection.calls[1][0],
        )

    def test_transition_rejects_invalid_axis_state_and_revision(self):
        store = PostgresIncidentStore(FakePool(FakeConnection()), TENANT_ID)
        base = {
            "incident_id": INCIDENT_ID,
            "idempotency_key": "bad-transition",
            "axis": "case",
            "from_state": "candidate",
            "to_state": "open",
            "incident_revision": 2,
            "transitioned_at_ms": 2_100,
            "source_kind": "operator_review",
        }
        with self.assertRaisesRegex(ValueError, "to_state"):
            store.append_transition({**base, "to_state": "observed"})
        with self.assertRaisesRegex(ValueError, "incident_revision"):
            store.append_transition({**base, "incident_revision": 0})


class IncidentMigrationTests(unittest.TestCase):
    def test_migration_is_additive_unknown_safe_and_preserves_legacy_state(self):
        legacy_source = LEGACY_MIGRATION.read_text(encoding="utf-8")
        source = MIGRATION.read_text(encoding="utf-8")
        self.assertIn(f'revision: str = "{INCIDENT_STORAGE_REVISION}"', source)
        self.assertIn('down_revision: str | None = "20260801_0011"', source)
        self.assertIn("ALTER TABLE archive.incidents", source)
        self.assertIn("ADD COLUMN perception_state", source)
        self.assertIn("ADD COLUMN risk_state", source)
        self.assertIn("ADD COLUMN case_state", source)
        self.assertIn("ADD COLUMN attention_state", source)
        self.assertGreaterEqual(source.count("DEFAULT 'unknown'"), 4)
        self.assertNotIn("UPDATE archive.incidents", source)
        self.assertIn("CREATE TABLE archive.incidents", legacy_source)
        self.assertIn("'candidate', 'draft', 'following'", legacy_source)

    def test_temporal_ledgers_have_composite_fks_rls_bounds_and_append_only_grants(self):
        source = MIGRATION.read_text(encoding="utf-8")
        for table in (
            "incident_observations",
            "incident_episodes",
            "incident_relations",
            "incident_transitions",
        ):
            self.assertIn(f"CREATE TABLE archive.{table}", source)
            self.assertIn(f'"{table}"', source)
        self.assertIn("CREATE POLICY archive_{table}_tenant_isolation", source)
        self.assertIn("_tenant_policy(table)", source)
        self.assertGreaterEqual(
            source.count("FOREIGN KEY (tenant_id, incident_id)"),
            3,
        )
        self.assertIn("FOREIGN KEY (tenant_id, subject_incident_id)", source)
        self.assertIn("FOREIGN KEY (tenant_id, object_incident_id)", source)
        self.assertGreaterEqual(source.count("idempotency_key text NOT NULL"), 4)
        self.assertIn("GRANT SELECT, INSERT ON", source)
        grant_section = source.split("GRANT SELECT, INSERT ON", 1)[1].split(
            "TO eva_api, eva_worker", 1
        )[0]
        self.assertNotIn("UPDATE", grant_section)
        self.assertNotIn("DELETE", grant_section)
        self.assertIn("octet_length(payload_json::text) <= 262144", source)

    def test_identity_indexes_and_lifecycle_lookup_index_are_tenant_scoped(self):
        source = MIGRATION.read_text(encoding="utf-8")
        self.assertIn("ux_archive_incidents_identity_key", source)
        self.assertIn("ux_archive_incidents_idempotency_key", source)
        self.assertIn("ON archive.incidents (tenant_id, identity_key)", source)
        self.assertIn("ON archive.incidents (tenant_id, idempotency_key)", source)
        self.assertIn("ix_archive_incidents_lifecycle_updated", source)

    def test_legacy_migration_keeps_rls_indexes_and_permission(self):
        source = LEGACY_MIGRATION.read_text(encoding="utf-8")
        self.assertIn("ENABLE ROW LEVEL SECURITY", source)
        self.assertIn("FORCE ROW LEVEL SECURITY", source)
        self.assertIn("archive_incidents_tenant_isolation", source)
        self.assertIn("USING gin (channel_ids)", source)
        self.assertIn("ix_archive_incidents_time", source)
        self.assertIn("ix_archive_incidents_state_updated", source)
        self.assertIn("'incidents:manage'", source)
        self.assertIn("'admin', 'engineer', 'operator'", source)
        self.assertNotIn("thumbnail_b64", source)
        self.assertNotIn("image_bytes", source)


if __name__ == "__main__":
    unittest.main()
