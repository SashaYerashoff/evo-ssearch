from __future__ import annotations

import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from incident_store import (
    INCIDENT_STORAGE_REVISION,
    IncidentRevisionConflict,
    PostgresIncidentStore,
)


ROOT = Path(__file__).resolve().parent.parent
MIGRATION = ROOT / "migrations" / "versions" / "20260801_0011_incidents.py"
TENANT_ID = "f3c3533e-bf17-46a1-a543-696d95b8cf6f"
ACTOR_ID = "85f620ba-fc37-4f67-ad1a-8fcf3d983461"
INCIDENT_ID = "384be4e4-8c4e-4aa8-941a-17807168cb8c"


def incident_row(*, revision: int = 1, title: str = "Craft near port gate"):
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
        ACTOR_ID,
        ACTOR_ID,
        now,
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


class IncidentMigrationTests(unittest.TestCase):
    def test_migration_has_single_head_rls_indexes_and_permission(self):
        source = MIGRATION.read_text(encoding="utf-8")
        self.assertIn(f'revision: str = "{INCIDENT_STORAGE_REVISION}"', source)
        self.assertIn('down_revision: str | None = "20260727_0010"', source)
        self.assertIn("CREATE TABLE archive.incidents", source)
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
