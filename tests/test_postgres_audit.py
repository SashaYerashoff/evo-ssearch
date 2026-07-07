from __future__ import annotations

import json
import os
import unittest
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from security import REDACTED, AuditEvent
from security.postgres_audit import PostgresAuditWriter


TENANT_ID = "59da6ca3-51b7-4d91-9190-aae06b76d846"
ACTOR_ID = "361fe45f-f277-42f8-ae35-eaa0fc81cf38"


def make_event(**overrides: Any) -> AuditEvent:
    values = {
        "timestamp": datetime(2026, 6, 9, 12, 30, tzinfo=timezone.utc),
        "request_id": "request-42",
        "actor_user_id": ACTOR_ID,
        "actor_roles": ("operator",),
        "tenant_id": TENANT_ID,
        "source_ip": "192.0.2.25",
        "action": "detections.read",
        "target_type": "channel",
        "target_id": "camera-17",
        "channel_id": 17,
        "result": "success",
        "details": {"rows": 3},
    }
    values.update(overrides)
    return AuditEvent(**values)


class _Cursor:
    def __init__(self, row: tuple[Any, ...] | None = None) -> None:
        self._row = row

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._row


class _Connection:
    def __init__(self, *, fail_insert: Exception | None = None) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.fail_insert = fail_insert

    def execute(self, query: str, params: Any = None) -> _Cursor:
        self.calls.append((query, params))
        if "INSERT INTO audit.events" not in query:
            return _Cursor()
        if self.fail_insert is not None:
            raise self.fail_insert
        return _Cursor((params[0],))


class _Pool:
    def __init__(self, connection: _Connection | None = None) -> None:
        self.connection = connection or _Connection()
        self.transactions = 0

    @contextmanager
    def transaction(self, context=None, *, readonly=False):
        self.transactions += 1
        yield self.connection


class PostgresAuditWriterTests(unittest.TestCase):
    def test_callable_inserts_all_event_fields_and_returns_uuid(self) -> None:
        pool = _Pool()
        event = make_event()

        event_id = PostgresAuditWriter(pool)(event)

        self.assertIsInstance(event_id, uuid.UUID)
        self.assertEqual(pool.transactions, 1)
        context_params = pool.connection.calls[0][1]
        self.assertEqual(context_params, (TENANT_ID, ACTOR_ID, "request-42"))

        insert_params = pool.connection.calls[1][1]
        self.assertEqual(insert_params[0], event_id)
        self.assertEqual(insert_params[1], uuid.UUID(TENANT_ID))
        self.assertEqual(insert_params[2], event.timestamp)
        self.assertEqual(insert_params[3], event.request_id)
        self.assertEqual(insert_params[4], uuid.UUID(ACTOR_ID))
        self.assertEqual(insert_params[5], ["operator"])
        self.assertEqual(insert_params[6], event.source_ip)
        self.assertEqual(insert_params[7], event.action)
        self.assertEqual(insert_params[8], event.target_type)
        self.assertEqual(insert_params[9], event.target_id)
        self.assertEqual(insert_params[10], event.channel_id)
        self.assertEqual(insert_params[11], "success")
        self.assertEqual(json.loads(insert_params[12]), {"rows": 3})

    def test_result_aliases_are_normalized_and_unknown_values_fail_closed(self) -> None:
        aliases = {
            "ok": "success",
            "allow": "success",
            "result": "success",
            "error": "failure",
            "failed": "failure",
            "deny": "denied",
            "forbidden": "denied",
        }
        for alias, expected in aliases.items():
            with self.subTest(alias=alias):
                pool = _Pool()
                PostgresAuditWriter(pool).write(make_event(result=alias))
                self.assertEqual(pool.connection.calls[1][1][11], expected)

        pool = _Pool()
        with self.assertRaisesRegex(ValueError, "unsupported audit result"):
            PostgresAuditWriter(pool).write(make_event(result="maybe"))
        self.assertEqual(pool.transactions, 0)

    def test_requires_customer_tenant_before_opening_transaction(self) -> None:
        for tenant_id in (None, "", "not-a-uuid"):
            with self.subTest(tenant_id=tenant_id):
                pool = _Pool()
                with self.assertRaisesRegex(ValueError, "tenant_id"):
                    PostgresAuditWriter(pool).write(
                        make_event(tenant_id=tenant_id)
                    )
                self.assertEqual(pool.transactions, 0)

    def test_rejects_non_positive_and_non_integer_channel_ids(self) -> None:
        for channel_id in (0, -1, "17", 17.0, True):
            with self.subTest(channel_id=channel_id):
                pool = _Pool()
                with self.assertRaisesRegex(ValueError, "channel_id"):
                    PostgresAuditWriter(pool).write(
                        make_event(channel_id=channel_id)
                    )
                self.assertEqual(pool.transactions, 0)

    def test_redacts_untrusted_details_before_json_serialization(self) -> None:
        pool = _Pool()
        details = {
            "token": "bearer-secret",
            "nested": {
                "password": "db-secret",
                "image_bytes": b"raw-frame",
                "safe": "retained",
            },
        }

        PostgresAuditWriter(pool).write(make_event(details=details))

        serialized = pool.connection.calls[1][1][12]
        self.assertNotIn("bearer-secret", serialized)
        self.assertNotIn("db-secret", serialized)
        self.assertNotIn("raw-frame", serialized)
        self.assertEqual(
            json.loads(serialized),
            {
                "nested": {
                    "image_bytes": REDACTED,
                    "password": REDACTED,
                    "safe": "retained",
                },
                "token": REDACTED,
            },
        )

    def test_database_errors_are_propagated(self) -> None:
        database_error = RuntimeError("database unavailable")
        pool = _Pool(_Connection(fail_insert=database_error))

        with self.assertRaises(RuntimeError) as raised:
            PostgresAuditWriter(pool).write(make_event())

        self.assertIs(raised.exception, database_error)

    def test_missing_returned_id_fails_closed(self) -> None:
        class NoRowConnection(_Connection):
            def execute(self, query: str, params: Any = None) -> _Cursor:
                self.calls.append((query, params))
                return _Cursor()

        with self.assertRaisesRegex(RuntimeError, "returned no event id"):
            PostgresAuditWriter(_Pool(NoRowConnection())).write(make_event())


@unittest.skipUnless(
    os.getenv("EVA_TEST_DATABASE_DSN"),
    "set EVA_TEST_DATABASE_DSN for live PostgreSQL audit tests",
)
class PostgresAuditWriterLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import psycopg

        from eva_db import DatabaseSettings, PsycopgPool

        cls.psycopg = psycopg
        cls.dsn = os.environ["EVA_TEST_DATABASE_DSN"]
        cls.pool = PsycopgPool(
            DatabaseSettings(dsn=cls.dsn, pool_min_size=0, pool_max_size=1)
        )
        cls.writer = PostgresAuditWriter(cls.pool)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.pool.close()

    def test_live_insert_and_append_only_trigger(self) -> None:
        tenant_id = str(uuid.uuid4())
        actor_id = str(uuid.uuid4())
        event = make_event(
            tenant_id=tenant_id,
            actor_user_id=actor_id,
            result="ok",
            details={"token": "secret", "safe": "visible"},
        )

        event_id = self.writer.write(event)

        with self.psycopg.connect(self.dsn) as connection:
            row = connection.execute(
                """
                SELECT tenant_id, actor_user_id, channel_id, result, safe_details
                FROM audit.events
                WHERE id = %s
                """,
                (event_id,),
            ).fetchone()
            self.assertEqual(
                row,
                (
                    uuid.UUID(tenant_id),
                    uuid.UUID(actor_id),
                    17,
                    "success",
                    {"safe": "visible", "token": REDACTED},
                ),
            )

        mutations = (
            "UPDATE audit.events SET result = 'failure' WHERE id = %s",
            "DELETE FROM audit.events WHERE id = %s",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation.split()[0]):
                with self.psycopg.connect(self.dsn) as connection:
                    with self.assertRaises(
                        self.psycopg.errors.ObjectNotInPrerequisiteState
                    ):
                        connection.execute(mutation, (event_id,))


if __name__ == "__main__":
    unittest.main()
