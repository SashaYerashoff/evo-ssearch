from __future__ import annotations

import unittest
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from security.postgres_throttling import PostgresLoginThrottleRepository
from security.throttling import LoginThrottleRecord


class Result:
    def __init__(self, *, row=None, rowcount=0) -> None:
        self._row = row
        self.rowcount = rowcount

    def fetchone(self):
        return self._row


class ScriptedConnection:
    def __init__(self, steps) -> None:
        self.steps = list(steps)
        self.executions = []

    def execute(self, sql, params=None):
        compact = " ".join(str(sql).split())
        self.executions.append((compact, params))
        if not self.steps:
            raise AssertionError(f"unexpected SQL: {compact}")
        expected, result = self.steps.pop(0)
        if expected not in compact:
            raise AssertionError(f"expected {expected!r}, got {compact!r}")
        return result

    def assert_finished(self) -> None:
        if self.steps:
            raise AssertionError(f"unconsumed SQL steps: {self.steps!r}")


class FakePool:
    def __init__(self, *connections) -> None:
        self.connections = list(connections)
        self.contexts = []
        self.readonly = []

    @contextmanager
    def transaction(self, context=None, *, readonly=False):
        self.contexts.append(context)
        self.readonly.append(readonly)
        if not self.connections:
            raise AssertionError("unexpected transaction")
        yield self.connections.pop(0)


class PostgresThrottleTests(unittest.TestCase):
    def test_get_save_and_delete_use_tenant_context(self):
        tenant_id = uuid.uuid4()
        window_started = datetime.fromtimestamp(100.0, timezone.utc)
        locked_until = datetime.fromtimestamp(200.0, timezone.utc)
        get_connection = ScriptedConnection(
            [
                (
                    "FROM iam.login_attempts",
                    Result(row=(2, window_started, locked_until)),
                )
            ]
        )
        save_connection = ScriptedConnection(
            [("ON CONFLICT (tenant_id, throttle_key)", Result(rowcount=1))]
        )
        delete_connection = ScriptedConnection(
            [("DELETE FROM iam.login_attempts", Result(rowcount=1))]
        )
        pool = FakePool(get_connection, save_connection, delete_connection)
        repository = PostgresLoginThrottleRepository(pool, tenant_id)

        record = repository.get(" tenant:user:ip ")
        repository.save(
            "tenant:user:ip",
            LoginThrottleRecord(
                failed_attempts=3,
                window_started_at=300.0,
                locked_until=None,
            ),
        )
        repository.delete("tenant:user:ip")

        self.assertEqual(
            record,
            LoginThrottleRecord(
                failed_attempts=2,
                window_started_at=100.0,
                locked_until=200.0,
            ),
        )
        self.assertEqual(pool.contexts[0].tenant_id, tenant_id)
        self.assertEqual(pool.contexts[0].actor_id, uuid.UUID(int=0))
        self.assertEqual(pool.readonly, [True, False, False])
        self.assertEqual(get_connection.executions[0][1][1], "tenant:user:ip")
        self.assertIsNone(save_connection.executions[0][1][4])
        get_connection.assert_finished()
        save_connection.assert_finished()
        delete_connection.assert_finished()


if __name__ == "__main__":
    unittest.main()
