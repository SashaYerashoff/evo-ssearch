import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from security import ALL_CHANNELS, AuthContext, REDACTED
from security.postgres_audit_reader import PostgresAuditReader


TENANT_ID = "59da6ca3-51b7-4d91-9190-aae06b76d846"
ACTOR_ID = "361fe45f-f277-42f8-ae35-eaa0fc81cf38"
OTHER_ACTOR_ID = "4b3d96cc-81ce-4cfd-8d44-6f26f4195386"


class _Cursor:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows


class _Connection:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, Any]] = []

    def execute(self, query: str, params: Any = None) -> _Cursor:
        self.calls.append((query, params))
        return _Cursor(self.rows)


class _Pool:
    def __init__(self, rows: list[tuple[Any, ...]] | None = None) -> None:
        self.connection = _Connection(rows or [])
        self.transactions: list[tuple[Any, bool]] = []

    @contextmanager
    def transaction(self, context=None, *, readonly=False):
        self.transactions.append((context, readonly))
        yield self.connection


def _context() -> AuthContext:
    return AuthContext(
        user_id=ACTOR_ID,
        tenant_id=TENANT_ID,
        roles=frozenset({"admin"}),
        permissions=frozenset({"audit:view"}),
        allowed_channel_ids=frozenset({ALL_CHANNELS}),
        request_id="request-42",
    )


def _row(
    sequence_number: int,
    occurred_at: datetime,
    *,
    details: Any = None,
) -> tuple[Any, ...]:
    return (
        f"00000000-0000-0000-0000-{sequence_number:012d}",
        sequence_number,
        TENANT_ID,
        occurred_at,
        "request-42",
        ACTOR_ID,
        ["admin"],
        "192.0.2.10",
        "auth.login",
        "session",
        "session-1",
        7,
        "success",
        details if details is not None else {"rows": 3},
    )


class PostgresAuditReaderTests(unittest.TestCase):
    def test_lists_events_with_readonly_transaction_and_redacted_details(self) -> None:
        rows = [
            _row(12, datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc), details={"token": "secret", "safe": "visible"}),
            _row(11, datetime(2026, 6, 10, 11, 59, tzinfo=timezone.utc)),
            _row(10, datetime(2026, 6, 10, 11, 58, tzinfo=timezone.utc)),
        ]
        pool = _Pool(rows)

        page = PostgresAuditReader(pool).list_events(_context(), limit=2)

        self.assertEqual(len(page.events), 2)
        self.assertIsNotNone(page.next_cursor)
        self.assertEqual(page.events[0].details, {"safe": "visible", "token": REDACTED})
        self.assertEqual(len(pool.transactions), 1)
        tx_context, readonly = pool.transactions[0]
        self.assertTrue(readonly)
        self.assertEqual(tx_context.tenant_id, TENANT_ID)
        self.assertEqual(tx_context.actor_id, ACTOR_ID)
        query, params = pool.connection.calls[0]
        self.assertIn("FROM audit.events", query)
        self.assertIn("safe_details", query)
        self.assertIn("ORDER BY occurred_at DESC, sequence_number DESC", query)
        self.assertEqual(params[0], TENANT_ID)
        self.assertEqual(params[-1], 3)

    def test_filters_are_allow_listed_and_parameterized(self) -> None:
        pool = _Pool()

        PostgresAuditReader(pool).list_events(
            _context(),
            limit="25",
            since="2026-06-10T10:00:00Z",
            until="2026-06-10T11:00:00+00:00",
            actor_user_id=OTHER_ACTOR_ID,
            action="auth.login",
            target_type="session",
            target_id="session-1",
            channel_id="7",
            result="success",
            request_id="request-99",
        )

        query, params = pool.connection.calls[0]
        for fragment in (
            "tenant_id = %s",
            "occurred_at >= %s",
            "occurred_at <= %s",
            "actor_user_id = %s",
            "channel_id = %s",
            "result = %s",
            "action = %s",
            "target_type = %s",
            "target_id = %s",
            "request_id = %s",
        ):
            self.assertIn(fragment, query)
        self.assertIn(OTHER_ACTOR_ID, params)
        self.assertIn(7, params)
        self.assertIn("success", params)
        self.assertEqual(params[-1], 26)

    def test_cursor_adds_keyset_predicate(self) -> None:
        first_pool = _Pool([
            _row(12, datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc)),
            _row(11, datetime(2026, 6, 10, 11, 59, tzinfo=timezone.utc)),
        ])
        cursor = PostgresAuditReader(first_pool).list_events(_context(), limit=1).next_cursor
        self.assertIsNotNone(cursor)

        second_pool = _Pool()
        PostgresAuditReader(second_pool).list_events(_context(), limit=1, cursor=cursor)

        query, params = second_pool.connection.calls[0]
        self.assertIn("(occurred_at, sequence_number) < (%s, %s)", query)
        self.assertEqual(params[-1], 2)
        self.assertEqual(params[-2], 12)

    def test_invalid_inputs_fail_before_transaction(self) -> None:
        invalid_cases = [
            {"limit": "101"},
            {"cursor": "not-base64"},
            {"since": "2026-06-10T10:00:00"},
            {"actor_user_id": "not-a-uuid"},
            {"channel_id": "07"},
            {"result": "maybe"},
            {"request_id": "x" * 129},
        ]
        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs):
                pool = _Pool()
                with self.assertRaises(ValueError):
                    PostgresAuditReader(pool).list_events(_context(), **kwargs)
                self.assertEqual(pool.transactions, [])


if __name__ == "__main__":
    unittest.main()
