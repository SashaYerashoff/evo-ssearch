import copy
import unittest
from contextlib import contextmanager

from agent_postgres_store import PostgresAgentStore


TENANT_ID = "59da6ca3-51b7-4d91-9190-aae06b76d846"
ACTOR_ID = "361fe45f-f277-42f8-ae35-eaa0fc81cf38"
SESSION_ID = "5fe748ee-1e4d-4afd-aa67-2e80bc1b5d76"


class _Cursor:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _Connection:
    def __init__(self):
        self.research_state = None
        self.calls = []

    def execute(self, query, params=()):
        normalized = " ".join(str(query).split())
        self.calls.append((normalized, params))
        if normalized.startswith("SELECT metadata ->"):
            return _Cursor(
                (copy.deepcopy(self.research_state),)
                if self.research_state is not None
                else (None,)
            )
        if "jsonb_set" in normalized:
            self.research_state = copy.deepcopy(params[1].obj)
            return _Cursor((SESSION_ID,))
        if "metadata = metadata -" in normalized:
            self.research_state = None
            return _Cursor((SESSION_ID,))
        raise AssertionError(f"unexpected SQL: {normalized}")


class _Pool:
    def __init__(self):
        self.connection = _Connection()
        self.transactions = []

    @contextmanager
    def transaction(self, context=None, *, readonly=False):
        self.transactions.append((context, readonly))
        yield self.connection


class PostgresAgentResearchStateTests(unittest.TestCase):
    def setUp(self):
        self.pool = _Pool()
        self.store = PostgresAgentStore(
            self.pool,
            max_sessions=100,
            max_messages_per_session=200,
            session_ttl_days=30,
        )
        self.owner = {"tenant_id": TENANT_ID, "actor_id": ACTOR_ID}

    def test_research_state_round_trip_is_owner_scoped_and_copied(self):
        state = {
            "version": 1,
            "window": {"from_ts": 100.0, "to_ts": 200.0},
            "completed_channel_ids": [112],
            "remaining_channel_ids": [118, 120],
        }

        self.store.save_research_state(SESSION_ID, state, **self.owner)
        loaded = self.store.load_research_state(SESSION_ID, **self.owner)

        self.assertEqual(loaded, state)
        self.assertIsNot(loaded, self.pool.connection.research_state)
        loaded["remaining_channel_ids"].append(999)
        self.assertEqual(
            self.pool.connection.research_state["remaining_channel_ids"],
            [118, 120],
        )
        self.assertFalse(self.pool.transactions[0][1])
        self.assertTrue(self.pool.transactions[1][1])
        update_params = self.pool.connection.calls[0][1]
        self.assertEqual(update_params[0], "research_state")
        self.assertEqual(update_params[2:], (TENANT_ID, ACTOR_ID, SESSION_ID))

    def test_clear_research_state_removes_only_server_ledger(self):
        self.pool.connection.research_state = {"version": 1}

        self.store.clear_research_state(SESSION_ID, **self.owner)

        self.assertIsNone(self.pool.connection.research_state)
        params = self.pool.connection.calls[-1][1]
        self.assertEqual(params, ("research_state", TENANT_ID, ACTOR_ID, SESSION_ID))

    def test_research_state_is_bounded(self):
        oversized = {"remaining_channel_ids": ["x" * 65_000]}

        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.store.save_research_state(SESSION_ID, oversized, **self.owner)

        self.assertEqual(self.pool.connection.calls, [])


if __name__ == "__main__":
    unittest.main()
