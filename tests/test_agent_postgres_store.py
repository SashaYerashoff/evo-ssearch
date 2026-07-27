import copy
import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
import uuid

from agent_postgres_store import PostgresAgentStore, record_agent_tool_run_audit
from agent_security.audit import ToolAuditEvent


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


class _ToolRunConnection:
    def __init__(self):
        self.calls = []

    def execute(self, query, params=()):
        normalized = " ".join(str(query).split())
        self.calls.append((normalized, params))
        if normalized.startswith("UPDATE agent.tool_runs"):
            return _Cursor(("run-1",))
        return _Cursor(None)


class _ToolRunPool:
    def __init__(self):
        self.connection = _ToolRunConnection()
        self.contexts = []

    @contextmanager
    def transaction(self, context=None, *, readonly=False):
        self.contexts.append(context)
        yield self.connection


class AgentToolRunAuditTests(unittest.TestCase):
    def _event(self, phase, **overrides):
        values = {
            "timestamp": datetime(2026, 7, 26, 7, 30, tzinfo=timezone.utc),
            "phase": phase,
            "operation": "execute",
            "tool_name": "list_video_summary_channels",
            "actor_id": ACTOR_ID,
            "tenant_id": TENANT_ID,
            "request_id": "request-42",
            "session_id": SESSION_ID,
            "actor_roles": ("operator",),
            "risk": "read",
            "required_permission": "streams:view",
            "arguments_hash": "ab" * 32,
            "duration_ms": 12.6,
        }
        values.update(overrides)
        return ToolAuditEvent(**values)

    def test_allow_and_result_are_projected_to_one_queryable_tool_run(self):
        pool = _ToolRunPool()

        record_agent_tool_run_audit(pool, self._event("allow"), uuid.uuid4())
        record_agent_tool_run_audit(pool, self._event("result"), uuid.uuid4())

        insert = next(
            call for call in pool.connection.calls
            if call[0].startswith("INSERT INTO agent.tool_runs")
        )
        update = next(
            call for call in pool.connection.calls
            if call[0].startswith("UPDATE agent.tool_runs")
        )
        self.assertEqual(insert[1][2], SESSION_ID)
        self.assertEqual(insert[1][5], "list_video_summary_channels")
        self.assertEqual(insert[1][8], "allow")
        self.assertEqual(insert[1][14], False)
        self.assertEqual(update[1][0], 13)
        self.assertEqual(update[1][1], "success")
        self.assertEqual(update[1][6], SESSION_ID)
        self.assertEqual(len(pool.contexts), 2)
        self.assertEqual(pool.contexts[0].agent_session_id, SESSION_ID)

    def test_denial_is_recorded_as_a_finished_denied_run(self):
        pool = _ToolRunPool()

        record_agent_tool_run_audit(
            pool,
            self._event("deny", code="permission_denied"),
            uuid.uuid4(),
        )

        insert = pool.connection.calls[0]
        self.assertEqual(insert[1][8], "deny")
        self.assertEqual(insert[1][10], "permission_denied")
        self.assertEqual(insert[1][14], True)

    def test_events_without_agent_session_remain_in_audit_only(self):
        pool = _ToolRunPool()

        record_agent_tool_run_audit(
            pool,
            self._event("allow", session_id=None),
            uuid.uuid4(),
        )

        self.assertEqual(pool.connection.calls, [])
        self.assertEqual(pool.contexts, [])


if __name__ == "__main__":
    unittest.main()
