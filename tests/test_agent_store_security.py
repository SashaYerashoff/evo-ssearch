import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from agent import AgentStore


class AgentStoreSecurityTests(unittest.TestCase):
    def test_existing_store_is_migrated_and_sessions_are_owner_scoped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "agent.sqlite3"
            connection = sqlite3.connect(path)
            connection.execute(
                """
                CREATE TABLE agent_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE agent_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL
                        REFERENCES agent_sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT,
                    tool_calls TEXT,
                    tool_call_id TEXT,
                    tool_name TEXT,
                    tool_result TEXT,
                    created_at INTEGER NOT NULL
                )
                """
            )
            now_ms = int(time.time() * 1000)
            connection.execute(
                """
                INSERT INTO agent_sessions (
                    id, title, created_at, updated_at
                )
                VALUES ('legacy', 'Legacy', ?, ?)
                """,
                (now_ms, now_ms),
            )
            connection.commit()
            connection.close()

            store = AgentStore(path=str(path))
            alice = store.create_session(
                tenant_id="tenant-1",
                actor_id="alice",
            )
            bob = store.create_session(
                tenant_id="tenant-1",
                actor_id="bob",
            )
            store.add_message(alice, role="user", content="alice secret")
            store.add_message(bob, role="user", content="bob secret")

            self.assertTrue(
                store.session_exists(
                    alice,
                    tenant_id="tenant-1",
                    actor_id="alice",
                )
            )
            self.assertFalse(
                store.session_exists(
                    alice,
                    tenant_id="tenant-1",
                    actor_id="bob",
                )
            )
            self.assertIsNone(
                store.get_session(
                    "legacy",
                    tenant_id="tenant-1",
                    actor_id="alice",
                )
            )
            self.assertEqual(
                [
                    item["id"]
                    for item in store.list_sessions(
                        tenant_id="tenant-1",
                        actor_id="alice",
                    )
                ],
                [alice],
            )
            self.assertFalse(
                store.delete_session(
                    alice,
                    tenant_id="tenant-1",
                    actor_id="bob",
                )
            )
            self.assertTrue(
                store.delete_session(
                    alice,
                    tenant_id="tenant-1",
                    actor_id="alice",
                )
            )
            self.assertEqual(
                {item["id"] for item in store.list_sessions()},
                {"legacy", bob},
            )


if __name__ == "__main__":
    unittest.main()
