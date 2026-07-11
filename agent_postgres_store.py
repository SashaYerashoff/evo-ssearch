"""PostgreSQL-backed agent chat session store."""

from __future__ import annotations

import copy
import json
import threading
import uuid
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

from eva_db import PsycopgPool, TransactionContext


def _jsonb(value: Any) -> Any:
    from psycopg.types.json import Jsonb

    return Jsonb(_plain_value(value))


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_plain_value(item) for item in sorted(value, key=repr)]
    return value


def _uuid_text(value: str | uuid.UUID | None, field_name: str) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except Exception as exc:
        raise ValueError(f"{field_name} is required and must be a UUID") from exc


class PostgresAgentStore:
    """AgentStore-compatible implementation using the secure `agent` schema."""

    backend = "postgres"
    RESEARCH_STATE_KEY = "research_state"
    MAX_RESEARCH_STATE_BYTES = 64_000

    def __init__(
        self,
        pool: PsycopgPool,
        *,
        max_sessions: int,
        max_messages_per_session: int,
        session_ttl_days: int,
    ) -> None:
        self.pool = pool
        self.max_sessions = max(10, int(max_sessions))
        self.max_msg = max(20, int(max_messages_per_session))
        self.ttl_ms = int(session_ttl_days) * 86_400_000
        self._lock = threading.RLock()

    def _context(
        self,
        *,
        tenant_id: str | None,
        actor_id: str | None,
        session_id: str | None = None,
    ) -> TransactionContext:
        return TransactionContext(
            tenant_id=_uuid_text(tenant_id, "tenant_id"),
            actor_id=_uuid_text(actor_id, "actor_id"),
            agent_session_id=session_id,
        )

    def create_session(
        self,
        *,
        tenant_id: Optional[str] = None,
        actor_id: Optional[str] = None,
    ) -> str:
        sid = str(uuid.uuid4())
        context = self._context(tenant_id=tenant_id, actor_id=actor_id, session_id=sid)
        with self._lock:
            with self.pool.transaction(context) as connection:
                self._gc_owner_locked(connection, context.tenant_id, context.actor_id)
                connection.execute(
                    """
                    INSERT INTO agent.sessions (
                        id,
                        tenant_id,
                        user_id,
                        status,
                        metadata
                    )
                    VALUES (%s, %s, %s, 'active', '{}'::jsonb)
                    """,
                    (sid, context.tenant_id, context.actor_id),
                )
        return sid

    def touch_session(
        self,
        session_id: str,
        title: Optional[str] = None,
        *,
        tenant_id: Optional[str] = None,
        actor_id: Optional[str] = None,
    ) -> None:
        context = self._context(tenant_id=tenant_id, actor_id=actor_id, session_id=session_id)
        with self._lock:
            with self.pool.transaction(context) as connection:
                row = connection.execute(
                    """
                    UPDATE agent.sessions
                    SET updated_at = clock_timestamp(),
                        title = COALESCE(title, %s)
                    WHERE tenant_id = %s
                      AND user_id = %s
                      AND id = %s
                      AND status = 'active'
                    RETURNING id
                    """,
                    (title, context.tenant_id, context.actor_id, session_id),
                ).fetchone()
                if row is None:
                    raise KeyError("agent session not found")

    @classmethod
    def _validated_research_state(cls, state: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(state, Mapping):
            raise TypeError("research state must be an object")
        payload = _plain_value(dict(state))
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > cls.MAX_RESEARCH_STATE_BYTES:
            raise ValueError(
                f"research state exceeds {cls.MAX_RESEARCH_STATE_BYTES} bytes"
            )
        return payload

    def load_research_state(
        self,
        session_id: str,
        *,
        tenant_id: Optional[str] = None,
        actor_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the trusted server-side continuation ledger for a chat session."""

        context = self._context(
            tenant_id=tenant_id,
            actor_id=actor_id,
            session_id=session_id,
        )
        with self.pool.transaction(context, readonly=True) as connection:
            row = connection.execute(
                """
                SELECT metadata -> %s
                FROM agent.sessions
                WHERE tenant_id = %s
                  AND user_id = %s
                  AND id = %s
                  AND status = 'active'
                """,
                (
                    self.RESEARCH_STATE_KEY,
                    context.tenant_id,
                    context.actor_id,
                    session_id,
                ),
            ).fetchone()
        if row is None or not isinstance(row[0], Mapping):
            return None
        return copy.deepcopy(dict(row[0]))

    def save_research_state(
        self,
        session_id: str,
        state: Mapping[str, Any],
        *,
        tenant_id: Optional[str] = None,
        actor_id: Optional[str] = None,
    ) -> None:
        """Persist a bounded continuation ledger without adding model-visible chat."""

        payload = self._validated_research_state(state)
        context = self._context(
            tenant_id=tenant_id,
            actor_id=actor_id,
            session_id=session_id,
        )
        with self._lock:
            with self.pool.transaction(context) as connection:
                row = connection.execute(
                    """
                    UPDATE agent.sessions
                    SET metadata = jsonb_set(
                            metadata,
                            ARRAY[%s]::text[],
                            %s,
                            true
                        ),
                        updated_at = clock_timestamp()
                    WHERE tenant_id = %s
                      AND user_id = %s
                      AND id = %s
                      AND status = 'active'
                    RETURNING id
                    """,
                    (
                        self.RESEARCH_STATE_KEY,
                        _jsonb(payload),
                        context.tenant_id,
                        context.actor_id,
                        session_id,
                    ),
                ).fetchone()
                if row is None:
                    raise KeyError("agent session not found")

    def clear_research_state(
        self,
        session_id: str,
        *,
        tenant_id: Optional[str] = None,
        actor_id: Optional[str] = None,
    ) -> None:
        context = self._context(
            tenant_id=tenant_id,
            actor_id=actor_id,
            session_id=session_id,
        )
        with self._lock:
            with self.pool.transaction(context) as connection:
                row = connection.execute(
                    """
                    UPDATE agent.sessions
                    SET metadata = metadata - %s,
                        updated_at = clock_timestamp()
                    WHERE tenant_id = %s
                      AND user_id = %s
                      AND id = %s
                      AND status = 'active'
                    RETURNING id
                    """,
                    (
                        self.RESEARCH_STATE_KEY,
                        context.tenant_id,
                        context.actor_id,
                        session_id,
                    ),
                ).fetchone()
                if row is None:
                    raise KeyError("agent session not found")

    def session_exists(
        self,
        session_id: str,
        *,
        tenant_id: Optional[str] = None,
        actor_id: Optional[str] = None,
    ) -> bool:
        context = self._context(tenant_id=tenant_id, actor_id=actor_id, session_id=session_id)
        with self.pool.transaction(context, readonly=True) as connection:
            row = connection.execute(
                """
                SELECT id
                FROM agent.sessions
                WHERE tenant_id = %s
                  AND user_id = %s
                  AND id = %s
                  AND status = 'active'
                """,
                (context.tenant_id, context.actor_id, session_id),
            ).fetchone()
        return row is not None

    def list_sessions(
        self,
        *,
        tenant_id: Optional[str] = None,
        actor_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        context = self._context(tenant_id=tenant_id, actor_id=actor_id)
        with self.pool.transaction(context, readonly=True) as connection:
            rows = connection.execute(
                """
                SELECT
                    s.id::text,
                    s.title,
                    (EXTRACT(EPOCH FROM s.created_at) * 1000)::bigint AS created_ms,
                    (EXTRACT(EPOCH FROM s.updated_at) * 1000)::bigint AS updated_ms,
                    COUNT(m.id)::bigint AS message_count
                FROM agent.sessions s
                LEFT JOIN agent.messages m
                  ON m.tenant_id = s.tenant_id
                 AND m.session_id = s.id
                WHERE s.tenant_id = %s
                  AND s.user_id = %s
                  AND s.status = 'active'
                GROUP BY s.id, s.title, s.created_at, s.updated_at
                ORDER BY s.updated_at DESC
                """,
                (context.tenant_id, context.actor_id),
            ).fetchall()
        return [
            {
                "id": str(row[0]),
                "title": row[1],
                "created_at": int(row[2] or 0),
                "updated_at": int(row[3] or 0),
                "message_count": int(row[4] or 0),
            }
            for row in rows
        ]

    def get_session(
        self,
        session_id: str,
        *,
        tenant_id: Optional[str] = None,
        actor_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        context = self._context(tenant_id=tenant_id, actor_id=actor_id, session_id=session_id)
        with self.pool.transaction(context, readonly=True) as connection:
            row = connection.execute(
                """
                SELECT
                    id::text,
                    title,
                    (EXTRACT(EPOCH FROM created_at) * 1000)::bigint AS created_ms,
                    (EXTRACT(EPOCH FROM updated_at) * 1000)::bigint AS updated_ms
                FROM agent.sessions
                WHERE tenant_id = %s
                  AND user_id = %s
                  AND id = %s
                  AND status = 'active'
                """,
                (context.tenant_id, context.actor_id, session_id),
            ).fetchone()
            if row is None:
                return None
            messages = connection.execute(
                """
                SELECT
                    sequence_number,
                    role,
                    content,
                    tool_call_id,
                    metadata,
                    (EXTRACT(EPOCH FROM created_at) * 1000)::bigint AS created_ms
                FROM agent.messages
                WHERE tenant_id = %s AND session_id = %s
                ORDER BY sequence_number ASC
                """,
                (context.tenant_id, session_id),
            ).fetchall()
        return {
            "session_id": str(row[0]),
            "title": row[1],
            "created_at": int(row[2] or 0),
            "updated_at": int(row[3] or 0),
            "messages": [self._message_row_to_dict(item) for item in messages],
        }

    def delete_session(
        self,
        session_id: str,
        *,
        tenant_id: Optional[str] = None,
        actor_id: Optional[str] = None,
    ) -> bool:
        context = self._context(tenant_id=tenant_id, actor_id=actor_id, session_id=session_id)
        with self._lock:
            with self.pool.transaction(context) as connection:
                cursor = connection.execute(
                    """
                    UPDATE agent.sessions
                    SET status = 'deleted',
                        deleted_at = clock_timestamp(),
                        updated_at = clock_timestamp()
                    WHERE tenant_id = %s
                      AND user_id = %s
                      AND id = %s
                      AND status = 'active'
                    """,
                    (context.tenant_id, context.actor_id, session_id),
                )
        return int(cursor.rowcount or 0) > 0

    def add_message(
        self,
        session_id: str,
        role: str,
        content: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_result: Optional[Any] = None,
        *,
        tenant_id: Optional[str] = None,
        actor_id: Optional[str] = None,
    ) -> int:
        context = self._context(tenant_id=tenant_id, actor_id=actor_id, session_id=session_id)
        metadata: Dict[str, Any] = {}
        if tool_calls:
            metadata["tool_calls"] = tool_calls
        if tool_name:
            metadata["tool_name"] = tool_name
        if tool_result is not None:
            metadata["tool_result"] = tool_result
        with self._lock:
            with self.pool.transaction(context) as connection:
                row = connection.execute(
                    """
                    SELECT COALESCE(MAX(sequence_number), -1) + 1
                    FROM agent.messages
                    WHERE tenant_id = %s AND session_id = %s
                    """,
                    (context.tenant_id, session_id),
                ).fetchone()
                sequence = int(row[0] or 0)
                connection.execute(
                    """
                    INSERT INTO agent.messages (
                        id,
                        tenant_id,
                        session_id,
                        sequence_number,
                        role,
                        content,
                        tool_call_id,
                        metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(uuid.uuid4()),
                        context.tenant_id,
                        session_id,
                        sequence,
                        str(role or "user"),
                        str(content or ""),
                        tool_call_id,
                        _jsonb(metadata),
                    ),
                )
                self._prune_messages_locked(connection, context.tenant_id, session_id)
                connection.execute(
                    """
                    UPDATE agent.sessions
                    SET updated_at = clock_timestamp()
                    WHERE tenant_id = %s AND user_id = %s AND id = %s
                    """,
                    (context.tenant_id, context.actor_id, session_id),
                )
        return sequence

    def load_history(
        self,
        session_id: str,
        *,
        tenant_id: Optional[str] = None,
        actor_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        context = self._context(tenant_id=tenant_id, actor_id=actor_id, session_id=session_id)
        with self.pool.transaction(context, readonly=True) as connection:
            rows = connection.execute(
                """
                SELECT role, content, metadata
                FROM agent.messages
                WHERE tenant_id = %s AND session_id = %s
                ORDER BY sequence_number DESC
                LIMIT %s
                """,
                (context.tenant_id, session_id, 80),
            ).fetchall()
        rows = list(reversed(rows))
        messages: List[Dict[str, Any]] = []
        for row in rows:
            role = str(row[0] or "")
            content = str(row[1] or "").strip()
            trusted_system_receipt = (
                role == "system"
                and content.startswith("Trusted server action receipt:")
            )
            if role not in {"user", "assistant"} and not trusted_system_receipt:
                continue
            if not content:
                continue
            if trusted_system_receipt:
                messages.append({"role": "system", "content": content})
                continue
            if messages and str(messages[-1].get("role") or "") == role:
                prev_content = str(messages[-1].get("content") or "")
                if prev_content.strip() == content:
                    continue
                messages[-1]["content"] = f"{prev_content}\n\n{content}".strip()
                continue
            messages.append({"role": role, "content": content})
        messages = messages[-20:]
        while messages and messages[0].get("role") != "user":
            messages.pop(0)
        return messages

    def _gc_owner_locked(self, connection: Any, tenant_id: str, actor_id: str) -> None:
        connection.execute(
            """
            DELETE FROM agent.sessions
            WHERE tenant_id = %s
              AND user_id = %s
              AND updated_at < (clock_timestamp() - make_interval(secs => %s))
            """,
            (tenant_id, actor_id, self.ttl_ms / 1000.0),
        )
        connection.execute(
            """
            UPDATE agent.sessions
            SET status = 'deleted',
                deleted_at = COALESCE(deleted_at, clock_timestamp()),
                updated_at = clock_timestamp()
            WHERE tenant_id = %s
              AND user_id = %s
              AND status = 'active'
              AND id IN (
                  SELECT id
                  FROM agent.sessions
                  WHERE tenant_id = %s
                    AND user_id = %s
                    AND status = 'active'
                  ORDER BY updated_at DESC
                  OFFSET %s
              )
            """,
            (tenant_id, actor_id, tenant_id, actor_id, self.max_sessions),
        )

    def _prune_messages_locked(
        self,
        connection: Any,
        tenant_id: str,
        session_id: str,
    ) -> None:
        connection.execute(
            """
            DELETE FROM agent.messages
            WHERE tenant_id = %s
              AND session_id = %s
              AND sequence_number IN (
                  SELECT sequence_number
                  FROM agent.messages
                  WHERE tenant_id = %s AND session_id = %s
                  ORDER BY sequence_number DESC
                  OFFSET %s
              )
            """,
            (tenant_id, session_id, tenant_id, session_id, self.max_msg),
        )

    @staticmethod
    def _message_row_to_dict(row: Any) -> Dict[str, Any]:
        metadata = row[4] if isinstance(row[4], dict) else {}
        payload: Dict[str, Any] = {
            "id": int(row[0] or 0),
            "role": str(row[1] or ""),
            "content": str(row[2] or ""),
            "tool_call_id": row[3],
            "created_at": int(row[5] or 0),
        }
        if isinstance(metadata, dict):
            if "tool_calls" in metadata:
                payload["tool_calls"] = metadata["tool_calls"]
            if "tool_name" in metadata:
                payload["tool_name"] = metadata["tool_name"]
            if "tool_result" in metadata:
                payload["tool_result"] = metadata["tool_result"]
        return payload
