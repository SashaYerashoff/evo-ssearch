"""
agent.py — AI agent backend for evo-ssearch.

Provides:
  AgentStore   — SQLite session/message persistence
  AgentTools   — tool dispatcher (one method per tool)
  AgentRunner  — tool-calling loop + SSE streaming generator
  _AgentLMClient — LM Studio client (tools + streaming), separate from _call_lm_chat

Instantiated once in oldapp.py after all globals are ready, injected with
callables so this module never imports from oldapp.py.
"""

from __future__ import annotations

import base64
import collections
import copy
import json
import sqlite3
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_MAX_HISTORY_MESSAGES = 20       # last N messages kept as context
AGENT_MAX_TOOL_ITERATIONS  = 8        # runaway loop guard
AGENT_HEARTBEAT_INTERVAL   = 15       # seconds between SSE heartbeats
AGENT_SESSION_TTL_DAYS     = 30       # sessions older than this are GC'd
AGENT_MAX_SESSIONS         = 100      # sessions kept per store (GC oldest)
AGENT_MAX_MESSAGES_PER_SESSION = 200  # messages kept per session (prune oldest)

_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_archive",
            "description": (
                "Semantic text search over either an indexed folder (CLIP/FAISS) or "
                "the detections archive (vector search over stored detections). "
                "Returns ranked results with file paths, similarity scores, and metadata. "
                "TODO v1.1: add image_path/detection_id params for image-similarity search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query, e.g. 'person in red jacket near door'.",
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["indexed_folder", "detections"],
                        "description": (
                            "indexed_folder: searches a FAISS-indexed image directory. "
                            "detections: searches the detections archive via stored CLIP vectors."
                        ),
                    },
                    "folder": {
                        "type": "string",
                        "description": "Required when scope=indexed_folder. Absolute path to an indexed folder.",
                    },
                    "probe_id": {
                        "type": "string",
                        "description": "Optional. When scope=detections, restrict to detections from this probe.",
                    },
                    "channel_id": {
                        "type": "integer",
                        "description": "Optional. Restrict detections search to this channel.",
                    },
                    "since_hours": {
                        "type": "number",
                        "description": "Only include results from the past N hours. Default: 24.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return. Default: 12, max: 48.",
                    },
                },
                "required": ["query", "scope"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_detections",
            "description": (
                "Retrieve recent detection events from the archive for one or more probes. "
                "Returns a list with timestamps, confidence scores, and image paths. "
                "Use to inspect what a probe has been triggering on."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "probe_id": {
                        "type": "string",
                        "description": "Probe ID to filter by. If omitted, returns detections across all probes.",
                    },
                    "probe_name": {
                        "type": "string",
                        "description": "Human-readable probe name. Resolved to probe_id internally.",
                    },
                    "channel_id": {
                        "type": "integer",
                        "description": "Optional. Restrict to detections from this channel.",
                    },
                    "since_hours": {
                        "type": "number",
                        "description": "Fetch detections from the past N hours. Default: 24.",
                    },
                    "until_hours": {
                        "type": "number",
                        "description": "Optional upper bound: detections older than N hours ago.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max detections to return. Default: 20, max: 100.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_detection_summary",
            "description": (
                "Aggregate detection statistics grouped by probe. Returns hit counts and "
                "latest activity timestamps. Use for an overview before drilling into specific events."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "since_hours": {
                        "type": "number",
                        "description": "Summarize detections from the past N hours. Default: 24.",
                    },
                    "channel_id": {
                        "type": "integer",
                        "description": "Optional. Restrict summary to one channel.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_probe",
            "description": (
                "Modify a probe's configuration: text pairs (positives/negatives), "
                "detection thresholds, or enabled state. "
                "IMPORTANT: always call with preview=true first to show the user a diff. "
                "Only call with preview=false after the user has explicitly confirmed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "probe_name": {
                        "type": "string",
                        "description": "Human-readable name of the probe to modify.",
                    },
                    "probe_id": {
                        "type": "string",
                        "description": "Probe ID. Use if probe_name is ambiguous.",
                    },
                    "changes": {
                        "type": "object",
                        "description": "Partial update — only the fields to change. All other probe settings are preserved.",
                        "properties": {
                            "positives": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "New list of positive text descriptions (replaces current list).",
                            },
                            "negatives": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "New list of negative/exclusion text descriptions.",
                            },
                            "pos_floor": {
                                "type": "number",
                                "description": "Minimum positive similarity score to trigger (0.0–1.0).",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                            "margin_thr": {
                                "type": "number",
                                "description": "Minimum margin (pos_score - neg_score) to trigger.",
                                "minimum": 0.0,
                            },
                            "top_k": {"type": "integer", "minimum": 1},
                            "window_sec": {"type": "number", "minimum": 0.0},
                            "enabled": {"type": "boolean"},
                            "severity": {
                                "type": "string",
                                "enum": ["info", "low", "normal", "high", "critical"],
                            },
                            "bookmark_enabled": {"type": "boolean"},
                        },
                        "additionalProperties": False,
                    },
                    "preview": {
                        "type": "boolean",
                        "description": "If true, return a diff without applying. If false, apply the changes. Default: true.",
                    },
                },
                "required": ["changes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_frame",
            "description": (
                "Send a specific image frame to the vision language model for a detailed description. "
                "Use to understand what happened in a detection, or to analyze a specific image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Absolute filesystem path to the image.",
                    },
                    "detection_id": {
                        "type": "integer",
                        "description": "Detection record ID. The stored thumbnail will be used.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Optional instruction to the VLM. Defaults to a general scene description.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_bookmark",
            "description": (
                "Create a bookmark in Luxriot EVA on a camera channel, visible in the Luxriot client. "
                "Use to flag significant events for operator review. "
                "Only call when there is a concrete reason — do not create bookmarks speculatively."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "integer",
                        "description": "Luxriot channel (camera) ID to attach the bookmark to.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Short bookmark title (max 80 characters).",
                        "maxLength": 80,
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed event description (max 240 characters).",
                        "maxLength": 240,
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["info", "low", "normal", "high", "critical"],
                        "description": "Event severity level. Default: normal.",
                    },
                    "timestamp_ms": {
                        "type": "integer",
                        "description": "Optional event timestamp in milliseconds. Defaults to now.",
                    },
                },
                "required": ["channel_id", "title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": (
                "Compile a structured detection report for a time period. "
                "Aggregates per-probe statistics, highlights peak activity hours, "
                "and lists the most significant detection events. "
                "Use for daily/weekly summaries or to answer 'give me a report on this week'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "since_hours": {
                        "type": "number",
                        "description": "Report covers detections from the past N hours. Default: 24.",
                    },
                    "until_hours": {
                        "type": "number",
                        "description": "Optional upper bound. Omit for up-to-now.",
                    },
                    "channel_id": {
                        "type": "integer",
                        "description": "Optional. Restrict report to one channel.",
                    },
                    "include_probes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Probe names to include. If omitted, includes all active probes.",
                    },
                    "top_events": {
                        "type": "integer",
                        "description": "Include the N highest-margin events per probe. Default: 5.",
                    },
                },
                "required": [],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# LM client
# ---------------------------------------------------------------------------

@dataclass
class _ToolCall:
    id: str
    name: str
    args: Dict[str, Any]


@dataclass
class _LMResponse:
    content: Optional[str]
    tool_calls: List[_ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"


class _AgentLMClient:
    """
    Minimal OpenAI-compatible client for the tool-calling loop.
    Completely separate from _call_lm_chat in oldapp.py.
    """

    def __init__(self, base_url: str, model: str, api_key: str, timeout: int) -> None:
        if not base_url:
            raise ValueError("LM base URL is not configured (EVOSSEARCH_LM_BASE_URL).")
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.model    = model
        self.timeout  = timeout
        self.headers: Dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def call_with_tools(self, messages: List[Dict[str, Any]]) -> _LMResponse:
        """Blocking non-streaming call with tools. Returns parsed response."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": _TOOL_SCHEMAS,
            "tool_choice": "auto",
            "stream": False,
        }
        resp = requests.post(
            self.endpoint, json=payload, headers=self.headers, timeout=self.timeout
        )
        resp.raise_for_status()
        data   = resp.json()
        choice = data["choices"][0]
        msg    = choice.get("message", {}) or {}
        finish = choice.get("finish_reason", "stop")

        # Parse tool_calls
        tool_calls: List[_ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, KeyError, TypeError):
                args = {}
            tool_calls.append(_ToolCall(
                id=tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                name=tc["function"]["name"],
                args=args,
            ))

        # content may be None on pure tool-call turns
        raw_content = msg.get("content")
        if isinstance(raw_content, list):
            raw_content = " ".join(
                p.get("text", "") for p in raw_content
                if isinstance(p, dict) and p.get("type") == "text"
            ).strip() or None

        return _LMResponse(
            content=raw_content,
            tool_calls=tool_calls,
            finish_reason=finish,
        )

    def stream_text(self, messages: List[Dict[str, Any]]) -> Iterator[str]:
        """Streaming call without tools. Yields text delta chunks."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        with requests.post(
            self.endpoint, json=payload, headers=self.headers,
            timeout=self.timeout, stream=True
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                try:
                    chunk  = json.loads(line)
                    delta  = chunk["choices"][0]["delta"]
                    text   = delta.get("content")
                    if text:
                        yield text
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue


# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------

class AgentStore:
    """SQLite-backed session and message store."""

    def __init__(
        self,
        path: str = "agent_sessions.sqlite3",
        max_sessions: int = AGENT_MAX_SESSIONS,
        max_messages_per_session: int = AGENT_MAX_MESSAGES_PER_SESSION,
        session_ttl_days: int = AGENT_SESSION_TTL_DAYS,
    ) -> None:
        self.path = Path(path)
        self.max_sessions = max(10, int(max_sessions))
        self.max_msg = max(20, int(max_messages_per_session))
        self.ttl_ms = int(session_ttl_days) * 86_400_000
        self._lock = threading.RLock()
        self._init_db()
        self._gc()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS agent_sessions (
                        id          TEXT PRIMARY KEY,
                        title       TEXT,
                        created_at  INTEGER NOT NULL,
                        updated_at  INTEGER NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS agent_messages (
                        id              INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id      TEXT NOT NULL
                                            REFERENCES agent_sessions(id) ON DELETE CASCADE,
                        role            TEXT NOT NULL,
                        content         TEXT,
                        tool_calls      TEXT,
                        tool_call_id    TEXT,
                        tool_name       TEXT,
                        tool_result     TEXT,
                        created_at      INTEGER NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_agent_msg_session
                        ON agent_messages (session_id, id ASC);
                    CREATE INDEX IF NOT EXISTS idx_agent_sessions_updated
                        ON agent_sessions (updated_at DESC);
                """)
                conn.commit()
            finally:
                conn.close()

    # ── Session CRUD ────────────────────────────────────────────────────────

    def create_session(self) -> str:
        sid = uuid.uuid4().hex
        now = _now_ms()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO agent_sessions (id, title, created_at, updated_at) VALUES (?,?,?,?)",
                    (sid, None, now, now),
                )
                conn.commit()
            finally:
                conn.close()
        return sid

    def touch_session(self, session_id: str, title: Optional[str] = None) -> None:
        now = _now_ms()
        with self._lock:
            conn = self._connect()
            try:
                if title is not None:
                    conn.execute(
                        "UPDATE agent_sessions SET updated_at=?, title=COALESCE(title,?) WHERE id=?",
                        (now, title, session_id),
                    )
                else:
                    conn.execute(
                        "UPDATE agent_sessions SET updated_at=? WHERE id=?",
                        (now, session_id),
                    )
                conn.commit()
            finally:
                conn.close()

    def session_exists(self, session_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT id FROM agent_sessions WHERE id=?", (session_id,)
                ).fetchone()
                return row is not None
            finally:
                conn.close()

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute("""
                    SELECT s.id, s.title, s.created_at, s.updated_at,
                           COUNT(m.id) AS message_count
                    FROM agent_sessions s
                    LEFT JOIN agent_messages m ON m.session_id = s.id
                    GROUP BY s.id
                    ORDER BY s.updated_at DESC
                """).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT id, title, created_at, updated_at FROM agent_sessions WHERE id=?",
                    (session_id,),
                ).fetchone()
                if not row:
                    return None
                messages = conn.execute(
                    "SELECT * FROM agent_messages WHERE session_id=? ORDER BY id ASC",
                    (session_id,),
                ).fetchall()
                return {
                    "session_id": row["id"],
                    "title": row["title"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "messages": [_msg_row_to_dict(m) for m in messages],
                }
            finally:
                conn.close()

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "DELETE FROM agent_sessions WHERE id=?", (session_id,)
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    # ── Message CRUD ────────────────────────────────────────────────────────

    def add_message(
        self,
        session_id: str,
        role: str,
        content: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_result: Optional[Any] = None,
    ) -> int:
        now = _now_ms()
        tc_json = json.dumps(tool_calls) if tool_calls else None
        tr_json = json.dumps(tool_result, default=str) if tool_result is not None else None
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """INSERT INTO agent_messages
                       (session_id, role, content, tool_calls, tool_call_id, tool_name, tool_result, created_at)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (session_id, role, content, tc_json, tool_call_id, tool_name, tr_json, now),
                )
                msg_id = cur.lastrowid
                self._prune_messages_locked(conn, session_id)
                conn.commit()
            finally:
                conn.close()
        self.touch_session(session_id)
        return msg_id or 0

    def _prune_messages_locked(self, conn: sqlite3.Connection, session_id: str) -> None:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM agent_messages WHERE session_id=?", (session_id,)
        ).fetchone()
        total = int(row["c"]) if row else 0
        if total <= self.max_msg:
            return
        excess = total - self.max_msg
        conn.execute(
            """DELETE FROM agent_messages WHERE id IN (
               SELECT id FROM agent_messages WHERE session_id=? ORDER BY id ASC LIMIT ?
            )""",
            (session_id, excess),
        )

    def load_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Return the last AGENT_MAX_HISTORY_MESSAGES messages for this session
        as a list of OpenAI-compatible message dicts, with orphan-scan applied.
        """
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """SELECT * FROM agent_messages WHERE session_id=?
                       ORDER BY id DESC LIMIT ?""",
                    (session_id, AGENT_MAX_HISTORY_MESSAGES),
                ).fetchall()
            finally:
                conn.close()

        rows = list(reversed(rows))  # restore chronological order
        messages = [_msg_row_to_openai(r) for r in rows]

        # Orphan-scan: drop leading tool-result messages that have no
        # preceding tool-call (would confuse the model).
        while messages and messages[0].get("role") == "tool":
            messages.pop(0)

        return messages

    # ── GC ──────────────────────────────────────────────────────────────────

    def _gc(self) -> None:
        cutoff = _now_ms() - self.ttl_ms
        with self._lock:
            conn = self._connect()
            try:
                # Delete expired sessions (cascade removes their messages)
                conn.execute(
                    "DELETE FROM agent_sessions WHERE updated_at < ?", (cutoff,)
                )
                # If still too many sessions, remove the oldest ones
                conn.execute(
                    """DELETE FROM agent_sessions WHERE id IN (
                       SELECT id FROM agent_sessions
                       ORDER BY updated_at DESC
                       LIMIT -1 OFFSET ?
                    )""",
                    (self.max_sessions,),
                )
                conn.commit()
            finally:
                conn.close()


# ---------------------------------------------------------------------------
# Tool errors
# ---------------------------------------------------------------------------

class ToolError(Exception):
    """Raised by AgentTools to signal a user-facing error to the model."""


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------

class AgentTools:
    """
    Executes agent tools. All dependencies are injected at construction.
    Never imports from oldapp.py.
    """

    def __init__(
        self,
        *,
        detections_store: Any,
        probes_store: Any,
        luxriot_manager: Any,
        embed_text_fn: Callable[[str], np.ndarray],
        embed_image_fn: Callable[[Image.Image], np.ndarray],
        call_lm_fn: Callable[..., str],
        encode_jpeg_fn: Callable[..., str],
        search_indexed_folder_fn: Callable[..., List[Dict[str, Any]]],
        search_detections_fn: Callable[..., List[Dict[str, Any]]],
    ) -> None:
        self._ds   = detections_store
        self._ps   = probes_store
        self._lxm  = luxriot_manager
        self._emb_text  = embed_text_fn
        self._emb_image = embed_image_fn
        self._lm   = call_lm_fn
        self._jpeg = encode_jpeg_fn
        self._search_folder    = search_indexed_folder_fn
        self._search_det       = search_detections_fn

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch to the named tool. Returns a dict always."""
        dispatch = {
            "search_archive":       self._search_archive,
            "get_detections":       self._get_detections,
            "get_detection_summary": self._get_detection_summary,
            "update_probe":         self._update_probe,
            "describe_frame":       self._describe_frame,
            "create_bookmark":      self._create_bookmark,
            "generate_report":      self._generate_report,
        }
        fn = dispatch.get(name)
        if fn is None:
            raise ToolError(f"Unknown tool: {name!r}")
        return fn(args)

    # ── search_archive ──────────────────────────────────────────────────────

    def _search_archive(self, args: Dict[str, Any]) -> Dict[str, Any]:
        query  = str(args.get("query") or "").strip()
        scope  = str(args.get("scope") or "detections")
        limit  = max(1, min(48, int(args.get("limit") or 12)))

        if not query:
            raise ToolError("'query' is required.")

        if scope == "indexed_folder":
            folder = str(args.get("folder") or "").strip()
            if not folder:
                raise ToolError("'folder' is required when scope='indexed_folder'.")
            results = self._search_folder(query=query, folder=folder, limit=limit)
            return {"scope": scope, "count": len(results), "results": _strip_thumbnails(results)}

        # scope == "detections"
        since_hours = float(args.get("since_hours") or 24)
        since_ms    = int(time.time() * 1000 - since_hours * 3_600_000)
        probe_id    = str(args.get("probe_id") or "").strip() or None
        channel_id  = _opt_int(args.get("channel_id"))

        results = self._search_det(
            query=query,
            probe_id=probe_id,
            channel_id=channel_id,
            since_ms=since_ms,
            limit=limit,
        )
        return {"scope": scope, "count": len(results), "results": _strip_thumbnails(results)}

    # ── get_detections ──────────────────────────────────────────────────────

    def _get_detections(self, args: Dict[str, Any]) -> Dict[str, Any]:
        probe_id = str(args.get("probe_id") or "").strip() or None

        # Resolve probe_name → probe_id if needed
        probe_name_raw = str(args.get("probe_name") or "").strip()
        if not probe_id and probe_name_raw:
            probe_id = self._resolve_probe_id_by_name(probe_name_raw)

        channel_id  = _opt_int(args.get("channel_id"))
        since_hours = float(args.get("since_hours") or 24)
        until_hours = _opt_float(args.get("until_hours"))
        limit       = max(1, min(100, int(args.get("limit") or 20)))

        now_ms   = int(time.time() * 1000)
        since_ms = int(now_ms - since_hours * 3_600_000)
        until_ms = int(now_ms - until_hours * 3_600_000) if until_hours is not None else None

        rows, total = self._ds.list_detections(
            probe_id=probe_id,
            channel_id=channel_id,
            since_ms=since_ms,
            until_ms=until_ms,
            limit=limit,
            offset=0,
        )

        return {
            "total_in_window": total,
            "returned": len(rows),
            "detections": [_safe_detection(r) for r in rows],
        }

    # ── get_detection_summary ───────────────────────────────────────────────

    def _get_detection_summary(self, args: Dict[str, Any]) -> Dict[str, Any]:
        since_hours = float(args.get("since_hours") or 24)
        channel_id  = _opt_int(args.get("channel_id"))
        since_ms    = int(time.time() * 1000 - since_hours * 3_600_000)

        rows = self._ds.summarize_by_probe(since_ms=since_ms, channel_id=channel_id)
        return {
            "since_hours": since_hours,
            "probe_count": len(rows),
            "total_detections": sum(r.get("hit_count", 0) for r in rows),
            "by_probe": rows,
        }

    # ── update_probe ────────────────────────────────────────────────────────

    def _update_probe(self, args: Dict[str, Any]) -> Dict[str, Any]:
        changes  = args.get("changes") or {}
        preview  = bool(args.get("preview", True))
        probe_id = str(args.get("probe_id") or "").strip() or None
        probe_name_raw = str(args.get("probe_name") or "").strip()

        if not changes:
            raise ToolError("'changes' must contain at least one field to modify.")

        # Resolve probe
        if not probe_id and probe_name_raw:
            probe_id = self._resolve_probe_id_by_name(probe_name_raw)
        if not probe_id:
            raise ToolError("Provide 'probe_id' or 'probe_name'.")

        current = self._find_probe(probe_id)

        # Deep merge — only touch what changes specifies
        merged = _merge_probe(current, changes)

        # Validate merged object
        errors = _validate_probe(merged)
        if errors:
            raise ToolError("Validation failed: " + "; ".join(errors))

        # Build human-readable diff
        diff = _probe_diff(current, merged)

        if preview:
            return {
                "status": "preview",
                "probe_id": probe_id,
                "probe_name": current.get("name"),
                "diff": diff,
                "current": _probe_summary(current),
                "proposed": _probe_summary(merged),
            }

        # Apply
        self._ps.upsert_probe(merged)
        return {
            "status": "applied",
            "probe_id": probe_id,
            "probe_name": merged.get("name"),
            "diff": diff,
        }

    # ── describe_frame ──────────────────────────────────────────────────────

    def _describe_frame(self, args: Dict[str, Any]) -> Dict[str, Any]:
        image_path   = str(args.get("image_path") or "").strip() or None
        detection_id = _opt_int(args.get("detection_id"))
        prompt = str(args.get("prompt") or "").strip() or (
            "Describe what is happening in this image in detail. "
            "Note any people, vehicles, objects, or unusual activity."
        )

        if image_path is None and detection_id is None:
            raise ToolError("Provide either 'image_path' or 'detection_id'.")

        # Resolve image from detection record if needed
        resolved_path: Optional[str] = image_path
        if detection_id is not None and image_path is None:
            records = self._ds.fetch_detections_by_ids([detection_id], include_vectors=False)
            if not records:
                raise ToolError(f"Detection ID {detection_id} not found.")
            rec = records[0]
            resolved_path = rec.get("image_path")
            if not resolved_path:
                # Fall back to thumbnail
                thumb = rec.get("thumbnail")
                if not thumb:
                    raise ToolError(f"Detection {detection_id} has no image_path or thumbnail.")
                return self._describe_from_thumb_b64(thumb, prompt)

        if not resolved_path or not Path(resolved_path).exists():
            raise ToolError(f"Image file not found: {resolved_path!r}")

        # Build messages inline (mirrors _build_image_messages in oldapp.py)
        try:
            encoded = self._jpeg(Image.open(resolved_path), max_edge=960, quality=88)
        except Exception as exc:
            raise ToolError(f"Could not open image: {exc}") from exc

        messages = [
            {"role": "system", "content": "You are an expert visual analyst. Be concise and factual."},
            {"role": "user", "content": [
                {"type": "text", "text": f"Image: {Path(resolved_path).name}\n\nTask: {prompt}"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded}",
                    "detail": "high",
                }},
            ]},
        ]
        description = self._lm(messages)
        return {"description": description, "image_path": resolved_path}

    def _describe_from_thumb_b64(self, thumb_b64: str, prompt: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": "You are an expert visual analyst. Be concise and factual."},
            {"role": "user", "content": [
                {"type": "text", "text": f"Task: {prompt}"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{thumb_b64}",
                    "detail": "high",
                }},
            ]},
        ]
        description = self._lm(messages)
        return {"description": description, "image_path": None, "note": "low-res thumbnail used"}

    # ── create_bookmark ─────────────────────────────────────────────────────

    def _create_bookmark(self, args: Dict[str, Any]) -> Dict[str, Any]:
        channel_id = _opt_int(args.get("channel_id"))
        if channel_id is None:
            raise ToolError("'channel_id' is required.")
        title = str(args.get("title") or "").strip()[:80]
        if not title:
            raise ToolError("'title' is required.")
        description = str(args.get("description") or "").strip()[:240]
        severity    = str(args.get("severity") or "normal").lower()
        timestamp_ms = _opt_int(args.get("timestamp_ms")) or int(time.time() * 1000)

        if not hasattr(self._lxm, "send_bookmark_event"):
            raise ToolError("Luxriot manager is not available or not configured.")

        try:
            self._lxm.send_bookmark_event(
                channel_id=channel_id,
                title=title,
                description=description,
                severity=severity,
                state="new",
                timestamp_ms=timestamp_ms,
            )
        except Exception as exc:
            raise ToolError(f"Bookmark creation failed: {exc}") from exc

        return {
            "status": "created",
            "channel_id": channel_id,
            "title": title,
            "severity": severity,
        }

    # ── generate_report ─────────────────────────────────────────────────────

    def _generate_report(self, args: Dict[str, Any]) -> Dict[str, Any]:
        since_hours = float(args.get("since_hours") or 24)
        until_hours = _opt_float(args.get("until_hours"))
        channel_id  = _opt_int(args.get("channel_id"))
        include_probes: Optional[List[str]] = args.get("include_probes") or None
        top_events  = max(1, min(20, int(args.get("top_events") or 5)))

        now_ms   = int(time.time() * 1000)
        since_ms = int(now_ms - since_hours * 3_600_000)
        until_ms = int(now_ms - until_hours * 3_600_000) if until_hours is not None else None

        summary_rows = self._ds.summarize_by_probe(since_ms=since_ms, channel_id=channel_id)

        # Filter probes if requested
        if include_probes:
            names_lower = {n.lower() for n in include_probes}
            summary_rows = [
                r for r in summary_rows
                if str(r.get("probe_name") or r.get("probe_id") or "").lower() in names_lower
            ]

        probes_data = []
        activity_by_hour: Dict[str, int] = collections.defaultdict(int)
        total_detections = 0

        for row in summary_rows:
            pid = row["probe_id"]
            total_detections += row.get("hit_count", 0)

            # Fetch top events for this probe
            top_rows, _ = self._ds.list_detections(
                probe_id=pid,
                channel_id=channel_id,
                since_ms=since_ms,
                until_ms=until_ms,
                limit=top_events,
                offset=0,
            )

            # Accumulate hourly buckets
            all_rows, _ = self._ds.list_detections(
                probe_id=pid,
                channel_id=channel_id,
                since_ms=since_ms,
                until_ms=until_ms,
                limit=500,
                offset=0,
            )
            for det in all_rows:
                ts = det.get("timestamp_ms") or 0
                hour_key = time.strftime("%Y-%m-%d %H:00", time.localtime(ts / 1000))
                activity_by_hour[hour_key] += 1

            probes_data.append({
                "probe_id": pid,
                "probe_name": row.get("probe_name"),
                "channel_id": row.get("channel_id"),
                "hit_count": row.get("hit_count", 0),
                "latest_ts": row.get("latest_timestamp_ms"),
                "top_events": [_safe_detection(r) for r in top_rows],
            })

        return {
            "period": {
                "since_ms": since_ms,
                "until_ms": until_ms,
                "since_hours": since_hours,
            },
            "total_detections": total_detections,
            "probe_count": len(probes_data),
            "probes": probes_data,
            "activity_by_hour": dict(sorted(activity_by_hour.items())),
        }

    # ── helpers ─────────────────────────────────────────────────────────────

    def _resolve_probe_id_by_name(self, name: str) -> str:
        name_lower = name.lower()
        matches = [
            p for p in self._ps.list_probes()
            if str(p.get("name") or "").lower() == name_lower
        ]
        if not matches:
            raise ToolError(f"No probe found with name {name!r}.")
        if len(matches) > 1:
            ids = [p.get("id") for p in matches]
            raise ToolError(
                f"Multiple probes named {name!r}: {ids}. Use probe_id to be specific."
            )
        return str(matches[0]["id"])

    def _find_probe(self, probe_id: str) -> Dict[str, Any]:
        for p in self._ps.list_probes():
            if str(p.get("id")) == probe_id:
                return copy.deepcopy(p)
        raise ToolError(f"Probe not found: {probe_id!r}")


# ---------------------------------------------------------------------------
# Probe merge / validate helpers
# ---------------------------------------------------------------------------

_SCALAR_PROBE_FIELDS = (
    "pos_floor", "margin", "top_k", "window_sec",
    "enabled", "severity", "bookmark",
    "name", "channel_id",
)

def _merge_probe(current: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-merge agent changes onto the full probe object.
    Only the fields present in 'changes' are updated.
    ROI and image_probe are never touched — they are preserved as-is.
    """
    merged = copy.deepcopy(current)

    # Map agent field name 'margin_thr' → stored field 'margin'
    if "margin_thr" in changes:
        changes = dict(changes)
        changes["margin"] = changes.pop("margin_thr")
    if "bookmark_enabled" in changes:
        changes = dict(changes)
        changes["bookmark"] = changes.pop("bookmark_enabled")

    # Scalar fields
    for f in _SCALAR_PROBE_FIELDS:
        if f in changes:
            merged[f] = changes[f]

    # List fields: full replacement, cleaned
    for f in ("positives", "negatives"):
        if f in changes:
            raw = changes[f]
            if not isinstance(raw, list):
                raise ToolError(f"'{f}' must be a list of strings.")
            merged[f] = [str(s).strip() for s in raw if str(s).strip()]

    # ROI and image_probe: intentionally not in the schema — preserved unchanged.

    return merged


def _validate_probe(probe: Dict[str, Any]) -> List[str]:
    """Return a list of validation error strings (empty = valid)."""
    errors: List[str] = []
    positives  = probe.get("positives") or []
    image_data = (probe.get("image_probe") or {}).get("data")
    image_en   = (probe.get("image_probe") or {}).get("enabled", True)

    if not positives and not (image_data and image_en is not False):
        errors.append("Probe must have at least one positive text or an enabled image probe.")

    pos_floor = probe.get("pos_floor")
    if pos_floor is not None:
        try:
            v = float(pos_floor)
            if not (0.0 <= v <= 1.0):
                errors.append(f"pos_floor must be between 0.0 and 1.0, got {v}.")
        except (TypeError, ValueError):
            errors.append(f"pos_floor must be a number, got {pos_floor!r}.")

    margin = probe.get("margin")
    if margin is not None:
        try:
            v = float(margin)
            if v < 0.0:
                errors.append(f"margin must be >= 0.0, got {v}.")
        except (TypeError, ValueError):
            errors.append(f"margin must be a number, got {margin!r}.")

    return errors


def _probe_diff(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """Return {field: {before, after}} for changed fields."""
    diff: Dict[str, Any] = {}
    all_keys = set(before) | set(after)
    skip = {"bookmark_gate", "bookmark_gate_updated_at_ms", "last_hit", "recent_hits", "image_probe", "roi_norm"}
    for k in sorted(all_keys - skip):
        if before.get(k) != after.get(k):
            diff[k] = {"before": before.get(k), "after": after.get(k)}
    return diff


def _probe_summary(probe: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name":        probe.get("name"),
        "enabled":     probe.get("enabled"),
        "positives":   probe.get("positives"),
        "negatives":   probe.get("negatives"),
        "pos_floor":   probe.get("pos_floor"),
        "margin":      probe.get("margin"),
        "top_k":       probe.get("top_k"),
        "window_sec":  probe.get("window_sec"),
        "severity":    probe.get("severity"),
        "bookmark":    probe.get("bookmark"),
    }


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def build_system_prompt(
    probes_store: Any,
    detections_store: Any,
    luxriot_manager: Any,
) -> str:
    now_str = time.strftime("%Y-%m-%d %H:%M")

    # Active probes
    try:
        probes = probes_store.list_probes()
    except Exception:
        probes = []

    # Recent detection counts (last 24h)
    since_ms = int(time.time() * 1000 - 86_400_000)
    try:
        probe_summary = detections_store.summarize_by_probe(since_ms=since_ms)
        hit_by_probe  = {r["probe_id"]: r["hit_count"] for r in probe_summary}
    except Exception:
        hit_by_probe  = {}

    # Available channels
    try:
        raw_channels = getattr(luxriot_manager, "channels", None) or []
        channels_str = ", ".join(
            f"{c.get('id')} ({c.get('title', 'unknown')})"
            for c in (raw_channels if isinstance(raw_channels, list) else [])
        ) or "unknown (Luxriot not connected)"
    except Exception:
        channels_str = "unknown"

    probe_lines = []
    for p in probes:
        pid    = p.get("id", "?")
        pname  = p.get("name", pid)
        ch     = p.get("channel_id", "?")
        floor  = p.get("pos_floor", "?")
        margin = p.get("margin", "?")
        en     = "enabled" if p.get("enabled", True) else "disabled"
        hits   = hit_by_probe.get(pid, 0)
        probe_lines.append(
            f"  - \"{pname}\" [id={pid}, ch={ch}]  "
            f"pos_floor={floor}, margin={margin}, {en}, {hits} hits/24h"
        )

    probe_block = "\n".join(probe_lines) if probe_lines else "  (no probes configured)"

    return (
        f"You are the AI operations assistant for Luxriot EVA AI — a CCTV intelligent "
        f"monitoring platform. You have tools to search the archive, inspect detections, "
        f"tune probes, describe frames, create bookmarks, and compile reports.\n"
        f"Be concise and operator-focused. Never fabricate detection data.\n\n"
        f"Current time: {now_str}\n\n"
        f"Active probes ({len(probes)} total):\n{probe_block}\n\n"
        f"Available channels: {channels_str}\n\n"
        f"Rules:\n"
        f"- For probe modifications: always call update_probe with preview=true first, "
        f"show the user the diff, and only apply after explicit confirmation.\n"
        f"- When returning search results or detections, summarize; don't dump raw lists.\n"
        f"- Use markdown for structure."
    )


# ---------------------------------------------------------------------------
# Agent runner (tool loop + SSE)
# ---------------------------------------------------------------------------

class AgentRunner:
    """
    Drives the tool-calling loop and yields SSE events as strings.
    Instantiated once in oldapp.py.
    """

    def __init__(
        self,
        *,
        embed_text_fn: Callable[[str], np.ndarray],
        embed_image_fn: Callable[[Image.Image], np.ndarray],
        call_lm_fn: Callable[..., str],
        encode_jpeg_fn: Callable[..., str],
        probes_store: Any,
        detections_store: Any,
        luxriot_manager: Any,
        search_indexed_folder_fn: Callable[..., List[Dict[str, Any]]],
        search_detections_fn: Callable[..., List[Dict[str, Any]]],
        lm_base_url: str,
        lm_model: str,
        lm_api_key: str,
        lm_timeout: int,
        store_path: str = "agent_sessions.sqlite3",
    ) -> None:
        self._ps  = probes_store
        self._ds  = detections_store
        self._lxm = luxriot_manager
        self.store = AgentStore(path=store_path)
        self._lm_client = _AgentLMClient(
            base_url=lm_base_url,
            model=lm_model,
            api_key=lm_api_key,
            timeout=lm_timeout,
        )
        self._tools = AgentTools(
            detections_store=detections_store,
            probes_store=probes_store,
            luxriot_manager=luxriot_manager,
            embed_text_fn=embed_text_fn,
            embed_image_fn=embed_image_fn,
            call_lm_fn=call_lm_fn,
            encode_jpeg_fn=encode_jpeg_fn,
            search_indexed_folder_fn=search_indexed_folder_fn,
            search_detections_fn=search_detections_fn,
        )

    def stream_chat(
        self,
        session_id: Optional[str],
        message: str,
        image_b64: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Main entry point. Yields SSE-formatted strings.
        Each event is:  'data: <json>\\n\\n'
        """
        # ── session setup ──────────────────────────────────────────────────
        if not session_id or not self.store.session_exists(session_id):
            session_id = self.store.create_session()

        title = message[:60].strip() or "Chat"
        self.store.touch_session(session_id, title=title)

        yield _sse({"type": "session", "session_id": session_id})

        # ── persist user message ───────────────────────────────────────────
        user_content: Any
        if image_b64:
            user_content = [
                {"type": "text", "text": message},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}",
                    "detail": "high",
                }},
            ]
        else:
            user_content = message

        self.store.add_message(session_id, role="user", content=message)

        # ── build messages for LM ──────────────────────────────────────────
        system_prompt = build_system_prompt(self._ps, self._ds, self._lxm)
        history = self.store.load_history(session_id)

        # Replace the stored user content with the full (possibly image-bearing) one
        in_flight: List[Dict[str, Any]] = (
            [{"role": "system", "content": system_prompt}]
            + history[:-1]                          # all but the just-added user msg
            + [{"role": "user", "content": user_content}]
        )

        # Accumulated messages from this turn (to persist after streaming)
        new_assistant_messages: List[Dict[str, Any]] = []

        # ── tool loop ──────────────────────────────────────────────────────
        for iteration in range(AGENT_MAX_TOOL_ITERATIONS):
            # Run the blocking LM call in a thread so we can emit heartbeats
            lm_response: _LMResponse
            try:
                lm_response = yield from _run_with_heartbeats(
                    fn=lambda: self._lm_client.call_with_tools(in_flight),
                    heartbeat_interval=AGENT_HEARTBEAT_INTERVAL,
                )
            except Exception as exc:
                yield _sse({"type": "error", "message": f"LM error: {exc}"})
                yield _sse({"type": "done", "session_id": session_id})
                return

            if lm_response.finish_reason != "tool_calls" or not lm_response.tool_calls:
                # Model wants to respond with text — break out to streaming phase
                break

            # Append assistant turn with tool_calls to in-flight history
            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": lm_response.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.args)},
                    }
                    for tc in lm_response.tool_calls
                ],
            }
            in_flight.append(assistant_msg)
            new_assistant_messages.append(assistant_msg)

            # Execute each tool call
            for tc in lm_response.tool_calls:
                yield _sse({"type": "tool_call", "name": tc.name, "args": tc.args})

                try:
                    result = yield from _run_with_heartbeats(
                        fn=lambda tc=tc: self._tools.execute(tc.name, tc.args),
                        heartbeat_interval=AGENT_HEARTBEAT_INTERVAL,
                    )
                    result_msg = {"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                  "content": json.dumps(result, default=str)}
                    yield _sse({"type": "tool_result", "name": tc.name, "result": result})
                except ToolError as exc:
                    error_payload = {"error": str(exc)}
                    result_msg = {"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                  "content": json.dumps(error_payload)}
                    yield _sse({"type": "tool_result", "name": tc.name,
                                "result": error_payload, "error": str(exc)})
                except Exception as exc:
                    error_payload = {"error": f"Internal tool error: {exc}"}
                    result_msg = {"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                  "content": json.dumps(error_payload)}
                    yield _sse({"type": "tool_result", "name": tc.name,
                                "result": error_payload, "error": str(exc)})

                in_flight.append(result_msg)
                new_assistant_messages.append(result_msg)

        else:
            # Hit iteration limit
            yield _sse({"type": "error",
                        "message": f"Tool loop exceeded {AGENT_MAX_TOOL_ITERATIONS} iterations."})
            yield _sse({"type": "done", "session_id": session_id})
            return

        # ── final streaming text response ──────────────────────────────────
        full_text_parts: List[str] = []
        try:
            for chunk in self._lm_client.stream_text(in_flight):
                full_text_parts.append(chunk)
                yield _sse({"type": "text", "content": chunk})
        except Exception as exc:
            yield _sse({"type": "error", "message": f"Streaming error: {exc}"})
            yield _sse({"type": "done", "session_id": session_id})
            return

        final_text = "".join(full_text_parts)

        # ── persist assistant turn ─────────────────────────────────────────
        # Persist intermediate tool-call/result pairs
        for msg in new_assistant_messages:
            role = msg.get("role", "assistant")
            if role == "assistant":
                tcs = msg.get("tool_calls")
                self.store.add_message(
                    session_id,
                    role="assistant",
                    content=msg.get("content"),
                    tool_calls=tcs,
                )
            elif role == "tool":
                # Strip large binary content before persisting
                raw_content = msg.get("content", "{}")
                try:
                    parsed = json.loads(raw_content)
                    parsed = _strip_thumbnails_deep(parsed)
                    content_to_store = json.dumps(parsed, default=str)
                except Exception:
                    content_to_store = raw_content
                self.store.add_message(
                    session_id,
                    role="tool",
                    tool_call_id=msg.get("tool_call_id"),
                    tool_name=msg.get("name"),
                    tool_result=content_to_store,
                )

        # Persist final assistant text
        if final_text:
            self.store.add_message(session_id, role="assistant", content=final_text)

        yield _sse({"type": "done", "session_id": session_id})


# ---------------------------------------------------------------------------
# Heartbeat helper
# ---------------------------------------------------------------------------

def _run_with_heartbeats(
    fn: Callable[[], Any],
    heartbeat_interval: float = AGENT_HEARTBEAT_INTERVAL,
) -> Generator[str, None, Any]:
    """
    Run fn() in a thread. Yield SSE heartbeat events every heartbeat_interval
    seconds while waiting. Return the result when done.
    Usage:  result = yield from _run_with_heartbeats(fn)
    """
    result_holder: Dict[str, Any] = {}
    exc_holder: Dict[str, Any] = {}

    def _run() -> None:
        try:
            result_holder["v"] = fn()
        except Exception as exc:
            exc_holder["v"] = exc

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    while t.is_alive():
        t.join(timeout=heartbeat_interval)
        if t.is_alive():
            yield _sse({"type": "heartbeat"})

    if "v" in exc_holder:
        raise exc_holder["v"]
    return result_holder.get("v")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _now_ms() -> int:
    return int(time.time() * 1000)


def _sse(obj: Dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, default=str)}\n\n"


def _opt_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _opt_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_detection(r: Dict[str, Any]) -> Dict[str, Any]:
    """Return detection dict with thumbnail stripped (use image_path instead)."""
    out = {k: v for k, v in r.items() if k not in ("thumbnail", "clip_vec", "dino_vec")}
    out["has_thumbnail"] = bool(r.get("thumbnail"))
    return out


def _strip_thumbnails(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {k: v for k, v in r.items()
         if k not in ("thumbnail", "thumbnail_b64", "clip_vec", "dino_vec")}
        for r in results
    ]


def _strip_thumbnails_deep(obj: Any) -> Any:
    """Recursively remove thumbnail/vector fields from nested dicts/lists."""
    if isinstance(obj, dict):
        return {
            k: _strip_thumbnails_deep(v)
            for k, v in obj.items()
            if k not in ("thumbnail", "thumbnail_b64", "clip_vec", "dino_vec")
        }
    if isinstance(obj, list):
        return [_strip_thumbnails_deep(item) for item in obj]
    return obj


def _msg_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    for field_name in ("tool_calls", "tool_result"):
        raw = d.get(field_name)
        if raw:
            try:
                d[field_name] = json.loads(raw)
            except Exception:
                pass
    return d


def _msg_row_to_openai(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a DB row to an OpenAI-compatible message dict."""
    d = dict(row)
    role = d.get("role", "user")

    if role == "assistant":
        msg: Dict[str, Any] = {"role": "assistant"}
        if d.get("content"):
            msg["content"] = d["content"]
        tcs_raw = d.get("tool_calls")
        if tcs_raw:
            try:
                msg["tool_calls"] = json.loads(tcs_raw)
            except Exception:
                pass
        return msg

    if role == "tool":
        raw_result = d.get("tool_result") or "{}"
        return {
            "role": "tool",
            "tool_call_id": d.get("tool_call_id", ""),
            "name": d.get("tool_name", ""),
            "content": raw_result if isinstance(raw_result, str) else json.dumps(raw_result),
        }

    # user
    return {"role": "user", "content": d.get("content") or ""}
