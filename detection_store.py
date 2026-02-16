import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


class DetectionsStore:
    """SQLite-backed event store for probe detections."""

    def __init__(self, path: str = "detections_store.sqlite3", max_records: int = 20000) -> None:
        self.path = Path(path)
        self.max_records = max(1000, int(max_records or 20000))
        self.lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self.lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS detections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        dedupe_key TEXT UNIQUE,
                        event_timestamp_ms INTEGER NOT NULL,
                        recorded_at_ms INTEGER NOT NULL,
                        probe_id TEXT NOT NULL,
                        probe_name TEXT,
                        channel_id INTEGER NOT NULL,
                        severity TEXT,
                        bookmark_enabled INTEGER NOT NULL DEFAULT 0,
                        bookmark_sent INTEGER NOT NULL DEFAULT 0,
                        pos_score REAL,
                        neg_score REAL,
                        margin REAL,
                        thumbnail_b64 TEXT,
                        source TEXT,
                        payload_json TEXT
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_detections_probe_ts ON detections (probe_id, event_timestamp_ms DESC)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_detections_channel_ts ON detections (channel_id, event_timestamp_ms DESC)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_detections_recorded ON detections (recorded_at_ms DESC)"
                )
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _normalize_detection(record: Dict[str, Any]) -> Dict[str, Any]:
        now_ms = int(time.time() * 1000)
        event_ts = int(record.get("timestamp_ms") or now_ms)
        probe_id = str(record.get("probe_id") or "").strip()
        if not probe_id:
            raise ValueError("probe_id is required")
        try:
            channel_id = int(record.get("channel_id"))
        except Exception as exc:
            raise ValueError("channel_id is required") from exc
        pos_score = float(record.get("pos_score", 0.0))
        neg_score = float(record.get("neg_score", 0.0))
        margin = float(record.get("margin", 0.0))
        source = str(record.get("source") or "probe").strip().lower()
        dedupe_key = str(
            record.get("dedupe_key")
            or f"{probe_id}:{event_ts}:{pos_score:.3f}:{neg_score:.3f}:{margin:.3f}:{source}"
        )
        payload = record.get("payload")
        payload_json = json.dumps(payload, ensure_ascii=True) if isinstance(payload, (dict, list)) else None
        return {
            "dedupe_key": dedupe_key,
            "event_timestamp_ms": event_ts,
            "recorded_at_ms": int(record.get("recorded_at_ms") or now_ms),
            "probe_id": probe_id,
            "probe_name": str(record.get("probe_name") or "").strip() or None,
            "channel_id": channel_id,
            "severity": str(record.get("severity") or "").strip().lower() or None,
            "bookmark_enabled": 1 if bool(record.get("bookmark_enabled", False)) else 0,
            "bookmark_sent": 1 if bool(record.get("bookmark_sent", False)) else 0,
            "pos_score": pos_score,
            "neg_score": neg_score,
            "margin": margin,
            "thumbnail_b64": str(record.get("thumbnail_b64") or "").strip() or None,
            "source": source,
            "payload_json": payload_json,
        }

    def _trim_to_cap(self, conn: sqlite3.Connection) -> None:
        row = conn.execute("SELECT COUNT(*) AS c FROM detections").fetchone()
        total = int(row["c"]) if row else 0
        if total <= self.max_records:
            return
        excess = total - self.max_records
        conn.execute(
            "DELETE FROM detections WHERE id IN (SELECT id FROM detections ORDER BY id ASC LIMIT ?)",
            (excess,),
        )

    def add_detection(self, record: Dict[str, Any]) -> bool:
        normalized = self._normalize_detection(record)
        with self.lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO detections (
                        dedupe_key,
                        event_timestamp_ms,
                        recorded_at_ms,
                        probe_id,
                        probe_name,
                        channel_id,
                        severity,
                        bookmark_enabled,
                        bookmark_sent,
                        pos_score,
                        neg_score,
                        margin,
                        thumbnail_b64,
                        source,
                        payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        normalized["dedupe_key"],
                        normalized["event_timestamp_ms"],
                        normalized["recorded_at_ms"],
                        normalized["probe_id"],
                        normalized["probe_name"],
                        normalized["channel_id"],
                        normalized["severity"],
                        normalized["bookmark_enabled"],
                        normalized["bookmark_sent"],
                        normalized["pos_score"],
                        normalized["neg_score"],
                        normalized["margin"],
                        normalized["thumbnail_b64"],
                        normalized["source"],
                        normalized["payload_json"],
                    ),
                )
                self._trim_to_cap(conn)
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def add_detections(self, records: Sequence[Dict[str, Any]]) -> int:
        inserted = 0
        for record in records:
            try:
                if self.add_detection(record):
                    inserted += 1
            except Exception:
                continue
        return inserted

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        payload_json = row["payload_json"]
        payload: Optional[Any] = None
        if payload_json:
            try:
                payload = json.loads(payload_json)
            except Exception:
                payload = None
        return {
            "id": row["id"],
            "timestamp_ms": row["event_timestamp_ms"],
            "recorded_at_ms": row["recorded_at_ms"],
            "probe_id": row["probe_id"],
            "probe_name": row["probe_name"],
            "channel_id": row["channel_id"],
            "severity": row["severity"],
            "bookmark_enabled": bool(row["bookmark_enabled"]),
            "bookmark_sent": bool(row["bookmark_sent"]),
            "pos_score": float(row["pos_score"] or 0.0),
            "neg_score": float(row["neg_score"] or 0.0),
            "margin": float(row["margin"] or 0.0),
            "thumbnail": row["thumbnail_b64"],
            "source": row["source"],
            "payload": payload,
        }

    def list_detections(
        self,
        probe_id: Optional[str] = None,
        channel_id: Optional[int] = None,
        since_ms: Optional[int] = None,
        until_ms: Optional[int] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        limit = max(1, min(500, int(limit or 50)))
        offset = max(0, int(offset or 0))
        where: List[str] = []
        params: List[Any] = []
        if probe_id:
            where.append("probe_id = ?")
            params.append(str(probe_id))
        if channel_id is not None:
            where.append("channel_id = ?")
            params.append(int(channel_id))
        if since_ms is not None:
            where.append("event_timestamp_ms >= ?")
            params.append(int(since_ms))
        if until_ms is not None:
            where.append("event_timestamp_ms <= ?")
            params.append(int(until_ms))
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""

        with self.lock:
            conn = self._connect()
            try:
                count_row = conn.execute(
                    f"SELECT COUNT(*) AS c FROM detections {where_sql}",
                    tuple(params),
                ).fetchone()
                total = int(count_row["c"]) if count_row else 0
                rows = conn.execute(
                    f"""
                    SELECT
                        id,
                        event_timestamp_ms,
                        recorded_at_ms,
                        probe_id,
                        probe_name,
                        channel_id,
                        severity,
                        bookmark_enabled,
                        bookmark_sent,
                        pos_score,
                        neg_score,
                        margin,
                        thumbnail_b64,
                        source,
                        payload_json
                    FROM detections
                    {where_sql}
                    ORDER BY event_timestamp_ms DESC, id DESC
                    LIMIT ? OFFSET ?
                    """,
                    tuple(params + [limit, offset]),
                ).fetchall()
                return [self._row_to_dict(row) for row in rows], total
            finally:
                conn.close()

    def summarize_by_probe(
        self,
        since_ms: Optional[int] = None,
        channel_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        limit = max(1, min(500, int(limit or 100)))
        where: List[str] = []
        params: List[Any] = []
        if since_ms is not None:
            where.append("event_timestamp_ms >= ?")
            params.append(int(since_ms))
        if channel_id is not None:
            where.append("channel_id = ?")
            params.append(int(channel_id))
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""

        with self.lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"""
                    SELECT
                        probe_id,
                        MAX(probe_name) AS probe_name,
                        channel_id,
                        COUNT(*) AS hit_count,
                        MAX(event_timestamp_ms) AS latest_timestamp_ms
                    FROM detections
                    {where_sql}
                    GROUP BY probe_id, channel_id
                    ORDER BY latest_timestamp_ms DESC
                    LIMIT ?
                    """,
                    tuple(params + [limit]),
                ).fetchall()
                return [
                    {
                        "probe_id": row["probe_id"],
                        "probe_name": row["probe_name"],
                        "channel_id": row["channel_id"],
                        "hit_count": int(row["hit_count"] or 0),
                        "latest_timestamp_ms": int(row["latest_timestamp_ms"] or 0),
                    }
                    for row in rows
                ]
            finally:
                conn.close()
