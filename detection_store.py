import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np


PROBE_SOURCE_ALIASES = frozenset({"probe", "probes_run", "probes_query", "probe_daemon"})
ARCHIVE_SOURCE_ALIASES = {
    "probe": "probe",
    "probes_run": "probe",
    "probes_query": "probe",
    "probe_daemon": "probe",
    "vlm_summary": "vlm_summary",
    "vlm_alert": "vlm_alert",
}
SOURCE_GROUP_SQL = (
    "CASE WHEN source IN ('probe', 'probes_run', 'probes_query', 'probe_daemon') "
    "THEN 'probe' ELSE COALESCE(NULLIF(source, ''), 'unknown') END"
)


class DetectionsStore:
    """SQLite-backed event store for probe detections plus optional CLIP/DINO vectors."""

    backend = "sqlite"

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
                        payload_json TEXT,
                        shard_key TEXT,
                        image_path TEXT,
                        clip_vec BLOB,
                        dino_vec BLOB,
                        dino_ready INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
                self._migrate_columns(conn)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_detections_probe_ts ON detections (probe_id, event_timestamp_ms DESC)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_detections_channel_ts ON detections (channel_id, event_timestamp_ms DESC)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_detections_recorded ON detections (recorded_at_ms DESC)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_detections_shard_ts ON detections (shard_key, event_timestamp_ms DESC)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_detections_shard_id ON detections (shard_key, id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_detections_dino_ready ON detections (dino_ready, id DESC)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_detections_image_path ON detections (image_path)"
                )
                conn.commit()
            finally:
                conn.close()

    def health(self) -> Dict[str, Any]:
        try:
            with self.lock:
                conn = self._connect()
                try:
                    conn.execute("SELECT 1").fetchone()
                finally:
                    conn.close()
            return {
                "ok": True,
                "status": "reachable",
                "backend": self.backend,
                "path": str(self.path),
            }
        except Exception as exc:
            return {
                "ok": False,
                "status": "error",
                "backend": self.backend,
                "path": str(self.path),
                "error": str(exc),
            }

    @staticmethod
    def _vec_to_blob(vec: Any) -> Optional[bytes]:
        if vec is None:
            return None
        try:
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if arr.size == 0:
            return None
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        return arr.astype(np.float32, copy=False).tobytes()

    @staticmethod
    def _blob_to_vec(blob: Optional[bytes]) -> Optional[np.ndarray]:
        if not blob:
            return None
        arr = np.frombuffer(blob, dtype=np.float32)
        if arr.size == 0:
            return None
        # copy detaches from sqlite-managed memory
        return arr.astype(np.float32, copy=True)

    @staticmethod
    def _shard_key(channel_id: int, event_ts_ms: int) -> str:
        ts_sec = max(0.0, float(event_ts_ms) / 1000.0)
        date_key = time.strftime("%Y%m%d", time.localtime(ts_sec))
        return f"ch{channel_id}:{date_key}"

    def _migrate_columns(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute("PRAGMA table_info(detections)").fetchall()
        existing = {str(row["name"]) for row in rows}
        migrations: List[Tuple[str, str]] = [
            ("shard_key", "TEXT"),
            ("image_path", "TEXT"),
            ("clip_vec", "BLOB"),
            ("dino_vec", "BLOB"),
            ("dino_ready", "INTEGER NOT NULL DEFAULT 0"),
        ]
        for column, ddl in migrations:
            if column in existing:
                continue
            conn.execute(f"ALTER TABLE detections ADD COLUMN {column} {ddl}")

    @staticmethod
    def _normalize_detection(record: Dict[str, Any]) -> Dict[str, Any]:
        now_ms = int(time.time() * 1000)
        event_ts = int(record.get("timestamp_ms") or now_ms)
        probe_id = str(record.get("probe_id") or "").strip()
        if not probe_id:
            raise ValueError("probe_id is required")
        channel_raw = record.get("channel_id")
        if channel_raw is None:
            raise ValueError("channel_id is required")
        try:
            channel_id = int(cast(Any, channel_raw))
        except Exception as exc:
            raise ValueError("channel_id is required") from exc

        pos_score = float(record.get("pos_score", 0.0))
        neg_score = float(record.get("neg_score", 0.0))
        margin = float(record.get("margin", 0.0))
        source = str(record.get("source") or "probe").strip().lower()
        source = ARCHIVE_SOURCE_ALIASES.get(source, source)
        dedupe_key = str(
            record.get("dedupe_key")
            or f"{probe_id}:{event_ts}:{pos_score:.3f}:{neg_score:.3f}:{margin:.3f}:{source}"
        )

        payload_obj = record.get("payload")
        payload_json = json.dumps(payload_obj, ensure_ascii=True) if isinstance(payload_obj, (dict, list)) else None

        image_path = str(record.get("image_path") or "").strip() or None
        if image_path is None and isinstance(payload_obj, dict):
            image_path = str(payload_obj.get("image_path") or "").strip() or None

        clip_value = record.get("clip_vec")
        if clip_value is None:
            clip_value = record.get("clip_embedding")
        dino_value = record.get("dino_vec")
        if dino_value is None:
            dino_value = record.get("dino_embedding")
        clip_vec_blob = DetectionsStore._vec_to_blob(clip_value)
        dino_vec_blob = DetectionsStore._vec_to_blob(dino_value)

        shard_key = str(record.get("shard_key") or "").strip() or DetectionsStore._shard_key(channel_id, event_ts)

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
            "shard_key": shard_key,
            "image_path": image_path,
            "clip_vec": clip_vec_blob,
            "dino_vec": dino_vec_blob,
            "dino_ready": 1 if dino_vec_blob is not None else 0,
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
                        payload_json,
                        shard_key,
                        image_path,
                        clip_vec,
                        dino_vec,
                        dino_ready
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        normalized["shard_key"],
                        normalized["image_path"],
                        normalized["clip_vec"],
                        normalized["dino_vec"],
                        normalized["dino_ready"],
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
    def _decode_payload(payload_json: Optional[str]) -> Optional[Any]:
        if not payload_json:
            return None
        try:
            return json.loads(payload_json)
        except Exception:
            return None

    @classmethod
    def _row_to_dict(cls, row: sqlite3.Row, include_vectors: bool = False) -> Dict[str, Any]:
        payload = cls._decode_payload(row["payload_json"]) if "payload_json" in row.keys() else None
        data: Dict[str, Any] = {
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
            "shard_key": row["shard_key"] if "shard_key" in row.keys() else None,
            "image_path": row["image_path"] if "image_path" in row.keys() else None,
            "has_clip": bool(row["clip_vec"]) if "clip_vec" in row.keys() else False,
            "has_dino": bool(row["dino_vec"]) if "dino_vec" in row.keys() else False,
        }
        if include_vectors:
            data["clip_vec"] = cls._blob_to_vec(row["clip_vec"] if "clip_vec" in row.keys() else None)
            data["dino_vec"] = cls._blob_to_vec(row["dino_vec"] if "dino_vec" in row.keys() else None)
        return data

    @staticmethod
    def _build_where(
        probe_id: Optional[str] = None,
        channel_id: Optional[int] = None,
        source: Optional[str] = None,
        since_ms: Optional[int] = None,
        until_ms: Optional[int] = None,
        only_with_clip: bool = False,
        only_with_dino: bool = False,
        ids: Optional[Sequence[int]] = None,
    ) -> Tuple[str, List[Any]]:
        where: List[str] = []
        params: List[Any] = []
        if probe_id:
            where.append("probe_id = ?")
            params.append(str(probe_id))
        if channel_id is not None:
            where.append("channel_id = ?")
            params.append(int(channel_id))
        if source:
            normalized_source = ARCHIVE_SOURCE_ALIASES.get(str(source).strip().lower(), str(source).strip().lower())
            if normalized_source == "probe":
                placeholders = ",".join("?" for _ in PROBE_SOURCE_ALIASES)
                where.append(f"source IN ({placeholders})")
                params.extend(sorted(PROBE_SOURCE_ALIASES))
            else:
                where.append("source = ?")
                params.append(normalized_source)
        if since_ms is not None:
            where.append("event_timestamp_ms >= ?")
            params.append(int(since_ms))
        if until_ms is not None:
            where.append("event_timestamp_ms <= ?")
            params.append(int(until_ms))
        if only_with_clip:
            where.append("clip_vec IS NOT NULL")
        if only_with_dino:
            where.append("dino_vec IS NOT NULL")
        if ids is not None:
            ids_clean = [int(i) for i in ids]
            if not ids_clean:
                where.append("1 = 0")
            else:
                placeholders = ",".join("?" for _ in ids_clean)
                where.append(f"id IN ({placeholders})")
                params.extend(ids_clean)
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        return where_sql, params

    def list_detections(
        self,
        probe_id: Optional[str] = None,
        channel_id: Optional[int] = None,
        since_ms: Optional[int] = None,
        until_ms: Optional[int] = None,
        limit: int = 50,
        offset: int = 0,
        source: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        limit = max(1, min(500, int(limit or 50)))
        offset = max(0, int(offset or 0))
        where_sql, params = self._build_where(
            probe_id=probe_id,
            channel_id=channel_id,
            source=source,
            since_ms=since_ms,
            until_ms=until_ms,
        )

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
                        payload_json,
                        shard_key,
                        image_path,
                        clip_vec,
                        dino_vec
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

    def list_vector_candidates(
        self,
        probe_id: Optional[str] = None,
        channel_id: Optional[int] = None,
        since_ms: Optional[int] = None,
        until_ms: Optional[int] = None,
        limit: int = 20000,
        only_with_clip: bool = True,
        include_vectors: bool = False,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        limit = max(1, min(100000, int(limit or 20000)))
        where_sql, params = self._build_where(
            probe_id=probe_id,
            channel_id=channel_id,
            source=source,
            since_ms=since_ms,
            until_ms=until_ms,
            only_with_clip=only_with_clip,
        )
        with self.lock:
            conn = self._connect()
            try:
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
                        payload_json,
                        shard_key,
                        image_path,
                        clip_vec,
                        dino_vec
                    FROM detections
                    {where_sql}
                    ORDER BY event_timestamp_ms DESC, id DESC
                    LIMIT ?
                    """,
                    tuple(params + [limit]),
                ).fetchall()
                return [self._row_to_dict(row, include_vectors=include_vectors) for row in rows]
            finally:
                conn.close()

    def fetch_detections_by_ids(self, ids: Sequence[int], include_vectors: bool = True) -> List[Dict[str, Any]]:
        ids_clean = [int(item) for item in ids if item is not None]
        if not ids_clean:
            return []
        where_sql, params = self._build_where(ids=ids_clean)
        with self.lock:
            conn = self._connect()
            try:
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
                        payload_json,
                        shard_key,
                        image_path,
                        clip_vec,
                        dino_vec
                    FROM detections
                    {where_sql}
                    """,
                    tuple(params),
                ).fetchall()
                return [self._row_to_dict(row, include_vectors=include_vectors) for row in rows]
            finally:
                conn.close()

    def channel_ids_for_image_path(self, image_path: str) -> frozenset[int]:
        normalized = str(image_path or "").strip()
        if not normalized:
            return frozenset()
        with self.lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT DISTINCT channel_id
                    FROM detections
                    WHERE image_path = ?
                    """,
                    (normalized,),
                ).fetchall()
                return frozenset(
                    int(row["channel_id"])
                    for row in rows
                    if row["channel_id"] is not None
                )
            finally:
                conn.close()

    def update_dino_embeddings(self, rows: Sequence[Tuple[int, Sequence[float]]]) -> int:
        if not rows:
            return 0
        payload: List[Tuple[bytes, int]] = []
        for det_id, vec in rows:
            blob = self._vec_to_blob(vec)
            if blob is None:
                continue
            payload.append((blob, int(det_id)))
        if not payload:
            return 0
        with self.lock:
            conn = self._connect()
            try:
                conn.executemany(
                    "UPDATE detections SET dino_vec = ?, dino_ready = 1 WHERE id = ?",
                    payload,
                )
                conn.commit()
                return conn.total_changes
            finally:
                conn.close()

    def update_clip_embeddings(self, rows: Sequence[Tuple[int, Sequence[float]]]) -> int:
        if not rows:
            return 0
        payload: List[Tuple[bytes, int]] = []
        for det_id, vec in rows:
            blob = self._vec_to_blob(vec)
            if blob is None:
                continue
            payload.append((blob, int(det_id)))
        if not payload:
            return 0
        with self.lock:
            conn = self._connect()
            try:
                conn.executemany(
                    "UPDATE detections SET clip_vec = ? WHERE id = ?",
                    payload,
                )
                conn.commit()
                return conn.total_changes
            finally:
                conn.close()

    def shard_version(self, shard_key: str, embedder: str = "clip") -> Tuple[int, int]:
        column = "dino_vec" if embedder == "dino" else "clip_vec"
        with self.lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    f"SELECT COUNT(*) AS c, MAX(id) AS max_id FROM detections WHERE shard_key = ? AND {column} IS NOT NULL",
                    (str(shard_key),),
                ).fetchone()
                if not row:
                    return 0, 0
                return int(row["c"] or 0), int(row["max_id"] or 0)
            finally:
                conn.close()

    def load_shard_vectors(self, shard_key: str, embedder: str = "clip") -> Tuple[List[int], np.ndarray]:
        column = "dino_vec" if embedder == "dino" else "clip_vec"
        with self.lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"SELECT id, {column} AS vec_blob FROM detections WHERE shard_key = ? AND {column} IS NOT NULL ORDER BY id ASC",
                    (str(shard_key),),
                ).fetchall()
                ids: List[int] = []
                vectors: List[np.ndarray] = []
                target_dim: Optional[int] = None
                for row in rows:
                    vec = self._blob_to_vec(row["vec_blob"])
                    if vec is None:
                        continue
                    if target_dim is None:
                        target_dim = int(vec.shape[0]) if vec.ndim == 1 else None
                    if target_dim is None:
                        continue
                    if vec.ndim != 1 or int(vec.shape[0]) != target_dim:
                        continue
                    ids.append(int(row["id"]))
                    vectors.append(vec)
                if not vectors:
                    return [], np.zeros((0, 0), dtype=np.float32)
                mat = np.stack(vectors, axis=0).astype(np.float32, copy=False)
                return ids, mat
            finally:
                conn.close()

    def summarize_by_probe(
        self,
        since_ms: Optional[int] = None,
        channel_id: Optional[int] = None,
        limit: int = 100,
        source: Optional[str] = None,
        until_ms: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        limit = max(1, min(500, int(limit or 100)))
        where_sql, params = self._build_where(
            channel_id=channel_id,
            source=source,
            since_ms=since_ms,
            until_ms=until_ms,
        )

        with self.lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"""
                    SELECT
                        probe_id,
                        MAX(probe_name) AS probe_name,
                        channel_id,
                        {SOURCE_GROUP_SQL} AS source_key,
                        COUNT(*) AS hit_count,
                        MAX(event_timestamp_ms) AS latest_timestamp_ms
                    FROM detections
                    {where_sql}
                    GROUP BY probe_id, channel_id, {SOURCE_GROUP_SQL}
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
                        "source": row["source_key"],
                        "hit_count": int(row["hit_count"] or 0),
                        "latest_timestamp_ms": int(row["latest_timestamp_ms"] or 0),
                    }
                    for row in rows
                ]
            finally:
                conn.close()

    def summarize_shards(
        self,
        probe_id: Optional[str] = None,
        channel_id: Optional[int] = None,
        since_ms: Optional[int] = None,
        until_ms: Optional[int] = None,
        limit: int = 2000,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        limit = max(1, min(5000, int(limit or 2000)))
        where_sql, params = self._build_where(
            probe_id=probe_id,
            channel_id=channel_id,
            source=source,
            since_ms=since_ms,
            until_ms=until_ms,
        )
        with self.lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"""
                    SELECT
                        shard_key,
                        COUNT(*) AS hit_count,
                        MIN(event_timestamp_ms) AS min_ts,
                        MAX(event_timestamp_ms) AS max_ts,
                        SUM(CASE WHEN clip_vec IS NOT NULL THEN 1 ELSE 0 END) AS clip_count,
                        SUM(CASE WHEN dino_vec IS NOT NULL THEN 1 ELSE 0 END) AS dino_count
                    FROM detections
                    {where_sql}
                    GROUP BY shard_key
                    ORDER BY max_ts DESC
                    LIMIT ?
                    """,
                    tuple(params + [limit]),
                ).fetchall()
                out: List[Dict[str, Any]] = []
                for row in rows:
                    out.append(
                        {
                            "shard_key": row["shard_key"],
                            "hit_count": int(row["hit_count"] or 0),
                            "min_ts": int(row["min_ts"] or 0),
                            "max_ts": int(row["max_ts"] or 0),
                            "clip_count": int(row["clip_count"] or 0),
                            "dino_count": int(row["dino_count"] or 0),
                        }
                    )
                return out
            finally:
                conn.close()

    def summarize_by_source(
        self,
        since_ms: Optional[int] = None,
        channel_id: Optional[int] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        limit = max(1, min(100, int(limit or 20)))
        where_sql, params = self._build_where(
            channel_id=channel_id,
            since_ms=since_ms,
        )
        with self.lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"""
                    SELECT
                        COALESCE(NULLIF(source, ''), 'unknown') AS source_key,
                        COUNT(*) AS row_count,
                        SUM(CASE WHEN thumbnail_b64 IS NOT NULL AND thumbnail_b64 <> '' THEN 1 ELSE 0 END) AS thumbnail_count,
                        SUM(CASE WHEN clip_vec IS NOT NULL THEN 1 ELSE 0 END) AS clip_count,
                        SUM(CASE WHEN dino_vec IS NOT NULL THEN 1 ELSE 0 END) AS dino_count,
                        COUNT(DISTINCT channel_id) AS channel_count,
                        MIN(event_timestamp_ms) AS oldest_timestamp_ms,
                        MAX(event_timestamp_ms) AS newest_timestamp_ms
                    FROM detections
                    {where_sql}
                    GROUP BY source_key
                    ORDER BY newest_timestamp_ms DESC
                    LIMIT ?
                    """,
                    tuple(params + [limit]),
                ).fetchall()
                return [
                    {
                        "source": row["source_key"],
                        "row_count": int(row["row_count"] or 0),
                        "thumbnail_count": int(row["thumbnail_count"] or 0),
                        "clip_count": int(row["clip_count"] or 0),
                        "dino_count": int(row["dino_count"] or 0),
                        "channel_count": int(row["channel_count"] or 0),
                        "oldest_timestamp_ms": int(row["oldest_timestamp_ms"] or 0),
                        "newest_timestamp_ms": int(row["newest_timestamp_ms"] or 0),
                    }
                    for row in rows
                ]
            finally:
                conn.close()

    def storage_summary(self) -> Dict[str, Any]:
        with self.lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT
                        COUNT(*) AS row_count,
                        COALESCE(SUM(LENGTH(thumbnail_b64)), 0) AS thumbnail_bytes,
                        COALESCE(SUM(LENGTH(clip_vec)), 0) AS clip_vec_bytes,
                        COALESCE(SUM(LENGTH(dino_vec)), 0) AS dino_vec_bytes,
                        COALESCE(SUM(LENGTH(payload_json)), 0) AS payload_json_bytes,
                        MIN(event_timestamp_ms) AS oldest_timestamp_ms,
                        MAX(event_timestamp_ms) AS newest_timestamp_ms
                    FROM detections
                    """
                ).fetchone()
            finally:
                conn.close()
        return {
            "backend": self.backend,
            "row_count": int(row["row_count"] or 0) if row else 0,
            "thumbnail_bytes": int(row["thumbnail_bytes"] or 0) if row else 0,
            "clip_vec_bytes": int(row["clip_vec_bytes"] or 0) if row else 0,
            "dino_vec_bytes": int(row["dino_vec_bytes"] or 0) if row else 0,
            "payload_json_bytes": int(row["payload_json_bytes"] or 0) if row else 0,
            "oldest_timestamp_ms": int(row["oldest_timestamp_ms"]) if row and row["oldest_timestamp_ms"] is not None else None,
            "newest_timestamp_ms": int(row["newest_timestamp_ms"]) if row and row["newest_timestamp_ms"] is not None else None,
        }
