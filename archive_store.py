"""PostgreSQL-backed archive repositories for deployment runtime data."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np

from eva_db import PsycopgPool, TransactionContext


NIL_UUID = uuid.UUID(int=0)
ARCHIVE_RUNTIME_REVISION = "20260612_0005"
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


class ArchiveStoreNotReady(RuntimeError):
    """Raised when the configured archive backend has not been migrated yet."""


def _is_missing_archive_relation(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return (
        "undefinedtable" in name
        or "undefined_table" in text
        or "does not exist" in text
    ) and "archive." in text


def _archive_not_ready(exc: Exception) -> ArchiveStoreNotReady:
    return ArchiveStoreNotReady(
        f"Archive storage is not migrated yet. Apply database migration {ARCHIVE_RUNTIME_REVISION}."
    )


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


def _blob_to_vec(blob: Optional[bytes]) -> Optional[np.ndarray]:
    if not blob:
        return None
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.size == 0:
        return None
    return arr.astype(np.float32, copy=True)


def _detection_shard_key(channel_id: int, event_ts_ms: int) -> str:
    ts_sec = max(0.0, float(event_ts_ms) / 1000.0)
    date_key = time.strftime("%Y%m%d", time.localtime(ts_sec))
    return f"ch{channel_id}:{date_key}"


def _decode_detection_payload(payload_json: Optional[str]) -> Optional[Any]:
    if not payload_json:
        return None
    try:
        return json.loads(payload_json)
    except Exception:
        return None


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
    payload_json = (
        json.dumps(payload_obj, ensure_ascii=True)
        if isinstance(payload_obj, (dict, list))
        else None
    )

    image_path = str(record.get("image_path") or "").strip() or None
    if image_path is None and isinstance(payload_obj, dict):
        image_path = str(payload_obj.get("image_path") or "").strip() or None

    clip_value = record.get("clip_vec")
    if clip_value is None:
        clip_value = record.get("clip_embedding")
    dino_value = record.get("dino_vec")
    if dino_value is None:
        dino_value = record.get("dino_embedding")
    clip_vec_blob = _vec_to_blob(clip_value)
    dino_vec_blob = _vec_to_blob(dino_value)

    shard_key = (
        str(record.get("shard_key") or "").strip()
        or _detection_shard_key(channel_id, event_ts)
    )

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


class _TenantRepository:
    backend = "postgres"

    def __init__(self, pool: PsycopgPool, tenant_id: str | uuid.UUID) -> None:
        self.pool = pool
        self.tenant_id = str(uuid.UUID(str(tenant_id)))
        self.lock = threading.RLock()
        self._last_state_hashes: Dict[str, str] = {}

    def _context(self) -> TransactionContext:
        return TransactionContext(tenant_id=self.tenant_id, actor_id=NIL_UUID)


class PostgresDetectionsStore(_TenantRepository):
    """PostgreSQL implementation for archived detection/search records."""

    def __init__(
        self,
        pool: PsycopgPool,
        tenant_id: str | uuid.UUID,
        max_records: int = 20000,
        cap_check_interval_sec: float = 60.0,
    ) -> None:
        super().__init__(pool, tenant_id)
        self.max_records = max(1000, int(max_records or 20000))
        self.cap_check_interval_sec = max(0.0, float(cap_check_interval_sec or 0.0))
        self._last_cap_check_monotonic = 0.0

    @staticmethod
    def _row_to_dict(row: Sequence[Any], include_vectors: bool = False) -> Dict[str, Any]:
        clip_blob = _blob_bytes(row[17])
        dino_blob = _blob_bytes(row[18])
        data: Dict[str, Any] = {
            "id": int(row[0]),
            "timestamp_ms": int(row[1]),
            "recorded_at_ms": int(row[2]),
            "probe_id": row[3],
            "probe_name": row[4],
            "channel_id": int(row[5]),
            "severity": row[6],
            "bookmark_enabled": bool(row[7]),
            "bookmark_sent": bool(row[8]),
            "pos_score": float(row[9] or 0.0),
            "neg_score": float(row[10] or 0.0),
            "margin": float(row[11] or 0.0),
            "thumbnail": row[12],
            "source": row[13],
            "payload": _decode_json_value(row[14]),
            "shard_key": row[15],
            "image_path": row[16],
            "has_clip": clip_blob is not None,
            "has_dino": dino_blob is not None,
        }
        if include_vectors:
            data["clip_vec"] = _blob_to_vec(clip_blob)
            data["dino_vec"] = _blob_to_vec(dino_blob)
        return data

    def health(self) -> Dict[str, Any]:
        try:
            with self.pool.transaction(self._context(), readonly=True) as connection:
                connection.execute("SELECT 1 FROM archive.detections LIMIT 1")
            return {
                "ok": True,
                "status": "reachable",
                "backend": self.backend,
                "tenant_id": self.tenant_id,
            }
        except Exception as exc:
            if _is_missing_archive_relation(exc):
                return {
                    "ok": False,
                    "status": "not_migrated",
                    "backend": self.backend,
                    "tenant_id": self.tenant_id,
                    "required_revision": ARCHIVE_RUNTIME_REVISION,
                }
            return {
                "ok": False,
                "status": "error",
                "backend": self.backend,
                "tenant_id": self.tenant_id,
                "error": type(exc).__name__,
            }

    def add_detection(self, record: Dict[str, Any]) -> bool:
        normalized = _normalize_detection(record)
        with self.lock:
            with self.pool.transaction(self._context()) as connection:
                inserted = self._insert_normalized(connection, normalized)
                self._trim_to_cap(connection)
                return inserted

    def add_detections(self, records: Sequence[Dict[str, Any]]) -> int:
        normalized_rows: List[Dict[str, Any]] = []
        for record in records:
            try:
                normalized_rows.append(_normalize_detection(record))
            except Exception:
                continue
        if not normalized_rows:
            return 0
        inserted = 0
        with self.lock:
            with self.pool.transaction(self._context()) as connection:
                for normalized in normalized_rows:
                    try:
                        if self._insert_normalized(connection, normalized):
                            inserted += 1
                    except Exception:
                        continue
                self._trim_to_cap(connection)
        return inserted

    def _insert_normalized(self, connection: Any, normalized: Mapping[str, Any]) -> bool:
        payload = _decode_detection_payload(
            normalized.get("payload_json") if isinstance(normalized.get("payload_json"), str) else None
        )
        cursor = connection.execute(
            """
            INSERT INTO archive.detections (
                tenant_id,
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
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s
            )
            ON CONFLICT (tenant_id, dedupe_key) DO NOTHING
            """,
            (
                self.tenant_id,
                normalized["dedupe_key"],
                int(normalized["event_timestamp_ms"]),
                int(normalized["recorded_at_ms"]),
                normalized["probe_id"],
                normalized["probe_name"],
                int(normalized["channel_id"]),
                normalized["severity"],
                bool(normalized["bookmark_enabled"]),
                bool(normalized["bookmark_sent"]),
                float(normalized["pos_score"]),
                float(normalized["neg_score"]),
                float(normalized["margin"]),
                normalized["thumbnail_b64"],
                normalized["source"],
                _jsonb(payload) if payload is not None else None,
                normalized["shard_key"],
                normalized["image_path"],
                normalized["clip_vec"],
                normalized["dino_vec"],
                bool(normalized["dino_ready"]),
            ),
        )
        return int(cursor.rowcount or 0) > 0

    def _trim_to_cap(self, connection: Any) -> None:
        now = time.monotonic()
        if (
            self.cap_check_interval_sec > 0
            and self._last_cap_check_monotonic > 0
            and now - self._last_cap_check_monotonic < self.cap_check_interval_sec
        ):
            return
        self._last_cap_check_monotonic = now
        row = connection.execute(
            "SELECT COUNT(*) FROM archive.detections WHERE tenant_id = %s",
            (self.tenant_id,),
        ).fetchone()
        total = int(row[0] or 0) if row else 0
        if total <= self.max_records:
            return
        excess = total - self.max_records
        connection.execute(
            """
            DELETE FROM archive.detections
            WHERE tenant_id = %s
              AND id IN (
                  SELECT id
                  FROM archive.detections
                  WHERE tenant_id = %s
                  ORDER BY id ASC
                  LIMIT %s
              )
            """,
            (self.tenant_id, self.tenant_id, int(excess)),
        )

    def apply_retention(
        self,
        *,
        row_retention_days: float,
        thumbnail_retention_days: float,
        max_records: Optional[int] = None,
        batch_size: int = 5000,
    ) -> Dict[str, Any]:
        batch = max(100, min(50000, int(batch_size or 5000)))
        now_ms = int(time.time() * 1000.0)
        delete_before_ms = _retention_cutoff_ms(now_ms, row_retention_days)
        thumbnail_before_ms = _retention_cutoff_ms(now_ms, thumbnail_retention_days)
        capped_max_records = max(1000, int(max_records or self.max_records))
        deleted_image_paths: List[str] = []
        rows_deleted = 0
        thumbnails_cleared = 0
        cap_rows_deleted = 0

        with self.lock:
            with self.pool.transaction(self._context()) as connection:
                if thumbnail_before_ms is not None:
                    thumb_rows = connection.execute(
                        """
                        SELECT id
                        FROM archive.detections
                        WHERE tenant_id = %s
                          AND thumbnail_b64 IS NOT NULL
                          AND event_timestamp_ms < %s
                        ORDER BY event_timestamp_ms ASC, id ASC
                        LIMIT %s
                        """,
                        (self.tenant_id, int(thumbnail_before_ms), batch),
                    ).fetchall()
                    thumb_ids = [int(row[0]) for row in thumb_rows]
                    if thumb_ids:
                        cursor = connection.execute(
                            f"""
                            UPDATE archive.detections
                            SET thumbnail_b64 = NULL,
                                updated_at = clock_timestamp()
                            WHERE tenant_id = %s
                              AND id IN ({','.join('%s' for _ in thumb_ids)})
                            """,
                            tuple([self.tenant_id] + thumb_ids),
                        )
                        thumbnails_cleared = max(0, int(cursor.rowcount or 0))

                if delete_before_ms is not None:
                    rows_deleted, paths = self._delete_oldest_rows(
                        connection,
                        where_sql="event_timestamp_ms < %s",
                        where_params=[int(delete_before_ms)],
                        limit=batch,
                    )
                    deleted_image_paths.extend(paths)

                count_row = connection.execute(
                    "SELECT COUNT(*) FROM archive.detections WHERE tenant_id = %s",
                    (self.tenant_id,),
                ).fetchone()
                total = int(count_row[0] or 0) if count_row else 0
                if total > capped_max_records:
                    cap_rows_deleted, paths = self._delete_oldest_rows(
                        connection,
                        where_sql="TRUE",
                        where_params=[],
                        limit=min(batch, total - capped_max_records),
                    )
                    deleted_image_paths.extend(paths)

        return {
            "ok": True,
            "backend": self.backend,
            "rows_deleted": rows_deleted,
            "cap_rows_deleted": cap_rows_deleted,
            "thumbnails_cleared": thumbnails_cleared,
            "deleted_image_paths": deleted_image_paths,
            "row_retention_days": float(row_retention_days),
            "thumbnail_retention_days": float(thumbnail_retention_days),
            "max_records": capped_max_records,
            "batch_size": batch,
        }

    def storage_summary(self) -> Dict[str, Any]:
        with self.lock:
            with self.pool.transaction(self._context(), readonly=True) as connection:
                row = connection.execute(
                    """
                    SELECT
                        COUNT(*),
                        COALESCE(SUM(octet_length(thumbnail_b64)), 0),
                        COALESCE(SUM(octet_length(clip_vec)), 0),
                        COALESCE(SUM(octet_length(dino_vec)), 0),
                        COALESCE(SUM(octet_length(payload_json::text)), 0),
                        MIN(event_timestamp_ms),
                        MAX(event_timestamp_ms)
                    FROM archive.detections
                    WHERE tenant_id = %s
                    """,
                    (self.tenant_id,),
                ).fetchone()
        return {
            "backend": self.backend,
            "row_count": int(row[0] or 0) if row else 0,
            "thumbnail_bytes": int(row[1] or 0) if row else 0,
            "clip_vec_bytes": int(row[2] or 0) if row else 0,
            "dino_vec_bytes": int(row[3] or 0) if row else 0,
            "payload_json_bytes": int(row[4] or 0) if row else 0,
            "oldest_timestamp_ms": int(row[5]) if row and row[5] is not None else None,
            "newest_timestamp_ms": int(row[6]) if row and row[6] is not None else None,
        }

    def summarize_by_source(
        self,
        since_ms: Optional[int] = None,
        channel_id: Optional[int] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        limit = max(1, min(100, int(limit or 20)))
        where_sql, params = self._build_where(channel_id=channel_id, since_ms=since_ms)
        try:
            with self.lock:
                with self.pool.transaction(self._context(), readonly=True) as connection:
                    rows = connection.execute(
                        f"""
                        SELECT
                            COALESCE(NULLIF(source, ''), 'unknown'),
                            COUNT(*),
                            SUM(CASE WHEN thumbnail_b64 IS NOT NULL AND thumbnail_b64 <> '' THEN 1 ELSE 0 END),
                            SUM(CASE WHEN clip_vec IS NOT NULL THEN 1 ELSE 0 END),
                            SUM(CASE WHEN dino_vec IS NOT NULL THEN 1 ELSE 0 END),
                            COUNT(DISTINCT channel_id),
                            MIN(event_timestamp_ms),
                            MAX(event_timestamp_ms)
                        FROM archive.detections
                        {where_sql}
                        GROUP BY COALESCE(NULLIF(source, ''), 'unknown')
                        ORDER BY MAX(event_timestamp_ms) DESC
                        LIMIT %s
                        """,
                        tuple(params + [limit]),
                    ).fetchall()
        except Exception as exc:
            if _is_missing_archive_relation(exc):
                raise _archive_not_ready(exc) from exc
            raise
        return [
            {
                "source": row[0],
                "row_count": int(row[1] or 0),
                "thumbnail_count": int(row[2] or 0),
                "clip_count": int(row[3] or 0),
                "dino_count": int(row[4] or 0),
                "channel_count": int(row[5] or 0),
                "oldest_timestamp_ms": int(row[6] or 0),
                "newest_timestamp_ms": int(row[7] or 0),
            }
            for row in rows
        ]

    def _delete_oldest_rows(
        self,
        connection: Any,
        *,
        where_sql: str,
        where_params: Sequence[Any],
        limit: int,
    ) -> Tuple[int, List[str]]:
        capped_limit = max(0, int(limit or 0))
        if capped_limit <= 0:
            return 0, []
        rows = connection.execute(
            f"""
            SELECT id, image_path
            FROM archive.detections
            WHERE tenant_id = %s
              AND {where_sql}
            ORDER BY event_timestamp_ms ASC, id ASC
            LIMIT %s
            """,
            tuple([self.tenant_id] + list(where_params) + [capped_limit]),
        ).fetchall()
        ids = [int(row[0]) for row in rows]
        paths = [str(row[1]) for row in rows if row[1]]
        if not ids:
            return 0, []
        cursor = connection.execute(
            f"""
            DELETE FROM archive.detections
            WHERE tenant_id = %s
              AND id IN ({','.join('%s' for _ in ids)})
            """,
            tuple([self.tenant_id] + ids),
        )
        return max(0, int(cursor.rowcount or 0)), paths

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
        try:
            with self.lock:
                with self.pool.transaction(self._context(), readonly=True) as connection:
                    total_row = connection.execute(
                        f"SELECT COUNT(*) FROM archive.detections {where_sql}",
                        tuple(params),
                    ).fetchone()
                    rows = connection.execute(
                        f"""
                        SELECT {self._select_columns()}
                        FROM archive.detections
                        {where_sql}
                        ORDER BY event_timestamp_ms DESC, id DESC
                        LIMIT %s OFFSET %s
                        """,
                        tuple(params + [limit, offset]),
                    ).fetchall()
        except Exception as exc:
            if _is_missing_archive_relation(exc):
                raise _archive_not_ready(exc) from exc
            raise
        total = int(total_row[0] or 0) if total_row else 0
        return [self._row_to_dict(row) for row in rows], total

    def list_vector_candidates(
        self,
        probe_id: Optional[str] = None,
        channel_id: Optional[int] = None,
        since_ms: Optional[int] = None,
        until_ms: Optional[int] = None,
        limit: int = 20000,
        only_with_clip: bool = True,
        include_vectors: bool = False,
        include_thumbnail: bool = True,
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
        try:
            with self.lock:
                with self.pool.transaction(self._context(), readonly=True) as connection:
                    rows = connection.execute(
                        f"""
                        SELECT {self._select_columns(include_thumbnail=include_thumbnail)}
                        FROM archive.detections
                        {where_sql}
                        ORDER BY event_timestamp_ms DESC, id DESC
                        LIMIT %s
                        """,
                        tuple(params + [limit]),
                    ).fetchall()
        except Exception as exc:
            if _is_missing_archive_relation(exc):
                raise _archive_not_ready(exc) from exc
            raise
        return [self._row_to_dict(row, include_vectors=include_vectors) for row in rows]

    def count_vector_candidates(
        self,
        probe_id: Optional[str] = None,
        channel_id: Optional[int] = None,
        since_ms: Optional[int] = None,
        until_ms: Optional[int] = None,
        only_with_clip: bool = True,
        source: Optional[str] = None,
    ) -> int:
        where_sql, params = self._build_where(
            probe_id=probe_id,
            channel_id=channel_id,
            source=source,
            since_ms=since_ms,
            until_ms=until_ms,
            only_with_clip=only_with_clip,
        )
        try:
            with self.lock:
                with self.pool.transaction(self._context(), readonly=True) as connection:
                    row = connection.execute(
                        f"SELECT COUNT(*) FROM archive.detections {where_sql}",
                        tuple(params),
                    ).fetchone()
        except Exception as exc:
            if _is_missing_archive_relation(exc):
                raise _archive_not_ready(exc) from exc
            raise
        return int(row[0] or 0) if row else 0

    def fetch_detections_by_ids(
        self,
        ids: Sequence[int],
        include_vectors: bool = True,
        include_thumbnail: bool = True,
    ) -> List[Dict[str, Any]]:
        ids_clean = [int(item) for item in ids if item is not None]
        if not ids_clean:
            return []
        where_sql, params = self._build_where(ids=ids_clean)
        try:
            with self.lock:
                with self.pool.transaction(self._context(), readonly=True) as connection:
                    rows = connection.execute(
                        f"""
                        SELECT {self._select_columns(include_thumbnail=include_thumbnail)}
                        FROM archive.detections
                        {where_sql}
                        """,
                        tuple(params),
                    ).fetchall()
        except Exception as exc:
            if _is_missing_archive_relation(exc):
                raise _archive_not_ready(exc) from exc
            raise
        return [self._row_to_dict(row, include_vectors=include_vectors) for row in rows]

    def channel_ids_for_image_path(self, image_path: str) -> frozenset[int]:
        normalized = str(image_path or "").strip()
        if not normalized:
            return frozenset()
        try:
            with self.lock:
                with self.pool.transaction(self._context(), readonly=True) as connection:
                    rows = connection.execute(
                        """
                        SELECT DISTINCT channel_id
                        FROM archive.detections
                        WHERE tenant_id = %s AND image_path = %s
                        """,
                        (self.tenant_id, normalized),
                    ).fetchall()
        except Exception as exc:
            if _is_missing_archive_relation(exc):
                raise _archive_not_ready(exc) from exc
            raise
        return frozenset(int(row[0]) for row in rows if row[0] is not None)

    def update_dino_embeddings(self, rows: Sequence[Tuple[int, Sequence[float]]]) -> int:
        return self._update_embeddings(rows, "dino_vec", set_dino_ready=True)

    def update_clip_embeddings(self, rows: Sequence[Tuple[int, Sequence[float]]]) -> int:
        return self._update_embeddings(rows, "clip_vec", set_dino_ready=False)

    def _update_embeddings(
        self,
        rows: Sequence[Tuple[int, Sequence[float]]],
        column: str,
        *,
        set_dino_ready: bool,
    ) -> int:
        if column not in {"clip_vec", "dino_vec"}:
            raise ValueError("unsupported vector column")
        payload: List[Tuple[bytes, int]] = []
        for det_id, vec in rows:
            blob = _vec_to_blob(vec)
            if blob is not None:
                payload.append((blob, int(det_id)))
        if not payload:
            return 0
        updated = 0
        with self.lock:
            with self.pool.transaction(self._context()) as connection:
                for blob, det_id in payload:
                    if set_dino_ready:
                        cursor = connection.execute(
                            """
                            UPDATE archive.detections
                            SET dino_vec = %s,
                                dino_ready = true,
                                updated_at = clock_timestamp()
                            WHERE tenant_id = %s AND id = %s
                            """,
                            (blob, self.tenant_id, det_id),
                        )
                    else:
                        cursor = connection.execute(
                            """
                            UPDATE archive.detections
                            SET clip_vec = %s,
                                updated_at = clock_timestamp()
                            WHERE tenant_id = %s AND id = %s
                            """,
                            (blob, self.tenant_id, det_id),
                        )
                    updated += max(0, int(cursor.rowcount or 0))
        return updated

    def shard_version(self, shard_key: str, embedder: str = "clip") -> Tuple[int, int]:
        column = "dino_vec" if embedder == "dino" else "clip_vec"
        try:
            with self.lock:
                with self.pool.transaction(self._context(), readonly=True) as connection:
                    row = connection.execute(
                        f"""
                        SELECT COUNT(*), COALESCE(MAX(id), 0)
                        FROM archive.detections
                        WHERE tenant_id = %s
                          AND shard_key = %s
                          AND {column} IS NOT NULL
                        """,
                        (self.tenant_id, str(shard_key)),
                    ).fetchone()
        except Exception as exc:
            if _is_missing_archive_relation(exc):
                raise _archive_not_ready(exc) from exc
            raise
        if not row:
            return 0, 0
        return int(row[0] or 0), int(row[1] or 0)

    def load_shard_vectors(self, shard_key: str, embedder: str = "clip") -> Tuple[List[int], np.ndarray]:
        column = "dino_vec" if embedder == "dino" else "clip_vec"
        try:
            with self.lock:
                with self.pool.transaction(self._context(), readonly=True) as connection:
                    rows = connection.execute(
                        f"""
                        SELECT id, {column}
                        FROM archive.detections
                        WHERE tenant_id = %s
                          AND shard_key = %s
                          AND {column} IS NOT NULL
                        ORDER BY id ASC
                        """,
                        (self.tenant_id, str(shard_key)),
                    ).fetchall()
        except Exception as exc:
            if _is_missing_archive_relation(exc):
                raise _archive_not_ready(exc) from exc
            raise
        ids: List[int] = []
        vectors: List[np.ndarray] = []
        target_dim: Optional[int] = None
        for row in rows:
            vec = _blob_to_vec(_blob_bytes(row[1]))
            if vec is None:
                continue
            if target_dim is None:
                target_dim = int(vec.shape[0]) if vec.ndim == 1 else None
            if target_dim is None or vec.ndim != 1 or int(vec.shape[0]) != target_dim:
                continue
            ids.append(int(row[0]))
            vectors.append(vec)
        if not vectors:
            return [], np.zeros((0, 0), dtype=np.float32)
        return ids, np.stack(vectors, axis=0).astype(np.float32, copy=False)

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
        try:
            with self.lock:
                with self.pool.transaction(self._context(), readonly=True) as connection:
                    rows = connection.execute(
                        f"""
                        SELECT
                            probe_id,
                            MAX(probe_name),
                            channel_id,
                            {SOURCE_GROUP_SQL} AS source_key,
                            COUNT(*),
                            MAX(event_timestamp_ms)
                        FROM archive.detections
                        {where_sql}
                        GROUP BY probe_id, channel_id, {SOURCE_GROUP_SQL}
                        ORDER BY MAX(event_timestamp_ms) DESC
                        LIMIT %s
                        """,
                        tuple(params + [limit]),
                    ).fetchall()
        except Exception as exc:
            if _is_missing_archive_relation(exc):
                raise _archive_not_ready(exc) from exc
            raise
        return [
            {
                "probe_id": row[0],
                "probe_name": row[1],
                "channel_id": int(row[2]),
                "source": row[3],
                "hit_count": int(row[4] or 0),
                "latest_timestamp_ms": int(row[5] or 0),
            }
            for row in rows
        ]

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
        try:
            with self.lock:
                with self.pool.transaction(self._context(), readonly=True) as connection:
                    rows = connection.execute(
                        f"""
                        SELECT
                            shard_key,
                            COUNT(*),
                            MIN(event_timestamp_ms),
                            MAX(event_timestamp_ms),
                            SUM(CASE WHEN clip_vec IS NOT NULL THEN 1 ELSE 0 END),
                            SUM(CASE WHEN dino_vec IS NOT NULL THEN 1 ELSE 0 END)
                        FROM archive.detections
                        {where_sql}
                        GROUP BY shard_key
                        ORDER BY MAX(event_timestamp_ms) DESC
                        LIMIT %s
                        """,
                        tuple(params + [limit]),
                    ).fetchall()
        except Exception as exc:
            if _is_missing_archive_relation(exc):
                raise _archive_not_ready(exc) from exc
            raise
        return [
            {
                "shard_key": row[0],
                "hit_count": int(row[1] or 0),
                "min_ts": int(row[2] or 0),
                "max_ts": int(row[3] or 0),
                "clip_count": int(row[4] or 0),
                "dino_count": int(row[5] or 0),
            }
            for row in rows
        ]

    @staticmethod
    def _select_columns(*, include_thumbnail: bool = True) -> str:
        thumbnail_expr = "thumbnail_b64" if include_thumbnail else "NULL::text AS thumbnail_b64"
        return (
            "id, event_timestamp_ms, recorded_at_ms, probe_id, probe_name, "
            "channel_id, severity, bookmark_enabled, bookmark_sent, pos_score, "
            f"neg_score, margin, {thumbnail_expr}, source, payload_json, shard_key, "
            "image_path, clip_vec, dino_vec"
        )

    def _build_where(
        self,
        probe_id: Optional[str] = None,
        channel_id: Optional[int] = None,
        source: Optional[str] = None,
        since_ms: Optional[int] = None,
        until_ms: Optional[int] = None,
        only_with_clip: bool = False,
        only_with_dino: bool = False,
        ids: Optional[Sequence[int]] = None,
    ) -> Tuple[str, List[Any]]:
        where: List[str] = ["tenant_id = %s"]
        params: List[Any] = [self.tenant_id]
        if probe_id:
            where.append("probe_id = %s")
            params.append(str(probe_id))
        if channel_id is not None:
            where.append("channel_id = %s")
            params.append(int(channel_id))
        if source:
            normalized_source = ARCHIVE_SOURCE_ALIASES.get(str(source).strip().lower(), str(source).strip().lower())
            if normalized_source == "probe":
                placeholders = ",".join("%s" for _ in PROBE_SOURCE_ALIASES)
                where.append(f"source IN ({placeholders})")
                params.extend(sorted(PROBE_SOURCE_ALIASES))
            else:
                where.append("source = %s")
                params.append(normalized_source)
        if since_ms is not None:
            where.append("event_timestamp_ms >= %s")
            params.append(int(since_ms))
        if until_ms is not None:
            where.append("event_timestamp_ms <= %s")
            params.append(int(until_ms))
        if only_with_clip:
            where.append("clip_vec IS NOT NULL")
        if only_with_dino:
            where.append("dino_vec IS NOT NULL")
        if ids is not None:
            ids_clean = [int(item) for item in ids]
            if ids_clean:
                where.append(f"id IN ({','.join('%s' for _ in ids_clean)})")
                params.extend(ids_clean)
            else:
                where.append("1 = 0")
        return f"WHERE {' AND '.join(where)}", params


class PostgresProbesStore(_TenantRepository):
    """PostgreSQL-backed replacement for the JSON probe store."""

    def health(self) -> Dict[str, Any]:
        try:
            with self.pool.transaction(self._context(), readonly=True) as connection:
                connection.execute("SELECT 1 FROM archive.probes LIMIT 1")
            return {
                "ok": True,
                "status": "reachable",
                "backend": self.backend,
                "tenant_id": self.tenant_id,
            }
        except Exception as exc:
            return {
                "ok": False,
                "status": "error",
                "backend": self.backend,
                "tenant_id": self.tenant_id,
                "error": type(exc).__name__,
            }

    def list_probes(self) -> List[Dict[str, Any]]:
        with self.lock:
            with self.pool.transaction(self._context(), readonly=True) as connection:
                rows = connection.execute(
                    """
                    SELECT payload_json
                    FROM archive.probes
                    WHERE tenant_id = %s
                    ORDER BY updated_at DESC, probe_id ASC
                    """,
                    (self.tenant_id,),
                ).fetchall()
        probes: List[Dict[str, Any]] = []
        for row in rows:
            payload = _decode_json_value(row[0])
            if isinstance(payload, dict):
                probes.append(copy.deepcopy(payload))
        return probes

    def upsert_probe(self, probe: Dict[str, Any]) -> Dict[str, Any]:
        stored_probe = copy.deepcopy(probe)
        if not stored_probe.get("id"):
            stored_probe["id"] = f"probe-{uuid.uuid4().hex[:12]}"
        probe_id = str(stored_probe.get("id") or "").strip()
        if not probe_id:
            raise ValueError("probe id is required")
        with self.lock:
            with self.pool.transaction(self._context()) as connection:
                connection.execute(
                    """
                    INSERT INTO archive.probes (
                        tenant_id,
                        probe_id,
                        payload_json
                    )
                    VALUES (%s, %s, %s)
                    ON CONFLICT (tenant_id, probe_id) DO UPDATE
                    SET payload_json = EXCLUDED.payload_json,
                        updated_at = clock_timestamp()
                    """,
                    (self.tenant_id, probe_id, _jsonb(stored_probe)),
                )
        return copy.deepcopy(stored_probe)

    def patch_probe_runtime(
        self,
        probe_id: str,
        changes: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Atomically merge daemon-owned runtime fields into an existing probe.

        This deliberately uses UPDATE rather than UPSERT: a late daemon result
        must not recreate a probe that the operator deleted, and it must not
        overwrite concurrently edited thresholds, prompts, or bookmark policy.
        """

        normalized = str(probe_id or "").strip()
        if not normalized:
            raise ValueError("probe id is required")
        allowed_runtime_fields = {
            "last_hit",
            "recent_hits",
            "bookmark_gate",
            "bookmark_gate_updated_at_ms",
        }
        runtime_patch = {
            str(key): copy.deepcopy(value)
            for key, value in dict(changes or {}).items()
            if str(key) in allowed_runtime_fields
        }
        if not runtime_patch:
            return None
        with self.lock:
            with self.pool.transaction(self._context()) as connection:
                row = connection.execute(
                    """
                    UPDATE archive.probes
                    SET payload_json = payload_json || %s,
                        updated_at = clock_timestamp()
                    WHERE tenant_id = %s AND probe_id = %s
                    RETURNING payload_json
                    """,
                    (_jsonb(runtime_patch), self.tenant_id, normalized),
                ).fetchone()
        if row is None:
            return None
        payload = _decode_json_value(row[0])
        return copy.deepcopy(payload) if isinstance(payload, dict) else None

    def delete_probe(self, probe_id: str) -> bool:
        normalized = str(probe_id or "").strip()
        if not normalized:
            return False
        with self.lock:
            with self.pool.transaction(self._context()) as connection:
                cursor = connection.execute(
                    """
                    DELETE FROM archive.probes
                    WHERE tenant_id = %s AND probe_id = %s
                    """,
                    (self.tenant_id, normalized),
                )
        return int(cursor.rowcount or 0) > 0


class PostgresRuntimeStateStore(_TenantRepository):
    """Small JSONB document store for Luxriot summary and rollup runtime state."""

    _SUMMARY_STATE_KEY = "luxriot_summary_state"
    _SUMMARY_META_KEY = "luxriot_summary_state:meta"
    _SUMMARY_HISTORY_PREFIX = "luxriot_summary_state:history:"
    _SUMMARY_RUNS_PREFIX = "luxriot_summary_state:runs:"

    def health(self) -> Dict[str, Any]:
        try:
            with self.pool.transaction(self._context(), readonly=True) as connection:
                connection.execute("SELECT 1 FROM archive.runtime_state LIMIT 1")
            return {
                "ok": True,
                "status": "reachable",
                "backend": self.backend,
                "tenant_id": self.tenant_id,
            }
        except Exception as exc:
            return {
                "ok": False,
                "status": "error",
                "backend": self.backend,
                "tenant_id": self.tenant_id,
                "error": type(exc).__name__,
            }

    def load_state(self, key: str) -> Optional[Dict[str, Any]]:
        normalized = _state_key(key)
        if normalized == self._SUMMARY_STATE_KEY:
            split_payload = self._load_split_summary_state()
            if split_payload is not None:
                return split_payload
        with self.lock:
            with self.pool.transaction(self._context(), readonly=True) as connection:
                row = connection.execute(
                    """
                    SELECT payload_json
                    FROM archive.runtime_state
                    WHERE tenant_id = %s AND state_key = %s
                    """,
                    (self.tenant_id, normalized),
                ).fetchone()
        if row is None:
            return None
        payload = _decode_json_value(row[0])
        return copy.deepcopy(payload) if isinstance(payload, dict) else None

    def save_state(self, key: str, payload: Mapping[str, Any]) -> None:
        normalized = _state_key(key)
        if normalized == self._SUMMARY_STATE_KEY:
            self._save_split_summary_state(payload)
            return
        self._save_state_exact(normalized, payload)

    def _save_state_exact(self, key: str, payload: Mapping[str, Any]) -> None:
        payload_dict = dict(payload)
        digest = _payload_digest(payload_dict)
        cache_key = f"exact:{key}"
        if self._last_state_hashes.get(cache_key) == digest:
            return
        with self.lock:
            with self.pool.transaction(self._context()) as connection:
                self._upsert_state_locked(connection, key, payload_dict)
                self._last_state_hashes[cache_key] = digest

    def _save_split_summary_state(self, payload: Mapping[str, Any]) -> None:
        history_raw = payload.get("summary_history")
        runs_raw = payload.get("summary_runs")
        meta_payload = {
            "version": payload.get("version", 2),
            "revision": payload.get("revision", 0),
            "updated_at": payload.get("updated_at"),
            "channel_routines": _plain_value(payload.get("channel_routines") or {}),
            "road_scene_calibrations": _plain_value(
                payload.get("road_scene_calibrations") or {}
            ),
            "capture_baselines": _plain_value(payload.get("capture_baselines") or {}),
            "prompt_settings": _plain_value(payload.get("prompt_settings") or {}),
        }
        entries: Dict[str, Dict[str, Any]] = {self._SUMMARY_META_KEY: meta_payload}
        if isinstance(history_raw, Mapping):
            for channel_id, logs in history_raw.items():
                entries[f"{self._SUMMARY_HISTORY_PREFIX}{channel_id}"] = {
                    "logs": _plain_value(
                        logs
                        if isinstance(logs, Sequence)
                        and not isinstance(logs, (str, bytes, bytearray))
                        else []
                    )
                }
        if isinstance(runs_raw, Mapping):
            for channel_id, runs in runs_raw.items():
                entries[f"{self._SUMMARY_RUNS_PREFIX}{channel_id}"] = {
                    "runs": _plain_value(
                        runs
                        if isinstance(runs, Sequence)
                        and not isinstance(runs, (str, bytes, bytearray))
                        else []
                    )
                }

        with self.lock:
            with self.pool.transaction(self._context()) as connection:
                for state_key, entry_payload in entries.items():
                    normalized_key = _state_key(state_key)
                    digest = _payload_digest(entry_payload)
                    cache_key = f"split:{normalized_key}"
                    if self._last_state_hashes.get(cache_key) == digest:
                        continue
                    self._upsert_state_locked(connection, normalized_key, entry_payload)
                    self._last_state_hashes[cache_key] = digest

    def _load_split_summary_state(self) -> Optional[Dict[str, Any]]:
        with self.lock:
            with self.pool.transaction(self._context(), readonly=True) as connection:
                rows = connection.execute(
                    """
                    SELECT state_key, payload_json
                    FROM archive.runtime_state
                    WHERE tenant_id = %s
                      AND (
                          state_key = %s
                          OR state_key LIKE %s
                      )
                    """,
                    (
                        self.tenant_id,
                        self._SUMMARY_META_KEY,
                        f"{self._SUMMARY_STATE_KEY}:%",
                    ),
                ).fetchall()
        if not rows:
            return None
        meta: Dict[str, Any] = {}
        summary_history: Dict[str, Any] = {}
        summary_runs: Dict[str, Any] = {}
        for key, raw_payload in rows:
            state_key = str(key or "")
            payload = _decode_json_value(raw_payload)
            if not isinstance(payload, dict):
                continue
            if state_key == self._SUMMARY_META_KEY:
                meta = payload
            elif state_key.startswith(self._SUMMARY_HISTORY_PREFIX):
                channel_id = state_key[len(self._SUMMARY_HISTORY_PREFIX) :]
                logs = payload.get("logs")
                if isinstance(logs, list):
                    summary_history[str(channel_id)] = logs
            elif state_key.startswith(self._SUMMARY_RUNS_PREFIX):
                channel_id = state_key[len(self._SUMMARY_RUNS_PREFIX) :]
                runs = payload.get("runs")
                if isinstance(runs, list):
                    summary_runs[str(channel_id)] = runs
        return {
            "version": meta.get("version", 2),
            "revision": meta.get("revision", 0),
            "updated_at": meta.get("updated_at"),
            "summary_history": summary_history,
            "summary_runs": summary_runs,
            "channel_routines": meta.get("channel_routines") or {},
            "road_scene_calibrations": meta.get("road_scene_calibrations") or {},
            "capture_baselines": meta.get("capture_baselines") or {},
            "prompt_settings": meta.get("prompt_settings") or {},
        }

    def _upsert_state_locked(
        self,
        connection: Any,
        key: str,
        payload: Mapping[str, Any],
    ) -> None:
        connection.execute(
            """
            INSERT INTO archive.runtime_state (
                tenant_id,
                state_key,
                payload_json
            )
            VALUES (%s, %s, %s)
            ON CONFLICT (tenant_id, state_key) DO UPDATE
            SET payload_json = EXCLUDED.payload_json,
                updated_at = clock_timestamp()
            """,
            (self.tenant_id, key, _jsonb(dict(payload))),
        )


def _state_key(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        raise ValueError("state key is required")
    if len(normalized) > 128:
        raise ValueError("state key is too long")
    return normalized


def _retention_cutoff_ms(now_ms: int, days: float) -> Optional[int]:
    try:
        value = float(days)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return int(now_ms - value * 86400.0 * 1000.0)


def _decode_json_value(value: Any) -> Optional[Any]:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        value = bytes(value).decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return None
    return None


def _blob_bytes(value: Any) -> Optional[bytes]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    return value


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_plain_value(item) for item in sorted(value, key=repr)]
    return value


def _payload_digest(value: Any) -> str:
    encoded = json.dumps(
        _plain_value(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonb(value: Any) -> Any:
    from psycopg.types.json import Jsonb

    payload = _plain_value(value)
    json.dumps(payload, allow_nan=False)
    return Jsonb(payload)
