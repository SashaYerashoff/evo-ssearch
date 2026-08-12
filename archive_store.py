"""PostgreSQL-backed archive repositories for deployment runtime data."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np

from eva_db import PsycopgPool, TransactionContext
from embedding_space import (
    embedding_space_fingerprint,
    embedding_space_requires_identity,
    normalize_embedding_space,
)


NIL_UUID = uuid.UUID(int=0)
ARCHIVE_RUNTIME_REVISION = "20260612_0005"
ALERT_FEEDBACK_REVISION = "20260725_0007"
ALERT_FEEDBACK_REASON_LABELS: Dict[str, str] = {
    "no_relevant_event": "No relevant event",
    "benign_activity": "Benign activity",
    "wrong_object_or_actor": "Wrong object or actor",
    "duplicate_or_stale": "Duplicate or stale alert",
    "poor_visual_quality": "Poor visual quality",
}
PROBE_SOURCE_ALIASES = frozenset({"probe", "probes_run", "probes_query", "probe_daemon"})
ARCHIVE_SOURCE_ALIASES = {
    "probe": "probe",
    "probes_run": "probe",
    "probes_query": "probe",
    "probe_daemon": "probe",
    "semantic_snapshot": "semantic_snapshot",
    "semantic_snapshots": "semantic_snapshot",
    "continuous_clip": "semantic_snapshot",
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


def _feedback_not_ready(exc: Exception) -> ArchiveStoreNotReady:
    return ArchiveStoreNotReady(
        f"Alert feedback storage is not migrated yet. Apply database migration {ALERT_FEEDBACK_REVISION}."
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

    def _context(self, actor_id: str | uuid.UUID | None = None) -> TransactionContext:
        return TransactionContext(
            tenant_id=self.tenant_id,
            actor_id=actor_id if actor_id is not None else NIL_UUID,
        )


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
        # Kept for constructor compatibility. Hard-cap deletion is performed
        # only by apply_retention(), which returns image paths to the caller so
        # database rows and filesystem snapshots are pruned together.
        self.cap_check_interval_sec = max(0.0, float(cap_check_interval_sec or 0.0))

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
        return inserted

    def ensure_detections(self, records: Sequence[Dict[str, Any]]) -> int:
        """Durably apply a validated idempotent batch.

        A deterministic dedupe conflict means that row is already durable, so
        it counts as success. Unlike the legacy best-effort bulk helper, this
        method propagates normalization/database errors and lets the caller
        retry or record an explicit coverage gap.
        """

        normalized_rows = [_normalize_detection(record) for record in records]
        if not normalized_rows:
            return 0
        with self.lock:
            with self.pool.transaction(self._context()) as connection:
                for normalized in normalized_rows:
                    self._insert_normalized(connection, normalized)
        return len(normalized_rows)

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
        include_thumbnail: bool = True,
        channel_ids: Optional[Sequence[int]] = None,
        batch_id: Optional[str] = None,
        parent_alert_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        limit = max(1, min(500, int(limit or 50)))
        offset = max(0, int(offset or 0))
        where_sql, params = self._build_where(
            probe_id=probe_id,
            channel_id=channel_id,
            channel_ids=channel_ids,
            source=source,
            since_ms=since_ms,
            until_ms=until_ms,
            batch_id=batch_id,
            parent_alert_id=parent_alert_id,
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
                        SELECT {self._select_columns(include_thumbnail=include_thumbnail)}
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

    def list_vlm_summary_batches(
        self,
        *,
        channel_id: int,
        since_ms: Optional[int] = None,
        until_ms: Optional[int] = None,
        limit: int = 120,
        offset: int = 0,
        return_page_info: bool = False,
    ) -> (
        Tuple[List[Dict[str, Any]], int]
        | Tuple[List[Dict[str, Any]], int, Dict[str, Any]]
    ):
        """Return one compact text row per archived VLM batch.

        Each archived batch has several evidence-frame rows. Grouping them in
        PostgreSQL avoids sending the same summary payload and large inline
        thumbnail columns repeatedly to the operator history reader. The
        returned thumbnail id points to one frame that was actually submitted
        to the VLM; burst/selector evidence is preferred when available.
        """
        limit = max(1, min(1000, int(limit or 120)))
        offset = max(0, int(offset or 0))
        where_sql, params = self._build_where(
            channel_id=channel_id,
            source="vlm_summary",
            since_ms=since_ms,
            until_ms=until_ms,
        )
        # A summary batch currently produces up to 16 archive evidence rows.  The
        # previous query grouped the entire lifetime of a channel and ran
        # COUNT(*) OVER() before returning its first page.  On long-lived streams
        # that turned a cheap history read into a 15-25 second archive lock which
        # also stalled live L0 writes.  Read the source/channel time index in
        # bounded chunks instead.  `total` is an honest lower bound until the scan
        # reaches the end; callers that need paging metadata can request it.
        target_batches = offset + limit + 1
        # Four archived evidence rows is the normal batch density.  Start near
        # one requested page and grow by another chunk only for denser batches;
        # reading twelve rows per batch made the first page needlessly pull
        # several hundred kilobytes of JSON from PostgreSQL.
        chunk_size = max(256, min(1024, target_batches * 4))
        max_scan_rows = max(4096, target_batches * 32)
        raw_offset = 0
        raw_has_more = True
        batch_rows: Dict[str, Sequence[Any]] = {}
        batch_priorities: Dict[str, Tuple[int, int, int, int, int]] = {}
        role_rank = {
            "burst_companion": 0,
            "burst_apex": 1,
            "sample": 2,
            "only": 3,
            "last": 4,
            "first": 5,
        }
        try:
            # PostgreSQL supplies a consistent readonly snapshot; serializing it
            # behind the writer mutex only makes operator history wait for CLIP /
            # SigLIP archive writes and can stall live ingestion in the opposite
            # direction.
            with self.pool.transaction(self._context(), readonly=True) as connection:
                    while (
                        raw_has_more
                        and len(batch_rows) < target_batches
                        and raw_offset < max_scan_rows
                    ):
                        fetch_limit = min(chunk_size, max_scan_rows - raw_offset)
                        rows = connection.execute(
                            f"""
                            SELECT
                                id,
                                event_timestamp_ms,
                                COALESCE(NULLIF(payload_json->>'batch_id', ''), '') AS batch_id,
                                thumbnail_b64 IS NOT NULL AS has_thumbnail,
                                COALESCE(NULLIF(payload_json->>'anchor_role', ''), 'sample') AS anchor_role,
                                COALESCE(NULLIF(payload_json->>'frame_index', '')::integer, 0) AS frame_index,
                                COALESCE(NULLIF(payload_json->>'run_id', ''), 'manual') AS run_id,
                                COALESCE(
                                    NULLIF(payload_json->>'batch_start_ms', '')::bigint,
                                    event_timestamp_ms
                                ) AS batch_start_ms,
                                COALESCE(
                                    NULLIF(payload_json->>'batch_end_ms', '')::bigint,
                                    event_timestamp_ms
                                ) AS batch_end_ms,
                                lower(COALESCE(payload_json->>'is_cover', 'false'))
                                    IN ('true', '1', 'yes', 'on') AS is_cover
                            FROM archive.detections
                            {where_sql}
                            ORDER BY
                                event_timestamp_ms DESC,
                                id DESC
                            LIMIT %s OFFSET %s
                            """,
                            tuple(params + [fetch_limit, raw_offset]),
                        ).fetchall()
                        raw_offset += len(rows)
                        raw_has_more = len(rows) == fetch_limit
                        for row in rows:
                            run_id = str(row[6] or "manual")
                            batch_start_ms = int(row[7] or row[1] or 0)
                            batch_end_ms = int(row[8] or row[1] or batch_start_ms)
                            batch_key = str(row[2] or "").strip()
                            if not batch_key:
                                batch_key = f"{run_id}:{batch_start_ms}:{batch_end_ms}"
                            anchor_role = str(row[4] or "sample")
                            priority = (
                                1 if bool(row[3]) else 0,
                                1 if bool(row[9]) else 0,
                                -role_rank.get(anchor_role, 6),
                                int(row[1] or 0),
                                int(row[0] or 0),
                            )
                            if priority > batch_priorities.get(
                                batch_key,
                                (-1, -1, -99, -1, -1),
                            ):
                                batch_rows[batch_key] = row
                                batch_priorities[batch_key] = priority
        except Exception as exc:
            if _is_missing_archive_relation(exc):
                raise _archive_not_ready(exc) from exc
            raise

        ordered_rows = sorted(
            batch_rows.values(),
            key=lambda row: (int(row[8] or row[1] or 0), int(row[0] or 0)),
            reverse=True,
        )
        page_rows = ordered_rows[offset : offset + limit]
        payload_by_id: Dict[int, Mapping[str, Any]] = {}
        page_ids = [int(row[0]) for row in page_rows]
        if page_ids:
            try:
                with self.pool.transaction(
                    self._context(),
                    readonly=True,
                ) as connection:
                    payload_rows = connection.execute(
                        f"""
                        SELECT id, payload_json
                        FROM archive.detections
                        WHERE tenant_id = %s
                          AND id IN ({','.join('%s' for _ in page_ids)})
                        """,
                        tuple([self.tenant_id] + page_ids),
                    ).fetchall()
                for payload_row in payload_rows:
                    payload = _decode_json_value(payload_row[1])
                    if isinstance(payload, Mapping):
                        payload_by_id[int(payload_row[0])] = payload
            except Exception as exc:
                if _is_missing_archive_relation(exc):
                    raise _archive_not_ready(exc) from exc
                raise
        total_exact = not raw_has_more
        has_more = len(ordered_rows) > offset + limit or raw_has_more
        total = (
            len(ordered_rows)
            if total_exact
            else max(len(ordered_rows), offset + len(page_rows) + (1 if has_more else 0))
        )
        logs: List[Dict[str, Any]] = []
        for row in page_rows:
            payload = payload_by_id.get(int(row[0]))
            if not isinstance(payload, Mapping):
                continue
            vector_signal_raw = payload.get("vector_signal")
            capture_attention = (
                vector_signal_raw.get("capture_attention")
                if isinstance(vector_signal_raw, Mapping)
                else None
            )
            compact_vector_signal = (
                {"capture_attention": dict(capture_attention)}
                if isinstance(capture_attention, Mapping)
                else {}
            )
            batch_start_ms = int(row[7] or payload.get("batch_start_ms") or row[1] or 0)
            batch_end_ms = int(row[8] or payload.get("batch_end_ms") or row[1] or batch_start_ms)
            logs.append(
                {
                    "archive_id": int(row[0]),
                    "channel_id": int(channel_id),
                    "run_id": str(payload.get("run_id") or "").strip(),
                    "created_at": float(batch_end_ms) / 1000.0,
                    "batch_start_ms": batch_start_ms,
                    "batch_end_ms": batch_end_ms,
                    "frame_count": int(payload.get("frame_count") or payload.get("batch_size") or 0),
                    "batch_size": int(payload.get("batch_size") or 0),
                    "duration_sec": float(payload.get("duration_sec") or 0.0),
                    "summary": str(payload.get("summary") or ""),
                    "alert_counts": dict(payload.get("alert_counts") or {})
                    if isinstance(payload.get("alert_counts"), Mapping)
                    else {},
                    "alert_total": int(payload.get("alert_total") or 0),
                    "bookmarks_sent": int(payload.get("bookmarks_sent") or 0),
                    "vector_signal": compact_vector_signal,
                    "thumbnail_detection_id": int(row[0]) if bool(row[3]) else None,
                    "thumbnail_role": str(row[4] or "sample"),
                    "thumbnail_frame_index": int(row[5] or 0),
                    "thumbnail_selection_source": str(
                        payload.get("cover_source")
                        or payload.get("selection_source")
                        or ""
                    ).strip(),
                    "thumbnail_is_cover": bool(payload.get("is_cover")),
                    "thumbnail_snapshot_index": int(
                        payload.get("snapshot_index")
                        or int(row[5] or 0) + 1
                    ),
                    "cover_kind": str(payload.get("cover_kind") or "").strip(),
                    "cover_reason": str(payload.get("cover_reason") or "").strip(),
                    "cover_confidence": str(
                        payload.get("cover_confidence") or ""
                    ).strip(),
                    "batch_id": str(payload.get("batch_id") or "").strip(),
                    "archive_backed": True,
                }
            )
        if return_page_info:
            return logs, total, {
                "has_more": bool(has_more),
                "total_exact": bool(total_exact),
                "scanned_rows": int(raw_offset),
            }
        return logs, total

    def list_vlm_summary_buckets(
        self,
        *,
        channel_id: int,
        since_ms: int,
        until_ms: int,
        bucket_sec: int = 900,
    ) -> List[Dict[str, Any]]:
        """List archive-backed L0 text coverage grouped into rollup windows."""

        normalized_bucket_sec = max(300, min(86400, int(bucket_sec)))
        bucket_ms = normalized_bucket_sec * 1000
        where_sql, params = self._build_where(
            channel_id=channel_id,
            source="vlm_summary",
            since_ms=int(since_ms),
            until_ms=int(until_ms),
        )
        try:
            with self.lock:
                with self.pool.transaction(self._context(), readonly=True) as connection:
                    rows = connection.execute(
                        f"""
                        WITH candidates AS (
                            SELECT DISTINCT ON (
                                COALESCE(NULLIF(payload_json->>'run_id', ''), 'manual'),
                                COALESCE(NULLIF(payload_json->>'batch_start_ms', '')::bigint, event_timestamp_ms),
                                COALESCE(NULLIF(payload_json->>'batch_end_ms', '')::bigint, event_timestamp_ms)
                            )
                                COALESCE(NULLIF(payload_json->>'batch_start_ms', '')::bigint, event_timestamp_ms) AS batch_start_ms,
                                COALESCE(NULLIF(payload_json->>'batch_end_ms', '')::bigint, event_timestamp_ms) AS batch_end_ms,
                                payload_json
                            FROM archive.detections
                            {where_sql}
                              AND COALESCE(payload_json->>'summary', '') <> ''
                            ORDER BY
                                COALESCE(NULLIF(payload_json->>'run_id', ''), 'manual'),
                                COALESCE(NULLIF(payload_json->>'batch_start_ms', '')::bigint, event_timestamp_ms),
                                COALESCE(NULLIF(payload_json->>'batch_end_ms', '')::bigint, event_timestamp_ms),
                                event_timestamp_ms DESC,
                                id DESC
                        )
                        SELECT
                            (FLOOR(batch_start_ms::numeric / %s) * %s)::bigint AS bucket_start_ms,
                            COUNT(*)::bigint AS batch_count,
                            MIN(batch_start_ms)::bigint AS first_batch_start_ms,
                            MAX(batch_end_ms)::bigint AS last_batch_end_ms,
                            SUM(COALESCE(NULLIF(payload_json->>'frame_count', '')::integer, 0))::bigint AS frame_count
                        FROM candidates
                        GROUP BY bucket_start_ms
                        ORDER BY bucket_start_ms ASC
                        """,
                        tuple(params + [bucket_ms, bucket_ms]),
                    ).fetchall()
        except Exception as exc:
            if _is_missing_archive_relation(exc):
                raise _archive_not_ready(exc) from exc
            raise
        return [
            {
                "window_start": float(int(row[0])) / 1000.0,
                "window_end": float(int(row[0]) + bucket_ms) / 1000.0,
                "batch_count": int(row[1] or 0),
                "first_batch_start_ms": int(row[2] or 0),
                "last_batch_end_ms": int(row[3] or 0),
                "frame_count": int(row[4] or 0),
            }
            for row in rows
        ]

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
        channel_ids: Optional[Sequence[int]] = None,
        embedding_space: Optional[Mapping[str, Any]] = None,
        allow_legacy_embedding_space: bool = False,
    ) -> List[Dict[str, Any]]:
        limit = max(1, min(100000, int(limit or 20000)))
        where_sql, params = self._build_where(
            probe_id=probe_id,
            channel_id=channel_id,
            channel_ids=channel_ids,
            source=source,
            since_ms=since_ms,
            until_ms=until_ms,
            only_with_clip=only_with_clip,
            embedding_space=embedding_space,
            allow_legacy_embedding_space=allow_legacy_embedding_space,
        )
        try:
            with self.lock:
                with self.pool.transaction(self._context(), readonly=True) as connection:
                    rows = connection.execute(
                        f"""
                        SELECT {self._select_columns(
                            include_thumbnail=include_thumbnail,
                            include_vectors=include_vectors,
                            include_payload=False,
                        )}
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
        channel_ids: Optional[Sequence[int]] = None,
        embedding_space: Optional[Mapping[str, Any]] = None,
        allow_legacy_embedding_space: bool = False,
    ) -> int:
        where_sql, params = self._build_where(
            probe_id=probe_id,
            channel_id=channel_id,
            channel_ids=channel_ids,
            source=source,
            since_ms=since_ms,
            until_ms=until_ms,
            only_with_clip=only_with_clip,
            embedding_space=embedding_space,
            allow_legacy_embedding_space=allow_legacy_embedding_space,
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
                        SELECT {self._select_columns(
                            include_thumbnail=include_thumbnail,
                            include_vectors=include_vectors,
                        )}
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

    def resolve_vlm_snapshot_refs(
        self,
        refs: Sequence[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Resolve durable ``<batch>:snapshot:<n>`` refs to archive rows.

        L0 incident memory intentionally stores compact, stable batch references
        instead of copying image payloads into the incident tables.  The review
        board still needs a thumbnail, so resolve every requested batch in one
        read-only query rather than issuing one archive query per incident card.
        """

        requested: set[tuple[str, int]] = set()
        for raw_ref in refs or ():
            match = re.fullmatch(r"(.+):snapshot:([1-9][0-9]*)", str(raw_ref or "").strip())
            if not match:
                continue
            batch_id = match.group(1).strip()
            if not batch_id or len(batch_id) > 200:
                continue
            requested.add((batch_id, int(match.group(2))))
        if not requested:
            return {}

        batch_ids = sorted({batch_id for batch_id, _ in requested})
        try:
            with self.lock:
                with self.pool.transaction(self._context(), readonly=True) as connection:
                    rows = connection.execute(
                        """
                        SELECT
                            id,
                            event_timestamp_ms,
                            payload_json->>'batch_id' AS batch_id,
                            (payload_json->>'snapshot_index')::integer AS snapshot_index
                        FROM archive.detections
                        WHERE tenant_id = %s
                          AND source = 'vlm_summary'
                          AND payload_json->>'batch_id' = ANY(%s)
                          AND COALESCE(payload_json->>'snapshot_index', '') ~ '^[1-9][0-9]*$'
                          AND (thumbnail_b64 IS NOT NULL OR image_path IS NOT NULL)
                        ORDER BY recorded_at_ms DESC, id DESC
                        """,
                        (self.tenant_id, batch_ids),
                    ).fetchall()
        except Exception as exc:
            if _is_missing_archive_relation(exc):
                raise _archive_not_ready(exc) from exc
            raise

        resolved: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            batch_id = str(row[2] or "").strip()
            snapshot_index = int(row[3] or 0)
            if (batch_id, snapshot_index) not in requested:
                continue
            ref = f"{batch_id}:snapshot:{snapshot_index}"
            resolved.setdefault(
                ref,
                {
                    "detection_id": int(row[0]),
                    "timestamp_ms": int(row[1] or 0) or None,
                },
            )
        return resolved

    def resolve_vlm_snapshot_cover_refs(
        self,
        hints: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Resolve card-cover refs without a tenant-wide JSON heap scan.

        PostgreSQL row-level security cannot safely push the non-leakproof JSON
        extraction used by ``resolve_vlm_snapshot_refs`` into the expression
        index.  Under a busy archive that turns a small review-board read into a
        multi-second heap scan.  Incident starts are trusted durable timestamps,
        so first select candidate ids through the ordinary channel/time index,
        then inspect only those bounded rows for exact batch/snapshot identity.
        """

        normalized: List[Tuple[str, str, int, int, int]] = []
        seen_refs: set[str] = set()
        for raw_hint in list(hints or ())[:500]:
            if not isinstance(raw_hint, Mapping):
                continue
            ref = str(raw_hint.get("ref") or "").strip()
            match = re.fullmatch(r"(.+):snapshot:([1-9][0-9]*)", ref)
            if not match or ref in seen_refs:
                continue
            try:
                channel_id = int(raw_hint.get("channel_id") or 0)
                timestamp_ms = int(raw_hint.get("timestamp_ms") or 0)
            except (TypeError, ValueError):
                continue
            batch_id = match.group(1).strip()
            snapshot_index = int(match.group(2))
            if (
                channel_id < 1
                or timestamp_ms < 1
                or not batch_id
                or len(batch_id) > 200
            ):
                continue
            seen_refs.add(ref)
            normalized.append(
                (ref, batch_id, snapshot_index, channel_id, timestamp_ms)
            )
        if not normalized:
            return {}

        resolved: Dict[str, Dict[str, Any]] = {}

        def resolve_window(
            connection: Any,
            request_indices: Sequence[int],
            *,
            before_ms: int,
            after_ms: int,
        ) -> None:
            if not request_indices:
                return
            values_sql = ",".join(["(%s,%s,%s,%s)"] * len(request_indices))
            params: List[Any] = []
            for request_index in request_indices:
                _ref, _batch_id, _snapshot_index, channel_id, timestamp_ms = (
                    normalized[int(request_index)]
                )
                params.extend(
                    (
                        int(request_index),
                        channel_id,
                        max(0, timestamp_ms - before_ms),
                        timestamp_ms + after_ms,
                    )
                )
            params.append(self.tenant_id)
            candidate_rows = connection.execute(
                f"""
                WITH requested(hint_id, channel_id, start_ms, end_ms) AS (
                    VALUES {values_sql}
                )
                SELECT requested.hint_id, detections.id
                FROM requested
                JOIN archive.detections AS detections
                  ON detections.tenant_id = %s
                 AND detections.source = 'vlm_summary'
                 AND detections.channel_id = requested.channel_id
                 AND detections.event_timestamp_ms
                     BETWEEN requested.start_ms AND requested.end_ms
                """,
                tuple(params),
            ).fetchall()
            candidate_ids = sorted(
                {
                    int(row[1])
                    for row in candidate_rows
                    if len(row) > 1 and int(row[1] or 0) > 0
                }
            )
            if not candidate_ids:
                return
            detail_rows = connection.execute(
                """
                SELECT
                    id,
                    event_timestamp_ms,
                    payload_json->>'batch_id' AS batch_id,
                    payload_json->>'snapshot_index' AS snapshot_index,
                    (thumbnail_b64 IS NOT NULL OR image_path IS NOT NULL)
                        AS has_media
                FROM archive.detections
                WHERE tenant_id = %s
                  AND id = ANY(%s)
                """,
                (self.tenant_id, candidate_ids),
            ).fetchall()
            details = {
                int(row[0]): row
                for row in detail_rows
                if row and int(row[0] or 0) > 0
            }
            candidates_by_hint: Dict[int, List[Sequence[Any]]] = {}
            for raw_hint_id, raw_detection_id in candidate_rows:
                hint_id = int(raw_hint_id)
                detail = details.get(int(raw_detection_id))
                if detail is not None:
                    candidates_by_hint.setdefault(hint_id, []).append(detail)
            for request_index in request_indices:
                ref, batch_id, snapshot_index, _channel_id, _timestamp_ms = (
                    normalized[int(request_index)]
                )
                matches = [
                    row
                    for row in candidates_by_hint.get(int(request_index), [])
                    if bool(row[4])
                    and str(row[2] or "") == batch_id
                    and str(row[3] or "") == str(snapshot_index)
                ]
                if not matches:
                    continue
                best = max(matches, key=lambda row: (int(row[1] or 0), int(row[0])))
                resolved[ref] = {
                    "detection_id": int(best[0]),
                    "timestamp_ms": int(best[1] or 0) or None,
                }

        try:
            with self.lock:
                with self.pool.transaction(self._context(), readonly=True) as connection:
                    all_indices = list(range(len(normalized)))
                    resolve_window(
                        connection,
                        all_indices,
                        before_ms=10_000,
                        after_ms=90_000,
                    )
                    unresolved_indices = [
                        index
                        for index, item in enumerate(normalized)
                        if item[0] not in resolved
                    ]
                    # A merged/late-arriving incident can place its first durable
                    # evidence slightly outside the tight start window.  Widen
                    # only those misses; never fall back to the tenant-wide JSON
                    # scan that caused the review board timeout.
                    resolve_window(
                        connection,
                        unresolved_indices,
                        before_ms=60_000,
                        after_ms=300_000,
                    )
        except Exception as exc:
            if _is_missing_archive_relation(exc):
                raise _archive_not_ready(exc) from exc
            raise
        return resolved

    @staticmethod
    def _feedback_row_to_dict(row: Sequence[Any]) -> Dict[str, Any]:
        return {
            "id": int(row[0]),
            "detection_id": int(row[1]),
            "channel_id": int(row[2]),
            "alert_timestamp_ms": int(row[3]),
            "actor_id": str(row[4]),
            "reason_code": str(row[5]),
            "reason_label": ALERT_FEEDBACK_REASON_LABELS.get(
                str(row[5]),
                str(row[5]).replace("_", " ").title(),
            ),
            "note": str(row[6] or ""),
            "alert_title": str(row[7]),
            "alert_snapshot": _decode_json_value(row[8]) or {},
            "submitted_at_ms": int(row[9] or 0),
            "updated_at_ms": int(row[10] or 0),
        }

    def upsert_alert_feedback(
        self,
        *,
        detection_id: int,
        reason_code: str,
        note: str = "",
        actor_id: str | uuid.UUID | None = None,
    ) -> Dict[str, Any]:
        detection_id = int(detection_id)
        if detection_id <= 0:
            raise ValueError("detection_id must be positive")
        reason = str(reason_code or "").strip().lower()
        if reason not in ALERT_FEEDBACK_REASON_LABELS:
            raise ValueError(
                "reason_code must be one of: "
                + ", ".join(ALERT_FEEDBACK_REASON_LABELS)
            )
        clean_note = str(note or "").strip()
        if len(clean_note) > 1000:
            raise ValueError("note must be at most 1000 characters")
        clean_actor = actor_id if actor_id is not None else NIL_UUID
        try:
            with self.lock:
                with self.pool.transaction(self._context(clean_actor)) as connection:
                    row = connection.execute(
                        """
                        WITH alert AS (
                            SELECT
                                id,
                                channel_id,
                                event_timestamp_ms,
                                left(
                                    COALESCE(
                                        NULLIF(payload_json #>> '{alert_event,title}', ''),
                                        NULLIF(probe_name, ''),
                                        'VLM alert'
                                    ),
                                    160
                                ) AS alert_title,
                                jsonb_strip_nulls(
                                    jsonb_build_object(
                                        'source', source,
                                        'severity', severity,
                                        'probe_id', probe_id,
                                        'run_id', payload_json->>'run_id',
                                        'batch_start_ms', payload_json->'batch_start_ms',
                                        'batch_end_ms', payload_json->'batch_end_ms',
                                        'anchor_frame_index', payload_json->'anchor_frame_index',
                                        'anchor_frame_timestamp_ms', payload_json->'anchor_frame_timestamp_ms',
                                        'alert_event', payload_json->'alert_event',
                                        'summary', left(COALESCE(payload_json->>'summary', ''), 1000)
                                    )
                                ) AS alert_snapshot
                            FROM archive.detections
                            WHERE tenant_id = %s
                              AND id = %s
                              AND source = 'vlm_alert'
                        )
                        INSERT INTO archive.alert_feedback (
                            tenant_id,
                            detection_id,
                            channel_id,
                            alert_timestamp_ms,
                            actor_id,
                            reason_code,
                            note,
                            alert_title,
                            alert_snapshot
                        )
                        SELECT
                            %s,
                            alert.id,
                            alert.channel_id,
                            alert.event_timestamp_ms,
                            %s,
                            %s,
                            NULLIF(%s, ''),
                            alert.alert_title,
                            alert.alert_snapshot
                        FROM alert
                        ON CONFLICT (tenant_id, detection_id, actor_id)
                        DO UPDATE SET
                            reason_code = EXCLUDED.reason_code,
                            note = EXCLUDED.note,
                            alert_title = EXCLUDED.alert_title,
                            alert_snapshot = EXCLUDED.alert_snapshot,
                            updated_at = clock_timestamp()
                        RETURNING
                            id,
                            detection_id,
                            channel_id,
                            alert_timestamp_ms,
                            actor_id,
                            reason_code,
                            note,
                            alert_title,
                            alert_snapshot,
                            (extract(epoch FROM submitted_at) * 1000)::bigint,
                            (extract(epoch FROM updated_at) * 1000)::bigint
                        """,
                        (
                            self.tenant_id,
                            detection_id,
                            self.tenant_id,
                            str(uuid.UUID(str(clean_actor))),
                            reason,
                            clean_note,
                        ),
                    ).fetchone()
        except Exception as exc:
            if _is_missing_archive_relation(exc):
                raise _feedback_not_ready(exc) from exc
            raise
        if not row:
            raise LookupError("VLM alert detection was not found")
        return self._feedback_row_to_dict(row)

    def get_alert_feedback(
        self,
        *,
        detection_id: int,
        actor_id: str | uuid.UUID | None = None,
    ) -> Optional[Dict[str, Any]]:
        params: List[Any] = [self.tenant_id, int(detection_id)]
        actor_sql = ""
        if actor_id is not None:
            actor_sql = "AND actor_id = %s"
            params.append(str(uuid.UUID(str(actor_id))))
        try:
            with self.lock:
                with self.pool.transaction(self._context(actor_id), readonly=True) as connection:
                    row = connection.execute(
                        f"""
                        SELECT
                            id,
                            detection_id,
                            channel_id,
                            alert_timestamp_ms,
                            actor_id,
                            reason_code,
                            note,
                            alert_title,
                            alert_snapshot,
                            (extract(epoch FROM submitted_at) * 1000)::bigint,
                            (extract(epoch FROM updated_at) * 1000)::bigint
                        FROM archive.alert_feedback
                        WHERE tenant_id = %s
                          AND detection_id = %s
                          {actor_sql}
                        ORDER BY updated_at DESC, id DESC
                        LIMIT 1
                        """,
                        tuple(params),
                    ).fetchone()
        except Exception as exc:
            if _is_missing_archive_relation(exc):
                raise _feedback_not_ready(exc) from exc
            raise
        return self._feedback_row_to_dict(row) if row else None

    @staticmethod
    def _feedback_where(
        *,
        tenant_id: str,
        channel_id: Optional[int] = None,
        channel_ids: Optional[Sequence[int]] = None,
        since_ms: Optional[int] = None,
        until_ms: Optional[int] = None,
        reason_code: Optional[str] = None,
    ) -> Tuple[str, List[Any]]:
        where = ["tenant_id = %s"]
        params: List[Any] = [tenant_id]
        if channel_id is not None:
            where.append("channel_id = %s")
            params.append(int(channel_id))
        elif channel_ids is not None:
            clean_channels = sorted({int(item) for item in channel_ids if int(item) > 0})
            if clean_channels:
                where.append(
                    f"channel_id IN ({','.join('%s' for _ in clean_channels)})"
                )
                params.extend(clean_channels)
            else:
                where.append("1 = 0")
        if since_ms is not None:
            where.append("alert_timestamp_ms >= %s")
            params.append(int(since_ms))
        if until_ms is not None:
            where.append("alert_timestamp_ms <= %s")
            params.append(int(until_ms))
        if reason_code:
            normalized_reason = str(reason_code).strip().lower()
            if normalized_reason not in ALERT_FEEDBACK_REASON_LABELS:
                raise ValueError(
                    "reason_code must be one of: "
                    + ", ".join(ALERT_FEEDBACK_REASON_LABELS)
                )
            where.append("reason_code = %s")
            params.append(normalized_reason)
        return "WHERE " + " AND ".join(where), params

    def list_alert_feedback(
        self,
        *,
        channel_id: Optional[int] = None,
        channel_ids: Optional[Sequence[int]] = None,
        since_ms: Optional[int] = None,
        until_ms: Optional[int] = None,
        reason_code: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        limit = max(1, min(500, int(limit or 100)))
        offset = max(0, int(offset or 0))
        where_sql, params = self._feedback_where(
            tenant_id=self.tenant_id,
            channel_id=channel_id,
            channel_ids=channel_ids,
            since_ms=since_ms,
            until_ms=until_ms,
            reason_code=reason_code,
        )
        try:
            with self.lock:
                with self.pool.transaction(self._context(), readonly=True) as connection:
                    total_row = connection.execute(
                        f"SELECT COUNT(*) FROM archive.alert_feedback {where_sql}",
                        tuple(params),
                    ).fetchone()
                    rows = connection.execute(
                        f"""
                        SELECT
                            id,
                            detection_id,
                            channel_id,
                            alert_timestamp_ms,
                            actor_id,
                            reason_code,
                            note,
                            alert_title,
                            alert_snapshot,
                            (extract(epoch FROM submitted_at) * 1000)::bigint,
                            (extract(epoch FROM updated_at) * 1000)::bigint
                        FROM archive.alert_feedback
                        {where_sql}
                        ORDER BY alert_timestamp_ms DESC, id DESC
                        LIMIT %s OFFSET %s
                        """,
                        tuple(params + [limit, offset]),
                    ).fetchall()
        except Exception as exc:
            if _is_missing_archive_relation(exc):
                raise _feedback_not_ready(exc) from exc
            raise
        total = int(total_row[0] or 0) if total_row else 0
        return [self._feedback_row_to_dict(row) for row in rows], total

    def generate_false_positive_report(
        self,
        *,
        channel_id: Optional[int] = None,
        channel_ids: Optional[Sequence[int]] = None,
        since_ms: Optional[int] = None,
        until_ms: Optional[int] = None,
        reason_code: Optional[str] = None,
        item_limit: int = 100,
    ) -> Dict[str, Any]:
        where_sql, params = self._feedback_where(
            tenant_id=self.tenant_id,
            channel_id=channel_id,
            channel_ids=channel_ids,
            since_ms=since_ms,
            until_ms=until_ms,
            reason_code=reason_code,
        )
        try:
            with self.lock:
                with self.pool.transaction(self._context(), readonly=True) as connection:
                    summary_row = connection.execute(
                        f"""
                        SELECT
                            COUNT(*),
                            COUNT(DISTINCT detection_id),
                            COUNT(DISTINCT actor_id),
                            COUNT(DISTINCT channel_id),
                            MIN(alert_timestamp_ms),
                            MAX(alert_timestamp_ms),
                            MIN((extract(epoch FROM submitted_at) * 1000)::bigint),
                            MAX((extract(epoch FROM updated_at) * 1000)::bigint)
                        FROM archive.alert_feedback
                        {where_sql}
                        """,
                        tuple(params),
                    ).fetchone()
                    reason_rows = connection.execute(
                        f"""
                        SELECT reason_code, COUNT(*)
                        FROM archive.alert_feedback
                        {where_sql}
                        GROUP BY reason_code
                        ORDER BY COUNT(*) DESC, reason_code ASC
                        """,
                        tuple(params),
                    ).fetchall()
                    channel_rows = connection.execute(
                        f"""
                        SELECT channel_id, COUNT(*)
                        FROM archive.alert_feedback
                        {where_sql}
                        GROUP BY channel_id
                        ORDER BY COUNT(*) DESC, channel_id ASC
                        """,
                        tuple(params),
                    ).fetchall()
        except Exception as exc:
            if _is_missing_archive_relation(exc):
                raise _feedback_not_ready(exc) from exc
            raise

        feedback, total = self.list_alert_feedback(
            channel_id=channel_id,
            channel_ids=channel_ids,
            since_ms=since_ms,
            until_ms=until_ms,
            reason_code=reason_code,
            limit=item_limit,
            offset=0,
        )
        summary = {
            "annotation_count": int(summary_row[0] or 0) if summary_row else 0,
            "distinct_alert_count": int(summary_row[1] or 0) if summary_row else 0,
            "reviewer_count": int(summary_row[2] or 0) if summary_row else 0,
            "channel_count": int(summary_row[3] or 0) if summary_row else 0,
        }
        reason_counts = [
            {
                "reason_code": str(row[0]),
                "reason_label": ALERT_FEEDBACK_REASON_LABELS.get(
                    str(row[0]),
                    str(row[0]).replace("_", " ").title(),
                ),
                "count": int(row[1] or 0),
            }
            for row in reason_rows
        ]
        channel_counts = [
            {"channel_id": int(row[0]), "count": int(row[1] or 0)}
            for row in channel_rows
        ]
        lines = [
            "# False-positive operator feedback report",
            "",
            (
                f"- Period: {since_ms if since_ms is not None else 'unbounded'}"
                f" to {until_ms if until_ms is not None else 'now'} (Unix ms)"
            ),
            f"- Operator annotations: {summary['annotation_count']}",
            f"- Distinct alerts: {summary['distinct_alert_count']}",
            f"- Channels: {summary['channel_count']}",
            f"- Reviewers: {summary['reviewer_count']}",
            "",
            "## Reasons",
            "",
        ]
        if reason_counts:
            lines.extend(
                f"- {row['reason_label']}: {row['count']}"
                for row in reason_counts
            )
        else:
            lines.append("- No operator false-positive annotations in this period.")
        lines.extend(
            [
                "",
                "> These counts are operator annotations, not independent ground truth.",
            ]
        )
        return {
            "report_type": "false_positives",
            "period": {
                "since_ms": since_ms,
                "until_ms": until_ms,
                "first_alert_timestamp_ms": int(summary_row[4] or 0) if summary_row else 0,
                "last_alert_timestamp_ms": int(summary_row[5] or 0) if summary_row else 0,
                "first_submitted_at_ms": int(summary_row[6] or 0) if summary_row else 0,
                "last_updated_at_ms": int(summary_row[7] or 0) if summary_row else 0,
            },
            "coverage": {
                "status": "covered" if total else "no_annotations",
                "annotation_count": total,
                "returned_count": len(feedback),
                "truncated": len(feedback) < total,
                "scope": "operator_false_positive_annotations",
                "ground_truth_status": "operator_annotation_only",
            },
            "summary": summary,
            "reason_counts": reason_counts,
            "channel_counts": channel_counts,
            "feedback": feedback,
            "report": "\n".join(lines),
        }

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
        channel_ids: Optional[Sequence[int]] = None,
    ) -> List[Dict[str, Any]]:
        limit = max(1, min(500, int(limit or 100)))
        where_sql, params = self._build_where(
            channel_id=channel_id,
            channel_ids=channel_ids,
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
        channel_ids: Optional[Sequence[int]] = None,
        since_ms: Optional[int] = None,
        until_ms: Optional[int] = None,
        limit: int = 2000,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        limit = max(1, min(5000, int(limit or 2000)))
        where_sql, params = self._build_where(
            probe_id=probe_id,
            channel_id=channel_id,
            channel_ids=channel_ids,
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
    def _select_columns(
        *,
        include_thumbnail: bool = True,
        include_vectors: bool = False,
        include_payload: bool = True,
    ) -> str:
        thumbnail_expr = "thumbnail_b64" if include_thumbnail else "NULL::text AS thumbnail_b64"
        # Candidate enumeration only needs the embedding-space identity from
        # payload_json. VLM summaries can contain several KB of repeated text,
        # so selecting the whole payload for tens of thousands of candidates
        # can exceed the request statement timeout before FAISS even runs.
        payload_expr = (
            "payload_json"
            if include_payload
            else (
                "CASE WHEN payload_json ? 'embedding_space' "
                "THEN jsonb_build_object('embedding_space', payload_json->'embedding_space') "
                "ELSE NULL::jsonb END AS payload_json"
            )
        )
        # Preserve the cheap has_clip/has_dino flags consumed by _row_to_dict
        # without transferring the full vectors unless the caller explicitly
        # needs them. A SigLIP2 vector is several KB per row.
        clip_expr = (
            "clip_vec"
            if include_vectors
            else (
                "CASE WHEN clip_vec IS NULL THEN NULL::bytea "
                "ELSE decode('00', 'hex') END AS clip_vec"
            )
        )
        dino_expr = (
            "dino_vec"
            if include_vectors
            else (
                "CASE WHEN dino_vec IS NULL THEN NULL::bytea "
                "ELSE decode('00', 'hex') END AS dino_vec"
            )
        )
        return (
            "id, event_timestamp_ms, recorded_at_ms, probe_id, probe_name, "
            "channel_id, severity, bookmark_enabled, bookmark_sent, pos_score, "
            f"neg_score, margin, {thumbnail_expr}, source, {payload_expr}, shard_key, "
            f"image_path, {clip_expr}, {dino_expr}"
        )

    def _build_where(
        self,
        probe_id: Optional[str] = None,
        channel_id: Optional[int] = None,
        channel_ids: Optional[Sequence[int]] = None,
        source: Optional[str] = None,
        since_ms: Optional[int] = None,
        until_ms: Optional[int] = None,
        only_with_clip: bool = False,
        only_with_dino: bool = False,
        ids: Optional[Sequence[int]] = None,
        batch_id: Optional[str] = None,
        parent_alert_id: Optional[str] = None,
        embedding_space: Optional[Mapping[str, Any]] = None,
        allow_legacy_embedding_space: bool = False,
    ) -> Tuple[str, List[Any]]:
        where: List[str] = ["tenant_id = %s"]
        params: List[Any] = [self.tenant_id]
        if probe_id:
            where.append("probe_id = %s")
            params.append(str(probe_id))
        normalized_channel_ids = {
            int(item)
            for item in (channel_ids or [])
            if item is not None and int(item) > 0
        }
        if channel_id is not None and int(channel_id) > 0:
            normalized_channel_ids.add(int(channel_id))
        if normalized_channel_ids:
            ordered_channel_ids = sorted(normalized_channel_ids)
            if len(ordered_channel_ids) == 1:
                where.append("channel_id = %s")
                params.append(ordered_channel_ids[0])
            else:
                where.append(
                    f"channel_id IN ({','.join('%s' for _ in ordered_channel_ids)})"
                )
                params.extend(ordered_channel_ids)
        elif channel_ids is not None:
            where.append("1 = 0")
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
        normalized_batch_id = str(batch_id or "").strip()
        if normalized_batch_id:
            # Keep this explicit existence predicate aligned with the partial
            # ix_archive_detections_vlm_batch index. PostgreSQL does not infer
            # `payload_json ? 'batch_id'` from an equality on `->>`, so without
            # it even a six-frame immutable batch falls back to two full-table
            # scans (COUNT + page read) across thumbnail-heavy archive rows.
            where.append("payload_json ? 'batch_id'")
            where.append("payload_json->>'batch_id' = %s")
            params.append(normalized_batch_id)
        normalized_parent_alert_id = str(parent_alert_id or "").strip()
        if normalized_parent_alert_id:
            where.append("payload_json->>'parent_alert_id' = %s")
            params.append(normalized_parent_alert_id)
        if only_with_clip:
            where.append("clip_vec IS NOT NULL")
        if only_with_dino:
            where.append("dino_vec IS NOT NULL")
        normalized_space = normalize_embedding_space(embedding_space)
        if normalized_space:
            fingerprint = embedding_space_fingerprint(normalized_space)
            fingerprint_sql = (
                "payload_json#>>'{embedding_space,fingerprint}' = %s"
            )
            if (
                allow_legacy_embedding_space
                and not embedding_space_requires_identity(normalized_space)
            ):
                legacy_clauses = [
                    "payload_json#>>'{embedding_space,fingerprint}' IS NULL",
                    "(payload_json#>>'{embedding_space,backend}' IS NULL "
                    "OR payload_json#>>'{embedding_space,backend}' = %s)",
                    "(payload_json#>>'{embedding_space,model}' IS NULL "
                    "OR payload_json#>>'{embedding_space,model}' = %s)",
                ]
                legacy_params: List[Any] = [
                    str(normalized_space.get("backend") or "openai_clip"),
                    str(normalized_space.get("model") or ""),
                ]
                if normalized_space.get("dimension") is not None:
                    legacy_clauses.append(
                        "(payload_json#>>'{embedding_space,dimension}' IS NULL "
                        "OR payload_json#>>'{embedding_space,dimension}' = %s)"
                    )
                    legacy_params.append(str(normalized_space["dimension"]))
                where.append(
                    f"({fingerprint_sql} OR ({' AND '.join(legacy_clauses)}))"
                )
                params.extend([fingerprint, *legacy_params])
            else:
                where.append(fingerprint_sql)
                params.append(fingerprint)
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

    def delete_probes(self, probe_ids: Sequence[str]) -> int:
        normalized = sorted(
            {
                str(probe_id or "").strip()
                for probe_id in probe_ids
                if str(probe_id or "").strip()
            }
        )
        if not normalized:
            return 0
        with self.lock:
            with self.pool.transaction(self._context()) as connection:
                cursor = connection.execute(
                    """
                    DELETE FROM archive.probes
                    WHERE tenant_id = %s
                      AND probe_id = ANY(%s)
                    """,
                    (self.tenant_id, normalized),
                )
        return int(cursor.rowcount or 0)


class PostgresRuntimeStateStore(_TenantRepository):
    """Small JSONB document store for Luxriot summary and rollup runtime state."""

    _SUMMARY_STATE_KEY = "luxriot_summary_state"
    _SUMMARY_META_KEY = "luxriot_summary_state:meta"
    _SUMMARY_HISTORY_PREFIX = "luxriot_summary_state:history:"
    _SUMMARY_HISTORY_ITEM_PREFIX = "luxriot_summary_state:history_item:"
    _SUMMARY_RUNS_PREFIX = "luxriot_summary_state:runs:"
    _ROLLUP_PREFIX = "luxriot_rollup:"

    def __init__(self, pool: PsycopgPool, tenant_id: str | uuid.UUID) -> None:
        super().__init__(pool, tenant_id)
        self._summary_history_items_last_pruned_at = 0.0

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

    @classmethod
    def _rollup_state_key(cls, payload: Mapping[str, Any]) -> str:
        channel_id = int(payload.get("channel_id") or 0)
        level = str(payload.get("level") or "").strip().lower()
        window_start = float(payload.get("window_start"))
        if channel_id < 1 or level not in {"l1", "l2", "l3"}:
            raise ValueError("rollup requires a positive channel_id and level L1, L2, or L3")
        return _state_key(
            f"{cls._ROLLUP_PREFIX}{channel_id}:{level}:{int(window_start)}"
        )

    def save_rollup(self, payload: Mapping[str, Any]) -> None:
        """Persist one semantic rollup as an independently queryable state row."""

        payload_dict = dict(payload)
        key = self._rollup_state_key(payload_dict)
        self._save_state_exact(key, payload_dict)

    def save_rollups(self, payloads: Sequence[Mapping[str, Any]]) -> int:
        """Promote legacy hot-cache rows without replacing newer durable rows."""

        entries: List[Tuple[str, Dict[str, Any], str]] = []
        for payload in payloads:
            payload_dict = dict(payload)
            key = self._rollup_state_key(payload_dict)
            entries.append((key, payload_dict, _payload_digest(payload_dict)))
        if not entries:
            return 0
        written = 0
        with self.lock:
            with self.pool.transaction(self._context()) as connection:
                for key, payload_dict, digest in entries:
                    cache_key = f"exact:{key}"
                    if self._last_state_hashes.get(cache_key) == digest:
                        continue
                    cursor = connection.execute(
                        """
                        INSERT INTO archive.runtime_state (
                            tenant_id,
                            state_key,
                            payload_json
                        )
                        VALUES (%s, %s, %s)
                        ON CONFLICT (tenant_id, state_key) DO NOTHING
                        """,
                        (self.tenant_id, key, _jsonb(payload_dict)),
                    )
                    if int(cursor.rowcount or 0) > 0:
                        self._last_state_hashes[cache_key] = digest
                        written += 1
        return written

    def load_rollup(self, rollup_id: str) -> Optional[Dict[str, Any]]:
        """Load a canonical L1-L3 rollup without scanning the shared hot cache."""

        match = re.fullmatch(
            r"(?P<level>l[123])-ch(?P<channel>\d+)-w\d+-(?P<start>\d+)",
            str(rollup_id or "").strip().lower(),
        )
        if match is None:
            return None
        key = _state_key(
            f"{self._ROLLUP_PREFIX}{int(match.group('channel'))}:"
            f"{match.group('level')}:{int(match.group('start'))}"
        )
        payload = self.load_state(key)
        if not isinstance(payload, dict):
            return None
        if str(payload.get("rollup_id") or "").strip().lower() != str(rollup_id or "").strip().lower():
            return None
        return payload

    def list_rollups(
        self,
        *,
        channel_id: int,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
        levels: Optional[Sequence[str]] = None,
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        """Read persisted rollups overlapping a requested time window."""

        normalized_channel = int(channel_id)
        if normalized_channel < 1:
            return []
        normalized_levels = {
            str(level or "").strip().upper()
            for level in (levels or ("L1", "L2", "L3"))
            if str(level or "").strip().upper() in {"L1", "L2", "L3"}
        }
        if not normalized_levels:
            return []
        clauses = ["tenant_id = %s", "state_key LIKE %s"]
        params: List[Any] = [
            self.tenant_id,
            f"{self._ROLLUP_PREFIX}{normalized_channel}:%",
        ]
        level_placeholders = ", ".join(["%s"] * len(normalized_levels))
        clauses.append(f"upper(payload_json->>'level') IN ({level_placeholders})")
        params.extend(sorted(normalized_levels))
        if start_ts is not None:
            clauses.append("(payload_json->>'window_end')::double precision >= %s")
            params.append(float(start_ts))
        if end_ts is not None:
            clauses.append("(payload_json->>'window_start')::double precision <= %s")
            params.append(float(end_ts))
        bounded_limit = max(1, min(50000, int(limit)))
        params.append(bounded_limit)
        query = f"""
            SELECT payload_json
            FROM archive.runtime_state
            WHERE {' AND '.join(clauses)}
            ORDER BY (payload_json->>'window_start')::double precision ASC
            LIMIT %s
        """
        # PostgreSQL provides a consistent MVCC snapshot for this range read.
        # Do not serialize it behind the repository's writer lock: background
        # rollup/cache persistence can hold that lock for many seconds while
        # the agent is only trying to inspect already-committed summaries.
        with self.pool.transaction(self._context(), readonly=True) as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            payload = _decode_json_value(row[0])
            if isinstance(payload, dict):
                out.append(copy.deepcopy(payload))
        return out

    def prune_rollups(self, cutoff_ts: float) -> int:
        """Remove semantic rollups whose covered window ended before retention."""

        with self.lock:
            with self.pool.transaction(self._context()) as connection:
                cursor = connection.execute(
                    """
                    DELETE FROM archive.runtime_state
                    WHERE tenant_id = %s
                      AND state_key LIKE %s
                      AND (payload_json->>'window_end')::double precision < %s
                    """,
                    (self.tenant_id, f"{self._ROLLUP_PREFIX}%", float(cutoff_ts)),
                )
        return int(cursor.rowcount or 0)

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
        history_storage = str(payload.get("summary_history_storage") or "").strip()
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
                normalized_logs = (
                    logs
                    if isinstance(logs, Sequence)
                    and not isinstance(logs, (str, bytes, bytearray))
                    else []
                )
                if history_storage == "upsert_items":
                    for log in normalized_logs:
                        if not isinstance(log, Mapping):
                            continue
                        identity = {
                            "channel_id": str(channel_id),
                            "batch_id": str(log.get("batch_id") or "").strip(),
                            "run_id": str(log.get("run_id") or "").strip(),
                            "created_at": log.get("created_at"),
                            "frame_count": log.get("frame_count"),
                            "summary": str(log.get("summary") or "").strip()[:160],
                        }
                        item_id = hashlib.sha256(
                            json.dumps(
                                identity,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                                default=str,
                            ).encode("utf-8")
                        ).hexdigest()[:32]
                        entries[
                            f"{self._SUMMARY_HISTORY_ITEM_PREFIX}{channel_id}:{item_id}"
                        ] = {
                            "channel_id": str(channel_id),
                            "log": _plain_value(log),
                        }
                else:
                    entries[f"{self._SUMMARY_HISTORY_PREFIX}{channel_id}"] = {
                        "logs": _plain_value(normalized_logs)
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
                retention_cutoff = payload.get("summary_history_retention_cutoff")
                prune_now = time.monotonic()
                if (
                    history_storage == "upsert_items"
                    and isinstance(retention_cutoff, (int, float))
                    and (
                        self._summary_history_items_last_pruned_at <= 0
                        or prune_now - self._summary_history_items_last_pruned_at >= 3600.0
                    )
                ):
                    connection.execute(
                        """
                        DELETE FROM archive.runtime_state
                        WHERE tenant_id = %s
                          AND state_key LIKE %s
                          AND (payload_json->'log'->>'created_at')::double precision < %s
                        """,
                        (
                            self.tenant_id,
                            f"{self._SUMMARY_HISTORY_ITEM_PREFIX}%",
                            float(retention_cutoff),
                        ),
                    )
                    self._summary_history_items_last_pruned_at = prune_now

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
        summary_history_items: Dict[str, List[Any]] = {}
        summary_runs: Dict[str, Any] = {}
        for key, raw_payload in rows:
            state_key = str(key or "")
            payload = _decode_json_value(raw_payload)
            if not isinstance(payload, dict):
                continue
            if state_key == self._SUMMARY_META_KEY:
                meta = payload
            elif state_key.startswith(self._SUMMARY_HISTORY_ITEM_PREFIX):
                channel_id = str(payload.get("channel_id") or "").strip()
                log = payload.get("log")
                if channel_id and isinstance(log, dict):
                    summary_history_items.setdefault(channel_id, []).append(log)
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
        for channel_id, logs in summary_history_items.items():
            summary_history.setdefault(channel_id, []).extend(logs)
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
