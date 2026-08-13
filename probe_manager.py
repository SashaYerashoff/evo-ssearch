import base64
import os
import threading
import time
from io import BytesIO
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
from PIL import Image

from config import config
from embedding_space import embedding_space_fingerprint
from semantic_presence import SemanticPresenceTracker


_SEMANTIC_PATCH_METADATA_KEY = "_semantic_patch_presence_v1"


class _FaissTypingStub:
    IndexFlatIP = Any


faiss: Any = _FaissTypingStub()
_faiss_module: Optional[Any] = None


def _get_faiss() -> Any:
    global faiss, _faiss_module
    if str(os.getenv("EVOSSEARCH_LOCAL_VISION_ENABLED", "true") or "true").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        raise RuntimeError(
            "Local vision stack is disabled. Set EVOSSEARCH_LOCAL_VISION_ENABLED=true "
            "on a host that supports the installed CLIP/FAISS wheels."
        )
    if _faiss_module is None:
        import importlib

        _faiss_module = importlib.import_module("faiss")
        faiss = _faiss_module
    return _faiss_module


def _faiss_add(index: faiss.IndexFlatIP, vectors: np.ndarray) -> None:
    """FAISS Python API monkey-patches add(), but stubs may expose C++ signature."""
    cast(Any, index).add(vectors)


def _to_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    try:
        return int(cast(Any, value))
    except Exception:
        return None


class ProbeBuffer:
    def __init__(self, max_frames: int, thumb_edge: int) -> None:
        self.max_frames = max_frames
        self.thumb_edge = thumb_edge
        self.embeddings: List[np.ndarray] = []
        self.meta: List[Dict[str, Any]] = []
        self.index: Optional[faiss.IndexFlatIP] = None
        self.next_uid = 1
        self.roi_cache: Dict[str, Dict[int, np.ndarray]] = {}
        self.roi_cache_order: List[str] = []
        self.roi_cache_max_entries = 16

    def _rebuild_index(self) -> None:
        if not self.embeddings:
            self.index = None
            return
        mat = np.stack(self.embeddings, axis=0).astype(np.float32)
        self.index = _get_faiss().IndexFlatIP(mat.shape[1])
        _faiss_add(self.index, mat)

    def clear(self) -> None:
        self.embeddings = []
        self.meta = []
        self.index = None
        self.roi_cache = {}
        self.roi_cache_order = []

    @staticmethod
    def _normalize_vec(vec: np.ndarray) -> np.ndarray:
        arr = vec.flatten().astype(np.float32) if vec.ndim != 1 else vec.astype(np.float32)
        return arr / max(float(np.linalg.norm(arr)), 1e-8)

    @staticmethod
    def _roi_key(roi_norm: Tuple[float, float, float, float]) -> str:
        x, y, w, h = roi_norm
        return f"{x:.4f}:{y:.4f}:{w:.4f}:{h:.4f}"

    def _remember_roi_key(self, key: str) -> None:
        if key in self.roi_cache_order:
            self.roi_cache_order = [item for item in self.roi_cache_order if item != key]
        self.roi_cache_order.append(key)
        while len(self.roi_cache_order) > self.roi_cache_max_entries:
            evicted = self.roi_cache_order.pop(0)
            self.roi_cache.pop(evicted, None)

    def _prune_roi_cache_uids(self, removed_uids: Sequence[int]) -> None:
        if not removed_uids or not self.roi_cache:
            return
        stale = set(int(uid) for uid in removed_uids)
        if not stale:
            return
        empty_keys: List[str] = []
        for key, cache_map in self.roi_cache.items():
            for uid in stale:
                cache_map.pop(uid, None)
            if not cache_map:
                empty_keys.append(key)
        if empty_keys:
            for key in empty_keys:
                self.roi_cache.pop(key, None)
            self.roi_cache_order = [key for key in self.roi_cache_order if key not in set(empty_keys)]

    def add(
        self,
        embedding: np.ndarray,
        timestamp_ms: int,
        channel_id: int,
        thumb: str,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> int:
        emb = self._normalize_vec(embedding)
        if self.embeddings and emb.shape != self.embeddings[0].shape:
            self.clear()
        frame_uid = int(self.next_uid)
        self.next_uid += 1
        self.embeddings.append(emb)
        meta_row: Dict[str, Any] = {
            "uid": frame_uid,
            "timestamp_ms": timestamp_ms,
            "channel_id": channel_id,
            "thumb": thumb,
        }
        if isinstance(provenance, Mapping):
            selection_provenance: Dict[str, Any] = {}
            for key in (
                "version",
                "policy",
                "bucket_start_ms",
                "source_frame_indices",
                "source_timestamps_ms",
                "source_frame_hashes",
                "selected_source_frame_index",
                "selected_timestamp_ms",
                "selected_frame_hash",
                "selection_source",
                "selection_score",
                "score_source",
                "apex_available",
                "fallback_reason",
            ):
                if key in provenance:
                    selection_provenance[key] = provenance.get(key)
            if selection_provenance:
                meta_row["selection_provenance"] = selection_provenance
        self.meta.append(meta_row)
        if len(self.embeddings) > self.max_frames:
            excess = len(self.embeddings) - self.max_frames
            if excess > 0:
                removed = self.meta[:excess]
                self.embeddings = self.embeddings[excess:]
                self.meta = self.meta[excess:]
                removed_uids = [uid for uid in (_to_optional_int(item.get("uid")) for item in removed) if uid is not None]
                self._prune_roi_cache_uids(removed_uids)
        # Query and P/N/M scoring below operate on a bounded dense matrix and do
        # not consume ``self.index``. Rebuilding a complete FAISS index for every
        # one-Hz append was therefore pure O(n) work performed under the global
        # ProbeManager lock, blocking every channel and the L0 scorer. Keep the
        # legacy attribute invalidated for callers that inspect it; a future
        # large-buffer ANN path can rebuild lazily when it actually uses FAISS.
        self.index = None
        return frame_uid

    def read_snapshot(self, *, include_roi_cache: bool = False) -> "ProbeBuffer":
        """Return a stable, zero-copy vector snapshot for lock-free scoring."""

        snapshot = ProbeBuffer(self.max_frames, self.thumb_edge)
        snapshot.embeddings = list(self.embeddings)
        snapshot.meta = [dict(row) for row in self.meta]
        snapshot.next_uid = int(self.next_uid)
        if include_roi_cache:
            snapshot.roi_cache = {
                key: dict(cache)
                for key, cache in self.roi_cache.items()
            }
            snapshot.roi_cache_order = list(self.roi_cache_order)
        return snapshot

    def cache_roi_embedding(
        self,
        roi_norm: Tuple[float, float, float, float],
        frame_uid: int,
        embedding: np.ndarray,
    ) -> None:
        key = self._roi_key(roi_norm)
        cache = self.roi_cache.setdefault(key, {})
        cache[int(frame_uid)] = self._normalize_vec(embedding)
        self._remember_roi_key(key)

    def merge_roi_cache_from(self, snapshot: "ProbeBuffer") -> None:
        valid_uids = {int(row.get("uid") or 0) for row in self.meta}
        for key, snapshot_cache in snapshot.roi_cache.items():
            if not snapshot_cache:
                continue
            cache = self.roi_cache.setdefault(key, {})
            cache.update(
                {
                    int(uid): vector
                    for uid, vector in snapshot_cache.items()
                    if int(uid) in valid_uids
                }
            )
            if cache:
                self._remember_roi_key(key)

    def score(
        self,
        pos_embs: np.ndarray,
        neg_embs: np.ndarray,
        *,
        min_ts_ms: Optional[int] = None,
        max_ts_ms: Optional[int] = None,
        roi_norm: Optional[Tuple[float, float, float, float]] = None,
        embed_image_fn: Optional[Callable[[Image.Image], np.ndarray]] = None,
        roi_padding: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """Return P/N/M for every buffered embedding in a bounded time window."""

        if not self.embeddings or pos_embs.ndim != 2 or pos_embs.shape[0] == 0:
            return []
        selected_idx: List[int] = []
        for idx, row in enumerate(self.meta):
            timestamp_ms = _to_optional_int(row.get("timestamp_ms"))
            if timestamp_ms is None:
                continue
            if min_ts_ms is not None and timestamp_ms < int(min_ts_ms):
                continue
            if max_ts_ms is not None and timestamp_ms > int(max_ts_ms):
                continue
            selected_idx.append(idx)
        if not selected_idx:
            return []
        if roi_norm is None or embed_image_fn is None:
            mat = np.stack(
                [self.embeddings[idx] for idx in selected_idx],
                axis=0,
            ).astype(np.float32)
        else:
            roi_key = self._roi_key(roi_norm)
            cache = self.roi_cache.setdefault(roi_key, {})
            self._remember_roi_key(roi_key)
            vectors: List[np.ndarray] = []
            for idx in selected_idx:
                meta_row = self.meta[idx]
                uid = int(meta_row.get("uid") or 0)
                cached = cache.get(uid)
                if cached is not None:
                    vectors.append(cached)
                    continue
                embedded = self._embed_roi_thumb(
                    str(meta_row.get("thumb") or ""),
                    roi_norm=roi_norm,
                    embed_image_fn=embed_image_fn,
                    roi_padding=roi_padding,
                )
                vec = embedded if embedded is not None else self.embeddings[idx]
                cache[uid] = vec
                vectors.append(vec)
            mat = np.stack(vectors, axis=0).astype(np.float32)
        if int(mat.shape[1]) != int(pos_embs.shape[1]):
            raise ValueError(
                "Probe vector dimension mismatch. Clear the live probe buffer after changing the CLIP model."
            )
        if neg_embs.size > 0 and (
            neg_embs.ndim != 2
            or int(neg_embs.shape[1]) != int(pos_embs.shape[1])
        ):
            raise ValueError(
                "Positive and negative probe vectors use different embedding dimensions."
            )
        pos_max = (mat @ pos_embs.T).max(axis=1)
        neg_max = (
            (mat @ neg_embs.T).max(axis=1)
            if neg_embs.size > 0
            else np.zeros_like(pos_max)
        )
        margin = pos_max - neg_max
        results: List[Dict[str, Any]] = []
        for output_index, meta_index in enumerate(selected_idx):
            meta_row = self.meta[meta_index]
            result: Dict[str, Any] = {
                "frame_uid": int(meta_row.get("uid") or 0),
                "timestamp_ms": int(meta_row.get("timestamp_ms") or 0),
                "channel_id": int(meta_row.get("channel_id") or 0),
                "pos_score": float(pos_max[output_index]),
                "neg_score": float(neg_max[output_index]),
                "margin": float(margin[output_index]),
            }
            if isinstance(meta_row.get("selection_provenance"), Mapping):
                result["selection_provenance"] = dict(
                    meta_row.get("selection_provenance") or {}
                )
            results.append(result)
        return results

    def _embed_roi_thumb(
        self,
        thumb_b64: str,
        roi_norm: Tuple[float, float, float, float],
        embed_image_fn: Callable[[Image.Image], np.ndarray],
        roi_padding: float,
    ) -> Optional[np.ndarray]:
        raw = str(thumb_b64 or "").strip()
        if not raw:
            return None
        try:
            data = base64.b64decode(raw)
            with Image.open(BytesIO(data)) as img:
                rgb = img.convert("RGB")
        except Exception:
            return None
        width, height = rgb.size
        if width < 2 or height < 2:
            return None
        x, y, w, h = roi_norm
        pad = max(0.0, float(roi_padding))
        x0 = max(0.0, x - w * pad)
        y0 = max(0.0, y - h * pad)
        x1 = min(1.0, x + w + w * pad)
        y1 = min(1.0, y + h + h * pad)
        left = int(round(x0 * width))
        top = int(round(y0 * height))
        right = int(round(x1 * width))
        bottom = int(round(y1 * height))
        right = min(width, max(left + 1, right))
        bottom = min(height, max(top + 1, bottom))
        if right - left < 2 or bottom - top < 2:
            return None
        crop = rgb.crop((left, top, right, bottom))
        try:
            return self._normalize_vec(embed_image_fn(crop))
        except Exception:
            return None

    def _mat_for_query(
        self,
        min_ts_ms: Optional[int],
        roi_norm: Optional[Tuple[float, float, float, float]],
        embed_image_fn: Optional[Callable[[Image.Image], np.ndarray]],
        roi_padding: float,
        roi_embedding_budget: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        if not self.embeddings:
            return np.zeros((0, 0), dtype=np.float32), []
        selected_idx: List[int] = []
        for idx, row in enumerate(self.meta):
            ts = row.get("timestamp_ms")
            if min_ts_ms is not None and ts is not None and int(ts) < int(min_ts_ms):
                continue
            selected_idx.append(idx)
        if not selected_idx:
            return np.zeros((0, 0), dtype=np.float32), []
        if roi_norm is None or embed_image_fn is None:
            selected_meta = [self.meta[idx] for idx in selected_idx]
            mat = np.stack([self.embeddings[idx] for idx in selected_idx], axis=0).astype(np.float32)
            return mat, selected_meta

        roi_key = self._roi_key(roi_norm)
        cache = self.roi_cache.setdefault(roi_key, {})
        self._remember_roi_key(roi_key)
        missing_indices = [
            idx
            for idx in selected_idx
            if int(self.meta[idx].get("uid") or 0) not in cache
        ]
        if roi_embedding_budget is None:
            allowed_missing = set(missing_indices)
        else:
            budget = max(0, int(roi_embedding_budget))
            # Fresh evidence is more useful than cold historical backfill. The
            # cache accumulates across daemon passes, so older rows can still be
            # filled gradually without flooding the shared SigLIP batcher.
            allowed_missing = set(missing_indices[-budget:] if budget else [])
        vectors: List[np.ndarray] = []
        selected_meta: List[Dict[str, Any]] = []
        for idx in selected_idx:
            row = self.meta[idx]
            uid = int(row.get("uid") or 0)
            cached = cache.get(uid)
            if cached is not None:
                vectors.append(cached)
                selected_meta.append(row)
                continue
            if idx not in allowed_missing:
                continue
            embedded = self._embed_roi_thumb(
                str(row.get("thumb") or ""),
                roi_norm=roi_norm,
                embed_image_fn=embed_image_fn,
                roi_padding=roi_padding,
            )
            if embedded is None:
                continue
            cache[uid] = embedded
            vectors.append(embedded)
            selected_meta.append(row)
        if not vectors:
            return np.zeros((0, 0), dtype=np.float32), []
        mat = np.stack(vectors, axis=0).astype(np.float32)
        return mat, selected_meta

    def query(
        self,
        pos_embs: np.ndarray,
        neg_embs: np.ndarray,
        pos_floor: float,
        margin_thr: float,
        top_k: int,
        min_ts_ms: Optional[int] = None,
        roi_norm: Optional[Tuple[float, float, float, float]] = None,
        embed_image_fn: Optional[Callable[[Image.Image], np.ndarray]] = None,
        roi_padding: float = 0.05,
        roi_embedding_budget: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not self.embeddings:
            return []
        mat, meta = self._mat_for_query(
            min_ts_ms=min_ts_ms,
            roi_norm=roi_norm,
            embed_image_fn=embed_image_fn,
            roi_padding=roi_padding,
            roi_embedding_budget=roi_embedding_budget,
        )
        if mat.size == 0 or not meta:
            return []
        if pos_embs.ndim != 2 or pos_embs.shape[0] == 0:
            return []
        if int(mat.shape[1]) != int(pos_embs.shape[1]):
            raise ValueError(
                "Probe vector dimension mismatch. Clear the live probe buffer after changing the CLIP model."
            )
        if neg_embs.size > 0 and (neg_embs.ndim != 2 or int(neg_embs.shape[1]) != int(pos_embs.shape[1])):
            raise ValueError(
                "Positive and negative probe vectors use different embedding dimensions."
            )
        pos_scores = mat @ pos_embs.T
        pos_max = pos_scores.max(axis=1)
        if neg_embs.size > 0:
            neg_scores = mat @ neg_embs.T
            neg_max = neg_scores.max(axis=1)
        else:
            neg_max = np.zeros_like(pos_max)
        margin = pos_max - neg_max
        mask = (pos_max >= pos_floor) & (margin >= margin_thr)
        idxs = np.nonzero(mask)[0]
        if idxs.size == 0:
            return []
        selected = [(int(i), float(pos_max[i]), float(neg_max[i]), float(margin[i])) for i in idxs]
        selected.sort(key=lambda x: x[3], reverse=True)
        selected = selected[:top_k]
        results: List[Dict[str, Any]] = []
        for i, p, n, m in selected:
            meta_row = meta[i]
            results.append(
                {
                    "timestamp_ms": meta_row.get("timestamp_ms"),
                    "channel_id": meta_row.get("channel_id"),
                    "thumbnail": meta_row.get("thumb"),
                    "pos_score": p,
                    "neg_score": n,
                    "margin": m,
                    **(
                        {"selection_provenance": dict(meta_row.get("selection_provenance") or {})}
                        if isinstance(meta_row.get("selection_provenance"), Mapping)
                        else {}
                    ),
                }
            )
        return results

    def status(self) -> Dict[str, Any]:
        if not self.meta:
            return {"frames": 0, "time_range_ms": None, "last_timestamp_ms": None, "first_timestamp_ms": None}
        return {
            "frames": len(self.meta),
            "time_range_ms": (self.meta[0]["timestamp_ms"], self.meta[-1]["timestamp_ms"]),
            "last_timestamp_ms": self.meta[-1]["timestamp_ms"],
            "first_timestamp_ms": self.meta[0]["timestamp_ms"],
        }


class ProbeManager:
    def __init__(
        self,
        embed_image_fn: Callable[[Image.Image], np.ndarray],
        embed_text_fn: Callable[[str], np.ndarray],
        jpeg_encoder: Callable[..., str],
        embed_image_with_metadata_fn: Optional[
            Callable[[Image.Image], Tuple[np.ndarray, Mapping[str, Any]]]
        ] = None,
        embedding_space_fn: Optional[Callable[[], Mapping[str, Any]]] = None,
        embed_texts_fn: Optional[Callable[[Sequence[str]], np.ndarray]] = None,
        semantic_presence_enabled: bool = False,
        semantic_presence_classes: Optional[Sequence[str]] = None,
        patch_attention_fn: Optional[
            Callable[[Image.Image, np.ndarray, str], Mapping[str, Any]]
        ] = None,
        patch_attention_enabled: bool = False,
    ) -> None:
        self.buffers: Dict[int, ProbeBuffer] = {}
        self.lock = threading.Lock()
        self.max_frames = config.PROBE_MAX_FRAMES
        self.thumb_edge = config.PROBE_THUMB_MAX_EDGE
        self.embed_image_fn = embed_image_fn
        self.embed_image_with_metadata_fn = embed_image_with_metadata_fn
        self.embedding_space_fn = embedding_space_fn
        self.embed_texts_fn = embed_texts_fn
        self._buffer_embedding_fingerprints: Dict[int, str] = {}
        self._buffer_embedding_spaces: Dict[int, Dict[str, Any]] = {}
        self._embedding_space_fingerprint = ""
        self._embedding_space_fingerprint_lock = threading.Lock()
        self.embed_text_fn = embed_text_fn
        self.jpeg_encoder = jpeg_encoder
        self._text_embedding_cache: Dict[str, np.ndarray] = {}
        self._text_embedding_cache_lock = threading.Lock()
        self._text_embedding_encode_lock = threading.Lock()
        self._text_embedding_cache_limit = 512
        self._text_prewarm_lock = threading.Lock()
        self._text_prewarm_pending: Dict[str, str] = {}
        self._text_prewarm_worker_active = False
        self.semantic_presence_enabled = bool(semantic_presence_enabled)
        self.semantic_presence_tracker = SemanticPresenceTracker(
            semantic_presence_classes or (),
            maximum_classes=int(
                getattr(config, "SEMANTIC_PRESENCE_MAX_CLASSES", 10)
            ),
            warmup_samples=int(
                getattr(config, "SEMANTIC_PRESENCE_WARMUP_SAMPLES", 30)
            ),
        )
        self.semantic_patch_presence_tracker = SemanticPresenceTracker(
            semantic_presence_classes or (),
            maximum_classes=int(
                getattr(config, "SEMANTIC_PRESENCE_MAX_CLASSES", 10)
            ),
            warmup_samples=int(
                getattr(config, "SEMANTIC_PRESENCE_WARMUP_SAMPLES", 30)
            ),
            noise_floor=0.005,
        )
        self.patch_attention_fn = patch_attention_fn
        self.patch_attention_enabled = bool(patch_attention_enabled)
        self._patch_attention_lock = threading.Lock()
        self._patch_attention_inflight: set[int] = set()
        self._patch_attention_last_started: Dict[int, float] = {}
        try:
            patch_attention_interval = float(
                getattr(config, "PROBE_PATCH_ATTENTION_MIN_INTERVAL_SEC", 0.75)
            )
        except (TypeError, ValueError):
            patch_attention_interval = 0.75
        self.patch_attention_min_interval_sec = max(
            0.0,
            min(10.0, patch_attention_interval),
        )
        try:
            roi_query_budget = int(
                getattr(config, "PROBE_ROI_QUERY_EMBED_BUDGET", 2)
            )
        except (TypeError, ValueError):
            roi_query_budget = 2
        self.roi_query_embed_budget = max(0, min(16, roi_query_budget))

    @staticmethod
    def _space_fingerprint(value: Any) -> str:
        if not isinstance(value, Mapping) or not value:
            return ""
        durable = embedding_space_fingerprint(value)
        runtime_generation = str(value.get("runtime_generation") or "").strip()
        # The durable fingerprint intentionally remains model/revision based so
        # archive vectors survive a healthy process restart. In-memory frame
        # and text caches are stricter: a partial model reload must never leave
        # vectors from two loaded instances comparable merely because their
        # metadata is identical.
        return (
            f"{durable}@{runtime_generation}"
            if runtime_generation
            else durable
        )

    def _current_space_fingerprint(self) -> str:
        # A cache hit must not wait for the image encoder. SigLIP image and text
        # inference share one lifecycle lock, and one-Hz archive indexing can
        # otherwise starve L0 probe scoring even though all text vectors are
        # already cached. ``clear_all`` resets this identity whenever the
        # embedding runtime changes; ``add_frame`` refreshes it from encoder
        # metadata captured with the actual image vector.
        with self._embedding_space_fingerprint_lock:
            cached = self._embedding_space_fingerprint
        if cached:
            return cached
        if self.embedding_space_fn is None:
            return ""
        try:
            fingerprint = self._space_fingerprint(self.embedding_space_fn())
        except Exception:
            return ""
        if fingerprint:
            with self._embedding_space_fingerprint_lock:
                self._embedding_space_fingerprint = fingerprint
        return fingerprint

    def _remember_space_fingerprint(self, fingerprint: str) -> None:
        normalized = str(fingerprint or "").strip()
        if not normalized:
            return
        with self._embedding_space_fingerprint_lock:
            self._embedding_space_fingerprint = normalized

    def _buffer_matches_current_space(self, channel_id: int) -> bool:
        current = self._current_space_fingerprint()
        if not current:
            return True
        with self.lock:
            buffered = self._buffer_embedding_fingerprints.get(int(channel_id))
            if not buffered or buffered == current:
                return True
            self.buffers.pop(int(channel_id), None)
            self._buffer_embedding_fingerprints.pop(int(channel_id), None)
            self._buffer_embedding_spaces.pop(int(channel_id), None)
        return False

    def _buffer(self, channel_id: int) -> ProbeBuffer:
        if channel_id not in self.buffers:
            self.buffers[channel_id] = ProbeBuffer(self.max_frames, self.thumb_edge)
        return self.buffers[channel_id]

    def add_frame(
        self,
        channel_id: int,
        pil_image: Image.Image,
        timestamp_ms: Optional[int],
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        ts_ms = timestamp_ms or int(time.time() * 1000)
        embedding_space: Dict[str, Any] = {}
        spatial_presence: Mapping[str, Any] = {}
        if self.embed_image_with_metadata_fn is not None:
            emb, raw_metadata = self.embed_image_with_metadata_fn(pil_image)
            if isinstance(raw_metadata, Mapping):
                embedding_space = dict(raw_metadata)
                raw_spatial_presence = embedding_space.pop(
                    _SEMANTIC_PATCH_METADATA_KEY,
                    None,
                )
                if isinstance(raw_spatial_presence, Mapping):
                    spatial_presence = raw_spatial_presence
        else:
            emb = self.embed_image_fn(pil_image)
        frame_fingerprint = self._space_fingerprint(embedding_space)
        self._remember_space_fingerprint(frame_fingerprint)
        thumb = self.jpeg_encoder(pil_image, max_edge=self.thumb_edge, quality=70)
        reset_presence = False
        with self.lock:
            previous_fingerprint = self._buffer_embedding_fingerprints.get(
                int(channel_id)
            )
            if (
                frame_fingerprint
                and previous_fingerprint
                and frame_fingerprint != previous_fingerprint
            ):
                self.buffers.pop(int(channel_id), None)
                self._buffer_embedding_spaces.pop(int(channel_id), None)
                reset_presence = True
            buf = self._buffer(channel_id)
            if frame_fingerprint:
                self._buffer_embedding_fingerprints[int(channel_id)] = (
                    frame_fingerprint
                )
            if embedding_space:
                self._buffer_embedding_spaces[int(channel_id)] = dict(
                    embedding_space
                )
            frame_uid = buf.add(
                emb,
                ts_ms,
                channel_id,
                thumb,
                provenance=provenance,
            )
        if reset_presence:
            self.semantic_presence_tracker.clear_channel(int(channel_id))
            self.semantic_patch_presence_tracker.clear_channel(int(channel_id))
        presence = self._observe_semantic_presence(
            int(channel_id),
            int(ts_ms),
            np.asarray(emb, dtype=np.float32),
            spatial_presence=spatial_presence,
        )
        result = {
            "channel_id": int(channel_id),
            "frame_uid": int(frame_uid),
            "timestamp_ms": int(ts_ms),
            "embedding_ref": f"probe-buffer:{int(channel_id)}:{int(frame_uid)}",
            "embedding": emb,
            "embedding_space": embedding_space,
            "thumbnail": thumb,
        }
        if presence:
            result["semantic_presence"] = presence
        return result

    def _observe_semantic_presence(
        self,
        channel_id: int,
        timestamp_ms: int,
        embedding: np.ndarray,
        *,
        spatial_presence: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.semantic_presence_enabled:
            return {}
        spatial_status = self._observe_spatial_presence(
            channel_id,
            timestamp_ms,
            spatial_presence,
        )
        plan = self.semantic_presence_tracker.prompt_plan(channel_id)
        phrases = [prompt for _label, prompts in plan for prompt in prompts]
        if not phrases:
            return self._merge_spatial_presence(
                self.semantic_presence_tracker.status(
                    channel_id,
                    now_ms=timestamp_ms,
                ),
                spatial_status,
            )
        try:
            if not self.texts_cached(phrases):
                self.prewarm_texts_async(phrases)
                return self._merge_spatial_presence(
                    self.semantic_presence_tracker.status(
                        channel_id,
                        now_ms=timestamp_ms,
                    ),
                    spatial_status,
                )
            text_matrix = self._embed_texts(phrases)
            image_vector = ProbeBuffer._normalize_vec(
                np.asarray(embedding, dtype=np.float32).flatten()
            )
            if (
                text_matrix.ndim != 2
                or not text_matrix.size
                or int(text_matrix.shape[1]) != int(image_vector.shape[0])
            ):
                raise ValueError("Semantic presence vector dimension mismatch")
            prompt_scores = text_matrix @ image_vector
            offset = 0
            scores: Dict[str, float] = {}
            for label, prompts in plan:
                width = len(prompts)
                scores[label] = float(prompt_scores[offset : offset + width].max())
                offset += width
            return self._merge_spatial_presence(
                self.semantic_presence_tracker.update(
                    channel_id,
                    timestamp_ms,
                    scores,
                ),
                spatial_status,
            )
        except Exception as exc:
            self.semantic_presence_tracker.note_error(channel_id, exc)
            return self._merge_spatial_presence(
                self.semantic_presence_tracker.status(
                    channel_id,
                    now_ms=timestamp_ms,
                ),
                spatial_status,
            )

    def _observe_spatial_presence(
        self,
        channel_id: int,
        timestamp_ms: int,
        payload: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        classes = payload.get("classes") if isinstance(payload, Mapping) else None
        scores: Dict[str, float] = {}
        contrasts: Dict[str, float] = {}
        if isinstance(classes, Mapping):
            for label, raw in classes.items():
                if not isinstance(raw, Mapping):
                    continue
                try:
                    scores[str(label)] = float(raw.get("score"))
                    contrasts[str(label)] = float(raw.get("contrast"))
                except (TypeError, ValueError):
                    continue
        if scores:
            status = self.semantic_patch_presence_tracker.update(
                channel_id,
                timestamp_ms,
                scores,
            )
        else:
            status = self.semantic_patch_presence_tracker.status(
                channel_id,
                now_ms=timestamp_ms,
            )
        if contrasts:
            status = dict(status)
            status["contrasts"] = contrasts
        status["semantics"] = str(
            (payload or {}).get("semantics")
            or "same_forward_top_patch_text_affinity_shadow_v1"
        )
        return status

    @staticmethod
    def _merge_spatial_presence(
        pooled_status: Mapping[str, Any],
        spatial_status: Mapping[str, Any],
    ) -> Dict[str, Any]:
        result = dict(pooled_status)
        spatial_classes = {
            str(item.get("key") or ""): item
            for item in spatial_status.get("classes", [])
            if isinstance(item, Mapping) and item.get("key")
        }
        contrasts = spatial_status.get("contrasts")
        merged_classes: List[Dict[str, Any]] = []
        for raw in result.get("classes", []):
            if not isinstance(raw, Mapping):
                continue
            item = dict(raw)
            spatial = spatial_classes.get(str(item.get("key") or ""))
            if spatial is not None and int(spatial.get("samples") or 0) > 0:
                for key in (
                    "score",
                    "baseline",
                    "deviation",
                    "delta",
                    "z",
                    "state",
                    "warmup",
                    "samples",
                    "timestamp_ms",
                    "history",
                ):
                    if key in spatial:
                        item[f"spatial_{key}"] = spatial[key]
                if isinstance(contrasts, Mapping) and item.get("key") in contrasts:
                    item["spatial_contrast"] = contrasts[item["key"]]
            merged_classes.append(item)
        result["classes"] = merged_classes
        result["spatial_semantics"] = spatial_status.get("semantics")
        result["spatial_state"] = spatial_status.get("state")
        return result

    def set_semantic_presence_classes(
        self,
        channel_id: int,
        labels: Sequence[str],
    ) -> Tuple[str, ...]:
        """Bounded seam for future evidence-backed L1 class proposals."""

        normalized = self.semantic_presence_tracker.set_channel_labels(
            channel_id,
            labels,
        )
        self.semantic_patch_presence_tracker.set_channel_labels(channel_id, labels)
        return normalized

    def patch_attention(
        self,
        channel_id: int,
        timestamp_ms: int,
        class_key: str,
    ) -> Dict[str, Any]:
        """Inspect one exact frame; never persist or feed the resulting map."""

        channel = int(channel_id)
        timestamp = int(timestamp_ms)
        normalized_key = " ".join(str(class_key or "").lower().split())
        if not self.patch_attention_enabled or self.patch_attention_fn is None:
            return {
                "error": "Patch inspection is unavailable.",
                "error_code": "patch_attention_unavailable",
            }
        prompt_plan = dict(self.semantic_presence_tracker.prompt_plan(channel))
        prompts = tuple(prompt_plan.get(normalized_key) or ())
        if not prompts:
            return {
                "error": "Choose a class from this channel's semantic presence bank.",
                "error_code": "patch_attention_class_invalid",
                "allowed_classes": list(prompt_plan),
            }
        thumbnail = self.frame_thumbnail(channel, timestamp)
        frame_vector, _embedding_space = self.frame_embedding(channel, timestamp)
        if not thumbnail or frame_vector is None:
            return {
                "error": "The exact scored frame is no longer buffered.",
                "error_code": "patch_attention_frame_unavailable",
            }
        try:
            raw = base64.b64decode(thumbnail)
            with Image.open(BytesIO(raw)) as opened:
                image = opened.convert("RGB")
        except Exception:
            return {
                "error": "The exact scored frame could not be decoded.",
                "error_code": "patch_attention_frame_invalid",
            }

        now = time.monotonic()
        with self._patch_attention_lock:
            previous = float(self._patch_attention_last_started.get(channel) or 0.0)
            if channel in self._patch_attention_inflight:
                return {
                    "error": "A patch inspection is already running for this channel.",
                    "error_code": "patch_attention_busy",
                    "retry_after_ms": 250,
                }
            remaining = self.patch_attention_min_interval_sec - (now - previous)
            if remaining > 0:
                return {
                    "error": "Patch inspection is rate limited.",
                    "error_code": "patch_attention_rate_limited",
                    "retry_after_ms": int(round(remaining * 1000.0)),
                }
            self._patch_attention_inflight.add(channel)
            self._patch_attention_last_started[channel] = now

        try:
            text_matrix = self._embed_texts(prompts)
            image_vector = ProbeBuffer._normalize_vec(
                np.asarray(frame_vector, dtype=np.float32)
            )
            if (
                text_matrix.ndim != 2
                or not text_matrix.size
                or int(text_matrix.shape[1]) != int(image_vector.shape[0])
            ):
                raise ValueError("Patch inspection vector dimension mismatch")
            prompt_index = int(np.argmax(text_matrix @ image_vector))
            selected_prompt = prompts[prompt_index]
            payload = dict(
                self.patch_attention_fn(
                    image,
                    np.asarray(text_matrix[prompt_index], dtype=np.float32),
                    selected_prompt,
                )
            )
            payload.update(
                {
                    "channel_id": channel,
                    "timestamp_ms": timestamp,
                    "class_key": normalized_key,
                    "label": normalized_key,
                    "prompt": selected_prompt,
                    "frame_url": f"/probes/signal_frame/{channel}/{timestamp}",
                    "ephemeral": True,
                    "shadow": True,
                }
            )
            return payload
        except Exception as exc:
            return {
                "error": f"Patch inspection failed: {type(exc).__name__}",
                "error_code": "patch_attention_failed",
            }
        finally:
            with self._patch_attention_lock:
                self._patch_attention_inflight.discard(channel)

    @staticmethod
    def _prepare_texts(texts: Sequence[str]) -> List[str]:
        return [
            " ".join(str(text or "").split())
            for text in texts
            if " ".join(str(text or "").split())
        ]

    def _text_cache_key(self, text: str, *, space_fingerprint: Optional[str] = None) -> str:
        fingerprint = (
            self._current_space_fingerprint()
            if space_fingerprint is None
            else str(space_fingerprint or "")
        )
        return f"{fingerprint}:{str(text or '').casefold()}"

    def texts_cached(self, texts: Sequence[str]) -> bool:
        """Return immediately; never enter the shared SigLIP encoder lock."""

        prepared = self._prepare_texts(texts)
        if not prepared:
            return True
        space_fingerprint = self._current_space_fingerprint()
        with self._text_embedding_cache_lock:
            return all(
                self._text_cache_key(text, space_fingerprint=space_fingerprint)
                in self._text_embedding_cache
                for text in prepared
            )

    def _embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        prepared = self._prepare_texts(texts)
        if not prepared:
            return np.zeros((0, 0), dtype=np.float32)
        space_fingerprint = self._current_space_fingerprint()

        def cache_key(text: str) -> str:
            return self._text_cache_key(
                text,
                space_fingerprint=space_fingerprint,
            )

        unique_missing: List[str] = []
        with self._text_embedding_cache_lock:
            for text in prepared:
                key = cache_key(text)
                if key not in self._text_embedding_cache and text not in unique_missing:
                    unique_missing.append(text)

        if unique_missing:
            # Single-flight cold text encoding across simultaneous channel
            # summaries. Recheck after taking the lock because another channel
            # may have populated the same probe phrases while we waited.
            with self._text_embedding_encode_lock:
                with self._text_embedding_cache_lock:
                    missing = [
                        text
                        for text in unique_missing
                        if cache_key(text) not in self._text_embedding_cache
                    ]
                if missing:
                    if self.embed_texts_fn is not None:
                        matrix = np.asarray(
                            self.embed_texts_fn(missing),
                            dtype=np.float32,
                        )
                        if matrix.ndim == 1 and len(missing) == 1:
                            matrix = matrix.reshape(1, -1)
                        if matrix.ndim != 2 or int(matrix.shape[0]) != len(missing):
                            raise ValueError(
                                "Batch text embedder returned an invalid matrix shape"
                            )
                        encoded = [matrix[index].flatten() for index in range(len(missing))]
                    else:
                        encoded = [
                            np.asarray(self.embed_text_fn(text), dtype=np.float32).flatten()
                            for text in missing
                        ]
                    with self._text_embedding_cache_lock:
                        for text, vector in zip(missing, encoded):
                            norm = max(float(np.linalg.norm(vector)), 1e-8)
                            self._text_embedding_cache[cache_key(text)] = vector / norm
                        while len(self._text_embedding_cache) > self._text_embedding_cache_limit:
                            oldest = next(iter(self._text_embedding_cache))
                            self._text_embedding_cache.pop(oldest, None)

        with self._text_embedding_cache_lock:
            embs = [self._text_embedding_cache[cache_key(text)] for text in prepared]
        if not embs:
            return np.zeros((0, 0), dtype=np.float32)
        mat = np.stack(embs, axis=0).astype(np.float32)
        mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
        return mat

    def prewarm_texts(self, texts: Sequence[str]) -> int:
        """Encode all unique probe phrases in one model call when possible."""

        unique = list(dict.fromkeys(self._prepare_texts(texts)))
        if unique:
            self._embed_texts(unique)
        return len(unique)

    def prewarm_texts_async(self, texts: Sequence[str]) -> int:
        """Queue cold text vectors without blocking the live L0 batch.

        SigLIP image and text encoding share a model lifecycle lock. Continuous
        one-Hz archive image indexing may therefore make a cold probe phrase
        wait several seconds. A daemon, single-flight worker lets the current
        L0 batch proceed without P/N/M and makes the vectors available to the
        next batch instead of turning that encoder wait into VLM latency.
        """

        unique = list(dict.fromkeys(self._prepare_texts(texts)))
        if not unique:
            return 0
        space_fingerprint = self._current_space_fingerprint()
        with self._text_embedding_cache_lock:
            missing = [
                text
                for text in unique
                if self._text_cache_key(
                    text,
                    space_fingerprint=space_fingerprint,
                )
                not in self._text_embedding_cache
            ]
        if not missing:
            return 0

        scheduled = 0
        start_worker = False
        with self._text_prewarm_lock:
            for text in missing:
                key = self._text_cache_key(
                    text,
                    space_fingerprint=space_fingerprint,
                )
                if key in self._text_prewarm_pending:
                    continue
                self._text_prewarm_pending[key] = text
                scheduled += 1
            if scheduled and not self._text_prewarm_worker_active:
                self._text_prewarm_worker_active = True
                start_worker = True
        if start_worker:
            threading.Thread(
                target=self._run_text_prewarm_worker,
                name="eva-probe-text-prewarm",
                daemon=True,
            ).start()
        return scheduled

    def _run_text_prewarm_worker(self) -> None:
        while True:
            with self._text_prewarm_lock:
                batch = list(self._text_prewarm_pending.items())
                if not batch:
                    self._text_prewarm_worker_active = False
                    return
            try:
                self._embed_texts([text for _key, text in batch])
            except Exception:
                # The next L0 pass may schedule the phrases again. Probe text
                # warmup is advisory and must never take down archive capture.
                pass
            finally:
                with self._text_prewarm_lock:
                    for key, _text in batch:
                        self._text_prewarm_pending.pop(key, None)

    def score_current_frame(
        self,
        channel_id: int,
        timestamp_ms: int,
        positives: Sequence[str],
        negatives: Sequence[str],
        *,
        embedding: np.ndarray,
        thumbnail_b64: str = "",
        roi_norm: Optional[Tuple[float, float, float, float]] = None,
        roi_padding: float = 0.05,
    ) -> Dict[str, Any]:
        """Score one completed semantic frame without re-entering the buffer lock.

        The capture callback already owns the exact full-frame vector. ROI
        probes require one additional cropped embedding, but that work must not
        hold ``ProbeManager.lock``: doing so can block subsequent 1 Hz capture
        insertions behind the shared image microbatcher and turn fresh alarm
        evidence into a stale FIFO.
        """

        pos_texts = [str(item).strip() for item in positives if str(item).strip()]
        neg_texts = [str(item).strip() for item in negatives if str(item).strip()]
        if not pos_texts:
            return {"error": "Provide at least one positive probe."}
        vector = ProbeBuffer._normalize_vec(
            np.asarray(embedding, dtype=np.float32).flatten()
        )
        if roi_norm is not None:
            scratch = ProbeBuffer(max_frames=1, thumb_edge=self.thumb_edge)
            cropped = scratch._embed_roi_thumb(
                str(thumbnail_b64 or ""),
                roi_norm=roi_norm,
                embed_image_fn=self.embed_image_fn,
                roi_padding=roi_padding,
            )
            if cropped is None:
                return {"error": "Current ROI frame could not be embedded."}
            vector = cropped
            self._remember_roi_frame_embedding(
                int(channel_id),
                int(timestamp_ms),
                roi_norm,
                vector,
            )
        pos_embs = self._embed_texts(pos_texts)
        neg_embs = self._embed_texts(neg_texts)
        if pos_embs.size == 0 or int(pos_embs.shape[1]) != int(vector.shape[0]):
            return {"error": "Probe vector dimension mismatch."}
        if neg_embs.size and int(neg_embs.shape[1]) != int(vector.shape[0]):
            return {"error": "Negative probe vector dimension mismatch."}
        pos_score = float((pos_embs @ vector).max())
        neg_score = float((neg_embs @ vector).max()) if neg_embs.size else 0.0
        return {
            "result": {
                "timestamp_ms": int(timestamp_ms),
                "channel_id": int(channel_id),
                "pos_score": pos_score,
                "neg_score": neg_score,
                "margin": pos_score - neg_score,
            },
            "scoring_embedding": vector,
        }

    def _remember_roi_frame_embedding(
        self,
        channel_id: int,
        timestamp_ms: int,
        roi_norm: Tuple[float, float, float, float],
        embedding: np.ndarray,
    ) -> bool:
        """Attach a realtime ROI crop to the matching one-Hz buffer row."""

        with self.lock:
            buffer = self.buffers.get(int(channel_id))
            if buffer is None:
                return False
            for row in reversed(buffer.meta):
                if int(row.get("timestamp_ms") or 0) != int(timestamp_ms):
                    continue
                frame_uid = int(row.get("uid") or 0)
                if frame_uid <= 0:
                    return False
                buffer.cache_roi_embedding(roi_norm, frame_uid, embedding)
                return True
        return False

    def score_frames(
        self,
        channel_id: int,
        positives: Sequence[str],
        negatives: Sequence[str],
        *,
        min_ts_ms: Optional[int] = None,
        max_ts_ms: Optional[int] = None,
        roi_norm: Optional[Tuple[float, float, float, float]] = None,
        roi_padding: float = 0.05,
    ) -> Dict[str, Any]:
        """Score all saved embedding snapshots without applying hit thresholds."""

        if not self._buffer_matches_current_space(channel_id):
            return {
                "error": "Probe buffer belonged to a previous embedding space and was cleared.",
                "frames_indexed": 0,
            }
        pos_texts = [str(item).strip() for item in positives if str(item).strip()]
        neg_texts = [str(item).strip() for item in negatives if str(item).strip()]
        if not pos_texts:
            return {"error": "Provide at least one positive probe."}
        pos_embs = self._embed_texts(pos_texts)
        neg_embs = self._embed_texts(neg_texts)
        with self.lock:
            live_buffer = self.buffers.get(int(channel_id))
            if live_buffer is None:
                return {"results": [], "frames_indexed": 0}
            if roi_norm is None:
                scoring_buffer = live_buffer.read_snapshot()
                status = live_buffer.status()
            else:
                scoring_buffer = live_buffer
                status = live_buffer.status()
        try:
            if roi_norm is None:
                results = scoring_buffer.score(
                    pos_embs,
                    neg_embs,
                    min_ts_ms=min_ts_ms,
                    max_ts_ms=max_ts_ms,
                )
            else:
                # ROI vectors are cached on the live buffer; retain the lock for
                # that uncommon path until ROI embedding receives its own
                # immutable cache generation.
                with self.lock:
                    results = scoring_buffer.score(
                        pos_embs,
                        neg_embs,
                        min_ts_ms=min_ts_ms,
                        max_ts_ms=max_ts_ms,
                        roi_norm=roi_norm,
                        embed_image_fn=self.embed_image_fn,
                        roi_padding=roi_padding,
                    )
                    status = scoring_buffer.status()
        except ValueError as exc:
            with self.lock:
                current = self.buffers.get(int(channel_id))
                if current is live_buffer:
                    current.clear()
            return {"error": str(exc), "frames_indexed": 0}
        return {
            "results": results,
            "status": status,
            "frames_indexed": status.get("frames", 0),
        }

    def _embed_image_base64(self, data: str) -> Optional[np.ndarray]:
        try:
            raw = base64.b64decode(data)
            img = Image.open(BytesIO(raw)).convert("RGB")
            return self.embed_image_fn(img)
        except Exception:
            return None

    def query(
        self,
        channel_id: int,
        positives: Sequence[str],
        negatives: Sequence[str],
        pos_floor: float,
        margin_thr: float,
        top_k: int,
        window_sec: Optional[float] = None,
        image_probe: Optional[Dict[str, Any]] = None,
        roi_norm: Optional[Tuple[float, float, float, float]] = None,
        roi_padding: float = 0.05,
        roi_embedding_budget: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not self._buffer_matches_current_space(channel_id):
            return {
                "error": "Probe buffer belonged to a previous embedding space and was cleared.",
                "frames_indexed": 0,
            }
        pos_texts = [p.strip() for p in positives if str(p).strip()]
        neg_texts = [n.strip() for n in negatives if str(n).strip()]
        image_enabled = False
        image_pos_floor = None
        image_emb = None
        if image_probe and image_probe.get("data") and image_probe.get("enabled", True) is not False:
            image_emb = self._embed_image_base64(image_probe["data"])
            image_enabled = image_emb is not None
            if image_enabled:
                try:
                    image_pos_floor = float(image_probe.get("pos_floor", pos_floor))
                except Exception:
                    image_pos_floor = None
        if not pos_texts and not image_enabled:
            return {"error": "Provide at least one positive probe (text or image)."}
        # Negatives optional; if none, margin uses zero neg scores.
        pos_embs = self._embed_texts(pos_texts)
        if image_emb is not None:
            if image_emb.ndim != 1:
                image_emb = image_emb.flatten()
            if pos_embs.size and int(pos_embs.shape[1]) != int(image_emb.shape[0]):
                return {
                    "error": (
                        "Text and image probe embeddings use different dimensions. "
                        "Use one CLIP model and clear old probe buffers."
                    )
                }
            pos_embs = np.vstack([pos_embs, image_emb[np.newaxis, :]]) if pos_embs.size else image_emb[np.newaxis, :]
        neg_embs = self._embed_texts(neg_texts)
        if neg_embs.size and pos_embs.size and int(neg_embs.shape[1]) != int(pos_embs.shape[1]):
            return {
                "error": (
                    "Positive and negative probe embeddings use different dimensions. "
                    "Use one CLIP model and clear old probe buffers."
                )
            }
        if image_pos_floor is not None:
            pos_floor = max(pos_floor, image_pos_floor)
        min_ts_ms: Optional[int] = None
        if window_sec and window_sec > 0:
            min_ts_ms = int((time.time() - float(window_sec)) * 1000)
        with self.lock:
            live_buffer = self.buffers.get(channel_id)
            if not live_buffer:
                return {"results": [], "frames_indexed": 0}
            query_buffer = live_buffer.read_snapshot(
                include_roi_cache=roi_norm is not None
            )
            status = live_buffer.status()
        try:
            effective_roi_budget = (
                self.roi_query_embed_budget
                if roi_embedding_budget is None
                else max(0, int(roi_embedding_budget))
            )
            results = query_buffer.query(
                pos_embs,
                neg_embs,
                pos_floor,
                margin_thr,
                top_k,
                min_ts_ms=min_ts_ms,
                roi_norm=roi_norm,
                embed_image_fn=self.embed_image_fn,
                roi_padding=roi_padding,
                roi_embedding_budget=(
                    effective_roi_budget if roi_norm is not None else None
                ),
            )
            if roi_norm is not None:
                with self.lock:
                    current = self.buffers.get(channel_id)
                    if current is live_buffer:
                        current.merge_roi_cache_from(query_buffer)
        except ValueError as exc:
            with self.lock:
                current = self.buffers.get(channel_id)
                if current is live_buffer:
                    current.clear()
            return {"error": str(exc), "frames_indexed": 0}
        return {"results": results, "status": status, "frames_indexed": status.get("frames", 0)}

    def status(self, channel_id: int) -> Dict[str, Any]:
        with self.lock:
            buf = self.buffers.get(channel_id)
            if not buf:
                result = {"frames": 0, "time_range_ms": None}
            else:
                result = buf.status()
        if self.semantic_presence_enabled:
            spatial_status = self.semantic_patch_presence_tracker.status(channel_id)
            spatial_status["semantics"] = (
                "same_forward_top_patch_text_affinity_shadow_v1"
            )
            result["semantic_presence"] = self._merge_spatial_presence(
                self.semantic_presence_tracker.status(channel_id),
                spatial_status,
            )
        return result

    def frame_thumbnail(self, channel_id: int, timestamp_ms: int) -> Optional[str]:
        """Return the JPEG belonging to one exact scored semantic frame.

        Operator tuning must never pair a score with whichever capture frame
        happens to be newest.  Keep this lookup deliberately metadata-only:
        it neither enters the encoder lifecycle lock nor performs inference.
        """

        wanted_timestamp_ms = int(timestamp_ms)
        with self.lock:
            buf = self.buffers.get(int(channel_id))
            if buf is None:
                return None
            for row in reversed(buf.meta):
                if int(row.get("timestamp_ms") or 0) != wanted_timestamp_ms:
                    continue
                thumbnail = str(row.get("thumb") or "").strip()
                return thumbnail or None
        return None

    def frame_embedding(
        self,
        channel_id: int,
        timestamp_ms: int,
        *,
        roi_norm: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Return the already-computed vector for one exact buffered frame."""

        wanted_timestamp_ms = int(timestamp_ms)
        with self.lock:
            buf = self.buffers.get(int(channel_id))
            embedding_space = dict(
                self._buffer_embedding_spaces.get(int(channel_id)) or {}
            )
            if buf is None:
                return None, embedding_space
            for index in range(len(buf.meta) - 1, -1, -1):
                row = buf.meta[index]
                if int(row.get("timestamp_ms") or 0) != wanted_timestamp_ms:
                    continue
                if roi_norm is None:
                    if index >= len(buf.embeddings):
                        return None, embedding_space
                    return np.asarray(
                        buf.embeddings[index],
                        dtype=np.float32,
                    ).copy(), embedding_space
                frame_uid = int(row.get("uid") or 0)
                roi_key = buf._roi_key(roi_norm)
                cached = buf.roi_cache.get(roi_key, {}).get(frame_uid)
                if cached is None:
                    return None, embedding_space
                return np.asarray(cached, dtype=np.float32).copy(), embedding_space
        return None, embedding_space

    def clear(self, channel_id: int) -> None:
        with self.lock:
            self.buffers.pop(channel_id, None)
            self._buffer_embedding_fingerprints.pop(channel_id, None)
            self._buffer_embedding_spaces.pop(channel_id, None)
        self.semantic_presence_tracker.clear_channel(channel_id)
        self.semantic_patch_presence_tracker.clear_channel(channel_id)

    def clear_all(self) -> None:
        with self.lock:
            self.buffers.clear()
            self._buffer_embedding_fingerprints.clear()
            self._buffer_embedding_spaces.clear()
        with self._text_embedding_cache_lock:
            self._text_embedding_cache.clear()
        with self._embedding_space_fingerprint_lock:
            self._embedding_space_fingerprint = ""
        self.semantic_presence_tracker.clear()
        self.semantic_patch_presence_tracker.clear()
