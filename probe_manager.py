import base64
import os
import threading
import time
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
from PIL import Image

from config import config


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

    def add(self, embedding: np.ndarray, timestamp_ms: int, channel_id: int, thumb: str) -> None:
        emb = self._normalize_vec(embedding)
        frame_uid = int(self.next_uid)
        self.next_uid += 1
        self.embeddings.append(emb)
        self.meta.append(
            {
                "uid": frame_uid,
                "timestamp_ms": timestamp_ms,
                "channel_id": channel_id,
                "thumb": thumb,
            }
        )
        if len(self.embeddings) > self.max_frames:
            excess = len(self.embeddings) - self.max_frames
            if excess > 0:
                removed = self.meta[:excess]
                self.embeddings = self.embeddings[excess:]
                self.meta = self.meta[excess:]
                removed_uids = [uid for uid in (_to_optional_int(item.get("uid")) for item in removed) if uid is not None]
                self._prune_roi_cache_uids(removed_uids)
        self._rebuild_index()

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
        selected_meta = [self.meta[idx] for idx in selected_idx]
        if roi_norm is None or embed_image_fn is None:
            mat = np.stack([self.embeddings[idx] for idx in selected_idx], axis=0).astype(np.float32)
            return mat, selected_meta

        roi_key = self._roi_key(roi_norm)
        cache = self.roi_cache.setdefault(roi_key, {})
        self._remember_roi_key(roi_key)
        vectors: List[np.ndarray] = []
        for idx, row in zip(selected_idx, selected_meta):
            uid = int(row.get("uid") or 0)
            cached = cache.get(uid)
            if cached is not None:
                vectors.append(cached)
                continue
            embedded = self._embed_roi_thumb(
                str(row.get("thumb") or ""),
                roi_norm=roi_norm,
                embed_image_fn=embed_image_fn,
                roi_padding=roi_padding,
            )
            vec = embedded if embedded is not None else self.embeddings[idx]
            cache[uid] = vec
            vectors.append(vec)
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
    ) -> List[Dict[str, Any]]:
        if not self.embeddings:
            return []
        mat, meta = self._mat_for_query(
            min_ts_ms=min_ts_ms,
            roi_norm=roi_norm,
            embed_image_fn=embed_image_fn,
            roi_padding=roi_padding,
        )
        if mat.size == 0 or not meta:
            return []
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
    ) -> None:
        self.buffers: Dict[int, ProbeBuffer] = {}
        self.lock = threading.Lock()
        self.max_frames = config.PROBE_MAX_FRAMES
        self.thumb_edge = config.PROBE_THUMB_MAX_EDGE
        self.embed_image_fn = embed_image_fn
        self.embed_text_fn = embed_text_fn
        self.jpeg_encoder = jpeg_encoder

    def _buffer(self, channel_id: int) -> ProbeBuffer:
        if channel_id not in self.buffers:
            self.buffers[channel_id] = ProbeBuffer(self.max_frames, self.thumb_edge)
        return self.buffers[channel_id]

    def add_frame(self, channel_id: int, pil_image: Image.Image, timestamp_ms: Optional[int]) -> None:
        ts_ms = timestamp_ms or int(time.time() * 1000)
        emb = self.embed_image_fn(pil_image)
        thumb = self.jpeg_encoder(pil_image, max_edge=self.thumb_edge, quality=70)
        with self.lock:
            buf = self._buffer(channel_id)
            buf.add(emb, ts_ms, channel_id, thumb)

    def _embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        embs = []
        for t in texts:
            if not t or not str(t).strip():
                continue
            embs.append(self.embed_text_fn(str(t)))
        if not embs:
            return np.zeros((0, 512), dtype=np.float32)
        mat = np.stack(embs, axis=0).astype(np.float32)
        mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
        return mat

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
    ) -> Dict[str, Any]:
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
            pos_embs = np.vstack([pos_embs, image_emb[np.newaxis, :]]) if pos_embs.size else image_emb[np.newaxis, :]
        neg_embs = self._embed_texts(neg_texts)
        if image_pos_floor is not None:
            pos_floor = max(pos_floor, image_pos_floor)
        min_ts_ms: Optional[int] = None
        if window_sec and window_sec > 0:
            min_ts_ms = int((time.time() - float(window_sec)) * 1000)
        with self.lock:
            buf = self.buffers.get(channel_id)
            if not buf:
                return {"results": [], "frames_indexed": 0}
            results = buf.query(
                pos_embs,
                neg_embs,
                pos_floor,
                margin_thr,
                top_k,
                min_ts_ms=min_ts_ms,
                roi_norm=roi_norm,
                embed_image_fn=self.embed_image_fn,
                roi_padding=roi_padding,
            )
            status = buf.status()
        return {"results": results, "status": status, "frames_indexed": status.get("frames", 0)}

    def status(self, channel_id: int) -> Dict[str, Any]:
        with self.lock:
            buf = self.buffers.get(channel_id)
            if not buf:
                return {"frames": 0, "time_range_ms": None}
            return buf.status()

    def clear(self, channel_id: int) -> None:
        with self.lock:
            self.buffers.pop(channel_id, None)
