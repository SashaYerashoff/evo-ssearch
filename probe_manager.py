import base64
import threading
import time
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Sequence, cast

import faiss  # type: ignore
import numpy as np
from PIL import Image

from config import config


def _faiss_add(index: faiss.IndexFlatIP, vectors: np.ndarray) -> None:
    """FAISS Python API monkey-patches add(), but stubs may expose C++ signature."""
    cast(Any, index).add(vectors)


class ProbeBuffer:
    def __init__(self, max_frames: int, thumb_edge: int) -> None:
        self.max_frames = max_frames
        self.thumb_edge = thumb_edge
        self.embeddings: List[np.ndarray] = []
        self.meta: List[Dict[str, Any]] = []
        self.index: Optional[faiss.IndexFlatIP] = None

    def _rebuild_index(self) -> None:
        if not self.embeddings:
            self.index = None
            return
        mat = np.stack(self.embeddings, axis=0).astype(np.float32)
        self.index = faiss.IndexFlatIP(mat.shape[1])
        _faiss_add(self.index, mat)

    def add(self, embedding: np.ndarray, timestamp_ms: int, channel_id: int, thumb: str) -> None:
        if embedding.ndim != 1:
            embedding = embedding.flatten()
        emb = embedding.astype(np.float32)
        emb = emb / max(np.linalg.norm(emb), 1e-8)
        self.embeddings.append(emb)
        self.meta.append(
            {
                "timestamp_ms": timestamp_ms,
                "channel_id": channel_id,
                "thumb": thumb,
            }
        )
        if len(self.embeddings) > self.max_frames:
            excess = len(self.embeddings) - self.max_frames
            if excess > 0:
                self.embeddings = self.embeddings[excess:]
                self.meta = self.meta[excess:]
        self._rebuild_index()

    def query(
        self,
        pos_embs: np.ndarray,
        neg_embs: np.ndarray,
        pos_floor: float,
        margin_thr: float,
        top_k: int,
        min_ts_ms: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not self.embeddings or self.index is None:
            return []
        mat = np.stack(self.embeddings, axis=0).astype(np.float32)
        meta = list(self.meta)
        if min_ts_ms is not None:
            keep = [i for i, m in enumerate(meta) if m.get("timestamp_ms") and m["timestamp_ms"] >= min_ts_ms]
            if not keep:
                return []
            mat = mat[keep]
            meta = [meta[i] for i in keep]
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
            results = buf.query(pos_embs, neg_embs, pos_floor, margin_thr, top_k, min_ts_ms=min_ts_ms)
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
