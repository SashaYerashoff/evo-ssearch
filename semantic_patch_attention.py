"""Bounded helpers for experimental SigLIP2 patch/text affinity maps.

The pooled SigLIP2 embedding is trained for image/text matching. Individual
vision tokens are useful for an operator-facing localization hint, but they are
not calibrated detections or segmentation masks. This module therefore only
turns one ephemeral patch matrix into a compact relative heatmap; callers must
keep the result out of alerting and durable evidence.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def _smooth_grid(grid: np.ndarray) -> np.ndarray:
    """Apply one small edge-preserving-ish box pass without extra deps."""

    rows, cols = grid.shape
    padded = np.pad(grid, ((1, 1), (1, 1)), mode="edge")
    smoothed = np.zeros_like(grid, dtype=np.float32)
    weights = np.asarray(
        ((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0)),
        dtype=np.float32,
    )
    for row in range(rows):
        for col in range(cols):
            window = padded[row : row + 3, col : col + 3]
            smoothed[row, col] = float(np.sum(window * weights) / 16.0)
    return smoothed


def _peak_component_roi(
    heatmap: np.ndarray,
    *,
    minimum_contrast: float,
    raw_contrast: float,
) -> Optional[Dict[str, float]]:
    if heatmap.size == 0 or raw_contrast < minimum_contrast:
        return None
    rows, cols = heatmap.shape
    peak_index = int(np.argmax(heatmap))
    peak_row, peak_col = divmod(peak_index, cols)
    threshold = max(0.58, float(np.quantile(heatmap, 0.80)))
    active = heatmap >= threshold
    if not bool(active[peak_row, peak_col]):
        return None

    pending = deque([(peak_row, peak_col)])
    visited = {(peak_row, peak_col)}
    component: List[Tuple[int, int]] = []
    while pending:
        row, col = pending.popleft()
        if not bool(active[row, col]):
            continue
        component.append((row, col))
        for next_row, next_col in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if not (0 <= next_row < rows and 0 <= next_col < cols):
                continue
            point = (next_row, next_col)
            if point in visited:
                continue
            visited.add(point)
            pending.append(point)
    if not component:
        return None

    row_values = [point[0] for point in component]
    col_values = [point[1] for point in component]
    # Half a cell of context keeps the suggested crop from cutting directly
    # through the strongest patch while remaining bounded to the frame.
    x0 = max(0.0, (min(col_values) - 0.5) / cols)
    y0 = max(0.0, (min(row_values) - 0.5) / rows)
    x1 = min(1.0, (max(col_values) + 1.5) / cols)
    y1 = min(1.0, (max(row_values) + 1.5) / rows)
    return {
        "x": round(x0, 6),
        "y": round(y0, 6),
        "w": round(max(0.0, x1 - x0), 6),
        "h": round(max(0.0, y1 - y0), 6),
    }


def build_patch_affinity_payload(
    patch_embeddings: np.ndarray,
    text_embedding: np.ndarray,
    *,
    rows: int,
    cols: int,
    minimum_contrast: float = 0.005,
) -> Dict[str, Any]:
    """Return a bounded relative heatmap without exposing patch embeddings."""

    row_count = int(rows)
    col_count = int(cols)
    if row_count <= 0 or col_count <= 0 or row_count * col_count > 4096:
        raise ValueError("invalid SigLIP2 patch grid")
    patches = np.asarray(patch_embeddings, dtype=np.float32)
    text = np.asarray(text_embedding, dtype=np.float32).reshape(-1)
    expected = row_count * col_count
    if patches.ndim != 2 or int(patches.shape[0]) < expected:
        raise ValueError("SigLIP2 patch tensor does not match its spatial grid")
    if not text.size or int(patches.shape[1]) != int(text.shape[0]):
        raise ValueError("SigLIP2 patch and text dimensions do not match")

    normalized_patches = _normalize_rows(patches[:expected])
    normalized_text = text / max(float(np.linalg.norm(text)), 1e-8)
    raw_grid = (normalized_patches @ normalized_text).reshape(row_count, col_count)
    raw_grid = _smooth_grid(raw_grid)
    low = float(np.quantile(raw_grid, 0.10))
    high = float(np.quantile(raw_grid, 0.90))
    contrast = max(0.0, high - low)
    if contrast <= 1e-8:
        relative = np.zeros_like(raw_grid, dtype=np.float32)
    else:
        relative = np.clip((raw_grid - low) / contrast, 0.0, 1.0)
    peak_index = int(np.argmax(relative)) if relative.size else 0
    peak_row, peak_col = divmod(peak_index, col_count)
    suggested_roi = _peak_component_roi(
        relative,
        minimum_contrast=max(0.0, float(minimum_contrast)),
        raw_contrast=contrast,
    )
    return {
        "semantics": "experimental_relative_patch_text_affinity_not_detection",
        "grid": {"rows": row_count, "cols": col_count},
        "heatmap": [round(float(value), 4) for value in relative.reshape(-1)],
        "raw_range": {
            "p10": round(low, 6),
            "p90": round(high, 6),
            "contrast": round(contrast, 6),
        },
        "peak_cell": {"row": peak_row, "col": peak_col},
        "suggested_roi": suggested_roi,
    }

