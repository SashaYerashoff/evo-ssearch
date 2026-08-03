"""PTZ-aware camera-motion and recurring-view tracking.

This module deliberately stops at camera/coverage facts.  It does not infer a
maritime event from optical flow and it does not suppress the 1 Hz semantic
archive.  Consumers may use ``spatial_probes_enabled`` to prevent a probe tied
to one view from regulating attention while a PTZ camera is moving or showing
an unconfirmed view.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


CAMERA_MOTION_STATES = frozenset(
    {"steady", "pan", "tilt", "zoom", "preset_cut", "settling"}
)


def scene_fingerprint(frame: Any, *, size: int = 16) -> list[float]:
    """Return a compact, illumination-tolerant fingerprint for one view."""

    array = np.asarray(frame)
    if array.ndim == 3:
        if array.shape[2] == 4:
            array = array[:, :, :3]
        gray = cv2.cvtColor(array.astype(np.uint8, copy=False), cv2.COLOR_RGB2GRAY)
    elif array.ndim == 2:
        gray = array.astype(np.uint8, copy=False)
    else:
        raise ValueError("frame must be a 2D or RGB image array")
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    normalized = resized - float(np.mean(resized))
    norm = float(np.linalg.norm(normalized))
    if norm <= 1e-6:
        return [0.0] * (size * size)
    return (normalized.reshape(-1) / norm).astype(float).tolist()


def fingerprint_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-9:
        return 1.0 if np.allclose(a, b) else 0.0
    return max(-1.0, min(1.0, float(np.dot(a, b) / denominator)))


@dataclass(frozen=True)
class PtzSceneTrackerConfig:
    global_motion_floor_px: float = 0.65
    zoom_motion_floor_px: float = 0.42
    coherence_floor: float = 0.42
    preset_similarity_floor: float = 0.82
    stable_batches_to_confirm: int = 2
    maximum_presets: int = 16


def classify_camera_motion(
    frame_scores: Sequence[Mapping[str, Any]],
    config: PtzSceneTrackerConfig | None = None,
) -> str:
    """Classify a batch from camera-global motion diagnostics.

    The caller supplies values produced by ``RoadMotionAnalyzer``.  Object
    motion is intentionally ignored here unless it is coherent across most of
    the image and therefore looks like camera motion.
    """

    effective = config or PtzSceneTrackerConfig()
    if any(bool(row.get("scene_cut")) for row in frame_scores):
        return "preset_cut"
    candidates: list[Mapping[str, Any]] = []
    for row in frame_scores:
        if bool(row.get("warmup")) or bool(row.get("low_fps_suppressed")):
            continue
        try:
            magnitude = float(row.get("global_motion") or 0.0)
            zoom = abs(float(row.get("global_zoom") or 0.0))
            coherence = float(row.get("global_motion_coherence") or 0.0)
            zoom_coherence = float(row.get("global_zoom_coherence") or 0.0)
        except (TypeError, ValueError):
            continue
        if (
            magnitude >= effective.global_motion_floor_px
            and coherence >= effective.coherence_floor
        ) or (
            zoom >= effective.zoom_motion_floor_px
            and zoom_coherence >= effective.coherence_floor
        ):
            candidates.append(row)
    if not candidates:
        return "steady"
    dx = float(np.median([float(row.get("global_dx") or 0.0) for row in candidates]))
    dy = float(np.median([float(row.get("global_dy") or 0.0) for row in candidates]))
    zoom = float(
        np.median([abs(float(row.get("global_zoom") or 0.0)) for row in candidates])
    )
    translation = math.hypot(dx, dy)
    if zoom >= max(effective.zoom_motion_floor_px, translation * 0.70):
        return "zoom"
    return "pan" if abs(dx) >= abs(dy) else "tilt"


class PtzSceneTracker:
    """Track scene epochs and recurring PTZ views for one channel."""

    def __init__(
        self,
        channel_id: int,
        *,
        config: PtzSceneTrackerConfig | None = None,
        state: Mapping[str, Any] | None = None,
    ) -> None:
        self.channel_id = int(channel_id)
        self.config = config or PtzSceneTrackerConfig()
        self.scene_epoch = 0
        self.motion_state = "steady"
        self.current_preset_id: str | None = None
        self.stable_batches = 0
        self.presets: list[dict[str, Any]] = []
        if state:
            self._restore(state)

    def _restore(self, state: Mapping[str, Any]) -> None:
        self.scene_epoch = max(0, int(state.get("scene_epoch") or 0))
        motion = str(state.get("motion_state") or "steady").strip().lower()
        self.motion_state = motion if motion in CAMERA_MOTION_STATES else "steady"
        preset_id = str(state.get("current_preset_id") or "").strip()
        self.current_preset_id = preset_id or None
        self.stable_batches = max(0, int(state.get("stable_batches") or 0))
        for raw in state.get("presets") or []:
            if not isinstance(raw, Mapping):
                continue
            fingerprint = raw.get("fingerprint")
            if not isinstance(fingerprint, Sequence) or isinstance(
                fingerprint, (str, bytes, bytearray)
            ):
                continue
            values = [float(value) for value in fingerprint]
            if not values:
                continue
            self.presets.append(
                {
                    "preset_id": str(raw.get("preset_id") or f"view-{len(self.presets) + 1}"),
                    "label": str(raw.get("label") or "").strip(),
                    "fingerprint": values,
                    "observations": max(1, int(raw.get("observations") or 1)),
                    "last_seen_ms": max(0, int(raw.get("last_seen_ms") or 0)),
                }
            )
        self.presets = self.presets[: self.config.maximum_presets]

    def snapshot(self) -> dict[str, Any]:
        return {
            "version": 1,
            "channel_id": self.channel_id,
            "scene_epoch": self.scene_epoch,
            "motion_state": self.motion_state,
            "current_preset_id": self.current_preset_id,
            "stable_batches": self.stable_batches,
            "presets": [dict(preset) for preset in self.presets],
        }

    def _match(self, fingerprint: Sequence[float]) -> tuple[dict[str, Any] | None, float]:
        best: dict[str, Any] | None = None
        best_score = -1.0
        for preset in self.presets:
            score = fingerprint_similarity(preset["fingerprint"], fingerprint)
            if score > best_score:
                best = preset
                best_score = score
        if best is None or best_score < self.config.preset_similarity_floor:
            return None, best_score
        return best, best_score

    def observe(
        self,
        frame_scores: Sequence[Mapping[str, Any]],
        fingerprint: Sequence[float] | None,
        *,
        timestamp_ms: int,
    ) -> dict[str, Any]:
        classified = classify_camera_motion(frame_scores, self.config)
        previous_motion = self.motion_state
        moving = classified in {"pan", "tilt", "zoom", "preset_cut"}
        if moving:
            if previous_motion not in {"pan", "tilt", "zoom", "preset_cut"}:
                self.scene_epoch += 1
            self.motion_state = classified
            self.current_preset_id = None
            self.stable_batches = 0
            return self._observation(
                coverage_status="camera_moving",
                preset_status="unavailable",
                similarity=None,
            )

        if previous_motion in {"pan", "tilt", "zoom", "preset_cut", "settling"}:
            self.stable_batches += 1
            if self.stable_batches < self.config.stable_batches_to_confirm:
                self.motion_state = "settling"
                return self._observation(
                    coverage_status="settling",
                    preset_status="unconfirmed",
                    similarity=None,
                )
        else:
            self.stable_batches = max(self.stable_batches + 1, 1)
        self.motion_state = "steady"

        if not fingerprint:
            self.current_preset_id = None
            return self._observation(
                coverage_status="unknown_view",
                preset_status="unavailable",
                similarity=None,
            )
        matched, similarity = self._match(fingerprint)
        if matched is None:
            if len(self.presets) >= self.config.maximum_presets:
                self.current_preset_id = None
                return self._observation(
                    coverage_status="unknown_view",
                    preset_status="capacity_reached",
                    similarity=similarity,
                )
            matched = {
                "preset_id": f"view-{len(self.presets) + 1}",
                "label": "",
                "fingerprint": list(fingerprint),
                "observations": 1,
                "last_seen_ms": int(timestamp_ms),
            }
            self.presets.append(matched)
            preset_status = "provisional"
        else:
            observations = max(1, int(matched.get("observations") or 1)) + 1
            old = np.asarray(matched["fingerprint"], dtype=np.float32)
            new = np.asarray(fingerprint, dtype=np.float32)
            blended = old * 0.85 + new * 0.15
            norm = float(np.linalg.norm(blended))
            if norm > 1e-9:
                matched["fingerprint"] = (blended / norm).astype(float).tolist()
            matched["observations"] = observations
            matched["last_seen_ms"] = int(timestamp_ms)
            preset_status = "known" if observations >= 2 else "provisional"
        previous_preset = self.current_preset_id
        self.current_preset_id = str(matched["preset_id"])
        if previous_preset and previous_preset != self.current_preset_id:
            self.scene_epoch += 1
        coverage = "visible" if preset_status == "known" else "unknown_view"
        return self._observation(
            coverage_status=coverage,
            preset_status=preset_status,
            similarity=similarity if similarity >= 0.0 else None,
        )

    def _observation(
        self,
        *,
        coverage_status: str,
        preset_status: str,
        similarity: float | None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "version": 1,
            "channel_id": self.channel_id,
            "camera_motion": self.motion_state,
            "scene_epoch": self.scene_epoch,
            "coverage_status": coverage_status,
            "preset_id": self.current_preset_id,
            "preset_status": preset_status,
            "known_preset_count": sum(
                1 for preset in self.presets if int(preset.get("observations") or 0) >= 2
            ),
            "spatial_probes_enabled": (
                self.motion_state == "steady"
                and coverage_status == "visible"
                and preset_status == "known"
            ),
            "coverage_semantics": (
                "not_observed_when_view_unavailable;never_infer_absence_from_ptz_coverage_gap"
            ),
        }
        if similarity is not None and math.isfinite(similarity):
            result["preset_similarity"] = round(float(similarity), 4)
        return result
