from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .scene import RoadSceneCard, RoadZone, zone_mask


@dataclass(frozen=True)
class MotionAnalyzerConfig:
    max_edge: int = 360
    min_motion_px: float = 0.7
    active_ratio_floor: float = 0.012
    active_ratio_ceiling: float = 0.55
    mean_motion_floor: float = 0.18
    p90_motion_floor: float = 0.9
    max_motion_claim_interval_ms: int = 3500
    wrong_way_alignment: float = -0.45
    cross_flow_alignment_abs_ceiling: float = 0.25
    cross_flow_ratio_floor: float = 0.04
    directional_active_ratio_ceiling: float = 0.45
    compensate_global_motion: bool = True
    compensation_active_ratio_ceiling: float = 0.35
    scene_cut_mean_diff_floor: float = 0.075
    scene_cut_changed_ratio_floor: float = 0.18
    scene_cut_p95_diff_floor: float = 0.22
    scene_cut_hist_distance_floor: float = 0.18


@dataclass(frozen=True)
class RoadMotionCue:
    channel_id: int
    timestamp_ms: int
    zone_name: str
    cue_type: str
    score: float
    evidence: str
    metrics: dict[str, float] = field(default_factory=dict)
    frame_index: int | None = None


@dataclass(frozen=True)
class RoadMotionSample:
    channel_id: int
    timestamp_ms: int
    frame_index: int
    cues: tuple[RoadMotionCue, ...] = field(default_factory=tuple)
    zone_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    global_motion: dict[str, float] = field(default_factory=dict)
    quality: dict[str, float] = field(default_factory=dict)
    warmup: bool = False
    scene_cut: bool = False


def _to_rgb_array(frame: Any) -> np.ndarray:
    if isinstance(frame, Image.Image):
        return np.asarray(frame.convert("RGB"))
    arr = np.asarray(frame)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.ndim != 3:
        raise ValueError("frame must be a 2D or 3D image array")
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.shape[2] != 3:
        raise ValueError("frame must have 3 or 4 channels")
    return arr.astype(np.uint8, copy=False)


def _resize_gray(frame: Any, max_edge: int) -> np.ndarray:
    rgb = _to_rgb_array(frame)
    height, width = rgb.shape[:2]
    edge = max(height, width)
    if edge > max_edge:
        scale = float(max_edge) / float(edge)
        rgb = cv2.resize(
            rgb,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _cosine_alignment(dx: float, dy: float, expected: tuple[float, float] | None) -> float:
    if expected is None:
        return 0.0
    magnitude = (dx * dx + dy * dy) ** 0.5
    if magnitude <= 1e-9:
        return 0.0
    return (dx * expected[0] + dy * expected[1]) / magnitude


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class RoadMotionAnalyzer:
    """Frame-diff/optical-flow road cue extractor.

    The analyzer emits candidate cues.  It deliberately does not claim vehicle
    identity or violations; downstream CLIP/VLM/evidence review must confirm.
    """

    def __init__(self, scene_card: RoadSceneCard, config: MotionAnalyzerConfig | None = None):
        self.scene_card = scene_card
        self.config = config or MotionAnalyzerConfig()
        self._previous_gray: np.ndarray | None = None
        self._previous_timestamp_ms: int | None = None

    def reset(self) -> None:
        self._previous_gray = None
        self._previous_timestamp_ms = None

    def analyze_frame(self, frame: Any, timestamp_ms: int, frame_index: int = 0) -> RoadMotionSample:
        timestamp_ms = int(timestamp_ms)
        gray = _resize_gray(frame, max(64, int(self.config.max_edge)))
        frame_interval_ms = (
            max(0, timestamp_ms - self._previous_timestamp_ms)
            if self._previous_timestamp_ms is not None
            else None
        )
        if self._previous_gray is None:
            self._previous_gray = gray
            self._previous_timestamp_ms = timestamp_ms
            return RoadMotionSample(
                channel_id=self.scene_card.channel_id,
                timestamp_ms=timestamp_ms,
                frame_index=int(frame_index),
                quality={"frame_interval_ms": 0.0},
                warmup=True,
            )
        if self._previous_gray.shape != gray.shape:
            self._previous_gray = cv2.resize(self._previous_gray, (gray.shape[1], gray.shape[0]))
        cut_metrics = self._scene_cut_metrics(self._previous_gray, gray)
        if bool(cut_metrics["scene_cut"]):
            self._previous_gray = gray
            self._previous_timestamp_ms = timestamp_ms
            return RoadMotionSample(
                channel_id=self.scene_card.channel_id,
                timestamp_ms=timestamp_ms,
                frame_index=int(frame_index),
                global_motion=cut_metrics,
                quality={
                    "frame_interval_ms": float(frame_interval_ms or 0),
                    "low_fps_suppressed": 1.0
                    if (frame_interval_ms or 0) > self.config.max_motion_claim_interval_ms
                    else 0.0,
                },
                warmup=True,
                scene_cut=True,
            )

        flow = cv2.calcOpticalFlowFarneback(
            self._previous_gray,
            gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        self._previous_gray = gray
        self._previous_timestamp_ms = timestamp_ms
        raw_active_ratio = self._active_ratio(flow)
        global_metrics = self._global_motion(flow)
        global_metrics["active_ratio"] = raw_active_ratio
        compensated = False
        if (
            self.config.compensate_global_motion
            and raw_active_ratio <= self.config.compensation_active_ratio_ceiling
        ):
            flow = flow.copy()
            flow[:, :, 0] -= global_metrics["dx"]
            flow[:, :, 1] -= global_metrics["dy"]
            compensated = True
        global_metrics["compensated"] = 1.0 if compensated else 0.0
        low_fps_suppressed = (
            frame_interval_ms is not None
            and frame_interval_ms > self.config.max_motion_claim_interval_ms
        )
        quality = {
            "frame_interval_ms": float(frame_interval_ms or 0),
            "low_fps_suppressed": 1.0 if low_fps_suppressed else 0.0,
            "raw_active_ratio": raw_active_ratio,
        }

        cues: list[RoadMotionCue] = []
        zone_metrics: dict[str, dict[str, float]] = {}
        zones = self.scene_card.active_zones()
        if not zones:
            zones = (RoadZone(name="full_frame"),)
        for zone in zones:
            metrics = self._zone_metrics(flow, zone)
            metrics.update(quality)
            zone_metrics[zone.name] = metrics
            cues.extend(self._cues_for_zone(zone, metrics, int(timestamp_ms), int(frame_index)))
        return RoadMotionSample(
            channel_id=self.scene_card.channel_id,
            timestamp_ms=timestamp_ms,
            frame_index=int(frame_index),
            cues=tuple(cues),
            zone_metrics=zone_metrics,
            global_motion=global_metrics,
            quality=quality,
            warmup=False,
        )

    def _active_ratio(self, flow: np.ndarray) -> float:
        mag = np.linalg.norm(flow, axis=2)
        active = mag >= self.config.min_motion_px
        return float(np.count_nonzero(active) / max(1, active.size))

    def _global_motion(self, flow: np.ndarray) -> dict[str, float]:
        values = flow.reshape(-1, 2)
        dx = float(np.median(values[:, 0])) if values.size else 0.0
        dy = float(np.median(values[:, 1])) if values.size else 0.0
        magnitudes = np.linalg.norm(values, axis=1) if values.size else np.asarray([], dtype=np.float32)
        median_magnitude = float(np.median(magnitudes)) if magnitudes.size else 0.0
        translation = float((dx * dx + dy * dy) ** 0.5)

        # A PTZ zoom produces radial flow whose median dx/dy is near zero.  A
        # robust radial projection lets the camera-scene tracker distinguish
        # that global view change from local object motion without decoding a
        # second optical-flow pass.
        height, width = flow.shape[:2]
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        xx -= (max(1, width) - 1) / 2.0
        yy -= (max(1, height) - 1) / 2.0
        radius = np.sqrt(xx * xx + yy * yy)
        radius = np.maximum(radius, 1.0)
        residual_x = flow[:, :, 0] - dx
        residual_y = flow[:, :, 1] - dy
        radial_projection = (residual_x * xx + residual_y * yy) / radius
        zoom = float(np.median(radial_projection)) if radial_projection.size else 0.0
        median_abs_zoom = (
            float(np.median(np.abs(radial_projection)))
            if radial_projection.size
            else 0.0
        )
        return {
            "dx": dx,
            "dy": dy,
            "magnitude": translation,
            "median_flow_magnitude": median_magnitude,
            "coherence": translation / max(1e-6, median_magnitude),
            "zoom": zoom,
            "zoom_coherence": abs(zoom) / max(1e-6, median_abs_zoom),
        }

    def _scene_cut_metrics(self, previous: np.ndarray, current: np.ndarray) -> dict[str, float]:
        diff = cv2.absdiff(previous, current).astype(np.float32) / 255.0
        mean_diff = float(np.mean(diff)) if diff.size else 0.0
        changed_ratio = float(np.count_nonzero(diff >= 0.12) / max(1, diff.size))
        p95_diff = float(np.percentile(diff, 95)) if diff.size else 0.0
        prev_hist = cv2.calcHist([previous], [0], None, [32], [0, 256])
        curr_hist = cv2.calcHist([current], [0], None, [32], [0, 256])
        cv2.normalize(prev_hist, prev_hist)
        cv2.normalize(curr_hist, curr_hist)
        hist_distance = float(cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA))
        scene_cut = (
            mean_diff >= self.config.scene_cut_mean_diff_floor
            and changed_ratio >= self.config.scene_cut_changed_ratio_floor
            and p95_diff >= self.config.scene_cut_p95_diff_floor
            and hist_distance >= self.config.scene_cut_hist_distance_floor
        )
        return {
            "scene_cut": 1.0 if scene_cut else 0.0,
            "mean_absdiff": mean_diff,
            "changed_ratio": changed_ratio,
            "p95_absdiff": p95_diff,
            "hist_distance": hist_distance,
        }

    def _zone_metrics(self, flow: np.ndarray, zone: RoadZone) -> dict[str, float]:
        height, width = flow.shape[:2]
        mask = zone_mask(width, height, zone).astype(bool)
        zone_flow = flow[mask]
        if zone_flow.size == 0:
            return {
                "active_ratio": 0.0,
                "mean_motion": 0.0,
                "p90_motion": 0.0,
                "mean_dx": 0.0,
                "mean_dy": 0.0,
                "alignment": 0.0,
            }
        mag = np.linalg.norm(zone_flow, axis=1)
        active = mag >= self.config.min_motion_px
        active_ratio = float(np.count_nonzero(active) / max(1, mag.size))
        if np.any(active):
            active_flow = zone_flow[active]
            active_mag = mag[active]
        else:
            active_flow = zone_flow
            active_mag = mag
        mean_dx = float(np.mean(active_flow[:, 0])) if active_flow.size else 0.0
        mean_dy = float(np.mean(active_flow[:, 1])) if active_flow.size else 0.0
        mean_motion = float(np.mean(active_mag)) if active_mag.size else 0.0
        p90_motion = float(np.percentile(mag, 90)) if mag.size else 0.0
        return {
            "active_ratio": active_ratio,
            "mean_motion": mean_motion,
            "p90_motion": p90_motion,
            "mean_dx": mean_dx,
            "mean_dy": mean_dy,
            "alignment": _cosine_alignment(mean_dx, mean_dy, zone.expected_flow),
        }

    def _cues_for_zone(
        self,
        zone: RoadZone,
        metrics: dict[str, float],
        timestamp_ms: int,
        frame_index: int,
    ) -> list[RoadMotionCue]:
        cues: list[RoadMotionCue] = []
        active_ratio = metrics["active_ratio"]
        mean_motion = metrics["mean_motion"]
        p90_motion = metrics["p90_motion"]
        alignment = metrics["alignment"]
        low_fps = bool(metrics.get("low_fps_suppressed", 0.0) >= 1.0)
        too_wide = active_ratio > self.config.active_ratio_ceiling
        burst = (
            not low_fps
            and not too_wide
            and active_ratio >= self.config.active_ratio_floor
            and (
                p90_motion >= self.config.p90_motion_floor
                or mean_motion >= self.config.mean_motion_floor
            )
        )
        if burst:
            score = _clamp_score(
                max(
                    p90_motion / max(0.1, self.config.p90_motion_floor),
                    mean_motion / max(0.1, self.config.mean_motion_floor),
                )
                * min(1.0, active_ratio / max(1e-6, self.config.active_ratio_floor))
                / 2.0
            )
            cues.append(
                RoadMotionCue(
                    channel_id=self.scene_card.channel_id,
                    timestamp_ms=timestamp_ms,
                    zone_name=zone.name,
                    cue_type="road_motion_burst",
                    score=score,
                    evidence=(
                        f"motion burst in {zone.name}: active_ratio={active_ratio:.3f}, "
                        f"p90={p90_motion:.2f}px"
                    ),
                    metrics=dict(metrics),
                    frame_index=frame_index,
                )
            )
        directional_allowed = (
            zone.expected_flow is not None
            and burst
            and active_ratio <= self.config.directional_active_ratio_ceiling
        )
        if directional_allowed and alignment <= self.config.wrong_way_alignment:
            cues.append(
                RoadMotionCue(
                    channel_id=self.scene_card.channel_id,
                    timestamp_ms=timestamp_ms,
                    zone_name=zone.name,
                    cue_type="opposing_flow_candidate",
                    score=_clamp_score(abs(alignment)),
                    evidence=f"motion direction opposes expected flow in {zone.name}: alignment={alignment:.2f}",
                    metrics=dict(metrics),
                    frame_index=frame_index,
                )
            )
        if (
            directional_allowed
            and abs(alignment) <= self.config.cross_flow_alignment_abs_ceiling
            and active_ratio >= self.config.cross_flow_ratio_floor
        ):
            cues.append(
                RoadMotionCue(
                    channel_id=self.scene_card.channel_id,
                    timestamp_ms=timestamp_ms,
                    zone_name=zone.name,
                    cue_type="cross_flow_candidate",
                    score=_clamp_score(1.0 - abs(alignment)),
                    evidence=f"motion is mostly cross-flow in {zone.name}: alignment={alignment:.2f}",
                    metrics=dict(metrics),
                    frame_index=frame_index,
                )
            )
        return cues
