from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from .motion import MotionAnalyzerConfig, _resize_gray
from .scene import RoadSceneCard, RoadZone
from .video import DecodedVideoFrame


@dataclass(frozen=True)
class AutoSceneCardConfig:
    max_edge: int = 240
    min_motion_px: float = 0.8
    min_frames: int = 12
    min_motion_pairs: int = 4
    heat_threshold: float = 0.018
    min_zone_area_ratio: float = 0.015
    max_pair_active_ratio: float = 0.55
    padding_ratio: float = 0.08
    flow_dominance_floor: float = 0.35
    scene_cut_config: MotionAnalyzerConfig = MotionAnalyzerConfig(max_edge=240)


@dataclass(frozen=True)
class AutoSceneCardResult:
    scene_card: RoadSceneCard
    confidence: str
    reason: str
    frame_count: int
    motion_pair_count: int
    scene_cut_count: int
    zone_area_ratio: float
    flow_dominance: float

    def as_dict(self) -> dict:
        return {
            "scene_card": {
                "channel_id": self.scene_card.channel_id,
                "title": self.scene_card.title,
                "version": self.scene_card.version,
                "notes": self.scene_card.notes,
                "zones": [
                    {
                        "name": zone.name,
                        "polygon": [list(point) for point in zone.polygon],
                        "zone_type": zone.zone_type,
                        "expected_flow": list(zone.expected_flow) if zone.expected_flow else None,
                        "enabled": zone.enabled,
                    }
                    for zone in self.scene_card.zones
                ],
            },
            "confidence": self.confidence,
            "reason": self.reason,
            "frame_count": self.frame_count,
            "motion_pair_count": self.motion_pair_count,
            "scene_cut_count": self.scene_cut_count,
            "zone_area_ratio": round(float(self.zone_area_ratio), 4),
            "flow_dominance": round(float(self.flow_dominance), 4),
        }


def _scene_cut(previous: np.ndarray, current: np.ndarray, config: MotionAnalyzerConfig) -> bool:
    diff = cv2.absdiff(previous, current).astype(np.float32) / 255.0
    mean_diff = float(np.mean(diff)) if diff.size else 0.0
    changed_ratio = float(np.count_nonzero(diff >= 0.12) / max(1, diff.size))
    p95_diff = float(np.percentile(diff, 95)) if diff.size else 0.0
    prev_hist = cv2.calcHist([previous], [0], None, [32], [0, 256])
    curr_hist = cv2.calcHist([current], [0], None, [32], [0, 256])
    cv2.normalize(prev_hist, prev_hist)
    cv2.normalize(curr_hist, curr_hist)
    hist_distance = float(cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA))
    return (
        mean_diff >= config.scene_cut_mean_diff_floor
        and changed_ratio >= config.scene_cut_changed_ratio_floor
        and p95_diff >= config.scene_cut_p95_diff_floor
        and hist_distance >= config.scene_cut_hist_distance_floor
    )


def _full_frame_result(
    channel_id: int,
    title: str,
    *,
    reason: str,
    frame_count: int,
    motion_pair_count: int,
    scene_cut_count: int,
) -> AutoSceneCardResult:
    card = RoadSceneCard(
        channel_id=int(channel_id),
        title=title,
        zones=(
            RoadZone(
                name="auto_full_frame_degraded",
                polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
                zone_type="degraded_full_frame",
                expected_flow=None,
            ),
        ),
        notes=f"Auto scene inference degraded: {reason}. Use for motion candidates only; wrong-way semantics disabled.",
    )
    return AutoSceneCardResult(
        scene_card=card,
        confidence="low",
        reason=reason,
        frame_count=frame_count,
        motion_pair_count=motion_pair_count,
        scene_cut_count=scene_cut_count,
        zone_area_ratio=1.0,
        flow_dominance=0.0,
    )


def infer_scene_card_from_frames(
    channel_id: int,
    title: str,
    frames: Iterable[DecodedVideoFrame],
    *,
    config: AutoSceneCardConfig | None = None,
) -> AutoSceneCardResult:
    """Infer a conservative road zone from motion history.

    This is not semantic segmentation.  It is a bounded bootstrap step for
    unmapped channels: find the stable image region where vehicle-like motion
    repeatedly occurs and infer a dominant flow only when the motion vectors
    agree enough.  If the evidence is weak, return a degraded full-frame card.
    """

    cfg = config or AutoSceneCardConfig()
    frame_list = list(frames)
    frame_count = len(frame_list)
    if frame_count < max(2, int(cfg.min_frames)):
        return _full_frame_result(
            channel_id,
            title,
            reason=f"insufficient frames for auto mapping ({frame_count})",
            frame_count=frame_count,
            motion_pair_count=0,
            scene_cut_count=0,
        )

    previous_gray: np.ndarray | None = None
    heat: np.ndarray | None = None
    flow_sum = np.array([0.0, 0.0], dtype=np.float64)
    flow_mag_sum = 0.0
    flow_count = 0
    motion_pair_count = 0
    scene_cut_count = 0

    for frame in frame_list:
        gray = _resize_gray(frame.image, max(64, int(cfg.max_edge)))
        if previous_gray is None:
            previous_gray = gray
            heat = np.zeros(gray.shape, dtype=np.float32)
            continue
        if previous_gray.shape != gray.shape:
            previous_gray = cv2.resize(previous_gray, (gray.shape[1], gray.shape[0]))
        if _scene_cut(previous_gray, gray, cfg.scene_cut_config):
            scene_cut_count += 1
            previous_gray = gray
            continue
        flow = cv2.calcOpticalFlowFarneback(previous_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        previous_gray = gray
        values = flow.reshape(-1, 2)
        flow[:, :, 0] -= float(np.median(values[:, 0])) if values.size else 0.0
        flow[:, :, 1] -= float(np.median(values[:, 1])) if values.size else 0.0
        mag = np.linalg.norm(flow, axis=2)
        active = mag >= max(0.1, float(cfg.min_motion_px))
        active_ratio = float(np.count_nonzero(active) / max(1, active.size))
        if active_ratio <= 0.0 or active_ratio > float(cfg.max_pair_active_ratio):
            continue
        if heat is None:
            heat = np.zeros(gray.shape, dtype=np.float32)
        heat[active] += 1.0
        active_flow = flow[active]
        active_mag = mag[active]
        if active_flow.size:
            flow_sum += np.sum(active_flow, axis=0)
            flow_mag_sum += float(np.sum(active_mag))
            flow_count += int(active_flow.shape[0])
        motion_pair_count += 1

    if heat is None or motion_pair_count < int(cfg.min_motion_pairs):
        return _full_frame_result(
            channel_id,
            title,
            reason=f"insufficient stable motion pairs ({motion_pair_count})",
            frame_count=frame_count,
            motion_pair_count=motion_pair_count,
            scene_cut_count=scene_cut_count,
        )

    heat = heat / max(1, motion_pair_count)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=3.0)
    nonzero = heat[heat > 0]
    if nonzero.size == 0:
        return _full_frame_result(
            channel_id,
            title,
            reason="no stable motion heat after scene-cut filtering",
            frame_count=frame_count,
            motion_pair_count=motion_pair_count,
            scene_cut_count=scene_cut_count,
        )
    percentile_threshold = float(np.percentile(nonzero, 60))
    threshold = max(float(cfg.heat_threshold), percentile_threshold)
    mask = (heat >= threshold).astype(np.uint8) * 255
    kernel = np.ones((7, 7), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    components, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if components <= 1:
        return _full_frame_result(
            channel_id,
            title,
            reason="no connected motion zone",
            frame_count=frame_count,
            motion_pair_count=motion_pair_count,
            scene_cut_count=scene_cut_count,
        )
    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x = int(stats[largest_idx, cv2.CC_STAT_LEFT])
    y = int(stats[largest_idx, cv2.CC_STAT_TOP])
    width = int(stats[largest_idx, cv2.CC_STAT_WIDTH])
    height = int(stats[largest_idx, cv2.CC_STAT_HEIGHT])
    image_height, image_width = mask.shape[:2]
    pad_x = int(round(image_width * float(cfg.padding_ratio)))
    pad_y = int(round(image_height * float(cfg.padding_ratio)))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(image_width - 1, x + width + pad_x)
    y1 = min(image_height - 1, y + height + pad_y)
    zone_area_ratio = float(((x1 - x0 + 1) * (y1 - y0 + 1)) / max(1, image_width * image_height))
    if zone_area_ratio < float(cfg.min_zone_area_ratio):
        return _full_frame_result(
            channel_id,
            title,
            reason=f"motion zone too small ({zone_area_ratio:.3f})",
            frame_count=frame_count,
            motion_pair_count=motion_pair_count,
            scene_cut_count=scene_cut_count,
        )

    inferred_flow = None
    flow_dominance = 0.0
    if flow_count > 0 and flow_mag_sum > 1e-9:
        mean_vector = flow_sum / float(flow_count)
        mean_magnitude = float((mean_vector[0] ** 2 + mean_vector[1] ** 2) ** 0.5)
        mean_active_magnitude = flow_mag_sum / float(flow_count)
        flow_dominance = mean_magnitude / max(1e-9, mean_active_magnitude)
        if flow_dominance >= float(cfg.flow_dominance_floor) and mean_magnitude > 1e-6:
            inferred_flow = (float(mean_vector[0] / mean_magnitude), float(mean_vector[1] / mean_magnitude))

    polygon = (
        (x0 / max(1, image_width - 1), y0 / max(1, image_height - 1)),
        (x1 / max(1, image_width - 1), y0 / max(1, image_height - 1)),
        (x1 / max(1, image_width - 1), y1 / max(1, image_height - 1)),
        (x0 / max(1, image_width - 1), y1 / max(1, image_height - 1)),
    )
    confidence = "medium" if inferred_flow is not None and zone_area_ratio < 0.75 else "low"
    expected_flow = inferred_flow if confidence == "medium" else None
    reason = (
        "motion-history zone inferred"
        if expected_flow is not None
        else "motion-history zone inferred without dominant flow; wrong-way semantics disabled"
    )
    card = RoadSceneCard(
        channel_id=int(channel_id),
        title=title,
        zones=(
            RoadZone(
                name="auto_motion_zone",
                polygon=polygon,
                zone_type="auto_motion_road",
                expected_flow=expected_flow,
            ),
        ),
        notes=(
            f"Auto-inferred from {motion_pair_count} stable motion pairs; "
            f"scene_cuts={scene_cut_count}; confidence={confidence}; {reason}."
        ),
    )
    return AutoSceneCardResult(
        scene_card=card,
        confidence=confidence,
        reason=reason,
        frame_count=frame_count,
        motion_pair_count=motion_pair_count,
        scene_cut_count=scene_cut_count,
        zone_area_ratio=zone_area_ratio,
        flow_dominance=flow_dominance,
    )
