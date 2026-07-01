from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .auto_scene import AutoSceneCardResult
from .scene import RoadSceneCard, RoadZone


@dataclass(frozen=True)
class SceneCalibrationConfig:
    min_zone_samples: int = 4
    min_flow_samples: int = 3
    zone_iou_floor: float = 0.45
    flow_agreement_floor: float = 0.72
    high_confidence_zone_agreement: float = 0.75


@dataclass(frozen=True)
class SceneCalibrationResult:
    scene_card: RoadSceneCard
    confidence: str
    reason: str
    sample_count: int
    usable_zone_samples: int
    usable_flow_samples: int
    zone_agreement: float
    flow_agreement: float

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
            "sample_count": self.sample_count,
            "usable_zone_samples": self.usable_zone_samples,
            "usable_flow_samples": self.usable_flow_samples,
            "zone_agreement": round(float(self.zone_agreement), 4),
            "flow_agreement": round(float(self.flow_agreement), 4),
        }


def _bbox_from_zone(zone: RoadZone) -> tuple[float, float, float, float] | None:
    if not zone.polygon:
        return None
    xs = [float(point[0]) for point in zone.polygon]
    ys = [float(point[1]) for point in zone.polygon]
    x0 = max(0.0, min(xs))
    y0 = max(0.0, min(ys))
    x1 = min(1.0, max(xs))
    y1 = min(1.0, max(ys))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return float(intersection / max(1e-9, area_a + area_b - intersection))


def _degraded_result(
    channel_id: int,
    title: str,
    *,
    reason: str,
    sample_count: int,
    usable_zone_samples: int,
    usable_flow_samples: int,
    zone_agreement: float = 0.0,
    flow_agreement: float = 0.0,
) -> SceneCalibrationResult:
    card = RoadSceneCard(
        channel_id=int(channel_id),
        title=title,
        zones=(
            RoadZone(
                name="calibrated_full_frame_degraded",
                polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
                zone_type="degraded_full_frame",
                expected_flow=None,
            ),
        ),
        notes=f"Scene calibration degraded: {reason}. Wrong-way semantics disabled.",
    )
    return SceneCalibrationResult(
        scene_card=card,
        confidence="low",
        reason=reason,
        sample_count=sample_count,
        usable_zone_samples=usable_zone_samples,
        usable_flow_samples=usable_flow_samples,
        zone_agreement=zone_agreement,
        flow_agreement=flow_agreement,
    )


def calibrate_scene_card_from_results(
    channel_id: int,
    title: str,
    samples: Iterable[AutoSceneCardResult],
    *,
    config: SceneCalibrationConfig | None = None,
) -> SceneCalibrationResult:
    """Aggregate many short auto-scene samples into a stable client-stream card."""

    cfg = config or SceneCalibrationConfig()
    sample_list = list(samples)
    zone_boxes: list[tuple[float, float, float, float]] = []
    flow_vectors: list[tuple[float, float]] = []
    for sample in sample_list:
        zones = sample.scene_card.active_zones()
        if not zones:
            continue
        zone = zones[0]
        if zone.zone_type == "degraded_full_frame":
            continue
        bbox = _bbox_from_zone(zone)
        if bbox is None:
            continue
        zone_boxes.append(bbox)
        if zone.expected_flow is not None and sample.confidence in {"medium", "high"}:
            flow_vectors.append(zone.expected_flow)

    sample_count = len(sample_list)
    usable_zone_samples = len(zone_boxes)
    usable_flow_samples = len(flow_vectors)
    if usable_zone_samples < int(cfg.min_zone_samples):
        return _degraded_result(
            channel_id,
            title,
            reason=f"insufficient usable motion-zone samples ({usable_zone_samples}/{sample_count})",
            sample_count=sample_count,
            usable_zone_samples=usable_zone_samples,
            usable_flow_samples=usable_flow_samples,
        )

    boxes = np.array(zone_boxes, dtype=np.float64)
    median_box = tuple(float(value) for value in np.median(boxes, axis=0))
    zone_agreement = float(
        sum(1 for box in zone_boxes if _bbox_iou(box, median_box) >= float(cfg.zone_iou_floor))
        / max(1, usable_zone_samples)
    )
    if zone_agreement < 0.5:
        return _degraded_result(
            channel_id,
            title,
            reason=f"unstable motion-zone geometry (agreement={zone_agreement:.2f})",
            sample_count=sample_count,
            usable_zone_samples=usable_zone_samples,
            usable_flow_samples=usable_flow_samples,
            zone_agreement=zone_agreement,
        )

    expected_flow = None
    flow_agreement = 0.0
    if usable_flow_samples >= int(cfg.min_flow_samples):
        flow = np.array(flow_vectors, dtype=np.float64)
        mean_flow = np.mean(flow, axis=0)
        magnitude = float(np.linalg.norm(mean_flow))
        if magnitude > 1e-9:
            normalized = mean_flow / magnitude
            dots = np.clip(flow @ normalized, -1.0, 1.0)
            flow_agreement = float(np.mean(dots >= float(cfg.flow_agreement_floor)))
            if flow_agreement >= float(cfg.flow_agreement_floor):
                expected_flow = (float(normalized[0]), float(normalized[1]))

    confidence = "high" if expected_flow is not None and zone_agreement >= float(cfg.high_confidence_zone_agreement) else "medium"
    reason = (
        "stable motion zone and dominant flow calibrated"
        if expected_flow is not None
        else "stable motion zone calibrated without reliable dominant flow; wrong-way semantics disabled"
    )
    x0, y0, x1, y1 = median_box
    polygon = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    card = RoadSceneCard(
        channel_id=int(channel_id),
        title=title,
        zones=(
            RoadZone(
                name="calibrated_motion_zone",
                polygon=polygon,
                zone_type="calibrated_motion_road",
                expected_flow=expected_flow,
            ),
        ),
        notes=(
            f"Calibrated from {usable_zone_samples}/{sample_count} usable samples; "
            f"zone_agreement={zone_agreement:.2f}; flow_agreement={flow_agreement:.2f}; "
            f"confidence={confidence}; {reason}."
        ),
    )
    return SceneCalibrationResult(
        scene_card=card,
        confidence=confidence,
        reason=reason,
        sample_count=sample_count,
        usable_zone_samples=usable_zone_samples,
        usable_flow_samples=usable_flow_samples,
        zone_agreement=zone_agreement,
        flow_agreement=flow_agreement,
    )
