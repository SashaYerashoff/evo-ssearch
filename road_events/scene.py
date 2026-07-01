from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

Point = tuple[float, float]
Vector = tuple[float, float]


def _coerce_point(value: Sequence[Any]) -> Point:
    if len(value) != 2:
        raise ValueError("scene-card polygon points must be [x, y]")
    x = float(value[0])
    y = float(value[1])
    return (min(1.0, max(0.0, x)), min(1.0, max(0.0, y)))


def _coerce_vector(value: Sequence[Any] | None) -> Vector | None:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError("expected_flow must be [dx, dy]")
    dx = float(value[0])
    dy = float(value[1])
    magnitude = (dx * dx + dy * dy) ** 0.5
    if magnitude <= 1e-9:
        return None
    return (dx / magnitude, dy / magnitude)


@dataclass(frozen=True)
class RoadZone:
    """A normalized polygon where road-motion semantics should be measured.

    ``expected_flow`` is optional and normalized.  It describes the legal/usual
    vehicle direction in image coordinates, e.g. ``[1, 0]`` for left-to-right.
    """

    name: str
    polygon: tuple[Point, ...] = field(default_factory=tuple)
    zone_type: str = "road"
    expected_flow: Vector | None = None
    enabled: bool = True

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RoadZone":
        name = str(payload.get("name") or "").strip()
        if not name:
            raise ValueError("road zone requires a name")
        polygon_raw = payload.get("polygon") or payload.get("points") or []
        polygon = tuple(_coerce_point(point) for point in polygon_raw)
        return cls(
            name=name,
            polygon=polygon,
            zone_type=str(payload.get("zone_type") or payload.get("type") or "road").strip() or "road",
            expected_flow=_coerce_vector(payload.get("expected_flow")),
            enabled=payload.get("enabled", True) is not False,
        )


@dataclass(frozen=True)
class RoadSceneCard:
    """Operator-maintained traffic semantics for one camera channel."""

    channel_id: int
    title: str = ""
    zones: tuple[RoadZone, ...] = field(default_factory=tuple)
    notes: str = ""
    version: int = 1

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RoadSceneCard":
        channel_id = int(payload.get("channel_id"))
        zones = tuple(RoadZone.from_mapping(item) for item in payload.get("zones", []) or [])
        return cls(
            channel_id=channel_id,
            title=str(payload.get("title") or "").strip(),
            zones=zones,
            notes=str(payload.get("notes") or "").strip(),
            version=int(payload.get("version") or 1),
        )

    def active_zones(self) -> tuple[RoadZone, ...]:
        return tuple(zone for zone in self.zones if zone.enabled)


def load_scene_cards(path: str | Path) -> dict[int, RoadSceneCard]:
    """Load JSON scene cards keyed by channel id.

    Supported JSON shapes:
    - ``[{...card...}, {...card...}]``
    - ``{"channels": [{...card...}]}``
    - ``{"112": {...card without channel_id...}}``
    """

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    items: Iterable[Any]
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict) and isinstance(payload.get("channels"), list):
        items = payload["channels"]
    elif isinstance(payload, dict):
        expanded: list[dict[str, Any]] = []
        for key, value in payload.items():
            if not isinstance(value, dict):
                continue
            item = dict(value)
            item.setdefault("channel_id", int(key))
            expanded.append(item)
        items = expanded
    else:
        raise ValueError("scene-card file must be a JSON list or object")
    cards = [RoadSceneCard.from_mapping(item) for item in items]
    return {card.channel_id: card for card in cards}


def zone_mask(width: int, height: int, zone: RoadZone) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if not zone.polygon:
        mask[:, :] = 255
        return mask
    points = np.array(
        [[int(round(x * (width - 1))), int(round(y * (height - 1)))] for x, y in zone.polygon],
        dtype=np.int32,
    )
    if points.shape[0] < 3:
        mask[:, :] = 255
        return mask
    cv2.fillPoly(mask, [points], 255)
    return mask
