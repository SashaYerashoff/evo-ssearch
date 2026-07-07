from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable

from .motion import RoadMotionCue, RoadMotionSample


@dataclass(frozen=True)
class RoadEventCue:
    source: str
    cue_type: str
    timestamp_ms: int
    channel_id: int
    zone_name: str = ""
    score: float = 0.0
    label: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoadEpisode:
    channel_id: int
    episode_id: str
    start_ms: int
    end_ms: int
    zone_name: str
    event_type: str
    confidence: str
    score: float
    cues: tuple[RoadEventCue, ...] = field(default_factory=tuple)
    evidence_timestamps: tuple[int, ...] = field(default_factory=tuple)
    status: str = "candidate"


@dataclass(frozen=True)
class RoadEpisodeAggregatorConfig:
    window_ms: int = 90_000
    close_after_ms: int = 45_000
    min_cues_for_medium: int = 2
    min_sources_for_high: int = 2
    min_cues_for_high: int = 3
    max_recent_cues: int = 2000


_CUE_EVENT_MAP = {
    "opposing_flow_candidate": "wrong_way_candidate",
    "cross_flow_candidate": "aggressive_vehicle_motion_candidate",
    "road_motion_burst": "aggressive_vehicle_motion_candidate",
    "clip_vehicle_drift": "drift_burnout_candidate",
    "clip_tire_smoke": "drift_burnout_candidate",
    "clip_burnout": "drift_burnout_candidate",
    "vlm_vehicle_drift": "drift_burnout_candidate",
    "vlm_aggressive_driving": "aggressive_vehicle_motion_candidate",
}


def _episode_family(cue_type: str) -> str:
    return _CUE_EVENT_MAP.get(cue_type, cue_type)


def _confidence(cues: Iterable[RoadEventCue], config: RoadEpisodeAggregatorConfig) -> str:
    cue_list = list(cues)
    sources = {cue.source for cue in cue_list if cue.source}
    best_score = max((cue.score for cue in cue_list), default=0.0)
    if len(cue_list) >= config.min_cues_for_high and len(sources) >= config.min_sources_for_high:
        return "high"
    if len(cue_list) >= 2 and len(sources) >= config.min_sources_for_high and best_score >= 0.75:
        return "high"
    if len(cue_list) >= config.min_cues_for_medium:
        return "medium"
    return "low"


def _episode_id(channel_id: int, zone_name: str, event_type: str, start_ms: int) -> str:
    digest = hashlib.sha1(f"{channel_id}:{zone_name}:{event_type}:{start_ms}".encode("utf-8")).hexdigest()
    return f"road-{digest[:12]}"


class RoadEpisodeAggregator:
    """Groups road CV/CLIP/VLM cues into bounded candidate episodes."""

    def __init__(self, config: RoadEpisodeAggregatorConfig | None = None):
        self.config = config or RoadEpisodeAggregatorConfig()
        self._recent: list[RoadEventCue] = []

    def reset(self) -> None:
        self._recent.clear()

    def add_motion_sample(self, sample: RoadMotionSample) -> tuple[RoadEpisode, ...]:
        return self.add_cues(
            RoadEventCue(
                source="cv_motion",
                cue_type=cue.cue_type,
                timestamp_ms=cue.timestamp_ms,
                channel_id=cue.channel_id,
                zone_name=cue.zone_name,
                score=cue.score,
                label=cue.evidence,
                evidence={
                    "frame_index": cue.frame_index,
                    "metrics": dict(cue.metrics),
                },
            )
            for cue in sample.cues
        )

    def add_cue(self, cue: RoadEventCue) -> tuple[RoadEpisode, ...]:
        return self.add_cues((cue,))

    def add_cues(self, cues: Iterable[RoadEventCue]) -> tuple[RoadEpisode, ...]:
        added = [cue for cue in cues]
        if not added:
            return self.current_episodes()
        self._recent.extend(added)
        latest_ms = max(cue.timestamp_ms for cue in added)
        floor = latest_ms - max(self.config.window_ms, self.config.close_after_ms)
        self._recent = [cue for cue in self._recent if cue.timestamp_ms >= floor]
        if len(self._recent) > self.config.max_recent_cues:
            self._recent = self._recent[-self.config.max_recent_cues :]
        return self.current_episodes(now_ms=latest_ms)

    def current_episodes(self, now_ms: int | None = None) -> tuple[RoadEpisode, ...]:
        if not self._recent:
            return ()
        effective_now = int(now_ms if now_ms is not None else max(cue.timestamp_ms for cue in self._recent))
        groups: dict[tuple[int, str, str], list[RoadEventCue]] = {}
        for cue in self._recent:
            if cue.timestamp_ms < effective_now - self.config.window_ms:
                continue
            key = (cue.channel_id, cue.zone_name, _episode_family(cue.cue_type))
            groups.setdefault(key, []).append(cue)
        episodes: list[RoadEpisode] = []
        for (channel_id, zone_name, event_type), cues in groups.items():
            cues.sort(key=lambda item: item.timestamp_ms)
            start_ms = cues[0].timestamp_ms
            end_ms = cues[-1].timestamp_ms
            confidence = _confidence(cues, self.config)
            status = "active" if effective_now - end_ms <= self.config.close_after_ms else "candidate"
            score = max((cue.score for cue in cues), default=0.0)
            episodes.append(
                RoadEpisode(
                    channel_id=channel_id,
                    episode_id=_episode_id(channel_id, zone_name, event_type, start_ms),
                    start_ms=start_ms,
                    end_ms=end_ms,
                    zone_name=zone_name,
                    event_type=event_type,
                    confidence=confidence,
                    score=score,
                    cues=tuple(cues),
                    evidence_timestamps=tuple(sorted({cue.timestamp_ms for cue in cues})),
                    status=status,
                )
            )
        episodes.sort(key=lambda item: (item.start_ms, item.event_type, item.zone_name))
        return tuple(episodes)


def road_event_cue_from_clip(
    *,
    channel_id: int,
    timestamp_ms: int,
    cue_type: str,
    score: float,
    zone_name: str = "",
    label: str = "",
    evidence: dict[str, Any] | None = None,
) -> RoadEventCue:
    return RoadEventCue(
        source="clip_probe",
        cue_type=cue_type,
        timestamp_ms=int(timestamp_ms),
        channel_id=int(channel_id),
        zone_name=zone_name,
        score=float(score),
        label=label,
        evidence=dict(evidence or {}),
    )
