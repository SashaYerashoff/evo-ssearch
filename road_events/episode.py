from __future__ import annotations

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
    apex_timestamp_ms: int | None = None
    apex_frame: int | None = None
    status: str = "candidate"


@dataclass(frozen=True)
class RoadEpisodeAggregatorConfig:
    window_ms: int = 90_000
    close_after_ms: int = 45_000
    max_inter_cue_gap_ms: int = 20_000
    min_cues_for_medium: int = 2
    min_sources_for_high: int = 2
    min_cues_for_high: int = 3
    max_recent_cues: int = 2000
    max_recent_episodes: int = 200


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


@dataclass
class _EpisodeState:
    channel_id: int
    episode_id: str
    zone_name: str
    event_type: str
    start_ms: int
    end_ms: int
    cues: list[RoadEventCue] = field(default_factory=list)

    def append(self, cue: RoadEventCue, max_recent_cues: int) -> None:
        self.cues.append(cue)
        if len(self.cues) > max_recent_cues:
            self.cues = self.cues[-max_recent_cues:]
        self.start_ms = min(self.start_ms, cue.timestamp_ms)
        self.end_ms = max(self.end_ms, cue.timestamp_ms)


class RoadEpisodeAggregator:
    """Groups road CV/CLIP/VLM cues into bounded candidate episodes."""

    def __init__(self, config: RoadEpisodeAggregatorConfig | None = None):
        self.config = config or RoadEpisodeAggregatorConfig()
        self._active: dict[tuple[int, str, str], _EpisodeState] = {}
        self._closed: list[RoadEpisode] = []
        self._next_sequence_by_channel: dict[int, int] = {}

    def reset(self) -> None:
        self._active.clear()
        self._closed.clear()

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
        added = sorted([cue for cue in cues], key=lambda item: item.timestamp_ms)
        if not added:
            return self.current_episodes()
        for cue in added:
            self._add_one(cue)
        latest_ms = max(cue.timestamp_ms for cue in added)
        self._close_stale(latest_ms)
        return self.current_episodes(now_ms=latest_ms)

    def _next_episode_id(self, channel_id: int) -> str:
        next_value = int(self._next_sequence_by_channel.get(channel_id, 0)) + 1
        self._next_sequence_by_channel[channel_id] = next_value
        return f"road-{channel_id}-{next_value:06d}"

    def _add_one(self, cue: RoadEventCue) -> None:
        family = _episode_family(cue.cue_type)
        key = (cue.channel_id, cue.zone_name, family)
        state = self._active.get(key)
        if state is not None and cue.timestamp_ms - state.end_ms > self.config.max_inter_cue_gap_ms:
            self._closed.append(self._state_to_episode(state, now_ms=cue.timestamp_ms, closed=True))
            self._active.pop(key, None)
            state = None
        if state is None:
            state = _EpisodeState(
                channel_id=cue.channel_id,
                episode_id=self._next_episode_id(cue.channel_id),
                zone_name=cue.zone_name,
                event_type=family,
                start_ms=cue.timestamp_ms,
                end_ms=cue.timestamp_ms,
            )
            self._active[key] = state
        state.append(cue, max_recent_cues=max(1, int(self.config.max_recent_cues)))
        if len(self._closed) > self.config.max_recent_episodes:
            self._closed = self._closed[-self.config.max_recent_episodes :]

    def _close_stale(self, now_ms: int) -> None:
        for key, state in list(self._active.items()):
            if now_ms - state.end_ms <= self.config.close_after_ms:
                continue
            self._closed.append(self._state_to_episode(state, now_ms=now_ms, closed=True))
            self._active.pop(key, None)
        if len(self._closed) > self.config.max_recent_episodes:
            self._closed = self._closed[-self.config.max_recent_episodes :]

    def _state_to_episode(self, state: _EpisodeState, *, now_ms: int, closed: bool = False) -> RoadEpisode:
        cues = sorted(state.cues, key=lambda item: item.timestamp_ms)
        confidence = _confidence(cues, self.config)
        score = max((cue.score for cue in cues), default=0.0)
        apex_cue = max(cues, key=lambda item: item.score, default=None)
        apex_frame = None
        if apex_cue is not None and isinstance(apex_cue.evidence, dict):
            raw_frame = apex_cue.evidence.get("frame_index") or apex_cue.evidence.get("apex_frame")
            try:
                apex_frame = int(raw_frame)
            except Exception:
                apex_frame = None
        active = (not closed) and now_ms - state.end_ms <= self.config.close_after_ms
        return RoadEpisode(
            channel_id=state.channel_id,
            episode_id=state.episode_id,
            start_ms=state.start_ms,
            end_ms=state.end_ms,
            zone_name=state.zone_name,
            event_type=state.event_type,
            confidence=confidence,
            score=score,
            cues=tuple(cues),
            evidence_timestamps=tuple(sorted({cue.timestamp_ms for cue in cues})),
            apex_timestamp_ms=apex_cue.timestamp_ms if apex_cue is not None else None,
            apex_frame=apex_frame,
            status="active" if active else "closed",
        )

    def current_episodes(self, now_ms: int | None = None) -> tuple[RoadEpisode, ...]:
        if not self._active and not self._closed:
            return ()
        latest_active = max((state.end_ms for state in self._active.values()), default=0)
        latest_closed = max((episode.end_ms for episode in self._closed), default=0)
        effective_now = int(now_ms if now_ms is not None else max(latest_active, latest_closed))
        self._close_stale(effective_now)
        floor = effective_now - max(self.config.window_ms, self.config.close_after_ms)
        episodes: list[RoadEpisode] = []
        episodes.extend(episode for episode in self._closed if episode.end_ms >= floor)
        episodes.extend(
            self._state_to_episode(state, now_ms=effective_now)
            for state in self._active.values()
            if state.end_ms >= floor
        )
        episodes.sort(key=lambda item: (item.start_ms, item.event_type, item.zone_name, item.episode_id))
        if len(episodes) > self.config.max_recent_episodes:
            episodes = episodes[-self.config.max_recent_episodes :]
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
