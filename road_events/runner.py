from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .episode import RoadEpisode, RoadEpisodeAggregator, RoadEpisodeAggregatorConfig
from .motion import MotionAnalyzerConfig, RoadMotionAnalyzer, RoadMotionCue, RoadMotionSample
from .scene import RoadSceneCard
from .video import DecodedVideoFrame, iter_video_frames

FrameIteratorFactory = Callable[..., Iterable[DecodedVideoFrame]]


@dataclass(frozen=True)
class RoadCvChannelConfig:
    channel_id: int
    source: str | int
    scene_card: RoadSceneCard
    every_n: int = 1
    wallclock_timestamps: bool = False
    motion_config: MotionAnalyzerConfig = field(default_factory=MotionAnalyzerConfig)
    episode_config: RoadEpisodeAggregatorConfig = field(default_factory=RoadEpisodeAggregatorConfig)


@dataclass(frozen=True)
class RoadCvStatus:
    channel_id: int
    running: bool
    frame_count: int
    cue_count: int
    episode_count: int
    scene_cut_count: int
    last_timestamp_ms: int | None
    last_error: str | None
    recent_cues: tuple[dict, ...] = field(default_factory=tuple)
    recent_episodes: tuple[dict, ...] = field(default_factory=tuple)


def _cue_to_dict(cue: RoadMotionCue) -> dict:
    return {
        "channel_id": cue.channel_id,
        "timestamp_ms": cue.timestamp_ms,
        "zone_name": cue.zone_name,
        "cue_type": cue.cue_type,
        "score": round(float(cue.score), 4),
        "evidence": cue.evidence,
        "frame_index": cue.frame_index,
        "metrics": {key: round(float(value), 4) for key, value in cue.metrics.items()},
    }


def _episode_to_dict(episode: RoadEpisode) -> dict:
    return {
        "channel_id": episode.channel_id,
        "episode_id": episode.episode_id,
        "start_ms": episode.start_ms,
        "end_ms": episode.end_ms,
        "zone_name": episode.zone_name,
        "event_type": episode.event_type,
        "confidence": episode.confidence,
        "score": round(float(episode.score), 4),
        "status": episode.status,
        "evidence_timestamps": list(episode.evidence_timestamps),
        "cue_count": len(episode.cues),
        "sources": sorted({cue.source for cue in episode.cues}),
    }


class RoadCvRunner:
    """Owns one road-CV stream and a bounded status snapshot."""

    def __init__(
        self,
        config: RoadCvChannelConfig,
        *,
        frame_iter_factory: FrameIteratorFactory = iter_video_frames,
        max_recent_cues: int = 40,
        max_recent_episodes: int = 20,
    ):
        self.config = config
        self.frame_iter_factory = frame_iter_factory
        self.max_recent_cues = max(1, int(max_recent_cues))
        self.max_recent_episodes = max(1, int(max_recent_episodes))
        self.analyzer = RoadMotionAnalyzer(config.scene_card, config.motion_config)
        self.aggregator = RoadEpisodeAggregator(config.episode_config)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._frame_count = 0
        self._cue_count = 0
        self._scene_cut_count = 0
        self._last_timestamp_ms: int | None = None
        self._last_error: str | None = None
        self._recent_cues: list[dict] = []
        self._recent_episodes: list[dict] = []

    def start(self) -> RoadCvStatus:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return self.status()
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_background,
                name=f"road-cv-{self.config.channel_id}",
                daemon=True,
            )
            self._running = True
            self._thread.start()
            return self.status()

    def stop(self, timeout: float = 5.0) -> RoadCvStatus:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=max(0.1, float(timeout)))
        with self._lock:
            self._running = False
            return self.status()

    def run_once(self, *, max_frames: int | None = None) -> RoadCvStatus:
        with self._lock:
            self._running = True
            self._last_error = None
        try:
            self._consume(max_frames=max_frames)
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
        finally:
            with self._lock:
                self._running = False
        return self.status()

    def status(self) -> RoadCvStatus:
        with self._lock:
            return RoadCvStatus(
                channel_id=self.config.channel_id,
                running=self._running,
                frame_count=self._frame_count,
                cue_count=self._cue_count,
                episode_count=len(self._recent_episodes),
                scene_cut_count=self._scene_cut_count,
                last_timestamp_ms=self._last_timestamp_ms,
                last_error=self._last_error,
                recent_cues=tuple(self._recent_cues[-self.max_recent_cues :]),
                recent_episodes=tuple(self._recent_episodes[-self.max_recent_episodes :]),
            )

    def status_dict(self) -> dict:
        status = self.status()
        return {
            "channel_id": status.channel_id,
            "running": status.running,
            "frame_count": status.frame_count,
            "cue_count": status.cue_count,
            "episode_count": status.episode_count,
            "scene_cut_count": status.scene_cut_count,
            "last_timestamp_ms": status.last_timestamp_ms,
            "last_error": status.last_error,
            "recent_cues": list(status.recent_cues),
            "recent_episodes": list(status.recent_episodes),
        }

    def _run_background(self) -> None:
        try:
            self._consume(max_frames=None)
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
        finally:
            with self._lock:
                self._running = False

    def _consume(self, *, max_frames: int | None) -> None:
        iterator = self.frame_iter_factory(
            self.config.source,
            every_n=self.config.every_n,
            max_frames=max_frames,
            wallclock_timestamps=self.config.wallclock_timestamps,
        )
        for frame in iterator:
            if self._stop_event.is_set():
                break
            sample = self.analyzer.analyze_frame(
                frame.image,
                timestamp_ms=frame.timestamp_ms,
                frame_index=frame.frame_index,
            )
            episodes = self.aggregator.add_motion_sample(sample)
            self._record(sample, episodes)
            if max_frames is None:
                time.sleep(0)

    def _record(self, sample: RoadMotionSample, episodes: tuple[RoadEpisode, ...]) -> None:
        with self._lock:
            self._frame_count += 1
            self._last_timestamp_ms = sample.timestamp_ms
            if sample.scene_cut:
                self._scene_cut_count += 1
            if sample.cues:
                self._cue_count += len(sample.cues)
                self._recent_cues.extend(_cue_to_dict(cue) for cue in sample.cues)
                self._recent_cues = self._recent_cues[-self.max_recent_cues :]
            if episodes:
                by_id = {item["episode_id"]: item for item in self._recent_episodes}
                for episode in episodes:
                    by_id[episode.episode_id] = _episode_to_dict(episode)
                ordered = sorted(by_id.values(), key=lambda item: (item["start_ms"], item["episode_id"]))
                self._recent_episodes = ordered[-self.max_recent_episodes :]
