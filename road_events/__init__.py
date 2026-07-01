"""Road-event CV primitives for traffic-focused EVA workflows.

This package is intentionally independent from the live Luxriot runtime.  It
turns decoded frames plus a scene card into bounded motion cues, then groups
those cues into candidate episodes that VLM/probes can confirm.
"""

from .episode import (
    RoadEventCue,
    RoadEpisode,
    RoadEpisodeAggregator,
    RoadEpisodeAggregatorConfig,
    road_event_cue_from_clip,
)
from .auto_scene import AutoSceneCardConfig, AutoSceneCardResult, infer_scene_card_from_frames
from .calibration import (
    SceneCalibrationConfig,
    SceneCalibrationResult,
    calibrate_scene_card_from_results,
)
from .luxriot_source import (
    LuxriotLiveSegment,
    capture_luxriot_archive_mp4_segment,
    capture_luxriot_live_mp4_segment,
    iter_luxriot_archive_segment_frames,
    iter_luxriot_archive_snapshots,
    iter_luxriot_live_segment_frames,
)
from .motion import (
    MotionAnalyzerConfig,
    RoadMotionAnalyzer,
    RoadMotionCue,
    RoadMotionSample,
)
from .scene import RoadSceneCard, RoadZone, load_scene_cards
from .runner import RoadCvChannelConfig, RoadCvRunner, RoadCvStatus
from .video import DecodedVideoFrame, iter_video_frames

__all__ = [
    "DecodedVideoFrame",
    "AutoSceneCardConfig",
    "AutoSceneCardResult",
    "LuxriotLiveSegment",
    "MotionAnalyzerConfig",
    "RoadCvChannelConfig",
    "RoadCvRunner",
    "RoadCvStatus",
    "RoadEventCue",
    "RoadEpisode",
    "RoadEpisodeAggregator",
    "RoadEpisodeAggregatorConfig",
    "RoadMotionAnalyzer",
    "RoadMotionCue",
    "RoadMotionSample",
    "RoadSceneCard",
    "RoadZone",
    "SceneCalibrationConfig",
    "SceneCalibrationResult",
    "calibrate_scene_card_from_results",
    "capture_luxriot_archive_mp4_segment",
    "capture_luxriot_live_mp4_segment",
    "iter_luxriot_archive_segment_frames",
    "iter_luxriot_archive_snapshots",
    "iter_luxriot_live_segment_frames",
    "infer_scene_card_from_frames",
    "iter_video_frames",
    "load_scene_cards",
    "road_event_cue_from_clip",
]
