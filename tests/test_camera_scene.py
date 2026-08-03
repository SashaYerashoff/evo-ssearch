import cv2
import numpy as np

from camera_scene import (
    PtzSceneTracker,
    classify_camera_motion,
    fingerprint_similarity,
    scene_fingerprint,
)


def _view(offset: int = 0) -> np.ndarray:
    frame = np.zeros((96, 128, 3), dtype=np.uint8)
    cv2.rectangle(frame, (12 + offset, 20), (76 + offset, 68), (210, 210, 210), -1)
    cv2.line(frame, (0, 80), (127, 80), (100, 100, 100), 3)
    return frame


def test_scene_fingerprint_matches_same_view_and_rejects_other_view():
    first = scene_fingerprint(_view())
    same = scene_fingerprint(np.clip(_view().astype(np.int16) + 15, 0, 255).astype(np.uint8))
    other = scene_fingerprint(np.rot90(_view()).copy())

    assert fingerprint_similarity(first, same) > 0.98
    assert fingerprint_similarity(first, other) < 0.80


def test_camera_motion_classifies_pan_tilt_zoom_and_cut():
    assert classify_camera_motion([{"scene_cut": True}]) == "preset_cut"
    assert classify_camera_motion([
        {"global_motion": 2.0, "global_dx": 2.0, "global_dy": 0.2, "global_motion_coherence": 0.9}
    ]) == "pan"
    assert classify_camera_motion([
        {"global_motion": 2.0, "global_dx": 0.1, "global_dy": -2.0, "global_motion_coherence": 0.9}
    ]) == "tilt"
    assert classify_camera_motion([
        {"global_motion": 0.1, "global_zoom": 1.2, "global_zoom_coherence": 0.9}
    ]) == "zoom"


def test_ptz_tracker_freezes_spatial_probes_until_view_is_confirmed():
    tracker = PtzSceneTracker(112)
    fingerprint = scene_fingerprint(_view())

    first = tracker.observe([], fingerprint, timestamp_ms=1_000)
    second = tracker.observe([], fingerprint, timestamp_ms=2_000)
    moving = tracker.observe(
        [{"global_motion": 2.0, "global_dx": 2.0, "global_motion_coherence": 0.9}],
        fingerprint,
        timestamp_ms=3_000,
    )
    settling = tracker.observe([], fingerprint, timestamp_ms=4_000)
    recovered = tracker.observe([], fingerprint, timestamp_ms=5_000)

    assert first["preset_status"] == "provisional"
    assert first["spatial_probes_enabled"] is False
    assert second["preset_status"] == "known"
    assert second["spatial_probes_enabled"] is True
    assert moving["camera_motion"] == "pan"
    assert moving["coverage_status"] == "camera_moving"
    assert moving["spatial_probes_enabled"] is False
    assert settling["camera_motion"] == "settling"
    assert recovered["preset_id"] == second["preset_id"]
    assert recovered["spatial_probes_enabled"] is True
    assert recovered["scene_epoch"] == 1


def test_ptz_tracker_state_round_trip_preserves_known_views():
    tracker = PtzSceneTracker(7)
    fingerprint = scene_fingerprint(_view())
    tracker.observe([], fingerprint, timestamp_ms=1_000)
    tracker.observe([], fingerprint, timestamp_ms=2_000)

    restored = PtzSceneTracker(7, state=tracker.snapshot())
    observation = restored.observe([], fingerprint, timestamp_ms=3_000)

    assert observation["preset_status"] == "known"
    assert observation["known_preset_count"] == 1
    assert observation["spatial_probes_enabled"] is True
