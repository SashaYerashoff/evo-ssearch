import json
from pathlib import Path

import numpy as np

from road_events import (
    AutoSceneCardConfig,
    DecodedVideoFrame,
    MotionAnalyzerConfig,
    RoadCvChannelConfig,
    RoadCvRunner,
    RoadEventCue,
    RoadEpisodeAggregator,
    RoadEpisodeAggregatorConfig,
    RoadMotionAnalyzer,
    RoadSceneCard,
    RoadZone,
    calibrate_scene_card_from_results,
    infer_scene_card_from_frames,
    iter_video_frames,
    load_scene_cards,
)
from road_events.auto_scene import AutoSceneCardResult


def _frame_with_vehicle(x: int, y: int = 46, *, width: int = 120, height: int = 80) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[y : y + 12, x : x + 20, :] = 255
    return frame


def _scene(expected_flow=(1.0, 0.0)) -> RoadSceneCard:
    return RoadSceneCard(
        channel_id=7,
        title="test road",
        zones=(
            RoadZone(
                name="main_lane",
                polygon=((0.0, 0.35), (1.0, 0.35), (1.0, 0.9), (0.0, 0.9)),
                expected_flow=expected_flow,
            ),
        ),
    )


def _analyzer() -> RoadMotionAnalyzer:
    return RoadMotionAnalyzer(
        _scene(),
        MotionAnalyzerConfig(
            max_edge=160,
            min_motion_px=0.25,
            active_ratio_floor=0.003,
            mean_motion_floor=0.05,
            p90_motion_floor=0.1,
            wrong_way_alignment=-0.35,
            compensate_global_motion=True,
        ),
    )


def test_motion_analyzer_warms_up_on_first_frame():
    analyzer = _analyzer()

    sample = analyzer.analyze_frame(_frame_with_vehicle(20), timestamp_ms=1000, frame_index=1)

    assert sample.warmup is True
    assert sample.cues == ()


def test_global_motion_exposes_translation_coherence_and_radial_zoom():
    analyzer = _analyzer()
    pan_flow = np.zeros((40, 60, 2), dtype=np.float32)
    pan_flow[:, :, 0] = 2.0
    pan = analyzer._global_motion(pan_flow)

    yy, xx = np.mgrid[0:40, 0:60].astype(np.float32)
    xx -= 29.5
    yy -= 19.5
    radius = np.maximum(np.sqrt(xx * xx + yy * yy), 1.0)
    zoom_flow = np.stack((xx / radius, yy / radius), axis=2) * 1.5
    zoom = analyzer._global_motion(zoom_flow.astype(np.float32))

    assert pan["magnitude"] > 1.9
    assert pan["coherence"] > 0.95
    assert abs(zoom["zoom"]) > 1.3
    assert zoom["zoom_coherence"] > 0.9


def test_motion_analyzer_emits_burst_for_vehicle_motion_in_road_zone():
    analyzer = _analyzer()
    analyzer.analyze_frame(_frame_with_vehicle(20), timestamp_ms=1000, frame_index=1)

    sample = analyzer.analyze_frame(_frame_with_vehicle(32), timestamp_ms=2000, frame_index=2)

    assert any(cue.cue_type == "road_motion_burst" for cue in sample.cues)
    assert sample.zone_metrics["main_lane"]["active_ratio"] > 0
    assert sample.zone_metrics["main_lane"]["alignment"] > 0.5


def test_motion_analyzer_flags_opposing_flow_against_scene_card():
    analyzer = _analyzer()
    analyzer.analyze_frame(_frame_with_vehicle(52), timestamp_ms=1000, frame_index=1)

    sample = analyzer.analyze_frame(_frame_with_vehicle(36), timestamp_ms=2000, frame_index=2)

    assert any(cue.cue_type == "opposing_flow_candidate" for cue in sample.cues)
    assert sample.zone_metrics["main_lane"]["alignment"] < -0.35


def test_motion_analyzer_does_not_emit_directional_cues_without_expected_flow():
    analyzer = RoadMotionAnalyzer(
        _scene(expected_flow=None),
        MotionAnalyzerConfig(
            max_edge=160,
            min_motion_px=0.25,
            active_ratio_floor=0.003,
            mean_motion_floor=0.05,
            p90_motion_floor=0.1,
            wrong_way_alignment=-0.35,
            compensate_global_motion=True,
        ),
    )
    analyzer.analyze_frame(_frame_with_vehicle(52), timestamp_ms=1000, frame_index=1)

    sample = analyzer.analyze_frame(_frame_with_vehicle(36), timestamp_ms=2000, frame_index=2)

    assert any(cue.cue_type == "road_motion_burst" for cue in sample.cues)
    assert not any(cue.cue_type == "opposing_flow_candidate" for cue in sample.cues)
    assert not any(cue.cue_type == "cross_flow_candidate" for cue in sample.cues)


def test_motion_analyzer_suppresses_motion_claims_at_low_fps_intervals():
    analyzer = _analyzer()
    analyzer.analyze_frame(_frame_with_vehicle(20), timestamp_ms=1000, frame_index=1)

    sample = analyzer.analyze_frame(_frame_with_vehicle(52), timestamp_ms=7000, frame_index=2)

    assert sample.quality["low_fps_suppressed"] == 1.0
    assert sample.zone_metrics["main_lane"]["frame_interval_ms"] == 6000
    assert sample.cues == ()


def test_motion_analyzer_does_not_flag_expected_direction_as_wrong_way():
    analyzer = _analyzer()
    analyzer.analyze_frame(_frame_with_vehicle(20), timestamp_ms=1000, frame_index=1)

    sample = analyzer.analyze_frame(_frame_with_vehicle(36), timestamp_ms=2000, frame_index=2)

    assert not any(cue.cue_type == "opposing_flow_candidate" for cue in sample.cues)


def test_motion_analyzer_resets_on_hard_scene_cut_without_motion_cue():
    analyzer = _analyzer()
    analyzer.analyze_frame(_frame_with_vehicle(20), timestamp_ms=1000, frame_index=1)
    cut_frame = np.full((80, 120, 3), 255, dtype=np.uint8)

    cut_sample = analyzer.analyze_frame(cut_frame, timestamp_ms=2000, frame_index=2)

    assert cut_sample.scene_cut is True
    assert cut_sample.warmup is True
    assert cut_sample.cues == ()
    assert cut_sample.global_motion["mean_absdiff"] > 0.18


def test_motion_analyzer_resumes_after_scene_cut_warmup():
    analyzer = _analyzer()
    analyzer.analyze_frame(_frame_with_vehicle(20), timestamp_ms=1000, frame_index=1)
    analyzer.analyze_frame(np.full((80, 120, 3), 255, dtype=np.uint8), timestamp_ms=2000, frame_index=2)

    post_cut_warmup = _frame_with_vehicle(20)
    resumed = _frame_with_vehicle(36)
    analyzer.analyze_frame(post_cut_warmup, timestamp_ms=3000, frame_index=3)
    sample = analyzer.analyze_frame(resumed, timestamp_ms=4000, frame_index=4)

    assert sample.scene_cut is False
    assert any(cue.cue_type == "road_motion_burst" for cue in sample.cues)


def test_scene_card_json_loader(tmp_path: Path):
    path = tmp_path / "scenes.json"
    path.write_text(
        json.dumps(
            {
                "channels": [
                    {
                        "channel_id": 112,
                        "title": "Avenue",
                        "zones": [
                            {
                                "name": "lane_a",
                                "polygon": [[0, 0.4], [1, 0.4], [1, 1], [0, 1]],
                                "expected_flow": [0, -1],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    cards = load_scene_cards(path)

    assert list(cards) == [112]
    assert cards[112].zones[0].expected_flow == (0.0, -1.0)


def test_auto_scene_inference_builds_motion_zone_and_flow():
    frames = [
        DecodedVideoFrame(
            frame_index=idx,
            timestamp_ms=1000 + idx * 1000,
            image=_frame_with_vehicle(12 + idx * 8),
        )
        for idx in range(8)
    ]

    result = infer_scene_card_from_frames(
        7,
        "Synthetic road",
        frames,
        config=AutoSceneCardConfig(
            max_edge=120,
            min_frames=6,
            min_motion_pairs=2,
            min_motion_px=0.2,
            heat_threshold=0.005,
            flow_dominance_floor=0.2,
        ),
    )

    zone = result.scene_card.zones[0]
    assert result.confidence == "medium"
    assert zone.name == "auto_motion_zone"
    assert zone.zone_type == "auto_motion_road"
    assert zone.expected_flow is not None
    assert zone.expected_flow[0] > 0.5
    assert result.motion_pair_count >= 2
    assert result.zone_area_ratio < 1.0


def test_auto_scene_inference_degrades_when_motion_budget_is_too_small():
    frames = [
        DecodedVideoFrame(frame_index=0, timestamp_ms=1000, image=_frame_with_vehicle(12)),
        DecodedVideoFrame(frame_index=1, timestamp_ms=2000, image=_frame_with_vehicle(20)),
    ]

    result = infer_scene_card_from_frames(
        7,
        "Synthetic road",
        frames,
        config=AutoSceneCardConfig(max_edge=120, min_frames=6),
    )

    zone = result.scene_card.zones[0]
    assert result.confidence == "low"
    assert zone.zone_type == "degraded_full_frame"
    assert zone.expected_flow is None
    assert "insufficient frames" in result.reason


def test_auto_scene_low_confidence_wide_zone_disables_expected_flow():
    frames = [
        DecodedVideoFrame(
            frame_index=idx,
            timestamp_ms=1000 + idx * 1000,
            image=_frame_with_vehicle(2 + idx * 8, y=4, width=96, height=40),
        )
        for idx in range(10)
    ]

    result = infer_scene_card_from_frames(
        7,
        "Wide motion",
        frames,
        config=AutoSceneCardConfig(
            max_edge=96,
            min_frames=6,
            min_motion_pairs=2,
            min_motion_px=0.2,
            heat_threshold=0.001,
            flow_dominance_floor=0.2,
            padding_ratio=0.5,
        ),
    )

    assert result.confidence == "low"
    assert result.zone_area_ratio >= 0.75
    assert result.scene_card.zones[0].expected_flow is None


def _auto_sample(
    *,
    polygon=((0.1, 0.4), (0.9, 0.4), (0.9, 0.9), (0.1, 0.9)),
    expected_flow=(1.0, 0.0),
    confidence="medium",
) -> AutoSceneCardResult:
    card = RoadSceneCard(
        channel_id=7,
        title="calibration sample",
        zones=(
            RoadZone(
                name="auto_motion_zone",
                polygon=polygon,
                zone_type="auto_motion_road",
                expected_flow=expected_flow,
            ),
        ),
    )
    return AutoSceneCardResult(
        scene_card=card,
        confidence=confidence,
        reason="sample",
        frame_count=40,
        motion_pair_count=20,
        scene_cut_count=0,
        zone_area_ratio=0.4,
        flow_dominance=0.8,
    )


def test_calibrate_scene_card_aggregates_stable_zone_and_flow():
    samples = [
        _auto_sample(polygon=((0.1, 0.4), (0.9, 0.4), (0.9, 0.9), (0.1, 0.9)), expected_flow=(1.0, 0.0)),
        _auto_sample(polygon=((0.12, 0.42), (0.88, 0.42), (0.88, 0.92), (0.12, 0.92)), expected_flow=(0.98, 0.1)),
        _auto_sample(polygon=((0.09, 0.39), (0.91, 0.39), (0.91, 0.88), (0.09, 0.88)), expected_flow=(0.97, -0.05)),
        _auto_sample(polygon=((0.11, 0.41), (0.89, 0.41), (0.89, 0.91), (0.11, 0.91)), expected_flow=(1.0, 0.0)),
    ]

    result = calibrate_scene_card_from_results(7, "Avenue", samples)

    zone = result.scene_card.zones[0]
    assert result.confidence == "high"
    assert zone.name == "calibrated_motion_zone"
    assert zone.expected_flow is not None
    assert zone.expected_flow[0] > 0.99
    assert result.zone_agreement == 1.0
    assert result.flow_agreement == 1.0


def test_calibrate_scene_card_keeps_zone_but_disables_conflicting_flow():
    samples = [
        _auto_sample(expected_flow=(1.0, 0.0)),
        _auto_sample(expected_flow=(-1.0, 0.0)),
        _auto_sample(expected_flow=(0.0, 1.0)),
        _auto_sample(expected_flow=(0.0, -1.0)),
    ]

    result = calibrate_scene_card_from_results(7, "Avenue", samples)

    assert result.confidence == "medium"
    assert result.scene_card.zones[0].zone_type == "calibrated_motion_road"
    assert result.scene_card.zones[0].expected_flow is None
    assert "without reliable dominant flow" in result.reason


def test_calibrate_scene_card_degrades_on_insufficient_samples():
    result = calibrate_scene_card_from_results(7, "Avenue", [_auto_sample()])

    assert result.confidence == "low"
    assert result.scene_card.zones[0].zone_type == "degraded_full_frame"
    assert result.scene_card.zones[0].expected_flow is None


def test_episode_aggregator_promotes_repeated_motion_to_medium_candidate():
    aggregator = RoadEpisodeAggregator(
        RoadEpisodeAggregatorConfig(window_ms=60_000, min_cues_for_medium=2)
    )
    cue1 = RoadEventCue(
        source="cv_motion",
        cue_type="road_motion_burst",
        timestamp_ms=10_000,
        channel_id=7,
        zone_name="main_lane",
        score=0.4,
    )
    cue2 = RoadEventCue(
        source="cv_motion",
        cue_type="road_motion_burst",
        timestamp_ms=20_000,
        channel_id=7,
        zone_name="main_lane",
        score=0.5,
    )

    aggregator.add_cue(cue1)
    episodes = aggregator.add_cue(cue2)

    assert len(episodes) == 1
    assert episodes[0].event_type == "aggressive_vehicle_motion_candidate"
    assert episodes[0].confidence == "medium"
    assert episodes[0].evidence_timestamps == (10_000, 20_000)


def test_episode_aggregator_keeps_episode_id_as_window_slides():
    aggregator = RoadEpisodeAggregator(
        RoadEpisodeAggregatorConfig(
            window_ms=15_000,
            close_after_ms=30_000,
            max_inter_cue_gap_ms=20_000,
            min_cues_for_medium=2,
        )
    )
    cue1 = RoadEventCue(
        source="cv_motion",
        cue_type="road_motion_burst",
        timestamp_ms=10_000,
        channel_id=7,
        zone_name="main_lane",
        score=0.4,
    )
    cue2 = RoadEventCue(
        source="cv_motion",
        cue_type="road_motion_burst",
        timestamp_ms=20_000,
        channel_id=7,
        zone_name="main_lane",
        score=0.5,
    )
    cue3 = RoadEventCue(
        source="cv_motion",
        cue_type="road_motion_burst",
        timestamp_ms=30_000,
        channel_id=7,
        zone_name="main_lane",
        score=0.6,
    )

    first = aggregator.add_cues([cue1, cue2])[0]
    second = aggregator.add_cue(cue3)[0]

    assert second.episode_id == first.episode_id
    assert second.start_ms == 10_000
    assert second.end_ms == 30_000
    assert second.confidence == "medium"


def test_episode_aggregator_starts_new_episode_after_quiet_gap():
    aggregator = RoadEpisodeAggregator(
        RoadEpisodeAggregatorConfig(
            window_ms=90_000,
            close_after_ms=30_000,
            max_inter_cue_gap_ms=5_000,
        )
    )
    cue1 = RoadEventCue(
        source="cv_motion",
        cue_type="road_motion_burst",
        timestamp_ms=10_000,
        channel_id=7,
        zone_name="main_lane",
        score=0.4,
    )
    cue2 = RoadEventCue(
        source="cv_motion",
        cue_type="road_motion_burst",
        timestamp_ms=20_000,
        channel_id=7,
        zone_name="main_lane",
        score=0.5,
    )

    first_id = aggregator.add_cue(cue1)[0].episode_id
    episodes = aggregator.add_cue(cue2)

    assert len(episodes) == 2
    assert {episode.status for episode in episodes} == {"active", "closed"}
    assert any(episode.episode_id == first_id and episode.status == "closed" for episode in episodes)
    assert any(episode.episode_id != first_id and episode.status == "active" for episode in episodes)


def test_episode_aggregator_promotes_appearance_drift_without_motion_cue():
    aggregator = RoadEpisodeAggregator(
        RoadEpisodeAggregatorConfig(window_ms=90_000, min_sources_for_high=2)
    )

    episodes = aggregator.add_cues(
        [
            RoadEventCue(
                source="clip_probe",
                cue_type="clip_tire_smoke",
                timestamp_ms=10_000,
                channel_id=7,
                zone_name="intersection",
                score=0.7,
            ),
            RoadEventCue(
                source="vlm_alert",
                cue_type="vlm_vehicle_drift",
                timestamp_ms=12_000,
                channel_id=7,
                zone_name="intersection",
                score=0.8,
            ),
        ]
    )

    drift = [episode for episode in episodes if episode.event_type == "drift_burnout_candidate"][0]
    assert drift.confidence == "high"
    assert {cue.source for cue in drift.cues} == {"clip_probe", "vlm_alert"}


def test_episode_aggregator_promotes_multi_source_drift_to_high_candidate():
    aggregator = RoadEpisodeAggregator(
        RoadEpisodeAggregatorConfig(window_ms=90_000, min_cues_for_high=3, min_sources_for_high=2)
    )
    cues = [
        RoadEventCue(
            source="cv_motion",
            cue_type="road_motion_burst",
            timestamp_ms=10_000,
            channel_id=7,
            zone_name="intersection",
            score=0.6,
        ),
        RoadEventCue(
            source="clip_probe",
            cue_type="clip_tire_smoke",
            timestamp_ms=12_000,
            channel_id=7,
            zone_name="intersection",
            score=0.7,
        ),
        RoadEventCue(
            source="vlm_alert",
            cue_type="vlm_vehicle_drift",
            timestamp_ms=14_000,
            channel_id=7,
            zone_name="intersection",
            score=0.8,
        ),
    ]

    episodes = aggregator.add_cues(cues)

    assert len(episodes) == 2
    drift = [episode for episode in episodes if episode.event_type == "drift_burnout_candidate"][0]
    assert drift.confidence == "high"
    assert {cue.source for cue in drift.cues} == {"clip_probe", "vlm_alert"}


def test_iter_video_frames_decodes_file_with_opencv(tmp_path: Path):
    cv2 = __import__("cv2")
    path = str(tmp_path / "sample.avi")
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (64, 48),
    )
    try:
        for idx in range(4):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[:, idx * 5 : idx * 5 + 8, :] = 255
            writer.write(frame)
    finally:
        writer.release()

    frames = list(iter_video_frames(path, every_n=2, max_frames=2))

    assert len(frames) == 2
    assert frames[0].frame_index == 0
    assert frames[1].frame_index == 2
    assert frames[0].image.shape == (48, 64, 3)


def test_road_cv_runner_collects_bounded_status_from_injected_frames():
    frames = [
        DecodedVideoFrame(frame_index=0, timestamp_ms=1000, image=_frame_with_vehicle(20)),
        DecodedVideoFrame(frame_index=1, timestamp_ms=2000, image=_frame_with_vehicle(36)),
        DecodedVideoFrame(frame_index=2, timestamp_ms=3000, image=_frame_with_vehicle(52)),
    ]

    def frame_iter(_source, **_kwargs):
        return iter(frames)

    runner = RoadCvRunner(
        RoadCvChannelConfig(
            channel_id=7,
            source="synthetic",
            scene_card=_scene(),
            motion_config=MotionAnalyzerConfig(
                max_edge=160,
                min_motion_px=0.25,
                active_ratio_floor=0.003,
                mean_motion_floor=0.05,
                p90_motion_floor=0.1,
            ),
        ),
        frame_iter_factory=frame_iter,
        max_recent_cues=3,
    )

    status = runner.run_once(max_frames=3)
    payload = runner.status_dict()

    assert status.running is False
    assert status.frame_count == 3
    assert status.cue_count >= 2
    assert payload["scene_cut_count"] == 0
    assert payload["recent_cues"]
    assert payload["recent_episodes"]
    assert payload["recent_episodes"][0]["event_type"] == "aggressive_vehicle_motion_candidate"
