#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import Config
from luxriot_connector import LuxriotClient
from road_events import (
    AutoSceneCardConfig,
    SceneCalibrationConfig,
    calibrate_scene_card_from_results,
    infer_scene_card_from_frames,
    iter_luxriot_archive_segment_frames,
    iter_luxriot_archive_snapshots,
)


def _resolve_channel(client: LuxriotClient, channel_ref: str) -> tuple[int, str]:
    ref = str(channel_ref or "").strip()
    if not ref:
        raise SystemExit("channel is required")
    channels = client.get_channels()
    if ref.isdigit():
        channel_id = int(ref)
        for channel in channels:
            if channel.get("id") == channel_id:
                return channel_id, str(channel.get("title") or f"Channel {channel_id}")
        return channel_id, f"Channel {channel_id}"
    lowered = ref.lower()
    matches = [
        channel
        for channel in channels
        if lowered in str(channel.get("title") or "").lower()
    ]
    if not matches:
        titles = ", ".join(f"{item.get('id')}:{item.get('title')}" for item in channels)
        raise SystemExit(f"channel {ref!r} not found. Available: {titles}")
    if len(matches) > 1:
        titles = ", ".join(f"{item.get('id')}:{item.get('title')}" for item in matches)
        raise SystemExit(f"channel {ref!r} is ambiguous: {titles}")
    channel = matches[0]
    return int(channel["id"]), str(channel.get("title") or f"Channel {channel['id']}")


def _archive_window(
    client: LuxriotClient,
    channel_id: int,
    *,
    stream_type: str,
    hours: float,
) -> tuple[int, int, dict]:
    boundaries = client.get_archive_boundaries(channel_id, stream_type=stream_type)
    main = boundaries.get("main") or boundaries.get("use") or {}
    archive_start = int(main.get("from") or 0)
    archive_end = int(main.get("to") or 0)
    if archive_start <= 0 or archive_end <= 0 or archive_end < archive_start:
        raise SystemExit(f"channel {channel_id} has no archive boundaries for {stream_type}: {boundaries}")
    start = max(archive_start, archive_end - int(max(0.1, float(hours)) * 3_600_000))
    return start, archive_end, boundaries


def _sample_starts(start_ms: int, end_ms: int, count: int, window_ms: int) -> list[int]:
    count = max(1, int(count))
    if count == 1 or end_ms - start_ms <= window_ms:
        return [start_ms]
    span = max(1, end_ms - start_ms - window_ms)
    return [start_ms + int(round(span * idx / max(1, count - 1))) for idx in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate a Luxriot road scene-card from archive motion history."
    )
    parser.add_argument("--channel", required=True, help="Channel id or unique title substring")
    parser.add_argument("--mode", choices=("archive-video", "archive-snapshots"), default="archive-video")
    parser.add_argument("--stream", default="mainStream")
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--window-sec", type=float, default=12.0)
    parser.add_argument("--frames-per-sample", type=int, default=72)
    parser.add_argument("--every-n", type=int, default=6)
    parser.add_argument("--segment-mb", type=float, default=16.0)
    parser.add_argument("--segment-sec", type=float, default=30.0)
    parser.add_argument("--snapshot-interval-ms", type=int, default=1000)
    parser.add_argument("--max-edge", type=int, default=240)
    parser.add_argument("--output", default="", help="Optional scene-card JSON output path")
    args = parser.parse_args()

    client = LuxriotClient(Config.LUXRIOT_BASE_URL, Config.LUXRIOT_USERNAME, Config.LUXRIOT_PASSWORD)
    channel_id, channel_title = _resolve_channel(client, args.channel)
    start_ms, end_ms, boundaries = _archive_window(
        client,
        channel_id,
        stream_type=args.stream,
        hours=args.hours,
    )
    window_ms = max(1000, int(float(args.window_sec) * 1000))
    starts = _sample_starts(start_ms, end_ms, args.samples, window_ms)
    auto_results = []
    sample_reports = []
    for index, sample_start in enumerate(starts):
        sample_end = min(end_ms, sample_start + window_ms)
        if args.mode == "archive-snapshots":
            frames = list(
                iter_luxriot_archive_snapshots(
                    client,
                    channel_id,
                    start_ms=sample_start,
                    end_ms=sample_end,
                    interval_ms=max(1, int(args.snapshot_interval_ms)),
                    stream_type=args.stream,
                    max_frames=max(1, int(args.frames_per_sample)),
                )
            )
        else:
            frames = list(
                iter_luxriot_archive_segment_frames(
                    client,
                    channel_id,
                    start_ms=sample_start,
                    end_ms=sample_end,
                    stream_type=args.stream,
                    segment_bytes=int(max(0.1, float(args.segment_mb)) * 1024 * 1024),
                    segment_seconds=max(0.25, float(args.segment_sec)),
                    every_n=max(1, int(args.every_n)),
                    max_frames=max(1, int(args.frames_per_sample)),
                )
            )
        auto_result = infer_scene_card_from_frames(
            channel_id,
            channel_title,
            frames,
            config=AutoSceneCardConfig(
                max_edge=max(96, int(args.max_edge)),
                min_frames=min(24, max(6, len(frames) // 2)) if frames else 24,
                min_motion_pairs=4,
            ),
        )
        auto_results.append(auto_result)
        sample_reports.append(
            {
                "index": index,
                "start_ms": sample_start,
                "end_ms": sample_end,
                "frame_count": len(frames),
                "confidence": auto_result.confidence,
                "reason": auto_result.reason,
                "motion_pair_count": auto_result.motion_pair_count,
                "scene_cut_count": auto_result.scene_cut_count,
                "zone_area_ratio": round(float(auto_result.zone_area_ratio), 4),
                "flow_dominance": round(float(auto_result.flow_dominance), 4),
            }
        )

    calibration = calibrate_scene_card_from_results(
        channel_id,
        channel_title,
        auto_results,
        config=SceneCalibrationConfig(),
    )
    scene_payload = calibration.as_dict()["scene_card"]
    output_payload = {
        "channel_id": channel_id,
        "channel_title": channel_title,
        "mode": args.mode,
        "stream": args.stream,
        "archive_start_ms": start_ms,
        "archive_end_ms": end_ms,
        "archive_boundaries": boundaries,
        "calibration": calibration.as_dict(),
        "samples": sample_reports,
        "scene_cards_json": {"channels": [scene_payload]},
    }
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"channels": [scene_payload]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        output_payload["output"] = str(output_path)
    print(json.dumps(output_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
