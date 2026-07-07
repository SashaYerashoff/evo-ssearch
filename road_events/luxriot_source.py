from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .video import DecodedVideoFrame, iter_video_frames


@dataclass(frozen=True)
class LuxriotLiveSegment:
    path: Path
    channel_id: int
    stream: str
    started_at_ms: int
    bytes_written: int
    content_type: str
    source_start_ms: int | None = None
    source_end_ms: int | None = None


def iter_luxriot_archive_snapshots(
    client: Any,
    channel_id: int,
    *,
    start_ms: int,
    end_ms: int,
    interval_ms: int = 1000,
    stream_type: str = "mainStream",
    max_frames: int | None = None,
    align_to_recorded_frames: bool = True,
) -> Iterator[DecodedVideoFrame]:
    """Yield Luxriot archived JPEG snapshots as road-CV frames.

    Luxriot archive video chunks use a private stream framing.  Archived JPEG
    snapshots are simpler and stable enough for sparse investigation sweeps:
    sample one recorded frame per interval, then let the CV layer group cues
    into candidate episodes for VLM/CLIP confirmation.
    """

    start_value = int(start_ms)
    end_value = int(end_ms)
    if end_value < start_value:
        raise ValueError("end_ms must be greater than or equal to start_ms")
    step = max(1, int(interval_ms))
    if max_frames is not None and int(max_frames) <= 0:
        return
    emitted = 0
    frame_index = 0
    cursor = start_value
    last_frame_time: int | None = None
    while cursor <= end_value:
        sample_time = cursor
        if align_to_recorded_frames:
            next_time = client.get_next_archive_frame_time(
                int(channel_id),
                cursor,
                stream_type=stream_type,
            )
            if next_time is None or next_time > end_value:
                break
            sample_time = int(next_time)
            if last_frame_time is not None and sample_time <= last_frame_time:
                sample_time = last_frame_time + step
        image = client.get_archive_snapshot(
            int(channel_id),
            sample_time,
            stream_type=stream_type,
        )
        yield DecodedVideoFrame(
            frame_index=frame_index,
            timestamp_ms=sample_time,
            source_timestamp_ms=sample_time,
            image=image,
        )
        emitted += 1
        frame_index += 1
        if max_frames is not None and emitted >= max(0, int(max_frames)):
            break
        last_frame_time = sample_time
        cursor = sample_time + step


def capture_luxriot_live_mp4_segment(
    client: Any,
    channel_id: int,
    output_path: str | Path,
    *,
    stream: str = "mainStream",
    max_bytes: int = 4 * 1024 * 1024,
    max_seconds: float = 15.0,
) -> LuxriotLiveSegment:
    """Capture a bounded Luxriot live MP4 segment through the authenticated client."""

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    started_at_ms = int(time.time() * 1000)
    byte_budget = max(1, int(max_bytes))
    second_budget = max(0.25, float(max_seconds))
    written = 0
    response = client.open_live_stream(int(channel_id), stream=stream)
    content_type = str(response.headers.get("Content-Type") or response.headers.get("content-type") or "")
    deadline = time.monotonic() + second_budget
    try:
        with target.open("wb") as fh:
            for chunk in response.iter_content(65536):
                if not chunk:
                    if time.monotonic() >= deadline:
                        break
                    continue
                remaining = byte_budget - written
                if remaining <= 0:
                    break
                fh.write(chunk[:remaining])
                written += min(len(chunk), remaining)
                if written >= byte_budget or time.monotonic() >= deadline:
                    break
    finally:
        response.close()
    if written <= 0:
        raise RuntimeError(f"Luxriot live stream for channel {channel_id} returned no video bytes.")
    return LuxriotLiveSegment(
        path=target,
        channel_id=int(channel_id),
        stream=stream,
        started_at_ms=started_at_ms,
        bytes_written=written,
        content_type=content_type,
        source_start_ms=started_at_ms,
    )


def _parse_header_int(headers: Any, name: str) -> int | None:
    value = None
    try:
        value = headers.get(name) or headers.get(name.lower())
    except Exception:
        value = None
    if value is None:
        return None
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def capture_luxriot_archive_mp4_segment(
    client: Any,
    channel_id: int,
    time_ms: int,
    output_path: str | Path,
    *,
    stream_type: str = "mainStream",
    max_bytes: int = 64 * 1024 * 1024,
    max_seconds: float = 30.0,
) -> LuxriotLiveSegment:
    """Capture a bounded Luxriot archive stream segment.

    Luxriot may return either MP4 bytes or an internal stream framing depending
    on the camera/codec.  The caller validates decodability through OpenCV.
    """

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    request_started_at_ms = int(time.time() * 1000)
    byte_budget = max(1, int(max_bytes))
    second_budget = max(0.25, float(max_seconds))
    written = 0
    response = client.open_archive_stream(
        int(channel_id),
        int(time_ms),
        stream_type=stream_type,
    )
    content_type = str(response.headers.get("Content-Type") or response.headers.get("content-type") or "")
    source_start_ms = _parse_header_int(response.headers, "X-Stream-Start-Time")
    source_end_ms = _parse_header_int(response.headers, "X-Stream-End-Time")
    deadline = time.monotonic() + second_budget
    try:
        with target.open("wb") as fh:
            for chunk in response.iter_content(65536):
                if not chunk:
                    if time.monotonic() >= deadline:
                        break
                    continue
                remaining = byte_budget - written
                if remaining <= 0:
                    break
                fh.write(chunk[:remaining])
                written += min(len(chunk), remaining)
                if written >= byte_budget or time.monotonic() >= deadline:
                    break
    finally:
        response.close()
    if written <= 0:
        raise RuntimeError(f"Luxriot archive stream for channel {channel_id} returned no video bytes.")
    return LuxriotLiveSegment(
        path=target,
        channel_id=int(channel_id),
        stream=stream_type,
        started_at_ms=request_started_at_ms,
        bytes_written=written,
        content_type=content_type,
        source_start_ms=source_start_ms,
        source_end_ms=source_end_ms,
    )


def iter_luxriot_live_segment_frames(
    client: Any,
    channel_id: int,
    *,
    stream: str = "mainStream",
    segment_bytes: int = 4 * 1024 * 1024,
    segment_seconds: float = 15.0,
    every_n: int = 1,
    max_frames: int | None = None,
    keep_segment_path: str | Path | None = None,
) -> Iterator[DecodedVideoFrame]:
    """Capture a short live MP4 segment and yield decoded frames.

    This is a bounded smoke/investigation primitive, not the final long-lived
    service loop.  It avoids teaching OpenCV Luxriot digest auth while still
    exercising the real live-video endpoint.
    """

    cleanup = keep_segment_path is None
    if keep_segment_path is None:
        tmp = tempfile.NamedTemporaryFile(prefix=f"luxriot-ch{channel_id}-", suffix=".mp4", delete=False)
        segment_path = Path(tmp.name)
        tmp.close()
    else:
        segment_path = Path(keep_segment_path)
    try:
        segment = capture_luxriot_live_mp4_segment(
            client,
            int(channel_id),
            segment_path,
            stream=stream,
            max_bytes=segment_bytes,
            max_seconds=segment_seconds,
        )
        for frame in iter_video_frames(
            str(segment.path),
            every_n=max(1, int(every_n)),
            max_frames=max_frames,
            wallclock_timestamps=False,
        ):
            source_ts = int(frame.source_timestamp_ms or 0)
            yield DecodedVideoFrame(
                frame_index=frame.frame_index,
                timestamp_ms=segment.started_at_ms + source_ts if source_ts > 0 else int(time.time() * 1000),
                source_timestamp_ms=source_ts if source_ts > 0 else None,
                image=frame.image,
            )
    finally:
        if cleanup:
            try:
                segment_path.unlink()
            except FileNotFoundError:
                pass


def iter_luxriot_archive_segment_frames(
    client: Any,
    channel_id: int,
    *,
    start_ms: int,
    end_ms: int,
    stream_type: str = "mainStream",
    segment_bytes: int = 64 * 1024 * 1024,
    segment_seconds: float = 30.0,
    every_n: int = 1,
    max_frames: int | None = None,
    keep_segment_dir: str | Path | None = None,
) -> Iterator[DecodedVideoFrame]:
    """Yield decoded frames from Luxriot archive stream segments."""

    start_value = int(start_ms)
    end_value = int(end_ms)
    if end_value < start_value:
        raise ValueError("end_ms must be greater than or equal to start_ms")
    if max_frames is not None and int(max_frames) <= 0:
        return
    cleanup = keep_segment_dir is None
    if keep_segment_dir is None:
        temp_dir = Path(tempfile.mkdtemp(prefix=f"luxriot-archive-ch{channel_id}-"))
    else:
        temp_dir = Path(keep_segment_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
    emitted = 0
    emitted_frame_index = 0
    cursor = start_value
    segment_index = 0
    try:
        while cursor <= end_value:
            segment_path = temp_dir / f"ch{channel_id}-{segment_index:04d}-{cursor}.mp4"
            segment = capture_luxriot_archive_mp4_segment(
                client,
                int(channel_id),
                cursor,
                segment_path,
                stream_type=stream_type,
                max_bytes=segment_bytes,
                max_seconds=segment_seconds,
            )
            segment_start = int(segment.source_start_ms or cursor)
            segment_end = int(segment.source_end_ms or segment_start)
            yielded_from_segment = False
            for frame in iter_video_frames(
                str(segment.path),
                every_n=max(1, int(every_n)),
                max_frames=None,
                wallclock_timestamps=False,
            ):
                source_offset_ms = int(frame.source_timestamp_ms or 0)
                timestamp_ms = segment_start + source_offset_ms
                if timestamp_ms < start_value:
                    continue
                if timestamp_ms > end_value:
                    break
                yielded_from_segment = True
                yield DecodedVideoFrame(
                    frame_index=emitted_frame_index,
                    timestamp_ms=timestamp_ms,
                    source_timestamp_ms=timestamp_ms,
                    image=frame.image,
                )
                emitted += 1
                emitted_frame_index += 1
                if max_frames is not None and emitted >= int(max_frames):
                    return
            if segment_end <= cursor:
                cursor += 1000
            else:
                cursor = segment_end + 1
            if not yielded_from_segment and cursor <= start_value:
                cursor = start_value + 1000
            segment_index += 1
    finally:
        if cleanup:
            for path in temp_dir.glob("*"):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            try:
                temp_dir.rmdir()
            except OSError:
                pass
