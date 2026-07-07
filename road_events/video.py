from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class DecodedVideoFrame:
    frame_index: int
    timestamp_ms: int
    image: np.ndarray
    source_timestamp_ms: int | None = None


def iter_video_frames(
    source: str | int,
    *,
    every_n: int = 1,
    max_frames: int | None = None,
    wallclock_timestamps: bool = False,
) -> Iterator[DecodedVideoFrame]:
    """Yield RGB frames from a video file, device, or RTSP URL via OpenCV.

    The function is a low-level decoder primitive.  It does not reconnect or
    own service lifecycle; the Luxriot-facing runner will wrap it with policy.
    """

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"could not open video source: {source!r}")
    try:
        frame_index = 0
        emitted = 0
        step = max(1, int(every_n))
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if frame_index % step != 0:
                frame_index += 1
                continue
            source_ts = int(capture.get(cv2.CAP_PROP_POS_MSEC) or 0)
            timestamp_ms = int(time.time() * 1000) if wallclock_timestamps or source_ts <= 0 else source_ts
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            yield DecodedVideoFrame(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                source_timestamp_ms=source_ts if source_ts > 0 else None,
                image=frame_rgb,
            )
            emitted += 1
            frame_index += 1
            if max_frames is not None and emitted >= max_frames:
                break
    finally:
        capture.release()
