from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

from PIL import Image


@dataclass(frozen=True)
class LocalVideoSource:
    channel_id: int
    title: str
    device: str
    input_format: str = "mjpeg"
    width: int = 1280
    height: int = 720
    fps: float = 15.0
    preview_fps: float = 8.0

    def channel_dict(self) -> Dict[str, Any]:
        return {
            "id": int(self.channel_id),
            "guid": f"local-v4l2:{self.device}",
            "title": self.title,
            "server": "local-v4l2",
            "ptzCapabilities": None,
            "source": "local_v4l2",
            "device": self.device,
            "archive_available": False,
        }


def parse_local_video_sources(raw: object) -> Tuple[Dict[str, Any], ...]:
    """Parse trusted operator configuration without probing any devices."""

    text = str(raw or "").strip()
    if not text:
        return ()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("EVOSSEARCH_LOCAL_VIDEO_SOURCES_JSON must be valid JSON") from exc
    if not isinstance(payload, list):
        raise ValueError("EVOSSEARCH_LOCAL_VIDEO_SOURCES_JSON must contain a JSON array")

    parsed = []
    seen_ids = set()
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise ValueError(f"local video source #{index + 1} must be an object")
        try:
            channel_id = int(item.get("id") or item.get("channel_id"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"local video source #{index + 1} requires a positive numeric id") from exc
        if channel_id <= 0:
            raise ValueError(f"local video source #{index + 1} requires a positive numeric id")
        if channel_id in seen_ids:
            raise ValueError(f"duplicate local video channel id: {channel_id}")
        seen_ids.add(channel_id)

        device = str(item.get("device") or "").strip()
        if not device.startswith("/dev/video") or not device[len("/dev/video") :].isdigit():
            raise ValueError(f"local video source #{index + 1} requires a /dev/videoN device")
        title = str(item.get("title") or f"Local camera {channel_id}").strip()
        input_format = str(item.get("input_format") or "mjpeg").strip().lower()
        if input_format not in {"mjpeg", "h264", "yuyv422"}:
            raise ValueError(f"unsupported local video input_format: {input_format}")

        try:
            width = int(item.get("width") or 1280)
            height = int(item.get("height") or 720)
            fps = float(item.get("fps") or 15.0)
            preview_fps = float(item.get("preview_fps") or min(fps, 8.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"local video source #{index + 1} has invalid dimensions or fps") from exc
        if not 160 <= width <= 7680 or not 120 <= height <= 4320:
            raise ValueError(f"local video source #{index + 1} has unsupported dimensions")
        if not 0.2 <= fps <= 120.0 or not 0.2 <= preview_fps <= 30.0:
            raise ValueError(f"local video source #{index + 1} has unsupported fps")

        parsed.append(
            {
                "id": channel_id,
                "title": title,
                "device": device,
                "input_format": input_format,
                "width": width,
                "height": height,
                "fps": fps,
                "preview_fps": preview_fps,
            }
        )
    return tuple(parsed)


def _ffmpeg_binary() -> str:
    configured = str(os.getenv("EVOSSEARCH_FFMPEG_BIN") or "").strip()
    if configured:
        return configured
    bundled = Path(__file__).resolve().parent / ".eva-runtime" / "bin" / "ffmpeg"
    if bundled.is_file() and os.access(bundled, os.X_OK):
        return str(bundled)
    return shutil.which("ffmpeg") or "ffmpeg"


class LocalMjpegResponse:
    """Small requests.Response-compatible wrapper around a local FFmpeg pipe."""

    def __init__(self, process: subprocess.Popen[bytes], boundary: str) -> None:
        self.process = process
        self.status_code = 200
        self.headers = {"Content-Type": f"multipart/x-mixed-replace; boundary={boundary}"}
        self._eva_live_transport = "local_v4l2_ffmpeg"
        self._eva_media_source = "local-v4l2"
        self._closed = False
        self._close_lock = threading.Lock()

    def iter_content(self, chunk_size: int = 64 * 1024) -> Iterator[bytes]:
        stdout = self.process.stdout
        if stdout is None:
            return
        while True:
            chunk = stdout.read(max(1, int(chunk_size)))
            if not chunk:
                break
            yield chunk

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=1.0)
            if self.process.stdout is not None:
                self.process.stdout.close()


class LocalVideoClient:
    def __init__(self, source: LocalVideoSource) -> None:
        self.source = source

    def _assert_channel(self, channel_id: int) -> None:
        if int(channel_id) != int(self.source.channel_id):
            raise ValueError(f"local video client does not provide channel {channel_id}")

    def _input_args(self) -> list[str]:
        source = self.source
        return [
            "-f",
            "v4l2",
            "-input_format",
            source.input_format,
            "-video_size",
            f"{source.width}x{source.height}",
            "-framerate",
            f"{source.fps:g}",
            "-i",
            source.device,
        ]

    def get_snapshot(
        self,
        channel_id: int,
        stream: str = "mainStream",
        *,
        timeout: Optional[float] = None,
    ) -> Image.Image:
        del stream
        self._assert_channel(channel_id)
        command = [
            _ffmpeg_binary(),
            "-hide_banner",
            "-loglevel",
            "error",
            *self._input_args(),
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "pipe:1",
        ]
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=max(1.0, float(timeout or 5.0)),
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"local camera {self.source.device} snapshot timed out") from exc
        if result.returncode != 0 or not result.stdout:
            detail = result.stderr.decode("utf-8", errors="replace").strip()[-400:]
            raise RuntimeError(detail or f"local camera {self.source.device} returned no frame")
        try:
            with Image.open(BytesIO(result.stdout)) as image:
                return image.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"local camera {self.source.device} returned an invalid image") from exc

    def open_live_stream(
        self,
        channel_id: int,
        *,
        stream: str = "mainStream",
        timeout: Optional[Any] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> LocalMjpegResponse:
        del stream, timeout, headers
        self._assert_channel(channel_id)
        boundary = "eva-local-frame"
        command = [
            _ffmpeg_binary(),
            "-hide_banner",
            "-loglevel",
            "error",
            *self._input_args(),
            "-an",
            "-vf",
            f"fps={self.source.preview_fps:g}",
            "-q:v",
            "5",
            "-f",
            "mpjpeg",
            "-boundary_tag",
            boundary,
            "pipe:1",
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return LocalMjpegResponse(process, boundary)


class LocalVideoSourceRegistry:
    def __init__(self, rows: Sequence[Mapping[str, Any]] = ()) -> None:
        self._sources: Dict[int, LocalVideoSource] = {}
        for row in rows:
            source = LocalVideoSource(
                channel_id=int(row["id"]),
                title=str(row["title"]),
                device=str(row["device"]),
                input_format=str(row.get("input_format") or "mjpeg"),
                width=int(row.get("width") or 1280),
                height=int(row.get("height") or 720),
                fps=float(row.get("fps") or 15.0),
                preview_fps=float(row.get("preview_fps") or 8.0),
            )
            if source.channel_id in self._sources:
                raise ValueError(f"duplicate local video channel id: {source.channel_id}")
            self._sources[source.channel_id] = source

    def has_channel(self, channel_id: int) -> bool:
        return int(channel_id) in self._sources

    def channels(self) -> list[Dict[str, Any]]:
        return [self._sources[key].channel_dict() for key in sorted(self._sources)]

    def client_for(self, channel_id: int) -> Optional[LocalVideoClient]:
        source = self._sources.get(int(channel_id))
        return LocalVideoClient(source) if source is not None else None
