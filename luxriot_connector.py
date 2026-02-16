import base64
import json
import threading
import time
from io import BytesIO
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Set, Tuple, cast

import requests
from PIL import Image
from requests.auth import HTTPDigestAuth


class ProbeManagerLike(Protocol):
    def add_frame(self, channel_id: int, pil_image: Image.Image, timestamp_ms: Optional[int]) -> Any: ...


def _parse_optional_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    try:
        return int(cast(Any, value))
    except Exception:
        return None


class LuxriotClient:
    """Thin Luxriot Evo HTTP API client with digest auth and SSE-friendly helpers."""

    def __init__(self, base_url: str, username: str, password: str, timeout: int = 15) -> None:
        if not base_url:
            raise ValueError("Luxriot base URL is not configured.")
        self.base_url = base_url.rstrip("/")
        self.username = username or ""
        self.password = password or ""
        self.session = requests.Session()
        self.session.auth = HTTPDigestAuth(self.username, self.password)
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        url = f"{self.base_url}{path}"
        headers = kwargs.pop("headers", {}) or {}
        headers.setdefault("Accept", "application/json")
        try:
            resp = self.session.request(
                method,
                url,
                headers=headers,
                timeout=kwargs.pop("timeout", self.timeout),
                **kwargs,
            )
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:  # surface a clearer upstream error
            raise RuntimeError(f"Luxriot request failed ({url}): {exc}") from exc

    @staticmethod
    def _extract_first_json(lines: Iterable[str], max_chunks: int = 64) -> Optional[Any]:
        buffer = ""
        for idx, raw_line in enumerate(lines):
            if max_chunks and idx > max_chunks:
                break
            if raw_line is None:
                continue
            line = str(raw_line).strip()
            if not line:
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            buffer += line
            try:
                return json.loads(buffer)
            except json.JSONDecodeError:
                if len(buffer) > 50000:
                    break
                continue
        return None

    def get_channels(self) -> List[Dict[str, Any]]:
        resp = self._request(
            "GET",
            "/channels",
            params={"health": 0},
            headers={"Accept": "application/json"},
            stream=True,
        )
        try:
            payload = self._extract_first_json(resp.iter_lines(decode_unicode=True))
        finally:
            resp.close()
        if payload is None:
            raise RuntimeError("Luxriot /channels returned no data.")

        channels: Any = None
        if isinstance(payload, dict):
            if isinstance(payload.get("channels"), list):
                channels = payload["channels"]
            elif isinstance(payload.get("added"), dict) and isinstance(payload["added"].get("channels"), list):
                channels = payload["added"]["channels"]
            elif isinstance(payload.get("data"), dict) and isinstance(payload["data"].get("channels"), list):
                channels = payload["data"]["channels"]
        else:
            channels = payload

        if not isinstance(channels, list):
            raise RuntimeError(f"Unexpected /channels payload: {payload}")
        cleaned: List[Dict[str, Any]] = []
        for item in channels:
            if not isinstance(item, dict):
                continue
            item_map = cast(Mapping[str, object], item)
            channel_id = _parse_optional_int(item_map.get("id"))
            title_value = item_map.get("title")
            title = str(title_value).strip() if title_value is not None else ""
            if not title:
                title = f"Channel {channel_id if channel_id is not None else 'unknown'}"
            cleaned.append(
                {
                    "id": channel_id,
                    "guid": item_map.get("guid"),
                    "title": title,
                    "server": item_map.get("server"),
                    "ptzCapabilities": item_map.get("ptzCapabilities"),
                }
            )
        return cleaned

    def get_snapshot(self, channel_id: int, stream: str = "mainStream") -> Image.Image:
        resp = self._request(
            "GET",
            f"/live/{channel_id}/snapshot",
            params={"stream": stream},
            headers={"Accept": "image/jpeg"},
            stream=False,
            timeout=max(10, self.timeout),
        )
        image = Image.open(BytesIO(resp.content))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def create_bookmark(
        self,
        channel_id: int,
        title: str,
        description: str = "",
        timestamp_ms: Optional[int] = None,
        severity: str = "critical",
        state: str = "new",
    ) -> None:
        params = {
            "title": title,
            "channel": channel_id,
            "time": timestamp_ms or int(time.time() * 1000),
            "severity": severity,
            "state": state,
        }
        # Description must be plain text in POST body per docs
        self._request(
            "POST",
            "/createBookmark",
            params=params,
            headers={"Content-Type": "text/plain"},
            data=description or "",
        )


class LuxriotCaptureSession:
    """Background snapshot-to-summary loop for a single channel."""

    def __init__(
        self,
        manager: "LuxriotManager",
        channel_id: int,
        batch_size: int,
        prompt: str,
        model_hint: Optional[str] = None,
        interval_override: Optional[float] = None,
        summarization_enabled: bool = True,
        capture_kind: str = "video",
    ) -> None:
        self.manager = manager
        self.channel_id = channel_id
        self.batch_size = batch_size
        self.prompt = prompt
        self.model_hint = model_hint
        self.summarization_enabled = bool(summarization_enabled)
        self.capture_kind = (capture_kind or "video").strip().lower()
        if interval_override and interval_override > 0:
            self.interval = max(0.2, float(interval_override))
        else:
            self.interval = max(1, int(getattr(manager.config, "LUXRIOT_SNAPSHOT_INTERVAL", 5)))
        self.max_edge = int(getattr(manager.config, "LUXRIOT_SNAPSHOT_MAX_EDGE", 800))
        self.max_buffer = int(getattr(manager.config, "LUXRIOT_MAX_BUFFER_FRAMES", 180))
        self.client = manager.build_client()

        self.frames: List[Dict[str, Any]] = []
        self.logs: List[Dict[str, Any]] = []
        self.total_flushes = 0
        self.dropped_frames = 0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.last_error: Optional[str] = None

    def start(self) -> None:
        if not self.thread.is_alive():
            self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=0.75)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                snapshot = self.client.get_snapshot(self.channel_id)
                captured_at = time.time()
                frame = {
                    "thumbnail": self.manager.jpeg_encoder(snapshot, max_edge=self.max_edge, quality=85),
                    "captured_at": captured_at,
                    "time_sec": captured_at,
                    "width": snapshot.width,
                    "height": snapshot.height,
                }
                with self.lock:
                    self.frames.append(frame)
                    self._enforce_buffer_locked()
                try:
                    probe_manager = self.manager.probe_manager
                    if probe_manager is not None:
                        ts_ms = int(captured_at * 1000)
                        probe_manager.add_frame(self.channel_id, snapshot, ts_ms)
                except Exception as pm_exc:
                    self.last_error = str(pm_exc)
                if self.summarization_enabled and len(self.frames) >= self.batch_size:
                    self._summarize_batch()
                    with self.lock:
                        self.frames.clear()
                self.last_error = None
            except Exception as exc:
                self.last_error = str(exc)
            self.stop_event.wait(self.interval)

    def _summarize_batch(self) -> None:
        with self.lock:
            frames_copy = list(self.frames)
        if not frames_copy:
            return
        started = time.time()
        try:
            system_prompt = getattr(self.manager, "system_prompt", "") or ""
            messages = self.manager.message_builder(f"#{self.channel_id}", frames_copy, self.prompt, system_prompt)
            summary = self.manager.lm_callback(messages, self.model_hint)
            duration = time.time() - started
            entry = {
                "channel_id": self.channel_id,
                "summary": summary,
                "frame_count": len(frames_copy),
                "batch_size": self.batch_size,
                "created_at": time.time(),
                "duration_sec": duration,
                "prompt": self.prompt,
            }
            # Bookmarks from video understanding are disabled (hidden automation)
            with self.lock:
                self.logs.append(entry)
                self.total_flushes += 1
                if len(self.logs) > 50:
                    self.logs = self.logs[-50:]
        except Exception as exc:
            self.last_error = str(exc)

    def _enforce_buffer_locked(self) -> None:
        """Ensure frame buffer does not grow unbounded."""
        if self.max_buffer and len(self.frames) > self.max_buffer:
            overflow = len(self.frames) - self.max_buffer
            # Drop oldest frames to cap size; keep last max_buffer frames
            self.frames = self.frames[-self.max_buffer :]
            self.dropped_frames += overflow

    def flush_now(self) -> None:
        """Force a summary of current buffer."""
        if self.summarization_enabled:
            self._summarize_batch()
        with self.lock:
            self.frames.clear()

    def status(self) -> Dict[str, Any]:
        with self.lock:
            logs_copy = list(self.logs)
            pending_frames = len(self.frames)
        return {
            "running": not self.stop_event.is_set() and self.thread.is_alive(),
            "channel_id": self.channel_id,
            "batch_size": self.batch_size,
            "pending_frames": pending_frames,
            "interval_sec": self.interval,
            "max_edge": self.max_edge,
            "max_buffer_frames": self.max_buffer,
            "capture_kind": self.capture_kind,
            "summarization_enabled": self.summarization_enabled,
            "dropped_frames": self.dropped_frames,
            "flush_count": self.total_flushes,
            "last_error": self.last_error,
            "logs": logs_copy,
            "prompt": self.prompt,
            "model": self.model_hint,
        }


class LuxriotManager:
    """Coordinator for Luxriot snapshots, summaries, and channel helpers."""

    def __init__(
        self,
        config: Any,
        lm_callback: Callable[[List[Dict[str, Any]], Optional[str]], str],
        message_builder: Callable[[str, List[Dict[str, Any]], str, str], List[Dict[str, Any]]],
        jpeg_encoder: Callable[..., str],
        alert_parser: Optional[Callable[[str, int], List[Dict[str, Any]]]] = None,
        probe_manager: Optional[ProbeManagerLike] = None,
    ) -> None:
        self.config = config
        self.lm_callback = lm_callback
        self.message_builder = message_builder
        self.jpeg_encoder = jpeg_encoder
        self.alert_parser = alert_parser
        self.probe_manager: Optional[ProbeManagerLike] = probe_manager
        self.system_prompt = getattr(config, "LUXRIOT_SYSTEM_PROMPT_DEFAULT", "")

        self.sessions: Dict[int, LuxriotCaptureSession] = {}
        self.probe_sessions: Dict[int, LuxriotCaptureSession] = {}
        self.paused_probe_channels: Set[int] = set()
        self.cache_lock = threading.Lock()
        self.channels_cache: Optional[Tuple[float, List[Dict[str, Any]]]] = None

    def build_client(self) -> LuxriotClient:
        return LuxriotClient(
            base_url=self.config.LUXRIOT_BASE_URL,
            username=self.config.LUXRIOT_USERNAME,
            password=self.config.LUXRIOT_PASSWORD,
        )

    def get_channels(self, force: bool = False) -> List[Dict[str, Any]]:
        now = time.time()
        with self.cache_lock:
            if not force and self.channels_cache and now - self.channels_cache[0] < 30:
                return list(self.channels_cache[1])
        client = self.build_client()
        channels = client.get_channels()
        with self.cache_lock:
            self.channels_cache = (time.time(), channels)
        return channels

    def get_snapshot_base64(self, channel_id: int, stream_type: str = "mainStream") -> Tuple[str, Dict[str, Any]]:
        client = self.build_client()
        snapshot = client.get_snapshot(channel_id, stream=stream_type)
        encoded = self.jpeg_encoder(snapshot, max_edge=self.config.LUXRIOT_SNAPSHOT_MAX_EDGE, quality=85)
        return encoded, {"width": snapshot.width, "height": snapshot.height}

    def send_bookmark_event(
        self,
        channel_id: int,
        title: str,
        description: str,
        severity: str = "critical",
        state: str = "new",
        timestamp_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        client = self.build_client()
        sev_map = getattr(self.config, "LUXRIOT_SEVERITY_MAP", {}) or {}
        severity = str(severity).lower()
        severity = sev_map.get(severity, severity)
        client.create_bookmark(
            channel_id=channel_id,
            title=title,
            description=description or "",
            timestamp_ms=timestamp_ms,
            severity=severity,
            state=state,
        )
        return {"success": True, "channel_id": channel_id, "severity": severity, "state": state}

    def start_session(
        self,
        channel_id: int,
        batch_size: Optional[int] = None,
        prompt: str = "",
        model_hint: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        sizes = list(getattr(self.config, "LUXRIOT_BATCH_SIZES", (12, 24, 36)))
        default_size = sizes[0] if sizes else 12
        batch = default_size
        try:
            if batch_size:
                candidate = int(batch_size)
                if candidate in sizes:
                    batch = candidate
        except Exception:
            batch = default_size
        prompt = prompt or ""
        with self.cache_lock:
            existing = self.sessions.pop(channel_id, None)
            if existing:
                existing.stop()
            if system_prompt:
                self.system_prompt = system_prompt
            session = LuxriotCaptureSession(
                self,
                channel_id,
                batch,
                prompt,
                model_hint=model_hint,
                summarization_enabled=True,
                capture_kind="video",
            )
            self.sessions[channel_id] = session
            session.start()
            return session.status()

    def stop_session(self, channel_id: int) -> Dict[str, Any]:
        with self.cache_lock:
            session = self.sessions.pop(channel_id, None)
        if session:
            session.stop()
            return {"channel_id": channel_id, "running": False}
        return {"channel_id": channel_id, "running": False, "message": "No active session"}

    def start_probe_capture(self, channel_id: int, fps: Optional[float] = None, clear_pause: bool = True) -> Dict[str, Any]:
        with self.cache_lock:
            if clear_pause:
                self.paused_probe_channels.discard(channel_id)
            elif channel_id in self.paused_probe_channels:
                return {
                    "channel_id": channel_id,
                    "running": False,
                    "paused": True,
                    "message": "Probe capture paused",
                    "capture_kind": "analytics",
                    "summarization_enabled": False,
                }
            existing = self.probe_sessions.get(channel_id)
            if existing:
                return existing.status()
            interval = None
            if fps and fps > 0:
                interval = 1.0 / float(fps)
            session = LuxriotCaptureSession(
                self,
                channel_id,
                batch_size=1,
                prompt="",
                model_hint=None,
                interval_override=interval,
                summarization_enabled=False,
                capture_kind="analytics",
            )
            self.probe_sessions[channel_id] = session
            session.start()
            status = session.status()
            status["paused"] = False
            return status

    def stop_probe_capture(self, channel_id: int, pause: bool = True) -> Dict[str, Any]:
        with self.cache_lock:
            if pause:
                self.paused_probe_channels.add(channel_id)
            else:
                self.paused_probe_channels.discard(channel_id)
            session = self.probe_sessions.pop(channel_id, None)
        if session:
            session.stop()
            return {"channel_id": channel_id, "running": False, "paused": pause}
        return {
            "channel_id": channel_id,
            "running": False,
            "paused": pause,
            "message": "No active probe capture",
        }

    def is_probe_capture_paused(self, channel_id: int) -> bool:
        with self.cache_lock:
            return channel_id in self.paused_probe_channels

    def flush_session(self, channel_id: int) -> Dict[str, Any]:
        with self.cache_lock:
            session = self.sessions.get(channel_id)
        if not session:
            return {"success": False, "message": "No active session"}
        session.flush_now()
        return {"success": True, "message": "Flushed buffered frames", "status": session.status()}

    def session_status(self, channel_id: int) -> Dict[str, Any]:
        with self.cache_lock:
            session = self.sessions.get(channel_id)
        if session:
            return session.status()
        return {
            "running": False,
            "channel_id": channel_id,
            "batch_size": None,
            "pending_frames": 0,
            "interval_sec": getattr(self.config, "LUXRIOT_SNAPSHOT_INTERVAL", 5),
            "max_edge": getattr(self.config, "LUXRIOT_SNAPSHOT_MAX_EDGE", 800),
            "capture_kind": "video",
            "summarization_enabled": True,
            "last_error": None,
            "logs": [],
        }

    @staticmethod
    def _compact_stream_status(
        stream_type: str,
        status: Dict[str, Any],
        paused_channels: Optional[Set[int]] = None,
    ) -> Dict[str, Any]:
        compact = dict(status)
        logs = compact.pop("logs", None)
        compact["log_count"] = len(logs) if isinstance(logs, list) else 0
        compact["stream_type"] = stream_type
        if paused_channels is not None:
            channel_id = compact.get("channel_id")
            compact["paused"] = bool(isinstance(channel_id, int) and channel_id in paused_channels)
        return compact

    def streams_status(self) -> Dict[str, Any]:
        with self.cache_lock:
            video_items = list(self.sessions.items())
            analytics_items = list(self.probe_sessions.items())
            paused = set(self.paused_probe_channels)
        video_streams = [
            self._compact_stream_status("video", session.status(), paused)
            for _, session in video_items
        ]
        analytics_streams = [
            self._compact_stream_status("analytics", session.status(), paused)
            for _, session in analytics_items
        ]
        return {
            "video_streams": sorted(video_streams, key=lambda item: int(item.get("channel_id", 0))),
            "analytics_streams": sorted(analytics_streams, key=lambda item: int(item.get("channel_id", 0))),
            "paused_analytics_channels": sorted(paused),
            "running_total": len(video_streams) + len(analytics_streams),
        }

    def stop_stream(self, channel_id: int, stream_type: str = "both", pause_analytics: bool = True) -> Dict[str, Any]:
        normalized = (stream_type or "both").strip().lower()
        result: Dict[str, Any] = {"channel_id": channel_id, "stream_type": normalized}
        if normalized in {"video", "summary", "summaries"}:
            result["video"] = self.stop_session(channel_id)
        elif normalized in {"analytics", "probe", "probes"}:
            result["analytics"] = self.stop_probe_capture(channel_id, pause=pause_analytics)
        elif normalized in {"both", "all"}:
            result["video"] = self.stop_session(channel_id)
            result["analytics"] = self.stop_probe_capture(channel_id, pause=pause_analytics)
        else:
            raise ValueError("stream_type must be one of: video, analytics, both")
        return result

    def stop_all_streams(
        self,
        stop_video: bool = True,
        stop_analytics: bool = True,
        pause_analytics: bool = True,
    ) -> Dict[str, Any]:
        with self.cache_lock:
            video_channels = list(self.sessions.keys()) if stop_video else []
            analytics_channels = list(self.probe_sessions.keys()) if stop_analytics else []
        stopped_video = [self.stop_session(ch) for ch in video_channels]
        stopped_analytics = [self.stop_probe_capture(ch, pause=pause_analytics) for ch in analytics_channels]
        return {
            "stopped_video_count": len(stopped_video),
            "stopped_analytics_count": len(stopped_analytics),
            "video": stopped_video,
            "analytics": stopped_analytics,
        }
