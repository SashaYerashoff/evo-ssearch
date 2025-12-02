import base64
import json
import threading
import time
from io import BytesIO
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import requests
from PIL import Image
from requests.auth import HTTPDigestAuth


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
            channel_id = item.get("id")
            try:
                channel_id = int(channel_id)
            except Exception:
                pass
            cleaned.append(
                {
                    "id": channel_id,
                    "guid": item.get("guid"),
                    "title": item.get("title") or f"Channel {channel_id}",
                    "server": item.get("server"),
                    "ptzCapabilities": item.get("ptzCapabilities"),
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
    ) -> None:
        self.manager = manager
        self.channel_id = channel_id
        self.batch_size = batch_size
        self.prompt = prompt
        self.model_hint = model_hint
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
                if len(self.frames) >= self.batch_size:
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
            messages = self.manager.message_builder(f"#{self.channel_id}", frames_copy, self.prompt)
            summary = self.manager.lm_callback(messages, model_override=self.model_hint)
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
        message_builder: Callable[[str, List[Dict[str, Any]], str], List[Dict[str, Any]]],
        jpeg_encoder: Callable[..., str],
    ) -> None:
        self.config = config
        self.lm_callback = lm_callback
        self.message_builder = message_builder
        self.jpeg_encoder = jpeg_encoder

        self.sessions: Dict[int, LuxriotCaptureSession] = {}
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
            session = LuxriotCaptureSession(self, channel_id, batch, prompt, model_hint=model_hint)
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
            "last_error": None,
            "logs": [],
        }
