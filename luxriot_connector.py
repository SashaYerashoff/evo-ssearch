import base64
import hashlib
import json
import logging
import math
import subprocess
import tempfile
import re
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Set, Tuple, cast
from urllib.parse import quote, urlsplit, urlunsplit

import requests
from PIL import Image
from requests.auth import HTTPDigestAuth
try:
    from road_events import (
        AutoSceneCardConfig,
        DecodedVideoFrame,
        RoadMotionAnalyzer,
        infer_scene_card_from_frames,
        iter_luxriot_live_segment_frames,
    )
except Exception:  # pragma: no cover - road CV is optional in minimal installs
    AutoSceneCardConfig = None  # type: ignore[assignment]
    DecodedVideoFrame = None  # type: ignore[assignment]
    RoadMotionAnalyzer = None  # type: ignore[assignment]
    infer_scene_card_from_frames = None  # type: ignore[assignment]
    iter_luxriot_live_segment_frames = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

DEFAULT_ALERTS_JSON_PROMPT = (
    "Machine-readable alert output for operator review:\n"
    "- Always append exactly one block at the end, prefixed with ALERTS_JSON:.\n"
    "- If no trigger matches, use {\"alerts\": []}.\n"
    "- If one or more triggers match, include one alert object per distinct visible trigger using this schema:\n"
    "ALERTS_JSON:\n"
    "{\n"
    "  \"alerts\": [\n"
    "    {\n"
    "      \"title\": \"Short event title\",\n"
    "      \"description\": \"<= 240 chars, concrete and actionable\",\n"
    "      \"severity\": \"info|low|normal|high|critical\",\n"
    "      \"state\": \"new\",\n"
    "      \"channel_id\": {channel_id},\n"
    "      \"timestamp_ms\": 0\n"
    "    }\n"
    "  ]\n"
    "}\n"
    "Alert candidates are defined by the Alert review policy and by visible immediate safety/security hazards. "
    "General hazards include physical violence, a person falling/collapsing or appearing to need urgent help, "
    "dangerous vehicle behavior, forced entry, property damage, theft-like tampering, weapon/fire/smoke, "
    "critical camera obstruction, or crowd escalation. "
    "Do not classify behavior as illegal/unlawful; describe visible facts requiring operator review. "
    "Do not alert on littering, loitering, casual gestures, routine walking/parking/deliveries, or ambiguous movement "
    "unless the Alert review policy explicitly asks for that review signal. "
    "Rules: emit one alert object per distinct visible trigger in the batch, up to 8 objects; "
    "do not merge unrelated triggers into one alert; do not output a prose-only Alerts section or Warning Level list; "
    "if a matching event is described anywhere in the prose summary, it must also appear in ALERTS_JSON; "
    "evaluate every operator-defined trigger independently against the current snapshots; "
    "if two distinct triggers are visible in the same batch, emit two alert objects; "
    "timestamp_ms should be observed batch epoch in milliseconds (or 0 if unknown)."
)

DEFAULT_ALERT_POLICY_PROMPT = (
    "Alert review policy:\n"
    "- Evaluate general safety/security hazards even if the operator did not list them explicitly. "
    "Use severity high/critical only for immediate visible danger; use low/normal for review-worthy but non-urgent events.\n"
    "- Evaluate channel-specific operator criteria as first-class alert triggers. Operator criteria may describe "
    "non-security review signals, vulnerable-person monitoring, site-specific rules, or temporary watch items.\n"
    "- If operator criteria mention health, age, impairment, intent, legality, or identity, do not diagnose or accuse. "
    "Alert only on visible facts such as falling, collapse, distress, immobility, unsafe movement, obstruction, "
    "or a requested visible object/action.\n"
    "- If evidence is ambiguous but relevant to an explicit operator criterion, emit a low/info alert with uncertainty "
    "instead of silently dropping it.\n"
    "Channel-specific operator alert criteria:\n"
    "{operator_alert_policy}"
)

LIVE_OBSERVATION_STATE_PROMPT = (
    "Current-batch observation contract:\n"
    "- Treat prior channel memory as context only, never as visual evidence for the current batch.\n"
    "- Evaluate every watched entity and operator-defined trigger independently against the CURRENT snapshots.\n"
    "- Before ALERTS_JSON, include a concise 'Current observed state' section for watched entities/triggers: "
    "present|absent|uncertain with snapshot numbers or timestamps as evidence.\n"
    "- If two distinct triggers are visible in the same batch, report both and emit two alert objects.\n"
    "- Claim enter/leave only when the current snapshots show a before/after transition; otherwise report the "
    "current state and let backend continuity tools compare adjacent batches."
)

VECTOR_SIGNAL_PROMPT_PREFIX = (
    "Current vector/homeostasis signal contract:\n"
    "- VECTOR_SIGNALS_JSON is a secondary attention/arousal signal from CLIP probes and lightweight CV, not visual proof.\n"
    "- Use it to decide which current snapshots deserve extra scrutiny; verify any candidate directly in the current images.\n"
    "- If a vector cue and the current snapshots support an Alert review policy trigger, emit the normal ALERTS_JSON alert.\n"
    "- If the cue is not visually supported, mention uncertainty briefly and do not create an alert from the vector cue alone.\n"
)

_OUTDATED_ALERT_PROMPT_MARKERS = (
    "if no trigger match: emit no json block",
    "rules: max 3 alerts",
    "\"timestamp_ms\": 1772202050000",
    "also emit operator-defined low/normal test triggers when the stream prompt explicitly asks",
)

ALERT_SEVERITY_ORDER = ("critical", "high", "normal", "low", "info")
ALERT_SEVERITY_SET = set(ALERT_SEVERITY_ORDER)


class ProbeManagerLike(Protocol):
    def add_frame(self, channel_id: int, pil_image: Image.Image, timestamp_ms: Optional[int]) -> Any: ...
    def query(
        self,
        channel_id: int,
        positives: Sequence[str],
        negatives: Sequence[str],
        pos_floor: float,
        margin_thr: float,
        top_k: int,
        window_sec: Optional[float] = None,
        image_probe: Optional[Dict[str, Any]] = None,
        roi_norm: Optional[Tuple[float, float, float, float]] = None,
        roi_padding: float = 0.05,
    ) -> Mapping[str, Any]: ...


class ProbeStoreLike(Protocol):
    def list_probes(self) -> List[Dict[str, Any]]: ...


AlertParserFn = Callable[[str, int, Optional[int]], List[Dict[str, Any]]]
SummaryDispatcherFn = Callable[[Mapping[str, Any], str], Mapping[str, Any]]
SummaryArchiveFn = Callable[[Mapping[str, Any]], Optional[Mapping[str, Any]]]


class AlertDeliveryResult(int):
    """Integer-compatible delivery result with alert/bookmark diagnostics."""

    def __new__(
        cls,
        sent: int = 0,
        *,
        parsed: int = 0,
        json_alert_count: int = 0,
        prose_alert_count: int = 0,
        failed: int = 0,
        skipped_duplicate: int = 0,
        last_error: Optional[str] = None,
        alerts_detected: bool = False,
        parser_error: Optional[str] = None,
        alert_events: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> "AlertDeliveryResult":
        obj = int.__new__(cls, int(max(0, sent)))
        obj.parsed = int(max(0, parsed))
        obj.json_alert_count = int(max(0, json_alert_count))
        obj.prose_alert_count = int(max(0, prose_alert_count))
        obj.failed = int(max(0, failed))
        obj.skipped_duplicate = int(max(0, skipped_duplicate))
        obj.last_error = str(last_error or "").strip() or None
        obj.alerts_detected = bool(alerts_detected)
        obj.parser_error = str(parser_error or "").strip() or None
        cleaned_events: List[Dict[str, Any]] = []
        if isinstance(alert_events, Sequence) and not isinstance(alert_events, (str, bytes, bytearray)):
            for raw_event in alert_events[:32]:
                if not isinstance(raw_event, Mapping):
                    continue
                title = str(raw_event.get("title") or "Event").strip()[:120] or "Event"
                description = str(raw_event.get("description") or "").strip()[:300]
                severity = str(raw_event.get("severity") or "normal").strip().lower()[:20] or "normal"
                state = str(raw_event.get("state") or "new").strip().lower()[:20] or "new"
                event: Dict[str, Any] = {
                    "title": title,
                    "description": description,
                    "severity": severity,
                    "state": state,
                }
                channel_id = _parse_optional_int(raw_event.get("channel_id"))
                if channel_id is not None:
                    event["channel_id"] = int(channel_id)
                timestamp_ms = _parse_optional_int(raw_event.get("timestamp_ms"))
                if timestamp_ms is not None:
                    event["timestamp_ms"] = int(timestamp_ms)
                status = str(raw_event.get("delivery_status") or "").strip().lower()
                if status:
                    event["delivery_status"] = status[:40]
                error = str(raw_event.get("error") or "").strip()
                if error:
                    event["error"] = error[:240]
                cleaned_events.append(event)
        obj.alert_events = cleaned_events
        return obj

    def as_dict(self) -> Dict[str, Any]:
        return {
            "alerts_detected": self.alerts_detected,
            "alerts_parsed": self.parsed,
            "parser_alert_count": self.parsed,
            "json_alert_count": self.json_alert_count,
            "prose_alert_count": self.prose_alert_count,
            "bookmarks_sent": int(self),
            "bookmark_failed_count": self.failed,
            "bookmark_skipped_duplicate_count": self.skipped_duplicate,
            "bookmark_cooldown_skipped_count": self.skipped_duplicate,
            "bookmark_last_error": self.last_error,
            "alert_parser_error": self.parser_error,
            "alert_events": list(self.alert_events),
        }


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
        try:
            with Image.open(BytesIO(resp.content)) as opened:
                opened.load()
                image = opened.convert("RGB") if opened.mode != "RGB" else opened.copy()
        except Exception as exc:
            raise RuntimeError(
                f"Luxriot snapshot decode failed for channel {channel_id}: {exc}"
            ) from exc
        if image.width <= 0 or image.height <= 0:
            raise RuntimeError(f"Luxriot snapshot for channel {channel_id} has invalid dimensions.")
        return image

    @staticmethod
    def _decode_jpeg_response(resp: requests.Response, *, label: str) -> Image.Image:
        try:
            with Image.open(BytesIO(resp.content)) as opened:
                opened.load()
                image = opened.convert("RGB") if opened.mode != "RGB" else opened.copy()
        except Exception as exc:
            raise RuntimeError(f"{label} decode failed: {exc}") from exc
        if image.width <= 0 or image.height <= 0:
            raise RuntimeError(f"{label} has invalid dimensions.")
        return image

    def get_archive_boundaries(
        self,
        channel_id: int,
        *,
        stream_type: str = "mainStream",
    ) -> Dict[str, Dict[str, int]]:
        resp = self._request(
            "GET",
            f"/archive/{channel_id}/boundaries",
            params={"streamType": stream_type},
            headers={"Accept": "application/json"},
            timeout=max(10, self.timeout),
        )
        payload = resp.json()
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"Unexpected Luxriot archive boundaries payload: {payload!r}")
        normalized: Dict[str, Dict[str, int]] = {}
        for key in ("use", "main", "sub", "edge"):
            raw_range = payload.get(key)
            if not isinstance(raw_range, Mapping):
                continue
            from_ms = _parse_optional_int(raw_range.get("from")) or 0
            to_ms = _parse_optional_int(raw_range.get("to")) or 0
            normalized[key] = {"from": int(from_ms), "to": int(to_ms)}
        return normalized

    def get_archive_timeline(
        self,
        channel_id: int,
        start_ms: int,
        end_ms: int,
        *,
        interval_ms: int = 5000,
        stream_type: str = "mainStream",
    ) -> List[Tuple[int, int]]:
        resp = self._request(
            "GET",
            f"/archive/{channel_id}/timeline",
            params={
                "start": int(start_ms),
                "end": int(end_ms),
                "interval": max(1, int(interval_ms)),
                "streamType": stream_type,
            },
            headers={"Accept": "application/json"},
            timeout=max(10, self.timeout),
        )
        payload = resp.json()
        if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
            raise RuntimeError(f"Unexpected Luxriot archive timeline payload: {payload!r}")
        ranges: List[Tuple[int, int]] = []
        for item in payload:
            if not isinstance(item, Sequence) or isinstance(item, (str, bytes, bytearray)) or len(item) < 2:
                continue
            start_value = _parse_optional_int(item[0])
            end_value = _parse_optional_int(item[1])
            if start_value is None or end_value is None:
                continue
            ranges.append((int(start_value), int(end_value)))
        return ranges

    def get_archive_frame_time(
        self,
        channel_id: int,
        time_ms: int,
        *,
        direction: str = "next",
        stream_type: str = "mainStream",
    ) -> Optional[int]:
        endpoint = "nextFrameTime" if str(direction).lower().strip() != "prev" else "prevFrameTime"
        resp = self._request(
            "GET",
            f"/archive/{channel_id}/{endpoint}",
            params={"time": int(time_ms), "streamType": stream_type},
            headers={"Accept": "text/plain"},
            timeout=max(10, self.timeout),
        )
        frame_time = _parse_optional_int(resp.text)
        if frame_time is None or frame_time <= 0:
            return None
        return int(frame_time)

    def get_next_archive_frame_time(
        self,
        channel_id: int,
        time_ms: int,
        *,
        stream_type: str = "mainStream",
    ) -> Optional[int]:
        return self.get_archive_frame_time(
            channel_id,
            time_ms,
            direction="next",
            stream_type=stream_type,
        )

    def get_prev_archive_frame_time(
        self,
        channel_id: int,
        time_ms: int,
        *,
        stream_type: str = "mainStream",
    ) -> Optional[int]:
        return self.get_archive_frame_time(
            channel_id,
            time_ms,
            direction="prev",
            stream_type=stream_type,
        )

    def get_archive_snapshot(
        self,
        channel_id: int,
        time_ms: int,
        *,
        stream_type: str = "mainStream",
    ) -> Image.Image:
        resp = self._request(
            "GET",
            f"/archive/{channel_id}/snapshot",
            params={"time": int(time_ms), "streamType": stream_type},
            headers={"Accept": "image/jpeg"},
            stream=False,
            timeout=max(10, self.timeout),
        )
        return self._decode_jpeg_response(resp, label=f"Luxriot archive snapshot for channel {channel_id}")

    def open_live_stream(
        self,
        channel_id: int,
        *,
        stream: str = "mainStream",
        timeout: Optional[int] = None,
    ) -> requests.Response:
        return self._request(
            "GET",
            f"/live/{channel_id}/{stream}",
            headers={"Accept": "video/mp4,video/webm,application/octet-stream,*/*"},
            stream=True,
            timeout=timeout or max(30, self.timeout),
        )

    def open_archive_stream(
        self,
        channel_id: int,
        time_ms: int,
        *,
        stream_type: str = "mainStream",
        timeout: Optional[int] = None,
    ) -> requests.Response:
        return self._request(
            "GET",
            f"/archive/{channel_id}/stream",
            params={"time": int(time_ms), "streamType": stream_type},
            headers={
                "Accept": "video/mp4,application/octet-stream,*/*",
                "Streaming-Web-Ver": "1.3.0",
            },
            stream=True,
            timeout=timeout or max(30, self.timeout),
        )

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
        run_id: Optional[str] = None,
        run_started_at: Optional[float] = None,
        model_hint: Optional[str] = None,
        interval_override: Optional[float] = None,
        summarization_enabled: bool = True,
        capture_kind: str = "video",
    ) -> None:
        self.manager = manager
        self.channel_id = channel_id
        self.batch_size = batch_size
        self.prompt = prompt
        self.run_id = str(run_id or "").strip()
        self.run_started_at = float(run_started_at) if run_started_at else time.time()
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
        self.recent_frames: List[Dict[str, Any]] = []
        self.recent_max_buffer = max(
            int(self.batch_size),
            min(int(self.max_buffer or 0) or 36, max(36, int(self.batch_size) * 3)),
        )
        self.logs: List[Dict[str, Any]] = []
        self.total_flushes = 0
        self.dropped_frames = 0
        self.queue_submissions = 0
        self.queue_dropped_batches = 0
        self.last_queue_job_id: Optional[str] = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.last_error: Optional[str] = None
        self.snapshot_count = 0
        self.snapshot_failed_count = 0
        self.slow_snapshot_count = 0
        self.last_snapshot_latency_sec: Optional[float] = None
        self.avg_snapshot_latency_sec: Optional[float] = None
        self.max_snapshot_latency_sec: Optional[float] = None
        self.last_snapshot_at: Optional[float] = None
        self.snapshot_slow_threshold_sec = max(2.0, float(self.interval) * 2.0)
        self.capture_source_mode = str(getattr(manager.config, "LUXRIOT_CAPTURE_SOURCE", "auto") or "auto").strip().lower()
        if self.capture_source_mode not in {"auto", "snapshot", "live_segment"}:
            self.capture_source_mode = "auto"
        self.active_capture_source = "snapshot"
        self.live_segment_count = 0
        self.live_segment_failed_count = 0
        self.live_segment_frame_count = 0
        self.last_live_segment_latency_sec: Optional[float] = None
        self.last_live_segment_frames = 0
        self.last_live_segment_error: Optional[str] = None
        self._last_frame_hash: Optional[str] = None
        self._same_frame_started_at: Optional[float] = None
        self._same_frame_count = 0
        self.frozen_signal = False
        self.frozen_signal_since: Optional[float] = None
        self.frozen_frame_count = 0
        self.frozen_frame_hash: Optional[str] = None
        self.frozen_frame_dropped_count = 0

    def start(self) -> None:
        if not self.thread.is_alive():
            self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=0.75)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            loop_started = time.monotonic()
            try:
                if self._should_use_live_segment():
                    handled = self._run_live_segment_once()
                    if not handled and self.capture_source_mode == "auto":
                        self._run_snapshot_once()
                else:
                    self._run_snapshot_once()
            except Exception as exc:
                self.last_error = str(exc)
                self._record_snapshot_result(
                    max(0.0, time.monotonic() - loop_started),
                    success=False,
                )
            elapsed = max(0.0, time.monotonic() - loop_started)
            self.stop_event.wait(max(0.0, float(self.interval) - elapsed))

    def _should_use_live_segment(self) -> bool:
        if self.capture_source_mode == "snapshot":
            return False
        if self.capture_source_mode == "live_segment":
            return True
        with self.lock:
            slow_count = int(self.slow_snapshot_count)
            last_latency = self.last_snapshot_latency_sec
            threshold = float(self.snapshot_slow_threshold_sec)
        return slow_count > 0 or (
            last_latency is not None
            and threshold > 0
            and float(last_latency) >= threshold
        )

    def _run_snapshot_once(self) -> None:
        snapshot_started = time.monotonic()
        snapshot = self.client.get_snapshot(self.channel_id)
        snapshot_latency = max(0.0, time.monotonic() - snapshot_started)
        self._record_snapshot_result(snapshot_latency, success=True)
        self.active_capture_source = "snapshot"
        self._accept_captured_frame(snapshot, int(time.time() * 1000))

    def _run_live_segment_once(self) -> bool:
        ffmpeg_result = self._run_ffmpeg_live_segment_once()
        if ffmpeg_result is not None:
            return ffmpeg_result
        if iter_luxriot_live_segment_frames is None:
            self.last_live_segment_error = "road_events live segment decoder is unavailable"
            if self.capture_source_mode == "live_segment":
                raise RuntimeError(self.last_live_segment_error)
            return False
        segment_seconds = max(
            float(getattr(self.manager.config, "LUXRIOT_LIVE_SEGMENT_SECONDS", 15.0)),
            min(60.0, max(2.0, float(self.batch_size) * max(0.2, float(self.interval)) * 1.15)),
        )
        segment_bytes = int(float(getattr(self.manager.config, "LUXRIOT_LIVE_SEGMENT_MB", 8.0)) * 1024 * 1024)
        every_n = int(getattr(self.manager.config, "LUXRIOT_LIVE_SEGMENT_EVERY_N", 25))
        frame_limit = max(1, int(self.batch_size))
        started = time.monotonic()
        accepted = 0
        try:
            for decoded in iter_luxriot_live_segment_frames(
                self.client,
                self.channel_id,
                stream="mainStream",
                segment_bytes=segment_bytes,
                segment_seconds=segment_seconds,
                every_n=max(1, every_n),
                max_frames=frame_limit,
            ):
                if self.stop_event.is_set():
                    break
                try:
                    image = Image.fromarray(decoded.image).convert("RGB")
                except Exception:
                    continue
                timestamp_ms = int(decoded.timestamp_ms or int(time.time() * 1000))
                self._accept_captured_frame(image, timestamp_ms, summarize=False)
                accepted += 1
            latency = max(0.0, time.monotonic() - started)
            with self.lock:
                self.active_capture_source = "live_segment"
                self.live_segment_count += 1
                self.live_segment_frame_count += accepted
                self.last_live_segment_latency_sec = latency
                self.last_live_segment_frames = accepted
                self.last_live_segment_error = None if accepted > 0 else "live segment produced no decoded frames"
            self._summarize_if_ready()
            return accepted > 0
        except Exception as exc:
            latency = max(0.0, time.monotonic() - started)
            with self.lock:
                self.active_capture_source = "live_segment"
                self.live_segment_failed_count += 1
                self.last_live_segment_latency_sec = latency
                self.last_live_segment_frames = accepted
                self.last_live_segment_error = str(exc)[:240] or exc.__class__.__name__
            if self.capture_source_mode == "live_segment":
                raise
            return False

    def _live_stream_url_with_credentials(self) -> str:
        parsed = urlsplit(str(self.client.base_url or "").rstrip("/"))
        username = quote(str(self.client.username or ""), safe="")
        password = quote(str(self.client.password or ""), safe="")
        netloc = parsed.netloc
        if username or password:
            netloc = f"{username}:{password}@{parsed.netloc}"
        base_path = parsed.path.rstrip("/")
        live_path = f"{base_path}/live/{int(self.channel_id)}/mainStream"
        return urlunsplit((parsed.scheme or "http", netloc, live_path, "", ""))

    def _run_ffmpeg_live_segment_once(self) -> Optional[bool]:
        frame_limit = max(1, int(self.batch_size))
        fps = float(getattr(self.manager.config, "LUXRIOT_LIVE_SEGMENT_FPS", 2.0))
        fps = max(0.2, min(10.0, fps))
        segment_seconds = float(getattr(self.manager.config, "LUXRIOT_LIVE_SEGMENT_SECONDS", 15.0))
        timeout_sec = max(12.0, segment_seconds + 15.0, (float(frame_limit) / fps) + 12.0)
        started = time.monotonic()
        accepted = 0
        try:
            with tempfile.TemporaryDirectory(prefix=f"eva-live-ch{self.channel_id}-") as temp_dir:
                output_pattern = str(Path(temp_dir) / "frame-%04d.jpg")
                cmd = [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-i",
                    self._live_stream_url_with_credentials(),
                    "-vf",
                    f"fps={fps:g}",
                    "-frames:v",
                    str(frame_limit),
                    "-q:v",
                    "4",
                    output_pattern,
                ]
                completed = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout_sec,
                    check=False,
                )
                frame_paths = sorted(Path(temp_dir).glob("frame-*.jpg"))
                base_ts = int(time.time() * 1000) - int((len(frame_paths) / fps) * 1000) if frame_paths else int(time.time() * 1000)
                for index, path in enumerate(frame_paths[:frame_limit]):
                    if self.stop_event.is_set():
                        break
                    try:
                        with Image.open(path) as opened:
                            opened.load()
                            image = opened.convert("RGB")
                    except Exception:
                        continue
                    timestamp_ms = base_ts + int((index / fps) * 1000)
                    self._accept_captured_frame(image, timestamp_ms, summarize=False)
                    accepted += 1
                latency = max(0.0, time.monotonic() - started)
                stderr = str(completed.stderr or "").strip()
                with self.lock:
                    self.active_capture_source = "live_segment"
                    self.live_segment_count += 1 if accepted > 0 else 0
                    self.live_segment_frame_count += accepted
                    self.last_live_segment_latency_sec = latency
                    self.last_live_segment_frames = accepted
                    if accepted > 0:
                        self.last_live_segment_error = None
                    else:
                        self.live_segment_failed_count += 1
                        self.last_live_segment_error = stderr[:240] or f"ffmpeg exited {completed.returncode} without frames"
                self._summarize_if_ready()
                if accepted > 0:
                    return True
                if completed.returncode == 127:
                    return None
                return False
        except FileNotFoundError:
            return None
        except subprocess.TimeoutExpired as exc:
            latency = max(0.0, time.monotonic() - started)
            with self.lock:
                self.active_capture_source = "live_segment"
                self.live_segment_failed_count += 1
                self.last_live_segment_latency_sec = latency
                self.last_live_segment_frames = accepted
                self.last_live_segment_error = f"ffmpeg live segment timed out after {timeout_sec:.1f}s"
            if self.capture_source_mode == "live_segment":
                raise RuntimeError(self.last_live_segment_error) from exc
            return False
        except Exception as exc:
            latency = max(0.0, time.monotonic() - started)
            with self.lock:
                self.active_capture_source = "live_segment"
                self.live_segment_failed_count += 1
                self.last_live_segment_latency_sec = latency
                self.last_live_segment_frames = accepted
                self.last_live_segment_error = str(exc)[:240] or exc.__class__.__name__
            if self.capture_source_mode == "live_segment":
                raise
            return False

    def _record_frame_hash_locked(self, frame_hash: str, observed_at: float) -> bool:
        if self._last_frame_hash != frame_hash:
            self._last_frame_hash = frame_hash
            self._same_frame_started_at = observed_at
            self._same_frame_count = 1
            self.frozen_signal = False
            self.frozen_signal_since = None
            self.frozen_frame_count = 0
            self.frozen_frame_hash = None
            return False

        if self._same_frame_started_at is None:
            self._same_frame_started_at = observed_at
        self._same_frame_count += 1
        frozen_after_sec = max(5.0, float(getattr(self.manager.config, "LUXRIOT_FROZEN_FRAME_MAX_SEC", 20.0)))
        frozen_min_count = max(2, int(getattr(self.manager.config, "LUXRIOT_FROZEN_FRAME_MIN_COUNT", 3)))
        frozen_duration = max(0.0, float(observed_at) - float(self._same_frame_started_at))
        if self._same_frame_count >= frozen_min_count and frozen_duration >= frozen_after_sec:
            if not self.frozen_signal:
                self.frozen_signal_since = observed_at
            self.frozen_signal = True
            self.frozen_frame_count = int(self._same_frame_count)
            self.frozen_frame_hash = frame_hash
        return self.frozen_signal

    def _accept_captured_frame(self, snapshot: Image.Image, timestamp_ms: int, *, summarize: bool = True) -> None:
        captured_at = max(0.0, float(timestamp_ms) / 1000.0)
        thumbnail = self.manager.jpeg_encoder(snapshot, max_edge=self.max_edge, quality=85)
        frame_hash = hashlib.sha1(str(thumbnail or "").encode("ascii", errors="ignore")).hexdigest()[:16]
        observed_at = time.time()
        frame = {
            "thumbnail": thumbnail,
            "captured_at": captured_at,
            "time_sec": captured_at,
            "width": snapshot.width,
            "height": snapshot.height,
            "frame_hash": frame_hash,
        }
        with self.lock:
            frozen_now = self._record_frame_hash_locked(frame_hash, observed_at)
            if frozen_now:
                self.frozen_frame_dropped_count += 1
                return
            self.frames.append(frame)
            self.recent_frames.append(frame)
            self._enforce_buffer_locked()
        try:
            probe_manager = self.manager.probe_manager
            if probe_manager is not None:
                probe_manager.add_frame(self.channel_id, snapshot, int(timestamp_ms))
        except Exception as pm_exc:
            self.last_error = str(pm_exc)
        if summarize:
            self._summarize_if_ready()

    def _summarize_if_ready(self) -> None:
        with self.lock:
            ready_to_summarize = len(self.frames) >= self.batch_size
        summarized_ok = True
        if self.summarization_enabled and ready_to_summarize:
            summarized_ok = self._summarize_batch()
        if summarized_ok:
            self.last_error = None

    def _record_snapshot_result(self, latency_sec: float, *, success: bool) -> None:
        latency = max(0.0, float(latency_sec))
        with self.lock:
            if success:
                self.snapshot_count += 1
                previous = float(self.avg_snapshot_latency_sec or 0.0)
                count = max(1, int(self.snapshot_count))
                self.avg_snapshot_latency_sec = (
                    latency if count == 1 else previous + ((latency - previous) / float(count))
                )
                current_max = float(self.max_snapshot_latency_sec or 0.0)
                self.max_snapshot_latency_sec = max(current_max, latency)
                self.last_snapshot_at = time.time()
            else:
                self.snapshot_failed_count += 1
            self.last_snapshot_latency_sec = latency
            if latency >= float(self.snapshot_slow_threshold_sec):
                self.slow_snapshot_count += 1

    def _summarize_batch(self, workload_class: str = "heartbeat") -> bool:
        with self.lock:
            frames_copy = list(self.frames)
            self.frames.clear()
        if not frames_copy:
            return True
        try:
            batch = self.manager.create_summary_batch(
                channel_id=self.channel_id,
                run_id=self.run_id,
                batch_size=self.batch_size,
                prompt=self.prompt,
                model_hint=self.model_hint,
                interval_sec=self.interval,
                frames=frames_copy,
            )
            outcome = self.manager.dispatch_summary_batch(
                batch,
                workload_class=workload_class,
            )
            if bool(outcome.get("queued")):
                with self.lock:
                    self.queue_submissions += 1
                    self.last_queue_job_id = (
                        str(outcome.get("job_id") or "").strip() or None
                    )
                    if not bool(outcome.get("accepted", True)):
                        self.queue_dropped_batches += 1
            return True
        except Exception as exc:
            self.last_error = str(exc)
            with self.lock:
                self.frames = frames_copy + self.frames
                self._enforce_buffer_locked()
                self.queue_dropped_batches += 1
            return False

    def _enforce_buffer_locked(self) -> None:
        """Ensure frame buffer does not grow unbounded."""
        if self.max_buffer and len(self.frames) > self.max_buffer:
            overflow = len(self.frames) - self.max_buffer
            # Drop oldest frames to cap size; keep last max_buffer frames
            self.frames = self.frames[-self.max_buffer :]
            self.dropped_frames += overflow
        if self.recent_max_buffer and len(self.recent_frames) > self.recent_max_buffer:
            self.recent_frames = self.recent_frames[-self.recent_max_buffer :]

    def flush_now(self) -> None:
        """Force a summary of current buffer."""
        if self.summarization_enabled:
            self._summarize_batch(workload_class="manual")

    def status(self) -> Dict[str, Any]:
        with self.lock:
            logs_copy = list(self.logs)
            pending_frames = len(self.frames)
            recent_frame_count = len(self.recent_frames)
            last_snapshot_latency_sec = self.last_snapshot_latency_sec
            avg_snapshot_latency_sec = self.avg_snapshot_latency_sec
            max_snapshot_latency_sec = self.max_snapshot_latency_sec
            snapshot_count = self.snapshot_count
            snapshot_failed_count = self.snapshot_failed_count
            slow_snapshot_count = self.slow_snapshot_count
            last_snapshot_at = self.last_snapshot_at
            active_capture_source = self.active_capture_source
            live_segment_count = self.live_segment_count
            live_segment_failed_count = self.live_segment_failed_count
            live_segment_frame_count = self.live_segment_frame_count
            last_live_segment_latency_sec = self.last_live_segment_latency_sec
            last_live_segment_frames = self.last_live_segment_frames
            last_live_segment_error = self.last_live_segment_error
            frozen_signal = self.frozen_signal
            frozen_signal_since = self.frozen_signal_since
            frozen_frame_count = self.frozen_frame_count
            frozen_frame_hash = self.frozen_frame_hash
            frozen_frame_dropped_count = self.frozen_frame_dropped_count
        return {
            "running": not self.stop_event.is_set() and self.thread.is_alive(),
            "channel_id": self.channel_id,
            "run_id": self.run_id,
            "run_started_at": self.run_started_at,
            "batch_size": self.batch_size,
            "pending_frames": pending_frames,
            "recent_frame_count": recent_frame_count,
            "interval_sec": self.interval,
            "max_edge": self.max_edge,
            "max_buffer_frames": self.max_buffer,
            "capture_kind": self.capture_kind,
            "capture_source_mode": self.capture_source_mode,
            "active_capture_source": active_capture_source,
            "summarization_enabled": self.summarization_enabled,
            "dropped_frames": self.dropped_frames,
            "flush_count": self.total_flushes,
            "queue_submissions": self.queue_submissions,
            "queue_dropped_batches": self.queue_dropped_batches,
            "last_queue_job_id": self.last_queue_job_id,
            "last_error": self.last_error,
            "snapshot_count": snapshot_count,
            "snapshot_failed_count": snapshot_failed_count,
            "slow_snapshot_count": slow_snapshot_count,
            "snapshot_slow_threshold_sec": round(float(self.snapshot_slow_threshold_sec), 3),
            "last_snapshot_latency_sec": round(float(last_snapshot_latency_sec), 3)
            if last_snapshot_latency_sec is not None
            else None,
            "avg_snapshot_latency_sec": round(float(avg_snapshot_latency_sec), 3)
            if avg_snapshot_latency_sec is not None
            else None,
            "max_snapshot_latency_sec": round(float(max_snapshot_latency_sec), 3)
            if max_snapshot_latency_sec is not None
            else None,
            "last_snapshot_at": last_snapshot_at,
            "live_segment_count": live_segment_count,
            "live_segment_failed_count": live_segment_failed_count,
            "live_segment_frame_count": live_segment_frame_count,
            "last_live_segment_latency_sec": round(float(last_live_segment_latency_sec), 3)
            if last_live_segment_latency_sec is not None
            else None,
            "last_live_segment_frames": last_live_segment_frames,
            "last_live_segment_error": last_live_segment_error,
            "frozen_signal": frozen_signal,
            "frozen_signal_since": frozen_signal_since,
            "frozen_signal_age_sec": (
                round(max(0.0, time.time() - float(frozen_signal_since)), 3)
                if frozen_signal_since is not None
                else None
            ),
            "frozen_frame_count": frozen_frame_count,
            "frozen_frame_hash": frozen_frame_hash,
            "frozen_frame_dropped_count": frozen_frame_dropped_count,
            "logs": logs_copy,
            "prompt": self.prompt,
            "model": self.model_hint,
        }

    def nearest_frame_thumbnail(self, timestamp_ms: Optional[int] = None) -> Optional[str]:
        with self.lock:
            frames_copy = list(self.recent_frames or self.frames)
        if not frames_copy:
            return None
        if timestamp_ms is None:
            raw = frames_copy[-1].get("thumbnail")
            value = str(raw or "").strip()
            return value or None
        target_sec = float(timestamp_ms) / 1000.0
        best = min(
            frames_copy,
            key=lambda frame: abs(float(frame.get("time_sec") or frame.get("captured_at") or 0.0) - target_sec),
        )
        raw = best.get("thumbnail")
        value = str(raw or "").strip()
        return value or None

    def recent_frame_items(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self.lock:
            frames_copy = list(self.recent_frames or self.frames)
        if isinstance(limit, int) and limit > 0:
            return frames_copy[-limit:]
        return frames_copy


class LuxriotManager:
    """Coordinator for Luxriot snapshots, summaries, and channel helpers."""

    DESIRED_LIVE_SESSIONS_KEY = "luxriot_live_sessions:v1"

    @staticmethod
    def _normalize_json_alert_prompt(prompt_text: object) -> str:
        text = str(prompt_text or "").strip()
        if not text:
            return DEFAULT_ALERTS_JSON_PROMPT
        lowered = text.lower()
        if any(marker in lowered for marker in _OUTDATED_ALERT_PROMPT_MARKERS):
            return DEFAULT_ALERTS_JSON_PROMPT
        if (
            "do not merge unrelated triggers into one alert" in lowered
            and "evaluate every operator-defined trigger independently" not in lowered
        ):
            return DEFAULT_ALERTS_JSON_PROMPT
        return text

    @staticmethod
    def _render_channel_memory_prompt(routine_text: str) -> str:
        routine = str(routine_text or "").strip()
        if not routine:
            return ""
        return (
            "Active Channel Memory - Prior Context (not a current observation):\n"
            f"{routine}\n"
            "Use this memory only as prior context for routine-vs-deviation judgment. "
            "Current snapshots override this memory. Do not assert that a person, animal, vehicle, "
            "object, or action is present from memory alone; verify presence in the current snapshots. "
            "Do not let routine baseline suppress visible security/safety alerts or operator-defined triggers. "
            "Preserve new deviations, concrete operator-review incidents, and alert tuning signals."
        )

    @staticmethod
    def _strip_suffix_prompt(prompt_text: str, suffix: str) -> str:
        prompt = str(prompt_text or "").rstrip()
        rendered_suffix = str(suffix or "").strip()
        if not prompt or not rendered_suffix:
            return prompt
        if prompt.endswith(rendered_suffix):
            return prompt[: len(prompt) - len(rendered_suffix)].rstrip()
        return prompt

    @staticmethod
    def _render_alert_policy_prompt(prompt_text: object) -> str:
        criteria = str(prompt_text or "").strip()
        rendered_criteria = criteria if criteria else "None provided. Use only the general safety/security hazards above."
        return DEFAULT_ALERT_POLICY_PROMPT.replace("{operator_alert_policy}", rendered_criteria).strip()

    @staticmethod
    def _compact_prompt_text(text: object) -> str:
        lines = [line.rstrip() for line in str(text or "").splitlines()]
        compacted: List[str] = []
        blank = False
        for line in lines:
            if not line.strip():
                if not blank and compacted:
                    compacted.append("")
                blank = True
                continue
            compacted.append(line)
            blank = False
        return "\n".join(compacted).strip()

    @staticmethod
    def _strip_prompt_bullet(line: str) -> str:
        return re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", str(line or "")).strip()

    @classmethod
    def _legacy_alert_prompt_health(
        cls,
        stream_prompt: object,
        alert_policy_prompt: object,
    ) -> Dict[str, Any]:
        stream_text = str(stream_prompt or "")
        current_policy = str(alert_policy_prompt or "").strip()
        if not stream_text.strip():
            return {
                "needs_migration": False,
                "legacy_prose_alert_format": False,
                "legacy_alert_criteria_in_stream": False,
                "warnings": [],
            }

        legacy_format = False
        legacy_criteria = False
        candidate_policy_lines: List[str] = []
        cleaned_lines: List[str] = []
        for raw_line in stream_text.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            lowered = stripped.lower()
            is_json_contract = "alerts_json" in lowered or '"alerts"' in lowered
            is_prose_alert_heading = bool(
                re.match(r"^\s*(?:#{1,6}\s*)?(?:alerts?|alerts/signals)\s*:?\s*$", stripped, flags=re.IGNORECASE)
            )
            is_prose_level_line = bool(
                re.match(
                    r"^\s*(?:[-*•]|\d+[.)])?\s*"
                    r"(?:info(?:rmation(?:al)?)?|low|warn(?:ing)?|normal|moderate|high|critical|danger|emergency)"
                    r"\s*(?:level|alert|severity)\s*[:\-–]",
                    stripped,
                    flags=re.IGNORECASE,
                )
            )
            is_alert_section_instruction = bool(
                re.search(r"\breturn\s+markdown\b", lowered)
                and re.search(r"\balerts?(?:/signals)?\b", lowered)
            )
            criteria_hint = bool(
                re.search(
                    r"\b(alert|alerts|trigger|triggers|watch|flag|bookmark|notify|raise|pay attention|look for|monitor)\b",
                    lowered,
                )
            )
            if is_json_contract or is_prose_alert_heading or is_prose_level_line or is_alert_section_instruction:
                legacy_format = True
                continue
            if criteria_hint:
                clean_candidate = cls._strip_prompt_bullet(line)
                if clean_candidate and clean_candidate not in candidate_policy_lines:
                    candidate_policy_lines.append(clean_candidate)
                legacy_criteria = True
                continue
            cleaned_lines.append(line)

        suggested_policy = current_policy
        if candidate_policy_lines:
            existing_lower = suggested_policy.lower()
            new_lines = [
                line
                for line in candidate_policy_lines
                if line.lower() not in existing_lower
            ]
            if new_lines:
                suggested_policy = cls._compact_prompt_text(
                    "\n".join([part for part in (suggested_policy, "\n".join(f"- {line}" for line in new_lines)) if part.strip()])
                )
        suggested_stream = cls._compact_prompt_text("\n".join(cleaned_lines))
        needs_migration = bool(legacy_format or legacy_criteria)
        warnings: List[str] = []
        if legacy_format:
            warnings.append(
                "Legacy prose alert formatting was detected in the stream prompt. "
                "Move alert output requirements to the backend ALERTS_JSON contract."
            )
        if legacy_criteria:
            warnings.append(
                "Alert/watch criteria were detected inside the stream prompt. "
                "Move them to Alert Criteria so L0 role text does not compete with machine-readable alerts."
            )
        return {
            "needs_migration": needs_migration,
            "legacy_prose_alert_format": legacy_format,
            "legacy_alert_criteria_in_stream": legacy_criteria,
            "warnings": warnings,
            "candidate_alert_policy_lines": candidate_policy_lines[:24],
            "suggested_stream_system_prompt": suggested_stream,
            "suggested_alert_policy_prompt": suggested_policy,
        }

    def __init__(
        self,
        config: Any,
        lm_callback: Callable[[List[Dict[str, Any]], Optional[str]], str],
        message_builder: Callable[[str, List[Dict[str, Any]], str, str], List[Dict[str, Any]]],
        jpeg_encoder: Callable[..., str],
        alert_parser: Optional[AlertParserFn] = None,
        probe_manager: Optional[ProbeManagerLike] = None,
        runtime_state_store: Optional[Any] = None,
        summary_archive_callback: Optional[SummaryArchiveFn] = None,
    ) -> None:
        self.config = config
        self.lm_callback = lm_callback
        self.message_builder = message_builder
        self.jpeg_encoder = jpeg_encoder
        self.alert_parser = alert_parser
        self.probe_manager: Optional[ProbeManagerLike] = probe_manager
        self.probes_store: Optional[ProbeStoreLike] = None
        self.runtime_state_store = runtime_state_store
        self.summary_dispatcher: Optional[SummaryDispatcherFn] = None
        self.summary_archive_callback: Optional[SummaryArchiveFn] = summary_archive_callback
        self.system_prompt = getattr(config, "LUXRIOT_SYSTEM_PROMPT_DEFAULT", "")
        self.alert_policy_prompt = str(getattr(config, "LUXRIOT_ALERT_POLICY_PROMPT", "") or "")

        self.sessions: Dict[int, LuxriotCaptureSession] = {}
        self.probe_sessions: Dict[int, LuxriotCaptureSession] = {}
        self.shared_probe_channels: Set[int] = set()
        self.paused_probe_channels: Set[int] = set()
        self.cache_lock = threading.Lock()
        self.channels_cache: Optional[Tuple[float, List[Dict[str, Any]]]] = None
        try:
            history_limit = int(getattr(config, "LUXRIOT_SUMMARY_HISTORY_LIMIT", 600))
        except Exception:
            history_limit = 600
        self.summary_history_limit = max(40, history_limit)
        try:
            archive_frames_per_batch = int(
                getattr(config, "LUXRIOT_SUMMARY_ARCHIVE_FRAMES_PER_BATCH", 4)
            )
        except Exception:
            archive_frames_per_batch = 4
        self.summary_archive_frames_per_batch = max(1, min(16, archive_frames_per_batch))
        try:
            summary_retention_days = float(getattr(config, "LUXRIOT_SUMMARY_RETENTION_DAYS", 7.0))
        except Exception:
            summary_retention_days = 7.0
        self.summary_retention_days = max(0.0, summary_retention_days)
        self.summary_history: Dict[int, List[Dict[str, Any]]] = {}
        self.channel_status_digest: Dict[int, Dict[str, Any]] = {}
        self.summary_runs: Dict[int, List[Dict[str, Any]]] = {}
        self.active_summary_runs: Dict[int, str] = {}
        self.channel_routine_context: Dict[int, Dict[str, Any]] = {}
        self.channel_observed_state_tracker: Dict[int, Dict[str, Dict[str, Any]]] = {}
        self.channel_prompt_overrides: Dict[int, Dict[str, Any]] = {}
        self._summary_state_last_persist = 0.0
        self._summary_state_dirty = False
        try:
            persist_interval = float(getattr(config, "LUXRIOT_SUMMARY_STATE_PERSIST_INTERVAL_SEC", 15.0))
        except Exception:
            persist_interval = 15.0
        self.summary_state_persist_interval_sec = max(0.0, persist_interval)
        self.live_session_restore_errors: Dict[int, str] = {}
        self.channel_bookmark_fingerprints: Dict[int, Dict[str, int]] = {}
        self.default_bookmark_enabled = bool(getattr(config, "LUXRIOT_AUTO_BOOKMARKS", False))
        try:
            cooldown_value = float(getattr(config, "LUXRIOT_BOOKMARK_COOLDOWN_SEC", 60.0))
        except Exception:
            cooldown_value = 60.0
        self.default_bookmark_cooldown_sec = max(0.0, cooldown_value)
        try:
            max_alerts_value = int(getattr(config, "LUXRIOT_ALERTS_MAX_PER_BATCH", 8))
        except Exception:
            max_alerts_value = 8
        self.alerts_max_per_batch = max(1, min(32, max_alerts_value))
        self.default_json_alert_prompt = self._normalize_json_alert_prompt(
            getattr(config, "LUXRIOT_ALERTS_JSON_PROMPT", DEFAULT_ALERTS_JSON_PROMPT)
        )
        self.state_transitions_enabled = bool(
            getattr(config, "LUXRIOT_STATE_TRANSITIONS_ENABLED", True)
        )
        try:
            confirm_batches = int(getattr(config, "LUXRIOT_STATE_TRANSITION_CONFIRM_BATCHES", 2))
        except Exception:
            confirm_batches = 2
        self.state_transition_confirm_batches = max(1, min(6, confirm_batches))
        self.state_transition_alert_events_enabled = bool(
            getattr(config, "LUXRIOT_STATE_TRANSITION_ALERT_EVENTS", True)
        )
        self.vector_signals_enabled = bool(
            getattr(config, "LUXRIOT_VECTOR_SIGNALS_ENABLED", True)
        )
        try:
            vector_probe_limit = int(getattr(config, "LUXRIOT_VECTOR_SIGNAL_PROBE_LIMIT", 6))
        except Exception:
            vector_probe_limit = 6
        self.vector_signal_probe_limit = max(0, min(16, vector_probe_limit))
        try:
            vector_top_hits = int(getattr(config, "LUXRIOT_VECTOR_SIGNAL_TOP_HITS", 2))
        except Exception:
            vector_top_hits = 2
        self.vector_signal_top_hits = max(1, min(5, vector_top_hits))
        self.road_cv_batch_signals_enabled = bool(
            getattr(config, "LUXRIOT_ROAD_CV_BATCH_SIGNALS", True)
        )
        try:
            road_cv_max_frames = int(getattr(config, "LUXRIOT_ROAD_CV_BATCH_MAX_FRAMES", 24))
        except Exception:
            road_cv_max_frames = 24
        self.road_cv_batch_max_frames = max(4, min(48, road_cv_max_frames))
        try:
            road_cv_max_edge = int(getattr(config, "LUXRIOT_ROAD_CV_BATCH_MAX_EDGE", 240))
        except Exception:
            road_cv_max_edge = 240
        self.road_cv_batch_max_edge = max(96, min(480, road_cv_max_edge))
        try:
            self.lm_input_warning_chars = int(getattr(config, "LM_VIDEO_INPUT_WARNING_CHARS", 24000))
        except (TypeError, ValueError):
            self.lm_input_warning_chars = 24000
        self.lm_input_warning_chars = max(1, self.lm_input_warning_chars)
        try:
            self.lm_image_payload_warning_chars = int(
                getattr(config, "LM_VIDEO_IMAGE_PAYLOAD_WARNING_CHARS", 2500000)
            )
        except (TypeError, ValueError):
            self.lm_image_payload_warning_chars = 2500000
        self.lm_image_payload_warning_chars = max(1, self.lm_image_payload_warning_chars)
        self.rollup_time_only = bool(getattr(config, "LUXRIOT_ROLLUP_TIME_ONLY", True))
        summary_state_raw = str(getattr(config, "LUXRIOT_SUMMARY_STATE_FILE", "luxriot_summary_state.json") or "").strip()
        if not summary_state_raw:
            summary_state_raw = "luxriot_summary_state.json"
        summary_state_path = Path(summary_state_raw).expanduser()
        if not summary_state_path.is_absolute():
            summary_state_path = Path.cwd() / summary_state_path
        self.summary_state_file = summary_state_path
        try:
            l1_window = int(getattr(config, "LUXRIOT_ROLLUP_L1_WINDOW_SEC", 900))
        except Exception:
            l1_window = 900
        try:
            l2_window = int(getattr(config, "LUXRIOT_ROLLUP_L2_WINDOW_SEC", 3600))
        except Exception:
            l2_window = 3600
        try:
            l3_window = int(getattr(config, "LUXRIOT_ROLLUP_L3_WINDOW_SEC", 21600))
        except Exception:
            l3_window = 21600
        self.rollup_windows: Dict[str, int] = {
            "L1": max(300, l1_window),
            "L2": max(900, l2_window),
            "L3": max(1800, l3_window),
        }
        try:
            self.rollup_highlight_limit = max(1, int(getattr(config, "LUXRIOT_ROLLUP_HIGHLIGHTS", 3)))
        except Exception:
            self.rollup_highlight_limit = 3
        # Backward-compatible single-level flag is still supported, but we default to all levels.
        legacy_l1_enabled = bool(getattr(config, "LUXRIOT_ROLLUP_L1_LLM_ENABLED", True))
        llm_levels_raw = str(getattr(config, "LUXRIOT_ROLLUP_LLM_LEVELS", "L1,L2,L3") or "")
        llm_level_tokens = [token.strip().upper() for token in llm_levels_raw.split(",") if token.strip()]
        allowed_levels = {"L1", "L2", "L3"}
        if any(token in {"NONE", "OFF"} for token in llm_level_tokens):
            self.rollup_llm_levels: Set[str] = set()
        else:
            parsed_levels = set(llm_level_tokens)
            self.rollup_llm_levels = parsed_levels.intersection(allowed_levels) if parsed_levels else {"L1", "L2", "L3"}
        if not legacy_l1_enabled and "L1" in self.rollup_llm_levels:
            self.rollup_llm_levels.discard("L1")
        try:
            self.rollup_min_source_tokens = int(getattr(config, "LUXRIOT_ROLLUP_MIN_SOURCE_TOKENS", 8000))
        except Exception:
            self.rollup_min_source_tokens = 8000
        self.rollup_min_source_tokens = max(512, self.rollup_min_source_tokens)
        try:
            self.rollup_llm_char_budget = int(
                getattr(
                    config,
                    "LUXRIOT_ROLLUP_LLM_CHAR_BUDGET",
                    getattr(config, "LUXRIOT_ROLLUP_L1_CHAR_BUDGET", 12000),
                )
            )
        except Exception:
            self.rollup_llm_char_budget = 12000
        self.rollup_llm_char_budget = max(2000, self.rollup_llm_char_budget)
        try:
            self.rollup_llm_max_new_per_call = int(
                getattr(
                    config,
                    "LUXRIOT_ROLLUP_LLM_MAX_NEW_PER_CALL",
                    getattr(config, "LUXRIOT_ROLLUP_L1_MAX_NEW_PER_CALL", 2),
                )
            )
        except Exception:
            self.rollup_llm_max_new_per_call = 2
        self.rollup_llm_max_new_per_call = max(1, self.rollup_llm_max_new_per_call)
        try:
            self.rollup_summary_cache_limit = int(getattr(config, "LUXRIOT_ROLLUP_SUMMARY_CACHE_LIMIT", 800))
        except Exception:
            self.rollup_summary_cache_limit = 800
        self.rollup_summary_cache_limit = max(100, self.rollup_summary_cache_limit)
        self.rollup_llm_model_hint = (
            str(
                getattr(
                    config,
                    "LUXRIOT_ROLLUP_LLM_MODEL",
                    getattr(config, "LUXRIOT_ROLLUP_L1_MODEL", ""),
                )
                or ""
            ).strip()
            or None
        )
        default_rollup_system_prompt = (
            "You are a CCTV operations analyst. Summarize lower-level notes into operator-facing window reports. "
            "Use structured Markdown sections, keep concrete timestamps/events, and avoid repetitive rollup wording."
        )
        self.rollup_llm_system_prompt = str(
            getattr(
                config,
                "LUXRIOT_ROLLUP_LLM_SYSTEM_PROMPT",
                getattr(
                    config,
                    "LUXRIOT_ROLLUP_L1_SYSTEM_PROMPT",
                    "",
                )
                or default_rollup_system_prompt,
            )
            or ""
        ).strip()
        if not self.rollup_llm_system_prompt:
            self.rollup_llm_system_prompt = default_rollup_system_prompt
        self.rollup_llm_system_prompts: Dict[str, str] = {}
        for level in ("L1", "L2", "L3"):
            configured = str(getattr(config, f"LUXRIOT_ROLLUP_{level}_SYSTEM_PROMPT", "") or "").strip()
            self.rollup_llm_system_prompts[level] = configured or self.rollup_llm_system_prompt
        self.rollup_summary_cache: Dict[str, Dict[str, Any]] = {}
        cache_file_raw = str(getattr(config, "LUXRIOT_ROLLUP_CACHE_FILE", "luxriot_rollups_cache.json") or "").strip()
        if not cache_file_raw:
            cache_file_raw = "luxriot_rollups_cache.json"
        cache_path = Path(cache_file_raw).expanduser()
        if not cache_path.is_absolute():
            cache_path = Path.cwd() / cache_path
        self.rollup_cache_file = cache_path
        self._load_summary_state_from_disk()
        self._load_rollup_cache_from_disk()

    @staticmethod
    def _summary_log_key(log: Mapping[str, Any]) -> Tuple[str, str, str, str]:
        created_num = LuxriotManager._coerce_float(log.get("created_at"))
        created = f"{created_num:.6f}" if created_num is not None else "0.000000"
        run_id = str(log.get("run_id") or "").strip()
        frame_count = str(log.get("frame_count") or "")
        summary = str(log.get("summary") or "").strip()
        return (created, run_id, frame_count, summary[:160])

    @staticmethod
    def _summary_log_bounds_seconds(log: Mapping[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        created = LuxriotManager._coerce_float(log.get("created_at"))
        fallback_ms = int(float(created) * 1000.0) if created is not None else None
        start_ms = _parse_optional_int(log.get("batch_start_ms"))
        end_ms = _parse_optional_int(log.get("batch_end_ms"))
        if start_ms is None:
            start_ms = end_ms if end_ms is not None else fallback_ms
        if end_ms is None:
            end_ms = start_ms if start_ms is not None else fallback_ms
        if start_ms is None or end_ms is None:
            return None, None
        if end_ms < start_ms:
            start_ms, end_ms = end_ms, start_ms
        return float(start_ms) / 1000.0, float(end_ms) / 1000.0

    @classmethod
    def _combine_summary_logs(
        cls,
        history_logs: Sequence[Mapping[str, Any]],
        current_logs: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        ordered: List[Tuple[str, str, str, str]] = []
        for item in list(history_logs) + list(current_logs):
            if not isinstance(item, Mapping):
                continue
            key = cls._summary_log_key(item)
            if key not in merged:
                ordered.append(key)
            incoming = dict(item)
            existing = merged.get(key)
            if existing is not None:
                existing_meta = cls._alert_meta_from_counts(existing.get("alert_counts"))
                incoming_meta = cls._alert_meta_from_counts(incoming.get("alert_counts"))
                existing_total = int(existing_meta.get("alert_total") or 0)
                incoming_total = int(incoming_meta.get("alert_total") or 0)
                if existing_total > 0 and incoming_total <= 0:
                    incoming["alert_counts"] = dict(existing_meta.get("alert_counts") or {})
                    incoming["alert_total"] = existing_total
                    incoming["alert_severities"] = list(existing_meta.get("alert_severities") or [])
                if isinstance(existing.get("signal_digest"), Mapping) and not isinstance(incoming.get("signal_digest"), Mapping):
                    incoming["signal_digest"] = dict(cast(Mapping[str, Any], existing.get("signal_digest")))
                cls._preserve_summary_provenance_on_merge(existing, incoming)
            merged[key] = incoming
        ordered.sort(key=lambda key: float(key[0]))
        return [merged[key] for key in ordered]

    @staticmethod
    def _has_non_empty_sequence_field(item: Mapping[str, Any], key: str) -> bool:
        value = item.get(key)
        return bool(isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) > 0)

    @staticmethod
    def _has_positive_count_field(item: Mapping[str, Any], key: str) -> bool:
        return int(_parse_optional_int(item.get(key)) or 0) > 0

    @classmethod
    def _preserve_summary_provenance_on_merge(
        cls,
        existing: Mapping[str, Any],
        incoming: Dict[str, Any],
    ) -> None:
        for key in ("alert_events", "state_observations", "state_transition_events"):
            if cls._has_non_empty_sequence_field(existing, key) and not cls._has_non_empty_sequence_field(incoming, key):
                incoming[key] = list(cast(Sequence[Any], existing.get(key)))
        for key in (
            "parser_alert_count",
            "json_alert_count",
            "prose_alert_count",
            "bookmark_failed_count",
            "bookmark_skipped_duplicate_count",
            "bookmark_cooldown_skipped_count",
            "state_transition_total",
            "bookmarks_sent",
            "alerts_parsed",
        ):
            if cls._has_positive_count_field(existing, key) and not cls._has_positive_count_field(incoming, key):
                incoming[key] = existing.get(key)
        for key in ("bookmark_last_error", "alert_parser_error"):
            if str(existing.get(key) or "").strip() and not str(incoming.get(key) or "").strip():
                incoming[key] = existing.get(key)
        for key in ("llm_input_stats", "alert_delivery_breakdown", "alert_parser_breakdown", "vector_signal"):
            if isinstance(existing.get(key), Mapping) and not isinstance(incoming.get(key), Mapping):
                incoming[key] = dict(cast(Mapping[str, Any], existing.get(key)))

    @staticmethod
    def _normalize_alert_severity(value: Any) -> str:
        severity = str(value or "").strip().lower()
        severity_aliases = {
            "information": "info",
            "informational": "info",
            "warn": "low",
            "warning": "low",
            "medium": "normal",
            "moderate": "normal",
            "danger": "high",
            "emergency": "critical",
        }
        severity = severity_aliases.get(severity, severity)
        return severity if severity in ALERT_SEVERITY_SET else "normal"

    @classmethod
    def _normalize_alert_counts(cls, raw_counts: Any) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        if isinstance(raw_counts, Mapping):
            for raw_severity, raw_count in raw_counts.items():
                severity = cls._normalize_alert_severity(raw_severity)
                count = _parse_optional_int(raw_count) or 0
                if count > 0:
                    counts[severity] = counts.get(severity, 0) + count
        return {
            severity: int(counts[severity])
            for severity in ALERT_SEVERITY_ORDER
            if counts.get(severity, 0) > 0
        }

    @classmethod
    def _alert_meta_from_counts(cls, raw_counts: Any) -> Dict[str, Any]:
        counts = cls._normalize_alert_counts(raw_counts)
        total = int(sum(counts.values()))
        return {
            "alert_counts": counts,
            "alert_total": total,
            "alert_severities": [severity for severity in ALERT_SEVERITY_ORDER if counts.get(severity, 0) > 0],
        }

    @classmethod
    def _format_alert_counts(cls, raw_counts: Any) -> str:
        counts = cls._normalize_alert_counts(raw_counts)
        if not counts:
            return ""
        return ", ".join(
            f"{severity}={counts[severity]}"
            for severity in ALERT_SEVERITY_ORDER
            if counts.get(severity, 0) > 0
        )

    @classmethod
    def _rollup_backend_instruction_lines(cls, level: str) -> List[str]:
        normalized_level = cls._normalize_rollup_level(level) or str(level or "").strip().upper() or "rollup"
        return [
            "Context constraints:",
            "- All source entries are from the same channel and continuous timeline window above.",
            "- Source entries may be model-generated summaries from a lower level; avoid compounding uncertainty.",
            "- Window signal digest is a compact routing map for alerts, deviations, watch items, missing data, and uncertainty; use it to keep continuity, but treat source summaries as evidence.",
            "- Preserve rare but important events even if they appear once.",
            "- Keep numeric facts aligned with metadata above (item_count/frame_count/window).",
            "- Never compress alerts, deviations, or operator-review incidents into routine.",
            "- Alert Ledger must mention every source alert/count, including normal/info alerts, even when the event is routine or needs no action.",
            "- Do not classify behavior as illegal/unlawful; describe observable security/safety facts.",
            "",
            "Task:",
            f"- Write one concise {normalized_level} summary for operators.",
            "- Deduplicate repeated scene descriptions and boilerplate.",
            "- Keep meaningful changes in short timeline bullets across the full window.",
            "- Mention risks/signals only when grounded in source text.",
            "- If activity is routine, say so clearly without repeating identical details.",
            "- Keep routine baseline separate from preserved deviations and alert ledger.",
            "- Do not invent entities, times, or counts.",
            "",
            "Output format (Markdown):",
            "### Window Snapshot",
            "### Routine Baseline",
            "### Preserved Deviations",
            "### Alert Ledger",
            "### Alert Tuning Notes",
            "### Alerts/Signals",
            "### Operator Notes",
            "",
            "Append exactly one compact machine-readable memory block:",
            "MEMORY_UPDATE_JSON:",
            "{",
            "  \"routine_baseline\": \"normal pattern for this channel, if grounded\",",
            "  \"active_watchlist\": [\"short-lived items to watch in the next window\"],",
            "  \"preserved_deviations\": [",
            "    {\"time\": \"HH:MM-HH:MM\", \"severity\": \"info|low|normal|high|critical\", \"event\": \"observable event\", \"evidence\": \"visible evidence\"}",
            "  ],",
            "  \"alert_tuning_notes\": [\"what should/should not trigger next time\"],",
            "  \"ignore_as_routine\": [\"recurring benign activity/noise\"]",
            "}",
            "Rules for MEMORY_UPDATE_JSON: keep every field concise; use [] or \"\" when absent; preserve real deviations even if the window is mostly routine.",
        ]

    @classmethod
    def _rollup_backend_instruction_text(cls, level: str) -> str:
        return "\n".join(cls._rollup_backend_instruction_lines(level)).strip()

    def _summary_alert_metadata(
        self,
        summary_text: str,
        *,
        channel_id: int,
        timestamp_ms: int,
        fallback: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if fallback is not None:
            counts = self._normalize_alert_counts(fallback.get("alert_counts"))
            if counts:
                return self._alert_meta_from_counts(counts)
            raw_total = _parse_optional_int(fallback.get("alert_total")) or 0
            if raw_total > 0:
                severity = self._normalize_alert_severity(fallback.get("severity") or "normal")
                return self._alert_meta_from_counts({severity: raw_total})

        text = str(summary_text or "")
        if not text or not self.alert_parser or not self._contains_alerts_json(text):
            return self._alert_meta_from_counts({})
        try:
            parsed_alerts = self.alert_parser(text, int(channel_id), int(timestamp_ms))
        except TypeError:
            parsed_alerts = cast(Any, self.alert_parser)(text, int(channel_id))
        except Exception:
            parsed_alerts = []

        counts: Dict[str, int] = {}
        if isinstance(parsed_alerts, Sequence) and not isinstance(parsed_alerts, (str, bytes, bytearray)):
            for raw_alert in parsed_alerts:
                if not isinstance(raw_alert, Mapping):
                    continue
                severity = self._normalize_alert_severity(raw_alert.get("severity"))
                counts[severity] = counts.get(severity, 0) + 1
        return self._alert_meta_from_counts(counts)

    @staticmethod
    def _normalize_observed_state_key(label: str) -> str:
        text = str(label or "").strip().lower()
        text = re.sub(r"[*_`#]+", " ", text)
        text = re.sub(r"\([^)]*\)", " ", text)
        text = text.replace("/", " ")
        text = re.sub(r"[^a-zа-яё0-9]+", " ", text, flags=re.IGNORECASE).strip()
        if not text:
            return ""
        return text[:80]

    @staticmethod
    def _normalize_observed_state_value(text: str) -> Optional[str]:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return None
        if re.search(r"\b(?:uncertain|unknown|unclear|ambiguous|not sure|cannot determine|can't determine)\b", lowered):
            return "unknown"
        if re.search(
            r"\b(?:absent|not\s+visible|not\s+present|no\s+visible|none\s+visible|missing|gone|out\s+of\s+frame|not\s+detected)\b",
            lowered,
        ):
            return "absent"
        if re.search(r"\b(?:present|visible|detected|observed|seen|yes|active|in\s+frame)\b", lowered):
            return "present"
        return None

    @classmethod
    def _extract_current_observed_states(cls, summary_text: str) -> List[Dict[str, Any]]:
        text = str(summary_text or "")
        if not text.strip():
            return []
        header = re.search(
            r"^\s*(?:#{1,6}\s*)?current observed state\b.*$",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if not header:
            return []
        tail = text[header.end() :]
        stop = re.search(
            r"^\s*(?:#{1,6}\s+\S|ALERTS_JSON:|MEMORY_UPDATE_JSON:)",
            tail,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        section = tail[: stop.start()] if stop else tail
        observations: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for raw_line in section.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line)
            line = re.sub(r"^\s*\|", "", line).strip()
            if not line or line.lower().startswith(("alert", "summary", "scene", "activity")):
                continue
            match = re.match(
                r"^\s*\**(?P<label>[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9 _/().,'-]{1,90}?)\**\s*(?:[:|–-])\s*(?P<state>.+?)\s*$",
                line,
            )
            if not match:
                continue
            label = " ".join(str(match.group("label") or "").split()).strip(" -*_`|")
            state_text = str(match.group("state") or "").strip()
            state = cls._normalize_observed_state_value(state_text)
            key = cls._normalize_observed_state_key(label)
            if not key or state is None or key in seen:
                continue
            seen.add(key)
            observations.append(
                {
                    "key": key,
                    "label": label[:120],
                    "state": state,
                    "evidence": state_text[:300],
                }
            )
        return observations

    def _update_observed_state_tracker(
        self,
        channel_id: int,
        observations: Sequence[Mapping[str, Any]],
        timestamp_ms: int,
    ) -> List[Dict[str, Any]]:
        if not self.state_transitions_enabled:
            return []
        transitions: List[Dict[str, Any]] = []
        if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes, bytearray)):
            return transitions
        confirm_required = max(1, int(self.state_transition_confirm_batches or 1))
        with self.cache_lock:
            channel_states = self.channel_observed_state_tracker.setdefault(int(channel_id), {})
            for raw_obs in observations[:64]:
                if not isinstance(raw_obs, Mapping):
                    continue
                key = str(raw_obs.get("key") or "").strip()
                label = str(raw_obs.get("label") or key).strip() or key
                state = str(raw_obs.get("state") or "").strip().lower()
                if not key or state not in {"present", "absent", "unknown"}:
                    continue
                tracker = channel_states.get(key)
                if tracker is None:
                    channel_states[key] = {
                        "key": key,
                        "label": label,
                        "stable_state": state,
                        "last_observed_state": state,
                        "last_observed_ms": int(timestamp_ms),
                        "last_changed_ms": int(timestamp_ms),
                        "candidate_state": None,
                        "candidate_count": 0,
                        "updated_at": time.time(),
                    }
                    continue
                tracker["label"] = label or tracker.get("label") or key
                tracker["last_observed_state"] = state
                tracker["last_observed_ms"] = int(timestamp_ms)
                tracker["updated_at"] = time.time()
                if state == "unknown":
                    tracker["candidate_state"] = None
                    tracker["candidate_count"] = 0
                    continue
                stable_state = str(tracker.get("stable_state") or "unknown")
                if stable_state == "unknown":
                    tracker["stable_state"] = state
                    tracker["last_changed_ms"] = int(timestamp_ms)
                    tracker["candidate_state"] = None
                    tracker["candidate_count"] = 0
                    continue
                if state == stable_state:
                    tracker["candidate_state"] = None
                    tracker["candidate_count"] = 0
                    continue
                candidate_state = str(tracker.get("candidate_state") or "")
                if candidate_state == state:
                    candidate_count = int(_parse_optional_int(tracker.get("candidate_count")) or 0) + 1
                else:
                    candidate_count = 1
                tracker["candidate_state"] = state
                tracker["candidate_count"] = candidate_count
                if candidate_count < confirm_required:
                    continue
                event_type = "state_change"
                if stable_state != "present" and state == "present":
                    event_type = "appearance"
                elif stable_state == "present" and state == "absent":
                    event_type = "disappearance"
                transition = {
                    "key": key,
                    "label": label,
                    "event_type": event_type,
                    "from_state": stable_state,
                    "to_state": state,
                    "timestamp_ms": int(timestamp_ms),
                    "confirmations": candidate_count,
                    "required_confirmations": confirm_required,
                    "evidence": str(raw_obs.get("evidence") or "").strip()[:300],
                    "source": "vlm_current_observed_state",
                }
                transitions.append(transition)
                tracker["stable_state"] = state
                tracker["last_changed_ms"] = int(timestamp_ms)
                tracker["candidate_state"] = None
                tracker["candidate_count"] = 0
        return transitions

    @staticmethod
    def _transition_alert_events(transitions: Sequence[Mapping[str, Any]], channel_id: int) -> List[Dict[str, Any]]:
        if not isinstance(transitions, Sequence) or isinstance(transitions, (str, bytes, bytearray)):
            return []
        events: List[Dict[str, Any]] = []
        for transition in transitions[:32]:
            if not isinstance(transition, Mapping):
                continue
            event_type = str(transition.get("event_type") or "state_change").strip().lower()
            label = str(transition.get("label") or transition.get("key") or "Observed state").strip() or "Observed state"
            title = f"{label} {event_type.replace('_', ' ')}".strip()
            evidence = str(transition.get("evidence") or "").strip()
            from_state = str(transition.get("from_state") or "unknown")
            to_state = str(transition.get("to_state") or "unknown")
            description = (
                f"Backend state tracker confirmed {label}: {from_state} -> {to_state} "
                f"from current observed state."
            )
            if evidence:
                description = f"{description} Evidence: {evidence}"
            timestamp_ms = _parse_optional_int(transition.get("timestamp_ms")) or int(time.time() * 1000)
            events.append(
                {
                    "title": title[:120],
                    "description": description[:300],
                    "severity": "info",
                    "state": "new",
                    "channel_id": int(channel_id),
                    "timestamp_ms": int(timestamp_ms),
                    "delivery_status": "state_tracker",
                }
            )
        return events

    def _summary_alert_signal_items(
        self,
        summary_text: object,
        *,
        channel_id: int,
        timestamp_ms: int,
        max_items: int = 5,
    ) -> List[str]:
        text = str(summary_text or "")
        if not text or not self.alert_parser or not self._contains_alerts_json(text):
            return []
        try:
            parsed_alerts = self.alert_parser(text, int(channel_id), int(timestamp_ms))
        except TypeError:
            parsed_alerts = cast(Any, self.alert_parser)(text, int(channel_id))
        except Exception:
            parsed_alerts = []
        if not isinstance(parsed_alerts, Sequence) or isinstance(parsed_alerts, (str, bytes, bytearray)):
            return []
        out: List[str] = []
        seen: Set[str] = set()
        for raw_alert in parsed_alerts:
            if not isinstance(raw_alert, Mapping):
                continue
            severity = self._normalize_alert_severity(raw_alert.get("severity"))
            title = self._truncate_text(raw_alert.get("title") or raw_alert.get("description"), 120)
            description = self._truncate_text(raw_alert.get("description"), 160)
            if not title and not description:
                continue
            text_item = f"{severity}: {title or description}"
            if description and description.lower() != title.lower():
                text_item = f"{text_item} | {description}"
            text_item = self._truncate_text(text_item, 220)
            key = text_item.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(text_item)
            if len(out) >= max(1, int(max_items)):
                break
        return out

    @classmethod
    def _extract_signal_sentences(
        cls,
        text: object,
        patterns: Sequence[str],
        *,
        max_items: int = 3,
        max_len: int = 180,
    ) -> List[str]:
        raw = str(text or "")
        if not raw:
            return []
        raw = re.sub(r"```.*?```", " ", raw, flags=re.DOTALL)
        raw = re.sub(r"ALERTS_JSON:\s*\{.*", " ", raw, flags=re.IGNORECASE | re.DOTALL)
        raw = re.sub(r"MEMORY_UPDATE_JSON:\s*\{.*", " ", raw, flags=re.IGNORECASE | re.DOTALL)
        candidates = re.split(r"(?<=[.!?])\s+|\n+|;\s+", raw)
        regexes = [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]
        out: List[str] = []
        seen: Set[str] = set()
        for candidate in candidates:
            text_item = cls._truncate_text(candidate, max_len)
            if not text_item:
                continue
            if not any(regex.search(text_item) for regex in regexes):
                continue
            key = text_item.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(text_item)
            if len(out) >= max(1, int(max_items)):
                break
        return out

    def _summary_signal_digest(
        self,
        summary_text: object,
        *,
        channel_id: int,
        timestamp_ms: int,
        alert_counts: Optional[Mapping[str, Any]] = None,
        alert_total: Optional[int] = None,
    ) -> Dict[str, Any]:
        digest: Dict[str, Any] = {}
        counts = self._normalize_alert_counts(alert_counts)
        if not counts and int(alert_total or 0) > 0:
            counts = self._normalize_alert_counts({"normal": int(alert_total or 0)})
        if counts:
            digest["alerts"] = counts
            alert_events = self._summary_alert_signal_items(
                summary_text,
                channel_id=channel_id,
                timestamp_ms=timestamp_ms,
                max_items=5,
            )
            if alert_events:
                digest["alert_events"] = alert_events

        text_upper = str(summary_text or "").upper()
        has_structured_memory = "MEMORY_UPDATE_JSON:" in text_upper or bool(
            re.search(
                r"^###\s*(Routine Baseline|Preserved Deviations|Active Watchlist|Alert Tuning Notes|Alerts/Signals|Operator Notes)\s*$",
                str(summary_text or ""),
                flags=re.IGNORECASE | re.MULTILINE,
            )
        )
        if has_structured_memory:
            memory = self._extract_memory_update(summary_text)
            routine = self._truncate_text(memory.get("routine_baseline"), 260)
            if routine:
                digest["routine"] = routine
            for source_key, target_key, max_items, max_len in (
                ("active_watchlist", "watchlist", 4, 180),
                ("preserved_deviations", "deviations", 5, 220),
                ("alert_tuning_notes", "tuning", 3, 180),
                ("ignore_as_routine", "routine_noise", 3, 160),
            ):
                items = self._coerce_memory_items(memory.get(source_key), max_items=max_items, max_len=max_len)
                if items:
                    digest[target_key] = items

        missing = self._extract_signal_sentences(
            summary_text,
            (
                r"\b(no source|no frames|frame gap|dropped frames?|preview failed|load failed|unavailable|not loaded|not ready)\b",
                r"\b(camera|stream|feed|snapshot|frame).{0,40}\b(failed|missing|unavailable|stale|frozen)\b",
                r"\b(occluded|obstructed|blackout)\b",
            ),
            max_items=3,
            max_len=180,
        )
        if missing:
            digest["missing_data"] = missing

        uncertainty = self._extract_signal_sentences(
            summary_text,
            (
                r"\b(uncertain|unknown|ambiguous|unclear|possibly|probably|appears to|seems to|may be|might be)\b",
                r"\b(cannot determine|not enough|insufficient evidence)\b",
            ),
            max_items=3,
            max_len=180,
        )
        if uncertainty:
            digest["uncertainty"] = uncertainty
        return {key: value for key, value in digest.items() if value}

    @classmethod
    def _merge_signal_digest_items(
        cls,
        children: Sequence[Mapping[str, Any]],
        field: str,
        *,
        max_items: int,
        max_len: int = 220,
    ) -> List[str]:
        grouped: List[Sequence[str]] = []
        for child in children:
            if not isinstance(child, Mapping):
                continue
            digest = child.get("signal_digest")
            if not isinstance(digest, Mapping):
                continue
            grouped.append(cls._coerce_memory_items(digest.get(field), max_items=max_items, max_len=max_len))
        return cls._dedupe_memory_items(*grouped, max_items=max_items)

    @classmethod
    def _aggregate_signal_digest(
        cls,
        children: Sequence[Mapping[str, Any]],
        alert_counts: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        digest: Dict[str, Any] = {}
        counts = cls._normalize_alert_counts(alert_counts)
        if counts:
            digest["alerts"] = counts
        for field, limit, max_len in (
            ("alert_events", 6, 220),
            ("deviations", 6, 220),
            ("watchlist", 5, 180),
            ("missing_data", 4, 180),
            ("uncertainty", 4, 180),
            ("tuning", 4, 180),
            ("routine_noise", 4, 160),
            ("routine", 2, 220),
        ):
            items = cls._merge_signal_digest_items(children, field, max_items=limit, max_len=max_len)
            if items:
                digest[field] = items if field != "routine" else items[0]
        return {key: value for key, value in digest.items() if value}

    @classmethod
    def _render_signal_digest(cls, digest_raw: object, *, max_len: int = 1000) -> str:
        if not isinstance(digest_raw, Mapping):
            return ""
        digest = cast(Mapping[str, Any], digest_raw)
        lines: List[str] = []
        alerts = cls._format_alert_counts(digest.get("alerts"))
        if alerts:
            lines.append(f"Alerts: {alerts}")
        for key, label, limit in (
            ("alert_events", "Alert events", 5),
            ("deviations", "Preserved deviations", 5),
            ("watchlist", "Watchlist", 4),
            ("missing_data", "Missing/data-quality", 3),
            ("uncertainty", "Uncertainty", 3),
            ("tuning", "Alert tuning", 3),
            ("routine_noise", "Routine/noise", 3),
        ):
            items = cls._coerce_memory_items(digest.get(key), max_items=limit, max_len=180)
            if items:
                lines.append(f"{label}: " + "; ".join(items))
        routine = cls._truncate_text(digest.get("routine"), 220)
        if routine:
            lines.append(f"Routine cue: {routine}")
        rendered = "\n".join(lines).strip()
        return cls._truncate_text(rendered, max_len)

    @classmethod
    def _merge_alert_metadata(cls, nodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            node_counts = cls._normalize_alert_counts(node.get("alert_counts"))
            if node_counts:
                for severity, count in node_counts.items():
                    counts[severity] = counts.get(severity, 0) + int(count)
                continue
            total = _parse_optional_int(node.get("alert_total")) or 0
            if total > 0:
                severity = cls._normalize_alert_severity(node.get("severity") or "normal")
                counts[severity] = counts.get(severity, 0) + total
        return cls._alert_meta_from_counts(counts)

    @staticmethod
    def _compact_llm_input_stats(value: object) -> Dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        out: Dict[str, Any] = {}
        for key in (
            "phase",
            "level",
            "source_level",
            "frame_count",
            "batch_size",
            "message_count",
            "text_chars",
            "image_parts",
            "high_detail_images",
            "image_url_chars",
            "total_payload_chars",
            "system_prompt_chars",
            "task_prompt_chars",
            "total_image_base64_chars",
            "largest_frame_base64_chars",
            "source_lines_selected",
            "source_lines_available",
            "source_chars_selected",
            "source_char_budget",
            "backend_instruction_chars",
            "routine_context_chars",
            "signal_digest_chars",
            "vector_signal_chars",
            "warning_text_chars",
            "warning_image_payload_chars",
        ):
            if key not in value:
                continue
            item = value.get(key)
            if isinstance(item, str):
                out[key] = item[:120]
            elif isinstance(item, (int, float, bool)) or item is None:
                out[key] = item
            else:
                parsed = _parse_optional_int(item)
                if parsed is not None:
                    out[key] = parsed
        warnings = value.get("warnings")
        if isinstance(warnings, Sequence) and not isinstance(warnings, (str, bytes, bytearray)):
            out["warnings"] = [str(item)[:180] for item in warnings[:6] if str(item or "").strip()]
        return out

    @staticmethod
    def _compact_alert_events(value: object) -> List[Dict[str, Any]]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            return []
        events: List[Dict[str, Any]] = []
        for raw_event in value[:32]:
            if not isinstance(raw_event, Mapping):
                continue
            title = str(raw_event.get("title") or "Event").strip()[:120] or "Event"
            description = str(raw_event.get("description") or "").strip()[:300]
            severity = str(raw_event.get("severity") or "normal").strip().lower()[:20] or "normal"
            state = str(raw_event.get("state") or "new").strip().lower()[:20] or "new"
            event: Dict[str, Any] = {
                "title": title,
                "description": description,
                "severity": severity,
                "state": state,
            }
            channel_id = _parse_optional_int(raw_event.get("channel_id"))
            if channel_id is not None:
                event["channel_id"] = int(channel_id)
            timestamp_ms = _parse_optional_int(raw_event.get("timestamp_ms"))
            if timestamp_ms is not None:
                event["timestamp_ms"] = int(timestamp_ms)
            status = str(raw_event.get("delivery_status") or "").strip().lower()
            if status:
                event["delivery_status"] = status[:40]
            error = str(raw_event.get("error") or "").strip()
            if error:
                event["error"] = error[:240]
            events.append(event)
        return events

    @staticmethod
    def _compact_state_observations(value: object) -> List[Dict[str, Any]]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            return []
        out: List[Dict[str, Any]] = []
        for raw_obs in value[:64]:
            if not isinstance(raw_obs, Mapping):
                continue
            key = str(raw_obs.get("key") or "").strip()[:80]
            state = str(raw_obs.get("state") or "").strip().lower()
            if not key or state not in {"present", "absent", "unknown"}:
                continue
            out.append(
                {
                    "key": key,
                    "label": str(raw_obs.get("label") or key).strip()[:120],
                    "state": state,
                    "evidence": str(raw_obs.get("evidence") or "").strip()[:300],
                }
            )
        return out

    @staticmethod
    def _compact_state_transition_events(value: object) -> List[Dict[str, Any]]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            return []
        out: List[Dict[str, Any]] = []
        for raw_event in value[:32]:
            if not isinstance(raw_event, Mapping):
                continue
            key = str(raw_event.get("key") or "").strip()[:80]
            event_type = str(raw_event.get("event_type") or "").strip().lower()
            if not key or event_type not in {"appearance", "disappearance", "state_change"}:
                continue
            event: Dict[str, Any] = {
                "key": key,
                "label": str(raw_event.get("label") or key).strip()[:120],
                "event_type": event_type,
                "from_state": str(raw_event.get("from_state") or "unknown").strip().lower()[:20],
                "to_state": str(raw_event.get("to_state") or "unknown").strip().lower()[:20],
                "evidence": str(raw_event.get("evidence") or "").strip()[:300],
                "source": str(raw_event.get("source") or "vlm_current_observed_state").strip()[:80],
            }
            timestamp_ms = _parse_optional_int(raw_event.get("timestamp_ms"))
            if timestamp_ms is not None:
                event["timestamp_ms"] = int(timestamp_ms)
            for field in ("confirmations", "required_confirmations"):
                parsed = _parse_optional_int(raw_event.get(field))
                if parsed is not None:
                    event[field] = int(parsed)
            out.append(event)
        return out

    @staticmethod
    def _finite_float(value: object) -> Optional[float]:
        try:
            number = float(cast(Any, value))
        except Exception:
            return None
        if not math.isfinite(number):
            return None
        return number

    @classmethod
    def _compact_vector_signal(cls, value: object) -> Dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        out: Dict[str, Any] = {
            "version": int(_parse_optional_int(value.get("version")) or 1),
            "semantics": "vector_homeostasis_attention_signal_not_visual_proof",
        }
        channel_id = _parse_optional_int(value.get("channel_id"))
        if channel_id is not None:
            out["channel_id"] = int(channel_id)
        for key in ("batch_start_ms", "batch_end_ms"):
            parsed = _parse_optional_int(value.get(key))
            if parsed is not None:
                out[key] = int(parsed)

        clip_items: List[Dict[str, Any]] = []
        raw_clip = value.get("clip_probe_signals")
        if isinstance(raw_clip, Sequence) and not isinstance(raw_clip, (str, bytes, bytearray)):
            for raw in raw_clip[:12]:
                if not isinstance(raw, Mapping):
                    continue
                name = str(raw.get("name") or raw.get("probe_name") or "").strip()[:120]
                if not name:
                    continue
                item: Dict[str, Any] = {
                    "name": name,
                    "state": str(raw.get("state") or "positive_candidate").strip().lower()[:40],
                    "score_semantics": "clip_pnm_attention_signal_not_visual_proof",
                }
                for text_key in ("probe_id", "severity"):
                    text = str(raw.get(text_key) or "").strip()
                    if text:
                        item[text_key] = text[:80]
                for score_key, out_key in (
                    ("p", "p"),
                    ("n", "n"),
                    ("m", "m"),
                    ("pos_score", "p"),
                    ("negative_score", "n"),
                    ("neg_score", "n"),
                    ("margin", "m"),
                ):
                    number = cls._finite_float(raw.get(score_key))
                    if number is not None:
                        item[out_key] = round(float(number), 4)
                for int_key in ("timestamp_ms", "apex_frame", "hit_count"):
                    parsed = _parse_optional_int(raw.get(int_key))
                    if parsed is not None:
                        item[int_key] = int(parsed)
                clip_items.append(item)
        if clip_items:
            out["clip_probe_signals"] = clip_items[:8]

        road_items: List[Dict[str, Any]] = []
        raw_road = value.get("road_cv_cues")
        if isinstance(raw_road, Sequence) and not isinstance(raw_road, (str, bytes, bytearray)):
            for raw in raw_road[:12]:
                if not isinstance(raw, Mapping):
                    continue
                cue_type = str(raw.get("cue_type") or raw.get("type") or "").strip()[:80]
                if not cue_type:
                    continue
                item = {
                    "cue_type": cue_type,
                    "score_semantics": "road_cv_motion_attention_signal_not_visual_proof",
                }
                for text_key in ("zone_name", "evidence"):
                    text = str(raw.get(text_key) or "").strip()
                    if text:
                        item[text_key] = text[:180 if text_key == "evidence" else 80]
                score = cls._finite_float(raw.get("score"))
                if score is not None:
                    item["score"] = round(float(score), 4)
                for int_key in ("timestamp_ms", "frame_index", "apex_frame"):
                    parsed = _parse_optional_int(raw.get(int_key))
                    if parsed is not None:
                        item[int_key] = int(parsed)
                road_items.append(item)
        if road_items:
            out["road_cv_cues"] = road_items[:8]

        scene = value.get("road_cv_scene")
        if isinstance(scene, Mapping):
            scene_out: Dict[str, Any] = {}
            for text_key in ("confidence", "reason"):
                text = str(scene.get(text_key) or "").strip()
                if text:
                    scene_out[text_key] = text[:180]
            for int_key in ("frame_count", "motion_pair_count", "scene_cut_count"):
                parsed = _parse_optional_int(scene.get(int_key))
                if parsed is not None:
                    scene_out[int_key] = int(parsed)
            for score_key in ("zone_area_ratio", "flow_dominance"):
                number = cls._finite_float(scene.get(score_key))
                if number is not None:
                    scene_out[score_key] = round(float(number), 4)
            if scene_out:
                out["road_cv_scene"] = scene_out

        health = value.get("health")
        if isinstance(health, Mapping):
            health_out: Dict[str, Any] = {}
            for raw_key, raw_value in health.items():
                key = str(raw_key or "").strip().lower()[:80]
                if not key:
                    continue
                if isinstance(raw_value, bool):
                    health_out[key] = raw_value
                    continue
                parsed_int = _parse_optional_int(raw_value)
                if parsed_int is not None:
                    health_out[key] = int(parsed_int)
                    continue
                parsed_float = cls._finite_float(raw_value)
                if parsed_float is not None:
                    health_out[key] = round(float(parsed_float), 4)
                    continue
                text = str(raw_value or "").strip()
                if text:
                    health_out[key] = text[:160]
            if health_out:
                out["health"] = health_out

        has_signal_payload = any(key in out for key in ("clip_probe_signals", "road_cv_cues"))
        if not has_signal_payload:
            return {}
        return out

    @staticmethod
    def _batch_frame_timestamp_ms(frame: Mapping[str, Any]) -> Optional[int]:
        for key in ("timestamp_ms", "captured_at_ms"):
            parsed = _parse_optional_int(frame.get(key))
            if parsed is not None:
                return int(parsed)
        for key in ("captured_at", "time_sec"):
            raw = frame.get(key)
            if isinstance(raw, (int, float)):
                return int(float(raw) * 1000.0)
        return None

    @classmethod
    def _nearest_batch_frame_index(cls, frames: Sequence[Mapping[str, Any]], timestamp_ms: Optional[int]) -> Optional[int]:
        if timestamp_ms is None or not frames:
            return None
        best_idx: Optional[int] = None
        best_delta: Optional[int] = None
        for idx, frame in enumerate(frames, start=1):
            parsed = cls._batch_frame_timestamp_ms(frame)
            if parsed is None:
                continue
            delta = abs(int(parsed) - int(timestamp_ms))
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_idx = idx
        return best_idx

    @staticmethod
    def _decode_frame_thumbnail_to_rgb_array(frame: Mapping[str, Any]) -> Optional[Any]:
        raw = str(frame.get("thumbnail") or frame.get("thumbnail_b64") or "").strip()
        if not raw:
            return None
        if "," in raw and raw.lower().startswith("data:"):
            raw = raw.split(",", 1)[1]
        try:
            data = base64.b64decode(raw, validate=False)
            with Image.open(BytesIO(data)) as image:
                rgb = image.convert("RGB")
                import numpy as np  # Local import keeps minimal installs lighter.

                return np.asarray(rgb)
        except Exception:
            return None

    def _clip_probe_vector_signals(
        self,
        channel_id: int,
        frames: Sequence[Mapping[str, Any]],
        *,
        batch_start_ms: Optional[int],
        batch_end_ms: Optional[int],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        health: Dict[str, Any] = {}
        if self.probe_manager is None:
            health["clip_probe_status"] = "probe_manager_unavailable"
            return [], health
        store = self.probes_store
        if store is None:
            health["clip_probe_status"] = "probe_store_unavailable"
            return [], health
        try:
            probes = store.list_probes()
        except Exception as exc:
            health["clip_probe_error"] = str(exc)[:160] or exc.__class__.__name__
            return [], health
        active: List[Mapping[str, Any]] = []
        for raw_probe in probes if isinstance(probes, list) else []:
            if not isinstance(raw_probe, Mapping):
                continue
            if raw_probe.get("enabled") is False:
                continue
            probe_channel = _parse_optional_int(raw_probe.get("channel_id"))
            if probe_channel != int(channel_id):
                continue
            positives = [str(item).strip() for item in (raw_probe.get("positives") or []) if str(item).strip()]
            image_probe = raw_probe.get("image_probe") if isinstance(raw_probe.get("image_probe"), Mapping) else None
            if not positives and not (image_probe and image_probe.get("data")):
                continue
            active.append(raw_probe)
        health["clip_probe_configured"] = len(active)
        if not active or self.vector_signal_probe_limit <= 0:
            return [], health
        active = active[: self.vector_signal_probe_limit]
        health["clip_probe_scanned"] = len(active)

        duration_sec = 30.0
        if batch_start_ms is not None and batch_end_ms is not None and batch_end_ms >= batch_start_ms:
            duration_sec = max(10.0, min(180.0, (batch_end_ms - batch_start_ms) / 1000.0 + 15.0))
        signals: List[Dict[str, Any]] = []
        for probe in active:
            try:
                positives = [str(item).strip() for item in (probe.get("positives") or []) if str(item).strip()]
                negatives = [str(item).strip() for item in (probe.get("negatives") or []) if str(item).strip()]
                pos_floor = float(probe.get("pos_floor", 0.2))
                margin_thr = float(probe.get("margin", 0.05))
                result = self.probe_manager.query(
                    int(channel_id),
                    positives,
                    negatives,
                    pos_floor,
                    margin_thr,
                    max(1, self.vector_signal_top_hits),
                    window_sec=duration_sec,
                    image_probe=cast(Optional[Dict[str, Any]], probe.get("image_probe") if isinstance(probe.get("image_probe"), Mapping) else None),
                )
            except Exception as exc:
                health["clip_probe_query_error"] = str(exc)[:160] or exc.__class__.__name__
                continue
            if not isinstance(result, Mapping):
                continue
            frames_indexed = _parse_optional_int(result.get("frames_indexed"))
            if frames_indexed is not None:
                health["clip_frames_indexed"] = max(int(health.get("clip_frames_indexed") or 0), int(frames_indexed))
            hits = result.get("results")
            if not isinstance(hits, Sequence) or isinstance(hits, (str, bytes, bytearray)) or not hits:
                continue
            best = next((item for item in hits if isinstance(item, Mapping)), None)
            if best is None:
                continue
            timestamp_ms = _parse_optional_int(best.get("timestamp_ms"))
            apex_frame = self._nearest_batch_frame_index(frames, timestamp_ms)
            signal: Dict[str, Any] = {
                "name": str(probe.get("name") or probe.get("id") or "CLIP probe").strip()[:120],
                "probe_id": str(probe.get("id") or "").strip()[:80],
                "severity": str(probe.get("severity") or "normal").strip().lower()[:20] or "normal",
                "state": "positive_candidate",
                "hit_count": len([item for item in hits if isinstance(item, Mapping)]),
                "score_semantics": "clip_pnm_attention_signal_not_visual_proof",
            }
            for src_key, out_key in (("pos_score", "pos_score"), ("neg_score", "negative_score"), ("margin", "margin")):
                number = self._finite_float(best.get(src_key))
                if number is not None:
                    signal[out_key] = round(float(number), 4)
            if timestamp_ms is not None:
                signal["timestamp_ms"] = int(timestamp_ms)
            if apex_frame is not None:
                signal["apex_frame"] = int(apex_frame)
            signals.append(signal)
        signals.sort(
            key=lambda item: (
                ALERT_SEVERITY_ORDER.index(str(item.get("severity") or "info")) if str(item.get("severity") or "info") in ALERT_SEVERITY_ORDER else 99,
                -float(item.get("margin") or 0.0),
            )
        )
        return signals[:8], health

    def _road_cv_vector_signals(
        self,
        channel_id: int,
        frames: Sequence[Mapping[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        health: Dict[str, Any] = {}
        if not self.road_cv_batch_signals_enabled:
            health["road_cv_status"] = "disabled"
            return [], {}, health
        if (
            DecodedVideoFrame is None
            or AutoSceneCardConfig is None
            or infer_scene_card_from_frames is None
            or RoadMotionAnalyzer is None
        ):
            health["road_cv_status"] = "unavailable"
            return [], {}, health
        decoded: List[Any] = []
        sampled = list(frames)[-self.road_cv_batch_max_frames :]
        for idx, frame in enumerate(sampled, start=1):
            if not isinstance(frame, Mapping):
                continue
            image = self._decode_frame_thumbnail_to_rgb_array(frame)
            if image is None:
                continue
            timestamp_ms = self._batch_frame_timestamp_ms(frame)
            if timestamp_ms is None:
                timestamp_ms = int(time.time() * 1000.0)
            decoded.append(DecodedVideoFrame(frame_index=idx, timestamp_ms=int(timestamp_ms), image=image))
        health["road_cv_decoded_frames"] = len(decoded)
        if len(decoded) < 3:
            return [], {}, health
        try:
            scene_result = infer_scene_card_from_frames(
                int(channel_id),
                f"Channel {channel_id}",
                decoded,
                config=AutoSceneCardConfig(max_edge=int(self.road_cv_batch_max_edge), min_frames=min(12, max(3, len(decoded)))),
            )
            analyzer = RoadMotionAnalyzer(scene_result.scene_card)
            cues: List[Dict[str, Any]] = []
            for decoded_frame in decoded:
                sample = analyzer.analyze_frame(
                    decoded_frame.image,
                    timestamp_ms=int(decoded_frame.timestamp_ms),
                    frame_index=int(decoded_frame.frame_index),
                )
                for cue in sample.cues:
                    cues.append(
                        {
                            "cue_type": cue.cue_type,
                            "zone_name": cue.zone_name,
                            "score": round(float(cue.score), 4),
                            "evidence": cue.evidence,
                            "timestamp_ms": int(cue.timestamp_ms),
                            "frame_index": int(cue.frame_index or decoded_frame.frame_index),
                            "apex_frame": int(cue.frame_index or decoded_frame.frame_index),
                        }
                    )
            cues.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
            scene = scene_result.as_dict()
            scene_compact = {
                "confidence": scene.get("confidence"),
                "reason": scene.get("reason"),
                "frame_count": scene.get("frame_count"),
                "motion_pair_count": scene.get("motion_pair_count"),
                "scene_cut_count": scene.get("scene_cut_count"),
                "zone_area_ratio": scene.get("zone_area_ratio"),
                "flow_dominance": scene.get("flow_dominance"),
            }
            return cues[:8], scene_compact, health
        except Exception as exc:
            health["road_cv_error"] = str(exc)[:160] or exc.__class__.__name__
            return [], {}, health

    @staticmethod
    def _compact_count_breakdown(value: object) -> Dict[str, int]:
        if not isinstance(value, Mapping):
            return {}
        out: Dict[str, int] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key or "").strip().lower()
            if not key:
                continue
            parsed = _parse_optional_int(raw_value)
            if parsed is None or parsed <= 0:
                continue
            out[key[:80]] = out.get(key[:80], 0) + int(parsed)
        return out

    @classmethod
    def _alert_parser_breakdown_from_entry(cls, entry: Mapping[str, Any]) -> Dict[str, int]:
        parser_count = _parse_optional_int(entry.get("parser_alert_count"))
        if parser_count is None:
            parser_count = _parse_optional_int(entry.get("alerts_parsed")) or 0
        json_count = _parse_optional_int(entry.get("json_alert_count")) or 0
        prose_count = _parse_optional_int(entry.get("prose_alert_count")) or 0
        breakdown = {
            "parser_alert_count": int(max(0, parser_count or 0)),
            "json_alert_count": int(max(0, json_count)),
            "prose_alert_count": int(max(0, prose_count)),
            "prose_only_signal_count": int(max(0, prose_count - json_count)),
        }
        return {key: value for key, value in breakdown.items() if value > 0}

    @classmethod
    def _alert_delivery_breakdown_from_entry(cls, entry: Mapping[str, Any]) -> Dict[str, int]:
        breakdown: Dict[str, int] = {}
        for event in cls._compact_alert_events(entry.get("alert_events")):
            status = str(event.get("delivery_status") or "unknown").strip().lower() or "unknown"
            breakdown[status] = breakdown.get(status, 0) + 1
        if not breakdown:
            for source_key, status in (
                ("bookmarks_sent", "sent"),
                ("bookmark_failed_count", "failed"),
                ("bookmark_cooldown_skipped_count", "cooldown_skipped"),
                ("bookmark_skipped_duplicate_count", "cooldown_skipped"),
            ):
                parsed = _parse_optional_int(entry.get(source_key)) or 0
                if parsed > 0:
                    breakdown[status] = breakdown.get(status, 0) + int(parsed)
        total = sum(value for value in breakdown.values() if isinstance(value, int) and value > 0)
        if total > 0:
            breakdown["total"] = total
        return breakdown

    @classmethod
    def _merge_count_breakdowns(
        cls,
        children: Sequence[Mapping[str, Any]],
        field: str,
    ) -> Dict[str, int]:
        merged: Dict[str, int] = {}
        for child in children:
            if not isinstance(child, Mapping):
                continue
            breakdown = cls._compact_count_breakdown(child.get(field))
            for key, value in breakdown.items():
                merged[key] = merged.get(key, 0) + int(value)
        return merged

    @classmethod
    def _aggregate_provenance_metadata(cls, children: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        delivery = cls._merge_count_breakdowns(children, "alert_delivery_breakdown")
        parser = cls._merge_count_breakdowns(children, "alert_parser_breakdown")
        state_transition_total = 0
        vector_signal_total = 0
        for child in children:
            if not isinstance(child, Mapping):
                continue
            state_transition_total += int(_parse_optional_int(child.get("state_transition_total")) or 0)
            direct_vector_total = _parse_optional_int(child.get("vector_signal_total"))
            if direct_vector_total is not None:
                vector_signal_total += max(0, int(direct_vector_total))
                continue
            vector_signal = cls._compact_vector_signal(child.get("vector_signal"))
            if vector_signal:
                clip_count = len(vector_signal.get("clip_probe_signals") or []) if isinstance(vector_signal.get("clip_probe_signals"), list) else 0
                road_count = len(vector_signal.get("road_cv_cues") or []) if isinstance(vector_signal.get("road_cv_cues"), list) else 0
                vector_signal_total += int(clip_count + road_count)
        meta: Dict[str, Any] = {}
        if delivery:
            meta["alert_delivery_breakdown"] = delivery
        if parser:
            meta["alert_parser_breakdown"] = parser
        if state_transition_total > 0:
            meta["state_transition_total"] = int(state_transition_total)
        if vector_signal_total > 0:
            meta["vector_signal_total"] = int(vector_signal_total)
        return meta

    @staticmethod
    def _merge_int_breakdown(target: Dict[str, int], source: Mapping[str, Any]) -> None:
        for raw_key, raw_value in source.items():
            key = str(raw_key or "").strip().lower()
            parsed = _parse_optional_int(raw_value)
            if not key or parsed is None or parsed <= 0:
                continue
            target[key[:80]] = target.get(key[:80], 0) + int(parsed)

    @classmethod
    def _channel_status_digest_from_logs(
        cls,
        channel_id: int,
        logs: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        ordered = sorted(
            [dict(log) for log in logs if isinstance(log, Mapping)],
            key=lambda row: float(cls._coerce_float(row.get("created_at")) or 0.0),
        )
        alert_counts: Dict[str, int] = {}
        parser_breakdown: Dict[str, int] = {}
        delivery_breakdown: Dict[str, int] = {}
        recent_alerts: List[Dict[str, Any]] = []
        recent_transitions: List[Dict[str, Any]] = []
        recent_vector_signals: List[Dict[str, Any]] = []
        current_state: List[Dict[str, Any]] = []
        first_ts: Optional[float] = None
        last_ts: Optional[float] = None
        last_batch_end_ms: Optional[int] = None
        frame_count = 0
        state_transition_total = 0
        vector_signal_total = 0

        for log in ordered:
            start_ts, end_ts = cls._summary_log_bounds_seconds(log)
            if start_ts is not None:
                first_ts = start_ts if first_ts is None else min(first_ts, start_ts)
            created = cls._coerce_float(log.get("created_at"))
            latest_candidate = end_ts if end_ts is not None else created
            if latest_candidate is not None:
                last_ts = latest_candidate if last_ts is None else max(last_ts, latest_candidate)
            parsed_batch_end = _parse_optional_int(log.get("batch_end_ms"))
            if parsed_batch_end is not None:
                last_batch_end_ms = parsed_batch_end if last_batch_end_ms is None else max(last_batch_end_ms, parsed_batch_end)
            frame_count += int(_parse_optional_int(log.get("frame_count")) or 0)

            raw_counts = log.get("alert_counts")
            if isinstance(raw_counts, Mapping):
                cls._merge_int_breakdown(alert_counts, raw_counts)
            else:
                total = _parse_optional_int(log.get("alert_total")) or 0
                if total > 0:
                    severity = str(log.get("severity") or "normal").strip().lower() or "normal"
                    alert_counts[severity] = alert_counts.get(severity, 0) + int(total)

            cls._merge_int_breakdown(parser_breakdown, cls._alert_parser_breakdown_from_entry(log))
            cls._merge_int_breakdown(delivery_breakdown, cls._alert_delivery_breakdown_from_entry(log))
            state_transition_total += int(_parse_optional_int(log.get("state_transition_total")) or 0)

            event_ts_fallback = parsed_batch_end
            for event in cls._compact_alert_events(log.get("alert_events")):
                timestamp_ms = _parse_optional_int(event.get("timestamp_ms"))
                if timestamp_ms is None:
                    timestamp_ms = event_ts_fallback
                item = {
                    "title": str(event.get("title") or "Event").strip()[:120] or "Event",
                    "severity": str(event.get("severity") or "normal").strip().lower()[:20] or "normal",
                    "delivery_status": str(event.get("delivery_status") or "unknown").strip().lower()[:40] or "unknown",
                }
                if timestamp_ms is not None:
                    item["timestamp_ms"] = int(timestamp_ms)
                recent_alerts.append(item)

            for transition in cls._compact_state_transition_events(log.get("state_transition_events")):
                recent_transitions.append(transition)
            state_observations = cls._compact_state_observations(log.get("state_observations"))
            if state_observations:
                current_state = state_observations[:16]
            vector_signal = cls._compact_vector_signal(log.get("vector_signal"))
            if vector_signal:
                clip_count = len(vector_signal.get("clip_probe_signals") or []) if isinstance(vector_signal.get("clip_probe_signals"), list) else 0
                road_count = len(vector_signal.get("road_cv_cues") or []) if isinstance(vector_signal.get("road_cv_cues"), list) else 0
                vector_signal_total += int(clip_count + road_count)
                vector_item = {
                    "timestamp_ms": parsed_batch_end or (int(latest_candidate * 1000.0) if latest_candidate is not None else None),
                    "clip_probe_signal_count": clip_count,
                    "road_cv_cue_count": road_count,
                    "health": vector_signal.get("health") if isinstance(vector_signal.get("health"), Mapping) else {},
                }
                clip_signals = vector_signal.get("clip_probe_signals")
                if isinstance(clip_signals, list) and clip_signals:
                    vector_item["top_clip_probe"] = clip_signals[0]
                road_cues = vector_signal.get("road_cv_cues")
                if isinstance(road_cues, list) and road_cues:
                    vector_item["top_road_cv_cue"] = road_cues[0]
                recent_vector_signals.append(vector_item)

        recent_alerts.sort(
            key=lambda row: int(_parse_optional_int(row.get("timestamp_ms")) or 0),
            reverse=True,
        )
        recent_transitions.sort(
            key=lambda row: int(_parse_optional_int(row.get("timestamp_ms")) or 0),
            reverse=True,
        )
        recent_vector_signals.sort(
            key=lambda row: int(_parse_optional_int(row.get("timestamp_ms")) or 0),
            reverse=True,
        )
        alert_total = int(sum(value for value in alert_counts.values() if isinstance(value, int) and value > 0))
        updated_at = time.time()
        return {
            "channel_id": int(channel_id),
            "summary_count": len(ordered),
            "frame_count": int(frame_count),
            "first_summary_ts": first_ts,
            "last_summary_ts": last_ts,
            "last_summary_batch_end_ms": last_batch_end_ms,
            "alert_total": alert_total,
            "alert_counts_by_severity": dict(alert_counts),
            "recent_alerts": recent_alerts[:10],
            "alert_delivery_breakdown": dict(delivery_breakdown),
            "alert_parser_breakdown": dict(parser_breakdown),
            "state_transition_total": int(state_transition_total),
            "recent_state_transitions": recent_transitions[:10],
            "current_observed_state": current_state[:16],
            "vector_signal_total": int(vector_signal_total),
            "recent_vector_signals": recent_vector_signals[:10],
            "updated_at": updated_at,
            "rebuilt_from_history": True,
            "source": "summary_history",
        }

    def _rebuild_channel_status_digest_locked(self) -> None:
        self.channel_status_digest = {
            int(channel_id): self._channel_status_digest_from_logs(int(channel_id), logs)
            for channel_id, logs in self.summary_history.items()
            if isinstance(logs, Sequence) and not isinstance(logs, (str, bytes, bytearray)) and logs
        }

    def _update_channel_status_digest_locked(
        self,
        channel_id: int,
        logs: Sequence[Mapping[str, Any]],
    ) -> None:
        if not logs:
            self.channel_status_digest.pop(int(channel_id), None)
            return
        self.channel_status_digest[int(channel_id)] = self._channel_status_digest_from_logs(int(channel_id), logs)

    @staticmethod
    def _overlay_stream_runtime_on_digest(
        digest: Dict[str, Any],
        runtime: Mapping[str, Any],
    ) -> None:
        latest_log_ts: Optional[float] = None
        logs = runtime.get("logs")
        if isinstance(logs, Sequence) and not isinstance(logs, (str, bytes, bytearray)) and logs:
            latest_log = logs[-1] if isinstance(logs[-1], Mapping) else None
            if isinstance(latest_log, Mapping):
                latest_log_ts = LuxriotManager._coerce_float(latest_log.get("created_at"))
        if latest_log_ts is None:
            latest_log_ts = LuxriotManager._coerce_float(runtime.get("last_summary_at"))
        digest_ts = LuxriotManager._coerce_float(digest.get("last_summary_ts"))
        if latest_log_ts is not None and digest_ts is not None and latest_log_ts > digest_ts:
            for field in (
                "recent_alerts",
                "alert_counts_by_severity",
                "alert_delivery_breakdown",
                "alert_parser_breakdown",
                "recent_state_transitions",
                "current_observed_state",
            ):
                digest.pop(field, None)
            digest["stale_digest"] = True
        digest["running"] = bool(runtime.get("running"))
        digest["video_lm"] = str(runtime.get("model") or "").strip() or None
        digest["pending_frames"] = _parse_optional_int(runtime.get("pending_frames")) or 0
        digest["dropped_frames"] = _parse_optional_int(runtime.get("dropped_frames")) or 0
        digest["dropped_batches"] = _parse_optional_int(runtime.get("queue_dropped_batches")) or 0
        for field in (
            "snapshot_count",
            "snapshot_failed_count",
            "slow_snapshot_count",
            "snapshot_slow_threshold_sec",
            "last_snapshot_latency_sec",
            "avg_snapshot_latency_sec",
            "max_snapshot_latency_sec",
            "last_snapshot_at",
            "capture_source_mode",
            "active_capture_source",
            "live_segment_count",
            "live_segment_failed_count",
            "live_segment_frame_count",
            "last_live_segment_latency_sec",
            "last_live_segment_frames",
            "last_live_segment_error",
            "frozen_signal",
            "frozen_signal_since",
            "frozen_signal_age_sec",
            "frozen_frame_count",
            "frozen_frame_hash",
            "frozen_frame_dropped_count",
        ):
            if field in runtime:
                digest[field] = runtime.get(field)
        last_error = str(runtime.get("last_error") or runtime.get("last_restore_error") or "").strip()
        digest["last_error"] = last_error[:240] or None
        digest["runtime_updated_at"] = time.time()

    def _normalize_summary_log_entry(self, entry: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        channel_id = _parse_optional_int(entry.get("channel_id"))
        if channel_id is None:
            return None
        summary = str(entry.get("summary") or "").strip()
        if not summary:
            return None
        created_at = self._coerce_float(entry.get("created_at"))
        if created_at is None:
            created_at = time.time()
        frame_count = _parse_optional_int(entry.get("frame_count")) or 0
        batch_size = _parse_optional_int(entry.get("batch_size")) or 0
        duration_sec = self._coerce_float(entry.get("duration_sec")) or 0.0
        created_ms = int(float(created_at) * 1000.0)
        batch_start_ms = _parse_optional_int(entry.get("batch_start_ms"))
        if batch_start_ms is None:
            batch_start_ms = created_ms
        batch_end_ms = _parse_optional_int(entry.get("batch_end_ms"))
        if batch_end_ms is None:
            batch_end_ms = batch_start_ms
        if batch_end_ms < batch_start_ms:
            batch_start_ms, batch_end_ms = batch_end_ms, batch_start_ms
        alert_meta = self._summary_alert_metadata(
            summary,
            channel_id=int(channel_id),
            timestamp_ms=int(batch_end_ms),
            fallback=entry,
        )
        signal_digest = self._summary_signal_digest(
            summary,
            channel_id=int(channel_id),
            timestamp_ms=int(batch_end_ms),
            alert_counts=cast(Mapping[str, Any], alert_meta.get("alert_counts") or {}),
            alert_total=int(alert_meta.get("alert_total") or 0),
        )
        bookmarks_sent = _parse_optional_int(entry.get("bookmarks_sent")) or 0
        alerts_parsed_count = int(max(0, _parse_optional_int(entry.get("alerts_parsed")) or 0))
        parser_alert_count = _parse_optional_int(entry.get("parser_alert_count"))
        if parser_alert_count is None:
            parser_alert_count = alerts_parsed_count
        duplicate_skip_count = int(max(0, _parse_optional_int(entry.get("bookmark_skipped_duplicate_count")) or 0))
        cooldown_skip_count = _parse_optional_int(entry.get("bookmark_cooldown_skipped_count"))
        if cooldown_skip_count is None:
            cooldown_skip_count = duplicate_skip_count
        return {
            "channel_id": int(channel_id),
            "run_id": str(entry.get("run_id") or "").strip(),
            "summary": summary,
            "frame_count": int(max(0, frame_count)),
            "batch_size": int(max(0, batch_size)),
            "created_at": float(created_at),
            "batch_start_ms": int(batch_start_ms),
            "batch_end_ms": int(batch_end_ms),
            "duration_sec": float(max(0.0, duration_sec)),
            "prompt": str(entry.get("prompt") or ""),
            "bookmarks_sent": int(max(0, bookmarks_sent)),
            "alerts_detected": bool(entry.get("alerts_detected")),
            "alerts_parsed": alerts_parsed_count,
            "parser_alert_count": int(max(0, parser_alert_count)),
            "json_alert_count": int(max(0, _parse_optional_int(entry.get("json_alert_count")) or 0)),
            "prose_alert_count": int(max(0, _parse_optional_int(entry.get("prose_alert_count")) or 0)),
            "bookmark_failed_count": int(max(0, _parse_optional_int(entry.get("bookmark_failed_count")) or 0)),
            "bookmark_skipped_duplicate_count": duplicate_skip_count,
            "bookmark_cooldown_skipped_count": int(max(0, cooldown_skip_count)),
            "bookmark_last_error": self._truncate_text(entry.get("bookmark_last_error"), 240),
            "alert_parser_error": self._truncate_text(entry.get("alert_parser_error"), 240),
            "alert_events": self._compact_alert_events(entry.get("alert_events")),
            "state_observations": self._compact_state_observations(entry.get("state_observations")),
            "state_transition_events": self._compact_state_transition_events(entry.get("state_transition_events")),
            "state_transition_total": int(max(0, _parse_optional_int(entry.get("state_transition_total")) or 0)),
            "vector_signal": self._compact_vector_signal(entry.get("vector_signal")),
            "llm_input_stats": self._compact_llm_input_stats(entry.get("llm_input_stats")),
            "signal_digest": signal_digest,
            **alert_meta,
        }

    def _normalize_summary_run_entry(self, entry: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        channel_id = _parse_optional_int(entry.get("channel_id"))
        run_id = str(entry.get("run_id") or "").strip()
        if channel_id is None or not run_id:
            return None
        started_at = self._coerce_float(entry.get("started_at"))
        if started_at is None:
            started_at = time.time()
        ended_at = self._coerce_float(entry.get("ended_at"))
        batch_size = _parse_optional_int(entry.get("batch_size"))
        return {
            "run_id": run_id,
            "channel_id": int(channel_id),
            "started_at": float(started_at),
            "ended_at": ended_at,
            "running": bool(entry.get("running")),
            "batch_size": int(batch_size) if batch_size is not None else 0,
            "interval_sec": self._normalize_capture_interval_sec(entry.get("interval_sec")),
            "model": str(entry.get("model") or "").strip() or None,
            "prompt": str(entry.get("prompt") or ""),
            "system_prompt": str(entry.get("system_prompt") or ""),
        }

    def _summary_retention_cutoff(self) -> Optional[float]:
        if self.summary_retention_days <= 0:
            return None
        return time.time() - self.summary_retention_days * 86400.0

    def _filter_summary_history_retention(
        self,
        logs: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        cutoff = self._summary_retention_cutoff()
        out: List[Dict[str, Any]] = []
        for log in logs:
            normalized = self._normalize_summary_log_entry(log)
            if normalized is None:
                continue
            created = self._coerce_float(normalized.get("created_at"))
            if cutoff is not None and created is not None and created < cutoff:
                continue
            out.append(normalized)
        return out

    def _filter_normalized_summary_history_retention(
        self,
        logs: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        cutoff = self._summary_retention_cutoff()
        out: List[Dict[str, Any]] = []
        for log in logs:
            if not isinstance(log, Mapping):
                continue
            created = self._coerce_float(log.get("created_at"))
            if cutoff is not None and created is not None and created < cutoff:
                continue
            out.append(dict(log))
        return out

    def _filter_summary_runs_retention(
        self,
        runs: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        cutoff = self._summary_retention_cutoff()
        out: List[Dict[str, Any]] = []
        for run in runs:
            normalized = self._normalize_summary_run_entry(run)
            if normalized is None:
                continue
            ended = self._coerce_float(normalized.get("ended_at"))
            started = self._coerce_float(normalized.get("started_at"))
            anchor = ended if ended is not None else started
            if cutoff is not None and anchor is not None and anchor < cutoff:
                continue
            out.append(normalized)
        return out

    def _persist_summary_state_locked(self) -> None:
        history_payload: Dict[str, List[Dict[str, Any]]] = {}
        for channel_id, logs in self.summary_history.items():
            if not logs:
                continue
            history_payload[str(channel_id)] = [dict(log) for log in logs if isinstance(log, Mapping)]
        runs_payload: Dict[str, List[Dict[str, Any]]] = {}
        for channel_id, runs in self.summary_runs.items():
            if not runs:
                continue
            runs_payload[str(channel_id)] = [dict(run) for run in runs if isinstance(run, Mapping)]
        routine_payload: Dict[str, Dict[str, Any]] = {}
        for channel_id, routine in self.channel_routine_context.items():
            if not isinstance(routine, Mapping):
                continue
            routine_payload[str(channel_id)] = dict(routine)
        prompt_payload = {
            "stream_system_prompt": str(self.system_prompt or ""),
            "alert_policy_prompt": str(self.alert_policy_prompt or ""),
            "rollup_prompts": {
                "L1": str(self.rollup_llm_system_prompts.get("L1") or ""),
                "L2": str(self.rollup_llm_system_prompts.get("L2") or ""),
                "L3": str(self.rollup_llm_system_prompts.get("L3") or ""),
            },
            "bookmark_enabled": bool(self.default_bookmark_enabled),
            "bookmark_cooldown_sec": float(self.default_bookmark_cooldown_sec),
            "json_alert_prompt": str(self.default_json_alert_prompt or DEFAULT_ALERTS_JSON_PROMPT),
            "channel_overrides": {
                str(channel_id): dict(settings)
                for channel_id, settings in self.channel_prompt_overrides.items()
                if isinstance(settings, Mapping)
            },
        }
        payload = {
            "version": 2,
            "updated_at": time.time(),
            "summary_history": history_payload,
            "summary_runs": runs_payload,
            "channel_routines": routine_payload,
            "prompt_settings": prompt_payload,
        }
        state_store = getattr(self, "runtime_state_store", None)
        if state_store is not None:
            try:
                state_store.save_state("luxriot_summary_state", payload)
            except Exception:
                return
            return
        path = self.summary_state_file
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = path.with_suffix(f"{path.suffix}.tmp")
            tmp_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_file.replace(path)
        except Exception:
            return

    def _persist_summary_state_if_due_locked(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if (
            not force
            and self.summary_state_persist_interval_sec > 0
            and self._summary_state_last_persist > 0
            and now - self._summary_state_last_persist < self.summary_state_persist_interval_sec
        ):
            self._summary_state_dirty = True
            return
        self._summary_state_last_persist = now
        self._summary_state_dirty = False
        self._persist_summary_state_locked()

    def _load_summary_state_from_disk(self) -> None:
        payload: Optional[Dict[str, Any]] = None
        state_store = getattr(self, "runtime_state_store", None)
        if state_store is not None:
            try:
                loaded_payload = state_store.load_state("luxriot_summary_state")
                if isinstance(loaded_payload, Mapping):
                    payload = dict(loaded_payload)
            except Exception:
                payload = None
        if payload is None:
            path = self.summary_state_file
            if not path.exists():
                return
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return
        history_raw = payload.get("summary_history") if isinstance(payload, Mapping) else None
        runs_raw = payload.get("summary_runs") if isinstance(payload, Mapping) else None
        routines_raw = payload.get("channel_routines") if isinstance(payload, Mapping) else None
        prompt_settings_raw = payload.get("prompt_settings") if isinstance(payload, Mapping) else None
        loaded_history: Dict[int, List[Dict[str, Any]]] = {}
        if isinstance(history_raw, Mapping):
            for channel_key, logs_value in history_raw.items():
                channel_id = _parse_optional_int(channel_key)
                if channel_id is None or not isinstance(logs_value, Sequence) or isinstance(logs_value, (str, bytes, bytearray)):
                    continue
                normalized_logs: List[Dict[str, Any]] = []
                for raw_log in logs_value:
                    if not isinstance(raw_log, Mapping):
                        continue
                    normalized = self._normalize_summary_log_entry(cast(Mapping[str, Any], raw_log))
                    if normalized is not None:
                        normalized_logs.append(normalized)
                if not normalized_logs:
                    continue
                combined = self._filter_summary_history_retention(
                    self._combine_summary_logs([], normalized_logs)
                )
                if len(combined) > self.summary_history_limit:
                    combined = combined[-self.summary_history_limit :]
                loaded_history[int(channel_id)] = combined
        loaded_runs: Dict[int, List[Dict[str, Any]]] = {}
        if isinstance(runs_raw, Mapping):
            for channel_key, runs_value in runs_raw.items():
                channel_id = _parse_optional_int(channel_key)
                if channel_id is None or not isinstance(runs_value, Sequence) or isinstance(runs_value, (str, bytes, bytearray)):
                    continue
                dedup: Dict[str, Dict[str, Any]] = {}
                for raw_run in runs_value:
                    if not isinstance(raw_run, Mapping):
                        continue
                    normalized_run = self._normalize_summary_run_entry(cast(Mapping[str, Any], raw_run))
                    if normalized_run is None:
                        continue
                    normalized_run["running"] = False
                    if normalized_run.get("ended_at") is None:
                        normalized_run["ended_at"] = normalized_run.get("started_at")
                    dedup[str(normalized_run["run_id"])] = normalized_run
                runs_list = sorted(
                    self._filter_summary_runs_retention(dedup.values()),
                    key=lambda row: float(self._coerce_float(row.get("started_at")) or 0.0),
                    reverse=True,
                )
                if runs_list:
                    loaded_runs[int(channel_id)] = runs_list
        loaded_routines: Dict[int, Dict[str, Any]] = {}
        if isinstance(routines_raw, Mapping):
            for channel_key, routine_value in routines_raw.items():
                channel_id = _parse_optional_int(channel_key)
                if channel_id is None or not isinstance(routine_value, Mapping):
                    continue
                routine_text = str(routine_value.get("routine") or "").strip()
                if not routine_text:
                    continue
                loaded_entry: Dict[str, Any] = {
                    "channel_id": int(channel_id),
                    "rollup_id": str(routine_value.get("rollup_id") or "").strip(),
                    "source_level": str(routine_value.get("source_level") or "").strip(),
                    "window_end": float(self._coerce_float(routine_value.get("window_end")) or 0.0),
                    "routine": routine_text,
                    "updated_at": float(self._coerce_float(routine_value.get("updated_at")) or time.time()),
                }
                memory_raw = routine_value.get("memory")
                if isinstance(memory_raw, Mapping):
                    loaded_entry["memory"] = dict(memory_raw)
                loaded_routines[int(channel_id)] = loaded_entry
        loaded_stream_system_prompt: Optional[str] = None
        loaded_alert_policy_prompt: Optional[str] = None
        loaded_rollup_prompts: Dict[str, str] = {}
        loaded_channel_prompt_overrides: Dict[int, Dict[str, Any]] = {}
        loaded_default_bookmark_enabled: Optional[bool] = None
        loaded_default_bookmark_cooldown_sec: Optional[float] = None
        loaded_default_json_alert_prompt: Optional[str] = None
        if isinstance(prompt_settings_raw, Mapping):
            if "stream_system_prompt" in prompt_settings_raw:
                loaded_stream_system_prompt = str(prompt_settings_raw.get("stream_system_prompt") or "")
            elif "system_prompt" in prompt_settings_raw:
                loaded_stream_system_prompt = str(prompt_settings_raw.get("system_prompt") or "")
            if "alert_policy_prompt" in prompt_settings_raw:
                loaded_alert_policy_prompt = str(prompt_settings_raw.get("alert_policy_prompt") or "")
            if "bookmark_enabled" in prompt_settings_raw:
                loaded_default_bookmark_enabled = bool(prompt_settings_raw.get("bookmark_enabled"))
            if "bookmark_cooldown_sec" in prompt_settings_raw:
                raw_cooldown = self._coerce_float(prompt_settings_raw.get("bookmark_cooldown_sec"))
                loaded_default_bookmark_cooldown_sec = max(0.0, raw_cooldown if raw_cooldown is not None else 0.0)
            if "json_alert_prompt" in prompt_settings_raw:
                loaded_default_json_alert_prompt = self._normalize_json_alert_prompt(
                    prompt_settings_raw.get("json_alert_prompt")
                )
            rollup_prompts_raw = prompt_settings_raw.get("rollup_prompts")
            if isinstance(rollup_prompts_raw, Mapping):
                for raw_level, raw_prompt in rollup_prompts_raw.items():
                    level = self._normalize_rollup_level(raw_level)
                    if level in {"L1", "L2", "L3"}:
                        loaded_rollup_prompts[level] = str(raw_prompt or "").strip()
            # Backward-compatibility for flat keys.
            for level in ("L1", "L2", "L3"):
                flat_key = f"rollup_{level.lower()}_system_prompt"
                if level in loaded_rollup_prompts:
                    continue
                if flat_key in prompt_settings_raw:
                    loaded_rollup_prompts[level] = str(prompt_settings_raw.get(flat_key) or "").strip()
            channel_overrides_raw = prompt_settings_raw.get("channel_overrides")
            if isinstance(channel_overrides_raw, Mapping):
                for channel_key, channel_payload in channel_overrides_raw.items():
                    channel_id = _parse_optional_int(channel_key)
                    if channel_id is None or not isinstance(channel_payload, Mapping):
                        continue
                    parsed_channel_payload: Dict[str, Any] = {}
                    if "stream_system_prompt" in channel_payload:
                        parsed_channel_payload["stream_system_prompt"] = str(channel_payload.get("stream_system_prompt") or "")
                    if "alert_policy_prompt" in channel_payload:
                        parsed_channel_payload["alert_policy_prompt"] = str(channel_payload.get("alert_policy_prompt") or "")
                    if "bookmark_enabled" in channel_payload:
                        parsed_channel_payload["bookmark_enabled"] = bool(channel_payload.get("bookmark_enabled"))
                    if "bookmark_cooldown_sec" in channel_payload:
                        raw_channel_cooldown = self._coerce_float(channel_payload.get("bookmark_cooldown_sec"))
                        parsed_channel_payload["bookmark_cooldown_sec"] = max(
                            0.0, raw_channel_cooldown if raw_channel_cooldown is not None else 0.0
                        )
                    if "capture_interval_sec" in channel_payload:
                        channel_interval = self._normalize_capture_interval_sec(
                            channel_payload.get("capture_interval_sec")
                        )
                        if channel_interval is not None:
                            parsed_channel_payload["capture_interval_sec"] = channel_interval
                    if "json_alert_prompt" in channel_payload:
                        parsed_channel_payload["json_alert_prompt"] = self._normalize_json_alert_prompt(
                            channel_payload.get("json_alert_prompt")
                        )
                    channel_rollup_prompts_raw = channel_payload.get("rollup_prompts")
                    if isinstance(channel_rollup_prompts_raw, Mapping):
                        parsed_rollup_prompts: Dict[str, str] = {}
                        for raw_level, raw_prompt in channel_rollup_prompts_raw.items():
                            level = self._normalize_rollup_level(raw_level)
                            if level in {"L1", "L2", "L3"}:
                                parsed_rollup_prompts[level] = str(raw_prompt or "").strip()
                        if parsed_rollup_prompts:
                            parsed_channel_payload["rollup_prompts"] = parsed_rollup_prompts
                    for level in ("L1", "L2", "L3"):
                        flat_key = f"rollup_{level.lower()}_system_prompt"
                        if flat_key not in channel_payload:
                            continue
                        rollup_payload = parsed_channel_payload.setdefault("rollup_prompts", {})
                        if isinstance(rollup_payload, dict):
                            rollup_payload[level] = str(channel_payload.get(flat_key) or "").strip()
                    if parsed_channel_payload:
                        loaded_channel_prompt_overrides[int(channel_id)] = parsed_channel_payload
        with self.cache_lock:
            self.summary_history = loaded_history
            self.summary_runs = loaded_runs
            self.channel_routine_context = loaded_routines
            self.active_summary_runs = {}
            self.channel_prompt_overrides = loaded_channel_prompt_overrides
            self._rebuild_channel_status_digest_locked()
            if loaded_stream_system_prompt is not None:
                self.system_prompt = loaded_stream_system_prompt
            if loaded_alert_policy_prompt is not None:
                self.alert_policy_prompt = loaded_alert_policy_prompt
            if loaded_default_bookmark_enabled is not None:
                self.default_bookmark_enabled = loaded_default_bookmark_enabled
            if loaded_default_bookmark_cooldown_sec is not None:
                self.default_bookmark_cooldown_sec = loaded_default_bookmark_cooldown_sec
            if loaded_default_json_alert_prompt is not None:
                self.default_json_alert_prompt = loaded_default_json_alert_prompt or self.default_json_alert_prompt
            for level, prompt_text in loaded_rollup_prompts.items():
                self.rollup_llm_system_prompts[level] = prompt_text

    def persist_summary_state(self) -> None:
        with self.cache_lock:
            self._persist_summary_state_locked()

    def _load_desired_live_sessions(self) -> Dict[int, Dict[str, Any]]:
        state_store = getattr(self, "runtime_state_store", None)
        if state_store is None:
            return {}
        payload = state_store.load_state(self.DESIRED_LIVE_SESSIONS_KEY)
        if not isinstance(payload, Mapping):
            return {}
        sessions_raw = payload.get("sessions")
        if not isinstance(sessions_raw, Mapping):
            return {}
        desired: Dict[int, Dict[str, Any]] = {}
        for raw_channel_id, raw_state in sessions_raw.items():
            channel_id = _parse_optional_int(raw_channel_id)
            if channel_id is None or channel_id <= 0 or not isinstance(raw_state, Mapping):
                continue
            desired[channel_id] = dict(raw_state)
        return desired

    def _save_desired_live_sessions(self, sessions: Mapping[int, Mapping[str, Any]]) -> None:
        state_store = getattr(self, "runtime_state_store", None)
        if state_store is None:
            return
        payload = {
            "version": 1,
            "updated_at": time.time(),
            "sessions": {
                str(int(channel_id)): dict(state)
                for channel_id, state in sessions.items()
                if int(channel_id) > 0
            },
        }
        state_store.save_state(self.DESIRED_LIVE_SESSIONS_KEY, payload)

    def _set_desired_live_session(
        self,
        channel_id: int,
        *,
        enabled: bool,
        batch_size: Optional[int] = None,
        prompt: Optional[str] = None,
        model_hint: Optional[str] = None,
        interval_sec: Optional[float] = None,
        restore_error: Optional[str] = None,
    ) -> None:
        desired = self._load_desired_live_sessions()
        now = time.time()
        current = dict(desired.get(int(channel_id)) or {})
        current.update(
            {
                "enabled": bool(enabled),
                "channel_id": int(channel_id),
                "updated_at": now,
            }
        )
        if batch_size is not None:
            current["batch_size"] = int(batch_size)
        if prompt is not None:
            current["prompt"] = str(prompt)
        if model_hint is not None:
            current["model_hint"] = str(model_hint)
        elif enabled:
            current["model_hint"] = ""
        if interval_sec is not None:
            current["interval_sec"] = float(interval_sec)
        if restore_error:
            current["last_restore_error"] = str(restore_error)[:500]
            current["last_restore_error_at"] = now
            current["restore_attempts"] = int(current.get("restore_attempts") or 0) + 1
        elif enabled:
            current.pop("last_restore_error", None)
            current.pop("last_restore_error_at", None)
        desired[int(channel_id)] = current
        self._save_desired_live_sessions(desired)

    def restore_desired_live_sessions(self, *, max_channels: Optional[int] = None) -> Dict[str, Any]:
        try:
            desired = self._load_desired_live_sessions()
        except Exception as exc:
            return {
                "ok": False,
                "status": "desired_state_unavailable",
                "error": type(exc).__name__,
            }
        enabled_items = [
            (channel_id, state)
            for channel_id, state in sorted(desired.items())
            if bool(state.get("enabled"))
        ]
        if max_channels is not None and max_channels > 0:
            enabled_items = enabled_items[: int(max_channels)]
        restored: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []
        skipped: List[int] = []
        for index, (channel_id, state) in enumerate(enabled_items):
            with self.cache_lock:
                already_running = channel_id in self.sessions
            if already_running:
                skipped.append(channel_id)
                continue
            if index:
                time.sleep(0.2)
            try:
                status = self.start_session(
                    channel_id=channel_id,
                    batch_size=_parse_optional_int(state.get("batch_size")),
                    prompt=str(state.get("prompt") or ""),
                    model_hint=str(state.get("model_hint") or "").strip() or None,
                    interval_sec=self._normalize_capture_interval_sec(state.get("interval_sec")),
                    update_desired=True,
                )
                restored.append(
                    {
                        "channel_id": channel_id,
                        "model": status.get("model"),
                        "batch_size": status.get("batch_size"),
                    }
                )
                with self.cache_lock:
                    self.live_session_restore_errors.pop(channel_id, None)
            except Exception as exc:
                message = str(exc)[:500]
                failed.append(
                    {
                        "channel_id": channel_id,
                        "error": type(exc).__name__,
                        "message": message,
                    }
                )
                with self.cache_lock:
                    self.live_session_restore_errors[channel_id] = message
                try:
                    self._set_desired_live_session(
                        channel_id,
                        enabled=True,
                        restore_error=message,
                    )
                except Exception:
                    pass
        return {
            "ok": not failed,
            "status": "restored" if not failed else "partial",
            "desired_count": len(enabled_items),
            "restored_count": len(restored),
            "skipped_count": len(skipped),
            "failed_count": len(failed),
            "restored": restored,
            "skipped": skipped,
            "failed": failed,
        }

    @staticmethod
    def _coerce_float(value: object) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            num = float(value)
            return num if math.isfinite(num) else None
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                num = float(text)
            except ValueError:
                return None
            return num if math.isfinite(num) else None
        try:
            num = float(cast(Any, value))
        except Exception:
            return None
        return num if math.isfinite(num) else None

    def _normalize_capture_interval_sec(self, value: object) -> Optional[float]:
        interval = self._coerce_float(value)
        if interval is None or interval <= 0:
            return None
        return max(0.2, min(300.0, float(interval)))

    def _default_capture_interval_sec(self) -> float:
        return self._normalize_capture_interval_sec(
            getattr(self.config, "LUXRIOT_SNAPSHOT_INTERVAL", 5)
        ) or 5.0

    def _get_capture_interval_sec_locked(self, channel_id: Optional[int] = None) -> float:
        interval = self._default_capture_interval_sec()
        if channel_id is not None:
            overrides = self.channel_prompt_overrides.get(int(channel_id))
            if isinstance(overrides, Mapping):
                channel_interval = self._normalize_capture_interval_sec(
                    overrides.get("capture_interval_sec")
                )
                if channel_interval is not None:
                    interval = channel_interval
        return interval

    def _generate_run_id_locked(self, channel_id: int) -> str:
        base = f"ch{channel_id}-{int(time.time() * 1000)}"
        existing = {
            str(run.get("run_id") or "").strip()
            for run in self.summary_runs.get(channel_id, [])
            if isinstance(run, Mapping)
        }
        run_id = base
        suffix = 1
        while run_id in existing:
            run_id = f"{base}-{suffix}"
            suffix += 1
        return run_id

    def _open_run_locked(
        self,
        channel_id: int,
        batch_size: int,
        prompt: str,
        model_hint: Optional[str],
        system_prompt: Optional[str],
        interval_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        started_at = time.time()
        run = {
            "run_id": self._generate_run_id_locked(channel_id),
            "channel_id": channel_id,
            "started_at": started_at,
            "ended_at": None,
            "running": True,
            "batch_size": int(batch_size),
            "interval_sec": self._normalize_capture_interval_sec(interval_sec),
            "model": (model_hint or "").strip() or None,
            "prompt": prompt or "",
            "system_prompt": system_prompt or "",
        }
        self.summary_runs.setdefault(channel_id, []).append(run)
        self.active_summary_runs[channel_id] = str(run["run_id"])
        self._persist_summary_state_locked()
        return dict(run)

    def _close_run_locked(self, channel_id: int, run_id: Optional[str]) -> None:
        normalized_run_id = str(run_id or "").strip()
        active_run_id = str(self.active_summary_runs.get(channel_id) or "").strip()
        target_run_id = normalized_run_id or active_run_id
        if not target_run_id:
            return
        runs = self.summary_runs.get(channel_id, [])
        for run in runs:
            if str(run.get("run_id") or "").strip() == target_run_id:
                run["running"] = False
                if not run.get("ended_at"):
                    run["ended_at"] = time.time()
                break
        self.summary_runs[channel_id] = self._filter_summary_runs_retention(runs)
        if active_run_id == target_run_id:
            self.active_summary_runs.pop(channel_id, None)
        self._persist_summary_state_locked()

    @staticmethod
    def _filter_summary_logs(
        logs: Sequence[Mapping[str, Any]],
        run_id: Optional[str] = None,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        normalized_run_id = str(run_id or "").strip()
        filtered: List[Dict[str, Any]] = []
        for item in logs:
            if not isinstance(item, Mapping):
                continue
            if normalized_run_id:
                item_run = str(item.get("run_id") or "").strip()
                if item_run != normalized_run_id:
                    continue
            item_start, item_end = LuxriotManager._summary_log_bounds_seconds(item)
            if start_ts is not None and (item_end is None or item_end < start_ts):
                continue
            if end_ts is not None and (item_start is None or item_start > end_ts):
                continue
            filtered.append(dict(item))
        return filtered

    @staticmethod
    def _resolve_run_selector(
        run_selector: Optional[str],
        runs: Sequence[Mapping[str, Any]],
        running_run_id: Optional[str],
    ) -> Tuple[str, Optional[str], Optional[str]]:
        normalized_selector = str(run_selector or "").strip()
        if not normalized_selector:
            normalized_selector = "latest"
        available = [
            str(run.get("run_id") or "").strip()
            for run in runs
            if isinstance(run, Mapping) and str(run.get("run_id") or "").strip()
        ]
        latest_run_id = available[0] if available else None
        running_id = str(running_run_id or "").strip() or None
        lowered = normalized_selector.lower()
        if lowered == "all":
            return "all", None, latest_run_id
        if lowered == "live":
            target = running_id or latest_run_id
            return "live", target, latest_run_id
        if lowered == "latest":
            return "latest", latest_run_id, latest_run_id
        if normalized_selector in available:
            return normalized_selector, normalized_selector, latest_run_id
        return "latest", latest_run_id, latest_run_id

    @staticmethod
    def _stable_id(parts: Sequence[str], length: int = 12) -> str:
        payload = "|".join(str(part) for part in parts).encode("utf-8", errors="ignore")
        digest = hashlib.sha1(payload).hexdigest()
        return digest[: max(6, int(length))]

    def _source_signature(self, source_ids: Sequence[str]) -> str:
        normalized = [str(item or "").strip() for item in source_ids if str(item or "").strip()]
        if not normalized:
            return ""
        return self._stable_id(normalized, length=16)

    @classmethod
    def _signature_value(cls, value: object) -> object:
        if isinstance(value, Mapping):
            return {
                str(key): cls._signature_value(item_value)
                for key, item_value in sorted(value.items(), key=lambda item: str(item[0]))
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [cls._signature_value(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @classmethod
    def _signature_json(cls, value: object) -> str:
        try:
            return json.dumps(
                cls._signature_value(value),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except Exception:
            return str(cls._signature_value(value))

    @staticmethod
    def _text_hash(text: object, length: int = 16) -> str:
        payload = str(text or "").strip().encode("utf-8", errors="ignore")
        if not payload:
            return ""
        return hashlib.sha1(payload).hexdigest()[: max(6, int(length))]

    def _rollup_source_signature(
        self,
        children: Sequence[Mapping[str, Any]],
        source_ids: Sequence[str],
    ) -> str:
        normalized_source_ids = [
            str(item or "").strip()
            for item in source_ids
            if str(item or "").strip()
        ]
        child_payloads: List[Dict[str, Any]] = []
        for child in children:
            if not isinstance(child, Mapping):
                continue
            signal_digest = child.get("signal_digest")
            child_payloads.append(
                {
                    "source_id": str(child.get("rollup_id") or "").strip(),
                    "summary_hash": self._text_hash(child.get("summary")),
                    "alert_counts": self._normalize_alert_counts(child.get("alert_counts")),
                    "signal_digest": signal_digest if isinstance(signal_digest, Mapping) else {},
                    "source_signature": str(child.get("source_signature") or "").strip(),
                }
            )
        if not child_payloads:
            return self._source_signature(normalized_source_ids)
        return self._stable_id(
            [
                self._signature_json(
                    {
                        "source_ids": normalized_source_ids,
                        "children": child_payloads,
                    }
                )
            ],
            length=16,
        )

    @staticmethod
    def _summary_headline(text: object, max_len: int = 180) -> str:
        normalized = " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())
        if not normalized:
            return ""
        if len(normalized) <= max_len:
            return normalized
        return f"{normalized[: max_len - 3].rstrip()}..."

    @staticmethod
    def _sanitize_l0_summary(text: object, max_len: int = 520) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        upper_raw = raw.upper()
        cut_points = [
            idx
            for marker in ("ALERTS_JSON:", "MEMORY_UPDATE_JSON:")
            for idx in [upper_raw.find(marker)]
            if idx >= 0
        ]
        if cut_points:
            raw = raw[: min(cut_points)].rstrip()
        # Remove frequent boilerplate prefixes produced by VLM.
        cleaned = re.sub(
            r"^\s*As a security expert(?:[^:.\n]*[:.])\s*",
            "",
            raw,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^\s*(Summary|Scene Overview|Detailed Analysis)\s*:\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        normalized = " ".join(cleaned.replace("\r", " ").replace("\n", " ").split())
        if len(normalized) <= max_len:
            return normalized
        return f"{normalized[: max_len - 3].rstrip()}..."

    @staticmethod
    def _window_label(start_ts: Optional[float], end_ts: Optional[float]) -> str:
        if isinstance(start_ts, (int, float)):
            start_label = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(start_ts)))
        else:
            start_label = "n/a"
        if isinstance(end_ts, (int, float)):
            end_label = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(end_ts)))
        else:
            end_label = "n/a"
        return f"{start_label} -> {end_label}"

    @staticmethod
    def _extract_markdown_section(text: str, heading: str) -> str:
        pattern = re.compile(
            rf"^###\s*{re.escape(heading)}\s*$\n(?P<body>.*?)(?:\n###\s|\Z)",
            re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        match = pattern.search(text)
        if not match:
            return ""
        body = match.group("body").strip()
        return body

    @staticmethod
    def _truncate_text(text: object, max_len: int) -> str:
        normalized = " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())
        if not normalized:
            return ""
        limit = max(16, int(max_len))
        if len(normalized) <= limit:
            return normalized
        return f"{normalized[: limit - 3].rstrip()}..."

    @classmethod
    def _extract_json_marker_payload(cls, text: str, marker: str) -> Optional[Mapping[str, Any]]:
        haystack = str(text or "")
        marker_text = str(marker or "").strip()
        if not haystack or not marker_text:
            return None
        idx = haystack.upper().find(marker_text.upper())
        if idx < 0:
            return None
        tail = haystack[idx + len(marker_text) :].strip()
        if not tail:
            return None
        try:
            payload, _offset = json.JSONDecoder().raw_decode(tail)
        except Exception:
            return None
        if isinstance(payload, Mapping):
            return cast(Mapping[str, Any], payload)
        return None

    @staticmethod
    def _compact_json_key(value: object) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())

    @classmethod
    def _payload_value(cls, payload: Mapping[str, Any], *aliases: str) -> Any:
        for alias in aliases:
            if alias in payload:
                return payload.get(alias)
        compact_payload = {
            cls._compact_json_key(key): value
            for key, value in payload.items()
        }
        for alias in aliases:
            compact_alias = cls._compact_json_key(alias)
            if compact_alias in compact_payload:
                return compact_payload[compact_alias]
        return None

    @classmethod
    def _coerce_memory_items(cls, value: object, *, max_items: int = 5, max_len: int = 220) -> List[str]:
        if value is None:
            return []
        if isinstance(value, Mapping):
            raw_items: Sequence[object] = [value]
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            raw_items = cast(Sequence[object], value)
        else:
            raw_items = [value]
        out: List[str] = []
        seen: Set[str] = set()
        for raw_item in raw_items:
            if isinstance(raw_item, Mapping):
                parts: List[str] = []
                for key in ("time", "severity", "event", "evidence", "note", "reason"):
                    if key not in raw_item:
                        continue
                    part = cls._truncate_text(raw_item.get(key), 120 if key != "event" else 180)
                    if part:
                        parts.append(part)
                text = " | ".join(parts)
            else:
                text = cls._truncate_text(raw_item, max_len)
            text = cls._truncate_text(text, max_len)
            if not text:
                continue
            dedupe_key = text.lower()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            out.append(text)
            if len(out) >= max(1, int(max_items)):
                break
        return out

    @staticmethod
    def _dedupe_memory_items(*groups: Sequence[str], max_items: int = 6) -> List[str]:
        out: List[str] = []
        seen: Set[str] = set()
        for group in groups:
            for item in group:
                text = " ".join(str(item or "").split())
                if not text:
                    continue
                key = text.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(text)
                if len(out) >= max(1, int(max_items)):
                    return out
        return out

    @classmethod
    def _render_channel_memory_text(cls, memory: Mapping[str, Any], max_len: int = 1600) -> str:
        routine = cls._truncate_text(memory.get("routine_baseline"), 420)
        active = cls._coerce_memory_items(memory.get("active_watchlist"), max_items=4, max_len=180)
        deviations = cls._coerce_memory_items(memory.get("preserved_deviations"), max_items=5, max_len=220)
        tuning = cls._coerce_memory_items(memory.get("alert_tuning_notes"), max_items=4, max_len=200)
        ignore = cls._coerce_memory_items(memory.get("ignore_as_routine"), max_items=4, max_len=180)
        lines: List[str] = []
        if routine:
            lines.append(f"Routine baseline: {routine}")
        if active:
            lines.append("Active watchlist: " + "; ".join(active))
        if deviations:
            lines.append("Preserved deviations: " + "; ".join(deviations))
        if tuning:
            lines.append("Alert tuning notes: " + "; ".join(tuning))
        if ignore:
            lines.append("Ignore as routine/noise: " + "; ".join(ignore))
        rendered = "\n".join(lines).strip()
        return cls._truncate_text(rendered, max_len)

    def _extract_memory_update(self, summary_text: object) -> Dict[str, Any]:
        text = str(summary_text or "").strip()
        if not text:
            return {}
        payload = self._extract_json_marker_payload(text, "MEMORY_UPDATE_JSON:")
        if payload:
            memory = {
                "routine_baseline": self._truncate_text(
                    self._payload_value(
                        payload,
                        "routine_baseline",
                        "routineBaseline",
                        "scene_baseline",
                        "sceneBaseline",
                        "persistent_patterns",
                        "persistentPatterns",
                    ),
                    420,
                ),
                "active_watchlist": self._coerce_memory_items(
                    self._payload_value(payload, "active_watchlist", "activeWatchlist", "watchlist"),
                    max_items=4,
                    max_len=180,
                ),
                "preserved_deviations": self._coerce_memory_items(
                    self._payload_value(
                        payload,
                        "preserved_deviations",
                        "preservedDeviations",
                        "recent_deviations",
                        "recentDeviations",
                        "notable_events",
                        "notableEvents",
                    ),
                    max_items=5,
                    max_len=220,
                ),
                "alert_tuning_notes": self._coerce_memory_items(
                    self._payload_value(
                        payload,
                        "alert_tuning_notes",
                        "alertTuningNotes",
                        "alerttuningnotes",
                        "alert_tuning",
                        "alertTuning",
                    ),
                    max_items=4,
                    max_len=200,
                ),
                "ignore_as_routine": self._coerce_memory_items(
                    self._payload_value(
                        payload,
                        "ignore_as_routine",
                        "ignoreAsRoutine",
                        "ignoreasroutine",
                        "routine_noise",
                        "routineNoise",
                        "recurring_false_positives",
                        "recurringFalsePositives",
                    ),
                    max_items=4,
                    max_len=180,
                ),
            }
            memory["text"] = self._render_channel_memory_text(memory)
            return {key: value for key, value in memory.items() if value}

        section_map = {
            "routine_baseline": (
                "Routine Baseline",
                "Scene Baseline",
                "Persistent Patterns",
                "Time-of-day Routine",
            ),
            "active_watchlist": ("Active Watchlist",),
            "preserved_deviations": (
                "Preserved Deviations",
                "Significant Changes",
                "Key Changes",
                "Notable Events",
                "Risks and Follow-ups",
            ),
            "alert_tuning_notes": (
                "Alert Tuning Notes",
                "Alerts/Signals",
                "Alert Ledger",
                "Operator Notes",
            ),
            "ignore_as_routine": (
                "Recurring False Positives",
                "Ignore as Routine",
            ),
        }
        memory: Dict[str, Any] = {}
        for target_key, headings in section_map.items():
            chunks: List[str] = []
            for heading in headings:
                body = self._extract_markdown_section(text, heading)
                if body:
                    chunks.append(body)
            if not chunks:
                continue
            if target_key == "routine_baseline":
                memory[target_key] = self._truncate_text(" ".join(chunks), 420)
            else:
                memory[target_key] = self._coerce_memory_items(chunks, max_items=4, max_len=220)
        if not memory:
            cleaned = self._sanitize_l0_summary(text, max_len=520)
            if cleaned:
                memory["routine_baseline"] = cleaned
        memory["text"] = self._render_channel_memory_text(memory)
        return {key: value for key, value in memory.items() if value}

    def _update_channel_routine_context(
        self,
        channel_id: int,
        rollup_id: str,
        summary_text: object,
        window_end: Optional[float],
        level: Optional[str] = None,
    ) -> None:
        extracted_memory = self._extract_memory_update(summary_text)
        if not extracted_memory:
            return
        channel_key = int(channel_id)
        rollup_key = str(rollup_id or "").strip()
        window_end_value = self._coerce_float(window_end) or 0.0
        source_level = self._normalize_rollup_level(level) or ""
        changed = False
        with self.cache_lock:
            current = self.channel_routine_context.get(channel_key)
            current_window_end = self._coerce_float(current.get("window_end")) if isinstance(current, Mapping) else None
            current_window_end_value = float(current_window_end or 0.0)
            incoming_is_older = current is not None and current_window_end is not None and window_end_value < current_window_end_value
            current_rollup_id = str(current.get("rollup_id") or "").strip() if isinstance(current, Mapping) else ""
            current_hint = str(current.get("routine") or "").strip() if isinstance(current, Mapping) else ""
            current_memory_raw = current.get("memory") if isinstance(current, Mapping) else None
            current_memory = dict(current_memory_raw) if isinstance(current_memory_raw, Mapping) else {}
            merged_memory = dict(current_memory)
            routine_baseline = str(extracted_memory.get("routine_baseline") or "").strip()
            current_routine_baseline = str(merged_memory.get("routine_baseline") or "").strip()
            if routine_baseline and (not incoming_is_older or not current_routine_baseline):
                merged_memory["routine_baseline"] = routine_baseline
            for key, limit in (
                ("active_watchlist", 5),
                ("preserved_deviations", 7),
                ("alert_tuning_notes", 5),
                ("ignore_as_routine", 5),
            ):
                incoming_items = self._coerce_memory_items(extracted_memory.get(key), max_items=limit, max_len=220)
                existing = self._coerce_memory_items(merged_memory.get(key), max_items=limit, max_len=220)
                if incoming_is_older:
                    merged = self._dedupe_memory_items(existing, incoming_items, max_items=limit)
                else:
                    merged = self._dedupe_memory_items(incoming_items, existing, max_items=limit)
                if merged:
                    merged_memory[key] = merged
                elif key in merged_memory:
                    merged_memory.pop(key, None)
            routine_hint = self._render_channel_memory_text(merged_memory)
            if not routine_hint:
                routine_hint = str(extracted_memory.get("text") or "").strip()
            should_update = (
                current is None
                or window_end_value > current_window_end_value
                or (current_rollup_id == rollup_key and current_hint != routine_hint)
                or current_hint != routine_hint
            )
            if should_update:
                if incoming_is_older:
                    stored_rollup_id = current_rollup_id or rollup_key
                    stored_source_level = str(current.get("source_level") or "").strip() if isinstance(current, Mapping) else source_level
                    stored_window_end = current_window_end_value
                else:
                    stored_rollup_id = rollup_key
                    stored_source_level = source_level
                    stored_window_end = window_end_value
                self.channel_routine_context[channel_key] = {
                    "channel_id": channel_key,
                    "rollup_id": stored_rollup_id,
                    "source_level": stored_source_level,
                    "window_end": stored_window_end,
                    "routine": routine_hint,
                    "memory": merged_memory,
                    "updated_at": time.time(),
                }
                changed = True
            if changed:
                self._persist_summary_state_locked()

    def _get_channel_routine_prompt(self, channel_id: int) -> str:
        with self.cache_lock:
            current = self.channel_routine_context.get(int(channel_id))
        if not isinstance(current, Mapping):
            return ""
        routine = str(current.get("routine") or "").strip()
        return self._render_channel_memory_prompt(routine)

    def _build_vector_signal_bundle(
        self,
        channel_id: int,
        frames: Sequence[Mapping[str, Any]],
        *,
        batch_start_ms: Optional[int] = None,
        batch_end_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not self.vector_signals_enabled:
            return {}
        health: Dict[str, Any] = {"enabled": True}
        clip_signals, clip_health = self._clip_probe_vector_signals(
            int(channel_id),
            frames,
            batch_start_ms=batch_start_ms,
            batch_end_ms=batch_end_ms,
        )
        health.update(clip_health)
        road_cues, road_scene, road_health = self._road_cv_vector_signals(int(channel_id), frames)
        health.update(road_health)
        bundle: Dict[str, Any] = {
            "version": 1,
            "channel_id": int(channel_id),
            "semantics": "vector_homeostasis_attention_signal_not_visual_proof",
            "health": health,
        }
        if batch_start_ms is not None:
            bundle["batch_start_ms"] = int(batch_start_ms)
        if batch_end_ms is not None:
            bundle["batch_end_ms"] = int(batch_end_ms)
        if clip_signals:
            bundle["clip_probe_signals"] = clip_signals
        if road_cues:
            bundle["road_cv_cues"] = road_cues
        if road_scene:
            bundle["road_cv_scene"] = road_scene
        return self._compact_vector_signal(bundle)

    def _render_vector_signal_prompt(self, vector_signal: object) -> str:
        compact = self._compact_vector_signal(vector_signal)
        if not compact:
            return ""
        try:
            payload = json.dumps(compact, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except Exception:
            return ""
        return f"{VECTOR_SIGNAL_PROMPT_PREFIX}\nVECTOR_SIGNALS_JSON:\n{payload}"

    def compose_live_system_prompt(
        self,
        channel_id: int,
        base_prompt: Optional[str],
        vector_signal: Optional[Mapping[str, Any]] = None,
    ) -> str:
        rendered_json_prompt = self._get_rendered_json_alert_prompt(channel_id)
        base = self._strip_suffix_prompt(str(base_prompt or ""), rendered_json_prompt).strip()
        alert_policy = self._get_rendered_alert_policy_prompt(channel_id)
        routine = self._get_channel_routine_prompt(channel_id)
        vector_prompt = self._render_vector_signal_prompt(vector_signal)
        parts = [
            part
            for part in (base, alert_policy, routine, vector_prompt, LIVE_OBSERVATION_STATE_PROMPT, rendered_json_prompt)
            if str(part or "").strip()
        ]
        return "\n\n".join(str(part).strip() for part in parts)

    def _default_rollup_prompt_for_level_locked(self, level: str) -> str:
        normalized_level = self._normalize_rollup_level(level)
        by_level = str(self.rollup_llm_system_prompts.get(normalized_level) or "").strip()
        fallback = str(self.rollup_llm_system_prompt or "").strip()
        return by_level or fallback

    def _get_stream_system_prompt_locked(self, channel_id: Optional[int] = None) -> str:
        if channel_id is not None:
            overrides = self.channel_prompt_overrides.get(int(channel_id))
            if isinstance(overrides, Mapping) and "stream_system_prompt" in overrides:
                return str(overrides.get("stream_system_prompt") or "")
        return str(self.system_prompt or "")

    def _get_alert_policy_prompt_locked(self, channel_id: Optional[int] = None) -> str:
        if channel_id is not None:
            overrides = self.channel_prompt_overrides.get(int(channel_id))
            if isinstance(overrides, Mapping) and "alert_policy_prompt" in overrides:
                return str(overrides.get("alert_policy_prompt") or "")
        return str(self.alert_policy_prompt or "")

    def _get_channel_bookmark_settings_locked(self, channel_id: Optional[int] = None) -> Dict[str, Any]:
        enabled = bool(self.default_bookmark_enabled)
        cooldown_sec = float(max(0.0, self.default_bookmark_cooldown_sec))
        json_prompt = self._normalize_json_alert_prompt(self.default_json_alert_prompt)
        if channel_id is not None:
            overrides = self.channel_prompt_overrides.get(int(channel_id))
            if isinstance(overrides, Mapping):
                if "bookmark_enabled" in overrides:
                    enabled = bool(overrides.get("bookmark_enabled"))
                if "bookmark_cooldown_sec" in overrides:
                    raw_cooldown = self._coerce_float(overrides.get("bookmark_cooldown_sec"))
                    cooldown_sec = max(0.0, raw_cooldown if raw_cooldown is not None else 0.0)
                if "json_alert_prompt" in overrides:
                    json_prompt = str(overrides.get("json_alert_prompt") or "")
        json_prompt = self._normalize_json_alert_prompt(json_prompt)
        return {
            "bookmark_enabled": enabled,
            "bookmark_cooldown_sec": cooldown_sec,
            "json_alert_prompt": json_prompt,
        }

    @staticmethod
    def _render_json_alert_prompt(prompt_text: str, channel_id: int) -> str:
        rendered = str(prompt_text or "")
        replacement = str(int(channel_id))
        return (
            rendered
            .replace("{channel_id}", replacement)
            .replace("{CHANNEL_ID}", replacement)
            .replace("<CHANNEL_ID>", replacement)
        )

    def _get_rendered_json_alert_prompt(self, channel_id: int) -> str:
        with self.cache_lock:
            bookmark_settings = self._get_channel_bookmark_settings_locked(channel_id)
        json_prompt = str(bookmark_settings.get("json_alert_prompt") or "").strip()
        if not json_prompt:
            return ""
        return self._render_json_alert_prompt(json_prompt, int(channel_id)).strip()

    def _get_rendered_alert_policy_prompt(self, channel_id: int) -> str:
        with self.cache_lock:
            policy_prompt = self._get_alert_policy_prompt_locked(channel_id)
        return self._render_alert_policy_prompt(policy_prompt)

    def _get_rollup_system_prompt_locked(self, level: str, channel_id: Optional[int] = None) -> str:
        normalized_level = self._normalize_rollup_level(level)
        if channel_id is not None:
            overrides = self.channel_prompt_overrides.get(int(channel_id))
            if isinstance(overrides, Mapping):
                rollup_prompts = overrides.get("rollup_prompts")
                if isinstance(rollup_prompts, Mapping) and normalized_level in rollup_prompts:
                    return str(rollup_prompts.get(normalized_level) or "")
        return self._default_rollup_prompt_for_level_locked(normalized_level)

    def _get_rollup_model_hint_locked(self, channel_id: Optional[int] = None) -> Optional[str]:
        if channel_id is not None:
            overrides = self.channel_prompt_overrides.get(int(channel_id))
            if isinstance(overrides, Mapping):
                override_hint = str(overrides.get("model_hint") or "").strip()
                if override_hint:
                    return override_hint
            runs = self.summary_runs.get(int(channel_id), [])
            for run in reversed(runs):
                if not isinstance(run, Mapping):
                    continue
                model_hint = str(run.get("model") or "").strip()
                if model_hint:
                    return model_hint
        fallback_hint = str(self.rollup_llm_model_hint or "").strip()
        return fallback_hint or None

    def get_stream_system_prompt(self, channel_id: Optional[int] = None) -> str:
        with self.cache_lock:
            return self._get_stream_system_prompt_locked(channel_id)

    def get_effective_stream_system_prompt(self, channel_id: int) -> str:
        with self.cache_lock:
            base_prompt = self._get_stream_system_prompt_locked(channel_id)
            bookmark_settings = self._get_channel_bookmark_settings_locked(channel_id)
        json_prompt = str(bookmark_settings.get("json_alert_prompt") or "").strip()
        if json_prompt:
            json_prompt = self._render_json_alert_prompt(json_prompt, int(channel_id))
            return f"{base_prompt}\n\n{json_prompt}" if base_prompt else json_prompt
        return base_prompt

    def get_prompt_settings(self, channel_id: Optional[int] = None) -> Dict[str, Any]:
        with self.cache_lock:
            defaults_bookmark = self._get_channel_bookmark_settings_locked(None)
            defaults = {
                "stream_system_prompt": str(self.system_prompt or ""),
                "alert_policy_prompt": str(self.alert_policy_prompt or ""),
                "rollup_prompts": {
                    "L1": self._default_rollup_prompt_for_level_locked("L1"),
                    "L2": self._default_rollup_prompt_for_level_locked("L2"),
                    "L3": self._default_rollup_prompt_for_level_locked("L3"),
                },
                "capture_interval_sec": self._default_capture_interval_sec(),
                "bookmark_enabled": bool(defaults_bookmark.get("bookmark_enabled")),
                "bookmark_cooldown_sec": float(defaults_bookmark.get("bookmark_cooldown_sec") or 0.0),
                "json_alert_prompt": str(defaults_bookmark.get("json_alert_prompt") or DEFAULT_ALERTS_JSON_PROMPT),
            }
            effective_stream_prompt = self._get_stream_system_prompt_locked(channel_id)
            effective_alert_policy_prompt = self._get_alert_policy_prompt_locked(channel_id)
            prompt_health = self._legacy_alert_prompt_health(
                effective_stream_prompt,
                effective_alert_policy_prompt,
            )
            effective_rollup_prompts = {
                "L1": self._get_rollup_system_prompt_locked("L1", channel_id),
                "L2": self._get_rollup_system_prompt_locked("L2", channel_id),
                "L3": self._get_rollup_system_prompt_locked("L3", channel_id),
            }
            effective_capture_interval_sec = self._get_capture_interval_sec_locked(channel_id)
            effective_bookmark = self._get_channel_bookmark_settings_locked(channel_id)
            active_memory = ""
            if channel_id is not None:
                current_memory = self.channel_routine_context.get(int(channel_id))
                if isinstance(current_memory, Mapping):
                    routine_text = str(current_memory.get("routine") or "").strip()
                    if routine_text:
                        active_memory = self._render_channel_memory_prompt(routine_text)
            has_channel_override = bool(
                channel_id is not None and isinstance(self.channel_prompt_overrides.get(int(channel_id)), Mapping)
            )
        prompt_layers = {
            "stream": {
                "editable_prompt": effective_stream_prompt,
                "backend_memory": active_memory,
                "warnings": list(prompt_health.get("warnings") or []),
                "notes": [
                    "Live L0 summaries use the editable stream prompt.",
                    "If channel memory exists, EVA AI appends it as prior context, not current visual evidence.",
                    "Current-batch observation instructions follow memory, and ALERTS_JSON instructions are appended last.",
                    "Bookmark settings only control Luxriot bookmark side effects.",
                ],
            },
            "alerts": {
                "editable_prompt": effective_alert_policy_prompt,
                "backend_instructions": self._render_alert_policy_prompt(effective_alert_policy_prompt),
                "warnings": list(prompt_health.get("warnings") or []),
                "notes": [
                    "Use this layer for channel-specific alert criteria and temporary watch items.",
                    "General safety/security hazards are always evaluated by EVA AI.",
                    "Do not put JSON schema text here; EVA AI appends the machine-readable ALERTS_JSON contract last.",
                    "Visible facts should be alerted; diagnoses, accusations, and hidden intent should not.",
                ],
            },
            "json": {
                "editable_prompt": str(effective_bookmark.get("json_alert_prompt") or DEFAULT_ALERTS_JSON_PROMPT),
                "notes": [
                    "Advanced machine-readable output contract.",
                    "Use Alert Criteria for ordinary operator watch conditions.",
                    "This JSON layer is appended last and is parsed for VLM alert events.",
                ],
            },
            "rollups": {
                level: {
                    "editable_prompt": effective_rollup_prompts.get(level, ""),
                    "backend_instructions": self._rollup_backend_instruction_text(level),
                    "active_memory": active_memory,
                    "notes": [
                        "The editable prompt is the system prompt.",
                        "Backend instructions are always appended as the user task layer.",
                        "Alert Ledger must preserve alert counts even when a window is routine.",
                    ],
                }
                for level in ("L1", "L2", "L3")
            },
        }
        return {
            "channel_id": int(channel_id) if channel_id is not None else None,
            "stream_system_prompt": effective_stream_prompt,
            "alert_policy_prompt": effective_alert_policy_prompt,
            "rollup_prompts": effective_rollup_prompts,
            "capture_interval_sec": effective_capture_interval_sec,
            "bookmark_enabled": bool(effective_bookmark.get("bookmark_enabled")),
            "bookmark_cooldown_sec": float(effective_bookmark.get("bookmark_cooldown_sec") or 0.0),
            "json_alert_prompt": str(effective_bookmark.get("json_alert_prompt") or DEFAULT_ALERTS_JSON_PROMPT),
            "defaults": defaults,
            "has_channel_override": has_channel_override,
            "prompt_layers": prompt_layers,
            "prompt_health": prompt_health,
        }

    def update_prompt_settings(
        self,
        channel_id: Optional[int] = None,
        stream_system_prompt: Optional[str] = None,
        alert_policy_prompt: Optional[str] = None,
        rollup_prompts: Optional[Mapping[str, Any]] = None,
        json_alert_prompt: Optional[str] = None,
        bookmark_enabled: Optional[bool] = None,
        bookmark_cooldown_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        changed = False
        target_channel_id = int(channel_id) if channel_id is not None else None
        channel_overrides: Dict[str, Any] = {}
        with self.cache_lock:
            if target_channel_id is None:
                if stream_system_prompt is not None:
                    next_stream_prompt = str(stream_system_prompt)
                    if next_stream_prompt != str(self.system_prompt or ""):
                        self.system_prompt = next_stream_prompt
                        changed = True
                if alert_policy_prompt is not None:
                    next_alert_policy_prompt = str(alert_policy_prompt)
                    if next_alert_policy_prompt != str(self.alert_policy_prompt or ""):
                        self.alert_policy_prompt = next_alert_policy_prompt
                        changed = True
                if json_alert_prompt is not None:
                    next_json_prompt = self._normalize_json_alert_prompt(json_alert_prompt)
                    if next_json_prompt != str(self.default_json_alert_prompt):
                        self.default_json_alert_prompt = next_json_prompt
                        changed = True
                if bookmark_enabled is not None:
                    next_enabled = bool(bookmark_enabled)
                    if next_enabled != bool(self.default_bookmark_enabled):
                        self.default_bookmark_enabled = next_enabled
                        changed = True
                if bookmark_cooldown_sec is not None:
                    next_cooldown = max(0.0, float(bookmark_cooldown_sec))
                    if next_cooldown != float(self.default_bookmark_cooldown_sec):
                        self.default_bookmark_cooldown_sec = next_cooldown
                        changed = True
            else:
                current_overrides_raw = self.channel_prompt_overrides.get(target_channel_id)
                channel_overrides = dict(current_overrides_raw) if isinstance(current_overrides_raw, Mapping) else {}
                if stream_system_prompt is not None:
                    next_stream_prompt = str(stream_system_prompt)
                    if next_stream_prompt != str(channel_overrides.get("stream_system_prompt") or ""):
                        channel_overrides["stream_system_prompt"] = next_stream_prompt
                        changed = True
                if alert_policy_prompt is not None:
                    next_alert_policy_prompt = str(alert_policy_prompt)
                    if next_alert_policy_prompt != str(channel_overrides.get("alert_policy_prompt") or ""):
                        channel_overrides["alert_policy_prompt"] = next_alert_policy_prompt
                        changed = True
                if json_alert_prompt is not None:
                    next_json_prompt = self._normalize_json_alert_prompt(json_alert_prompt)
                    if next_json_prompt != str(channel_overrides.get("json_alert_prompt") or ""):
                        channel_overrides["json_alert_prompt"] = next_json_prompt
                        changed = True
                if bookmark_enabled is not None:
                    next_enabled = bool(bookmark_enabled)
                    if next_enabled != bool(channel_overrides.get("bookmark_enabled", False)):
                        channel_overrides["bookmark_enabled"] = next_enabled
                        changed = True
                if bookmark_cooldown_sec is not None:
                    next_cooldown = max(0.0, float(bookmark_cooldown_sec))
                    if next_cooldown != float(channel_overrides.get("bookmark_cooldown_sec", 0.0) or 0.0):
                        channel_overrides["bookmark_cooldown_sec"] = next_cooldown
                        changed = True
            if isinstance(rollup_prompts, Mapping):
                for raw_level, raw_prompt in rollup_prompts.items():
                    level = self._normalize_rollup_level(raw_level)
                    if level not in {"L1", "L2", "L3"}:
                        continue
                    next_prompt = str(raw_prompt or "").strip()
                    if target_channel_id is None:
                        if next_prompt != str(self.rollup_llm_system_prompts.get(level) or ""):
                            self.rollup_llm_system_prompts[level] = next_prompt
                            changed = True
                    else:
                        existing_rollups_raw = channel_overrides.get("rollup_prompts")
                        channel_rollups = dict(existing_rollups_raw) if isinstance(existing_rollups_raw, Mapping) else {}
                        if next_prompt != str(channel_rollups.get(level) or ""):
                            channel_rollups[level] = next_prompt
                            channel_overrides["rollup_prompts"] = channel_rollups
                            changed = True
            if target_channel_id is not None and changed:
                self.channel_prompt_overrides[target_channel_id] = channel_overrides
            if changed:
                self._persist_summary_state_locked()
        return self.get_prompt_settings(channel_id=target_channel_id)

    def _get_rollup_system_prompt(self, level: str, channel_id: Optional[int] = None) -> str:
        with self.cache_lock:
            return self._get_rollup_system_prompt_locked(level, channel_id)

    @staticmethod
    def _bucket_start(ts: float, window_sec: int) -> int:
        window = max(1, int(window_sec))
        return int(ts // window) * window

    @staticmethod
    def _collect_highlights(nodes: Sequence[Mapping[str, Any]], max_items: int) -> List[str]:
        output: List[str] = []
        seen: Set[str] = set()
        for node in nodes:
            highlights = node.get("highlights")
            candidates: List[str] = []
            if isinstance(highlights, list):
                for item in highlights:
                    text = str(item or "").strip()
                    if text:
                        candidates.append(text)
            if not candidates:
                summary = str(node.get("summary") or "").strip()
                if summary:
                    candidates.append(LuxriotManager._summary_headline(summary))
            for candidate in candidates:
                if not candidate or candidate in seen:
                    continue
                seen.add(candidate)
                output.append(candidate)
                if len(output) >= max_items:
                    return output
        return output

    def _format_rollup_signal_text(self, alert_counts: object, signal_digest: object) -> str:
        parts: List[str] = []
        alert_text = self._format_alert_counts(alert_counts)
        if alert_text:
            parts.append(f"Alert counts: {alert_text}.")
        if isinstance(signal_digest, Mapping):
            deviations = self._coerce_memory_items(
                signal_digest.get("deviations"),
                max_items=5,
                max_len=180,
            )
            if deviations:
                parts.append("Preserved deviations: " + "; ".join(deviations) + ".")
        digest_text = self._render_signal_digest(signal_digest, max_len=700)
        if digest_text:
            digest_line = self._truncate_text(digest_text.replace("\n", " | "), 760)
            if digest_line and digest_line[-1] not in ".!?":
                digest_line += "."
            parts.append(f"Signal digest: {digest_line}")
        return " ".join(parts).strip()

    def _compose_rollup_summary(
        self,
        level: str,
        source_level: str,
        item_count: int,
        frame_count: int,
        run_ids: Sequence[str],
        highlights: Sequence[str],
        window_sec: int,
        alert_counts: object = None,
        signal_digest: object = None,
    ) -> str:
        base = (
            f"{level} rollup from {source_level}: {item_count} items over ~{max(1, int(window_sec // 60))} min"
            if window_sec < 3600
            else f"{level} rollup from {source_level}: {item_count} items over ~{max(1, int(window_sec // 3600))} hr"
        )
        if frame_count > 0:
            base += f" ({frame_count} frames)"
        if run_ids:
            base += f", {len(run_ids)} run(s)"
        if highlights:
            base += ". Highlights: " + "; ".join(highlights[: self.rollup_highlight_limit])
        else:
            base += "."
        signal_text = self._format_rollup_signal_text(alert_counts, signal_digest)
        if signal_text:
            base += " " + signal_text
        return base

    @classmethod
    def _rollup_child_salience_score(cls, child: Mapping[str, Any]) -> int:
        score = 0
        counts = cls._normalize_alert_counts(child.get("alert_counts"))
        score += counts.get("critical", 0) * 120
        score += counts.get("high", 0) * 100
        score += counts.get("normal", 0) * 70
        score += counts.get("low", 0) * 45
        score += counts.get("info", 0) * 30
        digest = child.get("signal_digest")
        if isinstance(digest, Mapping):
            for field, weight in (
                ("alert_events", 90),
                ("deviations", 80),
                ("missing_data", 55),
                ("uncertainty", 35),
                ("watchlist", 25),
                ("tuning", 20),
            ):
                items = cls._coerce_memory_items(digest.get(field), max_items=8, max_len=120)
                if items:
                    score += weight + min(20, len(items) * 5)
        summary = str(child.get("summary") or "")
        if re.search(r"\b(alert|deviation|incident|hazard|fire|smoke|fight|drift|weapon)\b", summary, flags=re.IGNORECASE):
            score += 20
        return int(score)

    @staticmethod
    def _estimate_token_count(text: object) -> int:
        normalized = " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())
        if not normalized:
            return 0
        return max(1, int(len(normalized) / 4))

    @staticmethod
    def _normalize_rollup_level(level: object) -> str:
        text = str(level or "").strip().upper()
        if text in {"L1", "L2", "L3"}:
            return text
        return ""

    @staticmethod
    def _coerce_str_list(values: object) -> List[str]:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            return []
        out: List[str] = []
        for value in values:
            text = str(value or "").strip()
            if text:
                out.append(text)
        return out

    @staticmethod
    def _infer_channel_id_from_rollup_id(rollup_id: str) -> Optional[int]:
        match = re.search(r"-ch(\d+)-", str(rollup_id or ""))
        if not match:
            return None
        try:
            return int(match.group(1))
        except Exception:
            return None

    @staticmethod
    def _canonical_rollup_id(level: str, channel_id: int, window_start: float, window_sec: int) -> str:
        normalized_level = str(level or "").strip().lower()
        start_bucket = int(float(window_start))
        normalized_window = max(0, int(window_sec))
        return f"{normalized_level}-ch{int(channel_id)}-w{normalized_window}-{start_bucket}"

    def _rollup_identity_key(self, row: Mapping[str, Any]) -> Optional[str]:
        level = self._normalize_rollup_level(row.get("level"))
        channel_id = _parse_optional_int(row.get("channel_id"))
        window_start = self._coerce_float(row.get("window_start"))
        window_end = self._coerce_float(row.get("window_end"))
        window_sec = _parse_optional_int(row.get("window_sec"))
        if (
            level in {"L1", "L2", "L3"}
            and channel_id is not None
            and window_start is not None
        ):
            if window_sec is None and window_end is not None:
                try:
                    window_sec = max(1, int(window_end - window_start))
                except Exception:
                    window_sec = 0
            return self._canonical_rollup_id(level, int(channel_id), float(window_start), int(window_sec or 0))
        rollup_id = str(row.get("rollup_id") or "").strip()
        if rollup_id:
            return rollup_id
        return None

    def _normalize_cached_rollup_entry(self, entry: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        rollup_id = str(entry.get("rollup_id") or "").strip()
        summary = str(entry.get("summary") or "").strip()
        if not summary:
            return None
        level = self._normalize_rollup_level(entry.get("level"))
        if not level:
            return None
        channel_id = _parse_optional_int(entry.get("channel_id"))
        if channel_id is None:
            channel_id = self._infer_channel_id_from_rollup_id(rollup_id)
        if channel_id is None:
            return None
        source_level = self._normalize_rollup_level(entry.get("source_level")) or None
        window_start = self._coerce_float(entry.get("window_start"))
        window_end = self._coerce_float(entry.get("window_end"))
        window_sec = _parse_optional_int(entry.get("window_sec"))
        if window_sec is None:
            if window_start is not None and window_end is not None:
                try:
                    window_sec = max(1, int(window_end - window_start))
                except Exception:
                    window_sec = 0
            else:
                window_sec = 0
        if level in {"L1", "L2", "L3"} and window_start is not None:
            rollup_id = self._canonical_rollup_id(level, int(channel_id), float(window_start), int(window_sec or 0))
        if not rollup_id:
            return None
        item_count = _parse_optional_int(entry.get("item_count")) or 0
        frame_count = _parse_optional_int(entry.get("frame_count")) or 0
        source_tokens = _parse_optional_int(entry.get("source_tokens")) or 0
        run_ids = self._coerce_str_list(entry.get("run_ids"))
        source_ids = self._coerce_str_list(entry.get("source_ids"))
        source_signature = str(entry.get("source_signature") or "").strip()
        if not source_signature:
            source_signature = self._source_signature(source_ids)
        highlights = self._coerce_str_list(entry.get("highlights"))
        summary_kind = str(entry.get("summary_kind") or "").strip() or "llm_cached"
        alert_meta = self._alert_meta_from_counts(entry.get("alert_counts"))
        if not alert_meta.get("alert_total"):
            raw_total = _parse_optional_int(entry.get("alert_total")) or 0
            if raw_total > 0:
                alert_meta = self._alert_meta_from_counts({"normal": raw_total})
        signal_digest = entry.get("signal_digest")
        if not isinstance(signal_digest, Mapping):
            signal_digest = {}
        alert_delivery_breakdown = self._compact_count_breakdown(entry.get("alert_delivery_breakdown"))
        alert_parser_breakdown = self._compact_count_breakdown(entry.get("alert_parser_breakdown"))
        state_transition_total = int(max(0, _parse_optional_int(entry.get("state_transition_total")) or 0))
        vector_signal_total = int(max(0, _parse_optional_int(entry.get("vector_signal_total")) or 0))
        created_at = self._coerce_float(entry.get("created_at"))
        if created_at is None:
            created_at = time.time()
        normalized = {
            "rollup_id": rollup_id,
            "channel_id": int(channel_id),
            "level": level,
            "source_level": source_level,
            "source_ids": source_ids,
            "window_start": window_start,
            "window_end": window_end,
            "window_sec": int(max(0, window_sec)),
            "item_count": int(max(0, item_count)),
            "frame_count": int(max(0, frame_count)),
            "source_tokens": int(max(0, source_tokens)),
            "run_ids": run_ids,
            "highlights": highlights,
            "source_signature": source_signature,
            "summary": summary,
            "summary_kind": summary_kind,
            "created_at": float(created_at),
            "signal_digest": dict(signal_digest),
            **alert_meta,
        }
        if alert_delivery_breakdown:
            normalized["alert_delivery_breakdown"] = alert_delivery_breakdown
        if alert_parser_breakdown:
            normalized["alert_parser_breakdown"] = alert_parser_breakdown
        if state_transition_total > 0:
            normalized["state_transition_total"] = state_transition_total
        if vector_signal_total > 0:
            normalized["vector_signal_total"] = vector_signal_total
        return normalized

    def _filter_rollup_cache_retention(
        self,
        entries: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        cutoff = self._summary_retention_cutoff()
        out: List[Dict[str, Any]] = []
        for entry in entries:
            normalized = self._normalize_cached_rollup_entry(entry)
            if normalized is None:
                continue
            created = self._coerce_float(normalized.get("created_at"))
            if cutoff is not None and created is not None and created < cutoff:
                continue
            out.append(normalized)
        return out

    def _prune_rollup_cache_retention_locked(self) -> None:
        cutoff = self._summary_retention_cutoff()
        if cutoff is None:
            return
        for key, entry in list(self.rollup_summary_cache.items()):
            if not isinstance(entry, Mapping):
                self.rollup_summary_cache.pop(key, None)
                continue
            created = self._coerce_float(entry.get("created_at"))
            if created is not None and created < cutoff:
                self.rollup_summary_cache.pop(key, None)

    def _persist_rollup_cache_locked(self) -> None:
        payload_entries = [
            dict(entry)
            for entry in self.rollup_summary_cache.values()
            if isinstance(entry, Mapping)
        ]
        cache_file = self.rollup_cache_file
        payload = {"version": 1, "updated_at": time.time(), "entries": payload_entries}
        state_store = getattr(self, "runtime_state_store", None)
        if state_store is not None:
            try:
                state_store.save_state("luxriot_rollup_cache", payload)
            except Exception:
                return
            return
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = cache_file.with_suffix(f"{cache_file.suffix}.tmp")
            tmp_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_file.replace(cache_file)
        except Exception:
            # Best-effort persistence should never interrupt the live stream loop.
            return

    def _load_rollup_cache_from_disk(self) -> None:
        payload: Optional[Any] = None
        state_store = getattr(self, "runtime_state_store", None)
        if state_store is not None:
            try:
                payload = state_store.load_state("luxriot_rollup_cache")
            except Exception:
                payload = None
        if payload is None:
            cache_file = self.rollup_cache_file
            if not cache_file.exists():
                return
            try:
                payload = json.loads(cache_file.read_text(encoding="utf-8"))
            except Exception:
                return
        raw_entries: Sequence[object]
        if isinstance(payload, Mapping):
            data_entries = payload.get("entries")
            if isinstance(data_entries, Sequence) and not isinstance(data_entries, (str, bytes, bytearray)):
                raw_entries = list(data_entries)
            else:
                raw_entries = [
                    value
                    for value in payload.values()
                    if isinstance(value, Mapping) and value.get("rollup_id")
                ]
        elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
            raw_entries = list(payload)
        else:
            raw_entries = []
        normalized_entries: List[Dict[str, Any]] = []
        for item in raw_entries:
            if not isinstance(item, Mapping):
                continue
            normalized = self._normalize_cached_rollup_entry(cast(Mapping[str, Any], item))
            if normalized is None:
                continue
            normalized_entries.append(normalized)
        normalized_entries.sort(key=lambda row: float(self._coerce_float(row.get("created_at")) or 0.0))
        with self.cache_lock:
            self.rollup_summary_cache.clear()
            retained_entries = self._filter_rollup_cache_retention(normalized_entries)
            for entry in retained_entries[-self.rollup_summary_cache_limit :]:
                self.rollup_summary_cache[str(entry["rollup_id"])] = entry

    def persist_rollup_cache(self) -> None:
        with self.cache_lock:
            self._persist_rollup_cache_locked()

    def _get_cached_rollup_record(self, rollup_id: str) -> Optional[Dict[str, Any]]:
        key = str(rollup_id or "").strip()
        if not key:
            return None
        with self.cache_lock:
            cached = self.rollup_summary_cache.get(key)
        if not isinstance(cached, Mapping):
            return None
        return dict(cached)

    def _put_cached_rollup_summary(self, rollup_id: str, summary: str, **meta: Any) -> None:
        key = str(rollup_id or "").strip()
        text = str(summary or "").strip()
        if not key or not text:
            return
        payload: Dict[str, Any] = {"rollup_id": key, "summary": text, "created_at": time.time()}
        payload.update(meta)
        normalized_payload = self._normalize_cached_rollup_entry(payload)
        if normalized_payload is None:
            return
        store_key = str(normalized_payload.get("rollup_id") or "").strip()
        if not store_key:
            return
        with self.cache_lock:
            self.rollup_summary_cache[store_key] = normalized_payload
            self._prune_rollup_cache_retention_locked()
            while len(self.rollup_summary_cache) > self.rollup_summary_cache_limit:
                oldest_key = next(iter(self.rollup_summary_cache))
                if oldest_key == store_key and len(self.rollup_summary_cache) == 1:
                    break
                self.rollup_summary_cache.pop(oldest_key, None)
            self._persist_rollup_cache_locked()

    def _list_cached_rollups(
        self,
        channel_id: int,
        start_ts: Optional[float],
        end_ts: Optional[float],
    ) -> List[Dict[str, Any]]:
        with self.cache_lock:
            entries = [dict(val) for val in self.rollup_summary_cache.values() if isinstance(val, Mapping)]
        out: List[Dict[str, Any]] = []
        for entry in entries:
            if _parse_optional_int(entry.get("channel_id")) != channel_id:
                continue
            window_start = self._coerce_float(entry.get("window_start"))
            window_end = self._coerce_float(entry.get("window_end"))
            if start_ts is not None and (window_end is None or window_end < start_ts):
                continue
            if end_ts is not None and (window_start is None or window_start > end_ts):
                continue
            out.append(entry)
        out.sort(key=lambda item: float(self._coerce_float(item.get("window_start")) or 0.0))
        return out

    @staticmethod
    def _rollup_matches_run_selector(entry: Mapping[str, Any], run_id: Optional[str]) -> bool:
        selected = str(run_id or "").strip()
        if not selected:
            return True
        run_ids = entry.get("run_ids")
        if not isinstance(run_ids, Sequence) or isinstance(run_ids, (str, bytes, bytearray)):
            return False
        return selected in {str(item or "").strip() for item in run_ids}

    def _merge_rollup_rows(
        self,
        generated_rows: Sequence[Mapping[str, Any]],
        stored_rows: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged_by_id: Dict[str, Dict[str, Any]] = {}
        anonymous_rows: List[Dict[str, Any]] = []
        for row in stored_rows:
            row_dict = dict(row)
            row_id = self._rollup_identity_key(row_dict)
            if row_id:
                merged_by_id[row_id] = row_dict
            else:
                anonymous_rows.append(row_dict)
        for row in generated_rows:
            row_dict = dict(row)
            row_id = self._rollup_identity_key(row_dict)
            if row_id:
                current = merged_by_id.get(row_id)
                if current:
                    merged = dict(current)
                    merged.update(row_dict)
                    generated_kind = str(row_dict.get("summary_kind") or "").strip().lower()
                    current_kind = str(current.get("summary_kind") or "").strip().lower()
                    generated_signature = str(row_dict.get("source_signature") or "").strip()
                    current_signature = str(current.get("source_signature") or "").strip()
                    if (
                        generated_kind == "pending_context"
                        and current_kind in {"llm", "llm_cached"}
                        and generated_signature
                        and generated_signature == current_signature
                    ):
                        merged["summary"] = str(current.get("summary") or "")
                        merged["summary_kind"] = current_kind
                    merged["rollup_id"] = row_id
                    merged_by_id[row_id] = merged
                else:
                    row_dict["rollup_id"] = row_id
                    merged_by_id[row_id] = row_dict
            else:
                anonymous_rows.append(row_dict)
        merged = list(merged_by_id.values()) + anonymous_rows
        merged.sort(
            key=lambda item: (
                float(self._coerce_float(item.get("window_start")) or 0.0),
                float(self._coerce_float(item.get("created_at")) or 0.0),
            )
        )
        return merged

    def _refresh_channel_routine_from_l2(self, channel_id: int, l2_rows: Sequence[Mapping[str, Any]]) -> None:
        if not l2_rows:
            return
        latest = max(
            l2_rows,
            key=lambda row: float(self._coerce_float(row.get("window_end")) or 0.0),
        )
        summary_kind = str(latest.get("summary_kind") or "").strip().lower()
        if summary_kind not in {"llm", "llm_cached"}:
            return
        rollup_id = str(latest.get("rollup_id") or "").strip()
        summary = str(latest.get("summary") or "").strip()
        if not rollup_id or not summary:
            return
        self._update_channel_routine_context(
            channel_id=channel_id,
            rollup_id=rollup_id,
            summary_text=summary,
            window_end=self._coerce_float(latest.get("window_end")),
            level="L2",
        )

    def _refresh_channel_memory_from_rollups(
        self,
        channel_id: int,
        rollup_rows: Sequence[Mapping[str, Any]],
    ) -> None:
        candidates: List[Mapping[str, Any]] = []
        for row in rollup_rows:
            if not isinstance(row, Mapping):
                continue
            summary_kind = str(row.get("summary_kind") or "").strip().lower()
            if summary_kind not in {"llm", "llm_cached"}:
                continue
            level = self._normalize_rollup_level(row.get("level"))
            if level not in {"L1", "L2", "L3"}:
                continue
            summary = str(row.get("summary") or "").strip()
            rollup_id = str(row.get("rollup_id") or "").strip()
            if not summary or not rollup_id:
                continue
            candidates.append(row)
        if not candidates:
            return
        candidates = sorted(
            candidates,
            key=lambda row: (
                float(self._coerce_float(row.get("window_end")) or 0.0),
                {"L1": 1, "L2": 2, "L3": 3}.get(self._normalize_rollup_level(row.get("level")), 0),
            ),
        )
        for row in candidates:
            self._update_channel_routine_context(
                channel_id=channel_id,
                rollup_id=str(row.get("rollup_id") or ""),
                summary_text=row.get("summary"),
                window_end=self._coerce_float(row.get("window_end")),
                level=self._normalize_rollup_level(row.get("level")),
            )

    def _select_rollup_source_lines(
        self,
        children: Sequence[Mapping[str, Any]],
        char_budget: int,
    ) -> List[str]:
        items: List[Tuple[float, str, int]] = []
        for child in sorted(children, key=lambda item: float(self._coerce_float(item.get("window_start")) or 0.0)):
            ts = self._coerce_float(child.get("window_start"))
            if ts is None:
                continue
            summary = self._sanitize_l0_summary(child.get("summary"), max_len=420)
            if not summary:
                continue
            ts_label = time.strftime("%H:%M:%S", time.localtime(ts))
            alert_counts = self._format_alert_counts(child.get("alert_counts"))
            if not alert_counts:
                raw_total = _parse_optional_int(child.get("alert_total")) or 0
                if raw_total > 0:
                    alert_counts = self._format_alert_counts({"normal": raw_total})
            if alert_counts:
                summary = f"[SOURCE_ALERTS {alert_counts}] {summary}"
            items.append((ts, f"- {ts_label} | {summary}", self._rollup_child_salience_score(child)))
        if not items:
            return ["- No valid lower-level summaries in this window."]
        if len(items) == 1:
            return [items[0][1]]
        # First pass: keep everything if it fits.
        joined_len = sum(len(line) + 1 for _, line, _score in items)
        if joined_len <= char_budget:
            return [line for _, line, _score in items]
        # Second pass: salience first, then even timeline sampling + first/last anchors.
        max_lines = max(4, min(len(items), int(char_budget / 260)))
        if max_lines >= len(items):
            selected_indexes = list(range(len(items)))
        else:
            selected_indexes = {0, len(items) - 1}
            salient_indexes = [
                index
                for index, (_ts, _line, score) in sorted(
                    enumerate(items),
                    key=lambda item: (-item[1][2], item[1][0]),
                )
                if score > 0
            ]
            salient_budget = max(1, min(len(salient_indexes), max_lines // 2))
            for index in salient_indexes[:salient_budget]:
                selected_indexes.add(index)
                if len(selected_indexes) >= max_lines:
                    break
            span = len(items) - 1
            remaining_slots = max(0, max_lines - len(selected_indexes))
            for step in range(1, remaining_slots + 1):
                idx = int(round((step * span) / max(1, max_lines - 1)))
                selected_indexes.add(max(0, min(len(items) - 1, idx)))
                if len(selected_indexes) >= max_lines:
                    break
            selected_indexes = sorted(selected_indexes)
        lines = [items[idx][1] for idx in cast(Sequence[int], selected_indexes)]
        # Final pass: trim trailing lines to budget.
        out: List[str] = []
        consumed = 0
        for line in lines:
            next_len = consumed + len(line) + 1
            if next_len > char_budget and out:
                break
            out.append(line)
            consumed = next_len
        return out or [lines[0]]

    def _build_rollup_messages(
        self,
        channel_id: int,
        level: str,
        source_level: str,
        node: Mapping[str, Any],
        children: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        system_msg = self._get_rollup_system_prompt(level, channel_id=channel_id) or (
            "You summarize CCTV batches into concise operator-facing rollups."
        )
        window_start = float(self._coerce_float(node.get("window_start")) or 0.0)
        window_end = float(self._coerce_float(node.get("window_end")) or 0.0)
        frame_count = _parse_optional_int(node.get("frame_count")) or 0
        item_count = _parse_optional_int(node.get("item_count")) or 0
        run_ids_raw = node.get("run_ids")
        run_ids = run_ids_raw if isinstance(run_ids_raw, list) else []
        run_text = ", ".join(sorted({str(run).strip() for run in run_ids if str(run).strip()})) or "n/a"
        source_tokens = _parse_optional_int(node.get("source_tokens")) or 0
        lines = self._select_rollup_source_lines(children, self.rollup_llm_char_budget)
        routine_context = self._get_channel_routine_prompt(channel_id)
        window_alert_counts = self._format_alert_counts(node.get("alert_counts"))
        window_signal_digest = self._render_signal_digest(node.get("signal_digest"), max_len=1000)
        backend_instruction_lines = self._rollup_backend_instruction_lines(level)
        user_text = "\n".join(
            [
                f"Channel: {channel_id}",
                f"Target level: {level}",
                f"Source level: {source_level}",
                f"Window: {self._window_label(window_start, window_end)}",
                f"Item count: {int(item_count)}",
                f"Frame count: {int(frame_count)}",
                f"Runs: {run_text}",
                f"Approx source tokens: {int(source_tokens)}",
                f"Source alert totals: {window_alert_counts or 'none'}",
                "",
                "Window signal digest (compact continuity map):",
                window_signal_digest or "none",
                "",
                *backend_instruction_lines,
                "",
                "Window Snapshot must begin with:",
                f"`Channel {channel_id} — {time.strftime('%H:%M', time.localtime(window_start))}-{time.strftime('%H:%M', time.localtime(window_end))}, {int(frame_count)} frames, {int(item_count)} items.`",
                "",
                "Known long-window routine context (if available):",
                routine_context or "n/a",
                "",
                f"{source_level} summaries:",
                *lines,
            ]
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_msg}]},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ]
        input_stats = self._estimate_message_payload_chars(cast(Sequence[Mapping[str, Any]], messages))
        input_stats.update(
            {
                "phase": "rollup_request_built",
                "level": level,
                "source_level": source_level,
                "source_lines_selected": len(lines),
                "source_lines_available": len(children),
                "source_chars_selected": int(sum(len(line) + 1 for line in lines)),
                "source_char_budget": int(self.rollup_llm_char_budget),
                "system_prompt_chars": len(system_msg),
                "backend_instruction_chars": sum(len(line) + 1 for line in backend_instruction_lines),
                "routine_context_chars": len(routine_context),
                "signal_digest_chars": len(window_signal_digest),
                "warning_text_chars": self.lm_input_warning_chars,
            }
        )
        warnings = self._summary_input_warnings(input_stats)
        if warnings:
            input_stats["warnings"] = warnings
        if isinstance(node, dict):
            node["llm_input_stats"] = self._compact_llm_input_stats(input_stats)
        return messages

    def _synthesize_rollup_summary(
        self,
        channel_id: int,
        level: str,
        source_level: str,
        node: Mapping[str, Any],
        children: Sequence[Mapping[str, Any]],
        fallback_summary: str,
    ) -> str:
        try:
            messages = self._build_rollup_messages(
                channel_id=channel_id,
                level=level,
                source_level=source_level,
                node=node,
                children=children,
            )
            with self.cache_lock:
                model_hint = self._get_rollup_model_hint_locked(channel_id)
            summary = str(self.lm_callback(messages, model_hint)).strip()
            return summary or fallback_summary
        except Exception:
            return fallback_summary

    def _compose_pending_rollup_summary(
        self,
        level: str,
        source_level: str,
        source_tokens: int,
        min_tokens: int,
        item_count: int,
        frame_count: int,
        alert_counts: object = None,
        signal_digest: object = None,
    ) -> str:
        base = (
            f"{level} pending from {source_level}: {item_count} items ({frame_count} frames). "
            f"Collecting context {source_tokens}/{min_tokens} tokens."
        )
        signal_text = self._format_rollup_signal_text(alert_counts, signal_digest)
        if signal_text:
            base += " " + signal_text
        return base

    def _apply_rollup_llm_summaries(
        self,
        channel_id: int,
        level: str,
        source_level: str,
        node_children_pairs: Sequence[Tuple[Dict[str, Any], Sequence[Mapping[str, Any]]]],
    ) -> None:
        if level not in self.rollup_llm_levels or not node_children_pairs:
            return
        remaining_budget = self.rollup_llm_max_new_per_call
        pairs = sorted(
            node_children_pairs,
            key=lambda pair: float(self._coerce_float(pair[0].get("window_start")) or 0.0),
            reverse=True,
        )
        for node, children in pairs:
            rollup_id = str(node.get("rollup_id") or "").strip()
            if not rollup_id:
                continue
            source_tokens = _parse_optional_int(node.get("source_tokens")) or 0
            source_signature = str(node.get("source_signature") or "").strip()
            if not source_signature:
                source_signature = self._rollup_source_signature(
                    cast(Sequence[Mapping[str, Any]], children),
                    self._coerce_str_list(node.get("source_ids")),
                )
                if source_signature:
                    node["source_signature"] = source_signature
            if (not self.rollup_time_only) and source_tokens < self.rollup_min_source_tokens:
                node["summary"] = self._compose_pending_rollup_summary(
                    level=level,
                    source_level=source_level,
                    source_tokens=source_tokens,
                    min_tokens=self.rollup_min_source_tokens,
                    item_count=_parse_optional_int(node.get("item_count")) or 0,
                    frame_count=_parse_optional_int(node.get("frame_count")) or 0,
                    alert_counts=node.get("alert_counts"),
                    signal_digest=node.get("signal_digest"),
                )
                node["summary_kind"] = "pending_context"
                continue
            cached = self._get_cached_rollup_record(rollup_id)
            if cached:
                cached_summary = str(cached.get("summary") or "").strip()
                cached_signature = str(cached.get("source_signature") or "").strip()
                if cached_summary and cached_signature and cached_signature == source_signature:
                    node["summary"] = cached_summary
                    node["summary_kind"] = "llm_cached"
                    self._update_channel_routine_context(
                        channel_id=channel_id,
                        rollup_id=rollup_id,
                        summary_text=node.get("summary"),
                        window_end=self._coerce_float(node.get("window_end")),
                        level=level,
                    )
                    continue
            if remaining_budget <= 0:
                continue
            fallback = str(node.get("summary") or "").strip()
            summary = self._synthesize_rollup_summary(
                channel_id=channel_id,
                level=level,
                source_level=source_level,
                node=node,
                children=children,
                fallback_summary=fallback,
            )
            if summary and summary != fallback:
                self._put_cached_rollup_summary(
                    rollup_id,
                    summary,
                    channel_id=channel_id,
                    level=level,
                    source_level=source_level,
                    window_start=self._coerce_float(node.get("window_start")),
                    window_end=self._coerce_float(node.get("window_end")),
                    window_sec=_parse_optional_int(node.get("window_sec")) or 0,
                    item_count=_parse_optional_int(node.get("item_count")) or 0,
                    frame_count=_parse_optional_int(node.get("frame_count")) or 0,
                    source_tokens=source_tokens,
                    run_ids=node.get("run_ids"),
                    source_ids=node.get("source_ids"),
                    source_signature=source_signature,
                    highlights=node.get("highlights"),
                    alert_counts=node.get("alert_counts"),
                    alert_total=node.get("alert_total"),
                    alert_severities=node.get("alert_severities"),
                    signal_digest=node.get("signal_digest"),
                    alert_delivery_breakdown=node.get("alert_delivery_breakdown"),
                    alert_parser_breakdown=node.get("alert_parser_breakdown"),
                    state_transition_total=node.get("state_transition_total"),
                    summary_kind="llm",
                )
                node["summary"] = summary
                node["summary_kind"] = "llm"
                self._update_channel_routine_context(
                    channel_id=channel_id,
                    rollup_id=rollup_id,
                    summary_text=summary,
                    window_end=self._coerce_float(node.get("window_end")),
                    level=level,
                )
            remaining_budget -= 1

    def _l0_nodes_from_logs(self, channel_id: int, logs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        nodes: List[Dict[str, Any]] = []
        for log in logs:
            if not isinstance(log, Mapping):
                continue
            created = self._coerce_float(log.get("created_at"))
            if created is None:
                continue
            created_ms = int(float(created) * 1000.0)
            batch_start_ms = _parse_optional_int(log.get("batch_start_ms"))
            if batch_start_ms is None:
                batch_start_ms = created_ms
            batch_end_ms = _parse_optional_int(log.get("batch_end_ms"))
            if batch_end_ms is None:
                batch_end_ms = batch_start_ms
            if batch_end_ms < batch_start_ms:
                batch_start_ms, batch_end_ms = batch_end_ms, batch_start_ms
            window_start = float(batch_start_ms) / 1000.0
            window_end = float(batch_end_ms) / 1000.0
            frame_count = _parse_optional_int(log.get("frame_count")) or 0
            run_id = str(log.get("run_id") or "").strip()
            summary = str(log.get("summary") or "").strip()
            headline = self._summary_headline(summary)
            key = self._summary_log_key(log)
            rollup_id = f"l0-ch{channel_id}-{self._stable_id(key, length=14)}"
            alert_counts = dict(log.get("alert_counts") or {})
            alert_total = int(_parse_optional_int(log.get("alert_total")) or 0)
            alert_events = self._compact_alert_events(log.get("alert_events"))
            state_observations = self._compact_state_observations(log.get("state_observations"))
            state_transition_events = self._compact_state_transition_events(log.get("state_transition_events"))
            vector_signal = self._compact_vector_signal(log.get("vector_signal"))
            state_transition_total = int(
                max(
                    0,
                    _parse_optional_int(log.get("state_transition_total")) or len(state_transition_events),
                )
            )
            alert_delivery_breakdown = self._alert_delivery_breakdown_from_entry(log)
            alert_parser_breakdown = self._alert_parser_breakdown_from_entry(log)
            signal_digest = self._summary_signal_digest(
                summary,
                channel_id=channel_id,
                timestamp_ms=int(batch_end_ms),
                alert_counts=alert_counts,
                alert_total=alert_total,
            )
            node = {
                "rollup_id": rollup_id,
                "channel_id": channel_id,
                "level": "L0",
                "source_level": None,
                "source_ids": [],
                "window_start": window_start,
                "window_end": window_end,
                "window_sec": max(0, int(round(window_end - window_start))),
                "item_count": 1,
                "frame_count": int(frame_count),
                "run_ids": [run_id] if run_id else [],
                "highlights": [headline] if headline else [],
                "summary": summary,
                "created_at": created,
                "alert_counts": alert_counts,
                "alert_total": alert_total,
                "alert_severities": self._coerce_str_list(log.get("alert_severities")),
                "signal_digest": signal_digest,
            }
            if alert_events:
                node["alert_events"] = alert_events
            if state_observations:
                node["state_observations"] = state_observations
            if state_transition_events:
                node["state_transition_events"] = state_transition_events
            if state_transition_total > 0:
                node["state_transition_total"] = state_transition_total
            if alert_delivery_breakdown:
                node["alert_delivery_breakdown"] = alert_delivery_breakdown
            if alert_parser_breakdown:
                node["alert_parser_breakdown"] = alert_parser_breakdown
            if vector_signal:
                node["vector_signal"] = vector_signal
                clip_count = len(vector_signal.get("clip_probe_signals") or []) if isinstance(vector_signal.get("clip_probe_signals"), list) else 0
                road_count = len(vector_signal.get("road_cv_cues") or []) if isinstance(vector_signal.get("road_cv_cues"), list) else 0
                node["vector_signal_total"] = int(clip_count + road_count)
            nodes.append(node)
        nodes.sort(key=lambda item: float(item.get("window_start") or 0.0))
        return nodes

    def _build_rollup_level(
        self,
        channel_id: int,
        level: str,
        source_level: str,
        window_sec: int,
        source_nodes: Sequence[Mapping[str, Any]],
        *,
        synthesize: bool = True,
    ) -> List[Dict[str, Any]]:
        if not source_nodes:
            return []
        buckets: Dict[int, List[Mapping[str, Any]]] = {}
        for node in source_nodes:
            ts = self._coerce_float(node.get("window_start"))
            if ts is None:
                continue
            bucket = self._bucket_start(ts, window_sec)
            buckets.setdefault(bucket, []).append(node)
        out: List[Dict[str, Any]] = []
        llm_pairs: List[Tuple[Dict[str, Any], Sequence[Mapping[str, Any]]]] = []
        for bucket_start in sorted(buckets.keys()):
            children = sorted(
                buckets[bucket_start],
                key=lambda item: float(self._coerce_float(item.get("window_start")) or 0.0),
            )
            source_ids: List[str] = []
            frame_count = 0
            item_count = 0
            source_tokens = 0
            run_ids: Set[str] = set()
            for child in children:
                child_id = str(child.get("rollup_id") or "").strip()
                if child_id:
                    source_ids.append(child_id)
                frame_count += _parse_optional_int(child.get("frame_count")) or 0
                item_count += _parse_optional_int(child.get("item_count")) or 0
                source_tokens += self._estimate_token_count(child.get("summary"))
                child_runs = child.get("run_ids")
                if isinstance(child_runs, list):
                    for run in child_runs:
                        run_text = str(run or "").strip()
                        if run_text:
                            run_ids.add(run_text)
            highlights = self._collect_highlights(children, self.rollup_highlight_limit)
            alert_meta = self._merge_alert_metadata(children)
            signal_digest = self._aggregate_signal_digest(
                children,
                alert_counts=cast(Mapping[str, Any], alert_meta.get("alert_counts") or {}),
            )
            provenance_meta = self._aggregate_provenance_metadata(children)
            source_signature = self._rollup_source_signature(children, source_ids)
            summary = self._compose_rollup_summary(
                level=level,
                source_level=source_level,
                item_count=item_count,
                frame_count=frame_count,
                run_ids=sorted(run_ids),
                highlights=highlights,
                window_sec=window_sec,
                alert_counts=alert_meta.get("alert_counts"),
                signal_digest=signal_digest,
            )
            end_ts = float(bucket_start + max(1, int(window_sec)))
            rollup_id = self._canonical_rollup_id(level, channel_id, float(bucket_start), int(window_sec))
            out.append(
                {
                    "rollup_id": rollup_id,
                    "channel_id": channel_id,
                    "level": level,
                    "source_level": source_level,
                    "source_ids": source_ids,
                    "window_start": float(bucket_start),
                    "window_end": end_ts,
                    "window_sec": int(window_sec),
                    "item_count": int(item_count),
                    "frame_count": int(frame_count),
                    "source_tokens": int(source_tokens),
                    "run_ids": sorted(run_ids),
                    "highlights": highlights,
                    "source_signature": source_signature,
                    "summary": summary,
                    "created_at": end_ts,
                    "signal_digest": signal_digest,
                    **alert_meta,
                    **provenance_meta,
                }
            )
            if synthesize and level in self.rollup_llm_levels:
                llm_pairs.append((out[-1], children))
        if synthesize and level in self.rollup_llm_levels and llm_pairs:
            self._apply_rollup_llm_summaries(
                channel_id=channel_id,
                level=level,
                source_level=source_level,
                node_children_pairs=llm_pairs,
            )
        return out

    def _merge_summary_history_locked(self, channel_id: int, logs: Sequence[Mapping[str, Any]]) -> None:
        if not logs:
            return
        existing = self._filter_normalized_summary_history_retention(
            self.summary_history.get(channel_id, [])
        )
        merged: List[Dict[str, Any]] = []
        key_to_index: Dict[Tuple[str, str, str, str], int] = {}
        last_created: Optional[float] = None
        out_of_order = False

        for item in existing:
            key = self._summary_log_key(item)
            if key in key_to_index:
                continue
            key_to_index[key] = len(merged)
            merged.append(item)
            created = self._coerce_float(item.get("created_at"))
            if created is not None:
                if last_created is not None and created < last_created:
                    out_of_order = True
                last_created = created

        for raw_log in logs:
            if not isinstance(raw_log, Mapping):
                continue
            incoming = dict(raw_log)
            key = self._summary_log_key(incoming)
            index = key_to_index.get(key)
            if index is not None:
                existing_item = merged[index]
                existing_meta = self._alert_meta_from_counts(existing_item.get("alert_counts"))
                incoming_meta = self._alert_meta_from_counts(incoming.get("alert_counts"))
                existing_total = int(existing_meta.get("alert_total") or 0)
                incoming_total = int(incoming_meta.get("alert_total") or 0)
                if existing_total > 0 and incoming_total <= 0:
                    incoming["alert_counts"] = dict(existing_meta.get("alert_counts") or {})
                    incoming["alert_total"] = existing_total
                    incoming["alert_severities"] = list(existing_meta.get("alert_severities") or [])
                if (
                    isinstance(existing_item.get("signal_digest"), Mapping)
                    and not isinstance(incoming.get("signal_digest"), Mapping)
                ):
                    incoming["signal_digest"] = dict(cast(Mapping[str, Any], existing_item.get("signal_digest")))
                self._preserve_summary_provenance_on_merge(existing_item, incoming)
                merged[index] = incoming
                continue
            created = self._coerce_float(incoming.get("created_at"))
            if created is not None and last_created is not None and created < last_created:
                out_of_order = True
            if created is not None:
                last_created = created
            key_to_index[key] = len(merged)
            merged.append(incoming)

        if out_of_order:
            merged.sort(key=lambda item: float(self._coerce_float(item.get("created_at")) or 0.0))
        if len(merged) > self.summary_history_limit:
            merged = merged[-self.summary_history_limit :]
        self.summary_history[channel_id] = merged
        self._update_channel_status_digest_locked(channel_id, merged)
        self._persist_summary_state_if_due_locked()

    def record_summary_log(self, channel_id: int, entry: Mapping[str, Any]) -> None:
        normalized = self._normalize_summary_log_entry(entry)
        if normalized is None:
            return
        with self.cache_lock:
            self._merge_summary_history_locked(channel_id, [normalized])

    def set_summary_dispatcher(
        self,
        dispatcher: Optional[SummaryDispatcherFn],
    ) -> None:
        self.summary_dispatcher = dispatcher

    def set_summary_archive_callback(
        self,
        callback: Optional[SummaryArchiveFn],
    ) -> None:
        self.summary_archive_callback = callback

    @classmethod
    def _frame_timestamp_ms(cls, frame: Mapping[str, Any], fallback_ms: int) -> int:
        raw_ms = _parse_optional_int(frame.get("timestamp_ms"))
        if raw_ms is not None and raw_ms >= 0:
            return int(raw_ms)
        raw_value = frame.get("captured_at")
        if not isinstance(raw_value, (int, float)):
            raw_value = frame.get("time_sec")
        if isinstance(raw_value, (int, float)):
            try:
                numeric = float(raw_value)
                if numeric > 10_000_000_000:
                    return int(numeric)
                if numeric >= 0:
                    return int(numeric * 1000.0)
            except Exception:
                pass
        return int(max(0, fallback_ms))

    @classmethod
    def _summary_archive_frames(
        cls,
        frames: Sequence[Mapping[str, Any]],
        *,
        batch_start_ms: int,
        batch_end_ms: int,
        sample_count: int = 4,
    ) -> List[Dict[str, Any]]:
        if not frames:
            return []
        last_index = len(frames) - 1
        try:
            sample_count = int(sample_count)
        except Exception:
            sample_count = 4
        sample_count = max(1, min(16, sample_count))
        if last_index == 0:
            anchors: List[Tuple[str, int]] = [("only", 0)]
        elif sample_count <= 2:
            anchors = [("first", 0), ("last", last_index)]
        else:
            raw_indices = {
                int(round(last_index * (index / float(sample_count - 1))))
                for index in range(sample_count)
            }
            indices = sorted(index for index in raw_indices if 0 <= index <= last_index)
            anchors = []
            for index in indices:
                if index == 0:
                    role = "first"
                elif index == last_index:
                    role = "last"
                else:
                    role = "sample"
                anchors.append((role, index))
        out: List[Dict[str, Any]] = []
        for role, index in anchors:
            frame = frames[index]
            thumbnail = str(frame.get("thumbnail") or "").strip()
            if not thumbnail:
                continue
            fallback_ms = batch_start_ms if role in {"first", "only"} else batch_end_ms
            captured_at = cls._coerce_float(frame.get("captured_at"))
            if captured_at is None:
                captured_at = cls._coerce_float(frame.get("time_sec"))
            item: Dict[str, Any] = {
                "anchor_role": role,
                "frame_index": int(index),
                "timestamp_ms": cls._frame_timestamp_ms(frame, fallback_ms),
                "thumbnail": thumbnail,
            }
            if captured_at is not None:
                item["captured_at"] = float(captured_at)
            for key in ("width", "height"):
                value = _parse_optional_int(frame.get(key))
                if value is not None and value > 0:
                    item[key] = int(value)
            out.append(item)
        return out

    def _archive_summary_entry(self, entry: Mapping[str, Any]) -> Dict[str, Any]:
        callback = self.summary_archive_callback
        if callback is None:
            return {}
        try:
            result = callback(entry)
        except Exception as exc:
            return {"error": str(exc)[:240] or exc.__class__.__name__}
        if isinstance(result, Mapping):
            return dict(result)
        return {}

    @staticmethod
    def _estimate_message_payload_chars(messages: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        text_chars = 0
        image_parts = 0
        high_detail_images = 0
        image_url_chars = 0
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            content = message.get("content")
            if isinstance(content, str):
                text_chars += len(content)
                continue
            if not isinstance(content, Sequence) or isinstance(content, (str, bytes, bytearray)):
                continue
            for part in content:
                if not isinstance(part, Mapping):
                    continue
                part_type = str(part.get("type") or "")
                if part_type == "text":
                    text_chars += len(str(part.get("text") or ""))
                    continue
                if part_type == "image_url":
                    image_parts += 1
                    image_url = part.get("image_url")
                    if isinstance(image_url, Mapping):
                        if str(image_url.get("detail") or "").lower() == "high":
                            high_detail_images += 1
                        image_url_chars += len(str(image_url.get("url") or ""))
        try:
            total_payload_chars = len(
                json.dumps(
                    list(messages),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
            )
        except Exception:
            total_payload_chars = text_chars + image_url_chars
        return {
            "message_count": len(messages),
            "text_chars": int(text_chars),
            "image_parts": int(image_parts),
            "high_detail_images": int(high_detail_images),
            "image_url_chars": int(image_url_chars),
            "total_payload_chars": int(total_payload_chars),
        }

    def _summary_input_warnings(self, stats: Mapping[str, Any]) -> List[str]:
        warnings: List[str] = []
        text_chars = _parse_optional_int(stats.get("text_chars")) or 0
        image_url_chars = _parse_optional_int(stats.get("image_url_chars")) or 0
        if text_chars >= self.lm_input_warning_chars:
            warnings.append(
                f"text_input_chars {text_chars} >= warning {self.lm_input_warning_chars}"
            )
        if image_url_chars >= self.lm_image_payload_warning_chars:
            warnings.append(
                f"image_payload_chars {image_url_chars} >= warning {self.lm_image_payload_warning_chars}"
            )
        return warnings

    def create_summary_batch(
        self,
        *,
        channel_id: int,
        run_id: str,
        batch_size: int,
        prompt: str,
        model_hint: Optional[str],
        interval_sec: float,
        frames: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        frame_items = [dict(frame) for frame in frames if isinstance(frame, Mapping)]
        if not frame_items:
            raise ValueError("summary batch requires at least one frame")
        frame_ts_ms: List[int] = []
        for frame in frame_items:
            raw_ts = frame.get("captured_at") or frame.get("time_sec")
            if not isinstance(raw_ts, (int, float)):
                continue
            try:
                frame_ts_ms.append(int(float(raw_ts) * 1000.0))
            except Exception:
                continue
        submitted_at = time.time()
        submitted_at_ms = int(submitted_at * 1000.0)
        batch_start_ms = min(frame_ts_ms) if frame_ts_ms else submitted_at_ms
        batch_end_ms = max(frame_ts_ms) if frame_ts_ms else submitted_at_ms
        vector_signal = self._build_vector_signal_bundle(
            int(channel_id),
            cast(Sequence[Mapping[str, Any]], frame_items),
            batch_start_ms=batch_start_ms,
            batch_end_ms=batch_end_ms,
        )
        base_system_prompt = self.get_effective_stream_system_prompt(channel_id)
        system_prompt = self.compose_live_system_prompt(channel_id, base_system_prompt, vector_signal=vector_signal)
        frame_b64_lengths = [
            len(str(frame.get("thumbnail") or ""))
            for frame in frame_items
            if isinstance(frame, Mapping)
        ]
        llm_input_stats = {
            "phase": "summary_batch_created",
            "frame_count": len(frame_items),
            "batch_size": int(batch_size),
            "system_prompt_chars": len(system_prompt),
            "task_prompt_chars": len(str(prompt or "")),
            "vector_signal_chars": len(json.dumps(vector_signal, ensure_ascii=False, sort_keys=True)) if vector_signal else 0,
            "total_image_base64_chars": int(sum(frame_b64_lengths)),
            "largest_frame_base64_chars": int(max(frame_b64_lengths) if frame_b64_lengths else 0),
            "warning_text_chars": self.lm_input_warning_chars,
            "warning_image_payload_chars": self.lm_image_payload_warning_chars,
        }
        return {
            "version": 1,
            "channel_id": int(channel_id),
            "run_id": str(run_id or "").strip(),
            "batch_size": int(batch_size),
            "prompt": str(prompt or ""),
            "model_hint": str(model_hint or "").strip() or None,
            "system_prompt": system_prompt,
            "vector_signal": vector_signal,
            "interval_sec": max(0.2, float(interval_sec)),
            "frames": frame_items,
            "frame_count": len(frame_items),
            "batch_start_ms": batch_start_ms,
            "batch_end_ms": batch_end_ms,
            "submitted_at": submitted_at,
            "llm_input_stats": llm_input_stats,
        }

    def run_summary_batch(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        channel_id = _parse_optional_int(batch.get("channel_id"))
        frames = batch.get("frames")
        if channel_id is None or channel_id < 1:
            raise ValueError("summary batch channel_id must be positive")
        if (
            not isinstance(frames, Sequence)
            or isinstance(frames, (str, bytes, bytearray))
            or not frames
        ):
            raise ValueError("summary batch frames are missing")
        frame_items = [
            dict(frame)
            for frame in frames
            if isinstance(frame, Mapping)
        ]
        if not frame_items:
            raise ValueError("summary batch has no valid frames")
        started = time.time()
        messages = self.message_builder(
            f"#{channel_id}",
            frame_items,
            str(batch.get("prompt") or ""),
            str(batch.get("system_prompt") or ""),
        )
        llm_input_stats = dict(batch.get("llm_input_stats") or {})
        message_stats = self._estimate_message_payload_chars(cast(Sequence[Mapping[str, Any]], messages))
        llm_input_stats.update(
            {
                "phase": "summary_request_built",
                "message_count": message_stats.get("message_count"),
                "text_chars": message_stats.get("text_chars"),
                "image_parts": message_stats.get("image_parts"),
                "high_detail_images": message_stats.get("high_detail_images"),
                "image_url_chars": message_stats.get("image_url_chars"),
                "total_payload_chars": message_stats.get("total_payload_chars"),
            }
        )
        warnings = self._summary_input_warnings(llm_input_stats)
        if warnings:
            llm_input_stats["warnings"] = warnings
        model_hint = str(batch.get("model_hint") or "").strip() or None
        summary = self.lm_callback(messages, model_hint)
        submitted_at = self._coerce_float(batch.get("submitted_at"))
        created_at = submitted_at if submitted_at is not None else time.time()
        batch_start_ms = _parse_optional_int(batch.get("batch_start_ms"))
        if batch_start_ms is None:
            batch_start_ms = int(created_at * 1000.0)
        batch_end_ms = _parse_optional_int(batch.get("batch_end_ms"))
        if batch_end_ms is None:
            batch_end_ms = int(created_at * 1000.0)
        archive_frames = self._summary_archive_frames(
            frame_items,
            batch_start_ms=batch_start_ms,
            batch_end_ms=batch_end_ms,
            sample_count=getattr(self, "summary_archive_frames_per_batch", 4),
        )
        entry = {
            "channel_id": int(channel_id),
            "run_id": str(batch.get("run_id") or "").strip(),
            "summary": summary,
            "frame_count": len(frame_items),
            "batch_size": max(
                len(frame_items),
                _parse_optional_int(batch.get("batch_size")) or 0,
            ),
            "created_at": float(created_at),
            "batch_start_ms": batch_start_ms,
            "batch_end_ms": batch_end_ms,
            "duration_sec": max(0.0, time.time() - started),
            "prompt": str(batch.get("prompt") or ""),
            "interval_sec": max(
                0.2,
                self._coerce_float(batch.get("interval_sec")) or 1.0,
            ),
            "llm_input_stats": llm_input_stats,
        }
        vector_signal = self._compact_vector_signal(batch.get("vector_signal"))
        if vector_signal:
            entry["vector_signal"] = vector_signal
        if archive_frames:
            entry["archive_frames"] = archive_frames
        return entry

    def accept_summary_entry(self, entry: Mapping[str, Any]) -> Dict[str, Any]:
        normalized = self._normalize_summary_log_entry(entry)
        if normalized is None:
            raise ValueError("invalid summary result")
        accepted = dict(entry)
        accepted.update(normalized)
        channel_id = int(normalized["channel_id"])
        batch_start_ms = _parse_optional_int(accepted.get("batch_start_ms"))
        if batch_start_ms is None:
            batch_start_ms = int(float(normalized["created_at"]) * 1000.0)
        batch_end_ms = _parse_optional_int(accepted.get("batch_end_ms"))
        if batch_end_ms is None:
            batch_end_ms = batch_start_ms
        interval_sec = max(
            0.2,
            self._coerce_float(accepted.get("interval_sec")) or 1.0,
        )
        tolerance_ms = max(1000, int(interval_sec * 1000.0))
        try:
            alert_delivery = self.process_summary_alerts(
                channel_id,
                str(normalized["summary"]),
                default_ts_ms=batch_end_ms,
                min_ts_ms=batch_start_ms - tolerance_ms,
                max_ts_ms=batch_end_ms + tolerance_ms,
            )
        except Exception as exc:
            alert_delivery = AlertDeliveryResult(
                0,
                failed=1,
                last_error=str(exc)[:240] or exc.__class__.__name__,
            )
        accepted.update(alert_delivery.as_dict())

        state_observations = self._extract_current_observed_states(str(normalized["summary"]))
        state_transitions = self._update_observed_state_tracker(
            channel_id,
            state_observations,
            int(batch_end_ms),
        )
        if state_observations:
            accepted["state_observations"] = state_observations
        if state_transitions:
            accepted["state_transition_events"] = state_transitions
            accepted["state_transition_total"] = len(state_transitions)
            if self.state_transition_alert_events_enabled:
                transition_alert_events = self._transition_alert_events(state_transitions, channel_id)
                if transition_alert_events:
                    existing_events = self._compact_alert_events(accepted.get("alert_events"))
                    accepted["alert_events"] = existing_events + transition_alert_events
                    counts = self._normalize_alert_counts(accepted.get("alert_counts"))
                    for event in transition_alert_events:
                        severity = self._normalize_alert_severity(event.get("severity"))
                        counts[severity] = counts.get(severity, 0) + 1
                    alert_meta = self._alert_meta_from_counts(counts)
                    accepted.update(alert_meta)

        archive_meta = self._archive_summary_entry(accepted)
        accepted.pop("archive_frames", None)
        if archive_meta:
            for key, value in archive_meta.items():
                if key in {"attempted", "inserted", "summary_frames", "alert_frames", "error"}:
                    accepted[f"archive_{key}"] = value

        with self.cache_lock:
            session = self.sessions.get(channel_id)
        if session is not None:
            run_id = str(accepted.get("run_id") or "").strip()
            if not run_id or run_id == session.run_id:
                with session.lock:
                    entry_key = self._summary_log_key(accepted)
                    if all(
                        self._summary_log_key(existing) != entry_key
                        for existing in session.logs
                    ):
                        session.logs.append(dict(accepted))
                        session.total_flushes += 1
                        session.last_error = None
                        if len(session.logs) > 50:
                            session.logs = session.logs[-50:]
        self.record_summary_log(channel_id, accepted)
        return accepted

    def dispatch_summary_batch(
        self,
        batch: Mapping[str, Any],
        *,
        workload_class: str = "heartbeat",
    ) -> Dict[str, Any]:
        dispatcher = self.summary_dispatcher
        if dispatcher is not None:
            return dict(dispatcher(batch, workload_class))
        entry = self.run_summary_batch(batch)
        self.accept_summary_entry(entry)
        return {
            "queued": False,
            "accepted": True,
            "status": "completed",
        }

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
        captured_at_ms = int(time.time() * 1000)
        encoded = self.jpeg_encoder(snapshot, max_edge=self.config.LUXRIOT_SNAPSHOT_MAX_EDGE, quality=85)
        return encoded, {
            "width": snapshot.width,
            "height": snapshot.height,
            "captured_at_ms": captured_at_ms,
            "sha1": hashlib.sha1(encoded.encode("ascii")).hexdigest(),
        }

    def capture_snapshot_base64(
        self,
        channel_id: int,
        stream_type: str = "mainStream",
        roi_norm: Optional[Tuple[float, float, float, float]] = None,
        quality: int = 92,
    ) -> Tuple[str, Dict[str, Any]]:
        client = self.build_client()
        snapshot = client.get_snapshot(channel_id, stream=stream_type)
        captured_at_ms = int(time.time() * 1000)
        original_width = int(snapshot.width)
        original_height = int(snapshot.height)
        crop_box: Optional[Tuple[int, int, int, int]] = None
        if roi_norm is not None:
            x, y, w, h = roi_norm
            sx = max(0, min(original_width - 1, int(round(float(x) * original_width))))
            sy = max(0, min(original_height - 1, int(round(float(y) * original_height))))
            sw = max(1, min(original_width - sx, int(round(float(w) * original_width))))
            sh = max(1, min(original_height - sy, int(round(float(h) * original_height))))
            crop_box = (sx, sy, sx + sw, sy + sh)
            snapshot = snapshot.crop(crop_box)
        encoded = self.jpeg_encoder(
            snapshot,
            max_edge=self.config.LUXRIOT_SNAPSHOT_MAX_EDGE,
            quality=max(60, min(95, int(quality))),
        )
        return encoded, {
            "channel_id": int(channel_id),
            "stream": stream_type,
            "width": int(snapshot.width),
            "height": int(snapshot.height),
            "original_width": original_width,
            "original_height": original_height,
            "captured_at_ms": captured_at_ms,
            "sha1": hashlib.sha1(encoded.encode("ascii")).hexdigest(),
            "roi": {
                "x": crop_box[0],
                "y": crop_box[1],
                "w": crop_box[2] - crop_box[0],
                "h": crop_box[3] - crop_box[1],
            }
            if crop_box is not None
            else None,
        }

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

    @staticmethod
    def _normalize_alert_timestamp_ms(raw_value: Any, fallback_ts_ms: int) -> int:
        try:
            ts_ms = int(raw_value)
        except Exception:
            ts_ms = int(fallback_ts_ms)
        return ts_ms if ts_ms > 0 else int(fallback_ts_ms)

    @staticmethod
    def _normalize_alert_timestamp_ms_bounded(
        raw_value: Any,
        fallback_ts_ms: int,
        min_ts_ms: Optional[int] = None,
        max_ts_ms: Optional[int] = None,
    ) -> int:
        fallback = int(fallback_ts_ms) if int(fallback_ts_ms) > 0 else int(time.time() * 1000)

        # Default timestamp from prompt examples; ignore and use observed batch time instead.
        DEFAULT_PROMPT_TS_MS = 1772202050000

        parsed_ts_ms: Optional[int] = None
        if raw_value is not None:
            try:
                numeric = float(raw_value)
                if numeric > 0:
                    # If model returns unix seconds (10 digits), convert to milliseconds.
                    if numeric < 1_000_000_000_000:
                        numeric *= 1000.0
                    parsed_ts_ms = int(numeric)
            except Exception:
                parsed_ts_ms = None

        min_bound: Optional[int] = None
        max_bound: Optional[int] = None
        if min_ts_ms is not None:
            try:
                min_bound = int(min_ts_ms)
            except Exception:
                min_bound = None
        if max_ts_ms is not None:
            try:
                max_bound = int(max_ts_ms)
            except Exception:
                max_bound = None

        ts_ms = parsed_ts_ms if parsed_ts_ms is not None else fallback
        if ts_ms == DEFAULT_PROMPT_TS_MS:
            # Treat the baked-in template literal as invalid unless it actually
            # falls inside the observed batch window.
            in_window = True
            if min_bound is not None and ts_ms < min_bound:
                in_window = False
            if max_bound is not None and ts_ms > max_bound:
                in_window = False
            if not in_window:
                ts_ms = fallback

        # Reject obviously implausible epochs for runtime events.
        if ts_ms < 946684800000 or ts_ms > 4102444800000:  # 2000-01-01 .. 2100-01-01
            ts_ms = fallback
        if min_bound is not None and max_bound is not None and min_bound > max_bound:
            min_bound, max_bound = max_bound, min_bound
        if min_bound is not None and ts_ms < min_bound:
            ts_ms = min_bound
        if max_bound is not None and ts_ms > max_bound:
            ts_ms = max_bound
        return ts_ms if ts_ms > 0 else fallback

    @staticmethod
    def _bookmark_fingerprint(alert: Mapping[str, Any]) -> str:
        title = str(alert.get("title") or "").strip().lower()
        description = str(alert.get("description") or "").strip().lower()[:180]
        severity = str(alert.get("severity") or "").strip().lower()
        state = str(alert.get("state") or "").strip().lower()
        payload = f"{title}|{description}|{severity}|{state}"
        return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()

    @staticmethod
    def _extract_balanced_json_blob(blob: str, start_idx: int) -> Optional[Tuple[str, int]]:
        if not isinstance(blob, str) or start_idx < 0 or start_idx >= len(blob):
            return None
        idx = start_idx
        while idx < len(blob) and blob[idx] != "{":
            idx += 1
        if idx >= len(blob):
            return None
        depth = 0
        in_string = False
        escaped = False
        end_idx = idx
        while end_idx < len(blob):
            ch = blob[end_idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return blob[idx : end_idx + 1], end_idx + 1
            end_idx += 1
        return None

    @classmethod
    def _alert_output_diagnostics(cls, summary_text: str) -> Dict[str, Any]:
        text = str(summary_text or "")
        json_count = 0
        seen_json: Set[str] = set()

        def add_json_candidate(raw: Any) -> None:
            nonlocal json_count
            if not isinstance(raw, Mapping):
                return
            alerts = raw.get("alerts")
            if not isinstance(alerts, Sequence) or isinstance(alerts, (str, bytes, bytearray)):
                return
            try:
                key = json.dumps(raw, ensure_ascii=False, sort_keys=True)
            except Exception:
                key = repr(raw)
            if key in seen_json:
                return
            seen_json.add(key)
            json_count += sum(1 for alert in alerts if isinstance(alert, Mapping))

        for match in re.finditer(r"```json(.*?)```", text, flags=re.DOTALL | re.IGNORECASE):
            try:
                add_json_candidate(json.loads(match.group(1)))
            except Exception:
                continue

        lowered = text.lower()
        marker = "alerts_json:"
        search_pos = 0
        while True:
            marker_idx = lowered.find(marker, search_pos)
            if marker_idx < 0:
                break
            chunk = cls._extract_balanced_json_blob(text, marker_idx + len(marker))
            if not chunk:
                search_pos = marker_idx + 1
                continue
            json_blob, next_idx = chunk
            try:
                add_json_candidate(json.loads(json_blob))
            except Exception:
                pass
            search_pos = max(next_idx, marker_idx + 1)

        for match in re.finditer(r"\{\s*\"alerts\"\s*:", text, flags=re.IGNORECASE):
            chunk = cls._extract_balanced_json_blob(text, match.start())
            if not chunk:
                continue
            try:
                add_json_candidate(json.loads(chunk[0]))
            except Exception:
                continue

        prose_count = len(
            re.findall(
                r"^\s*(?:[-*•]|\d+[.)])?\s*"
                r"(?:info(?:rmation(?:al)?)?|low|warn(?:ing)?|normal|moderate|high|critical|danger|emergency)"
                r"\s*(?:level|alert|severity)?\s*[:\-–]\s*\S+",
                text,
                flags=re.IGNORECASE | re.MULTILINE,
            )
        )
        return {
            "alerts_detected": cls._contains_alerts_json(text),
            "json_alert_count": int(max(0, json_count)),
            "prose_alert_count": int(max(0, prose_count)),
        }

    @staticmethod
    def _contains_alerts_json(summary_text: str) -> bool:
        text = str(summary_text or "")
        lowered = text.lower()
        if "```json" in lowered or "alerts_json:" in lowered:
            return True
        if re.search(r'^\s*\{\s*["\']alerts["\']\s*:', text, flags=re.IGNORECASE | re.MULTILINE):
            return True
        if re.search(
            r"^\s*(?:[-*•]|\d+[.)])?\s*"
            r"(?:info(?:rmation(?:al)?)?|low|warn(?:ing)?|normal|moderate|high|critical|danger|emergency)"
            r"\s*(?:level|alert|severity)?\s*[:\-–]\s*\S+",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        ):
            return True
        return False

    def _bookmark_recently_sent_locked(self, channel_id: int, fingerprint: str, now_ms: int, cooldown_sec: float) -> bool:
        if cooldown_sec <= 0:
            return False
        channel_key = int(channel_id)
        channel_cache = self.channel_bookmark_fingerprints.get(channel_key) or {}
        last_ts = channel_cache.get(fingerprint)
        if isinstance(last_ts, int) and (now_ms - last_ts) < int(cooldown_sec * 1000):
            return True
        return False

    @classmethod
    def _bookmark_cooldown_for_severity(cls, base_cooldown_sec: float, severity: Any) -> float:
        base = max(0.0, float(base_cooldown_sec or 0.0))
        normalized = cls._normalize_alert_severity(severity)
        if normalized in {"critical", "high"}:
            return 0.0
        return base

    def _mark_bookmark_sent_locked(self, channel_id: int, fingerprint: str, ts_ms: int) -> None:
        channel_key = int(channel_id)
        channel_cache = self.channel_bookmark_fingerprints.setdefault(channel_key, {})
        channel_cache[fingerprint] = int(ts_ms)
        # Prevent unbounded growth.
        prune_before = int(ts_ms) - 86400000
        stale_keys = [key for key, value in channel_cache.items() if isinstance(value, int) and value < prune_before]
        for key in stale_keys:
            channel_cache.pop(key, None)
        if len(channel_cache) > 2000:
            newest = sorted(channel_cache.items(), key=lambda item: item[1], reverse=True)[:1200]
            self.channel_bookmark_fingerprints[channel_key] = dict(newest)

    def process_summary_alerts(
        self,
        channel_id: int,
        summary_text: str,
        default_ts_ms: Optional[int] = None,
        min_ts_ms: Optional[int] = None,
        max_ts_ms: Optional[int] = None,
    ) -> AlertDeliveryResult:
        diagnostics = self._alert_output_diagnostics(summary_text)
        if not self.alert_parser:
            return AlertDeliveryResult(
                alerts_detected=bool(diagnostics.get("alerts_detected")),
                json_alert_count=int(diagnostics.get("json_alert_count") or 0),
                prose_alert_count=int(diagnostics.get("prose_alert_count") or 0),
            )
        base_ts_ms = self._normalize_alert_timestamp_ms_bounded(
            default_ts_ms,
            int(time.time() * 1000),
            min_ts_ms=min_ts_ms,
            max_ts_ms=max_ts_ms,
        )
        with self.cache_lock:
            settings = self._get_channel_bookmark_settings_locked(channel_id)
        cooldown_sec = float(settings.get("bookmark_cooldown_sec") or 0.0)
        if not bool(diagnostics.get("alerts_detected")):
            return AlertDeliveryResult(
                alerts_detected=False,
                json_alert_count=int(diagnostics.get("json_alert_count") or 0),
                prose_alert_count=int(diagnostics.get("prose_alert_count") or 0),
            )
        try:
            parsed_alerts = self.alert_parser(summary_text, int(channel_id), base_ts_ms)
        except TypeError:
            try:
                parsed_alerts = cast(Any, self.alert_parser)(summary_text, int(channel_id))
            except Exception as exc:
                return AlertDeliveryResult(
                    alerts_detected=True,
                    json_alert_count=int(diagnostics.get("json_alert_count") or 0),
                    prose_alert_count=int(diagnostics.get("prose_alert_count") or 0),
                    parser_error=str(exc)[:240] or exc.__class__.__name__,
                )
        except Exception as exc:
            return AlertDeliveryResult(
                alerts_detected=True,
                json_alert_count=int(diagnostics.get("json_alert_count") or 0),
                prose_alert_count=int(diagnostics.get("prose_alert_count") or 0),
                parser_error=str(exc)[:240] or exc.__class__.__name__,
            )
        if not isinstance(parsed_alerts, list) or not parsed_alerts:
            return AlertDeliveryResult(
                alerts_detected=True,
                json_alert_count=int(diagnostics.get("json_alert_count") or 0),
                prose_alert_count=int(diagnostics.get("prose_alert_count") or 0),
            )

        sent_count = 0
        failed_count = 0
        skipped_duplicate_count = 0
        last_error: Optional[str] = None
        alert_events: List[Dict[str, Any]] = []
        max_alerts = max(1, min(32, int(getattr(self, "alerts_max_per_batch", 8) or 8)))
        for raw_alert in parsed_alerts:
            if len(alert_events) >= max_alerts:
                break
            if not isinstance(raw_alert, Mapping):
                continue
            alert = {
                "title": str(raw_alert.get("title") or "Event"),
                "description": str(raw_alert.get("description") or ""),
                "severity": str(raw_alert.get("severity") or "normal"),
                "state": str(raw_alert.get("state") or "new"),
                "channel_id": int(channel_id),  # force observed stream channel
                "timestamp_ms": self._normalize_alert_timestamp_ms_bounded(
                    raw_alert.get("timestamp_ms"),
                    base_ts_ms,
                    min_ts_ms=min_ts_ms,
                    max_ts_ms=max_ts_ms,
                ),
            }
            if not bool(settings.get("bookmark_enabled")):
                alert_events.append({**alert, "delivery_status": "bookmark_disabled"})
                continue
            fingerprint = self._bookmark_fingerprint(alert)
            now_ms = int(time.time() * 1000)
            alert_cooldown_sec = self._bookmark_cooldown_for_severity(cooldown_sec, alert["severity"])
            with self.cache_lock:
                if self._bookmark_recently_sent_locked(int(channel_id), fingerprint, now_ms, alert_cooldown_sec):
                    skipped_duplicate_count += 1
                    alert_events.append({**alert, "delivery_status": "cooldown_skipped"})
                    continue
            try:
                self.send_bookmark_event(
                    channel_id=int(channel_id),
                    title=str(alert["title"]),
                    description=str(alert["description"]),
                    severity=str(alert["severity"]),
                    state=str(alert["state"]),
                    timestamp_ms=int(alert["timestamp_ms"]),
                )
            except Exception as exc:
                failed_count += 1
                last_error = str(exc)[:240] or exc.__class__.__name__
                alert_events.append({**alert, "delivery_status": "failed", "error": last_error})
                LOGGER.warning(
                    "Luxriot bookmark send failed channel_id=%s title=%r severity=%s error=%s",
                    channel_id,
                    alert["title"],
                    alert["severity"],
                    last_error,
                )
                continue
            with self.cache_lock:
                self._mark_bookmark_sent_locked(int(channel_id), fingerprint, now_ms)
            sent_count += 1
            alert_events.append({**alert, "delivery_status": "sent"})
        return AlertDeliveryResult(
            sent_count,
            parsed=len(parsed_alerts),
            json_alert_count=int(diagnostics.get("json_alert_count") or 0),
            prose_alert_count=int(diagnostics.get("prose_alert_count") or 0),
            failed=failed_count,
            skipped_duplicate=skipped_duplicate_count,
            last_error=last_error,
            alerts_detected=True,
            alert_events=alert_events,
        )

    def start_session(
        self,
        channel_id: int,
        batch_size: Optional[int] = None,
        prompt: str = "",
        model_hint: Optional[str] = None,
        system_prompt: Optional[str] = None,
        interval_sec: Optional[float] = None,
        update_desired: bool = True,
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
        normalized_model_hint = str(model_hint or "").strip() or None
        requested_interval_sec = self._normalize_capture_interval_sec(interval_sec)
        with self.cache_lock:
            existing = self.sessions.pop(channel_id, None)
            if existing:
                existing_status = existing.status()
                existing_logs = existing_status.get("logs")
                if isinstance(existing_logs, list):
                    self._merge_summary_history_locked(channel_id, existing_logs)
                self._close_run_locked(channel_id, existing_status.get("run_id"))
                existing.stop()
            if system_prompt is not None:
                overrides_raw = self.channel_prompt_overrides.get(channel_id)
                channel_overrides = dict(overrides_raw) if isinstance(overrides_raw, Mapping) else {}
                next_stream_prompt = str(system_prompt)
                if next_stream_prompt != str(channel_overrides.get("stream_system_prompt") or ""):
                    channel_overrides["stream_system_prompt"] = next_stream_prompt
            else:
                overrides_raw = self.channel_prompt_overrides.get(channel_id)
                channel_overrides = dict(overrides_raw) if isinstance(overrides_raw, Mapping) else {}
            if normalized_model_hint and normalized_model_hint != str(channel_overrides.get("model_hint") or ""):
                channel_overrides["model_hint"] = normalized_model_hint
            if requested_interval_sec is not None:
                current_interval = self._normalize_capture_interval_sec(
                    channel_overrides.get("capture_interval_sec")
                )
                if current_interval != requested_interval_sec:
                    channel_overrides["capture_interval_sec"] = requested_interval_sec
            if channel_overrides != dict(self.channel_prompt_overrides.get(channel_id) or {}):
                self.channel_prompt_overrides[channel_id] = channel_overrides
                self._persist_summary_state_locked()
            effective_interval_sec = self._get_capture_interval_sec_locked(channel_id)
            effective_system_prompt = self._get_stream_system_prompt_locked(channel_id)
            run = self._open_run_locked(
                channel_id=channel_id,
                batch_size=batch,
                prompt=prompt,
                model_hint=normalized_model_hint,
                system_prompt=effective_system_prompt,
                interval_sec=effective_interval_sec,
            )
            session = LuxriotCaptureSession(
                self,
                channel_id,
                batch,
                prompt,
                run_id=run.get("run_id"),
                run_started_at=run.get("started_at"),
                model_hint=normalized_model_hint,
                interval_override=effective_interval_sec,
                summarization_enabled=True,
                capture_kind="video",
            )
            existing_probe = self.probe_sessions.pop(channel_id, None)
            if existing_probe is not None:
                existing_probe.stop()
                if channel_id not in self.paused_probe_channels:
                    self.shared_probe_channels.add(channel_id)
            if update_desired:
                self._set_desired_live_session(
                    channel_id,
                    enabled=True,
                    batch_size=batch,
                    prompt=prompt,
                    model_hint=normalized_model_hint,
                    interval_sec=effective_interval_sec,
                )
            self.sessions[channel_id] = session
            session.start()
            return session.status()

    def stop_session(self, channel_id: int, *, update_desired: bool = True) -> Dict[str, Any]:
        with self.cache_lock:
            session = self.sessions.pop(channel_id, None)
        if session:
            status = session.status()
            logs = status.get("logs")
            session.stop()
            with self.cache_lock:
                if isinstance(logs, list):
                    self._merge_summary_history_locked(channel_id, logs)
                self._close_run_locked(channel_id, status.get("run_id"))
                archived_count = len(self.summary_history.get(channel_id, []))
            if update_desired:
                self._set_desired_live_session(channel_id, enabled=False)
            return {
                "channel_id": channel_id,
                "run_id": status.get("run_id"),
                "running": False,
                "archived_log_count": archived_count,
            }
        with self.cache_lock:
            self._close_run_locked(channel_id, None)
            archived_count = len(self.summary_history.get(channel_id, []))
        if update_desired:
            self._set_desired_live_session(channel_id, enabled=False)
        return {
            "channel_id": channel_id,
            "running": False,
            "archived_log_count": archived_count,
            "message": "No active session",
        }

    def start_probe_capture(self, channel_id: int, fps: Optional[float] = None, clear_pause: bool = True) -> Dict[str, Any]:
        with self.cache_lock:
            if clear_pause:
                self.paused_probe_channels.discard(channel_id)
            elif channel_id in self.paused_probe_channels:
                self.shared_probe_channels.discard(channel_id)
                return {
                    "channel_id": channel_id,
                    "running": False,
                    "paused": True,
                    "message": "Probe capture paused",
                    "capture_kind": "analytics",
                    "summarization_enabled": False,
                }
            existing = self.probe_sessions.get(channel_id)
            video_session = self.sessions.get(channel_id)
            if video_session is not None:
                if existing is not None:
                    self.probe_sessions.pop(channel_id, None)
                    existing.stop()
                self.shared_probe_channels.add(channel_id)
                return self._shared_probe_capture_status_locked(
                    channel_id,
                    video_session,
                    requested_fps=fps,
                    paused=False,
                )
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

    def _shared_probe_capture_status_locked(
        self,
        channel_id: int,
        video_session: Optional[LuxriotCaptureSession] = None,
        *,
        requested_fps: Optional[float] = None,
        paused: bool = False,
    ) -> Dict[str, Any]:
        source = video_session or self.sessions.get(channel_id)
        if source is None:
            return {
                "channel_id": channel_id,
                "running": False,
                "paused": bool(paused),
                "capture_kind": "analytics",
                "summarization_enabled": False,
                "shared_capture": True,
                "shared_source_stream_type": "video",
                "message": "No active video summary capture to share",
            }
        status = self._compact_stream_status("analytics", source.status(), self.paused_probe_channels)
        status["stream_type"] = "analytics"
        status["capture_kind"] = "analytics"
        status["summarization_enabled"] = False
        status["shared_capture"] = True
        status["shared_source_stream_type"] = "video"
        status["shared_source_capture_kind"] = "video"
        status["running"] = bool(status.get("running")) and not bool(paused)
        status["paused"] = bool(paused)
        if requested_fps and requested_fps > 0:
            status["requested_fps"] = round(float(requested_fps), 3)
        status["message"] = "Probe capture is shared with the active video-summary capture loop"
        return status

    def stop_probe_capture(self, channel_id: int, pause: bool = True) -> Dict[str, Any]:
        with self.cache_lock:
            if pause:
                self.paused_probe_channels.add(channel_id)
            else:
                self.paused_probe_channels.discard(channel_id)
            shared = channel_id in self.shared_probe_channels
            self.shared_probe_channels.discard(channel_id)
            session = self.probe_sessions.pop(channel_id, None)
        if session:
            session.stop()
            return {"channel_id": channel_id, "running": False, "paused": pause}
        if shared:
            return {
                "channel_id": channel_id,
                "running": False,
                "paused": pause,
                "shared_capture": True,
                "message": "Stopped shared probe capture",
            }
        return {
            "channel_id": channel_id,
            "running": False,
            "paused": pause,
            "message": "No active probe capture",
        }

    def is_probe_capture_paused(self, channel_id: int) -> bool:
        with self.cache_lock:
            return channel_id in self.paused_probe_channels

    def probe_frame_thumbnail(self, channel_id: int, timestamp_ms: Optional[int] = None) -> Optional[str]:
        with self.cache_lock:
            session = self.probe_sessions.get(channel_id)
            video_session = self.sessions.get(channel_id)
        if session is None:
            session = video_session
        if session is None:
            return None
        try:
            return session.nearest_frame_thumbnail(timestamp_ms)
        except Exception:
            return None

    def flush_session(self, channel_id: int) -> Dict[str, Any]:
        with self.cache_lock:
            session = self.sessions.get(channel_id)
        if not session:
            return {"success": False, "message": "No active session"}
        session.flush_now()
        return {"success": True, "message": "Flushed buffered frames", "status": session.status()}

    def session_status(
        self,
        channel_id: int,
        run_selector: Optional[str] = None,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        if start_ts is not None and end_ts is not None and start_ts > end_ts:
            start_ts, end_ts = end_ts, start_ts
        with self.cache_lock:
            session = self.sessions.get(channel_id)
            history_logs = list(self.summary_history.get(channel_id, []))
            run_items = [dict(run) for run in self.summary_runs.get(channel_id, []) if isinstance(run, Mapping)]
            active_run_id = str(self.active_summary_runs.get(channel_id) or "").strip() or None
        if session:
            status = session.status()
            current_logs = status.get("logs")
            current_list = current_logs if isinstance(current_logs, list) else []
            all_logs = self._combine_summary_logs(history_logs, current_list)
            running_run_id = str(status.get("run_id") or "").strip() or active_run_id
            if running_run_id:
                active_run_id = running_run_id
            log_count_by_run: Dict[str, int] = {}
            for entry in all_logs:
                run_id = str(entry.get("run_id") or "").strip()
                if run_id:
                    log_count_by_run[run_id] = log_count_by_run.get(run_id, 0) + 1
            for run in run_items:
                run_id = str(run.get("run_id") or "").strip()
                run["log_count"] = int(log_count_by_run.get(run_id, 0))
                run["running"] = bool(run_id and run_id == running_run_id)
                if run["running"]:
                    run["ended_at"] = None
            run_items.sort(key=lambda item: float(item.get("started_at") or 0.0), reverse=True)
            selected_run, selected_run_id, latest_run_id = self._resolve_run_selector(run_selector, run_items, running_run_id)
            filtered_logs = self._filter_summary_logs(all_logs, selected_run_id, start_ts, end_ts)
            if isinstance(limit, int) and limit > 0 and len(filtered_logs) > limit:
                filtered_logs = filtered_logs[-limit:]
            status["logs"] = filtered_logs
            status["logs_total"] = len(all_logs)
            status["logs_filtered"] = len(filtered_logs)
            status["archived_log_count"] = len(history_logs)
            status["runs"] = run_items
            status["running_run_id"] = running_run_id
            status["latest_run_id"] = latest_run_id
            status["selected_run"] = selected_run
            status["run_filter_id"] = selected_run_id
            status["from_ts"] = start_ts
            status["to_ts"] = end_ts
            status["limit"] = limit
            return status
        all_logs = list(history_logs)
        log_count_by_run: Dict[str, int] = {}
        for entry in all_logs:
            run_id = str(entry.get("run_id") or "").strip()
            if run_id:
                log_count_by_run[run_id] = log_count_by_run.get(run_id, 0) + 1
        for run in run_items:
            run_id = str(run.get("run_id") or "").strip()
            run["log_count"] = int(log_count_by_run.get(run_id, 0))
            run["running"] = bool(run_id and run_id == active_run_id)
            if run["running"]:
                run["ended_at"] = None
        run_items.sort(key=lambda item: float(item.get("started_at") or 0.0), reverse=True)
        selected_run, selected_run_id, latest_run_id = self._resolve_run_selector(run_selector, run_items, active_run_id)
        filtered_logs = self._filter_summary_logs(all_logs, selected_run_id, start_ts, end_ts)
        if isinstance(limit, int) and limit > 0 and len(filtered_logs) > limit:
            filtered_logs = filtered_logs[-limit:]
        return {
            "running": False,
            "channel_id": channel_id,
            "run_id": active_run_id,
            "batch_size": None,
            "pending_frames": 0,
            "interval_sec": getattr(self.config, "LUXRIOT_SNAPSHOT_INTERVAL", 5),
            "max_edge": getattr(self.config, "LUXRIOT_SNAPSHOT_MAX_EDGE", 800),
            "capture_kind": "video",
            "summarization_enabled": True,
            "last_error": None,
            "logs": filtered_logs,
            "logs_total": len(all_logs),
            "logs_filtered": len(filtered_logs),
            "archived_log_count": len(history_logs),
            "runs": run_items,
            "running_run_id": active_run_id,
            "latest_run_id": latest_run_id,
            "selected_run": selected_run,
            "run_filter_id": selected_run_id,
            "from_ts": start_ts,
            "to_ts": end_ts,
            "limit": limit,
        }

    def summary_rollups(
        self,
        channel_id: int,
        run_selector: Optional[str] = None,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
        level_limit: Optional[int] = 60,
        synthesize: bool = True,
    ) -> Dict[str, Any]:
        status = self.session_status(
            channel_id=channel_id,
            run_selector=run_selector,
            start_ts=start_ts,
            end_ts=end_ts,
            limit=None,
        )
        logs_raw = status.get("logs")
        logs = logs_raw if isinstance(logs_raw, list) else []

        l0_nodes = self._l0_nodes_from_logs(channel_id, logs)
        l1_nodes = self._build_rollup_level(
            channel_id=channel_id,
            level="L1",
            source_level="L0",
            window_sec=self.rollup_windows["L1"],
            source_nodes=l0_nodes,
            synthesize=synthesize,
        )
        l2_nodes = self._build_rollup_level(
            channel_id=channel_id,
            level="L2",
            source_level="L1",
            window_sec=self.rollup_windows["L2"],
            source_nodes=l1_nodes,
            synthesize=synthesize,
        )
        l3_nodes = self._build_rollup_level(
            channel_id=channel_id,
            level="L3",
            source_level="L2",
            window_sec=self.rollup_windows["L3"],
            source_nodes=l2_nodes,
            synthesize=synthesize,
        )

        selected_run_id = str(status.get("run_filter_id") or "").strip() or None
        stored_rollups = self._list_cached_rollups(channel_id=channel_id, start_ts=start_ts, end_ts=end_ts)
        if selected_run_id:
            stored_rollups = [row for row in stored_rollups if self._rollup_matches_run_selector(row, selected_run_id)]
        stored_by_level: Dict[str, List[Dict[str, Any]]] = {"L1": [], "L2": [], "L3": []}
        for row in stored_rollups:
            level = self._normalize_rollup_level(row.get("level"))
            if level in stored_by_level:
                stored_by_level[level].append(dict(row))

        l1_nodes = self._merge_rollup_rows(l1_nodes, stored_by_level["L1"])
        l2_nodes = self._merge_rollup_rows(l2_nodes, stored_by_level["L2"])
        l3_nodes = self._merge_rollup_rows(l3_nodes, stored_by_level["L3"])

        if isinstance(level_limit, int) and level_limit > 0:
            l0_nodes = l0_nodes[-level_limit:]
            l1_nodes = l1_nodes[-level_limit:]
            l2_nodes = l2_nodes[-level_limit:]
            l3_nodes = l3_nodes[-level_limit:]
        self._refresh_channel_memory_from_rollups(
            channel_id,
            [*l1_nodes, *l2_nodes, *l3_nodes],
        )
        stored_counts: Dict[str, int] = {}
        for entry in stored_rollups:
            level = str(entry.get("level") or "").strip().upper() or "UNKNOWN"
            stored_counts[level] = stored_counts.get(level, 0) + 1
        with self.cache_lock:
            routine_context = dict(self.channel_routine_context.get(channel_id, {}))

        return {
            "channel_id": channel_id,
            "running": bool(status.get("running")),
            "runs": status.get("runs"),
            "selected_run": status.get("selected_run"),
            "run_filter_id": status.get("run_filter_id"),
            "running_run_id": status.get("running_run_id"),
            "latest_run_id": status.get("latest_run_id"),
            "from_ts": start_ts,
            "to_ts": end_ts,
            "window_sec": dict(self.rollup_windows),
            "level_limit": level_limit,
            "rollup_mode": "time-only" if self.rollup_time_only else "token-gated",
            "min_source_tokens": self.rollup_min_source_tokens,
            "stored_rollups_count": len(stored_rollups),
            "stored_counts": stored_counts,
            "routine_context": routine_context,
            "source_counts": {
                "L0": len(l0_nodes),
                "L1": len(l1_nodes),
                "L2": len(l2_nodes),
                "L3": len(l3_nodes),
            },
            "levels": {
                "L0": l0_nodes,
                "L1": l1_nodes,
                "L2": l2_nodes,
                "L3": l3_nodes,
            },
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
        latest_log = logs[-1] if isinstance(logs, list) and logs else None
        if isinstance(latest_log, Mapping):
            compact["last_summary_at"] = latest_log.get("created_at")
            compact["last_summary_batch_end_ms"] = latest_log.get("batch_end_ms")
            compact["last_alert_total"] = latest_log.get("alert_total")
            compact["last_alert_counts"] = latest_log.get("alert_counts")
            compact["last_alert_severities"] = latest_log.get("alert_severities")
            compact["last_bookmark_failed_count"] = latest_log.get("bookmark_failed_count")
            compact["last_bookmark_last_error"] = latest_log.get("bookmark_last_error")
        for field in (
            "snapshot_count",
            "snapshot_failed_count",
            "slow_snapshot_count",
            "snapshot_slow_threshold_sec",
            "last_snapshot_latency_sec",
            "avg_snapshot_latency_sec",
            "max_snapshot_latency_sec",
            "last_snapshot_at",
            "capture_source_mode",
            "active_capture_source",
            "live_segment_count",
            "live_segment_failed_count",
            "live_segment_frame_count",
            "last_live_segment_latency_sec",
            "last_live_segment_frames",
            "last_live_segment_error",
            "frozen_signal",
            "frozen_signal_since",
            "frozen_signal_age_sec",
            "frozen_frame_count",
            "frozen_frame_hash",
            "frozen_frame_dropped_count",
        ):
            if field in status:
                compact[field] = status.get(field)
        compact["stream_type"] = stream_type
        if paused_channels is not None:
            channel_id = compact.get("channel_id")
            compact["paused"] = bool(isinstance(channel_id, int) and channel_id in paused_channels)
        return compact

    def streams_status(self) -> Dict[str, Any]:
        try:
            desired_live = self._load_desired_live_sessions()
        except Exception:
            desired_live = {}
        with self.cache_lock:
            video_items = list(self.sessions.items())
            analytics_items = list(self.probe_sessions.items())
            shared_probe_channels = set(self.shared_probe_channels)
            paused = set(self.paused_probe_channels)
            history_channels = sorted(channel_id for channel_id, logs in self.summary_history.items() if logs)
            restore_errors = dict(self.live_session_restore_errors)
            status_digest = {
                int(channel_id): dict(digest)
                for channel_id, digest in self.channel_status_digest.items()
                if isinstance(digest, Mapping)
            }
        video_streams = [
            self._compact_stream_status("video", session.status(), paused)
            for _, session in video_items
        ]
        running_video_channels = {
            int(item.get("channel_id") or 0)
            for item in video_streams
        }
        desired_video_channels = sorted(
            channel_id
            for channel_id, state in desired_live.items()
            if bool(state.get("enabled"))
        )
        desired_missing = [
            {
                "channel_id": channel_id,
                "desired": True,
                "running": False,
                "last_restore_error": restore_errors.get(channel_id)
                or str(desired_live.get(channel_id, {}).get("last_restore_error") or "").strip()
                or None,
            }
            for channel_id in desired_video_channels
            if channel_id not in running_video_channels
        ]
        for item in video_streams:
            channel_id = int(item.get("channel_id") or 0)
            item["desired"] = channel_id in desired_video_channels
            item["last_restore_error"] = restore_errors.get(channel_id)
            digest = status_digest.setdefault(
                channel_id,
                {
                    "channel_id": channel_id,
                    "summary_count": int(item.get("log_count") or 0),
                    "alert_total": int(item.get("last_alert_total") or 0),
                    "alert_counts_by_severity": dict(item.get("last_alert_counts") or {}),
                    "recent_alerts": [],
                    "alert_delivery_breakdown": {},
                    "alert_parser_breakdown": {},
                    "state_transition_total": 0,
                    "current_observed_state": [],
                    "recent_state_transitions": [],
                    "vector_signal_total": 0,
                    "recent_vector_signals": [],
                    "rebuilt_from_history": False,
                    "source": "runtime",
                },
            )
            digest["desired"] = channel_id in desired_video_channels
            digest["last_restore_error"] = restore_errors.get(channel_id)
            self._overlay_stream_runtime_on_digest(digest, item)
        for channel_id in desired_video_channels:
            digest = status_digest.setdefault(
                int(channel_id),
                {
                    "channel_id": int(channel_id),
                    "summary_count": 0,
                    "alert_total": 0,
                    "alert_counts_by_severity": {},
                    "recent_alerts": [],
                    "alert_delivery_breakdown": {},
                    "alert_parser_breakdown": {},
                    "state_transition_total": 0,
                    "current_observed_state": [],
                    "recent_state_transitions": [],
                    "vector_signal_total": 0,
                    "recent_vector_signals": [],
                    "rebuilt_from_history": False,
                    "source": "desired",
                },
            )
            digest["desired"] = True
            digest.setdefault("running", False)
            digest.setdefault("last_restore_error", restore_errors.get(int(channel_id)))
        analytics_streams = [
            self._compact_stream_status("analytics", session.status(), paused)
            for _, session in analytics_items
        ]
        analytics_channels = {
            int(item.get("channel_id") or 0)
            for item in analytics_streams
        }
        video_sessions_by_channel = {int(channel_id): session for channel_id, session in video_items}
        for channel_id in sorted(shared_probe_channels):
            if channel_id in analytics_channels:
                continue
            video_session = video_sessions_by_channel.get(int(channel_id))
            if video_session is None:
                continue
            analytics_streams.append(
                self._shared_probe_capture_status_locked(
                    int(channel_id),
                    video_session,
                    paused=int(channel_id) in paused,
                )
            )
        return {
            "video_streams": sorted(video_streams, key=lambda item: int(item.get("channel_id", 0))),
            "analytics_streams": sorted(analytics_streams, key=lambda item: int(item.get("channel_id", 0))),
            "channel_status_digest": sorted(status_digest.values(), key=lambda item: int(item.get("channel_id", 0))),
            "desired_video_channels": desired_video_channels,
            "desired_video_missing": desired_missing,
            "paused_analytics_channels": sorted(paused),
            "video_history_channels": history_channels,
            "running_total": len(video_streams) + len(analytics_streams),
            "capture_thread_total": len(video_streams) + len(analytics_items),
            "shared_analytics_count": sum(1 for item in analytics_streams if bool(item.get("shared_capture"))),
        }

    def system_status_digest(
        self,
        channel_ids: Optional[Sequence[int]] = None,
        *,
        compact: bool = True,
    ) -> Dict[str, Any]:
        status = self.streams_status()
        rows = status.get("channel_status_digest") if isinstance(status, Mapping) else []
        wanted = {
            int(item)
            for item in (channel_ids or [])
            if _parse_optional_int(item) is not None and int(item) > 0
        }
        output: List[Dict[str, Any]] = []
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, Mapping):
                continue
            channel_id = _parse_optional_int(row.get("channel_id"))
            if channel_id is None:
                continue
            if wanted and int(channel_id) not in wanted:
                continue
            item = dict(row)
            if compact:
                item["recent_alerts"] = list(item.get("recent_alerts") or [])[:5]
                item["recent_state_transitions"] = list(item.get("recent_state_transitions") or [])[:5]
                item["current_observed_state"] = list(item.get("current_observed_state") or [])[:8]
                item["recent_vector_signals"] = list(item.get("recent_vector_signals") or [])[:5]
            output.append(item)
        return {
            "channels": output,
            "count": len(output),
            "compact": bool(compact),
            "source": "luxriot_channel_status_digest",
        }

    def stop_stream(
        self,
        channel_id: int,
        stream_type: str = "both",
        pause_analytics: bool = True,
        update_desired: bool = True,
    ) -> Dict[str, Any]:
        normalized = (stream_type or "both").strip().lower()
        result: Dict[str, Any] = {"channel_id": channel_id, "stream_type": normalized}
        if normalized in {"video", "summary", "summaries"}:
            result["video"] = self.stop_session(channel_id, update_desired=update_desired)
        elif normalized in {"analytics", "probe", "probes"}:
            result["analytics"] = self.stop_probe_capture(channel_id, pause=pause_analytics)
        elif normalized in {"both", "all"}:
            result["video"] = self.stop_session(channel_id, update_desired=update_desired)
            result["analytics"] = self.stop_probe_capture(channel_id, pause=pause_analytics)
        else:
            raise ValueError("stream_type must be one of: video, analytics, both")
        return result

    def stop_all_streams(
        self,
        stop_video: bool = True,
        stop_analytics: bool = True,
        pause_analytics: bool = True,
        update_desired: bool = True,
    ) -> Dict[str, Any]:
        with self.cache_lock:
            video_channels = list(self.sessions.keys()) if stop_video else []
            analytics_channels = (
                sorted(set(self.probe_sessions.keys()) | set(self.shared_probe_channels))
                if stop_analytics
                else []
            )
        stopped_video = [
            self.stop_session(ch, update_desired=update_desired)
            for ch in video_channels
        ]
        stopped_analytics = [self.stop_probe_capture(ch, pause=pause_analytics) for ch in analytics_channels]
        return {
            "stopped_video_count": len(stopped_video),
            "stopped_analytics_count": len(stopped_analytics),
            "video": stopped_video,
            "analytics": stopped_analytics,
        }
