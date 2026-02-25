import hashlib
import json
import math
import re
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Set, Tuple, cast

import requests
from PIL import Image
from requests.auth import HTTPDigestAuth

DEFAULT_ALERTS_JSON_PROMPT = (
    "Optional bookmark output (emit only when a Task-defined trigger is observed in this batch):\n"
    "- If no trigger match: emit no JSON block.\n"
    "- If a trigger matches: append exactly one block at the end, prefixed with ALERTS_JSON:, using this schema:\n"
    "ALERTS_JSON:\n"
    "{\n"
    "  \"alerts\": [\n"
    "    {\n"
    "      \"title\": \"Short event title\",\n"
    "      \"description\": \"<= 240 chars, concrete and actionable\",\n"
    "      \"severity\": \"info|low|normal|high|critical\",\n"
    "      \"state\": \"new\",\n"
    "      \"channel_id\": {channel_id},\n"
    "      \"timestamp_ms\": 1772202050000\n"
    "    }\n"
    "  ]\n"
    "}\n"
    "Rules: max 3 alerts; do not alert routine micro-movements unless explicitly requested; timestamp_ms should be batch time in ms."
)


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
            base_system_prompt = self.manager.get_effective_stream_system_prompt(self.channel_id)
            system_prompt = self.manager.compose_live_system_prompt(self.channel_id, base_system_prompt)
            messages = self.manager.message_builder(f"#{self.channel_id}", frames_copy, self.prompt, system_prompt)
            summary = self.manager.lm_callback(messages, self.model_hint)
            duration = time.time() - started
            created_at = time.time()
            entry = {
                "channel_id": self.channel_id,
                "run_id": self.run_id,
                "summary": summary,
                "frame_count": len(frames_copy),
                "batch_size": self.batch_size,
                "created_at": created_at,
                "duration_sec": duration,
                "prompt": self.prompt,
            }
            try:
                sent_alerts = int(self.manager.process_summary_alerts(self.channel_id, summary, int(created_at * 1000)))
            except Exception:
                sent_alerts = 0
            entry["bookmarks_sent"] = sent_alerts
            with self.lock:
                self.logs.append(entry)
                self.total_flushes += 1
                if len(self.logs) > 50:
                    self.logs = self.logs[-50:]
            try:
                self.manager.record_summary_log(self.channel_id, entry)
            except Exception:
                pass
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
            "run_id": self.run_id,
            "run_started_at": self.run_started_at,
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

    def nearest_frame_thumbnail(self, timestamp_ms: Optional[int] = None) -> Optional[str]:
        with self.lock:
            frames_copy = list(self.frames)
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
        try:
            history_limit = int(getattr(config, "LUXRIOT_SUMMARY_HISTORY_LIMIT", 600))
        except Exception:
            history_limit = 600
        self.summary_history_limit = max(40, history_limit)
        self.summary_history: Dict[int, List[Dict[str, Any]]] = {}
        self.summary_runs: Dict[int, List[Dict[str, Any]]] = {}
        self.active_summary_runs: Dict[int, str] = {}
        self.channel_routine_context: Dict[int, Dict[str, Any]] = {}
        self.channel_prompt_overrides: Dict[int, Dict[str, Any]] = {}
        self.channel_bookmark_fingerprints: Dict[int, Dict[str, int]] = {}
        self.default_bookmark_enabled = bool(getattr(config, "LUXRIOT_AUTO_BOOKMARKS", False))
        try:
            cooldown_value = float(getattr(config, "LUXRIOT_BOOKMARK_COOLDOWN_SEC", 60.0))
        except Exception:
            cooldown_value = 60.0
        self.default_bookmark_cooldown_sec = max(0.0, cooldown_value)
        self.default_json_alert_prompt = str(
            getattr(config, "LUXRIOT_ALERTS_JSON_PROMPT", DEFAULT_ALERTS_JSON_PROMPT) or DEFAULT_ALERTS_JSON_PROMPT
        ).strip() or DEFAULT_ALERTS_JSON_PROMPT
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
        parsed_levels = {token.strip().upper() for token in llm_levels_raw.split(",") if token.strip()}
        allowed_levels = {"L1", "L2", "L3"}
        self.rollup_llm_levels: Set[str] = parsed_levels.intersection(allowed_levels) if parsed_levels else {"L1", "L2", "L3"}
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
            merged[key] = dict(item)
        ordered.sort(key=lambda key: float(key[0]))
        return [merged[key] for key in ordered]

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
        return {
            "channel_id": int(channel_id),
            "run_id": str(entry.get("run_id") or "").strip(),
            "summary": summary,
            "frame_count": int(max(0, frame_count)),
            "batch_size": int(max(0, batch_size)),
            "created_at": float(created_at),
            "duration_sec": float(max(0.0, duration_sec)),
            "prompt": str(entry.get("prompt") or ""),
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
            "model": str(entry.get("model") or "").strip() or None,
            "prompt": str(entry.get("prompt") or ""),
            "system_prompt": str(entry.get("system_prompt") or ""),
        }

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
        path = self.summary_state_file
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = path.with_suffix(f"{path.suffix}.tmp")
            tmp_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_file.replace(path)
        except Exception:
            return

    def _load_summary_state_from_disk(self) -> None:
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
                combined = self._combine_summary_logs([], normalized_logs)
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
                    dedup.values(),
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
                loaded_routines[int(channel_id)] = {
                    "channel_id": int(channel_id),
                    "rollup_id": str(routine_value.get("rollup_id") or "").strip(),
                    "window_end": float(self._coerce_float(routine_value.get("window_end")) or 0.0),
                    "routine": routine_text,
                    "updated_at": float(self._coerce_float(routine_value.get("updated_at")) or time.time()),
                }
        loaded_stream_system_prompt: Optional[str] = None
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
            if "bookmark_enabled" in prompt_settings_raw:
                loaded_default_bookmark_enabled = bool(prompt_settings_raw.get("bookmark_enabled"))
            if "bookmark_cooldown_sec" in prompt_settings_raw:
                try:
                    loaded_default_bookmark_cooldown_sec = max(0.0, float(prompt_settings_raw.get("bookmark_cooldown_sec")))
                except Exception:
                    loaded_default_bookmark_cooldown_sec = 0.0
            if "json_alert_prompt" in prompt_settings_raw:
                loaded_default_json_alert_prompt = str(prompt_settings_raw.get("json_alert_prompt") or "").strip()
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
                    if "bookmark_enabled" in channel_payload:
                        parsed_channel_payload["bookmark_enabled"] = bool(channel_payload.get("bookmark_enabled"))
                    if "bookmark_cooldown_sec" in channel_payload:
                        try:
                            parsed_channel_payload["bookmark_cooldown_sec"] = max(0.0, float(channel_payload.get("bookmark_cooldown_sec")))
                        except Exception:
                            parsed_channel_payload["bookmark_cooldown_sec"] = 0.0
                    if "json_alert_prompt" in channel_payload:
                        parsed_channel_payload["json_alert_prompt"] = str(channel_payload.get("json_alert_prompt") or "")
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
            if loaded_stream_system_prompt is not None:
                self.system_prompt = loaded_stream_system_prompt
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
    ) -> Dict[str, Any]:
        started_at = time.time()
        run = {
            "run_id": self._generate_run_id_locked(channel_id),
            "channel_id": channel_id,
            "started_at": started_at,
            "ended_at": None,
            "running": True,
            "batch_size": int(batch_size),
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
            created = LuxriotManager._coerce_float(item.get("created_at"))
            if start_ts is not None and (created is None or created < start_ts):
                continue
            if end_ts is not None and (created is None or created > end_ts):
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

    def _extract_routine_hint(self, summary_text: object) -> str:
        text = str(summary_text or "").strip()
        if not text:
            return ""
        baseline = self._extract_markdown_section(text, "Scene Baseline")
        notes = self._extract_markdown_section(text, "Operator Notes")
        chunks: List[str] = []
        if baseline:
            chunks.append(f"Scene baseline: {' '.join(baseline.split())}")
        if notes:
            chunks.append(f"Operator notes: {' '.join(notes.split())}")
        if not chunks:
            cleaned = self._sanitize_l0_summary(text, max_len=420)
            if cleaned:
                chunks.append(cleaned)
        hint = " ".join(chunks).strip()
        if len(hint) > 900:
            hint = f"{hint[:897].rstrip()}..."
        return hint

    def _update_channel_routine_context(
        self,
        channel_id: int,
        rollup_id: str,
        summary_text: object,
        window_end: Optional[float],
    ) -> None:
        routine_hint = self._extract_routine_hint(summary_text)
        if not routine_hint:
            return
        channel_key = int(channel_id)
        rollup_key = str(rollup_id or "").strip()
        window_end_value = self._coerce_float(window_end) or 0.0
        changed = False
        with self.cache_lock:
            current = self.channel_routine_context.get(channel_key)
            current_window_end = self._coerce_float(current.get("window_end")) if isinstance(current, Mapping) else None
            current_rollup_id = str(current.get("rollup_id") or "").strip() if isinstance(current, Mapping) else ""
            current_hint = str(current.get("routine") or "").strip() if isinstance(current, Mapping) else ""
            should_replace = (
                current is None
                or window_end_value > float(current_window_end or 0.0)
                or (current_rollup_id == rollup_key and current_hint != routine_hint)
            )
            if should_replace:
                self.channel_routine_context[channel_key] = {
                    "channel_id": channel_key,
                    "rollup_id": rollup_key,
                    "window_end": window_end_value,
                    "routine": routine_hint,
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
        if not routine:
            return ""
        return (
            "Channel routine baseline from prior long-window summaries:\n"
            f"{routine}\n"
            "Use this as context for what is typical in this stream. Preserve key deviations and anomalies."
        )

    def compose_live_system_prompt(self, channel_id: int, base_prompt: Optional[str]) -> str:
        base = str(base_prompt or "").strip()
        routine = self._get_channel_routine_prompt(channel_id)
        if routine:
            return f"{base}\n\n{routine}" if base else routine
        return base

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

    def _get_channel_bookmark_settings_locked(self, channel_id: Optional[int] = None) -> Dict[str, Any]:
        enabled = bool(self.default_bookmark_enabled)
        cooldown_sec = float(max(0.0, self.default_bookmark_cooldown_sec))
        json_prompt = str(self.default_json_alert_prompt or DEFAULT_ALERTS_JSON_PROMPT)
        if channel_id is not None:
            overrides = self.channel_prompt_overrides.get(int(channel_id))
            if isinstance(overrides, Mapping):
                if "bookmark_enabled" in overrides:
                    enabled = bool(overrides.get("bookmark_enabled"))
                if "bookmark_cooldown_sec" in overrides:
                    try:
                        cooldown_sec = max(0.0, float(overrides.get("bookmark_cooldown_sec")))
                    except Exception:
                        cooldown_sec = 0.0
                if "json_alert_prompt" in overrides:
                    json_prompt = str(overrides.get("json_alert_prompt") or "")
        return {
            "bookmark_enabled": enabled,
            "bookmark_cooldown_sec": cooldown_sec,
            "json_alert_prompt": json_prompt or DEFAULT_ALERTS_JSON_PROMPT,
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

    def _get_rollup_system_prompt_locked(self, level: str, channel_id: Optional[int] = None) -> str:
        normalized_level = self._normalize_rollup_level(level)
        if channel_id is not None:
            overrides = self.channel_prompt_overrides.get(int(channel_id))
            if isinstance(overrides, Mapping):
                rollup_prompts = overrides.get("rollup_prompts")
                if isinstance(rollup_prompts, Mapping) and normalized_level in rollup_prompts:
                    return str(rollup_prompts.get(normalized_level) or "")
        return self._default_rollup_prompt_for_level_locked(normalized_level)

    def get_stream_system_prompt(self, channel_id: Optional[int] = None) -> str:
        with self.cache_lock:
            return self._get_stream_system_prompt_locked(channel_id)

    def get_effective_stream_system_prompt(self, channel_id: int) -> str:
        with self.cache_lock:
            base_prompt = self._get_stream_system_prompt_locked(channel_id)
            bookmark_settings = self._get_channel_bookmark_settings_locked(channel_id)
        if bool(bookmark_settings.get("bookmark_enabled")):
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
                "rollup_prompts": {
                    "L1": self._default_rollup_prompt_for_level_locked("L1"),
                    "L2": self._default_rollup_prompt_for_level_locked("L2"),
                    "L3": self._default_rollup_prompt_for_level_locked("L3"),
                },
                "bookmark_enabled": bool(defaults_bookmark.get("bookmark_enabled")),
                "bookmark_cooldown_sec": float(defaults_bookmark.get("bookmark_cooldown_sec") or 0.0),
                "json_alert_prompt": str(defaults_bookmark.get("json_alert_prompt") or DEFAULT_ALERTS_JSON_PROMPT),
            }
            effective_stream_prompt = self._get_stream_system_prompt_locked(channel_id)
            effective_rollup_prompts = {
                "L1": self._get_rollup_system_prompt_locked("L1", channel_id),
                "L2": self._get_rollup_system_prompt_locked("L2", channel_id),
                "L3": self._get_rollup_system_prompt_locked("L3", channel_id),
            }
            effective_bookmark = self._get_channel_bookmark_settings_locked(channel_id)
            has_channel_override = bool(
                channel_id is not None and isinstance(self.channel_prompt_overrides.get(int(channel_id)), Mapping)
            )
        return {
            "channel_id": int(channel_id) if channel_id is not None else None,
            "stream_system_prompt": effective_stream_prompt,
            "rollup_prompts": effective_rollup_prompts,
            "bookmark_enabled": bool(effective_bookmark.get("bookmark_enabled")),
            "bookmark_cooldown_sec": float(effective_bookmark.get("bookmark_cooldown_sec") or 0.0),
            "json_alert_prompt": str(effective_bookmark.get("json_alert_prompt") or DEFAULT_ALERTS_JSON_PROMPT),
            "defaults": defaults,
            "has_channel_override": has_channel_override,
        }

    def update_prompt_settings(
        self,
        channel_id: Optional[int] = None,
        stream_system_prompt: Optional[str] = None,
        rollup_prompts: Optional[Mapping[str, Any]] = None,
        json_alert_prompt: Optional[str] = None,
        bookmark_enabled: Optional[bool] = None,
        bookmark_cooldown_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        changed = False
        target_channel_id = int(channel_id) if channel_id is not None else None
        with self.cache_lock:
            if target_channel_id is None:
                if stream_system_prompt is not None:
                    next_stream_prompt = str(stream_system_prompt)
                    if next_stream_prompt != str(self.system_prompt or ""):
                        self.system_prompt = next_stream_prompt
                        changed = True
                if json_alert_prompt is not None:
                    next_json_prompt = str(json_alert_prompt)
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
                if json_alert_prompt is not None:
                    next_json_prompt = str(json_alert_prompt)
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

    def _compose_rollup_summary(
        self,
        level: str,
        source_level: str,
        item_count: int,
        frame_count: int,
        run_ids: Sequence[str],
        highlights: Sequence[str],
        window_sec: int,
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
        return base

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
        created_at = self._coerce_float(entry.get("created_at"))
        if created_at is None:
            created_at = time.time()
        return {
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
        }

    def _persist_rollup_cache_locked(self) -> None:
        payload_entries = [
            dict(entry)
            for entry in self.rollup_summary_cache.values()
            if isinstance(entry, Mapping)
        ]
        cache_file = self.rollup_cache_file
        payload = {"version": 1, "updated_at": time.time(), "entries": payload_entries}
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = cache_file.with_suffix(f"{cache_file.suffix}.tmp")
            tmp_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_file.replace(cache_file)
        except Exception:
            # Best-effort persistence should never interrupt the live stream loop.
            return

    def _load_rollup_cache_from_disk(self) -> None:
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
            for entry in normalized_entries[-self.rollup_summary_cache_limit :]:
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
        )

    def _select_rollup_source_lines(
        self,
        children: Sequence[Mapping[str, Any]],
        char_budget: int,
    ) -> List[str]:
        items: List[Tuple[float, str]] = []
        for child in sorted(children, key=lambda item: float(self._coerce_float(item.get("window_start")) or 0.0)):
            ts = self._coerce_float(child.get("window_start"))
            if ts is None:
                continue
            summary = self._sanitize_l0_summary(child.get("summary"), max_len=420)
            if not summary:
                continue
            ts_label = time.strftime("%H:%M:%S", time.localtime(ts))
            items.append((ts, f"- {ts_label} | {summary}"))
        if not items:
            return ["- No valid lower-level summaries in this window."]
        if len(items) == 1:
            return [items[0][1]]
        # First pass: keep everything if it fits.
        joined_len = sum(len(line) + 1 for _, line in items)
        if joined_len <= char_budget:
            return [line for _, line in items]
        # Second pass: even timeline sampling + first/last anchors.
        max_lines = max(8, min(len(items), int(char_budget / 180)))
        if max_lines >= len(items):
            selected_indexes = list(range(len(items)))
        else:
            selected_indexes = {0, len(items) - 1}
            span = len(items) - 1
            for step in range(1, max_lines - 1):
                idx = int(round((step * span) / max(1, max_lines - 1)))
                selected_indexes.add(max(0, min(len(items) - 1, idx)))
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
                "",
                "Context constraints:",
                "- All source entries are from the same channel and continuous timeline window above.",
                "- Source entries may be model-generated summaries from a lower level; avoid compounding uncertainty.",
                "- Preserve rare but important events even if they appear once.",
                "- Keep numeric facts aligned with metadata above (item_count/frame_count/window).",
                "",
                "Task:",
                f"- Write one concise {level} summary for operators.",
                "- Deduplicate repeated scene descriptions and boilerplate.",
                "- Keep meaningful changes in short timeline bullets across the full window.",
                "- Mention risks/signals only when grounded in source text.",
                "- If activity is routine, say so clearly without repeating identical details.",
                "- Do not invent entities, times, or counts.",
                "",
                "Output format (Markdown):",
                "### Window Snapshot",
                "### Scene Baseline",
                "### Key Changes",
                "### Alerts/Signals",
                "### Operator Notes",
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
        return [
            {"role": "system", "content": [{"type": "text", "text": system_msg}]},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ]

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
            summary = str(self.lm_callback(messages, self.rollup_llm_model_hint)).strip()
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
    ) -> str:
        return (
            f"{level} pending from {source_level}: {item_count} items ({frame_count} frames). "
            f"Collecting context {source_tokens}/{min_tokens} tokens."
        )

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
                source_signature = self._source_signature(self._coerce_str_list(node.get("source_ids")))
            if (not self.rollup_time_only) and source_tokens < self.rollup_min_source_tokens:
                node["summary"] = self._compose_pending_rollup_summary(
                    level=level,
                    source_level=source_level,
                    source_tokens=source_tokens,
                    min_tokens=self.rollup_min_source_tokens,
                    item_count=_parse_optional_int(node.get("item_count")) or 0,
                    frame_count=_parse_optional_int(node.get("frame_count")) or 0,
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
                    if level == "L2":
                        self._update_channel_routine_context(
                            channel_id=channel_id,
                            rollup_id=rollup_id,
                            summary_text=node.get("summary"),
                            window_end=self._coerce_float(node.get("window_end")),
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
                    source_signature=str(node.get("source_signature") or "").strip(),
                    highlights=node.get("highlights"),
                    summary_kind="llm",
                )
                node["summary"] = summary
                node["summary_kind"] = "llm"
                if level == "L2":
                    self._update_channel_routine_context(
                        channel_id=channel_id,
                        rollup_id=rollup_id,
                        summary_text=summary,
                        window_end=self._coerce_float(node.get("window_end")),
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
            frame_count = _parse_optional_int(log.get("frame_count")) or 0
            run_id = str(log.get("run_id") or "").strip()
            summary = str(log.get("summary") or "").strip()
            headline = self._summary_headline(summary)
            key = self._summary_log_key(log)
            rollup_id = f"l0-ch{channel_id}-{self._stable_id(key, length=14)}"
            nodes.append(
                {
                    "rollup_id": rollup_id,
                    "channel_id": channel_id,
                    "level": "L0",
                    "source_level": None,
                    "source_ids": [],
                    "window_start": created,
                    "window_end": created,
                    "window_sec": 0,
                    "item_count": 1,
                    "frame_count": int(frame_count),
                    "run_ids": [run_id] if run_id else [],
                    "highlights": [headline] if headline else [],
                    "summary": summary,
                    "created_at": created,
                }
            )
        nodes.sort(key=lambda item: float(item.get("window_start") or 0.0))
        return nodes

    def _build_rollup_level(
        self,
        channel_id: int,
        level: str,
        source_level: str,
        window_sec: int,
        source_nodes: Sequence[Mapping[str, Any]],
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
            source_signature = self._source_signature(source_ids)
            summary = self._compose_rollup_summary(
                level=level,
                source_level=source_level,
                item_count=item_count,
                frame_count=frame_count,
                run_ids=sorted(run_ids),
                highlights=highlights,
                window_sec=window_sec,
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
                }
            )
            if level in self.rollup_llm_levels:
                llm_pairs.append((out[-1], children))
        if level in self.rollup_llm_levels and llm_pairs:
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
        existing = self.summary_history.get(channel_id, [])
        combined = self._combine_summary_logs(existing, logs)
        if len(combined) > self.summary_history_limit:
            combined = combined[-self.summary_history_limit :]
        self.summary_history[channel_id] = combined
        self._persist_summary_state_locked()

    def record_summary_log(self, channel_id: int, entry: Mapping[str, Any]) -> None:
        normalized = self._normalize_summary_log_entry(entry)
        if normalized is None:
            return
        with self.cache_lock:
            self._merge_summary_history_locked(channel_id, [normalized])

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

    @staticmethod
    def _normalize_alert_timestamp_ms(raw_value: Any, fallback_ts_ms: int) -> int:
        try:
            ts_ms = int(raw_value)
        except Exception:
            ts_ms = int(fallback_ts_ms)
        return ts_ms if ts_ms > 0 else int(fallback_ts_ms)

    @staticmethod
    def _bookmark_fingerprint(alert: Mapping[str, Any]) -> str:
        title = str(alert.get("title") or "").strip().lower()
        description = str(alert.get("description") or "").strip().lower()[:180]
        severity = str(alert.get("severity") or "").strip().lower()
        state = str(alert.get("state") or "").strip().lower()
        payload = f"{title}|{description}|{severity}|{state}"
        return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()

    @staticmethod
    def _contains_alerts_json(summary_text: str) -> bool:
        text = str(summary_text or "")
        lowered = text.lower()
        if "```json" in lowered or "alerts_json:" in lowered:
            return True
        if re.search(r'^\s*\{\s*["\']alerts["\']\s*:', text, flags=re.IGNORECASE | re.MULTILINE):
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

    def process_summary_alerts(self, channel_id: int, summary_text: str, default_ts_ms: Optional[int] = None) -> int:
        if not self.alert_parser:
            return 0
        base_ts_ms = self._normalize_alert_timestamp_ms(default_ts_ms, int(time.time() * 1000))
        with self.cache_lock:
            settings = self._get_channel_bookmark_settings_locked(channel_id)
        if not bool(settings.get("bookmark_enabled")):
            return 0
        cooldown_sec = float(settings.get("bookmark_cooldown_sec") or 0.0)
        if not self._contains_alerts_json(summary_text):
            return 0
        parsed_alerts = self.alert_parser(summary_text, int(channel_id), base_ts_ms)
        if not isinstance(parsed_alerts, list) or not parsed_alerts:
            return 0

        sent_count = 0
        for raw_alert in parsed_alerts:
            if sent_count >= 3:
                break
            if not isinstance(raw_alert, Mapping):
                continue
            alert = {
                "title": str(raw_alert.get("title") or "Event"),
                "description": str(raw_alert.get("description") or ""),
                "severity": str(raw_alert.get("severity") or "normal"),
                "state": str(raw_alert.get("state") or "new"),
                "channel_id": int(channel_id),  # force observed stream channel
                "timestamp_ms": self._normalize_alert_timestamp_ms(raw_alert.get("timestamp_ms"), base_ts_ms),
            }
            fingerprint = self._bookmark_fingerprint(alert)
            now_ms = int(time.time() * 1000)
            with self.cache_lock:
                if self._bookmark_recently_sent_locked(int(channel_id), fingerprint, now_ms, cooldown_sec):
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
            except Exception:
                continue
            with self.cache_lock:
                self._mark_bookmark_sent_locked(int(channel_id), fingerprint, now_ms)
            sent_count += 1
        return sent_count

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
                    self.channel_prompt_overrides[channel_id] = channel_overrides
                    self._persist_summary_state_locked()
            effective_system_prompt = self._get_stream_system_prompt_locked(channel_id)
            run = self._open_run_locked(
                channel_id=channel_id,
                batch_size=batch,
                prompt=prompt,
                model_hint=model_hint,
                system_prompt=effective_system_prompt,
            )
            session = LuxriotCaptureSession(
                self,
                channel_id,
                batch,
                prompt,
                run_id=run.get("run_id"),
                run_started_at=run.get("started_at"),
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
            status = session.status()
            logs = status.get("logs")
            session.stop()
            with self.cache_lock:
                if isinstance(logs, list):
                    self._merge_summary_history_locked(channel_id, logs)
                self._close_run_locked(channel_id, status.get("run_id"))
                archived_count = len(self.summary_history.get(channel_id, []))
            return {
                "channel_id": channel_id,
                "run_id": status.get("run_id"),
                "running": False,
                "archived_log_count": archived_count,
            }
        with self.cache_lock:
            self._close_run_locked(channel_id, None)
            archived_count = len(self.summary_history.get(channel_id, []))
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

    def probe_frame_thumbnail(self, channel_id: int, timestamp_ms: Optional[int] = None) -> Optional[str]:
        with self.cache_lock:
            session = self.probe_sessions.get(channel_id)
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
        )
        l2_nodes = self._build_rollup_level(
            channel_id=channel_id,
            level="L2",
            source_level="L1",
            window_sec=self.rollup_windows["L2"],
            source_nodes=l1_nodes,
        )
        l3_nodes = self._build_rollup_level(
            channel_id=channel_id,
            level="L3",
            source_level="L2",
            window_sec=self.rollup_windows["L3"],
            source_nodes=l2_nodes,
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
        self._refresh_channel_routine_from_l2(channel_id, l2_nodes)
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
            history_channels = sorted(channel_id for channel_id, logs in self.summary_history.items() if logs)
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
            "video_history_channels": history_channels,
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
