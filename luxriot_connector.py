import base64
import copy
import hashlib
import json
import logging
import math
import os
import queue
import subprocess
import tempfile
import re
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Set, Tuple, cast
from uuid import uuid4

import requests
from PIL import Image, ImageChops, ImageFilter, ImageStat
from requests.auth import HTTPDigestAuth
try:
    from road_events import (
        AutoSceneCardConfig,
        DecodedVideoFrame,
        RoadEventCue,
        RoadEpisodeAggregator,
        RoadEpisodeAggregatorConfig,
        RoadMotionAnalyzer,
        RoadSceneCard,
        RoadZone,
        SceneCalibrationConfig,
        calibrate_scene_card_from_results,
        infer_scene_card_from_frames,
        iter_luxriot_live_segment_frames,
    )
except Exception:  # pragma: no cover - road CV is optional in minimal installs
    AutoSceneCardConfig = None  # type: ignore[assignment]
    DecodedVideoFrame = None  # type: ignore[assignment]
    RoadEventCue = None  # type: ignore[assignment]
    RoadEpisodeAggregator = None  # type: ignore[assignment]
    RoadEpisodeAggregatorConfig = None  # type: ignore[assignment]
    RoadMotionAnalyzer = None  # type: ignore[assignment]
    RoadSceneCard = None  # type: ignore[assignment]
    RoadZone = None  # type: ignore[assignment]
    SceneCalibrationConfig = None  # type: ignore[assignment]
    calibrate_scene_card_from_results = None  # type: ignore[assignment]
    infer_scene_card_from_frames = None  # type: ignore[assignment]
    iter_luxriot_live_segment_frames = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)
ROLLUP_OPERATOR_FORMAT_VERSION = 2


class SummaryBatchSuperseded(RuntimeError):
    """A queued summary belongs to a stopped or replaced channel generation."""

    superseded = True

    def __init__(self, reason: str) -> None:
        self.reason = str(reason or "summary batch was superseded").strip()
        super().__init__(self.reason)

_ERROR_URL_USERINFO_RE = re.compile(
    r"(?P<scheme>\b[a-z][a-z0-9+.-]*://)[^\s/@]+@",
    re.IGNORECASE,
)
_ERROR_QUERY_SECRET_RE = re.compile(
    r"(?P<prefix>[?&](?:password|passwd|pwd|token|access_token|api[_-]?key)=)[^&\s\"'<>]+",
    re.IGNORECASE,
)
_ERROR_ASSIGNMENT_SECRET_RE = re.compile(
    r"(?P<prefix>\b(?:password|passwd|pwd|token|access_token|api[_-]?key)\s*[:=]\s*)"
    r"(?P<quote>[\"']?)[^\s,;\"']+(?P=quote)",
    re.IGNORECASE,
)
_ERROR_AUTH_HEADER_RE = re.compile(
    r"(?P<prefix>\bauthorization\s*[:=]\s*(?:basic|bearer)\s+)[^\s,;]+",
    re.IGNORECASE,
)


def _safe_error_text(value: object, max_len: int = 500) -> str:
    """Return a bounded diagnostic string with URL/header credentials removed."""

    text = str(value or "").strip()
    if not text:
        return ""
    text = _ERROR_URL_USERINFO_RE.sub(r"\g<scheme><redacted>@", text)
    text = _ERROR_QUERY_SECRET_RE.sub(r"\g<prefix><redacted>", text)
    text = _ERROR_ASSIGNMENT_SECRET_RE.sub(r"\g<prefix><redacted>", text)
    text = _ERROR_AUTH_HEADER_RE.sub(r"\g<prefix><redacted>", text)
    return text[: max(1, int(max_len))]


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
    "current state and let backend continuity tools compare adjacent batches.\n"
    "- Never assert intent or skill: words like 'intentional', 'controlled', 'stunt', 'deliberate', 'showing off' "
    "are conclusions cameras cannot prove. Describe the visible dynamics (speed, trajectory, smoke, proximity to "
    "people/objects) and let severity reflect the visible risk.\n"
    "- Never declare 'no safety hazard' or 'no danger': absence of visible harm in sampled snapshots is not proof "
    "of safety. Say what is visible and what remains uncertain.\n"
    "- A vector/attention cue never confirms an event by itself; only current snapshots confirm. If the images do "
    "not support the cue, say the cue is visually unconfirmed."
)

VECTOR_SIGNAL_PROMPT_PREFIX = (
    "Current vector/homeostasis signal contract:\n"
    "- VECTOR_SIGNALS_JSON is a secondary attention/arousal signal from CLIP probes and lightweight CV, not visual proof.\n"
    "- Use it to decide which current snapshots deserve extra scrutiny; verify any candidate directly in the current images.\n"
    "- capture_attention marks snapshots whose motion is far above this channel's measured norm (activity_x = times above "
    "typical). Motion blur on burst snapshots is expected physics of fast events - describe the action itself; use sharper "
    "neighboring snapshots (or a provided sharper companion frame) for identity details.\n"
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

# Per-second CV apex decider (policy v2). Fast events produce only motion-blurred
# frames, so sharpness may only steer the choice when it actually discriminates
# inside the bucket; burst seconds are judged against the channel's own measured
# activity baseline, never an absolute threshold.
CAPTURE_APEX_POLICY = "capture_per_second_cv_apex_v2"
CAPTURE_SELECTOR_BIASES = ("auto", "action", "clarity")
_CAPTURE_BASELINE_ALPHA = 0.005          # EMA weight per finalized second (~3 min horizon)
_CAPTURE_BASELINE_WARMUP_BUCKETS = 90    # seconds of history before burst mode is trusted
_CAPTURE_NORMAL_ACTIVITY_BAND = 0.65     # normal mode picks sharpest among frames >= band * bucket peak
_CAPTURE_SHARPNESS_DISCRIMINATION = 1.15 # in-bucket sharpness spread required to influence the choice
_CAPTURE_COMPANION_ACTIVITY_BAND = 0.5   # companion must still belong to the action
_CAPTURE_COMPANION_SHARPNESS_GAIN = 1.3  # companion must be meaningfully sharper than the apex

# LM backpressure: instead of silently dropping the oldest queued 12-second
# window, adjacent windows are coalesced into one wider batch (graceful loss
# of temporal resolution). One merged batch may span at most this many
# original batches; beyond that the oldest window is dropped WITH an explicit
# coverage-gap history entry.
_SUMMARY_COALESCE_MAX_BATCHES = 4


class ProbeManagerLike(Protocol):
    def add_frame(
        self,
        channel_id: int,
        pil_image: Image.Image,
        timestamp_ms: Optional[int],
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> Any: ...
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
SummaryArchiveHistoryLoaderFn = Callable[[int, float, float], Tuple[List[Dict[str, Any]], int]]
SummaryArchiveBucketLoaderFn = Callable[[int, float, float, int], List[Dict[str, Any]]]


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
        obj.last_error = _safe_error_text(last_error, 240) or None
        obj.alerts_detected = bool(alerts_detected)
        obj.parser_error = _safe_error_text(parser_error, 240) or None
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
                error = _safe_error_text(raw_event.get("error"), 240)
                if error:
                    event["error"] = error
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

    CHANNEL_STREAM_SETTLE_SEC = 0.20
    CHANNEL_STREAM_MAX_PAYLOADS = 256

    def __init__(self, base_url: str, username: str, password: str, timeout: int = 15) -> None:
        if not base_url:
            raise ValueError("Luxriot base URL is not configured.")
        self.base_url = base_url.rstrip("/")
        self.username = username or ""
        self.password = password or ""
        self.session = requests.Session()
        self.session.auth = HTTPDigestAuth(self.username, self.password)
        self.timeout = timeout
        self.channel_inventory_meta: Dict[str, Any] = {
            "complete": None,
            "completion": "not_loaded",
            "payload_count": 0,
        }

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
            safe_url = _safe_error_text(url, 1000)
            safe_error = _safe_error_text(exc, 1000)
            raise RuntimeError(f"Luxriot request failed ({safe_url}): {safe_error}") from exc

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

    @staticmethod
    def _iter_json_documents(lines: Iterable[str], max_chunks: int = 5_000_000) -> Iterable[Any]:
        """Decode newline/SSE or concatenated JSON documents without dropping trailing data."""

        decoder = json.JSONDecoder()
        buffer = ""
        for idx, raw_line in enumerate(lines):
            if max_chunks and idx >= max_chunks:
                raise RuntimeError("Luxriot /channels resource stream exceeded its initial-state chunk limit.")
            if raw_line is None:
                continue
            if isinstance(raw_line, bytes):
                line = raw_line.decode("utf-8", errors="replace")
            else:
                line = str(raw_line)
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(":") or stripped.startswith(("event:", "id:", "retry:")):
                continue
            if stripped.startswith("data:"):
                stripped = stripped[5:].strip()
            if not stripped or stripped == "[DONE]":
                continue
            buffer += stripped
            if len(buffer) > 5_000_000:
                raise RuntimeError("Luxriot /channels initial-state payload exceeded 5 MB.")
            while buffer:
                candidate = buffer.lstrip()
                try:
                    payload, end = decoder.raw_decode(candidate)
                except json.JSONDecodeError:
                    break
                yield payload
                buffer = candidate[end:]
        if buffer.strip():
            raise RuntimeError("Luxriot /channels resource stream ended with incomplete JSON.")

    @staticmethod
    def _channel_stream_flags(payload: Any) -> Tuple[bool, bool]:
        """Return (explicitly_incomplete, explicitly_complete) for known stream markers."""

        if not isinstance(payload, Mapping):
            return False, False
        candidates: List[Mapping[str, Any]] = [cast(Mapping[str, Any], payload)]
        for key in ("meta", "metadata", "stream"):
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                candidates.append(cast(Mapping[str, Any], nested))
        incomplete = False
        complete = False
        for item in candidates:
            for key in ("hasMore", "has_more"):
                if key in item:
                    if bool(item.get(key)):
                        incomplete = True
                    else:
                        complete = True
            for key in ("partial", "isPartial", "is_partial"):
                if key in item and bool(item.get(key)):
                    incomplete = True
            for key in (
                "initialStateComplete",
                "initial_state_complete",
                "isInitialStateComplete",
                "endOfInitialData",
                "end_of_initial_data",
            ):
                if key in item:
                    if bool(item.get(key)):
                        complete = True
                    else:
                        incomplete = True
            phase = str(item.get("phase") or item.get("type") or item.get("event") or "").strip().lower()
            if phase in {"initial", "initial_state", "initial-state", "snapshot_part"}:
                incomplete = True
            elif phase in {
                "initial_complete",
                "initial-state-complete",
                "initial_state_complete",
                "snapshot_complete",
                "ready",
            }:
                complete = True
        if complete:
            incomplete = False
        return incomplete, complete

    @staticmethod
    def _channel_payload_is_full_snapshot(payload: Any) -> bool:
        if isinstance(payload, list):
            return True
        if not isinstance(payload, Mapping):
            return False
        if isinstance(payload.get("channels"), list):
            return True
        data = payload.get("data")
        return bool(
            isinstance(data, Mapping)
            and isinstance(data.get("channels"), list)
            and not any(key in data for key in ("added", "updated", "removed"))
        )

    def _collect_channel_resource_payloads(self, resp: requests.Response) -> Tuple[List[Any], Dict[str, Any]]:
        """Collect the bounded initial burst of a long-lived Luxriot resource stream."""

        messages: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        stop_reader = threading.Event()

        def read_stream() -> None:
            try:
                raw_read1 = getattr(getattr(resp, "raw", None), "read1", None)
                if callable(raw_read1):
                    def raw_chunks() -> Iterable[bytes]:
                        while not stop_reader.is_set():
                            chunk = raw_read1(65_536)
                            if not chunk:
                                return
                            yield chunk

                    chunks: Iterable[Any] = raw_chunks()
                else:
                    try:
                        chunks = resp.iter_content(chunk_size=1, decode_unicode=True)
                    except TypeError:  # compatibility with minimal response fakes/adapters
                        chunks = resp.iter_lines(decode_unicode=True)
                for payload in self._iter_json_documents(chunks):
                    messages.put(("payload", payload))
                    _, complete = self._channel_stream_flags(payload)
                    if complete or self._channel_payload_is_full_snapshot(payload):
                        break
                    if stop_reader.is_set():
                        break
                messages.put(("eof", None))
            except Exception as exc:
                messages.put(("error", exc))
            finally:
                try:
                    resp.close()
                except Exception:
                    pass

        reader = threading.Thread(
            target=read_stream,
            name="luxriot-channel-inventory",
            daemon=True,
        )
        reader.start()
        payloads: List[Any] = []
        explicitly_incomplete = False
        explicitly_complete = False
        completion = "unknown"
        try:
            while len(payloads) < self.CHANNEL_STREAM_MAX_PAYLOADS:
                wait_sec = max(1.0, float(self.timeout)) if not payloads else self.CHANNEL_STREAM_SETTLE_SEC
                try:
                    kind, value = messages.get(timeout=wait_sec)
                except queue.Empty:
                    if not payloads:
                        raise RuntimeError("Luxriot /channels returned no initial-state data before timeout.")
                    if explicitly_incomplete and not explicitly_complete:
                        raise RuntimeError("Luxriot /channels initial state remained incomplete.")
                    completion = "settled"
                    break
                if kind == "error":
                    raise RuntimeError(
                        f"Luxriot /channels resource stream failed: {_safe_error_text(value, 500)}"
                    ) from cast(Exception, value)
                if kind == "eof":
                    if explicitly_incomplete and not explicitly_complete:
                        raise RuntimeError("Luxriot /channels resource stream ended before initial state completed.")
                    completion = "eof"
                    break
                payloads.append(value)
                incomplete, complete = self._channel_stream_flags(value)
                explicitly_incomplete = explicitly_incomplete or incomplete
                explicitly_complete = explicitly_complete or complete
                if complete:
                    completion = "explicit"
                    break
                if self._channel_payload_is_full_snapshot(value):
                    completion = "snapshot"
                    break
            else:
                raise RuntimeError("Luxriot /channels resource stream exceeded its initial-state payload limit.")
        finally:
            stop_reader.set()
            reader.join(timeout=0.02)
        return payloads, {
            "complete": True if completion in {"explicit", "snapshot", "eof"} else None,
            "completion": completion,
            "payload_count": len(payloads),
        }

    @staticmethod
    def _channel_resource_sections(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        data = payload.get("data")
        if isinstance(data, Mapping) and any(key in data for key in ("added", "updated", "removed")):
            return cast(Mapping[str, Any], data)
        return payload

    @classmethod
    def _apply_channel_resource_payload(
        cls,
        payload: Any,
        state: Dict[str, Dict[str, Any]],
        order: List[str],
    ) -> bool:
        """Apply full snapshots and added/updated/removed resource-stream deltas."""

        recognized = False

        def row_key(row: Mapping[str, Any], *, anonymous_index: int) -> str:
            channel_id = _parse_optional_int(row.get("id"))
            return str(channel_id) if channel_id is not None else f"anonymous:{anonymous_index}"

        def replace_snapshot(rows: Sequence[Any]) -> None:
            state.clear()
            order.clear()
            for index, raw in enumerate(rows):
                if not isinstance(raw, Mapping):
                    continue
                key = row_key(cast(Mapping[str, Any], raw), anonymous_index=index)
                state[key] = dict(raw)
                order.append(key)

        def apply_rows(rows: Any, *, update: bool) -> None:
            if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
                return
            for index, raw in enumerate(rows):
                if not isinstance(raw, Mapping):
                    continue
                key = row_key(cast(Mapping[str, Any], raw), anonymous_index=len(order) + index)
                existing = state.get(key) if update else None
                merged = dict(existing or {})
                merged.update(dict(raw))
                state[key] = merged
                if key not in order:
                    order.append(key)

        def remove_rows(rows: Any) -> None:
            if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
                return
            for raw in rows:
                channel_id = _parse_optional_int(raw.get("id")) if isinstance(raw, Mapping) else _parse_optional_int(raw)
                if channel_id is None:
                    continue
                key = str(channel_id)
                state.pop(key, None)
                if key in order:
                    order.remove(key)

        if isinstance(payload, list):
            replace_snapshot(payload)
            return True
        if not isinstance(payload, Mapping):
            return False
        if isinstance(payload.get("channels"), list):
            replace_snapshot(cast(Sequence[Any], payload.get("channels")))
            recognized = True
        data = payload.get("data")
        if (
            isinstance(data, Mapping)
            and isinstance(data.get("channels"), list)
            and not any(key in data for key in ("added", "updated", "removed"))
        ):
            replace_snapshot(cast(Sequence[Any], data.get("channels")))
            recognized = True
        sections = cls._channel_resource_sections(cast(Mapping[str, Any], payload))
        for section_name, updater in (("added", False), ("updated", True)):
            section = sections.get(section_name)
            if not isinstance(section, Mapping) or not isinstance(section.get("channels"), list):
                continue
            apply_rows(section.get("channels"), update=updater)
            recognized = True
        removed = sections.get("removed")
        if isinstance(removed, Mapping) and isinstance(removed.get("channels"), list):
            remove_rows(removed.get("channels"))
            recognized = True
        return recognized

    def get_channels(self) -> List[Dict[str, Any]]:
        resp = self._request(
            "GET",
            "/channels",
            params={"health": 0},
            headers={"Accept": "application/json"},
            timeout=(self.timeout, max(2.0, self.CHANNEL_STREAM_SETTLE_SEC * 5.0)),
            stream=True,
        )
        try:
            payloads, stream_meta = self._collect_channel_resource_payloads(resp)
            state: Dict[str, Dict[str, Any]] = {}
            order: List[str] = []
            recognized = False
            for payload in payloads:
                recognized = self._apply_channel_resource_payload(payload, state, order) or recognized
            if not recognized:
                raise RuntimeError("Unexpected Luxriot /channels resource-stream payload.")
            channels = [state[key] for key in order if key in state]
            self.channel_inventory_meta = {
                **stream_meta,
                "channel_count": len(channels),
                "error": None,
            }
        except Exception as exc:
            self.channel_inventory_meta = {
                "complete": False,
                "completion": "error",
                "payload_count": 0,
                "channel_count": 0,
                "error": _safe_error_text(exc, 500),
            }
            raise
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

    def get_snapshot(
        self,
        channel_id: int,
        stream: str = "mainStream",
        *,
        timeout: Optional[float] = None,
    ) -> Image.Image:
        request_timeout = float(timeout) if timeout is not None else max(10, self.timeout)
        resp = self._request(
            "GET",
            f"/live/{channel_id}/snapshot",
            params={"stream": stream},
            headers={"Accept": "image/jpeg"},
            stream=False,
            timeout=max(1.0, request_timeout),
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
        normalized_stream = str(stream_type or "mainStream").strip().lower()
        snapshot_type = {
            "mainstream": "video1",
            "main": "video1",
            "video1": "video1",
            "substream": "video2",
            "sub": "video2",
            "video2": "video2",
            "edgestream": "video3",
            "edge": "video3",
            "video3": "video3",
        }.get(normalized_stream, "video1")
        try:
            resp = self._request(
                "GET",
                f"/archive/{channel_id}/snapshot",
                params={"time": int(time_ms), "type": snapshot_type},
                headers={"Accept": "image/jpeg"},
                stream=False,
                timeout=max(10, self.timeout),
            )
        except Exception:
            # Evo variants predating the documented video1/video2 selector used
            # streamType. Keep the compatibility retry local to archive JPEGs.
            resp = self._request(
                "GET",
                f"/archive/{channel_id}/snapshot",
                params={"time": int(time_ms), "streamType": stream_type},
                headers={"Accept": "image/jpeg"},
                stream=False,
                timeout=max(10, self.timeout),
            )
        try:
            return self._decode_jpeg_response(
                resp,
                label=f"Luxriot archive snapshot for channel {channel_id}",
            )
        finally:
            try:
                resp.close()
            except Exception:
                pass

    def open_live_stream(
        self,
        channel_id: int,
        *,
        stream: str = "mainStream",
        timeout: Optional[Any] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> requests.Response:
        """Open live media through Evo's short-lived stream-token transport.

        The token avoids a Digest challenge on the long-running media response.
        It remains server-side and is redacted by the generic request-error
        sanitizer. Older Evo builds transparently fall back to the legacy direct
        live path.
        """

        live_timeout = timeout or max(30, self.timeout)
        request_headers = {
            "Accept": "video/mp4,video/webm,multipart/x-mixed-replace,application/octet-stream,*/*",
            **dict(headers or {}),
        }
        if isinstance(live_timeout, (tuple, list)) and len(live_timeout) >= 2:
            token_setup_timeout: Any = (
                max(0.25, float(live_timeout[0])),
                max(1.0, min(10.0, float(live_timeout[1]))),
            )
        else:
            token_setup_timeout = max(1, min(10, int(live_timeout)))
        token = uuid4().hex
        token_failure: Optional[Exception] = None
        add_response: Optional[requests.Response] = None
        try:
            add_response = self._request(
                "GET",
                f"/live/{channel_id}/addStreamToken",
                params={"token": token, "stream": stream},
                headers={"Accept": "*/*", "Accept-Encoding": "identity"},
                stream=False,
                timeout=token_setup_timeout,
            )
            add_response.close()
            add_response = None
            response = self._request(
                "GET",
                "/retrieveLiveStreamByToken",
                params={"token": token},
                headers=request_headers,
                stream=True,
                timeout=live_timeout,
            )
            setattr(response, "_eva_live_transport", "token")
            return response
        except Exception as exc:
            token_failure = exc
        finally:
            if add_response is not None:
                try:
                    add_response.close()
                except Exception:
                    pass

        response = self._request(
            "GET",
            f"/live/{channel_id}/{stream}",
            headers=request_headers,
            stream=True,
            timeout=live_timeout,
        )
        setattr(response, "_eva_live_transport", "digest_direct_fallback")
        if token_failure is not None:
            setattr(response, "_eva_live_transport_fallback_reason", type(token_failure).__name__)
        return response

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
        session_generation: Optional[str] = None,
    ) -> None:
        self.manager = manager
        self.channel_id = channel_id
        self.batch_size = batch_size
        self.prompt = prompt
        self.run_id = str(run_id or "").strip()
        self.run_started_at = float(run_started_at) if run_started_at else time.time()
        self.model_hint = model_hint
        self.session_generation = str(session_generation or "").strip() or None
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
        self.summary_queue_max_batches = max(
            1,
            min(12, int(getattr(manager.config, "LUXRIOT_SUMMARY_QUEUE_MAX_BATCHES", 2) or 2)),
        )
        self.summary_condition = threading.Condition(self.lock)
        self.summary_queue: List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]] = []
        self.summary_coalesced_batches = 0
        self.summary_inflight = False
        self.summary_worker_thread = threading.Thread(target=self._summary_worker, daemon=True)
        self.last_error: Optional[str] = None
        self.capture_last_error: Optional[str] = None
        self.probe_last_error: Optional[str] = None
        self.summary_last_error: Optional[str] = None
        self.summary_failed_batches = 0
        self.summary_last_failure_at: Optional[float] = None
        self.summary_last_success_at: Optional[float] = None
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
        self.last_live_segment_target_seconds = 0.0
        self.last_live_segment_summary_target_seconds = 0.0
        self.last_live_segment_raw_frame_budget = 0
        self.last_live_segment_byte_budget = 0
        self.last_live_segment_streamed_bytes = 0
        self.last_live_segment_represented_seconds = 0.0
        self.last_live_segment_completed_at: Optional[float] = None
        self.last_live_segment_source_start_timestamp_ms: Optional[int] = None
        self.last_live_segment_last_source_timestamp_ms: Optional[int] = None
        self.last_live_segment_timestamp_source: Optional[str] = None
        self.live_segment_inflight = False
        self.live_segment_capture_started_at: Optional[float] = None
        self.live_segment_inflight_target_seconds = 0.0
        self.live_segment_inflight_raw_frame_budget = 0
        self.live_segment_inflight_frames = 0
        self.live_segment_inflight_represented_seconds = 0.0
        self._last_live_source_timestamp_ms: Optional[int] = None
        self.live_segment_backoff_until = 0.0
        self._last_frame_hash: Optional[str] = None
        self._same_frame_started_at: Optional[float] = None
        self._same_frame_count = 0
        self.frozen_signal = False
        self.frozen_signal_since: Optional[float] = None
        self.frozen_frame_count = 0
        self.frozen_frame_hash: Optional[str] = None
        self.frozen_frame_dropped_count = 0
        self._capture_source_sequence = 0
        self._capture_apex_bucket_start_ms: Optional[int] = None
        self._capture_apex_bucket: List[Dict[str, Any]] = []
        self._capture_cv_previous_gray: Optional[Image.Image] = None
        self.capture_apex_raw_frame_count = 0
        self.capture_apex_selected_count = 0
        self.capture_apex_fallback_count = 0
        self.capture_apex_probe_dispatch_count = 0
        self.capture_apex_probe_failure_count = 0
        self.capture_apex_probe_skipped_count = 0
        self.capture_apex_selection_sources: Dict[str, int] = {}
        self.capture_apex_last_selection: Dict[str, Any] = {}
        self.capture_selector_bias = "auto"
        self.capture_activity_baseline_level: Optional[float] = None
        self.capture_activity_baseline_dev = 0.0
        self.capture_activity_baseline_buckets = 0
        self.capture_apex_mode_counts: Dict[str, int] = {}
        self.capture_apex_companion_count = 0
        baseline_getter = getattr(manager, "get_persisted_capture_baseline", None)
        if callable(baseline_getter):
            try:
                persisted_baseline = baseline_getter(channel_id)
            except Exception:
                persisted_baseline = None
            if isinstance(persisted_baseline, Mapping):
                persisted_level = manager._finite_float(persisted_baseline.get("level"))
                if persisted_level is not None and float(persisted_level) >= 0.0:
                    self.capture_activity_baseline_level = float(persisted_level)
                    self.capture_activity_baseline_dev = max(
                        0.0,
                        float(manager._finite_float(persisted_baseline.get("dev")) or 0.0),
                    )
                    self.capture_activity_baseline_buckets = max(
                        0,
                        int(_parse_optional_int(persisted_baseline.get("buckets")) or 0),
                    )

    def _refresh_last_error_locked(self) -> None:
        self.last_error = self.summary_last_error or self.capture_last_error or self.probe_last_error

    def _set_capture_error(self, error: object) -> None:
        with self.lock:
            self.capture_last_error = _safe_error_text(error, 500) or error.__class__.__name__
            self._refresh_last_error_locked()

    def _clear_capture_error(self) -> None:
        with self.lock:
            self.capture_last_error = None
            self._refresh_last_error_locked()

    def _set_probe_error(self, error: object) -> None:
        with self.lock:
            self.probe_last_error = _safe_error_text(error, 500) or error.__class__.__name__
            self._refresh_last_error_locked()

    def _clear_probe_error(self) -> None:
        with self.lock:
            self.probe_last_error = None
            self._refresh_last_error_locked()

    def _record_summary_failure_locked(
        self,
        error: object,
        *,
        dropped_frames: int = 0,
        increment_dropped_batch: bool = True,
    ) -> None:
        message = _safe_error_text(error, 500) or error.__class__.__name__
        self.summary_last_error = message
        self.summary_last_failure_at = time.time()
        self.summary_failed_batches += 1
        if increment_dropped_batch:
            self.queue_dropped_batches += 1
        if dropped_frames > 0:
            self.dropped_frames += int(dropped_frames)
        self._refresh_last_error_locked()

    def _mark_summary_success_locked(self) -> None:
        self.summary_last_error = None
        self.summary_last_success_at = time.time()
        self._refresh_last_error_locked()

    def start(self) -> None:
        if self.summarization_enabled and not self.summary_worker_thread.is_alive():
            self.summary_worker_thread.start()
        if not self.thread.is_alive():
            self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        with self.summary_condition:
            self.summary_condition.notify_all()
        if self.thread.is_alive():
            self.thread.join(timeout=0.75)
        self._flush_capture_apex_bucket()
        if self.summary_worker_thread.is_alive():
            self.summary_worker_thread.join(timeout=0.75)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            loop_started = time.monotonic()
            try:
                if self._should_use_live_segment():
                    handled = self._run_live_segment_once()
                    if (
                        not handled
                        and self.capture_source_mode == "auto"
                        and not self._snapshot_unavailable_in_auto()
                    ):
                        self._run_snapshot_once()
                else:
                    try:
                        self._run_snapshot_once()
                    except Exception as exc:
                        if self.capture_source_mode != "auto":
                            raise
                        self._set_capture_error(exc)
                        self._record_snapshot_result(
                            max(0.0, time.monotonic() - loop_started),
                            success=False,
                        )
                        handled = self._run_live_segment_once()
                        if handled:
                            self._clear_capture_error()
            except Exception as exc:
                self._set_capture_error(exc)
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
            snapshot_count = int(self.snapshot_count)
            snapshot_failed_count = int(self.snapshot_failed_count)
            last_latency = self.last_snapshot_latency_sec
            threshold = float(self.snapshot_slow_threshold_sec)
        return (snapshot_count <= 0 and snapshot_failed_count > 0) or slow_count > 0 or (
            last_latency is not None
            and threshold > 0
            and float(last_latency) >= threshold
        )

    def _snapshot_unavailable_in_auto(self) -> bool:
        if self.capture_source_mode != "auto":
            return False
        with self.lock:
            return int(self.snapshot_count) <= 0 and int(self.snapshot_failed_count) > 0

    def _run_snapshot_once(self) -> None:
        snapshot_started = time.monotonic()
        capture_timeout = float(getattr(self.manager.config, "LUXRIOT_CAPTURE_REQUEST_TIMEOUT_SEC", 5.0))
        snapshot = self.client.get_snapshot(self.channel_id, timeout=capture_timeout)
        snapshot_latency = max(0.0, time.monotonic() - snapshot_started)
        self._record_snapshot_result(snapshot_latency, success=True)
        self._clear_capture_error()
        self.active_capture_source = "snapshot"
        self._accept_captured_frame(snapshot, int(time.time() * 1000))

    def _live_segment_capture_budget(self) -> Dict[str, Any]:
        """Return bounded dense-capture budgets for one analytics batch window."""

        fps = max(
            0.2,
            min(10.0, float(getattr(self.manager.config, "LUXRIOT_LIVE_SEGMENT_FPS", 2.0))),
        )
        configured_seconds = max(
            2.0,
            min(60.0, float(getattr(self.manager.config, "LUXRIOT_LIVE_SEGMENT_SECONDS", 15.0))),
        )
        summary_target_seconds = max(
            1.0,
            min(60.0, float(self.batch_size) * max(0.2, float(self.interval))),
        )
        stream_seconds = configured_seconds
        raw_frame_budget = min(
            600,
            max(1, int(math.ceil(stream_seconds * fps))),
        )
        configured_bytes = int(
            max(0.25, float(getattr(self.manager.config, "LUXRIOT_LIVE_SEGMENT_MB", 8.0)))
            * 1024
            * 1024
        )
        # The response is piped rather than retained, so allow a conservative
        # 32 Mbps transport envelope for the represented window.  The old fixed
        # 8 MiB download cap truncated high-bitrate channels long before enough
        # seconds existed for a complete per-second apex batch.
        transport_bytes = int(math.ceil(stream_seconds * 4.0 * 1024 * 1024))
        byte_budget = min(256 * 1024 * 1024, max(configured_bytes, transport_bytes))
        return {
            "fps": float(fps),
            "represented_seconds": float(stream_seconds),
            "stream_seconds": float(stream_seconds),
            "summary_target_seconds": float(summary_target_seconds),
            "raw_frame_budget": int(raw_frame_budget),
            "byte_budget": int(byte_budget),
        }

    def _mark_live_segment_inflight(self, budget: Mapping[str, Any]) -> None:
        with self.lock:
            self.active_capture_source = "live_segment"
            self.live_segment_inflight = True
            self.live_segment_capture_started_at = time.time()
            self.live_segment_inflight_target_seconds = float(
                budget.get("represented_seconds") or 0.0
            )
            self.live_segment_inflight_raw_frame_budget = int(
                budget.get("raw_frame_budget") or 0
            )
            self.live_segment_inflight_frames = 0
            self.live_segment_inflight_represented_seconds = 0.0

    def _complete_live_segment_budget(
        self,
        budget: Mapping[str, Any],
        *,
        streamed_bytes: int,
        represented_seconds: float,
        source_start_timestamp_ms: Optional[int] = None,
        last_source_timestamp_ms: Optional[int] = None,
        timestamp_source: Optional[str] = None,
    ) -> None:
        with self.lock:
            self.last_live_segment_target_seconds = float(
                budget.get("stream_seconds") or budget.get("represented_seconds") or 0.0
            )
            self.last_live_segment_summary_target_seconds = float(
                budget.get("summary_target_seconds") or 0.0
            )
            self.last_live_segment_raw_frame_budget = int(
                budget.get("raw_frame_budget") or 0
            )
            self.last_live_segment_byte_budget = int(budget.get("byte_budget") or 0)
            self.last_live_segment_streamed_bytes = max(0, int(streamed_bytes))
            self.last_live_segment_represented_seconds = round(
                max(0.0, float(represented_seconds)),
                3,
            )
            self.last_live_segment_completed_at = time.time()
            self.last_live_segment_source_start_timestamp_ms = (
                int(source_start_timestamp_ms)
                if source_start_timestamp_ms is not None
                else None
            )
            self.last_live_segment_last_source_timestamp_ms = (
                int(last_source_timestamp_ms)
                if last_source_timestamp_ms is not None
                else None
            )
            self.last_live_segment_timestamp_source = (
                str(timestamp_source or "").strip() or None
            )
            self.live_segment_inflight = False
            self.live_segment_capture_started_at = None
            self.live_segment_inflight_target_seconds = 0.0
            self.live_segment_inflight_raw_frame_budget = 0
            self.live_segment_inflight_frames = 0
            self.live_segment_inflight_represented_seconds = 0.0

    def _next_live_source_timestamp_ms(
        self,
        *,
        source_anchor_ms: int,
        frame_index: int,
        fps: float,
    ) -> int:
        cadence_ms = max(1, int(round(1000.0 / max(0.2, float(fps)))))
        candidate = int(source_anchor_ms) + int(round(float(frame_index) * 1000.0 / float(fps)))
        with self.lock:
            previous = self._last_live_source_timestamp_ms
            if previous is not None and candidate <= int(previous):
                candidate = int(previous) + cadence_ms
            self._last_live_source_timestamp_ms = int(candidate)
        return int(candidate)

    def _cancel_live_segment_inflight(self) -> None:
        with self.lock:
            self.live_segment_inflight = False
            self.live_segment_capture_started_at = None
            self.live_segment_inflight_target_seconds = 0.0
            self.live_segment_inflight_raw_frame_budget = 0
            self.live_segment_inflight_frames = 0
            self.live_segment_inflight_represented_seconds = 0.0

    def _run_live_segment_once(self) -> bool:
        if self._live_segment_backoff_active():
            return False
        ffmpeg_result = self._run_ffmpeg_live_segment_once()
        if ffmpeg_result is not None:
            return ffmpeg_result
        if iter_luxriot_live_segment_frames is None:
            self.last_live_segment_error = "road_events live segment decoder is unavailable"
            self._set_live_segment_backoff(failed=True)
            if self.capture_source_mode == "live_segment":
                raise RuntimeError(self.last_live_segment_error)
            return False
        budget = self._live_segment_capture_budget()
        segment_seconds = float(budget["stream_seconds"])
        segment_bytes = int(budget["byte_budget"])
        every_n = int(getattr(self.manager.config, "LUXRIOT_LIVE_SEGMENT_EVERY_N", 25))
        frame_limit = int(budget["raw_frame_budget"])
        started = time.monotonic()
        accepted = 0
        first_timestamp_ms: Optional[int] = None
        last_timestamp_ms: Optional[int] = None
        self._mark_live_segment_inflight(budget)
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
                decoded_timestamp_ms = int(decoded.timestamp_ms or int(time.time() * 1000))
                timestamp_ms = self._next_live_source_timestamp_ms(
                    source_anchor_ms=decoded_timestamp_ms,
                    frame_index=0,
                    fps=float(budget["fps"]),
                )
                self._accept_captured_frame(image, timestamp_ms, summarize=False)
                if first_timestamp_ms is None:
                    first_timestamp_ms = timestamp_ms
                last_timestamp_ms = timestamp_ms
                accepted += 1
                with self.lock:
                    self.live_segment_inflight_frames = int(accepted)
                    self.live_segment_inflight_represented_seconds = round(
                        max(
                            0.0,
                            float(last_timestamp_ms - first_timestamp_ms) / 1000.0,
                        ),
                        3,
                    )
            latency = max(0.0, time.monotonic() - started)
            with self.lock:
                self.active_capture_source = "live_segment"
                self.live_segment_count += 1
                self.live_segment_frame_count += accepted
                self.last_live_segment_latency_sec = latency
                self.last_live_segment_frames = accepted
                self.last_live_segment_error = None if accepted > 0 else "live segment produced no decoded frames"
            represented_seconds = (
                max(0.0, float(last_timestamp_ms - first_timestamp_ms) / 1000.0)
                if first_timestamp_ms is not None and last_timestamp_ms is not None
                else 0.0
            )
            self._complete_live_segment_budget(
                budget,
                streamed_bytes=0,
                represented_seconds=represented_seconds,
                source_start_timestamp_ms=first_timestamp_ms,
                last_source_timestamp_ms=last_timestamp_ms,
                timestamp_source="decoder_source_timestamp_ms",
            )
            self._set_live_segment_backoff(failed=accepted <= 0)
            self._flush_capture_apex_bucket()
            self._summarize_if_ready()
            if accepted > 0:
                self._clear_capture_error()
            return accepted > 0
        except Exception as exc:
            latency = max(0.0, time.monotonic() - started)
            with self.lock:
                self.active_capture_source = "live_segment"
                self.live_segment_failed_count += 1
                self.last_live_segment_latency_sec = latency
                self.last_live_segment_frames = accepted
                self.last_live_segment_error = _safe_error_text(exc, 240) or exc.__class__.__name__
            represented_seconds = (
                max(0.0, float(last_timestamp_ms - first_timestamp_ms) / 1000.0)
                if first_timestamp_ms is not None and last_timestamp_ms is not None
                else 0.0
            )
            self._complete_live_segment_budget(
                budget,
                streamed_bytes=0,
                represented_seconds=represented_seconds,
                source_start_timestamp_ms=first_timestamp_ms,
                last_source_timestamp_ms=last_timestamp_ms,
                timestamp_source="decoder_source_timestamp_ms",
            )
            self._set_live_segment_backoff(failed=True)
            if self.capture_source_mode == "live_segment":
                raise
            return False

    def _pipe_authenticated_live_segment(
        self,
        process: Any,
        *,
        max_bytes: int,
        max_seconds: float,
        read_timeout_sec: float,
        cancel_event: Optional[threading.Event] = None,
        state: Optional[Dict[str, Any]] = None,
        ready_event: Optional[threading.Event] = None,
        default_source_anchor_ms: Optional[int] = None,
    ) -> int:
        """Pipe a bounded DigestAuth response to ffmpeg without credential argv."""

        response = self.client.open_live_stream(
            int(self.channel_id),
            stream="mainStream",
            timeout=max(1, int(math.ceil(read_timeout_sec))),
        )
        shared_state = state if isinstance(state, dict) else {}
        shared_state["response"] = response
        shared_state["source_anchor_ms"] = int(
            default_source_anchor_ms or int(time.time() * 1000)
        )
        shared_state["timestamp_source"] = "capture_window_started_at"
        upstream_headers = getattr(response, "headers", {}) or {}
        stream_start_ms = _parse_optional_int(
            upstream_headers.get("X-Stream-Start-Time")
            or upstream_headers.get("x-stream-start-time")
        )
        if stream_start_ms is not None and stream_start_ms > 0:
            shared_state["source_anchor_ms"] = int(stream_start_ms)
            shared_state["timestamp_source"] = "evo_x_stream_start_time"
        if ready_event is not None:
            ready_event.set()
        process_input = getattr(process, "stdin", None)
        if process_input is None:
            try:
                response.close()
            except Exception:
                pass
            shared_state["response"] = None
            raise RuntimeError("ffmpeg stdin pipe is unavailable")
        byte_budget = max(1, int(max_bytes))
        deadline = time.monotonic() + max(0.25, float(max_seconds))
        written = 0
        pipe_closed = False
        try:
            for raw_chunk in response.iter_content(65_536):
                if (
                    self.stop_event.is_set()
                    or (cancel_event is not None and cancel_event.is_set())
                    or time.monotonic() >= deadline
                ):
                    break
                if callable(getattr(process, "poll", None)) and process.poll() is not None:
                    break
                if not raw_chunk:
                    continue
                chunk = raw_chunk.encode("utf-8") if isinstance(raw_chunk, str) else bytes(raw_chunk)
                remaining = byte_budget - written
                if remaining <= 0:
                    break
                view = memoryview(chunk[:remaining])
                while view:
                    if (
                        self.stop_event.is_set()
                        or (cancel_event is not None and cancel_event.is_set())
                        or time.monotonic() >= deadline
                    ):
                        view = view[:0]
                        break
                    try:
                        count = process_input.write(view)
                    except (BrokenPipeError, OSError):
                        pipe_closed = True
                        view = view[:0]
                        break
                    if not count:
                        pipe_closed = True
                        view = view[:0]
                        break
                    written += int(count)
                    view = view[int(count) :]
                if pipe_closed or written >= byte_budget:
                    break
        finally:
            shared_state["bytes"] = int(written)
            try:
                process_input.close()
            except Exception:
                pass
            try:
                response.close()
            except Exception:
                pass
            shared_state["response"] = None
            if ready_event is not None:
                ready_event.set()
        if written <= 0:
            raise RuntimeError(
                f"Luxriot live stream for channel {self.channel_id} returned no video bytes."
            )
        return int(written)

    @staticmethod
    def _bounded_file_tail(path: Path, limit: int = 8192) -> str:
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - max(1, int(limit))), os.SEEK_SET)
                return handle.read(max(1, int(limit))).decode("utf-8", errors="replace").strip()
        except Exception:
            return ""

    @staticmethod
    def _terminate_ffmpeg_process(process: Any) -> None:
        if process is None:
            return
        try:
            if process.poll() is None:
                process.kill()
        except Exception:
            pass
        try:
            process.wait(timeout=1.0)
        except Exception:
            pass

    def _live_segment_backoff_active(self) -> bool:
        with self.lock:
            return float(self.live_segment_backoff_until or 0.0) > time.monotonic()

    def _set_live_segment_backoff(self, *, failed: bool) -> None:
        with self.lock:
            if failed:
                delay_sec = max(2.0, min(10.0, float(self.interval) * 5.0))
                self.live_segment_backoff_until = time.monotonic() + delay_sec
            else:
                self.live_segment_backoff_until = 0.0

    @staticmethod
    def _extract_complete_jpegs(buffer: bytearray) -> List[bytes]:
        frames: List[bytes] = []
        while True:
            start = buffer.find(b"\xff\xd8")
            if start < 0:
                if len(buffer) > 1:
                    del buffer[:-1]
                break
            if start > 0:
                del buffer[:start]
            end = buffer.find(b"\xff\xd9", 2)
            if end < 0:
                break
            frames.append(bytes(buffer[: end + 2]))
            del buffer[: end + 2]
        return frames

    def _run_ffmpeg_live_segment_once(self) -> Optional[bool]:
        budget = self._live_segment_capture_budget()
        frame_limit = int(budget["raw_frame_budget"])
        fps = float(budget["fps"])
        segment_seconds = float(budget["stream_seconds"])
        read_timeout_sec = float(getattr(self.manager.config, "LUXRIOT_LIVE_SEGMENT_READ_TIMEOUT_SEC", 5.0))
        read_timeout_sec = max(1.0, min(30.0, read_timeout_sec))
        segment_bytes = int(budget["byte_budget"])
        timeout_sec = max(6.0, segment_seconds + read_timeout_sec + 2.0, (float(frame_limit) / fps) + read_timeout_sec + 2.0)
        timeout_sec = min(90.0, timeout_sec)
        rw_timeout_us = str(int(read_timeout_sec * 1_000_000))
        started = time.monotonic()
        accepted = 0
        decoded_frame_count = 0
        streamed_bytes = 0
        process: Any = None
        feeder_thread: Optional[threading.Thread] = None
        stdout_thread: Optional[threading.Thread] = None
        window_cancel = threading.Event()
        feed_ready = threading.Event()
        feed_done = threading.Event()
        stdout_done = threading.Event()
        feed_state: Dict[str, Any] = {
            "bytes": 0,
            "error": None,
            "response": None,
            "source_anchor_ms": int(time.time() * 1000),
            "timestamp_source": "capture_window_started_at",
        }
        stdout_state: Dict[str, Any] = {"error": None}
        stdout_queue: queue.Queue[Any] = queue.Queue(maxsize=16)
        stdout_sentinel = object()
        first_source_timestamp_ms: Optional[int] = None
        last_source_timestamp_ms: Optional[int] = None
        stopped = False
        self._mark_live_segment_inflight(budget)

        def close_window_io(*, terminate: bool) -> None:
            window_cancel.set()
            response = feed_state.get("response")
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
            process_input = getattr(process, "stdin", None)
            if process_input is not None:
                try:
                    process_input.close()
                except Exception:
                    pass
            if terminate:
                self._terminate_ffmpeg_process(process)

        def feed_process() -> None:
            try:
                feed_state["bytes"] = self._pipe_authenticated_live_segment(
                    process,
                    max_bytes=segment_bytes,
                    max_seconds=segment_seconds,
                    read_timeout_sec=read_timeout_sec,
                    cancel_event=window_cancel,
                    state=feed_state,
                    ready_event=feed_ready,
                )
            except BaseException as exc:  # thread boundary; inspected in capture loop
                feed_state["error"] = exc
            finally:
                feed_ready.set()
                feed_done.set()

        def read_stdout() -> None:
            try:
                process_stdout = getattr(process, "stdout", None)
                if process_stdout is None:
                    raise RuntimeError("ffmpeg stdout pipe is unavailable")
                while not window_cancel.is_set():
                    chunk = process_stdout.read(65_536)
                    if not chunk:
                        break
                    while not window_cancel.is_set():
                        try:
                            stdout_queue.put(bytes(chunk), timeout=0.1)
                            break
                        except queue.Full:
                            continue
            except BaseException as exc:  # thread boundary; inspected in capture loop
                if not window_cancel.is_set():
                    stdout_state["error"] = exc
            finally:
                stdout_done.set()
                try:
                    stdout_queue.put(stdout_sentinel, timeout=0.1)
                except queue.Full:
                    pass

        try:
            with tempfile.TemporaryDirectory(prefix=f"eva-live-ch{self.channel_id}-") as temp_dir:
                stderr_path = Path(temp_dir) / "ffmpeg.stderr"
                dense_max_edge = max(160, min(1280, int(self.max_edge or 800)))
                cmd = [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-rw_timeout",
                    rw_timeout_us,
                    "-i",
                    "pipe:0",
                    "-vf",
                    (
                        f"fps={fps:g},scale={dense_max_edge}:{dense_max_edge}:"
                        "force_original_aspect_ratio=decrease"
                    ),
                    "-frames:v",
                    str(frame_limit),
                    "-q:v",
                    "4",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "mjpeg",
                    "pipe:1",
                ]
                with stderr_path.open("wb") as stderr_output:
                    process = subprocess.Popen(
                        cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=stderr_output,
                        bufsize=0,
                    )
                    feeder_thread = threading.Thread(
                        target=feed_process,
                        name=f"luxriot-feed-{self.channel_id}",
                        daemon=True,
                    )
                    stdout_thread = threading.Thread(
                        target=read_stdout,
                        name=f"luxriot-decode-{self.channel_id}",
                        daemon=True,
                    )
                    feeder_thread.start()
                    stdout_thread.start()

                    jpeg_buffer = bytearray()
                    process_deadline = started + timeout_sec
                    while True:
                        if self.stop_event.is_set():
                            stopped = True
                            close_window_io(terminate=True)
                            break
                        if time.monotonic() >= process_deadline:
                            close_window_io(terminate=True)
                            raise subprocess.TimeoutExpired(cmd, timeout_sec)
                        try:
                            chunk = stdout_queue.get(timeout=0.1)
                        except queue.Empty:
                            if stdout_done.is_set() and stdout_queue.empty():
                                break
                            continue
                        if chunk is stdout_sentinel:
                            break
                        jpeg_buffer.extend(chunk)
                        if len(jpeg_buffer) > 32 * 1024 * 1024:
                            raise RuntimeError("ffmpeg JPEG pipe exceeded the bounded frame buffer")
                        for jpeg_bytes in self._extract_complete_jpegs(jpeg_buffer):
                            frame_index = decoded_frame_count
                            decoded_frame_count += 1
                            with self.lock:
                                self.live_segment_inflight_frames = int(decoded_frame_count)
                                self.live_segment_inflight_represented_seconds = round(
                                    float(decoded_frame_count) / fps,
                                    3,
                                )
                            try:
                                with Image.open(BytesIO(jpeg_bytes)) as opened:
                                    opened.load()
                                    image = opened.convert("RGB")
                            except Exception:
                                continue
                            source_anchor_ms = int(
                                _parse_optional_int(feed_state.get("source_anchor_ms"))
                                or int(time.time() * 1000)
                            )
                            timestamp_ms = self._next_live_source_timestamp_ms(
                                source_anchor_ms=source_anchor_ms,
                                frame_index=frame_index,
                                fps=fps,
                            )
                            if first_source_timestamp_ms is None:
                                first_source_timestamp_ms = timestamp_ms
                            last_source_timestamp_ms = timestamp_ms
                            self._accept_captured_frame(image, timestamp_ms, summarize=False)
                            accepted += 1
                            # A newly crossed second may have finalized one apex.
                            # Dispatch complete summary batches while this same
                            # authenticated stream and ffmpeg process stay open.
                            self._summarize_if_ready()

                    if not stopped:
                        while True:
                            if self.stop_event.is_set():
                                stopped = True
                                close_window_io(terminate=True)
                                returncode = int(getattr(process, "returncode", 0) or 0)
                                break
                            remaining_timeout = process_deadline - time.monotonic()
                            if remaining_timeout <= 0:
                                close_window_io(terminate=True)
                                raise subprocess.TimeoutExpired(cmd, timeout_sec)
                            try:
                                returncode = int(
                                    process.wait(timeout=min(0.1, max(0.01, remaining_timeout)))
                                )
                                break
                            except subprocess.TimeoutExpired:
                                continue
                    else:
                        returncode = int(getattr(process, "returncode", 0) or 0)

                close_window_io(terminate=False)
                if feeder_thread is not None:
                    feeder_thread.join(timeout=0.5)
                if stdout_thread is not None:
                    stdout_thread.join(timeout=0.5)
                if callable(getattr(process, "poll", None)) and process.poll() is None:
                    self._terminate_ffmpeg_process(process)
                streamed_bytes = int(feed_state.get("bytes") or 0)
                latency = max(0.0, time.monotonic() - started)
                stderr = self._bounded_file_tail(stderr_path)
                feed_error = feed_state.get("error")
                stdout_error = stdout_state.get("error")
                process_error = None
                if not stopped:
                    process_error = feed_error or stdout_error
                    if process_error is None and returncode not in {0, 127}:
                        process_error = RuntimeError(
                            _safe_error_text(stderr, 240)
                            or f"ffmpeg exited {returncode}"
                        )
                if not stopped and accepted <= 0 and returncode == 127:
                    self._cancel_live_segment_inflight()
                    return None
                with self.lock:
                    self.active_capture_source = "live_segment"
                    self.live_segment_count += 1 if accepted > 0 else 0
                    self.live_segment_frame_count += accepted
                    self.last_live_segment_latency_sec = latency
                    self.last_live_segment_frames = accepted
                    if stopped:
                        pass
                    elif accepted > 0 and process_error is None:
                        self.last_live_segment_error = None
                    else:
                        self.live_segment_failed_count += 1
                        self.last_live_segment_error = (
                            _safe_error_text(process_error, 240)
                            or _safe_error_text(stderr, 240)
                            or f"ffmpeg exited {returncode} without frames"
                        )
                self._complete_live_segment_budget(
                    budget,
                    streamed_bytes=streamed_bytes,
                    represented_seconds=float(decoded_frame_count) / fps,
                    source_start_timestamp_ms=first_source_timestamp_ms,
                    last_source_timestamp_ms=last_source_timestamp_ms,
                    timestamp_source=str(feed_state.get("timestamp_source") or "capture_window_started_at"),
                )
                self._set_live_segment_backoff(
                    failed=not stopped and (accepted <= 0 or process_error is not None)
                )
                if not stopped:
                    self._flush_capture_apex_bucket()
                    self._summarize_if_ready()
                if accepted > 0 and process_error is None:
                    self._clear_capture_error()
                    return True
                if accepted > 0:
                    if process_error is not None:
                        self._set_capture_error(process_error)
                    return True
                if process_error is not None:
                    self._set_capture_error(process_error)
                return False
        except FileNotFoundError:
            close_window_io(terminate=True)
            self._cancel_live_segment_inflight()
            return None
        except subprocess.TimeoutExpired as exc:
            close_window_io(terminate=True)
            if feeder_thread is not None:
                feeder_thread.join(timeout=0.5)
            if stdout_thread is not None:
                stdout_thread.join(timeout=0.5)
            latency = max(0.0, time.monotonic() - started)
            with self.lock:
                self.active_capture_source = "live_segment"
                self.live_segment_failed_count += 1
                self.last_live_segment_latency_sec = latency
                self.last_live_segment_frames = accepted
                self.last_live_segment_error = f"ffmpeg live segment timed out after {timeout_sec:.1f}s"
            self._complete_live_segment_budget(
                budget,
                streamed_bytes=int(feed_state.get("bytes") or streamed_bytes),
                represented_seconds=float(decoded_frame_count) / fps,
                source_start_timestamp_ms=first_source_timestamp_ms,
                last_source_timestamp_ms=last_source_timestamp_ms,
                timestamp_source=str(feed_state.get("timestamp_source") or "capture_window_started_at"),
            )
            self._flush_capture_apex_bucket()
            self._summarize_if_ready()
            self._set_live_segment_backoff(failed=True)
            if self.capture_source_mode == "live_segment":
                raise RuntimeError(self.last_live_segment_error) from exc
            return False
        except Exception as exc:
            close_window_io(terminate=True)
            if feeder_thread is not None:
                feeder_thread.join(timeout=0.5)
            if stdout_thread is not None:
                stdout_thread.join(timeout=0.5)
            latency = max(0.0, time.monotonic() - started)
            with self.lock:
                self.active_capture_source = "live_segment"
                self.live_segment_failed_count += 1
                self.last_live_segment_latency_sec = latency
                self.last_live_segment_frames = accepted
                self.last_live_segment_error = _safe_error_text(exc, 240) or exc.__class__.__name__
            self._complete_live_segment_budget(
                budget,
                streamed_bytes=int(feed_state.get("bytes") or streamed_bytes),
                represented_seconds=float(decoded_frame_count) / fps,
                source_start_timestamp_ms=first_source_timestamp_ms,
                last_source_timestamp_ms=last_source_timestamp_ms,
                timestamp_source=str(feed_state.get("timestamp_source") or "capture_window_started_at"),
            )
            self._flush_capture_apex_bucket()
            self._summarize_if_ready()
            self._set_live_segment_backoff(failed=True)
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

    @staticmethod
    def _capture_cv_gray(snapshot: object, max_edge: int = 160) -> Optional[Image.Image]:
        if not isinstance(snapshot, Image.Image):
            return None
        try:
            gray = snapshot.convert("L")
            edge = max(gray.size)
            if edge > max(32, int(max_edge)):
                scale = float(max_edge) / float(edge)
                size = (
                    max(1, int(round(gray.width * scale))),
                    max(1, int(round(gray.height * scale))),
                )
                resampling = getattr(Image, "Resampling", Image)
                gray = gray.resize(size, resample=resampling.BILINEAR)
            return gray
        except Exception:
            return None

    @staticmethod
    def _capture_cv_sharpness_score(gray: Optional[Image.Image]) -> Optional[float]:
        """Edge-energy variance on the downscaled gray frame (motion-blur proxy)."""

        if gray is None:
            return None
        try:
            stat = ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES))
            if not stat.var:
                return None
            return max(0.0, float(stat.var[0]))
        except Exception:
            return None

    @staticmethod
    def _sharpness_discriminates(values: Sequence[Optional[float]]) -> bool:
        present = [float(value) for value in values if value is not None]
        if len(present) < 2:
            return False
        top = max(present)
        if top <= 0.0:
            return False
        low = min(present)
        if low <= 0.0:
            return True
        return (top / low) >= _CAPTURE_SHARPNESS_DISCRIMINATION

    @staticmethod
    def _capture_cv_delta_score(previous: Optional[Image.Image], current: Optional[Image.Image]) -> Optional[float]:
        if previous is None or current is None:
            return None
        try:
            if previous.size != current.size:
                resampling = getattr(Image, "Resampling", Image)
                previous = previous.resize(current.size, resample=resampling.BILINEAR)
            difference = ImageChops.difference(previous, current)
            mean = ImageStat.Stat(difference).mean
            if not mean:
                return 0.0
            return max(0.0, min(1.0, float(mean[0]) / 255.0))
        except Exception:
            return None

    def _add_selected_probe_frame(
        self,
        image: object,
        timestamp_ms: int,
        provenance: Mapping[str, Any],
    ) -> None:
        probe_manager = self.manager.probe_manager
        if probe_manager is None or not isinstance(image, Image.Image):
            with self.lock:
                self.capture_apex_probe_skipped_count += 1
            return
        if not self.manager.should_dispatch_probe_frame(
            self.channel_id,
            capture_kind=self.capture_kind,
        ):
            with self.lock:
                self.capture_apex_probe_skipped_count += 1
            return
        try:
            try:
                probe_manager.add_frame(
                    self.channel_id,
                    image,
                    int(timestamp_ms),
                    provenance=dict(provenance),
                )
            except TypeError as exc:
                message = str(exc).lower()
                if "provenance" not in message or "keyword" not in message:
                    raise
                probe_manager.add_frame(self.channel_id, image, int(timestamp_ms))
            self._clear_probe_error()
            with self.lock:
                self.capture_apex_probe_dispatch_count += 1
        except Exception as exc:
            with self.lock:
                self.capture_apex_probe_failure_count += 1
            self._set_probe_error(exc)

    def _effective_selector_bias(self) -> str:
        """Read the live channel setting; overrides apply without a restart."""

        bias = "auto"
        getter = getattr(self.manager, "get_capture_selector_bias", None)
        if callable(getter):
            try:
                bias = str(getter(self.channel_id) or "auto").strip().lower()
            except Exception:
                bias = "auto"
        else:
            bias = str(self.capture_selector_bias or "auto").strip().lower()
        if bias not in CAPTURE_SELECTOR_BIASES:
            bias = "auto"
        self.capture_selector_bias = bias
        return bias

    def _capture_noise_floor(self) -> float:
        try:
            floor = float(getattr(self.manager.config, "LUXRIOT_CAPTURE_ACTIVITY_NOISE_FLOOR", 0.004))
        except (TypeError, ValueError):
            floor = 0.004
        return max(0.0, min(0.25, floor))

    def _capture_burst_zscore(self) -> float:
        try:
            zscore = float(getattr(self.manager.config, "LUXRIOT_CAPTURE_BURST_ZSCORE", 6.0))
        except (TypeError, ValueError):
            zscore = 6.0
        return max(1.0, min(50.0, zscore))

    def _capture_baseline_snapshot_locked(self) -> Dict[str, Any]:
        level = self.capture_activity_baseline_level
        return {
            "level": round(float(level), 6) if level is not None else None,
            "dev": round(float(self.capture_activity_baseline_dev), 6),
            "buckets": int(self.capture_activity_baseline_buckets),
            "warmup": int(self.capture_activity_baseline_buckets) < _CAPTURE_BASELINE_WARMUP_BUCKETS,
        }

    def _update_capture_activity_baseline_locked(self, bucket_peak: float) -> None:
        sample = max(0.0, float(bucket_peak))
        level = self.capture_activity_baseline_level
        noise_floor = self._capture_noise_floor()
        if level is None:
            self.capture_activity_baseline_level = sample
            self.capture_activity_baseline_dev = 0.0
        else:
            # Winsorize the sample so a burst does not immediately raise the
            # very baseline that detected it; genuine regime shifts still adapt.
            ceiling = float(level) + 3.0 * max(float(self.capture_activity_baseline_dev), noise_floor / 2.0)
            clamped = min(sample, ceiling)
            alpha = _CAPTURE_BASELINE_ALPHA
            next_level = ((1.0 - alpha) * float(level)) + (alpha * clamped)
            self.capture_activity_baseline_level = next_level
            deviation = abs(clamped - next_level)
            self.capture_activity_baseline_dev = (
                ((1.0 - alpha) * float(self.capture_activity_baseline_dev)) + (alpha * deviation)
            )
        self.capture_activity_baseline_buckets += 1

    def _classify_capture_bucket_mode(
        self,
        bucket_peak: float,
        baseline: Mapping[str, Any],
        bias: str,
    ) -> str:
        noise_floor = self._capture_noise_floor()
        if bias == "clarity":
            return "quiet"
        if bias == "action":
            return "burst" if bucket_peak > noise_floor else "quiet"
        if bucket_peak <= noise_floor:
            return "quiet"
        level = baseline.get("level")
        if not bool(baseline.get("warmup")) and level is not None:
            threshold = float(level) + self._capture_burst_zscore() * max(
                float(baseline.get("dev") or 0.0),
                noise_floor / 2.0,
            )
            if bucket_peak > threshold:
                return "burst"
        return "normal"

    def _select_burst_companion(
        self,
        scored: Sequence[Mapping[str, Any]],
        selected: Mapping[str, Any],
        activity_peak: float,
    ) -> Optional[Dict[str, Any]]:
        """Pick a meaningfully sharper frame of the same action second, if any."""

        threshold = _CAPTURE_COMPANION_ACTIVITY_BAND * max(0.0, float(activity_peak))
        pool: List[Tuple[float, Mapping[str, Any]]] = []
        for item in scored:
            if item is selected:
                continue
            sharpness = self.manager._finite_float(item.get("cv_sharpness_score"))
            if sharpness is None or float(sharpness) <= 0.0:
                continue
            if float(self.manager._finite_float(item.get("cv_attention_score")) or 0.0) < threshold:
                continue
            pool.append((float(sharpness), item))
        if not pool:
            return None
        best_sharpness, best = max(
            pool,
            key=lambda entry: (
                entry[0],
                -int(_parse_optional_int(entry[1].get("source_frame_index")) or 0),
            ),
        )
        selected_sharpness = float(self.manager._finite_float(selected.get("cv_sharpness_score")) or 0.0)
        if selected_sharpness > 0.0 and (best_sharpness / selected_sharpness) < _CAPTURE_COMPANION_SHARPNESS_GAIN:
            return None
        return dict(best)

    def _encode_burst_companion(self, companion: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        image = companion.get("image")
        if image is None:
            return None
        try:
            thumbnail = self.manager.jpeg_encoder(image, max_edge=self.max_edge, quality=85)
        except Exception:
            return None
        if not thumbnail:
            return None
        return {
            "role": "burst_sharp_companion",
            "thumbnail": thumbnail,
            "timestamp_ms": int(_parse_optional_int(companion.get("timestamp_ms")) or 0),
            "source_frame_index": int(_parse_optional_int(companion.get("source_frame_index")) or 0),
            "frame_hash": str(companion.get("frame_hash") or "")[:40],
            "sharpness": round(float(self.manager._finite_float(companion.get("cv_sharpness_score")) or 0.0), 6),
            "activity": round(float(self.manager._finite_float(companion.get("cv_attention_score")) or 0.0), 6),
        }

    def _finalize_capture_apex_bucket(self, bucket: Sequence[Mapping[str, Any]]) -> bool:
        candidates = [dict(item) for item in bucket if isinstance(item, Mapping)]
        if not candidates:
            return False
        candidates.sort(
            key=lambda item: (
                int(_parse_optional_int(item.get("timestamp_ms")) or 0),
                int(_parse_optional_int(item.get("source_frame_index")) or 0),
            )
        )
        def activity_of(item: Mapping[str, Any]) -> float:
            return float(self.manager._finite_float(item.get("cv_attention_score")) or 0.0)

        def sharpness_of(item: Mapping[str, Any]) -> Optional[float]:
            return self.manager._finite_float(item.get("cv_sharpness_score"))

        def order_of(item: Mapping[str, Any]) -> int:
            # Negated so that max() prefers the earliest source frame on ties.
            return -int(_parse_optional_int(item.get("source_frame_index")) or 0)

        activity_peak = max((activity_of(item) for item in candidates), default=0.0)
        bias = self._effective_selector_bias()
        with self.lock:
            baseline = self._capture_baseline_snapshot_locked()
            self._update_capture_activity_baseline_locked(activity_peak)
        selection_mode = self._classify_capture_bucket_mode(activity_peak, baseline, bias)
        activity_x: Optional[float] = None
        baseline_level = baseline.get("level")
        if baseline_level is not None and float(baseline_level) > 0.0:
            activity_x = activity_peak / float(baseline_level)

        fallback_reason = ""
        selected_score: Optional[float] = None
        score_source = ""
        companion: Optional[Dict[str, Any]] = None
        if len(candidates) == 1:
            selected = candidates[0]
            selection_source = "single_frame"
            fallback_reason = "single_frame_only_no_intra_second_choice"
            apex_available = False
        else:
            scored = [item for item in candidates if activity_of(item) > 0.0]
            all_sharpness = [sharpness_of(item) for item in candidates]
            if selection_mode == "burst" and scored:
                # Fast events only exist as motion-blurred frames; the blur is
                # the evidence, so the motion peak wins outright.
                selected = max(
                    scored,
                    key=lambda item: (activity_of(item), sharpness_of(item) or 0.0, order_of(item)),
                )
                selection_source = "capture_cv_frame_delta"
                selected_score = activity_of(selected)
                score_source = "mean_absolute_grayscale_frame_delta"
                apex_available = True
                companion = self._select_burst_companion(scored, selected, activity_peak)
            elif selection_mode == "normal" and scored:
                band = [
                    item
                    for item in scored
                    if activity_of(item) >= _CAPTURE_NORMAL_ACTIVITY_BAND * activity_peak
                ]
                if band and self._sharpness_discriminates([sharpness_of(item) for item in band]):
                    selected = max(
                        band,
                        key=lambda item: (sharpness_of(item) or 0.0, activity_of(item), order_of(item)),
                    )
                    selection_source = "capture_cv_sharp_active"
                    selected_score = sharpness_of(selected)
                    score_source = "find_edges_variance"
                else:
                    selected = max(
                        scored,
                        key=lambda item: (activity_of(item), sharpness_of(item) or 0.0, order_of(item)),
                    )
                    selection_source = "capture_cv_frame_delta"
                    selected_score = activity_of(selected)
                    score_source = "mean_absolute_grayscale_frame_delta"
                apex_available = True
            elif self._sharpness_discriminates(all_sharpness):
                # Quiet second (or clarity bias): ship the clearest frame.
                selected = max(
                    candidates,
                    key=lambda item: (sharpness_of(item) or 0.0, activity_of(item), order_of(item)),
                )
                selection_source = "capture_cv_sharpest"
                selected_score = sharpness_of(selected)
                score_source = "find_edges_variance"
                apex_available = True
            else:
                timestamps = [
                    int(_parse_optional_int(item.get("timestamp_ms")) or 0)
                    for item in candidates
                ]
                midpoint_ms = (min(timestamps) + max(timestamps)) / 2.0
                selected = min(
                    candidates,
                    key=lambda item: (
                        abs(float(_parse_optional_int(item.get("timestamp_ms")) or 0) - midpoint_ms),
                        int(_parse_optional_int(item.get("source_frame_index")) or 0),
                    ),
                )
                selection_source = "deterministic_temporal_midpoint"
                fallback_reason = "no_positive_capture_cv_attention_score"
                apex_available = False

        source_indices = [
            int(_parse_optional_int(item.get("source_frame_index")) or 0)
            for item in candidates
        ]
        source_timestamps = [
            int(_parse_optional_int(item.get("timestamp_ms")) or 0)
            for item in candidates
        ]
        source_hashes = [str(item.get("frame_hash") or "")[:40] for item in candidates]
        selected_index = int(_parse_optional_int(selected.get("source_frame_index")) or 0)
        selected_timestamp = int(_parse_optional_int(selected.get("timestamp_ms")) or 0)
        selected_hash = str(selected.get("frame_hash") or "")[:40]
        bucket_start_ms = (source_timestamps[0] // 1000) * 1000
        provenance: Dict[str, Any] = {
            "version": 2,
            "policy": CAPTURE_APEX_POLICY,
            "frame_hash_source": "normalized_grayscale_sha1",
            "channel_id": int(self.channel_id),
            "bucket_start_ms": int(bucket_start_ms),
            "source_frame_indices": source_indices,
            "source_timestamps_ms": source_timestamps,
            "source_frame_hashes": source_hashes,
            "selected_source_frame_index": int(selected_index),
            "selected_timestamp_ms": int(selected_timestamp),
            "selected_frame_hash": selected_hash,
            "selection_source": selection_source,
            "apex_available": bool(apex_available),
        }
        provenance["selection_mode"] = selection_mode
        provenance["activity_peak"] = round(float(activity_peak), 6)
        if activity_x is not None:
            provenance["activity_x"] = round(float(activity_x), 3)
        provenance["baseline"] = dict(baseline)
        if bias != "auto":
            provenance["selector_bias"] = bias
        selected_sharpness_value = self.manager._finite_float(selected.get("cv_sharpness_score"))
        if selected_sharpness_value is not None:
            provenance["selected_sharpness"] = round(float(selected_sharpness_value), 6)
        if selected_score is not None:
            provenance["selection_score"] = round(float(selected_score), 6)
            provenance["score_source"] = score_source or "mean_absolute_grayscale_frame_delta"
        if fallback_reason:
            provenance["fallback_reason"] = fallback_reason

        frame = dict(selected.get("frame") or {})
        selected_image = selected.get("image")
        if not frame.get("thumbnail") and selected_image is not None:
            # JPEG/base64 work belongs only to the selected per-second apex.
            # Encoding every dense candidate throttles the stdout pipe and can
            # make a healthy live source appear slower than real time.
            try:
                frame["thumbnail"] = self.manager.jpeg_encoder(
                    selected_image,
                    max_edge=self.max_edge,
                    quality=85,
                )
            except Exception:
                # An unencodable apex must not kill the capture loop; the frame
                # keeps its provenance and flows on without a preview.
                pass
        if companion is not None:
            companion_payload = self._encode_burst_companion(companion)
            if companion_payload:
                frame["burst_companion"] = companion_payload
                provenance["companion"] = {
                    key: companion_payload[key]
                    for key in ("timestamp_ms", "source_frame_index", "frame_hash", "sharpness", "activity")
                }
        frame["capture_selection"] = dict(provenance)
        frame["capture_source_frame_count"] = len(candidates)
        frame["capture_selected_source_frame_index"] = int(selected_index)
        frame["capture_selection_source"] = selection_source
        frame["capture_selection_apex_available"] = bool(apex_available)
        if selected_score is not None:
            frame["capture_selection_score"] = round(float(selected_score), 6)
        if fallback_reason:
            frame["capture_selection_fallback_reason"] = fallback_reason

        with self.lock:
            self.frames.append(frame)
            self.recent_frames.append(frame)
            self.capture_apex_selected_count += 1
            if not apex_available:
                self.capture_apex_fallback_count += 1
            self.capture_apex_selection_sources[selection_source] = (
                self.capture_apex_selection_sources.get(selection_source, 0) + 1
            )
            self.capture_apex_mode_counts[selection_mode] = (
                self.capture_apex_mode_counts.get(selection_mode, 0) + 1
            )
            if frame.get("burst_companion"):
                self.capture_apex_companion_count += 1
            self.capture_apex_last_selection = dict(provenance)
            baseline_snapshot = self._capture_baseline_snapshot_locked()
            self._enforce_buffer_locked()

        note_baseline = getattr(self.manager, "note_capture_baseline", None)
        if callable(note_baseline):
            try:
                note_baseline(self.channel_id, baseline_snapshot)
            except Exception:
                pass

        self._add_selected_probe_frame(
            selected.get("image"),
            selected_timestamp,
            provenance,
        )
        return True

    def _flush_capture_apex_bucket(self) -> bool:
        with self.lock:
            bucket = list(self._capture_apex_bucket)
            self._capture_apex_bucket.clear()
            self._capture_apex_bucket_start_ms = None
        return self._finalize_capture_apex_bucket(bucket)

    def _accept_captured_frame(self, snapshot: Image.Image, timestamp_ms: int, *, summarize: bool = True) -> None:
        captured_at = max(0.0, float(timestamp_ms) / 1000.0)
        observed_at = time.time()
        current_gray = self._capture_cv_gray(snapshot)
        fallback_thumbnail: Optional[str] = None
        if current_gray is not None:
            hash_payload = (
                f"{current_gray.width}x{current_gray.height}:".encode("ascii")
                + current_gray.tobytes()
            )
        else:
            # The frame identity must stay content-based: mixing the timestamp
            # in would make every frame unique and silently disable frozen
            # signal detection whenever grayscale conversion is unavailable.
            # Without grayscale pixels the encoded frame is the identity
            # source, exactly as in the pre-apex pipeline.
            try:
                fallback_thumbnail = self.manager.jpeg_encoder(
                    snapshot,
                    max_edge=self.max_edge,
                    quality=85,
                )
            except Exception:
                fallback_thumbnail = None
            if fallback_thumbnail:
                hash_payload = str(fallback_thumbnail).encode("utf-8", errors="ignore")
            else:
                try:
                    raw_bytes = snapshot.tobytes()
                except Exception:
                    raw_bytes = b""
                hash_payload = (
                    f"{snapshot.width}x{snapshot.height}:".encode("ascii") + raw_bytes
                )
        frame_hash = hashlib.sha1(hash_payload).hexdigest()[:16]
        frame = {
            "captured_at": captured_at,
            "time_sec": captured_at,
            "width": snapshot.width,
            "height": snapshot.height,
            "frame_hash": frame_hash,
        }
        if fallback_thumbnail:
            # Already paid for during identity hashing; the apex selector
            # skips re-encoding frames that carry a thumbnail.
            frame["thumbnail"] = fallback_thumbnail
        completed_bucket: List[Dict[str, Any]] = []
        with self.lock:
            if self.stop_event.is_set():
                return
            frozen_now = self._record_frame_hash_locked(frame_hash, observed_at)
            if frozen_now:
                self.frozen_frame_dropped_count += 1
                return
            self._capture_source_sequence += 1
            source_frame_index = int(self._capture_source_sequence)
            cv_attention_score = self._capture_cv_delta_score(
                self._capture_cv_previous_gray,
                current_gray,
            )
            cv_sharpness_score = self._capture_cv_sharpness_score(current_gray)
            self._capture_cv_previous_gray = current_gray
            bucket_start_ms = (int(timestamp_ms) // 1000) * 1000
            candidate: Dict[str, Any] = {
                "frame": frame,
                "image": snapshot.copy() if isinstance(snapshot, Image.Image) else snapshot,
                "timestamp_ms": int(timestamp_ms),
                "source_frame_index": source_frame_index,
                "frame_hash": frame_hash,
            }
            if cv_attention_score is not None:
                candidate["cv_attention_score"] = round(float(cv_attention_score), 6)
            if cv_sharpness_score is not None:
                candidate["cv_sharpness_score"] = round(float(cv_sharpness_score), 6)
            if (
                self._capture_apex_bucket_start_ms is not None
                and int(self._capture_apex_bucket_start_ms) != bucket_start_ms
            ):
                completed_bucket = list(self._capture_apex_bucket)
                self._capture_apex_bucket.clear()
            self._capture_apex_bucket_start_ms = int(bucket_start_ms)
            self._capture_apex_bucket.append(candidate)
            self.capture_apex_raw_frame_count += 1
        if completed_bucket:
            self._finalize_capture_apex_bucket(completed_bucket)
        if summarize and float(self.interval) >= 1.0:
            # Snapshot/1-fps capture has no intra-second choice to defer.  Commit
            # immediately so the configured cadence and batch timing stay unchanged.
            self._flush_capture_apex_bucket()
        if summarize:
            self._summarize_if_ready()

    def _summarize_if_ready(self) -> None:
        if not self.summarization_enabled:
            return
        while True:
            with self.lock:
                ready_to_summarize = len(self.frames) >= self.batch_size
            if not ready_to_summarize:
                return
            if not self._enqueue_summary_batch(frame_limit=self.batch_size):
                return

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

    def _dispatch_summary_frames(
        self,
        frames_copy: Sequence[Mapping[str, Any]],
        *,
        workload_class: str = "heartbeat",
        restore_on_failure: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        frame_items = [dict(frame) for frame in frames_copy if isinstance(frame, Mapping)]
        if not frame_items:
            return True
        job_meta = dict(metadata or {})
        try:
            batch = self.manager.create_summary_batch(
                channel_id=self.channel_id,
                run_id=str(job_meta.get("run_id") or self.run_id),
                batch_size=int(job_meta.get("batch_size") or self.batch_size),
                prompt=str(job_meta.get("prompt") if "prompt" in job_meta else self.prompt),
                model_hint=cast(Optional[str], job_meta.get("model_hint")) if job_meta.get("model_hint") else self.model_hint,
                interval_sec=float(job_meta.get("interval_sec") or self.interval),
                frames=frame_items,
                session_generation=(
                    str(job_meta.get("session_generation") or self.session_generation or "").strip()
                    or None
                ),
            )
            coalesced_meta = job_meta.get("coalesced")
            if isinstance(coalesced_meta, Mapping):
                batch["coalesced"] = {
                    "batches": max(1, int(_parse_optional_int(coalesced_meta.get("batches")) or 1)),
                    "omitted_frames": max(0, int(_parse_optional_int(coalesced_meta.get("omitted_frames")) or 0)),
                }
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
                if str(outcome.get("status") or "").strip().lower() in {
                    "superseded",
                    "stale_session",
                }:
                    return True
                with self.lock:
                    self._record_summary_failure_locked(
                        f"summary batch was not accepted ({outcome.get('status') or 'rejected'})",
                        dropped_frames=len(frame_items),
                    )
                return False
            return True
        except Exception as exc:
            with self.lock:
                if restore_on_failure:
                    self.frames = frame_items + self.frames
                    self._enforce_buffer_locked()
                self._record_summary_failure_locked(
                    exc,
                    dropped_frames=0 if restore_on_failure else len(frame_items),
                )
            return False

    def _summarize_batch(self, workload_class: str = "heartbeat") -> bool:
        with self.lock:
            frames_copy = list(self.frames)
            self.frames.clear()
        return self._dispatch_summary_frames(
            frames_copy,
            workload_class=workload_class,
            restore_on_failure=True,
        )

    def _summary_job_metadata(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "batch_size": int(self.batch_size),
            "prompt": self.prompt,
            "model_hint": self.model_hint,
            "interval_sec": float(self.interval),
            "session_generation": self.session_generation,
        }

    @staticmethod
    def _frame_capture_mode(frame: Mapping[str, Any]) -> str:
        selection = frame.get("capture_selection")
        if isinstance(selection, Mapping):
            return str(selection.get("selection_mode") or "").strip().lower()
        return ""

    @staticmethod
    def _frame_activity_x(frame: Mapping[str, Any]) -> float:
        selection = frame.get("capture_selection")
        if isinstance(selection, Mapping):
            try:
                return float(selection.get("activity_x") or 0.0)
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    @staticmethod
    def _frame_captured_at(frame: Mapping[str, Any]) -> float:
        for key in ("captured_at", "time_sec"):
            try:
                value = float(frame.get(key))
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        return 0.0

    @classmethod
    def _subsample_coalesced_frames(
        cls,
        frames: Sequence[Mapping[str, Any]],
        target_count: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Thin a merged window to target size, never sacrificing burst seconds."""

        items = [dict(frame) for frame in frames if isinstance(frame, Mapping)]
        items.sort(key=cls._frame_captured_at)
        target = max(1, int(target_count))
        if len(items) <= target:
            return items, 0
        keep: Set[int] = set()
        burst_indices = [
            index for index, frame in enumerate(items)
            if cls._frame_capture_mode(frame) == "burst"
        ]
        for index in burst_indices[:target]:
            keep.add(index)
        if len(keep) < target:
            # Strong normal seconds may claim at most half the remaining
            # slots; the rest is even temporal fill so a coalesced window
            # stays representative when activity does not discriminate.
            normal_slots = (target - len(keep)) // 2
            normal_indices = sorted(
                (
                    index for index, frame in enumerate(items)
                    if index not in keep
                    and cls._frame_capture_mode(frame) == "normal"
                    and cls._frame_activity_x(frame) > 0.0
                ),
                key=lambda index: -cls._frame_activity_x(items[index]),
            )
            for index in normal_indices[:normal_slots]:
                keep.add(index)
        if len(keep) < target:
            remaining = [index for index in range(len(items)) if index not in keep]
            need = target - len(keep)
            step = len(remaining) / float(need)
            for slot in range(need):
                keep.add(remaining[min(len(remaining) - 1, int(slot * step))])
        kept = [items[index] for index in sorted(keep)]
        return kept, len(items) - len(kept)

    def _coalesce_oldest_queued_batches_locked(self) -> bool:
        if len(self.summary_queue) < 2:
            return False
        frames_a, workload_a, meta_a = self.summary_queue[0]
        frames_b, _workload_b, meta_b = self.summary_queue[1]

        def merged_batch_count(meta: Mapping[str, Any]) -> int:
            info = meta.get("coalesced")
            if isinstance(info, Mapping):
                return max(1, int(_parse_optional_int(info.get("batches")) or 1))
            return 1

        def omitted_count(meta: Mapping[str, Any]) -> int:
            info = meta.get("coalesced")
            if isinstance(info, Mapping):
                return max(0, int(_parse_optional_int(info.get("omitted_frames")) or 0))
            return 0

        total_batches = merged_batch_count(meta_a) + merged_batch_count(meta_b)
        if total_batches > _SUMMARY_COALESCE_MAX_BATCHES:
            return False
        target = max(1, int(_parse_optional_int(meta_a.get("batch_size")) or self.batch_size))
        merged_frames, omitted = self._subsample_coalesced_frames(
            list(frames_a) + list(frames_b),
            target,
        )
        merged_meta = dict(meta_a)
        merged_meta["coalesced"] = {
            "batches": total_batches,
            "omitted_frames": omitted_count(meta_a) + omitted_count(meta_b) + int(omitted),
        }
        self.summary_queue[0:2] = [(merged_frames, workload_a, merged_meta)]
        self.summary_coalesced_batches += 1
        return True

    def _note_summary_coverage_gap(self, frames: Sequence[Mapping[str, Any]]) -> None:
        """Write an explicit history gap for a window dropped under backpressure.

        Must be called WITHOUT holding the session lock: the manager recorder
        takes its own cache lock and the safe order is manager -> session.
        """

        recorder = getattr(self.manager, "record_summary_log", None)
        if not callable(recorder) or not frames:
            return
        stamps = [
            self._frame_captured_at(frame)
            for frame in frames
            if isinstance(frame, Mapping)
        ]
        stamps = [stamp for stamp in stamps if stamp > 0]
        now = time.time()
        start_sec = min(stamps) if stamps else now
        end_sec = max(stamps) if stamps else now
        entry = {
            "channel_id": int(self.channel_id),
            "run_id": self.run_id,
            "summary": (
                "[coverage gap] L0 batch dropped under LM backpressure; "
                "this interval has no description."
            ),
            "coverage_gap": True,
            "gap_reason": "lm_backpressure_dropped_batch",
            "frame_count": len(frames),
            "batch_size": int(self.batch_size),
            "batch_start_ms": int(start_sec * 1000.0),
            "batch_end_ms": int(end_sec * 1000.0),
            "created_at": now,
        }
        try:
            recorder(self.channel_id, entry)
        except Exception:
            # The counters already recorded the drop; the gap entry is honesty,
            # not a second failure channel.
            pass

    def _enqueue_summary_batch(
        self,
        workload_class: str = "heartbeat",
        *,
        frame_limit: Optional[int] = None,
    ) -> bool:
        gap_batches: List[List[Dict[str, Any]]] = []
        queued = False
        with self.summary_condition:
            take_count = len(self.frames)
            if frame_limit is not None:
                take_count = min(take_count, max(1, int(frame_limit)))
            frames_copy = list(self.frames[:take_count])
            del self.frames[:take_count]
            if not frames_copy:
                return True
            metadata = self._summary_job_metadata()
            if not self.summary_worker_thread.is_alive():
                # Unit tests and manual direct calls often exercise the loop without
                # start(); preserve the previous synchronous semantics there.
                pass
            else:
                while len(self.summary_queue) >= self.summary_queue_max_batches:
                    if self._coalesce_oldest_queued_batches_locked():
                        continue
                    dropped_frames, _, _ = self.summary_queue.pop(0)
                    self._record_summary_failure_locked(
                        "summary queue overflow: oldest pending batch dropped",
                        dropped_frames=len(dropped_frames),
                    )
                    gap_batches.append(dropped_frames)
                self.summary_queue.append((frames_copy, str(workload_class or "heartbeat"), metadata))
                self.summary_condition.notify_all()
                queued = True
        for dropped_frames in gap_batches:
            self._note_summary_coverage_gap(dropped_frames)
        if queued:
            return True
        return self._dispatch_summary_frames(
            frames_copy,
            workload_class=workload_class,
            restore_on_failure=True,
            metadata=metadata,
        )

    def _summary_worker(self) -> None:
        while True:
            with self.summary_condition:
                while not self.summary_queue and not self.stop_event.is_set():
                    self.summary_condition.wait(timeout=0.5)
                if not self.summary_queue and self.stop_event.is_set():
                    return
                frames_copy, workload_class, metadata = self.summary_queue.pop(0)
                self.summary_inflight = True
            try:
                self._dispatch_summary_frames(
                    frames_copy,
                    workload_class=workload_class,
                    restore_on_failure=False,
                    metadata=metadata,
                )
            finally:
                with self.summary_condition:
                    self.summary_inflight = False
                    self.summary_condition.notify_all()

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
            self._flush_capture_apex_bucket()
            self._summarize_batch(workload_class="manual")

    def status(self) -> Dict[str, Any]:
        with self.lock:
            logs_copy = list(self.logs)
            capture_apex_pending_frames = len(self._capture_apex_bucket)
            pending_frames = len(self.frames) + capture_apex_pending_frames
            summary_queue_depth = len(self.summary_queue)
            summary_queue_frame_count = sum(len(item[0]) for item in self.summary_queue)
            summary_inflight = bool(self.summary_inflight)
            summary_worker_alive = self.summary_worker_thread.is_alive()
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
            last_live_segment_target_seconds = self.last_live_segment_target_seconds
            last_live_segment_summary_target_seconds = self.last_live_segment_summary_target_seconds
            last_live_segment_raw_frame_budget = self.last_live_segment_raw_frame_budget
            last_live_segment_byte_budget = self.last_live_segment_byte_budget
            last_live_segment_streamed_bytes = self.last_live_segment_streamed_bytes
            last_live_segment_represented_seconds = self.last_live_segment_represented_seconds
            last_live_segment_completed_at = self.last_live_segment_completed_at
            last_live_segment_source_start_timestamp_ms = self.last_live_segment_source_start_timestamp_ms
            last_live_segment_last_source_timestamp_ms = self.last_live_segment_last_source_timestamp_ms
            last_live_segment_timestamp_source = self.last_live_segment_timestamp_source
            live_segment_inflight = self.live_segment_inflight
            live_segment_capture_started_at = self.live_segment_capture_started_at
            live_segment_inflight_target_seconds = self.live_segment_inflight_target_seconds
            live_segment_inflight_raw_frame_budget = self.live_segment_inflight_raw_frame_budget
            live_segment_inflight_frames = self.live_segment_inflight_frames
            live_segment_inflight_represented_seconds = self.live_segment_inflight_represented_seconds
            live_segment_backoff_sec = max(0.0, float(self.live_segment_backoff_until or 0.0) - time.monotonic())
            frozen_signal = self.frozen_signal
            frozen_signal_since = self.frozen_signal_since
            frozen_frame_count = self.frozen_frame_count
            frozen_frame_hash = self.frozen_frame_hash
            frozen_frame_dropped_count = self.frozen_frame_dropped_count
            capture_last_error = self.capture_last_error
            probe_last_error = self.probe_last_error
            summary_last_error = self.summary_last_error
            summary_failed_batches = self.summary_failed_batches
            summary_last_failure_at = self.summary_last_failure_at
            summary_last_success_at = self.summary_last_success_at
            last_error = self.last_error
            capture_apex_raw_frame_count = self.capture_apex_raw_frame_count
            capture_apex_selected_count = self.capture_apex_selected_count
            capture_apex_fallback_count = self.capture_apex_fallback_count
            capture_apex_probe_dispatch_count = self.capture_apex_probe_dispatch_count
            capture_apex_probe_failure_count = self.capture_apex_probe_failure_count
            capture_apex_probe_skipped_count = self.capture_apex_probe_skipped_count
            capture_apex_selection_sources = dict(self.capture_apex_selection_sources)
            capture_apex_last_selection = dict(self.capture_apex_last_selection)
            capture_apex_mode_counts = dict(self.capture_apex_mode_counts)
            capture_apex_companion_count = self.capture_apex_companion_count
            capture_activity_baseline = self._capture_baseline_snapshot_locked()
            capture_selector_bias = str(self.capture_selector_bias or "auto")
        return {
            "running": not self.stop_event.is_set() and self.thread.is_alive(),
            "channel_id": self.channel_id,
            "run_id": self.run_id,
            "run_started_at": self.run_started_at,
            "session_generation": self.session_generation,
            "batch_size": self.batch_size,
            "pending_frames": pending_frames,
            "capture_apex_pending_frames": capture_apex_pending_frames,
            "capture_apex_raw_frame_count": capture_apex_raw_frame_count,
            "capture_apex_selected_count": capture_apex_selected_count,
            "capture_apex_fallback_count": capture_apex_fallback_count,
            "capture_apex_probe_dispatch_count": capture_apex_probe_dispatch_count,
            "capture_apex_probe_failure_count": capture_apex_probe_failure_count,
            "capture_apex_probe_skipped_count": capture_apex_probe_skipped_count,
            "capture_apex_selection_sources": capture_apex_selection_sources,
            "capture_apex_last_selection": capture_apex_last_selection,
            "capture_apex_mode_counts": capture_apex_mode_counts,
            "capture_apex_companion_count": capture_apex_companion_count,
            "capture_activity_baseline": capture_activity_baseline,
            "capture_selector_bias": capture_selector_bias,
            "summary_queue_depth": summary_queue_depth,
            "summary_queue_frame_count": summary_queue_frame_count,
            "summary_queue_max_batches": self.summary_queue_max_batches,
            "summary_inflight": summary_inflight,
            "summary_worker_alive": summary_worker_alive,
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
            "summary_coalesced_batches": self.summary_coalesced_batches,
            "last_queue_job_id": self.last_queue_job_id,
            "last_error": last_error,
            "capture_last_error": capture_last_error,
            "probe_last_error": probe_last_error,
            "summary_last_error": summary_last_error,
            "summary_failed_batches": summary_failed_batches,
            "summary_last_failure_at": summary_last_failure_at,
            "summary_last_success_at": summary_last_success_at,
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
            "last_live_segment_target_seconds": round(float(last_live_segment_target_seconds), 3),
            "last_live_segment_summary_target_seconds": round(float(last_live_segment_summary_target_seconds), 3),
            "last_live_segment_raw_frame_budget": int(last_live_segment_raw_frame_budget),
            "last_live_segment_byte_budget": int(last_live_segment_byte_budget),
            "last_live_segment_streamed_bytes": int(last_live_segment_streamed_bytes),
            "last_live_segment_represented_seconds": round(float(last_live_segment_represented_seconds), 3),
            "last_live_segment_completed_at": last_live_segment_completed_at,
            "last_live_segment_source_start_timestamp_ms": last_live_segment_source_start_timestamp_ms,
            "last_live_segment_last_source_timestamp_ms": last_live_segment_last_source_timestamp_ms,
            "last_live_segment_timestamp_source": last_live_segment_timestamp_source,
            "live_segment_inflight": bool(live_segment_inflight),
            "live_segment_capture_started_at": live_segment_capture_started_at,
            "live_segment_inflight_target_seconds": round(float(live_segment_inflight_target_seconds), 3),
            "live_segment_inflight_raw_frame_budget": int(live_segment_inflight_raw_frame_budget),
            "live_segment_inflight_frames": int(live_segment_inflight_frames),
            "live_segment_inflight_represented_seconds": round(float(live_segment_inflight_represented_seconds), 3),
            "live_segment_backoff_sec": round(float(live_segment_backoff_sec), 3),
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
    ROLLUP_BACKFILL_STATE_KEY = "luxriot_rollup_backfill:active"
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
        self.summary_archive_history_loader: Optional[SummaryArchiveHistoryLoaderFn] = None
        self.summary_archive_bucket_loader: Optional[SummaryArchiveBucketLoaderFn] = None
        self.system_prompt = getattr(config, "LUXRIOT_SYSTEM_PROMPT_DEFAULT", "")
        self.alert_policy_prompt = str(getattr(config, "LUXRIOT_ALERT_POLICY_PROMPT", "") or "")

        self.sessions: Dict[int, LuxriotCaptureSession] = {}
        self.probe_sessions: Dict[int, LuxriotCaptureSession] = {}
        self.shared_probe_channels: Set[int] = set()
        self.paused_probe_channels: Set[int] = set()
        # Manager helpers are layered (status/prompt/rollup paths call compact
        # helpers that may need the same cache). A re-entrant lock prevents one
        # request/background callback from permanently owning the global state
        # when those layers meet.
        self.cache_lock = threading.RLock()
        self.channels_cache: Optional[Tuple[float, List[Dict[str, Any]]]] = None
        self.channels_cache_stale = False
        self.channels_cache_last_error: Optional[str] = None
        self.channels_cache_last_attempt_at: Optional[float] = None
        self.channels_cache_last_success_at: Optional[float] = None
        self.channels_cache_stream_meta: Dict[str, Any] = {}
        self._session_generation_epoch = uuid4().hex
        self._session_generation_guard = threading.Lock()
        self._session_generation_counters: Dict[int, int] = {}
        self._session_generations: Dict[int, str] = {}
        self._session_side_effect_locks: Dict[int, Any] = {}
        self._desired_live_sessions_lock = threading.RLock()
        try:
            history_limit = int(getattr(config, "LUXRIOT_SUMMARY_HISTORY_LIMIT", 600))
        except Exception:
            history_limit = 600
        try:
            hot_history_limit = int(getattr(config, "LUXRIOT_SUMMARY_STATE_HOT_LIMIT", 2160))
        except Exception:
            hot_history_limit = 2160
        self.summary_state_hot_limit = max(240, min(10000, hot_history_limit))
        self.summary_history_limit = min(max(40, history_limit), self.summary_state_hot_limit)
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
        try:
            rollup_retention_days = float(
                getattr(
                    config,
                    "LUXRIOT_ROLLUP_RETENTION_DAYS",
                    self.summary_retention_days,
                )
            )
        except Exception:
            rollup_retention_days = self.summary_retention_days
        self.rollup_retention_days = max(0.0, rollup_retention_days)
        self.summary_history: Dict[int, List[Dict[str, Any]]] = {}
        self.channel_status_digest: Dict[int, Dict[str, Any]] = {}
        self.summary_runs: Dict[int, List[Dict[str, Any]]] = {}
        self.active_summary_runs: Dict[int, str] = {}
        self.channel_routine_context: Dict[int, Dict[str, Any]] = {}
        self.channel_observed_state_tracker: Dict[int, Dict[str, Dict[str, Any]]] = {}
        self.channel_prompt_overrides: Dict[int, Dict[str, Any]] = {}
        self._summary_state_last_persist = 0.0
        self._summary_state_dirty = False
        self.summary_state_revision = 0
        self._summary_state_revision_issued = 0
        self.summary_state_last_success_at: Optional[float] = None
        self.summary_state_last_error: Optional[str] = None
        self._summary_persist_condition = threading.Condition()
        self._summary_persist_pending: Optional[Dict[str, Any]] = None
        self._summary_persist_thread: Optional[threading.Thread] = None
        self.summary_state_backend = "runtime_state" if runtime_state_store is not None else "file"
        self._persisted_prompt_default_fields: Set[str] = set()
        try:
            persist_interval = float(getattr(config, "LUXRIOT_SUMMARY_STATE_PERSIST_INTERVAL_SEC", 15.0))
        except Exception:
            persist_interval = 15.0
        self.summary_state_persist_interval_sec = max(0.0, persist_interval)
        self.live_session_restore_errors: Dict[int, str] = {}
        self.channel_bookmark_fingerprints: Dict[int, Dict[str, int]] = {}
        self.channel_bookmark_content_keys: Dict[int, Dict[str, int]] = {}
        self.default_bookmark_enabled = bool(getattr(config, "LUXRIOT_AUTO_BOOKMARKS", False))
        try:
            cooldown_value = float(getattr(config, "LUXRIOT_BOOKMARK_COOLDOWN_SEC", 60.0))
        except Exception:
            cooldown_value = 60.0
        self.default_bookmark_cooldown_sec = max(0.0, cooldown_value)
        try:
            alert_dedupe_value = float(getattr(config, "LUXRIOT_ALERT_DEDUPE_WINDOW_SEC", 600.0))
        except Exception:
            alert_dedupe_value = 600.0
        self.alert_dedupe_window_sec = max(0.0, min(86400.0, alert_dedupe_value))
        try:
            max_alerts_value = int(getattr(config, "LUXRIOT_ALERTS_MAX_PER_BATCH", 8))
        except Exception:
            max_alerts_value = 8
        self.alerts_max_per_batch = max(1, min(32, max_alerts_value))
        self.default_json_alert_prompt = self._normalize_json_alert_prompt(
            getattr(config, "LUXRIOT_ALERTS_JSON_PROMPT", DEFAULT_ALERTS_JSON_PROMPT)
        )
        self.default_capture_selector_bias = (
            self._normalize_selector_bias(getattr(config, "LUXRIOT_CAPTURE_SELECTOR_BIAS", "auto"))
            or "auto"
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
            road_scene_samples = int(getattr(config, "LUXRIOT_ROAD_SCENE_CALIBRATION_SAMPLES", 8))
        except Exception:
            road_scene_samples = 8
        self.road_scene_calibration_samples = max(4, min(64, road_scene_samples))
        self.road_scene_auto_samples: Dict[int, List[Any]] = {}
        self.road_scene_calibrations: Dict[int, Dict[str, Any]] = {}
        self.road_episode_aggregators: Dict[int, Any] = {}
        # Measured per-channel motion homeostasis (capture decider baseline).
        self.capture_activity_baselines: Dict[int, Dict[str, Any]] = {}
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
        self.rollup_scheduler_enabled = bool(
            getattr(config, "LUXRIOT_ROLLUP_SCHEDULER_ENABLED", False)
        )
        try:
            scheduler_initial_delay = float(
                getattr(config, "LUXRIOT_ROLLUP_SCHEDULER_INITIAL_DELAY_SEC", 30.0)
            )
        except Exception:
            scheduler_initial_delay = 30.0
        self.rollup_scheduler_initial_delay_sec = max(1.0, min(600.0, scheduler_initial_delay))
        try:
            scheduler_spacing = float(
                getattr(config, "LUXRIOT_ROLLUP_SCHEDULER_SPACING_SEC", 15.0)
            )
        except Exception:
            scheduler_spacing = 15.0
        self.rollup_scheduler_spacing_sec = max(1.0, min(300.0, scheduler_spacing))
        try:
            scheduler_backfill_windows = int(
                getattr(config, "LUXRIOT_ROLLUP_SCHEDULER_BACKFILL_WINDOWS", 2)
            )
        except Exception:
            scheduler_backfill_windows = 2
        self.rollup_scheduler_backfill_windows = max(1, min(12, scheduler_backfill_windows))
        try:
            scheduler_max_deferral_windows = float(
                getattr(config, "LUXRIOT_ROLLUP_SCHEDULER_MAX_DEFERRAL_WINDOWS", 2.0)
            )
        except Exception:
            scheduler_max_deferral_windows = 2.0
        self.rollup_scheduler_max_deferral_windows = max(
            0.0,
            min(10.0, scheduler_max_deferral_windows),
        )
        try:
            backfill_spacing = float(
                getattr(config, "LUXRIOT_ROLLUP_BACKFILL_SPACING_SEC", 10.0)
            )
        except Exception:
            backfill_spacing = 10.0
        self.rollup_backfill_spacing_sec = max(1.0, min(300.0, backfill_spacing))
        try:
            backfill_max_attempts = int(
                getattr(config, "LUXRIOT_ROLLUP_BACKFILL_MAX_ATTEMPTS", 3)
            )
        except Exception:
            backfill_max_attempts = 3
        self.rollup_backfill_max_attempts = max(1, min(10, backfill_max_attempts))
        try:
            backfill_estimate_sec = float(
                getattr(config, "LUXRIOT_ROLLUP_BACKFILL_ESTIMATE_SEC", 45.0)
            )
        except Exception:
            backfill_estimate_sec = 45.0
        self.rollup_backfill_estimate_sec = max(1.0, min(900.0, backfill_estimate_sec))
        self._rollup_scheduler_due: Dict[Tuple[int, str], float] = {}
        self._rollup_scheduler_deferred_since: Dict[Tuple[int, str], float] = {}
        self._rollup_scheduler_stop = threading.Event()
        self._rollup_scheduler_thread: Optional[threading.Thread] = None
        self._rollup_scheduler_status: Dict[str, Any] = {
            "enabled": self.rollup_scheduler_enabled,
            "running": False,
            "jobs_completed": 0,
            "jobs_deferred_for_l0": 0,
            "jobs_forced_after_deferral": 0,
            "backfill_windows_generated": 0,
            "invalid_operator_contract": 0,
            "corrective_retries": 0,
            "corrective_retry_successes": 0,
            "semantic_guard_retries": 0,
            "semantic_guard_retry_successes": 0,
            "semantic_guard_failures": 0,
            "semantic_guard_sanitized": 0,
            "cached_semantic_guard_rejections": 0,
            "last_error": None,
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
        self._rollup_durable_last_prune_at = 0.0
        cache_file_raw = str(getattr(config, "LUXRIOT_ROLLUP_CACHE_FILE", "luxriot_rollups_cache.json") or "").strip()
        if not cache_file_raw:
            cache_file_raw = "luxriot_rollups_cache.json"
        cache_path = Path(cache_file_raw).expanduser()
        if not cache_path.is_absolute():
            cache_path = Path.cwd() / cache_path
        self.rollup_cache_file = cache_path
        self._load_summary_state_from_disk()
        self._summary_state_revision_issued = int(self.summary_state_revision)
        self._load_rollup_cache_from_disk()
        self._rollup_backfill_condition = threading.Condition(threading.RLock())
        self._rollup_backfill_stop = threading.Event()
        self._rollup_backfill_candidate_cache: Dict[Tuple[str, int, str], List[float]] = {}
        self._rollup_backfill_state = self._load_rollup_backfill_state()
        self._rollup_backfill_thread: Optional[threading.Thread] = None
        if runtime_state_store is not None:
            self._rollup_backfill_thread = threading.Thread(
                target=self._rollup_backfill_loop,
                daemon=True,
                name="eva-rollup-backfill",
            )
            self._rollup_backfill_thread.start()
        if self.rollup_scheduler_enabled:
            self._start_rollup_scheduler()

    def _start_rollup_scheduler(self) -> None:
        if self._rollup_scheduler_thread is not None and self._rollup_scheduler_thread.is_alive():
            return
        self._rollup_scheduler_thread = threading.Thread(
            target=self._rollup_scheduler_loop,
            daemon=True,
            name="eva-rollup-scheduler",
        )
        self._rollup_scheduler_thread.start()

    def _rollup_scheduler_channels(self) -> List[int]:
        with self.cache_lock:
            channels = {
                int(channel_id)
                for channel_id, logs in self.summary_history.items()
                if logs
            }
            channels.update(int(channel_id) for channel_id in self.sessions)
        try:
            desired = self._load_desired_live_sessions()
        except Exception:
            desired = {}
        channels.update(
            int(channel_id)
            for channel_id, state in desired.items()
            if bool(state.get("enabled"))
        )
        return sorted(channel_id for channel_id in channels if channel_id > 0)

    def _rollup_initial_due(self, channel_id: int, level: str, now: float, channel_count: int) -> float:
        window_sec = max(1, int(self.rollup_windows[level]))
        spread_sec = min(
            float(window_sec),
            max(60.0, float(max(1, channel_count)) * self.rollup_scheduler_spacing_sec),
        )
        digest = hashlib.sha1(f"{int(channel_id)}:{level}".encode("utf-8")).hexdigest()
        offset = int(digest[:12], 16) % max(1, int(spread_sec))
        return float(now) + self.rollup_scheduler_initial_delay_sec + float(offset)

    def _l0_backpressure_active(self, channel_id: Optional[int] = None) -> bool:
        """Return true only for a saturated queued backlog, not normal inference."""

        with self.cache_lock:
            if channel_id is None:
                sessions = list(self.sessions.values())
            else:
                session = self.sessions.get(int(channel_id))
                sessions = [session] if session is not None else []
        for session in sessions:
            try:
                status = session.status()
            except Exception:
                continue
            queue_depth = int(_parse_optional_int(status.get("summary_queue_depth")) or 0)
            queue_limit = int(_parse_optional_int(status.get("summary_queue_max_batches")) or 0)
            if queue_limit > 0 and queue_depth >= queue_limit:
                return True
        return False

    def _rollup_deferral_exhausted(
        self,
        key: Tuple[int, str],
        level: str,
        now: float,
    ) -> bool:
        max_deferral_sec = (
            float(self.rollup_windows[level])
            * float(self.rollup_scheduler_max_deferral_windows)
        )
        if max_deferral_sec <= 0:
            return True
        first_deferred_at = self._rollup_scheduler_deferred_since.get(key)
        return bool(
            first_deferred_at is not None
            and float(now) - float(first_deferred_at) >= max_deferral_sec
        )

    def _run_scheduled_rollup(self, channel_id: int, level: str, now: float) -> Dict[str, Any]:
        window_sec = max(1, int(self.rollup_windows[level]))
        closed_window_end = int(float(now) // window_sec) * window_sec
        if closed_window_end <= 0:
            return {"levels": {}, "source_counts": {}}
        return self.summary_rollups(
            channel_id=int(channel_id),
            run_selector="all",
            # Scan all retained hot L0 context. Cached windows are free; the
            # scheduler-specific generation budget drains newest missing
            # windows first without turning a restart into an LM stampede.
            start_ts=None,
            end_ts=float(closed_window_end) - 0.001,
            level_limit=None,
            synthesize=True,
            target_level=level,
            synthesize_levels={level},
            max_new_per_level=self.rollup_scheduler_backfill_windows,
        )

    def _rollup_scheduler_loop(self) -> None:
        while not self._rollup_scheduler_stop.is_set():
            now = time.time()
            channels = self._rollup_scheduler_channels()
            for channel_id in channels:
                for level in ("L1", "L2", "L3"):
                    key = (int(channel_id), level)
                    self._rollup_scheduler_due.setdefault(
                        key,
                        self._rollup_initial_due(channel_id, level, now, len(channels)),
                    )
            due_items = [
                (due_at, key)
                for key, due_at in self._rollup_scheduler_due.items()
                if key[0] in channels
            ]
            if not due_items:
                self._rollup_scheduler_stop.wait(10.0)
                continue
            due_at, (channel_id, level) = min(due_items, key=lambda item: item[0])
            wait_sec = float(due_at) - time.time()
            if wait_sec > 0:
                self._rollup_scheduler_stop.wait(min(10.0, wait_sec))
                continue
            scheduler_key = (channel_id, level)
            l0_backpressure = self._l0_backpressure_active(channel_id)
            deferral_exhausted = bool(
                l0_backpressure
                and self._rollup_deferral_exhausted(
                    scheduler_key,
                    level,
                    time.time(),
                )
            )
            if l0_backpressure and not deferral_exhausted:
                self._rollup_scheduler_deferred_since.setdefault(scheduler_key, time.time())
                self._rollup_scheduler_due[(channel_id, level)] = time.time() + max(
                    15.0,
                    self.rollup_scheduler_spacing_sec,
                )
                with self.cache_lock:
                    self._rollup_scheduler_status["jobs_deferred_for_l0"] = int(
                        self._rollup_scheduler_status.get("jobs_deferred_for_l0") or 0
                    ) + 1
                    self._rollup_scheduler_status["last_deferred_channel_id"] = int(channel_id)
                    self._rollup_scheduler_status["last_deferred_level"] = level
                    self._rollup_scheduler_status["last_deferred_at"] = time.time()
                continue
            self._rollup_scheduler_deferred_since.pop(scheduler_key, None)
            if deferral_exhausted:
                with self.cache_lock:
                    self._rollup_scheduler_status["jobs_forced_after_deferral"] = int(
                        self._rollup_scheduler_status.get("jobs_forced_after_deferral") or 0
                    ) + 1

            started_at = time.time()
            with self.cache_lock:
                self._rollup_scheduler_status.update(
                    {
                        "running": True,
                        "active_channel_id": int(channel_id),
                        "active_level": level,
                        "last_started_at": started_at,
                        "last_error": None,
                    }
                )
            error: Optional[str] = None
            generated_windows = 0
            try:
                result = self._run_scheduled_rollup(channel_id, level, started_at)
                levels_raw = result.get("levels") if isinstance(result, Mapping) else None
                rows_raw = levels_raw.get(level) if isinstance(levels_raw, Mapping) else None
                rows = rows_raw if isinstance(rows_raw, Sequence) and not isinstance(rows_raw, (str, bytes, bytearray)) else []
                generated_windows = sum(
                    1
                    for row in rows
                    if isinstance(row, Mapping)
                    and str(row.get("summary_kind") or "").strip().lower() == "llm"
                )
            except Exception as exc:
                error = _safe_error_text(exc, 240) or exc.__class__.__name__
                LOGGER.warning(
                    "Scheduled Luxriot rollup failed channel_id=%s level=%s error=%s",
                    channel_id,
                    level,
                    error,
                )
            completed_at = time.time()
            next_due = float(due_at) + float(self.rollup_windows[level])
            while next_due <= completed_at:
                next_due += float(self.rollup_windows[level])
            self._rollup_scheduler_due[(channel_id, level)] = next_due
            with self.cache_lock:
                self._rollup_scheduler_status.update(
                    {
                        "running": False,
                        "active_channel_id": None,
                        "active_level": None,
                        "last_completed_at": completed_at,
                        "last_duration_sec": round(max(0.0, completed_at - started_at), 3),
                        "last_channel_id": int(channel_id),
                        "last_level": level,
                        "last_generated_windows": int(generated_windows),
                        "last_error": error,
                        "backfill_windows_generated": int(
                            self._rollup_scheduler_status.get("backfill_windows_generated") or 0
                        ) + int(generated_windows),
                        "jobs_completed": int(
                            self._rollup_scheduler_status.get("jobs_completed") or 0
                        ) + (0 if error else 1),
                    }
                )
            self._rollup_scheduler_stop.wait(self.rollup_scheduler_spacing_sec)

    def set_summary_archive_readers(
        self,
        history_loader: Optional[SummaryArchiveHistoryLoaderFn],
        bucket_loader: Optional[SummaryArchiveBucketLoaderFn],
    ) -> None:
        self.summary_archive_history_loader = history_loader
        self.summary_archive_bucket_loader = bucket_loader
        with self._rollup_backfill_condition:
            self._rollup_backfill_condition.notify_all()

    def _load_rollup_backfill_state(self) -> Dict[str, Any]:
        store = getattr(self, "runtime_state_store", None)
        if store is None:
            return {}
        try:
            payload = store.load_state(self.ROLLUP_BACKFILL_STATE_KEY)
        except Exception:
            return {}
        if not isinstance(payload, Mapping):
            return {}
        state = dict(payload)
        if str(state.get("status") or "") in {"running", "waiting_live", "retrying"}:
            state["status"] = "queued"
            state["resume_reason"] = "process_restart"
        return state

    def _persist_rollup_backfill_state_locked(self) -> None:
        store = getattr(self, "runtime_state_store", None)
        if store is None or not self._rollup_backfill_state:
            return
        self._rollup_backfill_state["updated_at"] = time.time()
        store.save_state(
            self.ROLLUP_BACKFILL_STATE_KEY,
            dict(self._rollup_backfill_state),
        )

    @staticmethod
    def _backfill_terminal_status(status: object) -> bool:
        return str(status or "").strip().lower() in {
            "completed",
            "completed_with_gaps",
            "cancelled",
            "failed",
        }

    def _backfill_channel_ids(self, channel_ids: Optional[Sequence[int]]) -> List[int]:
        parsed = sorted(
            {
                int(channel_id)
                for channel_id in (channel_ids or [])
                if _parse_optional_int(channel_id) is not None and int(channel_id) > 0
            }
        )
        if parsed:
            return parsed
        try:
            inventory = self.get_channels(force=False)
        except Exception:
            inventory = []
        parsed = sorted(
            {
                int(channel_id)
                for channel_id in (
                    _parse_optional_int(row.get("id"))
                    for row in inventory
                    if isinstance(row, Mapping)
                )
                if channel_id is not None and channel_id > 0
            }
        )
        return parsed or self._rollup_scheduler_channels()

    def _backfill_candidate_starts(
        self,
        job_id: str,
        channel_id: int,
        level: str,
        start_ts: float,
        end_ts: float,
    ) -> List[float]:
        normalized_level = self._normalize_rollup_level(level)
        cache_key = (str(job_id), int(channel_id), normalized_level)
        cached = self._rollup_backfill_candidate_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        loader = self.summary_archive_bucket_loader
        if not callable(loader):
            return []
        l1_window = int(self.rollup_windows["L1"])
        rows = loader(int(channel_id), float(start_ts), float(end_ts), l1_window)
        l1_starts = sorted(
            {
                float(self._bucket_start(float(row.get("window_start")), l1_window))
                for row in rows
                if isinstance(row, Mapping)
                and self._coerce_float(row.get("window_start")) is not None
            }
        )
        window_sec = int(self.rollup_windows[normalized_level])
        closed_end = min(float(end_ts), time.time())
        starts = sorted(
            {
                float(self._bucket_start(start, window_sec))
                for start in l1_starts
                if float(self._bucket_start(start, window_sec)) + window_sec <= closed_end
                and float(self._bucket_start(start, window_sec)) + window_sec > float(start_ts)
            }
        )
        self._rollup_backfill_candidate_cache[cache_key] = starts
        return list(starts)

    def _rollup_semantic_ready(self, row: Optional[Mapping[str, Any]]) -> bool:
        if not isinstance(row, Mapping):
            return False
        kind = str(row.get("summary_kind") or "").strip().lower()
        format_version = int(_parse_optional_int(row.get("format_version")) or 1)
        summary = str(row.get("summary") or "").strip()
        return bool(
            kind in {"llm", "llm_cached"}
            and format_version >= ROLLUP_OPERATOR_FORMAT_VERSION
            and summary
            and not self._rollup_operator_semantic_guard_issues(summary)
        )

    def _backfill_ready_ids(
        self,
        channel_id: int,
        start_ts: float,
        end_ts: float,
    ) -> Set[str]:
        return {
            str(row.get("rollup_id") or "").strip()
            for row in self._list_cached_rollups(channel_id, start_ts, end_ts)
            if self._rollup_semantic_ready(row)
            and str(row.get("rollup_id") or "").strip()
        }

    @staticmethod
    def _closed_window_count(start_ts: float, end_ts: float, window_sec: int) -> int:
        first = int(math.floor(float(start_ts) / window_sec) * window_sec)
        closed_end = int(math.floor(min(float(end_ts), time.time()) / window_sec) * window_sec)
        if closed_end <= first:
            return 0
        return max(0, int((closed_end - first) // window_sec))

    def plan_rollup_backfill(
        self,
        *,
        channel_ids: Optional[Sequence[int]],
        start_ts: float,
        end_ts: float,
        levels: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        if not callable(self.summary_archive_bucket_loader):
            raise RuntimeError("archive-backed summary coverage reader is unavailable")
        normalized_channels = self._backfill_channel_ids(channel_ids)
        if not normalized_channels:
            raise ValueError("no channels are available for summary restoration")
        normalized_levels = [
            level
            for level in ("L1", "L2", "L3")
            if level in {
                self._normalize_rollup_level(item)
                for item in (levels or ("L2", "L3"))
            }
        ]
        if not normalized_levels:
            raise ValueError("levels must contain L1, L2, or L3")
        start = max(0.0, min(float(start_ts), float(end_ts)))
        end = max(float(start_ts), float(end_ts))
        audit_id = f"audit-{uuid4().hex[:12]}"
        totals = {
            "calendar_windows": 0,
            "source_windows": 0,
            "source_missing_windows": 0,
            "already_ready": 0,
            "missing_semantic": 0,
        }
        per_channel: List[Dict[str, Any]] = []
        for channel_id in normalized_channels:
            ready_ids = self._backfill_ready_ids(channel_id, start, end)
            level_rows: Dict[str, Dict[str, int]] = {}
            for level in normalized_levels:
                window_sec = int(self.rollup_windows[level])
                candidates = self._backfill_candidate_starts(
                    audit_id,
                    channel_id,
                    level,
                    start,
                    end,
                )
                ready = sum(
                    1
                    for window_start in candidates
                    if self._canonical_rollup_id(level, channel_id, window_start, window_sec)
                    in ready_ids
                )
                calendar = self._closed_window_count(start, end, window_sec)
                source_count = len(candidates)
                missing = max(0, source_count - ready)
                level_rows[level] = {
                    "calendar_windows": calendar,
                    "source_windows": source_count,
                    "source_missing_windows": max(0, calendar - source_count),
                    "already_ready": ready,
                    "missing_semantic": missing,
                }
                for key in totals:
                    totals[key] += int(level_rows[level][key])
            per_channel.append({"channel_id": channel_id, "levels": level_rows})
        channels_with_source = sum(
            1
            for row in per_channel
            if any(
                int(level_row.get("source_windows") or 0) > 0
                for level_row in (row.get("levels") or {}).values()
                if isinstance(level_row, Mapping)
            )
        )
        queueable_channels = sum(
            1
            for row in per_channel
            if any(
                int(level_row.get("missing_semantic") or 0) > 0
                for level_row in (row.get("levels") or {}).values()
                if isinstance(level_row, Mapping)
            )
        )
        with self._rollup_backfill_condition:
            current_avg = self._coerce_float(
                self._rollup_backfill_state.get("average_window_sec")
            )
        estimate_per_window = max(
            1.0,
            current_avg or self.rollup_backfill_estimate_sec,
        ) + self.rollup_backfill_spacing_sec
        estimated_sec = float(totals["missing_semantic"]) * estimate_per_window
        request_payload = {
            "channel_ids": normalized_channels,
            "from_bucket": int(start),
            "to_bucket": int(end),
            "levels": normalized_levels,
        }
        request_key = hashlib.sha256(
            json.dumps(request_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:24]
        self._rollup_backfill_candidate_cache = {
            key: value
            for key, value in self._rollup_backfill_candidate_cache.items()
            if key[0] != audit_id
        }
        return {
            "status": "preview",
            "request_key": request_key,
            "channel_ids": normalized_channels,
            "channel_count": len(normalized_channels),
            "levels": normalized_levels,
            "from_ts": start,
            "to_ts": end,
            "archive_source": "vlm_summary batch text",
            "totals": totals,
            "restoration_scope": {
                "queueable_windows": int(totals["missing_semantic"]),
                "already_semantic_windows": int(totals["already_ready"]),
                "archived_source_windows": int(totals["source_windows"]),
                "not_restorable_no_archived_source": int(totals["source_missing_windows"]),
                "calendar_windows": int(totals["calendar_windows"]),
                "channels_with_source": channels_with_source,
                "channels_without_source": max(0, len(normalized_channels) - channels_with_source),
                "queueable_channels": queueable_channels,
                "queue_contract": (
                    "Only queueable_windows are submitted to the worker and included in ETA. "
                    "not_restorable_no_archived_source is a coverage gap, not queued work."
                ),
            },
            "per_channel": per_channel,
            "estimated_window_sec": round(estimate_per_window, 2),
            "estimated_seconds": round(estimated_sec, 1),
            "estimated_hours": round(estimated_sec / 3600.0, 2),
            "estimated_hours_range": [
                round(estimated_sec * 0.6 / 3600.0, 2),
                round(estimated_sec * 1.8 / 3600.0, 2),
            ],
            "load_policy": "single background worker; live backlog wins; LM workload=background",
            "idempotent": True,
        }

    def start_rollup_backfill(
        self,
        *,
        channel_ids: Optional[Sequence[int]],
        start_ts: float,
        end_ts: float,
        levels: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        plan = self.plan_rollup_backfill(
            channel_ids=channel_ids,
            start_ts=start_ts,
            end_ts=end_ts,
            levels=levels,
        )
        with self._rollup_backfill_condition:
            current = dict(self._rollup_backfill_state)
            if current and not self._backfill_terminal_status(current.get("status")):
                result = self.rollup_backfill_status()
                result["idempotent_existing_job"] = True
                return result
            if (
                current
                and current.get("request_key") == plan.get("request_key")
                and str(current.get("status") or "") in {"completed", "completed_with_gaps"}
            ):
                result = self.rollup_backfill_status()
                result["idempotent_existing_job"] = True
                return result
            now = time.time()
            self._rollup_backfill_state = {
                "version": 1,
                "job_id": f"rollup-backfill-{uuid4().hex[:12]}",
                "request_key": plan["request_key"],
                "status": "queued",
                "created_at": now,
                "updated_at": now,
                "started_at": None,
                "completed_at": None,
                "from_ts": plan["from_ts"],
                "to_ts": plan["to_ts"],
                "channel_ids": list(plan["channel_ids"]),
                "levels": list(plan["levels"]),
                "plan": {
                    "totals": dict(plan["totals"]),
                    "estimated_seconds": plan["estimated_seconds"],
                    "estimated_hours": plan["estimated_hours"],
                    "estimated_hours_range": list(plan["estimated_hours_range"]),
                },
                "cursor": {
                    "level_index": 0,
                    "channel_index": 0,
                    "after_window_start": None,
                    "attempt": 0,
                },
                "progress": {
                    "processed": 0,
                    "restored": 0,
                    "already_ready": 0,
                    "source_missing": 0,
                    "failed": 0,
                    "retries": 0,
                },
                "average_window_sec": None,
                "last_error": None,
                "current_item": None,
            }
            self._rollup_backfill_candidate_cache.clear()
            self._persist_rollup_backfill_state_locked()
            self._rollup_backfill_condition.notify_all()
        return self.rollup_backfill_status()

    def rollup_backfill_status(self) -> Dict[str, Any]:
        with self._rollup_backfill_condition:
            state = copy.deepcopy(self._rollup_backfill_state)
        if not state:
            return {
                "status": "not_started",
                "worker_alive": bool(
                    self._rollup_backfill_thread
                    and self._rollup_backfill_thread.is_alive()
                ),
            }
        plan = state.get("plan") if isinstance(state.get("plan"), Mapping) else {}
        totals = plan.get("totals") if isinstance(plan.get("totals"), Mapping) else {}
        progress = state.get("progress") if isinstance(state.get("progress"), Mapping) else {}
        target = int(totals.get("missing_semantic") or 0)
        processed = int(progress.get("processed") or 0)
        remaining = max(0, target - processed)
        average = self._coerce_float(state.get("average_window_sec")) or self.rollup_backfill_estimate_sec
        eta_sec = remaining * (average + self.rollup_backfill_spacing_sec)
        completed = str(state.get("status") or "") in {"completed", "completed_with_gaps"}
        if completed:
            remaining = 0
        state["remaining"] = remaining
        state["progress_percent"] = round(
            100.0 if completed or target <= 0 else min(100.0, processed * 100.0 / target),
            2,
        )
        state["eta_seconds"] = round(eta_sec, 1)
        state["eta_hours"] = round(eta_sec / 3600.0, 2)
        state["worker_alive"] = bool(
            self._rollup_backfill_thread
            and self._rollup_backfill_thread.is_alive()
        )
        state["durable"] = getattr(self, "runtime_state_store", None) is not None
        return state

    def _next_rollup_backfill_item_locked(self) -> Optional[Dict[str, Any]]:
        state = self._rollup_backfill_state
        levels = [str(level) for level in state.get("levels") or []]
        channels = [int(channel_id) for channel_id in state.get("channel_ids") or []]
        cursor = state.setdefault("cursor", {})
        while int(cursor.get("level_index") or 0) < len(levels):
            level_index = int(cursor.get("level_index") or 0)
            channel_index = int(cursor.get("channel_index") or 0)
            if channel_index >= len(channels):
                cursor["level_index"] = level_index + 1
                cursor["channel_index"] = 0
                cursor["after_window_start"] = None
                cursor["attempt"] = 0
                self._persist_rollup_backfill_state_locked()
                continue
            level = levels[level_index]
            channel_id = channels[channel_index]
            starts = self._backfill_candidate_starts(
                str(state.get("job_id") or ""),
                channel_id,
                level,
                float(state.get("from_ts") or 0.0),
                float(state.get("to_ts") or time.time()),
            )
            after = self._coerce_float(cursor.get("after_window_start"))
            window_sec = int(self.rollup_windows[level])
            for window_start in starts:
                if after is not None and window_start <= after:
                    continue
                rollup_id = self._canonical_rollup_id(level, channel_id, window_start, window_sec)
                if self._rollup_semantic_ready(self._get_cached_rollup_record(rollup_id)):
                    cursor["after_window_start"] = window_start
                    progress = state.setdefault("progress", {})
                    progress["already_ready"] = int(progress.get("already_ready") or 0) + 1
                    continue
                return {
                    "channel_id": channel_id,
                    "level": level,
                    "window_start": window_start,
                    "window_end": window_start + window_sec,
                    "rollup_id": rollup_id,
                }
            cursor["channel_index"] = channel_index + 1
            cursor["after_window_start"] = None
            cursor["attempt"] = 0
            self._persist_rollup_backfill_state_locked()
        return None

    def _restore_rollup_window(self, item: Mapping[str, Any]) -> Dict[str, Any]:
        channel_id = int(item["channel_id"])
        level = self._normalize_rollup_level(item.get("level"))
        window_start = float(item["window_start"])
        window_end = float(item["window_end"])
        rollup_id = str(item["rollup_id"])
        cached = self._get_cached_rollup_record(rollup_id)
        if self._rollup_semantic_ready(cached):
            return {"status": "already_ready", "rollup_id": rollup_id}
        if level == "L1":
            loader = self.summary_archive_history_loader
            if not callable(loader):
                raise RuntimeError("archive-backed summary history reader is unavailable")
            logs, total = loader(channel_id, window_start, window_end - 0.001)
            if not logs:
                return {"status": "source_missing", "rollup_id": rollup_id, "source_count": int(total)}
            source_nodes = self._l0_nodes_from_logs(channel_id, logs)
            source_level = "L0"
        else:
            source_level = "L1" if level == "L2" else "L2"
            source_nodes = [
                row
                for row in self._list_cached_rollups(channel_id, window_start, window_end - 0.001)
                if self._normalize_rollup_level(row.get("level")) == source_level
                and self._rollup_semantic_ready(row)
            ]
            if not source_nodes and level == "L2":
                loader = self.summary_archive_history_loader
                if callable(loader):
                    logs, _total = loader(channel_id, window_start, window_end - 0.001)
                    source_nodes = self._l0_nodes_from_logs(channel_id, logs)
                    source_level = "L0"
            if not source_nodes:
                return {"status": "source_missing", "rollup_id": rollup_id, "source_count": 0}
        rows = self._build_rollup_level(
            channel_id=channel_id,
            level=level,
            source_level=source_level,
            window_sec=int(self.rollup_windows[level]),
            source_nodes=source_nodes,
            synthesize=True,
            max_new=1,
            workload_class="background",
        )
        restored = next(
            (
                row
                for row in rows
                if str(row.get("rollup_id") or "") == rollup_id
            ),
            None,
        )
        if self._rollup_semantic_ready(restored):
            return {
                "status": "restored",
                "rollup_id": rollup_id,
                "source_count": len(source_nodes),
            }
        error = str((restored or {}).get("generation_error") or "semantic generation did not complete")
        raise RuntimeError(error)

    def _rollup_backfill_loop(self) -> None:
        while not self._rollup_backfill_stop.is_set():
            with self._rollup_backfill_condition:
                status = str(self._rollup_backfill_state.get("status") or "")
                if not self._rollup_backfill_state or self._backfill_terminal_status(status):
                    self._rollup_backfill_condition.wait(timeout=10.0)
                    continue
                if not callable(self.summary_archive_bucket_loader) or not callable(self.summary_archive_history_loader):
                    self._rollup_backfill_state["status"] = "waiting_source_reader"
                    try:
                        self._persist_rollup_backfill_state_locked()
                    except Exception:
                        pass
                    self._rollup_backfill_condition.wait(timeout=10.0)
                    continue
                if self._rollup_backfill_state.get("started_at") is None:
                    self._rollup_backfill_state["started_at"] = time.time()
                self._rollup_backfill_state["status"] = "running"
                try:
                    item = self._next_rollup_backfill_item_locked()
                except Exception as exc:
                    self._rollup_backfill_state["status"] = "failed"
                    self._rollup_backfill_state["last_error"] = _safe_error_text(exc, 240) or exc.__class__.__name__
                    self._rollup_backfill_state["completed_at"] = time.time()
                    try:
                        self._persist_rollup_backfill_state_locked()
                    except Exception:
                        pass
                    continue
                if item is None:
                    progress = self._rollup_backfill_state.get("progress") or {}
                    final_status = "completed_with_gaps" if int(progress.get("failed") or 0) or int(progress.get("source_missing") or 0) else "completed"
                    self._rollup_backfill_state["status"] = final_status
                    self._rollup_backfill_state["completed_at"] = time.time()
                    self._rollup_backfill_state["current_item"] = None
                    self._persist_rollup_backfill_state_locked()
                    continue
                if self._l0_backpressure_active():
                    self._rollup_backfill_state["status"] = "waiting_live"
                    self._rollup_backfill_state["current_item"] = item
                    self._persist_rollup_backfill_state_locked()
                    self._rollup_backfill_condition.wait(timeout=max(10.0, self.rollup_backfill_spacing_sec))
                    continue
                self._rollup_backfill_state["current_item"] = item
                self._persist_rollup_backfill_state_locked()
            started = time.monotonic()
            try:
                outcome = self._restore_rollup_window(item)
                error: Optional[str] = None
            except Exception as exc:
                outcome = {"status": "failed", "rollup_id": item.get("rollup_id")}
                error = _safe_error_text(exc, 240) or exc.__class__.__name__
            elapsed = max(0.0, time.monotonic() - started)
            wait_sec = self.rollup_backfill_spacing_sec
            with self._rollup_backfill_condition:
                state = self._rollup_backfill_state
                cursor = state.setdefault("cursor", {})
                progress = state.setdefault("progress", {})
                outcome_status = str(outcome.get("status") or "failed")
                if outcome_status == "failed":
                    attempt = int(cursor.get("attempt") or 0) + 1
                    cursor["attempt"] = attempt
                    progress["retries"] = int(progress.get("retries") or 0) + 1
                    state["last_error"] = error
                    if attempt < self.rollup_backfill_max_attempts:
                        state["status"] = "retrying"
                        wait_sec = max(wait_sec, min(300.0, 15.0 * (2 ** (attempt - 1))))
                    else:
                        progress["failed"] = int(progress.get("failed") or 0) + 1
                        progress["processed"] = int(progress.get("processed") or 0) + 1
                        cursor["after_window_start"] = float(item["window_start"])
                        cursor["attempt"] = 0
                        state["status"] = "running"
                else:
                    progress["processed"] = int(progress.get("processed") or 0) + 1
                    if outcome_status == "restored":
                        progress["restored"] = int(progress.get("restored") or 0) + 1
                    elif outcome_status == "already_ready":
                        progress["already_ready"] = int(progress.get("already_ready") or 0) + 1
                    elif outcome_status == "source_missing":
                        progress["source_missing"] = int(progress.get("source_missing") or 0) + 1
                    cursor["after_window_start"] = float(item["window_start"])
                    cursor["attempt"] = 0
                    state["status"] = "running"
                    state["last_error"] = None
                    previous_avg = self._coerce_float(state.get("average_window_sec"))
                    state["average_window_sec"] = round(
                        elapsed if previous_avg is None else previous_avg * 0.8 + elapsed * 0.2,
                        3,
                    )
                state["last_outcome"] = dict(outcome)
                state["current_item"] = None
                try:
                    self._persist_rollup_backfill_state_locked()
                except Exception as exc:
                    state["status"] = "failed"
                    state["last_error"] = _safe_error_text(exc, 240) or exc.__class__.__name__
                    state["completed_at"] = time.time()
                self._rollup_backfill_condition.wait(timeout=wait_sec)

    def _session_side_effect_lock_for(self, channel_id: int) -> Any:
        channel_key = int(channel_id)
        with self._session_generation_guard:
            lock = self._session_side_effect_locks.get(channel_key)
            if lock is None:
                lock = threading.RLock()
                self._session_side_effect_locks[channel_key] = lock
            return lock

    def _advance_session_generation(self, channel_id: int) -> str:
        channel_key = int(channel_id)
        with self._session_generation_guard:
            counter = int(self._session_generation_counters.get(channel_key, 0)) + 1
            self._session_generation_counters[channel_key] = counter
            generation = f"{self._session_generation_epoch}:{counter}"
            self._session_generations[channel_key] = generation
            return generation

    def _current_session_generation(self, channel_id: int) -> Optional[str]:
        with self._session_generation_guard:
            return self._session_generations.get(int(channel_id))

    def _summary_entry_stale_reason(self, entry: Mapping[str, Any]) -> Optional[str]:
        channel_id = _parse_optional_int(entry.get("channel_id"))
        if channel_id is None:
            return "missing channel id"
        incoming_generation = str(entry.get("session_generation") or "").strip()
        current_generation = self._current_session_generation(channel_id)
        with self.cache_lock:
            current_session = self.sessions.get(int(channel_id))
        if incoming_generation and incoming_generation != str(current_generation or ""):
            return "session generation was superseded"
        if current_session is not None:
            session_generation = str(current_session.session_generation or "").strip()
            if incoming_generation and session_generation and incoming_generation != session_generation:
                return "session instance was superseded"
            incoming_run_id = str(entry.get("run_id") or "").strip()
            if incoming_run_id and incoming_run_id != str(current_session.run_id or "").strip():
                return "summary run was superseded"
        return None

    def _assert_summary_batch_current(self, batch: Mapping[str, Any]) -> None:
        stale_reason = self._summary_entry_stale_reason(batch)
        if stale_reason:
            raise SummaryBatchSuperseded(stale_reason)

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
        level_focus = {
            "L1": "Describe the behavior and changes across this 15-minute window as one short sequence.",
            "L2": "Describe hour-scale episodes, routine shifts, meaningful recurrence, and unresolved exceptions.",
            "L3": "Describe the longer operational pattern, repeated behavior, unresolved incidents, and coverage quality.",
        }.get(normalized_level, "Describe behavior and change across the complete period.")
        return [
            "Context constraints:",
            "- All source entries are from the same channel and continuous timeline window above.",
            "- Source entries may be model-generated summaries from a lower level; avoid compounding uncertainty.",
            "- Window signal digest and channel memory are internal routing context, not operator prose and not independent evidence.",
            "- Preserve rare but important events even if they appear once.",
            "- Keep numeric facts aligned with metadata above (item_count/frame_count/window).",
            "- Never compress alerts, deviations, coverage gaps, or operator-review incidents into routine.",
            "- Do not classify behavior as illegal/unlawful; describe observable security/safety facts.",
            "- Never infer intent, motive, identity, skill, or blame, and never ask the operator to confirm intent.",
            "- Sampled snapshots cannot prove complete scene coverage, absence outside the sampled frames, or the absence of blind spots.",
            "- If source prose and structured alerts conflict, state the conflict briefly instead of silently choosing one claim.",
            "",
            "Task:",
            f"- Write one readable {normalized_level} behavioral summary for a municipal CCTV operator.",
            f"- {level_focus}",
            "- Synthesize behavior over time; do not enumerate, concatenate, or paraphrase every source batch.",
            "- Use prose for the period narrative. Use bullets only for distinct observations, alerts, interruptions, or follow-up items.",
            "- Deduplicate repeated scene descriptions, names, boilerplate, and unchanged background.",
            "- Explain each alert's observable meaning and outcome; do not merely repeat severity counters.",
            "- Report camera/feed interruptions and missing coverage separately from observed behavior.",
            "- Distinguish 'no interruption recorded in metadata' from a claim that visual coverage was complete.",
            "- Recommend operator follow-up only for a grounded unresolved safety/security issue, not routine presence changes or low-confidence cues.",
            "- If the period is routine, say so plainly and keep the report short.",
            "- Do not invent entities, times, or counts.",
            "",
            "Operator output format (Markdown, use exactly these sections):",
            "### Period Overview",
            "### Routine and Behavior",
            "### Notable Observations and Exceptions",
            "### Alerts and Meaning",
            "### Coverage and Interruptions",
            "### Operator Takeaway",
            "Do not expose signal digests, prompt tuning, watchlist mechanics, memory terminology, source tokens, or internal detector plumbing in these sections.",
            "",
            "After the operator sections, append exactly one compact machine-readable block for EVA internal use. It is not shown to operators:",
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
            "source_frame_count",
            "selected_frame_count",
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
    def _compact_frame_selection(cls, value: object) -> Dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        groups: List[Dict[str, Any]] = []
        raw_groups = value.get("groups")
        if isinstance(raw_groups, Sequence) and not isinstance(raw_groups, (str, bytes, bytearray)):
            for raw_group in raw_groups[:256]:
                if not isinstance(raw_group, Mapping):
                    continue
                selected_index = _parse_optional_int(raw_group.get("selected_source_frame_index"))
                selection_source = str(raw_group.get("selection_source") or "").strip().lower()[:80]
                if selected_index is None or selected_index < 1 or not selection_source:
                    continue
                group: Dict[str, Any] = {
                    "selected_source_frame_index": int(selected_index),
                    "selection_source": selection_source,
                    "apex_available": bool(raw_group.get("apex_available")),
                }
                for key in ("bucket_start_ms", "selected_timestamp_ms"):
                    parsed = _parse_optional_int(raw_group.get(key))
                    if parsed is not None:
                        group[key] = int(parsed)
                source_indices: List[int] = []
                raw_indices = raw_group.get("source_frame_indices")
                if isinstance(raw_indices, Sequence) and not isinstance(raw_indices, (str, bytes, bytearray)):
                    for raw_index in raw_indices[:64]:
                        parsed = _parse_optional_int(raw_index)
                        if parsed is not None and parsed > 0:
                            source_indices.append(int(parsed))
                if source_indices:
                    group["source_frame_indices"] = source_indices
                source_timestamps: List[Optional[int]] = []
                raw_timestamps = raw_group.get("source_timestamps_ms")
                if isinstance(raw_timestamps, Sequence) and not isinstance(raw_timestamps, (str, bytes, bytearray)):
                    for raw_timestamp in raw_timestamps[:64]:
                        parsed = _parse_optional_int(raw_timestamp)
                        source_timestamps.append(int(parsed) if parsed is not None else None)
                if source_timestamps:
                    group["source_timestamps_ms"] = source_timestamps
                source_hashes: List[str] = []
                raw_hashes = raw_group.get("source_frame_hashes")
                if isinstance(raw_hashes, Sequence) and not isinstance(raw_hashes, (str, bytes, bytearray)):
                    source_hashes = [str(value or "")[:40] for value in raw_hashes[:64]]
                if source_hashes:
                    group["source_frame_hashes"] = source_hashes
                selected_frame_hash = str(raw_group.get("selected_frame_hash") or "").strip()[:40]
                if selected_frame_hash:
                    group["selected_frame_hash"] = selected_frame_hash
                score = cls._finite_float(raw_group.get("selection_score"))
                if score is not None:
                    group["selection_score"] = round(float(score), 6)
                fallback_reason = str(raw_group.get("fallback_reason") or "").strip().lower()[:160]
                if fallback_reason:
                    group["fallback_reason"] = fallback_reason
                score_source = str(raw_group.get("score_source") or "").strip().lower()[:80]
                if score_source:
                    group["score_source"] = score_source
                references: List[Dict[str, Any]] = []
                raw_references = raw_group.get("signal_references")
                if isinstance(raw_references, Sequence) and not isinstance(raw_references, (str, bytes, bytearray)):
                    for raw_reference in raw_references[:8]:
                        if not isinstance(raw_reference, Mapping):
                            continue
                        source = str(raw_reference.get("source") or "").strip().lower()[:80]
                        if not source:
                            continue
                        reference: Dict[str, Any] = {"source": source}
                        for key in ("source_frame_index", "timestamp_ms"):
                            parsed = _parse_optional_int(raw_reference.get(key))
                            if parsed is not None:
                                reference[key] = int(parsed)
                        reference_score = cls._finite_float(raw_reference.get("score"))
                        if reference_score is not None:
                            reference["score"] = round(float(reference_score), 6)
                        label = str(raw_reference.get("label") or "").strip()[:120]
                        if label:
                            reference["label"] = label
                        score_source = str(raw_reference.get("score_source") or "").strip().lower()[:80]
                        if score_source:
                            reference["score_source"] = score_source
                        references.append(reference)
                if references:
                    group["signal_references"] = references
                groups.append(group)
        if not groups:
            return {}
        out: Dict[str, Any] = {
            "version": int(_parse_optional_int(value.get("version")) or 1),
            "policy": str(value.get("policy") or "per_second_attention_apex_v1").strip()[:120],
            "time_bucket_ms": int(_parse_optional_int(value.get("time_bucket_ms")) or 1000),
            "groups": groups,
        }
        for key in (
            "source_frame_count",
            "selected_frame_count",
            "apex_selected_count",
            "fallback_count",
            "single_frame_count",
            "timestamp_unavailable_count",
        ):
            parsed = _parse_optional_int(value.get(key))
            if parsed is not None:
                out[key] = max(0, int(parsed))
        selection_sources = cls._compact_count_breakdown(value.get("selection_sources"))
        if selection_sources:
            out["selection_sources"] = selection_sources
        return out

    @classmethod
    def _attention_frame_candidates(
        cls,
        frames: Sequence[Mapping[str, Any]],
        vector_signal: object,
    ) -> Dict[int, List[Dict[str, Any]]]:
        if not isinstance(vector_signal, Mapping):
            return {}
        candidates: Dict[int, List[Dict[str, Any]]] = {}

        def target_index(raw: Mapping[str, Any], *, timestamp_key: str = "timestamp_ms") -> Optional[int]:
            for key in ("source_frame_index", "frame_index", "apex_frame"):
                parsed = _parse_optional_int(raw.get(key))
                if parsed is not None and 1 <= parsed <= len(frames):
                    return int(parsed)
            timestamp_ms = _parse_optional_int(raw.get(timestamp_key))
            return cls._nearest_batch_frame_index(frames, timestamp_ms)

        def add_candidate(
            raw: Mapping[str, Any],
            *,
            source: str,
            priority: int,
            score_keys: Sequence[str],
            timestamp_key: str = "timestamp_ms",
            label_key: Optional[str] = None,
        ) -> None:
            index = target_index(raw, timestamp_key=timestamp_key)
            if index is None:
                return
            score: Optional[float] = None
            for key in score_keys:
                score = cls._finite_float(raw.get(key))
                if score is not None:
                    break
            if score is None or score <= 0.0:
                return
            timestamp_ms = _parse_optional_int(raw.get(timestamp_key))
            reference: Dict[str, Any] = {
                "source": source,
                "source_frame_index": int(index),
                "score": float(score),
            }
            if timestamp_ms is not None:
                reference["timestamp_ms"] = int(timestamp_ms)
            if label_key:
                label = str(raw.get(label_key) or "").strip()[:120]
                if label:
                    reference["label"] = label
            score_source = str(raw.get("score_source") or "").strip().lower()[:80]
            if score_source:
                reference["score_source"] = score_source
            candidates.setdefault(int(index), []).append(
                {
                    "source": source,
                    "priority": int(priority),
                    "score": float(score),
                    "reference": reference,
                }
            )

        signal_specs = (
            ("road_cv_cues", "road_cv_cue", 0, ("score",), "timestamp_ms", "cue_type"),
            (
                "road_cv_frame_scores",
                "road_cv_frame_score",
                1,
                ("attention_score", "cue_score", "active_ratio"),
                "timestamp_ms",
                None,
            ),
            ("road_episodes", "road_episode", 2, ("score",), "apex_timestamp_ms", "event_type"),
            ("clip_probe_signals", "clip_probe", 3, ("m", "margin", "p", "pos_score"), "timestamp_ms", "name"),
        )
        for key, source, priority, score_keys, timestamp_key, label_key in signal_specs:
            raw_items = vector_signal.get(key)
            if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes, bytearray)):
                continue
            for raw in raw_items:
                if isinstance(raw, Mapping):
                    add_candidate(
                        raw,
                        source=source,
                        priority=priority,
                        score_keys=score_keys,
                        timestamp_key=timestamp_key,
                        label_key=label_key,
                    )
        return candidates

    @classmethod
    def _select_attention_frames(
        cls,
        frames: Sequence[Mapping[str, Any]],
        vector_signal: object,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        source_frames: List[Dict[str, Any]] = []
        for index, raw_frame in enumerate(frames, start=1):
            frame = dict(raw_frame)
            frame["batch_source_frame_index"] = int(index)
            capture_selection = frame.get("capture_selection")
            capture_source_index = (
                _parse_optional_int(capture_selection.get("selected_source_frame_index"))
                if isinstance(capture_selection, Mapping)
                else None
            )
            frame["source_frame_index"] = int(capture_source_index or index)
            timestamp_ms = cls._batch_frame_timestamp_ms(frame)
            if timestamp_ms is not None:
                frame["source_timestamp_ms"] = int(timestamp_ms)
            source_frames.append(frame)

        candidates = cls._attention_frame_candidates(source_frames, vector_signal)
        bucket_groups: Dict[int, List[Tuple[int, Dict[str, Any], int]]] = {}
        missing_timestamp: List[Tuple[int, Dict[str, Any]]] = []
        for index, frame in enumerate(source_frames, start=1):
            timestamp_ms = cls._batch_frame_timestamp_ms(frame)
            if timestamp_ms is None:
                missing_timestamp.append((index, frame))
                continue
            bucket_start_ms = (int(timestamp_ms) // 1000) * 1000
            bucket_groups.setdefault(bucket_start_ms, []).append((index, frame, int(timestamp_ms)))

        selected_rows: List[Tuple[int, Dict[str, Any]]] = []
        groups: List[Dict[str, Any]] = []
        selection_sources: Dict[str, int] = {}
        apex_selected_count = 0
        fallback_count = 0
        single_frame_count = 0

        for bucket_start_ms, items in bucket_groups.items():
            source_indices = [index for index, _frame, _timestamp in items]
            source_timestamps = [timestamp for _index, _frame, timestamp in items]
            source_hashes: List[str] = []
            fallback_reason = ""
            chosen_candidate: Optional[Dict[str, Any]] = None
            selection_score: Optional[float] = None
            capture_selection = (
                items[0][1].get("capture_selection")
                if len(items) == 1 and isinstance(items[0][1].get("capture_selection"), Mapping)
                else None
            )
            capture_source = (
                str(capture_selection.get("selection_source") or "").strip().lower()
                if isinstance(capture_selection, Mapping)
                else ""
            )
            if len(items) == 1 and capture_source:
                selected_index, selected_frame, selected_timestamp = items[0]
                selection_source = capture_source[:80]
                apex_available = bool(capture_selection.get("apex_available"))
                fallback_reason = str(capture_selection.get("fallback_reason") or "").strip().lower()[:160]
                selection_score = cls._finite_float(capture_selection.get("selection_score"))
                capture_indices = capture_selection.get("source_frame_indices")
                if isinstance(capture_indices, Sequence) and not isinstance(capture_indices, (str, bytes, bytearray)):
                    parsed_indices = [
                        int(parsed)
                        for parsed in (_parse_optional_int(value) for value in capture_indices)
                        if parsed is not None and parsed > 0
                    ]
                    if parsed_indices:
                        source_indices = parsed_indices
                capture_timestamps = capture_selection.get("source_timestamps_ms")
                if isinstance(capture_timestamps, Sequence) and not isinstance(capture_timestamps, (str, bytes, bytearray)):
                    parsed_timestamps = [
                        int(parsed)
                        for parsed in (_parse_optional_int(value) for value in capture_timestamps)
                        if parsed is not None
                    ]
                    if parsed_timestamps:
                        source_timestamps = parsed_timestamps
                capture_hashes = capture_selection.get("source_frame_hashes")
                if isinstance(capture_hashes, Sequence) and not isinstance(capture_hashes, (str, bytes, bytearray)):
                    source_hashes = [str(value or "")[:40] for value in capture_hashes[:64]]
                selected_provenance_index = int(
                    _parse_optional_int(capture_selection.get("selected_source_frame_index"))
                    or selected_frame.get("source_frame_index")
                    or selected_index
                )
                selected_timestamp = int(
                    _parse_optional_int(capture_selection.get("selected_timestamp_ms"))
                    or selected_timestamp
                )
                if apex_available:
                    apex_selected_count += 1
                else:
                    fallback_count += 1
                if len(source_indices) == 1:
                    single_frame_count += 1
            elif len(items) == 1:
                selected_index, selected_frame, selected_timestamp = items[0]
                selected_provenance_index = int(selected_frame.get("source_frame_index") or selected_index)
                selection_source = "single_frame"
                fallback_reason = "single_frame_only_no_intra_second_choice"
                apex_available = False
                single_frame_count += 1
                fallback_count += 1
            else:
                group_candidates: List[Tuple[int, Dict[str, Any]]] = []
                for source_index in source_indices:
                    for candidate in candidates.get(source_index, []):
                        group_candidates.append((source_index, candidate))
                if group_candidates:
                    selected_index, chosen_candidate = min(
                        group_candidates,
                        key=lambda item: (
                            int(item[1].get("priority") or 0),
                            -float(item[1].get("score") or 0.0),
                            int(item[0]),
                        ),
                    )
                    selected_frame = source_frames[selected_index - 1]
                    selected_timestamp = int(cls._batch_frame_timestamp_ms(selected_frame) or bucket_start_ms)
                    selected_provenance_index = int(selected_frame.get("source_frame_index") or selected_index)
                    selection_source = str(chosen_candidate.get("source") or "attention_signal")
                    apex_available = True
                    apex_selected_count += 1
                else:
                    midpoint_ms = (min(source_timestamps) + max(source_timestamps)) / 2.0
                    selected_index, selected_frame, selected_timestamp = min(
                        items,
                        key=lambda item: (abs(float(item[2]) - midpoint_ms), int(item[0])),
                    )
                    selected_provenance_index = int(selected_frame.get("source_frame_index") or selected_index)
                    selection_source = "deterministic_temporal_midpoint"
                    fallback_reason = "no_frame_level_attention_signal"
                    apex_available = False
                    fallback_count += 1

            selected = dict(selected_frame)
            selected["selection_bucket_start_ms"] = int(bucket_start_ms)
            selected["selection_source"] = selection_source
            selected["selection_apex_available"] = bool(apex_available)
            if chosen_candidate is not None:
                selection_score = float(chosen_candidate.get("score") or 0.0)
            if selection_score is not None:
                selected["selection_score"] = round(float(selection_score), 6)
            if fallback_reason:
                selected["selection_fallback_reason"] = fallback_reason
            selected_rows.append((int(selected_index), selected))

            group: Dict[str, Any] = {
                "bucket_start_ms": int(bucket_start_ms),
                "source_frame_indices": source_indices,
                "source_timestamps_ms": source_timestamps,
                "selected_source_frame_index": int(selected_provenance_index),
                "selected_timestamp_ms": int(selected_timestamp),
                "selection_source": selection_source,
                "apex_available": bool(apex_available),
            }
            if chosen_candidate is not None:
                group["signal_references"] = [dict(chosen_candidate.get("reference") or {})]
            if selection_score is not None:
                group["selection_score"] = float(selection_score)
            if source_hashes:
                group["source_frame_hashes"] = source_hashes
            if isinstance(capture_selection, Mapping):
                selected_frame_hash = str(capture_selection.get("selected_frame_hash") or "").strip()[:40]
                if selected_frame_hash:
                    group["selected_frame_hash"] = selected_frame_hash
                score_source = str(capture_selection.get("score_source") or "").strip().lower()[:80]
                if score_source:
                    group["score_source"] = score_source
            if fallback_reason:
                group["fallback_reason"] = fallback_reason
            groups.append(group)
            selection_sources[selection_source] = selection_sources.get(selection_source, 0) + 1

        for source_index, source_frame in missing_timestamp:
            selection_source = "timestamp_unavailable_passthrough"
            fallback_reason = "frame_timestamp_unavailable"
            selected = dict(source_frame)
            selected["selection_source"] = selection_source
            selected["selection_apex_available"] = False
            selected["selection_fallback_reason"] = fallback_reason
            selected_rows.append((int(source_index), selected))
            groups.append(
                {
                    "source_frame_indices": [int(source_index)],
                    "source_timestamps_ms": [None],
                    "selected_source_frame_index": int(source_index),
                    "selection_source": selection_source,
                    "apex_available": False,
                    "fallback_reason": fallback_reason,
                }
            )
            selection_sources[selection_source] = selection_sources.get(selection_source, 0) + 1
            fallback_count += 1

        selected_rows.sort(key=lambda item: item[0])
        groups.sort(key=lambda item: int(_parse_optional_int(item.get("selected_source_frame_index")) or 0))
        provenance_source_frame_count = sum(
            len(group.get("source_frame_indices") or [])
            for group in groups
            if isinstance(group, Mapping)
        )
        selection = cls._compact_frame_selection(
            {
                "version": 1,
                "policy": "per_second_attention_apex_v1",
                "time_bucket_ms": 1000,
                "source_frame_count": provenance_source_frame_count or len(source_frames),
                "selected_frame_count": len(selected_rows),
                "apex_selected_count": apex_selected_count,
                "fallback_count": fallback_count,
                "single_frame_count": single_frame_count,
                "timestamp_unavailable_count": len(missing_timestamp),
                "selection_sources": selection_sources,
                "groups": groups,
            }
        )
        return [frame for _index, frame in selected_rows], selection

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
                for text_key in (
                    "probe_id",
                    "severity",
                    "capture_selection_source",
                    "capture_frame_hash",
                    "capture_fallback_reason",
                ):
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
                for int_key in (
                    "timestamp_ms",
                    "apex_frame",
                    "hit_count",
                    "capture_selected_source_frame_index",
                ):
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
                active_ratio = cls._finite_float(raw.get("active_ratio"))
                if active_ratio is not None:
                    item["active_ratio"] = round(float(active_ratio), 4)
                for int_key in ("timestamp_ms", "frame_index", "apex_frame", "frame_interval_ms"):
                    parsed = _parse_optional_int(raw.get(int_key))
                    if parsed is not None:
                        item[int_key] = int(parsed)
                road_items.append(item)
        if road_items:
            out["road_cv_cues"] = road_items[:8]

        road_frame_items: List[Dict[str, Any]] = []
        raw_road_frames = value.get("road_cv_frame_scores")
        if isinstance(raw_road_frames, Sequence) and not isinstance(raw_road_frames, (str, bytes, bytearray)):
            for raw in raw_road_frames[:256]:
                if not isinstance(raw, Mapping):
                    continue
                source_frame_index = _parse_optional_int(
                    raw.get("source_frame_index") or raw.get("frame_index")
                )
                timestamp_ms = _parse_optional_int(raw.get("timestamp_ms"))
                if source_frame_index is None or source_frame_index < 1 or timestamp_ms is None:
                    continue
                item: Dict[str, Any] = {
                    "source_frame_index": int(source_frame_index),
                    "timestamp_ms": int(timestamp_ms),
                    "score_semantics": "road_cv_frame_attention_signal_not_visual_proof",
                }
                for score_key in (
                    "attention_score",
                    "cue_score",
                    "active_ratio",
                    "global_motion",
                ):
                    score = cls._finite_float(raw.get(score_key))
                    if score is not None:
                        item[score_key] = round(float(score), 6)
                score_source = str(raw.get("score_source") or "").strip().lower()[:80]
                if score_source:
                    item["score_source"] = score_source
                unavailable_reason = str(raw.get("unavailable_reason") or "").strip().lower()[:160]
                if unavailable_reason:
                    item["unavailable_reason"] = unavailable_reason
                for flag in ("warmup", "scene_cut", "low_fps_suppressed"):
                    if flag in raw:
                        item[flag] = bool(raw.get(flag))
                road_frame_items.append(item)
        if road_frame_items:
            out["road_cv_frame_scores"] = road_frame_items

        episode_items: List[Dict[str, Any]] = []
        raw_episodes = value.get("road_episodes")
        if isinstance(raw_episodes, Sequence) and not isinstance(raw_episodes, (str, bytes, bytearray)):
            for raw in raw_episodes[:12]:
                if not isinstance(raw, Mapping):
                    continue
                event_type = str(raw.get("event_type") or "").strip()[:80]
                episode_id = str(raw.get("episode_id") or "").strip()[:80]
                if not event_type or not episode_id:
                    continue
                item = {
                    "episode_id": episode_id,
                    "event_type": event_type,
                    "confidence": str(raw.get("confidence") or "low").strip().lower()[:20],
                    "status": str(raw.get("status") or "candidate").strip().lower()[:20],
                    "score_semantics": "road_episode_fusion_candidate_not_visual_proof",
                }
                for text_key in ("zone_name",):
                    text = str(raw.get(text_key) or "").strip()
                    if text:
                        item[text_key] = text[:80]
                score = cls._finite_float(raw.get("score"))
                if score is not None:
                    item["score"] = round(float(score), 4)
                sources = raw.get("sources")
                if isinstance(sources, Sequence) and not isinstance(sources, (str, bytes, bytearray)):
                    item["sources"] = [str(source).strip()[:40] for source in sources if str(source).strip()][:8]
                for int_key in ("channel_id", "start_ms", "end_ms", "apex_timestamp_ms", "apex_frame", "cue_count"):
                    parsed = _parse_optional_int(raw.get(int_key))
                    if parsed is not None:
                        item[int_key] = int(parsed)
                episode_items.append(item)
        if episode_items:
            out["road_episodes"] = episode_items[:8]

        scene = value.get("road_cv_scene")
        if isinstance(scene, Mapping):
            scene_out: Dict[str, Any] = {}
            for text_key in ("confidence", "reason", "status", "live_sample_confidence", "live_sample_reason"):
                text = str(scene.get(text_key) or "").strip()
                if text:
                    scene_out[text_key] = text[:180]
            if "directional_enabled" in scene:
                scene_out["directional_enabled"] = bool(scene.get("directional_enabled"))
            for int_key in ("frame_count", "motion_pair_count", "scene_cut_count", "sample_count", "usable_zone_samples", "usable_flow_samples"):
                parsed = _parse_optional_int(scene.get(int_key))
                if parsed is not None:
                    scene_out[int_key] = int(parsed)
            for score_key in ("zone_area_ratio", "flow_dominance", "zone_agreement", "flow_agreement"):
                number = cls._finite_float(scene.get(score_key))
                if number is not None:
                    scene_out[score_key] = round(float(number), 4)
            if scene_out:
                out["road_cv_scene"] = scene_out

        attention = value.get("capture_attention")
        if isinstance(attention, Mapping):
            attention_out: Dict[str, Any] = {}
            policy = str(attention.get("policy") or "").strip()[:60]
            if policy:
                attention_out["policy"] = policy
            attention_baseline = attention.get("baseline")
            if isinstance(attention_baseline, Mapping):
                baseline_out: Dict[str, Any] = {}
                baseline_level = cls._finite_float(attention_baseline.get("level"))
                if baseline_level is not None:
                    baseline_out["level"] = round(float(baseline_level), 6)
                if "warmup" in attention_baseline:
                    baseline_out["warmup"] = bool(attention_baseline.get("warmup"))
                if baseline_out:
                    attention_out["baseline"] = baseline_out
            second_items: List[Dict[str, Any]] = []
            raw_seconds = attention.get("seconds")
            if isinstance(raw_seconds, Sequence) and not isinstance(raw_seconds, (str, bytes, bytearray)):
                for raw in raw_seconds[:6]:
                    if not isinstance(raw, Mapping):
                        continue
                    snapshot = _parse_optional_int(raw.get("snapshot"))
                    mode = str(raw.get("mode") or "").strip().lower()[:20]
                    if snapshot is None or snapshot < 1 or mode not in {"burst", "normal"}:
                        continue
                    second_item: Dict[str, Any] = {"snapshot": int(snapshot), "mode": mode}
                    second_activity = cls._finite_float(raw.get("activity_x"))
                    if second_activity is not None:
                        second_item["activity_x"] = round(float(second_activity), 2)
                    blur = str(raw.get("blur") or "").strip().lower()[:40]
                    if blur:
                        second_item["blur"] = blur
                    if raw.get("sharper_companion"):
                        second_item["sharper_companion"] = True
                    second_items.append(second_item)
            if second_items:
                attention_out["seconds"] = second_items
            if attention_out.get("seconds"):
                out["capture_attention"] = attention_out

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

        has_signal_payload = any(
            key in out
            for key in (
                "clip_probe_signals",
                "road_cv_cues",
                "road_cv_frame_scores",
                "road_episodes",
                "road_cv_scene",
                "capture_attention",
            )
        )
        if not has_signal_payload:
            return {}
        return out

    @classmethod
    def _compact_summary_history_entry(cls, value: Mapping[str, Any]) -> Dict[str, Any]:
        """Keep operator/agent evidence while dropping per-candidate diagnostics.

        Exact candidate arrays and hashes have already been written to the frame
        archive before an L0 entry reaches history. Repeating them in every
        runtime-state row made ordinary feed reads and persistence scale with
        tens of megabytes per channel.
        """

        out = dict(value)
        frame_selection = cls._compact_frame_selection(value.get("frame_selection"))
        if frame_selection:
            compact_selection = {
                key: frame_selection[key]
                for key in (
                    "version",
                    "policy",
                    "time_bucket_ms",
                    "source_frame_count",
                    "selected_frame_count",
                    "apex_selected_count",
                    "fallback_count",
                    "single_frame_count",
                    "timestamp_unavailable_count",
                    "selection_sources",
                )
                if key in frame_selection
            }
            compact_groups: List[Dict[str, Any]] = []
            for raw_group in frame_selection.get("groups") or []:
                if not isinstance(raw_group, Mapping):
                    continue
                group = {
                    key: raw_group[key]
                    for key in (
                        "bucket_start_ms",
                        "selected_timestamp_ms",
                        "selected_source_frame_index",
                        "selection_source",
                        "apex_available",
                        "fallback_reason",
                    )
                    if key in raw_group
                }
                if group:
                    compact_groups.append(group)
            if compact_groups:
                compact_selection["groups"] = compact_groups[:64]
            out["frame_selection"] = compact_selection
        else:
            out.pop("frame_selection", None)

        vector_signal = cls._compact_vector_signal(value.get("vector_signal"))
        if vector_signal:
            history_signal: Dict[str, Any] = {
                key: vector_signal[key]
                for key in ("version", "semantics", "channel_id", "batch_start_ms", "batch_end_ms")
                if key in vector_signal
            }
            clip_signals = vector_signal.get("clip_probe_signals")
            if isinstance(clip_signals, list) and clip_signals:
                history_signal["clip_probe_signals"] = [dict(item) for item in clip_signals[:4]]
            road_cues = vector_signal.get("road_cv_cues")
            if isinstance(road_cues, list) and road_cues:
                history_signal["road_cv_cues"] = [dict(item) for item in road_cues[:4]]
            road_episodes = vector_signal.get("road_episodes")
            if isinstance(road_episodes, list) and road_episodes:
                history_signal["road_episodes"] = [dict(item) for item in road_episodes[:4]]
            attention = vector_signal.get("capture_attention")
            if isinstance(attention, Mapping):
                history_signal["capture_attention"] = dict(attention)
            if any(
                key in history_signal
                for key in ("clip_probe_signals", "road_cv_cues", "road_episodes", "capture_attention")
            ):
                out["vector_signal"] = history_signal
            else:
                out.pop("vector_signal", None)
        else:
            out.pop("vector_signal", None)
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
            mapped_hits = [item for item in hits if isinstance(item, Mapping)]
            if batch_start_ms is not None or batch_end_ms is not None:
                in_batch_hits: List[Mapping[str, Any]] = []
                for item in mapped_hits:
                    item_timestamp_ms = _parse_optional_int(item.get("timestamp_ms"))
                    if item_timestamp_ms is None:
                        continue
                    if batch_start_ms is not None and item_timestamp_ms < int(batch_start_ms):
                        continue
                    if batch_end_ms is not None and item_timestamp_ms > int(batch_end_ms):
                        continue
                    in_batch_hits.append(item)
                outside_count = len(mapped_hits) - len(in_batch_hits)
                if outside_count > 0:
                    health["clip_probe_hits_outside_batch"] = int(
                        health.get("clip_probe_hits_outside_batch") or 0
                    ) + outside_count
                mapped_hits = in_batch_hits
            best = mapped_hits[0] if mapped_hits else None
            if best is None:
                continue
            timestamp_ms = _parse_optional_int(best.get("timestamp_ms"))
            apex_frame = self._nearest_batch_frame_index(frames, timestamp_ms)
            signal: Dict[str, Any] = {
                "name": str(probe.get("name") or probe.get("id") or "CLIP probe").strip()[:120],
                "probe_id": str(probe.get("id") or "").strip()[:80],
                "severity": str(probe.get("severity") or "normal").strip().lower()[:20] or "normal",
                "state": "positive_candidate",
                "hit_count": len(mapped_hits),
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
            selection_provenance = best.get("selection_provenance")
            if isinstance(selection_provenance, Mapping):
                capture_source = str(selection_provenance.get("selection_source") or "").strip().lower()
                if capture_source:
                    signal["capture_selection_source"] = capture_source[:80]
                capture_index = _parse_optional_int(
                    selection_provenance.get("selected_source_frame_index")
                )
                if capture_index is not None:
                    signal["capture_selected_source_frame_index"] = int(capture_index)
                capture_hash = str(selection_provenance.get("selected_frame_hash") or "").strip()
                if capture_hash:
                    signal["capture_frame_hash"] = capture_hash[:40]
                capture_fallback = str(selection_provenance.get("fallback_reason") or "").strip().lower()
                if capture_fallback:
                    signal["capture_fallback_reason"] = capture_fallback[:160]
            signals.append(signal)
        signals.sort(
            key=lambda item: (
                ALERT_SEVERITY_ORDER.index(str(item.get("severity") or "info")) if str(item.get("severity") or "info") in ALERT_SEVERITY_ORDER else 99,
                -float(item.get("margin") or 0.0),
            )
        )
        return signals[:8], health

    @staticmethod
    def _road_scene_card_to_dict(card: Any) -> Dict[str, Any]:
        if card is None:
            return {}
        return {
            "channel_id": int(getattr(card, "channel_id", 0) or 0),
            "title": str(getattr(card, "title", "") or ""),
            "version": int(getattr(card, "version", 1) or 1),
            "notes": str(getattr(card, "notes", "") or ""),
            "zones": [
                {
                    "name": str(getattr(zone, "name", "") or ""),
                    "polygon": [list(point) for point in (getattr(zone, "polygon", ()) or ())],
                    "zone_type": str(getattr(zone, "zone_type", "") or "road"),
                    "expected_flow": list(getattr(zone, "expected_flow", None))
                    if getattr(zone, "expected_flow", None)
                    else None,
                    "enabled": bool(getattr(zone, "enabled", True)),
                }
                for zone in (getattr(card, "zones", ()) or ())
            ],
        }

    @staticmethod
    def _road_scene_card_without_expected_flow(card: Any) -> Any:
        if RoadSceneCard is None or RoadZone is None or card is None:
            return card
        try:
            zones = tuple(
                RoadZone(
                    name=str(getattr(zone, "name", "") or "road_zone"),
                    polygon=tuple(getattr(zone, "polygon", ()) or ()),
                    zone_type=str(getattr(zone, "zone_type", "") or "road"),
                    expected_flow=None,
                    enabled=bool(getattr(zone, "enabled", True)),
                )
                for zone in (getattr(card, "zones", ()) or ())
            )
            return RoadSceneCard(
                channel_id=int(getattr(card, "channel_id", 0) or 0),
                title=str(getattr(card, "title", "") or ""),
                zones=zones,
                notes=str(getattr(card, "notes", "") or ""),
                version=int(getattr(card, "version", 1) or 1),
            )
        except Exception:
            return card

    @classmethod
    def _road_calibration_state_from_result(cls, result: Any) -> Dict[str, Any]:
        payload = result.as_dict() if hasattr(result, "as_dict") else {}
        if not isinstance(payload, Mapping):
            payload = {}
        return {
            "channel_id": int(getattr(getattr(result, "scene_card", None), "channel_id", 0) or 0),
            "confidence": str(getattr(result, "confidence", payload.get("confidence", "low")) or "low").strip().lower(),
            "reason": str(getattr(result, "reason", payload.get("reason", "")) or "").strip()[:240],
            "sample_count": int(_parse_optional_int(payload.get("sample_count")) or _parse_optional_int(getattr(result, "sample_count", None)) or 0),
            "usable_zone_samples": int(_parse_optional_int(payload.get("usable_zone_samples")) or _parse_optional_int(getattr(result, "usable_zone_samples", None)) or 0),
            "usable_flow_samples": int(_parse_optional_int(payload.get("usable_flow_samples")) or _parse_optional_int(getattr(result, "usable_flow_samples", None)) or 0),
            "zone_agreement": round(float(cls._finite_float(payload.get("zone_agreement")) or cls._finite_float(getattr(result, "zone_agreement", None)) or 0.0), 4),
            "flow_agreement": round(float(cls._finite_float(payload.get("flow_agreement")) or cls._finite_float(getattr(result, "flow_agreement", None)) or 0.0), 4),
            "updated_at": time.time(),
            "scene_card": cls._road_scene_card_to_dict(getattr(result, "scene_card", None)),
        }

    @staticmethod
    def _road_scene_card_from_state(state: Mapping[str, Any]) -> Any:
        if RoadSceneCard is None:
            return None
        card_raw = state.get("scene_card")
        if not isinstance(card_raw, Mapping):
            return None
        try:
            return RoadSceneCard.from_mapping(card_raw)
        except Exception:
            return None

    def _update_road_scene_calibration(self, channel_id: int, sample_result: Any) -> Dict[str, Any]:
        if calibrate_scene_card_from_results is None or SceneCalibrationConfig is None:
            return {}
        with self.cache_lock:
            samples = self.road_scene_auto_samples.setdefault(int(channel_id), [])
            samples.append(sample_result)
            if len(samples) > self.road_scene_calibration_samples:
                del samples[: len(samples) - self.road_scene_calibration_samples]
            if len(samples) < self.road_scene_calibration_samples:
                existing = self.road_scene_calibrations.get(int(channel_id))
                return dict(existing) if isinstance(existing, Mapping) else {}
            sample_list = list(samples)
        try:
            result = calibrate_scene_card_from_results(
                int(channel_id),
                f"Channel {channel_id}",
                sample_list,
                config=SceneCalibrationConfig(),
            )
        except Exception:
            existing = self.road_scene_calibrations.get(int(channel_id))
            return dict(existing) if isinstance(existing, Mapping) else {}
        state = self._road_calibration_state_from_result(result)
        with self.cache_lock:
            self.road_scene_calibrations[int(channel_id)] = state
            self._summary_state_dirty = True
        return dict(state)

    def _road_episode_aggregator(self, channel_id: int) -> Any:
        if RoadEpisodeAggregator is None or RoadEpisodeAggregatorConfig is None:
            return None
        with self.cache_lock:
            existing = self.road_episode_aggregators.get(int(channel_id))
            if existing is not None:
                return existing
            aggregator = RoadEpisodeAggregator(
                RoadEpisodeAggregatorConfig(
                    window_ms=90_000,
                    close_after_ms=45_000,
                    max_inter_cue_gap_ms=20_000,
                )
            )
            self.road_episode_aggregators[int(channel_id)] = aggregator
            return aggregator

    @staticmethod
    def _road_cue_type_from_clip_signal(signal: Mapping[str, Any]) -> Optional[str]:
        text = " ".join(
            str(signal.get(key) or "")
            for key in ("name", "probe_id", "state")
        ).lower()
        if "burnout" in text:
            return "clip_burnout"
        if "tire smoke" in text or ("smoke" in text and ("car" in text or "vehicle" in text or "road" in text)):
            return "clip_tire_smoke"
        if "drift" in text or "sideways" in text or "sliding vehicle" in text:
            return "clip_vehicle_drift"
        return None

    @classmethod
    def _compact_road_episodes(cls, episodes: Sequence[Any]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for episode in episodes[:12]:
            if episode is None:
                continue
            item: Dict[str, Any] = {
                "episode_id": str(getattr(episode, "episode_id", "") or "")[:80],
                "event_type": str(getattr(episode, "event_type", "") or "")[:80],
                "zone_name": str(getattr(episode, "zone_name", "") or "")[:80],
                "confidence": str(getattr(episode, "confidence", "") or "low")[:20],
                "status": str(getattr(episode, "status", "") or "candidate")[:20],
                "score_semantics": "road_episode_fusion_candidate_not_visual_proof",
                "cue_count": len(getattr(episode, "cues", ()) or ()),
                "sources": sorted({str(getattr(cue, "source", "") or "") for cue in (getattr(episode, "cues", ()) or ()) if str(getattr(cue, "source", "") or "")})[:8],
            }
            for attr in ("channel_id", "start_ms", "end_ms", "apex_timestamp_ms", "apex_frame"):
                parsed = _parse_optional_int(getattr(episode, attr, None))
                if parsed is not None:
                    item[attr] = int(parsed)
            score = cls._finite_float(getattr(episode, "score", None))
            if score is not None:
                item["score"] = round(float(score), 4)
            if item["episode_id"] and item["event_type"]:
                items.append(item)
        items.sort(key=lambda row: (int(_parse_optional_int(row.get("end_ms")) or 0), float(row.get("score") or 0.0)), reverse=True)
        return items[:8]

    def _road_episode_vector_signals(
        self,
        channel_id: int,
        road_cues: Sequence[Mapping[str, Any]],
        clip_signals: Sequence[Mapping[str, Any]],
        *,
        now_ms: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if RoadEventCue is None:
            return []
        aggregator = self._road_episode_aggregator(int(channel_id))
        if aggregator is None:
            return []
        event_cues: List[Any] = []
        for cue in road_cues:
            if not isinstance(cue, Mapping):
                continue
            cue_type = str(cue.get("cue_type") or "").strip()
            timestamp_ms = _parse_optional_int(cue.get("timestamp_ms"))
            if not cue_type or timestamp_ms is None:
                continue
            event_cues.append(
                RoadEventCue(
                    source="cv_motion",
                    cue_type=cue_type,
                    timestamp_ms=int(timestamp_ms),
                    channel_id=int(channel_id),
                    zone_name=str(cue.get("zone_name") or "").strip(),
                    score=float(self._finite_float(cue.get("score")) or 0.0),
                    label=str(cue.get("evidence") or "").strip(),
                    evidence={
                        "frame_index": _parse_optional_int(cue.get("frame_index")),
                        "apex_frame": _parse_optional_int(cue.get("apex_frame")),
                    },
                )
            )
        for signal in clip_signals:
            if not isinstance(signal, Mapping):
                continue
            cue_type = self._road_cue_type_from_clip_signal(signal)
            timestamp_ms = _parse_optional_int(signal.get("timestamp_ms"))
            if not cue_type or timestamp_ms is None:
                continue
            event_cues.append(
                RoadEventCue(
                    source="clip_probe",
                    cue_type=cue_type,
                    timestamp_ms=int(timestamp_ms),
                    channel_id=int(channel_id),
                    zone_name="",
                    score=float(self._finite_float(signal.get("margin")) or self._finite_float(signal.get("m")) or 0.0),
                    label=str(signal.get("name") or "").strip(),
                    evidence={"apex_frame": _parse_optional_int(signal.get("apex_frame"))},
                )
            )
        if event_cues:
            episodes = aggregator.add_cues(event_cues)
        else:
            episodes = aggregator.current_episodes(now_ms=now_ms)
        return self._compact_road_episodes(list(episodes))

    def _road_cv_vector_signals(
        self,
        channel_id: int,
        frames: Sequence[Mapping[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        health: Dict[str, Any] = {}
        if not self.road_cv_batch_signals_enabled:
            health["road_cv_status"] = "disabled"
            return [], [], {}, health
        if (
            DecodedVideoFrame is None
            or AutoSceneCardConfig is None
            or infer_scene_card_from_frames is None
            or RoadMotionAnalyzer is None
        ):
            health["road_cv_status"] = "unavailable"
            return [], [], {}, health
        decoded: List[Any] = []
        indexed_frames = list(enumerate(frames, start=1))
        sampled = indexed_frames[-self.road_cv_batch_max_frames :]
        health["road_cv_source_frame_count"] = len(indexed_frames)
        health["road_cv_sampled_frame_count"] = len(sampled)
        for source_index, frame in sampled:
            if not isinstance(frame, Mapping):
                continue
            image = self._decode_frame_thumbnail_to_rgb_array(frame)
            if image is None:
                continue
            timestamp_ms = self._batch_frame_timestamp_ms(frame)
            if timestamp_ms is None:
                timestamp_ms = int(time.time() * 1000.0)
            decoded.append(
                DecodedVideoFrame(
                    frame_index=int(source_index),
                    timestamp_ms=int(timestamp_ms),
                    image=image,
                )
            )
        health["road_cv_decoded_frames"] = len(decoded)
        if len(decoded) < 3:
            return [], [], {}, health
        intervals = [
            int(decoded[idx].timestamp_ms) - int(decoded[idx - 1].timestamp_ms)
            for idx in range(1, len(decoded))
            if int(decoded[idx].timestamp_ms) >= int(decoded[idx - 1].timestamp_ms)
        ]
        if intervals:
            sorted_intervals = sorted(intervals)
            median_interval = sorted_intervals[len(sorted_intervals) // 2]
            p90_interval = sorted_intervals[min(len(sorted_intervals) - 1, int(round((len(sorted_intervals) - 1) * 0.9)))]
            health["road_cv_frame_interval_ms_median"] = int(median_interval)
            health["road_cv_frame_interval_ms_p90"] = int(p90_interval)
        try:
            scene_result = infer_scene_card_from_frames(
                int(channel_id),
                f"Channel {channel_id}",
                decoded,
                config=AutoSceneCardConfig(max_edge=int(self.road_cv_batch_max_edge), min_frames=min(12, max(3, len(decoded)))),
            )
            calibration_state = self._update_road_scene_calibration(int(channel_id), scene_result)
            calibrated_card = self._road_scene_card_from_state(calibration_state)
            calibration_confidence = str(calibration_state.get("confidence") or "").strip().lower()
            if calibrated_card is not None and calibration_confidence == "high":
                analysis_card = calibrated_card
                scene_status = "calibrated"
                directional_enabled = True
            else:
                analysis_card = self._road_scene_card_without_expected_flow(scene_result.scene_card)
                scene_status = "calibrating" if calibration_state else "uncalibrated"
                if calibration_confidence and calibration_confidence != "high":
                    scene_status = "low_confidence" if calibration_confidence == "low" else "calibrating"
                directional_enabled = False
            health["road_cv_scene_status"] = scene_status
            health["road_cv_directional_enabled"] = bool(directional_enabled)
            analyzer = RoadMotionAnalyzer(analysis_card)
            cues: List[Dict[str, Any]] = []
            frame_scores: List[Dict[str, Any]] = []
            active_ratios: List[float] = []
            global_motion_values: List[float] = []
            low_fps_suppressed = 0
            for decoded_frame in decoded:
                sample = analyzer.analyze_frame(
                    decoded_frame.image,
                    timestamp_ms=int(decoded_frame.timestamp_ms),
                    frame_index=int(decoded_frame.frame_index),
                )
                frame_global_motion: Optional[float] = None
                if sample.global_motion:
                    global_motion = self._finite_float(sample.global_motion.get("magnitude"))
                    if global_motion is not None:
                        frame_global_motion = float(global_motion)
                        global_motion_values.append(float(global_motion))
                sample_low_fps = bool(
                    sample.quality
                    and int(_parse_optional_int(sample.quality.get("low_fps_suppressed")) or 0) > 0
                )
                if sample_low_fps:
                    low_fps_suppressed += 1
                frame_active_ratios: List[float] = []
                for metrics in sample.zone_metrics.values():
                    if not isinstance(metrics, Mapping):
                        continue
                    active_ratio = self._finite_float(metrics.get("active_ratio"))
                    if active_ratio is not None:
                        frame_active_ratios.append(float(active_ratio))
                        active_ratios.append(float(active_ratio))
                frame_cue_scores = [
                    float(cue.score)
                    for cue in sample.cues
                    if self._finite_float(cue.score) is not None
                ]
                max_active_ratio = max(frame_active_ratios) if frame_active_ratios else None
                max_cue_score = max(frame_cue_scores) if frame_cue_scores else None
                frame_score: Dict[str, Any] = {
                    "source_frame_index": int(decoded_frame.frame_index),
                    "timestamp_ms": int(decoded_frame.timestamp_ms),
                    "warmup": bool(sample.warmup),
                    "scene_cut": bool(sample.scene_cut),
                    "low_fps_suppressed": bool(sample_low_fps),
                }
                if max_active_ratio is not None:
                    frame_score["active_ratio"] = round(float(max_active_ratio), 6)
                if max_cue_score is not None:
                    frame_score["cue_score"] = round(float(max_cue_score), 6)
                if frame_global_motion is not None:
                    frame_score["global_motion"] = round(float(frame_global_motion), 6)
                if sample.warmup:
                    frame_score["unavailable_reason"] = "road_cv_warmup_frame"
                elif sample.scene_cut:
                    frame_score["unavailable_reason"] = "road_cv_scene_cut"
                elif sample_low_fps:
                    frame_score["unavailable_reason"] = "road_cv_low_fps_suppressed"
                elif max_cue_score is not None and max_cue_score > 0.0:
                    frame_score["attention_score"] = round(float(max_cue_score), 6)
                    frame_score["score_source"] = "road_cv_cue"
                elif max_active_ratio is not None and max_active_ratio > 0.0:
                    frame_score["attention_score"] = round(float(max_active_ratio), 6)
                    frame_score["score_source"] = "road_cv_active_ratio"
                else:
                    frame_score["unavailable_reason"] = "no_positive_road_cv_attention_score"
                frame_scores.append(frame_score)
                for cue in sample.cues:
                    metrics = dict(cue.metrics)
                    cues.append(
                        {
                            "cue_type": cue.cue_type,
                            "zone_name": cue.zone_name,
                            "score": round(float(cue.score), 4),
                            "evidence": cue.evidence,
                            "active_ratio": round(float(metrics.get("active_ratio") or 0.0), 4),
                            "frame_interval_ms": int(_parse_optional_int(metrics.get("frame_interval_ms")) or 0),
                            "timestamp_ms": int(cue.timestamp_ms),
                            "frame_index": int(cue.frame_index or decoded_frame.frame_index),
                            "apex_frame": int(cue.frame_index or decoded_frame.frame_index),
                        }
                    )
            if active_ratios:
                health["road_cv_active_ratio_max"] = round(max(active_ratios), 4)
            if global_motion_values:
                health["road_cv_global_motion_max"] = round(max(global_motion_values), 4)
            if low_fps_suppressed:
                health["road_cv_low_fps_suppressed_frames"] = int(low_fps_suppressed)
            cues.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
            scene = scene_result.as_dict()
            scene_compact = {
                "confidence": calibration_confidence or scene.get("confidence"),
                "reason": calibration_state.get("reason") if calibration_state else scene.get("reason"),
                "status": scene_status,
                "directional_enabled": bool(directional_enabled),
                "live_sample_confidence": scene.get("confidence"),
                "live_sample_reason": scene.get("reason"),
                "frame_count": scene.get("frame_count"),
                "motion_pair_count": scene.get("motion_pair_count"),
                "scene_cut_count": scene.get("scene_cut_count"),
                "zone_area_ratio": scene.get("zone_area_ratio"),
                "flow_dominance": scene.get("flow_dominance"),
            }
            for key in ("sample_count", "usable_zone_samples", "usable_flow_samples", "zone_agreement", "flow_agreement"):
                if key in calibration_state:
                    scene_compact[key] = calibration_state.get(key)
            return cues[:8], frame_scores, scene_compact, health
        except Exception as exc:
            health["road_cv_error"] = str(exc)[:160] or exc.__class__.__name__
            return [], [], {}, health

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
                episode_count = len(vector_signal.get("road_episodes") or []) if isinstance(vector_signal.get("road_episodes"), list) else 0
                vector_signal_total += int(clip_count + road_count + episode_count)
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
        source_frame_count = 0
        selected_frame_count = 0
        selection_fallback_count = 0
        selection_sources: Dict[str, int] = {}
        last_frame_selection: Dict[str, Any] = {}
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
            source_frame_count += int(
                _parse_optional_int(log.get("source_frame_count"))
                or _parse_optional_int(log.get("frame_count"))
                or 0
            )
            selected_frame_count += int(
                _parse_optional_int(log.get("selected_frame_count"))
                or _parse_optional_int(log.get("frame_count"))
                or 0
            )
            frame_selection = cls._compact_frame_selection(log.get("frame_selection"))
            if frame_selection:
                last_frame_selection = frame_selection
                selection_fallback_count += int(
                    _parse_optional_int(frame_selection.get("fallback_count")) or 0
                )
                cls._merge_int_breakdown(
                    selection_sources,
                    cast(Mapping[str, Any], frame_selection.get("selection_sources") or {}),
                )

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
                episode_count = len(vector_signal.get("road_episodes") or []) if isinstance(vector_signal.get("road_episodes"), list) else 0
                vector_signal_total += int(clip_count + road_count + episode_count)
                vector_item = {
                    "timestamp_ms": parsed_batch_end or (int(latest_candidate * 1000.0) if latest_candidate is not None else None),
                    "clip_probe_signal_count": clip_count,
                    "road_cv_cue_count": road_count,
                    "road_episode_count": episode_count,
                    "health": vector_signal.get("health") if isinstance(vector_signal.get("health"), Mapping) else {},
                }
                clip_signals = vector_signal.get("clip_probe_signals")
                if isinstance(clip_signals, list) and clip_signals:
                    vector_item["top_clip_probe"] = clip_signals[0]
                road_cues = vector_signal.get("road_cv_cues")
                if isinstance(road_cues, list) and road_cues:
                    vector_item["top_road_cv_cue"] = road_cues[0]
                road_episodes = vector_signal.get("road_episodes")
                if isinstance(road_episodes, list) and road_episodes:
                    vector_item["top_road_episode"] = road_episodes[0]
                road_scene = vector_signal.get("road_cv_scene")
                if isinstance(road_scene, Mapping):
                    vector_item["road_cv_scene"] = dict(road_scene)
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
            "source_frame_count": int(source_frame_count),
            "selected_frame_count": int(selected_frame_count),
            "selection_fallback_count": int(selection_fallback_count),
            "selection_sources": dict(selection_sources),
            "last_frame_selection": last_frame_selection,
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
            "last_live_segment_target_seconds",
            "last_live_segment_summary_target_seconds",
            "last_live_segment_raw_frame_budget",
            "last_live_segment_byte_budget",
            "last_live_segment_streamed_bytes",
            "last_live_segment_represented_seconds",
            "last_live_segment_completed_at",
            "last_live_segment_source_start_timestamp_ms",
            "last_live_segment_last_source_timestamp_ms",
            "last_live_segment_timestamp_source",
            "live_segment_inflight",
            "live_segment_capture_started_at",
            "live_segment_inflight_target_seconds",
            "live_segment_inflight_raw_frame_budget",
            "live_segment_inflight_frames",
            "live_segment_inflight_represented_seconds",
            "live_segment_backoff_sec",
            "capture_last_error",
            "probe_last_error",
            "summary_last_error",
            "summary_failed_batches",
            "summary_last_failure_at",
            "summary_last_success_at",
            "frozen_signal",
            "frozen_signal_since",
            "frozen_signal_age_sec",
            "frozen_frame_count",
            "frozen_frame_hash",
            "frozen_frame_dropped_count",
            "capture_apex_pending_frames",
            "capture_apex_raw_frame_count",
            "capture_apex_selected_count",
            "capture_apex_fallback_count",
            "capture_apex_probe_dispatch_count",
            "capture_apex_probe_failure_count",
            "capture_apex_probe_skipped_count",
            "capture_apex_selection_sources",
            "capture_apex_last_selection",
        ):
            if field in runtime:
                digest[field] = (
                    _safe_error_text(runtime.get(field), 240)
                    if field in {
                        "last_live_segment_error",
                        "capture_last_error",
                        "probe_last_error",
                        "summary_last_error",
                    }
                    else runtime.get(field)
                )
        last_error = _safe_error_text(
            runtime.get("last_error") or runtime.get("last_restore_error"),
            240,
        )
        digest["last_error"] = last_error or None
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
        selected_frame_count = _parse_optional_int(entry.get("selected_frame_count"))
        if selected_frame_count is None:
            selected_frame_count = frame_count
        source_frame_count = _parse_optional_int(entry.get("source_frame_count"))
        if source_frame_count is None:
            source_frame_count = max(frame_count, selected_frame_count)
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
        normalized = {
            "channel_id": int(channel_id),
            "run_id": str(entry.get("run_id") or "").strip(),
            "summary": summary,
            "frame_count": int(max(0, frame_count)),
            "source_frame_count": int(max(0, source_frame_count)),
            "selected_frame_count": int(max(0, selected_frame_count)),
            "frame_selection": self._compact_frame_selection(entry.get("frame_selection")),
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
            "bookmark_last_error": _safe_error_text(entry.get("bookmark_last_error"), 240),
            "alert_parser_error": _safe_error_text(entry.get("alert_parser_error"), 240),
            "alert_events": self._compact_alert_events(entry.get("alert_events")),
            "state_observations": self._compact_state_observations(entry.get("state_observations")),
            "state_transition_events": self._compact_state_transition_events(entry.get("state_transition_events")),
            "state_transition_total": int(max(0, _parse_optional_int(entry.get("state_transition_total")) or 0)),
            "vector_signal": self._compact_vector_signal(entry.get("vector_signal")),
            "llm_input_stats": self._compact_llm_input_stats(entry.get("llm_input_stats")),
            "signal_digest": signal_digest,
            **alert_meta,
        }
        for field in (
            "archive_attempted",
            "archive_inserted",
            "archive_summary_frames",
            "archive_alert_frames",
        ):
            if field not in entry:
                continue
            value = _parse_optional_int(entry.get(field))
            if value is not None:
                normalized[field] = max(0, int(value))
        if "archive_error" in entry:
            normalized["archive_error"] = _safe_error_text(entry.get("archive_error"), 240)
        if entry.get("coverage_gap"):
            normalized["coverage_gap"] = True
            gap_reason = str(entry.get("gap_reason") or "").strip().lower()[:80]
            if gap_reason:
                normalized["gap_reason"] = gap_reason
        coalesced_raw = entry.get("coalesced")
        if isinstance(coalesced_raw, Mapping):
            coalesced_out: Dict[str, int] = {}
            for coalesced_key in ("batches", "omitted_frames"):
                parsed_value = _parse_optional_int(coalesced_raw.get(coalesced_key))
                if parsed_value is not None and parsed_value > 0:
                    coalesced_out[coalesced_key] = int(parsed_value)
            if coalesced_out.get("batches", 0) > 1:
                normalized["coalesced"] = coalesced_out
        return normalized

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

    def _rollup_retention_cutoff(self) -> Optional[float]:
        if self.rollup_retention_days <= 0:
            return None
        return time.time() - self.rollup_retention_days * 86400.0

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

    @staticmethod
    def _prompt_default_field_names() -> Set[str]:
        return {
            "stream_system_prompt",
            "alert_policy_prompt",
            "rollup_prompts.L1",
            "rollup_prompts.L2",
            "rollup_prompts.L3",
            "bookmark_enabled",
            "bookmark_cooldown_sec",
            "json_alert_prompt",
        }

    def _build_summary_state_payload_locked(self, revision: Optional[int] = None) -> Dict[str, Any]:
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
            "capture_selector_bias": str(self.default_capture_selector_bias or "auto"),
            "json_alert_prompt": str(self.default_json_alert_prompt or DEFAULT_ALERTS_JSON_PROMPT),
            "channel_overrides": {
                str(channel_id): dict(settings)
                for channel_id, settings in self.channel_prompt_overrides.items()
                if isinstance(settings, Mapping)
            },
        }
        payload = {
            "version": 2,
            "revision": int(
                revision
                if revision is not None
                else max(self.summary_state_revision, self._summary_state_revision_issued) + 1
            ),
            "updated_at": time.time(),
            "summary_history": history_payload,
            "summary_runs": runs_payload,
            "channel_routines": routine_payload,
            "road_scene_calibrations": {
                str(channel_id): dict(state)
                for channel_id, state in self.road_scene_calibrations.items()
                if isinstance(state, Mapping)
            },
            "capture_baselines": {
                str(channel_id): dict(state)
                for channel_id, state in self.capture_activity_baselines.items()
                if isinstance(state, Mapping)
            },
            "prompt_settings": prompt_payload,
        }
        return payload

    def _write_summary_state_payload(self, payload: Mapping[str, Any]) -> Optional[str]:
        state_store = getattr(self, "runtime_state_store", None)
        if state_store is not None:
            try:
                state_store.save_state("luxriot_summary_state", payload)
            except Exception as exc:
                return _safe_error_text(exc, 500) or exc.__class__.__name__
        else:
            path = self.summary_state_file
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp_file = path.with_suffix(f"{path.suffix}.tmp")
                tmp_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                tmp_file.replace(path)
            except Exception as exc:
                return _safe_error_text(exc, 500) or exc.__class__.__name__
        return None

    def _mark_summary_state_write_result_locked(
        self,
        payload: Mapping[str, Any],
        error: Optional[str],
    ) -> bool:
        revision = int(_parse_optional_int(payload.get("revision")) or 0)
        if error:
            self.summary_state_last_error = str(error)[:500]
            self._summary_state_dirty = True
            return False
        if revision >= int(self.summary_state_revision):
            self.summary_state_revision = revision
            self.summary_state_last_success_at = self._coerce_float(payload.get("updated_at")) or time.time()
            self.summary_state_last_error = None
        with self._summary_persist_condition:
            has_newer_pending = self._summary_persist_pending is not None
        self._summary_state_dirty = bool(
            has_newer_pending or revision < int(self._summary_state_revision_issued)
        )
        self._persisted_prompt_default_fields.update(self._prompt_default_field_names())
        return True

    def _persist_summary_state_locked(self) -> bool:
        revision = max(self.summary_state_revision, self._summary_state_revision_issued) + 1
        self._summary_state_revision_issued = int(revision)
        payload = self._build_summary_state_payload_locked(revision)
        error = self._write_summary_state_payload(payload)
        return self._mark_summary_state_write_result_locked(payload, error)

    def _summary_state_async_persistence_enabled(self) -> bool:
        state_store = getattr(self, "runtime_state_store", None)
        return str(getattr(state_store, "backend", "") or "").strip().lower() == "postgres"

    def _summary_persist_worker(self) -> None:
        while True:
            with self._summary_persist_condition:
                while self._summary_persist_pending is None:
                    self._summary_persist_condition.wait()
                payload = self._summary_persist_pending
                self._summary_persist_pending = None
            if not isinstance(payload, Mapping):
                continue
            revision = int(_parse_optional_int(payload.get("revision")) or 0)
            with self.cache_lock:
                issued_revision = int(self._summary_state_revision_issued)
            if revision < issued_revision:
                continue
            error = self._write_summary_state_payload(payload)
            with self.cache_lock:
                self._mark_summary_state_write_result_locked(payload, error)

    def _schedule_summary_state_persist_locked(self) -> None:
        revision = max(self.summary_state_revision, self._summary_state_revision_issued) + 1
        self._summary_state_revision_issued = int(revision)
        payload = self._build_summary_state_payload_locked(revision)
        self._summary_state_dirty = True
        with self._summary_persist_condition:
            self._summary_persist_pending = payload
            if self._summary_persist_thread is None or not self._summary_persist_thread.is_alive():
                self._summary_persist_thread = threading.Thread(
                    target=self._summary_persist_worker,
                    daemon=True,
                    name="eva-summary-state",
                )
                self._summary_persist_thread.start()
            self._summary_persist_condition.notify()

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
        if self._summary_state_async_persistence_enabled():
            self._schedule_summary_state_persist_locked()
        elif not self._persist_summary_state_locked():
            self._summary_state_dirty = True

    def _load_summary_state_from_disk(self) -> None:
        payload: Optional[Dict[str, Any]] = None
        state_store = getattr(self, "runtime_state_store", None)
        if state_store is not None:
            try:
                loaded_payload = state_store.load_state("luxriot_summary_state")
                if isinstance(loaded_payload, Mapping):
                    payload = dict(loaded_payload)
            except Exception as exc:
                self.summary_state_last_error = _safe_error_text(exc, 500) or exc.__class__.__name__
                payload = None
        if payload is None:
            path = self.summary_state_file
            if not path.exists():
                return
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                self.summary_state_last_error = _safe_error_text(exc, 500) or exc.__class__.__name__
                return
        history_raw = payload.get("summary_history") if isinstance(payload, Mapping) else None
        runs_raw = payload.get("summary_runs") if isinstance(payload, Mapping) else None
        routines_raw = payload.get("channel_routines") if isinstance(payload, Mapping) else None
        road_scene_raw = payload.get("road_scene_calibrations") if isinstance(payload, Mapping) else None
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
                        normalized_logs.append(self._compact_summary_history_entry(normalized))
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
        loaded_road_scene_calibrations: Dict[int, Dict[str, Any]] = {}
        if isinstance(road_scene_raw, Mapping):
            for channel_key, state_value in road_scene_raw.items():
                channel_id = _parse_optional_int(channel_key)
                if channel_id is None or not isinstance(state_value, Mapping):
                    continue
                card_raw = state_value.get("scene_card")
                if RoadSceneCard is None or not isinstance(card_raw, Mapping):
                    continue
                try:
                    card = RoadSceneCard.from_mapping(card_raw)
                except Exception:
                    continue
                normalized_state = {
                    "channel_id": int(channel_id),
                    "confidence": str(state_value.get("confidence") or "low").strip().lower() or "low",
                    "reason": str(state_value.get("reason") or "").strip()[:240],
                    "sample_count": int(_parse_optional_int(state_value.get("sample_count")) or 0),
                    "usable_zone_samples": int(_parse_optional_int(state_value.get("usable_zone_samples")) or 0),
                    "usable_flow_samples": int(_parse_optional_int(state_value.get("usable_flow_samples")) or 0),
                    "zone_agreement": float(self._coerce_float(state_value.get("zone_agreement")) or 0.0),
                    "flow_agreement": float(self._coerce_float(state_value.get("flow_agreement")) or 0.0),
                    "updated_at": float(self._coerce_float(state_value.get("updated_at")) or time.time()),
                    "scene_card": {
                        "channel_id": card.channel_id,
                        "title": card.title,
                        "version": card.version,
                        "notes": card.notes,
                        "zones": [
                            {
                                "name": zone.name,
                                "polygon": [list(point) for point in zone.polygon],
                                "zone_type": zone.zone_type,
                                "expected_flow": list(zone.expected_flow) if zone.expected_flow else None,
                                "enabled": zone.enabled,
                            }
                            for zone in card.zones
                        ],
                    },
                }
                loaded_road_scene_calibrations[int(channel_id)] = normalized_state
        loaded_capture_baselines: Dict[int, Dict[str, Any]] = {}
        capture_baselines_raw = payload.get("capture_baselines") if isinstance(payload, Mapping) else None
        if isinstance(capture_baselines_raw, Mapping):
            for channel_key, state_value in capture_baselines_raw.items():
                channel_id = _parse_optional_int(channel_key)
                if channel_id is None or channel_id <= 0 or not isinstance(state_value, Mapping):
                    continue
                level = self._coerce_float(state_value.get("level"))
                if level is None or level < 0:
                    continue
                loaded_capture_baselines[int(channel_id)] = {
                    "level": float(level),
                    "dev": max(0.0, float(self._coerce_float(state_value.get("dev")) or 0.0)),
                    "buckets": max(0, int(_parse_optional_int(state_value.get("buckets")) or 0)),
                    "updated_at": float(self._coerce_float(state_value.get("updated_at")) or time.time()),
                }
        loaded_stream_system_prompt: Optional[str] = None
        loaded_alert_policy_prompt: Optional[str] = None
        loaded_rollup_prompts: Dict[str, str] = {}
        loaded_channel_prompt_overrides: Dict[int, Dict[str, Any]] = {}
        loaded_default_bookmark_enabled: Optional[bool] = None
        loaded_default_bookmark_cooldown_sec: Optional[float] = None
        loaded_default_capture_selector_bias: Optional[str] = None
        loaded_default_json_alert_prompt: Optional[str] = None
        loaded_prompt_default_fields: Set[str] = set()
        if isinstance(prompt_settings_raw, Mapping):
            if "stream_system_prompt" in prompt_settings_raw:
                loaded_stream_system_prompt = str(prompt_settings_raw.get("stream_system_prompt") or "")
                loaded_prompt_default_fields.add("stream_system_prompt")
            elif "system_prompt" in prompt_settings_raw:
                loaded_stream_system_prompt = str(prompt_settings_raw.get("system_prompt") or "")
                loaded_prompt_default_fields.add("stream_system_prompt")
            if "alert_policy_prompt" in prompt_settings_raw:
                loaded_alert_policy_prompt = str(prompt_settings_raw.get("alert_policy_prompt") or "")
                loaded_prompt_default_fields.add("alert_policy_prompt")
            if "bookmark_enabled" in prompt_settings_raw:
                loaded_default_bookmark_enabled = bool(prompt_settings_raw.get("bookmark_enabled"))
                loaded_prompt_default_fields.add("bookmark_enabled")
            if "bookmark_cooldown_sec" in prompt_settings_raw:
                raw_cooldown = self._coerce_float(prompt_settings_raw.get("bookmark_cooldown_sec"))
                loaded_default_bookmark_cooldown_sec = max(0.0, raw_cooldown if raw_cooldown is not None else 0.0)
                loaded_prompt_default_fields.add("bookmark_cooldown_sec")
            if "capture_selector_bias" in prompt_settings_raw:
                loaded_default_capture_selector_bias = self._normalize_selector_bias(
                    prompt_settings_raw.get("capture_selector_bias")
                )
                if loaded_default_capture_selector_bias is not None:
                    loaded_prompt_default_fields.add("capture_selector_bias")
            if "json_alert_prompt" in prompt_settings_raw:
                loaded_default_json_alert_prompt = self._normalize_json_alert_prompt(
                    prompt_settings_raw.get("json_alert_prompt")
                )
                loaded_prompt_default_fields.add("json_alert_prompt")
            rollup_prompts_raw = prompt_settings_raw.get("rollup_prompts")
            if isinstance(rollup_prompts_raw, Mapping):
                for raw_level, raw_prompt in rollup_prompts_raw.items():
                    level = self._normalize_rollup_level(raw_level)
                    if level in {"L1", "L2", "L3"}:
                        loaded_rollup_prompts[level] = str(raw_prompt or "").strip()
                        loaded_prompt_default_fields.add(f"rollup_prompts.{level}")
            # Backward-compatibility for flat keys.
            for level in ("L1", "L2", "L3"):
                flat_key = f"rollup_{level.lower()}_system_prompt"
                if level in loaded_rollup_prompts:
                    continue
                if flat_key in prompt_settings_raw:
                    loaded_rollup_prompts[level] = str(prompt_settings_raw.get(flat_key) or "").strip()
                    loaded_prompt_default_fields.add(f"rollup_prompts.{level}")
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
                    if "model_hint" in channel_payload:
                        channel_model_hint = str(channel_payload.get("model_hint") or "").strip()
                        if channel_model_hint:
                            parsed_channel_payload["model_hint"] = channel_model_hint
                    if "capture_selector_bias" in channel_payload:
                        channel_selector_bias = self._normalize_selector_bias(
                            channel_payload.get("capture_selector_bias")
                        )
                        if channel_selector_bias is not None:
                            parsed_channel_payload["capture_selector_bias"] = channel_selector_bias
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
            self.road_scene_calibrations = loaded_road_scene_calibrations
            self.capture_activity_baselines = loaded_capture_baselines
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
            if loaded_default_capture_selector_bias is not None:
                self.default_capture_selector_bias = loaded_default_capture_selector_bias
            if loaded_default_json_alert_prompt is not None:
                self.default_json_alert_prompt = loaded_default_json_alert_prompt or self.default_json_alert_prompt
            for level, prompt_text in loaded_rollup_prompts.items():
                self.rollup_llm_system_prompts[level] = prompt_text
            loaded_revision = _parse_optional_int(payload.get("revision"))
            self.summary_state_revision = int(
                loaded_revision
                if loaded_revision is not None
                else (1 if isinstance(prompt_settings_raw, Mapping) else 0)
            )
            self.summary_state_last_success_at = self._coerce_float(payload.get("updated_at"))
            self.summary_state_last_error = None
            self._persisted_prompt_default_fields = set(loaded_prompt_default_fields)

    def persist_summary_state(self) -> bool:
        with self.cache_lock:
            return self._persist_summary_state_locked()

    def get_persisted_capture_baseline(self, channel_id: int) -> Optional[Dict[str, Any]]:
        with self.cache_lock:
            state = self.capture_activity_baselines.get(int(channel_id))
            return dict(state) if isinstance(state, Mapping) else None

    def note_capture_baseline(self, channel_id: int, snapshot: Mapping[str, Any]) -> None:
        """Record the channel's measured motion baseline for persistence/prompting."""

        level = self._coerce_float(snapshot.get("level"))
        if level is None or level < 0:
            return
        record = {
            "level": float(level),
            "dev": max(0.0, float(self._coerce_float(snapshot.get("dev")) or 0.0)),
            "buckets": max(0, int(_parse_optional_int(snapshot.get("buckets")) or 0)),
            "updated_at": time.time(),
        }
        with self.cache_lock:
            self.capture_activity_baselines[int(channel_id)] = record

    @staticmethod
    def _activity_level_label(level: float) -> str:
        if level < 0.01:
            return "low"
        if level < 0.05:
            return "moderate"
        return "high"

    def _render_capture_homeostasis_prompt(self, channel_id: int) -> str:
        with self.cache_lock:
            state = self.capture_activity_baselines.get(int(channel_id))
            baseline = dict(state) if isinstance(state, Mapping) else None
        if not baseline:
            return ""
        buckets = max(0, int(_parse_optional_int(baseline.get("buckets")) or 0))
        if buckets < _CAPTURE_BASELINE_WARMUP_BUCKETS:
            return ""
        level = float(self._coerce_float(baseline.get("level")) or 0.0)
        minutes = max(1, int(round(buckets / 60.0)))
        return (
            "Measured motion homeostasis (server-computed, trusted): typical per-second motion on this "
            f"channel is {self._activity_level_label(level)} (activity level {level:.4f}, ~{minutes} min of history). "
            "VECTOR_SIGNALS_JSON.capture_attention reports how current snapshots compare to this norm."
        )

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
            normalized_state = dict(raw_state)
            if "last_restore_error" in normalized_state:
                normalized_state["last_restore_error"] = _safe_error_text(
                    normalized_state.get("last_restore_error"),
                    500,
                )
            desired[channel_id] = normalized_state
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
        # Different channel operations use different side-effect locks.  Serialize
        # this read/modify/write map so starting B cannot overwrite concurrently
        # persisted desired state for A.
        with self._desired_live_sessions_lock:
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
                current["last_restore_error"] = _safe_error_text(restore_error, 500)
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
                message = _safe_error_text(exc, 500) or exc.__class__.__name__
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
        runs_were_present = channel_id in self.summary_runs
        previous_runs = copy.deepcopy(self.summary_runs.get(channel_id, []))
        previous_active_run_id = self.active_summary_runs.get(channel_id)
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
        if not self._persist_summary_state_locked():
            if runs_were_present:
                self.summary_runs[channel_id] = previous_runs
            else:
                self.summary_runs.pop(channel_id, None)
            if previous_active_run_id is not None:
                self.active_summary_runs[channel_id] = previous_active_run_id
            else:
                self.active_summary_runs.pop(channel_id, None)
            persistence_error = (
                self.summary_state_last_error
                or "runtime state backend rejected the new summary run"
            )
            raise RuntimeError(
                f"Summary session was not started because its run state could not be persisted: {persistence_error}"
            )
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

    def _split_rollup_operator_output(self, value: object) -> Tuple[str, Dict[str, Any]]:
        """Separate human-facing Markdown from the machine-only memory block."""

        text = str(value or "").strip()
        if not text:
            return "", {}
        marker = "MEMORY_UPDATE_JSON:"
        marker_index = text.upper().find(marker)
        operator_summary = text[:marker_index].strip() if marker_index >= 0 else text
        raw_memory = self._extract_json_marker_payload(text, marker)
        if not isinstance(raw_memory, Mapping):
            return operator_summary, {}
        if not raw_memory:
            return operator_summary, {}
        normalized = self._extract_memory_update(
            marker + "\n" + json.dumps(dict(raw_memory), ensure_ascii=False)
        )
        normalized.pop("text", None)
        return operator_summary, normalized

    @staticmethod
    def _normalize_rollup_operator_headings(value: object) -> str:
        """Normalize harmless small-model heading drift to the v2 contract."""

        heading_specs = (
            ("Period Overview", r"period\s+overview"),
            ("Routine and Behavior", r"routine\s+(?:and|&)\s+behavio[u]?r"),
            (
                "Notable Observations and Exceptions",
                r"notable\s+observations?\s+(?:and|&)\s+exceptions?",
            ),
            ("Alerts and Meaning", r"alerts?\s+(?:and|&)\s+meaning"),
            (
                "Coverage and Interruptions",
                r"coverage\s+(?:and|&)\s+interruptions?",
            ),
            ("Operator Takeaway", r"operator\s+takeaway"),
        )
        normalized_lines: List[str] = []
        for line in str(value or "").splitlines():
            replacement: Optional[str] = None
            for canonical, body_pattern in heading_specs:
                if re.match(
                    rf"^\s*#{{2,3}}\s*(?:\d+\s*[.)-]\s*)?{body_pattern}\s*[:\-–—]?\s*$",
                    line,
                    flags=re.IGNORECASE,
                ):
                    replacement = f"### {canonical}"
                    break
            normalized_lines.append(replacement or line)
        return "\n".join(normalized_lines).strip()

    @classmethod
    def _rollup_operator_contract_valid(cls, value: object) -> bool:
        normalized = cls._normalize_rollup_operator_headings(value)
        expected = [
            "### Period Overview",
            "### Routine and Behavior",
            "### Notable Observations and Exceptions",
            "### Alerts and Meaning",
            "### Coverage and Interruptions",
            "### Operator Takeaway",
        ]
        observed = [line.strip() for line in normalized.splitlines() if line.strip() in expected]
        return observed == expected

    def _increment_rollup_status_counter(self, name: str) -> None:
        key = str(name or "").strip()
        if not key:
            return
        with self.cache_lock:
            self._rollup_scheduler_status[key] = int(
                self._rollup_scheduler_status.get(key) or 0
            ) + 1

    @staticmethod
    def _rollup_operator_semantic_guard_issues(value: object) -> List[str]:
        """Catch a few high-risk claims that sampled CCTV summaries cannot support."""

        text = " ".join(str(value or "").split()).casefold()
        issues: List[str] = []
        patterns = (
            (
                "intent_followup",
                r"\b(?:confirm|determine|verify|establish)\s+(?:the\s+)?(?:person(?:'s)?\s+)?intent\b",
            ),
            (
                "complete_coverage_claim",
                r"\b(?:no\s+blind\s+spots?|no\s+missing\s+coverage|complete\s+(?:visual\s+)?coverage|coverage\s+(?:was|is)\s+complete)\b",
            ),
            (
                "categorical_safety_absence",
                r"\bno\s+(?:immediate\s+)?(?:safety|security)(?:\s+(?:or|and)\s+(?:safety|security))?\s+(?:concerns?|hazards?|risks?|issues?)\b",
            ),
        )
        for issue, pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            if issue == "complete_coverage_claim":
                prefix = text[max(0, match.start() - 56) : match.start()]
                if re.search(
                    r"\b(?:not|cannot|can't|do\s+not|does\s+not|never)\b[^.]{0,48}$",
                    prefix,
                    flags=re.IGNORECASE,
                ):
                    continue
            issues.append(issue)
        return issues

    @classmethod
    def _sanitize_rollup_operator_overclaims(cls, value: object) -> str:
        """Replace a narrow set of unsafe sampled-evidence overclaims."""

        output: List[str] = []
        for raw_line in str(value or "").splitlines():
            line = raw_line
            issues = set(cls._rollup_operator_semantic_guard_issues(line))
            prefix_match = re.match(r"^(\s*(?:[-*•]\s+|\d+[.)]\s+)?)", line)
            prefix = prefix_match.group(1) if prefix_match else ""
            if "complete_coverage_claim" in issues:
                line = (
                    prefix
                    + "No camera/feed interruption was recorded in metadata; sampled frames are partial evidence."
                )
                issues.discard("complete_coverage_claim")
            if "categorical_safety_absence" in issues:
                line = (
                    prefix
                    + "No immediate safety/security issue was identified in the sampled observations; this does not establish absence outside them."
                )
                issues.discard("categorical_safety_absence")
            if "intent_followup" in issues:
                line = re.sub(
                    r"\b(?:confirm|determine|verify|establish)\s+(?:the\s+)?(?:person(?:'s)?\s+)?intent(?:\s+or\s+context)?\b",
                    "review the observable sequence and context",
                    line,
                    flags=re.IGNORECASE,
                )
            output.append(line)
        return "\n".join(output).strip()

    @staticmethod
    def _is_legacy_fallback_rollup(value: object) -> bool:
        text = " ".join(str(value or "").strip().split()).lower()
        return bool(re.match(r"^l[123] rollup from l[012]:", text))

    def _update_channel_routine_context(
        self,
        channel_id: int,
        rollup_id: str,
        summary_text: object,
        window_end: Optional[float],
        level: Optional[str] = None,
        memory_update: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if isinstance(memory_update, Mapping):
            if not memory_update:
                return
            extracted_memory = self._extract_memory_update(
                "MEMORY_UPDATE_JSON:\n"
                + json.dumps(dict(memory_update), ensure_ascii=False)
            )
        else:
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
        road_cues, road_frame_scores, road_scene, road_health = self._road_cv_vector_signals(
            int(channel_id),
            frames,
        )
        health.update(road_health)
        road_episodes = self._road_episode_vector_signals(
            int(channel_id),
            road_cues,
            clip_signals,
            now_ms=batch_end_ms,
        )
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
        if road_frame_scores:
            bundle["road_cv_frame_scores"] = road_frame_scores
        if road_episodes:
            bundle["road_episodes"] = road_episodes
        if road_scene:
            bundle["road_cv_scene"] = road_scene
        return self._compact_vector_signal(bundle)

    def _capture_attention_signal(self, frames: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        """Summarize capture-decider modes for the exact snapshot numbering the model sees."""

        seconds: List[Dict[str, Any]] = []
        baseline_info: Optional[Dict[str, Any]] = None
        policy = ""
        for index, frame in enumerate(frames, start=1):
            if not isinstance(frame, Mapping):
                continue
            selection = frame.get("capture_selection")
            if not isinstance(selection, Mapping):
                continue
            policy = policy or str(selection.get("policy") or "")
            baseline_raw = selection.get("baseline")
            if baseline_info is None and isinstance(baseline_raw, Mapping):
                baseline_info = {
                    "level": self._coerce_float(baseline_raw.get("level")),
                    "warmup": bool(baseline_raw.get("warmup")),
                }
            mode = str(selection.get("selection_mode") or "").strip().lower()
            activity_x = self._coerce_float(selection.get("activity_x"))
            if mode == "burst":
                pass
            elif mode == "normal" and activity_x is not None and float(activity_x) >= 3.0:
                pass
            else:
                continue
            entry: Dict[str, Any] = {"snapshot": int(index), "mode": mode}
            if activity_x is not None:
                entry["activity_x"] = round(float(activity_x), 2)
            if mode == "burst":
                entry["blur"] = "expected_motion"
                if isinstance(selection.get("companion"), Mapping) or frame.get("burst_companion"):
                    entry["sharper_companion"] = True
            seconds.append(entry)
        if not seconds:
            return {}
        # Token discipline: bursts first, then the strongest normals, max six.
        seconds.sort(
            key=lambda item: (
                0 if item.get("mode") == "burst" else 1,
                -float(item.get("activity_x") or 0.0),
            )
        )
        seconds = seconds[:6]
        seconds.sort(key=lambda item: int(item.get("snapshot") or 0))
        signal: Dict[str, Any] = {
            "policy": policy or CAPTURE_APEX_POLICY,
            "seconds": seconds,
        }
        if baseline_info is not None:
            signal["baseline"] = baseline_info
        return signal

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
        homeostasis = self._render_capture_homeostasis_prompt(channel_id)
        vector_prompt = self._render_vector_signal_prompt(vector_signal)
        parts = [
            part
            for part in (
                base,
                alert_policy,
                routine,
                homeostasis,
                vector_prompt,
                LIVE_OBSERVATION_STATE_PROMPT,
                rendered_json_prompt,
            )
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

    @staticmethod
    def _normalize_selector_bias(value: object, *, strict: bool = False) -> Optional[str]:
        bias = str(value or "").strip().lower()
        if not bias:
            return None
        if bias not in CAPTURE_SELECTOR_BIASES:
            if strict:
                raise ValueError(
                    "capture_selector_bias must be one of: " + ", ".join(CAPTURE_SELECTOR_BIASES)
                )
            return None
        return bias

    def _get_capture_selector_bias_locked(self, channel_id: Optional[int] = None) -> str:
        if channel_id is not None:
            overrides = self.channel_prompt_overrides.get(int(channel_id))
            if isinstance(overrides, Mapping) and "capture_selector_bias" in overrides:
                bias = self._normalize_selector_bias(overrides.get("capture_selector_bias"))
                if bias:
                    return bias
        return self._normalize_selector_bias(self.default_capture_selector_bias) or "auto"

    def get_capture_selector_bias(self, channel_id: Optional[int] = None) -> str:
        with self.cache_lock:
            return self._get_capture_selector_bias_locked(channel_id)

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
                "capture_selector_bias": self._normalize_selector_bias(self.default_capture_selector_bias) or "auto",
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
            effective_capture_selector_bias = self._get_capture_selector_bias_locked(channel_id)
            effective_bookmark = self._get_channel_bookmark_settings_locked(channel_id)
            active_memory = ""
            if channel_id is not None:
                current_memory = self.channel_routine_context.get(int(channel_id))
                if isinstance(current_memory, Mapping):
                    routine_text = str(current_memory.get("routine") or "").strip()
                    if routine_text:
                        active_memory = self._render_channel_memory_prompt(routine_text)
            raw_channel_overrides = (
                self.channel_prompt_overrides.get(int(channel_id))
                if channel_id is not None
                else None
            )
            channel_overrides = (
                dict(raw_channel_overrides)
                if isinstance(raw_channel_overrides, Mapping)
                else {}
            )
            override_fields: Set[str] = {
                field
                for field in (
                    "stream_system_prompt",
                    "alert_policy_prompt",
                    "capture_interval_sec",
                    "capture_selector_bias",
                    "bookmark_enabled",
                    "bookmark_cooldown_sec",
                    "json_alert_prompt",
                )
                if field in channel_overrides
            }
            channel_rollup_overrides = channel_overrides.get("rollup_prompts")
            if isinstance(channel_rollup_overrides, Mapping):
                for level in ("L1", "L2", "L3"):
                    if level in channel_rollup_overrides:
                        override_fields.add(f"rollup_prompts.{level}")
            has_channel_override = bool(override_fields)
            persisted_default_fields = set(self._persisted_prompt_default_fields)
            persistence = {
                "backend": self.summary_state_backend,
                "persisted": self.summary_state_last_error is None
                and self.summary_state_last_success_at is not None,
                "revision": int(self.summary_state_revision),
                "last_success_at": self.summary_state_last_success_at,
                "last_error": self.summary_state_last_error,
                "dirty": bool(self._summary_state_dirty),
            }

        def setting_source(field: str) -> str:
            if field in override_fields:
                return "channel_override"
            if field in persisted_default_fields:
                return "persisted_runtime_default"
            return "config_default"

        setting_sources = {
            "stream_system_prompt": setting_source("stream_system_prompt"),
            "alert_policy_prompt": setting_source("alert_policy_prompt"),
            "capture_interval_sec": setting_source("capture_interval_sec"),
            "capture_selector_bias": setting_source("capture_selector_bias"),
            "bookmark_enabled": setting_source("bookmark_enabled"),
            "bookmark_cooldown_sec": setting_source("bookmark_cooldown_sec"),
            "json_alert_prompt": setting_source("json_alert_prompt"),
            "rollup_prompts": {
                level: setting_source(f"rollup_prompts.{level}")
                for level in ("L1", "L2", "L3")
            },
        }
        prompt_layers = {
            "stream": {
                "editable_prompt": effective_stream_prompt,
                "source": setting_sources["stream_system_prompt"],
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
                "source": setting_sources["alert_policy_prompt"],
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
                "source": setting_sources["json_alert_prompt"],
                "notes": [
                    "Advanced machine-readable output contract.",
                    "Use Alert Criteria for ordinary operator watch conditions.",
                    "This JSON layer is appended last and is parsed for VLM alert events.",
                ],
            },
            "rollups": {
                level: {
                    "editable_prompt": effective_rollup_prompts.get(level, ""),
                    "source": setting_sources["rollup_prompts"][level],
                    "backend_instructions": self._rollup_backend_instruction_text(level),
                    "active_memory": active_memory,
                    "notes": [
                        "The editable prompt is the system prompt.",
                        "The operator-format contract is always appended to the system prompt.",
                        "Alerts remain a separate operator section and must explain their observable meaning.",
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
            "capture_selector_bias": effective_capture_selector_bias,
            "bookmark_enabled": bool(effective_bookmark.get("bookmark_enabled")),
            "bookmark_cooldown_sec": float(effective_bookmark.get("bookmark_cooldown_sec") or 0.0),
            "json_alert_prompt": str(effective_bookmark.get("json_alert_prompt") or DEFAULT_ALERTS_JSON_PROMPT),
            "defaults": defaults,
            "has_channel_override": has_channel_override,
            "override_fields": sorted(override_fields),
            "setting_sources": setting_sources,
            "persistence": persistence,
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
        capture_selector_bias: Optional[str] = None,
        clear_override_fields: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        changed = False
        target_channel_id = int(channel_id) if channel_id is not None else None
        normalized_selector_bias: Optional[str] = None
        if capture_selector_bias is not None:
            normalized_selector_bias = self._normalize_selector_bias(capture_selector_bias, strict=True)
        clear_fields: Set[str] = set()
        if clear_override_fields is not None:
            if isinstance(clear_override_fields, (str, bytes)):
                raise ValueError("clear_override_fields must be a list of setting names")
            allowed_clear_fields = {
                "stream_system_prompt",
                "alert_policy_prompt",
                "capture_interval_sec",
                "capture_selector_bias",
                "bookmark_enabled",
                "bookmark_cooldown_sec",
                "json_alert_prompt",
                "rollup_prompts.L1",
                "rollup_prompts.L2",
                "rollup_prompts.L3",
            }
            for raw_field in clear_override_fields:
                field = str(raw_field or "").strip()
                if field.lower().startswith("rollup_prompts."):
                    level = self._normalize_rollup_level(field.rsplit(".", 1)[-1])
                    field = f"rollup_prompts.{level}"
                if field not in allowed_clear_fields:
                    raise ValueError(f"Unsupported prompt override field: {field or '<empty>'}")
                clear_fields.add(field)
            if clear_fields and target_channel_id is None:
                raise ValueError("clear_override_fields requires a channel_id")
        updated_fields: Set[str] = set()
        if stream_system_prompt is not None:
            updated_fields.add("stream_system_prompt")
        if alert_policy_prompt is not None:
            updated_fields.add("alert_policy_prompt")
        if json_alert_prompt is not None:
            updated_fields.add("json_alert_prompt")
        if bookmark_enabled is not None:
            updated_fields.add("bookmark_enabled")
        if bookmark_cooldown_sec is not None:
            updated_fields.add("bookmark_cooldown_sec")
        if normalized_selector_bias is not None:
            updated_fields.add("capture_selector_bias")
        if isinstance(rollup_prompts, Mapping):
            for raw_level in rollup_prompts:
                level = self._normalize_rollup_level(raw_level)
                if level in {"L1", "L2", "L3"}:
                    updated_fields.add(f"rollup_prompts.{level}")
        overlapping_fields = clear_fields & updated_fields
        if overlapping_fields:
            raise ValueError(
                "A setting cannot be updated and reset to inherited in the same request: "
                + ", ".join(sorted(overlapping_fields))
            )
        channel_overrides: Dict[str, Any] = {}
        with self.cache_lock:
            previous_state = {
                "system_prompt": self.system_prompt,
                "alert_policy_prompt": self.alert_policy_prompt,
                "default_json_alert_prompt": self.default_json_alert_prompt,
                "default_bookmark_enabled": self.default_bookmark_enabled,
                "default_bookmark_cooldown_sec": self.default_bookmark_cooldown_sec,
                "default_capture_selector_bias": self.default_capture_selector_bias,
                "rollup_llm_system_prompts": copy.deepcopy(self.rollup_llm_system_prompts),
                "channel_override_present": bool(
                    target_channel_id is not None
                    and target_channel_id in self.channel_prompt_overrides
                ),
                "channel_override": copy.deepcopy(
                    self.channel_prompt_overrides.get(target_channel_id)
                    if target_channel_id is not None
                    else None
                ),
            }
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
                if normalized_selector_bias is not None:
                    if normalized_selector_bias != str(self.default_capture_selector_bias or "auto"):
                        self.default_capture_selector_bias = normalized_selector_bias
                        changed = True
            else:
                current_overrides_raw = self.channel_prompt_overrides.get(target_channel_id)
                channel_overrides = dict(current_overrides_raw) if isinstance(current_overrides_raw, Mapping) else {}
                for field in sorted(clear_fields):
                    if field.startswith("rollup_prompts."):
                        level = field.rsplit(".", 1)[-1]
                        existing_rollups_raw = channel_overrides.get("rollup_prompts")
                        channel_rollups = (
                            dict(existing_rollups_raw)
                            if isinstance(existing_rollups_raw, Mapping)
                            else {}
                        )
                        if level in channel_rollups:
                            channel_rollups.pop(level, None)
                            changed = True
                            if channel_rollups:
                                channel_overrides["rollup_prompts"] = channel_rollups
                            else:
                                channel_overrides.pop("rollup_prompts", None)
                    elif field in channel_overrides:
                        channel_overrides.pop(field, None)
                        changed = True
                if stream_system_prompt is not None:
                    next_stream_prompt = str(stream_system_prompt)
                    if (
                        "stream_system_prompt" not in channel_overrides
                        or next_stream_prompt != str(channel_overrides.get("stream_system_prompt") or "")
                    ):
                        channel_overrides["stream_system_prompt"] = next_stream_prompt
                        changed = True
                if alert_policy_prompt is not None:
                    next_alert_policy_prompt = str(alert_policy_prompt)
                    if (
                        "alert_policy_prompt" not in channel_overrides
                        or next_alert_policy_prompt != str(channel_overrides.get("alert_policy_prompt") or "")
                    ):
                        channel_overrides["alert_policy_prompt"] = next_alert_policy_prompt
                        changed = True
                if json_alert_prompt is not None:
                    next_json_prompt = self._normalize_json_alert_prompt(json_alert_prompt)
                    if (
                        "json_alert_prompt" not in channel_overrides
                        or next_json_prompt != str(channel_overrides.get("json_alert_prompt") or "")
                    ):
                        channel_overrides["json_alert_prompt"] = next_json_prompt
                        changed = True
                if bookmark_enabled is not None:
                    next_enabled = bool(bookmark_enabled)
                    if (
                        "bookmark_enabled" not in channel_overrides
                        or next_enabled != bool(channel_overrides.get("bookmark_enabled"))
                    ):
                        channel_overrides["bookmark_enabled"] = next_enabled
                        changed = True
                if bookmark_cooldown_sec is not None:
                    next_cooldown = max(0.0, float(bookmark_cooldown_sec))
                    if (
                        "bookmark_cooldown_sec" not in channel_overrides
                        or next_cooldown != float(channel_overrides.get("bookmark_cooldown_sec") or 0.0)
                    ):
                        channel_overrides["bookmark_cooldown_sec"] = next_cooldown
                        changed = True
                if normalized_selector_bias is not None:
                    if (
                        "capture_selector_bias" not in channel_overrides
                        or normalized_selector_bias != str(channel_overrides.get("capture_selector_bias") or "")
                    ):
                        channel_overrides["capture_selector_bias"] = normalized_selector_bias
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
                        if level not in channel_rollups or next_prompt != str(channel_rollups.get(level) or ""):
                            channel_rollups[level] = next_prompt
                            channel_overrides["rollup_prompts"] = channel_rollups
                            changed = True
            if target_channel_id is not None and changed:
                if channel_overrides:
                    self.channel_prompt_overrides[target_channel_id] = channel_overrides
                else:
                    self.channel_prompt_overrides.pop(target_channel_id, None)
            if changed:
                if not self._persist_summary_state_locked():
                    self.system_prompt = str(previous_state["system_prompt"] or "")
                    self.alert_policy_prompt = str(previous_state["alert_policy_prompt"] or "")
                    self.default_json_alert_prompt = str(previous_state["default_json_alert_prompt"] or "")
                    self.default_bookmark_enabled = bool(previous_state["default_bookmark_enabled"])
                    self.default_bookmark_cooldown_sec = float(
                        previous_state["default_bookmark_cooldown_sec"] or 0.0
                    )
                    self.default_capture_selector_bias = str(
                        previous_state["default_capture_selector_bias"] or "auto"
                    )
                    self.rollup_llm_system_prompts = copy.deepcopy(
                        previous_state["rollup_llm_system_prompts"]
                    )
                    if target_channel_id is not None:
                        if previous_state["channel_override_present"]:
                            previous_override = previous_state["channel_override"]
                            self.channel_prompt_overrides[target_channel_id] = (
                                copy.deepcopy(previous_override)
                                if isinstance(previous_override, Mapping)
                                else {}
                            )
                        else:
                            self.channel_prompt_overrides.pop(target_channel_id, None)
                    persistence_error = (
                        self.summary_state_last_error
                        or "runtime state backend rejected the update"
                    )
                    raise RuntimeError(
                        f"Prompt settings were not persisted; in-memory changes were rolled back: {persistence_error}"
                    )
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

    @classmethod
    def _collect_rollup_alert_events(
        cls,
        nodes: Sequence[Mapping[str, Any]],
        max_items: int = 16,
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str, int]] = set()
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            for event in cls._compact_alert_events(node.get("alert_events")):
                key = (
                    str(event.get("title") or "").strip().casefold(),
                    str(event.get("severity") or "").strip().casefold(),
                    int(_parse_optional_int(event.get("timestamp_ms")) or 0),
                )
                if key in seen:
                    continue
                seen.add(key)
                events.append(event)
                if len(events) >= max(1, int(max_items)):
                    return events
        return events

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
        duration = (
            f"about {max(1, int(window_sec // 60))} minutes"
            if window_sec < 3600
            else f"about {max(1, int(window_sec // 3600))} hours"
        )
        evidence = f"{item_count} source observations"
        if frame_count > 0:
            evidence += f" covering {frame_count} frames"
        alert_text = self._format_alert_counts(alert_counts)
        missing_items = self._coerce_memory_items(
            signal_digest.get("missing_data") if isinstance(signal_digest, Mapping) else None,
            max_items=3,
            max_len=180,
        )
        coverage_text = (
            "Reported coverage issues: " + "; ".join(missing_items)
            if missing_items
            else "No structured coverage interruption was recorded in the available source metadata."
        )
        alert_section = (
            f"The source contains {alert_text}. Their behavioral meaning requires the lower-level observations or a completed semantic rollup."
            if alert_text
            else "No structured alerts were recorded in the available source metadata."
        )
        return "\n\n".join(
            [
                "### Period Overview\n"
                f"Semantic {level} aggregation is queued for this {duration} window. "
                f"EVA retained {evidence} across {len(run_ids)} run(s).",
                "### Routine and Behavior\n"
                "The background semantic pass will describe routine and behavioral development. Source observations remain available while it runs.",
                "### Notable Observations and Exceptions\n"
                "The semantic exception narrative is pending; drill down for the current source observations.",
                "### Alerts and Meaning\n" + alert_section,
                "### Coverage and Interruptions\n" + coverage_text,
                "### Operator Takeaway\n"
                "Background aggregation is pending. Source observations can be reviewed now; the period-level narrative will replace this card automatically.",
            ]
        )

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
        raw_summary = str(entry.get("operator_summary") or entry.get("summary") or "").strip()
        summary, embedded_memory = self._split_rollup_operator_output(raw_summary)
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
        format_version = _parse_optional_int(entry.get("format_version")) or 1
        summary_kind = str(entry.get("summary_kind") or "").strip()
        if not summary_kind:
            summary_kind = "llm_cached" if format_version >= ROLLUP_OPERATOR_FORMAT_VERSION else "legacy_cached"
        if self._is_legacy_fallback_rollup(summary):
            summary_kind = "degraded"
        generation_status = str(entry.get("generation_status") or "").strip()
        if not generation_status:
            generation_status = "cached" if summary_kind in {"llm", "llm_cached"} else "stale"
        if self._rollup_operator_semantic_guard_issues(summary):
            summary_kind = "degraded"
            generation_status = "semantic_guard_rejected"
        memory_raw = entry.get("memory_update")
        memory_update = dict(memory_raw) if isinstance(memory_raw, Mapping) else embedded_memory
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
        alert_events = self._compact_alert_events(entry.get("alert_events"))
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
            "operator_summary": summary,
            "memory_update": memory_update,
            "summary_kind": summary_kind,
            "generation_status": generation_status,
            "format_version": int(format_version),
            "created_at": float(created_at),
            "signal_digest": dict(signal_digest),
            **alert_meta,
        }
        if alert_delivery_breakdown:
            normalized["alert_delivery_breakdown"] = alert_delivery_breakdown
        if alert_events:
            normalized["alert_events"] = alert_events
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
        cutoff = self._rollup_retention_cutoff()
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
        cutoff = self._rollup_retention_cutoff()
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
        bulk_saver = getattr(state_store, "save_rollups", None)
        if callable(bulk_saver) and retained_entries:
            try:
                promoted = int(bulk_saver(retained_entries) or 0)
                with self.cache_lock:
                    self._rollup_scheduler_status["durable_rollups_promoted"] = promoted
                    self._rollup_scheduler_status["durable_store_last_error"] = None
            except Exception as exc:
                with self.cache_lock:
                    self._rollup_scheduler_status["durable_store_last_error"] = (
                        _safe_error_text(exc, 240) or exc.__class__.__name__
                    )

    def persist_rollup_cache(self) -> None:
        with self.cache_lock:
            self._persist_rollup_cache_locked()

    def _get_cached_rollup_record(self, rollup_id: str) -> Optional[Dict[str, Any]]:
        key = str(rollup_id or "").strip()
        if not key:
            return None
        with self.cache_lock:
            cached = self.rollup_summary_cache.get(key)
        if isinstance(cached, Mapping):
            return dict(cached)
        state_store = getattr(self, "runtime_state_store", None)
        loader = getattr(state_store, "load_rollup", None)
        if not callable(loader):
            return None
        try:
            durable = loader(key)
        except Exception as exc:
            with self.cache_lock:
                self._rollup_scheduler_status["durable_store_last_error"] = (
                    _safe_error_text(exc, 240) or exc.__class__.__name__
                )
            return None
        if not isinstance(durable, Mapping):
            return None
        normalized = self._normalize_cached_rollup_entry(durable)
        if normalized is None:
            return None
        cutoff = self._rollup_retention_cutoff()
        created = self._coerce_float(normalized.get("created_at"))
        if cutoff is not None and created is not None and created < cutoff:
            return None
        with self.cache_lock:
            self.rollup_summary_cache[key] = normalized
            while len(self.rollup_summary_cache) > self.rollup_summary_cache_limit:
                self.rollup_summary_cache.pop(next(iter(self.rollup_summary_cache)), None)
            self._rollup_scheduler_status["durable_store_last_error"] = None
            self._rollup_scheduler_status["durable_rollup_cache_hits"] = int(
                self._rollup_scheduler_status.get("durable_rollup_cache_hits") or 0
            ) + 1
        return dict(normalized)

    def _save_durable_rollup(self, payload: Mapping[str, Any]) -> None:
        state_store = getattr(self, "runtime_state_store", None)
        saver = getattr(state_store, "save_rollup", None)
        if not callable(saver):
            return
        try:
            saver(payload)
            now = time.time()
            cutoff = self._rollup_retention_cutoff()
            pruner = getattr(state_store, "prune_rollups", None)
            pruned = 0
            if (
                cutoff is not None
                and callable(pruner)
                and now - self._rollup_durable_last_prune_at >= 3600.0
            ):
                pruned = int(pruner(cutoff) or 0)
                self._rollup_durable_last_prune_at = now
            with self.cache_lock:
                self._rollup_scheduler_status["durable_store_last_error"] = None
                self._rollup_scheduler_status["durable_rollups_written"] = int(
                    self._rollup_scheduler_status.get("durable_rollups_written") or 0
                ) + 1
                if pruned:
                    self._rollup_scheduler_status["durable_rollups_pruned"] = int(
                        self._rollup_scheduler_status.get("durable_rollups_pruned") or 0
                    ) + pruned
        except Exception as exc:
            with self.cache_lock:
                self._rollup_scheduler_status["durable_store_last_error"] = (
                    _safe_error_text(exc, 240) or exc.__class__.__name__
                )

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
        self._save_durable_rollup(normalized_payload)
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
            hot_entries = [dict(val) for val in self.rollup_summary_cache.values() if isinstance(val, Mapping)]
        entries: List[Dict[str, Any]] = []
        state_store = getattr(self, "runtime_state_store", None)
        durable_loader = getattr(state_store, "list_rollups", None)
        if callable(durable_loader):
            try:
                durable_entries = durable_loader(
                    channel_id=channel_id,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    levels=("L1", "L2", "L3"),
                    limit=50000,
                )
                if isinstance(durable_entries, Sequence) and not isinstance(
                    durable_entries,
                    (str, bytes, bytearray),
                ):
                    durable_rows = [
                        dict(entry)
                        for entry in durable_entries
                        if isinstance(entry, Mapping)
                    ]
                    entries.extend(durable_rows)
                    # The durable row is authoritative and also warms the small
                    # in-process cache, avoiding one DB lookup per rollup during
                    # the synthesis walk that follows this range read.
                    with self.cache_lock:
                        for durable_row in durable_rows:
                            durable_id = self._rollup_identity_key(durable_row)
                            if durable_id:
                                self.rollup_summary_cache[durable_id] = durable_row
                        while len(self.rollup_summary_cache) > self.rollup_summary_cache_limit:
                            self.rollup_summary_cache.pop(next(iter(self.rollup_summary_cache)), None)
                with self.cache_lock:
                    self._rollup_scheduler_status["durable_store_last_error"] = None
            except Exception as exc:
                with self.cache_lock:
                    self._rollup_scheduler_status["durable_store_last_error"] = (
                        _safe_error_text(exc, 240) or exc.__class__.__name__
                    )
        entries.extend(hot_entries)
        out: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        cutoff = self._rollup_retention_cutoff()
        for entry in entries:
            normalized = self._normalize_cached_rollup_entry(entry)
            if normalized is None:
                continue
            identity = self._rollup_identity_key(normalized)
            if identity and identity in seen:
                continue
            if _parse_optional_int(normalized.get("channel_id")) != channel_id:
                continue
            created = self._coerce_float(normalized.get("created_at"))
            if cutoff is not None and created is not None and created < cutoff:
                continue
            window_start = self._coerce_float(normalized.get("window_start"))
            window_end = self._coerce_float(normalized.get("window_end"))
            if start_ts is not None and (window_end is None or window_end < start_ts):
                continue
            if end_ts is not None and (window_start is None or window_start > end_ts):
                continue
            if identity:
                seen.add(identity)
            out.append(normalized)
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
                    current_semantic_valid = bool(
                        current_kind in {"llm", "llm_cached"}
                        and int(_parse_optional_int(current.get("format_version")) or 1)
                        >= ROLLUP_OPERATOR_FORMAT_VERSION
                        and not self._rollup_operator_semantic_guard_issues(
                            current.get("summary")
                        )
                    )
                    if (
                        current_semantic_valid
                        and generated_kind not in {"llm", "llm_cached"}
                    ):
                        for preserved_field in (
                            "summary",
                            "operator_summary",
                            "memory_update",
                            "summary_kind",
                            "generation_status",
                            "format_version",
                        ):
                            if preserved_field in current:
                                merged[preserved_field] = current.get(preserved_field)
                        if (
                            not generated_signature
                            or generated_signature != current_signature
                        ):
                            # Closed-window source can expand during bounded
                            # backfill. Keep the last useful narrative visible
                            # while the scheduler refreshes it, but mark that it
                            # does not yet include the new source signature.
                            merged["summary_kind"] = "llm_cached"
                            merged["generation_status"] = "refresh_pending"
                            merged["semantic_refresh_pending"] = True
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
            memory_update=latest.get("memory_update") if isinstance(latest.get("memory_update"), Mapping) else {},
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
                memory_update=row.get("memory_update") if isinstance(row.get("memory_update"), Mapping) else {},
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
        system_msg = (
            system_msg.rstrip()
            + "\n\nEVA operator rollup contract v2 (mandatory; overrides older section names):\n"
            + self._rollup_backend_instruction_text(level)
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
        alert_event_lines: List[str] = []
        for event in self._compact_alert_events(node.get("alert_events"))[:16]:
            timestamp_ms = _parse_optional_int(event.get("timestamp_ms"))
            time_label = (
                time.strftime("%H:%M:%S", time.localtime(float(timestamp_ms) / 1000.0))
                if timestamp_ms is not None and timestamp_ms > 0
                else "time n/a"
            )
            severity = self._normalize_alert_severity(event.get("severity"))
            title = self._truncate_text(event.get("title"), 120) or "Event"
            description = self._truncate_text(event.get("description"), 240)
            delivery = str(event.get("delivery_status") or "").strip().lower()
            detail = f"- {time_label} | {severity} | {title}"
            if description:
                detail += f" | {description}"
            if delivery:
                detail += f" | delivery={delivery}"
            alert_event_lines.append(detail)
        delivery_breakdown = self._compact_count_breakdown(node.get("alert_delivery_breakdown"))
        delivery_text = ", ".join(
            f"{key}={value}"
            for key, value in sorted(delivery_breakdown.items())
            if key != "total"
        )
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
                f"Bookmark/delivery outcomes: {delivery_text or 'none recorded'}",
                "Structured alert events:",
                *(alert_event_lines or ["- none"]),
                "",
                "Window signal digest (compact continuity map):",
                window_signal_digest or "none",
                "",
                "Period Overview must begin with:",
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
        workload_class: str = "rollup",
    ) -> Tuple[str, Dict[str, Any], Optional[str]]:
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
            if bool(getattr(self.lm_callback, "eva_workload_class", False)):
                raw_summary = str(
                    self.lm_callback(
                        messages,
                        model_hint,
                        workload_class=workload_class,
                    )
                ).strip()
            else:
                raw_summary = str(self.lm_callback(messages, model_hint)).strip()
            operator_summary, memory_update = self._split_rollup_operator_output(raw_summary)
            operator_summary = self._normalize_rollup_operator_headings(operator_summary)
            contract_valid = bool(
                operator_summary
                and self._rollup_operator_contract_valid(operator_summary)
            )
            semantic_issues = self._rollup_operator_semantic_guard_issues(operator_summary)
            if contract_valid and not semantic_issues:
                return operator_summary, memory_update, None

            if not contract_valid:
                self._increment_rollup_status_counter("invalid_operator_contract")
            if semantic_issues:
                self._increment_rollup_status_counter("semantic_guard_retries")
            self._increment_rollup_status_counter("corrective_retries")
            corrective_messages = list(messages)
            if raw_summary:
                corrective_messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": self._truncate_text(raw_summary, 8000),
                            }
                        ],
                    }
                )
            corrective_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Correct the complete operator report. Return it again with "
                                "exactly these six headings, in this order: ### Period Overview; "
                                "### Routine and Behavior; ### Notable Observations and Exceptions; "
                                "### Alerts and Meaning; ### Coverage and Interruptions; ### Operator Takeaway. "
                                "Keep grounded factual content, but resolve internal presence/absence contradictions. "
                                "Never claim complete coverage, no blind spots, or categorical absence of safety/security concerns from sampled frames. "
                                "Never ask anyone to confirm intent; describe only the observable sequence or uncertainty. "
                                "Append MEMORY_UPDATE_JSON only after all six sections."
                            ),
                        }
                    ],
                }
            )
            if bool(getattr(self.lm_callback, "eva_workload_class", False)):
                retry_raw = str(
                    self.lm_callback(
                        corrective_messages,
                        model_hint,
                        workload_class=workload_class,
                    )
                ).strip()
            else:
                retry_raw = str(self.lm_callback(corrective_messages, model_hint)).strip()
            retry_summary, retry_memory = self._split_rollup_operator_output(retry_raw)
            retry_summary = self._normalize_rollup_operator_headings(retry_summary)
            retry_contract_valid = bool(
                retry_summary
                and self._rollup_operator_contract_valid(retry_summary)
            )
            retry_semantic_issues = self._rollup_operator_semantic_guard_issues(retry_summary)
            if retry_contract_valid and not retry_semantic_issues:
                self._increment_rollup_status_counter("corrective_retry_successes")
                if semantic_issues:
                    self._increment_rollup_status_counter("semantic_guard_retry_successes")
                return retry_summary, retry_memory, None
            if retry_contract_valid and retry_semantic_issues:
                sanitized_summary = self._sanitize_rollup_operator_overclaims(
                    retry_summary
                )
                if not self._rollup_operator_semantic_guard_issues(sanitized_summary):
                    self._increment_rollup_status_counter("corrective_retry_successes")
                    self._increment_rollup_status_counter("semantic_guard_sanitized")
                    return sanitized_summary, retry_memory, None
                self._increment_rollup_status_counter("semantic_guard_failures")
                return fallback_summary, {}, "unsafe_operator_claims"
            if raw_summary or retry_raw:
                return fallback_summary, {}, "invalid_operator_contract"
            return fallback_summary, {}, "empty_operator_summary"
        except Exception as exc:
            error_code = type(exc).__name__
            LOGGER.warning(
                "Rollup synthesis failed channel_id=%s level=%s error=%s",
                channel_id,
                level,
                error_code,
            )
            return fallback_summary, {}, error_code

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
        alerts = self._format_alert_counts(alert_counts)
        return "\n\n".join(
            [
                "### Period Overview\n"
                f"EVA is still collecting context for this {level} window ({item_count} items, {frame_count} frames).",
                "### Routine and Behavior\nNot available until this aggregation window closes.",
                "### Notable Observations and Exceptions\nReview lower-level observations while aggregation is pending.",
                "### Alerts and Meaning\n"
                + (f"Structured alerts collected so far: {alerts}." if alerts else "No structured alerts collected so far."),
                "### Coverage and Interruptions\nCoverage assessment is pending with the aggregation.",
                "### Operator Takeaway\nAggregation in progress; no period-level behavioral conclusion is available yet.",
            ]
        )

    def _apply_rollup_llm_summaries(
        self,
        channel_id: int,
        level: str,
        source_level: str,
        node_children_pairs: Sequence[Tuple[Dict[str, Any], Sequence[Mapping[str, Any]]]],
        max_new: Optional[int] = None,
        workload_class: str = "rollup",
    ) -> None:
        if level not in self.rollup_llm_levels or not node_children_pairs:
            return
        remaining_budget = max(
            1,
            int(max_new if max_new is not None else self.rollup_llm_max_new_per_call),
        )
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
                node["operator_summary"] = node["summary"]
                node["summary_kind"] = "pending_context"
                node["generation_status"] = "pending"
                node["format_version"] = ROLLUP_OPERATOR_FORMAT_VERSION
                continue
            cached = self._get_cached_rollup_record(rollup_id)
            if cached:
                cached_summary = str(cached.get("summary") or "").strip()
                cached_signature = str(cached.get("source_signature") or "").strip()
                cached_format_version = _parse_optional_int(cached.get("format_version")) or 1
                cached_semantic_issues = self._rollup_operator_semantic_guard_issues(
                    cached_summary
                )
                if cached_semantic_issues:
                    self._increment_rollup_status_counter(
                        "cached_semantic_guard_rejections"
                    )
                if (
                    cached_summary
                    and cached_signature
                    and cached_signature == source_signature
                    and cached_format_version >= ROLLUP_OPERATOR_FORMAT_VERSION
                    and not cached_semantic_issues
                ):
                    node["summary"] = cached_summary
                    node["operator_summary"] = cached_summary
                    cached_memory = cached.get("memory_update")
                    node["memory_update"] = dict(cached_memory) if isinstance(cached_memory, Mapping) else {}
                    node["summary_kind"] = "llm_cached"
                    node["generation_status"] = "cached"
                    node["format_version"] = cached_format_version
                    self._update_channel_routine_context(
                        channel_id=channel_id,
                        rollup_id=rollup_id,
                        summary_text=node.get("summary"),
                        window_end=self._coerce_float(node.get("window_end")),
                        level=level,
                        memory_update=node.get("memory_update") if isinstance(node.get("memory_update"), Mapping) else {},
                    )
                    continue
            if remaining_budget <= 0:
                node["generation_status"] = "deferred"
                continue
            fallback = str(node.get("summary") or "").strip()
            summary, memory_update, generation_error = self._synthesize_rollup_summary(
                channel_id=channel_id,
                level=level,
                source_level=source_level,
                node=node,
                children=children,
                fallback_summary=fallback,
                workload_class=workload_class,
            )
            if summary and summary != fallback and generation_error is None:
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
                    alert_events=node.get("alert_events"),
                    alert_parser_breakdown=node.get("alert_parser_breakdown"),
                    state_transition_total=node.get("state_transition_total"),
                    summary_kind="llm",
                    operator_summary=summary,
                    memory_update=memory_update,
                    generation_status="ready",
                    format_version=ROLLUP_OPERATOR_FORMAT_VERSION,
                )
                node["summary"] = summary
                node["operator_summary"] = summary
                node["memory_update"] = memory_update
                node["summary_kind"] = "llm"
                node["generation_status"] = "ready"
                node["format_version"] = ROLLUP_OPERATOR_FORMAT_VERSION
                self._update_channel_routine_context(
                    channel_id=channel_id,
                    rollup_id=rollup_id,
                    summary_text=summary,
                    window_end=self._coerce_float(node.get("window_end")),
                    level=level,
                    memory_update=memory_update,
                )
            else:
                node["summary_kind"] = "degraded"
                node["generation_status"] = "failed" if generation_error else "degraded"
                if generation_error:
                    node["generation_error"] = generation_error
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
            source_frame_count = _parse_optional_int(log.get("source_frame_count")) or frame_count
            selected_frame_count = _parse_optional_int(log.get("selected_frame_count")) or frame_count
            frame_selection = self._compact_frame_selection(log.get("frame_selection"))
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
                "source_frame_count": int(source_frame_count),
                "selected_frame_count": int(selected_frame_count),
                "run_ids": [run_id] if run_id else [],
                "highlights": [headline] if headline else [],
                "summary": summary,
                "created_at": created,
                "alert_counts": alert_counts,
                "alert_total": alert_total,
                "alert_severities": self._coerce_str_list(log.get("alert_severities")),
                "signal_digest": signal_digest,
            }
            if frame_selection:
                node["frame_selection"] = frame_selection
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
                episode_count = len(vector_signal.get("road_episodes") or []) if isinstance(vector_signal.get("road_episodes"), list) else 0
                node["vector_signal_total"] = int(clip_count + road_count + episode_count)
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
        max_new: Optional[int] = None,
        workload_class: str = "rollup",
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
            alert_events = self._collect_rollup_alert_events(children)
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
                    "operator_summary": summary,
                    "summary_kind": "queued",
                    "generation_status": "queued",
                    "format_version": ROLLUP_OPERATOR_FORMAT_VERSION,
                    "created_at": end_ts,
                    "signal_digest": signal_digest,
                    **alert_meta,
                    **provenance_meta,
                }
            )
            if alert_events:
                out[-1]["alert_events"] = alert_events
            if synthesize and level in self.rollup_llm_levels:
                llm_pairs.append((out[-1], children))
        if synthesize and level in self.rollup_llm_levels and llm_pairs:
            self._apply_rollup_llm_summaries(
                channel_id=channel_id,
                level=level,
                source_level=source_level,
                node_children_pairs=llm_pairs,
                max_new=max_new,
                workload_class=workload_class,
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
            incoming = self._compact_summary_history_entry(raw_log)
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
            frame_hash = str(frame.get("frame_hash") or "").strip()[:40]
            if frame_hash:
                item["frame_hash"] = frame_hash
            for key in ("width", "height"):
                value = _parse_optional_int(frame.get(key))
                if value is not None and value > 0:
                    item[key] = int(value)
            for source_key, output_key in (
                ("source_frame_index", "source_frame_index"),
                ("source_timestamp_ms", "source_timestamp_ms"),
                ("selection_bucket_start_ms", "selection_bucket_start_ms"),
            ):
                value = _parse_optional_int(frame.get(source_key))
                if value is not None:
                    item[output_key] = int(value)
            selection_source = str(frame.get("selection_source") or "").strip().lower()[:80]
            if selection_source:
                item["selection_source"] = selection_source
            selection_score = cls._finite_float(frame.get("selection_score"))
            if selection_score is not None:
                item["selection_score"] = round(float(selection_score), 6)
            if "selection_apex_available" in frame:
                item["apex_available"] = bool(frame.get("selection_apex_available"))
            fallback_reason = str(frame.get("selection_fallback_reason") or "").strip().lower()[:160]
            if fallback_reason:
                item["fallback_reason"] = fallback_reason
            out.append(item)

        # Burst evidence rides along regardless of even sampling: the motion
        # peak (if it was skipped) and its sharper companion frame.
        sampled_indices = {index for _role, index in anchors}
        extras = 0
        for index, frame in enumerate(frames):
            if extras >= 4:
                break
            companion = frame.get("burst_companion")
            if not isinstance(companion, Mapping):
                continue
            frame_ts = cls._frame_timestamp_ms(frame, batch_end_ms)
            if index not in sampled_indices:
                apex_thumbnail = str(frame.get("thumbnail") or "").strip()
                if apex_thumbnail:
                    apex_item: Dict[str, Any] = {
                        "anchor_role": "burst_apex",
                        "frame_index": int(index),
                        "timestamp_ms": frame_ts,
                        "thumbnail": apex_thumbnail,
                    }
                    apex_hash = str(frame.get("frame_hash") or "").strip()[:40]
                    if apex_hash:
                        apex_item["frame_hash"] = apex_hash
                    apex_source_index = _parse_optional_int(frame.get("source_frame_index"))
                    if apex_source_index is not None:
                        apex_item["source_frame_index"] = int(apex_source_index)
                    out.append(apex_item)
                    sampled_indices.add(index)
                    extras += 1
            companion_thumbnail = str(companion.get("thumbnail") or "").strip()
            if companion_thumbnail and extras < 4:
                companion_item: Dict[str, Any] = {
                    "anchor_role": "burst_companion",
                    "frame_index": int(index),
                    "timestamp_ms": int(
                        _parse_optional_int(companion.get("timestamp_ms")) or frame_ts
                    ),
                    "thumbnail": companion_thumbnail,
                    "companion_of_timestamp_ms": frame_ts,
                }
                companion_hash = str(companion.get("frame_hash") or "").strip()[:40]
                if companion_hash:
                    companion_item["frame_hash"] = companion_hash
                companion_source_index = _parse_optional_int(companion.get("source_frame_index"))
                if companion_source_index is not None:
                    companion_item["source_frame_index"] = int(companion_source_index)
                for score_key in ("sharpness", "activity"):
                    score = cls._finite_float(companion.get(score_key))
                    if score is not None:
                        companion_item[score_key] = round(float(score), 6)
                out.append(companion_item)
                extras += 1
        return out

    def _archive_summary_entry(self, entry: Mapping[str, Any]) -> Dict[str, Any]:
        callback = self.summary_archive_callback
        if callback is None:
            return {}
        try:
            result = callback(entry)
        except Exception as exc:
            return {"error": _safe_error_text(exc, 240) or exc.__class__.__name__}
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
        # Rough context estimate for small-context VLMs (Qwen3-VL class):
        # ~4 chars per text token, ~300 visual tokens per <=640px image.
        estimated_tokens = int(round(text_chars / 4.0)) + int(image_parts) * 300
        return {
            "message_count": len(messages),
            "text_chars": int(text_chars),
            "image_parts": int(image_parts),
            "high_detail_images": int(high_detail_images),
            "image_url_chars": int(image_url_chars),
            "total_payload_chars": int(total_payload_chars),
            "estimated_context_tokens": int(estimated_tokens),
        }

    def _summary_context_tokens_warn(self) -> int:
        try:
            threshold = int(getattr(self.config, "LM_VIDEO_CONTEXT_TOKENS_WARN", 7000))
        except (TypeError, ValueError):
            threshold = 7000
        return max(1000, threshold)

    def _summary_input_warnings(self, stats: Mapping[str, Any]) -> List[str]:
        warnings: List[str] = []
        text_chars = _parse_optional_int(stats.get("text_chars")) or 0
        image_url_chars = _parse_optional_int(stats.get("image_url_chars")) or 0
        estimated_tokens = _parse_optional_int(stats.get("estimated_context_tokens")) or 0
        if text_chars >= self.lm_input_warning_chars:
            warnings.append(
                f"text_input_chars {text_chars} >= warning {self.lm_input_warning_chars}"
            )
        if image_url_chars >= self.lm_image_payload_warning_chars:
            warnings.append(
                f"image_payload_chars {image_url_chars} >= warning {self.lm_image_payload_warning_chars}"
            )
        tokens_warn = self._summary_context_tokens_warn()
        if estimated_tokens >= tokens_warn:
            warnings.append(
                f"estimated_context_tokens {estimated_tokens} >= warning {tokens_warn}; "
                "the VLM context may truncate this batch"
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
        session_generation: Optional[str] = None,
    ) -> Dict[str, Any]:
        source_frame_items = [dict(frame) for frame in frames if isinstance(frame, Mapping)]
        if not source_frame_items:
            raise ValueError("summary batch requires at least one frame")
        frame_ts_ms: List[int] = []
        for frame in source_frame_items:
            timestamp_ms = self._batch_frame_timestamp_ms(frame)
            if timestamp_ms is not None:
                frame_ts_ms.append(int(timestamp_ms))
        submitted_at = time.time()
        submitted_at_ms = int(submitted_at * 1000.0)
        batch_start_ms = min(frame_ts_ms) if frame_ts_ms else submitted_at_ms
        batch_end_ms = max(frame_ts_ms) if frame_ts_ms else submitted_at_ms
        raw_vector_signal = self._build_vector_signal_bundle(
            int(channel_id),
            cast(Sequence[Mapping[str, Any]], source_frame_items),
            batch_start_ms=batch_start_ms,
            batch_end_ms=batch_end_ms,
        )
        vector_signal = self._compact_vector_signal(raw_vector_signal)
        frame_items, frame_selection = self._select_attention_frames(
            cast(Sequence[Mapping[str, Any]], source_frame_items),
            raw_vector_signal,
        )
        if self.vector_signals_enabled:
            capture_attention = self._capture_attention_signal(frame_items)
            if capture_attention:
                enriched_signal = dict(vector_signal) if vector_signal else {
                    "version": 1,
                    "channel_id": int(channel_id),
                    "semantics": "vector_homeostasis_attention_signal_not_visual_proof",
                }
                enriched_signal["capture_attention"] = capture_attention
                vector_signal = self._compact_vector_signal(enriched_signal)
        provenance_source_frame_count = int(
            _parse_optional_int(frame_selection.get("source_frame_count"))
            or len(source_frame_items)
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
            "source_frame_count": provenance_source_frame_count,
            "selected_frame_count": len(frame_items),
            "batch_size": int(batch_size),
            "system_prompt_chars": len(system_prompt),
            "task_prompt_chars": len(str(prompt or "")),
            "vector_signal_chars": len(json.dumps(vector_signal, ensure_ascii=False, sort_keys=True)) if vector_signal else 0,
            "total_image_base64_chars": int(sum(frame_b64_lengths)),
            "largest_frame_base64_chars": int(max(frame_b64_lengths) if frame_b64_lengths else 0),
            "warning_text_chars": self.lm_input_warning_chars,
            "warning_image_payload_chars": self.lm_image_payload_warning_chars,
        }
        batch_payload: Dict[str, Any] = {
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
            "source_frame_count": provenance_source_frame_count,
            "selected_frame_count": len(frame_items),
            "frame_selection": frame_selection,
            "batch_start_ms": batch_start_ms,
            "batch_end_ms": batch_end_ms,
            "submitted_at": submitted_at,
            "llm_input_stats": llm_input_stats,
        }
        normalized_generation = str(session_generation or "").strip()
        if normalized_generation:
            batch_payload["session_generation"] = normalized_generation
        return batch_payload

    def run_summary_batch(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        channel_id = _parse_optional_int(batch.get("channel_id"))
        frames = batch.get("frames")
        if channel_id is None or channel_id < 1:
            raise ValueError("summary batch channel_id must be positive")
        self._assert_summary_batch_current(batch)
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
        if bool(getattr(self.lm_callback, "eva_generation_preflight", False)):
            summary = self.lm_callback(
                messages,
                model_hint,
                preflight=lambda: self._assert_summary_batch_current(batch),
            )
        else:
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
        source_frame_count = max(
            len(frame_items),
            int(_parse_optional_int(batch.get("source_frame_count")) or 0),
        )
        frame_selection = self._compact_frame_selection(batch.get("frame_selection"))
        entry: Dict[str, Any] = {
            "channel_id": int(channel_id),
            "run_id": str(batch.get("run_id") or "").strip(),
            "summary": summary,
            "frame_count": len(frame_items),
            "source_frame_count": int(source_frame_count),
            "selected_frame_count": len(frame_items),
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
        if frame_selection:
            entry["frame_selection"] = frame_selection
        coalesced_info = batch.get("coalesced")
        if isinstance(coalesced_info, Mapping):
            entry["coalesced"] = {
                "batches": max(1, int(_parse_optional_int(coalesced_info.get("batches")) or 1)),
                "omitted_frames": max(0, int(_parse_optional_int(coalesced_info.get("omitted_frames")) or 0)),
            }
        session_generation = str(batch.get("session_generation") or "").strip()
        if session_generation:
            entry["session_generation"] = session_generation
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
        side_effect_lock = self._session_side_effect_lock_for(channel_id)
        with side_effect_lock:
            stale_reason = self._summary_entry_stale_reason(accepted)
            if stale_reason:
                accepted.pop("archive_frames", None)
                accepted.update(
                    {
                        "accepted": False,
                        "stale_session": True,
                        "side_effects_skipped": True,
                        "stale_reason": stale_reason,
                    }
                )
                return accepted

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
                    last_error=_safe_error_text(exc, 240) or exc.__class__.__name__,
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
                            if len(session.logs) > 50:
                                session.logs = session.logs[-50:]
                        session._mark_summary_success_locked()
            self.record_summary_log(channel_id, accepted)
            accepted["accepted"] = True
            return accepted

    def dispatch_summary_batch(
        self,
        batch: Mapping[str, Any],
        *,
        workload_class: str = "heartbeat",
    ) -> Dict[str, Any]:
        stale_reason = self._summary_entry_stale_reason(batch)
        if stale_reason:
            return {
                "queued": False,
                "accepted": False,
                "status": "superseded",
                "stale_reason": stale_reason,
            }
        dispatcher = self.summary_dispatcher
        if dispatcher is not None:
            return dict(dispatcher(batch, workload_class))
        try:
            entry = self.run_summary_batch(batch)
        except SummaryBatchSuperseded as exc:
            return {
                "queued": False,
                "accepted": False,
                "status": "superseded",
                "stale_reason": exc.reason,
            }
        accepted = self.accept_summary_entry(entry)
        if not bool(accepted.get("accepted", True)):
            return {
                "queued": False,
                "accepted": False,
                "status": "stale_session",
                "stale_reason": accepted.get("stale_reason"),
            }
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

    def should_dispatch_probe_frame(self, channel_id: int, *, capture_kind: str = "video") -> bool:
        """Return whether a selected apex needs CLIP buffering for this channel."""

        if self.probe_manager is None:
            return False
        if str(capture_kind or "").strip().lower() == "analytics":
            return True
        with self.cache_lock:
            return int(channel_id) in self.shared_probe_channels

    def get_channels(self, force: bool = False) -> List[Dict[str, Any]]:
        now = time.time()
        with self.cache_lock:
            if not force and self.channels_cache and now - self.channels_cache[0] < 30:
                return [dict(channel) for channel in self.channels_cache[1]]
            self.channels_cache_last_attempt_at = now
        client = self.build_client()
        try:
            channels = client.get_channels()
        except Exception as exc:
            safe_error = _safe_error_text(exc, 500) or exc.__class__.__name__
            with self.cache_lock:
                self.channels_cache_stale = self.channels_cache is not None
                self.channels_cache_last_error = safe_error
                stream_meta = dict(getattr(client, "channel_inventory_meta", None) or {})
                if stream_meta.get("error"):
                    stream_meta["error"] = _safe_error_text(stream_meta.get("error"), 500)
                self.channels_cache_stream_meta = stream_meta
                cached = (
                    [dict(channel) for channel in self.channels_cache[1]]
                    if self.channels_cache is not None
                    else None
                )
            if cached is not None:
                LOGGER.warning(
                    "Luxriot channel inventory refresh failed; retaining stale cache: %s",
                    safe_error,
                )
                return cached
            raise
        refreshed_at = time.time()
        with self.cache_lock:
            self.channels_cache = (refreshed_at, [dict(channel) for channel in channels])
            self.channels_cache_stale = False
            self.channels_cache_last_error = None
            self.channels_cache_last_success_at = refreshed_at
            stream_meta = dict(getattr(client, "channel_inventory_meta", None) or {})
            if stream_meta.get("error"):
                stream_meta["error"] = _safe_error_text(stream_meta.get("error"), 500)
            self.channels_cache_stream_meta = stream_meta
        return [dict(channel) for channel in channels]

    def channel_inventory_status(self) -> Dict[str, Any]:
        """Expose cache freshness without changing the long-standing list return type."""

        now = time.time()
        with self.cache_lock:
            cached_at = self.channels_cache[0] if self.channels_cache is not None else None
            count = len(self.channels_cache[1]) if self.channels_cache is not None else 0
            return {
                "cached": self.channels_cache is not None,
                "count": count,
                "stale": bool(self.channels_cache_stale),
                "cache_age_sec": max(0.0, now - cached_at) if cached_at is not None else None,
                "last_attempt_at": self.channels_cache_last_attempt_at,
                "last_success_at": self.channels_cache_last_success_at,
                "last_error": self.channels_cache_last_error,
                "stream": dict(self.channels_cache_stream_meta),
            }

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
    def _bookmark_content_key(cls, alert: Mapping[str, Any]) -> str:
        title = " ".join(str(alert.get("title") or "").casefold().split())
        severity = cls._normalize_alert_severity(alert.get("severity"))
        return f"{title}|{severity}"

    def _bookmark_content_recently_sent_locked(self, channel_id: int, content_key: str, now_ms: int) -> bool:
        window_sec = float(getattr(self, "alert_dedupe_window_sec", 600.0) or 0.0)
        if window_sec <= 0:
            return False
        channel_cache = self.channel_bookmark_content_keys.get(int(channel_id)) or {}
        last_ts = channel_cache.get(content_key)
        return isinstance(last_ts, int) and (now_ms - last_ts) < int(window_sec * 1000)

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

    def _mark_bookmark_content_sent_locked(self, channel_id: int, content_key: str, ts_ms: int) -> None:
        window_sec = float(getattr(self, "alert_dedupe_window_sec", 600.0) or 0.0)
        if window_sec <= 0:
            return
        channel_key = int(channel_id)
        channel_cache = self.channel_bookmark_content_keys.setdefault(channel_key, {})
        channel_cache[content_key] = int(ts_ms)
        prune_before = int(ts_ms) - int(window_sec * 1000)
        stale_keys = [key for key, value in channel_cache.items() if isinstance(value, int) and value < prune_before]
        for key in stale_keys:
            channel_cache.pop(key, None)
        if len(channel_cache) > 2000:
            newest = sorted(channel_cache.items(), key=lambda item: item[1], reverse=True)[:1200]
            self.channel_bookmark_content_keys[channel_key] = dict(newest)

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
                    parser_error=_safe_error_text(exc, 240) or exc.__class__.__name__,
                )
        except Exception as exc:
            return AlertDeliveryResult(
                alerts_detected=True,
                json_alert_count=int(diagnostics.get("json_alert_count") or 0),
                prose_alert_count=int(diagnostics.get("prose_alert_count") or 0),
                parser_error=_safe_error_text(exc, 240) or exc.__class__.__name__,
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
            content_key = self._bookmark_content_key(alert)
            now_ms = int(time.time() * 1000)
            alert_cooldown_sec = self._bookmark_cooldown_for_severity(cooldown_sec, alert["severity"])
            with self.cache_lock:
                if self._bookmark_content_recently_sent_locked(int(channel_id), content_key, now_ms):
                    alert_events.append({**alert, "delivery_status": "deduplicated"})
                    continue
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
                last_error = _safe_error_text(exc, 240) or exc.__class__.__name__
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
                self._mark_bookmark_content_sent_locked(int(channel_id), content_key, now_ms)
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
        side_effect_lock = self._session_side_effect_lock_for(channel_id)
        with side_effect_lock:
            return self._start_session_for_generation(
                channel_id=channel_id,
                batch_size=batch_size,
                prompt=prompt,
                model_hint=model_hint,
                system_prompt=system_prompt,
                interval_sec=interval_sec,
                update_desired=update_desired,
                session_generation="",
            )

    def _start_session_for_generation(
        self,
        channel_id: int,
        batch_size: Optional[int] = None,
        prompt: str = "",
        model_hint: Optional[str] = None,
        system_prompt: Optional[str] = None,
        interval_sec: Optional[float] = None,
        update_desired: bool = True,
        session_generation: str = "",
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
        previous_override_present = False
        previous_channel_overrides: Dict[str, Any] = {}
        with self.cache_lock:
            previous_override_present = channel_id in self.channel_prompt_overrides
            previous_channel_overrides = copy.deepcopy(
                self.channel_prompt_overrides.get(channel_id) or {}
            )
            if system_prompt is not None:
                overrides_raw = self.channel_prompt_overrides.get(channel_id)
                channel_overrides = dict(overrides_raw) if isinstance(overrides_raw, Mapping) else {}
                next_stream_prompt = str(system_prompt)
                if (
                    "stream_system_prompt" not in channel_overrides
                    or next_stream_prompt != str(channel_overrides.get("stream_system_prompt") or "")
                ):
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
                if not self._persist_summary_state_locked():
                    if previous_override_present:
                        self.channel_prompt_overrides[channel_id] = previous_channel_overrides
                    else:
                        self.channel_prompt_overrides.pop(channel_id, None)
                    persistence_error = (
                        self.summary_state_last_error
                        or "runtime state backend rejected the channel settings"
                    )
                    raise RuntimeError(
                        "Summary session was not restarted because its channel settings "
                        f"could not be persisted: {persistence_error}"
                    )
            effective_interval_sec = self._get_capture_interval_sec_locked(channel_id)
            effective_system_prompt = self._get_stream_system_prompt_locked(channel_id)

        # Desired state must be durable before the current session is disrupted.
        # A failed write leaves the old session alive and on its current generation.
        if update_desired:
            self._set_desired_live_session(
                channel_id,
                enabled=True,
                batch_size=batch,
                prompt=prompt,
                model_hint=normalized_model_hint,
                interval_sec=effective_interval_sec,
            )

        # Persist the replacement run before superseding/stopping the current
        # generation.  A backend outage therefore leaves the current channel
        # running instead of turning a save failure into a video outage.
        with self.cache_lock:
            run = self._open_run_locked(
                channel_id=channel_id,
                batch_size=batch,
                prompt=prompt,
                model_hint=normalized_model_hint,
                system_prompt=effective_system_prompt,
                interval_sec=effective_interval_sec,
            )

        if not session_generation:
            session_generation = self._advance_session_generation(channel_id)

        with self.cache_lock:
            existing = self.sessions.pop(channel_id, None)
            existing_probe = self.probe_sessions.pop(channel_id, None)

        if existing is not None:
            existing.stop()
            existing_status = existing.status()
            existing_logs = existing_status.get("logs")
            with self.cache_lock:
                if isinstance(existing_logs, list):
                    self._merge_summary_history_locked(channel_id, existing_logs)
                self._close_run_locked(channel_id, existing_status.get("run_id"))

        if existing_probe is not None:
            existing_probe.stop()
            with self.cache_lock:
                if channel_id not in self.paused_probe_channels:
                    self.shared_probe_channels.add(channel_id)

        try:
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
                session_generation=session_generation,
            )
        except Exception:
            with self.cache_lock:
                self._close_run_locked(channel_id, run.get("run_id"))
            raise
        with self.cache_lock:
            self.sessions[channel_id] = session
        try:
            session.start()
        except Exception:
            with self.cache_lock:
                if self.sessions.get(channel_id) is session:
                    self.sessions.pop(channel_id, None)
                self._close_run_locked(channel_id, run.get("run_id"))
            raise
        return session.status()

    def stop_session(self, channel_id: int, *, update_desired: bool = True) -> Dict[str, Any]:
        side_effect_lock = self._session_side_effect_lock_for(channel_id)
        with side_effect_lock:
            self._advance_session_generation(channel_id)
            return self._stop_session_for_generation(channel_id, update_desired=update_desired)

    def _stop_session_for_generation(
        self,
        channel_id: int,
        *,
        update_desired: bool = True,
    ) -> Dict[str, Any]:
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
        compact_feed: bool = False,
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
            if compact_feed:
                status["logs"] = [self._compact_summary_feed_entry(log) for log in filtered_logs]
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
        result = {
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
        if compact_feed:
            result["logs"] = [self._compact_summary_feed_entry(log) for log in filtered_logs]
        return result

    @classmethod
    def _compact_summary_feed_entry(cls, value: Mapping[str, Any]) -> Dict[str, Any]:
        fields = (
            "channel_id",
            "run_id",
            "summary",
            "frame_count",
            "source_frame_count",
            "selected_frame_count",
            "batch_size",
            "created_at",
            "batch_start_ms",
            "batch_end_ms",
            "duration_sec",
            "model",
            "alert_counts",
            "alert_total",
            "alert_severities",
            "bookmarks_sent",
            "bookmark_failed_count",
            "bookmark_last_error",
            "coverage_gap",
            "gap_reason",
            "coalesced",
            "archive_inserted",
        )
        out = {key: value[key] for key in fields if key in value}
        vector_signal = cls._compact_vector_signal(value.get("vector_signal"))
        attention = vector_signal.get("capture_attention") if isinstance(vector_signal, Mapping) else None
        if isinstance(attention, Mapping):
            out["vector_signal"] = {"capture_attention": dict(attention)}
        return out

    def summary_rollups(
        self,
        channel_id: int,
        run_selector: Optional[str] = None,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
        level_limit: Optional[int] = 60,
        synthesize: bool = True,
        target_level: Optional[str] = None,
        synthesize_levels: Optional[Set[str]] = None,
        max_new_per_level: Optional[int] = None,
    ) -> Dict[str, Any]:
        aggregation_started = time.monotonic()
        raw_target_level = str(target_level or "").strip().upper()
        if raw_target_level and raw_target_level not in {"L0", "L1", "L2", "L3"}:
            raise ValueError("target_level must be one of L0, L1, L2, or L3")
        requested_target = raw_target_level or None
        requested_synthesis_levels = {
            self._normalize_rollup_level(level)
            for level in (synthesize_levels or set())
            if self._normalize_rollup_level(level) in {"L1", "L2", "L3"}
        }

        def should_synthesize(level: str) -> bool:
            if not synthesize:
                return False
            return not requested_synthesis_levels or level in requested_synthesis_levels

        target_rank = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}.get(
            requested_target or "L3",
            3,
        )
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
        selected_run_id = str(status.get("run_filter_id") or "").strip() or None
        stored_rollups = self._list_cached_rollups(
            channel_id=channel_id,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        if selected_run_id:
            stored_rollups = [
                row
                for row in stored_rollups
                if self._rollup_matches_run_selector(row, selected_run_id)
            ]
        stored_by_level: Dict[str, List[Dict[str, Any]]] = {
            "L1": [],
            "L2": [],
            "L3": [],
        }
        for row in stored_rollups:
            stored_level = self._normalize_rollup_level(row.get("level"))
            if stored_level in stored_by_level:
                stored_by_level[stored_level].append(dict(row))

        l1_nodes: List[Dict[str, Any]] = []
        l2_nodes: List[Dict[str, Any]] = []
        l3_nodes: List[Dict[str, Any]] = []
        if target_rank >= 1:
            l1_nodes = self._build_rollup_level(
                channel_id=channel_id,
                level="L1",
                source_level="L0",
                window_sec=self.rollup_windows["L1"],
                source_nodes=l0_nodes,
                synthesize=should_synthesize("L1"),
                max_new=max_new_per_level,
            )
            l1_nodes = self._merge_rollup_rows(l1_nodes, stored_by_level["L1"])
        if target_rank >= 2:
            l2_nodes = self._build_rollup_level(
                channel_id=channel_id,
                level="L2",
                source_level="L1",
                window_sec=self.rollup_windows["L2"],
                source_nodes=l1_nodes,
                synthesize=should_synthesize("L2"),
                max_new=max_new_per_level,
            )
            l2_nodes = self._merge_rollup_rows(l2_nodes, stored_by_level["L2"])
        if target_rank >= 3:
            l3_nodes = self._build_rollup_level(
                channel_id=channel_id,
                level="L3",
                source_level="L2",
                window_sec=self.rollup_windows["L3"],
                source_nodes=l2_nodes,
                synthesize=should_synthesize("L3"),
                max_new=max_new_per_level,
            )
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
        all_levels = ["L0", "L1", "L2", "L3"]
        computed_levels = all_levels[: target_rank + 1]
        not_requested_levels = all_levels[target_rank + 1 :]
        aggregation_elapsed_sec = max(0.0, time.monotonic() - aggregation_started)

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
            "target_level": requested_target,
            "computed_levels": computed_levels,
            "not_requested_levels": not_requested_levels,
            "aggregation": {
                "status": "ready",
                "requested_level": requested_target or "all",
                "computed_levels": computed_levels,
                "not_requested_levels": not_requested_levels,
                "elapsed_sec": round(aggregation_elapsed_sec, 3),
                "synthesize": bool(synthesize),
            },
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
            compact["last_source_frame_count"] = latest_log.get("source_frame_count")
            compact["last_selected_frame_count"] = latest_log.get("selected_frame_count")
            for archive_field in (
                "attempted",
                "inserted",
                "summary_frames",
                "alert_frames",
                "error",
            ):
                source_key = f"archive_{archive_field}"
                if source_key in latest_log:
                    compact[f"last_{source_key}"] = latest_log.get(source_key)
            frame_selection = LuxriotManager._compact_frame_selection(latest_log.get("frame_selection"))
            if frame_selection:
                compact["last_frame_selection"] = frame_selection
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
            "last_live_segment_target_seconds",
            "last_live_segment_summary_target_seconds",
            "last_live_segment_raw_frame_budget",
            "last_live_segment_byte_budget",
            "last_live_segment_streamed_bytes",
            "last_live_segment_represented_seconds",
            "last_live_segment_completed_at",
            "last_live_segment_source_start_timestamp_ms",
            "last_live_segment_last_source_timestamp_ms",
            "last_live_segment_timestamp_source",
            "live_segment_inflight",
            "live_segment_capture_started_at",
            "live_segment_inflight_target_seconds",
            "live_segment_inflight_raw_frame_budget",
            "live_segment_inflight_frames",
            "live_segment_inflight_represented_seconds",
            "frozen_signal",
            "frozen_signal_since",
            "frozen_signal_age_sec",
            "frozen_frame_count",
            "frozen_frame_hash",
            "frozen_frame_dropped_count",
            "capture_apex_pending_frames",
            "capture_apex_raw_frame_count",
            "capture_apex_selected_count",
            "capture_apex_fallback_count",
            "capture_apex_probe_dispatch_count",
            "capture_apex_probe_failure_count",
            "capture_apex_probe_skipped_count",
            "capture_apex_selection_sources",
            "capture_apex_last_selection",
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
            rollup_scheduler_status = dict(self._rollup_scheduler_status)
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
            "rollup_scheduler": rollup_scheduler_status,
            "rollup_backfill": self.rollup_backfill_status(),
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
