"""Trusted UI projections derived from completed EVA agent domain tools.

The model never emits these commands.  The harness maps validated tool
arguments/results onto a small closed vocabulary that the React console may
render.  Mutating tools produce navigation-only previews until a trusted Apply
receipt is present.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


UI_EFFECT_VERSION = 1
_ARCHIVE_TOOLS = frozenset({"get_detections", "search_archive"})
_VIDEO_PERIOD_TOOLS = frozenset(
    {
        "get_video_summaries",
        "count_video_summary_events",
        "list_attention_bursts",
        "track_visual_state_transitions",
        "get_visual_window_signals",
    }
)
_PROBE_WRITE_TOOLS = frozenset({"create_probe", "update_probe", "delete_probes"})


def derive_agent_ui_effects(
    tool_name: str,
    arguments: Mapping[str, Any] | None,
    result: Any,
    *,
    committed: bool = False,
    seed: str = "",
) -> list[dict[str, Any]]:
    """Return bounded, closed-vocabulary console effects for one tool result."""

    name = str(tool_name or "").strip()
    args = dict(arguments or {})
    row = dict(result) if isinstance(result, Mapping) else {}
    if not name or _result_failed(row):
        return []

    payload = _common_payload(args, row)
    action = ""
    target = ""

    if name in _ARCHIVE_TOOLS:
        target = "archive"
        action = "show_results"
        payload.update(_archive_payload(args, row))
    elif name == "describe_frame":
        target = "archive"
        action = "open_review"
        payload.update(_pick(args, row, keys=("detection_id", "image_path", "timestamp_ms")))
    elif name == "list_probes":
        target = "probes"
        action = "show_board"
        payload["result_count"] = _sequence_count(row, ("probes", "items", "results"))
    elif name in _PROBE_WRITE_TOOLS:
        target = "probes"
        action = "refresh" if committed and _has_applied_receipt(row) else "show_preview"
        payload.update(_probe_payload(args, row))
    elif name == "get_prompt_settings":
        target = "video"
        action = "open_prompt_settings"
    elif name == "update_prompt_settings":
        target = "video"
        action = "open_prompt_settings" if committed and _has_applied_receipt(row) else "show_prompt_preview"
    elif name in _VIDEO_PERIOD_TOOLS:
        target = "video"
        action = "show_period"
        payload.update(_video_payload(args, row))
    elif name == "list_video_summary_channels":
        target = "video"
        action = "show_channels"
        payload["result_count"] = _sequence_count(row, ("channels", "items", "results"))
    elif name in {"get_video_summary_restore_status", "restore_video_summary_history"}:
        target = "video"
        action = "show_restore_status" if name.startswith("get_") or committed else "show_restore_preview"
    else:
        return []

    effect_seed = seed or _effect_seed(name, args, row, committed)
    effect_id = hashlib.sha256(
        f"{effect_seed}\0{target}\0{action}".encode("utf-8", errors="replace")
    ).hexdigest()[:24]
    return [
        {
            "version": UI_EFFECT_VERSION,
            "effect_id": effect_id,
            "target": target,
            "action": action,
            "source": {
                "tool": name,
                "committed": bool(committed and _has_applied_receipt(row)),
            },
            "payload": payload,
        }
    ]


def _result_failed(result: Mapping[str, Any]) -> bool:
    if result.get("error"):
        return True
    return str(result.get("status") or "").strip().lower() in {"error", "failed", "denied"}


def _has_applied_receipt(result: Mapping[str, Any]) -> bool:
    receipt = result.get("action_receipt")
    return (
        isinstance(receipt, Mapping)
        and str(receipt.get("status") or "").strip().lower() == "applied"
    )


def _common_payload(arguments: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    payload = _pick(
        arguments,
        result,
        keys=(
            "channel_id",
            "probe_id",
            "source",
            "since_ms",
            "until_ms",
            "from_ts",
            "to_ts",
        ),
    )
    channel_ids = arguments.get("channel_ids")
    if channel_ids is None:
        channel_ids = result.get("channel_ids")
    bounded_ids = _bounded_int_list(channel_ids, limit=32)
    if bounded_ids:
        payload["channel_ids"] = bounded_ids
    return payload


def _archive_payload(arguments: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    payload = _pick(
        arguments,
        result,
        keys=("query", "sort_by", "limit", "offset"),
    )
    payload["result_count"] = _sequence_count(
        result,
        ("results", "detections", "items", "rows", "frames"),
    )
    return payload


def _probe_payload(arguments: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    probe = result.get("probe")
    if isinstance(probe, Mapping):
        payload.update(_pick(probe, {}, keys=("id", "probe_id", "channel_id", "name")))
    payload.update(_pick(arguments, result, keys=("probe_id", "channel_id")))
    probe_ids = arguments.get("probe_ids")
    if probe_ids is None:
        probe_ids = result.get("probe_ids")
    bounded = _bounded_str_list(probe_ids, limit=32)
    if bounded:
        payload["probe_ids"] = bounded
    return payload


def _video_payload(arguments: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    explicit_scope_keys = (
        "relative_range",
        "since_hours",
        "from_ts",
        "to_ts",
        "since_ms",
        "until_ms",
    )
    explicit_scope = any(
        arguments.get(key) is not None and str(arguments.get(key)).strip() != ""
        for key in explicit_scope_keys
    )
    payload = _pick(arguments, result, keys=("relative_range", "since_hours"))
    # A passive read with the tool's implicit default window may navigate to
    # Video, but must not overwrite the operator's current Live/L0 selection.
    # When the operator supplied a period, project the server-resolved bounds
    # so depth and time change together instead of producing Live + L1.
    if explicit_scope:
        payload.update(_pick(arguments, result, keys=("depth",)))
        time_window = result.get("time_window")
        if not isinstance(time_window, Mapping):
            time_window = result.get("period")
        if isinstance(time_window, Mapping):
            since_ms = _integer_or_none(time_window.get("since_ms"))
            until_ms = _integer_or_none(time_window.get("until_ms"))
            if since_ms is None:
                from_ts = _number_or_none(time_window.get("from_ts"))
                since_ms = int(from_ts * 1000.0) if from_ts is not None else None
            if until_ms is None:
                to_ts = _number_or_none(time_window.get("to_ts"))
                until_ms = int(to_ts * 1000.0) if to_ts is not None else None
            if since_ms is not None and until_ms is not None and until_ms >= since_ms:
                payload["since_ms"] = since_ms
                payload["until_ms"] = until_ms
    payload["result_count"] = _sequence_count(
        result,
        ("summaries", "items", "results", "events", "bursts"),
    )
    return payload


def _number_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _integer_or_none(value: Any) -> int | None:
    number = _number_or_none(value)
    return int(number) if number is not None else None


def _pick(
    primary: Mapping[str, Any],
    fallback: Mapping[str, Any],
    *,
    keys: Sequence[str],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in keys:
        value = primary.get(key)
        if value is None:
            value = fallback.get(key)
        if value is None or isinstance(value, (Mapping, list, tuple, set)):
            continue
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
            value = value[:512]
        payload[key] = value
    return payload


def _sequence_count(result: Mapping[str, Any], keys: Sequence[str]) -> int:
    for key in keys:
        value = result.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return min(100_000, len(value))
    for key in ("count", "total", "result_count"):
        try:
            return max(0, min(100_000, int(result.get(key) or 0)))
        except (TypeError, ValueError):
            continue
    return 0


def _bounded_int_list(value: Any, *, limit: int) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    result: list[int] = []
    for item in value:
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        if number not in result:
            result.append(number)
        if len(result) >= limit:
            break
    return result


def _bounded_str_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    result: list[str] = []
    for item in value:
        text = str(item or "").strip()[:128]
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _effect_seed(
    name: str,
    arguments: Mapping[str, Any],
    result: Mapping[str, Any],
    committed: bool,
) -> str:
    receipt = result.get("action_receipt")
    if isinstance(receipt, Mapping) and receipt.get("plan_id"):
        return f"plan:{receipt['plan_id']}"
    approval = result.get("approval")
    if isinstance(approval, Mapping) and approval.get("plan_id"):
        return f"preview:{approval['plan_id']}"
    bounded = json.dumps(
        {"tool": name, "arguments": dict(arguments), "committed": committed},
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return bounded[:4096]
