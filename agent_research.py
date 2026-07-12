"""Trusted, bounded continuation ledger for multi-turn video research."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Iterable, Optional, Set


RESEARCH_STATE_VERSION = 1
MAX_RESEARCH_CHANNELS = 1_000
RESEARCH_STATE_MAX_AGE_SEC = 24 * 60 * 60


def _channel_ids(values: Any) -> Set[int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return set()
    result: Set[int] = set()
    for value in values:
        if isinstance(value, Mapping):
            value = value.get("channel_id") if value.get("channel_id") is not None else value.get("id")
        try:
            channel_id = int(value)
        except (TypeError, ValueError):
            continue
        if channel_id > 0:
            result.add(channel_id)
        if len(result) >= MAX_RESEARCH_CHANNELS:
            break
    return result


def operator_requests_continuation(text: Any) -> bool:
    """Detect an explicit continuation command, not a merely similar word.

    Bare stems are dangerous: ``продолж`` matches «продолжительность»,
    ``остальн`` matches «опиши остальную сцену», bare ``remaining``/«дальше»
    match ordinary temporal questions. A false positive injects the frozen
    research window into an unrelated question, so every branch here requires
    either an imperative verb form or an explicit channels/scope context.
    """

    value = str(text or "").strip().casefold()
    return bool(
        re.search(
            r"\b(?:continue|resume)\b"
            r"|\bnext\s+(?:chunk|batch|channels?)\b"
            r"|\bremaining\s+(?:channels?|cameras?|ids?|scope|list)\b"
            r"|\bпродолж(?:и|ай|айте|им|ить|аем)\b"
            r"|\bдавай\s+дальше\b"
            r"|\bдальше\s+по\s+(?:списку|каналам|остальн\w*)\b"
            r"|следующ(?:ий|ая|ее|ие)\s+(?:чанк|пакет|канал)"
            r"|остальн\w*\s+(?:канал\w*|камер\w*)"
            r"|оставш\w*\s+(?:канал\w*|камер\w*)",
            value,
            flags=re.IGNORECASE,
        )
    )


def usable_research_state(state: Any, *, now: Optional[float] = None) -> bool:
    if not isinstance(state, Mapping):
        return False
    if int(state.get("version") or 0) != RESEARCH_STATE_VERSION:
        return False
    if str(state.get("kind") or "") != "video_summary_inventory":
        return False
    if not _channel_ids(state.get("remaining_channel_ids")):
        return False
    updated_at = state.get("updated_at")
    try:
        age = float(now if now is not None else time.time()) - float(updated_at)
    except (TypeError, ValueError):
        return False
    return -60.0 <= age <= RESEARCH_STATE_MAX_AGE_SEC


def _window(result: Mapping[str, Any]) -> Dict[str, float]:
    raw = result.get("time_window")
    source = raw if isinstance(raw, Mapping) else result
    window: Dict[str, float] = {}
    for key in ("from_ts", "to_ts"):
        try:
            value = float(source.get(key))
        except (TypeError, ValueError):
            continue
        window[key] = value
    if "from_ts" in window and "to_ts" in window and window["to_ts"] < window["from_ts"]:
        window["from_ts"], window["to_ts"] = window["to_ts"], window["from_ts"]
    return window


def research_state_from_inventory(
    result: Mapping[str, Any],
    *,
    previous: Optional[Mapping[str, Any]] = None,
    continuation: bool = False,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Build the next exact continuation state from one inventory tool result."""

    candidate = _channel_ids(result.get("candidate_channels"))
    inactive = _channel_ids(result.get("inactive_channel_ids"))
    checked = _channel_ids(result.get("checked_channel_ids"))
    deferred = _channel_ids(result.get("deferred_channel_ids"))
    unchecked = _channel_ids(result.get("unchecked_channel_ids"))
    errors = _channel_ids(result.get("errors"))
    explicit_requested = _channel_ids(result.get("requested_channel_ids"))
    accounted = candidate | inactive | checked | deferred | unchecked | errors
    current_scope = explicit_requested or accounted

    prior_requested = _channel_ids(previous.get("requested_channel_ids")) if isinstance(previous, Mapping) else set()
    prior_completed = _channel_ids(previous.get("completed_channel_ids")) if isinstance(previous, Mapping) else set()
    prior_remaining = _channel_ids(previous.get("remaining_channel_ids")) if isinstance(previous, Mapping) else set()
    if continuation and isinstance(previous, Mapping):
        requested = prior_requested or (prior_completed | prior_remaining | current_scope)
        completed = prior_completed | candidate | inactive
    else:
        requested = current_scope
        completed = candidate | inactive

    # ``checked_channel_ids`` means that the inventory/status lookup ran for a
    # channel.  It does not mean that the channel was returned inside the
    # bounded attention window: active rows beyond the per-turn limit are both
    # checked and deferred.  Deriving the remainder from the frozen requested
    # set preserves that distinction and cannot silently lose an errored,
    # unchecked, deferred, or otherwise unaccounted channel.
    remaining = requested.difference(completed)

    current_window = _window(result)
    prior_window = previous.get("frozen_window") if isinstance(previous, Mapping) else None
    window_mismatch = False
    if continuation and isinstance(prior_window, Mapping):
        frozen_window = {
            key: float(prior_window[key])
            for key in ("from_ts", "to_ts")
            if prior_window.get(key) is not None
        }
        if current_window and any(
            abs(float(current_window.get(key, frozen_window.get(key, 0.0))) - float(frozen_window.get(key, 0.0))) > 0.001
            for key in ("from_ts", "to_ts")
            if key in frozen_window
        ):
            window_mismatch = True
    else:
        frozen_window = current_window

    requested = set(sorted(requested)[:MAX_RESEARCH_CHANNELS])
    completed &= requested
    remaining &= requested
    state = {
        "version": RESEARCH_STATE_VERSION,
        "kind": "video_summary_inventory",
        "status": "pending" if remaining else "complete",
        "frozen_window": frozen_window,
        "requested_channel_ids": sorted(requested),
        "completed_channel_ids": sorted(completed),
        "remaining_channel_ids": sorted(remaining),
        "last_chunk_channel_ids": sorted(candidate | inactive | checked),
        "error_channel_ids": sorted(errors),
        "window_mismatch": window_mismatch,
        "updated_at": float(now if now is not None else time.time()),
    }
    return state


def continuation_tool_defaults(
    state: Mapping[str, Any],
    *,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    if not usable_research_state(state, now=now):
        return {}
    defaults: Dict[str, Any] = {
        "channel_ids": sorted(_channel_ids(state.get("remaining_channel_ids"))),
    }
    window = state.get("frozen_window")
    if isinstance(window, Mapping):
        for key in ("from_ts", "to_ts"):
            try:
                defaults[key] = float(window[key])
            except (KeyError, TypeError, ValueError):
                pass
    return defaults


def trusted_research_message(
    state: Mapping[str, Any],
    *,
    now: Optional[float] = None,
) -> str:
    defaults = continuation_tool_defaults(state, now=now)
    payload = {
        "kind": state.get("kind"),
        "status": state.get("status"),
        "requested_channel_ids": sorted(_channel_ids(state.get("requested_channel_ids"))),
        "completed_channel_ids": sorted(_channel_ids(state.get("completed_channel_ids"))),
        "remaining_channel_ids": defaults.get("channel_ids") or [],
        "frozen_window": {
            key: defaults[key]
            for key in ("from_ts", "to_ts")
            if key in defaults
        },
    }
    return (
        "Trusted server research continuation ledger. Treat this as authoritative; "
        "do not reconstruct IDs or time bounds from prose. Continue only remaining_channel_ids "
        "inside frozen_window and never repeat completed_channel_ids: "
        + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
