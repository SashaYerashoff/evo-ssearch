"""Validation and prompt-safe projection of the operator's current console state.

Console context is a harness-owned default for tool arguments, never operator
prose and never model-authored state. Explicit values in the operator request
or model tool call continue to win.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from typing import Any


CONSOLE_CONTEXT_VERSION = 1
_SECTIONS = frozenset({"home", "archive", "probes", "video"})
_ARCHIVE_SOURCES = frozenset(
    {"semantic_snapshot", "probe", "vlm_summary", "vlm_alert"}
)
_ARCHIVE_SORTS = frozenset({"similarity", "time"})


def normalize_agent_console_context(
    value: Any,
    *,
    allowed_channel_ids: Collection[str] | None = None,
) -> dict[str, Any]:
    """Return a bounded closed-schema console context or an empty mapping."""

    if not isinstance(value, Mapping):
        return {}
    try:
        version = int(value.get("version") or CONSOLE_CONTEXT_VERSION)
    except (TypeError, ValueError):
        return {}
    if version != CONSOLE_CONTEXT_VERSION:
        return {}

    section = str(value.get("section") or "").strip().lower()
    if section not in _SECTIONS:
        return {}

    result: dict[str, Any] = {
        "version": CONSOLE_CONTEXT_VERSION,
        "section": section,
    }
    if section != "archive":
        return result

    raw_archive = value.get("archive")
    if not isinstance(raw_archive, Mapping):
        return result
    archive: dict[str, Any] = {}

    channel_id = _positive_int(raw_archive.get("channel_id"))
    allowed = {str(item) for item in (allowed_channel_ids or ())}
    if channel_id is not None and (
        not allowed or "*" in allowed or str(channel_id) in allowed
    ):
        archive["channel_id"] = channel_id

    source = str(raw_archive.get("source") or "").strip().lower()
    if source in _ARCHIVE_SOURCES:
        archive["source"] = source

    probe_id = str(raw_archive.get("probe_id") or "").strip()[:128]
    if archive.get("source") == "probe" and probe_id:
        archive["probe_id"] = probe_id

    since_ms = _nonnegative_int(raw_archive.get("since_ms"))
    until_ms = _nonnegative_int(raw_archive.get("until_ms"))
    if since_ms is not None and until_ms is not None and since_ms <= until_ms:
        archive["since_ms"] = since_ms
        archive["until_ms"] = until_ms

    sort_by = str(raw_archive.get("sort_by") or "").strip().lower()
    if sort_by in _ARCHIVE_SORTS:
        archive["sort_by"] = sort_by

    rows = _positive_int(raw_archive.get("rows"))
    if rows is not None:
        archive["rows"] = min(rows, 100)

    if archive:
        result["archive"] = archive
    return result


def apply_console_context_defaults(
    turn_context: dict[str, Any],
    console_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge trusted console defaults without overriding operator-derived state."""

    if not isinstance(console_context, Mapping):
        return turn_context
    if str(console_context.get("section") or "") != "archive":
        return turn_context
    archive = console_context.get("archive")
    if not isinstance(archive, Mapping):
        return turn_context

    if turn_context.get("channel_id") is None and archive.get("channel_id") is not None:
        turn_context["channel_id"] = int(archive["channel_id"])
        turn_context["console_default_channel"] = True
    if (
        not turn_context.get("operator_relative_range")
        and not isinstance(turn_context.get("time_window"), Mapping)
        and archive.get("since_ms") is not None
        and archive.get("until_ms") is not None
    ):
        turn_context["time_window"] = {
            "since_ms": int(archive["since_ms"]),
            "until_ms": int(archive["until_ms"]),
        }
        turn_context["console_default_time_window"] = True
    if archive.get("source"):
        turn_context["console_archive_source"] = str(archive["source"])
    if archive.get("probe_id"):
        turn_context["console_archive_probe_id"] = str(archive["probe_id"])
    if archive.get("sort_by"):
        turn_context["console_archive_sort_by"] = str(archive["sort_by"])
    if archive.get("rows"):
        turn_context["console_archive_rows"] = int(archive["rows"])
    return turn_context


def trusted_console_context_message(value: Mapping[str, Any] | None) -> str:
    """Format validated state as a short harness-owned instruction."""

    if not value:
        return ""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        "Trusted operator-console context (current UI state, not visual evidence): "
        f"{payload}. Use it only as default tool scope. Explicit operator wording and "
        "explicit tool arguments override these defaults."
    )


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if 0 < parsed <= 2_147_483_647 else None


def _nonnegative_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if 0 <= parsed <= 10_000_000_000_000 else None
