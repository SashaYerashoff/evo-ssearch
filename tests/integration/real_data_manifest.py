"""Frozen, read-only truth manifest for live EVA agent acceptance tests.

The manifest deliberately uses the same authenticated HTTP endpoints as the
vanilla frontend.  Callers must provide an immutable time window so live streams
can continue adding rows without changing the assertions for the test run.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .eva_client import EvaSession


ARCHIVE_SOURCES = ("vlm_summary", "vlm_alert", "probe")


def _number(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _log_span(log: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
    start_ms = _number(log.get("batch_start_ms"))
    end_ms = _number(log.get("batch_end_ms"))
    if start_ms is not None or end_ms is not None:
        start_ms = end_ms if start_ms is None else start_ms
        end_ms = start_ms if end_ms is None else end_ms
        assert start_ms is not None and end_ms is not None
        start, end = start_ms / 1000.0, end_ms / 1000.0
    else:
        created = _number(log.get("created_at"))
        if created is None:
            return None
        if created > 100_000_000_000:
            created /= 1000.0
        start = end = created
    if end < start:
        start, end = end, start
    return start, end


def _coverage_from_logs(
    logs: Sequence[Mapping[str, Any]],
    *,
    from_ts: float,
    to_ts: float,
) -> Dict[str, Any]:
    spans: List[Tuple[float, float]] = []
    run_ids = set()
    for log in logs:
        span = _log_span(log)
        if span is None:
            continue
        start = max(float(from_ts), span[0])
        end = min(float(to_ts), span[1])
        if end < float(from_ts) or start > float(to_ts):
            continue
        spans.append((start, max(start, end)))
        run_id = str(log.get("run_id") or "").strip()
        if run_id:
            run_ids.add(run_id)

    spans.sort()
    merged: List[List[float]] = []
    for start, end in spans:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    gaps: List[Dict[str, float]] = []
    cursor = float(from_ts)
    for start, end in merged:
        if start > cursor:
            gaps.append({"from_ts": cursor, "to_ts": start, "duration_sec": start - cursor})
        cursor = max(cursor, end)
    if cursor < float(to_ts):
        gaps.append({"from_ts": cursor, "to_ts": float(to_ts), "duration_sec": float(to_ts) - cursor})

    requested = max(0.0, float(to_ts) - float(from_ts))
    covered = sum(max(0.0, end - start) for start, end in merged)
    return {
        "log_count": len(logs),
        "span_count": len(spans),
        "first_ts": spans[0][0] if spans else None,
        "last_ts": spans[-1][1] if spans else None,
        "run_ids": sorted(run_ids),
        "union_covered_sec": covered,
        "coverage_ratio": 1.0 if requested <= 0 else min(1.0, covered / requested),
        "gap_count": len(gaps),
        "gaps": gaps,
    }


def _channel_id(row: Mapping[str, Any]) -> Optional[int]:
    raw = row.get("id") if row.get("id") is not None else row.get("channel_id")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _archive_bounds(
    session: EvaSession,
    *,
    channel_id: int,
    source: str,
    since_ms: int,
    until_ms: int,
) -> Dict[str, Any]:
    params = {
        "channel_id": channel_id,
        "source": source,
        "since_ms": since_ms,
        "until_ms": until_ms,
        "limit": 1,
        "offset": 0,
    }
    newest_payload = session.get_json("/detections/list", params=params)
    total = int(newest_payload.get("total") or 0)
    newest_rows = newest_payload.get("detections") or []
    oldest_rows: Sequence[Mapping[str, Any]] = []
    if total > 1:
        oldest_payload = session.get_json(
            "/detections/list",
            params={**params, "offset": total - 1},
        )
        oldest_rows = oldest_payload.get("detections") or []
    elif total == 1:
        oldest_rows = newest_rows

    def timestamp(rows: Sequence[Mapping[str, Any]]) -> Optional[int]:
        if not rows or not isinstance(rows[0], Mapping):
            return None
        raw = rows[0].get("timestamp_ms")
        if raw is None:
            raw = rows[0].get("event_timestamp_ms")
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    return {
        "total": total,
        "first_timestamp_ms": timestamp(oldest_rows),
        "last_timestamp_ms": timestamp(newest_rows),
    }


def build_frozen_manifest(
    session: EvaSession,
    *,
    from_ts: float,
    to_ts: float,
    channel_ids: Optional[Iterable[int]] = None,
    archive_sources: Sequence[str] = ARCHIVE_SOURCES,
    summary_log_limit: int = 100_000,
    require_admin_all_channels: bool = True,
) -> Dict[str, Any]:
    """Collect immutable frontend-visible truth for later agent assertions."""

    from_ts, to_ts = float(from_ts), float(to_ts)
    if to_ts < from_ts:
        from_ts, to_ts = to_ts, from_ts
    if to_ts <= from_ts:
        raise ValueError("frozen manifest window must have positive duration")

    whoami = session.whoami()
    user = whoami.get("user") if isinstance(whoami.get("user"), Mapping) else {}
    allowed = {str(item) for item in user.get("allowedChannelIds") or []}
    if require_admin_all_channels and "*" not in allowed:
        raise PermissionError("live manifest requires named admin with all-channel access")

    inventory_payload = session.get_json("/luxriot/channels", params={"force": 1})
    inventory = inventory_payload.get("channels") or []
    streams = session.get_json("/luxriot/streams")
    requested = {int(item) for item in channel_ids or [] if int(item) > 0}

    video_by_channel = {
        int(row["channel_id"]): dict(row)
        for row in streams.get("video_streams") or []
        if isinstance(row, Mapping) and _number(row.get("channel_id")) is not None
    }
    desired_ids = {
        int(item)
        for item in streams.get("desired_video_channels") or []
        if _number(item) is not None
    }

    channels: Dict[str, Dict[str, Any]] = {}
    since_ms, until_ms = int(from_ts * 1000.0), int(to_ts * 1000.0)
    for raw_channel in inventory:
        if not isinstance(raw_channel, Mapping):
            continue
        channel_id = _channel_id(raw_channel)
        if channel_id is None or (requested and channel_id not in requested):
            continue
        status = session.get_json(
            "/luxriot/session",
            params={
                "channel_id": channel_id,
                "run": "all",
                "from_ts": from_ts,
                "to_ts": to_ts,
                "limit": max(1, int(summary_log_limit)),
            },
        )
        logs = [row for row in status.get("logs") or [] if isinstance(row, Mapping)]
        summary_memory = _coverage_from_logs(logs, from_ts=from_ts, to_ts=to_ts)
        summary_memory["possibly_truncated"] = len(logs) >= max(1, int(summary_log_limit))
        channels[str(channel_id)] = {
            "channel_id": channel_id,
            "title": str(raw_channel.get("title") or raw_channel.get("name") or f"channel-{channel_id}"),
            "inventory": dict(raw_channel),
            "desired": channel_id in desired_ids,
            "stream": video_by_channel.get(channel_id),
            "summary_memory": summary_memory,
            "archive": {
                source: _archive_bounds(
                    session,
                    channel_id=channel_id,
                    source=str(source),
                    since_ms=since_ms,
                    until_ms=until_ms,
                )
                for source in archive_sources
            },
        }

    missing_requested = sorted(requested.difference(int(key) for key in channels))
    return {
        "version": 1,
        "frozen_window": {
            "from_ts": from_ts,
            "to_ts": to_ts,
            "since_ms": since_ms,
            "until_ms": until_ms,
        },
        "auth": {
            "username": user.get("username"),
            "roles": list(user.get("roles") or []),
            "allowed_channel_ids": sorted(allowed),
        },
        "inventory_count": len(inventory),
        "requested_channel_ids": sorted(requested) if requested else sorted(int(key) for key in channels),
        "missing_requested_channel_ids": missing_requested,
        "channels": channels,
    }

