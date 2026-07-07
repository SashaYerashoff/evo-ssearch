"""Deterministic fakes for the EVA coherence benchmark harness."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Optional, Sequence

try:
    from .scenarios import (
        CHANNELS,
        DETECTION_ROWS,
        FIXTURE_RUN_ID,
        SUMMARY_LOGS_BY_CHANNEL,
        SUMMARY_ROLLUPS_BY_CHANNEL,
    )
except ImportError:  # pragma: no cover - useful when this file is imported directly.
    from scenarios import (  # type: ignore
        CHANNELS,
        DETECTION_ROWS,
        FIXTURE_RUN_ID,
        SUMMARY_LOGS_BY_CHANNEL,
        SUMMARY_ROLLUPS_BY_CHANNEL,
    )


def _row_ts_ms(row: Mapping[str, Any]) -> int:
    for key in ("timestamp_ms", "event_timestamp_ms", "recorded_at_ms"):
        value = row.get(key)
        if value is not None:
            return int(value)
    payload = row.get("payload")
    if isinstance(payload, Mapping) and payload.get("frame_timestamp_ms") is not None:
        return int(payload["frame_timestamp_ms"])
    return 0


def _node_start(node: Mapping[str, Any]) -> Optional[float]:
    value = node.get("window_start", node.get("start_ts"))
    return float(value) if value is not None else None


def _node_end(node: Mapping[str, Any]) -> Optional[float]:
    value = node.get("window_end", node.get("end_ts", node.get("created_at")))
    return float(value) if value is not None else None


def _node_overlaps(node: Mapping[str, Any], start_ts: Optional[float], end_ts: Optional[float]) -> bool:
    node_start = _node_start(node)
    node_end = _node_end(node)
    if node_start is None and node.get("created_at") is not None:
        node_start = float(node["created_at"])
    if node_end is None:
        node_end = node_start
    if node_start is None:
        return True
    if start_ts is not None and node_end is not None and node_end < float(start_ts):
        return False
    if end_ts is not None and node_start > float(end_ts):
        return False
    return True


def _log_ts(log: Mapping[str, Any]) -> Optional[float]:
    for key in ("created_at", "window_start", "start_ts"):
        value = log.get(key)
        if value is not None:
            return float(value)
    return None


def _matches_run(row: Mapping[str, Any], run_selector: Optional[str]) -> bool:
    if not run_selector or run_selector in {"all", "latest"}:
        return True
    return str(row.get("run_id") or "") == str(run_selector)


def _row_text(row: Mapping[str, Any]) -> str:
    payload = row.get("payload")
    payload_summary = payload.get("summary") if isinstance(payload, Mapping) else ""
    return " ".join(
        str(part or "")
        for part in (
            row.get("probe_name"),
            row.get("probe_id"),
            row.get("source"),
            row.get("severity"),
            payload_summary,
        )
    ).lower()


def _semantic_score(row: Mapping[str, Any], query: str) -> float:
    text = _row_text(row)
    query_text = str(query or "").lower()
    try:
        base = float(row.get("similarity") or row.get("margin") or 0.25)
    except Exception:
        base = 0.25

    score = base
    if "orlandina" in query_text:
        if "orlandina" in text and any(token in text for token in ("visible", "enters", "present", "sits")):
            score += 0.22
        if any(token in query_text for token in ("empty", "no cat", "absent")):
            if any(token in text for token in ("absent", "empty", "not visible", "no frame evidence")):
                score += 0.18
            if any(token in text for token in ("visible", "enters", "present", "sits")):
                score -= 0.35

    if "dog" in query_text:
        if "dog" in text:
            score += 0.08
        text_no_visible_tag = any(
            token in text
            for token in ("no visible ear tag", "without visible ear tag", "no ear tag is visible")
        )
        query_no_visible_tag = any(
            token in query_text
            for token in ("without visible ear tag", "without a visible ear tag", "no visible ear tag")
        )
        query_visible_tag = (
            "with visible ear tag" in query_text or "with a visible ear tag" in query_text
        ) and not query_no_visible_tag
        if query_no_visible_tag:
            if text_no_visible_tag:
                score += 0.22
            elif "ear tag" in text:
                score -= 0.12
        if query_visible_tag:
            if text_no_visible_tag:
                score -= 0.35
            elif "ear tag" in text:
                score += 0.16

    return max(0.0, min(1.0, score))


class FakeLuxriotManager:
    """Small in-memory manager with the methods used by ``AgentTools``."""

    def __init__(
        self,
        *,
        channels: Optional[Sequence[Mapping[str, Any]]] = None,
        rollups_by_channel: Optional[Mapping[int, Mapping[str, Sequence[Mapping[str, Any]]]]] = None,
        logs_by_channel: Optional[Mapping[int, Sequence[Mapping[str, Any]]]] = None,
    ) -> None:
        self.channels = [dict(channel) for channel in (channels or CHANNELS)]
        source_rollups = rollups_by_channel or SUMMARY_ROLLUPS_BY_CHANNEL
        self.rollups_by_channel: dict[int, dict[str, list[dict[str, Any]]]] = {
            int(channel_id): {
                str(level): [dict(node) for node in nodes]
                for level, nodes in levels.items()
            }
            for channel_id, levels in source_rollups.items()
        }
        source_logs = logs_by_channel or SUMMARY_LOGS_BY_CHANNEL
        self.logs_by_channel: dict[int, list[dict[str, Any]]] = {
            int(channel_id): [dict(row) for row in rows]
            for channel_id, rows in source_logs.items()
        }

    def get_channels(self, force: bool = False) -> list[dict[str, Any]]:
        return deepcopy(self.channels)

    def session_status(
        self,
        channel_id: int,
        run_selector: Optional[str] = None,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        deltas = []
        for row in self.logs_by_channel.get(int(channel_id), []):
            if not _matches_run(row, run_selector):
                continue
            created_at = _log_ts(row)
            if created_at is not None and start_ts is not None and created_at < float(start_ts):
                continue
            if created_at is not None and end_ts is not None and created_at > float(end_ts):
                continue
            deltas.append(dict(row))

        deltas.sort(key=lambda row: (_log_ts(row) or 0.0, str(row.get("summary") or "")))
        if isinstance(limit, int) and limit > 0:
            deltas = deltas[-limit:]

        latest_run_id = None
        for row in reversed(deltas):
            run_id = str(row.get("run_id") or "").strip()
            if run_id:
                latest_run_id = run_id
                break
        selected_run_id = None if run_selector in {None, "all"} else str(run_selector)
        if run_selector == "latest":
            selected_run_id = latest_run_id

        return {
            "running": False,
            "channel_id": int(channel_id),
            "run_id": latest_run_id,
            "logs": deepcopy(deltas),
            "logs_total": len(self.logs_by_channel.get(int(channel_id), [])),
            "logs_filtered": len(deltas),
            "runs": [
                {
                    "run_id": FIXTURE_RUN_ID,
                    "started_at": min((_log_ts(row) or 0.0) for row in deltas) if deltas else None,
                    "ended_at": max((_log_ts(row) or 0.0) for row in deltas) if deltas else None,
                    "log_count": len(deltas),
                    "running": False,
                }
            ],
            "selected_run": {"run_id": selected_run_id} if selected_run_id else None,
            "run_filter_id": selected_run_id,
            "latest_run_id": latest_run_id,
            "running_run_id": None,
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
    ) -> dict[str, Any]:
        source_levels = self.rollups_by_channel.get(int(channel_id), {})
        levels: dict[str, list[dict[str, Any]]] = {}
        for level in ("L0", "L1", "L2", "L3"):
            nodes = [
                dict(node)
                for node in source_levels.get(level, [])
                if _matches_run(node, run_selector) and _node_overlaps(node, start_ts, end_ts)
            ]
            nodes.sort(key=lambda node: (_node_start(node) or 0.0, _node_end(node) or 0.0))
            if isinstance(level_limit, int) and level_limit > 0:
                nodes = nodes[-level_limit:]
            levels[level] = nodes

        status = self.session_status(
            channel_id=channel_id,
            run_selector=run_selector,
            start_ts=start_ts,
            end_ts=end_ts,
            limit=None,
        )
        return {
            "channel_id": int(channel_id),
            "running": False,
            "runs": status.get("runs"),
            "selected_run": status.get("selected_run"),
            "run_filter_id": status.get("run_filter_id"),
            "running_run_id": None,
            "latest_run_id": status.get("latest_run_id"),
            "from_ts": start_ts,
            "to_ts": end_ts,
            "level_limit": level_limit,
            "source_counts": {level: len(nodes) for level, nodes in levels.items()},
            "levels": deepcopy(levels),
        }


class FakeDetectionStore:
    """Small in-memory detection archive with AgentTools-compatible methods."""

    def __init__(self, rows: Optional[Sequence[Mapping[str, Any]]] = None) -> None:
        self.rows = [dict(row) for row in (rows or DETECTION_ROWS)]

    def list_detections(
        self,
        probe_id: Optional[str] = None,
        channel_id: Optional[int] = None,
        since_ms: Optional[int] = None,
        until_ms: Optional[int] = None,
        limit: int = 50,
        offset: int = 0,
        source: Optional[str] = None,
    ) -> tuple[list[dict[str, Any]], int]:
        limit = max(1, min(500, int(limit or 50)))
        offset = max(0, int(offset or 0))
        filtered: list[dict[str, Any]] = []
        for row in self.rows:
            if probe_id is not None and str(row.get("probe_id") or "") != str(probe_id):
                continue
            if channel_id is not None and int(row.get("channel_id") or 0) != int(channel_id):
                continue
            if source is not None and str(row.get("source") or "") != str(source):
                continue
            timestamp_ms = _row_ts_ms(row)
            if since_ms is not None and timestamp_ms < int(since_ms):
                continue
            if until_ms is not None and timestamp_ms > int(until_ms):
                continue
            filtered.append(dict(row))

        filtered.sort(key=lambda row: (_row_ts_ms(row), int(row.get("id") or 0)), reverse=True)
        total = len(filtered)
        return deepcopy(filtered[offset : offset + limit]), total

    def summarize_by_probe(
        self,
        since_ms: Optional[int] = None,
        channel_id: Optional[int] = None,
        limit: int = 100,
        source: Optional[str] = None,
        until_ms: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, int, str], dict[str, Any]] = {}
        rows, _total = self.list_detections(
            probe_id=None,
            channel_id=channel_id,
            since_ms=since_ms,
            until_ms=until_ms,
            limit=500,
            offset=0,
            source=source,
        )
        for row in rows:
            source_key = str(row.get("source") or "")
            key = (str(row.get("probe_id") or ""), int(row.get("channel_id") or 0), source_key)
            timestamp_ms = _row_ts_ms(row)
            slot = grouped.setdefault(
                key,
                {
                    "probe_id": row.get("probe_id"),
                    "probe_name": row.get("probe_name"),
                    "channel_id": row.get("channel_id"),
                    "source": source_key,
                    "hit_count": 0,
                    "latest_timestamp_ms": 0,
                },
            )
            slot["hit_count"] += 1
            slot["latest_timestamp_ms"] = max(int(slot["latest_timestamp_ms"]), timestamp_ms)

        summary_rows = list(grouped.values())
        summary_rows.sort(
            key=lambda row: (int(row.get("latest_timestamp_ms") or 0), str(row.get("probe_id") or "")),
            reverse=True,
        )
        return deepcopy(summary_rows[: max(1, min(500, int(limit or 100)))])

    def fetch_detections_by_ids(
        self,
        ids: Sequence[int],
        include_vectors: bool = False,
    ) -> list[dict[str, Any]]:
        wanted = {int(item) for item in ids}
        rows: list[dict[str, Any]] = []
        for row in self.rows:
            row_id = row.get("id", row.get("detection_id"))
            if row_id is None or int(row_id) not in wanted:
                continue
            copied = dict(row)
            # The coherence harness does not ship real fixture image files.
            # Force AgentTools.describe_frame to use the embedded thumbnail path.
            copied["image_path"] = ""
            rows.append(copied)
        return deepcopy(rows)

    def search_detections(
        self,
        *,
        query: str,
        probe_id: Optional[str] = None,
        channel_id: Optional[int] = None,
        source: Optional[str] = None,
        since_ms: Optional[int] = None,
        until_ms: Optional[int] = None,
        limit: int = 12,
        sort_by: str = "similarity",
        candidate_limit: int = 20000,
        mode: str = "clip",
    ) -> list[dict[str, Any]]:
        rows, _total = self.list_detections(
            probe_id=probe_id,
            channel_id=channel_id,
            since_ms=since_ms,
            until_ms=until_ms,
            limit=min(max(int(candidate_limit or limit), int(limit)), 500),
            offset=0,
            source=source,
        )
        scored: list[dict[str, Any]] = []
        for row in rows:
            copied = dict(row)
            copied["similarity"] = _semantic_score(copied, query)
            copied["score"] = copied["similarity"]
            copied["search_mode"] = mode
            scored.append(copied)
        if sort_by == "time":
            scored.sort(key=lambda row: (_row_ts_ms(row), int(row.get("id") or 0)), reverse=True)
        else:
            scored.sort(
                key=lambda row: (
                    float(row.get("similarity") or 0.0),
                    _row_ts_ms(row),
                    int(row.get("id") or 0),
                ),
                reverse=True,
            )
        return deepcopy(scored[: max(1, min(500, int(limit or 12)))])


def build_environment() -> dict[str, Any]:
    """Return the minimal fake runtime environment consumed by the runner."""
    detections_store = FakeDetectionStore()
    return {
        "luxriot_manager": FakeLuxriotManager(),
        "detections_store": detections_store,
        "probes_store": None,
    }
