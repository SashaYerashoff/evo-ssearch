#!/usr/bin/env python3
"""Collect a secret-safe timing snapshot for the Georgia upgrade rehearsal.

The report keeps four different clocks separate:

* the small CPU CV apex calculation;
* SigLIP queue and CUDA inference reported by the live worker;
* VLM batching/inference;
* event-to-Evo bookmark acknowledgement for probes and VLM alerts.

It is read-only.  It does not create probes, alerts or bookmarks and never
prints database credentials or image data.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_ENV = Path("/home/sasha/Projects/eva-georgia-upgrade-repro/.env")
DEFAULT_BASE_URL = "http://127.0.0.1:5081"


def parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key.strip()] = value
    for _ in range(8):
        changed = False
        for key, value in tuple(values.items()):
            expanded = re.sub(
                r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}",
                lambda match: values.get(match.group(1), match.group(0)),
                value,
            )
            if expanded != value:
                values[key] = expanded
                changed = True
        if not changed:
            break
    return values


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def timing_stats(values: Iterable[Any]) -> dict[str, Any]:
    samples = [number for value in values if (number := finite_number(value)) is not None and number >= 0.0]
    if not samples:
        return {"samples": 0}
    return {
        "samples": len(samples),
        "min_ms": round(min(samples), 3),
        "p50_ms": round(float(percentile(samples, 0.50) or 0.0), 3),
        "p95_ms": round(float(percentile(samples, 0.95) or 0.0), 3),
        "max_ms": round(max(samples), 3),
        "mean_ms": round(statistics.fmean(samples), 3),
    }


def http_json(url: str, timeout: float = 10.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read(8 * 1024 * 1024).decode("utf-8"))


def git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def connect_database(values: Mapping[str, str]) -> Any:
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - depends on the field venv
        raise RuntimeError("psycopg is required; run with the EVA application venv") from exc
    dsn = str(values.get("EVA_DATABASE_DSN") or values.get("EVOSSEARCH_DATABASE_DSN") or "").strip()
    if not dsn:
        raise RuntimeError("EVA_DATABASE_DSN is not declared in the selected environment file")
    return psycopg.connect(dsn, connect_timeout=8)


def database_rows(
    connection: Any,
    tenant_id: str,
    lookback_hours: float,
    row_limit: int,
) -> tuple[str | None, list[dict[str, Any]]]:
    threshold_ms = int((time.time() - max(0.25, float(lookback_hours)) * 3600.0) * 1000.0)
    bounded_limit = max(16, min(10_000, int(row_limit)))
    with connection.cursor() as cursor:
        cursor.execute("SELECT set_config('eva.tenant_id', %s, true)", (tenant_id,))
        cursor.execute("SELECT version_num FROM public.alembic_version LIMIT 1")
        schema_row = cursor.fetchone()
        cursor.execute(
            """
            WITH selected AS (
              (
                SELECT id, source, channel_id, event_timestamp_ms, created_at, payload_json
                FROM archive.detections
                WHERE tenant_id = %s::uuid
                  AND source = 'probe'
                  AND event_timestamp_ms >= %s
                  AND payload_json #>> '{context,bookmark_gate,event_to_bookmark_ack_ms}' IS NOT NULL
                ORDER BY event_timestamp_ms DESC, id DESC
                LIMIT %s
              )
              UNION ALL
              (
                SELECT id, source, channel_id, event_timestamp_ms, created_at, payload_json
                FROM archive.detections
                WHERE tenant_id = %s::uuid
                  AND source = 'vlm_alert'
                  AND event_timestamp_ms >= %s
                ORDER BY event_timestamp_ms DESC, id DESC
                LIMIT %s
              )
            )
            SELECT
              id,
              source,
              channel_id,
              event_timestamp_ms,
              created_at,
              jsonb_build_object(
                'origin', payload_json -> 'origin',
                'fast_alert_phase', payload_json -> 'fast_alert_phase',
                'batch_id', payload_json -> 'batch_id',
                'context', jsonb_build_object(
                  'bookmark_gate', payload_json #> '{context,bookmark_gate}'
                ),
                'alert_event', payload_json -> 'alert_event',
                'latency_trace', payload_json -> 'latency_trace'
              )
            FROM selected
            ORDER BY id ASC
            """,
            (
                tenant_id,
                threshold_ms,
                bounded_limit,
                tenant_id,
                threshold_ms,
                bounded_limit,
            ),
        )
        rows = [
            {
                "id": row[0],
                "source": row[1],
                "channel_id": row[2],
                "event_timestamp_ms": row[3],
                "created_at": row[4].isoformat() if row[4] else None,
                "payload": row[5] if isinstance(row[5], dict) else {},
            }
            for row in cursor.fetchall()
        ]
    return (str(schema_row[0]) if schema_row else None), rows


def bookmark_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, dict[str, list[float]]] = {
        "probe_realtime": {"event_to_ack": [], "delivery": [], "embedding_age": []},
        "probe_daemon": {"event_to_ack": [], "delivery": [], "embedding_age": []},
        "vlm_fast": {"event_to_ack": [], "delivery": []},
        "vlm_full": {"event_to_ack": [], "delivery": []},
    }
    seen_bookmarks: set[tuple[str, int, int]] = set()
    for row in rows:
        payload = row.get("payload") if isinstance(row.get("payload"), Mapping) else {}
        if row.get("source") == "probe":
            context = payload.get("context") if isinstance(payload.get("context"), Mapping) else {}
            gate = context.get("bookmark_gate") if isinstance(context.get("bookmark_gate"), Mapping) else {}
            source = str(gate.get("source") or payload.get("origin") or "")
            if source not in {"probe_realtime", "probe_daemon"} or not gate.get("sent"):
                continue
            ack = int(finite_number(gate.get("bookmark_ack_at_ms")) or 0)
            key = (source, int(row.get("event_timestamp_ms") or 0), ack)
            if key in seen_bookmarks:
                continue
            seen_bookmarks.add(key)
            buckets[source]["event_to_ack"].append(gate.get("event_to_bookmark_ack_ms"))
            buckets[source]["delivery"].append(gate.get("bookmark_delivery_ms"))
            buckets[source]["embedding_age"].append(gate.get("embedding_event_age_ms"))
            continue

        alert_event = payload.get("alert_event") if isinstance(payload.get("alert_event"), Mapping) else {}
        if alert_event.get("delivery_status") != "sent":
            continue
        source = "vlm_fast" if payload.get("fast_alert_phase") is True else "vlm_full"
        ack = int(finite_number(alert_event.get("bookmark_ack_at_ms")) or 0)
        key = (source, int(alert_event.get("timestamp_ms") or row.get("event_timestamp_ms") or 0), ack)
        if key in seen_bookmarks:
            continue
        seen_bookmarks.add(key)
        buckets[source]["event_to_ack"].append(alert_event.get("event_to_bookmark_ack_ms"))
        buckets[source]["delivery"].append(alert_event.get("bookmark_delivery_ms"))

    return {
        source: {name: timing_stats(values) for name, values in metrics.items()}
        for source, metrics in buckets.items()
    }


def vlm_stage_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, dict[str, list[float]]] = {
        "vlm_fast": {"batch_span": [], "admission_wait": [], "http": [], "inference": [], "event_to_completion": []},
        "vlm_full": {"batch_span": [], "admission_wait": [], "http": [], "inference": [], "event_to_completion": []},
    }
    seen: set[tuple[str, int, int]] = set()
    for row in rows:
        if row.get("source") != "vlm_alert":
            continue
        payload = row.get("payload") if isinstance(row.get("payload"), Mapping) else {}
        trace = payload.get("latency_trace") if isinstance(payload.get("latency_trace"), Mapping) else {}
        source = "vlm_fast" if payload.get("fast_alert_phase") is True else "vlm_full"
        started = int(finite_number(trace.get("inference_started_at_ms")) or 0)
        completed = int(finite_number(trace.get("inference_completed_at_ms")) or 0)
        key = (source, started, completed)
        if key in seen or not any(key[1:]):
            continue
        seen.add(key)
        first = finite_number(trace.get("batch_first_frame_at_ms"))
        last = finite_number(trace.get("batch_last_frame_at_ms"))
        final = finite_number(trace.get("alert_processing_completed_at_ms")) or finite_number(trace.get("completed_at_ms"))
        if first is not None and last is not None:
            buckets[source]["batch_span"].append(max(0.0, last - first))
        if first is not None and final is not None:
            buckets[source]["event_to_completion"].append(max(0.0, final - first))
        buckets[source]["admission_wait"].append(trace.get("lm_admission_wait_ms"))
        buckets[source]["http"].append(trace.get("lm_http_ms"))
        buckets[source]["inference"].append(trace.get("inference_ms"))
    return {
        source: {name: timing_stats(values) for name, values in metrics.items()}
        for source, metrics in buckets.items()
    }


def rows_within_hours(
    rows: Sequence[Mapping[str, Any]],
    *,
    now: datetime,
    hours: float,
) -> list[Mapping[str, Any]]:
    threshold = now - timedelta(hours=max(0.25, float(hours)))
    selected: list[Mapping[str, Any]] = []
    for row in rows:
        raw = str(row.get("created_at") or "").strip()
        try:
            created_at = datetime.fromisoformat(raw)
        except ValueError:
            continue
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if created_at >= threshold:
            selected.append(row)
    return selected


def cv_apex_benchmark(connection: Any, tenant_id: str, frames: int, repeats: int) -> dict[str, Any]:
    try:
        from PIL import Image, ImageChops, ImageFilter, ImageStat
    except ImportError as exc:
        return {"ok": False, "error": f"benchmark import failed: {type(exc).__name__}"}

    # Keep this benchmark independent from the live runtime module. Importing
    # luxriot_connector also imports the CUDA stack and would itself perturb the
    # SigLIP window we are trying to observe. These are the exact three PIL
    # operations used by LuxriotCaptureSession's capture apex v2 helpers.
    def to_gray(snapshot: Any, max_edge: int = 160) -> Any:
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

    def delta_score(previous: Any, current: Any) -> float | None:
        if previous is None or current is None:
            return None
        if previous.size != current.size:
            resampling = getattr(Image, "Resampling", Image)
            previous = previous.resize(current.size, resample=resampling.BILINEAR)
        mean = ImageStat.Stat(ImageChops.difference(previous, current)).mean
        return max(0.0, min(1.0, float(mean[0]) / 255.0)) if mean else 0.0

    def sharpness_score(gray: Any) -> float | None:
        stat = ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES))
        return max(0.0, float(stat.var[0])) if stat.var else None

    with connection.cursor() as cursor:
        cursor.execute("SELECT set_config('eva.tenant_id', %s, true)", (tenant_id,))
        cursor.execute(
            """
            SELECT thumbnail_b64
            FROM archive.detections
            WHERE tenant_id = %s::uuid
              AND source = 'semantic_snapshot'
              AND thumbnail_b64 IS NOT NULL
            ORDER BY id DESC
            LIMIT %s
            """,
            (tenant_id, max(2, int(frames))),
        )
        encoded = [row[0] for row in cursor.fetchall()]
    images = []
    for raw in reversed(encoded):
        try:
            data = str(raw).split(",", 1)[-1]
            images.append(Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB"))
        except Exception:
            continue
    if len(images) < 2:
        return {"ok": False, "error": "not enough archived semantic thumbnails"}

    previous = None
    for image in images[: min(10, len(images))]:
        gray = to_gray(image)
        delta_score(previous, gray)
        sharpness_score(gray)
        previous = gray

    timings = {"gray": [], "delta": [], "sharpness": [], "total": []}
    previous = None
    for _ in range(max(1, int(repeats))):
        for image in images:
            started = time.perf_counter_ns()
            gray = to_gray(image)
            gray_done = time.perf_counter_ns()
            delta_score(previous, gray)
            delta_done = time.perf_counter_ns()
            sharpness_score(gray)
            sharp_done = time.perf_counter_ns()
            previous = gray
            timings["gray"].append((gray_done - started) / 1_000_000.0)
            timings["delta"].append((delta_done - gray_done) / 1_000_000.0)
            timings["sharpness"].append((sharp_done - delta_done) / 1_000_000.0)
            timings["total"].append((sharp_done - started) / 1_000_000.0)
    return {
        "ok": True,
        "algorithm": "capture_per_second_cv_apex_v2",
        "frames": len(images),
        "repeats": max(1, int(repeats)),
        "operations": {name: timing_stats(values) for name, values in timings.items()},
        "scope": "isolated CPU gray-160 + frame delta + edge variance; excludes capture, JPEG, SigLIP and VLM",
    }


def compact_runtime(ready: Mapping[str, Any]) -> dict[str, Any]:
    checks = ready.get("checks") if isinstance(ready.get("checks"), Mapping) else {}
    attention = checks.get("attention") if isinstance(checks.get("attention"), Mapping) else {}
    microbatch = attention.get("clip_microbatcher") if isinstance(attention.get("clip_microbatcher"), Mapping) else {}
    return {
        "status": ready.get("status"),
        "version": ready.get("version"),
        "database": checks.get("database"),
        "luxriot": checks.get("luxriot"),
        "luxriot_restore": checks.get("luxriot_restore"),
        "lm_profiles": checks.get("lm_profiles"),
        "siglip": {
            "backend": (checks.get("embedder") or {}).get("backend") if isinstance(checks.get("embedder"), Mapping) else None,
            "model": (checks.get("embedder") or {}).get("clip_model") if isinstance(checks.get("embedder"), Mapping) else None,
            "device": (checks.get("embedder") or {}).get("device") if isinstance(checks.get("embedder"), Mapping) else None,
            "recent": microbatch.get("recent"),
            "last_batch_compute_ms": microbatch.get("last_batch_compute_ms"),
            "last_batch_queue_wait_ms": microbatch.get("last_batch_queue_wait_ms"),
            "average_batch_size": microbatch.get("average_batch_size"),
            "queue_depth": microbatch.get("queue_depth"),
            "inflight": microbatch.get("inflight"),
            "cuda_graph": attention.get("siglip_cuda_graph"),
        },
        "probe_runtime": attention.get("realtime_probe_bookmarks"),
        "fast_vlm_runtime": attention.get("fast_vlm_alerts"),
        "capture_runtime": attention.get("capture_runtime"),
    }


def markdown_table_stat(stats: Mapping[str, Any], key: str) -> str:
    value = stats.get(key)
    return "—" if value is None else f"{float(value):.1f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    runtime = report.get("runtime") if isinstance(report.get("runtime"), Mapping) else {}
    siglip = runtime.get("siglip") if isinstance(runtime.get("siglip"), Mapping) else {}
    recent = siglip.get("recent") if isinstance(siglip.get("recent"), Mapping) else {}
    cv = report.get("cv_apex_benchmark") if isinstance(report.get("cv_apex_benchmark"), Mapping) else {}
    cv_total = ((cv.get("operations") or {}).get("total") or {}) if isinstance(cv.get("operations"), Mapping) else {}
    bookmarks = report.get("bookmark_latency_current") if isinstance(report.get("bookmark_latency_current"), Mapping) else {}
    historical_bookmarks = report.get("bookmark_latency_history") if isinstance(report.get("bookmark_latency_history"), Mapping) else {}
    vlm = report.get("vlm_stages_current") if isinstance(report.get("vlm_stages_current"), Mapping) else {}
    lines = [
        "# Georgia β 0.8.1 → β 0.8.7 latency snapshot",
        "",
        f"Collected: `{report.get('generated_at')}`  ",
        f"Source commit: `{report.get('source_commit') or 'unknown'}`  ",
        f"Live status: `{runtime.get('status')}`; schema: `{report.get('schema_revision')}`",
        f"Current window: `{report.get('current_hours')}` h; validation history: `{report.get('lookback_hours')}` h",
        f"Database sample cap: latest `{report.get('db_row_limit_per_source')}` rows per source",
        "",
        "## Processing cost",
        "",
        "| Stage | Samples | p50, ms | p95, ms | What is measured |",
        "|---|---:|---:|---:|---|",
        f"| CV apex | {cv_total.get('samples', 0)} | {markdown_table_stat(cv_total, 'p50_ms')} | {markdown_table_stat(cv_total, 'p95_ms')} | gray-160 + frame delta + edge variance, CPU |",
        f"| SigLIP batch compute | {recent.get('window_batches', 0)} | {float(recent.get('compute_ms_p50') or 0):.1f} | {float(recent.get('compute_ms_p95') or 0):.1f} | live recent worker window |",
        f"| SigLIP queue wait | {recent.get('window_batches', 0)} | {float(recent.get('queue_wait_ms_p50') or 0):.1f} | {float(recent.get('queue_wait_ms_p95') or 0):.1f} | live recent worker window |",
    ]
    for source, label in (("vlm_fast", "Fast VLM"), ("vlm_full", "Full L0 VLM")):
        metrics = vlm.get(source) if isinstance(vlm.get(source), Mapping) else {}
        inference = metrics.get("inference") if isinstance(metrics.get("inference"), Mapping) else {}
        total = metrics.get("event_to_completion") if isinstance(metrics.get("event_to_completion"), Mapping) else {}
        lines.append(
            f"| {label} inference | {inference.get('samples', 0)} | {markdown_table_stat(inference, 'p50_ms')} | {markdown_table_stat(inference, 'p95_ms')} | model execution only |"
        )
        lines.append(
            f"| {label} event → processed | {total.get('samples', 0)} | {markdown_table_stat(total, 'p50_ms')} | {markdown_table_stat(total, 'p95_ms')} | batching/roll + queue + inference + parse |"
        )
    lines.extend([
        "",
        "## Bookmark acknowledgement in Evo",
        "",
        "| Pipeline | Sent samples | Event → Evo ack p50, ms | p95, ms | EVA → Evo delivery p50, ms |",
        "|---|---:|---:|---:|---:|",
    ])
    for source, label in (
        ("probe_realtime", "Operator probe, direct lane"),
        ("probe_daemon", "Probe retrospective fallback"),
        ("vlm_fast", "Fast VLM alert"),
        ("vlm_full", "Full L0 VLM alert"),
    ):
        metrics = bookmarks.get(source) if isinstance(bookmarks.get(source), Mapping) else {}
        total = metrics.get("event_to_ack") if isinstance(metrics.get("event_to_ack"), Mapping) else {}
        delivery = metrics.get("delivery") if isinstance(metrics.get("delivery"), Mapping) else {}
        lines.append(
            f"| {label} | {total.get('samples', 0)} | {markdown_table_stat(total, 'p50_ms')} | {markdown_table_stat(total, 'p95_ms')} | {markdown_table_stat(delivery, 'p50_ms')} |"
        )
    lines.extend([
        "",
        "### Validation history",
        "",
        "This second table retains controlled probe/VLM checks which may not have a fresh hit in the current window.",
        "",
        "| Pipeline | Sent samples | Event → Evo ack p50, ms | p95, ms | EVA → Evo delivery p50, ms |",
        "|---|---:|---:|---:|---:|",
    ])
    for source, label in (
        ("probe_realtime", "Operator probe, direct lane"),
        ("probe_daemon", "Probe retrospective fallback"),
        ("vlm_fast", "Fast VLM alert"),
        ("vlm_full", "Full L0 VLM alert"),
    ):
        metrics = historical_bookmarks.get(source) if isinstance(historical_bookmarks.get(source), Mapping) else {}
        total = metrics.get("event_to_ack") if isinstance(metrics.get("event_to_ack"), Mapping) else {}
        delivery = metrics.get("delivery") if isinstance(metrics.get("delivery"), Mapping) else {}
        lines.append(
            f"| {label} | {total.get('samples', 0)} | {markdown_table_stat(total, 'p50_ms')} | {markdown_table_stat(total, 'p95_ms')} | {markdown_table_stat(delivery, 'p50_ms')} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- CV and SigLIP values are compute/queue costs, not camera-to-operator latency.",
        "- A direct operator probe still waits for the configured embedding cadence and hit confirmation; its model score itself is a millisecond-scale operation.",
        "- Full L0 VLM includes the batch observation window. Fast VLM includes post-roll and admission. Evo delivery after EVA decides is normally only tens of milliseconds.",
        "- Only rows with an actual `sent` acknowledgement enter the bookmark table. Cooldown/deduplicated alerts are intentionally excluded.",
        "- Small sample counts are shown rather than hidden. Re-run immediately after the controlled thumbs-up tests for release acceptance numbers.",
        "- Database reads are deliberately capped per source so a timing preflight does not decompress the full historical evidence archive.",
        "- Semantic presence is an affinity/homeostasis signal, not object detection. CV apex is an attention selector, not event truth.",
        "",
    ])
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    result.add_argument("--base-url", default=DEFAULT_BASE_URL)
    result.add_argument("--lookback-hours", type=float, default=168.0)
    result.add_argument("--current-hours", type=float, default=6.0)
    result.add_argument(
        "--db-row-limit",
        type=int,
        default=1024,
        help="Maximum recent rows read per source; bounds TOAST/evidence decompression.",
    )
    result.add_argument("--cv-frames", type=int, default=128)
    result.add_argument("--cv-repeats", type=int, default=4)
    result.add_argument("--output-json", type=Path)
    result.add_argument("--output-markdown", type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    values = parse_env(args.env_file)
    tenant_id = str(values.get("EVOSSEARCH_ARCHIVE_TENANT_ID") or "").strip()
    if not tenant_id:
        raise SystemExit("EVOSSEARCH_ARCHIVE_TENANT_ID is missing")
    ready = http_json(args.base_url.rstrip("/") + "/ready")
    connection = connect_database(values)
    try:
        with connection.transaction():
            schema, rows = database_rows(
                connection,
                tenant_id,
                args.lookback_hours,
                args.db_row_limit,
            )
        with connection.transaction():
            cv = cv_apex_benchmark(connection, tenant_id, args.cv_frames, args.cv_repeats)
    finally:
        connection.close()
    repo_root = Path(__file__).resolve().parents[1]
    generated_at = datetime.now(timezone.utc)
    current_rows = rows_within_hours(rows, now=generated_at, hours=args.current_hours)
    report = {
        "schema_version": 1,
        "generated_at": generated_at.isoformat(),
        "source_commit": git_commit(repo_root),
        "env_file": str(args.env_file),
        "base_url": args.base_url,
        "lookback_hours": args.lookback_hours,
        "current_hours": args.current_hours,
        "db_row_limit_per_source": args.db_row_limit,
        "schema_revision": schema,
        "runtime": compact_runtime(ready),
        "cv_apex_benchmark": cv,
        "vlm_stages_current": vlm_stage_metrics(current_rows),
        "bookmark_latency_current": bookmark_metrics(current_rows),
        "bookmark_latency_history": bookmark_metrics(rows),
    }
    markdown = render_markdown(report)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
