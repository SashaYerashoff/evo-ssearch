#!/usr/bin/env python3
"""Read-only live L0 timing report for the installed EVA appliance."""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


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


def number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def delta(end: Any, start: Any) -> float | None:
    end_number = number(end)
    start_number = number(start)
    if end_number is None or start_number is None:
        return None
    return max(0.0, end_number - start_number)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def stats(values: Iterable[Any]) -> str:
    samples = [value for raw in values if (value := number(raw)) is not None and value >= 0]
    if not samples:
        return "n=0"
    return (
        f"n={len(samples)} p50={percentile(samples, 0.50) / 1000.0:.2f}s "
        f"p95={percentile(samples, 0.95) / 1000.0:.2f}s "
        f"max={max(samples) / 1000.0:.2f}s avg={statistics.fmean(samples) / 1000.0:.2f}s"
    )


def extract_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = row.get("payload") if isinstance(row.get("payload"), Mapping) else {}
    trace = payload.get("latency_trace") if isinstance(payload.get("latency_trace"), Mapping) else {}
    response = payload.get("lm_response_stats") if isinstance(payload.get("lm_response_stats"), Mapping) else {}
    first = number(trace.get("batch_first_frame_at_ms")) or number(payload.get("batch_start_ms"))
    last = number(trace.get("batch_last_frame_at_ms")) or number(payload.get("batch_end_ms"))
    sealed = number(trace.get("batch_sealed_at_ms"))
    enqueued = number(trace.get("summary_enqueued_at_ms"))
    dispatch = number(trace.get("summary_dispatch_started_at_ms"))
    prepared = number(trace.get("summary_prepared_at_ms"))
    inference_start = number(trace.get("inference_started_at_ms"))
    inference_end = number(trace.get("inference_completed_at_ms"))
    persisted = number(row.get("recorded_at_ms"))
    return {
        "channel_id": int(row["channel_id"]),
        "batch_id": str(row.get("batch_id") or ""),
        "recorded_at_ms": persisted,
        "frames": int(number(payload.get("frame_count")) or 0),
        "observation_ms": delta(last, first),
        "first_frame_to_result_ms": delta(persisted, first),
        "last_frame_to_result_ms": delta(persisted, last),
        "last_frame_to_seal_ms": delta(sealed, last),
        "seal_to_enqueue_ms": delta(enqueued, sealed),
        "executor_queue_ms": delta(dispatch, enqueued),
        "prepare_ms": delta(prepared, dispatch),
        "admission_wait_ms": number(response.get("admission_wait_ms")),
        "model_call_ms": delta(inference_end, inference_start),
        "model_http_ms": number(response.get("http_ms")),
        "seal_to_result_ms": delta(persisted, sealed),
        "model_to_result_ms": delta(persisted, inference_end),
        "input_images": int(number((payload.get("llm_input_stats") or {}).get("image_parts")) or 0)
        if isinstance(payload.get("llm_input_stats"), Mapping)
        else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path("/mnt/eva-llamacpp-lab/factory-x64/config/eva-ai.env"),
    )
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--limit", type=int, default=240)
    args = parser.parse_args()

    values = parse_env(args.env_file)
    dsn = str(values.get("EVA_DATABASE_DSN") or values.get("EVOSSEARCH_DATABASE_DSN") or "").strip()
    tenant_id = str(values.get("EVOSSEARCH_ARCHIVE_TENANT_ID") or "").strip()
    if not dsn or not tenant_id:
        raise SystemExit("Database DSN or archive tenant ID is missing from the EVA environment")
    try:
        import psycopg
    except ImportError as exc:
        raise SystemExit(f"Run with the installed EVA Python: {exc}") from exc

    query = """
        WITH ranked AS (
          SELECT
            channel_id,
            payload_json ->> 'batch_id' AS batch_id,
            recorded_at_ms,
            payload_json,
            row_number() OVER (
              PARTITION BY channel_id, payload_json ->> 'batch_id'
              ORDER BY recorded_at_ms ASC, id ASC
            ) AS row_number
          FROM archive.detections
          WHERE tenant_id = %s::uuid
            AND source = 'vlm_summary'
            AND created_at >= clock_timestamp() - (%s * interval '1 minute')
            AND COALESCE(payload_json ->> 'batch_id', '') <> ''
        )
        SELECT channel_id, batch_id, recorded_at_ms, payload_json
        FROM ranked
        WHERE row_number = 1
        ORDER BY recorded_at_ms DESC
        LIMIT %s
    """
    with psycopg.connect(dsn, connect_timeout=8) as connection:
        with connection.transaction():
            with connection.cursor() as cursor:
                cursor.execute("SELECT set_config('eva.tenant_id', %s, true)", (tenant_id,))
                cursor.execute(query, (tenant_id, max(1, args.minutes), max(1, args.limit)))
                rows = [
                    {
                        "channel_id": row[0],
                        "batch_id": row[1],
                        "recorded_at_ms": row[2],
                        "payload": row[3] if isinstance(row[3], dict) else {},
                    }
                    for row in cursor.fetchall()
                ]

    metrics = [extract_metrics(row) for row in rows]
    print("EVA LIVE L0 END-TO-END REPORT")
    print("=" * 72)
    print(f"Window: last {max(1, args.minutes)} minutes; completed durable batches: {len(metrics)}")
    if not metrics:
        print("No completed vlm_summary batches found in this window.")
        return 2

    fields = (
        ("first_frame_to_result_ms", "first source frame -> durable result"),
        ("last_frame_to_result_ms", "latest source frame -> durable result"),
        ("observation_ms", "evidence observation span"),
        ("last_frame_to_seal_ms", "latest frame -> batch seal"),
        ("executor_queue_ms", "EVA executor queue"),
        ("prepare_ms", "evidence compose/prepare"),
        ("model_call_ms", "LM call incl. admission/retry"),
        ("model_to_result_ms", "model complete -> durable result"),
    )
    print("\nAll channels")
    for key, label in fields:
        print(f"  {label:39} {stats(item.get(key) for item in metrics)}")

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in metrics:
        grouped[int(item["channel_id"])].append(item)
    for channel_id in sorted(grouped):
        channel = grouped[channel_id]
        completed = sorted(
            value
            for item in channel
            if (value := number(item.get("recorded_at_ms"))) is not None
        )
        cadence = [current - previous for previous, current in zip(completed, completed[1:])]
        print(f"\nChannel {channel_id}: {len(channel)} batches")
        print(f"  durable result cadence                {stats(cadence)}")
        print(f"  latest source frame -> durable result  {stats(item.get('last_frame_to_result_ms') for item in channel)}")
        print(f"  first source frame -> durable result   {stats(item.get('first_frame_to_result_ms') for item in channel)}")
        print(f"  LM call incl. admission/retry          {stats(item.get('model_call_ms') for item in channel)}")

    print("\nLatest 12 completed batches (seconds)")
    print("  channel frames/images  observe  latest->result  queue  prepare  admission  model  post")
    for item in metrics[:12]:
        def seconds(key: str) -> str:
            value = number(item.get(key))
            return "   — " if value is None else f"{value / 1000.0:5.2f}"

        print(
            f"  {item['channel_id']:>7} {item['frames']:>2}/{item['input_images']:<2}"
            f"       {seconds('observation_ms')}          {seconds('last_frame_to_result_ms')}"
            f"   {seconds('executor_queue_ms')}   {seconds('prepare_ms')}"
            f"      {seconds('admission_wait_ms')}  {seconds('model_call_ms')} {seconds('model_to_result_ms')}"
        )
    print("\n`result` means the L0 archive transaction was started and its evidence rows were durable.")
    print("No credentials, prompts, summaries or image data are printed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
