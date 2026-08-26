#!/usr/bin/env python3
"""Print a secret-free trace of recent durable L0 batches."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import psycopg


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


def bounded(value: object, limit: int = 1800) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", type=int, default=112)
    parser.add_argument("--minutes", type=int, default=15)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path("/mnt/eva-llamacpp-lab/factory-x64/config/eva-ai.env"),
    )
    args = parser.parse_args()
    values = parse_env(args.env_file)
    dsn = values.get("EVA_DATABASE_DSN") or values.get("EVOSSEARCH_DATABASE_DSN")
    tenant = values.get("EVOSSEARCH_ARCHIVE_TENANT_ID")
    if not dsn or not tenant:
        raise SystemExit("Database DSN or archive tenant ID is missing")
    query = """
        WITH ranked AS (
          SELECT recorded_at_ms, payload_json,
                 row_number() OVER (
                   PARTITION BY payload_json ->> 'batch_id'
                   ORDER BY recorded_at_ms ASC, id ASC
                 ) AS row_number
          FROM archive.detections
          WHERE tenant_id = %s::uuid
            AND channel_id = %s
            AND source = 'vlm_summary'
            AND created_at >= clock_timestamp() - (%s * interval '1 minute')
        )
        SELECT recorded_at_ms, payload_json
        FROM ranked
        WHERE row_number = 1
        ORDER BY recorded_at_ms DESC
        LIMIT %s
    """
    with psycopg.connect(dsn, connect_timeout=8) as connection:
        with connection.transaction():
            with connection.cursor() as cursor:
                cursor.execute("SELECT set_config('eva.tenant_id', %s, true)", (tenant,))
                cursor.execute(
                    query,
                    (tenant, args.channel, max(1, args.minutes), max(1, args.limit)),
                )
                rows = cursor.fetchall()

    print(f"EVA RECENT L0 TRACE — channel {args.channel}; batches {len(rows)}")
    for recorded_at_ms, raw_payload in rows:
        payload = raw_payload if isinstance(raw_payload, dict) else {}
        stamp = datetime.fromtimestamp(recorded_at_ms / 1000.0).isoformat(timespec="milliseconds")
        state = payload.get("batch_state") if isinstance(payload.get("batch_state"), dict) else {}
        selection = payload.get("frame_selection") if isinstance(payload.get("frame_selection"), dict) else {}
        stats = payload.get("llm_input_stats") if isinstance(payload.get("llm_input_stats"), dict) else {}
        signal = payload.get("vector_signal") if isinstance(payload.get("vector_signal"), dict) else {}
        probe_signals = signal.get("clip_probe_signals")
        capture_attention = signal.get("capture_attention")
        safe_signal = {}
        if isinstance(probe_signals, list):
            safe_signal["clip_probe_signals"] = probe_signals[:8]
        if isinstance(capture_attention, dict):
            safe_signal["capture_attention"] = capture_attention
        print("\n" + "=" * 72)
        print(
            f"{stamp} batch={payload.get('batch_id')} "
            f"frames={payload.get('frame_count')}/{stats.get('image_parts')} "
            f"source={selection.get('source_frame_count')} "
            f"alerts={payload.get('alert_total')} bookmarks={payload.get('bookmarks_sent')}"
        )
        print("SUMMARY")
        print(bounded(payload.get("summary")))
        print("BATCH_STATE")
        print(json.dumps(state, ensure_ascii=False, sort_keys=True))
        print("FRAME_SELECTION")
        print(json.dumps(selection, ensure_ascii=False, sort_keys=True))
        print("ATTENTION_SIGNALS")
        print(json.dumps(safe_signal, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
