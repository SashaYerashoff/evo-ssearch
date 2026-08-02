#!/usr/bin/env python3
"""Seed small, deterministic archive fixtures for the live integration smoke.

DEV-ONLY. Run on the box, in the service's env (so it uses the same PostgreSQL
archive + CLIP embedder the running service reads fresh per query).

Seeds ONLY archive rows (read fresh from the DB → visible to a running service
without restart):
  - one "needle" detection with a distinct caption (for search_archive);
  - a small positive/negative frame set for one probe (for calibrate_probe_from_archive).

It does NOT seed summary history (that is in-memory, loaded at startup → would
need a restart). The prose-only / contamination behavior stays a golden test.

Idempotent within a UTC day: daily dedupe_keys keep the fixture inside a rolling
24-hour scenario instead of silently aging out, while repeated runs on the same
day do not duplicate. Use --dry-run to preview without writing. Verify the
imports below match this deployment before running (this script was authored
against archive_store.add_detections's record contract and the app's text
embedder).

Prints the env the live smoke expects:
    EVA_LIVE_CHANNEL_REF, EVA_LIVE_NEEDLE_QUERY, EVA_LIVE_PROBE_NAME
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SEED_TAG = "live-smoke-seed-v1"
NEEDLE_CAPTION = "a person lying motionless on the ground at night near a wall"
PROBE_POSITIVE = "a person making a thumbs up hand gesture"
PROBE_NEGATIVE = "a person typing on a keyboard with both hands"
PROBE_NAME = "smoke: thumbs up gesture"


def _now_ms() -> int:
    import time

    return int(time.time() * 1000)


def _vec(embed_text, caption: str) -> List[float]:
    """Use the SAME CLIP text embedder the search uses, stored as the frame vector.

    A text-embedding proxy (not a real image) is adequate for a *plumbing* smoke:
    it makes a same-concept query match deterministically. Not a CLIP-accuracy test.
    """
    arr = embed_text(caption)
    return [float(x) for x in list(arr)]


def _records(channel_id: int, embed_text) -> List[Dict[str, Any]]:
    base_ts = _now_ms() - 30 * 60 * 1000  # 30 min ago, inside typical windows
    # Keep repeat runs idempotent for the current UTC day while ensuring an old
    # seed does not silently age out of a "last 24 hours" acceptance scenario.
    seed_run = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    needle_vec = _vec(embed_text, NEEDLE_CAPTION)
    pos_vec = _vec(embed_text, PROBE_POSITIVE)
    neg_vec = _vec(embed_text, PROBE_NEGATIVE)
    out: List[Dict[str, Any]] = [
        {
            "dedupe_key": f"{SEED_TAG}:{seed_run}:needle:{channel_id}",
            "timestamp_ms": base_ts,
            "probe_id": f"{SEED_TAG}:needle",
            "probe_name": "smoke needle",
            "channel_id": channel_id,
            "severity": "info",
            "source": "vlm_summary",
            "clip_vec": needle_vec,
            "payload": {"caption": NEEDLE_CAPTION, "seed": SEED_TAG},
        }
    ]
    # a few positive + negative frames so calibration has a separable-ish set
    for i in range(4):
        out.append({
            "dedupe_key": f"{SEED_TAG}:{seed_run}:pos:{channel_id}:{i}",
            "timestamp_ms": base_ts + i * 1000,
            "probe_id": f"{SEED_TAG}:probe",
            "probe_name": PROBE_NAME,
            "channel_id": channel_id,
            "severity": "info",
            "source": "vlm_summary",
            "clip_vec": pos_vec,
            "payload": {"caption": PROBE_POSITIVE, "seed": SEED_TAG, "role": "positive"},
        })
    for i in range(8):
        out.append({
            "dedupe_key": f"{SEED_TAG}:{seed_run}:neg:{channel_id}:{i}",
            "timestamp_ms": base_ts + 100 + i * 1000,
            "probe_id": f"{SEED_TAG}:probe",
            "probe_name": PROBE_NAME,
            "channel_id": channel_id,
            "severity": "info",
            "source": "vlm_summary",
            "clip_vec": neg_vec,
            "payload": {"caption": PROBE_NEGATIVE, "seed": SEED_TAG, "role": "negative"},
        })
    return out


def _close_pool(owner: Any) -> None:
    try:
        pool = getattr(owner, "pool", None)
    except Exception:
        return
    try:
        close = getattr(pool, "close", None)
    except Exception:
        close = None
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _close_oldapp_pools(oldapp_module: Any) -> None:
    for value in vars(oldapp_module).values():
        _close_pool(value)
        try:
            close = getattr(value, "close", None)
            class_name = value.__class__.__name__
        except Exception:
            continue
        if callable(close) and class_name == "PsycopgPool":
            try:
                close()
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed live-smoke archive fixtures (DEV ONLY).")
    parser.add_argument("--channel-id", type=int, required=True, help="target channel id (also EVA_LIVE_CHANNEL_REF)")
    parser.add_argument("--dry-run", action="store_true", help="build records but do not write")
    args = parser.parse_args()

    # Import the app's already-built archive store + text embedder. Verify these
    # names against the deployment before running.
    try:
        import oldapp  # builds detections_store + embedders at import
    except Exception as exc:  # pragma: no cover
        print(f"failed to import oldapp (run inside the service env): {exc}", file=sys.stderr)
        return 2

    detections_store = getattr(oldapp, "detections_store", None)
    embed_text = getattr(oldapp, "get_text_embedding", None) or getattr(oldapp, "get_probe_text_embedding", None)
    if detections_store is None or embed_text is None:
        print("could not resolve detections_store / text embedder from oldapp", file=sys.stderr)
        return 2

    try:
        records = _records(args.channel_id, embed_text)
        print(f"prepared {len(records)} seed records for channel {args.channel_id} (tag={SEED_TAG})")
        if args.dry_run:
            print("dry-run: not writing")
        else:
            inserted = detections_store.add_detections(records)
            print(f"inserted={inserted}")

        print("\n# live-smoke env:")
        print(f"export EVA_LIVE_CHANNEL_REF={args.channel_id}")
        print(f'export EVA_LIVE_NEEDLE_QUERY="person lying on the ground at night"')
        print(f'export EVA_LIVE_PROBE_NAME="{PROBE_NAME}"')
        return 0
    finally:
        _close_pool(detections_store)
        _close_oldapp_pools(oldapp)


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Importing oldapp creates background DB pools; Python 3.14 may warn while
    # joining them during interpreter shutdown. We already close the pools we can
    # reach, then exit directly to keep this operator-facing helper quiet.
    import os

    os._exit(exit_code)
