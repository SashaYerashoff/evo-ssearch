#!/usr/bin/env python3
"""Probe VLM image perception and maintain a machine-readable health state."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm_vision_health import (  # noqa: E402
    VisionCanaryResult,
    is_timeout_error,
    probe_openai_liveness,
    probe_openai_workload,
    probe_vision,
    read_health_state,
    utc_now_iso,
    write_health_state,
)


def _env(*names: str) -> str:
    for name in names:
        value = str(os.getenv(name) or "").strip()
        if value:
            return value
    return ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check actual VLM image perception, not only HTTP liveness."
    )
    parser.add_argument(
        "--base-url",
        default=_env(
            "EVOSSEARCH_LM_PROFILE_VLM_BASE_URL",
            "EVOSSEARCH_LM_BASE_URL",
        ),
    )
    parser.add_argument(
        "--model",
        default=_env(
            "EVOSSEARCH_LM_PROFILE_VLM_MODEL",
            "EVOSSEARCH_LM_MODEL",
        ),
    )
    parser.add_argument(
        "--state-file",
        default=_env("EVOSSEARCH_LM_VISION_HEALTH_STATE_FILE"),
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--failure-threshold", type=int, default=2)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--fail-on-suspect",
        action="store_true",
        help="Return non-zero on the first mismatch (useful for startup gates).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not str(args.base_url).strip() or not str(args.model).strip():
        print("VLM vision watchdog requires a base URL and model.", file=sys.stderr)
        return 64
    if not str(args.state_file).strip():
        print("VLM vision watchdog requires --state-file.", file=sys.stderr)
        return 64

    previous = read_health_state(args.state_file)
    api_key = _env("EVOSSEARCH_LM_PROFILE_VLM_API_KEY", "EVOSSEARCH_LM_API_KEY")
    endpoint_liveness = probe_openai_liveness(
        args.base_url,
        args.model,
        api_key=api_key,
        timeout_sec=min(5.0, max(1.0, float(args.timeout) / 6.0)),
    )
    workload = probe_openai_workload(
        args.base_url,
        api_key=api_key,
        timeout_sec=min(3.0, max(1.0, float(args.timeout) / 10.0)),
    )
    preflight_busy = bool(
        endpoint_liveness.ok
        and (
            workload.busy
            or (not workload.known and is_timeout_error(workload.error))
        )
    )
    if preflight_busy:
        result = VisionCanaryResult(
            ok=False,
            expected="",
            observed="",
            latency_ms=workload.latency_ms,
            error="queue_busy",
        )
        busy = True
    elif not endpoint_liveness.ok:
        result = VisionCanaryResult(
            ok=False,
            expected="",
            observed="",
            latency_ms=endpoint_liveness.latency_ms,
            error=endpoint_liveness.error or "endpoint_not_live",
        )
        busy = False
    else:
        result = probe_vision(
            args.base_url,
            args.model,
            api_key=api_key,
            timeout_sec=max(1.0, float(args.timeout)),
            seed=args.seed,
        )
        busy = bool(
            not result.ok
            and is_timeout_error(result.error)
            and endpoint_liveness.ok
        )
    checked_at = utc_now_iso()
    previous_failures = int(previous.get("consecutive_failures") or 0)
    if result.ok:
        failures = 0
    elif busy:
        # The API is alive but the content canary waited behind real work.
        # Preserve sub-threshold semantic evidence without turning saturation
        # into a destructive restart. An already-actioned degraded counter is
        # cleared so a successful recovery cannot enter a restart loop.
        failures = (
            0
            if previous_failures >= max(1, int(args.failure_threshold))
            else previous_failures
        )
    else:
        failures = previous_failures + 1
    threshold = max(1, int(args.failure_threshold))
    if result.ok:
        status = "healthy"
    elif busy:
        status = "busy"
    elif failures >= threshold:
        status = "degraded"
    else:
        status = "suspect"
    state = {
        "version": 1,
        "ok": bool(result.ok or busy),
        "vision_ok": result.ok,
        "status": status,
        "checked_at": checked_at,
        "checked_at_epoch": __import__("time").time(),
        "last_success_at": checked_at if result.ok else previous.get("last_success_at"),
        "last_failure_at": (
            checked_at
            if not result.ok and not busy
            else previous.get("last_failure_at")
        ),
        "consecutive_failures": failures,
        "failure_threshold": threshold,
        "latency_ms": result.latency_ms,
        "expected": result.expected,
        "observed": result.observed,
        "error": result.error,
        "endpoint_liveness_ok": (
            endpoint_liveness.ok if endpoint_liveness is not None else None
        ),
        "endpoint_liveness_ms": (
            endpoint_liveness.latency_ms
            if endpoint_liveness is not None
            else None
        ),
        "endpoint_liveness_error": endpoint_liveness.error,
        "workload_known": workload.known,
        "workload_busy": workload.busy,
        "workload_processing": workload.processing,
        "workload_deferred": workload.deferred,
        "workload_latency_ms": workload.latency_ms,
        "workload_error": workload.error,
        "base_url": str(args.base_url).rstrip("/"),
        "model": str(args.model).strip(),
    }
    write_health_state(args.state_file, state)
    print(
        f"VLM vision {status}: expected={result.expected!r} "
        f"observed={result.observed!r} latency_ms={result.latency_ms}"
    )
    if result.ok or busy:
        return 0
    if args.fail_on_suspect or status == "degraded":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
