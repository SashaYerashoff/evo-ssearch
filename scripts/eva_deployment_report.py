#!/usr/bin/env python3
"""Write a secret-safe, human-readable EVA AI deployment acceptance report."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_SCHEMA = "20260805_0013"
DEFAULT_ENV = Path("/etc/eva-ai/eva-ai.env")
DEFAULT_APP = Path("/opt/eva-ai/evo-ssearch")


def parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("export "):
            text = text[7:].lstrip()
        key, separator, value = text.partition("=")
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


def _command(argv: Sequence[str], *, env: Mapping[str, str] | None = None, timeout: int = 15) -> dict[str, Any]:
    try:
        result = subprocess.run(
            list(argv),
            env=dict(env) if env is not None else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": type(exc).__name__}
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip()[:8000],
        "stderr": result.stderr.strip()[:1000],
    }


def _http_json(url: str, timeout: int = 8) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            raw = response.read(2 * 1024 * 1024)
            return {
                "ok": 200 <= response.status < 400,
                "status_code": response.status,
                "payload": json.loads(raw.decode("utf-8")),
            }
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read(2 * 1024 * 1024).decode("utf-8"))
        except Exception:
            payload = {}
        return {"ok": False, "status_code": exc.code, "payload": payload}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__, "payload": {}}


def _ui_status(base_url: str) -> dict[str, Any]:
    try:
        request = urllib.request.Request(base_url.rstrip("/") + "/", method="GET")
        with urllib.request.urlopen(request, timeout=8) as response:
            body = response.read(1024 * 1024).decode("utf-8", errors="replace")
            mode = str(response.headers.get("X-EVA-UI") or "").strip().lower()
            react = mode == "react" and "<div id=\"root\"></div>" in body
            return {
                "ok": react,
                "status_code": response.status,
                "mode": mode or "unknown",
                "react_root": "<div id=\"root\"></div>" in body,
            }
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__, "mode": "unavailable"}


def _service_status(name: str) -> dict[str, Any]:
    result = _command(
        (
            "systemctl",
            "show",
            f"{name}.service",
            "--property=LoadState,ActiveState,SubState,UnitFileState,Result",
        )
    )
    properties = dict(
        line.split("=", 1)
        for line in str(result.get("stdout") or "").splitlines()
        if "=" in line
    )
    loaded = properties.get("LoadState") == "loaded"
    return {
        "present": loaded,
        "ok": bool(loaded and properties.get("ActiveState") == "active"),
        "properties": properties,
    }


def _db_dsn(values: Mapping[str, str]) -> str:
    return str(
        os.environ.get("EVA_INSTALL_MIGRATION_DSN")
        or values.get("EVA_MIGRATION_DATABASE_DSN")
        or values.get("EVA_DATABASE_DSN")
        or values.get("EVOSSEARCH_DATABASE_DSN")
        or ""
    ).strip()


def _psql(dsn: str, sql: str) -> dict[str, Any]:
    if not dsn:
        return {"ok": False, "error": "database DSN unavailable"}
    env = dict(os.environ)
    env["PGDATABASE"] = dsn
    env["PGCONNECT_TIMEOUT"] = "8"
    result = _command(("psql", "-X", "-A", "-t", "-F", "\t", "-c", sql), env=env, timeout=20)
    if not result.get("ok"):
        result["stdout"] = ""
        result["stderr"] = str(result.get("stderr") or "").splitlines()[-1:] or []
    return result


def _schema_status(values: Mapping[str, str]) -> dict[str, Any]:
    result = _psql(_db_dsn(values), "SELECT version_num FROM alembic_version LIMIT 1")
    revision = str(result.get("stdout") or "").strip().splitlines()[-1:] or [""]
    current = revision[0].strip()
    return {
        "ok": bool(result.get("ok") and current == EXPECTED_SCHEMA),
        "current_revision": current or None,
        "expected_revision": EXPECTED_SCHEMA,
        **({"error": result.get("error") or result.get("stderr")} if not result.get("ok") else {}),
    }


def _summary_activity(values: Mapping[str, str], *, minutes: int = 15) -> dict[str, Any]:
    tenant_raw = str(values.get("EVOSSEARCH_ARCHIVE_TENANT_ID") or values.get("EVOSSEARCH_AUTH_TENANT_ID") or "").strip()
    try:
        tenant = str(uuid.UUID(tenant_raw))
    except ValueError:
        return {"ok": False, "error": "archive tenant UUID unavailable", "window_minutes": minutes}
    threshold_ms = max(1, int(minutes)) * 60 * 1000
    sql = f"""
WITH tenant_context AS (
  SELECT set_config('eva.tenant_id', '{tenant}', false)
)
SELECT
  COUNT(*),
  COUNT(DISTINCT channel_id),
  COALESCE(MAX(event_timestamp_ms), 0),
  COALESCE((EXTRACT(EPOCH FROM clock_timestamp()) * 1000)::bigint - MAX(event_timestamp_ms), 0)
FROM archive.detections, tenant_context
WHERE tenant_id = '{tenant}'::uuid
  AND source = 'vlm_summary'
  AND event_timestamp_ms >= (EXTRACT(EPOCH FROM clock_timestamp()) * 1000)::bigint - {threshold_ms}
""".strip()
    result = _psql(_db_dsn(values), sql)
    parts = str(result.get("stdout") or "").strip().split("\t")
    if not result.get("ok") or len(parts) != 4:
        return {
            "ok": False,
            "error": result.get("error") or result.get("stderr") or "query failed",
            "window_minutes": minutes,
        }
    try:
        records, channels, latest_ms, age_ms = (int(float(value or 0)) for value in parts)
    except ValueError:
        return {"ok": False, "error": "invalid activity query result", "window_minutes": minutes}
    return {
        "ok": True,
        "window_minutes": minutes,
        "records": records,
        "channels": channels,
        "latest_event_timestamp_ms": latest_ms,
        "latest_age_sec": round(max(0, age_ms) / 1000.0, 1) if latest_ms else None,
    }


def _gpu_status() -> dict[str, Any]:
    result = _command(
        (
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        )
    )
    rows = [row.strip() for row in str(result.get("stdout") or "").splitlines() if row.strip()]
    return {"ok": bool(result.get("ok") and rows), "devices": rows}


def _profile_ids(values: Mapping[str, str]) -> list[str]:
    raw = str(values.get("EVOSSEARCH_LM_PROFILES") or "agent,vlm")
    return list(dict.fromkeys(item.strip() for item in raw.split(",") if item.strip()))


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(str(value or "0").strip()))
    except (TypeError, ValueError):
        return 0


def _profile_status(values: Mapping[str, str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for profile_id in _profile_ids(values):
        env_id = re.sub(r"[^A-Za-z0-9]+", "_", profile_id).strip("_").upper()
        base_url = str(
            values.get(f"EVOSSEARCH_LM_PROFILE_{env_id}_BASE_URL")
            or values.get("EVOSSEARCH_LM_BASE_URL")
            or ""
        ).strip().rstrip("/")
        model = str(
            values.get(f"EVOSSEARCH_LM_PROFILE_{env_id}_MODEL")
            or values.get("EVOSSEARCH_LM_MODEL")
            or ""
        ).strip()
        api_key = str(
            values.get(f"EVOSSEARCH_LM_PROFILE_{env_id}_API_KEY")
            or values.get("EVOSSEARCH_LM_API_KEY")
            or ""
        ).strip()
        if not base_url:
            output.append({"id": profile_id, "ok": False, "model": model, "status": "not_configured"})
            continue
        request = urllib.request.Request(base_url + "/models", headers={"Accept": "application/json"})
        if api_key:
            request.add_header("Authorization", f"Bearer {api_key}")
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                payload = json.loads(response.read(1024 * 1024).decode("utf-8"))
            served = [
                str(row.get("id") or row.get("model") or "")
                for row in payload.get("data", [])
                if isinstance(row, dict)
            ] if isinstance(payload, dict) else []
            output.append(
                {
                    "id": profile_id,
                    "ok": bool(200 <= response.status < 400 and served and (not model or model in served)),
                    "status": "reachable",
                    "base_url": base_url,
                    "model": model,
                    "kind": str(values.get(f"EVOSSEARCH_LM_PROFILE_{env_id}_KIND") or "").strip(),
                    "max_inflight": _nonnegative_int(values.get(f"EVOSSEARCH_LM_PROFILE_{env_id}_MAX_INFLIGHT")),
                    "served_models": served,
                }
            )
        except Exception as exc:
            output.append(
                {
                    "id": profile_id,
                    "ok": False,
                    "status": "unavailable",
                    "base_url": base_url,
                    "model": model,
                    "kind": str(values.get(f"EVOSSEARCH_LM_PROFILE_{env_id}_KIND") or "").strip(),
                    "max_inflight": _nonnegative_int(values.get(f"EVOSSEARCH_LM_PROFILE_{env_id}_MAX_INFLIGHT")),
                    "error": type(exc).__name__,
                }
            )
    return output


def collect(
    *,
    env_file: Path,
    app_dir: Path,
    service: str,
    base_url: str,
    activity_minutes: int = 15,
) -> dict[str, Any]:
    values = parse_env(env_file)
    health = _http_json(base_url.rstrip("/") + "/health")
    ready = _http_json(base_url.rstrip("/") + "/ready")
    ready_payload = ready.get("payload") if isinstance(ready.get("payload"), dict) else {}
    checks = ready_payload.get("checks") if isinstance(ready_payload.get("checks"), dict) else {}
    version = ""
    version_path = app_dir / "VERSION"
    if version_path.is_file():
        version = version_path.read_text(encoding="utf-8").strip()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "hostname": platform.node(),
            "kernel": platform.release(),
            "architecture": platform.machine(),
            "cpu": platform.processor(),
            "memory": _command(("free", "-h")),
            "gpu": _gpu_status(),
        },
        "eva": {
            "version": version,
            "service": _service_status(service),
            "health": health,
            "ready": ready,
            "ui": _ui_status(base_url),
        },
        "database": _schema_status(values),
        "luxriot": {
            "configured": bool(values.get("EVOSSEARCH_LUXRIOT_BASE_URL") and values.get("EVOSSEARCH_LUXRIOT_USERNAME")),
            "ready_check": checks.get("luxriot") if isinstance(checks.get("luxriot"), dict) else {},
        },
        "inference": {
            "ready_check": checks.get("lm_profiles") if isinstance(checks.get("lm_profiles"), dict) else {},
            "profiles": _profile_status(values),
        },
        "semantic": {
            "ready_check": checks.get("embedder") if isinstance(checks.get("embedder"), dict) else {},
        },
        "streams": _summary_activity(values, minutes=activity_minutes),
    }


def _baseline_expected(baseline: Mapping[str, Any] | None) -> tuple[bool, int]:
    if not isinstance(baseline, Mapping):
        return False, 0
    streams = baseline.get("streams") if isinstance(baseline.get("streams"), Mapping) else {}
    channels = int(streams.get("channels") or 0)
    latest = int(streams.get("latest_event_timestamp_ms") or 0)
    return channels > 0 and latest > 0, latest


def _wait_for_stream_progress(
    report: dict[str, Any],
    *,
    baseline: Mapping[str, Any] | None,
    deadline_seconds: int,
    collect_args: Mapping[str, Any],
) -> dict[str, Any]:
    expected, previous_latest = _baseline_expected(baseline)
    if not expected or deadline_seconds <= 0:
        return report
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        streams = report.get("streams") if isinstance(report.get("streams"), dict) else {}
        if int(streams.get("latest_event_timestamp_ms") or 0) > previous_latest:
            return report
        time.sleep(5)
        report = collect(**collect_args)
    return report


def evaluate(report: Mapping[str, Any], baseline: Mapping[str, Any] | None = None) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    eva = report.get("eva") if isinstance(report.get("eva"), Mapping) else {}
    service = eva.get("service") if isinstance(eva.get("service"), Mapping) else {}
    health = eva.get("health") if isinstance(eva.get("health"), Mapping) else {}
    ready = eva.get("ready") if isinstance(eva.get("ready"), Mapping) else {}
    ui = eva.get("ui") if isinstance(eva.get("ui"), Mapping) else {}
    database = report.get("database") if isinstance(report.get("database"), Mapping) else {}
    luxriot = report.get("luxriot") if isinstance(report.get("luxriot"), Mapping) else {}
    luxriot_ready = luxriot.get("ready_check") if isinstance(luxriot.get("ready_check"), Mapping) else {}
    inference = report.get("inference") if isinstance(report.get("inference"), Mapping) else {}
    profiles = inference.get("profiles") if isinstance(inference.get("profiles"), list) else []
    streams = report.get("streams") if isinstance(report.get("streams"), Mapping) else {}
    semantic = report.get("semantic") if isinstance(report.get("semantic"), Mapping) else {}
    semantic_ready = semantic.get("ready_check") if isinstance(semantic.get("ready_check"), Mapping) else {}

    if not service.get("ok"):
        failures.append("eva-ai.service is not active")
    if not health.get("ok"):
        failures.append("EVA /health is unavailable")
    if not ready.get("ok"):
        failures.append("EVA /ready is not ready")
    if not ui.get("ok"):
        failures.append("React UI is not the active root interface")
    if not database.get("ok"):
        failures.append(
            f"database schema is {database.get('current_revision') or 'unknown'}, expected {EXPECTED_SCHEMA}"
        )
    if bool(luxriot.get("configured")) and not luxriot_ready.get("ok"):
        failures.append("Luxriot Evo is configured but not reachable through EVA")
    unavailable_profiles = [str(row.get("id")) for row in profiles if isinstance(row, Mapping) and not row.get("ok")]
    if unavailable_profiles:
        failures.append("inference profiles unavailable: " + ", ".join(unavailable_profiles))

    expected_streams, previous_latest = _baseline_expected(baseline)
    if expected_streams:
        if int(streams.get("latest_event_timestamp_ms") or 0) <= previous_latest:
            failures.append("previously active video-summary streams did not produce a new post-update record")
    elif not streams.get("ok"):
        warnings.append("stream activity could not be measured")
    elif int(streams.get("channels") or 0) == 0:
        warnings.append("no active pre-update stream baseline; configure/start streams before acceptance")
    semantic_status = str(semantic_ready.get("status") or "unknown")
    if semantic_status == "not_loaded":
        warnings.append("semantic backend is idle and will load on the first enabled semantic probe")
    elif semantic_ready and not semantic_ready.get("ok"):
        failures.append(f"semantic backend is unhealthy: {semantic_status}")
    return {
        "status": "FAIL" if failures else ("WARN" if warnings else "PASS"),
        "failures": failures,
        "warnings": warnings,
    }


def render_text(report: Mapping[str, Any], assessment: Mapping[str, Any]) -> str:
    host = report.get("host") if isinstance(report.get("host"), Mapping) else {}
    eva = report.get("eva") if isinstance(report.get("eva"), Mapping) else {}
    ui = eva.get("ui") if isinstance(eva.get("ui"), Mapping) else {}
    database = report.get("database") if isinstance(report.get("database"), Mapping) else {}
    luxriot = report.get("luxriot") if isinstance(report.get("luxriot"), Mapping) else {}
    luxriot_ready = luxriot.get("ready_check") if isinstance(luxriot.get("ready_check"), Mapping) else {}
    streams = report.get("streams") if isinstance(report.get("streams"), Mapping) else {}
    inference = report.get("inference") if isinstance(report.get("inference"), Mapping) else {}
    profiles = inference.get("profiles") if isinstance(inference.get("profiles"), list) else []
    semantic = report.get("semantic") if isinstance(report.get("semantic"), Mapping) else {}
    semantic_ready = semantic.get("ready_check") if isinstance(semantic.get("ready_check"), Mapping) else {}
    gpu = host.get("gpu") if isinstance(host.get("gpu"), Mapping) else {}
    rows = [
        "EVA AI DEPLOYMENT REPORT",
        "=" * 60,
        f"RESULT: {assessment.get('status')}",
        f"Generated: {report.get('generated_at')}",
        f"Host: {host.get('hostname') or 'unknown'}",
        f"Kernel version: {host.get('kernel') or 'unknown'}",
        f"GPU: {', '.join(gpu.get('devices') or []) or 'not detected'}",
        f"EVA version: {eva.get('version') or 'unknown'}",
        f"EVA service: {'ACTIVE' if (eva.get('service') or {}).get('ok') else 'FAILED'}",
        f"EVA readiness: {'READY' if (eva.get('ready') or {}).get('ok') else 'NOT READY'}",
        f"UI updated and running: {'YES (React)' if ui.get('ok') else 'NO'}",
        (
            "Migrations successful: YES "
            f"({database.get('current_revision') or 'unknown'})"
            if database.get("ok")
            else f"Migrations successful: NO ({database.get('current_revision') or 'unknown'})"
        ),
        f"Luxriot Evo: {'REACHABLE' if luxriot_ready.get('ok') else ('NOT CONFIGURED' if not luxriot.get('configured') else 'UNAVAILABLE')}",
        "Inference: " + ", ".join(
            f"{row.get('id')}={'READY' if row.get('ok') else 'FAILED'} "
            f"({row.get('base_url') or 'no URL'} · {row.get('model') or 'no model'} · max {row.get('max_inflight') or '?'})"
            for row in profiles if isinstance(row, Mapping)
        ),
        (
            "Semantic: READY"
            if semantic_ready.get("ok")
            else "Semantic: IDLE — loads on first enabled semantic probe"
            if semantic_ready.get("status") == "not_loaded"
            else f"Semantic: {str(semantic_ready.get('status') or 'UNKNOWN').upper()}"
        ),
        (
            "Streams: WORKING — "
            f"{streams.get('channels', 0)} channel(s), {streams.get('records', 0)} summary frame(s) "
            f"in the last {streams.get('window_minutes', 15)} minutes"
            if int(streams.get("channels") or 0) > 0
            else "Streams: no recent video-summary records"
        ),
    ]
    for failure in assessment.get("failures") or []:
        rows.append(f"FAIL: {failure}")
    for warning in assessment.get("warnings") or []:
        rows.append(f"WARN: {warning}")
    rows.extend(("=" * 60, "Secrets are intentionally omitted from this report."))
    return "\n".join(rows) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--app-dir", type=Path, default=DEFAULT_APP)
    parser.add_argument("--service", default="eva-ai")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--wait-streams", type=int, default=0)
    parser.add_argument("--activity-minutes", type=int, default=15)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--text-output", type=Path)
    args = parser.parse_args(argv)
    baseline: dict[str, Any] | None = None
    if args.baseline and args.baseline.is_file():
        try:
            baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            baseline = None
    collect_args = {
        "env_file": args.env_file,
        "app_dir": args.app_dir,
        "service": args.service,
        "base_url": args.base_url,
        "activity_minutes": max(1, args.activity_minutes),
    }
    report = collect(**collect_args)
    report = _wait_for_stream_progress(
        report,
        baseline=baseline,
        deadline_seconds=max(0, args.wait_streams),
        collect_args=collect_args,
    )
    assessment = evaluate(report, baseline)
    report["assessment"] = assessment
    rendered_json = json.dumps(report, indent=2, sort_keys=True) + "\n"
    rendered_text = render_text(report, assessment)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered_json, encoding="utf-8")
    if args.text_output:
        args.text_output.parent.mkdir(parents=True, exist_ok=True)
        args.text_output.write_text(rendered_text, encoding="utf-8")
    print(rendered_text, end="")
    return 1 if assessment["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
