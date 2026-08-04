#!/usr/bin/env python3
"""Collect a secret-safe EVA appliance installation and runtime diagnosis."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from validate_appliance_config import parse_env_file, validate


DEFAULT_ENV = Path("/etc/eva-ai/eva-ai.env")
DEFAULT_STATE = Path("/var/lib/eva-ai-installer/install-state.json")
SERVICES = (
    "postgresql",
    "eva-vllm",
    "eva-deep-review",
    "eva-ai",
    "nginx",
)


def _vllm_tool_calling_contract(unit_text: str) -> dict[str, Any]:
    """Verify that the installed VLM unit accepts OpenAI native tool calls."""
    exec_lines = [
        line.strip()
        for line in str(unit_text or "").splitlines()
        if line.strip().startswith("ExecStart=") and "vllm" in line
    ]
    command = exec_lines[-1] if exec_lines else ""
    auto_choice = "--enable-auto-tool-choice" in command
    parser_match = command.split("--tool-call-parser", 1)
    parser = ""
    if len(parser_match) == 2:
        parser = parser_match[1].strip().split(None, 1)[0] if parser_match[1].strip() else ""
    return {
        "ok": bool(command and auto_choice and parser),
        "unit_exec_start_present": bool(command),
        "auto_tool_choice": auto_choice,
        "tool_call_parser": parser or None,
    }


def _command(argv: tuple[str, ...], timeout: int = 15) -> dict[str, Any]:
    try:
        result = subprocess.run(
            argv,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": type(exc).__name__}
    output = result.stdout.strip()
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        **({"output": output[:4000]} if output else {}),
    }


def _json_endpoint(url: str, timeout: int = 8) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            raw = response.read(1024 * 1024)
        payload = json.loads(raw.decode("utf-8"))
        result: dict[str, Any] = {
            "ok": 200 <= response.status < 400,
            "status_code": response.status,
        }
        if isinstance(payload, dict):
            if "status" in payload:
                result["status"] = payload["status"]
            if isinstance(payload.get("data"), list):
                result["models"] = [
                    str(item.get("id") or "")
                    for item in payload["data"]
                    if isinstance(item, dict)
                ]
            if isinstance(payload.get("checks"), dict):
                result["checks"] = {
                    name: {
                        key: check.get(key)
                        for key in ("ok", "status", "required")
                        if key in check
                    }
                    for name, check in payload["checks"].items()
                    if isinstance(check, dict)
                }
        return result
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read(1024 * 1024).decode("utf-8"))
        except Exception:
            payload = {}
        return {
            "ok": False,
            "status_code": exc.code,
            **(
                {"status": payload.get("status")}
                if isinstance(payload, dict) and payload.get("status")
                else {}
            ),
        }
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__}


def _service_status(service: str) -> dict[str, Any]:
    result = _command(
        (
            "systemctl",
            "show",
            f"{service}.service",
            "--property=LoadState,ActiveState,SubState,UnitFileState,Result",
        )
    )
    lines = str(result.pop("output", "")).splitlines()
    result["properties"] = dict(
        line.split("=", 1) for line in lines if "=" in line
    )
    result["ok"] = (
        result.get("ok", False)
        and result["properties"].get("ActiveState") == "active"
    )
    return result


def _vllm_runtime_contract(values: Mapping[str, str]) -> dict[str, Any]:
    base_url = str(values.get("EVOSSEARCH_LM_PROFILE_VLM_BASE_URL") or "").strip()
    try:
        hostname = str(urlsplit(base_url).hostname or "").lower()
    except ValueError:
        hostname = ""
    if hostname and hostname not in {"127.0.0.1", "localhost", "::1"}:
        return {
            "ok": True,
            "status": "external_profile",
            "local_unit_required": False,
        }
    result = _command(("systemctl", "cat", "eva-vllm.service", "--no-pager"))
    contract = _vllm_tool_calling_contract(str(result.get("output") or ""))
    if not result.get("ok"):
        contract["unit_read_error"] = result.get("error") or result.get("returncode")
        contract["ok"] = False
    return contract


def _intel_qsv_status() -> dict[str, Any]:
    devices = sorted(Path("/dev/dri").glob("renderD*"))
    intel_device: Path | None = None
    for device in devices:
        vendor_path = Path("/sys/class/drm") / device.name / "device" / "vendor"
        try:
            vendor = vendor_path.read_text(encoding="utf-8").strip().lower()
        except OSError:
            continue
        if vendor == "0x8086":
            intel_device = device
            break
    if intel_device is None:
        return {"ok": False, "status": "Intel DRM render node not found"}
    if not shutil.which("vainfo"):
        return {
            "ok": False,
            "device": str(intel_device),
            "status": "vainfo not installed",
        }
    result = _command(
        (
            "vainfo",
            "--display",
            "drm",
            "--device",
            str(intel_device),
        )
    )
    result["device"] = str(intel_device)
    result["status"] = "ready" if result.get("ok") else "driver initialization failed"
    return result


def _safe_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"present": False}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"present": True, "valid": False, "error": type(exc).__name__}
    target = payload.get("target") if isinstance(payload.get("target"), dict) else {}
    return {
        "present": True,
        "valid": True,
        "version": payload.get("version"),
        "bundle_id": payload.get("bundle_id"),
        "attempts": payload.get("attempts"),
        "status": payload.get("status"),
        "last_completed_phase": payload.get("last_completed_phase"),
        "failed_phase": payload.get("failed_phase"),
        "target": {
            key: target.get(key)
            for key in (
                "install_root",
                "data_root",
                "config_root",
                "local_vlm",
                "local_deep",
                "timezone",
            )
        },
    }


def collect(env_file: Path, state_file: Path) -> dict[str, Any]:
    values: dict[str, str] = {}
    config_read_error = ""
    if env_file.is_file():
        try:
            values = parse_env_file(env_file)
        except OSError as exc:
            config_read_error = type(exc).__name__
    config_errors = (
        validate(values, check_files=True)
        if values
        else [
            (
                f"configuration file cannot be read: {config_read_error}"
                if config_read_error
                else f"configuration file is missing: {env_file}"
            )
        ]
    )
    mode = None
    if env_file.exists():
        mode = oct(stat.S_IMODE(env_file.stat().st_mode))

    endpoints: dict[str, Any] = {
        "eva_health": _json_endpoint("http://127.0.0.1:5000/health"),
        "eva_ready": _json_endpoint("http://127.0.0.1:5000/ready"),
    }
    vlm_url = str(values.get("EVOSSEARCH_LM_PROFILE_VLM_BASE_URL") or "").rstrip("/")
    if vlm_url:
        endpoints["vlm_models"] = _json_endpoint(vlm_url + "/models")
    deep_url = str(
        values.get("EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_BASE_URL") or ""
    ).rstrip("/")
    if deep_url:
        endpoints["deep_models"] = _json_endpoint(deep_url + "/models")

    schema = {"ok": False, "status": "psql unavailable"}
    if shutil.which("psql"):
        argv = (
            "runuser",
            "-u",
            "postgres",
            "--",
            "psql",
            "-d",
            "eva",
            "-Atqc",
            "SELECT version_num FROM alembic_version LIMIT 1",
        )
        if os.geteuid() == 0:
            schema = _command(argv)
        elif shutil.which("sudo"):
            schema = _command(("sudo", "-n", *argv))

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "os_release": (
                Path("/etc/os-release").read_text(encoding="utf-8")[:4000]
                if Path("/etc/os-release").is_file()
                else ""
            ),
            "kernel": _command(("uname", "-a")),
            "disk": _command(("df", "-h", "/opt/eva-ai", "/var/lib/eva-ai")),
            "gpu": _command(
                (
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version",
                    "--format=csv,noheader",
                )
            )
            if shutil.which("nvidia-smi")
            else {"ok": False, "error": "nvidia-smi not found"},
            "intel_qsv": _intel_qsv_status(),
        },
        "installer": _safe_state(state_file),
        "configuration": {
            "present": env_file.is_file(),
            "mode": mode,
            "valid": not config_errors,
            "errors": config_errors,
            "auth_enabled": str(values.get("EVOSSEARCH_AUTH_ENABLED") or "").lower(),
            "tenant_ids_present": {
                key: bool(values.get(key))
                for key in (
                    "EVOSSEARCH_AUTH_TENANT_ID",
                    "EVOSSEARCH_ARCHIVE_TENANT_ID",
                    "EVOSSEARCH_INFERENCE_QUEUE_TENANT_ID",
                )
            },
            "database_dsns_present": {
                key: bool(values.get(key))
                for key in (
                    "EVA_DATABASE_DSN",
                    "EVA_AUDIT_DATABASE_DSN",
                    "EVA_WORKER_DATABASE_DSN",
                    "EVA_MIGRATION_DATABASE_DSN",
                )
            },
        },
        "schema": schema,
        "services": {service: _service_status(service) for service in SERVICES},
        "runtime_contracts": {
            "vlm_native_tool_calling": _vllm_runtime_contract(values),
        },
        "endpoints": endpoints,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = collect(args.env_file, args.state_file)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Diagnostic report written to {args.output}")
    else:
        print(rendered, end="")
    core_ready = bool(
        report["configuration"]["valid"]
        and report["endpoints"]["eva_ready"].get("ok")
        and report["runtime_contracts"]["vlm_native_tool_calling"].get("ok")
    )
    return 0 if core_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
