#!/usr/bin/env python3
"""Fail-closed validation for an EVA AI appliance runtime environment.

This command intentionally performs no migrations and prints no configuration
values.  systemd runs it as ExecStartPre so a partially generated environment
cannot start an apparently healthy but unusable EVA process.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit


TENANT_KEYS = (
    "EVOSSEARCH_AUTH_TENANT_ID",
    "EVOSSEARCH_ARCHIVE_TENANT_ID",
    "EVOSSEARCH_INFERENCE_QUEUE_TENANT_ID",
)
REQUIRED_KEYS = (
    "EVA_DATABASE_DSN",
    "EVA_AUDIT_DATABASE_DSN",
    "EVA_WORKER_DATABASE_DSN",
    "EVA_MIGRATION_DATABASE_DSN",
    "EVOSSEARCH_LUXRIOT_BASE_URL",
    "EVOSSEARCH_LUXRIOT_USERNAME",
    "EVOSSEARCH_LUXRIOT_PASSWORD",
    "EVOSSEARCH_LM_PROFILE_AGENT_BASE_URL",
    "EVOSSEARCH_LM_PROFILE_AGENT_MODEL",
    "EVOSSEARCH_LM_PROFILE_VLM_BASE_URL",
    "EVOSSEARCH_LM_PROFILE_VLM_MODEL",
    *TENANT_KEYS,
)
TRUE_VALUES = {"1", "true", "yes", "on"}


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, raw = stripped.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key.strip()):
            continue
        value = raw.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _valid_http_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname)


def _valid_postgres_dsn(value: str) -> bool:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return parsed.scheme in {"postgres", "postgresql"} and bool(
        parsed.hostname or parsed.query
    )


def validate(values: Mapping[str, str], *, check_files: bool = True) -> list[str]:
    errors: list[str] = []
    missing = [key for key in REQUIRED_KEYS if not str(values.get(key) or "").strip()]
    if missing:
        errors.append("missing required settings: " + ", ".join(sorted(missing)))

    if str(values.get("EVOSSEARCH_AUTH_ENABLED") or "").strip().lower() not in TRUE_VALUES:
        errors.append("named-user authentication must be enabled")
    if str(values.get("EVOSSEARCH_SECURE_DEPLOYMENT_REQUIRED") or "").strip().lower() not in TRUE_VALUES:
        errors.append("secure deployment mode must be enabled")
    if str(values.get("EVOSSEARCH_AUTH_COOKIE_SECURE") or "").strip().lower() not in TRUE_VALUES:
        errors.append("secure authentication cookies must be enabled behind appliance TLS")
    if str(values.get("EVOSSEARCH_ADMIN_TOKEN") or "").strip():
        errors.append("legacy EVOSSEARCH_ADMIN_TOKEN must not be present")

    tenant_ids: set[str] = set()
    for key in TENANT_KEYS:
        raw = str(values.get(key) or "").strip()
        if not raw:
            continue
        try:
            tenant_ids.add(str(uuid.UUID(raw)))
        except ValueError:
            errors.append(f"{key} must be a UUID")
    if len(tenant_ids) > 1:
        errors.append("auth, archive and inference queue tenant IDs must match")

    for key in (
        "EVOSSEARCH_LUXRIOT_BASE_URL",
        "EVOSSEARCH_LM_PROFILE_AGENT_BASE_URL",
        "EVOSSEARCH_LM_PROFILE_VLM_BASE_URL",
    ):
        raw = str(values.get(key) or "").strip()
        if raw and not _valid_http_url(raw):
            errors.append(f"{key} must be an HTTP(S) URL")

    for key in (
        "EVA_DATABASE_DSN",
        "EVA_AUDIT_DATABASE_DSN",
        "EVA_WORKER_DATABASE_DSN",
        "EVA_MIGRATION_DATABASE_DSN",
    ):
        raw = str(values.get(key) or "").strip()
        if raw and not _valid_postgres_dsn(raw):
            errors.append(f"{key} must be a PostgreSQL URI")

    runtime_dsn = str(values.get("EVA_DATABASE_DSN") or "").strip()
    migration_dsn = str(values.get("EVA_MIGRATION_DATABASE_DSN") or "").strip()
    if runtime_dsn and runtime_dsn == migration_dsn:
        errors.append("migration and runtime DSNs must use distinct roles")

    if check_files:
        clip_model = str(values.get("EVOSSEARCH_CLIP_MODEL") or "").strip()
        if "siglip2" in clip_model.lower():
            model_cache = str(values.get("EVOSSEARCH_MODEL_CACHE_DIR") or "").strip()
            revision = str(values.get("EVOSSEARCH_CLIP_MODEL_REVISION") or "").strip()
            snapshot = (
                Path(model_cache)
                / "models--google--siglip2-base-patch16-224"
                / "snapshots"
                / revision
            ) if model_cache and revision else None
            required = (
                "config.json",
                "model.safetensors",
                "preprocessor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
            )
            if snapshot is None or any(
                not (snapshot / filename).is_file()
                for filename in required
            ):
                errors.append(
                    "SigLIP2 model, processor and tokenizer are missing from "
                    "the configured offline Hugging Face cache/revision"
                )
        else:
            clip_cache = str(values.get("EVOSSEARCH_OPENAI_CLIP_CACHE_DIR") or "").strip()
            if clip_cache and not (Path(clip_cache) / "ViT-B-32.pt").is_file():
                errors.append("CLIP ViT-B-32.pt is missing from the configured offline cache")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path("/etc/eva-ai/eva-ai.env"),
    )
    parser.add_argument(
        "--from-environment",
        action="store_true",
        help="Validate the environment inherited from systemd.",
    )
    parser.add_argument("--no-file-checks", action="store_true")
    args = parser.parse_args(argv)

    if args.from_environment:
        values = dict(os.environ)
    else:
        if not args.env_file.is_file():
            print(f"Configuration file is missing: {args.env_file}", file=sys.stderr)
            return 2
        values = parse_env_file(args.env_file)

    errors = validate(values, check_files=not args.no_file_checks)
    if errors:
        print("EVA appliance configuration is not ready:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 2
    print("EVA appliance configuration preflight OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
