#!/usr/bin/env python3
"""Dry-run-first, offline EVA AI installer.

The installer deliberately orchestrates the existing field-proven mechanisms:
``preflight_patch.sh`` for the baseline, ``install_patch.sh`` for backup and
copy, Alembic for transactional schema changes, ``verify_patch.sh`` for health,
and ``rollback.sh`` for the operator handoff.

No dependency command in this file may access a package index. A fresh install
therefore requires a bundle wheelhouse; an upgrade may reuse its existing venv.
"""

from __future__ import annotations

import argparse
import fcntl
import getpass
import grp
import os
import pwd
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping, Sequence
from urllib.parse import urlsplit


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent


def _expected_version() -> str:
    """The bundled VERSION file is authoritative; the constant is a fallback.

    The installer ships inside the source tree it installs, so a hard-coded
    version string silently rots on every release bump.
    """

    try:
        text = (REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        text = ""
    return text or "β 0.8.5"


EXPECTED_VERSION = _expected_version()
EXPECTED_SCHEMA = "20260726_0009"
DEFAULT_APP_DIR = Path("/opt/eva-ai/evo-ssearch")
DEFAULT_ENV_FILE = Path("/etc/eva-ai/eva-ai.env")
DEFAULT_BACKUP_ROOT = Path("/var/backups/eva-ai")
DEFAULT_UNIT_FILE = Path("/etc/systemd/system/eva-ai.service")
DEFAULT_LOCK_FILE = Path("/run/lock/eva-ai-083-installer.lock")

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SECRET_MARKERS = ("PASSWORD", "SECRET", "TOKEN", "API_KEY", "DSN", "DATABASE_URL")
_VLM_ENDPOINT_RE = re.compile(r"^EVOSSEARCH_LM_PROFILE_(?!AGENT(?:_|$)).+_BASE_URL$")
_ENV_REFERENCE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_INSTALLER_MANAGED_ENV_KEYS = frozenset({"EVOSSEARCH_APP_VERSION"})


class InstallerError(RuntimeError):
    """An expected, operator-actionable installer failure."""


@dataclass(frozen=True)
class PromptSpec:
    key: str
    label: str
    secret: bool = False
    default: str | None = None


@dataclass(frozen=True)
class EnvResolution:
    source: Path | None
    target: Path
    raw: str
    existing: Mapping[str, str]

    @property
    def source_kind(self) -> str:
        if self.source is None:
            return "new"
        if self.source.resolve(strict=False) == self.target.resolve(strict=False):
            return "in-place"
        return "copy"


@dataclass(frozen=True)
class Finding:
    level: str
    message: str


@dataclass(frozen=True)
class PlanAction:
    phase: str
    description: str


@dataclass
class InstallerOptions:
    source_dir: Path
    bundle_dir: Path
    app_dir: Path
    env_file: Path | None
    backup_root: Path
    service_name: str
    service_user: str
    service_group: str
    unit_file: Path
    unit_template: Path
    lock_file: Path
    base_url: str
    python_bin: str
    dry_run: bool
    non_interactive: bool
    migrate: bool
    start: bool
    verify: bool
    adopt_existing_config: bool = False


@dataclass
class PreparedInstall:
    options: InstallerOptions
    env: EnvResolution
    values: dict[str, str]
    updates: dict[str, str]
    migration_dsn: str | None = field(default=None, repr=False)
    migration_dsn_source: str | None = None
    findings: list[Finding] = field(default_factory=list)
    actions: list[PlanAction] = field(default_factory=list)


_PROMPTS: tuple[PromptSpec, ...] = (
    PromptSpec("EVOSSEARCH_LUXRIOT_BASE_URL", "Luxriot Evo base URL"),
    PromptSpec("EVOSSEARCH_LUXRIOT_USERNAME", "Luxriot Evo username"),
    PromptSpec("EVOSSEARCH_LUXRIOT_PASSWORD", "Luxriot Evo password", secret=True),
    PromptSpec("EVA_DATABASE_DSN", "PostgreSQL API/runtime DSN", secret=True),
    PromptSpec("EVA_AUDIT_DATABASE_DSN", "PostgreSQL audit-writer DSN", secret=True),
    PromptSpec("EVA_WORKER_DATABASE_DSN", "PostgreSQL worker DSN", secret=True),
    PromptSpec(
        "EVOSSEARCH_LM_PROFILE_AGENT_BASE_URL",
        "Agent OpenAI-compatible endpoint",
        default="http://127.0.0.1:1234/v1",
    ),
    PromptSpec(
        "EVOSSEARCH_LM_PROFILE_AGENT_MODEL",
        "Agent model id",
        default="qwen3.5-9b-mtp",
    ),
    PromptSpec("EVOSSEARCH_LM_PROFILE_VLM_BASE_URL", "VLM OpenAI-compatible endpoint"),
    PromptSpec(
        "EVOSSEARCH_LM_PROFILE_VLM_MODEL",
        "VLM model id",
        default="qwen/qwen3-vl-4b",
    ),
)


def is_secret_key(key: str) -> bool:
    upper = str(key).upper()
    return any(marker in upper for marker in _SECRET_MARKERS)


def redact_text(text: str, secret_values: Iterable[str]) -> str:
    safe = str(text or "")
    values = sorted(
        {str(value) for value in secret_values if value and len(str(value)) >= 4},
        key=len,
        reverse=True,
    )
    for value in values:
        safe = safe.replace(value, "***")
    return safe


def _decode_env_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        quote = value[0]
        value = value[1:-1]
        if quote == '"':
            value = (
                value.replace(r"\n", "\n")
                .replace(r"\r", "\r")
                .replace(r"\"", '"')
                .replace(r"\\", "\\")
            )
    return value


def parse_env_text(raw: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in str(raw or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        key, separator, value = stripped.partition("=")
        key = key.strip()
        if not separator or not _ENV_KEY_RE.fullmatch(key):
            continue
        values[key] = _decode_env_value(value)
    return values


def expand_env_references(
    values: Mapping[str, str],
    environ: Mapping[str, str],
) -> dict[str, str]:
    expanded = dict(values)
    for _iteration in range(8):
        changed = False
        for key, value in tuple(expanded.items()):
            resolved = _ENV_REFERENCE_RE.sub(
                lambda match: str(expanded.get(match.group(1), environ.get(match.group(1), match.group(0)))),
                value,
            )
            if resolved != value:
                expanded[key] = resolved
                changed = True
        if not changed:
            break
    return expanded


def read_env_file(path: Path) -> tuple[str, dict[str, str]]:
    raw = path.read_text(encoding="utf-8")
    return raw, parse_env_text(raw)


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)


def discover_env_file(
    *,
    explicit: Path | None,
    app_dir: Path,
    source_dir: Path,
    environ: Mapping[str, str],
) -> EnvResolution:
    env_override = str(environ.get("EVA_ENV_FILE") or "").strip()
    target_hint = explicit or (Path(env_override) if env_override else DEFAULT_ENV_FILE)

    candidates: list[Path] = [target_hint]
    candidates.extend(
        (
            app_dir / "eva-ai.env",
            app_dir / ".env",
            source_dir / "eva-ai.env",
            source_dir / ".env",
        )
    )
    seen: set[Path] = set()
    source: Path | None = None
    for candidate in candidates:
        normalized = candidate.resolve(strict=False)
        if normalized in seen:
            continue
        seen.add(normalized)
        if candidate.is_file():
            source = candidate
            break

    if explicit is not None:
        target = explicit
    elif source is not None and (
        _same_path(source.parent, app_dir)
        or _same_path(source, DEFAULT_ENV_FILE)
        or (env_override and _same_path(source, Path(env_override)))
    ):
        target = source
    else:
        target = target_hint

    if source is None:
        return EnvResolution(source=None, target=target, raw="", existing={})
    raw, existing = read_env_file(source)
    return EnvResolution(source=source, target=target, raw=raw, existing=existing)


def _quote_env_value(value: str) -> str:
    if "\x00" in value or "\n" in value or "\r" in value:
        raise InstallerError("Environment values must be single-line and contain no NUL bytes.")
    if "'" in value or "${" in value:
        raise InstallerError(
            "New environment values may not contain a single quote or '${...}' reference; "
            "place complex values in the reviewed env file before running the installer."
        )
    # Single quotes are understood by both bash `source` (used by the existing
    # migration scripts) and python-dotenv, without expanding ordinary '$'.
    return f"'{value}'"


def render_env_update(raw: str, updates: Mapping[str, str]) -> str:
    content = str(raw or "")
    existing_keys = set(parse_env_text(content))
    for key in sorted(_INSTALLER_MANAGED_ENV_KEYS.intersection(updates)):
        replacement = f"{key}={_quote_env_value(str(updates[key]))}"
        pattern = re.compile(
            rf"(?m)^[ \t]*(?:export[ \t]+)?{re.escape(key)}[ \t]*=.*$"
        )
        content = pattern.sub(replacement, content)
    pending_updates = {
        key: value for key, value in updates.items()
        if key not in existing_keys
    }
    if content and not content.endswith("\n"):
        content += "\n"
    if pending_updates:
        if content:
            content += "\n"
        content += "# Added by EVA AI 0.8.3 offline installer; existing keys were preserved.\n"
        for key, value in pending_updates.items():
            if not _ENV_KEY_RE.fullmatch(key):
                raise InstallerError(f"Unsafe environment key: {key!r}")
            content += f"{key}={_quote_env_value(str(value))}\n"
    return content


def _prompt(spec: PromptSpec, *, input_fn: Callable[[str], str]) -> str:
    suffix = f" [{spec.default}]" if spec.default else ""
    label = f"{spec.label}{suffix}: "
    value = getpass.getpass(label) if spec.secret else input_fn(label)
    value = value.strip()
    if not value and spec.default is not None:
        value = spec.default
    return value


def _selected_agent_profile_key(values: Mapping[str, str], suffix: str) -> str:
    profile_id = str(values.get("EVOSSEARCH_LM_AGENT_PROFILE_ID") or "").strip()
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", profile_id).strip("_").upper()
    if not normalized:
        return ""
    return f"EVOSSEARCH_LM_PROFILE_{normalized}_{suffix}"


def _has_agent_endpoint(values: Mapping[str, str]) -> bool:
    candidates = (
        "EVOSSEARCH_LM_PROFILE_AGENT_BASE_URL",
        "EVOSSEARCH_LM_BASE_URL",
        _selected_agent_profile_key(values, "BASE_URL"),
    )
    return any(str(values.get(key) or "").strip() for key in candidates if key)


def _has_agent_model(values: Mapping[str, str]) -> bool:
    candidates = (
        "EVOSSEARCH_LM_PROFILE_AGENT_MODEL",
        "EVOSSEARCH_LM_MODEL",
        _selected_agent_profile_key(values, "MODEL"),
    )
    return any(str(values.get(key) or "").strip() for key in candidates if key)


def _has_vlm_endpoint(values: Mapping[str, str]) -> bool:
    if str(values.get("EVOSSEARCH_LM_PROFILE_VLM_BASE_URL") or "").strip():
        return True
    return any(
        _VLM_ENDPOINT_RE.fullmatch(key) and str(value or "").strip()
        for key, value in values.items()
    )


def _has_vlm_model(values: Mapping[str, str]) -> bool:
    if str(values.get("EVOSSEARCH_LM_PROFILE_VLM_MODEL") or "").strip():
        return True
    return any(
        key.startswith("EVOSSEARCH_LM_PROFILE_")
        and key.endswith("_MODEL")
        and "_AGENT_" not in key
        and str(value or "").strip()
        for key, value in values.items()
    )


def prepare_env_values(
    resolution: EnvResolution,
    *,
    environ: Mapping[str, str],
    non_interactive: bool,
    input_fn: Callable[[str], str] = input,
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    values = expand_env_references(resolution.existing, environ)
    updates: dict[str, str] = {}

    def add_missing(key: str, value: str) -> None:
        if not str(values.get(key) or "").strip() and str(value or "").strip():
            values[key] = str(value)
            updates[key] = str(value)

    defaults: dict[str, str] = {}
    if not resolution.existing:
        tenant_id = str(uuid.uuid4())
        defaults.update({
            "EVOSSEARCH_APP_VERSION": EXPECTED_VERSION,
            "EVOSSEARCH_SECURE_DEPLOYMENT_REQUIRED": "true",
            "EVOSSEARCH_AUTH_ENABLED": "true",
            "EVOSSEARCH_DB_STRICT_RUNTIME_ROLES": "true",
            "EVOSSEARCH_ARCHIVE_STORE": "postgres",
            "EVOSSEARCH_EMBEDDER": "clip",
            "EVOSSEARCH_DINO_SEGMENTS_ENABLED": "false",
            "EVOSSEARCH_EXPERIMENTAL_EMBEDDERS_ENABLED": "false",
            "EVOSSEARCH_GUNICORN_WORKERS": "1",
            "EVOSSEARCH_GUNICORN_THREADS": "8",
            "EVOSSEARCH_LUXRIOT_LIVE_MEDIA_MAX_SECONDS": "120",
            "EVOSSEARCH_LUXRIOT_LIVE_MEDIA_MAX_BYTES": "268435456",
            "EVOSSEARCH_LUXRIOT_LIVE_SEGMENT_SECONDS": "60",
            "EVOSSEARCH_HOST": "127.0.0.1",
            "EVOSSEARCH_PORT": "5000",
            "EVOSSEARCH_AUTH_COOKIE_SECURE": "true",
            "EVOSSEARCH_AUTH_TENANT_ID": tenant_id,
            "EVOSSEARCH_ARCHIVE_TENANT_ID": tenant_id,
            "EVOSSEARCH_LM_PROFILES": "agent,vlm",
            "EVOSSEARCH_LM_AGENT_PROFILE_ID": "agent",
            "EVOSSEARCH_LM_VLM_PROFILE_ID": "vlm",
            "EVOSSEARCH_LM_PROFILE_AGENT_ENABLED": "true",
            "EVOSSEARCH_LM_PROFILE_AGENT_KIND": "agent",
            "EVOSSEARCH_LM_PROFILE_VLM_ENABLED": "true",
            "EVOSSEARCH_LM_PROFILE_VLM_KIND": "vlm",
        })
    for key, value in defaults.items():
        add_missing(key, value)

    # The release identity belongs to the installed code, not to site
    # configuration.  It is the sole reviewed key that an adopt upgrade may
    # replace; all operational settings remain preserve/append-only.
    if "EVOSSEARCH_APP_VERSION" in resolution.existing:
        current_version = str(values.get("EVOSSEARCH_APP_VERSION") or "").strip()
        if current_version != EXPECTED_VERSION:
            values["EVOSSEARCH_APP_VERSION"] = EXPECTED_VERSION
            updates["EVOSSEARCH_APP_VERSION"] = EXPECTED_VERSION

    for spec in _PROMPTS:
        if spec.key == "EVOSSEARCH_LM_PROFILE_AGENT_BASE_URL" and _has_agent_endpoint(values):
            continue
        if spec.key == "EVOSSEARCH_LM_PROFILE_AGENT_MODEL" and _has_agent_model(values):
            continue
        if spec.key == "EVOSSEARCH_LM_PROFILE_VLM_BASE_URL" and _has_vlm_endpoint(values):
            continue
        if spec.key == "EVOSSEARCH_LM_PROFILE_VLM_MODEL" and _has_vlm_model(values):
            continue
        if str(values.get(spec.key) or "").strip():
            continue
        supplied = str(environ.get(spec.key) or "").strip()
        if supplied:
            add_missing(spec.key, supplied)
            continue
        if non_interactive:
            continue
        entered = _prompt(spec, input_fn=input_fn)
        add_missing(spec.key, entered)

    missing: list[str] = []
    exact_required = (
        "EVOSSEARCH_LUXRIOT_BASE_URL",
        "EVOSSEARCH_LUXRIOT_USERNAME",
        "EVOSSEARCH_LUXRIOT_PASSWORD",
        "EVA_DATABASE_DSN",
        "EVA_AUDIT_DATABASE_DSN",
        "EVA_WORKER_DATABASE_DSN",
    )
    for key in exact_required:
        if not str(values.get(key) or "").strip():
            missing.append(key)
    if not _has_agent_endpoint(values):
        missing.append("EVOSSEARCH_LM_PROFILE_AGENT_BASE_URL (or EVOSSEARCH_LM_BASE_URL)")
    if not _has_agent_model(values):
        missing.append("EVOSSEARCH_LM_PROFILE_AGENT_MODEL (or EVOSSEARCH_LM_MODEL)")
    if not _has_vlm_endpoint(values):
        missing.append("EVOSSEARCH_LM_PROFILE_<VLM>_BASE_URL")
    if not _has_vlm_model(values):
        missing.append("EVOSSEARCH_LM_PROFILE_<VLM>_MODEL")
    return values, updates, missing


def prepare_migration_dsn(
    values: MutableMapping[str, str],
    updates: MutableMapping[str, str],
    *,
    environ: Mapping[str, str],
    migrate: bool,
    non_interactive: bool,
) -> tuple[str | None, str | None, str | None]:
    if not migrate:
        return None, None, None
    process_only = str(environ.get("EVA_INSTALL_MIGRATION_DSN") or "").strip()
    if process_only:
        migration_dsn = process_only
        source = "EVA_INSTALL_MIGRATION_DSN (process-only)"
    else:
        migration_dsn = str(
            values.get("EVA_MIGRATION_DATABASE_DSN")
            or environ.get("EVA_MIGRATION_DATABASE_DSN")
            or ""
        ).strip()
        source = "EVA_MIGRATION_DATABASE_DSN"
        if migration_dsn and not str(values.get("EVA_MIGRATION_DATABASE_DSN") or "").strip():
            values["EVA_MIGRATION_DATABASE_DSN"] = migration_dsn
            updates["EVA_MIGRATION_DATABASE_DSN"] = migration_dsn
        if not migration_dsn and not non_interactive:
            migration_dsn = getpass.getpass(
                "Privileged PostgreSQL migration DSN "
                "(stored as EVA_MIGRATION_DATABASE_DSN): "
            ).strip()
            if migration_dsn:
                values["EVA_MIGRATION_DATABASE_DSN"] = migration_dsn
                updates["EVA_MIGRATION_DATABASE_DSN"] = migration_dsn
    if not migration_dsn:
        return None, None, (
            "EVA_INSTALL_MIGRATION_DSN (process-only) or "
            "EVA_MIGRATION_DATABASE_DSN is required with --migrate"
        )
    runtime_dsn = str(values.get("EVA_DATABASE_DSN") or "").strip()
    if runtime_dsn and migration_dsn == runtime_dsn:
        return migration_dsn, source, (
            "migration DSN must be distinct from runtime EVA_DATABASE_DSN"
        )
    return migration_dsn, source, None


def _looks_like_placeholder(key: str, value: str) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    compact = re.sub(r"\s+", "", text)
    exact = {
        "changeme",
        "change-me",
        "change_me",
        "password",
        "admin:123",
        "example",
        "placeholder",
        "replace-me",
        "replace_me",
        "<...>",
        "<password>",
        "<secret>",
        "[field]",
    }
    if compact in exact:
        return True
    if re.search(r"<[^>]+>", text):
        return True
    markers = (
        "changeme",
        "placeholder",
        "replace-me",
        "replace_me",
        "<...>",
        "<password>",
        "<secret>",
        "example.com",
        "://example",
        "luxriot-host",
        "your-host",
        "your_password",
        "your-password",
        "[field]",
    )
    if any(marker in compact for marker in markers):
        return True
    upper_key = key.upper()
    if "PASSWORD" in upper_key and compact in {"secret", "test", "123", "123456"}:
        return True
    if "DSN" in upper_key and any(marker in compact for marker in (":password@", ":changeme@")):
        return True
    return False


def _valid_http_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname)


def _valid_postgres_dsn(value: str) -> bool:
    text = str(value or "").strip()
    if "://" not in text:
        return "=" in text
    try:
        return urlsplit(text).scheme.lower() in {"postgres", "postgresql"}
    except ValueError:
        return False


def _wheelhouse_files(bundle_dir: Path) -> list[Path]:
    wheelhouse = bundle_dir / "wheelhouse"
    if not wheelhouse.is_dir():
        return []
    patterns = ("*.whl", "*.tar.gz", "*.zip")
    return sorted({path for pattern in patterns for path in wheelhouse.glob(pattern)})


def _version(source_dir: Path) -> str:
    version_file = source_dir / "VERSION"
    if not version_file.is_file():
        return ""
    return version_file.read_text(encoding="utf-8").strip()


def collect_preflight(
    options: InstallerOptions,
    resolution: EnvResolution,
    values: Mapping[str, str],
    missing: Sequence[str],
    migration_dsn: str | None,
    migration_dsn_source: str | None,
) -> list[Finding]:
    findings: list[Finding] = []

    def add(level: str, message: str) -> None:
        findings.append(Finding(level, message))

    required_paths = (
        options.source_dir / "run_prod.sh",
        options.source_dir / "wsgi.py",
        options.source_dir / "requirements.txt",
        options.source_dir / "alembic.ini",
        options.source_dir / "migrations",
        options.source_dir / "static" / "js" / "app.js",
        options.source_dir / "templates" / "index.html",
        options.source_dir / "scripts" / "preflight_patch.sh",
        options.source_dir / "scripts" / "install_patch.sh",
        options.source_dir / "scripts" / "verify_patch.sh",
        options.source_dir / "scripts" / "rollback.sh",
        options.unit_template,
    )
    for path in required_paths:
        if not path.exists():
            add("FAIL", f"required offline payload path is missing: {path}")
        elif path.suffix == ".sh" and not os.access(path, os.X_OK):
            add("FAIL", f"installer helper is not executable: {path}")
    source_version = _version(options.source_dir)
    if source_version == EXPECTED_VERSION:
        add("OK", f"source version is {EXPECTED_VERSION}")
    elif source_version:
        add("FAIL", f"source version is {source_version!r}; expected {EXPECTED_VERSION!r}")
    else:
        add("FAIL", "source VERSION is missing")

    if missing:
        add("FAIL", "required configuration keys are missing: " + ", ".join(missing))
    else:
        add("OK", "required Evo/PostgreSQL/agent/VLM configuration is present")

    for key in (
        "EVOSSEARCH_LUXRIOT_BASE_URL",
        "EVOSSEARCH_LM_PROFILE_AGENT_BASE_URL",
        "EVOSSEARCH_LM_PROFILE_VLM_BASE_URL",
    ):
        value = str(values.get(key) or "").strip()
        if value and not _valid_http_url(value):
            add("FAIL", f"{key} must be an http(s) URL")
    for key, value in values.items():
        if _VLM_ENDPOINT_RE.fullmatch(key) and value and not _valid_http_url(value):
            add("FAIL", f"{key} must be an http(s) URL")
    for key in ("EVA_DATABASE_DSN", "EVA_AUDIT_DATABASE_DSN", "EVA_WORKER_DATABASE_DSN"):
        value = str(values.get(key) or "").strip()
        if value and not _valid_postgres_dsn(value):
            add("FAIL", f"{key} must be a PostgreSQL URI or libpq conninfo")
        if "${" in value:
            add("FAIL", f"{key} contains an unresolved environment reference")
    placeholder_keys = {
        "EVOSSEARCH_LUXRIOT_BASE_URL",
        "EVOSSEARCH_LUXRIOT_USERNAME",
        "EVOSSEARCH_LUXRIOT_PASSWORD",
        "EVA_DATABASE_DSN",
        "EVA_AUDIT_DATABASE_DSN",
        "EVA_WORKER_DATABASE_DSN",
        "EVOSSEARCH_LM_PROFILE_AGENT_BASE_URL",
        "EVOSSEARCH_LM_PROFILE_AGENT_MODEL",
        "EVOSSEARCH_LM_BASE_URL",
        "EVOSSEARCH_LM_MODEL",
    }
    placeholder_keys.update(
        key for key in values
        if _VLM_ENDPOINT_RE.fullmatch(key)
        or (
            key.startswith("EVOSSEARCH_LM_PROFILE_")
            and key.endswith("_MODEL")
        )
    )
    verified_adopt = bool(
        options.adopt_existing_config
        and not options.migrate
        and resolution.source is not None
        and resolution.source.resolve(strict=False) == resolution.target.resolve(strict=False)
        and (options.app_dir / "VERSION").is_file()
    )
    for key in sorted(placeholder_keys):
        value = str(values.get(key) or "").strip()
        if value and _looks_like_placeholder(key, value):
            if verified_adopt:
                add(
                    "WARN",
                    f"{key} looks like a placeholder but is preserved by verified code-only adopt",
                )
            else:
                add("FAIL", f"{key} contains an obvious placeholder value")

    if options.migrate and migration_dsn:
        if not _valid_postgres_dsn(migration_dsn):
            add("FAIL", "privileged migration DSN must be a PostgreSQL URI or libpq conninfo")
        elif _looks_like_placeholder("EVA_MIGRATION_DATABASE_DSN", migration_dsn):
            add("FAIL", "privileged migration DSN contains an obvious placeholder value")
        else:
            add("OK", f"distinct privileged migration DSN supplied via {migration_dsn_source}")

    workers = str(values.get("EVOSSEARCH_GUNICORN_WORKERS") or "1").strip()
    if workers != "1":
        add("FAIL", "EVOSSEARCH_GUNICORN_WORKERS must be 1")
    if not _valid_http_url(options.base_url):
        add("FAIL", "--base-url must be an http(s) URL")
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+", options.service_name):
        add("FAIL", "--service-name contains unsafe characters")
    for flag, value in (("--service-user", options.service_user), ("--service-group", options.service_group)):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", value):
            add("FAIL", f"{flag} contains unsafe characters")

    if resolution.source is None:
        add("WARN", f"no existing eva-ai.env/.env found; a new file would be created at {resolution.target}")
    elif resolution.source_kind == "copy":
        add(
            "OK",
            "existing environment will be copied without overwriting keys: "
            f"{resolution.source} -> {resolution.target}",
        )
    else:
        add("OK", f"existing environment will be preserved in place: {resolution.target}")

    target_venv = options.app_dir / ".venv" / "bin" / "python"
    wheels = _wheelhouse_files(options.bundle_dir)
    if wheels:
        add("OK", f"offline wheelhouse contains {len(wheels)} artifact(s)")
    elif target_venv.is_file() and os.access(target_venv, os.X_OK):
        add("WARN", "no wheelhouse found; existing target venv would be reused without downloads")
    else:
        add("FAIL", "fresh install requires bundle/wheelhouse; online dependency downloads are forbidden")

    try:
        disk_path = options.app_dir.parent if options.app_dir.parent.exists() else Path("/")
        free_mb = shutil.disk_usage(disk_path).free // (1024 * 1024)
        if free_mb < 4096:
            add("FAIL", f"less than 4096 MB free on target filesystem ({free_mb} MB)")
        else:
            add("OK", f"target filesystem has {free_mb} MB free")
    except OSError as exc:
        add("WARN", f"could not inspect target free space: {exc}")

    if shutil.which(options.python_bin) is None and not Path(options.python_bin).is_file():
        add("FAIL", f"Python executable not found: {options.python_bin}")
    for command in ("bash", "tar"):
        if shutil.which(command) is None:
            add("FAIL", f"required host command not found: {command}")
    try:
        grp.getgrnam(options.service_group)
    except KeyError:
        if shutil.which("groupadd") is None:
            add("FAIL", f"group {options.service_group!r} is absent and groupadd is unavailable")
    try:
        pwd.getpwnam(options.service_user)
    except KeyError:
        if shutil.which("useradd") is None:
            add("FAIL", f"user {options.service_user!r} is absent and useradd is unavailable")
    if options.migrate and shutil.which("pg_dump") is None:
        add("FAIL", "pg_dump is required before migrations; no unsafe skip is provided")
    if options.dry_run:
        add("OK", "dry-run mode: no filesystem, database, service, or network state will change")
    else:
        if os.geteuid() != 0:
            add("FAIL", "--apply requires root (run with sudo)")
        if shutil.which("systemctl") is None:
            add("FAIL", "systemctl is required for --apply")
        if _same_path(options.source_dir, options.app_dir):
            add("FAIL", "source-dir and app-dir must differ for --apply")
        if options.unit_file.name != options.service_name + ".service":
            add("FAIL", "--unit-file basename must match <service-name>.service")
    return findings


def build_plan(prepared: PreparedInstall) -> list[PlanAction]:
    options = prepared.options
    env_action = (
        f"preserve {prepared.env.source} and append only {len(prepared.updates)} missing key(s)"
        if prepared.env.source_kind == "in-place"
        else (
            f"copy {prepared.env.source} to {prepared.env.target}, preserving all existing keys"
            if prepared.env.source_kind == "copy"
            else f"create {prepared.env.target} with mode 0600"
        )
    )
    unit_action = (
        f"preserve existing systemd unit {options.unit_file} unchanged"
        if options.unit_file.is_file()
        else f"render new {options.unit_file} from installer template and daemon-reload"
    )
    actions = [
        PlanAction(
            "lock",
            f"acquire nonblocking apply lock {options.lock_file} through verification/handoff",
        ),
        PlanAction("configuration", env_action),
        PlanAction(
            "host",
            (
                "preserve the account selected by the existing systemd unit and ensure target directories"
                if options.unit_file.is_file()
                else f"ensure service account {options.service_user}:{options.service_group} and target directories"
            ),
        ),
        PlanAction(
            "dependencies",
            "reuse target .venv or create it, then install only from bundle/wheelhouse with --no-index",
        ),
        PlanAction(
            "systemd",
            unit_action,
        ),
        PlanAction(
            "preflight",
            "run existing scripts/preflight_patch.sh without stopping the service",
        ),
        PlanAction(
            "install",
            "run existing scripts/install_patch.sh for code/env/unit/DB backup and static copy; keep service stopped",
        ),
    ]
    if options.migrate:
        actions.append(PlanAction(
            "database",
            f"require postgres.dump, then run Alembic current -> upgrade head -> current ({EXPECTED_SCHEMA})",
        ))
    else:
        actions.append(PlanAction("database", "skip migrations by explicit operator request"))
    if options.start:
        actions.append(PlanAction("service", f"enable and restart {options.service_name}.service"))
    else:
        actions.append(PlanAction("service", "leave service stopped by explicit operator request"))
    if options.verify:
        actions.append(PlanAction("health", "run existing scripts/verify_patch.sh for /health and /ready"))
    actions.append(PlanAction(
        "rollback",
        f"handoff scripts/rollback.sh with the backup recorded under {options.backup_root}/LATEST",
    ))
    return actions


def _derive_bundle_dir(source_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    parent = source_dir.parent
    if source_dir.name == "repo" and ((parent / "wheelhouse").is_dir() or (parent / "manifest.txt").is_file()):
        return parent
    return source_dir


def render_unit(template: str, options: InstallerOptions) -> str:
    replacements = {
        "@SERVICE_NAME@": options.service_name,
        "@SERVICE_USER@": options.service_user,
        "@SERVICE_GROUP@": options.service_group,
        "@APP_DIR@": str(options.app_dir),
        "@ENV_FILE@": str(options.env_file or DEFAULT_ENV_FILE),
    }
    rendered = str(template)
    for marker, value in replacements.items():
        if "\n" in value or "\r" in value:
            raise InstallerError(f"Unsafe newline in systemd template value for {marker}")
        rendered = rendered.replace(marker, value)
    leftovers = sorted(set(re.findall(r"@[A-Z_]+@", rendered)))
    if leftovers:
        raise InstallerError("Unresolved systemd template markers: " + ", ".join(leftovers))
    return rendered


def _atomic_write(path: Path, content: str, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_path, mode)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


class CommandRunner:
    def __init__(self, secret_values: Iterable[str]) -> None:
        self._secrets = tuple(value for value in secret_values if value)

    def run(
        self,
        command: Sequence[str | Path],
        *,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        argv = [str(item) for item in command]
        rendered_command = "[RUN] " + " ".join(shlex.quote(item) for item in argv)
        print(redact_text(rendered_command, self._secrets))
        completed = subprocess.run(
            argv,
            cwd=str(cwd) if cwd is not None else None,
            env=dict(env) if env is not None else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        stdout = redact_text(completed.stdout, self._secrets)
        stderr = redact_text(completed.stderr, self._secrets)
        if stdout:
            print(stdout, end="" if stdout.endswith("\n") else "\n")
        if stderr:
            print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr)
        if completed.returncode != 0:
            raise InstallerError(f"Command failed with exit {completed.returncode}: {argv[0]}")
        return completed


def _ensure_service_account(options: InstallerOptions, runner: CommandRunner) -> None:
    try:
        grp.getgrnam(options.service_group)
    except KeyError:
        runner.run(("groupadd", "--system", options.service_group))
    try:
        pwd.getpwnam(options.service_user)
    except KeyError:
        runner.run((
            "useradd",
            "--system",
            "--gid",
            options.service_group,
            "--home-dir",
            str(options.app_dir.parent),
            "--shell",
            "/usr/sbin/nologin",
            options.service_user,
        ))


def _chown_tree(path: Path, user: str, group: str) -> None:
    uid = pwd.getpwnam(user).pw_uid
    gid = grp.getgrnam(group).gr_gid
    for root, dirs, files in os.walk(path):
        os.lchown(root, uid, gid)
        for name in dirs:
            os.lchown(Path(root) / name, uid, gid)
        for name in files:
            os.lchown(Path(root) / name, uid, gid)


def _backup_file(path: Path) -> Path | None:
    if not path.is_file():
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.preinstall-{timestamp}.bak")
    shutil.copy2(path, backup)
    return backup


def _migration_environment(prepared: PreparedInstall) -> dict[str, str]:
    env = dict(os.environ)
    env.update(prepared.values)
    if not prepared.migration_dsn:
        raise InstallerError("A distinct privileged migration DSN is required")
    env["EVA_DATABASE_DSN"] = prepared.migration_dsn
    return env


def _latest_backup(backup_root: Path) -> Path:
    latest = backup_root / "LATEST"
    if not latest.is_file():
        raise InstallerError(f"install_patch did not record {latest}")
    raw = latest.read_text(encoding="utf-8").strip()
    backup = Path(raw).resolve(strict=False)
    root = backup_root.resolve(strict=False)
    if root not in backup.parents:
        raise InstallerError("Refusing backup path outside configured backup root")
    if not backup.is_dir():
        raise InstallerError(f"Recorded backup directory is missing: {backup}")
    return backup


@contextmanager
def install_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise InstallerError(
                f"another EVA AI installer holds apply lock {path}"
            ) from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def apply_install(prepared: PreparedInstall) -> Path:
    options = prepared.options
    if options.dry_run:
        raise InstallerError("Internal error: apply_install called during dry-run")
    if os.geteuid() != 0:
        raise InstallerError("--apply requires root (run with sudo)")
    if options.migrate and not prepared.migration_dsn:
        raise InstallerError(
            "Refusing --migrate without distinct EVA_INSTALL_MIGRATION_DSN or "
            "EVA_MIGRATION_DATABASE_DSN"
        )

    secret_values = [value for key, value in prepared.values.items() if is_secret_key(key)]
    migration_dsn = str(prepared.migration_dsn or "").strip()
    if migration_dsn:
        secret_values.append(migration_dsn)
    runner = CommandRunner(secret_values)
    backup_dir: Path | None = None
    app_preexisted = options.app_dir.exists()
    env_preexisted = prepared.env.target.exists()
    unit_preexisted = options.unit_file.exists()
    env_preinstall_backup: Path | None = None

    try:
        # A code-only adopt upgrade must not silently replace the site's
        # reviewed service identity or hardening.  Existing units are preserved;
        # the installer account/template are only for a fresh service.
        if not unit_preexisted:
            _ensure_service_account(options, runner)
        options.app_dir.mkdir(parents=True, exist_ok=True)
        options.backup_root.mkdir(parents=True, exist_ok=True)
        if not app_preexisted:
            uid = pwd.getpwnam(options.service_user).pw_uid
            gid = grp.getgrnam(options.service_group).gr_gid
            os.chown(options.app_dir, uid, gid)

        env_content = render_env_update(prepared.env.raw, prepared.updates)
        current_env_content = (
            prepared.env.target.read_text(encoding="utf-8")
            if prepared.env.target.is_file()
            else None
        )
        if current_env_content != env_content:
            env_preinstall_backup = _backup_file(prepared.env.target)
            _atomic_write(prepared.env.target, env_content, 0o600)
        else:
            os.chmod(prepared.env.target, 0o600)

        venv_python = options.app_dir / ".venv" / "bin" / "python"
        if not venv_python.is_file():
            runner.run((options.python_bin, "-m", "venv", options.app_dir / ".venv"))
            _chown_tree(options.app_dir / ".venv", options.service_user, options.service_group)

        prepared.options.env_file = prepared.env.target
        if not unit_preexisted:
            template = options.unit_template.read_text(encoding="utf-8")
            unit_content = render_unit(template, options)
            _atomic_write(options.unit_file, unit_content, 0o644)
            runner.run(("systemctl", "daemon-reload"))

        preflight = options.source_dir / "scripts" / "preflight_patch.sh"
        runner.run((
            preflight,
            "--bundle-dir", options.bundle_dir,
            "--app-dir", options.app_dir,
            "--env-file", prepared.env.target,
            "--service", options.service_name,
            "--base-url", options.base_url,
            "--backup-root", options.backup_root,
            "--expected-version", EXPECTED_VERSION,
            "--expected-schema", EXPECTED_SCHEMA,
            "--skip-service",
        ))

        install_env = dict(os.environ)
        if migration_dsn:
            install_env["EVA_PATCH_PG_DSN"] = migration_dsn
        install_patch = options.source_dir / "scripts" / "install_patch.sh"
        runner.run((
            install_patch,
            "--bundle-dir", options.bundle_dir,
            "--source-dir", options.source_dir,
            "--app-dir", options.app_dir,
            "--env-file", prepared.env.target,
            "--service", options.service_name,
            "--base-url", options.base_url,
            "--backup-root", options.backup_root,
            "--no-start",
            "--no-verify",
        ), env=install_env)
        backup_dir = _latest_backup(options.backup_root)
        if env_preexisted and env_preinstall_backup is not None:
            shutil.copy2(env_preinstall_backup, backup_dir / "eva-ai.env")
            env_preinstall_backup.unlink(missing_ok=True)
        state = (
            f"created_at={datetime.now(timezone.utc).isoformat()}\n"
            f"installation_mode={'upgrade' if app_preexisted else 'fresh'}\n"
            f"app_preexisted={'true' if app_preexisted else 'false'}\n"
            f"env_preexisted={'true' if env_preexisted else 'false'}\n"
            f"unit_preexisted={'true' if unit_preexisted else 'false'}\n"
            f"app_dir={options.app_dir}\n"
            f"env_file={prepared.env.target}\n"
            f"unit_file={options.unit_file}\n"
            f"service_name={options.service_name}\n"
        )
        _atomic_write(backup_dir / "offline-installer-state.txt", state, 0o600)

        if options.migrate:
            db_dump = backup_dir / "postgres.dump"
            if not db_dump.is_file() or db_dump.stat().st_size <= 0:
                raise InstallerError(
                    "PostgreSQL backup is absent; refusing to run migrations. "
                    "Fix pg_dump/permissions and retry."
                )
            alembic = options.app_dir / ".venv" / "bin" / "alembic"
            if not alembic.is_file():
                raise InstallerError(f"Alembic is missing after dependency setup: {alembic}")
            migration_env = _migration_environment(prepared)
            runner.run((alembic, "current"), cwd=options.app_dir, env=migration_env)
            runner.run((alembic, "upgrade", "head"), cwd=options.app_dir, env=migration_env)
            current = runner.run((alembic, "current"), cwd=options.app_dir, env=migration_env)
            if EXPECTED_SCHEMA not in current.stdout:
                raise InstallerError(
                    f"Alembic current did not report expected schema {EXPECTED_SCHEMA}"
                )

        if options.start:
            runner.run(("systemctl", "enable", options.service_name + ".service"))
            runner.run(("systemctl", "restart", options.service_name + ".service"))
        if options.verify:
            verify = options.app_dir / "scripts" / "verify_patch.sh"
            runner.run((
                verify,
                "--service", options.service_name,
                "--base-url", options.base_url,
                "--timeout", "90",
            ))
        return backup_dir
    except Exception:
        if backup_dir is None:
            latest = options.backup_root / "LATEST"
            if latest.is_file():
                try:
                    backup_dir = _latest_backup(options.backup_root)
                except InstallerError:
                    backup_dir = None
        # If install_patch never established a rollback snapshot, undo the
        # small amount of staging performed by this orchestrator itself.  If
        # it did establish one but then failed, make sure that snapshot holds
        # the *pre-orchestrator* env rather than the appended staging copy.
        if env_preinstall_backup is not None and env_preinstall_backup.is_file():
            if backup_dir is not None:
                shutil.copy2(env_preinstall_backup, backup_dir / "eva-ai.env")
            elif env_preexisted:
                shutil.copy2(env_preinstall_backup, prepared.env.target)
            env_preinstall_backup.unlink(missing_ok=True)
        elif backup_dir is None and not env_preexisted:
            prepared.env.target.unlink(missing_ok=True)
        if backup_dir is None and not unit_preexisted:
            options.unit_file.unlink(missing_ok=True)
            subprocess.run(
                ("systemctl", "daemon-reload"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        if backup_dir is not None:
            print(
                "ROLLBACK HANDOFF: sudo "
                f"{shlex.quote(str(options.source_dir / 'scripts' / 'rollback.sh'))} "
                f"--backup-dir {shlex.quote(str(backup_dir))} "
                f"--app-dir {shlex.quote(str(options.app_dir))} "
                f"--env-file {shlex.quote(str(prepared.env.target))} "
                f"--service {shlex.quote(options.service_name)}",
                file=sys.stderr,
            )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run-first, offline EVA AI installer.",
    )
    parser.add_argument("--source-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--app-dir", type=Path, default=DEFAULT_APP_DIR)
    parser.add_argument("--env-file", type=Path, help="Use/create this env file; otherwise discover eva-ai.env/.env.")
    parser.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT)
    parser.add_argument("--service-name", default="eva-ai")
    parser.add_argument("--service-user", default="eva")
    parser.add_argument("--service-group", default="eva")
    parser.add_argument("--unit-file", type=Path, default=DEFAULT_UNIT_FILE)
    parser.add_argument("--unit-template", type=Path)
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK_FILE)
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--python", dest="python_bin", default="python3")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Plan only (the default).")
    mode.add_argument("--apply", action="store_true", help="Perform the reviewed plan; requires root.")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Never prompt; fail when required env values are absent.",
    )
    parser.add_argument(
        "--no-migrate",
        action="store_true",
        help="Explicitly skip Alembic migration (not recommended).",
    )
    parser.add_argument(
        "--verified-adopt-existing-config",
        action="store_true",
        help=(
            "Preserve an existing running deployment's config and downgrade only "
            "placeholder-like values to warnings. Requires an in-place env, installed "
            "VERSION, and --no-migrate; intended for a live-verified orchestrator."
        ),
    )
    parser.add_argument("--no-start", action="store_true", help="Install but leave the service stopped.")
    parser.add_argument("--no-verify", action="store_true", help="Skip post-start /health and /ready checks.")
    return parser


def _options(args: argparse.Namespace) -> InstallerOptions:
    source_dir = args.source_dir.resolve(strict=False)
    bundle_dir = _derive_bundle_dir(source_dir, args.bundle_dir)
    return InstallerOptions(
        source_dir=source_dir,
        bundle_dir=bundle_dir.resolve(strict=False),
        app_dir=args.app_dir.resolve(strict=False),
        env_file=args.env_file.resolve(strict=False) if args.env_file else None,
        backup_root=args.backup_root.resolve(strict=False),
        service_name=str(args.service_name),
        service_user=str(args.service_user),
        service_group=str(args.service_group),
        unit_file=args.unit_file.resolve(strict=False),
        unit_template=(
            args.unit_template
            or (source_dir / "scripts" / "install_assets" / "eva-ai.service.in")
        ).resolve(strict=False),
        lock_file=args.lock_file.resolve(strict=False),
        base_url=str(args.base_url).rstrip("/"),
        python_bin=str(args.python_bin),
        dry_run=not bool(args.apply),
        non_interactive=bool(args.non_interactive),
        migrate=not bool(args.no_migrate),
        start=not bool(args.no_start),
        verify=not bool(args.no_verify) and not bool(args.no_start),
        adopt_existing_config=bool(args.verified_adopt_existing_config),
    )


def prepare_install(options: InstallerOptions, environ: Mapping[str, str] | None = None) -> PreparedInstall:
    source_env = os.environ if environ is None else environ
    resolution = discover_env_file(
        explicit=options.env_file,
        app_dir=options.app_dir,
        source_dir=options.source_dir,
        environ=source_env,
    )
    values, updates, missing = prepare_env_values(
        resolution,
        environ=source_env,
        non_interactive=options.non_interactive,
    )
    migration_dsn, migration_dsn_source, migration_error = prepare_migration_dsn(
        values,
        updates,
        environ=source_env,
        migrate=options.migrate,
        non_interactive=options.non_interactive,
    )
    if migration_error:
        missing.append(migration_error)
    findings = collect_preflight(
        options,
        resolution,
        values,
        missing,
        migration_dsn,
        migration_dsn_source,
    )
    prepared = PreparedInstall(
        options=options,
        env=resolution,
        values=values,
        updates=updates,
        migration_dsn=migration_dsn,
        migration_dsn_source=migration_dsn_source,
        findings=findings,
    )
    prepared.actions = build_plan(prepared)
    return prepared


def print_prepared(prepared: PreparedInstall) -> None:
    print(f"EVA AI {EXPECTED_VERSION} offline installer")
    print("MODE: DRY-RUN (no changes)" if prepared.options.dry_run else "MODE: APPLY")
    print(f"source_dir={prepared.options.source_dir}")
    print(f"app_dir={prepared.options.app_dir}")
    print(f"env_file={prepared.env.target}")
    print(f"env_source={prepared.env.source or '[new]'}")
    configuration_state = [
        (key, bool(str(prepared.values.get(key) or "").strip()))
        for key in (
            "EVOSSEARCH_LUXRIOT_BASE_URL",
            "EVOSSEARCH_LUXRIOT_USERNAME",
            "EVOSSEARCH_LUXRIOT_PASSWORD",
            "EVA_DATABASE_DSN",
            "EVA_AUDIT_DATABASE_DSN",
            "EVA_WORKER_DATABASE_DSN",
        )
    ]
    configuration_state.extend((
        ("agent_lm_endpoint", _has_agent_endpoint(prepared.values)),
        ("agent_lm_model", _has_agent_model(prepared.values)),
        ("vlm_endpoint", _has_vlm_endpoint(prepared.values)),
        ("vlm_model", _has_vlm_model(prepared.values)),
    ))
    if prepared.options.migrate:
        configuration_state.append(("privileged_migration_dsn", bool(prepared.migration_dsn)))
    print("configuration: " + ", ".join(
        f"{key}={'[set]' if present else '[missing]'}"
        for key, present in configuration_state
    ))
    print("\nPreflight:")
    for finding in prepared.findings:
        print(f"{finding.level}: {finding.message}")
    print("\nPlan:")
    for index, action in enumerate(prepared.actions, 1):
        print(f"{index:02d}. [{action.phase}] {action.description}")
    print("\nNo package-index or source-control network operation is part of this plan.")
    if prepared.options.dry_run:
        print("Review the plan, then rerun the same command with --apply under sudo.")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    options = _options(args)
    try:
        prepared = prepare_install(options)
        print_prepared(prepared)
        failures = [finding for finding in prepared.findings if finding.level == "FAIL"]
        if failures:
            print(f"\nBLOCKED: {len(failures)} preflight failure(s); no changes made.", file=sys.stderr)
            return 2
        if options.dry_run:
            return 0
        with install_lock(options.lock_file):
            backup_dir = apply_install(prepared)
            print("\nINSTALL COMPLETE")
            print(f"backup_dir={backup_dir}")
            print(
                "rollback_command=sudo "
                f"{options.app_dir / 'scripts' / 'rollback.sh'} "
                f"--backup-dir {backup_dir} --app-dir {options.app_dir} "
                f"--env-file {prepared.env.target} --service {options.service_name}"
            )
        return 0
    except (InstallerError, OSError) as exc:
        safe_secrets: list[str] = []
        try:
            safe_secrets = [
                value for key, value in prepared.values.items()  # type: ignore[possibly-undefined]
                if is_secret_key(key)
            ]
            if prepared.migration_dsn:  # type: ignore[possibly-undefined]
                safe_secrets.append(prepared.migration_dsn)  # type: ignore[possibly-undefined]
        except UnboundLocalError:
            pass
        print("INSTALLER ERROR: " + redact_text(str(exc), safe_secrets), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
