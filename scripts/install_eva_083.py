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
import hashlib
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
from urllib.request import (
    HTTPDigestAuthHandler,
    HTTPPasswordMgrWithDefaultRealm,
    Request,
    build_opener,
)


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

from pg_with_dsn import DsnError, postgres_environment


def _expected_version() -> str:
    """The bundled VERSION file is authoritative; the constant is a fallback.

    The installer ships inside the source tree it installs, so a hard-coded
    version string silently rots on every release bump.
    """

    try:
        text = (REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        text = ""
    return text or "β 0.8.7"


EXPECTED_VERSION = _expected_version()
EXPECTED_SCHEMA = "20260805_0013"
DEFAULT_APP_DIR = Path("/opt/eva-ai/evo-ssearch")
DEFAULT_ENV_FILE = Path("/etc/eva-ai/eva-ai.env")
DEFAULT_BACKUP_ROOT = Path("/var/backups/eva-ai")
DEFAULT_UNIT_FILE = Path("/etc/systemd/system/eva-ai.service")
DEFAULT_LOCK_FILE = Path("/run/lock/eva-ai-083-installer.lock")
SIGLIP2_MODEL = "google/siglip2-base-patch16-224"
SIGLIP2_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
DEFAULT_RUNTIME_ROOT = Path("/var/lib/eva-ai")
_LEGACY_OPENAI_CLIP_MODELS = frozenset(
    {
        "vit-b/32",
        "vit-b-32",
        "openai/clip-vit-base-patch32",
    }
)

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SECRET_MARKERS = ("PASSWORD", "SECRET", "TOKEN", "API_KEY", "DSN", "DATABASE_URL")
_VLM_ENDPOINT_RE = re.compile(r"^EVOSSEARCH_LM_PROFILE_(?!AGENT(?:_|$)).+_BASE_URL$")
_ENV_REFERENCE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_INSTALLER_MANAGED_ENV_KEYS = frozenset(
    {
        "EVOSSEARCH_APP_VERSION",
        "EVOSSEARCH_UI_MODE",
        "EVOSSEARCH_ARCHIVE_RETENTION_ENABLED",
        # These are release-owned embedding-space coordinates.  They may be
        # replaced only when prepare_env_values has positively identified the
        # 0.8.1 OpenAI CLIP default; arbitrary site-selected models/caches are
        # still preserved.
        "EVOSSEARCH_PRODUCTION_CLIP_MODEL",
        "EVOSSEARCH_CLIP_MODEL",
        "EVOSSEARCH_CLIP_MODEL_REVISION",
        # These runtime gates are part of the same release-owned SigLIP2
        # migration.  prepare_env_values only updates them after ruling out a
        # site-selected embedding model, so legacy false/auto values must be
        # replaceable instead of surviving as disconnected probe settings.
        "EVOSSEARCH_CLIP_DEVICE",
        "EVOSSEARCH_EMBEDDER_REQUIRED",
        "EVOSSEARCH_EXPERIMENTAL_EMBEDDERS_ENABLED",
        "EVOSSEARCH_EMBEDDER_FALLBACK_ENABLED",
        "EVOSSEARCH_PROBE_POS_FLOOR_DEFAULT",
        "EVOSSEARCH_PROBE_MARGIN_DEFAULT",
    }
)

_INFERENCE_POLICY_PREFIXES = (
    "EVOSSEARCH_LM_",
    "EVOSSEARCH_AGENT_",
    "EVOSSEARCH_INFERENCE_",
)
_INFERENCE_POLICY_EXACT_KEYS = frozenset({"CUDA_VISIBLE_DEVICES"})

_ARCHIVE_RETENTION_POLICY_KEYS = (
    "EVOSSEARCH_ARCHIVE_RETENTION_ENABLED",
    "EVOSSEARCH_ARCHIVE_ROW_RETENTION_DAYS",
    "EVOSSEARCH_ARCHIVE_THUMBNAIL_RETENTION_DAYS",
)


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
    verify_luxriot_credential: bool = False


@dataclass
class PreparedInstall:
    options: InstallerOptions
    env: EnvResolution
    values: dict[str, str]
    updates: dict[str, str]
    migration_dsn: str | None = field(default=None, repr=False)
    migration_dsn_source: str | None = None
    inference_policy_hash: str | None = field(default=None, repr=False)
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
        content += "# Added by the EVA AI universal offline installer; existing site keys were preserved.\n"
        for key, value in pending_updates.items():
            if not _ENV_KEY_RE.fullmatch(key):
                raise InstallerError(f"Unsafe environment key: {key!r}")
            content += f"{key}={_quote_env_value(str(value))}\n"
    return content


def inference_policy_fingerprint(values: Mapping[str, str]) -> str:
    """Hash the complete EVA-to-inference contract without exposing secrets.

    An upgrade may inspect configured OpenAI-compatible endpoints, but it must
    not alter their addresses, models, API keys, timeouts, concurrency,
    context limits, queue policy, video budgets, or GPU visibility. Hashing
    parsed values makes the guard insensitive to comments and quoting that do
    not affect runtime behavior.
    """

    selected = {
        str(key): str(value)
        for key, value in values.items()
        if (
            str(key) in _INFERENCE_POLICY_EXACT_KEYS
            or str(key).startswith(_INFERENCE_POLICY_PREFIXES)
        )
    }
    canonical = "".join(f"{key}={selected[key]}\n" for key in sorted(selected))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _assert_inference_policy_file(
    path: Path,
    expected_hash: str,
    *,
    phase: str,
) -> None:
    if not path.is_file():
        raise InstallerError(
            f"Inference policy guard failed during {phase}: environment file is missing."
        )
    actual_hash = inference_policy_fingerprint(parse_env_text(path.read_text(encoding="utf-8")))
    if actual_hash != expected_hash:
        raise InstallerError(
            f"Inference policy changed during {phase}; refusing to continue."
        )


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
            "EVOSSEARCH_OFFLINE_MODE": "true",
            "EVOSSEARCH_MODEL_CACHE_DIR": "/var/lib/eva-ai/models/huggingface",
            "EVOSSEARCH_OPENAI_CLIP_CACHE_DIR": "/var/lib/eva-ai/models/clip",
            "EVOSSEARCH_TRUSTED_PROXY_HOPS": "1",
            "EVOSSEARCH_PROBE_POS_FLOOR_DEFAULT": "0.05",
            "EVOSSEARCH_PROBE_MARGIN_DEFAULT": "0.02",
            "EVOSSEARCH_PROBE_CAPTURE_WARMUP_SEC": "2.5",
            "EVOSSEARCH_ARCHIVE_DISK_MIN_FREE_GB": "2.0",
            "EVOSSEARCH_ARCHIVE_DISK_MIN_FREE_PERCENT": "5.0",
            "EVOSSEARCH_ARCHIVE_RETENTION_ENABLED": "true",
            "EVOSSEARCH_ARCHIVE_ROW_RETENTION_DAYS": "90",
            "EVOSSEARCH_ARCHIVE_THUMBNAIL_RETENTION_DAYS": "14",
            "EVOSSEARCH_INFERENCE_QUEUE_ENABLED": "true",
            "EVOSSEARCH_INFERENCE_QUEUE_SPOOL_DIR": "/var/lib/eva-ai/inference-spool",
            "EVOSSEARCH_INFERENCE_QUEUE_CAPACITY": "200",
            "EVOSSEARCH_INFERENCE_WORKER_COUNT": "3",
            "EVOSSEARCH_DB_STRICT_RUNTIME_ROLES": "true",
            "EVOSSEARCH_ARCHIVE_STORE": "postgres",
            "EVOSSEARCH_EMBEDDER": "clip",
            "EVOSSEARCH_DINO_SEGMENTS_ENABLED": "false",
            "EVOSSEARCH_EXPERIMENTAL_EMBEDDERS_ENABLED": "true",
            "EVOSSEARCH_PRODUCTION_CLIP_MODEL": SIGLIP2_MODEL,
            "EVOSSEARCH_CLIP_MODEL": SIGLIP2_MODEL,
            "EVOSSEARCH_CLIP_MODEL_REVISION": SIGLIP2_REVISION,
            "EVOSSEARCH_CLIP_DEVICE": "cuda",
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
    elif not any(
        str(resolution.existing.get(key) or "").strip()
        for key in _ARCHIVE_RETENTION_POLICY_KEYS
    ):
        # Releases before the PostgreSQL archive introduced retention did not
        # carry these keys.  Letting a new default silently become active on
        # first boot can erase months of otherwise valid visual evidence.  An
        # upgrade therefore starts conservatively and asks the administrator
        # to opt in to a reviewed retention window later.
        values["EVOSSEARCH_ARCHIVE_RETENTION_ENABLED"] = "false"
        updates["EVOSSEARCH_ARCHIVE_RETENTION_ENABLED"] = "false"
    for key, value in defaults.items():
        add_missing(key, value)

    # 0.8.7 makes SigLIP2 the production semantic space.  Missing coordinates
    # are appended.  The known 0.8.1 OpenAI CLIP default is an explicit release
    # migration and is replaced atomically with the environment backup; an
    # arbitrary site-selected model/cache remains untouched.
    configured_clip_models = [
        str(values.get(key) or "").strip()
        for key in ("EVOSSEARCH_PRODUCTION_CLIP_MODEL", "EVOSSEARCH_CLIP_MODEL")
        if str(values.get(key) or "").strip()
    ]
    custom_clip_model = any(
        model.casefold() not in _LEGACY_OPENAI_CLIP_MODELS
        and model.casefold() != SIGLIP2_MODEL.casefold()
        for model in configured_clip_models
    )
    legacy_clip_migration = bool(
        not custom_clip_model
        and any(model.casefold() in _LEGACY_OPENAI_CLIP_MODELS for model in configured_clip_models)
    )
    siglip2_release_managed = not custom_clip_model
    if siglip2_release_managed:
        for key, value in {
            "EVOSSEARCH_MODEL_CACHE_DIR": "/var/lib/eva-ai/models/huggingface",
            "EVOSSEARCH_PRODUCTION_CLIP_MODEL": SIGLIP2_MODEL,
            "EVOSSEARCH_CLIP_MODEL": SIGLIP2_MODEL,
            "EVOSSEARCH_CLIP_MODEL_REVISION": SIGLIP2_REVISION,
            "EVOSSEARCH_CLIP_DEVICE": "cuda",
            "EVOSSEARCH_EMBEDDER_FALLBACK_ENABLED": "false",
            # The 0.8.7 operator probe contract is a continuous one-Hz
            # semantic stream.  Older site environments do not contain these
            # coordinates because the feature did not exist yet.  Append the
            # release defaults without replacing an administrator's explicit
            # attention policy.
            "EVOSSEARCH_LUXRIOT_ATTENTION_SCHEDULER_ENABLED": "true",
            "EVOSSEARCH_LUXRIOT_ATTENTION_EPISODE_DISPATCH_ENABLED": "false",
            "EVOSSEARCH_LUXRIOT_ATTENTION_EMBED_ALL_CHANNELS": "true",
            "EVOSSEARCH_LUXRIOT_ATTENTION_EMBEDDING_CADENCE_MS": "1000",
            "EVOSSEARCH_LUXRIOT_ATTENTION_STORAGE_ENABLED": "true",
            "EVOSSEARCH_SEMANTIC_SNAPSHOT_ARCHIVE_ENABLED": "true",
            "EVOSSEARCH_PROBE_REALTIME_BOOKMARK_ENABLED": "true",
        }.items():
            add_missing(key, value)
        # SigLIP2 is no longer an optional experimental sidecar in this
        # release.  A preserved 0.8.1 "light runtime" value must not make the
        # updater report success while probes are silently disconnected.
        for key in (
            "EVOSSEARCH_EMBEDDER_REQUIRED",
            "EVOSSEARCH_EXPERIMENTAL_EMBEDDERS_ENABLED",
        ):
            if str(values.get(key) or "").strip().lower() != "true":
                values[key] = "true"
                updates[key] = "true"
        if str(values.get("EVOSSEARCH_CLIP_DEVICE") or "").strip().lower() == "auto":
            values["EVOSSEARCH_CLIP_DEVICE"] = "cuda"
            updates["EVOSSEARCH_CLIP_DEVICE"] = "cuda"
    if legacy_clip_migration:
        for key in ("EVOSSEARCH_PRODUCTION_CLIP_MODEL", "EVOSSEARCH_CLIP_MODEL"):
            current = str(values.get(key) or "").strip()
            if not current or current.casefold() in _LEGACY_OPENAI_CLIP_MODELS:
                values[key] = SIGLIP2_MODEL
                updates[key] = SIGLIP2_MODEL
        values["EVOSSEARCH_CLIP_MODEL_REVISION"] = SIGLIP2_REVISION
        updates["EVOSSEARCH_CLIP_MODEL_REVISION"] = SIGLIP2_REVISION
        values["EVOSSEARCH_EMBEDDER_FALLBACK_ENABLED"] = "false"
        updates["EVOSSEARCH_EMBEDDER_FALLBACK_ENABLED"] = "false"
        for key, legacy_default, siglip_default in (
            ("EVOSSEARCH_PROBE_POS_FLOOR_DEFAULT", "0.28", "0.05"),
            ("EVOSSEARCH_PROBE_MARGIN_DEFAULT", "0.08", "0.02"),
        ):
            current = str(values.get(key) or "").strip()
            if not current or current == legacy_default:
                values[key] = siglip_default
                updates[key] = siglip_default

    # Release identity and the accepted console belong to installed code, not
    # site topology.  External LM endpoints/models, channels, credentials and
    # tenant values remain preserve/append-only.  The legacy console remains available at
    # /?ui=legacy for emergency recovery.
    managed = {
        "EVOSSEARCH_APP_VERSION": EXPECTED_VERSION,
        "EVOSSEARCH_UI_MODE": "react",
    }
    for key, expected in managed.items():
        if str(values.get(key) or "").strip() != expected:
            values[key] = expected
            updates[key] = expected

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
                "(used only by this installer run): "
            ).strip()
            if migration_dsn:
                source = "interactive process-only value"
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


def _verify_luxriot_credential(
    values: Mapping[str, str],
    *,
    timeout_sec: float = 5.0,
) -> tuple[bool, str]:
    """Verify a heuristic-matched site credential without exposing the secret.

    Evo uses HTTP Digest authentication.  This read-only inventory request is
    deliberately opt-in: a credential that merely *looks* like a placeholder
    may be accepted only when the configured endpoint authenticates it.
    """

    base_url = str(values.get("EVOSSEARCH_LUXRIOT_BASE_URL") or "").strip().rstrip("/")
    username = str(values.get("EVOSSEARCH_LUXRIOT_USERNAME") or "").strip()
    password = str(values.get("EVOSSEARCH_LUXRIOT_PASSWORD") or "")
    if not base_url or not username or not password:
        return False, "missing_config"
    if not _valid_http_url(base_url):
        return False, "invalid_url"
    target = f"{base_url}/channels?health=0"
    password_manager = HTTPPasswordMgrWithDefaultRealm()
    password_manager.add_password(None, base_url + "/", username, password)
    opener = build_opener(HTTPDigestAuthHandler(password_manager))
    request = Request(target, headers={"Accept": "application/json"})
    try:
        with opener.open(request, timeout=max(1.0, float(timeout_sec))) as response:
            status = int(getattr(response, "status", response.getcode()))
    except Exception as exc:
        return False, type(exc).__name__
    return 200 <= status < 400, f"http_{status}"


def _wheelhouse_files(bundle_dir: Path) -> list[Path]:
    wheelhouse = bundle_dir / "wheelhouse"
    if not wheelhouse.is_dir():
        return []
    patterns = ("*.whl", "*.tar.gz", "*.zip")
    return sorted({path for pattern in patterns for path in wheelhouse.glob(pattern)})


def _siglip2_cache_snapshot(cache_root: Path, revision: str) -> Path | None:
    """Return a complete local SigLIP2 snapshot, never consulting the network."""

    repository = cache_root / "models--google--siglip2-base-patch16-224"
    snapshots = repository / "snapshots"
    candidates: list[Path] = []
    if revision:
        candidates.append(snapshots / revision)
    elif snapshots.is_dir():
        candidates.extend(path for path in snapshots.iterdir() if path.is_dir())
    for candidate in candidates:
        if (candidate / "config.json").is_file() and any(
            candidate.glob("*.safetensors")
        ):
            return candidate
    return None


def _siglip2_cache_findings(cache_root: Path, revision: str) -> list[Finding]:
    """Verify the complete, checksummed offline semantic-model payload."""

    checksum_file = cache_root / "SHA256SUMS"
    snapshot = _siglip2_cache_snapshot(cache_root, revision)
    if snapshot is None:
        return [Finding(
            "FAIL",
            f"offline bundle has no complete SigLIP2 snapshot for revision {revision}",
        )]
    required_snapshot_files = (
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    repository = cache_root / "models--google--siglip2-base-patch16-224"
    try:
        repository_root = repository.resolve(strict=True)
        for name in required_snapshot_files:
            candidate = snapshot / name
            resolved = candidate.resolve(strict=True)
            if repository_root != resolved and repository_root not in resolved.parents:
                raise ValueError(f"snapshot link escapes model repository: {name}")
            if not candidate.is_file() or candidate.stat().st_size <= 0:
                raise ValueError(f"required snapshot file is empty: {name}")
        if not checksum_file.is_file():
            raise ValueError("SHA256SUMS is missing")
        checked = 0
        for raw_line in checksum_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.+)", line)
            if match is None:
                raise ValueError("SHA256SUMS contains an invalid row")
            expected, relative = match.groups()
            candidate = cache_root / relative
            resolved = candidate.resolve(strict=True)
            cache_root_resolved = cache_root.resolve(strict=True)
            if cache_root_resolved != resolved and cache_root_resolved not in resolved.parents:
                raise ValueError(f"checksum path escapes model cache: {relative}")
            digest = hashlib.sha256()
            with candidate.open("rb") as handle:
                for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest().lower() != expected.lower():
                raise ValueError(f"checksum mismatch: {relative}")
            checked += 1
        if checked == 0:
            raise ValueError("SHA256SUMS is empty")
    except (OSError, RuntimeError, ValueError) as exc:
        return [Finding("FAIL", f"bundled SigLIP2 verification failed: {exc}")]
    return [Finding(
        "OK",
        f"offline SigLIP2 revision {revision} verified ({checked} checksums)",
    )]


def _siglip2_runtime_findings(
    options: InstallerOptions,
    values: Mapping[str, str],
) -> list[Finding]:
    """Prove SigLIP2 is usable now or installable offline before stopping EVA."""

    model = str(values.get("EVOSSEARCH_CLIP_MODEL") or "").strip().lower()
    if "siglip2" not in model:
        return []
    clip_device = str(
        values.get("EVOSSEARCH_CLIP_DEVICE") or "cuda"
    ).strip().lower()
    if not clip_device.startswith("cuda"):
        return [Finding(
            "FAIL",
            "release-managed SigLIP2 requires EVOSSEARCH_CLIP_DEVICE=cuda; "
            f"found {clip_device or 'unset'}",
        )]
    visible_devices = str(values.get("CUDA_VISIBLE_DEVICES") or "").strip().lower()
    if visible_devices in {"-1", "none", "void"}:
        return [Finding(
            "FAIL",
            "release-managed SigLIP2 cannot start while CUDA_VISIBLE_DEVICES "
            f"hides every GPU ({visible_devices}); select a reviewed GPU before apply",
        )]
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return [Finding(
            "FAIL",
            "release-managed SigLIP2 requires a working NVIDIA driver, but "
            "nvidia-smi is unavailable; repair the host driver before apply",
        )]
    driver_probe = subprocess.run(
        (nvidia_smi, "--query-gpu=index", "--format=csv,noheader"),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if driver_probe.returncode != 0 or not driver_probe.stdout.strip():
        return [Finding(
            "FAIL",
            "release-managed SigLIP2 requires a working NVIDIA driver, but "
            "nvidia-smi found no usable GPU; repair the host driver before apply",
        )]
    python = options.app_dir / ".venv" / "bin" / "python"
    if not python.is_file() or not os.access(python, os.X_OK):
        return []
    probe = subprocess.run(
        (
            str(python),
            "-c",
            (
                "import importlib.util,re; from importlib.metadata import version; "
                "v=version('transformers'); "
                "parts=tuple(int(x) for x in re.findall(r'\\d+', v)[:2]); "
                "assert parts >= (4,52), v; "
                "assert importlib.util.find_spec('transformers.models.siglip2') is not None; "
                "import torch; assert torch.cuda.is_available(), torch.__version__"
            ),
        ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if probe.returncode != 0:
        wheelhouse = options.bundle_dir / "wheelhouse"
        if not _wheelhouse_files(options.bundle_dir):
            return [Finding(
                "FAIL",
                "target venv cannot load the SigLIP2 runtime contract "
                "(requires transformers>=4.52) and the bundle has no wheelhouse",
            )]
        requirement_files = [options.source_dir / "requirements.txt"]
        database_requirements = options.source_dir / "requirements-db.txt"
        if database_requirements.is_file():
            requirement_files.append(database_requirements)
        cuda_requirements = options.source_dir / "requirements-cuda.txt"
        if not cuda_requirements.is_file():
            return [Finding(
                "FAIL",
                "target venv has no CUDA-capable torch and the bundle has no "
                "reviewed requirements-cuda.txt repair contract",
            )]
        requirement_files.append(cuda_requirements)
        common_args = [
            "--dry-run",
            "--no-index",
            "--find-links",
            str(wheelhouse),
        ]
        for requirement_file in requirement_files:
            common_args.extend(("-r", str(requirement_file)))
        commands: list[list[str]] = []
        venv_pip = options.app_dir / ".venv" / "bin" / "pip"
        if venv_pip.is_file() and os.access(venv_pip, os.X_OK):
            commands.append([str(venv_pip), "install", *common_args])
        commands.append([str(python), "-m", "pip", "install", *common_args])
        uv = shutil.which("uv")
        if uv:
            commands.append([
                uv,
                "pip",
                "install",
                "--python",
                str(python),
                *common_args,
            ])
        errors: list[str] = []
        for command in commands:
            resolution = subprocess.run(
                tuple(command),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if resolution.returncode == 0:
                return [Finding(
                    "OK",
                    "target venv needs the SigLIP2 CUDA runtime repair; bundled "
                    "wheelhouse resolves all Python requirements offline",
                )]
            detail_rows = [
                row.strip()
                for row in (resolution.stderr or resolution.stdout or "").splitlines()
                if row.strip()
            ]
            if detail_rows:
                errors.append(detail_rows[-1][:300])
        detail = errors[-1] if errors else "no compatible pip/uv resolver is available"
        return [Finding(
            "FAIL",
            "target venv cannot provide CUDA SigLIP2 and bundled wheelhouse resolution failed: "
            + detail,
        )]
    return [Finding("OK", "target venv supports the SigLIP2 CUDA runtime contract")]


def _media_runtime_findings(bundle_dir: Path) -> list[Finding]:
    """Validate an optional self-contained media payload before apply.

    Patch bundles may omit the payload when a complete wheelhouse/system media
    stack is supplied.  When ``runtime/`` is present, however, it is an atomic
    contract: helper, FFmpeg binaries, one OpenCV wheel and every recorded
    checksum must be valid before the live service is stopped.
    """

    runtime_dir = bundle_dir / "runtime"
    if not runtime_dir.exists():
        return [Finding("WARN", "bundle has no self-contained media runtime; existing dependencies will be reused")]
    required = (
        bundle_dir / "scripts" / "install_media_runtime.sh",
        runtime_dir / "SHA256SUMS",
        runtime_dir / "manifest.txt",
        runtime_dir / "ffmpeg" / "bin" / "ffmpeg",
        runtime_dir / "ffmpeg" / "bin" / "ffprobe",
        runtime_dir / "ffmpeg" / "LICENSE.txt",
    )
    failures = [path for path in required if not path.is_file()]
    if failures:
        return [
            Finding("FAIL", f"bundled media runtime path is missing: {path}")
            for path in failures
        ]
    helper = required[0]
    if not os.access(helper, os.X_OK):
        return [Finding("FAIL", f"bundled media installer is not executable: {helper}")]
    wheels = sorted((runtime_dir / "opencv").glob("opencv_python_headless-*.whl"))
    if len(wheels) != 1:
        return [Finding("FAIL", "bundled media runtime must contain exactly one OpenCV wheel")]

    checksum_file = runtime_dir / "SHA256SUMS"
    try:
        records = checksum_file.read_text(encoding="utf-8").splitlines()
        checked = 0
        for record in records:
            expected, separator, relative = record.strip().partition("  ")
            if not separator or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
                raise ValueError("malformed SHA256SUMS record")
            candidate = (runtime_dir / relative).resolve(strict=False)
            if runtime_dir.resolve(strict=False) not in candidate.parents or not candidate.is_file():
                raise ValueError(f"unsafe or missing checksum target: {relative}")
            digest = hashlib.sha256()
            with candidate.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual = digest.hexdigest()
            if actual.lower() != expected.lower():
                raise ValueError(f"checksum mismatch: {relative}")
            checked += 1
        if checked == 0:
            raise ValueError("SHA256SUMS is empty")
    except (OSError, ValueError) as exc:
        return [Finding("FAIL", f"bundled media runtime verification failed: {exc}")]
    return [Finding("OK", f"self-contained media runtime verified ({checked} checksums)")]


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
        options.source_dir / "react-ui" / "dist" / "index.html",
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
    findings.extend(_media_runtime_findings(options.bundle_dir))
    findings.extend(_siglip2_runtime_findings(options, values))
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
            elif key == "EVOSSEARCH_LUXRIOT_PASSWORD" and options.verify_luxriot_credential:
                verified, status = _verify_luxriot_credential(values)
                if verified:
                    add(
                        "WARN",
                        "EVOSSEARCH_LUXRIOT_PASSWORD matched the placeholder heuristic "
                        "but was accepted after an authenticated read-only Evo /channels check",
                    )
                else:
                    add(
                        "FAIL",
                        "EVOSSEARCH_LUXRIOT_PASSWORD matched the placeholder heuristic and "
                        f"the authenticated Evo check failed ({status})",
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
            "existing environment will be copied with only reviewed release-owned keys migrated: "
            f"{resolution.source} -> {resolution.target}",
        )
    else:
        add("OK", f"existing environment will be preserved in place: {resolution.target}")
    if resolution.existing and not any(
        str(resolution.existing.get(key) or "").strip()
        for key in _ARCHIVE_RETENTION_POLICY_KEYS
    ):
        add(
            "WARN",
            "legacy configuration has no explicit archive retention policy; "
            "automatic pruning will be disabled during upgrade to prevent silent evidence loss",
        )

    target_venv = options.app_dir / ".venv" / "bin" / "python"
    wheels = _wheelhouse_files(options.bundle_dir)
    if wheels:
        add("OK", f"offline wheelhouse contains {len(wheels)} artifact(s)")
    elif target_venv.is_file() and os.access(target_venv, os.X_OK):
        add("WARN", "no wheelhouse found; existing target venv would be reused without downloads")
    else:
        add("FAIL", "fresh install requires bundle/wheelhouse; online dependency downloads are forbidden")

    clip_model = str(values.get("EVOSSEARCH_CLIP_MODEL") or "").strip().lower()
    if "siglip2" in clip_model:
        revision = str(values.get("EVOSSEARCH_CLIP_MODEL_REVISION") or "").strip()
        target_cache = Path(
            str(values.get("EVOSSEARCH_MODEL_CACHE_DIR") or "/var/lib/eva-ai/models/huggingface")
        ).expanduser()
        bundled_cache = options.bundle_dir / "models" / "huggingface"
        target_snapshot = _siglip2_cache_snapshot(target_cache, revision)
        findings.extend(_siglip2_cache_findings(bundled_cache, revision))
        if target_snapshot is not None:
            add("OK", f"SigLIP2 is already cached locally at {target_snapshot}")
        else:
            add("OK", f"SigLIP2 will be installed into local cache {target_cache}")

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
    if options.migrate and shutil.which("pg_restore") is None:
        add("FAIL", "pg_restore is required to validate the pre-migration dump")
    if options.migrate and shutil.which("psql") is None:
        add("FAIL", "psql is required to prove migration-table access before apply")
    if options.dry_run:
        detail = "dry-run mode: no filesystem, database, or service state will change"
        if options.verify_luxriot_credential:
            detail += "; Luxriot credential verification uses a read-only network request"
        add("OK", detail)
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
        f"preserve {prepared.env.source} and apply {len(prepared.updates)} reviewed release default/migration key(s)"
        if prepared.env.source_kind == "in-place"
        else (
            f"copy {prepared.env.source} to {prepared.env.target}, preserving site keys and applying reviewed release migrations"
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
            "inference",
            (
                "preserve the complete existing external agent/VLM inference policy; reject any "
                "endpoint/model/context/queue/GPU rewrite before startup"
                if prepared.inference_policy_hash is not None
                else "create the reviewed inference policy for a fresh installation"
            ),
        ),
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
            (
                "prove privileged PostgreSQL access to public.alembic_version, then run "
                "scripts/preflight_patch.sh without stopping the service"
                if options.migrate
                else "run existing scripts/preflight_patch.sh without stopping the service"
            ),
        ),
        PlanAction(
            "install",
            "run existing scripts/install_patch.sh for code/env/unit/DB backup and static copy; keep service stopped",
        ),
        PlanAction(
            "media",
            "install the checksummed offline FFmpeg runtime and add the bundled OpenCV overlay only when the target venv lacks it",
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
        actions.append(PlanAction(
            "health",
            (
                "run scripts/verify_patch.sh for /health and /ready; once the new service is active, "
                "report dependency failures for in-place repair without automatic rollback"
            ),
        ))
    actions.append(PlanAction(
        "rollback",
        (
            "allow automatic rollback only before the new service becomes active; hand off the explicit "
            f"manual scripts/rollback.sh command using {options.backup_root}/LATEST for disaster recovery"
        ),
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


def _ensure_runtime_directories(
    values: Mapping[str, str],
    *,
    user: str,
    group: str,
) -> list[Path]:
    uid = pwd.getpwnam(user).pw_uid
    gid = grp.getgrnam(group).gr_gid
    created: list[Path] = []
    for key in (
        "EVOSSEARCH_MODEL_CACHE_DIR",
        "EVOSSEARCH_OPENAI_CLIP_CACHE_DIR",
        "EVOSSEARCH_INFERENCE_QUEUE_SPOOL_DIR",
    ):
        raw = str(values.get(key) or "").strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        if not path.is_absolute() or path == Path("/"):
            raise InstallerError(
                f"{key} must be a safe absolute runtime directory"
            )
        managed_paths = [path]
        if path == DEFAULT_RUNTIME_ROOT or DEFAULT_RUNTIME_ROOT in path.parents:
            relative = path.relative_to(DEFAULT_RUNTIME_ROOT)
            managed_paths = [DEFAULT_RUNTIME_ROOT]
            current = DEFAULT_RUNTIME_ROOT
            for part in relative.parts:
                current /= part
                managed_paths.append(current)
        for managed_path in managed_paths:
            managed_path.mkdir(parents=True, exist_ok=True, mode=0o750)
            os.chown(managed_path, uid, gid)
            os.chmod(managed_path, 0o750)
        created.append(path)
    return created


def _ensure_runtime_env_access(path: Path, *, user: str, group: str) -> None:
    """Keep secrets private while making a legacy env readable by its service."""

    if not path.is_file():
        raise InstallerError(f"Runtime environment file is missing: {path}")
    uid = pwd.getpwnam(user).pw_uid
    gid = grp.getgrnam(group).gr_gid
    parent = path.parent
    parent_stat = parent.stat()
    # Preserve the legacy administrator owner while granting only the selected
    # service group enough access to traverse the configuration directory.
    os.chown(parent, parent_stat.st_uid, gid)
    os.chmod(parent, 0o750)
    # The EVA process reads this file during config bootstrap and may persist
    # reviewed Settings changes.  No account except the service identity (and
    # root) should be able to read the embedded credentials.
    os.chown(path, uid, gid)
    os.chmod(path, 0o600)


def _backup_file(path: Path) -> Path | None:
    if not path.is_file():
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.preinstall-{timestamp}.bak")
    _copy2_with_ownership(path, backup)
    return backup


def _copy2_with_ownership(source: Path, destination: Path) -> None:
    """Copy a file as root without changing its original owner/group."""

    source_stat = source.stat()
    shutil.copy2(source, destination)
    os.chown(destination, source_stat.st_uid, source_stat.st_gid)


def _migration_environment(prepared: PreparedInstall) -> dict[str, str]:
    env = dict(os.environ)
    env.update(prepared.values)
    if not prepared.migration_dsn:
        raise InstallerError("A distinct privileged migration DSN is required")
    env["EVA_DATABASE_DSN"] = prepared.migration_dsn
    return env


def _verify_migration_capability(
    runner: CommandRunner,
    migration_dsn: str,
) -> None:
    """Exercise Alembic's revision-table privileges without lasting mutation."""

    sql = """
        BEGIN;
        SELECT version_num FROM public.alembic_version LIMIT 1;
        UPDATE public.alembic_version SET version_num = version_num;
        SET LOCAL row_security = off;
        DO $eva_row_visibility$
        DECLARE
            protected record;
        BEGIN
            FOR protected IN
                SELECT 1
                     , namespace.nspname AS schema_name
                     , relation.relname AS table_name
                FROM pg_class relation
                JOIN pg_namespace namespace ON namespace.oid = relation.relnamespace
                WHERE namespace.nspname = ANY(ARRAY['agent', 'archive', 'audit', 'iam', 'jobs'])
                  AND relation.relkind IN ('r', 'p')
                  AND relation.relforcerowsecurity
            LOOP
                EXECUTE format(
                    'SELECT 1 FROM %I.%I LIMIT 1',
                    protected.schema_name,
                    protected.table_name
                );
            END LOOP;
        END
        $eva_row_visibility$;
        SET LOCAL ROLE eva_owner;
        DO $eva_preflight$
        DECLARE
            expected_schemas text[] := ARRAY['agent', 'archive', 'audit', 'iam', 'jobs'];
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM unnest(expected_schemas) AS expected(name)
                LEFT JOIN pg_namespace namespace ON namespace.nspname = expected.name
                WHERE namespace.oid IS NULL
                   OR pg_get_userbyid(namespace.nspowner) <> current_user
            ) THEN
                RAISE EXCEPTION 'EVA schemas are absent or are not owned by eva_owner';
            END IF;
            IF EXISTS (
                SELECT 1
                FROM pg_class relation
                JOIN pg_namespace namespace ON namespace.oid = relation.relnamespace
                WHERE namespace.nspname = ANY(expected_schemas)
                  AND relation.relkind IN ('r', 'p', 'S', 'v', 'm', 'i')
                  AND pg_get_userbyid(relation.relowner) <> current_user
            ) THEN
                RAISE EXCEPTION 'EVA relations are not owned by eva_owner';
            END IF;
        END
        $eva_preflight$;
        CREATE TABLE archive.__eva_migration_preflight (id integer);
        ROLLBACK;
    """
    try:
        postgres_env = postgres_environment(migration_dsn, os.environ)
    except DsnError as exc:
        raise InstallerError(
            "privileged migration DSN cannot be represented safely for libpq"
        ) from exc
    try:
        runner.run((
            "psql",
            "--no-psqlrc",
            "--set", "ON_ERROR_STOP=1",
            "--command", sql,
        ), env=postgres_env)
    except InstallerError as exc:
        raise InstallerError(
            "privileged migration DSN cannot update public.alembic_version, "
            "see rows behind FORCE ROW LEVEL SECURITY, SET ROLE eva_owner, or "
            "modify EVA-owned schemas; live files and service were not changed"
        ) from exc


def _automatic_rollback(
    prepared: PreparedInstall,
    runner: CommandRunner,
    backup_dir: Path,
) -> bool:
    """Restore the last runnable snapshot after any post-backup apply failure."""

    options = prepared.options
    command: list[str | Path] = [
        options.source_dir / "scripts" / "rollback.sh",
        "--backup-dir", backup_dir,
        "--backup-root", options.backup_root,
        "--app-dir", options.app_dir,
        "--env-file", prepared.env.target,
        "--service", options.service_name,
        "--base-url", options.base_url,
        "--no-verify",
    ]
    rollback_env = dict(os.environ)
    if options.migrate:
        command.append("--restore-db")
        rollback_env["EVA_PATCH_CONFIRM_DB_RESTORE"] = "yes"
        if prepared.migration_dsn:
            rollback_env["EVA_PATCH_PG_DSN"] = prepared.migration_dsn
    if not options.start:
        command.append("--no-start")
    try:
        runner.run(command, env=rollback_env)
    except Exception as exc:
        print(
            "AUTOMATIC ROLLBACK FAILED: previous code/database could not be fully "
            f"restored ({type(exc).__name__}); backup_dir={backup_dir}",
            file=sys.stderr,
        )
        if options.start:
            subprocess.run(
                ("systemctl", "start", options.service_name + ".service"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        return False
    # Only claim what this invocation actually restored: --restore-db is passed
    # for migrating updates alone, and a code-only rollback leaves the database
    # untouched by design.
    restored = (
        "previous code, configuration and database"
        if options.migrate
        else "previous code and configuration (database was not part of this rollback)"
    )
    print(
        f"AUTOMATIC ROLLBACK COMPLETE: {restored} were restored from {backup_dir}",
        file=sys.stderr,
    )
    return True


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


def _latest_backup_marker(backup_root: Path) -> str:
    latest = backup_root / "LATEST"
    if not latest.is_file():
        return ""
    try:
        raw = latest.read_text(encoding="utf-8").strip()
        return str(Path(raw).resolve(strict=False)) if raw else ""
    except OSError:
        return ""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_database_backup(backup_dir: Path, runner: CommandRunner) -> Path:
    """Prove the quiescent pre-migration dump is present and self-consistent."""

    dump = backup_dir / "postgres.dump"
    checksum = backup_dir / "postgres.dump.sha256"
    if not dump.is_file() or dump.stat().st_size <= 0:
        raise InstallerError(
            "PostgreSQL backup is absent; refusing to run migrations. "
            "Fix pg_dump/permissions and retry."
        )
    if not checksum.is_file():
        raise InstallerError(
            "PostgreSQL backup checksum is absent; refusing to run migrations."
        )
    row = checksum.read_text(encoding="utf-8").strip().split()
    if len(row) != 2 or row[1] != "postgres.dump" or not re.fullmatch(r"[0-9a-fA-F]{64}", row[0]):
        raise InstallerError("PostgreSQL backup checksum manifest is invalid")
    if _sha256_file(dump) != row[0].lower():
        raise InstallerError("PostgreSQL backup checksum verification failed")
    runner.run(("pg_restore", "--list", "--file=/dev/null", dump))
    return dump


def _venv_has_healthy_opencv(python: Path) -> bool:
    """Probe only the target venv, deliberately excluding an old overlay."""

    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        (
            str(python),
            "-c",
            (
                "import cv2,numpy as np; "
                "image=np.zeros((8,8,3),dtype=np.uint8); "
                "assert cv2.cvtColor(image,cv2.COLOR_BGR2RGB).shape==(8,8,3)"
            ),
        ),
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def _install_media_runtime(prepared: PreparedInstall, runner: CommandRunner) -> bool:
    """Install an included media payload after backup, before DB/service work."""

    options = prepared.options
    runtime_dir = options.bundle_dir / "runtime"
    if not runtime_dir.is_dir():
        return False
    python = options.app_dir / ".venv" / "bin" / "python"
    command: list[str | Path] = [
        options.bundle_dir / "scripts" / "install_media_runtime.sh",
        "--bundle-dir", options.bundle_dir,
        "--app-dir", options.app_dir,
        "--python", python,
        "--owner", f"{options.service_user}:{options.service_group}",
    ]
    if not _venv_has_healthy_opencv(python):
        command.append("--with-opencv-overlay")
    runner.run(command)
    return True


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

    if prepared.inference_policy_hash is not None:
        policy_source = prepared.env.target if prepared.env.target.is_file() else prepared.env.source
        if policy_source is None:
            raise InstallerError(
                "Inference policy guard has no environment source to verify before apply."
            )
        _assert_inference_policy_file(
            policy_source,
            prepared.inference_policy_hash,
            phase="apply preflight",
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
    latest_before_apply = _latest_backup_marker(options.backup_root)
    runtime_started = False

    try:
        # This must precede even the env staging below.  Alembic itself needs
        # access to its revision row before migration scripts can SET ROLE.
        if options.migrate:
            _verify_migration_capability(runner, migration_dsn)

        # A code-only adopt upgrade must not silently replace the site's
        # reviewed service identity or hardening.  Existing units are preserved;
        # the installer account/template are only for a fresh service.
        if not unit_preexisted:
            _ensure_service_account(options, runner)
        options.app_dir.mkdir(parents=True, exist_ok=True)
        options.backup_root.mkdir(parents=True, exist_ok=True)
        # Existing 0.8.1 environments receive a release-managed SigLIP cache
        # during migration too.  Ensure both its leaf and the EVA-owned
        # /var/lib/eva-ai ancestors are traversable by the service account;
        # otherwise the checksummed model is present but runtime loading fails
        # with a misleading embedding-model error.
        _ensure_runtime_directories(
            prepared.values,
            user=options.service_user,
            group=options.service_group,
        )
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
        if prepared.inference_policy_hash is not None:
            _assert_inference_policy_file(
                prepared.env.target,
                prepared.inference_policy_hash,
                phase="environment staging",
            )

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
        install_command: list[str | Path] = [
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
        ]
        if options.migrate:
            install_command.append("--require-pg-dump")
        runner.run(install_command, env=install_env)
        backup_dir = _latest_backup(options.backup_root)
        if prepared.inference_policy_hash is not None:
            _assert_inference_policy_file(
                prepared.env.target,
                prepared.inference_policy_hash,
                phase="code installation",
            )
        if env_preexisted and env_preinstall_backup is not None:
            _copy2_with_ownership(env_preinstall_backup, backup_dir / "eva-ai.env")
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

        # ``oldapp.py`` imports cv2 during process bootstrap.  The universal
        # bundle keeps rescue OpenCV out of the site's legacy venv and exposes
        # it through .eva-runtime/python in run_prod.sh.  This must happen
        # before migrations and, critically, before the first service start.
        _install_media_runtime(prepared, runner)

        if options.migrate:
            _validate_database_backup(backup_dir, runner)
            alembic = options.app_dir / ".venv" / "bin" / "alembic"
            if not alembic.is_file():
                raise InstallerError(f"Alembic is missing after dependency setup: {alembic}")
            migration_env = _migration_environment(prepared)
            preservation_guard = options.app_dir / "scripts" / "database_preservation_guard.py"
            if not preservation_guard.is_file():
                raise InstallerError(
                    "database preservation guard is absent; refusing to run migrations"
                )
            preservation_manifest = backup_dir / "database-preservation.json"
            runner.run(
                (
                    venv_python,
                    preservation_guard,
                    "capture",
                    "--output",
                    preservation_manifest,
                ),
                cwd=options.app_dir,
                env=migration_env,
            )
            runner.run((alembic, "current"), cwd=options.app_dir, env=migration_env)
            runner.run((alembic, "upgrade", "head"), cwd=options.app_dir, env=migration_env)
            current = runner.run((alembic, "current"), cwd=options.app_dir, env=migration_env)
            if EXPECTED_SCHEMA not in current.stdout:
                raise InstallerError(
                    f"Alembic current did not report expected schema {EXPECTED_SCHEMA}"
                )
            runner.run(
                (
                    venv_python,
                    preservation_guard,
                    "verify",
                    "--input",
                    preservation_manifest,
                ),
                cwd=options.app_dir,
                env=migration_env,
            )

        _ensure_runtime_env_access(
            prepared.env.target,
            user=options.service_user,
            group=options.service_group,
        )
        if options.start:
            runner.run(("systemctl", "enable", options.service_name + ".service"))
            runner.run(("systemctl", "restart", options.service_name + ".service"))
            runner.run(("systemctl", "is-active", options.service_name + ".service"))
            # From this point the new runtime may accept frames and mutate the
            # archive. Dependency acceptance errors are repaired in place;
            # an automatic rollback could discard those new writes.
            runtime_started = True
        if options.verify:
            verify = options.app_dir / "scripts" / "verify_patch.sh"
            runner.run((
                verify,
                "--service", options.service_name,
                "--base-url", options.base_url,
                # A production worker eagerly loads and exercises SigLIP before
                # exposing /health.  Field cold starts have reached 240 seconds
                # on the shared rehearsal host; a shorter verification budget
                # falsely declares a healthy install failed and triggers an
                # automatic rollback while the worker is still warming.
                "--timeout", "300",
            ))
        if prepared.inference_policy_hash is not None:
            _assert_inference_policy_file(
                prepared.env.target,
                prepared.inference_policy_hash,
                phase="post-update verification",
            )
        return backup_dir
    except Exception as exc:
        if runtime_started and backup_dir is not None:
            safe_error = redact_text(str(exc), secret_values)
            warning_path = backup_dir / "post-start-acceptance-warning.txt"
            _atomic_write(
                warning_path,
                (
                    "The new EVA runtime became active. Automatic rollback was disabled "
                    "before dependency acceptance checks.\n"
                    f"{type(exc).__name__}: {safe_error}\n"
                ),
                0o600,
            )
            print(
                "POST-START ACCEPTANCE ERROR: the new EVA runtime is active; "
                "code and database were left in place for repair. "
                f"Details: {warning_path}",
                file=sys.stderr,
            )
            return backup_dir
        if backup_dir is None:
            latest = options.backup_root / "LATEST"
            if latest.is_file():
                try:
                    candidate = _latest_backup(options.backup_root)
                    if str(candidate) != latest_before_apply:
                        backup_dir = candidate
                except InstallerError:
                    backup_dir = None
        # If install_patch never established a rollback snapshot, undo the
        # small amount of staging performed by this orchestrator itself.  If
        # it did establish one but then failed, make sure that snapshot holds
        # the *pre-orchestrator* env rather than the appended staging copy.
        if env_preinstall_backup is not None and env_preinstall_backup.is_file():
            if backup_dir is not None:
                _copy2_with_ownership(env_preinstall_backup, backup_dir / "eva-ai.env")
            elif env_preexisted:
                _copy2_with_ownership(env_preinstall_backup, prepared.env.target)
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
        if backup_dir is None and unit_preexisted and options.start:
            # install_patch now quiesces writers before a required pg_dump. If
            # that dump fails before LATEST is armed, the code/database are
            # unchanged; restore the staged env above and bring the old service
            # back instead of leaving the site dark.
            subprocess.run(
                ("systemctl", "start", options.service_name + ".service"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        rollback_succeeded = False
        if backup_dir is not None:
            rollback_succeeded = _automatic_rollback(prepared, runner, backup_dir)
        if backup_dir is not None and not rollback_succeeded:
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
    parser.add_argument(
        "--verify-luxriot-credential",
        action="store_true",
        help=(
            "When the Evo password matches the placeholder heuristic, accept it only "
            "after an authenticated read-only Digest /channels check."
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
        verify_luxriot_credential=bool(args.verify_luxriot_credential),
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
    inference_policy_hash: str | None = None
    if resolution.source is not None:
        inference_policy_hash = inference_policy_fingerprint(resolution.existing)
        projected = parse_env_text(render_env_update(resolution.raw, updates))
        projected_hash = inference_policy_fingerprint(projected)
        if projected_hash != inference_policy_hash:
            findings.append(Finding(
                "FAIL",
                "upgrade plan would change the existing inference policy; no changes were made",
            ))
        else:
            findings.append(Finding(
                "OK",
                f"existing inference policy is protected by fingerprint {inference_policy_hash}",
            ))
    prepared = PreparedInstall(
        options=options,
        env=resolution,
        values=values,
        updates=updates,
        migration_dsn=migration_dsn,
        migration_dsn_source=migration_dsn_source,
        inference_policy_hash=inference_policy_hash,
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
