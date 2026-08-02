#!/usr/bin/env python3
"""Interactive offline installer for the EVA AI eight-channel appliance.

The script is intentionally standard-library-only.  It is copied to the root
of the field USB and can therefore run on a fresh Ubuntu Server 24.04 host
before any Python packages have been installed.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import json
import os
import platform
import re
import secrets
import shlex
import shutil
import socket
import struct
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import uuid
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.parse import urlsplit


VERSION = "β 0.8.5"
EXPECTED_SCHEMA = "20260801_0011"
DEFAULT_ROOT = Path("/opt/eva-ai")
DEFAULT_DATA = Path("/var/lib/eva-ai")
DEFAULT_CONFIG = Path("/etc/eva-ai")
DEFAULT_BACKUPS = Path("/var/backups/eva-ai")
DEFAULT_INSTALLER_STATE = Path("/var/lib/eva-ai-installer/install-state.json")
DEFAULT_OFFLINE_APT_ROOT = Path("/var/cache/eva-ai-offline-apt")
DEFAULT_VLM_URL = "http://127.0.0.1:1234/v1"
DEFAULT_VLM_MODEL = "qwen/qwen3-vl-4b"
DEFAULT_DEEP_URL = "http://127.0.0.1:1236/v1"
DEFAULT_DEEP_MODEL = "qwen3.5-9b-mtp"
DEFAULT_SIGLIP2_MODEL = "google/siglip2-base-patch16-224"
DEFAULT_SIGLIP2_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
MIN_FREE_GIB = 45

PORT_ENV = {
    "EVOSSEARCH_APP_VERSION": VERSION,
    "EVOSSEARCH_HOST": "127.0.0.1",
    "EVOSSEARCH_PORT": "5000",
    "EVOSSEARCH_DEBUG": "false",
    # Keep the mature console as the appliance default until React parity is
    # accepted. Operators can pilot React at /?ui=react without a restart.
    "EVOSSEARCH_UI_MODE": "legacy",
    "EVOSSEARCH_GUNICORN_WORKERS": "1",
    "EVOSSEARCH_GUNICORN_THREADS": "8",
    "EVOSSEARCH_GUNICORN_TIMEOUT": "240",
    "EVOSSEARCH_GUNICORN_GRACEFUL_TIMEOUT": "20",
    "EVOSSEARCH_OFFLINE_MODE": "true",
    "EVOSSEARCH_SECURE_DEPLOYMENT_REQUIRED": "true",
    "EVOSSEARCH_AUTH_ENABLED": "true",
    "EVOSSEARCH_AUTH_COOKIE_SECURE": "true",
    "EVOSSEARCH_TRUSTED_PROXY_HOPS": "1",
    "EVOSSEARCH_DB_STRICT_RUNTIME_ROLES": "true",
    "EVOSSEARCH_EMBEDDER": "clip",
    "EVOSSEARCH_EMBEDDER_EAGER_LOAD": "true",
    "EVOSSEARCH_INDEX_MODE": "clip",
    "EVOSSEARCH_PRODUCTION_CLIP_MODEL": DEFAULT_SIGLIP2_MODEL,
    "EVOSSEARCH_CLIP_MODEL": DEFAULT_SIGLIP2_MODEL,
    "EVOSSEARCH_CLIP_MODEL_REVISION": DEFAULT_SIGLIP2_REVISION,
    "EVOSSEARCH_EXPERIMENTAL_EMBEDDERS_ENABLED": "true",
    "EVOSSEARCH_CLIP_DEVICE": "cuda",
    "EVOSSEARCH_PROBE_POS_FLOOR_DEFAULT": "0.05",
    "EVOSSEARCH_PROBE_MARGIN_DEFAULT": "0.02",
    "EVOSSEARCH_DINO_SEGMENTS_ENABLED": "false",
    "EVOSSEARCH_M2F_ENABLED": "false",
    # SigLIP2 base needs GPU placement to sustain the eight-channel 1 Hz
    # semantic archive. It shares the 4070 Super with vLLM under a bounded
    # vLLM allocation; CPU is an explicit fallback profile, not the default.
    "CUDA_VISIBLE_DEVICES": "0",
    "EVOSSEARCH_ARCHIVE_STORE": "postgres",
    "EVOSSEARCH_ARCHIVE_MAX_RECORDS": "10000000",
    "EVOSSEARCH_ARCHIVE_ROW_RETENTION_DAYS": "14",
    "EVOSSEARCH_ARCHIVE_THUMBNAIL_RETENTION_DAYS": "3",
    "EVOSSEARCH_ARCHIVE_RETENTION_ENABLED": "true",
    "EVOSSEARCH_ARCHIVE_RETENTION_PRUNE_INTERVAL_SEC": "3600",
    "EVOSSEARCH_ARCHIVE_RETENTION_BATCH_SIZE": "5000",
    "EVOSSEARCH_ARCHIVE_ESTIMATE_CHANNELS": "8",
    "EVOSSEARCH_INFERENCE_QUEUE_ENABLED": "true",
    "EVOSSEARCH_INFERENCE_QUEUE_CAPACITY": "200",
    "EVOSSEARCH_INFERENCE_WORKER_COUNT": "1",
    "EVOSSEARCH_LM_VIDEO_REPETITION_PENALTY": "1.08",
    "EVOSSEARCH_LUXRIOT_CAPTURE_SOURCE": "live_segment",
    "EVOSSEARCH_LUXRIOT_LIVE_SEGMENT_SECONDS": "60",
    "EVOSSEARCH_LUXRIOT_LIVE_SEGMENT_FPS": "4",
    "EVOSSEARCH_LUXRIOT_SUMMARY_MAX_BATCH_FRAMES": "16",
    "EVOSSEARCH_LUXRIOT_SUMMARY_MAX_WINDOW_SEC": "60",
    "EVOSSEARCH_LUXRIOT_SUMMARY_QUIET_CADENCE_SEC": "10",
    "EVOSSEARCH_LUXRIOT_SUMMARY_NORMAL_CADENCE_SEC": "5",
    "EVOSSEARCH_LUXRIOT_SUMMARY_BURST_CADENCE_SEC": "1",
    "EVOSSEARCH_LUXRIOT_SUMMARY_QUEUE_MAX_BATCHES": "2",
    "EVOSSEARCH_LUXRIOT_ATTENTION_SCHEDULER_ENABLED": "true",
    "EVOSSEARCH_LUXRIOT_ATTENTION_EPISODE_DISPATCH_ENABLED": "false",
    "EVOSSEARCH_LUXRIOT_ATTENTION_EMBED_ALL_CHANNELS": "true",
    "EVOSSEARCH_LUXRIOT_ATTENTION_EMBEDDING_CADENCE_MS": "1000",
    "EVOSSEARCH_LUXRIOT_ATTENTION_STORAGE_ENABLED": "true",
    "EVOSSEARCH_LUXRIOT_ATTENTION_REQUESTS_PER_MINUTE": "6",
    "EVOSSEARCH_LUXRIOT_ATTENTION_L0_SLOT_PARALLELISM": "3",
    "EVOSSEARCH_LUXRIOT_ATTENTION_MAX_OUTSTANDING": "1",
    "EVOSSEARCH_LUXRIOT_ATTENTION_RING_SECONDS": "90",
    "EVOSSEARCH_LUXRIOT_ATTENTION_POSTROLL_SEC": "3",
    "EVOSSEARCH_LUXRIOT_ATTENTION_MAX_VLM_FRAMES": "16",
    "EVOSSEARCH_LUXRIOT_CLIP_ASYNC_ENABLED": "true",
    "EVOSSEARCH_LUXRIOT_CLIP_ASYNC_WORKERS": "8",
    "EVOSSEARCH_LUXRIOT_CLIP_ASYNC_QUEUE_CAPACITY": "64",
    "EVOSSEARCH_LIVE_CLIP_BATCH_SIZE": "8",
    "EVOSSEARCH_LIVE_CLIP_BATCH_WAIT_MS": "75",
    "EVOSSEARCH_LIVE_CLIP_BATCH_QUEUE_CAPACITY": "128",
    "EVOSSEARCH_LIVE_CLIP_BATCH_TIMEOUT_SEC": "15",
    "EVOSSEARCH_SEMANTIC_SNAPSHOT_ARCHIVE_ENABLED": "true",
    "EVOSSEARCH_SEMANTIC_SNAPSHOT_ARCHIVE_QUEUE": "512",
    "EVOSSEARCH_SEMANTIC_SNAPSHOT_ARCHIVE_BATCH_SIZE": "32",
    "EVOSSEARCH_PROBE_REALTIME_BOOKMARK_ENABLED": "true",
    "EVOSSEARCH_PROBE_REALTIME_CONFIRM_HITS": "2",
    "EVOSSEARCH_PROBE_REALTIME_CONFIRM_WINDOW_SEC": "3.2",
    "EVOSSEARCH_PROBE_REALTIME_MAX_EVENT_AGE_SEC": "5",
    "EVOSSEARCH_VLM_FAST_ALERT_ENABLED": "true",
    "EVOSSEARCH_VLM_FAST_ALERT_POST_ROLL_SEC": "2.5",
    "EVOSSEARCH_VLM_FAST_ALERT_COOLDOWN_SEC": "12",
    "EVOSSEARCH_VLM_FAST_ALERT_MAX_FRAMES": "6",
    "EVOSSEARCH_VLM_FAST_ALERT_MAX_TOKENS": "128",
    "EVOSSEARCH_VLM_FAST_ALERT_WORKERS": "2",
    "EVOSSEARCH_VLM_FAST_ALERT_SEMANTIC_DELTA": "0.22",
    "EVOSSEARCH_VLM_FAST_ALERT_MIN_MOVING_FRACTION": "0.15",
    "EVOSSEARCH_VLM_FAST_ALERT_DEDUPE_WINDOW_SEC": "12",
    "EVOSSEARCH_LUXRIOT_ROLLUP_SCHEDULER_ENABLED": "true",
    "EVOSSEARCH_LUXRIOT_ROLLUP_LLM_LEVELS": "L1,L2,L3",
    "EVOSSEARCH_LUXRIOT_ROLLUP_LLM_MODEL": "agent",
    "EVOSSEARCH_LUXRIOT_ROLLUP_CONTEXT_LIMIT_TOKENS": "32768",
    "EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_ENABLED": "true",
    "EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_CONNECT_TIMEOUT_SEC": "5",
    "EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_READ_TIMEOUT_SEC": "600",
    "EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_MAX_TOKENS": "3072",
    "EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_TEMPERATURE": "0.1",
    "EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_QUEUE_CAPACITY": "64",
    "EVOSSEARCH_LUXRIOT_ROLLUP_L3_QUIET_WINDOW_ENABLED": "false",
    "EVOSSEARCH_AGENT_CONTEXT_LIMIT_TOKENS": "32768",
    "EVOSSEARCH_AGENT_CONTEXT_WARNING_TOKENS": "26000",
    "EVOSSEARCH_AGENT_CONTEXT_HARD_TOKENS": "30000",
    "EVOSSEARCH_AGENT_CONTEXT_HISTORY_BUDGET_TOKENS": "8000",
    "EVOSSEARCH_AGENT_MAX_OUTPUT_TOKENS": "2048",
    "EVOSSEARCH_OFFLINE_VIDEO_ENABLED": "false",
    "EVOSSEARCH_PROBE_SNAP_ENABLED": "false",
    "EVOSSEARCH_INDEXED_FOLDER_ENABLED": "false",
}

APT_PACKAGES = (
    "ca-certificates",
    "curl",
    "ffmpeg",
    "jq",
    "nginx",
    "openssl",
    "postgresql",
    "postgresql-contrib",
    "python3",
    "python3-pip",
    "python3-venv",
    "rsync",
    "vainfo",
    "intel-media-va-driver-non-free",
    "build-essential",
    "cmake",
    "ninja-build",
)


class InstallError(RuntimeError):
    """Operator-actionable installation failure."""


@dataclass
class InstallJournal:
    """Secret-free, crash-safe record of installer phase boundaries.

    Completed phases are deliberately replayed on a retry.  Every phase in this
    installer is idempotent, and replaying it is safer than trusting a stale
    "completed" marker after a power loss.
    """

    path: Path = DEFAULT_INSTALLER_STATE
    dry_run: bool = False
    secrets_to_redact: tuple[str, ...] = ()
    payload: dict = field(default_factory=dict)

    def add_secrets(self, values: Iterable[str]) -> None:
        additions = tuple(str(value) for value in values if value)
        self.secrets_to_redact = tuple(
            dict.fromkeys((*self.secrets_to_redact, *additions))
        )

    def _safe(self, value: str) -> str:
        safe = str(value)
        for secret in sorted(self.secrets_to_redact, key=len, reverse=True):
            safe = safe.replace(secret, "***")
        return safe

    def _write(self) -> None:
        if self.dry_run:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        _atomic_write(
            self.path,
            json.dumps(self.payload, indent=2, sort_keys=True) + "\n",
            0o600,
        )

    def begin(self, bundle_root: Path, answers: "Answers") -> None:
        previous = {}
        if self.path.is_file():
            try:
                previous = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous = {}
        attempts = int(previous.get("attempts") or 0) + 1
        manifest_path = bundle_root / "manifest.json"
        bundle_id = ""
        if manifest_path.is_file():
            bundle_id = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        self.payload = {
            "format": 1,
            "version": VERSION,
            "bundle_id": bundle_id,
            "attempts": attempts,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "last_completed_phase": previous.get("last_completed_phase"),
            "target": {
                "install_root": str(answers.install_root),
                "data_root": str(answers.data_root),
                "config_root": str(answers.config_root),
                "evo_url": answers.evo_url,
                "evo_username": answers.evo_username,
                "local_vlm": answers.local_vlm,
                "vlm_url": answers.vlm_url,
                "local_deep": answers.local_deep,
                "deep_url": answers.deep_url,
                "timezone": answers.timezone,
                "admin_username": answers.admin_username,
            },
            "phases": {},
        }
        self._write()

    def mark(self, phase: str, status: str, detail: str = "") -> None:
        now = datetime.now(timezone.utc).isoformat()
        phases = self.payload.setdefault("phases", {})
        phases[phase] = {
            "status": status,
            "updated_at": now,
            **({"detail": self._safe(detail)[:500]} if detail else {}),
        }
        if status == "completed":
            self.payload["last_completed_phase"] = phase
        elif status == "failed":
            self.payload["status"] = "failed"
            self.payload["failed_phase"] = phase
        self.payload["updated_at"] = now
        self._write()

    def complete(self) -> None:
        self.payload["status"] = "complete"
        self.payload["completed_at"] = datetime.now(timezone.utc).isoformat()
        self.payload.pop("failed_phase", None)
        self._write()


@dataclass
class Hardware:
    gpu_lines: list[str] = field(default_factory=list)
    nvidia_pci: bool = False
    intel_display: bool = False

    @property
    def nvidia_ready(self) -> bool:
        return bool(self.gpu_lines)


@dataclass
class Answers:
    install_root: Path
    data_root: Path
    config_root: Path
    evo_url: str
    evo_username: str
    evo_password: str = field(repr=False)
    local_vlm: bool = True
    vlm_url: str = DEFAULT_VLM_URL
    vlm_model: str = DEFAULT_VLM_MODEL
    local_deep: bool = True
    deep_url: str = DEFAULT_DEEP_URL
    deep_model: str = DEFAULT_DEEP_MODEL
    timezone: str = "Europe/Riga"
    quiet_enabled: bool = False
    quiet_start: str = "01:00"
    quiet_end: str = "05:00"
    admin_username: str = "admin"
    admin_display_name: str = "EVA Administrator"
    admin_password: str = field(default="", repr=False)


def local_siglip2_cuda_selected(
    env: Mapping[str, str] | None = None,
) -> bool:
    """Return whether the appliance keeps a local CUDA embedding workload."""

    values = PORT_ENV if env is None else env
    embedder = str(values.get("EVOSSEARCH_EMBEDDER") or "").strip().lower()
    model = str(values.get("EVOSSEARCH_CLIP_MODEL") or "").strip().lower()
    clip_device = str(values.get("EVOSSEARCH_CLIP_DEVICE") or "").strip().lower()
    return (
        embedder in {"clip", "fusion"}
        and "siglip2" in model
        and clip_device.startswith("cuda")
    )


def requires_local_nvidia(answers: Answers) -> bool:
    """VLM placement is independent from the local semantic embedder."""

    return bool(answers.local_vlm or local_siglip2_cuda_selected())


class Runner:
    def __init__(self, *, dry_run: bool, secrets_to_redact: Iterable[str] = ()) -> None:
        self.dry_run = dry_run
        self.secrets = tuple(value for value in secrets_to_redact if value)

    def _safe(self, text: str) -> str:
        safe = str(text)
        for secret in sorted(self.secrets, key=len, reverse=True):
            safe = safe.replace(secret, "***")
        return safe

    def add_secrets(self, values: Iterable[str]) -> None:
        additions = tuple(str(value) for value in values if value)
        self.secrets = tuple(dict.fromkeys((*self.secrets, *additions)))

    def run(
        self,
        command: Sequence[str | Path],
        *,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        capture: bool = False,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        argv = [str(item) for item in command]
        print(self._safe("+ " + shlex.join(argv)))
        if self.dry_run:
            return subprocess.CompletedProcess(argv, 0, "", "")
        completed = subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            env=dict(env) if env is not None else None,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            check=False,
        )
        if capture:
            if completed.stdout:
                print(self._safe(completed.stdout.rstrip()))
            if completed.stderr and completed.returncode:
                print(self._safe(completed.stderr.rstrip()), file=sys.stderr)
        if check and completed.returncode:
            raise InstallError(f"Command failed ({completed.returncode}): {argv[0]}")
        return completed


def run_phase(journal: InstallJournal, name: str, operation):
    print(f"\n== {name} ==")
    journal.mark(name, "running")
    try:
        result = operation()
    except Exception as exc:
        journal.mark(name, "failed", f"{type(exc).__name__}: {exc}")
        raise
    journal.mark(name, "completed")
    return result


def _prompt(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    entered = input(f"{label}{suffix}: ").strip()
    return entered or str(default or "")


def _prompt_secret(label: str, *, minimum: int = 1) -> str:
    while True:
        value = getpass.getpass(f"{label}: ").strip()
        if len(value) >= minimum:
            return value
        print(f"Please enter at least {minimum} characters.")


def _read_secret_file(path_value: str | None, label: str, *, minimum: int) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    try:
        value = path.read_text(encoding="utf-8").rstrip("\r\n")
    except OSError as exc:
        raise InstallError(f"Cannot read {label} file {path}: {exc}") from exc
    if len(value) < minimum:
        raise InstallError(f"{label} must contain at least {minimum} characters.")
    if "\x00" in value or "\n" in value or "\r" in value:
        raise InstallError(f"{label} file must contain exactly one line.")
    return value


def _yes_no(label: str, default: bool = True) -> bool:
    marker = "Y/n" if default else "y/N"
    entered = input(f"{label} [{marker}]: ").strip().lower()
    if not entered:
        return default
    return entered in {"y", "yes"}


def _valid_clock(value: str) -> bool:
    match = re.fullmatch(r"(\d{2}):(\d{2})", value)
    return bool(match and int(match.group(1)) < 24 and int(match.group(2)) < 60)


def _url(value: str) -> str:
    text = value.strip()
    if "://" not in text:
        text = "http://" + text
    parsed = urlsplit(text)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise InstallError(f"Not a valid HTTP(S) URL: {value!r}")
    if parsed.username or parsed.password:
        raise InstallError("URLs containing embedded credentials are not supported.")
    return text.rstrip("/")


def detect_hardware() -> Hardware:
    gpu_lines: list[str] = []
    if shutil.which("nvidia-smi"):
        completed = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if completed.returncode == 0:
            gpu_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    pci = ""
    if shutil.which("lspci"):
        completed = subprocess.run(
            ("lspci",),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        pci = completed.stdout.lower()
    nvidia_pci = "nvidia" in pci
    intel_display = "intel" in pci and any(token in pci for token in ("vga", "display"))
    # A minimal Ubuntu Server install may not have pciutils yet.  The kernel's
    # sysfs PCI inventory is sufficient to avoid rejecting the local-inference
    # path before the offline package repository has installed lspci/drivers.
    if not nvidia_pci:
        for vendor_file in Path("/sys/bus/pci/devices").glob("*/vendor"):
            try:
                if vendor_file.read_text(encoding="ascii").strip().lower() == "0x10de":
                    nvidia_pci = True
                    break
            except OSError:
                continue
    return Hardware(
        gpu_lines=gpu_lines,
        nvidia_pci=nvidia_pci,
        intel_display=intel_display,
    )


def nearest_existing(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def disk_free_gib(path: Path) -> float:
    return shutil.disk_usage(nearest_existing(path)).free / (1024**3)


def validate_target_host(os_release: Path = Path("/etc/os-release")) -> None:
    machine = platform.machine().lower()
    if machine not in {"x86_64", "amd64"}:
        raise InstallError(
            f"This bundle targets Ubuntu 24.04 amd64; detected architecture {machine!r}."
        )
    values: dict[str, str] = {}
    if os_release.is_file():
        for line in os_release.read_text(encoding="utf-8").splitlines():
            key, separator, raw = line.partition("=")
            if separator:
                values[key] = raw.strip().strip("\"'")
    if values.get("ID") != "ubuntu" or values.get("VERSION_ID") != "24.04":
        raise InstallError(
            "This appliance bundle requires Ubuntu Server 24.04. "
            f"Detected {values.get('ID', 'unknown')} {values.get('VERSION_ID', 'unknown')}."
        )


def evo_reachable(url: str) -> tuple[bool, str]:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=4) as response:
            return True, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        return True, f"HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, str(exc.reason if isinstance(exc, urllib.error.URLError) else exc)


def read_manifest(bundle_root: Path) -> dict:
    path = bundle_root / "manifest.json"
    if not path.is_file():
        raise InstallError(f"Bundle manifest is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InstallError(f"Bundle manifest is invalid: {exc}") from exc
    if str(payload.get("version") or "") != VERSION:
        raise InstallError(
            f"Bundle version {payload.get('version')!r} does not match installer {VERSION!r}"
        )
    return payload


def verify_critical_payload(bundle_root: Path, manifest: Mapping) -> None:
    required = (
        "repo/VERSION",
        "repo/alembic.ini",
        "repo/migrations/versions/20260801_0011_incidents.py",
        "wheelhouse",
        "apt/Packages.gz",
        "models/qwen3-vl-4b-awq/model.safetensors",
        "models/qwen3.5-9b-mtp/Qwen3.5-9B-Q4_K_M.gguf",
        "models/clip/ViT-B-32.pt",
        (
            "models/huggingface/models--google--siglip2-base-patch16-224/"
            f"snapshots/{DEFAULT_SIGLIP2_REVISION}/model.safetensors"
        ),
        *(
            (
                "models/huggingface/models--google--siglip2-base-patch16-224/"
                f"snapshots/{DEFAULT_SIGLIP2_REVISION}/{filename}"
            )
            for filename in (
                "config.json",
                "preprocessor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
            )
        ),
        "llama.cpp/CMakeLists.txt",
    )
    missing = [item for item in required if not (bundle_root / item).exists()]
    if missing:
        raise InstallError("Offline payload is incomplete: " + ", ".join(missing))
    for relative, expected in dict(manifest.get("critical_sha256") or {}).items():
        path = bundle_root / relative
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise InstallError(f"Checksum mismatch: {relative}")


def current_schema() -> str | None:
    if not shutil.which("psql"):
        return None
    command = (
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
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    value = completed.stdout.strip()
    return value or None


def database_exists() -> bool:
    if not shutil.which("psql"):
        return False
    completed = subprocess.run(
        (
            "runuser",
            "-u",
            "postgres",
            "--",
            "psql",
            "-Atqc",
            "SELECT 1 FROM pg_database WHERE datname='eva'",
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0 and completed.stdout.strip() == "1"


def gather_answers(non_interactive: bool, args: argparse.Namespace) -> Answers:
    if non_interactive:
        if args.evo_password and args.evo_password_file:
            raise InstallError(
                "Use either --evo-password or --evo-password-file, not both."
            )
        evo_password = args.evo_password or _read_secret_file(
            args.evo_password_file,
            "Evo password",
            minimum=1,
        )
        admin_password = _read_secret_file(
            args.admin_password_file,
            "administrator password",
            minimum=12,
        )
        required = (args.evo_url, args.evo_username, evo_password, admin_password)
        if not all(required):
            raise InstallError(
                "--non-interactive requires Evo URL/username/password and "
                "--admin-password-file"
            )
        if args.no_deep_review and args.external_deep_url:
            raise InstallError(
                "--no-deep-review cannot be combined with --external-deep-url"
            )
        root = Path(args.install_root or DEFAULT_ROOT)
        local_deep = not bool(args.external_deep_url) and not args.no_deep_review
        deep_url = (
            ""
            if args.no_deep_review
            else _url(args.external_deep_url or DEFAULT_DEEP_URL)
        )
        deep_model = (
            ""
            if not deep_url
            else args.external_deep_model or DEFAULT_DEEP_MODEL
        )
        quiet_enabled = bool(
            deep_url and args.quiet_window_start and args.quiet_window_end
        )
        if bool(args.quiet_window_start) != bool(args.quiet_window_end):
            raise InstallError(
                "Both --quiet-window-start and --quiet-window-end are required."
            )
        if quiet_enabled and (
            not _valid_clock(args.quiet_window_start)
            or not _valid_clock(args.quiet_window_end)
        ):
            raise InstallError("Quiet-window times must use valid HH:MM values.")
        answers = Answers(
            install_root=root,
            data_root=Path(args.data_root or DEFAULT_DATA),
            config_root=Path(args.config_root or DEFAULT_CONFIG),
            evo_url=_url(args.evo_url),
            evo_username=args.evo_username,
            evo_password=evo_password,
            local_vlm=not bool(args.external_vlm_url),
            vlm_url=_url(args.external_vlm_url or DEFAULT_VLM_URL),
            vlm_model=args.external_vlm_model or DEFAULT_VLM_MODEL,
            local_deep=local_deep,
            deep_url=deep_url,
            deep_model=deep_model,
            timezone=args.timezone or "Europe/Riga",
            quiet_enabled=quiet_enabled,
            quiet_start=args.quiet_window_start or "01:00",
            quiet_end=args.quiet_window_end or "05:00",
            admin_username=args.admin_username or "admin",
            admin_display_name=args.admin_display_name or "EVA Administrator",
            admin_password=admin_password,
        )
        return answers

    print("\nConnect the Luxriot Evo server to the same network before continuing.")
    input("Press Enter when Evo is connected and you know its credentials...")
    evo_url = _url(_prompt("Luxriot Evo IP address or URL"))
    evo_username = _prompt("Luxriot Evo username")
    evo_password = _prompt_secret("Luxriot Evo password")

    print("\nDefault filesystem layout:")
    print(f"  application and inference: {DEFAULT_ROOT}")
    print(f"  database-adjacent runtime data: {DEFAULT_DATA}")
    print(f"  configuration: {DEFAULT_CONFIG}")
    if _yes_no("Use this layout?", True):
        install_root = DEFAULT_ROOT
        data_root = DEFAULT_DATA
        config_root = DEFAULT_CONFIG
    else:
        install_root = Path(_prompt("Application/inference root", str(DEFAULT_ROOT)))
        data_root = Path(_prompt("Runtime data root", str(DEFAULT_DATA)))
        config_root = Path(_prompt("Configuration root", str(DEFAULT_CONFIG)))

    print("\nInference placement:")
    print("  Recommended local profile: RTX 4070 Super, Qwen3-VL-4B AWQ/vLLM,")
    print("  32K context, FP8 KV, 4 sequences, 4096 batched tokens, ~10 GB VRAM.")
    local_vlm = _yes_no("Install and run the VLM on this computer?", True)
    if local_vlm:
        vlm_url, vlm_model = DEFAULT_VLM_URL, DEFAULT_VLM_MODEL
    else:
        vlm_url = _url(_prompt("External OpenAI-compatible VLM URL"))
        vlm_model = _prompt("External VLM model id", DEFAULT_VLM_MODEL)

    local_deep = _yes_no(
        "Install the CPU Qwen3.5-9B-MTP endpoint for preemptible L3 review?",
        True,
    )
    if local_deep:
        deep_url, deep_model = DEFAULT_DEEP_URL, DEFAULT_DEEP_MODEL
    else:
        deep_url_raw = _prompt(
            "External deep-review endpoint (leave empty to disable)",
            "",
        )
        deep_url = _url(deep_url_raw) if deep_url_raw else ""
        deep_model = (
            _prompt("External deep-review model id", DEFAULT_DEEP_MODEL)
            if deep_url
            else ""
        )

    timezone_name = _prompt("Site timezone", "Europe/Riga")
    quiet_enabled = bool(deep_url) and _yes_no(
        "Configure a quiet window for 9B consolidation now?",
        False,
    )
    quiet_start, quiet_end = "01:00", "05:00"
    if quiet_enabled:
        quiet_start = _prompt("Quiet window start (HH:MM)", "01:00")
        quiet_end = _prompt("Quiet window end (HH:MM)", "05:00")
        if not _valid_clock(quiet_start) or not _valid_clock(quiet_end):
            raise InstallError("Quiet-window times must use valid HH:MM values.")

    print("\nCreate the first EVA administrator:")
    admin_username = _prompt("Admin username", "admin")
    admin_display_name = _prompt("Admin display name", "EVA Administrator")
    while True:
        admin_password = _prompt_secret("Admin password", minimum=12)
        if secrets.compare_digest(
            admin_password,
            getpass.getpass("Confirm admin password: "),
        ):
            break
        print("Passwords do not match.")

    return Answers(
        install_root=install_root,
        data_root=data_root,
        config_root=config_root,
        evo_url=evo_url,
        evo_username=evo_username,
        evo_password=evo_password,
        local_vlm=local_vlm,
        vlm_url=vlm_url,
        vlm_model=vlm_model,
        local_deep=local_deep,
        deep_url=deep_url,
        deep_model=deep_model,
        timezone=timezone_name,
        quiet_enabled=quiet_enabled,
        quiet_start=quiet_start,
        quiet_end=quiet_end,
        admin_username=admin_username,
        admin_display_name=admin_display_name,
        admin_password=admin_password,
    )


def print_plan(
    answers: Answers,
    hardware: Hardware,
    *,
    db_exists: bool,
    schema: str | None,
    required_gib: float,
) -> None:
    print("\nInstallation plan")
    print(f"  EVA AI: {VERSION}, schema head {EXPECTED_SCHEMA}")
    print(f"  App/inference root: {answers.install_root}")
    print(f"  Runtime data: {answers.data_root}")
    print(f"  Configuration: {answers.config_root}")
    print(f"  Evo: {answers.evo_url} (user {answers.evo_username})")
    print(
        "  VLM: "
        + (
            "local Qwen3-VL-4B AWQ/vLLM on port 1234"
            if answers.local_vlm
            else f"external {answers.vlm_url}"
        )
    )
    print(
        "  Deep L3: "
        + (
            "local Qwen3.5-9B-MTP/llama.cpp CPU on port 1236"
            if answers.local_deep
            else (f"external {answers.deep_url}" if answers.deep_url else "disabled")
        )
    )
    print(
        "  Semantic embedder: local SigLIP2 on CUDA"
        if local_siglip2_cuda_selected()
        else "  Semantic embedder: configured without local CUDA"
    )
    print(
        "  Quiet window: "
        + (
            f"{answers.quiet_start}-{answers.quiet_end} {answers.timezone}"
            if answers.quiet_enabled
            else "disabled; operator can configure it later"
        )
    )
    print(f"  PostgreSQL: {'existing' if db_exists else 'fresh local database'}")
    if db_exists:
        print(f"  Current schema: {schema or 'unknown'}; backup will be taken first")
    print(f"  Free space required by this plan: at least {required_gib:.1f} GiB")
    if hardware.gpu_lines:
        for line in hardware.gpu_lines:
            print(f"  GPU: {line}")
    elif hardware.nvidia_pci:
        print("  GPU: NVIDIA PCI device found; driver is not active yet")
    else:
        print("  GPU: no NVIDIA device detected")
    print(
        "  Intel QSV: "
        + ("Intel display controller detected" if hardware.intel_display else "not detected")
    )


def _atomic_write(path: Path, text: str, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def _env_quote(value: str) -> str:
    if any(char in value for char in ("\n", "\r", "\x00", "'")):
        raise InstallError("A configuration value contains unsupported characters.")
    return "'" + value + "'"


def parse_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
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


def render_env(values: Mapping[str, str]) -> str:
    lines = [
        "# EVA AI eight-channel appliance configuration.",
        "# Generated by the offline port installer; chmod 0600.",
    ]
    for key in sorted(values):
        lines.append(f"{key}={_env_quote(str(values[key]))}")
    return "\n".join(lines) + "\n"


def install_offline_apt(
    bundle_root: Path,
    runner: Runner,
    *,
    include_nvidia: bool,
    apt_root: Path = DEFAULT_OFFLINE_APT_ROOT,
) -> None:
    source_repo = bundle_root / "apt"
    digest_source = source_repo / "Packages.gz"
    if not digest_source.is_file():
        digest_source = source_repo / "package-names.txt"
    packages_digest = hashlib.sha256(digest_source.read_bytes()).hexdigest()[:16]
    repos_root = apt_root / "repos"
    repo = repos_root / packages_digest
    runner.run(("install", "-d", "-o", "root", "-g", "root", "-m", "0755", apt_root))
    runner.run(
        ("install", "-d", "-o", "root", "-g", "root", "-m", "0755", repos_root)
    )
    if runner.dry_run:
        print(f"+ stage offline APT repository {source_repo} -> {repo}")
    elif not repo.is_dir():
        candidate = repos_root / f".{packages_digest}.{os.getpid()}.tmp"
        if candidate.exists():
            shutil.rmtree(candidate)
        shutil.copytree(source_repo, candidate, symlinks=False)
        for path in candidate.rglob("*"):
            current_mode = path.stat().st_mode & 0o777
            os.chmod(path, current_mode | (0o055 if path.is_dir() else 0o044))
        os.chmod(candidate, 0o755)
        os.replace(candidate, repo)

    package_file = (
        repo / "package-names.txt"
        if (repo / "package-names.txt").is_file()
        else source_repo / "package-names.txt"
    )
    packages = (
        [
            line.strip()
            for line in package_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        if package_file.is_file()
        else list(APT_PACKAGES)
    )
    if not include_nvidia:
        packages = [
            package
            for package in packages
            if not package.startswith("nvidia-")
        ]
    source_file = apt_root / "eva-ai-offline.list"
    lists_dir = apt_root / "lists"
    cache_dir = apt_root / "cache"
    runner.run(
        ("install", "-d", "-o", "root", "-g", "root", "-m", "0755", lists_dir)
    )
    runner.run(
        (
            "install",
            "-d",
            "-o",
            "_apt",
            "-g",
            "root",
            "-m",
            "0700",
            lists_dir / "partial",
        )
    )
    runner.run(
        (
            "install",
            "-d",
            "-o",
            "root",
            "-g",
            "root",
            "-m",
            "0755",
            cache_dir,
            cache_dir / "archives",
        )
    )
    runner.run(
        (
            "install",
            "-d",
            "-o",
            "_apt",
            "-g",
            "root",
            "-m",
            "0700",
            cache_dir / "archives" / "partial",
        )
    )
    source_text = f"deb [trusted=yes] file:{repo.resolve()} ./\n"
    if not runner.dry_run:
        _atomic_write(source_file, source_text, 0o644)
    apt_options = (
        "-o",
        f"Dir::Etc::sourcelist={source_file}",
        "-o",
        "Dir::Etc::sourceparts=-",
        "-o",
        f"Dir::State::lists={lists_dir}",
        "-o",
        f"Dir::Cache={cache_dir}",
        "-o",
        "Acquire::Languages=none",
        "-o",
        "Acquire::Retries=0",
    )
    apt_env = dict(os.environ)
    apt_env["DEBIAN_FRONTEND"] = "noninteractive"
    apt_env["APT_LISTCHANGES_FRONTEND"] = "none"
    runner.run(("apt-get", *apt_options, "update"), env=apt_env)
    runner.run(
        (
            "apt-get",
            *apt_options,
            "--no-install-recommends",
            "-y",
            "install",
            *packages,
        ),
        env=apt_env,
    )


def ensure_accounts_and_dirs(answers: Answers, runner: Runner) -> None:
    runner.run(("getent", "group", "eva"), check=False)
    if not runner.dry_run and subprocess.run(("getent", "group", "eva"), check=False).returncode:
        runner.run(("groupadd", "--system", "eva"))
    runner.run(("getent", "passwd", "eva"), check=False)
    if not runner.dry_run and subprocess.run(("getent", "passwd", "eva"), check=False).returncode:
        runner.run(
            (
                "useradd",
                "--system",
                "--gid",
                "eva",
                "--home-dir",
                str(answers.data_root),
                "--shell",
                "/usr/sbin/nologin",
                "eva",
            )
        )
    directories = (
        answers.install_root,
        answers.data_root,
        answers.data_root / "models",
        answers.data_root / "models" / "clip",
        answers.data_root / "models" / "huggingface",
        answers.data_root / "state",
        answers.data_root / "inference-spool",
        answers.data_root / "detections_archive",
        answers.config_root,
        DEFAULT_BACKUPS,
    )
    runner.run(("mkdir", "-p", *directories))
    runner.run(("chown", "-R", "eva:eva", answers.install_root, answers.data_root))
    runner.run(("chmod", "0755", answers.install_root))
    runner.run(("chmod", "0750", answers.data_root, answers.config_root))


def quiesce_existing_runtime(runner: Runner) -> None:
    """Stop only EVA-owned services before replacing code and configuration."""

    for service in ("eva-ai", "eva-vllm", "eva-deep-review"):
        runner.run(("systemctl", "stop", service), check=False)


def backup_existing(
    answers: Answers,
    runner: Runner,
    *,
    db_exists: bool,
) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup = DEFAULT_BACKUPS / f"port-install-{stamp}"
    runner.run(("mkdir", "-p", backup))
    env_file = answers.config_root / "eva-ai.env"
    if env_file.is_file():
        runner.run(("cp", "-a", env_file, backup / "eva-ai.env"))
    app_dir = answers.install_root / "app"
    if app_dir.is_dir():
        runner.run(
            (
                "tar",
                "--exclude=.venv",
                "--exclude=__pycache__",
                "-czf",
                backup / "app-source.tar.gz",
                "-C",
                app_dir,
                ".",
            )
        )
    if db_exists:
        database_backup_dir = backup / "postgres"
        runner.run(
            (
                "install",
                "-d",
                "-o",
                "postgres",
                "-g",
                "postgres",
                "-m",
                "0700",
                database_backup_dir,
            )
        )
        runner.run(
            (
                "runuser",
                "-u",
                "postgres",
                "--",
                "pg_dump",
                "-Fc",
                "-f",
                database_backup_dir / "eva.postgres.dump",
                "eva",
            )
        )
        runner.run(("chown", "-R", "root:root", database_backup_dir))
    return backup


def sync_payload(bundle_root: Path, answers: Answers, runner: Runner) -> None:
    app_dir = answers.install_root / "app"
    runner.run(("mkdir", "-p", app_dir))
    runner.run(
        (
            "rsync",
            "-a",
            "--delete",
            "--delete-delay",
            "--exclude=.venv",
            "--exclude=.git",
            str(bundle_root / "repo") + "/",
            str(app_dir) + "/",
        )
    )
    if answers.local_vlm:
        runner.run(
            (
                "rsync",
                "-a",
                "--delete",
                str(bundle_root / "models" / "qwen3-vl-4b-awq") + "/",
                str(answers.data_root / "models" / "qwen3-vl-4b-awq") + "/",
            )
        )
    if answers.local_deep:
        runner.run(
            (
                "rsync",
                "-a",
                "--delete",
                str(bundle_root / "models" / "qwen3.5-9b-mtp") + "/",
                str(answers.data_root / "models" / "qwen3.5-9b-mtp") + "/",
            )
        )
    runner.run(
        (
            "install",
            "-m",
            "0644",
            bundle_root / "models" / "clip" / "ViT-B-32.pt",
            answers.data_root / "models" / "clip" / "ViT-B-32.pt",
        )
    )
    runner.run(
        (
            "rsync",
            "-a",
            "--delete",
            str(bundle_root / "models" / "huggingface") + "/",
            str(answers.data_root / "models" / "huggingface") + "/",
        )
    )
    if answers.local_deep:
        runner.run(
            (
                "rsync",
                "-a",
                "--delete",
                str(bundle_root / "llama.cpp") + "/",
                str(answers.install_root / "llama.cpp") + "/",
            )
        )
    runner.run(("chown", "-R", "eva:eva", answers.install_root, answers.data_root))


def install_python_envs(bundle_root: Path, answers: Answers, runner: Runner) -> None:
    app_dir = answers.install_root / "app"
    app_python = app_dir / ".venv" / "bin" / "python"
    if not app_python.exists() or runner.dry_run:
        runner.run(("python3", "-m", "venv", app_dir / ".venv"))
    common_pip = (
        "--no-index",
        "--find-links",
        bundle_root / "wheelhouse",
        "--constraint",
        bundle_root / "constraints-port-4070s.txt",
    )
    runner.run(
        (
            app_python,
            "-m",
            "pip",
            "install",
            *common_pip,
            "-r",
            app_dir / "requirements.txt",
            "-r",
            app_dir / "requirements-db.txt",
        )
    )
    vllm_dir = answers.install_root / "vllm"
    vllm_python = vllm_dir / ".venv" / "bin" / "python"
    if answers.local_vlm:
        if not vllm_python.exists() or runner.dry_run:
            runner.run(("python3", "-m", "venv", vllm_dir / ".venv"))
        runner.run(
            (
                vllm_python,
                "-m",
                "pip",
                "install",
                *common_pip,
                "vllm==0.25.0",
            )
        )
    runner.run(("chown", "-R", "eva:eva", answers.install_root))


def prepare_database(
    answers: Answers,
    runner: Runner,
    *,
    db_was_present: bool,
    existing_env: Mapping[str, str],
) -> dict[str, str]:
    runner.run(("systemctl", "enable", "--now", "postgresql"))
    if not db_was_present:
        runner.run(("runuser", "-u", "postgres", "--", "createdb", "eva"))

    app_dir = answers.install_root / "app"
    migration_env = dict(os.environ)
    migration_env["EVA_DATABASE_DSN"] = "postgresql:///eva?host=/var/run/postgresql"
    runner.run(
        (
            "runuser",
            "--preserve-environment",
            "-u",
            "postgres",
            "--",
            app_dir / ".venv" / "bin" / "alembic",
            "upgrade",
            "head",
        ),
        cwd=app_dir,
        env=migration_env,
    )

    password_keys = {
        "EVA_MIGRATOR_PASSWORD": "eva_migrator_login",
        "EVA_API_PASSWORD": "eva_api_login",
        "EVA_AUDIT_PASSWORD": "eva_audit_login",
        "EVA_WORKER_PASSWORD": "eva_worker_login",
        "EVA_BACKUP_PASSWORD": "eva_backup_login",
    }
    passwords: dict[str, str] = {}
    for key in password_keys:
        passwords[key] = existing_env.get(key) or secrets.token_hex(32)

    role_env = dict(migration_env)
    role_env.update(passwords)
    runner.run(
        (
            "runuser",
            "--preserve-environment",
            "-u",
            "postgres",
            "--",
            app_dir / ".venv" / "bin" / "python",
            "scripts/bootstrap_db_roles.py",
        ),
        cwd=app_dir,
        env=role_env,
    )
    return passwords


def build_llama_cpp(answers: Answers, runner: Runner) -> None:
    if not answers.local_deep:
        return
    source = answers.install_root / "llama.cpp"
    build = source / "build-port-cpu"
    runner.run(
        (
            "cmake",
            "-S",
            source,
            "-B",
            build,
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_NATIVE=OFF",
            "-DGGML_AVX=ON",
            "-DGGML_AVX2=ON",
            "-DGGML_AVX512=OFF",
            "-DGGML_CUDA=OFF",
            "-DLLAMA_CURL=OFF",
        )
    )
    runner.run(
        (
            "cmake",
            "--build",
            build,
            "--target",
            "llama-server",
            "-j",
            str(min(12, os.cpu_count() or 4)),
        )
    )
    runner.run(("chown", "-R", "eva:eva", source))


TENANT_ID_KEYS = (
    "EVOSSEARCH_AUTH_TENANT_ID",
    "EVOSSEARCH_ARCHIVE_TENANT_ID",
    "EVOSSEARCH_INFERENCE_QUEUE_TENANT_ID",
)

OBSOLETE_OR_UNSAFE_ENV_KEYS = {
    # Named-user authentication is mandatory for the appliance.  Carrying a
    # token from an old developer install silently re-enables a second auth
    # model and was one of the field-install failure modes.
    "EVOSSEARCH_ADMIN_TOKEN",
}


def resolve_tenant_id(existing: Mapping[str, str]) -> str:
    configured = {
        key: str(existing.get(key) or "").strip()
        for key in TENANT_ID_KEYS
        if str(existing.get(key) or "").strip()
    }
    normalized: set[str] = set()
    for key, value in configured.items():
        try:
            normalized.add(str(uuid.UUID(value)))
        except ValueError as exc:
            raise InstallError(
                f"{key} is not a valid UUID; refusing to change tenant identity."
            ) from exc
    if len(normalized) > 1:
        raise InstallError(
            "The configured auth, archive and inference tenant IDs disagree; "
            "refusing to merge tenant data automatically."
        )
    return next(iter(normalized), str(uuid.uuid4()))


def render_runtime_env(
    answers: Answers,
    existing: Mapping[str, str],
    passwords: Mapping[str, str],
) -> dict[str, str]:
    values = dict(PORT_ENV)
    values.update(
        {
            "EVOSSEARCH_LUXRIOT_BASE_URL": answers.evo_url,
            "EVOSSEARCH_LUXRIOT_USERNAME": answers.evo_username,
            "EVOSSEARCH_LUXRIOT_PASSWORD": answers.evo_password,
            "EVOSSEARCH_SITE_TIMEZONE": answers.timezone,
            "EVOSSEARCH_MODEL_CACHE_DIR": str(answers.data_root / "models" / "huggingface"),
            "EVOSSEARCH_OPENAI_CLIP_CACHE_DIR": str(answers.data_root / "models" / "clip"),
            "EVOSSEARCH_ALLOWED_ROOTS": str(answers.data_root),
            "EVOSSEARCH_DETECTIONS_ARCHIVE_DIR": str(
                answers.data_root / "detections_archive"
            ),
            "EVOSSEARCH_INFERENCE_QUEUE_SPOOL_DIR": str(
                answers.data_root / "inference-spool"
            ),
            "EVOSSEARCH_LM_VISION_HEALTH_STATE_FILE": str(
                answers.data_root / "state" / "vlm-vision-health.json"
            ),
            "EVOSSEARCH_LM_VISION_HEALTH_MAX_AGE_SEC": "180",
            "EVOSSEARCH_PROBE_CHANNEL_GROUPS_FILE": str(
                answers.data_root / "state" / "probe_channel_groups.json"
            ),
            "EVOSSEARCH_LUXRIOT_SUMMARY_STATE_FILE": str(
                answers.data_root / "state" / "luxriot_summary_state.json"
            ),
            "EVOSSEARCH_LUXRIOT_ROLLUP_CACHE_FILE": str(
                answers.data_root / "state" / "luxriot_rollups_cache.json"
            ),
            "EVOSSEARCH_LM_PROFILES": "agent,vlm",
            "EVOSSEARCH_LM_AGENT_PROFILE_ID": "agent",
            "EVOSSEARCH_LM_VLM_PROFILE_ID": "vlm",
            "EVOSSEARCH_LM_PROFILE_AGENT_KIND": "agent",
            "EVOSSEARCH_LM_PROFILE_AGENT_ENABLED": "true",
            "EVOSSEARCH_LM_PROFILE_AGENT_BASE_URL": answers.vlm_url,
            "EVOSSEARCH_LM_PROFILE_AGENT_MODEL": answers.vlm_model,
            "EVOSSEARCH_LM_PROFILE_AGENT_TIMEOUT": "600",
            "EVOSSEARCH_LM_PROFILE_AGENT_MAX_INFLIGHT": "8",
            "EVOSSEARCH_LM_PROFILE_VLM_KIND": "vlm",
            "EVOSSEARCH_LM_PROFILE_VLM_ENABLED": "true",
            "EVOSSEARCH_LM_PROFILE_VLM_BASE_URL": answers.vlm_url,
            "EVOSSEARCH_LM_PROFILE_VLM_MODEL": answers.vlm_model,
            "EVOSSEARCH_LM_PROFILE_VLM_TIMEOUT": "600",
            "EVOSSEARCH_LM_PROFILE_VLM_MAX_INFLIGHT": "8",
            "EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_BASE_URL": answers.deep_url,
            "EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_MODEL": answers.deep_model,
            "EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_ENABLED": (
                "true" if answers.deep_url else "false"
            ),
            "EVOSSEARCH_LUXRIOT_ROLLUP_L3_QUIET_WINDOW_ENABLED": (
                "true" if answers.quiet_enabled else "false"
            ),
            "EVOSSEARCH_LUXRIOT_ROLLUP_L3_QUIET_WINDOW_TIMEZONE": answers.timezone,
            "EVOSSEARCH_LUXRIOT_ROLLUP_L3_QUIET_WINDOW_START": answers.quiet_start,
            "EVOSSEARCH_LUXRIOT_ROLLUP_L3_QUIET_WINDOW_END": answers.quiet_end,
        }
    )
    tenant_id = resolve_tenant_id(existing)
    for key in TENANT_ID_KEYS:
        values[key] = tenant_id
    values.update(passwords)
    values.update(
        {
            "EVA_DATABASE_DSN": (
                f"postgresql://eva_api_login:{passwords['EVA_API_PASSWORD']}"
                "@127.0.0.1:5432/eva"
            ),
            "EVA_AUDIT_DATABASE_DSN": (
                f"postgresql://eva_audit_login:{passwords['EVA_AUDIT_PASSWORD']}"
                "@127.0.0.1:5432/eva"
            ),
            "EVA_WORKER_DATABASE_DSN": (
                f"postgresql://eva_worker_login:{passwords['EVA_WORKER_PASSWORD']}"
                "@127.0.0.1:5432/eva"
            ),
            "EVA_MIGRATION_DATABASE_DSN": (
                f"postgresql://eva_migrator_login:{passwords['EVA_MIGRATOR_PASSWORD']}"
                "@127.0.0.1:5432/eva"
            ),
        }
    )
    for key, value in existing.items():
        if key not in values and key not in OBSOLETE_OR_UNSAFE_ENV_KEYS:
            values[key] = value
    return values


def validate_runtime_config(answers: Answers, runner: Runner) -> None:
    runner.run(
        (
            answers.install_root / "app" / ".venv" / "bin" / "python",
            answers.install_root / "app" / "scripts" / "validate_appliance_config.py",
            "--env-file",
            answers.config_root / "eva-ai.env",
        )
    )


def install_systemd_units(answers: Answers, runner: Runner) -> None:
    app_dir = answers.install_root / "app"
    env_file = answers.config_root / "eva-ai.env"
    validate_config = app_dir / "scripts" / "validate_appliance_config.py"
    units: dict[Path, str] = {}
    local_vlm_dependencies = ""
    if answers.local_vlm:
        local_vlm_dependencies = (
            "After=eva-vllm.service\n"
            "Wants=eva-vllm.service\n"
        )
    units[Path("/etc/systemd/system/eva-ai.service")] = f"""[Unit]
Description=EVA AI eight-channel appliance
After=network-online.target postgresql.service
Wants=network-online.target
Requires=postgresql.service
{local_vlm_dependencies}

[Service]
Type=simple
User=eva
Group=eva
WorkingDirectory={app_dir}
EnvironmentFile={env_file}
Environment=EVOSSEARCH_CONFIG_ENV_FILE={env_file}
ExecStartPre={app_dir}/.venv/bin/python {validate_config} --from-environment
ExecStartPre={app_dir}/.venv/bin/python {app_dir}/scripts/wait_openai_endpoint.py --timeout 600
ExecStart={app_dir}/run_prod.sh
Restart=on-failure
RestartSec=5
TimeoutStartSec=660
TimeoutStopSec=120
KillSignal=SIGTERM
UMask=0077
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
"""
    if answers.local_vlm:
        model = answers.data_root / "models" / "qwen3-vl-4b-awq"
        vllm = answers.install_root / "vllm" / ".venv" / "bin" / "vllm"
        units[Path("/etc/systemd/system/eva-vllm.service")] = f"""[Unit]
Description=EVA Qwen3-VL-4B AWQ vLLM
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=eva
Group=eva
WorkingDirectory={answers.install_root}/vllm
Environment=HF_HOME={answers.data_root}/models/huggingface
Environment=HF_HUB_OFFLINE=1
Environment=TRANSFORMERS_OFFLINE=1
Environment=VLLM_USE_FLASHINFER_SAMPLER=0
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EnvironmentFile={env_file}
ExecStart={vllm} serve {model} --served-model-name {DEFAULT_VLM_MODEL} --host 127.0.0.1 --port 1234 --max-model-len 32768 --gpu-memory-utilization 0.75 --max-num-seqs 8 --max-num-batched-tokens 4096 --kv-cache-dtype fp8 --attention-backend TRITON_ATTN --mm-encoder-attn-backend FLASH_ATTN --mm-processor-cache-gb 0 --limit-mm-per-prompt.image 16 --limit-mm-per-prompt.video 0 --mm-processor-kwargs.max_pixels 100352 --enable-auto-tool-choice --tool-call-parser hermes
ExecStartPost={app_dir}/.venv/bin/python {app_dir}/scripts/wait_openai_endpoint.py --timeout 240
Restart=on-failure
RestartSec=10
TimeoutStartSec=300
TimeoutStopSec=60
KillMode=mixed

[Install]
WantedBy=multi-user.target
"""
        vision_state = answers.data_root / "state" / "vlm-vision-health.json"
        watchdog = app_dir / "scripts" / "vlm_vision_watchdog.py"
        python = app_dir / ".venv" / "bin" / "python"
        units[Path("/etc/systemd/system/eva-vlm-vision-watchdog.service")] = f"""[Unit]
Description=EVA content-aware VLM vision watchdog
After=eva-vllm.service
Requires=eva-vllm.service
OnFailure=eva-vlm-vision-recover.service

[Service]
Type=oneshot
User=eva
Group=eva
WorkingDirectory={app_dir}
EnvironmentFile={env_file}
Environment=EVOSSEARCH_CONFIG_ENV_FILE={env_file}
ExecStart={python} {watchdog} --state-file {vision_state} --failure-threshold 2 --timeout 30
Nice=10
NoNewPrivileges=true
PrivateTmp=true
"""
        units[Path("/etc/systemd/system/eva-vlm-vision-watchdog.timer")] = """[Unit]
Description=Run EVA VLM vision watchdog every minute

[Timer]
OnBootSec=120s
OnUnitActiveSec=60s
AccuracySec=5s
Persistent=true
Unit=eva-vlm-vision-watchdog.service

[Install]
WantedBy=timers.target
"""
        units[Path("/etc/systemd/system/eva-vlm-vision-recover.service")] = """[Unit]
Description=Recover EVA VLM after confirmed visual inference failure
StartLimitIntervalSec=3600
StartLimitBurst=3

[Service]
Type=oneshot
ExecStart=/usr/bin/systemctl restart eva-vllm.service
"""
    if answers.local_deep:
        binary = answers.install_root / "llama.cpp" / "build-port-cpu" / "bin" / "llama-server"
        model = answers.data_root / "models" / "qwen3.5-9b-mtp" / "Qwen3.5-9B-Q4_K_M.gguf"
        units[Path("/etc/systemd/system/eva-deep-review.service")] = f"""[Unit]
Description=EVA CPU Qwen3.5-9B-MTP deep review
After=network-online.target

[Service]
Type=simple
User=eva
Group=eva
WorkingDirectory={answers.install_root}/llama.cpp
ExecStart={binary} -m {model} -a {DEFAULT_DEEP_MODEL} --spec-type draft-mtp --spec-draft-n-max 4 -c 65536 -ngl 0 -fa on -ctk q8_0 -ctv q8_0 -np 1 -cb --threads 12 --threads-batch 16 --jinja --metrics --host 127.0.0.1 --port 1236
Restart=on-failure
RestartSec=10
TimeoutStartSec=300
TimeoutStopSec=60
KillMode=mixed

[Install]
WantedBy=multi-user.target
"""
    for path, content in units.items():
        if runner.dry_run:
            print(f"+ write {path}")
        else:
            _atomic_write(path, content, 0o644)
    retired_units = []
    if not answers.local_vlm:
        retired_units.append(("eva-vllm", Path("/etc/systemd/system/eva-vllm.service")))
        retired_units.extend(
            (
                ("eva-vlm-vision-watchdog.timer", Path("/etc/systemd/system/eva-vlm-vision-watchdog.timer")),
                ("eva-vlm-vision-watchdog.service", Path("/etc/systemd/system/eva-vlm-vision-watchdog.service")),
                ("eva-vlm-vision-recover", Path("/etc/systemd/system/eva-vlm-vision-recover.service")),
            )
        )
    if not answers.local_deep:
        retired_units.append(
            ("eva-deep-review", Path("/etc/systemd/system/eva-deep-review.service"))
        )
    for service, path in retired_units:
        runner.run(("systemctl", "disable", "--now", service), check=False)
        runner.run(("rm", "-f", path))
    runner.run(("systemctl", "daemon-reload"))
    services = ["postgresql", "eva-ai"]
    if answers.local_vlm:
        services.append("eva-vllm")
        services.append("eva-vlm-vision-watchdog.timer")
    if answers.local_deep:
        services.append("eva-deep-review")
    runner.run(("systemctl", "enable", *services))
    runner.run(
        (
            "ln",
            "-sfn",
            app_dir / "scripts" / "eva_appliance_doctor.py",
            "/usr/local/sbin/eva-ai-doctor",
        )
    )


def configure_nginx(answers: Answers, runner: Runner) -> None:
    cert_dir = answers.config_root / "tls"
    cert = cert_dir / "eva-ai.crt"
    key = cert_dir / "eva-ai.key"
    runner.run(("mkdir", "-p", cert_dir))
    if not cert.exists() or not key.exists() or runner.dry_run:
        hostname = socket.getfqdn() or socket.gethostname() or "eva-ai"
        san_entries = [f"DNS:{hostname}", f"DNS:{socket.gethostname()}"]
        try:
            addresses = {
                info[4][0]
                for info in socket.getaddrinfo(
                    socket.gethostname(),
                    None,
                    socket.AF_INET,
                )
                if not info[4][0].startswith("127.")
            }
        except socket.gaierror:
            addresses = set()
        san_entries.extend(f"IP:{address}" for address in sorted(addresses))
        runner.run(
            (
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:3072",
                "-sha256",
                "-days",
                "825",
                "-nodes",
                "-subj",
                f"/CN={hostname}",
                "-addext",
                "subjectAltName=" + ",".join(san_entries),
                "-keyout",
                key,
                "-out",
                cert,
            )
        )
        runner.run(("chmod", "0600", key))
    nginx = f"""server {{
    listen 443 ssl;
    listen [::]:443 ssl;
    server_name _;
    ssl_certificate {cert};
    ssl_certificate_key {key};
    client_max_body_size 256m;

    location / {{
        proxy_pass http://127.0.0.1:5000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
        proxy_buffering off;
    }}
}}
"""
    site = Path("/etc/nginx/sites-available/eva-ai")
    if runner.dry_run:
        print(f"+ write {site}")
    else:
        _atomic_write(site, nginx, 0o644)
    runner.run(("ln", "-sfn", site, "/etc/nginx/sites-enabled/eva-ai"))
    runner.run(("rm", "-f", "/etc/nginx/sites-enabled/default"))
    runner.run(("nginx", "-t"))
    runner.run(("systemctl", "enable", "--now", "nginx"))


def _wait_for_json_endpoint(
    url: str,
    *,
    label: str,
    timeout_sec: int,
    expected_status: str | None = None,
    expected_model: str | None = None,
) -> dict:
    deadline = time.monotonic() + timeout_sec
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=8) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if expected_status and payload.get("status") != expected_status:
                last_error = f"status={payload.get('status')!r}"
            elif expected_model:
                model_ids = {
                    str(item.get("id") or "")
                    for item in payload.get("data", [])
                    if isinstance(item, Mapping)
                }
                if expected_model not in model_ids:
                    last_error = (
                        f"expected model {expected_model!r}; available={sorted(model_ids)!r}"
                    )
                else:
                    print(f"{label} ready: {expected_model}")
                    return payload
            else:
                print(f"{label} ready.")
                return payload
        except Exception as exc:  # bounded readiness retry
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(3)
    raise InstallError(f"{label} did not become ready within {timeout_sec}s: {last_error}")


def _vision_smoke_png() -> bytes:
    """Build a deterministic PNG without depending on Pillow on a fresh host."""

    width, height = 640, 360
    pixels = bytearray([255, 255, 255] * width * height)

    def rectangle(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        for y in range(max(0, y0), min(height, y1)):
            row = y * width * 3
            for x in range(max(0, x0), min(width, x1)):
                offset = row + x * 3
                pixels[offset : offset + 3] = bytes(color)

    # A three-colour sequence catches an image encoder returning unrelated features.
    rectangle(70, 35, 210, 145, (220, 35, 45))
    rectangle(250, 35, 390, 145, (30, 170, 75))
    rectangle(430, 35, 570, 145, (35, 90, 220))

    glyphs = {
        "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
        "3": ("11110", "00010", "00010", "01110", "00010", "00010", "11110"),
        "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
        "9": ("01110", "10001", "10001", "01111", "00001", "00010", "11100"),
    }
    scale = 18
    digit_width = 5 * scale
    gap = 2 * scale
    code = "7391"
    total_width = len(code) * digit_width + (len(code) - 1) * gap
    cursor_x = (width - total_width) // 2
    for digit in code:
        for row_index, row in enumerate(glyphs[digit]):
            for column_index, enabled in enumerate(row):
                if enabled == "1":
                    rectangle(
                        cursor_x + column_index * scale,
                        190 + row_index * scale,
                        cursor_x + (column_index + 1) * scale,
                        190 + (row_index + 1) * scale,
                        (10, 10, 10),
                    )
        cursor_x += digit_width + gap

    raw = b"".join(
        b"\x00" + bytes(pixels[y * width * 3 : (y + 1) * width * 3])
        for y in range(height)
    )

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=9))
        + chunk(b"IEND", b"")
    )


def _verify_vlm_vision(base_url: str, model: str, *, timeout_sec: int = 90) -> None:
    image = base64.b64encode(_vision_smoke_png()).decode("ascii")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Read the four black digits and the three coloured blocks "
                            "from left to right. Reply on one line exactly as: "
                            "VISION_OK <digits> <COLOR> <COLOR> <COLOR>."
                        ),
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 64,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            result = json.loads(response.read().decode("utf-8"))
        content = str(result["choices"][0]["message"]["content"])
    except Exception as exc:
        raise InstallError(
            "VLM text endpoint is ready, but the required vision smoke test failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    normalized = " ".join(content.upper().replace(",", " ").split())
    if "7391" not in normalized or not all(
        color in normalized for color in ("RED", "GREEN", "BLUE")
    ):
        raise InstallError(
            "VLM responded to an image but did not perceive the control frame. "
            f"Expected code 7391 and RED/GREEN/BLUE; received {content!r}. "
            "Check the multimodal encoder/attention backend before starting EVA."
        )
    print("VLM vision smoke passed: code 7391, RED/GREEN/BLUE.")


def start_and_verify(answers: Answers, runner: Runner) -> None:
    if answers.local_vlm:
        runner.run(("systemctl", "restart", "eva-vllm"))
    if not runner.dry_run:
        _wait_for_json_endpoint(
            answers.vlm_url.rstrip("/") + "/models",
            label="Local VLM" if answers.local_vlm else "External VLM",
            timeout_sec=600 if answers.local_vlm else 60,
            expected_model=answers.vlm_model,
        )
        _verify_vlm_vision(answers.vlm_url, answers.vlm_model)
    if answers.local_vlm:
        runner.run(("systemctl", "restart", "eva-vlm-vision-watchdog.service"))
        runner.run(("systemctl", "restart", "eva-vlm-vision-watchdog.timer"))
    if answers.local_deep:
        runner.run(("systemctl", "restart", "eva-deep-review"))
    if answers.deep_url and not runner.dry_run:
        _wait_for_json_endpoint(
            answers.deep_url.rstrip("/") + "/models",
            label=(
                "Local deep-review model"
                if answers.local_deep
                else "External deep-review model"
            ),
            timeout_sec=300 if answers.local_deep else 60,
            expected_model=answers.deep_model,
        )
    runner.run(("systemctl", "restart", "eva-ai"))
    if runner.dry_run:
        return
    _wait_for_json_endpoint(
        "http://127.0.0.1:5000/ready",
        label="EVA and required dependencies",
        timeout_sec=300,
        expected_status="ready",
    )


def bootstrap_admin(
    answers: Answers,
    values: Mapping[str, str],
    runner: Runner,
) -> None:
    env = dict(os.environ)
    env.update(values)
    env["EVA_BOOTSTRAP_ADMIN_PASSWORD"] = answers.admin_password
    runner.run(
        (
            answers.install_root / "app" / ".venv" / "bin" / "python",
            answers.install_root / "app" / "scripts" / "bootstrap_admin.py",
            "--tenant-id",
            values["EVOSSEARCH_AUTH_TENANT_ID"],
            "--username",
            answers.admin_username,
            "--display-name",
            answers.admin_display_name,
        ),
        env=env,
    )


def apply_install(
    bundle_root: Path,
    answers: Answers,
    hardware: Hardware,
    *,
    dry_run: bool,
    db_was_present: bool,
) -> None:
    if not dry_run and os.geteuid() != 0:
        raise InstallError("Installation requires root. Run ./install.sh with sudo access.")
    runner = Runner(
        dry_run=dry_run,
        secrets_to_redact=(answers.evo_password, answers.admin_password),
    )
    journal = InstallJournal(
        dry_run=dry_run,
        secrets_to_redact=(answers.evo_password, answers.admin_password),
    )
    journal.begin(bundle_root, answers)
    try:
        run_phase(
            journal,
            "offline_apt",
            lambda: install_offline_apt(
                bundle_root,
                runner,
                include_nvidia=requires_local_nvidia(answers),
            ),
        )

        def confirm_database_discovery() -> None:
            if not dry_run and not db_was_present and database_exists():
                raise InstallError(
                    "An existing EVA database became visible after PostgreSQL tools "
                    "were installed. Rerun the installer so it can report the schema "
                    "and request backup/migration approval before changing it."
                )

        run_phase(journal, "database_discovery", confirm_database_discovery)
        run_phase(
            journal,
            "filesystem",
            lambda: ensure_accounts_and_dirs(answers, runner),
        )

        def activate_gpu() -> None:
            if not requires_local_nvidia(answers) or hardware.nvidia_ready:
                return
            if not hardware.nvidia_pci:
                raise InstallError(
                    "A local CUDA workload (VLM and/or SigLIP2) was selected, "
                    "but no NVIDIA GPU is visible on PCI. External VLM mode "
                    "still requires a local GPU while SigLIP2 uses CUDA."
                )
            runner.run(("modprobe", "nvidia"), check=False)
            refreshed = detect_hardware() if not dry_run else hardware
            if not dry_run and not refreshed.nvidia_ready:
                raise InstallError(
                    "The NVIDIA driver is installed but the GPU is not active. "
                    "Reboot and rerun this installer; completed phases are safe to replay."
                )

        run_phase(journal, "gpu", activate_gpu)
        run_phase(
            journal,
            "quiesce_runtime",
            lambda: quiesce_existing_runtime(runner),
        )

        env_file = answers.config_root / "eva-ai.env"
        existing_env = parse_env(env_file)
        backup = run_phase(
            journal,
            "backup",
            lambda: backup_existing(
                answers,
                runner,
                db_exists=db_was_present,
            ),
        )
        print(f"Backup directory: {backup}")
        run_phase(
            journal,
            "application_payload",
            lambda: sync_payload(bundle_root, answers, runner),
        )
        run_phase(
            journal,
            "python_environments",
            lambda: install_python_envs(bundle_root, answers, runner),
        )
        run_phase(
            journal,
            "deep_review_runtime",
            lambda: build_llama_cpp(answers, runner),
        )
        passwords = run_phase(
            journal,
            "database",
            lambda: prepare_database(
                answers,
                runner,
                db_was_present=db_was_present,
                existing_env=existing_env,
            ),
        )
        runner.add_secrets(passwords.values())
        journal.add_secrets(passwords.values())
        values = render_runtime_env(answers, existing_env, passwords)

        def write_configuration() -> None:
            if runner.dry_run:
                print(f"+ write {env_file} (secrets redacted)")
            else:
                _atomic_write(env_file, render_env(values), 0o600)

        run_phase(journal, "configuration", write_configuration)
        run_phase(
            journal,
            "configuration_preflight",
            lambda: validate_runtime_config(answers, runner),
        )
        run_phase(
            journal,
            "systemd_units",
            lambda: install_systemd_units(answers, runner),
        )
        run_phase(
            journal,
            "reverse_proxy",
            lambda: configure_nginx(answers, runner),
        )
        run_phase(
            journal,
            "administrator",
            lambda: bootstrap_admin(answers, values, runner),
        )
        run_phase(
            journal,
            "services_and_readiness",
            lambda: start_and_verify(answers, runner),
        )

        def verify_schema() -> None:
            if runner.dry_run:
                return
            revision = current_schema()
            if revision != EXPECTED_SCHEMA:
                raise InstallError(
                    f"Installed database revision is {revision!r}; "
                    f"expected {EXPECTED_SCHEMA}"
                )

        run_phase(journal, "schema_verification", verify_schema)
    except Exception:
        print(
            f"Installer state: {journal.path}. Fix the reported cause and rerun; "
            "completed phases are idempotent.",
            file=sys.stderr,
        )
        raise
    journal.complete()
    print("\nINSTALLATION COMPLETE")
    print("Open EVA AI at: https://<this-server-ip>/")
    print("The TLS certificate is locally generated; import/trust it on operator workstations.")
    print("Next: log in, run 'Protocol: Deploy', select up to eight Evo channels,")
    print("and configure/confirm the 9B quiet window if it was left disabled.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install the EVA AI eight-channel port appliance from an offline USB.",
    )
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip final confirmation.")
    parser.add_argument("--install-root")
    parser.add_argument("--data-root")
    parser.add_argument("--config-root")
    parser.add_argument("--evo-url")
    parser.add_argument("--evo-username")
    parser.add_argument("--evo-password")
    parser.add_argument("--evo-password-file")
    parser.add_argument("--external-vlm-url")
    parser.add_argument("--external-vlm-model")
    parser.add_argument("--external-deep-url")
    parser.add_argument("--external-deep-model")
    parser.add_argument("--no-deep-review", action="store_true")
    parser.add_argument("--quiet-window-start")
    parser.add_argument("--quiet-window-end")
    parser.add_argument("--admin-username")
    parser.add_argument("--admin-display-name")
    parser.add_argument("--admin-password-file")
    parser.add_argument("--timezone")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    script_parent = Path(__file__).resolve().parent
    default_bundle_root = (
        script_parent
        if (script_parent / "manifest.json").is_file()
        else script_parent.parent
    )
    bundle_root = args.bundle_root.resolve() if args.bundle_root else default_bundle_root
    try:
        print(f"EVA AI {VERSION} offline port-appliance installer")
        print("Target: RTX 4070 Super, Intel Core i9 14th Gen, 64 GB RAM, Ubuntu 24.04.")
        validate_target_host()
        manifest = read_manifest(bundle_root)
        verify_critical_payload(bundle_root, manifest)
        answers = gather_answers(args.non_interactive, args)
        required_gib = max(
            MIN_FREE_GIB,
            float(manifest.get("minimum_free_bytes") or 0) / (1024**3),
        )
        free_gib = disk_free_gib(answers.install_root)
        print(
            f"\nDisk check: {free_gib:.1f} GiB free at "
            f"{nearest_existing(answers.install_root)}"
        )
        if free_gib < required_gib:
            raise InstallError(
                f"At least {required_gib:.1f} GiB free is required; "
                f"only {free_gib:.1f} GiB is available."
            )

        reachable, detail = evo_reachable(answers.evo_url)
        print(f"Evo reachability: {'OK' if reachable else 'WARNING'} ({detail})")
        if not reachable and not args.non_interactive:
            if not _yes_no("Continue and configure Evo even though it is not reachable?", False):
                raise InstallError("Installation cancelled until Evo is reachable.")

        hardware = detect_hardware()
        db_present = database_exists()
        schema = current_schema() if db_present else None
        print_plan(
            answers,
            hardware,
            db_exists=db_present,
            schema=schema,
            required_gib=required_gib,
        )
        if requires_local_nvidia(answers) and not (
            hardware.nvidia_ready or hardware.nvidia_pci
        ):
            raise InstallError(
                "A local CUDA workload was selected, but no NVIDIA GPU was detected. "
                "Using an external VLM does not move the local SigLIP2 embedder."
            )
        if db_present and schema != EXPECTED_SCHEMA:
            print(
                f"NOTICE: the existing database will be backed up and migrated "
                f"from {schema or 'an unknown revision'} to {EXPECTED_SCHEMA}."
            )
            if not args.non_interactive and not _yes_no(
                "Approve backup and database migration?",
                False,
            ):
                raise InstallError("Database migration was not approved.")
        if args.dry_run:
            apply_install(
                bundle_root,
                answers,
                hardware,
                dry_run=True,
                db_was_present=db_present,
            )
            return 0
        if not (args.yes or args.non_interactive) and not _yes_no(
            "Proceed with installation?",
            False,
        ):
            print("No changes made.")
            return 0
        apply_install(
            bundle_root,
            answers,
            hardware,
            dry_run=False,
            db_was_present=db_present,
        )
        return 0
    except (InstallError, OSError, ValueError) as exc:
        print(f"\nINSTALLER ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
