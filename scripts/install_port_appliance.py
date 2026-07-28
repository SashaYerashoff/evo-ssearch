#!/usr/bin/env python3
"""Interactive offline installer for the EVA AI eight-channel appliance.

The script is intentionally standard-library-only.  It is copied to the root
of the field USB and can therefore run on a fresh Ubuntu Server 24.04 host
before any Python packages have been installed.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.parse import urlsplit


VERSION = "β 0.8.5"
EXPECTED_SCHEMA = "20260727_0010"
DEFAULT_ROOT = Path("/opt/eva-ai")
DEFAULT_DATA = Path("/var/lib/eva-ai")
DEFAULT_CONFIG = Path("/etc/eva-ai")
DEFAULT_BACKUPS = Path("/var/backups/eva-ai")
DEFAULT_VLM_URL = "http://127.0.0.1:1234/v1"
DEFAULT_VLM_MODEL = "qwen/qwen3-vl-4b"
DEFAULT_DEEP_URL = "http://127.0.0.1:1236/v1"
DEFAULT_DEEP_MODEL = "qwen3.5-9b-mtp"
MIN_FREE_GIB = 45

PORT_ENV = {
    "EVOSSEARCH_APP_VERSION": VERSION,
    "EVOSSEARCH_HOST": "127.0.0.1",
    "EVOSSEARCH_PORT": "5000",
    "EVOSSEARCH_DEBUG": "false",
    "EVOSSEARCH_GUNICORN_WORKERS": "1",
    "EVOSSEARCH_GUNICORN_THREADS": "8",
    "EVOSSEARCH_GUNICORN_TIMEOUT": "240",
    "EVOSSEARCH_OFFLINE_MODE": "true",
    "EVOSSEARCH_SECURE_DEPLOYMENT_REQUIRED": "true",
    "EVOSSEARCH_AUTH_ENABLED": "true",
    "EVOSSEARCH_AUTH_COOKIE_SECURE": "true",
    "EVOSSEARCH_TRUSTED_PROXY_HOPS": "1",
    "EVOSSEARCH_DB_STRICT_RUNTIME_ROLES": "true",
    "EVOSSEARCH_EMBEDDER": "clip",
    "EVOSSEARCH_INDEX_MODE": "clip",
    "EVOSSEARCH_PRODUCTION_CLIP_MODEL": "ViT-B/32",
    "EVOSSEARCH_CLIP_MODEL": "ViT-B/32",
    "EVOSSEARCH_EXPERIMENTAL_EMBEDDERS_ENABLED": "false",
    "EVOSSEARCH_DINO_SEGMENTS_ENABLED": "false",
    "EVOSSEARCH_M2F_ENABLED": "false",
    "CUDA_VISIBLE_DEVICES": "-1",
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
    "EVOSSEARCH_LUXRIOT_ROLLUP_SCHEDULER_ENABLED": "true",
    "EVOSSEARCH_LUXRIOT_ROLLUP_LLM_LEVELS": "L1,L2,L3",
    "EVOSSEARCH_LUXRIOT_ROLLUP_LLM_MODEL": "agent",
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


class Runner:
    def __init__(self, *, dry_run: bool, secrets_to_redact: Iterable[str] = ()) -> None:
        self.dry_run = dry_run
        self.secrets = tuple(value for value in secrets_to_redact if value)

    def _safe(self, text: str) -> str:
        safe = str(text)
        for secret in sorted(self.secrets, key=len, reverse=True):
            safe = safe.replace(secret, "***")
        return safe

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
        "repo/migrations/versions/20260727_0010_audit_hash_chain.py",
        "wheelhouse",
        "apt/Packages.gz",
        "models/qwen3-vl-4b-awq/model.safetensors",
        "models/qwen3.5-9b-mtp/Qwen3.5-9B-Q4_K_M.gguf",
        "models/clip/ViT-B-32.pt",
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
        required = (args.evo_url, args.evo_username, args.evo_password)
        if not all(required):
            raise InstallError(
                "--non-interactive requires --evo-url, --evo-username and --evo-password"
            )
        root = Path(args.install_root or DEFAULT_ROOT)
        answers = Answers(
            install_root=root,
            data_root=Path(args.data_root or DEFAULT_DATA),
            config_root=Path(args.config_root or DEFAULT_CONFIG),
            evo_url=_url(args.evo_url),
            evo_username=args.evo_username,
            evo_password=args.evo_password,
            local_vlm=not bool(args.external_vlm_url),
            vlm_url=_url(args.external_vlm_url or DEFAULT_VLM_URL),
            vlm_model=args.external_vlm_model or DEFAULT_VLM_MODEL,
            local_deep=not bool(args.external_deep_url),
            deep_url=_url(args.external_deep_url or DEFAULT_DEEP_URL),
            deep_model=args.external_deep_model or DEFAULT_DEEP_MODEL,
            timezone=args.timezone or "Europe/Riga",
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
) -> None:
    repo = bundle_root / "apt"
    package_file = repo / "package-names.txt"
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
    apt_state = Path("/var/cache/eva-ai-offline-apt")
    source_file = apt_state / "eva-ai-offline.list"
    lists_dir = apt_state / "lists"
    cache_dir = apt_state / "cache"
    runner.run(("mkdir", "-p", lists_dir / "partial", cache_dir / "archives" / "partial"))
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
    )
    runner.run(("apt-get", *apt_options, "update"))
    runner.run(
        (
            "apt-get",
            *apt_options,
            "--no-install-recommends",
            "-y",
            "install",
            *packages,
        )
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
                str(bundle_root / "models" / "qwen3-vl-4b-awq") + "/",
                str(answers.data_root / "models" / "qwen3-vl-4b-awq") + "/",
            )
        )
    if answers.local_deep:
        runner.run(
            (
                "rsync",
                "-a",
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
            "EVOSSEARCH_LM_PROFILE_VLM_KIND": "vlm",
            "EVOSSEARCH_LM_PROFILE_VLM_ENABLED": "true",
            "EVOSSEARCH_LM_PROFILE_VLM_BASE_URL": answers.vlm_url,
            "EVOSSEARCH_LM_PROFILE_VLM_MODEL": answers.vlm_model,
            "EVOSSEARCH_LM_PROFILE_VLM_TIMEOUT": "600",
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
    tenant_id = existing.get("EVOSSEARCH_AUTH_TENANT_ID") or str(uuid.uuid4())
    values["EVOSSEARCH_AUTH_TENANT_ID"] = tenant_id
    values["EVOSSEARCH_ARCHIVE_TENANT_ID"] = tenant_id
    values["EVOSSEARCH_INFERENCE_QUEUE_TENANT_ID"] = tenant_id
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
        if key not in values:
            values[key] = value
    return values


def install_systemd_units(answers: Answers, runner: Runner) -> None:
    app_dir = answers.install_root / "app"
    env_file = answers.config_root / "eva-ai.env"
    units: dict[Path, str] = {}
    units[Path("/etc/systemd/system/eva-ai.service")] = f"""[Unit]
Description=EVA AI eight-channel appliance
After=network-online.target postgresql.service
Wants=network-online.target
Requires=postgresql.service

[Service]
Type=simple
User=eva
Group=eva
WorkingDirectory={app_dir}
EnvironmentFile={env_file}
Environment=EVOSSEARCH_CONFIG_ENV_FILE={env_file}
ExecStart={app_dir}/run_prod.sh
Restart=on-failure
RestartSec=5
TimeoutStartSec=180
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
ExecStart={vllm} serve {model} --served-model-name {DEFAULT_VLM_MODEL} --host 127.0.0.1 --port 1234 --max-model-len 32768 --gpu-memory-utilization 0.82 --max-num-seqs 4 --max-num-batched-tokens 4096 --kv-cache-dtype fp8 --enforce-eager --attention-backend TRITON_ATTN --limit-mm-per-prompt.image 16 --limit-mm-per-prompt.video 0 --mm-processor-kwargs.max_pixels 100352 --enable-auto-tool-choice --tool-call-parser hermes
Restart=on-failure
RestartSec=10
TimeoutStartSec=300
TimeoutStopSec=60
KillMode=mixed

[Install]
WantedBy=multi-user.target
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
    runner.run(("systemctl", "daemon-reload"))
    services = ["postgresql", "eva-ai"]
    if answers.local_vlm:
        services.append("eva-vllm")
    if answers.local_deep:
        services.append("eva-deep-review")
    runner.run(("systemctl", "enable", *services))


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


def start_and_verify(answers: Answers, runner: Runner) -> None:
    if answers.local_vlm:
        runner.run(("systemctl", "restart", "eva-vllm"))
    if answers.local_deep:
        runner.run(("systemctl", "restart", "eva-deep-review"))
    runner.run(("systemctl", "restart", "eva-ai"))
    if runner.dry_run:
        return
    deadline = time.monotonic() + 180
    health_url = "http://127.0.0.1:5000/health"
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("status") == "ok":
                print(f"EVA health OK: {payload}")
                return
        except Exception as exc:  # bounded readiness retry
            last_error = str(exc)
        time.sleep(3)
    raise InstallError(f"EVA did not become healthy within 180 seconds: {last_error}")


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
    install_offline_apt(
        bundle_root,
        runner,
        include_nvidia=answers.local_vlm,
    )
    ensure_accounts_and_dirs(answers, runner)

    if answers.local_vlm and not hardware.nvidia_ready:
        if not hardware.nvidia_pci:
            raise InstallError(
                "Local VLM was selected but no NVIDIA GPU is visible on PCI. "
                "Rerun and select an external endpoint."
            )
        runner.run(("modprobe", "nvidia"), check=False)
        refreshed = detect_hardware() if not dry_run else hardware
        if not dry_run and not refreshed.nvidia_ready:
            raise InstallError(
                "The NVIDIA driver packages were installed but the GPU is not active. "
                "Reboot, rerun this installer, and keep the same answers."
            )

    env_file = answers.config_root / "eva-ai.env"
    existing_env = parse_env(env_file)
    backup = backup_existing(
        answers,
        runner,
        db_exists=db_was_present,
    )
    print(f"Backup directory: {backup}")
    sync_payload(bundle_root, answers, runner)
    install_python_envs(bundle_root, answers, runner)
    build_llama_cpp(answers, runner)
    passwords = prepare_database(
        answers,
        runner,
        db_was_present=db_was_present,
        existing_env=existing_env,
    )
    values = render_runtime_env(answers, existing_env, passwords)
    if runner.dry_run:
        print(f"+ write {env_file} (secrets redacted)")
    else:
        _atomic_write(env_file, render_env(values), 0o600)
    install_systemd_units(answers, runner)
    configure_nginx(answers, runner)
    start_and_verify(answers, runner)
    bootstrap_admin(answers, values, runner)

    if not runner.dry_run:
        revision = current_schema()
        if revision != EXPECTED_SCHEMA:
            raise InstallError(
                f"Installed database revision is {revision!r}; expected {EXPECTED_SCHEMA}"
            )
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
    parser.add_argument("--external-vlm-url")
    parser.add_argument("--external-vlm-model")
    parser.add_argument("--external-deep-url")
    parser.add_argument("--external-deep-model")
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
        if answers.local_vlm and not (hardware.nvidia_ready or hardware.nvidia_pci):
            raise InstallError(
                "Local inference was selected, but no NVIDIA GPU was detected."
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
