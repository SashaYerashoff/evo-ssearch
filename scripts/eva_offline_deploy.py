#!/usr/bin/env python3
"""Single entry point for an offline EVA AI fresh install or in-place update.

The USB bundle deliberately keeps the two proven mutation engines separate:

* a fresh host is provisioned by ``install_port_appliance.py`` (APT, PostgreSQL,
  local inference, systemd, TLS and the first administrator);
* an existing host is upgraded by ``install_eva_083.py`` (preserve site config,
  mandatory database backup, transactional Alembic upgrade and verification).

This file only detects the target, makes the choice visible, runs a read-only
preflight first and emits a post-deployment report.  It is standard-library-only
so it can run before the offline Python environment exists.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import pwd
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


DEFAULT_BACKUP_ROOT = Path("/var/backups/eva-ai")
DEFAULT_REPORT_ROOT = Path("/var/lib/eva-ai-installer")
DEFAULT_SERVICE = "eva-ai"
EXPECTED_SCHEMA = "20260801_0011"
EXPECTED_FLAVOR = "universal-offline"


class DeployError(RuntimeError):
    """An operator-actionable deployment error."""


@dataclass(frozen=True)
class ExistingDeployment:
    service: str
    app_dir: Path
    env_file: Path
    unit_file: Path
    service_user: str
    service_group: str
    base_url: str


def _run(
    argv: Sequence[str | Path],
    *,
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [str(item) for item in argv]
    print("\n$ " + " ".join(shlex.quote(item) for item in command))
    completed = subprocess.run(command, env=dict(env) if env is not None else None, text=True)
    if check and completed.returncode:
        raise DeployError(f"Command failed with exit {completed.returncode}: {command[0]}")
    return completed


def _capture(argv: Sequence[str | Path]) -> str:
    try:
        completed = subprocess.run(
            [str(item) for item in argv],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _bundle_root(explicit: Path | None) -> Path:
    if explicit is not None:
        root = explicit.resolve()
    else:
        script_dir = Path(__file__).resolve().parent
        candidates = (script_dir, script_dir.parent)
        root = next(
            (
                candidate
                for candidate in candidates
                if (candidate / "repo").is_dir()
                and ((candidate / "manifest.json").is_file() or (candidate / "manifest.txt").is_file())
            ),
            script_dir.parent,
        ).resolve()
    if not (root / "repo").is_dir():
        raise DeployError(f"Offline bundle repo/ is missing under {root}")
    return root


def _verify_bundle(root: Path) -> None:
    """Reject an incomplete, mismatched or corrupted universal payload.

    Fresh installs perform their own payload validation as well, but updates used
    to trust the copied source tree.  Keeping this check in the common entry point
    gives both paths the same fail-before-mutation boundary.
    """

    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DeployError(f"Offline bundle manifest is missing: {manifest_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise DeployError(f"Offline bundle manifest is invalid: {exc}") from exc
    if str(manifest.get("release_flavor") or "") != EXPECTED_FLAVOR:
        raise DeployError(
            "This entry point requires an EVA AI universal-offline bundle; "
            f"found {manifest.get('release_flavor') or 'unknown'}"
        )
    if str(manifest.get("schema_head") or "") != EXPECTED_SCHEMA:
        raise DeployError(
            f"Bundle schema head is {manifest.get('schema_head') or 'unknown'}, "
            f"expected {EXPECTED_SCHEMA}"
        )
    required = (
        "SOURCE_REVISION.json",
        "START_EVA_AI.sh",
        "eva_offline_deploy.py",
        "install_port_appliance.py",
        "migration-plans/0006-to-0011.sql",
        "apt/Packages.gz",
        "wheelhouse",
        "repo/VERSION",
        "repo/react-ui/dist/index.html",
        "repo/migrations/versions/20260801_0011_incidents.py",
    )
    missing = [relative for relative in required if not (root / relative).exists()]
    if missing:
        raise DeployError("Offline payload is incomplete: " + ", ".join(missing))
    critical = manifest.get("critical_sha256")
    if not isinstance(critical, dict) or not critical:
        raise DeployError("Bundle manifest has no critical checksums")
    resolved_root = root.resolve()
    for relative, expected in critical.items():
        candidate = (root / str(relative)).resolve()
        if not candidate.is_relative_to(resolved_root) or not candidate.is_file():
            raise DeployError(f"Invalid critical payload path: {relative}")
        digest = hashlib.sha256()
        with candidate.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != str(expected):
            raise DeployError(f"Checksum mismatch: {relative}")


def _parse_env(path: Path) -> dict[str, str]:
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
    return values


def _systemd_property(service: str, name: str) -> str:
    return _capture(("systemctl", "show", f"{service}.service", f"--property={name}", "--value"))


def _environment_file_from_systemd(service: str) -> Path | None:
    raw = _systemd_property(service, "EnvironmentFiles")
    for token in raw.replace("(ignore_errors=yes)", "").replace("(ignore_errors=no)", "").split():
        candidate = Path(token.lstrip("-"))
        if candidate.is_file():
            return candidate
    return None


def detect_existing(service: str = DEFAULT_SERVICE) -> ExistingDeployment | None:
    load_state = _systemd_property(service, "LoadState")
    working_directory = _systemd_property(service, "WorkingDirectory")
    app_candidates = [
        Path(working_directory) if working_directory else Path("/__missing__"),
        Path("/opt/eva-ai/evo-ssearch"),
        Path("/opt/eva-ai/app"),
    ]
    app_dir = next(
        (path for path in app_candidates if path.is_dir() and (path / "VERSION").is_file()),
        None,
    )
    if load_state != "loaded" and app_dir is None:
        return None
    if app_dir is None:
        raise DeployError(
            f"{service}.service exists, but its EVA WorkingDirectory could not be identified"
        )
    env_file = _environment_file_from_systemd(service)
    if env_file is None:
        env_file = next(
            (
                path
                for path in (
                    Path("/etc/eva-ai/eva-ai.env"),
                    app_dir / "eva-ai.env",
                    app_dir / ".env",
                )
                if path.is_file()
            ),
            None,
        )
    if env_file is None:
        raise DeployError("Existing EVA was detected, but its environment file was not found")
    values = _parse_env(env_file)
    port = str(values.get("EVOSSEARCH_PORT") or "5000").strip()
    unit_file_raw = _systemd_property(service, "FragmentPath")
    unit_file = Path(unit_file_raw or f"/etc/systemd/system/{service}.service")
    service_user = _systemd_property(service, "User") or "eva"
    service_group = _systemd_property(service, "Group") or service_user
    return ExistingDeployment(
        service=service,
        app_dir=app_dir,
        env_file=env_file,
        unit_file=unit_file,
        service_user=service_user,
        service_group=service_group,
        base_url=f"http://127.0.0.1:{port}",
    )


def _require_root() -> None:
    if os.geteuid() != 0:
        raise DeployError("Run this entry point with sudo so backups and system services are protected")


def _manifest_commit(bundle_root: Path) -> str:
    json_manifest = bundle_root / "SOURCE_REVISION.json"
    if json_manifest.is_file():
        try:
            return str(json.loads(json_manifest.read_text(encoding="utf-8")).get("commit") or "")
        except (OSError, json.JSONDecodeError):
            return ""
    manifest = bundle_root / "manifest.txt"
    for line in manifest.read_text(encoding="utf-8").splitlines() if manifest.is_file() else ():
        if line.startswith("git_commit="):
            return line.partition("=")[2].strip()
    return ""


def _report(
    bundle_root: Path,
    deployment: ExistingDeployment,
    *,
    output_prefix: Path,
    baseline: Path | None = None,
    wait_streams: int = 0,
    check: bool = True,
) -> None:
    report_script = bundle_root / "repo" / "scripts" / "eva_deployment_report.py"
    if not report_script.is_file():
        print("WARNING: deployment report script is not present in this bundle", file=sys.stderr)
        return
    command: list[str | Path] = [
        sys.executable,
        report_script,
        "--env-file",
        deployment.env_file,
        "--app-dir",
        deployment.app_dir,
        "--service",
        deployment.service,
        "--base-url",
        deployment.base_url,
        "--json-output",
        output_prefix.with_suffix(".json"),
        "--text-output",
        output_prefix.with_suffix(".txt"),
    ]
    if baseline is not None:
        command.extend(("--baseline", baseline))
    if wait_streams > 0:
        command.extend(("--wait-streams", str(wait_streams)))
    _run(command, check=check)


def _update(
    bundle_root: Path,
    deployment: ExistingDeployment,
    *,
    assume_yes: bool,
    wait_streams: int,
) -> None:
    _require_root()
    source = bundle_root / "repo"
    installer = source / "scripts" / "install_eva_083.py"
    if not installer.is_file():
        raise DeployError(f"Update engine is missing: {installer}")
    values = _parse_env(deployment.env_file)
    process_env = dict(os.environ)
    if not process_env.get("EVA_INSTALL_MIGRATION_DSN") and not values.get(
        "EVA_MIGRATION_DATABASE_DSN"
    ):
        migration_dsn = getpass.getpass(
            "Privileged PostgreSQL migration DSN (used only for this update): "
        ).strip()
        if not migration_dsn:
            raise DeployError(
                "A privileged migration DSN is required; it is not written to eva-ai.env"
            )
        process_env["EVA_INSTALL_MIGRATION_DSN"] = migration_dsn

    DEFAULT_REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    baseline = DEFAULT_REPORT_ROOT / "pre-update-baseline.json"
    _report(
        bundle_root,
        deployment,
        output_prefix=DEFAULT_REPORT_ROOT / "pre-update-baseline",
        check=False,
    )

    common: list[str | Path] = [
        sys.executable,
        installer,
        "--non-interactive",
        "--source-dir",
        source,
        "--bundle-dir",
        bundle_root,
        "--app-dir",
        deployment.app_dir,
        "--env-file",
        deployment.env_file,
        "--backup-root",
        DEFAULT_BACKUP_ROOT,
        "--service-name",
        deployment.service,
        "--service-user",
        deployment.service_user,
        "--service-group",
        deployment.service_group,
        "--unit-file",
        deployment.unit_file,
        "--base-url",
        deployment.base_url,
    ]
    print("\nUPDATE PREFLIGHT (read-only)")
    _run((*common, "--dry-run"), env=process_env)
    if not assume_yes:
        answer = input(
            "\nApply the reviewed update, database backup and migrations now? [y/N]: "
        ).strip().lower()
        if answer not in {"y", "yes"}:
            print("No changes made.")
            return
    print("\nAPPLYING EVA AI UPDATE")
    _run((*common, "--apply"), env=process_env)

    commit = _manifest_commit(bundle_root)
    if commit:
        marker = deployment.app_dir / ".eva-bundle-commit"
        marker.write_text(commit + "\n", encoding="utf-8")
        try:
            identity = pwd.getpwnam(deployment.service_user)
            os.chown(marker, identity.pw_uid, identity.pw_gid)
        except (KeyError, PermissionError):
            pass

    print("\nPOST-UPDATE ACCEPTANCE")
    _report(
        bundle_root,
        deployment,
        output_prefix=DEFAULT_REPORT_ROOT / "last-deployment-report",
        baseline=baseline if baseline.is_file() else None,
        wait_streams=wait_streams,
    )


def _fresh(
    bundle_root: Path,
    *,
    assume_yes: bool,
    passthrough: Sequence[str],
) -> None:
    _require_root()
    installer = bundle_root / "install_port_appliance.py"
    if not installer.is_file():
        installer = bundle_root / "repo" / "scripts" / "install_port_appliance.py"
    if not installer.is_file():
        raise DeployError(f"Fresh-install engine is missing: {installer}")
    command: list[str | Path] = [sys.executable, installer, "--bundle-root", bundle_root]
    if assume_yes:
        command.append("--yes")
    command.extend(passthrough)
    _run(command)
    deployment = detect_existing()
    if deployment is not None:
        DEFAULT_REPORT_ROOT.mkdir(parents=True, exist_ok=True)
        _report(
            bundle_root,
            deployment,
            output_prefix=DEFAULT_REPORT_ROOT / "last-deployment-report",
            wait_streams=0,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install or update EVA AI from one offline USB bundle."
    )
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--mode", choices=("auto", "install", "update", "report"), default="auto")
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--yes", action="store_true", help="Accept the final reviewed mutation plan.")
    parser.add_argument(
        "--wait-streams",
        type=int,
        default=180,
        help="Seconds to wait for previously active summary streams after an update.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, passthrough = parser.parse_known_args(argv)
    try:
        bundle_root = _bundle_root(args.bundle_root)
        _verify_bundle(bundle_root)
        existing = detect_existing(args.service)
        selected_mode = args.mode
        if selected_mode == "auto":
            selected_mode = "update" if existing is not None else "install"
        print("EVA AI UNIVERSAL OFFLINE DEPLOYMENT")
        print(f"Bundle: {bundle_root}")
        print(f"Mode:   {selected_mode.upper()}")
        if selected_mode == "install":
            if existing is not None:
                raise DeployError(
                    "Existing EVA was detected; use --mode update or remove the ambiguity explicitly"
                )
            _fresh(bundle_root, assume_yes=args.yes, passthrough=passthrough)
        elif selected_mode == "update":
            if passthrough:
                raise DeployError(
                    "Fresh-install options were supplied during an update; "
                    "remove them and rerun --mode update"
                )
            if existing is None:
                raise DeployError("No existing EVA deployment was detected; use --mode install")
            _update(
                bundle_root,
                existing,
                assume_yes=args.yes,
                wait_streams=max(0, args.wait_streams),
            )
        else:
            if passthrough:
                raise DeployError("Unsupported options were supplied for --mode report")
            if existing is None:
                raise DeployError("No existing EVA deployment was detected for reporting")
            DEFAULT_REPORT_ROOT.mkdir(parents=True, exist_ok=True)
            _report(
                bundle_root,
                existing,
                output_prefix=DEFAULT_REPORT_ROOT / "last-deployment-report",
            )
        return 0
    except (DeployError, OSError, ValueError) as exc:
        print(f"\nDEPLOYMENT ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
