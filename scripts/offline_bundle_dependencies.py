#!/usr/bin/env python3
"""Build-time and field-time validation for EVA's offline dependencies.

The first field bundles treated ``apt/`` and ``wheelhouse/`` as informal
directories.  A bundle could therefore be finalized successfully and fail on
the target only after the running service had already been touched.  This
module turns both directories into a versioned, checksummed payload.

Only the standard library is required for field verification.  The optional
pip resolver is a build-time check and is deliberately not used on a fresh
host before ``python3-pip`` has been installed from the bundle.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


MANIFEST_NAME = "offline-dependencies.json"
MANIFEST_FORMAT = 1
TARGET = {
    "os": "Ubuntu 24.04 LTS",
    "architecture": "amd64",
    "python": "CPython 3.12",
}


class DependencyError(RuntimeError):
    """An incomplete or corrupt offline dependency payload."""


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _safe_relative(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise DependencyError(f"Dependency path escapes the bundle: {relative}")
    return path


def _requested_apt_packages(path: Path) -> list[str]:
    if not path.is_file():
        raise DependencyError(f"APT package list is missing: {path}")
    packages = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not packages:
        raise DependencyError("APT package list is empty")
    if len(packages) != len(set(packages)):
        raise DependencyError("APT package list contains duplicate package names")
    return packages


def _package_stanzas(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise DependencyError(f"APT index is missing: {path}")
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="strict") as handle:
            raw = handle.read()
    except (OSError, UnicodeError) as exc:
        raise DependencyError(f"APT index is unreadable: {exc}") from exc
    stanzas: list[dict[str, str]] = []
    for block in raw.split("\n\n"):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            if line.startswith((" ", "\t")) or ":" not in line:
                continue
            key, value = line.split(":", 1)
            fields[key] = value.strip()
        if fields.get("Package"):
            stanzas.append(fields)
    if not stanzas:
        raise DependencyError("APT index contains no package records")
    return stanzas


def _validate_apt(bundle: Path) -> dict[str, object]:
    apt_root = bundle / "apt"
    packages_path = apt_root / "package-names.txt"
    index_path = apt_root / "Packages.gz"
    requested = _requested_apt_packages(packages_path)
    stanzas = _package_stanzas(index_path)
    indexed = {row["Package"] for row in stanzas}
    missing = sorted(set(requested) - indexed)
    if missing:
        raise DependencyError(
            "APT index does not provide requested packages: " + ", ".join(missing)
        )

    artifacts: list[dict[str, object]] = []
    seen_files: set[str] = set()
    for row in stanzas:
        relative = row.get("Filename", "").removeprefix("./")
        expected_sha = row.get("SHA256", "")
        expected_size = row.get("Size", "")
        if not relative or not expected_sha or not expected_size:
            raise DependencyError(
                f"APT record for {row['Package']} lacks Filename, Size or SHA256"
            )
        if relative in seen_files:
            continue
        seen_files.add(relative)
        path = _safe_relative(apt_root, relative)
        if not path.is_file():
            raise DependencyError(f"APT artifact is missing: apt/{relative}")
        actual_size = path.stat().st_size
        try:
            recorded_size = int(expected_size)
        except ValueError as exc:
            raise DependencyError(f"Invalid APT Size for {relative}") from exc
        if actual_size != recorded_size:
            raise DependencyError(
                f"APT artifact size mismatch: apt/{relative} "
                f"({actual_size} != {recorded_size})"
            )
        actual_sha = _digest(path)
        if actual_sha != expected_sha:
            raise DependencyError(f"APT artifact checksum mismatch: apt/{relative}")
        artifacts.append(
            {
                "path": f"apt/{relative}",
                "size": actual_size,
                "sha256": actual_sha,
            }
        )

    actual_debs = {
        str(path.relative_to(apt_root))
        for path in apt_root.rglob("*.deb")
        if path.is_file()
    }
    stale = sorted(actual_debs - seen_files)
    if stale:
        raise DependencyError(
            "APT directory contains artifacts absent from Packages.gz: "
            + ", ".join(stale)
        )

    return {
        "requested_packages": requested,
        "indexed_packages": len(indexed),
        "artifacts": sorted(artifacts, key=lambda item: str(item["path"])),
        "packages_index_sha256": _digest(index_path),
        "package_list_sha256": _digest(packages_path),
    }


def _wheel_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise DependencyError(f"Python wheelhouse is missing: {root}")
    wheels = sorted(path for path in root.iterdir() if path.is_file() and path.suffix == ".whl")
    unsupported = sorted(
        path.name
        for path in root.iterdir()
        if path.is_file() and path.suffix != ".whl"
    )
    if unsupported:
        raise DependencyError(
            "Wheelhouse must be binary-only; remove source archives: "
            + ", ".join(unsupported)
        )
    if not wheels:
        raise DependencyError("Python wheelhouse contains no wheels")
    return wheels


def _validate_wheels(bundle: Path) -> dict[str, object]:
    wheelhouse = bundle / "wheelhouse"
    artifacts: list[dict[str, object]] = []
    for path in _wheel_files(wheelhouse):
        try:
            with zipfile.ZipFile(path) as archive:
                if not any(name.endswith(".dist-info/METADATA") for name in archive.namelist()):
                    raise DependencyError(f"Wheel has no package metadata: {path.name}")
                broken = archive.testzip()
        except (OSError, zipfile.BadZipFile) as exc:
            raise DependencyError(f"Wheel is unreadable: {path.name}: {exc}") from exc
        if broken is not None:
            raise DependencyError(f"Wheel CRC check failed: {path.name}: {broken}")
        artifacts.append(
            {
                "path": f"wheelhouse/{path.name}",
                "size": path.stat().st_size,
                "sha256": _digest(path),
            }
        )
    return {"artifacts": artifacts}


def _requirements_fingerprints(repo_root: Path, bundle: Path) -> dict[str, str]:
    paths = {
        "requirements.txt": repo_root / "requirements.txt",
        "requirements-db.txt": repo_root / "requirements-db.txt",
        "requirements-cuda.txt": repo_root / "requirements-cuda.txt",
        "constraints-port-4070s.txt": bundle / "constraints-port-4070s.txt",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise DependencyError("Dependency declarations are missing: " + ", ".join(missing))
    return {name: _digest(path) for name, path in paths.items()}


def _resolve_with_pip(bundle: Path, repo_root: Path, python: str) -> None:
    """Prove that both EVA environments resolve without network access."""

    wheelhouse = bundle / "wheelhouse"
    constraints = bundle / "constraints-port-4070s.txt"
    base = [
        python,
        "-m",
        "pip",
        "install",
        "--dry-run",
        "--ignore-installed",
        "--break-system-packages",
        "--no-index",
        "--find-links",
        str(wheelhouse),
        "--constraint",
        str(constraints),
        "--python-version",
        "3.12",
        "--only-binary=:all:",
    ]
    with tempfile.TemporaryDirectory(prefix="eva-offline-resolve-") as temporary:
        report_path = Path(temporary) / "pip-report.json"
        command = [
            *base,
            "--report",
            str(report_path),
            "-r",
            str(repo_root / "requirements.txt"),
            "-r",
            str(repo_root / "requirements-db.txt"),
            "-r",
            str(repo_root / "requirements-cuda.txt"),
            "vllm==0.25.0",
        ]
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
            check=False,
        )
        if completed.returncode:
            tail = "\n".join(completed.stdout.splitlines()[-30:])
            raise DependencyError(
                "Offline Python dependency resolution failed for CPython 3.12:\n" + tail
            )
        if not report_path.is_file():
            raise DependencyError("pip completed without writing its offline resolution report")


def build_manifest(
    bundle: Path,
    *,
    repo_root: Path,
    resolve: bool = False,
    python: str = sys.executable,
) -> dict[str, object]:
    apt = _validate_apt(bundle)
    wheels = _validate_wheels(bundle)
    requirements = _requirements_fingerprints(repo_root, bundle)
    if resolve:
        _resolve_with_pip(bundle, repo_root, python)
    return {
        "format": MANIFEST_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": TARGET,
        "requirements_sha256": requirements,
        "apt": apt,
        "wheelhouse": wheels,
        "pip_resolution": {
            "target_python": "3.12",
            "vllm": "0.25.0",
            "verified": bool(resolve),
        },
    }


def _iter_artifacts(payload: Mapping[str, object]) -> Iterable[Mapping[str, object]]:
    for section_name in ("apt", "wheelhouse"):
        section = payload.get(section_name)
        if not isinstance(section, Mapping):
            raise DependencyError(f"Dependency manifest has no {section_name} section")
        artifacts = section.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise DependencyError(f"Dependency manifest has no {section_name} artifacts")
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                raise DependencyError(f"Invalid {section_name} artifact record")
            yield artifact


def verify_manifest(bundle: Path, *, repo_root: Path | None = None) -> dict[str, object]:
    path = bundle / MANIFEST_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DependencyError(f"Offline dependency manifest is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise DependencyError(f"Offline dependency manifest is invalid: {exc}") from exc
    if payload.get("format") != MANIFEST_FORMAT:
        raise DependencyError(
            f"Unsupported offline dependency manifest format: {payload.get('format')!r}"
        )
    if payload.get("target") != TARGET:
        raise DependencyError("Offline dependencies target a different OS/Python platform")
    resolution = payload.get("pip_resolution")
    if not isinstance(resolution, Mapping) or resolution.get("verified") is not True:
        raise DependencyError("Wheelhouse was not proven by the offline pip resolver")

    seen: set[str] = set()
    for artifact in _iter_artifacts(payload):
        relative = str(artifact.get("path") or "")
        if not relative or relative in seen:
            raise DependencyError(f"Invalid or duplicate dependency artifact path: {relative!r}")
        seen.add(relative)
        path = _safe_relative(bundle, relative)
        if not path.is_file():
            raise DependencyError(f"Offline dependency artifact is missing: {relative}")
        expected_size = int(artifact.get("size") or -1)
        if path.stat().st_size != expected_size:
            raise DependencyError(f"Offline dependency artifact size mismatch: {relative}")
        if _digest(path) != str(artifact.get("sha256") or ""):
            raise DependencyError(f"Offline dependency artifact checksum mismatch: {relative}")
    if repo_root is not None:
        current = _requirements_fingerprints(repo_root, bundle)
        if payload.get("requirements_sha256") != current:
            raise DependencyError(
                "Wheelhouse was resolved for different requirements or constraints"
            )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and inventory EVA AI offline APT/Python dependencies."
    )
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--verify-manifest", action="store_true")
    parser.add_argument("--resolve", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    bundle = args.bundle.resolve()
    repo_root = (args.repo_root or bundle / "repo").resolve()
    try:
        if args.verify_manifest:
            payload = verify_manifest(bundle, repo_root=repo_root)
        else:
            payload = build_manifest(
                bundle,
                repo_root=repo_root,
                resolve=bool(args.resolve),
                python=str(args.python),
            )
            if args.write_manifest:
                (bundle / MANIFEST_NAME).write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        print(
            "Offline dependencies OK: "
            f"{len(payload['apt']['artifacts'])} APT artifacts, "
            f"{len(payload['wheelhouse']['artifacts'])} wheels, "
            f"pip resolver={'yes' if payload['pip_resolution']['verified'] else 'no'}"
        )
        return 0
    except (DependencyError, OSError, ValueError) as exc:
        print(f"OFFLINE DEPENDENCY ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
