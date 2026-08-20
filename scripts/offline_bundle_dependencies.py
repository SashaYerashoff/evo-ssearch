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
import platform
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


MANIFEST_NAME = "offline-dependencies.json"
MANIFEST_FORMAT = 1
SUPPORTED_ARCHITECTURES = {"amd64", "arm64"}
SPARK_VENDOR_RUNTIME_PACKAGES = {
    "numpy": "2.1.0",
    "torch": "2.11.0",
    "torchvision": "0.26.0",
}
SUPPORTED_OS_RELEASES = {
    "24.04": "Ubuntu 24.04 LTS",
    "26.04": "Ubuntu 26.04 LTS",
}
SUPPORTED_PYTHON_VERSIONS = {"3.12", "3.13", "3.14"}


def normalize_architecture(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {
        "x86_64": "amd64",
        "amd64": "amd64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise DependencyError(f"Unsupported offline dependency architecture: {value!r}") from exc


def normalize_os_release(value: str) -> str:
    normalized = value.strip()
    if normalized not in SUPPORTED_OS_RELEASES:
        raise DependencyError(f"Unsupported offline dependency OS release: {value!r}")
    return normalized


def normalize_python_versions(values: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
    unsupported = sorted(set(normalized) - SUPPORTED_PYTHON_VERSIONS)
    if unsupported:
        raise DependencyError(
            "Unsupported offline dependency Python version(s): " + ", ".join(unsupported)
        )
    if not normalized:
        raise DependencyError("At least one offline dependency Python version is required")
    return normalized


def normalize_python_version(value: str) -> str:
    return normalize_python_versions((value,))[0]


def default_python_for_os(os_release: str) -> str:
    return "3.14" if normalize_os_release(os_release) == "26.04" else "3.12"


def dependency_target(
    architecture: str,
    os_release: str = "24.04",
) -> dict[str, str]:
    os_release = normalize_os_release(os_release)
    return {
        "os": SUPPORTED_OS_RELEASES[os_release],
        "os_release": os_release,
        "architecture": normalize_architecture(architecture),
        "python": f"CPython {default_python_for_os(os_release)}",
    }


def constraints_filename(architecture: str) -> str:
    return (
        "constraints-spark-gb10.txt"
        if normalize_architecture(architecture) == "arm64"
        else "constraints-port-4070s.txt"
    )


# Compatibility name for callers/tests which imported the original single
# target constant. New bundles record their explicit architecture in-manifest.
TARGET = dependency_target("amd64")


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


def _write_resolver_stub(directory: Path, name: str, version: str) -> Path:
    """Create a resolver-only wheel representing a vendor-owned ARM package."""

    normalized = name.replace("-", "_")
    path = directory / f"{normalized}-{version}-py3-none-any.whl"
    dist_info = f"{normalized}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{normalized}/__init__.py", "")
        archive.writestr(
            f"{dist_info}/METADATA",
            "\n".join(
                (
                    "Metadata-Version: 2.1",
                    f"Name: {name}",
                    f"Version: {version}",
                    "Summary: resolver-only placeholder for vendor ARM runtime",
                    "",
                )
            ),
        )
        archive.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\nGenerator: eva-offline-resolver\n"
            "Root-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(f"{dist_info}/RECORD", "")
    return path


def _requirements_fingerprints(
    repo_root: Path,
    bundle: Path,
    *,
    architecture: str,
) -> dict[str, str]:
    constraints_name = constraints_filename(architecture)
    paths = {
        "requirements.txt": repo_root / "requirements.txt",
        "requirements-db.txt": repo_root / "requirements-db.txt",
        "requirements-cuda.txt": repo_root / "requirements-cuda.txt",
        constraints_name: bundle / constraints_name,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise DependencyError("Dependency declarations are missing: " + ", ".join(missing))
    return {name: _digest(path) for name, path in paths.items()}


def _resolve_with_pip(
    bundle: Path,
    repo_root: Path,
    python: str,
    *,
    architecture: str,
    include_vllm: bool,
    python_versions: Sequence[str],
    vllm_python_version: str | None = None,
) -> None:
    """Prove that both EVA environments resolve without network access."""

    wheelhouse = bundle / "wheelhouse"
    architecture = normalize_architecture(architecture)
    declared_constraints = bundle / constraints_filename(architecture)
    python_versions = normalize_python_versions(python_versions)
    if include_vllm:
        vllm_python_version = normalize_python_version(
            vllm_python_version or python_versions[0]
        )
        if vllm_python_version not in python_versions:
            raise DependencyError(
                "Local vLLM Python must be present in the update compatibility matrix"
            )
    if architecture == "arm64" and python_versions != ("3.12",):
        raise DependencyError("Spark ARM64 dependency payload is pinned to CPython 3.12")
    with tempfile.TemporaryDirectory(prefix="eva-offline-resolve-") as temporary:
        temporary_root = Path(temporary)
        constraints = declared_constraints
        extra_find_links: list[str] = []
        if architecture == "arm64":
            stubs = temporary_root / "vendor-runtime-stubs"
            stubs.mkdir()
            vendor_stubs = {
                package: _write_resolver_stub(stubs, package, version)
                for package, version in SPARK_VENDOR_RUNTIME_PACKAGES.items()
            }
            constraints = temporary_root / "resolver-constraints.txt"
            constraints.write_text(
                declared_constraints.read_text(encoding="utf-8")
                + "\n"
                + "\n".join(
                    f"{package} @ {path.as_uri()}"
                    for package, path in sorted(vendor_stubs.items())
                )
                + "\n",
                encoding="utf-8",
            )
            extra_find_links.extend(("--find-links", str(stubs)))
        for python_version in python_versions:
            report_path = temporary_root / f"pip-report-{python_version}.json"
            base = [
                python,
                "-m",
                "pip",
                "install",
                "--dry-run",
                "--break-system-packages",
                "--no-index",
                "--find-links",
                str(wheelhouse),
                *extra_find_links,
                "--constraint",
                str(constraints),
                "--only-binary=:all:",
                "--python-version",
                python_version,
            ]
            if architecture == "amd64":
                base.append("--ignore-installed")
            if architecture == "arm64" and normalize_architecture(platform.machine()) != "arm64":
                base.extend(
                    (
                        "--platform",
                        "manylinux_2_28_aarch64",
                        "--platform",
                        "manylinux_2_27_aarch64",
                        "--platform",
                        "manylinux2014_aarch64",
                        "--platform",
                        "manylinux_2_17_aarch64",
                        "--implementation",
                        "cp",
                        "--abi",
                        "cp312",
                    )
                )
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
                *(
                    ("vllm==0.25.0",)
                    if include_vllm and python_version == vllm_python_version
                    else ()
                ),
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
                    f"Offline Python dependency resolution failed for CPython {python_version}:\n"
                    + tail
                )
            if not report_path.is_file():
                raise DependencyError(
                    f"pip completed without writing its CPython {python_version} resolution report"
                )


def build_manifest(
    bundle: Path,
    *,
    repo_root: Path,
    resolve: bool = False,
    python: str = sys.executable,
    architecture: str = "amd64",
    include_vllm: bool | None = None,
    vllm_python_version: str | None = None,
    target_os_release: str = "24.04",
    update_os_releases: Sequence[str] | None = None,
    update_python_versions: Sequence[str] | None = None,
) -> dict[str, object]:
    architecture = normalize_architecture(architecture)
    target_os_release = normalize_os_release(target_os_release)
    target_python = default_python_for_os(target_os_release)
    if update_os_releases is None:
        update_os_releases = (target_os_release,)
    normalized_os_releases = tuple(
        dict.fromkeys(normalize_os_release(value) for value in update_os_releases)
    )
    if target_os_release not in normalized_os_releases:
        raise DependencyError("Fresh-install OS must also be present in update compatibility")
    if update_python_versions is None:
        update_python_versions = (target_python,)
    normalized_python_versions = normalize_python_versions(update_python_versions)
    if target_python not in normalized_python_versions:
        raise DependencyError("Fresh-install Python must also be present in update compatibility")
    if architecture == "arm64" and (
        normalized_os_releases != ("24.04",)
        or normalized_python_versions != ("3.12",)
    ):
        raise DependencyError("Spark ARM64 release remains pinned to Ubuntu 24.04 / CPython 3.12")
    if include_vllm is None:
        include_vllm = architecture == "amd64"
    if include_vllm:
        vllm_python_version = normalize_python_version(
            vllm_python_version or target_python
        )
        if vllm_python_version not in normalized_python_versions:
            raise DependencyError(
                "Local vLLM Python must be present in update compatibility"
            )
    else:
        vllm_python_version = None
    apt = _validate_apt(bundle)
    wheels = _validate_wheels(bundle)
    requirements = _requirements_fingerprints(
        repo_root,
        bundle,
        architecture=architecture,
    )
    if resolve:
        _resolve_with_pip(
            bundle,
            repo_root,
            python,
            architecture=architecture,
            include_vllm=include_vllm,
            python_versions=normalized_python_versions,
            vllm_python_version=vllm_python_version,
        )
    return {
        "format": MANIFEST_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": dependency_target(architecture, target_os_release),
        "update_compatibility": {
            "os_releases": list(normalized_os_releases),
            "python_versions": list(normalized_python_versions),
        },
        "requirements_sha256": requirements,
        "apt": apt,
        "wheelhouse": wheels,
        "pip_resolution": {
            "target_python": target_python,
            "target_pythons": list(normalized_python_versions),
            "vllm": (
                "pinned-ngc-container"
                if architecture == "arm64"
                else "0.25.0" if include_vllm else "external"
            ),
            "vllm_python": (
                vllm_python_version
                if include_vllm and architecture == "amd64"
                else None
            ),
            "container_packages": (
                [
                    "numpy==2.1.0",
                    "torch>=2.1.0",
                    "torchvision>=0.16.0",
                    "CUDA-enabled NVIDIA runtime",
                ]
                if architecture == "arm64"
                else []
            ),
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
    target = payload.get("target")
    if not isinstance(target, Mapping):
        raise DependencyError("Offline dependencies have no target platform")
    try:
        architecture = normalize_architecture(str(target.get("architecture") or ""))
    except DependencyError as exc:
        raise DependencyError("Offline dependencies target an unsupported architecture") from exc
    os_release = str(target.get("os_release") or "")
    if dict(target) != dependency_target(architecture, os_release):
        raise DependencyError("Offline dependencies target a different OS/Python platform")
    compatibility = payload.get("update_compatibility")
    if not isinstance(compatibility, Mapping):
        raise DependencyError("Offline dependencies have no update compatibility matrix")
    os_releases = tuple(str(value) for value in compatibility.get("os_releases") or ())
    python_versions = tuple(str(value) for value in compatibility.get("python_versions") or ())
    normalized_os_releases = tuple(normalize_os_release(value) for value in os_releases)
    normalized_python_versions = normalize_python_versions(python_versions)
    if os_release not in normalized_os_releases:
        raise DependencyError("Fresh-install OS is absent from update compatibility")
    if default_python_for_os(os_release) not in normalized_python_versions:
        raise DependencyError("Fresh-install Python is absent from update compatibility")
    resolution = payload.get("pip_resolution")
    if not isinstance(resolution, Mapping) or resolution.get("verified") is not True:
        raise DependencyError("Wheelhouse was not proven by the offline pip resolver")
    if list(normalized_python_versions) != list(resolution.get("target_pythons") or ()):
        raise DependencyError("Wheelhouse resolver coverage differs from update compatibility")

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
        current = _requirements_fingerprints(
            repo_root,
            bundle,
            architecture=architecture,
        )
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
    parser.add_argument(
        "--architecture",
        default="amd64",
        choices=sorted(SUPPORTED_ARCHITECTURES),
    )
    parser.add_argument(
        "--external-vllm",
        action="store_true",
        help="Resolve the EVA application only; the target must provide its VLM endpoint.",
    )
    parser.add_argument(
        "--vllm-python-version",
        choices=sorted(SUPPORTED_PYTHON_VERSIONS),
        help=(
            "Pinned Python runtime used by local x64 vLLM; may differ from the "
            "fresh host's system Python."
        ),
    )
    parser.add_argument(
        "--target-os-release",
        default="24.04",
        choices=sorted(SUPPORTED_OS_RELEASES),
        help="Fresh-install Ubuntu release carried by apt/.",
    )
    parser.add_argument(
        "--update-os-release",
        action="append",
        choices=sorted(SUPPORTED_OS_RELEASES),
        help="Ubuntu release accepted for in-place updates; repeat as needed.",
    )
    parser.add_argument(
        "--update-python-version",
        action="append",
        choices=sorted(SUPPORTED_PYTHON_VERSIONS),
        help="Existing venv Python accepted for updates; repeat as needed.",
    )
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
                architecture=str(args.architecture),
                include_vllm=not bool(args.external_vllm),
                vllm_python_version=args.vllm_python_version,
                target_os_release=str(args.target_os_release),
                update_os_releases=args.update_os_release,
                update_python_versions=args.update_python_version,
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
