#!/usr/bin/env python3
"""Create the portable manifest and checksum inventory for the port USB."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from offline_bundle_dependencies import DependencyError, verify_manifest


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _key_value_manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = raw.partition("=")
        if separator and key.strip():
            values[key.strip()] = value.strip()
    return values


def bundled_update_packages(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate optional standalone update packs and return critical paths.

    A universal USB may carry client-specific update packs beside the generic
    fresh/update engine.  Treating updates/ as an untracked convenience folder
    allowed a stale or partially copied updater to survive a new installer
    build.  Every present pack is now content-bound into the universal manifest.
    """

    updates_root = root / "updates"
    if not updates_root.exists():
        return [], []
    if not updates_root.is_dir():
        raise SystemExit("Invalid updates payload: updates/ is not a directory")

    packages: list[dict[str, Any]] = []
    critical: list[str] = []
    for package_dir in sorted(path for path in updates_root.iterdir() if path.is_dir()):
        archives = sorted(package_dir.glob("*.tar.gz"))
        if len(archives) != 1:
            raise SystemExit(
                f"Update pack {package_dir.name} must contain exactly one .tar.gz archive"
            )
        archive = archives[0]
        checksum = Path(f"{archive}.sha256")
        if not checksum.is_file():
            raise SystemExit(f"Update checksum is missing: {checksum.relative_to(root)}")
        checksum_fields = checksum.read_text(encoding="utf-8").strip().split()
        if len(checksum_fields) < 2:
            raise SystemExit(f"Invalid update checksum file: {checksum.relative_to(root)}")
        expected_digest = checksum_fields[0].lower()
        expected_name = checksum_fields[-1].lstrip("*")
        if expected_name != archive.name or not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
            raise SystemExit(f"Invalid update checksum contract: {checksum.relative_to(root)}")
        actual_digest = digest(archive)
        if actual_digest != expected_digest:
            raise SystemExit(f"Update archive checksum mismatch: {archive.relative_to(root)}")

        bundle_name = archive.name.removesuffix(".tar.gz")
        expanded = package_dir / bundle_name
        expanded_manifest = expanded / "manifest.txt"
        if not expanded_manifest.is_file():
            raise SystemExit(
                f"Expanded update manifest is missing: {expanded_manifest.relative_to(root)}"
            )
        update_manifest = _key_value_manifest(expanded_manifest)
        if update_manifest.get("bundle_name") != bundle_name:
            raise SystemExit(f"Expanded update identity mismatch: {expanded_manifest.relative_to(root)}")
        if update_manifest.get("working_tree_status") != "clean":
            raise SystemExit(f"Update pack was not built from a clean tree: {package_dir.name}")
        commit = str(update_manifest.get("git_commit") or "").strip()
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise SystemExit(f"Update pack has an invalid git commit: {package_dir.name}")

        safety_files = (
            expanded / "repo" / "scripts" / "database_preservation_guard.py",
            expanded / "repo" / "scripts" / "pg_with_dsn.py",
        )
        for path in safety_files:
            if not path.is_file():
                raise SystemExit(f"Update data-safety payload is missing: {path.relative_to(root)}")
        launchers = sorted(package_dir.glob("START*.sh"))
        if not launchers:
            raise SystemExit(f"Update pack has no START*.sh launcher: {package_dir.name}")

        package_critical = [archive, checksum, expanded_manifest, *safety_files, *launchers]
        critical.extend(str(path.relative_to(root)) for path in package_critical)
        packages.append(
            {
                "name": package_dir.name,
                "bundle_name": bundle_name,
                "version": update_manifest.get("version"),
                "git_commit": commit,
                "archive": str(archive.relative_to(root)),
                "archive_sha256": actual_digest,
                "checksum": str(checksum.relative_to(root)),
                "expanded_manifest": str(expanded_manifest.relative_to(root)),
                "launchers": [str(path.relative_to(root)) for path in launchers],
            }
        )
    return packages, critical


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    root = args.bundle.resolve()
    version = (root / "repo" / "VERSION").read_text(encoding="utf-8").strip()
    source_path = root / "SOURCE_REVISION.json"
    if not source_path.is_file():
        raise SystemExit("Missing critical payload: SOURCE_REVISION.json")
    try:
        source = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid SOURCE_REVISION.json: {exc}") from exc
    if source.get("working_tree_clean") is not True:
        raise SystemExit("Refusing to finalize an uncommitted port client bundle.")
    release_flavor = str(source.get("release_flavor") or "").strip()
    supported_flavors = {"ventspils-maritime-client", "universal-offline"}
    if release_flavor not in supported_flavors:
        raise SystemExit(
            "Unexpected release flavor in SOURCE_REVISION.json: "
            f"{release_flavor or '[missing]'}"
        )
    try:
        dependency_manifest = verify_manifest(root, repo_root=root / "repo")
    except DependencyError as exc:
        raise SystemExit(f"Offline dependencies are not releasable: {exc}") from exc
    update_packages, update_critical_files = bundled_update_packages(root)
    version_match = re.search(r"\d+(?:\.\d+)+", version)
    if version_match is None:
        raise SystemExit(f"Cannot derive a Debian version from {version!r}")
    debian_version = version_match.group(0)
    critical_files = (
        "SOURCE_REVISION.json",
        "repo/VERSION",
        "repo/camera_scene.py",
        "repo/maritime_profiles.py",
        "repo/docs/maritime_port_profile.md",
        "repo/react-ui/dist/index.html",
        "repo/requirements-cuda.txt",
        "repo/scripts/database_preservation_guard.py",
        "repo/scripts/pg_with_dsn.py",
        "repo/migrations/versions/20260801_0011_incidents.py",
        "repo/migrations/versions/20260805_0012_incident_temporal_memory.py",
        "repo/migrations/versions/20260805_0013_archive_source_channel_page_index.py",
        "eva_offline_deploy.py",
        "offline_bundle_dependencies.py",
        "offline-dependencies.json",
        "START_EVA_AI.sh",
        *(("migration-plans/0006-to-0013.sql",) if release_flavor == "universal-offline" else ()),
        "models/qwen3-vl-4b-awq/model.safetensors",
        "models/qwen3.5-9b-mtp/Qwen3.5-9B-Q4_K_M.gguf",
        "models/clip/ViT-B-32.pt",
        (
            "models/huggingface/models--google--siglip2-base-patch16-224/"
            "snapshots/75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2/"
            "model.safetensors"
        ),
        *(
            (
                "models/huggingface/models--google--siglip2-base-patch16-224/"
                "snapshots/75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2/"
                f"{filename}"
            )
            for filename in (
                "config.json",
                "preprocessor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
            )
        ),
        (
            "installer-deb/"
            f"eva-ai-appliance-installer_{debian_version}_amd64.deb"
        ),
        *update_critical_files,
    )
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise SystemExit("Bundle is empty.")
    critical = {}
    for relative in critical_files:
        path = root / relative
        if not path.is_file():
            raise SystemExit(f"Missing critical payload: {relative}")
        critical[relative] = digest(path)
    payload_bytes = sum(path.stat().st_size for path in files)
    target = (
        {
            "os": "Ubuntu 24.04 LTS amd64",
            "gpu": "NVIDIA RTX 4070 Super / RTX 5070 Ti or newer (12+ GB VRAM)",
            "cpu": "x86_64 with AVX2; 16+ logical CPUs recommended",
            "ram_gib": 64,
            "channels": "site profile (8 local single-GPU; external VLM scale-out supported)",
        }
        if release_flavor == "universal-offline"
        else {
            "os": "Ubuntu Server 24.04 amd64",
            "gpu": "NVIDIA GeForce RTX 4070 Super 12 GB",
            "cpu": "Intel Core i9 14th Gen",
            "ram_gib": 64,
            "channels": 8,
        }
    )
    manifest = {
        "format": 2,
        "version": version,
        "schema_head": "20260805_0013",
        "release_flavor": release_flavor,
        "source": source,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": target,
        "payload_bytes": payload_bytes,
        "minimum_free_bytes": max(45 * 1024**3, payload_bytes + 25 * 1024**3),
        "installation_modes": ["fresh", "update", "report"],
        "update_packages": update_packages,
        "offline_dependencies": {
            "manifest": "offline-dependencies.json",
            "apt_artifacts": len(dependency_manifest["apt"]["artifacts"]),
            "wheels": len(dependency_manifest["wheelhouse"]["artifacts"]),
            "target": dependency_manifest["target"],
        },
        "critical_sha256": critical,
        "models": {
            "live_vlm": "Qwen3-VL-4B-Instruct AWQ / vLLM 0.25.0",
            "deep_review": "Qwen3.5-9B-MTP Q4_K_M / llama.cpp CPU",
            "semantic_index": (
                "Google SigLIP2 base patch16 224 FP16 / shared CUDA; "
                "OpenAI CLIP ViT-B/32 retained for comparison"
            ),
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (root / "manifest.txt").write_text(
        "\n".join(
            (
                f"bundle_name=eva-ai-{release_flavor}",
                f"created_at={manifest['created_at']}",
                f"git_branch={source.get('branch') or 'unknown'}",
                f"git_commit={source.get('commit') or 'unknown'}",
                f"version={version}",
                "working_tree_status=clean",
                "wheelhouse=included",
                "media_runtime=included",
                "media_runtime_platform=linux-x86_64",
                f"schema_head={manifest['schema_head']}",
                f"release_flavor={release_flavor}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    files = sorted(path for path in root.rglob("*") if path.is_file())
    checksum_path = root / "SHA256SUMS"
    with checksum_path.open("w", encoding="utf-8") as handle:
        for path in files:
            if path.name in {"SHA256SUMS", "manifest.json"}:
                continue
            handle.write(f"{digest(path)}  {path.relative_to(root)}\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
