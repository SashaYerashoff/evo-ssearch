#!/usr/bin/env python3
"""Create the portable manifest and checksum inventory for the port USB."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from offline_bundle_dependencies import DependencyError, verify_manifest


SPARK_RUNTIME_BASE_IMAGE = "nvcr.io/nvidia/vllm:26.07-py3"
SPARK_RUNTIME_BASE_MANIFEST_DIGEST = (
    "sha256:1de8e6bfdb4c81c1f31a806cc9b13b5c6352714a7cec87f4d24964bcc91159b2"
)
SPARK_RUNTIME_BASE_IMAGE_ID = (
    "sha256:4c704f1343c7cb3aa7ea5cc57cab5fa1ed1a2160daf4f57d1e0c06fc1e2c7dbb"
)
SPARK_RUNTIME_IMAGE = "eva-ai/spark-runtime:0.8.7-arm64"
SPARK_RUNTIME_IMAGE_ID = (
    "sha256:5f79999e8001200efe1bacff71758a1ac459c83707f4ddab74311996863e17ba"
)
SPARK_RUNTIME_IMAGE_MANIFEST_DIGEST = (
    "sha256:2652b56f319448cb89d7f0307bb897b95004a4f19e9d179a72d0af75b07cddd3"
)
SPARK_RUNTIME_ARCHIVE = "container/eva-spark-runtime-0.8.7-arm64.tar.zst"
SPARK_VLM_REPO = "Qwen/Qwen3-VL-4B-Instruct"
SPARK_VLM_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
SPARK_NUMPY_VERSION = "2.1.0"
SPARK_PIP_CONSTRAINT = "/etc/pip/constraint.txt"
SPARK_SIGLIP_DTYPE = "float32"
SPARK_FFMPEG_BIN = "/usr/local/bin/eva-ffmpeg"
SPARK_FFMPEG_WRAPPER_RELATIVE = (
    "repo/deployment/spark_gb10/runtime-image/eva-ffmpeg"
)
X64_VLLM_PYTHON_VERSION = "3.12.13"
X64_VLLM_PYTHON_DIRECTORY = "cpython-3.12.13-linux-x86_64-gnu"
X64_VLLM_PYTHON_ARCHIVE = "python/cpython-3.12.13-linux-x86_64-gnu.tar.gz"
X64_VLLM_PYTHON_ARCHIVE_SHA256 = (
    "22803d96bc57ce0645aff383b4ab5076f7d19ea5ece5b64583ca2448841ed261"
)


def semantic_index_description(architecture: str) -> str:
    """Describe the packaged embedder without lying about its runtime precision."""

    if architecture == "arm64":
        return "Google SigLIP2 base patch16 224 FP32 / NVIDIA GB10 CUDA"
    return (
        "Google SigLIP2 base patch16 224 FP16 / shared CUDA; "
        "OpenAI CLIP ViT-B/32 retained for comparison"
    )


@lru_cache(maxsize=None)
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


def verify_siglip2_checksum_manifest(root: Path) -> int:
    """Fail closed unless the packaged SigLIP2 cache is content-complete."""

    cache_root = root / "models" / "huggingface"
    checksum_path = cache_root / "SHA256SUMS"
    if not checksum_path.is_file():
        raise SystemExit("SigLIP2 checksum manifest is missing: models/huggingface/SHA256SUMS")
    try:
        cache_root_resolved = cache_root.resolve(strict=True)
        checked = 0
        for raw_line in checksum_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.+)", line)
            if match is None:
                raise ValueError("contains an invalid row")
            expected, relative = match.groups()
            candidate = cache_root / relative
            resolved = candidate.resolve(strict=True)
            if cache_root_resolved != resolved and cache_root_resolved not in resolved.parents:
                raise ValueError(f"path escapes model cache: {relative}")
            if not candidate.is_file():
                raise ValueError(f"path is not a file: {relative}")
            if digest(candidate).lower() != expected.lower():
                raise ValueError(f"checksum mismatch: {relative}")
            checked += 1
        if checked == 0:
            raise ValueError("is empty")
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Invalid SigLIP2 checksum manifest: {exc}") from exc
    return checked


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


def spark_runtime_payload(root: Path, architecture: str) -> tuple[dict[str, Any] | None, list[str]]:
    if architecture != "arm64":
        return None, []
    contract_path = root / "runtime-container.json"
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid Spark runtime contract: {exc}") from exc
    expected = {
        "format": 1,
        "engine": "docker",
        "base_image": SPARK_RUNTIME_BASE_IMAGE,
        "base_manifest_digest": SPARK_RUNTIME_BASE_MANIFEST_DIGEST,
        "base_image_id": SPARK_RUNTIME_BASE_IMAGE_ID,
        "image": SPARK_RUNTIME_IMAGE,
        "image_id": SPARK_RUNTIME_IMAGE_ID,
        "image_manifest_digest": SPARK_RUNTIME_IMAGE_MANIFEST_DIGEST,
        "platform": "linux/arm64",
        "model": SPARK_VLM_REPO,
        "model_revision": SPARK_VLM_REVISION,
        "numpy": SPARK_NUMPY_VERSION,
        "pip_constraint": SPARK_PIP_CONSTRAINT,
        "weight_quantization": "online-fp8-w8a8",
        "kv_cache_dtype": "bfloat16",
        "vision_attention_dtype": "bfloat16",
        "siglip_dtype": SPARK_SIGLIP_DTYPE,
        "ffmpeg_bin": SPARK_FFMPEG_BIN,
        "ffmpeg_h264_decoder": "required",
    }
    if contract != expected:
        raise SystemExit("Spark runtime contract does not match the pinned release image.")
    wrapper = root / SPARK_FFMPEG_WRAPPER_RELATIVE
    if not wrapper.is_file() or not (wrapper.stat().st_mode & 0o111):
        raise SystemExit(
            "ARM64 release is incomplete: the Spark FFmpeg wrapper is missing "
            "or is not executable."
        )
    critical = ["runtime-container.json", SPARK_FFMPEG_WRAPPER_RELATIVE]
    archive = root / SPARK_RUNTIME_ARCHIVE
    if not archive.is_file():
        raise SystemExit(
            "ARM64 release is incomplete: the pinned NGC runtime archive is missing."
        )
    archive_manifest = subprocess.run(
        ("tar", "--zstd", "-xOf", archive, "manifest.json"),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if archive_manifest.returncode or SPARK_RUNTIME_IMAGE_ID.removeprefix("sha256:") not in (
        archive_manifest.stdout
    ):
        raise SystemExit(
            "Spark runtime archive does not contain the pinned ARM64 image ID."
        )
    archive_index = subprocess.run(
        ("tar", "--zstd", "-xOf", archive, "index.json"),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    try:
        index = json.loads(archive_index.stdout) if not archive_index.returncode else {}
        manifest_digests = {
            str(item.get("digest") or "")
            for item in index.get("manifests", [])
            if isinstance(item, dict)
        }
    except json.JSONDecodeError:
        manifest_digests = set()
    if SPARK_RUNTIME_IMAGE_MANIFEST_DIGEST not in manifest_digests:
        raise SystemExit(
            "Spark runtime archive does not contain the pinned ARM64 manifest digest."
        )
    contract["archive"] = SPARK_RUNTIME_ARCHIVE
    contract["archive_sha256"] = digest(archive)
    critical.append(SPARK_RUNTIME_ARCHIVE)
    return contract, critical


def x64_python_runtime_payload(
    root: Path,
    architecture: str,
) -> tuple[dict[str, str] | None, list[str]]:
    """Bind the standalone local-vLLM interpreter into an x64 release."""

    if architecture != "amd64":
        return None, []
    archive = root / X64_VLLM_PYTHON_ARCHIVE
    if not archive.is_file():
        raise SystemExit(
            "x64 release is incomplete: the local vLLM Python runtime is missing."
        )
    archive_digest = digest(archive)
    if archive_digest != X64_VLLM_PYTHON_ARCHIVE_SHA256:
        raise SystemExit(
            "x64 local vLLM Python runtime checksum does not match the pinned release."
        )
    required = {
        f"{X64_VLLM_PYTHON_DIRECTORY}/BUILD",
        f"{X64_VLLM_PYTHON_DIRECTORY}/bin/python3.12",
        f"{X64_VLLM_PYTHON_DIRECTORY}/lib/libpython3.12.so.1.0",
    }
    try:
        with tarfile.open(archive, "r:gz") as payload:
            names = {member.name.removeprefix("./") for member in payload.getmembers()}
    except (OSError, tarfile.TarError) as exc:
        raise SystemExit(f"Invalid x64 local vLLM Python runtime: {exc}") from exc
    missing = sorted(required - names)
    if missing:
        raise SystemExit(
            "x64 local vLLM Python runtime is incomplete: " + ", ".join(missing)
        )
    contract = {
        "implementation": "cpython",
        "version": X64_VLLM_PYTHON_VERSION,
        "platform": "linux/x86_64",
        "directory": X64_VLLM_PYTHON_DIRECTORY,
        "archive": X64_VLLM_PYTHON_ARCHIVE,
        "archive_sha256": archive_digest,
    }
    return contract, [X64_VLLM_PYTHON_ARCHIVE]


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
    architecture = str(dependency_manifest["target"]["architecture"])
    target_os_release = str(dependency_manifest["target"].get("os_release") or "")
    target_os_label = str(dependency_manifest["target"].get("os") or "")
    local_vllm_mode = str(dependency_manifest["pip_resolution"].get("vllm") or "")
    verify_siglip2_checksum_manifest(root)
    update_packages, update_critical_files = bundled_update_packages(root)
    container_runtime, container_critical_files = spark_runtime_payload(root, architecture)
    python_runtime, python_runtime_critical_files = x64_python_runtime_payload(
        root,
        architecture,
    )
    spark_model_critical_files = (
        [
            str(path.relative_to(root))
            for path in sorted((root / "models" / "qwen3-vl-4b").rglob("*"))
            if path.is_file()
        ]
        if architecture == "arm64"
        else []
    )
    if architecture == "arm64" and not any(
        path.endswith(".safetensors") for path in spark_model_critical_files
    ):
        raise SystemExit("ARM64 release has no Qwen3-VL-4B safetensors payload.")
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
        *(
            ("repo/deployment/spark_gb10/FACTORY_ACCEPTANCE.md",)
            if architecture == "arm64"
            else ()
        ),
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
        "models/huggingface/SHA256SUMS",
        *(("migration-plans/0006-to-0013.sql",) if release_flavor == "universal-offline" else ()),
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
            f"eva-ai-appliance-installer_{debian_version}_{architecture}.deb"
        ),
        *(
            (
                "models/qwen3-vl-4b-awq/model.safetensors",
                "models/qwen3.5-9b-mtp/Qwen3.5-9B-Q4_K_M.gguf",
                "models/clip/ViT-B-32.pt",
            )
            if architecture == "amd64"
            else ()
        ),
        *container_critical_files,
        *python_runtime_critical_files,
        *spark_model_critical_files,
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
            "os": f"{target_os_label} ARM64",
            "gpu": "NVIDIA GB10 / Spark-class integrated CUDA GPU",
            "cpu": "ARM64 vendor appliance platform",
            "ram_gib": 120,
            "channels": "site profile; bundled local Qwen3-VL-4B inference",
        }
        if architecture == "arm64"
        else
        {
            "os": f"{target_os_label} amd64",
            "gpu": "NVIDIA RTX 4070 Super / RTX 5070 Ti or newer (12+ GB VRAM)",
            "cpu": "x86_64 with AVX2; 16+ logical CPUs recommended",
            "ram_gib": 64,
            "channels": "site profile (8 local single-GPU; external VLM scale-out supported)",
        }
        if release_flavor == "universal-offline"
        else {
            "os": f"Ubuntu Server {target_os_release} amd64",
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
        "minimum_free_bytes": (
            max(70 * 1024**3, payload_bytes + 35 * 1024**3)
            if architecture == "arm64"
            else max(45 * 1024**3, payload_bytes + 25 * 1024**3)
        ),
        "installation_modes": ["fresh", "resume", "update", "report"],
        "update_packages": update_packages,
        "offline_dependencies": {
            "manifest": "offline-dependencies.json",
            "apt_artifacts": len(dependency_manifest["apt"]["artifacts"]),
            "wheels": len(dependency_manifest["wheelhouse"]["artifacts"]),
            "target": dependency_manifest["target"],
            "update_compatibility": dependency_manifest["update_compatibility"],
        },
        **({"container_runtime": container_runtime} if container_runtime else {}),
        **({"python_runtime": python_runtime} if python_runtime else {}),
        "critical_sha256": critical,
        "models": {
            "live_vlm": (
                "Qwen3-VL-4B-Instruct online FP8 / pinned NGC vLLM container"
                if architecture == "arm64"
                else (
                    "external OpenAI-compatible VLM endpoint required"
                    if local_vllm_mode == "external"
                    else "Qwen3-VL-4B-Instruct AWQ / vLLM 0.25.0"
                )
            ),
            "deep_review": (
                "external endpoint or disabled"
                if architecture == "arm64"
                else "Qwen3.5-9B-MTP Q4_K_M / llama.cpp CPU"
            ),
            "semantic_index": semantic_index_description(architecture),
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
                (
                    "media_runtime=pinned-ngc-container"
                    if architecture == "arm64"
                    else "media_runtime=system-apt"
                ),
                f"media_runtime_platform=linux-{'arm64' if architecture == 'arm64' else 'x86_64'}",
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
