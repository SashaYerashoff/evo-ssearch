#!/usr/bin/env python3
"""Create the portable manifest and checksum inventory for the port USB."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


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
    if str(source.get("release_flavor") or "") != "ventspils-maritime-client":
        raise SystemExit("Unexpected release flavor in SOURCE_REVISION.json")
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
        "repo/migrations/versions/20260801_0011_incidents.py",
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
    manifest = {
        "format": 1,
        "version": version,
        "schema_head": "20260801_0011",
        "release_flavor": "ventspils-maritime-client",
        "source": source,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": {
            "os": "Ubuntu Server 24.04 amd64",
            "gpu": "NVIDIA GeForce RTX 4070 Super 12 GB",
            "cpu": "Intel Core i9 14th Gen",
            "ram_gib": 64,
            "channels": 8,
        },
        "payload_bytes": payload_bytes,
        "minimum_free_bytes": max(45 * 1024**3, payload_bytes + 25 * 1024**3),
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
