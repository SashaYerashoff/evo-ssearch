#!/usr/bin/env python3
"""Restore a code snapshot exactly while preserving adopted runtime data."""

from __future__ import annotations

import argparse
import fnmatch
import shutil
import tarfile
import tempfile
from pathlib import Path


PRESERVED_ROOTS = {
    ".git", ".local", ".venv", "dist", "node_modules",
    "detections_archive", "video", "models",
}
PRESERVED_NAMES = {
    "probes_store.json", "luxriot_summary_state.json", "luxriot_rollups_cache.json",
}
PRESERVED_PATTERNS = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.sqlite3", "*.db", "*.log")


def preserved(relative: Path) -> bool:
    parts = relative.parts
    if not parts:
        return False
    if parts[0] in PRESERVED_ROOTS or "__pycache__" in parts:
        return True
    name = parts[-1]
    return name in PRESERVED_NAMES or any(fnmatch.fnmatch(name, pattern) for pattern in PRESERVED_PATTERNS)


def remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def clean_new_code(target: Path, source: Path, relative: Path = Path()) -> None:
    if preserved(relative):
        return
    if not target.exists() and not target.is_symlink():
        return
    if not source.exists() and not source.is_symlink():
        remove(target)
        return
    target_is_dir = target.is_dir() and not target.is_symlink()
    source_is_dir = source.is_dir() and not source.is_symlink()
    if target_is_dir != source_is_dir:
        remove(target)
        return
    if target_is_dir:
        for child in list(target.iterdir()):
            clean_new_code(child, source / child.name, relative / child.name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--app-dir", required=True, type=Path)
    args = parser.parse_args()
    archive = args.archive.resolve()
    app_dir = args.app_dir.resolve()
    app_parent = app_dir.parent
    app_name = app_dir.name

    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        if not members or any(Path(member.name).parts[:1] != (app_name,) for member in members):
            raise SystemExit("snapshot has an unexpected top-level path")
        if any(member.name.startswith("/") or ".." in Path(member.name).parts for member in members):
            raise SystemExit("snapshot contains an unsafe path")
        with tempfile.TemporaryDirectory(prefix="eva-rollback-") as temp_name:
            temp_root = Path(temp_name)
            handle.extractall(temp_root, filter="data")
            snapshot_app = temp_root / app_name
            app_dir.mkdir(parents=True, exist_ok=True)
            clean_new_code(app_dir, snapshot_app)

    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(app_parent, filter="data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
