from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "offline_bundle_dependencies_test",
    ROOT / "scripts" / "offline_bundle_dependencies.py",
)
assert SPEC and SPEC.loader
dependencies = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dependencies
SPEC.loader.exec_module(dependencies)


def _wheel(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("demo/__init__.py", "")
        archive.writestr(
            "demo-1.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n",
        )
        archive.writestr(
            "demo-1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nTag: py3-none-any\n",
        )


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    apt = bundle / "apt"
    wheelhouse = bundle / "wheelhouse"
    repo = bundle / "repo"
    apt.mkdir(parents=True)
    wheelhouse.mkdir()
    repo.mkdir()
    deb = apt / "demo_1_amd64.deb"
    deb.write_bytes(b"offline-deb")
    digest = hashlib.sha256(deb.read_bytes()).hexdigest()
    stanza = "\n".join(
        (
            "Package: demo",
            "Version: 1",
            "Architecture: amd64",
            "Filename: ./demo_1_amd64.deb",
            f"Size: {deb.stat().st_size}",
            f"SHA256: {digest}",
            "",
        )
    )
    with gzip.open(apt / "Packages.gz", "wt", encoding="utf-8") as handle:
        handle.write(stanza)
    (apt / "package-names.txt").write_text("demo\n", encoding="utf-8")
    _wheel(wheelhouse / "demo-1.0-py3-none-any.whl")
    (repo / "requirements.txt").write_text("demo==1.0\n", encoding="utf-8")
    (repo / "requirements-db.txt").write_text("", encoding="utf-8")
    (bundle / "constraints-port-4070s.txt").write_text("demo==1.0\n", encoding="utf-8")
    return bundle


def test_dependency_manifest_detects_artifact_corruption(tmp_path):
    bundle = _bundle(tmp_path)
    payload = dependencies.build_manifest(bundle, repo_root=bundle / "repo")
    payload["pip_resolution"]["verified"] = True
    (bundle / dependencies.MANIFEST_NAME).write_text(
        json.dumps(payload), encoding="utf-8"
    )

    dependencies.verify_manifest(bundle, repo_root=bundle / "repo")
    (bundle / "apt" / "demo_1_amd64.deb").write_bytes(b"corrupt")

    with pytest.raises(dependencies.DependencyError, match="size mismatch"):
        dependencies.verify_manifest(bundle, repo_root=bundle / "repo")


def test_dependency_manifest_rejects_requirements_drift(tmp_path):
    bundle = _bundle(tmp_path)
    payload = dependencies.build_manifest(bundle, repo_root=bundle / "repo")
    payload["pip_resolution"]["verified"] = True
    (bundle / dependencies.MANIFEST_NAME).write_text(
        json.dumps(payload), encoding="utf-8"
    )
    (bundle / "repo" / "requirements.txt").write_text("demo==2.0\n", encoding="utf-8")

    with pytest.raises(dependencies.DependencyError, match="different requirements"):
        dependencies.verify_manifest(bundle, repo_root=bundle / "repo")


def test_build_rejects_requested_apt_package_absent_from_index(tmp_path):
    bundle = _bundle(tmp_path)
    (bundle / "apt" / "package-names.txt").write_text("missing\n", encoding="utf-8")

    with pytest.raises(dependencies.DependencyError, match="requested packages"):
        dependencies.build_manifest(bundle, repo_root=bundle / "repo")


def test_build_rejects_stale_unindexed_deb(tmp_path):
    bundle = _bundle(tmp_path)
    (bundle / "apt" / "stale_1_amd64.deb").write_bytes(b"stale")

    with pytest.raises(dependencies.DependencyError, match="absent from Packages.gz"):
        dependencies.build_manifest(bundle, repo_root=bundle / "repo")
