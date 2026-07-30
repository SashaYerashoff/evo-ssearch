#!/usr/bin/env python3
"""Build the small, non-interactive EVA appliance installer Debian package.

Application source, wheels and model assets intentionally remain outside this
package.  The package installs a stable bootstrap command; that command then
consumes a versioned offline bundle from USB, ISO or a local mirror.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "eva-ai-appliance-installer"
VERSION_TEXT = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
VERSION_MATCH = re.search(r"\d+(?:\.\d+)+", VERSION_TEXT)
if VERSION_MATCH is None:
    raise RuntimeError(f"Cannot derive a Debian version from {VERSION_TEXT!r}")
DEFAULT_VERSION = VERSION_MATCH.group(0)


def _write(path: Path, text: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)


def build(output_dir: Path, version: str, architecture: str) -> Path:
    dpkg_deb = shutil.which("dpkg-deb")
    if not dpkg_deb:
        raise SystemExit("dpkg-deb is required to build the installer package.")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{PACKAGE}_{version}_{architecture}.deb"
    with tempfile.TemporaryDirectory(prefix="eva-ai-installer-deb-") as temporary:
        tree = Path(temporary) / "root"
        library = tree / "usr/lib/eva-ai-installer"
        library.mkdir(parents=True)
        for name in (
            "install_port_appliance.py",
            "validate_appliance_config.py",
            "eva_appliance_doctor.py",
            "wait_openai_endpoint.py",
        ):
            shutil.copy2(ROOT / "scripts" / name, library / name)
            (library / name).chmod(0o755)

        _write(
            tree / "usr/sbin/eva-ai-install",
            """#!/bin/sh
set -eu
exec python3 /usr/lib/eva-ai-installer/install_port_appliance.py "$@"
""",
            0o755,
        )
        _write(
            tree / "usr/sbin/eva-ai-doctor",
            """#!/bin/sh
set -eu
exec python3 /usr/lib/eva-ai-installer/eva_appliance_doctor.py "$@"
""",
            0o755,
        )
        _write(
            tree / "DEBIAN/control",
            f"""Package: {PACKAGE}
Version: {version}
Section: admin
Priority: optional
Architecture: {architecture}
Maintainer: EVA AI Engineering
Depends: python3 (>= 3.12)
Description: Offline EVA AI appliance bootstrap and diagnostics
 Installs stable, non-interactive bootstrap and diagnostic commands.
 Application code, Python wheels and model assets are consumed from a
 separately versioned and checksummed offline bundle.
""",
        )
        _write(
            tree / "DEBIAN/postinst",
            """#!/bin/sh
set -eu
install -d -o root -g root -m 0700 /var/lib/eva-ai-installer
echo "EVA installer ready. Run: sudo eva-ai-install --bundle-root /path/to/bundle"
""",
            0o755,
        )
        documentation = tree / "usr/share/doc" / PACKAGE
        documentation.mkdir(parents=True)
        shutil.copy2(
            ROOT / "deployment" / "debian" / "README.md",
            documentation / "README.md",
        )
        tree.chmod(0o755)
        for path in tree.rglob("*"):
            if path.is_dir():
                path.chmod(0o755)
            elif path.stat().st_mode & 0o111:
                path.chmod(0o755)
            else:
                path.chmod(0o644)
        subprocess.run(
            (
                dpkg_deb,
                "--build",
                "--root-owner-group",
                str(tree),
                str(destination),
            ),
            check=True,
        )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "dist" / "debian",
    )
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--architecture", default="amd64")
    args = parser.parse_args()
    package = build(args.output_dir.resolve(), args.version, args.architecture)
    print(package)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
