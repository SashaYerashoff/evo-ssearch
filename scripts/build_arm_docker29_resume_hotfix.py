#!/usr/bin/env python3
"""Build the single-file Docker 29 resume hotfix for ARM bundle 9063d2f."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "scripts" / "install_port_appliance.py"
SOURCE_COMMIT = "9063d2ffce9d835c22ef3ee22fd5ae5b2ca455b7"
ORIGINAL_INSTALLER_SHA256 = (
    "673cf816c2f9e9b29b5edb10926da77f7ca726fc9636ebf9c09eb3a0a393dcea"
)
IMAGE_CONFIG_DIGEST = (
    "sha256:5f79999e8001200efe1bacff71758a1ac459c83707f4ddab74311996863e17ba"
)
IMAGE_MANIFEST_DIGEST = (
    "sha256:2652b56f319448cb89d7f0307bb897b95004a4f19e9d179a72d0af75b07cddd3"
)
IMAGE_TAG = "eva-ai/spark-runtime:0.8.7-arm64"


def render() -> str:
    installer = INSTALLER.read_bytes()
    installer_sha256 = hashlib.sha256(installer).hexdigest()
    compressed = gzip.compress(installer, compresslevel=9, mtime=0)
    encoded = base64.b64encode(compressed).decode("ascii")
    encoded_lines = "\n".join(textwrap.wrap(encoded, 76))
    return f"""#!/usr/bin/env bash
set -Eeuo pipefail

# EVA AI field hotfix: resume only the interrupted 0.8.7 ARM64 fresh install
# from bundle 9063d2f on Docker classic or containerd-backed image stores.
# It does not edit Docker configuration, the bundle, PostgreSQL, or EVA data.

EXPECTED_SOURCE_COMMIT={SOURCE_COMMIT!r}
EXPECTED_ORIGINAL_INSTALLER_SHA256={ORIGINAL_INSTALLER_SHA256!r}
EXPECTED_FIXED_INSTALLER_SHA256={installer_sha256!r}
EXPECTED_IMAGE_CONFIG_DIGEST={IMAGE_CONFIG_DIGEST!r}
EXPECTED_IMAGE_MANIFEST_DIGEST={IMAGE_MANIFEST_DIGEST!r}
EXPECTED_IMAGE_TAG={IMAGE_TAG!r}
STATE_FILE=/var/lib/eva-ai-installer/install-state.json
HOTFIX_ROOT=/var/lib/eva-ai-installer/hotfixes/docker29-9063d2f

die() {{
  echo "HOTFIX ERROR: $*" >&2
  exit 1
}}

if [[ $EUID -ne 0 ]]; then
  exec sudo -- "$0" "$@"
fi

BUNDLE_ROOT=${{1:-$PWD}}
BUNDLE_ROOT=$(readlink -f -- "$BUNDLE_ROOT")
[[ -d "$BUNDLE_ROOT" ]] || die "bundle directory not found: $BUNDLE_ROOT"
[[ -f "$BUNDLE_ROOT/manifest.json" ]] || die "manifest.json not found under $BUNDLE_ROOT"
[[ -f "$BUNDLE_ROOT/SOURCE_REVISION.json" ]] || die "SOURCE_REVISION.json not found under $BUNDLE_ROOT"
[[ -f "$BUNDLE_ROOT/install_port_appliance.py" ]] || die "original installer not found under $BUNDLE_ROOT"
[[ -f "$BUNDLE_ROOT/offline_bundle_dependencies.py" ]] || die "offline dependency verifier not found under $BUNDLE_ROOT"
[[ -f "$STATE_FILE" ]] || die "no interrupted fresh-install journal at $STATE_FILE"

echo "EVA AI ARM64 DOCKER IMAGE-STORE RESUME HOTFIX"
echo "Bundle: $BUNDLE_ROOT"
echo "This accepts only the exact 9063d2f bundle and its pinned runtime image."

python3 - "$BUNDLE_ROOT" "$STATE_FILE" \\
  "$EXPECTED_SOURCE_COMMIT" "$EXPECTED_ORIGINAL_INSTALLER_SHA256" \\
  "$EXPECTED_IMAGE_CONFIG_DIGEST" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
state_path = Path(sys.argv[2])
expected_commit = sys.argv[3]
expected_installer = sys.argv[4]
expected_config = sys.argv[5]

def fail(message):
    raise SystemExit(f"HOTFIX ERROR: {{message}}")

try:
    source = json.loads((root / "SOURCE_REVISION.json").read_text(encoding="utf-8"))
    manifest_path = root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    journal = json.loads(state_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    fail(f"cannot read bundle identity or installer journal: {{exc}}")

if source.get("commit") != expected_commit or source.get("working_tree_clean") is not True:
    fail("this hotfix is only for the clean ARM bundle built from 9063d2f")
if str(manifest.get("release_flavor") or "") != "universal-offline":
    fail("bundle is not the universal offline release")
target = (manifest.get("offline_dependencies") or {{}}).get("target") or {{}}
if str(target.get("architecture") or "") != "arm64" or str(target.get("os_release") or "") != "24.04":
    fail("bundle target is not Ubuntu 24.04 ARM64")
runtime = manifest.get("container_runtime") or {{}}
if str(runtime.get("image_id") or "") != expected_config:
    fail("bundle does not pin the expected 0.8.7 Spark image config digest")

installer_digest = hashlib.sha256((root / "install_port_appliance.py").read_bytes()).hexdigest()
if installer_digest != expected_installer:
    fail("the original bundle installer was modified or is not the affected 9063d2f version")
manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
if journal.get("bundle_id") != manifest_digest:
    fail("installer journal belongs to a different bundle")
if journal.get("status") != "failed" or journal.get("failed_phase") != "spark_container_runtime":
    fail("journal is not the known interrupted spark_container_runtime installation")
print("Bundle and interrupted-install journal identity: OK")
PY

command -v docker >/dev/null || die "docker command is missing"
docker info >/dev/null 2>&1 || die "Docker daemon is not available to root"
IMAGE_FACTS=$(docker image inspect --format '{{{{.Architecture}}}}|{{{{.Id}}}}' "$EXPECTED_IMAGE_TAG" 2>/dev/null) \\
  || die "the pinned Spark runtime tag is not loaded"
IMAGE_ARCH=${{IMAGE_FACTS%%|*}}
IMAGE_ID=${{IMAGE_FACTS#*|}}
[[ "$IMAGE_ARCH" == "arm64" ]] || die "loaded runtime architecture is $IMAGE_ARCH, expected arm64"
if [[ "$IMAGE_ID" != "$EXPECTED_IMAGE_CONFIG_DIGEST" && "$IMAGE_ID" != "$EXPECTED_IMAGE_MANIFEST_DIGEST" ]]; then
  die "loaded runtime identity $IMAGE_ID is not pinned by this release"
fi
echo "Pinned runtime image: OK ($IMAGE_ID)"

install -d -o root -g root -m 0700 "$HOTFIX_ROOT"
PAYLOAD="$HOTFIX_ROOT/install_port_appliance.docker29.py"
base64 -d <<'PAYLOAD_B64' | gzip -dc >"$PAYLOAD"
{encoded_lines}
PAYLOAD_B64
chmod 0700 "$PAYLOAD"
ACTUAL_FIXED_INSTALLER_SHA256=$(sha256sum "$PAYLOAD" | awk '{{print $1}}')
[[ "$ACTUAL_FIXED_INSTALLER_SHA256" == "$EXPECTED_FIXED_INSTALLER_SHA256" ]] \\
  || die "embedded fixed installer checksum mismatch"

MANIFEST_SHA256=$(sha256sum "$BUNDLE_ROOT/manifest.json" | awk '{{print $1}}')
echo "Fixed installer staged outside the release bundle: $PAYLOAD"
echo "The original START_EVA_AI.sh already verified this exact bundle before writing the matched journal."
echo "Continuing the idempotent fresh-install engine; you will be asked for the installation answers again."

set +e
PYTHONPATH="$BUNDLE_ROOT" \\
EVA_OFFLINE_BUNDLE_PREFLIGHT_SHA256="$MANIFEST_SHA256" \\
/usr/bin/python3 "$PAYLOAD" --bundle-root "$BUNDLE_ROOT"
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  die "fixed installer stopped with exit $rc; preserve the new terminal output and journal"
fi

echo
echo "HOTFIXED INSTALLATION COMPLETE"
echo "Generate the standard deployment report with:"
echo "  cd '$BUNDLE_ROOT'"
echo "  sudo ./START_EVA_AI.sh --mode report"
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path.cwd() / "EVA_ARM_DOCKER29_RESUME_FIX_9063d2f.sh",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(), encoding="utf-8")
    output.chmod(0o755)
    output_digest = hashlib.sha256(output.read_bytes()).hexdigest()
    checksum = output.with_name(output.name + ".sha256")
    checksum.write_text(f"{output_digest}  {output.name}\n", encoding="utf-8")
    print(output)
    print(checksum)
    print(output_digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
