#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="${1:-/mnt/eva-llamacpp-lab/spark-arm64-dependency-seed}"
PYTHON_BIN="${PYTHON:-python3}"

case "${OUTPUT_ROOT}" in
  /|/home|/home/*/Projects|/mnt|/run/media/*)
    printf 'ERROR: dependency output is too broad: %s\n' "${OUTPUT_ROOT}" >&2
    exit 1
    ;;
esac

for command in docker dpkg-scanpackages gzip "${PYTHON_BIN}"; do
  command -v "${command}" >/dev/null 2>&1 || {
    printf 'ERROR: required build command is missing: %s\n' "${command}" >&2
    exit 1
  }
done

TEMP_ROOT="$(mktemp -d /mnt/eva-llamacpp-lab/eva-arm64-seed.XXXXXX)"
cleanup() {
  rm -rf "${TEMP_ROOT}"
}
trap cleanup EXIT

rm -rf "${OUTPUT_ROOT}/apt" "${OUTPUT_ROOT}/wheelhouse"
mkdir -p "${OUTPUT_ROOT}/apt" "${OUTPUT_ROOT}/wheelhouse"

STUB_ROOT="${TEMP_ROOT}/resolver-stubs"
mkdir -p "${STUB_ROOT}"
"${PYTHON_BIN}" - "${SCRIPT_DIR}/offline_bundle_dependencies.py" "${STUB_ROOT}" <<'PY'
import importlib.util
import sys
from pathlib import Path

module_path = Path(sys.argv[1])
output = Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("eva_offline_dependencies", module_path)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
for package, version in module.SPARK_VENDOR_RUNTIME_PACKAGES.items():
    module._write_resolver_stub(output, package, version)
PY

RESOLVER_CONSTRAINTS="${TEMP_ROOT}/resolver-constraints.txt"
"${PYTHON_BIN}" - \
  "${REPO_ROOT}/deployment/spark_gb10/constraints-spark-gb10.txt" \
  "${STUB_ROOT}" \
  "${RESOLVER_CONSTRAINTS}" <<'PY'
import sys
from pathlib import Path

declared, stubs, output = map(Path, sys.argv[1:])
torch = next(stubs.glob("torch-*.whl"))
torchvision = next(stubs.glob("torchvision-*.whl"))
numpy = next(stubs.glob("numpy-*.whl"))
output.write_text(
    declared.read_text(encoding="utf-8")
    + f"\nnumpy @ {numpy.as_uri()}\n"
    + f"torch @ {torch.as_uri()}\n"
    + f"torchvision @ {torchvision.as_uri()}\n",
    encoding="utf-8",
)
PY

"${PYTHON_BIN}" -m pip download \
  --dest "${OUTPUT_ROOT}/wheelhouse" \
  --only-binary=:all: \
  --platform manylinux_2_28_aarch64 \
  --platform manylinux_2_27_aarch64 \
  --platform manylinux2014_aarch64 \
  --platform manylinux_2_17_aarch64 \
  --implementation cp \
  --python-version 3.12 \
  --abi cp312 \
  --find-links "${STUB_ROOT}" \
  --constraint "${RESOLVER_CONSTRAINTS}" \
  --requirement "${REPO_ROOT}/requirements.txt" \
  --requirement "${REPO_ROOT}/requirements-db.txt" \
  --requirement "${REPO_ROOT}/requirements-cuda.txt"

# Resolver stubs prove the dependency graph while representing packages owned
# by the pinned NVIDIA container. They must never enter the installable payload.
rm -f \
  "${OUTPUT_ROOT}/wheelhouse/numpy-2.1.0-py3-none-any.whl" \
  "${OUTPUT_ROOT}/wheelhouse/torch-2.11.0-py3-none-any.whl" \
  "${OUTPUT_ROOT}/wheelhouse/torchvision-0.26.0-py3-none-any.whl"
if find "${OUTPUT_ROOT}/wheelhouse" -maxdepth 1 -type f \
    \( -iname 'numpy-*.whl' -o -iname 'torch-*.whl' -o -iname 'torchvision-*.whl' -o -iname 'vllm-*.whl' \) \
    -print -quit | grep -q .; then
  printf 'ERROR: vendor-owned NumPy/torch/vLLM wheel leaked into ARM wheelhouse.\n' >&2
  exit 1
fi

APT_WORK="${TEMP_ROOT}/apt-work"
mkdir -p "${APT_WORK}" "${OUTPUT_ROOT}/apt"
docker run --rm \
  --platform linux/amd64 \
  --env "HOST_UID=$(id -u)" \
  --env "HOST_GID=$(id -g)" \
  --mount "type=bind,src=${APT_WORK},dst=/work" \
  --mount "type=bind,src=${OUTPUT_ROOT}/apt,dst=/out" \
  --mount "type=bind,src=${REPO_ROOT}/deployment/spark_gb10/apt-packages-ubuntu-24.04.txt,dst=/package-names.txt,readonly" \
  ubuntu:24.04 \
  bash -Eeuc '
    mkdir -p /work/etc/apt /work/lists/partial /work/cache/archives/partial /work/state
    printf "%s\n" \
      "deb [trusted=yes] http://ports.ubuntu.com/ubuntu-ports noble main universe multiverse restricted" \
      "deb [trusted=yes] http://ports.ubuntu.com/ubuntu-ports noble-updates main universe multiverse restricted" \
      "deb [trusted=yes] http://ports.ubuntu.com/ubuntu-ports noble-security main universe multiverse restricted" \
      > /work/etc/apt/sources.list
    : > /work/state/status
    apt_options=(
      -o APT::Architecture=arm64
      -o APT::Architectures::=arm64
      -o Dir::Etc::sourcelist=/work/etc/apt/sources.list
      -o Dir::Etc::sourceparts=-
      -o Dir::State::status=/work/state/status
      -o Dir::State::lists=/work/lists
      -o Dir::Cache=/work/cache
      -o Acquire::Languages=none
      -o Acquire::Retries=3
    )
    apt-get "${apt_options[@]}" update
    mapfile -t packages < <(sed -e "/^[[:space:]]*#/d" -e "/^[[:space:]]*$/d" /package-names.txt)
    apt-get "${apt_options[@]}" --download-only --no-install-recommends -y install "${packages[@]}"
    cp /work/cache/archives/*.deb /out/
    chown -R "${HOST_UID}:${HOST_GID}" /work /out
  '

install -m 0644 \
  "${REPO_ROOT}/deployment/spark_gb10/apt-packages-ubuntu-24.04.txt" \
  "${OUTPUT_ROOT}/apt/package-names.txt"
(
  cd "${OUTPUT_ROOT}/apt"
  dpkg-scanpackages . /dev/null | gzip -9 > Packages.gz
)

while IFS= read -r -d '' package; do
  package_architecture="$(dpkg-deb -f "${package}" Architecture)"
  if [[ "${package_architecture}" != "arm64" && "${package_architecture}" != "all" ]]; then
    printf 'ERROR: non-ARM package found in Spark APT payload: %s (%s)\n' \
      "${package}" "${package_architecture}" >&2
    exit 1
  fi
done < <(find "${OUTPUT_ROOT}/apt" -maxdepth 1 -type f -name '*.deb' -print0)

printf 'ARM64 dependency seed ready: %s\n' "${OUTPUT_ROOT}"
printf '  apt artifacts: %s\n' "$(find "${OUTPUT_ROOT}/apt" -maxdepth 1 -name '*.deb' | wc -l)"
printf '  Python wheels: %s\n' "$(find "${OUTPUT_ROOT}/wheelhouse" -maxdepth 1 -name '*.whl' | wc -l)"
du -sh "${OUTPUT_ROOT}"
