#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="${1:-/mnt/eva-llamacpp-lab/x64-multi-python-dependency-seed}"
TARGET_OS_RELEASE="${2:-${EVA_PORT_OS_RELEASE:-24.04}}"
PYTHON_BIN="${PYTHON:-python3}"
PYTHON_VERSIONS="${EVA_X64_PYTHON_VERSIONS:-3.12 3.13 3.14}"
VLLM_PYTHON_VERSION="${EVA_X64_VLLM_PYTHON_VERSION:-3.12}"
RESUME_BUILD="${EVA_X64_SEED_RESUME:-0}"
PIP_CACHE_DIR="${EVA_X64_PIP_CACHE_DIR:-/mnt/eva-llamacpp-lab/pip-cache}"
export PIP_CACHE_DIR PIP_DISABLE_PIP_VERSION_CHECK=1

case "${TARGET_OS_RELEASE}" in
  24.04) UBUNTU_CODENAME=noble ;;
  26.04) UBUNTU_CODENAME=resolute ;;
  *)
    printf 'ERROR: unsupported Ubuntu release: %s\n' "${TARGET_OS_RELEASE}" >&2
    exit 1
    ;;
esac

case "${OUTPUT_ROOT}" in
  /|/home|/home/*/Projects|/mnt|/run/media/*)
    printf 'ERROR: dependency output is too broad: %s\n' "${OUTPUT_ROOT}" >&2
    exit 1
    ;;
esac

for command in docker dpkg-scanpackages dpkg-deb gzip "${PYTHON_BIN}"; do
  command -v "${command}" >/dev/null 2>&1 || {
    printf 'ERROR: required build command is missing: %s\n' "${command}" >&2
    exit 1
  }
done

PACKAGE_LIST="${REPO_ROOT}/deployment/port_4070s/apt-packages-ubuntu-${TARGET_OS_RELEASE}.txt"
[[ -f "${PACKAGE_LIST}" ]] || {
  printf 'ERROR: package list is missing: %s\n' "${PACKAGE_LIST}" >&2
  exit 1
}

TEMP_ROOT="$(mktemp -d /mnt/eva-llamacpp-lab/eva-x64-seed.XXXXXX)"
cleanup() {
  rm -rf "${TEMP_ROOT}"
}
trap cleanup EXIT

if [[ "${RESUME_BUILD}" != "1" ]]; then
  rm -rf "${OUTPUT_ROOT}/apt" "${OUTPUT_ROOT}/wheelhouse"
fi
mkdir -p "${OUTPUT_ROOT}/apt" "${OUTPUT_ROOT}/wheelhouse"

platform_args=()
for minor in $(seq 39 -1 17); do
  platform_args+=(--platform "manylinux_2_${minor}_x86_64")
done
platform_args+=(--platform manylinux2014_x86_64)

for python_version in ${PYTHON_VERSIONS}; do
  case "${python_version}" in
    3.12|3.13|3.14) ;;
    *)
      printf 'ERROR: unsupported CPython target: %s\n' "${python_version}" >&2
      exit 1
      ;;
  esac
  abi="cp${python_version/.}"
  printf '\nDownloading CPython %s x86_64 wheel set...\n' "${python_version}"
  requirements=(
    --requirement "${REPO_ROOT}/requirements.txt"
    --requirement "${REPO_ROOT}/requirements-db.txt"
    --requirement "${REPO_ROOT}/requirements-cuda.txt"
  )
  # vLLM is a separate inference runtime, not an EVA app dependency.  Keep one
  # reviewed local-inference ABI in the union wheelhouse while still supporting
  # existing EVA application venvs on all declared Python minors.  This avoids
  # letting vLLM's CUDA dependency graph block an otherwise safe application
  # update or mutate a working external inference service.
  if [[ "${python_version}" == "${VLLM_PYTHON_VERSION}" ]]; then
    requirements+=('vllm==0.25.0')
  fi
  "${PYTHON_BIN}" -m pip download \
    --dest "${OUTPUT_ROOT}/wheelhouse" \
    --progress-bar off \
    --only-binary=:all: \
    "${platform_args[@]}" \
    --implementation cp \
    --python-version "${python_version}" \
    --abi "${abi}" \
    --constraint "${REPO_ROOT}/deployment/port_4070s/constraints-port-4070s.txt" \
    "${requirements[@]}"
done

APT_WORK="${TEMP_ROOT}/apt-work"
mkdir -p "${APT_WORK}"
docker run --rm \
  --platform linux/amd64 \
  --env "HOST_UID=$(id -u)" \
  --env "HOST_GID=$(id -g)" \
  --env "UBUNTU_CODENAME=${UBUNTU_CODENAME}" \
  --mount "type=bind,src=${APT_WORK},dst=/work" \
  --mount "type=bind,src=${OUTPUT_ROOT}/apt,dst=/out" \
  --mount "type=bind,src=${PACKAGE_LIST},dst=/package-names.txt,readonly" \
  "ubuntu:${TARGET_OS_RELEASE}" \
  bash -Eeuc '
    mkdir -p /work/etc/apt /work/lists/partial /work/cache/archives/partial /work/state
    printf "%s\n" \
      "deb [trusted=yes] http://archive.ubuntu.com/ubuntu ${UBUNTU_CODENAME} main universe multiverse restricted" \
      "deb [trusted=yes] http://archive.ubuntu.com/ubuntu ${UBUNTU_CODENAME}-updates main universe multiverse restricted" \
      "deb [trusted=yes] http://security.ubuntu.com/ubuntu ${UBUNTU_CODENAME}-security main universe multiverse restricted" \
      > /work/etc/apt/sources.list
    : > /work/state/status
    apt_options=(
      -o APT::Architecture=amd64
      -o APT::Architectures::=amd64
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

install -m 0644 "${PACKAGE_LIST}" "${OUTPUT_ROOT}/apt/package-names.txt"
(
  cd "${OUTPUT_ROOT}/apt"
  dpkg-scanpackages . /dev/null | gzip -9 > Packages.gz
)

while IFS= read -r -d '' package; do
  package_architecture="$(dpkg-deb -f "${package}" Architecture)"
  if [[ "${package_architecture}" != "amd64" && "${package_architecture}" != "all" ]]; then
    printf 'ERROR: non-amd64 package found: %s (%s)\n' \
      "${package}" "${package_architecture}" >&2
    exit 1
  fi
done < <(find "${OUTPUT_ROOT}/apt" -maxdepth 1 -type f -name '*.deb' -print0)

printf '\nx86_64 dependency seed ready: %s\n' "${OUTPUT_ROOT}"
printf '  Ubuntu fresh-install target: %s\n' "${TARGET_OS_RELEASE}"
printf '  update Python ABIs: %s\n' "${PYTHON_VERSIONS}"
printf '  local vLLM wheel ABI: CPython %s\n' "${VLLM_PYTHON_VERSION}"
printf '  apt artifacts: %s\n' "$(find "${OUTPUT_ROOT}/apt" -maxdepth 1 -name '*.deb' | wc -l)"
printf '  Python wheels: %s\n' "$(find "${OUTPUT_ROOT}/wheelhouse" -maxdepth 1 -name '*.whl' | wc -l)"
du -sh "${OUTPUT_ROOT}"
