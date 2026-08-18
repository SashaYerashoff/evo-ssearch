#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_IMAGE="nvcr.io/nvidia/vllm:26.07-py3"
BASE_MANIFEST_DIGEST="sha256:1de8e6bfdb4c81c1f31a806cc9b13b5c6352714a7cec87f4d24964bcc91159b2"
BASE_IMAGE_ID="sha256:4c704f1343c7cb3aa7ea5cc57cab5fa1ed1a2160daf4f57d1e0c06fc1e2c7dbb"
RUNTIME_IMAGE="eva-ai/spark-runtime:0.8.7-arm64"
OUTPUT_ARCHIVE="${1:-/mnt/eva-llamacpp-lab/spark-runtime/nvidia-vllm-26.07-py3-eva-0.8.7-arm64.tar.zst}"

for command in docker dpkg-deb python3 sha256sum tar zstd; do
    command -v "${command}" >/dev/null 2>&1 || {
        printf 'ERROR: required command is missing: %s\n' "${command}" >&2
        exit 1
    }
done

case "${OUTPUT_ARCHIVE}" in
    /|/home|/mnt|/run/media/*)
        printf 'ERROR: runtime archive target is too broad: %s\n' "${OUTPUT_ARCHIVE}" >&2
        exit 1
        ;;
esac

if [[ -n "${EVA_SPARK_RUNTIME_BUILD_ROOT:-}" ]]; then
    TEMP_ROOT="$(readlink -f "${EVA_SPARK_RUNTIME_BUILD_ROOT}")"
    case "${TEMP_ROOT}" in
        /mnt/eva-llamacpp-lab/eva-spark-runtime.*) ;;
        *)
            printf 'ERROR: unsafe reusable Spark build root: %s\n' "${TEMP_ROOT}" >&2
            exit 1
            ;;
    esac
    if [[ ! -d "${TEMP_ROOT}" ]]; then
        printf 'ERROR: reusable Spark build root does not exist: %s\n' "${TEMP_ROOT}" >&2
        exit 1
    fi
else
    TEMP_ROOT="$(mktemp -d /mnt/eva-llamacpp-lab/eva-spark-runtime.XXXXXX)"
fi
DIND_CONTAINER="eva-spark-runtime-build-dind-$$"
BASE_INSPECT_CONTAINER="eva-spark-runtime-base-inspect"
BUILD_SUCCEEDED=false
cleanup() {
    status=$?
    docker rm -f "${DIND_CONTAINER}" >/dev/null 2>&1 || true
    if [[ "${BUILD_SUCCEEDED}" == "true" ]]; then
        docker run --rm \
            --mount "type=bind,src=${TEMP_ROOT},dst=/cleanup" \
            --entrypoint sh \
            docker:29-dind \
            -c 'find /cleanup -mindepth 1 -delete' >/dev/null
        rmdir "${TEMP_ROOT}"
    else
        printf 'Spark build root retained for retry: %s\n' "${TEMP_ROOT}" >&2
    fi
    return "${status}"
}
trap cleanup EXIT

mkdir -p "${TEMP_ROOT}/context/rootfs" "${TEMP_ROOT}/debs" "${TEMP_ROOT}/apt"
mkdir -p "${TEMP_ROOT}/docker-data"
docker run --detach --privileged \
    --name "${DIND_CONTAINER}" \
    --env DOCKER_TLS_CERTDIR= \
    --mount "type=bind,src=${TEMP_ROOT}/docker-data,dst=/var/lib/docker" \
    --mount "type=bind,src=${TEMP_ROOT}/context,dst=/work/context,readonly" \
    docker:29-dind \
    --host=unix:///var/run/docker.sock \
    --feature containerd-snapshotter=false >/dev/null
for _attempt in $(seq 1 60); do
    if docker exec "${DIND_CONTAINER}" docker info >/dev/null 2>&1; then
        break
    fi
    sleep 1
done
docker exec "${DIND_CONTAINER}" docker info >/dev/null
docker exec "${DIND_CONTAINER}" \
    docker pull --platform linux/arm64 "${BASE_IMAGE}"
base_reference="${BASE_IMAGE}@${BASE_MANIFEST_DIGEST}"
docker exec "${DIND_CONTAINER}" \
    docker pull --platform linux/arm64 "${base_reference}" >/dev/null
actual_base_config_id="$(
    docker exec "${DIND_CONTAINER}" \
        docker buildx imagetools inspect --raw "${base_reference}" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["config"]["digest"])'
)"
if [[ "${actual_base_config_id}" != "${BASE_IMAGE_ID}" ]]; then
    printf 'ERROR: ARM64 NGC manifest %s has config %s, expected %s\n' \
        "${BASE_MANIFEST_DIGEST}" \
        "${actual_base_config_id:-nothing}" \
        "${BASE_IMAGE_ID}" >&2
    exit 1
fi

if [[ ! -s "${TEMP_ROOT}/apt/status" ]]; then
    docker exec "${DIND_CONTAINER}" \
        docker rm -f "${BASE_INSPECT_CONTAINER}" >/dev/null 2>&1 || true
    docker exec "${DIND_CONTAINER}" \
        docker create \
            --platform linux/arm64 \
            --name "${BASE_INSPECT_CONTAINER}" \
            --entrypoint /bin/true \
            "${base_reference}" >/dev/null
    docker exec "${DIND_CONTAINER}" \
        docker cp "${BASE_INSPECT_CONTAINER}:/var/lib/dpkg/status" - \
        | tar -xOf - > "${TEMP_ROOT}/apt/status"
    docker exec "${DIND_CONTAINER}" \
        docker rm "${BASE_INSPECT_CONTAINER}" >/dev/null
fi

if [[ ! -x "${TEMP_ROOT}/context/rootfs/usr/bin/ffmpeg" ]]; then
    docker run --rm \
        --platform linux/amd64 \
        --env "HOST_UID=$(id -u)" \
        --env "HOST_GID=$(id -g)" \
        --mount "type=bind,src=${TEMP_ROOT}/apt,dst=/work" \
        --mount "type=bind,src=${TEMP_ROOT}/debs,dst=/out" \
        ubuntu:24.04 \
        bash -Eeuc '
            mkdir -p /work/etc/apt /work/lists/partial /work/cache/archives/partial
            printf "%s\n" \
              "deb [trusted=yes] http://ports.ubuntu.com/ubuntu-ports noble main universe multiverse restricted" \
              "deb [trusted=yes] http://ports.ubuntu.com/ubuntu-ports noble-updates main universe multiverse restricted" \
              "deb [trusted=yes] http://ports.ubuntu.com/ubuntu-ports noble-security main universe multiverse restricted" \
              > /work/etc/apt/sources.list
            apt_options=(
              -o APT::Architecture=arm64
              -o APT::Architectures::=arm64
              -o Dir::Etc::sourcelist=/work/etc/apt/sources.list
              -o Dir::Etc::sourceparts=-
              -o Dir::State::status=/work/status
              -o Dir::State::lists=/work/lists
              -o Dir::Cache=/work/cache
              -o Acquire::Languages=none
              -o Acquire::Retries=3
            )
            apt-get "${apt_options[@]}" update
            apt-get "${apt_options[@]}" \
              --download-only --no-install-recommends --no-upgrade -y install ffmpeg
            cp /work/cache/archives/*.deb /out/
            chown -R "${HOST_UID}:${HOST_GID}" /work /out
        '

    while IFS= read -r -d '' package; do
        package_architecture="$(dpkg-deb -f "${package}" Architecture)"
        if [[ "${package_architecture}" != "arm64" && "${package_architecture}" != "all" ]]; then
            printf 'ERROR: non-ARM package entered runtime layer: %s (%s)\n' \
                "${package}" "${package_architecture}" >&2
            exit 1
        fi
        dpkg-deb --extract "${package}" "${TEMP_ROOT}/context/rootfs"
    done < <(find "${TEMP_ROOT}/debs" -maxdepth 1 -type f -name '*.deb' -print0)
fi

if [[ ! -x "${TEMP_ROOT}/context/rootfs/usr/bin/ffmpeg" ]]; then
    printf 'ERROR: resolved runtime layer does not contain /usr/bin/ffmpeg\n' >&2
    exit 1
fi

find "${TEMP_ROOT}/context/rootfs" -exec touch -h -d '@0' {} +
install -m 0644 \
    "${REPO_ROOT}/deployment/spark_gb10/runtime-image/Dockerfile" \
    "${TEMP_ROOT}/context/Dockerfile"
docker exec "${DIND_CONTAINER}" \
    docker build \
        --platform linux/arm64 \
        --build-arg "BASE_IMAGE=${base_reference}" \
        --tag "${RUNTIME_IMAGE}" \
        /work/context

runtime_id="$(
    docker exec "${DIND_CONTAINER}" \
        docker image inspect --format '{{.Id}}' "${RUNTIME_IMAGE}"
)"
runtime_arch="$(
    docker exec "${DIND_CONTAINER}" \
        docker image inspect --format '{{.Architecture}}' "${RUNTIME_IMAGE}"
)"
if [[ "${runtime_arch}" != "arm64" ]]; then
    printf 'ERROR: derived runtime architecture is %s, expected arm64\n' "${runtime_arch}" >&2
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT_ARCHIVE}")"
partial_archive="${OUTPUT_ARCHIVE}.partial"
rm -f "${partial_archive}"
docker exec "${DIND_CONTAINER}" \
    docker image save --platform linux/arm64 "${RUNTIME_IMAGE}" \
    | zstd -T0 -7 -f -o "${partial_archive}"
mv -f "${partial_archive}" "${OUTPUT_ARCHIVE}"

printf 'Spark runtime ready\n'
printf '  image:   %s\n' "${RUNTIME_IMAGE}"
printf '  image_id:%s\n' "${runtime_id}"
printf '  archive: %s\n' "${OUTPUT_ARCHIVE}"
sha256sum "${OUTPUT_ARCHIVE}"
du -sh "${OUTPUT_ARCHIVE}"
BUILD_SUCCEEDED=true
