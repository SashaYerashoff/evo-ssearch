#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STAGING_ROOT="${1:-/mnt/eva-llamacpp-lab/port-usb-staging}"
MODEL_VLM="${EVA_PORT_VLM_MODEL_DIR:-${REPO_ROOT}/.local/inference/models/qwen3-vl-4b-awq}"
MODEL_VLM_SPARK="${EVA_SPARK_VLM_MODEL_DIR:-/mnt/eva-llamacpp-lab/models/huggingface/Qwen/Qwen3-VL-4B-Instruct}"
MODEL_VLM_SPARK_REVISION="ebb281ec70b05090aa6165b016eac8ec08e71b17"
MODEL_9B="${EVA_PORT_9B_MODEL_DIR:-/mnt/eva-llamacpp-lab/models/unsloth/Qwen3.5-9B-MTP-GGUF}"
CLIP_WEIGHT="${EVA_PORT_CLIP_WEIGHT:-/home/sasha/.cache/clip/ViT-B-32.pt}"
SIGLIP2_REVISION="${EVA_PORT_SIGLIP2_REVISION:-75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2}"
SIGLIP2_CACHE_REPO="${EVA_PORT_SIGLIP2_CACHE_REPO:-/mnt/eva-llamacpp-lab/models/huggingface/models--google--siglip2-base-patch16-224}"
LLAMA_SOURCE="${EVA_PORT_LLAMA_SOURCE:-/mnt/eva-llamacpp-lab/src/llama.cpp}"
REACT_UI_ROOT="${REPO_ROOT}/react-ui"
EXPECTED_BRANCH="${EVA_PORT_EXPECTED_BRANCH:-feature/maritime-port-specs}"
RELEASE_FLAVOR="${EVA_PORT_RELEASE_FLAVOR:-ventspils-maritime-client}"
TARGET_ARCHITECTURE="${EVA_PORT_ARCHITECTURE:-amd64}"
TARGET_OS_RELEASE="${EVA_PORT_OS_RELEASE:-24.04}"
SPARK_RUNTIME_ARCHIVE="${EVA_SPARK_RUNTIME_ARCHIVE:-}"
SPARK_RUNTIME_ARCHIVE_NAME="eva-spark-runtime-0.8.7-arm64.tar.zst"
X64_VLLM_PYTHON_ARCHIVE="${EVA_X64_VLLM_PYTHON_ARCHIVE:-/mnt/eva-llamacpp-lab/x64-vllm-runtime/cpython-3.12.13-linux-x86_64-gnu.tar.gz}"
X64_VLLM_PYTHON_ARCHIVE_NAME="cpython-3.12.13-linux-x86_64-gnu.tar.gz"
STAGING_HARDLINKS="${EVA_PORT_STAGING_HARDLINKS:-0}"
SOURCE_BRANCH="$(git -C "${REPO_ROOT}" branch --show-current)"
SOURCE_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
SOURCE_DIRTY="$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=normal)"
WORKTREE_CLEAN=true

case "${STAGING_HARDLINKS}" in
    0|1) ;;
    *)
        echo "ERROR: EVA_PORT_STAGING_HARDLINKS must be 0 or 1." >&2
        exit 1
        ;;
esac

stage_tree_payload() {
    local source="$1"
    local destination="$2"
    if [[ "${STAGING_HARDLINKS}" == "1" ]] \
        && [[ "$(stat -c '%d' "${source}")" == "$(stat -c '%d' "$(dirname "${destination}")")" ]]; then
        rm -rf "${destination}"
        mkdir -p "${destination}"
        cp -al "${source}/." "${destination}/"
        printf 'Hard-linked immutable staging payload: %s\n' "${destination}"
        return
    fi
    rsync -a --delete "${source}/" "${destination}/"
}

stage_file_payload() {
    local source="$1"
    local destination="$2"
    if [[ "${STAGING_HARDLINKS}" == "1" ]] \
        && [[ "$(stat -c '%d' "${source}")" == "$(stat -c '%d' "$(dirname "${destination}")")" ]]; then
        ln -f "${source}" "${destination}"
        printf 'Hard-linked immutable staging payload: %s\n' "${destination}"
        return
    fi
    rsync -a --info=progress2 "${source}" "${destination}"
}

case "${TARGET_ARCHITECTURE}" in
    amd64)
        PROFILE_DIR="${REPO_ROOT}/deployment/port_4070s"
        CONSTRAINTS_SOURCE="${PROFILE_DIR}/constraints-port-4070s.txt"
        CONSTRAINTS_NAME="constraints-port-4070s.txt"
        ;;
    arm64)
        PROFILE_DIR="${REPO_ROOT}/deployment/spark_gb10"
        CONSTRAINTS_SOURCE="${PROFILE_DIR}/constraints-spark-gb10.txt"
        CONSTRAINTS_NAME="constraints-spark-gb10.txt"
        ;;
    *)
        echo "ERROR: unsupported EVA_PORT_ARCHITECTURE=${TARGET_ARCHITECTURE}" >&2
        exit 1
        ;;
esac
case "${TARGET_OS_RELEASE}" in
    24.04|26.04) ;;
    *)
        echo "ERROR: unsupported EVA_PORT_OS_RELEASE=${TARGET_OS_RELEASE}" >&2
        exit 1
        ;;
esac
if [[ "${TARGET_ARCHITECTURE}" == "arm64" && "${TARGET_OS_RELEASE}" != "24.04" ]]; then
    echo "ERROR: Spark ARM64 remains pinned to Ubuntu 24.04." >&2
    exit 1
fi

if [[ "${SOURCE_BRANCH}" != "${EXPECTED_BRANCH}" && "${EVA_PORT_ALLOW_OTHER_BRANCH:-0}" != "1" ]]; then
    echo "ERROR: port client bundle must be built from ${EXPECTED_BRANCH}; current branch is ${SOURCE_BRANCH}." >&2
    echo "Set EVA_PORT_ALLOW_OTHER_BRANCH=1 only for an explicitly reviewed recovery build." >&2
    exit 1
fi
if [[ -n "${SOURCE_DIRTY}" && "${EVA_PORT_ALLOW_DIRTY:-0}" != "1" ]]; then
    echo "ERROR: port client bundle requires a clean committed working tree." >&2
    echo "Commit the release candidate, or set EVA_PORT_ALLOW_DIRTY=1 only for a labelled diagnostic build." >&2
    exit 1
fi
if [[ -n "${SOURCE_DIRTY}" ]]; then
    WORKTREE_CLEAN=false
    echo "WARNING: building a diagnostic payload from a dirty tree; finalization will refuse it." >&2
fi

REQUIRED_PAYLOAD=(
    "${SIGLIP2_CACHE_REPO}/snapshots/${SIGLIP2_REVISION}/model.safetensors"
    "${SIGLIP2_CACHE_REPO}/snapshots/${SIGLIP2_REVISION}/config.json"
    "${SIGLIP2_CACHE_REPO}/snapshots/${SIGLIP2_REVISION}/preprocessor_config.json"
    "${SIGLIP2_CACHE_REPO}/snapshots/${SIGLIP2_REVISION}/tokenizer.json"
    "${SIGLIP2_CACHE_REPO}/snapshots/${SIGLIP2_REVISION}/tokenizer_config.json"
)
if [[ "${TARGET_ARCHITECTURE}" == "amd64" ]]; then
    REQUIRED_PAYLOAD+=(
        "${CLIP_WEIGHT}"
        "${MODEL_VLM}/model.safetensors"
        "${MODEL_9B}/Qwen3.5-9B-Q4_K_M.gguf"
        "${LLAMA_SOURCE}/CMakeLists.txt"
        "${X64_VLLM_PYTHON_ARCHIVE}"
    )
else
    REQUIRED_PAYLOAD+=(
        "${PROFILE_DIR}/runtime-container.json"
        "${MODEL_VLM_SPARK}/config.json"
        "${MODEL_VLM_SPARK}/tokenizer.json"
        "${MODEL_VLM_SPARK}/.cache/huggingface/download/config.json.metadata"
    )
fi
for required in "${REQUIRED_PAYLOAD[@]}"; do
    if [[ ! -e "${required}" ]]; then
        echo "ERROR: required payload is missing: ${required}" >&2
        exit 1
    fi
done
if [[ "${TARGET_ARCHITECTURE}" == "arm64" ]] \
    && [[ "$(head -n 1 "${MODEL_VLM_SPARK}/.cache/huggingface/download/config.json.metadata")" \
        != "${MODEL_VLM_SPARK_REVISION}" ]]; then
    echo "ERROR: Spark Qwen model is not the pinned revision ${MODEL_VLM_SPARK_REVISION}." >&2
    exit 1
fi
if [[ "${TARGET_ARCHITECTURE}" == "arm64" ]] \
    && ! find "${MODEL_VLM_SPARK}" -maxdepth 1 -type f -name '*.safetensors' -print -quit \
        | grep -q .; then
    echo "ERROR: Spark Qwen model weights are missing: ${MODEL_VLM_SPARK}" >&2
    exit 1
fi

if [[ ! -x "${REACT_UI_ROOT}/node_modules/.bin/vite" ]]; then
    echo "ERROR: React build dependencies are missing. Run: npm --prefix ${REACT_UI_ROOT} ci" >&2
    exit 1
fi
npm --prefix "${REACT_UI_ROOT}" run build
if [[ ! -f "${REACT_UI_ROOT}/dist/index.html" ]]; then
    echo "ERROR: React production build did not produce dist/index.html" >&2
    exit 1
fi

mkdir -p "${STAGING_ROOT}"
mkdir -p "${STAGING_ROOT}/repo"
mkdir -p "${STAGING_ROOT}/models/qwen3-vl-4b-awq"
mkdir -p "${STAGING_ROOT}/models/qwen3-vl-4b"
mkdir -p "${STAGING_ROOT}/models/qwen3.5-9b-mtp"
mkdir -p "${STAGING_ROOT}/models/clip"
mkdir -p "${STAGING_ROOT}/models/huggingface"
mkdir -p "${STAGING_ROOT}/llama.cpp"
mkdir -p "${STAGING_ROOT}/wheelhouse"
mkdir -p "${STAGING_ROOT}/apt"
mkdir -p "${STAGING_ROOT}/installer-deb"
mkdir -p "${STAGING_ROOT}/repository-backup"
mkdir -p "${STAGING_ROOT}/container"
rm -f \
    "${STAGING_ROOT}/constraints-port-4070s.txt" \
    "${STAGING_ROOT}/constraints-spark-gb10.txt"
find "${STAGING_ROOT}/installer-deb" -maxdepth 1 -type f \
    -name 'eva-ai-appliance-installer_*.deb' -delete

rsync -a --delete --delete-excluded \
    --exclude=.git \
    --exclude='.env*' \
    --exclude='.venv*' \
    --exclude=.local \
    --exclude=.claude \
    --exclude=.eva-bundle-commit \
    --exclude=.eva-runtime \
    --exclude=.clip_index \
    --exclude=.pytest_cache \
    --exclude=__pycache__ \
    --exclude=react-ui/node_modules/ \
    --exclude=node_modules/ \
    --exclude=/dist/ \
    --exclude=detections_archive/ \
    --exclude=inference_spool/ \
    --exclude=/video/ \
    --exclude=probes_store.json \
    --exclude=probe_channel_groups.json \
    --exclude=luxriot_rollups_cache.json \
    --exclude=luxriot_summary_state.json \
    --exclude='*.sqlite3' \
    --exclude='*.sqlite3-shm' \
    --exclude='*.sqlite3-wal' \
    --exclude='*.pyc' \
    --exclude='*.dump' \
    --exclude='*.bak' \
    --exclude='docs/*.pdf' \
    "${REPO_ROOT}/" "${STAGING_ROOT}/repo/"

stage_tree_payload \
    "${SIGLIP2_CACHE_REPO}" \
    "${STAGING_ROOT}/models/huggingface/models--google--siglip2-base-patch16-224"
SIGLIP2_CACHE_TARGET="${STAGING_ROOT}/models/huggingface"
(
    cd "${SIGLIP2_CACHE_TARGET}"
    find models--google--siglip2-base-patch16-224 -type f -print0 \
        | sort -z \
        | xargs -0 -r sha256sum > SHA256SUMS
    if [[ ! -s SHA256SUMS ]]; then
        echo "ERROR: staged SigLIP2 cache produced an empty SHA256SUMS." >&2
        exit 1
    fi
    sha256sum -c SHA256SUMS >/dev/null
)
if [[ "${TARGET_ARCHITECTURE}" == "amd64" ]]; then
    rsync -a "${CLIP_WEIGHT}" "${STAGING_ROOT}/models/clip/ViT-B-32.pt"
    rsync -a --delete "${MODEL_VLM}/" "${STAGING_ROOT}/models/qwen3-vl-4b-awq/"
    rsync -a \
        "${MODEL_9B}/Qwen3.5-9B-Q4_K_M.gguf" \
        "${STAGING_ROOT}/models/qwen3.5-9b-mtp/Qwen3.5-9B-Q4_K_M.gguf"
    rsync -a --delete \
        --exclude=.git \
        --exclude=build \
        --exclude=build-port-cpu \
        --exclude='*.o' \
        "${LLAMA_SOURCE}/" "${STAGING_ROOT}/llama.cpp/"
    rm -rf "${STAGING_ROOT}/python"
    mkdir -p "${STAGING_ROOT}/python"
    stage_file_payload \
        "${X64_VLLM_PYTHON_ARCHIVE}" \
        "${STAGING_ROOT}/python/${X64_VLLM_PYTHON_ARCHIVE_NAME}"
    rm -rf "${STAGING_ROOT}/container"
    rm -rf "${STAGING_ROOT}/models/qwen3-vl-4b"
else
    rm -rf \
        "${STAGING_ROOT}/models/qwen3-vl-4b-awq" \
        "${STAGING_ROOT}/models/qwen3.5-9b-mtp" \
        "${STAGING_ROOT}/llama.cpp" \
        "${STAGING_ROOT}/models/clip" \
        "${STAGING_ROOT}/python"
    mkdir -p "${STAGING_ROOT}/models/qwen3-vl-4b"
    if [[ "${STAGING_HARDLINKS}" == "1" ]]; then
        stage_tree_payload \
            "${MODEL_VLM_SPARK}" \
            "${STAGING_ROOT}/models/qwen3-vl-4b"
        rm -rf "${STAGING_ROOT}/models/qwen3-vl-4b/.cache"
    else
        rsync -a --delete --exclude=.cache \
            "${MODEL_VLM_SPARK}/" \
            "${STAGING_ROOT}/models/qwen3-vl-4b/"
    fi
    install -p -m 0644 \
        "${PROFILE_DIR}/runtime-container.json" \
        "${STAGING_ROOT}/runtime-container.json"
    rm -rf "${STAGING_ROOT}/container"
    mkdir -p "${STAGING_ROOT}/container"
    if [[ -z "${SPARK_RUNTIME_ARCHIVE}" || ! -f "${SPARK_RUNTIME_ARCHIVE}" ]]; then
        echo "ERROR: a factory-complete ARM bundle requires the pinned NGC image archive." >&2
        echo "Set EVA_SPARK_RUNTIME_ARCHIVE=/path/to/eva-spark-runtime.tar.zst." >&2
        exit 1
    fi
    stage_file_payload \
        "${SPARK_RUNTIME_ARCHIVE}" \
        "${STAGING_ROOT}/container/${SPARK_RUNTIME_ARCHIVE_NAME}"
fi

install -p -m 0755 "${SCRIPT_DIR}/install_port_appliance.py" "${STAGING_ROOT}/install_port_appliance.py"
install -p -m 0755 "${SCRIPT_DIR}/install_port_appliance.sh" "${STAGING_ROOT}/install.sh"
install -p -m 0755 "${SCRIPT_DIR}/eva_offline_deploy.py" "${STAGING_ROOT}/eva_offline_deploy.py"
install -p -m 0755 "${SCRIPT_DIR}/eva_offline_deploy.sh" "${STAGING_ROOT}/START_EVA_AI.sh"
install -p -m 0755 \
    "${SCRIPT_DIR}/offline_bundle_dependencies.py" \
    "${STAGING_ROOT}/offline_bundle_dependencies.py"
install -p -m 0644 \
    "${CONSTRAINTS_SOURCE}" \
    "${STAGING_ROOT}/${CONSTRAINTS_NAME}"
if [[ "${RELEASE_FLAVOR}" == "universal-offline" ]]; then
    if [[ "${TARGET_ARCHITECTURE}" == "arm64" ]]; then
        install -p -m 0644 \
            "${PROFILE_DIR}/START_HERE.md" \
            "${STAGING_ROOT}/START_HERE.md"
    else
        install -p -m 0644 \
            "${REPO_ROOT}/deployment/universal/START_HERE.md" \
            "${STAGING_ROOT}/START_HERE.md"
    fi
else
    install -p -m 0644 \
        "${REPO_ROOT}/deployment/port_4070s/START_HERE.txt" \
        "${STAGING_ROOT}/START_HERE.txt"
fi
install -p -m 0644 \
    "${PROFILE_DIR}/apt-packages-ubuntu-${TARGET_OS_RELEASE}.txt" \
    "${STAGING_ROOT}/apt/package-names.txt"
install -p -m 0644 \
    "${REPO_ROOT}/deployment/port_4070s/REPOSITORY_BACKUP.txt" \
    "${STAGING_ROOT}/repository-backup/README.txt"
python3 - "${STAGING_ROOT}/SOURCE_REVISION.json" "${SOURCE_BRANCH}" "${SOURCE_COMMIT}" "${RELEASE_FLAVOR}" "${WORKTREE_CLEAN}" <<'PY'
import json
import sys
from pathlib import Path

target, branch, commit, flavor, clean = sys.argv[1:]
Path(target).write_text(
    json.dumps(
        {
            "format": 1,
            "release_flavor": flavor,
            "branch": branch,
            "commit": commit,
            "working_tree_clean": clean == "true",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
git -C "${REPO_ROOT}" bundle create \
    "${STAGING_ROOT}/repository-backup/evo-ssearch-all-refs.bundle" \
    --all
python3 "${SCRIPT_DIR}/build_appliance_installer_deb.py" \
    --output-dir "${STAGING_ROOT}/installer-deb" \
    --architecture "${TARGET_ARCHITECTURE}"

echo "Base payload prepared at ${STAGING_ROOT}"
echo "Universal entry point: ${STAGING_ROOT}/START_EVA_AI.sh"
echo "Base payload only: populate and validate wheelhouse/ and apt/ before finalization."
echo "For a complete fresh/update bundle, use scripts/build_universal_usb_bundle.sh."
