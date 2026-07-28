#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STAGING_ROOT="${1:-/mnt/eva-llamacpp-lab/port-usb-staging}"
MODEL_VLM="${EVA_PORT_VLM_MODEL_DIR:-${REPO_ROOT}/.local/inference/models/qwen3-vl-4b-awq}"
MODEL_9B="${EVA_PORT_9B_MODEL_DIR:-/mnt/eva-llamacpp-lab/models/unsloth/Qwen3.5-9B-MTP-GGUF}"
CLIP_WEIGHT="${EVA_PORT_CLIP_WEIGHT:-/home/sasha/.cache/clip/ViT-B-32.pt}"
LLAMA_SOURCE="${EVA_PORT_LLAMA_SOURCE:-/mnt/eva-llamacpp-lab/src/llama.cpp}"

for required in \
    "${MODEL_VLM}/model.safetensors" \
    "${MODEL_9B}/Qwen3.5-9B-Q4_K_M.gguf" \
    "${CLIP_WEIGHT}" \
    "${LLAMA_SOURCE}/CMakeLists.txt"; do
    if [[ ! -e "${required}" ]]; then
        echo "ERROR: required payload is missing: ${required}" >&2
        exit 1
    fi
done

mkdir -p "${STAGING_ROOT}"
mkdir -p "${STAGING_ROOT}/repo"
mkdir -p "${STAGING_ROOT}/models/qwen3-vl-4b-awq"
mkdir -p "${STAGING_ROOT}/models/qwen3.5-9b-mtp"
mkdir -p "${STAGING_ROOT}/models/clip"
mkdir -p "${STAGING_ROOT}/llama.cpp"
mkdir -p "${STAGING_ROOT}/wheelhouse"
mkdir -p "${STAGING_ROOT}/apt"
mkdir -p "${STAGING_ROOT}/repository-backup"

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
    --exclude=react-ui/ \
    --exclude=node_modules/ \
    --exclude=dist/ \
    --exclude=detections_archive/ \
    --exclude=inference_spool/ \
    --exclude=video/ \
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

rsync -a --delete "${MODEL_VLM}/" "${STAGING_ROOT}/models/qwen3-vl-4b-awq/"
install -p -m 0644 \
    "${MODEL_9B}/Qwen3.5-9B-Q4_K_M.gguf" \
    "${STAGING_ROOT}/models/qwen3.5-9b-mtp/Qwen3.5-9B-Q4_K_M.gguf"
install -p -m 0644 "${CLIP_WEIGHT}" "${STAGING_ROOT}/models/clip/ViT-B-32.pt"
rsync -a --delete \
    --exclude=.git \
    --exclude=build \
    --exclude=build-port-cpu \
    --exclude='*.o' \
    "${LLAMA_SOURCE}/" "${STAGING_ROOT}/llama.cpp/"

install -p -m 0755 "${SCRIPT_DIR}/install_port_appliance.py" "${STAGING_ROOT}/install_port_appliance.py"
install -p -m 0755 "${SCRIPT_DIR}/install_port_appliance.sh" "${STAGING_ROOT}/install.sh"
install -p -m 0644 \
    "${REPO_ROOT}/deployment/port_4070s/constraints-port-4070s.txt" \
    "${STAGING_ROOT}/constraints-port-4070s.txt"
install -p -m 0644 \
    "${REPO_ROOT}/deployment/port_4070s/START_HERE.txt" \
    "${STAGING_ROOT}/START_HERE.txt"
install -p -m 0644 \
    "${REPO_ROOT}/deployment/port_4070s/apt-packages-ubuntu-24.04.txt" \
    "${STAGING_ROOT}/apt/package-names.txt"
install -p -m 0644 \
    "${REPO_ROOT}/deployment/port_4070s/REPOSITORY_BACKUP.txt" \
    "${STAGING_ROOT}/repository-backup/README.txt"
git -C "${REPO_ROOT}" bundle create \
    "${STAGING_ROOT}/repository-backup/evo-ssearch-all-refs.bundle" \
    --all

echo "Base payload prepared at ${STAGING_ROOT}"
echo "Populate wheelhouse/ and apt/, then run scripts/finalize_port_usb_bundle.py."
