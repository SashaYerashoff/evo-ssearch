#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STAGING_ROOT="${1:-/mnt/eva-llamacpp-lab/port-usb-staging}"
MODEL_VLM="${EVA_PORT_VLM_MODEL_DIR:-${REPO_ROOT}/.local/inference/models/qwen3-vl-4b-awq}"
MODEL_9B="${EVA_PORT_9B_MODEL_DIR:-/mnt/eva-llamacpp-lab/models/unsloth/Qwen3.5-9B-MTP-GGUF}"
CLIP_WEIGHT="${EVA_PORT_CLIP_WEIGHT:-/home/sasha/.cache/clip/ViT-B-32.pt}"
SIGLIP2_REVISION="${EVA_PORT_SIGLIP2_REVISION:-75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2}"
SIGLIP2_CACHE_REPO="${EVA_PORT_SIGLIP2_CACHE_REPO:-/mnt/eva-llamacpp-lab/models/huggingface/models--google--siglip2-base-patch16-224}"
LLAMA_SOURCE="${EVA_PORT_LLAMA_SOURCE:-/mnt/eva-llamacpp-lab/src/llama.cpp}"
REACT_UI_ROOT="${REPO_ROOT}/react-ui"
EXPECTED_BRANCH="${EVA_PORT_EXPECTED_BRANCH:-feature/maritime-port-specs}"
RELEASE_FLAVOR="${EVA_PORT_RELEASE_FLAVOR:-ventspils-maritime-client}"
SOURCE_BRANCH="$(git -C "${REPO_ROOT}" branch --show-current)"
SOURCE_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
SOURCE_DIRTY="$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=normal)"
WORKTREE_CLEAN=true

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

for required in \
    "${MODEL_VLM}/model.safetensors" \
    "${MODEL_9B}/Qwen3.5-9B-Q4_K_M.gguf" \
    "${CLIP_WEIGHT}" \
    "${SIGLIP2_CACHE_REPO}/snapshots/${SIGLIP2_REVISION}/model.safetensors" \
    "${SIGLIP2_CACHE_REPO}/snapshots/${SIGLIP2_REVISION}/config.json" \
    "${SIGLIP2_CACHE_REPO}/snapshots/${SIGLIP2_REVISION}/preprocessor_config.json" \
    "${SIGLIP2_CACHE_REPO}/snapshots/${SIGLIP2_REVISION}/tokenizer.json" \
    "${SIGLIP2_CACHE_REPO}/snapshots/${SIGLIP2_REVISION}/tokenizer_config.json" \
    "${LLAMA_SOURCE}/CMakeLists.txt"; do
    if [[ ! -e "${required}" ]]; then
        echo "ERROR: required payload is missing: ${required}" >&2
        exit 1
    fi
done

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
mkdir -p "${STAGING_ROOT}/models/qwen3.5-9b-mtp"
mkdir -p "${STAGING_ROOT}/models/clip"
mkdir -p "${STAGING_ROOT}/models/huggingface"
mkdir -p "${STAGING_ROOT}/llama.cpp"
mkdir -p "${STAGING_ROOT}/wheelhouse"
mkdir -p "${STAGING_ROOT}/apt"
mkdir -p "${STAGING_ROOT}/installer-deb"
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
    --exclude=react-ui/node_modules/ \
    --exclude=node_modules/ \
    --exclude=/dist/ \
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
rsync -a \
    "${MODEL_9B}/Qwen3.5-9B-Q4_K_M.gguf" \
    "${STAGING_ROOT}/models/qwen3.5-9b-mtp/Qwen3.5-9B-Q4_K_M.gguf"
rsync -a "${CLIP_WEIGHT}" "${STAGING_ROOT}/models/clip/ViT-B-32.pt"
rsync -a --delete \
    "${SIGLIP2_CACHE_REPO}/" \
    "${STAGING_ROOT}/models/huggingface/models--google--siglip2-base-patch16-224/"
rsync -a --delete \
    --exclude=.git \
    --exclude=build \
    --exclude=build-port-cpu \
    --exclude='*.o' \
    "${LLAMA_SOURCE}/" "${STAGING_ROOT}/llama.cpp/"

install -p -m 0755 "${SCRIPT_DIR}/install_port_appliance.py" "${STAGING_ROOT}/install_port_appliance.py"
install -p -m 0755 "${SCRIPT_DIR}/install_port_appliance.sh" "${STAGING_ROOT}/install.sh"
install -p -m 0755 "${SCRIPT_DIR}/eva_offline_deploy.py" "${STAGING_ROOT}/eva_offline_deploy.py"
install -p -m 0755 "${SCRIPT_DIR}/eva_offline_deploy.sh" "${STAGING_ROOT}/START_EVA_AI.sh"
install -p -m 0755 \
    "${SCRIPT_DIR}/offline_bundle_dependencies.py" \
    "${STAGING_ROOT}/offline_bundle_dependencies.py"
install -p -m 0644 \
    "${REPO_ROOT}/deployment/port_4070s/constraints-port-4070s.txt" \
    "${STAGING_ROOT}/constraints-port-4070s.txt"
if [[ "${RELEASE_FLAVOR}" == "universal-offline" ]]; then
    install -p -m 0644 \
        "${REPO_ROOT}/deployment/universal/START_HERE.md" \
        "${STAGING_ROOT}/START_HERE.md"
else
    install -p -m 0644 \
        "${REPO_ROOT}/deployment/port_4070s/START_HERE.txt" \
        "${STAGING_ROOT}/START_HERE.txt"
fi
install -p -m 0644 \
    "${REPO_ROOT}/deployment/port_4070s/apt-packages-ubuntu-24.04.txt" \
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
    --output-dir "${STAGING_ROOT}/installer-deb"

echo "Base payload prepared at ${STAGING_ROOT}"
echo "Universal entry point: ${STAGING_ROOT}/START_EVA_AI.sh"
echo "Base payload only: populate and validate wheelhouse/ and apt/ before finalization."
echo "For a complete fresh/update bundle, use scripts/build_universal_usb_bundle.sh."
