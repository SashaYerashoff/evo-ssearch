#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/dist}"
BUNDLE_NAME="${BUNDLE_NAME:-eva-ai-patch-$(date +%Y%m%d-%H%M%S)}"
INCLUDE_WHEELHOUSE=false
WHEELHOUSE_DIR=""
FFMPEG_ARCHIVE=""
OPENCV_WHEEL=""
MEDIA_RUNTIME_DIR=""
SIGLIP2_CACHE_REPO=""
PYTHON_BIN="${PYTHON:-python3}"

log() {
  printf '[INFO] %s\n' "$*"
}

ok() {
  printf 'OK: %s\n' "$*"
}

fail() {
  printf 'FAIL: %s\n' "$*" >&2
}

usage() {
  cat <<'USAGE'
Usage: scripts/build_patch_bundle.sh [options]

Options:
  --output-dir DIR      Directory for the generated tarball.
  --name NAME           Bundle directory/archive name.
  --with-wheelhouse     Download Python wheels into bundle/wheelhouse.
  --wheelhouse-dir DIR  Copy an existing wheelhouse directory into the bundle.
  --ffmpeg-archive FILE Include a BtbN linux64 FFmpeg .tar.xz payload.
  --opencv-wheel FILE   Include an x86_64 opencv-python-headless wheel.
  --media-runtime-dir DIR
                        Reuse an already checksummed runtime/ payload from a
                        previously validated offline bundle.
  --siglip2-cache-repo DIR
                        Include the complete Hugging Face cache repository for
                        google/siglip2-base-patch16-224 (blobs/refs/snapshots).
  --python-bin PATH     Python used for pip download. Default: $PYTHON or python3.
  -h, --help            Show this help.

Environment:
  OUTPUT_DIR            Same as --output-dir.
  BUNDLE_NAME           Same as --name.
  PYTHON                Same as --python-bin.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --name)
      BUNDLE_NAME="$2"
      shift 2
      ;;
    --with-wheelhouse)
      INCLUDE_WHEELHOUSE=true
      shift
      ;;
    --wheelhouse-dir)
      WHEELHOUSE_DIR="$2"
      INCLUDE_WHEELHOUSE=true
      shift 2
      ;;
    --ffmpeg-archive)
      FFMPEG_ARCHIVE="$2"
      shift 2
      ;;
    --opencv-wheel)
      OPENCV_WHEEL="$2"
      shift 2
      ;;
    --media-runtime-dir)
      MEDIA_RUNTIME_DIR="$2"
      shift 2
      ;;
    --siglip2-cache-repo)
      SIGLIP2_CACHE_REPO="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    fail "Required command not found: $1"
    exit 1
  fi
}

need_cmd tar
need_cmd date
need_cmd mktemp

if [[ -n "${MEDIA_RUNTIME_DIR}" && ( -n "${FFMPEG_ARCHIVE}" || -n "${OPENCV_WHEEL}" ) ]]; then
  fail "Use either --media-runtime-dir or --ffmpeg-archive/--opencv-wheel, not both"
  exit 2
fi

if [[ -n "${FFMPEG_ARCHIVE}" || -n "${OPENCV_WHEEL}" ]]; then
  [[ -f "${FFMPEG_ARCHIVE}" ]] || {
    fail "FFmpeg archive not found: ${FFMPEG_ARCHIVE:-<not supplied>}"
    exit 1
  }
  [[ -f "${OPENCV_WHEEL}" ]] || {
    fail "OpenCV wheel not found: ${OPENCV_WHEEL:-<not supplied>}"
    exit 1
  }
  need_cmd sha256sum
fi

if [[ -n "${MEDIA_RUNTIME_DIR}" ]]; then
  for relative in SHA256SUMS manifest.txt ffmpeg/bin/ffmpeg ffmpeg/bin/ffprobe ffmpeg/LICENSE.txt; do
    [[ -f "${MEDIA_RUNTIME_DIR}/${relative}" ]] || {
      fail "Media runtime is incomplete: ${MEDIA_RUNTIME_DIR}/${relative}"
      exit 1
    }
  done
  [[ -n "$(find "${MEDIA_RUNTIME_DIR}/opencv" -maxdepth 1 -type f -name 'opencv_python_headless-*.whl' -print -quit)" ]] || {
    fail "Media runtime has no OpenCV wheel: ${MEDIA_RUNTIME_DIR}/opencv"
    exit 1
  }
  (cd "${MEDIA_RUNTIME_DIR}" && sha256sum -c SHA256SUMS >/dev/null) || {
    fail "Media runtime checksum verification failed: ${MEDIA_RUNTIME_DIR}"
    exit 1
  }
fi

if [[ -n "${SIGLIP2_CACHE_REPO}" ]]; then
  [[ -d "${SIGLIP2_CACHE_REPO}/blobs" && -d "${SIGLIP2_CACHE_REPO}/snapshots" ]] || {
    fail "SigLIP2 cache repository is incomplete: ${SIGLIP2_CACHE_REPO}"
    exit 1
  }
  SIGLIP2_REVISION="75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
  SIGLIP2_SNAPSHOT="${SIGLIP2_CACHE_REPO}/snapshots/${SIGLIP2_REVISION}"
  for required in config.json model.safetensors preprocessor_config.json tokenizer.json tokenizer_config.json; do
    [[ -s "${SIGLIP2_SNAPSHOT}/${required}" ]] || {
      fail "SigLIP2 revision ${SIGLIP2_REVISION} is missing ${required}"
      exit 1
    }
  done
  need_cmd sha256sum
fi

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

BUNDLE_DIR="${TMP_DIR}/${BUNDLE_NAME}"
SNAPSHOT_DIR="${BUNDLE_DIR}/repo"
mkdir -p "${SNAPSHOT_DIR}" "${BUNDLE_DIR}/scripts" "${OUTPUT_DIR}"

if git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  GIT_BRANCH="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || true)"
  GIT_STATUS="$(git -C "${REPO_ROOT}" status --short 2>/dev/null || true)"
else
  GIT_BRANCH="unknown"
  GIT_COMMIT="unknown"
  GIT_STATUS=""
fi

VERSION_VALUE="unknown"
if [[ -f "${REPO_ROOT}/VERSION" ]]; then
  VERSION_VALUE="$(tr -d '\r\n' < "${REPO_ROOT}/VERSION")"
fi

log "Creating working-tree snapshot from ${REPO_ROOT}"

COMMON_EXCLUDES=(
  "--exclude=.git"
  "--exclude=.local"
  "--exclude=.venv"
  "--exclude=.venv*"
  "--exclude=__pycache__"
  "--exclude=*.pyc"
  "--exclude=.pytest_cache"
  "--exclude=.mypy_cache"
  "--exclude=.ruff_cache"
  "--exclude=.coverage"
  "--exclude=htmlcov"
  "--exclude=dist"
  "--exclude=node_modules"
  "--exclude=detections_archive"
  "--exclude=/video/"
  "--exclude=/models/"
  "--exclude=qwen-cookbooks"
  "--exclude=*.mp4"
  "--exclude=*.avi"
  "--exclude=*.mov"
  "--exclude=*.mkv"
  "--exclude=probes_store.json"
  "--exclude=probe_channel_groups.json"
  "--exclude=luxriot_summary_state.json"
  "--exclude=luxriot_rollups_cache.json"
  "--exclude=.env"
  "--exclude=.env.*"
  "--exclude=*.sqlite3"
  "--exclude=*.db"
  "--exclude=*.log"
  "--exclude=*.pid"
  "--exclude=*.sock"
)

if command -v rsync >/dev/null 2>&1; then
  rsync -a "${COMMON_EXCLUDES[@]}" "${REPO_ROOT}/" "${SNAPSHOT_DIR}/"
else
  tar "${COMMON_EXCLUDES[@]}" -cf - -C "${REPO_ROOT}" . | tar -xf - -C "${SNAPSHOT_DIR}"
fi

# An rsync/tar exclude intended for mutable root runtime data must never match a
# same-named tracked source directory (for example react-ui/.../video). The
# manifest's clean commit is only meaningful when every tracked file made it
# into the snapshot.
"${PYTHON_BIN}" - "${REPO_ROOT}" "${SNAPSHOT_DIR}" <<'PY'
import subprocess
import sys
from pathlib import Path

repo = Path(sys.argv[1])
snapshot = Path(sys.argv[2])
tracked = subprocess.check_output(
    ("git", "-C", str(repo), "ls-files", "-z")
).split(b"\0")
missing = [
    raw.decode("utf-8", errors="surrogateescape")
    for raw in tracked
    if raw and not (snapshot / raw.decode("utf-8", errors="surrogateescape")).is_file()
]
if missing:
    print("FAIL: tracked source file missing from bundle snapshot:", file=sys.stderr)
    for path in missing[:40]:
        print(f"  {path}", file=sys.stderr)
    raise SystemExit(1)
print(f"OK: tracked source snapshot complete ({len(tracked) - 1} files)")
PY

if [[ ! -f "${REPO_ROOT}/react-ui/dist/index.html" ]]; then
  fail "React production build is missing: ${REPO_ROOT}/react-ui/dist/index.html"
  exit 1
fi
mkdir -p "${SNAPSHOT_DIR}/react-ui/dist"
cp -a "${REPO_ROOT}/react-ui/dist/." "${SNAPSHOT_DIR}/react-ui/dist/"

for script_name in install_patch.sh install_media_runtime.sh restore_code_snapshot.py verify_patch.sh rollback.sh set_site_ips.sh client_diagnostics.sh preflight_patch.sh pg_with_dsn.py; do
  if [[ -f "${REPO_ROOT}/scripts/${script_name}" ]]; then
    cp "${REPO_ROOT}/scripts/${script_name}" "${BUNDLE_DIR}/scripts/${script_name}"
    chmod 0755 "${BUNDLE_DIR}/scripts/${script_name}"
  fi
done

if [[ -f "${REPO_ROOT}/scripts/update_bundle.sh" ]]; then
  cp "${REPO_ROOT}/scripts/update_bundle.sh" "${BUNDLE_DIR}/update.sh"
  chmod 0755 "${BUNDLE_DIR}/update.sh"
fi

if [[ "${INCLUDE_WHEELHOUSE}" == true ]]; then
  mkdir -p "${BUNDLE_DIR}/wheelhouse"
  if [[ -n "${WHEELHOUSE_DIR}" ]]; then
    [[ -d "${WHEELHOUSE_DIR}" ]] || {
      fail "Wheelhouse directory not found: ${WHEELHOUSE_DIR}"
      exit 1
    }
    log "Copying existing wheelhouse from ${WHEELHOUSE_DIR}"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a "${WHEELHOUSE_DIR}/" "${BUNDLE_DIR}/wheelhouse/"
    else
      tar -cf - -C "${WHEELHOUSE_DIR}" . | tar -xf - -C "${BUNDLE_DIR}/wheelhouse"
    fi
  else
    log "Downloading offline wheels with ${PYTHON_BIN}"
    if [[ -x "${PYTHON_BIN}" ]]; then
      PIP_PYTHON="${PYTHON_BIN}"
    elif command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
      PIP_PYTHON="$(command -v "${PYTHON_BIN}")"
    else
      fail "Python not found for wheel download: ${PYTHON_BIN}"
      exit 1
    fi
    "${PIP_PYTHON}" -m pip download --dest "${BUNDLE_DIR}/wheelhouse" -r "${REPO_ROOT}/requirements.txt"
    if [[ -f "${REPO_ROOT}/requirements-db.txt" ]]; then
      "${PIP_PYTHON}" -m pip download --dest "${BUNDLE_DIR}/wheelhouse" -r "${REPO_ROOT}/requirements-db.txt"
    fi
    if [[ -f "${REPO_ROOT}/requirements-cuda.txt" ]]; then
      "${PIP_PYTHON}" -m pip download --dest "${BUNDLE_DIR}/wheelhouse" -r "${REPO_ROOT}/requirements-cuda.txt"
    fi
  fi
  {
    printf 'created_at=%s\n' "$(date -Is)"
    printf 'python_bin=%s\n' "${PYTHON_BIN}"
    if command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
      "${PYTHON_BIN}" --version 2>&1 | sed 's/^/python_version=/'
    elif [[ -x "${PYTHON_BIN}" ]]; then
      "${PYTHON_BIN}" --version 2>&1 | sed 's/^/python_version=/'
    fi
    printf 'requirements=requirements.txt requirements-db.txt requirements-cuda.txt\n'
    if [[ -n "${WHEELHOUSE_DIR}" ]]; then
      printf 'wheelhouse_source_dir=%s\n' "${WHEELHOUSE_DIR}"
    fi
    printf 'wheel_count=%s\n' "$(find "${BUNDLE_DIR}/wheelhouse" -type f \( -name '*.whl' -o -name '*.tar.gz' -o -name '*.zip' \) | wc -l | tr -d ' ')"
  } > "${BUNDLE_DIR}/wheelhouse_manifest.txt"
  ok "wheelhouse included"
fi

if [[ -n "${FFMPEG_ARCHIVE}" ]]; then
  log "Including self-contained FFmpeg/ffprobe and OpenCV rescue payload"
  MEDIA_DIR="${BUNDLE_DIR}/runtime"
  FFMPEG_EXTRACT_DIR="${TMP_DIR}/ffmpeg-extract"
  mkdir -p "${MEDIA_DIR}/ffmpeg/bin" "${MEDIA_DIR}/opencv" "${FFMPEG_EXTRACT_DIR}"
  FFMPEG_ROOT="$(tar -tJf "${FFMPEG_ARCHIVE}" | sed -n '1{s:/*$::;p;}')"
  [[ -n "${FFMPEG_ROOT}" ]] || {
    fail "FFmpeg archive is empty: ${FFMPEG_ARCHIVE}"
    exit 1
  }
  tar -xJf "${FFMPEG_ARCHIVE}" -C "${FFMPEG_EXTRACT_DIR}" \
    "${FFMPEG_ROOT}/bin/ffmpeg" \
    "${FFMPEG_ROOT}/bin/ffprobe" \
    "${FFMPEG_ROOT}/LICENSE.txt"
  install -m 0755 "${FFMPEG_EXTRACT_DIR}/${FFMPEG_ROOT}/bin/ffmpeg" "${MEDIA_DIR}/ffmpeg/bin/ffmpeg"
  install -m 0755 "${FFMPEG_EXTRACT_DIR}/${FFMPEG_ROOT}/bin/ffprobe" "${MEDIA_DIR}/ffmpeg/bin/ffprobe"
  install -m 0644 "${FFMPEG_EXTRACT_DIR}/${FFMPEG_ROOT}/LICENSE.txt" "${MEDIA_DIR}/ffmpeg/LICENSE.txt"
  install -m 0644 "${OPENCV_WHEEL}" "${MEDIA_DIR}/opencv/$(basename "${OPENCV_WHEEL}")"
  {
    printf 'platform=linux-x86_64\n'
    printf 'ffmpeg_source_archive=%s\n' "$(basename "${FFMPEG_ARCHIVE}")"
    printf 'ffmpeg_source_sha256=%s\n' "$(sha256sum "${FFMPEG_ARCHIVE}" | awk '{print $1}')"
    printf 'opencv_wheel=%s\n' "$(basename "${OPENCV_WHEEL}")"
    printf 'opencv_wheel_sha256=%s\n' "$(sha256sum "${OPENCV_WHEEL}" | awk '{print $1}')"
  } > "${MEDIA_DIR}/manifest.txt"
  (
    cd "${MEDIA_DIR}"
    find ffmpeg opencv -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
  )
  ok "media runtime included"
elif [[ -n "${MEDIA_RUNTIME_DIR}" ]]; then
  log "Reusing checksummed offline media runtime"
  mkdir -p "${BUNDLE_DIR}/runtime"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "${MEDIA_RUNTIME_DIR}/" "${BUNDLE_DIR}/runtime/"
  else
    tar -cf - -C "${MEDIA_RUNTIME_DIR}" . | tar -xf - -C "${BUNDLE_DIR}/runtime"
  fi
  ok "validated media runtime included"
fi

if [[ -n "${SIGLIP2_CACHE_REPO}" ]]; then
  SIGLIP2_TARGET="${BUNDLE_DIR}/models/huggingface/models--google--siglip2-base-patch16-224"
  mkdir -p "$(dirname "${SIGLIP2_TARGET}")"
  log "Including offline SigLIP2 model cache"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "${SIGLIP2_CACHE_REPO}/" "${SIGLIP2_TARGET}/"
  else
    tar -cf - -C "${SIGLIP2_CACHE_REPO}" . | tar -xf - -C "${SIGLIP2_TARGET}"
  fi
  [[ -n "$(find "${SIGLIP2_TARGET}/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit)" ]] || {
    fail "SigLIP2 cache has no materialized snapshot"
    exit 1
  }
  (
    cd "${BUNDLE_DIR}/models/huggingface"
    find models--google--siglip2-base-patch16-224 -type f -print0 \
      | sort -z \
      | xargs -0 sha256sum > SHA256SUMS
    sha256sum -c SHA256SUMS >/dev/null
  )
  ok "checksummed SigLIP2 offline model included"
fi

{
  printf 'bundle_name=%s\n' "${BUNDLE_NAME}"
  printf 'created_at=%s\n' "$(date -Is)"
  printf 'source_path=%s\n' "${REPO_ROOT}"
  printf 'git_branch=%s\n' "${GIT_BRANCH}"
  printf 'git_commit=%s\n' "${GIT_COMMIT}"
  printf 'version=%s\n' "${VERSION_VALUE}"
  if [[ "${INCLUDE_WHEELHOUSE}" == true ]]; then
    printf 'wheelhouse=included\n'
  else
    printf 'wheelhouse=not_included\n'
  fi
  if [[ -n "${FFMPEG_ARCHIVE}" || -n "${MEDIA_RUNTIME_DIR}" ]]; then
    printf 'media_runtime=included\n'
    printf 'media_runtime_platform=linux-x86_64\n'
  else
    printf 'media_runtime=not_included\n'
  fi
  if [[ -n "${SIGLIP2_CACHE_REPO}" ]]; then
    printf 'siglip2_model=included\n'
  else
    printf 'siglip2_model=not_included\n'
  fi
  if [[ -n "${GIT_STATUS}" ]]; then
    printf 'working_tree_status=dirty\n'
    printf '\nChanged files at build time:\n'
    printf '%s\n' "${GIT_STATUS}"
  else
    printf 'working_tree_status=clean\n'
  fi
} > "${BUNDLE_DIR}/manifest.txt"

ARCHIVE_PATH="${OUTPUT_DIR}/${BUNDLE_NAME}.tar.gz"
if command -v pigz >/dev/null 2>&1; then
  COMPRESS_PROGRAM="pigz -1"
else
  COMPRESS_PROGRAM="gzip -1"
fi
log "Compressing bundle with ${COMPRESS_PROGRAM}"
tar --use-compress-program="${COMPRESS_PROGRAM}" -cf "${ARCHIVE_PATH}" \
  -C "${TMP_DIR}" "${BUNDLE_NAME}"

if command -v sha256sum >/dev/null 2>&1; then
  (cd "${OUTPUT_DIR}" && sha256sum "${BUNDLE_NAME}.tar.gz" > "${BUNDLE_NAME}.tar.gz.sha256")
  ok "wrote ${ARCHIVE_PATH}.sha256"
fi

ok "wrote ${ARCHIVE_PATH}"
log "Copy the .tar.gz and .sha256 files to the USB drive."
