#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

BUNDLE_DIR=""
APP_DIR=""
PYTHON_BIN=""
OWNER=""
WITH_OPENCV=false

die() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle-dir) BUNDLE_DIR="$2"; shift 2 ;;
    --app-dir) APP_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --owner) OWNER="$2"; shift 2 ;;
    --with-opencv-overlay) WITH_OPENCV=true; shift ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -d "${BUNDLE_DIR}/runtime" ]] || die "bundle media runtime is missing"
[[ -d "${APP_DIR}" ]] || die "application directory is missing: ${APP_DIR}"
[[ -x "${PYTHON_BIN}" ]] || die "target Python is missing: ${PYTHON_BIN}"
[[ "$(uname -m)" == "x86_64" ]] || die "media runtime requires Linux x86_64"

(
  cd "${BUNDLE_DIR}/runtime"
  sha256sum -c SHA256SUMS >/dev/null
) || die "media runtime checksum verification failed"

FFMPEG_SOURCE="${BUNDLE_DIR}/runtime/ffmpeg/bin/ffmpeg"
FFPROBE_SOURCE="${BUNDLE_DIR}/runtime/ffmpeg/bin/ffprobe"
[[ -x "${FFMPEG_SOURCE}" && -x "${FFPROBE_SOURCE}" ]] || die "ffmpeg/ffprobe payload is incomplete"

RUNTIME_DIR="${APP_DIR}/.eva-runtime"
STAGE_DIR="${APP_DIR}/.eva-runtime.new.$$"
PREVIOUS_DIR="${APP_DIR}/.eva-runtime.previous.$$"
cleanup() {
  rm -rf "${STAGE_DIR}"
  if [[ -d "${PREVIOUS_DIR}" && ! -e "${RUNTIME_DIR}" ]]; then
    mv "${PREVIOUS_DIR}" "${RUNTIME_DIR}"
  fi
}
trap cleanup EXIT

rm -rf "${STAGE_DIR}" "${PREVIOUS_DIR}"
mkdir -p "${STAGE_DIR}/bin"
install -m 0755 "${FFMPEG_SOURCE}" "${STAGE_DIR}/bin/ffmpeg"
install -m 0755 "${FFPROBE_SOURCE}" "${STAGE_DIR}/bin/ffprobe"
install -m 0644 "${BUNDLE_DIR}/runtime/ffmpeg/LICENSE.txt" "${STAGE_DIR}/FFMPEG-LICENSE.txt"
install -m 0644 "${BUNDLE_DIR}/runtime/manifest.txt" "${STAGE_DIR}/manifest.txt"

if [[ "${WITH_OPENCV}" == true ]]; then
  mapfile -t OPENCV_WHEELS < <(find "${BUNDLE_DIR}/runtime/opencv" -maxdepth 1 -type f -name 'opencv_python_headless-*.whl' -print)
  [[ "${#OPENCV_WHEELS[@]}" -eq 1 ]] || die "expected exactly one OpenCV wheel"
  mkdir -p "${STAGE_DIR}/python"
  "${PYTHON_BIN}" -m zipfile -e "${OPENCV_WHEELS[0]}" "${STAGE_DIR}/python"
  "${PYTHON_BIN}" - "${STAGE_DIR}/python" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import cv2
import numpy as np
image = np.zeros((8, 8, 3), dtype=np.uint8)
assert cv2.cvtColor(image, cv2.COLOR_BGR2RGB).shape == (8, 8, 3)
PY
fi

"${STAGE_DIR}/bin/ffmpeg" -v error -f lavfi -i color=c=black:s=16x16:d=0.05 \
  -frames:v 1 -f image2pipe -vcodec mjpeg - >/dev/null
"${STAGE_DIR}/bin/ffprobe" -version >/dev/null

if [[ -e "${RUNTIME_DIR}" ]]; then
  mv "${RUNTIME_DIR}" "${PREVIOUS_DIR}"
fi
mv "${STAGE_DIR}" "${RUNTIME_DIR}"
if [[ -n "${OWNER}" ]]; then
  chown -R "${OWNER}" "${RUNTIME_DIR}"
fi
rm -rf "${PREVIOUS_DIR}"
trap - EXIT
printf 'OK: offline FFmpeg runtime installed at %s\n' "${RUNTIME_DIR}"
if [[ "${WITH_OPENCV}" == true ]]; then
  printf 'OK: OpenCV rescue overlay installed\n'
else
  printf 'OK: existing OpenCV installation retained\n'
fi
