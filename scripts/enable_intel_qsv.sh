#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Run this script with sudo." >&2
  exit 1
fi

TARGET_USER="${SUDO_USER:-admins}"
if [[ "${TARGET_USER}" == "root" ]]; then
  TARGET_USER="admins"
fi
QSV_DEVICE="${EVOSSEARCH_LUXRIOT_FFMPEG_QSV_DEVICE:-/dev/dri/renderD128}"

if [[ ! -e "${QSV_DEVICE}" ]]; then
  echo "Intel render node is missing: ${QSV_DEVICE}" >&2
  exit 1
fi

vendor_path="/sys/class/drm/$(basename "${QSV_DEVICE}")/device/vendor"
vendor="$(tr '[:upper:]' '[:lower:]' < "${vendor_path}" 2>/dev/null || true)"
if [[ "${vendor}" != "0x8086" ]]; then
  echo "Refusing non-Intel render node ${QSV_DEVICE} (vendor ${vendor:-unknown})." >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
if ! apt-get install -y intel-media-va-driver-non-free vainfo; then
  apt-get install -y intel-media-va-driver vainfo
fi

getent group render >/dev/null && usermod -aG render "${TARGET_USER}"
getent group video >/dev/null && usermod -aG video "${TARGET_USER}"

runuser -u "${TARGET_USER}" -- vainfo --display drm --device "${QSV_DEVICE}" >/tmp/eva-vainfo.txt
if runuser -u "${TARGET_USER}" -- ffmpeg \
    -hide_banner -loglevel error \
    -init_hw_device "qsv=eva_qsv:${QSV_DEVICE}" \
    -filter_hw_device eva_qsv \
    -f lavfi -i 'color=size=64x64:rate=1:duration=0.1' \
    -vf 'format=nv12,hwupload=extra_hw_frames=8,vpp_qsv=w=32:h=32:format=nv12' \
    -frames:v 1 -f null -; then
  backend="qsv"
else
  echo "Direct QSV initialization is unavailable; testing Intel VAAPI fallback."
  runuser -u "${TARGET_USER}" -- ffmpeg \
    -hide_banner -loglevel error \
    -init_hw_device "vaapi=eva_va:${QSV_DEVICE}" \
    -filter_hw_device eva_va \
    -f lavfi -i 'color=size=64x64:rate=1:duration=0.1' \
    -vf 'format=nv12,hwupload=extra_hw_frames=8,scale_vaapi=w=32:h=32:format=nv12' \
    -frames:v 1 -f null -
  backend="vaapi"
fi

echo "Intel media runtime is ready on ${QSV_DEVICE} for ${TARGET_USER} (${backend})."
echo "Restart eva-ai after deploying the QSV-aware application build."
