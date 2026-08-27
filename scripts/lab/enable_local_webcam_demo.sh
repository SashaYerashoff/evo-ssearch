#!/usr/bin/env bash
set -Eeuo pipefail

SERVICE_NAME="${EVA_DEMO_SERVICE:-eva-ai.service}"
DEVICE="${EVA_DEMO_CAMERA_DEVICE:-/dev/video0}"
CHANNEL_ID="${EVA_DEMO_CAMERA_ID:-900001}"
CHANNEL_TITLE="${EVA_DEMO_CAMERA_TITLE:-Demo — local USB camera}"
WIDTH="${EVA_DEMO_CAMERA_WIDTH:-1280}"
HEIGHT="${EVA_DEMO_CAMERA_HEIGHT:-720}"
FPS="${EVA_DEMO_CAMERA_FPS:-15}"
PREVIEW_FPS="${EVA_DEMO_CAMERA_PREVIEW_FPS:-8}"
BACKUP_PARENT="/var/backups/eva-ai"

fail() {
  printf 'LOCAL WEBCAM DEMO ERROR: %s\n' "$*" >&2
  exit 1
}

if (( EUID != 0 )); then
  exec sudo -- "$0" "$@"
fi

[[ "${DEVICE}" =~ ^/dev/video[0-9]+$ ]] \
  || fail "unsafe V4L2 device path: ${DEVICE}"
[[ "${CHANNEL_ID}" =~ ^[0-9]+$ ]] && (( CHANNEL_ID > 0 )) \
  || fail "camera channel ID must be a positive integer"
[[ -c "${DEVICE}" ]] || fail "camera device is unavailable: ${DEVICE}"

app_dir="$(systemctl show "${SERVICE_NAME}" -p WorkingDirectory --value)"
env_spec="$(systemctl show "${SERVICE_NAME}" -p EnvironmentFiles --value)"
env_file="${env_spec%% *}"
service_user="$(systemctl show "${SERVICE_NAME}" -p User --value)"

[[ -n "${app_dir}" && -d "${app_dir}" ]] \
  || fail "could not resolve the EVA application directory from ${SERVICE_NAME}"
[[ -n "${env_file}" && -f "${env_file}" ]] \
  || fail "could not resolve the EVA environment file from ${SERVICE_NAME}"
[[ "${service_user}" == "eva" ]] \
  || fail "refusing unexpected service user: ${service_user:-missing}"
[[ -x "${app_dir}/.venv/bin/python" ]] \
  || fail "installed EVA Python is unavailable"

install -d -m 0750 "${BACKUP_PARENT}"
stamp="$(date -u +%Y%m%d-%H%M%S)"
backup_dir="${BACKUP_PARENT}/local-webcam-demo-${stamp}"
install -d -m 0750 "${backup_dir}"
cp -a -- "${env_file}" "${backup_dir}/eva-ai.env"

python3 - \
  "${env_file}" "${DEVICE}" "${CHANNEL_ID}" "${CHANNEL_TITLE}" \
  "${WIDTH}" "${HEIGHT}" "${FPS}" "${PREVIEW_FPS}" <<'PY'
import json
import os
import stat
import sys
import tempfile
from pathlib import Path

(
    raw_path,
    device,
    channel_id,
    title,
    width,
    height,
    fps,
    preview_fps,
) = sys.argv[1:]
path = Path(raw_path)
key = "EVOSSEARCH_LOCAL_VIDEO_SOURCES_JSON"
original = path.read_text(encoding="utf-8")
lines = original.splitlines()
existing_raw = ""
kept: list[str] = []
for line in lines:
    if line.startswith(f"{key}="):
        if not existing_raw:
            existing_raw = line.split("=", 1)[1].strip()
        continue
    kept.append(line)

if len(existing_raw) >= 2 and existing_raw[0] == existing_raw[-1] \
        and existing_raw[0] in {"'", '"'}:
    existing_raw = existing_raw[1:-1]
try:
    sources = json.loads(existing_raw) if existing_raw else []
except json.JSONDecodeError as exc:
    raise SystemExit(f"Existing {key} is invalid JSON: {exc}") from exc
if not isinstance(sources, list):
    raise SystemExit(f"Existing {key} must contain a JSON array")

camera = {
    "id": int(channel_id),
    "title": title,
    "device": device,
    "input_format": "mjpeg",
    "width": int(width),
    "height": int(height),
    "fps": int(fps),
    "preview_fps": int(preview_fps),
}
sources = [
    item for item in sources
    if not (
        isinstance(item, dict)
        and (
            str(item.get("id") or "") == str(camera["id"])
            or item.get("device") == device
        )
    )
]
sources.append(camera)
compact = json.dumps(sources, ensure_ascii=False, separators=(",", ":"))
kept.append(f"{key}='{compact}'")
updated = "\n".join(kept) + "\n"

metadata = path.stat()
descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
temporary = Path(temporary_name)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(updated)
        handle.flush()
        os.fsync(handle.fileno())
    os.chown(temporary, metadata.st_uid, metadata.st_gid)
    os.chmod(temporary, stat.S_IMODE(metadata.st_mode))
    os.replace(temporary, path)
finally:
    temporary.unlink(missing_ok=True)
PY

usermod -aG video eva

ffmpeg_bin="$(command -v ffmpeg || true)"
[[ -x "${ffmpeg_bin}" ]] || fail "system FFmpeg is unavailable"
printf 'Checking one real MJPEG frame from %s as user eva...\n' "${DEVICE}"
timeout 20s runuser -u eva -- "${ffmpeg_bin}" \
  -hide_banner -loglevel error -nostdin \
  -f v4l2 -input_format mjpeg -video_size "${WIDTH}x${HEIGHT}" \
  -framerate "${FPS}" -i "${DEVICE}" \
  -frames:v 1 -f null - \
  || fail "the EVA service user could not capture a frame from ${DEVICE}"

systemctl restart "${SERVICE_NAME}"

ready=false
for _ in $(seq 1 60); do
  if curl -fsS --max-time 2 http://127.0.0.1:5000/ready \
      | python3 -c 'import json,sys; raise SystemExit(json.load(sys.stdin).get("status") != "ready")' \
      >/dev/null 2>&1; then
    ready=true
    break
  fi
  sleep 2
done
[[ "${ready}" == true ]] || fail "EVA did not become ready after the restart"

server_ip="$(hostname -I | awk '{print $1}')"
printf '\nLOCAL WEBCAM DEMO READY\n'
printf '  EVA:     https://%s/\n' "${server_ip:-127.0.0.1}"
printf '  channel: %s (#%s)\n' "${CHANNEL_TITLE}" "${CHANNEL_ID}"
printf '  device:  %s · %sx%s · %s fps\n' "${DEVICE}" "${WIDTH}" "${HEIGHT}" "${FPS}"
printf '  backup:  %s\n' "${backup_dir}"
printf '\nOpen Stream summaries or Probes, select the local camera, and start the stream.\n'
