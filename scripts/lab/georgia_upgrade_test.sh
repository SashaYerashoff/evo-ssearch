#!/usr/bin/env bash
set -Eeuo pipefail

BUNDLE_NAME="eva-ai-georgia-upgrade-0.8.1-to-0.8.7-1ce3eb0"
BUNDLE_ARCHIVE="/run/media/sasha/writable/EVA-AI-0.8.7-UNIVERSAL/updates/georgia-0.8.1-to-0.8.7-1ce3eb0/${BUNDLE_NAME}.tar.gz"
BUNDLE_CHECKSUM="${BUNDLE_ARCHIVE}.sha256"
BUNDLE_ROOT="/run/media/sasha/writable/EVA-AI-0.8.7-UNIVERSAL/updates/georgia-0.8.1-to-0.8.7-1ce3eb0/${BUNDLE_NAME}"
SOURCE_ROOT="${BUNDLE_ROOT}/repo"
TARGET_ROOT="/home/sasha/Projects/eva-georgia-upgrade-repro"
ENV_FILE="${TARGET_ROOT}/.env"
SERVICE_NAME="eva-ai-georgia-repro"
UNIT_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
BACKUP_ROOT="/var/backups/${SERVICE_NAME}"
LOCK_FILE="/run/lock/${SERVICE_NAME}.lock"
BASE_URL="http://127.0.0.1:5081"
DB_CONTAINER="eva-tbilisi-repro-postgres"
PG_CLIENT_ROOT="${TARGET_ROOT}/.pg-client"
PG_CLIENT_BIN="${PG_CLIENT_ROOT}/usr/lib/postgresql/16/bin"
PG_CLIENT_LIB="${PG_CLIENT_ROOT}/usr/lib/x86_64-linux-gnu"
EXPECTED_BASELINE_VERSION="β 0.8.1"
EXPECTED_BASELINE_SCHEMA="20260614_0006"
EXPECTED_TARGET_VERSION="β 0.8.7"
EXPECTED_TARGET_SCHEMA="20260805_0013"
EXPECTED_COMMIT="1ce3eb0a481c874b2cd20e9dfc213d11d9d6b7f2"

export PATH="${PG_CLIENT_BIN}:${PATH}"
export LD_LIBRARY_PATH="${PG_CLIENT_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

[[ -f "${BUNDLE_ARCHIVE}" ]] || die "upgrade archive is missing"
[[ -f "${BUNDLE_CHECKSUM}" ]] || die "upgrade checksum is missing"
if [[ "${EVA_REHEARSAL_ARCHIVE_VERIFIED:-0}" != "1" ]]; then
  (cd "$(dirname "${BUNDLE_ARCHIVE}")" && sha256sum -c "$(basename "${BUNDLE_CHECKSUM}")")
fi

if [[ ! -f "${BUNDLE_ROOT}/manifest.txt" ]]; then
  tar -xzf "${BUNDLE_ARCHIVE}" -C "$(dirname "${BUNDLE_ROOT}")"
fi

manifest_value() {
  local key="$1"
  sed -n "s/^${key}=//p" "${BUNDLE_ROOT}/manifest.txt" | tail -n 1
}

[[ "$(manifest_value working_tree_status)" == "clean" ]] || die "bundle was built from a dirty tree"
[[ "$(manifest_value git_commit)" == "${EXPECTED_COMMIT}" ]] || die "unexpected bundle commit"
[[ "$(manifest_value version)" == "${EXPECTED_TARGET_VERSION}" ]] || die "unexpected bundle version"
[[ "$(manifest_value media_runtime)" == "included" ]] || die "offline media runtime is missing"
[[ "$(manifest_value siglip2_model)" == "included" ]] || die "offline SigLIP2 model is missing"
[[ -f "${SOURCE_ROOT}/react-ui/dist/index.html" ]] || die "React production build is missing"
(cd "${BUNDLE_ROOT}/runtime" && sha256sum -c SHA256SUMS)

[[ -f "${TARGET_ROOT}/VERSION" ]] || die "Georgia baseline application is missing"
[[ -f "${ENV_FILE}" ]] || die "Georgia baseline environment is missing"
[[ -L "${TARGET_ROOT}/.venv" ]] || die "Georgia baseline venv is not the expected symlink"
docker inspect "${DB_CONTAINER}" >/dev/null 2>&1 || die "PostgreSQL reproduction container is unavailable"

container_value() {
  local key="$1"
  docker inspect "${DB_CONTAINER}" \
    --format '{{range .Config.Env}}{{println .}}{{end}}' \
    | sed -n "s/^${key}=//p" \
    | head -n 1
}

PG_USER="$(container_value POSTGRES_USER)"
PG_PASSWORD="$(container_value POSTGRES_PASSWORD)"
PG_DATABASE="$(container_value POSTGRES_DB)"
[[ -n "${PG_USER}" && -n "${PG_PASSWORD}" && -n "${PG_DATABASE}" ]] \
  || die "could not resolve reproduction database credentials"
MIGRATION_DSN="host=127.0.0.1 port=15433 dbname=${PG_DATABASE} user=${PG_USER} password=${PG_PASSWORD}"

BASELINE_VERSION="$(tr -d '\r\n' < "${TARGET_ROOT}/VERSION")"
BASELINE_SCHEMA="$(docker exec "${DB_CONTAINER}" psql -U "${PG_USER}" -d "${PG_DATABASE}" -Atc 'select version_num from public.alembic_version')"
BASELINE_ARCHIVE="$(docker exec "${DB_CONTAINER}" psql -U "${PG_USER}" -d "${PG_DATABASE}" -Atc \
  "select count(*) || '|' || count(thumbnail_b64) || '|' || coalesce(max(id),0) || '|' || coalesce(max(event_timestamp_ms),0) from archive.detections")"
IFS='|' read -r BASELINE_ROWS BASELINE_THUMBNAILS BASELINE_MAX_ID BASELINE_MAX_TIMESTAMP <<< "${BASELINE_ARCHIVE}"

[[ "${BASELINE_VERSION}" == "${EXPECTED_BASELINE_VERSION}" ]] || die "baseline is ${BASELINE_VERSION}, run RESET first"
[[ "${BASELINE_SCHEMA}" == "${EXPECTED_BASELINE_SCHEMA}" ]] || die "baseline schema is ${BASELINE_SCHEMA}, run RESET first"
[[ "${BASELINE_ROWS}" == "8683" && "${BASELINE_THUMBNAILS}" == "8683" ]] \
  || die "baseline archive is incomplete (${BASELINE_ROWS}/${BASELINE_THUMBNAILS})"
[[ ! -e "${UNIT_FILE}" ]] || die "rehearsal service already exists, run RESET first"

hash_inference_configuration() {
  find /home/sasha/.config/systemd/user \
    -maxdepth 3 -type f \
    \( -name 'eva-vllm-qwen3-vl-4b.service' \
       -o -path '*/eva-vllm-qwen3-vl-4b.service.d/*' \
       -o -name 'eva-llama-qwen3-vl-4b.service' \
       -o -path '*/eva-llama-qwen3-vl-4b.service.d/*' \
       -o -name 'eva-llama-qwen35-mtp.service' \
       -o -path '*/eva-llama-qwen35-mtp.service.d/*' \
       -o -name 'eva-vlm-vision-watchdog.service' \
       -o -name 'eva-vlm-vision-watchdog.timer' \
       -o -path '*/eva-vlm-vision-watchdog.service.d/*' \
       -o -path '*/eva-vlm-vision-watchdog.timer.d/*' \) \
    -print0 \
    | sort -z \
    | xargs -0 -r sha256sum \
    | sha256sum \
    | awk '{print $1}'
}

hash_protected_site_configuration() {
  sed -n -E \
    '/^[[:space:]]*(export[[:space:]]+)?(EVA_DATABASE_DSN|EVOSSEARCH_(HOST|PORT|DATABASE_DSN|ARCHIVE_TENANT_ID|LUXRIOT_[A-Za-z0-9_]+|LM_[A-Za-z0-9_]+|LIVE_CLIP_[A-Za-z0-9_]+)|LUXRIOT_[A-Za-z0-9_]+|LIVE_CLIP_[A-Za-z0-9_]+)[[:space:]]*=/p' \
    "${ENV_FILE}" \
    | sed -E 's/^[[:space:]]*export[[:space:]]+//' \
    | sort \
    | sha256sum \
    | awk '{print $1}'
}

BEFORE_INFERENCE_HASH="$(hash_inference_configuration)"
BEFORE_SITE_CONFIG_HASH="$(hash_protected_site_configuration)"

COMMON_ARGS=(
  --non-interactive
  --source-dir "${SOURCE_ROOT}"
  --bundle-dir "${BUNDLE_ROOT}"
  --app-dir "${TARGET_ROOT}"
  --env-file "${ENV_FILE}"
  --backup-root "${BACKUP_ROOT}"
  --service-name "${SERVICE_NAME}"
  --service-user sasha
  --service-group sasha
  --unit-file "${UNIT_FILE}"
  --lock-file "${LOCK_FILE}"
  --base-url "${BASE_URL}"
  --verify-luxriot-credential
)

printf '\nEVA AI Georgia upgrade rehearsal\n'
printf '  updater:           %s (%s)\n' "${EXPECTED_TARGET_VERSION}" "${EXPECTED_COMMIT:0:7}"
printf '  installed:         %s\n' "${BASELINE_VERSION}"
printf '  database:          %s\n' "${BASELINE_SCHEMA}"
printf '  preserved archive: %s rows / %s thumbnails\n' "${BASELINE_ROWS}" "${BASELINE_THUMBNAILS}"
printf '  VLM primary:       http://192.168.1.110:8080/v1 (capacity 4)\n'
printf '  VLM local:         http://127.0.0.1:1234/v1 (capacity 1)\n'
printf '  agent:             http://127.0.0.1:1235/v1\n'

printf '\nREAD-ONLY PREFLIGHT\n'
EVA_INSTALL_MIGRATION_DSN="${MIGRATION_DSN}" \
  python3 "${SOURCE_ROOT}/scripts/install_eva_083.py" \
  "${COMMON_ARGS[@]}" --dry-run

printf '\nThe preflight above made no changes.\n'
read -r -p 'Apply the checked 0.8.1 -> 0.8.7 update now? [y/N] ' answer
case "${answer}" in
  y|Y|yes|YES|Yes) ;;
  *) printf 'Cancelled; no update was applied.\n'; exit 0 ;;
esac

printf '\nAPPLYING UPDATE\n'
sudo env \
  PATH="${PATH}" \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
  EVA_INSTALL_MIGRATION_DSN="${MIGRATION_DSN}" \
  python3 "${SOURCE_ROOT}/scripts/install_eva_083.py" \
  "${COMMON_ARGS[@]}" --apply

AFTER_INFERENCE_HASH="$(hash_inference_configuration)"
AFTER_SITE_CONFIG_HASH="$(hash_protected_site_configuration)"
[[ "${BEFORE_INFERENCE_HASH}" == "${AFTER_INFERENCE_HASH}" ]] \
  || die "inference systemd configuration changed during the EVA update"
[[ "${BEFORE_SITE_CONFIG_HASH}" == "${AFTER_SITE_CONFIG_HASH}" ]] \
  || die "protected site/external-inference/Luxriot values in .env changed during the EVA update"

mapfile -t SEMANTIC_CONFIG < <("${TARGET_ROOT}/.venv/bin/python" - "${ENV_FILE}" <<'PY'
import sys
from pathlib import Path

values = {}
for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    if "=" not in raw or raw.lstrip().startswith("#"):
        continue
    key, value = raw.split("=", 1)
    values[key.strip()] = value.strip().strip("\"").strip("'")
print(values.get("EVOSSEARCH_CLIP_MODEL", ""))
print(values.get("EVOSSEARCH_PRODUCTION_CLIP_MODEL", ""))
print(values.get("EVOSSEARCH_CLIP_MODEL_REVISION", ""))
print(values.get("EVOSSEARCH_MODEL_CACHE_DIR", ""))
print(values.get("EVOSSEARCH_EMBEDDER_FALLBACK_ENABLED", ""))
PY
)
[[ "${SEMANTIC_CONFIG[0]:-}" == "google/siglip2-base-patch16-224" ]] || die "CLIP backend was not migrated to SigLIP2"
[[ "${SEMANTIC_CONFIG[1]:-}" == "google/siglip2-base-patch16-224" ]] || die "production semantic backend was not migrated to SigLIP2"
[[ "${SEMANTIC_CONFIG[2]:-}" == "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2" ]] || die "unexpected SigLIP2 revision"
[[ "${SEMANTIC_CONFIG[3]:-}" == /* && "${SEMANTIC_CONFIG[3]:-}" != "/" ]] || die "invalid local model cache path"
[[ "${SEMANTIC_CONFIG[4]:-}" == "false" ]] || die "cross-embedding fallback was not disabled"
SIGLIP_SNAPSHOT="${SEMANTIC_CONFIG[3]}/models--google--siglip2-base-patch16-224/snapshots/${SEMANTIC_CONFIG[2]}"
[[ -s "${SIGLIP_SNAPSHOT}/config.json" && -s "${SIGLIP_SNAPSHOT}/model.safetensors" ]] \
  || die "installed SigLIP2 cache is incomplete"

printf '\nPOST-UPDATE ACCEPTANCE\n'
health_payload="$(mktemp)"
ready_payload="$(mktemp)"
cleanup_files() {
  unlink "${health_payload}" 2>/dev/null || true
  unlink "${ready_payload}" 2>/dev/null || true
}
trap cleanup_files EXIT

deadline=$((SECONDS + 300))
until curl -fsS --max-time 5 "${BASE_URL}/health" > "${health_payload}" 2>/dev/null; do
  if systemctl is-failed --quiet "${SERVICE_NAME}.service"; then
    systemctl status "${SERVICE_NAME}.service" --no-pager -l || true
    journalctl -u "${SERVICE_NAME}.service" -n 120 --no-pager || true
    die "EVA service failed while waiting for /health"
  fi
  (( SECONDS < deadline )) || die "EVA /health did not become ready within 300 seconds"
  sleep 2
done

curl -fsS --max-time 10 "${BASE_URL}/ready" > "${ready_payload}" \
  || die "EVA /ready is not healthy"

python3 - "${health_payload}" "${ready_payload}" <<'PY'
import json
import sys
health = json.load(open(sys.argv[1], encoding="utf-8"))
ready = json.load(open(sys.argv[2], encoding="utf-8"))
assert health.get("status") == "ok", health
assert health.get("version") == "β 0.8.7", health
assert ready.get("status") == "ready", ready
assert ready.get("checks", {}).get("embedder", {}).get("ok") is True, ready.get("checks", {}).get("embedder")
assert ready.get("checks", {}).get("luxriot", {}).get("ok") is True, ready.get("checks", {}).get("luxriot")
retention = ready.get("checks", {}).get("database", {}).get("retention", {})
assert retention.get("enabled") is False, retention
print("health: ready β 0.8.7")
print("Luxriot: reachable")
print("semantic backend: SigLIP2 loaded")
print("legacy archive retention: safely disabled pending review")
PY

TARGET_SCHEMA="$(docker exec "${DB_CONTAINER}" psql -U "${PG_USER}" -d "${PG_DATABASE}" -Atc 'select version_num from public.alembic_version')"
[[ "${TARGET_SCHEMA}" == "${EXPECTED_TARGET_SCHEMA}" ]] || die "migration stopped at ${TARGET_SCHEMA}"

ARCHIVE_AFTER="$(docker exec "${DB_CONTAINER}" psql -U "${PG_USER}" -d "${PG_DATABASE}" -Atc \
  "select count(*), count(thumbnail_b64), count(*) filter (where id <= ${BASELINE_MAX_ID} and thumbnail_b64 is null), count(*) filter (where id <= ${BASELINE_MAX_ID}) from archive.detections")"
IFS='|' read -r AFTER_ROWS AFTER_THUMBNAILS LOST_BASELINE_THUMBNAILS PRESERVED_BASELINE_ROWS <<< "${ARCHIVE_AFTER// /|}"
[[ "${PRESERVED_BASELINE_ROWS}" == "${BASELINE_ROWS}" ]] || die "baseline archive rows changed"
[[ "${LOST_BASELINE_THUMBNAILS}" == "0" ]] || die "${LOST_BASELINE_THUMBNAILS} baseline thumbnails were lost"

ROOT_HEADERS="$(curl -sS -D - -o /dev/null "${BASE_URL}/")"
printf '%s' "${ROOT_HEADERS}" | grep -qi '^X-EVA-UI: react' || die "accepted React UI is not active"
[[ -f "${TARGET_ROOT}/react-ui/dist/index.html" ]] || die "installed React payload is absent"

PYTHONPATH="${TARGET_ROOT}/.eva-runtime/python" \
  "${TARGET_ROOT}/.venv/bin/python" -c 'import cv2; print("OpenCV overlay:", cv2.__version__)'
curl -fsS http://127.0.0.1:1234/v1/models \
  | python3 -c 'import json,sys; print("VLM local:", json.load(sys.stdin)["data"][0]["id"])'
curl -fsS http://192.168.1.110:8080/v1/models \
  | python3 -c 'import json,sys; print("VLM primary:", json.load(sys.stdin)["data"][0]["id"])'
curl -fsS http://127.0.0.1:1235/v1/models \
  | python3 -c 'import json,sys; print("Agent:", json.load(sys.stdin)["data"][0]["id"])'

"${TARGET_ROOT}/.venv/bin/python" - "${ENV_FILE}" "${TARGET_ROOT}" <<'PY'
import sys
from pathlib import Path
env = {}
for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    if "=" in raw and not raw.lstrip().startswith("#"):
        key, value = raw.split("=", 1)
        env[key.strip()] = value.strip().strip("\"").strip("'")
sys.path.insert(0, sys.argv[2])
from luxriot_connector import LuxriotClient
client = LuxriotClient(
    env["EVOSSEARCH_LUXRIOT_BASE_URL"],
    env["EVOSSEARCH_LUXRIOT_USERNAME"],
    env["EVOSSEARCH_LUXRIOT_PASSWORD"],
    timeout=15,
)
channels = client.get_channels()
assert channels, "Evo returned no channels"
print("Evo channels:", len(channels), [row.get("id") for row in channels[:12]])
PY

printf 'Waiting for one fresh live archive batch...\n'
live_deadline=$((SECONDS + 90))
while (( SECONDS < live_deadline )); do
  latest_timestamp="$(docker exec "${DB_CONTAINER}" psql -U "${PG_USER}" -d "${PG_DATABASE}" -Atc \
    'select coalesce(max(event_timestamp_ms),0) from archive.detections')"
  if (( latest_timestamp > BASELINE_MAX_TIMESTAMP )); then
    break
  fi
  sleep 3
done
(( latest_timestamp > BASELINE_MAX_TIMESTAMP )) || die "no fresh Evo archive row arrived within 90 seconds"

printf '\nUPGRADE ACCEPTED\n'
printf '  EVA:               %s, React UI\n' "${EXPECTED_TARGET_VERSION}"
printf '  database:          %s\n' "${TARGET_SCHEMA}"
printf '  old archive:       %s/%s rows preserved, 0 thumbnails lost\n' "${PRESERVED_BASELINE_ROWS}" "${BASELINE_ROWS}"
printf '  live archive:      new data arrived\n'
printf '  inference config:  unchanged (%s)\n' "${AFTER_INFERENCE_HASH}"
printf '  protected site env: unchanged (%s)\n' "${AFTER_SITE_CONFIG_HASH}"
printf '  semantic backend:  SigLIP2 %s\n' "${SEMANTIC_CONFIG[2]:0:12}"
printf '  URL:               %s\n' "${BASE_URL}"
