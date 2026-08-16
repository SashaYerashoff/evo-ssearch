#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

APP_DIR="${EVA_APP_DIR:-/opt/eva-ai/evo-ssearch}"
ENV_FILE="${EVA_ENV_FILE:-/etc/eva-ai/eva-ai.env}"
SERVICE_NAME="${EVA_SERVICE_NAME:-eva-ai}"
BASE_URL="${EVA_BASE_URL:-http://127.0.0.1:5000}"
BACKUP_ROOT="${EVA_BACKUP_ROOT:-/var/backups/eva-ai}"
PG_DATABASE="${EVA_PG_DATABASE:-eva}"
START_SERVICE=true
RUN_VERIFY=true
RUN_MIGRATIONS="${EVA_PATCH_RUN_MIGRATIONS:-false}"
SKIP_PG_DUMP=false
BUNDLE_DIR=""
SOURCE_DIR=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() {
  printf '[INFO] %s\n' "$*"
}

ok() {
  printf 'OK: %s\n' "$*"
}

warn() {
  printf 'WARN: %s\n' "$*" >&2
}

fail() {
  printf 'FAIL: %s\n' "$*" >&2
}

die() {
  fail "$*"
  exit 1
}

usage() {
  cat <<'USAGE'
Usage: sudo scripts/install_patch.sh [options]

Options:
  --bundle-dir DIR       Unpacked bundle directory. Defaults to script parent.
  --source-dir DIR       Source repo directory. Defaults to BUNDLE_DIR/repo.
  --app-dir DIR          Target app directory. Default: /opt/eva-ai/evo-ssearch.
  --env-file FILE        Runtime env file. Default: /etc/eva-ai/eva-ai.env.
  --service NAME         systemd service name. Default: eva-ai.
  --base-url URL         Local app base URL for verification.
  --backup-root DIR      Backup root. Default: /var/backups/eva-ai.
  --pg-database NAME     Local PostgreSQL database fallback. Default: eva.
  --skip-pg-dump         Do not attempt PostgreSQL dump.
  --run-migrations       Run alembic upgrade head after code copy.
  --no-start             Do not start service after install.
  --no-verify            Do not call verify_patch.sh after install.
  -h, --help             Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle-dir)
      BUNDLE_DIR="$2"
      shift 2
      ;;
    --source-dir)
      SOURCE_DIR="$2"
      shift 2
      ;;
    --app-dir)
      APP_DIR="$2"
      shift 2
      ;;
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --service)
      SERVICE_NAME="$2"
      shift 2
      ;;
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --backup-root)
      BACKUP_ROOT="$2"
      shift 2
      ;;
    --pg-database)
      PG_DATABASE="$2"
      shift 2
      ;;
    --skip-pg-dump)
      SKIP_PG_DUMP=true
      shift
      ;;
    --run-migrations)
      RUN_MIGRATIONS=true
      shift
      ;;
    --no-start)
      START_SERVICE=false
      shift
      ;;
    --no-verify)
      RUN_VERIFY=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

if [[ "${EUID}" -ne 0 ]]; then
  die "Run this script with sudo/root so it can backup /etc, /opt, and systemd state."
fi

if [[ -z "${BUNDLE_DIR}" ]]; then
  if [[ -d "${SCRIPT_DIR}/../repo" ]]; then
    BUNDLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
  else
    BUNDLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
  fi
fi

if [[ -z "${SOURCE_DIR}" ]]; then
  if [[ -d "${BUNDLE_DIR}/repo" ]]; then
    SOURCE_DIR="${BUNDLE_DIR}/repo"
  else
    SOURCE_DIR="${BUNDLE_DIR}"
  fi
fi

[[ -d "${SOURCE_DIR}" ]] || die "Source directory not found: ${SOURCE_DIR}"
[[ -f "${SOURCE_DIR}/run_prod.sh" || -f "${SOURCE_DIR}/wsgi.py" ]] || die "Source directory does not look like evo-ssearch: ${SOURCE_DIR}"
[[ -d "${APP_DIR}" ]] || die "Target app directory not found: ${APP_DIR}"

read_env_var() {
  local key="$1"
  local file="$2"
  local line value
  [[ -f "${file}" ]] || return 0
  line="$(grep -E "^[[:space:]]*${key}=" "${file}" | tail -n 1 || true)"
  [[ -n "${line}" ]] || return 0
  value="${line#*=}"
  value="${value%$'\r'}"
  value="${value%\"}"
  value="${value#\"}"
  value="${value%\'}"
  value="${value#\'}"
  printf '%s' "${value}"
}

service_exists() {
  systemctl list-unit-files "${SERVICE_NAME}.service" --no-legend >/dev/null 2>&1 \
    || systemctl status "${SERVICE_NAME}.service" >/dev/null 2>&1
}

run_as_user() {
  local user="$1"
  shift
  if command -v runuser >/dev/null 2>&1; then
    runuser -u "${user}" -- "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo -u "${user}" "$@"
  else
    die "runuser or sudo is required to run commands as ${user}"
  fi
}

timestamp="$(date +%Y%m%d-%H%M%S)"
BACKUP_DIR="${BACKUP_ROOT}/patch-${timestamp}"
install -d -m 0700 "${BACKUP_DIR}"

{
  printf 'created_at=%s\n' "$(date -Is)"
  printf 'app_dir=%s\n' "${APP_DIR}"
  printf 'env_file=%s\n' "${ENV_FILE}"
  printf 'service_name=%s\n' "${SERVICE_NAME}"
  printf 'source_dir=%s\n' "${SOURCE_DIR}"
  printf 'base_url=%s\n' "${BASE_URL}"
} > "${BACKUP_DIR}/metadata.txt"

ok "created backup directory ${BACKUP_DIR}"

if [[ -f "${ENV_FILE}" ]]; then
  cp -a "${ENV_FILE}" "${BACKUP_DIR}/eva-ai.env"
  ok "backed up env file"
else
  warn "env file not found, skipping env backup: ${ENV_FILE}"
fi

UNIT_PATH="$(systemctl show -p FragmentPath --value "${SERVICE_NAME}.service" 2>/dev/null || true)"
if [[ -n "${UNIT_PATH}" && -f "${UNIT_PATH}" ]]; then
  cp -a "${UNIT_PATH}" "${BACKUP_DIR}/$(basename "${UNIT_PATH}")"
  printf '%s\n' "${UNIT_PATH}" > "${BACKUP_DIR}/systemd_unit_path.txt"
  ok "backed up systemd unit ${UNIT_PATH}"
else
  warn "systemd unit path not found for ${SERVICE_NAME}"
fi

DROPIN_DIR="/etc/systemd/system/${SERVICE_NAME}.service.d"
if [[ -d "${DROPIN_DIR}" ]]; then
  tar -czf "${BACKUP_DIR}/systemd-dropins.tgz" -C "$(dirname "${DROPIN_DIR}")" "$(basename "${DROPIN_DIR}")"
  ok "backed up systemd drop-ins"
fi

APP_PARENT="$(dirname "${APP_DIR}")"
APP_BASE="$(basename "${APP_DIR}")"
tar \
  --exclude="${APP_BASE}/.git" \
  --exclude="${APP_BASE}/*/.git" \
  --exclude="${APP_BASE}/.local" \
  --exclude="${APP_BASE}/*/.local" \
  --exclude="${APP_BASE}/.venv*" \
  --exclude="${APP_BASE}/*/.venv*" \
  --exclude="${APP_BASE}/.env" \
  --exclude="${APP_BASE}/.env.*" \
  --exclude="${APP_BASE}/dist" \
  --exclude="${APP_BASE}/__pycache__" \
  --exclude="${APP_BASE}/.pytest_cache" \
  --exclude="${APP_BASE}/node_modules" \
  --exclude="${APP_BASE}/*/node_modules" \
  --exclude="${APP_BASE}/detections_archive" \
  --exclude="${APP_BASE}/video" \
  --exclude="${APP_BASE}/models" \
  --exclude="${APP_BASE}/*.mp4" \
  --exclude="${APP_BASE}/*.avi" \
  --exclude="${APP_BASE}/*.mov" \
  --exclude="${APP_BASE}/*.mkv" \
  --exclude="${APP_BASE}/probes_store.json" \
  --exclude="${APP_BASE}/probe_channel_groups.json" \
  --exclude="${APP_BASE}/luxriot_summary_state.json" \
  --exclude="${APP_BASE}/luxriot_rollups_cache.json" \
  --exclude="${APP_BASE}/*.sqlite3" \
  --exclude="${APP_BASE}/*.db" \
  --exclude="${APP_BASE}/*.log" \
  -czf "${BACKUP_DIR}/code.tgz" \
  -C "${APP_PARENT}" "${APP_BASE}"
ok "backed up current code"
if [[ -f "${APP_DIR}/probe_channel_groups.json" ]]; then
  install -m 0600 "${APP_DIR}/probe_channel_groups.json" \
    "${BACKUP_DIR}/probe_channel_groups.json"
  ok "backed up probe channel groups"
fi

if [[ "${SKIP_PG_DUMP}" != true ]]; then
  if command -v pg_dump >/dev/null 2>&1; then
    PG_DSN="${EVA_PATCH_PG_DSN:-$(read_env_var EVA_DATABASE_DSN "${ENV_FILE}")}"
    if [[ -z "${PG_DSN}" ]]; then
      PG_DSN="$(read_env_var EVOSSEARCH_DATABASE_DSN "${ENV_FILE}")"
    fi

    if [[ -n "${PG_DSN}" ]]; then
      if pg_dump --dbname="${PG_DSN}" --format=custom --file="${BACKUP_DIR}/postgres.dump"; then
        ok "created PostgreSQL dump from env DSN"
        DB_REVISION="$(psql --no-psqlrc --tuples-only --no-align --dbname="${PG_DSN}" \
          --command='SELECT version_num FROM public.alembic_version LIMIT 1' 2>/dev/null \
          | head -n 1 || true)"
        if [[ "${DB_REVISION}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
          printf '%s\n' "${DB_REVISION}" > "${BACKUP_DIR}/database_revision.txt"
          ok "recorded pre-update database revision"
        else
          warn "could not record pre-update database revision"
        fi
      else
        warn "pg_dump via env DSN failed; continuing with code/env backup"
        rm -f "${BACKUP_DIR}/postgres.dump"
      fi
    elif id postgres >/dev/null 2>&1; then
      LOCAL_DUMP="/tmp/eva-ai-${timestamp}-postgres.dump"
      rm -f "${LOCAL_DUMP}"
      if run_as_user postgres pg_dump --format=custom --file="${LOCAL_DUMP}" "${PG_DATABASE}"; then
        install -m 0600 "${LOCAL_DUMP}" "${BACKUP_DIR}/postgres.dump"
        rm -f "${LOCAL_DUMP}"
        ok "created PostgreSQL dump for local database ${PG_DATABASE}"
        DB_REVISION="$(run_as_user postgres psql --no-psqlrc --tuples-only --no-align \
          --dbname="${PG_DATABASE}" \
          --command='SELECT version_num FROM public.alembic_version LIMIT 1' 2>/dev/null \
          | head -n 1 || true)"
        if [[ "${DB_REVISION}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
          printf '%s\n' "${DB_REVISION}" > "${BACKUP_DIR}/database_revision.txt"
          ok "recorded pre-update database revision"
        else
          warn "could not record pre-update database revision"
        fi
      else
        warn "local pg_dump failed; continuing with code/env backup"
        rm -f "${LOCAL_DUMP}"
        rm -f "${BACKUP_DIR}/postgres.dump"
      fi
    else
      warn "no DSN and no postgres OS user; skipping PostgreSQL dump"
    fi
  else
    warn "pg_dump not found; skipping PostgreSQL dump"
  fi
else
  warn "PostgreSQL dump skipped by operator request"
fi

printf '%s\n' "${BACKUP_DIR}" > "${BACKUP_ROOT}/LATEST"

if service_exists; then
  log "Stopping ${SERVICE_NAME}"
  systemctl stop "${SERVICE_NAME}.service"
  ok "stopped ${SERVICE_NAME}"
else
  warn "service ${SERVICE_NAME} not found; code will be copied without stop/start"
  START_SERVICE=false
fi

# Runtime policy is client data, not application code.  In particular the
# channel alert criteria live in the Luxriot summary state on older installs
# (newer PostgreSQL-backed installs are already covered by pg_dump).  Snapshot
# configured external files after the service has stopped so a rollback cannot
# silently preserve code while losing the operator's VLM alert policy.
RUNTIME_STATE_DIR="${BACKUP_DIR}/runtime-state"
RUNTIME_STATE_MANIFEST="${BACKUP_DIR}/runtime-state.tsv"
install -d -m 0700 "${RUNTIME_STATE_DIR}"
: > "${RUNTIME_STATE_MANIFEST}"

backup_runtime_state_file() {
  local label="$1"
  local configured="$2"
  local fallback="$3"
  local source_path="${configured}"
  if [[ -z "${source_path}" ]]; then
    source_path="${APP_DIR}/${fallback}"
  elif [[ "${source_path}" != /* ]]; then
    source_path="${APP_DIR}/${source_path}"
  fi
  [[ "${source_path}" = /* && "${source_path}" != "/" ]] \
    || die "Refusing unsafe runtime-state path for ${label}: ${source_path}"
  if [[ -f "${source_path}" ]]; then
    install -m 0600 "${source_path}" "${RUNTIME_STATE_DIR}/${label}"
    printf '%s\t%s\n' "${label}" "${source_path}" >> "${RUNTIME_STATE_MANIFEST}"
    ok "backed up runtime state ${label} from ${source_path}"
  fi
}

backup_runtime_state_file \
  "luxriot_summary_state.json" \
  "$(read_env_var EVOSSEARCH_LUXRIOT_SUMMARY_STATE_FILE "${ENV_FILE}")" \
  "luxriot_summary_state.json"
backup_runtime_state_file \
  "luxriot_rollups_cache.json" \
  "$(read_env_var EVOSSEARCH_LUXRIOT_ROLLUP_CACHE_FILE "${ENV_FILE}")" \
  "luxriot_rollups_cache.json"
backup_runtime_state_file \
  "probe_channel_groups.json" \
  "$(read_env_var EVOSSEARCH_PROBE_CHANNEL_GROUPS_FILE "${ENV_FILE}")" \
  "probe_channel_groups.json"
backup_runtime_state_file \
  "probes_store.json" \
  "" \
  "probes_store.json"

if [[ ! -s "${RUNTIME_STATE_MANIFEST}" ]]; then
  rm -f "${RUNTIME_STATE_MANIFEST}"
  rmdir "${RUNTIME_STATE_DIR}" 2>/dev/null || true
  warn "no file-backed runtime policy found; PostgreSQL backup remains authoritative"
fi

APP_OWNER="${EVA_APP_OWNER:-$(stat -c '%U:%G' "${APP_DIR}")}"
log "Copying patch files from ${SOURCE_DIR} to ${APP_DIR}"

RSYNC_EXCLUDES=(
  "--exclude=.git"
  "--exclude=.venv"
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
)

if command -v rsync >/dev/null 2>&1; then
  rsync -a "${RSYNC_EXCLUDES[@]}" "${SOURCE_DIR}/" "${APP_DIR}/"
else
  tar "${RSYNC_EXCLUDES[@]}" -cf - -C "${SOURCE_DIR}" . | tar -xf - -C "${APP_DIR}"
fi

REACT_BUILD_SOURCE="${SOURCE_DIR}/react-ui/dist"
REACT_BUILD_TARGET="${APP_DIR}/react-ui/dist"
[[ -f "${REACT_BUILD_SOURCE}/index.html" ]] \
  || die "React production build is missing: ${REACT_BUILD_SOURCE}/index.html"
rm -rf -- "${REACT_BUILD_TARGET}"
mkdir -p "${REACT_BUILD_TARGET}"
cp -a "${REACT_BUILD_SOURCE}/." "${REACT_BUILD_TARGET}/"
ok "installed React production build"

find "${APP_DIR}" \
  \( \
    -path "${APP_DIR}/.venv" \
    -o -path "${APP_DIR}/node_modules" \
    -o -path "${APP_DIR}/detections_archive" \
    -o -path "${APP_DIR}/video" \
    -o -path "${APP_DIR}/models" \
  \) -prune \
  -o -exec chown "${APP_OWNER}" {} +
ok "copied patch files"

SIGLIP2_SOURCE="${BUNDLE_DIR}/models/huggingface/models--google--siglip2-base-patch16-224"
if [[ -d "${SIGLIP2_SOURCE}/blobs" && -d "${SIGLIP2_SOURCE}/snapshots" ]]; then
  SIGLIP2_CHECKSUM_ROOT="${BUNDLE_DIR}/models/huggingface"
  [[ -s "${SIGLIP2_CHECKSUM_ROOT}/SHA256SUMS" ]] \
    || die "offline SigLIP2 checksum manifest is missing"
  (
    cd "${SIGLIP2_CHECKSUM_ROOT}"
    sha256sum -c SHA256SUMS >/dev/null
  ) || die "offline SigLIP2 checksum verification failed"
  MODEL_CACHE_DIR="$(read_env_var EVOSSEARCH_MODEL_CACHE_DIR "${ENV_FILE}")"
  if [[ -z "${MODEL_CACHE_DIR}" ]]; then
    MODEL_CACHE_DIR="/var/lib/eva-ai/models/huggingface"
  elif [[ "${MODEL_CACHE_DIR}" != /* ]]; then
    die "EVOSSEARCH_MODEL_CACHE_DIR must be absolute for offline model install"
  fi
  SIGLIP2_TARGET="${MODEL_CACHE_DIR}/models--google--siglip2-base-patch16-224"
  install -d -m 0750 -o "${APP_OWNER%%:*}" -g "${APP_OWNER##*:}" "${MODEL_CACHE_DIR}" "${SIGLIP2_TARGET}"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "${SIGLIP2_SOURCE}/" "${SIGLIP2_TARGET}/"
  else
    tar -cf - -C "${SIGLIP2_SOURCE}" . | tar -xf - -C "${SIGLIP2_TARGET}"
  fi
  chown -R "${APP_OWNER}" "${SIGLIP2_TARGET}"
  ok "installed offline SigLIP2 cache at ${SIGLIP2_TARGET}"
fi

if [[ -d "${BUNDLE_DIR}/wheelhouse" && -x "${APP_DIR}/.venv/bin/pip" ]]; then
  log "Installing offline wheels from bundle wheelhouse"
  run_as_user "${APP_OWNER%%:*}" bash -lc \
    "cd '${APP_DIR}' && .venv/bin/pip install --no-index --find-links '${BUNDLE_DIR}/wheelhouse' -r requirements.txt"
  if [[ -f "${APP_DIR}/requirements-db.txt" ]]; then
    run_as_user "${APP_OWNER%%:*}" bash -lc \
      "cd '${APP_DIR}' && .venv/bin/pip install --no-index --find-links '${BUNDLE_DIR}/wheelhouse' -r requirements-db.txt"
  fi
  ok "offline dependency install completed"
else
  warn "no wheelhouse found or .venv missing; dependency install skipped"
fi

if [[ "${RUN_MIGRATIONS}" == true ]]; then
  if [[ -x "${APP_DIR}/.venv/bin/alembic" && -f "${APP_DIR}/alembic.ini" ]]; then
    log "Running alembic upgrade head"
    run_as_user "${APP_OWNER%%:*}" bash -lc \
      "set -a; source '${ENV_FILE}'; set +a; cd '${APP_DIR}' && .venv/bin/alembic upgrade head"
    ok "migrations completed"
  else
    die "migrations requested, but alembic is not available in ${APP_DIR}/.venv"
  fi
fi

if [[ "${START_SERVICE}" == true ]]; then
  systemctl daemon-reload
  log "Starting ${SERVICE_NAME}"
  systemctl start "${SERVICE_NAME}.service"
  sleep "${EVA_PATCH_START_WAIT_SECONDS:-10}"
  ok "started ${SERVICE_NAME}"
fi

if [[ "${RUN_VERIFY}" == true ]]; then
  VERIFY_SCRIPT="${SCRIPT_DIR}/verify_patch.sh"
  if [[ ! -x "${VERIFY_SCRIPT}" && -x "${APP_DIR}/scripts/verify_patch.sh" ]]; then
    VERIFY_SCRIPT="${APP_DIR}/scripts/verify_patch.sh"
  fi
  [[ -x "${VERIFY_SCRIPT}" ]] || die "verify script not found"
  "${VERIFY_SCRIPT}" --service "${SERVICE_NAME}" --base-url "${BASE_URL}" --timeout 300
fi

ok "install completed"
log "Backup directory: ${BACKUP_DIR}"
