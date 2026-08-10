#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

APP_DIR="${EVA_APP_DIR:-/opt/eva-ai/evo-ssearch}"
ENV_FILE="${EVA_ENV_FILE:-/etc/eva-ai/eva-ai.env}"
SERVICE_NAME="${EVA_SERVICE_NAME:-eva-ai}"
BASE_URL="${EVA_BASE_URL:-http://127.0.0.1:5000}"
BACKUP_ROOT="${EVA_BACKUP_ROOT:-/var/backups/eva-ai}"
PG_DATABASE="${EVA_PG_DATABASE:-eva}"
BACKUP_DIR=""
RESTORE_DB=false
RUN_VERIFY=true
START_SERVICE=true
MODE="system"

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
Usage: sudo scripts/rollback.sh [options]

Options:
  --backup-dir DIR    Backup directory. Default: value from /var/backups/eva-ai/LATEST.
  --backup-root DIR   Backup root containing LATEST.
  --user              Use the current user's systemd manager (rehearsal mode).
  --app-dir DIR       Target app directory. Default: /opt/eva-ai/evo-ssearch.
  --env-file FILE     Runtime env file. Default: /etc/eva-ai/eva-ai.env.
  --service NAME      systemd service name. Default: eva-ai.
  --base-url URL      Local app base URL for verification.
  --pg-database NAME  Local PostgreSQL database fallback. Default: eva.
  --restore-db        Restore postgres.dump. Requires EVA_PATCH_CONFIRM_DB_RESTORE=yes.
  --no-start          Do not start service after rollback.
  --no-verify         Do not call verify_patch.sh after rollback.
  -h, --help          Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backup-dir)
      BACKUP_DIR="$2"
      shift 2
      ;;
    --backup-root)
      BACKUP_ROOT="$2"
      shift 2
      ;;
    --user)
      MODE="user"
      shift
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
    --pg-database)
      PG_DATABASE="$2"
      shift 2
      ;;
    --restore-db)
      RESTORE_DB=true
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

if [[ "${MODE}" == "system" && "${EUID}" -ne 0 ]]; then
  die "Run this script with sudo/root so it can restore /etc, /opt, and systemd state."
fi

if [[ "${MODE}" == "user" ]]; then
  SYSTEMCTL=(systemctl --user)
else
  SYSTEMCTL=(systemctl)
fi

if [[ -z "${BACKUP_DIR}" ]]; then
  if [[ -f "${BACKUP_ROOT}/LATEST" ]]; then
    BACKUP_DIR="$(cat "${BACKUP_ROOT}/LATEST")"
  else
    die "Backup directory was not specified and ${BACKUP_ROOT}/LATEST does not exist"
  fi
fi

[[ -d "${BACKUP_DIR}" ]] || die "Backup directory not found: ${BACKUP_DIR}"
[[ -f "${BACKUP_DIR}/code.tgz" ]] || die "Backup code archive not found: ${BACKUP_DIR}/code.tgz"

read_state_var() {
  local key="$1"
  local state_file="${BACKUP_DIR}/offline-installer-state.txt"
  local line
  [[ -f "${state_file}" ]] || return 0
  line="$(grep -E "^${key}=" "${state_file}" | tail -n 1 || true)"
  [[ -n "${line}" ]] || return 0
  printf '%s' "${line#*=}"
}

UNIT_PREEXISTED="$(read_state_var unit_preexisted)"
ORIGINAL_UNIT_FILE="$(read_state_var unit_file)"

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
  "${SYSTEMCTL[@]}" list-unit-files "${SERVICE_NAME}.service" --no-legend >/dev/null 2>&1 \
    || "${SYSTEMCTL[@]}" status "${SERVICE_NAME}.service" >/dev/null 2>&1
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

if service_exists; then
  log "Stopping ${SERVICE_NAME}"
  "${SYSTEMCTL[@]}" stop "${SERVICE_NAME}.service" || true
  ok "stopped ${SERVICE_NAME}"
else
  warn "service ${SERVICE_NAME} not found"
  START_SERVICE=false
fi

if [[ -f "${BACKUP_DIR}/eva-ai.env" ]]; then
  install -d -m 0750 "$(dirname "${ENV_FILE}")"
  cp -a -- "${BACKUP_DIR}/eva-ai.env" "${ENV_FILE}"
  chmod 0600 "${ENV_FILE}"
  ok "restored env file"
else
  warn "env backup not found; env restore skipped"
fi

if [[ "${UNIT_PREEXISTED}" == "false" ]]; then
  if [[ -n "${ORIGINAL_UNIT_FILE}" ]]; then
    [[ "${ORIGINAL_UNIT_FILE}" = /* && "${ORIGINAL_UNIT_FILE}" != "/" ]] \
      || die "Refusing unsafe generated unit path from rollback state"
    [[ "$(basename "${ORIGINAL_UNIT_FILE}")" == "${SERVICE_NAME}.service" ]] \
      || die "Generated unit path does not match service ${SERVICE_NAME}"
    rm -f -- "${ORIGINAL_UNIT_FILE}"
    ok "removed service unit created by the failed installation"
  fi
  START_SERVICE=false
elif [[ -f "${BACKUP_DIR}/systemd_unit_path.txt" ]]; then
  UNIT_PATH="$(cat "${BACKUP_DIR}/systemd_unit_path.txt")"
  UNIT_BACKUP="${BACKUP_DIR}/$(basename "${UNIT_PATH}")"
  if [[ -f "${UNIT_BACKUP}" ]]; then
    install -D -m 0644 "${UNIT_BACKUP}" "${UNIT_PATH}"
    ok "restored systemd unit ${UNIT_PATH}"
  fi
fi

if [[ -f "${BACKUP_DIR}/systemd-dropins.tgz" ]]; then
  tar -xzf "${BACKUP_DIR}/systemd-dropins.tgz" -C /etc/systemd/system
  ok "restored systemd drop-ins"
fi

APP_PARENT="$(dirname "${APP_DIR}")"
APP_BASE="$(basename "${APP_DIR}")"
REPLACED_ARCHIVE="${APP_PARENT}/${APP_BASE}.rollback-current-code-$(date +%Y%m%d-%H%M%S).tgz"
if [[ -e "${APP_DIR}" ]]; then
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
    -czf "${REPLACED_ARCHIVE}" \
    -C "${APP_PARENT}" "${APP_BASE}"
  ok "captured current app code at ${REPLACED_ARCHIVE}"
fi
mkdir -p "${APP_PARENT}"
RESTORE_HELPER="${SCRIPT_DIR}/restore_code_snapshot.py"
[[ -f "${RESTORE_HELPER}" ]] || die "restore helper not found: ${RESTORE_HELPER}"
if [[ -x "${APP_DIR}/.venv/bin/python" ]]; then
  RESTORE_PYTHON="${APP_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  RESTORE_PYTHON="$(command -v python3)"
else
  die "Python is required for safe code rollback"
fi
"${RESTORE_PYTHON}" "${RESTORE_HELPER}" --archive "${BACKUP_DIR}/code.tgz" --app-dir "${APP_DIR}"
ok "restored exact code snapshot while preserving runtime data"

RUNTIME_STATE_MANIFEST="${BACKUP_DIR}/runtime-state.tsv"
RUNTIME_STATE_DIR="${BACKUP_DIR}/runtime-state"
if [[ -f "${RUNTIME_STATE_MANIFEST}" ]]; then
  while IFS=$'\t' read -r state_name state_path; do
    [[ -n "${state_name}" && -n "${state_path}" ]] || continue
    [[ "${state_path}" = /* && "${state_path}" != "/" ]] \
      || die "Refusing unsafe runtime-state restore path: ${state_path}"
    state_backup="${RUNTIME_STATE_DIR}/${state_name}"
    [[ -f "${state_backup}" ]] \
      || die "Runtime-state backup is missing: ${state_backup}"
    install -D -m 0600 "${state_backup}" "${state_path}"
    ok "restored runtime state ${state_name} to ${state_path}"
  done < "${RUNTIME_STATE_MANIFEST}"
fi

if [[ "${RESTORE_DB}" == true ]]; then
  [[ "${EVA_PATCH_CONFIRM_DB_RESTORE:-}" == "yes" ]] || die "Database restore requires EVA_PATCH_CONFIRM_DB_RESTORE=yes"
  [[ -f "${BACKUP_DIR}/postgres.dump" ]] || die "PostgreSQL dump not found in ${BACKUP_DIR}"
  if ! command -v pg_restore >/dev/null 2>&1; then
    die "pg_restore not found"
  fi
  if ! command -v psql >/dev/null 2>&1; then
    die "psql not found"
  fi

  PG_DSN="${EVA_PATCH_PG_DSN:-$(read_env_var EVA_DATABASE_DSN "${ENV_FILE}")}"
  if [[ -z "${PG_DSN}" ]]; then
    PG_DSN="$(read_env_var EVOSSEARCH_DATABASE_DSN "${ENV_FILE}")"
  fi

  if [[ -n "${PG_DSN}" ]]; then
    DB_REVISION="$(cat "${BACKUP_DIR}/database_revision.txt" 2>/dev/null || true)"
    CURRENT_DB_REVISION="$(psql --no-psqlrc --tuples-only --no-align --dbname="${PG_DSN}" \
      --command='SELECT version_num FROM public.alembic_version LIMIT 1' 2>/dev/null \
      | head -n 1 || true)"
    if [[ "${DB_REVISION}" =~ ^[A-Za-z0-9_.-]+$ \
       && "${CURRENT_DB_REVISION}" == "${DB_REVISION}" ]]; then
      ok "database is already at the recorded pre-update revision; dump restore skipped"
    else
    DB_SUPERUSER="$(psql --no-psqlrc --tuples-only --no-align --dbname="${PG_DSN}" \
      --command='SELECT rolsuper FROM pg_roles WHERE rolname = current_user' \
      | head -n 1)"
    if [[ "${DB_SUPERUSER}" == "t" ]]; then
      psql --no-psqlrc --set ON_ERROR_STOP=on --dbname="${PG_DSN}" --command="
        DROP SCHEMA IF EXISTS agent CASCADE;
        DROP SCHEMA IF EXISTS archive CASCADE;
        DROP SCHEMA IF EXISTS audit CASCADE;
        DROP SCHEMA IF EXISTS iam CASCADE;
        DROP SCHEMA IF EXISTS jobs CASCADE;
        DROP TABLE IF EXISTS public.alembic_version CASCADE;
      "
      pg_restore --exit-on-error --dbname="${PG_DSN}" "${BACKUP_DIR}/postgres.dump"
      ok "restored complete PostgreSQL dump with original ownership"
    else
      die "Database advanced beyond the recorded revision; exact dump restore requires a PostgreSQL superuser DSN"
    fi
    fi
  elif id postgres >/dev/null 2>&1; then
    run_as_user postgres psql --set ON_ERROR_STOP=on --dbname="${PG_DATABASE}" --command="
      DROP SCHEMA IF EXISTS agent CASCADE;
      DROP SCHEMA IF EXISTS archive CASCADE;
      DROP SCHEMA IF EXISTS audit CASCADE;
      DROP SCHEMA IF EXISTS iam CASCADE;
      DROP SCHEMA IF EXISTS jobs CASCADE;
      DROP TABLE IF EXISTS public.alembic_version CASCADE;
    "
    run_as_user postgres pg_restore --exit-on-error \
      --dbname="${PG_DATABASE}" "${BACKUP_DIR}/postgres.dump"
    ok "restored complete PostgreSQL dump to local database ${PG_DATABASE}"
  else
    die "No DSN and no postgres OS user available for database restore"
  fi
else
  warn "database restore not requested; PostgreSQL left unchanged"
fi

"${SYSTEMCTL[@]}" daemon-reload || true
if [[ "${START_SERVICE}" == true ]]; then
  log "Starting ${SERVICE_NAME}"
  "${SYSTEMCTL[@]}" start "${SERVICE_NAME}.service"
  sleep "${EVA_PATCH_START_WAIT_SECONDS:-10}"
  ok "started ${SERVICE_NAME}"
fi

if [[ "${RUN_VERIFY}" == true ]]; then
  VERIFY_SCRIPT="${SCRIPT_DIR}/verify_patch.sh"
  if [[ ! -x "${VERIFY_SCRIPT}" && -x "${APP_DIR}/scripts/verify_patch.sh" ]]; then
    VERIFY_SCRIPT="${APP_DIR}/scripts/verify_patch.sh"
  fi
  [[ -x "${VERIFY_SCRIPT}" ]] || die "verify script not found"
  "${VERIFY_SCRIPT}" --service "${SERVICE_NAME}" --base-url "${BASE_URL}" --timeout 60
fi

ok "rollback completed"
