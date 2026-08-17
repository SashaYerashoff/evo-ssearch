#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${EVA_APP_DIR:-/opt/eva-ai/evo-ssearch}"
ENV_FILE="${EVA_ENV_FILE:-/etc/eva-ai/eva-ai.env}"
SERVICE_NAME="${EVA_SERVICE_NAME:-eva-ai}"
BASE_URL="${EVA_BASE_URL:-http://127.0.0.1:5000}"
BACKUP_ROOT="${EVA_BACKUP_ROOT:-/var/backups/eva-ai}"
PG_DATABASE="${EVA_PG_DATABASE:-eva}"
EXPECTED_VERSION="${EVA_EXPECTED_VERSION:-}"
EXPECTED_SCHEMA="${EVA_EXPECTED_SCHEMA:-20260805_0013}"
BUNDLE_DIR=""
STRICT=false
CHECK_SERVICE=true
USER_SERVICE=false
CURL_INSECURE=true
MIN_BACKUP_FREE_MB="${EVA_PREFLIGHT_MIN_BACKUP_FREE_MB:-20480}"

OK_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0

ok() {
  OK_COUNT=$((OK_COUNT + 1))
  printf 'OK: %s\n' "$*"
}

warn() {
  WARN_COUNT=$((WARN_COUNT + 1))
  printf 'WARN: %s\n' "$*" >&2
}

fail() {
  FAIL_COUNT=$((FAIL_COUNT + 1))
  printf 'FAIL: %s\n' "$*" >&2
}

usage() {
  cat <<'USAGE'
Usage: scripts/preflight_patch.sh [options]

Safe preflight for an offline EVA AI patch. It does not stop services, copy code,
run migrations, or edit configuration.

Options:
  --bundle-dir DIR        Unpacked patch bundle directory.
  --app-dir DIR           Target app directory. Default: /opt/eva-ai/evo-ssearch.
  --env-file FILE         Runtime env file. Default: /etc/eva-ai/eva-ai.env.
  --service NAME          systemd service name. Default: eva-ai.
  --base-url URL          Local app base URL. Default: http://127.0.0.1:5000.
  --backup-root DIR       Backup root to check. Default: /var/backups/eva-ai.
  --pg-database NAME      Local PostgreSQL database fallback. Default: eva.
  --expected-version STR  Expected patch/app version, for example "β 0.8.7".
  --expected-schema REV   Expected Alembic revision. Default: 20260805_0013.
  --min-backup-free-mb N  Warn if backup filesystem has less free space.
  --user-service          Check a per-user systemd service with systemctl --user.
  --skip-service          Do not check systemd service state.
  --curl-insecure         Allow self-signed HTTPS checks. Enabled by default.
  --no-curl-insecure      Disable curl -k for HTTPS endpoint checks.
  --strict                Treat warnings as failures.
  -h, --help              Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle-dir)
      BUNDLE_DIR="$2"
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
    --expected-version)
      EXPECTED_VERSION="$2"
      shift 2
      ;;
    --expected-schema)
      EXPECTED_SCHEMA="$2"
      shift 2
      ;;
    --min-backup-free-mb)
      MIN_BACKUP_FREE_MB="$2"
      shift 2
      ;;
    --user-service)
      USER_SERVICE=true
      shift
      ;;
    --skip-service)
      CHECK_SERVICE=false
      shift
      ;;
    --curl-insecure)
      CURL_INSECURE=true
      shift
      ;;
    --no-curl-insecure)
      CURL_INSECURE=false
      shift
      ;;
    --strict)
      STRICT=true
      shift
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

run_as_user() {
  local user="$1"
  shift
  if command -v runuser >/dev/null 2>&1; then
    runuser -u "${user}" -- "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo -u "${user}" "$@"
  else
    return 127
  fi
}

run_pg_dsn() {
  local dsn="$1"
  shift
  [[ -f "${SCRIPT_DIR}/pg_with_dsn.py" ]] || return 127
  EVA_PG_CONNECT_DSN="${dsn}" python3 "${SCRIPT_DIR}/pg_with_dsn.py" -- "$@"
}

redact_value() {
  local key="$1"
  local value="$2"
  case "${key}" in
    *PASSWORD*|*SECRET*|*TOKEN*|*KEY*|*DSN*)
      if [[ -n "${value}" ]]; then
        printf '[set]'
      else
        printf '[empty]'
      fi
      ;;
    *)
      printf '%s' "${value:-[empty]}"
      ;;
  esac
}

print_header() {
  printf '\n== %s ==\n' "$1"
}

check_path() {
  local label="$1"
  local path="$2"
  if [[ -e "${path}" ]]; then
    ok "${label}: ${path}"
  else
    fail "${label} missing: ${path}"
  fi
}

check_command() {
  local cmd="$1"
  if command -v "${cmd}" >/dev/null 2>&1; then
    ok "command available: ${cmd}"
  else
    warn "command not found: ${cmd}"
  fi
}

json_value() {
  local file="$1"
  local query="$2"
  if command -v jq >/dev/null 2>&1; then
    jq -r "${query} // empty" "${file}" 2>/dev/null || true
  fi
}

print_header "Patch Bundle"
if [[ -n "${BUNDLE_DIR}" ]]; then
  if [[ -d "${BUNDLE_DIR}" ]]; then
    ok "bundle directory exists: ${BUNDLE_DIR}"
    if [[ -f "${BUNDLE_DIR}/manifest.txt" ]]; then
      ok "bundle manifest found"
      sed -n '1,20p' "${BUNDLE_DIR}/manifest.txt"
      bundle_version="$(grep -E '^version=' "${BUNDLE_DIR}/manifest.txt" | tail -n 1 | cut -d= -f2- || true)"
      if [[ -n "${EXPECTED_VERSION}" && -n "${bundle_version}" && "${bundle_version}" != "${EXPECTED_VERSION}" ]]; then
        warn "bundle version is ${bundle_version}, expected ${EXPECTED_VERSION}"
      fi
    else
      warn "bundle manifest not found"
    fi
    if [[ -d "${BUNDLE_DIR}/wheelhouse" ]]; then
      wheel_count="$(find "${BUNDLE_DIR}/wheelhouse" -type f \( -name '*.whl' -o -name '*.tar.gz' -o -name '*.zip' \) | wc -l | tr -d ' ')"
      ok "wheelhouse found with ${wheel_count} files"
    else
      warn "no wheelhouse in bundle; outer installer must prove the existing venv satisfies release dependencies"
    fi
  else
    fail "bundle directory not found: ${BUNDLE_DIR}"
  fi
else
  warn "bundle directory not supplied; patch manifest and wheelhouse not checked"
fi

print_header "Runtime Paths"
check_path "app directory" "${APP_DIR}"
check_path "env file" "${ENV_FILE}"
if [[ -d "${APP_DIR}" ]]; then
  if [[ -f "${APP_DIR}/VERSION" ]]; then
    current_version="$(tr -d '\r\n' < "${APP_DIR}/VERSION")"
    ok "current VERSION: ${current_version}"
  else
    warn "VERSION file missing in app dir"
  fi
  if [[ -x "${APP_DIR}/.venv/bin/python" ]]; then
    ok "venv python exists"
    "${APP_DIR}/.venv/bin/python" --version || true
  else
    warn "venv python missing: ${APP_DIR}/.venv/bin/python"
  fi
fi

print_header "Systemd"
SYSTEMCTL=(systemctl)
if [[ "${USER_SERVICE}" == true ]]; then
  SYSTEMCTL=(systemctl --user)
fi
if [[ "${CHECK_SERVICE}" != true ]]; then
  warn "systemd service check skipped"
elif command -v systemctl >/dev/null 2>&1; then
  if "${SYSTEMCTL[@]}" list-unit-files "${SERVICE_NAME}.service" --no-legend >/dev/null 2>&1 \
    || "${SYSTEMCTL[@]}" cat "${SERVICE_NAME}.service" >/dev/null 2>&1
  then
    if [[ "${USER_SERVICE}" == true ]]; then
      ok "user systemd service exists: ${SERVICE_NAME}.service"
    else
      ok "systemd service exists: ${SERVICE_NAME}.service"
    fi
    unit_path="$("${SYSTEMCTL[@]}" show -p FragmentPath --value "${SERVICE_NAME}.service" 2>/dev/null || true)"
    work_dir="$("${SYSTEMCTL[@]}" show -p WorkingDirectory --value "${SERVICE_NAME}.service" 2>/dev/null || true)"
    user_name="$("${SYSTEMCTL[@]}" show -p User --value "${SERVICE_NAME}.service" 2>/dev/null || true)"
    group_name="$("${SYSTEMCTL[@]}" show -p Group --value "${SERVICE_NAME}.service" 2>/dev/null || true)"
    printf 'unit_path=%s\nworking_directory=%s\nuser=%s\ngroup=%s\n' \
      "${unit_path:-unknown}" "${work_dir:-unknown}" "${user_name:-unknown}" "${group_name:-unknown}"
    if [[ -n "${work_dir}" && "${work_dir}" != "${APP_DIR}" ]]; then
      warn "systemd WorkingDirectory differs from app dir"
    fi
    if "${SYSTEMCTL[@]}" is-active --quiet "${SERVICE_NAME}.service"; then
      ok "service is active"
    else
      warn "service is not active before patch"
    fi
  else
    fail "systemd service not found: ${SERVICE_NAME}.service"
  fi
else
  warn "systemctl not available"
fi

print_header "Runtime Env"
if [[ -f "${ENV_FILE}" ]]; then
  for key in \
    EVOSSEARCH_APP_VERSION \
    EVOSSEARCH_HOST \
    EVOSSEARCH_PORT \
    EVOSSEARCH_GUNICORN_WORKERS \
    EVOSSEARCH_AUTH_COOKIE_SECURE \
    EVOSSEARCH_LUXRIOT_BASE_URL \
    EVOSSEARCH_LUXRIOT_USERNAME \
    EVOSSEARCH_LUXRIOT_PASSWORD \
    EVOSSEARCH_LM_PROFILES \
    EVOSSEARCH_LM_VLM_BALANCER_ENABLED \
    EVOSSEARCH_LM_VLM_BALANCER_PROFILES \
    EVA_DATABASE_DSN \
    EVOSSEARCH_DATABASE_DSN
  do
    value="$(read_env_var "${key}" "${ENV_FILE}")"
    if [[ -n "${value}" ]]; then
      printf '%s=%s\n' "${key}" "$(redact_value "${key}" "${value}")"
    fi
  done
  profile_ids="$(read_env_var EVOSSEARCH_LM_PROFILES "${ENV_FILE}")"
  for profile_id in ${profile_ids//,/ }; do
    [[ -n "${profile_id}" ]] || continue
    profile_env_id="$(printf '%s' "${profile_id}" | tr '[:lower:]' '[:upper:]' | tr -c 'A-Z0-9' '_')"
    for suffix in BASE_URL MODEL KIND MAX_INFLIGHT; do
      key="EVOSSEARCH_LM_PROFILE_${profile_env_id}_${suffix}"
      value="$(read_env_var "${key}" "${ENV_FILE}")"
      if [[ -n "${value}" ]]; then
        printf '%s=%s\n' "${key}" "$(redact_value "${key}" "${value}")"
      fi
    done
  done
  workers="$(read_env_var EVOSSEARCH_GUNICORN_WORKERS "${ENV_FILE}")"
  if [[ -n "${workers}" && "${workers}" != "1" ]]; then
    fail "EVOSSEARCH_GUNICORN_WORKERS must be 1 for current runtime; found ${workers}"
  else
    ok "gunicorn worker count is compatible"
  fi
  app_version_override="$(read_env_var EVOSSEARCH_APP_VERSION "${ENV_FILE}")"
  if [[ -n "${EXPECTED_VERSION}" && -n "${app_version_override}" && "${app_version_override}" != "${EXPECTED_VERSION}" ]]; then
    warn "EVOSSEARCH_APP_VERSION is ${app_version_override}; expected ${EXPECTED_VERSION}"
  fi
fi

print_header "Disk And Backup"
backup_parent="$(dirname "${BACKUP_ROOT}")"
if [[ -d "${backup_parent}" ]]; then
  ok "backup parent exists: ${backup_parent}"
  free_mb="$(df -Pm "${backup_parent}" | awk 'NR==2 {print $4}')"
  printf 'backup_parent_free_mb=%s\n' "${free_mb:-unknown}"
  if [[ "${free_mb:-0}" =~ ^[0-9]+$ ]] && (( free_mb < MIN_BACKUP_FREE_MB )); then
    warn "backup filesystem has less than ${MIN_BACKUP_FREE_MB} MB free"
  else
    ok "backup filesystem free-space threshold passed"
  fi
else
  warn "backup parent does not exist: ${backup_parent}"
fi

print_header "PostgreSQL"
check_command pg_dump
check_command pg_restore
check_command psql
PG_DSN="$(read_env_var EVA_DATABASE_DSN "${ENV_FILE}")"
if [[ -z "${PG_DSN}" ]]; then
  PG_DSN="$(read_env_var EVOSSEARCH_DATABASE_DSN "${ENV_FILE}")"
fi
if [[ -n "${PG_DSN}" ]]; then
  ok "database DSN present in env"
  if command -v psql >/dev/null 2>&1; then
    schema="$(run_pg_dsn "${PG_DSN}" psql -Atc "select version_num from alembic_version limit 1" 2>/dev/null || true)"
    db_size="$(run_pg_dsn "${PG_DSN}" psql -Atc "select pg_size_pretty(pg_database_size(current_database()))" 2>/dev/null || true)"
  fi
elif id postgres >/dev/null 2>&1; then
  ok "postgres OS user exists for local database fallback"
  if command -v psql >/dev/null 2>&1; then
    schema="$(run_as_user postgres psql -d "${PG_DATABASE}" -Atc "select version_num from alembic_version limit 1" 2>/dev/null || true)"
    db_size="$(run_as_user postgres psql -d "${PG_DATABASE}" -Atc "select pg_size_pretty(pg_database_size(current_database()))" 2>/dev/null || true)"
  fi
else
  warn "no database DSN and no postgres OS user; installer may skip DB dump"
fi
if [[ -n "${schema:-}" ]]; then
  if [[ "${schema}" == "${EXPECTED_SCHEMA}" ]]; then
    ok "database schema revision: ${schema}"
  else
    warn "database schema revision is ${schema}; expected ${EXPECTED_SCHEMA}"
  fi
else
  warn "database schema revision not readable"
fi
if [[ -n "${db_size:-}" ]]; then
  printf 'database_size=%s\n' "${db_size}"
fi

print_header "HTTP Health"
if command -v curl >/dev/null 2>&1; then
  health_body="$(mktemp)"
  ready_body="$(mktemp)"
  CURL_OPTS=(-sS --max-time 8)
  if [[ "${CURL_INSECURE}" == true ]]; then
    CURL_OPTS+=(-k)
  fi
  if curl "${CURL_OPTS[@]}" "${BASE_URL}/health" > "${health_body}" 2>/tmp/eva-preflight-curl.err; then
    ok "health reachable at ${BASE_URL}/health"
    health_version="$(json_value "${health_body}" '.version')"
    health_status="$(json_value "${health_body}" '.status')"
    [[ -n "${health_status}" ]] && printf 'health_status=%s\n' "${health_status}"
    [[ -n "${health_version}" ]] && printf 'health_version=%s\n' "${health_version}"
  else
    warn "health not reachable at ${BASE_URL}/health"
    [[ -s /tmp/eva-preflight-curl.err ]] && sed -n '1,3p' /tmp/eva-preflight-curl.err >&2
  fi
  if curl "${CURL_OPTS[@]}" "${BASE_URL}/ready" > "${ready_body}" 2>/tmp/eva-preflight-curl.err; then
    ok "ready reachable at ${BASE_URL}/ready"
    ready_status="$(json_value "${ready_body}" '.status')"
    [[ -n "${ready_status}" ]] && printf 'ready_status=%s\n' "${ready_status}"
    if [[ -n "${ready_status}" && "${ready_status}" != "ready" ]]; then
      warn "ready status is ${ready_status}; record whether this predates the patch"
    fi
  else
    warn "ready not reachable at ${BASE_URL}/ready"
    [[ -s /tmp/eva-preflight-curl.err ]] && sed -n '1,3p' /tmp/eva-preflight-curl.err >&2
  fi
  rm -f "${health_body}" "${ready_body}" /tmp/eva-preflight-curl.err
else
  warn "curl not available"
fi

print_header "Summary"
printf 'ok=%s warn=%s fail=%s strict=%s\n' "${OK_COUNT}" "${WARN_COUNT}" "${FAIL_COUNT}" "${STRICT}"

if [[ "${STRICT}" == true && "${WARN_COUNT}" -gt 0 ]]; then
  FAIL_COUNT=$((FAIL_COUNT + WARN_COUNT))
fi

if [[ "${FAIL_COUNT}" -gt 0 ]]; then
  printf 'Preflight result: FAIL\n' >&2
  exit 1
fi

printf 'Preflight result: OK_WITH_WARNINGS_OR_OK\n'
