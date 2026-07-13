#!/usr/bin/env bash
# One-command EVA AI 0.8.4 adopt upgrade from an unpacked offline bundle.

set -Eeuo pipefail
umask 077

EXPECTED_VERSION="β 0.8.4"
EXPECTED_SCHEMA="20260614_0006"
MODE="auto"
APP_DIR=""
ENV_FILE=""
SERVICE_NAME=""
BASE_URL=""
BACKUP_ROOT=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/manifest.txt" && -d "${SCRIPT_DIR}/repo" ]]; then
  BUNDLE_DIR="${SCRIPT_DIR}"
elif [[ -f "${SCRIPT_DIR}/../manifest.txt" && -d "${SCRIPT_DIR}/../repo" ]]; then
  BUNDLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  printf 'STOP: update.sh must be run from an unpacked EVA AI bundle.\n' >&2
  exit 1
fi
SOURCE_DIR="${BUNDLE_DIR}/repo"

usage() {
  cat <<'USAGE'
Usage: ./update.sh [options]

Normally no options are needed. The script detects a user or system service.

Options:
  --user                 Use systemctl --user.
  --system               Use the system service manager.
  --service NAME         Service name (with or without .service).
  --app-dir DIR          Existing EVA AI application directory.
  --env-file FILE        Existing EVA AI environment file.
  --base-url URL         Local health URL.
  --backup-root DIR      Backup directory root.
  -h, --help             Show this help.

Run this script directly, not with sudo. It requests sudo only when required.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user) MODE="user"; shift ;;
    --system) MODE="system"; shift ;;
    --service) SERVICE_NAME="${2%.service}"; shift 2 ;;
    --app-dir) APP_DIR="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --base-url) BASE_URL="${2%/}"; shift 2 ;;
    --backup-root) BACKUP_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'STOP: unknown option: %s\n' "$1" >&2; usage; exit 2 ;;
  esac
done

say() { printf '\n== %s\n' "$*"; }
ok() { printf 'OK: %s\n' "$*"; }
stop() { printf 'STOP: %s\n' "$*" >&2; exit 1; }

as_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  else
    command -v sudo >/dev/null 2>&1 || stop "sudo is required for the code backup/install step"
    sudo "$@"
  fi
}

for command_name in cmp curl grep sed systemctl tar; do
  command -v "${command_name}" >/dev/null 2>&1 || stop "required command is missing: ${command_name}"
done

if [[ "$(id -u)" -eq 0 && -n "${SUDO_USER:-}" && "${MODE}" != "system" ]]; then
  stop "run ./update.sh without sudo so user systemd can be detected"
fi

find_user_service() {
  local unit
  while read -r unit _rest; do
    [[ "${unit}" =~ ^eva-ai.*\.service$ ]] || continue
    if systemctl --user is-active --quiet "${unit}"; then
      printf '%s\n' "${unit%.service}"
      return 0
    fi
  done < <(systemctl --user list-units --type=service --all --no-legend 2>/dev/null || true)
  return 1
}

find_system_service() {
  local unit
  if systemctl is-active --quiet eva-ai.service 2>/dev/null; then
    printf 'eva-ai\n'
    return 0
  fi
  while read -r unit _rest; do
    [[ "${unit}" =~ ^eva-ai.*\.service$ ]] || continue
    if systemctl is-active --quiet "${unit}" 2>/dev/null; then
      printf '%s\n' "${unit%.service}"
      return 0
    fi
  done < <(systemctl list-units --type=service --all --no-legend 2>/dev/null || true)
  return 1
}

if [[ "${MODE}" == "auto" ]]; then
  if detected_service="$(find_user_service)"; then
    MODE="user"
    [[ -n "${SERVICE_NAME}" ]] || SERVICE_NAME="${detected_service}"
  elif detected_service="$(find_system_service)"; then
    MODE="system"
    [[ -n "${SERVICE_NAME}" ]] || SERVICE_NAME="${detected_service}"
  elif [[ -d /opt/eva-ai/evo-ssearch ]]; then
    MODE="system"
    [[ -n "${SERVICE_NAME}" ]] || SERVICE_NAME="eva-ai"
  else
    stop "could not detect an EVA AI service; pass --user/--system and --service"
  fi
fi

[[ -n "${SERVICE_NAME}" ]] || SERVICE_NAME="eva-ai"
SERVICE_NAME="${SERVICE_NAME%.service}"
if [[ "${MODE}" == "user" ]]; then
  SYSTEMCTL=(systemctl --user)
  SERVICE_USER="$(id -un)"
  SERVICE_GROUP="$(id -gn)"
else
  SYSTEMCTL=(systemctl)
  SERVICE_USER="$(${SYSTEMCTL[@]} show "${SERVICE_NAME}.service" -p User --value 2>/dev/null || true)"
  SERVICE_GROUP="$(${SYSTEMCTL[@]} show "${SERVICE_NAME}.service" -p Group --value 2>/dev/null || true)"
  SERVICE_USER="${SERVICE_USER:-eva}"
  SERVICE_GROUP="${SERVICE_GROUP:-${SERVICE_USER}}"
fi

if [[ -z "${BACKUP_ROOT}" ]]; then
  if [[ "${MODE}" == "user" ]]; then
    BACKUP_ROOT="${HOME}/.local/state/eva-ai/0.8.4-backups"
  else
    BACKUP_ROOT="/var/tmp/eva-ai-0.8.4-backups"
  fi
fi

systemctl_read() {
  if [[ "${MODE}" == "system" && "$(id -u)" -ne 0 ]]; then
    as_root "${SYSTEMCTL[@]}" "$@"
  else
    "${SYSTEMCTL[@]}" "$@"
  fi
}

systemctl_write() {
  if [[ "${MODE}" == "system" ]]; then
    as_root "${SYSTEMCTL[@]}" "$@"
  else
    "${SYSTEMCTL[@]}" "$@"
  fi
}

if [[ -z "${APP_DIR}" ]]; then
  APP_DIR="$(systemctl_read show "${SERVICE_NAME}.service" -p WorkingDirectory --value 2>/dev/null || true)"
  [[ -n "${APP_DIR}" ]] || APP_DIR="/opt/eva-ai/evo-ssearch"
fi
if [[ -z "${ENV_FILE}" ]]; then
  if [[ -f "${APP_DIR}/.env" ]]; then
    ENV_FILE="${APP_DIR}/.env"
  else
    ENV_FILE="/etc/eva-ai/eva-ai.env"
  fi
fi
UNIT_FILE="$(systemctl_read show "${SERVICE_NAME}.service" -p FragmentPath --value 2>/dev/null || true)"
if [[ -z "${UNIT_FILE}" ]]; then
  if [[ "${MODE}" == "user" ]]; then
    UNIT_FILE="${HOME}/.config/systemd/user/${SERVICE_NAME}.service"
  else
    UNIT_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
  fi
fi

read_env_value() {
  local key="$1"
  local reader=(sed -n "s/^[[:space:]]*${key}[[:space:]]*=[[:space:]]*//p" "${ENV_FILE}")
  local value
  if [[ -r "${ENV_FILE}" ]]; then
    value="$("${reader[@]}" | tail -n 1)"
  else
    value="$(as_root "${reader[@]}" | tail -n 1)"
  fi
  value="${value%$'\r'}"
  value="${value%\"}"; value="${value#\"}"
  value="${value%\'}"; value="${value#\'}"
  printf '%s' "${value}"
}

if [[ -z "${BASE_URL}" ]]; then
  PORT="$(read_env_value EVOSSEARCH_PORT)"
  if [[ -z "${PORT}" && "${SERVICE_NAME}" =~ -([0-9]+)$ ]]; then
    PORT="${BASH_REMATCH[1]}"
  fi
  PORT="${PORT:-5000}"
  for candidate in "https://127.0.0.1:${PORT}" "http://127.0.0.1:${PORT}"; do
    if curl -skfS --max-time 3 "${candidate}/ready" >/dev/null 2>&1; then
      BASE_URL="${candidate}"
      break
    fi
  done
  [[ -n "${BASE_URL}" ]] || {
    if [[ "${PORT}" == "5000" ]]; then
      BASE_URL="http://127.0.0.1:${PORT}"
    else
      BASE_URL="https://127.0.0.1:${PORT}"
    fi
  }
fi

say "EVA AI ${EXPECTED_VERSION} offline update"
printf 'Mode:       %s systemd\n' "${MODE}"
printf 'Service:    %s.service\n' "${SERVICE_NAME}"
printf 'Application: %s\n' "${APP_DIR}"
printf 'Config:      %s\n' "${ENV_FILE}"
printf 'Health URL:  %s\n' "${BASE_URL}"

[[ -d "${APP_DIR}" ]] || stop "application directory not found: ${APP_DIR}"
[[ -f "${ENV_FILE}" ]] || stop "environment file not found: ${ENV_FILE}"
[[ -x "${APP_DIR}/.venv/bin/python" ]] || stop "existing .venv is missing; adopt upgrade is not possible"
[[ -f "${SOURCE_DIR}/VERSION" ]] || stop "bundle VERSION is missing"
[[ "$(tr -d '\r\n' < "${SOURCE_DIR}/VERSION")" == "${EXPECTED_VERSION}" ]] || stop "bundle VERSION is not ${EXPECTED_VERSION}"
MANIFEST_VERSION="$(sed -n 's/^version=//p' "${BUNDLE_DIR}/manifest.txt" | tail -n 1)"
MANIFEST_STATUS="$(sed -n 's/^working_tree_status=//p' "${BUNDLE_DIR}/manifest.txt" | tail -n 1)"
[[ "${MANIFEST_VERSION}" == "${EXPECTED_VERSION}" ]] || stop "manifest version is not ${EXPECTED_VERSION}"
[[ "${MANIFEST_STATUS}" == "clean" ]] || stop "bundle was built from a dirty working tree"

DEPLOYED_VERSION="$(tr -d '\r\n' < "${APP_DIR}/VERSION" 2>/dev/null || true)"
case "${DEPLOYED_VERSION}" in
  "β 0.8.0"|"β 0.8.1") ok "supported installed version: ${DEPLOYED_VERSION}" ;;
  "${EXPECTED_VERSION}") stop "${EXPECTED_VERSION} is already installed" ;;
  *) stop "unsupported installed version: ${DEPLOYED_VERSION:-missing}" ;;
esac

for requirements_file in requirements.txt requirements-db.txt; do
  [[ -f "${APP_DIR}/${requirements_file}" ]] || stop "installed ${requirements_file} is missing"
  cmp -s "${APP_DIR}/${requirements_file}" "${SOURCE_DIR}/${requirements_file}" \
    || stop "${requirements_file} changed; this bundle needs a reviewed wheelhouse"
done
if "${APP_DIR}/.venv/bin/python" -m pip --version >/dev/null 2>&1; then
  "${APP_DIR}/.venv/bin/python" -m pip check >/dev/null \
    || stop "existing .venv failed pip check"
elif command -v uv >/dev/null 2>&1; then
  uv pip check --python "${APP_DIR}/.venv/bin/python" >/dev/null \
    || stop "existing .venv failed uv pip check"
else
  stop "cannot verify .venv: neither pip nor uv is available"
fi
ok "dependencies are unchanged and existing .venv is healthy"

target_python() {
  if [[ "${MODE}" == "system" || ! -r "${ENV_FILE}" ]]; then
    as_root "${APP_DIR}/.venv/bin/python" "$@"
  else
    "${APP_DIR}/.venv/bin/python" "$@"
  fi
}

SCHEMA_VERSION="$(target_python - "${ENV_FILE}" <<'PY'
import os
import re
import sys

values = {}
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    for line in handle:
        match = re.match(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line.strip())
        if match:
            values[match.group(1)] = match.group(2).strip().strip('"').strip("'")
for _ in range(8):
    changed = False
    for key, value in tuple(values.items()):
        expanded = re.sub(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}",
            lambda match: values.get(match.group(1), os.environ.get(match.group(1), match.group(0))),
            value,
        )
        if expanded != value:
            values[key] = expanded
            changed = True
    if not changed:
        break
dsn = values.get("EVA_DATABASE_DSN") or values.get("EVOSSEARCH_DATABASE_DSN")
if not dsn:
    print("NO_DSN")
    raise SystemExit
try:
    import psycopg
    with psycopg.connect(dsn, connect_timeout=10, options="-c default_transaction_read_only=on") as conn:
        row = conn.execute("SELECT version_num FROM alembic_version").fetchone()
except Exception as exc:
    print(f"ERROR:{type(exc).__name__}")
    raise SystemExit
print(row[0] if row else "EMPTY")
PY
)"
[[ "${SCHEMA_VERSION}" == "${EXPECTED_SCHEMA}" ]] \
  || stop "database schema is ${SCHEMA_VERSION}; expected ${EXPECTED_SCHEMA}. No migration was attempted."
ok "database schema is already ${EXPECTED_SCHEMA}; database will not be changed"

DRY_RUN=(
  "${SOURCE_DIR}/scripts/install_eva_083.py"
  --dry-run --non-interactive --no-migrate --no-start --no-verify
  --source-dir "${SOURCE_DIR}"
  --bundle-dir "${BUNDLE_DIR}"
  --app-dir "${APP_DIR}"
  --env-file "${ENV_FILE}"
  --backup-root "${BACKUP_ROOT}"
  --service-name "${SERVICE_NAME}"
  --service-user "${SERVICE_USER}"
  --service-group "${SERVICE_GROUP}"
  --unit-file "${UNIT_FILE}"
  --base-url "${BASE_URL}"
)
if [[ "${MODE}" == "user" ]]; then
  say "Local rehearsal preflight"
  printf 'WARN: user-systemd dev mode skips the production credential-placeholder policy.\n'
  printf '      Version, bundle, venv, requirements and database schema checks passed.\n'
else
  say "Production installer dry-run"
  if [[ ! -r "${ENV_FILE}" ]]; then
    as_root "${DRY_RUN[@]}" || stop "installer dry-run failed"
  else
    "${DRY_RUN[@]}" || stop "installer dry-run failed"
  fi
fi

printf '\nType UPDATE to install %s (database and runtime data stay unchanged): ' "${EXPECTED_VERSION}"
read -r CONFIRMATION
[[ "${CONFIRMATION}" == "UPDATE" ]] || stop "confirmation not received; nothing was changed"

say "Stopping ${SERVICE_NAME}.service"
systemctl_write stop "${SERVICE_NAME}.service"

say "Backing up and installing code"
if [[ "${MODE}" == "user" ]]; then
  timestamp="$(date +%Y%m%d-%H%M%S)"
  BACKUP_DIR="${BACKUP_ROOT}/patch-${timestamp}"
  mkdir -p "${BACKUP_DIR}"
  cp -a "${ENV_FILE}" "${BACKUP_DIR}/eva-ai.env"
  APP_PARENT="$(dirname "${APP_DIR}")"
  APP_BASE="$(basename "${APP_DIR}")"
  tar \
    --exclude="${APP_BASE}/.git" \
    --exclude="${APP_BASE}/.local" \
    --exclude="${APP_BASE}/.venv" \
    --exclude="${APP_BASE}/dist" \
    --exclude="${APP_BASE}/__pycache__" \
    --exclude="${APP_BASE}/.pytest_cache" \
    --exclude="${APP_BASE}/node_modules" \
    --exclude="${APP_BASE}/detections_archive" \
    --exclude="${APP_BASE}/video" \
    --exclude="${APP_BASE}/models" \
    --exclude="${APP_BASE}/*.mp4" \
    --exclude="${APP_BASE}/*.avi" \
    --exclude="${APP_BASE}/*.mov" \
    --exclude="${APP_BASE}/*.mkv" \
    --exclude="${APP_BASE}/probes_store.json" \
    --exclude="${APP_BASE}/luxriot_summary_state.json" \
    --exclude="${APP_BASE}/luxriot_rollups_cache.json" \
    -czf "${BACKUP_DIR}/code.tgz" \
    -C "${APP_PARENT}" "${APP_BASE}"
  printf '%s\n' "${BACKUP_DIR}" > "${BACKUP_ROOT}/LATEST"

  COPY_EXCLUDES=(
    --exclude=.git --exclude=.local --exclude=.venv
    --exclude=__pycache__ --exclude='*.pyc' --exclude=.pytest_cache
    --exclude=dist --exclude=node_modules --exclude=detections_archive
    --exclude=video --exclude=models --exclude='*.mp4' --exclude='*.avi'
    --exclude='*.mov' --exclude='*.mkv' --exclude=probes_store.json
    --exclude=luxriot_summary_state.json --exclude=luxriot_rollups_cache.json
    --exclude=.env --exclude='.env.*' --exclude='*.sqlite3' --exclude='*.db'
    --exclude='*.log'
  )
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "${COPY_EXCLUDES[@]}" "${SOURCE_DIR}/" "${APP_DIR}/"
  else
    tar "${COPY_EXCLUDES[@]}" -cf - -C "${SOURCE_DIR}" . | tar -xf - -C "${APP_DIR}"
  fi
  ok "local code backup: ${BACKUP_DIR}"
else
  as_root "${BUNDLE_DIR}/scripts/install_patch.sh" \
    --bundle-dir "${BUNDLE_DIR}" \
    --source-dir "${SOURCE_DIR}" \
    --app-dir "${APP_DIR}" \
    --env-file "${ENV_FILE}" \
    --service "${SERVICE_NAME}" \
    --base-url "${BASE_URL}" \
    --backup-root "${BACKUP_ROOT}" \
    --skip-pg-dump --no-start --no-verify
fi

target_python - "${ENV_FILE}" "${EXPECTED_VERSION}" <<'PY'
import os
import re
import stat
import sys
import tempfile
from pathlib import Path

path = Path(sys.argv[1])
version = sys.argv[2]
original = path.read_text(encoding="utf-8")
replacement = f'EVOSSEARCH_APP_VERSION="{version}"'
updated, count = re.subn(
    r"(?m)^[ \t]*EVOSSEARCH_APP_VERSION[ \t]*=.*$",
    replacement,
    original,
)
if count == 0:
    updated = original.rstrip("\n") + "\n" + replacement + "\n"
st = path.stat()
fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(updated)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temp_name, stat.S_IMODE(st.st_mode))
    if os.geteuid() == 0:
        os.chown(temp_name, st.st_uid, st.st_gid)
    os.replace(temp_name, path)
finally:
    try:
        os.unlink(temp_name)
    except FileNotFoundError:
        pass
PY
ok "code installed; database and runtime data were not changed"

printf '\nRestart %s.service now? [Y/n]: ' "${SERVICE_NAME}"
read -r RESTART_ANSWER
case "${RESTART_ANSWER}" in
  ""|y|Y|yes|YES) ;;
  *)
    printf '\nUpdate installed. Service remains stopped.\n'
    if [[ "${MODE}" == "user" ]]; then
      printf 'Start it with: systemctl --user start %s.service\n' "${SERVICE_NAME}"
    else
      printf 'Start it with: sudo systemctl start %s.service\n' "${SERVICE_NAME}"
    fi
    exit 0
    ;;
esac

say "Starting and checking EVA AI"
systemctl_write start "${SERVICE_NAME}.service"
READY_JSON=""
for _attempt in {1..18}; do
  if READY_JSON="$(curl -skfS --max-time 5 "${BASE_URL}/ready" 2>/dev/null)"; then
    if printf '%s' "${READY_JSON}" | grep -Eq '"status"[[:space:]]*:[[:space:]]*"ready"' \
       && printf '%s' "${READY_JSON}" | grep -Fq "${EXPECTED_VERSION}"; then
      break
    fi
  fi
  READY_JSON=""
  sleep 5
done

SERVICE_STATE="$(systemctl_read is-active "${SERVICE_NAME}.service" 2>/dev/null || true)"
[[ "${SERVICE_STATE}" == "active" ]] || stop "service state is ${SERVICE_STATE:-unknown}; backup root: ${BACKUP_ROOT}"
[[ -n "${READY_JSON}" ]] || stop "service started but /ready did not report ${EXPECTED_VERSION}; backup root: ${BACKUP_ROOT}"
curl -skfS --max-time 5 "${BASE_URL}/health" >/dev/null \
  || stop "service is active but /health failed; backup root: ${BACKUP_ROOT}"

printf '\n============================================================\n'
printf 'OK: EVA AI %s is up and running\n' "${EXPECTED_VERSION}"
printf 'URL: %s\n' "${BASE_URL}"
printf 'Service: %s.service (%s systemd)\n' "${SERVICE_NAME}" "${MODE}"
if [[ "${MODE}" == "user" ]]; then
  LATEST_BACKUP="$(cat "${BACKUP_ROOT}/LATEST" 2>/dev/null || printf '%s' "${BACKUP_ROOT}")"
else
  LATEST_BACKUP="$(as_root cat "${BACKUP_ROOT}/LATEST" 2>/dev/null || printf '%s' "${BACKUP_ROOT}")"
fi
printf 'Backup: %s\n' "${LATEST_BACKUP}"
printf '============================================================\n'
