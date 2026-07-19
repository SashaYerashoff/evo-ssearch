#!/usr/bin/env bash
# One-command EVA AI 0.8.4 adopt upgrade from an unpacked offline bundle.
#
# Model/server policy: the preflight only *describes* the configured LM
# topology (remote vLLM for VLM streams, local LM Studio/llama.cpp for the
# agent). Configuration that cannot be verified produces warnings, never a
# stop, and the updater never writes model or server settings.

set -Eeuo pipefail
umask 077

EXPECTED_VERSION="β 0.8.4"
EXPECTED_SCHEMA="20260614_0006"
MODE="auto"
APP_DIR=""
ENV_FILE=""
ENV_FILE_SOURCE=""
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
    --env-file) ENV_FILE="$2"; ENV_FILE_SOURCE="command line"; shift 2 ;;
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

for command_name in cmp curl env grep sed sha256sum systemctl tar; do
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

path_is_file() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    return 0
  fi
  if [[ "${MODE}" == "system" && "$(id -u)" -ne 0 ]]; then
    as_root test -f "${path}"
  else
    return 1
  fi
}

discover_systemd_env_file() {
  local raw candidate
  raw="$(systemctl_read show "${SERVICE_NAME}.service" -p EnvironmentFiles --value 2>/dev/null || true)"
  while IFS= read -r candidate; do
    candidate="${candidate#-}"
    candidate="${candidate#${candidate%%[![:space:]]*}}"
    candidate="${candidate%${candidate##*[![:space:]]}}"
    [[ -n "${candidate}" ]] || continue
    if path_is_file "${candidate}"; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done < <(
    printf '%s\n' "${raw}" \
      | sed -E 's/[[:space:]]+\(ignore_errors=(yes|no)\)/\n/g'
  )
  return 1
}

if [[ -z "${APP_DIR}" ]]; then
  APP_DIR="$(systemctl_read show "${SERVICE_NAME}.service" -p WorkingDirectory --value 2>/dev/null || true)"
  [[ -n "${APP_DIR}" ]] || APP_DIR="/opt/eva-ai/evo-ssearch"
fi
if [[ -z "${ENV_FILE}" ]]; then
  if discovered_env_file="$(discover_systemd_env_file)"; then
    ENV_FILE="${discovered_env_file}"
    ENV_FILE_SOURCE="systemd EnvironmentFiles"
  elif path_is_file "/etc/eva-ai/eva-ai.env"; then
    ENV_FILE="/etc/eva-ai/eva-ai.env"
    ENV_FILE_SOURCE="standard system config"
  elif path_is_file "${APP_DIR}/.env"; then
    ENV_FILE="${APP_DIR}/.env"
    ENV_FILE_SOURCE="application fallback"
  else
    ENV_FILE="/etc/eva-ai/eva-ai.env"
    ENV_FILE_SOURCE="unresolved fallback"
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

read_env_file_value() {
  local file="$1"
  local key="$2"
  local reader=(sed -n -E "s/^[[:space:]]*(export[[:space:]]+)?${key}[[:space:]]*=[[:space:]]*//p" "${file}")
  local value
  if [[ -r "${file}" ]]; then
    value="$("${reader[@]}" | tail -n 1)"
  elif path_is_file "${file}"; then
    value="$(as_root "${reader[@]}" | tail -n 1)"
  else
    value=""
  fi
  value="${value%$'\r'}"
  value="${value%\"}"; value="${value#\"}"
  value="${value%\'}"; value="${value#\'}"
  printf '%s' "${value}"
}

read_env_value() {
  read_env_file_value "${ENV_FILE}" "$1"
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
printf 'Config source: %s\n' "${ENV_FILE_SOURCE:-preselected}"
printf 'Health URL:  %s\n' "${BASE_URL}"

[[ -d "${APP_DIR}" ]] || stop "application directory not found: ${APP_DIR}"
path_is_file "${ENV_FILE}" || stop "environment file not found: ${ENV_FILE}"
[[ -x "${APP_DIR}/.venv/bin/python" ]] || stop "existing .venv is missing; adopt upgrade is not possible"
[[ -f "${SOURCE_DIR}/VERSION" ]] || stop "bundle VERSION is missing"
[[ "$(tr -d '\r\n' < "${SOURCE_DIR}/VERSION")" == "${EXPECTED_VERSION}" ]] || stop "bundle VERSION is not ${EXPECTED_VERSION}"
MANIFEST_VERSION="$(sed -n 's/^version=//p' "${BUNDLE_DIR}/manifest.txt" | tail -n 1)"
MANIFEST_STATUS="$(sed -n 's/^working_tree_status=//p' "${BUNDLE_DIR}/manifest.txt" | tail -n 1)"
BUNDLE_COMMIT="$(sed -n 's/^git_commit=//p' "${BUNDLE_DIR}/manifest.txt" | tail -n 1)"
MEDIA_RUNTIME="$(sed -n 's/^media_runtime=//p' "${BUNDLE_DIR}/manifest.txt" | tail -n 1)"
MEDIA_PLATFORM="$(sed -n 's/^media_runtime_platform=//p' "${BUNDLE_DIR}/manifest.txt" | tail -n 1)"
[[ "${MANIFEST_VERSION}" == "${EXPECTED_VERSION}" ]] || stop "manifest version is not ${EXPECTED_VERSION}"
[[ "${MANIFEST_STATUS}" == "clean" ]] || stop "bundle was built from a dirty working tree"
[[ "${BUNDLE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || stop "manifest git_commit is missing or invalid"
[[ "${MEDIA_RUNTIME}" == "included" ]] || stop "offline FFmpeg/OpenCV runtime is missing from this bundle"
[[ "${MEDIA_PLATFORM}" == "linux-x86_64" && "$(uname -m)" == "x86_64" ]] \
  || stop "media runtime requires Linux x86_64"
[[ -x "${BUNDLE_DIR}/runtime/ffmpeg/bin/ffmpeg" ]] || stop "bundled ffmpeg is missing"
[[ -x "${BUNDLE_DIR}/runtime/ffmpeg/bin/ffprobe" ]] || stop "bundled ffprobe is missing"
(
  cd "${BUNDLE_DIR}/runtime"
  sha256sum -c SHA256SUMS >/dev/null
) || stop "media runtime checksum verification failed"
"${BUNDLE_DIR}/runtime/ffmpeg/bin/ffmpeg" -v error \
  -f lavfi -i color=c=black:s=16x16:d=0.05 -frames:v 1 \
  -f image2pipe -vcodec mjpeg - >/dev/null \
  || stop "bundled ffmpeg failed the decode smoke test"
"${BUNDLE_DIR}/runtime/ffmpeg/bin/ffprobe" -version >/dev/null \
  || stop "bundled ffprobe failed to start"
ok "offline FFmpeg/ffprobe payload is intact and executable"

DEPLOYED_VERSION="$(tr -d '\r\n' < "${APP_DIR}/VERSION" 2>/dev/null || true)"
[[ -n "${DEPLOYED_VERSION}" ]] || stop "installed VERSION is missing; cannot create a verifiable rollback handoff"
INSTALLED_BUNDLE_COMMIT=""
if [[ -f "${APP_DIR}/.eva-bundle-commit" ]]; then
  INSTALLED_BUNDLE_COMMIT="$(tr -d '\r\n' < "${APP_DIR}/.eva-bundle-commit")"
fi
if [[ "${INSTALLED_BUNDLE_COMMIT}" == "${BUNDLE_COMMIT}" ]]; then
  stop "this exact ${EXPECTED_VERSION} bundle is already installed (${BUNDLE_COMMIT:0:7})"
fi
if [[ "${DEPLOYED_VERSION}" == "${EXPECTED_VERSION}" ]]; then
  INSTALLED_BUNDLE_LABEL="${INSTALLED_BUNDLE_COMMIT:0:7}"
  ok "same-version hotfix: ${INSTALLED_BUNDLE_LABEL:-unmarked} -> ${BUNDLE_COMMIT:0:7}"
else
  ok "adopt-upgrade candidate: ${DEPLOYED_VERSION} -> ${EXPECTED_VERSION}"
  printf '      Compatibility is determined by the exact requirements and read-only schema gates below.\n'
fi

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

remove_temp_path() {
  local path="$1"
  if [[ "${MODE}" == "system" && "$(id -u)" -ne 0 ]]; then
    as_root rm -rf -- "${path}"
  else
    rm -rf -- "${path}"
  fi
}

ready_json_matches_version() {
  local expected_version="$1"
  target_python -c '
import json
import sys

try:
    payload = json.load(sys.stdin)
except Exception:
    raise SystemExit(1)
raise SystemExit(
    0
    if payload.get("status") == "ready" and payload.get("version") == sys.argv[1]
    else 1
)
' "${expected_version}"
}

ready_json_reports_version() {
  local expected_version="$1"
  target_python -c '
import json
import sys

try:
    payload = json.load(sys.stdin)
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if payload.get("version") == sys.argv[1] else 1)
' "${expected_version}"
}

ACTIVE_READY_BODY="$(mktemp)"
if ! curl -skfS --max-time 8 "${BASE_URL}/ready" > "${ACTIVE_READY_BODY}" 2>/dev/null; then
  rm -f "${ACTIVE_READY_BODY}"
  stop "installed EVA /ready is unavailable; restore the existing service before upgrading"
fi
mapfile -t ACTIVE_RUNTIME_FACTS < <(target_python - "${ACTIVE_READY_BODY}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
checks = payload.get("checks") or {}
postgres = checks.get("postgresql") or {}
database = checks.get("database") or {}
profiles = (checks.get("lm_profiles") or {}).get("profiles") or []
agent = next((row for row in profiles if row.get("kind") == "agent" or row.get("id") == "agent"), {})
luxriot = checks.get("luxriot") or {}
db_gate_ok = bool(postgres.get("ok") and database.get("ok") and postgres.get("runtime_role_ok", True))
print(str(payload.get("status") or ""))
print(str(payload.get("version") or ""))
print(str(postgres.get("current_revision") or ""))
print("true" if db_gate_ok else "false")
print(str(agent.get("base_url") or ""))
print(str(luxriot.get("base_url") or ""))
print(str(agent.get("model") or agent.get("configured_model") or ""))
PY
)
rm -f "${ACTIVE_READY_BODY}"
ACTIVE_RUNTIME_STATUS="${ACTIVE_RUNTIME_FACTS[0]:-}"
ACTIVE_RUNTIME_VERSION="${ACTIVE_RUNTIME_FACTS[1]:-}"
ACTIVE_RUNTIME_SCHEMA="${ACTIVE_RUNTIME_FACTS[2]:-}"
ACTIVE_RUNTIME_DB_OK="${ACTIVE_RUNTIME_FACTS[3]:-false}"
ACTIVE_RUNTIME_AGENT_BASE_URL="${ACTIVE_RUNTIME_FACTS[4]:-}"
ACTIVE_RUNTIME_LUXRIOT_BASE_URL="${ACTIVE_RUNTIME_FACTS[5]:-}"
ACTIVE_RUNTIME_AGENT_MODEL="${ACTIVE_RUNTIME_FACTS[6]:-}"
ok "active runtime identity loaded from /ready (${ACTIVE_RUNTIME_VERSION:-unknown version})"
PREUPGRADE_DEGRADED=false
if [[ "${ACTIVE_RUNTIME_STATUS}" != "ready" ]]; then
  PREUPGRADE_DEGRADED=true
  printf 'WARN: installed EVA reports %s before upgrade (a dependency such as a remote VLM server may be offline).\n' \
    "${ACTIVE_RUNTIME_STATUS:-unknown}" >&2
  printf '      The update continues; post-update verification will accept the same pre-existing degraded dependencies.\n' >&2
fi
if [[ -n "${ACTIVE_RUNTIME_VERSION}" && "${ACTIVE_RUNTIME_VERSION}" != "${DEPLOYED_VERSION}" ]]; then
  ENV_VERSION_OVERRIDE="$(read_env_value EVOSSEARCH_APP_VERSION)"
  if [[ -n "${ENV_VERSION_OVERRIDE}" && "${ENV_VERSION_OVERRIDE}" == "${ACTIVE_RUNTIME_VERSION}" ]]; then
    ok "runtime reports ${ACTIVE_RUNTIME_VERSION} via the EVOSSEARCH_APP_VERSION override; code tree is ${DEPLOYED_VERSION}"
  else
    printf 'WARN: active service reports %s while %s/VERSION is %s; continuing (field builds may brand the runtime version differently).\n' \
      "${ACTIVE_RUNTIME_VERSION}" "${APP_DIR}" "${DEPLOYED_VERSION}" >&2
  fi
fi

EXPECTED_AGENT_CONTEXT=65536
CONFIGURED_AGENT_CONTEXT="$(read_env_value EVOSSEARCH_AGENT_CONTEXT_LIMIT_TOKENS)"
CONFIGURED_AGENT_CONTEXT="${CONFIGURED_AGENT_CONTEXT:-${EXPECTED_AGENT_CONTEXT}}"
CONFIGURED_AGENT_MODEL="$(read_env_value EVOSSEARCH_LM_PROFILE_AGENT_MODEL)"
[[ -n "${CONFIGURED_AGENT_MODEL}" ]] || CONFIGURED_AGENT_MODEL="$(read_env_value EVOSSEARCH_LM_MODEL)"
LEGACY_AGENT_MODEL=""
if [[ "${APP_DIR}/.env" != "${ENV_FILE}" ]] && path_is_file "${APP_DIR}/.env"; then
  LEGACY_AGENT_MODEL="$(read_env_file_value "${APP_DIR}/.env" EVOSSEARCH_LM_PROFILE_AGENT_MODEL)"
  [[ -n "${LEGACY_AGENT_MODEL}" ]] \
    || LEGACY_AGENT_MODEL="$(read_env_file_value "${APP_DIR}/.env" EVOSSEARCH_LM_MODEL)"
fi
SERVED_AGENT_MODELS=()
SERVED_AGENT_CONTEXT="UNKNOWN"
TEMPORARY_AGENT_CONTEXT=""
CONTEXT_FORCE_REQUIRED=false
CONTEXT_UNKNOWN_REQUIRED=false
CONTEXT_UNKNOWN_ACCEPTED=false
AGENT_LM_BASE_URL="$(read_env_value EVOSSEARCH_LM_PROFILE_AGENT_BASE_URL)"
[[ -n "${AGENT_LM_BASE_URL}" ]] || AGENT_LM_BASE_URL="$(read_env_value EVOSSEARCH_LM_BASE_URL)"
AGENT_LM_API_KEY="$(read_env_value EVOSSEARCH_LM_PROFILE_AGENT_API_KEY)"
[[ -n "${AGENT_LM_API_KEY}" ]] || AGENT_LM_API_KEY="$(read_env_value EVOSSEARCH_LM_API_KEY)"
LUXRIOT_BASE_URL="$(read_env_value EVOSSEARCH_LUXRIOT_BASE_URL)"
ENDPOINTS_UNDERSTOOD=true
if [[ -n "${ACTIVE_RUNTIME_AGENT_BASE_URL}" && -n "${AGENT_LM_BASE_URL}" \
  && "${ACTIVE_RUNTIME_AGENT_BASE_URL%/}" != "${AGENT_LM_BASE_URL%/}" ]]; then
  ENDPOINTS_UNDERSTOOD=false
  printf 'WARN: selected config does not match the active runtime agent profile (%s != %s).\n' \
    "${AGENT_LM_BASE_URL%/}" "${ACTIVE_RUNTIME_AGENT_BASE_URL%/}" >&2
  printf '      Continuing; this updater never rewrites model or server endpoints, so the running configuration stays authoritative.\n' >&2
fi
if [[ -n "${ACTIVE_RUNTIME_LUXRIOT_BASE_URL}" && -n "${LUXRIOT_BASE_URL}" \
  && "${ACTIVE_RUNTIME_LUXRIOT_BASE_URL%/}" != "${LUXRIOT_BASE_URL%/}" ]]; then
  ENDPOINTS_UNDERSTOOD=false
  printf 'WARN: selected config does not match the active Luxriot endpoint; continuing without changing it.\n' >&2
fi
if [[ "${ENDPOINTS_UNDERSTOOD}" == true ]]; then
  ok "selected config matches the active runtime endpoints"
fi
if [[ "${CONFIGURED_AGENT_CONTEXT}" =~ ^[0-9]+$ ]] && (( CONFIGURED_AGENT_CONTEXT < EXPECTED_AGENT_CONTEXT )); then
  CONTEXT_FORCE_REQUIRED=true
  TEMPORARY_AGENT_CONTEXT="${CONFIGURED_AGENT_CONTEXT}"
fi
if [[ -n "${AGENT_LM_BASE_URL}" ]]; then
  AGENT_MODELS_BODY="$(mktemp)"
  AGENT_MODELS_CURL=(curl -fsS --max-time 5)
  if [[ -n "${AGENT_LM_API_KEY}" ]]; then
    AGENT_MODELS_CURL+=(-H "Authorization: Bearer ${AGENT_LM_API_KEY}")
  fi
  AGENT_MODELS_CURL+=("${AGENT_LM_BASE_URL%/}/models")
  if "${AGENT_MODELS_CURL[@]}" > "${AGENT_MODELS_BODY}" 2>/dev/null; then
    mapfile -t AGENT_SERVER_FACTS < <(target_python - "${AGENT_MODELS_BODY}" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = [row for row in (payload.get("data") or []) if isinstance(row, dict)]
    row = rows[0] if rows else {}
    value = row.get("max_model_len") or (row.get("meta") or {}).get("n_ctx")
    print(int(value) if value is not None else "UNKNOWN")
    seen = set()
    for item in rows:
        model_id = str(item.get("id") or item.get("model") or "").strip()
        if model_id and model_id not in seen:
            seen.add(model_id)
            print(model_id)
except Exception:
    print("UNKNOWN")
PY
)
    SERVED_AGENT_CONTEXT="${AGENT_SERVER_FACTS[0]:-UNKNOWN}"
    if (( ${#AGENT_SERVER_FACTS[@]} > 1 )); then
      SERVED_AGENT_MODELS=("${AGENT_SERVER_FACTS[@]:1}")
    fi
    if [[ "${SERVED_AGENT_CONTEXT}" =~ ^[0-9]+$ ]] && (( SERVED_AGENT_CONTEXT < EXPECTED_AGENT_CONTEXT )); then
      CONTEXT_FORCE_REQUIRED=true
      if [[ -z "${TEMPORARY_AGENT_CONTEXT}" ]] \
        || (( SERVED_AGENT_CONTEXT < TEMPORARY_AGENT_CONTEXT )); then
        TEMPORARY_AGENT_CONTEXT="${SERVED_AGENT_CONTEXT}"
      fi
    elif [[ "${SERVED_AGENT_CONTEXT}" =~ ^[0-9]+$ ]]; then
      ok "agent inference context is ${SERVED_AGENT_CONTEXT} (required: ${EXPECTED_AGENT_CONTEXT})"
    else
      CONTEXT_UNKNOWN_REQUIRED=true
      printf 'WARN: agent inference server did not report n_ctx/max_model_len; verify it is at least %s.\n' \
        "${EXPECTED_AGENT_CONTEXT}" >&2
    fi
  else
    CONTEXT_UNKNOWN_REQUIRED=true
    printf 'WARN: could not read agent inference context from %s/models.\n' "${AGENT_LM_BASE_URL%/}" >&2
  fi
  rm -f "${AGENT_MODELS_BODY}"
fi

say "Model/server configuration preflight (read-only)"
printf 'This updater never writes model or server settings; every finding below is informational.\n'
if [[ -z "${CONFIGURED_AGENT_MODEL}" ]]; then
  if [[ -n "${LEGACY_AGENT_MODEL}" ]]; then
    printf 'WARN: Agent model is set only in the legacy %s/.env (%s), not in %s; EVA keeps using its configured defaults.\n' \
      "${APP_DIR}" "${LEGACY_AGENT_MODEL}" "${ENV_FILE}" >&2
  else
    printf 'WARN: no explicit Agent model in %s; EVA keeps using its configured profile defaults.\n' "${ENV_FILE}" >&2
  fi
else
  ok "Agent model stays as configured: ${CONFIGURED_AGENT_MODEL}"
fi
describe_lm_profile() {
  local profile_id="$1"
  local env_id base_url model kind role models_body served_models
  env_id="$(printf '%s' "${profile_id}" | sed -E 's/[^A-Za-z0-9]+/_/g; s/^_+//; s/_+$//' | tr '[:lower:]' '[:upper:]')"
  base_url="$(read_env_value "EVOSSEARCH_LM_PROFILE_${env_id}_BASE_URL")"
  [[ -n "${base_url}" ]] || base_url="$(read_env_value EVOSSEARCH_LM_BASE_URL)"
  model="$(read_env_value "EVOSSEARCH_LM_PROFILE_${env_id}_MODEL")"
  [[ -n "${model}" ]] || model="$(read_env_value EVOSSEARCH_LM_MODEL)"
  kind="$(read_env_value "EVOSSEARCH_LM_PROFILE_${env_id}_KIND")"
  case "${kind}" in
    vlm|vision|video) role="VLM stream inference (typically a dedicated vLLM server)" ;;
    agent) role="Agent reasoning (typically LM Studio or llama.cpp beside EVA)" ;;
    *) role="general" ;;
  esac
  printf 'Profile %-12s kind=%-8s base=%s model=%s\n' "${profile_id}" "${kind:-general}" \
    "${base_url:-unset}" "${model:-unset}"
  printf '  Role: %s\n' "${role}"
  if [[ -z "${base_url}" ]]; then
    printf 'WARN: profile %s has no base URL configured; skipping the reachability probe.\n' "${profile_id}" >&2
    return 0
  fi
  models_body="$(mktemp)"
  if curl -fsS --max-time 5 "${base_url%/}/models" > "${models_body}" 2>/dev/null; then
    served_models="$(target_python - "${models_body}" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = [row for row in (payload.get("data") or []) if isinstance(row, dict)]
    ids = []
    for row in rows:
        model_id = str(row.get("id") or row.get("model") or "").strip()
        if model_id and model_id not in ids:
            ids.append(model_id)
    print(", ".join(ids))
except Exception:
    print("")
PY
)"
    if [[ -n "${served_models}" ]]; then
      printf '  Serving: %s\n' "${served_models}"
      if [[ -n "${model}" ]] && ! printf '%s' "${served_models}" | grep -Fq "${model}"; then
        printf 'WARN: profile %s is configured for model "%s" but the server at %s currently serves: %s.\n' \
          "${profile_id}" "${model}" "${base_url%/}" "${served_models}" >&2
        printf '      Continuing; verify the intended model is loaded after the update.\n' >&2
      fi
    else
      printf 'WARN: profile %s endpoint answered but reported no served models; continuing.\n' "${profile_id}" >&2
    fi
  else
    printf 'WARN: could not reach %s for profile %s; continuing — the endpoint configuration will not be changed.\n' \
      "${base_url%/}/models" "${profile_id}" >&2
  fi
  rm -f "${models_body}"
}
CONFIGURED_LM_PROFILE_IDS="$(read_env_value EVOSSEARCH_LM_PROFILES)"
if [[ -n "${CONFIGURED_LM_PROFILE_IDS}" ]]; then
  IFS=',' read -ra LM_PROFILE_ID_LIST <<< "${CONFIGURED_LM_PROFILE_IDS}"
  for lm_profile_id in "${LM_PROFILE_ID_LIST[@]}"; do
    lm_profile_id="$(printf '%s' "${lm_profile_id}" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')"
    [[ -n "${lm_profile_id}" ]] || continue
    describe_lm_profile "${lm_profile_id}"
  done
else
  printf 'WARN: EVOSSEARCH_LM_PROFILES is not set in %s; EVA runs on the single default LM profile.\n' "${ENV_FILE}" >&2
fi
ok "model/server preflight finished; no configuration was or will be modified"

say "Python dependency preflight (read-only)"
printf 'Checking that the existing .venv can import every %s runtime dependency.\n' "${EXPECTED_VERSION}"
printf 'OpenCV is deliberately excluded: the bundle deploys it into .eva-runtime/python when missing.\n'
MISSING_IMPORTS="$(target_python - <<'PY'
import importlib

modules = [
    "flask", "flask_cors", "dotenv", "numpy", "torch", "torchvision",
    "PIL", "transformers", "clip", "faiss", "requests", "psutil",
    "psycopg", "gunicorn",
]
missing = []
for name in modules:
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)
print(",".join(missing))
PY
)"
if [[ -n "${MISSING_IMPORTS}" ]]; then
  stop "existing .venv cannot import modules required by ${EXPECTED_VERSION}: ${MISSING_IMPORTS}. The requirements files match, but these packages are not actually installed, and this adopt bundle carries no wheelhouse. Install them into the venv (or build a --with-wheelhouse bundle) before updating; nothing was changed."
fi
ok "existing .venv imports every ${EXPECTED_VERSION} runtime dependency (OpenCV comes from the bundle when needed)"

CV_OVERLAY_REQUIRED=false
if target_python - <<'PY' >/dev/null 2>&1
import cv2
import numpy as np
image = np.zeros((8, 8, 3), dtype=np.uint8)
assert cv2.cvtColor(image, cv2.COLOR_BGR2RGB).shape == (8, 8, 3)
PY
then
  ok "existing OpenCV runtime is healthy"
else
  CV_OVERLAY_REQUIRED=true
  ok "existing OpenCV is unavailable; the bundled rescue wheel will be used"
fi

mapfile -t OPENCV_WHEELS < <(find "${BUNDLE_DIR}/runtime/opencv" -maxdepth 1 -type f -name 'opencv_python_headless-*.whl' -print)
[[ "${#OPENCV_WHEELS[@]}" -eq 1 ]] || stop "expected exactly one bundled OpenCV wheel"
CV_PAYLOAD_TEST_DIR="$(mktemp -d)"
if ! target_python -m zipfile -e "${OPENCV_WHEELS[0]}" "${CV_PAYLOAD_TEST_DIR}"; then
  remove_temp_path "${CV_PAYLOAD_TEST_DIR}"
  stop "bundled OpenCV wheel could not be unpacked"
fi
if ! target_python - "${CV_PAYLOAD_TEST_DIR}" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import cv2
import numpy as np
image = np.zeros((8, 8, 3), dtype=np.uint8)
assert cv2.cvtColor(image, cv2.COLOR_BGR2RGB).shape == (8, 8, 3)
PY
then
  remove_temp_path "${CV_PAYLOAD_TEST_DIR}"
  stop "bundled OpenCV wheel is incompatible with the target Python/OS"
fi
remove_temp_path "${CV_PAYLOAD_TEST_DIR}"
ok "bundled OpenCV rescue payload is compatible"

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
SCHEMA_SOURCE="direct read-only DSN query"
if [[ "${SCHEMA_VERSION}" == "NO_DSN" && "${ACTIVE_RUNTIME_DB_OK}" == "true" \
  && "${ACTIVE_RUNTIME_SCHEMA}" == "${EXPECTED_SCHEMA}" ]]; then
  SCHEMA_VERSION="${ACTIVE_RUNTIME_SCHEMA}"
  SCHEMA_SOURCE="active runtime /ready (no DSN stored in selected file)"
fi
[[ "${SCHEMA_VERSION}" == "${EXPECTED_SCHEMA}" ]] \
  || stop "database schema is ${SCHEMA_VERSION}; expected ${EXPECTED_SCHEMA}. No migration was attempted."
ok "database schema is already ${EXPECTED_SCHEMA} via ${SCHEMA_SOURCE}; database will not be changed"

DRY_RUN=(
  "${SOURCE_DIR}/scripts/install_eva_083.py"
  --dry-run --non-interactive --no-migrate --no-start --no-verify
  --verified-adopt-existing-config
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
DRY_RUN_COMMAND=("${DRY_RUN[@]}")
if [[ "${MODE}" == "user" ]]; then
  say "Local rehearsal preflight"
  printf 'WARN: user-systemd dev mode skips the production credential-placeholder policy.\n'
  printf '      Version, bundle, venv, requirements and database schema checks passed.\n'
else
  say "Production installer dry-run"
  if [[ ! -r "${ENV_FILE}" ]]; then
    as_root "${DRY_RUN_COMMAND[@]}" || stop "installer dry-run failed"
  else
    "${DRY_RUN_COMMAND[@]}" || stop "installer dry-run failed"
  fi
fi

if [[ "${CONTEXT_UNKNOWN_REQUIRED}" == true ]]; then
  say "Agent context verification decision"
  printf 'WARN: the updater could not verify the context served by the agent LM.\n' >&2
  printf 'Configured in EVA: %s tokens\n' "${CONFIGURED_AGENT_CONTEXT}" >&2
  printf 'Required for this release: %s tokens\n' "${EXPECTED_AGENT_CONTEXT}" >&2
  printf 'Continue with the operator-verified Agent configuration? [y/N]: '
  read -r CONTEXT_UNKNOWN_DECISION
  [[ "${CONTEXT_UNKNOWN_DECISION}" =~ ^([yY]|[yY][eE][sS])$ ]] \
    || stop "unknown agent context declined; nothing was changed"
  CONTEXT_UNKNOWN_ACCEPTED=true
  ok "operator explicitly accepted an unverified agent context"
fi

if [[ "${CONTEXT_FORCE_REQUIRED}" == true ]]; then
  say "Agent context compatibility decision"
  printf 'WARN: this release is designed for an agent context of %s tokens.\n' "${EXPECTED_AGENT_CONTEXT}" >&2
  printf 'Configured in EVA: %s tokens\n' "${CONFIGURED_AGENT_CONTEXT}" >&2
  printf 'Reported by agent LM: %s tokens\n' "${SERVED_AGENT_CONTEXT}" >&2
  printf 'Safe temporary EVA cap: %s tokens\n' "${TEMPORARY_AGENT_CONTEXT}" >&2
  printf 'The agent will have less room for history and multi-step research until LM Studio is reconfigured.\n' >&2
  printf 'Continue with the temporary context cap? [y/N]: '
  read -r CONTEXT_DECISION
  [[ "${CONTEXT_DECISION}" =~ ^([yY]|[yY][eE][sS])$ ]] \
    || stop "short-context update declined; nothing was changed"
  ok "operator accepted temporary ${TEMPORARY_AGENT_CONTEXT}-token agent context"
fi

if [[ "${MODE}" == "system" && "$(id -u)" -ne 0 ]]; then
  say "Checking sudo access before service stop"
  sudo -v || stop "sudo authentication failed; service was not stopped"
  ok "sudo access confirmed"
fi

read_latest_backup() {
  if [[ "${MODE}" == "system" && "$(id -u)" -ne 0 ]]; then
    as_root cat "${BACKUP_ROOT}/LATEST" 2>/dev/null || true
  else
    cat "${BACKUP_ROOT}/LATEST" 2>/dev/null || true
  fi
}

backup_has_snapshot() {
  local backup_dir="$1"
  if [[ "${MODE}" == "system" && "$(id -u)" -ne 0 ]]; then
    as_root test -f "${backup_dir}/code.tgz"
  else
    test -f "${backup_dir}/code.tgz"
  fi
}

LATEST_BEFORE_UPDATE="$(read_latest_backup)"
ROLLBACK_ARMED=false
ROLLBACK_RUNNING=false

automatic_rollback() {
  local exit_status=$?
  trap - EXIT
  if [[ "${ROLLBACK_ARMED}" != true || "${ROLLBACK_RUNNING}" == true ]]; then
    exit "${exit_status}"
  fi
  ROLLBACK_RUNNING=true
  printf '\n== Update failed; restoring %s automatically\n' "${DEPLOYED_VERSION:-previous EVA AI version}" >&2
  local latest_backup
  latest_backup="$(read_latest_backup)"
  if [[ -z "${latest_backup}" || "${latest_backup}" == "${LATEST_BEFORE_UPDATE}" ]] \
    || ! backup_has_snapshot "${latest_backup}"; then
    printf 'WARN: no new complete code snapshot was recorded; restarting the unchanged service.\n' >&2
    systemctl_write start "${SERVICE_NAME}.service" || true
    exit "${exit_status}"
  fi
  local rollback_command=(
    "${BUNDLE_DIR}/scripts/rollback.sh"
    --backup-dir "${latest_backup}"
    --backup-root "${BACKUP_ROOT}"
    --app-dir "${APP_DIR}"
    --env-file "${ENV_FILE}"
    --service "${SERVICE_NAME}"
    --base-url "${BASE_URL}"
    --no-verify
  )
  if [[ "${MODE}" == "user" ]]; then
    rollback_command+=(--user)
    "${rollback_command[@]}" || {
      printf 'FAIL: automatic rollback failed; backup: %s\n' "${latest_backup}" >&2
      exit "${exit_status}"
    }
  else
    as_root "${rollback_command[@]}" || {
      printf 'FAIL: automatic rollback failed; backup: %s\n' "${latest_backup}" >&2
      exit "${exit_status}"
    }
  fi
  printf 'OK: previous code and configuration restored; database and runtime data were untouched.\n' >&2
  local rollback_deadline=$((SECONDS + 60))
  local rollback_response=""
  while (( SECONDS < rollback_deadline )); do
    if rollback_response="$(curl -skS --max-time 5 "${BASE_URL}/ready?load=1" 2>/dev/null)" \
      && printf '%s' "${rollback_response}" | ready_json_reports_version "${DEPLOYED_VERSION}"; then
      if printf '%s' "${rollback_response}" | ready_json_matches_version "${DEPLOYED_VERSION}"; then
        printf 'OK: %s is back up at %s\n' "${DEPLOYED_VERSION}" "${BASE_URL}" >&2
      else
        printf 'WARN: restored %s is running at %s but /ready remains degraded; inspect required dependencies.\n' \
          "${DEPLOYED_VERSION}" "${BASE_URL}" >&2
      fi
      exit "${exit_status}"
    fi
    sleep 5
  done
  printf 'WARN: previous code was restored, but %s did not report version %s within 60 seconds.\n' \
    "${BASE_URL}" "${DEPLOYED_VERSION}" >&2
  exit "${exit_status}"
}
trap automatic_rollback EXIT

printf '\nInstall %s now? Database and runtime data stay unchanged. [y/N]: ' "${EXPECTED_VERSION}"
read -r CONFIRMATION
[[ "${CONFIRMATION}" =~ ^([yY]|[yY][eE][sS])$ ]] \
  || stop "confirmation not received; nothing was changed"

say "Stopping ${SERVICE_NAME}.service"
systemctl_write stop "${SERVICE_NAME}.service"
ROLLBACK_ARMED=true

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
    --exclude="${APP_BASE}/*/.git" \
    --exclude="${APP_BASE}/.local" \
    --exclude="${APP_BASE}/*/.local" \
    --exclude="${APP_BASE}/.venv*" \
    --exclude="${APP_BASE}/*/.venv*" \
    --exclude="${APP_BASE}/.env" \
    --exclude="${APP_BASE}/.env.*" \
    --exclude="${APP_BASE}/dist" \
    --exclude="${APP_BASE}/*/dist" \
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
    --exclude="${APP_BASE}/luxriot_summary_state.json" \
    --exclude="${APP_BASE}/luxriot_rollups_cache.json" \
    --exclude="${APP_BASE}/*.sqlite3" \
    --exclude="${APP_BASE}/*.db" \
    --exclude="${APP_BASE}/*.log" \
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

MEDIA_INSTALL=(
  "${BUNDLE_DIR}/scripts/install_media_runtime.sh"
  --bundle-dir "${BUNDLE_DIR}"
  --app-dir "${APP_DIR}"
  --python "${APP_DIR}/.venv/bin/python"
  --owner "${SERVICE_USER}:${SERVICE_GROUP}"
)
if [[ "${CV_OVERLAY_REQUIRED}" == true ]]; then
  MEDIA_INSTALL+=(--with-opencv-overlay)
fi
if [[ "${MODE}" == "user" ]]; then
  "${MEDIA_INSTALL[@]}"
else
  as_root "${MEDIA_INSTALL[@]}"
fi

target_python - "${ENV_FILE}" "${EXPECTED_VERSION}" "${APP_DIR}/.eva-bundle-commit" "${BUNDLE_COMMIT}" "${TEMPORARY_AGENT_CONTEXT}" <<'PY'
import os
import re
import stat
import sys
import tempfile
from pathlib import Path

path = Path(sys.argv[1])
version = sys.argv[2]
marker_path = Path(sys.argv[3])
bundle_commit = sys.argv[4]
temporary_agent_context = sys.argv[5].strip()
original = path.read_text(encoding="utf-8")
replacement = f'EVOSSEARCH_APP_VERSION="{version}"'
updated, count = re.subn(
    r"(?m)^[ \t]*EVOSSEARCH_APP_VERSION[ \t]*=.*$",
    replacement,
    original,
)
if count == 0:
    updated = original.rstrip("\n") + "\n" + replacement + "\n"
if temporary_agent_context:
    context_replacement = f"EVOSSEARCH_AGENT_CONTEXT_LIMIT_TOKENS={temporary_agent_context}"
    updated, count = re.subn(
        r"(?m)^[ \t]*EVOSSEARCH_AGENT_CONTEXT_LIMIT_TOKENS[ \t]*=.*$",
        context_replacement,
        updated,
    )
    if count == 0:
        updated = updated.rstrip("\n") + "\n" + context_replacement + "\n"
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
marker_path.write_text(bundle_commit + "\n", encoding="utf-8")
PY
ok "code installed; database and runtime data were not changed"

printf '\nRestart %s.service now? [Y/n]: ' "${SERVICE_NAME}"
read -r RESTART_ANSWER
case "${RESTART_ANSWER}" in
  ""|y|Y|yes|YES) ;;
  *)
    ROLLBACK_ARMED=false
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
POST_UPDATE_DEGRADED=false
READY_DEADLINE=$((SECONDS + 240))
while (( SECONDS < READY_DEADLINE )); do
  if READY_JSON="$(curl -skfS --max-time 5 "${BASE_URL}/ready?load=1" 2>/dev/null)"; then
    if printf '%s' "${READY_JSON}" | ready_json_matches_version "${EXPECTED_VERSION}"; then
      break
    fi
    if [[ "${PREUPGRADE_DEGRADED}" == true ]] \
      && printf '%s' "${READY_JSON}" | ready_json_reports_version "${EXPECTED_VERSION}"; then
      POST_UPDATE_DEGRADED=true
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
ROLLBACK_ARMED=false

printf '\n============================================================\n'
printf 'OK: EVA AI %s is up and running\n' "${EXPECTED_VERSION}"
if [[ "${POST_UPDATE_DEGRADED}" == true ]]; then
  printf 'WARN: /ready is degraded, matching the pre-update state (an external dependency such as the VLM server is still unavailable).\n'
  printf '      Model and server settings were not changed by this update.\n'
fi
printf 'URL: %s\n' "${BASE_URL}"
printf 'Service: %s.service (%s systemd)\n' "${SERVICE_NAME}" "${MODE}"
if [[ -n "${TEMPORARY_AGENT_CONTEXT}" ]]; then
  printf 'Agent context: %s tokens (TEMPORARY FORCED CAP; target is %s)\n' \
    "${TEMPORARY_AGENT_CONTEXT}" "${EXPECTED_AGENT_CONTEXT}"
  printf 'Next: raise LM Studio context, then set EVOSSEARCH_AGENT_CONTEXT_LIMIT_TOKENS=%s and restart EVA.\n' \
    "${EXPECTED_AGENT_CONTEXT}"
elif [[ "${CONTEXT_UNKNOWN_ACCEPTED}" == true ]]; then
  printf 'Agent context: UNVERIFIED (operator accepted the warning)\n'
  printf 'WARN: EVA is up, but agent LM availability/context still requires manual verification.\n'
else
  printf 'Agent context: %s tokens\n' "${SERVED_AGENT_CONTEXT}"
fi
if [[ "${MODE}" == "user" ]]; then
  LATEST_BACKUP="$(cat "${BACKUP_ROOT}/LATEST" 2>/dev/null || printf '%s' "${BACKUP_ROOT}")"
else
  LATEST_BACKUP="$(as_root cat "${BACKUP_ROOT}/LATEST" 2>/dev/null || printf '%s' "${BACKUP_ROOT}")"
fi
printf 'Backup: %s\n' "${LATEST_BACKUP}"
printf '============================================================\n'
