#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

ENV_FILE="${EVA_ENV_FILE:-/etc/eva-ai/eva-ai.env}"
SERVICE_NAME="${EVA_SERVICE_NAME:-eva-ai}"
LUXRIOT_IP="${LUXRIOT_EVO_IP:-}"
LUXRIOT_PORT="${LUXRIOT_EVO_PORT:-8080}"
INFERENCE_A_IP="${INFERENCE_A_IP:-}"
INFERENCE_B_IP="${INFERENCE_B_IP:-}"
AGENT_BASE_URL="${AGENT_BASE_URL:-}"
RESTART_SERVICE=false

ok() {
  printf 'OK: %s\n' "$*"
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
Usage: sudo scripts/set_site_ips.sh [options]

Options:
  --env-file FILE          Runtime env file. Default: /etc/eva-ai/eva-ai.env.
  --service NAME           systemd service name. Default: eva-ai.
  --luxriot-ip IP          Luxriot Evo server IP or hostname.
  --luxriot-port PORT      Luxriot Evo HTTP port. Default: 8080.
  --inference-a-ip IP      First vLLM server IP or hostname.
  --inference-b-ip IP      Second vLLM server IP or hostname.
  --agent-base-url URL     OpenAI-compatible agent base URL.
  --restart                Restart systemd service after env update.
  -h, --help               Show this help.

Environment alternatives:
  LUXRIOT_EVO_IP, LUXRIOT_EVO_PORT, INFERENCE_A_IP, INFERENCE_B_IP,
  AGENT_BASE_URL.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --service)
      SERVICE_NAME="$2"
      shift 2
      ;;
    --luxriot-ip)
      LUXRIOT_IP="$2"
      shift 2
      ;;
    --luxriot-port)
      LUXRIOT_PORT="$2"
      shift 2
      ;;
    --inference-a-ip)
      INFERENCE_A_IP="$2"
      shift 2
      ;;
    --inference-b-ip)
      INFERENCE_B_IP="$2"
      shift 2
      ;;
    --agent-base-url)
      AGENT_BASE_URL="$2"
      shift 2
      ;;
    --restart)
      RESTART_SERVICE=true
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
  die "Run this script with sudo/root so it can update ${ENV_FILE}."
fi

[[ -f "${ENV_FILE}" ]] || die "env file not found: ${ENV_FILE}"
[[ -n "${LUXRIOT_IP}" ]] || die "--luxriot-ip or LUXRIOT_EVO_IP is required"
[[ -n "${INFERENCE_A_IP}" ]] || die "--inference-a-ip or INFERENCE_A_IP is required"
[[ -n "${INFERENCE_B_IP}" ]] || die "--inference-b-ip or INFERENCE_B_IP is required"
[[ -n "${AGENT_BASE_URL}" ]] || die "--agent-base-url or AGENT_BASE_URL is required"

BACKUP_PATH="${ENV_FILE}.bak.$(date +%Y%m%d-%H%M%S)"
cp -a "${ENV_FILE}" "${BACKUP_PATH}"
ok "backed up env file to ${BACKUP_PATH}"

TMP_FILE="$(mktemp)"
cp "${ENV_FILE}" "${TMP_FILE}"

set_env_key() {
  local key="$1"
  local value="$2"
  local next_file
  next_file="$(mktemp)"
  awk -v key="${key}" -v value="${value}" '
    BEGIN { found = 0 }
    $0 ~ "^[[:space:]]*" key "=" {
      print key "=" value
      found = 1
      next
    }
    { print }
    END {
      if (found == 0) {
        print key "=" value
      }
    }
  ' "${TMP_FILE}" > "${next_file}"
  mv "${next_file}" "${TMP_FILE}"
}

set_env_key "EVOSSEARCH_LUXRIOT_BASE_URL" "http://${LUXRIOT_IP}:${LUXRIOT_PORT}"
set_env_key "EVOSSEARCH_LM_PROFILE_AGENT_BASE_URL" "${AGENT_BASE_URL}"
set_env_key "EVOSSEARCH_LM_PROFILE_VLM_A1_BASE_URL" "http://${INFERENCE_A_IP}:8001/v1"
set_env_key "EVOSSEARCH_LM_PROFILE_VLM_A0_BASE_URL" "http://${INFERENCE_A_IP}:8002/v1"
set_env_key "EVOSSEARCH_LM_PROFILE_VLM_B1_BASE_URL" "http://${INFERENCE_B_IP}:8001/v1"
set_env_key "EVOSSEARCH_LM_PROFILE_VLM_B0_BASE_URL" "http://${INFERENCE_B_IP}:8002/v1"

ENV_MODE="$(stat -c '%a' "${ENV_FILE}")"
ENV_OWNER="$(stat -c '%U:%G' "${ENV_FILE}")"
install -m "${ENV_MODE}" -o "${ENV_OWNER%%:*}" -g "${ENV_OWNER##*:}" "${TMP_FILE}" "${ENV_FILE}"
rm -f "${TMP_FILE}"

ok "updated site IP settings in ${ENV_FILE}"
printf 'OK: EVOSSEARCH_LUXRIOT_BASE_URL=http://%s:%s\n' "${LUXRIOT_IP}" "${LUXRIOT_PORT}"
printf 'OK: EVOSSEARCH_LM_PROFILE_AGENT_BASE_URL=%s\n' "${AGENT_BASE_URL}"
printf 'OK: VLM A=http://%s:8001/v1,http://%s:8002/v1\n' "${INFERENCE_A_IP}" "${INFERENCE_A_IP}"
printf 'OK: VLM B=http://%s:8001/v1,http://%s:8002/v1\n' "${INFERENCE_B_IP}" "${INFERENCE_B_IP}"

if [[ "${RESTART_SERVICE}" == true ]]; then
  systemctl restart "${SERVICE_NAME}.service"
  ok "restarted ${SERVICE_NAME}"
else
  printf 'OK: service not restarted; run: sudo systemctl restart %s\n' "${SERVICE_NAME}"
fi
