#!/usr/bin/env bash
set -Eeuo pipefail

SERVICE_NAME="${EVA_SERVICE_NAME:-eva-ai}"
BASE_URL="${EVA_BASE_URL:-http://127.0.0.1:5000}"
TIMEOUT_SECONDS="${EVA_VERIFY_TIMEOUT_SECONDS:-300}"
CHECK_SERVICE=true
USER_SERVICE=false
CURL_INSECURE="${EVA_PATCH_CURL_INSECURE:-false}"

ok() {
  printf 'OK: %s\n' "$*"
}

fail() {
  printf 'FAIL: %s\n' "$*" >&2
}

warn() {
  printf 'WARN: %s\n' "$*" >&2
}

usage() {
  cat <<'USAGE'
Usage: scripts/verify_patch.sh [options]

Options:
  --service NAME      systemd service name. Default: eva-ai.
  --base-url URL      App base URL. Default: http://127.0.0.1:5000.
  --timeout SECONDS   Wait timeout for endpoints. Default: 300.
  --user-service      Check a per-user systemd service with systemctl --user.
  --skip-service      Do not check systemd service state.
  --curl-insecure     Allow self-signed HTTPS checks.
  -h, --help          Show this help.

Environment:
  EVA_PATCH_CURL_INSECURE=true allows self-signed HTTPS checks.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --service)
      SERVICE_NAME="$2"
      shift 2
      ;;
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT_SECONDS="$2"
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

if ! command -v curl >/dev/null 2>&1; then
  fail "curl is required"
  exit 1
fi

CURL_OPTS=(-sS --max-time 8)
if [[ "${CURL_INSECURE}" == true ]]; then
  CURL_OPTS+=(-k)
fi

RESULT=0

if [[ "${CHECK_SERVICE}" == true ]]; then
  if command -v systemctl >/dev/null 2>&1; then
    SYSTEMCTL=(systemctl)
    if [[ "${USER_SERVICE}" == true ]]; then
      SYSTEMCTL=(systemctl --user)
    fi
    if "${SYSTEMCTL[@]}" is-active --quiet "${SERVICE_NAME}.service"; then
      ok "service ${SERVICE_NAME} is active"
    else
      fail "service ${SERVICE_NAME} is not active"
      RESULT=1
    fi
  else
    warn "systemctl not found; service check skipped"
  fi
fi

check_endpoint() {
  local name="$1"
  local path="$2"
  local deadline now code body

  deadline=$((SECONDS + TIMEOUT_SECONDS))
  body="$(mktemp)"
  while true; do
    code="$(curl "${CURL_OPTS[@]}" -o "${body}" -w '%{http_code}' "${BASE_URL}${path}" 2>/tmp/eva-patch-curl.err || true)"
    if [[ "${code}" == "200" ]]; then
      ok "${name} endpoint returned HTTP 200"
      if command -v jq >/dev/null 2>&1 && jq -e . "${body}" >/dev/null 2>&1; then
        status="$(jq -r '.status // .state // .ok // empty' "${body}" 2>/dev/null || true)"
        if [[ -n "${status}" ]]; then
          ok "${name} status: ${status}"
        fi
      fi
      rm -f "${body}" /tmp/eva-patch-curl.err
      return 0
    fi

    now="${SECONDS}"
    if (( now >= deadline )); then
      fail "${name} endpoint failed at ${BASE_URL}${path} with HTTP ${code:-000}"
      if [[ -s /tmp/eva-patch-curl.err ]]; then
        sed -n '1,3p' /tmp/eva-patch-curl.err >&2
      fi
      if [[ -s "${body}" ]]; then
        sed -n '1,8p' "${body}" >&2
      fi
      rm -f "${body}" /tmp/eva-patch-curl.err
      return 1
    fi
    sleep 2
  done
}

check_endpoint "health" "/health" || RESULT=1
check_endpoint "ready" "/ready" || RESULT=1

check_react_ui() {
  local body code asset asset_code
  body="$(mktemp)"
  code="$(curl "${CURL_OPTS[@]}" -o "${body}" -w '%{http_code}' "${BASE_URL}/" 2>/tmp/eva-patch-curl.err || true)"
  if [[ "${code}" != "200" ]]; then
    fail "React UI shell failed at ${BASE_URL}/ with HTTP ${code:-000}"
    rm -f "${body}" /tmp/eva-patch-curl.err
    return 1
  fi
  if ! grep -q '<div id="root"></div>' "${body}" \
    || ! grep -q '/ui-assets/assets/' "${body}"
  then
    fail "root page is not the React command console (legacy or incomplete UI detected)"
    rm -f "${body}" /tmp/eva-patch-curl.err
    return 1
  fi
  asset="$(grep -Eo '/ui-assets/assets/[^"[:space:]>]+' "${body}" | head -n 1 || true)"
  if [[ -z "${asset}" ]]; then
    fail "React UI shell contains no hashed asset reference"
    rm -f "${body}" /tmp/eva-patch-curl.err
    return 1
  fi
  asset_code="$(curl "${CURL_OPTS[@]}" -o /dev/null -w '%{http_code}' "${BASE_URL}${asset}" 2>/tmp/eva-patch-curl.err || true)"
  rm -f "${body}" /tmp/eva-patch-curl.err
  if [[ "${asset_code}" != "200" ]]; then
    fail "React UI asset ${asset} failed with HTTP ${asset_code:-000}"
    return 1
  fi
  ok "React command console and hashed frontend asset are served"
}

check_react_ui || RESULT=1

if [[ "${RESULT}" -eq 0 ]]; then
  ok "verification completed"
else
  fail "verification failed"
fi

exit "${RESULT}"
