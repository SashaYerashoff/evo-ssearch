#!/usr/bin/env bash
set -Eeuo pipefail

umask 077
IFS=$'\n\t'

SCRIPT_VERSION="2026-06-22"
SERVICE_NAME="${EVA_DIAG_SERVICE:-eva-ai}"
SNAPSHOT_COUNT="${EVA_DIAG_SNAPSHOT_COUNT:-3}"
JOURNAL_LINES="${EVA_DIAG_JOURNAL_LINES:-300}"
CONNECT_TIMEOUT="${EVA_DIAG_CONNECT_TIMEOUT:-3}"
MAX_TIME="${EVA_DIAG_MAX_TIME:-20}"
OUT_PARENT="${EVA_DIAG_OUTPUT_DIR:-/tmp}"
ENV_FILE="${EVA_DIAG_ENV_FILE:-/etc/eva-ai/eva-ai.env}"
INCLUDE_DOTENV="${EVA_DIAG_INCLUDE_DOTENV:-0}"
FORCE_CHANNELS="${EVA_DIAG_FORCE_CHANNELS:-0}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/client_diagnostics.sh

Optional environment:
  EVA_BASE_URL=http://127.0.0.1:5000
  EVA_COOKIE_FILE=/tmp/eva.cookies
  EVA_COOKIE='session=...'
  EVA_COOKIE_HEADER='session=...'
  EVA_DIAG_SNAPSHOT_COUNT=3
  EVA_DIAG_JOURNAL_LINES=300
  EVA_DIAG_OUTPUT_DIR=/tmp
  EVA_DIAG_ENV_FILE=/etc/eva-ai/eva-ai.env
  EVA_DIAG_VLLM_BASE_URLS='http://host-a:8001/v1,http://host-a:8002/v1'

The script is read-only against EVA AI: it uses systemctl/journalctl and HTTP GETs.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! [[ "$SNAPSHOT_COUNT" =~ ^[0-9]+$ ]]; then
  echo "EVA_DIAG_SNAPSHOT_COUNT must be a non-negative integer" >&2
  exit 2
fi

if ! [[ "$JOURNAL_LINES" =~ ^[0-9]+$ ]]; then
  echo "EVA_DIAG_JOURNAL_LINES must be a non-negative integer" >&2
  exit 2
fi

have() {
  command -v "$1" >/dev/null 2>&1
}

trim() {
  sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' <<<"$1"
}

redact_stream() {
  sed -E \
    -e 's#(postgres(ql)?://[^:/@[:space:]]+:)[^@[:space:]]+@#\1<redacted>@#g' \
    -e 's#(https?://[^:/@[:space:]]+:)[^@[:space:]]+@#\1<redacted>@#g' \
    -e 's#([?&](access_)?token=)[^&[:space:]]+#\1<redacted>#Ig' \
    -e 's#([?&](api[_-]?key|password|secret|cookie|auth)=)[^&[:space:]]+#\1<redacted>#Ig' \
    -e 's#(Authorization:[[:space:]]*)(Bearer|Basic)[[:space:]]+[^[:space:]]+#\1\2 <redacted>#Ig' \
    -e 's#(Cookie:[[:space:]]*).*#\1<redacted>#Ig' \
    -e 's#(Set-Cookie:[[:space:]]*)[^;[:space:]]+#\1<redacted>#Ig' \
    -e 's#("?(password|passwd|secret|token|access_token|api[_-]?key|authorization|cookie|csrf[_-]?token|session[_-]?id)"?[[:space:]]*[:=][[:space:]]*)("[^"]*"|[^,[:space:]}]+)#\1<redacted>#Ig' \
    -e 's#^([A-Za-z_][A-Za-z0-9_]*(PASSWORD|PASS|SECRET|TOKEN|COOKIE|API_KEY|PRIVATE_KEY|CREDENTIAL|AUTH)[A-Za-z0-9_]*=).*#\1<redacted>#Ig'
}

redact_inline() {
  redact_stream <<<"$1" | tr '\n' ' ' | sed -e 's/[[:space:]]*$//'
}

redact_env_value() {
  local key="$1"
  local value="$2"

  if [[ "$key" =~ (PASSWORD|PASS|SECRET|TOKEN|COOKIE|API_KEY|PRIVATE_KEY|CREDENTIAL|AUTH) ]]; then
    printf '<redacted>'
    return
  fi

  redact_inline "$value"
}

safe_name() {
  tr -c 'A-Za-z0-9_.-' '_' <<<"$1" | sed -e 's/^_*//' -e 's/_*$//' -e 's/__*/_/g'
}

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_PARENT%/}/eva-ai-diagnostics-${timestamp}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

mkdir -p "$OUT_DIR/system" "$OUT_DIR/api" "$OUT_DIR/env" "$OUT_DIR/vllm" "$OUT_DIR/snapshots"

declare -A ENV_VALUES=()
declare -A ENV_SOURCES=()
declare -a ENV_SOURCE_NOTES=()

set_env_value() {
  local key="$1"
  local value="$2"
  local source="$3"

  [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || return 0
  ENV_VALUES["$key"]="$value"
  if [[ -n "${ENV_SOURCES[$key]+x}" ]]; then
    ENV_SOURCES["$key"]="${ENV_SOURCES[$key]},${source}"
  else
    ENV_SOURCES["$key"]="$source"
  fi
}

parse_env_assignment() {
  local raw="$1"
  local source="$2"
  local line key value last

  line="$(trim "$raw")"
  [[ -z "$line" || "${line:0:1}" == "#" ]] && return 0
  if [[ "$line" == export[[:space:]]* ]]; then
    line="$(trim "${line#export}")"
  fi
  [[ "$line" == *"="* ]] || return 0

  key="$(trim "${line%%=*}")"
  value="$(trim "${line#*=}")"
  [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || return 0

  if (( ${#value} >= 2 )); then
    last="${value: -1}"
    if [[ "${value:0:1}" == "\"" && "$last" == "\"" ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "${value:0:1}" == "'" && "$last" == "'" ]]; then
      value="${value:1:${#value}-2}"
    fi
  fi

  set_env_value "$key" "$value" "$source"
}

load_env_file() {
  local file="$1"
  local source="$2"
  local raw

  if [[ ! -e "$file" ]]; then
    ENV_SOURCE_NOTES+=("${file}: missing")
    return
  fi
  if [[ ! -r "$file" ]]; then
    ENV_SOURCE_NOTES+=("${file}: not readable by $(id -un)")
    return
  fi

  ENV_SOURCE_NOTES+=("${file}: readable")
  while IFS= read -r raw || [[ -n "$raw" ]]; do
    parse_env_assignment "$raw" "$source"
  done <"$file"
}

load_systemd_environment() {
  local raw item
  local old_ifs="$IFS"

  if ! have systemctl; then
    ENV_SOURCE_NOTES+=("systemctl: unavailable")
    return
  fi

  raw="$(systemctl show "$SERVICE_NAME" --property=Environment --value 2>/dev/null || true)"
  if [[ -z "$raw" ]]; then
    ENV_SOURCE_NOTES+=("systemd ${SERVICE_NAME} Environment: empty or unavailable")
    return
  fi

  ENV_SOURCE_NOTES+=("systemd ${SERVICE_NAME} Environment: readable")
  IFS=' '
  for item in $raw; do
    parse_env_assignment "$item" "systemd:${SERVICE_NAME}"
  done
  IFS="$old_ifs"
}

load_process_environment() {
  local key value

  while IFS='=' read -r key value; do
    [[ -n "$key" ]] || continue
    set_env_value "$key" "$value" "process"
  done < <(env)
}

load_env_file "$ENV_FILE" "$ENV_FILE"
if [[ "$INCLUDE_DOTENV" == "1" ]]; then
  load_env_file ".env" ".env"
fi
load_systemd_environment
load_process_environment

if [[ -n "${EVA_BASE_URL:-}" ]]; then
  BASE_URL="${EVA_BASE_URL%/}"
else
  port="${ENV_VALUES[EVOSSEARCH_PORT]:-5000}"
  BASE_URL="http://127.0.0.1:${port}"
fi

declare -a COOKIE_ARGS=()
AUTH_AVAILABLE=0
AUTH_SOURCE="none"

if [[ -n "${EVA_COOKIE_FILE:-}" ]]; then
  if [[ -r "$EVA_COOKIE_FILE" ]]; then
    COOKIE_ARGS=(-b "$EVA_COOKIE_FILE")
    AUTH_AVAILABLE=1
    AUTH_SOURCE="cookie-file"
  else
    AUTH_SOURCE="cookie-file-not-readable"
  fi
elif [[ -n "${EVA_COOKIE_HEADER:-}" ]]; then
  cookie_header="${EVA_COOKIE_HEADER#Cookie:}"
  cookie_header="$(trim "$cookie_header")"
  COOKIE_ARGS=(-H "Cookie: ${cookie_header}")
  AUTH_AVAILABLE=1
  AUTH_SOURCE="cookie-header"
elif [[ -n "${EVA_COOKIE:-}" ]]; then
  COOKIE_ARGS=(-b "$EVA_COOKIE")
  AUTH_AVAILABLE=1
  AUTH_SOURCE="cookie-env"
fi

run_capture() {
  local outfile="$1"
  shift
  local status

  {
    printf '# command:'
    printf ' %q' "$@"
    printf '\n# started_utc: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    set +e
    "$@" 2>&1
    status=$?
    set -e
    printf '\n# exit_status: %s\n' "$status"
    printf '# finished_utc: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } | redact_stream >"$outfile"
}

write_command_unavailable() {
  local outfile="$1"
  local command_name="$2"
  {
    printf '# command unavailable: %s\n' "$command_name"
    printf '# host: %s\n' "$(hostname 2>/dev/null || printf unknown)"
  } >"$outfile"
}

pretty_body() {
  local body="$1"
  local pretty="$TMP_DIR/pretty-body"

  if [[ ! -f "$body" ]]; then
    printf '# no response body captured\n'
    return
  fi

  if have jq && jq . "$body" >"$pretty" 2>/dev/null; then
    cat "$pretty"
  else
    cat "$body"
  fi
}

curl_get_endpoint() {
  local path="$1"
  local outfile="$2"
  local auth_mode="${3:-noauth}"
  local body_copy="${4:-}"
  local url="${BASE_URL}${path}"
  local body headers meta err curl_status
  local -a args

  if ! have curl; then
    write_command_unavailable "$outfile" "curl"
    return
  fi

  body="$TMP_DIR/body-$(safe_name "$path")"
  headers="$TMP_DIR/headers-$(safe_name "$path")"
  meta="$TMP_DIR/meta-$(safe_name "$path")"
  err="$TMP_DIR/curl-err-$(safe_name "$path")"
  : >"$body"
  : >"$headers"
  : >"$meta"
  : >"$err"

  args=(curl -sS --connect-timeout "$CONNECT_TIMEOUT" --max-time "$MAX_TIME" -D "$headers" -o "$body" -w $'http_code=%{http_code}\ntime_total=%{time_total}\nsize_download=%{size_download}\n')
  if [[ "$auth_mode" == "auth" ]]; then
    args+=("${COOKIE_ARGS[@]}")
  fi
  args+=("$url")

  set +e
  "${args[@]}" >"$meta" 2>"$err"
  curl_status=$?
  set -e

  {
    printf '# endpoint: %s\n' "$path"
    printf '# url: %s\n' "$(redact_inline "$url")"
    printf '# auth: %s\n' "$auth_mode"
    printf '# auth_source: %s\n' "$AUTH_SOURCE"
    printf '# curl_exit_status: %s\n' "$curl_status"
    cat "$meta"
    if [[ -s "$err" ]]; then
      printf '# curl_stderr:\n'
      cat "$err"
    fi
    printf '# response_headers:\n'
    cat "$headers" 2>/dev/null || true
    printf '\n# response_body:\n'
    pretty_body "$body"
  } | redact_stream >"$outfile"

  if [[ -n "$body_copy" ]]; then
    cp "$body" "$body_copy"
  fi
}

sha1_file() {
  local file="$1"

  if [[ ! -f "$file" ]]; then
    printf 'missing-file'
    return
  fi

  if have sha1sum; then
    sha1sum "$file" | awk '{print $1}'
  elif have shasum; then
    shasum -a 1 "$file" | awk '{print $1}'
  else
    printf 'sha1-tool-unavailable'
  fi
}

header_value() {
  local header_file="$1"
  local header_name="$2"

  awk -v name="$header_name" '
    BEGIN { IGNORECASE = 1 }
    index($0, name ":") == 1 {
      sub(/\r$/, "")
      sub(/^[^:]+:[[:space:]]*/, "")
      value = $0
    }
    END { print value }
  ' "$header_file"
}

extract_channel_ids() {
  local body="$1"

  if have jq; then
    jq -r '
      def ids:
        if type == "array" then .[]
        elif type == "object" then (.channels // .data // .video_streams // .streams // [])
          | if type == "array" then .[] else empty end
        else empty end;
      ids | (.id // .channel_id // .channelId // empty)
    ' "$body" 2>/dev/null | awk 'NF && !seen[$0]++' || true
  else
    {
      grep -Eo '"(id|channel_id|channelId)"[[:space:]]*:[[:space:]]*"?[0-9]+' "$body" 2>/dev/null \
      | grep -Eo '[0-9]+$' \
      | awk 'NF && !seen[$0]++'
    } || true
  fi
}

tsv_safe() {
  tr '\t\r\n' '   ' <<<"$1" | sed -e 's/[[:space:]]*$//'
}

fetch_snapshot_digest() {
  local channel_id="$1"
  local image_file headers err meta curl_status
  local http_code time_total size_download size_bytes sha1 content_type width height error_text url
  local -a args

  image_file="$TMP_DIR/snapshot-${channel_id}.bin"
  headers="$TMP_DIR/snapshot-${channel_id}.headers"
  err="$TMP_DIR/snapshot-${channel_id}.err"
  url="${BASE_URL}/luxriot/snapshot/${channel_id}"
  : >"$image_file"
  : >"$headers"
  : >"$err"

  args=(curl -sS --connect-timeout "$CONNECT_TIMEOUT" --max-time "$MAX_TIME" -D "$headers" -o "$image_file" -w $'http_code=%{http_code}\ttime_total=%{time_total}\tsize_download=%{size_download}')
  args+=("${COOKIE_ARGS[@]}")
  args+=("$url")

  set +e
  meta="$("${args[@]}" 2>"$err")"
  curl_status=$?
  set -e

  http_code="$(tr '\t' '\n' <<<"$meta" | awk -F= '$1 == "http_code" { print $2 }')"
  time_total="$(tr '\t' '\n' <<<"$meta" | awk -F= '$1 == "time_total" { print $2 }')"
  size_download="$(tr '\t' '\n' <<<"$meta" | awk -F= '$1 == "size_download" { print $2 }')"
  size_bytes="0"
  if [[ -f "$image_file" ]]; then
    size_bytes="$(wc -c <"$image_file" | tr -d '[:space:]')"
  fi
  [[ -n "$size_download" ]] || size_download="$size_bytes"
  sha1="$(sha1_file "$image_file")"
  content_type="$(header_value "$headers" "Content-Type")"
  width="$(header_value "$headers" "X-Image-Width")"
  height="$(header_value "$headers" "X-Image-Height")"
  error_text=""
  if [[ -s "$err" ]]; then
    error_text="$(redact_inline "$(cat "$err")")"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(tsv_safe "$channel_id")" \
    "$(tsv_safe "${http_code:-unknown}")" \
    "$(tsv_safe "$curl_status")" \
    "$(tsv_safe "$size_bytes")" \
    "$(tsv_safe "$sha1")" \
    "$(tsv_safe "${time_total:-unknown}")" \
    "$(tsv_safe "$content_type")" \
    "$(tsv_safe "$width")" \
    "$(tsv_safe "$height")" \
    "$(tsv_safe "$error_text")"
}

collect_vllm_urls() {
  local key value base_list item

  if [[ -n "${EVA_DIAG_VLLM_BASE_URLS:-}" ]]; then
    base_list="$(tr ',' '\n' <<<"$EVA_DIAG_VLLM_BASE_URLS")"
    while IFS= read -r item; do
      item="$(trim "$item")"
      [[ -n "$item" ]] && printf 'EVA_DIAG_VLLM_BASE_URLS\t%s\n' "$item"
    done <<<"$base_list"
  fi

  for key in "${!ENV_VALUES[@]}"; do
    value="${ENV_VALUES[$key]}"
    [[ -n "$value" ]] || continue
    if [[ "$key" =~ ^EVOSSEARCH_LM_PROFILE_.*_BASE_URL$ ]]; then
      printf '%s\t%s\n' "$key" "$value"
    elif [[ "$key" =~ (VLLM|LM_STUDIO|LMSTUDIO).*BASE_URL$ ]]; then
      printf '%s\t%s\n' "$key" "$value"
    fi
  done | awk -F '\t' 'NF == 2 && !seen[$2]++'
}

profile_api_key_for_base_key() {
  local base_key="$1"
  local api_key_name

  if [[ "$base_key" =~ ^EVOSSEARCH_LM_PROFILE_(.*)_BASE_URL$ ]]; then
    api_key_name="EVOSSEARCH_LM_PROFILE_${BASH_REMATCH[1]}_API_KEY"
    printf '%s' "${ENV_VALUES[$api_key_name]:-}"
  fi
  return 0
}

curl_model_endpoint() {
  local label="$1"
  local base_url="$2"
  local api_key="$3"
  local outfile="$4"
  local models_url body headers meta err curl_status
  local -a args

  if ! have curl; then
    write_command_unavailable "$outfile" "curl"
    return
  fi

  if [[ "$base_url" == */models ]]; then
    models_url="$base_url"
  else
    models_url="${base_url%/}/models"
  fi

  body="$TMP_DIR/vllm-body-$(safe_name "$label")"
  headers="$TMP_DIR/vllm-headers-$(safe_name "$label")"
  meta="$TMP_DIR/vllm-meta-$(safe_name "$label")"
  err="$TMP_DIR/vllm-err-$(safe_name "$label")"
  : >"$body"
  : >"$headers"
  : >"$meta"
  : >"$err"

  args=(curl -sS --connect-timeout "$CONNECT_TIMEOUT" --max-time "$MAX_TIME" -D "$headers" -o "$body" -w $'http_code=%{http_code}\ntime_total=%{time_total}\nsize_download=%{size_download}\n')
  if [[ -n "$api_key" ]]; then
    args+=(-H "Authorization: Bearer ${api_key}")
  fi
  args+=("$models_url")

  set +e
  "${args[@]}" >"$meta" 2>"$err"
  curl_status=$?
  set -e

  {
    printf '# env_key: %s\n' "$label"
    printf '# models_url: %s\n' "$(redact_inline "$models_url")"
    printf '# authorization_header: %s\n' "$([[ -n "$api_key" ]] && printf present || printf absent)"
    printf '# curl_exit_status: %s\n' "$curl_status"
    cat "$meta"
    if [[ -s "$err" ]]; then
      printf '# curl_stderr:\n'
      cat "$err"
    fi
    printf '# response_headers:\n'
    cat "$headers" 2>/dev/null || true
    printf '\n# response_body:\n'
    pretty_body "$body"
  } | redact_stream >"$outfile"
}

{
  printf 'script_version=%s\n' "$SCRIPT_VERSION"
  printf 'created_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'host=%s\n' "$(hostname 2>/dev/null || printf unknown)"
  printf 'user=%s\n' "$(id -un 2>/dev/null || printf unknown)"
  printf 'uid=%s\n' "$(id -u 2>/dev/null || printf unknown)"
  printf 'cwd=%s\n' "$(pwd)"
  printf 'service_name=%s\n' "$SERVICE_NAME"
  printf 'base_url=%s\n' "$(redact_inline "$BASE_URL")"
  printf 'auth_available=%s\n' "$AUTH_AVAILABLE"
  printf 'auth_source=%s\n' "$AUTH_SOURCE"
  printf 'snapshot_count=%s\n' "$SNAPSHOT_COUNT"
  printf 'journal_lines=%s\n' "$JOURNAL_LINES"
  printf 'connect_timeout=%s\n' "$CONNECT_TIMEOUT"
  printf 'max_time=%s\n' "$MAX_TIME"
} >"$OUT_DIR/metadata.txt"

{
  printf '# Env sources\n'
  for note in "${ENV_SOURCE_NOTES[@]}"; do
    printf '%s\n' "$note"
  done
  printf '\n# Redacted env summary\n'
  for key in "${!ENV_VALUES[@]}"; do
    if [[ "$key" =~ ^(EVA|EVOSSEARCH|VLLM|CUDA|HF|HUGGINGFACE|TRANSFORMERS|TOKENIZERS|OPENAI|LM_STUDIO|LMSTUDIO|PYTHON|GUNICORN|UVICORN|POSTGRES|PG)[A-Za-z0-9_]*$ ]]; then
      printf '%s=%s\t# sources: %s\n' "$key" "$(redact_env_value "$key" "${ENV_VALUES[$key]}")" "${ENV_SOURCES[$key]}"
    fi
  done | sort
} >"$OUT_DIR/env/redacted_env_summary.txt"

if have systemctl; then
  run_capture "$OUT_DIR/system/systemctl-status-${SERVICE_NAME}.txt" systemctl status "$SERVICE_NAME" --no-pager -l
  run_capture "$OUT_DIR/system/systemctl-show-${SERVICE_NAME}-redacted.txt" systemctl show "$SERVICE_NAME" --no-pager
  run_capture "$OUT_DIR/system/systemctl-cat-${SERVICE_NAME}-redacted.txt" systemctl cat "$SERVICE_NAME" --no-pager
else
  write_command_unavailable "$OUT_DIR/system/systemctl-status-${SERVICE_NAME}.txt" "systemctl"
fi

if have journalctl; then
  run_capture "$OUT_DIR/system/journalctl-${SERVICE_NAME}-last.txt" journalctl -u "$SERVICE_NAME" -n "$JOURNAL_LINES" --no-pager -o short-iso
  run_capture "$OUT_DIR/system/journalctl-${SERVICE_NAME}-warnings.txt" journalctl -u "$SERVICE_NAME" -p warning -n "$JOURNAL_LINES" --no-pager -o short-iso
else
  write_command_unavailable "$OUT_DIR/system/journalctl-${SERVICE_NAME}-last.txt" "journalctl"
fi

run_capture "$OUT_DIR/system/uname.txt" uname -a
if have uptime; then
  run_capture "$OUT_DIR/system/uptime.txt" uptime
fi
if have df; then
  run_capture "$OUT_DIR/system/df-h.txt" df -h
fi
if have free; then
  run_capture "$OUT_DIR/system/free-h.txt" free -h
fi
if have ss; then
  run_capture "$OUT_DIR/system/ss-listeners.txt" ss -ltnp
fi
if have nvidia-smi; then
  run_capture "$OUT_DIR/system/nvidia-smi.txt" nvidia-smi
fi

curl_get_endpoint "/health" "$OUT_DIR/api/health.txt" "noauth"
curl_get_endpoint "/ready" "$OUT_DIR/api/ready.txt" "noauth"
curl_get_endpoint "/lm/models" "$OUT_DIR/api/lm-models.txt" "noauth"

channels_body="$TMP_DIR/channels.json"
channels_path="/luxriot/channels"
if [[ "$FORCE_CHANNELS" == "1" ]]; then
  channels_path="/luxriot/channels?force=1"
fi

if [[ "$AUTH_AVAILABLE" == "1" ]]; then
  curl_get_endpoint "/auth/me" "$OUT_DIR/api/auth-me.txt" "auth"
  curl_get_endpoint "$channels_path" "$OUT_DIR/api/luxriot-channels.txt" "auth" "$channels_body"
  curl_get_endpoint "/luxriot/streams" "$OUT_DIR/api/luxriot-streams.txt" "auth"

  {
    printf 'channel_id\thttp_code\tcurl_exit_status\tsize_bytes\tsha1\ttime_total_sec\tcontent_type\tx_image_width\tx_image_height\terror\n'
    if (( SNAPSHOT_COUNT > 0 )); then
      mapfile -t channel_ids < <(extract_channel_ids "$channels_body" | head -n "$SNAPSHOT_COUNT")
      if (( ${#channel_ids[@]} == 0 )); then
        printf '# no channel IDs parsed from /luxriot/channels response\n'
      else
        for channel_id in "${channel_ids[@]}"; do
          fetch_snapshot_digest "$channel_id"
        done
      fi
    fi
  } >"$OUT_DIR/snapshots/snapshot-digests.tsv"
else
  cat >"$OUT_DIR/api/AUTH_REQUIRED.txt" <<EOF
Authenticated cookie was not provided, so the script skipped:
- /auth/me
- /luxriot/channels
- /luxriot/streams
- /luxriot/snapshot/<channel_id> digest checks

Create a temporary cookie jar on the EVA AI host, then rerun:

  BASE_URL=$(redact_inline "$BASE_URL")
  read -rsp "EVA password: " EVA_PASSWORD; echo
  curl -sS -c /tmp/eva.cookies \\
    -H 'Content-Type: application/json' \\
    -X POST "\${BASE_URL}/auth/login" \\
    -d "{\\"username\\":\\"admin\\",\\"password\\":\\"\${EVA_PASSWORD}\\"}"
  unset EVA_PASSWORD
  EVA_BASE_URL="\${BASE_URL}" EVA_COOKIE_FILE=/tmp/eva.cookies scripts/client_diagnostics.sh
  rm -f /tmp/eva.cookies

Do not send the cookie jar or password to EVA AI support.
EOF
fi

vllm_index="$OUT_DIR/vllm/discovered-endpoints.tsv"
{
  printf 'env_key\tbase_url\n'
  collect_vllm_urls | while IFS=$'\t' read -r key value; do
    printf '%s\t%s\n' "$key" "$(redact_inline "$value")"
  done
} >"$vllm_index"

vllm_found=0
while IFS=$'\t' read -r key value; do
  [[ -n "${key:-}" && -n "${value:-}" ]] || continue
  vllm_found=1
  safe_key="$(safe_name "$key")"
  api_key="$(profile_api_key_for_base_key "$key")"
  curl_model_endpoint "$key" "$value" "$api_key" "$OUT_DIR/vllm/models-${safe_key}.txt"
done < <(collect_vllm_urls)

if [[ "$vllm_found" == "0" ]]; then
  cat >"$OUT_DIR/vllm/NO_ENDPOINTS_DISCOVERED.txt" <<'EOF'
No vLLM/OpenAI-compatible base URLs were discovered from the readable env.

Rerun as root if /etc/eva-ai/eva-ai.env is not readable, or pass endpoints explicitly:

  EVA_DIAG_VLLM_BASE_URLS='http://host-a:8001/v1,http://host-a:8002/v1' scripts/client_diagnostics.sh
EOF
fi

archive_path="${OUT_DIR}.tar.gz"
if have tar; then
  tar -C "$(dirname "$OUT_DIR")" -czf "$archive_path" "$(basename "$OUT_DIR")"
else
  archive_path="tar-unavailable; send directory ${OUT_DIR}"
fi

cat <<EOF
Diagnostics collected.
Directory: ${OUT_DIR}
Package:   ${archive_path}

Send the .tar.gz package only. Do not send cookie jars, passwords, or raw env files.
EOF
