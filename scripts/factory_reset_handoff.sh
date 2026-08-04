#!/usr/bin/env bash
set -Eeuo pipefail

# Destructive, operator-authorized appliance handoff reset.
# Keeps schema, users, roles and model/runtime configuration; removes all
# collected evidence, summaries, probes, incidents, chats and home-site data.
# No database or runtime backup is retained on the appliance.

APP_DIR="/opt/eva-ai"
SYSTEM_ENV="/etc/eva-ai/eva-ai.env"
PROJECT_ENV="${APP_DIR}/.env"
SERVICE_USER="admins"
CONFIRMATION="--yes-clean-operational-data"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

if [[ ${EUID} -ne 0 ]]; then
  die "run with sudo"
fi
if [[ "${1:-}" != "${CONFIRMATION}" ]]; then
  die "refusing destructive reset; pass ${CONFIRMATION}"
fi
if [[ ! -d "${APP_DIR}/.git" || ! -f "${APP_DIR}/oldapp.py" ]]; then
  die "${APP_DIR} is not the expected EVA AI checkout"
fi
if [[ ! -f "${SYSTEM_ENV}" ]]; then
  die "missing ${SYSTEM_ENV}"
fi
if ! id "${SERVICE_USER}" >/dev/null 2>&1; then
  die "service user ${SERVICE_USER} does not exist"
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"

set -a
# This is a root-managed installation file produced by the EVA installer.
# shellcheck disable=SC1090
. "${SYSTEM_ENV}"
set +a

database_dsn="${EVA_MIGRATION_DATABASE_DSN:-${EVA_DATABASE_DSN:-${EVOSSEARCH_DATABASE_DSN:-${DATABASE_URL:-}}}}"
if [[ -z "${database_dsn}" ]]; then
  die "no PostgreSQL DSN was found in ${SYSTEM_ENV}"
fi

database_name="$(DATABASE_DSN="${database_dsn}" python3 <<'PY'
import os
import shlex
from urllib.parse import unquote, urlsplit

dsn = os.environ["DATABASE_DSN"].strip()
if "://" in dsn:
    name = unquote(urlsplit(dsn).path.lstrip("/").split("/", 1)[0])
else:
    values = {}
    for token in shlex.split(dsn):
        if "=" in token:
            key, value = token.split("=", 1)
            values[key.strip()] = value.strip()
    name = values.get("dbname") or values.get("database") or ""
if not name or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in name):
    raise SystemExit("could not safely determine the PostgreSQL database name")
print(name)
PY
)"

printf 'Stopping writers...\n'
systemctl stop eva-deep-review.service eva-ai.service

printf 'Removing operational PostgreSQL data while preserving IAM principals and schema...\n'
runuser -u postgres -- psql --dbname="${database_name}" -v ON_ERROR_STOP=1 <<'SQL'
DO $handoff$
DECLARE
    targets text;
BEGIN
    SELECT string_agg(format('%I.%I', schemaname, tablename), ', ' ORDER BY schemaname, tablename)
      INTO targets
      FROM pg_tables
     WHERE schemaname IN ('archive', 'agent', 'jobs', 'audit');
    IF targets IS NOT NULL THEN
        EXECUTE 'TRUNCATE TABLE ' || targets || ' RESTART IDENTITY CASCADE';
    END IF;
END
$handoff$;

TRUNCATE TABLE
    iam.sessions,
    iam.login_attempts,
    iam.user_channel_grants
RESTART IDENTITY CASCADE;
SQL

safe_empty_dir() {
  local raw_path="$1"
  local resolved
  [[ -n "${raw_path}" ]] || return 0
  if [[ "${raw_path}" != /* ]]; then
    raw_path="${APP_DIR}/${raw_path}"
  fi
  resolved="$(realpath -m -- "${raw_path}")"
  case "${resolved}" in
    "${APP_DIR}"/*|/var/lib/eva-ai/*|"/home/${SERVICE_USER}/.local/share/eva-ai"/*)
      if [[ -d "${resolved}" ]]; then
        find "${resolved}" -mindepth 1 -delete
      fi
      ;;
    *)
      die "refusing to empty unexpected runtime path: ${resolved}"
      ;;
  esac
}

safe_remove_file() {
  local raw_path="$1"
  local resolved
  [[ -n "${raw_path}" ]] || return 0
  if [[ "${raw_path}" != /* ]]; then
    raw_path="${APP_DIR}/${raw_path}"
  fi
  resolved="$(realpath -m -- "${raw_path}")"
  case "${resolved}" in
    "${APP_DIR}"/*|/var/lib/eva-ai/*|"/home/${SERVICE_USER}/.local/share/eva-ai"/*)
      rm -f -- "${resolved}"
      ;;
    *)
      die "refusing to remove unexpected runtime file: ${resolved}"
      ;;
  esac
}

printf 'Removing file-backed runtime state...\n'
safe_empty_dir "${EVOSSEARCH_DETECTIONS_ARCHIVE_DIR:-detections_archive}"
safe_empty_dir "${EVOSSEARCH_INFERENCE_QUEUE_SPOOL_DIR:-inference-spool}"
safe_remove_file "${EVOSSEARCH_PROBE_CHANNEL_GROUPS_FILE:-probe_channel_groups.json}"
safe_remove_file "${EVOSSEARCH_LUXRIOT_SUMMARY_STATE_FILE:-luxriot_summary_state.json}"
safe_remove_file "${EVOSSEARCH_LUXRIOT_ROLLUP_CACHE_FILE:-luxriot_rollups_cache.json}"
safe_remove_file "${EVOSSEARCH_LM_VISION_HEALTH_STATE_FILE:-vlm-vision-health.json}"
safe_remove_file "probes_store.json"

printf 'Creating a sanitized, UI-writable canonical service environment...\n'
python3 - "${SYSTEM_ENV}" "${PROJECT_ENV}" <<'PY'
from pathlib import Path
import re
import sys

source = Path(sys.argv[1])
target = Path(sys.argv[2])
updates = {
    "EVOSSEARCH_LUXRIOT_BASE_URL": "http://127.0.0.1:8080",
    "EVOSSEARCH_LUXRIOT_USERNAME": "",
    "EVOSSEARCH_LUXRIOT_PASSWORD": "",
    "EVOSSEARCH_LUXRIOT_DEFAULT_CHANNEL_ID": "1",
    "EVOSSEARCH_LUXRIOT_ALERT_POLICY_PROMPT": "",
    "EVOSSEARCH_LUXRIOT_SYSTEM_PROMPT_DEFAULT": "",
    "EVOSSEARCH_LOCAL_VIDEO_SOURCES": "",
}
lines = source.read_text(encoding="utf-8").splitlines()
seen = set()
rendered = []
for line in lines:
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=", line)
    if match and match.group(1) in updates:
        key = match.group(1)
        rendered.append(f'{key}="{updates[key]}"')
        seen.add(key)
    else:
        rendered.append(line)
for key, value in updates.items():
    if key not in seen:
        rendered.append(f'{key}="{value}"')
target.write_text("\n".join(rendered) + "\n", encoding="utf-8")
PY
chown "${SERVICE_USER}:${SERVICE_USER}" "${PROJECT_ENV}"
chmod 0600 "${PROJECT_ENV}"

# Make the same sanitized values visible in the retained root copy so an old
# service snapshot cannot reconnect to the office Evo by accident.
install -m 0600 -o root -g root "${PROJECT_ENV}" "${SYSTEM_ENV}"

dropin_dir="/etc/systemd/system/eva-ai.service.d"
install -d -m 0755 "${dropin_dir}"
cat > "${dropin_dir}/10-ui-managed-env.conf" <<EOF
[Service]
EnvironmentFile=
EnvironmentFile=${PROJECT_ENV}
Environment=EVOSSEARCH_CONFIG_ENV_FILE=${PROJECT_ENV}
EOF
chmod 0644 "${dropin_dir}/10-ui-managed-env.conf"
systemctl daemon-reload

printf 'Removing private, superseded server-side copies...\n'
rm -rf -- "/home/${SERVICE_USER}/eva-backups"
rm -rf -- "/home/${SERVICE_USER}/eva-release-a49b71b"
find "/home/${SERVICE_USER}" -maxdepth 1 -type d -name 'eva-handoff-private-*' -exec rm -rf -- {} +
if [[ -d "/home/${SERVICE_USER}/Pictures/Screenshots" ]]; then
  find "/home/${SERVICE_USER}/Pictures/Screenshots" -mindepth 1 -delete
fi
find /tmp -maxdepth 1 -type f \( -name 'eva*.pgcustom' -o -name 'eva*.dump' \) -delete
find /tmp -maxdepth 1 -type d -name 'eva-upgrade.*' -exec rm -rf -- {} +

# This workstation was used only for appliance development and testing. Clear
# cached EVA frames, browser histories and localStorage so a handed-over GUI
# cannot resurrect private thumbnails, chat/session selectors or channel IDs.
for browser_path in \
  "/home/${SERVICE_USER}/.cache/chromium" \
  "/home/${SERVICE_USER}/.cache/google-chrome" \
  "/home/${SERVICE_USER}/.cache/mozilla"; do
  if [[ -d "${browser_path}" ]]; then
    find "${browser_path}" -mindepth 1 -delete
  fi
done
for chromium_profile in \
  "/home/${SERVICE_USER}/.config/chromium/Default" \
  "/home/${SERVICE_USER}/.config/google-chrome/Default"; do
  if [[ -d "${chromium_profile}" ]]; then
    rm -rf -- \
      "${chromium_profile}/Cache" \
      "${chromium_profile}/Code Cache" \
      "${chromium_profile}/Service Worker" \
      "${chromium_profile}/IndexedDB" \
      "${chromium_profile}/Local Storage" \
      "${chromium_profile}/Session Storage" \
      "${chromium_profile}/History" \
      "${chromium_profile}/History-journal"
  fi
done
rm -f -- \
  "/home/${SERVICE_USER}/.bash_history" \
  "/home/${SERVICE_USER}/.local/share/recently-used.xbel"

desktop_dir="/home/${SERVICE_USER}/Desktop"
install -d -m 0755 -o "${SERVICE_USER}" -g "${SERVICE_USER}" "${desktop_dir}"
install -m 0644 -o "${SERVICE_USER}" -g "${SERVICE_USER}" \
  "${APP_DIR}/react-ui/dist/quick-start.html" \
  "${desktop_dir}/EVA AI Operator Quick Start.html"
rm -f -- "${desktop_dir}/RUN EVA CLEAN HANDOFF.txt"

printf 'Starting EVA services...\n'
systemctl start eva-ai.service eva-deep-review.service

for _attempt in $(seq 1 45); do
  if python3 - <<'PY' >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://127.0.0.1:5000/health", timeout=2).read()
PY
  then
    break
  fi
  sleep 1
done
python3 - <<'PY' >/dev/null || die "EVA health endpoint did not recover"
import urllib.request
urllib.request.urlopen("http://127.0.0.1:5000/health", timeout=5).read()
PY

report="/home/${SERVICE_USER}/eva-handoff-report.txt"
{
  printf 'EVA AI handoff reset: %s\n' "${stamp}"
  printf 'Source revision: %s\n' "$(git -C "${APP_DIR}" rev-parse --short HEAD)"
  printf 'Database migration: '
  runuser -u postgres -- psql --dbname="${database_name}" -Atqc 'SELECT version_num FROM alembic_version LIMIT 1'
  printf 'Preserved users: '
  runuser -u postgres -- psql --dbname="${database_name}" -Atqc 'SELECT count(*) FROM iam.users'
  printf 'Operational rows remaining: '
  runuser -u postgres -- psql --dbname="${database_name}" -Atqc "SELECT coalesce(sum(n_live_tup), 0)::bigint FROM pg_stat_user_tables WHERE schemaname IN ('archive','agent','jobs')"
  printf 'Health: OK\n'
  printf 'Server-side private backups: none\n'
  printf 'Quick start: /home/%s/Desktop/EVA AI Operator Quick Start.html\n' "${SERVICE_USER}"
} > "${report}"
chown "${SERVICE_USER}:${SERVICE_USER}" "${report}"
chmod 0644 "${report}"

printf '\nHandoff reset completed.\n'
cat "${report}"
