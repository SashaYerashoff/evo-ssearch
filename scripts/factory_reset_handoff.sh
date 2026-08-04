#!/usr/bin/env bash
set -Eeuo pipefail

# Destructive, operator-authorized appliance handoff reset.
# Keeps schema, users, roles and model/runtime configuration; removes all
# collected evidence, summaries, probes, incidents, chats and home-site data.

APP_DIR="/opt/eva-ai"
SYSTEM_ENV="/etc/eva-ai/eva-ai.env"
PROJECT_ENV="${APP_DIR}/.env"
BACKUP_OWNER="admins"
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
if ! id "${BACKUP_OWNER}" >/dev/null 2>&1; then
  die "backup owner ${BACKUP_OWNER} does not exist"
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
backup_dir="/home/${BACKUP_OWNER}/eva-handoff-private-${stamp}"
install -d -m 0700 -o "${BACKUP_OWNER}" -g "${BACKUP_OWNER}" "${backup_dir}"

set -a
# This is a root-managed installation file produced by the EVA installer.
# shellcheck disable=SC1090
. "${SYSTEM_ENV}"
set +a

database_dsn="${EVA_MIGRATION_DATABASE_DSN:-${EVA_DATABASE_DSN:-${EVOSSEARCH_DATABASE_DSN:-${DATABASE_URL:-}}}}"
if [[ -z "${database_dsn}" ]]; then
  die "no PostgreSQL DSN was found in ${SYSTEM_ENV}"
fi

printf 'Creating final private backup in %s\n' "${backup_dir}"
pg_dump --format=custom --file="${backup_dir}/eva.pgcustom" "${database_dsn}"
cp --preserve=mode,timestamps "${SYSTEM_ENV}" "${backup_dir}/system.env"
if [[ -f "${PROJECT_ENV}" ]]; then
  cp --preserve=mode,timestamps "${PROJECT_ENV}" "${backup_dir}/application.env"
fi
if [[ -d "/home/${BACKUP_OWNER}/Pictures/Screenshots" ]]; then
  tar -C "/home/${BACKUP_OWNER}/Pictures" -czf "${backup_dir}/server-screenshots.tar.gz" Screenshots
fi
git -C "${APP_DIR}" bundle create "${backup_dir}/source.bundle" HEAD
sha256sum "${backup_dir}"/* > "${backup_dir}/SHA256SUMS"
chown -R "${BACKUP_OWNER}:${BACKUP_OWNER}" "${backup_dir}"
chmod -R go-rwx "${backup_dir}"

printf 'Stopping writers...\n'
systemctl stop eva-deep-review.service eva-ai.service

printf 'Removing operational PostgreSQL data while preserving IAM principals and schema...\n'
psql "${database_dsn}" -v ON_ERROR_STOP=1 <<'SQL'
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
    "${APP_DIR}"/*|/var/lib/eva-ai/*|"/home/${BACKUP_OWNER}/.local/share/eva-ai"/*)
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
    "${APP_DIR}"/*|/var/lib/eva-ai/*|"/home/${BACKUP_OWNER}/.local/share/eva-ai"/*)
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
chown "${BACKUP_OWNER}:${BACKUP_OWNER}" "${PROJECT_ENV}"
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
rm -rf -- "/home/${BACKUP_OWNER}/eva-backups"
rm -rf -- "/home/${BACKUP_OWNER}/eva-release-a49b71b"
if [[ -d "/home/${BACKUP_OWNER}/Pictures/Screenshots" ]]; then
  find "/home/${BACKUP_OWNER}/Pictures/Screenshots" -mindepth 1 -delete
fi
find /tmp -maxdepth 1 -type f \( -name 'eva*.pgcustom' -o -name 'eva*.dump' \) -delete
find /tmp -maxdepth 1 -type d -name 'eva-upgrade.*' -exec rm -rf -- {} +

desktop_dir="/home/${BACKUP_OWNER}/Desktop"
install -d -m 0755 -o "${BACKUP_OWNER}" -g "${BACKUP_OWNER}" "${desktop_dir}"
install -m 0644 -o "${BACKUP_OWNER}" -g "${BACKUP_OWNER}" \
  "${APP_DIR}/react-ui/dist/quick-start.html" \
  "${desktop_dir}/EVA AI Operator Quick Start.html"

printf 'Starting EVA services...\n'
systemctl start eva-ai.service eva-deep-review.service

for _attempt in $(seq 1 45); do
  if curl -fsS --max-time 2 http://127.0.0.1:5000/health >/dev/null; then
    break
  fi
  sleep 1
done
curl -fsS --max-time 5 http://127.0.0.1:5000/health >/dev/null \
  || die "EVA health endpoint did not recover"

report="/home/${BACKUP_OWNER}/eva-handoff-report.txt"
{
  printf 'EVA AI handoff reset: %s\n' "${stamp}"
  printf 'Source revision: %s\n' "$(git -C "${APP_DIR}" rev-parse --short HEAD)"
  printf 'Database migration: '
  psql "${database_dsn}" -Atqc 'SELECT version_num FROM alembic_version LIMIT 1'
  printf 'Preserved users: '
  psql "${database_dsn}" -Atqc 'SELECT count(*) FROM iam.users'
  printf 'Operational rows remaining: '
  psql "${database_dsn}" -Atqc "SELECT sum(n_live_tup)::bigint FROM pg_stat_user_tables WHERE schemaname IN ('archive','agent','jobs')"
  printf 'Health: OK\n'
  printf 'Private backup awaiting off-machine copy: %s\n' "${backup_dir}"
  printf 'Quick start: /home/%s/Desktop/EVA AI Operator Quick Start.html\n' "${BACKUP_OWNER}"
} > "${report}"
chown "${BACKUP_OWNER}:${BACKUP_OWNER}" "${report}"
chmod 0644 "${report}"

printf '\nHandoff reset completed.\n'
printf 'PRIVATE_BACKUP=%s\n' "${backup_dir}"
printf 'Copy that directory off the appliance, verify its SHA256SUMS, then remove it.\n'
cat "${report}"
