#!/usr/bin/env bash
set -Eeuo pipefail

EXPECTED_ARCH="aarch64"
DATABASE="eva"
INSTALL_ROOT="/opt/eva-ai"
DATA_ROOT="/var/lib/eva-ai"
CONFIG_ROOT="/etc/eva-ai"
STATE_ROOT="/var/lib/eva-ai-installer"
APT_CACHE_ROOT="/var/cache/eva-ai-offline-apt"
SITE_BACKUPS_ROOT="/var/backups/eva-ai"
BACKUP_ROOT="/var/backups/eva-ai-factory-reset-$(date +%Y%m%d-%H%M%S)"
SPARK_IMAGE_ID="sha256:5f79999e8001200efe1bacff71758a1ac459c83707f4ddab74311996863e17ba"

UNITS=(
  eva-ai.service
  eva-vllm.service
  eva-deep-review.service
  eva-vlm-vision-watchdog.timer
  eva-vlm-vision-watchdog.service
  eva-vlm-vision-recover.service
)

DB_ROLES=(
  eva_migrator_login
  eva_api_login
  eva_audit_login
  eva_worker_login
  eva_backup_login
  eva_migrator
  eva_api
  eva_audit_writer
  eva_worker
  eva_backup
  eva_agent_reader
  eva_owner
)

fail() {
  printf 'FACTORY RESET ERROR: %s\n' "$*" >&2
  exit 1
}

[[ "${EUID}" -eq 0 ]] || fail "run with sudo"
[[ "$(uname -m)" == "${EXPECTED_ARCH}" ]] \
  || fail "this reset is pinned to ARM64/Spark-class hosts"

# Never follow an unexpected symlink into another part of the machine.
for root in "${INSTALL_ROOT}" "${DATA_ROOT}" "${CONFIG_ROOT}" \
  "${STATE_ROOT}" "${APT_CACHE_ROOT}" "${SITE_BACKUPS_ROOT}"
do
  [[ ! -L "${root}" ]] || fail "refusing symlinked EVA root: ${root}"
done

if [[ -f "${STATE_ROOT}/install-state.json" ]]; then
  python3 - "${STATE_ROOT}/install-state.json" <<'PY'
import json
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
target = state.get("target") or {}
expected = {
    "install_root": "/opt/eva-ai",
    "data_root": "/var/lib/eva-ai",
    "config_root": "/etc/eva-ai",
}
actual = {key: str(target.get(key) or "") for key in expected}
for key, value in actual.items():
    if value and value != expected[key]:
        raise SystemExit(
            f"installer journal uses unexpected {key}={value}; refusing factory reset"
        )
PY
fi

printf '\nEVA AI ARM64 FACTORY RESET\n'
printf '  application: %s\n' "${INSTALL_ROOT}"
printf '  data:        %s\n' "${DATA_ROOT}"
printf '  config:      %s\n' "${CONFIG_ROOT}"
printf '  database:    local PostgreSQL/%s\n' "${DATABASE}"
printf '  quarantine:  %s\n\n' "${BACKUP_ROOT}"
printf '%s\n' 'Docker/PostgreSQL/NVIDIA packages and unrelated containers are preserved.'
printf '%s\n\n' 'EVA files and a database dump are quarantined rather than irreversibly erased.'
read -r -p 'Type FACTORY RESET EVA to continue: ' confirmation
[[ "${confirmation}" == "FACTORY RESET EVA" ]] \
  || fail "confirmation did not match"

install -d -m 0711 "${BACKUP_ROOT}"
install -d -m 0700 "${BACKUP_ROOT}/systemd" "${BACKUP_ROOT}/nginx"
install -d -o postgres -g postgres -m 0700 "${BACKUP_ROOT}/postgres"

printf '\n[1/8] Stopping only EVA services and containers...\n'
systemctl disable --now "${UNITS[@]}" 2>/dev/null || true
if command -v docker >/dev/null 2>&1; then
  docker rm -f eva-ai-app eva-vllm 2>/dev/null || true
fi

command -v psql >/dev/null 2>&1 || fail "PostgreSQL client is absent; database state is unknown"
id postgres >/dev/null 2>&1 || fail "PostgreSQL service account is absent"
systemctl start postgresql.service
db_exists="$(runuser -u postgres -- psql -d postgres -Atqc \
  "SELECT 1 FROM pg_database WHERE datname='${DATABASE}'")" \
  || fail "could not inspect the local PostgreSQL cluster"
if [[ "${db_exists}" == "1" ]]; then
  printf '[2/8] Dumping and validating PostgreSQL database %s...\n' "${DATABASE}"
  runuser -u postgres -- pg_dump --format=custom \
    --file="${BACKUP_ROOT}/postgres/eva.dump" "${DATABASE}"
  runuser -u postgres -- pg_restore --list \
    "${BACKUP_ROOT}/postgres/eva.dump" >/dev/null
  chown root:root "${BACKUP_ROOT}/postgres/eva.dump"
  chmod 0600 "${BACKUP_ROOT}/postgres/eva.dump"
else
  printf '[2/8] Local PostgreSQL database %s is already absent.\n' "${DATABASE}"
fi

printf '[3/8] Quarantining EVA system integration files...\n'
for unit in "${UNITS[@]}"; do
  path="/etc/systemd/system/${unit}"
  if [[ -e "${path}" || -L "${path}" ]]; then
    cp -a "${path}" "${BACKUP_ROOT}/systemd/"
    rm -f -- "${path}"
  fi
done
if [[ -e /etc/nginx/sites-available/eva-ai ]]; then
  mv -- /etc/nginx/sites-available/eva-ai "${BACKUP_ROOT}/nginx/eva-ai.available"
fi
if [[ -e /etc/nginx/sites-enabled/eva-ai || -L /etc/nginx/sites-enabled/eva-ai ]]; then
  rm -f -- /etc/nginx/sites-enabled/eva-ai
fi
if [[ -e /etc/nginx/sites-available/default \
      && ! -e /etc/nginx/sites-enabled/default ]]; then
  ln -s ../sites-available/default /etc/nginx/sites-enabled/default
fi
if [[ -e /usr/local/sbin/eva-ai-doctor || -L /usr/local/sbin/eva-ai-doctor ]]; then
  mv -- /usr/local/sbin/eva-ai-doctor "${BACKUP_ROOT}/eva-ai-doctor"
fi

printf '[4/8] Quarantining EVA application, data, config and installer state...\n'
for entry in \
  "${INSTALL_ROOT}:install-root" \
  "${DATA_ROOT}:data-root" \
  "${CONFIG_ROOT}:config-root" \
  "${STATE_ROOT}:installer-state" \
  "${APT_CACHE_ROOT}:offline-apt-cache" \
  "${SITE_BACKUPS_ROOT}:site-backups"
do
  source_path="${entry%%:*}"
  backup_name="${entry#*:}"
  if [[ -e "${source_path}" ]]; then
    mv -- "${source_path}" "${BACKUP_ROOT}/${backup_name}"
  fi
done

if [[ "${db_exists}" == "1" ]]; then
  printf '[5/8] Dropping only local PostgreSQL database %s...\n' "${DATABASE}"
  runuser -u postgres -- psql -d postgres -v ON_ERROR_STOP=1 -c \
    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='${DATABASE}' AND pid <> pg_backend_pid();" \
    >/dev/null
  runuser -u postgres -- dropdb "${DATABASE}"
else
  printf '[5/8] Database already absent.\n'
fi

printf '[6/8] Removing EVA-only PostgreSQL roles when dependency-free...\n'
for role in "${DB_ROLES[@]}"; do
  runuser -u postgres -- psql -d postgres -v ON_ERROR_STOP=1 -c \
    "DROP ROLE IF EXISTS ${role};" >/dev/null 2>&1 || \
    printf '  retained role with external dependency: %s\n' "${role}"
done

printf '[7/8] Removing the exact EVA Spark runtime image tag...\n'
if command -v docker >/dev/null 2>&1; then
  docker image rm eva-ai/spark-runtime:0.8.7-arm64 "${SPARK_IMAGE_ID}" \
    2>/dev/null || true
fi

printf '[8/8] Reloading system integration...\n'
systemctl daemon-reload
systemctl reset-failed "${UNITS[@]}" 2>/dev/null || true
if command -v nginx >/dev/null 2>&1; then
  nginx -t
  systemctl reload nginx.service 2>/dev/null || true
fi

if id eva >/dev/null 2>&1; then
  userdel eva 2>/dev/null || true
fi
if getent group eva >/dev/null 2>&1; then
  groupdel eva 2>/dev/null || true
fi

cat >"${BACKUP_ROOT}/RESET_RECEIPT.txt" <<EOF
EVA AI ARM64 factory reset completed at $(date --iso-8601=seconds).
The prior EVA filesystem and a validated database dump were quarantined here.
System Docker, PostgreSQL, NVIDIA packages, and unrelated containers were preserved.
EOF
chmod 0600 "${BACKUP_ROOT}/RESET_RECEIPT.txt"
chmod 0700 "${BACKUP_ROOT}"

printf '\nEVA AI FACTORY RESET COMPLETE\n'
printf 'Quarantine: %s\n' "${BACKUP_ROOT}"
printf '%s\n' 'The universal offline installer should now select fresh INSTALL mode.'
