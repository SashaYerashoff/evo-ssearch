#!/usr/bin/env bash
set -Eeuo pipefail

FACTORY_ROOT="/mnt/eva-llamacpp-lab/factory-x64"
BACKUP_PARENT="/mnt/eva-llamacpp-lab"
DATABASE="eva"
STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP_ROOT="${BACKUP_PARENT}/factory-x64-retired-${STAMP}"
INSTALLER_STATE_ROOT="/var/lib/eva-ai-installer"

UNITS=(
  eva-ai.service
  eva-vllm.service
  eva-deep-review.service
  eva-vlm-vision-watchdog.timer
  eva-vlm-vision-watchdog.service
  eva-vlm-vision-recover.service
)

fail() {
  printf 'RESET ERROR: %s\n' "$*" >&2
  exit 1
}

if [[ "${EUID}" -ne 0 ]]; then
  fail "run this script with sudo"
fi

[[ "${FACTORY_ROOT}" == "/mnt/eva-llamacpp-lab/factory-x64" ]] \
  || fail "factory-root safety guard failed"
[[ -d "${FACTORY_ROOT}" ]] || fail "factory root does not exist: ${FACTORY_ROOT}"
[[ -d "${BACKUP_PARENT}" ]] || fail "backup parent does not exist: ${BACKUP_PARENT}"
[[ ! -e "${BACKUP_ROOT}" ]] || fail "backup target already exists: ${BACKUP_ROOT}"

unit_workdir="$(systemctl show eva-ai.service --property=WorkingDirectory --value 2>/dev/null || true)"
case "${unit_workdir}" in
  "${FACTORY_ROOT}"/*) ;;
  *)
    fail "eva-ai.service does not belong to the factory target (WorkingDirectory=${unit_workdir:-missing})"
    ;;
esac

db_exists="$(runuser -u postgres -- psql -Atqc \
  "SELECT 1 FROM pg_database WHERE datname = '${DATABASE}'" 2>/dev/null || true)"
[[ "${db_exists}" == "1" ]] || fail "factory database '${DATABASE}' was not found"

schema="$(runuser -u postgres -- psql -d "${DATABASE}" -Atqc \
  'SELECT version_num FROM alembic_version LIMIT 1' 2>/dev/null || true)"

printf '\nThis retires ONLY the disposable x64 factory appliance.\n'
printf '  filesystem: %s\n' "${FACTORY_ROOT}"
printf '  database:   %s (schema %s)\n' "${DATABASE}" "${schema:-unknown}"
printf '  backup:     %s\n' "${BACKUP_ROOT}"
printf '  preserved:  Georgia repro, development trees, and every other PostgreSQL database\n\n'
read -r -p 'Type RESET FACTORY X64 to continue: ' confirmation
[[ "${confirmation}" == "RESET FACTORY X64" ]] || fail "confirmation did not match"

# PostgreSQL writes the dump as its own unprivileged account.  The temporary
# execute bits let it traverse the root-owned backup directory; the directory
# is closed back to 0700 after the dump and filesystem retirement finish.
install -d -m 0711 "${BACKUP_ROOT}"
install -d -m 0700 "${BACKUP_ROOT}/systemd" "${BACKUP_ROOT}/nginx"
install -d -o postgres -g postgres -m 0700 "${BACKUP_ROOT}/postgres"

printf '\n[1/6] Stopping only factory EVA services...\n'
systemctl stop "${UNITS[@]}" 2>/dev/null || true
systemctl disable eva-ai.service eva-vllm.service eva-deep-review.service \
  eva-vlm-vision-watchdog.timer 2>/dev/null || true

printf '[2/6] Taking a recoverable PostgreSQL dump of database %s...\n' "${DATABASE}"
runuser -u postgres -- pg_dump --format=custom --file="${BACKUP_ROOT}/postgres/eva.dump" "${DATABASE}"
runuser -u postgres -- pg_restore --list "${BACKUP_ROOT}/postgres/eva.dump" >/dev/null
chown root:root "${BACKUP_ROOT}/postgres/eva.dump"
chmod 0600 "${BACKUP_ROOT}/postgres/eva.dump"

printf '[3/6] Saving exact system integration files...\n'
for unit in "${UNITS[@]}"; do
  unit_path="/etc/systemd/system/${unit}"
  if [[ -e "${unit_path}" || -L "${unit_path}" ]]; then
    cp -a "${unit_path}" "${BACKUP_ROOT}/systemd/"
    rm -f -- "${unit_path}"
  fi
done
if [[ -e /etc/nginx/sites-enabled/eva-ai || -L /etc/nginx/sites-enabled/eva-ai ]]; then
  cp -a /etc/nginx/sites-enabled/eva-ai "${BACKUP_ROOT}/nginx/eva-ai.enabled"
  rm -f -- /etc/nginx/sites-enabled/eva-ai
fi
if [[ -e /etc/nginx/sites-available/eva-ai ]]; then
  cp -a /etc/nginx/sites-available/eva-ai "${BACKUP_ROOT}/nginx/eva-ai.available"
  rm -f -- /etc/nginx/sites-available/eva-ai
fi

printf '[4/6] Saving installer receipt and factory filesystem...\n'
if [[ -e "${INSTALLER_STATE_ROOT}" ]]; then
  mv -- "${INSTALLER_STATE_ROOT}" "${BACKUP_ROOT}/installer-state"
fi
mv -- "${FACTORY_ROOT}" "${BACKUP_ROOT}/factory-x64"

printf '[5/6] Removing only the disposable database %s...\n' "${DATABASE}"
runuser -u postgres -- dropdb "${DATABASE}"

printf '[6/6] Reloading service manager and nginx...\n'
systemctl daemon-reload
systemctl reset-failed "${UNITS[@]}" 2>/dev/null || true
nginx -t
systemctl reload nginx.service 2>/dev/null || true

cat >"${BACKUP_ROOT}/RESTORE.txt" <<EOF
Factory x64 appliance retired at ${STAMP}.

The old filesystem, unit files, nginx site, installer state, and a verified
custom-format dump of PostgreSQL database '${DATABASE}' are stored here.

Do not restore this backup over a working replacement installation.  Stop the
replacement and inspect the paths first.
EOF
chmod 0600 "${BACKUP_ROOT}/RESTORE.txt"
chmod 0700 "${BACKUP_ROOT}"

printf '\nFACTORY X64 RESET COMPLETE\n'
printf 'Backup: %s\n' "${BACKUP_ROOT}"
printf 'The new offline bundle should now detect a fresh INSTALL target.\n'
