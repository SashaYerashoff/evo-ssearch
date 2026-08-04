#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/eva_offline_deploy.py" ]]; then
  ENTRYPOINT="${SCRIPT_DIR}/eva_offline_deploy.py"
elif [[ -f "${SCRIPT_DIR}/repo/scripts/eva_offline_deploy.py" ]]; then
  ENTRYPOINT="${SCRIPT_DIR}/repo/scripts/eva_offline_deploy.py"
else
  printf 'ERROR: eva_offline_deploy.py is missing from this bundle.\n' >&2
  exit 1
fi

if [[ "${EUID}" -ne 0 ]]; then
  exec sudo --preserve-env=TERM,EVA_INSTALL_MIGRATION_DSN python3 "${ENTRYPOINT}" "$@"
fi
exec python3 "${ENTRYPOINT}" "$@"
