#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALLER="${SCRIPT_DIR}/install_port_appliance.py"

if [[ ! -f "${INSTALLER}" ]]; then
    echo "ERROR: install_port_appliance.py is missing beside this launcher." >&2
    exit 1
fi

if [[ "${EUID}" -ne 0 && " $* " != *" --dry-run "* ]]; then
    echo "EVA AI needs administrator access to install PostgreSQL, systemd services,"
    echo "the offline Python environments, models, and the TLS reverse proxy."
    exec sudo --preserve-env=TERM python3 "${INSTALLER}" "$@"
fi

exec python3 "${INSTALLER}" "$@"
