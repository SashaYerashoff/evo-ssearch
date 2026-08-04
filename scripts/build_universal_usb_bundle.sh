#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STAGING_ROOT="${1:-/mnt/eva-llamacpp-lab/universal-usb-staging}"

export EVA_PORT_EXPECTED_BRANCH="${EVA_UNIVERSAL_EXPECTED_BRANCH:-feature/universal-offline-deploy}"
export EVA_PORT_RELEASE_FLAVOR="universal-offline"

"${SCRIPT_DIR}/build_port_usb_bundle.sh" "${STAGING_ROOT}"

ALEMBIC_BIN="${EVA_UNIVERSAL_ALEMBIC_BIN:-${REPO_ROOT}/.venv/bin/alembic}"
if [[ ! -x "${ALEMBIC_BIN}" ]]; then
  printf 'ERROR: Alembic is required to validate the 0006 -> head offline migration plan: %s\n' \
    "${ALEMBIC_BIN}" >&2
  exit 1
fi
mkdir -p "${STAGING_ROOT}/migration-plans"
EVA_DATABASE_DSN='postgresql://offline-validation/unused' \
  "${ALEMBIC_BIN}" -c "${REPO_ROOT}/alembic.ini" \
  upgrade 20260614_0006:20260801_0011 --sql \
  > "${STAGING_ROOT}/migration-plans/0006-to-0011.sql"
grep -Fq "version_num='20260801_0011'" \
  "${STAGING_ROOT}/migration-plans/0006-to-0011.sql" || {
    printf 'ERROR: generated migration plan does not reach 20260801_0011.\n' >&2
    exit 1
  }

cat <<EOF

Universal EVA AI staging tree created at:
  ${STAGING_ROOT}

Next steps:
  1. Populate/verify apt/ and wheelhouse/ for Ubuntu 24.04 amd64.
  2. Run:
       python3 ${REPO_ROOT}/scripts/finalize_port_usb_bundle.py ${STAGING_ROOT}
  3. Copy the finalized directory to the field USB.

Field entry point:
  sudo ./START_EVA_AI.sh
EOF
