#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STAGING_ROOT="${1:-/mnt/eva-llamacpp-lab/universal-usb-staging}"
DEPENDENCY_SEED="${EVA_UNIVERSAL_DEPENDENCY_SEED:-/mnt/eva-llamacpp-lab/universal-usb-staging}"
ALEMBIC_BIN="${EVA_UNIVERSAL_ALEMBIC_BIN:-}"

if [[ -z "${ALEMBIC_BIN}" ]]; then
  for candidate in \
    "${REPO_ROOT}/.venv/bin/alembic" \
    "$(command -v alembic 2>/dev/null || true)"; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      ALEMBIC_BIN="${candidate}"
      break
    fi
  done
fi
if [[ -z "${ALEMBIC_BIN}" ]]; then
  while IFS= read -r candidate; do
    if [[ -x "${candidate}" ]]; then
      ALEMBIC_BIN="${candidate}"
      break
    fi
  done < <(find "$(dirname "${REPO_ROOT}")" -maxdepth 4 -path '*/.venv/bin/alembic' -type f -print 2>/dev/null | sort)
fi
if [[ -z "${ALEMBIC_BIN}" || ! -x "${ALEMBIC_BIN}" ]]; then
  printf 'ERROR: Alembic is required before payload copying begins.\n' >&2
  printf 'Set EVA_UNIVERSAL_ALEMBIC_BIN=/path/to/.venv/bin/alembic.\n' >&2
  exit 1
fi

# A universal artifact is identified by its clean source commit and checksums,
# not by a long-lived client branch name.  The client-specific builder keeps its
# stricter branch guard.
export EVA_PORT_ALLOW_OTHER_BRANCH=1
export EVA_PORT_RELEASE_FLAVOR="universal-offline"

if [[ ! -f "${DEPENDENCY_SEED}/apt/Packages.gz" || ! -d "${DEPENDENCY_SEED}/wheelhouse" ]]; then
  printf 'ERROR: dependency seed is incomplete: %s\n' "${DEPENDENCY_SEED}" >&2
  printf 'Set EVA_UNIVERSAL_DEPENDENCY_SEED to a reviewed Ubuntu 24.04 amd64 cache.\n' >&2
  exit 1
fi

# If the selected seed is the staging tree itself, preserve it while the base
# payload is refreshed.  Otherwise synchronize it so stale packages cannot
# survive between releases.
if [[ "$(readlink -f "${DEPENDENCY_SEED}")" != "$(readlink -f "${STAGING_ROOT}")" ]]; then
  mkdir -p "${STAGING_ROOT}/apt" "${STAGING_ROOT}/wheelhouse"
  rsync -a --delete "${DEPENDENCY_SEED}/apt/" "${STAGING_ROOT}/apt/"
  rsync -a --delete "${DEPENDENCY_SEED}/wheelhouse/" "${STAGING_ROOT}/wheelhouse/"
fi

"${SCRIPT_DIR}/build_port_usb_bundle.sh" "${STAGING_ROOT}"

python3 "${SCRIPT_DIR}/offline_bundle_dependencies.py" \
  "${STAGING_ROOT}" \
  --repo-root "${STAGING_ROOT}/repo" \
  --write-manifest \
  --resolve

mkdir -p "${STAGING_ROOT}/migration-plans"
EVA_DATABASE_DSN='postgresql://offline-validation/unused' \
  "${ALEMBIC_BIN}" -c "${REPO_ROOT}/alembic.ini" \
  upgrade 20260614_0006:20260805_0013 --sql \
  > "${STAGING_ROOT}/migration-plans/0006-to-0013.sql"
grep -Fq "version_num='20260805_0013'" \
  "${STAGING_ROOT}/migration-plans/0006-to-0013.sql" || {
    printf 'ERROR: generated migration plan does not reach 20260805_0013.\n' >&2
    exit 1
  }

cat <<EOF

Universal EVA AI staging tree created at:
  ${STAGING_ROOT}

Next steps:
  1. Review the source revision and acceptance output above.
  2. Finalize a clean committed release with:
       python3 ${REPO_ROOT}/scripts/finalize_port_usb_bundle.py ${STAGING_ROOT}
  3. Copy the finalized directory to the field USB.

Field entry point:
  sudo ./START_EVA_AI.sh
EOF
