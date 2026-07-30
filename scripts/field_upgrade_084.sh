#!/usr/bin/env bash
# EVA AI 0.8.4 guided field upgrade.
#
# One entrypoint for the field engineer: verifies it runs from the unpacked
# bundle, snapshots the current service state, checks the DB schema version
# read-only, refuses to migrate anything by itself, dry-runs the offline
# installer, asks for one explicit confirmation, applies, verifies, and
# writes the exact rollback command to a file.
#
# The privileged migration path is deliberately NOT reachable from this
# script: if the database schema is not already at the expected head, the
# script stops and tells the engineer to call the responsible developer.

set -Eeuo pipefail

EXPECTED_SCHEMA="20260614_0006"
EXPECTED_VERSION="β 0.8.4"
APP_DIR="/opt/eva-ai/evo-ssearch"
ENV_FILE="/etc/eva-ai/eva-ai.env"
SERVICE_NAME="eva-ai"
BASE_URL="http://127.0.0.1:5000"

usage() {
  cat <<'USAGE'
Usage: sudo ./scripts/field_upgrade_084.sh [options]

Options:
  --app-dir DIR       Target application directory (default /opt/eva-ai/evo-ssearch)
  --env-file FILE     Site env file (default /etc/eva-ai/eva-ai.env)
  --service NAME      systemd service name (default eva-ai)
  --base-url URL      Local health URL (default http://127.0.0.1:5000)
  -h, --help          Show this help.
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --app-dir) APP_DIR="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --service) SERVICE_NAME="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

say()  { printf '\n== %s\n' "$*"; }
ok()   { printf 'OK: %s\n' "$*"; }
fail() { printf 'STOP: %s\n' "$*" >&2; }

stop_and_call() {
  fail "$1"
  printf '\nDo NOT continue. Call the responsible developer and read the STOP line above.\n' >&2
  printf 'Diagnostics to send: bash scripts/client_diagnostics.sh > diag.txt\n' >&2
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUNDLE_DIR="$(cd "${REPO_DIR}/.." && pwd)"

say "EVA AI 0.8.4 guided upgrade"

if [ "$(id -u)" -ne 0 ]; then
  stop_and_call "run with sudo: sudo ./scripts/field_upgrade_084.sh"
fi
for required_command in cmp curl find grep sed systemctl tee; do
  if ! command -v "${required_command}" >/dev/null 2>&1; then
    stop_and_call "required command is missing: ${required_command}"
  fi
done

# 1. We must be inside an unpacked offline bundle, not a random checkout.
if [ ! -f "${BUNDLE_DIR}/manifest.txt" ]; then
  stop_and_call "manifest.txt was not found next to repo/; this is not an unpacked bundle"
fi
BUNDLE_VERSION="$(tr -d '\r\n' < "${REPO_DIR}/VERSION" 2>/dev/null || true)"
MANIFEST_VERSION="$(sed -n 's/^version=//p' "${BUNDLE_DIR}/manifest.txt" | tail -n 1)"
MANIFEST_TREE_STATUS="$(sed -n 's/^working_tree_status=//p' "${BUNDLE_DIR}/manifest.txt" | tail -n 1)"
MANIFEST_COMMIT="$(sed -n 's/^git_commit=//p' "${BUNDLE_DIR}/manifest.txt" | tail -n 1)"
if [ "${BUNDLE_VERSION}" != "${EXPECTED_VERSION}" ]; then
  stop_and_call "bundle VERSION='${BUNDLE_VERSION:-missing}', expected '${EXPECTED_VERSION}'"
fi
if [ "${MANIFEST_VERSION}" != "${EXPECTED_VERSION}" ]; then
  stop_and_call "manifest version='${MANIFEST_VERSION:-missing}', expected '${EXPECTED_VERSION}'"
fi
if [ "${MANIFEST_TREE_STATUS}" != "clean" ]; then
  stop_and_call "bundle was built from a ${MANIFEST_TREE_STATUS:-unknown} working tree; release bundle must be clean"
fi
if ! [[ "${MANIFEST_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  stop_and_call "manifest git_commit is missing or invalid"
fi
ok "bundle version: ${BUNDLE_VERSION}"

if [ ! -f "${ENV_FILE}" ]; then
  stop_and_call "environment file not found: ${ENV_FILE}"
fi
if [ ! -x "${APP_DIR}/.venv/bin/python" ]; then
  stop_and_call "${APP_DIR}/.venv/bin/python is not executable; adopt upgrade requires the existing venv"
fi
DEPLOYED_VERSION="$(tr -d '\r\n' < "${APP_DIR}/VERSION" 2>/dev/null || true)"
if [ -z "${DEPLOYED_VERSION}" ]; then
  stop_and_call "deployed VERSION is missing; rollback verification would be ambiguous"
fi
INSTALLED_BUNDLE_COMMIT="$(tr -d '\r\n' < "${APP_DIR}/.eva-bundle-commit" 2>/dev/null || true)"
if [ "${INSTALLED_BUNDLE_COMMIT}" = "${MANIFEST_COMMIT}" ]; then
  stop_and_call "this exact ${EXPECTED_VERSION} bundle is already installed (${MANIFEST_COMMIT:0:7})"
fi
if [ "${DEPLOYED_VERSION}" = "${EXPECTED_VERSION}" ]; then
  ok "same-version hotfix candidate: ${INSTALLED_BUNDLE_COMMIT:0:7} -> ${MANIFEST_COMMIT:0:7}"
else
  ok "adopt-upgrade candidate: ${DEPLOYED_VERSION} -> ${EXPECTED_VERSION}"
  printf 'Compatibility will be decided by exact requirements and schema gates.\n'
fi

# A post-schema -> 0.8.4 adopt upgrade is dependency-neutral only when the
# declarations prove it. Without a usable wheelhouse, require byte-identical
# dependency declarations before reusing the existing venv (an empty
# wheelhouse directory does not bypass this gate).
WHEELHOUSE_ARTIFACT="$(find "${BUNDLE_DIR}/wheelhouse" -maxdepth 1 -type f \
  \( -name '*.whl' -o -name '*.tar.gz' -o -name '*.zip' \) -print -quit 2>/dev/null || true)"
if [ -z "${WHEELHOUSE_ARTIFACT}" ]; then
  for requirements_file in requirements.txt requirements-db.txt; do
    if [ ! -f "${APP_DIR}/${requirements_file}" ] \
       || ! cmp -s "${APP_DIR}/${requirements_file}" "${REPO_DIR}/${requirements_file}"; then
      stop_and_call "${requirements_file} differs from deployed tree; a reviewed wheelhouse is required"
    fi
  done
  if "${APP_DIR}/.venv/bin/python" -m pip --version >/dev/null 2>&1; then
    if ! "${APP_DIR}/.venv/bin/python" -m pip check >/dev/null 2>&1; then
      stop_and_call "existing venv failed 'pip check'; repair it or bring a reviewed wheelhouse"
    fi
  elif command -v uv >/dev/null 2>&1; then
    if ! uv pip check --python "${APP_DIR}/.venv/bin/python" >/dev/null 2>&1; then
      stop_and_call "existing venv failed 'uv pip check'; repair it or bring a reviewed wheelhouse"
    fi
  else
    stop_and_call "cannot verify existing venv: neither python -m pip nor uv is available"
  fi
  ok "requirements unchanged and existing venv passes pip check"
fi

EVIDENCE_DIR="/var/tmp/eva-upgrade-084-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${EVIDENCE_DIR}"
chmod 700 "${EVIDENCE_DIR}"
ok "evidence dir: ${EVIDENCE_DIR}"

# 2. Pre-upgrade snapshot (best-effort, read-only).
say "Pre-upgrade state snapshot"
systemctl is-active "${SERVICE_NAME}" > "${EVIDENCE_DIR}/pre_service_state.txt" 2>&1 || true
curl -sS -m 10 "${BASE_URL}/health" > "${EVIDENCE_DIR}/pre_health.json" 2>&1 || true
curl -sS -m 10 "${BASE_URL}/ready" > "${EVIDENCE_DIR}/pre_ready.json" 2>&1 || true
ok "pre_service_state=$(cat "${EVIDENCE_DIR}/pre_service_state.txt" 2>/dev/null || echo unknown)"

# 3. Read-only schema check with the runtime (non-DDL) login.
say "Read-only database schema check"
SCHEMA_VERSION="$("${APP_DIR}/.venv/bin/python" - "$ENV_FILE" <<'PYEOF' 2>>"${EVIDENCE_DIR}/schema_check.log"
import re
import os
import sys

env_path = sys.argv[1]
values = {}
with open(env_path, "r", encoding="utf-8") as handle:
    for line in handle:
        match = re.match(
            r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$",
            line.strip(),
        )
        if match:
            values[match.group(1)] = match.group(2).strip().strip('"').strip("'")
for _iteration in range(8):
    changed = False
    for key, value in tuple(values.items()):
        expanded = re.sub(
            r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}",
            lambda match: values.get(match.group(1), os.environ.get(match.group(1), match.group(0))),
            value,
        )
        if expanded != value:
            values[key] = expanded
            changed = True
    if not changed:
        break
dsn = values.get("EVA_DATABASE_DSN") or values.get("EVOSSEARCH_DATABASE_DSN") or ""
if not dsn:
    print("NO_DSN")
    sys.exit(0)
try:
    import psycopg
    with psycopg.connect(dsn, connect_timeout=10) as conn:
        row = conn.execute("SELECT version_num FROM alembic_version").fetchone()
except Exception as exc:  # noqa: BLE001 - single answer channel by design
    print(f"ERROR:{type(exc).__name__}")
    sys.exit(0)
print(row[0] if row else "EMPTY")
PYEOF
)"

case "${SCHEMA_VERSION}" in
  "${EXPECTED_SCHEMA}")
    ok "database schema is already ${EXPECTED_SCHEMA}; the database will NOT be changed (--no-migrate)"
    ;;
  NO_DSN)
    stop_and_call "EVA_DATABASE_DSN is absent from the environment file; schema cannot be verified"
    ;;
  ERROR:*)
    stop_and_call "schema could not be read (${SCHEMA_VERSION}); log: ${EVIDENCE_DIR}/schema_check.log"
    ;;
  *)
    stop_and_call "database schema '${SCHEMA_VERSION}' does not match ${EXPECTED_SCHEMA}; ONLY the responsible developer may migrate it"
    ;;
esac

# 4. Installer dry-run must be fully green before anything changes.
say "Installer dry-run (no changes)"
set +e
"${REPO_DIR}/scripts/install_eva_083.py" \
  --dry-run --non-interactive --no-migrate \
  --source-dir "${REPO_DIR}" \
  --bundle-dir "${BUNDLE_DIR}" \
  --app-dir "${APP_DIR}" \
  --env-file "${ENV_FILE}" \
  --service-name "${SERVICE_NAME}" \
  2>&1 | tee "${EVIDENCE_DIR}/dry_run.txt"
DRY_STATUS=${PIPESTATUS[0]}
set -e
if [ "${DRY_STATUS}" -ne 0 ]; then
  stop_and_call "dry-run was blocked; see the FAIL lines above and ${EVIDENCE_DIR}/dry_run.txt"
fi
ok "dry-run passed"

# 5. One explicit confirmation, typed by a human.
say "Confirmation"
printf 'Service %s will be stopped and upgraded to %s. The database will not be changed.\n' "${SERVICE_NAME}" "${BUNDLE_VERSION}"
printf 'Type UPGRADE and press Enter: '
read -r CONFIRMATION
if [ "${CONFIRMATION}" != "UPGRADE" ]; then
  stop_and_call "confirmation was not received; nothing was changed"
fi

# 6. Apply the reviewed plan.
say "Applying update (log: ${EVIDENCE_DIR}/apply.txt)"
set +e
"${REPO_DIR}/scripts/install_eva_083.py" \
  --apply --non-interactive --no-migrate \
  --source-dir "${REPO_DIR}" \
  --bundle-dir "${BUNDLE_DIR}" \
  --app-dir "${APP_DIR}" \
  --env-file "${ENV_FILE}" \
  --service-name "${SERVICE_NAME}" \
  2>&1 | tee "${EVIDENCE_DIR}/apply.txt"
APPLY_STATUS=${PIPESTATUS[0]}
set -e

# 7. Record the rollback command regardless of outcome.
grep -Ei "^(ROLLBACK HANDOFF: |rollback_command=)" "${EVIDENCE_DIR}/apply.txt" \
  | tail -n 1 \
  | sed -E 's/^ROLLBACK HANDOFF: //; s/^rollback_command=//' \
  > "${EVIDENCE_DIR}/ROLLBACK_COMMAND.txt" 2>/dev/null || true
if [ -s "${EVIDENCE_DIR}/ROLLBACK_COMMAND.txt" ]; then
  ok "rollback command saved: ${EVIDENCE_DIR}/ROLLBACK_COMMAND.txt"
fi

if [ "${APPLY_STATUS}" -ne 0 ]; then
  stop_and_call "apply failed; log: ${EVIDENCE_DIR}/apply.txt"
fi
if [ ! -s "${EVIDENCE_DIR}/ROLLBACK_COMMAND.txt" ]; then
  stop_and_call "apply completed but no rollback command was recorded; log: ${EVIDENCE_DIR}/apply.txt"
fi

# 8. Post-upgrade verification.
say "Post-upgrade verification"
sleep 5
systemctl is-active "${SERVICE_NAME}" > "${EVIDENCE_DIR}/post_service_state.txt" 2>&1 || true
HEALTH_OK=false
for _attempt in 1 2 3 4 5 6 7 8 9; do
  if curl -fsS -m 10 "${BASE_URL}/health" > "${EVIDENCE_DIR}/post_health.json" 2>/dev/null \
     && curl -fsS -m 10 "${BASE_URL}/ready?load=1" > "${EVIDENCE_DIR}/post_ready.json" 2>/dev/null; then
    if "${APP_DIR}/.venv/bin/python" - "${EXPECTED_VERSION}" "${EVIDENCE_DIR}/post_ready.json" <<'PYEOF'
import json
import sys

try:
    with open(sys.argv[2], "r", encoding="utf-8") as handle:
        payload = json.load(handle)
except Exception:
    raise SystemExit(1)
raise SystemExit(
    0
    if payload.get("status") == "ready" and payload.get("version") == sys.argv[1]
    else 1
)
PYEOF
    then
      HEALTH_OK=true
      break
    fi
  fi
  sleep 10
done
systemctl is-active "${SERVICE_NAME}" > "${EVIDENCE_DIR}/post_service_state.txt" 2>&1 || true
POST_STATE="$(cat "${EVIDENCE_DIR}/post_service_state.txt" 2>/dev/null || echo unknown)"

if [ "${POST_STATE}" != "active" ] || [ "${HEALTH_OK}" != "true" ]; then
  fail "service did not confirm readiness (state=${POST_STATE})"
  printf 'Rollback: run the command stored in %s\n' "${EVIDENCE_DIR}/ROLLBACK_COMMAND.txt" >&2
  exit 1
fi
ok "service=active, /health responds, /ready status=ready and version=${EXPECTED_VERSION}"
printf '%s\n' "${MANIFEST_COMMIT}" > "${APP_DIR}/.eva-bundle-commit"
ok "installed bundle marker: ${MANIFEST_COMMIT:0:7}"

say "COMPLETE"
cat <<DONE
1. Open the UI and the Video tab. Channels with enabled summaries should
   restore automatically within about two minutes. Channel Runtime shows any
   restore failure.
2. Run the smoke test in readiness/EVA_AI_0.8.4_R4_FIELD_UPDATE_EN.md.
3. All logs from this run: ${EVIDENCE_DIR}
4. If anything is wrong, the rollback command is stored in
   ${EVIDENCE_DIR}/ROLLBACK_COMMAND.txt. Collect diagnostics with:
   bash scripts/client_diagnostics.sh > diag.txt
DONE
