#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTEST="${PYTEST:-${ROOT}/.venv/bin/pytest}"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"

BASE_URL="${EVA_LIVE_BASE_URL:-https://127.0.0.1:5443}"
CHANNEL_REF="${EVA_LIVE_CHANNEL_REF:-112}"
NEEDLE_QUERY="${EVA_LIVE_NEEDLE_QUERY:-person lying on the ground at night}"
PROBE_NAME="${EVA_LIVE_PROBE_NAME:-smoke: thumbs up gesture}"

RUN_LIVE="${EVA_PREDEPLOY_RUN_LIVE:-false}"
RUN_OPERATOR="${EVA_PREDEPLOY_RUN_OPERATOR:-false}"
RUN_SEED="${EVA_PREDEPLOY_RUN_SEED:-false}"
SKIP_FULL_TESTS="${EVA_PREDEPLOY_SKIP_FULL_TESTS:-false}"

log() {
  printf '\n[predeploy] %s\n' "$*"
}

die() {
  printf '[predeploy] FAIL: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'USAGE'
Usage: scripts/predeploy_acceptance.sh

Runs deterministic predeploy gates by default:
  - git diff whitespace check
  - docs drift guard
  - full pytest suite

Optional live smoke:
  EVA_PREDEPLOY_RUN_SEED=true       seed archive fixtures first
  EVA_PREDEPLOY_RUN_LIVE=true       run admin live smoke
  EVA_PREDEPLOY_RUN_OPERATOR=true   create/update operator-smoke and run non-admin smoke

Required env for admin live smoke:
  EVA_LIVE_PASSWORD                 admin password

Required env for operator live smoke:
  EVA_LIVE_OPERATOR_PASSWORD        operator-smoke password

Useful env:
  EVA_LIVE_BASE_URL=https://127.0.0.1:5443
  EVA_LIVE_CHANNEL_REF=112
  EVA_LIVE_NEEDLE_QUERY="person lying on the ground at night"
  EVA_LIVE_PROBE_NAME="smoke: thumbs up gesture"
  EVA_PREDEPLOY_SKIP_FULL_TESTS=true
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

cd "${ROOT}"

[[ -x "${PYTEST}" ]] || die "pytest not found/executable: ${PYTEST}"
[[ -x "${PYTHON}" ]] || die "python not found/executable: ${PYTHON}"

log "version"
tr -d '\n' < VERSION
printf '\n'

log "git diff whitespace check"
git diff --check

log "docs drift guard"
bash scripts/check_docs_drift.sh

if [[ "${SKIP_FULL_TESTS}" == "true" ]]; then
  log "full pytest skipped by EVA_PREDEPLOY_SKIP_FULL_TESTS=true"
else
  log "full pytest"
  "${PYTEST}" -q
fi

if [[ "${RUN_SEED}" == "true" ]]; then
  log "seed live-smoke archive fixtures on channel ${CHANNEL_REF}"
  "${PYTHON}" scripts/seed_demo_fixtures.py --channel-id "${CHANNEL_REF}"
fi

if [[ "${RUN_LIVE}" == "true" ]]; then
  [[ -n "${EVA_LIVE_PASSWORD:-}" ]] || die "EVA_LIVE_PASSWORD is required for admin live smoke"
  log "admin live smoke"
  EVA_LIVE_BASE_URL="${BASE_URL}" \
  EVA_LIVE_USER="${EVA_LIVE_USER:-admin}" \
  EVA_LIVE_PASSWORD="${EVA_LIVE_PASSWORD}" \
  EVA_LIVE_CHANNEL_REF="${CHANNEL_REF}" \
  EVA_LIVE_NEEDLE_QUERY="${NEEDLE_QUERY}" \
  EVA_LIVE_PROBE_NAME="${PROBE_NAME}" \
  EVA_LIVE_INCLUDE=seed \
  "${PYTEST}" -q tests/integration/test_live_agent.py -s
fi

if [[ "${RUN_OPERATOR}" == "true" ]]; then
  [[ -n "${EVA_LIVE_OPERATOR_PASSWORD:-}" ]] || die "EVA_LIVE_OPERATOR_PASSWORD is required for operator live smoke"
  log "bootstrap non-admin live-smoke operator"
  EVA_LIVE_OPERATOR_PASSWORD="${EVA_LIVE_OPERATOR_PASSWORD}" \
  "${PYTHON}" scripts/bootstrap_live_smoke_operator.py --channel-id "${CHANNEL_REF}" --set-password --base-url "${BASE_URL}"

  log "non-admin live smoke"
  EVA_LIVE_BASE_URL="${BASE_URL}" \
  EVA_LIVE_USER=operator-smoke \
  EVA_LIVE_PASSWORD="${EVA_LIVE_OPERATOR_PASSWORD}" \
  EVA_LIVE_CHANNEL_REF="${CHANNEL_REF}" \
  EVA_LIVE_INCLUDE=non_admin \
  "${PYTEST}" -q tests/integration/test_live_agent.py -s
fi

log "predeploy acceptance completed"
