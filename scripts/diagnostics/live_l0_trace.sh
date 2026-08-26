#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
app_dir="$(systemctl show eva-ai.service -p WorkingDirectory --value)"
env_spec="$(systemctl show eva-ai.service -p EnvironmentFiles --value)"
env_file="${env_spec%% *}"

if [[ -z "${app_dir}" || ! -x "${app_dir}/.venv/bin/python" ]]; then
  echo "ERROR: could not resolve the installed EVA Python from eva-ai.service" >&2
  exit 1
fi
if [[ -z "${env_file}" || ! -r "${env_file}" ]]; then
  echo "ERROR: could not read the installed EVA environment file" >&2
  exit 1
fi

exec "${app_dir}/.venv/bin/python" \
  "${SCRIPT_DIR}/live_l0_trace.py" \
  --env-file "${env_file}" \
  "$@"
