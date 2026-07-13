#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="${APP_DIR}/.eva-runtime"
if [[ -x "${RUNTIME_DIR}/bin/ffmpeg" ]]; then
  export PATH="${RUNTIME_DIR}/bin:${PATH}"
fi
if [[ -d "${RUNTIME_DIR}/python" ]]; then
  export PYTHONPATH="${RUNTIME_DIR}/python${PYTHONPATH:+:${PYTHONPATH}}"
fi

HOST="${EVOSSEARCH_HOST:-0.0.0.0}"
PORT="${EVOSSEARCH_PORT:-5000}"
WORKERS="${EVOSSEARCH_GUNICORN_WORKERS:-1}"
THREADS="${EVOSSEARCH_GUNICORN_THREADS:-8}"
TIMEOUT="${EVOSSEARCH_GUNICORN_TIMEOUT:-180}"
GUNICORN_BIN="${EVOSSEARCH_GUNICORN_BIN:-}"
GUNICORN_CONFIG="${EVOSSEARCH_GUNICORN_CONFIG:-gunicorn_conf.py}"
SECURE_REQUIRED="${EVOSSEARCH_SECURE_DEPLOYMENT_REQUIRED:-false}"

case "${SECURE_REQUIRED,,}" in
  1|true|yes|on)
    if [[ "${WORKERS}" != "1" ]]; then
      echo "EVOSSEARCH_SECURE_DEPLOYMENT_REQUIRED requires EVOSSEARCH_GUNICORN_WORKERS=1; shared in-process schedulers are not multi-worker safe." >&2
      exit 1
    fi
    ;;
esac

if [[ -z "${GUNICORN_BIN}" ]]; then
  if [[ -x ".venv/bin/gunicorn" ]]; then
    GUNICORN_BIN=".venv/bin/gunicorn"
  elif command -v gunicorn >/dev/null 2>&1; then
    GUNICORN_BIN="gunicorn"
  else
    echo "gunicorn not found. Install requirements or set EVOSSEARCH_GUNICORN_BIN." >&2
    exit 1
  fi
fi

GUNICORN_ARGS=()

if [[ -n "${GUNICORN_CONFIG}" ]]; then
  GUNICORN_ARGS+=(--config "${GUNICORN_CONFIG}")
fi

GUNICORN_ARGS+=(
  "wsgi:app" \
  --bind "${HOST}:${PORT}" \
  --workers "${WORKERS}" \
  --threads "${THREADS}" \
  --timeout "${TIMEOUT}" \
  --worker-class gthread
)

exec "${GUNICORN_BIN}" "${GUNICORN_ARGS[@]}"
