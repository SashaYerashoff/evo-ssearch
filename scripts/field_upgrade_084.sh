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
  printf '\nДальше НЕ продолжай. Позвони разработчику и продиктуй строку выше.\n' >&2
  printf 'Диагностика для отправки: bash scripts/client_diagnostics.sh > diag.txt\n' >&2
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUNDLE_DIR="$(cd "${REPO_DIR}/.." && pwd)"

say "EVA AI 0.8.4 guided upgrade"

if [ "$(id -u)" -ne 0 ]; then
  stop_and_call "запусти через sudo: sudo ./scripts/field_upgrade_084.sh"
fi

# 1. We must be inside an unpacked offline bundle, not a random checkout.
if [ ! -f "${BUNDLE_DIR}/manifest.txt" ]; then
  stop_and_call "manifest.txt не найден рядом с repo/ — это не распакованный bundle"
fi
BUNDLE_VERSION="$(tr -d '\n' < "${REPO_DIR}/VERSION" 2>/dev/null || true)"
ok "bundle version: ${BUNDLE_VERSION:-unknown}"

if [ ! -f "${ENV_FILE}" ]; then
  stop_and_call "env-файл не найден: ${ENV_FILE}"
fi
if [ ! -x "${APP_DIR}/.venv/bin/python" ]; then
  stop_and_call "нет исполняемого ${APP_DIR}/.venv/bin/python — adopt-апгрейд невозможен без venv"
fi

EVIDENCE_DIR="/var/tmp/eva-upgrade-084-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${EVIDENCE_DIR}"
chmod 700 "${EVIDENCE_DIR}"
ok "evidence dir: ${EVIDENCE_DIR}"

# 2. Pre-upgrade snapshot (best-effort, read-only).
say "Снимок состояния до апгрейда"
systemctl is-active "${SERVICE_NAME}" > "${EVIDENCE_DIR}/pre_service_state.txt" 2>&1 || true
curl -sS -m 10 "${BASE_URL}/health" > "${EVIDENCE_DIR}/pre_health.json" 2>&1 || true
curl -sS -m 10 "${BASE_URL}/ready" > "${EVIDENCE_DIR}/pre_ready.json" 2>&1 || true
ok "pre_service_state=$(cat "${EVIDENCE_DIR}/pre_service_state.txt" 2>/dev/null || echo unknown)"

# 3. Read-only schema check with the runtime (non-DDL) login.
say "Проверка версии схемы БД (только чтение)"
SCHEMA_VERSION="$("${APP_DIR}/.venv/bin/python" - "$ENV_FILE" <<'PYEOF' 2>>"${EVIDENCE_DIR}/schema_check.log"
import re
import sys

env_path = sys.argv[1]
dsn = ""
with open(env_path, "r", encoding="utf-8") as handle:
    for line in handle:
        match = re.match(r"^EVA_DATABASE_DSN=(.*)$", line.strip())
        if match:
            dsn = match.group(1).strip().strip('"').strip("'")
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
    ok "схема БД уже на ожидаемой голове ${EXPECTED_SCHEMA}; база данных изменяться НЕ будет (--no-migrate)"
    ;;
  NO_DSN)
    stop_and_call "в env нет EVA_DATABASE_DSN — проверить версию схемы невозможно"
    ;;
  ERROR:*)
    stop_and_call "не удалось прочитать версию схемы (${SCHEMA_VERSION}); лог: ${EVIDENCE_DIR}/schema_check.log"
    ;;
  *)
    stop_and_call "схема БД '${SCHEMA_VERSION}' не совпадает с ожидаемой ${EXPECTED_SCHEMA}; миграцию выполняет ТОЛЬКО разработчик"
    ;;
esac

# 4. Installer dry-run must be fully green before anything changes.
say "Пробный прогон инсталлера (dry-run, без изменений)"
set +e
"${REPO_DIR}/scripts/install_eva_083.py" \
  --dry-run --non-interactive --no-migrate \
  --source-dir "${REPO_DIR}" \
  --bundle-dir "${BUNDLE_DIR}" \
  --app-dir "${APP_DIR}" \
  --env-file "${ENV_FILE}" \
  --service-name "${SERVICE_NAME}" \
  | tee "${EVIDENCE_DIR}/dry_run.txt"
DRY_STATUS=${PIPESTATUS[0]}
set -e
if [ "${DRY_STATUS}" -ne 0 ]; then
  stop_and_call "dry-run заблокирован (см. строки FAIL выше и ${EVIDENCE_DIR}/dry_run.txt)"
fi
ok "dry-run чистый"

# 5. One explicit confirmation, typed by a human.
say "Подтверждение"
printf 'Сервис %s будет остановлен и обновлён до %s. База данных не изменяется.\n' "${SERVICE_NAME}" "${BUNDLE_VERSION}"
printf 'Набери слово UPGRADE и нажми Enter: '
read -r CONFIRMATION
if [ "${CONFIRMATION}" != "UPGRADE" ]; then
  stop_and_call "подтверждение не получено — ничего не изменено"
fi

# 6. Apply the reviewed plan.
say "Применение (журнал: ${EVIDENCE_DIR}/apply.txt)"
set +e
"${REPO_DIR}/scripts/install_eva_083.py" \
  --apply --non-interactive --no-migrate \
  --source-dir "${REPO_DIR}" \
  --bundle-dir "${BUNDLE_DIR}" \
  --app-dir "${APP_DIR}" \
  --env-file "${ENV_FILE}" \
  --service-name "${SERVICE_NAME}" \
  | tee "${EVIDENCE_DIR}/apply.txt"
APPLY_STATUS=${PIPESTATUS[0]}
set -e

# 7. Record the rollback command regardless of outcome.
grep -E "rollback" "${EVIDENCE_DIR}/apply.txt" > "${EVIDENCE_DIR}/ROLLBACK_COMMAND.txt" 2>/dev/null || true
if [ -s "${EVIDENCE_DIR}/ROLLBACK_COMMAND.txt" ]; then
  ok "команда отката сохранена: ${EVIDENCE_DIR}/ROLLBACK_COMMAND.txt"
fi

if [ "${APPLY_STATUS}" -ne 0 ]; then
  stop_and_call "apply завершился с ошибкой; журнал: ${EVIDENCE_DIR}/apply.txt"
fi

# 8. Post-upgrade verification.
say "Проверка после апгрейда"
sleep 5
systemctl is-active "${SERVICE_NAME}" > "${EVIDENCE_DIR}/post_service_state.txt" 2>&1 || true
POST_STATE="$(cat "${EVIDENCE_DIR}/post_service_state.txt" 2>/dev/null || echo unknown)"
HEALTH_OK=false
for _attempt in 1 2 3 4 5 6 7 8 9; do
  if curl -sS -m 10 "${BASE_URL}/health" > "${EVIDENCE_DIR}/post_health.json" 2>/dev/null \
     && curl -sS -m 10 "${BASE_URL}/ready" > "${EVIDENCE_DIR}/post_ready.json" 2>/dev/null; then
    HEALTH_OK=true
    break
  fi
  sleep 10
done

if [ "${POST_STATE}" != "active" ] || [ "${HEALTH_OK}" != "true" ]; then
  fail "сервис не подтвердил здоровье (state=${POST_STATE})"
  printf 'Откат: выполни команду из %s\n' "${EVIDENCE_DIR}/ROLLBACK_COMMAND.txt" >&2
  exit 1
fi
ok "service=active, /health и /ready отвечают"

say "ГОТОВО"
cat <<DONE
1. Открой UI и вкладку Video: каналы с включёнными summaries должны сами
   подняться в течение ~2 минут (persisted desired state). Если канал не
   поднялся — Channel Runtime покажет ошибку restore.
2. Пройди smoke из readiness/UPGRADE_084_FIELD_CHECKLIST_RU.md (10 минут).
3. Все журналы этого запуска: ${EVIDENCE_DIR}
4. Если что-то не так: команда отката лежит в
   ${EVIDENCE_DIR}/ROLLBACK_COMMAND.txt, диагностика -
   bash scripts/client_diagnostics.sh > diag.txt (отправь файл).
DONE
