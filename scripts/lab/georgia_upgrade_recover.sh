#!/usr/bin/env bash
set -Eeuo pipefail

TARGET_ROOT="/home/sasha/Projects/eva-georgia-upgrade-repro"
SOURCE_ROOT="/home/sasha/Projects/evo-ssearch-office-demo"
LOCAL_PROFILE_ENV="/home/sasha/Projects/evo-ssearch/.env"
BASELINE_VENV="/home/sasha/Projects/evo-ssearch-tbilisi-field/.venv"
ENV_FILE="${TARGET_ROOT}/.env"
SERVICE_NAME="eva-ai-georgia-repro"
UNIT_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
BACKUP_DIR="/var/backups/${SERVICE_NAME}/patch-20260810-132911"
DATABASE_DUMP="${TARGET_ROOT}/backups/eva-0.8.1-schema-0006.pgcustom"
DATABASE_DUMP_SHA256="d81e9a17f04ce33751a117b878ad502608c552111bce8e5f474db9dbd63952eb"
DB_CONTAINER="eva-tbilisi-repro-postgres"
EXPECTED_DATABASE="eva_tbilisi_repro"
EXPECTED_REVISION="20260614_0006"
EXPECTED_VERSION="β 0.8.1"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

if (( EUID != 0 )); then
  exec sudo -- "$0" "$@"
fi

[[ "${TARGET_ROOT}" == "/home/sasha/Projects/eva-georgia-upgrade-repro" ]] \
  || die "unsafe target path"
[[ -f "${BACKUP_DIR}/code.tgz" ]] || die "baseline code backup is missing"
[[ -f "${BACKUP_DIR}/eva-ai.env" ]] || die "baseline environment backup is missing"
[[ -f "${DATABASE_DUMP}" ]] || die "baseline database dump is missing"
[[ -x "${BASELINE_VENV}/bin/python" ]] || die "baseline Georgia venv is missing"
[[ -f "${LOCAL_PROFILE_ENV}" ]] || die "verified local Evo profile is missing"
docker inspect "${DB_CONTAINER}" >/dev/null 2>&1 || die "rehearsal PostgreSQL is unavailable"

actual_dump_sha="$(sha256sum "${DATABASE_DUMP}" | awk '{print $1}')"
[[ "${actual_dump_sha}" == "${DATABASE_DUMP_SHA256}" ]] \
  || die "baseline database dump checksum mismatch"

printf '\nReset EVA Georgia rehearsal to the exact pre-upgrade baseline\n'
printf '  target:   %s\n' "${TARGET_ROOT}"
printf '  version:  %s\n' "${EXPECTED_VERSION}"
printf '  database: %s @ %s\n\n' "${EXPECTED_DATABASE}" "${EXPECTED_REVISION}"

systemctl stop "${SERVICE_NAME}.service" 2>/dev/null || true
systemctl disable "${SERVICE_NAME}.service" >/dev/null 2>&1 || true

python3 "${SOURCE_ROOT}/scripts/restore_code_snapshot.py" \
  --archive "${BACKUP_DIR}/code.tgz" \
  --app-dir "${TARGET_ROOT}"

cp -a -- "${BACKUP_DIR}/eva-ai.env" "${ENV_FILE}"

# Keep the restored site topology, but point the lab rehearsal at the verified
# local Evo. Secrets are copied in memory and never printed.
python3 - "${LOCAL_PROFILE_ENV}" "${ENV_FILE}" <<'PY'
import os
import re
import stat
import sys
import tempfile
from pathlib import Path

source_path = Path(sys.argv[1])
target_path = Path(sys.argv[2])

def parse(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw or raw.lstrip().startswith("#"):
            continue
        key, value = raw.split("=", 1)
        result[key.strip()] = value.strip()
    return result

source = parse(source_path)
required = (
    "EVOSSEARCH_LUXRIOT_BASE_URL",
    "EVOSSEARCH_LUXRIOT_USERNAME",
    "EVOSSEARCH_LUXRIOT_PASSWORD",
)
if any(not source.get(key) for key in required):
    raise SystemExit("verified local Evo profile is incomplete")

text = target_path.read_text(encoding="utf-8")
for key in required:
    replacement = f"{key}={source[key]}"
    text, count = re.subn(
        rf"(?m)^[ \t]*(?:export[ \t]+)?{re.escape(key)}[ \t]*=.*$",
        replacement,
        text,
    )
    if count == 0:
        text = text.rstrip("\n") + "\n" + replacement + "\n"

for key in (
    "EVOSSEARCH_UI_MODE",
    "EVOSSEARCH_ARCHIVE_RETENTION_ENABLED",
    "EVOSSEARCH_ARCHIVE_ROW_RETENTION_DAYS",
    "EVOSSEARCH_ARCHIVE_THUMBNAIL_RETENTION_DAYS",
):
    text = re.sub(
        rf"(?m)^[ \t]*(?:export[ \t]+)?{re.escape(key)}[ \t]*=.*\n?",
        "",
        text,
    )

text, count = re.subn(
    r"(?m)^[ \t]*EVOSSEARCH_APP_VERSION[ \t]*=.*$",
    "EVOSSEARCH_APP_VERSION='β 0.8.1'",
    text,
)
if count == 0:
    text = text.rstrip("\n") + "\nEVOSSEARCH_APP_VERSION='β 0.8.1'\n"

fd, temp_name = tempfile.mkstemp(prefix=".eva-georgia-baseline.", dir=target_path.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temp_name, 0o600)
    os.chown(temp_name, 1000, 1000)
    os.replace(temp_name, target_path)
finally:
    try:
        os.unlink(temp_name)
    except FileNotFoundError:
        pass
PY

if [[ -e "${TARGET_ROOT}/.venv" && ! -L "${TARGET_ROOT}/.venv" ]]; then
  die "refusing to replace non-symlink target venv"
fi
if [[ -L "${TARGET_ROOT}/.venv" ]]; then
  unlink "${TARGET_ROOT}/.venv"
fi
ln -s "${BASELINE_VENV}" "${TARGET_ROOT}/.venv"
chown -h sasha:sasha "${TARGET_ROOT}/.venv"

docker exec "${DB_CONTAINER}" psql -v ON_ERROR_STOP=1 -U postgres -d postgres \
  -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='${EXPECTED_DATABASE}' AND pid <> pg_backend_pid();" \
  >/dev/null
docker exec "${DB_CONTAINER}" dropdb -U postgres --if-exists "${EXPECTED_DATABASE}"
docker exec "${DB_CONTAINER}" createdb -U postgres "${EXPECTED_DATABASE}"
docker exec -i "${DB_CONTAINER}" pg_restore \
  --exit-on-error \
  -U postgres \
  -d "${EXPECTED_DATABASE}" \
  < "${DATABASE_DUMP}"

if [[ -e "${UNIT_FILE}" || -L "${UNIT_FILE}" ]]; then
  [[ "${UNIT_FILE}" == "/etc/systemd/system/eva-ai-georgia-repro.service" ]] \
    || die "unsafe service unit path"
  unlink "${UNIT_FILE}"
fi
systemctl daemon-reload
systemctl reset-failed "${SERVICE_NAME}.service" 2>/dev/null || true

chown sasha:sasha "${ENV_FILE}"
chmod 0600 "${ENV_FILE}"

actual_version="$(tr -d '\r\n' < "${TARGET_ROOT}/VERSION")"
actual_revision="$(docker exec "${DB_CONTAINER}" psql -U postgres -d "${EXPECTED_DATABASE}" -Atc 'select version_num from public.alembic_version')"
archive_facts="$(docker exec "${DB_CONTAINER}" psql -U postgres -d "${EXPECTED_DATABASE}" -Atc \
  "select count(*) || '|' || count(thumbnail_b64) || '|' || count(image_path) from archive.detections")"

[[ "${actual_version}" == "${EXPECTED_VERSION}" ]] || die "code restore produced ${actual_version}"
[[ "${actual_revision}" == "${EXPECTED_REVISION}" ]] || die "database restore produced ${actual_revision}"
[[ "${archive_facts}" == "8683|8683|5" ]] || die "archive restore produced unexpected counts: ${archive_facts}"
[[ ! -e "${UNIT_FILE}" ]] || die "rehearsal service unit still exists"
if "${TARGET_ROOT}/.venv/bin/python" -c 'import cv2' >/dev/null 2>&1; then
  die "baseline venv unexpectedly contains the post-upgrade OpenCV runtime"
fi

printf '\nRESET COMPLETE\n'
printf '  EVA version:       %s\n' "${actual_version}"
printf '  database revision: %s\n' "${actual_revision}"
printf '  archive:           8683 rows, 8683 thumbnails, 5 sidecars\n'
printf '  service:           absent (the updater must create it)\n'
printf '  venv:              original Georgia environment, OpenCV absent\n'
printf '\nNext: run /home/sasha/Projects/evo-ssearch/scripts/lab/georgia_upgrade_test.sh\n'
