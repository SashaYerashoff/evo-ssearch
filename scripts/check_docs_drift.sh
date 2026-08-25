#!/usr/bin/env bash
# Anti-drift guard for current documentation.
# Fails if current docs contain forbidden stale claims, or if the canon disagrees
# with the code. Exempt: readiness/history/ (snapshots), docs/gtm/ (separate
# reconcile track), RELEASE_NOTES_* (legitimately reference migration history).
#
# Run in CI and before each release.
set -uo pipefail
cd "$(dirname "$0")/.."

fail=0

patterns=(
  'sqlite'                                   # store is PostgreSQL; SQLite only valid historically
  'Secure mutation paths via admin token'    # legacy auth framing
)
scan_paths=(docs README.md CHANGELOG.md)
mapfile -t readiness_current < <(find readiness -maxdepth 1 -name '*.md' ! -name 'RELEASE_NOTES_*')

for pat in "${patterns[@]}"; do
  if hits=$(grep -rinI --exclude-dir=history --exclude-dir=gtm \
              "$pat" "${scan_paths[@]}" "${readiness_current[@]}" 2>/dev/null); then
    echo "FORBIDDEN (stale): pattern '$pat'"
    echo "$hits"
    fail=1
  fi
done

# Canon vs code consistency (keep internal spaces, e.g. "β 0.8.1").
ver_file=$(tr -d '\n' < VERSION 2>/dev/null | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')
if [ -n "${ver_file:-}" ] && ! grep -qF "$ver_file" docs/00_CANON/facts.md 2>/dev/null; then
  echo "DRIFT: VERSION ('$ver_file') not reflected in docs/00_CANON/facts.md"
  fail=1
fi

code_head=$(grep -oE '[0-9]{8}_[0-9]{4}' eva_db/settings.py 2>/dev/null | head -1)
if [ -n "${code_head:-}" ] && ! grep -qF "$code_head" docs/00_CANON/facts.md 2>/dev/null; then
  echo "DRIFT: schema head '$code_head' (eva_db/settings.py) not in docs/00_CANON/facts.md"
  fail=1
fi

# UI screenshots: every guide picture must come from a scene in docs/ui/shots.json
# and still exist. Stdlib-only, no browser, no running service.
py=""
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then py="$candidate"; break; fi
done
if [ -n "$py" ]; then
  if ! "$py" scripts/ui_shots.py validate; then
    fail=1
  fi
else
  echo "ui shots: no python interpreter found; screenshot manifest not checked"
fi

if [ "$fail" -eq 0 ]; then
  echo "docs drift check: OK"
fi
exit "$fail"
