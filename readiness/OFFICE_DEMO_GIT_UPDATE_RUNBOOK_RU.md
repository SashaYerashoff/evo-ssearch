# EVA AI β 0.8.2 - office demo update from git

Audience: project owner / engineer updating the office demo machine.  
Target: office demo installation with 20+ video-loop channels.  
Branch: `feature/secure-50-channel-foundation`.  
Release: `β 0.8.2`.  
Migration: **none**.

This runbook updates code from git while preserving the current PostgreSQL
database, archive, users, prompts, and existing `.env`.

## 0. Paste This First In The Same Terminal

Run this block first. Keep using the same terminal after it. The later command
blocks rely on these exported variables.

```bash
export APP_DIR=/opt/eva-ai/evo-ssearch
export SERVICE=eva-ai
export ENV_FILE=/etc/eva-ai/eva-ai.env
export DB_NAME=eva
export BRANCH=feature/secure-50-channel-foundation
export BASE_URL=https://127.0.0.1:5443

printf 'APP_DIR=<%s>\nSERVICE=<%s>\nENV_FILE=<%s>\nDB_NAME=<%s>\nBRANCH=<%s>\nBASE_URL=<%s>\n' \
  "$APP_DIR" "$SERVICE" "$ENV_FILE" "$DB_NAME" "$BRANCH" "$BASE_URL"

: "${APP_DIR:?APP_DIR is empty}"
: "${SERVICE:?SERVICE is empty}"
: "${ENV_FILE:?ENV_FILE is empty}"
: "${DB_NAME:?DB_NAME is empty}"
: "${BRANCH:?BRANCH is empty}"
: "${BASE_URL:?BASE_URL is empty}"
```

If even simple commands like `ls -la` show no output while the prompt still
appears, stdout/stderr were probably redirected in the current root shell. Fix
the shell before continuing:

```bash
exec >/dev/tty 2>&1
printf 'stdout ok\n'
printf 'stderr ok\n' >&2
```

If the service is a user service instead of a system service, replace:

```bash
sudo systemctl ...
```

with:

```bash
systemctl --user ...
```

## 1. Preflight

```bash
: "${APP_DIR:?APP_DIR is empty; rerun section 0}"
: "${SERVICE:?SERVICE is empty; rerun section 0}"

cd "$APP_DIR"

git config --global --add safe.directory /opt/eva-ai/evo-ssearch
git branch --show-current
git rev-parse --short HEAD
git status --short
```

Expected:

- branch is `feature/secure-50-channel-foundation`;
- working tree is clean, or only known local ignored files are present.

If `git status --short` shows modified tracked files, stop and ask before pulling.

If git says `detected dubious ownership`, make sure section 0 was run and then
run this exact command:

```bash
git config --global --add safe.directory /opt/eva-ai/evo-ssearch
```

## 2. Stop EVA AI

```bash
: "${SERVICE:?SERVICE is empty; rerun section 0}"

sudo systemctl stop "$SERVICE"
sudo systemctl status "$SERVICE" --no-pager -l || true
```

If systemd says `Failed to mangle name: Invalid argument`, `$SERVICE` is empty.
Rerun section 0 in the same terminal. Do not run `sudo --user systemctl`;
`--user` belongs to `systemctl`, not to `sudo`.

If `systemctl stop` ends with `Result: timeout` and then `SIGKILL`, the service
is still stopped enough for the update. Continue with the backup step unless
processes remain running.

## 3. Backup Database And Env

This fixes the old permission trap by making the backup directory writable by
the `postgres` OS user.

```bash
TS="$(date +%Y%m%d-%H%M%S)"

sudo install -d -o postgres -g postgres -m 700 /var/lib/eva-ai/backups

BACKUP="/var/lib/eva-ai/backups/eva-before-0.8.2-${TS}.dump"
sudo -u postgres pg_dump -Fc -d "$DB_NAME" -f "$BACKUP"
sudo ls -lh "$BACKUP"

sudo cp -a "$ENV_FILE" "${ENV_FILE}.bak.${TS}"
sudo ls -lh "${ENV_FILE}.bak.${TS}"
```

Do not continue if `pg_dump` fails.

## 4. Update Code From Git

```bash
cd "$APP_DIR"

git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

git rev-parse --short HEAD
cat VERSION
```

Expected:

- `cat VERSION` prints `β 0.8.2`.
- Pull is fast-forward only. If it is not, stop and ask.

## 5. Preserve Env, But Update Visible Version

If `EVOSSEARCH_APP_VERSION` is set in the env file, it overrides `VERSION`.
Update it so UI/health show the deployed code version.

```bash
if sudo grep -q '^EVOSSEARCH_APP_VERSION=' "$ENV_FILE"; then
  sudo sed -i 's/^EVOSSEARCH_APP_VERSION=.*/EVOSSEARCH_APP_VERSION="β 0.8.2"/' "$ENV_FILE"
else
  echo 'EVOSSEARCH_APP_VERSION="β 0.8.2"' | sudo tee -a "$ENV_FILE" >/dev/null
fi
```

Do not overwrite site-specific Luxriot IPs, LM Studio/vLLM URLs, tenant IDs,
or passwords.

## 6. Install/Verify Python Dependencies If Needed

Usually no dependency changes are needed for `β 0.8.2`.

Run only if the venv was changed or missing packages are reported:

```bash
cd "$APP_DIR"
.venv/bin/python -m pip install -r requirements.txt
```

## 7. Run Deterministic Predeploy Gates

This does not call the slow live agent smoke unless explicit env flags are set.

```bash
cd "$APP_DIR"
scripts/predeploy_acceptance.sh
```

Expected:

- docs drift OK;
- pytest OK.

If this is too slow on the office machine and you already tested locally, at
minimum run:

```bash
bash scripts/check_docs_drift.sh
.venv/bin/pytest -q tests/integration tests/test_agent_tool_loop.py tests/test_eva_agent_adapter.py
```

## 8. Start EVA AI

```bash
sudo systemctl daemon-reload
sudo systemctl start "$SERVICE"
sleep 10
sudo systemctl status "$SERVICE" --no-pager -l
```

## 9. Health / Ready

Use `-k` for the local self-signed certificate.

```bash
curl -k -sS "$BASE_URL/health" | jq
curl -k -sS "$BASE_URL/ready" | jq
```

Expected:

- `/health.version` is `β 0.8.2`;
- database/auth/postgresql/embedder checks are OK;
- Luxriot should be reachable if the local Evo loops are running;
- deployment security may be site-specific, but should not be ignored for client
  deployment.

## 10. Optional Live Smoke On Office Machine

Seed deterministic archive fixtures on a test/demo channel:

```bash
EVA_LIVE_CHANNEL_REF=112
.venv/bin/python scripts/seed_demo_fixtures.py --channel-id "$EVA_LIVE_CHANNEL_REF"
```

Admin live smoke. This can take a long time; 5-10 minutes per complex turn is
possible on LM Studio demo hardware.

```bash
EVA_LIVE_BASE_URL="$BASE_URL" \
EVA_LIVE_USER=admin EVA_LIVE_PASSWORD='[ADMIN_PASSWORD]' \
EVA_LIVE_CHANNEL_REF="$EVA_LIVE_CHANNEL_REF" \
EVA_LIVE_NEEDLE_QUERY="person lying on the ground at night" \
EVA_LIVE_PROBE_NAME="smoke: thumbs up gesture" \
EVA_LIVE_INCLUDE=seed \
.venv/bin/pytest -q tests/integration/test_live_agent.py -s
```

Non-admin smoke:

```bash
EVA_LIVE_OPERATOR_PASSWORD='[OPERATOR_PASSWORD]' \
.venv/bin/python scripts/bootstrap_live_smoke_operator.py --channel-id "$EVA_LIVE_CHANNEL_REF" --set-password --base-url "$BASE_URL"

EVA_LIVE_BASE_URL="$BASE_URL" \
EVA_LIVE_USER=operator-smoke EVA_LIVE_PASSWORD='[OPERATOR_PASSWORD]' \
EVA_LIVE_CHANNEL_REF="$EVA_LIVE_CHANNEL_REF" \
EVA_LIVE_INCLUDE=non_admin \
.venv/bin/pytest -q tests/integration/test_live_agent.py -s
```

## 11. Manual Test Handoff

Give the PM/intern:

```text
readiness/MANUAL_TEST_SCENARIO_0.8.2_OFFICE_DEMO_RU.md
```

Ask them to fill the channel map first:

- weapon/threat channel;
- street/public-order channel;
- lobby/entrance channel;
- normal/low-activity channel;
- 3-8 channel group for broad search.

Main manual focus:

- VLM alerts appear as structured events, not only prose.
- Agent reports are objective and include coverage/pipeline health.
- Evidence frames/thumbnails are accessible.
- Probe calibration uses archive P/N/M, visible negative/background prompts,
  warnings, and `safe_to_apply`.
- Probe/prompt changes remain preview-only until UI Apply.

## 12. Rollback

Fast rollback to previous commit:

```bash
cd "$APP_DIR"
sudo systemctl stop "$SERVICE"
git log --oneline -5
git checkout <PREVIOUS_GOOD_COMMIT>
sudo cp -a "${ENV_FILE}.bak.${TS}" "$ENV_FILE"   # if env was changed
sudo systemctl start "$SERVICE"
curl -k -sS "$BASE_URL/health" | jq
```

DB restore is normally not needed for this code-only patch. If DB restore is
required, stop and ask before restoring the dump.

## 13. What To Report Back

Send back:

```text
Office demo update result

Before commit:
After commit:
VERSION:
/health:
/ready:
pytest:
docs drift:
admin live smoke:
operator live smoke:
manual test file sent: yes/no
issues:
```
