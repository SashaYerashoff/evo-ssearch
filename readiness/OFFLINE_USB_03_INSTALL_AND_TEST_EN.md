# EVA AI offline update: 03 - deployment scenarios and manual post-install test

Goal: install the offline patch, verify the service, and run the minimum manual
test required before field/demo use.

Target release: `β 0.8.3`  
Schema head: `20260614_0006`  
Database migration for `β 0.8.2.1 -> β 0.8.3`: **none**

## 1. Common variables

Inside the unpacked bundle:

```bash
cd ~/eva-ai-patch/eva-ai-patch-0.8.3-*

export APP_DIR=/opt/eva-ai/evo-ssearch
export SERVICE=eva-ai
export ENV_FILE=/etc/eva-ai/eva-ai.env
export BASE_URL=http://127.0.0.1:5000
export DB_NAME=eva
```

If preflight showed HTTPS:

```bash
export BASE_URL=https://127.0.0.1:5443
export EVA_PATCH_CURL_INSECURE=true
```

## 2. Scenario A: standard offline install

Use this when preflight is OK, disk space is sufficient, checksum is OK, and
the bundle includes a wheelhouse or the existing `.venv` is known-good.

```bash
sudo scripts/install_patch.sh \
  --bundle-dir "$PWD" \
  --app-dir "$APP_DIR" \
  --env-file "$ENV_FILE" \
  --service "$SERVICE" \
  --base-url "$BASE_URL" \
  --pg-database "$DB_NAME"
```

For `β 0.8.3`, do not add `--run-migrations`: there is no migration.

## 3. Scenario B: code-only / reuse existing venv

Use only when the bundle has no wheelhouse, there is no internet, but preflight
confirmed that `.venv/bin/python` exists and the current installation worked.

Use the same command as Scenario A. The installer preserves the existing
`.venv`. If the bundle has no wheelhouse, dependencies will not be reinstalled.

```bash
sudo scripts/install_patch.sh \
  --bundle-dir "$PWD" \
  --app-dir "$APP_DIR" \
  --env-file "$ENV_FILE" \
  --service "$SERVICE" \
  --base-url "$BASE_URL" \
  --pg-database "$DB_NAME"
```

After install, pay extra attention to `/health`, `/ready`, and module import
checks through `verify_patch.sh`.

## 4. Scenario C: install without pg_dump

Use only when:

- there is a current external backup, or
- the database contains no critical data, or
- the responsible engineer explicitly approved skipping DB dump.

```bash
sudo scripts/install_patch.sh \
  --bundle-dir "$PWD" \
  --app-dir "$APP_DIR" \
  --env-file "$ENV_FILE" \
  --service "$SERVICE" \
  --base-url "$BASE_URL" \
  --pg-database "$DB_NAME" \
  --skip-pg-dump
```

This still backs up code/env/systemd. Only PostgreSQL dump is skipped.

## 5. Verify after install

```bash
scripts/verify_patch.sh \
  --service "$SERVICE" \
  --base-url "$BASE_URL" \
  --timeout 90
```

For HTTPS with a self-signed certificate:

```bash
EVA_PATCH_CURL_INSECURE=true scripts/verify_patch.sh \
  --service "$SERVICE" \
  --base-url "$BASE_URL" \
  --timeout 90
```

Check systemd:

```bash
systemctl status "$SERVICE" --no-pager -l
journalctl -u "$SERVICE" -n 120 --no-pager
```

Check endpoints:

```bash
curl -sS "$BASE_URL/health" | jq
curl -sS "$BASE_URL/ready" | jq '.status, .checks.postgresql, .checks.authentication, .checks.luxriot, .checks.lm_profiles'
```

For HTTPS, add `-k`:

```bash
curl -k -sS "$BASE_URL/health" | jq
curl -k -sS "$BASE_URL/ready" | jq '.status, .checks.postgresql, .checks.authentication, .checks.luxriot, .checks.lm_profiles'
```

Expected:

- `/health.version` = `β 0.8.3`;
- service active;
- PostgreSQL/auth checks ready;
- Luxriot reachable if Evo server is online;
- LM/VLM profiles reachable if inference servers are online;
- `inference_queue.status=disabled` is acceptable for the current pilot.

## 6. Verify model routing after install

In `/ready` or Admin Settings, verify:

- VLM/video-description profiles: vLLM servers with `qwen3-vl-4b-fp8`;
- agent/chat profile: LM Studio / EVA AI host with `qwen3.5-9b-mtp`;
- `.env` did not overwrite the user's UI model selection.

Command:

```bash
curl -sS "$BASE_URL/ready" \
  | jq '.checks.lm_profiles'
```

If the response does not show all profiles, check in the UI:

```text
Admin / Settings -> LM profiles / Video descriptions / Agent model
```

## 7. Minimum manual UI test

Open EVA AI in a browser or Luxriot EVO Monitor web tile. Hard-refresh after
installation.

### 7.1 Login and version

1. Log in as an operator/admin user.
2. Confirm the UI opens without infinite loading.
3. Confirm version `β 0.8.3`, if shown in UI/status.

### 7.2 Video tab / live signal honesty

1. Open video monitoring.
2. Select a live channel.
3. Confirm preview updates and does not replay old buffered video.
4. If a channel is disabled in Luxriot, UI must show `Signal lost` /
   `No fresh EVA frame`, not an old image.
5. While VLM processes a batch, `slow` / `processing delay` is acceptable; it
   must not become a false `signal lost`.

### 7.3 VLM feed and alerts

1. Check recent video summaries.
2. Check that alerts have evidence/thumbnail when an event was detected.
3. For road/street channels, road/drift outputs must be candidate/evidence
   wording, not legal conclusions.

### 7.4 Road mask / grounding

1. On a road channel, click `Ground road mask`.
2. Overlay must use fresh EVA frames.
3. If no fresh frames are available, UI must show an error, not an old image.

### 7.5 Agent

Ask:

```text
Show recent VLM alerts and notable video-summary events for the last hour.
```

Expected:

- agent uses video-description tools;
- agent reports coverage / partial coverage;
- agent does not claim the whole period was reviewed when coverage is partial.

Ask:

```text
Check live video-description channel status and report signal problems.
```

Expected:

- agent reports live signal/runtime problems;
- agent distinguishes detected/delivered/cooldown/failed when data exists;
- agent does not hide disabled/frozen/stale channels as calm.

### 7.6 Probe preview/apply

1. Ask the agent to create or modify a test probe.
2. A separate preview/apply card must appear.
3. Apply must go through UI approval, not direct model execution.
4. A receipt appears after Apply.

### 7.7 Archive evidence modal

1. Open found alert/evidence.
2. Confirm image is displayed.
3. If alert anchor is not the best frame, batch-frame navigation should allow
   viewing nearby frames.
4. `Open VLM feed` should navigate to the relevant time/context, not only
   switch tabs.

## 8. Rollback

If the service does not start or `/health` does not come up:

```bash
sudo scripts/rollback.sh \
  --app-dir "$APP_DIR" \
  --env-file "$ENV_FILE" \
  --service "$SERVICE" \
  --base-url "$BASE_URL" \
  --pg-database "$DB_NAME"
```

Restore PostgreSQL dump only after explicit approval:

```bash
sudo EVA_PATCH_CONFIRM_DB_RESTORE=yes scripts/rollback.sh \
  --app-dir "$APP_DIR" \
  --env-file "$ENV_FILE" \
  --service "$SERVICE" \
  --base-url "$BASE_URL" \
  --pg-database "$DB_NAME" \
  --restore-db
```

After rollback:

```bash
scripts/verify_patch.sh --service "$SERVICE" --base-url "$BASE_URL" --timeout 90
```

## 9. Send back after the work

Without secrets:

- `manifest.txt` photo/text;
- `sha256sum -c` result;
- preflight result;
- selected scenario A/B/C command;
- `/health` JSON;
- short `/ready` summary;
- `systemctl status eva-ai --no-pager -l`;
- 2-3 UI screenshots: live channel, disabled/stale channel, agent status/alert report;
- rollback backup directory and reason, if rollback was used.

