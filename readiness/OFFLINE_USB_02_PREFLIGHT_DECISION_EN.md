# EVA AI offline update: 02 - preflight and deployment scenario decision

Goal: before stopping the service, decide whether the patch can be installed,
which installation scenario to use, and which risks must be recorded.

Target release: `β 0.8.3`  
Schema head: `20260614_0006`  
Database migration for `β 0.8.2.1 -> β 0.8.3`: **none**

## 1. Known client-site parameters

Use these values as the starting point. If the real site differs, record the
difference and use the real value in commands.

| Parameter | Current client/site record |
| --- | --- |
| EVA AI app dir | `/opt/eva-ai/evo-ssearch` |
| systemd service | `eva-ai` |
| service user/group | `eva:eva` |
| env file | `/etc/eva-ai/eva-ai.env` |
| internal app URL | `http://127.0.0.1:5000` |
| PostgreSQL DB fallback name | `eva` |
| expected schema | `20260614_0006` |
| expected app version after patch | `β 0.8.3` |
| worker count | `EVOSSEARCH_GUNICORN_WORKERS=1` |
| Luxriot Evo observed pilot URL | `http://192.168.3.27:8080` |
| inference A | `192.168.3.104`, ports `8001` / `8002` |
| inference B | `192.168.3.11`, ports `8001` / `8002` |
| VLM model | `qwen3-vl-4b-fp8` |
| agent model | `qwen3.5-9b-mtp` |

Important: `https://127.0.0.1:5443` is for local/dev or a separate TLS-enabled
service. On the client control-plane, use `http://127.0.0.1:5000` unless
`systemctl status eva-ai` shows a different bind.

## 2. Run safe preflight

Go to the unpacked bundle:

```bash
cd ~/eva-ai-patch/eva-ai-patch-0.8.3-*
```

Run preflight. It does not stop services, copy code, or edit configuration.

```bash
sudo scripts/preflight_patch.sh \
  --bundle-dir "$PWD" \
  --app-dir /opt/eva-ai/evo-ssearch \
  --env-file /etc/eva-ai/eva-ai.env \
  --service eva-ai \
  --base-url http://127.0.0.1:5000 \
  --pg-database eva \
  --expected-version "β 0.8.3" \
  --expected-schema 20260614_0006
```

If the site uses a local HTTPS service:

```bash
sudo EVA_PATCH_CURL_INSECURE=true scripts/preflight_patch.sh \
  --bundle-dir "$PWD" \
  --app-dir /opt/eva-ai/evo-ssearch \
  --env-file /etc/eva-ai/eva-ai.env \
  --service eva-ai \
  --base-url https://127.0.0.1:5443 \
  --pg-database eva \
  --expected-version "β 0.8.3" \
  --expected-schema 20260614_0006
```

## 3. Save from preflight

Save the command output or screenshots, especially:

- bundle manifest;
- current `VERSION`;
- systemd service exists;
- `WorkingDirectory`, `User`, `Group`;
- venv python exists;
- PostgreSQL / schema revision;
- free disk space;
- `/health` and `/ready`;
- wheelhouse presence;
- warnings and failures.

Do not send the full `/etc/eva-ai/eva-ai.env`.

## 4. Decision table

| Preflight result | Action |
| --- | --- |
| Required checks OK, `wheelhouse found`, enough disk space | Scenario A: standard offline install with backup. |
| No `wheelhouse`, but existing `.venv` exists and the current install worked | Scenario B: code-only / reuse existing venv. Acceptable for urgent patching; record the risk. |
| `pg_dump` unavailable, but there is an external backup or no critical data | Scenario C: install with `--skip-pg-dump` only after responsible approval. |
| `/ready` was already `not_ready` because Luxriot/LM/vLLM/network was down | Install may proceed, but record the baseline. Do not blame old external issues on the patch. |
| service `eva-ai` was not active before install | Proceed only if expected. Record it and verify service startup after install. |
| app dir, service name, or env file differs | Do not use default commands. Pass real `--app-dir`, `--service`, `--env-file`. |
| schema revision is not `20260614_0006` | Stop. Engineering review required. This patch should not change schema. |
| not enough space for backup / bundle / wheelhouse | Stop. Free space or choose another backup root. |
| checksum is not OK | Stop. Bundle is damaged or copied incompletely. |
| manifest version is not `β 0.8.3` | Stop. Use the correct bundle. |

## 5. Choose base URL

Check how the service listens:

```bash
systemctl status eva-ai --no-pager -l
```

If it shows:

```text
Listening at: http://0.0.0.0:5000
```

use:

```text
http://127.0.0.1:5000
```

If it shows:

```text
Listening at: https://0.0.0.0:5443
```

use:

```text
https://127.0.0.1:5443
```

and set `EVA_PATCH_CURL_INSECURE=true` for self-signed TLS checks.

## 6. Check model routing before install

If the service is running:

```bash
curl -sS http://127.0.0.1:5000/ready \
  | jq '.checks.lm_profiles'
```

For local HTTPS:

```bash
curl -k -sS https://127.0.0.1:5443/ready \
  | jq '.checks.lm_profiles'
```

Expected routing:

- VLM/video-description profiles point to vLLM servers with
  `qwen3-vl-4b-fp8`;
- agent/chat profile points to EVA AI / LM Studio host with
  `qwen3.5-9b-mtp`;
- `.env` should not overwrite the operator's model selection in UI.

If `/ready` does not expose enough details, verify in Admin Settings after
install. Do not edit `.env` blindly.

## 7. Ready for install

Before moving to document `03`, know these values:

```text
APP_DIR=/opt/eva-ai/evo-ssearch
SERVICE=eva-ai
ENV_FILE=/etc/eva-ai/eva-ai.env
BASE_URL=http://127.0.0.1:5000
DB_NAME=eva
SCENARIO=A|B|C
```

Next:

```text
readiness/OFFLINE_USB_03_INSTALL_AND_TEST_EN.md
```

