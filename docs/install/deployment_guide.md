# Deployment Guide (sanitized)

Third-party-shareable install guide for an EVA AI pilot on Ubuntu LTS. **Contains
no client data** — all site specifics are placeholders. The filled, internal
version lives in `install/field_rollout_demo.md` (not shared).

Invariants: [facts](../00_CANON/facts.md). All variables:
[config_reference](../00_CANON/config_reference.md). Placeholders: `<...>`.

## Prerequisites

- Ubuntu LTS host for the app (also runs CLIP + the agent LM client + can co-host
  PostgreSQL for a single-node pilot).
- One or more GPU hosts running vLLM for the VLM (`qwen3-vl-4b`), and an agent LM
  endpoint (`qwen3.5-9b` class). See [inference_topology](inference_topology.md).
- PostgreSQL reachable, with the EVA roles/schemas (API/audit/worker/migration).
- Luxriot Evo reachable on the closed network.
- Closed network (the pilot security model assumes isolation).

## 1. Install code & dependencies

```bash
git clone <repo> /opt/eva-ai && cd /opt/eva-ai
python -m venv .venv && . .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
# (Closed network: install from an offline wheelhouse instead.)
```

## 2. Database

```bash
# create roles/schemas, then migrate to head
set -a; . /etc/eva-ai/eva-ai.env; set +a
alembic upgrade head
alembic current   # expect: 20260614_0006
```

Use `scripts/bootstrap_db_roles.py` for the separated runtime roles `[VERIFY]`.

## 3. Configure `.env` (secrets, mode 0600)

Set the **secure-pilot required set** (see config_reference) plus connection
details:

```env
EVOSSEARCH_SECURE_DEPLOYMENT_REQUIRED=true
EVOSSEARCH_AUTH_ENABLED=true
EVA_DB_STRICT_RUNTIME_ROLES=true
EVOSSEARCH_ARCHIVE_STORE=postgres
EVOSSEARCH_GUNICORN_WORKERS=1
EVOSSEARCH_AUTH_COOKIE_SECURE=true

EVOSSEARCH_PORT=<port>            # e.g. 5443 (TLS)
EVA_DATABASE_DSN=<...>
EVA_AUDIT_DATABASE_DSN=<...>
EVA_WORKER_DATABASE_DSN=<...>
EVOSSEARCH_LUXRIOT_BASE_URL=<...>
EVOSSEARCH_LUXRIOT_USERNAME=<...>
EVOSSEARCH_LUXRIOT_PASSWORD=<...>
# LM profiles (agent + vlm) — see inference_topology
```

## 4. First admin & verification

```bash
python scripts/bootstrap_admin.py   # create the first admin  [VERIFY]
```

- Start the service (systemd unit, gunicorn via `run_prod.sh` → single worker,
  `gunicorn_conf.py` hooks).
- `GET /health` → ok; `GET /ready` → all components ready.

## 5. First-run setup

1. Log in as admin; create operator/viewer accounts; assign **channel grants**.
2. In the Video tab, pick channels and confirm a live preview.
3. Start a video-description session on one channel with a conservative cadence.
4. Confirm L0 descriptions appear and (if any trigger) alert badges show.
5. Confirm video-description frames appear in Archive Research as
   "video-description frame".
6. Verify operator/viewer UI hiding and channel scope.
7. Only then scale toward the target channel count.

## 6. Running

```bash
./run_prod.sh   # gunicorn, 1 worker (gthread), --config gunicorn_conf.py
```

Single worker is **required** (in-process schedulers). See
[system_architecture](../architecture/system_architecture.md).

## 7. Upgrades

- **Code-only patch** (e.g. β 0.8.0 → 0.8.1): apply files, restart. No migration.
- **DB-touching patch:** run `alembic upgrade head` explicitly; the installer must
  refuse unsafe startup if DB revision < code-expected revision.
- Always reversible via `scripts/rollback.sh`.
- For closed-network field patches via USB, follow the offline patch SOP
  (internal field-rollout doc).

## 8. Operations

- Health/coverage: [observability](../admin/observability.md).
- Backup/restore: [backup_recovery](../admin/backup_recovery.md).
- Users/roles/retention: [admin_guide](../admin/admin_guide.md).
