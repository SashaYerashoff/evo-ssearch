# Deployment Guide (sanitized)

Third-party-shareable install guide for an EVA AI pilot on Ubuntu LTS. **Contains
no client data** — all site specifics are placeholders. The filled, internal
version lives in `install/field_rollout_demo.md` (not shared).

For the reviewed β 0.8.5 source branch, release-specific migration and
verification constraints are in the
[β 0.8.5 release notes](../../readiness/RELEASE_NOTES_0.8.5.md). The older
[0.8.4 Git guide](git_install_084.md) is historical and must not be reused as a
0.8.5 upgrade recipe because 0.8.5 adds database migrations. Do not run
`git pull` inside a live production checkout.

Invariants: [facts](../00_CANON/facts.md). All variables:
[config_reference](../00_CANON/config_reference.md). Production browser/internal
URL rules: [production_settings](production_settings.md). Placeholders: `<...>`.

## Offline single-node port appliance

For the reviewed RTX 4070 Super / Intel i9 14th Gen / 64-GB target, use the
generated `EVA-AI-0.8.5-PORT` directory on the writable partition of the Ubuntu
Server 24.04 installer USB:

```bash
cd EVA-AI-0.8.5-PORT
sha256sum -c SHA256SUMS
./install.sh
```

The English wizard asks for Evo connectivity and credentials, installation
paths, local versus external inference endpoints, the optional preemptible 9B
quiet window, and the first EVA administrator. It checks free space, GPU and
PostgreSQL/Alembic state before changing the host. A clean local install uses
Qwen3-VL-4B AWQ/vLLM on the GPU, Qwen3.5-9B-MTP Q4/llama.cpp on CPU for deep
review, and CPU/iGPU for dense CV plus continuous one-hertz CLIP indexing.

The USB is self-contained: local APT repository, Python 3.12 wheels, NVIDIA
driver/HWE kernel, PostgreSQL 16, Nginx/TLS support, both model payloads, CLIP
weights, and llama.cpp source are included. The default paths are:
`/opt/eva-ai`, `/var/lib/eva-ai`, `/etc/eva-ai`, and
`/var/backups/eva-ai`.

## Prerequisites

- Ubuntu LTS host for the app (also runs CLIP + the agent LM client + can co-host
  PostgreSQL for a single-node pilot).
- One or more GPU hosts running vLLM for `qwen3-vl-4b`. The constrained
  eight-channel appliance may share it for VLM + agent profiles under protected
  admission; scale-out deployments may add a separate agent endpoint and an
  optional quiet-window 9B deep-L3 endpoint. See
  [inference_topology](inference_topology.md).
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
alembic current   # expect: 20260727_0010
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

EVOSSEARCH_HOST=127.0.0.1         # recommended when Nginx/reverse proxy fronts the app
EVOSSEARCH_PORT=5000              # internal Gunicorn HTTP port
EVA_DATABASE_DSN=<...>
EVA_AUDIT_DATABASE_DSN=<...>
EVA_WORKER_DATABASE_DSN=<...>
EVOSSEARCH_LUXRIOT_BASE_URL=<...>
EVOSSEARCH_LUXRIOT_USERNAME=<...>
EVOSSEARCH_LUXRIOT_PASSWORD=<...>
# LM profiles (agent + vlm) — see inference_topology
```

Production browser access should be HTTPS/TLS at the reverse proxy or site
boundary, for example:

```
browser https://<eva-host>/  ->  Nginx/TLS  ->  http://127.0.0.1:5000
```

`run_prod.sh` does not create TLS by itself; it starts Gunicorn over HTTP. If a
temporary lab/demo opens `http://<eva-host>:5000` directly, set
`EVOSSEARCH_AUTH_COOKIE_SECURE=false` or browser login cookies will not work.
Do not keep HTTP-only mode for a client-facing deployment.

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

- **Code-only patch** (e.g. β 0.8.1 → 0.8.2): apply files, restart. No migration.
- **DB-touching patch:** run `alembic upgrade head` explicitly; the installer must
  refuse unsafe startup if DB revision < code-expected revision.
- **β 0.8.4 → β 0.8.5:** back up PostgreSQL and apply Alembic revisions
  `20260725_0007` through `20260727_0010`; this is not a code-only update.
- Always reversible via `scripts/rollback.sh`.
- For closed-network field patches via USB, follow the offline patch SOP
  (internal field-rollout doc).

## 8. Operations

- Health/coverage: [observability](../admin/observability.md).
- Backup/restore: [backup_recovery](../admin/backup_recovery.md).
- Users/roles/retention: [admin_guide](../admin/admin_guide.md).
