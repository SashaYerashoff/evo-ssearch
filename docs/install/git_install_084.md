# EVA AI 0.8.4 — Git Installation and Update Guide

This guide installs the reviewed 50-channel EVA AI line from Git on a connected
Ubuntu host. All terminal output and operator prompts in this procedure are in
English.

## Release source of truth

```text
Repository: https://github.com/SashaYerashoff/evo-ssearch.git
Branch:     feature/secure-50-channel-foundation
Version:    β 0.8.4
```

The branch name selects the product line, but a branch is mutable. Record the
exact commit printed by `git rev-parse HEAD` in the installation record. For a
client deployment, install only a commit reviewed by the EVA AI team.

Do not run `git pull` inside a live `/opt/eva-ai/evo-ssearch` installation. Use
a separate source checkout and the dry-run-first installer below so that the
existing environment, service definition, code backup, and rollback handoff are
preserved.

The installer entry point retains its historical filename
`install_eva_083.py`; its current source-version gate requires β 0.8.4.

## Scope and prerequisites

This procedure installs the EVA AI control plane. It does not install or
configure Luxriot Evo, NVIDIA drivers, Agent/VLM inference servers, PostgreSQL
roles, TLS certificates, or model weights.

Required before installation:

- Ubuntu LTS on amd64 with `git`, Python 3, `python3-venv`, `pip`, `curl`, `jq`,
  `ffmpeg`, and the PostgreSQL client tools including `pg_dump`;
- a reachable Luxriot Evo server;
- reachable OpenAI-compatible Agent and VLM endpoints;
- a PostgreSQL database with separate API, audit, and worker login roles;
- a privileged migration DSN for a fresh database or an approved migration;
- root access on the EVA AI host;
- `/etc/eva-ai/eva-ai.env` populated with real site values and mode `0600`.

See [Configuration Reference](../00_CANON/config_reference.md) and
[Inference Topology](inference_topology.md) before filling the environment.
Never commit the environment file or paste its secrets into an installation
record.

## 1. Clone the correct branch into a staging directory

Run as the logged-in maintenance user:

```bash
export EVA_REPO='https://github.com/SashaYerashoff/evo-ssearch.git'
export EVA_BRANCH='feature/secure-50-channel-foundation'
export EVA_SOURCE="/var/tmp/eva-ai-git-source-$USER"

rm -rf "$EVA_SOURCE"
git clone --branch "$EVA_BRANCH" --single-branch "$EVA_REPO" "$EVA_SOURCE"
cd "$EVA_SOURCE"

git status --short --branch
git branch --show-current
git rev-parse HEAD
git show -s --format='%H %s' HEAD
cat VERSION
```

Expected results:

- the branch is `feature/secure-50-channel-foundation`;
- the working tree is clean;
- `VERSION` reports `β 0.8.4`;
- the full commit SHA is copied into the installation record.

Stop if any of these checks fail.

## 2A. Update an existing EVA AI installation

Use this path when `/opt/eva-ai/evo-ssearch/.venv`, the systemd service, the
environment file, and database already belong to a working EVA AI deployment.
The 0.8.4 code expects schema revision `20260614_0006`.

First inspect the active deployment without changing it:

```bash
sudo systemctl show eva-ai.service \
  -p FragmentPath -p WorkingDirectory -p EnvironmentFiles -p MainPID \
  --no-pager

sudo systemctl is-active eva-ai.service
curl -sS --max-time 10 http://127.0.0.1:5000/ready | jq -c .
```

If the service uses a different application path, env file, service name, or
health URL, replace those values in every command below.

Run the installer in dry-run mode. This command does not stop the service,
write files, run migrations, or change the database:

```bash
cd "$EVA_SOURCE"

sudo ./scripts/install_eva_083.py \
  --dry-run \
  --non-interactive \
  --source-dir "$EVA_SOURCE" \
  --app-dir /opt/eva-ai/evo-ssearch \
  --env-file /etc/eva-ai/eva-ai.env \
  --service-name eva-ai \
  --service-user eva \
  --service-group eva \
  --base-url http://127.0.0.1:5000 \
  --no-migrate \
  --verified-adopt-existing-config
```

Review all output. `FAIL` or `BLOCKED` means nothing should be installed. A
warning that no wheelhouse is present is expected only when the existing target
`.venv` is healthy and the exact requirements are already satisfied.

After the dry-run passes, repeat the reviewed command with `--apply`:

```bash
cd "$EVA_SOURCE"

sudo ./scripts/install_eva_083.py \
  --apply \
  --non-interactive \
  --source-dir "$EVA_SOURCE" \
  --app-dir /opt/eva-ai/evo-ssearch \
  --env-file /etc/eva-ai/eva-ai.env \
  --service-name eva-ai \
  --service-user eva \
  --service-group eva \
  --base-url http://127.0.0.1:5000 \
  --no-migrate \
  --verified-adopt-existing-config
```

The installer prints the backup directory and rollback command. Save both in
the installation record before closing the terminal.

## 2B. Install on a fresh EVA AI host

Use this path only after the site database, runtime roles, inference endpoints,
Luxriot credentials, storage, and TLS plan are ready.

Create the service account and target virtual environment, then install Python
dependencies from the connected source host:

```bash
getent group eva >/dev/null || sudo groupadd --system eva
id -u eva >/dev/null 2>&1 || sudo useradd --system --gid eva \
  --home-dir /opt/eva-ai --create-home --shell /usr/sbin/nologin eva
sudo install -d -o eva -g eva -m 0755 /opt/eva-ai/evo-ssearch
sudo install -d -o root -g root -m 0750 /etc/eva-ai
sudo install -d -o eva -g eva -m 0750 /var/lib/eva-ai

sudo -u eva python3 -m venv /opt/eva-ai/evo-ssearch/.venv
sudo -u eva /opt/eva-ai/evo-ssearch/.venv/bin/python -m pip install \
  --upgrade pip setuptools wheel
sudo -u eva /opt/eva-ai/evo-ssearch/.venv/bin/pip install \
  -r "$EVA_SOURCE/requirements.txt" \
  -r "$EVA_SOURCE/requirements-db.txt"
```

Create the site environment without exposing secrets in Git:

```bash
sudo install -o root -g root -m 0600 /dev/null /etc/eva-ai/eva-ai.env
sudoedit /etc/eva-ai/eva-ai.env
```

At minimum, configure the secure deployment flags, tenant UUID, three runtime
DSNs, Luxriot connection, Agent profile, VLM profile(s), archive path, and
retention. Use the canonical configuration reference rather than copying values
from another client.

Supply the privileged migration DSN only for the installer process:

```bash
read -rsp 'Privileged PostgreSQL migration DSN: ' EVA_INSTALL_MIGRATION_DSN
echo
export EVA_INSTALL_MIGRATION_DSN
```

Run the fresh-install dry-run:

```bash
cd "$EVA_SOURCE"

sudo --preserve-env=EVA_INSTALL_MIGRATION_DSN \
  ./scripts/install_eva_083.py \
    --dry-run \
    --non-interactive \
    --source-dir "$EVA_SOURCE" \
    --app-dir /opt/eva-ai/evo-ssearch \
    --env-file /etc/eva-ai/eva-ai.env \
    --service-name eva-ai \
    --service-user eva \
    --service-group eva \
    --base-url http://127.0.0.1:5000
```

Do not continue until every `FAIL` is resolved. The apply stage requires a
successful non-empty PostgreSQL backup before Alembic may change the schema.

Repeat the same reviewed command with `--apply`:

```bash
cd "$EVA_SOURCE"

sudo --preserve-env=EVA_INSTALL_MIGRATION_DSN \
  ./scripts/install_eva_083.py \
    --apply \
    --non-interactive \
    --source-dir "$EVA_SOURCE" \
    --app-dir /opt/eva-ai/evo-ssearch \
    --env-file /etc/eva-ai/eva-ai.env \
    --service-name eva-ai \
    --service-user eva \
    --service-group eva \
    --base-url http://127.0.0.1:5000

unset EVA_INSTALL_MIGRATION_DSN
```

If apply fails, keep the terminal output and use only the exact rollback command
printed by the installer. Do not improvise a database restore.

## 3. Create the first administrator on a fresh host

Skip this section for an existing deployment.

```bash
sudo bash -lc '
  set -a
  . /etc/eva-ai/eva-ai.env
  set +a
  cd /opt/eva-ai/evo-ssearch
  exec sudo -u eva -E .venv/bin/python scripts/bootstrap_admin.py
'
```

Do not share the generated/admin password in chat or in the installation record.

## 4. Verify service, API, inference, and source identity

```bash
sudo systemctl status eva-ai.service --no-pager -l
sudo journalctl -u eva-ai.service -n 100 --no-pager

curl -sS --max-time 10 http://127.0.0.1:5000/health | jq -c .
curl -sS --max-time 10 http://127.0.0.1:5000/ready | jq -c .

cd "$EVA_SOURCE"
printf 'installed_source_commit=%s\n' "$(git rev-parse HEAD)"
```

Acceptance requires:

- systemd reports `active (running)`;
- `/health` is healthy and `/ready` reports β 0.8.4;
- Agent and VLM endpoints in `/ready` are the intended site endpoints/models;
- the operator can log in over the intended HTTPS URL;
- one granted channel has live preview and a completed VLM description;
- an Agent query completes with evidence;
- archive playback and a stored evidence thumbnail both work.

Only after this smoke test should channels be enabled in batches toward the
accepted site capacity.

## 5. Future Git updates

Refresh only the staging checkout, inspect the new commit, and repeat the
dry-run/apply procedure:

```bash
cd "$EVA_SOURCE"
git fetch origin "$EVA_BRANCH"
git status --short
git diff --stat HEAD.."origin/$EVA_BRANCH"
git log --oneline --decorate HEAD.."origin/$EVA_BRANCH"
git merge --ff-only "origin/$EVA_BRANCH"
git rev-parse HEAD
```

If `git status --short` is not empty or the merge is not fast-forward, stop and
ask the EVA AI team to prepare a clean reviewed source checkout.
