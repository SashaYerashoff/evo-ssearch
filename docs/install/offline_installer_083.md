# EVA AI 0.8.3 Offline Installer

`scripts/install_eva_083.py` is the dry-run-first installer for a fresh host or
an existing EVA AI deployment. It does not run `git`, `apt`, or an online
`pip install`. A fresh installation therefore needs an offline wheelhouse in
the bundle; an upgrade may reuse its existing `.venv`.

The installer reuses the established field mechanisms instead of introducing
another update path:

- `preflight_patch.sh` captures the read-only baseline;
- `install_patch.sh` backs up PostgreSQL, env, unit, and code, then copies the
  release and installs wheels with `--no-index`;
- Alembic performs transactional migrations to `20260614_0006`;
- `verify_patch.sh` checks systemd, `/health`, and `/ready`;
- `rollback.sh` consumes the recorded backup if the deployment must be reverted.

## 1. Prepare the offline bundle

The bundle should contain:

```text
manifest.txt
repo/
wheelhouse/                 # required on a fresh host
```

The existing bundle builder can package a prepared wheelhouse:

```bash
scripts/build_patch_bundle.sh \
  --name eva-ai-0.8.3-offline \
  --wheelhouse-dir /path/to/prebuilt/wheelhouse
```

Building/downloading the wheelhouse happens on a connected preparation host,
never on the client server and never during installer tests.

The PostgreSQL server/database and the deployment/runtime login roles must
already be provisioned and reachable. The installer migrates EVA schemas; it
does not install PostgreSQL or invent site password policy.

## 2. Dry-run first

From the unpacked bundle:

```bash
cd eva-ai-0.8.3-offline/repo

./scripts/install_eva_083.py \
  --dry-run \
  --non-interactive \
  --source-dir "$PWD" \
  --bundle-dir "$PWD/.." \
  --app-dir /opt/eva-ai/evo-ssearch
```

Dry-run is the default even if `--dry-run` is omitted. It performs no writes,
does not stop a service, does not run migrations, and does not contact a package
index. The output contains only configuration key status (`[set]`/`[missing]`),
never passwords, API keys, or DSNs.

Review every `FAIL` and `WARN`. The installer refuses `--apply` while any
preflight failure remains.

Non-empty placeholder values such as `changeme`, `admin:123`, `example.com`,
`<...>`, or `[FIELD]` are failures, not configuration. Diagnostics identify the
key but redact the value.

## 3. Environment discovery and preservation

Use `--env-file` when the deployment path is known:

```bash
--env-file /etc/eva-ai/eva-ai.env
```

Without it, the installer searches the configured `EVA_ENV_FILE`, the canonical
`/etc/eva-ai/eva-ai.env`, then `eva-ai.env`/`.env` in the target app and source
directories. Existing content, comments, unknown keys, model routing, and
operator settings are preserved. The installer only appends keys that are
absent; it never replaces an existing value.

If no env exists, interactive mode asks for:

- Luxriot Evo URL, username, and password;
- separate PostgreSQL API, audit, and worker DSNs;
- a distinct privileged PostgreSQL migration DSN when migrations are enabled;
- agent LM endpoint and model;
- VLM endpoint and model.

```bash
./scripts/install_eva_083.py \
  --dry-run \
  --source-dir "$PWD" \
  --bundle-dir "$PWD/.." \
  --env-file /etc/eva-ai/eva-ai.env
```

For automation, create a mode-0600 env file first and use
`--non-interactive --env-file ...`. Non-interactive mode fails instead of
guessing a missing mandatory endpoint or secret.

## 4. PostgreSQL migration identity

By default the installer runs:

```text
alembic current -> alembic upgrade head -> alembic current
```

The existing Alembic environment uses one transaction per migration. A valid,
non-empty `postgres.dump` is mandatory before `upgrade head`; there is no unsafe
"skip backup" option.

`EVA_DATABASE_DSN` is intentionally a least-privilege, non-DDL API login. The
installer never falls back to it for `pg_dump` or Alembic. With migrations
enabled, supply a distinct privileged DSN either as persisted
`EVA_MIGRATION_DATABASE_DSN` in the reviewed env file or transiently as
`EVA_INSTALL_MIGRATION_DSN`. The transient form is not written into
`eva-ai.env`:

```bash
read -rsp 'Migration DSN: ' EVA_INSTALL_MIGRATION_DSN
echo
export EVA_INSTALL_MIGRATION_DSN

sudo --preserve-env=EVA_INSTALL_MIGRATION_DSN \
  ./scripts/install_eva_083.py \
    --apply \
    --non-interactive \
    --source-dir "$PWD" \
    --bundle-dir "$PWD/.." \
    --app-dir /opt/eva-ai/evo-ssearch \
    --env-file /etc/eva-ai/eva-ai.env

unset EVA_INSTALL_MIGRATION_DSN
```

`--no-migrate` exists for an explicitly reviewed code-only deployment. It is
not the normal installation path, but it does not require a migration DSN.

## 5. Apply the reviewed plan

Rerun the same dry-run command with `sudo` and `--apply`:

```bash
sudo ./scripts/install_eva_083.py \
  --apply \
  --non-interactive \
  --source-dir "$PWD" \
  --bundle-dir "$PWD/.." \
  --app-dir /opt/eva-ai/evo-ssearch \
  --env-file /etc/eva-ai/eva-ai.env \
  --service-name eva-ai \
  --service-user eva \
  --service-group eva
```

The apply path is idempotent:

- the existing env is retained and backed up;
- the target venv is reused, or created once;
- dependencies come only from `wheelhouse/` with `--no-index`;
- static/templates are copied with the release by `install_patch.sh`;
- the systemd unit is replaced only when its rendered content differs;
- the unit exports `EVOSSEARCH_CONFIG_ENV_FILE` with the same path as
  `EnvironmentFile`, so Settings can report pending-restart provenance without
  reading or exposing secrets;
- Alembic `upgrade head` is safe to repeat;
- systemd `enable`/`restart` is safe to repeat.

Apply is serialized by a nonblocking lock at
`/run/lock/eva-ai-083-installer.lock` (override with `--lock-file`). A second
apply fails before account/config/code/service mutation. Dry-run never creates
the lock file.

Use `--no-start` to leave the service stopped; this also suppresses endpoint
verification. Use `--no-verify` only when health verification will be performed
immediately by another approved mechanism.

## 6. Health and operator handoff

Successful apply runs the existing verifier with a 90-second timeout. Confirm:

```bash
systemctl status eva-ai --no-pager -l
curl -sS http://127.0.0.1:5000/health
curl -sS http://127.0.0.1:5000/ready
```

For a fresh deployment, create the first named admin after the service and DB
are ready:

```bash
set -a
. /etc/eva-ai/eva-ai.env
set +a
/opt/eva-ai/evo-ssearch/.venv/bin/python \
  /opt/eva-ai/evo-ssearch/scripts/bootstrap_admin.py
```

## 7. Rollback handoff

The installer prints `backup_dir` and an exact rollback command. It also writes
`offline-installer-state.txt` into that backup, recording whether the app, env,
and unit existed before installation without recording any secret.

For an upgrade rollback:

```bash
sudo /opt/eva-ai/evo-ssearch/scripts/rollback.sh \
  --backup-dir /var/backups/eva-ai/patch-YYYYMMDD-HHMMSS \
  --app-dir /opt/eva-ai/evo-ssearch \
  --env-file /etc/eva-ai/eva-ai.env \
  --service eva-ai
```

Database restore is deliberately separate and destructive. Use
`--restore-db` only after reviewing the dump and setting the confirmation
required by `rollback.sh`.

For a fresh install (`installation_mode=fresh`), `rollback.sh` restores the
captured pre-copy tree but intentionally preserves `.venv` and runtime-data
paths. After rollback, the responsible engineer may disable/remove the new
unit, env, and empty target directory according to the three `*_preexisted`
flags in `offline-installer-state.txt`. This cleanup is never automatic because
the installer must not delete newly accumulated evidence or operator data.
