# EVA PostgreSQL foundation runbook

Date: 2026-06-10

## Scope

This foundation provides the control-plane schemas for:

- users, roles, permissions, sessions, and channel grants;
- agent sessions, tool runs, plans, and one-time approvals;
- append-only security audit events;
- bounded inference jobs, attempts, and an outbox.

Existing detection SQLite and JSON stores remain active until their import and
repository adapters are implemented.

## Install

```bash
.venv/bin/pip install -r requirements-db.txt
```

Set a privileged deployment PostgreSQL DSN through `EVA_DATABASE_DSN` or
`EVOSSEARCH_DATABASE_DSN` only while running migrations and role bootstrap.
Do not place the DSN in source control or expose it through the settings API.

```bash
export EVA_DATABASE_DSN='postgresql://postgres:...@db/eva'
```

After migrations, create least-privilege login principals. Passwords are read
from environment variables and are not printed:

```bash
export EVA_MIGRATOR_PASSWORD='...'
export EVA_API_PASSWORD='...'
export EVA_AUDIT_PASSWORD='...'
export EVA_WORKER_PASSWORD='...'
export EVA_BACKUP_PASSWORD='...'
.venv/bin/python scripts/bootstrap_db_roles.py
```

Switch runtime DSNs to the login principals and enable strict runtime role
checks:

```bash
export EVA_DATABASE_DSN='postgresql://eva_api_login:...@db/eva'
export EVA_AUDIT_DATABASE_DSN='postgresql://eva_audit_login:...@db/eva'
export EVA_WORKER_DATABASE_DSN='postgresql://eva_worker_login:...@db/eva'
export EVOSSEARCH_DB_STRICT_RUNTIME_ROLES=true
```

Configure a separate login principal and DSN for the append-only writer:

```bash
export EVA_AUDIT_DATABASE_DSN='postgresql://eva_audit_login:...@db/eva'
```

Grant the application login membership in `eva_api` and the audit login
membership in `eva_audit_writer`. Do not grant both roles to the same
principal.

Inference workers use a third login principal granted only `eva_worker`.
Construct API-side and worker-side
`PostgresInferenceQueueRepository` instances with their respective bounded
pools; do not share a privileged DSN between them.

## Migrate

```bash
EVA_DATABASE_DSN='postgresql://...' .venv/bin/alembic upgrade head
```

Expected revision:

```text
20260610_0004
```

## Bootstrap named-user authentication

Set one stable tenant UUID and enable authentication:

```bash
export EVOSSEARCH_AUTH_TENANT_ID='00000000-0000-0000-0000-000000000000'
export EVOSSEARCH_AUTH_ENABLED=true
export EVOSSEARCH_AUTH_COOKIE_SECURE=true
```

Use a generated UUID, not the zero UUID shown above. With the database DSN
configured, create the first administrator:

```bash
.venv/bin/python scripts/bootstrap_admin.py \
  --tenant-id "$EVOSSEARCH_AUTH_TENANT_ID" \
  --username admin
```

The command prompts for the password without placing it in shell history.
`EVA_BOOTSTRAP_ADMIN_PASSWORD` is available for one-time unattended
provisioning and must be removed immediately afterward.

`EVOSSEARCH_AUTH_COOKIE_SECURE=true` requires HTTPS. Set it to `false` only for
local HTTP development.

Use the operational CLI for pilot users, roles, channel grants, and session
revocation:

```bash
.venv/bin/python scripts/manage_users.py list
EVA_USER_PASSWORD='temporary password value' \
  .venv/bin/python scripts/manage_users.py create operator-1 \
  --role operator \
  --channels 1,2,3,4
.venv/bin/python scripts/manage_users.py revoke-sessions operator-1
```

The application `/ready` response includes a required `postgresql` component
when a DSN is configured. It distinguishes an unavailable database from a
schema revision mismatch and reports `runtime_user`, `runtime_role_ok`, and
`runtime_unsafe_reason`. With `EVOSSEARCH_DB_STRICT_RUNTIME_ROLES=true`,
`postgres`, `eva_owner`, `eva_migrator`, superuser, role-creator,
database-creator, or RLS-bypass runtime principals fail readiness.

## Verify

Run unit tests without PostgreSQL:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Run the live migration and security checks against a migrated disposable
database:

```bash
EVA_TEST_DATABASE_DSN='postgresql://...' \
  .venv/bin/python -m unittest \
  tests.test_database_foundation \
  tests.test_database_live_security
```

The live checks verify runtime role restrictions, tenant RLS isolation, schema
readiness, user lifecycle, durable approvals, queue grants, and append-only
audit enforcement.

## Security boundary

- Runtime roles are `NOLOGIN`, non-superuser, and cannot bypass RLS.
- The deployment layer creates login principals and grants only the required
  runtime role.
- Runtime readiness fails in strict mode if the application connects as a
  privileged role.
- Every transaction must receive server-derived tenant and actor context.
- Agent tool writes and external actions must fail before handler execution if
  the durable audit sink is unavailable.
- Audit rows cannot be updated or deleted through the application schema.

## Rollback

For an empty or disposable environment:

```bash
EVA_DATABASE_DSN='postgresql://...' .venv/bin/alembic downgrade base
```

The downgrade removes the EVA schemas but intentionally retains cluster roles,
because deployment-managed login principals or backup tooling may reference
them. Customer data rollback requires a backup and restore procedure; it must
not rely on destructive downgrade.

## Remaining pilot work

- Session inventory UI/API and targeted single-session revocation.
- SQLite/JSON import and comparison report.
- Backup/restore automation and a tested restore.
- Complete route ownership review and outcome audit after handler execution.
