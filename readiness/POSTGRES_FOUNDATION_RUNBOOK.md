# EVA PostgreSQL foundation runbook

Date: 2026-06-09

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

Set a PostgreSQL DSN through `EVA_DATABASE_DSN` or
`EVOSSEARCH_DATABASE_DSN`. Do not place the DSN in source control or expose it
through the settings API.

Configure a separate login principal and DSN for the append-only writer:

```bash
export EVA_AUDIT_DATABASE_DSN='postgresql://eva_audit_login:...@db/eva'
```

Grant the application login `eva_api` and the audit login
`eva_audit_writer`. Do not grant both roles to the same principal.

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
20260609_0002
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

The application `/ready` response includes a required `postgresql` component
when a DSN is configured. It distinguishes an unavailable database from a
schema revision mismatch.

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
readiness, and append-only audit enforcement.

## Security boundary

- Runtime roles are `NOLOGIN`, non-superuser, and cannot bypass RLS.
- The deployment layer creates login principals and grants only the required
  runtime role.
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

- Durable PostgreSQL repositories for agent plans/approvals.
- User administration, session revocation UI, and distributed login throttling.
- Capture producer and inference worker integration with the PostgreSQL queue.
- SQLite/JSON import and comparison report.
- Backup/restore automation and a tested restore.
- Complete route ownership review and outcome audit after handler execution.
