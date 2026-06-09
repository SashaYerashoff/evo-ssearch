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

## Migrate

```bash
EVA_DATABASE_DSN='postgresql://...' .venv/bin/alembic upgrade head
```

Expected revision:

```text
20260609_0001
```

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

- Durable PostgreSQL repositories for IAM and agent plans/approvals.
- Login/logout/me routes, secure cookies, CSRF, and endpoint authorization.
- PostgreSQL inference queue adapter and worker integration.
- SQLite/JSON import and comparison report.
- Backup/restore automation and a tested restore.
- Route and tool audit instrumentation with request IDs.
