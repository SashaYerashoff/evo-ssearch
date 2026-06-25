# Backup & Recovery

The asset is **PostgreSQL**. The collected video-descriptions, alerts, archive
frames, IAM, audit, and runtime state all live there. Back it up; everything else
is re-deployable. Invariants: [facts](../00_CANON/facts.md).

## What to back up

| Item | Why |
|---|---|
| **PostgreSQL database** (all EVA schemas: iam, agent, audit, archive) | The data + runtime state. Primary backup. |
| `.env` (on-host, `0600`) | Secrets + config. Store securely, separately. |
| `inference_spool/` (if queue enabled) | In-flight summary batches. |
| TLS material / service unit files | Faster rebuild. |

Code is in git / the patch bundle — not part of data backup.

## Critical warning

`archive.runtime_state` holds **prompt settings, desired live sessions, summary
history, and rollup cache** — not just caches. **Never `TRUNCATE
archive.runtime_state`** as a cleanup step; you will lose prompts and which
channels should be running. Targeted cleanup only.

## Backup procedure

1. Prefer an online `pg_dump` of the EVA database (RLS-aware: run as a role that
   can read all tenant rows, or dump per schema as required by your role setup
   — `[VERIFY]` against the deployed role layout).
2. Schedule daily (at minimum) during the pilot; keep several days of dumps.
3. Verify each dump is restorable on a scratch instance periodically.
4. Back up `.env` to a secure secrets store, separate from DB dumps.

```bash
# illustrative — confirm role/DSN against the deployment
set -a; . /etc/eva-ai/eva-ai.env; set +a
pg_dump "$EVA_DATABASE_DSN" -Fc -f /backup/eva_$(date +%F).dump
```

## Restore procedure

1. Stop the service (`systemctl stop eva-ai` / `eva-ai-local-5443`) so capture
   loops aren't writing.
2. Restore the database dump into a clean DB with the same schemas/roles.
3. Confirm Alembic head matches code: `alembic current` == `20260614_0006`
   (see facts). Run `alembic upgrade head` only if behind.
4. Restore `.env`.
5. Start the service; check `GET /ready` (all components) and that desired live
   sessions resume.
6. Spot-check: a recent video-description query returns data; archive search works.

## Disaster scenarios

- **App host lost:** rebuild host, redeploy code (patch bundle), restore DB + .env,
  start, verify `/ready`.
- **DB corruption:** restore latest good dump; accept loss since last dump
  (summary history since last hard write may differ by up to the persist-debounce
  interval on an unclean stop).
- **Accidental data wipe:** restore DB; never recover from `TRUNCATE
  archive.runtime_state` without a dump — hence daily dumps.

## Retention vs backup

Retention (pruning old frames) is **not** backup. Set retention to the operational
window ([config_reference](../00_CANON/config_reference.md)); keep DB dumps for
recovery independently.
