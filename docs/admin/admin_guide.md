# Admin Guide

Day-to-day administration: users, roles, channel access, audit, retention, and
prompt/alert settings. `[VERIFY]` marks exact CLI/UI flows to confirm against the
deployed build before relying on them. Invariants:
[facts](../00_CANON/facts.md). Variables:
[config_reference](../00_CANON/config_reference.md).

## Roles & permissions

| Role | Can |
|---|---|
| **admin** | Manage users/roles/grants, settings, read audit, retention |
| **engineer** | Configuration, prompt/probe tuning |
| **operator** | Monitor, use the agent, scoped channels |
| **viewer** | Read-only, scoped channels |

Access is enforced at the API/tool gateway **and** by PostgreSQL row-level
security; a user never exceeds their channel grants regardless of UI or agent.

## User lifecycle

Via the Admin UI and/or `scripts/manage_users.py` `[VERIFY exact subcommands]`:

- **Create** a user (role + initial password).
- **Reset password**, **enable/disable**, **revoke sessions**.
- **Channel grants** — assign specific channel IDs, or the all-channel grant for
  non-admin users who legitimately need every channel.
- First admin is created with `scripts/bootstrap_admin.py` `[VERIFY]`.

Login throttling and a session inventory are built in; disable shared-token
workflows (legacy admin token is not the current auth model).

## Channel grants

- Grant the minimum channels a user needs (data minimization + privacy).
- Operators/viewers see only granted channels in every surface (Video, Archive,
  Agent). If a channel is "missing" for a user, it's a grant issue.

## Audit

- Sensitive endpoints and **agent tool calls** are audited.
- Read the audit log via the protected admin/diagnostics reader `[VERIFY UI path]`
  — who did what, when, including agent actions.

## Retention administration

Set retention to the operational window (see config_reference):

- Frame archive rows: `EVOSSEARCH_ARCHIVE_ROW_RETENTION_DAYS`
- Thumbnails: `EVOSSEARCH_ARCHIVE_THUMBNAIL_RETENTION_DAYS`
- Row cap: `EVOSSEARCH_ARCHIVE_MAX_RECORDS` (raise for multi-week windows)
- Summary history: `EVOSSEARCH_LUXRIOT_SUMMARY_RETENTION_DAYS`

Retention runs on a schedule. **Never `TRUNCATE archive.runtime_state`** — it holds
prompt settings, desired live sessions, and summary state (see
[backup_recovery](backup_recovery.md)).

## Video-description & alert settings

- Stream/L0 prompt, rollup prompts (L1/L2/L3), and the structured `ALERTS_JSON`
  template are editable per-channel or as defaults (Settings, or agent
  `update_prompt_settings` — preview-only in secure mode).
- `bookmark_enabled` gates pushing alerts to Luxriot; alert **extraction** (badges,
  counts) is always on regardless.
- Watch the bookmark delivery metrics (observability) to confirm alerts reach
  Luxriot.

## Probes (engineer/agent-managed)

- Probes are primarily an agent/engineer tool now. Engineers can create/tune via
  the probe editor (preview the diff before applying). A curated watch-list probe
  set may be cast across demo channels as a parallel detector.

## Secrets rotation

- All secrets live in the on-host `.env` (mode `0600`). Rotate by updating `.env`
  and restarting the service. Never commit secrets or place them in shareable docs.

## Routine checks

- After any restart/patch: `GET /ready` all-green; desired sessions resumed.
- Daily during pilot: coverage gaps, archive growth vs. cap, bookmark failures
  (see [observability](observability.md)); a DB backup exists (see
  [backup_recovery](backup_recovery.md)).
