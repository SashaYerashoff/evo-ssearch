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

Use **Settings → Users** in the UI for normal administration, or
`scripts/manage_users.py` for CLI maintenance flows `[VERIFY exact subcommands]`:

- **Create** a user (role + initial password).
- **Reset password**, **enable/disable**, **revoke sessions**.
- **Channel grants** — assign specific channel IDs, or the all-channel grant for
  non-admin users who legitimately need every channel.
- First admin is created with `scripts/bootstrap_admin.py` `[VERIFY]`.

Login throttling and a session inventory are built in; disable shared-token
workflows (legacy admin token is not the current auth model).

If an operator asks the agent how to reset another user's password or assign
channel grants, the agent should redirect them to an admin/engineer rather than
reciting procedure steps. Admin/engineer users may receive sanitized procedure
help through the documentation lookup.

## Channel grants

- Grant the minimum channels a user needs (data minimization + privacy).
- Operators/viewers see only granted channels in every surface (Video, Archive,
  Agent). If a channel is "missing" for a user, it's a grant issue.

## Audit

- Sensitive endpoints and **agent tool calls** are audited.
- Read the audit log via **Settings → Audit** / protected diagnostics reader
  `[VERIFY exact UI path]` — who did what, when, including agent actions.

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

- Stream/L0 prompt controls description style; rollup prompts (L1/L2/L3) control
  summary aggregation.
- **Alert Criteria** (`alert_policy_prompt`) is the plain-language per-channel or
  default watch policy. Put "watch for / alert on" conditions here, not in the
  L0 stream prompt.
- The structured `ALERTS_JSON` / `json_alert_prompt` template is only the
  machine-readable alert-output contract, not the place for operator criteria.
- These fields are editable per-channel or as defaults (Settings, or agent
  `update_prompt_settings` — preview-only in secure mode). If `prompt_health`
  reports legacy alert/watch text in the stream prompt, preview
  `migrate_legacy_alert_policy` before other prompt edits.
- In secure mode, a chat confirmation is not an apply operation. The operator
  commits prompt/probe previews with the UI Apply button; the server records a
  trusted action receipt for the next agent turn.
- `bookmark_enabled` gates pushing alerts to Luxriot; alert **extraction** (badges,
  counts) is always on regardless.
- Watch the bookmark delivery metrics (observability) to confirm alerts reach
  Luxriot.
- **Start summaries** persists the desired live video-description session for a
  channel; restored sessions should come back after service restart. If a channel
  is quiet after restart, check stream health and desired-session restore status
  before assuming "no activity".

## Probes (engineer/agent-managed)

- Probes are primarily an agent/engineer tool now. Engineers can create/tune via
  the probe editor (preview the diff before applying). A curated watch-list probe
  set may be cast across demo channels as a parallel detector.
- Probe calibration from archive is read-only and audited. It proposes thresholds
  from archived CLIP P/N/M evidence, then requires the normal preview/apply flow
  before any probe setting changes.
- Probe negatives must be visible contrast/background states, never literal
  absence phrases such as "no weapon". Refusal to create an unsafe preview is a
  correct outcome.

## Secrets rotation

- All secrets live in the on-host `.env` (mode `0600`). Rotate by updating `.env`
  and restarting the service. Never commit secrets or place them in shareable docs.
- For deployments behind HTTPS/TLS, set `EVOSSEARCH_AUTH_COOKIE_SECURE=true`.
  Office demo systems may run plain HTTP internally, but client-facing systems
  should not keep insecure-cookie settings. See
  [production_settings](../install/production_settings.md).

## Routine checks

- After any restart/patch: `GET /ready` all-green; desired sessions resumed.
- Daily during pilot: coverage gaps, archive growth vs. cap, bookmark failures
  (see [observability](observability.md)); a DB backup exists (see
  [backup_recovery](backup_recovery.md)).
