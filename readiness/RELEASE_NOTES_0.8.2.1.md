# Luxriot EVA AI β 0.8.2.1 Internal Release Notes

Release date: 2026-06-30  
Amended: 2026-07-01  
Release type: office-demo UI/evidence and agent-workflow patch
Previous baseline: `β 0.8.2`  
Schema head: `20260614_0006`  
Database migration: **none**

`β 0.8.2.1` is a small stabilization patch on top of the office-demo build. It
does not change schema or deployment topology. The goal is to remove operator
confusion in the chat approval flow, clean up evidence/UI behavior observed
during manual testing in the Luxriot EVO Monitor embedded browser, and harden
agent investigation workflows that were prone to false "Luxriot not connected"
or latest-slice answers on ad-hoc onboarding installs.

## What Changed Since β 0.8.2

### Agent Runtime and Channel Inventory

- Agent startup context now reads channel inventory through
  `luxriot_manager.get_channels(force=False)`, the same production contract used
  by `/luxriot/channels` and the UI.
- The agent no longer writes `Luxriot not connected` from a missing legacy
  `.channels` attribute. If startup context cannot list channels, it tells the
  model to verify with `list_channels`.
- Channel-reference resolution (`channel_ref`, channel title/name lookup) also
  uses `get_channels()`, so office/local installs that use the real
  `LuxriotManager` behave consistently with test/dev managers.
- `list_channels {"now": true}` is accepted and normalized to `force=true`
  before security-gateway dispatch. Unknown arguments remain denied generally;
  this is a narrow compatibility alias for common "refresh now" wording.

### Agent Period Investigation Workflow

- Broad video-summary and incident-review instructions now explicitly forbid
  answering from only the newest summary entry or newest archive hits.
- The intended flow is now spelled out in the system prompt and video-summary
  skills: establish coverage/alert/probe health, inspect L2/L1 across the
  requested window, choose 2-3 candidate windows spanning the period or carrying
  alerts/deviations, then drill into L0/frames for evidence.
- If only the latest slice is actually available because history retention or
  coverage is short, the agent must report that as a coverage limitation rather
  than implying the full requested period was reviewed.

### Last-Week Reports And Archive Fallback

- `normalize_time_window` now treats `last week`, `last 7 days`, and equivalent
  relative wording as a rolling 7-day period. The agent prompt and tool schema
  explicitly instruct the model to use `relative_range` for these phrases rather
  than inventing a calendar date.
- `generate_report` now accepts `since_ms` / `until_ms` aliases and normalizes
  them to `from_ts` / `to_ts` before secure tool dispatch. This removes the
  `unknown tool arguments: since_ms` failure seen in manual chat traces.
- `list_video_summary_channels` no longer aborts the whole report when the live
  Luxriot channel inventory is temporarily unreachable. If local video-summary
  history/runtime digest exists, the tool returns `channel_inventory_status:
  archive_fallback` and builds candidate channels from local history.
- Video-description reports surface that fallback explicitly. The agent should
  say "live inventory unavailable, report uses local video-summary/runtime
  history" instead of claiming that no channels or no incidents exist.

### Period-Wide Evidence Sampling

- Archive-window reads with `sort_by="oldest"` now reach the true start of the
  requested window instead of sorting only the newest backend page.
- Video-description reports now sample VLM alert evidence frames across the
  requested period instead of returning only newest alert frames.
- Probe reports now include `representative_events` sampled across the requested
  period. Compact model-facing report output keeps these representative events
  so the agent can explain probe evidence from more than `latest_ts`.
- Existing `get_video_summaries` period sampling remains in place: L1/L2 entries
  are selected across the period with alert/deviation priority.

### Agent Chat Approval Flow

- Probe mutation previews and receipts (`create_probe`, `update_probe`,
  `delete_probes`) are rendered as standalone approval cards outside the
  collapsible Research trace.
- Empty Research trace blocks are hidden.
- Legacy probe preview cards that still appear inside the trace are promoted out
  of it and restyled as approval cards.
- The Apply button still executes the existing server-side action plan endpoint;
  there is no direct chat-side apply bypass.

### Video Summary Machine Blocks

- L0 machine-readable blocks are labeled by provenance:
  - `System message` for short non-alert JSON / empty-alert blocks.
  - first alert title plus severity/line metadata for alert JSON.
  - `Memory/homeostasis` for memory / baseline / homeostasis payloads.
- The raw JSON remains available to admin/engineer roles.

### Monitor UI

- Removed the selected-probe "Latest CLIP Hits" filmstrip from the monitor
  inspector. Recent hit data remains available internally and in the probe editor
  flow, but the unstable visual strip is no longer rendered in the operator
  monitor layout.
- Kept the CEF-safe repeated-card layout contracts for probe cards and archive
  result grids.

### Evidence Thumbnail Handling

- Detection rows now expose `has_thumbnail` and only receive `image_url` when an
  image path or thumbnail is known to exist.
- Metadata-only detections no longer masquerade as visual evidence.
- Agent thumbnail grids fall back to a non-clickable `No image #id` tile if an
  image request fails.

### Database Schema Audit

- Added `readiness/db_schema_audit_0_8_2_1_ru.md` as a current Russian audit of
  the PostgreSQL schema, tenant/RLS layout, agent/audit/archive tables, and
  screenshot/image storage surfaces.
- The audit is descriptive only: it documents the current implementation and
  intentionally does not propose integration changes.

## Upgrade Notes

- Code-only patch from `β 0.8.2`.
- No Alembic migration.
- Restart the service after update.
- Because `app.js`, `app.css`, and `templates/index.html` changed, testers must
  hard-refresh the browser or reload the Luxriot EVO Monitor web tile.
- If `/etc/eva-ai/eva-ai.env` sets `EVOSSEARCH_APP_VERSION`, update it to:

```env
EVOSSEARCH_APP_VERSION="β 0.8.2.1"
```

## Minimum Verification

Run on the target machine after updating:

```bash
bash scripts/check_docs_drift.sh
.venv/bin/python -m pytest -q \
  tests/test_ui_css_contract.py \
  tests/test_api_dataflow_smoke.py \
  tests/test_agent_video_summary_tools.py \
  tests/test_eva_agent_adapter.py \
  tests/test_agent_tool_loop.py
node --check static/js/app.js
```

Expected:

- docs drift check passes;
- targeted tests pass;
- JavaScript syntax check passes;
- `/health.version` reports `β 0.8.2.1` after restart/env override update.

Full local validation after the July 1 agent-report patch:

- `360 passed, 19 skipped, 127 subtests passed`
- docs drift check passed
- `py_compile` passed for touched agent/security/test modules

## Manual Regression Focus

- Ask the agent to introduce or list available channels on a real
  `LuxriotManager` install. It should not claim Luxriot is disconnected unless a
  tool call in that turn confirms a connection failure.
- Ask for a channel's last-day video-description report. The answer should state
  coverage, inspect L2/L1 over the period, and avoid basing conclusions only on
  the latest summary entry.
- Ask for a probe report over a long period. The answer should use
  representative probe events across the period when discussing evidence, with
  `latest_ts` treated as metadata only.
- Ask for a video-summary report for active channels for the last week. The
  period should be rolling 7 days, `generate_report` must not reject
  `since_ms/until_ms`, and live inventory failures should be reported as
  `archive_fallback` when local summaries exist.

## Manual Test Documents

- `readiness/MANUAL_TEST_SCENARIO_0.8.2.1_AGENT_REPORTS_RU.md` - agent/report
  acceptance scenario for VLM alerts, last-week reports, coverage gaps,
  archive fallback, probe calibration, and prompt tuning discipline.
- `readiness/MANUAL_TEST_PROTOCOL_0.8.2.1_UI_PATCH_RU.md` - UI regression
  protocol for probe approval cards, machine JSON labels, monitor probe cards,
  missing-image handling, and archive review modal behavior.
