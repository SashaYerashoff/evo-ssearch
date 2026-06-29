# Luxriot EVA AI β 0.8.2.1 Internal Release Notes

Release date: 2026-06-30  
Release type: office-demo UI/evidence patch  
Previous baseline: `β 0.8.2`  
Schema head: `20260614_0006`  
Database migration: **none**

`β 0.8.2.1` is a small stabilization patch on top of the office-demo build. It
does not change schema or deployment topology. The goal is to remove operator
confusion in the chat approval flow and clean up evidence/UI behavior observed
during manual testing in the Luxriot EVO Monitor embedded browser.

## What Changed Since β 0.8.2

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
  tests/test_agent_video_summary_tools.py
node --check static/js/app.js
```

Expected:

- docs drift check passes;
- targeted tests pass;
- JavaScript syntax check passes;
- `/health.version` reports `β 0.8.2.1` after restart/env override update.

