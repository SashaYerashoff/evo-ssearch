# Changelog

Notable changes per release. Detail lives in `readiness/RELEASE_NOTES_<version>.md`.
Authoritative current state: [docs/00_CANON/facts.md](docs/00_CANON/facts.md).
Format loosely follows Keep a Changelog.

## β 0.8.3 — 2026-07-02 (road-event and live-signal stabilization)

- **Road-event foundation:** added lightweight road-motion CV primitives,
  scene-card bootstrap, motion-zone/flow cues, and Luxriot snapshot/live-video
  smoke tooling for drift, burnout, wrong-way, and aggressive-traffic candidate
  workflows.
- **Vector signals:** compact CLIP/probe and road-CV cues can now condition L0
  video descriptions as secondary attention signals; they remain candidate
  signals and must be verified against current frames.
- **Live preview honesty:** stale and exact-frozen Luxriot buffers now surface as
  `Signal lost` / `Signal frozen` instead of replaying old frames in the UI or
  feeding VLM/probe loops.
- **Agent runtime status:** video-summary inventory now carries explicit
  `runtime_problem_channels` so stale/frozen/error capture issues are visible to
  the agent even when no summaries exist for that channel.
- **Road grounding UI:** engineer/admin monitoring can render a fresh-frame road
  mask overlay for scene grounding and diagnostics.
- **Acceptance:** added a cumulative β0.8.0→β0.8.3 manual acceptance scenario
  focused on alerts, reports, probe control, signal loss, and road-event
  candidate handling.
- **Upgrade:** code-only from β 0.8.2.1 — **no database migration**.
- Schema head: `20260614_0006`. See `readiness/RELEASE_NOTES_0.8.3.md`.

## β 0.8.2.1 — 2026-06-30, amended 2026-07-01 (UI evidence, approval, and agent workflow polish)

- **Agent chat UI:** probe create/update/delete previews and receipts render as
  standalone approval cards outside the collapsible research trace; legacy probe
  preview cards are promoted out of the trace if encountered.
- **Agent channel inventory:** startup context and channel-reference resolution
  now use the production `get_channels()` contract instead of the legacy/test
  `.channels` attribute, avoiding false "Luxriot not connected" answers.
- **Agent period investigations:** broad video-summary/report workflows are
  explicitly guarded against latest-slice answers; reports now use period-wide
  representative evidence instead of newest-only archive hits.
- **Probe reports:** compact agent results include representative probe events
  across the requested period, not only aggregate counts and `latest_ts`.
- **Tool-call resilience:** `list_channels {"now": true}` is accepted as a safe
  alias for `force=true` and normalized before gateway dispatch.
- **Video-summary UI:** machine JSON blocks are labeled by provenance
  (`System message`, alert title, or `Memory/homeostasis`) instead of a generic
  machine label.
- **Monitor UI:** removed the unstable selected-probe filmstrip from the monitor
  inspector; probe cards and repeated grids keep the CEF-safe layout contracts.
- **Evidence UI:** metadata-only detections no longer masquerade as visual
  evidence; missing thumbnails render as non-clickable `No image` tiles.
- **Upgrade:** code-only from β 0.8.2 — **no database migration**.
- Schema head: `20260614_0006`. See `readiness/RELEASE_NOTES_0.8.2.1.md`.

## β 0.8.2 — 2026-06-28 (office-demo hardening)

- **VLM alerts:** split live-feed role text from channel alert criteria; structured
  alert parsing/delivery is now observable; backend state-transition events are
  exposed to the agent with provenance so prose-only findings are treated as
  unconfirmed until frame evidence is checked.
- **Agent reports:** status/report flow is video-description-first, with pipeline
  health separated from incident findings and live runtime status routed to
  runtime tools instead of documentation lookup.
- **Probe control:** archive-based CLIP P/N/M calibration returns deterministic
  `safe_to_apply`, `recommended_action`, warnings, and pass-through preview args;
  unsafe/over-firing calibrations do not produce apply-ready probe changes.
- **Acceptance:** added seeded live-smoke fixtures plus admin and non-admin
  live-smoke paths for preview/apply, documentation RBAC, archive search,
  calibration, and broad-channel chunking.
- **Upgrade:** code-only from β 0.8.1 — **no database migration**.
- Schema head: `20260614_0006`. See `readiness/RELEASE_NOTES_0.8.2.md`.

## β 0.8.1 — 2026-06 (production-pilot stabilization)

- **Agent:** video-description-first status/reports; probes secondary;
  `generate_report` defaults to video-descriptions (`report_type=probes` for probe
  reports); new `track_visual_state_transitions` tool; read-only rollup reads (no
  LLM synthesis on investigation); coverage/truncation contracts; CLIP-safe
  negative-state handling; automatic evidence thumbnails.
- **Runtime:** persistence hot-path cost reduced (no full per-channel history
  re-normalization per batch; alert metadata preserved on duplicate merge);
  Gunicorn worker shutdown hooks flush summary/rollup state on graceful restart;
  bookmark delivery observability (`bookmark_failed_count` / `bookmark_last_error`).
- **Docs:** removed demo-specific examples from agent-visible contracts.
- **Upgrade:** code-only from β 0.8.0 — **no database migration**.
- Schema head: `20260614_0006`. See `readiness/RELEASE_NOTES_0.8.1.md`.

## β 0.8.0 — 2026-06-14 (production-pilot beta)

- Named users + RBAC + audit; PostgreSQL control plane; RLS; channel grants.
- PostgreSQL frame archive (probe / vlm_summary / vlm_alert) with search + retention.
- L0–L3 video-summary rollups; VLM profile routing; live-summary restore.
- Secured agent tool gateway; durable approval flow; `/health`, `/ready`.
- Schema head: `20260614_0006`. See `readiness/RELEASE_NOTES_0.8.0.md`.

---

Earlier α history and point-in-time engineering snapshots: `readiness/history/`.
