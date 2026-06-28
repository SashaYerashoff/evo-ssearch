# Changelog

Notable changes per release. Detail lives in `readiness/RELEASE_NOTES_<version>.md`.
Authoritative current state: [docs/00_CANON/facts.md](docs/00_CANON/facts.md).
Format loosely follows Keep a Changelog.

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
