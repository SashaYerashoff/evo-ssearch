# Changelog

Notable changes per release. Detail lives in `readiness/RELEASE_NOTES_<version>.md`.
Authoritative current state: [docs/00_CANON/facts.md](docs/00_CANON/facts.md).
Format loosely follows Keep a Changelog.

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
