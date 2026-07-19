# Changelog

Notable changes per release. Detail lives in `readiness/RELEASE_NOTES_<version>.md`.
Authoritative current state: [docs/00_CANON/facts.md](docs/00_CANON/facts.md).
Format loosely follows Keep a Changelog.

## β 0.8.4 — unreleased (attention decider, media broker, stabilization)

Scaffold for release notes; final prose to be assembled from commits
`728081c..HEAD` before tagging.

- **Capture apex decider v2:** per-second quiet/normal/burst classification
  against a persisted per-channel motion baseline (homeostasis); sharpness-aware
  frame selection; burst companion frames to archive and (one per batch) to the
  VLM; `capture_attention` in `VECTOR_SIGNALS_JSON`; measured-homeostasis line in
  channel memory prompts; `capture_selector_bias` channel setting
  (auto/action/clarity).
- **Operator media:** tokenized same-origin live/archive broker with lease
  renewal and stall watchdogs; shared EVA attention preview (`Model view`) with
  60 s freshness window and auto-recovery; archive review modal is
  evidence-first with opt-in playback; archive segments support bounded
  `duration_sec`. Optional local V4L2/USB sources can be added as independent
  live channels, using the bundled FFmpeg runtime directly without Evo; their
  lack of recorder archive and bookmark delivery is reported explicitly.
- **Agent:** activated runbooks (skills) now force-expose the tools they name
  through the intent gate — previously RU phrasings of a runbook's own trigger
  phrases could inject the runbook while exposing zero tools; runbook SKILL.md
  steps now name their tools explicitly (`multi_channel_event_sweep`,
  `cross_channel_correlation`, `video_incident_timeline`). Persisted research
  continuation ledger, composite channel
  inventory, shared LM admission queue, context/token budget observability
  (`estimated_context_tokens`); an approval-gated post-upgrade command can now
  audit and durably restore missing L2/L3 semantic history from archived L0
  text, with restart-safe progress, source-gap accounting, live-load priority,
  and ETA reporting.
- **UI:** vertical workspace layout rework, burst attention badges,
  review-modal filmstrip roles.
- **Stability:** capture thumbnail/frozen-signal contract restoration,
  duck-typed live media open, attention-stream keepalive, strict archive
  source filters, explicit-intent agent research continuation; L1–L3 operator
  narratives are separated from machine homeostasis, legacy concatenation is
  rejected, queued semantic aggregation is reported without alarm styling, and
  explicit per-window retry remains available. The background scheduler now
  distinguishes ordinary inference from a saturated L0 backlog, has a bounded
  deferral ceiling, and backfills newest missing closed windows after downtime.
  Genuine 0.8.0/0.8.1 LM rollups are adopted as labelled legacy semantics and
  promoted to durable rows without model regeneration; deterministic fallback
  cards remain excluded from semantic history. Scheduled L1–L3 text rollups
  use the dedicated agent profile with model thinking disabled. Interactive
  agent tool decisions and final answers also request direct/non-thinking
  output so Qwen3.5 cannot spend the completion budget before finishing the
  operator-visible tool loop.
  Rollup backpressure is LM-resource-aware, and scheduler spacing is measured
  start-to-start so a 50-channel cadence is not defeated by mandatory idle time.
- **Install:** offline dry-run-first installer orchestrator; version gate
  reads the bundled VERSION file; adopt upgrade rehearsed against a field
  tree copy (code-only, schema stays at head, no wheelhouse needed); guided
  field upgrade script (`scripts/field_upgrade_084.sh`) with a read-only
  schema gate, dry-run-then-confirm flow, and recorded rollback command
  (`readiness/UPGRADE_084_FIELD_CHECKLIST_RU.md`).
- **Compatible adopt upgrades:** deployed versions are no longer allowlisted;
  any non-empty post-schema build can proceed when exact requirements, healthy
  venv, and read-only schema-head gates pass. Exact bundle reruns remain blocked
  by commit marker. Code/rollback snapshots now exclude nested private/runtime
  trees, legacy absolute venv links are safely ignored during restore, and an
  automatically restored service reports external readiness degradation without
  another four-minute wait. System-mode media preflight cleans privileged
  OpenCV staging safely; the agent-context probe uses profile authentication
  and requires an explicit decision when the served context is unknown.
  Adopt-upgrade config discovery now follows the active systemd
  `EnvironmentFiles` contract and cross-checks selected agent/Luxriot endpoints
  against `/ready`, instead of preferring a possibly stale application `.env`.
  A verified code-only adopt preserves an already-running configuration and
  reports placeholder-like existing values as warnings; fresh installs and
  migrations keep the strict placeholder failure policy.
- Schema head: `20260614_0006` (unchanged from 0.8.3 unless noted before tag).
  See `readiness/RELEASE_NOTES_0.8.4.md`.

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
