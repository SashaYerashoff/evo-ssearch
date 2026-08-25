# Changelog

Notable changes per release. Detail lives in `readiness/RELEASE_NOTES_<version>.md`.
Authoritative current state: [docs/00_CANON/facts.md](docs/00_CANON/facts.md).
Format loosely follows Keep a Changelog.

## Unreleased

- **No RLS-blind appliance updates or backups:** fresh-install migrator and
  full-site backup logins now receive explicit `BYPASSRLS`, while API, worker,
  and audit logins remain tenant-filtered. Existing local appliances update
  through a random process-only migration identity created after approval via
  PostgreSQL peer auth, limited to a two-hour password lifetime, and removed
  before handoff. This repairs older installer-created `eva_migrator_login`
  roles without persisting a superuser DSN or weakening tenant runtime roles.

- **Incident Review is explicitly feature-in-progress:** its Video workspace tab
  now carries a visible FiP badge and operator warning. Settings → Features has
  a workstation-local `Show incidents (FiP)` switch which immediately unmounts
  and hides the review surface when its output is not operationally useful,
  without deleting incident history or changing backend processing.

- **Legacy incident noise no longer consumes operator review:** automatic
  pre-admission `vlm_l0_temporal` candidates which were never grounded,
  followed, or acted on remain in the full incident ledger and detail API but
  are excluded in PostgreSQL before Incident Review count and pagination.
  Manual drafts, operator lifecycle history, Follow cases, and every saved
  operator/safety signal remain visible.

- **Grounded cross-window incident hierarchy:** L3 may now attach bounded
  context across adjacent L2 windows only when an existing top-level safety or
  high-priority operator incident anchors the scene. Existing L2 children are
  reused by server-owned episode identity, context-only children stay out of
  review pagination, and observed coverage gaps no longer stretch an episode
  to the timestamp of a much later observation.

- **Site-specific CSRF cookies work in React:** the authentication response now
  publishes the configured non-secret CSRF cookie name, and the shared React
  transport applies it to every mutation and agent stream. Adopted sites no
  longer lose Incident Report and other writes when their cookie prefix differs
  from the default `eva_csrf`.

- **Bundled media is now part of the Python installer transaction:** when an
  offline bundle contains the self-contained runtime, its FFmpeg/OpenCV files
  and checksums are validated during read-only preflight and installed after
  the rollback snapshot but before Alembic or the first service start. Legacy
  venvs without OpenCV use the isolated `.eva-runtime/python` overlay instead
  of being modified in place.

- **Legacy 0006 upgrade safety:** Alembic now accepts both PostgreSQL URLs and
  libpq conninfo without leaking or mangling credentials, while preflight also
  proves schema ownership, `eva_owner` role switching, and transactional DDL
  before touching the live tree. Rollback records the original revision and
  restores the complete owner-preserving dump only when the database actually
  advanced.
- **No silent evidence pruning:** an existing installation without an explicit
  archive-retention policy is upgraded with pruning disabled and a warning,
  instead of immediately applying the newer 14-day thumbnail default. Fresh
  installations receive explicit 90-day row and 14-day thumbnail windows.
- **Complete accepted UI payload:** patch snapshots now carry and explicitly
  install `react-ui/dist`; the updater refuses a bundle without it, so an
  otherwise successful upgrade cannot fall back to the legacy console.
- **Durable workflow boundary:** an unfinished Protocol Deploy draft no longer
  captures a new archive/video request merely because it mentions a channel,
  webcam, alert, or time window.
- **Office installer follow-up:** the Ubuntu 24.04 offline package input now
  includes `python3-dev`, so native Python dependencies have the matching
  `Python.h` headers on a clean appliance. After validating the generated TLS
  site with `nginx -t`, the installer explicitly restarts Nginx instead of
  relying on `enable --now`, which does not reload an already-running
  pre-install process.
- The checksummed 0.8.7 USB payload must be rebuilt and finalized to carry this
  dependency change; released bundles remain immutable and reject in-place
  edits by design.

## β 0.8.7 — 2026-08-06 release candidate (temporal incidents and field hardening)

- **Temporal incident memory:** EVA now keeps server-owned observation,
  episode, incident and recurrence-series identities across L0–L3. Covered
  high-signal L0 events can open bounded review candidates; exact semantic
  tracks continue without merging parallel incidents, explicit routine
  boundaries end only the perceptual episode, and coverage gaps never prove
  resolution. The 2/4/8 allocator bounds prompt cognition while overflow stays
  durable and retrievable.
- **Independent lifecycle and review:** perception, risk, case and attention
  states have an append-only transition ledger with optimistic revisions.
  Operators can confirm, resolve, dismiss, mark false-positive or reopen a
  case, and confirm/reject candidate recurrence links without merging incident
  IDs. The secured agent exposes the same mutations through preview/apply
  approval with server-resolved channel, revision and relation bindings.
- **Incident operations:** Follow produces a durable human-readable outcome;
  Incident Review separates active, needs-review and historical queues; detail
  shows evidence covers, temporal episodes, candidate series and lifecycle
  history. Markdown/XML reports lead with an operator synopsis and compressed
  homeostatic response instead of dumping every low-level event.

- **Reliable Protocol Deploy on 4B heads:** active deployment drafts are now
  rehydrated from compact trusted tool receipts across chat turns. Operator
  channel/group selections deterministically advance through scope
  configuration and the bounded scene survey instead of being misrouted into
  generic L1/L2 research. The surveyed phase now injects a server-owned receipt
  with bounded scene fingerprints, pins the durable deployment ID on every
  subsequent tool call, and prevents a repeated scope payload from erasing the
  survey while the agent asks for policy requirements. Authorization-only
  channel scope no longer leaks into the workflow payload and clears saved
  groups during requirements collection. Requirements and preview phases expose
  only their single valid tool, so a small head cannot merely narrate an update
  without persisting it or skip the operator-review preview. Duplicate overlapping
  policy packs (including a hallucinated `quiet_window` pack) are discarded with
  a draft warning, while mixed count-and-duration intent is preserved. Final
  receipts distinguish a generated preview from an applied plan and prohibit
  coverage claims from the sparse survey. Partial requirements are durable and
  preview is blocked until every selected channel has a policy pack; the final
  partial and preview receipts are rendered from server truth instead of model
  prose, and general deployments discard model-invented maritime roles/starter
  policy fields. Operator corrections at `plan_ready` merge alerts by
  channel-pack and alert name, preserving sibling rules before regenerating the
  preview. Explicit quoted Rule/Alert names form a server-side allowlist, so
  model-invented sibling policies cannot enter the durable draft. Appliance Doctor also
  fails readiness when an installed vLLM unit lacks native tool-choice/parser
  flags. The React console now exposes the workflow instead of hiding it in a
  research trace: a searchable inventory picker handles up to 100 visible Evo
  channels while enforcing the eight-channel attention budget, a scene-survey
  card offers per-group alert drafting or an explicit no-alert choice, and the
  final card reviews VLM policies and P/N probes/counters scope by scope before
  revealing the sole atomic Apply action. Explicit probe rejection rebuilds the
  draft without discarding its VLM alert policy. The full authorized channel-ID
  inventory now survives chat-history reloads even when only the first 16 titles
  are exposed to the 4B head, and the maritime picker requires an explicit
  closed role for every selected channel before survey; the agent cannot invent
  a port/coast/PTZ role to advance the draft. Partial configuration returns a
  fresh review card containing only the still-missing scopes, and the operator's
  explicit `none`/`shadow` starter choice plus preemptible 9B quiet window survive
  reloads instead of being inferred by the model.
- **Operator-mode boundary:** `Operator OFF` now suppresses all agent-chat
  console-driving effects server-side, with a React-side mixed-version guard.
  Read-only research and chat answers remain available; explicit operator Apply
  actions still refresh the affected workspace after a trusted receipt.
- **Grounded archive search:** SigLIP archive search is explicitly a ranked
  candidate stage rather than a binary detector. EVA preserves bounded text
  evidence and server-resolved time/coverage metadata, deduplicates equivalent
  frames, and verifies the top 6–9 candidates in one multi-image VLM call.
  Positive and negative visual conclusions now require parsed per-frame
  verdicts; an uncertain or failed vision batch cannot become a confident
  “nothing found”, and the reviewed top batch is never presented as exhaustive
  proof about the whole archive.
- **Maritime deployment profile:** Protocol Deploy can assign each selected
  channel a port-gate, coastline, or mixed-PTZ role and preview bounded
  maritime L0–L3 prompts plus four role-specific shadow probes. Global pan,
  tilt, zoom, preset cuts, and settling create explicit scene epochs and
  coverage state instead of object-motion bursts; recurring views are matched
  before preset-specific probes regain authority, while independent one-Hz
  SigLIP2 indexing continues. React operator chrome can be switched between
  English and Latvian without translating evidence or model content.
- **Ventspils client freeze boundary:** the offline port appliance defaults to
  the React console with a legacy URL fallback, records the exact clean client
  branch revision in its manifest, and refuses an accidental dirty or `main`
  bundle unless an explicit recovery override is supplied.
- **Console polish:** Research traces are closed by default and only operator
  clicks change their disclosure state. Appearance can save, recall, overwrite,
  and delete up to twelve named browser-local custom presets after contrast
  validation.
- **Database and verification:** schema head is `20260805_0013`; `0012` adds
  incident lifecycle and temporal ledgers, while `0013` repairs archive
  source/channel paging on upgraded databases. Full release verification:
  1227 backend tests passed (23 skipped, 169 subtests) after the release fixes;
  83 React tests passed and the production UI build completed.
- **Universal offline deployment:** one `START_EVA_AI.sh` now detects fresh
  versus update deployments without Git or internet access. Bundle manifest v2
  covers fresh/update/report, refuses dirty source snapshots, inventories and
  verifies every APT package and Python wheel, proves the CPython 3.12 + vLLM
  dependency solve, carries the `0006 → 0013` SQL plan, backs up before update,
  automatically rolls back a failed apply and emits a secret-free acceptance
  report. The accepted React UI is release-managed while all site inference,
  channel, tenant, retention and Luxriot settings remain preserved.

## β 0.8.5 — 2026-07-27 release candidate (adaptive attention and memory)

- **Adaptive L0 delivery:** each channel now owns one bounded accumulator.
  Embedding-backed frames are admitted at quiet/normal/burst cadences of
  5/2/1 seconds, a full batch is sent immediately, no request exceeds 16
  snapshots, and every non-empty batch is flushed within 60 seconds. Frozen
  inputs remain explicit coverage heartbeats rather than silent gaps.
- **Attention audit plane:** dense CV is reduced to compact motion intervals;
  one frame per configured embedding cadence is linked to CLIP and probe P/N/M
  scores, attention episodes, scheduler decisions, and temporary-probe lineage.
  Dense raw CV frames are not archived by this telemetry path.
- **Unified visual memory:** L0 emits one `BATCH_STATE_JSON` contract for cover
  selection, scene match, episode continuity, watched states, routine,
  pass-up memory, and alerts. L1/L2/L3 prompts now describe 15-minute,
  one-hour, and eight-hour consolidation roles without inventing a human
  specialist persona.
- **Operator false-positive feedback:** archived VLM alerts accept one of five
  bounded reason codes plus an optional note. Reports can be generated by the
  agent or exported as Markdown/XML, and L3 may analyze reviewed failure modes
  without treating unreviewed alerts as false.
- **Archive and probes:** archive search supports multiple channels; batch
  identity keeps alert anchors, covers, and neighboring frames together.
  Temporary alert-derived probes are grouped by source channel and removed
  from the live board after expiry while their terminal lineage remains
  durable.
- **Agent stability:** terse continuations inherit the prior research scope;
  required inventory/detail steps are enforced for video research; duplicate
  read calls are suppressed; video turns stop after a bounded 10 tool calls
  and synthesize from completed evidence instead of running until the chat
  disconnects. EVA now discovers vLLM `max_model_len` and budgets against the
  smaller served/configured context, runbooks expose explicit tool allowlists
  with per-scenario call ceilings, and a failed final model synthesis returns
  and persists an evidence-only result instead of discarding completed calls.
- **UI:** the archive review modal pins preview, filmstrip, feedback, and
  summary to distinct grid rows, preserving a large snapshot preview while
  long L0 text scrolls independently. Agent research traces now preserve the
  operator's expand/collapse choice during streaming and after session reload;
  archive stream scope is a full-width searchable multi-select instead of a
  clipped half-column checkbox list. Opening a VLM-feed cover now renders its
  stored thumbnail immediately, loads batch neighbors from either the Video or
  Archive workspace, and exits the loading state with a bounded fallback if
  neighbor lookup stalls.
- **Probes tab:** the Monitoring tab is now **Probes**. Probes carry an
  `origin` of `operator`, `agent`, or `auto`, backfilled on read for probes
  stored earlier, so an approved agent probe is no longer indistinguishable
  from a hand-made one; an operator edit no longer strips a probe's authorship
  or its alert lineage. The board nests operator-defined channel group →
  channel → probes with grid and list layouts, author/state/search filters, an
  activity sparkline in place of the empty preview placeholder, an expiry
  countdown on temporary probes, and a jump from a background probe to its
  parent alert in the archive. Channel groups are EVA-side file-backed state
  (`EVOSSEARCH_PROBE_CHANNEL_GROUPS_FILE`); deleting a group never deletes
  probes.
- **Protocol Deploy:** the agent now commissions up to eight authorized
  channels through a durable inventory → groups → visual survey → operator
  policy → composite preview/apply workflow designed for a 4B head. Apply
  preserves unrelated Alert Criteria and installs bounded starter probes,
  counted-state profiles, the operator's preemptible 9B quiet window, and
  optional live starts. The first 15-minute pass requests one bounded L1,
  calibrates P/N/M from continuous semantic snapshots, and emits
  proposal-only threshold/cadence changes. Counted transition/dwell queries
  remain independent of alert cooldown and delivery.
- **Incident attention v0:** an operator can turn an archived alert or grounded
  L0 summary into a durable incident draft. EVA reconstructs bounded start/end
  candidates from neighboring summaries, structured transitions, alert anchors,
  CV intervals, and stored evidence links; coverage gaps and uncertainty remain
  explicit, while P/N/M stays labelled as attention rather than visual proof.
  Markdown/XML export is available. A TTL-bounded Follow/Critical lease raises
  only the affected channels to active/burst sampling, never masks a degraded
  source, never bypasses coverage fairness, and leaves independent 1 Hz semantic
  indexing running. Approved operator/agent probe scores can regulate subsequent
  observations for 60 seconds; automatic, temporary, attention-only, or
  embedding-space-mismatched probes remain shadow signals.
- **Offline port appliance:** an English interactive Ubuntu Server 24.04
  installer and USB-bundle builder now cover a clean RTX 4070 Super / i9 /
  64-GB deployment without network access. The bundle carries a local APT
  repository, Python 3.12 wheelhouse, PostgreSQL 16, NVIDIA driver and HWE
  kernel, Qwen3-VL-4B AWQ for vLLM, Qwen3.5-9B-MTP Q4 for preemptible CPU
  review, CLIP weights, and portable llama.cpp source. The installer verifies
  disk/GPU/schema state, preserves or backs up existing state, writes the
  runtime `.env`, provisions systemd and TLS, and hands off to Protocol Deploy.
- **Upgrade:** database migration required from β 0.8.4. Apply revisions
  `20260725_0007` through `20260801_0011`; resulting schema head is
  `20260801_0011`. New audit writes form a concurrency-safe, tenant-scoped
  SHA-256 chain, and incident drafts persist as bounded evidence references
  with optimistic revisions.
- **Verification:** 997 passed, 23 skipped, 161 subtests passed.
  See `readiness/RELEASE_NOTES_0.8.5.md`.

## β 0.8.4 — 2026-07-15 (attention decider, media broker, stabilization)

Release notes: `readiness/RELEASE_NOTES_0.8.4.md`.

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
