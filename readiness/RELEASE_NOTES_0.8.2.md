# Luxriot EVA AI β 0.8.2 Internal Release Notes

Release date: 2026-06-28  
Release type: office-demo hardening patch  
Previous baseline: `β 0.8.1`  
Schema head: `20260614_0006`  
Database migration: **none**
Runtime version source: `VERSION`; update `EVOSSEARCH_APP_VERSION` too if a
site-specific env file overrides it.

`β 0.8.2` is the stabilization build for the office demo with 20+ video-loop
channels. The operational goal is narrow and practical: video-description alerts
must be observable, the agent must aggregate objective reports from tool data,
and probe control must use archive-calibrated P/N/M rather than quick
under-tuned probe creation.

## Deployment Focus

- Primary workflow: **video descriptions and VLM alerts**.
- Agent reports should aggregate active channels, coverage gaps, dropped frames /
  batches, recent alert titles, delivery status, and pipeline health separately
  from incident narratives.
- Probes are a **secondary attention layer**. They are useful only when their
  positive / negative prompts and thresholds are tuned against channel context
  and visible background alternatives.
- Chat must stay preview-only for probe/prompt changes. Applying changes remains
  an explicit UI Apply / action-plan execution step.

## What Changed Since β 0.8.1

### VLM Alert Reliability

- Split live-feed prompt layers:
  - `stream_system_prompt` = L0 role / style / summary behavior.
  - `alert_policy_prompt` = channel-specific watch / alert criteria.
  - `json_alert_prompt` = structured alert-output contract.
- Added prompt-health migration support so legacy alert criteria embedded in the
  stream prompt can be moved into `alert_policy_prompt`.
- Added per-alert delivery status:
  - `sent`
  - `cooldown_skipped`
  - `bookmark_disabled`
  - `failed`
  - `state_tracker`
- Added parser diagnostics:
  - JSON alerts parsed
  - prose fallback alerts parsed
  - parser alert total
- Added backend state-transition tracking for structured L0 observations, with
  debounce/hysteresis for appearance/disappearance style events.
- Exposed provenance bundles to the agent so L0 prose, structured alert events,
  backend transitions, and routine-memory priors are not collapsed into one
  undifferentiated memory blob.

### Agent Reporting And Status

- Added a per-channel status digest used by runtime tools:
  - running / desired state
  - video-description model label
  - pending / dropped frames / dropped batches
  - last error
  - recent alert titles
  - alert delivery breakdown
  - parser breakdown
  - state transition counts
- `generate_report` now separates **Detection pipeline health** from incident
  findings.
- Current runtime status questions are routed to runtime tools, not to
  documentation lookup or static startup context.
- Agent rules now explicitly rank evidence:
  - routine memory = prior, not current evidence
  - L0 prose = current but can be contaminated by prior
  - structured alert/state events = stronger than prose
  - backend state transitions = confirmed cross-batch structured signal
  - frame evidence + `describe_frame` = strongest visual confirmation in chat

### Probe Control And Calibration

- Added server-side probe calibration workflow for multi-probe / multi-channel
  P/N/M review.
- Calibration returns deterministic tool decisions:
  - `calibration_status`
  - `separation_quality`
  - `safe_to_apply`
  - `recommended_action`
  - `warnings`
  - `recommended_probe_args` only when safe.
- Over-firing is treated as risk, not "excellent separation".
- Unsafe calibrations do **not** return apply-ready probe args.
- Negated contrast prompts such as `no weapon`, `not a person`, or
  `without smoke` are rejected/cleaned; contrast must describe visible
  alternatives or background state.
- Batch calibration stores progress server-side; the model receives a compact
  ledger and continuation handle instead of reconstructing long checklists from
  chat context.

### Live Smoke / Acceptance

- Added deterministic integration harness over the real HTTP/SSE contract.
- Added seeded archive fixtures for live-smoke archive search and probe
  calibration plumbing.
- Added admin live-smoke path for:
  - live status
  - archive search
  - probe calibration
  - preview-only apply lifecycle
  - report pipeline health
  - broad-channel chunking
- Added non-admin live-smoke path for documentation/RBAC restricted-help checks.
- Added `scripts/bootstrap_live_smoke_operator.py` for repeatable non-admin
  smoke setup.

## How To Test Before Office Demo

If the target machine has an env override, update it before restart:

```bash
sudo sed -i 's/^EVOSSEARCH_APP_VERSION=.*/EVOSSEARCH_APP_VERSION="β 0.8.2"/' /etc/eva-ai/eva-ai.env
```

Run deterministic tests:

```bash
.venv/bin/pytest -q
bash scripts/check_docs_drift.sh
```

Seed deterministic archive fixtures on the target demo channel:

```bash
.venv/bin/python scripts/seed_demo_fixtures.py --channel-id 112
```

Run admin live smoke:

```bash
EVA_LIVE_BASE_URL=https://127.0.0.1:5443 \
EVA_LIVE_USER=admin EVA_LIVE_PASSWORD='...' \
EVA_LIVE_CHANNEL_REF=112 \
EVA_LIVE_NEEDLE_QUERY="person lying on the ground at night" \
EVA_LIVE_PROBE_NAME="smoke: thumbs up gesture" \
EVA_LIVE_INCLUDE=seed \
.venv/bin/pytest -q tests/integration/test_live_agent.py -s
```

Create/update non-admin smoke user:

```bash
EVA_LIVE_OPERATOR_PASSWORD='...' \
.venv/bin/python scripts/bootstrap_live_smoke_operator.py --channel-id 112 --set-password
```

Run non-admin live smoke:

```bash
EVA_LIVE_BASE_URL=https://127.0.0.1:5443 \
EVA_LIVE_USER=operator-smoke EVA_LIVE_PASSWORD='...' \
EVA_LIVE_CHANNEL_REF=112 \
EVA_LIVE_INCLUDE=non_admin \
.venv/bin/pytest -q tests/integration/test_live_agent.py -s
```

## Manual Test Focus For PM / Intern

Use the 20+ video-loop channels to test:

- Do video-description alerts appear as structured events, not only prose?
- Do alert reports separate incidents from pipeline health?
- Does the agent state coverage gaps and unchecked channels?
- Can the agent find evidence frames and open/describe them?
- Does probe calibration flag over-firing, weak contrast, target-absent cases,
  or unsafe tuning instead of producing apply-ready changes?
- Does the agent keep probe changes preview-only and direct the operator to UI
  Apply for commits?

## Known Risks / Non-Goals

- Local LM Studio responses can take 5-10 minutes per complex agent turn on the
  dev/demo machine. This is expected for acceptance smoke and is not by itself a
  failure.
- Probe creation alone is not quality. A probe is meaningful only after channel
  context, positive examples, visible contrast/background negatives, thresholds,
  and representative frames are reviewed.
- CLIP P/N/M is a secondary attention signal, not proof. Use frame evidence and
  VLM/frame description before making operational conclusions.
- Routine memory decay is still a future hardening item. The current build gives
  the agent enough provenance to flag possible memory contamination, but it does
  not fully solve baseline decay.
- No schema migration is included. If a future patch touches schema, this release
  note does not apply.

## Verification Snapshot

Last local verification for the β 0.8.2 preparation pass:

- Admin live-smoke with seeded fixtures: passed.
- Non-admin live-smoke restricted-help path: passed.
- Full pytest suite: passed.
- Docs drift guard: passed.
