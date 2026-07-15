# Luxriot EVA AI β 0.8.4 Release Notes

Release date: release candidate prepared 2026-07-15  
Previous baseline: `β 0.8.3`  
Schema head: `20260614_0006`  
Database migration: **none**

`β 0.8.4` is the attention, media, semantic-history, agent-stability, and field
upgrade release. It keeps the PostgreSQL schema and Python dependency set from
`0.8.3`, while substantially changing live frame selection, archive review,
background rollups, agent context/tool routing, and offline deployment.

## Highlights

- Channel-relative quiet/normal/burst capture attention with sharpness-aware
  action/companion frame selection.
- Tokenized same-origin live/archive media broker and evidence-first archive
  review with explicit video playback.
- Durable L1–L3 semantic rollups, legacy rollup adoption, and approval-gated
  restoration of missing summary history from archived L0 text.
- Resource-aware LM admission/backpressure and visible coverage gaps.
- Intent-routed agent tools, compact evidence-preserving tool results,
  non-thinking Qwen calls, completion recovery, and a 65 536-token target
  context.
- One-command offline update with bundled FFmpeg/OpenCV, user/system systemd
  detection, read-only schema gate, backup, readiness verification, and
  automatic rollback.

## Capture Attention And Video Runtime

- The apex decider measures per-second motion against the learned baseline of
  each channel and emits quiet/normal/burst attention markers.
- Frame selection supports `auto`, `action`, and `clarity` bias.
- Burst seconds can retain an action frame plus a sharper companion frame.
- `capture_attention` enters bounded `VECTOR_SIGNALS_JSON` and is available to
  the agent through `list_attention_bursts`.
- Burst is an attention candidate, not semantic proof. VLM/frame evidence is
  still required before describing an incident.
- L0 backpressure coalesces bounded windows rather than failing silently; gaps
  are exposed to summaries and the agent.
- Rollup scheduling is LM-resource-aware so a busy VLM endpoint does not idle a
  separate agent endpoint.

## Operator Media And UI

- Live/archive playback uses a tokenized same-origin media broker with bounded
  upstream timeouts, lease renewal, and stall recovery.
- `Model view` reuses the EVA attention stream instead of opening a second
  recorder session.
- Archive review opens on the stored evidence frame and filmstrip; video is an
  explicit operator action.
- Browser playback timeout/gap keeps the stored evidence visible and reports
  the limitation honestly.
- Filmstrip roles distinguish alert/action/companion/context frames.
- Metadata-only rows remain `No image`; they are not presented as visual proof.
- Video-history navigation and the vertical review workspace were redesigned.

## Semantic History

- L1–L3 operator narratives are stored separately from machine homeostasis.
- Scheduled semantic rollups use the dedicated agent LM profile with model
  thinking disabled.
- Genuine `0.8.0/0.8.1` LM-generated cached rollups are adopted as labelled
  legacy semantics and promoted to durable rows without regeneration.
- Mechanical fallback strings are excluded from semantic history.
- An approval-gated restore workflow can queue missing L2/L3 (and explicitly
  requested L1) windows from archived L0 text.
- Restore progress survives restart, yields to live descriptions, reports ETA,
  and separates queueable work from non-restorable source gaps.

## Agent Reliability

- Relevant tool schemas are selected from operator intent. Plain chat receives
  no tools; runtime, help, archive, probe, prompt, and video research receive
  bounded task-specific sets.
- Broad video research inventories scope before detail tools, avoiding unscoped
  summary calls and accidental all-tool fan-out.
- Security sanitization preserves semantic entries and `image_url` while byte
  and row limits remain bounded.
- Verbose time/coverage/result contracts are compacted before the local model
  without discarding key evidence or truncation state.
- Empty or planning-only final model output is replaced by an evidence-only
  completion from trusted executed tools.
- Qwen3.5 calls request `enable_thinking=false` for tool decisions and final
  answers.
- Default agent context target is 65 536 tokens; history budget 16k, warning
  52k, hard tool stop 60k.
- The updater reads the context reported by the agent endpoint. An operator can
  explicitly `FORCE-CONTEXT` when LM Studio is shorter; EVA is then temporarily
  capped to the actual compatible value and the result is marked degraded.
- The context probe sends the configured agent-profile API key when present. If
  `/models` is unreachable or does not expose a context value, installation
  requires a separate `FORCE-UNKNOWN-CONTEXT` decision before service stop.
- System deployments discover the active configuration from the service's
  `EnvironmentFiles` property before standard-path or application `.env`
  fallbacks. The selected agent and Luxriot endpoints are cross-checked against
  the running `/ready` identity, and the baseline service must be ready before
  it is stopped. A runtime schema report can safely supply the read-only schema
  gate when a deployment injects its DSN outside the selected file.
- Field hotfix artifacts use a unique revisioned filename instead of replacing
  another archive with the same name, preventing stale Google Drive/browser
  cache entries and mixed extracted directories from passing the wrong
  operator instructions.

## Offline Update

- The bundle root contains `./update.sh`; run it without `sudo`.
- It detects user or system systemd and requests sudo only for required system
  operations.
- Compatibility is determined by invariants, not a deployed-version allowlist:
  - installed `VERSION` is present for rollback verification;
  - bundle VERSION/manifest/commit are valid and clean;
  - `requirements.txt` and `requirements-db.txt` match exactly;
  - the existing venv passes pip/uv dependency checks;
  - database schema is exactly `20260614_0006` through a read-only query.
- This admits `0.8.2`, `0.8.2.1`, `0.8.3`, and compatible intermediate
  post-schema builds. Exact same-bundle reruns remain blocked by
  `.eva-bundle-commit`; another `0.8.4` commit is treated as a hotfix.
- The updater backs up code and env, never migrates the database, preserves
  runtime data, and automatically restores code/env after a post-stop failure.
- Code and emergency rollback snapshots exclude nested Git, venv, env, build,
  model, media, database, log, and runtime-data trees. Restore also ignores
  those entries in older snapshots, including retired absolute `.venv*`
  symlinks, instead of copying them over adopted runtime state.
- Automatic rollback distinguishes a fully ready restored release from a
  restored release whose external required dependency remains unavailable; it
  reports the degraded dependency state without a second four-minute wait.
- Success requires active service, `/health`, and `/ready?load=1` reporting
  `status=ready` and `β 0.8.4`.
- Linux x86_64 FFmpeg/ffprobe and an OpenCV rescue wheel are included and
  checksum/smoke tested offline.

## Upgrade Compatibility From Office β 0.8.3

The office-demo branch `stable/office-demo` at `9e39392` was inspected:

- `VERSION` is `β 0.8.3`;
- migration `20260614_0006` is in its ancestry;
- `requirements.txt` is byte-identical to `0.8.4`;
- `requirements-db.txt` is byte-identical to `0.8.4`;
- no source files are deleted by the `0.8.3 → 0.8.4` code overlay.

The target database must still pass the live read-only schema gate; repository
ancestry is not a substitute for checking the deployed database.

## Operator Acceptance

Use `readiness/MANUAL_TEST_PLAN_0.8.4_OFFICE_DEMO_RU.md`. The required focus is:

- evidence-grounded completed agent answers;
- coverage/gap honesty;
- relevant tool routing;
- correct candidate/uncertainty language;
- live lost/frozen recovery;
- archive evidence and video playback;
- preview/Apply discipline;
- restart and desired-state restoration.

## Known Limits

- A forced context below 65 536 reduces the room for long multi-channel turns.
- Pending L0 work remains in process memory when the PostgreSQL inference queue
  is disabled; coalescing makes loss bounded and visible, not durable.
- Browser-playable archive video depends on upstream Luxriot archive coverage.
- Road-CV, CLIP, and capture-attention signals are candidates, not proof or
  legal conclusions.
- The bundle does not install GPU drivers, models, CUDA, LM Studio, vLLM, or
  llama.cpp services.

## Verification

- Installer/agent targeted suite after compatibility changes: 171/171.
- Exact office `0.8.3 / 9e39392` fixture with a working deployment env passed
  bundle media, requirements, venv, 65k agent-context, and live read-only schema
  gates through the final human confirmation. The same fixture with the source
  office `.env` stopped safely at `NO_DSN`, revealing that the source-worktree
  env is not sufficient evidence of the deployed service configuration.
- Clean-worktree full pytest snapshot: 657 passed, 23 skipped, 133 subtests;
  three known failures require a configured PostgreSQL archive/probe store and
  demonstrate that those tests are not yet hermetic without deployment env.
- Archive UI/media focused suite after timeout-contract repair: 65/65.
- A same-version live failure rehearsal correctly entered automatic rollback
  when the configured Luxriot endpoint was unreachable. It restored the prior
  bundle marker and active service without touching database/runtime data. The
  rehearsal exposed an over-broad emergency snapshot and a retired absolute
  venv symlink in the older backup; both rollback paths are now bounded and
  regression-tested.
- Previous `0.8.4` deterministic predeploy gate: 626 passed, 18 skipped, 134
  subtests passed.
- Agent/updater hotfix targeted suite before this documentation pass: 141/141.
- Final installer/agent/media regression selection after systemd discovery and
  English-only runtime/UI enforcement: 234/234.
- A real code-only snapshot of the current dev tree was 113 MB with no Git,
  local state, venv, env, dist, or node_modules members; the pre-fix emergency
  snapshot had grown to 3.9 GB before being interrupted.
- The revisioned `r3` candidate is built clean with bundled FFmpeg/OpenCV. Its
  manifest commit and SHA-256 are recorded in the operator handoff before
  deployment.
