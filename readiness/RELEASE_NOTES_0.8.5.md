# Luxriot EVA AI β 0.8.5 Release Notes

Release date: release candidate prepared 2026-07-27  
Previous baseline: `β 0.8.4`  
Schema head: `20260726_0009`  
Database migration: **required**

`β 0.8.5` is the adaptive-attention, bounded L0 memory, operator-feedback,
and port-deployment preparation release. It turns dense CV into compact
homeostatic signals, keeps VLM work bounded per channel, makes the semantic
memory hierarchy explicit, and closes several agent and archive workflows
observed during the live office soak.

## Highlights

- Per-channel L0 batches with activity-sensitive 5/2/1-second admission,
  maximum 16 snapshots, and a hard 60-second flush.
- Compact durable attention telemetry linked to embedding snapshots instead of
  an archive of dense CV frames.
- One authoritative `BATCH_STATE_JSON` for covers, episode continuity,
  watched-state evidence, memory pass-up, and alerts.
- Operator false-positive annotation, agent reporting, and Markdown/XML export.
- Multi-channel archive search with stable VLM batch identity and neighboring
  evidence navigation.
- Bounded temporary alert-derived probes with terminal lineage rather than a
  monitor full of disabled cards.
- Evidence-enforced agent research with duplicate-call suppression and a
  10-tool video-research ceiling.
- Correct archive-review layout with a persistent large snapshot preview.

## Adaptive Live Attention And L0 Delivery

- Dense capture CV continues to evaluate the configured live stream rate
  (approximately 4 fps in the office profile), but those dense raw images are
  not stored by the attention telemetry path.
- Each one-second bucket is reduced to a compact motion interval: quiet/mixed/
  motion state, sample count, mean/max/p95/integral activity, moving fraction,
  channel-relative activity, and peak time.
- Exactly one selected frame per configured embedding cadence is embedded and
  retained in the bounded runtime evidence ring.
- The coordinator assigns `quiet`, `watch`, `active`, `burst`, or `degraded`,
  together with coverage debt, priority, and a target revisit interval. These
  values regulate attention and never substitute for visible evidence.
- Every channel owns one L0 accumulator:
  - quiet intervals admit a frame every 5 seconds;
  - normal activity admits every 2 seconds;
  - bursts admit every 1 second;
  - reaching the configured target sends immediately;
  - no request contains more than 16 snapshots;
  - every non-empty batch is sent after at most 60 seconds by source-time and
    wall-clock accounting.
- A snapshot arriving beyond the 60-second boundary starts the next batch
  rather than stretching the old one.
- An unchanged or frozen source remains an explicit quiet/coverage heartbeat.
  It is not silently interpreted as a calm, continuously observed scene.
- Experimental coordinator-owned sparse episode dispatch remains disabled in
  the stable profile so one-frame work cannot bypass L0 aggregation.

## Homeostatic Signals And Durable Audit

- Every admitted embedding can carry P/N/M values for every active text probe,
  including below-threshold results.
- VLM frames receive bounded dense-motion aggregates, attention mode, coverage
  state, and probe signals as a sidecar. The VLM still decides scene semantics.
- Attention state can move between the dense CV path, L0–L3 memory, the live
  operator view, and alert-derived probe feedback without treating CLIP or
  motion as proof.
- A bounded background writer stores immutable telemetry without blocking the
  capture hot path.
- `/ready` exposes attention storage state, queue/drop/failure counters,
  coordinator status, and active temporary-probe counts.

## Unified Visual Memory Contract

- The legacy-named JSON prompt field now contains one unified
  `BATCH_STATE_JSON` block rather than separate temporary cover/alert formats.
- Every L0 response chooses one representative cover and records why it was
  selected. A cover is a navigation thumbnail, not evidence by itself.
- The same block records:
  - current scene match/mismatch/uncertainty;
  - new, continuing, resolved, or uncertain episodes;
  - watched entities and operator triggers as present/absent/unknown with
    current snapshot indices;
  - routines visibly reinforced in the batch;
  - grounded items to pass to later consolidation;
  - distinct alert candidates tied to current snapshots.
- Archive investigations distinguish semantic-search match, model-selected
  cover, alert anchor, and neighboring context frames.
- L1 is a 15-minute episodic consolidation layer.
- L2 is a one-hour routine and recurrence layer.
- L3 is an eight-hour audit/regulatory-memory layer. It can compare hierarchy
  consistency, use bounded L0 drills, and separately analyze reviewed
  false-positive modes.
- None of the levels may invent visual observations, silently suppress general
  hazards, or treat sampled drill windows as complete coverage.

## Operator False-Positive Feedback

- An operator can open an archived VLM alert, review its batch evidence, and
  select one reason:
  - no relevant event;
  - benign activity;
  - wrong object or actor;
  - duplicate or stale alert;
  - poor visual quality.
- An optional note is stored with the exact detection, channel, alert snapshot,
  operator identity, and timestamps.
- Repeated submission by the same operator updates the annotation instead of
  creating an ambiguous duplicate.
- Reports support time and multi-channel filters, reason/channel counts, item
  evidence, explicit truncation state, and operator-annotation provenance.
- Operators with export permission can download Markdown or XML reports.
- The agent can generate the same bounded false-positive report from tools.
- Feedback is privileged review input, not automatic ground truth. Unreviewed
  alerts remain unclassified.

## Archive And Evidence Navigation

- Archive filters accept multiple channels without requiring one query per
  stream.
- VLM batch rows carry a stable `batch_id`; the database indexes tenant,
  channel, batch, timestamp, and detection order for bounded neighbor lookup.
- The review filmstrip retains model cover, alert anchor, and nearby batch
  frames so an alert can be inspected in temporal context.
- The large selected snapshot and the long L0 description now occupy explicit
  grid rows. Hidden feedback UI can no longer move the description into an
  unconstrained row and collapse the preview.
- Stored evidence remains primary. Archive video playback is still an explicit
  operator action and depends on recorder coverage.

## Alert-Derived Probe Lifecycle

- A direct VLM alert may admit two to four temporary generation-zero probes
  when the feature is enabled.
- Automatic probes are positive-only, low-confidence, `attention-only`, and
  expire after five minutes by default.
- They influence P/N/M and attention selection but do not create recorder
  bookmarks or independent detection rows.
- Probe hits cannot recursively create another probe generation.
- Per-channel/global caps, semantic deduplication, cooldown, TTL, and lineage
  bound the loop.
- Active temporary checks appear collapsed under their source channel.
- Expired checks are removed from the live registry after terminal lineage is
  persisted; the monitor no longer becomes a cemetery of disabled probes.

## Agent Reliability

- The agent and VLM are described as complementary intellectual-core
  functions, without assigning a fictional human-security-specialist role.
- Terse follow-ups such as a time-window choice or `continue` inherit the
  immediately preceding operator research scope.
- Broad video research must complete the required inventory and detail steps
  before producing a factual report.
- Read-only tool results are cached within the turn. Repeating the same tool
  call is suppressed and the model is instructed to answer from cached
  evidence.
- Video-research turns are capped at 10 tool calls by default (configurable
  from 5 to 16). The model then synthesizes from completed evidence instead of
  continuing an unbounded loop after the browser disconnects.
- Planning-only or evidence-free video answers trigger bounded recovery.
- Tool results exposed to the UI use the same compact envelope as the model
  for large video-summary inventories, preventing another oversized stream
  after successful background work.
- Tool-run audit phases are projected into the durable agent store.

## Database Migration

Upgrade from `β 0.8.4` requires all three revisions:

1. `20260725_0007` — `archive.alert_feedback`, indexes, RLS, and role grants.
2. `20260726_0008` — embedding snapshots, probe scores, motion intervals,
   evidence links, attention episodes, scheduler decisions, and probe lineage.
3. `20260726_0009` — stable VLM batch-identity index.

Required final head:

```text
20260726_0009
```

Before migration:

- stop new operator changes;
- retain the deployment environment and model endpoints;
- take a PostgreSQL backup;
- use the distinct privileged migration DSN;
- verify that the installed release is `β 0.8.4` and record current
  `alembic current`.

Apply from the `0.8.5` source tree:

```bash
set -a
. /etc/eva-ai/eva-ai.env
set +a
.venv/bin/alembic current
.venv/bin/alembic upgrade head
.venv/bin/alembic current
```

Do not use the historical `field_upgrade_084.sh` or the 0.8.4 Git guide as an
0.8.5 recipe. The generic `update_bundle.sh` remains deliberately
non-migrating and accepts the code overlay only after its read-only gate sees
`20260726_0009`. The migration-capable offline installer is
`scripts/install_eva_083.py`; despite its legacy filename, its version is read
from `VERSION` and its expected schema is now `20260726_0009`.

## Operator Acceptance Focus

- Confirm `/health.version` and `/ready.version` report `β 0.8.5`.
- Confirm `/ready` reports database revision `20260726_0009`.
- Run a live channel through quiet, normal, and burst motion:
  - no batch exceeds 16 snapshots;
  - a quiet non-empty batch arrives within 60 seconds;
  - motion increases frame admission without erasing coverage honesty.
- Verify the frame sent to the VLM matches the archived embedding snapshot and
  that dense motion intervals link to the intended snapshot/apex.
- Open a VLM alert, navigate neighboring frames, submit a false-positive
  reason, and export a report.
- Confirm a temporary alert-derived probe expires from the monitor while its
  terminal lineage remains queryable.
- Ask the agent for a bounded last-night report and verify visible tool evidence,
  completion within the video-research call limit, and an operator-useful final
  answer.
- Verify L1/L2/L3 cadence and coverage: 15 minutes, one hour, and eight hours.

## Known Limits

- Capacity for 4–6 simultaneous VLM description streams on the target
  4070-class host still requires a measured port soak; this release establishes
  the controls and telemetry but does not claim that capacity in advance.
- Dense CV and CLIP/P/N/M signals are attention candidates, not visual or legal
  proof.
- Attention storage, sparse episode dispatch, embedding-all-channels, and
  alert-derived probes are separately configurable. A disabled subsystem must
  be reported as disabled rather than silently assumed active.
- The generic adopt updater does not run migrations.
- Local USB cameras still have no Luxriot recorder archive or bookmark target.
- Browser-playable archive video still depends on upstream Luxriot coverage.
- Model servers, GPU drivers, CUDA, vLLM, and llama.cpp are outside the release
  bundle.

## Verification

- Full project suite: **782 passed, 23 skipped, 138 subtests passed**.
- Archive modal regression contract: **45 passed** in the focused UI/CSS suite.
- Visual archive-modal smoke: Chromium at `1040×896`, with a long L0 summary
  and hidden feedback panel, retained the large preview and independent summary
  scroll.
- `git diff --check`: clean before the `0.8.5` metadata pass.
- Office EVA runtime completed the preceding overnight soak without a reported
  application incident; port capacity remains a separate acceptance gate.
