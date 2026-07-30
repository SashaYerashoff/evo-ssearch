# Adaptive live-attention runtime

Status: deployed on the EVA office runtime, 2026-07-26.

This is the runtime bridge between the ИОС framing in
`docs/Инженерные Основы Со-знания.md` and EVA's existing capture, CLIP, VLM,
archive, and probe subsystems. It treats CV/CLIP as attention signals, never
as visual proof; the VLM remains the semantic observer.

## Data flow and retention

1. Capture CV evaluates the dense stream at its configured rate (currently
   about 4 fps for Luxriot live segments). Dense frames are not archived.
2. Each one-second bucket becomes a compact `motion_interval`: quiet/mixed/
   motion state, sample count, mean/max/p95/integral, moving fraction,
   channel-relative `activity_x`, and peak time.
3. Exactly one selected frame per configured embedding cadence (1 Hz in the
   port preset) is embedded, kept in the bounded runtime ring, and persisted
   as `source=semantic_snapshot`. This archive path is independent of probe
   hits, alerts, and VLM admission, so operator semantic search retains the
   continuous indexed history. The interval links to this snapshot by UUID;
   it may also link to a VLM-apex reference. Continuous rows use per-channel
   hourly FAISS shards. `All evidence` fans retrieval out by source, while
   continuous retrieval ranks every matching shard instead of silently
   limiting the time window to the newest 20,000 rows.
4. The coordinator assigns homeostatic modes (`quiet`, `watch`, `active`,
   `burst`, `degraded`), coverage debt, priority, and a target revisit
   interval. These signals regulate frame admission; they do not replace the
   per-channel delivery contract.
5. Each channel owns one bounded L0 accumulator. The `port-4070s-8ch` profiles
   are: quiet `10 s / 120 s / 6-8-8`, watch `5 s / 90 s / 6-8-10`, active
   `2.5 s / 60 s / 8-12-12`, burst `1 s / 30 s / 10-16-16`, and degraded
   `15 s / 120 s / 4-6-6` (cadence/deadline/min-target-max frames). The hard
   accumulator cap is 16 in every mode. A newly arrived snapshot beyond the
   active mode deadline starts the next batch instead of stretching the
   previous one. An unchanged or frozen source remains an explicit degraded
   coverage heartbeat; it does not become a silent archive gap.
6. The model sidecar contains the dense interval aggregates and P/N/M for
   every active text probe on every admitted embedding, including values
   below thresholds. Experimental coordinator-owned sparse episode dispatch
   remains available behind a separate flag, but is off in the stable
   profile so one-frame coverage jobs cannot bypass L0 aggregation.

Dense CV frames are never archived. Only the selected 1 Hz thumbnail and its
already-computed normalized CLIP vector are written to the semantic archive;
the writer never recomputes CLIP.
Readiness exposes per-channel observed Hz, last-frame staleness, wall-cadence
gaps, and source-timestamp gaps separately from database write failures.
Local V4L2 analytics uses one direct ffmpeg process; it does not re-encode the
camera into multipart MJPEG and decode it again. Luxriot/Evo streams retain the
authenticated pipe path.

## Alert-derived probes

A direct VLM alert may admit two to four temporary generation-zero probes.
Fallback probes are positive-only, explicitly low-confidence, expire after
five minutes by default, and are `attention-only`: they influence P/N/M and
episode selection but do not create recorder bookmarks or separate detection
archive rows. Probe hits cannot create another generation of probes.

Per-channel and global caps, semantic deduplication, cooldown, TTL, and
lineage records bound this loop. Expired temporary definitions are removed
from the live probe registry after their terminal lineage is submitted; the
lineage table, not a disabled card, is their durable history. Active temporary
checks remain grouped under their source channel in a collapsed monitor
section.

## Durable audit plane

Alembic revision `20260726_0008` adds tenant-isolated tables for:

- embedding snapshot references and per-probe P/N/M;
- dense motion intervals and evidence links;
- attention episodes and scheduler decisions;
- temporary-probe lineage.

The capture hot path submits immutable records to a bounded background writer.
`/ready` reports the PostgreSQL store, queue/drop/failure counters, scheduler
state, and active temporary-probe counts. `/luxriot/streams` exposes the
attention coordinator with channel scoping applied by the auth layer.

## Port eight-channel defaults

- Luxriot capture window: 60 seconds at 4 CV samples per second.
- Embedding and semantic-archive cadence: one second on every enabled channel,
  independent of alert state.
- CLIP execution: cross-channel microbatches up to 8; no dynamic thinning.
- L0 VLM admission: mode profiles listed above, with a hard 16-frame cap.
- Global L0 cost budget: six reference requests/minute across tokens and
  slot-seconds; bursts may borrow two reference requests and repay that debt
  before ordinary credit is restored.
- Shared VLM admission: agent, alert/describe, and rollup work has one
  protected slot; L0 may borrow it while no protected work is waiting.
- Runtime evidence ring: 90 seconds.
- Coordinator-owned sparse VLM episode dispatch: disabled.
- Temporary alert-probe TTL: 300 seconds.
- L1/L2 use the agent profile. Optional deep L3 uses a separate CPU endpoint,
  concurrency one, only inside an operator-defined quiet window and only when
  activity, alerts, and L0 debt permit it. L3 output is proposal-only.

## Eight-channel commissioning

`Protocol: Deploy` persists its inventory, operator-selected scope/groups,
scene-survey receipt, policies, composite plan, and apply receipt outside the
chat context. This keeps the workflow predictable when the active head is a
small 4B model: tool schemas use closed enums, the server owns channel
iteration, and compact results expose one `next_action`.

Apply is one approval-gated composite operation. It preserves unrelated Alert
Criteria, creates no more than four starter probes per channel, and can start
the selected live sessions. Fifteen-minute commissioning waits for at least
120 continuous `semantic_snapshot` rows per channel, requests at most one new
L1 synthesis per channel, calibrates P/N/M against the independent archive,
and estimates episode cadence for cooldown/dedup. All resulting changes remain
proposals; semantic rephrasing, severity changes, and numerical updates require
operator review.

Counted-state profiles aggregate confirmed sampled transitions and state
segments from the continuous archive. They deliberately ignore bookmark
delivery, cooldown, and deduplication, and report unknown/uncovered time
separately from observed dwell.

These are deployment controls, not model facts. They should be tuned from the
stored scheduler decisions, coverage debt, latency, and false-positive
feedback when the 4070-class port profile is measured.
