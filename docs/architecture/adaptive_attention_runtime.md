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
3. Exactly one selected frame per configured embedding cadence is embedded
   and kept in the bounded runtime ring. The interval links to this snapshot
   by UUID; it may also link to a VLM-apex reference.
4. The coordinator assigns homeostatic modes (`quiet`, `watch`, `active`,
   `burst`, `degraded`), coverage debt, priority, and a target revisit
   interval. These signals regulate frame admission; they do not replace the
   per-channel delivery contract.
5. Each channel owns one bounded L0 accumulator. Embedding-backed snapshots
   are admitted at 5 seconds in quiet intervals, 2 seconds during normal
   activity, and 1 second during bursts. The configured batch target is hard
   capped at 16 snapshots. A full batch is dispatched immediately; every
   non-empty batch is dispatched after at most 60 seconds by both source-time
   and wall-clock accounting. A newly arrived snapshot beyond that boundary
   starts the next batch instead of stretching the previous one. An unchanged
   or frozen source remains an explicit quiet/coverage heartbeat at the quiet
   cadence; it does not become a silent archive gap.
6. The model sidecar contains the dense interval aggregates and P/N/M for
   every active text probe on every admitted embedding, including values
   below thresholds. Experimental coordinator-owned sparse episode dispatch
   remains available behind a separate flag, but is off in the stable
   profile so one-frame coverage jobs cannot bypass L0 aggregation.

No raw dense-CV image, base64 image payload, or independent snapshot is
written by the attention telemetry path.

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

## Office defaults

- Luxriot capture window: 60 seconds at 4 CV samples per second.
- Embedding cadence: the channel snapshot interval (currently one second).
- L0 VLM admission cadence: quiet 5 seconds, normal 2 seconds, burst 1 second.
- L0 delivery: target batch capped at 16 snapshots; hard flush at 60 seconds.
- Runtime evidence ring: 90 seconds.
- Coordinator-owned sparse VLM episode dispatch: disabled.
- Experimental coordinator budget when enabled: 6 requests/minute, at most
  one outstanding request, 8 saved frames and 3 seconds of post-roll.
- Temporary alert-probe TTL: 300 seconds.

These are deployment controls, not model facts. They should be tuned from the
stored scheduler decisions, coverage debt, latency, and false-positive
feedback when the 4070-class port profile is measured.
