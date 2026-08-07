# Incident temporal memory

Status: implementation contract for `feature/incident-temporal-memory`.

## Purpose

EVA must preserve distinct real-world incidents while it consolidates visual
observations from L0 into L1, L2, and L3. A repeated label is not an identity,
returning to routine is not automatically risk resolution, and a coverage gap
is never evidence that an incident ended.

The temporal layer is server-owned. The VLM proposes grounded semantic
observations; it does not mint durable identifiers, merge incidents, resolve
channel ownership, or close cases by omission.

L0 display prose and state/routine ledgers may retain ordinary posture, gaze,
and scene-motion observations. The incident chronology is intentionally
narrower: generic movement, head turns, seated/resting state, motion blur, and
similar micro-motion do not create episodes by themselves. Alert records keep
their independent path. Accepted VLM episode labels are normalized to a stable
primary entity/action key (for example `vehicle maneuver` or
`cat jump_climb`), and routine `applies_to_event_keys` are normalized through
the same function before they can close an episode.

## Identity hierarchy

| Identifier | Meaning | Lifetime |
|---|---|---|
| `observation_id` | One accepted, grounded batch observation | Immutable |
| `episode_id` | One continuous observable episode | Until a confirmed boundary |
| `incident_id` | Durable case/risk object | Across one or more episodes |
| `series_id` | Relation between distinct incidents | Durable, non-merging |

IDs are generated deterministically or by the server. Model-provided event
labels are semantic hints and cannot be used as durable IDs by themselves.

## Independent state axes

An incident has independent state dimensions:

- `perception_state`: `unknown`, `observed`, `not_observed`, `ended`;
- `risk_state`: `unknown`, `active`, `contained`, `occurred`, `resolved`;
- `case_state`: `unknown`, `candidate`, `open`, `closed`, `dismissed`,
  `false_positive`;
- `attention_state`: `unknown`, `inactive`, `follow`, `critical`.

Foreground/hot/parked is a transient allocator tier recorded in runtime
telemetry; it is not written into the durable operator attention state.

Follow mode and criticality affect attention scheduling. They are not case,
risk, or perception states.

No state transition may be inferred solely because an item was omitted from a
model response.

## Routine and coverage

A routine boundary is scale-relative and semantic-track-relative:

- a covered return to routine can close an `episode`;
- it does not by itself resolve risk or close the case;
- `routine_at_L1` with explicit L2 baseline mismatch is a
  `long_incident_candidate`;
- separate episodes with a compatible series key and covered routine gaps are a
  `series_candidate`;
- missing/degraded/PTZ-away coverage is `coverage_gap`, never `routine_gap`.

Every child episode presented to a rollup must receive one disposition:

- `routine_at_this_scale`;
- `long_incident_candidate`;
- `series_candidate`;
- `continuing_incident`;
- `resolved_incident`;
- `absorbed_into_baseline_proposal`;
- `insufficient_coverage`;
- `unclassified_keep`.

The backend supplies `unclassified_keep` for omitted or invalid dispositions.

## Heartbeats

Each accepted L0 batch produces one idempotent observation heartbeat for every
hot followed incident on that channel, bounded by the eight-item hot set. The
conservative heartbeat uses `perception_state=unknown`: a current channel batch
is not automatically proof about every parallel incident. Grounded episode
association or operator review may append a later `observed`, `not_observed`,
or `ended` observation. Heartbeats are append-only. The incident row is a
materialized current projection, not the observation history.

## Operator synopsis and Follow result

The durable ledger and the operator report are intentionally different views
of the same incident. The ledger retains signal intervals, evidence references,
and every immutable L0 heartbeat. The default report projects only:

- a grounded three-to-five-word title;
- a short episode description assembled from semantic milestones;
- coverage and confidence;
- a compressed homeostatic response (activity apex, elevated duration,
  settling time, burst count, and probe support);
- no more than five key semantic moments.

CV/homeostatic values describe the system's attention response and are never
presented as visual proof. Signal-only motion intervals remain available in the
collapsed technical ledger and do not become a list of operator-visible
events. Historical incident rows can reconstruct the same digest from their
stored CV interval labels, so this presentation change requires no data
migration.

Each Follow invocation owns a run identifier and ends with a durable result.
The result reports whether grounded L0 observations supported continuation,
showed an explicit resolution, encountered a coverage gap, or remained
inconclusive. Silence or an unrelated batch never proves absence. Watching an
old incident is recorded as a recurrence watch rather than a continuation of
the historical episode. TTL expiry is reconciled by a bounded background tick,
and defensively on incident reads and the next accepted L0 heartbeat, making
the inactive state durable after restart even on a silent channel.

Creating an operator incident materializes one append-only primary episode.
An episode lasting at least fifteen observed minutes is labelled
`long_incident_candidate`; an open shorter episode remains
`continuing_incident`. If a later, non-overlapping incident on the same channel
has the same grounded semantic track, EVA appends one `series_member` candidate
relation to the nearest earlier incident. This is a review hint only:
`automatic_merge=false`, incident IDs remain distinct, and unrelated parallel
incidents are unaffected. Transport/generic keys such as `vlm_alert`,
`transition`, and `batch_state` cannot form a series. If an older build created
such a candidate, maintenance appends a `rejected` correction referencing the
bad relation instead of rewriting the append-only ledger.

Covered high-signal L0 observations may also open a durable
`case_state=candidate` record automatically. The server, not the VLM, assigns
the incident ID; an exact normalized semantic track continues that candidate,
while a different track remains a parallel incident. A grounded resolved event
or an explicit covered routine boundary ends the perceptual episode but leaves
the operator case in the review queue. Automatic creation is bounded to four
new candidates per accepted batch and to the per-channel hot-set admission
limit; it has no bookmark or external alarm side effect.

## Operator lifecycle review

Incident Review exposes five explicit, revision-guarded operator decisions:

- `confirm` opens the case and marks the incident reported without inventing a
  perception state or risk assessment;
- `resolve` closes the case and marks the risk resolved;
- `dismiss` closes a non-actionable case while preserving all evidence;
- `false_positive` closes the case under an explicit false-positive label;
- `reopen` returns a historical case to the open queue without starting Follow
  or asserting that the old visual episode is currently present.

Every decision writes the materialized lifecycle projection, an audit event,
and an immutable `operator_review` observation containing the previous and new
independent state axes. An optional operator note is bounded to 1,000
characters. Concurrent stale dialogs fail with a revision conflict instead of
overwriting a newer review. Closing an actively followed incident first
materializes the Follow result and releases its runtime attention lease.
Operator review never deletes evidence, rewrites temporal episodes, confirms a
series relation, or merges incident IDs.

The same lifecycle actions are available to the secured EVA agent as a
preview/apply tool. Channel ownership, optimistic revision and candidate series
relation IDs are resolved from durable state by the gateway; the model cannot
author those bindings. Every apply requires the ordinary operator approval
plan used by the UI mutation surface.

Markdown and XML exports carry the operator synopsis, compressed homeostatic
attention, Follow outcome, bounded episode/series projection and immutable
lifecycle history. Full raw observations remain queryable in the technical
ledger and are not repeated as the primary human narrative.

## Attention admission: 2 / 4 / 8

The limits bound hot cognition, not durable storage:

- up to 2 normal foreground incidents per channel receive detailed L0 context;
- up to 4 active incidents may enter the VLM prompt; critical or
  operator-selected incidents may use all four foreground positions;
- up to 8 unresolved incidents remain hot, receive bounded heartbeats, and
  compete for verification;
- overflow is parked or cold in durable storage and is never marked resolved
  because of capacity.

The implemented ranking is deterministic and considers operator selection,
criticality, unresolved state, incumbency, resolution debt, and recency. The
temporal ledgers retain grounded evidence, novelty, coverage, and risk as
separate inputs for a later scoring revision; they are not silently collapsed
into today's allocator. Incumbency provides basic hysteresis. Preemption changes
only the transient allocator tier, never lifecycle state.

The sampling level for a channel is the strongest active focus lease. Bounded
prompt memory must nevertheless retain all selected parallel incident IDs and
contexts rather than discarding lower-level leases.

## Context envelope

The 2/4/8 hot state is not copied wholesale into the model prompt.

- two normal foreground incidents share the configured incident envelope;
- an operator-selected or critical competition may admit up to four semantic
  stubs, upgraded to compact detail only while the envelope still fits;
- hot incidents five through eight remain scheduler/heartbeat state and do not
  receive automatic prompt prose;
- parked incidents remain durable retrieval candidates;
- cold incidents: retrieval by ID/query only.

Every live request has separately measurable budgets for system text, channel
memory, homeostasis/probes, incident context, vision input, and output reserve.
The initial live target is an 8--10k-token request inside a 12--16k model
context; deployment measurements remain authoritative.

Compaction order is deterministic:

1. remove decorative/repeated memory prose;
2. reduce non-foreground incidents to heartbeats;
3. remove visually duplicate frames while preserving pre/onset/apex/post/control;
4. reduce old evidence to IDs and one-line summaries;
5. reduce secondary-frame resolution;
6. split the batch rather than truncating protected suffixes.

Alert criteria and the final `BATCH_STATE_JSON` contract are protected and may
not be tail-truncated. Telemetry records text/vision/output budgets, selected
and dropped frames, selected incident tiers, compaction reasons, queue wait,
and inference latency.

## Safety and compatibility

- Existing migration `20260801_0011` remains immutable; lifecycle persistence
  is additive in `0012`.
- Existing rows backfill new semantic states as `unknown`; migration does not
  invent historical meaning.
- Automatic temporal episode aggregation runs through L0-L3 rollup metadata,
  while high-signal L0 events may create bounded durable operator-review
  candidates. Neither path creates bookmarks or external alarm side effects.
- Ambiguous matching creates a candidate relation, not an automatic merge.
- The pinned tuktuk grammar remains unchanged until agent-visible incident
  lifecycle tools and compact envelopes are reviewed explicitly.
- The recurring L1-L3 scheduler considers only running or desired-live
  channels. History-only channels are repaired through explicit backfill.
- At process start, L1/L2 wait for the first successful L0 summary on every
  healthy live channel (with a bounded grace period). Their first automatic
  run is placed at the next canonical 15-minute/hour boundary rather than
  immediately replaying every level after a restart, and recurring jobs read
  only a bounded recent window instead of rescanning the entire archive.

## Acceptance scenarios

1. A 47-minute fire and a two-frame pickpocket event on the same channel have
   distinct incident IDs.
2. `new -> continuing -> resolved -> covered routine -> new` creates two
   episodes even when labels match.
3. A coverage gap does not close an episode or resolve a risk.
4. A short ended theft remains `risk_state=occurred`, `case_state=open`.
5. A PTZ departure from a fire yields `not_observed/unknown`, not `resolved`.
6. Critical focus does not erase a parallel Follow context.
7. Repeated processing is idempotent and process restart preserves identities.
8. Missing rollup output becomes `unclassified_keep`.
