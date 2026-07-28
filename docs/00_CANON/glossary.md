# Glossary

Canonical terminology. These terms are used with **exactly** these meanings in
all EVA AI documentation. The biggest source of confusion is conflating
**probes**, **video-descriptions**, **detections**, and **alerts** — they are
distinct. When in doubt, link here.

## Perception & description

- **Channel** — one Luxriot Evo camera/stream EVA AI can observe.
- **Snapshot** — a single frame pulled from a channel at the capture cadence
  (`EVOSSEARCH_LUXRIOT_SNAPSHOT_INTERVAL`).
- **Batch** — a group of snapshots (default 12) sent together to the VLM.
- **Video-description (L0 / live summary)** — the VLM's natural-language
  description of a batch. This is the system's primary always-on perception of a
  channel. Produced by `qwen3-vl-4b`.
- **VLM** — the vision-language model that produces video-descriptions.

## Aggregation

- **Rollup** — an aggregated summary over a time window, built from lower levels.
- **L0 / L1 / L2 / L3** — summary levels by time window:
  - **L0** = per-batch live video-description (frame-time anchored).
  - **L1 / L2 / L3** = rollups over progressively larger windows
    (default ~15 min / 1 h / 8 h). L1–L3 may be LLM-synthesized or deterministic.
- **Read-only rollup** — reading rollups for investigation **without** triggering
  LLM synthesis (used by agent investigation tools).
- **Salience-weighted compression** — when a rollup exceeds its budget, alert /
  deviation lines are kept before routine lines, so incidents are not averaged away.

## Alerts & bookmarks

- **BATCH_STATE_JSON** — the single structured current-batch contract appended by
  the VLM. It carries the chosen cover, scene/episode continuity, observed states,
  routine and memory-pass items, and zero or more grounded alerts. Always requested;
  independent of bookmark settings.
- **ALERTS_JSON** — legacy input format accepted while reading older summaries; it
  is no longer requested from the VLM.
- **Alert** — one parsed entry from `BATCH_STATE_JSON.alerts` (title, description,
  severity, state, timestamp, and snapshot references).
- **Severity** — `info | low | normal | high | critical`.
- **Bookmark** — an event pushed **into Luxriot Evo** from an alert. Gated by
  `bookmark_enabled` and a per-channel cooldown. A bookmark can fail to deliver
  (counted as `bookmark_failed_count`) without losing the in-EVA alert evidence.

## Probes (CLIP matching)

- **Probe** — a CLIP text/image matcher with positive (and optional negative)
  phrases scored against a channel's frame buffer. Used for semantic
  bulk-comparison; in this deployment, primarily an **agent-invoked** tool.
- **pos_floor / margin** — probe thresholds. Raising either makes the probe
  **stricter**; lowering makes it more permissive.
- **Watch-list probes** — a curated probe set (e.g. fighting, vehicle drifting)
  cast across channels as a parallel, VLM-independent detector.
- **Probe origin** — probe authorship, stored as `origin` on the probe and shown
  as a badge on the Probes board. Exactly one of:
  - **`operator`** — created through the operator UI. Probes stored before this
    field existed are backfilled to `operator` on read.
  - **`agent`** — proposed by the agent and written only once an operator
    approved the change. An agent editing an existing probe does not take over
    its authorship.
  - **`auto`** — a temporary follow-up created by the alert-probe lifecycle from
    a VLM alert. Distinct from the lifecycle's own `source: vlm_alert` lineage
    guard, which answers a different question.
- **Channel group** — an operator-defined label grouping channels on the Probes
  board. EVA-side only; Luxriot exposes no group concept. A channel belongs to
  at most one group, and ungrouped channels render under "Ungrouped".
- **Counted-state profile** — an operator-approved pair of visible states plus
  a transition/dwell rule. EVA evaluates it over continuous archived semantic
  snapshots; its episode count is independent of whether an alert/bookmark was
  sent, deduplicated, or suppressed by cooldown.
- **Protocol Deploy** — the durable, approval-gated initial commissioning
  workflow for at most eight channels: inventory → scope/groups → visual survey
  → operator policy → composite preview/apply → proposal-only commissioning.

## Archive & search

- **Detection** — a stored row in the frame archive. Sources:
  - **`probe`** — an actual probe hit.
  - **`vlm_summary`** — a frame sampled from a video-description batch.
  - **`vlm_alert`** — a frame anchored to a VLM alert.
  - **`semantic_snapshot`** — the continuous cadence-selected CLIP snapshot,
    archived independently of probes, VLM alerts, and VLM admission.
- **Frame archive** — PostgreSQL `archive.detections`; each row carries a CLIP
  vector (`bytea`) + thumbnail, indexed by time/channel/source.
- **Semantic search** — text/image → CLIP vector → ranked against candidate
  frames. Ranking uses an in-memory FAISS flat index per `ch:date` shard;
  **recall is bounded by the candidate window** (`ORDER BY ts DESC LIMIT N`), so
  broad searches must be scoped by channel + time.
- **Coverage contract** — metadata returned with a query: period requested vs.
  period inspected vs. entries returned vs. truncated. Prevents a search from
  silently reporting "nothing found" when it only inspected a recent slice.

## Sessions & runtime

- **Run / session** — a capture session for one channel. Runs have start/end
  times; a drop + reconnect produces a closed run and a new run. Source of
  "what connected / dropped over a period."
- **Inference queue / spool** — durable queue for summary batches. The code
  default is off for unconfigured development; the clean appliance profile
  enables it with one worker.
- **Tenant** — RLS isolation key; all archive/agent/audit rows are tenant-scoped.

## Concepts

- **CLIP vs VLM split** — CLIP is fast, cheap, semantically-steerable retrieval
  (probes, search, evidence ranking); the VLM is the slower, richer perception
  that produces the always-on video-descriptions and alerts. Reports and the
  agent center are anchored on **video-descriptions**; CLIP/probes serve the
  agent as a retrieval signal.
