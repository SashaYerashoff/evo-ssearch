# Operator Scenarios

Worked examples for the public-order monitoring the client cares about: urban
streets, squares, and house frontage — any dangerous expression a patrol officer
would act on. Each scenario gives the **ask**, the **tool path** the agent takes,
the **evidence** to expect, and how to **confirm**.

Terms: [glossary](../00_CANON/glossary.md). What the agent can/can't do:
[agent_capabilities](agent_capabilities.md).

## How to ask well (applies to every scenario)

- Always give a **period** and a **channel** (or "active channels"). Use natural
  relative time ("last 2 hours", "yesterday evening") — the agent normalizes it.
- For "find" questions, the agent searches archived frames; **scope by channel +
  time** so recall is complete within the window.
- For "how many times X happened" questions, expect **candidate** transitions
  with boundary-frame evidence — confirm visually before reporting as fact.

## Incident scenarios

### Fight / assault
- **Ask:** *"On channel N in the last 3 hours, were there any fights or physical
  altercations?"* or *"Search the archive for people fighting."*
- **Path:** video-summary read (alerts) → archive semantic search on
  `vlm_alert`/`vlm_summary` frames → `describe_frame` on top matches.
- **Confirm:** open boundary/top frames, describe them, check Luxriot bookmark.

### Dangerous / reckless driving (drift, burnout, speeding into a crowd)
- **Ask:** *"Find a vehicle drifting or doing a burnout at the square today."*
- **Path:** semantic search (CLIP catches the distinctive smoke/skid) + alerts.
- **Note:** the watch-list probe for "vehicle burnout/drift" is the parallel
  detector — use its hits as entry points if the VLM under-reported.

### Crowd gathering / escalation
- **Ask:** *"How many times did a crowd gather and disperse on channel N in the
  last 6 hours?"*
- **Path:** `track_visual_state_transitions` with positive "crowd gathered" and a
  visible-background negative ("empty square") — **not** "no crowd".
- **Confirm:** boundary frames; treat counts as candidates.

### Weapon / fire / smoke / immediate hazard
- **Ask:** *"Any visible weapons, fire, or smoke across active channels today?"*
- **Path:** cross-channel alert report (`list_video_summary_channels` sorted by
  alert total) → drill into the flagged channel → describe frames.

### Vandalism / property damage / theft-like tampering
- **Ask:** *"Search for someone damaging property or tampering with a vehicle/gate
  on channel N tonight."*
- **Path:** semantic search + alerts; confirm with describe_frame.

### Person down / medical / possible victim
- **Ask:** *"Did anyone fall or end up lying on the ground on channel N?"*
- **Path:** `track_visual_state_transitions` (positive "person lying on the
  ground", negative "people standing/walking") + semantic search.

### After-hours loitering / suspicious presence
- **Ask:** *"What unusual presence or loitering happened in front of the houses on
  channel N between 01:00 and 05:00?"*
- **Path:** video-summary read over the night window; rollups as a map, L0 for
  exact moments.

## Operational scenarios (status & reporting)

### "What happened over a period" (the core daily question)
- **Ask:** *"Summarize what happened on channel N yesterday evening."*
- **Expect:** L2 context → L1 candidate windows → L0 specifics, with evidence
  frames and a coverage line. Read-only (no LLM rollup synthesis triggered).

### "Which channels are active / what dropped or reconnected"
- **Ask:** *"Which channels produced video-descriptions in the last 12 hours, and
  did any drop or reconnect?"*
- **Expect:** per-channel running state, alert counts by severity, and coverage
  gaps. (If session connect/drop detail is thin, cross-check the stream health
  panel — see roadmap note in agent_capabilities.)

### Cross-channel sweep
- **Ask:** *"Across all active channels, where was the most concerning activity in
  the last hour?"* → channels ranked by alert total/severity.

### Find-similar
- Open any frame → *Find similar* / *describe* → semantic neighbors from the
  archive within the scoped window.

## Caveats to repeat to the client
- CLIP-based matches and state transitions are **candidates**; the VLM
  description and your eyes on the boundary frames are the confirmation.
- Search recall is complete **within the scoped window**; broad unscoped searches
  see only the most recent slice — always scope.
- A "quiet" answer is trustworthy only when coverage shows no gaps; the system
  reports its own coverage so you can tell.

## Backlog: versioned industry scenario packs

Design and ship a versioned catalog of preconfigured AI Operator scenarios. A
pack must be selectable from both the operator UI and the agent deployment flow,
but must never be enabled automatically on arbitrary channels. Deployment should
ask for channel/zone scope and expose the proposed semantics, thresholds,
deduplication/cooldown, evidence path, and expected latency before approval.

Each scenario template should declare whether it is best served by a semantic
probe, a VLM alert, or a hybrid path; positive and hard-negative semantics;
recommended cadence and evidence window; default threshold/margin; bookmark and
incident policy; known false-positive conditions; minimum scene/camera
requirements; localization; and a schema/version identifier. Site-specific
calibration remains separate from the immutable catalog default.

### Army / Defense / Border Control

- Pirate boat detection.
- Small boat carrying many people.
- People climbing or forcing their way over a border barrier.

### Public Safety (Police / 911)

- Drifting detection.
- Motorcycle riding on one wheel.
- Traffic moving in the wrong direction.
- Multiple people riding the same e-scooter.
- E-scooter left in the road.
- Scooter riding on the pavement.
- Vehicle parked at a charging point without charging.
- Vehicle accelerating or speeding suddenly.
- Graffiti in progress.
- Dog attack.
- Car stopped on tram tracks.

### Industrial / Construction

- Falling container.
- Person falling from scaffolding.
- Work on scaffolding without required PPE.
- Fire and smoke.
- General accident.
- Leaking pipe.
- Floor leakage or pooling liquid.

### Acceptance questions for the design pass

- Where the catalog and its migrations live, and how an offline update adds or
  revises templates without overwriting site-calibrated instances.
- How one-click UI deployment and agent deployment produce the same inspectable
  draft and approval transaction.
- Which scenarios require zones, direction lines, speed calibration, temporal
  tracking, or external signals and therefore cannot honestly be represented by
  text prompts alone.
- How evaluation clips, false-positive/false-negative feedback, model/version
  compatibility, and rollback are attached to each catalog version.
