# Demo Runbook

Audience: the operator running a live research demo for the client. Goal: show
that EVA AI can find and explain real public-order incidents from urban street
and square cameras (fights, dangerous driving, crowd escalation, weapons,
fire/smoke, vandalism — anything an honest patrol officer would care about).

Terminology: see [glossary](../00_CANON/glossary.md). Invariants:
[facts](../00_CANON/facts.md).

`[FIELD]` = fill in before the demo from the site.

## 0. Field inputs to fill before the demo

- Demo channels (street/square, in front of houses): `[FIELD] channel IDs + names`
- Luxriot Evo host / EVA URL: `[FIELD]`
- Data window available: `[FIELD]` (how many days of collected video-descriptions)
- Operator account used for the demo: `[FIELD]`

## 1. Pre-demo checklist (run 30+ min before)

1. `GET /health` → ok; `GET /ready` → all components ready.
2. **Video-descriptions are running**: ask the agent *"Which channels are
   producing video-descriptions right now, and how many alerts in the last 6
   hours?"* — expect a list with `running=true` and per-channel alert counts.
3. **Coverage is honest**: confirm no large coverage gaps / dropped-batch badges
   on the demo channels (stream health panel). A quiet channel must be *quiet*,
   not *blind*.
4. **Watch-list probes** for the public-order set are cast on the demo channels
   (fighting, vehicle burnout/drift, crowd brawl, fire/smoke, person on ground)
   — this is the VLM-independent safety net for the demo. If archive coverage
   exists, mark each probe set as calibrated from archive P/N/M or explicitly
   uncalibrated.
5. There is at least one real or seeded incident in the window. If the live
   window is thin, pick a window that contains known activity (see §4 fallback).

## 2. Framing (say this out loud)

> "EVA watches every channel continuously and writes a description of what it
> sees. When it sees something a patrol officer would care about, it raises a
> structured alert and can drop a bookmark into Luxriot Evo. You can then ask it,
> in plain language, what happened — and it answers with evidence frames and tells
> you exactly what period and channels it actually inspected."

Emphasize: **video-descriptions are the always-on perception**; semantic search
and probes are how the agent digs through large volumes on request.

## 3. Demo flow

Run these in order. Each step names the expected behavior so you can tell if it
landed.

1. **Live perception.** Open the Video tab on a busy street channel. Show the L0
   live video-description ticking and the VLM feed. *"This is the system
   describing the street in real time."*

2. **"What happened" over a period.** Ask the agent:
   > *"What happened on channel `[FIELD]` in the last 2 hours?"*
   Expect: read-only rollup (L2 as map → L1 candidate windows), a concise
   timeline, evidence thumbnails, and a coverage line (period inspected vs.
   returned).

3. **Cross-channel alert report.** Ask:
   > *"Across all active channels, were there any public-order or safety alerts
   > today? List the channels with the most."*
   Expect: channels sorted by alert total, severities, and which are
   active/inactive. This is the video-description-first report.

4. **Semantic archive search (the "magic" moment).** Ask:
   > *"Search the archive for a vehicle doing a burnout or drifting."* / *"…for
   > people fighting."*
   Expect: ranked frame matches from `vlm_summary`/`vlm_alert` frames, newest
   first within the scoped window. Click into a result → preview → *describe
   frame* for a fresh VLM read.

5. **State-change question.** Ask:
   > *"At the square on channel `[FIELD]`, how many times did a crowd gather and
   > then disperse in the last 6 hours?"*
   Expect: `track_visual_state_transitions` — candidate transitions with boundary
   frame evidence, explicitly labelled as CLIP candidates (not ground truth),
   with a "describe these frames to confirm" prompt.

6. **Evidence → Luxriot.** Show a bookmark created from an alert appearing in
   Luxriot Evo on the channel timeline. *"The officer sees this in the system
   they already use."*

7. **Honesty close.** Point at the coverage contract in an answer: *"It tells you
   what it inspected and what it didn't — it won't pretend the street was quiet
   if it actually lost coverage."* This is a trust differentiator for a
   government buyer.

## 4. Fallback if a window looks thin

- Re-scope to a window/channel known to contain activity (`[FIELD]`).
- Use a **watch-list probe** hit as the entry point ("show me the burnout the
  probe flagged"), then confirm with *describe frame*.
- Worst case, *describe the current live frame* on a busy channel to show the
  perception quality, then run the "what happened" query on a longer window.

## 5. Do NOT

- Do **not** promise real-time push alerting SLAs, PDF/CSV/email export, or
  background report queues — the system does not do these (see
  [agent_capabilities](../operator/agent_capabilities.md) limits).
- Do **not** present CLIP state-transition counts as ground truth — always frame
  them as candidates confirmed by the boundary frames.
- Do **not** ask a broad "search the last two weeks" without a channel/time scope
  — recall is bounded by the candidate window; scope it.
- Do **not** claim a bookmark reached Luxriot unless you can see it there.

## 6. If something breaks

- Agent slow/timeouts: the VLM hosts may be saturated; reduce concurrent demo
  queries, prefer read-only rollup questions over fresh `describe_frame` bursts.
- "Nothing found": widen the time window, confirm the channel had coverage in
  that window (stream health), retry scoped.
- Service issue: see `install/field_rollout_demo.md` recovery + `scripts/rollback.sh`.
