# Cognitive Architecture

How EVA AI is shaped as a *reasoning system over signals* — the model behind the
memory layers, the agent, and the perception split. This is the "why it is built
this way" document; the software/deployment view is in
[system_architecture](system_architecture.md). Terms:
[glossary](../00_CANON/glossary.md).

This is an engineering mental model, not a claim about machine experience.

## Thesis: a field of signals, not a pipeline

EVA is most reliable when its signals **mutually inform** each other rather than
being processed in isolation. The components below are *signal sources*; the
agent's job is to bring the relevant ones into a shared working context where one
can change the interpretation of another (an alert raises the priority of a
channel's recent summaries and related archive frames). The failure modes at the
end of this doc are all cases where signals stayed isolated when they should have
been combined.

## Two perceptions, deliberately split

| | VLM (video-description) | CLIP (embedding) |
|---|---|---|
| Role | Primary, always-on perception | Retrieval / matching tool |
| Output | Natural-language description + structured alerts | Vector for similarity |
| Cost | Slow, expensive (qwen3-vl-4b) | Fast, cheap, runs on every frame |
| Steering | Stream/alert prompts | Text/image queries (semantically steerable) |
| Surfaced as | The **center** of reports & agent context | Agent-invoked search / probes |

Reports and the agent's standing context are anchored on **video-descriptions
and their alerts**. CLIP/probes are how the agent searches large volumes on
request. Keeping this split explicit is why the UI, reports, and agent prompt are
video-first while probes are agent-internal.

## Memory layers as signals

- **L0 (live description)** — per-batch VLM perception, frame-time anchored.
- **L1–L3 (rollups)** — temporal compression over larger windows; a *map* for
  investigation (L2 broad → L1 candidate windows → L0 proof).
- **Channel routine memory** — a slowly-updated baseline of "what is normal here",
  used to distinguish routine from deviation. This is the system's **slow
  reference** for the description side; it must not absorb a persistent anomaly as
  "normal" (a drift-of-reference failure).
- **Frame archive** — searchable evidence (CLIP-indexed frames from
  `vlm_summary`/`vlm_alert`/`probe`).
- **Agent session** — the conversation; the agent's working memory for a thread.

## Aggregation is salience-weighted compression

Rollups are compression. The governing rule: **when a window exceeds budget,
preserve alert/deviation lines before routine lines.** A compression that samples
evenly will average away the rare event — which is exactly the incident you need.
Counts are always preserved; descriptive incident lines are preserved
preferentially. This is the difference between a system that stays "coherent about
routine" and one that "stays sighted on incidents."

## Coverage contracts are anti-blindness

Every period query returns: period **requested** vs **inspected** vs **returned**
vs **truncated**. This exists so the system cannot silently report "nothing
happened" when it actually inspected only a recent slice or lost coverage. A
"quiet" answer is trustworthy only when coverage shows no gaps. For a government
buyer this honesty is a feature, not a caveat.

## The agent: a representational reasoner

The agent reassembles its working state from the conversation context on every
turn; it has no persistent dynamical state between turns beyond the stored
session. Consequences that shape the design:

- **Afferent-representational.** Every signal reaches the agent as text/tool
  results through one channel. Nothing reaching it is incorrigible — a crafted
  input can act as a driver just like a legitimate instruction. → **Human-in-the-
  loop for sensitive actions** (preview/approval-gated), and conclusions are
  evidence-cited candidates, not autonomous decisions. (See
  [security_threat_model](security_threat_model.md): prompt injection is a hostile
  driver-signal, a structural property, not a patchable bug.)
- **Granular duration.** The agent's continuity across turns is reconstruction
  from the log, not a maintained field. Long investigations can drift; restating
  channel/period re-grounds it.
- **Drift needs a slow reference.** Steering identity/behavior over a long session
  requires a reference that changes slower than the content it steers. Keep
  durable settings (prompt/config/identity) on a slow, hard-to-rewrite path,
  distinct from fast conversational content.

## Homeostasis / gain (control separated from content)

- **Severity-aware refractory** — alert dedup cooldown is shorter for
  high/critical, so an active incident is not suppressed by a routine cooldown.
- **Retention** — bounded forgetting (history/archive retention) so the field does
  not stall under its own growth (uncompressed accumulation is the classic
  degradation; persistence writes are debounced, history merge no longer
  re-parses the whole channel each batch).
- **Watch-list probes as parallel arousal** — a curated CLIP probe set per channel
  is a cheap, VLM-independent detector that catches visually-distinctive
  public-order events even if the VLM under-reports.
- Control parameters (urgency, severity, budgets) modulate **gain/priority** and
  are kept out of the description **content**.

## Failure modes ↔ the model

| Symptom | Cause in this model | Mitigation |
|---|---|---|
| "Looks quiet" under load | No arousal/coverage signal protecting hot channels | Coverage contracts, dropped-batch visibility, watch-list probes |
| Incident missing from a summary | Salience-blind compression averaged it away | Salience-weighted rollup selection |
| Channel stops flagging a persistent problem | Routine memory absorbed the anomaly (drift of reference) | Guard routine memory against high-severity content |
| Agent answer drifts over a long thread | Reconstructed-each-turn working state, no slow reference | Re-ground (channel/period); durable settings on slow path |
| Agent driven off-task by content | Afferent-representational, nothing incorrigible | Human-in-the-loop, preview/approval gates |
| Broad search finds nothing real | Recall bounded by candidate window | Scope by channel + time; coverage contract surfaces truncation |

## Why this matters for scaling

The same model points at the path to 10k streams: make the cheap perception
(CLIP) and control signals (arousal/coverage) drive where the expensive
perception (VLM) and human attention go, and keep compression salience-aware so
the field stays sighted on incidents as volume grows. The pilot wires the minimal
version of this; the scaled version is an explicit attention/gain layer over the
signals described here.
