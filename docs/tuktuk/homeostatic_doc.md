# Homeostatic Context Document — Spec (Phase 1, Task 3)

Status: v0.1.0 draft. Schema and rules are data: [`homeostate.yaml`](homeostate.yaml).
Companions: [`grammar.md`](grammar.md) (references this as the third encoder
segment), [`tool_inventory.md`](tool_inventory.md).

## 1. Framing (ИОС mapping, do not re-litigate)

probe = receptor · homeostatic doc = interoception · tuktuk = reflex arc ·
agent = escalation target. The document is what the reflex *feels* at decision
time. Two laws from the brief, made operational here:

1. **Context in, gain out.** Measured state (baselines, modes, backlogs,
   freshness) goes into the document. Policies and thresholds that *act* on
   state (capture decider's `activity_x >= 3.0`, escalation cutoffs, admission
   capacity rules) stay in the harness and never appear as model inputs —
   the model sees "activity is 4.2× this channel's norm", never "threshold
   is 3.0".
2. **Printability.** The harness records the exact rendered document plus the
   selection-rule ids that produced it (`decisions.jsonl`). Any decision is
   reconstructable byte-for-byte; this same log is the future shadow-mode
   label source for the confidence head.

## 2. Grounding — every field has a real source in EVA today

| Document field | Source (verified on branch, 2026-07-19) |
|---|---|
| `system.lm_load` | `lm_admission` stats: queued, by-workload, wait ages |
| `system.runtime` | composite channel inventory (`active_runtime_streams`, `runtime_problem_channels`) |
| `system.archive` | `semantic_pending/failed` counts, restore worker status |
| `channels[].motion` | capture apex decider: `capture_activity_baselines` {level, warmup}, `capture_attention` modes/`activity_x` |
| `channels[].bursts_24h` | attention burst history (`list_attention_bursts` backend) |
| `channels[].probes` | probes store: enabled, `hit_count_24h`, latest hit |
| `channels[].alerts_24h` | VLM alert event records (archive, `source=vlm_alert`) |
| `channels[].coverage` | rollup freshness from summary state + coverage objects |
| `channels[].neighbors` | **new artifact** `channel_topology.yaml` — does not exist yet; absence degrades gracefully (empty neighbor lists) |

The renderer is assembly-only: reads existing state, computes nothing new.
That makes it a deterministic harness component (~pure function of state +
query), testable with fixtures, knocknock-style.

## 3. Source selection is retrieval, not attention

Which channels enter the document is decided by rules s1–s6 (see yaml), keyed
off the query/trigger — *before* the model runs. Deliberate consequence: the
model never learns "which channels to feel"; it learns "given these feelings,
what to do". Rule s5 matters most: an unscoped research query gets **no**
per-channel docs — pre-feeding a guessed subset would leak a source-selection
decision into the model that the grammar assigns to block `C`.

## 4. Encoder rendering

Compact JSON, sorted keys, fixed key vocabulary (the ~40 keys in the schema
are vocab tokens in the tuktuk tokenizer — a design input for Phase 3's small
domain vocab). Budget enforced at render time (`budget` section): 8 channels,
~2000 chars target; dropped channels recorded in `meta`. Numeric values are
rounded (`activity_x` to 0.1, ages to coarse buckets at render time) — the
reflex needs magnitudes, not precision, and coarse buckets compress the vocab.

## 5. Collection plan (office machine, starts 2026-07-20)

Prereqs on the office demo machine, in order:
1. Deploy the current build **including the intent-gate fix** (`f8754e9`) —
   traces collected against the broken gate would poison the routing labels.
2. Drop a hand-sketched `channel_topology.yaml` (Sasha, at update time).
3. Enable the two logs (both JSONL, daily rotation, local disk):
   - `decisions.jsonl` — per decision: rendered doc + chain + terminal +
     operator followup (schema in yaml `decision_log`).
   - `homeostate_snapshots.jsonl` — the system+channels state every 5 min
     regardless of activity: this is the junk-track baseline (Task 4) and the
     drift record for confidence calibration.
4. Daily "отжим" to the gym: copy both files + record the EVA commit hash and
   `tool_surface.json` hash into a manifest per day. The gym treats them as
   **validation/calibration data only** — never training (data philosophy:
   corpus is synthetic-from-grammar; real traffic calibrates and evaluates).

The dev machine (laptop Evo) is explicitly out of collection scope — its
traffic is developer noise; useful only as extra junk-track color if ever
labeled, never as behavioral baseline.

## 6. Interface to the grammar and the corpus generator

- The corpus generator (gym) synthesizes homeostatic documents from this
  schema: for each scenario it samples plausible system/channel states
  (quiet site, noisy site, degraded runtime, cold-start warmup, backlogged
  archive...) and the selection rules pick the channel subset — so state
  variation is a *corpus dimension*, measured like block coverage.
- Devalidations d03 (no coverage), d07 (runtime problems), d10 (semantic
  pending) become *predictable from the document before the first call* —
  the generator must include chains where the reflex reads the document and
  skips a doomed call (e.g. goes straight to "no VLM coverage" TERM, or picks
  the archive source because rollup freshness is stale). That skill —
  interoception shortening the chain — is the homeostatic payoff and needs
  explicit corpus stratum.
- Confidence head: trained later on shadow-mode disagreements from
  `decisions.jsonl` (`operator_followup` + agent-vs-reflex divergence);
  synthetic floor starts with the `junk` scenario per grammar.md §4.

## 7. Open items

- Topology artifact format is minimal (`neighbors`, `zone`); enough for s2/s3.
  Zones may later drive group scenarios ("check the parking zone") — out of
  v0.1 scope.
- `operator_followup` labeling needs a tiny UI affordance eventually
  (accepted/corrected); until then it is derived heuristically from the next
  operator turn — mark derived labels as such in the log.
- Snapshot cadence (5 min) and rounding buckets are first guesses; revisit
  after the first week of office data.
