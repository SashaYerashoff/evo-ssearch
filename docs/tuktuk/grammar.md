# tuktuk Scenario Grammar — Spec (Phase 1, Task 2 — part 2)

Status: v0.1.0 draft for review. The grammar itself is **data**:
[`grammar.yaml`](grammar.yaml). This file explains how to read it and records
the coverage obligations. Companions: [`tool_inventory.md`](tool_inventory.md)
(tool surface + normalization), [`scenario_audit.md`](scenario_audit.md)
(where the grammar came from).

## 1. Objects

- **Block** — a typed step in a chain. A block instance = block id + params.
  Params come from exactly three sources: `extract` (copied from operator
  text), `scratchpad` (copied from a prior result), `enum` (chosen from a
  closed set). **A block never computes a value** — that is the extractability
  law from the project brief, enforced by construction.
- **Chain** — ordered block instances ending in `TERM`. Canonical research
  spine: `W · C · MAP · RANK · DRILL · AGG · CAL · MUT · TERM` (each optional
  per scenario; order fixed).
- **Gate** — a predicate over chain history that caps what `TERM` may claim
  (`g1`: "visually confirmed" needs `describe_frame` on the cited frame this
  turn; `g3`: "changed" needs a trusted receipt; …). Gates are not steps;
  they are checkable mechanically by the harness — decision: CONFIRM is a
  gate, not a block.
- **Devalidation** — observable symptom in a compacted tool result →
  implicated block → correction. The central training object: tuktuk's
  "думалка" mode is exactly (chain state + result + symptom) → corrected next
  call. `owner: harness` rows (mechanical continuation) are never model
  decisions and never corpus examples.
- **Scenario** — a composition template + characteristic gates + the set of
  devalidations it can realistically exercise. 9 scenarios inherit from
  runbooks; `status_glance` and `bookmark_capture` cover intent groups that
  had no runbook; `junk` is the validation-track scenario.

## 2. Why d01 vs d02 is the model's most important lesson

`total_in_window` and filtered `count` are both present in archive results
today, so "empty result" splits into two *distinguishable* symptoms:

- `total_in_window = 0` → the **window** is wrong (d01) → widen W.
- `count = 0, total_in_window > 0` → the **filter** is wrong (d02) → relax or
  rephrase the query; touching the window is a *wrong correction*.

The corpus must contain minimal pairs of these two, because they have
identical surface appearance ("nothing found") and different corrections —
this is the exact failure mode where a cheap model guesses and an expensive
model reasons. Here the signal is extractable, so the reflex can get it right.

## 3. Coverage matrix (scenarios × devalidations)

Generated from `grammar.yaml` (script: scratchpad; regenerate on every yaml
change — do not hand-edit). Every devalidation is exercised by ≥1 scenario;
`junk` intentionally exercises none.

| scenario | d01 | d02 | d03 | d04 | d05 | d06 | d07 | d08 | d09 | d10 | d11 | d12 | d13 | d14 | d15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| video_event_check | x | x | x | · | x | · | · | · | · | · | · | · | x | x | · |
| video_summary_review | x | · | x | x | · | x | x | · | · | x | · | · | · | · | · |
| video_incident_timeline | x | · | x | x | · | · | · | · | · | · | · | · | x | · | · |
| multi_channel_event_sweep | x | x | · | · | · | x | x | · | x | · | x | · | · | · | · |
| cross_channel_correlation | x | x | · | · | · | · | · | · | · | · | · | · | x | · | · |
| archive_research | x | x | · | · | · | · | · | · | · | x | · | x | · | · | · |
| probe_tuning | · | · | · | · | · | · | · | x | x | · | · | · | · | · | x |
| prompt_tuning | · | · | x | · | x | · | · | · | · | · | · | · | · | · | x |
| protocol_deploy | · | · | · | · | · | x | · | x | x | · | x | · | · | · | x |
| status_glance | · | · | · | · | · | · | x | · | · | · | · | · | · | · | · |
| bookmark_capture | x | · | · | · | x | · | · | · | · | · | · | · | · | · | · |
| junk | · | · | · | · | · | · | · | · | · | · | · | · | · | · | · |

(d09 appears in three scenarios but is harness-owned: those cells mean "the
scenario's *state* includes continuation, which the generator must simulate";
the *correction* is never a model output.)

## 4. Corpus generation contract (Phase 2 consumes this)

1. **Positive chains**: for each scenario, sample compositions within the
   template (optional blocks in/out, enum params, RU/EN/mixed query
   paraphrases, channel/time surface forms). Every sample is
   (query, homeostatic doc, signatures) → full chain with per-step expected
   calls.
2. **Mutation matrix** (knocknock): take a valid chain, inject one
   devalidation at one block (simulate the symptom in the tool result), pair
   with the corrected continuation from the `devalidations` table. Coverage
   target = the matrix above, measured, not vibed.
3. **Single-shot track**: chains of length 1–2 (the "secretary" mode) are the
   high-volume stratum; long chains are rarer but carry the correction
   lessons.
4. **Confidence labels**: `junk` scenario + deliberately out-of-grammar
   queries → expected head output "not my pattern". Shadow-mode disagreement
   labels come later from deployment; the synthetic floor starts here.
5. **Signature variability**: every example rendered against legacy AND
   normalized signatures (tool_inventory §4-5) with light description
   paraphrases — signatures ride in the encoder as data.
6. **Terminal split**: `answer_show` terminals become trainable only after
   the `ui_*` family exists (tool_inventory §4a); until then the generator
   emits `answer_text` with a `show_intent` annotation so the corpus can be
   re-terminated later without regeneration.

## 5. Known gaps / next

- Blocks `MAP(domain=live_survey)` (deployment) and prompt_tuning's
  `fault_surface` decision head are the two residuals that stretch the block
  set; both are typed and closed, watch them during corpus statistics.
- Trigger phrases are dead as routing (Finding C) but live as paraphrase-storm
  seeds and junk-track anchors — the generator should import them from
  `skills/*/SKILL.md` as data.
- Task 3 (homeostatic document spec) defines the third encoder segment; the
  grammar references it only as "homeostatic doc" for now.
- Coverage table regeneration should become a checked-in script once the
  tuktuk repo exists (Phase 2); until then the scratchpad script + this note
  is the contract.
