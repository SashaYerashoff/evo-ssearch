# GYM: Grammar-Driven Corpus Generator for tuktuk — Codex Handoff

## Who you are in this project
You are Codex (Sol), the bulk executor. Work in
`/home/sasha/Projects/rabbithole/knocknock` — you are turning this repo into
the **gym**: the corpus generator + (later) training workshop for **tuktuk**,
a small distilled tool-orchestration reflex model for EVA AI. Design authority
lives elsewhere: Sasha + Claude in the EVA repo (`evo-ssearch`, branch
`research/tuktuk`). You build; you do not redesign.

## Inputs (read all before writing code)
From `evo-ssearch` branch `research/tuktuk`, directory `docs/tuktuk/`:
- `grammar.yaml` — **the primary artifact.** Blocks, gates, devalidations
  (with `owner: model|harness`), 12 scenario compositions. The corpus is a
  function of this file.
- `grammar.md` — how to read the yaml + the corpus generation contract (§4)
  + the coverage matrix you must reproduce.
- `homeostate.yaml` — schema of the homeostatic context document (third
  encoder segment), deterministic selection rules s1–s6, render budget.
- `homeostatic_doc.md` — §6 defines your interface: synthesize documents from
  the schema; state variation is a measured corpus dimension.
- `tool_surface.json` — all 28 tool schemas + per-tool
  `compact_result_keys` (the ONLY result vocabulary you may emit) +
  permissions/exposure metadata.
- `tool_inventory.md` — §4 normalized signatures (you render every example
  against BOTH legacy and normalized signature sets), §4a `ui_*` family
  (not yet real: emit `answer_text` + `show_intent` annotation instead).
Also from `evo-ssearch` root: `skills/*/SKILL.md` — trigger phrases and
runbook wording are your paraphrase seeds and junk-track anchors.

## Architecture of the generator
1. **Imports layer** (`imports/` + sync script): copies the files above from
   a *pinned* evo-ssearch commit; the manifest records the source commit hash
   and per-file hashes. Imports are read-only — never hand-edited, never
   "fixed". If an import looks wrong or incomplete, write it to
   `QUESTIONS.md` and continue with what is derivable; do not guess silently
   and do not patch the grammar.
2. **Chain sampler**: for each scenario, sample compositions within the
   template (optional blocks, enum params, channel/time surface forms).
   Param values obey the three-source law: extract (from the generated query
   text), scratchpad (from a prior simulated result), enum. **The generator
   must never produce an example where the model would have to compute a
   value** — that includes epoch timestamps: queries carry symbolic time
   («вчера вечером», "last night", "2026-07-18 01:30–03:00").
3. **Result synthesizer**: simulated compacted tool results using ONLY keys
   from `compact_result_keys`. This is where devalidation symptoms live
   (`total_in_window`, `count`, `coverage`, `safe_to_apply`,
   `deferred_channel_ids`, `semantic_pending_count`, ...). Fidelity of these
   fields matters more than realism of anything else. No images, text only.
4. **Homeostatic doc synthesizer**: sample site states per `homeostate.yaml`
   (profiles: quiet / noisy / degraded-runtime / cold-start-warmup /
   backlogged-archive), apply selection rules s1–s6 deterministically,
   render with the budget discipline.
5. **Emitter**: JSONL, one row per **step**, chains linked by id (see record
   schema). Steps, not whole chains, are the training unit — tuktuk's two
   modes (single-shot expert; chainer given query+scratchpad+last result) are
   both step predictions.

## Record schema (proposal — flag concerns in QUESTIONS.md, then follow it)
```json
{
  "example_id": "…", "chain_id": "…", "step_index": 0,
  "seed": 12345, "scenario": "video_event_check",
  "stratum": "positive | mutation | shortcut | junk",
  "lang": "ru | en | mixed",
  "signatures_variant": "legacy | normalized",
  "input": {
    "query": "…",
    "scratchpad": "compact digest of prior steps ('' at step 0)",
    "last_result": {"…": "compacted result or null"},
    "homeostatic_doc": {"…": "rendered per homeostate.yaml"},
    "tools": ["…subset visible this turn…"]
  },
  "target": {
    "kind": "call | terminal",
    "call": {"tool": "…", "args": {}},
    "terminal": {"kind": "answer_text", "claim_level": "candidate",
                 "show_intent": null},
    "correction_of": "d02 | null"
  },
  "confidence_label": "in_pattern | not_my_pattern",
  "provenance": {"grammar_version": "0.1.0", "imports_manifest": "sha…"}
}
```

## Strata (from grammar.md §4 — all six are mandatory)
1. **Positive chains** — high volume at chain length 1–2, decaying tail to
   full compositions.
2. **Mutation matrix** — valid chain, one devalidation injected into one
   simulated result, target = the correction from `grammar.yaml`
   `devalidations`. Coverage target = the scenarios × devalidations matrix in
   `grammar.md` §3, **measured and reported**. Rows with `owner: harness`
   (d09) appear only as *state* (continuation fields present in results),
   never as model targets. The d01/d02 minimal pairs (`grammar.md` §2) get
   dedicated oversampling.
3. **Homeostatic shortcut** — the doc predicts d03/d07/d10 before any call;
   target skips the doomed call (straight to no-coverage TERM, or source
   switch). This stratum is the payoff of interoception; do not skimp.
4. **Junk** — noise, smalltalk, cat-on-keyboard, out-of-grammar requests →
   `terminal: noop | escalate`, `confidence_label: not_my_pattern`. Seed from
   SKILL.md pilot artifacts («почтальон», «когда наблевали», magic phrases)
   and from adversarial fuzz in knocknock idiom.
5. **Signature variability** — every chain rendered against legacy AND
   normalized signatures; light paraphrases of tool descriptions.
6. **Paraphrase storms** — RU / EN / mixed, sloppy/colloquial included;
   trigger phrases as anchors. **All paraphrases of one base chain carry the
   same group key and must land in the same split** (leakage rule).

## House rules (knocknock idiom — keep them)
- Seeded everything; byte-identical output for identical (seed, imports).
- Manifest: generator source hash, git commit, imports hashes, strata counts,
  split counts, coverage matrix, leakage checks.
- Splits: group-aware (by chain_id and paraphrase group), scenario-stratified.
- Tests: determinism, coverage completeness (every devalidation × scenario
  cell from the matrix present), leakage, schema validation of every row,
  three-source law (no target arg value absent from query/scratchpad/enums).
- CLI in knocknock style: a `corpus-tuktuk` subcommand (or successor name)
  alongside existing commands; dashboard panel wiring can wait.
- English for code and artifacts.

## Do NOT
- Do not train models (that phase starts after corpus statistics are
  reviewed).
- Do not invent tools, blocks, gates, devalidations, or scenario steps not in
  the imports. Gaps → `QUESTIONS.md`.
- Do not touch the evo-ssearch repo.
- Do not use real site data — this corpus is 100% synthetic by design; real
  office-machine data arrives separately as validation only.

## Definition of done (first iteration)
`python -m knocknock <corpus-cmd> --seed N --out runs/tuktuk_corpus.jsonl`
produces: corpus + vocab + manifest + coverage report; tests green; a
`REPORT.md` with strata counts, coverage matrix as generated vs. required,
open questions. Sasha reviews the report before any scale-up run.
