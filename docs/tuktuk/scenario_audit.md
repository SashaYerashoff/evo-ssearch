# EVA Scenario Audit (tuktuk Phase 1, Task 2 — part 1: audit)

Status: audit findings + draft block decomposition. The full grammar
(`grammar.md` + blocks-as-data) follows once the block set below is agreed.
Generated 2026-07-19 from branch `feature/secure-50-channel-foundation`
(+ main-branch React UI for the front-end surface).

## 1. Where scenarios actually live

| Surface | Location | Content |
|---|---|---|
| Runbooks ("skills") | `skills/*/SKILL.md` — 9 files, 387 lines | Goal / Trigger phrases / Default order / Output / Trust hierarchy / Rules |
| Skill CRUD | React UI AgentPanel → `agentApi.createSkill/updateSkill`; served from the same directory | operators can author new runbooks at runtime |
| Quick buttons | `AgentPanel.tsx SUGGESTIONS` (4 canned queries) | plain prompts, no skill binding |
| Skill button | UI `runSkill` → prepends `Use playbook "<slug>"` to the operator text | explicit activation path |
| Auto-activation | `_extract_requested_skill_slugs`: substring match of each skill's Trigger phrases against the user text | implicit activation path |
| Injection | index (slug + first line) always in system prompt; activated skill docs injected in full (budget-capped) | `_format_runtime_skill_index_for_prompt`, `_format_active_skill_docs_for_prompt` |

The 9 runbooks: `archive_research`, `cross_channel_correlation`,
`multi_channel_event_sweep`, `probe_tuning`, `prompt_tuning`,
`protocol_deploy`, `video_event_check`, `video_incident_timeline`,
`video_summary_review`.

Quality surprise: these are NOT "просто закинуто" — they are disciplined,
with consistent trust hierarchies and anti-overclaim rules. The mess is not
in the runbooks; it is in the **routing** around them.

## 2. Finding A — three classifiers, zero coordination (severity: high, live product bug)

Three independent mechanisms decide what happens on a turn:

1. **UI button** → prepends `Use playbook "slug"` (operator-chosen).
2. **Trigger-phrase substring match** → injects skill doc (implicit).
3. **Regex intent gate** (`_classify_tool_intents` → `_TOOL_INTENT_GROUPS`) →
   decides which tool schemas the model may call.

(2) and (3) run on the same text but share no vocabulary. Empirical check
(scratchpad `intent_gap_test.py`, regexes copied verbatim; canonical queries
built from the runbooks' own trigger phrases):

| Query (canonical for its runbook) | Skill activated | Intents | Tools missing vs runbook prescription |
|---|---|---|---|
| «проверь канал 115, был ли почтальон вчера вечером?» | video_event_check (+ video_summary_review — «был ли» is in both) | **∅ → zero tools** | all 6 prescribed |
| "check channel 12: was there a delivery van this morning?" | video_event_check | **∅ → zero tools** | all 6 |
| «что происходило ночью на складе?» | video_summary_review | **∅** («происходило» ≠ regex «что\s+произош») | all 7 |
| «опиши происшествия за сегодня последовательно» | video_incident_timeline | **∅** («происшестви» not in regex) | all 5 |
| «проверь по всем каналам, где появлялась белая газель» | multi_channel_event_sweep | **∅** | all 5 |
| «сравни канал 3 и канал 7 — та же машина?» | cross_channel_correlation | **∅** | all 4 |
| «найди в архиве человека в красной куртке» | archive_research | archive_research | — (OK) |
| «проба слишком шумная, затюнь пороги» | probe_tuning | probe_management | get_video_summaries, build_research_batch |
| «настрой промпт описаний для канала 115» | prompt_tuning | prompt_policy | get_video_summaries |
| "Protocol: Deploy" (either form) | protocol_deploy | **∅** | all 5 — `survey_channels`/`deploy_summary` are in **no** intent group at all |

**8 of 9 runbooks prescribe tools the gate withholds on their own canonical
triggers.** In the worst (and common) case the model receives a full runbook
saying "call get_video_summaries, then describe_frame..." and an empty tools
array. `protocol_deploy` cannot run from chat at all.

Aggravator: skill activation matches RU trigger phrases fine (they are
substrings), but the intent gate's RU regexes are much narrower than its EN
ones — so the failure is systematically worse in Russian.

Why this matters beyond the bug: this is the strongest empirical argument for
tuktuk's architecture. Scenario selection, tool exposure, and call assembly
are one decision and must be made by one component reading one grammar.
The corpus inherits exactly this: (query features → scenario → block chain →
calls) as a single supervised object.

**Interim fix — implemented 2026-07-19** (decision: Sasha, same day). An
activated runbook force-adds the tools it names to the exposed set:
`_skill_tool_names()` scans the activated SKILL.md content for registered tool
names (works for operator-authored runbooks too); the chat turn stores them in
`turn_tool_context["skill_tool_names"]`; `_select_relevant_tool_schemas` unions
them into `allowed_names` *before* the video-research inventory clamp. Safety:
widening happens strictly within the already permission-filtered schema list,
so a runbook can never escape the security envelope. Three runbooks
(`multi_channel_event_sweep`, `cross_channel_correlation`,
`video_incident_timeline`) described steps in prose without naming tools and
were edited to name them — a runbook that names its tools is also a better
instruction for the big agent. Verified by replaying every failing query above
(all now expose their prescribed sets; plain chat still exposes zero) and by
`tests/test_agent_tool_loop.py` (22 passed, incl. 2 new cases).

Known accepted quirk: a tool mentioned in a *prohibition* rule ("Do not run
`calibrate_probe_from_archive` during a normal sweep unless...") is also
exposed. Harmless — permission- and preview-gated, and the injected rule
itself forbids the call — but a `Tools:`/`Forbidden:` typed section in
SKILL.md would clean this up; deferred to the grammar work.

## 3. Finding B — runbooks are block compositions + repeated invariants

Duplication counts across the 9 files:

| Invariant (verbatim or near-verbatim) | Repeats |
|---|---|
| "Normalize the time window first" | 8 |
| "Do not say 'confirmed visually' unless `describe_frame` ran this turn" | 6 |
| Channel chunking + checked/unchecked ledger + confirm-if-over-limit | 5 |
| Trust hierarchy (routine memory < rollups < structured events < frames+describe) | 4 (near-identical) |
| "Do not auto-calibrate probes unless explicitly asked" | 3 |
| "P/N/M is triage/attention, not proof" | 6 |
| preview=true → UI Apply → trust only the receipt | 3 |
| No accusation / no hidden-state inference; rephrase to visible evidence | 4 |

These are **global invariants**, not scenario steps. In the grammar they
become: (a) harness-enforced gates (e.g. the `describe_frame`-before-
"visually confirmed" rule is checkable mechanically), (b) properties of the
block definitions themselves, (c) system-prompt residue for the big agent
only. The per-scenario residue after factoring out invariants is small: a
composition string plus a handful of parameters — which is precisely the
block-grammar hypothesis, confirmed on real material.

## 4. Finding C — trigger-phrase pathology

- **Overlaps:** «был ли» activates two runbooks simultaneously (both get
  injected, wasting budget); «визуальные доказательства» vs «предоставь
  доказательства» split across two runbooks arbitrarily.
- **Substring false positives:** matching is bare `in lower`: «то же»
  fires inside «в то же время», "compare" inside "compared to what you said
  earlier". No word boundaries.
- **Pilot artifacts as routing signals:** «почтальон», «когда наблевали»,
  «центральная площадь», magic phrase «потому что мама так сказала!». These
  are demo/site-specific hacks living inside general routing vocabulary.
  For the corpus they are actually valuable — as documented *junk-track*
  seeds and paraphrase-storm anchors — but they must leave the routing layer.
- **Two runbooks have no trigger phrases at all** (`archive_research`,
  `prompt_tuning`) — reachable only via explicit UI selection or lucky
  regex overlap.

## 5. Draft block set (for agreement before grammar.md)

Blocks (typed; each with parameters, entry condition, devalidation rules):

| Block | Meaning | Realization (current tools) | Key devalidations → correction |
|---|---|---|---|
| `W` window | resolve symbolic time window | `normalize_time_window` / harness | `total_in_window=0` → widen; operator override → re-resolve |
| `C` channels | resolve refs; inventory when unnamed | `list_channels`, `list_video_summary_channels` | unknown ref → clarify; count > per_turn_limit → chunk + confirm; runtime_problem_channels → mark coverage |
| `MAP` coarse read | cheap aggregate state | `get_video_summaries L2/L1`, `get_detection_summary`, `get_prompt_settings`, `list_probes` | no coverage → say no-coverage (≠ no-activity) or fall back to `RANK` over archive |
| `RANK` attention | rank windows/frames/channels | `get_visual_window_signals`, `list_attention_bursts`, `search_archive` (as ranking) | count=0 & total>0 → relax filter; total=0 → devalidate `W`; weak margin → rephrase queries |
| `DRILL` evidence | exact-time evidence retrieval | `get_video_summaries L1/L0 + evidence_frames`, `get_detections` | empty at exact window → widen locally or switch source; `semantic_pending` → wait/fallback |
| `AGG` quantify | counting / stratified stats | `count_video_summary_events`, `track_visual_state_transitions`, `build_research_batch` | `backend_truncated` → narrow window or accept sampled + report |
| `CONFIRM` visual gate | VLM look at specific frame | `describe_frame` | contradicts hypothesis → downgrade claim; unavailable → cap claim level |
| `CAL` calibrate | P/N/M calibration job | `calibrate_probe(+batch)` | `safe_to_apply=false` → rephrase or frame review; `deferred/requires_continue` → continue job (harness, not model — see tool_inventory §2 note) |
| `MUT` mutate | preview → approval → receipt | `create/update/delete probe`, `update_prompt_settings`, `restore_history` | no receipt → not applied; approval denied → stop |
| `TERM` terminal | close the signal | text answer, `ui_*` show (§4a of tool_inventory), `create_bookmark`, `deploy_summary` card, clarify-question, escalate-to-agent | claim level capped by which gates ran (CONFIRM etc.) |

Scenario decompositions (runbook → composition):

```
video_event_check        W · C(1) · MAP · [RANK] · DRILL · [CONFIRM] · TERM(verdict+ledger)
video_summary_review     W · C(named | inventory+confirm) · MAP(L2) · DRILL(L1→L0) · [RANK] · [CONFIRM] · TERM(report)
video_incident_timeline  W · C · MAP(L2) · DRILL(L1, 2-3 windows) · [CONFIRM] · TERM(chronology)
multi_channel_sweep      W · C(group, chunked) · ∀ch[ MAP · RANK · DRILL ] · [CONFIRM] · TERM(ledger + next chunk)
cross_channel_corr       W · C(set) · ∀ch[ MAP · DRILL ] · RANK(weak) · CONFIRM(each side) · TERM(correlation table)
archive_research         W · [C] · RANK(search as retrieval) · [AGG] · [CONFIRM] · TERM
probe_tuning             MAP(det_summary) · CAL(job) · [DRILL·CONFIRM] · [AGG(3 horizons)] · MUT(probe) · TERM(receipt-wait)
prompt_tuning            C · MAP(prompts + summaries) · diagnose · MUT(prompts) · TERM(receipt-wait)
protocol_deploy          C(inventory, chunked) · SURVEY · TERM(propose) · [MUT(prompts)] · [CAL·MUT(probes)] · TERM(deploy_summary)
```

Residual scenario-specific bits that resist the 10 blocks: `SURVEY`
(deployment live-look, = MAP via VLM on live streams) and prompt_tuning's
`diagnose` step (classify which prompt surface is at fault — candidate for a
typed enum output, not free reasoning). Both are contained, neither breaks
the model.

Cross-check against the corpus plan: every devalidation in the table above is
observable in current tool results (see tool_inventory §6) — so mutation-
matrix generation (valid chain → mutate one block → pair with correction) is
groundable in real result shapes for all 10 blocks.

## 6. Decisions (Sasha, 2026-07-19)

1. **CONFIRM is a gate, not a block** — it never changes what is retrieved,
   only the claim level TERM may emit. The grammar models it as a predicate on
   TERM ("visually-confirmed claims require describe_frame on the cited frame
   in this turn"), alongside the other claim-level caps from the trust
   hierarchies. Block chain shrinks to:
   `W · C · MAP · RANK · DRILL · AGG · CAL · MUT · TERM` + gates.
2. **Finding A fixed now** — see §2, implemented same day.
3. **Chunk-loop split confirmed** — mechanical continuation
   (`requires_continue`, deferred channels, `job_id` resume) is deterministic
   harness; tuktuk makes only per-item decisions (accept/reject calibration,
   candidate vs no candidate, escalate). The corpus must not contain trivial
   "continue" examples.
