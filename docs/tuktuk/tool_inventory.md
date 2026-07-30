# EVA Agent Tool Surface — Inventory & Normalization (tuktuk Phase 1, Task 1)

Status: design doc, no live refactor. Generated 2026-07-19 from branch
`feature/secure-50-channel-foundation`.

Companion typed artifact: [`tool_surface.json`](tool_surface.json) — machine-readable
merge of `agent.py::_TOOL_SCHEMAS`, dispatch, intent groups, security policy
(`agent_security/eva_adapter.py`), and per-tool compacted result keys. Regenerate it
from source when the tool surface changes; this doc holds the judgments, the JSON
holds the facts.

---

## 1. How a tool call actually happens (chat surface)

The LM never sees the raw tool surface. The pipeline per turn:

1. **Security filter** — `EvaAgentToolAdapter.available_tool_schemas(ctx)`:
   drops tools the actor lacks permission for, hides `create_bookmark` until
   approval flow, hides global restore status from channel-scoped actors.
2. **Intent gate** — `_seed_turn_tool_context` + `_classify_tool_intents`
   (regex, RU/EN) map the operator text to intent groups
   (`_TOOL_INTENT_GROUPS`); `_select_relevant_tool_schemas` exposes only those
   groups' tools. Empty intent list ⇒ **zero tools** (deliberate: cheaper and
   safer than a 40k-token schema dump).
3. **Progressive disclosure** — `video_research` without a resolved channel
   exposes only `normalize_time_window` + `list_video_summary_channels` until an
   inventory result lands in the turn context.
4. **Deterministic arg injection** (harness, before dispatch): operator
   `relative_range` → summary tools; remembered `time_window` → `since_ms/until_ms`
   or `from_ts/to_ts` per tool family; research-continuation channel lists;
   `runtime_only` flag from turn context; channel mention in user text resolved
   via `_resolve_channel_id` into turn context.
5. **Gateway execution** — per-tool policy: preview-gating on write tools
   (apply requires an approval plan), channel scoping for scoped actors,
   timeouts (45s default, up to 300s), row caps.
6. **Result compaction** — `_compact_tool_result_for_model` reshapes every raw
   result before the LM sees it. **The compacted shape, not the handler return,
   is the ground truth for tuktuk's "last tool result" input.** Keys per tool
   are in `tool_surface.json → compact_result_keys`.
7. **Budgets** — `AGENT_MAX_TOOL_CALLS_PER_TURN`, context token budget with
   warning/hard-stop system messages.

Implication for tuktuk: stages 1–5 and 7 are exactly the "homeostasis-as-gain
stays outside" harness. The regex intent classifier (stage 2) is the component
tuktuk subsumes and generalizes; the arg-injection layer (stage 4) must NOT be
absorbed — tuktuk should emit symbolic args and let the harness keep resolving
them. The corpus must be generated against schemas *post* stage 1–3 filtering,
i.e. tuktuk should learn "which tools exist right now" from its encoder input,
not from weights.

## 2. Inventory summary

28 tools registered in `_TOOL_SCHEMAS`, 1:1 dispatch in `AgentTools.execute`.
Full signatures in `tool_surface.json`. Families:

| Family | Tools | Notes |
|---|---|---|
| Docs/help | `lookup_help` | exclusive intent: help queries expose only this |
| Time | `normalize_time_window` | deterministic, no I/O — really harness code shaped like a tool |
| Channel inventory | `list_channels`, `list_video_summary_channels` | second one is 3 tools in a trenchcoat (see F6) |
| Archive retrieval | `search_archive`, `get_detections`, `get_detection_summary`, `build_research_batch` | overlapping selection semantics (see F7) |
| Attention signals | `get_visual_window_signals`, `list_attention_bursts` | P/N/M scoring; explicit "signal, not proof" |
| Video summaries | `get_video_summaries`, `count_video_summary_events`, `track_visual_state_transitions` | analytics-grade, 12–22 params |
| Summary restore | `restore_video_summary_history`, `get_video_summary_restore_status` | write + status pair |
| Probe lifecycle | `list_probes`, `calibrate_probe_from_archive`, `prepare_probe_calibration_batch`, `create_probe`, `update_probe`, `delete_probes` | calibrate→create chain is the canonical scenario |
| VLM | `describe_frame`, `survey_channels` | describe_frame = escalation to expensive channel |
| Prompt policy | `get_prompt_settings`, `update_prompt_settings` | |
| Bookmarks | `create_bookmark` | hidden until approval flow |
| Reporting | `generate_report`, `deploy_summary` | |

Write tools (preview-gated): `create_probe`, `update_probe`, `delete_probes`,
`update_prompt_settings`, `restore_video_summary_history`; plus `create_bookmark`
(hidden-until-approval). All follow the same `preview: true` default → action-plan
→ approve → apply cycle. This is already uniform and should be kept verbatim in
the normalized surface.

**Chat-unreachable tools:** `survey_channels` and `deploy_summary` are in
dispatch and in the security policy but in **no intent group**, so the chat loop
never exposes their schemas today. **Decision (Sasha, 2026-07-19): keep them in
scope and design for variability.** Rationale: at rollout scale (target:
10 000 channels) stream survey + initial pre-configuration is a selling
capability, and it is reflex territory par excellence — a long, boring,
homogeneous chain (survey batch → calibrate → create probes → record summary)
where the expensive agent must not be the bottleneck. Add a `deployment` intent
group; the deployment loop becomes a first-class corpus scenario family, with
batch continuation (`deferred_channel_ids` / `requires_continue`) as its core
devalidation pattern.

## 3. Critique of the factorization, as seen by a ~50M model

Criterion (from the project brief): tool name predictable from intent; arguments
extractable from query + scratchpad without computation. Findings ranked by
corpus impact.

### F1. Time-window Babel — the single biggest defect
Four incompatible conventions coexist, distributed inconsistently:

| Convention | Used by | Semantics |
|---|---|---|
| `since_hours` / `until_hours` | most tools | relative, hours, float |
| `since_ms` / `until_ms` | archive family | absolute epoch **milliseconds** |
| `from_ts` / `to_ts` | summary/calibration family | absolute epoch **seconds** — "milliseconds are accepted" (silent unit sniffing) |
| `relative_range` / `date`+`day_hint`+`start_time`+`end_time` | `normalize_time_window`, summary tools | symbolic/local |

A small model asked to emit epoch milliseconds must *compute* — direct violation
of the extractability criterion. The harness already knows this: it injects
`time_window` into prepared args per family, and `normalize_time_window` exists
precisely to move the computation out of the LM. But the schemas still advertise
all four conventions, so the current agent (and any corpus copied from its
traces) mixes them.

**Target:** every tool takes one optional typed `window` argument:
`{relative_range} | {date, start_time, end_time} | {from_iso, to_iso}` — all
symbolic, all extractable from operator text. Epoch resolution happens in the
harness (existing `normalize_time_window` logic becomes a pure library call).
Tuktuk never emits a number it didn't copy from input.

### F2. Channel addressing — three spellings, per-tool lottery
`channel_id` (int) vs `channel_ref` (string `"#115"`/`"115"`/title) vs
`channel_ids` (int[]). Some tools take all three (`calibrate_probe_from_archive`),
some only id+ref, some only id (`search_archive` — no ref!), `create_bookmark`
takes id+ref but not ids. The harness already resolves refs deterministically
(`_resolve_channel_id`, title matching). 

**Target:** one field, `channels: string[]` (refs — ids-as-strings are valid
refs), resolved in harness. Single-channel tools take `channel: string`.
Extraction becomes copy-paste from operator text; resolution failures become a
typed devalidation (`unknown_channel`) with a deterministic correction path
(re-ask or `list_channels`).

### F3. Alias pollution
`event_query`/`positive_query` and `contrast_query`/`negative_query` are
declared as aliases *in the same schema* (`calibrate_probe_from_archive`,
`prepare_probe_calibration_batch` items). `sort_by` means
`similarity|time` in `search_archive` but `newest|oldest` in `get_detections`.
`since_hours` defaults differ (24 vs 6) across sibling tools. Aliases double the
token patterns a small model must treat as equivalent — pure corpus tax with
zero expressive gain.

**Target:** one canonical name per concept: `positive_query`, `contrast_query`,
`sort_by: relevance|time_desc|time_asc`. (Note the existing
`ARCHIVE_SOURCE_ALIASES` table in agent.py — the codebase already treats alias
normalization as harness work; extend that pattern, don't feed aliases to the
model.)

### F4. One tool with a mode parameter: `search_archive`
`scope=indexed_folder` and `scope=detections` are different tools: different
required params (`folder` only for the first), different backends, different
result semantics, disjoint filter sets (probe_id/source/channel meaningless for
folders). The enum forces the model to learn conditional requiredness that the
schema cannot express.

**Target:** split into `search_folder(query, folder, …)` and
`search_frames(query, source?, channels?, window?, …)`. Name predicts intent
("search the archive" vs "search that indexed folder") — exactly the property
tuktuk needs.

### F5. Duplicate workflow pair: `calibrate_probe_from_archive` vs `prepare_probe_calibration_batch`
Same operation (P/N/M calibration over archived frames), two entry points —
single-shot vs stateful job. The batch tool's `items[]` schema re-declares the
whole single-tool schema *including its aliases*, plus `job_id` continuation.
Both already return the same continuation vocabulary
(`requires_continue`, `deferred_channel_ids`, `next_batch_hint`).

**Target:** one tool, `calibrate_probe`, with server-side batching as it already
exists; `job_id` continuation is the norm, first call creates the job. The
"continue the job" reflex (result says `requires_continue: true` → next call is
same tool with `job_id`) is the cleanest devalidation-correction training pair
in the whole surface — one tool makes it learnable, two makes it a coin flip.

### F6. Tools that do two-or-three things
- `list_video_summary_channels`: channel inventory + runtime stream health +
  coverage/backfill status. 34 compact result keys. The harness itself patches
  in `runtime_only=true` when the operator asked a runtime question — i.e. the
  harness already knows these are different intents. Split: `list_summary_channels`
  (inventory/coverage) and `get_runtime_status` (streams, models, queues, drops).
- `get_video_summaries`: rollup retrieval + evidence-frame selection + semantic
  indexing status. Evidence frames are an escalation artifact (feed to
  `describe_frame`); keep them behind `include_evidence_frames` but move
  semantic-status reporting to the homeostatic document (Task 3) — it is system
  interoception, not query result.
- `count_video_summary_events`: counting + timeline sampling + transition events
  + anchor filtering. Defensible as one analytic tool, but its `event_kind` +
  `entity_query` + `anchor_query` triple needs explicit grammar coverage — it is
  the least predictable-from-intent tool in the surface.

### F7. Overlapping retrieval family — selection must become grammar, not vibes
`search_frames` (semantic query) vs `get_detections` (enumerate by filters) vs
`build_research_batch` (stratified sampling for research) vs `get_video_summaries`
(rollup text) all answer "what happened". Today the big agent picks by taste.
For tuktuk this is THE source-selection block of the grammar; the decision
features are extractable: *has semantic phrase* → search; *wants
recent list/count* → get_detections/summary; *wants period comparison* →
research batch; *wants narrative* → summaries. Write these as deterministic
grammar rules; where features conflict, that's a legitimate low-confidence
escalation, not a guess.

### F8. Args that require computation (violations of the extractability rule)
- `create_probe.pos_floor/margin_thr`: computable only from calibration output.
  Legal in a chain (copy from `calibrate_probe` result in scratchpad) — the
  grammar must *forbid* the single-shot form without a calibration block.
- `track_visual_state_transitions` thresholds (`positive_floor`,
  `margin_threshold`, `min_state_duration_sec`, `merge_gap_sec`, 22 params
  total): defaults must be trusted; corpus should never emit them unless copied
  from operator text or a prior result.
- `update_probe.changes` / `update_prompt_settings.changes`: **untyped objects.**
  For the big agent this is flexible; for a distilled model it is an open-ended
  generation surface with no schema constraint. Target: enumerate permitted keys
  in the schema (they are finite and known: probe fields / `stream_system_prompt`,
  `alert_policy_prompt`, sampling fields).
- `deploy_summary`: arguments are a composed report (overview, notes,
  per-channel lines). Split by field: the structured fields (`mode`, `wipe`,
  `elapsed_sec`, `channels[]`, `probes[]`, `prompt_targets[]`) are extractable
  from the deployment job scratchpad — reflex-fillable; the narrative
  `overview`/`notes` are authorship — template-fill for tuktuk ("surveyed N
  channels, created M probes, K deferred"), free-form only via the big agent.
- `create_bookmark.title/description`: short authorship. Acceptable as a
  template slot ("<event phrase> on <channel>") but flag as a quality-risk
  head; low confidence here should escalate.

### F9. Mutual exclusivity the schema can't say
`describe_frame` requires exactly one of `channel_id`/`channel_ref` |
`image_path` | `detection_id` — schema says `required: []`. Same pattern in
`normalize_time_window` (`date` xor `day_hint`, `relative_range` overrides
both). JSON Schema `oneOf` is available and the corpus generator needs the
constraint explicitly anyway; encode it in the normalized signatures.

### F10. Result-shape inconsistencies that matter for chaining
Time echoes come back in mixed units too: `since_ms/until_ms` in archive
results, `time_window` objects in summary results, `from_local/to_local` +
`from_ts/to_ts` in `normalize_time_window`. Devalidation vocabulary is
near-uniform but not quite: `requires_continue` + `deferred_channel_ids` +
`next_batch_hint` (calibration, summaries) vs `requires_confirmation`
(channel listing) vs `coverage` objects vs bare `errors` arrays. Normalize the
*result envelope*: every tool result gets optional
`{window, coverage, continuation{required, hint, deferred}, errors[]}` blocks
with fixed names. Free win: `_compact_tool_result_for_model` is already the
single choke point where this envelope can be imposed without touching
handlers.

## 4. Normalized target surface (proposal)

Shared typed blocks (encoder segments, reused across all signatures):

```
Window   := {relative_range: str} | {date: str, start_time?: str, end_time?: str}
          | {from_iso: str, to_iso: str}                  # symbolic only, no epoch
Channels := {channel: str} | {channels: str[]}            # refs; harness resolves
Source   := probe | vlm_summary | vlm_alert               # unchanged, already canonical
Paging   := {limit?: int, offset?: int}
Preview  := {preview: bool = true}                        # write tools, unchanged
Envelope := {window?, coverage?, continuation?{required, hint, deferred_channels},
             errors?: str[]}                              # uniform result wrapper
```

Target signatures (26 tools; renames marked ←):

```
help          lookup_help(query, top_k?)
time          — removed as a model-visible tool; becomes harness resolution of Window
inventory     list_channels(force?)
              list_summary_channels(Channels?, Window?, depth?)      ← list_video_summary_channels (inventory half)
              get_runtime_status(Channels?)                          ← list_video_summary_channels (runtime half)
retrieval     search_frames(query, Source?, Channels?, Window?, sort_by?, mode?, Paging?)   ← search_archive scope=detections
              list_frames(Source?, Channels?, probe?, Window?, sort_by?, Paging?)           ← get_detections
              summarize_frames(Source?, Channels?, Window?)          ← get_detection_summary
              build_research_batch(probe?, Channels?, Window?, periods?, bands?, limits...)
signals       get_window_signals(positive_query, contrast_query?, Channels, Window?, sources?, limits...)  ← get_visual_window_signals
              list_attention_bursts(Channels?, Window?, min_activity_x?, limit?)
summaries     get_video_summaries(Channels, Window?, depth?, include_evidence_frames?, limits...)
              count_events(entity_query, anchor_query?, event_kind?, Channels, Window?, depth?, limits...) ← count_video_summary_events
              track_state_transitions(positive_state_query, ..., Channels, Window?, sources?)             ← track_visual_state_transitions (thresholds harness-defaulted)
restore       restore_summary_history(Channels?, Window?, levels?, Preview)   ← restore_video_summary_history
              get_restore_status()                                   ← get_video_summary_restore_status
probes        list_probes(Window?)
              calibrate_probe(positive_query, contrast_query?, Channels?, Window?, sources?, job_id?, limits...)
                                          ← merges calibrate_probe_from_archive + prepare_probe_calibration_batch
              create_probe(name, Channels, positives, negatives?, thresholds_from_calibration, severity?, bookmark...?, Preview)
              update_probe(probe, changes{typed fields}, Preview)
              delete_probes(probe_ids | all, Preview)
vlm           describe_frame(target: oneOf{channel | image_path | detection_id}, prompt?)
prompts       get_prompt_settings(channel?)
              update_prompt_settings(channel?, changes{stream_system_prompt?, alert_policy_prompt?, ...}, Preview)
bookmarks     create_bookmark(title, channel, Window?/timestamp?, description?, severity?)
reports       generate_report(report_type?, Channels?, Window?, include_probes?, top_events?)
deployment    survey_channels(Channels?, fast_mode?, duration_sec?, sample_count?, prompt?)
              deploy_summary(mode, structured fields..., overview_template?)   # see F8 split
dormant       index_folder(folder) + search_folder(query, folder, ...)
              # operator folder-search workflow is dead (Sasha, 2026-07-19), but
              # "point the embedder at a directory" stays as a utility capability;
              # keep out of tuktuk v1 corpus, keep the backend alive
```

Signature-variability requirement (Needle-style, from the brief): the corpus
generator should emit each training example against BOTH the legacy and the
normalized signature set (and light paraphrases of descriptions), since
signatures ride in the encoder as data. The legacy↔normalized mapping table
that the migration shim needs is therefore also a corpus asset — one artifact,
two consumers.

## 4a. Missing tool family: UI control (design-ahead, main-branch React UI)

The React console (main branch, Ivan) runs the agent in a half-screen panel;
the other half — operator working area with sections `archive | video |
monitoring` (`LeftRail.SectionId`), archive sub-tools
(`filters | search | text | image`), and monitoring actions. There is **no tool
surface for the agent to drive that half yet**, but the transport already
exists: `App.tsx` has one-way "agent → console mirroring" (`AgentDrive`), which
currently hard-routes every agent action to the archive screen.

Proposed family (thin, enumerable, zero server-side risk — ideal reflex
terminals; "show, don't tell" beats describing results in text):

```
ui            ui_navigate(section: archive|video|monitoring)
              ui_show_archive(filters?: {Source?, Channels?, Window?, query?})   # generalizes AgentDrive
              ui_open_channel(channel)                       # video screen, focus one stream
              ui_open_probe(probe)                           # monitoring, ProbeInspector
              ui_present_frames(detection_ids[] | image_urls[])   # evidence handoff to operator
```

Properties that make this family tuktuk-friendly: closed enums, all arguments
copied from scratchpad (detection ids, channel refs, probe names from prior
results), instant + reversible, no approval flow needed. It also gives the
grammar a **new terminal block**: `to-user` splits into `to-user:text` and
`to-user:show`, and half of "покажи..." queries terminate in a UI call instead
of a prose answer. Requires: extending `AgentDrive` from hard-coded
archive-routing to a typed command channel (front-end work — coordinate with
Ivan; the agent side is just new no-op-server tools emitting SSE events).

## 5. Migration notes (when we get there — not now)

1. **Shim, not rewrite.** A pure-function adapter normalized→legacy args in
   front of `AgentTools.execute` (mirror image of the alias tables that already
   exist: `ARCHIVE_SOURCE_ALIASES`, ref resolution, time-window injection).
   Legacy schemas stay live for the big agent until parity is proven.
2. **Envelope at the choke point.** Impose the uniform result envelope inside
   `_compact_tool_result_for_model` — zero handler changes.
3. **Schema `oneOf`** for describe_frame/window variants; typed `changes`
   objects for the two update tools — these are additive schema edits, safe
   before any rename.
4. Renames and the `search_archive` split land last, behind the shim.
5. `normalize_time_window` stays callable during migration; tuktuk's harness
   calls it as a library function, the big agent keeps the tool.

## 6. Bridge to Task 2 (block grammar)

The working block hypothesis maps cleanly onto the normalized surface:

| Block | Realization | Devalidation signals already emitted today |
|---|---|---|
| source selection | tool choice + `Source`/`sources` | `coverage` empty, `source_counts` zero, `search_errors`, `runtime_problem_channels` |
| time window | `Window` | `total_in_window: 0` (widen), `backend_truncated`/`truncated` (tighten or paginate), `coverage` gaps |
| filter | queries, probe/channel filters | zero results with nonzero window total (relax filter, not window — distinguishable today!), `semantic_pending_count` (wait/fallback) |
| aggregation | `depth`, `level_limit`, counts vs list vs summary | `deferred_count`, `requires_continue` (continue), `error_count` |
| terminal | to-user text / `describe_frame` escalation / `create_bookmark` / write-tool preview | `requires_confirmation`, approval flow, low-confidence head |

Note the third row: because archive results return both `total_in_window` and
filtered `count`/`returned`, the "empty result: window or filter?" ambiguity
from the brief is *resolvable from the result shape* — the corpus can teach the
distinction with real signals instead of synthetic labels. This is the
strongest argument for imposing the uniform envelope before corpus generation.

Decisions log (Sasha, 2026-07-19):
1. Deployment tools **in scope** — deployment (survey + pre-configuration at
   rollout scale, target 10 000 channels) is a selling capability and a core
   reflex scenario family. See §2 and the `deployment` family in §4.
2. `count_video_summary_events` — **keep as one tool** (delegated decision).
   Splitting counting from timeline/transitions triples source-selection
   branching for marginal gain; instead the grammar covers the
   `event_kind`/`entity_query`/`anchor_query` triple explicitly.
3. Folder search as an operator workflow is **dead**; the "point the embedder
   at a directory" capability is kept as a dormant utility pair
   (`index_folder`/`search_folder`), out of tuktuk v1 corpus.
4. New requirement: **UI-control tool family** for the React console — see §4a.
