# tuktuk Grammar Review Questions

Use this file for tool/skill changes that conflict with the pinned tuktuk
grammar and need Sasha-level review before the grammar or implementation moves.

## Open

(none)

## Resolved

- **Write-tool compact result envelope** (resolved 2026-07-19, Claude+Sasha).
  The empty `compact_result_keys` for write tools in `tool_surface.json` were an
  extraction bug, not an EVA gap: `_compact_tool_result_for_model` already has
  compaction branches for `create_probe`, `update_probe`, `delete_probes`,
  and `update_prompt_settings`, but they return via
  `_attach_action_plan_hint({...})` / incremental dict builds that the v1
  ast extractor could not see. The extractor is fixed and `tool_surface.json`
  regenerated — no tool has an empty envelope now. The actual `MUT` envelope:
  `{status, action?, probe_id?, probe_name?, channel_id?, diff?,
  conflicts?/targets? (bounded), action_plan?: {plan_id, action,
  "awaiting_ui_apply", next_step_hint}}`; raw `approval` is stripped before the
  model sees it. Grammar note: `MUT.emits` says `preview_diff`/`plan_id` —
  read as `diff`/`action_plan.plan_id` (rename lands with the next grammar
  revision). **d15 stays harness/receipt state by design**: apply happens in
  the UI outside the model loop, so approval denial/expiry never arrives as a
  tool result — the model sees `awaiting_ui_apply` and then either a trusted
  action receipt (system message) or nothing. The gym's current representation
  (scratchpad/operator state) is correct; a regenerated `tool_surface.json`
  import unblocks realistic MUT-result synthesis. Regression test:
  `tests/test_agent_tool_loop.py::test_write_tool_compaction_returns_stable_preview_envelope`
  (on the feature branch).

- **search_archive coverage truncation** (resolved 2026-07-20, Claude+Sasha).
  Live agent traces on the tbilisi repro stand showed `search_archive` results
  reporting "Coverage: not reported by the backend" and a spurious
  "Result truncation: yes" on searches that returned full, untruncated result
  sets. Root cause: `agent_security.output.sanitize_output` enforces a single
  shared item budget (`ToolPolicy.max_output_items`, generic default 500)
  across an entire tool result by walking dict/list keys in insertion order
  and hard-stopping once the budget is spent. `AgentTools._search_archive`
  (`agent.py`) returned `coverage` as the *last* top-level key, after
  `results` -- and each un-compacted archive row (full vlm_summary/vlm_alert
  payload, including `state_observations`/`state_transition_events` arrays)
  is large enough that a normal 12-row page burns through 500 items well
  before the sanitizer reaches `coverage`, silently dropping it and setting
  `_truncated: true`. This broke the coverage-honesty gate
  (grammar_pin.md item 5) for both the operator UI and the model itself
  (`_compact_tool_result_for_model` runs on the already-sanitized result).
  Same failure class already recognized and fixed for `get_video_summaries`/
  `list_video_summary_channels` in `eva_adapter.py`'s `_max_output_items`
  (see the comment there) -- `search_archive` was the one archive/detection
  tool with the coverage-tail shape that hadn't been covered by that bump.
  Fix, part 1 (items): `search_archive` added to `_max_output_items` at
  4,000, matching the existing `get_video_summaries` precedent; defense in
  depth, `_search_archive`'s return dict now orders `coverage` before
  `results` so a future oversized row set clips results, not the
  honesty-critical field.

  Fix, part 2 (bytes -- found while verifying part 1 against live repro
  data): the item-count fix alone was not sufficient. `sanitize_output`
  has a SECOND, independent cap -- `_bound_serialized` -- that replaces
  the *entire* result with a useless `{"_truncated": true, "preview":
  "<64KB of raw JSON>"}` envelope once serialized size exceeds
  `max_output_bytes`, regardless of key order or the item budget. This
  was a flat, hardcoded 96,000 bytes for every tool with no per-tool
  override (unlike items/rows/timeout, which already had one). Measured
  against live tbilisi-repro data: a `search_archive` page at its own
  default (12 rows) serializes to ~141KB; `get_detections` at ITS default
  page (20 rows) measures ~221KB, scaling to ~1MB at its max_rows=100.
  Both routinely exceeded 96,000 bytes in ordinary operation -- not an
  edge case. `get_detections` is one of the most frequently called tools,
  so this was silently wiping a large fraction of ordinary calls to
  `{}`-shaped nulls, independent of the search_archive-specific
  coverage/results ordering issue. Added `_max_output_bytes` to
  `EvaAgentToolAdapter`, mirroring the existing `_max_output_items`
  pattern, wired into the `ToolPolicy` construction in place of the flat
  constant. `search_archive` and `get_detections` both raised to
  2,000,000 bytes (comfortable headroom over their measured max_rows
  worst case); both also added to `_max_output_items` at 4,000.
  `get_video_summaries`/`list_video_summary_channels` were measured too
  (54KB/15KB at typical depth) and are not currently affected --
  untouched. `build_research_batch` returned no rows for the args tried;
  no live evidence of it being broken, left untouched pending a concrete
  symptom. No tool schema, compact envelope, or intent-group change.

  Regression tests: `tests/test_agent_tool_gateway.py::
  SearchArchiveCoverageBudgetTests` (4 cases: raised item budget,
  pre-fix-order failure repro, key-order assertion against the real
  `AgentTools._search_archive`, adapter item-budget value) and
  `::OutputByteBudgetTests` (4 cases: realistic-size fixture would have
  been wiped at the old byte default even with the item fix applied,
  raised byte budget preserves a realistic page, `get_detections` gets
  both raised budgets, unlisted tools keep the conservative defaults).
