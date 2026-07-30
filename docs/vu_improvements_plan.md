# VU Improvements Plan (Working Notes)

Last updated: 2026-07-12
Active release branch: `stabilization/0.8.3-pre-react`

## Post-pilot revision marker (recorded 2026-07-12)

After the 0.8.4 pilot release, revisit the Video history reader as an
investigation workspace rather than adding more controls to the release UI:

- add an event/alert density timeline with brush-to-select;
- use server cursors instead of offset paging for deep archive traversal;
- expose archive/hot/rollup coverage as one honest coverage bar;
- consider two-channel comparison and shareable investigation URLs;
- measure the `Auto` resolution thresholds on real 50-channel operator queries
  before changing the release defaults.

This is deliberately deferred until after the pilot. The release reader keeps
the smaller operator model: **Period / Resolution / Follow live**, indexed
PostgreSQL history pages, and explicit loading/coverage feedback.

## Objective
Improve Video Understanding usability for multi-channel operation (4-8 channels) and make summaries readable, persistent, and retrievable.

## Implemented in this branch
- Channel runtime panel is channel-centric (not probe-centric labels).
- Per-channel probe controls support pause/resume semantics.
- Monitoring start capture now checks live runtime state before deciding a channel is already running.
- Live summaries no longer vanish on stop:
  - Backend keeps summary history per channel (`LuxriotManager.summary_history`).
  - `/luxriot/session` now returns archived logs when stream is stopped.
  - `/luxriot/streams` includes `video_history_channels`.
- Live summaries reader UX upgraded:
  - Dedicated full-width summaries row.
  - Independent summary channel selector.
  - Channel runtime cards include `View summaries` shortcut.
  - `Follow live` toggle.
  - `Pause updates` toggle.
  - `Jump to latest` for unread batches.
  - Scroll no longer yanks when user is reading older entries.
- Reader quality enhancements:
  - Compact/expanded summary view mode.
  - Per-entry collapse/expand and collapse-all control.
  - Per-entry copy/export actions.
- Retrieval model:
  - Persisted run IDs for video sessions (`run_id` on capture sessions and summary entries).
  - `/luxriot/session` supports run selectors (`latest`, `live`, `all`, explicit run ID).
  - `/luxriot/session` supports time-window filtering (`from_ts`, `to_ts`) and limit.
  - Reader toolbar exposes run selector and time-window filters (`from`, `to`) with apply/clear flow.
- Hierarchical summarization (backend scaffolding):
  - Added `/luxriot/rollups` endpoint for layered rollups (`L0`, `L1`, `L2`, `L3`).
  - L1/L2/L3 rollups aggregate prior level windows and expose provenance (`source_ids`).
  - Rollups support run/time filtering (`run`, `from_ts`, `to_ts`) and per-level limit.
- Hierarchical summarization (reader wiring):
  - Live Summaries toolbar now includes level selector (`L0`-`L3`) and drill-back navigation.
  - Reader supports drill-down via provenance (`source_ids`) from higher rollup layers.
  - Rollup rows support copy/export actions and markdown rendering.
  - Summary refresh path now auto-dispatches between session logs (`L0`) and rollup view (`L1`-`L3` or drilled context).
  - Rollup API now includes run metadata (`runs`) so toolbar run selection remains consistent in rollup mode.
- Hierarchical summarization (L1 quality pass):
  - L1 rollups now support optional LLM synthesis over L0 summaries (dedupe + timeline-focused output).
  - L1 synthesis is cache-backed by `rollup_id` and throttled per request to avoid overload.
  - Tunables added via env: char budget, max new rollups per call, cache limit, and optional model override.
- Hierarchical summarization (quality gate + retrieval):
  - LLM synthesis now supports `L1/L2/L3` levels (configurable).
  - Higher-level synthesis is gated by minimum source-context size (~token threshold) to avoid weak/truncated summaries.
  - Generated rollup summaries are stored in cache with channel/level/window metadata and exposed in period-filtered rollup responses.

## Next implementation slices
1. Hierarchical summarization pipeline (phase 2)
- Add optional LM-generated semantic rollup text over each window (token-budgeted).
- Persist rollups for retrieval/history instead of on-demand only.
- Add reader UX for level selection and drill-down from L3 -> L0.

2. Multi-channel operator UX
- Optional split reader for 2 channels.
- Unread counters per channel in left/runtime panel.
- Severity/event highlighting in summary cards.

## Constraints to keep in mind
- Target hardware includes lower-end RTX cards (e.g., 4060/5060).
- Must coexist with CLIP real-time probing, probe capture loops, and optional VLM summarization load.
- Summary refresh should remain non-blocking and not degrade monitoring responsiveness.
