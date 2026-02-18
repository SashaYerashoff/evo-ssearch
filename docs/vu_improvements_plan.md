# VU Improvements Plan (Working Notes)

Last updated: 2026-02-18
Branch: `vu-improvements`

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
  - Reader toolbar run/time controls deferred until next UI stabilization pass.
- Hierarchical summarization (backend scaffolding):
  - Added `/luxriot/rollups` endpoint for layered rollups (`L0`, `L1`, `L2`, `L3`).
  - L1/L2/L3 rollups aggregate prior level windows and expose provenance (`source_ids`).
  - Rollups support run/time filtering (`run`, `from_ts`, `to_ts`) and per-level limit.

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
