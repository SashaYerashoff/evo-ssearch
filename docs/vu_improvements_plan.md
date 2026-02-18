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

## Next implementation slices
1. Reader quality
- Add compact/expanded summary card mode.
- Add optional per-entry collapse.
- Add copy/export actions per summary entry.

2. Retrieval model
- Add persisted run IDs for video sessions.
- Add UI run selector (latest/live/previous runs).
- Add search/filter by time window and channel.

3. Hierarchical summarization pipeline
- L0: batch summaries (e.g., every 30s).
- L1: periodic rollups over L0 (10-20 min windows, 12-16k token budget).
- L2: hourly rollups over L1.
- L3: shift/day rollups over L2.
- Keep provenance links: L2 -> L1 -> L0 IDs.

4. Multi-channel operator UX
- Optional split reader for 2 channels.
- Unread counters per channel in left/runtime panel.
- Severity/event highlighting in summary cards.

## Constraints to keep in mind
- Target hardware includes lower-end RTX cards (e.g., 4060/5060).
- Must coexist with CLIP real-time probing, probe capture loops, and optional VLM summarization load.
- Summary refresh should remain non-blocking and not degrade monitoring responsiveness.
