# Video Summary Review

Trigger phrases:
- `review video descriptions`
- `review video-descriptions`
- `video summary report`
- `video summaries for`
- `last night`
- `who was in the space`
- `presence review`

Goal: answer period-based questions from VLM video summaries with explicit coverage, timeline, and uncertainty.

Default order:
1. Normalize the operator's period with `normalize_time_window` unless exact Unix timestamps are already supplied.
2. If the operator named one or more channels, resolve those channels and review only those channels.
3. If no channel was named, call `list_video_summary_channels` for the normalized period.
4. If more than the returned `per_turn_channel_limit` channels are active, do not read all summaries immediately. Present the candidate channels and ask the operator to choose channels or confirm full multi-turn research.
5. For selected channels, call `get_video_summaries` with `depth="L1"` first.
6. Drill into `depth="live"`/`L0` only for suspicious, ambiguous, or high-value windows.
7. Use `search_archive`/`get_detections` only as corroboration. Keep archive detections separate from VLM summary evidence.
8. Return a report with: scope, coverage, timeline, direct observations, indirect indicators, archive corroboration, gaps, and confidence.

Rules:
- Always state the resolved local time window in the report.
- Never include summaries outside the requested time window.
- Never imply that missing summary windows were reviewed.
- If a period has no summaries, say "no VLM summary coverage" instead of "no activity".
- For presence questions, separate:
  - direct person sightings
  - indirect occupancy indicators such as lighting changes, object movement, doors, devices, screens, or appliances
  - no-evidence windows
  - unreviewed or missing-coverage windows
- If no channel is specified and many channels are active, ask for confirmation before broad review.
- When broad review is confirmed, work in channel chunks and end with unchecked channels plus the next recommended chunk.
