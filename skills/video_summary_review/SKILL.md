# Video Summary Review

Trigger phrases:
- `review video descriptions`
- `review video-descriptions`
- `video summary report`
- `video summaries for`
- `last night`
- `who was in the space`
- `presence review`
- `проверь видеоописания`
- `видео-описания`
- `что происходило`
- `когда был`
- `был ли`
- `визуальные доказательства`

Goal: answer period-based questions from VLM video summaries with explicit coverage, timeline, and uncertainty.

Default order:
1. Normalize the operator's period with `normalize_time_window` unless exact Unix timestamps are already supplied.
2. If the operator named one or more channels, resolve those channels and review only those channels.
3. If no channel was named, call `list_video_summary_channels` for the normalized period.
4. If more than the returned `per_turn_channel_limit` channels are active, do not read all summaries immediately. Present the candidate channels and ask the operator to choose channels or confirm full multi-turn research.
5. For selected channels and periods longer than about one hour, call `get_video_summaries` with `depth="L2"` first to get the background map. For shorter periods, use `depth="L1"` first.
6. Use L2 as context, not proof. Drill from L2 into `depth="L1"` around candidate windows.
7. Drill into `depth="live"`/`L0` only for suspicious, ambiguous, or high-value windows that need exact evidence.
8. For event-oriented review, optionally call `get_visual_window_signals` with visible positive/negative phrases to prioritize candidate frames/windows.
9. When visual proof is requested, call `get_video_summaries` with `include_evidence_frames=true` or `get_detections` with `source="vlm_summary"`/`source="vlm_alert"` for the same channel and exact time window.
10. Before saying visual confirmation, call `describe_frame` on the relevant returned `detection_id` or frame image and use that description as the visual basis.
11. Use `search_archive` only for semantic discovery. Keep archive detections separate from VLM summary evidence.
12. Return a report with: scope, coverage, timeline, direct observations, indirect indicators, archive corroboration, gaps, and confidence.

Trust hierarchy:
- Routine memory or repeated background is a prior, not proof of a new event.
- L2/L1 rollups are maps; L0 prose is an unconfirmed candidate if structured alert/state data is missing.
- Structured `alert_events`, `state_observations`, and `state_transition_events` are stronger than prose. Backend `state_transition_events` are confirmed across batches, but still operator-review candidates.
- VLM summary/alert frames anchor exact times. Visual proof requires a returned frame or `image_url` plus `describe_frame` in this turn.
- P/N/M and archive semantic search are attention signals, not conclusions.

Rules:
- Always state the resolved local time window in the report.
- Never include summaries outside the requested time window.
- Never imply that missing summary windows were reviewed.
- If `coverage.truncated` is true, state that the tool returned a sampled subset and what remains unchecked.
- If a period has no summaries, say "no VLM summary coverage" instead of "no activity".
- P/N/M is a CLIP retrieval cue, not a conclusion. Mention it as "candidate signal" only when useful.
- Do not say "confirmed visually" unless `describe_frame` analyzed the relevant frame(s) in this turn.
- For presence questions, separate:
  - direct person sightings
  - indirect activity indicators such as lighting changes, object movement, gates, barriers, signage, or vehicles
  - no-evidence windows
  - unreviewed or missing-coverage windows
- If no channel is specified and many channels are active, ask for confirmation before broad review.
- When broad review is confirmed, work in channel chunks and end with unchecked channels plus the next recommended chunk.
- Do not auto-calibrate probes from ordinary review/report questions. Use `calibrate_probe_from_archive` only if the operator explicitly asks to create, duplicate, or tune probes from the reviewed events.
- Do not accuse or infer hidden state. Rephrase sensitive operator wording into visible candidate evidence:
  - "criminal act" -> "visible operator-review incident"
  - "illegal dumping" -> "person leaving an object or waste behind"
  - "intoxicated/unconscious" -> "person lying still / unstable movement / needs review"
