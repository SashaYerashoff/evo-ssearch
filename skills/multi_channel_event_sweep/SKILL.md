# Multi Channel Event Sweep

Trigger phrases:
- `check channels`
- `group of channels`
- `all cameras`
- `all active channels`
- `проверь группу каналов`
- `по всем каналам`
- `все камеры`
- `несколько каналов`

Goal: search a channel group for one event without pretending to review more channels than the turn can handle.

Default order:
1. Normalize the time window with `normalize_time_window`.
2. Resolve explicit channel IDs/titles if provided.
3. If no explicit group is provided, call `list_video_summary_channels` for the period.
4. If active channels exceed `per_turn_channel_limit`, present candidate channels and ask the operator to choose or confirm full multi-turn research.
5. For each selected channel, start with `get_video_summaries` at L2/L1 depending on period length, then use `get_visual_window_signals` for the event phrase if candidate windows are unclear.
6. Drill only candidate windows.
7. If visual confirmation is needed, pull VLM summary/alert frames with `get_detections` (`source="vlm_summary"`/`source="vlm_alert"`) and call `describe_frame` on the relevant returned `detection_id` or frame image before calling anything visually confirmed.
8. Keep a per-channel ledger with checked, candidates found, no coverage, and unchecked.

Output:
- Scope: checked channels vs unchecked channels.
- Candidate ledger grouped by channel.
- Evidence frames only from VLM summary/alert archive for the same channel/time.
- Next chunk recommendation if research is incomplete.

Rules:
- Never say "all channels were checked" unless the tool results cover all requested channels.
- If the event is sensitive or accusatory, phrase results as candidates for operator review.
- P/N/M can rank channels/windows, but it cannot confirm an event by itself.
- Do not run `calibrate_probe_from_archive` during a normal sweep unless the operator explicitly asks to create, duplicate, or tune probes for the event class.
- Do not say "confirmed visually" unless `describe_frame` analyzed the relevant frame(s) in this turn.
