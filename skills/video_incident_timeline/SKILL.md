# Video Incident Timeline

Trigger phrases:
- `describe incidents`
- `timeline`
- `what happened`
- `visual evidence`
- `опиши происшествия`
- `последовательно`
- `что сегодня происходило`
- `центральная площадь`
- `предоставь доказательства`

Goal: produce a chronological operator report for notable events on one channel or a small named channel set.

Default order:
1. Normalize the period with `normalize_time_window`.
2. Resolve channels. If none are named and many channels are active, use `list_video_summary_channels` and ask for confirmation before full research.
3. Start with `get_video_summaries depth="L2"` for the broad map.
4. Use `depth="L1"` for windows with alerts, deviations, or user-relevant scene changes.
5. Use `get_visual_window_signals` for user-requested event types when the summaries are too broad or ambiguous. Treat P/N/M as attention, not proof.
6. Do not build the timeline from the latest entries only. Select 2-3 candidate windows across the requested period, prioritizing alert/deviation windows plus first/last coverage anchors when needed.
7. Use `depth="live"`/`L0` only around selected event windows.
8. Pull VLM archive frames with `include_evidence_frames=true` or `get_detections source="vlm_summary"/"vlm_alert"`.
9. Before saying visual confirmation, call `describe_frame` on the relevant returned `detection_id` or frame image and use that description as the visual basis.

Output:
- Scope and coverage.
- Chronological event table: time, channel, event, severity, source, confidence, evidence.
- Routine background only once; do not repeat static scene descriptions.
- Unchecked windows and missing evidence.

Trust hierarchy:
- Routine memory is a prior for normal background, not event proof.
- L2/L1 rollups are maps; L0 prose is unconfirmed when no structured alert/state record supports it.
- Structured `alert_events`, `state_observations`, and backend `state_transition_events` outrank prose. Treat `state_transition_events` as cross-batch confirmed candidates for operator review.
- Visual proof requires an evidence frame or `image_url` and `describe_frame` in this turn.
- P/N/M and semantic archive hits only prioritize windows.

Rules:
- Preserve low/info events if they are user-relevant, but do not escalate them into safety incidents.
- Separate "routine", "deviation", "operator-review incident", and "insufficient evidence".
- Use P/N/M only to prioritize candidate windows; never present it as factual confirmation.
- A thumbnail is evidence of a frame, not proof of intent.
- Do not say "confirmed visually" unless `describe_frame` analyzed the relevant frame(s) in this turn.
