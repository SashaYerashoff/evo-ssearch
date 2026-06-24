# Video Event Check

Trigger phrases:
- `check channel`
- `check whether`
- `when did`
- `was there`
- `проверь канал`
- `проверь когда`
- `был ли`
- `во сколько`
- `когда наблевали`
- `почтальон`

Goal: check one named channel for one requested event inside one time window without overclaiming.

Default order:
1. Normalize the time window with `normalize_time_window`.
2. Resolve the named channel. If the channel is ambiguous, list likely candidates and ask one short clarification.
3. Call `get_video_summaries` with `depth="L2"` for context when the window is longer than about one hour; otherwise start with `depth="L1"`.
4. Convert the user event into visible evidence terms before searching. Example: "unlawful" becomes concrete observable actions; "unvaccinated dog" becomes "dog without visible ear tag".
5. Call `get_visual_window_signals` with the visible positive phrase and a concrete negative contrast phrase when useful. Treat P/N/M as triage only.
6. Search `source="vlm_summary"` and `source="vlm_alert"` with the same channel and exact time window for candidate frames.
7. Drill candidate windows with `get_video_summaries depth="live"` or `depth="L1"` and `include_evidence_frames=true`.
8. Before saying visual confirmation, call `describe_frame` on the relevant returned `detection_id` or frame image and use that description as the visual basis.
9. Return earliest-before / first-visible / last-visible timing when available.

Output:
- Scope and coverage.
- Event query rewritten as visible evidence.
- Findings: found / not found / possible / insufficient coverage.
- Evidence ledger with time, source, confidence, and image thumbnails when available.
- Gaps and next suggested drill-down.

Rules:
- Do not call candidate VLM summary frames "probe detections".
- P/N/M is not proof. Use it to choose frames/windows, then verify with summaries and `describe_frame`.
- Do not say "confirmed visually" unless this turn returned `image_url` rows for the relevant channel/time and `describe_frame` analyzed the relevant frame(s).
- Do not accuse people or infer intent, guilt, vaccination, intoxication, or legality from video.
