# Cross Channel Correlation

Trigger phrases:
- `compare`
- `same car`
- `same person`
- `correlate`
- `possible connection`
- `сравни`
- `то же`
- `та же`
- `тот же`
- `сопоставь`
- `возможные связи`

Goal: compare candidate events across channels and propose possible links with explicit uncertainty.

Tools:
- `normalize_time_window`
- `list_video_summary_channels`
- `get_video_summaries`
- `get_detections`
- `get_visual_window_signals`
- `describe_frame`

Default order:
1. Normalize the period with `normalize_time_window`. Resolve mentioned channels directly; when the operator asks for all/available channels, enumerate the authorized scope with `list_video_summary_channels` first.
2. Build per-channel event candidates using VLM summaries first (`get_video_summaries`) and VLM archive frames second (`get_detections` with `source="vlm_summary"`/`source="vlm_alert"`).
3. Use `get_visual_window_signals` only as a weak ranking cue when searching for candidate appearances across channels.
4. Compare candidate events by time proximity, direction of movement, object/person description, color, make/model when visible, and repeated distinctive features.
5. Before saying a link is visually confirmed, call `describe_frame` on the relevant returned `detection_id` or frame image for each side of the link.
6. If image-similarity by detection/frame id is not available, say so and rely only on described frames, returned thumbnails, and summary text.
7. Return links as `confirmed`, `likely`, `possible`, or `insufficient evidence`.

Output:
- Candidate event table.
- Correlation table: event A, event B, timing gap, shared visual features, conflicting features, confidence.
- Explicit uncertainty and recommended next check.

Rules:
- Do not assert identity of a person, vehicle, or animal unless there is direct evidence strong enough in returned frames.
- P/N/M may support ranking, but identity correlation still requires frame descriptions and explicit uncertainty.
- Do not say "confirmed visually" unless `describe_frame` analyzed the relevant frame(s) in this turn.
- License plates, faces, and legal conclusions require operator review; the agent can only flag visible candidates.
