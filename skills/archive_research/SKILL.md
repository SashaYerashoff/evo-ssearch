# Archive Research

Trigger phrases:
- video-description archive
- search the archive
- archive search
- find similar
- поиск по архиву
- найти в архиве

Goal: answer archive questions with evidence, not guesses.

Tools:
- `normalize_time_window`
- `search_archive`
- `get_detections`
- `build_research_batch`
- `describe_frame`

Default order:
1. Normalize the request into absolute time windows when possible.
2. For operational camera-history questions, start with `search_archive` over `source="vlm_summary"` and `source="vlm_alert"` when the source is not otherwise specified.
3. Use `get_detections` with `source="vlm_summary"`/`source="vlm_alert"` to inspect actual video-description evidence frames in the selected window.
4. Use probe hits (`source="probe"`) only when the operator explicitly asks about probes or when you need a secondary CLIP/P/N/M semantic corroboration signal.
5. Use `build_research_batch` when the question spans multiple periods, confidence bands, or source classes.
6. EVA selects up to eight visually distinct top SigLIP candidates and calls
   `describe_frame` once with `detection_ids`; treat the returned per-frame
   verdicts as the bounded visual verification stage.

Rules:
- Prefer `scope=detections` for operational questions about camera history; within that scope, prefer video-description sources (`vlm_summary`, `vlm_alert`) before probe hits.
- Use `scope=indexed_folder` only for indexed image folders.
- If the request mentions a date or period, pass `since_ms` and `until_ms` instead of vague relative windows.
- Separate `Video-description frame`, `VLM alert frame`, and `Probe hit` in the answer; do not call VLM frames probe detections.
- If results are sparse or ambiguous, ask the operator a clarifying question before making claims.
- When returning findings, separate: what matched, what did not match, and what remains uncertain.
- `search_archive` returns ranked candidates, not binary matches. Never infer
  absence from a low score or missing words in a candidate label.
- Positive and negative visual claims are symmetric: when the batch is missing,
  unparsed, or uncertain, report candidates and uncertainty instead of saying
  the entity/event is absent. A clean batch still covers only the reviewed top
  candidates, not every unsampled frame in the archive.
- Do not run `calibrate_probe_from_archive` during ordinary archive research unless the operator explicitly pivots to creating/tuning probes or duplicating VLM alerts with probes.
