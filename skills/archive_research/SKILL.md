# Archive Research

Goal: answer archive questions with evidence, not guesses.

Default order:
1. Normalize the request into absolute time windows when possible.
2. For operational camera-history questions, start with `search_archive` over `source="vlm_summary"` and `source="vlm_alert"` when the source is not otherwise specified.
3. Use `get_detections` with `source="vlm_summary"`/`source="vlm_alert"` to inspect actual video-description evidence frames in the selected window.
4. Use probe hits (`source="probe"`) only when the operator explicitly asks about probes or when you need a secondary CLIP/P/N/M semantic corroboration signal.
5. Use `build_research_batch` when the question spans multiple periods, confidence bands, or source classes.
6. Use `describe_frame` only for representative samples that need visual confirmation.

Rules:
- Prefer `scope=detections` for operational questions about camera history; within that scope, prefer video-description sources (`vlm_summary`, `vlm_alert`) before probe hits.
- Use `scope=indexed_folder` only for indexed image folders.
- If the request mentions a date or period, pass `since_ms` and `until_ms` instead of vague relative windows.
- Separate `Video-description frame`, `VLM alert frame`, and `Probe hit` in the answer; do not call VLM frames probe detections.
- If results are sparse or ambiguous, ask the operator a clarifying question before making claims.
- When returning findings, separate: what matched, what did not match, and what remains uncertain.
