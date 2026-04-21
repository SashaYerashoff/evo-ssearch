# Archive Research

Goal: answer archive questions with evidence, not guesses.

Default order:
1. Normalize the request into absolute time windows when possible.
2. Use `search_archive` for semantic discovery by text.
3. Use `get_detections` to inspect the actual hits in the selected window.
4. Use `get_detection_summary` to understand probe/channel distribution.
5. Use `build_research_batch` when the question spans multiple periods, confidence bands, or probes.
6. Use `describe_frame` only for representative samples that need visual confirmation.

Rules:
- Prefer `scope=detections` for operational questions about camera history.
- Use `scope=indexed_folder` only for indexed image folders.
- If the request mentions a date or period, pass `since_ms` and `until_ms` instead of vague relative windows.
- If results are sparse or ambiguous, ask the operator a clarifying question before making claims.
- When returning findings, separate: what matched, what did not match, and what remains uncertain.
