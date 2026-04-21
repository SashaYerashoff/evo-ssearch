# Prompt Tuning

Goal: refine VLM descriptions and alert prompts with explicit evidence.

Default order:
1. Read `get_prompt_settings` for either global defaults or the target channel.
2. Read `get_video_summaries` across the relevant depth levels (`L0`, `L1`, `L2`, `L3`) for the target period.
3. Compare the generated summaries with the operator's intent and with observed detections.
4. Identify whether the problem is:
   - stream prompt quality
   - rollup prompt quality at a specific level
   - alert JSON prompt quality
   - bookmark gating settings
5. Propose `update_prompt_settings` with `preview=true`.
6. Apply only after explicit operator confirmation.

Rules:
- Tune the smallest surface first: one level or one channel before changing global defaults.
- When a request is about descriptions over a date range, prefer absolute `from_ts`/`to_ts`.
- If summaries are missing, stale, or too sparse, say so explicitly instead of pretending the prompt is wrong.
