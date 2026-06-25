# Stream Prompt Tuning

Goal: refine VLM descriptions and alert prompts with explicit evidence.

Default order:
1. Resolve the target channel first. Use `channel_id` when explicit, otherwise pass `channel_ref` such as `#115` or `stream`.
2. Read `get_prompt_settings` for either global defaults or the target channel.
   Mapping:
   - `L0` / live feed prompt = `stream_system_prompt`
   - `L1/L2/L3` = `rollup_prompts.L1/.L2/.L3`
   - behavioral bookmark / alert rules = lines inside `stream_system_prompt`
   - `json_alert_prompt` = structured alert-output template only
3. Read `get_video_summaries` across the relevant depth levels (`L0`, `L1`, `L2`, `L3`) for the target period.
4. Compare the generated summaries with the operator's intent, VLM alert frames, and archive evidence from the same channel/time. Use probe hits only as secondary corroboration when explicitly relevant.
5. Identify whether the problem is:
   - stream prompt quality
   - rollup prompt quality at a specific level
   - alert JSON template quality
   - bookmark gating settings
6. Propose `update_prompt_settings` with `preview=true`.
7. Apply only after explicit operator confirmation.

Rules:
- Tune the smallest surface first: one level or one channel before changing global defaults.
- When a request is about descriptions over a date range, prefer absolute `from_ts`/`to_ts`.
- If summaries are missing, stale, or too sparse, say so explicitly instead of pretending the prompt is wrong.
- For the current pilot, optimize stream prompts and rollup prompts for video-description reporting first. Do not convert a prompt-tuning request into probe tuning unless the operator explicitly asks for probes.
- Do not say a bookmark rule exists unless `update_prompt_settings` actually applied the underlying `stream_system_prompt` change.
- Do not rewrite `json_alert_prompt` unless the operator explicitly asks to change the structured alert/parsing template.
