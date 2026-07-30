# Stream Prompt Tuning

Goal: refine VLM descriptions and alert prompts with explicit evidence.

Tools:
- `normalize_time_window`
- `get_prompt_settings`
- `get_video_summaries`
- `get_detections`
- `update_prompt_settings`

Default order:
1. Resolve the target channel first. Use `channel_id` when explicit, otherwise pass `channel_ref` such as `#115` or `stream`.
2. Read `get_prompt_settings` for either global defaults or the target channel.
   Mapping:
   - `L0` / live feed prompt = `stream_system_prompt`
   - channel-specific alert/watch criteria = `alert_policy_prompt`
   - `L1/L2/L3` = `rollup_prompts.L1/.L2/.L3`
   - `json_alert_prompt` = structured alert-output template only
3. If `prompt_health.needs_migration=true`, propose `update_prompt_settings` with `changes.migrate_legacy_alert_policy=true` and `preview=true` before other edits. This moves legacy alert/watch prose out of `stream_system_prompt` into `alert_policy_prompt`.
4. Read `get_video_summaries` across the relevant depth levels (`L0`, `L1`, `L2`, `L3`) for the target period.
5. Compare the generated summaries with the operator's intent, VLM alert frames, and archive evidence from the same channel/time. Use probe hits only as secondary corroboration when explicitly relevant.
6. Identify whether the problem is:
   - stream prompt quality
   - alert criteria quality
   - rollup prompt quality at a specific level
   - alert JSON template quality
   - bookmark gating settings
7. Propose `update_prompt_settings` with `preview=true`.
8. Do not apply from chat. Tell the operator to use the UI Apply button on the preview card; treat the later trusted action receipt as the only proof that prompt settings changed.

Rules:
- Tune the smallest surface first: one level or one channel before changing global defaults.
- When a request is about descriptions over a date range, prefer absolute `from_ts`/`to_ts`.
- If summaries are missing, stale, or too sparse, say so explicitly instead of pretending the prompt is wrong.
- For the current pilot, optimize stream prompts and rollup prompts for video-description reporting first. Do not convert a prompt-tuning request into probe tuning unless the operator explicitly asks for probes.
- When the operator asks to watch for a condition or add alert behavior, update `alert_policy_prompt`, not `stream_system_prompt`.
- In the UI, `alert_policy_prompt` is Alert Criteria: plain-language channel/default watch criteria, separate from L0 description style and separate from the `json_alert_prompt` parser contract.
- Do not say a bookmark rule exists unless `update_prompt_settings` actually applied the underlying `alert_policy_prompt` or bookmark setting change.
- Do not rewrite `json_alert_prompt` unless the operator explicitly asks to change the structured alert/parsing template.
