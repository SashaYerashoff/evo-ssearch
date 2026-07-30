# Probe Tuning

Trigger phrases:
- `calibrate probe`
- `tune thresholds`
- `probe too noisy`
- `probe missing events`
- `reduce false positives`
- `reduce false negatives`
- `double-check video-description alerts with probes`
- `duplicate VLM alerts with probes`
- `продублировать алерты пробами`
- `затюнить пробу`

Goal: tune probes using evidence from short-term, medium-term, and broader history.

Tools:
- `list_probes`
- `get_detection_summary`
- `prepare_probe_calibration_batch`
- `calibrate_probe_from_archive`
- `build_research_batch`
- `get_video_summaries`
- `describe_frame`
- `create_probe`
- `update_probe`

Use this skill only when the operator explicitly asks to tune, inspect, create, cast, or report on probes. For ordinary status reports, incident reports, channel-health checks, and video-history questions, use video-summary workflows first and treat probes only as a secondary CLIP/P/N/M corroboration signal.

Default order:

1. Read `get_detection_summary` for the last 24h and for the target channel if known.
2. For more than one probe/event/channel, run `prepare_probe_calibration_batch` instead of manually fanning out calibration calls. Continue with the returned `job_id` until `remaining_count=0`.
3. For a single probe/event/channel, run `calibrate_probe_from_archive` with a visible event query and visible contrast query when archive CLIP frames exist.
4. Inspect representative calibration frames with `describe_frame` when the recommendation is ambiguous, high-impact, or surprising.
5. Build a representative manual sample with `build_research_batch` only when calibrating an existing probe's historical hits or when calibration coverage is thin.
6. Compare recent, medium, and longer windows instead of tuning from a single moment.
7. Compare the probe behavior to `get_video_summaries` if LLM summaries exist for that period.
8. Only then propose `create_probe` or `update_probe` with `preview=true`. If a batch result returned non-null `recommended_probe_args`, pass those args through; do not rewrite them into calibration-shaped arguments. If `safe_to_apply=false` or `recommended_probe_args` is null, do not propose probe changes yet.
9. Do not apply from chat. Tell the operator to use the UI Apply button on the preview card; treat the later trusted action receipt as the only proof that the probe changed.

Embedding-model transition:

- Treat a CLIP/SigLIP2 model change as a new calibration epoch. Similarity
  scores, `pos_floor`, `margin`, archived vectors, and image probes are not
  portable between embedding spaces.
- Do not tune SigLIP2 from legacy CLIP vectors. The archive calibration tool
  skips vectors whose recorded embedding space is missing or mismatched when
  SigLIP2 is active; report that rejected coverage instead of treating it as
  target absence.
- Start SigLIP2 probes in `shadow`: they may prioritize frame review but must
  not regulate attention or corroborate an alert until their current embedding
  space has been stamped by an operator-approved create/update preview.
- Accumulate independent semantic snapshots before calibration. Use at least
  15 minutes/120 samples per channel for a first provisional pass, then compare
  a recent window with a broader routine window. Rare events need operator
  examples or reviewed VLM-alert frames; routine-only data cannot calibrate
  their positive threshold.
- For each probe, compare a visible positive phrase with a visible contrast
  phrase, inspect representative high-margin and ambiguous frames, and accept
  thresholds only when `safe_to_apply=true`. One threshold set per channel is
  the default.
- After Apply, observe a fresh window and compare hit rate, false positives,
  missed reviewed events, attention debt, and VLM batch admission. Roll back
  the probe to shadow if the new space over-fires or suppresses unrelated
  novelty.

VLM-alert to probe workflow:
- Use this when the operator asks to double-check video-description alerts with probes.
- First read the relevant L0/live prompt, VLM alerts, or video summaries; do not invent alert classes from memory.
- Convert each distinct VLM alert class into one probe per channel/event. Use `create_probe` with `preview=true` and `update_existing=true` so repeated alert classes do not create duplicate probes.
- Name probes after the observable event, not after private subjects. Good: `two people fighting`, `vehicle burnout or drift`, `person lying on ground`, `visible fire or smoke`.
- Positives must be CLIP-visible object/action/state descriptions. Avoid personal names, legality, intent, guilt, medical conclusions, or other hidden-state claims.
- Negatives are visible contrast/background states, not logical absence. Do not use `no person`, `no vehicle`, `without smoke`, or `object absent`. Use descriptions such as `clear sidewalk`, `parked vehicles on clear roadway`, `people walking normally`, `clear roadway with normal traffic`, or `empty public entrance`.
- Run `calibrate_probe_from_archive` before the preview when archive frames exist. Treat its `calibration_status`, `separation_quality`, `safe_to_apply`, `recommended_action`, and warnings as the source of truth. Use `suggested_thresholds.pos_floor` and `suggested_thresholds.margin_thr` only when `safe_to_apply=true`; otherwise request frame review or rephrase the positive/contrast queries.
- For multiple alert classes or channels, use `prepare_probe_calibration_batch` and report `job_id`, processed items, and remaining items. On "continue", resume the same `job_id`; do not reconstruct the checklist from chat.
- Treat the probe layer as a cheap secondary attention signal. It can corroborate or surface candidates for review; it is not proof and should be tuned from observed hits.

Broad-channel calibration:
- Work in chunks of at most 8 channels per turn.
- If `calibrate_probe_from_archive` returns `deferred_channel_ids`, report checked/deferred channels and ask the operator to continue calibration.
- Do not claim all channels are calibrated until every chunk has returned coverage and calibration results.
- Recommendations are per-channel unless the returned distributions are clearly similar across channels.

Tuning heuristics:
- For false positives on humans, inspect `positives`, `negatives`, and real detections before changing thresholds.
- Look for min/max ranges of `pos_score` and `margin` across the sampled windows.
- Do not treat a large positive-like count as quality. If almost every frame is positive-like, the probe may be over-broad or the contrast may be weak. Separation quality comes from margins and the tool's verdict.
- Evaluate text pairs over three horizons when possible:
  - recent: hours
  - medium: days
  - broad: the largest safe historical window available
- If data is too thin or contradictory, say so and ask the operator for a target scenario or more time range context.
- Threshold semantics are strict:
  - raising `pos_floor` makes the probe stricter
  - lowering `pos_floor` makes the probe more permissive
  - raising `margin` makes the probe stricter
  - lowering `margin` makes the probe more permissive
- Never describe lowering `margin` as "tightening" or "filtering more".
- A 24h hit count is historical archive data. After applying new thresholds, do not claim that the 24h count already improved unless you explicitly measured a post-change window.
- If the operator asks for probe status right after an update, report the stored thresholds and say that impact on live volume still needs post-change observation unless a fresh post-change query was run.
