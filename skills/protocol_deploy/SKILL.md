# Protocol Deploy

Trigger phrases:

- `Protocol: Deploy`
- `protocol deploy`
- `протокол деплой`

Goal: commission a fresh EVA installation for at most eight
homeostatically-regulated channels without keeping deployment state in chat.
The current agent head may be Qwen3-VL-4B. Keep every turn literal, short, and
phase-bound; use IDs and enums copied from tool/operator results.

Tools:
- `start_deployment`
- `configure_deployment`
- `survey_deployment`
- `apply_deployment_plan`
- `get_deployment_status`
- `normalize_time_window`
- `query_counted_state_metric`

## Durable flow

1. Call `start_deployment` with `target_channel_count=8`.
   - If it resumes an unfinished deployment, continue from `next_action`.
   - Never invent a channel ID.
2. Show the compact inventory. Ask the operator which 1–8 channels to use and
   whether to group them by operational role. One channel may be in at most one
   group.
3. Call `configure_deployment` with the chosen `channel_ids` and optional
   `groups`. This only updates the durable draft.
4. Call `survey_deployment` once. Use `fast_mode=false` unless the operator asks
   for the fastest possible setup.
   - Treat its scene fingerprints as provisional visual observations.
   - Do not infer identity, intent, ownership, or a security rule from them.
5. Ask one compact requirements question covering:
   - expected visible routine for each channel/group;
   - concrete visible alert conditions and their severities;
   - response to otherwise-unexpected visible activity:
     `ignore|log|info|low|normal|high|critical` and novelty
     `low|balanced|high`;
   - any state whose transitions or dwell time should be counted;
   - the preferred local quiet window for preemptible 9B consolidation.
6. Translate only the operator's answer into `requirements` and `quiet_window`,
   then call `configure_deployment` again.
   - Alert descriptions belong to VLM Alert Criteria, not the L0 role prompt.
   - CLIP `positive_query` and `contrast_query` must describe two visible
     alternatives. Do not use literal negation such as “no person”.
   - A counted state requires both queries. Choose
     `count_transitions`, `measure_duration`, or `count_and_duration`.
   - The quiet window is a preferred admission window, not a blind period:
     live L0 monitoring continues, and 9B work is preempted/deferred by active
     incidents or attention debt.
7. Call `apply_deployment_plan` with `preview=true`,
   `commissioning_after_minutes=15`, and the operator's `start_live` choice.
   Summarize its bounded diff and ask the operator to use the UI Apply action.
   Never call it with `preview=false`.
8. After Apply, call `get_deployment_status` when asked for progress. The
   server owns the timer and resumes it after EVA restarts.

## What Apply installs

- channel groups;
- per-channel VLM Alert Criteria while preserving unrelated existing policy;
- at most four starter homeostatic probes per channel, idempotent by
  deployment/name/channel;
- counted-state profiles whose event counts are independent of alert delivery,
  cooldown, and deduplication;
- the operator's 9B consolidation window;
- selected live streams when requested.

The first commissioning pass waits for at least 120 independent archived
semantic snapshots per channel. It requests one bounded L1 synthesis on the
agent profile, searches the continuous embedding archive, checks P/N/M separation,
estimates episode cadence for cooldown/dedup, and stores proposal-only probe
adjustments. It does not silently rewrite semantic meaning, severity, or alert
policy. Report coverage failures as `waiting_coverage`, not as an empty scene.

If the configured attention embedder is SigLIP2, treat commissioning as a new
embedding epoch. Starter probes remain shadow until enough SigLIP2-tagged
semantic snapshots exist and the operator approves the calibrated proposal.
Never reuse CLIP thresholds or interpret rejected legacy vectors as evidence
that an event did not occur. Continue live L0 from CV-selected frames while
this calibration debt is open.

## Count and duration questions

Use `query_counted_state_metric` for questions such as “how many times did the
person leave the workstation and how long was it occupied?”. Prefer
`metric_id`; use `metric_name` plus `channel_id` when names repeat. Normalize
the requested time with `normalize_time_window` when needed.

Report:

- episode/transition count;
- dwell duration for the configured state;
- unknown or uncovered time separately;
- that these are sampled visual-state estimates, not identity or attendance
  records.

## Guardrails

- Do not reconstruct this workflow from the legacy survey, individual
  prompt-edit, or individual probe-create primitives.
- Do not fan out dozens of tool calls; the composite tools own iteration and
  persist full receipts outside the model context.
- Do not create more than eight channels, eight groups, six alert conditions
  per requirement pack, or four starter probes per channel.
- Do not claim deployment is applied from a preview. Application requires the
  operator-approved plan.
- If the operator only asks whether cameras connect, use the normal channel
  inventory/status tools; do not start or resume Protocol Deploy.
