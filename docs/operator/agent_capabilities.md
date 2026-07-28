# Agent Capabilities & Limits

What the EVA AI agent can do, what it cannot, and how to ask so you get good
answers. The agent is **video-description-centered**: its standing context and
reports lead with live descriptions and their alerts; probes/CLIP are tools it
uses to dig through volume.

## What it can do

**Perceive & describe**
- `describe_frame` — fresh VLM description of a live snapshot, an archived
  detection, or an uploaded/path image.

**Read aggregated history (read-only — does not trigger LLM rollup synthesis)**
- `list_video_summary_channels` — which channels are producing descriptions,
  alert counts by severity, active/inactive, recent alert titles from the
  channel status digest, and pipeline health over a period.
- `get_video_summaries` — L0–L3 timeline for a channel/period, with evidence
  frames, structured alert/state fields, and a coverage contract.
- `count_video_summary_events` — count mentions/events in summaries for a channel.

**Find in the archive (semantic)**
- `detections_search_text` / `detections_search_image` — CLIP search over
  archived frames (`vlm_summary`, `vlm_alert`, `probe`).
- `get_detections` — list archive rows by channel/source/time.
- `track_visual_state_transitions` — appear/disappear, open/close, leave/return,
  gather/disperse over archived CLIP-scored frames, with boundary-frame evidence.

**Report**
- `generate_report` — **video-description-first** by default; probe report only
  via explicit `report_type=probes`.

**Act (gated)**
- `create_bookmark` — push an event to Luxriot Evo (subject to the approval
  workflow; may be unavailable in secure mode until approval is enabled).
- `update_prompt_settings`, `update_probe` — **preview-only** from chat in
  secure mode; the UI Apply button commits the server-owned action plan and
  writes a trusted receipt.
- Prompt settings are split by purpose: L0 description behavior is
  `stream_system_prompt`; **Alert Criteria** is `alert_policy_prompt`; the
  structured alert parser contract is `json_alert_prompt`.
- `Protocol: Deploy` starts a durable commissioning workflow for at most eight
  channels. The agent inventories authorized channels, persists the selected
  scope/groups, performs one bounded scene survey, collects operator-grounded
  routine/alert/counting policy, and returns one composite preview. The UI
  Apply action installs the channel groups, Alert Criteria, bounded starter
  probes, counted-state profiles, quiet-window preference, and optional live
  starts. A service restart does not lose the deployment stage.

**Internal tools (agent-invoked, for semantic bulk comparison)**
- `query_probe` / `list_probes` / `get_visual_window_signals` — CLIP P/N/M
  signals and probe matching. These serve the agent's search; they are not the
  operator's primary workflow.
- The agent can turn concrete video-description alert classes into preview
  probes as a second attention layer. It translates names and prose into generic
  visible classes/actions (for example, "two people fighting", "vehicle burnout
  or drift", or "person lying on ground") and uses visible contrast states
  instead of negation. These probes are corroborating candidates, not proof.
- `calibrate_probe_from_archive` — read-only archive calibration for proposed
  probe text. It scores archived frames with CLIP P/N/M, suggests initial
  `pos_floor`/`margin_thr` values, returns a `calibration_status` verdict, and
  returns representative frames. Threshold suggestions are applyable only when
  `safe_to_apply=true`; otherwise they are diagnostic and the operator should
  inspect frames or rephrase the probe text. For 50 channels it works in chunks
  of at most 8 channels and reports deferred channels.
- `prepare_probe_calibration_batch` — stateful multi-probe/multi-channel
  calibration. It returns a `job_id`, compact decision ledger, remaining items,
  and pass-through preview args only for safe verdicts, so the agent does not
  reconstruct a long checklist from chat. For multiple alert classes or many
  channels, the agent should continue by `job_id` instead of dumping raw P/N/M
  traces.
- `query_counted_state_metric` — answers transition-count and dwell-time
  questions from the continuous `semantic_snapshot` archive. Counts are state
  episodes, not delivered alerts, so bookmark cooldown/dedup does not change
  the answer. Unknown/no-coverage time is reported separately.

**Time & scope helpers**
- `normalize_time_window` — turns "last 2 hours", "yesterday evening" into exact
  timestamps. The agent calls this before period queries.

## How to ask well

- Give a **period** and a **channel** (or "active channels"). Natural language
  time is fine.
- For "find" questions, the agent scopes by channel + time so recall is complete
  in that window. Unscoped "search everything for two weeks" only inspects the
  most recent slice — always scope.
- For "how many times X" questions, expect **candidates** with boundary frames;
  ask it to describe those frames to confirm. A configured counted-state
  profile makes the query reproducible across operators and keeps alert
  delivery out of the count.
- Read its **coverage line**: it states the period requested, period inspected,
  entries returned, and whether the result was truncated.

## Evidence confidence

- Routine memory/background is only a prior.
- L0/L1/L2 prose is useful for discovery; L0 prose is unconfirmed when no
  structured alert/state data exists.
- Structured `alert_events`, `state_observations`, and backend
  `state_transition_events` are stronger. `state_transition_events` are confirmed
  across batches, but remain candidates for operator review.
- Visual proof requires an evidence frame or `image_url` and a fresh
  `describe_frame` result in the same turn.
- CLIP P/N/M, probes, and semantic archive search are attention signals, not
  standalone proof.

## What it cannot do (do not promise these)

- **No** PDF / CSV / email export, no async/background report queues, no file
  links — only structured chat reports.
- **No** ground-truth counting from CLIP — state transitions are candidates.
- **No** complete recall on unscoped broad searches — bounded by the candidate
  window.
- **No** chat-side apply — probe/prompt edits are preview-only from chat. Apply
  happens via the UI Apply button and a trusted server receipt.
- **No** autonomous sensitive actions — by design the agent proposes, a human
  confirms (see safety posture).

## Operating characteristics

- Up to 64 tool calls per turn; broad multi-channel research is chunked across
  turns and reports which channels remain unchecked.
- Ordinary investigations reassemble working state from the conversation.
  Protocol Deploy and its first commissioning receipt are separately persisted
  in the tenant runtime store, so the workflow can resume after a new chat or
  EVA restart.

## Safety posture (why human-in-the-loop)

The agent reasons over representations (text, tool results); every signal reaches
it through that one channel. That makes it powerful at synthesis but means
nothing reaching it is incorrigible — a crafted input can act as a "driver" the
same way a legitimate instruction does (prompt injection). Therefore sensitive
actions (bookmarks, prompt/probe changes) are **preview/approval-gated**, and the
agent's conclusions about incidents are **evidence-cited candidates** for an
operator to confirm. See [security_threat_model](../architecture/security_threat_model.md)
and [cognitive_architecture](../architecture/cognitive_architecture.md).
