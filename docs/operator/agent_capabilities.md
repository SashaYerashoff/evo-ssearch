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
  alert counts by severity, active/inactive, over a period.
- `get_video_summaries` — L0–L3 timeline for a channel/period, with evidence
  frames and a coverage contract.
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
- `update_prompt_settings`, `update_probe` — **preview-only** in secure mode;
  shows a diff, never applies silently.

**Internal tools (agent-invoked, for semantic bulk comparison)**
- `query_probe` / `list_probes` / `get_visual_window_signals` — CLIP P/N/M
  signals and probe matching. These serve the agent's search; they are not the
  operator's primary workflow.

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
  ask it to describe those frames to confirm.
- Read its **coverage line**: it states the period requested, period inspected,
  entries returned, and whether the result was truncated.

## What it cannot do (do not promise these)

- **No** PDF / CSV / email export, no async/background report queues, no file
  links — only structured chat reports.
- **No** ground-truth counting from CLIP — state transitions are candidates.
- **No** complete recall on unscoped broad searches — bounded by the candidate
  window.
- **No** silent changes — probe/prompt edits are preview-only; bookmark creation
  is approval-gated.
- **No** autonomous sensitive actions — by design the agent proposes, a human
  confirms (see safety posture).

## Operating characteristics

- Up to 64 tool calls per turn; broad multi-channel research is chunked across
  turns and reports which channels remain unchecked.
- The agent reassembles its working state from the conversation each turn; it has
  no persistent self between turns beyond the stored session. Long investigations
  may need re-grounding — restate the channel/period if a thread drifts.

## Safety posture (why human-in-the-loop)

The agent reasons over representations (text, tool results); every signal reaches
it through that one channel. That makes it powerful at synthesis but means
nothing reaching it is incorrigible — a crafted input can act as a "driver" the
same way a legitimate instruction does (prompt injection). Therefore sensitive
actions (bookmarks, prompt/probe changes) are **preview/approval-gated**, and the
agent's conclusions about incidents are **evidence-cited candidates** for an
operator to confirm. See [security_threat_model](../architecture/security_threat_model.md)
and [cognitive_architecture](../architecture/cognitive_architecture.md).
