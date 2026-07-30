# Operator Guide

A hands-on guide to **driving** EVA AI — not just watching it. After this you can
run an investigation, read video-descriptions and alerts, search the archive, and
ask the agent the right way. EVA is **video-description-first**: the system
continuously describes each channel and raises alerts; probes/CLIP are mostly the
agent's search tools.

Terms: [glossary](../00_CANON/glossary.md). Ready-made scenarios:
[operator_scenarios](operator_scenarios.md). Agent limits:
[agent_capabilities](agent_capabilities.md).

## 1. Sign in & what you can touch

- Open the deployment URL you were given. Office demo installs may use plain
  HTTP on port `5000`; client deployments should normally be behind HTTPS/TLS.
- Log in with your named account. Your **role** (admin / engineer / operator /
  viewer) and **channel grants** decide what you see. If a tab or channel is
  missing, you don't have the grant — ask an admin.
- The workspace has tabs: **Video**, **Archive Research**, **Probes**,
  **Agent**. For public-order monitoring you mostly live in **Video** and
  **Agent**.

## 2. Video tab — the live perception

This is where the system's always-on description of each channel lives.

- **Live Stream Control** — pick a channel; see its live preview.
- **Start summaries / Stop summaries** — starts or stops the live
  video-description loop for the selected channel. Starting a channel persists
  the desired live session, so the service can restore it after restart.
- **Stream context** — channel, cadence, batch size, model, queue, and runtime
  state.
- **VLM Feed controls** keep three choices separate: **Period** (Live, Today,
  Yesterday, a calendar range), **Resolution** (Auto, observations, 15 min,
  1 h, 6 h), and **Follow live**. Historical periods span service restarts and
  are read from the PostgreSQL archive in pages; `Load earlier` retrieves the
  next page without restarting analysis.
- **VLM Feed / L0–L3** — the running **video-descriptions** (L0 = per-batch
  observations) and the rollups (L1/L2/L3 = 15 min / 1 h / 8 h summaries).
  `Auto` chooses a practical resolution for the selected period. Collapsed
  summary rows show **alert badges** when something was flagged.
- Rollup status is explicit: **semantic** is a completed operator narrative,
  **aggregation pending** is an open window, and **semantic unavailable** is a
  coverage/status fallback rather than a behavioral conclusion. Use **Generate
  semantic** to retry one degraded historical window, or **Drill L0** to inspect
  its source observations. Internal homeostasis/memory payloads are not shown in
  operator summaries.
- **Channel Runtime / stream health** — whether the channel is actively
  producing descriptions and whether there are **coverage gaps / dropped
  batches**. A channel that is *quiet* is fine; a channel that is *blind* (gaps)
  is not — trust the health panel, not silence.

Prompt/settings note: **Alert Criteria** is the plain-language "watch for /
alert on" policy. The L0 stream prompt is for description style/role. The
legacy-named JSON template is the unified machine-readable `BATCH_STATE_JSON`
contract and should not be edited unless an engineer is intentionally changing
parser/schema behavior.

What to do here: confirm your demo channels are running, skim recent L1/L2 for
context, check coverage/health, and note any alert badges before you ask the
agent.

## 3. Agent tab — ask in plain language

This is the main investigation surface.

- **Ask with a period and a channel.** "What happened on channel 12 in the last 2
  hours?" Natural time works ("yesterday evening", "last 90 minutes").
- **Sessions** — persistent investigation threads. Start a new session for a new
  incident or demo track; reopen an old session when you want the agent to keep
  the same task context. If the thread starts drifting, restate the channel and
  period or start a fresh session.
- **Quick-question chips** are video-description-first — use them as starting
  points and edit the channel/period.
- **Read the coverage line** in answers: it tells you the period requested vs.
  actually inspected vs. returned, and whether results were truncated. This is how
  you know an answer is complete.
- **Evidence frames / thumbnails** appear with answers that mention frames or
  detections — click to enlarge, then ask the agent to **describe** a frame for a
  fresh read.
- For "how many times X happened" questions, the agent returns **candidates** with
  boundary frames — confirm visually before treating as fact.
- Trust evidence by provenance: routine background is context; L0/L1/L2 prose is
  a candidate unless structured alert/state data supports it; backend
  `state_transition_events` are stronger cross-batch candidates; visual proof
  requires a frame plus a fresh `describe_frame` read.

Good asks (see [operator_scenarios](operator_scenarios.md) for more):
- "Across active channels, were there public-order alerts today? List the worst."
- "Search the archive for a vehicle drifting at the square."
- "At channel 7, how many times did a crowd gather and disperse in 6 hours?"

If a thread drifts, restate the channel and period — that re-grounds the agent.

## 4. Archive Research — semantic frame search

For digging through stored frames directly.

- **Frame Archive filters** — channel, source, time range.
- **Semantic text search** — type what you're looking for ("people fighting").
  Results are CLIP-ranked frames. **Scope by channel + time** — broad unscoped
  searches only see the most recent slice.
- **Image search** — find frames similar to an uploaded image.
- **Min match slider** — raise it to keep only stronger matches.
- **Result card** — score details, source label (video-description frame vs probe
  hit), **Preview** and **describe** for a VLM read, and **Find similar**.

## 5. Probes tab (mostly agent-driven now)

Probes are CLIP matchers. In this deployment they're primarily the agent's
bulk-search tool; as an operator you'll mainly **read** probe hits if shown. You
can view the Probe Board; creating/tuning probes is an engineer/agent task. (A
curated watch-list probe set may be cast on demo channels as a safety-net
detector — see the demo runbook.)

The board nests **channel group → channel → probes**. Channel groups are
EVA-side labels you create yourself ("Perimeter", "Berth 3"); a channel belongs
to one group, and channels you have not grouped appear under **Ungrouped**.
Deleting a group never deletes probes.

Every probe card carries a badge saying who created it:

- **OP** — an operator created it by hand.
- **AI** — the agent proposed it and an operator approved the change.
- **VLM** — a temporary follow-up raised automatically from a video-description
  alert. These carry a countdown to their expiry, never write recorder
  bookmarks, and the inspector links back to the parent alert in the archive.

Use the **Created by** and **State** filters plus the search box to narrow the
board; **Grid** suits a few probes per channel and **List** suits many. Filtering
changes only what you see, never probe state.

Important: a probe **negative** is not "no X". CLIP negatives must be visible
contrast/background states, such as "people standing normally with empty hands"
or "clear lobby with ordinary pedestrian movement". If the agent refuses
`negative: no weapon` and asks for a visible alternative, that is correct safety
behavior.

If you want a second layer for video-description alerts, ask the agent to create
**probe previews** from the alert classes. The agent should translate the alert
into generic visual wording (not private names) and use visible background states
as contrasts. Example: *"Create preview probes to double-check the current
video-description alerts on active channels."* Review the preview before any
change is applied.

For higher confidence, ask for calibration first: *"Calibrate preview probes for
fighting alerts on active channels using the last 24 hours of archive evidence."*
The agent will process up to 8 channels at a time, show which channels were
deferred, and propose thresholds from archive P/N/M signals. Calibration is still
a candidate signal — inspect representative frames before applying changes.
`NOT SAFE TO APPLY`, `weak_separation`, or `manual frame review required` can be
the correct result when the archive evidence is weak.

## 6. Settings / Admin

Server, Search, Models, Advanced, plus user/role/audit for admins. See
[admin_guide](../admin/admin_guide.md). Most demo operation needs none of this.
For alert behavior, **Alert Criteria** is the plain-language watch policy
(`alert_policy_prompt`); it is separate from the L0 description prompt and the
structured alert-output template.

## 7. Run your first investigation in 5 minutes

1. **Video tab** → pick a busy street/square channel → confirm it's running and
   has no coverage gaps.
2. **Agent tab** → ask: *"What happened on channel `N` in the last 2 hours?"* →
   read the timeline + coverage line, open an evidence frame.
3. Ask: *"Search the archive on channel `N` for any fight or dangerous driving in
   that window."* → open a top match → ask the agent to **describe** it.
4. Ask: *"Across active channels, where was the most concerning activity in the
   last hour?"* → note the ranked channels.
5. If something real is found, check whether a **bookmark** appears in Luxriot Evo.

That's a complete loop: perceive → ask → search → confirm → evidence.

## 8. Troubleshooting (quick)

| Symptom | Do this |
|---|---|
| Agent says "nothing found" | Widen the window; confirm the channel had coverage; retry scoped by channel+time |
| Answer feels partial | Read the coverage line; ask it to continue / narrow scope |
| Agent slow/timeout | VLM/agent hosts may be busy; avoid bursts of fresh `describe_frame`; prefer read-only "what happened" |
| Channel looks quiet | Check stream health — is it quiet or blind (gaps)? |
| Bookmark not in Luxriot | Bookmark delivery can fail; the in-EVA alert/evidence is still there — tell an admin to check delivery metrics |

For the live demo specifically, follow [demo_runbook](demo_runbook.md).
