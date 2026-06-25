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

- Log in with your named account. Your **role** (admin / engineer / operator /
  viewer) and **channel grants** decide what you see. If a tab or channel is
  missing, you don't have the grant — ask an admin.
- The workspace has tabs: **Video**, **Archive Research**, **Monitoring**,
  **Agent**. For public-order monitoring you mostly live in **Video** and
  **Agent**.

## 2. Video tab — the live perception

This is where the system's always-on description of each channel lives.

- **Live Stream Control** — pick a channel; see its live preview.
- **Stream context** — channel, cadence, batch size, model, queue, probe state.
- **VLM Feed / L0–L3** — the running **video-descriptions** (L0 = live per-batch)
  and the rollups (L1/L2/L3 = 15 min / 1 h / 6 h summaries). Collapsed summary
  rows show **alert badges** when something was flagged.
- **Channel Runtime / stream health** — whether the channel is actively
  producing descriptions and whether there are **coverage gaps / dropped
  batches**. A channel that is *quiet* is fine; a channel that is *blind* (gaps)
  is not — trust the health panel, not silence.

What to do here: confirm your demo channels are running, skim recent L1/L2 for
context, and note any alert badges before you ask the agent.

## 3. Agent tab — ask in plain language

This is the main investigation surface.

- **Ask with a period and a channel.** "What happened on channel 12 in the last 2
  hours?" Natural time works ("yesterday evening", "last 90 minutes").
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

## 5. Monitoring / Probes (mostly agent-driven now)

Probes are CLIP matchers. In this deployment they're primarily the agent's
bulk-search tool; as an operator you'll mainly **read** probe hits if shown. You
can view the Probe Board; creating/tuning probes is an engineer/agent task. (A
curated watch-list probe set may be cast on demo channels as a safety-net
detector — see the demo runbook.)

## 6. Settings / Admin

Server, Search, Models, Advanced, plus user/role/audit for admins. See
[admin_guide](../admin/admin_guide.md). Most demo operation needs none of this.

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
