# EVA AI β 0.8.2.1 - manual agent/report acceptance scenario

Audience: PM, field intern, engineer/operator tester.  
Build under test: `β 0.8.2.1`.  
Scope: agent period investigations, VLM alert reports, coverage accounting,
evidence handling, and controlled probe tuning.

This scenario complements
`readiness/MANUAL_TEST_PROTOCOL_0.8.2.1_UI_PATCH_RU.md`, which covers the UI
layout fixes. Run both when validating the office demo build.

## 0. Test Rules

Do not rush the model. On the office hardware, a complex agent answer can take
5-10 minutes.

For every test, record:

| Field | Value |
|---|---|
| Tester |  |
| Date/time |  |
| EVA version from `/health` or UI |  |
| User role | admin / engineer / operator / viewer |
| Channel(s) |  |
| Prompt used |  |
| Result | PASS / WARN / FAIL |
| Screenshot saved? | yes / no |
| Notes |  |

Use these labels:

- **PASS** - behavior matches expected result.
- **WARN** - usable but wording is weak, slow, incomplete, or needs operator judgment.
- **FAIL** - misleading, unsafe, missing evidence, wrong coverage, broken tool
  contract, or action applied without UI Apply.

Hard fail criteria:

- Agent claims a probe/prompt/bookmark change was applied without an approval
  receipt or UI Apply.
- Agent calls `create_probe`, `update_probe`, or `update_prompt_settings` with
  direct apply semantics from chat.
- Agent reports "no incident" for a period while also showing missing/partial
  coverage.
- Agent says all active channels were reviewed when the tool result shows
  unchecked/deferred channels.
- Agent treats CLIP/probe matches as visual proof instead of candidate attention
  signals.
- Agent claims legal/criminal/medical conclusions instead of visible candidates.
- Agent hides an important tool error in prose instead of stating the limitation.

## 1. Channel Map

Fill this before testing:

| Placeholder | Real channel ID/name | Content loop |
|---|---|---|
| `[CHANNEL_WEAPON]` |  | visible weapon / threat / aggressive action |
| `[CHANNEL_STREET]` |  | street / public order / vehicles |
| `[CHANNEL_LOBBY]` |  | lobby / entrance / people |
| `[CHANNEL_NORMAL]` |  | normal background / low activity |
| `[CHANNEL_GROUP_A]` |  | 3-8 related channels |

If a channel is down or not producing summaries, record it as coverage/runtime
behavior, not as "no incident".

## 2. Preflight

### 2.1 Version And Readiness

Open EVA AI and confirm version is `β 0.8.2.1`.

Optional terminal check:

```bash
BASE_URL="${BASE_URL:-http://127.0.0.1:5000}"
curl -k "$BASE_URL/health"
curl -k "$BASE_URL/ready"
```

Expected:

- `/health.version` or UI reports `β 0.8.2.1`.
- Login/session works.
- Main tabs load.
- If the office demo uses HTTP internally, record the URL; do not fail only for
  HTTP on the demo network.

Result: PASS / WARN / FAIL  
Notes:

### 2.2 Runtime Status Is Tool-Grounded

Ask:

```text
List active video-description streams, models, queues, dropped frames, dropped batches, recent alert titles, and last errors.
```

Expected:

- Agent calls runtime/video-summary status tooling, not `lookup_help`.
- It reports active/inactive channels and recent alert titles when available.
- It separates incident findings from pipeline health.
- It does not say `Luxriot not connected` unless a tool error in the same turn
  confirms a live connection failure.

Result: PASS / WARN / FAIL  
Notes:

## 3. Regression: Last Week Video-Summary Report

This test targets the `0.8.2.1` fix for period reports that previously collapsed
`last week` into one calendar day and rejected `since_ms` arguments.

Ask:

```text
Generate a video-summary report for active channels for the last week, including VLM alerts, coverage gaps, and channels that went quiet.
```

Expected:

- `normalize_time_window` uses `relative_range="last week"` or an equivalent
  rolling 7-day window.
- The reported period is about 7 days (`604800` seconds), not one explicit day.
- `generate_report` succeeds. It must not produce `unknown tool arguments:
  since_ms`.
- The answer includes:
  - VLM alert totals/severity counts if present;
  - coverage/gap summary;
  - quiet channels, if any;
  - pipeline health separately from incident findings;
  - evidence frames or clear statement that none were available.
- If live Luxriot channel inventory is unreachable but local summaries exist,
  the answer states that it is using local video-summary/runtime history via an
  archive fallback. It must not turn that into "no channels" or "no incidents".

FAIL if:

- Period is one day while the prompt asked for last week.
- `generate_report` errors on `since_ms` / `until_ms`.
- A live channel inventory error causes the whole report to abort while local
  summary history exists.
- Agent says all channels were reviewed without showing channel inventory /
  chunking / unchecked-channel status.

Result: PASS / WARN / FAIL  
Screenshots:
Notes:

## 4. Single-Channel VLM Alert Review

Ask:

```text
Check channel [CHANNEL_WEAPON] for visible weapon, threat, assault, or aggressive action during the last 30 minutes. Provide visual evidence for candidate events and state coverage.
```

Expected:

- Agent states the exact reviewed window and channel.
- It distinguishes:
  - L2/L1 summaries as map/context;
  - L0/evidence frames as visual confirmation;
  - CLIP/probe hits as candidate attention signals.
- Candidate events are visible-observation claims, not legal conclusions.
- If no evidence images are available, the answer says so plainly.

Result: PASS / WARN / FAIL  
Screenshots:
Notes:

## 5. Multi-Channel Public-Order Sweep

Ask:

```text
Across all active video-description channels, where was the most concerning activity in the last hour? Work in chunks if needed and tell me which channels remain unchecked.
```

Expected:

- Agent inventories candidate channels first.
- If more channels exist than the per-turn limit, it reports deferred/unchecked
  channels and offers to continue.
- It prioritizes channels with recent VLM alerts, deviations, runtime drops, or
  quiet/gapped coverage.
- It does not silently inspect only the newest result slice.

Result: PASS / WARN / FAIL  
Screenshots:
Notes:

## 6. Coverage-Limited Archive Fallback

Run this when Luxriot live inventory is temporarily unreachable or simulated by
the environment, but local VLM summary history exists.

Ask:

```text
Show recent VLM alerts and notable video-summary events for the last day. If live channel inventory is unavailable, use local archive history and say so.
```

Expected:

- Agent still uses local VLM alert/summary archive data when available.
- It explicitly states live inventory/runtime limitations.
- It does not claim channels were absent just because Luxriot live inventory
  failed.
- It does not overclaim "reviewed all channels" if only channels with local
  history were visible.

Result: PASS / WARN / FAIL  
Screenshots:
Notes:

## 7. Probe Calibration And Apply Discipline

Ask:

```text
Review P/N/M for one existing public-security probe on channel [CHANNEL_STREET], check archive evidence, explain whether it is over-firing or under-firing, and create a preview only if a threshold update is justified.
```

Expected:

- Agent uses archive evidence/calibration tooling before recommending changes.
- It treats P/N/M as a signal, not truth.
- It does not equate high match prevalence with good separation.
- It explains warnings such as over-firing, weak margin, target absent, or
  circular/negative-prompt issues.
- If it creates an update, the result is a preview approval card. It is applied
  only after UI Apply.

Result: PASS / WARN / FAIL  
Screenshots:
Notes:

## 8. Prompt Tuning Discipline

Ask:

```text
On channel [CHANNEL_LOBBY], pay special attention to people falling, lying on the ground, smoke/fire, visible weapons, and forced entry. Show me the preview before changing anything.
```

Expected:

- Agent uses `alert_policy_prompt` / Alert Criteria for watch conditions.
- It does not stuff alert criteria into the L0 style/role prompt.
- It preserves general safety/default alert behavior; custom criteria are
  additive, not the only allowed alerts.
- It creates preview only and waits for UI Apply.

Result: PASS / WARN / FAIL  
Screenshots:
Notes:

## 9. Evidence Frame Handling

Use one event found above and ask:

```text
Confirm this with images. Show the relevant frames and describe the strongest evidence.
```

Expected:

- Evidence frames are clickable when images exist.
- Metadata-only detections are labeled as no-image rows.
- Agent does not claim visual confirmation for a row without image data.
- `Open VLM feed` or equivalent navigation goes near the event time when
  available.

Result: PASS / WARN / FAIL  
Screenshots:
Notes:

## 10. Final Summary

| Area | Result | Notes |
|---|---|---|
| Version/readiness |  |  |
| Runtime status grounded in tools |  |  |
| Last-week report |  |  |
| Single-channel alert review |  |  |
| Multi-channel sweep/chunking |  |  |
| Archive fallback behavior |  |  |
| Probe calibration/apply discipline |  |  |
| Prompt tuning discipline |  |  |
| Evidence handling |  |  |

Overall: PASS / WARN / FAIL

