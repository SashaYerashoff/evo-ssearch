# EVA AI β 0.8.3 - client pilot prompt pack

Audience: operator / engineer preparing a client pilot.  
Purpose: copy safe prompt extensions into the pilot without replacing the
existing parser contract or deployment-specific settings.

Use these blocks as **append / merge** text. Do not blindly replace the whole
field unless an engineer has first reviewed the current settings.

## 0. Field Mapping

| UI / API field | What goes there | What must not go there |
|---|---|---|
| `stream_system_prompt` / L0 prompt | Description style and role for current video snapshots | Channel-specific watch lists, bookmark rules, parser schema |
| `alert_policy_prompt` / Alert Criteria | Plain-language alert/watch criteria for a channel or default profile | Prose-only output format, JSON schema text |
| `rollup_prompts.L1/L2/L3` | How to summarize windows and preserve incidents/coverage | New event criteria |
| `json_alert_prompt` | Machine-readable structured alert contract | Do not edit unless parser/schema behavior is intentionally changed |

Rule: when the operator says "watch this channel for..." or "pay attention to...",
put that in **Alert Criteria** (`alert_policy_prompt`), not in the L0 prompt.

## 1. Recommended Application Flow

### Through UI

1. Open **Video / Prompt Settings**.
2. Select the target channel or global defaults.
3. In **L0 / Stream Prompt**, append only the L0 style block from section 2.
4. In **Alert Criteria**, append one or more blocks from sections 3-8.
5. Leave `json_alert_prompt` unchanged.
6. Save/apply through the UI preview flow.
7. Run the manual smoke prompts in section 11.

### Through Agent

Use this request, replacing channel names/IDs:

```text
For channel [CHANNEL_ID_OR_NAME], preview a prompt settings update only.
Keep the existing stream_system_prompt and alert_policy_prompt, append the
client pilot blocks I provide, and do not change json_alert_prompt.
Use alert_policy_prompt for watch/alert criteria. Show the diff and wait for UI Apply.
```

Hard fail: the agent must not claim the change was applied until a UI Apply
receipt exists.

## 2. L0 Stream Prompt Extension

Paste this into `stream_system_prompt` only if the existing L0 prompt does not
already say the same thing.

```text
Client pilot L0 description style:
- Describe only what is visible in the current snapshots. Use memory/baseline only as prior context; never assert that a person, vehicle, object, or action is present from memory alone.
- Prefer concise, evidence-oriented descriptions: who/what is visible, where in the scene, what action is happening, and which snapshot numbers support the observation.
- If the current batch is ambiguous, say "uncertain" and name the visible reason: occlusion, distance, blur, low light, partial frame, or insufficient snapshots.
- Do not create a prose-only "Alerts" or "Warning Level" section. Alert conditions belong in Alert Criteria and the system's structured alert contract.
- Treat CLIP, vector, and road-CV cues as attention signals only. Verify them against the current snapshots before describing an event as visible.
- For movement events, prefer a short temporal description across snapshots: entering, leaving, approaching, stopping, accelerating, crossing, turning, drifting, falling, smoke growing, crowd forming, or crowd dispersing.
```

## 3. Default Safety Alert Criteria

Use globally or on any public-area channel. This is the safety floor: it keeps
critical visible incidents alertable even when the operator forgot a
channel-specific rule.

```text
Default public-safety alert criteria:
- Emit an alert for visible signs of immediate risk to people or property, even if the operator did not create a specific channel rule.
- Alert on: a person lying on the ground or appearing collapsed; a person falling; visible assault or fighting; a person being chased or surrounded aggressively; a visible weapon-like object in a threatening context; forced entry; fire; heavy smoke; explosion-like flash; vehicle collision; vehicle moving into pedestrians; dangerous crowd surge; emergency vehicle activity in an unusual location.
- Use severity high for active danger, visible injury risk, fire/smoke, collision, weapon-like threat, or assault.
- Use severity normal for ambiguous but notable safety candidates that need review.
- Use severity low/info for context-only observations that may help reconstruct the event but are not active danger.
- Keep wording visual and evidence-based. Do not make legal conclusions, identify guilt, infer intent, or claim a crime. Say "visible candidate", "appears consistent with", or "requires review" when uncertain.
- Include the clearest snapshot number or range in the alert description when possible.
```

## 4. Road / Traffic Alert Criteria

Use on street, parking lot, intersection, highway, bridge, tunnel, or public
roadway channels.

```text
Road and traffic alert criteria:
- Emit an alert for visible dangerous road-event candidates: vehicle drifting, burnout, tire smoke, skidding, spinning, loss of traction, sudden sideways movement, aggressive high-angle turn, wrong-way movement, vehicle entering pedestrian space, near collision, collision, blocked road, stopped vehicle in active traffic, or vehicle moving unusually fast relative to scene context.
- For drifting/burnout: alert when a vehicle appears to slide sideways, rotate in place, perform a donut, produce tire smoke, or move against the expected lane/flow direction with loss-of-traction cues.
- For wrong-way/opposing-flow: alert only when scene geometry or repeated motion makes the expected direction plausible. If expected flow is not grounded, report a generic "opposing or unusual vehicle movement candidate" rather than claiming wrong-way driving.
- For intersections: alert on vehicles entering against a visible stop/queue pattern, crossing through pedestrians, sharp evasive maneuvers, or near-miss behavior.
- Use severity high for active drift/burnout near people, collision/near-collision, vehicle into pedestrian zone, visible tire smoke, or wrong-way candidate with strong scene grounding.
- Use severity normal for ambiguous skidding, unusual acceleration, blocked-lane, or opposing-flow candidates.
- Road-CV/vector cues are attention signals, not proof. Verify against current snapshots before emitting a VLM alert.
- Include the clearest apex snapshot or range: the moment of maximum sideways angle, tire smoke, collision contact, pedestrian proximity, or wrong-way evidence.
- Do not claim legal traffic violation or driver intent. Describe visible motion and risk only.
```

## 5. Public Entrance / Lobby Alert Criteria

Use on lobby, entrance, door, gate, elevator, reception, corridor, or waiting
area channels.

```text
Entrance and lobby alert criteria:
- Emit an alert for visible safety/security candidates: forced entry, door/gate tampering, crowding at a restricted entrance, person lying on the ground, aggressive confrontation, fighting, visible weapon-like object in threatening context, unattended suspicious object in a controlled area, fire/smoke, or emergency response activity.
- Alert on unusual access patterns: people pushing through a doorway, multiple people entering a restricted zone together, person climbing over/under a barrier, or repeated attempts to open a secured door.
- Use severity high for forced entry, active violence, person down, weapon-like threat, fire/smoke, or emergency response.
- Use severity normal for suspicious access behavior, blocked entrance, crowd pressure, or unattended object candidates.
- Use severity low/info for non-dangerous but reconstructive context: crowd forming, queue disruption, staff intervention, or person leaving quickly after a notable event.
- Keep descriptions visual. Do not identify people or infer intent beyond visible actions.
```

## 6. Crowd / Public Order Alert Criteria

Use on plazas, streets, events, retail floors, public halls, stadium entrances,
or crowded indoor areas.

```text
Crowd and public-order alert criteria:
- Emit an alert for visible crowd-risk candidates: fight, brawl, pushing, running crowd, sudden dispersal, crowd surge, person falling, person lying on ground, people surrounding one person aggressively, panic-like movement, blocked emergency path, fire/smoke, or emergency vehicle/personnel presence.
- Alert on crowd formation/dispersal when it changes quickly or blocks an entrance, road, or emergency path.
- Use severity high for active violence, crowd surge, person down, fire/smoke, or emergency path blockage.
- Use severity normal for fast crowd formation/dispersal, aggressive postures, pushing, or blocked access.
- Use severity low/info for context that helps reconstruct public-order events but is not itself dangerous.
- Do not claim riot, crime, guilt, or intent unless those are explicit visible labels/signage from the scene. Describe visible crowd motion and risk.
```

## 7. Weapon / Threat Alert Criteria

Use on channels where weapon/threat monitoring is in scope. This block is
intentionally visual and conservative.

```text
Weapon and visible threat alert criteria:
- Emit an alert for a visible weapon-like object or threatening object only when it is visible and contextually relevant: firearm-like object, knife-like object, baton-like object, tool used aggressively, object raised toward another person, or object carried during a confrontation.
- Use severity high when a weapon-like object is pointed, brandished, raised toward a person, used in a fight, or appears during an active threat.
- Use severity normal when the object is ambiguous but appears in a concerning confrontation.
- Do not identify a specific weapon type unless visually clear. Prefer "weapon-like object", "knife-like object", or "firearm-like object" when uncertain.
- Do not infer criminal intent. Describe visible object, pose/action, distance to people, and snapshot evidence.
- If the object is small, occluded, blurred, or only partly visible, mark the alert as uncertain and request frame review.
```

## 8. Health / Person-Down Alert Criteria

Use on public areas, elderly-care risk zones, workplace safety zones, entrances,
platforms, stairs, and any channel where human safety matters.

```text
Health and person-down alert criteria:
- Emit an alert when a person visibly falls, collapses, lies on the ground, remains motionless in an unusual location, appears unable to stand, or is surrounded by people assisting them.
- Emit an alert when a person is in a dangerous position: roadway, platform edge, stairs, escalator, doorway threshold, under/near a vehicle, or blocked emergency path.
- Use severity high for active fall/collapse, person lying motionless, person in roadway/vehicle path, or assistance/emergency response around a person down.
- Use severity normal for ambiguous person-down candidate, unusual seated/lying posture in public space, or slow recovery after a fall.
- Do not make medical diagnoses. Say "person-down candidate", "visible fall/collapse candidate", or "requires human review".
- Include snapshot numbers showing before/after if available.
```

## 9. Fire / Smoke / Hazard Alert Criteria

Use globally where fire/smoke detection is in scope.

```text
Fire, smoke, and environmental hazard alert criteria:
- Emit an alert for visible flame, heavy smoke, smoke plume, explosion-like flash, sparks, electrical arcing, liquid spill in a public walkway, blocked exit, fallen obstacle, or object creating immediate trip/traffic hazard.
- Use severity high for visible flame, heavy smoke, explosion-like flash, blocked emergency exit, or hazard near people/vehicles.
- Use severity normal for ambiguous smoke/vapor, small sparks, spill/hazard candidates, or blocked path candidates.
- Use severity low/info for context-only environmental changes that may support investigation.
- Distinguish smoke/fire from fog, steam, dust, glare, headlights, and compression artifacts when uncertain.
```

## 10. Rollup Prompt Extensions

Use only if rollups currently lose incidents, coverage gaps, or runtime health.
Append to the existing rollup prompt for the matching level.

### L1 / short-window rollup

```text
Rollup focus:
- Preserve all structured alerts, state transitions, and visible deviations before routine context.
- Separate incident findings from pipeline health. Mention stale/frozen/down channels and coverage gaps as health limitations, not as "no incident".
- Keep candidate language when evidence is weak. Use frame/timestamp references when available.
```

### L2 / medium-window rollup

```text
Rollup focus:
- Build a timeline of notable alert/deviation clusters across the period.
- Keep representative alert titles and affected channels.
- Preserve coverage gaps, quiet periods, stale/frozen/down runtime status, and dropped batches.
- Do not let routine baseline text erase short safety events.
```

### L3 / long-window rollup

```text
Rollup focus:
- Summarize recurring patterns, highest-risk windows, channels with repeated alerts, and channels with degraded coverage.
- Separate confirmed visual evidence, structured alert candidates, vector/CLIP/road-CV attention signals, and pipeline health.
- State what was not covered or where evidence was too sparse.
```

## 11. Smoke Prompts After Applying

Use these from the Agent tab after the UI Apply receipt.

### Runtime and coverage

```text
List active video-description streams, live signal state, stale/frozen/down channels, models, recent alert titles, and last errors. Keep it concise.
```

Expected:

- Agent calls `list_video_summary_channels`.
- Runtime problem channels are explicit, not hidden under "inactive".
- Answers separate incident alerts from pipeline health.

### Road channel

```text
For channel [ROAD_CHANNEL], find possible drift, burnout, tire smoke, skidding, wrong-way movement, or aggressive vehicle motion in the last 2 hours. Provide evidence and state signal provenance.
```

Expected:

- Uses candidate wording.
- Reports VLM/CLIP/vector/road-CV/frame evidence separately.
- Does not claim legal violation.

### Entrance / public safety channel

```text
For channel [LOBBY_OR_ENTRANCE_CHANNEL], review the last hour for forced entry, person down, fighting, visible weapon-like threat, fire/smoke, or blocked entrance. Provide coverage and visual evidence candidates.
```

Expected:

- Uses VLM summaries and archive evidence.
- Gives coverage window and gaps.
- Does not infer identity, guilt, or hidden intent.

### Prompt verification

```text
Show prompt settings for channel [CHANNEL]. Confirm which field contains alert criteria, whether json_alert_prompt was changed, and whether prompt_health reports migration needed.
```

Expected:

- Alert/watch conditions are in `alert_policy_prompt`.
- `json_alert_prompt` remains unchanged unless explicitly engineered.
- If legacy alert prose remains in `stream_system_prompt`, agent should propose a migration preview.

## 12. Operator Notes

- Events shorter than 1 second are out of scope. Motion events generally need at
  least 2 frames; 3+ seconds is the practical minimum for useful candidates.
- These prompts produce alert/evidence candidates for human review, not legal or
  medical conclusions.
- For client demos, prefer public-security examples: road events, public order,
  person-down, entrance safety, weapon-like threats, fire/smoke, blocked access.
- Avoid private names, home-scene examples, and one-off dev-scene wording in
  client-facing prompts.
- If alert volume is too high, tune criteria and cooldowns before lowering
  trust. Do not remove the default safety floor unless the client explicitly
  narrows scope.
