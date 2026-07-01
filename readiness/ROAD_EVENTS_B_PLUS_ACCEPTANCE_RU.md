# Road Events B+ Acceptance Track

Target date: Friday before the July 10 demo rehearsal.

Goal: EVA should help an operator find archived traffic incidents and surface fresh road-event candidates quickly enough for manual confirmation. The system must not promise sub-second events or legal conclusions.

## Scope Boundaries

- Events shorter than 1 second are out of scope for this pipeline.
- For movement events, useful evidence normally needs at least 2 frames, so 3+ seconds is the practical minimum for reliable candidate detection.
- CV/probe/VLM outputs are candidate signals. Final wording must stay visual: wrong-way candidate, aggressive vehicle motion candidate, burnout/drift candidate, smoke/skid candidate.
- Bookmarks are evidence/navigation aids, not enforcement findings.

## B+ Acceptance Criteria

- A road scene card can describe at least one channel with a road polygon and expected flow direction.
- A scene card can be bootstrapped automatically from bounded motion history when no human-authored card exists.
- The CV layer can decode video frames and emit bounded motion cues for:
  - road motion burst;
  - opposing-flow candidate;
  - cross-flow/aggressive-motion candidate.
- The episode layer can combine CV, CLIP, and VLM cues into a bounded candidate episode with confidence `low|medium|high`.
- The archive workflow can use existing VLM summaries/probes plus new road cue contracts to produce 2-3 candidate moments with evidence links.
- Fresh events can be turned into a bookmark candidate within roughly 60-90 seconds after enough cues exist.
- Operator-facing reports must state coverage, gaps, and signal provenance.

## Core Track

Owner: Sasha + Codex.

- Keep CV primitives independent from the live L0/VLM hot path until the runner is validated.
- Wire Luxriot video/RTSP runner only behind `EVOSSEARCH_ROAD_CV_ENABLED=false` default.
- Add scene-card loader, automatic scene bootstrap, motion analyzer, and episode aggregator tests before runtime integration.
- Add probe templates for public-road visual cues: tire smoke, burnout/skid marks, vehicle crossing lane flow, vehicle stopped in intersection, near collision.
- Keep confidence conservative: single CV cue is `low`; repeated CV cues are `medium`; multi-source confirmation is `high`.
- Treat automatic scene cards as candidate geometry:
  - `medium` only when a stable motion zone and dominant flow are inferred;
  - `low` when a motion zone exists but dominant flow is weak or multi-directional;
  - degraded full-frame mode disables wrong-way semantics.

## Automatic Scene Bootstrap Budget

Manual polygon painting is not a product path. For an unmapped channel, the
system should spend a bounded bootstrap budget:

- sample 30-120 decoded frames from recent archive or live video;
- ignore hard scene cuts and pairs with excessive whole-frame motion;
- build a motion-history heatmap;
- infer one conservative motion zone plus expected flow only if flow direction is dominant;
- otherwise keep a degraded full-frame card with `expected_flow=null`.

If the bootstrap cannot produce at least `low` confidence in the available
budget, the channel remains usable for generic motion/drift candidates but not
for wrong-way or lane-direction claims. The operator report must say this
explicitly.

For client streams with real daily traffic, run archive calibration over a
longer window. This samples short windows across the day and aggregates only
stable motion geometry/flow:

```bash
.venv/bin/python scripts/road_cv_luxriot_calibrate_scene.py \
  --channel <street-channel> \
  --mode archive-video \
  --hours 24 \
  --samples 24 \
  --window-sec 12 \
  --frames-per-sample 72 \
  --every-n 6 \
  --output /tmp/eva-road-scene-cards.json
```

Acceptance rule: use wrong-way semantics only when calibration reports stable
zone geometry and dominant flow (`confidence=high` with `expected_flow`). If it
returns a calibrated zone without flow, use it for generic road-motion/drift
candidates only.

## Ivan Track

Owner: Ivan, with review by Sasha.

- Prepare 5-10 public-road scene cards from office demo channels:
  - channel id/title;
  - visible road/intersection polygon;
  - expected vehicle flow vector;
  - notes about occlusion, reflections, night lighting, traffic density.
- Prepare manual QA checklist:
  - archive query for drift/burnout;
  - archive query for wrong-way traffic;
  - archive query for aggressive intersection behavior;
  - fresh-event bookmark verification;
  - false-positive log for rain, headlights, camera shake, shadows.
- Keep UI work simple: expose scene-card status and latest road candidates; do not redesign core archive views during this track.

## Functional Smoke

No real drift sample is required for the first smoke. A webcam or any moving
video source is enough to verify decoding, motion cue creation, bounded status,
and episode grouping:

```bash
.venv/bin/python scripts/road_cv_smoke.py \
  --source 0 \
  --channel-id 999 \
  --max-frames 120
```

For a Luxriot/RTSP source, pass the URL as `--source`. Without a scene-card file
the script uses a full-frame zone, which is enough for functionality but not for
wrong-way semantics. Use `--scene-cards path/to/scenes.json` to test expected
flow and road zones.

If only Luxriot snapshots are available, use the snapshot smoke against a live
channel:

```bash
.venv/bin/python scripts/road_cv_luxriot_snapshot_smoke.py \
  --channel emu-1 \
  --frames 60 \
  --interval-sec 0.5 \
  --save-dir /tmp/eva-road-emu1
```

For Luxriot live video, use the authenticated live MP4 API. This path samples
multiple adjacent frames from the same stream and is the preferred functional
check for movement events:

```bash
.venv/bin/python scripts/road_cv_luxriot_api_smoke.py \
  --channel emu-1 \
  --mode live-video \
  --frames 120 \
  --every-n 6 \
  --segment-mb 8 \
  --segment-sec 20
```

For archived investigation, prefer `archive-video` when the channel stores MP4
archive chunks. Use `archive-snapshots` only when the channel's archived JPEG
snapshot endpoint works reliably:

```bash
.venv/bin/python scripts/road_cv_luxriot_api_smoke.py \
  --channel emu-1 \
  --mode archive-video \
  --last-minutes 10 \
  --auto-scene \
  --auto-scene-frames 60 \
  --frames 120 \
  --every-n 6 \
  --segment-mb 16 \
  --segment-sec 30
```

The smoke script also prints archive boundaries. If boundaries are empty, Luxriot
has no recorded archive for that channel yet, even if the live stream is visible.

Edited multi-angle clips are useful for verifying `scene_cut_count`: hard cuts
must reset the motion analyzer and must not become road-motion episodes. They
are not sufficient to validate continuous drift/wrong-way motion unless the
decoder samples multiple consecutive frames from the same camera angle.

## Demo Operator Script

Suggested queries:

- "Show recent road-event candidates on street channels for the last 24 hours."
- "Find possible burnout or drift events and provide visual evidence."
- "Check if any vehicles moved against the expected direction on channel X today."
- "Summarize aggressive intersection behavior candidates and separate confirmed evidence from weak cues."

Expected answer shape:

- reviewed window and coverage;
- candidate list with time/channel/event type/confidence;
- signal provenance: CV motion, CLIP probe, VLM alert/summary;
- evidence links or next action to inspect frames;
- explicit limitations if coverage or cue strength is weak.

## Not In B+ Yet

- Legal violation classification.
- License plate or person identification.
- City-scale 1000+ channel optimization.
- Codec motion-vector backend as the default; OpenCV frame decoding is the first backend.
- Persistent auto-scene registry and UI approval workflow.
