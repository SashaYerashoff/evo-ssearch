# Luxriot EVA AI β 0.8.3 Internal Release Notes

Release date: 2026-07-02  
Release type: road-event / vector-signal / live-signal stabilization build  
Previous baseline: `β 0.8.2.1`  
Schema head: `20260614_0006`  
Database migration: **none**

`β 0.8.3` is the first road-event perception foundation build. It extends the
office-demo hardening line with lightweight road-motion CV, vector-signal
conditioning for L0 video descriptions, Luxriot video ingest smoke tooling, and
honest live-preview behavior when Luxriot channels are down or frozen.

This build does **not** claim legal traffic violation detection. Road outputs are
candidate/evidence/navigation signals for human review.

## What Changed Since β 0.8.2.1

### Vector Signals Into L0 Video Descriptions

- L0 video-description prompts can now receive compact `VECTOR_SIGNALS_JSON`
  cues from active CLIP probes and lightweight road-CV motion analysis.
- Vector signals are explicitly marked as secondary attention/arousal signals,
  not visual proof. The VLM must verify them against current frames before
  emitting normal `ALERTS_JSON`.
- Agent video-summary/status tools expose compact vector-signal metadata so the
  agent can explain why a batch received extra attention.

### Road-Event B+ Foundation

- Added `road_events/` primitives for:
  - decoded video frames;
  - scene cards and motion zones;
  - automatic scene-card bootstrap from motion history;
  - optical-flow / frame-motion cues;
  - episode candidate aggregation.
- Added smoke/diagnostic scripts for local video, Luxriot snapshot sampling,
  Luxriot live-video sampling, archive-video sampling, and scene calibration.
- Added road-event tests for scene cards, motion analysis, calibration, episode
  behavior, and Luxriot API helper paths.

### Luxriot Video Ingest And Motion Cues

- Live-summary capture can use Luxriot live-segment video sampling in addition
  to snapshot capture.
- Road-CV batch signals can be computed from the current L0 batch and passed to
  VLM as bounded cues.
- New configuration knobs cover live-segment size/FPS, vector-signal limits,
  and road-CV cue budgets.

### Road Mask Grounding UI

- Engineer/admin video monitoring UI can request a road grounding overlay from
  fresh EVA-captured frames.
- The overlay shows inferred road-motion zone and expected flow when available.
- The panel is diagnostic: it supports grounding and review, not automatic
  enforcement.

### Honest Live Preview / Signal Loss

- `/luxriot/recent_frame/<channel>` now serves only fresh EVA-captured frames by
  default.
- Stale buffers return typed JSON errors instead of replaying old frames:
  `no_eva_frame` / `signal_lost`.
- Capture sessions now detect exact repeated-frame freezes. If a source keeps
  returning the same frame beyond the configured threshold, the runtime marks
  `frozen_signal=true`, drops repeated frozen frames, and `/luxriot/recent_frame`
  returns `signal_frozen`.
- UI preview clears the old image and renders `Signal lost` or `Signal frozen`
  instead of continuing to display stale/frozen history.
- Road mask grounding also refuses stale/frozen EVA buffers.
- Agent live-status inventory now exposes `runtime_problem_channels` for
  stale/frozen/error/stopped capture issues, including runtime channels that do
  not have candidate summaries in the requested window.
- Agent live-status rows include live signal state, recent-frame counts,
  capture source, frozen/stale flags, and current model labels where available.

### Live Capture Stability Addendum

- Live capture no longer blocks on slow VLM batch processing. Per-channel
  summary dispatch now uses a bounded latest-wins queue, so fresh frames keep
  reaching the UI/model even while older batches are still being summarized.
- Added short Luxriot snapshot and live-segment read timeouts to prevent one
  stalled capture call from freezing the live preview loop.
- `auto` capture now fails over from snapshot to live-segment when snapshot
  capture is unavailable for a channel.
- Live-segment failures use a short backoff, reducing repeated ffmpeg/API churn
  on disabled or unauthorized channels.
- Stream status now exposes summary queue depth, inflight state, live-segment
  backoff, and dropped latest-wins batches for diagnostics.

### Offline Deployment Package

- Added split offline patch runbooks:
  - `readiness/OFFLINE_USB_01_PREPARE_MEDIA_RU.md`
  - `readiness/OFFLINE_USB_02_PREFLIGHT_DECISION_RU.md`
  - `readiness/OFFLINE_USB_03_INSTALL_AND_TEST_RU.md`
- Added physical client topology docs:
  - `readiness/CLIENT_PHYSICAL_TOPOLOGY_0.8.3_RU.md`
  - `readiness/CLIENT_PHYSICAL_TOPOLOGY_0.8.3.svg`
- `scripts/build_patch_bundle.sh` includes `preflight_patch.sh` in generated
  bundles when present.

### Live Validation

Live validation on the dev machine before packaging:

- `/health` reachable, `/ready` reached all required components except expected
  local deployment-security warning.
- LM Studio model list reachable and includes `qwen3.5-9b-mtp`.
- Luxriot channel inventory reachable: `pixel9a`, `Zenbook webcam`, `stream`,
  `emu-1`, `emu-2`.
- `emu-1` disabled in Luxriot returns `no_eva_frame` / `no_fresh_eva_frames`
  instead of image replay.
- `emu-2` live drift loop:
  - `/luxriot/recent_frame/120` returned fresh JPEG;
  - `/road/scene_overlay/120` returned a road grounding overlay;
  - live VLM drift smoke passed: VLM alert JSON and archived `vlm_alert`
    evidence thumbnail were produced.
- Agent LM live smoke passed:
  - runtime status question called `list_video_summary_channels`;
  - runtime-problem smoke reported `emu-1` snapshot 404 as a live signal error;
  - documentation/how-to question called `lookup_help`.
- Additional live-capture soak with active preview polling showed bounded
  buffers (`recent_frame_count=36`, `summary_queue_depth<=2`) and no linear
  memory growth over the smoke window.

## Upgrade Notes

- Code-only patch from `β 0.8.2.1`.
- No Alembic migration.
- Restart the service after update.
- Because backend runtime, `app.js`, `app.css`, and `templates/index.html`
  changed, reload the browser / Luxriot EVO Monitor web tile after restart.
- If `/etc/eva-ai/eva-ai.env` or a local `.env` sets `EVOSSEARCH_APP_VERSION`,
  update it to:

```env
EVOSSEARCH_APP_VERSION="β 0.8.3"
```

## New / Changed Configuration

Important new knobs:

- `EVOSSEARCH_LUXRIOT_CAPTURE_SOURCE`
- `EVOSSEARCH_LUXRIOT_LIVE_SEGMENT_SECONDS`
- `EVOSSEARCH_LUXRIOT_LIVE_SEGMENT_MB`
- `EVOSSEARCH_LUXRIOT_LIVE_SEGMENT_EVERY_N`
- `EVOSSEARCH_LUXRIOT_LIVE_SEGMENT_FPS`
- `EVOSSEARCH_LUXRIOT_CAPTURE_REQUEST_TIMEOUT_SEC`
- `EVOSSEARCH_LUXRIOT_LIVE_SEGMENT_READ_TIMEOUT_SEC`
- `EVOSSEARCH_LUXRIOT_SUMMARY_QUEUE_MAX_BATCHES`
- `EVOSSEARCH_LUXRIOT_VECTOR_SIGNALS_ENABLED`
- `EVOSSEARCH_LUXRIOT_VECTOR_SIGNAL_PROBE_LIMIT`
- `EVOSSEARCH_LUXRIOT_VECTOR_SIGNAL_TOP_HITS`
- `EVOSSEARCH_LUXRIOT_ROAD_CV_BATCH_SIGNALS`
- `EVOSSEARCH_LUXRIOT_ROAD_CV_BATCH_MAX_FRAMES`
- `EVOSSEARCH_LUXRIOT_ROAD_CV_BATCH_MAX_EDGE`
- `EVOSSEARCH_LUXRIOT_RECENT_FRAME_MAX_AGE_SEC`
- `EVOSSEARCH_LUXRIOT_FROZEN_FRAME_MAX_SEC`
- `EVOSSEARCH_LUXRIOT_FROZEN_FRAME_MIN_COUNT`
- `EVOSSEARCH_ROAD_CV_*`

See `docs/00_CANON/config_reference.md`.

## Minimum Verification

```bash
bash scripts/check_docs_drift.sh
.venv/bin/python -m py_compile \
  agent.py config.py luxriot_connector.py oldapp.py wsgi.py
node --check static/js/app.js
.venv/bin/python -m pytest -q \
  tests/test_road_events.py \
  tests/test_luxriot_api_extensions.py \
  tests/test_vlm_alert_contract.py \
  tests/test_api_dataflow_smoke.py \
  tests/test_http_auth_routes.py \
  tests/test_luxriot_inference_runtime.py
```

Optional live smoke on a dev/demo box with Luxriot and LM Studio:

```bash
EVA_LIVE_BASE_URL=https://127.0.0.1:5443 \
EVA_LIVE_USER=admin EVA_LIVE_PASSWORD='[FIELD]' \
EVA_LIVE_INCLUDE=drift \
EVA_LIVE_DRIFT_CHANNEL_ID=120 \
.venv/bin/python -m pytest -q tests/integration/test_live_vlm_drift_emu2.py -s
```

## Manual Regression Focus

Use `readiness/MANUAL_TEST_SCENARIO_0.8.3_CUMULATIVE_RU.md`.

Highest-priority checks:

- disabled Luxriot channel shows `Signal lost` / `Signal frozen`, not buffered
  replay;
- emu/street road channel can produce road grounding overlay;
- drift/burnout candidate can be found in archive and produces evidence;
- agent report remains video-description-first and coverage-aware;
- probe tuning remains preview/apply gated and does not bypass UI approval;
- road outputs stay candidate/evidence wording, not legal conclusions.

## Known Limits

- Events shorter than 1 second are out of scope. Movement events generally need
  at least 2 frames; 3+ seconds is the practical minimum for candidate quality.
- Road scene auto-calibration is diagnostic. Wrong-way semantics require stable
  scene geometry and dominant flow; otherwise the system must report degraded
  generic motion/drift candidates only.
- Vector signals are not proof. They are attention cues for VLM and agent
  investigation.
- Luxriot/LM Studio live smoke depends on local demo media and can be slow.
