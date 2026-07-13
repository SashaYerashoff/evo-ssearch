# Configuration Reference

Canonical list of deployment-relevant environment variables. **Source of truth
for behavior is `config.py` and the direct consumers named below**; this table is
the human reference. Secrets
(`*_PASSWORD`, `*_API_KEY`, `*_DSN`, `*_ADMIN_TOKEN`) live only in the on-host
`.env` (mode `0600`) and are **never** committed or placed in shareable docs.

Defaults shown are the code defaults, not the pilot values. Pilot/field values
live in the internal field-rollout doc with `[FIELD]` markers.

Last reviewed: 2026-07-11 (β 0.8.3 stabilization RC)

## Secure-pilot required set

These must be set for a secure deployment (see release notes):

```env
EVOSSEARCH_SECURE_DEPLOYMENT_REQUIRED=true
EVOSSEARCH_AUTH_ENABLED=true
EVA_DB_STRICT_RUNTIME_ROLES=true
EVOSSEARCH_ARCHIVE_STORE=postgres
EVOSSEARCH_EMBEDDER=clip
EVOSSEARCH_DINO_SEGMENTS_ENABLED=false
EVOSSEARCH_EXPERIMENTAL_EMBEDDERS_ENABLED=false
EVOSSEARCH_GUNICORN_WORKERS=1
EVOSSEARCH_GUNICORN_THREADS=8
EVOSSEARCH_AUTH_COOKIE_SECURE=true   # when TLS terminates at app or proxy
```

## Database / DSN (secrets — `.env` only)

| Var | Purpose |
|---|---|
| `EVA_DATABASE_DSN` | API role DSN |
| `EVA_AUDIT_DATABASE_DSN` | Audit role DSN |
| `EVA_WORKER_DATABASE_DSN` | Worker role DSN (required if local inference workers enabled) |
| `EVA_DB_STRICT_RUNTIME_ROLES` (`false`) | Enforce separated runtime roles |

## Server & deployment

| Var (default) | Notes |
|---|---|
| `EVOSSEARCH_HOST` (`0.0.0.0`) | Bind host |
| `EVOSSEARCH_PORT` (`5000`) | Internal Gunicorn HTTP port. TLS is provided by reverse proxy/TLS boundary, not by this variable |
| `EVOSSEARCH_DEBUG` (`false`) | Keep false in prod |
| `EVOSSEARCH_APP_VERSION` | Overrides `VERSION` only if set; keep in sync with release |
| `EVOSSEARCH_SECURE_DEPLOYMENT_REQUIRED` | Gate for secure-mode checks + single-worker enforcement |
| `EVOSSEARCH_GUNICORN_WORKERS` (`1`) | Must stay `1` |
| `EVOSSEARCH_GUNICORN_THREADS` (`8`) | HTTP request threads inside the single required worker. Eight leaves capacity for bounded live-media responses plus Agent/status traffic |
| `EVOSSEARCH_SETTINGS_LOCAL_ONLY` (`true`) | Restrict settings writes |
| `EVOSSEARCH_CONFIG_ENV_FILE` | Absolute path declaration for Settings precedence/provenance, normally identical to systemd `EnvironmentFile`. It does not load or retarget the Settings editor by itself |
| `EVOSSEARCH_SITE_TIMEZONE` (`UTC`) | Optional neutral fallback for agent calendar normalization. Omit it unless the deployment explicitly configures a timezone; operator-facing UI uses the browser timezone without displaying a location label. |

## Auth

| Var (default) | Notes |
|---|---|
| `EVOSSEARCH_AUTH_ENABLED` (`false`) | Must be `true` in pilot |
| `EVOSSEARCH_AUTH_TENANT_ID` | Tenant UUID |
| `EVOSSEARCH_AUTH_COOKIE_SECURE` (`true`) | Requires the browser-facing URL to be HTTPS. Use `false` only for HTTP-only lab/demo |
| `EVOSSEARCH_AUTH_SESSION_TTL_HOURS` (`12`) | Session lifetime |
| `EVOSSEARCH_AUTH_SESSION_COOKIE` / `_CSRF_COOKIE` | Cookie names |
| `EVOSSEARCH_ADMIN_TOKEN` | **Legacy**; not the current auth model |

## Luxriot integration (password is a secret)

| Var (default) | Notes |
|---|---|
| `EVOSSEARCH_LUXRIOT_BASE_URL` | Luxriot Evo host `[FIELD]` |
| `EVOSSEARCH_LUXRIOT_USERNAME` / `_PASSWORD` | Credentials `[FIELD]` |
| `EVOSSEARCH_LUXRIOT_DEFAULT_CHANNEL_ID` (`1`) | Default channel |
| `EVOSSEARCH_LUXRIOT_SNAPSHOT_INTERVAL` (`5`) | Capture cadence (s). Pilot uses aggressive values `[FIELD]` — see sizing |
| `EVOSSEARCH_LUXRIOT_SNAPSHOT_MAX_EDGE` (`800`) | Snapshot max edge px |
| `EVOSSEARCH_LUXRIOT_CAPTURE_SOURCE` (`auto`) | `snapshot`, `live_segment`, or automatic fallback. A true intra-second CV apex requires `live_segment` |
| `EVOSSEARCH_LUXRIOT_LIVE_SEGMENT_SECONDS` (`60`) | Bounded lifetime of one incremental dense-capture pipe. Summaries are emitted inside the window; the longer lease amortizes recorder-open latency |
| `EVOSSEARCH_LUXRIOT_LIVE_SEGMENT_FPS` (`3`) | Raw dense candidates per source-second before one CV apex is selected |
| `EVOSSEARCH_LUXRIOT_CAPTURE_REQUEST_TIMEOUT_SEC` (`5`) | Short timeout for per-frame snapshot capture; prevents stale UI when Luxriot stalls |
| `EVOSSEARCH_LUXRIOT_LIVE_SEGMENT_READ_TIMEOUT_SEC` (`5`) | HTTP read timeout passed to ffmpeg live-segment capture |
| `EVOSSEARCH_LUXRIOT_MAX_BUFFER_FRAMES` (`180`) | Per-channel frame buffer cap |
| `EVOSSEARCH_LUXRIOT_RECENT_FRAME_MAX_AGE_SEC` (`45`) | Max age for UI live-preview EVA frames; stale buffers render as signal loss instead of replay |
| `EVOSSEARCH_LUXRIOT_FROZEN_FRAME_MAX_SEC` (`20`) | Exact repeated-frame duration before a live source is marked frozen |
| `EVOSSEARCH_LUXRIOT_FROZEN_FRAME_MIN_COUNT` (`3`) | Minimum identical captured frames before frozen-source detection can trigger |
| `EVOSSEARCH_LUXRIOT_MEDIA_CONNECT_TIMEOUT_SEC` (`3.0`) | Same-origin media broker connect timeout; clamped to 0.25–30 s |
| `EVOSSEARCH_LUXRIOT_MEDIA_READ_TIMEOUT_SEC` (`8.0`) | Same-origin media broker upstream read timeout; clamped to 0.5–60 s |
| `EVOSSEARCH_LUXRIOT_ARCHIVE_PREPARE_TIMEOUT_SEC` (`90.0`) | First-byte timeout while Evo assembles a browser-compatible archive fragment; clamped to 15–180 s |
| `EVOSSEARCH_LUXRIOT_LIVE_MEDIA_MAX_SECONDS` (`120.0`) | Maximum lifetime of one bounded live broker response; UI renews it proactively at 75%; clamped to 1–120 s |
| `EVOSSEARCH_LUXRIOT_LIVE_MEDIA_MAX_BYTES` (`268435456`) | Maximum streamed bytes in one live broker response (256 MiB default and cap; data is relayed rather than retained) |
| `EVOSSEARCH_LUXRIOT_ARCHIVE_MEDIA_MAX_SECONDS` (`45.0`) | Maximum lifetime of one bounded archive broker response; clamped to 1–300 s |
| `EVOSSEARCH_LUXRIOT_ARCHIVE_MEDIA_MAX_BYTES` (`134217728`) | Maximum bytes in one archive broker response (128 MiB default; clamp 1 KiB–512 MiB) |
| `EVOSSEARCH_LUXRIOT_AUTO_BOOKMARKS` (`false`) | Push alerts as Luxriot bookmarks |
| `EVOSSEARCH_LUXRIOT_BOOKMARK_COOLDOWN_SEC` (`60`) | Dedup cooldown |
| `EVOSSEARCH_LUXRIOT_ALERT_DEDUPE_WINDOW_SEC` (`600`) | Per-channel bookmark delivery dedupe by normalized alert title + severity; `0` disables, clamped to 0–86400 s. Alert records remain in history/archive |
| `EVOSSEARCH_LUXRIOT_ALERTS_MAX_PER_BATCH` (`8`) | Max alerts per batch |
| `EVOSSEARCH_LUXRIOT_SEV_*` | Severity token mapping to Luxriot |

Live media uses Evo's server-side `addStreamToken` / `retrieveLiveStreamByToken`
flow with a secret-safe direct-Digest fallback. When summaries are running, the UI
defaults to the shared `/luxriot/attention_stream/<channel>` model view so operator
preview does not open a second recorder stream; `Full live` is an explicit opt-in.

## Video-description / summaries

| Var (default) | Notes |
|---|---|
| `EVOSSEARCH_LUXRIOT_SUMMARY_RETENTION_DAYS` (`7`) | Retention for hot L0 runtime history; older L0 text may remain reconstructable from archive rows |
| `EVOSSEARCH_LUXRIOT_ROLLUP_RETENTION_DAYS` (`archive row retention`, normally `90`) | Retention for independently queryable PostgreSQL L1–L3 rows |
| `EVOSSEARCH_LUXRIOT_SUMMARY_HISTORY_LIMIT` | Per-channel history cap |
| `EVOSSEARCH_LUXRIOT_SUMMARY_STATE_HOT_LIMIT` (`2160`) | Bounded hot L0 rows per channel (about 6 h at 12 frames × 1 fps plus margin); older evidence stays in archive and closed context survives as L1–L3 |
| `EVOSSEARCH_LUXRIOT_SUMMARY_ARCHIVE_FRAMES_PER_BATCH` (`4`) | Frames archived per batch for search |
| `EVOSSEARCH_LUXRIOT_SUMMARY_QUEUE_MAX_BATCHES` (`2`) | Per-channel VLM summary backlog cap; live capture keeps refreshing while older queued batches may be dropped under load |
| `EVOSSEARCH_LUXRIOT_ROLLUP_L1_WINDOW_SEC` (`900`) | L1 aggregation window and proactive cadence (15 min) |
| `EVOSSEARCH_LUXRIOT_ROLLUP_L2_WINDOW_SEC` (`3600`) | L2 aggregation window and proactive cadence (60 min) |
| `EVOSSEARCH_LUXRIOT_ROLLUP_L3_WINDOW_SEC` (`21600`) | L3 aggregation window and proactive cadence (6 h) |
| `EVOSSEARCH_LUXRIOT_ROLLUP_SCHEDULER_ENABLED` (`true`) | Build closed L1–L3 windows in the background instead of waiting for the first operator view |
| `EVOSSEARCH_LUXRIOT_ROLLUP_SCHEDULER_INITIAL_DELAY_SEC` (`30`) | Startup grace before staggered rollup work begins |
| `EVOSSEARCH_LUXRIOT_ROLLUP_SCHEDULER_SPACING_SEC` (`5`) | Minimum start-to-start spacing between channel/level rollup jobs (inference time counts toward it); deterministic channel phases spread fleet load and LM admission keeps interactive work ahead of rollups |
| `EVOSSEARCH_LUXRIOT_ROLLUP_SCHEDULER_BACKFILL_WINDOWS` (`2`) | Maximum newest missing windows synthesized by one scheduled level job while cached windows are skipped |
| `EVOSSEARCH_LUXRIOT_ROLLUP_SCHEDULER_MAX_DEFERRAL_WINDOWS` (`2`) | Maximum time, in target-level windows, that a saturated L0 queue may defer rollups before one job is admitted anyway; `0` disables deferral |
| `EVOSSEARCH_LUXRIOT_ROLLUP_BACKFILL_SPACING_SEC` (`10`) | Minimum pause between post-upgrade historical restoration windows; live backlog still pauses the worker completely |
| `EVOSSEARCH_LUXRIOT_ROLLUP_BACKFILL_MAX_ATTEMPTS` (`3`) | Bounded semantic retries per historical window before recording a failed gap and continuing |
| `EVOSSEARCH_LUXRIOT_ROLLUP_BACKFILL_ESTIMATE_SEC` (`45`) | Initial per-window ETA estimate until the durable worker measures the deployed LM |
| `EVOSSEARCH_LUXRIOT_ROLLUP_LLM_LEVELS` (`L1,L2,L3`) | Which levels get LLM synthesis |
| `EVOSSEARCH_LUXRIOT_ROLLUP_LLM_MODEL` (`agent` profile when configured) | Text-only L1–L3 model/profile selector; intentionally independent of each live channel's VLM selector |
| `EVOSSEARCH_LUXRIOT_ROLLUP_TIME_ONLY` (`true`) | Window labeling |
| `EVOSSEARCH_LUXRIOT_ALERTS_JSON_PROMPT` / `_SYSTEM_PROMPT_DEFAULT` | Prompt templates |
| `EVOSSEARCH_LUXRIOT_ALERT_POLICY_PROMPT` (`empty`) | Optional default operator alert criteria appended separately from role/summary prompt |
| `EVOSSEARCH_LUXRIOT_STATE_TRANSITIONS_ENABLED` (`true`) | Backend diff of L0 current-observed-state rows across batches |
| `EVOSSEARCH_LUXRIOT_STATE_TRANSITION_CONFIRM_BATCHES` (`2`) | Confirmation hysteresis before appearance/disappearance is emitted |
| `EVOSSEARCH_LUXRIOT_STATE_TRANSITION_ALERT_EVENTS` (`true`) | Store confirmed transitions as internal VLM alert events/evidence; does not send Luxriot bookmarks by itself |

## LM profiles (API keys are secrets)

| Var (default) | Notes |
|---|---|
| `EVOSSEARCH_LM_PROFILES` | e.g. `agent,vlm` |
| `EVOSSEARCH_LM_PROFILE_<ID>_*` | Per-profile base URL / model / timeout / kind `[FIELD]` |
| `EVOSSEARCH_LM_MAX_INFLIGHT` (`1`) | Endpoint-scoped in-process admission capacity fallback; clamped to 1–64 and valid only with the required single Gunicorn worker |
| `EVOSSEARCH_LM_PROFILE_<ID>_MAX_INFLIGHT` | Per-profile admission override, falling back to `EVOSSEARCH_LM_MAX_INFLIGHT`; profiles sharing an endpoint use the smallest configured capacity |
| `EVOSSEARCH_LM_VLM_BALANCER_ENABLED` | Static channel→profile routing across multiple VLM hosts |
| `EVOSSEARCH_LM_VIDEO_DEFAULT_FRAMES` / `_MAX_FRAMES` | Offline/video-description frame limits |
| `EVOSSEARCH_LM_VIDEO_MAX_EDGE` | Resize max edge before sending images to VLM |
| `EVOSSEARCH_LM_VIDEO_MAX_TOKENS` / `_TEMPERATURE` | VLM output sampling limits |
| `EVOSSEARCH_LM_VIDEO_INPUT_WARNING_CHARS` (`24000`) | Warning threshold for text-side VLM/rollup input payloads |
| `EVOSSEARCH_LM_VIDEO_IMAGE_PAYLOAD_WARNING_CHARS` (`2500000`) | Warning threshold for base64 image payload size |
| `EVOSSEARCH_LM_VIDEO_CONTEXT_TOKENS_WARN` (`7000`) | Rough VLM context estimate (chars/4 + ~300 visual tokens per image) that adds an `llm_input_stats` warning before the model truncates; align with the serving `--max-model-len` |

## Agent context budget

| Var (default) | Notes |
|---|---|
| `EVOSSEARCH_AGENT_CONTEXT_LIMIT_TOKENS` (`65536`) | Actual context served by the agent model; the inference server must expose the same or a larger context |
| `EVOSSEARCH_AGENT_MAX_OUTPUT_TOKENS` (`2048`) | Reserved maximum final-answer budget |
| `EVOSSEARCH_AGENT_CONTEXT_CHARS_PER_TOKEN` (`3`) | Conservative JSON/tool-result token estimator divisor |
| `EVOSSEARCH_AGENT_CONTEXT_HISTORY_BUDGET_TOKENS` (`16000`) | Old chat history budget before trimming |
| `EVOSSEARCH_AGENT_CONTEXT_WARNING_TOKENS` (`52000`) | Adds an internal compact-answer warning; includes tool-schema estimates during tool decisions |
| `EVOSSEARCH_AGENT_CONTEXT_HARD_TOKENS` (`60000`) | Stops further tool use and compacts tool payloads before the final model call |

## Inference queue (disabled by default)

| Var (default) | Notes |
|---|---|
| `EVOSSEARCH_INFERENCE_QUEUE_ENABLED` (`false`) | Durable summary queue; keep off until validated |
| `EVOSSEARCH_INFERENCE_QUEUE_CAPACITY` (`200`) | Max queued batches |
| `EVOSSEARCH_INFERENCE_WORKER_COUNT` (`0`) | Local worker threads |
| `EVOSSEARCH_INFERENCE_QUEUE_TENANT_ID` / `_SPOOL_DIR` | Tenant + spool |

## Frame archive & retention

| Var (default) | Notes |
|---|---|
| `EVOSSEARCH_ARCHIVE_STORE` (`postgres`) | Must be postgres in pilot |
| `EVOSSEARCH_ARCHIVE_TENANT_ID` | Tenant UUID |
| `EVOSSEARCH_ARCHIVE_MAX_RECORDS` (`5000000`) | Row cap. **Raise for 2-week window** (see sizing) |
| `EVOSSEARCH_ARCHIVE_ROW_RETENTION_DAYS` (`90`) | Row time-retention |
| `EVOSSEARCH_ARCHIVE_THUMBNAIL_RETENTION_DAYS` (`14`) | Thumbnail retention |
| `EVOSSEARCH_ARCHIVE_RETENTION_PRUNE_INTERVAL_SEC` (`3600`) | Prune cadence |
| `EVOSSEARCH_ARCHIVE_ESTIMATE_*` | Capacity-estimator inputs |

## Probes & detections archive

| Var (default) | Notes |
|---|---|
| `EVOSSEARCH_PROBE_MAX_FRAMES` (`2000`) | Per-channel probe buffer |
| `EVOSSEARCH_PROBE_THUMB_MAX_EDGE` (`256`) | Probe thumbnail size |
| `EVOSSEARCH_PROBE_BOOKMARK_*` | Probe bookmark cooldown/dedup/thresholds |
| `EVOSSEARCH_LUXRIOT_VECTOR_SIGNALS_ENABLED` (`true`) | Feed compact CLIP/road-CV attention cues into L0 video-description prompts |
| `EVOSSEARCH_LUXRIOT_VECTOR_SIGNAL_PROBE_LIMIT` (`6`) | Max active channel probes scanned per L0 batch for vector/homeostasis cues |
| `EVOSSEARCH_LUXRIOT_VECTOR_SIGNAL_TOP_HITS` (`2`) | Max live CLIP hits considered per probe signal |
| `EVOSSEARCH_DETECTIONS_ARCHIVE_ENABLED` (`true`) | Persist detection frames |
| `EVOSSEARCH_DETECTIONS_RETENTION_*` | Dedup/keep windows + similarity thresholds |

## Capture apex decider (per-second CV frame selection)

| Var (default) | Notes |
|---|---|
| `EVOSSEARCH_LUXRIOT_CAPTURE_BURST_ZSCORE` (`6.0`) | Robust z-score over the channel's own measured activity baseline that marks a second as `burst` (motion peak wins outright, blur expected) |
| `EVOSSEARCH_LUXRIOT_CAPTURE_ACTIVITY_NOISE_FLOOR` (`0.004`) | Mean-abs grayscale delta below this is sensor noise, not motion; such seconds are `quiet` and ship the sharpest frame |
| `EVOSSEARCH_LUXRIOT_CAPTURE_SELECTOR_BIAS` (`auto`) | Site default for the decider: `auto` (adaptive per-channel baseline), `action` (always motion peak), `clarity` (always sharpest). Per-channel override lives in channel prompt settings |

The decider classifies every capture second as `quiet`/`normal`/`burst` relative to
the channel's persisted motion baseline (homeostasis). `normal` seconds ship the
sharpest frame of the action band; `burst` seconds ship the motion peak and may
attach one sharper companion frame of the same second (archived as
`burst_companion`, offered to the VLM as one extra labeled snapshot). Burst/normal
markers reach the model via `VECTOR_SIGNALS_JSON.capture_attention`.

## Road CV primitives (experimental)

| Var (default) | Notes |
|---|---|
| `EVOSSEARCH_LUXRIOT_ROAD_CV_BATCH_SIGNALS` (`true`) | Adds bounded road-motion cues to L0 vector signals from the current batch |
| `EVOSSEARCH_LUXRIOT_ROAD_CV_BATCH_MAX_FRAMES` (`24`) | Max frames sampled per L0 batch for road-CV cue extraction |
| `EVOSSEARCH_LUXRIOT_ROAD_CV_BATCH_MAX_EDGE` (`240`) | Max edge used for L0 batch road-CV cue extraction |
| `EVOSSEARCH_LUXRIOT_ROAD_SCENE_CALIBRATION_SAMPLES` (`8`) | Per-channel auto-scene samples required before high-confidence frozen road direction can enable wrong-way/cross-flow cues |
| `EVOSSEARCH_ROAD_CV_ENABLED` (`false`) | Reserved for dedicated road-event CV runners |
| `EVOSSEARCH_ROAD_CV_SCENE_CARDS` (`empty`) | JSON scene-card path with channel road zones and expected flow vectors |
| `EVOSSEARCH_ROAD_CV_MAX_EDGE` (`360`) | Max edge for motion analysis frames |
| `EVOSSEARCH_ROAD_CV_MIN_MOTION_PX` (`0.7`) | Optical-flow magnitude threshold for active pixels |
| `EVOSSEARCH_ROAD_CV_ACTIVE_RATIO_FLOOR` (`0.012`) | Minimum moving-pixel ratio inside a road zone |
| `EVOSSEARCH_ROAD_CV_WRONG_WAY_ALIGNMENT` (`-0.45`) | Cosine alignment threshold for opposing-flow candidates |

## Embedder / vision

| Var (default) | Notes |
|---|---|
| `EVOSSEARCH_EMBEDDER` (`clip`) | Production embedder |
| `EVOSSEARCH_PRODUCTION_CLIP_MODEL` / `EVOSSEARCH_CLIP_MODEL` (`ViT-B/32`) | CLIP model |
| `EVOSSEARCH_RERANK_ENABLED` / `_TOP_K` | Re-rank toggle |
| `EVOSSEARCH_DINO_*`, `EVOSSEARCH_M2F_*`, `EVOSSEARCH_FUSION_*` | Experimental; disabled in prod |
| `EVOSSEARCH_INDEXED_FOLDER_ENABLED` / `_OFFLINE_VIDEO_ENABLED` / `_PROBE_SNAP_ENABLED` (`false`) | Legacy/hidden feature flags |

## Feature flags off in client pilot

`EVOSSEARCH_OFFLINE_VIDEO_ENABLED=false`, `EVOSSEARCH_PROBE_SNAP_ENABLED=false`,
`EVOSSEARCH_INDEXED_FOLDER_ENABLED=false` — these return 404 server-side and hide
the corresponding UI.
