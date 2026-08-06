# Configuration Reference

Canonical list of deployment-relevant environment variables. **Source of truth
for behavior is `config.py` and the direct consumers named below**; this table is
the human reference. Secrets
(`*_PASSWORD`, `*_API_KEY`, `*_DSN`, `*_ADMIN_TOKEN`) live only in the on-host
`.env` (mode `0600`) and are **never** committed or placed in shareable docs.

Defaults shown are the code defaults, not the pilot values. Pilot/field values
live in the internal field-rollout doc with `[FIELD]` markers.

Last reviewed: 2026-08-06 (β 0.8.7 release preparation)

## Secure-pilot required set

These must be set for a secure deployment (see release notes):

```env
EVOSSEARCH_SECURE_DEPLOYMENT_REQUIRED=true
EVOSSEARCH_AUTH_ENABLED=true
EVA_DB_STRICT_RUNTIME_ROLES=true
EVOSSEARCH_ARCHIVE_STORE=postgres
EVOSSEARCH_EMBEDDER=clip
EVOSSEARCH_DINO_SEGMENTS_ENABLED=false
EVOSSEARCH_EXPERIMENTAL_EMBEDDERS_ENABLED=true
EVOSSEARCH_CLIP_MODEL=google/siglip2-base-patch16-224
EVOSSEARCH_CLIP_MODEL_REVISION=75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2
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
| `EVOSSEARCH_UI_MODE` (`legacy`) | Browser shell: `legacy` or `react`. During parity soak, `/?ui=react` and `/?ui=legacy` provide a per-request override without restarting EVA. React production assets are prebuilt; Node.js is not required on the appliance |
| `EVOSSEARCH_SECURE_DEPLOYMENT_REQUIRED` | Gate for secure-mode checks + single-worker enforcement |
| `EVOSSEARCH_GUNICORN_WORKERS` (`1`) | Must stay `1` |
| `EVOSSEARCH_GUNICORN_THREADS` (`8`) | HTTP request threads inside the single required worker. Eight leaves capacity for bounded live-media responses plus Agent/status traffic |
| `EVOSSEARCH_SETTINGS_LOCAL_ONLY` (`true`) | Restrict settings writes |
| `EVOSSEARCH_TRUSTED_PROXY_HOPS` (`0`) | Number of reverse-proxy hops trusted for client IP, scheme, and host. Keep `0` when EVA is directly reachable; the clean appliance installer binds EVA to loopback and sets `1` for its local proxy |
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
| `EVOSSEARCH_LUXRIOT_FFMPEG_HWACCEL` (`auto`) | `auto` probes QSV, then Intel VAAPI, and uses the first working hardware decode/VPP path; any channel-level failure is retried in software. Use `qsv`, `vaapi`, or `software`/`off` to force a guarded backend |
| `EVOSSEARCH_LUXRIOT_FFMPEG_INTEL_DEVICE` (auto-discovered) | Optional Intel DRM render node such as `/dev/dri/renderD128`. Auto-discovery verifies PCI vendor `0x8086` and never selects the NVIDIA render node. The legacy `...QSV_DEVICE` name remains a compatibility alias |
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

## Direct local V4L2 video sources (optional)

| Var (default) | Notes |
|---|---|
| `EVOSSEARCH_LOCAL_VIDEO_SOURCES_JSON` (`[]`) | JSON array of local USB/V4L2 cameras. Each row accepts `id`, `title`, `device`, `input_format`, `width`, `height`, `fps`, and `preview_fps` |

Example: `[{"id":900001,"title":"Direct USB webcam","device":"/dev/video0","input_format":"mjpeg","width":1280,"height":720,"fps":15,"preview_fps":8}]`.
The channel ID must not collide with an Evo channel. The service account must be
able to read the configured `/dev/videoN` device (normally through the `video`
group), and FFmpeg must be available from the bundled runtime or
`EVOSSEARCH_FFMPEG_BIN`. Local channels support live preview, snapshots, EVA/VLM
capture, probes, summaries, and locally retained alert evidence. They do not
provide recorder archive playback or an Evo bookmark destination.

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
| `EVOSSEARCH_LUXRIOT_ROLLUP_L3_WINDOW_SEC` (`28800`) | L3 aggregation window and proactive cadence (8 h) |
| `EVOSSEARCH_LUXRIOT_ROLLUP_SCHEDULER_ENABLED` (`true`) | Build closed L1–L3 windows in the background instead of waiting for the first operator view |
| `EVOSSEARCH_LUXRIOT_ROLLUP_SCHEDULER_INITIAL_DELAY_SEC` (`30`) | Startup grace before staggered rollup work begins |
| `EVOSSEARCH_LUXRIOT_ROLLUP_SCHEDULER_SPACING_SEC` (`5`) | Minimum start-to-start spacing between channel/level rollup jobs (inference time counts toward it); after the fast startup pass, recurring jobs align to canonical L1/L2/L3 window boundaries and deterministic channel phases spread fleet load |
| `EVOSSEARCH_LUXRIOT_ROLLUP_SCHEDULER_BACKFILL_WINDOWS` (`2`) | Maximum newest missing windows synthesized by one scheduled level job while cached windows are skipped |
| `EVOSSEARCH_LUXRIOT_ROLLUP_SCHEDULER_MAX_DEFERRAL_WINDOWS` (`2`) | Maximum time, in target-level windows, that a saturated L0 queue may defer rollups before one job is admitted anyway; `0` disables deferral |
| `EVOSSEARCH_LUXRIOT_ROLLUP_SCHEDULER_MAX_DEFERRAL_SEC` (`180`) | Absolute ceiling on that L0 deferral. This keeps L1 semantic windows from starving on a continuously busy shared model; `0` leaves only the window-based ceiling |
| `EVOSSEARCH_LUXRIOT_ROLLUP_BACKFILL_SPACING_SEC` (`10`) | Minimum pause between post-upgrade historical restoration windows; live backlog still pauses the worker completely |
| `EVOSSEARCH_LUXRIOT_ROLLUP_BACKFILL_MAX_ATTEMPTS` (`3`) | Bounded semantic retries per historical window before recording a failed gap and continuing |
| `EVOSSEARCH_LUXRIOT_ROLLUP_BACKFILL_ESTIMATE_SEC` (`45`) | Initial per-window ETA estimate until the durable worker measures the deployed LM |
| `EVOSSEARCH_LUXRIOT_ROLLUP_LLM_LEVELS` (`L1,L2,L3`) | Which levels get LLM synthesis |
| `EVOSSEARCH_LUXRIOT_ROLLUP_LLM_MODEL` (`agent` profile when configured) | Text-only L1–L3 model/profile selector; intentionally independent of each live channel's VLM selector |
| `EVOSSEARCH_LUXRIOT_ROLLUP_L1_MAX_TOKENS` (`768`) / `_L2_` (`1024`) / `_L3_` (`2048`) | Independent text completion budgets for normal L1–L3 synthesis. They do not enlarge the live visual L0 completion budget |
| `EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_ENABLED` (`false`) | Route L3 only to a separate OpenAI-compatible text endpoint. L1/L2 remain on `ROLLUP_LLM_MODEL`; there is no fallback/offload to the live model |
| `EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_BASE_URL` / `_MODEL` / `_API_KEY` | Separate CPU/deep-review endpoint (for example a local llama.cpp 9B server); the key is optional for a local unauthenticated endpoint |
| `EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_CONNECT_TIMEOUT_SEC` (`5`) / `_READ_TIMEOUT_SEC` (`600`) | Bounded connect/read timeouts for deep L3 |
| `EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_MAX_TOKENS` (`3072`) / `_TEMPERATURE` (`0.1`) | Deep-review generation bounds |
| `EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_QUEUE_CAPACITY` (`64`) | Dedicated bounded L3 queue. It has one worker and cannot block live L0 or scheduled L1/L2 |
| `EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_MAX_ATTEMPTS` (`3`) / `_BACKOFF_INITIAL_SEC` (`30`) / `_BACKOFF_MAX_SEC` (`900`) | Bounded retries with exponential backoff; terminal failure retains a deterministic proposal-only L3 closure |
| `EVOSSEARCH_LUXRIOT_ROLLUP_L3_QUIET_WINDOW_ENABLED` (`false`) | Fail-closed operator switch for scheduled deep L3. Enabling deep routing alone does not invent a fixed “night” schedule |
| `EVOSSEARCH_LUXRIOT_ROLLUP_L3_QUIET_WINDOW_TIMEZONE` (`EVOSSEARCH_SITE_TIMEZONE` or `UTC`) / `_START` (`01:00`) / `_END` (`05:00`) / `_DAYS` (all days) | Operator-defined local quiet window. Cross-midnight windows are assigned to the day on which they start |
| `EVOSSEARCH_LUXRIOT_ROLLUP_L3_QUIET_MAX_DEFERRAL_SEC` (`86400`) / `_POLL_SEC` (`60`) | Bound activity-gate deferral and its recheck cadence |
| `EVOSSEARCH_LUXRIOT_ROLLUP_L3_QUIET_MAX_ACTIVITY_X` (`1.5`) / `_ALERT_LOOKBACK_SEC` (`900`) / `_MAX_L0_DEBT` (`0.75`) | Deep L3 runs only with quiet attention, no recent alerts, and bounded L0 coverage debt |
| `EVOSSEARCH_LUXRIOT_ROLLUP_TIME_ONLY` (`true`) | Window labeling |
| `EVOSSEARCH_LUXRIOT_ALERTS_JSON_PROMPT` / `_SYSTEM_PROMPT_DEFAULT` | Prompt templates; the legacy-named alerts field now stores the unified `BATCH_STATE_JSON` contract |
| `EVOSSEARCH_LUXRIOT_ALERT_POLICY_PROMPT` (`empty`) | Optional default operator alert criteria appended separately from role/summary prompt |
| `EVOSSEARCH_LUXRIOT_STATE_TRANSITIONS_ENABLED` (`true`) | Backend diff of L0 current-observed-state rows across batches |
| `EVOSSEARCH_LUXRIOT_STATE_TRANSITION_CONFIRM_BATCHES` (`2`) | Confirmation hysteresis before appearance/disappearance is emitted |
| `EVOSSEARCH_LUXRIOT_STATE_TRANSITION_ALERT_EVENTS` (`true`) | Store confirmed transitions as internal VLM alert events/evidence; does not send Luxriot bookmarks by itself |

The L3 quiet-window object is also available through
`LuxriotManager.get_rollup_l3_deep_schedule()` and can be validated/persisted by
`set_rollup_l3_deep_schedule(...)`. These are backend hooks for a future
authorized operator UI/API; they are intentionally not agent tools. Every L3
row is marked `review_only`, `proposals_only`, and `mutations_applied=false`.
L3 memory/tuning suggestions are retained for review but never alter probes,
thresholds, alert policy, live sampling, or the live routine context.

## LM profiles (API keys are secrets)

| Var (default) | Notes |
|---|---|
| `EVOSSEARCH_LM_PROFILES` | e.g. `agent,vlm` |
| `EVOSSEARCH_LM_PROFILE_<ID>_*` | Per-profile base URL / model / timeout / kind `[FIELD]` |
| `EVOSSEARCH_LM_MAX_INFLIGHT` (`1`) | Endpoint-scoped in-process admission capacity fallback; clamped to 1–64 and valid only with the required single Gunicorn worker. The port appliance writes `8` per agent/VLM profile to match its vLLM `--max-num-seqs 8`; one protected lane remains reserved for interactive/alert work |
| `EVOSSEARCH_LM_PROFILE_<ID>_MAX_INFLIGHT` | Per-profile admission override, falling back to `EVOSSEARCH_LM_MAX_INFLIGHT`; profiles sharing an endpoint use the smallest configured capacity |
| Shared-endpoint protected lane | With capacity >1, EVA keeps one physical request slot free for interactive agent or fast-alert work. L0 and L1–L3 cannot borrow it; capacity-one backends remain serial and do not deadlock |
| LM admission order | On a shared endpoint: interactive agent, realtime alert, live L0, then L1-L3/background rollup. Rollups do not own a protected slot; the port topology should still route them to the separate agent/CPU endpoint |
| `EVOSSEARCH_LUXRIOT_ROLLUP_CONTEXT_LIMIT_TOKENS` (`32768`) | Conservative text-only L1-L3 context ceiling. Oversized source/corrective blocks are middle-compacted before the request while preserving metadata at the head and recent evidence/instructions at the tail |
| `EVOSSEARCH_LUXRIOT_INCIDENT_FOREGROUND_LIMIT` / `_FOREGROUND_HARD_LIMIT` / `_HOT_LIMIT` (`2/4/8`) | Ordinary full-focus incidents, hard model-envelope cap, and unresolved scheduler hot set. Parking is attention-only and never resolves an incident |
| `EVOSSEARCH_LUXRIOT_INCIDENT_TRACKED_LIMIT` (`64`) | Process-local safety ceiling for durable incident leases; exceeding it is explicit backpressure, not silent lifecycle mutation |
| `EVOSSEARCH_INCIDENT_MAINTENANCE_ENABLED` (`true`) | Reconcile expired Follow leases durably even when a channel produces no new L0 batch |
| `EVOSSEARCH_INCIDENT_MAINTENANCE_INTERVAL_SEC` (`15`) | Bounded background reconciliation interval, clamped to 1–300 seconds |
| `EVOSSEARCH_LUXRIOT_L0_CONTEXT_WINDOW_TOKENS` (`16384`) | Measurable L0 context envelope used by the prompt planner |
| `EVOSSEARCH_LUXRIOT_L0_TEXT_BUDGET_TOKENS` / `_VISION_BUDGET_TOKENS` / `_OUTPUT_BUDGET_TOKENS` (`5000/5500/1536`) | Separate L0 budgets. Alert criteria and `BATCH_STATE_JSON` are protected atomic blocks; incident context is semantically compacted first |
| `EVOSSEARCH_LUXRIOT_L0_INCIDENT_BUDGET_TOKENS` (`900`) | Sub-budget shared by at most four incident contexts in an L0 request; incidents 5–8 remain scheduler state only |
| `EVOSSEARCH_LUXRIOT_L0_VISION_TOKENS_PER_IMAGE_ESTIMATE` (`300`) | Conservative accounting estimate per selected frame for telemetry and fail-before-send budget checks |
| `EVOSSEARCH_LM_VLM_BALANCER_ENABLED` | Static channel→profile routing across multiple VLM hosts |
| `EVOSSEARCH_LM_VIDEO_DEFAULT_FRAMES` / `_MAX_FRAMES` | Offline/video-description frame limits |
| `EVOSSEARCH_LM_VIDEO_MAX_EDGE` | Resize max edge before sending images to VLM |
| `EVOSSEARCH_LM_VIDEO_MAX_TOKENS` / `_TEMPERATURE` | VLM output sampling limits |
| `EVOSSEARCH_LM_VIDEO_INPUT_WARNING_CHARS` (`24000`) | Warning threshold for text-side VLM/rollup input payloads |
| `EVOSSEARCH_LM_VIDEO_IMAGE_PAYLOAD_WARNING_CHARS` (`2500000`) | Warning threshold for base64 image payload size |
| `EVOSSEARCH_LM_VIDEO_CONTEXT_TOKENS_WARN` (`7000`) | Rough VLM context estimate (chars/4 + ~300 visual tokens per image) that adds an `llm_input_stats` warning before the model truncates; align with the serving `--max-model-len` |
| `EVOSSEARCH_LM_VISION_HEALTH_STATE_FILE` | Optional content-aware VLM watchdog state. A first suspect result is warning-only when a recent successful canary exists and the watchdog threshold has not been reached; degraded, stale, or never-successful vision quarantines L0 results before memory or alerts |
| `EVOSSEARCH_LM_VISION_HEALTH_MAX_AGE_SEC` (`180`) | Maximum accepted age of a successful dynamic vision canary; stale watchdog state fails the VLM gate closed. |
| `EVOSSEARCH_GUNICORN_GRACEFUL_TIMEOUT` (`20`) | Maximum worker drain time during an appliance restart. Bounds a stuck ffmpeg/embedding child below the systemd stop timeout while preserving state through Gunicorn lifecycle hooks |

## Agent context budget

| Var (default) | Notes |
|---|---|
| `EVOSSEARCH_AGENT_CONTEXT_LIMIT_TOKENS` (`65536`) | Requested EVA-side ceiling. EVA reads `max_model_len` from OpenAI-compatible `/v1/models` when available and budgets against the smaller served/configured value; servers without that field must still be configured to the same or larger context |
| `EVOSSEARCH_AGENT_MAX_OUTPUT_TOKENS` (`2048`) | Reserved maximum final-answer budget |
| `EVOSSEARCH_AGENT_CONTEXT_CHARS_PER_TOKEN` (`3`) | Conservative JSON/tool-result token estimator divisor |
| `EVOSSEARCH_AGENT_CONTEXT_HISTORY_BUDGET_TOKENS` (`16000`) | Old chat history budget before trimming |
| `EVOSSEARCH_AGENT_CONTEXT_WARNING_TOKENS` (`52000`) | Adds an internal compact-answer warning; includes tool-schema estimates during tool decisions |
| `EVOSSEARCH_AGENT_CONTEXT_HARD_TOKENS` (`60000`) | Stops further tool use and compacts tool payloads before the final model call |
| `EVOSSEARCH_AGENT_ARCHIVE_VISION_BATCH_SIZE` (`8`, clamped `6`–`9`) | Number of diverse top SigLIP archive candidates verified together in one multi-image VLM request. This is a bounded visual drill, not exhaustive archive coverage |

## Inference queue

| Var (default) | Notes |
|---|---|
| `EVOSSEARCH_INFERENCE_QUEUE_ENABLED` (`false`) | Durable summary queue. Code default stays off for unconfigured development; the clean appliance installer enables it |
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
| `EVOSSEARCH_PROBE_ROI_QUERY_EMBED_BUDGET` (`2`) | Maximum previously uncached ROI thumbnails encoded by one retrospective probe query. Realtime operator ROI results seed the same cache; bounded cold backfill prevents the 5-second daemon from flooding the shared SigLIP batcher |
| `EVOSSEARCH_PROBE_POS_FLOOR_DEFAULT` (`0.05` for SigLIP2; `0.28` for OpenAI CLIP) | Backend-sensitive raw-cosine floor for newly created and ad-hoc probes. Scores are not transferable between embedding spaces; existing saved probes stay shadowed until their fingerprint matches and thresholds are recalibrated |
| `EVOSSEARCH_PROBE_MARGIN_DEFAULT` (`0.02` for SigLIP2; `0.08` for OpenAI CLIP) | Backend-sensitive positive-minus-negative margin for newly created and ad-hoc probes |
| `EVOSSEARCH_PROBE_CAPTURE_WARMUP_SEC` (`2.5`) | Maximum first-frame wait before an empty manual probe query returns an explicit capture-warming state |
| `EVOSSEARCH_ARCHIVE_DISK_MIN_FREE_GB` (`2.0`) | Stop writing new filesystem snapshots below this free-space floor while continuing metadata rows |
| `EVOSSEARCH_ARCHIVE_DISK_MIN_FREE_PERCENT` (`5.0`) | Stop writing new filesystem snapshots below this filesystem free-space percentage |
| `EVOSSEARCH_PROBE_BOOKMARK_*` | Probe bookmark cooldown/dedup/thresholds |
| `EVOSSEARCH_PROBE_REALTIME_BOOKMARK_ENABLED` (`true`) | Evaluate only operator-authored, bookmark-enabled text probes on each completed 1 Hz semantic apex. ROI text probes use a fresh crop and ROI embedding cache; automatic/VLM-derived and image-reference probes remain on the retrospective daemon and cannot enter this alarm lane |
| `EVOSSEARCH_PROBE_REALTIME_CONFIRM_HITS` (`2`) / `_CONFIRM_WINDOW_SEC` (`3.2`) / `_MAX_EVENT_AGE_SEC` (`5`) | Require repeated current-frame evidence before a direct probe bookmark and reject stale embedding completions. A match exceeding both P and M floors by `_STRONG_SCORE_BOOST` (`0.06`) may pass immediately |
| `EVOSSEARCH_PROBE_REALTIME_WORKERS` (`2`) / `_QUEUE_CAPACITY` (`32`) | Bounded asynchronous scoring/bookmark delivery; saturation drops the acceleration attempt, never the independent semantic archive or normal probe daemon |
| `EVOSSEARCH_PROBE_CHANNEL_GROUPS_FILE` (`probe_channel_groups.json`) | Operator-defined channel groups for the Probes board. File-backed presentation state, not tenant archive data; losing it only un-groups the board and never affects probes |
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
| `EVOSSEARCH_LUXRIOT_CAPTURE_SELECTOR_ENABLED` (`true`) | Site default for adaptive CV frame selection. Operators with prompt-management access can override it per channel in Live Stream Control; disabled channels use a deterministic temporal midpoint and pause homeostasis updates |

The decider classifies every capture second as `quiet`/`normal`/`burst` relative to
the channel's persisted motion baseline (homeostasis). `normal` seconds ship the
sharpest frame of the action band; `burst` seconds ship the motion peak and may
attach one sharper companion frame of the same second (archived as
`burst_companion`, offered to the VLM as one extra labeled snapshot). Burst/normal
markers reach the model via `VECTOR_SIGNALS_JSON.capture_attention`.

## Adaptive live attention

| Var (default) | Notes |
|---|---|
| `EVOSSEARCH_LUXRIOT_SUMMARY_MAX_BATCH_FRAMES` (`16`) | Hard upper bound for saved snapshots in one L0 VLM batch |
| `EVOSSEARCH_LUXRIOT_SUMMARY_MAX_WINDOW_SEC` (`60`) | Hard source/wall-clock deadline for a non-empty L0 batch |
| `EVOSSEARCH_LUXRIOT_SUMMARY_QUIET_CADENCE_SEC` (`5`) | Saved-snapshot cadence admitted to the VLM batch during quiet intervals |
| `EVOSSEARCH_LUXRIOT_SUMMARY_NORMAL_CADENCE_SEC` (`2`) | Saved-snapshot cadence admitted during normal activity |
| `EVOSSEARCH_LUXRIOT_SUMMARY_BURST_CADENCE_SEC` (`1`) | Saved-snapshot cadence admitted during bursts |
| `EVOSSEARCH_LUXRIOT_ATTENTION_SCHEDULER_ENABLED` (`false`) | Enable homeostatic CV/embedding attention telemetry and adaptive L0 frame admission |
| `EVOSSEARCH_LUXRIOT_ATTENTION_EPISODE_DISPATCH_ENABLED` (`false`) | Experimental sparse coordinator-owned VLM dispatch; normally off because L0 delivery is owned by the bounded per-channel batch accumulator |
| `EVOSSEARCH_LUXRIOT_ATTENTION_EMBED_ALL_CHANNELS` (`false`) | Produce CLIP embeddings for every enabled live video channel, independently of alerts and VLM admission |
| `EVOSSEARCH_LUXRIOT_ATTENTION_EMBEDDING_CADENCE_MS` (`1000`) | Durable semantic-index cadence. Port preset invariant is one embedding-backed snapshot per second/channel; changing VLM cadence never changes this archive path |
| `EVOSSEARCH_LUXRIOT_CLIP_ASYNC_ENABLED` (`true`) / `_WORKERS` (`8`) / `_QUEUE_CAPACITY` (`64`) | Bounded decoder-to-CLIP dispatch. Keeps synchronous embedding latency from backpressuring ffmpeg capture; one worker can wait per channel while the shared CLIP batcher combines cross-channel requests |
| `EVOSSEARCH_LIVE_CLIP_BATCH_SIZE` (`8`) / `_BATCH_WAIT_MS` (`75`) / `_BATCH_QUEUE_CAPACITY` (`128`) / `_BATCH_TIMEOUT_SEC` (`15`) | Cross-channel CLIP microbatch execution. Every submitted cadence slot receives one result or an explicit error; batching is not sampling |
| `EVOSSEARCH_SEMANTIC_SNAPSHOT_ARCHIVE_ENABLED` (`true`) | Persist every cadence embedding+thumbnail as `source=semantic_snapshot`, whether or not a probe/alert matched |
| `EVOSSEARCH_SEMANTIC_SNAPSHOT_ARCHIVE_QUEUE` (`512`) / `_BATCH_SIZE` (`32`) | Bounded asynchronous PostgreSQL writer for semantic snapshots; backpressure and write failures are exposed as archive gaps |
| `EVOSSEARCH_LUXRIOT_ATTENTION_STORAGE_ENABLED` (`false`) | Require PostgreSQL attention telemetry (`20260726_0008`) |
| `EVOSSEARCH_LUXRIOT_ATTENTION_RING_SECONDS` (`90`) | Bounded in-memory evidence ring; stores selected embedding frames, never dense CV frames |
| `EVOSSEARCH_LUXRIOT_ATTENTION_REQUESTS_PER_MINUTE` (`6`) | Global VLM token-bucket refill rate across channels |
| `EVOSSEARCH_LUXRIOT_ATTENTION_L0_SLOT_PARALLELISM` (auto: VLM max-inflight minus one, capped at `3`) | Converts measured L0 wall time into effective batched slot cost. The reserved endpoint lane remains available to interactive agent and alert work; set `1` for a serial llama.cpp/CPU endpoint |
| `EVOSSEARCH_LUXRIOT_ATTENTION_MAX_OUTSTANDING` (`1`) | Global queued/in-flight VLM episode limit |
| `EVOSSEARCH_LUXRIOT_ATTENTION_POSTROLL_SEC` (`3`) | Burst post-roll collected before episode dispatch |
| `EVOSSEARCH_LUXRIOT_ATTENTION_MAX_VLM_FRAMES` (`8`) | Maximum saved embedding frames in one episode |
| `EVOSSEARCH_VLM_FAST_ALERT_ENABLED` (`true`) | Run a separate alert-only VLM phase after a measured CV burst; this does not replace or enter the visible full L0 memory stream |
| `EVOSSEARCH_VLM_FAST_ALERT_POST_ROLL_SEC` (`2.5`) / `_MAX_FRAMES` (`6`) / `_MAX_TOKENS` (`128`) | Bound the control/pre/onset/apex/post evidence set and compact completion length. The short post-roll trades 1.5 seconds for enough visual trajectory to distinguish an event from a scene-change edge |
| `EVOSSEARCH_VLM_FAST_ALERT_WORKERS` (`2`) | Admit two independent burst checks concurrently; the global LM admission controller still gives interactive agent work priority and bounds total inference pressure |
| `EVOSSEARCH_VLM_FAST_ALERT_SEMANTIC_DELTA` (`0.22`) / `_MIN_MOVING_FRACTION` (`0.15`) | Also validate a large consecutive SigLIP scene change when CV confirms distributed motion. This catches meaningful changes on continuously active channels whose motion has become baseline; the vector delta only routes frames and is never alert proof |
| `EVOSSEARCH_VLM_FAST_ALERT_COOLDOWN_SEC` (`12`) / `_DEDUPE_WINDOW_SEC` (`12`) | Bound repeated burst passes and suppress an identical fast-phase/full-L0 bookmark replay without suppressing differently titled hazards |
| `EVOSSEARCH_LUXRIOT_ALERT_DERIVED_PROBES_ENABLED` (`false`) | Admit bounded, temporary attention-only probes from direct VLM alerts |
| `EVOSSEARCH_LUXRIOT_ALERT_DERIVED_PROBE_TTL_SEC` (`300`) | TTL for alert-derived probes |

With the attention scheduler enabled, the `port-4070s-8ch` contract overrides
the three legacy cadence/window values above: quiet `10 s / 120 s / 6-8-8`,
watch `5 s / 90 s / 6-8-10`, active `2.5 s / 60 s / 8-12-12`, burst
`1 s / 30 s / 10-16-16`, degraded `15 s / 120 s / 4-6-6`. Every mode retains
the independent 1 Hz semantic archive and the hard 16-frame accumulator cap.

See `docs/architecture/adaptive_attention_runtime.md` for retention, evidence
links, P/N/M semantics, and the deployed office profile.

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
| `EVOSSEARCH_PRODUCTION_CLIP_MODEL` / `EVOSSEARCH_CLIP_MODEL` (`google/siglip2-base-patch16-224`) | CLIP-like semantic model; OpenAI CLIP remains a selectable legacy/A-B backend |
| `EVOSSEARCH_CLIP_MODEL_REVISION` (`75de2d55...`) | Immutable Hugging Face revision included in the embedding-space fingerprint and offline bundle |
| `EVOSSEARCH_CLIP_DEVICE` (`auto`; appliance `cuda`) | Device for the CLIP-like attention embedder. SigLIP2 base on CPU does not sustain the eight-channel 1 Hz target; the single-4070S appliance reserves GPU headroom for it |
| `EVOSSEARCH_EMBEDDER_FALLBACK_ENABLED` (`false`) | Allows an explicit fallback to a different embedding model after load failure. Keep `false` in production: model changes invalidate archive vectors and probe thresholds |
| `EVOSSEARCH_OFFLINE_MODE` (`true`) | Blocks Hugging Face/Transformers and OpenAI CLIP downloads; missing artifacts fail closed |
| `EVOSSEARCH_MODEL_CACHE_DIR` (`~/.cache/eva-ai/models`) | Local Hugging Face/Transformers model cache |
| `EVOSSEARCH_OPENAI_CLIP_CACHE_DIR` (`~/.cache/clip`) | Local OpenAI CLIP weights cache |
| `EVOSSEARCH_RERANK_ENABLED` / `_TOP_K` | Re-rank toggle |
| `EVOSSEARCH_EXPERIMENTAL_EMBEDDERS_ENABLED` (`true`) | Enables the production SigLIP2 path and optional DINO/fusion selectors; DINO/fusion remain disabled separately |
| `EVOSSEARCH_DINO_*`, `EVOSSEARCH_M2F_*`, `EVOSSEARCH_FUSION_*` | Optional; disabled in the port profile |
| `EVOSSEARCH_INDEXED_FOLDER_ENABLED` / `_OFFLINE_VIDEO_ENABLED` / `_PROBE_SNAP_ENABLED` (`false`) | Legacy/hidden feature flags |

## Feature flags off in client pilot

`EVOSSEARCH_OFFLINE_VIDEO_ENABLED=false`, `EVOSSEARCH_PROBE_SNAP_ENABLED=false`,
`EVOSSEARCH_INDEXED_FOLDER_ENABLED=false` — these return 404 server-side and hide
the corresponding UI.
