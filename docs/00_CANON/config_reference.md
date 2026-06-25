# Configuration Reference

Canonical list of deployment-relevant environment variables. **Source of truth
for behavior is `config.py`**; this table is the human reference. Secrets
(`*_PASSWORD`, `*_API_KEY`, `*_DSN`, `*_ADMIN_TOKEN`) live only in the on-host
`.env` (mode `0600`) and are **never** committed or placed in shareable docs.

Defaults shown are the code defaults, not the pilot values. Pilot/field values
live in the internal field-rollout doc with `[FIELD]` markers.

Last reviewed: 2026-06-25 (β 0.8.1)

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
| `EVOSSEARCH_PORT` (`5000`) | Pilot uses `5443` (TLS) `[FIELD]` |
| `EVOSSEARCH_DEBUG` (`false`) | Keep false in prod |
| `EVOSSEARCH_APP_VERSION` | Overrides `VERSION` only if set; keep in sync with release |
| `EVOSSEARCH_SECURE_DEPLOYMENT_REQUIRED` | Gate for secure-mode checks + single-worker enforcement |
| `EVOSSEARCH_GUNICORN_WORKERS` (`1`) | Must stay `1` |
| `EVOSSEARCH_SETTINGS_LOCAL_ONLY` (`true`) | Restrict settings writes |

## Auth

| Var (default) | Notes |
|---|---|
| `EVOSSEARCH_AUTH_ENABLED` (`false`) | Must be `true` in pilot |
| `EVOSSEARCH_AUTH_TENANT_ID` | Tenant UUID |
| `EVOSSEARCH_AUTH_COOKIE_SECURE` (`true`) | Requires HTTPS |
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
| `EVOSSEARCH_LUXRIOT_MAX_BUFFER_FRAMES` (`180`) | Per-channel frame buffer cap |
| `EVOSSEARCH_LUXRIOT_AUTO_BOOKMARKS` (`false`) | Push alerts as Luxriot bookmarks |
| `EVOSSEARCH_LUXRIOT_BOOKMARK_COOLDOWN_SEC` (`60`) | Dedup cooldown |
| `EVOSSEARCH_LUXRIOT_ALERTS_MAX_PER_BATCH` (`8`) | Max alerts per batch |
| `EVOSSEARCH_LUXRIOT_SEV_*` | Severity token mapping to Luxriot |

## Video-description / summaries

| Var (default) | Notes |
|---|---|
| `EVOSSEARCH_LUXRIOT_SUMMARY_RETENTION_DAYS` (`7`) | History retention |
| `EVOSSEARCH_LUXRIOT_SUMMARY_HISTORY_LIMIT` | Per-channel history cap |
| `EVOSSEARCH_LUXRIOT_SUMMARY_ARCHIVE_FRAMES_PER_BATCH` (`4`) | Frames archived per batch for search |
| `EVOSSEARCH_LUXRIOT_ROLLUP_LLM_LEVELS` (`L1,L2,L3`) | Which levels get LLM synthesis |
| `EVOSSEARCH_LUXRIOT_ROLLUP_TIME_ONLY` (`true`) | Window labeling |
| `EVOSSEARCH_LUXRIOT_ALERTS_JSON_PROMPT` / `_SYSTEM_PROMPT_DEFAULT` | Prompt templates |

## LM profiles (API keys are secrets)

| Var (default) | Notes |
|---|---|
| `EVOSSEARCH_LM_PROFILES` | e.g. `agent,vlm` |
| `EVOSSEARCH_LM_*_PROFILE_*` | Per-profile base_url / model / timeout / kind `[FIELD]` |
| `EVOSSEARCH_LM_VLM_BALANCER_ENABLED` | Static channel→profile routing across multiple VLM hosts |
| `EVOSSEARCH_LM_VIDEO_*` | Frame count / max edge / tokens / temperature |

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
| `EVOSSEARCH_DETECTIONS_ARCHIVE_ENABLED` (`true`) | Persist detection frames |
| `EVOSSEARCH_DETECTIONS_RETENTION_*` | Dedup/keep windows + similarity thresholds |

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
