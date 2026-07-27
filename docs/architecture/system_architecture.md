# System Architecture (Software)

The software/deployment view: components, data flow, storage, runtime model. The
reasoning model ("why") is in [cognitive_architecture](cognitive_architecture.md).
Invariants: [facts](../00_CANON/facts.md). Config: [config_reference](../00_CANON/config_reference.md).

## Components

- **EVA AI app** (`oldapp.py`, Flask under Gunicorn) — internal HTTP API + UI,
  capture scheduler, probe daemon, summary pipeline, archive access, agent
  gateway. TLS is normally terminated before Gunicorn (Nginx/site proxy).
- **PostgreSQL** — control plane: IAM, agent sessions, audit, archive
  (`archive.detections`, `archive.probes`, `archive.runtime_state`). RLS forced.
- **CLIP embedder** — runs on the app host; embeds every captured frame and
  search queries (`ViT-B/32`).
- **VLM inference** (vLLM, `qwen3-vl-4b`) — produces video-descriptions; runs on
  dedicated host(s).
- **Agent LM** (`qwen3.5-9b` class) — the conversational agent; separate endpoint.
- **Luxriot Evo** — source of channels/snapshots and sink for bookmarks.

## Deployment topology (pilot)

```
[Operator browser] --TLS--> [Nginx/site TLS boundary]
                              |
                              v HTTP localhost
                         [EVA AI host: Flask/Gunicorn(1 worker, gthread)
                          + CLIP + PostgreSQL + Agent LM client]
                              |                       |
        snapshots/bookmarks   | SQL (RLS, 3 roles)    | HTTP
                              v                       v
                         [Luxriot Evo NVR]     [vLLM host(s): qwen3-vl-4b]
                                               [Agent LM host: qwen3.5-9b]
```

Client specifics (hosts/IPs/ports) live in `install/field_rollout_demo.md`
(`[FIELD]`); the sanitized `install/deployment_guide.md` uses placeholders.

Office/demo systems may temporarily expose Gunicorn directly over HTTP
(`http://<host>:5000`). Client-facing systems should use TLS at the browser
boundary and `EVOSSEARCH_AUTH_COOKIE_SECURE=true`.

## Data flows

**Capture → description (per channel, in-process loop)**
1. Pull snapshot at `SNAPSHOT_INTERVAL`; CLIP-embed it (feeds the probe buffer).
2. When the batch fills (default 12 frames), build a summary batch.
3. Dispatch to the VLM (synchronous by default; durable queue available but off).
4. VLM returns a description plus one `BATCH_STATE_JSON` block containing the
   cover, episode state, memory pass, observations, and zero or more alerts.
5. Accept: validate the unified batch state and snapshot references → optional
   Luxriot bookmarks (gated, instrumented); retain the cover and evidence frames
   under one stable batch id in the CLIP-indexed archive; record compact state
   into per-channel history (frame-time anchored; debounced persist).

**Aggregation**
- L0 history → L1/L2/L3 rollups (deterministic + optional LLM synthesis).
  Agent investigation reads are **read-only** (no LLM synthesis triggered).

**Search**
- Query → CLIP vector → candidate frames (time/channel/source filter,
  `ORDER BY ts DESC LIMIT N`) → ranked via in-memory FAISS flat index per
  `ch:date` shard → top-k thumbnails fetched. Recall bounded by the candidate
  window.

**Agent**
- HTTP (SSE streaming) → tool gateway (per-tool authz, channel scope, rate/row
  limits, audit) → tools over manager/archive/probes → streamed answer with
  evidence.

## Storage

| Data | Where | Retention |
|---|---|---|
| Frame archive (vectors + thumbnails) | `archive.detections` (Postgres) | row + thumbnail retention (configurable) |
| Probe definitions | `archive.probes` | persistent |
| Summary history / semantic L1–L3 rows / hot rollup cache / prompt settings / desired sessions | `archive.runtime_state` (Postgres) | summary retention days |
| IAM / sessions / audit | dedicated schemas (RLS) | per policy |

## Runtime model & durability

- **Single Gunicorn worker** (gthread). Capture/probe/summary schedulers are
  in-process and not multi-worker safe.
- **Graceful restart** flushes summary state + rollup cache via Gunicorn worker
  hooks (`gunicorn_conf.py`). Desired live sessions are restored on startup.
- **Hard kill (SIGKILL)** can lose up to the persist-debounce interval of summary
  history; durable settings/sessions use immediate writes.
- **Inference queue** (`inference_queue/`) exists for durable, decoupled VLM
  dispatch with a worker pool; disabled by default, enable only after load
  validation.

## Known scaling ceilings (pilot → 10k)

- Search recall bounded by candidate window → ANN index (pgvector) for true
  multi-day recall.
- Archive growth → time-partitioning for `DROP PARTITION` retention.
- Single-worker in-process schedulers → split capture/VLM workers via the durable
  queue with DB leases.
- Synchronous VLM dispatch under high channel counts → enable the queue + bounded
  worker pool.

See [observability](../admin/observability.md) and
[backup_recovery](../admin/backup_recovery.md) for operations.
