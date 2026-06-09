# EVA AI: security and data readiness for a 50-channel pilot

Date: 2026-06-08

## Executive decision

The current build is suitable for an internal demo, but not yet for deployment on a
customer network with sensitive video data.

For a single-server 50-channel pilot, hardened SQLite remains acceptable if the
application runs as one process on local NVMe and all writes are serialized and
batched. This is a pilot constraint, not the architecture for 8,000 channels.

The minimum customer baseline is:

1. Named users, server-side sessions, and role-based authorization.
2. Security audit events plus structured operational logs.
3. No secrets returned to the browser or stored in browser localStorage.
4. A bounded VLM work queue decoupled from snapshot capture.
5. Database-backed summary and probe runtime state instead of whole-file JSON
   rewrites.
6. Explicit data retention, disk quota, purge, backup, and restore procedures.
7. A 24-48 hour soak test with the actual 50-channel workload.

Estimated implementation and verification effort: 8-12 engineering days,
excluding model and GPU tuning and excluding container packaging.

## Current security posture

| Area | Current state | Pilot verdict |
|---|---|---|
| Authentication | One shared `EVOSSEARCH_ADMIN_TOKEN` | Red |
| Browser credential storage | Admin token in `localStorage` | Red |
| Authorization | Most mutations require the shared token; most reads are open | Red |
| Users and roles | Absent | Red |
| Security audit | Absent | Red |
| Operational logging | `print()` and tracebacks, no request correlation | Red |
| Secrets | `/settings/env` can return effective environment values, including secrets, to an admin client | Red |
| Health checks | `/health` and `/ready` exist | Green |
| Rate limiting | Absent | Red |
| TLS | Not provided by the application | Must be supplied by deployment |

### Minimum role model

Implement permissions and map the initial roles to them:

| Role | Intended permissions |
|---|---|
| `admin` | User management, system settings, all data and audit access |
| `engineer` | Models, prompts, probes, capture configuration, diagnostics |
| `operator` | View streams/detections, use agent, run approved probes, create bookmarks |
| `viewer` | Read-only access to assigned channels and reports |

Permissions should be checked on every endpoint, including image, snapshot,
detection, session, and settings reads. Probe/prompt changes are privileged
operations because they can redirect monitoring or generate external bookmarks.

For the pilot, channel-level access can be represented in the user record as an
allow-list. It may be `all` for the first customer, but the authorization layer
must receive the channel ID so this can be enforced later without rewriting the
API.

### Authentication implementation

- Store users and sessions server-side.
- Hash passwords with Argon2id.
- Use an opaque random session ID in an `HttpOnly`, `Secure`, `SameSite=Strict`
  cookie. Do not put credentials in `localStorage`.
- Protect state-changing requests with CSRF tokens.
- Add login throttling, failed-login audit events, session expiry, logout, and
  administrator session revocation.
- Bootstrap the first administrator with a one-time CLI command or one-time
  secret, then disable the bootstrap path.
- Keep the current admin token only as a temporary service/bootstrap mechanism;
  do not expose it in the normal UI.
- Mask all secrets in settings responses. A saved secret is write-only from the
  browser's perspective.
- Deploy behind HTTPS, even on a customer LAN.

### Required audit events

Each event must record timestamp, request ID, actor user and role, source IP,
action, target type/ID, channel ID where applicable, result, and safe structured
details. Do not record passwords, tokens, raw images, or complete sensitive
prompts in the audit log.

At minimum audit:

- Login success/failure, logout, session revocation.
- User creation, disablement, role and channel-scope changes.
- Probe create/update/delete/run and threshold changes.
- Prompt, model, capture, rollup, bookmark, and system setting changes.
- Agent session deletion and skill creation/update.
- Manual bookmark and external event creation.
- Sensitive data export/download and audit-log access.

Operational logs and audit logs are separate. Operational logs go to JSON stdout
with request IDs, latency, endpoint, status, queue depth, and exception details.
Audit events are durable application data with a retention policy.

## 50-channel workload

With the current defaults:

- Snapshot interval: 5 seconds.
- Batch: 12 frames.
- Aggregate capture: 10 snapshots/second for 50 channels.
- VLM traffic: one 12-image request per channel per minute.
- Aggregate VLM demand: about 50 requests/minute and 10 images/second.

The current `LuxriotCaptureSession` performs capture, CLIP indexing, and the
synchronous VLM request in the same per-channel thread. While the VLM call is
running, that channel is not capturing. Fifty channels also tend to create
bursty, uncontrolled concurrency against the inference server.

Before the pilot:

1. Keep capture loops independent from VLM latency.
2. Put VLM jobs into a bounded central queue.
3. Limit inference concurrency per model/GPU.
4. Define overload behavior: coalesce old jobs, keep the newest frame/batch, and
   expose dropped/coalesced job counters.
5. Add per-channel cadence and priority.

The recommended pilot profile is hybrid:

- CLIP probes run at the required lightweight cadence.
- VLM heartbeat descriptions run every 30-60 seconds.
- Probe hits create short higher-frequency VLM bursts.
- Critical channels may receive a higher fixed priority.

Continuous 1 FPS VLM description on all 50 channels means 50 images/second and
is a different hardware class from the current pilot target.

## Data storage assessment

Measured local state on 2026-06-08:

- `detections_store.sqlite3`: 20,000 rows, 254 MiB.
- Embedded detection thumbnails: 164 MiB.
- Embedded CLIP vectors: 46.8 MiB.
- `detections_archive`: 136,967 files, 7.4 GiB.
- `luxriot_summary_state.json`: 1,200 summaries for two channels, 1.2 MiB.
- `probes_store.json`: six probes and 150 recent hits, 1.5 MiB.

| Store | Current behavior | 50-channel verdict |
|---|---|---|
| Agent sessions SQLite | WAL, low write volume | Green after user ownership/access scope |
| Detection SQLite | 20k hard cap, rollback journal, one transaction and full count per record, thumbnails duplicated in DB | Red/Yellow |
| Summary state JSON | Entire history file rewritten for every summary | Red |
| Probe JSON | Config and thumbnail-rich runtime hits rewritten together | Red |
| Rollup cache JSON | Entire cache rewritten on updates | Yellow/Red |
| Probe buffers | 2,000 frames/channel in RAM; global lock; FAISS index rebuilt for every frame but queries use matrix multiplication | Red/Yellow |
| Detection archive | No global quota/purge; DB trimming does not remove archived files | Red |
| Schema migrations | Inline column checks, no versioned migration tool | Red |
| Backup/restore | No verified procedure | Red |

### Why the JSON stores do not scale to 50

At the configured history limit, 50 channels can retain 30,000 summaries in one
JSON document. Based on the current data this is roughly 30 MiB. At one summary
per channel per minute, rewriting that document for every summary can produce
about 1.5 GiB of writes per minute once full.

Probe runtime data has the same problem. Current recent-hit thumbnails account
for most of a 1.5 MiB file with only six probes. Probe definitions and runtime
hits must be stored separately.

### Probe memory estimate

At 2,000 frames for 50 channels there may be 100,000 frames in memory:

- CLIP vectors at 512 float32 values: about 195 MiB.
- A second FAISS copy: about 195 MiB.
- Current thumbnail sizes imply roughly 0.9 GiB of base64 data.
- Python objects and metadata add further overhead.
- ROI caches can add several GiB in a worst case.

At the default five-second cadence this buffer covers about 2.8 hours. At 1 FPS
it covers only 33 minutes and the current full-index rebuild becomes a material
CPU and allocation bottleneck. The FAISS index is currently rebuilt but not used
by `query()`, so that work should be removed or replaced by a real incremental
index.

## Required storage changes

### P0: before customer deployment

1. Add a versioned schema and migrations.
2. Move summary history, runs, rollups, probe definitions, and probe runtime
   state into SQLite tables.
3. Use one database writer queue or otherwise serialize writes explicitly.
4. Batch detection inserts in one transaction; do not count and trim on every
   row.
5. Store one image copy. Keep the filesystem path and a small UI thumbnail only
   where necessary.
6. Replace the 20,000-row cap with time/size retention and per-customer limits.
7. Purge archive files together with their database records and expose disk
   usage and purge status.
8. Add periodic consistent backup and perform a restore test.
9. Pin a SQLite build containing the March 2026 WAL race fix before enabling WAL
   broadly. The current Python and CLI SQLite version is 3.46.1.

### SQLite pilot constraints

SQLite is acceptable for this pilot only with:

- One application instance/process.
- Local disk, never a network filesystem.
- A fixed SQLite version with the WAL fix.
- Short transactions, busy timeout, WAL checkpoint monitoring, and bounded
  queues.
- Automated backups and tested restore.

PostgreSQL is not required to prove the 50-channel pilot, but it becomes the
preferred next step as soon as there are multiple application instances,
multiple writers, centralized multi-node operation, or an 8,000-channel plan.

## Two-week implementation sequence

### Days 1-3: identity and access

- User/session schema, Argon2id passwords, login/logout/me.
- Permission decorator and endpoint access matrix.
- Login UI and removal of the browser admin token.
- Protect every sensitive read and mutation.
- Mask secrets and make secret fields write-only.
- Authentication and authorization tests.

Acceptance: two users with different roles cannot access or mutate each other's
forbidden resources; all existing routes have an explicit access policy.

### Days 3-4: logging and audit

- JSON operational logging and request IDs.
- Durable audit-event table and instrumentation of P0 actions.
- Login throttling and security event coverage.

Acceptance: a probe change, settings change, login failure, bookmark, and user
role change can be attributed to a named user and request.

### Days 5-7: storage hardening

- Database migrations and new summary/probe/rollup tables.
- Batched detection writes and retention.
- Archive quota/purge and disk metrics.
- Backup and restore command/runbook.

Acceptance: no high-frequency path rewrites a whole JSON state file; data
survives restart and restore.

### Days 7-9: 50-channel scheduler

- Decoupled capture and bounded VLM queue.
- Per-GPU concurrency, channel priority, coalescing, retry/backoff.
- Queue depth, latency, dropped/coalesced jobs, and channel freshness metrics.

Acceptance: an overloaded model slows descriptions but does not stop snapshot
capture or exhaust RAM.

### Days 9-10: soak and deployment gate

- 24-48 hour run with 50 real or replayed channels.
- Verify RAM plateau, DB and archive growth, queue stability, p95 description
  age, restart recovery, backup/restore, and access controls.
- Record the exact supported cadence and retention profile for the customer.

## Container preparation without containerizing yet

Container packaging can wait, but these boundaries should be introduced now:

- All mutable state under one configurable `EVOSSEARCH_DATA_DIR`.
- Separate configurable model/cache directory.
- Logs to stdout; audit data to the application database.
- Secrets supplied outside the settings export and never baked into an image.
- No runtime dependency on the repository working directory.
- Existing `/health` and `/ready` endpoints remain the deployment probes.

## Boundary for 8,000 channels

The 50-channel pilot can be a hardened single-node product. An 8,000-channel
deployment is a distributed system and must not be represented as a larger
SQLite configuration.

It will require, at minimum:

- Stateless API instances and centralized identity.
- Distributed capture/inference workers with a durable queue.
- PostgreSQL or another partitioned metadata/event store.
- Object storage with lifecycle policies for images.
- Sharded vector/search services.
- Per-tenant and per-channel authorization.
- Fleet-level metrics, audit aggregation, capacity management, and failure
  isolation.

The pilot work above is still reusable if authentication, authorization, audit,
storage repositories, and inference scheduling are kept behind explicit
interfaces.

## References

- OWASP Password Storage Cheat Sheet:
  https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html
- OWASP Session Management Cheat Sheet:
  https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html
- OWASP Logging Cheat Sheet:
  https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html
- SQLite Write-Ahead Logging:
  https://www.sqlite.org/wal.html
