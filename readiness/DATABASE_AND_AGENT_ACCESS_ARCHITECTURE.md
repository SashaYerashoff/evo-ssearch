# EVA AI database and agent access architecture

Date: 2026-06-08
Status: proposed

## Decision

Use PostgreSQL as the system of record for the 50-channel pilot and the first
production deployments.

Recommended initial stack:

- PostgreSQL 18.x; PostgreSQL 17.x is an acceptable compatibility fallback.
- pgvector 0.8.2 for event-level CLIP/DINO/summary embeddings.
- Psycopg 3 with a bounded connection pool.
- SQLAlchemy Core and Alembic for schema definitions and migrations. Avoid a
  large ORM conversion during the pilot.
- A `MediaStore` abstraction backed by local NVMe for the pilot and by an
  S3-compatible object store when deployment becomes distributed.
- A PostgreSQL job table with `FOR UPDATE SKIP LOCKED` for the first bounded VLM
  worker queue.
- A transactional outbox from the first schema version.

Do not add ClickHouse, Kafka, a dedicated vector database, or Redis to the first
customer deployment unless measurements require them. The outbox and repository
boundaries provide a migration path without making the pilot operate a
distributed data platform.

## Why PostgreSQL

PostgreSQL fits the current product because most data is transactional and
relational:

- users, roles, sessions, and channel grants;
- channels, probes, prompt versions, and model configuration;
- detections, summaries, bookmarks, and agent sessions;
- VLM jobs and their state transitions;
- append-only security audit events.

It gives the pilot:

- safe concurrent readers and writers;
- transactions across configuration, jobs, audit, and outbox records;
- declarative time partitioning and efficient retention;
- row-level security as defense in depth;
- mature backup, point-in-time recovery, and replication;
- pgvector search without operating another database immediately.

At the default 50-channel profile, one 12-frame summary per channel per minute
produces about 72,000 summary rows/day. A 30-second heartbeat produces about
144,000 rows/day. PostgreSQL can handle this comfortably with short batched
transactions, correct indexes, and bounded payloads.

## What PostgreSQL must not store

Do not store full JPEGs or base64 thumbnails in database rows.

PostgreSQL stores media metadata:

- object ID;
- tenant and channel IDs;
- storage URI/key;
- content hash;
- MIME type, size, width, and height;
- captured timestamp;
- retention class and deletion state.

Image bytes live in `MediaStore`. A small thumbnail is also a media object. API
responses use authenticated media endpoints or short-lived signed URLs, not
filesystem paths.

This removes the current duplicate image storage from detections, JSON state,
agent results, and the archive.

## Initial schemas

### `iam`

- `users`
- `roles`
- `permissions`
- `user_roles`
- `user_channel_grants`
- `sessions`
- `login_attempts`

### `core`

- `tenants`
- `channels`
- `probes`
- `probe_versions`
- `prompt_sets`
- `prompt_versions`
- `model_configs`

Probe definitions and probe runtime state must be separate. Editing a probe
creates a version and an audit event. Runtime hits never rewrite the definition.

### `events`

- `detections`
- `summaries`
- `summary_rollups`
- `bookmarks`
- `capture_runs`
- `probe_runs`

`detections`, `summaries`, and audit events are range-partitioned by event time.
Start with monthly partitions for 50 channels. Revisit daily partitions only
after measured volume justifies them.

Every event row contains `tenant_id`, `channel_id`, event time, ingestion time,
source/model version, and a stable external deduplication key.

### `vectors`

- `detection_embeddings`
- `summary_embeddings`

Keep vectors separate from the hot event row:

- `event_id`
- `embedding_kind`
- `model_id`
- `model_version`
- `dimensions`
- `embedding vector(...)` or `halfvec(...)`

Only retained detections and useful summaries receive persistent embeddings.
Do not persist every sampled frame for every channel.

Use exact search initially. Add HNSW per active partition after measuring query
latency and index size. Filtering by tenant, channel, and time is mandatory.

### `media`

- `objects`
- `retention_policies`
- `deletion_jobs`

Deleting an event and deleting media are coordinated through a deletion job.
Media deletion must be retryable and idempotent.

### `jobs`

- `inference_jobs`
- `job_attempts`
- `dead_letters`
- `outbox`

Capture produces a job without waiting for the VLM. Workers claim jobs with
`FOR UPDATE SKIP LOCKED`.

Required job fields:

- tenant/channel and workload class;
- model and prompt version;
- media object IDs;
- priority and deadline;
- state, attempt count, lease owner, and lease expiry;
- idempotency key;
- created, started, and finished timestamps;
- compact result/error metadata.

Queue overload policy is explicit: coalesce replaceable heartbeat jobs, preserve
event-triggered jobs, and report dropped/coalesced counts.

### `agent`

- `sessions`
- `messages`
- `tool_runs`
- `action_plans`
- `action_approvals`

Agent sessions belong to a user and tenant. Tool runs record the actor, tool,
normalized arguments hash, permission decision, duration, result class, and
related audit event.

### `audit`

- `events`

The application audit writer receives INSERT only. It cannot update or delete
audit rows. Partitions are archived under a documented retention policy.

For stronger evidence, chain event hashes and regularly export signed checkpoints
or audit partitions to storage outside the application administrator's control.
Hash chaining alone is tamper-evident, not tamper-proof.

## Database roles

Use separate credentials and pools:

| Role | Access |
|---|---|
| `eva_owner` | Owns schemas; no normal application login |
| `eva_migrator` | DDL during controlled deployment only |
| `eva_api` | Required application DML; no DDL or audit deletion |
| `eva_worker` | Claim jobs, write results/events, read required configuration |
| `eva_agent_reader` | SELECT on curated views/functions only |
| `eva_audit_writer` | INSERT on audit events only |
| `eva_backup` | Backup-specific access, not used by the application |

The API and agent must never connect as the schema owner, superuser, or a role
with `BYPASSRLS`.

Use TLS and SCRAM-SHA-256, restrict `pg_hba.conf`, keep PostgreSQL off public
interfaces, and load credentials from deployment secrets. Apply connection,
statement, lock, and idle transaction timeouts.

## Row and tenant isolation

Every customer-owned row contains `tenant_id`. Channel-scoped rows also contain
`channel_id`.

Application authorization remains the primary control. PostgreSQL row-level
security is an additional control:

- enable and force RLS on sensitive tables;
- use an owner role different from runtime roles;
- set authenticated tenant/user context transaction-locally;
- test policies with cross-tenant and cross-channel negative cases;
- never accept tenant identity from request JSON or tool arguments.

For the first on-prem single-tenant deployment, there may be one tenant row, but
the column and policy boundary should exist from the first migration.

## Safe agent tool architecture

### Core rule

The LLM never receives SQL credentials and never executes SQL.

Agent tools call typed application services. Services call repositories with
parameterized queries. The database sees the authenticated actor and tenant
context supplied by the server, not values asserted by the model.

Do not add a general `query_sql`, `execute_sql`, shell, arbitrary file reader, or
arbitrary URL tool.

### Execution context

Every tool call receives a server-created context:

```text
actor_id
tenant_id
roles and permissions
allowed_channel_ids
agent_session_id
request_id
client_ip
```

The context is not part of the model-generated arguments and cannot be
overridden by them.

### Tool registry

Each tool has server-side metadata:

```text
required_permission
risk: read | write | external_side_effect
approval_required
channel_scope_resolver
maximum_rows and maximum_time_window
timeout and rate limit
output_data_classification
audit_event_type
```

Execution order:

1. Validate arguments against a closed schema.
2. Authorize the tool permission.
3. Resolve and verify tenant/channel scope.
4. Enforce row, time, cost, and rate limits.
5. Require a valid approval for high-impact actions.
6. Call a typed service.
7. Redact and bound the result.
8. Write tool-run and audit records.

The model sees only tools allowed for the current user's role, but tool-list
filtering is usability, not the security boundary. Authorization is repeated for
every execution.

### Mutating tools and approval

The current `preview=true` convention is not a security control. A model can
call the same tool with `preview=false`.

Replace it with a two-step server protocol:

1. The agent calls `plan_*`. The server stores normalized arguments, diff,
   actor, required permission, expiry, and an arguments hash. It returns
   `plan_id`.
2. The UI renders an approval dialog from trusted structured fields. The user
   approves through a normal authenticated API endpoint outside the LLM
   conversation.
3. The server creates a one-time `approval_id` bound to the user, plan hash,
   action, and short expiry.
4. `execute_*` accepts only the `approval_id`; it loads the stored plan rather
   than accepting new model-generated mutation arguments.
5. Execution consumes the approval and writes an audit event in the same
   transaction where possible.

Require this for:

- probe create/update/delete;
- prompt and model configuration changes;
- bookmarks/external events if customer policy requires it;
- exports;
- retention changes;
- user and permission changes.

An approval is never represented only by a user's natural-language message.

### Read tools

Expose narrow operations:

- `search_detections(filters, query, limit)`
- `aggregate_detections(filters, dimensions, metrics)`
- `get_summaries(channel_ids, time_window, depth, limit)`
- `get_detection_media(detection_id)`
- `list_probes(channel_scope)`

All require bounded time windows and row limits. Cross-channel queries intersect
requested channels with the user's channel grants.

If free-form analytics becomes necessary, implement a validated query DSL:
allow-listed dimensions, metrics, filters, sort fields, and maximum cost. Compile
the validated AST to parameterized SQL. Do not execute model-generated SQL.

### Media and filesystem safety

Remove arbitrary `image_path` and absolute `folder` arguments from model-facing
tools.

Use:

- `detection_id`
- `media_object_id`
- registered `dataset_id`
- authorized `channel_id`

The server resolves these IDs to storage objects after authorization. The model
does not see local paths.

### Untrusted data

Detection text, summaries, comments, filenames, OCR, and external metadata are
untrusted content and may contain indirect prompt injection.

- Mark tool output as data, not instructions.
- Keep authorization decisions outside the LLM.
- Strip control markup and bound text length.
- Do not expose secrets or internal paths in tool results.
- Do not let retrieved content expand the available tool set.

## Changes required in the current agent

1. Pass authenticated `ToolExecutionContext` from `/agent/chat` to
   `AgentRunner.stream_chat()` and every tool execution.
2. Add `user_id` and `tenant_id` ownership to agent sessions and messages.
3. Replace the static all-tools list with role-filtered schemas.
4. Add a policy-enforcing `ToolGateway` around `AgentTools`.
5. Split read tools from command tools.
6. Replace `preview=false` mutations with stored plans and external approvals.
7. Remove model-facing arbitrary filesystem paths.
8. Add channel-scope checks to every read and write tool.
9. Add tool timeout, cancellation, rate, row, and time-window limits.
10. Persist tool runs and security audit events.

## Growth path

### 50 channels

Use one PostgreSQL primary on local NVMe, pgvector, local `MediaStore`, and
separate API/VLM worker processes. This remains operationally simple.

### Hundreds of channels

- Add PostgreSQL standby and point-in-time recovery.
- Move media to S3-compatible object storage.
- Separate API, capture, probe, and VLM worker deployments.
- Add PgBouncer if instance count makes application pools inefficient.
- Export outbox events to an analytics store.

### Thousands of channels

At 8,000 channels:

- one summary/channel/minute is 11.52 million rows/day;
- a 30-second heartbeat is 23.04 million rows/day;
- persisting one embedding/second/channel would be 691.2 million vectors/day and
  is not a valid retention strategy.

Keep PostgreSQL as the control plane and source of truth for configuration,
identity, jobs, approvals, and audit indexes.

Add:

- a durable distributed event bus;
- ClickHouse for high-volume summaries, detections, telemetry, and aggregate
  queries;
- distributed object storage with lifecycle policies;
- a dedicated vector service only if retained vector volume and query
  measurements outgrow partitioned pgvector;
- horizontally scaled capture and inference workers.

Feed these systems from the transactional outbox. Do not dual-write from request
handlers.

## Migration sequence

### Phase 0: repository boundary

- Add repository interfaces for users, probes, detections, summaries, agent
  sessions, jobs, media metadata, and audit.
- Keep SQLite/JSON adapters temporarily.
- Add `DATABASE_URL` and migration tooling.

### Phase 1: security/control plane

- Create PostgreSQL schemas and roles.
- Move users, sessions, roles, channel grants, audit, agent sessions, probes,
  prompt versions, and model configuration.
- Introduce `ToolExecutionContext` and `ToolGateway`.

### Phase 2: event plane

- Move summaries, rollups, detections, and vector rows.
- Introduce `MediaStore`; backfill archive files and content hashes.
- Stop writing thumbnails into database and JSON fields.

### Phase 3: worker queue

- Insert inference jobs transactionally.
- Run dedicated VLM workers with bounded leases and idempotent results.
- Add overload/coalescing behavior and queue metrics.

### Phase 4: cutover

- Dual-write briefly through repositories.
- Compare row counts, checksums, and sampled query results.
- Switch reads to PostgreSQL.
- Freeze SQLite/JSON stores as rollback artifacts.
- Run a restore test and a 24-48 hour 50-channel soak.

Do not maintain indefinite dual writes. Every phase must have a defined cutover
and rollback point.

## Immediate implementation slice

The first code slice should be deliberately narrow:

1. Add PostgreSQL dependencies, connection pool, Alembic, and `/ready` DB check.
2. Create `iam`, `agent`, and `audit` schemas.
3. Implement users, secure sessions, roles, channel grants, and audit writer.
4. Move agent sessions to PostgreSQL with ownership.
5. Add `ToolExecutionContext` and deny tools without explicit permissions.
6. Implement plan/approval for probe and prompt mutations.

This creates the security boundary first. Detections and summaries can then be
migrated behind repositories without exposing a more powerful database to the
current unrestricted agent.

## References

- PostgreSQL 18 documentation:
  https://www.postgresql.org/docs/18/
- PostgreSQL row-level security:
  https://www.postgresql.org/docs/18/ddl-rowsecurity.html
- PostgreSQL table partitioning:
  https://www.postgresql.org/docs/18/ddl-partitioning.html
- PostgreSQL password authentication:
  https://www.postgresql.org/docs/18/auth-password.html
- pgvector:
  https://github.com/pgvector/pgvector
- Psycopg connection pools:
  https://www.psycopg.org/psycopg3/docs/advanced/pool.html
- OWASP LLM Excessive Agency:
  https://genai.owasp.org/llmrisk/llm062025-excessive-agency/
