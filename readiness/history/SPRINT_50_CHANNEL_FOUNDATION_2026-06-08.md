# EVA AI sprint: secure 50-channel foundation

Sprint length: 10 working days
Start: 2026-06-08
Target finish: 2026-06-19

## Sprint goal

Produce a production-shaped control plane for the first 50-channel customer
deployment:

- PostgreSQL and versioned migrations;
- named users, roles, channel grants, and secure sessions;
- durable security audit and structured operational logging;
- an authorization boundary around agent tools;
- PostgreSQL-backed agent sessions, probes, prompts, summaries, and VLM jobs;
- bounded inference scheduling that cannot stop capture under load;
- a repeatable migration and pilot acceptance test.

Sprint invariant: every sensitive read, mutation, export, external action, and
agent tool execution leaves an attributable audit fingerprint. Sensitive writes
and external side effects fail closed when durable audit recording is
unavailable.

Container packaging and the 50-100 channel agent navigation loop are explicitly
out of scope. The code must nevertheless use configurable data/model paths,
stdout logging, secrets, and health checks so containerization does not require
another persistence redesign.

## Parallelism

Use four implementation sub-agents plus one lead/integrator.

| Owner | Primary scope | Files/modules owned during parallel work |
|---|---|---|
| Lead/integrator | Schema contracts, architecture decisions, integration, reviews, cutover | Cross-cutting changes and final merges |
| DB agent | PostgreSQL bootstrap, Alembic, repositories, migrations, backup/restore | New `db/`, `repositories/`, migrations |
| IAM agent | Users, roles, sessions, channel grants, login UI, endpoint policy | New `security/`, auth routes and UI |
| Agent-safety agent | `ToolExecutionContext`, `ToolGateway`, role-filtered tools, approvals | New gateway/policy modules, focused `agent.py` edits |
| Data-plane agent | Summary/probe migration, MediaStore boundary, VLM job queue | New stores/workers, focused connector changes |

A fifth sub-agent can be used temporarily as a verifier for tests, migration
comparison, security review, and documentation. It should not own a fifth
cross-cutting implementation stream.

More than four concurrent implementers is counterproductive until `oldapp.py`
has stable service/repository boundaries.

## Working rules

1. Schema and repository interfaces are frozen by the lead at the end of Day 1.
2. Only the lead resolves cross-module contracts and edits shared bootstrap
   wiring during integration.
3. Every migration has upgrade and downgrade behavior or a documented
   irreversible reason.
4. No agent receives SQL, shell, arbitrary URL, or arbitrary filesystem tools.
5. No authentication or authorization decision is delegated to an LLM.
6. New code must work with PostgreSQL; compatibility adapters may keep existing
   SQLite/JSON data readable during migration.
7. Dual writes are temporary, measured, and removed after cutover.

## Backlog

### P0: must complete

| ID | Work item | Owner | Estimate | Acceptance |
|---|---|---|---:|---|
| DB-1 | PostgreSQL connection pool, config, `/ready` check | DB | 1.0d | Startup and readiness distinguish unavailable DB from migration mismatch |
| DB-2 | Alembic and initial schemas | DB | 1.5d | Empty DB upgrades reproducibly to current revision |
| IAM-1 | Users, Argon2id passwords, login/logout/me | IAM | 2.0d | No shared browser admin token is required for normal UI operation |
| IAM-2 | Roles, permissions, channel grants | IAM | 2.0d | Cross-role and cross-channel negative tests pass |
| IAM-3 | Secure cookie sessions and CSRF | IAM | 1.0d | Cookies are HttpOnly/Secure/SameSite and mutations reject missing CSRF |
| AUD-1 | Append-only audit writer and event schema | Lead + IAM | 1.0d | Login, role, probe, prompt, bookmark, and approval actions are attributable |
| LOG-1 | JSON logs and request IDs | Verifier/lead | 1.0d | Requests, failures, queue latency, and tool runs share a request ID |
| AG-1 | Authenticated `ToolExecutionContext` | Agent safety | 1.0d | Actor/tenant/channel context is server-created and reaches every tool |
| AG-2 | `ToolGateway` authorization and limits | Agent safety | 2.0d | Tools deny missing permission and forbidden channels server-side |
| AG-3 | Stored plan and one-time approval flow | Agent safety | 2.0d | Mutations cannot execute by changing model arguments after approval |
| AG-4 | Remove arbitrary filesystem paths from agent tools | Agent safety | 0.5d | Agent accepts IDs, not `image_path` or arbitrary folder paths |
| DATA-1 | PostgreSQL probe definitions and prompt versions | Data plane | 1.5d | Definitions no longer depend on whole-file JSON writes |
| DATA-2 | PostgreSQL summary/run/rollup tables | Data plane | 2.0d | Live summary updates do not rewrite a complete JSON history |
| JOB-1 | Bounded PostgreSQL inference queue | Data plane | 2.0d | Capture enqueues without waiting for VLM; workers use leases/idempotency |
| JOB-2 | Priority, coalescing, and overload counters | Data plane | 1.0d | Heartbeats coalesce; event jobs survive; overload is visible |
| MIG-1 | SQLite/JSON import command and comparison report | DB + Data plane | 1.5d | Counts and sampled rows match; rerun is idempotent |
| OPS-1 | Backup and restore runbook plus restore test | DB | 1.0d | A fresh DB restores and passes readiness |
| TEST-1 | Security, migration, tool-policy, and queue tests | All | 2.0d distributed | CI-reproducible suite covers every P0 boundary |

### P1: complete if P0 remains on schedule

| ID | Work item | Owner | Estimate | Acceptance |
|---|---|---|---:|---|
| DATA-3 | PostgreSQL detections and pgvector adapter | DB + Data plane | 2.0d | Existing search API returns equivalent sampled results |
| MEDIA-1 | `MediaStore` interface and local NVMe adapter | Data plane | 1.5d | New detections store media IDs, not base64/path payloads |
| RET-1 | Retention and coordinated media deletion | Data plane | 1.0d | DB and files purge together with retryable deletion jobs |
| RATE-1 | Login/tool/expensive endpoint throttling | IAM | 0.5d | Limits are keyed by actor/IP and produce auditable failures |

### Deferred

- ClickHouse, Kafka, Redis, and a dedicated vector database.
- Full container image and NVIDIA container validation.
- HA PostgreSQL and object storage.
- Agent planner for hierarchical navigation across 50-100 channels.
- Decomposition of the entire `oldapp.py`.

## Daily sequence

### Day 1: contracts

- Freeze schemas, IDs, repository interfaces, permissions, and endpoint policy.
- Bootstrap PostgreSQL/Alembic and test database.
- Define `ToolExecutionContext` and tool policy metadata.
- Inventory current JSON/SQLite import fields.

Gate: no parallel implementation continues with conflicting table or permission
names.

### Days 2-3: control plane

- DB agent creates schemas, roles, migrations, and repository skeletons.
- IAM agent implements user/session primitives and login.
- Agent-safety agent adds context and read-tool authorization.
- Data-plane agent implements summary/probe repositories and import mapping.

Gate: authenticated request can open a transaction with tenant/user context;
unauthenticated sensitive reads fail.

### Days 4-5: application integration

- Protect routes and add channel-scoped authorization.
- Move agent sessions, probes, prompts, summaries, and rollups to PostgreSQL.
- Add audit writer and instrument P0 mutations.
- Add structured logs and request correlation.
- Remove arbitrary filesystem arguments from model-facing tools.

Gate: restart loses no migrated state; two roles receive different tool lists
and cannot bypass execution authorization.

### Days 6-7: safe commands and scheduling

- Implement action plan, UI approval, one-time execution token, and audit.
- Add PostgreSQL VLM jobs, workers, leases, retries, and idempotency.
- Decouple capture from synchronous VLM calls.
- Add priority and coalescing.

Gate: a stalled inference worker does not stop capture or permit duplicate
results; model-generated mutations cannot execute without a bound approval.

### Day 8: migration and operations

- Import existing SQLite/JSON data.
- Compare counts, hashes, timestamps, and sampled API results.
- Add backup/restore commands and runbook.
- Add retention and MediaStore work if P0 is green.

Gate: import can be rerun, rollback artifacts are retained, and restore works on
a clean database.

### Day 9: integration hardening

- Full test suite, authorization matrix, failure injection, queue overload test.
- Check secret masking, log redaction, transaction timeouts, and audit coverage.
- Run a replay workload approximating 50 channels.

Gate: no P0 security bypass, unbounded queue, whole-file JSON hot write, or
capture-blocking VLM path remains.

### Day 10: pilot gate

- Deploy candidate to a clean environment.
- Run restart, backup/restore, worker crash, DB outage, and model outage tests.
- Begin 24-48 hour soak.
- Record supported cadence, queue concurrency, retention, disk forecast, and
  known limitations.

Sprint completion requires the deployment candidate and acceptance evidence, not
only merged code.

## Agent loop change

The fixed eight-round tool limit was removed on 2026-06-08. The loop now
continues until the model returns a non-tool response.

This sprint must add protections that do not depend on a tool-call count:

- user cancellation;
- total turn deadline;
- per-tool timeout;
- repeated identical-call detection;
- actor/model cost and rate budgets;
- bounded tool outputs;
- full tool-run audit.

The later 50-100 channel navigation loop should use hierarchy and batching:
channel inventory, group selection, bounded parallel survey, evidence reduction,
then targeted drill-down. It should not issue one serial tool call per channel.

## Definition of done

- A named operator logs in and can access only granted channels.
- An engineer can plan a probe/prompt change; an authorized user approves it;
  the exact approved change executes once.
- Every sensitive action is attributable in audit data.
- Sensitive reads, searches, image access, exports, and every agent tool run are
  attributable; no silent unaudited path remains.
- Capture remains live while VLM workers are slow or unavailable.
- Summary/probe hot paths no longer rewrite whole JSON files.
- Database migration, backup, restore, and restart are demonstrated.
- Existing and new tests pass.
- The 50-channel replay/soak metrics and limits are documented.

## Expected capacity

Four sub-agents over ten days provide roughly 28-32 useful implementation
agent-days after integration/review overhead, not 40. This backlog is near the
upper edge of that capacity.

If schedule slips, cut in this order:

1. Defer pgvector/detection migration and keep a read-only compatibility adapter.
2. Defer `MediaStore` backfill while stopping new base64 writes.
3. Defer general rate limiting except login and agent tools.

Do not cut users/RBAC, audit, tool authorization, approvals, bounded inference
queue, migrations, or restore testing.
