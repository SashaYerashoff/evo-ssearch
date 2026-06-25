# Secure 50-channel foundation status

Date: 2026-06-09
Branch: `feature/secure-50-channel-foundation`

## Parallel implementation result

| Stream | Owner | Implemented now | Still required |
|---|---|---|---|
| Integration | Lead | Readiness branch merged locally; auth and route policy wired; live PostgreSQL role flow verified | State import/cutover, backup/restore, pilot acceptance |
| Database | DB sub-agent + lead | Bounded pool, Alembic, schemas, least-privilege roles, RLS, append-only audit writer, `/ready` integration | State repositories, backup/restore, customer deployment principals |
| IAM | IAM sub-agent + lead | PostgreSQL identities/sessions, Argon2id, login/logout/me UI, secure cookies, CSRF, route and channel policy | User administration, durable distributed throttling, complete route matrix review |
| Agent safety | Agent-safety sub-agent + lead | Existing loop uses role-filtered `ToolGateway`; channel grants, audit, safe schemas, preview-safe writes, owned sessions, PostgreSQL durable one-time approvals for apply actions, operator Apply route/card | Broader UX polish and stale-diff conflict checks |
| Data plane | Data-plane sub-agent + lead | PostgreSQL bounded queue, `SKIP LOCKED`, leases, retries, idempotency, coalescing, overload metrics, separate worker role, Luxriot L0 summary admission/worker runtime with file spool | GPU concurrency controls, operational dashboards, soak tuning |

## Verified

- Clean PostgreSQL 16 upgrade, downgrade, and repeat upgrade.
- Runtime roles have no superuser, database creation, role creation, or RLS
  bypass rights.
- Tenant A cannot read tenant B rows through the API role.
- Audit rows reject mutation.
- `/ready` reports the expected schema revision.
- API, audit writer, and queue worker pass with three separate login principals.
- Luxriot L0 summaries can be detached from capture, enqueued durably, consumed
  by worker leases, applied back into summary history, and protected from spool
  leaks on coalesced/evicted heartbeat work.
- Luxriot queue runtime passes a separate-role gate: `eva_api` admits work,
  `eva_worker` claims/completes it, and the result is applied without sharing
  worker privileges with the API login.
- Agent apply operations now have durable PostgreSQL action plans and hashed
  one-time approvals. Preview calls do not need approval; `preview=false`
  requires a stored plan/approval and executes stored arguments only.
- Durable approvals passed an `eva_api` login gate: raw approval token stayed
  out of the database, approval was consumed once, and the plan was marked
  executed before the side effect.
- Agent preview cards now carry plan metadata only and render an `Apply`
  command. The UI calls a server-side route that approves and executes stored
  arguments; it never sends apply arguments or approval tokens.
- The rendered browser JavaScript passes `node --check`.
- Agent tools fail closed when durable audit is unavailable.
- Agent sessions and channel-bearing HTTP resources enforce user ownership.
- Full live suite before Luxriot runtime wiring: 129 tests passed against
  PostgreSQL 16.
- Fast local suite after Luxriot runtime and approval UI wiring: 139 tests
  passed.
- Disposable PostgreSQL 16 live suite after durable approval wiring: 52 tests
  passed, plus the separate-role Luxriot runtime and `eva_api` approval gates.
- Alembic `downgrade base -> upgrade head` passed through revision
  `20260609_0003`.

## Next execution order

1. Run the 50-channel replay/soak gate and record queue/GPU/storage limits.
2. Add stale-diff checks for apply operations where legacy handlers can expose
   current object versions.
3. Import agent/probe/prompt/summary state with an idempotent comparison report.
4. Add user administration, session revocation, and shared login throttling.
5. Add backup/restore automation and perform a clean restore drill.

The ethical and engineering invariant remains: touching sensitive system state
must leave an attributable, durable fingerprint. No model output is trusted as
identity, authorization, scope, or approval.
