# EVA AI GTM Backlog Triage - 2026-06-04

Purpose: meeting-ready comparison of `readiness/GTM_BACKLOG.md` against the current codebase and the hardening already started.

Current working branches of note:
- `hardening/readiness-low-hanging`: `/health`, `/ready`, route smoke-test update, Chrome Agent stream fix.
- `bench/llamacpp-lab`: independent llama.cpp hardware benchmark lab.

## Executive Summary

The backlog is mostly real, not hallucinated. The strongest framing for the meeting is:

- The product is demo-strong, but pilot risk is concentrated in operability, legal/data clarity, and security evidence.
- We already closed the lowest-effort operational gap: liveness/readiness endpoints and a Chrome Agent UX blocker.
- Some backlog wording should be tightened: a few items are partially implemented, but not production-grade.
- A colleague can help meaningfully if given bounded, evidence-producing tasks instead of broad architecture work.

## Comparative Table

| ID | Backlog claim | Current reality | Already improved | Recommended next action | Good delegate task? |
|---|---|---|---|---|---|
| L-1 License + EULA | Missing legal licensing story blocks customer/legal review. | Confirmed: no root `LICENSE`; EULA needs legal/product decision. | Not touched. | Decide license posture; create root `LICENSE`; legal drafts EULA. | Yes, if colleague can collect requirements/template, but final owner must be legal/lead. |
| L-2 Model attribution / NOTICES | Need model provenance and licenses. | Confirmed gap, but "all models permissive" should be verified, not assumed. README names CLIP, DINO, Mask2Former, SigLIP2. | Not touched. | Build `NOTICES.md` with exact package/model, source URL, license, usage mode. Verify each model card. | Strong yes. Bounded research/doc task. |
| L-3 Data governance + retention | Snapshots and vectors may be PII/biometric-adjacent; need retention/deletion/export policy and tooling. | Confirmed: `detections_archive/`, SQLite, image paths, CLIP/DINO vectors. Retention exists, but deletion/export/purge policy does not. | Not touched. | Draft policy first; implement admin purge/export endpoint/tool second. | Yes for policy inventory and data map. Backend purge tool should be reviewed by us. |
| R-1 Structured logging | Many `print()` calls and raw traceback; weak remote debugging. | Confirmed: prints in `oldapp.py`/`config.py`, `traceback.print_exc()` in search path. | Not touched. | Add app logger, request id, JSON-ish production output; replace high-risk prints first. | Partial. Good for mechanical print inventory; implementation needs careful review. |
| R-2 Health/readiness | Missing `/health` and `/ready`. | Was true. | Done on `hardening/readiness-low-hanging`: `/health`, `/ready` with DB/embedder/Luxriot checks; tests updated. | Review semantics: should `/ready` require embedder preloaded by default, or only with `?load=1`? | Done except review. Not a good remaining task. |
| R-3 Luxriot connector resilience | Single-shot HTTP calls, no retry/backoff, short stop join. | Mostly true. There are HTTP timeouts already, but no retry/backoff. `join(timeout=0.75)` exists. | Not touched. | Add retry/backoff around channels/snapshot/bookmark; separate "stop requested" from "capture thread died"; keep status visible. | Yes if scoped to retry/backoff tests; needs integration review. |
| R-4 LLM streaming watchdog | Hung LM stream can hang SSE. | Partially true. Requests timeouts exist, but streaming read timeout can be very long and no stalled-token watchdog. Frontend Chrome stream completion bug fixed. | Chrome UI bug fixed on `hardening/readiness-low-hanging`: frontend unlocks on SSE `done`. | Backend watchdog: abort if no stream chunk/heartbeat after N seconds; surface clean SSE error. | Yes for small backend patch if tests are required. |
| R-5 Graceful shutdown + process model | No proper SIGTERM drain; worker/VRAM topology undocumented. | Mostly true. There is `atexit` cleanup and `run_prod.sh` Gunicorn, but no documented safe topology or drain semantics. | Not touched. | Document supported process model now: 1 worker, gthread, GPU model memory caveats, shutdown behavior. Implement drain later. | Strong yes for documentation. |
| R-6 Basic metrics | No metrics endpoint. | Confirmed: no `/metrics` or Prometheus dependency. | Not touched. | Add minimal counters first: requests, latency, Agent/Luxriot/probe events. GPU metrics can be later. | Yes, if kept minimal and no huge observability framework. |
| A-1 Thread-safe embedder state | Globals reassigned without a single lock. | Confirmed: CLIP/DINO globals and settings save can reset runtime state while threads may use it. | Not touched. | Add an embedder service/lock before broad concurrency or multi-worker claims. | No for "half participation"; too easy to regress runtime ML paths. |
| A-2 Decompose `oldapp.py` | Monolith blocks maintainability. | Confirmed: ~6400 LOC mixing routes, ML, daemon, persistence. | Not touched. | Do not start a rewrite before pilot. Extract only where it reduces immediate risk. | No as a broad task. Could delegate route inventory only. |
| A-3 Test suite + CI | Smoke tests narrow, no CI. | Confirmed. Existing smoke tests now pass; route scan was stale and was fixed. | Improved: tests now 13/13, route scan reads Python + HTML + JS. | Add CI running current smoke tests; then add focused unit tests for retention/probes/agent tools. | Strong yes. Good bounded task. |
| A-4 Lock dependencies | Unpinned heavy deps, no lockfile; `setuptools<81` workaround fragile. | Confirmed. Test run still warns about `pkg_resources` deprecation. | Not touched. | Pick `uv` or pip-tools; generate lock on known-good environment; document GPU install caveat. | Strong yes. Good DevOps task. |
| A-5 Containerize | No Dockerfile. | Confirmed. `run_prod.sh` exists but no repeatable image. | Not touched. | Start with CPU/dev Dockerfile or NVIDIA runtime draft; do not promise full GPU image until tested. | Yes if colleague has Docker/GPU runtime experience. Otherwise documentation-only. |
| A-6 Data layer hardening / Postgres path | SQLite + unversioned migrations will hit scale limits. | Confirmed directionally. Store has locks and SQLite timeouts, but migrations are inline `ALTER TABLE`. | Not touched. | Add migration version table first; document Postgres/pgvector path for larger deployments. | Partial. Migration versioning can be delegated; Postgres design needs senior review. |
| S-1 Real auth/authz | Shared admin token only; no roles/users. | Confirmed. Mutating endpoints have shared token guard, but no users/roles/entropy policy. | Not touched. | Pilot: enforce token presence/entropy and document. GA: users/roles. | Yes for token entropy check + docs. Real auth should not be half-owned. |
| S-2 Rate limiting | Expensive/mutating endpoints unthrottled. | Confirmed. | Not touched. | Add simple per-token/per-IP limits for Agent, search, indexing, probe run. | Yes, if using a small dependency or simple in-memory limiter for pilot. |
| S-3 Audit logging | No audit trail for sensitive changes. | Confirmed. Sessions/messages are persisted, but not an actor/action audit log. | Not touched. | Add append-only audit events for settings, probes, bookmarks, prompt changes, skills. | Strong yes. Bounded backend/data task. |

## What We Already Pulled Up

| Area | Result | Branch / commit |
|---|---|---|
| Health/readiness | Added `/health` and `/ready`; `/ready` reports DB, embedder, Luxriot status. | `hardening/readiness-low-hanging` / `f972efb` |
| Smoke tests | Route test now scans `oldapp.py`, `templates/index.html`, `static/js/app.js`; tests pass. | `hardening/readiness-low-hanging` / `f972efb` |
| Agent Chrome UX | Fixed Chrome-on-Ubuntu issue where chat looked complete but next send was blocked. | `hardening/readiness-low-hanging` / `e185677` |
| Hardware benchmark lab | Separate llama.cpp lab for local GPU profiling and capacity estimates. | `bench/llamacpp-lab` / `1022569` |

## Suggested Scope For The Colleague

Good tasks for "part-time but useful" contribution:

| Task | Why it fits | Expected output |
|---|---|---|
| Model/license attribution inventory | Bounded research; low risk to product behavior. | `NOTICES.md` draft with model/package, source URL, license, usage notes, uncertainty flags. |
| Data map for governance | Bounded evidence task; helps legal and purge tooling. | Table of persisted data: file/DB/table, contains PII?, retention config, deletion gap. |
| CI for current smoke tests | Clear pass/fail output; low architecture risk. | `.github/workflows/tests.yml` running `python -m unittest discover -s tests`. |
| Dependency lock draft | Useful and self-contained. | `requirements.in` + generated lock or `uv.lock`, plus notes on CUDA/GPU install assumptions. |
| Process model doc | Needs careful writing more than deep code changes. | `docs/ops/process_model.md`: Gunicorn workers/threads, VRAM caveats, startup/shutdown notes. |
| Audit event inventory | Pre-work for S-3 without risky implementation. | List of sensitive actions and payload fields to log, mapped to endpoint/tool. |
| Rate-limit proposal | Easy to review before code. | Short design: endpoints, limits, keying by token/IP, failure response. |

Avoid delegating as broad tasks:

| Area | Reason |
|---|---|
| Embedder thread-safety refactor | Touches ML runtime, settings reset, probe daemon, search paths. High regression risk. |
| `oldapp.py` decomposition | Too easy to turn into a rewrite. Needs a narrow extraction plan. |
| Real auth/users/roles | Product/security design decision, not just code. |
| Postgres/pgvector migration | Needs scale assumptions and deployment target clarity. |

## Meeting Position

Recommended line:

> The audit is broadly correct. We already closed the obvious operational hole and one Chrome Agent blocker. The best way for you to help is not to take a vague chunk of architecture, but to own evidence-producing readiness items: NOTICES, data map, CI, process model, audit/rate-limit plan. That gives us customer-test credibility without destabilizing the demo.

## Near-Term Priority

Before broader customer tests:

1. Merge/review `hardening/readiness-low-hanging`.
2. Add `LICENSE`/EULA direction and `NOTICES.md`.
3. Add data governance one-pager + deletion/export/purge plan.
4. Add CI for current smoke tests.
5. Add minimal audit log and rate limits for mutating/expensive endpoints.
6. Patch Luxriot retry/backoff and LLM stalled-stream watchdog.
