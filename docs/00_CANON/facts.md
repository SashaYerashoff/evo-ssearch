# Canonical Facts

**This file is the single source of truth for version, schema, and runtime
invariants.** Other docs must reference these facts, not restate them. When a
fact changes, change it here first.

Markers: `[FIELD]` = client-specific, filled only in the internal field-rollout
doc, never in shareable docs. `[VERIFY]` = confirm before relying on it.

Last reviewed: 2026-07-27 (β 0.8.5)

## Product & version

| Fact | Value |
|---|---|
| Product | Luxriot EVA AI |
| Current version | `β 0.8.5` |
| Release class | Production-pilot beta (supervised, closed network) |
| Version source of truth | `VERSION` file; `EVOSSEARCH_APP_VERSION` overrides only if set |
| Previous baseline | `β 0.8.4` |

## Database

| Fact | Value |
|---|---|
| Control plane | PostgreSQL (required in secure deployment) |
| Alembic schema head | `20260801_0011` |
| Code-expected revision | `CURRENT_SCHEMA_REVISION` in `eva_db/settings.py` = `20260801_0011` |
| Migration needed for this working tree | **Yes**: run `alembic upgrade head` |
| Archive store | PostgreSQL, forced in secure mode (`EVOSSEARCH_ARCHIVE_STORE=postgres`) |
| Row-level security | Enabled and forced on `iam`, `agent`, `audit`, `archive` schemas |
| Runtime DB roles | Separate DSNs for API, audit, worker, migration |

## Authentication & access

| Fact | Value |
|---|---|
| Auth model | Named users + role-based access (admin / engineer / operator / viewer) |
| Legacy admin-token | **Not** the current auth model; do not document it as current |
| Channel scope | Per-user channel grants; all-channel grant supported |
| Audit | Sensitive endpoints and agent tool calls are audited; new events form a tenant-scoped SHA-256 hash chain |

## Runtime model

| Fact | Value |
|---|---|
| WSGI server | Gunicorn, `gthread` worker class |
| Worker count | **1** (required; in-process capture/probe/summary schedulers are not multi-worker safe) |
| App bind | Gunicorn serves plain HTTP on `EVOSSEARCH_HOST:EVOSSEARCH_PORT` (`5000` default) |
| Browser entrypoint | HTTPS/TLS reverse proxy or site TLS boundary `[FIELD]`; office/demo may use HTTP-only internally |
| Browser UI rollout | Legacy is the appliance default during React parity soak. `EVOSSEARCH_UI_MODE=react` changes the default; `/?ui=legacy` remains an emergency per-request fallback |
| Liveness / readiness | `GET /health`, `GET /ready` |
| Inference queue | Code default is off for unconfigured development; the clean appliance installer enables the PostgreSQL queue, one worker, and `/var/lib/eva-ai/inference-spool` |
| Rollup durability | Closed semantic L1–L3 windows are stored as queryable `archive.runtime_state` rows; a bounded hot cache is also flushed by Gunicorn worker hooks (`gunicorn_conf.py`) |

HTTP/TLS invariant: port number alone does not make the app HTTPS. If operators
open EVA AI through HTTPS (reverse proxy or TLS-terminating service), set
`EVOSSEARCH_AUTH_COOKIE_SECURE=true`. If a lab/demo opens the app directly over
plain HTTP, `EVOSSEARCH_AUTH_COOKIE_SECURE` must be `false` for browser login,
and that mode is not client-facing.

## Models & embedders

| Fact | Value |
|---|---|
| Production embedder | CLIP `ViT-B/32` |
| DINO / fusion / Mask2Former segments | Experimental, disabled in production |
| VLM (video-description) model | Configured by `EVOSSEARCH_LM_PROFILE_VLM_MODEL`; the Ventspils eight-channel deployment target is `Qwen3-VL-4B` |
| Agent LM model | Configured by the agent LM profile; the Ventspils eight-channel deployment initially shares `Qwen3-VL-4B` with the VLM under protected admission |
| Optional deep L3 model | A separate proposal-only 9B-class endpoint admitted only in the operator-defined quiet window and deferred by live attention/alert debt |
| Inference topology | VLM on dedicated vLLM host(s); app + CLIP + agent + DB on a separate host `[FIELD]` |

## Supported platform

| Fact | Value |
|---|---|
| OS target | Ubuntu LTS (pilot built/observed on Ubuntu 26.04) |
| Python | `[VERIFY]` pin for production (dev observed 3.14.x post-OS-upgrade) |
| Reproducible deps | `[VERIFY]` constraints/lock + offline wheelhouse for closed-network patches |

## Field-patch invariants

- Patches are applied offline via USB by an on-site operator ("cosmonaut").
- Database must be preserved across patches; a code-only patch must not run migrations.
- Any DB-touching patch must run `alembic upgrade head` explicitly; the installer
  must refuse unsafe startup when DB revision < code-expected revision.
- Every patch must be reversible via `scripts/rollback.sh`.
