# Canonical Facts

**This file is the single source of truth for version, schema, and runtime
invariants.** Other docs must reference these facts, not restate them. When a
fact changes, change it here first.

Markers: `[FIELD]` = client-specific, filled only in the internal field-rollout
doc, never in shareable docs. `[VERIFY]` = confirm before relying on it.

Last reviewed: 2026-06-25 (β 0.8.1)

## Product & version

| Fact | Value |
|---|---|
| Product | Luxriot EVA AI |
| Current version | `β 0.8.1` |
| Release class | Production-pilot beta (supervised, closed network) |
| Version source of truth | `VERSION` file; `EVOSSEARCH_APP_VERSION` overrides only if set |
| Previous baseline | `β 0.8.0` |

## Database

| Fact | Value |
|---|---|
| Control plane | PostgreSQL (required in secure deployment) |
| Alembic schema head | `20260614_0006` |
| Code-expected revision | `CURRENT_SCHEMA_REVISION` in `eva_db/settings.py` = `20260614_0006` |
| Migration needed for 0.8.0 → 0.8.1 | **No** (code-only upgrade) |
| Archive store | PostgreSQL, forced in secure mode (`EVOSSEARCH_ARCHIVE_STORE=postgres`) |
| Row-level security | Enabled and forced on `iam`, `agent`, `audit`, `archive` schemas |
| Runtime DB roles | Separate DSNs for API, audit, worker, migration |

## Authentication & access

| Fact | Value |
|---|---|
| Auth model | Named users + role-based access (admin / engineer / operator / viewer) |
| Legacy admin-token | **Not** the current auth model; do not document it as current |
| Channel scope | Per-user channel grants; all-channel grant supported |
| Audit | Sensitive endpoints and agent tool calls are audited |

## Runtime model

| Fact | Value |
|---|---|
| WSGI server | Gunicorn, `gthread` worker class |
| Worker count | **1** (required; in-process capture/probe/summary schedulers are not multi-worker safe) |
| App port | `5443` (TLS) `[FIELD]` confirm per site |
| Liveness / readiness | `GET /health`, `GET /ready` |
| Inference queue | Present but **disabled by default**; summary dispatch is synchronous in-process |
| Graceful-restart durability | Gunicorn worker hooks flush summary state + rollup cache (`gunicorn_conf.py`) |

## Models & embedders

| Fact | Value |
|---|---|
| Production embedder | CLIP `ViT-B/32` |
| DINO / fusion / Mask2Former segments | Experimental, disabled in production |
| VLM (video-description) model | `qwen/qwen3-vl-4b` |
| Agent LM model | `qwen3.5-9b` class |
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
