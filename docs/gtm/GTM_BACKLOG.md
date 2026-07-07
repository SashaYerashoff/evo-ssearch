# Luxriot EVA AI — Go-to-Market Readiness Backlog

**Status:** Current build is `β 0.8.2` — a production-pilot build for controlled client deployment.
**Premise:** The gap to a sellable product is hardening, operability, and legal/data
clarity — **not features**. Items are ordered by recommended sequence, not just severity.

**Effort key:** S = days · M = 1–2 weeks · L = 3+ weeks

---

## Tier 0 — Commercial & Legal Blockers

Cheap and fast, but fatal if skipped. Must exist before any customer conversation.

### L-1 — Add project license + EULA
- **Effort:** S · **Owner:** Lead + legal
- **Why:** Default state is "all rights reserved," which blocks a clean licensing
  story when a customer's legal team asks what they're buying.
- **Acceptance criteria:**
  - `LICENSE` file present at repo root.
  - Commercial license / EULA drafted defining customer usage rights.

### L-2 — Add model attribution / NOTICES file
- **Effort:** S · **Owner:** Any dev
- **Why:** Undocumented model provenance is a hard no for regulated buyers. All
  bundled models are commercially permissive — they just need documenting.
- **Acceptance criteria:**
  - `NOTICES` (or `THIRD_PARTY_LICENSES`) file lists each bundled model and license:
    CLIP (MIT), DINOv3 / Mask2Former / SigLIP2 (Apache-2.0).
  - File is referenced from the README.

### L-3 — Write data governance & retention policy
- **Effort:** M · **Owner:** Lead + legal
- **Why:** The product persists snapshots of identifiable people plus embedding
  vectors (`detections_archive/`, SQLite) — biometric/PII under GDPR / BIPA / state law.
- **Acceptance criteria:**
  - Written retention, deletion, and export policy.
  - Deletion endpoint/tooling exists and is verified to purge snapshots + vectors + DB rows.

---

## Tier 1 — Reliability & Operability

The biggest real gap — this is what breaks at a customer site. Minimum bar for a paid pilot.

### R-1 — Structured logging
- **Effort:** M · **Owner:** Backend
- **Why:** ~40+ `print()` calls, no structured logging, full tracebacks leaked to
  stdout (`oldapp.py:~4221`). Remote-site debugging is currently blind.
- **Acceptance criteria:**
  - All `print()` replaced with a configured logger (JSON to stdout, log levels).
  - Request/correlation IDs propagated across the threading boundary.
  - No raw stack traces returned to clients or printed in production mode.

### R-2 — Health / readiness endpoints
- **Effort:** S · **Owner:** Backend
- **Why:** No `/health` or `/ready` for load balancers / orchestrators / ops monitoring.
- **Acceptance criteria:**
  - `/health` (liveness) returns 200 when the process is up.
  - `/ready` reports model-loaded + DB-reachable + Luxriot-reachable and fails clearly when not.

### R-3 — Luxriot connector resilience
- **Effort:** M · **Owner:** Backend
- **Why:** Single-shot HTTP calls with no retry/backoff; `thread.join(timeout=0.75s)`
  abandons snapshot capture on a slow VMS (`luxriot_connector.py`). Camera/VMS networks
  are inherently flaky.
- **Acceptance criteria:**
  - Retry with exponential backoff on all Luxriot HTTP calls (channels, snapshot, bookmark).
  - Snapshot-capture join timeout fixed so slow-but-successful captures aren't dropped.
  - Transient VMS outages recover without a process restart.

### R-4 — LLM streaming watchdog
- **Effort:** S · **Owner:** Backend
- **Why:** No timeout around the streaming LM call (`agent.py:~3156`); a hung
  LM Studio/vLLM endpoint hangs the SSE stream indefinitely.
- **Acceptance criteria:**
  - Watchdog/timeout aborts a stalled LM stream and surfaces a clean error to the client.

### R-5 — Graceful shutdown + documented process model
- **Effort:** M · **Owner:** Backend / DevOps
- **Why:** No SIGTERM handling for in-flight requests. GPU model state is per-process —
  multiple Gunicorn workers each load CLIP/DINO into VRAM, so workers can't be scaled naively.
- **Acceptance criteria:**
  - SIGTERM drains in-flight requests before exit.
  - Supported worker/threading topology documented with VRAM implications.

### R-6 — Basic metrics
- **Effort:** M · **Owner:** Backend / DevOps
- **Why:** No observability into throughput or resource use.
- **Acceptance criteria:**
  - Metrics exposed (Prometheus `/metrics` or equivalent): request count/latency,
    search QPS, probe throughput, GPU/VRAM usage.

---

## Tier 2 — Architecture & Maintainability

Required to call it GA. Slows every future fix until done.

### A-1 — Thread-safe embedder state
- **Effort:** M · **Owner:** Backend (senior)
- **Why:** `clip_model` / `dino_encoder` globals are reassigned under `global` with no
  lock while the probe daemon and HTTP threads touch shared state — data-race hazard.
- **Acceptance criteria:**
  - Embedder state wrapped in a thread-safe service/singleton with a single lock.
  - Concurrent re-init / search exercised under test without races.

### A-2 — Decompose the oldapp.py monolith
- **Effort:** L · **Owner:** Backend (senior)
- **Why:** 6,400 lines mixing routes, ML init, caching, persistence, and the background
  daemon. Also sets up the previously-agreed sidecar-extraction path (no rewrite).
- **Acceptance criteria:**
  - Data-plane (embedding, indexing, detection store, probe daemon) extracted behind a
    service boundary, separate from Flask routing.
  - Route handlers moved out of the monolith into per-domain modules.

### A-3 — Real test suite + CI
- **Effort:** L · **Owner:** Backend + QA
- **Why:** ~12 smoke tests, <15% coverage, no CI (`.github/workflows` absent).
- **Acceptance criteria:**
  - Unit tests for embedding pipeline, detection dedup/retention, probe logic, agent
    tool dispatch.
  - CI pipeline runs tests on every PR.
  - Coverage floor enforced (target ≥40%).

### A-4 — Lock dependencies
- **Effort:** S · **Owner:** DevOps
- **Why:** `torch` / `transformers` / `faiss` / `numpy` are unpinned ranges, no lockfile;
  `setuptools<81` is a fragile CLIP workaround. Prod will silently drift from dev.
- **Acceptance criteria:**
  - Committed lockfile (uv or pip-tools) producing reproducible builds.

### A-5 — Containerize
- **Effort:** M · **Owner:** DevOps
- **Why:** No Dockerfile; on-prem security appliance needs a repeatable image.
- **Acceptance criteria:**
  - Dockerfile + documented GPU runtime image; documented build/run instructions.

### A-6 — Data layer hardening / Postgres path
- **Effort:** L · **Owner:** Backend (senior)
- **Why:** SQLite hits write-lock contention with the daemon + threads at scale;
  migrations are unversioned `ALTER TABLE` hacks.
- **Acceptance criteria:**
  - Migration versioning added now while schema is small.
  - Documented Postgres + pgvector migration path for multi-camera/multi-user sites.

---

## Tier 3 — Security Hardening

Scrutinized heavily because this is a security product.

### S-1 — Real auth/authz model
- **Effort:** L · **Owner:** Backend (senior)
- **Why:** Single shared admin token with no entropy requirement; no roles/users.
- **Acceptance criteria:**
  - Per-user authentication with roles.
  - Token/credential entropy enforced.

### S-2 — Rate limiting
- **Effort:** S · **Owner:** Backend
- **Why:** All endpoints unthrottled — runaway-query and abuse risk.
- **Acceptance criteria:**
  - Per-IP / per-token throttling on mutating and expensive endpoints.

### S-3 — Audit logging
- **Effort:** M · **Owner:** Backend
- **Why:** No record of who changed probes, created bookmarks, or edited config.
- **Acceptance criteria:**
  - Audit trail capturing actor + action + timestamp for sensitive operations.

---

## Suggested Sequencing

- **Pilot-ready (weeks):** Tier 0 (L-1 → L-3) + Tier 1 (R-1 → R-5). Mostly hardening
  of pieces that already exist.
- **GA-ready (months):** add A-1 → A-6 and the Tier 3 items.
