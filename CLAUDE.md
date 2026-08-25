# EVA AI Repository Orientation

This is the engineering map for coding assistants working in this repository.
Canonical product and deployment facts live in `docs/00_CANON/facts.md`; do not
copy changing versions, schema revisions, or field topology into this file.

## What EVA AI Is

EVA AI is a video-description-first monitoring layer for Luxriot Evo. It:

- captures bounded evidence from live channels and local video sources;
- reduces dense CV into compact attention/homeostasis signals;
- sends activity-sensitive L0 batches to a vision-language model;
- consolidates L0 into durable L1/L2/L3 semantic memory;
- raises and archives alerts with operator false-positive feedback;
- indexes sparse CLIP evidence for probes and semantic retrieval;
- gives an operator an evidence-grounded agent for archive and stream research.

The original folder-image search remains as a secondary/lab surface. It is not
the product architecture.

## Read Before Changing Things

- Current facts and runtime invariants: `docs/00_CANON/facts.md`
- Configuration: `docs/00_CANON/config_reference.md`
- System/data flow: `docs/architecture/system_architecture.md`
- Security model: `docs/architecture/security_threat_model.md`
- Retention/privacy: `docs/architecture/data_retention_privacy.md`
- Deployment: `docs/install/deployment_guide.md`
- Known limits: `docs/known_limitations.md`

Before changing agent tools, tool result compaction, intent groups, or
`skills/*/SKILL.md`, read `AGENTS.md` and `docs/tuktuk/grammar_pin.md`. The
pinned tuktuk grammar and extractability law are design gates. Record a real
conflict in `docs/tuktuk/grammar_review_questions.md`; do not silently bend the
grammar or implementation.

## Main Components

- `oldapp.py` — Flask HTTP surface, auth guards, archive/probe routes, runtime
  assembly, and the legacy UI backend.
- `luxriot_connector.py` — channel inventory, live capture, adaptive L0
  batching, alerts/bookmarks, attention integration, and L1–L3 scheduling.
- `agent.py`, `agent_research.py`, `agent_security/` — operator agent, evidence
  workflow, tool loop, approvals, and policy.
- `archive_store.py`, `attention_store.py` — PostgreSQL-backed evidence,
  summaries, feedback, probes, and compact attention telemetry.
- `inference_queue/`, `lm_admission.py` — bounded model admission, durable
  inference spool/retry, and workload isolation.
- `security/`, `eva_db/`, `migrations/` — named auth/RBAC/RLS, audit,
  tenant-scoped persistence, and schema revisions.
- `static/`, `templates/` — the current single-page operator UI.
- `scripts/` — preflight, install/update/rollback, acceptance, and live-soak
  utilities.

## Runtime Invariants

- Production uses one Gunicorn worker. Capture, probe, rollup, attention, and
  retention schedulers are in-process and are not multi-worker safe.
- Secure deployment uses named authentication, PostgreSQL, forced RLS, separate
  runtime/audit/worker DSNs, and a privileged migration DSN.
- The app binds plain HTTP behind the deployment TLS boundary. Trust forwarded
  headers only for the explicitly configured local proxy hops.
- Appliance mode is offline by default. Missing model artifacts must fail
  closed rather than download at runtime.
- Dense CV images are not an archive. One sparse embedding snapshot per
  configured cadence links compact motion/probe signals to VLM evidence.
- L0 work is bounded to 16 snapshots and a 60-second non-empty flush. Durable
  queue loss must create an explicit coverage-gap record.
- `BATCH_STATE_JSON` is the single machine state contract for L0 continuity,
  cover selection, pass-up memory, and alerts.
- Settings that are runtime-safe must reach active sessions. Environment-backed
  changes must be reported as restart-required.
- Global skill files are system-admin-managed process-wide playbooks, not
  tenant prompt-editor content.

## Local Development

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-test.txt
bash scripts/check_docs_drift.sh
python -m pytest -q
python oldapp.py
```

The default server bind in code is for development. Use `run_prod.sh` and the
deployment guide for a pilot appliance. Do not use the historical
`scripts/field_upgrade_084.sh` for the current release.

Tests requiring real PostgreSQL, Luxriot, VLM, or agent endpoints are skipped
unless their explicit environment variables are present. Never turn a skipped
live test into a passing claim.

## Change Discipline

- Preserve unrelated working-tree changes and local evidence files.
- Add a regression test for every bug fix; for browser behavior, prefer an
  executable browser test (Playwright harness: `docs/ui/README.md`).
- UI screenshots in the guides are generated from `docs/ui/shots.json` via
  `scripts/ui_shots.py`. Never hand-take one, and never commit a screenshot
  containing real evidence frames.
- Run `scripts/check_docs_drift.sh` whenever canonical facts, deployment, tools,
  permissions, or schema revisions change.
- Schema changes require an Alembic revision plus updates to the code-expected
  head, installer/preflight gates, canonical facts, and release notes.
- Do not put credentials, raw images, prompts, tokens, or DSNs in audit details,
  logs, fixtures, screenshots, or committed configuration.
- Do not claim model accuracy, capacity, latency, TLS validation, or soak
  stability without a measured run on the intended hardware and topology.
