# Luxriot EVA AI

AI monitoring for Luxriot Evo: EVA continuously **describes** each camera channel
with a vision-language model, raises **alerts** on public-order/safety events,
makes the evidence **semantically searchable**, and lets an operator **ask**, in
plain language, what happened — with evidence and explicit coverage.

Production-pilot beta. Current version: see [`VERSION`](VERSION) and
[docs/00_CANON/facts.md](docs/00_CANON/facts.md).

## Documentation

Start here — the docs are the source of truth, not this README:

- **Canon (authoritative facts):** [facts](docs/00_CANON/facts.md) ·
  [glossary](docs/00_CANON/glossary.md) ·
  [config reference](docs/00_CANON/config_reference.md)
- **Operators:** [operator guide](docs/operator/operator_guide.md) ·
  [scenarios](docs/operator/operator_scenarios.md) ·
  [agent capabilities](docs/operator/agent_capabilities.md) ·
  [demo runbook](docs/operator/demo_runbook.md)
- **Admins:** [admin guide](docs/admin/admin_guide.md) ·
  [observability](docs/admin/observability.md) ·
  [backup & recovery](docs/admin/backup_recovery.md)
- **Install:** [deployment guide](docs/install/deployment_guide.md) ·
  [Git install/update](docs/install/git_install_084.md) ·
  [inference topology](docs/install/inference_topology.md)
- **Architecture:** [system](docs/architecture/system_architecture.md) ·
  [cognitive](docs/architecture/cognitive_architecture.md) ·
  [security/threat model](docs/architecture/security_threat_model.md) ·
  [data retention & privacy](docs/architecture/data_retention_privacy.md)
- **Limits & history:** [known limitations](docs/known_limitations.md) ·
  [CHANGELOG](CHANGELOG.md) · [historical snapshots](readiness/history/)

## What it is

- **Video-description-first.** The VLM describes every channel continuously; that
  perception and its alerts are the center of reports and the agent.
- **Semantic archive.** Sampled frames are CLIP-indexed; search by text/image.
- **Agent.** Read-only investigation over summaries and archive, with coverage
  contracts and evidence; sensitive actions are preview/approval-gated.
- **Probes (CLIP/P-N-M)** are a secondary, mostly agent-invoked semantic tool.

## Install (summary)

Secure pilot on Ubuntu LTS, PostgreSQL, named auth + RLS, single Gunicorn worker,
separate VLM/agent inference hosts. Full steps:
[deployment guide](docs/install/deployment_guide.md). Required settings and all
variables: [config reference](docs/00_CANON/config_reference.md).

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# configure /etc/eva-ai/.env (secrets, 0600); alembic upgrade head; ./run_prod.sh
```

> Auth is **named users + RBAC + row-level security**. The legacy
> `EVOSSEARCH_ADMIN_TOKEN` is **not** the production auth model; do not rely on it
> for the secure pilot.

## Repository

Core: `oldapp.py`, `config.py`, `luxriot_connector.py`, `agent.py`,
`probe_manager.py`, `archive_store.py`, `inference_queue/`, `eva_db/`,
`migrations/`, `static/`, `templates/`, `scripts/`, `tests/`. Run production via
`run_prod.sh` (single worker, `gunicorn_conf.py` durability hooks).

## License & notices

Third-party dependency and model-weight licensing is being formalized for GA —
see [docs/legal/dependency_license_audit.md](docs/legal/dependency_license_audit.md).
