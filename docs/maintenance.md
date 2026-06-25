# Documentation Maintenance & Anti-Drift

How the docs stay true. The whole set drifted before because facts were restated
in many places and snapshots mixed with current docs. The rules below keep it
from happening again.

## Principles

1. **Canon is the only source of truth.** Version, schema head, config vars, and
   topology live in [docs/00_CANON/](00_CANON/). Everything else **links**, never
   restates them.
2. **Snapshots go to `readiness/history/`.** Anything point-in-time (audits,
   sprints, status reports) is archival and never edited. On conflict, canon wins.
3. **Markers, not guesses.** `[FIELD]` (client-specific, never in shareable docs),
   `[VERIFY]` (confirm against code/build), `[NEEDS LEGAL]` (PM/legal).
4. **Shareable vs internal.** Sanitized docs (e.g. `deployment_guide.md`) carry no
   client data; the filled versions (`field_rollout_demo.md`) are internal only.

## Anti-drift guard

`scripts/check_docs_drift.sh` fails CI if current docs contain forbidden stale
claims (e.g. a retired store named as current, legacy admin-token framing) or if the canon
disagrees with the code (VERSION / schema head). Run it in CI and before release:

```bash
bash scripts/check_docs_drift.sh
```

Extend the `patterns` list as new stale claims are retired (sparingly, to avoid
false positives).

## Release checklist (doc side)

On every release:
1. Update [`CHANGELOG.md`](../CHANGELOG.md) and add `readiness/RELEASE_NOTES_<v>.md`.
2. Update [facts.md](00_CANON/facts.md): version, schema head (if changed),
   migration-needed yes/no.
3. Update any guide whose behavior changed; update
   [config_reference.md](00_CANON/config_reference.md) for new/changed vars.
4. Move any superseded snapshot into `readiness/history/`.
5. Run `scripts/check_docs_drift.sh`.

## Why accuracy is operational

If the agent self-help feature ships (BM25 over these docs — see
[agent_self_help_design](architecture/agent_self_help_design.md)), a wrong
instruction in a doc becomes a wrong answer from the agent. Treat operator-doc
accuracy as part of the product, not just paperwork.
