# Luxriot EVA AI β 0.8.7 Release Notes

Release date: release candidate prepared 2026-08-06  
Previous baseline: `β 0.8.5`  
Schema head: `20260805_0013`  
Database migration: **required**

`β 0.8.7` closes the first complete incident lifecycle over EVA's adaptive
attention stack and hardens the React, archive, deployment and upgrade paths
used by the office and maritime pilots.

## Incident temporal memory

- Durable identities now separate an observation, a continuous episode, an
  operator case and a recurrence-series relation.
- Independent perception, risk, case and attention states prevent a covered
  return to routine from silently resolving risk or closing a case.
- High-signal L0 events may open bounded operator-review candidates. Exact
  semantic continuity updates the same candidate; unrelated overlapping tracks
  remain distinct.
- L1–L3 carry server-owned incident IDs, episode boundaries and total child
  dispositions. Missing model output becomes `unclassified_keep`; coverage
  gaps never become routine.
- A per-channel 2/4/8 attention policy limits foreground detail, active prompt
  tracks and unresolved hot records without deleting or resolving overflow.
- Append-only episode, relation, observation and lifecycle-transition ledgers
  are replay-safe and protected by optimistic incident revisions.

## Operator and agent workflow

- Incident Review provides Active, Needs review and History queues, grounded
  covers, duration, evidence count, independent states and technical temporal
  history.
- Operators can confirm, resolve, dismiss, mark false-positive or reopen a case
  and can confirm/reject candidate recurrence links without merging cases.
- Follow/Critical focus has a bounded lease and durable final result describing
  continuation, explicit resolution, coverage failure or inconclusive review.
- The EVA agent can read compact incident/temporal context and request the same
  lifecycle or series decisions through the standard preview/apply approval
  gate. Channel ownership, revision and relation candidates are resolved from
  durable state rather than model arguments.
- Markdown and XML exports contain the human synopsis, key moments,
  homeostatic attention statistics, Follow outcome, episode/series projection
  and lifecycle history. P/N/M remains labelled as attention, not visual proof.

## Other field hardening

- SigLIP archive matches are candidates verified in one bounded multi-image VLM
  pass; failed or uncertain vision cannot become a confident negative answer.
- Protocol Deploy, React operator workflow, maritime scene epochs, Latvian UI
  chrome and custom appearance presets are included in this release candidate.
- PostgreSQL archive storage is fail-closed in production and archive paging is
  repaired by migration `0013`.
- Offline deployment reports and installers expect schema `20260805_0013` and
  the React console.
- The universal Ubuntu 24.04 bundle now has one field entry point for a fresh
  appliance or an in-place upgrade. Manifest v2 records the clean source
  commit, fresh/update/report modes, every offline APT/wheel artifact and the
  migration plan. Build-time pip resolution proves the CPython 3.12 + vLLM
  environment without network access; field preflight verifies checksums before
  touching a running EVA. Failed update applies restore the backed-up source,
  environment and PostgreSQL database automatically.

## Upgrade

From the finalized universal bundle run:

```bash
sudo ./START_EVA_AI.sh
```

It detects the installed deployment, runs a read-only preflight, asks for a
privileged migration DSN only when the existing environment does not already
provide one, creates the application/database backup, applies Alembic through
`20260805_0013`, installs rebuilt React assets, verifies health/readiness and
compares live stream progress with the pre-update baseline.

Do not reuse a β 0.8.5 manifest, dependency cache inventory or migration plan
with the β 0.8.7 tree.

## Verification

- Backend: 1227 passed, 23 skipped, 169 subtests passed.
- React: 83 tests passed; TypeScript/Vite production build completed.
- Alembic: one head, `20260805_0013`.
- Documentation drift and whitespace checks: clean.
