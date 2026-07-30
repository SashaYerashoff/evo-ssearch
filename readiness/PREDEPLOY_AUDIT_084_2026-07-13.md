# EVA AI β 0.8.4 pre-deploy audit — 2026-07-13

Scope: `stabilization/0.8.3-pre-react`, local secure dev deployment, live Luxriot
Evo, PostgreSQL, Qwen3-VL-4B/vLLM and Qwen3.5-9B-MTP/llama.cpp. This is a
release decision record, not a claim that the Tbilisi host has the same
hardware or runtime state.

## Verified green

- Version: `β 0.8.4`; schema: `20260614_0006` (single Alembic head, code-only
  upgrade from the field baseline).
- Combined deterministic gate: 626 passed, 18 skipped, 134 subtests passed.
- Live admin agent acceptance: all runnable structural scenarios passed;
  seed-only and non-admin-only cases were intentionally skipped.
- Both configured models match the actually served model IDs. VLM reports an
  8192-token context; llama.cpp agent reports 32768 through `meta.n_ctx`.
- `/health`, authenticated `/ready?details=1`, PostgreSQL strict runtime roles,
  archive/runtime stores, Luxriot reachability and model endpoints are healthy.
- Live same-origin MJPEG broker returned a continuous token-authenticated stream.
- Durable rollup store and backfill resume survived an EVA process restart.
- No tracked field/dev credential remains in the live integration example.
- Local `.env` and TLS private key are mode 0600. Browser responses now include
  nosniff, SAMEORIGIN, no-referrer and camera/microphone/geolocation denial.

## Defects fixed in this audit

- Scoped users no longer receive other channels through runtime/desired/problem
  fields of `list_video_summary_channels`.
- Deployment-global restoration status is hidden from channel-scoped actors
  instead of leaking other channel IDs or presenting misleading partial totals.
- `runtime_only` no longer performs one historical summary scan per channel.
- Exact EN/RU runtime-status questions expose only the authoritative runtime
  inventory tool, reducing the local agent prompt by about 10k tokens.
- Operator-relative time is authoritative for `generate_report`, as it already
  was for video-summary tools.
- Malformed native tool calls no longer raise on a missing function name or
  non-object arguments.
- llama.cpp `data[].meta.n_ctx` is shown by LM admission.
- Rollup scheduler yields to saturated L0 queues only when they share its LM
  resource; a busy VLM endpoint no longer idles the separate text/agent GPU.
- Genuine 0.8.0/0.8.1 L1-L3 LM summaries are adopted as labelled legacy
  semantics and promoted to independently queryable durable rows without model
  regeneration. Mechanical fallback cards are not promoted as semantics.
- Text-only L1-L3 work is routed to the dedicated agent profile. Background
  rollups and the interactive agent request direct/non-thinking output; this
  preserves completion budget for tool calls and final operator answers.
  Scheduler spacing is start-to-start with a five-second
  default, avoiding a mathematically impossible 50-channel idle budget.
- A completed backfill with transient LM failures can be explicitly retried;
  true source gaps remain terminal/idempotent.
- Rollup failures now retain a bounded, credential-redacted provider detail.
- Local capture config no longer has conflicting 15/60 second segment values;
  it now runs 15 second segments with a 15 second read timeout.
- Field updater now rejects bad HTTP post-checks, preserves stderr/rollback
  handoff, preserves an adopted hardened systemd unit, restores staging env on
  failure, gates exact version/clean bundle/dependencies, and updates only the
  release-managed version key.
- The updater can validate a pip-less venv with `uv pip check` when available.

## Release decisions required before packaging

1. **Durable L0 queue.** `/ready` reports the inference queue disabled. Pending
   L0 batches are in process memory and reset on reboot. Under the live VLM +
   rollup/backfill load CH120 reached queue 2/2 and dropped 48 frames before the
   scheduler fix. Decide whether the field release enables PostgreSQL workers
   now or explicitly accepts bounded/coalesced loss.
2. **Preview concurrency.** A VLM alert-policy preview stores the full old policy
   plus the new criterion. Another operator can edit the policy before Apply and
   be overwritten. Add a revision/hash precondition (409 stale preview) or an
   atomic append operation.
3. **Legacy auth mode.** With `AUTH_ENABLED=false`, agent mutations can bypass
   the secure adapter's preview/Apply lifecycle. The field installer should
   require named auth, or the runner must force preview independently of auth.
4. **Archive playback acceptance.** The local Evo returned `archive_gap` for
   recent CH112/CH120 evidence and correctly fell back to the stored frame.
   Therefore a real recorded channel must prove that the modal plays and loops
   the requested ~15 second batch before release.
5. **Bookmark semantic dedupe.** Exact normalized title+severity dedupe works,
   but wording variants such as “performs burnout” / “performing burnout” can
   still create two bookmarks within the 600 second window. Prefer a stable
   event key in the alert schema.

## Operational warnings

- Local desired-state restoration is working; persisted desired video state at
  the final audit contains CH112 and CH120. Absence of any other channel after
  reboot is configuration state, not a restore failure.
- The local filesystem is 86% used (63 GiB free). The repo is 23 GiB, mostly a
  14 GiB local inference environment plus a 3.3 GiB Git object store. Git has
  about 1.8 GiB of unreachable packed blobs; cleanup is safe only after a backup.
- PostgreSQL is about 3.5 GiB: roughly 95k detection rows (3.2 GiB with images/
  vectors) and 400k audit events (238 MiB). Retention is active: 90-day rows,
  14-day thumbnails.
- RTX 4060 runs near its 8 GiB VRAM ceiling (~7.5 GiB). Do not colocate another
  CUDA process on it. The local agent model uses ~9-11 GiB host memory including
  mmap pages; current system availability is adequate but not generous.
- Local user services have `Linger=no`; they start after user login, not as a
  headless boot guarantee. Field system services/inference units need an actual
  reboot acceptance.
- The local self-signed TLS certificate expires 2026-09-21. Field certificate
  expiry must be checked independently.
- The field weak Evo password is intentionally rejected by secure installer
  preflight. Change it on site or make an explicit, documented exception; do not
  silently weaken the placeholder gate.

## Field acceptance before the engineer leaves

- Reboot; verify EVA, PostgreSQL, every VLM endpoint and the agent endpoint are
  active without a desktop/LM Studio click.
- Confirm all intended desired channels restore, then change batch/interval,
  restart summaries, and verify the running values rather than only controls.
- Run 50-channel inventory, CH112/118/120 status, one archive period query, one
  L1/L2 drill, one agent report, one explicit CLIP probe request and one VLM
  alert-policy preview/Apply.
- Toggle bookmarks off/on and prove zero/single Evo delivery respectively.
- Play a real recorded 15 second archive batch in the modal and leave it looping.
- Observe L0 queue, VLM admission, dropped/coalesced batches and rollup state for
  at least one complete L1 interval under concurrent agent use.
- Preserve diagnostics, `/ready?details=1`, installer evidence and rollback
  command before disconnecting.
