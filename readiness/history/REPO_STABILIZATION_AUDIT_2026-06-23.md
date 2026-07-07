# EVA AI repository stabilization audit - 2026-06-23

Branch under review: `feature/secure-50-channel-foundation`  
Working tree state: dirty; this report is based on the local repository state on 2026-06-23.

## Executive summary

The branch is much closer to a production pilot than the original PoC: named users,
RBAC, CSRF, audit logging, PostgreSQL-backed runtime state, archive search, VLM
profile balancing, live-summary restore, and agent tool gating are all present.
The architecture direction is correct.

The branch is not yet stable enough to call green:

- full unit suite is red: `254 tests`, `1 failure`, `1 error`, `18 skipped`;
- dependency/runtime setup is not reproducible after the Ubuntu 26.04 upgrade;
- several “hidden” client features are hidden only in the UI, not disabled server-side;
- VLM/agent evidence handling has a few concrete correctness bugs;
- runbooks/docs still mix PoC-era SQLite/admin-token assumptions with the secure beta;
- local runtime artifacts and generated TLS files sit in the repo tree and can confuse patch/bundle work.

The most important stabilization target is not a large refactor. It is tightening
contracts, removing misleading legacy paths from the active deployment surface,
and making the release path deterministic.

## Verified checks

Commands run locally:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m benchmarks.eva_coherence.runner --scenario all
git diff --check
node --check static/js/app.js
curl -k -sS https://127.0.0.1:5443/ready
```

Results:

- `unittest discover`: failed with 1 failure and 1 error.
- `benchmarks.eva_coherence`: passed, 3/3 scenarios.
- `git diff --check`: passed.
- `node --check static/js/app.js`: passed.
- Local EVA AI service is running on `https://127.0.0.1:5443`.
- `/ready` is `not_ready` only because `deployment_security` rejects weak Luxriot secret; database, auth, embedder, Luxriot, and LM profiles are reachable/ready.

Runtime versions observed after the Ubuntu 26.04 refresh:

```text
Python 3.14.4
flask 3.0.3
torch 2.12.1+cu130
torchvision 0.27.1+cu130
transformers 5.12.1
numpy 2.5.0
opencv 4.13.0
faiss 1.14.3
Pillow 12.2.0
requests 2.34.2
gunicorn 26.0.0
psycopg 3.3.4
SQLAlchemy 2.0.51
alembic 1.18.4
```

## Current branch and repository shape

Tracked changes are large:

```text
18 tracked files changed
5496 insertions, 1828 deletions
deleted: detection_store.py
deleted: tests/test_agent_store_security.py
deleted: tests/test_detection_store_security.py
```

Important untracked files/directories:

```text
.local/
agent_postgres_store.py
benchmarks/
readiness/*client/runbook/release docs*
scripts/build_patch_bundle.sh
scripts/client_diagnostics.sh
scripts/install_patch.sh
scripts/rollback.sh
scripts/set_site_ips.sh
scripts/verify_patch.sh
skills/cross_channel_correlation/
skills/multi_channel_event_sweep/
skills/video_event_check/
skills/video_incident_timeline/
```

Large/hot modules:

```text
static/js/app.js        ~11665 lines
oldapp.py              ~11480 lines
static/css/app.css      ~8785 lines
agent.py                ~5375 lines
luxriot_connector.py    ~4370 lines
archive_store.py        ~1292 lines
```

This is manageable for a pilot but fragile for repeated hot fixes. The
stabilization strategy should prioritize contract tests and small guardrails over
large modular refactors before the next field patch.

## P0 / must fix before calling branch stable

### 1. Full test suite is red

Evidence:

- [tests/test_agent_tool_loop.py](/home/sasha/Projects/evo-ssearch/tests/test_agent_tool_loop.py:160)
- [agent.py](/home/sasha/Projects/evo-ssearch/agent.py:3844)
- [tests/test_api_dataflow_smoke.py](/home/sasha/Projects/evo-ssearch/tests/test_api_dataflow_smoke.py:388)

Failures:

```text
TypeError: _FakeStore.touch_session() got an unexpected keyword argument 'tenant_id'
AssertionError: luxriot_snapshot_capture POST mutates but is not guarded
```

Interpretation:

- `AgentStore` contract became tenant-aware, but the fake store in tests still
  implements the old shape.
- `/luxriot/snapshot/<id>/capture` is centrally protected as a sensitive route,
  but the smoke-test classifier treats POST as mutation and does not accept the
  sensitive-route guard.

Fix:

- update `_FakeStore.touch_session/add_message/...` to accept `tenant_id`,
  `actor_id`, and owner kwargs;
- either add explicit mutation guard to `luxriot_snapshot_capture`, or update
  the smoke contract to classify this endpoint as safe POST/read-like snapshot
  capture with `streams:view`.

### 2. Dependency stack is not reproducible

Evidence:

- [requirements.txt](/home/sasha/Projects/evo-ssearch/requirements.txt:5) uses broad `>=` pins for heavy ML/runtime dependencies.
- Local runtime moved to Python 3.14.4 and CUDA 13 wheels after OS upgrade.
- `clip-anytorch` emits `pkg_resources` warnings, and torch warns that
  `torch.jit.load` is not supported in Python 3.14+.

Risk:

Closed-network deployments cannot depend on whatever `uv/pip` resolves that day.
This is especially risky for CLIP, torch, torchvision, transformers, faiss, and
opencv.

Fix:

- add a tested constraints/lock file for Ubuntu 26.04;
- decide whether prod supports Python 3.14 or pins Python 3.12/3.13;
- build an offline wheelhouse for patch bundles if dependencies change;
- add a small `scripts/verify_dev_env.sh` / `scripts/verify_runtime_env.sh`.

### 3. Offline patch runbook can install code without migrations

Evidence:

- [scripts/install_patch.sh](/home/sasha/Projects/evo-ssearch/scripts/install_patch.sh:346) runs migrations only when requested.
- [readiness/OFFLINE_USB_PATCH_OPERATOR_RUNBOOK_RU.md](/home/sasha/Projects/evo-ssearch/readiness/OFFLINE_USB_PATCH_OPERATOR_RUNBOOK_RU.md:103) does not force `--run-migrations`.

Risk:

This is exactly the “cosmonaut with a flash drive” failure mode: code is updated,
database is not, service boots into confusing runtime errors.

Fix:

- installer should compare current DB revision with code head and refuse unsafe
  startup unless migrations were run or explicitly waived;
- runbook must show exact command with `--run-migrations` for any DB-touching patch.

### 4. Gunicorn multi-worker mode is unsafe for current state model

Evidence:

- [run_prod.sh](/home/sasha/Projects/evo-ssearch/run_prod.sh:6) allows `EVOSSEARCH_GUNICORN_WORKERS > 1`.
- [wsgi.py](/home/sasha/Projects/evo-ssearch/wsgi.py:4) starts background daemons at import.
- [oldapp.py](/home/sasha/Projects/evo-ssearch/oldapp.py:3607) restores desired live sessions at import.
- [luxriot_connector.py](/home/sasha/Projects/evo-ssearch/luxriot_connector.py:430) keeps live sessions in memory.

Risk:

Multiple gunicorn workers can duplicate capture loops, probes, restored live
sessions, and in-memory state. The current safe production profile is one worker
with threads.

Fix:

- fail fast in secure/prod mode if `EVOSSEARCH_GUNICORN_WORKERS != 1`;
- longer term: split API and capture/probe/VLM workers with a DB lease/singleton.

## P1 / high priority stabilization bugs

### 5. `describe_frame(detection_id=...)` can route to live snapshot

Evidence:

- [agent_security/eva_adapter.py](/home/sasha/Projects/evo-ssearch/agent_security/eva_adapter.py:407) resolves detection ownership by adding `channel_id`.
- [agent.py](/home/sasha/Projects/evo-ssearch/agent.py:2373) prioritizes `channel_id` live snapshot before `detection_id`/`image_path`.

Risk:

The agent can ask to describe an archived detection and get the current live
camera frame instead. That breaks evidence trust.

Fix:

- in `_describe_frame`, prioritize `detection_id` and `image_path` before live
  `channel_id`;
- keep channel ownership as metadata, not as source selection.

### 6. L0-L3 summary windows can drift to processing time

Evidence:

- batch has `batch_start_ms`/`batch_end_ms` in [luxriot_connector.py](/home/sasha/Projects/evo-ssearch/luxriot_connector.py:3437).
- normalization currently derives alert/signal timestamps from `created_at` in [luxriot_connector.py](/home/sasha/Projects/evo-ssearch/luxriot_connector.py:1013).
- L0 nodes derive `created` from `created_at` in [luxriot_connector.py](/home/sasha/Projects/evo-ssearch/luxriot_connector.py:3138).

Risk:

Under queue delay/load, the operator may see events attributed to the time of
processing, not the time of the frames. This directly hurts “what happened when”
queries.

Fix:

- persist `batch_start_ms` and `batch_end_ms` in summary history;
- build L0 node windows and evidence anchors from frame timestamps;
- keep `created_at` only as processing/ingest time.

### 7. Structured alerts are coupled to auto-bookmarks

Evidence:

- [luxriot_connector.py](/home/sasha/Projects/evo-ssearch/luxriot_connector.py:2219) appends the JSON alert prompt only if `bookmark_enabled`.
- [config.py](/home/sasha/Projects/evo-ssearch/config.py:535) defaults auto bookmarks to false.

Risk:

If bookmarks are disabled, summaries can stop producing `ALERTS_JSON`. That
breaks severity badges, rollup alert counts, and downstream evidence even though
the system still needs structured alert extraction.

Fix:

- always request structured `ALERTS_JSON` from VLM summaries;
- gate only the side effect of sending Luxriot bookmarks.

### 8. Server-side disable flags are missing for hidden deployment features

Status: fixed in the stabilization pass after this audit. The server now has
`EVOSSEARCH_OFFLINE_VIDEO_ENABLED`, `EVOSSEARCH_PROBE_SNAP_ENABLED`, and
`EVOSSEARCH_INDEXED_FOLDER_ENABLED`; disabled endpoints return 404 and template
visibility uses the same flags.

Original evidence:

- Offline Video Analysis and Probe Snap are hidden in UI via deployment CSS/template.
- Server endpoints remain available:
  - [oldapp.py](/home/sasha/Projects/evo-ssearch/oldapp.py:7434) `/video_understanding`
  - [oldapp.py](/home/sasha/Projects/evo-ssearch/oldapp.py:8329) `/luxriot/snapshot/<id>/capture`

Risk:

For client deployment this is not disabled functionality, only hidden controls.
Users with route knowledge or automation can still call it.

Fix:

- add server-side feature flags:
  - `EVOSSEARCH_OFFLINE_VIDEO_ENABLED=false`
  - `EVOSSEARCH_PROBE_SNAP_ENABLED=false`
  - optionally `EVOSSEARCH_INDEXED_FOLDER_ENABLED=false`
- return 404/403 when disabled and render UI from the same flags.

### 9. Settings `.env` writing is not robust enough

Evidence:

- [oldapp.py](/home/sasha/Projects/evo-ssearch/oldapp.py:10007) serializes env as `KEY=value`.
- [oldapp.py](/home/sasha/Projects/evo-ssearch/oldapp.py:10505) writes `.env` directly.
- local `.env` permissions are `664`.

Risk:

Passwords/DSNs containing `#`, spaces, quotes, or newlines can corrupt `.env`.
Secrets can be group/world-readable on local/dev systems.

Fix:

- use dotenv-compatible quoting/escaping;
- write through temp file + `os.replace`;
- set mode `0600`;
- add tests with a Luxriot password/API key containing special characters.

### 10. Archive retention can orphan JPEG files

Evidence:

- [archive_store.py](/home/sasha/Projects/evo-ssearch/archive_store.py:340) `_trim_to_cap()` deletes old rows but does not return `image_path`.
- filesystem cleanup exists only in [oldapp.py](/home/sasha/Projects/evo-ssearch/oldapp.py:5290) through retention results.

Risk:

At 50 channels, orphaned previews can slowly fill disk while DB row counts look
healthy.

Fix:

- remove insert-time `_trim_to_cap()` or route it through the same deletion path
  that returns image paths;
- add orphan-free retention test;
- add scheduled/startup retention pass, not only opportunistic pruning on writes.

## API and contract audit

Observed:

- Flask routes: 73.
- frontend fetch call sites: 80.
- Central route security maps are in [oldapp.py](/home/sasha/Projects/evo-ssearch/oldapp.py:221) and [oldapp.py](/home/sasha/Projects/evo-ssearch/oldapp.py:249).

Good:

- sensitive and mutating route protection is centralized;
- auth routes self-guard;
- route-level security tests exist and are broad.

Gaps:

- no generated OpenAPI/JSON schema contract;
- frontend/backend shape contracts are mostly implicit;
- model catalog can cache fallback after pre-login `/lm/models` 401:
  - [static/js/app.js](/home/sasha/Projects/evo-ssearch/static/js/app.js:431)
  - [static/js/app.js](/home/sasha/Projects/evo-ssearch/static/js/app.js:596)
  - [oldapp.py](/home/sasha/Projects/evo-ssearch/oldapp.py:11305)
- agent skill detail GET exposes skill content to any `agent:use` user, while
  save/create are `prompts:manage`.

Recommended contract tests:

- `/luxriot/rollups`
- `/luxriot/streams`
- `/luxriot/snapshot/<id>/capture`
- `/detections/list`
- `/detections/search_text`
- `/detections/search_image`
- `/settings/archive_capacity`
- `/agent/sessions`
- `/agent/skills/<slug>`
- `/lm/models` before and after login.

## Database and security audit

Verified good:

- Alembic head is `20260614_0006`.
- `eva_db/settings.py` expects `CURRENT_SCHEMA_REVISION = "20260614_0006"`.
- PostgreSQL runtime roles are separated for API/audit/worker.
- RLS is enabled and forced on `iam`, `agent`, `audit`, `archive`.
- archive store is forced to Postgres in [config.py](/home/sasha/Projects/evo-ssearch/config.py:268).
- unavailable Postgres archive fail-closes through `_UnavailablePostgresStore` in [oldapp.py](/home/sasha/Projects/evo-ssearch/oldapp.py:505).
- agent sessions are backed by [agent_postgres_store.py](/home/sasha/Projects/evo-ssearch/agent_postgres_store.py:37).

Risks:

- live PostgreSQL tests are optional and skipped without `EVA_TEST_DATABASE_DSN`;
- runtime JSON fallback still exists for Luxriot summary/rollup state:
  - [luxriot_connector.py](/home/sasha/Projects/evo-ssearch/luxriot_connector.py:1168)
  - [luxriot_connector.py](/home/sasha/Projects/evo-ssearch/luxriot_connector.py:2657)
- archive insert errors are partly swallowed in [archive_store.py](/home/sasha/Projects/evo-ssearch/archive_store.py:258);
- local `.env` is `664`;
- root runtime files still exist:
  - `detections_store.sqlite3` 254 MiB
  - `agent_sessions.sqlite3` 1.6 MiB
  - `probes_store.json` 1.5 MiB
  - `luxriot_summary_state.json`
  - `luxriot_rollups_cache.json`

Recommendation:

- for secure mode, remove disk fallback or gate it behind an explicit one-shot
  import command;
- make live PostgreSQL security tests mandatory in release CI;
- add startup warning/fail if SQLite/JSON runtime artifacts are detected in
  secure deployment root;
- do not use broad `TRUNCATE archive.runtime_state` in client cleanup because it
  stores prompt settings, desired live sessions, and runtime summary state.

## Agent and VLM audit

Good:

- agent tool loop can exceed the old 8-tool limit;
- coherence benchmark passes:
  - Orlandina absent/present transition;
  - multi-channel confirmation workflow;
  - sensitive wording for visible dog tags instead of vaccination claims;
- route/user/channel authorization is wired into tool execution;
- new `get_visual_window_signals` concept exists for CLIP P/N/M-style signals.

Critical correctness items:

- `describe_frame(detection_id=...)` source priority bug, see P1 item 5.
- L0-L3 frame-time vs processing-time drift, see P1 item 6.
- structured alerts tied to bookmark flag, see P1 item 7.

Further quality issues:

- evidence frames for `get_video_summaries` are selected for the whole period,
  not per event/window;
- channel name resolution in [agent.py](/home/sasha/Projects/evo-ssearch/agent.py:2990) reads `self._lxm.channels`, which is fragile unless channels were preloaded;
- rollup source selection in [luxriot_connector.py](/home/sasha/Projects/evo-ssearch/luxriot_connector.py:2880) samples evenly and can drop rare alert/deviation source lines under budget;
- 50-channel overload still risks looking quiet: dropped heartbeat batches are
  counted, but operator-visible gap/coverage UI is thin;
- probe snap is still not synchronized with VLM frame buffers.

Recommendation:

- fix the three correctness bugs first;
- then add “coverage contract” to agent answers: period requested, period
  inspected, levels used, entries available, entries returned, evidence frames
  attached, and whether the result is complete/truncated;
- prioritize alert/deviation source lines before routine in rollups.

## Dependencies and runtime audit

Issues:

- no lock/constraints file for prod;
- `requirements.txt` uses broad ranges for critical packages;
- `.venv` after Ubuntu 26.04 contains Python 3.14.4;
- CLIP stack emits Python 3.14 warnings;
- `.venv/bin/python -m pip` is not guaranteed available in the new environment;
- [run_prod.sh](/home/sasha/Projects/evo-ssearch/run_prod.sh:11) prefers global `gunicorn` before `.venv/bin/gunicorn`;
- patch bundle scripts do not build a wheelhouse by default.

Fixes:

- prefer `.venv/bin/gunicorn` in `run_prod.sh`;
- create locked runtime profile for Ubuntu 26.04;
- decide supported Python version;
- add wheelhouse/offline dependency story;
- document venv rebuild after OS upgrade using `uv` or `ensurepip` explicitly.

## Legacy / lost functions / stale surface

Legacy or confusing remnants:

- deleted `detection_store.py`, but root SQLite files still exist;
- deleted SQLite tests, but replacement Postgres live tests are optional;
- old JSON `ProbesStore`/runtime fallback paths remain in the app;
- indexed-folder/FAISS routes are still live:
  - `/check_index`
  - `/index`
  - `/index_segments`
  - `/search`
  - `/search_by_image`
  - `/search_by_mask`
  - `/segment_from_point`
- comments endpoints still exist for indexed-folder workflow;
- docs and presales collateral still mention SQLite/admin token as current.

Recommendation:

- add server-side `EVOSSEARCH_INDEXED_FOLDER_ENABLED=false` for client deploys;
- keep legacy code only as dev/legacy feature if explicitly enabled;
- move old SQLite/JSON docs to “historical PoC” or update them;
- add CI grep guard for forbidden “SQLite current store” claims in current docs.

## Documentation audit

High-impact mismatches:

- [README.md](/home/sasha/Projects/evo-ssearch/README.md:23) still reads like PoC/admin-token docs.
- [readiness/POSTGRES_FOUNDATION_RUNBOOK.md](/home/sasha/Projects/evo-ssearch/readiness/POSTGRES_FOUNDATION_RUNBOOK.md:14) is stale around current archive/runtime state and schema head.
- [readiness/OFFLINE_USB_PATCH_OPERATOR_RUNBOOK_RU.md](/home/sasha/Projects/evo-ssearch/readiness/OFFLINE_USB_PATCH_OPERATOR_RUNBOOK_RU.md:103) misses mandatory migrations for DB-touching patches.
- [readiness/CLIENT_RESTART_AND_IP_CHANGE_RUNBOOK.md](/home/sasha/Projects/evo-ssearch/readiness/CLIENT_RESTART_AND_IP_CHANGE_RUNBOOK.md:249) and release notes still imply live video-description sessions are lost on restart, while code now restores desired sessions.
- [docs/product_marketing_vision_ru.md](/home/sasha/Projects/evo-ssearch/docs/product_marketing_vision_ru.md:86) and [docs/eva_ai_presales_onepager.html](/home/sasha/Projects/evo-ssearch/docs/eva_ai_presales_onepager.html:1855) still mention SQLite/admin-token era architecture.
- [readiness/GTM_BACKLOG.md](/home/sasha/Projects/evo-ssearch/readiness/GTM_BACKLOG.md:3) still presents fixed gaps as current.

Recommendation:

- split docs into:
  - current secure beta architecture;
  - operator runbook;
  - offline patch runbook;
  - historical audit/backlog snapshot.
- Every deployment doc should state:
  - current Alembic head;
  - auth required;
  - Postgres required;
  - how to verify `/ready`;
  - how to verify live session restore;
  - how to rotate/reset admin password;
  - what data cleanup does and does not delete.

## Prioritized stabilization plan

### Next fix batch: before branch stabilization

1. Fix the two red tests.
2. Fix `describe_frame(detection_id)` source priority.
3. Persist/use `batch_start_ms` and `batch_end_ms` for L0 windows.
4. Decouple `ALERTS_JSON` from auto-bookmarks.
5. Add server-side feature flags for offline video, probe snap, and indexed-folder.
6. Force single gunicorn worker in secure/prod mode.
7. Make `.env` writes atomic, quoted, and `0600`.
8. Add `.local/` to `.gitignore` or move local TLS/service scripts outside repo.

### Next patch batch: before the cosmonaut-style field patch

1. Update offline patch runbook and installer migration safety.
2. Add runtime verification script for Python/deps/DB/schema/permissions.
3. Add wheelhouse/constraints story for closed network patches.
4. Add retention cleanup for orphan JPEG prevention.
5. Add `/ready` or diagnostics signal for stale retention and VLM coverage gaps.
6. Update current architecture docs and mark old docs historical.

### After pilot

1. Split `oldapp.py`, `agent.py`, `luxriot_connector.py`, and `static/js/app.js`
   into smaller modules with contract boundaries.
2. Generate OpenAPI or a simple route/schema registry.
3. Move capture/probe/VLM workers out of gunicorn import-time startup.
4. Replace/modernize CLIP backend if Python 3.14 remains the target.
5. Add mandatory Postgres integration CI.

## Bottom line

The secure beta foundation is real: auth, audit, Postgres, RLS, channel grants,
VLM profile routing, live summary restore, archive frame search, and agent tool
guarding are in place.

The branch should not be frozen as stable yet. The highest-value work is a short
stabilization pass that fixes two red tests, three VLM/agent evidence bugs,
server-side feature flags, single-worker enforcement, env-secret handling, and
offline patch safety. That is the shortest path to a branch we can trust for
field patches without remote access.
