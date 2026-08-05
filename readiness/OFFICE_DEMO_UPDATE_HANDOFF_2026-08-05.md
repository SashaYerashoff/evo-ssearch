# Office-demo update and agent troubleshooting handoff

## Report metadata

| Field | Value |
| --- | --- |
| Report date | 2026-08-05 |
| Demo host | `RTX-LLM` |
| Source checkout | `/home/luxriot/projects/eva-ai-agent` |
| Live checkout | `/opt/eva-ai/evo-ssearch` |
| Operational branch | `stable/office-demo` |
| Deployed baseline | `66fd7b8dfc157b071e9dfecbfc0fa84bce97e77f` |
| Application version | `beta 0.8.5` (rendered by the application as `β 0.8.5`) |
| Previous database revision | `20260614_0006` |
| Current database revision | `20260801_0011` |
| Report evidence cutoff | 2026-08-05 10:20 EEST |

This is the development-machine handoff for the office demo update. It records the installation path, failures, recovery, verified runtime state, and the agent defects discovered after the update. It contains no passwords, API keys, generated migration credentials, or complete DSNs.

## Executive summary

The office installation was successfully moved to `stable/office-demo` at commit `66fd7b8`, upgraded from schema `20260614_0006` to `20260801_0011`, and switched to the React UI. The `eva-ai.service` unit is active and enabled, `/health` reports `β 0.8.5`, `/ready` reports `ready`, the React response header is present, PostgreSQL and named runtime roles are ready, and the four intended video-summary streams were restored.

The update was not clean on the first attempt. The installer initially blocked on a missing privileged migration DSN and a false-positive Evo password placeholder check. A temporary, process-only migration identity solved the DSN requirement. The requested fast path intentionally skipped filesystem and PostgreSQL backups. It switched the live Git branch and React assets, then stopped at the database migration because `eva_migrator_login` could not read `public.alembic_version`. This temporarily left the new code and React configuration in place while the service was inactive and the schema was still old. A resume helper granted the migrator group access to `alembic_version`, completed all migrations, set the expected schema revision, restarted the service, and verified the result.

The main confirmed post-update agent failure is a backend validation defect. The archive vision model returned parseable verdict JSON, but repeated one evidence sentence across all snapshots. The generic prose repetition guard rejected that structured result, retried once, rejected the retry, and raised an error before the archive verdict parser could use it. The bounded workflow then submitted the same `describe_frame` batch repeatedly because a failed batch never marks archive vision as completed. This produces the repeated `DESCRIBE FRAME` errors visible in the React research trace and consumes the turn tool budget.

No live configuration, service, database, model, or source changes were made during agent troubleshooting. Only read-only status, source, journal, and LM Studio log inspection was performed.

## Requested operating decision

`stable/office-demo` is the operational mainline for this installation from this update forward. Future work intended for the demo installation should be based on this branch, reviewed and tested on the development machine, then deployed deliberately to the live checkout. Do not restore the pre-switch feature branch over the live checkout unless performing an explicit rollback.

## Installation starting point

The update began from a working office installation with the following relevant state:

- Live application expected by the fast-update preflight: `β 0.8.4`.
- PostgreSQL Alembic revision: `20260614_0006`.
- Live code carried local work on `feature/secure-50-channel-foundation`.
- The source target was `origin/stable/office-demo` at `66fd7b8`.
- The existing environment file and systemd unit were to be preserved.
- The operator explicitly selected a fast update with no filesystem or PostgreSQL backup.
- The new React production bundle was already present in `react-ui/dist`.

The initial database role audit showed:

| Role | Login | Purpose |
| --- | --- | --- |
| `eva_owner` | no | Object owner used through `SET ROLE` in migrations |
| `eva_migrator` | no | Privileged migration group |
| `eva_migrator_login` | yes | Temporary-login identity used for this migration |

`eva_migrator_login` was confirmed as a member of `eva_migrator`.

## Installation and recovery timeline

### 1. Original installer dry-run

The first dry-run used the tracked installer directly:

```bash
sudo /usr/bin/python3 /home/luxriot/projects/eva-ai-agent/scripts/install_eva_083.py \
  --source-dir /home/luxriot/projects/eva-ai-agent \
  --bundle-dir /home/luxriot/projects/eva-ai-agent \
  --app-dir /opt/eva-ai/evo-ssearch \
  --env-file /etc/eva-ai/eva-ai.env \
  --service-name eva-ai \
  --dry-run
```

Positive preflight findings:

- Source version was `β 0.8.5`.
- Existing `/etc/eva-ai/eva-ai.env` would be preserved.
- Evo, PostgreSQL runtime, agent, and VLM settings were present.
- The target filesystem had ample free space.
- Dry-run mode made no changes.

Blocking findings:

1. No privileged migration DSN was supplied. The installer requires either process-only `EVA_INSTALL_MIGRATION_DSN` or persistent `EVA_MIGRATION_DATABASE_DSN` when migration is enabled. It also correctly requires this credential to differ from the runtime database DSN.
2. `EVOSSEARCH_LUXRIOT_PASSWORD` matched the installer's lexical placeholder heuristic even though the operator confirmed the credential was real and the running application could authenticate to the Evo/Luxriot endpoint.

Non-blocking finding:

- No wheelhouse was present. The update would have to reuse the target virtual environment without downloads.

### 2. Process-only migration identity

The database was inspected without exposing credentials. A site helper then:

1. Generated a random temporary password for `eva_migrator_login`.
2. Verified membership in `eva_migrator`.
3. Verified a PostgreSQL connection as `eva_migrator_login`.
4. Exported the generated DSN only as `EVA_INSTALL_MIGRATION_DSN` for the installer process.
5. Removed the password and DSN from the process environment on exit.

The site-specific launcher `/home/luxriot/eva-office-demo-installer.py` waived only the lexical placeholder check for `EVOSSEARCH_LUXRIOT_PASSWORD`. This was based on a successful live `/ready` authentication check. Every other installer gate remained enabled. The wrapped dry-run then passed configuration and privileged-DSN checks.

This wrapper is an emergency site artifact, not a tracked product fix. The main installer should eventually support a first-class, auditable connectivity-verified override rather than runtime monkey-patching.

### 3. Fast branch and React switch

The operator requested the no-backup fast path:

```bash
sudo /usr/bin/bash /home/luxriot/eva-office-demo-fast-apply.sh
```

The helper:

- fetched `stable/office-demo` from the canonical GitHub remote;
- stashed pre-existing live checkout work with message `pre-stable-office-demo-fast-switch-20260805-091945`;
- switched the live checkout to track `origin/stable/office-demo`;
- verified the live commit against the reviewed source commit;
- copied the React production bundle;
- set `EVOSSEARCH_UI_MODE=react` and `EVOSSEARCH_APP_VERSION=β 0.8.5` in the preserved environment;
- deliberately created no filesystem or PostgreSQL backup;
- stopped the service before migration.

The intermediate state was explicitly verified as:

| Check | Intermediate value |
| --- | --- |
| Live branch | `stable/office-demo` |
| Upstream | `origin/stable/office-demo` |
| Live commit | `66fd7b8` |
| Schema | `20260614_0006` |
| UI mode | `react` |
| Version setting | `β 0.8.5` |
| React `index.html` | present |
| Service | inactive |

### 4. Migration permission failure

The first resume attempt failed at `alembic current` with:

```text
psycopg.errors.InsufficientPrivilege: permission denied for table alembic_version
```

Cause: the original office database had created `public.alembic_version` under `postgres` without granting access to `eva_migrator`. Membership in the migration group was valid, and the migrations themselves use `SET LOCAL ROLE eva_owner`, but Alembic must read and update its own revision row before and after individual migrations. Therefore the process failed before migration code could use `eva_owner`.

This exposed a sequencing defect in the fast helper: privileged migration capability was not fully exercised before switching code/assets and stopping the service. Because the fast path had no backup and no restart-on-failure recovery, the installation remained partially transitioned until the resume helper was corrected and rerun.

### 5. Recovery and schema migration

`/home/luxriot/eva-office-demo-resume.sh` recovered the installation by:

1. Verifying branch, upstream, commit, React distribution, environment version, and UI mode.
2. Granting `SELECT`, `INSERT`, `UPDATE`, and `DELETE` on `public.alembic_version` to `eva_migrator`.
3. Creating and verifying another temporary `eva_migrator_login` password.
4. Running Alembic with the temporary DSN.
5. Updating `EVA_DB_EXPECTED_SCHEMA_REVISION=20260801_0011`.
6. Restarting `eva-ai.service`.
7. Waiting for health and verifying version, readiness, schema, and the `X-EVA-UI: react` response header.

The completed migration chain was:

| Revision | Change |
| --- | --- |
| `20260725_0007` | Durable operator feedback for archived VLM alerts |
| `20260726_0008` | Compact attention telemetry and scheduler audit storage |
| `20260726_0009` | Stable VLM batch membership index |
| `20260727_0010` | Tenant-scoped audit hash-chain preparation |
| `20260801_0011` | Durable incident records and incident operator permission |

Current readiness independently confirms both current and expected revision `20260801_0011`.

### 6. Channel restoration and warm-up

Immediately after restart, the UI briefly showed no channels. The channels later appeared without another deployment change. The current readiness response reports all four desired video-summary streams restored:

| Channel | Configured selector/model label | Batch size |
| --- | --- | --- |
| 129 | `qwen3.5-9b-mtp` | 12 |
| 420 | `vlm-rtx6000` | 12 |
| 453 | `vlm-rtx6000` | 12 |
| 455 | `vlm-rtx6000` | 12 |

The service journal repeatedly recorded Luxriot `/channels` stream read timeouts while retaining a stale channel cache. Current `/ready` nevertheless reports the Luxriot endpoint reachable with HTTP 200 and all four configured streams restored. The brief empty-channel UI state was therefore consistent with restart/warm-up plus intermittent inventory refresh timeouts, not permanent configuration loss.

## Current verified system state

Evidence below was collected read-only at the report cutoff.

| Component | Current state |
| --- | --- |
| Source branch | `stable/office-demo` |
| Source baseline before this report commit | `66fd7b8dfc157b071e9dfecbfc0fa84bce97e77f` |
| Source upstream | `origin/stable/office-demo`, also at `66fd7b8` before this report commit |
| Live branch/upstream | `stable/office-demo` / `origin/stable/office-demo` |
| Live deployed commit | `66fd7b8` |
| Git state before report creation | clean |
| Service | active, enabled, `SubState=running` |
| Service start | 2026-08-05 09:32:07 EEST |
| Service restarts since start | 0 |
| Web process | Gunicorn on `0.0.0.0:5000`, one worker, 32 threads |
| `/health` | `status=ok`, `version=β 0.8.5` |
| `/ready` | `status=ready`, `version=β 0.8.5` |
| UI | HTTP 200 and `X-EVA-UI: react` |
| PostgreSQL | ready, strict runtime roles enabled |
| Database revision | current=`20260801_0011`, expected=`20260801_0011` |
| Runtime DB identity | `eva_api_login` |
| Audit DB identity | `eva_audit_login` |
| Archive store | PostgreSQL, reachable |
| Authentication | ready |
| Luxriot | reachable at readiness check |
| Video-summary restore | 4 desired, 4 restored, 0 failed |
| Embedder | OpenAI CLIP `ViT-B/32` on CUDA, loaded |

### LM topology

All three configured logical profiles currently resolve to the same LM Studio endpoint and actual loaded model:

| Logical profile | Kind | Endpoint | Actual model |
| --- | --- | --- | --- |
| `agent` | agent | `http://127.0.0.1:1234/v1` | `qwen3.5-9b-mtp` |
| `vlm-rtx6000` | VLM | `http://127.0.0.1:1234/v1` | `qwen3.5-9b-mtp` |
| `qwen/qwen3-vl-4b` | VLM | `http://127.0.0.1:1234/v1` | `qwen3.5-9b-mtp` |

LM Studio was observed running the model with multimodal support and MTP speculative decoding. Bounded VLM calls set `enable_thinking=false`; LM Studio responses showed zero reasoning tokens, so hidden thinking is not the cause of the current archive error. Sharing one model endpoint across agent, interactive VLM, and background VLM workloads may affect contention and quality, but it is not required to explain the deterministic validation failure documented below.

### Current non-blocking warnings

These conditions do not currently make `/ready` fail but should be kept in the dev backlog:

- The runtime security diagnostic still says `EVOSSEARCH_LUXRIOT_PASSWORD appears to be a placeholder`. The operator confirms it is a real credential and Luxriot connectivity succeeds. This is a false-positive weakness heuristic, separate from the installer's lexical heuristic.
- `EVOSSEARCH_AUTH_COOKIE_SECURE` is false. This is appropriate only while the appliance is not serving authenticated traffic behind TLS; it must be true behind TLS.
- The optional inference queue and `vlm_vision` readiness components are disabled.
- The attention scheduler is disabled, while the live capture/video-summary paths are running independently.
- Fast VLM alerts recorded `fast VLM episode has fewer than two evidence frames` after one submitted trigger.
- Realtime probe bookmark evaluation has recently reported a stale semantic apex.
- Luxriot channel inventory refreshes have intermittently timed out, with stale-cache retention.

## Recovery assets and rollback notes

The operator explicitly waived filesystem and PostgreSQL backups for this fast update. There is therefore no update-time database dump or filesystem archive to rely on.

Git recovery artifacts do exist:

- The live checkout reported a stash created from `feature/secure-50-channel-foundation` with message `pre-stable-office-demo-fast-switch-20260805-091945`.
- The source checkout currently has a stash with message `pre-stable-office-demo-switch-2026-08-05`.

Do not blindly apply either stash to `stable/office-demo`. Inspect each stash in its own checkout and original branch context first. The live stash requires the `eva` account and `/opt/eva-ai/evo-ssearch` checkout.

One-off site helpers remain outside the repository on the demo host:

| Path | SHA-256 |
| --- | --- |
| `/home/luxriot/eva-office-demo-installer.py` | `d14c4ee004d8ad2d4dcbfa45619369076d6b8c08ee9d3522132894d2daa72693` |
| `/home/luxriot/eva-office-demo-db-installer.sh` | `7c98aa84c6b00cceeeb237f82f7fd280257c444f856a300fef82f5bcf66405e8` |
| `/home/luxriot/eva-office-demo-live-apply.sh` | `8821d2d20f06b8b140caed99aea4b259c3c5019ed10445bac6ca451414a3f1e1` |
| `/home/luxriot/eva-office-demo-fast-apply.sh` | `36015b2e6d329c3a678cdb62402d58c637731cdce62c79c3c3ca7bee65529c1b` |
| `/home/luxriot/eva-office-demo-resume.sh` | `80dcc4b3dfa5f6ad0a01df3c708345419ec54f43743dd0b5d037b0a6dd9b536f` |

These hashes record what was used on this host. The helpers should not be treated as a supported installer or copied into product flows without review.

## Agent issue investigation

### User-visible symptoms

The post-update operator transcript contains several distinct symptoms:

1. An archive search for a red car returned ranked channel 129 candidates, then displayed repeated `DESCRIBE FRAME` failures:

   ```text
   LM response rejected after one guarded retry: repeated sentence (finish_reason=stop)
   ```

2. The research trace showed the same failed visual-verification step repeatedly until the turn ended.
3. Some archive and alert requests produced no visible narrative.
4. An all-channel latest-alert request returned an unsupported channel 211/no-alert conclusion even though active summary channels were known.
5. A Dublin Road follow-up asking for a deeper dive fell back to a broad multi-channel inventory and generic L1 samples rather than the selected event.
6. A chair search reported eight visually verified frames while the displayed research trace exposed only the archive search step, indicating either trace rendering/collapse mismatch or missing surfaced tool detail.

Only symptoms 1 and 2 have a confirmed root cause in this investigation. Symptoms 3-6 are retained as separate grounding, continuation-routing, response-recovery, or UI trace issues requiring development-machine reproduction.

### Confirmed defect A: the prose repetition guard rejects structured verdict JSON

The archive vision batch asks the VLM for one JSON object containing one verdict per snapshot. In the observed red-car run, the raw LM response was parseable and contained:

- 8 verdict objects;
- 8 `match` verdicts;
- 1 unique `visible_evidence` string copied across all 8 objects.

The repeated evidence was:

```text
A red car is clearly visible in the center of the frame, driving on a street with tram tracks.
```

The generic `_lm_repetition_issue` function in `oldapp.py` operates before the archive parser. For responses of at least 400 characters, it splits the response into sentence/line units and rejects a long unit when it occurs three times. Although its docstring says it avoids penalizing structured JSON arrays, it excludes only the named `BATCH_STATE_JSON`, `ALERTS_JSON`, and `MEMORY_UPDATE_JSON` suffixes. It does not detect or exclude the standalone `{"verdicts": [...]}` archive contract.

Execution order:

1. `agent.py::_describe_detection_batch` builds a bounded 1-9 image request and requires verdict JSON.
2. Its LM callback reaches `oldapp.py::_call_lm_chat`.
3. `_call_lm_chat` receives parseable verdict JSON but runs `_lm_repetition_issue` first.
4. The repeated JSON evidence triggers `repeated sentence`.
5. The guarded retry uses temperature 0 and a higher repetition penalty, but the model again copies a concise evidence sentence across similar frames.
6. `_call_lm_chat` raises `RuntimeError` after the retry.
7. `_describe_detection_batch` never reaches `_extract_first_json_mapping`, schema normalization, or verdict return.

Relevant source locations at baseline `66fd7b8`:

- `oldapp.py:4140-4169` - generic repetition detection;
- `oldapp.py:4327-4345` - pre-parser rejection and guarded retry;
- `agent.py:6900-6938` - archive JSON contract and LM call;
- `agent.py:6937-6978` - parser and normalized verdict result that are never reached on this error.

This is a backend false positive. The raw output may still be low-quality because its evidence is copied, but it should be handled by contract-aware verdict validation, not discarded as runaway prose before parsing.

### Confirmed defect B: a failed required vision batch is resubmitted until budget exhaustion

After `search_archive` succeeds, `_remember_turn_tool_result` records candidate IDs, sets `archive_vision_required=true`, and sets `archive_vision_completed=false`. The deterministic archive workflow then requires `describe_frame` on that bounded candidate list.

Only a successful result with `source=archive_candidate_batch` and `vision_checked=true` sets `archive_vision_completed=true`. An exception produces an error tool result but records no failed/attempted terminal state. On the next loop iteration, `_required_bounded_workflow_tool_call` sees the same incomplete state and creates the same required batch again with a new call ID. Failed reads are not inserted into the successful read cache, so the duplicate-read guard does not stop them.

Relevant source locations:

- `agent.py:10940-10964` - deterministic reissue while vision is incomplete;
- `agent.py:11691-11718` - success-only archive state transitions;
- the tool execution exception path records an error for the ledger/UI but does not terminate or latch this workflow step.

This directly explains why a single archive search produces multiple identical `DESCRIBE FRAME` failures and why the final answer lacks usable visual verdicts.

### Findings ruled out as the immediate cause

- **Database/schema failure:** ruled out for the current agent run. `/ready` confirms PostgreSQL and revision `20260801_0011` are ready.
- **React UI selection:** ruled out. The root response is HTTP 200 with `X-EVA-UI: react`.
- **Missing no-thinking configuration:** ruled out. Bounded VLM calls set `chat_template_kwargs={"enable_thinking": false}`, and LM Studio reported zero reasoning tokens.
- **Late system-message placement:** ruled out in the current stable code. `_coalesce_system_messages` collects system content into one leading system message immediately before both tool-decision and final streaming requests (`agent.py:2370-2410`, `agent.py:2543`, and `agent.py:2627`). The older stashed system-order patch is therefore already covered by a broader implementation.
- **MTP speculative decoding:** enabled, but not established as causal. Do not reload the model or disable MTP on the demo host merely to address the deterministic JSON/guard defect.

### Additional agent findings requiring reproduction

#### Grounding of all-channel alert answers

The transcript contains an all-channel alert response about channel 211 and no active alert streams, conflicting with the already observed active summary channels. Determine whether the wrong tool was selected, a prior session result leaked into synthesis, the fallback formatter used stale context, or the model invented the channel in final prose. Add an assertion that every channel ID and alert count in the final answer is present in a current-turn trusted result.

#### Follow-up/continuation routing

The Dublin Road follow-up should have retained channel 420 and the selected overtaking event. Instead, the recovery response broadened into inventory and L1 samples across multiple channels. Reproduce session continuation state, selected event references, and `_required_video_research_tool_call` behavior. The acceptance behavior is a bounded channel-420/time-window drill or an explicit request for a missing timestamp, never an unexplained multi-channel expansion.

#### Missing final narratives and fallback quality

Some requests show only operator messages or research trace steps. Confirm whether the server emitted an empty/incomplete final stream, the React SSE consumer dropped text events, or completion recovery returned no operator text. When recovery is required, the fallback must remain scoped to the operator's channel/query and clearly state which tools failed.

#### Research trace completeness

The chair-search narrative claimed visual verification of eight candidates, while the visible trace showed only `SEARCH ARCHIVE`. Compare backend SSE `tool_call`/`tool_result` events with React trace rendering, especially required server-owned calls and collapsed research messages. The UI should expose at least one bounded batch verification step and its candidate count without duplicating it per candidate unless that is intentional.

#### Observability gap

The exact archive-tool rejection was visible in the UI and inferable from LM Studio raw predictions, but it was not present as a clear structured service-journal error. Add request/session/call IDs and safe structured logging for tool name, workload, retry reason, final error class, and bounded candidate count. Do not log images, thumbnails, credentials, complete DSNs, or private tool payloads.

## Recommended development fixes

### Priority 0: archive verdict validation

1. Parse the archive verdict JSON before applying generic prose-loop detection.
2. Validate the contract explicitly:
   - one verdict per supplied snapshot;
   - unique and in-range `snapshot_index` values;
   - verdict in `match`, `no_match`, or `uncertain`;
   - bounded non-empty evidence;
   - no omitted snapshots.
3. Keep the prose-loop guard for unstructured output and narrative portions.
4. Decide contract-aware handling for duplicated evidence. Reasonable options are to accept it with a quality flag, downgrade indistinguishable/copied entries to `uncertain`, or perform one targeted repair request. Do not turn valid verdict JSON into an internal tool error solely because frames share a sentence.

### Priority 0: failed-step circuit breaker

1. Record archive vision as attempted, with `succeeded`, `failed`, or `unparsed` status.
2. Allow only the internal guarded LM retry inside one `describe_frame` execution.
3. Do not resubmit the identical batch in the same agent turn after that execution fails.
4. Produce a grounded partial response: candidates were ranked, visual verification failed, and presence/absence cannot be concluded.

This changes deterministic tool-loop termination behavior and is grammar-affecting. Before implementation, follow `AGENTS.md` and review `docs/tuktuk/grammar_pin.md`. The intended change should remain within the pinned `RANK -> DRILL -> TERM` flow: the arguments remain extractable from the prior search result, while a failed `DRILL` must terminate safely rather than loop.

### Priority 1: regression tests

Add tests covering:

- parseable `verdicts` JSON with the same long evidence sentence repeated across 8 rows;
- an actual unstructured prose loop that must still be rejected;
- fenced and unfenced JSON;
- malformed, missing, duplicate, and out-of-range snapshot indices;
- a failed archive VLM batch executing once, not repeatedly to the turn limit;
- a failed visual batch producing no positive or negative visual claim;
- React trace rendering for one bounded batch with candidate count;
- current-turn channel-ID grounding for all-channel alerts;
- channel/event preservation on the Dublin Road follow-up.

### Priority 1: installer hardening

1. Test the privileged DSN against `public.alembic_version` during preflight, before changing the live checkout or stopping the service.
2. Ensure a failed fast apply restarts the last runnable service state and reports the exact partial transition.
3. Provide a supported connectivity-verified path for a password that a heuristic considers weak, without logging or persisting it.
4. Unify installer and runtime weak-secret diagnostics so one accepted site credential does not pass one gate and continue warning in another without explanation.
5. Produce a wheelhouse for genuinely offline reproducible upgrades instead of depending on the installed target virtual environment.
6. For no-backup mode, make the irreversible risk explicit and require confirmation in a supported updater rather than a one-off helper.

### Priority 2: topology and performance isolation

After the deterministic software fixes, evaluate whether the agent and background VLM should use separate loaded model instances, admission resources, or endpoints. Only then perform a controlled MTP on/off comparison using identical prompts and images. Do not infer an MTP defect from the current logs alone.

## Development-machine reproduction plan

1. Check out `stable/office-demo` including this report commit.
2. Read `docs/tuktuk/grammar_pin.md` before changing tool-loop state or termination behavior.
3. Reproduce the repetition false positive with a mocked LM response containing eight valid verdict rows and one repeated evidence sentence.
4. Reproduce the retry storm with a mocked `describe_frame` exception after a successful `search_archive`; count executions until the current turn terminates.
5. Implement contract-aware validation and the failed-step circuit breaker.
6. Run focused agent, LM-profile, archive-tool, and React trace tests.
7. Run the broader unit suite relevant to `agent.py` and `oldapp.py`.
8. Exercise the four transcript scenarios in a non-demo environment:
   - Dashcam/red car archive search;
   - man sitting on a chair;
   - latest VLM alerts across all channels;
   - Dublin Road overtaking follow-up.
9. Verify every factual channel, timestamp, count, and verdict in final prose is traceable to a current-turn tool result.
10. Deploy only during a maintenance window because code delivery requires a service restart. Ask the operator to paste the required `sudo` deployment/restart commands; do not assume unattended privilege.

## Acceptance criteria for the next demo build

- One archive query causes one `search_archive` and at most one bounded `describe_frame` tool execution, with its internal retry hidden as implementation detail.
- Valid verdict JSON reaches the archive parser even when similar frames legitimately share evidence wording.
- A failed visual drill ends with an explicit, scoped unknown result and does not exhaust the turn budget.
- No final answer claims visual presence or absence without parsed verdicts for the reviewed candidates.
- All-channel alerts mention only channels returned by current-turn trusted data.
- Follow-up questions retain the selected channel/event/time scope.
- The React research trace displays server-owned required steps accurately and without misleading duplication.
- Tool failures are available in safe structured logs with request/session/call correlation.
- Installer preflight proves Alembic revision-table access before mutating the live installation.
- The supported updater leaves the service running or performs a documented rollback when an update step fails.

## Final handoff state

The demo is operational on `stable/office-demo` with React enabled and schema `20260801_0011`. The current archive-agent error is understood well enough to fix on the development machine without experimenting on the live demo. The first engineering task should be the JSON-aware archive verdict guard plus the failed-step circuit breaker, followed by the scoped grounding and continuation regressions recorded above.
