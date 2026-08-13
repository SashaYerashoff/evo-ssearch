# EVA AI handoff — Georgia upgrade and inference-preservation work

Date: 2026-08-12 (Europe/Riga)

## Live rehearsal state after single-instance cleanup

Sasha explicitly requested that only the Georgia pilot rehearsal EVA remain
running. The resulting local topology is:

- `eva-ai-georgia-repro.service`: active, one Gunicorn worker, `127.0.0.1:5081`;
- `eva-tbilisi-repro-postgres`: active, `127.0.0.1:15433`;
- `eva-llama-qwen3-vl-4b.service`: active, required VLM, `127.0.0.1:1234`;
- `eva-llama-qwen35-mtp.service`: active, required agent LM, `127.0.0.1:1235`;
- `eva-vlm-vision-watchdog.timer`: active because it protects the required VLM
  and reads the Georgia rehearsal `.env`.

The duplicate `eva-ai-local-5443.service` is stopped and disabled. Its separate
`eva-ai-postgres` container on port 15432 is stopped. Alternative vLLM, Bonsai,
Qwen profiles, the older Tbilisi unit and the Georgia dev unit are inactive. The
old `eva-overnight-probe.timer` was stopped and disabled because its unit was
still tied to the removed 5443/vLLM baseline. Do not restart any of those while
stabilizing this rehearsal.

The rehearsal `.env` was not edited. Its verification hash remains:

```text
2c254527143f62bbdbcf7a14914872e2a6f1e0f4f776ef02024c0f27aac76325
```

After cleanup, `/ready` returned HTTP 200, both local inference `/health`
endpoints returned HTTP 200, about 9.8 GiB RAM was available, swap usage was
1.8/4.0 GiB, and there was no active swap-in/swap-out storm. The unrelated
`slopai-*` Docker stack was left untouched.

Two additional fixes are deployed in the rehearsal and are now committed in
`15ad09f`:

- incident covers now resolve through bounded indexed time windows before JSON
  filtering; an authenticated live check returned grounded covers for all 38/38
  incident cards in 1.872 seconds;
- the probe preview now advances only after image load/error rather than replacing
  an in-flight request every four seconds. The React suite passed 94/94 and the
  production build completed successfully.

Probe embedding backpressure and the recurring live-segment timeout were fixed and
deployed in the final runtime pass documented below. The rehearsal `.env` was not
edited to obtain that result.

## Bootstrap for the next Codex session

Open the session with this repository as its workspace:

```text
/home/sasha/Projects/evo-ssearch-office-demo
```

Then ask Codex to read this file completely and continue from **Next work**.

The stabilization source is committed. Do not reset, clean, stash, rebase, or
discard later user changes; inspect the current tree before editing.

## Current objective

Finish and validate the universal EVA AI `0.8.1 -> 0.8.7` offline update for the
Georgia deployment without changing or destabilizing its existing inference
topology.

The immediate goal is not to optimize the local RTX 4060 rehearsal inference.
The local inference only emulates the production APIs. The production invariant is:

- preserve the exact VLM and agent endpoint URLs;
- preserve model IDs and API keys;
- preserve context limits, timeouts, max-inflight values, queue policy, video
  limits and GPU visibility;
- do not install, restart or rewrite vLLM/llama.cpp/LM Studio services;
- only EVA code, React UI, database schema and bundled application dependencies
  may be upgraded;
- any inference-policy drift must stop the update or trigger automatic rollback.

Session continuation note: Sasha explicitly authorized replacing the **local**
rehearsal VLM with llama.cpp and stabilizing it for channels 112/118. This did not
broaden permission to change either Georgia production inference server or any
production/rehearsal `.env`. The production invariant above remains unchanged.

## Repository state

Primary working repository:

```text
path:   /home/sasha/Projects/evo-ssearch-office-demo
branch: main
latest release-source commit: 1a69745 fix: keep archive batch review immutable
```

The reviewed stabilization tail includes:

```text
15ad09f fix: stabilize Georgia upgrade rehearsal runtime
645f4f5 fix: bound rollup and operator-mode reads
2adc0cb fix: avoid monolithic rollup cache rewrites
8969893 fix: ground probe inventory and compact list payloads
4f5842a fix: make settings provenance explicit
1d6d518 fix: keep probe preview responsive
47c70d5 fix: correlate live probe frames and scores
c6417c7 fix: reuse buffered vectors in probe daemon
8884b86 docs: record backend realtime stabilization
9066706 fix: isolate live semantic capture latency
21412fe fix: align live inference with served capacity
623cec4 fix: preserve service during cold worker reload
4394e6a docs: record readiness-gated reload rehearsal
760fdc2 fix: allow verified cold model startup
8a476dc test: register probe signal frame route
1a69745 fix: keep archive batch review immutable
```

The working tree is expected to be clean after committing this handoff update.
No commit was pushed from this session.

Other repositories:

```text
/home/sasha/Projects/evo-ssearch
  branch: feature/universal-offline-deploy
  HEAD:   66fd7b8
  dirty:  requirements.txt
  role:   older/live development runtime; do not overwrite from office-demo blindly

/home/sasha/Projects/eva-georgia-upgrade-repro
  current installed VERSION: beta 0.8.7
  role: Georgia upgrade rehearsal target
```

## Georgia rehearsal controls

Desktop entry points:

```text
/home/sasha/Desktop/EVA_GEORGIA_UPGRADE_RECOVER.sh
/home/sasha/Desktop/EVA_GEORGIA_UPGRADE_TEST.sh
```

`EVA_GEORGIA_UPGRADE_RECOVER.sh` restores the exact pre-upgrade rehearsal baseline:

```text
EVA version:       beta 0.8.1
database revision: 20260614_0006
archive:           8683 rows, 8683 thumbnails, 5 sidecars
service:           absent (the updater must create it)
venv:              original Georgia environment, OpenCV absent
```

The rehearsal target is currently upgraded to `0.8.7`; run RECOVER before the next
honest upgrade attempt. Do not run RECOVER against any production path.

The last successful rehearsal reached:

```text
health:             HTTP 200, beta 0.8.7
database migration: 20260805_0013
VLM:                qwen/qwen3-vl-4b
Agent:              qwen3.5-9b-mtp
```

Earlier rehearsal failures that have already influenced the installer design:

- legacy React UI was served after upgrade;
- missing OpenCV/Python header/runtime dependencies;
- archive thumbnails/sidecars and old data needed explicit preservation checks;
- migration from `0006` needed a privileged, process-only migration DSN;
- service creation/start and rollback needed to work when the old service was absent;
- CSRF-protected React mutations needed verification after upgrade;
- the updater must not silently inherit or rewrite model configuration.

## Current local runtime (diagnostic only)

Current endpoints after the continuation pass:

```text
http://127.0.0.1:5081  Georgia rehearsal EVA, beta 0.8.7
http://127.0.0.1:1234  qwen/qwen3-vl-4b via llama.cpp on RTX 4060
http://127.0.0.1:1235  qwen3.5-9b-mtp via llama.cpp on RTX 5060 Ti
```

The old local `eva-vllm-qwen3-vl-4b.service` is inactive. The replacement user
unit is:

```text
/home/sasha/.config/systemd/user/eva-llama-qwen3-vl-4b.service
```

It uses the official ready-made Unsloth files, not a local quantization:

```text
Qwen3-VL-4B-Instruct-UD-Q4_K_XL.gguf
  bytes:  2546342176
  sha256: a5563711a524defc06d67438a497ce4ba9a0fb126f28456a9f207e2ad104eb75
mmproj-F16.gguf
  bytes:  836180640
  sha256: 1b9f4e92f0fbda14d7d7b58baed86039b8a980fe503d9d6a9393f25c0028f1fc
```

Important working flags are 16384 context, one slot, Q8 KV, flash attention,
temperature 0.1, mmproj offload and `--fit-target 512`. The original 1024 MiB fit
margin produced a pathological hybrid placement: long generations ran at roughly
8-12 token/s while llama.cpp consumed 280-370% CPU and the GPU showed large idle
gaps. With 512 MiB margin, the same real 6k-8k-token, eight-image requests run at
about 48-53 token/s and 8-13 seconds total. Prompt evaluation is approximately
1.45k-1.75k token/s. Total RTX 4060 allocation, including the other local EVA and
desktop processes, stayed near 6.8/8.2 GiB during the observed soak. Short bursts
reach deferred=2 but drain back to zero; there is no accumulating queue.

This is a local rehearsal unit only. Georgia production uses established external
inference servers and must keep their endpoints, model IDs, contexts, batching and
GPU visibility unchanged.

## Continuation stabilization results

### Watchdog

The local watchdog/recovery units now depend on and restart llama.cpp, load the
rehearsal env read-only, and wait for its OpenAI endpoint after recovery. The
semantic canary gates on the randomized color order; OCR digits are diagnostic
only because Qwen3-VL-4B read the colors reliably but not all random digits.

`vlm_vision_health.py` now reads llama.cpp/vLLM queue metrics before enqueueing a
canary. A live endpoint with processing/deferred work returns `busy` in milliseconds
and does not consume another inference slot. Confirmed failures still require three
consecutive failures before recovery. A false-restart loop was observed while
developing this fix; the timer was stopped immediately, the counter/busy behavior
was corrected, and the timer was re-enabled only after targeted and live checks.
The current timer runs every 120 seconds; recent busy checks completed successfully
without invoking recovery.

### Live channels, alerts and probes

The two requested Luxriot channels are:

```text
112  Zenbook webcam
118  emu-1
```

After the fit correction, 26 distinct live VLM batches were inspected. All 26 had
non-empty `vector_signal_chars`, proving that probe/homeostasis signals were added
to the VLM request as a secondary signal. Twenty batches returned a complete parsed
contract; six returned a safe `partial_prefix`, with alerts intentionally placed
first so alert actions survived output truncation.

Descriptions were grounded enough for the rehearsal: channel 112 reported the
person/desk/monitor/cat scene and visible entries/gestures; channel 118 reported the
night intersection, drifting vehicles and smoke. The main remaining quality issue
is occasional overconfident wording such as `controlled drift`; intent/skill wording
must remain prohibited in prompts and semantic guard tests.

Alerts are persisted in EVA and bookmark delivery works. The observed Luxriot API
call itself took roughly 35-114 ms. Post-fix alert acknowledgement from the end of a
completed batch ranged from about 14 to 87 seconds depending on capture/seal and
short queue bursts; the pre-fix values were commonly 2-4 minutes. Deduplicated
alerts correctly omit a second bookmark. This is a major improvement, but latency
acceptance should be repeated in the final appliance topology rather than inferred
from replay timestamps on this overloaded desktop.

### L1-L3 consolidation

Durable rollup state is in PostgreSQL `archive.runtime_state`, so the absence of
local `luxriot_rollups_cache.json` and `luxriot_summary_state.json` files is expected
for this runtime and is not data loss. Both channels have current `ready` semantic
rollups with correct lineage:

```text
L1  source_level=None  latest window 11:00-11:15
L2  source_level=L1    latest window 10:00-11:00
L3  source_level=L2    latest window 03:00-11:00
```

Older L3 rows with `generation_status=review_queued` are proposal-only deep review
records and intentionally do not mutate live memory. Targeted tests confirmed
lineage, durable cache promotion, hot-cache eviction recovery, agent depth selection,
operator/API redaction and L3 proposal-only behavior.

### Settings and operator guides

Settings saves now use dirty-field PATCH semantics. They write the env file declared
by the service (`EVOSSEARCH_CONFIG_ENV_FILE`/systemd `EnvironmentFile`), preserve
unsubmitted profiles and unrelated `EVA_*` lines, report env/process precedence and
only reset heavy runtime components whose settings actually changed. The React
modal exposes the active VLM/agent profile IDs and does not silently flatten other
configured inference profiles.

The canonical operator guide is now English:

```text
react-ui/public/quick-start.html
react-ui/public/quick-start.ka.html  Georgian
react-ui/public/quick-start.lv.html  Latvian original
```

All three are linked to each other; the production React build copied all three.
The verified build was subsequently overlaid onto the live rehearsal
`/home/sasha/Projects/eva-georgia-upgrade-repro/react-ui/dist` without restarting
EVA. The previous dist is recoverable at
`react-ui/dist.pre-stabilization-20260811-1146`. Live HTTP verification on port 5081
returned English/Georgian/Latvian for the three guide URLs, and the served index and
hashed JavaScript matched the reviewed build. The rehearsal `.env` SHA-256 was
identical before and after this static-only deployment.

### Continuation verification

Passed during this continuation:

```text
tests/test_security_smoke.py                         19 passed
targeted Settings persistence/precedence tests       3 passed
tests/test_vlm_vision_health.py                       7 passed
targeted live-memory/L1-L3/agent contract tests       8 passed
installer/update/watchdog/gate unittest bundle        82 passed
tests/test_vlm_alert_contract.py                      36 passed
React production build                               passed (existing chunk warning only)
React Vitest suite                                   88 passed (13 files)
HTMLParser over English/Georgian/Latvian guides       passed
git diff --check                                     passed
```

The Georgia rehearsal EVA process and its `.env` were not restarted or rewritten
by this continuation. Only the local llama.cpp service was deliberately restarted
while changing its fit margin. No Georgia production service or env was accessed.

## Latest inference-preservation fix

### Universal Python installer

`scripts/install_eva_083.py` now fingerprints the complete existing EVA inference
policy without printing values or credentials. Covered keys:

```text
EVOSSEARCH_LM_*
EVOSSEARCH_AGENT_*
EVOSSEARCH_INFERENCE_*
CUDA_VISIBLE_DEVICES
```

This includes endpoints, models, API keys, profile selection, timeouts, inflight
limits, agent context, video budgets, queue configuration and GPU mapping.

For an existing deployment the installer:

1. computes the current fingerprint during planning;
2. computes the projected `.env` and blocks preflight if the policy would change;
3. verifies the source again before any apply mutation;
4. verifies after `.env` staging;
5. verifies after code installation;
6. verifies after post-update health checks;
7. invokes the existing rollback path if drift occurs after a backup exists.

Fresh installations may still create a reviewed default inference policy.

### Shell updater

`scripts/update_bundle.sh` has the same inference-policy fingerprint before and
after code installation. It no longer writes a temporary
`EVOSSEARCH_AGENT_CONTEXT_LIMIT_TOKENS` value when the served context is below the
recommended 65k. It now:

- reports the configured and served context read-only;
- asks whether the operator wants to continue with the existing configuration;
- preserves it exactly;
- tells the operator to reconfigure inference separately after acceptance if
  desired.

### Actual Georgia env dry-run result

The current rehearsal `.env` was checked without printing its contents:

```text
georgia_policy_before=53eaaf460255fd1c8548a12f74087f2a11a1e96d1348a0fdb8d8d6259be8fd6b
georgia_policy_after =53eaaf460255fd1c8548a12f74087f2a11a1e96d1348a0fdb8d8d6259be8fd6b
preserved=True
sensitive_updates=[]
missing_required_count=0
```

The hash is safe to log; the underlying values and credentials are not.

## Runtime stabilization included in `15ad09f`

The commit also contains related product fixes from the current stabilization
series:

- L0 stopped synchronously re-embedding archive frames with SigLIP after every VLM
  response; it reuses the independent one-Hz semantic embeddings instead.
- L0 latency/queue telemetry and bounded output budgets were added.
- L1 source watermarks, failed-window retry and recovery were added/fixed.
- VLM alert/probe/runtime contracts and relevant tests were expanded.
- The visual watchdog distinguishes a busy inference endpoint from a dead one:
  a canary timeout followed by a healthy read-only `/models` response becomes
  `busy`, not an automatic vLLM restart. Genuine visual mismatches still escalate.
- The health gate accepts `busy` only while endpoint liveness and a recent visual
  success are both present.

Important diagnosis: previous two-to-three-minute L0 gaps were amplified by the
one-minute visual watchdog restarting a healthy but busy vLLM. Restart then sometimes
failed because the RTX 4060 desktop load left too little free VRAM for KV allocation.
This was not evidence that ordinary VLM generation itself required three minutes.

These product watchdog changes are source changes. They do not modify the Georgia
inference servers during an upgrade.

## Verification already completed

Most recent exact command:

```bash
/home/sasha/Projects/evo-ssearch/.venv/bin/python -m unittest \
  tests.test_install_eva_083 \
  tests.test_update_bundle \
  tests.test_vlm_vision_health \
  tests.test_vlm_vision_gate
```

Result:

```text
Ran 82 tests in 2.649s
OK
```

Also passed:

```bash
bash -n scripts/update_bundle.sh
git diff --check
```

The recurring message `Failed to create stream fd: Operation not permitted` came
from the previous Codex sandbox/harness, not from the EVA test suite.

## Afternoon stabilization pass: agent synthesis, bounded archive scope and probe UI

The rehearsal instance was updated in place from the reviewed dirty source. Georgia
production was not touched and the rehearsal `.env` remained byte-for-byte unchanged:

```text
2c254527143f62bbdbcf7a14914872e2a6f1e0f4f776ef02024c0f27aac76325
```

### Agent behavior fixed in source and deployed to rehearsal

- Archive requests with exactly two operator-supplied ISO timestamps now normalize
  that interval before search. The LLM cannot silently replace it with `last24h`.
- A quoted archive query is kept separate from channel, interval and response-style
  clauses.
- The protected archive read timeout is 180 seconds. A timed-out cacheable read is
  not started a second time in the same turn while its worker may still be running.
- Compact archive evidence carries server-formatted `timestamp_utc`; model-generated
  timestamps that are not present in trusted tool results force deterministic recovery.
- Natural synthesis is the default: report-shaped headings, tables, long bullet dumps
  and generic follow-up menus force the compact fallback unless the operator explicitly
  requested structured output.
- The fallback leads with the finding, caps detailed visual examples and states bounded
  coverage once. It no longer conflates all scanned candidates with the much smaller
  ranked/vision-reviewed set.
- Markdown rendering no longer turns every blank source line into an extra vertical
  block. Horizontal rules and paragraph/list spacing are compact.

The exact live archive smoke used channel 112, query `sphynx cat`, and the interval
`2026-08-09T21:46:30Z` through `2026-08-10T21:47:55Z`. The tool call preserved all
four values exactly, ran only once, and returned after about 90 seconds. The subsequent
vision drill reviewed eight frames. The final quality gate was added after that smoke;
its deterministic behavior is covered by tests, but a second expensive end-to-end
archive run has not yet been repeated.

One important open quality check remains: the current llama.cpp Q4 vision run labelled
all eight frames as matches, whereas an earlier run classified the same batch as
2 match / 5 no-match / 1 uncertain. Inspect those eight thumbnails manually before
declaring the Q4 visual judge accepted.

### Probe editor layout fixed and deployed

The positive/negative pair grid now uses shrinking `minmax(0, 1fr)` columns, reserves
a fixed 32 px action column, and gives each input `width: 100%; min-width: 0`. The
scroll owner has a 12 px right inset plus `scrollbar-gutter: stable`, so the negative
look-alike input and delete action stay inside the accent card and away from the
scrollbar.

Deployed React assets:

```text
index-B244Ig2m.js
index-DIOk0MDO.css
```

Source and rehearsal hashes for `agent.py` match. Source and rehearsal hashes for
`index-DIOk0MDO.css` also match. Recoverable backups are:

```text
/home/sasha/Projects/eva-georgia-upgrade-repro/agent.py.pre-agent-quality-20260811-1352
/home/sasha/Projects/eva-georgia-upgrade-repro/react-ui/dist.pre-probe-layout-20260811-1352
```

### Verification and runtime observations

```text
tests/test_agent_tool_loop.py:             57 passed, 13 subtests passed
tests/test_agent_video_summary_tools.py:   88 passed, 3 subtests passed
tests/test_eva_agent_adapter.py:           38 passed
React agent Markdown/transcript:           6 passed
React production build:                   passed (6240 modules)
git diff --check:                          passed
```

The Gunicorn master was gracefully reloaded with `HUP` because this shell has no
interactive PolicyKit authorization for the system unit. The old worker exited and
exactly one new worker remains. Post-reload `/health` is HTTP 200 and `/ready` reports
`ready`; database, PostgreSQL/schema, authentication, deployment security, embedder,
LM profiles, Luxriot and Luxriot restore are green.

Both llama.cpp user services are active with `NRestarts=0`. For the one-slot VLM and
two live channels, six consecutive queue samples showed `processing=1` and `deferred`
alternating between 0 and 1; the queue did not grow. The RTX 4060 sample while both
streams were active was about 6.97/8.19 GiB and 82% GPU utilization.

Cold/reload readiness is still too slow: SigLIP initialization can make `/health`
temporarily unreachable, and even after health recovers `/ready` took about 15 seconds.
Treat startup probe isolation and reload behavior as the next reliability defect,
not as an inference-model sizing failure.

## Late stabilization pass: probe runtime, Operator Mode, settings PATCH and typography

Four operator-visible regressions were reproduced and corrected in source on the
same day.  The React build is deployed to the rehearsal.  Backend files and a
CUDA-capable rehearsal venv are staged, but the Gunicorn system unit still needs
one privileged restart before those backend/runtime changes become active.

### Probe degradation is a CPU-environment defect, not a VRAM sizing defect

Detailed readiness telemetry showed both channels failing probe work with
`embedding batch request timed out`.  The shared CLIP microbatcher had thousands
of timeouts, seconds of queue wait, and tens of seconds of CPU compute; realtime
probe evaluation had stopped advancing.  PostgreSQL and the probe store remained
reachable.

The service process was launched through:

```text
/home/sasha/Projects/evo-ssearch-tbilisi-field/.venv
torch 2.13.0+cpu
CUDA unavailable
```

The CUDA-capable environment at `/home/sasha/Projects/evo-ssearch/.venv` imports
the complete runtime stack, uses `torch 2.12.1+cu130`, and sees both GPUs.  An
isolated SigLIP2 batch of eight fitted on the RTX 5060 Ti with about 731 MiB of
weights and 810 MiB reserved; first compute was about five seconds.  The existing
`CUDA_VISIBLE_DEVICES=1` maps to that RTX 5060 Ti on this host, so no `.env`
change is needed and the already-full RTX 4060 remains dedicated to the VLM.

The rehearsal `.venv` symlink is now staged to the CUDA-capable environment.  Its
recoverable previous symlink is:

```text
/home/sasha/Projects/eva-georgia-upgrade-repro/.venv.pre-probe-cuda-20260811-164716
```

The running Gunicorn master still has the old CPU interpreter because a true
system-unit restart requires interactive authorization.  A normal `systemctl
restart` timed out at PolicyKit and `sudo -n` was rejected; neither attempt changed
the service PID or restart counter.  Complete activation with:

```bash
sudo systemctl restart eva-ai-georgia-repro.service
```

Do not use `HUP` for this transition: it reloads code but retains the master's
old interpreter/environment.  After restart, verify the embedder reports CUDA,
both channel probe timeout counters stop increasing, dispatch backlog drains,
and realtime probe evaluation advances.

### Operator Mode Live/L1 routing

Database history confirmed the failure.  For `Show the latest VLM alerts across
all channels`, the harness inherited default channel 112 and emitted
`get_video_summaries(channel_id=112, depth=L1)`.  The trusted UI effect then changed
depth while leaving the period selector on Live.

The runner now treats explicit `all channels` as a hard scope that cannot inherit
a default/previous channel.  It inventories channels first, then drills latest
alerts at `depth=live`.  Passive unscoped video reads no longer overwrite the
operator's current depth; explicit historical scope projects depth together with
the server-resolved interval.  This remains within the pinned tuktuk sequence
`W -> MAP/DRILL -> TERM`; no tool schema, approval gate, or argument-extraction
rule changed.

### Settings host/port PATCH

Changing only host could transiently serialize the numeric port control as an
empty string even while the UI still displayed `5081`.  The UI now emits only
dirty writable fields and omits blank secrets/blank transient port values.  The
backend independently treats blank port as omitted before computing submitted
fields, preserving the exact persisted port and unrelated `.env` keys.

### Agent message typography

Consecutive plain source lines are now one paragraph with controlled `<br>` line
breaks.  Blank source lines delimit paragraphs without becoming blank DOM spacer
elements.  Paragraphs/lists/headings have modest block spacing and the message
line height is 1.42, avoiding both the earlier double spacing and the compressed
follow-up.

### Deployment and checks

The rehearsal now serves:

```text
index-BOHmfVMa.js
index-DENWuc8k.css
```

The staged backend files exactly match source: `agent.py`, `agent_ui_effects.py`
and `oldapp.py`.  Recoverable copies and the previous React build are:

```text
/home/sasha/Projects/eva-georgia-upgrade-repro/agent.py.pre-runtime-stabilization-20260811-164716
/home/sasha/Projects/eva-georgia-upgrade-repro/agent_ui_effects.py.pre-runtime-stabilization-20260811-164716
/home/sasha/Projects/eva-georgia-upgrade-repro/oldapp.py.pre-runtime-stabilization-20260811-164716
/home/sasha/Projects/eva-georgia-upgrade-repro/react-ui/dist.pre-runtime-stabilization-20260811-164716
```

Checks passed:

```text
Agent/tool/UI-effect suites: 190 passed, 16 subtests passed
Flask dataflow/auth suites:   107 passed, 127 subtests passed
React complete suite:         94 passed
React production build:       passed (6241 modules)
Python compile/diff check:     passed
```

The rehearsal `.env` remained byte-for-byte unchanged:

```text
2c254527143f62bbdbcf7a14914872e2a6f1e0f4f776ef02024c0f27aac76325
```

Both llama.cpp units remained `active/running`, `NRestarts=0`, and both health
endpoints returned `ok` after staging.

## Final runtime pass: CLIP backpressure and stream timeouts

This section supersedes the earlier note that the CUDA/backend transition was only
staged. The running rehearsal worker now uses the CUDA-capable environment, and the
final capture changes are active in `eva-ai-georgia-repro.service`.

### Root cause

The emu1 source was not spending material time obtaining its Luxriot token or
opening the media response: an isolated authenticated open took about 0.042 s. The
failure was local backpressure after the response opened:

- a 60-second live window requested 180 dense JPEG candidates at 3 fps;
- PIL/CV/frame admission plus downstream signal work slowed consumption of the
  ffmpeg stdout pipe;
- the authenticated feeder could then block on ffmpeg stdin and could not check
  its own 60-second deadline;
- the outer process loop killed the still-progressing window at the fixed 67-second
  wall-clock deadline;
- auto mode used cumulative `slow_snapshot_count > 0`, so one historical snapshot
  over the four-second threshold permanently switched that channel to live mode.

The same RTX 5060 Ti is shared by SigLIP2 and the Qwen3.5 agent llama.cpp server.
Agent generations produced legitimate SigLIP tails up to roughly 23-29 seconds.
The previous async dispatcher admitted a stale FIFO faster than that GPU could
drain it, then its 15-second caller timeout misreported healthy-but-late requests as
probe failures. This was contention and queue policy, not Q4 model VRAM sizing.

### Deployed corrections

- probe CLIP work is latest-only per channel: at most one active and one newest
  pending observation; superseded frames are marked unavailable, not failed;
- queue saturation never falls back to synchronous CLIP on the capture thread;
- the default CLIP batch request timeout is 45 seconds, above the measured shared-
  GPU tail, while queue depth remains bounded;
- live dense sampling defaults to 2 fps instead of 3 fps, preserving two apex
  candidates per represented second while removing one third of JPEG/PIL work;
- a minute live window has a 127-second hard bound but is terminated after 15
  seconds with no decoded progress, so a slow active pipe is distinct from a stall;
- auto mode enters live only after two consecutive slow/failed snapshots and retries
  the snapshot endpoint after every successful live window.

Source and rehearsal hashes match for `luxriot_connector.py` and `config.py`.
Recoverable pre-final-pass copies are:

```text
/home/sasha/Projects/eva-georgia-upgrade-repro/luxriot_connector.py.pre-20260811-194900-stream-progress-timeout
/home/sasha/Projects/eva-georgia-upgrade-repro/config.py.pre-20260811-194900-stream-progress-timeout
```

The Gunicorn master was HUP-reloaded. As in the prior reload, the retired worker
did not obey the 20-second graceful timeout or SIGTERM. Only its verified exact PID
`2204996` was sent SIGKILL; master `2014970` and new worker `2232488` remained. The
unit is active with `NRestarts=0` and exactly one worker. This shutdown defect still
needs a separate fix.

### Live acceptance evidence

The full Luxriot runtime file passed `203 passed in 82.94s`; the focused live,
latest-only dispatch and auto-hysteresis selection passed `7 passed`.

After reload, channel 118 naturally crossed the consecutive-slow threshold twice.
Both fallback windows completed without a stream failure and returned to snapshot:

```text
window 1: 120 frames, 60.0 represented seconds, 62.370 s wall time
window 2: 113 frames, 56.5 represented seconds, 63.206 s wall time
live_segment_count=2
live_segment_failed_count=0
last_live_segment_error=null
active_capture_source=snapshot
snapshot_slow_streak=0
```

Channel 112 remained on snapshot with no capture error. Both channels reported
`probe_last_error=null` and `summary_last_error=null`. The CLIP batch queue stayed
at 0-1; latest-only async state stayed bounded to the two channels. There were no
post-reload live-segment or embedding request timeout messages in the service log.

Both llama.cpp units remained active with `NRestarts=0`. The rehearsal `.env` hash
remained exactly:

```text
2c254527143f62bbdbcf7a14914872e2a6f1e0f4f776ef02024c0f27aac76325
```

### Resolved UI polling latency

The 1.8-19.3-second `/luxriot/streams` tail was not caused by stream status
serialization, the manager lock, network, swap, or PostgreSQL loss. Added runtime
timing showed:

```text
LuxriotManager lock wait:       ~0.003 ms
per-channel status:             ~0.1-1.5 ms
complete streams_status():      ~2-15 ms normally
```

A 12-request concurrent reproduction with one operator cookie made every request
take 25-27 seconds. `pg_stat_activity` showed `Lock/transactionid` waits on the same
`iam.sessions` row. Every authenticated GET used `SELECT ... FOR UPDATE`, updated
`last_seen_at`, and held one of the four Gunicorn request threads. Remaining requests
then waited for a worker. Concurrent status calls also repeatedly loaded the desired
live-session document through the runtime-state store lock.

The deployed correction keeps authorization fail-closed and database-backed on
every request, including expiry, revocation, user-active and user-lock checks, but:

- session resolution is now an MVCC read without `FOR UPDATE`;
- `last_seen_at` is touched at most once per 30 seconds instead of on every GET;
- desired live-session state is loaded once into a copy-on-read manager cache;
- start/stop mutations still persist synchronously and update the cache only after
  the durable write succeeds;
- status stage timings remain exposed as `status_timing_ms` for diagnosis.

The exact same 12-request acceptance test after deployment completed in 59-292 ms;
manager time was 1.7-2.9 ms, desired-state lookup 0.02-0.04 ms, and PostgreSQL showed
no `Lock/transactionid` waits. A normal authenticated poll then took 72.6 ms. Five
`/health` samples were 27-67 ms. `/ready` remains a deliberately heavier dependency
probe (about 1.3-4.2 s under inference load) and should be optimized separately if
the UI calls it as routine polling.

Full auth and Luxriot tests passed:

```text
276 passed, 3 skipped, 2 warnings in 82.74 s
```

Current recoverable copies from this pass:

```text
/home/sasha/Projects/eva-georgia-upgrade-repro/luxriot_connector.py.pre-20260811-200500-status-timing
/home/sasha/Projects/eva-georgia-upgrade-repro/luxriot_connector.py.pre-20260811-201500-auth-poll-lock
/home/sasha/Projects/eva-georgia-upgrade-repro/security/postgres_identity.py.pre-20260811-201500-auth-poll-lock
```

That pass used worker `2282732` under master `2014970`. It has since been
superseded by the rollup/agent pass below.

## Final rollup and agent-health pass

### Historical summary timeout root causes

A 24-hour summary read was slow even at L0: channel 112 took about 83 seconds,
while L3 commonly took 80-88 seconds. Direct PostgreSQL range selection was only
about 0.38 seconds, so the inference endpoints and the amount of stored data were
not the primary bottleneck. Three application defects compounded:

- read-only `get_video_summaries` calls refreshed channel memory from historical
  rollups, producing writes during an operator read; memory refresh now occurs only
  for an explicit synthesis pass;
- `PostgresRuntimeStateStore.list_rollups` waited behind the process-wide writer
  lock even though PostgreSQL MVCC makes the range read safe without it;
- the hot-cache merge normalized all channels and durable duplicates before
  applying channel/time filters. It now filters cheaply, deduplicates identities,
  and only then normalizes relevant rows.

Live acceptance after deployment:

```text
channel 112: L0 9.549 s, L3 17.892 s
channel 118: L0 3.964 s, L3 11.432 s
both L3 reads together: about 30 s, 4/4 windows per channel, no timeout
```

Read-only calls no longer update historical memory. Tests cover that invariant,
concurrent DB range reads bypassing the writer lock, and hot/durable duplicate plus
off-channel filtering. The full authentication and Luxriot selection completed:

```text
276 passed, 3 skipped, 2 warnings in 97.20 s
```

Recoverable target copies from this work are:

```text
/home/sasha/Projects/eva-georgia-upgrade-repro/luxriot_connector.py.pre-20260811-204200-rollup-readonly
/home/sasha/Projects/eva-georgia-upgrade-repro/archive_store.py.pre-20260811-205300-rollup-read-concurrency
/home/sasha/Projects/eva-georgia-upgrade-repro/luxriot_connector.py.pre-20260811-210200-rollup-hot-filter
```

### Cross-channel agent path and final presentation

The exact operator request `Compare the video summaries from all available
channels ... at L3 depth` originally exposed only time normalization. The word
`Compare` activated the bundled `cross_channel_correlation` playbook, but that
playbook did not expose `list_video_summary_channels`, so the progressive tool
surface could not inventory an unnamed authorized scope. The playbook now includes
the inventory tool and explicitly inventories `all/available` scope before bounded
per-channel reads.

The server-owned path also now:

- recognizes English and Russian all/available/authorized channel wording;
- extracts an explicit `live`/`L1`/`L2`/`L3` depth from English or Russian operator
  text as a closed enum;
- carries that exact depth through inventory and detail reads;
- terminates an explicit cross-channel summary plan after every candidate channel
  has been read, preventing the small local model from repeating completed reads;
- renders completion recovery as compact prose in the operator's language, with
  one block per channel and truthful semantic pending/failure state.

The final authenticated Operator Mode acceptance used this exact Russian request:

```text
Сравни видео-сводки по всем доступным каналам за последние 24 часа на глубине L3.
Проверь свежесть данных и здоровье консолидации L0–L3. Только чтение: ничего не меняй.
```

It completed in 45.859 seconds with exactly:

```text
normalize_time_window
list_video_summary_channels(depth=L3)
get_video_summaries(channel_id=118, depth=L3) -> 4/4, semantic_status=partial, pending=4
get_video_summaries(channel_id=112, depth=L3) -> 4/4, semantic_status=partial, pending=3
```

There were no tool errors, repeated reads or timeouts; the bounded-plan completion
event and final `done` event were both present. The local model's final synthesis
was rejected by the quality gate, and deterministic recovery produced a compact
Russian report containing both channels exactly once. `partial` is a real data
state for open/pending semantic windows, not a failed L3 read.

Current agent-focused verification is:

```text
206 passed, 16 subtests passed in 14.79 s
```

The changes remain compatible with the pinned tuktuk grammar: cross-channel
inventory is `W`, per-channel summary reads are bounded `DRILL`, the explicit depth
comes from operator text through a closed enum, and the compact recovery is `TERM`.
No grammar-pin conflict or review question was introduced.

Relevant recoverable target copies are:

```text
/home/sasha/Projects/eva-georgia-upgrade-repro/agent.py.pre-20260811-211500-cross-channel-inventory
/home/sasha/Projects/eva-georgia-upgrade-repro/skills/cross_channel_correlation/SKILL.md.pre-20260811-211500-cross-channel-inventory
/home/sasha/Projects/eva-georgia-upgrade-repro/agent.py.pre-20260811-212300-human-video-fallback
```

The current worker is `2373043` under master `2014970`. The previous worker
`2361760` retired without intervention after HUP; the unit remains active with
`NRestarts=0`. The rehearsal `.env` hash is still:

```text
2c254527143f62bbdbcf7a14914872e2a6f1e0f4f776ef02024c0f27aac76325
```

Final runtime spot-check after the agent acceptance:

```text
EVA /health (3 samples): 37-51 ms, HTTP 200
VLM 1234 /health:        1.6 ms, HTTP 200, PID 1499650, NRestarts=0
agent 1235 /health:      5.3 ms, HTTP 200, PID 1142087, NRestarts=0
RAM available:          about 10 GiB
swap allocated:         1.7 GiB; current vmstat intervals si/so=0/0
stream-status route:    5.153 ms manager total
```

Channels 112 and 118 were both desired/running with no capture, probe, summary or
live-segment error. CLIP queue depth was 1 with one request in flight and no error;
the semantic snapshot queue was empty. A fresh channel-118 fallback segment then
completed with 120 frames representing 60.0 seconds in 60.399 seconds. Its success
count advanced from 1 to 2, `live_segment_failed_count` stayed zero, and capture
returned to snapshot. This is fresh evidence that the former fixed 67-second kill
no longer aborts a progressing one-minute stream window.

### Post-commit bounded-rollup and Operator Mode acceptance

The remaining historical rollup cost came from rebuilding already durable target
levels and from materializing all L0 temporal structures even when only a small
number of missing higher-level buckets needed synthesis. Commit `645f4f5` makes
durable read-only target buckets authoritative, reuses lower durable levels for
L2/L3 reads, and materializes L0 structures only for missing L1 buckets. It retains
the complete L0 source count for observability.

Authenticated live 24-hour acceptance after deployment was:

```text
channel 112: L1 0.992 s, L2 1.019 s, L3 3.273 s
channel 118: L1 2.967 s, L2 8.440 s, L3 7.692 s
```

Each channel returned exactly four L3 windows: three closed `llm/ready` windows
and only the current open window as `pending_context/pending`. The false
`refresh_pending` state is gone. A few historical closed L1/L2 gaps are still
`deterministic/queued` and should be allowed to converge through the real scheduler
rather than repaired by an ad-hoc database mutation.

The same commit fixes the agent's attention-burst route and fail-closed behavior
when a llama.cpp server ignores required tool choice. `live` is translated to the
canonical closed depth `L0`; the Russian word `активность` no longer collides with
the runtime-status intent; and completed attention reads terminate without falling
through to unrelated L1 summaries. The output explains that attention bursts are a
statistical visual-change marker rather than evidence of an event.

Final authenticated Operator Mode acceptance used:

```text
На канале emu-1 (118) покажи самые сильные всплески визуальной активности за
последний час. Это статистический сигнал, не делай выводов о событиях. Только чтение.
```

It completed in 11.355 seconds with exactly two successful calls:

```text
normalize_time_window(relative_range="последний час")
list_attention_bursts(channel_id=118, target_level=L0) -> 0 bursts / 99 L0 windows
```

There were no summary calls, duplicate guards, tool errors or timeouts. The final
Russian response truthfully said that the server marker found no sharp deviation
and explicitly said this does not mean no events occurred. A separate cross-channel
L3 acceptance completed in 46.361 seconds with one inventory and one bounded read
per channel, no errors or repeats, and a compact Russian report covering both
channels exactly once.

Final relevant verification:

```text
tests/test_agent_video_summary_tools.py + tests/test_agent_tool_loop.py +
tests/test_luxriot_inference_runtime.py: 360 passed, 16 subtests passed
final agent rerun:                 150 passed, 16 subtests passed
git show --check 645f4f5:         clean
added-line credential scan:       no matches
```

These changes still fit the pinned tuktuk grammar: the attention lookup and
per-channel rollups are bounded `DRILL` reads; the channel/time arguments come from
operator text and server normalization; `L0` is a canonical closed enum; and no
tool schema, intent group or result envelope changed. No grammar review question
was required.

After the final HUP, worker `2487723` is active under master `2014970` with strict
readiness HTTP 200. Both user-scoped llama.cpp units and the watchdog timer are
active with `NRestarts=0`; ports 1234 and 1235 return HTTP 200. About 11 GiB RAM is
available. EVA itself has approximately 5 MiB swapped, so the host's allocated
2.2 GiB swap is not evidence of active EVA swap pressure. The rehearsal `.env`
hash remains unchanged.

### Rollup persistence/UI starvation follow-up

The first real scheduled L1 pass after worker `2487723` started exposed a second
latency component. Channel 112 needed 127.692 seconds for one L1 window and channel
118 needed 134.036 seconds for two. llama.cpp logs showed only 12-24 seconds of
model work; EVA then rewrote the former monolithic 259-entry rollup-cache JSON after
every independently durable rollup while holding the manager cache lock. Status
requests accumulated behind that lock and, together with bounded preview streams,
could exhaust all four Gunicorn request threads. Even `/health` briefly timed out.

Commit `2adc0cb` stops rewriting the legacy monolith when the runtime store supports
per-rollup durability. The legacy payload is now an insert-only migration source,
so it cannot overwrite a newer independently durable row during restart; durable
rows are overlaid onto the migration snapshot before the hot cache becomes visible.
If an individual durable save fails, EVA retains the monolithic fallback path.

Verification before deployment:

```text
tests/test_luxriot_inference_runtime.py: 212 passed in 59.43 s
```

The target copies are:

```text
/home/sasha/Projects/eva-georgia-upgrade-repro/archive_store.py.pre-20260811-232900-rollup-cache-split
/home/sasha/Projects/eva-georgia-upgrade-repro/luxriot_connector.py.pre-20260811-232900-rollup-cache-split
```

The HUP started worker `2522849`. Cold SigLIP initialization kept port 5081
unavailable for about 159 seconds, confirming that graceful reload startup remains
a separate pilot reliability defect. Both llama.cpp PIDs stayed unchanged and both
units retained `NRestarts=0`; the `.env` hash stayed unchanged. On startup the new
worker loaded 263 cache rows with `durable_rollups_promoted=0`, proving the old
monolith did not overwrite current durable rows.

A bounded authenticated operator synthesis then repaired two missing channel-112
L1 windows. Both became `llm/ready`; the request completed in 64.575 seconds. The
two llama.cpp generations themselves took about 14 seconds combined. During the
entire operation, 103/103 concurrent `/health` samples succeeded with no timeout,
maximum latency 1.539 seconds and p95 0.792 seconds. Persistence remains sequential
and visible in total synthesis latency, but it no longer freezes liveness or the UI.

### Probe-list payload, alert/probe path and grounded agent acceptance

The persistent probe store was healthy, but `/probes/list` returned all 30 stored
base64 thumbnails for every probe. With only two probes the authenticated response
was 560,189 bytes, of which about 502 KiB was duplicate recent-hit imagery. The
monitoring board needs one current card image and the P/N/M series, not 30 images.
Commit `8969893` keeps the latest card thumbnail and all numeric history while
omitting thumbnails and embedding arrays from `recent_hits` in the collection
response. Stored evidence is unchanged.

Live acceptance after deployment:

```text
/probes/list: 55,300 bytes (-90.1%), 84 ms, HTTP 200
CH 118 probe: runtime=running, semantic=ready, no capture/semantic error, 164 ms
CH 112 probe: runtime=running, semantic=ready, no capture/semantic error, 609 ms
recent_hits: 30 score rows per probe, zero repeated thumbnails
```

The fast VLM alert lane was also observed before the final reload. It accepted and
completed 18 motion/semantic-change episodes with no queue rejection or runtime
error; all produced `alerts: []`, so zero bookmarks was a grounded no-alert result,
not a dead lane. Measured Luxriot bookmark calls were about 63-165 ms. The dominant
event-to-ack latency remained capture/batch closure, queue wait and VLM inference:
roughly 55 seconds median and 106 seconds p95 over the sampled 24-hour alert rows.

Both current operator probes have `bookmark=false`. The separate realtime bookmark
lane therefore has no eligible probe to deliver. Under shared-GPU contention it
also observed embedding ages above its five-second freshness gate; a bookmark-enabled
probe still needs a deliberate live acceptance before this lane is considered pilot-
ready. Do not solve that by silently changing `.env` or weakening the freshness gate.

The first live Operator Mode probe audit used one successful `list_probes` call but
the local model added an unsupported claim about active VLM errors and took 62.304
seconds. The agent now treats an explicit probe inventory/status request as one
bounded `MAP -> TERM` read, exposes only `list_probes`, and renders a compact trusted
receipt. Explicit negation such as `без изменений`, `ничего не меняй`, or `только
чтение` keeps the route read-only even though the phrase contains an edit-related
word stem.

The exact post-fix request was:

```text
Проверь без изменений, какие семантические пробы настроены на каналах 112 и 118,
включены ли они, и были ли у них срабатывания за последние 24 часа. Используй пробы
только как дополнительный статистический сигнал; не делай по ним выводов о событиях.
Только чтение.
```

It completed in 5.898 seconds with exactly one `list_probes(since_hours=24)` call,
no errors, retries or recovery, and a compact Russian answer covering both probes.
It explicitly says persisted semantic hits are secondary statistics, not event proof
or evidence of current VLM/stream health. The progressive-disclosure change remains
inside the pinned tuktuk `MAP -> TERM` path: no tool schema, result envelope or legal
argument source changed, and source honesty is strengthened.

Verification for this commit:

```text
agent tool-loop + video-summary suites: 151 passed, 16 subtests passed
probe list/lineage focus:                3 passed, 52 deselected
full API dataflow suite:                 55 passed, 127 subtests passed
git diff --check:                        passed
```

Two HUPs were required because the first live acceptance exposed the read-only
negation edge case. They made port 5081 unavailable for about 139 and 132 seconds,
respectively. Both llama.cpp PIDs stayed unchanged (`1499650` VLM and `1142087`
agent), and the rehearsal `.env` hash remained unchanged. The current worker after
the second HUP is `2567031` under master `2014970`. Recoverable copies are:

```text
/home/sasha/Projects/eva-georgia-upgrade-repro/oldapp.py.pre-20260812-probe-list-compaction
/home/sasha/Projects/eva-georgia-upgrade-repro/agent.py.pre-20260812-probe-agent-grounding
/home/sasha/Projects/eva-georgia-upgrade-repro/agent.py.pre-20260812-probe-agent-readonly-negation
```

### Bookmark-enabled probe, VLM latency, and Settings provenance acceptance

Sasha completed the remaining manual bookmark-enabled semantic-probe acceptance
with a thumbs-up criterion. A bookmark became visible in the Evo monitor about five
seconds after the gesture. The SigLIP2 positive/negative signal separated a thumbs-up
from a victory gesture by a meaningful margin. The probe capture, scoring, bookmark,
and operator-visible delivery path are therefore accepted for this rehearsal.

Fresh persisted VLM traces were measured independently. For 30 `sent` VLM alerts in
a bounded 30-minute sample, event-frame to Evo bookmark acknowledgement was median
39.41 s, p95 89.37 s, and maximum 146.20 s. Evo `createBookmark` delivery itself was
fast: median 55 ms. A per-stage sample showed that the latency is accumulated before
delivery: event-to-batch-end median 16.30 s, queue/preparation median 7.55 s, VLM
inference median 21.77 s, and post-inference-to-ack median 0.11 s. Stage medians are
not additive percentiles. Channel 112 was materially faster than channel 118 in the
same sample (total median 28.53 s versus 47.42 s). The latest one-hour archive had
37 sent and 29 deduplicated alerts with no failed deliveries. Eleven failed rows in
the six-hour view were older transient connection failures to Luxriot and must not be
mixed into the current successful latency distribution.

Commit `4f5842a` removes the Settings source ambiguity. The root defect was that
`config.py` froze pre-dotenv provenance at module level while the API looked for it
on the `Config` instance. The service had always declared the correct source:

```text
EnvironmentFile=/home/sasha/Projects/eva-georgia-upgrade-repro/.env
EVOSSEARCH_CONFIG_ENV_FILE=/home/sasha/Projects/eva-georgia-upgrade-repro/.env
```

The fixed API now reports that absolute file as the persistence source, the started
process as the effective runtime source, and startup/file differences explicitly.
Secure deployments fail closed with HTTP 409 instead of writing an ambiguous `.env`
when no persistence source is declared. The Environment tab now round-trips the file
it edits rather than overlaying startup values. Server `host`, `port`, and `debug`
are persistence-only/restart-required and are no longer mutated in the Python config
as if Gunicorn had rebound live. Other Settings writes retain surgical PATCH
semantics and report runtime-applied versus restart-required fields.

Verification and live deployment acceptance:

```text
focused backend Settings/provenance: 6 passed
security + Settings focus:           4 passed
full security/UI contract suites:    82 passed
React Settings tests:                2 passed
React production build:              passed
live /settings: declared_aligned, write_allowed=true, different_count=0
live /settings/env: declared source, file host=127.0.0.1, port=5081
strict /ready:                        HTTP 200
```

The controlled HUP took about 160 seconds because of the known cold SigLIP startup.
The active worker is `3201333` under master `2014970`. Both llama.cpp processes kept
their PIDs (`1499650` VLM and `2916440` agent), and the rehearsal `.env` hash remains
unchanged. Recoverable copies are:

```text
/home/sasha/Projects/eva-georgia-upgrade-repro/config.py.pre-20260812-settings-provenance
/home/sasha/Projects/eva-georgia-upgrade-repro/oldapp.py.pre-20260812-settings-provenance
/home/sasha/Projects/eva-georgia-upgrade-repro/react-ui/dist.pre-20260812-settings-provenance
```

### Immediate next stabilization work

A final read-only 24-hour `run=all` durable check showed that ordinary current
boundaries are represented correctly, while old gaps are not automatically swept:

```text
CH 112 L1: 68 ready, 4 queued, 1 current pending
CH 112 L2: 15 ready, 4 queued, 1 current pending
CH 112 L3:  3 ready, 0 queued, 1 current pending
CH 118 L1: 65 ready, 3 queued, 1 current pending
CH 118 L2: 15 ready, 3 queued, 1 current pending
CH 118 L3:  3 ready, 0 queued, 1 current pending
```

The pending rows are exactly the open windows: L1 `21:00-21:15 UTC`, L2
`21:00-22:00 UTC`, and L3 `16:00-00:00 UTC`. The remaining queued rows are older
closed windows, including several around the disruptive HUPs. The normal boundary
scheduler does not perform an arbitrary historical sweep, so these now require the
existing explicit restore/backfill preview and trusted UI Apply if the operator wants
them repaired. Do not silently synthesize them from this handoff.

The bookmark-enabled semantic-probe path, ordinary probe board, agent inventory,
VLM secondary-signal path and fast-alert execution are accepted. The measured VLM
delivery latency is dominated by batching/queueing/inference, not Evo delivery; use
the stored stage trace when deciding whether to tune cadence or concurrency. Keep the
slow Gunicorn HUP/SigLIP startup behavior as a separate reliability defect; do not
mask it by weakening health checks or changing the preserved inference policy.

## Live semantic signal correlation and probe-daemon cleanup

The probe editor now treats one scored image and its P/N/M values as an atomic
operator evidence unit. `/probes/status` reports semantic age/staleness and an
exact timestamped frame URL; `/probes/signal_frame/<channel>/<timestamp>` serves
only the JPEG stored with that embedding and never substitutes a newer preview.
React commits the image and values together, retains the last complete pair on a
transient failure, labels the scored timestamp/age and uses single-flight polling.
The hidden probe board no longer competes with the open modal, and the unused
one-second full-board rerender was removed.

The embedding benchmark now separates encoder work from shared-lock wait, warms
up before measuring, repeats within a five-second diagnostic budget and exposes
the actual CUDA device name. On this host the EVA/SigLIP process is on the RTX
5060 Ti while the VLM llama.cpp process is on the RTX 4060; watching the 4060
during an embedding benchmark was therefore misleading. The benchmark still
shares the production encoder and is intentionally bounded.

Live lock/work telemetry then exposed a legacy functional ghost. Every five
seconds the old probe daemon queried top hits and re-embedded their thumbnails
even though the exact vectors were already in `ProbeBuffer`. Before the fix,
39 image calls accompanied only 27 microbatch calls, average image lock wait was
about 1.15 seconds and maximum wait was 9.6 seconds. The daemon now reuses the
exact full-frame or ROI vector and its embedding-space identity for bookmark
dedupe and archive persistence. After deployment, image calls exactly equalled
microbatch batches and the latest warmed lock waits fell to near zero.

The remaining variance is inside SigLIP image work rather than lock contention:
warmed calls ranged from roughly 0.24 seconds to occasional 4-10 second outliers
while the RTX 5060 Ti monitor remained almost idle and the EVA process showed
CPU/OpenMP activity. This points to CPU preprocessing/thread scheduling as the
next bottleneck. Do not change inference server flags or the rehearsal `.env` to
mask it. First split processor/model/materialization stages in the existing
telemetry or profile them in the one live worker; a second parallel Transformers
runtime is too expensive while swap is full.

Deployment details:

```text
source commits: 47c70d5, c6417c7
active worker after HUP: one worker under eva-ai-georgia-repro.service
service NRestarts: 0
.env sha256: 2c254527143f62bbdbcf7a14914872e2a6f1e0f4f776ef02024c0f27aac76325
llama.cpp PIDs: unchanged across both HUP reloads
```

Recoverable overlays are in:

```text
/home/sasha/Projects/eva-georgia-upgrade-repro/oldapp.py.pre-20260812-live-signal-correlation
/home/sasha/Projects/eva-georgia-upgrade-repro/probe_manager.py.pre-20260812-live-signal-correlation
/home/sasha/Projects/eva-georgia-upgrade-repro/react-ui/dist.pre-20260812-live-signal-correlation
/home/sasha/Projects/eva-georgia-upgrade-repro/oldapp.py.pre-20260812-probe-vector-reuse
/home/sasha/Projects/eva-georgia-upgrade-repro/probe_manager.py.pre-20260812-probe-vector-reuse
```

Both HUPs reproduced the cold/reload reliability defect: old and new workers
overlapped, `/health` stopped responding and readiness recovered only after about
two to three minutes. This is not acceptable as the final appliance deployment
strategy even though the process recovered to one worker and `/ready=200`.

Verification completed for this pass: Python compilation, 14/14 focused
ProbeManager tests, React production build, and all 94 React tests (run in bounded
groups because the full process was terminated under the already-full swap
pressure). The heavyweight API smoke could not import a second `oldapp` safely;
complete the authenticated live contract check in the existing worker after the
operator hard-refreshes the new assets.

## 2026-08-12 backend realtime follow-up

The test appliance remains the only active EVA runtime. Its inference `.env`
was not edited; SHA-256 remained
`2c254527143f62bbdbcf7a14914872e2a6f1e0f4f776ef02024c0f27aac76325`.
The two llama.cpp servers and their model/context flags were not restarted or
changed.

Four additional stabilization commits were deployed:

- `461a878 fix: remove summary history from realtime hot path`
- `2a82eb3 fix: stop replaying settled incident projections`
- `7ba6ff6 perf: bound incident projection backfill`
- `2d8a4ee fix: honor disabled probe bookmark cooldown`

The principal backend starvation source was not CV or React. PostgreSQL summary
state persistence rebuilt, serialized and hashed the complete per-channel L0
history every persist interval: 9.1 MB for channel 118 and 7.2 MB for channel
112. The `eva-summary-state` thread consumed about 55% of one CPU core and the
same synchronous path made `/luxriot/start_capture` exceed 30 seconds. New L0
history is now stored as independent idempotent rows while the two legacy history
documents remain readable and untouched. At the final check, 59 new item rows
used 244 KB total; the 16.2 MB of legacy history had not been rewritten.
`eva-summary-state` fell to roughly 0.1-0.3% CPU and a channel restart completed
in 4.10 seconds.

Incident temporal-projection backfill was another periodic CPU consumer: it
replayed expensive episode/relation reads for unchanged incidents every 15
seconds. It now caches completed incident revisions and processes at most eight
backfill records per pass. Measured `eva-incident-maintenance` CPU fell from
about 12% to 0.3% in the observed steady interval. Follow-TTL reconciliation is
unchanged and retains its configured page size.

CV/semantic queue audit found no duplicate encoder path. In the final 20-second
steady window channel 112 accepted 19 semantic frames and channel 118 accepted
20; the microbatcher received 40 frames, completed 39 and had exactly one
in-flight at the sample boundary with queue depth zero. The last image encode
was 78 ms, including 45 ms attributed to the CUDA stage, with 8 ms queue wait;
realtime probe evaluation was 64 ms. The first cold encode after each worker
reload remains very slow because probe text embeddings and CUDA warm-up share
the encoder lock. Cold startup/first-frame latency is still an explicit defect.

Both persisted channel overrides and desired-session state now specify a
one-second capture interval for channels 112 and 118. Reload restored both as
running, batch size 8, VLM profile selected. Channel 112 uses the snapshot lane;
channel 118 uses the dense live-segment fallback because its snapshot endpoint
still has multi-second stalls.

Probe cooldown/dedupe executes after P/N/M scoring and temporal confirmation; it
cannot slow the live score. A live Thumbs-up test produced two confirmed hits:
the first bookmark was sent and the second, 1015 ms later, was correctly marked
`cooldown` under its configured eight-second gate. Gate check, send and mark are
serialized across realtime and legacy daemon lanes. A separate correctness bug
where explicit cooldown `0` reverted to the global default was fixed and tested.

Verification for this follow-up: 214/214 Luxriot inference runtime tests, 57/57
additional Luxriot/archive tests, 22/22 incident maintenance/command tests, and
4/4 focused bookmark dataflow tests. Final service state was `/ready=200`, one
Gunicorn worker, and systemd `NRestarts=0`.

## 2026-08-12 dense capture and encoder isolation follow-up

The deployment `.env` was again left byte-for-byte unchanged; SHA-256 remained
`2c254527143f62bbdbcf7a14914872e2a6f1e0f4f776ef02024c0f27aac76325`.
Neither llama.cpp server was restarted or reconfigured.

The remaining periodic channel-118 gap was a capture-source state-machine bug.
After every successful 60-second dense live segment, auto mode deliberately
lowered `snapshot_slow_streak` and synchronously retried the snapshot endpoint.
That endpoint takes approximately five seconds on emu1, so the retry inserted a
five-second hole between otherwise healthy live segments. Auto mode is now
sticky after two slow snapshots. It still falls back to snapshot if live capture
actually fails, and a restarted or reconfigured session evaluates both sources
from a clean state. Live verification crossed a segment boundary with
`snapshot_count=6`, `slow_snapshot_count=2` and `snapshot_slow_streak=2` all
unchanged while the next segment started immediately.

Cold SigLIP startup now runs one synthetic image through the exact image tower
and prewarms all persisted positive/negative probe phrases before desired camera
sessions are restored. The synthetic frame is not added to a probe buffer or
archive. Probe-registry failure defers only text prewarm; image-runtime failure
still leaves startup explicitly unready. This removes the known image-versus-
probe-text startup race, but a subsequent cold rehearsal still recorded one
unattributed 16-second `_clip_init_lock` wait on the first live work. Steady state
recovered immediately. Add caller-level lock telemetry before claiming cold
startup fully solved.

The synthetic `/probes/bench` endpoint was also able to hold the same production
encoder lock for up to its ten-second diagnostic budget. It is now rejected with
HTTP 409 whenever any video or analytics capture session is active and returns
the current live encoder/batcher telemetry instead. The UI displays the explicit
reason. An idle appliance can still run the synthetic benchmark.

In a clean 20-second steady sample, the two channels completed 39 embeddings,
the microbatch queue remained empty, and lock wait added approximately 1.4 ms in
total across the window. Derived mean image work was about 103 ms per frame and
the CUDA portion about 59 ms; the last observed complete encode/evaluation was
20-21 ms. Channel 112 staleness ranged from roughly 0.18 to 0.79 seconds and
channel 118 from 0.30 to 0.62 seconds away from a segment boundary. There were no
archive drops or failures. CV apex selection remains CPU-only and dispatches one
selected frame into the shared SigLIP batcher; it does not create a second
embedding path.

GPU sampling showed SigLIP on the RTX 5060 Ti in short bursts with 0-2% at the
one-second sampling instants. The RTX 4060 was continuously 76-100% busy in the
same window under the separate Qwen/VLM server; it did not increase the SigLIP
queue because the runtimes are on different GPUs. Host memory was 16 GiB used
with 14 GiB available. Swap still contained about 3.9 GiB of old pages, but a
five-second `vmstat` sample showed zero active swap-in/swap-out after startup.

Operationally, Gunicorn HUP is still not truly graceful for this heavy worker:
the superseded worker is forced out after about eight seconds while the new
worker needs about two minutes to import and load transformers. This creates an
HTTP/capture outage during reload even though systemd remains active and reports
`NRestarts=0`. Treat this as an upgrade-runbook risk; do not advertise zero-
downtime reload until worker readiness handover is redesigned.

Verification for this pass: 319/319 combined runtime-bootstrap, embedding-
batcher, embedding-policy, ProbeManager-attention, Luxriot inference and UI
contract tests. This includes the active-capture benchmark guard. The deployed
service returned `/ready=200`, restored channels 112 and 118 as VLM/batch-8
sessions, retained one worker and reported systemd `NRestarts=0`.

## 2026-08-12 served-capacity, alert latency and live timestamp follow-up

The deployment inference configuration was again preserved byte-for-byte. The
`.env` SHA-256 remained
`2c254527143f62bbdbcf7a14914872e2a6f1e0f4f776ef02024c0f27aac76325`.
Neither inference server was restarted or reconfigured. The VLM remains
Qwen3-VL-4B Q4_K_XL on llama.cpp with 16K context and one parallel slot; the
agent remains Qwen3.5-9B with 65K context on the other GPU/server.

The EVA VLM profile was configured with `max_inflight=3`, while llama.cpp
reported exactly one `/slots` entry and runs with `-np 1`. EVA therefore
admitted up to three requests into llama.cpp's opaque FIFO, where urgent fast
alerts could sit behind normal L0 work. Served llama capacity is now discovered
from the llama-style model metadata plus `/slots`, cached across the probe TTL,
and used as a conservative admission clamp:
`effective_capacity=min(configured_capacity, served_capacity)`. Startup primes
this before restoring streams. Live traces reported configured 3, served 1 and
effective 1; llama metrics subsequently showed `requests_processing=1` and
`requests_deferred=0` under active work.

Fast VLM alert records now carry stage timestamps and durations for source
observation, post-roll, executor wait, evidence preparation, EVA admission,
HTTP inference, postprocessing/bookmark delivery and full event-to-ACK latency,
including token counts and the configured/served/effective capacity. Measured
bookmark HTTP delivery itself was normally only about 50-228 ms. Representative
end-to-end decisions were 8.6-15.5 seconds; post-roll was 2.5-3 seconds,
admission waits reached 2.2-6.7 seconds, and llama inference was about 3.0-6.1
seconds depending on the 2.5K-4.2K-token visual prompt. Evo bookmark delivery
was not the dominant latency.

The async semantic path now reports queue, work and callback timing plus frame
age at submission/work start/work completion. This separated a cold CUDA/lock
outlier from the steady path. The persistent emu1 age was a timestamp contract
bug: when Evo omitted `X-Stream-Start-Time`, EVA anchored synthetic timestamps
before authenticated stream/decoder startup, making freshly decoded frames
appear several seconds stale. Frames without an upstream clock now use their
actual decode-observation time with a monotonic 1 ms tie break; a genuine Evo
source timestamp remains authoritative. A recovered stale-frame rejection also
clears its transient health message instead of leaving the UI degraded forever.

The final post-reload 30-second sample was clean:

```text
channel 112  snapshot      29 semantic frames  0 skipped
channel 118  live_segment  30 semantic frames  0 skipped
async lane                  59 submitted/completed, 0 coalesced, 0 failed
selected frame age          642 ms on both channels at sample end
queue/work/callback         1.5 ms / 85 ms / 1 ms on the last work item
SigLIP microbatch           33 ms compute, 8 ms queue wait, queue depth 0
realtime probe evaluation   33 ms, event age 160 ms, no current error
emu1 timestamp source       decoded_frame_observed_at
```

A separate GPU sample before the timestamp fix showed the shared SigLIP GPU at
about 33% average utilization while both channels still completed 59/59 frames
in 30 seconds. React was not the cause of the semantic delay. The first live
encode after a cold worker start can still take 16-28 seconds, but steady CUDA
work returned to roughly 25-75 ms and no queue accumulated.

Durable PostgreSQL rollups remained healthy after the final reload. At the
check time both channels had L1 `13:00-13:15 UTC`, L2 `12:00-13:00 UTC` with
`source_level=L1`, and L3 `00:00-08:00 UTC` with `source_level=L2`,
`generation_route=agent_profile`, `proposals_only=true`, and
`mutations_applied=false`. A direct app-compatible agent smoke with thinking
disabled returned exactly `EVA_AGENT_OK`; `/health` was OK and the model endpoint
reported 65,536 context.

The final HUP again produced a 137-second HTTP outage. The old worker was forced
out before the new worker completed transformers/SigLIP initialization. On this
reload `/ready` briefly returned 200 while its embedder component still said
`not_loaded` because that component is marked optional, although restored probe
sessions already depend on it. Zero-downtime worker handover and the readiness
contract are still P1 deployment defects; do not use HUP during a live pilot
without an explicit maintenance window.

Verification for this follow-up: Python compilation, `git diff --check`, five
focused timestamp/latest-only/realtime tests, and a final 28/28 explicit
regression set covering LM profiles and capacity caching, runtime bootstrap,
settings provenance, fast-alert timing, realtime probe recovery, dense capture
and async CLIP dispatch. Live service acceptance additionally covered both
channels, the SigLIP path, llama queue metrics, agent inference and durable
L1-L3 lineage.

## 2026-08-12 readiness-gated Gunicorn handover

Commit `623cec4` removes the known hard HTTP outage during a one-worker Gunicorn
HUP without changing the inference topology or `.env`. Gunicorn normally creates
the replacement and immediately retires the old worker. EVA now temporarily keeps
both processes while the replacement performs cold import, CUDA model load, exact
image-path warm-up, persisted probe-text warm-up, and LM capacity discovery. The
old worker remains the sole owner of cameras, inference queues, rollup workers and
maintenance daemons during that interval.

The replacement does not restore camera/background ownership until all of the
following are true:

- its configured embedder loaded and the synthetic live-image warm-up succeeded;
- its durable writers started;
- the previous worker retired after a master-side `TTOU` transition;
- no sibling worker remains under the Gunicorn master.

During overlap, writes routed to the warming process fail with an explicit HTTP
503 `runtime_handover_in_progress`. Readiness now requires both the embedder and
desired Luxriot-session restoration whenever live analytics sessions exist, even
if `EVOSSEARCH_EMBEDDER_REQUIRED=false`. A failed replacement leaves the previous
worker serving and resets the temporary worker target. If the previous worker dies
during warm-up, Gunicorn keeps only the candidate and that candidate acquires the
deferred runtime after it is ready; it cannot remain permanently ownerless.

The live rehearsal HUP began at `17:15:11`. The replacement and serving worker
coexisted during a roughly 240-second cold bootstrap; the master requested `TTOU`
at `17:19:11`, the superseded process retired about ten seconds later, and exactly
one replacement remained. Under a deliberately harsh one-second curl deadline,
14 of 117 `/health` samples exceeded the deadline and the longest bad streak was
two samples. This is still a noticeable latency tail from running a cold
Transformers load beside the active process, but it replaces the former continuous
132-160-second connection outage. Do not call HUP latency-free: the old worker
preserves service while the candidate warms, but the shared host is busier.

Post-handover live state was:

```text
EVA workers:                1
systemd NRestarts:          0
/health:                    HTTP 200, about 18 ms in the spot check
/ready:                     HTTP 200, about 0.96-1.3 s
embedder:                   loaded on CUDA
desired/restored sessions:  2 / 2 (channels 112 and 118)
SigLIP queue:               depth 0, one in flight at the sample boundary
probe/summary errors:       none on either channel
VLM PID / Agent PID:        unchanged
```

`/ready` is intentionally a heavyweight dependency check and must not be used as
a routine high-frequency UI poll. Its one-second deadline can still expire while
it checks PostgreSQL, inference profiles and Luxriot. `/health` is the lightweight
liveness surface.

The same pass made rollup/incident workers explicit single-owner services instead
of constructor/import side effects. It also closes an L0 publication race: manager
history is now canonical before the per-session status feed advertises completion,
so the operator cannot observe a completed summary that rollup/archive readers do
not yet see.

Verification for this pass:

```text
Luxriot runtime + lifecycle/bootstrap: 230 passed
focused lifecycle/bootstrap rerun:      17 passed
installer/update/watchdog bundle:       82 passed
L0 publication race repeated:           5/5 passed
Python compilation / git diff check:     passed
```

All four staged runtime files match the reviewed source by SHA-256. The final
post-rehearsal edge-case and L0 ordering edits are copied into the rehearsal tree
but are not active in worker memory until its next controlled reload/restart; do
not trigger another cold HUP solely for those two non-current edge cases. The live
worker already runs the primary readiness handover implementation. The rehearsal
`.env` remains byte-for-byte unchanged:

```text
2c254527143f62bbdbcf7a14914872e2a6f1e0f4f776ef02024c0f27aac76325
```

Settings provenance was also rechecked read-only. The system unit and current
worker both declare `/home/sasha/Projects/eva-georgia-upgrade-repro/.env` through
`EnvironmentFile` and `EVOSSEARCH_CONFIG_ENV_FILE`; the worker has 66
`EVOSSEARCH_*` startup keys. Therefore the React Settings banner should report the
declared source after a fresh settings request. React already sends only dirty
writable fields and omits blank port/write-only secrets. If the old undeclared
banner persists after a hard refresh, inspect the authenticated `/settings`
response and asset cache rather than writing the env file.

## 2026-08-12 verified cold-start rehearsal bundle

The updater originally allowed only 45-90 seconds for post-install verification,
while the measured one-worker SigLIP cold start on this shared rehearsal host took
roughly 132-240 seconds. A correct install could therefore be declared failed and
automatically rolled back while the replacement worker was still warming. Commit
`760fdc2` gives the universal installer, patch installer, rollback verifier and
standalone verifier a consistent 300-second readiness budget. This does not alter
Gunicorn's request timeout, inference endpoints or model settings.

The final clean release snapshot also includes `8a476dc`, which registers the
timestamped semantic-signal frame route in the API dataflow contract. The route
was already implemented and used by `ProbeSettingsModal`; only the static contract
allowlist was stale.

Commit `1a69745` fixes the Archive research review filmstrip at both layers. The
React modal previously depended on the complete `channels` array. App-level
inventory polling replaces that array every 30 seconds, so an open immutable batch
was cleared to its selected frame and fetched again on every inventory refresh.
The modal now depends only on the selected evidence/batch identity, coalesces
concurrent reads, and keeps a bounded least-recently-used cache of eight completed
batches.

The same request was slow on PostgreSQL despite the existing
`ix_archive_detections_vlm_batch` index. The index is partial and requires
`payload_json ? 'batch_id'`, but the generated query only had an equality on
`payload_json->>'batch_id'`; PostgreSQL cannot infer the partial predicate. On a
live six-frame batch the COUNT and page query therefore used parallel sequential
scans and took 7.904 and 6.141 seconds. With the explicit predicate they use the
existing index and measured 1.473 and 22.587 milliseconds respectively. No schema
migration or new index was needed.

The resulting checked archive is:

```text
/home/sasha/Downloads/eva-ai-georgia-upgrade-0.8.1-to-0.8.7-1a69745.tar.gz
git commit: 1a697459c329bb47733a7e096a59904bee793701
size:       169628563 bytes
SHA-256:    a38c809234a13e5f3ee7f2e08c3528991cce72be712889f53fae3bfa9db11a5d
```

Its outer checksum and every checksum in `runtime/SHA256SUMS` pass. The manifest
reports a clean `main` snapshot, beta 0.8.7 and an included Linux x86-64 media
runtime. The React payload contains `index-OsWNZcqb.js` and
`index-CzNIZODy.css`; the four staged runtime files match the reviewed source
hashes. As with the previously successful rehearsal artifact, this incremental
upgrade bundle reuses the preserved application venv: it does not include a full
wheelhouse or a SigLIP model.

`/home/sasha/Desktop/EVA_GEORGIA_UPGRADE_TEST.sh` now pins this archive and exact
commit and also waits up to 300 seconds in its independent post-update health
gate. Shell syntax validation passes. The launcher still requires an explicit
interactive confirmation after its read-only preflight.

The archive-review patch is also active on the rehearsal service. A
readiness-gated HUP overlapped old worker `3837700` with replacement `3945636`
from `18:36:00` until `18:39:57`; every five-second one-second-deadline health
sample succeeded. Afterwards `/ready` reported the CUDA embedder loaded, Luxriot
reachable and both desired channel sessions restored (2/2). `NRestarts` remained
zero, the two inference PIDs were unchanged, and the served React index references
`index-OsWNZcqb.js`.

Focused verification for this fix was `86 passed, 2 skipped` with `128` subtests,
plus the React suite at 95/95 and a successful TypeScript/Vite production build.

No destructive reset or upgrade rehearsal was run while preparing the archive.
The current beta 0.8.7 service remained available, and the rehearsal `.env` still
hashes to the invariant value recorded above. The next state-changing action is
therefore deliberately still `EVA_GEORGIA_UPGRADE_RECOVER.sh` followed by the
interactive test launcher.

## 2026-08-12 bounded incident admission and review navigation

Commits `f1b2959` and `cdeb6bb` make the Incident Review entry path bounded and
separate operator/safety attention from ordinary L0 episode memory. The review
toolbar now lives in the upper Incident tab panel. Its safe first-open query is
one channel over 24 hours, not all channels over 30 days. Channel and period are
persisted in browser storage; completed query results have a four-entry LRU cache
and concurrent identical requests are coalesced. Switching away and back keeps
the mounted result instead of issuing the default wide query again. Explicit
Refresh remains the only forced reload for the current filter.

The live incident flood was not a React problem. Structured `alerts` produced by
the VLM had been accepted as operator incidents even when they did not match a
saved channel criterion, and ordinary meaningful `events` opened durable cases
at L0. The new server-owned admission boundary is:

- ordinary entry/exit, gesture, object-handling and similar transitions remain
  `episode_event` evidence for L1-L3 rollups;
- a deterministic saved-policy match may create `operator_alert` with
  `priority=operator_criterion`;
- an independently actionable grounded hazard may create `safety_event` or
  `safety_alert` with `priority=safety`;
- an ordinary episode cannot create or refresh a context-only legacy candidate;
- a later grounded operator/safety observation may only upgrade an existing
  matching candidate's provenance and priority;
- event and alert phrasings share canonical entity/action keys, including plural
  entities, so one batch cannot create separate `scene maneuver` and
  `vehicle maneuver` incidents for the same drifting cars.

No historical incident was deleted, closed or bulk-reclassified. Immediately
before the final live handover there were 144 candidate rows. A subsequent real
configured thumbs-up produced one new row with `source=operator_alert_l0` and
`priority=operator_criterion`, so the total became 145. Repeated live emu1 drift
batches updated one existing `vehicle maneuver` safety case and did not create a
new row; its older `scene maneuver` duplicate stopped receiving updates. The old
noise queue still needs a separately reviewed reconciliation action.

The final live worker is `4078032` under master `2014970`. `/ready?details=1`
reports the CUDA embedder loaded, Luxriot sessions restored 2/2, database and LM
profiles ready, and both semantic/CLIP queues at zero. `NRestarts=0`. The VLM and
agent llama.cpp processes retained PIDs `1499650` and `2916440`. The rehearsal
`.env` was not edited and still hashes to:

```text
2c254527143f62bbdbcf7a14914872e2a6f1e0f4f776ef02024c0f27aac76325
```

Both controlled HUPs kept the previous worker serving during the roughly
3 minute 40 second cold bootstrap. One-second health probes each observed one or
two isolated deadline misses during Transformers load or ownership transfer, but
there was no continuous outage. This is the known host latency tail, not an
incident-path queue, and inference flags were not changed. Recoverable copies are
stored beside the deployed files with suffixes
`pre-20260812-incident-admission-f1b2959` and
`pre-20260812-incident-dedupe-cdeb6bb`; the previous React dist is
`react-ui/dist.pre-20260812-incident-admission-f1b2959`.

Verification for this pass:

```text
focused admission/dedup regression: 23 passed
expanded incident/runtime backend:  288 passed
React suite:                        96 passed
TypeScript/Vite production build:   passed
Python compilation / diff check:    passed
```

The new immutable rehearsal archive is:

```text
/home/sasha/Downloads/eva-ai-georgia-upgrade-0.8.1-to-0.8.7-cdeb6bb.tar.gz
git commit: cdeb6bbcf18f99460c7f61f5ee330c58b47f2cce
size:       169632741 bytes
SHA-256:    b1829f603787a126297da250cc720755b0209ca9073323f3a10309477d1fb978
```

Its outer checksum, every `runtime/SHA256SUMS` entry, the staged backend hashes,
and React assets `index-ClBMw_wN.js` / `index-CKjVPy8G.css` were verified. The
manifest records a clean `main` snapshot, beta 0.8.7 and the Linux x86-64 media
runtime; as before, it reuses the target venv and does not contain a wheelhouse or
SigLIP model. `/home/sasha/Desktop/EVA_GEORGIA_UPGRADE_TEST.sh` now pins this
archive and exact commit, and `bash -n` passes. No destructive reset or upgrade
rehearsal was run.

The larger incident hierarchy is not falsely marked complete. L1-L3 already
retain temporal/routine evidence and deterministic scale dispositions, but
automatic durable composition of nested multi-action cases (for example one
traffic collision containing a smaller phone-call episode) remains the next
production implementation step. Card information architecture also remains to
be tightened now that source and priority are trustworthy.

Commit `4529b67` completes the first card information-architecture pass. Incident
cards are now horizontal triage rows rather than a large three-column poster
gallery. They lead with the server-owned reason for attention (operator
criterion, safety signal, or context candidate), then severity, event title,
channel and last-evidence time. Only meaningful lifecycle deviations are shown:
ongoing/ended evidence, active/critical risk, open case, and active Follow. The
four raw state axes remain available in the incident report but no longer consume
the scanning surface. A summary identical to the title is suppressed, full UUIDs
are shortened visually, and the grounded cover is a compact evidence anchor on
the left. All new labels have English and Latvian translations.

This UI-only change was copied live without a worker reload. The service stayed
on worker `4078032`, `NRestarts=0`, the inference PIDs and `.env` invariant stayed
unchanged, and the served assets are `index-DT6XXN8D.js` and
`index-D60Ji9I7.css`. The previous UI is recoverable at
`react-ui/dist.pre-20260812-incident-cards-4529b67`. The React suite passed 97/97,
the focused card test passed 4/4 after the final risk-label addition, and the
TypeScript/Vite production build passed.

The Desktop rehearsal launcher now points to the final checked archive:

```text
/home/sasha/Downloads/eva-ai-georgia-upgrade-0.8.1-to-0.8.7-4529b67.tar.gz
git commit: 4529b67f077727adb54cfc603c1f8cd72a3dacc5
size:       169635001 bytes
SHA-256:    eb859c4d08ca4712e81c747477c8d1f914334da267419cdde0aeadbc01d4ec22
```

The outer checksum, clean manifest, all runtime checksums, backend source hashes
and React asset names passed. The intermediate `cdeb6bb` archive remains
immutable but is no longer selected by the launcher. No RECOVER or upgrade
rehearsal was run.

## 2026-08-12 grounded L2 incident memory and nested episode projection

Runtime commits `ffe9815`, `4d5c3d9`, and `b18876b` complete the first
production-safe L2 composition slice. Investigation found that successful L1
and L2 generation built temporal observations, episode dispositions and routine
ledgers in memory but omitted them from the durable rollup write. Consequently,
almost every successful stored L1/L2 row retained only its narrative, while one
degraded L3 row happened to preserve the temporal fields. Successful, degraded
and review-pending writes now retain the same bounded ledgers. A semantic cached
rollup can be enriched from deterministic children without another LM call.

The automatic boundary remains conservative:

- ordinary L0 episode evidence still cannot create a durable case;
- L2 may attach nested episodes only to an already existing incident grounded
  by independent safety evidence or a high/critical saved operator criterion;
- each child must be within five minutes of the grounded parent itself;
  child-to-child proximity cannot form a transitive scene-wide chain;
- a safety parent may retain cross-entity scene context, while an operator-rule
  parent is restricted to the same primary entity;
- replay uses deterministic composition, episode and observation keys;
- no incident ID is merged, no case or risk state is changed, and every nested
  episode remains operator-review required with `automatic_merge=false`.

This last same-entity restriction was prompted by live data before the hourly
L2 write. An earlier draft would have allowed a high `cat enter` operator alert
to absorb nearby person exits, gestures and arm movement through a transitive
five-minute chain. The parent-bounded implementation rejected that semantic
soup. At the natural hourly live pass, channel 112 stored a new ready L2 with 27
temporal observations and 9 episodes, while channel 118 stored 94 observations
and 2 episodes. Both produced zero compositions: the cat episodes were too far
apart, and emu1 contained a recurrence series of one `vehicle maneuver`
semantic track rather than distinct nested events. No new incident card and no
L2 composition episode was created. This is a positive conservative acceptance,
not a missing callback.

The preceding natural L1 pass also verified the repaired durable path on both
channels. New ready rows for channel 112 retained 8 observations and 3-4
episodes per sampled window; channel 118 retained 20-25 observations and 1-2
episodes. Before this rollout, successful L1 and L2 rows on these channels had
zero temporal/incident ledgers.

Policy matching now treats irregular `left` as `leave` only in explicit egress
phrases such as `left the scene` or `left the camera view`. The historical
`Person turning head` candidate remains untouched for audit, but directional
`turned head left` text can no longer satisfy the configured entering/leaving
criterion. The same historical row subsequently received genuinely grounded
exit observations, so it must not be bulk-deleted automatically.

Incident detail now selects an explicitly marked composition parent instead of
blindly displaying `episodes[0]`, and shows up to eight nested episodes as a
short operator sequence. The copy states that these are attached context and
that no incidents were automatically merged. Full evidence remains under
technical details.

Three readiness-gated HUPs rolled these backend steps onto the rehearsal service.
The final worker is `5849` under unchanged master `2014970`; `NRestarts=0`,
readiness is green, the CUDA SigLIP2 embedder is loaded, and Luxriot restored
2/2 desired sessions. The inference PIDs remained `1499650` and `2916440`, and
their command-line flags were not changed. The rehearsal `.env` still hashes to:

```text
2c254527143f62bbdbcf7a14914872e2a6f1e0f4f776ef02024c0f27aac76325
```

Backend rollbacks are beside the live files with suffixes
`pre-20260812-incident-composition-ffe9815`,
`pre-20260812-nested-episodes-4d5c3d9`, and
`pre-20260812-parent-bounded-composition-b18876b`. The prior React dist is
`react-ui/dist.pre-20260812-nested-episodes-4d5c3d9`. Served assets are
`index-BjX4TzO_.js` and `index-BzONcufY.css`.

Verification:

```text
focused composition/policy checks:  6 passed
incident command suite:             20 passed
combined backend incident/runtime:  327 passed
React suite:                         97 passed
TypeScript/Vite production build:   passed
Python compilation / diff check:    passed
```

The final immutable rehearsal archive selected by the Desktop launcher is:

```text
/home/sasha/Downloads/eva-ai-georgia-upgrade-0.8.1-to-0.8.7-b18876b.tar.gz
git commit: b18876b168903a8b6033d732781c037357940448
size:       169647872 bytes
SHA-256:    41afc3689e15a951b396089779f35efac11f03acedd282c5ec6a2593ce676401
```

Its outer checksum, clean manifest, reused checked media-runtime checksum
manifest, backend source hashes, and React asset names passed. The prior
`4529b67` and `4d5c3d9` archives remain immutable but are no longer selected.
No RECOVER or interactive upgrade rehearsal was run.

Complete multi-incident scene understanding is not claimed. Separate durable
subincident IDs (for example a bystander's phone call inside a collision),
candidate causal/concurrent relations, L3 cross-window composition, and learned
weekday/time-of-day baselines remain the next incident-logic layers. The
architecture contract and research basis are recorded in
`docs/architecture/incident_temporal_memory.md`.

## 2026-08-13 natural episode replay and SigLIP runtime identity

Three additional backend commits are deployed to the rehearsal:

```text
2603969 fix: preserve grounded incident transitions
95b3dff fix: detect live embedding space drift
13d2c21 fix: infer grounded returns to routine
```

The natural channel-112 sequence included drinking, phone use, head resting on
the desk, return to typing, exit/return and a later hand-to-face action. The old
L0 prose saw most of this, but its structured contract omitted the phone action,
misclassified the continuing head-on-desk posture as routine and never emitted
an explicit return boundary. It also allowed a later `info` duplicate to lower a
saved `high` alert and matched a cat criterion to a person interacting with a cat
statue.

The fixed contract requires every grounded action in the prose episode update to
exist in structured events and requires explicit `returned` routine evidence.
Phone use is retained as a distinct context episode, head-on-desk/slumped posture
enters the safety path, alert dedupe preserves the highest severity, and operator
policy matching rejects mismatched primary entities. As a conservative fallback,
L1 now infers `ended_by_routine` only after two later covered L0 children show
ordinary activity for the same entity. An abnormal routine label that independently
passes the incident gate does not count; coverage gaps reset confirmation.

A read-only replay of the preserved 08:15-08:30 EEST L0 records produced a
`person fall` safety episode ending at the two later `person seated at desk`
windows. The intermediate `Person slumped over desk` was not accepted as recovery.
The replay was not written back to PostgreSQL. The historical phone call cannot be
reconstructed deterministically because it exists only in prose; the stronger
structured-output prompt applies to future batches.

The overnight headphones probe regression was not a threshold-editing problem.
The saved probe remains the intended contrastive pair `Person in headphones`
versus `Person`. Its model, revision, processor contract and 768-dimensional
fingerprint had not changed, but historical one-Hz vectors formed incompatible
clusters under that same metadata identity. A clean CPU encoding matched an old
stored vector at cosine 0.994, while a fresh encoding of a current frame and its
nearest live vector were only about 0.447. A separate deterministic control then
proved that the pinned CPU and RTX 5060 Ti FP16 model agree at cosine 0.999999 and
that GPU batch sizes 1/2/3/8, including mixed aspect ratios, are stable. Do not
change the probe phrases, thresholds or SigLIP dtype on the basis of this incident.

The runtime now has a stricter in-memory identity than the durable model identity.
Every model load receives a `runtime_generation`; ProbeManager includes it in frame
and text-cache keys but the durable model/revision fingerprint remains unchanged.
A deterministic image and two deterministic text controls are re-encoded every
120 seconds and compared with the startup generation. Drift fails readiness as
`runtime_drift` instead of persisting misleading scores. After both readiness-gated
HUPs the repeated live canary remained healthy at image/text cosine 1.0. The
rehearsal has one worker (`855105`), both channels restored, and no probe/capture
errors. The VLM and agent llama.cpp PIDs remained `1499650` and `2916440`.

The rehearsal `.env` was not changed and still hashes to:

```text
2c254527143f62bbdbcf7a14914872e2a6f1e0f4f776ef02024c0f27aac76325
```

Recoverable live copies use suffixes
`pre-20260813-siglip-incident-95b3dff` and
`pre-20260813-routine-boundary-13d2c21`. Verification completed with 174/174
Luxriot capture/temporal tests, 21/21 embedding/space/batcher tests, Python
compilation and `git diff --check`. Manual post-reload acceptance of the same
headphones probe is still required; compare a short no-headphones / headphones /
no-headphones cycle and do not infer acceptance from the canary alone.

## Next work

### 1. Confirm the committed baseline

Start with:

```bash
cd /home/sasha/Projects/evo-ssearch-office-demo
git status --short
git diff --check
```

The expected result is a clean tree at the committed stabilization baseline. If
new changes appear, inspect and preserve them.

### 2. Preserve the recorded regression baseline

The focused installer/update/watchdog bundle passed 83 tests. A broader pass also
covered:

- installer and update bundle;
- migration `0006 -> 0013`;
- auth/CSRF mutation routes;
- React console asset verification;
- archive row/thumbnail preservation;
- LM profiles and inference queue;
- L0/L1 rollups, alerts and probe behavior;
- incident creation, covers and review;
- watchdog and readiness gates.

That exact broader command completed with `236 passed, 2 skipped` and `128`
subtests passed. Its only two warnings were upstream Python 3.14 deprecations from
`clip`/`pkg_resources` and `torch.jit.load`.

Do not claim the repository's entire test corpus is green; only the recorded
focused and broader commands were run.

### 3. Keep the verified bundle immutable

The Desktop wrapper points at the checked `1a69745` archive above. Do not edit that
tarball in place. If release-source code changes, build a newly named bundle and
repeat the manifest, outer checksum, internal runtime checksum and React-dist
checks before changing the wrapper.

Build only from the reviewed committed source; do not fold unrelated later changes
into the release archive.

### 4. Reset and perform the next manual Georgia rehearsal

Run, in order:

```bash
/home/sasha/Desktop/EVA_GEORGIA_UPGRADE_RECOVER.sh
/home/sasha/Desktop/EVA_GEORGIA_UPGRADE_TEST.sh
```

Before apply, record without exposing secrets:

- EVA version `0.8.1`;
- Alembic revision `20260614_0006`;
- archive row, thumbnail and sidecar counts;
- inference-policy fingerprint;
- hashes of any local inference systemd units/drop-ins;
- `/models` IDs and reported context from both endpoints;
- service/unit existence state.

After apply, require:

- EVA `0.8.7` and React console served on port 5081;
- Alembic revision `20260805_0013`;
- archive data and available thumbnails preserved;
- existing channel configuration still present and channels can reconnect;
- the exact inference-policy fingerprint is unchanged;
- local inference unit/drop-in hashes are unchanged;
- model IDs/endpoints/contexts are unchanged;
- no vLLM/llama service restart was caused by the updater;
- authentication and CSRF-protected mutations work;
- incident report, probe mutation and settings saves work;
- `/health` succeeds and `/ready?details=1` explains only genuine external-state
  degradation;
- rollback command is printed and points at a valid backup.

### 5. Focused behavioral acceptance after updater safety

Once the update path is proven, retest rather than assume the recent fixes:

- L0 alert criteria actually produce alert badges/bookmarks;
- stream batch/cadence changes apply and runtime state reflects them;
- L0 latency stages are visible and no artificial watchdog restart occurs;
- 15-minute L1 passes run and are visible;
- incident cards have grounded cover images;
- probe settings expose live sub-threshold P/N/M signals where implemented;
- React has no large overlay/glitch and remains responsive when the display GPU is
  saturated by local rehearsal inference.

## Explicit non-goals for the next pass

- Do not change the now-stable local RTX 4060 flags merely for benchmark numbers;
  preserve the measured 16K/one-slot/Q4_K_XL baseline unless an acceptance test
  demonstrates a regression.
- Do not change Georgia production vLLM flags, model IDs, context, batching or GPU
  layout as part of the EVA update.
- Do not migrate Georgia to SigLIP2 by silently replacing an explicit existing
  embedding configuration. Bundled SigLIP assets may be installed for EVA use only
  under the reviewed append-only rules.
- Do not fold unrelated later work into the reviewed stabilization commits or
  into the rehearsal bundle.
- Do not print `.env`, database DSNs, Evo passwords or LM API keys in logs/handoff.

## Permission/harness note

The previous Codex session was opened with workspace root
`/home/sasha/Projects/rabbithole/bonsai` even though the active work was in
`evo-ssearch-office-demo`. Its sandbox repeatedly requested permission for ordinary
edits outside the stale root. The user had already granted full access. This was a
session/workspace-root mismatch, not an EVA issue. Start the next session from the
primary repository path above.

## 2026-08-12 disk cleanup and nested-incident source checkpoint

Disk cleanup was completed before the next incident-logic step. Free space rose
from about 21 GiB / 4.65% to 27.29 GB / 5.57%, so the archive retention disk
check returned from `low_space` to `ready` before the operator intentionally
stopped EVA for about 30 minutes.

Removed, with exact bounded targets:

- ten superseded Georgia upgrade archives in `~/Downloads`; the current
  `b18876b` archive remains and still hashes to
  `41afc3689e15a951b396089779f35efac11f03acedd282c5ec6a2593ce676401`;
- the unused vLLM torch compile cache;
- 1.027 GB of Docker build cache and 529.5 MB of unused Docker image layers.

Preserved deliberately:

- `/home/sasha/Downloads/eva-checked-media-runtime-4529b67`;
- all model files, both llama.cpp command lines, the Georgia PostgreSQL volume,
  and `.env` (control hash remained
  `2c254527143f62bbdbcf7a14914872e2a6f1e0f4f776ef02024c0f27aac76325`);
- the stopped legacy `eva_ai_postgres_data` volume, about 40.04 GB. It is not a
  cache and must not be deleted without an explicit data-retention decision;
- the Ubuntu installer ISO and unrelated user files.

About 4.1 GB of persistent journal plus large rotated `syslog`/`kern.log` files
remain because journal vacuum and rsyslog rotation require interactive `sudo`.
Inspection showed two main sources: verbose llama-server request timing in
`syslog`, and repeated AppArmor denials from the Mission Center snap in the
kernel log. Cleanup did not alter inference flags or logging configuration.

Source commit `f25028e` (`feat: materialize nested incident context`) implements
the next conservative hierarchy slice:

- L2 nested tracks receive deterministic, replay-safe incident IDs;
- each child remains a first-class directly addressable incident linked to the
  grounded parent by candidate `concurrent_with` context; this asserts neither
  causality nor merge;
- nested children are excluded in PostgreSQL before review-board
  `COUNT/LIMIT/OFFSET`, so they cannot flood or sparsify top-level review pages;
- the parent modal can navigate to each separate child and the child back to its
  parent;
- explicit operator `confirm`, `reopen`, or `follow` promotes a child to the
  top-level review board. Rollup/model output cannot promote it.

Local acceptance at this commit:

- 51 focused incident command/store/API tests passed;
- 322 related incident and Luxriot inference-runtime tests passed;
- 97 React tests passed;
- TypeScript/Vite production build passed, producing
  `index-BJEko90E.js` and `index-ToIv3F5P.css`;
- Python compile and `git diff --check` passed.

This commit has **not** been deployed to the rehearsal instance and no new
upgrade archive or Desktop launcher pin has been created. The operator
explicitly switched EVA off during local development. When EVA is available
again, first confirm the service/process ownership and `.env` hash, then perform
the ordinary readiness-gated rollout, live API/UI acceptance, immutable bundle
verification, launcher update, and final handoff amendment.

## 2026-08-13 nested-incident live rollout

The operator brought EVA/Luxriot back and the pre-rollout control checks passed:
the service was ready on beta 0.8.7, PostgreSQL remained at `20260805_0013`,
Luxriot restored channels 112 and 118, `.env` retained its control hash, and the
VLM/agent llama.cpp processes retained PIDs `1499650` and `2916440` with their
recorded command lines.

Commit `f25028e` was staged with bounded rollback copies:

- backend suffix `pre-20260813-nested-incidents-f25028e`;
- React rollback directory
  `react-ui/dist.pre-20260813-nested-incidents-f25028e`.

One graceful HUP was sent only to Gunicorn master `2014970`. Old worker `5849`
continued serving while replacement `170651` loaded CUDA/runtime state. After
the old worker retired, readiness briefly returned 503 while the sole new worker
acquired background ownership, then returned 200 with both desired channels
restored. `NRestarts=0`; neither inference process was restarted.

Live acceptance after handover:

- `/health=200`, `/ready=200`, beta 0.8.7;
- CUDA SigLIP2 loaded, PostgreSQL reachable, Luxriot reachable/restored 2/2;
- CLIP, semantic archive and realtime-probe pending queues were zero;
- served React assets are `index-BJEko90E.js` and `index-ToIv3F5P.css`;
- staged backend and React SHA-256 values match reviewed source/build output;
- authenticated operator login, review list, full list, temporal context and
  logout returned 200;
- review and full totals were both 149; PostgreSQL contained zero nested-only
  children immediately after rollout, proving the change did not retroactively
  multiply historical records;
- temporal output includes the new `nested_incidents` list contract.

No synthetic incident was inserted into the live operator queue. Separate child
materialization and replay behavior remain covered by the 51 focused tests and
322 related incident/Luxriot runtime tests recorded above; a natural eligible L2
composition will exercise the same server path without test-data pollution.

The final immutable rehearsal archive for this pass is:

```text
/home/sasha/Downloads/eva-ai-georgia-upgrade-0.8.1-to-0.8.7-2fae764.tar.gz
git commit: 2fae764ac842501f73934571e47706c5066aed43
size:       169657772 bytes
SHA-256:    dd3ae8939b5e5c4d407bcf88a1ad7a4f8e30e9aa795b2ddd5449c754b4a39a23
```

The outer checksum, clean manifest, every reused media-runtime checksum, staged
backend source hash, and React assets `index-BJEko90E.js` /
`index-ToIv3F5P.css` passed. The bundle contains the checked Linux x86-64 media
runtime but deliberately contains neither a wheelhouse nor the SigLIP model; it
preserves and reuses the accepted target venv/model configuration.

`/home/sasha/Desktop/EVA_GEORGIA_UPGRADE_TEST.sh` now pins this archive and the
exact manifest commit; `bash -n` and a final outer checksum check pass. The prior
`b18876b` archive remains immutable as the previous recovery artifact. No
destructive RECOVER or interactive upgrade rehearsal was run while producing
this archive.

## 2026-08-13 grounded incident hierarchy invariant

Commit `181c4f8` (`fix: keep grounded incidents in top-level review`) repairs a
live hierarchy defect found while resuming incident work. The read-only channel
112 audit contained this chain:

- top-level grounded `cat exit` operator incident;
- `cat enter`, originally nested under it, later upgraded by direct saved-policy
  L0 observations to `priority=operator_criterion` but still carrying the stale
  `presentation.scope=nested` marker;
- a new context-only `cat movement` child nested below that stale nested row.

This made a configured alert disappear from the top-level review board and
allowed a nested incident to become the parent of another automatic composition.
The fixed invariant is:

- active semantic-track selection prefers an existing top-level incident over a
  recently updated nested context row;
- L2 composition rejects every nested-marked candidate as an automatic parent;
- a later server-classified `operator_alert`, `safety_alert`, or `safety_event`
  promotes a matching nested row to `top_level`, retains its old parent link for
  audit/navigation, and preserves the highest grounded severity;
- PostgreSQL review filtering defensively includes older nested-marked rows that
  already carry direct `operator_criterion` or `safety` priority. Context-only
  children remain excluded before `COUNT/LIMIT/OFFSET`.

No historical row was rewritten or bulk-reclassified. After the live rollout,
the four-hour channel-112 query returned eight full records and seven review
records: the stale direct `cat enter` alert is visible in review, while the new
context-only `cat movement` child remains hidden and directly addressable. A
future grounded observation will persist the repaired `top_level` presentation
on the old row; visibility no longer depends on waiting for it.

Verification and rollout controls:

```text
focused incident command/store tests:             50 passed
expanded API/maintenance/temporal/admission tests: 84 passed
source commit:                                     181c4f8
live Gunicorn master / worker:                     2014970 / 913845
VLM llama.cpp PID:                                 1499650
agent llama.cpp PID:                               2916440
```

One readiness-gated HUP replaced worker `855105` with `913845`. The old worker
served `/health=200` throughout the roughly 3 minute 46 second cold SigLIP load;
after ownership transfer `/ready` returned from 503 to 200 in about four seconds.
Both channels restored, PostgreSQL and Luxriot are reachable, the CUDA SigLIP2
canary reports image/text cosine 1.0, and there are no capture/probe errors. The
inference processes were not restarted. The rehearsal `.env` remains unchanged
at SHA-256:

```text
2c254527143f62bbdbcf7a14914872e2a6f1e0f4f776ef02024c0f27aac76325
```

Recoverable backend copies are beside the live files with suffix
`pre-20260813-incident-top-level-181c4f8`. No new immutable upgrade archive or
Desktop launcher pin was produced in this slice.
