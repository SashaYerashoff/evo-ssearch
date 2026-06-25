# EVA AI Agent / Archive Deep Audit

Дата: 2026-06-14

Контекст: перед клиентским пилотом на 30-50 каналов, с перспективой дальнейшего масштабирования. Цель аудита - проверить, что агент видит VLM-сводки, понимает пробы, не ломается на массовых сценариях и что архив реально копится для поиска.

## Краткий вывод

Система уже достаточно сильна для управляемого пилота: пользователи/права, audit trail, secure adapter для agent tools, Postgres archive/runtime state, probe archive и VLM rollups присутствуют.

Главный разрыв: VLM-видеоописания и кадры, которые модель смотрит для summary, сейчас не становятся долговременным searchable CLIP-архивом сами по себе. Searchable archive пополняется устойчиво через probe hits. Live VLM capture эмбеддит кадры в RAM-буфер проб, но это не durable archive и не переживает рестарт.

Второй разрыв: агент умеет рассуждать и читать данные, но массовые операции на 50-100 каналов пока не имеют хороших bulk-инструментов и бюджетов. Старый лимит в 8 tool calls убран, но вместо него пока нет безопасного большого лимита/таймаута/чанкинга.

## Что уже есть

- Агент видит VLM-сводки через `get_video_summaries`.
- `get_video_summaries` читает L0/live, L1, L2, L3 через `LuxriotManager.summary_rollups`.
- Summary history/runs сохраняются в `archive.runtime_state`, если включен Postgres archive mode; иначе fallback в JSON state file.
- VLM alert metadata уже считается для L0-L3: `alert_counts`, `alert_total`, `alert_severities`.
- UI уже может показывать alert badges на свернутых summary rows.
- Probe runtime семантика понятная:
  - `pos_score = max(all positives)`
  - `neg_score = max(all negatives)`
  - hit если `pos_score >= pos_floor` и `pos_score - neg_score >= margin`
  - без negatives `neg_score = 0`
- Probe hits пишутся в detection archive через `_store_probe_hits`, включая thumbnail, `clip_vec`, severity, bookmark flags, payload.
- `/probes/cast` в HTTP API уже есть: channel list, conflict policy, cap 500 channels, audit completion.
- Secure agent adapter:
  - режет filesystem archive для агента;
  - enforce channel scope для scoped users;
  - write tools preview-only;
  - ограничивает rows/output/rate.
- Relevant tests passed:
  - `tests.test_agent_tool_loop`
  - `tests.test_eva_agent_adapter`
  - `tests.test_agent_tool_gateway`
  - `tests.test_embedding_policy`
  - `tests.test_luxriot_inference_runtime`
  - `tests.test_api_dataflow_smoke`
  - `tests.test_http_auth_routes`

## Главные риски

### P0 - VLM frames are not durable searchable archive

`LuxriotCaptureSession` отправляет каждый snapshot в `probe_manager.add_frame`, но это RAM buffer. Durable archive row создается только при probe hit:

- `probe_daemon`
- `/probes/query`
- `/probes/run`

Следствие: если канал только описывается VLM, но на нем нет проб или пробы не сработали, по его кадрам потом нельзя искать через detection archive.

Нужно: сохранять минимум 2 anchor frames на каждый VLM batch в Postgres archive или отдельную `archive.frames`.

### P0 - VLM alerts do not persist alert anchor frame

`accept_summary_entry` вызывает `process_summary_alerts`; тот отправляет Luxriot bookmark и обновляет fingerprint cache. Но он не сохраняет ближайший frame как searchable archive record.

Нужно: при parsed alert сохранять nearest batch frame как `source='vlm_alert'`, с severity, alert payload, summary/run linkage.

### P0 - Agent has no hard tool budget

Лимит 8 убран, тест подтверждает >8 tool rounds. Но `AgentRunner.stream_chat` теперь может крутиться до тех пор, пока модель продолжает просить tools. Gateway умеет timeout, но adapter не задает `timeout_seconds`.

Нужно: высокий, но конечный budget:

- max tool calls per turn: 64 или 128;
- max wall-clock per turn;
- per-tool timeout;
- repeated-call guard;
- graceful "остановился по бюджету, могу продолжить".

### P1 - Bulk probe workflows missing from agent surface

HTTP `/probes/cast` есть, но agent tools имеют только per-channel `create_probe`. Для 50-100 каналов агент будет делать десятки create calls, теряя grouping/approval UX.

Нужно: `cast_probe` tool:

- preview-only in secure mode;
- channel_ids / channel_refs;
- conflict: skip/create/update;
- copy_roi flag;
- counts created/updated/skipped/failed;
- audit and approval plan.

### P1 - `survey_channels` is serial and all-channel by default

Если не передать `channel_ids`, агент может опросить все каналы. Defaults: 12 sec, 4 samples, одна VLM-операция на канал. На 50-100 каналов это одна длинная tool call без пагинации.

Нужно:

- explicit confirmation if target channel count > 10;
- `limit/offset` or cursor;
- fast inventory mode separately from expensive VLM survey;
- bounded parallelism later.

### P1 - Agent sees only compacted subsets for large outputs

Model-side compaction currently keeps:

- first 12 channels from `list_channels`;
- first 12 probes from `list_probes`;
- first 8 channels from `survey_channels`;
- first 8 archive/search results.

UI can receive the full tool result, but the final reasoning step sees only compacted subset.

Нужно: compaction with aggregate summaries and omitted counts, plus pagination/chunk tools.

### P1 - Image probe threshold affects text probes

Image probe embedding is appended into positives, then global `pos_floor` becomes `max(text_floor, image_pos_floor)`. Default image floor around 0.7 can accidentally make text positives at 0.2 unusable.

Нужно: separate text and image acceptance:

- `text_ok = text_pos >= text_floor`
- `image_ok = image_pos >= image_floor`
- final positive condition: `text_ok OR image_ok`
- margin still applied against negatives.

### P2 - `/probes/query` accepts string positives/negatives poorly

Save/cast normalize positives via `_probe_text_values`, but `/probes/query` passes raw `data.get('positives')`. A JSON string `"person"` can become character-level prompts.

Нужно: normalize or reject non-list values in `/probes/query`.

### P2 - Agent cannot manage ROI/image probe

UI/API can save ROI and image probes. Agent `create_probe` hard-codes image probe off and ROI off; `update_probe` preserves those fields.

This is safe, but limited. For pilot, acceptable if UI owns ROI/image-probe setup. For agent-led setup, add explicit preview tools later.

## Implementation plan

### Phase 1 - Durable VLM archive

1. Add a small VLM frame archiver around summary batches.
2. Persist 2 deterministic frames per VLM batch:
   - first frame;
   - last frame;
   - optionally middle/alert-nearest frame.
3. Use existing detection store initially:
   - `source='vlm_summary'`;
   - synthetic `probe_id='vlm_summary'` or nullable probe identity;
   - payload includes `run_id`, `batch_start_ms`, `batch_end_ms`, `frame_index`, `summary_ref`.
4. On VLM alert, persist one extra row:
   - `source='vlm_alert'`;
   - severity from alert;
   - payload includes alert title/description/state/timestamp and summary linkage.
5. Extend archive UI/search filters to distinguish probe hits vs VLM frames vs VLM alerts.
6. Add tests:
   - summary batch stores two frame records;
   - alert stores alert anchor;
   - search_archive can retrieve `vlm_summary`/`vlm_alert` rows;
   - retention sees these rows.

### Phase 2 - Agent bulk safety

1. Add finite tool budgets to `AgentRunner`.
2. Add adapter `timeout_seconds` per tool.
3. Add `cast_probe` agent tool backed by current cast logic or shared service.
4. Add `survey_channels` channel cap and pagination.
5. Add multi-channel summary rollup tool:
   - `get_video_summary_rollups(channel_ids, depth, since_hours, limit_per_channel)`;
   - returns freshness, counts, alert totals, latest summaries.
6. Update playbooks:
   - `protocol_deploy`: chunk channels, do not survey all by default;
   - `probe_tuning`: explain positive/negative examples as flat examples, not pairwise scoring;
   - `archive_research`: VLM frame archive vs probe hits.

### Phase 3 - Probe semantics cleanup

1. Split image/text positive floor logic.
2. Normalize `/probes/query` positives/negatives.
3. Decide UI wording:
   - either "positive examples / negative exclusions";
   - or implement true pair-aware scoring later.
4. Decide disabled-probe manual run behavior:
   - document as manual override;
   - or require `force=true`.

### Phase 4 - Pilot hardening

1. Make `/health`/`/ready` fail loud if `SECURE_DEPLOYMENT_REQUIRED=true` and archive migration is missing.
2. Add archive capacity estimate defaults for 50-channel pilot.
3. Add a runbook section for clearing CLIP/SigLIP buffers after embedder switch.
4. Add a small admin-facing "archive ingest health" metric:
   - VLM batches seen;
   - frames archived;
   - alert anchors archived;
   - probe hits archived;
   - last archive write error.

## Current capacity note

The built-in estimate for 50 channels, 5s snapshot interval, batch size 12, and 2.5 retained frames per batch gives about 192.5k frame/archive rows per day. With current `ARCHIVE_MAX_RECORDS=5_000_000`, that cap is reached before 90 days. This is acceptable only if the pilot has explicit retention expectations and enough disk, or if we lower frame retention/frames per batch.

## Bottom line for pilot

Before Georgia deploy, close these first:

1. Durable VLM frame archiving: 2 frames per batch + alert anchor.
2. Agent tool budget/timeouts.
3. Agent `cast_probe` or at least a non-agent bulk cast workflow tested end-to-end.
4. Survey/channel chunking guard.
5. `/probes/query` normalization and image/text threshold fix.

Everything else can be staged after the first controlled deployment.
