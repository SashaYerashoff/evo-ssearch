# EVA AI beta 0.8.3: implementation audit и release-candidate gates

Дата: 2026-07-11

Статус: **implementation complete / release candidate в текущем working tree**

Baseline: `9e39392ae6bfce9252715113603547806517f1a1`

Рабочая ветка: `stabilization/0.8.3-pre-react`

Этот документ описывает фактически реализованное состояние, а не первоначальный
план работ. RC пока не является неизменяемым релизом: изменения ещё должны быть
зафиксированы commit/tag, собраны в проверяемый offline bundle и пройти полевой soak.

## 1. Какая точка выбрана

Точка определена по Git tree, а не по названию ветки:

- `stable/office-demo`;
- `feature/secure-50-channel-foundation`;
- `origin/feature/secure-50-channel-foundation`.

Все три ссылки указывали на commit `9e39392` и tree
`29ecbafba4fa47f0df056a943e4483d8cef65cc6`. React впервые появляется отдельно в
`c0a0c37` (`feature/react-ui-prototype`), а затем в истории `main`. Выбранный baseline
— последняя pre-React 0.8.3 foundation, одновременно последний `office-demo`.

Реализация ниже наложена на этот baseline в ветке
`stabilization/0.8.3-pre-react`; React в RC не добавлялся.

## 2. Решение по готовности

RC устраняет подтверждённые correctness-дефекты и добавляет наблюдаемые границы
нагрузки. Он пригоден для контролируемого показа и следующего этапа полевой
валидации, но ещё не даёт основания обещать безусловную production-работу 50
одновременных browser video players или полную семантическую реконструкцию событий
при входе 1 FPS.

| Контур | Реализовано в RC | Честная граница |
| --- | --- | --- |
| Agent: каналы и период | Составной inventory, Unicode resolver, точные bounds, coverage и period/event evidence | Полнота ответа видима, но не создаёт отсутствующие upstream/history данные |
| Agent: continuation | Server-side persisted research ledger с frozen window и remaining IDs | Ledger рассчитан на bounded research, а не на произвольный долговечный workflow engine |
| LM contention | Общая in-process admission queue для Agent/VLM/rollups и диагностика `/lm/admission` | Не distributed scheduler; действует внутри обязательного single-worker процесса |
| Channel settings | Explicit `false`/`0`, reset-to-inherit, source/revision/persistence status, stale-response guards | Внешний systemd environment по-прежнему выше `.env` и требует осознанного restart |
| Restart/desired state | Per-channel generation/side-effect locks и сериализованный desired-state RMW | Locks process-local; несколько Gunicorn workers по-прежнему запрещены |
| CV/CLIP/VLM apex | Incremental 60-second dense pipe; exact batch slices внутри окна; один выбранный attention frame на представленный source-second с provenance; тот же frame идёт в configured CLIP Probe, VLM и archive | 1 FPS — deterministic fallback, **не** semantic intra-second apex; low-cadence source честно получает `apex-lag` |
| Rollups | `target_level`, отсутствие скрытого расчёта более высоких levels, elapsed/progress/LM queue hint | Синтез всё ещё выполняется в HTTP request, а не background materializer |
| Operator media | Tokenized same-origin live/archive broker плюс shared `EVA attention preview`; явный lease, proactive renewal/watchdog, Range, archive alignment и honest degraded states | `Full live` расходует второй recorder stream; это не 50-player media plane или universal transcoder |
| Offline install | Dry-run-first orchestrator, env adoption, lock, privileged migration identity, backup/verify/rollback handoff | PostgreSQL/TLS/model assets не провиженятся; clean-host apply ещё не принят |

## 3. Agent: каналы, период, evidence и continuation

### 3.1 Реализованный контракт

- Luxriot `/channels` больше не обрывается на первом JSON object: initial fragments
  и resource deltas объединяются; явно незавершённый initial state считается
  ошибкой. При refresh failure сохранённый inventory остаётся доступен как stale с
  диагностикой, а не исчезает молча.
- Agent строит inventory как объединение live inventory, runtime streams, desired
  state, summary history/status digest и явно запрошенных numeric IDs. Для каждого
  найденного канала возвращается `inventory_provenance`.
- Broad scope сохраняет `requested`, `checked`, `inactive`, `errors`, `unchecked` и
  `deferred` IDs и их counts. Ограниченные списки не выдаются за полный результат.
- Channel title resolver использует Unicode NFKC + `casefold`; грузинские и
  кириллические названия не отбрасываются ASCII-регуляркой. Numeric ID остаётся
  наиболее надёжным операторским ключом.
- `EVOSSEARCH_SITE_TIMEZONE` при необходимости явно задаёт calendar semantics
  Agent; нейтральный default — `UTC`. Если модель передала только одну границу уже frozen
  окна, server дополняет вторую границу, а не заменяет период случайным default.
- `get_video_summaries` выбирает evidence по периоду: начало/событийные или
  deviation/state/vector окна/конец. Архивные результаты сохраняют `coverage` и
  similarity в model compaction и turn signal ledger.
- Continuation хранится отдельно от chat prose в PostgreSQL metadata с owner scope:
  immutable `frozen_window`, requested/completed/remaining IDs, ошибки и признак
  window drift. Команда `continue` получает server-trusted remaining scope и не
  восстанавливает его из пересказа модели.

### 3.2 Что это гарантирует и чего не гарантирует

Agent теперь обязан структурно сказать, какую часть периода и списка каналов он
реально проверил. `partial`, `truncated`, `no_data`, `unchecked` и `deferred` нельзя
переводить в «ничего не происходило».

При этом остаются разные retention-контуры: оперативная L0 memory может уже не
содержать период, хотя PostgreSQL Archive содержит кадры. Архивный semantic search
также имеет bounded candidate window. Coverage — честный отчёт о просмотренном, а
не математическое доказательство отсутствия события.

Per-turn attention по умолчанию остаётся ограниченным; 50-channel review должен
идти чанками через persisted ledger. Максимальные ID lists и ledger также bounded,
чтобы один запрос не раздувал model context без контроля: inventory payload хранит
до 100 IDs на список, ledger — до 1 000. Грузинский scope из 50 камер помещается в
эти границы.

### 3.3 Реальные acceptance hooks

Добавлены:

- frontend-equivalent authenticated client в `tests/integration/eva_client.py`;
- frozen read-only manifest в `tests/integration/real_data_manifest.py`;
- opt-in contracts в `tests/integration/test_live_attention_contract.py`;
- randomized Agent/VLM contention soak в `scripts/live_attention_soak.py`.

Они проверяют сохранение requested scope, exact window, evidence bounds,
continuation без повторения первого chunk, SSE/tool errors, desired-state drift и LM
queue peaks.

Authenticated live Agent acceptance на frozen реальном окне теперь запускался через
обычный admin login, без сброса пароля и обхода auth:

```text
2 passed, 1 skipped, 3 subtests passed
```

Пройдены conservation явно запрошенного channel scope и exact-window/evidence
bounds для трёх проверенных каналов. Continuation test был корректно skipped:
выбранный live scope содержал только три канала, а этому contract нужен scope больше
восьми, чтобы появился второй chunk. Это acceptance HTTP/tool wiring и фактических
данных, но **не** 20–30-минутный randomized Agent/VLM contention soak; он остаётся
отдельным RC gate. Credentials в test output и evidence не сохраняются.

## 4. Channel settings, persistence, restart и late results

### 4.1 Настройки и bookmarks

- `bookmark_enabled=false`, нулевой cooldown и пустые значения сохраняются как
  explicit channel overrides, даже если effective default совпадает с новым
  значением.
- Есть отдельная операция `clear_override_fields`: reset удаляет channel override и
  возвращает наследование default. Один field нельзя одновременно update и reset.
- GET settings возвращает effective values, defaults, `override_fields`,
  `setting_sources`, persistence backend/revision/last error/dirty state.
- UI Apply отправляет только поля, реально изменённые относительно загруженного
  channel; inherited defaults больше не закрепляются скрытым override.
- Persistence failure откатывает RAM update и возвращает ошибку. Restart не
  останавливает текущую session, пока новые settings, desired state и run state не
  сохранены.
- Channel `model_hint`, interval, prompts, bookmark policy и road calibration
  восстанавливаются из persisted summary state.

### 4.2 Precedence и environment source

UI показывает безопасный, без значений секретов, precedence report:

```text
process/systemd environment > project .env > runtime default
                                  + persisted runtime defaults
                                  + per-channel overrides
```

`EVOSSEARCH_CONFIG_ENV_FILE` декларирует путь, загруженный systemd
`EnvironmentFile`, и помогает диагностике назвать внешний источник. Это **не**
альтернативный dotenv loader и не обещание, что Settings editor перепишет внешний
файл: текущий editor читает/пишет project `.env`. При внешнем env оператор должен
менять канонический site file через installer/approved host procedure и выполнять
restart; process environment всё равно имеет приоритет.

### 4.3 Изоляция restart и side effects

- Start/stop/restart одного channel сериализованы per-channel side-effect lock.
- Desired live-session map защищён отдельным lock на весь read/modify/write, поэтому
  параллельный start B не теряет persisted state A.
- Каждая session имеет generation. Generation проверяется после ожидания LM
  admission непосредственно перед HTTP inference и ещё раз до bookmark, state,
  archive и history side effects.
- Late completion superseded generation возвращает явный stale/superseded status и
  не создаёт bookmark, current state, archive event или обычную history запись.
- Capture error и summary error разделены: новый snapshot больше не маскирует
  неуспешное описание до следующего успешного summary.
- UI settings, summaries, rollups, player и archive seek используют Abort/generation
  guards, поэтому поздний response A не должен отрисоваться или сохраниться в B.

Эти locks рассчитаны на `EVOSSEARCH_GUNICORN_WORKERS=1`. Они не являются
межпроцессным CAS; увеличение worker count создаст независимые managers и остаётся
unsupported.

## 5. CV apex, CLIP и VLM

### 5.1 Что реализовано

Capture path группирует входные frames в channel-local секундные buckets. При
нескольких frames лёгкий CV selector сравнивает grayscale frame deltas и выбирает
один attention candidate; если положительного score нет, используется
deterministic temporal midpoint. Selection сохраняет:

- policy/version и channel;
- bucket start;
- source frame indices/timestamps/hashes;
- selected index/timestamp/hash;
- selection source, score source и fallback reason.

В ProbeManager/CLIP попадает выбранный frame, а не все сырые frames bucket. Summary
layer сохраняет один выбранный frame на секунду и может использовать road-CV/CLIP
signals как provenance для attention selection. Tests трассируют один и тот же
selected hash через CLIP/probe input, VLM message и archive anchor; raw non-apex
frames в CLIP buffer не индексируются.

Dense `live_segment` path больше не пытается сначала удержать целый segment в
старом фиксированном 8 MiB download budget и не переоткрывает ffmpeg на каждый
12-frame summary batch. Один bounded capture window вычисляет:

- capture lease (default 60 s) и отдельный summary cadence как
  `batch_size * interval`;
- bounded raw-frame budget как capture-window seconds × dense FPS;
- transport budget, достаточный для всего capture window, с верхним пределом;
- stream/read/process timeouts и inflight diagnostics.

Server-side tokenized Luxriot response подаётся в `ffmpeg` через stdin pipe;
credentials и short-lived stream token не попадают в argv, browser URL или
временный video file. `image2pipe` декодируется инкрементально: second buckets
финализируются во время того же открытого 60-second window, а exact 12-apex summary
batches уходят в VLM worker без ожидания EOF. В CLIP/probe, VLM и archive проходит
ровно один selected apex на каждый фактически представленный source-second. Если
source-second отсутствует, EVA его не выдумывает. JPEG/base64 кодируется только для
выбранного apex, а CLIP embedding не запускается на канале без configured Probe.

### 5.2 Критическое ограничение 1 FPS

При одном входном frame в секунду система честно записывает:

```text
selection_source=single_frame
fallback_reason=single_frame_only_no_intra_second_choice
apex_available=false
```

То есть **1 FPS — deterministic fallback, а не semantic intra-second apex**. На
текущей грузинской раскатке 1 FPS EVA не может увидеть и выбрать кульминацию между
двумя snapshots. Разрешение кадра этого не исправляет. Для настоящего intra-second
apex нужен более плотный low-resolution input и полевой sizing на 50 каналов.

Текущий selector — attention heuristic на frame delta плюс доступные road/CLIP
signals, а не обученный универсальный классификатор драк, дрифта или нарушений. Его
результат определяет, куда эскалировать внимание; визуальное доказательство остаётся
за VLM/frame review и оператором.

### 5.3 Фактический throughput и локальная tuning-конфигурация

Исторические measurements до incremental pipe/token transport показывали:

```text
CH112: 0.60x realtime
CH118: 0.84x realtime
```

После исправления и при browser model-view реальный CH118 (`emu-1`) дал:

```text
60 s browser smoke: 102 raw candidates, 52 selected apex за 50 s между samples
completed windows: 60.0 / 60.688 s = 0.99x
                   57.5 / 60.616 s = 0.95x
```

CH112 остаётся source-specific underfill примерно `35/60 s` (`~0.58x`) даже без
второго recorder stream. Это низкая фактическая cadence данного webcam/MJPEG source,
а не число, которое EVA имеет право дорисовать. UI показывает его как `apex-lag`.
CH118 result — point measurement, **не** capacity claim на 50 камер.

На этой машине в git-ignored `.env` сейчас выбраны site-local значения:

```text
EVOSSEARCH_LUXRIOT_CAPTURE_SOURCE=live_segment
EVOSSEARCH_LUXRIOT_LIVE_SEGMENT_FPS=2
EVOSSEARCH_LUXRIOT_LIVE_SEGMENT_SECONDS=60
EVOSSEARCH_LUXRIOT_SNAPSHOT_MAX_EDGE=640
```

Site-local `capture_source`, FPS и max edge не являются release defaults. Defaults в
коде остаются `capture_source=auto`, dense FPS `3.0` и max edge `800`; bounded
incremental capture lease теперь default `60 s`. Installer при adopt сохраняет site
env. Сравнивать результат можно только вместе с точными effective settings.

Runtime status теперь публикует completed/inflight target seconds, represented
seconds, raw-frame/byte budgets и segment latency. UI показывает represented/target,
отношение к real time и отдельный `apex-lag` warning при underfill или заметном
отставании. Это observability, а не автоматическое доказательство пропускной
способности.

## 6. Shared LM admission и rollups

Agent, VLM и rollup calls, направленные на один OpenAI-compatible base URL, проходят
через один in-process admission resource. Model ID намеренно не разделяет resource:
разные model aliases одного LM Studio/llama.cpp процесса всё равно делят GPU/process.

- Default capacity — 1.
- `EVOSSEARCH_LM_MAX_INFLIGHT` задаёт общий fallback.
- `EVOSSEARCH_LM_PROFILE_<ID>_MAX_INFLIGHT` задаёт profile override.
- Если разные profiles указывают на один endpoint, применяется наиболее
  консервативная capacity.
- Interactive/agent work имеет приоритет, background/rollup ниже; aging не даёт
  background ждать бесконечно.
- `/lm/admission` и Agent sidebar показывают active, queued, workload breakdown,
  oldest queue age и wait counters без URL credentials.

Admission делает contention наблюдаемым и не позволяет случайно атаковать
capacity-one LM несколькими threads. Он не ускоряет модель, не освобождает VRAM и не
координирует несколько EVA processes/hosts.

Rollup endpoint принимает `target_level=L0|L1|L2|L3`; UI запрашивает выбранный level,
а более высокие скрытые levels не синтезируются. Пока request выполняется, UI
показывает aggregation elapsed time и optional shared-LM queue hint, а stale render
отбрасывается generation guard.

Это честный RC compromise, но не завершённый async materializer: нужный rollup всё
ещё может синхронно вызвать LM и занять HTTP thread. `pending_context` означает
недостаточный source window, а не сбой GPU.

## 7. Operator media

Добавлен authenticated same-origin endpoint
`/luxriot/media/<live|archive>/<channel_id>`. Live open использует документированный
Evo `addStreamToken` → `retrieveLiveStreamByToken` server-side flow с secret-safe
fallback на legacy direct Digest path. Upstream URL, credentials и bearer-like token
не попадают в browser. Broker проверяет channel ACL, поддерживает один HTTP byte
Range, сохраняет 200/206/416 semantics и ограничивает stream по времени и bytes.

Для archive broker запрашивает `nextFrameTime`, выравнивает начало на реальный sample,
использует `duration=1&html5compatible=true`, передаёт
`X-Stream-Last-Sample-Timestamp` и даёт UI guarded continuation от `last + 1`.
Snapshot fallback явно маркируется как static/degraded, а не video. UI различает
`loading`, `playing`, `degraded` и `error` и отбрасывает late player/seek responses.

Локальный smoke текущей системы получил HTTP 200 `video/mp4` и для live, и для
archive; archive вернул last-sample header. Это подтверждает текущий endpoint/codec,
но не является codec/topology matrix для всех 50 камер.

Подтверждённая причина прежнего «видео замирает примерно через 30 секунд» была не в
analytics channel и не в VLM: broker намеренно завершает live response по default
30-second safety lease старой конфигурации. Browser получал ожидаемый EOF bounded response; кроме того,
forward upstream `Content-Length` был небезопасен, если lease обрывал response раньше
объявленной длины.

RC делает этот transport contract явным:

- live response отдаёт `X-EVA-Media-Lease-Seconds` и
  `X-EVA-Media-Renew-After-Ms`;
- default live lease теперь 120 seconds / 256 MiB и предлагает renewal через 90 s;
- un-ranged live response больше не обещает upstream `Content-Length`, который не
  успеет доставить целиком до cutoff;
- основной Video preview и Probe preview proactively renew connection до lease EOF;
- `ended`, `waiting` и `stalled` вооружают bounded reconnect/stall watchdog, а
  progress/timeupdate его снимают;
- generation/Abort guards не дают старому reconnect отрисоваться после смены
  channel.

Когда video summaries уже запущены, UI по умолчанию использует
`/luxriot/attention_stream/<channel_id>`: bounded MJPEG из **тех же выбранных apex**,
которые идут в CV/CLIP/VLM/archive. Он не открывает второй Evo stream и поэтому не
отбирает throughput у analytics. Кнопка `Full live` явно открывает smooth recorder
media; `Model view` возвращает shared attention preview. На реальном CH118 model-view
60 секунд оставался `playing` без `degraded/error`, а analytics одновременно держал
`0.95–0.99x`. На CH112 full live был визуально стабилен, но второй stream снизил
dense capture примерно до `0.44x`; поэтому он больше не является безопасным default.

Player lifecycle отделён от EVA analytics lifecycle: broker использует только
authenticated `GET/HEAD`, UI reconnect не вызывает start/stop summaries или probes,
а logout regression не останавливает background captures. В live check
desired/running analytics state оставался неизменным при player renewal. Это
подтверждает независимость preview transport от analytics state, но не обещает
нулевого visual gap между двумя bounded browser connections.

### 7.1 Граница обещания

Broker работает **в том же single-worker Gunicorn process**; default thread count
увеличен до 8, чтобы bounded media не вытеснял Agent/status traffic. Defaults
ограничивают live request 120 секундами/256 MiB, archive request 45 секундами/128
MiB. Это bounded operator path, **не 50-player media plane**, не async media sidecar
и не universal remux/transcode service. Shared attention preview снимает contention
для model-view, но `Full live` всё ещё расходует отдельный upstream.

В текущем tree отсутствует каталог `docs/http-api-guide-2026-07-10`, поэтому он не
может быть самостоятельным release artifact этой ветки. Endpoint contract был
сверен с восстановленным guide и фактическим Evo smoke: CH112 tokenized MJPEG first
chunk `~0.12 s`, CH118 tokenized MP4 `~0.05 s`. Owner-recorder routing и native
Luxriot Monitor switching всё ещё не реализованы и **не обещаются**; multi-recorder
site требует отдельного routing evidence.

## 8. Offline installer

`scripts/install_eva_083.py` — dry-run-first orchestrator для existing/adopt и
ограниченного fresh app-host режима. Он:

- обнаруживает и сохраняет существующий env, неизвестные keys, comments и secrets;
- спрашивает недостающие Evo/PostgreSQL/Agent/VLM параметры или fail-closed в
  non-interactive mode;
- не выполняет `git`, `apt` или online `pip`; fresh venv требует offline wheelhouse;
- создаёт/reuses service account, directories, venv и systemd unit;
- устанавливает dependencies только с `--no-index`;
- переиспользует `preflight_patch.sh`, `install_patch.sh`, Alembic,
  `verify_patch.sh` и `rollback.sh`;
- требует отдельный privileged migration DSN и никогда не подменяет его runtime
  `EVA_DATABASE_DSN`;
- требует non-empty `postgres.dump` до migration;
- держит nonblocking `flock` от apply до verify/handoff;
- передаёт точный env path и в `EnvironmentFile`, и в
  `EVOSSEARCH_CONFIG_ENV_FILE`;
- по умолчанию только строит plan; mutation требует явного `--apply` и root.

### 8.1 Текущее fail-closed состояние на этом host

Реальный dry-run 2026-07-11 завершился `BLOCKED` без изменений по трём причинам:

1. не задан отдельный `EVA_INSTALL_MIGRATION_DSN` или
   `EVA_MIGRATION_DATABASE_DSN`;
2. на host отсутствует `pg_dump`;
3. `EVOSSEARCH_LUXRIOT_PASSWORD` распознан как placeholder (значение не выводилось).

Это ожидаемый safety result. До устранения всех трёх причин `--apply` запускать
нельзя.

### 8.2 Что installer пока не обещает

- установку и hardening PostgreSQL server/roles/password policy;
- автоматическое создание tenant/admin;
- OS packages, CUDA/driver, TLS/reverse proxy;
- поставку всех CLIP/VLM/LLM model assets;
- signed/hash-locked platform bundle;
- полностью автоматический destructive DB rollback;
- доказанный fresh offline apply/power-loss recovery на disposable VM.

Это installer RC поверх существующего patch/rollback контура, а не готовый generic
appliance installer для произвольного Ubuntu host.

## 9. Проверки и evidence

Полный локальный test run на финальном working tree:

```text
513 passed, 23 skipped, 132 subtests passed
```

Warnings относятся к deprecated `pkg_resources` в dependency `clip` и Python 3.14
warning для `torch.jit.load`; test failures нет.

Важные regression contracts включают:

- partial/delta channel inventory, stale cache и URL credential redaction;
- exact time bounds, Unicode titles, period/event sampling, search coverage и
  persisted continuation ledger;
- explicit false/reset/source/revision/persistence failure и restart recovery;
- desired-state serialization и stale generation без side effects;
- dense `live_segment` budget/ffmpeg pipe, один selected apex на представленный
  source-second, одинаковый apex hash в CLIP/VLM/archive и отдельный 1 FPS fallback;
- LM capacity/priority/timeout/status и generation preflight после admission;
- targeted rollup без hidden higher levels и UI progress generation guard;
- media lease/renewal/watchdog, Range, codec sniffing, archive gap/sample
  continuation, analytics independence, ACL и credential isolation;
- UI completed/inflight dense-capture diagnostics и `apex-lag` observability;
- installer env adoption, privileged migration identity, placeholder/pg_dump gates,
  lock, dry-run immutability и rollback handoff.

Отдельный authenticated live Agent acceptance дал
`2 passed, 1 skipped, 3 subtests passed`: scope conservation и
exact-window/evidence bounds пройдены, continuation
skipped из-за scope в три канала. Следовательно, локальные `513 passed` плюс этот
acceptance всё равно нельзя интерпретировать как пройденный production Agent/VLM
soak или как real-time throughput доказательство.

## 10. Разрешённые обещания для пилота

Можно обещать:

- attention escalation по событиям, продолжающимся достаточно долго, чтобы попасть
  в фактическую sampling cadence;
- channel-scoped live descriptions, alerts, probes, archive search и reports с
  явным coverage;
- multi-turn review, который сохраняет frozen период и показывает deferred/unchecked
  scope;
- детерминированный один attention frame на секунду с provenance при наличии более
  плотного входа;
- operator media для подтверждённых browser-compatible Luxriot responses с честным
  degraded fallback;
- контролируемый offline adopt/update workflow после чистого dry-run.

Нельзя обещать:

- profiling/identification людей;
- frame-perfect распознавание мгновенных событий при 1 FPS;
- отсутствие VLM/CLIP false positives или визуальное доказательство без frame review;
- полное покрытие периода, если API вернул `partial/truncated/no_data/unchecked`;
- 50 одновременных browser players через текущий Gunicorn broker;
- поддержку любого codec за счёт EVA transcoding;
- owner-recorder routing или remote/native Luxriot Monitor switching;
- unattended schema-changing installation до успешного site dry-run, backup и
  disposable-host acceptance.

## 11. Оставшиеся RC gates

1. Зафиксировать working tree в reviewable commit/tag и собрать immutable offline
   bundle.
2. Сохранить уже пройденный authenticated frozen-window result как redacted release
   evidence и отдельно прогнать randomized Agent/VLM soak 20–30 минут на трёх
   streams, не меняя auth.
3. Проверить invariants: requested scope conserved, exact window не дрейфует,
   `checked + unchecked/errors` объясняет scope, desired/running set стабилен, SSE и
   tool errors не молчат.
4. Прогнать lease renewal/stall recovery и live/archive player по нескольким
   реальным каналам и codec variants; отдельно зафиксировать, что broker bounded и
   не используется как 50-player wall.
5. После site-local `live_segment`/2 FPS/640 downscale повторно измерить represented
   seconds / latency на CH112 и CH118 в steady state. До результата не обещать
   real-time; на 1 FPS показывать fallback provenance и не называть его
   intra-second apex.
6. Устранить три installer blockers, повторить dry-run, затем выполнить upgrade и
   rollback rehearsal на disposable VM до клиентской schema migration.
7. Не увеличивать Gunicorn workers, не обещать owner-recorder/native Monitor и не
   включать новые внешние контуры без отдельного evidence/acceptance.
