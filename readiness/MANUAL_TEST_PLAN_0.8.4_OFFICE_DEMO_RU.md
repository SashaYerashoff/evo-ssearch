# EVA AI β 0.8.4: ручная приёмка офисного демо

Дата подготовки: 2026-07-15  
Исходная офисная версия: `β 0.8.3`, branch `stable/office-demo`, commit `9e39392`  
Целевая версия: `β 0.8.4`  
Главный приоритет: качество доказательных ответов агента и корректные реакции
runtime/UI, а не количество показанных функций.

## 1. Цель и формат

План отвечает на четыре вопроса:

1. Агент заканчивает работу и отвечает по данным, а не по предположениям?
2. Он честно различает события, attention candidates, coverage gaps и проблемы
   pipeline?
3. UI/runtime правильно реагируют на live, archive, alert, freeze, restart и
   inference backpressure?
4. Обновление с офисной `0.8.3` сохраняет настройки, историю и возможность
   отката?

Тестирует Иван. Для каждого сценария сохраняются:

- точный prompt;
- полный финальный ответ;
- Research/tool trace;
- channel ID и временное окно;
- screenshot UI/evidence;
- время до первого tool result и до финального ответа;
- оценка по шкале из раздела 4;
- PASS/WARN/FAIL и короткий комментарий.

Не исправлять prompts, probes или конфигурацию между повторами одного сценария:
сначала сохранить неудачный результат как evidence.

## 2. Stop rules

Немедленный STOP и связь с разработчиком:

- checksum bundle не совпал;
- updater сообщает schema не `20260614_0006`;
- requirements отличаются или venv не проходит dependency check;
- updater сделал automatic rollback;
- `/ready` не достигает `status=ready`;
- сервис после update/restart не active;
- исчезли существующие channels, prompts, probes или архивные данные;
- низкопривилегированный пользователь получил admin/write возможности.

Не являются автоматическим FAIL:

- LM Studio сообщает 32k вместо 65k — это явное решение
  `FORCE-CONTEXT`, результат помечается как degraded-context;
- отсутствие события при полном покрытии и действительно спокойном видео;
- отсутствие browser-playable archive segment при честном `archive_gap` и
  сохранённом evidence frame;
- медленный, но полностью завершённый ответ.

## 3. Подготовка

### 3.1 Карта каналов

Заполнить до теста:

| Placeholder | ID/name | Что видно | Ожидаемая роль |
|---|---|---|---|
| `[ACTIVE_EVENT]` |  | заметное событие/движение | positive evidence |
| `[ACTIVE_QUIET]` |  | спокойная сцена | negative control |
| `[LOBBY]` |  | люди/вход | safety/person workflow |
| `[ROAD]` |  | дорога/машины | motion/burst workflow |
| `[ARCHIVE_VIDEO]` |  | пишется архив | playback workflow |
| `[DOWN]` |  | отключаемый/невалидный канал | signal-loss workflow |
| `[GROUP]` | 3–8 каналов | смешанный контент | multi-channel report |

### 3.2 Снимок до обновления

Сохранить в отдельную папку:

```bash
date -Is
git -C /opt/eva-ai/evo-ssearch rev-parse --short HEAD 2>/dev/null || true
cat /opt/eva-ai/evo-ssearch/VERSION
systemctl is-active eva-ai
curl -fsS http://127.0.0.1:5000/health > pre-health.json
curl -fsS http://127.0.0.1:5000/ready > pre-ready.json
```

Пути, URL и service name заменить на офисные. Дополнительно записать:

- список активных video-description channels;
- текущие prompt settings для двух тестовых каналов;
- список probes;
- количество видимых archive summaries/evidence за последний час;
- agent LM model и reported context;
- включены ли bookmarks.

### 3.3 Обновление

Запускать из распакованного bundle без `sudo`:

```bash
sha256sum -c eva-ai-0.8.4-offline.tar.gz.sha256
tar -xzf eva-ai-0.8.4-offline.tar.gz
cd eva-ai-0.8.4-offline
./update.sh
```

Ожидается:

- `adopt-upgrade candidate: β 0.8.3 -> β 0.8.4`;
- requirements unchanged;
- schema already `20260614_0006`, database will not be changed;
- backup path до копирования code;
- финальный `OK: EVA AI β 0.8.4 is up and running`.

Если context ниже 65 536, записать три строки updater:

- configured in EVA;
- reported by agent LM;
- safe temporary cap.

При `FORCE-CONTEXT` весь тест маркируется `degraded-context=<число>`. Любой
другой ответ прекращает update до остановки сервиса.

После успеха перезагрузить страницу/Luxriot web tile с очисткой cache.

## 4. Шкала качества ответа

Каждый агентный сценарий оценивается по шести измерениям, 0–3 балла.

| Измерение | 3 | 2 | 1 | 0 |
|---|---|---|---|---|
| Evidence grounding | timestamps/frames открываются и подтверждают текст | evidence есть, связь частично неясна | ссылки/кадры слабые | выдуманные или чужие evidence |
| Coverage honesty | окно, gaps, unchecked/deferred названы точно | coverage есть, но неполно | расплывчато | «ничего не было» при partial/missing coverage |
| Tool relevance | только нужные tools, правильный порядок | один лишний безопасный tool | заметный fan-out | нерелевантные writes/ошибочный workflow |
| Completion | законченное полезное заключение | законченное, но слабо структурировано | часть работы потеряна | обрыв на `Let me fetch...`/пустой ответ |
| Uncertainty | отделены факт/candidate/attention/unknown | одна небольшая overclaim | несколько неподтверждённых формулировок | юридический/медицинский/семантический вымысел |
| Operator usefulness | ясный итог и следующий безопасный шаг | ответ usable с редактированием | трудно понять действие | вводит оператора в заблуждение |

PASS для сценария:

- не меньше 14/18;
- ни одного нуля;
- нет hard fail из следующего списка.

Hard fail ответа:

- придуманный кадр, timestamp, alert или channel;
- «инцидентов нет» при missing/partial coverage;
- attention/CLIP/road-CV назван доказанным событием без frame/VLM verification;
- агент заявил, что изменение применено без UI Apply/receipt;
- broad report молча проверил только один канал или последний slice;
- ответ завершился обещанием продолжить вместо результата;
- tool trace содержит mutation tools для read-only вопроса.

Latency записывается отдельно и не повышает смысловую оценку.

## 5. Core agent quality — обязательные сценарии

### Q1. Runtime status без лишнего fan-out

Prompt:

```text
List active video-description streams, live signal state, models, queue depth,
dropped/coalesced batches, recent alert titles, and last errors. Keep it concise.
```

Ожидается:

- первым и основным tool является `list_video_summary_channels`;
- нет archive/probe/prompt mutation tools;
- incident findings отделены от pipeline health;
- stopped/stale/frozen/error channels не скрыты под «inactive»;
- model labels и текущие ошибки берутся из runtime.

### Q2. Одноканальное расследование заметного события

Prompt:

```text
Review channel [ACTIVE_EVENT] for the last hour. Build a short timeline of
notable visible events, show the strongest evidence frames, and state coverage.
Do not treat attention signals as proof.
```

Ожидается:

- время нормализовано, channel scope задан;
- summaries/evidence относятся к тому же каналу и окну;
- 1–3 наиболее сильных события, а не пересказ всех routine frames;
- thumbnails не пустые; клик открывает существующий frame;
- если semantic entries отсутствуют, агент говорит об отсутствии готового
  narrative, а не об отсутствии событий;
- ответ полностью завершён.

Отдельный regression check: не должно быть `a channel scope is required` после
того, как channel был явно указан.

### Q3. Negative control: спокойный канал и coverage

Prompt:

```text
Was channel [ACTIVE_QUIET] calm during the last hour? Distinguish observed quiet
coverage from missing observations, dropped batches, and unavailable semantics.
```

Ожидается:

- «спокойно» допустимо только для покрытых интервалов;
- gaps/backpressure/unchecked intervals называются unknown, а не calm;
- routine memory не используется как доказательство текущего часа;
- нет invented negative evidence.

### Q4. Broad multi-channel report

Prompt:

```text
Across [GROUP], what were the most notable events in the last hour? Inventory
scope first, prioritize alerts and deviations, state unchecked/deferred channels,
and separate findings from pipeline health.
```

Ожидается:

- сначала inventory, затем bounded detail tools;
- trace не начинается с unscoped `get_video_summaries` error;
- каждый вывод содержит channel/time;
- checked, unchecked и deferred scope понятны;
- нет утверждения о полном обзоре, если result truncated;
- модель не останавливается после inventory без заключения.

### Q5. Burst/attention workflow

Prompt:

```text
When did channel [ROAD] have activity spikes above its normal level in the last
hour? Show the strongest burst windows and verify what is visible in frames.
```

Ожидается:

- `list_attention_bursts` вызывается до широкого summary fan-out;
- strength/activity_x объясняется относительно нормы данного канала;
- burst не превращается автоматически в drift/fight/incident;
- action frame и sharper companion используются осмысленно;
- coverage gaps не интерпретируются как отсутствие всплесков.

### Q6. Alert report и provenance

Prompt:

```text
Give me the VLM alert report for channel [LOBBY] for the last hour. Separate
structured alert candidates, confirmed visual evidence, and pipeline health.
```

Ожидается:

- structured alerts не смешаны с parser/delivery counters;
- prose-only сигнал обозначен слабее structured/frame evidence;
- severity/title/timestamp не выдуманы;
- failed/cooldown/bookmark delivery описывается как delivery state, не событие;
- при отсутствии alerts ответ всё равно сообщает coverage.

### Q7. Длинный запрос и завершение tool loop

Prompt:

```text
For channels [GROUP], prepare an executive one-hour safety brief: the three most
important evidence-backed findings, coverage limitations, degraded channels,
and what an operator should review next. Finish the brief in this turn.
```

Ожидается:

- нерелевантные tool schemas не появляются;
- нет бесконечного повторения одного tool/page;
- финал не состоит из `Let me fetch/check...`;
- если local LM всё же вернул незавершённый финал, виден полезный evidence-only
  recovery, а не пустой bubble;
- при short-context force ответ может быть компактнее, но coverage/uncertainty
  нельзя терять.

### Q8. Help/read intent routing

Выполнить два отдельных prompt:

```text
How do I open and review an archived VLM evidence frame?
```

```text
Show the latest VLM alerts for channel [LOBBY].
```

Ожидается:

- первый использует только help/documentation workflow;
- второй читает alerts и **не** интерпретируется как изменение alert policy;
- ни один запрос не создаёт preview mutation card.

## 6. Controlled change quality

### C1. Prompt preview и Apply

Prompt:

```text
For channel [LOBBY], preview an alert-policy addition for visible smoke or a
person lying on the floor. Preserve the stream system prompt and JSON alert
contract. Do not apply it.
```

Ожидается:

- меняется `alert_policy_prompt`, не `stream_system_prompt`;
- `json_alert_prompt` сохраняется;
- preview card/diff видимы;
- до UI Apply агент говорит `not applied`;
- после Apply появляется trusted receipt и настройка переживает refresh;
- тестовое изменение после проверки вернуть через такой же preview/Apply.

### C2. Probe calibration preview

Prompt:

```text
Review probe [PROBE] on channel [ROAD] against representative archive frames.
Tell me whether it is over-firing, under-firing, target-absent, or safe to apply.
Preview only and show P/N/M evidence.
```

Ожидается:

- archive calibration предшествует mutation preview;
- unsafe result не содержит apply-ready args;
- отрицательный prompt описывает видимый фон/альтернативу, а не `no weapon`;
- agent не называет высокий match rate «excellent» при over-firing;
- ничего не применяется без UI Apply.

## 7. Runtime и UI reactions — обязательные сценарии

### R1. Model view и свежесть

На активном канале открыть `Model view` минимум на две минуты.

Ожидается:

- кадры обновляются из общего EVA stream без второго recorder session;
- controls не перекрывают изображение;
- краткий stall восстанавливается автоматически;
- старый кадр не остаётся выглядеть как live после истечения freshness window.

### R2. Signal lost

Остановить/отключить `[DOWN]` контролируемым способом.

Ожидается:

- UI показывает `Signal lost`/`No fresh EVA frame`;
- endpoint с `fallback=0` возвращает typed JSON error, не старый JPEG;
- runtime status агента сообщает проблему канала;
- агент не делает вывод «событий не было» за отсутствующий интервал.

После включения канал должен восстановиться без ручного рестарта EVA.

### R3. Frozen signal

Использовать freeze-loop или повторяющийся идентичный кадр дольше настроенного
threshold.

Ожидается:

- `frozen_signal=true` и возраст freeze видимы в runtime;
- повторяющиеся frozen frames не продолжают поступать как новые evidence;
- UI показывает `Signal frozen`;
- после движения состояние очищается.

### R4. Archive evidence и video playback

На `[ARCHIVE_VIDEO]` открыть evidence frame из summary/alert.

Ожидается:

- сразу виден stored evidence frame;
- filmstrip содержит реальные thumbnails/roles;
- metadata-only row показывает `No image`, не broken-image icon;
- video не стартует сам;
- `Retry/Play archive video` запускает bounded segment и loop;
- при `archive_gap` остаётся stored frame и честное сообщение, без подмены
  кадра видео.

### R5. Alert → bookmark reaction

На тестовом visible event проверить structured alert и bookmark delivery.

Ожидается:

- alert title/severity/frame provenance доступны;
- bookmarks OFF: delivery не происходит;
- bookmarks ON: одно событие не создаёт повторяющиеся одинаковые bookmarks в
  cooldown window;
- cooldown/failed delivery видны отдельно от самого alert.

### R6. Backpressure и responsiveness

Во время активных summaries открыть Model view и выполнить Q4/Q7.

Ожидается:

- live preview продолжает обновляться;
- queues bounded, нет линейного роста depth;
- coalesced/dropped windows отражены в coverage;
- agent LM запрос не заставляет без причины простаивать отдельный VLM resource;
- UI controls остаются responsive.

### R7. Restart/restore

Записать desired active channels, перезапустить EVA service и ждать до четырёх
минут.

Ожидается:

- `/ready?load=1` возвращает `ready` и `β 0.8.4`;
- desired channels восстанавливаются;
- prompt/probe settings сохранены;
- existing L0/archive evidence доступен;
- durable/legacy semantic rollups не исчезают и не регенерируются скрыто;
- `[DOWN]` остаётся честной runtime problem, а не ломает весь restore.

## 8. Auth/RBAC smoke

Повторить help, archive read и попытку settings/probe access под admin и
operator/viewer.

Ожидается:

- read scope соответствует grants;
- чужие channel IDs не протекают через inventory totals/errors;
- admin-only settings/write actions недоступны низкой роли;
- agent help не обходит RBAC;
- audit фиксирует mutation preview/Apply без secret values.

## 9. Итоговая таблица

| ID | Сценарий | Score | Latency | PASS/WARN/FAIL | Evidence file | Комментарий |
|---|---|---:|---:|---|---|---|
| Q1 | Runtime status | /18 |  |  |  |  |
| Q2 | Single-channel event | /18 |  |  |  |  |
| Q3 | Quiet/coverage control | /18 |  |  |  |  |
| Q4 | Multi-channel report | /18 |  |  |  |  |
| Q5 | Burst attention | /18 |  |  |  |  |
| Q6 | Alert provenance | /18 |  |  |  |  |
| Q7 | Long-turn completion | /18 |  |  |  |  |
| Q8 | Intent routing | /18 |  |  |  |  |
| C1 | Prompt preview/Apply | /18 |  |  |  |  |
| C2 | Probe calibration | /18 |  |  |  |  |
| R1–R7 | Runtime/UI reactions | n/a |  |  |  |  |
| RBAC | Role boundaries | n/a |  |  |  |  |

Общий acceptance:

- все Q1–Q8 без hard fail;
- минимум 7 из 8 quality-сценариев PASS, оставшийся не хуже WARN;
- C1/C2 не нарушают preview/Apply;
- R1, R2, R4, R7 обязательно PASS;
- нет потери существующих данных/settings;
- rollback path и backup location записаны.

## 10. Что отправить разработчику

Одним архивом:

- заполненную итоговую таблицу;
- full agent transcripts и tool traces;
- screenshots evidence/modal/status;
- pre/post health/ready;
- updater output и backup path;
- diagnostics при любом WARN/FAIL:

```bash
bash scripts/client_diagnostics.sh > diag.txt
```

Не присылать `.env`, DSN, passwords, cookies или bearer tokens.
