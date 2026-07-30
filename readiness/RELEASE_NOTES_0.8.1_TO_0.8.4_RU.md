# EVA AI β 0.8.1 → β 0.8.4: изменения после грузинского пилота

Дата среза: 2026-07-14  
Пилотная база: `β 0.8.1`, commit `9ed94ce` от 2026-06-24  
Текущий release candidate: `β 0.8.4`, commit `7d0fd86` от 2026-07-13  
Schema head: `20260614_0006`  
Миграция БД: **нет**

## Короткий вывод

Между грузинским пилотом и текущей `0.8.4` произошёл большой функциональный
скачок при сравнительно консервативном контуре обновления.

`0.8.1` была стабилизацией уже существующей 50-канальной пилотной основы:
PostgreSQL/RLS, архив кадров, L0–L3 summaries, защищённые agent tools и базовые
video-description-first расследования уже существовали. В `0.8.2–0.8.4`
поверх этой основы появились:

- наблюдаемая структурированная VLM-alert цепочка;
- периодные и многоканальные расследования агента;
- архивная калибровка CLIP-проб;
- живой video ingest с честной диагностикой потери/заморозки сигнала;
- road-motion и channel-relative attention слой;
- полноценный media broker и evidence-first архивный review;
- durable semantic rollups и управляемое восстановление истории;
- bounded backpressure/admission для VLM и agent LM;
- офлайн-обновление с флешки, backup, automatic rollback и bundled media
  runtime.

Поэтому корректная оценка масштаба: **функционально это почти новый продуктовый
слой, но по данным и установочной топологии — совместимый code-only upgrade**.

## Масштаб изменения

Сравнение `9ed94ce..7d0fd86`:

| Метрика | Значение |
|---|---:|
| Коммиты после пилотной базы | 68 |
| Затронутые файлы | 158 |
| Добавлено строк | 58 084 |
| Удалено строк | 3 332 |
| Период разработки | 2026-06-25 — 2026-07-13 |

Распределение diff:

| Слой | Файлы | Добавлено | Удалено |
|---|---:|---:|---:|
| Runtime/backend | 33 | 21 907 | 1 453 |
| UI | 3 | 6 858 | 678 |
| Installer/operations scripts | 20 | 4 985 | 15 |
| Tests | 34 | 14 652 | 682 |
| Docs/readiness | 68 | 9 682 | 504 |

Цифры нельзя читать как линейный рост продукта: в них много тестовых контрактов,
acceptance-сценариев и полевых инструкций. Однако и без них runtime+UI составляют
примерно 28,8 тысячи новых строк, прежде всего в `luxriot_connector.py`,
`agent.py`, `oldapp.py` и операторском интерфейсе.

## Эволюция по релизам

### β 0.8.2 — VLM alerts, отчёты и безопасная калибровка

Основная цель: превратить office-demo поток video descriptions в наблюдаемую и
проверяемую operational цепочку.

- Разделены L0 system prompt, channel alert policy и JSON alert contract.
- Добавлены structured alert parsing, delivery status и parser diagnostics.
- Появились backend state transitions с debounce/hysteresis.
- Агент начал разделять incident findings и здоровье pipeline.
- Runtime-status запросы направляются в живые runtime tools, а не в статическую
  документацию.
- Введён порядок доверия к evidence: память → L0 prose → structured events →
  backend transitions → кадр и `describe_frame`.
- CLIP P/N/M calibration стала серверным workflow с `safe_to_apply`,
  `recommended_action`, warnings и запретом apply-ready результата при слабой
  или over-firing калибровке.
- Появились deterministic live smoke, seeded fixtures и admin/non-admin
  acceptance paths.

Практический эффект: алерт перестал быть просто строкой в VLM-прозе, а агент
получил возможность объяснять происхождение сигнала и честно отделять событие
от проблем самого pipeline.

### β 0.8.2.1 — evidence/UI и периодные расследования

Основная цель: убрать ложные ответы и операторскую путаницу, найденные при
ручном тестировании.

- Исправлен production channel inventory contract; агент перестал ложно
  сообщать `Luxriot not connected` из-за отсутствия test-only атрибута.
- Периодные расследования запрещено строить только по последнему summary или
  последней archive page.
- Добавлены rolling seven-day ranges и period-wide evidence sampling.
- При недоступном live inventory отчёт может честно использовать локальную
  summary/runtime history с явным `archive_fallback`.
- Probe mutation previews/receipts вынесены в отдельные approval cards;
  Apply остаётся отдельным серверным действием.
- Machine blocks получили provenance labels.
- Metadata-only detection больше не изображает из себя визуальную улику;
  отсутствующая картинка показывается как `No image`.

Практический эффект: агент стал заметно честнее по coverage, а UI перестал
создавать впечатление, что несуществующий thumbnail является evidence.

### β 0.8.3 — live-signal и road-event perception foundation

Основная цель: добавить лёгкий CV attention слой и перестать маскировать
проблемы живого видеосигнала старыми кадрами.

- Создан пакет `road_events/`: scene cards, motion zones, optical-flow/frame
  motion cues, auto scene bootstrap и episode candidates.
- CLIP/probe и road-CV сигналы могут попадать в L0 через bounded
  `VECTOR_SIGNALS_JSON` как attention cues, но не как доказательство.
- Live capture поддерживает Luxriot live segments и snapshot→segment failover.
- Медленный VLM больше не блокирует acquisition loop: используется bounded
  latest-wins queue.
- Добавлены короткие ingest timeouts, backoff и queue/drop observability.
- Fresh/stale/frozen signal contract стал явным. UI показывает `Signal lost` и
  `Signal frozen`, а не бесконечно повторяет последний успешный кадр.
- Агент видит runtime-problem channels, включая каналы без summaries в окне.
- Engineer/admin UI получил диагностический road grounding overlay.
- Появились offline USB runbooks, физическая топология клиента и cumulative
  acceptance сценарии.

Практический эффект: EVA стала не только анализировать уже полученные данные,
но и честно описывать качество/свежесть входного сигнала и выдавать bounded
motion-кандидаты для дальнейшей VLM/человеческой проверки.

### β 0.8.4 — attention, media, durable history и полевое обновление

Основная цель: сделать live/archive работу отзывчивой под нагрузкой, а выпуск —
реально переносимым на офлайн-клиента.

#### Capture attention

- Per-second apex decider классифицирует активность как quiet/normal/burst
  относительно сохранённой нормы конкретного канала.
- Frame selection учитывает motion и sharpness; режимы оператора:
  `auto`, `action`, `clarity`.
- Для burst сохраняется action frame и более резкий companion/anchor frame.
- `capture_attention` передаётся в `VECTOR_SIGNALS_JSON` и доступен агенту.
- `list_attention_bursts` даёт bounded список сильнейших всплесков без полного
  fan-out по summaries.
- Burst означает статистически необычное движение, а не семантический инцидент;
  вывод подтверждается кадрами/VLM.

#### Media и операторский UI

- Добавлен same-origin tokenized live/archive media broker с lease renewal,
  timeout/stall watchdog и bounded archive duration.
- `Model view` использует общий EVA attention stream и автоматически
  восстанавливается после краткого stall.
- Archive review стал evidence-first: сначала сохранённый кадр и filmstrip,
  video playback запускается явно оператором.
- Исправлена browser-playable archive video ветка и fallback на evidence frame.
- Рабочая область и history navigation переработаны под вертикальный review;
  появились burst badges и роли кадров в filmstrip.

#### Агент

- Persisted research continuation не даёт модели самовольно продолжать широкий
  fan-out без явного намерения оператора.
- Composite channel inventory объединяет live runtime и сохранённую историю.
- Tool schemas выдаются по intent: нерелевантные инструменты больше не занимают
  контекст и не провоцируют случайный workflow.
- Video-summary tool boundary сохраняет semantic entries и `image_url`; длинная
  coverage metadata больше не съедает evidence preview.
- Tool results компактируются перед моделью с сохранением coverage/evidence.
- Если локальная модель вернула пустой ответ или только обещание продолжить,
  агент формирует evidence-only completion из уже выполненных trusted tools.
- Для Qwen3.5 отключено thinking и в tool loop, и в final response.
- Контекст агента поднят с 32 768 до 65 536; history budget — 16k, warning —
  52k, hard stop — 60k.
- Updater показывает фактический `n_ctx`. При коротком LM context оператор может
  явно выбрать `FORCE-CONTEXT`; EVA временно ограничивается реально доступным
  значением. Отказ происходит до остановки сервиса.

#### Semantic rollups и нагрузка

- L1–L3 operator narratives отделены от machine homeostasis.
- Scheduled semantic rollups сохраняются durable и используют отдельный agent
  LM profile с thinking disabled.
- Genuine legacy LM rollups из `0.8.0/0.8.1` принимаются как
  `legacy_cached` и раскладываются в durable rows без повторной генерации.
- Mechanical fallback strings семантикой не считаются.
- Restore worker может после явного approval восстановить недостающие L2/L3
  окна из архивного L0; progress переживает restart и отличает queueable work
  от настоящих source gaps.
- L0 backpressure coalesces окна вместо молчаливого drop; gaps становятся
  видимыми.
- LM admission и scheduler учитывают конкретный LM resource: busy VLM не должен
  без причины останавливать отдельный agent GPU.

#### Field upgrade

- Финальный bundle запускается через корневой `./update.sh` и различает user и
  system systemd.
- Проверяются version/commit/clean manifest, `.venv`, неизменность requirements,
  schema head, media payload, LM context и `/ready`.
- До остановки production service проверяется `sudo`.
- Создаётся code/env backup; при post-stop failure выполняется automatic
  rollback.
- Post-start проверка парсит JSON, прогревает lazy embedder через
  `/ready?load=1` и принимает успех только при `status=ready` и версии `0.8.4`.
- В офлайн bundle включены Linux x86_64 FFmpeg/ffprobe и совместимый OpenCV
  rescue wheel. Python requirements и модели не дублируются.
- Репетиционный bundle `7d0fd86` имеет размер около 151 MB и был установлен на
  dev-конфиг полным one-command маршрутом.

## Что оператор заметит сразу

1. **Живое видео честнее.** Старый/замороженный кадр больше не выглядит как
   продолжающийся live feed.
2. **Архив стал исследовательским workspace.** Есть evidence frame, filmstrip,
   роли кадров и явный запуск видео.
3. **Агент лучше держит период.** Он обязан показывать coverage и gaps, а не
   делать вывод по последнему кадру.
4. **Отчёт показывает здоровье pipeline отдельно от событий.** Видны queue,
   drops, errors, parser/delivery и signal state.
5. **Всплески активности находятся без полного перебора истории.** Но burst
   остаётся attention marker, а не готовым алертом.
6. **Изменения probes/prompts остаются approval-gated.** Chat готовит preview,
   оператор отдельно применяет действие.
7. **Обновление стало полевым продуктом.** Проверка, backup, restart, readiness
   и rollback собраны в одном сценарии.

## Что принципиально не изменилось

- Schema head остаётся `20260614_0006`; Alembic migration не требуется.
- `requirements.txt` и `requirements-db.txt` между `0.8.1` и `0.8.4` не
  изменились; существующий здоровый `.venv` переиспользуется.
- PostgreSQL, архивные данные, runtime state и site-specific `.env` не должны
  удаляться или пересоздаваться adopt-upgrade маршрутом.
- Существующий hardened systemd unit в adopt mode сохраняется.
- Модели, CUDA, LM Studio/vLLM/llama.cpp и их service topology bundle не
  переустанавливает.
- Road-CV, CLIP и capture-attention сигналы остаются candidate/attention
  evidence, а не юридически значимым определением события.

## Совместимость старых данных

- Существующие L0 VLM summaries и evidence frames остаются источником истории.
- Настоящие старые L1–L3 LM rollups могут быть приняты как legacy semantics без
  повторного вызова модели.
- Если старое retention-окно уже физически удалило summaries, upgrade не может
  восстановить никогда не сохранённый источник.
- Restore semantic history — отдельное operator-approved действие, а не скрытая
  часть установки.
- Adopt-upgrade выполняет read-only schema gate и не трогает БД.

## Оставшиеся ограничения и риски

1. **Короткий context LM Studio.** Рекомендуется 65 536. Explicit force на
   меньшем контексте поддержан, но длинные многоканальные расследования будут
   менее устойчивыми до перенастройки LM Studio.
2. **In-memory L0 queue.** Если PostgreSQL inference queue не включена, pending
   L0 batches теряются при process restart. Coalescing и gap accounting делают
   потерю bounded/видимой, но не durable.
3. **Archive playback зависит от Luxriot archive coverage.** При upstream gap
   модалка честно остаётся на stored evidence frame.
4. **Road semantics ограничены.** Короткие события и wrong-way без устойчивой
   геометрии сцены нельзя интерпретировать уверенно.
5. **Inference runtime остаётся site responsibility.** Bundle приносит media
   runtime, но не GPU drivers, модели или универсальную конфигурацию LM Studio.
6. **Широкий agent report всё ещё дорог.** Intent routing, 65k context и
   compaction сильно снижают хрупкость, но coverage нескольких десятков каналов
   должен выполняться bounded chunks, а не одним неограниченным fan-out.

## Проверка и evidence

- `0.8.2.1` release snapshot: 360 tests passed, 19 skipped, 127 subtests passed.
- `0.8.4` predeploy audit до последних field hotfixes: 626 passed, 18 skipped,
  134 subtests passed.
- Финальный agent/updater hotfix `7d0fd86`: targeted suite 141/141 passed.
- Живая dev-репетиция финального bundle:
  - EVA service active;
  - `/ready` = `ready`, version `β 0.8.4`;
  - agent LM active, reported `n_ctx=65536`;
  - bundled FFmpeg/OpenCV smoke passed;
  - database schema уже `20260614_0006` и не изменялась;
  - updater завершился `OK: EVA AI β 0.8.4 is up and running`.

Эти snapshots относятся к разным этапам и не суммируются в одну цифру.

## Итоговая оценка

С точки зрения пользователя и архитектора разница между грузинским `0.8.1` и
текущим `0.8.4` большая:

- агент превратился из набора защищённых инструментов в bounded investigation
  runtime с coverage, continuation, evidence и context budgeting;
- video pipeline получил собственный attention/perception слой и честную
  диагностику живого сигнала;
- UI превратился из monitor/demo поверхности в evidence review workspace;
- background summaries стали durable и восстанавливаемыми;
- deployment превратился из инженерного patching runbook в переносимый offline
  upgrade с rollback.

При этом сохранены главные свойства безопасного пилотного апгрейда: та же схема
БД, те же Python dependencies, сохранение site config и данных, отсутствие
скрытой перегенерации истории и возможность вернуться к code/env backup.
