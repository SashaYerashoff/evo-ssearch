# EVA AI β 0.8.4: полевой adopt-upgrade — чек-лист и результаты репетиции

Дата репетиции: 2026-07-12. Инженер несёт флешку; этот документ — его маршрут.

## Что было отрепетировано (локально, без root)

1. **Bundle** собран `scripts/build_patch_bundle.sh` из рабочего дерева
   (`manifest.txt + repo/`, ~17 MB без wheelhouse; tar.gz + sha256).
2. **Adopt dry-run** инсталлера прогнан против точной копии полевого дерева
   (`evo-ssearch-tbilisi-field`, VERSION β 0.8.0) с его реальным `.env`:
   сначала честный `BLOCKED` (см. находки ниже), после подготовки env —
   **полностью зелёный preflight** с планом из 11 шагов и WARN
   `no wheelhouse found; existing target venv would be reused` (это и есть
   ожидаемый adopt-путь).
3. **Миграционная репетиция** на одноразовой копии схемы боевой БД:
   `alembic current` → `20260614_0006 (head)`; `upgrade head` — **no-op**.
   Полевое дерево уже несёт голову схемы: **апгрейд 0.8.4 — code-only,
   без изменения схемы БД**. Дамп перед миграцией всё равно обязателен
   (инсталлер не даст его пропустить).
4. `requirements.txt`/`requirements-db.txt` полевого и нового дерева
   **идентичны** → wheelhouse для адопт-апгрейда не нужен. Он нужен только
   для fresh-хоста.

Не репетировалось (нужен root/стенд): сам `--apply` (создание service-user,
systemd, рестарт), power-loss recovery. `--apply` на дев-машине не выполнялся
сознательно.

## Находки репетиции — проверь ДО поездки

1. **Полевой env 0.8.0-эры неполон для инсталлера.** В нём отсутствуют:
   `EVA_DATABASE_DSN`, `EVA_AUDIT_DATABASE_DSN`, `EVA_WORKER_DATABASE_DSN`,
   `EVOSSEARCH_LM_PROFILE_VLM_BASE_URL`, `EVOSSEARCH_LM_PROFILE_VLM_MODEL`.
   Если на реальном хосте env такой же — подготовь значения заранее
   (DSN трёх runtime-ролей + VLM endpoint/модель). Non-interactive режим
   упадёт, интерактивный спросит — но лучше не подбирать пароли в серверной.
2. **Placeholder-правило отвергает слабые НАСТОЯЩИЕ пароли.** Если пароль Evo
   на объекте реально `123`/`changeme` — preflight выдаст FAIL
   `contains an obvious placeholder value`. Это не баг: смени пароль Evo на
   объекте до апгрейда, значение в диагностике не печатается.
3. **`pg_dump` нужен для маршрута с миграцией**, но не блокирует
   штатный code-only `field_upgrade_084.sh --no-migrate`: этот маршрут
   вообще не изменяет БД. Если схема не `20260614_0006`, проводник
   остановится; тогда разработчику нужны `postgresql-client` и
   отдельный migration/backup DSN.
4. **Приведённый migration DSN** — отдельная привилегированная роль PostgreSQL.
   Передавать транзиентно: `EVA_INSTALL_MIGRATION_DSN` через
   `sudo --preserve-env=...` (в env-файл не пишется). `EVA_DATABASE_DSN`
   инсталлер для DDL не использует и не возьмёт.
5. **Adopt-режим определяется по исполняемому `target/.venv/bin/python`.**
   Если venv на хосте переносили/ломали — инсталлер честно потребует
   wheelhouse как для fresh. Проводник также сам сравнит
   `requirements.txt`/`requirements-db.txt` с полевым деревом и запустит
   `.venv/bin/python -m pip check` (либо `uv pip check`, если venv собран без
   модуля pip); без wheelhouse любое расхождение — `STOP`.
6. Версионный гейт инсталлера теперь читает `VERSION` из bundle
   и сверяет его с `manifest.txt`; оба должны быть ровно `β 0.8.4`.
   `working_tree_status` в manifest должен быть `clean`; черновой bundle из
   dirty-дерева в поле будет отвергнут.
   Если в старом env явно записан `EVOSSEARCH_APP_VERSION`, инсталлер заменит
   только этот release-managed ключ; IP, DSN, модели и прочие
   site-настройки не перетираются.
7. **Существующий systemd unit в adopt-режиме не переписывается.** Его
   `User`/`Group`, hardening, drop-ins и зависимости остаются полевыми. Новый
   unit из шаблона создаётся только на fresh-хосте.
8. **Совместимость определяется не строкой версии.** `β 0.8.2`, `β 0.8.2.1`,
   `β 0.8.3` и промежуточные post-schema builds допускаются как кандидаты.
   Реальные жёсткие gates: схема ровно `20260614_0006`, побайтно одинаковые
   requirements, здоровый существующий venv и валидный clean bundle. Повторный
   запуск того же exact bundle commit запрещён по `.eva-bundle-commit`, чтобы
   не заменить точку отката бэкапом уже обновлённой версии. Другой commit той
   же `β 0.8.4` проходит как same-version hotfix.
9. **Сохранённые L1–L3 из 0.8.1 не перегенерировать.** На первом старте
   0.8.4 читает прежний `luxriot_rollup_cache`, помечает настоящие
   LM-generated записи как `legacy_cached` и раскладывает их в отдельные
   durable rows `luxriot_rollup:*`. Они сразу доступны UI и агенту; scheduler
   и restore-worker считают их готовыми и не вызывают модель повторно.
   Механическая строка вида `L1 rollup from L0: ...` семантикой не считается.
   В `/luxriot/streams` после старта проверить
   `rollup_scheduler.rollup_cache_entries_loaded`,
   `legacy_rollups_adopted` и `legacy_rollups_adopted_by_level`.
   Важно: старый cache был ограничен retention/размером (обычно 7 дней и
   800 записей), поэтому импорт сохраняет всё реально записанное 0.8.1, но
   не может восстановить никогда не созданные окна. Для таких окон источником
   остаются архивные L0 `vlm_summary`; их генерация должна быть отдельным,
   подтверждённым оператором restore, а не частью апгрейда.

   До остановки 0.8.1 зафиксировать, сколько настоящих LM-rollup текстов
   физически лежит в старом общем cache (видимая в UI динамическая карточка
   сама по себе этого не гарантирует):

   ```bash
   sudo -u postgres psql -X -v ON_ERROR_STOP=1 -d eva <<'SQL'
   WITH entries AS (
     SELECT jsonb_array_elements(
       COALESCE(payload_json -> 'entries', '[]'::jsonb)
     ) AS item
     FROM archive.runtime_state
     WHERE state_key = 'luxriot_rollup_cache'
   )
   SELECT
     COALESCE(item ->> 'level', '?') AS level,
     count(*) AS cached_rows,
     count(*) FILTER (
       WHERE lower(COALESCE(item ->> 'summary_kind', 'llm_cached'))
             IN ('llm', 'llm_cached')
         AND COALESCE(item ->> 'summary', '') NOT LIKE 'L1 rollup from L0:%'
     ) AS semantic_rows,
     to_timestamp(min((item ->> 'window_start')::double precision)) AS oldest,
     to_timestamp(max((item ->> 'window_start')::double precision)) AS newest
   FROM entries
   GROUP BY 1
   ORDER BY 1;
   SQL
   ```

   Сохранить вывод в evidence. После первого старта сумма
   `legacy_rollups_adopted_by_level` должна соответствовать найденным
   семантическим строкам, оставшимся внутри новой rollup-retention.

## Маршрут инженера (основной — корневой update.sh)

0.8.4 — code-only релиз: при схеме на голове `20260614_0006` база данных не
изменяется вообще. Основной маршрут идёт через корневой `./update.sh`, который
сам различает user/system systemd, проверяет bundle/media/requirements/venv,
**read-only** читает schema head, показывает реальный agent LM context, делает
code/env backup, устанавливает без миграции, прогревает lazy embedder, требует
`/ready` со статусом `ready` и автоматически откатывает code/env при ошибке
после остановки сервиса.

`repo/scripts/field_upgrade_084.sh` остаётся расширенным production-маршрутом
для root-инсталляции, когда нужен отдельный evidence-каталог и записанная
rollback-команда. Он использует те же schema/dependency gates и также допускает
любой непустой deployed VERSION, а не только старый allowlist.

```bash
# 0. Дома: собрать и проверить bundle
scripts/build_patch_bundle.sh --name eva-ai-0.8.4-r4-offline
(cd dist && sha256sum -c eva-ai-0.8.4-r4-offline.tar.gz.sha256)

# 1. На хосте: ещё раз проверить файл после USB-копирования,
#    затем распаковать и запустить основной updater БЕЗ sudo
sha256sum -c eva-ai-0.8.4-r4-offline.tar.gz.sha256
tar xzf eva-ai-0.8.4-r4-offline.tar.gz && cd eva-ai-0.8.4-r4-offline
./update.sh

# 2. Скрипт сам остановится на любой проблеме и скажет, что диктовать по
#    телефону. Ничего не «чинить творчески» на месте.
```

Правила для инженера в серверной:

- Любая строка `STOP:` до остановки сервиса означает, что ничего не изменено.
  После остановки сервиса root updater сам выполняет automatic rollback.
- Для расширенного `field_upgrade_084.sh` команда отката записывается в
  evidence-каталог (`/var/tmp/eva-upgrade-084-*/ROLLBACK_COMMAND.txt`) —
  выполнять её только по инструкции с телефона.
- `bash scripts/client_diagnostics.sh > diag.txt` — единственная команда
  диагностики, файл отправить разработчику.

## Ручной маршрут (только разработчик)

Полный маршрут с миграциями и транзиентным `EVA_INSTALL_MIGRATION_DSN` —
в `docs/install/offline_installer_083.md` (§4–§5). В поле он нужен только
если схема оказалась НЕ на голове — тогда апгрейд ведёт разработчик, не
инженер с флешкой.

## После апгрейда — smoke на месте (10 минут)

1. Логин оператора, вкладка Video: канал с summaries → `Model view` играет,
   значок STATIC FALLBACK при живых кадрах не залипает (автовосстановление ≤12 с).
2. Настройки канала: селектор `Frame selector` (Auto/Action/Clarity) виден,
   Apply/Reset работают.
3. Archive: открыть VLM-кадр → кадр-улика, filmstrip; видео — только по кнопке
   `Play archive video`.
4. Runtime-строка канала: `capture_apex_mode_counts` в `/luxriot/streams`
   (burst/normal/quiet растут), при живом источнике ≥2 fps —
   `capture_cv_sharp_active` появляется в selection_sources.
5. Агент: «когда были всплески активности выше нормы на канале N за час?» —
   ответ ссылается на burst-окна, не на прозу.

## Проверка после reboot (до отъезда)

```bash
sudo systemctl is-enabled eva-ai
sudo systemctl reboot
# после возврата SSH/TeamViewer:
systemctl is-active eva-ai
curl -fsS http://127.0.0.1:5000/health
curl -fsS http://127.0.0.1:5000/ready
systemctl --failed --no-pager
```

Имена inference-сервисов зависят от объекта; их `is-enabled`/`is-active`
проверяются по installation record. Сам апгрейд не переустанавливает
vLLM/llama.cpp, модели и CUDA runtime: он сохраняет уже развёрнутую
inference-топологию и её env endpoints.
