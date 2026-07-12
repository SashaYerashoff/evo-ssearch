# EVA AI β 0.8.4: полевой апгрейд с 0.8.0/0.8.1 — чек-лист и результаты репетиции

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
3. **`pg_dump` должен быть на app-хосте** (`apt install postgresql-client`
   заранее, оффлайн-хосту — deb на ту же флешку). Без него preflight FAIL.
4. **Приведённый migration DSN** — отдельная привилегированная роль PostgreSQL.
   Передавать транзиентно: `EVA_INSTALL_MIGRATION_DSN` через
   `sudo --preserve-env=...` (в env-файл не пишется). `EVA_DATABASE_DSN`
   инсталлер для DDL не использует и не возьмёт.
5. **Adopt-режим определяется по исполняемому `target/.venv/bin/python`.**
   Если venv на хосте переносили/ломали — инсталлер честно потребует
   wheelhouse как для fresh. Проверь `test -x /opt/eva-ai/evo-ssearch/.venv/bin/python`.
6. Версионный гейт инсталлера теперь читает `VERSION` из bundle
   (β 0.8.4) — отдельно ничего править не нужно.

## Маршрут инженера

```bash
# 0. Дома: собрать и проверить bundle
scripts/build_patch_bundle.sh --name eva-ai-0.8.4-offline
sha256sum -c dist/eva-ai-0.8.4-offline.tar.gz.sha256

# 1. На хосте: распаковать, dry-run (по умолчанию), читать каждый FAIL/WARN
tar xzf eva-ai-0.8.4-offline.tar.gz && cd eva-ai-0.8.4-offline/repo
./scripts/install_eva_083.py --dry-run --non-interactive \
  --source-dir "$PWD" --bundle-dir "$PWD/.." \
  --app-dir /opt/eva-ai/evo-ssearch --env-file /etc/eva-ai/eva-ai.env

# 2. Устранить все FAIL (см. находки выше), повторить dry-run до чистого плана

# 3. Apply — только после чистого dry-run
read -rsp 'Migration DSN: ' EVA_INSTALL_MIGRATION_DSN; echo; export EVA_INSTALL_MIGRATION_DSN
sudo --preserve-env=EVA_INSTALL_MIGRATION_DSN ./scripts/install_eva_083.py --apply --non-interactive \
  --source-dir "$PWD" --bundle-dir "$PWD/.." \
  --app-dir /opt/eva-ai/evo-ssearch --env-file /etc/eva-ai/eva-ai.env
unset EVA_INSTALL_MIGRATION_DSN

# 4. Проверка
systemctl status eva-ai --no-pager -l
curl -sS http://127.0.0.1:5000/health && curl -sS http://127.0.0.1:5000/ready

# 5. Если плохо — rollback командой, которую напечатал инсталлер
#    (backup_dir указан в выводе; БД restore — отдельное осознанное действие)
```

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
