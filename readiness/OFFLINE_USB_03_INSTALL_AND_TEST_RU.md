# EVA AI offline update: 03 - варианты раскатки и ручной тест после установки

Цель: установить offline-патч, проверить сервис и выполнить минимальный
ручной тест, достаточный для допуска к полевому демо.

Целевой релиз: `β 0.8.3`  
Schema head: `20260614_0006`  
Миграция БД для `β 0.8.2.1 -> β 0.8.3`: **нет**

## 1. Общие переменные

В распакованном bundle:

```bash
cd ~/eva-ai-patch/eva-ai-patch-0.8.3-*

export APP_DIR=/opt/eva-ai/evo-ssearch
export SERVICE=eva-ai
export ENV_FILE=/etc/eva-ai/eva-ai.env
export BASE_URL=http://127.0.0.1:5000
export DB_NAME=eva
```

Если preflight показал HTTPS service:

```bash
export BASE_URL=https://127.0.0.1:5443
export EVA_PATCH_CURL_INSECURE=true
```

## 2. Сценарий A: стандартная offline-установка

Используйте, если preflight OK, места достаточно, checksum OK, bundle содержит
wheelhouse или существующий `.venv` рабочий.

```bash
sudo scripts/install_patch.sh \
  --bundle-dir "$PWD" \
  --app-dir "$APP_DIR" \
  --env-file "$ENV_FILE" \
  --service "$SERVICE" \
  --base-url "$BASE_URL" \
  --pg-database "$DB_NAME"
```

Для `β 0.8.3` не добавляйте `--run-migrations`: миграции нет.

## 3. Сценарий B: code-only / reuse existing venv

Используйте только если bundle без wheelhouse, интернета нет, но preflight
подтвердил существующий `.venv/bin/python` и текущая установка уже работала.

Команда такая же, как в сценарии A. Installer сохраняет существующий `.venv`.
Если wheelhouse отсутствует, он не будет переустанавливать зависимости.

```bash
sudo scripts/install_patch.sh \
  --bundle-dir "$PWD" \
  --app-dir "$APP_DIR" \
  --env-file "$ENV_FILE" \
  --service "$SERVICE" \
  --base-url "$BASE_URL" \
  --pg-database "$DB_NAME"
```

После установки особенно внимательно проверьте `/health`, `/ready` и импорт
ключевых модулей через `verify_patch.sh`.

## 4. Сценарий C: установка без pg_dump

Используйте только если:

- есть актуальный внешний backup, или
- база не содержит принципиально нужных данных, или
- ответственный инженер явно разрешил пропустить DB dump.

```bash
sudo scripts/install_patch.sh \
  --bundle-dir "$PWD" \
  --app-dir "$APP_DIR" \
  --env-file "$ENV_FILE" \
  --service "$SERVICE" \
  --base-url "$BASE_URL" \
  --pg-database "$DB_NAME" \
  --skip-pg-dump
```

Это не отключает backup кода/env/systemd. Пропускается только PostgreSQL dump.

## 5. Проверка после установки

```bash
scripts/verify_patch.sh \
  --service "$SERVICE" \
  --base-url "$BASE_URL" \
  --timeout 90
```

Если HTTPS с self-signed:

```bash
EVA_PATCH_CURL_INSECURE=true scripts/verify_patch.sh \
  --service "$SERVICE" \
  --base-url "$BASE_URL" \
  --timeout 90
```

Проверьте systemd:

```bash
systemctl status "$SERVICE" --no-pager -l
journalctl -u "$SERVICE" -n 120 --no-pager
```

Проверьте endpoints:

```bash
curl -sS "$BASE_URL/health" | jq
curl -sS "$BASE_URL/ready" | jq '.status, .checks.postgresql, .checks.authentication, .checks.luxriot, .checks.lm_profiles'
```

Если `BASE_URL=https://...`, добавьте `-k`:

```bash
curl -k -sS "$BASE_URL/health" | jq
curl -k -sS "$BASE_URL/ready" | jq '.status, .checks.postgresql, .checks.authentication, .checks.luxriot, .checks.lm_profiles'
```

Ожидаемое:

- `/health.version` = `β 0.8.3`;
- service active;
- PostgreSQL/auth checks готовы;
- Luxriot reachable, если Evo server включён;
- LM/VLM profiles reachable, если inference servers включены;
- `inference_queue.status=disabled` допустим для текущего пилота.

## 6. Проверка маршрутизации моделей после установки

В `/ready` или в Admin Settings проверьте:

- VLM/video-description profiles: vLLM servers с `qwen3-vl-4b-fp8`;
- agent/chat profile: LM Studio / EVA AI host с `qwen3.5-9b-mtp`;
- `.env` не перетёр пользовательский выбор модели в UI.

Команда:

```bash
curl -sS "$BASE_URL/ready" \
  | jq '.checks.lm_profiles'
```

Если структура ответа не показывает все profiles, сделайте проверку через UI:

```text
Admin / Settings -> LM profiles / Video descriptions / Agent model
```

## 7. Минимальный ручной тест UI

Откройте EVA AI в браузере или Luxriot EVO Monitor web tile. После установки
сделайте hard refresh.

### 7.1 Login и версия

1. Войти пользователем с правами оператора/админа.
2. Проверить, что UI открывается без бесконечного loading.
3. Проверить версию `β 0.8.3`, если она выводится в UI/status.

### 7.2 Video tab / live signal honesty

1. Открыть video monitoring.
2. Выбрать живой канал.
3. Убедиться, что preview обновляется и не крутит старый буфер.
4. Если канал выключен в Luxriot, UI должен показать `Signal lost` /
   `No fresh EVA frame`, а не продолжать показывать старое видео.
5. Если VLM обрабатывает батч, допустим `slow` / `processing delay`; это не
   должно превращаться в ложный `signal lost`.

### 7.3 VLM feed и alerts

1. Проверить последние video summaries.
2. Проверить, что alerts имеют evidence/thumbnail, если событие было найдено.
3. Для road/street канала проверить, что road/drift outputs сформулированы как
   candidate/evidence, не как юридическое заключение.

### 7.4 Road mask / grounding

1. На дорожном канале нажать `Ground road mask`.
2. Overlay должен строиться только по свежим EVA frames.
3. Если свежих кадров нет, UI должен показать ошибку, а не старую картинку.

### 7.5 Agent

Спросить:

```text
Show recent VLM alerts and notable video-summary events for the last hour.
```

Ожидаемое:

- агент использует video-description tools;
- явно говорит о coverage/partial coverage;
- не утверждает, что просмотрел весь период, если coverage неполное.

Спросить:

```text
Check live video-description channel status and report signal problems.
```

Ожидаемое:

- агент показывает live signal/runtime problems;
- отличает `detected`, `delivered`, `cooldown`, `failed`, если есть данные;
- не прячет disabled/frozen/stale канал как спокойный.

### 7.6 Probes preview/apply

1. Попросить агента создать или изменить тестовую probe.
2. Должна появиться отдельная preview/apply карточка.
3. Apply выполняется через UI approval, не через прямое применение моделью.
4. После Apply появляется receipt.

### 7.7 Archive evidence modal

1. Открыть найденный alert/evidence.
2. Проверить, что картинка отображается.
3. Если alert anchor не самый информативный кадр, листание batch frames должно
   позволить посмотреть соседние кадры.
4. `Open VLM feed` должен вести к соответствующему времени/контексту, а не
   просто переключать вкладку без ориентира.

## 8. Rollback

Если после установки сервис не стартует или `/health` не поднимается:

```bash
sudo scripts/rollback.sh \
  --app-dir "$APP_DIR" \
  --env-file "$ENV_FILE" \
  --service "$SERVICE" \
  --base-url "$BASE_URL" \
  --pg-database "$DB_NAME"
```

Если нужно восстановить PostgreSQL dump, используйте только после отдельного
подтверждения:

```bash
sudo EVA_PATCH_CONFIRM_DB_RESTORE=yes scripts/rollback.sh \
  --app-dir "$APP_DIR" \
  --env-file "$ENV_FILE" \
  --service "$SERVICE" \
  --base-url "$BASE_URL" \
  --pg-database "$DB_NAME" \
  --restore-db
```

После rollback:

```bash
scripts/verify_patch.sh --service "$SERVICE" --base-url "$BASE_URL" --timeout 90
```

## 9. Что отправить инженеру после работ

Без секретов:

- фото/текст `manifest.txt`;
- результат `sha256sum -c`;
- результат preflight;
- команда сценария A/B/C, который был выбран;
- `/health` JSON;
- краткий `/ready` summary;
- `systemctl status eva-ai --no-pager -l`;
- 2-3 скриншота UI: live channel, disabled/stale channel, agent status/alert report;
- если был rollback: путь backup directory и причина.

