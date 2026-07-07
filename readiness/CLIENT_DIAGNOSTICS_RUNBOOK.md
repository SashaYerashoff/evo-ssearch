# EVA AI: клиентский диагностический пакет

Этот runbook нужен для короткого окна доступа к закрытой клиентской системе. Скрипт собирает факты без изменения состояния EVA AI: `systemctl`/`journalctl`, read-only HTTP `GET` endpoints, прямые проверки OpenAI-compatible/vLLM `/models` и digest-метаданные snapshot'ов через EVA endpoint.

Скрипт не запускает и не останавливает capture, не вызывает `start_capture`, `stop_capture`, `flush_capture`, bookmarks или любые POST/DELETE операции приложения.

## Что подготовить

- Выполнять на EVA AI control-plane host, где доступен `eva-ai.service`.
- Желательно запускать через `sudo`, чтобы прочитать `/etc/eva-ai/eva-ai.env` и полный journal.
- Нужны `bash`, `curl`, `tar`; `jq` опционален, но улучшает читаемость JSON в пакете.
- По умолчанию EVA API ожидается на `http://127.0.0.1:5000`. Если адрес другой, передайте `EVA_BASE_URL`.

## Быстрый запуск без авторизованных Luxriot endpoints

```bash
cd /opt/eva-ai/evo-ssearch
sudo EVA_BASE_URL=http://127.0.0.1:5000 scripts/client_diagnostics.sh
```

Такой запуск соберет `/health`, `/ready`, `/lm/models`, состояние `eva-ai.service`, journal и vLLM endpoints из доступного env. `/luxriot/channels`, `/luxriot/streams` и snapshot digests будут пропущены, потому что им нужна cookie авторизованного пользователя.

Для текущего пилота `/luxriot/streams` - главный read-only снимок video-description runtime: какие каналы пишут summaries, какая модель назначена, сколько кадров в очереди, были ли dropped frames/batches, last_error и desired-but-not-running каналы. Probe status вторичен и нужен только для задач, где явно включены probes.

## Запуск с временной cookie

Используйте учетную запись с правом видеть нужные каналы. Для полной диагностики лучше admin/operator с `allowedChannelIds: ["*"]`.

```bash
cd /opt/eva-ai/evo-ssearch
BASE_URL=http://127.0.0.1:5000

read -rsp "EVA password: " EVA_PASSWORD; echo
curl -sS -c /tmp/eva.cookies \
  -H 'Content-Type: application/json' \
  -X POST "${BASE_URL}/auth/login" \
  -d "{\"username\":\"admin\",\"password\":\"${EVA_PASSWORD}\"}"
unset EVA_PASSWORD

sudo EVA_BASE_URL="${BASE_URL}" \
  EVA_COOKIE_FILE=/tmp/eva.cookies \
  EVA_DIAG_SNAPSHOT_COUNT=3 \
  scripts/client_diagnostics.sh

rm -f /tmp/eva.cookies
```

Скрипт не кладет сами JPEG snapshot'ы в пакет. Для первых `EVA_DIAG_SNAPSHOT_COUNT` каналов из `/luxriot/channels` он сохраняет только `channel_id`, HTTP status, размер ответа, SHA1, latency и image headers. Для расследования "отваливались ли video descriptions" сопоставляйте `/luxriot/streams` с agent/video-summary coverage: отсутствие summaries за период означает gap покрытия, а не доказанный сетевой outage без `last_error` или journal evidence.

## Если vLLM endpoints не найдены

Если `/etc/eva-ai/eva-ai.env` недоступен или profile URLs временно нужно задать вручную:

```bash
sudo EVA_BASE_URL=http://127.0.0.1:5000 \
  EVA_DIAG_VLLM_BASE_URLS='http://192.168.3.104:8001/v1,http://192.168.3.104:8002/v1,http://192.168.3.11:8001/v1,http://192.168.3.11:8002/v1' \
  scripts/client_diagnostics.sh
```

Скрипт проверяет `${base_url}/models`. Для `EVOSSEARCH_LM_PROFILE_*_BASE_URL` он также использует соответствующий `EVOSSEARCH_LM_PROFILE_*_API_KEY`, если ключ задан, но ключ не записывает в пакет.

## Полезные параметры

- `EVA_BASE_URL` - адрес EVA AI API, например `http://127.0.0.1:5000` или `https://127.0.0.1`.
- `EVA_COOKIE_FILE` - cookie jar от `curl -c`, предпочтительный способ авторизации.
- `EVA_COOKIE` или `EVA_COOKIE_HEADER` - запасной способ передать cookie через env; используйте только если cookie jar неудобен.
- `EVA_DIAG_SNAPSHOT_COUNT=3` - сколько snapshot digest-проверок сделать.
- `EVA_DIAG_JOURNAL_LINES=300` - сколько последних строк journal собрать.
- `EVA_DIAG_OUTPUT_DIR=/tmp` - куда положить каталог и `.tar.gz`.
- `EVA_DIAG_ENV_FILE=/etc/eva-ai/eva-ai.env` - runtime env file, который надо разобрать без `source`.
- `EVA_DIAG_VLLM_BASE_URLS='http://host:8001/v1,http://host:8002/v1'` - явные vLLM/OpenAI-compatible endpoints.

## Что передать нам

После завершения скрипт напечатает:

```text
Directory: /tmp/eva-ai-diagnostics-YYYYMMDDTHHMMSSZ
Package:   /tmp/eva-ai-diagnostics-YYYYMMDDTHHMMSSZ.tar.gz
```

Передайте нам только `.tar.gz` package. Если возможно, добавьте в письме/тикете краткий контекст: дата и локальное время запуска, hostname, какие inference-серверы должны быть активны, и что именно наблюдал клиент.

## Как не засветить секреты

- Не отправляйте `/etc/eva-ai/eva-ai.env`, `.env`, cookie jar, пароли, screenshots из browser devtools или вывод `curl -v`.
- Временную `/tmp/eva.cookies` удалите сразу после запуска.
- Скрипт редактирует secret-like значения (`PASSWORD`, `TOKEN`, `COOKIE`, `API_KEY`, `AUTH`, DSN passwords, `Authorization`, `Set-Cookie`), но пакет все равно стоит просмотреть перед отправкой.
- Для быстрой проверки содержимого:

```bash
tar -tzf /tmp/eva-ai-diagnostics-YYYYMMDDTHHMMSSZ.tar.gz
mkdir -p /tmp/eva-ai-diag-review
tar -xzf /tmp/eva-ai-diagnostics-YYYYMMDDTHHMMSSZ.tar.gz -C /tmp/eva-ai-diag-review
rg -n "password|secret|token|cookie|Authorization|Bearer|postgresql://[^ ]+:[^<]" /tmp/eva-ai-diag-review
```

Если `rg` нашел реальные значения секретов, не отправляйте пакет. Удалите чувствительный фрагмент или повторите запуск без соответствующего env/cookie способа.

## Ожидаемые ограничения

- Без cookie будут 401/skip для `/luxriot/channels`, `/luxriot/streams` и snapshot digest-проверок.
- При 403 на snapshot'ах пользователь авторизован, но не имеет доступа к выбранным каналам.
- Если journal пустой или обрезан, запустите скрипт через `sudo`.
- Если vLLM endpoints не обнаружены, запустите через `sudo` или передайте `EVA_DIAG_VLLM_BASE_URLS`.
