# EVA AI offline update: 02 - preflight и выбор сценария раскатки

Цель: до остановки сервиса понять, можно ли ставить патч, какой режим установки
выбрать и какие риски зафиксировать.

Целевой релиз: `β 0.8.3`  
Schema head: `20260614_0006`  
Миграция БД для `β 0.8.2.1 -> β 0.8.3`: **нет**

## 1. Известные параметры клиентского стенда

Используйте эти значения как стартовые. Если на объекте факт отличается, не
угадывайте: зафиксируйте отличие и используйте реальное значение в командах.

| Параметр | Значение по текущему client/site record |
| --- | --- |
| EVA AI app dir | `/opt/eva-ai/evo-ssearch` |
| systemd service | `eva-ai` |
| service user/group | `eva:eva` |
| env file | `/etc/eva-ai/eva-ai.env` |
| internal app URL | `http://127.0.0.1:5000` |
| PostgreSQL DB fallback name | `eva` |
| expected schema | `20260614_0006` |
| expected app version after patch | `β 0.8.3` |
| worker count | `EVOSSEARCH_GUNICORN_WORKERS=1` |
| Luxriot Evo observed pilot URL | `http://192.168.3.27:8080` |
| inference A | `192.168.3.104`, ports `8001` / `8002` |
| inference B | `192.168.3.11`, ports `8001` / `8002` |
| VLM model | `qwen3-vl-4b-fp8` |
| agent model | `qwen3.5-9b-mtp` |

Важно: `https://127.0.0.1:5443` относится к local/dev или отдельному
TLS-enabled service. На клиентском control-plane по умолчанию проверяйте
`http://127.0.0.1:5000`, пока `systemctl status eva-ai` не показывает другое.

## 2. Запуск safe preflight

Перейдите в распакованный bundle:

```bash
cd ~/eva-ai-patch/eva-ai-patch-0.8.3-*
```

Запустите preflight. Он ничего не останавливает, не копирует и не меняет.

```bash
sudo scripts/preflight_patch.sh \
  --bundle-dir "$PWD" \
  --app-dir /opt/eva-ai/evo-ssearch \
  --env-file /etc/eva-ai/eva-ai.env \
  --service eva-ai \
  --base-url http://127.0.0.1:5000 \
  --pg-database eva \
  --expected-version "β 0.8.3" \
  --expected-schema 20260614_0006
```

Если site использует локальный HTTPS service:

```bash
sudo EVA_PATCH_CURL_INSECURE=true scripts/preflight_patch.sh \
  --bundle-dir "$PWD" \
  --app-dir /opt/eva-ai/evo-ssearch \
  --env-file /etc/eva-ai/eva-ai.env \
  --service eva-ai \
  --base-url https://127.0.0.1:5443 \
  --pg-database eva \
  --expected-version "β 0.8.3" \
  --expected-schema 20260614_0006
```

## 3. Что сохранить из preflight

Сохраните вывод команды в файл или скриншот, особенно:

- `bundle manifest`;
- `current VERSION`;
- `systemd service exists`;
- `WorkingDirectory`, `User`, `Group`;
- `venv python exists`;
- `PostgreSQL` / schema revision;
- свободное место;
- `/health` и `/ready`;
- наличие `wheelhouse`;
- предупреждения и `FAIL`.

Не отправляйте полный `/etc/eva-ai/eva-ai.env`.

## 4. Decision table

| Preflight результат | Что делать |
| --- | --- |
| Все required checks OK, `wheelhouse found`, места достаточно | Сценарий A: стандартная offline-установка с backup. |
| `wheelhouse` отсутствует, но существующий `.venv` есть и зависимости уже стоят | Сценарий B: code-only / reuse existing venv. Допустимо для срочного патча, но зафиксируйте риск. |
| `pg_dump` недоступен, но данных принципиально не сохраняли или есть актуальный backup | Сценарий C: установка с `--skip-pg-dump` только после явного решения ответственного. |
| `/ready` был `not_ready` до установки из-за Luxriot/LM/vLLM/сети | Установка возможна, но зафиксируйте baseline. После патча нельзя считать старую внешнюю проблему новой. |
| service `eva-ai` не active до установки | Установка возможна только если это ожидаемо. Зафиксируйте состояние и после установки проверяйте отдельно запуск сервиса. |
| app dir, service name, env file отличаются от таблицы | Не ставьте с дефолтными командами. Подставьте реальные `--app-dir`, `--service`, `--env-file`. |
| schema revision не `20260614_0006` | Стоп. Нужна инженерная проверка; этот patch не должен менять схему. |
| нет места под backup / bundle / wheelhouse | Стоп. Освободить место или выбрать другой backup root. |
| checksum не OK | Стоп. Bundle повреждён или скопирован не полностью. |
| manifest version не `β 0.8.3` | Стоп. Нужен правильный bundle. |

## 5. Выбор base URL

Посмотрите, как сервис слушает порт:

```bash
systemctl status eva-ai --no-pager -l
```

Если видите:

```text
Listening at: http://0.0.0.0:5000
```

используйте:

```text
http://127.0.0.1:5000
```

Если видите:

```text
Listening at: https://0.0.0.0:5443
```

используйте:

```text
https://127.0.0.1:5443
```

и добавляйте `EVA_PATCH_CURL_INSECURE=true` для self-signed TLS checks.

## 6. Проверка маршрутизации моделей до установки

Если сервис уже работает, проверьте текущую маршрутизацию:

```bash
curl -sS http://127.0.0.1:5000/ready \
  | jq '.checks.lm_profiles'
```

Если локальный HTTPS:

```bash
curl -k -sS https://127.0.0.1:5443/ready \
  | jq '.checks.lm_profiles'
```

Ожидаемая логика:

- VLM/video-description profiles указывают на vLLM servers с
  `qwen3-vl-4b-fp8`;
- agent/chat profile указывает на EVA AI / LM Studio host с
  `qwen3.5-9b-mtp`;
- `.env` не должен сбрасывать UI-настройки модели на другой endpoint при
  обновлении кода.

Если вывод `/ready` не раскрывает profiles достаточно подробно, проверьте
админский UI Settings после установки. Не меняйте `.env` вслепую.

## 7. Готовность к установке

Перед переходом к документу `03` должны быть понятны:

```text
APP_DIR=/opt/eva-ai/evo-ssearch
SERVICE=eva-ai
ENV_FILE=/etc/eva-ai/eva-ai.env
BASE_URL=http://127.0.0.1:5000
DB_NAME=eva
SCENARIO=A|B|C
```

Дальше используйте:

```text
readiness/OFFLINE_USB_03_INSTALL_AND_TEST_RU.md
```

