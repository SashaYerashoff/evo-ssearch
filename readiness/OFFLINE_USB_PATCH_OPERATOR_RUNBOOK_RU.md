# EVA AI: offline-патч с USB-накопителя

Дата: 2026-07-02
Целевой релиз: `β 0.8.3`
Schema head: `20260614_0006`
Миграция БД: **нет** для обновления `β 0.8.2.1 -> β 0.8.3`

Статус: этот файл оставлен как полный legacy-runbook. Для полевой установки
предпочтительно использовать раздельные документы. Для космонавта используйте
английскую версию:

1. `readiness/OFFLINE_USB_01_PREPARE_MEDIA_EN.md` - USB preparation,
   Linux mount steps, and terminal cheat sheet.
2. `readiness/OFFLINE_USB_02_PREFLIGHT_DECISION_EN.md` - preflight and
   deployment scenario decision.
3. `readiness/OFFLINE_USB_03_INSTALL_AND_TEST_EN.md` - install variants,
   rollback, and manual post-install test.
4. `readiness/CLIENT_PHYSICAL_TOPOLOGY_0.8.3_EN.md` and
   `readiness/CLIENT_PHYSICAL_TOPOLOGY_0.8.3.svg` - client physical topology.

Русские версии оставлены рядом:

1. `readiness/OFFLINE_USB_01_PREPARE_MEDIA_RU.md` - подготовка флешки,
   поиск/монтирование USB под Linux и терминальная шпаргалка.
2. `readiness/OFFLINE_USB_02_PREFLIGHT_DECISION_RU.md` - preflight и выбор
   сценария установки.
3. `readiness/OFFLINE_USB_03_INSTALL_AND_TEST_RU.md` - варианты раскатки,
   rollback и ручной тест после установки.
4. `readiness/CLIENT_PHYSICAL_TOPOLOGY_0.8.3_RU.md` и
   `readiness/CLIENT_PHYSICAL_TOPOLOGY_0.8.3.svg` - физическая схема клиента.

Этот runbook рассчитан на инженера/оператора на клиентской EVA AI машине без
доступа в интернет. Цель: привезти patch bundle на флешке, выполнить preflight,
сделать backup, установить код, проверить `/health` и `/ready`, а при проблеме
откатиться.

Не копируйте и не отправляйте содержимое `/etc/eva-ai/eva-ai.env`: там могут
быть пароли и DSN. В терминал вставляйте только команды из блоков.

## 0. Что мы уже знаем о клиентской системе

Эти значения взяты из предыдущего клиентского installation record и текущих
полевых runbook-ов. Если на объекте факт отличается, сначала зафиксируйте
расхождение, потом продолжайте по реальному значению.

| Вопрос preflight | Ответ по текущему client/site record |
| --- | --- |
| EVA AI app dir | `/opt/eva-ai/evo-ssearch` |
| systemd service | `eva-ai` |
| service user/group | `eva:eva` |
| env file | `/etc/eva-ai/eva-ai.env` |
| internal app URL | `http://127.0.0.1:5000` |
| browser/client URL | site HTTPS/TLS boundary `[FIELD]`; internal Gunicorn remains HTTP |
| PostgreSQL DB fallback name | `eva` |
| expected schema | `20260614_0006` |
| worker count | `EVOSSEARCH_GUNICORN_WORKERS=1` |
| Luxriot Evo observed pilot URL | `http://192.168.3.27:8080` |
| inference A | `192.168.3.104`, ports `8001`/`8002` |
| inference B | `192.168.3.11`, ports `8001`/`8002` |
| VLM model | `qwen3-vl-4b-fp8` |
| VLM batch shape | 12 images per L0 batch, endpoint limit 16 images |

Важно: `https://127.0.0.1:5443` относится к local/dev или отдельному
TLS-enabled service. На клиентском control-plane и office/demo по умолчанию
проверяйте `http://127.0.0.1:5000`.

## 1. Что должно быть на флешке

На инженерной машине заранее собирается архив:

```text
eva-ai-patch-0.8.3-YYYYMMDD-HHMMSS.tar.gz
eva-ai-patch-0.8.3-YYYYMMDD-HHMMSS.tar.gz.sha256
```

Внутри архива:

```text
manifest.txt
repo/
scripts/preflight_patch.sh
scripts/install_patch.sh
scripts/verify_patch.sh
scripts/rollback.sh
scripts/set_site_ips.sh
scripts/client_diagnostics.sh
repo/readiness/OFFLINE_USB_PATCH_OPERATOR_RUNBOOK_RU.md
repo/readiness/CLIENT_DIAGNOSTICS_RUNBOOK.md
```

Если на клиенте может не хватить Python packages, в bundle также должен быть:

```text
wheelhouse/
wheelhouse_manifest.txt
```

Сборка на инженерной машине:

```bash
cd /home/sasha/Projects/evo-ssearch

scripts/build_patch_bundle.sh \
  --output-dir /tmp/eva-ai-usb \
  --name "eva-ai-patch-0.8.3-$(date +%Y%m%d-%H%M%S)" \
  --with-wheelhouse
```

`--with-wheelhouse` может сделать большой архив: `torch`, `opencv`,
`transformers` и связанные wheels тяжелые. Для закрытой сети это ожидаемо.
Собирать wheelhouse нужно на совместимой Linux/Python платформе. Если wheelhouse
уже подготовлен отдельно:

```bash
scripts/build_patch_bundle.sh \
  --output-dir /tmp/eva-ai-usb \
  --name "eva-ai-patch-0.8.3-$(date +%Y%m%d-%H%M%S)" \
  --wheelhouse-dir /path/to/wheelhouse
```

Скопируйте `.tar.gz` и `.sha256` на USB-накопитель.

## 2. Подготовка на клиентской машине

Вставьте флешку. Найдите путь к ней:

```bash
lsblk -f
```

В примерах ниже путь флешки обозначен как `/media/$USER/EVA_USB`. Замените его
на реальный путь.

Создайте рабочую директорию и скопируйте архив локально:

```bash
mkdir -p ~/eva-ai-patch
cp /media/$USER/EVA_USB/eva-ai-patch-*.tar.gz* ~/eva-ai-patch/
cd ~/eva-ai-patch
```

Проверьте контрольную сумму:

```bash
sha256sum -c eva-ai-patch-*.tar.gz.sha256
```

Ожидаемый результат: строка заканчивается на `OK`.

Распакуйте архив:

```bash
tar -xzf eva-ai-patch-*.tar.gz
cd eva-ai-patch-*
cat manifest.txt
```

Проверьте, что manifest показывает целевой релиз:

```text
version=β 0.8.3
```

## 3. Preflight до остановки сервиса

Запустите безопасную предпроверку. Она ничего не останавливает, не копирует и
не меняет.

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

Нормальный результат:

```text
Preflight result: OK_WITH_WARNINGS_OR_OK
```

Предупреждения допустимы, если они объяснены текущим состоянием объекта,
например Luxriot Evo или inference host временно выключен до работ. `FAIL`
нужно разобрать до установки.

Особенно проверьте:

- `wheelhouse found`, если на объекте нет интернета и могут понадобиться wheels;
- `EVOSSEARCH_GUNICORN_WORKERS` равен `1`;
- schema revision `20260614_0006`;
- backup filesystem имеет достаточно места;
- `/health` отвечает на том URL, который вы будете использовать после установки.

Если `/ready` уже не `ready` до установки, зафиксируйте это в акте работ. Патч
можно ставить, но после установки нельзя считать старую внешнюю проблему новой.

## 4. Установка патча

Запустите установку от root:

```bash
sudo scripts/install_patch.sh \
  --bundle-dir "$PWD" \
  --app-dir /opt/eva-ai/evo-ssearch \
  --env-file /etc/eva-ai/eva-ai.env \
  --service eva-ai \
  --base-url http://127.0.0.1:5000 \
  --pg-database eva
```

Что делает installer:

- создаёт backup в `/var/backups/eva-ai/patch-YYYYMMDD-HHMMSS`;
- сохраняет env-файл, systemd unit/drop-ins и текущий код;
- делает `pg_dump`, если доступен `pg_dump` и найден DSN или локальная база;
- останавливает `eva-ai`;
- копирует код из bundle в `/opt/eva-ai/evo-ssearch`;
- сохраняет существующие `.venv`, `.env`, archive/state/db файлы;
- ставит wheels из `wheelhouse/`, если он есть в bundle;
- запускает `eva-ai`;
- проверяет `/health` и `/ready`.

Для `β 0.8.3` не добавляйте `--run-migrations`: миграции нет.

Если installer напечатал `FAIL`, не продолжайте ручные правки. Сохраните вывод
команды и переходите к rollback.

## 5. Проверка после установки

Повторите verification:

```bash
scripts/verify_patch.sh \
  --service eva-ai \
  --base-url http://127.0.0.1:5000 \
  --timeout 60
```

Проверьте фактический bind:

```bash
systemctl status eva-ai --no-pager -l
```

Если `Listening at:` показывает `http://0.0.0.0:5000`, используйте
`http://127.0.0.1:5000`. Если показывает `https://0.0.0.0:5443`, только тогда
используйте `https://127.0.0.1:5443` и `curl -k`.

Проверьте health/readiness:

```bash
curl -sS http://127.0.0.1:5000/health | jq
curl -sS http://127.0.0.1:5000/ready | jq '.status, .checks.deployment_security, .checks.luxriot, .checks.lm_profiles, .checks.postgresql'
```

Ожидаемое:

- `/health.version` = `β 0.8.3`;
- PostgreSQL/auth checks готовы;
- `deployment_security` не содержит неожиданных placeholder/HTTP/cookie проблем;
- Luxriot и LM/VLM profiles reachable, если соответствующие машины включены;
- `inference_queue.status=disabled` допустим в текущем пилоте;
- browser UI открывается, login работает, live preview честно показывает signal
  lost/frozen вместо старого буфера.

### Проверка маршрутизации моделей

После установки отдельно проверьте, что VLM-balancer смотрит на vLLM endpoints с
`qwen3-vl-4b-fp8`, а агент смотрит на локальный agent endpoint EVA AI host с
`qwen3.5-9b-mtp`.

Проверка через runtime `/ready`:

```bash
curl -sS http://127.0.0.1:5000/ready \
  | jq '.checks.lm_profiles.profiles[]
    | {id, kind, model, base_url, required, ok, status}'
```

Ожидаемая картина для клиентского пилота:

```text
agent:
  kind=agent
  base_url=http://127.0.0.1:1234/v1
  model=qwen3.5-9b-mtp

vlm-a1:
  kind=vlm
  base_url=http://192.168.3.104:8001/v1
  model=qwen3-vl-4b-fp8

vlm-a0:
  kind=vlm
  base_url=http://192.168.3.104:8002/v1
  model=qwen3-vl-4b-fp8

vlm-b1:
  kind=vlm
  base_url=http://192.168.3.11:8001/v1
  model=qwen3-vl-4b-fp8

vlm-b0:
  kind=vlm
  base_url=http://192.168.3.11:8002/v1
  model=qwen3-vl-4b-fp8
```

Также проверьте, что balancer включен и перечисляет VLM profiles:

```bash
sudo grep -E \
  '^(EVOSSEARCH_LM_PROFILES|EVOSSEARCH_LM_AGENT_PROFILE_ID|EVOSSEARCH_LM_VLM_PROFILE_ID|EVOSSEARCH_LM_VLM_BALANCER_ENABLED|EVOSSEARCH_LM_VLM_BALANCER_PROFILES)=' \
  /etc/eva-ai/eva-ai.env
```

Ожидаемо:

```text
EVOSSEARCH_LM_PROFILES=agent,vlm-a1,vlm-a0,vlm-b1,vlm-b0
EVOSSEARCH_LM_AGENT_PROFILE_ID=agent
EVOSSEARCH_LM_VLM_BALANCER_ENABLED=true
EVOSSEARCH_LM_VLM_BALANCER_PROFILES=vlm-a1,vlm-a0,vlm-b1,vlm-b0
```

Если `agent` указывает на `qwen3-vl-4b-fp8` или один из `vlm-*` указывает на
`qwen3.5-9b-mtp`, остановитесь и исправьте `/etc/eva-ai/eva-ai.env` до запуска
массовых video descriptions. Если balancer выключен, каналы могут пойти в один
default VLM endpoint вместо распределения по четырем GPU.

## 6. Если на объекте изменились IP-адреса

Узнайте актуальные IP:

```bash
hostname -I
ip -br addr
```

Примените новые адреса к env-файлу. Подставьте реальные значения:

```bash
sudo scripts/set_site_ips.sh \
  --env-file /etc/eva-ai/eva-ai.env \
  --service eva-ai \
  --luxriot-ip 192.168.3.27 \
  --luxriot-port 8080 \
  --inference-a-ip 192.168.3.104 \
  --inference-b-ip 192.168.3.11 \
  --agent-base-url http://127.0.0.1:1234/v1 \
  --restart
```

Скрипт не меняет логины и пароли. Если изменились Luxriot credentials, правьте
их вручную через:

```bash
sudo nano /etc/eva-ai/eva-ai.env
```

После смены IP повторите preflight/verify:

```bash
sudo scripts/preflight_patch.sh \
  --app-dir /opt/eva-ai/evo-ssearch \
  --env-file /etc/eva-ai/eva-ai.env \
  --service eva-ai \
  --base-url http://127.0.0.1:5000 \
  --pg-database eva \
  --expected-schema 20260614_0006

scripts/verify_patch.sh \
  --service eva-ai \
  --base-url http://127.0.0.1:5000
```

## 7. Минимальная UI/agent проверка

После restart live video descriptions могут требовать повторного запуска на
нужных каналах. В UI проверьте:

- версия в `/health` или UI: `β 0.8.3`;
- `Video Monitoring` показывает реальный live/signal lost/frozen, а не старый
  кадр из буфера;
- agent `Stream status` говорит о video-description streams, live signal state,
  dropped/pending/last_error и runtime problem channels;
- agent report остаётся video-description-first и отдельно показывает pipeline
  health;
- probe preview/apply остаётся gated: агент не применяет изменения без UI Apply;
- road/drift outputs формулируются как candidates/evidence, не как юридическое
  заключение о нарушении.

## 8. Rollback без восстановления базы

Обычный rollback возвращает код, env и systemd unit из последнего backup. Базу
данных он не трогает.

```bash
sudo scripts/rollback.sh \
  --app-dir /opt/eva-ai/evo-ssearch \
  --env-file /etc/eva-ai/eva-ai.env \
  --service eva-ai \
  --base-url http://127.0.0.1:5000
```

Если нужно откатиться не на последний backup, укажите директорию явно:

```bash
sudo scripts/rollback.sh \
  --backup-dir /var/backups/eva-ai/patch-YYYYMMDD-HHMMSS \
  --app-dir /opt/eva-ai/evo-ssearch \
  --env-file /etc/eva-ai/eva-ai.env \
  --service eva-ai \
  --base-url http://127.0.0.1:5000
```

После rollback выполните проверку из раздела 5.

## 9. Rollback базы данных только по согласованию

Восстановление PostgreSQL dump является разрушительной операцией: текущие
данные в базе могут быть заменены состоянием на момент backup. Делайте это
только после явного согласования с ответственным инженером.

Команда требует отдельного подтверждения:

```bash
sudo EVA_PATCH_CONFIRM_DB_RESTORE=yes scripts/rollback.sh \
  --restore-db \
  --backup-dir /var/backups/eva-ai/patch-YYYYMMDD-HHMMSS \
  --app-dir /opt/eva-ai/evo-ssearch \
  --env-file /etc/eva-ai/eva-ai.env \
  --service eva-ai \
  --base-url http://127.0.0.1:5000
```

## 10. Что отправить инженеру после работ

Отправьте только безопасные артефакты:

```bash
cat ~/eva-ai-patch/eva-ai-patch-*/manifest.txt
sudo ls -la /var/backups/eva-ai
scripts/verify_patch.sh --service eva-ai --base-url http://127.0.0.1:5000
```

Если что-то пошло не так:

```bash
sudo journalctl -u eva-ai -n 160 --no-pager -l
```

Не отправляйте `/etc/eva-ai/eva-ai.env`, DB dumps, cookies, passwords, tokens или
полные DSN.
