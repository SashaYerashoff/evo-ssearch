# EVA AI: запись фактической пилотной инсталляции, июнь 2026

Дата фиксации: 2026-06-22  
Ветка: `feature/secure-50-channel-foundation`  
Версия приложения на пилоте: `β 0.8.1`  
Назначение: зафиксировать не желаемую архитектуру, а фактически поднятую
трехсерверную схему, параметры сервисов и ошибки, на которых мы уже
споткнулись во время развертывания.

Связанные runbook-и:

- `readiness/UBUNTU_LTS_INSTALL_GUIDE_RU.md`
- `readiness/VLLM_QWEN3_VL_INFERENCE_SERVER_RUNBOOK.md`
- `readiness/CLIENT_RESTART_AND_IP_CHANGE_RUNBOOK.md`
- `readiness/POSTGRES_FOUNDATION_RUNBOOK.md`
- `readiness/RELEASE_NOTES_0.8.1.md`
- `readiness/RELEASE_NOTES_0.8.0.md`

## 1. Общая схема

Пилот собран как три EVA AI машины плюс существующий Luxriot Evo как источник
каналов, snapshots и bookmarks.

| Узел | Фактическая роль | Основные сервисы | Сетевые порты |
| --- | --- | --- | --- |
| EVA AI control-plane, hostname `Luxriot` | Web UI/API, PostgreSQL, auth/RBAC, audit, archive, CLIP, агентная модель через OpenAI-compatible endpoint | `eva-ai.service`, `postgresql`, LM Studio/agent endpoint | `5000` для EVA AI, `1234` для agent endpoint если LM Studio локально |
| Inference A, `Luxriot1`, наблюдаемый IP `192.168.3.104` | VLM inference на двух RTX 5080 | `eva-vllm-gpu1.service`, `eva-vllm-gpu0.service` | `8001` GPU1, `8002` GPU0 |
| Inference B, `Luxriot2`, наблюдаемый IP `192.168.3.11` | VLM inference на двух RTX 5080 | `eva-vllm-gpu1.service`, `eva-vllm-gpu0.service` | `8001` GPU1, `8002` GPU0 |
| Luxriot Evo, наблюдаемый URL `http://192.168.3.27:8080` | Источник channel list, snapshots и bookmarks | Luxriot Evo | `8080` |

Фактическая модель распределения:

- агент и control-plane живут на EVA AI машине;
- live video descriptions идут в четыре VLM endpoint-а;
- VLM balancer в EVA AI статически распределяет каналы по profile id;
- Postgres находится на EVA AI control-plane машине;
- CLIP/archive работают на EVA AI control-plane машине;
- Luxriot Evo остается внешней системой, EVA AI ходит к нему по HTTP API.

Полевое наблюдение: на 50 live video-description каналов, 1 fps на канал и
около 30 секунд задержки/цикла, две inference-машины с суммарно 4 x RTX 5080
держали нагрузку примерно на уровне 70%. Это не гарантия емкости, но полезная
стартовая точка для клиентского пилота.

## 2. vLLM inference-сервисы

Обе inference-машины подняты одинаково:

- OS: свежая Ubuntu, amd64;
- GPU: 2 x NVIDIA GeForce RTX 5080 16 GB;
- driver на первой машине наблюдался как `595.71.05`;
- vLLM: `0.23.0`;
- Torch: `2.11.0+cu130`;
- модель: `Qwen/Qwen3-VL-4B-Instruct-FP8`;
- local path: `/opt/eva-vllm/models/qwen3-vl-4b-fp8`;
- Hugging Face cache: `/opt/eva-vllm/hf`;
- Python venv: `/opt/eva-vllm/.venv`;
- сервисы systemd: `/etc/systemd/system/eva-vllm-gpu1.service` и
  `/etc/systemd/system/eva-vllm-gpu0.service`.

Финальные рабочие endpoint-ы:

| Profile | Машина | GPU | Port | Base URL |
| --- | --- | --- | --- | --- |
| `vlm-a1` | `Luxriot1` | GPU1 | `8001` | `http://192.168.3.104:8001/v1` |
| `vlm-a0` | `Luxriot1` | GPU0 | `8002` | `http://192.168.3.104:8002/v1` |
| `vlm-b1` | `Luxriot2` | GPU1 | `8001` | `http://192.168.3.11:8001/v1` |
| `vlm-b0` | `Luxriot2` | GPU0 | `8002` | `http://192.168.3.11:8002/v1` |

### Финальные параметры vLLM

Критично: старый ранний smoke-test был `4096/image 8`, но для реального
пилота это недостаточно. Live summaries EVA AI отправляют batch из 12
изображений. Финальное рабочее состояние:

```text
--served-model-name qwen3-vl-4b-fp8
--host 0.0.0.0
--port 8001 или 8002
--max-model-len 8192
--gpu-memory-utilization 0.82
--max-num-seqs 4
--limit-mm-per-prompt.video 0
--limit-mm-per-prompt.image 16
--mm-processor-cache-gb 0
--trust-remote-code
```

Финальные environment-переменные в unit-файлах:

```ini
Environment=HF_HOME=/opt/eva-vllm/hf
Environment=HF_HUB_DISABLE_XET=1
Environment=OMP_NUM_THREADS=1
Environment=CUDA_VISIBLE_DEVICES=1
Environment=VLLM_USE_FLASHINFER_SAMPLER=0
```

Для GPU0 отличается только `CUDA_VISIBLE_DEVICES=0` и `--port 8002`.

Если endpoint падает по OOM, первым снижать только concurrency:

```text
--max-num-seqs 4 -> --max-num-seqs 2
```

Не снижать `--limit-mm-per-prompt.image` ниже `12`, иначе live
video-description снова начнет копить `pending_frames` и не будет писать
summary.

### Проверка vLLM после рестарта

На каждой inference-машине:

```bash
sudo systemctl restart eva-vllm-gpu1 eva-vllm-gpu0
sleep 90

sudo systemctl status eva-vllm-gpu1 --no-pager -l
sudo systemctl status eva-vllm-gpu0 --no-pager -l

curl -sS http://127.0.0.1:8001/v1/models | jq '.data[0] | {id,max_model_len}'
curl -sS http://127.0.0.1:8002/v1/models | jq '.data[0] | {id,max_model_len}'

nvidia-smi
```

Ожидаемо:

- оба сервиса `active (running)`;
- оба `/v1/models` показывают `qwen3-vl-4b-fp8`;
- оба `/v1/models` показывают `"max_model_len": 8192`;
- `nvidia-smi` показывает по одному vLLM process на каждую GPU.

## 3. EVA AI control-plane

Основной сервис:

```text
/etc/systemd/system/eva-ai.service
/etc/eva-ai/eva-ai.env
/opt/eva-ai/evo-ssearch/run_prod.sh
```

Сервис работает под пользователем `eva`:

```ini
[Service]
Type=simple
User=eva
Group=eva
WorkingDirectory=/opt/eva-ai/evo-ssearch
EnvironmentFile=/etc/eva-ai/eva-ai.env
ExecStart=/opt/eva-ai/evo-ssearch/run_prod.sh
Restart=on-failure
RestartSec=5
TimeoutStopSec=30
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=true
ReadWritePaths=/opt/eva-ai/evo-ssearch /var/lib/eva-ai /var/log/eva-ai
```

Для клиентского доступа по локальной сети control-plane поднят не только на
localhost:

```env
EVOSSEARCH_HOST=0.0.0.0
EVOSSEARCH_PORT=5000
EVOSSEARCH_DEBUG=false
EVOSSEARCH_GUNICORN_WORKERS=1
EVOSSEARCH_GUNICORN_THREADS=4
EVOSSEARCH_GUNICORN_TIMEOUT=240
```

`EVOSSEARCH_GUNICORN_WORKERS` должен оставаться `1`. Live capture и probe
runtime пока in-process; несколько gunicorn workers могут раздвоить runtime
state.

## 4. EVA AI inference profiles

Фактическая схема profiles:

```env
EVOSSEARCH_LM_PROFILES=agent,vlm-a1,vlm-a0,vlm-b1,vlm-b0
EVOSSEARCH_LM_AGENT_PROFILE_ID=agent
EVOSSEARCH_LM_VLM_PROFILE_ID=vlm-a1
EVOSSEARCH_LM_VLM_BALANCER_ENABLED=true
EVOSSEARCH_LM_VLM_BALANCER_PROFILES=vlm-a1,vlm-a0,vlm-b1,vlm-b0
```

Agent profile:

```env
EVOSSEARCH_LM_PROFILE_AGENT_KIND=agent
EVOSSEARCH_LM_PROFILE_AGENT_BASE_URL=http://127.0.0.1:1234/v1
EVOSSEARCH_LM_PROFILE_AGENT_MODEL=<agent-model-id>
EVOSSEARCH_LM_PROFILE_AGENT_API_KEY=
EVOSSEARCH_LM_PROFILE_AGENT_TIMEOUT=600
EVOSSEARCH_LM_PROFILE_AGENT_ENABLED=true
```

VLM profiles:

```env
EVOSSEARCH_LM_PROFILE_VLM_A1_KIND=vlm
EVOSSEARCH_LM_PROFILE_VLM_A1_BASE_URL=http://192.168.3.104:8001/v1
EVOSSEARCH_LM_PROFILE_VLM_A1_MODEL=qwen3-vl-4b-fp8
EVOSSEARCH_LM_PROFILE_VLM_A1_API_KEY=
EVOSSEARCH_LM_PROFILE_VLM_A1_TIMEOUT=240
EVOSSEARCH_LM_PROFILE_VLM_A1_ENABLED=true
EVOSSEARCH_LM_PROFILE_VLM_A1_GPU=server-a:1

EVOSSEARCH_LM_PROFILE_VLM_A0_KIND=vlm
EVOSSEARCH_LM_PROFILE_VLM_A0_BASE_URL=http://192.168.3.104:8002/v1
EVOSSEARCH_LM_PROFILE_VLM_A0_MODEL=qwen3-vl-4b-fp8
EVOSSEARCH_LM_PROFILE_VLM_A0_API_KEY=
EVOSSEARCH_LM_PROFILE_VLM_A0_TIMEOUT=240
EVOSSEARCH_LM_PROFILE_VLM_A0_ENABLED=true
EVOSSEARCH_LM_PROFILE_VLM_A0_GPU=server-a:0

EVOSSEARCH_LM_PROFILE_VLM_B1_KIND=vlm
EVOSSEARCH_LM_PROFILE_VLM_B1_BASE_URL=http://192.168.3.11:8001/v1
EVOSSEARCH_LM_PROFILE_VLM_B1_MODEL=qwen3-vl-4b-fp8
EVOSSEARCH_LM_PROFILE_VLM_B1_API_KEY=
EVOSSEARCH_LM_PROFILE_VLM_B1_TIMEOUT=240
EVOSSEARCH_LM_PROFILE_VLM_B1_ENABLED=true
EVOSSEARCH_LM_PROFILE_VLM_B1_GPU=server-b:1

EVOSSEARCH_LM_PROFILE_VLM_B0_KIND=vlm
EVOSSEARCH_LM_PROFILE_VLM_B0_BASE_URL=http://192.168.3.11:8002/v1
EVOSSEARCH_LM_PROFILE_VLM_B0_MODEL=qwen3-vl-4b-fp8
EVOSSEARCH_LM_PROFILE_VLM_B0_API_KEY=
EVOSSEARCH_LM_PROFILE_VLM_B0_TIMEOUT=240
EVOSSEARCH_LM_PROFILE_VLM_B0_ENABLED=true
EVOSSEARCH_LM_PROFILE_VLM_B0_GPU=server-b:0
```

Важно: текущий balancer - статическое распределение по channel id/profile id,
а не динамический least-loaded scheduler и не health-aware failover. На малом
числе каналов распределение может быть неровным; на 50 каналах оно
распределяется лучше, но при проблемном endpoint-е канал может зависнуть на
этом profile до ручного вмешательства.

## 5. Luxriot Evo integration

Фактический client-site endpoint во время пилота:

```env
EVOSSEARCH_LUXRIOT_BASE_URL=http://192.168.3.27:8080
EVOSSEARCH_LUXRIOT_USERNAME=admin
EVOSSEARCH_LUXRIOT_PASSWORD=<site-secret>
EVOSSEARCH_LUXRIOT_DEFAULT_CHANNEL_ID=105
EVOSSEARCH_LUXRIOT_SNAPSHOT_INTERVAL=1
EVOSSEARCH_LUXRIOT_SNAPSHOT_MAX_EDGE=800
EVOSSEARCH_LUXRIOT_MAX_BUFFER_FRAMES=120
EVOSSEARCH_LUXRIOT_AUTO_BOOKMARKS=true
EVOSSEARCH_LUXRIOT_SEV_INFO=info
EVOSSEARCH_LUXRIOT_SEV_LOW=low
EVOSSEARCH_LUXRIOT_SEV_NORMAL=normal
EVOSSEARCH_LUXRIOT_SEV_HIGH=high
EVOSSEARCH_LUXRIOT_SEV_CRITICAL=critical
```

Проверка:

```bash
curl -sS http://127.0.0.1:5000/ready \
  | jq '.status, .checks.luxriot, .checks.lm_profiles'
```

Проверка channel list требует terminal login, см. раздел про auth cookies.

## 6. PostgreSQL, auth, audit, archive

Пилот переведен с disposable SQLite/JSON подхода на PostgreSQL control-plane.
Ожидаемая Alembic head revision:

```text
20260614_0006
```

Ключевые настройки:

```env
EVOSSEARCH_SECURE_DEPLOYMENT_REQUIRED=true
EVOSSEARCH_AUTH_ENABLED=true
EVOSSEARCH_AUTH_TENANT_ID=<stable-tenant-uuid>
EVOSSEARCH_AUTH_COOKIE_SECURE=<true for HTTPS, false only for HTTP-only lab>
EVOSSEARCH_DB_STRICT_RUNTIME_ROLES=true

EVA_DATABASE_DSN=postgresql://eva_api_login:<secret>@127.0.0.1:5432/eva
EVA_AUDIT_DATABASE_DSN=postgresql://eva_audit_login:<secret>@127.0.0.1:5432/eva
EVA_WORKER_DATABASE_DSN=postgresql://eva_worker_login:<secret>@127.0.0.1:5432/eva
```

Runtime роли разделены:

- `eva_api_login` - API/runtime доступ;
- `eva_audit_login` - append-only audit writer;
- `eva_worker_login` - worker/runtime queue path;
- migration/bootstrap выполняются отдельно и не должны оставаться runtime DSN.

Readiness должен показывать:

- `postgresql.status = ready`;
- `postgresql.current_revision = 20260614_0006`;
- `postgresql.runtime_user = eva_api_login`;
- `authentication.status = ready`;
- `authentication.audit_runtime_user = eva_audit_login`.

Архив:

```env
EVOSSEARCH_ARCHIVE_STORE=postgres
EVOSSEARCH_ARCHIVE_TENANT_ID=<same-tenant-uuid>
EVOSSEARCH_ARCHIVE_RETENTION_ENABLED=true
EVOSSEARCH_ARCHIVE_MAX_RECORDS=5000000
EVOSSEARCH_ARCHIVE_ROW_RETENTION_DAYS=90
EVOSSEARCH_ARCHIVE_THUMBNAIL_RETENTION_DAYS=14
EVOSSEARCH_ARCHIVE_RETENTION_PRUNE_INTERVAL_SEC=3600
EVOSSEARCH_ARCHIVE_RETENTION_BATCH_SIZE=5000
EVOSSEARCH_ARCHIVE_ESTIMATE_CHANNELS=50
EVOSSEARCH_ARCHIVE_ESTIMATE_FRAMES_PER_BATCH=2.5
EVOSSEARCH_ARCHIVE_ESTIMATE_AVG_JPEG_KB=100
EVOSSEARCH_ARCHIVE_ESTIMATE_PROBE_RECORDS_PER_CHANNEL_DAY=250
EVOSSEARCH_DETECTIONS_ARCHIVE_DIR=/var/lib/eva-ai/detections_archive
EVOSSEARCH_DETECTIONS_RETENTION_DROP_SKIPPED=true
EVOSSEARCH_LUXRIOT_SUMMARY_RETENTION_DAYS=7
EVOSSEARCH_LUXRIOT_SUMMARY_HISTORY_LIMIT=10080
```

Фактически в Archive Research попадают:

- probe hits;
- sampled frames из live video-description batch;
- alert anchor frames, если VLM summary содержит alert.

## 7. Важная операционная особенность: live sessions

После рестарта `eva-ai` live video-description sessions не восстанавливаются
сами. Это in-process runtime state.

После каждого reboot/restart нужно:

1. запустить `eva-ai.service`;
2. проверить `/ready`;
3. залогиниться в UI;
4. вручную включить summaries на нужных каналах или запустить bulk curl;
5. проверить `/luxriot/streams`.

Пример bulk start:

```bash
CSRF="$(awk '$6=="eva_csrf" {print $7}' /tmp/eva.cookies | tail -1)"

for ch in 105 109 110 111; do
  echo "Starting channel ${ch}"
  curl -sS -b /tmp/eva.cookies \
    -H "X-CSRF-Token: ${CSRF}" \
    -H 'Content-Type: application/json' \
    -X POST http://127.0.0.1:5000/luxriot/start_capture \
    -d "{\"channel_id\":${ch},\"batch_size\":12,\"model\":\"__auto__\",\"prompt\":\"Describe visible activity, people, vehicles, objects, and notable changes.\"}" \
    | jq '{success, channel_id:.session.channel_id, model:.session.model, assigned:.session.assigned_profile_id, running:.session.running}'
done
```

Заменить список каналов на реальные channel IDs площадки.

## 8. Проверка auto-balancer и активных каналов

```bash
curl -sS -b /tmp/eva.cookies \
  http://127.0.0.1:5000/luxriot/streams \
  | jq -r '.video_streams[] | [.channel_id,.model,.pending_frames,.queue_submissions,.queue_dropped_batches,.last_error] | @tsv'

curl -sS -b /tmp/eva.cookies \
  http://127.0.0.1:5000/luxriot/streams \
  | jq -r '.video_streams[].model' | sort | uniq -c
```

Нормальное состояние:

- `pending_frames` периодически растет и сбрасывается;
- `queue_dropped_batches` не растет пачками;
- `last_error = null`;
- `logs_total` по session detail больше нуля после первого batch;
- archive counters в summary log показывают `archive_inserted`.

Если один канал стоит на `pending_frames: 120`, смотреть его `model`/profile и
проверять конкретный endpoint напрямую.

## 9. Terminal auth для curl

Браузерная сессия не создает `/tmp/eva.cookies`. Для terminal curl нужен
отдельный login:

```bash
read -rsp "EVA admin password: " EVA_PASSWORD
echo

curl -sS -c /tmp/eva.cookies \
  -H 'Content-Type: application/json' \
  -X POST http://127.0.0.1:5000/auth/login \
  -d "{\"username\":\"admin\",\"password\":\"${EVA_PASSWORD}\"}" \
  | jq '.success, .user.username, .user.allowedChannelIds'

unset EVA_PASSWORD
```

CSRF для mutating requests:

```bash
CSRF="$(awk '$6=="eva_csrf" {print $7}' /tmp/eva.cookies | tail -1)"
```

Проверка текущего пользователя:

```bash
curl -sS -b /tmp/eva.cookies http://127.0.0.1:5000/auth/me \
  | jq '.user | {username,roles,allowedChannelIds,permissions}'
```

Для setup/admin сценариев ожидается:

```json
"allowedChannelIds": ["*"]
```

Агентные tools фильтруют каналы по `allowedChannelIds`. Если агент "не видит"
канал, это может быть не LLM-ошибка, а IAM/channel visibility.

## 10. Очистка тестовых данных перед передачей клиенту

Перед передачей системы клиенту мы обсуждали clean operational purge:
сохраняем пользователей/роли/конфиг, но удаляем тестовые detections, VLM
frames, saved probes, runtime summary state, agent chats, queued jobs, sessions
и JPEG previews.

Перед purge обязательно остановить сервис и сделать backup:

```bash
sudo systemctl stop eva-ai

sudo -u postgres pg_dump -Fc -d eva -f /tmp/eva-before-client-clean.dump
sudo mkdir -p /var/lib/eva-ai/backups
sudo mv /tmp/eva-before-client-clean.dump /var/lib/eva-ai/backups/
sudo ls -lh /var/lib/eva-ai/backups/eva-before-client-clean.dump
```

Операционный purge:

```sql
BEGIN;

TRUNCATE TABLE
  archive.detections,
  archive.probes,
  archive.runtime_state,
  agent.action_approvals,
  agent.action_plans,
  agent.tool_runs,
  agent.messages,
  agent.sessions,
  jobs.job_attempts,
  jobs.outbox,
  jobs.inference_jobs,
  iam.sessions,
  iam.login_attempts
RESTART IDENTITY CASCADE;

COMMIT;
```

Очистка JPEG/previews:

```bash
sudo bash -lc '
  set -a
  . /etc/eva-ai/eva-ai.env
  set +a

  ARCHIVE_DIR="${EVOSSEARCH_DETECTIONS_ARCHIVE_DIR:-/var/lib/eva-ai/detections_archive}"

  case "$ARCHIVE_DIR" in
    /*) ;;
    *) ARCHIVE_DIR="/opt/eva-ai/evo-ssearch/$ARCHIVE_DIR" ;;
  esac

  echo "Cleaning archive dir: $ARCHIVE_DIR"

  if [ -d "$ARCHIVE_DIR" ] && [ "$ARCHIVE_DIR" != "/" ]; then
    find "$ARCHIVE_DIR" -mindepth 1 -delete
  fi
'
```

Audit можно чистить только до начала реальной клиентской эксплуатации:

```sql
TRUNCATE TABLE audit.events RESTART IDENTITY CASCADE;
```

После purge:

```bash
sudo systemctl start eva-ai
sleep 10
curl -sS http://127.0.0.1:5000/ready \
  | jq '.status, .checks.postgresql, .checks.authentication, .checks.luxriot'
```

Так как `iam.sessions` очищается, пользователи должны залогиниться заново.

## 11. Известные ловушки из фактического развертывания

### `status=217/USER` у vLLM service

Причина: в unit-файле указан неправильный Linux user. На `Luxriot2` фактический
пользователь был `pc2`, а unit был создан с `User=pc`.

Проверка:

```bash
id -un
id -gn
sudo systemctl cat eva-vllm-gpu0 --no-pager
sudo systemctl cat eva-vllm-gpu1 --no-pager
```

`User=` и `Group=` должны совпадать с фактическим account на этой машине.

### `/v1/models` показывает `max_model_len: 4096`

Причина: старый manual vLLM process еще слушает порт, даже если systemd unit
уже переписан на `8192`.

Починка:

```bash
sudo systemctl stop eva-vllm-gpu1 eva-vllm-gpu0 2>/dev/null || true

for port in 8001 8002; do
  pids="$(sudo ss -ltnp "sport = :${port}" | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u)"
  if [ -n "$pids" ]; then
    echo "Killing stale listener(s) on port ${port}: ${pids}"
    sudo kill $pids || true
  fi
done

sleep 5
sudo systemctl restart eva-vllm-gpu1 eva-vllm-gpu0
```

### `systemctl status` визуально обрезает длинный `ExecStart`

В `status` строка может заканчиваться символом `>`. Это может быть только
визуальное сокращение, а может быть реально сломанный unit после плохого
copy-paste. Проверять так:

```bash
sudo systemctl cat eva-vllm-gpu1 --no-pager
sudo systemctl status eva-vllm-gpu1 --no-pager -l
```

### Hugging Face download/XET stall

Симптом: download висит часами, `.incomplete` файлы не растут.

Рабочий подход:

```bash
find /opt/eva-vllm/hf/hub/.locks -type f -delete 2>/dev/null || true
export HF_HOME=/opt/eva-vllm/hf
export HF_HUB_DISABLE_XET=1

hf download Qwen/Qwen3-VL-4B-Instruct-FP8 \
  --local-dir /opt/eva-vllm/models/qwen3-vl-4b-fp8
```

Если одна inference-машина уже скачала модель, предпочтительнее копировать
через `rsync` внутри локальной сети.

### FlashInfer sampler требует `nvcc`

Симптом:

```text
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

Фикс уже в systemd unit:

```ini
Environment=VLLM_USE_FLASHINFER_SAMPLER=0
```

### vLLM `image 8` ломает live batch из 12 кадров

Симптомы:

- live stream виден;
- frames копятся;
- `pending_frames` доходит до `120`;
- summaries не появляются;
- канал может выглядеть "running", но `logs_total = 0`.

Причина: EVA AI отправляет live batch из 12 кадров, а vLLM был поднят с:

```text
--limit-mm-per-prompt.image 8
```

Финальный фикс:

```text
--max-model-len 8192
--limit-mm-per-prompt.image 16
```

### Channel visibility и `allowedChannelIds`

Если UI или агент видит не все каналы:

```bash
curl -sS -b /tmp/eva.cookies http://127.0.0.1:5000/auth/me \
  | jq '.user.allowedChannelIds'

curl -sS -b /tmp/eva.cookies \
  "http://127.0.0.1:5000/luxriot/channels?force=1" \
  | jq '.channels[] | {id,title}'
```

Для setup проще использовать admin с:

```json
["*"]
```

Если channel отсутствует даже у admin, смотреть Luxriot credentials, Luxriot
permissions и `EVOSSEARCH_LUXRIOT_BASE_URL`.

### Случайный `channel_id: 1`

Во время ручного curl был оставлен placeholder `PUT_4TH_CHANNEL_ID_HERE`.
Система в итоге подняла лишнюю session на `channel_id: 1`. Перед клиентским
показом проверять и останавливать мусорные sessions:

```bash
CSRF="$(awk '$6=="eva_csrf" {print $7}' /tmp/eva.cookies | tail -1)"

curl -sS -b /tmp/eva.cookies \
  -H "X-CSRF-Token: ${CSRF}" \
  -H 'Content-Type: application/json' \
  -X POST http://127.0.0.1:5000/luxriot/stop_capture \
  -d '{"channel_id":1}' | jq
```

### Raw model name вместо profile id

Для live descriptions на этом пилоте безопаснее выбирать `Auto balance` или
конкретный VLM profile (`vlm-a1`, `vlm-a0`, `vlm-b1`, `vlm-b0`), а не raw model
name. Agent/text model и VLM model имеют разные endpoint-ы и разную
способность принимать изображения.

### Terminal copy-paste

Во время развертывания в shell несколько раз попал поясняющий текст вроде
`Now clear stale processes` или `If one fails...`, что приводило к
`command not found`. Для операторов в runbook-ах нужно копировать только
закрытые code blocks.

## 12. Временные UI-ограничения перед закрытой сетью

Перед передачей в закрытую сеть обсуждалось скрыть frontend-only две функции,
которые не были обещаны клиенту и требуют отдельного тестирования:

- Offline Video Analysis;
- Probe Snap.

Безопасный ручной вариант на установленной машине - не удалять HTML, а только
добавить inline `style="display: none;"`:

```html
<div class="video-analysis-shell studio-panel" style="display: none;">
<button id="probeSnapBtn" type="button" class="feature-btn" style="display: none;">Snap</button>
<div id="probeSnapModal" class="settings-modal" style="display: none;">
```

Важно: не удалять `videoModel` и `videoPrompt` из DOM, потому что Archive
Research description path использует общую функцию `describeImageWithLM(...)`
и endpoint `/describe_image`. Скрытие блока не должно ломать рабочую функцию
"описать картинку/detection image" из архива.

После ручного изменения `templates/index.html` на установленной машине:

```bash
sudo systemctl restart eva-ai
```

Затем в браузере сделать hard refresh.

## 13. Минимальный checklist после перевозки или смены IP

1. На обеих inference-машинах запустить `eva-vllm-gpu1` и `eva-vllm-gpu0`.
2. Проверить `8001/8002`, модель `qwen3-vl-4b-fp8` и `max_model_len=8192`.
3. На EVA AI машине обновить `/etc/eva-ai/eva-ai.env`:
   - `EVOSSEARCH_LUXRIOT_BASE_URL`;
   - `EVOSSEARCH_LM_PROFILE_AGENT_BASE_URL`;
   - `EVOSSEARCH_LM_PROFILE_VLM_A1_BASE_URL`;
   - `EVOSSEARCH_LM_PROFILE_VLM_A0_BASE_URL`;
   - `EVOSSEARCH_LM_PROFILE_VLM_B1_BASE_URL`;
   - `EVOSSEARCH_LM_PROFILE_VLM_B0_BASE_URL`.
4. Перезапустить `postgresql` и `eva-ai`.
5. Проверить `/health` и `/ready`.
6. Залогиниться terminal curl-ом, проверить `/luxriot/channels?force=1`.
7. Запустить live summaries заново.
8. Проверить `/luxriot/streams` и распределение по profiles.
9. Проверить Archive Research: VLM frames и probe hits различаются по source.
10. Проверить, что пользователи/roles/channel grants соответствуют клиентской
    схеме доступа.
