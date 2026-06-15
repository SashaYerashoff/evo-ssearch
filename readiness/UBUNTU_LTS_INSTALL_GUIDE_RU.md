# EVA AI: установка control-plane на чистую Ubuntu Server LTS

Дата: 2026-06-10

## Scope

Этот runbook поднимает EVA AI web/API, PostgreSQL, пользователей, роли,
audit и systemd-service на чистой Ubuntu Server LTS. Он подходит для Ubuntu
24.04 LTS и 26.04 LTS. Настройка LLM/VLM inference, llama.cpp/LM Studio/Ollama,
CUDA-бенчей и лицензий моделей здесь намеренно вынесена за скобки.

На дату документа последняя стабильная Ubuntu Server LTS на официальном сайте
Canonical — Ubuntu 26.04 LTS. Canonical указывает, что LTS-релизы выходят раз
в два года и имеют 5 лет стандартного security maintenance.

Ссылки:

- https://ubuntu.com/download/server
- https://ubuntu.com/about/release-cycle

## Целевая схема

- OS: Ubuntu Server 24.04 LTS или 26.04 LTS, amd64.
- App user: `eva`.
- App path: `/opt/eva-ai/evo-ssearch`.
- Data path: `/var/lib/eva-ai`.
- Runtime env: `/etc/eva-ai/eva-ai.env`, `0600`, owner `root`.
- PostgreSQL: локально на том же сервере для первого пилота.
- Web: Gunicorn через `run_prod.sh`, один worker, несколько threads.
- HTTPS: Nginx reverse proxy. Для production использовать нормальный TLS
  сертификат; self-signed допустим только для закрытого стенда.

## 1. Базовая подготовка Ubuntu

```bash
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y \
  ca-certificates curl git jq openssl \
  python3 python3-venv python3-pip \
  build-essential cmake pkg-config \
  ffmpeg libglib2.0-0 libgl1 \
  postgresql postgresql-contrib \
  nginx
sudo reboot
```

После reboot:

```bash
uname -a
python3 --version
systemctl status postgresql --no-pager
```

GPU-драйверы для inference на этом шаге не настраиваем. Если на сервере уже
должны быть NVIDIA-драйверы для CLIP/DINO/GPU-проверок, минимальная проверка:

```bash
nvidia-smi
```

## 2. Системный пользователь и директории

```bash
sudo adduser --system --group --home /opt/eva-ai eva
sudo install -d -o eva -g eva -m 0755 /opt/eva-ai
sudo install -d -o eva -g eva -m 0750 /var/lib/eva-ai
sudo install -d -o eva -g eva -m 0750 /var/lib/eva-ai/detections_archive
sudo install -d -o eva -g eva -m 0750 /var/log/eva-ai
sudo install -d -o root -g root -m 0750 /etc/eva-ai
```

## 3. Код и Python-окружение

Подставить реальный URL репозитория/ветки:

```bash
sudo -u eva git clone <REPO_URL> /opt/eva-ai/evo-ssearch
cd /opt/eva-ai/evo-ssearch
sudo -u eva git checkout feature/secure-50-channel-foundation

sudo -u eva python3 -m venv .venv
sudo -u eva .venv/bin/python -m pip install --upgrade pip setuptools wheel
sudo -u eva .venv/bin/pip install -r requirements.txt -r requirements-db.txt
```

Быстрая проверка без live PostgreSQL:

```bash
sudo -u eva -H bash -lc \
  'cd /opt/eva-ai/evo-ssearch && .venv/bin/python -m unittest discover tests'
```

## 4. PostgreSQL: база, миграции, runtime-роли

Создать пустую базу:

```bash
sudo -u postgres createdb eva
```

Прогнать миграции привилегированным локальным подключением:

```bash
cd /opt/eva-ai/evo-ssearch
sudo -u postgres env \
  EVA_DATABASE_DSN='postgresql:///eva?host=/var/run/postgresql' \
  .venv/bin/alembic upgrade head
```

Сгенерировать пароли login-ролей. Используем `hex`, чтобы DSN не требовал URL
encoding:

```bash
export EVA_MIGRATOR_PASSWORD="$(openssl rand -hex 32)"
export EVA_API_PASSWORD="$(openssl rand -hex 32)"
export EVA_AUDIT_PASSWORD="$(openssl rand -hex 32)"
export EVA_WORKER_PASSWORD="$(openssl rand -hex 32)"
export EVA_BACKUP_PASSWORD="$(openssl rand -hex 32)"
```

Создать least-privilege login-роли:

```bash
cd /opt/eva-ai/evo-ssearch
sudo -u postgres env \
  EVA_DATABASE_DSN='postgresql:///eva?host=/var/run/postgresql' \
  EVA_MIGRATOR_PASSWORD="$EVA_MIGRATOR_PASSWORD" \
  EVA_API_PASSWORD="$EVA_API_PASSWORD" \
  EVA_AUDIT_PASSWORD="$EVA_AUDIT_PASSWORD" \
  EVA_WORKER_PASSWORD="$EVA_WORKER_PASSWORD" \
  EVA_BACKUP_PASSWORD="$EVA_BACKUP_PASSWORD" \
  .venv/bin/python scripts/bootstrap_db_roles.py
```

## 5. Runtime env

Сгенерировать tenant UUID:

```bash
export EVOSSEARCH_AUTH_TENANT_ID="$(python3 - <<'PY'
import uuid
print(uuid.uuid4())
PY
)"
```

Создать `/etc/eva-ai/eva-ai.env`:

```bash
sudo tee /etc/eva-ai/eva-ai.env >/dev/null <<EOF
EVOSSEARCH_HOST=127.0.0.1
EVOSSEARCH_PORT=5000
EVOSSEARCH_DEBUG=false
EVOSSEARCH_GUNICORN_WORKERS=1
EVOSSEARCH_GUNICORN_THREADS=4
EVOSSEARCH_GUNICORN_TIMEOUT=240

EVOSSEARCH_EMBEDDER=clip
EVOSSEARCH_INDEX_MODE=clip
EVOSSEARCH_DINO_SEGMENTS_ENABLED=false
EVOSSEARCH_M2F_ENABLED=false

EVOSSEARCH_SECURE_DEPLOYMENT_REQUIRED=true
EVOSSEARCH_AUTH_ENABLED=true
EVOSSEARCH_AUTH_TENANT_ID=${EVOSSEARCH_AUTH_TENANT_ID}
EVOSSEARCH_AUTH_COOKIE_SECURE=true
EVOSSEARCH_DB_STRICT_RUNTIME_ROLES=true

EVA_DATABASE_DSN=postgresql://eva_api_login:${EVA_API_PASSWORD}@127.0.0.1:5432/eva
EVA_AUDIT_DATABASE_DSN=postgresql://eva_audit_login:${EVA_AUDIT_PASSWORD}@127.0.0.1:5432/eva
EVA_WORKER_DATABASE_DSN=postgresql://eva_worker_login:${EVA_WORKER_PASSWORD}@127.0.0.1:5432/eva

EVOSSEARCH_ARCHIVE_STORE=postgres
EVOSSEARCH_ARCHIVE_TENANT_ID=${EVOSSEARCH_AUTH_TENANT_ID}
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

# Luxriot Evo integration. Заменить адрес, учетку и default channel под площадку.
EVOSSEARCH_LUXRIOT_BASE_URL=http://LUXRIOT_EVO_HOST:9090
EVOSSEARCH_LUXRIOT_USERNAME=admin
EVOSSEARCH_LUXRIOT_PASSWORD=CHANGE_ME
EVOSSEARCH_LUXRIOT_DEFAULT_CHANNEL_ID=
EVOSSEARCH_LUXRIOT_SNAPSHOT_INTERVAL=1
EVOSSEARCH_LUXRIOT_SNAPSHOT_MAX_EDGE=800
EVOSSEARCH_LUXRIOT_MAX_BUFFER_FRAMES=120
EVOSSEARCH_LUXRIOT_AUTO_BOOKMARKS=true
EVOSSEARCH_LUXRIOT_SEV_INFO=info
EVOSSEARCH_LUXRIOT_SEV_LOW=low
EVOSSEARCH_LUXRIOT_SEV_NORMAL=normal
EVOSSEARCH_LUXRIOT_SEV_HIGH=high
EVOSSEARCH_LUXRIOT_SEV_CRITICAL=critical

EVOSSEARCH_LUXRIOT_SUMMARY_RETENTION_DAYS=7
EVOSSEARCH_LUXRIOT_SUMMARY_HISTORY_LIMIT=10080
EVOSSEARCH_ALLOWED_ROOTS=/var/lib/eva-ai
EVOSSEARCH_SETTINGS_LOCAL_ONLY=true
EVOSSEARCH_CORS_ALLOWED_ORIGINS=

# Inference placeholder. Настроить отдельно перед включением VLM/agent flows.
EVOSSEARCH_LM_BASE_URL=http://127.0.0.1:1234/v1
EVOSSEARCH_LM_MODEL=
EVOSSEARCH_LM_API_KEY=

# OpenAI-compatible inference profiles.
# Для демо-схемы: один endpoint под агента и четыре VLM endpoint-а под live summaries.
# Profile id `vlm-1` в EVOSSEARCH_LM_PROFILES превращается в env prefix
# EVOSSEARCH_LM_PROFILE_VLM_1_*.
# EVOSSEARCH_LM_PROFILES=agent,vlm-1,vlm-2,vlm-3,vlm-4
# EVOSSEARCH_LM_AGENT_PROFILE_ID=agent
# EVOSSEARCH_LM_VLM_PROFILE_ID=vlm-1
# EVOSSEARCH_LM_VLM_BALANCER_ENABLED=true
# EVOSSEARCH_LM_VLM_BALANCER_PROFILES=vlm-1,vlm-2,vlm-3,vlm-4
#
# EVOSSEARCH_LM_PROFILE_AGENT_KIND=agent
# EVOSSEARCH_LM_PROFILE_AGENT_BASE_URL=http://<agent-host>:1234/v1
# EVOSSEARCH_LM_PROFILE_AGENT_MODEL=<agent-model-id>
# EVOSSEARCH_LM_PROFILE_AGENT_API_KEY=
# EVOSSEARCH_LM_PROFILE_AGENT_TIMEOUT=600
# EVOSSEARCH_LM_PROFILE_AGENT_ENABLED=true
#
# EVOSSEARCH_LM_PROFILE_VLM_1_KIND=vlm
# EVOSSEARCH_LM_PROFILE_VLM_1_BASE_URL=http://<vlm-1-host>:1234/v1
# EVOSSEARCH_LM_PROFILE_VLM_1_MODEL=<vlm-model-id>
# EVOSSEARCH_LM_PROFILE_VLM_1_API_KEY=
# EVOSSEARCH_LM_PROFILE_VLM_1_TIMEOUT=240
# EVOSSEARCH_LM_PROFILE_VLM_1_ENABLED=true
# EVOSSEARCH_LM_PROFILE_VLM_1_GPU=server-a:0
#
# Повторить VLM_2/VLM_3/VLM_4 для остальных endpoints.
EOF

sudo chmod 0600 /etc/eva-ai/eva-ai.env
sudo chown root:root /etc/eva-ai/eva-ai.env
```

Для прямого доступа к web/API по локальной сети без nginx заменить
`EVOSSEARCH_HOST=127.0.0.1` на `EVOSSEARCH_HOST=0.0.0.0`.

Важно: `EVOSSEARCH_AUTH_COOKIE_SECURE=true` требует HTTPS с точки зрения
браузера. Для локального HTTP-only стенда временно можно поставить `false`, но
не для клиентского демо. Если EVA AI открывается через Luxriot Evo Monitor
crosslink по HTTP, это тоже считается HTTP-only режимом для cookie.

Офисный пример, который использовался при проверке: Luxriot Evo на
`http://192.168.2.180:9090`; в том стенде default channel был не старый `112`,
а актуальный ID из списка `/channels`. Перед деплоем всегда проверять реальные
ID каналов через UI или `/channels`.

## 6. Первый администратор

```bash
sudo bash -lc '
  set -a
  . /etc/eva-ai/eva-ai.env
  set +a
  cd /opt/eva-ai/evo-ssearch
  sudo -u eva -E .venv/bin/python scripts/bootstrap_admin.py \
    --tenant-id "$EVOSSEARCH_AUTH_TENANT_ID" \
    --username admin \
    --display-name "EVA Admin"
'
```

Пароль вводится интерактивно и не попадает в shell history.

Создание пилотного оператора на каналы 1-50:

```bash
read -rsp 'Operator temporary password: ' EVA_USER_PASSWORD
echo
export EVA_USER_PASSWORD
CHANNELS_1_50="$(seq -s, 1 50)"
sudo EVA_USER_PASSWORD="$EVA_USER_PASSWORD" CHANNELS_1_50="$CHANNELS_1_50" bash -lc '
  set -a
  . /etc/eva-ai/eva-ai.env
  set +a
  cd /opt/eva-ai/evo-ssearch
  sudo -u eva -E \
    EVA_USER_PASSWORD="$EVA_USER_PASSWORD" \
    EVA_ADMIN_USERNAME=admin \
    .venv/bin/python scripts/manage_users.py create operator-1 \
    --role operator \
    --channels "$CHANNELS_1_50"
'
unset EVA_USER_PASSWORD
```

## 7. systemd service

```bash
sudo tee /etc/systemd/system/eva-ai.service >/dev/null <<'EOF'
[Unit]
Description=EVA AI control-plane
After=network-online.target postgresql.service
Wants=network-online.target

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

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now eva-ai
sudo systemctl status eva-ai --no-pager
```

Проверка:

```bash
curl -sS http://127.0.0.1:5000/ready | jq
journalctl -u eva-ai -n 100 --no-pager
```

В secure режиме `/ready` должен быть `ok=true`; если нет, смотреть компоненты
`authentication`, `postgresql`, `audit` и `runtime_role_*`.

## 8. Nginx reverse proxy

Для production поставить нормальный сертификат. Для закрытого демо можно
временно сделать self-signed:

```bash
sudo install -d -m 0750 /etc/eva-ai/tls
sudo openssl req -x509 -nodes -newkey rsa:4096 -days 30 \
  -keyout /etc/eva-ai/tls/eva-ai.key \
  -out /etc/eva-ai/tls/eva-ai.crt \
  -subj "/CN=eva-ai.local"
sudo chmod 0600 /etc/eva-ai/tls/eva-ai.key
```

Nginx site:

```bash
sudo tee /etc/nginx/sites-available/eva-ai >/dev/null <<'EOF'
server {
    listen 80;
    server_name _;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name _;

    ssl_certificate /etc/eva-ai/tls/eva-ai.crt;
    ssl_certificate_key /etc/eva-ai/tls/eva-ai.key;

    client_max_body_size 64m;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 300s;
        proxy_buffering off;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/eva-ai /etc/nginx/sites-enabled/eva-ai
sudo nginx -t
sudo systemctl reload nginx
```

Проверка с сервера:

```bash
curl -k -sS https://127.0.0.1/ready | jq
```

## 9. Smoke checklist перед клиентским демо

```bash
sudo -u eva -H bash -lc \
  'cd /opt/eva-ai/evo-ssearch && .venv/bin/python -m unittest discover tests'
sudo bash -lc '
  set -a
  . /etc/eva-ai/eva-ai.env
  set +a
  export EVA_TEST_DATABASE_DSN="$EVA_DATABASE_DSN"
  cd /opt/eva-ai/evo-ssearch
  sudo -u eva -E .venv/bin/python -m unittest tests.test_postgres_identity
'
curl -k -sS https://127.0.0.1/ready | jq
sudo bash -lc '
  set -a
  . /etc/eva-ai/eva-ai.env
  set +a
  cd /opt/eva-ai/evo-ssearch
  sudo -u eva -E EVA_ADMIN_USERNAME=admin .venv/bin/python scripts/manage_users.py list
'
```

Ручная проверка в UI:

- зайти через HTTPS;
- войти admin-пользователем;
- создать/проверить оператора с ограничением на 1-50 каналов;
- проверить, что `/auth/sessions` показывает сессии;
- сделать logout/login;
- убедиться, что audit DB пишет события login, route access и completion events.

Минимальный preflight для выезда:

```bash
curl -k -fsS https://127.0.0.1/health | jq -e '.status == "ok"'
curl -k -fsS 'https://127.0.0.1/ready?strict=true' | jq -e '
  .status == "ready"
  and (.checks.authentication.ok == true)
  and (.checks.postgresql.ok == true)
  and (.checks.lm_profiles.ok == true or .checks.lm_profiles.required == false)
  and (.checks.luxriot.ok == true)
'
sudo journalctl -u eva-ai -n 120 --no-pager
```

Для схемы с включённым `EVOSSEARCH_LM_VLM_BALANCER_ENABLED=true`
`checks.lm_profiles.required` должен быть `true`, а все profile id из
`checks.lm_profiles.required_profile_ids` должны иметь `ok=true`.

## 10. Что не забыть перед реальным rollout

- Заменить self-signed TLS на клиентский сертификат.
- Ограничить firewall: наружу только 443, SSH только с админских адресов.
- Сохранить `/etc/eva-ai/eva-ai.env` в клиентском secret vault, не в git.
- Настроить backup PostgreSQL и тест restore.
- Зафиксировать версию ветки/коммита, который ставится клиенту.
- Отдельно подключить inference endpoints и лицензии моделей.
- Для multi-server схемы вынести PostgreSQL/agent/VLM workers по ролям, но
  оставить тот же принцип: отдельные runtime DSN, audit append-only, named users.
