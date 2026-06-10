# EVA AI: установка control-plane на чистую Ubuntu Server LTS

Дата: 2026-06-10

## Scope

Этот runbook поднимает EVA AI web/API, PostgreSQL, пользователей, роли,
audit и systemd-service на чистой Ubuntu Server LTS. Настройка LLM/VLM
inference, llama.cpp/LM Studio/Ollama, CUDA-бенчей и лицензий моделей здесь
намеренно вынесена за скобки.

На дату документа последняя стабильная Ubuntu Server LTS на официальном сайте
Canonical — Ubuntu 26.04 LTS. Canonical указывает, что LTS-релизы выходят раз
в два года и имеют 5 лет стандартного security maintenance.

Ссылки:

- https://ubuntu.com/download/server
- https://ubuntu.com/about/release-cycle

## Целевая схема

- OS: Ubuntu Server 26.04 LTS, amd64.
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
sudo install -d -o eva -g eva -m 0750 /opt/eva-ai
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

EVOSSEARCH_SECURE_DEPLOYMENT_REQUIRED=true
EVOSSEARCH_AUTH_ENABLED=true
EVOSSEARCH_AUTH_TENANT_ID=${EVOSSEARCH_AUTH_TENANT_ID}
EVOSSEARCH_AUTH_COOKIE_SECURE=true
EVOSSEARCH_DB_STRICT_RUNTIME_ROLES=true

EVA_DATABASE_DSN=postgresql://eva_api_login:${EVA_API_PASSWORD}@127.0.0.1:5432/eva
EVA_AUDIT_DATABASE_DSN=postgresql://eva_audit_login:${EVA_AUDIT_PASSWORD}@127.0.0.1:5432/eva
EVA_WORKER_DATABASE_DSN=postgresql://eva_worker_login:${EVA_WORKER_PASSWORD}@127.0.0.1:5432/eva

EVOSSEARCH_DETECTIONS_ARCHIVE_DIR=/var/lib/eva-ai/detections_archive
EVOSSEARCH_ALLOWED_ROOTS=/var/lib/eva-ai
EVOSSEARCH_SETTINGS_LOCAL_ONLY=true
EVOSSEARCH_CORS_ALLOWED_ORIGINS=

# Inference placeholder. Настроить отдельно перед включением VLM/agent flows.
EVOSSEARCH_LM_BASE_URL=http://127.0.0.1:1234/v1
EVOSSEARCH_LM_MODEL=
EVOSSEARCH_LM_API_KEY=
EOF

sudo chmod 0600 /etc/eva-ai/eva-ai.env
sudo chown root:root /etc/eva-ai/eva-ai.env
```

Важно: `EVOSSEARCH_AUTH_COOKIE_SECURE=true` требует HTTPS с точки зрения
браузера. Для локального HTTP-only стенда временно можно поставить `false`, но
не для клиентского демо.

## 6. Первый администратор

```bash
cd /opt/eva-ai/evo-ssearch
set -a
. /etc/eva-ai/eva-ai.env
set +a
sudo -u eva -E .venv/bin/python scripts/bootstrap_admin.py \
  --tenant-id "$EVOSSEARCH_AUTH_TENANT_ID" \
  --username admin \
  --display-name "EVA Admin"
```

Пароль вводится интерактивно и не попадает в shell history.

Создание пилотного оператора на каналы 1-50:

```bash
cd /opt/eva-ai/evo-ssearch
set -a
. /etc/eva-ai/eva-ai.env
set +a
read -rsp 'Operator temporary password: ' EVA_USER_PASSWORD
echo
export EVA_USER_PASSWORD
CHANNELS_1_50="$(seq -s, 1 50)"
sudo -u eva -E .venv/bin/python scripts/manage_users.py create operator-1 \
  --role operator \
  --channels "$CHANNELS_1_50"
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
cd /opt/eva-ai/evo-ssearch
set -a
. /etc/eva-ai/eva-ai.env
set +a

sudo -u eva -E .venv/bin/python -m unittest discover tests
sudo -u eva -E .venv/bin/python -m unittest tests.test_postgres_identity
curl -k -sS https://127.0.0.1/ready | jq
sudo -u eva -E .venv/bin/python scripts/manage_users.py list
```

Ручная проверка в UI:

- зайти через HTTPS;
- войти admin-пользователем;
- создать/проверить оператора с ограничением на 1-50 каналов;
- проверить, что `/auth/sessions` показывает сессии;
- сделать logout/login;
- убедиться, что audit DB пишет события login, route access и completion events.

## 10. Что не забыть перед реальным rollout

- Заменить self-signed TLS на клиентский сертификат.
- Ограничить firewall: наружу только 443, SSH только с админских адресов.
- Сохранить `/etc/eva-ai/eva-ai.env` в клиентском secret vault, не в git.
- Настроить backup PostgreSQL и тест restore.
- Зафиксировать версию ветки/коммита, который ставится клиенту.
- Отдельно подключить inference endpoints и лицензии моделей.
- Для multi-server схемы вынести PostgreSQL/agent/VLM workers по ролям, но
  оставить тот же принцип: отдельные runtime DSN, audit append-only, named users.
