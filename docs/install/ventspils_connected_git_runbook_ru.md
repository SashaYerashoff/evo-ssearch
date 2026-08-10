# EVA AI β 0.8.5: проверенная установка Ventspils из Git на Ubuntu 24.04

Дата проверенной установки: **6 августа 2026 года**.

Этот runbook воспроизводит фактически работающую установку EVA AI на одном
Ubuntu-хосте с локальными PostgreSQL, VLM и deep-review. Это не инструкция для
offline USB installer и не инструкция обновления уже работающего стенда.

## 1. Зафиксированный профиль установки

| Компонент | Проверенное значение |
|---|---|
| Ubuntu | Ubuntu 24.04.4 LTS, amd64, kernel `6.8.0-136-generic` |
| CPU / RAM | Intel Core i7-14700K, 20 cores / 28 threads, 62 GiB RAM |
| GPU | 2 × NVIDIA RTX A4000, 16 GiB, driver `595.84` |
| Репозиторий | `https://github.com/SashaYerashoff/Luxriot-EVA-AI.git` |
| Ветка | `deploy/ventslpils-osta` (опечатка в имени ветки сохранена) |
| Коммит | `0316c5d4dc601fa29785ed2aa3f07ca86feaf9b7` |
| EVA AI | `β 0.8.5` |
| PostgreSQL | локальная БД `eva`, schema revision `20260727_0010` |
| VLM / Agent | Qwen3-VL-4B-Instruct AWQ-4bit, vLLM `0.25.0`, `127.0.0.1:1234` |
| Deep review | Qwen3.5-9B-MTP Q4_K_M, llama.cpp `b9330`, `127.0.0.1:1236` |
| EVA app | Gunicorn, `127.0.0.1:5000` |
| Внешний URL | Nginx HTTPS, порт `443` |
| Основные пути | `/opt/eva-ai`, `/var/lib/eva-ai`, `/etc/eva-ai` |

Установленный коммит содержит исходники `react-ui`, но в этой конкретной
сборке production assets React не собирались. В браузере ожидается legacy UI.

## 2. Перед началом

Понадобятся:

- sudo-доступ к Ubuntu;
- физический доступ к UEFI либо человек на площадке, если включён Secure Boot;
- доступ Ubuntu к GitHub, PyPI и Hugging Face;
- URL, логин и пароль Luxriot Evo;
- свободный TCP `443`; внутренние `1234`, `1236`, `5000` наружу не открываются;
- минимум 45 GiB свободного места, лучше 400–500 GiB под архив.

Не публиковать `/etc/eva-ai/eva-ai.env`, DB secrets, пароль Luxriot или пароль
администратора EVA в Git, почте, чатах и диагностических файлах.

### Важное ограничение по дискам

На проверенном сервере Ubuntu находилась на LVM-разделе HDD, а Samsung 990 PRO
содержал два NTFS-раздела. Для EVA они **не удалялись и не форматировались**.
Место добавили существующему root LV из свободного пространства `ubuntu-vg`.

Никогда не выполняйте команды изменения дисков, пока вывод ниже не совпадает с
ожидаемой схемой:

```bash
lsblk -o NAME,SIZE,FSTYPE,TYPE,MOUNTPOINTS,MODEL
sudo pvs
sudo vgs
sudo lvs -o lv_name,vg_name,lv_size,devices
df -hT /
```

На проверенном хосте целевой LV был `/dev/ubuntu-vg/ubuntu-lv`, а в VG было
около 1.72 TiB свободно. Его расширили до 500 GiB:

```bash
sudo lvextend -L 500G -r /dev/ubuntu-vg/ubuntu-lv
df -hT /
```

Если имя VG/LV или раскладка отличаются, эту команду не выполнять.

## 3. Инвентаризация хоста

```bash
echo '=== OS ==='
. /etc/os-release
echo "$PRETTY_NAME"
uname -m
uname -r

echo '=== CPU / RAM ==='
lscpu | grep -E 'Model name|Socket|Core|Thread|CPU\(s\)|Architecture'
free -h

echo '=== GPU / DISPLAY ==='
lspci -nnk -d 10de: | grep -E 'VGA|3D|Kernel driver in use|Kernel modules'
command -v nvidia-smi >/dev/null && nvidia-smi || true

echo '=== SECURE BOOT ==='
command -v mokutil >/dev/null && mokutil --sb-state || true

echo '=== PORTS ==='
sudo ss -lntup | grep -E ':(80|443|1234|1236|5000|5432)\b' || true
```

На проверенном хосте порт `80` уже занимал Nextcloud Apache. Его не отключали:
Nginx EVA слушает только `443`.

Рекомендуется закрепить DHCP lease для EVA-хоста. После любого reboot локально
узнать адрес можно одной командой:

```bash
hostname -I
```

## 4. NVIDIA driver и Secure Boot

RTX A4000 сначала работали через `nouveau`. Был установлен пакет:

```bash
sudo apt-get update
sudo apt-get install -y nvidia-driver-595-open mokutil
```

При включённом Secure Boot неподписанный DKMS-модуль не загрузится:

```text
modprobe: ERROR: could not insert 'nvidia': Key was rejected by service
```

Есть два варианта, оба требуют действий при физической загрузке:

1. зарегистрировать созданный MOK в синем UEFI/MOK Manager;
2. отключить Secure Boot в UEFI — так сделано на этом стенде.

После изменения UEFI и reboot:

```bash
mokutil --sb-state
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
lspci -nnk -d 10de: | grep -E 'VGA|3D|Kernel driver in use'
```

Ожидаются две RTX A4000, а `Kernel driver in use` — `nvidia`. Пока
`nvidia-smi` не работает, к установке vLLM не переходить.

## 5. Системные пакеты, пользователь и каталоги

```bash
sudo apt-get update
sudo apt-get install -y \
  ca-certificates curl git jq openssl acl rsync \
  python3 python3-venv python3-pip \
  build-essential cmake ninja-build pkg-config \
  ffmpeg libglib2.0-0 libgl1 \
  postgresql postgresql-contrib libpq-dev \
  nginx

getent group eva >/dev/null || sudo groupadd --system eva
id -u eva >/dev/null 2>&1 || sudo useradd --system \
  --gid eva \
  --home-dir /var/lib/eva-ai \
  --create-home \
  --shell /usr/sbin/nologin \
  eva

sudo install -d -o eva  -g eva -m 0755 /opt/eva-ai
sudo install -d -o eva  -g eva -m 0750 /var/lib/eva-ai
sudo install -d -o root -g eva -m 0750 /etc/eva-ai
sudo install -d -o eva  -g eva -m 0750 \
  /var/lib/eva-ai/models \
  /var/lib/eva-ai/models/huggingface \
  /var/lib/eva-ai/models/clip \
  /var/lib/eva-ai/detections_archive \
  /var/lib/eva-ai/inference-spool \
  /var/lib/eva-ai/state
```

Если установка `nginx` пожаловалась на занятый порт 80, это ожидаемо при
существующем Apache. Сначала выполнить раздел 12, затем:

```bash
sudo dpkg --configure -a
sudo systemctl restart nginx
```

## 6. Установка точного кода EVA AI

Ветка изменяема, поэтому после клонирования обязательно закрепить проверенный
commit:

```bash
sudo -u eva git clone \
  --branch deploy/ventslpils-osta \
  --single-branch \
  https://github.com/SashaYerashoff/Luxriot-EVA-AI.git \
  /opt/eva-ai/app

sudo -u eva git -C /opt/eva-ai/app checkout \
  0316c5d4dc601fa29785ed2aa3f07ca86feaf9b7

sudo -u eva git -C /opt/eva-ai/app status --short --branch
sudo -u eva git -C /opt/eva-ai/app rev-parse HEAD
cat /opt/eva-ai/app/VERSION
```

Ожидается чистое дерево, точный SHA выше и `β 0.8.5`.

```bash
sudo -u eva python3 -m venv /opt/eva-ai/app/.venv
sudo -u eva /opt/eva-ai/app/.venv/bin/python -m pip install \
  --upgrade pip 'setuptools<81' wheel
sudo -u eva /opt/eva-ai/app/.venv/bin/pip install \
  -r /opt/eva-ai/app/requirements.txt \
  -r /opt/eva-ai/app/requirements-db.txt

sudo chmod +x /opt/eva-ai/app/run_prod.sh
sudo -u eva /opt/eva-ai/app/.venv/bin/pip check
```

## 7. Qwen3-VL-4B AWQ и vLLM

Точный checkpoint, совпадающий с проверенной установкой по составу 15 файлов и
размеру `model.safetensors` (`4430808624` bytes):

```text
cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit
```

```bash
sudo install -d -o eva -g eva \
  /opt/eva-ai/vllm \
  /var/lib/eva-ai/models/qwen3-vl-4b-awq \
  /var/lib/eva-ai/models/huggingface

sudo -u eva python3 -m venv /opt/eva-ai/vllm/.venv
sudo -u eva /opt/eva-ai/vllm/.venv/bin/python -m pip install \
  --upgrade pip setuptools wheel
sudo -u eva /opt/eva-ai/vllm/.venv/bin/pip install \
  'vllm==0.25.0' huggingface_hub

sudo -u eva env HF_HOME=/var/lib/eva-ai/models/huggingface \
  /opt/eva-ai/vllm/.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit",
    local_dir="/var/lib/eva-ai/models/qwen3-vl-4b-awq",
)
print(path)
PY

sudo du -sh /var/lib/eva-ai/models/qwen3-vl-4b-awq
sudo stat -c '%U:%G %a %s %n' \
  /var/lib/eva-ai/models/qwen3-vl-4b-awq/model.safetensors
```

Создать `/etc/systemd/system/eva-vllm.service`:

```ini
[Unit]
Description=EVA Qwen3-VL-4B AWQ vLLM
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=eva
Group=eva
WorkingDirectory=/opt/eva-ai/vllm
Environment=HF_HOME=/var/lib/eva-ai/models/huggingface
Environment=HF_HUB_OFFLINE=1
Environment=TRANSFORMERS_OFFLINE=1
Environment=VLLM_USE_FLASHINFER_SAMPLER=0
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ExecStart=/opt/eva-ai/vllm/.venv/bin/vllm serve /var/lib/eva-ai/models/qwen3-vl-4b-awq --served-model-name qwen/qwen3-vl-4b --host 127.0.0.1 --port 1234 --max-model-len 32768 --gpu-memory-utilization 0.82 --max-num-seqs 4 --max-num-batched-tokens 4096 --kv-cache-dtype bfloat16 --enforce-eager --attention-backend TRITON_ATTN --limit-mm-per-prompt.image 16 --limit-mm-per-prompt.video 0 --mm-processor-kwargs.max_pixels 100352 --enable-auto-tool-choice --tool-call-parser hermes
Restart=on-failure
RestartSec=10
TimeoutStartSec=300
TimeoutStopSec=60
KillMode=mixed

[Install]
WantedBy=multi-user.target
```

Для RTX A4000 обязательно используется `--kv-cache-dtype bfloat16`. Значение
`fp8` падает, потому что native FP8 для Triton требует SM89+, а A4000 — SM86.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now eva-vllm

for i in $(seq 1 60); do
  if response=$(curl -fsS http://127.0.0.1:1234/v1/models 2>/dev/null); then
    echo "$response" | jq
    break
  fi
  systemctl is-active --quiet eva-vllm || {
    systemctl status eva-vllm --no-pager -l
    sudo journalctl -u eva-vllm -n 120 --no-pager
    break
  }
  echo "Loading vLLM... ($i/60)"
  sleep 10
done

nvidia-smi
```

Ожидается модель `qwen/qwen3-vl-4b`; на GPU 0 занято около 13.5 GiB.

## 8. Qwen3.5-9B-MTP Q4 и llama.cpp на CPU

```bash
sudo install -d -o eva -g eva \
  /var/lib/eva-ai/models/qwen3.5-9b-mtp \
  /var/lib/eva-ai/models/huggingface

sudo -u eva env HF_HOME=/var/lib/eva-ai/models/huggingface \
  /opt/eva-ai/vllm/.venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="unsloth/Qwen3.5-9B-MTP-GGUF",
    filename="Qwen3.5-9B-Q4_K_M.gguf",
    local_dir="/var/lib/eva-ai/models/qwen3.5-9b-mtp",
)
print(path)
PY

sudo stat -c '%U:%G %a %s %n' \
  /var/lib/eva-ai/models/qwen3.5-9b-mtp/Qwen3.5-9B-Q4_K_M.gguf
```

Проверенный размер GGUF: `5868826976` bytes.

```bash
sudo -u eva git clone --depth 1 --branch b9330 \
  https://github.com/ggml-org/llama.cpp.git \
  /opt/eva-ai/llama.cpp

sudo -u eva cmake \
  -S /opt/eva-ai/llama.cpp \
  -B /opt/eva-ai/llama.cpp/build-port-cpu \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF \
  -DGGML_AVX=ON \
  -DGGML_AVX2=ON \
  -DGGML_AVX512=OFF \
  -DGGML_CUDA=OFF \
  -DLLAMA_CURL=OFF

sudo -u eva cmake \
  --build /opt/eva-ai/llama.cpp/build-port-cpu \
  --target llama-server \
  -j 12

/opt/eva-ai/llama.cpp/build-port-cpu/bin/llama-server --version
```

Ожидается commit llama.cpp `328874d` из tag `b9330`.

Создать `/etc/systemd/system/eva-deep-review.service`:

```ini
[Unit]
Description=EVA CPU Qwen3.5-9B-MTP deep review
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=eva
Group=eva
WorkingDirectory=/opt/eva-ai/llama.cpp
ExecStart=/opt/eva-ai/llama.cpp/build-port-cpu/bin/llama-server -m /var/lib/eva-ai/models/qwen3.5-9b-mtp/Qwen3.5-9B-Q4_K_M.gguf -a qwen3.5-9b-mtp --spec-type draft-mtp --spec-draft-n-max 4 -c 65536 -ngl 0 -fa on -ctk q8_0 -ctv q8_0 -np 1 -cb --threads 12 --threads-batch 16 --jinja --metrics --host 127.0.0.1 --port 1236
Restart=on-failure
RestartSec=10
TimeoutStartSec=300
TimeoutStopSec=60
KillMode=mixed

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now eva-deep-review

for i in $(seq 1 90); do
  if response=$(curl -fsS http://127.0.0.1:1236/v1/models 2>/dev/null); then
    echo "$response" | jq
    break
  fi
  echo "Loading Qwen3.5-9B... ($i/90)"
  sleep 10
done
```

Проверить генерацию без внутреннего thinking:

```bash
curl -fsS http://127.0.0.1:1236/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3.5-9b-mtp",
    "messages":[{"role":"user","content":"Reply with exactly: EVA-DEEP-OK"}],
    "chat_template_kwargs":{"enable_thinking":false},
    "temperature":0,
    "max_tokens":64
  }' | jq '{content:.choices[0].message.content, finish_reason:.choices[0].finish_reason}'
```

Без `enable_thinking:false` модель может израсходовать лимит на reasoning и
вернуть пустой `content` с `finish_reason=length`.

## 9. CLIP ViT-B/32

```bash
sudo -u eva env CUDA_VISIBLE_DEVICES=-1 \
  /opt/eva-ai/app/.venv/bin/python - <<'PY'
import clip

cache = "/var/lib/eva-ai/models/clip"
model, _ = clip.load("ViT-B/32", device="cpu", download_root=cache)
print("CLIP loaded successfully on CPU")
del model
PY

sudo stat -c '%U:%G %a %s %n' \
  /var/lib/eva-ai/models/clip/ViT-B-32.pt
```

Проверенный размер: `353976522` bytes.

## 10. PostgreSQL, миграции и runtime-роли

```bash
sudo systemctl enable --now postgresql
sudo -u postgres psql -X -tAc "SELECT 1 FROM pg_database WHERE datname='eva'" \
  | grep -q 1 || sudo -u postgres createdb eva

sudo -u postgres -H env \
  EVA_DATABASE_DSN='postgresql:///eva?host=/var/run/postgresql' \
  bash -c 'cd /opt/eva-ai/app && .venv/bin/alembic upgrade head'

sudo -u postgres psql -X -d eva -tAc \
  'SELECT version_num FROM alembic_version;'
```

Ожидается `20260727_0010`.

Создать секреты. Пароли hex, поэтому их не нужно URL-encode в DSN:

```bash
sudo bash -c '
set -eu
umask 0077
{
  printf "EVA_MIGRATOR_PASSWORD=%s\n" "$(openssl rand -hex 32)"
  printf "EVA_API_PASSWORD=%s\n"      "$(openssl rand -hex 32)"
  printf "EVA_AUDIT_PASSWORD=%s\n"    "$(openssl rand -hex 32)"
  printf "EVA_WORKER_PASSWORD=%s\n"   "$(openssl rand -hex 32)"
  printf "EVA_BACKUP_PASSWORD=%s\n"   "$(openssl rand -hex 32)"
} > /etc/eva-ai/eva-db-secrets.env
chown root:eva /etc/eva-ai/eva-db-secrets.env
chmod 0640 /etc/eva-ai/eva-db-secrets.env
'
```

Создать least-privilege login roles:

```bash
sudo bash -c '
set -a
. /etc/eva-ai/eva-db-secrets.env
set +a
runuser -u postgres -- env \
  EVA_DATABASE_DSN="postgresql:///eva?host=/var/run/postgresql" \
  EVA_MIGRATOR_PASSWORD="$EVA_MIGRATOR_PASSWORD" \
  EVA_API_PASSWORD="$EVA_API_PASSWORD" \
  EVA_AUDIT_PASSWORD="$EVA_AUDIT_PASSWORD" \
  EVA_WORKER_PASSWORD="$EVA_WORKER_PASSWORD" \
  EVA_BACKUP_PASSWORD="$EVA_BACKUP_PASSWORD" \
  /opt/eva-ai/app/.venv/bin/python \
  /opt/eva-ai/app/scripts/bootstrap_db_roles.py
'

sudo -u postgres psql -X -d eva -tAc \
  "SELECT rolname FROM pg_roles WHERE rolname LIKE 'eva_%_login' ORDER BY rolname;"
```

Ожидаются `eva_api_login`, `eva_audit_login`, `eva_backup_login`,
`eva_migrator_login`, `eva_worker_login`.

## 11. Runtime-конфигурация EVA

Удобнее создать интерактивный генератор отдельным файлом. Не запускать такой
Python-код через `python - <<PY`: `input()`/`getpass()` должны читать терминал,
а не heredoc.

```bash
sudo tee /usr/local/sbin/eva-write-site-config >/dev/null <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import getpass
import importlib.util
import os
import sys
import uuid
from pathlib import Path
from urllib.parse import urlsplit

module_path = Path("/opt/eva-ai/app/scripts/install_port_appliance.py")
spec = importlib.util.spec_from_file_location("eva_port_installer", module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

secrets = module.parse_env(Path("/etc/eva-ai/eva-db-secrets.env"))
required = {
    "EVA_MIGRATOR_PASSWORD",
    "EVA_API_PASSWORD",
    "EVA_AUDIT_PASSWORD",
    "EVA_WORKER_PASSWORD",
    "EVA_BACKUP_PASSWORD",
}
missing = sorted(required - secrets.keys())
if missing:
    raise SystemExit("Missing DB secrets: " + ", ".join(missing))

while True:
    evo_url = input("Luxriot Evo base URL, including port: ").strip().rstrip("/")
    parsed = urlsplit(evo_url)
    try:
        valid_port = parsed.port is not None
    except ValueError:
        valid_port = False
    if parsed.scheme in {"http", "https"} and parsed.hostname and valid_port:
        break
    print("Expected http://host:port or https://host:port")

evo_username = input("Luxriot username: ").strip()
evo_password = getpass.getpass("Luxriot password: ")
timezone = input("Site timezone [Europe/Athens]: ").strip() or "Europe/Athens"
strict_answer = input("Require strict deployment security? [Y/n]: ").strip().lower()
strict_security = strict_answer not in {"n", "no"}

tenant_id = str(uuid.uuid4())
values = dict(module.PORT_ENV)
values.update(secrets)
values.update({
    "EVOSSEARCH_SECURE_DEPLOYMENT_REQUIRED": "true" if strict_security else "false",
    "EVOSSEARCH_LUXRIOT_BASE_URL": evo_url,
    "EVOSSEARCH_LUXRIOT_USERNAME": evo_username,
    "EVOSSEARCH_LUXRIOT_PASSWORD": evo_password,
    "EVOSSEARCH_SITE_TIMEZONE": timezone,
    "EVOSSEARCH_MODEL_CACHE_DIR": "/var/lib/eva-ai/models/huggingface",
    "EVOSSEARCH_OPENAI_CLIP_CACHE_DIR": "/var/lib/eva-ai/models/clip",
    "EVOSSEARCH_ALLOWED_ROOTS": "/var/lib/eva-ai",
    "EVOSSEARCH_DETECTIONS_ARCHIVE_DIR": "/var/lib/eva-ai/detections_archive",
    "EVOSSEARCH_INFERENCE_QUEUE_SPOOL_DIR": "/var/lib/eva-ai/inference-spool",
    "EVOSSEARCH_PROBE_CHANNEL_GROUPS_FILE": "/var/lib/eva-ai/state/probe_channel_groups.json",
    "EVOSSEARCH_LUXRIOT_SUMMARY_STATE_FILE": "/var/lib/eva-ai/state/luxriot_summary_state.json",
    "EVOSSEARCH_LUXRIOT_ROLLUP_CACHE_FILE": "/var/lib/eva-ai/state/luxriot_rollups_cache.json",
    "EVOSSEARCH_LM_PROFILES": "agent,vlm",
    "EVOSSEARCH_LM_AGENT_PROFILE_ID": "agent",
    "EVOSSEARCH_LM_VLM_PROFILE_ID": "vlm",
    "EVOSSEARCH_LM_PROFILE_AGENT_KIND": "agent",
    "EVOSSEARCH_LM_PROFILE_AGENT_ENABLED": "true",
    "EVOSSEARCH_LM_PROFILE_AGENT_BASE_URL": "http://127.0.0.1:1234/v1",
    "EVOSSEARCH_LM_PROFILE_AGENT_MODEL": "qwen/qwen3-vl-4b",
    "EVOSSEARCH_LM_PROFILE_AGENT_TIMEOUT": "600",
    "EVOSSEARCH_LM_PROFILE_VLM_KIND": "vlm",
    "EVOSSEARCH_LM_PROFILE_VLM_ENABLED": "true",
    "EVOSSEARCH_LM_PROFILE_VLM_BASE_URL": "http://127.0.0.1:1234/v1",
    "EVOSSEARCH_LM_PROFILE_VLM_MODEL": "qwen/qwen3-vl-4b",
    "EVOSSEARCH_LM_PROFILE_VLM_TIMEOUT": "600",
    "EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_BASE_URL": "http://127.0.0.1:1236/v1",
    "EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_MODEL": "qwen3.5-9b-mtp",
    "EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_ENABLED": "true",
    "EVA_DATABASE_DSN": f"postgresql://eva_api_login:{secrets['EVA_API_PASSWORD']}@127.0.0.1:5432/eva",
    "EVA_AUDIT_DATABASE_DSN": f"postgresql://eva_audit_login:{secrets['EVA_AUDIT_PASSWORD']}@127.0.0.1:5432/eva",
    "EVA_WORKER_DATABASE_DSN": f"postgresql://eva_worker_login:{secrets['EVA_WORKER_PASSWORD']}@127.0.0.1:5432/eva",
    "EVA_MIGRATION_DATABASE_DSN": f"postgresql://eva_migrator_login:{secrets['EVA_MIGRATOR_PASSWORD']}@127.0.0.1:5432/eva",
})
for key in module.TENANT_ID_KEYS:
    values[key] = tenant_id

target = Path("/etc/eva-ai/eva-ai.env")
target.write_text(module.render_env(values), encoding="utf-8")
os.chown(target, 0, 0)
os.chmod(target, 0o600)
print(f"Configuration written: {target}")
print(f"Variables written: {len(values)}")
PY

sudo chmod 0755 /usr/local/sbin/eva-write-site-config
sudo /usr/local/sbin/eva-write-site-config
sudo stat -c '%U:%G %a %n' /etc/eva-ai/eva-ai.env
```

Для обычной установки отвечать `Y` на strict security. Проверить файл без
вывода секретов:

```bash
sudo /opt/eva-ai/app/.venv/bin/python \
  /opt/eva-ai/app/scripts/validate_appliance_config.py \
  --env-file /etc/eva-ai/eva-ai.env
```

### Зафиксированное исключение Ventspils

На проверенном стенде существующий пароль Luxriot был короче требования EVA,
поэтому `/ready` помечал `deployment_security` как misconfigured. По решению
владельца стенда strict gate отключён, но named-user auth, HTTPS и secure cookie
оставлены включёнными.

Для точного повторения этого исключения в генераторе отвечают `n`. После этого
обычная file-validation закономерно откажется принимать конфигурацию. В unit
ниже строку validator меняют на:

```ini
ExecStartPre=/usr/bin/env EVOSSEARCH_SECURE_DEPLOYMENT_REQUIRED=true /opt/eva-ai/app/.venv/bin/python /opt/eva-ai/app/scripts/validate_appliance_config.py --from-environment
```

Так только preflight проверяет полноту файла в strict-режиме, а приложение
получает исходное `EVOSSEARCH_SECURE_DEPLOYMENT_REQUIRED=false`. Это осознанный
site-specific bypass, а не рекомендуемое production-значение.

## 12. Nginx HTTPS без захвата порта 80

Закрепить IP сервера и указать его явно:

```bash
EVA_SERVER_IP='<EVA_LAN_IP>'
EVA_HOSTNAME="$(hostname -f 2>/dev/null || hostname)"

sudo install -d -m 0750 /etc/eva-ai/tls
sudo openssl req -x509 -newkey rsa:3072 -sha256 -days 825 -nodes \
  -subj "/CN=${EVA_HOSTNAME}" \
  -addext "subjectAltName=DNS:${EVA_HOSTNAME},IP:${EVA_SERVER_IP}" \
  -keyout /etc/eva-ai/tls/eva-ai.key \
  -out /etc/eva-ai/tls/eva-ai.crt
sudo chmod 0600 /etc/eva-ai/tls/eva-ai.key
```

Создать `/etc/nginx/sites-available/eva-ai`:

```nginx
server {
    listen 443 ssl;
    listen [::]:443 ssl;
    server_name _;

    ssl_certificate /etc/eva-ai/tls/eva-ai.crt;
    ssl_certificate_key /etc/eva-ai/tls/eva-ai.key;
    client_max_body_size 256m;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
        proxy_buffering off;
    }
}
```

```bash
sudo ln -sfn /etc/nginx/sites-available/eva-ai /etc/nginx/sites-enabled/eva-ai
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl enable --now nginx
sudo ss -lntp | grep ':443 '
```

Удаление `sites-enabled/default` удаляет только default-конфигурацию Nginx и
не трогает Nextcloud/Apache на порту 80.

## 13. Первый администратор EVA

```bash
sudo bash -lc '
set -a
. /etc/eva-ai/eva-ai.env
set +a
exec sudo -u eva -E \
  /opt/eva-ai/app/.venv/bin/python \
  /opt/eva-ai/app/scripts/bootstrap_admin.py \
  --tenant-id "$EVOSSEARCH_AUTH_TENANT_ID" \
  --username admin \
  --display-name "EVA Administrator"
'
```

Пароль вводится дважды и должен быть не короче 12 символов.

## 14. systemd unit EVA AI

Создать `/etc/systemd/system/eva-ai.service`:

```ini
[Unit]
Description=EVA AI Ventspils appliance
After=network-online.target postgresql.service eva-vllm.service
Wants=network-online.target eva-vllm.service
Requires=postgresql.service

[Service]
Type=simple
User=eva
Group=eva
WorkingDirectory=/opt/eva-ai/app
EnvironmentFile=/etc/eva-ai/eva-ai.env
Environment=EVOSSEARCH_CONFIG_ENV_FILE=/etc/eva-ai/eva-ai.env
ExecStartPre=/opt/eva-ai/app/.venv/bin/python /opt/eva-ai/app/scripts/validate_appliance_config.py --from-environment
ExecStartPre=/opt/eva-ai/app/.venv/bin/python /opt/eva-ai/app/scripts/wait_openai_endpoint.py --timeout 600
ExecStart=/opt/eva-ai/app/run_prod.sh
Restart=on-failure
RestartSec=5
TimeoutStartSec=660
TimeoutStopSec=120
KillSignal=SIGTERM
UMask=0077
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

Если применено исключение из раздела 11, заменить первую `ExecStartPre` на
указанный там вариант с `/usr/bin/env`.

```bash
sudo systemctl daemon-reload
sudo systemctl enable eva-vllm eva-deep-review eva-ai
sudo systemctl restart eva-vllm eva-deep-review
sudo systemctl start eva-ai

systemctl status eva-vllm eva-deep-review eva-ai --no-pager -l
sudo ss -lntp | grep -E ':(1234|1236|5000|5432)\b'
```

## 15. Проверка установки

```bash
echo '=== MODELS ==='
curl -fsS http://127.0.0.1:1234/v1/models | jq
curl -fsS http://127.0.0.1:1236/v1/models | jq

echo '=== EVA ==='
curl -sS http://127.0.0.1:5000/health | jq
curl -sS http://127.0.0.1:5000/ready | jq '{status,version,checks}'

echo '=== SERVICES ==='
systemctl is-active postgresql eva-vllm eva-deep-review eva-ai nginx
systemctl is-enabled eva-vllm eva-deep-review eva-ai nginx

echo '=== GPU ==='
nvidia-smi
```

Acceptance criteria:

- `/ready.status` равно `ready`;
- version равна `β 0.8.5`;
- PostgreSQL revision равна `20260727_0010`;
- `qwen/qwen3-vl-4b` и `qwen3.5-9b-mtp` отвечают;
- `luxriot.status` — `reachable`;
- `embedder.status` — `loaded`;
- вход через `https://<EVA_LAN_IP>/` работает;
- Agent отвечает на простой запрос;
- список каналов Luxriot загружается автоматически.

Для site-specific security bypass нормальный результат:

```text
deployment_security.required = false
deployment_security.status   = ready
```

## 16. Первый VLM feed

Каналы вручную в EVA не добавляются: они автоматически подтягиваются из
Luxriot Evo. На каждый канал независимо можно включить Live Description,
создать Probe или использовать оба режима.

Smoke test:

1. открыть Video;
2. выбрать канал с работающим live preview;
3. оставить batch `12` и interval `5 s`;
4. нажать **Start summaries**;
5. ждать около 60–90 секунд до первой сводки.

Batch из 12 кадров с интервалом 5 секунд физически заполняется примерно минуту,
поэтому пустой feed сразу после запуска не является ошибкой.

Проверить HTTP-цепочку во время клика:

```bash
sudo tail -n 0 -F /var/log/nginx/access.log \
  | grep --line-buffered -E '/luxriot/(prompt_settings|start_capture|stop_capture|session)'
```

Исправный запуск показывает:

```text
GET  /luxriot/prompt_settings?... 200
POST /luxriot/start_capture        200
GET  /luxriot/session?...          200
```

Если POST есть, но через две минуты нет сводки:

```bash
sudo journalctl -u eva-vllm --since '5 minutes ago' --no-pager | tail -n 100
sudo journalctl -u eva-ai --since '5 minutes ago' --no-pager \
  | grep -Ei 'error|exception|failed|summary|inference|channel' | tail -n 120
```

## 17. Диагностика и эксплуатация

```bash
sudo journalctl -u eva-ai -n 150 --no-pager
sudo journalctl -u eva-vllm -n 150 --no-pager
sudo journalctl -u eva-deep-review -n 150 --no-pager
sudo nginx -t
sudo tail -n 100 /var/log/nginx/error.log
```

После reboot:

```bash
hostname -I
systemctl is-active postgresql eva-vllm eva-deep-review eva-ai nginx
curl -sS http://127.0.0.1:5000/ready | jq '.status,.version'
nvidia-smi
```

Резервная копия перед обновлением:

```bash
sudo install -d -m 0700 /var/backups/eva-ai
sudo bash -c 'sudo -u postgres pg_dump -Fc eva \
  > "/var/backups/eva-ai/eva-$(date +%Y%m%d-%H%M%S).dump"'
sudo cp -a /etc/eva-ai \
  "/var/backups/eva-ai/etc-eva-ai-$(date +%Y%m%d-%H%M%S)"
```

Не выполнять `git pull` в production-каталоге без проверки миграций. Для
обновления сначала зафиксировать текущие SHA и DB revision, сделать backup,
проверить разницу ветки в отдельном checkout и только затем планировать
обновление к новому проверенному commit.

## 18. Фактически подтверждённый итог

На завершённом стенде одновременно работали:

- `eva-vllm`: active/enabled, Qwen3-VL-4B на GPU 0;
- `eva-deep-review`: active/enabled, Qwen3.5-9B-MTP на CPU;
- `eva-ai`: active/enabled, Gunicorn на `127.0.0.1:5000`;
- PostgreSQL с пятью least-privilege login roles;
- Nginx HTTPS на `443`, без вмешательства в существующий порт `80`;
- Agent, Luxriot live preview и VLM summaries;
- `/ready.status = ready`.
