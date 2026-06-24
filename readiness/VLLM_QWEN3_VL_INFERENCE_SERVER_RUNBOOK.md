# EVA AI: vLLM Qwen3-VL inference server runbook

Date: 2026-06-16

## Scope

This runbook installs a standalone OpenAI-compatible VLM inference node for
EVA AI. It is based on the working `Luxriot1` setup:

- OS: Ubuntu 26.04 LTS, amd64.
- GPUs: 2 x NVIDIA GeForce RTX 5080 16 GB.
- Driver observed on the first server: `595.71.05`.
- vLLM: `0.23.0`.
- Torch: `2.11.0+cu130`.
- Model: `Qwen/Qwen3-VL-4B-Instruct-FP8`.
- Local model path: `/opt/eva-vllm/models/qwen3-vl-4b-fp8`.
- GPU1 endpoint: `http://<server-ip>:8001/v1`.
- GPU0 endpoint: `http://<server-ip>:8002/v1`.

The setup intentionally does not require CUDA Toolkit / `nvcc`. FlashInfer
sampling is disabled with `VLLM_USE_FLASHINFER_SAMPLER=0`, because the first
server failed during sampler JIT when `/usr/local/cuda` was absent.

## 1. Verify the machine

```bash
hostname -I
lsb_release -a || cat /etc/os-release
nvidia-smi
```

Expected:

- Two RTX 5080 GPUs are visible.
- Driver is installed.
- The machine has enough free disk for vLLM packages, model files, and compile
  cache. Keep at least 50 GB free.

If `nvidia-smi` is missing or does not show the GPUs, stop and fix the NVIDIA
driver before continuing.

## 2. Base packages

```bash
sudo apt update
sudo apt install -y \
  ca-certificates curl git jq openssl \
  build-essential python3 python3-venv python3-pip \
  rsync
```

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo install -m 0755 "$HOME/.local/bin/uv" /usr/local/bin/uv
uv --version
```

## 3. Create the vLLM workspace

This runbook uses the currently logged-in user for the service. On the first
server that user was `pc`.

```bash
export EVA_VLLM_USER="$USER"
export EVA_VLLM_GROUP="$(id -gn)"

sudo mkdir -p /opt/eva-vllm
sudo chown "$EVA_VLLM_USER:$EVA_VLLM_GROUP" /opt/eva-vllm

cd /opt/eva-vllm
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate

python --version
```

## 4. Install vLLM

```bash
cd /opt/eva-vllm
source .venv/bin/activate

uv pip install --upgrade pip setuptools wheel
uv pip install vllm --torch-backend auto
```

Verify CUDA visibility:

```bash
cd /opt/eva-vllm
source .venv/bin/activate

python - <<'PY'
import torch
import vllm
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda devices", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
print("vllm", vllm.__version__)
PY
```

Expected on the first server:

```text
torch 2.11.0+cu130
cuda available True
cuda devices 2
0 NVIDIA GeForce RTX 5080
1 NVIDIA GeForce RTX 5080
vllm 0.23.0
```

## 5. Install the model

### Option A: copy from the first inference server

Prefer this if the first server already has the model and the local network is
faster than internet access. Replace `192.168.3.104` and `pc` if needed.

```bash
mkdir -p /opt/eva-vllm/models/qwen3-vl-4b-fp8

rsync -avP \
  pc@192.168.3.104:/opt/eva-vllm/models/qwen3-vl-4b-fp8/ \
  /opt/eva-vllm/models/qwen3-vl-4b-fp8/
```

### Option B: download from Hugging Face

Use this if server-to-server copy is unavailable. `HF_HUB_DISABLE_XET=1`
avoids the stalled Xet download behavior observed on the first server.

```bash
cd /opt/eva-vllm
source .venv/bin/activate

mkdir -p /opt/eva-vllm/hf
mkdir -p /opt/eva-vllm/models/qwen3-vl-4b-fp8

export HF_HOME=/opt/eva-vllm/hf
export HF_HUB_DISABLE_XET=1

hf download Qwen/Qwen3-VL-4B-Instruct-FP8 \
  --local-dir /opt/eva-vllm/models/qwen3-vl-4b-fp8
```

If `hf` is unavailable:

```bash
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct-FP8 \
  --local-dir /opt/eva-vllm/models/qwen3-vl-4b-fp8
```

Verify:

```bash
du -sh /opt/eva-vllm/models/qwen3-vl-4b-fp8
find /opt/eva-vllm/models/qwen3-vl-4b-fp8 -maxdepth 1 -type f \
  -printf '%10s %p\n' | sort -n | tail
```

Expected model size is about `5.7G`, with two safetensor shards:

```text
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
```

## 6. Smoke-test GPU1 manually

Start GPU1 first. GPU0 often carries display/remote-desktop load.

```bash
cd /opt/eva-vllm
source .venv/bin/activate

export HF_HOME=/opt/eva-vllm/hf
export HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve /opt/eva-vllm/models/qwen3-vl-4b-fp8 \
  --served-model-name qwen3-vl-4b-fp8 \
  --host 0.0.0.0 \
  --port 8001 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.82 \
  --max-num-seqs 4 \
  --limit-mm-per-prompt.video 0 \
  --limit-mm-per-prompt.image 16 \
  --mm-processor-cache-gb 0 \
  --trust-remote-code
```

Wait for the API server to finish startup. In another terminal:

```bash
curl -sS http://127.0.0.1:8001/v1/models | jq
```

Text smoke-test:

```bash
curl -sS http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-vl-4b-fp8","messages":[{"role":"user","content":"Say OK."}],"max_tokens":32}' | jq
```

Image smoke-test without relying on external image URLs:

```bash
python - <<'PY'
from PIL import Image, ImageDraw

img = Image.new("RGB", (640, 360), (30, 55, 85))
d = ImageDraw.Draw(img)
d.rectangle((40, 190, 600, 340), fill=(45, 140, 75))
d.ellipse((480, 40, 570, 130), fill=(245, 210, 70))
d.rectangle((120, 130, 260, 300), fill=(160, 90, 55))
d.text((50, 40), "EVA AI VLM test image", fill=(255, 255, 255))
img.save("/tmp/eva-vlm-test.jpg", quality=90)
print("/tmp/eva-vlm-test.jpg")
PY

IMG="$(base64 -w0 /tmp/eva-vlm-test.jpg)"

curl -sS http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"qwen3-vl-4b-fp8\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Describe this image in one short sentence.\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,$IMG\"}}]}],\"max_tokens\":80}" | jq
```

If these tests pass, stop the manual server with `Ctrl+C` and create systemd
services.

## 7. GPU1 systemd service

```bash
export EVA_VLLM_USER="$USER"
export EVA_VLLM_GROUP="$(id -gn)"

sudo tee /etc/systemd/system/eva-vllm-gpu1.service >/dev/null <<EOF
[Unit]
Description=EVA vLLM Qwen3-VL 4B FP8 GPU1
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${EVA_VLLM_USER}
Group=${EVA_VLLM_GROUP}
WorkingDirectory=/opt/eva-vllm
Environment=HF_HOME=/opt/eva-vllm/hf
Environment=HF_HUB_DISABLE_XET=1
Environment=OMP_NUM_THREADS=1
Environment=CUDA_VISIBLE_DEVICES=1
Environment=VLLM_USE_FLASHINFER_SAMPLER=0
ExecStart=/opt/eva-vllm/.venv/bin/vllm serve /opt/eva-vllm/models/qwen3-vl-4b-fp8 --served-model-name qwen3-vl-4b-fp8 --host 0.0.0.0 --port 8001 --max-model-len 8192 --gpu-memory-utilization 0.82 --max-num-seqs 4 --limit-mm-per-prompt.video 0 --limit-mm-per-prompt.image 16 --mm-processor-cache-gb 0 --trust-remote-code
Restart=on-failure
RestartSec=10
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now eva-vllm-gpu1
```

Check:

```bash
sudo systemctl status eva-vllm-gpu1 --no-pager
sudo journalctl -u eva-vllm-gpu1 -n 120 --no-pager -l
curl -sS http://127.0.0.1:8001/v1/models | jq
```

## 8. GPU0 systemd service

GPU0 may have display or TeamViewer memory pressure. Keep the same image/context
limits as GPU1, but lower GPU memory utilization if this card carries desktop
or remote-access load.

```bash
export EVA_VLLM_USER="$USER"
export EVA_VLLM_GROUP="$(id -gn)"

sudo tee /etc/systemd/system/eva-vllm-gpu0.service >/dev/null <<EOF
[Unit]
Description=EVA vLLM Qwen3-VL 4B FP8 GPU0
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${EVA_VLLM_USER}
Group=${EVA_VLLM_GROUP}
WorkingDirectory=/opt/eva-vllm
Environment=HF_HOME=/opt/eva-vllm/hf
Environment=HF_HUB_DISABLE_XET=1
Environment=OMP_NUM_THREADS=1
Environment=CUDA_VISIBLE_DEVICES=0
Environment=VLLM_USE_FLASHINFER_SAMPLER=0
ExecStart=/opt/eva-vllm/.venv/bin/vllm serve /opt/eva-vllm/models/qwen3-vl-4b-fp8 --served-model-name qwen3-vl-4b-fp8 --host 0.0.0.0 --port 8002 --max-model-len 8192 --gpu-memory-utilization 0.82 --max-num-seqs 4 --limit-mm-per-prompt.video 0 --limit-mm-per-prompt.image 16 --mm-processor-cache-gb 0 --trust-remote-code
Restart=on-failure
RestartSec=10
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now eva-vllm-gpu0
```

Check:

```bash
sudo systemctl status eva-vllm-gpu0 --no-pager
sudo journalctl -u eva-vllm-gpu0 -n 120 --no-pager -l
curl -sS http://127.0.0.1:8002/v1/models | jq
nvidia-smi
```

If GPU0 fails due to memory pressure, reduce memory utilization first:

```text
--gpu-memory-utilization 0.78
```

If it still fails, reduce concurrency:

```text
--max-num-seqs 2
```

Then reload and restart:

```bash
sudo systemctl daemon-reload
sudo systemctl restart eva-vllm-gpu0
```

## 9. Remote checks from the EVA AI server

From the EVA AI control-plane machine:

```bash
curl -sS http://<inference-server-ip>:8001/v1/models | jq
curl -sS http://<inference-server-ip>:8002/v1/models | jq
```

If local checks pass but remote checks fail, inspect firewall rules:

```bash
sudo ufw status verbose
```

For a closed trusted lab network:

```bash
sudo ufw allow 8001/tcp
sudo ufw allow 8002/tcp
```

For a tighter deployment, allow only the EVA AI server IP:

```bash
sudo ufw allow from <eva-ai-server-ip> to any port 8001 proto tcp
sudo ufw allow from <eva-ai-server-ip> to any port 8002 proto tcp
```

## 10. EVA AI profile mapping

For this second inference server, add two more VLM profiles to
`/etc/eva-ai/eva-ai.env` on the EVA AI control-plane host.

Example for two inference servers:

```env
EVOSSEARCH_LM_PROFILES=agent,vlm-a1,vlm-a0,vlm-b1,vlm-b0
EVOSSEARCH_LM_AGENT_PROFILE_ID=agent
EVOSSEARCH_LM_VLM_PROFILE_ID=vlm-a1
EVOSSEARCH_LM_VLM_BALANCER_ENABLED=true
EVOSSEARCH_LM_VLM_BALANCER_PROFILES=vlm-a1,vlm-a0,vlm-b1,vlm-b0

EVOSSEARCH_LM_PROFILE_AGENT_KIND=agent
EVOSSEARCH_LM_PROFILE_AGENT_BASE_URL=http://<eva-ai-agent-host>:1234/v1
EVOSSEARCH_LM_PROFILE_AGENT_MODEL=<agent-model-id>
EVOSSEARCH_LM_PROFILE_AGENT_API_KEY=
EVOSSEARCH_LM_PROFILE_AGENT_TIMEOUT=600
EVOSSEARCH_LM_PROFILE_AGENT_ENABLED=true

EVOSSEARCH_LM_PROFILE_VLM_A1_KIND=vlm
EVOSSEARCH_LM_PROFILE_VLM_A1_BASE_URL=http://<first-inference-server-ip>:8001/v1
EVOSSEARCH_LM_PROFILE_VLM_A1_MODEL=qwen3-vl-4b-fp8
EVOSSEARCH_LM_PROFILE_VLM_A1_API_KEY=
EVOSSEARCH_LM_PROFILE_VLM_A1_TIMEOUT=240
EVOSSEARCH_LM_PROFILE_VLM_A1_ENABLED=true
EVOSSEARCH_LM_PROFILE_VLM_A1_GPU=server-a:1

EVOSSEARCH_LM_PROFILE_VLM_A0_KIND=vlm
EVOSSEARCH_LM_PROFILE_VLM_A0_BASE_URL=http://<first-inference-server-ip>:8002/v1
EVOSSEARCH_LM_PROFILE_VLM_A0_MODEL=qwen3-vl-4b-fp8
EVOSSEARCH_LM_PROFILE_VLM_A0_API_KEY=
EVOSSEARCH_LM_PROFILE_VLM_A0_TIMEOUT=240
EVOSSEARCH_LM_PROFILE_VLM_A0_ENABLED=true
EVOSSEARCH_LM_PROFILE_VLM_A0_GPU=server-a:0

EVOSSEARCH_LM_PROFILE_VLM_B1_KIND=vlm
EVOSSEARCH_LM_PROFILE_VLM_B1_BASE_URL=http://<second-inference-server-ip>:8001/v1
EVOSSEARCH_LM_PROFILE_VLM_B1_MODEL=qwen3-vl-4b-fp8
EVOSSEARCH_LM_PROFILE_VLM_B1_API_KEY=
EVOSSEARCH_LM_PROFILE_VLM_B1_TIMEOUT=240
EVOSSEARCH_LM_PROFILE_VLM_B1_ENABLED=true
EVOSSEARCH_LM_PROFILE_VLM_B1_GPU=server-b:1

EVOSSEARCH_LM_PROFILE_VLM_B0_KIND=vlm
EVOSSEARCH_LM_PROFILE_VLM_B0_BASE_URL=http://<second-inference-server-ip>:8002/v1
EVOSSEARCH_LM_PROFILE_VLM_B0_MODEL=qwen3-vl-4b-fp8
EVOSSEARCH_LM_PROFILE_VLM_B0_API_KEY=
EVOSSEARCH_LM_PROFILE_VLM_B0_TIMEOUT=240
EVOSSEARCH_LM_PROFILE_VLM_B0_ENABLED=true
EVOSSEARCH_LM_PROFILE_VLM_B0_GPU=server-b:0
```

Restart EVA AI after editing the env:

```bash
sudo systemctl restart eva-ai
curl -sS http://127.0.0.1:5000/ready | jq
```

## 11. Operational commands

```bash
sudo systemctl status eva-vllm-gpu1 --no-pager
sudo systemctl status eva-vllm-gpu0 --no-pager
sudo journalctl -u eva-vllm-gpu1 -n 120 --no-pager -l
sudo journalctl -u eva-vllm-gpu0 -n 120 --no-pager -l
curl -sS http://127.0.0.1:8001/v1/models | jq
curl -sS http://127.0.0.1:8002/v1/models | jq
nvidia-smi
```

Restart:

```bash
sudo systemctl restart eva-vllm-gpu1
sudo systemctl restart eva-vllm-gpu0
```

Stop:

```bash
sudo systemctl stop eva-vllm-gpu1
sudo systemctl stop eva-vllm-gpu0
```

## 12. Known issues from the first server

### Hugging Face download stalls

Symptom:

- vLLM process sits for hours after `model.safetensors.index.json`.
- `/opt/eva-vllm/hf` remains small.
- `.incomplete` files stop changing.

Fix:

```bash
pkill -f "vllm serve" || true
find /opt/eva-vllm/hf/hub/.locks -type f -delete 2>/dev/null || true
export HF_HUB_DISABLE_XET=1
hf download Qwen/Qwen3-VL-4B-Instruct-FP8 \
  --local-dir /opt/eva-vllm/models/qwen3-vl-4b-fp8
```

### FlashInfer sampler requires nvcc

Symptom:

```text
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

Fix:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

This is already included in the systemd services above.

### External image URL fails during smoke-test

Some image hosts reject generated thumbnail URLs. Use the base64 local JPEG
smoke-test in this runbook instead.
