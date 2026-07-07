# EVA AI client restart and IP-change runbook

This runbook is for operators who need to start the three-server EVA AI pilot
after transport, power-off, reboot, or a client network IP change.

Current deployment shape:

- EVA AI control-plane server: Flask UI/API, PostgreSQL, CLIP, agent endpoint.
- Inference server A: two vLLM services, GPU1 on `8001`, GPU0 on `8002`.
- Inference server B: two vLLM services, GPU1 on `8001`, GPU0 on `8002`.
- Luxriot Evo server: source of channels, snapshots, and bookmarks.

Do not paste explanatory text into the terminal. Paste only command blocks.

## 1. After Power-On: Start vLLM On Both Inference Servers

Run this on each inference server.

```bash
cd /opt/eva-vllm

sudo systemctl restart eva-vllm-gpu1 eva-vllm-gpu0

sleep 90

sudo systemctl status eva-vllm-gpu1 --no-pager -l
sudo systemctl status eva-vllm-gpu0 --no-pager -l

curl -sS http://127.0.0.1:8001/v1/models | jq '.data[0] | {id,max_model_len}'
curl -sS http://127.0.0.1:8002/v1/models | jq '.data[0] | {id,max_model_len}'

nvidia-smi
```

Expected:

- both services are `active (running)`;
- both model checks show `qwen3-vl-4b-fp8`;
- both model checks show `"max_model_len": 8192`;
- `nvidia-smi` shows one vLLM process on each GPU.

If `8001` or `8002` still shows `"max_model_len": 4096`, an old manual vLLM
process is still listening. Clear stale listeners and restart:

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
sleep 90

curl -sS http://127.0.0.1:8001/v1/models | jq '.data[0] | {id,max_model_len}'
curl -sS http://127.0.0.1:8002/v1/models | jq '.data[0] | {id,max_model_len}'
```

## 2. Find The New Server IP Addresses

Run on each server and write down the LAN IP.

```bash
hostname -I
ip -br addr
```

Required values:

- `LUXRIOT_EVO_IP`: Luxriot Evo server IP.
- `INFERENCE_A_IP`: first vLLM server IP.
- `INFERENCE_B_IP`: second vLLM server IP.
- `AGENT_BASE_URL`: OpenAI-compatible agent endpoint. If the agent is LM Studio
  on the EVA AI server, this is usually `http://127.0.0.1:1234/v1`.

## 3. Update EVA AI IP Settings

Run on the EVA AI control-plane server.

Set the actual client-site values first:

```bash
export LUXRIOT_EVO_IP="CHANGE_ME"
export LUXRIOT_EVO_PORT="8080"
export INFERENCE_A_IP="CHANGE_ME"
export INFERENCE_B_IP="CHANGE_ME"
export AGENT_BASE_URL="http://127.0.0.1:1234/v1"
```

Apply them to `/etc/eva-ai/eva-ai.env`:

```bash
sudo cp -a /etc/eva-ai/eva-ai.env "/etc/eva-ai/eva-ai.env.bak.$(date +%Y%m%d-%H%M%S)"

sudo sed -i -E \
  -e "s|^EVOSSEARCH_LUXRIOT_BASE_URL=.*|EVOSSEARCH_LUXRIOT_BASE_URL=http://${LUXRIOT_EVO_IP}:${LUXRIOT_EVO_PORT}|" \
  -e "s|^EVOSSEARCH_LM_PROFILE_AGENT_BASE_URL=.*|EVOSSEARCH_LM_PROFILE_AGENT_BASE_URL=${AGENT_BASE_URL}|" \
  -e "s|^EVOSSEARCH_LM_PROFILE_VLM_A1_BASE_URL=.*|EVOSSEARCH_LM_PROFILE_VLM_A1_BASE_URL=http://${INFERENCE_A_IP}:8001/v1|" \
  -e "s|^EVOSSEARCH_LM_PROFILE_VLM_A0_BASE_URL=.*|EVOSSEARCH_LM_PROFILE_VLM_A0_BASE_URL=http://${INFERENCE_A_IP}:8002/v1|" \
  -e "s|^EVOSSEARCH_LM_PROFILE_VLM_B1_BASE_URL=.*|EVOSSEARCH_LM_PROFILE_VLM_B1_BASE_URL=http://${INFERENCE_B_IP}:8001/v1|" \
  -e "s|^EVOSSEARCH_LM_PROFILE_VLM_B0_BASE_URL=.*|EVOSSEARCH_LM_PROFILE_VLM_B0_BASE_URL=http://${INFERENCE_B_IP}:8002/v1|" \
  /etc/eva-ai/eva-ai.env

sudo chmod 0600 /etc/eva-ai/eva-ai.env
sudo chown root:root /etc/eva-ai/eva-ai.env

sudo grep -E 'EVOSSEARCH_LUXRIOT_BASE_URL|EVOSSEARCH_LM_PROFILE_(AGENT|VLM_A1|VLM_A0|VLM_B1|VLM_B0)_BASE_URL|EVOSSEARCH_(OFFLINE_VIDEO|PROBE_SNAP|INDEXED_FOLDER)_ENABLED' /etc/eva-ai/eva-ai.env
```

If Luxriot credentials also changed, edit the env file:

```bash
sudo nano /etc/eva-ai/eva-ai.env
```

For the current client pilot, keep these feature gates disabled unless the patch
explicitly says otherwise:

```bash
EVOSSEARCH_OFFLINE_VIDEO_ENABLED=false
EVOSSEARCH_PROBE_SNAP_ENABLED=false
EVOSSEARCH_INDEXED_FOLDER_ENABLED=false
```

These flags hide the UI surfaces and return 404 from the server endpoints.
Archive Research and image description for found archive frames remain enabled.

Relevant fields:

```env
EVOSSEARCH_LUXRIOT_USERNAME=admin
EVOSSEARCH_LUXRIOT_PASSWORD=CHANGE_ME
EVOSSEARCH_LUXRIOT_DEFAULT_CHANNEL_ID=105
```

## 4. Start EVA AI

If the agent is LM Studio on the EVA AI server, start LM Studio first, load the
agent model, and start the OpenAI-compatible server on port `1234`.

Then run:

```bash
sudo systemctl restart postgresql
sudo systemctl restart eva-ai

sleep 15

sudo systemctl status eva-ai --no-pager -l
curl -sS http://127.0.0.1:5000/health | jq
curl -sS http://127.0.0.1:5000/ready | jq '.status, .checks.luxriot, .checks.lm_profiles, .checks.postgresql, .checks.authentication'
```

Expected:

- `postgresql`, `authentication`, and `luxriot` are ready.
- `lm_profiles` lists all configured VLM profiles as reachable.
- If the agent LM Studio endpoint is not running yet, agent calls will fail even
  if live video descriptions work.

Primary pilot signal after restart is live video descriptions, not probe counts.
In the UI open `Video` and `Agent`:

- `Video` -> confirm selected channels can `Start summaries`;
- `Agent` -> `Stream status` chip should report active video-description streams,
  assigned models, pending frames, dropped frames/batches, and last errors;
- if a channel was configured before restart but is not running now, treat it as
  `desired but not running` and check `/ready`, Luxriot snapshot access, and VLM
  endpoint reachability before looking at probes.

## 5. Verify All Four VLM Endpoints From EVA AI

Run on the EVA AI server:

```bash
export INFERENCE_A_IP="CHANGE_ME"
export INFERENCE_B_IP="CHANGE_ME"

set -e

for base in \
  "http://${INFERENCE_A_IP}:8001/v1" \
  "http://${INFERENCE_A_IP}:8002/v1" \
  "http://${INFERENCE_B_IP}:8001/v1" \
  "http://${INFERENCE_B_IP}:8002/v1"
do
  echo "=== ${base} ==="
  curl -sS "${base}/models" | jq '.data[0] | {id,max_model_len}'
done
```

Optional full 12-image test:

```bash
python3 - <<'PY' >/tmp/eva-local-12-images.json
import base64, io, json
from PIL import Image, ImageDraw

img = Image.new("RGB", (640, 360), (20, 28, 36))
draw = ImageDraw.Draw(img)
draw.rectangle((80, 90, 300, 260), fill=(40, 180, 120))
draw.ellipse((420, 80, 540, 200), fill=(230, 190, 40))
buf = io.BytesIO()
img.save(buf, format="JPEG", quality=85)
b64 = base64.b64encode(buf.getvalue()).decode()

content = [{"type":"text","text":"Describe these 12 frames in one short sentence."}]
for _ in range(12):
    content.append({"type":"image_url","image_url":{"url":"data:image/jpeg;base64," + b64}})

print(json.dumps({
  "model": "qwen3-vl-4b-fp8",
  "messages": [{"role":"user","content":content}],
  "max_tokens": 80
}))
PY

for base in \
  "http://${INFERENCE_A_IP}:8001/v1" \
  "http://${INFERENCE_A_IP}:8002/v1" \
  "http://${INFERENCE_B_IP}:8001/v1" \
  "http://${INFERENCE_B_IP}:8002/v1"
do
  echo "=== ${base} ==="
  curl -sS "${base}/chat/completions" \
    -H 'Content-Type: application/json' \
    --data-binary @/tmp/eva-local-12-images.json \
    | jq -r '.choices[0].message.content // .error.message'
done
```

All four endpoints must return a text description.

## 6. Verify Luxriot Channels

Log in from terminal. Use the EVA AI admin password.

```bash
read -rsp "EVA admin password: " EVA_PASSWORD
echo

curl -sS -c /tmp/eva.cookies \
  -H 'Content-Type: application/json' \
  -X POST http://127.0.0.1:5000/auth/login \
  -d "{\"username\":\"admin\",\"password\":\"${EVA_PASSWORD}\"}" | jq '.success, .user.username, .user.allowedChannelIds'

unset EVA_PASSWORD
```

List channels:

```bash
curl -sS -b /tmp/eva.cookies \
  "http://127.0.0.1:5000/luxriot/channels?force=1" \
  | jq '.channels[] | {id,title}'
```

If expected channels are missing, check:

- `EVOSSEARCH_LUXRIOT_BASE_URL`;
- `EVOSSEARCH_LUXRIOT_USERNAME`;
- `EVOSSEARCH_LUXRIOT_PASSWORD`;
- Luxriot Evo channel permissions.

## 7. Start Live Video Descriptions After EVA AI Restart

Important: live video-description sessions are in-process runtime state. After
`eva-ai` restarts, channels must be started again.

From UI:

1. Open EVA AI.
2. Log in.
3. Open Video tab.
4. Select a channel.
5. Select `Auto balance`.
6. Click `Start summaries`.
7. Repeat for required channels.

Bulk terminal start:

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

Replace the channel list with the client-site channel IDs.

## 8. Check Auto-Balancer Distribution

Run on EVA AI:

```bash
curl -sS -b /tmp/eva.cookies \
  http://127.0.0.1:5000/luxriot/streams \
  | jq -r '.video_streams[] | [.channel_id,.model,.pending_frames,.queue_submissions,.queue_dropped_batches,.last_error] | @tsv'

curl -sS -b /tmp/eva.cookies \
  http://127.0.0.1:5000/luxriot/streams \
  | jq -r '.video_streams[].model' | sort | uniq -c
```

Notes:

- The current auto-balancer uses stable channel-id hashing, not dynamic
  least-loaded scheduling.
- With only a few channels, distribution can be uneven.
- With many channels, distribution should spread better.
- If a channel is stuck at `pending_frames: 120`, inspect the profile shown in
  the `model` column and test that specific vLLM endpoint.

## 9. Stop Bad Or Accidental Channels

```bash
CSRF="$(awk '$6=="eva_csrf" {print $7}' /tmp/eva.cookies | tail -1)"

for ch in 1 999; do
  echo "Stopping channel ${ch}"
  curl -sS -b /tmp/eva.cookies \
    -H "X-CSRF-Token: ${CSRF}" \
    -H 'Content-Type: application/json' \
    -X POST http://127.0.0.1:5000/luxriot/stop_capture \
    -d "{\"channel_id\":${ch}}" | jq
done
```

## 10. Common Failure Patterns

### `systemctl` shows `status=217/USER`

The service unit has the wrong Linux user. Rewrite the vLLM service with the
actual account:

```bash
id -un
id -gn
sudo systemctl cat eva-vllm-gpu0 --no-pager
sudo systemctl cat eva-vllm-gpu1 --no-pager
```

The unit `User=` and `Group=` must match the actual account on that server.

### `/v1/models` shows `max_model_len: 4096`

An old vLLM process is still running. Use the stale-listener cleanup from
section 1.

### One live channel accumulates `pending_frames: 120`

The channel is running, but its assigned VLM profile is not consuming batches.
Check the profile name in `/luxriot/streams`, then test the matching endpoint.

### Terminal curl says `Authentication required`

The browser login does not create `/tmp/eva.cookies`. Run the terminal login
from section 6.

### Agent does not see a channel

The agent tools are filtered by the current user's `allowedChannelIds`. For
setup, use an admin account with `allowedChannelIds: ["*"]`.
