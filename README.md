# Luxriot EVA AI (Smart Image Search and Understanding)

Luxriot EVA AI is a production-pilot CLIP visual search and monitoring system for Luxriot-integrated workflows. It combines:

- Archive research over indexed folders and detections archive
- Video understanding via OpenAI-compatible vision models (LM Studio/vLLM)
- Live video-description monitoring with VLM alerts, coverage-aware reports, and bookmark actions
- Probe-based CLIP/P/N/M tracking as a secondary semantic signal for tuning and large archive comparison

## Version

- Current version: `β 0.8.1`
- Source of truth: `VERSION` (UI reads it at runtime; optional override via `EVOSSEARCH_APP_VERSION`)

### Versioning Policy

- Release label + `major.minor.patch` (shown in UI from `VERSION`, e.g. `β 0.8.1`)
- New feature branch work: increment `minor`
- Minor adjustments inside a branch: increment `patch`
- Merge to `main`: increment `major`, reset `minor` and `patch` to `0`

## Current Scope

This branch is a beta production-pilot build focused on:

- Stable end-to-end UX in 3 tabs: `Archive Research`, `Video Understanding`, `Monitoring`
- Secure mutation paths via admin token
- Runtime configuration through Settings + `.env` editor
- Video-description archive evidence, VLM alert frames, and retention controls for useful snapshots

## Core Features

### Archive Research

- Text and image query in one workspace
- Search scope switch:
  - `Indexed Folder`: FAISS search over local `.clip_index`
  - `Detections Archive`: search over persisted probe hits, VLM summary frames, and VLM alert frames
- Search modes: CLIP, DINO, or fusion (when enabled)
- Sort by similarity or time
- Expand result, copy path, and run "find similar"
- Comments and interactive segmentation for indexed-folder images
- LLM image description from expanded image, with save-as-comment

Note: Comments and segmentation are intentionally limited to indexed-folder images.

### Video Understanding

- Offline video analysis (`/video_understanding`) with sampled frames
- Configurable frames, sample FPS, prompt, and model id
- Luxriot live summaries panel with visible system prompt
- Active stream manager:
  - stop individual stream channels
  - stop all video streams
  - stop all analytics streams

### Monitoring (Luxriot + Probes)

- Live Luxriot preview by channel
- Saved probe cards with latest detections strip and stream controls
- Probe editor modal:
  - text pairs (positive/negative)
  - optional image probe
  - per-probe thresholds, severity, and bookmark behavior
- Background probe daemon over active channels
- CLIP throughput benchmark (`/probes/bench`)

### Detections Archive + Retention

- Probe hits, VLM-sampled summary frames, and VLM alert frames are persisted in PostgreSQL in secure deployments
- Optional adaptive retention keeps high-value snapshots and drops near-duplicates
- Archive search returns source labels, visual evidence URLs, and search similarity signals

## Prerequisites

- Python 3.10+ (3.13 is supported in current branch)
- Git
- CUDA GPU recommended for DINO/Mask2Former/probe-heavy workloads
- LM Studio or vLLM (optional, for video understanding)
- Luxriot Evo S (optional, for live monitoring/bookmarks)

## Installation

```bash
git clone <your-repo-url>
cd evo-ssearch
git checkout <feature-branch>

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

### DINO Weights

DINO requires local weights if you use DINO/fusion paths:

```bash
# example only; choose the weight file that matches EVOSSEARCH_DINO_MODEL
wget -O /path/to/dinov3_weights.pth <weights-url>
```

Then set `EVOSSEARCH_DINO_WEIGHTS_PATH`.

### Windows 11 Quickstart

Use this for the fastest first run on a demo laptop.

1. Open PowerShell in the repo folder.
2. Create and activate venv.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

3. Install dependencies.

```powershell
pip install -r requirements.txt
```

4. Optional stability-first `.env` profile (recommended if GPU/CUDA is limited):

```env
EVOSSEARCH_EMBEDDER=clip
EVOSSEARCH_DINO_SEGMENTS_ENABLED=false
EVOSSEARCH_M2F_ENABLED=false
```

5. Start Luxriot EVA AI.

```powershell
python oldapp.py
```

6. Open:

```text
http://localhost:5000
```

Windows notes:

- `run_prod.sh` is Linux-oriented; use `python oldapp.py` on Windows.
- If DINO/fusion is enabled, set a valid Windows path for `EVOSSEARCH_DINO_WEIGHTS_PATH`.
- If execution policy blocks activation, run:
  - `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

## Running

### Dev

```bash
python oldapp.py
```

Startup now validates port availability. If occupied, startup aborts with a clear message.

Default URL:

- `http://localhost:5000`

### Production (Gunicorn)

```bash
./run_prod.sh
```

Optional `run_prod.sh` knobs:

```bash
EVOSSEARCH_GUNICORN_WORKERS=2
EVOSSEARCH_GUNICORN_THREADS=4
EVOSSEARCH_GUNICORN_TIMEOUT=180
EVOSSEARCH_GUNICORN_BIN=/path/to/gunicorn
```

## Configuration

### Settings UI (Recommended)

- Click lock icon to set admin token in browser (`localStorage`)
- Click gear icon to open settings modal
- Save settings to `.env`
- Use Environment Variables section to view/edit all `EVOSSEARCH_*` keys
- Restart server for full env application

### Admin Token

Mutating endpoints require `EVOSSEARCH_ADMIN_TOKEN` when set.

Accepted headers:

```text
X-Admin-Token: <token>
Authorization: Bearer <token>
```

UI convenience:

- Open once with `?admin_token=<token>` to seed browser storage

### Luxriot Evo Bookmark Setup (Integration)

Use this checklist to ensure bookmarks sent by Luxriot EVA AI are accepted and visible in Luxriot Evo.

1. Prepare Luxriot API access.
   - Confirm Luxriot base URL is reachable from Luxriot EVA AI host (example: `http://<luxriot-host>:8080`).
   - Use a Luxriot user that has permissions to:
     - read channels/snapshots
     - create bookmarks/events
   - Luxriot EVA AI uses HTTP Digest auth for Luxriot API calls.
2. Configure Luxriot EVA AI connection values.
   - Set `EVOSSEARCH_LUXRIOT_BASE_URL`
   - Set `EVOSSEARCH_LUXRIOT_USERNAME`
   - Set `EVOSSEARCH_LUXRIOT_PASSWORD`
   - Set `EVOSSEARCH_LUXRIOT_DEFAULT_CHANNEL_ID`
   - Optionally align severity mapping:
     - `EVOSSEARCH_LUXRIOT_SEV_INFO`
     - `EVOSSEARCH_LUXRIOT_SEV_LOW`
     - `EVOSSEARCH_LUXRIOT_SEV_NORMAL`
     - `EVOSSEARCH_LUXRIOT_SEV_HIGH`
     - `EVOSSEARCH_LUXRIOT_SEV_CRITICAL`
3. Restart Luxriot EVA AI after config/env changes.
4. Verify Luxriot connectivity in UI.
   - Open `Video Understanding` or `Monitoring`.
   - Click channel reload and preview.
   - If channels/snapshots fail, fix connection/auth first.
5. Send a direct test bookmark through Luxriot EVA AI.

```bash
curl -X POST "http://localhost:5000/luxriot/bookmark" \
  -H "Content-Type: application/json" \
  -H "X-Admin-Token: <your-admin-token>" \
  -d '{
    "channel_id": 103,
    "title": "Luxriot EVA AI integration test",
    "description": "Bookmark created by Luxriot EVA AI /luxriot/bookmark",
    "severity": "normal",
    "state": "new"
  }'
```

Expected response contains `"success": true`.

6. Verify automatic bookmark sources.
   - Probe path: in `Monitoring` -> probe editor, keep `Make bookmarks` enabled, set severity, run/save probe.
   - Manual summary path: in `Video Understanding` live summaries, click bookmark on a summary row.
7. Confirm in Luxriot.
   - Open Luxriot bookmarks/events for the target channel.
   - Check event title, description, severity, and timestamp near the trigger time.

If direct test works but probe bookmarks do not, verify the probe has bookmarks enabled and returns hits.

### Environment Variables

Effective variables currently used by app/config:

```bash
# Server
EVOSSEARCH_HOST=0.0.0.0
EVOSSEARCH_PORT=5000
EVOSSEARCH_DEBUG=false
EVOSSEARCH_APP_VERSION="β 0.8.1"

# Embedder/index
EVOSSEARCH_EMBEDDER=clip              # clip|dino|fusion
EVOSSEARCH_CLIP_MODEL=ViT-B/32        # OpenAI CLIP or HF SigLIP2 model id (e.g. google/siglip2-base-patch16-224)
EVOSSEARCH_DINO_MODEL=dinov3_vith16plus
EVOSSEARCH_EMB_DIM_DINO=1280
EVOSSEARCH_DINO_WEIGHTS_PATH=/path/to/weights.pth
EVOSSEARCH_DINO_DEVICE=cuda:0
EVOSSEARCH_INDEX_MODE=clip            # clip|dino|dual
EVOSSEARCH_FUSION_ENABLED=false
EVOSSEARCH_FUSION_ALPHA=0.7
EVOSSEARCH_RERANK_ENABLED=false
EVOSSEARCH_RERANK_TOP_K=50
EVOSSEARCH_DINO_SEGMENTS_ENABLED=false
EVOSSEARCH_DINO_SEGMENT_MIN_PATCHES=3
EVOSSEARCH_DINO_HEATMAP_THRESHOLD=0.7

# Mask2Former
EVOSSEARCH_M2F_ENABLED=true
EVOSSEARCH_M2F_MODEL=facebook/mask2former-swin-base-ade-semantic
EVOSSEARCH_M2F_DEVICE=cuda:0
EVOSSEARCH_M2F_MAX_SIZE=1024

# Video LM
EVOSSEARCH_LM_BASE_URL=http://127.0.0.1:1234/v1
EVOSSEARCH_LM_MODEL=qwen/qwen3-vl-4b
EVOSSEARCH_LM_API_KEY=
EVOSSEARCH_LM_TIMEOUT=120
EVOSSEARCH_LM_VIDEO_DEFAULT_FRAMES=16
EVOSSEARCH_LM_VIDEO_MAX_FRAMES=64
EVOSSEARCH_LM_VIDEO_MAX_EDGE=960
EVOSSEARCH_LM_VIDEO_MAX_TOKENS=1536
EVOSSEARCH_LM_VIDEO_TEMPERATURE=0.2
EVOSSEARCH_OFFLINE_VIDEO_ENABLED=false
EVOSSEARCH_PROBE_SNAP_ENABLED=false
EVOSSEARCH_INDEXED_FOLDER_ENABLED=false

# Luxriot
EVOSSEARCH_LUXRIOT_BASE_URL=http://luxriot-host:8080
EVOSSEARCH_LUXRIOT_USERNAME=
EVOSSEARCH_LUXRIOT_PASSWORD=
EVOSSEARCH_LUXRIOT_DEFAULT_CHANNEL_ID=1
EVOSSEARCH_LUXRIOT_SNAPSHOT_INTERVAL=5
EVOSSEARCH_LUXRIOT_SNAPSHOT_MAX_EDGE=800
EVOSSEARCH_LUXRIOT_MAX_BUFFER_FRAMES=180
EVOSSEARCH_LUXRIOT_AUTO_BOOKMARKS=false
EVOSSEARCH_LUXRIOT_SEV_INFO=info
EVOSSEARCH_LUXRIOT_SEV_LOW=low
EVOSSEARCH_LUXRIOT_SEV_NORMAL=normal
EVOSSEARCH_LUXRIOT_SEV_HIGH=high
EVOSSEARCH_LUXRIOT_SEV_CRITICAL=critical

# Probe capture
EVOSSEARCH_PROBE_MAX_FRAMES=2000
EVOSSEARCH_PROBE_THUMB_MAX_EDGE=256

# Detections archive + retention
EVOSSEARCH_DETECTIONS_ARCHIVE_ENABLED=true
EVOSSEARCH_DETECTIONS_ARCHIVE_DIR=detections_archive
EVOSSEARCH_DETECTIONS_ARCHIVE_JPEG_QUALITY=88
EVOSSEARCH_DETECTIONS_RETENTION_ENABLED=true
EVOSSEARCH_DETECTIONS_RETENTION_DROP_SKIPPED=false
EVOSSEARCH_DETECTIONS_RETENTION_WINDOW_SEC=6
EVOSSEARCH_DETECTIONS_RETENTION_FORCE_KEEP_SEC=20
EVOSSEARCH_DETECTIONS_RETENTION_SIMILARITY_HIGH=0.985
EVOSSEARCH_DETECTIONS_RETENTION_SIMILARITY_LOW=0.94
EVOSSEARCH_DETECTIONS_RETENTION_MARGIN_DELTA=0.08
EVOSSEARCH_DETECTIONS_RETENTION_SCORE_DELTA=0.08

# Search limits and processing
EVOSSEARCH_MIN_RESULTS=3
EVOSSEARCH_MAX_RESULTS=48
EVOSSEARCH_DEFAULT_RESULTS=12
EVOSSEARCH_BATCH_SIZE=32
EVOSSEARCH_THUMBNAIL_QUALITY=85
EVOSSEARCH_INDEX_FOLDER=.clip_index
EVOSSEARCH_MAX_COMMENT_LENGTH=100
EVOSSEARCH_MAX_FILE_SIZE_MB=50

# Security/network
EVOSSEARCH_ADMIN_TOKEN=
EVOSSEARCH_SETTINGS_LOCAL_ONLY=true
EVOSSEARCH_CORS_ALLOWED_ORIGINS=
EVOSSEARCH_ALLOWED_ROOTS=
```

Notes:

- CORS is not globally open by default; set `EVOSSEARCH_CORS_ALLOWED_ORIGINS` explicitly.
- `.env` edits require restart for full consistency.
- `EVOSSEARCH_CLIP_MODEL` supports OpenAI CLIP names and SigLIP2 HF IDs.
- If a SigLIP2 model fails to load, Luxriot EVA AI auto-falls back to `ViT-B/32` so startup can continue.

## Typical Workflows

### 1) Index + Search Folder

1. Set folder path in top panel.
2. Click `Index Folder`.
3. In `Archive Research`, keep scope = `Indexed Folder`.
4. Run text query or image upload query.

### 2) Search Video-Description Evidence

1. In `Archive Research`, set scope = `Detections Archive`.
2. Filter by stream, source (`Video descriptions` / `VLM alerts`), and time range.
3. Run text/image search over saved VLM frames and open evidence thumbnails.

### 3) Live Video Descriptions

1. Open `Video`.
2. Select a Luxriot channel, cadence, batch size, and VLM model/profile.
3. Click `Start summaries`.
4. Review `VLM Feed`, alert badges, archive frames, and stream health.
5. Use `Agent` -> `Stream status` / `Video report` for operator-facing status and reports.

### 4) Monitoring + Probes

Use `Monitoring` when an engineer explicitly needs CLIP/P/N/M probes for semantic comparison, threshold tuning, or secondary corroboration. Probe state is not the default operator report center for the current pilot.

## Project Layout

```text
evo-ssearch/
├── oldapp.py
├── config.py
├── luxriot_connector.py
├── probe_manager.py
├── embedders/
├── heads/
├── tools/
├── tests/
├── run_prod.sh
├── wsgi.py
└── [indexed-folder]/.clip_index/
```

Runtime artifacts:

- PostgreSQL schemas for archive/search/probes/agent runtime state
- `/var/lib/eva-ai/detections_archive/` or configured retained snapshot storage

## Troubleshooting

- `Startup aborted: host:port is already in use`
  - stop previous process or change `EVOSSEARCH_PORT`
- CUDA `CUBLAS_STATUS_ALLOC_FAILED`
  - reduce concurrent streams/probes, lower frame sizes, or restart stale GPU processes
- DINO/fusion returns errors
  - verify `EVOSSEARCH_DINO_WEIGHTS_PATH` and CUDA device
- Remote settings blocked
  - expected when `EVOSSEARCH_SETTINGS_LOCAL_ONLY=true` and no valid admin token
- Mutating calls return `401/503`
  - verify admin token in lock panel or request headers

## Supported Image Formats

`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
