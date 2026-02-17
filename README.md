# SISU (Smart Image Search and Understanding)

SISU is a CLIP/DINO-powered search and monitoring toolkit with Luxriot Evo S integration, background probes, and vision LLM summaries. “Sisu” is also a Finnish word for grit, resilience, and determination—fitting for a system that keeps watching.

## Features

**Monitoring & Probes (Luxriot Evo S)**
- Live Luxriot channel preview and user-visible system prompt (no hidden LLM prompts)
- Continuous per-channel capture feeding per-probe FAISS buffers
- Text probes and image probes (image-only supported); enable/disable per probe
- Background probe runner executes all enabled probes across channels; per-probe FPS hint
- Bookmark sending to Luxriot with severity control and bookmark toggle
- Saved probe cards with thumbnails/status, inline “New probe” card
- Latest detections carousel with hit metadata

**Benchmarks & Layout**
- Built-in GPU embed benchmark (/probes/bench) surfaced in UI
- Monitoring layout optimized for many probes: cards/benchmark on top, editor/detections below

**Search & Browse (tools in the toolkit)**
- Text search (CLIP) and image search (upload/path) with FAISS indexing
- Expand, find similar, copy path, and comment on images (persistent)
- Sort by similarity or time; adjustable result count

**Configurable & Accessible**
- Settings modal writes `.env`; all prompts visible and editable
- Network-accessible; CORS enabled
- DINOv3 support for image search/segmentation (manual weights download; see below)

## Prerequisites
- Windows 10/11 or Ubuntu 20.04+
- Python 3.10+ (64-bit recommended)
- Git
- CUDA-capable GPU recommended for probes/monitoring (NVIDIA, drivers + CUDA toolkit)
- **LM Studio** with a vision-capable model (e.g., Qwen3-VL) running, if using Video Understanding
- **Luxriot Evo S** server up and reachable, if using monitoring/bookmarks
- **vLLM** (optional, faster Video Understanding) with a vision model such as Qwen/Qwen3-VL-4B-Instruct

## Installation & Setup

### 1. Open Terminal
- **Windows**: Press `Win + S`, type `PowerShell`, and open it.
- **Ubuntu**: Press `Ctrl + Alt + T`.

### 2. Clone the Repository
```sh
git clone https://github.com/SashaYerashoff/evo-ssearch.git
cd evo-ssearch
# Monitoring/VideoUnderstanding live on branch
git checkout lxrt-inntegration
```

### 3. Create a Virtual Environment
```sh
python -m venv .venv
```

### 4. Activate the Virtual Environment
- **Windows**:
  ```sh
  .venv\Scripts\Activate.ps1
  ```
- **Ubuntu**:
  ```sh
  source .venv/bin/activate
  ```

### 5. Install Dependencies
```sh
pip install --upgrade pip
pip install -r requirements.txt

# (Optional) DINOv3 weights (for image search/segmentation)
# Download weights manually and place in embedders/dino_encoder or per your config.
# Example (adjust model name/path as needed):
# wget -O embedders/dino_encoder/dinov3_vitb16_pretrain.pth https://dl.fbaipublicfiles.com/dinov3/dinov3_vitb16_pretrain.pth

# (Optional) vLLM for Video Understanding (vision)
# Clone HF model locally (example: Qwen/Qwen3-VL-4B-Instruct)
# git clone https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct /path/to/Qwen3-VL-4B-Instruct
# Start vLLM (CUDA):
# export VLLM_WORKER_USE_VISION_ENGINE=1
# vllm serve /path/to/Qwen3-VL-4B-Instruct --port 8000 --api-type openai --trust-remote-code --max-model-len 4096
# Then set EVOSSEARCH_LM_BASE_URL=http://localhost:8000/v1 and EVOSSEARCH_LM_MODEL=Qwen/Qwen3-VL-4B-Instruct
```

## Running the Application

```sh
python oldapp.py
```

The server will display available URLs on startup:
- **Local**: [http://localhost:5000](http://localhost:5000)
- **Network**: http://[your-ip]:5000 (accessible from other devices on your network)

### Production Run (Gunicorn)

```sh
# install dependencies first
./run_prod.sh
```

Optional runtime knobs:

```bash
EVOSSEARCH_GUNICORN_WORKERS=2
EVOSSEARCH_GUNICORN_THREADS=4
EVOSSEARCH_GUNICORN_TIMEOUT=180
```

## Configuration

### Frontend Settings Panel (Recommended)

**Easy Configuration via Web UI:**
- Click the settings gear icon (⚙️) in the top-right corner of the application
- Modify settings in the organized modal panel
- Click "Save Settings" to persist changes to `.env` file
- Restart the server to apply new configuration

**Available Settings:**
- **Server**: Host, Port, Debug Mode
- **Search**: Min/Max/Default result limits  
- **Model**: CLIP model variant, batch size, thumbnail quality
- **Advanced**: Comment length limits, file size limits, index folder name

Settings are automatically saved to `.env` file and persist across restarts.

### Environment Variables (Advanced)

For command-line configuration or CI/CD environments:

```bash
# Server settings
EVOSSEARCH_HOST=0.0.0.0          # Server host (0.0.0.0 for network access)
EVOSSEARCH_PORT=5000             # Server port
EVOSSEARCH_DEBUG=False           # Debug mode

# Search limits
EVOSSEARCH_MIN_RESULTS=3         # Minimum search results  
EVOSSEARCH_MAX_RESULTS=48        # Maximum search results
EVOSSEARCH_DEFAULT_RESULTS=12    # Default search results

# Model configuration
EVOSSEARCH_CLIP_MODEL=ViT-B/32   # CLIP model variant
EVOSSEARCH_BATCH_SIZE=32         # Processing batch size
EVOSSEARCH_THUMBNAIL_QUALITY=85  # JPEG quality (50-100)

# LM Studio video understanding (Qwen3-VL)
EVOSSEARCH_LM_BASE_URL=http://192.168.1.104:1234/v1
EVOSSEARCH_LM_MODEL=qwen/qwen3-vl-4b
EVOSSEARCH_LM_VIDEO_DEFAULT_FRAMES=16   # default frames to sample per video
EVOSSEARCH_LM_VIDEO_MAX_FRAMES=64       # hard cap on frames sent to the model
EVOSSEARCH_LM_VIDEO_MAX_EDGE=960        # resize frames before sending
EVOSSEARCH_LM_VIDEO_MAX_TOKENS=1536     # cap model output tokens (increase if responses are cut)
EVOSSEARCH_LM_VIDEO_TEMPERATURE=0.2     # decoding temperature
EVOSSEARCH_LM_TIMEOUT=120               # request timeout (seconds)

# Luxriot Evo S live integration
EVOSSEARCH_LUXRIOT_BASE_URL=http://192.168.1.102:8080
EVOSSEARCH_LUXRIOT_USERNAME=
EVOSSEARCH_LUXRIOT_PASSWORD=
EVOSSEARCH_LUXRIOT_DEFAULT_CHANNEL_ID=103
EVOSSEARCH_LUXRIOT_SNAPSHOT_INTERVAL=5
EVOSSEARCH_LUXRIOT_SNAPSHOT_MAX_EDGE=800
EVOSSEARCH_LUXRIOT_MAX_BUFFER_FRAMES=180   # cap buffered snapshots before forced flush
EVOSSEARCH_LUXRIOT_SYSTEM_PROMPT_DEFAULT="You summarize real-time CCTV snapshots..."
EVOSSEARCH_LUXRIOT_AUTO_BOOKMARKS=False     # hidden auto-bookmarks are disabled by default

# Advanced settings
EVOSSEARCH_MAX_COMMENT_LENGTH=500 # Max comment characters
EVOSSEARCH_MAX_FILE_SIZE_MB=50   # Max upload file size
EVOSSEARCH_INDEX_FOLDER=.clip_index # Index folder name
EVOSSEARCH_SETTINGS_LOCAL_ONLY=True  # If no token is set, /settings GET is localhost-only
EVOSSEARCH_ADMIN_TOKEN=change-me      # Required for mutating endpoints
EVOSSEARCH_CORS_ALLOWED_ORIGINS=      # Optional comma-separated CORS allowlist
EVOSSEARCH_ALLOWED_ROOTS=             # Optional pathsep-separated folder allowlist for indexing/search

# Probe runner
EVOSSEARCH_PROBE_MAX_FRAMES=500         # frames kept per channel buffer
EVOSSEARCH_PROBE_MAX_STORED_HITS=30     # recent hits retained per probe
EVOSSEARCH_PROBE_DAEMON_INTERVAL_SEC=5  # background runner interval
EVOSSEARCH_PROBE_BENCH_BATCH=16         # batch size for /probes/bench

# DINOv3 weights (manual download)
# Place weights in embedders/dino_encoder or configure the path in config.py

# vLLM (Video Understanding via OpenAI-compatible endpoint)
# Set the base URL/model to point to your vLLM server running a vision model
# e.g., EVOSSEARCH_LM_BASE_URL=http://localhost:8000/v1
#       EVOSSEARCH_LM_MODEL=Qwen/Qwen3-VL-4B-Instruct
```

### Admin Token For Mutating Endpoints

Set `EVOSSEARCH_ADMIN_TOKEN` to enable mutating endpoints (indexing, settings save, Luxriot capture control, probe save/delete/run, and comment writes).

In the web UI, click the lock icon in the header to store the token in browser localStorage (or open with `?admin_token=<token>` once).

Send the token as either:

```bash
Authorization: Bearer <token>
# or
X-Admin-Token: <token>
```

### Example Usage
```bash
# Run on different port
EVOSSEARCH_PORT=8080 python oldapp.py

# Use different CLIP model  
EVOSSEARCH_CLIP_MODEL=ViT-L/14 python oldapp.py

# Change result limits
EVOSSEARCH_MIN_RESULTS=5 EVOSSEARCH_MAX_RESULTS=60 python oldapp.py
```

## How to Use

1. **Index a Folder**: Enter the path to your image folder and click "Index Folder"
2. **Search Images**: 
   - **Text Mode**: Type a natural language description
   - **Image Mode**: Upload an image file OR enter an image path for similarity search
3. **Video Understanding**:
   - Switch to the **Video Understanding** tab
   - Provide a video path, choose how many frames to sample (16/32/64), optional sampling FPS, and a prompt (can be remembered)
   - System prompt is visible/editable; bookmarks are user-controlled
   - Click **Analyze Video** to send sampled frames to Qwen3-VL via LM Studio; the response supports basic markdown formatting
4. **Monitoring (Luxriot + Probes)**:
   - Switch to **Monitoring**
   - Set system prompt (LLM role), channel, batch/FPS, and start stream
   - Create/edit probes (text or image-only), enable/disable, and save; background runner executes all enabled probes per channel
   - View saved probes (grid), run/disable/delete from cards; detections carousel shows recent hits; optional bookmarks to Luxriot
   - Run GPU benchmark (button) to estimate throughput
5. **Configure Search**: 
   - Choose sorting by similarity or time (newest first)
   - Adjust the number of results using the dropdown
6. **Interact with Results**: 
   - **Expand**: Click the expand icon (⤢) in bottom-right corner of any image
   - **Find Similar**: Click the search icon (🔍) on expanded images to find similar images
   - **Copy Path**: Click the copy icon (📋) next to the filename
   - **Add Comments**: In expanded view, add comments that persist across searches
7. **View Commented Images**: Click "Show Commented Images" to see only images with comments

## UI Controls

| Icon | Location | Function |
|------|----------|----------|
| ⚙️ (settings) | Top-right of header | Open settings panel for configuration |
| ⤢ (expand) | Bottom-right of thumbnail | Expand to full view (takes full row, 900px min-width) |
| ⤡ (collapse) | Bottom-right of expanded image | Collapse to thumbnail |
| 🔍 (find similar) | Top-right of expanded image | Find visually similar images |
| 📋 (copy) | Next to filename | Copy full file path to clipboard |

## Technical Features

**CLIP & FAISS Integration:**
- Semantic similarity matching using OpenAI's CLIP
- Fast similarity search with persistent FAISS indexes
- Configurable CLIP model variants

**Data Management:**
- File metadata tracking (modification times, file sizes)
- Persistent comment storage with timestamps
- Robust error handling for corrupted or missing images

**Network & Security:**
- CORS enabled for cross-origin requests  
- Input validation and XSS protection
- Network accessibility with automatic IP detection
- Cross-platform compatibility (Windows/Linux)

## File Structure

```
evo-ssearch/
├── oldapp.py              # Main application
├── config.py              # Configuration with environment variable support
├── .env                   # Settings file (created by settings panel)
├── requirements.txt       # Python dependencies
├── images/                # SVG icons for UI controls
│   ├── expand_content_*.svg
│   ├── collapse_content_*.svg
│   ├── content_copy_*.svg
│   └── settings_*.svg
└── [indexed-folder]/
    └── .clip_index/       # Created automatically
        ├── index.faiss    # FAISS vector index
        ├── paths.pkl      # Image file paths
        ├── metadata.pkl   # File metadata
        └── comments.json  # User comments
```

**Supported Image Formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

## Troubleshooting
- **OpenMP Errors**: The application automatically handles OpenMP runtime issues on Windows
- **Dependencies**: Try upgrading pip and reinstalling requirements if you encounter issues
- **Fewer Results**: Check that images are in supported formats and properly indexed
- **Comments Not Saving**: Ensure write permissions to the indexed folder
- **Network Access**: Make sure firewall allows connections on the configured port
- **Luxriot/Probes idle**: Ensure probe FPS > 0, probes are enabled, and the correct channel is selected; use the benchmark to size concurrency

---

**Enjoy fast, natural language image search with modern SVG overlay controls and comprehensive image management!**
