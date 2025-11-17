# evo-ssearch

A CLIP-powered natural language image search application with semantic similarity using OpenAI's CLIP model and FAISS indexing.

## Features

**Modern Dark UI with SVG Overlay Controls:**
- Clean, minimal dark interface optimized for image browsing
- SVG overlay icons for intuitive image interaction
- Network-accessible with automatic IP detection

**Dual Search Modes:**
- **Text Search**: Natural language descriptions (e.g., "red car", "sunset over mountains")
- **Image Search**: Upload an image file OR enter an image path to find visually similar images

**Advanced Image Management:**
- **Expand/Collapse**: Click overlay icon (bottom-right) to toggle between thumbnail and full view
- **Full-Width Expanded View**: Expanded images take the complete row with 900px minimum width and no cropping
- **Find Similar**: Click search icon on expanded images to find visually similar images
- **Quick Copy**: Click copy icon next to filename to copy full file path to clipboard
- **Comment System**: Add timestamped comments to any image with persistent storage

**Configurable & Accessible:**
- Dynamic result limits (3-48 images, configurable via environment variables)
- Sort by similarity or modification time (newest first)
- Network access - server accessible from any device on local network
- Comprehensive configuration via `config.py` and environment variables

## Prerequisites
- Windows 10/11 or Ubuntu 20.04+
- Python 3.10 or newer (64-bit recommended)
- Git
- (Optional) CUDA-capable GPU for faster search (NVIDIA, with drivers installed)

## Installation & Setup

### 1. Open Terminal
- **Windows**: Press `Win + S`, type `PowerShell`, and open it.
- **Ubuntu**: Press `Ctrl + Alt + T`.

### 2. Clone the Repository
```sh
git clone https://github.com/SashaYerashoff/evo-ssearch.git
cd evo-ssearch
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
```

## Running the Application

```sh
python oldapp.py
```

The server will display available URLs on startup:
- **Local**: [http://localhost:5000](http://localhost:5000)
- **Network**: http://[your-ip]:5000 (accessible from other devices on your network)

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
EVOSSEARCH_LM_VIDEO_MAX_TOKENS=768      # cap model output tokens
EVOSSEARCH_LM_VIDEO_TEMPERATURE=0.2     # decoding temperature
EVOSSEARCH_LM_TIMEOUT=120               # request timeout (seconds)

# Advanced settings
EVOSSEARCH_MAX_COMMENT_LENGTH=500 # Max comment characters
EVOSSEARCH_MAX_FILE_SIZE_MB=50   # Max upload file size
EVOSSEARCH_INDEX_FOLDER=.clip_index # Index folder name
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
   - Click **Analyze Video** to send sampled frames to Qwen3-VL via LM Studio; the response supports basic markdown formatting
4. **Configure Search**: 
   - Choose sorting by similarity or time (newest first)
   - Adjust the number of results using the dropdown
5. **Interact with Results**: 
   - **Expand**: Click the expand icon (⤢) in bottom-right corner of any image
   - **Find Similar**: Click the search icon (🔍) on expanded images to find similar images
   - **Copy Path**: Click the copy icon (📋) next to the filename
   - **Add Comments**: In expanded view, add comments that persist across searches
6. **View Commented Images**: Click "Show Commented Images" to see only images with comments

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

---

**Enjoy fast, natural language image search with modern SVG overlay controls and comprehensive image management!**
