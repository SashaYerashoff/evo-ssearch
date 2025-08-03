# evo-ssearch

A CLIP-powered natural language image search application with semantic similarity using OpenAI's CLIP model and FAISS indexing.

## Two Application Versions:
- **oldapp.py**: **Main application** - Full-featured interface with comprehensive search, commenting, and management capabilities
- **app.py**: Minimal glassmorphic interface - Simplified version with dynamic background effects

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

## Running the Server

### Main Application (`oldapp.py`) - **Recommended**
```sh
python oldapp.py
```
- Open your browser and go to: [http://localhost:5000](http://localhost:5000)
- Features: Full-featured interface with advanced search controls, comment system, image management, and comprehensive functionality.

### Minimal Interface (`app.py`)
```sh
python app.py
```
- Open your browser and go to: [http://localhost:5000](http://localhost:5000)
- Features: Modern glassmorphic UI with dynamic background effects (last enlarged image becomes the background).

### Alternative Interface (`oldapp.py`) - **Legacy Name**
```sh
python oldapp.py
```
- Open your browser and go to: [http://localhost:5000](http://localhost:5000)
- Features: Comprehensive search and management interface with advanced functionality.

## Features

### Search Capabilities
- **Text Search**: Natural language descriptions (e.g., "red car", "sunset over mountains")
- **Image Search**: Upload an image to find visually similar images
- **Configurable Results**: Choose result limits from 6 to 60 images
- **Smart Limit Handling**: Returns available results even when limit exceeds total matches
- **Time-based Sorting**: Sort search results by modification time (newest first) or similarity
- **Commented Image View**: Display all images that have user comments with comment counts and latest comment previews

### Image Management
- **Original Image Viewing**: Click thumbnails to view full-resolution images
- **Comment System**: Add timestamped comments to any image
- **Comment History**: View all previous comments with timestamps
- **Persistent Storage**: Comments saved locally with search index
- **Clipboard Integration**: Copy image paths and filenames to clipboard with one click

### User Interface (Main Application - oldapp.py)
- **Dual Search Modes**: Toggle between text and image search
- **Advanced Controls**: Sort by similarity or time, adjustable result limits
- **Commented Images View**: Dedicated button to show only images with comments
- **Dark Theme**: Clean, modern dark interface
- **Responsive Grid**: Auto-adjusting image grid layout
- **Expandable Results**: Click to expand images and access features
- **Real-time Feedback**: Loading states and error handling

### Technical Features
- **CLIP Embeddings**: Semantic similarity matching using OpenAI's CLIP
- **FAISS Indexing**: Fast similarity search with persistent indexes
- **File Metadata Tracking**: Automatic capture of modification times and file sizes during indexing
- **Robust Error Handling**: Graceful handling of corrupted or missing images
- **Security**: Input validation and XSS protection
- **Cross-platform**: Works on Windows and Linux

## How to Use

1. **Index a Folder**: Enter the path to your image folder and click "Index Folder"
2. **Search Images**: 
   - Type a natural language description, or
   - Upload an image for similarity search
3. **Configure Search**: 
   - Choose sorting by similarity or time (newest first)
   - Adjust the number of results (6-60)
4. **View Results**: Browse thumbnail results with similarity scores or timestamps
5. **Expand Images**: Click any result to view the full-resolution image
6. **Add Comments**: In expanded view, add comments that persist across searches
7. **View Commented Images**: Click "Show Commented Images" to see only images with comments
8. **Copy Paths**: Use the copy button in expanded view to copy image paths to clipboard

## File Structure

- `oldapp.py` - **Main application** with full feature set and comprehensive interface
- `app.py` - Minimal glassmorphic interface version
- `requirements.txt` - Python dependencies
- `CLAUDE.md` - Development documentation
- `.clip_index/` - Created in indexed folders containing:
  - `index.faiss` - FAISS vector index
  - `paths.pkl` - Image file paths
  - `metadata.pkl` - File modification times and sizes
  - `comments.json` - User comments and timestamps

## Notes
- On Windows, the applications automatically handle OpenMP runtime issues for FAISS/CLIP.
- Indexing and searching work with folders containing images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`).
- For best performance, use a CUDA-capable GPU and the appropriate PyTorch version.
- Comments are stored locally in the indexed folder and persist across sessions.
- **Recommended**: Use `oldapp.py` for the full-featured experience with all capabilities.

## Troubleshooting
- If you see errors about OpenMP or `libiomp5md.dll`, ensure you are using the provided scripts (they set the required environment variable).
- If you have issues with dependencies, try upgrading pip and reinstalling requirements.
- If search returns fewer results than expected, check that images are in supported formats.
- Comments not saving? Ensure the application has write permissions to the indexed folder.

---

**Enjoy fast, natural language image search with CLIP and comprehensive image management using the main application (`oldapp.py`)!**
EOF < /dev/null
