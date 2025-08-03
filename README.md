# evo-ssearch

A CLIP-powered image search server with two UIs:
- **app.py**: Modern glassmorphic interface, uses the last enlarged image as the background.
- **oldapp.py**: Clean, dark UI with minimal design and search-by-image functionality.

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

### For the Glassmorphic Interface (`app.py`)
```sh
python app.py
```
- Open your browser and go to: [http://localhost:5000](http://localhost:5000)
- Features: Modern glassmorphic UI, last enlarged image becomes the background.

### For the Minimal Backend (`oldapp.py`)
```sh
python oldapp.py
```
- Open your browser and go to: [http://localhost:5000](http://localhost:5000)
- Features: Clean, dark UI with search-by-image functionality.

## Search Modes

### Text Search
- Enter a natural language description of what you're looking for
- Example: "a red car", "sunset over mountains", "people playing basketball"

### Image Search (oldapp.py only)
- Upload an image to find visually similar images in your indexed folder
- Supports common image formats (jpg, png, webp, etc.)
- Uses CLIP embeddings for semantic similarity matching

## Notes
- On Windows, the scripts automatically handle OpenMP runtime issues for FAISS/CLIP.
- Indexing and searching work with folders containing images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`).
- For best performance, use a CUDA-capable GPU and the appropriate PyTorch version.

## Troubleshooting
- If you see errors about OpenMP or `libiomp5md.dll`, ensure you are using the provided scripts (they set the required environment variable).
- If you have issues with dependencies, try upgrading pip and reinstalling requirements.

---

Enjoy fast, natural language image search with CLIP!
