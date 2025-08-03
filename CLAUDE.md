# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

evo-ssearch is a CLIP-powered image search server with two Flask-based interfaces:
- **app.py**: Modern glassmorphic UI with dynamic backgrounds
- **oldapp.py**: Clean, dark minimal UI with text and image search capabilities

## Development Commands

### Setup and Installation
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\Activate.ps1

# Activate virtual environment (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Running the Applications
```bash
# Run glassmorphic interface (app.py)
python app.py

# Run minimal interface (oldapp.py) 
python oldapp.py
```

Both applications run on `http://localhost:5000` with debug mode enabled.

## Architecture

### Core Components

**CLIP Integration**: Both applications use OpenAI's CLIP model (ViT-B/32) for:
- Text-to-image semantic search
- Image-to-image similarity search
- Feature extraction and embedding generation

**FAISS Indexing**: 
- Vector similarity search using Facebook's FAISS library
- IndexFlatIP for inner product (cosine similarity) searches
- Persistent storage in `.clip_index/` folders with `index.faiss` and `paths.pkl` files

**Flask Web Server**:
- Single-page applications with embedded HTML/CSS/JavaScript
- RESTful endpoints for indexing and searching
- CORS enabled for cross-origin requests

### Key Differences Between Applications

**app.py** (Glassmorphic):
- More compact code with embedded CSS/HTML template
- Background changes to last clicked image
- Text search only
- Batch processing with configurable BATCH_SIZE (32)
- Optimized CUDA device detection with capability checks

**oldapp.py** (Minimal):
- Comprehensive UI with dual search modes (text and image)
- Separate functions for PIL image processing
- Configurable result limits (6-60 images)
- More detailed error handling and status reporting

### API Endpoints

**Common Endpoints**:
- `GET /` - Serve frontend interface
- `POST /index` - Index folder for search
- `POST /search` - Text-based image search
- `POST /check_index` - Verify if folder is indexed

**oldapp.py Additional**:
- `POST /search_by_image` - Image-based similarity search

### File Structure

**Index Storage**: Each indexed folder gets a `.clip_index/` subdirectory containing:
- `index.faiss` - FAISS vector index
- `paths.pkl` - Pickled list of image file paths

**Supported Formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

## Environment Notes

**Windows Compatibility**: Both apps set `KMP_DUPLICATE_LIB_OK=TRUE` to handle OpenMP runtime conflicts with FAISS/CLIP on Windows.

**CUDA Support**: Automatic GPU detection with fallback to CPU. app.py includes specific CUDA capability filtering for optimal performance.

**Dependencies**: Core stack includes Flask, CORS, PyTorch, CLIP, FAISS, PIL, and NumPy.