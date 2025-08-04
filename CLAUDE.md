# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

evo-ssearch is a CLIP-powered image search server with a Flask-based interface:
- **oldapp.py**: Modern dark UI with text and image search capabilities, featuring SVG overlay controls and comprehensive image management

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

### Running the Application
```bash
# Run the image search server
python oldapp.py
```

**Network Access**: The server binds to all network interfaces and displays available URLs on startup:
- Local: `http://localhost:5000`
- Network: `http://192.168.1.104:5000` (or your machine's IP)
- IPv6: Available if configured

## Configuration

**config.py** provides centralized configuration for oldapp.py with environment variable support:

### Server Configuration
- `EVOSSEARCH_HOST` (default: '0.0.0.0') - Server host (0.0.0.0 for network access)
- `EVOSSEARCH_PORT` (default: 5000) - Server port
- `EVOSSEARCH_DEBUG` (default: False) - Debug mode

### Search Configuration  
- `EVOSSEARCH_MIN_RESULTS` (default: 3) - Minimum search results
- `EVOSSEARCH_MAX_RESULTS` (default: 48) - Maximum search results
- `EVOSSEARCH_DEFAULT_RESULTS` (default: 12) - Default search results

### Model Configuration
- `EVOSSEARCH_CLIP_MODEL` (default: 'ViT-B/32') - CLIP model variant

### Example Usage
```bash
# Run on different port
EVOSSEARCH_PORT=8080 python oldapp.py

# Use different CLIP model
EVOSSEARCH_CLIP_MODEL=ViT-L/14 python oldapp.py

# Disable debug mode
EVOSSEARCH_DEBUG=False python oldapp.py
```

## Architecture

### Core Components

**CLIP Integration**: Uses OpenAI's CLIP model (configurable, default ViT-B/32) for:
- Text-to-image semantic search
- Image-to-image similarity search
- Feature extraction and embedding generation

**FAISS Indexing**: 
- Vector similarity search using Facebook's FAISS library
- IndexFlatIP for inner product (cosine similarity) searches
- Persistent storage in `.clip_index/` folders with `index.faiss`, `paths.pkl`, `metadata.pkl`, and `comments.json`

**Flask Web Server**:
- Single-page application with embedded HTML/CSS/JavaScript
- RESTful endpoints for indexing and searching
- CORS enabled for cross-origin requests
- Network-accessible with automatic IP detection

### UI Features

**Modern Dark Interface**:
- Clean, minimal design with dark theme
- Dual search modes (text and image-based)
- Dynamic result limits (configurable 3-48 images)
- SVG overlay controls for better UX

**Image Management**:
- Thumbnail view with expand/collapse functionality
- Fit/fill toggle for expanded images (crop vs. contain)
- Inline copy functionality for file paths
- Image commenting system with timestamps
- Sort by similarity or modification time

### API Endpoints

**Core Endpoints**:
- `GET /` - Serve frontend interface
- `POST /index` - Index folder for search  
- `POST /search` - Text-based image search
- `POST /search_by_image` - Image-based similarity search
- `POST /check_index` - Verify if folder is indexed

**Image & Comment Management**:
- `GET /image/<path:filepath>` - Serve original images
- `GET /comments` - Get comments for specific image
- `POST /comments` - Save new comment for image
- `POST /commented_images` - Get all images with comments

### File Structure

**Index Storage**: Each indexed folder gets a `.clip_index/` subdirectory containing:
- `index.faiss` - FAISS vector index
- `paths.pkl` - Pickled list of image file paths
- `metadata.pkl` - Image metadata (modification time, file size)
- `comments.json` - User comments with timestamps

**UI Assets**: 
- `images/` - SVG icons for UI controls (expand, collapse, copy)

**Configuration**:
- `config.py` - Centralized configuration with environment variable support

**Supported Formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

## Environment Notes

**Windows Compatibility**: The application sets `KMP_DUPLICATE_LIB_OK=TRUE` to handle OpenMP runtime conflicts with FAISS/CLIP on Windows.

**CUDA Support**: Automatic GPU detection with fallback to CPU for optimal performance across different hardware configurations.

**Dependencies**: Core stack includes Flask, CORS, PyTorch, CLIP, FAISS, PIL, and NumPy.

**Network Access**: Server binds to all interfaces (0.0.0.0) by default for local network accessibility, with automatic IP detection and display.