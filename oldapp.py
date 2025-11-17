import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import base64
import json
import pickle
import time
import math
import requests
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
from urllib.parse import unquote
from threading import Lock

import numpy as np
import torch
import cv2
import clip
import faiss
from PIL import Image
from flask import Flask, request, jsonify, send_file, render_template_string, make_response
from flask_cors import CORS

from config import config
from embedders.dino_encoder import DINOEncoder
try:
    from heads.mask2former_head import Mask2FormerHead
except Exception:  # pragma: no cover - optional dependency
    Mask2FormerHead = None  # type: ignore[misc]

app = Flask(__name__)
CORS(app)

# Global embedder state
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model: Optional[torch.nn.Module] = None
clip_preprocess = None
dino_encoder: Optional[DINOEncoder] = None
mask2former_head: Optional["Mask2FormerHead"] = None
_mask2former_lock = Lock()
_mask2former_failed = False
SUPPORTED_EMBEDDERS = {"clip", "dino", "fusion"}
EMBEDDER_SUBDIRS: Dict[str, str] = {"clip": "clip", "dino": "dino"}
active_embedder = config.EMBEDDER if config.EMBEDDER in SUPPORTED_EMBEDDERS else "clip"
if active_embedder == "fusion" and not config.FUSION_ENABLED:
    active_embedder = "clip"


def init_clip() -> None:
    """Load the CLIP model lazily for embedding extraction."""
    global clip_model, clip_preprocess
    if clip_model is not None and clip_preprocess is not None:
        return
    clip_model, clip_preprocess = clip.load(config.CLIP_MODEL, device=device)
    clip_model.eval()


def init_dino() -> None:
    """Load the DINO encoder lazily."""
    global dino_encoder
    if dino_encoder is not None:
        return
    weights_path = (config.DINO_WEIGHTS_PATH or "").strip()
    if not weights_path:
        raise RuntimeError(
            "EVOSSEARCH_DINO_WEIGHTS_PATH is not set. Provide a local DINO checkpoint to keep inference on GPU."
        )
    weights_file = Path(weights_path).expanduser()
    if not weights_file.exists():
        raise FileNotFoundError(f"DINO weights file not found: {weights_file}")
    config.DINO_WEIGHTS_PATH = str(weights_file.resolve())

    device_hint = (config.DINO_DEVICE or "").strip()
    if not device_hint:
        if torch.cuda.is_available():
            device_hint = "cuda:0"
        else:
            raise RuntimeError("CUDA device required for DINO encoder; none detected.")
    if not device_hint.startswith("cuda"):
        raise RuntimeError("DINO encoder must run on CUDA. Set EVOSSEARCH_DINO_DEVICE accordingly.")
    config.DINO_DEVICE = device_hint

    dino_encoder = DINOEncoder(
        model_name=config.DINO_MODEL,
        batch_size=config.BATCH_SIZE,
        weights_path=config.DINO_WEIGHTS_PATH,
        device=config.DINO_DEVICE,
    )


def ensure_embedder_loaded(embedder: Optional[str] = None) -> None:
    target = embedder or active_embedder
    if target == "clip":
        init_clip()
    elif target == "dino":
        init_dino()
    elif target == "fusion":
        init_clip()
        init_dino()
    else:
        raise ValueError(f"Unsupported embedder: {target}")


def ensure_mask_head() -> Optional["Mask2FormerHead"]:
    global mask2former_head, _mask2former_failed
    if not config.MASK2FORMER_ENABLED:
        return None
    if Mask2FormerHead is None:
        if not _mask2former_failed:
            print("Mask2Former head unavailable: transformers vision deps missing; disabling head.")
            _mask2former_failed = True
        return None
    if mask2former_head is not None:
        return mask2former_head
    with _mask2former_lock:
        if mask2former_head is not None:
            return mask2former_head
        try:
            target_device = config.MASK2FORMER_DEVICE or ("cuda:0" if torch.cuda.is_available() else "cpu")
            mask2former_head = Mask2FormerHead(
                model_name=config.MASK2FORMER_MODEL,
                device=target_device,
                max_size=config.MASK2FORMER_MAX_SIZE,
            )
        except Exception as exc:
            _mask2former_failed = True
            print(f"Mask2Former head initialization failed: {exc}")
            config.MASK2FORMER_ENABLED = False
            return None
    return mask2former_head


def get_image_embedding(image_path: Union[str, Path], embedder: Optional[str] = None) -> np.ndarray:
    """Extract an embedding for an image path using the selected backend."""
    target = embedder or active_embedder
    if target == "fusion":
        target = "clip"
    ensure_embedder_loaded(target)
    if target == "clip":
        image = clip_preprocess(Image.open(image_path)).unsqueeze(0).to(device)  # type: ignore[attr-defined]
        with torch.no_grad():
            image_features = clip_model.encode_image(image)  # type: ignore[union-attr]
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()
    else:
        assert dino_encoder is not None
        features = dino_encoder.encode_images(image_path)
        return features[0]


def get_image_embedding_from_pil(pil_image: Image.Image, embedder: Optional[str] = None) -> np.ndarray:
    """Extract an embedding from a PIL image using the selected backend."""
    target = embedder or active_embedder
    if target == "fusion":
        target = "clip"
    ensure_embedder_loaded(target)
    if target == "clip":
        image = clip_preprocess(pil_image).unsqueeze(0).to(device)  # type: ignore[attr-defined]
        with torch.no_grad():
            image_features = clip_model.encode_image(image)  # type: ignore[union-attr]
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()
    else:
        assert dino_encoder is not None
        features = dino_encoder.encode_images(pil_image)
        return features[0]


def get_text_embedding(text: str) -> np.ndarray:
    """Extract a CLIP text embedding. Only available when CLIP or fusion backend is active."""
    if active_embedder not in {"clip", "fusion"}:
        raise RuntimeError("Text search is only supported when the CLIP backend is active.")
    ensure_embedder_loaded("clip")
    text_tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)  # type: ignore[union-attr]
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy().flatten()


def _build_index_metadata(embedder: str, additional: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ensure_embedder_loaded(embedder)
    base: Dict[str, Any]
    if embedder == "clip":
        embed_dim = int(getattr(getattr(clip_model, "visual", None), "output_dim", 512))  # type: ignore[union-attr]
        base = {
            "embedder": "clip",
            "model": config.CLIP_MODEL,
            "embedding_dim": embed_dim,
            "library": "openai/CLIP",
            "device": device,
        }
    else:
        assert dino_encoder is not None
        base = {
            "embedder": "dino",
            "model": dino_encoder.metadata.name,
            "embedding_dim": dino_encoder.metadata.embed_dim,
            "config_model": config.DINO_MODEL,
            "library": "facebookresearch/dinov3",
            "device": str(dino_encoder.device),
        }
    if additional:
        base.update(additional)
    base.setdefault("created_at", time.time())
    return base


def _index_targets() -> List[str]:
    if config.INDEX_MODE == 'dual':
        return list(EMBEDDER_SUBDIRS.keys())
    return [config.INDEX_MODE]


def _collect_image_entries(folder_path: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    entries: List[Tuple[Path, Dict[str, Any]]] = []
    for ext in config.SUPPORTED_EXTENSIONS:
        for img_path in sorted(folder_path.glob(f'*{ext}')):
            try:
                stat = img_path.stat()
            except FileNotFoundError:
                continue
            metadata = {
                'path': str(img_path),
                'mtime': stat.st_mtime,
                'size': stat.st_size,
            }
            entries.append((img_path, metadata))
    return entries


def _create_index_for_embedder(entries: List[Tuple[Path, Dict[str, Any]]], embedder: str) -> Optional[Tuple[faiss.Index, List[str], List[Dict[str, Any]], Dict[str, Any]]]:
    image_paths: List[str] = []
    image_metadata: List[Dict[str, Any]] = []
    embeddings: List[np.ndarray] = []

    for img_path, metadata in entries:
        try:
            embedding = get_image_embedding(img_path, embedder=embedder)
        except Exception as exc:
            print(f"Error processing {img_path} for {embedder}: {exc}")
            continue
        embeddings.append(embedding)
        image_paths.append(str(img_path))
        image_metadata.append(metadata)

    if not embeddings:
        return None

    embeddings_array = np.array(embeddings, dtype='float32')
    index = faiss.IndexFlatIP(embeddings_array.shape[1])
    index.add(embeddings_array)
    index_meta = _build_index_metadata(embedder, {'image_count': len(image_paths)})
    return index, image_paths, image_metadata, index_meta


def create_index(folder_path):
    """Create FAISS indexes for the configured embedder targets."""
    folder_path = Path(folder_path)
    entries = _collect_image_entries(folder_path)
    if not entries:
        return {}

    results: Dict[str, Tuple[faiss.Index, List[str], List[Dict[str, Any]], Dict[str, Any]]] = {}
    for embedder in _index_targets():
        if embedder not in EMBEDDER_SUBDIRS:
            continue
        data = _create_index_for_embedder(entries, embedder)
        if data is not None:
            results[embedder] = data
    return results


@app.route('/')
def home():
    """Serve the frontend"""
    # Generate result limit options dynamically based on config
    result_options = []
    
    # Create a reasonable set of options between min and max
    min_val = config.MIN_RESULTS
    max_val = config.MAX_RESULTS
    default_val = config.DEFAULT_RESULTS
    
    # Generate options with reasonable intervals
    options = set()
    
    # Always include min, default, and max
    options.add(min_val)
    options.add(default_val)
    options.add(max_val)
    
    # Add some intermediate values
    if max_val <= 20:
        # Small range: add every 2-3 values
        for i in range(min_val, max_val + 1):
            if i % 2 == 0 or i % 3 == 0:
                options.add(i)
    else:
        # Larger range: add multiples of 6, 12, etc.
        for i in [6, 12, 18, 24, 30]:
            if min_val <= i <= max_val:
                options.add(i)
    
    # Sort and create HTML options
    for i in sorted(options):
        selected = "selected" if i == default_val else ""
        result_options.append(f'<option value="{i}" {selected}>{i}</option>')
    
    result_options_html = '\n                            '.join(result_options)
    
    # Use string formatting for the result options
    html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Natural Language Image Search</title>
    <!-- Cache buster: {timestamp} -->
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a;
            color: #e0e0e0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        .container {
            max-width: 1200px;
            min-width: 900px;                     
            margin: 0 auto;
            padding: 2rem;
            flex: 1;
        }
        
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
        }
        
        h1 {
            font-size: 2rem;
            font-weight: 300;
            letter-spacing: -0.02em;
            margin: 0;
        }
        
        .settings-icon {
            cursor: pointer;
            padding: 8px;
            border-radius: 6px;
            transition: all 0.2s ease;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .settings-icon:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.2);
            transform: scale(1.05);
        }
        
        /* Settings Modal */
        .settings-modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            z-index: 1000;
            backdrop-filter: blur(4px);
        }
        
        .settings-modal-content {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #161616;
            border-radius: 12px;
            border: 1px solid #262626;
            width: 90%;
            max-width: 600px;
            max-height: 80vh;
            overflow-y: auto;
            padding: 2rem;
        }
        
        .settings-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid #333;
        }
        
        .settings-header h2 {
            font-size: 1.5rem;
            font-weight: 300;
            color: #e0e0e0;
            margin: 0;
        }
        
        .close-btn {
            background: none;
            border: none;
            color: #888;
            font-size: 1.5rem;
            cursor: pointer;
            padding: 4px;
            border-radius: 4px;
            transition: all 0.2s ease;
        }
        
        .close-btn:hover {
            color: #e0e0e0;
            background: rgba(255, 255, 255, 0.1);
        }
        
        .settings-section {
            margin-bottom: 2rem;
        }
        
        .settings-section h3 {
            font-size: 1.1rem;
            font-weight: 400;
            color: #e0e0e0;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #333;
        }
        
        .settings-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            gap: 1rem;
        }

        .backend-dino {
            display: none;
        }

        .mode-tab[disabled] {
            opacity: 0.4;
            cursor: not-allowed;
        }

        .settings-label {
            flex: 1;
            color: #ccc;
            font-size: 0.9rem;
            min-width: 120px;
        }
        
        .settings-input {
            flex: 1;
            background: #0a0a0a;
            border: 1px solid #333;
            padding: 0.5rem 0.75rem;
            border-radius: 6px;
            color: #e0e0e0;
            font-size: 0.9rem;
            transition: border-color 0.2s;
            max-width: 200px;
        }
        
        .settings-input:focus {
            outline: none;
            border-color: #555;
        }

        .settings-input.disabled {
            opacity: 0.5;
        }
        
        .settings-checkbox {
            width: 18px;
            height: 18px;
            accent-color: #4a4a4a;
        }
        
        .settings-select {
            background: #0a0a0a;
            border: 1px solid #333;
            padding: 0.5rem 0.75rem;
            border-radius: 6px;
            color: #e0e0e0;
            font-size: 0.9rem;
            cursor: pointer;
            max-width: 200px;
        }
        
        .settings-range {
            flex: 1;
            max-width: 150px;
            accent-color: #555;
        }
        
        .range-value {
            min-width: 30px;
            text-align: center;
            color: #888;
            font-size: 0.85rem;
        }

        .range-value.disabled {
            opacity: 0.5;
        }
        
        .settings-actions {
            display: flex;
            gap: 1rem;
            justify-content: flex-end;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #333;
        }
        
        .settings-btn {
            background: #2a2a2a;
            border: 1px solid #444;
            color: #e0e0e0;
            padding: 0.75rem 1.5rem;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.2s;
        }
        
        .settings-btn:hover {
            background: #333;
            border-color: #555;
        }
        
        .settings-btn.primary {
            background: #4a4a4a;
            border-color: #666;
        }
        
        .settings-btn.primary:hover {
            background: #555;
            border-color: #777;
        }
        
        .settings-status {
            margin-top: 1rem;
            padding: 0.75rem;
            border-radius: 6px;
            font-size: 0.9rem;
            display: none;
        }
        
        .settings-status.success {
            background: rgba(74, 222, 128, 0.1);
            border: 1px solid rgba(74, 222, 128, 0.3);
            color: #4ade80;
        }
        
        .settings-status.error {
            background: rgba(248, 113, 113, 0.1);
            border: 1px solid rgba(248, 113, 113, 0.3);
            color: #f87171;
        }
        
        .control-panel {
            background: #161616;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            border: 1px solid #262626;
        }
        
        .folder-select {
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        
        input[type="text"] {
            flex: 1;
            background: #0a0a0a;
            border: 1px solid #333;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            color: #e0e0e0;
            font-size: 0.95rem;
            transition: border-color 0.2s;
        }
        
        input[type="text"]:focus {
            outline: none;
            border-color: #555;
        }
        
        button {
            background: #1a1a1a;
            border: 1px solid #333;
            color: #e0e0e0;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.95rem;
            transition: all 0.2s;
        }
        
        button:hover {
            background: #222;
            border-color: #444;
        }
        
        button:active {
            transform: translateY(1px);
        }
        
        .status {
            font-size: 0.875rem;
            color: #888;
            margin-top: 0.5rem;
        }
        
        .status.success {
            color: #4ade80;
        }
        
        .status.error {
            color: #f87171;
        }
        
        .search-panel {
            background: #161616;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            border: 1px solid #262626;
        }
        
        .search-mode-tabs {
            display: flex;
            gap: 0;
            margin-bottom: 1rem;
            border-radius: 8px;
            overflow: hidden;
        }
        
        .mode-tab {
            flex: 1;
            background: #0a0a0a;
            border: 1px solid #333;
            color: #888;
            padding: 0.75rem 1rem;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.2s;
            border-radius: 0;
        }
        
        .mode-tab.active {
            background: #1a1a1a;
            color: #e0e0e0;
            border-color: #555;
        }
        
        .mode-tab:hover {
            background: #222;
            color: #e0e0e0;
        }
        
        .search-controls {
            margin-bottom: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
        }
        
        .control-group {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .feature-btn {
            background: #2a4a3a;
            border: 1px solid #3a5a4a;
            color: #e0e0e0;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.2s;
            white-space: nowrap;
        }
        
        .feature-btn:hover {
            background: #345a44;
            border-color: #4a6a54;
        }
        
        .feature-btn:active {
            transform: translateY(1px);
        }
        
        .sort-control {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .limit-control {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .segment-controls {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .segment-controls label {
            font-size: 0.85rem;
            color: #bbb;
        }
        
        .segment-threshold-control {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .segment-threshold-control input[type="range"] {
            width: 160px;
            accent-color: #ff9d1a;
        }
        
        .segment-threshold-value {
            font-variant-numeric: tabular-nums;
            min-width: 3.5ch;
            text-align: right;
            color: #e0e0e0;
        }
        
        .segment-threshold-control.disabled {
            opacity: 0.4;
            pointer-events: none;
        }
        
        .sort-control label,
        .limit-control label {
            color: #888;
            font-size: 0.9rem;
        }
        
        select {
            background: #0a0a0a;
            border: 1px solid #333;
            color: #e0e0e0;
            padding: 0.5rem 0.75rem;
            border-radius: 6px;
            font-size: 0.9rem;
            cursor: pointer;
            transition: border-color 0.2s;
        }
        
        select:focus {
            outline: none;
            border-color: #555;
        }
        
        select option {
            background: #0a0a0a;
            color: #e0e0e0;
        }
        
        .search-box {
            display: flex;
            gap: 1rem;
        }
        
        input[type="file"] {
            background: #0a0a0a;
            border: 1px solid #333;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            color: #e0e0e0;
            font-size: 0.95rem;
            transition: border-color 0.2s;
            width: 100%;
        }
        
        input[type="file"]:focus {
            outline: none;
            border-color: #555;
        }
        
        /* Image search layout */
        .image-search-inputs {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            flex: 1;
            margin-right: 1rem;
        }
        
        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .input-label {
            color: #888;
            font-size: 0.9rem;
            font-weight: 500;
        }
        
        .input-separator {
            text-align: center;
            color: #666;
            font-size: 0.8rem;
            font-weight: bold;
            padding: 0.25rem 0;
        }
        
        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 1.5rem;
        }
        
        .result-item {
            background: #161616;
            border-radius: 8px;
            overflow: hidden;
            cursor: pointer;
            transition: all 0.3s;
            border: 1px solid #262626;
        }
        
        .result-item:hover {
            transform: translateY(-2px);
            border-color: #444;
        }
        
        .result-item.expanded {
            grid-column: 1 / -1;
        }
        
        .thumbnail {
            width: 100%;
            height: 150px;
            object-fit: cover;
            display: block;
        }
        
        .thumbnail.segment-enabled {
            cursor: crosshair;
        }
        
        
        .result-info {
            padding: 0.75rem;
            font-size: 0.875rem;
        }
        
        .filename {
            color: #e0e0e0;
            margin-bottom: 0.25rem;
            word-break: break-all;
        }
        
        .similarity {
            color: #bbb;
            font-size: 0.85rem;
            display: flex;
            flex-direction: column;
            gap: 0.2rem;
        }

        .similarity .metric-line {
            display: flex;
            gap: 0.3rem;
            align-items: baseline;
            color: #999;
        }

        .metric-label {
            color: #e0e0e0;
            font-weight: 500;
        }

        .metric-note {
            color: #777;
            font-size: 0.75rem;
            margin-left: 0.35rem;
        }

        .settings-input.disabled,
        .settings-input:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .loading {
            text-align: center;
            padding: 2rem;
            color: #666;
        }
        
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid #333;
            border-top-color: #888;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        /* Comment System Styles */
        .comment-section {
            display: none;
            padding: 1rem;
            border-top: 1px solid #333;
            background: #0f0f0f;
        }
        
        .segments-panel {
            display: none;
            padding: 1rem;
            border-top: 1px solid #2a2a2a;
            background: #111;
        }
        
        .result-item.expanded .segments-panel {
            display: block;
        }
        
        .segments-status {
            font-size: 0.85rem;
            margin-bottom: 0.75rem;
            color: #bcbcbc;
        }
        
        .segments-status.success {
            color: #7dd97b;
        }
        
        .segments-status.error {
            color: #ff6b6b;
        }
        
        .segments-status.warning {
            color: #f4c066;
        }
        
        .segment-overlay-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            align-items: flex-start;
            margin-bottom: 0.75rem;
        }
        
        .segment-overlay-figure,
        .segment-segmap-figure {
            margin: 0;
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
            max-width: 280px;
        }
        
        .segment-overlay-figure figcaption,
        .segment-segmap-figure figcaption {
            color: #aaa;
            font-size: 0.78rem;
            letter-spacing: 0.01em;
        }
        
        .segment-overlay-stack {
            position: relative;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid #1f1f1f;
        }
        
        .segment-overlay-stack img {
            display: block;
            width: 100%;
            height: auto;
        }
        
        .segment-overlay-stack .overlay-layer {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }
        
        .segment-overlay-stack .overlay-heatmap {
            mix-blend-mode: screen;
            opacity: 0.85;
        }
        
        .segment-overlay-stack .overlay-mask {
            mix-blend-mode: multiply;
            opacity: 0.45;
        }
        
        .segment-overlay-stack .overlay-crosshair {
            position: absolute;
            width: 14px;
            height: 14px;
            margin-left: -7px;
            margin-top: -7px;
            border-radius: 50%;
            border: 2px solid #ffdf6b;
            box-shadow: 0 0 6px rgba(255, 223, 107, 0.7);
            pointer-events: none;
        }

        .segment-segmap {
            width: 100%;
            border-radius: 10px;
            border: 1px solid #1f1f1f;
            display: block;
        }
        
        .segment-legend {
            display: flex;
            flex-direction: column;
            gap: 0.3rem;
            margin-bottom: 0.75rem;
        }
        
        .segment-legend-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.78rem;
            color: #cfcfcf;
        }
        
        .segment-legend-item.highlight {
            color: #ffffff;
            font-weight: 600;
        }
        
        .segment-legend-swatch {
            width: 16px;
            height: 16px;
            border-radius: 3px;
            border: 1px solid rgba(0, 0, 0, 0.6);
            box-shadow: 0 0 4px rgba(0, 0, 0, 0.3);
        }
        
        .segment-legend-item.highlight .segment-legend-swatch {
            box-shadow: 0 0 6px rgba(255, 255, 255, 0.6);
        }
        
        .segment-overlay-images img {
            max-width: 160px;
            border: 1px solid #222;
            border-radius: 4px;
            background: #000;
        }
        
        .segment-results-list {
            list-style: none;
            padding: 0;
            margin: 0;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .segment-results-list li {
            background: #181818;
            border: 1px solid #242424;
            border-radius: 6px;
            padding: 0.6rem 0.75rem;
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
            font-size: 0.82rem;
        }
        
        .segment-title {
            color: #e0e0e0;
            font-weight: 600;
            letter-spacing: 0.01em;
        }
        
        .segment-meta {
            color: #888;
            font-size: 0.78rem;
        }
        
        .segment-match-list {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }
        
        .segment-match-row {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            background: #111;
            border: 1px solid #222;
            border-radius: 5px;
            padding: 0.35rem 0.5rem;
        }
        
        .segment-match-thumb {
            width: 44px;
            height: 44px;
            object-fit: cover;
            border-radius: 4px;
            border: 1px solid #1f1f1f;
        }
        
        .segment-match-thumb.placeholder {
            background: repeating-linear-gradient(45deg, #222, #222 6px, #1a1a1a 6px, #1a1a1a 12px);
        }
        
        .segment-match-meta {
            display: flex;
            flex-direction: column;
            font-size: 0.75rem;
            color: #cfcfcf;
            line-height: 1.35;
        }
        
        .result-item.expanded .comment-section {
            display: block;
        }
        
        .comments-list {
            max-height: 200px;
            overflow-y: auto;
            margin-bottom: 1rem;
            padding: 0.5rem;
            background: #1a1a1a;
            border-radius: 6px;
            border: 1px solid #333;
        }
        
        .comment-item {
            padding: 0.5rem;
            margin-bottom: 0.5rem;
            background: #222;
            border-radius: 4px;
            border-left: 3px solid #555;
            font-size: 0.85rem;
            line-height: 1.4;
            color: #e0e0e0;
        }
        
        .comment-item:last-child {
            margin-bottom: 0;
        }
        
        .comment-timestamp {
            color: #888;
            font-size: 0.75rem;
            font-weight: bold;
        }
        
        .comment-text {
            margin-top: 0.25rem;
            color: #ccc;
        }
        
        .comment-form {
            display: flex;
            gap: 0.5rem;
            align-items: flex-start;
        }
        
        .comment-input {
            flex: 1;
            background: #0a0a0a;
            border: 1px solid #333;
            padding: 0.5rem 0.75rem;
            border-radius: 6px;
            color: #e0e0e0;
            font-size: 0.85rem;
            resize: vertical;
            min-height: 60px;
            font-family: inherit;
        }
        
        .comment-input:focus {
            outline: none;
            border-color: #555;
        }
        
        .comment-input::placeholder {
            color: #666;
        }
        
        .save-comment-btn {
            background: #2a2a2a;
            border: 1px solid #444;
            color: #e0e0e0;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85rem;
            transition: all 0.2s;
            white-space: nowrap;
        }
        
        .save-comment-btn:hover {
            background: #333;
            border-color: #555;
        }
        
        .save-comment-btn:disabled {
            background: #1a1a1a;
            border-color: #333;
            color: #666;
            cursor: not-allowed;
        }
        
        .no-comments {
            text-align: center;
            color: #666;
            font-style: italic;
            padding: 1rem;
            font-size: 0.85rem;
        }
        
        .comment-loading {
            text-align: center;
            color: #888;
            font-size: 0.85rem;
            padding: 0.5rem;
        }
        
        
        /* Image Container and Overlay */
        .image-container {
            position: relative;
            display: block;
        }
        
        .image-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none; /* Allow clicks to pass through to image */
        }
        
        .expand-collapse-icon {
            position: absolute;
            bottom: 8px;
            right: 8px;
            background: rgba(0, 0, 0, 0.7);
            border-radius: 4px;
            padding: 4px;
            cursor: pointer;
            pointer-events: auto; /* Re-enable clicks for the icon */
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .expand-collapse-icon:hover {
            background: rgba(0, 0, 0, 0.9);
            transform: scale(1.1);
        }

        .find-similar-icon,
        .describe-icon {
            position: absolute;
            bottom: 8px;
            left: 8px;
            background: rgba(0, 0, 0, 0.7);
            border-radius: 4px;
            padding: 4px;
            cursor: pointer;
            pointer-events: auto;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .find-similar-icon {
            left: 40px;
        }

        .find-similar-icon:hover,
        .describe-icon:hover {
            background: rgba(0, 0, 0, 0.9);
            transform: scale(1.1);
        }

        /* Video understanding */
        .video-box {
            display: none;
            flex-direction: column;
            gap: 0.8rem;
            background: #111;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 1rem;
        }

        .video-row {
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
        }

        .video-row .input-group {
            flex: 1;
            min-width: 220px;
        }

        .video-prompt {
            width: 100%;
            min-height: 110px;
            background: #0f0f0f;
            border: 1px solid #2a2a2a;
            border-radius: 6px;
            color: #eaeaea;
            padding: 0.75rem;
            resize: vertical;
        }

        .video-controls {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            flex-wrap: wrap;
        }

        .video-status {
            font-size: 0.9rem;
            color: #ccc;
            min-height: 20px;
        }

        .video-output {
            background: #0f0f0f;
            border: 1px solid #222;
            border-radius: 8px;
            padding: 0.9rem;
            color: #eaeaea;
            line-height: 1.6;
            white-space: pre-wrap;
        }

        .video-frame-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .video-frame-grid img {
            width: 140px;
            height: auto;
            border-radius: 5px;
            border: 1px solid #222;
            background: #111;
        }
        
        
        .find-similar-icon {
            position: absolute;
            top: 8px;
            right: 8px;
            background: rgba(0, 0, 0, 0.7);
            border-radius: 4px;
            padding: 4px;
            cursor: pointer;
            pointer-events: auto; /* Re-enable clicks for the icon */
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .find-similar-icon:hover {
            background: rgba(0, 0, 0, 0.9);
            transform: scale(1.1);
        }
        
        /* Show find similar icon only when expanded */
        .result-item.expanded .find-similar-icon {
            display: flex !important;
        }
        
        /* Copy icon styling */
        .copy-icon {
            margin-left: 8px;
            cursor: pointer;
            transition: fill 0.2s ease;
            vertical-align: middle;
        }
        
        .copy-icon:hover {
            fill: #e0e0e0;
        }
        
        .filename {
            display: flex;
            align-items: center;
        }
        
        /* Expanded image display */
        .result-item.expanded .thumbnail {
            width: 100%;
            min-width: 900px;
            height: auto;
            object-fit: contain;
        }
        
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Natural Language Image Search</h1>
            <div class="settings-icon" id="settingsBtn">
                <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="#e3e3e3">
                    <path d="m370-80-16-128q-13-5-24.5-12T307-235l-119 50L78-375l103-78q-1-7-1-13.5v-27q0-6.5 1-13.5L78-585l110-190 119 50q11-8 23-15t24-12l16-128h220l16 128q13 5 24.5 12t22.5 15l119-50 110 190-103 78q1 7 1 13.5v27q0 6.5-1 13.5l103 78-110 190-119-50q-11 8-23 15t-24 12L590-80H370Zm70-80h79l14-106q31-8 57.5-23.5T639-327l99 41 39-68-86-65q5-14 7-29.5t2-31.5q0-16-2-31.5t-7-29.5l86-65-39-68-99 41q-22-23-48.5-38.5T533-694l-13-106h-79l-14 106q-31 8-57.5 23.5T321-633l-99-41-39 68 86 65q-5 14-7 29.5t-2 31.5q0 16 2 31.5t7 29.5l-86 65 39 68 99-41q22 23 48.5 38.5T427-266l13 106Zm42-180q58 0 99-41t41-99q0-58-41-99t-99-41q-59 0-99.5 41T342-480q0 58 40.5 99t99.5 41Zm-2-140Z"/>
                </svg>
            </div>
        </div>
        
        <div class="control-panel">
            <div class="folder-select">
                <input type="text" id="folderPath" placeholder="Enter folder path..." />
                <button id="indexBtn">Index Folder</button>
            </div>
            <div class="status" id="indexStatus"></div>
        </div>
        
            <div class="search-panel">
                <div class="search-mode-tabs">
                    <button id="textModeBtn" class="mode-tab active">Text Search</button>
                    <button id="imageModeBtn" class="mode-tab">Image Search</button>
                    <button id="videoModeBtn" class="mode-tab">Video Understanding</button>
                </div>
            <div class="search-controls">
                <div class="control-group">
                    <button id="showCommentedBtn" class="feature-btn">Show Commented Images</button>
                </div>
                <div class="control-group">
                    <div class="sort-control">
                        <label for="sortBy">Sort by:</label>
                        <select id="sortBy">
                            <option value="similarity" selected>Similarity</option>
                            <option value="time">Time (Newest First)</option>
                        </select>
                    </div>
                    <div class="limit-control">
                        <label for="resultLimit">Results:</label>
                        <select id="resultLimit">
                            {result_options_html}
                        </select>
                    </div>
                    <div class="segment-controls">
                        <label for="segmentThresholdSlider">Region threshold:</label>
                        <div class="segment-threshold-control" id="segmentThresholdControl">
                            <input type="range" id="segmentThresholdSlider" min="40" max="99" value="70" step="1">
                            <span class="segment-threshold-value" id="segmentThresholdValue">70%</span>
                        </div>
                    </div>
                </div>
            </div>
            <div id="textSearchBox" class="search-box">
                <input type="text" id="searchQuery" placeholder="Describe what you're looking for..." />
                <button id="searchBtn">Search</button>
            </div>
            <div id="imageSearchBox" class="search-box" style="display: none;">
                <div class="image-search-inputs">
                    <div class="input-group">
                        <label for="imageUpload" class="input-label">Upload File:</label>
                        <input type="file" id="imageUpload" accept="image/*" />
                    </div>
                    <div class="input-separator">OR</div>
                    <div class="input-group">
                        <label for="imagePath" class="input-label">Enter Image Path:</label>
                        <input type="text" id="imagePath" placeholder="C:\\path\\to\\image.jpg" />
                    </div>
                </div>
                <button id="imageSearchBtn">Search by Image</button>
            </div>
            <div id="videoBox" class="video-box" style="display: none;">
                <div class="video-row">
                    <div class="input-group">
                        <label for="videoPath" class="input-label">Video Path:</label>
                        <input type="text" id="videoPath" placeholder="/home/user/video.mp4" />
                    </div>
                </div>
                <div class="video-row">
                    <div class="input-group">
                        <label class="input-label" for="videoFrameCount">Frames to sample:</label>
                        <select id="videoFrameCount">
                            <option value="16">16</option>
                            <option value="32">32</option>
                            <option value="64">64</option>
                        </select>
                    </div>
                    <div class="input-group">
                        <label class="input-label" for="videoSampleFps">Target sample FPS (optional):</label>
                        <input type="number" id="videoSampleFps" min="0" step="0.1" placeholder="auto" />
                    </div>
                </div>
                <div class="input-group">
                    <label class="input-label" for="videoPrompt">Prompt:</label>
                    <textarea id="videoPrompt" class="video-prompt" placeholder="Describe the actions, key events, and any objects of interest."></textarea>
                    <label style="color: #aaa; font-size: 0.85rem;">
                        <input type="checkbox" id="saveVideoPrompt"> Remember this prompt
                    </label>
                </div>
                <div class="video-controls">
                    <button id="videoRunBtn" class="feature-btn primary">Analyze Video</button>
                    <button id="saveSummaryBtn" class="feature-btn" style="display:none;">Save summary as comment</button>
                    <div id="videoStatus" class="video-status"></div>
                </div>
                <div id="videoOutput" class="video-output" style="display: none;"></div>
                <div id="videoFrames" class="video-frame-grid"></div>
            </div>
        </div>
        
        <div id="results" class="results-grid"></div>
    </div>
    
    <!-- Settings Modal -->
    <div id="settingsModal" class="settings-modal">
        <div class="settings-modal-content">
            <div class="settings-header">
                <h2>Settings</h2>
                <button class="close-btn" id="closeSettings">&times;</button>
            </div>
            
            <div class="settings-section">
                <h3>Server Configuration</h3>
                <div class="settings-row">
                    <label class="settings-label">Host:</label>
                    <input type="text" id="host" class="settings-input" placeholder="0.0.0.0">
                </div>
                <div class="settings-row">
                    <label class="settings-label">Port:</label>
                    <input type="number" id="port" class="settings-input" min="1000" max="65535" placeholder="5000">
                </div>
                <div class="settings-row">
                    <label class="settings-label">Debug Mode:</label>
                    <input type="checkbox" id="debug" class="settings-checkbox">
                </div>
            </div>
            
            <div class="settings-section">
                <h3>Search Configuration</h3>
                <div class="settings-row">
                    <label class="settings-label">Min Results:</label>
                    <input type="number" id="minResults" class="settings-input" min="1" max="100" placeholder="3">
                </div>
                <div class="settings-row">
                    <label class="settings-label">Max Results:</label>
                    <input type="number" id="maxResults" class="settings-input" min="1" max="200" placeholder="48">
                </div>
                <div class="settings-row">
                    <label class="settings-label">Default Results:</label>
                    <input type="number" id="defaultResults" class="settings-input" min="1" max="100" placeholder="12">
                </div>
            </div>
            
            <div class="settings-section">
                <h3>Model & Processing</h3>
                <div class="settings-row">
                    <label class="settings-label">Backend:</label>
                    <select id="embedder" class="settings-select">
                        <option value="clip">CLIP</option>
                        <option value="dino">DINO</option>
                        <option value="fusion">Fusion</option>
                    </select>
                </div>
                <div class="settings-row">
                    <label class="settings-label">Fusion Enabled:</label>
                    <input type="checkbox" id="fusionEnabled" class="settings-checkbox">
                </div>
                <div class="settings-row">
                    <label class="settings-label">Fusion Alpha:</label>
                    <input type="range" id="fusionAlpha" class="settings-range" min="0" max="1" step="0.05" value="0.7">
                    <span class="range-value" id="fusionAlphaValue">0.70</span>
                </div>
                <div class="settings-row backend-dino">
                    <label class="settings-label">DINO Model:</label>
                    <input type="text" id="dinoModel" class="settings-input" placeholder="dinov3_vitb16">
                </div>
                <div class="settings-row backend-dino">
                    <label class="settings-label">DINO Embedding Dim:</label>
                    <input type="number" id="dinoEmbedDim" class="settings-input" min="128" max="4096" placeholder="1280">
                </div>
                <div class="settings-row backend-dino">
                    <label class="settings-label">DINO Weights Path:</label>
                    <input type="text" id="dinoWeightsPath" class="settings-input" placeholder="/path/to/dinov3">
                </div>
                <div class="settings-row backend-clip">
                    <label class="settings-label">CLIP Model:</label>
                    <select id="clipModel" class="settings-select">
                        <option value="ViT-B/32">ViT-B/32</option>
                        <option value="ViT-B/16">ViT-B/16</option>
                        <option value="ViT-L/14">ViT-L/14</option>
                    </select>
                </div>
                <div class="settings-row">
                    <label class="settings-label">Batch Size:</label>
                    <input type="number" id="batchSize" class="settings-input" min="1" max="128" placeholder="32">
                </div>
                <div class="settings-row">
                    <label class="settings-label">Thumbnail Quality:</label>
                    <input type="range" id="thumbnailQuality" class="settings-range" min="50" max="100" value="85">
                    <span class="range-value" id="qualityValue">85</span>
                </div>
            </div>
            
            <div class="settings-section">
                <h3>Advanced Settings</h3>
                <div class="settings-row">
                    <label class="settings-label">Max Comment Length:</label>
                    <input type="number" id="maxCommentLength" class="settings-input" min="50" max="2000" placeholder="100">
                </div>
                <div class="settings-row">
                    <label class="settings-label">Max File Size (MB):</label>
                    <input type="number" id="maxFileSize" class="settings-input" min="1" max="500" placeholder="50">
                </div>
                <div class="settings-row">
                    <label class="settings-label">Index Folder Name:</label>
                    <input type="text" id="indexFolderName" class="settings-input" placeholder=".clip_index">
                </div>
                <div class="settings-row">
                    <label class="settings-label">Index Mode:</label>
                    <select id="indexMode" class="settings-select">
                        <option value="clip">CLIP only</option>
                        <option value="dino">DINO only</option>
                        <option value="dual">Dual (CLIP & DINO)</option>
                    </select>
                </div>
                <div class="settings-row">
                    <label class="settings-label">Rerank Enabled:</label>
                    <input type="checkbox" id="rerankEnabled" class="settings-checkbox">
                </div>
                <div class="settings-row">
                    <label class="settings-label">Rerank Top-K:</label>
                    <input type="number" id="rerankTopK" class="settings-input" min="1" max="500" placeholder="50">
                </div>
                <div class="settings-row">
                    <label class="settings-label">Segment Embeddings:</label>
                    <input type="checkbox" id="segmentsEnabled" class="settings-checkbox">
                </div>
                <div class="settings-row">
                    <label class="settings-label">Min Segment Patches:</label>
                    <input type="number" id="segmentMinPatches" class="settings-input" min="1" max="256" placeholder="3">
                </div>
            </div>
            
            <div class="settings-actions">
                <button class="settings-btn" id="resetSettings">Reset to Defaults</button>
                <button class="settings-btn primary" id="saveSettings">Save Settings</button>
            </div>
            
            <div id="settingsStatus" class="settings-status"></div>
        </div>
    </div>
    
    <script>
        const folderInput = document.getElementById('folderPath');
        const indexBtn = document.getElementById('indexBtn');
        const indexStatus = document.getElementById('indexStatus');
        const searchInput = document.getElementById('searchQuery');
        const searchBtn = document.getElementById('searchBtn');
        const imageUpload = document.getElementById('imageUpload');
        const imagePath = document.getElementById('imagePath');
        const imageSearchBtn = document.getElementById('imageSearchBtn');
        const textModeBtn = document.getElementById('textModeBtn');
        const imageModeBtn = document.getElementById('imageModeBtn');
        const videoModeBtn = document.getElementById('videoModeBtn');
        const textSearchBox = document.getElementById('textSearchBox');
        const imageSearchBox = document.getElementById('imageSearchBox');
        const videoBox = document.getElementById('videoBox');
        const videoPathInput = document.getElementById('videoPath');
        const videoFrameCount = document.getElementById('videoFrameCount');
        const videoSampleFpsInput = document.getElementById('videoSampleFps');
        const videoPromptInput = document.getElementById('videoPrompt');
        const saveVideoPromptInput = document.getElementById('saveVideoPrompt');
        const videoRunBtn = document.getElementById('videoRunBtn');
        const videoStatus = document.getElementById('videoStatus');
        const videoOutput = document.getElementById('videoOutput');
        const videoFrames = document.getElementById('videoFrames');
        const saveSummaryBtn = document.getElementById('saveSummaryBtn');
        const resultLimitSelect = document.getElementById('resultLimit');
        const sortBySelect = document.getElementById('sortBy');
        const showCommentedBtn = document.getElementById('showCommentedBtn');
        const resultsContainer = document.getElementById('results');
        
        let currentFolder = '';
        let currentMode = 'text';
        let videoTimerHandle = null;
        let videoRequestStarted = 0;
        let lastSummaryText = '';
        let lastSummaryTarget = null;

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function renderMarkdown(text) {
            const safe = escapeHtml(text || '');
            return safe
                .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
                .replace(/`([^`]+)`/g, '<code>$1</code>')
                .replace(/\\n/g, '<br>');
        }

        function formatDuration(seconds) {
            if (!Number.isFinite(seconds)) return 'n/a';
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${mins}m ${secs}s`;
        }

        function startVideoTimer() {
            videoRequestStarted = performance.now();
            if (videoTimerHandle) clearInterval(videoTimerHandle);
            videoTimerHandle = setInterval(() => {
                const elapsed = (performance.now() - videoRequestStarted) / 1000;
                const base = videoStatus.dataset.base || '';
                videoStatus.textContent = `${base} · ${elapsed.toFixed(1)}s`;
            }, 200);
        }

        function stopVideoTimer(finalize = false) {
            const elapsed = videoRequestStarted ? (performance.now() - videoRequestStarted) / 1000 : 0;
            if (videoTimerHandle) {
                clearInterval(videoTimerHandle);
                videoTimerHandle = null;
            }
            if (finalize) {
                const base = videoStatus.dataset.base || '';
                videoStatus.textContent = `${base} · ${elapsed.toFixed(1)}s`;
            }
            videoRequestStarted = 0;
        }

        function setMode(mode) {
            currentMode = mode;
            textModeBtn.classList.toggle('active', mode === 'text');
            imageModeBtn.classList.toggle('active', mode === 'image');
            videoModeBtn.classList.toggle('active', mode === 'video');
            textSearchBox.style.display = mode === 'text' ? 'flex' : 'none';
            imageSearchBox.style.display = mode === 'image' ? 'flex' : 'none';
            videoBox.style.display = mode === 'video' ? 'flex' : 'none';
        }

        const savedVideoPrompt = localStorage.getItem('evs_video_prompt');
        if (savedVideoPrompt && videoPromptInput) {
            videoPromptInput.value = savedVideoPrompt;
            if (saveVideoPromptInput) {
                saveVideoPromptInput.checked = true;
            }
        }

        setMode(currentMode);
        
        // Settings modal elements
        const settingsBtn = document.getElementById('settingsBtn');
        const settingsModal = document.getElementById('settingsModal');
        const closeSettingsBtn = document.getElementById('closeSettings');
        const saveSettingsBtn = document.getElementById('saveSettings');
        const resetSettingsBtn = document.getElementById('resetSettings');
        const settingsStatus = document.getElementById('settingsStatus');
        const thumbnailQualitySlider = document.getElementById('thumbnailQuality');
        const qualityValue = document.getElementById('qualityValue');
        const embedderSelect = document.getElementById('embedder');
        const fusionEnabledInput = document.getElementById('fusionEnabled');
        const fusionAlphaInput = document.getElementById('fusionAlpha');
        const fusionAlphaValue = document.getElementById('fusionAlphaValue');
        const dinoModelInput = document.getElementById('dinoModel');
        const dinoEmbedDimInput = document.getElementById('dinoEmbedDim');
        const dinoWeightsInput = document.getElementById('dinoWeightsPath');
        const indexModeSelect = document.getElementById('indexMode');
        const rerankEnabledInput = document.getElementById('rerankEnabled');
        const rerankTopKInput = document.getElementById('rerankTopK');
        const segmentsEnabledInput = document.getElementById('segmentsEnabled');
        const segmentMinPatchesInput = document.getElementById('segmentMinPatches');
        const segmentThresholdSlider = document.getElementById('segmentThresholdSlider');
        const segmentThresholdValueEl = document.getElementById('segmentThresholdValue');
        const segmentThresholdControl = document.getElementById('segmentThresholdControl');
        
        let segmentThreshold = 0.7;

        function clampSegmentThreshold(value) {
            const numeric = Number.parseFloat(value);
            if (!Number.isFinite(numeric)) {
                return segmentThreshold;
            }
            return Math.min(0.99, Math.max(0.0, numeric));
        }

        function setSegmentThresholdFromPercent(percentValue) {
            const pct = Number.parseInt(percentValue, 10);
            const clamped = Math.min(99, Math.max(0, Number.isFinite(pct) ? pct : Math.round(segmentThreshold * 100)));
            segmentThreshold = clamped / 100;
            if (segmentThresholdSlider) {
                segmentThresholdSlider.value = String(clamped);
            }
            if (segmentThresholdValueEl) {
                segmentThresholdValueEl.textContent = `${clamped}%`;
            }
        }

        function formatPercent(value) {
            if (!Number.isFinite(value)) {
                return 'n/a';
            }
            return `${(value * 100).toFixed(1)}%`;
        }

        function buildSimilarityMetrics(result, isCommented = false) {
            if (isCommented) {
                const count = result.comment_count || 0;
                const latest = (result.latest_comment || '').toString();
                const trimmed = latest.length > 50 ? `${latest.substring(0, 50)}...` : latest;
                return `<div class="metric-line"><span class="metric-label">Comments:</span> ${count}${trimmed ? ` <span class="metric-note">Latest: ${trimmed}</span>` : ''}</div>`;
            }

            const lines = [];
            lines.push(`<div class="metric-line"><span class="metric-label">Final:</span> ${formatPercent(result.similarity)}</div>`);

            if (result.rerank) {
                const originalScore = formatPercent(result.rerank.original_score);
                if (Number.isFinite(result.rerank.original_score)) {
                    lines.push(`<div class="metric-line"><span class="metric-label">Original:</span> ${originalScore}</div>`);
                }

                if (Number.isFinite(result.rerank.score)) {
                    const rerankScore = formatPercent(result.rerank.score);
                    const note = result.rerank.applied ? '' : '<span class="metric-note">fallback</span>';
                    lines.push(`<div class="metric-line"><span class="metric-label">Rerank:</span> ${rerankScore}${note}</div>`);
                }
            }

            if (result.fusion) {
                if (Number.isFinite(result.fusion.clip_similarity)) {
                    lines.push(`<div class="metric-line"><span class="metric-label">CLIP:</span> ${formatPercent(result.fusion.clip_similarity)}</div>`);
                }
                if (Number.isFinite(result.fusion.dino_similarity)) {
                    lines.push(`<div class="metric-line"><span class="metric-label">DINO:</span> ${formatPercent(result.fusion.dino_similarity)}</div>`);
                }
                if (Number.isFinite(result.fusion.alpha)) {
                    lines.push(`<div class="metric-line"><span class="metric-label">Fusion α:</span> ${result.fusion.alpha.toFixed(2)}</div>`);
                }
            }

            if (!lines.length) {
                lines.push(`<div class="metric-line"><span class="metric-label">Similarity:</span> ${formatPercent(result.similarity)}</div>`);
            }

            return lines.join('');
        }
        
        // Settings modal functionality
        settingsBtn.addEventListener('click', () => {
            settingsModal.style.display = 'block';
            loadSettings();
        });
        
        closeSettingsBtn.addEventListener('click', () => {
            settingsModal.style.display = 'none';
        });
        
        // Close modal when clicking outside
        settingsModal.addEventListener('click', (e) => {
            if (e.target === settingsModal) {
                settingsModal.style.display = 'none';
            }
        });
        
        // Thumbnail quality slider update
        thumbnailQualitySlider.addEventListener('input', (e) => {
            qualityValue.textContent = e.target.value;
        });

        fusionAlphaInput.addEventListener('input', () => {
            fusionAlphaValue.textContent = Number(fusionAlphaInput.value).toFixed(2);
        });

        if (segmentThresholdSlider) {
            segmentThresholdSlider.addEventListener('input', (e) => {
                setSegmentThresholdFromPercent(e.target.value);
            });
            setSegmentThresholdFromPercent(segmentThresholdSlider.value);
        }

        fusionEnabledInput.addEventListener('change', () => {
            updateFusionUI(fusionEnabledInput.checked);
        });

        rerankEnabledInput.addEventListener('change', () => {
            updateRerankUI(rerankEnabledInput.checked);
        });

        segmentsEnabledInput.addEventListener('change', () => {
            updateSegmentsUI(segmentsEnabledInput.checked);
            refreshSegmentsPanels();
        });

        // Load current settings
        async function loadSettings() {
            try {
                const response = await fetch('/settings');
                const data = await response.json();

                if (data.success) {
                    const settings = data.settings;
                    document.getElementById('host').value = settings.host;
                    document.getElementById('port').value = settings.port;
                    document.getElementById('debug').checked = settings.debug;
                    embedderSelect.value = settings.embedder || 'clip';
                    fusionEnabledInput.checked = Boolean(settings.fusionEnabled);
                    const parsedFusionAlpha = parseFloat(settings.fusionAlpha);
                    const fusionAlpha = Number.isFinite(parsedFusionAlpha) ? parsedFusionAlpha : 0.7;
                    fusionAlphaInput.value = fusionAlpha.toFixed(2);
                    dinoModelInput.value = settings.dinoModel || 'dinov3_vitb16';
                    dinoEmbedDimInput.value = settings.dinoEmbedDim || 1280;
                    dinoWeightsInput.value = settings.dinoWeightsPath || '';
                    indexModeSelect.value = settings.indexMode || 'clip';
                    updateFusionUI(fusionEnabledInput.checked);
                    rerankEnabledInput.checked = Boolean(settings.rerankEnabled);
                    const parsedRerankTopK = parseInt(settings.rerankTopK, 10);
                    rerankTopKInput.value = Number.isFinite(parsedRerankTopK) ? parsedRerankTopK : 50;
                    updateRerankUI(rerankEnabledInput.checked);
                    document.getElementById('clipModel').value = settings.clipModel;
                    document.getElementById('minResults').value = settings.minResults;
                    document.getElementById('maxResults').value = settings.maxResults;
                    document.getElementById('defaultResults').value = settings.defaultResults;
                    document.getElementById('batchSize').value = settings.batchSize;
                    document.getElementById('thumbnailQuality').value = settings.thumbnailQuality;
                    document.getElementById('qualityValue').textContent = settings.thumbnailQuality;
                    document.getElementById('maxCommentLength').value = settings.maxCommentLength;
                    document.getElementById('maxFileSize').value = settings.maxFileSize;
                    document.getElementById('indexFolderName').value = settings.indexFolderName;
                    applyEmbedderUI(embedderSelect.value);
                    segmentsEnabledInput.checked = Boolean(settings.segmentsEnabled);
                    segmentMinPatchesInput.value = settings.segmentMinPatches || 3;
                    const thresholdRaw = clampSegmentThreshold(settings.segmentThreshold);
                    const pctValue = Math.round(thresholdRaw * 100);
                    setSegmentThresholdFromPercent(pctValue);
                    updateSegmentsUI(segmentsEnabledInput.checked);
                    refreshSegmentsPanels();
                } else {
                    showSettingsStatus('Error loading settings: ' + data.error, 'error');
                }
            } catch (error) {
                showSettingsStatus('Error loading settings: ' + error.message, 'error');
            }
        }
        
        // Save settings
        saveSettingsBtn.addEventListener('click', async () => {
            try {
                const settings = {
                    host: document.getElementById('host').value.trim(),
                    port: parseInt(document.getElementById('port').value),
                    debug: document.getElementById('debug').checked,
                    embedder: embedderSelect.value,
                    fusionEnabled: fusionEnabledInput.checked,
                    fusionAlpha: parseFloat(fusionAlphaInput.value),
                    rerankEnabled: rerankEnabledInput.checked,
                    rerankTopK: parseInt(rerankTopKInput.value),
                    segmentsEnabled: segmentsEnabledInput.checked,
                    segmentMinPatches: parseInt(segmentMinPatchesInput.value),
                    segmentThreshold: segmentThreshold,
                    clipModel: document.getElementById('clipModel').value,
                    dinoModel: dinoModelInput.value.trim(),
                    dinoEmbedDim: parseInt(dinoEmbedDimInput.value),
                    dinoWeightsPath: dinoWeightsInput.value.trim(),
                    indexMode: indexModeSelect.value,
                    minResults: parseInt(document.getElementById('minResults').value),
                    maxResults: parseInt(document.getElementById('maxResults').value),
                    defaultResults: parseInt(document.getElementById('defaultResults').value),
                    batchSize: parseInt(document.getElementById('batchSize').value),
                    thumbnailQuality: parseInt(document.getElementById('thumbnailQuality').value),
                    maxCommentLength: parseInt(document.getElementById('maxCommentLength').value),
                    maxFileSize: parseInt(document.getElementById('maxFileSize').value),
                    indexFolderName: document.getElementById('indexFolderName').value.trim()
                };
                
                // Basic validation
                if (!settings.host) {
                    showSettingsStatus('Host cannot be empty', 'error');
                    return;
                }
                
                if (settings.minResults >= settings.maxResults) {
                    showSettingsStatus('Min results must be less than max results', 'error');
                    return;
                }
                
                if (settings.defaultResults < settings.minResults || settings.defaultResults > settings.maxResults) {
                    showSettingsStatus('Default results must be between min and max results', 'error');
                    return;
                }

                if (!Number.isFinite(settings.dinoEmbedDim) || settings.dinoEmbedDim <= 0) {
                    settings.dinoEmbedDim = parseInt(dinoEmbedDimInput.placeholder) || 1280;
                }

                if (!Number.isFinite(settings.fusionAlpha) || settings.fusionAlpha < 0 || settings.fusionAlpha > 1) {
                    const defaultAlpha = parseFloat(fusionAlphaInput.defaultValue || '0.7');
                    settings.fusionAlpha = Number.isFinite(defaultAlpha) ? defaultAlpha : 0.7;
                }

                if (!settings.fusionEnabled && settings.embedder === 'fusion') {
                    settings.embedder = 'clip';
                }

                if (!Number.isFinite(settings.rerankTopK) || settings.rerankTopK < 1) {
                    const defaultTopK = parseInt(rerankTopKInput.placeholder) || 50;
                    settings.rerankTopK = Number.isFinite(defaultTopK) && defaultTopK > 0 ? defaultTopK : 50;
                }

                if (!Number.isFinite(settings.segmentMinPatches) || settings.segmentMinPatches < 1) {
                    const defaultSegments = parseInt(segmentMinPatchesInput.placeholder) || 3;
                    settings.segmentMinPatches = Number.isFinite(defaultSegments) && defaultSegments > 0 ? defaultSegments : 3;
                }

                settings.segmentThreshold = clampSegmentThreshold(settings.segmentThreshold);

                if (settings.embedder === 'dino' && !settings.dinoModel) {
                    showSettingsStatus('DINO model name is required when DINO backend is selected', 'error');
                    return;
                }
                
                saveSettingsBtn.disabled = true;
                saveSettingsBtn.textContent = 'Saving...';
                
                const response = await fetch('/settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(settings)
                });
                
                const data = await response.json();
                
                if (data.success) {
                    showSettingsStatus(data.message, 'success');
                } else {
                    showSettingsStatus('Error saving settings: ' + data.error, 'error');
                }
                
            } catch (error) {
                showSettingsStatus('Error saving settings: ' + error.message, 'error');
            } finally {
                saveSettingsBtn.disabled = false;
                saveSettingsBtn.textContent = 'Save Settings';
            }
        });
        
        // Reset settings to defaults
        resetSettingsBtn.addEventListener('click', () => {
            if (confirm('Reset all settings to default values?')) {
                document.getElementById('host').value = '0.0.0.0';
                document.getElementById('port').value = '5000';
                document.getElementById('debug').checked = false;
                embedderSelect.value = 'clip';
                fusionEnabledInput.checked = false;
                fusionAlphaInput.value = '0.70';
                fusionAlphaValue.textContent = '0.70';
                rerankEnabledInput.checked = false;
                rerankTopKInput.value = '50';
                segmentsEnabledInput.checked = false;
                segmentMinPatchesInput.value = '3';
                setSegmentThresholdFromPercent(70);
                dinoModelInput.value = 'dinov3_vitb16';
                dinoEmbedDimInput.value = '1280';
                dinoWeightsInput.value = '';
                indexModeSelect.value = 'clip';
                document.getElementById('clipModel').value = 'ViT-B/32';
                document.getElementById('minResults').value = '3';
                document.getElementById('maxResults').value = '48';
                document.getElementById('defaultResults').value = '12';
                document.getElementById('batchSize').value = '32';
                document.getElementById('thumbnailQuality').value = '85';
                document.getElementById('qualityValue').textContent = '85';
                document.getElementById('maxCommentLength').value = '100';
                document.getElementById('maxFileSize').value = '50';
                document.getElementById('indexFolderName').value = '.clip_index';
                updateFusionUI(false);
                updateRerankUI(false);
                updateSegmentsUI(false);
                refreshSegmentsPanels();
                applyEmbedderUI(embedderSelect.value);
            }
        });

        // Show settings status message
        function showSettingsStatus(message, type) {
            settingsStatus.textContent = message;
            settingsStatus.className = `settings-status ${type}`;
            settingsStatus.style.display = 'block';
            
            setTimeout(() => {
                settingsStatus.style.display = 'none';
            }, 5000);
        }

        function updateFusionUI(enabled) {
            fusionAlphaInput.disabled = !enabled;
            fusionAlphaValue.textContent = Number(fusionAlphaInput.value).toFixed(2);
            fusionAlphaValue.classList.toggle('disabled', !enabled);
            const fusionOption = embedderSelect.querySelector('option[value="fusion"]');
            if (fusionOption) {
                fusionOption.disabled = !enabled;
            }
            if (!enabled && embedderSelect.value === 'fusion') {
                embedderSelect.value = 'clip';
                applyEmbedderUI('clip');
            }
        }

        function updateRerankUI(enabled) {
            rerankTopKInput.disabled = !enabled;
            rerankTopKInput.classList.toggle('disabled', !enabled);
        }

        updateFusionUI(fusionEnabledInput.checked);
        updateRerankUI(rerankEnabledInput.checked);
        
        function updateSegmentsUI(enabled) {
            segmentMinPatchesInput.disabled = !enabled;
            segmentMinPatchesInput.classList.toggle('disabled', !enabled);
            updateSegmentControlsUI(enabled);
        }

        function updateSegmentControlsUI(enabled) {
            if (!segmentThresholdSlider || !segmentThresholdControl) return;
            segmentThresholdSlider.disabled = !enabled;
            segmentThresholdControl.classList.toggle('disabled', !enabled);
        }

        updateSegmentsUI(segmentsEnabledInput.checked);
        refreshSegmentsPanels();

        function applyEmbedderUI(embedder) {
            const showDino = embedder === 'dino' || embedder === 'fusion';
            const dinoRows = document.querySelectorAll('.backend-dino');
            dinoRows.forEach(row => {
                row.style.display = showDino ? 'flex' : 'none';
            });

            const clipRows = document.querySelectorAll('.backend-clip');
            clipRows.forEach(row => {
                row.style.display = embedder === 'dino' ? 'none' : 'flex';
            });

            if (embedder === 'dino') {
                textModeBtn.disabled = true;
                textModeBtn.title = 'Text search is only available with the CLIP backend.';
                textModeBtn.classList.remove('active');
                setMode('image');
            } else {
                textModeBtn.disabled = false;
                textModeBtn.title = '';
                if (currentMode === 'text') {
                    setMode('text');
                }
            }
        }

        embedderSelect.addEventListener('change', (event) => {
            applyEmbedderUI(event.target.value);
        });
        applyEmbedderUI(embedderSelect.value);
        
        // Mode switching
        textModeBtn.addEventListener('click', () => setMode('text'));
        imageModeBtn.addEventListener('click', () => setMode('image'));
        videoModeBtn.addEventListener('click', () => setMode('video'));
        
        // Check index status
        async function checkIndexStatus(folder) {
            try {
                const response = await fetch('/check_index', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ folder })
                });
                return await response.json();
            } catch (error) {
                return { indexed: false, available_modes: [] };
            }
        }
        
        // Index folder
        indexBtn.addEventListener('click', async () => {
            const folder = folderInput.value.trim();
            if (!folder) return;
            
            indexStatus.textContent = 'Indexing...';
            indexStatus.className = 'status';
            indexBtn.disabled = true;
            
            try {
                const response = await fetch('/index', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ folder })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    const counts = data.counts || {};
                    const summary = Object.keys(counts).length > 0
                        ? Object.entries(counts).map(([mode, count]) => `${mode}: ${count}`).join(' | ')
                        : `Active: ${data.count || 0}`;
                    indexStatus.textContent = `Indexed successfully (${summary})`;
                    indexStatus.className = 'status success';
                    currentFolder = folder;
                    const modes = data.modes || [];
                    if (modes.includes(embedderSelect.value)) {
                        applyEmbedderUI(embedderSelect.value);
                    }
                } else {
                    indexStatus.textContent = data.error || 'Indexing failed';
                    indexStatus.className = 'status error';
                }
            } catch (error) {
                indexStatus.textContent = 'Error: ' + error.message;
                indexStatus.className = 'status error';
            } finally {
                indexBtn.disabled = false;
            }
        });
        
        // Text search
        searchBtn.addEventListener('click', async () => {
            const query = searchInput.value.trim();
            const folder = folderInput.value.trim();
            const limit = resultLimitSelect.value;
            const sortBy = sortBySelect.value;
            
            if (!query || !folder) return;
            
            resultsContainer.innerHTML = '<div class="loading"><div class="spinner"></div> Searching...</div>';
            
            try {
                const response = await fetch('/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ folder, query, limit, sort_by: sortBy })
                });
                
                const data = await response.json();
                
                if (data.results && data.results.length > 0) {
                    displayResults(data.results);
                } else {
                    resultsContainer.innerHTML = '<div class="loading">No results found</div>';
                }
            } catch (error) {
                resultsContainer.innerHTML = '<div class="loading">Error: ' + error.message + '</div>';
            }
        });
        
        // Image search
        imageSearchBtn.addEventListener('click', async () => {
            const folder = folderInput.value.trim();
            const file = imageUpload.files[0];
            const imagePathValue = imagePath.value.trim();
            const limit = resultLimitSelect.value;
            const sortBy = sortBySelect.value;
            
            // Check if we have either a file or a path
            if (!folder || (!file && !imagePathValue)) {
                alert('Please select a folder and either upload an image file or enter an image path.');
                return;
            }
            
            resultsContainer.innerHTML = '<div class="loading"><div class="spinner"></div> Searching by image...</div>';
            
            try {
                const formData = new FormData();
                formData.append('folder', folder);
                formData.append('limit', limit);
                formData.append('sort_by', sortBy);
                
                // Prioritize file upload over path
                if (file) {
                    formData.append('image', file);
                } else if (imagePathValue) {
                    formData.append('image_path', imagePathValue);
                }
                
                const response = await fetch('/search_by_image', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.results && data.results.length > 0) {
                    displayResults(data.results);
                } else {
                    resultsContainer.innerHTML = '<div class="loading">No results found</div>';
                }
            } catch (error) {
                resultsContainer.innerHTML = '<div class="loading">Error: ' + error.message + '</div>';
            }
        });

        function renderVideoFrames(frames) {
            if (!videoFrames) return;
            if (!frames || !frames.length) {
                videoFrames.innerHTML = '';
                return;
            }
            const html = frames.map((frame, idx) => {
                const ts = typeof frame.time_sec === 'number' ? `${frame.time_sec.toFixed(2)}s` : 'n/a';
                return `<div title="Frame ${idx + 1} (${ts})"><img src="data:image/jpeg;base64,${frame.thumbnail}" alt="Frame ${idx + 1}" /></div>`;
            }).join('');
            videoFrames.innerHTML = html;
        }

        async function runVideoUnderstanding() {
            const videoPath = videoPathInput.value.trim();
            const frameCount = parseInt(videoFrameCount.value, 10) || 16;
            const sampleFpsValue = Number.parseFloat(videoSampleFpsInput.value);
            const prompt = videoPromptInput.value.trim();

            if (!videoPath) {
                videoStatus.textContent = 'Provide a video path.';
                videoStatus.className = 'video-status error';
                return;
            }

            if (saveVideoPromptInput && saveVideoPromptInput.checked) {
                localStorage.setItem('evs_video_prompt', prompt);
            } else {
                localStorage.removeItem('evs_video_prompt');
            }

            videoRunBtn.disabled = true;
            saveSummaryBtn.style.display = 'none';
            lastSummaryText = '';
            lastSummaryTarget = null;
            videoStatus.dataset.base = 'Sampling frames and querying the model...';
            videoStatus.textContent = videoStatus.dataset.base;
            videoStatus.className = 'video-status';
            videoOutput.style.display = 'none';
            videoOutput.innerHTML = '';
            renderVideoFrames([]);
            startVideoTimer();

            try {
                const payload = {
                    video: videoPath,
                    frame_count: frameCount,
                    prompt,
                };
                if (Number.isFinite(sampleFpsValue) && sampleFpsValue > 0) {
                    payload.sample_fps = sampleFpsValue;
                }
                const response = await fetch('/video_understanding', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                const data = await response.json();
                if (!response.ok || data.error) {
                    videoStatus.dataset.base = data.error || 'Video understanding request failed.';
                    videoStatus.textContent = videoStatus.dataset.base;
                    videoStatus.className = 'video-status error';
                    stopVideoTimer();
                    return;
                }
                const durationLabel = typeof data.duration_sec === 'number' ? ` · Duration: ${formatDuration(data.duration_sec)}` : '';
                videoStatus.dataset.base = `Model: ${data.model || 'LM Studio'} · Frames sent: ${(data.frames || []).length || frameCount}${durationLabel}`;
                videoStatus.textContent = videoStatus.dataset.base;
                if (data.summary) {
                    videoOutput.style.display = 'block';
                    videoOutput.innerHTML = renderMarkdown(data.summary);
                    lastSummaryText = data.summary;
                    lastSummaryTarget = null;
                    saveSummaryBtn.style.display = 'inline-flex';
                } else {
                    videoOutput.style.display = 'block';
                    videoOutput.textContent = '(No summary returned)';
                    lastSummaryText = '';
                    lastSummaryTarget = null;
                    saveSummaryBtn.style.display = 'none';
                }
                renderVideoFrames(data.frames || []);
                stopVideoTimer(true);
            } catch (error) {
                videoStatus.dataset.base = 'Error: ' + error.message;
                videoStatus.textContent = videoStatus.dataset.base;
                videoStatus.className = 'video-status error';
                stopVideoTimer(true);
            } finally {
                videoRunBtn.disabled = false;
            }
        }

        if (videoRunBtn) {
            videoRunBtn.addEventListener('click', runVideoUnderstanding);
        }

        async function saveSummaryAsComment() {
            if (!lastSummaryText || !lastSummaryTarget || !lastSummaryTarget.path) {
                alert('No summary or target image available to save.');
                return;
            }
            const folder = folderInput.value.trim();
            if (!folder) {
                alert('Please enter a folder path first.');
                return;
            }
            try {
                const response = await fetch('/comments', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        folder,
                        image_path: lastSummaryTarget.path,
                        comment: lastSummaryText,
                    }),
                });
                const data = await response.json();
                if (data.success) {
                    alert('Summary saved as comment.');
                } else {
                    alert('Failed to save comment: ' + (data.error || 'Unknown error'));
                }
            } catch (err) {
                alert('Failed to save comment: ' + err.message);
            }
        }

        if (saveSummaryBtn) {
            saveSummaryBtn.addEventListener('click', saveSummaryAsComment);
        }
        
        // Show commented images
        showCommentedBtn.addEventListener('click', async () => {
            const folder = folderInput.value.trim();
            
            if (!folder) {
                alert('Please enter a folder path first');
                return;
            }
            setMode('text');
            
            resultsContainer.innerHTML = '<div class="loading"><div class="spinner"></div> Loading commented images...</div>';
            
            try {
                const response = await fetch('/commented_images', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ folder })
                });
                
                const data = await response.json();
                
                if (data.results && data.results.length > 0) {
                    displayCommentedResults(data.results);
                } else {
                    resultsContainer.innerHTML = '<div class="loading">No commented images found</div>';
                }
            } catch (error) {
                resultsContainer.innerHTML = '<div class="loading">Error: ' + error.message + '</div>';
            }
        });
        
        // Generate common HTML structure for result items
        function generateResultItemHTML(result, index, isCommented = false) {
            const similarityMarkup = buildSimilarityMetrics(result, isCommented);
                
            return `
                <div class="image-container">
                    <img src="data:image/jpeg;base64,${result.thumbnail}" class="thumbnail" alt="" />
                    <div class="image-overlay">
                        <div class="expand-collapse-icon" data-index="${index}">
                            <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="#e3e3e3">
                                <path d="M240-240v-240h72v168h168v72H240Zm408-240v-168H480v-72h240v240h-72Z"/>
                            </svg>
                        </div>
                        <div class="describe-icon" data-index="${index}" data-path="${result.path || ''}">
                            <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="#e3e3e3">
                                <path d="M160-120q-33 0-56.5-23.5T80-200v-560q0-33 23.5-56.5T160-840h545q33 0 56.5 23.5T785-760v160h-80v-160H160v560h545v-160h80v160q0 33-23.5 56.5T705-120H160Zm520-240 57-57-143-143 143-143-57-57-143 143-143-143-57 57 143 143-143 143 57 57 143-143 143 143Z"/>
                            </svg>
                        </div>
                        <div class="find-similar-icon" data-index="${index}" data-path="${result.path}" style="display: none;">
                            <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="#e3e3e3">
                                <path d="M784-120 532-372q-30 24-69 38t-83 14q-109 0-184.5-75.5T120-580q0-109 75.5-184.5T380-840q109 0 184.5 75.5T640-580q0 44-14 83t-38 69l252 252-56 56ZM380-400q75 0 127.5-52.5T560-580q0-75-52.5-127.5T380-760q-75 0-127.5 52.5T200-580q0 75 52.5 127.5T380-400Z"/>
                            </svg>
                        </div>
                    </div>
                </div>
                <div class="result-info">
                    <div class="filename">
                        ${result.filename}
                        <svg class="copy-icon" xmlns="http://www.w3.org/2000/svg" height="16px" viewBox="0 -960 960 960" width="16px" fill="#888">
                            <path d="M360-240q-29.7 0-50.85-21.15Q288-282.3 288-312v-480q0-29.7 21.15-50.85Q330.3-864 360-864h384q29.7 0 50.85 21.15Q816-821.7 816-792v480q0 29.7-21.15 50.85Q773.7-240 744-240H360Zm0-72h384v-480H360v480ZM216-96q-29.7 0-50.85-21.15Q144-138.3 144-168v-552h72v552h456v72H216Zm144-216v-480 480Z"/>
                        </svg>
                    </div>
                    <div class="similarity">${similarityMarkup}</div>
                </div>
                <div class="segments-panel" id="segments-${index}">
                    <div class="segments-status warning">Segments disabled. Enable in settings to propose regions.</div>
                </div>
                <div class="comment-section">
                    <div class="comments-list" id="comments-${index}">
                        <div class="comment-loading">Loading comments...</div>
                    </div>
                    <div class="comment-form">
                        <textarea class="comment-input" placeholder="Add a comment..." id="comment-input-${index}"></textarea>
                        <button class="save-comment-btn" id="save-btn-${index}">Save</button>
                    </div>
                </div>
            `;
        }

        // Setup event handlers for result item
        function setupResultItemEventHandlers(item, result, index) {
            // Handle expand/collapse via overlay icon
            const expandCollapseIcon = item.querySelector('.expand-collapse-icon');
            expandCollapseIcon.addEventListener('click', (e) => {
                e.stopPropagation();
                toggleImageExpansion(item, result, index);
            });
            
            // Handle copy icon click
            const copyIcon = item.querySelector('.copy-icon');
            if (copyIcon) {
                if (result.path) {
                    copyIcon.addEventListener('click', (e) => {
                        e.stopPropagation();
                        copyImagePath(result.path);
                    });
                } else {
                    copyIcon.style.display = 'none';
                }
            }
            
            
            // Handle find similar button
            const findSimilarIcon = item.querySelector('.find-similar-icon');
            if (findSimilarIcon) {
                if (result.path) {
                    findSimilarIcon.addEventListener('click', (e) => {
                        e.stopPropagation();
                        findSimilarImages(result.path);
                    });
                } else {
                    findSimilarIcon.style.display = 'none';
                }
            }

            const describeIcon = item.querySelector('.describe-icon');
            if (describeIcon) {
                if (result.path) {
                    describeIcon.addEventListener('click', (e) => {
                        e.stopPropagation();
                        describeImageWithLM(result.path);
                    });
                } else {
                    describeIcon.style.display = 'none';
                }
            }
            
            // Add save comment functionality
            const saveBtn = item.querySelector(`#save-btn-${index}`);
            const commentInput = item.querySelector(`#comment-input-${index}`);
            
            if (saveBtn) {
                if (result.path) {
                    saveBtn.addEventListener('click', () => {
                        saveComment(index, result.path, folderInput.value.trim(), commentInput.value.trim());
                    });
                } else {
                    saveBtn.disabled = true;
                }
            }

            const img = item.querySelector('.thumbnail');
            if (img) {
                img.addEventListener('click', (e) => {
                    handleSegmentClick(e, result, index, item);
                });
            }
        }

        // Display results
        function displayResults(results) {
            resultsContainer.innerHTML = '';
            
            results.forEach((result, index) => {
                const item = document.createElement('div');
                item.className = 'result-item';
                item.dataset.resultIndex = index;
                item.innerHTML = generateResultItemHTML(result, index, false);
                
                setupResultItemEventHandlers(item, result, index);
                resetSegmentsPanel(item, index);
                resultsContainer.appendChild(item);
            });

            refreshSegmentsPanels();
        }
        
        // Display commented results (similar to displayResults but with comment info)
        function displayCommentedResults(results) {
            resultsContainer.innerHTML = '';
            
            results.forEach((result, index) => {
                const item = document.createElement('div');
                item.className = 'result-item';
                item.dataset.resultIndex = index;
                item.innerHTML = generateResultItemHTML(result, index, true);
                
                setupResultItemEventHandlers(item, result, index);
                resetSegmentsPanel(item, index);
                resultsContainer.appendChild(item);
            });

            refreshSegmentsPanels();
        }
        
        // Comment functionality
        async function loadComments(index, imagePath, folder) {
            const commentsContainer = document.getElementById(`comments-${index}`);
            
            try {
                const response = await fetch(`/comments?folder=${encodeURIComponent(folder)}&image_path=${encodeURIComponent(imagePath)}`);
                const data = await response.json();
                
                if (data.comments && data.comments.length > 0) {
                    displayComments(commentsContainer, data.comments);
                } else {
                    commentsContainer.innerHTML = '<div class="no-comments">No comments yet. Be the first to add one!</div>';
                }
            } catch (error) {
                console.error('Error loading comments:', error);
                commentsContainer.innerHTML = '<div class="no-comments">Error loading comments</div>';
            }
        }
        
        function displayComments(container, comments) {
            container.innerHTML = '';
            comments.forEach(comment => {
                const commentDiv = document.createElement('div');
                commentDiv.className = 'comment-item';
                
                // Parse timestamp and comment text
                const timestampMatch = comment.match(/^\\[(.*?)\\] (.*)$/);
                if (timestampMatch) {
                    const [, timestamp, text] = timestampMatch;
                    commentDiv.innerHTML = `
                        <div class="comment-timestamp">${timestamp}</div>
                        <div class="comment-text">${escapeHtml(text)}</div>
                    `;
                } else {
                    commentDiv.innerHTML = `<div class="comment-text">${escapeHtml(comment)}</div>`;
                }
                
                container.appendChild(commentDiv);
            });
        }
        
        async function saveComment(index, imagePath, folder, comment) {
            if (!comment) return;
            
            const saveBtn = document.getElementById(`save-btn-${index}`);
            const commentInput = document.getElementById(`comment-input-${index}`);
            
            // Disable button during save
            saveBtn.disabled = true;
            saveBtn.textContent = 'Saving...';
            
            try {
                const response = await fetch('/comments', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        folder: folder,
                        image_path: imagePath,
                        comment: comment
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    // Clear input and reload comments
                    commentInput.value = '';
                    const commentsContainer = document.getElementById(`comments-${index}`);
                    displayComments(commentsContainer, data.comments);
                } else {
                    alert('Error saving comment: ' + (data.error || 'Unknown error'));
                }
            } catch (error) {
                console.error('Error saving comment:', error);
                alert('Error saving comment: ' + error.message);
            } finally {
                // Re-enable button
                saveBtn.disabled = false;
                saveBtn.textContent = 'Save';
            }
        }
        
        function toggleImageExpansion(item, result, index) {
            const img = item.querySelector('.thumbnail');
            const expandCollapseIcon = item.querySelector('.expand-collapse-icon');
            const isExpanded = item.classList.contains('expanded');
            
            if (isExpanded) {
                // Collapse: switch back to thumbnail
                img.src = `data:image/jpeg;base64,${result.thumbnail}`;
                item.classList.remove('expanded');
                resetSegmentsPanel(item, index);
                img.classList.remove('segment-enabled');
                // Update icon to expand
                expandCollapseIcon.innerHTML = `
                    <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="#e3e3e3">
                        <path d="M240-240v-240h72v168h168v72H240Zm408-240v-168H480v-72h240v240h-72Z"/>
                    </svg>
                `;
            } else {
                // Expand: show original image and load comments
                const originalImageUrl = `/image/${encodeURIComponent(result.path)}`;
                img.src = originalImageUrl;
                item.classList.add('expanded');
                loadComments(index, result.path, folderInput.value.trim());
                prepareSegmentsPanel(item, result, index);
                if (segmentsEnabledInput.checked) {
                    img.classList.add('segment-enabled');
                }
                // Update icon to collapse
                expandCollapseIcon.innerHTML = `
                    <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="#e3e3e3">
                        <path d="M432-432v240h-72v-168H192v-72h240Zm168-336v168h168v72H528v-240h72Z"/>
                    </svg>
                `;
            }
        }

        function resetSegmentsPanel(item, index) {
            const panel = item.querySelector(`#segments-${index}`);
            if (!panel) return;
            if (!segmentsEnabledInput.checked) {
                panel.innerHTML = '<div class="segments-status warning">Segments disabled. Enable in settings to propose regions.</div>';
            } else {
                panel.innerHTML = '<div class="segments-status">Expand the image and click on an area to propose regions.</div>';
            }
        }

        function prepareSegmentsPanel(item, result, index) {
            const panel = item.querySelector(`#segments-${index}`);
            if (!panel) return;
            if (!segmentsEnabledInput.checked) {
                panel.innerHTML = '<div class="segments-status warning">Segments disabled. Enable in settings to propose regions.</div>';
                return;
            }
            panel.innerHTML = '<div class="segments-status">Click inside the image to propose a region near the selected point.</div>';
        }

        function refreshSegmentsPanels() {
            document.querySelectorAll('.result-item').forEach((item) => {
                const indexAttr = item.dataset.resultIndex;
                if (typeof indexAttr === 'undefined') return;
                const index = parseInt(indexAttr, 10);
                if (Number.isNaN(index)) return;
                if (item.classList.contains('expanded')) {
                    prepareSegmentsPanel(item, null, index);
                    const img = item.querySelector('.thumbnail');
                    if (img) {
                        if (segmentsEnabledInput.checked) {
                            img.classList.add('segment-enabled');
                        } else {
                            img.classList.remove('segment-enabled');
                        }
                    }
                } else {
                    resetSegmentsPanel(item, index);
                    const img = item.querySelector('.thumbnail');
                    if (img) {
                        img.classList.remove('segment-enabled');
                    }
                }
            });
        }

        function clamp01(value) {
            if (!Number.isFinite(value)) return 0;
            return Math.min(1, Math.max(0, value));
        }

        async function handleSegmentClick(event, result, index, item) {
            if (!segmentsEnabledInput.checked) return;
            if (!item.classList.contains('expanded')) return;

            const folder = folderInput.value.trim();
            if (!folder) {
                const panel = item.querySelector(`#segments-${index}`);
                if (panel) {
                    panel.innerHTML = '<div class="segments-status error">Provide a folder path before running region proposals.</div>';
                }
                return;
            }

            if (item.dataset.segmentLoading === '1') {
                return;
            }

            const panel = item.querySelector(`#segments-${index}`);
            if (!panel) return;

            const img = event.currentTarget;
            const rect = img.getBoundingClientRect();
            if (rect.width === 0 || rect.height === 0) return;

            const xNorm = clamp01((event.clientX - rect.left) / rect.width);
            const yNorm = clamp01((event.clientY - rect.top) / rect.height);

            const limitValue = parseInt(resultLimitSelect.value, 10);
            const payload = {
                folder,
                image_path: result.path,
                x: xNorm,
                y: yNorm,
                limit: Number.isFinite(limitValue) ? limitValue : 12,
                sort_by: sortBySelect.value || 'similarity',
                targets: ['images', 'segments'],
                threshold: segmentThreshold,
            };

            item.dataset.segmentLoading = '1';
            panel.innerHTML = '<div class="segments-status">Proposing region around the selected point...</div>';

            try {
                const response = await fetch('/segment_from_point', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                const data = await response.json();
                if (!response.ok || data.error) {
                    throw new Error(data.error || 'Region proposal failed');
                }
                renderSegmentResponse(panel, data, xNorm, yNorm, img.currentSrc || img.src);
            } catch (error) {
                panel.innerHTML = `<div class="segments-status error">Segment error: ${escapeHtml(error.message || String(error))}</div>`;
            } finally {
                delete item.dataset.segmentLoading;
            }
        }

        function renderSegmentResponse(panel, data, xNorm, yNorm, baseImageSrc) {
            const segments = Array.isArray(data && data.segments) ? data.segments : [];
            const overlay = data && data.overlay ? data.overlay : {};
            const pctX = (xNorm * 100).toFixed(1);
            const pctY = (yNorm * 100).toFixed(1);
            const safeBaseSrc = baseImageSrc ? escapeHtml(baseImageSrc) : '';

            const baseOverlayFigure = safeBaseSrc ? `
                <figure class="segment-overlay-figure">
                    <div class="segment-overlay-stack">
                        <img src="${safeBaseSrc}" alt="Expanded image region" />
                        ${overlay.heatmap_png ? `<img class="overlay-layer overlay-heatmap" src="data:image/png;base64,${overlay.heatmap_png}" alt="Heatmap overlay" />` : ''}
                        ${overlay.mask_png ? `<img class="overlay-layer overlay-mask" src="data:image/png;base64,${overlay.mask_png}" alt="Refined mask overlay" />` : ''}
                        <div class="overlay-crosshair" style="left: ${pctX}%; top: ${pctY}%"></div>
                    </div>
                    <figcaption>Region overlay</figcaption>
                </figure>
            ` : '';

            const segmentationFigure = overlay.segmentation_png ? `
                <figure class="segment-segmap-figure">
                    <img class="segment-segmap" src="data:image/png;base64,${overlay.segmentation_png}" alt="Semantic segmentation" />
                    <figcaption>Mask2Former segmentation</figcaption>
                </figure>
            ` : '';

            const legendItems = Array.isArray(overlay.legend)
                ? overlay.legend.map((entry) => {
                    const color = escapeHtml(String(entry.color || '#888'));
                    const labelText = entry.label ? escapeHtml(String(entry.label)) : escapeHtml(String(entry.id || 'class'));
                    const highlightClass = entry.highlight ? ' highlight' : '';
                    return `<div class="segment-legend-item${highlightClass}"><span class="segment-legend-swatch" style="background:${color};"></span><span>${labelText}</span></div>`;
                }).join('')
                : '';

            const legendHtml = legendItems ? `<div class="segment-legend">${legendItems}</div>` : '';

            const overlayHtml = (baseOverlayFigure || segmentationFigure)
                ? `<div class="segment-overlay-grid">${baseOverlayFigure}${segmentationFigure}</div>${legendHtml}`
                : legendHtml;

            const listItems = segments.slice(0, 3).map((segment, idx) => {
                const segId = escapeHtml(String(segment.segment_id || `region-${idx + 1}`));
                const fraction = typeof segment.patch_fraction === 'number'
                    ? `${(segment.patch_fraction * 100).toFixed(1)}% area`
                    : 'Area n/a';
                const patchCount = typeof segment.patch_count === 'number'
                    ? `${segment.patch_count} patch${segment.patch_count === 1 ? '' : 'es'}`
                    : '';
                const humanLabel = segment.label ? ` · ${escapeHtml(String(segment.label))}` : '';

                const matches = Array.isArray(segment.image_results) ? segment.image_results.slice(0, 3) : [];
                const matchRows = matches.map((match, matchIdx) => {
                    const label = escapeHtml(String(match.filename || match.path || `Match ${matchIdx + 1}`));
                    const score = typeof match.similarity === 'number' ? `${(match.similarity * 100).toFixed(1)}%` : 'n/a';
                    const thumb = match.thumbnail
                        ? `<img class="segment-match-thumb" src="data:image/jpeg;base64,${match.thumbnail}" alt="${label}" />`
                        : '<div class="segment-match-thumb placeholder"></div>';
                    return `
                        <div class="segment-match-row">
                            ${thumb}
                            <div class="segment-match-meta">
                                <span>${label}</span>
                                <span>Similarity: ${score}</span>
                            </div>
                        </div>
                    `;
                }).join('');

                const matchList = matchRows || '<div class="segments-status warning">No close matches for this region.</div>';

                return `
                    <li>
                        <span class="segment-title">#${idx + 1} · ${segId}${humanLabel}</span>
                        <span class="segment-meta">${fraction}${patchCount ? ` · ${patchCount}` : ''}</span>
                        <div class="segment-match-list">
                            ${matchList}
                        </div>
                    </li>
                `;
            }).join('');

            const refinementNote = overlay.refinement
                ? `<div class="segment-meta">Mask source: ${escapeHtml(String(overlay.refinement))}${overlay.refined_label ? ` · ${escapeHtml(String(overlay.refined_label))}` : ''}</div>`
                : '';
            const areaNote = typeof overlay.mask_fraction === 'number'
                ? `<div class="segment-meta">Refined mask coverage: ${(overlay.mask_fraction * 100).toFixed(1)}%</div>`
                : '';
            const resultsHtml = listItems
                ? `<ul class="segment-results-list">${listItems}</ul>`
                : '<div class="segments-status warning">Region proposals returned no matches.</div>';

            panel.innerHTML = `
                <div class="segments-status success">Regions proposed near (${pctX}%, ${pctY}%) · ${segments.length} candidate(s)</div>
                ${overlayHtml}
                ${refinementNote}
                ${typeof overlay.threshold === 'number' ? `<div class="segment-meta">Heatmap threshold: ${(overlay.threshold * 100).toFixed(1)}%</div>` : ''}
                ${areaNote}
                ${resultsHtml}
            `;
        }
        
        async function copyImagePath(imagePath) {
            try {
                const textToCopy = imagePath;
                
                if (navigator.clipboard && window.isSecureContext) {
                    // Use modern clipboard API
                    await navigator.clipboard.writeText(textToCopy);
                } else {
                    // Fallback for older browsers
                    const textArea = document.createElement('textarea');
                    textArea.value = textToCopy;
                    textArea.style.position = 'fixed';
                    textArea.style.left = '-999999px';
                    textArea.style.top = '-999999px';
                    document.body.appendChild(textArea);
                    textArea.focus();
                    textArea.select();
                    document.execCommand('copy');
                    textArea.remove();
                }
                
                // Simple console feedback for now (could add toast notification)
                console.log('Copied to clipboard:', imagePath);
                
            } catch (error) {
                console.error('Failed to copy:', error);
            }
        }
        
        async function findSimilarImages(imagePath) {
            const folder = folderInput.value.trim();
            const limit = resultLimitSelect.value;
            const sortBy = sortBySelect.value;
            
            if (!folder) {
                alert('Please enter a folder path first');
                return;
            }
            
            // Show loading state
            indexStatus.textContent = 'Finding similar images...';
            indexStatus.className = 'status';
            
            try {
                // Fetch the image file from the server using existing image route
                const imageResponse = await fetch(`/image/${encodeURIComponent(imagePath)}`);
                if (!imageResponse.ok) {
                    throw new Error('Failed to load image file');
                }
                
                // Convert response to blob
                const imageBlob = await imageResponse.blob();
                
                // Create FormData to match existing search_by_image endpoint
                const formData = new FormData();
                formData.append('image', imageBlob, 'reference_image.jpg');
                formData.append('folder', folder);
                formData.append('limit', limit);
                formData.append('sort_by', sortBy);
                
                // Call existing search_by_image endpoint
                const response = await fetch('/search_by_image', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.results) {
                    if (data.results.length === 0) {
                        indexStatus.textContent = 'No similar images found';
                        indexStatus.className = 'status warning';
                    } else {
                        indexStatus.textContent = `Found ${data.results.length} similar images`;
                        indexStatus.className = 'status success';
                        displayResults(data.results);
                    }
                } else {
                    throw new Error(data.error || 'Unknown error');
                }
            } catch (error) {
                console.error('Find similar error:', error);
                indexStatus.textContent = 'Error finding similar images: ' + error.message;
                indexStatus.className = 'status error';
            }
        }

        async function describeImageWithLM(imagePath) {
            if (!imagePath) {
                alert('No filesystem path is available for this image.');
                return;
            }
            const prompt = videoPromptInput.value.trim();
            setMode('video');
            videoStatus.dataset.base = 'Querying model...';
            videoStatus.textContent = videoStatus.dataset.base;
            videoStatus.className = 'video-status';
            videoRunBtn.disabled = true;
            saveSummaryBtn.style.display = 'none';
            videoOutput.style.display = 'none';
            videoOutput.innerHTML = '';
            renderVideoFrames([]);
            startVideoTimer();

            try {
                const response = await fetch('/describe_image', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image_path: imagePath, prompt }),
                });
                const data = await response.json();
                if (!response.ok || data.error) {
                    videoStatus.dataset.base = data.error || 'Describe request failed.';
                    videoStatus.textContent = videoStatus.dataset.base;
                    videoStatus.className = 'video-status error';
                    stopVideoTimer();
                    return;
                }
                videoStatus.dataset.base = `Model: ${data.model || 'LM Studio'} · Image described`;
                videoStatus.textContent = videoStatus.dataset.base;
                if (data.summary) {
                    videoOutput.style.display = 'block';
                    videoOutput.innerHTML = renderMarkdown(data.summary);
                    lastSummaryText = data.summary;
                    lastSummaryTarget = { path: imagePath };
                    saveSummaryBtn.style.display = 'inline-flex';
                }
                if (data.thumbnail) {
                    videoFrames.innerHTML = `<div title="Image"><img src="data:image/jpeg;base64,${data.thumbnail}" alt="Image" /></div>`;
                }
                stopVideoTimer(true);
            } catch (err) {
                videoStatus.dataset.base = 'Error: ' + err.message;
                videoStatus.textContent = videoStatus.dataset.base;
                videoStatus.className = 'video-status error';
                stopVideoTimer(true);
            } finally {
                videoRunBtn.disabled = false;
            }
        }
        
        // Enter key support
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') searchBtn.click();
        });
        
        folderInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') indexBtn.click();
        });
        
        // Check index on folder change
        folderInput.addEventListener('blur', async () => {
            const folder = folderInput.value.trim();
            if (folder) {
                const status = await checkIndexStatus(folder);
                if (status.indexed) {
                    indexStatus.textContent = `Folder is indexed (${(status.available_modes || []).join(', ') || embedderSelect.value})`;
                    indexStatus.className = 'status success';
                } else {
                    const available = (status.available_modes || []).join(', ');
                    indexStatus.textContent = available ? `Folder indexed for: ${available}` : 'Folder not indexed';
                    indexStatus.className = available ? 'status warning' : 'status';
                }
            }
        });
    </script>
</body>
</html>
    '''
    
    # Replace the placeholder with actual options and timestamp
    current_timestamp = str(int(time.time()))
    response_html = html_template.replace('{result_options_html}', result_options_html)
    response_html = response_html.replace('{timestamp}', current_timestamp)
    
    # Create response with cache-busting headers
    response = make_response(response_html)
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    
    return response


@app.route('/image/<path:filepath>')
def serve_image(filepath):
    """Serve original images from absolute paths."""
    try:
        decoded = unquote(filepath)
        path_obj = Path(decoded)
        if '..' in path_obj.parts:
            return "Access denied", 403
        if not path_obj.is_absolute():
            path_obj = Path('/') / path_obj
        abs_path = path_obj.resolve()
        if not abs_path.exists() or not abs_path.is_file():
            return "Image not found", 404
        return send_file(str(abs_path))
    except Exception as exc:
        return f"Error serving image: {exc}", 500


def _index_directory(folder_path: Union[str, Path], embedder: str) -> Path:
    root = Path(folder_path) / config.INDEX_FOLDER_NAME
    return root / EMBEDDER_SUBDIRS[embedder]


def _segment_index_directory(folder_path: Union[str, Path]) -> Path:
    return _index_directory(folder_path, 'dino') / 'segments'


def _image_to_base64(img: Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()


def _encode_jpeg(img: Image.Image, max_edge: Optional[int] = None, quality: int = 85) -> str:
    if max_edge and max(img.size) > max_edge:
        scale = max_edge / float(max(img.size))
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=quality)
    return base64.b64encode(buffer.getvalue()).decode()


def _create_overlay_rgba(alpha_image: Image.Image, color: Tuple[int, int, int], opacity_scale: float = 1.0) -> Image.Image:
    alpha = alpha_image.convert('L')
    scale = float(opacity_scale)
    if scale <= 0:
        alpha = alpha.point(lambda _: 0)
    elif scale < 0.999:
        alpha = alpha.point(lambda v: int(max(0, min(255, v * scale))))
    overlay = Image.new('RGBA', alpha.size, color + (0,))
    overlay.putalpha(alpha)
    return overlay


_SEGMENT_COLOR_TABLE: Tuple[Tuple[int, int, int], ...] = (
    (244, 67, 54),
    (30, 136, 229),
    (102, 187, 106),
    (255, 202, 40),
    (171, 71, 188),
    (255, 112, 67),
    (66, 165, 245),
    (38, 166, 154),
    (156, 204, 101),
    (255, 238, 88),
    (239, 83, 80),
    (126, 87, 194),
    (0, 188, 212),
    (255, 171, 145),
    (156, 39, 176),
    (124, 179, 66),
)


def _class_color(class_id: int) -> Tuple[int, int, int]:
    table = _SEGMENT_COLOR_TABLE
    return table[class_id % len(table)]


def _sample_video_frames(
    video_path: Union[str, Path],
    max_frames: int,
    sample_fps: Optional[float],
    max_edge: int,
) -> Tuple[List[Dict[str, Any]], float, Optional[float]]:
    path_obj = Path(video_path)
    cap = cv2.VideoCapture(str(path_obj))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    duration = (total_frames / fps) if fps > 0 else None
    target_frames = max(1, min(int(max_frames), config.LM_VIDEO_MAX_FRAMES))

    if total_frames > 0:
        if sample_fps and sample_fps > 0 and fps > 0:
            step = max(1, int(round(fps / sample_fps)))
            frame_indices = list(range(0, total_frames, step))
        else:
            frame_indices = list(np.linspace(0, total_frames - 1, num=target_frames, dtype=int))
        frame_indices = sorted(set(frame_indices))[:target_frames]
    else:
        frame_indices = list(range(target_frames))

    frames: List[Dict[str, Any]] = []
    try:
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            frames.append(
                {
                    'index': int(idx),
                    'time_sec': round(idx / fps, 2) if fps > 0 else None,
                    'thumbnail': _encode_jpeg(pil_img, max_edge=max_edge, quality=85),
                    'width': pil_img.width,
                    'height': pil_img.height,
                }
            )
            if len(frames) >= target_frames:
                break
    finally:
        cap.release()
    return frames, fps, duration


def _build_video_messages(video_path: str, frames: List[Dict[str, Any]], user_prompt: str) -> List[Dict[str, Any]]:
    prompt = (user_prompt or '').strip() or "Summarize the key events, people, and objects in this video."
    intro = f"Video file: {Path(video_path).name}. {len(frames)} sampled frames are provided."
    user_content: List[Dict[str, Any]] = [{'type': 'text', 'text': f"{intro}\n\nTask: {prompt}"}]
    for idx, frame in enumerate(frames):
        ts = frame.get('time_sec')
        ts_label = f"{ts:.2f}s" if isinstance(ts, (int, float)) else "n/a"
        user_content.append({'type': 'text', 'text': f"Frame {idx + 1} (t={ts_label})"})
        user_content.append(
            {
                'type': 'image_url',
                'image_url': {
                    'url': f"data:image/jpeg;base64,{frame['thumbnail']}",
                    'detail': 'high',
                },
            }
        )
    system_msg = (
        "You analyze short videos using sampled frames. Provide a concise, factual summary, key events, "
        "notable objects, people, and any scene changes. Avoid repeating the same detail for each frame."
    )
    return [
        {'role': 'system', 'content': [{'type': 'text', 'text': system_msg}]},
        {'role': 'user', 'content': user_content},
    ]


def _call_lm_chat(messages: List[Dict[str, Any]]) -> str:
    base_url = (config.LM_BASE_URL or '').rstrip('/')
    if not base_url:
        raise RuntimeError("EVOSSEARCH_LM_BASE_URL is not configured.")
    endpoint = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if config.LM_API_KEY:
        headers["Authorization"] = f"Bearer {config.LM_API_KEY}"

    payload = {
        "model": config.LM_MODEL,
        "messages": messages,
        "temperature": float(config.LM_VIDEO_TEMPERATURE),
        "max_tokens": int(config.LM_VIDEO_MAX_TOKENS),
    }
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=config.LM_TIMEOUT)
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"LM Studio request failed: {exc}") from exc

    data = response.json()
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message", {}) or {}
    content = message.get("content", "")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        content_text = "\n".join(parts).strip()
    else:
        content_text = str(content).strip()
    return content_text or "(empty response from model)"


def _call_video_understanding(messages: List[Dict[str, Any]]) -> str:
    return _call_lm_chat(messages)


def _build_image_messages(image_path: str, prompt: str) -> List[Dict[str, Any]]:
    user_prompt = (prompt or '').strip() or "Describe the main content of this image clearly and concisely."
    user_content = [
        {'type': 'text', 'text': f"Image: {Path(image_path).name}\n\nTask: {user_prompt}"},
        {
            'type': 'image_url',
            'image_url': {
                'url': f"data:image/jpeg;base64,{_encode_jpeg(Image.open(image_path), max_edge=config.THUMBNAIL_SIZE[0])}",
                'detail': 'high',
            },
        },
    ]
    return [
        {'role': 'system', 'content': [{'type': 'text', 'text': 'You are an expert visual analyst. Be concise and factual.'}]},
        {'role': 'user', 'content': user_content},
    ]


def _rgb_to_hex(color: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)


def _render_segmentation_overlay(
    seg_map: np.ndarray,
    label_lookup: Optional[Dict[int, str]],
    highlight_id: Optional[int],
) -> Tuple[Optional[Image.Image], List[Dict[str, Any]]]:
    if seg_map is None or not isinstance(seg_map, np.ndarray):
        return None, []
    seg_int = np.asarray(seg_map, dtype=np.int32)
    if seg_int.ndim != 2:
        return None, []
    height, width = seg_int.shape
    if height == 0 or width == 0:
        return None, []

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    legend: List[Dict[str, Any]] = []

    labels = label_lookup or {}
    unique_ids = np.unique(seg_int)
    for class_id in unique_ids:
        class_int = int(class_id)
        color = _class_color(class_int)
        mask = seg_int == class_int
        alpha = 220 if highlight_id is not None and class_int == highlight_id else 150
        rgba[..., :3][mask] = color
        rgba[..., 3][mask] = alpha
        legend.append(
            {
                'id': class_int,
                'label': labels.get(class_int, f'class_{class_int}'),
                'color': _rgb_to_hex(color),
                'highlight': bool(highlight_id is not None and class_int == highlight_id),
            }
        )

    legend.sort(key=lambda entry: (0 if entry['highlight'] else 1, entry['label']))

    overlay_img = Image.fromarray(rgba, mode='RGBA')
    return overlay_img, legend


def _available_indexes(folder_path: Union[str, Path]) -> List[str]:
    available: List[str] = []
    root = Path(folder_path) / config.INDEX_FOLDER_NAME
    has_clip = False
    has_dino = False
    for embedder in EMBEDDER_SUBDIRS:
        embed_dir = _index_directory(folder_path, embedder)
        if (embed_dir / 'index.faiss').exists():
            available.append(embedder)
            if embedder == 'clip':
                has_clip = True
            elif embedder == 'dino':
                has_dino = True
            continue
        if embedder == 'clip' and (root / 'index.faiss').exists():
            available.append(embedder)
            has_clip = True
    if config.FUSION_ENABLED and has_clip and has_dino:
        available.append('fusion')
    return available


def save_index(index_results: Dict[str, Tuple[faiss.Index, List[str], List[Dict[str, Any]], Dict[str, Any]]], folder_path) -> None:
    """Persist FAISS indexes for one or more embedders."""
    for embedder, (index, image_paths, image_metadata, index_meta) in index_results.items():
        embed_dir = _index_directory(folder_path, embedder)
        embed_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(embed_dir / 'index.faiss'))

        with open(embed_dir / 'paths.pkl', 'wb') as f:
            pickle.dump(image_paths, f)

        with open(embed_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(image_metadata, f)

        meta_path = embed_dir / 'meta.json'
        meta_path.write_text(json.dumps(index_meta, indent=2), encoding='utf-8')


def load_index(folder_path, embedder: Optional[str] = None):
    """Load FAISS index, metadata, and embedder info for the requested backend."""
    target = embedder or active_embedder
    if target not in EMBEDDER_SUBDIRS:
        return None, None, None, {}

    root = Path(folder_path) / config.INDEX_FOLDER_NAME
    embed_dir = _index_directory(folder_path, target)
    legacy_dir = root if target == 'clip' else None
    if not embed_dir.exists():
        if legacy_dir is None or not (legacy_dir / 'index.faiss').exists():
            return None, None, None, {}
        embed_dir = legacy_dir

    meta: Dict[str, Any] = {}
    meta_path = embed_dir / 'meta.json'
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            meta = {}

    try:
        index = faiss.read_index(str(embed_dir / 'index.faiss'))

        with open(embed_dir / 'paths.pkl', 'rb') as f:
            image_paths = pickle.load(f)

        image_metadata = None
        metadata_file = embed_dir / 'metadata.pkl'
        if metadata_file.exists():
            try:
                with open(metadata_file, 'rb') as f:
                    image_metadata = pickle.load(f)
            except Exception:
                image_metadata = None

        return index, image_paths, image_metadata, meta
    except Exception as exc:
        print(f"Error loading index for {target}: {exc}")
        return None, None, None, meta


def save_segment_index(
    folder_path: Union[str, Path],
    embeddings: np.ndarray,
    segment_metadata: List[Dict[str, Any]],
) -> None:
    segment_dir = _segment_index_directory(folder_path)
    segment_dir.mkdir(parents=True, exist_ok=True)

    index_path = segment_dir / 'segment_index.faiss'
    metadata_path = segment_dir / 'segments.pkl'
    info_path = segment_dir / 'meta.json'

    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError("Segment embeddings must be a 2D array")

    if index_path.exists():
        index = faiss.read_index(str(index_path))
        if index.d != embeddings.shape[1]:
            raise ValueError(
                f"Segment embedding dimension mismatch: existing index expects {index.d}, got {embeddings.shape[1]}"
            )
        with open(metadata_path, 'rb') as fh:
            existing_meta: List[Dict[str, Any]] = pickle.load(fh)
    else:
        index = faiss.IndexFlatIP(embeddings.shape[1])
        existing_meta = []

    index.add(embeddings)
    existing_meta.extend(segment_metadata)

    faiss.write_index(index, str(index_path))
    with open(metadata_path, 'wb') as fh:
        pickle.dump(existing_meta, fh)

    info = {
        'embedder': 'dino',
        'type': 'segments',
        'embedding_dim': embeddings.shape[1],
        'segment_count': len(existing_meta),
        'updated_at': time.time(),
    }
    info_path.write_text(json.dumps(info, indent=2), encoding='utf-8')


def load_segment_index(folder_path: Union[str, Path]):
    segment_dir = _segment_index_directory(folder_path)
    index_path = segment_dir / 'segment_index.faiss'
    metadata_path = segment_dir / 'segments.pkl'
    info_path = segment_dir / 'meta.json'

    if not index_path.exists() or not metadata_path.exists():
        return None, [], {}

    try:
        index = faiss.read_index(str(index_path))
        with open(metadata_path, 'rb') as fh:
            segment_meta: List[Dict[str, Any]] = pickle.load(fh)
        meta_info = {}
        if info_path.exists():
            try:
                meta_info = json.loads(info_path.read_text(encoding='utf-8'))
            except json.JSONDecodeError:
                meta_info = {}
        return index, segment_meta, meta_info
    except Exception as exc:
        print(f"Error loading segment index: {exc}")
        return None, [], {}


def load_comments(folder_path):
    """Load comments from JSON file."""
    index_path = Path(folder_path) / config.INDEX_FOLDER_NAME
    comments_file = index_path / 'comments.json'

    if not comments_file.exists():
        return {}

    try:
        return json.loads(comments_file.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {}


def save_comments(folder_path, comments_data):
    """Persist comments to JSON."""
    index_path = Path(folder_path) / config.INDEX_FOLDER_NAME
    index_path.mkdir(exist_ok=True)
    comments_file = index_path / 'comments.json'

    try:
        comments_file.write_text(json.dumps(comments_data, ensure_ascii=False, indent=2), encoding='utf-8')
        return True
    except Exception as exc:
        print(f"Error saving comments: {exc}")
        return False


def get_image_comments(folder_path, image_path):
    """Fetch list of comments for a particular image."""
    comments_data = load_comments(folder_path)
    return comments_data.get(image_path, [])


def add_image_comment(folder_path, image_path, comment):
    """Append a comment to the image's history."""
    comments_data = load_comments(folder_path)

    if image_path not in comments_data:
        comments_data[image_path] = []

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    comments_data[image_path].append(f"[{timestamp}] {comment}")

    return save_comments(folder_path, comments_data)


def _prepare_metadata_map(image_paths: Optional[List[str]], image_metadata: Optional[List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    meta_map: Dict[str, Dict[str, Any]] = {}
    if not image_paths:
        return meta_map
    for idx, path in enumerate(image_paths):
        info: Dict[str, Any] = {}
        if image_metadata and idx < len(image_metadata):
            meta = image_metadata[idx] or {}
            if isinstance(meta, dict):
                info = meta
        meta_map[path] = info
    return meta_map


def _build_result_entry(img_path: str, similarity: float, metadata: Optional[Dict[str, Any]] = None, extra: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    try:
        img = Image.open(img_path)
        img.thumbnail(config.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=config.THUMBNAIL_QUALITY)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        result = {
            'path': img_path,
            'filename': os.path.basename(img_path),
            'similarity': float(similarity),
            'thumbnail': img_base64,
            'metadata': metadata or {},
        }
        if extra:
            result.update(extra)
        return result
    except Exception as img_error:
        print(f"Error processing image {img_path}: {img_error}")
        return None


def _should_rerank(sort_by: str) -> bool:
    return config.RERANK_ENABLED and config.RERANK_TOP_K > 0 and sort_by != 'time'


def _candidate_pool_size(limit: int, total: int, sort_by: str) -> int:
    if total <= 0:
        return 0
    limit = max(0, min(limit, total))
    if limit == 0:
        return 0
    if _should_rerank(sort_by):
        return min(total, max(limit, config.RERANK_TOP_K))
    return limit


def _collect_candidates(indices: np.ndarray, similarities: np.ndarray, max_index: int) -> List[Tuple[int, float]]:
    seen: Set[int] = set()
    candidates: List[Tuple[int, float]] = []
    flat_indices = indices[0] if indices.ndim > 1 else indices
    flat_sims = similarities[0] if similarities.ndim > 1 else similarities
    for idx, sim in zip(flat_indices, flat_sims):
        idx_int = int(idx)
        if 0 <= idx_int < max_index and idx_int not in seen:
            seen.add(idx_int)
            candidates.append((idx_int, float(sim)))
    return candidates


def _reconstruct_vectors(index: faiss.Index, ids: Sequence[int]) -> Optional[np.ndarray]:
    vectors: List[np.ndarray] = []
    for idx in ids:
        try:
            vec = index.reconstruct(int(idx))
        except (AttributeError, RuntimeError, IndexError, TypeError):
            return None
        vectors.append(np.asarray(vec, dtype=np.float32))
    if not vectors:
        return None
    try:
        return np.stack(vectors, axis=0)
    except ValueError:
        return None


def _rerank_candidates(index: faiss.Index, query_vec: np.ndarray, candidates: List[Tuple[int, float]], sort_by: str) -> List[Tuple[int, float, float, bool]]:
    if not candidates:
        return []

    if not _should_rerank(sort_by):
        return [(idx, score, score, False) for idx, score in candidates]

    rerank_count = min(len(candidates), config.RERANK_TOP_K)
    pool_ids = [idx for idx, _ in candidates[:rerank_count]]
    reconstructed = _reconstruct_vectors(index, pool_ids)
    if reconstructed is None:
        return [(idx, score, score, False) for idx, score in candidates]

    pool_vectors = reconstructed.astype(np.float32, copy=False)
    pool_norms = np.linalg.norm(pool_vectors, axis=1, keepdims=True)
    pool_norms[pool_norms == 0] = 1.0
    pool_vectors = pool_vectors / pool_norms

    query = np.asarray(query_vec, dtype=np.float32)
    if query.ndim > 1:
        query = query.reshape(-1)
    q_norm = np.linalg.norm(query)
    if q_norm == 0.0:
        q_norm = 1.0
    query = query / q_norm

    rerank_scores = pool_vectors @ query
    reranked = [
        (pool_ids[i], float(rerank_scores[i]), candidates[i][1], True)
        for i in range(len(pool_ids))
    ]
    reranked.sort(key=lambda item: item[1], reverse=True)

    remaining = [
        (idx, score, score, False) for idx, score in candidates[len(pool_ids):]
    ]
    return reranked + remaining


def _build_ranked_results(
    index: faiss.Index,
    query_vec: np.ndarray,
    indices: np.ndarray,
    similarities: np.ndarray,
    image_paths: Sequence[str],
    metadata_map: Dict[str, Dict[str, Any]],
    limit: int,
    sort_by: str,
) -> List[Dict[str, Any]]:
    candidates = _collect_candidates(indices, similarities, len(image_paths))
    baseline = [(idx, score, score, False) for idx, score in candidates]
    ranked = _rerank_candidates(index, query_vec, candidates, sort_by)

    if not ranked and baseline:
        ranked = baseline

    if ranked and len(ranked) != len(baseline) and baseline:
        ranked = baseline

    if not ranked:
        return []

    results: List[Dict[str, Any]] = []
    max_results = min(limit, len(ranked)) if limit > 0 else len(ranked)

    for idx, score, original_score, applied in ranked[:max_results]:
        path = image_paths[idx]
        extra = None
        if config.RERANK_ENABLED:
            extra = {
                'rerank': {
                    'applied': applied,
                    'score': float(score),
                    'original_score': float(original_score),
                }
            }
        entry = _build_result_entry(path, float(score), metadata_map.get(path, {}), extra=extra)
        if entry:
            results.append(entry)

    return results


def _load_fusion_indexes(folder_path: Union[str, Path]):
    clip_data = load_index(folder_path, embedder='clip')
    dino_data = load_index(folder_path, embedder='dino')
    return clip_data, dino_data


def _merge_metadata_maps(primary: Dict[str, Dict[str, Any]], secondary: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    merged = {path: dict(meta) for path, meta in primary.items()}
    for path, meta in secondary.items():
        if path not in merged:
            merged[path] = dict(meta)
        else:
            for key, value in meta.items():
                merged[path].setdefault(key, value)
    return merged


def _fuse_results(clip_data, dino_data, clip_vec: np.ndarray, dino_vec: np.ndarray, limit: int, sort_by: str) -> List[Dict[str, Any]]:
    clip_index, clip_paths, clip_metadata, _ = clip_data
    dino_index, dino_paths, dino_metadata, _ = dino_data
    if clip_index is None or dino_index is None:
        return []

    clip_map = _prepare_metadata_map(clip_paths, clip_metadata)
    dino_map = _prepare_metadata_map(dino_paths, dino_metadata)
    metadata_map = _merge_metadata_maps(clip_map, dino_map)

    limit = max(1, limit)
    alpha = max(0.0, min(1.0, float(config.FUSION_ALPHA)))

    def _search(index, vec, paths):
        if index is None or not paths:
            return {}
        k = min(limit * 2, len(paths))
        sims, inds = index.search(vec.reshape(1, -1), k)
        scores = {}
        for idx, sim in zip(inds[0], sims[0]):
            if 0 <= idx < len(paths):
                scores[paths[idx]] = float(sim)
        return scores

    clip_scores = _search(clip_index, clip_vec, clip_paths)
    dino_scores = _search(dino_index, dino_vec, dino_paths)

    combined: List[Tuple[str, float, float, float]] = []
    for path in set(list(clip_scores.keys()) + list(dino_scores.keys())):
        clip_sim = clip_scores.get(path, 0.0)
        dino_sim = dino_scores.get(path, 0.0)
        combined_score = (1.0 - alpha) * clip_sim + alpha * dino_sim
        combined.append((path, combined_score, clip_sim, dino_sim))

    if sort_by == 'time':
        combined.sort(key=lambda item: metadata_map.get(item[0], {}).get('mtime', 0), reverse=True)
    else:
        combined.sort(key=lambda item: item[1], reverse=True)

    results: List[Dict[str, Any]] = []
    for path, score, clip_sim, dino_sim in combined:
        entry = _build_result_entry(
            path,
            score,
            metadata_map.get(path, {}),
            extra={
                'fusion': {
                    'clip_similarity': clip_sim,
                    'dino_similarity': dino_sim,
                    'alpha': alpha,
                }
            },
        )
        if entry:
            results.append(entry)
        if len(results) >= limit:
            break
    return results


def _build_segment_search_results(
    segment_index: Optional[faiss.Index],
    segment_metadata: Sequence[Dict[str, Any]],
    query_vec: np.ndarray,
    limit: int,
) -> List[Dict[str, Any]]:
    if segment_index is None or not segment_metadata:
        return []

    if limit <= 0:
        return []

    k = min(limit, len(segment_metadata))
    if k == 0:
        return []

    similarities, indices = segment_index.search(query_vec.reshape(1, -1), k)

    results: List[Dict[str, Any]] = []
    for idx, sim in zip(indices[0], similarities[0]):
        if not (0 <= idx < len(segment_metadata)):
            continue
        seg_meta = segment_metadata[idx] or {}
        image_path = seg_meta.get('image_path')
        if not image_path:
            continue
        metadata = {
            'segment_id': seg_meta.get('segment_id'),
            'segment_label': seg_meta.get('label'),
            'segment_area': seg_meta.get('area'),
            'segment_fraction': seg_meta.get('patch_fraction'),
        }
        entry = _build_result_entry(image_path, float(sim), metadata)
        if entry is None:
            continue
        entry['segment'] = {
            'id': seg_meta.get('segment_id'),
            'label': seg_meta.get('label'),
            'source': seg_meta.get('source'),
            'area': seg_meta.get('area'),
            'patch_fraction': seg_meta.get('patch_fraction'),
        }
        results.append(entry)
    return results


def _parse_targets(raw_value: Optional[Union[str, Sequence[str]]]) -> Set[str]:
    acceptable = {'images', 'segments'}
    if raw_value is None:
        return {'images'}
    if isinstance(raw_value, (list, tuple, set)):
        tokens = {str(item).lower() for item in raw_value}
    else:
        tokens = {str(raw_value).lower()}
    targets = {token for token in tokens if token in acceptable}
    return targets or {'images'}


def _mask_search_pipeline(
    image_input: Union[str, Path, Image.Image],
    mask_image: Image.Image,
    folder: Union[str, Path],
    limit: int,
    sort_by: str,
    target_modes: Set[str],
    segment_ids: Optional[List[str]] = None,
    label_map: Optional[Dict[str, Any]] = None,
):
    ensure_embedder_loaded('dino')
    if dino_encoder is None:
        raise RuntimeError('DINO encoder is not available')

    segments = dino_encoder.encode_masked(
        image_input,
        mask_image,
        segment_ids=segment_ids,
        min_patches=config.DINO_SEGMENT_MIN_PATCHES,
    )

    if not segments:
        return [], segments

    image_index, image_paths, image_metadata, _ = load_index(folder, embedder='dino')
    if image_index is None or not image_paths:
        raise RuntimeError('DINO index not available for the requested folder')

    metadata_map = _prepare_metadata_map(image_paths, image_metadata)
    segment_index, segment_metadata, _ = load_segment_index(folder)

    label_map = label_map or {}

    segments_response: List[Dict[str, Any]] = []
    for seg_id, info in segments.items():
        if seg_id == 'full':
            if segment_ids is not None and 'full' not in segment_ids:
                continue
            if segment_ids is None and any(key != 'full' for key in segments.keys()):
                continue

        embedding = np.asarray(info.get('embedding'))
        if embedding.size == 0:
            continue

        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm == 0.0:
            continue
        embedding = embedding / norm

        entry: Dict[str, Any] = {
            'segment_id': seg_id,
            'label': label_map.get(str(seg_id)) or label_map.get(seg_id),
            'patch_count': int(info.get('patch_count', 0)),
            'patch_fraction': float(info.get('patch_fraction', 0.0)),
        }

        if 'images' in target_modes:
            k = _candidate_pool_size(limit, len(image_paths), sort_by)
            similarities, indices = image_index.search(embedding.reshape(1, -1), k)
            entry['image_results'] = _build_ranked_results(
                image_index,
                embedding,
                indices,
                similarities,
                image_paths,
                metadata_map,
                limit,
                sort_by,
            )

        if 'segments' in target_modes:
            entry['segment_results'] = _build_segment_search_results(
                segment_index,
                segment_metadata,
                embedding,
                limit,
            )

        segments_response.append(entry)

    return segments_response, segments


@app.route('/comments', methods=['GET'])
def get_comments():
    """Get comments for a specific image"""
    folder = request.args.get('folder')
    image_path = request.args.get('image_path')
    
    if not folder or not image_path:
        return jsonify({'error': 'Missing folder or image_path parameter'}), 400
    
    try:
        comments = get_image_comments(folder, image_path)
        return jsonify({'comments': comments})
    except Exception as e:
        print(f"Error getting comments: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/comments', methods=['POST'])
def save_comment():
    """Save a new comment for an image"""
    data = request.json
    folder = data.get('folder')
    image_path = data.get('image_path')
    comment = data.get('comment', '').strip()
    
    if not folder or not image_path or not comment:
        return jsonify({'error': 'Missing folder, image_path, or comment'}), 400
    
    # Basic input sanitization
    if len(comment) > config.MAX_COMMENT_LENGTH:
        return jsonify({'error': f'Comment too long (max {config.MAX_COMMENT_LENGTH} characters)'}), 400
    
    try:
        success = add_image_comment(folder, image_path, comment)
        if success:
            comments = get_image_comments(folder, image_path)
            return jsonify({'success': True, 'comments': comments})
        else:
            return jsonify({'error': 'Failed to save comment'}), 500
    except Exception as e:
        print(f"Error saving comment: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/commented_images', methods=['POST'])
def get_commented_images():
    """Get all images that have comments in the indexed folder"""
    folder = request.json.get('folder')
    if not folder:
        return jsonify({'error': 'No folder specified'}), 400
    
    try:
        # Load index to get image paths
        index, image_paths, image_metadata, index_meta = load_index(folder, embedder=active_embedder)
        if index is None:
            message = 'Folder not indexed for the current backend'
            available = _available_indexes(folder)
            if available:
                message += f" (available: {', '.join(available)})"
            return jsonify({'error': message}), 400
        
        # Load comments
        comments_data = load_comments(folder)
        
        # Build results for images with comments
        results = []
        for image_path in comments_data.keys():
            if image_path in image_paths:
                try:
                    # Get index position for metadata lookup
                    idx = image_paths.index(image_path)
                    
                    # Create thumbnail
                    img = Image.open(image_path)
                    img.thumbnail(config.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                    
                    # Convert to base64
                    buffer = BytesIO()
                    img.save(buffer, format='JPEG', quality=config.THUMBNAIL_QUALITY)
                    img_base64 = base64.b64encode(buffer.getvalue()).decode()
                    
                    # Get metadata if available
                    metadata_info = {}
                    if image_metadata and idx < len(image_metadata):
                        meta = image_metadata[idx]
                        metadata_info = {
                            'mtime': meta.get('mtime', 0),
                            'size': meta.get('size', 0)
                        }
                    
                    results.append({
                        'path': image_path,
                        'filename': os.path.basename(image_path),
                        'thumbnail': img_base64,
                        'comment_count': len(comments_data[image_path]),
                        'latest_comment': comments_data[image_path][-1] if comments_data[image_path] else '',
                        'metadata': metadata_info
                    })
                except Exception as img_error:
                    print(f"Error processing commented image {image_path}: {img_error}")
                    continue
        
        # Sort by most recent comment first
        results.sort(key=lambda x: x['latest_comment'], reverse=True)
        
        return jsonify({'results': results})
    except Exception as e:
        print(f"Error getting commented images: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/check_index', methods=['POST'])
def check_index():
    """Check if folder is indexed"""
    folder = request.json.get('folder')
    if not folder:
        return jsonify({'error': 'No folder specified'}), 400
    
    available = _available_indexes(folder)
    if active_embedder == 'fusion':
        indexed = 'fusion' in available or ({'clip', 'dino'}.issubset(set(available)))
    else:
        indexed = active_embedder in available
    response = {
        'indexed': indexed,
        'available_modes': available,
    }
    if not response['indexed'] and available:
        response['existing_embedder'] = available
    return jsonify(response)

@app.route('/index', methods=['POST'])
def index_folder():
    """Index a folder"""
    folder = request.json.get('folder')
    if not folder or not os.path.exists(folder):
        return jsonify({'error': 'Invalid folder path'}), 400
    
    try:
        index_results = create_index(folder)
        if not index_results:
            return jsonify({'error': 'No images found in folder'}), 400

        save_index(index_results, folder)

        counts = {embedder: len(data[1]) for embedder, data in index_results.items()}
        metadata = {embedder: data[3] for embedder, data in index_results.items()}
        modes = list(index_results.keys())
        if config.FUSION_ENABLED and 'clip' in counts and 'dino' in counts:
            fusion_count = min(counts['clip'], counts['dino'])
            counts['fusion'] = fusion_count
            metadata['fusion'] = {'alpha': config.FUSION_ALPHA}
            modes.append('fusion')

        active_count = counts.get(active_embedder)
        if active_count is None and counts:
            active_count = next(iter(counts.values()))
        response = {
            'success': True,
            'counts': counts,
            'meta': metadata,
            'count': active_count or 0,
            'active_meta': metadata.get(active_embedder),
            'modes': modes,
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _parse_segment_ids(raw_value: Optional[Union[str, List[str], List[int]]]) -> Optional[List[str]]:
    if raw_value is None:
        return None
    if isinstance(raw_value, list):
        return [str(item) for item in raw_value]
    text = str(raw_value).strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return [part.strip() for part in text.split(',') if part.strip()]


def _parse_segment_labels(raw_value: Optional[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
    if raw_value is None:
        return {}
    if isinstance(raw_value, dict):
        return {str(k): v for k, v in raw_value.items()}
    try:
        parsed = json.loads(raw_value)
        if isinstance(parsed, dict):
            return {str(k): v for k, v in parsed.items()}
    except json.JSONDecodeError:
        pass
    return {}


def _load_mask_from_request() -> Optional[Image.Image]:
    mask_file = request.files.get('mask')
    if mask_file:
        return Image.open(mask_file.stream).convert('L')
    mask_base64 = request.form.get('mask') or (request.json or {}).get('mask') if request.is_json else None
    if mask_base64:
        try:
            mask_bytes = base64.b64decode(mask_base64)
            return Image.open(BytesIO(mask_bytes)).convert('L')
        except Exception as exc:
            raise ValueError(f"Invalid mask payload: {exc}")
    return None


@app.route('/index_segments', methods=['POST'])
def index_segments():
    """Index DINO segment embeddings derived from a mask."""
    if not config.DINO_SEGMENTS_ENABLED:
        return jsonify({'error': 'Segment indexing is disabled. Enable EVOSSEARCH_DINO_SEGMENTS_ENABLED to use this feature.'}), 400

    data = request.form if request.form else (request.json or {})

    folder = data.get('folder')
    image_path = data.get('image_path')
    if not folder or not image_path:
        return jsonify({'error': 'Both folder and image_path are required'}), 400

    if not Path(image_path).exists():
        return jsonify({'error': f'Image file not found: {image_path}'}), 400

    mask_image = _load_mask_from_request()
    if mask_image is None:
        return jsonify({'error': 'Mask is required for segment indexing'}), 400

    segment_ids = _parse_segment_ids(data.get('segment_ids'))
    label_map = _parse_segment_labels(data.get('segment_labels'))

    ensure_embedder_loaded('dino')
    segments = dino_encoder.encode_masked(
        image_path,
        mask_image,
        segment_ids=segment_ids,
        min_patches=config.DINO_SEGMENT_MIN_PATCHES,
    )

    if not segments:
        return jsonify({'error': 'No valid segments were produced from the provided mask'}), 400

    entries: List[Dict[str, Any]] = []
    embeddings: List[np.ndarray] = []
    for seg_id, info in segments.items():
        if seg_id == 'full':
            continue
        embedding = info.get('embedding')
        if embedding is None:
            continue
        embeddings.append(np.asarray(embedding, dtype=np.float32))
        entries.append(
            {
                'image_path': image_path,
                'segment_id': seg_id,
                'label': label_map.get(seg_id),
                'area': int(info.get('patch_count', 0)),
                'patch_fraction': float(info.get('patch_fraction', 0.0)),
                'source': 'index_segments',
                'created_at': time.time(),
            }
        )

    if not embeddings:
        return jsonify({'error': 'Mask did not yield any segments beyond the full image aggregate'}), 400

    embedding_matrix = np.stack(embeddings, axis=0)
    save_segment_index(folder, embedding_matrix, entries)

    return jsonify(
        {
            'success': True,
            'segments_indexed': [
                {
                    'segment_id': entry['segment_id'],
                    'label': entry.get('label'),
                    'patch_count': entry.get('area'),
                    'patch_fraction': entry.get('patch_fraction'),
                }
                for entry in entries
            ],
        }
    )


@app.route('/video_understanding', methods=['POST'])
def video_understanding():
    data = request.json or {}
    video_path = (data.get('video') or '').strip()
    if not video_path:
        return jsonify({'error': 'Provide a video path.'}), 400
    max_frames = data.get('frame_count') or config.LM_VIDEO_DEFAULT_FRAMES
    try:
        max_frames_int = int(max_frames)
    except (TypeError, ValueError):
        max_frames_int = config.LM_VIDEO_DEFAULT_FRAMES
    if max_frames_int < 1:
        max_frames_int = 1
    max_frames_int = min(max_frames_int, config.LM_VIDEO_MAX_FRAMES)

    sample_fps_raw = data.get('sample_fps')
    try:
        sample_fps_val = float(sample_fps_raw) if sample_fps_raw is not None else None
        if sample_fps_val is not None and sample_fps_val <= 0:
            sample_fps_val = None
    except (TypeError, ValueError):
        sample_fps_val = None

    user_prompt = data.get('prompt') or ''

    try:
        frames, fps, duration = _sample_video_frames(
            video_path,
            max_frames=max_frames_int,
            sample_fps=sample_fps_val,
            max_edge=config.LM_VIDEO_MAX_EDGE,
        )
        if not frames:
            return jsonify({'error': 'No frames could be extracted from the video.'}), 400
        messages = _build_video_messages(video_path, frames, user_prompt)
        summary = _call_video_understanding(messages)
        return jsonify(
            {
                'summary': summary,
                'frames': [
                    {
                        'index': f['index'],
                        'time_sec': f['time_sec'],
                        'thumbnail': f['thumbnail'],
                    }
                    for f in frames
                ],
                'fps': fps,
                'duration_sec': duration,
                'model': config.LM_MODEL,
            }
        )
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/describe_image', methods=['POST'])
def describe_image():
    data = request.json or {}
    image_path = (data.get('image_path') or '').strip()
    prompt = data.get('prompt') or ''
    if not image_path:
        return jsonify({'error': 'image_path is required'}), 400
    path_obj = Path(image_path)
    if not path_obj.exists():
        return jsonify({'error': f'Image not found: {image_path}'}), 400
    try:
        messages = _build_image_messages(image_path, prompt)
        summary = _call_lm_chat(messages)
        thumb = _encode_jpeg(Image.open(path_obj), max_edge=config.THUMBNAIL_SIZE[0])
        return jsonify(
            {
                'summary': summary,
                'thumbnail': thumb,
                'model': config.LM_MODEL,
                'image_path': image_path,
            }
        )
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
@app.route('/search', methods=['POST'])
def search():
    """Search for images using text queries."""
    folder = request.json.get('folder')
    query = request.json.get('query')
    limit = request.json.get('limit', 10)
    sort_by = request.json.get('sort_by', 'similarity')  # 'similarity' or 'time'
    print(f"Search request: folder={folder}, query={query}, limit={limit}, sort_by={sort_by}")

    if not folder or not query:
        return jsonify({'error': 'Missing folder or query'}), 400

    fusion_active = active_embedder == 'fusion' and config.FUSION_ENABLED
    search_mode = 'clip' if fusion_active else active_embedder

    if search_mode != 'clip':
        return jsonify({'error': 'Text search is only available when using the CLIP backend.'}), 400

    try:
        limit = int(limit)
        if limit < config.MIN_RESULTS or limit > config.MAX_RESULTS:
            limit = config.DEFAULT_RESULTS
    except (ValueError, TypeError):
        limit = config.DEFAULT_RESULTS

    index, image_paths, image_metadata, index_meta = load_index(folder, embedder=search_mode)
    if index is None:
        message = 'Folder not indexed for the current backend'
        available = _available_indexes(folder)
        if available:
            message += f" (available: {', '.join(available)})"
        return jsonify({'error': message}), 400

    try:
        text_embedding = get_text_embedding(query)
    except RuntimeError as err:
        return jsonify({'error': str(err)}), 400

    try:
        k = _candidate_pool_size(limit, len(image_paths), sort_by)
        if k == 0:
            return jsonify({'results': []})
        similarities, indices = index.search(text_embedding.reshape(1, -1), k)

        metadata_map = _prepare_metadata_map(image_paths, image_metadata)
        results = _build_ranked_results(
            index,
            text_embedding,
            indices,
            similarities,
            image_paths,
            metadata_map,
            limit,
            sort_by,
        )

        if sort_by == 'time':
            results.sort(key=lambda item: item['metadata'].get('mtime', 0), reverse=True)

        return jsonify({'results': results})
    except Exception as e:
        print(f"Text search error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/search_by_image', methods=['POST'])
def search_by_image():
    """Search for images using an uploaded image"""
    folder = request.form.get('folder')
    limit = request.form.get('limit', 12)
    sort_by = request.form.get('sort_by', 'similarity')  # 'similarity' or 'time'

    if not folder:
        return jsonify({'error': 'Missing folder'}), 400

    try:
        limit = int(limit)
        if limit < config.MIN_RESULTS or limit > config.MAX_RESULTS:
            limit = config.DEFAULT_RESULTS
    except (ValueError, TypeError):
        limit = config.DEFAULT_RESULTS

    file = request.files.get('image')
    image_path = request.form.get('image_path')

    if not file and not image_path:
        return jsonify({'error': 'No image uploaded or path provided'}), 400

    if file and file.filename == '':
        file = None

    fusion_active = active_embedder == 'fusion' and config.FUSION_ENABLED

    if fusion_active:
        clip_data, dino_data = _load_fusion_indexes(folder)
        clip_index, clip_paths, clip_metadata, _ = clip_data
        dino_index, dino_paths, dino_metadata, _ = dino_data
        if clip_index is None or dino_index is None:
            message = 'Fusion requires both CLIP and DINO indexes'
            available = _available_indexes(folder)
            if available:
                message += f" (available: {', '.join(available)})"
            return jsonify({'error': message}), 400
    else:
        index, image_paths, image_metadata, index_meta = load_index(folder, embedder=active_embedder)
        if index is None:
            message = 'Folder not indexed for the current backend'
            available = _available_indexes(folder)
            if available:
                message += f" (available: {', '.join(available)})"
            return jsonify({'error': message}), 400

    try:
        if file:
            uploaded_image = Image.open(file.stream)
            if uploaded_image.mode != 'RGB':
                uploaded_image = uploaded_image.convert('RGB')
            clip_vec = get_image_embedding_from_pil(uploaded_image, embedder='clip') if fusion_active else get_image_embedding_from_pil(uploaded_image, embedder=active_embedder)
            if fusion_active:
                dino_vec = get_image_embedding_from_pil(uploaded_image, embedder='dino')
        else:
            if not os.path.exists(image_path):
                return jsonify({'error': f'Image file not found: {image_path}'}), 400
            clip_vec = get_image_embedding(image_path, embedder='clip') if fusion_active else get_image_embedding(image_path, embedder=active_embedder)
            if fusion_active:
                dino_vec = get_image_embedding(image_path, embedder='dino')

        if fusion_active:
            results = _fuse_results(clip_data, dino_data, clip_vec, dino_vec, limit, sort_by)
            return jsonify({'results': results})

        k = _candidate_pool_size(limit, len(image_paths), sort_by)
        if k == 0:
            return jsonify({'results': []})
        similarities, indices = index.search(clip_vec.reshape(1, -1), k)

        metadata_map = _prepare_metadata_map(image_paths, image_metadata)
        results = _build_ranked_results(
            index,
            clip_vec,
            indices,
            similarities,
            image_paths,
            metadata_map,
            limit,
            sort_by,
        )

        if sort_by == 'time':
            results.sort(key=lambda item: item['metadata'].get('mtime', 0), reverse=True)

        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/search_by_mask', methods=['POST'])
def search_by_mask():
    """Search using a masked region of an image leveraging DINO segment embeddings."""
    data = request.form if request.form else (request.json or {})

    if not config.DINO_SEGMENTS_ENABLED:
        return jsonify({'error': 'Segment search is disabled. Enable EVOSSEARCH_DINO_SEGMENTS_ENABLED to use this feature.'}), 400

    folder = data.get('folder')
    if not folder:
        return jsonify({'error': 'Missing folder'}), 400

    limit = data.get('limit', config.DEFAULT_RESULTS)
    sort_by = data.get('sort_by', 'similarity')
    target_modes_raw = data.get('targets') or data.get('target')
    segment_ids = _parse_segment_ids(data.get('segment_ids'))
    label_map = _parse_segment_labels(data.get('segment_labels'))

    try:
        limit = int(limit)
    except (ValueError, TypeError):
        limit = config.DEFAULT_RESULTS

    try:
        limit = max(config.MIN_RESULTS, min(limit, config.MAX_RESULTS))
    except Exception:
        limit = config.DEFAULT_RESULTS

    target_modes = _parse_targets(target_modes_raw)

    image_source = data.get('image_path')
    uploaded_image = request.files.get('image')

    if not image_source and uploaded_image is None:
        return jsonify({'error': 'Provide image_path or upload an image file'}), 400

    mask_image = _load_mask_from_request()
    if mask_image is None:
        return jsonify({'error': 'Mask is required for masked search'}), 400

    ensure_embedder_loaded('dino')

    if uploaded_image:
        query_image = Image.open(uploaded_image.stream)
        if query_image.mode != 'RGB':
            query_image = query_image.convert('RGB')
        image_input: Union[Image.Image, str] = query_image
    else:
        if not Path(image_source).exists():
            return jsonify({'error': f'Image file not found: {image_source}'}), 400
        image_input = image_source

    try:
        segment_map = dino_encoder.encode_masked(
            image_input,
            mask_image,
            segment_ids=segment_ids,
            min_patches=config.DINO_SEGMENT_MIN_PATCHES,
        )
    except Exception as exc:
        return jsonify({'error': f'Failed to compute masked embeddings: {exc}'}), 500

    if not segment_map:
        return jsonify({'error': 'Mask did not produce any valid segments'}), 400

    try:
        segments_response, _ = _mask_search_pipeline(
            image_input,
            mask_image,
            folder,
            limit,
            sort_by,
            target_modes,
            segment_ids,
            label_map,
        )
    except RuntimeError as exc:
        return jsonify({'error': str(exc)}), 400

    if not segments_response:
        return jsonify({'error': 'No segment results produced'}), 400

    return jsonify(
        {
            'segments': segments_response,
            'targets': list(target_modes),
            'segment_count': len(segments_response),
        }
    )


@app.route('/segment_from_point', methods=['POST'])
def segment_from_point():
    """Derive a mask from a clicked point and run masked search."""
    data = request.json if request.is_json else request.form
    if data is None:
        return jsonify({'error': 'No data provided'}), 400

    folder = data.get('folder')
    if not folder:
        return jsonify({'error': 'Missing folder'}), 400

    image_path = data.get('image_path')
    uploaded_image = request.files.get('image')
    if not image_path and uploaded_image is None:
        return jsonify({'error': 'Provide image_path or upload an image file'}), 400

    try:
        x_norm = float(data.get('x'))
        y_norm = float(data.get('y'))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid or missing x/y coordinates'}), 400

    try:
        threshold = float(data.get('threshold', config.DINO_HEATMAP_THRESHOLD))
    except (TypeError, ValueError):
        threshold = config.DINO_HEATMAP_THRESHOLD
    threshold = min(max(threshold, 0.0), 0.99)

    limit = data.get('limit', config.DEFAULT_RESULTS)
    sort_by = data.get('sort_by', 'similarity')
    target_modes = _parse_targets(data.get('targets') or data.get('target'))
    segment_ids = _parse_segment_ids(data.get('segment_ids'))
    label_map = _parse_segment_labels(data.get('segment_labels'))
    if not isinstance(label_map, dict):
        label_map = {}

    try:
        limit = int(limit)
    except (ValueError, TypeError):
        limit = config.DEFAULT_RESULTS
    try:
        limit = max(config.MIN_RESULTS, min(limit, config.MAX_RESULTS))
    except Exception:
        limit = config.DEFAULT_RESULTS

    ensure_embedder_loaded('dino')
    if dino_encoder is None:
        return jsonify({'error': 'DINO encoder is not available'}), 500

    pil_image: Optional[Image.Image] = None
    if uploaded_image:
        pil_image = Image.open(uploaded_image.stream)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
    else:
        if not os.path.exists(image_path):
            return jsonify({'error': f'Image file not found: {image_path}'}), 400
        with Image.open(image_path) as src:
            pil_image = src.convert('RGB')

    assert pil_image is not None
    image_input: Union[str, Path, Image.Image] = pil_image

    try:
        heatmap, _, grid, patch_coords = dino_encoder.patch_similarity_map(
            image_input,
            x_norm,
            y_norm,
        )
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500

    flat = heatmap.reshape(-1)
    quantile = np.quantile(flat, threshold) if flat.size else 0.0
    mask_bool = heatmap >= quantile
    if not mask_bool.any() and flat.size:
        mask_bool.flat[int(np.argmax(flat))] = True

    coarse_mask_uint8 = (mask_bool.astype(np.uint8) * 255)
    mask_fraction = float(np.count_nonzero(coarse_mask_uint8)) / float(coarse_mask_uint8.size) if coarse_mask_uint8.size else 0.0

    heatmap_norm = heatmap - heatmap.min()
    if heatmap_norm.max() > 0:
        heatmap_norm = heatmap_norm / heatmap_norm.max()

    crop_size = getattr(dino_encoder, 'crop_size', 224)
    mask_img = Image.fromarray(coarse_mask_uint8).resize((crop_size, crop_size), resample=Image.NEAREST)

    base_size = pil_image.size
    overlay_mask_source = Image.fromarray(coarse_mask_uint8).resize(base_size, resample=Image.NEAREST)
    refinement_source = 'dino_heatmap'
    refined_label: Optional[str] = None
    segment_value = 255
    refine_result: Optional[Dict[str, Any]] = None

    head = ensure_mask_head()
    if head is not None:
        try:
            refine_result = head.refine(pil_image, coarse_mask_uint8, (x_norm, y_norm))
            if refine_result and isinstance(refine_result, dict):
                refined_mask_arr = refine_result.get('mask')
                if isinstance(refined_mask_arr, np.ndarray) and refined_mask_arr.any():
                    refinement_source = 'mask2former'
                    refined_label = refine_result.get('label')
                    mask_fraction = float(refine_result.get('mask_fraction', mask_fraction))
                    segment_value = int(refine_result.get('segment_value', 255))
                    overlay_mask_source = Image.fromarray(refined_mask_arr.astype(np.uint8), mode='L')
                    if overlay_mask_source.size != base_size:
                        overlay_mask_source = overlay_mask_source.resize(base_size, resample=Image.NEAREST)
                    mask_img = overlay_mask_source.resize((crop_size, crop_size), resample=Image.NEAREST)
                    if refined_label:
                        label_map[str(segment_value)] = refined_label
        except Exception as exc:
            print(f"Mask2Former refinement error: {exc}")

    # Update mask coverage fraction based on overlay image size
    overlay_mask_np = np.asarray(overlay_mask_source)
    if overlay_mask_np.size:
        mask_fraction = float(np.count_nonzero(overlay_mask_np)) / float(overlay_mask_np.size)

    heatmap_alpha = Image.fromarray((heatmap_norm * 255).astype(np.uint8)).resize(base_size, resample=Image.BILINEAR)
    heatmap_overlay = _create_overlay_rgba(heatmap_alpha, (255, 155, 40), 0.9)
    mask_overlay = _create_overlay_rgba(overlay_mask_source, (94, 196, 255), 0.6)
    segmentation_overlay_img: Optional[Image.Image] = None
    legend_entries: List[Dict[str, Any]] = []

    if refine_result:
        seg_map = refine_result.get('segmentation')
        class_labels_raw = refine_result.get('class_labels')
        class_labels = {}
        if isinstance(class_labels_raw, dict):
            class_labels = {int(k): str(v) for k, v in class_labels_raw.items()}
        seg_overlay, legend_entries = _render_segmentation_overlay(seg_map, class_labels, int(refine_result.get('class_id', segment_value)))
        if seg_overlay is not None:
            if seg_overlay.size != base_size:
                seg_overlay = seg_overlay.resize(base_size, resample=Image.NEAREST)
            segmentation_overlay_img = seg_overlay

    try:
        segments_response, segment_map = _mask_search_pipeline(
            image_input,
            mask_img,
            folder,
            limit,
            sort_by,
            target_modes,
            segment_ids,
            label_map,
        )
    except RuntimeError as exc:
        return jsonify({'error': str(exc)}), 400

    if not segments_response:
        return jsonify({'error': 'No segment results produced'}), 400

    selected_key = next((key for key in segment_map.keys() if key != 'full'), 'full')
    selected_meta = segment_map.get(selected_key, {})

    overlay = {
        'grid_size': grid,
        'patch_coords': {'x': patch_coords[0], 'y': patch_coords[1]},
        'threshold': threshold,
        'heatmap_png': _image_to_base64(heatmap_overlay),
        'mask_png': _image_to_base64(mask_overlay),
        'mask_fraction': mask_fraction,
        'refinement': refinement_source,
        'refined_label': refined_label,
        'segment_value': segment_value,
        'patch_count': int(selected_meta.get('patch_count', 0)),
        'patch_fraction': float(selected_meta.get('patch_fraction', 0.0)),
    }
    if segmentation_overlay_img is not None:
        overlay['segmentation_png'] = _image_to_base64(segmentation_overlay_img)
    if legend_entries:
        overlay['legend'] = legend_entries

    return jsonify(
        {
            'segments': segments_response,
            'targets': list(target_modes),
            'segment_count': len(segments_response),
            'overlay': overlay,
        }
    )

@app.route('/settings', methods=['GET'])
def get_settings():
    """Get current configuration settings"""
    try:
        requested_embedder = config.EMBEDDER if config.EMBEDDER in SUPPORTED_EMBEDDERS else active_embedder
        settings = {
            'host': config.HOST,
            'port': config.PORT,
            'debug': config.DEBUG,
            'embedder': requested_embedder,
            'clipModel': config.CLIP_MODEL,
            'dinoModel': config.DINO_MODEL,
            'dinoEmbedDim': config.EMB_DIM_DINO,
            'dinoWeightsPath': config.DINO_WEIGHTS_PATH,
            'indexMode': config.INDEX_MODE,
            'fusionEnabled': config.FUSION_ENABLED,
            'fusionAlpha': config.FUSION_ALPHA,
            'rerankEnabled': config.RERANK_ENABLED,
            'rerankTopK': config.RERANK_TOP_K,
            'segmentsEnabled': config.DINO_SEGMENTS_ENABLED,
            'segmentMinPatches': config.DINO_SEGMENT_MIN_PATCHES,
            'segmentThreshold': config.DINO_HEATMAP_THRESHOLD,
            'minResults': config.MIN_RESULTS,
            'maxResults': config.MAX_RESULTS,
            'defaultResults': config.DEFAULT_RESULTS,
            'batchSize': config.BATCH_SIZE,
            'thumbnailQuality': config.THUMBNAIL_QUALITY,
            'maxCommentLength': config.MAX_COMMENT_LENGTH,
            'maxFileSize': config.MAX_FILE_SIZE_MB,
            'indexFolderName': config.INDEX_FOLDER_NAME
        }
        return jsonify({'success': True, 'settings': settings})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/settings', methods=['POST'])
def save_settings():
    """Save configuration settings to .env file"""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        global active_embedder, clip_model, clip_preprocess, dino_encoder

        required_fields = ['host', 'port', 'debug', 'clipModel', 'minResults', 'maxResults', 'defaultResults']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400

        try:
            port = int(data['port'])
            if not (1000 <= port <= 65535):
                return jsonify({'success': False, 'error': 'Port must be between 1000 and 65535'}), 400

            min_results = int(data['minResults'])
            max_results = int(data['maxResults'])
            default_results = int(data['defaultResults'])

            if not (1 <= min_results <= max_results):
                return jsonify({'success': False, 'error': 'Min results must be less than or equal to max results'}), 400

            if not (min_results <= default_results <= max_results):
                return jsonify({'success': False, 'error': 'Default results must be between min and max results'}), 400
        except ValueError as e:
            return jsonify({'success': False, 'error': f'Invalid number format: {str(e)}'}), 400

        fusion_enabled_raw = data.get('fusionEnabled', config.FUSION_ENABLED)
        if isinstance(fusion_enabled_raw, str):
            fusion_enabled = fusion_enabled_raw.strip().lower() in {'true', '1', 'yes', 'on'}
        else:
            fusion_enabled = bool(fusion_enabled_raw)

        try:
            fusion_alpha = float(data.get('fusionAlpha', config.FUSION_ALPHA))
        except (TypeError, ValueError):
            fusion_alpha = config.FUSION_ALPHA
        fusion_alpha = min(1.0, max(0.0, fusion_alpha))

        rerank_enabled_raw = data.get('rerankEnabled', config.RERANK_ENABLED)
        if isinstance(rerank_enabled_raw, str):
            rerank_enabled = rerank_enabled_raw.strip().lower() in {'true', '1', 'yes', 'on'}
        else:
            rerank_enabled = bool(rerank_enabled_raw)

        try:
            rerank_top_k = int(data.get('rerankTopK', config.RERANK_TOP_K))
        except (TypeError, ValueError):
            rerank_top_k = config.RERANK_TOP_K
        if rerank_top_k < 1:
            rerank_top_k = 1

        segments_enabled_raw = data.get('segmentsEnabled', config.DINO_SEGMENTS_ENABLED)
        if isinstance(segments_enabled_raw, str):
            segments_enabled = segments_enabled_raw.strip().lower() in {'true', '1', 'yes', 'on'}
        else:
            segments_enabled = bool(segments_enabled_raw)

        try:
            segment_min_patches = int(data.get('segmentMinPatches', config.DINO_SEGMENT_MIN_PATCHES))
        except (TypeError, ValueError):
            segment_min_patches = config.DINO_SEGMENT_MIN_PATCHES
        if segment_min_patches < 1:
            segment_min_patches = 1

        try:
            segment_threshold = float(data.get('segmentThreshold', config.DINO_HEATMAP_THRESHOLD))
        except (TypeError, ValueError):
            segment_threshold = config.DINO_HEATMAP_THRESHOLD
        segment_threshold = min(0.99, max(0.0, segment_threshold))

        embedder = str(data.get('embedder', active_embedder)).strip().lower()
        if embedder == 'fusion' and not fusion_enabled:
            embedder = 'clip'
        if embedder not in SUPPORTED_EMBEDDERS:
            embedder = 'clip'
        dino_model = str(data.get('dinoModel', config.DINO_MODEL)).strip() or config.DINO_MODEL
        try:
            dino_dim = int(data.get('dinoEmbedDim', config.EMB_DIM_DINO))
        except (TypeError, ValueError):
            dino_dim = config.EMB_DIM_DINO

        dino_weights_path = data.get('dinoWeightsPath', config.DINO_WEIGHTS_PATH) or ''
        dino_device = str(data.get('dinoDevice', config.DINO_DEVICE)).strip()

        index_mode = str(data.get('indexMode', config.INDEX_MODE)).strip().lower()
        if index_mode not in {'clip', 'dino', 'dual'}:
            index_mode = 'clip'

        batch_size = int(data.get('batchSize', config.BATCH_SIZE))
        thumbnail_quality = int(data.get('thumbnailQuality', config.THUMBNAIL_QUALITY))
        max_comment_length = int(data.get('maxCommentLength', config.MAX_COMMENT_LENGTH))
        max_file_size = int(data.get('maxFileSize', config.MAX_FILE_SIZE_MB))
        index_folder = data.get('indexFolderName', config.INDEX_FOLDER_NAME)

        env_content = f"""# evo-ssearch Configuration
# Generated by settings panel

# Server Configuration
EVOSSEARCH_HOST={data['host']}
EVOSSEARCH_PORT={port}
EVOSSEARCH_DEBUG={str(data['debug']).lower()}

# Embedder configuration
EVOSSEARCH_EMBEDDER={embedder}
EVOSSEARCH_CLIP_MODEL={data['clipModel']}
EVOSSEARCH_DINO_MODEL={dino_model}
EVOSSEARCH_EMB_DIM_DINO={dino_dim}
EVOSSEARCH_DINO_WEIGHTS_PATH={dino_weights_path}
EVOSSEARCH_DINO_DEVICE={dino_device}
EVOSSEARCH_INDEX_MODE={index_mode}
EVOSSEARCH_FUSION_ENABLED={str(fusion_enabled).lower()}
EVOSSEARCH_FUSION_ALPHA={fusion_alpha:.4f}
EVOSSEARCH_RERANK_ENABLED={str(rerank_enabled).lower()}
EVOSSEARCH_RERANK_TOP_K={rerank_top_k}
EVOSSEARCH_DINO_SEGMENTS_ENABLED={str(segments_enabled).lower()}
EVOSSEARCH_DINO_SEGMENT_MIN_PATCHES={segment_min_patches}
EVOSSEARCH_DINO_HEATMAP_THRESHOLD={segment_threshold:.4f}
EVOSSEARCH_M2F_ENABLED={str(config.MASK2FORMER_ENABLED).lower()}
EVOSSEARCH_M2F_MODEL={config.MASK2FORMER_MODEL}
EVOSSEARCH_M2F_DEVICE={config.MASK2FORMER_DEVICE}
EVOSSEARCH_M2F_MAX_SIZE={config.MASK2FORMER_MAX_SIZE}

# Search result limits
EVOSSEARCH_MIN_RESULTS={min_results}
EVOSSEARCH_MAX_RESULTS={max_results}
EVOSSEARCH_DEFAULT_RESULTS={default_results}

# Processing configuration
EVOSSEARCH_BATCH_SIZE={batch_size}
EVOSSEARCH_THUMBNAIL_QUALITY={thumbnail_quality}

# File system configuration
EVOSSEARCH_INDEX_FOLDER={index_folder}

# Comment system configuration
EVOSSEARCH_MAX_COMMENT_LENGTH={max_comment_length}

# Security configuration
EVOSSEARCH_MAX_FILE_SIZE_MB={max_file_size}
"""

        with open('.env', 'w', encoding='utf-8') as f:
            f.write(env_content)

        config.HOST = data['host']
        config.PORT = port
        config.DEBUG = bool(data['debug'])
        config.EMBEDDER = embedder
        config.CLIP_MODEL = data['clipModel']
        config.DINO_MODEL = dino_model
        config.EMB_DIM_DINO = dino_dim
        config.DINO_WEIGHTS_PATH = dino_weights_path
        config.DINO_DEVICE = dino_device
        config.INDEX_MODE = index_mode
        config.MIN_RESULTS = min_results
        config.MAX_RESULTS = max_results
        config.DEFAULT_RESULTS = default_results
        config.BATCH_SIZE = batch_size
        config.THUMBNAIL_QUALITY = thumbnail_quality
        config.MAX_COMMENT_LENGTH = max_comment_length
        config.MAX_FILE_SIZE_MB = max_file_size
        config.INDEX_FOLDER_NAME = index_folder
        config.FUSION_ENABLED = fusion_enabled
        config.FUSION_ALPHA = fusion_alpha
        config.RERANK_ENABLED = rerank_enabled
        config.RERANK_TOP_K = rerank_top_k
        config.DINO_SEGMENTS_ENABLED = segments_enabled
        config.DINO_SEGMENT_MIN_PATCHES = segment_min_patches
        config.DINO_HEATMAP_THRESHOLD = segment_threshold

        active_embedder = embedder
        if active_embedder == 'fusion' and not config.FUSION_ENABLED:
            active_embedder = 'clip'
        clip_model = None
        clip_preprocess = None
        dino_encoder = None
        ensure_embedder_loaded()

        return jsonify({'success': True, 'message': 'Settings saved successfully. Restart the server if issues persist.'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    ensure_embedder_loaded()
    config.print_startup_info()
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
