import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import atexit
import base64
import copy
import json
import math
import pickle
import secrets
import threading
import time
import uuid
import requests
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, cast
from urllib.parse import unquote
from threading import Lock

import numpy as np
import torch
import cv2
import clip
import faiss
from PIL import Image
from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS

from config import config
from embedders.dino_encoder import DINOEncoder
from luxriot_connector import LuxriotManager
from probe_manager import ProbeManager
if TYPE_CHECKING:
    from heads.mask2former_head import Mask2FormerHead
try:
    from heads.mask2former_head import Mask2FormerHead as _Mask2FormerHeadRuntime
except Exception:  # pragma: no cover - optional dependency
    _Mask2FormerHeadRuntime = None  # type: ignore[misc]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = max(1, int(config.MAX_FILE_SIZE_MB)) * 1024 * 1024
if config.CORS_ALLOWED_ORIGINS:
    CORS(app, resources={r"/*": {"origins": list(config.CORS_ALLOWED_ORIGINS)}})

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
FaissIndexBundle = Tuple[Optional[faiss.Index], Optional[List[str]], Optional[List[Dict[str, Any]]], Dict[str, Any]]

if hasattr(Image, "Resampling"):
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
else:  # pragma: no cover - Pillow compatibility fallback
    RESAMPLE_NEAREST = Image.NEAREST  # type: ignore[attr-defined]
    RESAMPLE_BILINEAR = Image.BILINEAR  # type: ignore[attr-defined]
    RESAMPLE_LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]
active_embedder = config.EMBEDDER if config.EMBEDDER in SUPPORTED_EMBEDDERS else "clip"
if active_embedder == "fusion" and not config.FUSION_ENABLED:
    active_embedder = "clip"

LOCAL_HOSTS = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}
TRUE_BOOL_STRINGS = {"1", "true", "yes", "on"}
FALSE_BOOL_STRINGS = {"0", "false", "no", "off"}


def _json_body() -> Dict[str, Any]:
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in TRUE_BOOL_STRINGS:
            return True
        if normalized in FALSE_BOOL_STRINGS:
            return False
        return default
    return bool(value)


def _is_local_request() -> bool:
    return (request.remote_addr or "") in LOCAL_HOSTS


def _request_admin_token() -> str:
    auth_header = (request.headers.get("Authorization") or "").strip()
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return (request.headers.get("X-Admin-Token") or "").strip()


def _has_admin_access() -> bool:
    configured = (config.ADMIN_TOKEN or "").strip()
    if not configured:
        return False
    provided = _request_admin_token()
    if not provided:
        return False
    return secrets.compare_digest(provided, configured)


def _settings_guard(write: bool = False):
    if write:
        return _mutation_guard()
    if config.ADMIN_TOKEN:
        if _has_admin_access():
            return None
        return jsonify({"success": False, "error": "Admin token required"}), 401
    if config.SETTINGS_LOCAL_ONLY and not _is_local_request():
        return jsonify({"success": False, "error": "Remote settings access is disabled."}), 403
    return None


def _mutation_guard():
    configured = (config.ADMIN_TOKEN or "").strip()
    if not configured:
        return jsonify(
            {
                "success": False,
                "error": "Admin token is required for mutating endpoints. Set EVOSSEARCH_ADMIN_TOKEN.",
            }
        ), 503
    if _has_admin_access():
        return None
    return jsonify({"success": False, "error": "Admin token required"}), 401


def _mutation_guard_error():
    guard = _mutation_guard()
    if guard is None:
        return None
    body, status = guard
    payload = body.get_json(silent=True) if hasattr(body, "get_json") else {}
    message = (payload or {}).get("error") or "Admin token required"
    return jsonify({"error": message}), status


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_folder_path(folder_raw: Any, require_index: bool = False) -> Path:
    folder_text = str(folder_raw or "").strip()
    if not folder_text:
        raise ValueError("No folder specified")
    folder_path = Path(folder_text).expanduser().resolve()
    if not folder_path.exists() or not folder_path.is_dir():
        raise ValueError("Invalid folder path")
    if config.ALLOWED_ROOTS:
        allowed_roots = [Path(item).expanduser().resolve() for item in config.ALLOWED_ROOTS]
        if not any(_path_within(folder_path, root) for root in allowed_roots):
            raise ValueError("Folder path is outside configured allowed roots")
    if require_index:
        index_root = folder_path / config.INDEX_FOLDER_NAME
        has_index = (index_root / "index.faiss").exists() or any(
            (index_root / subdir / "index.faiss").exists() for subdir in EMBEDDER_SUBDIRS.values()
        )
        if not has_index:
            raise ValueError("Folder is not indexed")
    return folder_path


@app.errorhandler(413)
def payload_too_large(_: Exception):
    return jsonify({"error": f"Payload too large (max {config.MAX_FILE_SIZE_MB} MB)."}), 413


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
    if _Mask2FormerHeadRuntime is None:
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
            created_head = _Mask2FormerHeadRuntime(
                model_name=config.MASK2FORMER_MODEL,
                device=target_device,
                max_size=config.MASK2FORMER_MAX_SIZE,
            )
            mask2former_head = cast(Any, created_head)
        except Exception as exc:
            _mask2former_failed = True
            print(f"Mask2Former head initialization failed: {exc}")
            config.MASK2FORMER_ENABLED = False
            return None
    return mask2former_head


def _faiss_add_vectors(index: faiss.Index, vectors: np.ndarray) -> None:
    """Typed wrapper for FAISS add; FAISS runtime monkey-patches signatures."""
    cast(Any, index).add(vectors)


def _faiss_search(index: faiss.Index, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Typed wrapper for FAISS search; returns (distances, indices)."""
    distances, labels = cast(Any, index).search(query, int(k))
    return np.asarray(distances), np.asarray(labels)


def _faiss_reconstruct(index: faiss.Index, idx: int) -> np.ndarray:
    """Typed wrapper for FAISS reconstruct."""
    return np.asarray(cast(Any, index).reconstruct(int(idx)), dtype=np.float32)


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
    _faiss_add_vectors(index, embeddings_array)
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
    luxriot_batch_options = []
    for size in config.LUXRIOT_BATCH_SIZES:
        selected = "selected" if size == config.LUXRIOT_BATCH_SIZES[0] else ""
        luxriot_batch_options.append(f'<option value="{size}" {selected}>{size}</option>')
    luxriot_batch_options_html = '\n                            '.join(luxriot_batch_options)
    luxriot_default_batch = config.LUXRIOT_BATCH_SIZES[0] if config.LUXRIOT_BATCH_SIZES else 12
    default_video_frames = max(1, int(config.LM_VIDEO_DEFAULT_FRAMES))
    max_video_frames = max(default_video_frames, int(config.LM_VIDEO_MAX_FRAMES))
    video_frame_options_set: Set[int] = {default_video_frames}
    for raw_option in getattr(config, "LM_VIDEO_FRAME_OPTIONS", ()):
        try:
            option = int(raw_option)
        except (TypeError, ValueError):
            continue
        if 1 <= option <= max_video_frames:
            video_frame_options_set.add(option)
    video_frame_options = sorted(video_frame_options_set)
    video_frame_options_html = '\n                            '.join(
        f'<option value="{count}" {"selected" if count == default_video_frames else ""}>{count}</option>'
        for count in video_frame_options
    )
    segments_enabled_checked = "checked" if bool(config.DINO_SEGMENTS_ENABLED) else ""
    segment_min_patches_default = max(1, int(config.DINO_SEGMENT_MIN_PATCHES))
    segment_threshold_percent = min(99, max(40, int(round(float(config.DINO_HEATMAP_THRESHOLD) * 100))))
    
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
        :root {
            --bg: #0a0a0a;
            --panel: #161616;
            --panel-border: #262626;
            --surface: #0f0f0f;
            --field: #0a0a0a;
            --field-border: #333;
            --text: #e0e0e0;
            --muted: #8f8f8f;
            --radius-md: 8px;
            --radius-lg: 12px;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        .container {
            width: 100%;
            max-width: 1280px;
            min-width: 0;
            margin: 0 auto;
            padding: 1.5rem;
            flex: 1;
        }
        
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
        }

        .header-actions {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.9rem;
        }

        .brand-title {
            display: flex;
            flex-direction: column;
            gap: 0.2rem;
        }

        .brand-main {
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            margin: 0;
        }

        .brand-sub {
            color: #d8d8d8;
            font-size: 1.05rem;
            margin: 0;
        }

        .brand-note {
            color: #9c9c9c;
            font-size: 0.9rem;
            font-style: italic;
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

        .probe-editor-modal-content {
            max-width: 1040px;
            width: min(96vw, 1040px);
            max-height: 88vh;
        }

        .probe-editor-modal-body {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .probe-editor-layout {
            display: grid;
            grid-template-columns: minmax(280px, 0.9fr) minmax(500px, 1.1fr);
            gap: 1rem;
            align-items: start;
        }

        .probe-editor-settings {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .probe-editor-modal-actions {
            margin-top: 0.25rem;
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
            background: var(--panel);
            border-radius: var(--radius-lg);
            padding: 1.5rem;
            margin-bottom: 2rem;
            border: 1px solid var(--panel-border);
            position: static;
        }
        
        .folder-select {
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        
        input[type="text"] {
            flex: 1;
            background: var(--field);
            border: 1px solid var(--field-border);
            padding: 0.75rem 1rem;
            border-radius: var(--radius-md);
            color: var(--text);
            font-size: 0.95rem;
            transition: border-color 0.2s;
            min-width: 0;
        }
        
        input[type="text"]:focus {
            outline: none;
            border-color: #555;
        }
        
        button {
            background: #1a1a1a;
            border: 1px solid var(--field-border);
            color: var(--text);
            padding: 0.75rem 1.5rem;
            border-radius: var(--radius-md);
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

        .status.warning {
            color: #f4c066;
        }
        
        .search-panel {
            background: var(--panel);
            border-radius: var(--radius-lg);
            padding: 1.5rem;
            margin-bottom: 2rem;
            border: 1px solid var(--panel-border);
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        
        .search-mode-tabs {
            display: flex;
            gap: 0;
            border-radius: var(--radius-md);
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
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
        }

        .archive-box {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .archive-workspace {
            display: grid;
            grid-template-columns: 320px minmax(0, 1fr);
            gap: 1rem;
            align-items: start;
        }

        .archive-results-panel {
            background: #101010;
            border: 1px solid #232323;
            border-radius: var(--radius-md);
            padding: 0.9rem;
            min-height: 280px;
        }

        .archive-results-head {
            color: #bdbdbd;
            font-size: 0.82rem;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            margin-bottom: 0.75rem;
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

        .feature-btn.primary {
            background: #3a6346;
            border-color: #4e7a5b;
            color: #eef8f1;
        }

        .feature-btn.primary:hover {
            background: #447454;
            border-color: #5d8e6e;
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
            align-items: flex-end;
        }

        .archive-search-shell {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            position: sticky;
            top: 1rem;
            align-self: start;
        }

        .archive-section {
            background: var(--surface);
            border: 1px solid #222;
            border-radius: var(--radius-md);
            padding: 0.9rem;
            display: flex;
            flex-direction: column;
            gap: 0.7rem;
        }

        .archive-section-title {
            color: #bdbdbd;
            font-size: 0.85rem;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }
        
        input[type="file"] {
            background: var(--field);
            border: 1px solid var(--field-border);
            padding: 0.75rem 1rem;
            border-radius: var(--radius-md);
            color: var(--text);
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
            min-width: 0;
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
        
        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 1.5rem;
            align-content: start;
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

        .segment-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 0.75rem;
        }

        .segment-action-btn {
            border: 1px solid #2e2e2e;
            background: #151515;
            color: #ddd;
            border-radius: 999px;
            padding: 0.34rem 0.72rem;
            font-size: 0.76rem;
            cursor: pointer;
            transition: border-color 0.2s ease, background 0.2s ease, color 0.2s ease;
        }

        .segment-action-btn:hover {
            border-color: #4a4a4a;
            background: #1c1c1c;
            color: #fff;
        }

        .segment-action-btn.primary {
            border-color: #3f4d6a;
            background: #1a2235;
            color: #d7e3ff;
        }

        .segment-action-btn.primary:hover {
            border-color: #5e73a5;
            background: #22304d;
            color: #ffffff;
        }

        .segment-action-btn:disabled {
            opacity: 0.65;
            cursor: default;
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

        .lm-description {
            margin-bottom: 0.75rem;
        }

        .lm-description-actions {
            display: flex;
            justify-content: flex-end;
            margin-bottom: 0.75rem;
        }

        .lm-comment {
            border-left-color: #3f6d54;
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

        .is-hidden {
            display: none;
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

        .result-actions {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            margin-top: 0.35rem;
        }

        .action-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: #1a1a1a;
            border: 1px solid #2d2d2d;
            border-radius: 4px;
            padding: 4px;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .action-icon:hover {
            background: #202020;
            border-color: #3a3a3a;
        }
        /* Video understanding */
        .video-box {
            display: none;
            flex-direction: column;
            gap: 1rem;
            background: #111;
            border: 1px solid #222;
            border-radius: 10px;
            padding: 1rem;
        }

        .monitor-box {
            display: none;
            background: #111;
            border: 1px solid #222;
            border-radius: 10px;
            padding: 1rem;
        }

        .video-analysis-grid {
            display: grid;
            grid-template-columns: minmax(300px, 0.95fr) minmax(380px, 1.05fr);
            gap: 1rem;
            margin-top: 0.25rem;
        }

        .video-analysis-form {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .video-analysis-output {
            background: #0d0d0d;
            border: 1px solid #1f1f1f;
            border-radius: 8px;
            padding: 0.75rem;
            min-height: 240px;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
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

        .video-prompt-note {
            color: #aaa;
            font-size: 0.85rem;
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
        }

        .video-status {
            font-size: 0.9rem;
            color: #ccc;
            min-height: 20px;
        }

        .video-status.error {
            color: #ff9080;
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

        .video-output-wrap,
        .video-frame-block {
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
        }

        .video-block-title {
            color: #bdbdbd;
            font-size: 0.82rem;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }

        .video-frame-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 0.5rem;
            max-height: 360px;
            overflow-y: auto;
            padding-right: 0.25rem;
        }

        .video-frame-grid img {
            width: 100%;
            height: 92px;
            object-fit: cover;
            border-radius: 5px;
            border: 1px solid #222;
            background: #111;
        }

        /* Luxriot live view */
        .luxriot-grid {
            display: grid;
            grid-template-columns: 1.3fr 1fr;
            gap: 0.75rem;
            margin-bottom: 0.75rem;
        }

        @media (max-width: 1180px) {
            .luxriot-grid {
                grid-template-columns: 1fr;
            }
        }

        .luxriot-card {
            background: #0f0f0f;
            border: 1px solid #1f1f1f;
            border-radius: 10px;
            padding: 0.85rem;
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
        }

        .luxriot-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
        }

        .luxriot-header h4 {
            font-size: 1rem;
        }

        .luxriot-status {
            font-size: 0.9rem;
            color: #a6ffb0;
        }

        .luxriot-status.error {
            color: #ff9080;
        }

        .luxriot-row {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            flex-wrap: wrap;
        }

        .luxriot-row label {
            color: #aaa;
            font-size: 0.9rem;
        }

        .luxriot-viewport {
            position: relative;
            background: #050505;
            border: 1px solid #222;
            border-radius: 8px;
            min-height: 260px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }

        .luxriot-viewport img {
            width: 100%;
            max-height: 500px;
            object-fit: contain;
            display: block;
        }

        .luxriot-viewport .luxriot-overlay {
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #777;
            font-size: 0.9rem;
            pointer-events: none;
            background: linear-gradient(135deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
        }

        .luxriot-actions {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            flex-wrap: wrap;
        }

        .luxriot-prompt {
            width: 100%;
            min-height: 80px;
            background: #0f0f0f;
            border: 1px solid #2a2a2a;
            border-radius: 6px;
            color: #eaeaea;
            padding: 0.65rem;
            resize: vertical;
        }

        .luxriot-summaries {
            max-height: 340px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .luxriot-stream-manager {
            margin-top: 0.2rem;
            padding-top: 0.55rem;
            border-top: 1px solid #1f1f1f;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .luxriot-stream-manager-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem;
            flex-wrap: wrap;
        }

        .luxriot-stream-list {
            max-height: 220px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
            padding-right: 0.2rem;
        }

        .luxriot-stream-item {
            display: grid;
            grid-template-columns: auto 1fr auto;
            gap: 0.55rem;
            align-items: start;
            background: #0a0a0a;
            border: 1px solid #212121;
            border-radius: 6px;
            padding: 0.5rem 0.55rem;
        }

        .luxriot-stream-kind {
            font-size: 0.72rem;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            color: #c8d6cc;
            background: #162017;
            border: 1px solid #2d4b32;
            border-radius: 999px;
            padding: 0.14rem 0.45rem;
            white-space: nowrap;
        }

        .luxriot-stream-kind.analytics {
            color: #d3cfbf;
            background: #1d1a12;
            border-color: #4f4325;
        }

        .luxriot-stream-main {
            display: flex;
            flex-direction: column;
            gap: 0.18rem;
            min-width: 0;
        }

        .luxriot-stream-title {
            color: #ececec;
            font-size: 0.87rem;
            font-weight: 600;
        }

        .luxriot-stream-meta {
            color: #adadad;
            font-size: 0.8rem;
            line-height: 1.35;
        }

        .luxriot-stream-tag {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            border: 1px solid #2f5a3a;
            background: rgba(36, 70, 44, 0.28);
            color: #aed7b9;
            font-size: 0.72rem;
            padding: 0.1rem 0.4rem;
            width: fit-content;
        }

        .luxriot-stream-tag.paused {
            border-color: #5f5533;
            background: rgba(84, 71, 32, 0.3);
            color: #e5d29b;
        }

        .luxriot-stream-controls {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: 0.35rem;
            flex-wrap: wrap;
        }

        .luxriot-stream-controls .feature-btn {
            padding: 0.2rem 0.55rem;
            font-size: 0.75rem;
        }

        .luxriot-summary {
            background: #0a0a0a;
            border: 1px solid #222;
            border-radius: 6px;
            padding: 0.65rem;
        }

        .luxriot-summary-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem;
            margin-bottom: 0.35rem;
        }

        .luxriot-summary .timestamp {
            color: #aaa;
            font-size: 0.82rem;
            margin-bottom: 0;
        }

        .luxriot-bookmark-btn {
            font-size: 0.72rem;
            padding: 0.22rem 0.56rem;
            border-radius: 999px;
        }

        .summary-body {
            color: #d7d7d7;
            line-height: 1.45;
        }

        .luxriot-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            padding: 0.25rem 0.55rem;
            background: #161616;
            border: 1px solid #2b2b2b;
            border-radius: 999px;
            font-size: 0.8rem;
            color: #cfcfcf;
        }

        .luxriot-mini-input {
            min-width: 120px;
        }

        /* Probes */
        .probe-card {
            background: #0f0f0f;
            border: 1px solid #1f1f1f;
            border-radius: 10px;
            padding: 0.85rem;
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
            margin-top: 0.5rem;
        }

        .probe-row {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            align-items: center;
        }

        .probe-row.spread {
            justify-content: space-between;
        }

        .probe-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 0.65rem;
            flex-wrap: wrap;
            margin-bottom: 0.6rem;
        }

        .probe-header.split {
            align-items: center;
        }

        .probe-header h4 {
            margin: 0;
        }

        .probe-header-actions {
            display: flex;
            gap: 0.35rem;
            flex-wrap: wrap;
        }

        .probe-panel {
            background: #0f0f0f;
            border: 1px solid #1f1f1f;
            border-radius: 10px;
            padding: 0.85rem;
        }

        .probe-select-grow {
            flex: 1;
        }

        .probe-severity-wrap {
            margin-left: auto;
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }

        .small-label-group {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            color: #bdbdbd;
            font-size: 0.88rem;
        }

        .probe-short-input {
            max-width: 84px;
        }

        .inline-check {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
        }

        .probe-pairs-spacer {
            text-align: center;
        }

        .probe-remove-btn {
            width: 54px;
        }

        .probe-mini-main {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }

        .probe-row label {
            color: #aaa;
            font-size: 0.9rem;
        }

        .probe-textarea {
            width: 100%;
            min-height: 60px;
            background: #0f0f0f;
            border: 1px solid #2a2a2a;
            border-radius: 6px;
            color: #eaeaea;
            padding: 0.65rem;
            resize: vertical;
        }

        .probe-results {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 0.5rem;
        }

        .probe-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 0.75rem;
        }

        .bench-card {
            background: #0f0f0f;
            border: 1px solid #1f1f1f;
            border-radius: 10px;
            padding: 0.75rem;
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 0.5rem;
            align-items: center;
        }

        .bench-meta {
            color: #bcbcbc;
            font-size: 0.95rem;
            line-height: 1.35;
        }

        .probe-result {
            background: #0a0a0a;
            border: 1px solid #1d1d1d;
            border-radius: 8px;
            padding: 0.6rem;
        }

        .probe-result img {
            width: 100%;
            border-radius: 6px;
            margin-bottom: 0.4rem;
        }

        /* Monitoring mock-inspired layout */
        .monitor-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 1rem;
            margin-bottom: 1rem;
            align-items: start;
        }

        .monitor-panel {
            background: #111;
            border: 1px solid #222;
            border-radius: 10px;
            padding: 0.75rem;
        }

        .monitor-panel h4 {
            margin: 0 0 0.6rem 0;
        }

        .monitor-detections-panel .probe-nav {
            display: grid;
            grid-template-columns: auto 1fr auto;
            gap: 0.5rem;
            align-items: center;
        }

        .probe-nav-btn {
            background: #1a1a1a;
            border: 1px solid #2d2d2d;
            color: #cfcfcf;
            border-radius: 6px;
            padding: 0.4rem 0.55rem;
            min-width: 36px;
        }

        .probe-nav-btn:hover {
            background: #232323;
            border-color: #3a3a3a;
        }

        .probe-nav-btn:disabled {
            opacity: 0.4;
            cursor: default;
        }

        .monitor-detections-panel .probe-results {
            grid-template-columns: repeat(5, minmax(118px, 1fr));
            min-height: 190px;
            max-height: none;
            overflow: hidden;
            padding-right: 0;
        }

        .monitor-detections-panel .probe-results .loading {
            grid-column: 1 / -1;
            padding: 1.4rem 0.3rem;
        }

        .monitor-detections-panel .probe-result {
            padding: 0.42rem;
            min-width: 0;
            display: flex;
            flex-direction: column;
            gap: 0.28rem;
            border-color: #262626;
        }

        .monitor-detections-panel .probe-result img {
            margin-bottom: 0;
            width: 100%;
            aspect-ratio: 16 / 9;
            object-fit: cover;
            border-radius: 5px;
            border: 1px solid #202020;
        }

        .probe-result-time {
            font-size: 0.74rem;
            color: #d8d8d8;
            line-height: 1.25;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .probe-result-score {
            font-size: 0.7rem;
            color: #aaaaaa;
            line-height: 1.3;
        }

        .monitor-stream-preview {
            width: 100%;
            aspect-ratio: 16/9;
            background: #0a0a0a;
            border: 1px solid #222;
            border-radius: 6px;
            position: relative;
            overflow: hidden;
        }

        .monitor-stream-preview img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .monitor-stream-overlay {
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, rgba(0,0,0,0.35), rgba(0,0,0,0.2));
            color: #c8c8c8;
            font-weight: 600;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            pointer-events: none;
        }

        .monitor-inline {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            flex-wrap: wrap;
        }

        .monitor-btn-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.5rem;
            margin-top: 0.5rem;
        }

        .monitor-actions-row {
            display: flex;
            justify-content: space-between;
            gap: 0.5rem;
            flex-wrap: wrap;
            align-items: center;
        }

        .monitor-probe-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            flex-wrap: wrap;
        }

        .monitor-probe-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.8rem;
        }

        .monitor-probe-box {
            background: #0c0c0c;
            border: 1px solid #1f1f1f;
            border-radius: 8px;
            padding: 0.6rem;
        }

        .probe-shell {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .probe-mini-card {
            background: #0f0f0f;
            border: 1px solid #1f1f1f;
            border-radius: 10px;
            padding: 0;
            display: block;
            overflow: hidden;
            min-height: 220px;
        }

        .probe-mini-card.active {
            border-color: #4a7a58;
            box-shadow: 0 0 0 1px rgba(74, 122, 88, 0.4), 0 12px 24px rgba(0, 0, 0, 0.35);
        }

        .probe-mini-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.4rem;
        }

        .probe-mini-name {
            font-weight: 700;
            color: #f5f5f5;
            font-size: 0.98rem;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.85);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .probe-status-pill {
            padding: 0.15rem 0.45rem;
            border-radius: 999px;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.02em;
            border: 1px solid #2e2e2e;
        }

        .pill-running { background: rgba(58, 99, 70, 0.2); color: #9bd2a8; border-color: #3a6346; }
        .pill-paused { background: rgba(140, 120, 60, 0.15); color: #e4c47c; border-color: #8c783c; }
        .pill-idle { background: rgba(90, 90, 90, 0.2); color: #cfcfcf; border-color: #555; }
        .pill-disabled { background: rgba(110, 30, 30, 0.18); color: #e8a4a4; border-color: #8b0000; }

        .probe-mini-meta {
            color: #d4d4d4;
            font-size: 0.82rem;
            line-height: 1.35;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.85);
        }

        .probe-mini-actions {
            display: flex;
            gap: 0.35rem;
            flex-wrap: nowrap;
            justify-content: flex-end;
            align-items: center;
        }

        .probe-action-btn {
            width: 32px;
            height: 32px;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.35);
            background: rgba(8, 8, 8, 0.58);
            color: #f0f0f0;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0;
            cursor: pointer;
            backdrop-filter: blur(3px);
            transition: transform 0.15s ease, background 0.15s ease, border-color 0.15s ease;
        }

        .probe-action-btn svg {
            width: 17px;
            height: 17px;
            fill: currentColor;
        }

        .probe-action-btn:hover {
            transform: translateY(-1px);
            background: rgba(22, 22, 22, 0.8);
            border-color: rgba(255, 255, 255, 0.55);
        }

        .probe-action-btn.delete {
            border-color: rgba(255, 120, 120, 0.6);
            color: #ffd1d1;
            background: rgba(80, 18, 18, 0.45);
        }

        .probe-action-btn.delete:hover {
            border-color: rgba(255, 150, 150, 0.85);
            background: rgba(98, 26, 26, 0.65);
        }

        .probe-mini-thumb {
            position: relative;
            border: 1px solid #121212;
            border-radius: 10px;
            background: #040404;
            overflow: hidden;
            min-height: 220px;
            aspect-ratio: 16/10;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .probe-mini-thumb img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }

        .probe-mini-thumb.is-empty::before {
            content: "No preview";
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #a4a4a4;
            font-size: 0.86rem;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            background: repeating-linear-gradient(135deg, #0d0d0d 0 12px, #131313 12px 24px);
        }

        .probe-mini-overlay {
            position: absolute;
            inset: 0;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            padding: 0.55rem;
            background: linear-gradient(180deg, rgba(0, 0, 0, 0.35) 0%, rgba(0, 0, 0, 0.04) 38%, rgba(0, 0, 0, 0.78) 100%);
            pointer-events: none;
        }

        .probe-mini-top,
        .probe-mini-bottom {
            display: flex;
            align-items: center;
            gap: 0.45rem;
        }

        .probe-mini-top {
            justify-content: space-between;
            align-items: flex-start;
        }

        .probe-mini-bottom {
            flex-direction: column;
            align-items: flex-start;
            gap: 0.18rem;
        }

        .probe-mini-overlay .probe-status-pill,
        .probe-mini-overlay .probe-mini-actions {
            pointer-events: auto;
        }

        .probe-mini-score {
            color: #f0f0f0;
            font-size: 0.79rem;
            letter-spacing: 0.01em;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.85);
        }

        .probe-thumb-pill {
            position: absolute;
            top: 6px;
            right: 6px;
        }

        .new-probe-card {
            border: 1px dashed #2f5a3a;
            background: radial-gradient(circle at 20% 20%, #121b14, #070707);
            align-items: center;
            justify-content: center;
            text-align: center;
            min-height: 220px;
            display: flex;
        }

        .probe-new-btn {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.72rem 1.15rem;
            border-radius: 999px;
            border: 1px solid #3d6f4b;
            background: rgba(23, 41, 28, 0.65);
            color: #d7f0dc;
            font-weight: 600;
            cursor: pointer;
        }

        .probe-new-btn:hover {
            background: rgba(34, 57, 40, 0.9);
            border-color: #4f8a5f;
        }

        .probe-new-btn svg {
            width: 16px;
            height: 16px;
            fill: currentColor;
        }

        .probe-pairs {
            background: #0f0f0f;
            border: 1px solid #1f1f1f;
            border-radius: 10px;
            padding: 0.6rem;
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .probe-pairs-header {
            display: grid;
            grid-template-columns: 40px 1fr 1fr 60px;
            gap: 0.35rem;
            align-items: center;
            color: #bdbdbd;
            font-weight: 600;
        }

        .probe-pair-row {
            display: grid;
            grid-template-columns: 40px 1fr 1fr 60px;
            gap: 0.35rem;
            align-items: center;
        }

        .probe-pair-idx {
            color: #a5a5a5;
            text-align: center;
        }

        .probe-add-row {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            margin-top: 0.25rem;
        }

        .probe-meta {
            color: #9a9a9a;
            font-size: 0.9rem;
        }

        .image-probe-panel {
            background: #101010;
            border: 1px solid #203527;
            border-radius: 10px;
            padding: 0.75rem;
            display: grid;
            grid-template-columns: 65% 35%;
            gap: 0.75rem;
            align-items: center;
        }

        .image-probe-left {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .image-probe-row {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .image-probe-pos {
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .image-probe-actions {
            display: flex;
            gap: 0.6rem;
            align-items: center;
            flex-wrap: wrap;
        }

        .probe-preview {
            background: #0a0a0a;
            border: 1px solid #222;
            border-radius: 8px;
            min-height: 140px;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
        }

        .probe-preview img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }

        .probe-preview-overlay {
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, rgba(0,0,0,0.4), rgba(0,0,0,0.25));
            color: #cfcfcf;
            font-weight: 600;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            text-align: center;
            padding: 0.5rem;
        }

        .probe-preview.compact {
            max-width: 220px;
            min-height: 140px;
            justify-self: end;
        }

        .monitor-actions-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.75rem;
            flex-wrap: wrap;
        }

        .monitor-actions-main {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }

        .severity-row {
            display: flex;
            gap: 0.4rem;
            flex-wrap: wrap;
        }

        .settings-short-input {
            width: 110px;
        }

        .input-text {
            min-width: 200px;
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
            min-width: 0;
            max-height: 70vh;
            height: auto;
            object-fit: contain;
            background: #080808;
        }

        @media (max-width: 980px) {
            .container {
                padding: 1rem;
            }

            .header {
                flex-direction: column;
                align-items: flex-start;
                gap: 0.75rem;
            }

            .folder-select {
                flex-direction: column;
            }

            .control-panel {
                position: static;
            }

            .search-controls {
                flex-direction: column;
                align-items: stretch;
                gap: 0.75rem;
            }

            .archive-workspace {
                grid-template-columns: 1fr;
            }

            .archive-search-shell {
                position: static;
            }

            .archive-results-panel {
                padding: 0.65rem;
            }

            .control-group {
                width: 100%;
                flex-wrap: wrap;
                justify-content: flex-start;
            }

            .search-box {
                flex-direction: column;
                align-items: stretch;
            }

            .image-search-inputs {
                margin-right: 0;
            }

            .probe-editor-layout {
                grid-template-columns: 1fr;
            }

            .video-analysis-grid {
                grid-template-columns: 1fr;
            }

            .video-analysis-output {
                min-height: 0;
            }

            .monitor-grid {
                grid-template-columns: 1fr;
            }

            .monitor-detections-panel .probe-results {
                grid-template-columns: repeat(5, minmax(0, 1fr));
                min-height: 0;
            }

            .image-probe-panel {
                grid-template-columns: 1fr;
            }

            .probe-preview.compact {
                max-width: none;
                justify-self: stretch;
            }

            .result-item.expanded .thumbnail {
                max-height: 55vh;
            }
        }
        
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="brand">
                <div class="brand-title">
                    <div class="brand-main">SISU</div>
                    <div class="brand-sub">Smart Image Search and Understanding.</div>
                    <div class="brand-note">Also a Finnish word for a unique combination of courage, resilience, grit, and tenacious determination.</div>
                </div>
            </div>
            <div class="header-actions">
                <div class="settings-icon" id="authTokenBtn" title="Set admin token">
                    <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="#e3e3e3">
                        <path d="M240-160q-33 0-56.5-23.5T160-240v-480q0-33 23.5-56.5T240-800h360q33 0 56.5 23.5T680-720v160h40q33 0 56.5 23.5T800-480v240q0 33-23.5 56.5T720-160H240Zm0-80h480v-240H240v240Zm120-320h240v-160H360v160Zm120 200q17 0 28.5-11.5T520-400q0-17-11.5-28.5T480-440q-17 0-28.5 11.5T440-400q0 17 11.5 28.5T480-360Z"/>
                    </svg>
                </div>
                <div class="settings-icon" id="settingsBtn">
                    <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="#e3e3e3">
                        <path d="m370-80-16-128q-13-5-24.5-12T307-235l-119 50L78-375l103-78q-1-7-1-13.5v-27q0-6.5 1-13.5L78-585l110-190 119 50q11-8 23-15t24-12l16-128h220l16 128q13 5 24.5 12t22.5 15l119-50 110 190-103 78q1 7 1 13.5v27q0 6.5-1 13.5l103 78-110 190-119-50q-11 8-23 15t-24 12L590-80H370Zm70-80h79l14-106q31-8 57.5-23.5T639-327l99 41 39-68-86-65q5-14 7-29.5t2-31.5q0-16-2-31.5t-7-29.5l86-65-39-68-99 41q-22-23-48.5-38.5T533-694l-13-106h-79l-14 106q-31 8-57.5 23.5T321-633l-99-41-39 68 86 65q-5 14-7 29.5t-2 31.5q0 16 2 31.5t7 29.5l-86 65 39 68 99-41q22 23 48.5 38.5T427-266l13 106Zm42-180q58 0 99-41t41-99q0-58-41-99t-99-41q-59 0-99.5 41T342-480q0 58 40.5 99t99.5 41Zm-2-140Z"/>
                    </svg>
                </div>
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
                    <button id="archiveModeBtn" class="mode-tab active">Archive Research</button>
                    <button id="videoModeBtn" class="mode-tab">Video Understanding</button>
                    <button id="monitorModeBtn" class="mode-tab">Monitoring</button>
                </div>
            <div id="archiveBox" class="archive-box">
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
                                <input type="range" id="segmentThresholdSlider" min="40" max="99" value="{segment_threshold_percent}" step="1">
                                <span class="segment-threshold-value" id="segmentThresholdValue">{segment_threshold_percent}%</span>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="archive-workspace">
                    <div id="archiveSearchBox" class="archive-search-shell">
                        <div class="archive-section">
                            <div class="archive-section-title">Text Query</div>
                            <div id="textSearchBox" class="search-box">
                                <input type="text" id="searchQuery" placeholder="Describe what you're looking for..." />
                                <button id="searchBtn">Search</button>
                            </div>
                        </div>
                        <div class="archive-section">
                            <div class="archive-section-title">Image Query</div>
                            <div id="imageSearchBox" class="search-box">
                                <div class="image-search-inputs">
                                    <div class="input-group">
                                        <label for="imageUpload" class="input-label">Upload File:</label>
                                        <input type="file" id="imageUpload" accept="image/*" />
                                    </div>
                                </div>
                                <button id="imageSearchBtn">Search by Image</button>
                            </div>
                        </div>
                    </div>
                    <div class="archive-results-panel">
                        <div class="archive-results-head">Search Results</div>
                        <div id="results" class="results-grid"></div>
                    </div>
                </div>
            </div>
            <div id="videoBox" class="video-box">
        <div class="luxriot-grid">
            <div class="luxriot-card">
                <div class="luxriot-header">
                    <h4>Luxriot Live Preview</h4>
                    <div id="luxriotStatus" class="luxriot-status">Not connected</div>
                        </div>
                        <div class="luxriot-row">
                            <label for="luxriotChannelSelect">Channel:</label>
                            <select id="luxriotChannelSelect" class="luxriot-mini-input"></select>
                            <button id="luxriotRefreshChannels" class="feature-btn">Reload</button>
                            <span class="luxriot-pill">Batch:
                                <select id="luxriotBatchSize" class="luxriot-mini-input">
                                    {luxriot_batch_options}
                                </select>
                            </span>
                            <span class="luxriot-pill">~{luxriot_snapshot_interval}s · {luxriot_snapshot_max_edge}px</span>
                        </div>
                        <div class="luxriot-actions">
                            <button id="luxriotPreviewBtn" class="feature-btn">Preview</button>
                            <button id="luxriotStartCapture" class="feature-btn primary">Start summaries</button>
                            <button id="luxriotStopCapture" class="feature-btn">Stop</button>
                            <button id="luxriotFlushCapture" class="feature-btn">Flush now</button>
                        </div>
                <div class="luxriot-row">
                    <label for="luxriotPrompt">Prompt:</label>
                </div>
                <textarea id="luxriotPrompt" class="luxriot-prompt" placeholder="Describe ongoing activity, anomalies, people, vehicles..."></textarea>
                <div class="luxriot-row">
                    <label for="luxriotSystemPrompt">System prompt (LLM role):</label>
                </div>
                <textarea id="luxriotSystemPrompt" class="luxriot-prompt" placeholder="System prompt for summaries">{luxriot_system_prompt_default}</textarea>
                <div class="luxriot-viewport" id="luxriotViewport">
                    <img id="luxriotPreview" src="" alt="Luxriot live preview" />
                    <div class="luxriot-overlay" id="luxriotOverlay">Preview not started</div>
                </div>
            </div>
                    <div class="luxriot-card">
                        <div class="luxriot-header">
                            <h4>Live Summaries</h4>
                    <div class="luxriot-actions">
                        <button id="luxriotRefreshSummaries" class="feature-btn">Refresh</button>
                    </div>
                </div>
                <div id="luxriotSummaries" class="luxriot-summaries">
                    <div class="loading">No summaries yet.</div>
                </div>
                <div class="luxriot-stream-manager">
                    <div class="luxriot-stream-manager-head">
                        <div class="video-block-title">Active Streams</div>
                        <div class="luxriot-actions">
                            <button id="luxriotRefreshStreams" class="feature-btn">Refresh</button>
                            <button id="luxriotStopAllVideo" class="feature-btn">Stop video</button>
                            <button id="luxriotStopAllAnalytics" class="feature-btn">Stop analytics</button>
                        </div>
                    </div>
                    <div id="luxriotStreams" class="luxriot-stream-list">
                        <div class="loading">No active streams.</div>
                    </div>
                </div>
            </div>
        </div>
        <div class="video-analysis-grid">
            <div class="video-analysis-form">
                <div class="video-row">
                    <div class="input-group">
                        <label for="videoPath" class="input-label">Video Path:</label>
                        <input type="text" id="videoPath" placeholder="/home/user/video.mp4" />
                    </div>
                    <div class="input-group">
                        <label class="input-label" for="videoModel">Model ID:</label>
                        <input type="text" id="videoModel" placeholder="qwen/qwen3-vl-4b" value="{lm_model}" />
                    </div>
                </div>
                <div class="video-row">
                    <div class="input-group">
                        <label class="input-label" for="videoFrameCount">Frames to sample:</label>
                        <select id="videoFrameCount">
                            {video_frame_options_html}
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
                    <label class="video-prompt-note">
                        <input type="checkbox" id="saveVideoPrompt"> Remember this prompt
                    </label>
                </div>
                <div class="video-controls">
                    <button id="videoRunBtn" class="feature-btn primary">Analyze Video</button>
                    <button id="saveSummaryBtn" class="feature-btn is-hidden">Save summary as comment</button>
                    <div id="videoStatus" class="video-status"></div>
                </div>
            </div>
            <div class="video-analysis-output">
                <div class="video-output-wrap">
                    <div class="video-block-title">Summary</div>
                    <div id="videoOutput" class="video-output is-hidden"></div>
                </div>
                <div class="video-frame-block">
                    <div class="video-block-title">Sampled Frames</div>
                    <div id="videoFrames" class="video-frame-grid"></div>
                </div>
            </div>
        </div>
            </div>
                        <div id="monitorBox" class="monitor-box">
                <div class="probe-shell">
                    <div class="probe-panel">
                        <div class="probe-header">
                            <div>
                                <h4>Saved probes</h4>
                                <div class="probe-meta">Click to expand, run, or delete.</div>
                            </div>
                            <div class="probe-header-actions">
                                <button id="probeReloadBtn" class="feature-btn">Refresh list</button>
                                <button id="probeNewBtn" class="feature-btn primary">+ New Probe</button>
                            </div>
                        </div>
                        <div id="probeCards" class="probe-grid"></div>
                    </div>
                    <div class="bench-card">
                        <div>
                            <div class="bench-meta">GPU embed throughput (CLIP) estimate. Helps size total streams/probes.</div>
                            <div id="probeBenchOutput" class="bench-meta">Not run yet.</div>
                        </div>
                        <button id="probeBenchBtn" class="feature-btn primary">Run benchmark</button>
                    </div>
                    <div class="monitor-grid">
                        <div class="monitor-panel monitor-detections-panel">
                            <div class="probe-header split">
                                <h4>Latest Detections</h4>
                                <div id="probeHitsMeta" class="probe-meta">Frames: 0 · Hits: 0</div>
                            </div>
                            <div class="probe-nav">
                                <button class="probe-nav-btn" id="probeDetLeft">&#9664;</button>
                                <div id="probeResults" class="probe-results"></div>
                                <button class="probe-nav-btn" id="probeDetRight">&#9654;</button>
                            </div>
                        </div>
                    </div>
                    <div class="monitor-actions-bar">
                        <div class="monitor-actions-main">
                            <button id="probeEditBtn" class="feature-btn">Probe settings</button>
                            <button id="probeRunBtn" class="feature-btn primary">Run probe</button>
                        </div>
                        <button id="probeDeleteBtn" class="feature-btn">Delete Probe</button>
                    </div>
                </div>
            </div>
    </div>

    <div id="probeEditorModal" class="settings-modal">
        <div class="settings-modal-content probe-editor-modal-content">
            <div class="settings-header">
                <h2>Probe Settings</h2>
                <button class="close-btn" id="closeProbeEditor">&times;</button>
            </div>
            <div class="probe-editor-modal-body">
                <div class="probe-editor-layout">
                    <div class="monitor-panel">
                        <div class="probe-header split">
                            <h4>Live stream</h4>
                            <span id="probeStatus" class="luxriot-status">Idle</span>
                        </div>
                        <div class="probe-row">
                            <label>Channel:</label>
                            <select id="probeChannelSelect" class="luxriot-mini-input probe-select-grow"></select>
                        </div>
                        <div class="monitor-stream-preview">
                            <img id="probePreviewImg" src="" alt="" />
                            <div id="probePreviewOverlay" class="monitor-stream-overlay">No channel</div>
                        </div>
                        <div class="probe-meta" id="probeCaptureStatus">Frames: 0 · Range: n/a</div>
                        <div class="probe-meta" id="probeBufferInfo">Last snapshot: n/a</div>
                        <div class="probe-meta" id="probeStreamState"></div>
                        <div class="probe-row">
                            <label>FPS:</label>
                            <input type="number" id="probeFps" class="settings-input luxriot-mini-input" min="0" step="1" value="0" />
                            <label>Buffer (sec):</label>
                            <input type="number" id="probeWindowSec" class="settings-input luxriot-mini-input" min="0" value="300" />
                        </div>
                        <div class="monitor-btn-row">
                            <button id="probeStartCapture" class="feature-btn primary">Start Stream</button>
                            <button id="probeStopCapture" class="feature-btn">Pause</button>
                            <button id="probeStopAll" class="feature-btn">Stop</button>
                        </div>
                    </div>
                    <div class="probe-editor-settings">
                        <div class="monitor-probe-header">
                            <label>Probe name:</label>
                            <input type="text" id="probeName" class="input-text" placeholder="Provide descriptive name" />
                            <div class="small-label-group">Positive: <input type="number" id="probePosFloor" class="settings-input luxriot-mini-input probe-short-input" step="0.01" value="0.2" /></div>
                            <div class="small-label-group">Margin: <input type="number" id="probeMargin" class="settings-input luxriot-mini-input probe-short-input" step="0.01" value="0.05" /></div>
                            <label class="inline-check">
                                <input type="checkbox" id="probeEnableToggle" checked>
                                Enable probe
                            </label>
                        </div>
                        <div class="probe-row spread">
                            <label><input type="checkbox" id="probeBookmarkToggle" checked> Make bookmarks</label>
                            <div class="probe-severity-wrap">
                                <label>Severity:</label>
                                <select id="probeBookmarkSeverity" class="luxriot-mini-input">
                                    <option value="info">info</option>
                                    <option value="low">low</option>
                                    <option value="normal">normal</option>
                                    <option value="high">high</option>
                                    <option value="critical" selected>critical</option>
                                </select>
                            </div>
                        </div>
                        <div class="probe-pairs" id="probePairs">
                            <div class="probe-pairs-header">
                                <div></div>
                                <div>Positive Examples:</div>
                                <div>Negative Examples:</div>
                                <div class="probe-pairs-spacer">&nbsp;</div>
                            </div>
                        </div>
                        <div class="probe-add-row">
                            <span class="probe-pair-idx">+</span>
                            <button id="probeAddPair" class="feature-btn">Add pair</button>
                        </div>
                        <div class="image-probe-panel">
                            <div class="image-probe-left">
                                <div class="image-probe-row">
                                    <input type="file" id="probeImageFile" class="settings-input" accept="image/*" />
                                </div>
                                <div class="image-probe-pos">
                                    <label>Image Pos:</label>
                                    <input type="number" id="probeImagePos" class="settings-input luxriot-mini-input probe-short-input" step="0.01" min="0" max="1" value="0.7" />
                                </div>
                                <div class="image-probe-actions">
                                    <button id="probeImageEnable" class="feature-btn">Enable Image Probe</button>
                                    <span class="luxriot-status" id="probeImageStatus">Status: Disabled</span>
                                </div>
                            </div>
                            <div class="probe-preview compact">
                                <img id="probeImageThumb" src="" alt="" />
                                <div id="probeImageOverlay" class="probe-preview-overlay">No image selected</div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="settings-actions probe-editor-modal-actions">
                    <button id="probeEditorCloseBtn" class="settings-btn">Close</button>
                    <button id="probeSaveBtn" class="settings-btn primary">Save Probe</button>
                </div>
            </div>
        </div>
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
                <h3>Luxriot Evo</h3>
                <div class="settings-row">
                    <label class="settings-label">Base URL:</label>
                    <input type="text" id="luxriotBaseUrl" class="settings-input" placeholder="http://192.168.1.102:8080">
                </div>
                <div class="settings-row">
                    <label class="settings-label">Username:</label>
                    <input type="text" id="luxriotUsername" class="settings-input" placeholder="admin">
                </div>
                <div class="settings-row">
                    <label class="settings-label">Password:</label>
                    <input type="password" id="luxriotPassword" class="settings-input" placeholder="••••••">
                </div>
                <div class="settings-row">
                    <label class="settings-label">Default Channel ID:</label>
                    <input type="number" id="luxriotDefaultChannelId" class="settings-input" min="1" placeholder="103">
                </div>
                <div class="settings-row">
                    <label class="settings-label">Snapshot Interval (s):</label>
                    <input type="number" id="luxriotSnapshotInterval" class="settings-input" min="1" max="300" placeholder="5">
                </div>
                <div class="settings-row">
                    <label class="settings-label">Snapshot Max Edge (px):</label>
                    <input type="number" id="luxriotSnapshotMaxEdge" class="settings-input" min="640" max="1600" placeholder="800">
                </div>
                <div class="settings-row">
                    <label class="settings-label">Max Buffer Frames:</label>
                    <input type="number" id="luxriotMaxBufferFrames" class="settings-input" min="12" max="2000" placeholder="180">
                </div>
                <div class="settings-row">
                    <label class="settings-label">Auto Bookmark Alerts:</label>
                    <input type="checkbox" id="luxriotAutoBookmarks" class="settings-checkbox">
                </div>
                <div class="settings-row">
                    <label class="settings-label">Severity Mapping:</label>
                    <div class="severity-row">
                        <input type="text" id="luxriotSevInfo" class="settings-input settings-short-input" placeholder="info">
                        <input type="text" id="luxriotSevLow" class="settings-input settings-short-input" placeholder="low">
                        <input type="text" id="luxriotSevNormal" class="settings-input settings-short-input" placeholder="normal">
                        <input type="text" id="luxriotSevHigh" class="settings-input settings-short-input" placeholder="high">
                        <input type="text" id="luxriotSevCritical" class="settings-input settings-short-input" placeholder="critical">
                    </div>
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
                    <input type="checkbox" id="segmentsEnabled" class="settings-checkbox" {segments_enabled_checked}>
                </div>
                <div class="settings-row">
                    <label class="settings-label">Min Segment Patches:</label>
                    <input type="number" id="segmentMinPatches" class="settings-input" min="1" max="256" placeholder="3" value="{segment_min_patches_default}">
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
        const imageSearchBtn = document.getElementById('imageSearchBtn');
        const archiveModeBtn = document.getElementById('archiveModeBtn');
        const videoModeBtn = document.getElementById('videoModeBtn');
        const archiveBox = document.getElementById('archiveBox');
        const videoBox = document.getElementById('videoBox');
        const videoPathInput = document.getElementById('videoPath');
        const videoModelInput = document.getElementById('videoModel');
        const videoFrameCount = document.getElementById('videoFrameCount');
        const videoSampleFpsInput = document.getElementById('videoSampleFps');
        const videoPromptInput = document.getElementById('videoPrompt');
        const saveVideoPromptInput = document.getElementById('saveVideoPrompt');
        const videoRunBtn = document.getElementById('videoRunBtn');
        const videoStatus = document.getElementById('videoStatus');
        const videoOutput = document.getElementById('videoOutput');
        const videoFrames = document.getElementById('videoFrames');
        const saveSummaryBtn = document.getElementById('saveSummaryBtn');
        const monitorModeBtn = document.getElementById('monitorModeBtn');
        const monitorBox = document.getElementById('monitorBox');
        const luxriotChannelSelect = document.getElementById('luxriotChannelSelect');
        const luxriotRefreshChannelsBtn = document.getElementById('luxriotRefreshChannels');
        const luxriotBatchSizeSelect = document.getElementById('luxriotBatchSize');
        const luxriotStatusLabel = document.getElementById('luxriotStatus');
        const luxriotPreviewImg = document.getElementById('luxriotPreview');
        const luxriotOverlay = document.getElementById('luxriotOverlay');
        const luxriotPreviewBtn = document.getElementById('luxriotPreviewBtn');
        const luxriotStartCaptureBtn = document.getElementById('luxriotStartCapture');
        const luxriotStopCaptureBtn = document.getElementById('luxriotStopCapture');
        const luxriotFlushCaptureBtn = document.getElementById('luxriotFlushCapture');
        const luxriotRefreshSummariesBtn = document.getElementById('luxriotRefreshSummaries');
        const luxriotSummaries = document.getElementById('luxriotSummaries');
        const luxriotStreams = document.getElementById('luxriotStreams');
        const luxriotRefreshStreamsBtn = document.getElementById('luxriotRefreshStreams');
        const luxriotStopAllVideoBtn = document.getElementById('luxriotStopAllVideo');
        const luxriotStopAllAnalyticsBtn = document.getElementById('luxriotStopAllAnalytics');
        const luxriotPromptInput = document.getElementById('luxriotPrompt');
        const luxriotSystemPromptInput = document.getElementById('luxriotSystemPrompt');
        const probeChannelSelect = document.getElementById('probeChannelSelect');
        const probeTopKInput = document.getElementById('probeTopK');
        const probePosFloorInput = document.getElementById('probePosFloor');
        const probeMarginInput = document.getElementById('probeMargin');
        const probeNameInput = document.getElementById('probeName');
        const probeRunBtn = document.getElementById('probeRunBtn');
        const probeSaveBtn = document.getElementById('probeSaveBtn');
        const probeDeleteBtn = document.getElementById('probeDeleteBtn');
        const probeEditBtn = document.getElementById('probeEditBtn');
        const probeEditorModal = document.getElementById('probeEditorModal');
        const closeProbeEditorBtn = document.getElementById('closeProbeEditor');
        const probeEditorCloseBtn = document.getElementById('probeEditorCloseBtn');
        const probeResults = document.getElementById('probeResults');
        const probeStatus = document.getElementById('probeStatus');
        const probeBookmarkSeverityInput = document.getElementById('probeBookmarkSeverity');
        const probeBookmarkToggle = document.getElementById('probeBookmarkToggle');
        const probeStartCaptureBtn = document.getElementById('probeStartCapture');
        const probeStopCaptureBtn = document.getElementById('probeStopCapture');
        const probeStopAllBtn = document.getElementById('probeStopAll');
        const probeCaptureStatus = document.getElementById('probeCaptureStatus');
        const probeHitsMeta = document.getElementById('probeHitsMeta');
        const probeCards = document.getElementById('probeCards');
        const probeNewBtn = document.getElementById('probeNewBtn');
        const probeReloadBtn = document.getElementById('probeReloadBtn');
        const probePreviewImg = document.getElementById('probePreviewImg');
        const probePreviewOverlay = document.getElementById('probePreviewOverlay');
        const probePairsContainer = document.getElementById('probePairs');
        const probeAddPairBtn = document.getElementById('probeAddPair');
        const probeImageFile = document.getElementById('probeImageFile');
        const probeImageEnableBtn = document.getElementById('probeImageEnable');
        const probeImageStatus = document.getElementById('probeImageStatus');
        const probeImageThumb = document.getElementById('probeImageThumb');
        const probeImageOverlay = document.getElementById('probeImageOverlay');
        const probeImagePosInput = document.getElementById('probeImagePos');
        const probeDetLeftBtn = document.getElementById('probeDetLeft');
        const probeDetRightBtn = document.getElementById('probeDetRight');
        const resultLimitSelect = document.getElementById('resultLimit');
        const sortBySelect = document.getElementById('sortBy');
        const showCommentedBtn = document.getElementById('showCommentedBtn');
        const resultsContainer = document.getElementById('results');
        const probeBufferInfo = document.getElementById('probeBufferInfo');
        const probeStreamState = document.getElementById('probeStreamState');
        const probeEnableToggle = document.getElementById('probeEnableToggle');
        const probeBenchBtn = document.getElementById('probeBenchBtn');
        const probeBenchOutput = document.getElementById('probeBenchOutput');
        
        let currentFolder = '';
        let currentMode = 'archive';
        let videoTimerHandle = null;
        let videoRequestStarted = 0;
        let lastSummaryText = '';
        let lastSummaryTarget = null;
        let segmentContextByIndex = {};
        let luxriotSummaryLogCache = [];
        const luxriotDefaults = {
            channelId: {luxriot_default_channel},
            snapshotInterval: {luxriot_snapshot_interval},
            snapshotMaxEdge: {luxriot_snapshot_max_edge},
            baseUrl: '{luxriot_base_url}',
            batchSize: {luxriot_batch_default}
        };
        let luxriotActiveChannel = luxriotDefaults.channelId;
        let luxriotPreviewTimer = null;
        let luxriotSummaryTimer = null;
        let luxriotStreamsCache = [];
        let luxriotInitialized = false;
        const probeHitsCacheByKey = {};
        const probeHitsOffsetByKey = {};
        const probeFramesByKey = {};
        const probeHitsUpdatedByKey = {};
        const probeWindowSecByKey = {};
        let probePairsState = [];
        let probeImageState = null;
        let imageProbeEnabled = false;
        let probeList = [];
        let activeProbeId = null;
        const probeCaptureState = {};
        let probeRunTimer = null;
        let probeRunInFlight = false;
        let probePreviewTimer = null;
        let lastProbeRefresh = 0;
        let probeStatusTimer = null;
        const channelCaptureConfig = {};
        const channelFpsDesired = {};
        const ADMIN_TOKEN_STORAGE_KEY = 'evs_admin_token';

        function getAdminToken() {
            return (localStorage.getItem(ADMIN_TOKEN_STORAGE_KEY) || '').trim();
        }

        function saveAdminToken(token) {
            const clean = (token || '').trim();
            if (clean) {
                localStorage.setItem(ADMIN_TOKEN_STORAGE_KEY, clean);
            } else {
                localStorage.removeItem(ADMIN_TOKEN_STORAGE_KEY);
            }
        }

        (function seedAdminTokenFromQuery() {
            try {
                const url = new URL(window.location.href);
                const qp = (url.searchParams.get('admin_token') || '').trim();
                if (!qp) return;
                saveAdminToken(qp);
                url.searchParams.delete('admin_token');
                window.history.replaceState({}, '', url.toString());
            } catch (_) {
                // no-op
            }
        })();

        const rawFetch = window.fetch.bind(window);
        window.fetch = (input, init = {}) => {
            const options = init ? { ...init } : {};
            const token = getAdminToken();
            if (token) {
                const headers = new Headers(options.headers || {});
                if (!headers.has('X-Admin-Token') && !headers.has('Authorization')) {
                    headers.set('X-Admin-Token', token);
                }
                options.headers = headers;
            }
            return rawFetch(input, options);
        };

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
            archiveModeBtn.classList.toggle('active', mode === 'archive');
            videoModeBtn.classList.toggle('active', mode === 'video');
            monitorModeBtn.classList.toggle('active', mode === 'monitor');
            if (archiveBox) {
                archiveBox.style.display = mode === 'archive' ? 'flex' : 'none';
            }
            videoBox.style.display = mode === 'video' ? 'flex' : 'none';
            monitorBox.style.display = mode === 'monitor' ? 'block' : 'none';
            if (mode === 'video') {
                ensureLuxriotInit();
                startLuxriotPreview();
                refreshLuxriotSummaries();
                refreshLuxriotStreams();
                startLuxriotSummaryPoll();
                syncProbeChannelSelect();
            } else if (mode === 'monitor') {
                ensureLuxriotInit();
                syncProbeChannelSelect();
                startProbePreview(parseInt(probeChannelSelect?.value || luxriotActiveChannel, 10));
                refreshProbeStatus();
                loadProbeList();
                startProbeStatusPoll();
            } else {
                stopLuxriotPreview();
                stopLuxriotSummaryPoll();
                stopProbePreview();
                stopProbeRunLoop();
                stopProbeStatusPoll();
                if (probeEditorModal) {
                    probeEditorModal.style.display = 'none';
                }
            }
        }

        function setLuxriotStatus(text, isError = false) {
            if (!luxriotStatusLabel) return;
            luxriotStatusLabel.textContent = text;
            luxriotStatusLabel.classList.toggle('error', Boolean(isError));
            if (isError) {
                luxriotStatusLabel.title = text;
            } else {
                luxriotStatusLabel.removeAttribute('title');
            }
        }

        function stopLuxriotPreview() {
            if (luxriotPreviewTimer) {
                clearInterval(luxriotPreviewTimer);
                luxriotPreviewTimer = null;
            }
        }

        function stopLuxriotSummaryPoll() {
            if (luxriotSummaryTimer) {
                clearInterval(luxriotSummaryTimer);
                luxriotSummaryTimer = null;
            }
        }

        function getSelectedLuxriotChannel() {
            const raw = luxriotChannelSelect ? luxriotChannelSelect.value : '';
            const parsed = parseInt(raw || luxriotActiveChannel, 10);
            if (Number.isFinite(parsed)) {
                luxriotActiveChannel = parsed;
                return parsed;
            }
            return luxriotDefaults.channelId;
        }

        async function fetchLuxriotChannels(force = false) {
            if (!luxriotChannelSelect) return;
            luxriotChannelSelect.innerHTML = '<option>Loading...</option>';
            try {
                const response = await fetch(`/luxriot/channels${force ? '?force=1' : ''}`);
                const data = await response.json();
                if (data.error) {
                    throw new Error(data.error);
                }
                const channels = data.channels || [];
                if (!channels.length) {
                    luxriotChannelSelect.innerHTML = '<option value="">No channels</option>';
                    setLuxriotStatus('No channels available', true);
                    return;
                }
                const options = channels.map((ch) => {
                    const id = ch.id;
                    const label = ch.title || `Channel ${id}`;
                    const selected = String(id) === String(luxriotActiveChannel) ? 'selected' : '';
                    return `<option value="${id}" ${selected}>${label} (#${id})</option>`;
                });
                luxriotChannelSelect.innerHTML = options.join('');
                if (!channels.some((ch) => String(ch.id) === String(luxriotActiveChannel))) {
                    luxriotActiveChannel = channels[0].id;
                    luxriotChannelSelect.value = luxriotActiveChannel;
                }
                setLuxriotStatus(`Loaded ${channels.length} channels`);
            } catch (err) {
                luxriotChannelSelect.innerHTML = '<option value="">Load failed</option>';
                setLuxriotStatus('Channel load failed: ' + err.message, true);
            }
        }

        function startLuxriotPreview() {
            if (!luxriotPreviewImg) return;
            const channelId = getSelectedLuxriotChannel();
            if (!channelId) {
                setLuxriotStatus('Select a channel to preview', true);
                return;
            }
            const refresh = () => {
                if (luxriotOverlay) {
                    luxriotOverlay.textContent = 'Loading...';
                }
                luxriotPreviewImg.src = `/luxriot/snapshot/${channelId}?t=${Date.now()}`;
            };
            luxriotPreviewImg.onload = () => {
                if (luxriotOverlay) luxriotOverlay.textContent = '';
                setLuxriotStatus(`Previewing channel ${channelId}`);
            };
            luxriotPreviewImg.onerror = () => {
                if (luxriotOverlay) luxriotOverlay.textContent = 'Preview failed';
                setLuxriotStatus('Preview failed', true);
            };
            stopLuxriotPreview();
            refresh();
            const intervalMs = Math.max(2000, (luxriotDefaults.snapshotInterval || 5) * 1000);
            luxriotPreviewTimer = setInterval(refresh, intervalMs);
        }

        function renderLuxriotSummaries(logs) {
            if (!luxriotSummaries) return;
            if (!logs || !logs.length) {
                luxriotSummaryLogCache = [];
                luxriotSummaries.innerHTML = '<div class="loading">No summaries yet.</div>';
                return;
            }
            luxriotSummaryLogCache = logs.slice().reverse();
            const html = luxriotSummaryLogCache
                .map((log, idx) => {
                    const ts = Number(log.created_at) ? new Date(log.created_at * 1000) : null;
                    const tsLabel = ts ? ts.toLocaleString() : 'n/a';
                    const frameLabel = log.frame_count ? `${log.frame_count} frames` : '';
                    const summary = String(log.summary || '').trim();
                    const canBookmark = summary.length > 0;
                    return `
                        <div class="luxriot-summary">
                            <div class="luxriot-summary-head">
                                <div class="timestamp">${tsLabel}${frameLabel ? ` · ${frameLabel}` : ''}</div>
                                <button class="feature-btn luxriot-bookmark-btn" data-luxriot-bookmark="${idx}" ${canBookmark ? '' : 'disabled'}>
                                    Bookmark
                                </button>
                            </div>
                            <div class="summary-body">${renderMarkdown(summary)}</div>
                        </div>
                    `;
                })
                .join('');
            luxriotSummaries.innerHTML = html;
        }

        function renderLuxriotStreams(payload) {
            if (!luxriotStreams) return;
            const data = payload && typeof payload === 'object' ? payload : {};
            const videoStreams = Array.isArray(data.video_streams) ? data.video_streams : [];
            const analyticsStreams = Array.isArray(data.analytics_streams) ? data.analytics_streams : [];
            const pausedChannels = new Set(
                (Array.isArray(data.paused_analytics_channels) ? data.paused_analytics_channels : [])
                    .map((val) => parseInt(String(val), 10))
                    .filter((val) => Number.isFinite(val))
            );
            luxriotStreamsCache = [...videoStreams, ...analyticsStreams];
            if (!videoStreams.length && !analyticsStreams.length && !pausedChannels.size) {
                luxriotStreams.innerHTML = '<div class="loading">No active streams.</div>';
                return;
            }
            const rows = [];
            const sortedVideo = videoStreams
                .slice()
                .sort((a, b) => (Number(a.channel_id) || 0) - (Number(b.channel_id) || 0));
            const sortedAnalytics = analyticsStreams
                .slice()
                .sort((a, b) => (Number(a.channel_id) || 0) - (Number(b.channel_id) || 0));
            sortedVideo.forEach((stream) => {
                const channelId = Number(stream.channel_id) || 0;
                const batch = Number(stream.batch_size) || 0;
                const queued = Number(stream.pending_frames) || 0;
                const flushes = Number(stream.flush_count) || 0;
                const parts = [`Channel #${channelId}`];
                if (batch > 0) parts.push(`batch ${batch}`);
                parts.push(`${queued} queued`);
                if (flushes > 0) parts.push(`${flushes} flushes`);
                if (stream.last_error) parts.push('error');
                rows.push(`
                    <div class="luxriot-stream-item">
                        <div class="luxriot-stream-kind">Video</div>
                        <div class="luxriot-stream-main">
                            <div class="luxriot-stream-title">${escapeHtml(stream.running ? 'Running summary stream' : 'Stopped summary stream')}</div>
                            <div class="luxriot-stream-meta">${escapeHtml(parts.join(' · '))}</div>
                        </div>
                        <div class="luxriot-stream-controls">
                            <button class="feature-btn" data-stream-stop="${channelId}" data-stream-type="video">Stop</button>
                        </div>
                    </div>
                `);
            });
            sortedAnalytics.forEach((stream) => {
                const channelId = Number(stream.channel_id) || 0;
                const queued = Number(stream.pending_frames) || 0;
                const intervalSec = Number(stream.interval_sec);
                const fpsLabel = Number.isFinite(intervalSec) && intervalSec > 0 ? `${(1 / intervalSec).toFixed(2)} fps` : 'n/a fps';
                const isPaused = pausedChannels.has(channelId);
                const parts = [`Channel #${channelId}`, fpsLabel, `${queued} buffered`];
                if (stream.last_error) parts.push('error');
                rows.push(`
                    <div class="luxriot-stream-item">
                        <div class="luxriot-stream-kind analytics">Analytics</div>
                        <div class="luxriot-stream-main">
                            <div class="luxriot-stream-title">${escapeHtml(stream.running ? 'Running probe capture' : 'Stopped probe capture')}</div>
                            <div class="luxriot-stream-meta">${escapeHtml(parts.join(' · '))}</div>
                            ${isPaused ? '<span class="luxriot-stream-tag paused">paused</span>' : '<span class="luxriot-stream-tag">active</span>'}
                        </div>
                        <div class="luxriot-stream-controls">
                            <button class="feature-btn" data-stream-stop="${channelId}" data-stream-type="analytics">Stop</button>
                        </div>
                    </div>
                `);
            });
            const runningAnalyticsChannels = new Set(sortedAnalytics.map((stream) => Number(stream.channel_id) || 0));
            Array.from(pausedChannels)
                .filter((channelId) => !runningAnalyticsChannels.has(channelId))
                .sort((a, b) => a - b)
                .forEach((channelId) => {
                    rows.push(`
                        <div class="luxriot-stream-item">
                            <div class="luxriot-stream-kind analytics">Analytics</div>
                            <div class="luxriot-stream-main">
                                <div class="luxriot-stream-title">Paused probe capture</div>
                                <div class="luxriot-stream-meta">Channel #${channelId}</div>
                                <span class="luxriot-stream-tag paused">paused</span>
                            </div>
                            <div class="luxriot-stream-controls"></div>
                        </div>
                    `);
                });
            luxriotStreams.innerHTML = rows.join('');
        }

        async function refreshLuxriotStreams() {
            if (!luxriotStreams) return;
            try {
                const resp = await fetch('/luxriot/streams');
                const data = await resp.json();
                if (!resp.ok || data.error) {
                    throw new Error(data.error || 'Failed to fetch stream state');
                }
                renderLuxriotStreams(data);
            } catch (err) {
                luxriotStreams.innerHTML = `<div class="loading">Stream state unavailable: ${escapeHtml(err.message || 'Unknown error')}</div>`;
            }
        }

        async function stopLuxriotStream(channelId, streamType) {
            const parsedChannelId = parseInt(String(channelId || ''), 10);
            const normalizedType = String(streamType || '').trim().toLowerCase();
            if (!Number.isFinite(parsedChannelId)) {
                setLuxriotStatus('Invalid channel id for stream stop', true);
                return;
            }
            if (!['video', 'analytics', 'both'].includes(normalizedType)) {
                setLuxriotStatus('Invalid stream type', true);
                return;
            }
            setLuxriotStatus(`Stopping ${normalizedType} stream on channel ${parsedChannelId}...`);
            try {
                const response = await fetch('/luxriot/streams/stop', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        channel_id: parsedChannelId,
                        stream_type: normalizedType,
                        pause_analytics: true,
                    }),
                });
                const data = await parseApiJson(response, 'Stream stop failed');
                if (data.streams) {
                    renderLuxriotStreams(data.streams);
                } else {
                    await refreshLuxriotStreams();
                }
                if (normalizedType === 'video' || normalizedType === 'both') {
                    await refreshLuxriotSummaries(parsedChannelId);
                }
                if (normalizedType === 'analytics' || normalizedType === 'both') {
                    await refreshProbeStatus(parsedChannelId);
                }
                setLuxriotStatus(`Stopped ${normalizedType} stream on channel ${parsedChannelId}`);
            } catch (err) {
                setLuxriotStatus(err.message || 'Failed to stop stream', true);
            }
        }

        async function stopAllLuxriotStreams(streamType) {
            const normalizedType = String(streamType || '').trim().toLowerCase();
            const stopVideo = normalizedType === 'video' || normalizedType === 'both';
            const stopAnalytics = normalizedType === 'analytics' || normalizedType === 'both';
            if (!stopVideo && !stopAnalytics) return;
            setLuxriotStatus(`Stopping ${normalizedType} streams...`);
            try {
                const response = await fetch('/luxriot/streams/stop_all', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        stop_video: stopVideo,
                        stop_analytics: stopAnalytics,
                        pause_analytics: true,
                    }),
                });
                const data = await parseApiJson(response, 'Stop-all failed');
                if (data.streams) {
                    renderLuxriotStreams(data.streams);
                } else {
                    await refreshLuxriotStreams();
                }
                if (stopVideo) {
                    stopLuxriotSummaryPoll();
                    await refreshLuxriotSummaries();
                }
                if (stopAnalytics) {
                    await refreshProbeStatus();
                }
                setLuxriotStatus(`Stopped ${normalizedType} streams`);
            } catch (err) {
                setLuxriotStatus(err.message || 'Failed to stop streams', true);
            }
        }

        async function sendLuxriotBookmarkFromLog(logIndex, triggerBtn = null) {
            const idx = Number.isFinite(logIndex) ? logIndex : parseInt(String(logIndex || ''), 10);
            if (!Number.isFinite(idx) || idx < 0 || idx >= luxriotSummaryLogCache.length) {
                setLuxriotStatus('Invalid summary selection', true);
                return;
            }
            const log = luxriotSummaryLogCache[idx] || {};
            const summaryText = String(log.summary || '').trim();
            if (!summaryText) {
                setLuxriotStatus('No summary text to bookmark', true);
                return;
            }
            const channelId = Number(log.channel_id) || getSelectedLuxriotChannel() || luxriotDefaults.channelId;
            const firstLine = summaryText.split(/\\r?\\n/, 1)[0].trim();
            const titleBase = firstLine || `Channel ${channelId} summary`;
            const title = titleBase.length > 80 ? `${titleBase.slice(0, 77)}...` : titleBase;
            const description = summaryText.length > 2400 ? `${summaryText.slice(0, 2397)}...` : summaryText;
            const createdAtSec = Number(log.created_at);
            const timestampMs = Number.isFinite(createdAtSec) ? Math.round(createdAtSec * 1000) : null;

            const button = triggerBtn;
            const originalLabel = button ? button.textContent : '';
            if (button) {
                button.disabled = true;
                button.textContent = 'Saving...';
            }

            try {
                const response = await fetch('/luxriot/bookmark', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        channel_id: channelId,
                        title: `Live summary: ${title}`,
                        description,
                        severity: 'normal',
                        state: 'new',
                        timestamp_ms: timestampMs
                    }),
                });
                await parseApiJson(response, 'Bookmark failed');
                setLuxriotStatus(`Bookmark sent for channel ${channelId}`);
                if (button) {
                    button.textContent = 'Bookmarked';
                }
            } catch (err) {
                setLuxriotStatus(err.message || 'Bookmark failed', true);
                if (button) {
                    button.textContent = originalLabel || 'Bookmark';
                }
            } finally {
                if (button) {
                    button.disabled = false;
                }
            }
        }

        async function refreshLuxriotSummaries(channelId = getSelectedLuxriotChannel()) {
            if (!channelId) return;
            try {
                const resp = await fetch(`/luxriot/session?channel_id=${channelId}`);
                const data = await resp.json();
                if (data.error) {
                    throw new Error(data.error);
                }
                renderLuxriotSummaries(data.logs || []);
                let baseStatus = data.running ? `Summaries running · batch ${data.batch_size || ''}` : 'Summaries stopped';
                if (typeof data.pending_frames === 'number' && data.pending_frames > 0) {
                    baseStatus += ` · ${data.pending_frames} frames queued`;
                }
                setLuxriotStatus(baseStatus, Boolean(data.last_error));
                if (data.last_error) {
                    luxriotStatusLabel.title = data.last_error;
                }
            } catch (err) {
                setLuxriotStatus('Failed to fetch summaries: ' + err.message, true);
            }
        }

        function startLuxriotSummaryPoll(channelId = getSelectedLuxriotChannel()) {
            stopLuxriotSummaryPoll();
            luxriotSummaryTimer = setInterval(() => {
                refreshLuxriotSummaries(channelId);
                refreshLuxriotStreams();
            }, 8000);
        }

        async function startLuxriotCapture() {
            const channelId = getSelectedLuxriotChannel();
            if (!channelId) {
                setLuxriotStatus('Select a channel first', true);
                return;
            }
            const batchSize = luxriotBatchSizeSelect
                ? parseInt(luxriotBatchSizeSelect.value, 10)
                : luxriotDefaults.batchSize || 12;
            const prompt = luxriotPromptInput ? luxriotPromptInput.value.trim() : '';
            const systemPrompt = luxriotSystemPromptInput ? luxriotSystemPromptInput.value.trim() : '';
            const fallbackPrompt = videoPromptInput ? videoPromptInput.value.trim() : '';
            luxriotStartCaptureBtn.disabled = true;
            setLuxriotStatus('Starting summaries...');
            try {
                const resp = await fetch('/luxriot/start_capture', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        channel_id: channelId,
                        batch_size: batchSize,
                        prompt: prompt || fallbackPrompt,
                        model: videoModelInput ? videoModelInput.value.trim() : '',
                        system_prompt: systemPrompt
                    })
                });
                const data = await resp.json();
                if (!resp.ok || data.error) {
                    throw new Error(data.error || 'Luxriot start failed');
                }
                setLuxriotStatus(`Summaries running on channel ${channelId} (batch ${batchSize})`);
                refreshLuxriotSummaries(channelId);
                refreshLuxriotStreams();
                startLuxriotSummaryPoll(channelId);
            } catch (err) {
                setLuxriotStatus(err.message, true);
            } finally {
                luxriotStartCaptureBtn.disabled = false;
            }
        }

        async function stopLuxriotCapture() {
            const channelId = getSelectedLuxriotChannel();
            setLuxriotStatus('Stopping...');
            try {
                const resp = await fetch('/luxriot/stop_capture', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ channel_id: channelId })
                });
                const data = await resp.json();
                if (data.error) {
                    throw new Error(data.error);
                }
                setLuxriotStatus('Summaries stopped');
                refreshLuxriotSummaries(channelId);
                refreshLuxriotStreams();
            } catch (err) {
                setLuxriotStatus(err.message, true);
            }
        }

        async function flushLuxriotCapture() {
            const channelId = getSelectedLuxriotChannel();
            setLuxriotStatus('Flushing...');
            try {
                const resp = await fetch('/luxriot/flush_capture', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ channel_id: channelId })
                });
                const data = await resp.json();
                if (!resp.ok || data.error) {
                    throw new Error(data.error || data.message || 'Flush failed');
                }
                setLuxriotStatus('Buffer flushed');
                if (data.status) {
                    renderLuxriotSummaries(data.status.logs || []);
                }
                refreshLuxriotStreams();
            } catch (err) {
                setLuxriotStatus(err.message, true);
            }
        }

        async function ensureLuxriotInit() {
            if (luxriotInitialized) return;
            luxriotInitialized = true;
            await fetchLuxriotChannels();
            startLuxriotPreview();
            refreshLuxriotSummaries();
            refreshLuxriotStreams();
        }

        const savedVideoPrompt = localStorage.getItem('evs_video_prompt');
        if (savedVideoPrompt && videoPromptInput) {
            videoPromptInput.value = savedVideoPrompt;
            if (saveVideoPromptInput) {
                saveVideoPromptInput.checked = true;
            }
        }
        if (luxriotPromptInput && videoPromptInput && videoPromptInput.value && !luxriotPromptInput.value) {
            luxriotPromptInput.value = videoPromptInput.value;
        }
        function syncProbeChannelSelect() {
            if (probeChannelSelect && luxriotChannelSelect && luxriotChannelSelect.innerHTML) {
                probeChannelSelect.innerHTML = luxriotChannelSelect.innerHTML;
                probeChannelSelect.value = luxriotChannelSelect.value || luxriotDefaults.channelId;
            }
        }
        syncProbeChannelSelect();

        setMode(currentMode);
        
        // Settings modal elements
        const authTokenBtn = document.getElementById('authTokenBtn');
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
        const luxriotBaseUrlInput = document.getElementById('luxriotBaseUrl');
        const luxriotUsernameInput = document.getElementById('luxriotUsername');
        const luxriotPasswordInput = document.getElementById('luxriotPassword');
        const luxriotDefaultChannelIdInput = document.getElementById('luxriotDefaultChannelId');
        const luxriotSnapshotIntervalInput = document.getElementById('luxriotSnapshotInterval');
        const luxriotSnapshotMaxEdgeInput = document.getElementById('luxriotSnapshotMaxEdge');
        const luxriotMaxBufferFramesInput = document.getElementById('luxriotMaxBufferFrames');
        const luxriotAutoBookmarksInput = document.getElementById('luxriotAutoBookmarks');
        const luxriotSevInfoInput = document.getElementById('luxriotSevInfo');
        const luxriotSevLowInput = document.getElementById('luxriotSevLow');
        const luxriotSevNormalInput = document.getElementById('luxriotSevNormal');
        const luxriotSevHighInput = document.getElementById('luxriotSevHigh');
        const luxriotSevCriticalInput = document.getElementById('luxriotSevCritical');
        
        let segmentThreshold = 0.7;

        function toBool(value, fallback = false) {
            if (typeof value === 'boolean') return value;
            if (value === null || value === undefined) return fallback;
            if (typeof value === 'string') {
                const normalized = value.trim().toLowerCase();
                if (['1', 'true', 'yes', 'on'].includes(normalized)) return true;
                if (['0', 'false', 'no', 'off'].includes(normalized)) return false;
                return fallback;
            }
            return Boolean(value);
        }

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

        if (authTokenBtn) {
            authTokenBtn.addEventListener('click', () => {
                const existing = getAdminToken();
                const entered = window.prompt(
                    'Set admin token (stored in this browser for mutating API calls). Leave empty to clear.',
                    existing
                );
                if (entered === null) {
                    return;
                }
                saveAdminToken(entered);
                const hasToken = !!getAdminToken();
                authTokenBtn.style.opacity = hasToken ? '1' : '0.6';
                indexStatus.textContent = hasToken ? 'Admin token saved in browser.' : 'Admin token cleared.';
                indexStatus.className = hasToken ? 'status success' : 'status warning';
            });
            authTokenBtn.style.opacity = getAdminToken() ? '1' : '0.6';
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

        if (probeEditBtn && probeEditorModal) {
            probeEditBtn.addEventListener('click', () => {
                probeEditorModal.style.display = 'block';
            });
        }
        if (closeProbeEditorBtn && probeEditorModal) {
            closeProbeEditorBtn.addEventListener('click', () => {
                probeEditorModal.style.display = 'none';
            });
        }
        if (probeEditorCloseBtn && probeEditorModal) {
            probeEditorCloseBtn.addEventListener('click', () => {
                probeEditorModal.style.display = 'none';
            });
        }
        if (probeEditorModal) {
            probeEditorModal.addEventListener('click', (e) => {
                if (e.target === probeEditorModal) {
                    probeEditorModal.style.display = 'none';
                }
            });
        }
        
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
                    document.getElementById('debug').checked = toBool(settings.debug, false);
                    embedderSelect.value = settings.embedder || 'clip';
                    fusionEnabledInput.checked = toBool(settings.fusionEnabled, false);
                    const parsedFusionAlpha = parseFloat(settings.fusionAlpha);
                    const fusionAlpha = Number.isFinite(parsedFusionAlpha) ? parsedFusionAlpha : 0.7;
                    fusionAlphaInput.value = fusionAlpha.toFixed(2);
                    dinoModelInput.value = settings.dinoModel || 'dinov3_vitb16';
                    dinoEmbedDimInput.value = settings.dinoEmbedDim || 1280;
                    dinoWeightsInput.value = settings.dinoWeightsPath || '';
                    indexModeSelect.value = settings.indexMode || 'clip';
                    updateFusionUI(fusionEnabledInput.checked);
                    rerankEnabledInput.checked = toBool(settings.rerankEnabled, false);
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
                    if (luxriotBaseUrlInput) luxriotBaseUrlInput.value = settings.luxriotBaseUrl || '';
                    if (luxriotUsernameInput) luxriotUsernameInput.value = settings.luxriotUsername || '';
                    if (luxriotPasswordInput) luxriotPasswordInput.value = settings.luxriotPassword || '';
                    if (luxriotDefaultChannelIdInput) luxriotDefaultChannelIdInput.value = settings.luxriotDefaultChannelId || '';
                    if (luxriotSnapshotIntervalInput) luxriotSnapshotIntervalInput.value = settings.luxriotSnapshotInterval || 5;
                    if (luxriotSnapshotMaxEdgeInput) luxriotSnapshotMaxEdgeInput.value = settings.luxriotSnapshotMaxEdge || 800;
                    if (luxriotMaxBufferFramesInput) luxriotMaxBufferFramesInput.value = settings.luxriotMaxBufferFrames || 180;
                    if (luxriotAutoBookmarksInput) luxriotAutoBookmarksInput.checked = toBool(settings.luxriotAutoBookmarks, false);
                    if (settings.luxriotSeverityMap) {
                        if (luxriotSevInfoInput) luxriotSevInfoInput.value = settings.luxriotSeverityMap.info || 'info';
                        if (luxriotSevLowInput) luxriotSevLowInput.value = settings.luxriotSeverityMap.low || 'low';
                        if (luxriotSevNormalInput) luxriotSevNormalInput.value = settings.luxriotSeverityMap.normal || 'normal';
                        if (luxriotSevHighInput) luxriotSevHighInput.value = settings.luxriotSeverityMap.high || 'high';
                        if (luxriotSevCriticalInput) luxriotSevCriticalInput.value = settings.luxriotSeverityMap.critical || 'critical';
                    }
                    applyEmbedderUI(embedderSelect.value);
                    segmentsEnabledInput.checked = toBool(settings.segmentsEnabled, segmentsEnabledInput.checked);
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
                    indexFolderName: document.getElementById('indexFolderName').value.trim(),
                    luxriotBaseUrl: luxriotBaseUrlInput.value.trim(),
                    luxriotUsername: luxriotUsernameInput.value.trim(),
                    luxriotPassword: luxriotPasswordInput ? luxriotPasswordInput.value : '',
                    luxriotDefaultChannelId: parseInt(luxriotDefaultChannelIdInput ? luxriotDefaultChannelIdInput.value : config.LUXRIOT_DEFAULT_CHANNEL_ID),
                    luxriotSnapshotInterval: parseInt(luxriotSnapshotIntervalInput ? luxriotSnapshotIntervalInput.value : config.LUXRIOT_SNAPSHOT_INTERVAL),
                    luxriotSnapshotMaxEdge: parseInt(luxriotSnapshotMaxEdgeInput ? luxriotSnapshotMaxEdgeInput.value : config.LUXRIOT_SNAPSHOT_MAX_EDGE),
                    luxriotMaxBufferFrames: parseInt(luxriotMaxBufferFramesInput ? luxriotMaxBufferFramesInput.value : config.LUXRIOT_MAX_BUFFER_FRAMES),
                    luxriotAutoBookmarks: luxriotAutoBookmarksInput ? luxriotAutoBookmarksInput.checked : false,
                    luxriotSeverityMap: {
                        info: luxriotSevInfoInput ? (luxriotSevInfoInput.value.trim() || 'info') : 'info',
                        low: luxriotSevLowInput ? (luxriotSevLowInput.value.trim() || 'low') : 'low',
                        normal: luxriotSevNormalInput ? (luxriotSevNormalInput.value.trim() || 'normal') : 'normal',
                        high: luxriotSevHighInput ? (luxriotSevHighInput.value.trim() || 'high') : 'high',
                        critical: luxriotSevCriticalInput ? (luxriotSevCriticalInput.value.trim() || 'critical') : 'critical'
                    }
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
                luxriotBaseUrlInput.value = 'http://192.168.1.102:8080';
                luxriotUsernameInput.value = 'admin';
                luxriotPasswordInput.value = '123';
                luxriotDefaultChannelIdInput.value = '103';
                luxriotSnapshotIntervalInput.value = '5';
                luxriotSnapshotMaxEdgeInput.value = '800';
                luxriotMaxBufferFramesInput.value = '180';
                if (luxriotAutoBookmarksInput) luxriotAutoBookmarksInput.checked = false;
                if (luxriotSevInfoInput) luxriotSevInfoInput.value = 'info';
                if (luxriotSevLowInput) luxriotSevLowInput.value = 'low';
                if (luxriotSevNormalInput) luxriotSevNormalInput.value = 'normal';
                if (luxriotSevHighInput) luxriotSevHighInput.value = 'high';
                if (luxriotSevCriticalInput) luxriotSevCriticalInput.value = 'critical';
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

            const textSearchAvailable = embedder !== 'dino';
            searchInput.disabled = !textSearchAvailable;
            searchBtn.disabled = !textSearchAvailable;
            searchInput.placeholder = textSearchAvailable
                ? "Describe what you're looking for..."
                : 'Text search requires CLIP or Fusion backend.';
            searchBtn.title = textSearchAvailable ? '' : 'Text search is disabled when backend is DINO.';
        }

        embedderSelect.addEventListener('change', (event) => {
            applyEmbedderUI(event.target.value);
        });
        applyEmbedderUI(embedderSelect.value);
        if (luxriotRefreshChannelsBtn) {
            luxriotRefreshChannelsBtn.addEventListener('click', () => {
                fetchLuxriotChannels(true).then(syncProbeChannelSelect);
            });
        }
        if (luxriotPreviewBtn) {
            luxriotPreviewBtn.addEventListener('click', () => {
                fetchLuxriotChannels();
                syncProbeChannelSelect();
                startLuxriotPreview();
            });
        }
        if (luxriotStartCaptureBtn) {
            luxriotStartCaptureBtn.addEventListener('click', startLuxriotCapture);
        }
        if (luxriotStopCaptureBtn) {
            luxriotStopCaptureBtn.addEventListener('click', stopLuxriotCapture);
        }
        if (luxriotFlushCaptureBtn) {
            luxriotFlushCaptureBtn.addEventListener('click', flushLuxriotCapture);
        }
        if (luxriotRefreshSummariesBtn) {
            luxriotRefreshSummariesBtn.addEventListener('click', () => refreshLuxriotSummaries());
        }
        if (luxriotRefreshStreamsBtn) {
            luxriotRefreshStreamsBtn.addEventListener('click', () => refreshLuxriotStreams());
        }
        if (luxriotStopAllVideoBtn) {
            luxriotStopAllVideoBtn.addEventListener('click', () => stopAllLuxriotStreams('video'));
        }
        if (luxriotStopAllAnalyticsBtn) {
            luxriotStopAllAnalyticsBtn.addEventListener('click', () => stopAllLuxriotStreams('analytics'));
        }
        if (luxriotSummaries) {
            luxriotSummaries.addEventListener('click', (event) => {
                const target = event.target;
                if (!(target instanceof Element)) return;
                const button = target.closest('[data-luxriot-bookmark]');
                if (!(button instanceof HTMLButtonElement)) return;
                const idx = parseInt(button.dataset.luxriotBookmark || '', 10);
                if (!Number.isFinite(idx)) return;
                event.preventDefault();
                sendLuxriotBookmarkFromLog(idx, button);
            });
        }
        if (luxriotStreams) {
            luxriotStreams.addEventListener('click', (event) => {
                const target = event.target;
                if (!(target instanceof Element)) return;
                const button = target.closest('[data-stream-stop]');
                if (!(button instanceof HTMLButtonElement)) return;
                const channelId = parseInt(button.dataset.streamStop || '', 10);
                const streamType = (button.dataset.streamType || '').trim().toLowerCase();
                if (!Number.isFinite(channelId) || !streamType) return;
                event.preventDefault();
                stopLuxriotStream(channelId, streamType);
            });
        }
        if (luxriotChannelSelect) {
            luxriotChannelSelect.addEventListener('change', () => {
                luxriotActiveChannel = getSelectedLuxriotChannel();
                syncProbeChannelSelect();
                startLuxriotPreview();
                refreshLuxriotSummaries();
                refreshLuxriotStreams();
            });
        }
        
        // -------- Monitoring / Probes --------
        function setProbeStatus(message, isError = false) {
            if (!probeStatus) return;
            probeStatus.textContent = message;
            probeStatus.classList.toggle('error', Boolean(isError));
        }

        function setPreviewState(text, clearImage = false) {
            if (probePreviewOverlay) {
                probePreviewOverlay.style.display = text ? 'flex' : 'none';
                if (text) probePreviewOverlay.textContent = text;
            }
            if (clearImage && probePreviewImg) {
                probePreviewImg.src = '';
            }
        }

        function stopProbePreview() {
            if (probePreviewTimer) {
                clearInterval(probePreviewTimer);
                probePreviewTimer = null;
            }
        }

        function startProbePreview(channelId) {
            if (!probePreviewImg) return;
            stopProbePreview();
            if (!channelId && channelId !== 0) {
                setPreviewState('No channel', true);
                return;
            }
            if (probeStreamState) probeStreamState.textContent = `Streaming channel ${channelId}`;
            const refresh = () => {
                if (probePreviewOverlay) probePreviewOverlay.textContent = 'Loading...';
                probePreviewImg.src = `/luxriot/snapshot/${channelId}?t=${Date.now()}`;
            };
            probePreviewImg.onload = () => setPreviewState('');
            probePreviewImg.onerror = () => setPreviewState('Preview failed');
            refresh();
            const intervalMs = Math.max(2000, (luxriotDefaults.snapshotInterval || 5) * 1000);
            probePreviewTimer = setInterval(refresh, intervalMs);
        }

        function ensurePairsSeed() {
            if (!probePairsState || !probePairsState.length) {
                probePairsState = [
                    { pos: '', neg: '' },
                    { pos: '', neg: '' },
                    { pos: '', neg: '' },
                ];
            }
        }

        function renderPairs() {
            if (!probePairsContainer) return;
            ensurePairsSeed();
            const rows = probePairsState.map((row, idx) => {
                const canRemove = probePairsState.length > 1;
                const removeBtn = canRemove ? `<button class="feature-btn probe-remove-btn" data-remove="${idx}">×</button>` : '<div class="probe-pair-idx">–</div>';
                return `
                    <div class="probe-pair-row" data-idx="${idx}">
                        <div class="probe-pair-idx">${idx + 1}.</div>
                        <input type="text" class="settings-input probe-pos" data-idx="${idx}" value="${escapeHtml(row.pos || '')}" placeholder="Positive probe ${idx + 1}">
                        <input type="text" class="settings-input probe-neg" data-idx="${idx}" value="${escapeHtml(row.neg || '')}" placeholder="Negative probe ${idx + 1}">
                        ${removeBtn}
                    </div>
                `;
            }).join('');
            probePairsContainer.innerHTML = `
                <div class="probe-pairs-header">
                    <div></div>
                    <div>Positive Examples:</div>
                    <div>Negative Examples:</div>
                    <div class="probe-pairs-spacer">&nbsp;</div>
                </div>
                ${rows}
            `;
        }

        function applyImageThumb(base64) {
            if (!probeImageThumb || !probeImageOverlay) return;
            if (base64) {
                probeImageThumb.src = `data:image/jpeg;base64,${base64}`;
                probeImageOverlay.style.display = 'none';
            } else {
                probeImageThumb.src = '';
                probeImageOverlay.style.display = 'flex';
            }
        }

        function updateImageProbeStatus(enabled) {
            imageProbeEnabled = enabled && Boolean(probeImageState?.data);
            if (probeImageEnableBtn) {
                probeImageEnableBtn.textContent = imageProbeEnabled ? 'Disable Image Probe' : 'Enable Image Probe';
            }
            if (probeImageStatus) {
                probeImageStatus.textContent = `Status: ${imageProbeEnabled ? 'Enabled' : 'Disabled'}`;
            }
        }

        function collectProbeForm() {
            const positives = [];
            const negatives = [];
            ensurePairsSeed();
            probePairsState.forEach((row) => {
                if (row.pos?.trim()) positives.push(row.pos.trim());
                if (row.neg?.trim()) negatives.push(row.neg.trim());
            });
            const channelId = parseInt(probeChannelSelect?.value || luxriotActiveChannel, 10);
            return {
                id: activeProbeId,
                name: (probeNameInput?.value || '').trim(),
                channel_id: Number.isFinite(channelId) ? channelId : luxriotActiveChannel,
                pairs: probePairsState.slice(),
                positives,
                negatives,
                pos_floor: parseFloat(probePosFloorInput?.value) || 0.2,
                margin: parseFloat(probeMarginInput?.value) || 0.05,
                top_k: parseInt(probeTopKInput?.value || '6', 10) || 6,
                window_sec: parseFloat(probeWindowSec?.value) || 300,
                fps: parseFloat(probeFps?.value) || 0,
                severity: probeBookmarkSeverityInput ? probeBookmarkSeverityInput.value : 'critical',
                bookmark: probeBookmarkToggle ? probeBookmarkToggle.checked : true,
                enabled: probeEnableToggle ? probeEnableToggle.checked : true,
                image_probe: {
                    data: probeImageState?.data,
                    name: probeImageState?.name,
                    pos_floor: probeImagePosInput ? (parseFloat(probeImagePosInput.value) || 0.7) : 0.7,
                    enabled: imageProbeEnabled,
                },
            };
        }

        function probeHitsKey(probeId = activeProbeId) {
            return probeId ? `probe:${probeId}` : 'probe:draft';
        }

        function renderProbeHitsSlice(hits) {
            if (!hits || !hits.length) {
                return '<div class="loading">No matches</div>';
            }
            return hits.map((hit) => {
                const ts = hit.timestamp_ms ? new Date(hit.timestamp_ms).toLocaleString() : 'n/a';
                return `
                    <div class="probe-result">
                        ${hit.thumbnail ? `<img src="data:image/jpeg;base64,${hit.thumbnail}" alt="probe hit" />` : ''}
                        <div class="probe-result-time">${escapeHtml(ts)}</div>
                        <div class="probe-result-score">P ${(hit.pos_score || 0).toFixed(3)} · N ${(hit.neg_score || 0).toFixed(3)} · M ${(hit.margin || 0).toFixed(3)}</div>
                    </div>
                `;
            }).join('');
        }

        function renderProbeHitsPage(key = probeHitsKey()) {
            const pageSize = 5;
            const allHits = probeHitsCacheByKey[key] || [];
            const total = allHits.length;
            if (!Number.isFinite(probeHitsOffsetByKey[key])) probeHitsOffsetByKey[key] = 0;
            if (probeHitsOffsetByKey[key] > Math.max(0, total - 1)) {
                probeHitsOffsetByKey[key] = 0;
            }
            const offset = probeHitsOffsetByKey[key];
            const pageSlice = allHits.slice(offset, offset + pageSize);
            if (probeResults) {
                probeResults.innerHTML = renderProbeHitsSlice(pageSlice);
            }
            lastProbeRefresh = probeHitsUpdatedByKey[key] || Date.now();
            if (probeHitsMeta) {
                const tsLabel = new Date(lastProbeRefresh).toLocaleTimeString();
                const pageIdx = total ? Math.floor(offset / pageSize) + 1 : 1;
                const pageCount = Math.max(1, Math.ceil(total / pageSize));
                const frames = probeFramesByKey[key] || 0;
                probeHitsMeta.textContent = `Frames: ${frames} · Hits: ${total} · Page: ${pageIdx}/${pageCount} · Updated: ${tsLabel}`;
            }
            if (probeDetLeftBtn) {
                probeDetLeftBtn.disabled = offset <= 0;
            }
            if (probeDetRightBtn) {
                probeDetRightBtn.disabled = offset + pageSize >= total;
            }
        }

        function renderProbeHits(hits = [], framesIndexed = 0, windowSec = null, options = {}) {
            const key = options.key || probeHitsKey();
            const replace = options.replace === true;
            const now = Date.now();
            const parsedWindow = Number.parseFloat(windowSec);
            const effectiveWindowSec = Number.isFinite(parsedWindow)
                ? parsedWindow
                : Number.parseFloat(probeWindowSecByKey[key]);
            if (Number.isFinite(effectiveWindowSec) && effectiveWindowSec > 0) {
                probeWindowSecByKey[key] = effectiveWindowSec;
            }
            const minTs = Number.isFinite(effectiveWindowSec) && effectiveWindowSec > 0
                ? now - (effectiveWindowSec * 1000)
                : null;
            const merged = new Map();
            const addHit = (hit) => {
                if (!hit) return;
                if (minTs && hit.timestamp_ms && hit.timestamp_ms < minTs) return;
                const dedupeKey = `${hit.timestamp_ms || 0}-${(hit.pos_score || 0).toFixed(3)}-${(hit.neg_score || 0).toFixed(3)}-${(hit.margin || 0).toFixed(3)}`;
                merged.set(dedupeKey, hit);
            };
            if (!replace) {
                (probeHitsCacheByKey[key] || []).forEach(addHit);
            }
            (hits || []).forEach(addHit);
            const combined = Array.from(merged.values())
                .sort((a, b) => (b.timestamp_ms || 0) - (a.timestamp_ms || 0))
                .slice(0, 50);
            probeHitsCacheByKey[key] = combined;
            probeFramesByKey[key] = Number.isFinite(framesIndexed) ? framesIndexed : (probeFramesByKey[key] || 0);
            probeHitsUpdatedByKey[key] = now;
            if (options.resetOffset !== false) {
                probeHitsOffsetByKey[key] = 0;
            }
            if (key === probeHitsKey()) {
                renderProbeHitsPage(key);
            }
        }

        function probeActionIcon(action) {
            const icons = {
                expand: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="M240-240v-240h72v168h168v72H240Zm408-240v-168H480v-72h240v240h-72Z"/></svg>',
                run: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="m380-300 280-180-280-180v360Z"/></svg>',
                enable: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="m424-296 282-282-56-56-226 226-114-114-56 56 170 170Z"/></svg>',
                disable: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="M520-200v-560h160v560H520Zm-240 0v-560h160v560H280Z"/></svg>',
                delete: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="M280-120q-33 0-56.5-23.5T200-200v-520h-40v-80h200v-40h240v40h200v80h-40v520q0 33-23.5 56.5T680-120H280Zm400-600H280v520h400v-520ZM360-280h80v-360h-80v360Zm160 0h80v-360h-80v360Z"/></svg>',
                new: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960"><path d="M440-440H200v-80h240v-240h80v240h240v80H520v240h-80v-240Z"/></svg>',
            };
            return icons[action] || '';
        }

        function renderProbeCards() {
            if (!probeCards) return;
            if (!probeList.length) {
                probeCards.innerHTML = `
                    <div class="probe-mini-card new-probe-card">
                        <button class="probe-new-btn" data-action="new" aria-label="Create probe" title="Create probe">
                            ${probeActionIcon('new')}
                            <span>New Probe</span>
                        </button>
                    </div>`;
                return;
            }
            const cards = probeList.map((p) => {
                const last = p.last_hit;
                const ts = last?.timestamp_ms ? new Date(last.timestamp_ms).toLocaleTimeString() : 'n/a';
                const status = p.enabled === false ? 'disabled' : (p.enabled ? 'running' : 'idle');
                const pillClass = status === 'disabled' ? 'pill-disabled' : status === 'running' ? 'pill-running' : 'pill-idle';
                const thumbSrc = last?.thumbnail || p.image_probe?.data || '';
                const toggleAction = status === 'disabled' ? 'enable' : 'disable';
                const toggleTitle = status === 'disabled' ? 'Enable probe' : 'Disable probe';
                const scores = `P: ${Number.isFinite(last?.pos_score) ? last.pos_score.toFixed(3) : '—'} · N: ${Number.isFinite(last?.neg_score) ? last.neg_score.toFixed(3) : '—'} · M: ${Number.isFinite(last?.margin) ? last.margin.toFixed(3) : '—'}`;
                return `
                    <div class="probe-mini-card ${activeProbeId === p.id ? 'active' : ''}">
                        <div class="probe-mini-thumb ${thumbSrc ? '' : 'is-empty'}">
                            ${thumbSrc ? `<img src="data:image/jpeg;base64,${thumbSrc}" alt="${escapeHtml(p.name || 'probe preview')}" />` : ''}
                            <div class="probe-mini-overlay">
                                <div class="probe-mini-top">
                                    <div class="probe-status-pill ${pillClass}">${status}</div>
                                    <div class="probe-mini-actions">
                                        <button class="probe-action-btn" data-action="expand" data-id="${p.id}" title="Open probe" aria-label="Open probe">${probeActionIcon('expand')}</button>
                                        <button class="probe-action-btn" data-action="run" data-id="${p.id}" title="Run probe" aria-label="Run probe">${probeActionIcon('run')}</button>
                                        <button class="probe-action-btn" data-action="${toggleAction}" data-id="${p.id}" title="${toggleTitle}" aria-label="${toggleTitle}">${probeActionIcon(toggleAction)}</button>
                                        <button class="probe-action-btn delete" data-action="delete" data-id="${p.id}" title="Delete probe" aria-label="Delete probe">${probeActionIcon('delete')}</button>
                                    </div>
                                </div>
                                <div class="probe-mini-bottom">
                                    <div class="probe-mini-name">${escapeHtml(p.name || 'unnamed')}</div>
                                    <div class="probe-mini-meta">Ch ${p.channel_id || luxriotActiveChannel} · Last ${last ? ts : 'n/a'}</div>
                                    <div class="probe-mini-score">${scores}</div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            });
            cards.push(`
                <div class="probe-mini-card new-probe-card">
                    <button class="probe-new-btn" data-action="new" aria-label="Create probe" title="Create probe">
                        ${probeActionIcon('new')}
                        <span>New Probe</span>
                    </button>
                </div>
            `);
            probeCards.innerHTML = cards.join('');
        }

        function setActiveProbe(probe) {
            activeProbeId = probe && probe.id ? probe.id : null;
            if (probeNameInput) probeNameInput.value = (probe && probe.name) || '';
            if (probeChannelSelect && probe && probe.channel_id) {
                probeChannelSelect.value = probe.channel_id;
                startProbePreview(probe.channel_id);
            }
            if (probePosFloorInput) probePosFloorInput.value = probe?.pos_floor ?? 0.2;
            if (probeMarginInput) probeMarginInput.value = probe?.margin ?? 0.05;
            if (probeFps) probeFps.value = probe?.fps ?? 0;
            if (probeWindowSec) probeWindowSec.value = probe?.window_sec ?? 300;
            if (probeBookmarkSeverityInput) probeBookmarkSeverityInput.value = probe?.severity || 'critical';
            if (probeBookmarkToggle) probeBookmarkToggle.checked = probe?.bookmark !== false;
            if (probeEnableToggle) probeEnableToggle.checked = probe?.enabled !== false;
            probePairsState = (probe?.pairs && Array.isArray(probe.pairs) ? probe.pairs : null) || (probe ? [] : probePairsState);
            if (probe?.image_probe?.data) {
                probeImageState = { data: probe.image_probe.data, name: probe.image_probe.name };
                applyImageThumb(probe.image_probe.data);
                if (probeImagePosInput) probeImagePosInput.value = probe.image_probe.pos_floor || 0.7;
                const enabled = probe.image_probe.enabled !== false;
                updateImageProbeStatus(enabled);
            } else {
                probeImageState = null;
                applyImageThumb('');
                updateImageProbeStatus(false);
            }
            renderPairs();
            const initialHits = Array.isArray(probe?.recent_hits) && probe.recent_hits.length
                ? probe.recent_hits
                : (probe?.last_hit ? [probe.last_hit] : []);
            const key = probeHitsKey(activeProbeId);
            renderProbeHits(
                initialHits,
                initialHits.length || (probe?.last_hit ? 1 : 0),
                probe?.window_sec ?? null,
                { key, replace: true, resetOffset: true }
            );
            renderProbeCards();
            setProbeStatus(activeProbeId ? `Editing: ${probe?.name || probe?.id}` : 'New probe');
        }

        function updateRunButton(running) {
            if (!probeRunBtn) return;
            probeRunBtn.textContent = running ? 'Stop probe' : 'Run probe';
            probeRunBtn.classList.toggle('primary', running);
        }

        async function persistProbeEnabled(enabled) {
            if (!activeProbeId) return;
            const payload = collectProbeForm();
            payload.id = activeProbeId;
            payload.enabled = enabled;
            try {
                const resp = await fetch('/probes/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                const data = await resp.json();
                if (!resp.ok || data.error) throw new Error(data.error || 'Save failed');
                await loadProbeList();
                if (probeEnableToggle) probeEnableToggle.checked = enabled;
            } catch (err) {
                setProbeStatus(err.message, true);
            }
        }

        function stopProbeRunLoop(message) {
            if (probeRunTimer) {
                clearInterval(probeRunTimer);
                probeRunTimer = null;
            }
            updateRunButton(false);
            if (message) setProbeStatus(message);
        }

        async function loadProbeList(showStatus = false) {
            try {
                const resp = await fetch('/probes/list');
                const data = await resp.json();
                probeList = data.probes || [];
                if (showStatus) setProbeStatus(`Loaded ${probeList.length} probes`);
                const match = activeProbeId ? probeList.find(p => p.id === activeProbeId) : null;
                if (match) {
                    setActiveProbe(match);
                } else if (!activeProbeId && probeList.length) {
                    setActiveProbe(probeList[0]);
                } else {
                    renderProbeHits([], 0, null, { key: probeHitsKey(activeProbeId), replace: true, resetOffset: true });
                    renderProbeCards();
                }
            } catch (err) {
                setProbeStatus('Failed to load probes: ' + err.message, true);
            }
        }

        async function ensureProbeCapture(channelId, quiet = false) {
            if (!channelId && channelId !== 0) return;
            if (probeCaptureState[channelId]) {
                if (probeCaptureStatus && !quiet) probeCaptureStatus.textContent = `Streaming channel ${channelId}`;
                setPreviewState('');
                startProbePreview(channelId);
                return;
            }
            try {
                channelCaptureConfig[channelId] = {
                    fps: parseFloat(probeFps?.value) || 0,
                    windowSec: parseFloat(probeWindowSec?.value) || 300,
                };
                const resp = await fetch('/probes/start_capture', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ channel_id: channelId, fps: channelCaptureConfig[channelId].fps })
                });
                const data = await resp.json();
                if (!resp.ok || data.error) throw new Error(data.error || 'Failed to start capture');
                probeCaptureState[channelId] = true;
                if (probeCaptureStatus) probeCaptureStatus.textContent = `Streaming channel ${channelId}`;
                setPreviewState('');
                startProbePreview(channelId);
            } catch (err) {
                if (probeCaptureStatus) probeCaptureStatus.textContent = err.message;
                if (!quiet) setProbeStatus(err.message, true);
            }
        }

        async function stopProbeCapture(channelId, reason = 'stopped') {
            if (!channelId && channelId !== 0) return;
            try {
                const resp = await fetch('/probes/stop_capture', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ channel_id: channelId })
                });
                const data = await resp.json();
                if (!resp.ok || data.error) throw new Error(data.error || 'Failed to stop capture');
                delete probeCaptureState[channelId];
                if (reason === 'paused') {
                    setPreviewState('Paused');
                    if (probeCaptureStatus) probeCaptureStatus.textContent = 'Paused';
                } else {
                    setPreviewState('Stopped', true);
                    if (probeCaptureStatus) probeCaptureStatus.textContent = 'Stream stopped';
                }
                if (reason !== 'paused') stopProbePreview();
            } catch (err) {
                if (probeCaptureStatus) probeCaptureStatus.textContent = err.message;
                setProbeStatus(err.message, true);
            }
        }

        async function refreshProbeStatus(channelIdOverride) {
            const channelId = channelIdOverride || parseInt(probeChannelSelect?.value || luxriotActiveChannel, 10);
            try {
                const resp = await fetch(`/probes/status?channel_id=${channelId}`);
                const data = await resp.json();
                if (data.error) {
                    setProbeStatus(data.error, true);
                    return;
                }
                const range = data.time_range_ms && data.time_range_ms.length === 2
                    ? `${new Date(data.time_range_ms[0]).toLocaleTimeString()} - ${new Date(data.time_range_ms[1]).toLocaleTimeString()}`
                    : 'n/a';
                setProbeStatus(`Frames: ${data.frames || 0} · Range: ${range}`);
                if (probeCaptureStatus) {
                    probeCaptureStatus.textContent = data.frames ? `Streaming channel ${channelId}` : 'Stream idle';
                }
                if (probeBufferInfo) {
                    const lastTs = data.last_timestamp_ms ? new Date(data.last_timestamp_ms).toLocaleTimeString() : 'n/a';
                    probeBufferInfo.textContent = `Last snapshot: ${lastTs}`;
                }
                if (probeStreamState) {
                    const pill = data.frames ? `Streaming channel ${channelId}` : 'Stream idle';
                    probeStreamState.textContent = pill;
                }
            } catch (err) {
                setProbeStatus('Status error: ' + err.message, true);
            }
        }

        async function saveActiveProbe() {
            const payload = collectProbeForm();
            const hasPos = payload.positives.length > 0 || (payload.image_probe?.enabled && payload.image_probe?.data);
            if (!hasPos) {
                setProbeStatus('Add a text positive or enable an image probe.', true);
                return;
            }
            setProbeStatus('Saving...');
            try {
                const resp = await fetch('/probes/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                const data = await resp.json();
                if (!resp.ok || data.error) {
                    throw new Error(data.error || 'Save failed');
                }
                const saved = data.probe;
                activeProbeId = saved.id || activeProbeId;
            setProbeStatus(`Saved probe ${saved.name || saved.id}`);
            await loadProbeList();
            await ensureProbeCapture(saved.channel_id || payload.channel_id, true);
            } catch (err) {
                setProbeStatus(err.message, true);
            }
            return activeProbeId;
        }

        async function runActiveProbe(quiet = false) {
            if (probeRunInFlight) return;
            const payload = collectProbeForm();
            const hasPos = payload.positives.length > 0 || (payload.image_probe?.enabled && payload.image_probe?.data);
            if (!hasPos) {
                setProbeStatus('Add a text positive or enable an image probe.', true);
                if (probeRunTimer) stopProbeRunLoop();
                return;
            }
            const channelId = payload.channel_id;
            await ensureProbeCapture(channelId, true);
            if (!quiet) setProbeStatus('Running...');
            probeRunInFlight = true;
            try {
                let resp;
                if (activeProbeId) {
                    resp = await fetch('/probes/run', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ id: activeProbeId })
                    });
                } else {
                    resp = await fetch('/probes/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                }
                const data = await resp.json();
                if (!resp.ok || data.error) throw new Error(data.error || 'Probe failed');
                const hits = data.results || [];
                const framesCount = data.frames_indexed || data.status?.frames || 0;
                renderProbeHits(hits, framesCount, payload.window_sec);
                if (data.probe) {
                    activeProbeId = data.probe.id || activeProbeId;
                    await loadProbeList();
                } else {
                    renderProbeCards();
                }
                if (!quiet) setProbeStatus(`Hits: ${hits.length} · Frames: ${framesCount}`);
            } catch (err) {
                renderProbeHits([], 0);
                setProbeStatus(err.message, true);
            } finally {
                probeRunInFlight = false;
            }
        }

        function startProbeRunLoop(quiet = false) {
            stopProbeRunLoop();
            updateRunButton(true);
            runActiveProbe(quiet);
            const windowSec = parseFloat(probeWindowSec?.value) || 30;
            const intervalMs = Math.max(2000, Math.min(10000, (windowSec * 1000) / 2));
            probeRunTimer = setInterval(() => runActiveProbe(true), intervalMs);
            persistProbeEnabled(true);
        }

        function startProbeStatusPoll() {
            if (probeStatusTimer) return;
            refreshProbeStatus();
            probeStatusTimer = setInterval(() => refreshProbeStatus(), 8000);
        }

        function stopProbeStatusPoll() {
            if (probeStatusTimer) {
                clearInterval(probeStatusTimer);
                probeStatusTimer = null;
            }
        }

        async function deleteProbe(id) {
            if (!id) {
                setProbeStatus('No probe selected', true);
                return;
            }
            try {
                const resp = await fetch('/probes/delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ id })
                });
                const data = await resp.json();
                if (!resp.ok || data.error) throw new Error(data.error || 'Delete failed');
                if (activeProbeId === id) activeProbeId = null;
                setProbeStatus('Probe deleted');
                await loadProbeList(true);
                stopProbeRunLoop();
            } catch (err) {
                setProbeStatus(err.message, true);
            }
        }

        function handleProbeCardClick(event) {
            const btn = event.target.closest('button[data-action]');
            if (!btn) return;
            const id = btn.getAttribute('data-id');
            const action = btn.getAttribute('data-action');
            const probe = probeList.find(p => String(p.id) === String(id));
            if (!action) return;
            if (action === 'expand' && probe) {
                setActiveProbe(probe);
                if (probeEditorModal) {
                    probeEditorModal.style.display = 'block';
                }
            } else if (action === 'run' && probe) {
                setActiveProbe(probe);
                startProbeRunLoop();
            } else if (action === 'enable' && probe) {
                setActiveProbe(probe);
                persistProbeEnabled(true);
                ensureProbeCapture(probe.channel_id || luxriotActiveChannel, true);
            } else if (action === 'delete') {
                deleteProbe(id);
            } else if (action === 'disable' && probe) {
                setActiveProbe(probe);
                persistProbeEnabled(false);
                stopProbeRunLoop();
            } else if (action === 'new') {
                activeProbeId = null;
                probePairsState = [];
                probeImageState = null;
                applyImageThumb('');
                renderPairs();
                renderProbeHits([], 0, null, { key: probeHitsKey(null), replace: true, resetOffset: true });
                if (probeEnableToggle) probeEnableToggle.checked = true;
                setProbeStatus('New probe');
                if (probeEditorModal) {
                    probeEditorModal.style.display = 'block';
                }
            }
        }

        if (probeRunBtn) {
            probeRunBtn.addEventListener('click', () => {
                if (probeRunTimer) {
                    stopProbeRunLoop('Stopped probe loop');
                    persistProbeEnabled(false);
                } else {
                    if (!activeProbeId) {
                        saveActiveProbe().then(() => startProbeRunLoop());
                    } else {
                        startProbeRunLoop();
                    }
                }
            });
        }
        if (probeSaveBtn) {
            probeSaveBtn.addEventListener('click', async () => {
                const savedId = await saveActiveProbe();
                if (savedId && probeEditorModal) {
                    probeEditorModal.style.display = 'none';
                }
            });
        }
        if (probeDeleteBtn) probeDeleteBtn.addEventListener('click', () => {
            if (activeProbeId) deleteProbe(activeProbeId);
            else {
                probePairsState = [];
                probeImageState = null;
                applyImageThumb('');
                renderPairs();
                renderProbeHits([], 0, null, { key: probeHitsKey(null), replace: true, resetOffset: true });
                setProbeStatus('Cleared unsaved probe');
            }
        });
        if (probeStartCaptureBtn) {
            probeStartCaptureBtn.addEventListener('click', () => ensureProbeCapture(parseInt(probeChannelSelect?.value || luxriotActiveChannel, 10)));
        }
        if (probeStopCaptureBtn) {
            probeStopCaptureBtn.addEventListener('click', () => stopProbeCapture(parseInt(probeChannelSelect?.value || luxriotActiveChannel, 10), 'paused'));
        }
        if (probeStopAllBtn) {
            probeStopAllBtn.addEventListener('click', () => {
                Object.keys(probeCaptureState).forEach((cid) => stopProbeCapture(parseInt(cid, 10), 'stopped'));
                stopProbeRunLoop();
            });
        }
        if (probeChannelSelect) {
            probeChannelSelect.addEventListener('change', () => {
                const cid = parseInt(probeChannelSelect.value || luxriotActiveChannel, 10);
                startProbePreview(cid);
            });
        }
        if (probeCards) {
            probeCards.addEventListener('click', handleProbeCardClick);
        }
        if (probeNewBtn) {
            probeNewBtn.addEventListener('click', () => {
                activeProbeId = null;
                probePairsState = [];
                probeImageState = null;
                applyImageThumb('');
                renderPairs();
                renderProbeHits([], 0, null, { key: probeHitsKey(null), replace: true, resetOffset: true });
                setProbeStatus('New probe');
                if (probeEnableToggle) probeEnableToggle.checked = true;
                if (probeEditorModal) {
                    probeEditorModal.style.display = 'block';
                }
            });
        }
        if (probeReloadBtn) {
            probeReloadBtn.addEventListener('click', () => loadProbeList(true));
        }
        if (probeImageFile) {
            probeImageFile.addEventListener('change', () => {
                const file = probeImageFile.files && probeImageFile.files[0];
                if (!file) return;
                const reader = new FileReader();
                reader.onload = () => {
                    const base64 = reader.result.split(',')[1];
                    probeImageState = { name: file.name, data: base64 };
                    applyImageThumb(base64);
                    updateImageProbeStatus(imageProbeEnabled);
                };
                reader.readAsDataURL(file);
            });
        }
        if (probeAddPairBtn && probePairsContainer) {
            probeAddPairBtn.addEventListener('click', () => {
                probePairsState.push({ pos: '', neg: '' });
                renderPairs();
            });
            probePairsContainer.addEventListener('input', (e) => {
                const target = e.target;
                const idx = parseInt(target.getAttribute('data-idx') || '-1', 10);
                if (!Number.isFinite(idx) || idx < 0 || idx >= probePairsState.length) return;
                if (target.classList.contains('probe-pos')) {
                    probePairsState[idx].pos = target.value;
                } else if (target.classList.contains('probe-neg')) {
                    probePairsState[idx].neg = target.value;
                }
            });
            probePairsContainer.addEventListener('click', (e) => {
                const btn = e.target.closest('button[data-remove]');
                if (!btn) return;
                const idx = parseInt(btn.getAttribute('data-remove') || '-1', 10);
                if (!Number.isFinite(idx) || idx < 0 || probePairsState.length <= 1) return;
                probePairsState.splice(idx, 1);
                renderPairs();
            });
        }
        if (probeDetLeftBtn && probeResults) {
            probeDetLeftBtn.addEventListener('click', () => {
                const key = probeHitsKey();
                const allHits = probeHitsCacheByKey[key] || [];
                if (!allHits.length) return;
                const pageSize = 5;
                const currentOffset = Number.isFinite(probeHitsOffsetByKey[key]) ? probeHitsOffsetByKey[key] : 0;
                probeHitsOffsetByKey[key] = Math.max(0, currentOffset - pageSize);
                renderProbeHitsPage(key);
            });
        }
        if (probeDetRightBtn && probeResults) {
            probeDetRightBtn.addEventListener('click', () => {
                const key = probeHitsKey();
                const allHits = probeHitsCacheByKey[key] || [];
                if (!allHits.length) return;
                const pageSize = 5;
                const currentOffset = Number.isFinite(probeHitsOffsetByKey[key]) ? probeHitsOffsetByKey[key] : 0;
                if (currentOffset + pageSize < allHits.length) {
                    probeHitsOffsetByKey[key] = currentOffset + pageSize;
                }
                renderProbeHitsPage(key);
            });
        }
        if (probeEnableToggle) {
            probeEnableToggle.addEventListener('change', (e) => {
                const enabled = e.target.checked;
                persistProbeEnabled(enabled);
                if (enabled) {
                    ensureProbeCapture(parseInt(probeChannelSelect?.value || luxriotActiveChannel, 10), true);
                    runActiveProbe(true);
                } else {
                    stopProbeRunLoop('Probe disabled');
                }
            });
        }
        if (probeImageEnableBtn) {
            probeImageEnableBtn.addEventListener('click', () => {
                if (!probeImageState?.data) {
                    setProbeStatus('Select an image first.', true);
                    return;
                }
                updateImageProbeStatus(!imageProbeEnabled);
            });
        }
        if (probeBenchBtn && probeBenchOutput) {
            probeBenchBtn.addEventListener('click', async () => {
                probeBenchBtn.disabled = true;
                probeBenchOutput.textContent = 'Benchmark running...';
                try {
                    const resp = await fetch('/probes/bench');
                    const data = await resp.json();
                    if (!resp.ok || data.error) throw new Error(data.error || 'Benchmark failed');
                    probeBenchOutput.textContent = `~${data.approx_fps} fps @ batch ${data.batch} on ${data.device} (elapsed ${data.elapsed_sec}s)`;
                } catch (err) {
                    probeBenchOutput.textContent = `Benchmark failed: ${err.message}`;
                } finally {
                    probeBenchBtn.disabled = false;
                }
            });
        }

        // Mode switching
        if (archiveModeBtn) archiveModeBtn.addEventListener('click', () => setMode('archive'));
        if (videoModeBtn) videoModeBtn.addEventListener('click', () => setMode('video'));
        if (monitorModeBtn) monitorModeBtn.addEventListener('click', () => setMode('monitor'));
        
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

        async function parseApiJson(response, fallbackMessage) {
            let data = {};
            try {
                data = await response.json();
            } catch (_) {
                data = {};
            }
            if (!response.ok || data.error) {
                const message = data.error || `${fallbackMessage} (${response.status})`;
                throw new Error(message);
            }
            return data;
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
            setMode('archive');
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
                
                const data = await parseApiJson(response, 'Text search failed');
                
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
            setMode('archive');
            const folder = folderInput.value.trim();
            const file = imageUpload.files[0];
            const limit = resultLimitSelect.value;
            const sortBy = sortBySelect.value;
            
            if (!folder || !file) {
                alert('Please select a folder and upload an image file.');
                return;
            }
            
            resultsContainer.innerHTML = '<div class="loading"><div class="spinner"></div> Searching by image...</div>';
            
            try {
                const formData = new FormData();
                formData.append('folder', folder);
                formData.append('limit', limit);
                formData.append('sort_by', sortBy);
                formData.append('image', file);
                
                const response = await fetch('/search_by_image', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await parseApiJson(response, 'Image search failed');
                
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
            const modelId = videoModelInput ? videoModelInput.value.trim() : '';

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
                if (modelId) {
                    payload.model = modelId;
                }
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
                videoStatus.dataset.base = `Model: ${data.model || modelId || 'LM Studio'} · Frames sent: ${(data.frames || []).length || frameCount}${durationLabel}`;
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
            setMode('archive');
            
            resultsContainer.innerHTML = '<div class="loading"><div class="spinner"></div> Loading commented images...</div>';
            
            try {
                const response = await fetch('/commented_images', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ folder })
                });
                
                const data = await parseApiJson(response, 'Loading commented images failed');
                
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
            const safeFilename = escapeHtml(result.filename || 'unnamed');
            const rawPath = String(result.path || '');
            const safePath = escapeHtml(rawPath);
            const thumb = String(result.thumbnail || '').trim();
            const fallbackSvg = encodeURIComponent(
                '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="260">' +
                '<rect width="100%" height="100%" fill="#1f2026"/>' +
                '<text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#9aa0ad" font-size="18">No thumbnail</text>' +
                '</svg>'
            );
            const thumbnailSrc = thumb ? `data:image/jpeg;base64,${thumb}` : `data:image/svg+xml;charset=utf-8,${fallbackSvg}`;
                
            return `
                <div class="image-container">
                    <img src="${thumbnailSrc}" class="thumbnail" alt="" />
                    <div class="image-overlay">
                        <div class="expand-collapse-icon" data-index="${index}">
                            <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="#e3e3e3">
                                <path d="M240-240v-240h72v168h168v72H240Zm408-240v-168H480v-72h240v240h-72Z"/>
                            </svg>
                        </div>
                    </div>
                </div>
                <div class="result-info">
                    <div class="filename">
                        ${safeFilename}
                        <svg class="copy-icon" xmlns="http://www.w3.org/2000/svg" height="16px" viewBox="0 -960 960 960" width="16px" fill="#888">
                            <path d="M360-240q-29.7 0-50.85-21.15Q288-282.3 288-312v-480q0-29.7 21.15-50.85Q330.3-864 360-864h384q29.7 0 50.85 21.15Q816-821.7 816-792v480q0 29.7-21.15 50.85Q773.7-240 744-240H360Zm0-72h384v-480H360v480ZM216-96q-29.7 0-50.85-21.15Q144-138.3 144-168v-552h72v552h456v72H216Zm144-216v-480 480Z"/>
                        </svg>
                    </div>
                    <div class="similarity">${similarityMarkup}</div>
                    <div class="result-actions">
                        <button class="action-icon describe-icon" data-index="${index}" data-path="${safePath}" title="Describe with LM">
                            <svg xmlns="http://www.w3.org/2000/svg" height="18px" viewBox="0 -960 960 960" width="18px" fill="#e3e3e3">
                                <path d="M160-120q-33 0-56.5-23.5T80-200v-560q0-33 23.5-56.5T160-840h545q33 0 56.5 23.5T785-760v160h-80v-160H160v560h545v-160h80v160q0 33-23.5 56.5T705-120H160Zm520-240 57-57-143-143 143-143-57-57-143 143-143-143-57 57 143 143-143 143 57 57 143-143 143 143Z"/>
                            </svg>
                        </button>
                        <button class="action-icon find-similar-icon" data-index="${index}" data-path="${safePath}" title="Find similar">
                            <svg xmlns="http://www.w3.org/2000/svg" height="18px" viewBox="0 -960 960 960" width="18px" fill="#e3e3e3">
                                <path d="M784-120 532-372q-30 24-69 38t-83 14q-109 0-184.5-75.5T120-580q0-109 75.5-184.5T380-840q109 0 184.5 75.5T640-580q0 44-14 83t-38 69l252 252-56 56ZM380-400q75 0 127.5-52.5T560-580q0-75-52.5-127.5T380-760q-75 0-127.5 52.5T200-580q0 75 52.5 127.5T380-400Z"/>
                            </svg>
                        </button>
                    </div>
                </div>
                <div class="segments-panel" id="segments-${index}">
                    <div class="segments-status warning">Segments disabled. Enable in settings to propose regions.</div>
                </div>
                <div class="comment-section">
                    <div class="lm-description" id="lm-desc-${index}">
                        <div class="no-comments">No LLM description yet.</div>
                    </div>
                    <div class="lm-description-actions">
                        <button class="save-comment-btn is-hidden" id="lm-save-btn-${index}">Save LLM as comment</button>
                    </div>
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
                        describeImageWithLM(index, result.path, item, result);
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

            const lmSaveBtn = item.querySelector(`#lm-save-btn-${index}`);
            if (lmSaveBtn) {
                if (result.path) {
                    lmSaveBtn.addEventListener('click', () => {
                        saveLmDescriptionAsComment(index, result.path);
                    });
                } else {
                    lmSaveBtn.style.display = 'none';
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
            segmentContextByIndex = {};
            
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
            segmentContextByIndex = {};
            
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

        function renderLmDescription(index, summary, modelLabel = '') {
            const descContainer = document.getElementById(`lm-desc-${index}`);
            const saveBtn = document.getElementById(`lm-save-btn-${index}`);
            if (!descContainer || !saveBtn) return;

            const now = new Date().toLocaleString();
            const modelSuffix = modelLabel ? ` · ${escapeHtml(modelLabel)}` : '';
            descContainer.innerHTML = `
                <div class="comment-item lm-comment">
                    <div class="comment-timestamp">LLM Description${modelSuffix} · ${escapeHtml(now)}</div>
                    <div class="comment-text">${renderMarkdown(summary || '')}</div>
                </div>
            `;
            saveBtn.dataset.summary = summary || '';
            saveBtn.style.display = 'inline-flex';
            saveBtn.disabled = false;
            saveBtn.textContent = 'Save LLM as comment';
        }

        async function saveLmDescriptionAsComment(index, imagePath) {
            const saveBtn = document.getElementById(`lm-save-btn-${index}`);
            if (!saveBtn) return;
            const summary = (saveBtn.dataset.summary || '').trim();
            if (!summary) {
                alert('No LLM description to save yet.');
                return;
            }
            const folder = folderInput.value.trim();
            if (!folder) {
                alert('Please enter a folder path first.');
                return;
            }

            const originalText = saveBtn.textContent;
            saveBtn.disabled = true;
            saveBtn.textContent = 'Saving...';
            try {
                const response = await fetch('/comments', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        folder,
                        image_path: imagePath,
                        comment: summary,
                    }),
                });
                const data = await parseApiJson(response, 'Saving LLM description failed');
                const commentsContainer = document.getElementById(`comments-${index}`);
                if (commentsContainer && Array.isArray(data.comments)) {
                    displayComments(commentsContainer, data.comments);
                }
                indexStatus.textContent = 'LLM description saved as comment.';
                indexStatus.className = 'status success';
                saveBtn.textContent = 'Saved';
            } catch (err) {
                alert('Failed to save LLM description: ' + err.message);
                saveBtn.textContent = originalText;
            } finally {
                saveBtn.disabled = false;
            }
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
                delete segmentContextByIndex[index];
                img.classList.remove('segment-enabled');
                // Update icon to expand
                expandCollapseIcon.innerHTML = `
                    <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="#e3e3e3">
                        <path d="M240-240v-240h72v168h168v72H240Zm408-240v-168H480v-72h240v240h-72Z"/>
                    </svg>
                `;
            } else {
                // Expand: show original image and load comments
                const activeFolder = folderInput.value.trim();
                const params = new URLSearchParams();
                if (activeFolder) {
                    params.set('folder', activeFolder);
                }
                params.set('image_path', result.path || '');
                const originalImageUrl = `/image?${params.toString()}`;
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

        function stripBase64Payload(rawValue) {
            const text = String(rawValue || '').trim();
            if (!text) return '';
            if (text.startsWith('data:')) {
                const comma = text.indexOf(',');
                return comma >= 0 ? text.slice(comma + 1) : '';
            }
            return text;
        }

        function extractSegmentMeta(segments) {
            const ids = [];
            const labels = {};
            (segments || []).forEach((segment) => {
                if (!segment || segment.segment_id === undefined || segment.segment_id === null) return;
                const segId = String(segment.segment_id).trim();
                if (!segId) return;
                ids.push(segId);
                if (segment.label !== undefined && segment.label !== null) {
                    const label = String(segment.label).trim();
                    if (label) {
                        labels[segId] = label;
                    }
                }
            });
            return {
                segmentIds: [...new Set(ids)],
                segmentLabels: labels,
            };
        }

        function showSegmentPanelNotice(panel, message, level = 'success') {
            if (!panel) return;
            const safeLevel = ['success', 'warning', 'error'].includes(level) ? level : 'success';
            const notice = document.createElement('div');
            notice.className = `segments-status ${safeLevel}`;
            notice.textContent = message;
            panel.prepend(notice);
            setTimeout(() => {
                notice.remove();
            }, 5200);
        }

        function buildSegmentActionContext(result, data, xNorm, yNorm, baseImageSrc) {
            if (!result || !result.path) {
                return null;
            }
            const folder = folderInput.value.trim();
            if (!folder) {
                return null;
            }
            const overlay = data && data.overlay ? data.overlay : {};
            const maskBase64 = stripBase64Payload(overlay.mask_raw_png || overlay.mask_png || '');
            if (!maskBase64) {
                return null;
            }
            const segments = Array.isArray(data && data.segments) ? data.segments : [];
            const meta = extractSegmentMeta(segments);
            return {
                folder,
                imagePath: String(result.path),
                maskBase64,
                segmentIds: meta.segmentIds,
                segmentLabels: meta.segmentLabels,
                overlay,
                xNorm,
                yNorm,
                baseImageSrc: baseImageSrc || '',
            };
        }

        async function runMaskSearchFromSegment(index, panel, triggerBtn = null) {
            const context = segmentContextByIndex[index];
            if (!context || !context.maskBase64) {
                showSegmentPanelNotice(panel, 'Click the image to create a region mask first.', 'warning');
                return;
            }

            const payload = {
                folder: context.folder,
                image_path: context.imagePath,
                mask: context.maskBase64,
                limit: parseInt(resultLimitSelect.value, 10) || 12,
                sort_by: sortBySelect.value || 'similarity',
                targets: ['images', 'segments'],
            };
            if (context.segmentLabels && Object.keys(context.segmentLabels).length) {
                payload.segment_labels = context.segmentLabels;
            }

            const button = triggerBtn instanceof HTMLButtonElement ? triggerBtn : null;
            const originalLabel = button ? button.textContent : '';
            if (button) {
                button.disabled = true;
                button.textContent = 'Searching...';
            }

            try {
                const response = await fetch('/search_by_mask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                let data = {};
                try {
                    data = await response.json();
                } catch (_) {
                    data = {};
                }
                if (!response.ok || data.error) {
                    const hint = data.hint ? ` ${data.hint}` : '';
                    throw new Error(`${data.error || 'Mask search failed'}${hint}`);
                }
                const segments = Array.isArray(data.segments) ? data.segments : [];
                const meta = extractSegmentMeta(segments);
                const refreshedContext = {
                    ...context,
                    segmentIds: meta.segmentIds,
                    segmentLabels: meta.segmentLabels,
                };
                segmentContextByIndex[index] = refreshedContext;
                renderSegmentResponse(
                    panel,
                    { ...data, overlay: context.overlay || {} },
                    context.xNorm,
                    context.yNorm,
                    context.baseImageSrc || '',
                    { index, actionContext: refreshedContext, sourceLabel: 'Mask search' },
                );
                indexStatus.textContent = segments.length
                    ? `Mask search returned ${segments.length} region candidate(s).`
                    : 'Mask search returned no region candidates.';
                indexStatus.className = segments.length ? 'status success' : 'status warning';
            } catch (err) {
                showSegmentPanelNotice(panel, `Mask search failed: ${err.message || String(err)}`, 'error');
            } finally {
                if (button) {
                    button.disabled = false;
                    button.textContent = originalLabel || 'Search by mask';
                }
            }
        }

        async function indexSegmentsFromMask(index, panel, triggerBtn = null) {
            const context = segmentContextByIndex[index];
            if (!context || !context.maskBase64) {
                showSegmentPanelNotice(panel, 'Click the image to create a region mask first.', 'warning');
                return;
            }

            const payload = {
                folder: context.folder,
                image_path: context.imagePath,
                mask: context.maskBase64,
                segment_labels: context.segmentLabels || {},
            };

            const button = triggerBtn instanceof HTMLButtonElement ? triggerBtn : null;
            const originalLabel = button ? button.textContent : '';
            if (button) {
                button.disabled = true;
                button.textContent = 'Indexing...';
            }

            try {
                const response = await fetch('/index_segments', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                let data = {};
                try {
                    data = await response.json();
                } catch (_) {
                    data = {};
                }
                if (!response.ok || data.error) {
                    const hint = data.hint ? ` ${data.hint}` : '';
                    throw new Error(`${data.error || 'Segment indexing failed'}${hint}`);
                }
                const count = Array.isArray(data.segments_indexed)
                    ? data.segments_indexed.length
                    : Number(data.segment_count || 0);
                showSegmentPanelNotice(panel, `Indexed ${count} segment(s) for this image.`, 'success');
                const relaxedNote = data.min_patches_relaxed ? ' (min patch fallback used)' : '';
                indexStatus.textContent = `Segment index updated (${count} segment${count === 1 ? '' : 's'})${relaxedNote}.`;
                indexStatus.className = 'status success';
            } catch (err) {
                showSegmentPanelNotice(panel, `Segment indexing failed: ${err.message || String(err)}`, 'error');
                indexStatus.textContent = `Segment indexing failed: ${err.message || String(err)}`;
                indexStatus.className = 'status error';
            } finally {
                if (button) {
                    button.disabled = false;
                    button.textContent = originalLabel || 'Index segments';
                }
            }
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
                const actionContext = buildSegmentActionContext(result, data, xNorm, yNorm, img.currentSrc || img.src);
                if (actionContext) {
                    segmentContextByIndex[index] = actionContext;
                } else {
                    delete segmentContextByIndex[index];
                }
                renderSegmentResponse(panel, data, xNorm, yNorm, img.currentSrc || img.src, {
                    index,
                    actionContext,
                    sourceLabel: 'Region proposal',
                });
            } catch (error) {
                panel.innerHTML = `<div class="segments-status error">Segment error: ${escapeHtml(error.message || String(error))}</div>`;
            } finally {
                delete item.dataset.segmentLoading;
            }
        }

        function renderSegmentResponse(panel, data, xNorm, yNorm, baseImageSrc, options = {}) {
            const segments = Array.isArray(data && data.segments) ? data.segments : [];
            const overlay = data && data.overlay ? data.overlay : {};
            const pctX = (xNorm * 100).toFixed(1);
            const pctY = (yNorm * 100).toFixed(1);
            const safeBaseSrc = baseImageSrc ? escapeHtml(baseImageSrc) : '';
            const sourceLabel = escapeHtml(String(options.sourceLabel || 'Region proposal'));
            const parsedIndex = Number.isFinite(options.index)
                ? Number(options.index)
                : parseInt(String(options.index || ''), 10);
            const actionContext = options.actionContext || null;
            const hasActions = Number.isFinite(parsedIndex)
                && actionContext
                && actionContext.maskBase64
                && actionContext.folder
                && actionContext.imagePath;

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
            const actionsHtml = hasActions ? `
                <div class="segment-actions">
                    <button class="segment-action-btn" data-segment-mask-search="${parsedIndex}">Search by mask</button>
                    <button class="segment-action-btn primary" data-segment-index="${parsedIndex}">Index segments</button>
                </div>
            ` : '';

            panel.innerHTML = `
                <div class="segments-status success">${sourceLabel} near (${pctX}%, ${pctY}%) · ${segments.length} candidate(s)</div>
                ${actionsHtml}
                ${overlayHtml}
                ${refinementNote}
                ${typeof overlay.threshold === 'number' ? `<div class="segment-meta">Heatmap threshold: ${(overlay.threshold * 100).toFixed(1)}%</div>` : ''}
                ${areaNote}
                ${resultsHtml}
            `;

            if (hasActions) {
                const maskSearchBtn = panel.querySelector(`[data-segment-mask-search="${parsedIndex}"]`);
                if (maskSearchBtn) {
                    maskSearchBtn.addEventListener('click', (event) => {
                        event.preventDefault();
                        event.stopPropagation();
                        runMaskSearchFromSegment(parsedIndex, panel, maskSearchBtn);
                    });
                }
                const indexBtn = panel.querySelector(`[data-segment-index="${parsedIndex}"]`);
                if (indexBtn) {
                    indexBtn.addEventListener('click', (event) => {
                        event.preventDefault();
                        event.stopPropagation();
                        indexSegmentsFromMask(parsedIndex, panel, indexBtn);
                    });
                }
            }
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
                const activeFolder = folderInput.value.trim();
                const params = new URLSearchParams();
                if (activeFolder) {
                    params.set('folder', activeFolder);
                }
                params.set('image_path', imagePath);
                const imageResponse = await fetch(`/image?${params.toString()}`);
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

        async function describeImageWithLM(index, imagePath, item = null, result = null) {
            if (!imagePath) {
                alert('No filesystem path is available for this image.');
                return;
            }
            const folder = folderInput.value.trim();
            if (!folder) {
                alert('Please enter a folder path first.');
                return;
            }

            const prompt = videoPromptInput.value.trim();
            const modelId = videoModelInput ? videoModelInput.value.trim() : '';

            setMode('archive');
            const targetItem = item || document.querySelector(`.result-item[data-result-index="${index}"]`);
            if (targetItem && !targetItem.classList.contains('expanded') && result) {
                toggleImageExpansion(targetItem, result, index);
            }

            const descContainer = document.getElementById(`lm-desc-${index}`);
            const saveBtn = document.getElementById(`lm-save-btn-${index}`);
            if (!descContainer || !saveBtn) {
                alert('Unable to render LLM description panel for this result.');
                return;
            }

            descContainer.innerHTML = '<div class="comment-loading"><div class="spinner"></div> Generating LLM description...</div>';
            saveBtn.style.display = 'none';
            saveBtn.dataset.summary = '';

            try {
                const response = await fetch('/describe_image', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        folder,
                        image_path: imagePath,
                        prompt,
                        model: modelId
                    }),
                });
                const data = await parseApiJson(response, 'Describe request failed');
                if (data.summary) {
                    renderLmDescription(index, data.summary, data.model || modelId || 'LM Studio');
                    return;
                }
                descContainer.innerHTML = '<div class="no-comments">(No description returned)</div>';
            } catch (err) {
                descContainer.innerHTML = `<div class="no-comments">Error: ${escapeHtml(err.message || String(err))}</div>`;
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
    response_html = response_html.replace('{luxriot_batch_options}', luxriot_batch_options_html)
    response_html = response_html.replace('{video_frame_options_html}', video_frame_options_html)
    response_html = response_html.replace('{segment_threshold_percent}', str(segment_threshold_percent))
    response_html = response_html.replace('{segments_enabled_checked}', segments_enabled_checked)
    response_html = response_html.replace('{segment_min_patches_default}', str(segment_min_patches_default))
    response_html = response_html.replace('{timestamp}', current_timestamp)
    response_html = response_html.replace('{lm_model}', config.LM_MODEL)
    response_html = response_html.replace('{luxriot_default_channel}', str(config.LUXRIOT_DEFAULT_CHANNEL_ID))
    response_html = response_html.replace('{luxriot_base_url}', config.LUXRIOT_BASE_URL)
    response_html = response_html.replace('{luxriot_snapshot_interval}', str(config.LUXRIOT_SNAPSHOT_INTERVAL))
    response_html = response_html.replace('{luxriot_snapshot_max_edge}', str(config.LUXRIOT_SNAPSHOT_MAX_EDGE))
    response_html = response_html.replace('{luxriot_batch_default}', str(luxriot_default_batch))
    
    # Create response with cache-busting headers
    response = make_response(response_html)
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    
    return response


@app.route('/image', methods=['GET'])
@app.route('/image/<path:filepath>', methods=['GET'])
def serve_image(filepath: str = ""):
    """Serve image files only from indexed folders."""
    try:
        folder_raw = request.args.get('folder')
        if not folder_raw:
            return "Missing folder parameter", 400
        folder_path = _resolve_folder_path(folder_raw, require_index=True)

        source_path = request.args.get('image_path') or filepath
        if not source_path:
            return "Missing image path", 400

        decoded = unquote(source_path)
        path_obj = Path(decoded)
        if not path_obj.is_absolute():
            path_obj = folder_path / path_obj
        abs_path = path_obj.resolve()
        if abs_path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            return "Unsupported file type", 403
        if not _path_within(abs_path, folder_path):
            return "Access denied", 403
        if not abs_path.exists() or not abs_path.is_file():
            return "Image not found", 404
        return send_file(str(abs_path))
    except ValueError as exc:
        return str(exc), 400
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
        img = img.resize((int(img.width * scale), int(img.height * scale)), RESAMPLE_LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=quality)
    return base64.b64encode(buffer.getvalue()).decode()


def _create_overlay_rgba(alpha_image: Image.Image, color: Tuple[int, int, int], opacity_scale: float = 1.0) -> Image.Image:
    alpha = alpha_image.convert('L')
    scale = float(opacity_scale)
    alpha_arr = np.asarray(alpha, dtype=np.float32)
    if scale <= 0:
        alpha_arr.fill(0.0)
    elif scale < 0.999:
        alpha_arr = np.clip(alpha_arr * scale, 0.0, 255.0)
    alpha = Image.fromarray(alpha_arr.astype(np.uint8), mode='L')
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


def _build_luxriot_messages(channel_label: str, frames: List[Dict[str, Any]], user_prompt: str, system_prompt: str) -> List[Dict[str, Any]]:
    prompt = (user_prompt or '').strip() or "Describe notable activity, people, vehicles, and anomalies."
    intro = (
        f"Live snapshots from Luxriot channel {channel_label}. "
        f"{len(frames)} snapshots captured roughly every {config.LUXRIOT_SNAPSHOT_INTERVAL}s."
    )
    user_content: List[Dict[str, Any]] = [{'type': 'text', 'text': f"{intro}\n\nTask: {prompt}"}]
    for idx, frame in enumerate(frames):
        ts_raw = frame.get('captured_at') or frame.get('time_sec')
        ts_label = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts_raw)) if isinstance(ts_raw, (int, float)) else 'n/a'
        user_content.append({'type': 'text', 'text': f"Snapshot {idx + 1} (captured at {ts_label})"})
        thumbnail = frame.get('thumbnail')
        if thumbnail:
            user_content.append(
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': f"data:image/jpeg;base64,{thumbnail}",
                        'detail': 'high',
                    },
                }
            )
    system_msg = system_prompt.strip() or LUXRIOT_SYSTEM_PROMPT_DEFAULT
    return [
        {'role': 'system', 'content': [{'type': 'text', 'text': system_msg}]},
        {'role': 'user', 'content': user_content},
    ]


def _call_lm_chat(messages: List[Dict[str, Any]], model_override: Optional[str] = None) -> str:
    base_url = (config.LM_BASE_URL or '').rstrip('/')
    if not base_url:
        raise RuntimeError("EVOSSEARCH_LM_BASE_URL is not configured.")
    endpoint = f"{base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if config.LM_API_KEY:
        headers["Authorization"] = f"Bearer {config.LM_API_KEY}"

    target_model = (model_override or config.LM_MODEL).strip()
    payload = {
        "model": target_model,
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


def _call_video_understanding(messages: List[Dict[str, Any]], model_override: Optional[str] = None) -> str:
    return _call_lm_chat(messages, model_override=model_override)


def _parse_lm_alerts(text: str, default_channel_id: int, default_ts_ms: Optional[int] = None) -> List[Dict[str, Any]]:
    """Extract alert objects from LM output; expects optional JSON with an alerts array."""
    import json
    import re
    now_ms = int(time.time() * 1000)
    base_ts_ms = default_ts_ms or now_ms

    def _validate_alert(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(raw, dict):
            return None
        title = (raw.get('title') or '').strip() or 'External event'
        description = (raw.get('description') or '').strip()
        severity = str(raw.get('severity') or 'critical').lower()
        allowed_sev = {'info', 'low', 'normal', 'high', 'critical'}
        if severity not in allowed_sev:
            severity = 'critical'
        state = str(raw.get('state') or 'new').lower()
        allowed_state = {'none', 'new', 'inprogress', 'closed', 'hidden'}
        if state not in allowed_state:
            state = 'new'
        channel_id = raw.get('channel_id') or default_channel_id
        try:
            channel_id = int(channel_id)
        except Exception:
            channel_id = default_channel_id
        timestamp_ms = raw.get('timestamp_ms') or base_ts_ms
        try:
            timestamp_ms = int(timestamp_ms)
        except Exception:
            timestamp_ms = base_ts_ms
        return {
            'title': title,
            'description': description,
            'severity': severity,
            'state': state,
            'channel_id': channel_id,
            'timestamp_ms': timestamp_ms,
        }

    def _extract_candidates(blob: str) -> List[Any]:
        candidates: List[Any] = []
        try:
            parsed = json.loads(blob)
            candidates.append(parsed)
        except Exception:
            pass
        for match in re.finditer(r"```json(.*?)```", blob, flags=re.DOTALL | re.IGNORECASE):
            try:
                candidates.append(json.loads(match.group(1)))
            except Exception:
                continue
        for match in re.finditer(r"ALERTS_JSON:(\{.*?\})", blob, flags=re.DOTALL | re.IGNORECASE):
            try:
                candidates.append(json.loads(match.group(1)))
            except Exception:
                continue
        return candidates

    alerts: List[Dict[str, Any]] = []
    for candidate in _extract_candidates(text or ''):
        if isinstance(candidate, dict) and isinstance(candidate.get('alerts'), list):
            for raw_alert in candidate['alerts']:
                validated = _validate_alert(raw_alert)
                if validated:
                    alerts.append(validated)
            if alerts:
                break

    if not alerts:
        # Heuristic fallbacks
        lowered = (text or '').lower()
        threat_keywords = [
            'weapon', 'gun', 'handgun', 'pistol', 'rifle', 'knife', 'shoot', 'shot', 'firearm', 'aggression', 'fight', 'violence',
            'оружие', 'пистолет', 'револьвер', 'винтовк', 'нож', 'стрел', 'выстрел', 'агресс', 'драк', 'насили'
        ]
        phone_keywords = ['phone', 'call', 'talking on phone', 'звон', 'телефон', 'разговаривает по телефону']
        pet_keywords = ['orl', 'orland', 'maz', 'cat', 'кот', 'кошка', 'питом']

        def add_fallback(title: str, severity: str, reason: str) -> None:
            val = _validate_alert(
                {
                    'title': title,
                    'description': f"Heuristic trigger: {reason}. Summary snippet: {text.strip()[:200]}",
                    'severity': severity,
                    'state': 'new',
                    'channel_id': default_channel_id,
                    'timestamp_ms': now_ms,
                }
            )
            if val:
                alerts.append(val)

        if any(k in lowered for k in threat_keywords):
            add_fallback('Possible weapon or aggression detected', 'critical', 'weapon/aggression keywords')
        elif any(k in lowered for k in phone_keywords):
            add_fallback('Phone call detected', 'info', 'phone keywords')
        elif any(k in lowered for k in pet_keywords):
            add_fallback('Pet interaction detected', 'low', 'pet keywords')

    return alerts


luxriot_manager = LuxriotManager(
    config=config,
    lm_callback=_call_video_understanding,
    message_builder=cast(Any, _build_luxriot_messages),
    jpeg_encoder=_encode_jpeg,
    alert_parser=_parse_lm_alerts,
    probe_manager=None,  # will be assigned after probe_manager init
)

PROBE_MAX_STORED_HITS = getattr(config, 'PROBE_MAX_STORED_HITS', 30)
PROBE_DAEMON_INTERVAL_SEC = getattr(config, 'PROBE_DAEMON_INTERVAL_SEC', 5)
PROBE_BENCH_BATCH = getattr(config, 'PROBE_BENCH_BATCH', 16)
LUXRIOT_SYSTEM_PROMPT_DEFAULT = (
    "You summarize real-time CCTV snapshots. Focus on key actions, people, vehicles, time of day, and any risks. "
    "Keep it concise and avoid repetition across frames. Provide a free-form summary first. "
    "Always append a JSON block in this form (alerts may be empty, but must be present): "
    "{ 'alerts': [ { 'title': 'short alert title', 'description': '1-2 sentence description with any time/frame hints', "
    "'severity': 'info|low|normal|high|critical', 'state': 'new|inprogress|closed|hidden|none', "
    "'channel_id': <channel id>, 'timestamp_ms': <milliseconds since epoch> } ] }. "
    "Rules: mark weapons, fights, or aggression as critical; calm phone use as info; holding pets (Orlandina or Maz) as low; "
    "other notable but mild changes as normal/high as appropriate. If nothing notable, alerts is an empty array."
)

probe_manager = ProbeManager(
    embed_image_fn=lambda img: get_image_embedding_from_pil(img, embedder="clip"),
    embed_text_fn=lambda text: get_text_embedding(text),
    jpeg_encoder=_encode_jpeg,
)
luxriot_manager.probe_manager = probe_manager
probe_daemon_thread: Optional[threading.Thread] = None
probe_daemon_stop = threading.Event()


class ProbesStore:
    def __init__(self, path: Union[str, Path] = "probes_store.json") -> None:
        self.path = Path(path)
        self.data: Dict[str, Any] = {"probes": []}
        self.lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        with self.lock:
            if self.path.exists():
                try:
                    loaded = json.loads(self.path.read_text(encoding='utf-8'))
                    if isinstance(loaded, dict):
                        self.data = loaded
                    else:
                        self.data = {"probes": []}
                except Exception:
                    self.data = {"probes": []}

    def _save_locked(self) -> None:
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        tmp_path.write_text(json.dumps(self.data, indent=2), encoding='utf-8')
        tmp_path.replace(self.path)

    def list_probes(self) -> List[Dict[str, Any]]:
        with self.lock:
            probes = self.data.get("probes", [])
            return copy.deepcopy(probes if isinstance(probes, list) else [])

    def upsert_probe(self, probe: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            self.data.setdefault("probes", [])
            probe_list: List[Dict[str, Any]] = self.data["probes"]
            stored_probe = copy.deepcopy(probe)
            if not stored_probe.get("id"):
                stored_probe["id"] = f"probe-{uuid.uuid4().hex[:12]}"
            existing = None
            for idx, item in enumerate(probe_list):
                if item.get("id") == stored_probe["id"]:
                    existing = idx
                    break
            if existing is None:
                probe_list.append(stored_probe)
            else:
                probe_list[existing] = stored_probe
            self._save_locked()
            return copy.deepcopy(stored_probe)

    def delete_probe(self, probe_id: str) -> bool:
        with self.lock:
            probes = self.data.get("probes", [])
            new_probes = [p for p in probes if p.get("id") != probe_id]
            if len(new_probes) == len(probes):
                return False
            self.data["probes"] = new_probes
            self._save_locked()
            return True


probes_store = ProbesStore()


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


def _probe_daemon() -> None:
    """Background runner: execute enabled probes across channels."""
    while not probe_daemon_stop.is_set():
        try:
            probes = probes_store.list_probes()
            # Group probes by channel
            by_channel: Dict[int, List[Dict[str, Any]]] = {}
            for p in probes:
                if p.get("enabled") is False:
                    continue
                ch = int(p.get("channel_id", config.LUXRIOT_DEFAULT_CHANNEL_ID))
                by_channel.setdefault(ch, []).append(p)
            for ch, plist in by_channel.items():
                try:
                    if luxriot_manager.is_probe_capture_paused(ch):
                        continue
                    # Ensure capture running for this channel
                    try:
                        fps_desired = max([p.get('fps') or 0 for p in plist] or [0])
                        luxriot_manager.start_probe_capture(
                            ch,
                            fps=fps_desired if fps_desired > 0 else None,
                            clear_pause=False,
                        )
                    except Exception as exc:
                        print(f"Probe daemon failed to start capture for channel {ch}: {exc}")
                    for probe in plist:
                        result = probe_manager.query(
                            probe.get('channel_id', config.LUXRIOT_DEFAULT_CHANNEL_ID),
                            probe.get('positives', []),
                            probe.get('negatives', []),
                            probe.get('pos_floor', 0.2),
                            probe.get('margin', 0.05),
                            probe.get('top_k', 6),
                            window_sec=probe.get('window_sec', 300.0),
                            image_probe=probe.get('image_probe'),
                        )
                        if 'error' in result:
                            continue
                        hits = result.get('results') or []
                        if hits:
                            probe['last_hit'] = hits[0]
                            recent = probe.get('recent_hits') or []
                            recent = (hits + recent)[:PROBE_MAX_STORED_HITS]
                            probe['recent_hits'] = recent
                            if probe.get('bookmark'):
                                try:
                                    luxriot_manager.send_bookmark_event(
                                        channel_id=probe.get('channel_id', config.LUXRIOT_DEFAULT_CHANNEL_ID),
                                        title=f"Probe hit: {probe.get('name', 'probe')}",
                                        description=f"pos {hits[0].get('pos_score'):.3f} / neg {hits[0].get('neg_score'):.3f} · margin {hits[0].get('margin'):.3f}",
                                        severity=probe.get('severity', 'critical'),
                                        state='new',
                                        timestamp_ms=hits[0].get('timestamp_ms'),
                                    )
                                except Exception as exc:
                                    print(f"Probe daemon failed to send bookmark for probe {probe.get('id')}: {exc}")
                            probes_store.upsert_probe(probe)
                except Exception as exc:
                    print(f"Probe daemon channel loop error (channel {ch}): {exc}")
                    continue
        except Exception as exc:
            print(f"Probe daemon loop error: {exc}")
        probe_daemon_stop.wait(PROBE_DAEMON_INTERVAL_SEC)

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


def load_index(folder_path: Union[str, Path], embedder: Optional[str] = None) -> FaissIndexBundle:
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

    _faiss_add_vectors(index, embeddings)
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
    result = {
        'path': img_path,
        'filename': os.path.basename(img_path),
        'similarity': float(similarity),
        'thumbnail': '',
        'metadata': dict(metadata or {}),
    }
    try:
        with Image.open(img_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.thumbnail(config.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=config.THUMBNAIL_QUALITY)
            result['thumbnail'] = base64.b64encode(buffer.getvalue()).decode()
    except Exception as img_error:
        # Keep the result entry even if thumbnail creation fails so search never collapses to empty.
        # Frontend can still show filename/path and let operators inspect the source image directly.
        result['metadata']['thumbnail_error'] = str(img_error)
        print(f"Warning: thumbnail generation failed for {img_path}: {img_error}")
    if extra:
        result.update(extra)
    return result


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
            vec = _faiss_reconstruct(index, int(idx))
        except (AttributeError, RuntimeError, IndexError, TypeError):
            return None
        vectors.append(vec)
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
    skipped = 0

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
        else:
            skipped += 1

    if skipped:
        print(f"Warning: skipped {skipped}/{max_results} ranked entries while building results.")

    return results


def _load_fusion_indexes(folder_path: Union[str, Path]) -> Tuple[FaissIndexBundle, FaissIndexBundle]:
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


def _fuse_results(
    clip_data: FaissIndexBundle,
    dino_data: FaissIndexBundle,
    clip_vec: np.ndarray,
    dino_vec: np.ndarray,
    limit: int,
    sort_by: str,
) -> List[Dict[str, Any]]:
    clip_index, clip_paths, clip_metadata, _ = clip_data
    dino_index, dino_paths, dino_metadata, _ = dino_data
    if clip_index is None or dino_index is None:
        return []

    clip_map = _prepare_metadata_map(clip_paths, clip_metadata)
    dino_map = _prepare_metadata_map(dino_paths, dino_metadata)
    metadata_map = _merge_metadata_maps(clip_map, dino_map)

    limit = max(1, limit)
    alpha = max(0.0, min(1.0, float(config.FUSION_ALPHA)))

    def _search(index: Optional[faiss.Index], vec: np.ndarray, paths: Optional[Sequence[str]]) -> Dict[str, float]:
        if index is None or not paths:
            return {}
        k = min(limit * 2, len(paths))
        sims, inds = _faiss_search(index, vec.reshape(1, -1), k)
        scores: Dict[str, float] = {}
        for idx, sim in zip(inds[0], sims[0]):
            idx_int = int(idx)
            if 0 <= idx_int < len(paths):
                scores[paths[idx_int]] = float(sim)
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

    similarities, indices = _faiss_search(segment_index, query_vec.reshape(1, -1), k)

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


def _has_non_full_segments(segments: Dict[str, Any]) -> bool:
    return any(str(seg_id) != 'full' for seg_id in segments.keys())


def _encode_mask_segments_with_fallback(
    image_input: Union[str, Path, Image.Image],
    mask_image: Image.Image,
    segment_ids: Optional[List[str]] = None,
    min_patches: Optional[int] = None,
) -> Tuple[Dict[str, Dict[str, Union[np.ndarray, float, int]]], int, bool]:
    ensure_embedder_loaded('dino')
    if dino_encoder is None:
        raise RuntimeError('DINO encoder is not available')

    requested_min = max(1, int(min_patches if min_patches is not None else config.DINO_SEGMENT_MIN_PATCHES))
    segments = dino_encoder.encode_masked(
        image_input,
        mask_image,
        segment_ids=segment_ids,
        min_patches=requested_min,
    )
    if requested_min <= 1:
        return segments, requested_min, False

    require_non_full = True
    if segment_ids is not None:
        normalized_ids = {str(seg).strip().lower() for seg in segment_ids if str(seg).strip()}
        require_non_full = bool(normalized_ids - {'full'})

    if segments and (not require_non_full or _has_non_full_segments(segments)):
        return segments, requested_min, False

    relaxed = dino_encoder.encode_masked(
        image_input,
        mask_image,
        segment_ids=segment_ids,
        min_patches=1,
    )
    if relaxed and (not require_non_full or _has_non_full_segments(relaxed)):
        print(
            f"Mask encoding fallback: relaxed min_patches from {requested_min} to 1 "
            f"(segment_ids={segment_ids if segment_ids is not None else 'auto'})"
        )
        return relaxed, 1, True

    return segments or relaxed, requested_min, False


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
    segments, _, _ = _encode_mask_segments_with_fallback(
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
            similarities, indices = _faiss_search(image_index, embedding.reshape(1, -1), k)
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
    folder_raw = request.args.get('folder')
    image_path = request.args.get('image_path')
    
    if not folder_raw or not image_path:
        return jsonify({'error': 'Missing folder or image_path parameter'}), 400
    
    try:
        folder = str(_resolve_folder_path(folder_raw))
        comments = get_image_comments(folder, image_path)
        return jsonify({'comments': comments})
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as e:
        print(f"Error getting comments: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/comments', methods=['POST'])
def save_comment():
    """Save a new comment for an image"""
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    folder_raw = data.get('folder')
    image_path = data.get('image_path')
    comment = data.get('comment', '').strip()
    
    if not folder_raw or not image_path or not comment:
        return jsonify({'error': 'Missing folder, image_path, or comment'}), 400
    
    # Basic input sanitization
    if len(comment) > config.MAX_COMMENT_LENGTH:
        return jsonify({'error': f'Comment too long (max {config.MAX_COMMENT_LENGTH} characters)'}), 400
    
    try:
        folder = str(_resolve_folder_path(folder_raw))
        success = add_image_comment(folder, image_path, comment)
        if success:
            comments = get_image_comments(folder, image_path)
            return jsonify({'success': True, 'comments': comments})
        else:
            return jsonify({'error': 'Failed to save comment'}), 500
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as e:
        print(f"Error saving comment: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/commented_images', methods=['POST'])
def get_commented_images():
    """Get all images that have comments in the indexed folder"""
    payload = _json_body()
    folder_raw = payload.get('folder')
    if not folder_raw:
        return jsonify({'error': 'No folder specified'}), 400

    try:
        folder = str(_resolve_folder_path(folder_raw))
        available = _available_indexes(folder)
        image_paths: List[str] = []
        image_metadata: List[Dict[str, Any]] = []
        metadata_map: Dict[str, Dict[str, Any]] = {}

        if available:
            targets = [active_embedder] + [m for m in available if m != active_embedder]
            for emb in targets:
                idx_obj, paths, metas, meta_info = load_index(folder, embedder=emb)
                if idx_obj is not None:
                    image_paths = paths or []
                    image_metadata = metas or []
                    metadata_map = _prepare_metadata_map(image_paths, image_metadata)
                    break

        comments_data = load_comments(folder)
        if not comments_data:
            return jsonify({'results': []})

        results: List[Dict[str, Any]] = []
        for image_path, comment_list in comments_data.items():
            if not Path(image_path).exists():
                continue
            entry = _build_result_entry(
                image_path,
                similarity=1.0,
                metadata=metadata_map.get(image_path, {}),
                extra={
                    'comment_count': len(comment_list),
                    'latest_comment': comment_list[-1] if comment_list else '',
                },
            )
            if entry:
                results.append(entry)

        results.sort(key=lambda x: (x.get('metadata', {}).get('mtime', 0), x.get('comment_count', 0)), reverse=True)
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/check_index', methods=['POST'])
def check_index():
    """Check if folder is indexed"""
    data = _json_body()
    folder_raw = data.get('folder')
    if not folder_raw:
        return jsonify({'error': 'No folder specified'}), 400
    try:
        folder = str(_resolve_folder_path(folder_raw))
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    
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
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    folder_raw = data.get('folder')
    if not folder_raw:
        return jsonify({'error': 'Invalid folder path'}), 400
    
    try:
        folder = str(_resolve_folder_path(folder_raw))
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


def _normalize_mask_image(mask_img: Image.Image) -> Image.Image:
    """Normalize uploaded/decoded mask images to a meaningful single-channel mask.

    If an alpha channel exists and is informative (not fully opaque), use alpha because
    UI overlays are RGBA with semantic mask stored in transparency.
    """
    bands = mask_img.getbands()
    if "A" in bands:
        try:
            alpha = mask_img.getchannel("A")
            alpha_np = np.asarray(alpha)
            if alpha_np.size and np.any(alpha_np > 0) and np.any(alpha_np < 255):
                return alpha.convert("L")
        except Exception:
            pass
    if mask_img.mode != "L":
        return mask_img.convert("L")
    return mask_img


def _load_mask_from_request() -> Optional[Image.Image]:
    mask_file = request.files.get('mask')
    if mask_file:
        return _normalize_mask_image(Image.open(mask_file.stream))
    payload = _json_body()
    mask_base64 = request.form.get('mask') or payload.get('mask')
    if mask_base64:
        try:
            mask_bytes = base64.b64decode(mask_base64)
            return _normalize_mask_image(Image.open(BytesIO(mask_bytes)))
        except Exception as exc:
            raise ValueError(f"Invalid mask payload: {exc}")
    return None


@app.route('/index_segments', methods=['POST'])
def index_segments():
    """Index DINO segment embeddings derived from a mask."""
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    if not config.DINO_SEGMENTS_ENABLED:
        return jsonify({'error': 'Segment indexing is disabled. Enable EVOSSEARCH_DINO_SEGMENTS_ENABLED to use this feature.'}), 400

    data = request.form if request.form else _json_body()

    folder_raw = data.get('folder')
    image_path = data.get('image_path')
    if not folder_raw or not image_path:
        return jsonify({'error': 'Both folder and image_path are required'}), 400
    try:
        folder_path = _resolve_folder_path(folder_raw)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    image_obj = Path(image_path).expanduser().resolve()
    if not image_obj.exists() or not image_obj.is_file():
        return jsonify({'error': f'Image file not found: {image_path}'}), 400
    if image_obj.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
        return jsonify({'error': 'Unsupported image file type'}), 400
    if not _path_within(image_obj, folder_path):
        return jsonify({'error': 'image_path must be inside folder'}), 400

    mask_image = _load_mask_from_request()
    if mask_image is None:
        return jsonify({'error': 'Mask is required for segment indexing'}), 400

    segment_ids = _parse_segment_ids(data.get('segment_ids'))
    label_map = _parse_segment_labels(data.get('segment_labels'))

    try:
        segments, min_patches_used, min_patches_relaxed = _encode_mask_segments_with_fallback(
            str(image_obj),
            mask_image,
            segment_ids=segment_ids,
            min_patches=config.DINO_SEGMENT_MIN_PATCHES,
        )
    except RuntimeError as exc:
        return jsonify({'error': str(exc)}), 500

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
        segment_keys = [str(key) for key in segments.keys()]
        print(
            "index_segments: no non-full segments "
            f"(keys={segment_keys}, min_patches_used={min_patches_used}, relaxed={min_patches_relaxed})"
        )
        return jsonify(
            {
                'error': 'Mask did not yield any segments beyond the full image aggregate',
                'min_patches_used': min_patches_used,
                'min_patches_relaxed': min_patches_relaxed,
                'hint': 'Select a larger region or lower Min Segment Patches in settings.',
            }
        ), 400

    embedding_matrix = np.stack(embeddings, axis=0)
    save_segment_index(str(folder_path), embedding_matrix, entries)

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
            'min_patches_used': min_patches_used,
            'min_patches_relaxed': min_patches_relaxed,
        }
    )


@app.route('/video_understanding', methods=['POST'])
def video_understanding():
    data = _json_body()
    video_path = (data.get('video') or '').strip()
    if not video_path:
        return jsonify({'error': 'Provide a video path.'}), 400
    video_obj = Path(video_path).expanduser().resolve()
    if not video_obj.exists() or not video_obj.is_file():
        return jsonify({'error': f'Video file not found: {video_path}'}), 400
    if config.ALLOWED_ROOTS:
        allowed_roots = [Path(item).expanduser().resolve() for item in config.ALLOWED_ROOTS]
        if not any(_path_within(video_obj, root) for root in allowed_roots):
            return jsonify({'error': 'Video path is outside configured allowed roots'}), 400
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
    model_hint = (data.get('model') or '').strip()

    try:
        frames, fps, duration = _sample_video_frames(
            str(video_obj),
            max_frames=max_frames_int,
            sample_fps=sample_fps_val,
            max_edge=config.LM_VIDEO_MAX_EDGE,
        )
        if not frames:
            return jsonify({'error': 'No frames could be extracted from the video.'}), 400
        messages = _build_video_messages(str(video_obj), frames, user_prompt)
        summary = _call_video_understanding(messages, model_override=model_hint or None)
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
                'model': model_hint or config.LM_MODEL,
            }
        )
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/describe_image', methods=['POST'])
def describe_image():
    data = _json_body()
    folder_raw = data.get('folder')
    image_path = (data.get('image_path') or '').strip()
    prompt = data.get('prompt') or ''
    model_hint = (data.get('model') or '').strip()
    if not folder_raw or not image_path:
        return jsonify({'error': 'folder and image_path are required'}), 400
    try:
        folder_path = _resolve_folder_path(folder_raw, require_index=True)
        path_obj = Path(image_path).expanduser().resolve()
        if path_obj.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            return jsonify({'error': 'Unsupported image file type'}), 400
        if not path_obj.exists() or not path_obj.is_file():
            return jsonify({'error': f'Image not found: {image_path}'}), 400
        if not _path_within(path_obj, folder_path):
            return jsonify({'error': 'image_path must be inside folder'}), 400
        messages = _build_image_messages(str(path_obj), prompt)
        summary = _call_lm_chat(messages, model_override=model_hint or None)
        thumb = _encode_jpeg(Image.open(path_obj), max_edge=config.THUMBNAIL_SIZE[0])
        return jsonify(
            {
                'summary': summary,
                'thumbnail': thumb,
                'model': model_hint or config.LM_MODEL,
                'image_path': str(path_obj),
            }
        )
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
@app.route('/search', methods=['POST'])
def search():
    """Search for images using text queries."""
    data = _json_body()
    folder_raw = data.get('folder')
    query = data.get('query')
    limit = data.get('limit', 10)
    sort_by = data.get('sort_by', 'similarity')  # 'similarity' or 'time'
    if not folder_raw or not query:
        return jsonify({'error': 'Missing folder or query'}), 400
    try:
        folder = str(_resolve_folder_path(folder_raw, require_index=True))
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

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
    if index is None or not image_paths:
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
            print(f"Text search: query='{query}' candidates=0 returned=0 folder='{folder}'")
            return jsonify({'results': []})
        similarities, indices = _faiss_search(index, text_embedding.reshape(1, -1), k)

        metadata_map = _prepare_metadata_map(image_paths, image_metadata)
        candidate_count = len(_collect_candidates(indices, similarities, len(image_paths)))
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

        print(
            f"Text search: query='{query}' candidates={candidate_count} "
            f"returned={len(results)} folder='{folder}'"
        )
        return jsonify({'results': results})
    except Exception as e:
        print(f"Text search error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/search_by_image', methods=['POST'])
def search_by_image():
    """Search for images using an uploaded image"""
    folder_raw = request.form.get('folder')
    limit = request.form.get('limit', 12)
    sort_by = request.form.get('sort_by', 'similarity')  # 'similarity' or 'time'

    if not folder_raw:
        return jsonify({'error': 'Missing folder'}), 400
    try:
        folder = str(_resolve_folder_path(folder_raw, require_index=True))
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

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

    clip_data: Optional[FaissIndexBundle] = None
    dino_data: Optional[FaissIndexBundle] = None
    index: Optional[faiss.Index] = None
    image_paths: List[str] = []
    image_metadata: List[Dict[str, Any]] = []

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
        index, loaded_paths, loaded_metadata, index_meta = load_index(folder, embedder=active_embedder)
        if index is None or not loaded_paths:
            message = 'Folder not indexed for the current backend'
            available = _available_indexes(folder)
            if available:
                message += f" (available: {', '.join(available)})"
            return jsonify({'error': message}), 400
        image_paths = loaded_paths
        image_metadata = loaded_metadata or []

    try:
        dino_vec: Optional[np.ndarray] = None
        if file:
            uploaded_image = Image.open(file.stream)
            if uploaded_image.mode != 'RGB':
                uploaded_image = uploaded_image.convert('RGB')
            clip_vec = get_image_embedding_from_pil(uploaded_image, embedder='clip') if fusion_active else get_image_embedding_from_pil(uploaded_image, embedder=active_embedder)
            if fusion_active:
                dino_vec = get_image_embedding_from_pil(uploaded_image, embedder='dino')
        else:
            image_obj = Path(str(image_path)).expanduser().resolve()
            if image_obj.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
                return jsonify({'error': 'Unsupported image file type'}), 400
            if not image_obj.exists() or not image_obj.is_file():
                return jsonify({'error': f'Image file not found: {image_path}'}), 400
            if not _path_within(image_obj, Path(folder)):
                return jsonify({'error': 'image_path must be inside folder'}), 400
            clip_vec = get_image_embedding(image_obj, embedder='clip') if fusion_active else get_image_embedding(image_obj, embedder=active_embedder)
            if fusion_active:
                dino_vec = get_image_embedding(image_obj, embedder='dino')

        if fusion_active:
            if clip_data is None or dino_data is None or dino_vec is None:
                return jsonify({'error': 'Fusion search requires both CLIP and DINO query embeddings'}), 500
            results = _fuse_results(clip_data, dino_data, clip_vec, dino_vec, limit, sort_by)
            return jsonify({'results': results})

        if index is None or not image_paths:
            return jsonify({'error': 'Backend index is unavailable for image search'}), 400
        k = _candidate_pool_size(limit, len(image_paths), sort_by)
        if k == 0:
            print(f"Image search: candidates=0 returned=0 folder='{folder}'")
            return jsonify({'results': []})
        similarities, indices = _faiss_search(index, clip_vec.reshape(1, -1), k)

        metadata_map = _prepare_metadata_map(image_paths, image_metadata)
        candidate_count = len(_collect_candidates(indices, similarities, len(image_paths)))
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

        print(
            f"Image search: candidates={candidate_count} returned={len(results)} "
            f"folder='{folder}'"
        )
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/search_by_mask', methods=['POST'])
def search_by_mask():
    """Search using a masked region of an image leveraging DINO segment embeddings."""
    data = request.form if request.form else _json_body()

    if not config.DINO_SEGMENTS_ENABLED:
        return jsonify({'error': 'Segment search is disabled. Enable EVOSSEARCH_DINO_SEGMENTS_ENABLED to use this feature.'}), 400

    folder_raw = data.get('folder')
    if not folder_raw:
        return jsonify({'error': 'Missing folder'}), 400
    try:
        folder_path = _resolve_folder_path(folder_raw, require_index=True)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    folder = str(folder_path)

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
    if dino_encoder is None:
        return jsonify({'error': 'DINO encoder is not available'}), 500

    if uploaded_image:
        query_image = Image.open(uploaded_image.stream)
        if query_image.mode != 'RGB':
            query_image = query_image.convert('RGB')
        image_input: Union[Image.Image, str] = query_image
    else:
        image_obj = Path(str(image_source)).expanduser().resolve()
        if image_obj.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            return jsonify({'error': 'Unsupported image file type'}), 400
        if not image_obj.exists():
            return jsonify({'error': f'Image file not found: {image_source}'}), 400
        if not _path_within(image_obj, folder_path):
            return jsonify({'error': 'image_path must be inside folder'}), 400
        image_input = str(image_obj)

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
    data = request.form if request.form else _json_body()
    if data is None:
        return jsonify({'error': 'No data provided'}), 400

    folder_raw = data.get('folder')
    if not folder_raw:
        return jsonify({'error': 'Missing folder'}), 400
    try:
        folder_path = _resolve_folder_path(folder_raw, require_index=True)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    folder = str(folder_path)

    image_path = data.get('image_path')
    uploaded_image = request.files.get('image')
    if not image_path and uploaded_image is None:
        return jsonify({'error': 'Provide image_path or upload an image file'}), 400

    x_raw = data.get('x')
    y_raw = data.get('y')
    if x_raw is None or y_raw is None:
        return jsonify({'error': 'Invalid or missing x/y coordinates'}), 400
    try:
        x_norm = float(x_raw)
        y_norm = float(y_raw)
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
        image_obj = Path(str(image_path)).expanduser().resolve()
        if image_obj.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            return jsonify({'error': 'Unsupported image file type'}), 400
        if not image_obj.exists() or not image_obj.is_file():
            return jsonify({'error': f'Image file not found: {image_path}'}), 400
        if not _path_within(image_obj, folder_path):
            return jsonify({'error': 'image_path must be inside folder'}), 400
        with Image.open(image_obj) as src:
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
    mask_img = Image.fromarray(coarse_mask_uint8).resize((crop_size, crop_size), resample=RESAMPLE_NEAREST)

    base_size = pil_image.size
    overlay_mask_source = Image.fromarray(coarse_mask_uint8).resize(base_size, resample=RESAMPLE_NEAREST)
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
                        overlay_mask_source = overlay_mask_source.resize(base_size, resample=RESAMPLE_NEAREST)
                    mask_img = overlay_mask_source.resize((crop_size, crop_size), resample=RESAMPLE_NEAREST)
                    if refined_label:
                        label_map[str(segment_value)] = refined_label
        except Exception as exc:
            print(f"Mask2Former refinement error: {exc}")

    # Update mask coverage fraction based on overlay image size
    overlay_mask_np = np.asarray(overlay_mask_source)
    if overlay_mask_np.size:
        mask_fraction = float(np.count_nonzero(overlay_mask_np)) / float(overlay_mask_np.size)

    heatmap_alpha = Image.fromarray((heatmap_norm * 255).astype(np.uint8)).resize(base_size, resample=RESAMPLE_BILINEAR)
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
        seg_overlay: Optional[Image.Image] = None
        if isinstance(seg_map, np.ndarray):
            seg_overlay, legend_entries = _render_segmentation_overlay(
                seg_map,
                class_labels,
                int(refine_result.get('class_id', segment_value)),
            )
        if seg_overlay is not None:
            if seg_overlay.size != base_size:
                seg_overlay = seg_overlay.resize(base_size, resample=RESAMPLE_NEAREST)
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
        'mask_raw_png': _image_to_base64(overlay_mask_source.convert('L')),
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


@app.route('/luxriot/channels', methods=['GET'])
def luxriot_channels():
    """Fetch available Luxriot channels (cached briefly)."""
    force = str(request.args.get('force', '')).lower() in {'1', 'true', 'yes'}
    try:
        channels = luxriot_manager.get_channels(force=force)
        return jsonify({'channels': channels})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/luxriot/snapshot/<int:channel_id>', methods=['GET'])
def luxriot_snapshot(channel_id: int):
    stream_type = request.args.get('stream', 'mainStream')
    try:
        encoded, meta = luxriot_manager.get_snapshot_base64(channel_id, stream_type=stream_type)
        img_bytes = base64.b64decode(encoded)
        response = make_response(img_bytes)
        response.headers['Content-Type'] = 'image/jpeg'
        response.headers['Cache-Control'] = 'no-store, must-revalidate'
        response.headers['X-Image-Width'] = str(meta.get('width'))
        response.headers['X-Image-Height'] = str(meta.get('height'))
        return response
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/luxriot/start_capture', methods=['POST'])
def luxriot_start_capture():
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    try:
        channel_id = int(data.get('channel_id') or data.get('channel') or data.get('id') or config.LUXRIOT_DEFAULT_CHANNEL_ID)
    except Exception:
        return jsonify({'error': 'Provide a valid channel_id'}), 400
    batch_size = data.get('batch_size')
    prompt = data.get('prompt') or ''
    model_hint = (data.get('model') or '').strip() or None
    system_prompt = (data.get('system_prompt') or '').strip() or None
    try:
        status = luxriot_manager.start_session(
            channel_id,
            batch_size=batch_size,
            prompt=prompt,
            model_hint=model_hint,
            system_prompt=system_prompt,
        )
        return jsonify({'success': True, 'session': status})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/luxriot/stop_capture', methods=['POST'])
def luxriot_stop_capture():
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    try:
        channel_id = int(data.get('channel_id') or data.get('channel') or config.LUXRIOT_DEFAULT_CHANNEL_ID)
    except Exception:
        return jsonify({'error': 'Provide a valid channel_id'}), 400
    try:
        state = luxriot_manager.stop_session(channel_id)
        return jsonify({'success': True, 'session': state})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/luxriot/flush_capture', methods=['POST'])
def luxriot_flush_capture():
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    try:
        channel_id = int(data.get('channel_id') or data.get('channel') or config.LUXRIOT_DEFAULT_CHANNEL_ID)
    except Exception:
        return jsonify({'error': 'Provide a valid channel_id'}), 400
    try:
        result = luxriot_manager.flush_session(channel_id)
        if not result.get('success'):
            return jsonify(result), 400
        return jsonify(result)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/luxriot/session', methods=['GET'])
def luxriot_session_status():
    channel_id = request.args.get('channel_id', default=config.LUXRIOT_DEFAULT_CHANNEL_ID, type=int)
    try:
        status = luxriot_manager.session_status(channel_id)
        return jsonify(status)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/luxriot/streams', methods=['GET'])
def luxriot_streams_status():
    try:
        return jsonify(luxriot_manager.streams_status())
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/luxriot/streams/stop', methods=['POST'])
def luxriot_stop_stream():
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    try:
        channel_id = int(data.get('channel_id') or data.get('channel') or config.LUXRIOT_DEFAULT_CHANNEL_ID)
    except Exception:
        return jsonify({'error': 'Provide a valid channel_id'}), 400
    stream_type = (data.get('stream_type') or 'both').strip().lower()
    pause_analytics = _coerce_bool(data.get('pause_analytics'), True)
    try:
        result = luxriot_manager.stop_stream(channel_id, stream_type=stream_type, pause_analytics=pause_analytics)
        return jsonify({'success': True, 'result': result, 'streams': luxriot_manager.streams_status()})
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/luxriot/streams/stop_all', methods=['POST'])
def luxriot_stop_all_streams():
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    stop_video = _coerce_bool(data.get('stop_video'), True)
    stop_analytics = _coerce_bool(data.get('stop_analytics'), True)
    pause_analytics = _coerce_bool(data.get('pause_analytics'), True)
    if not stop_video and not stop_analytics:
        return jsonify({'error': 'Select at least one stream type to stop'}), 400
    try:
        result = luxriot_manager.stop_all_streams(
            stop_video=stop_video,
            stop_analytics=stop_analytics,
            pause_analytics=pause_analytics,
        )
        return jsonify({'success': True, 'result': result, 'streams': luxriot_manager.streams_status()})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/luxriot/bookmark', methods=['POST'])
def luxriot_bookmark():
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    try:
        channel_id = int(data.get('channel_id') or data.get('channel') or config.LUXRIOT_DEFAULT_CHANNEL_ID)
    except Exception:
        return jsonify({'error': 'Provide a valid channel_id'}), 400
    title = (data.get('title') or '').strip() or 'External event'
    description = data.get('description') or ''
    severity = (data.get('severity') or 'critical').strip().lower()
    state = (data.get('state') or 'new').strip().lower()
    timestamp_ms = data.get('timestamp_ms')
    try:
        if timestamp_ms is not None:
            timestamp_ms = int(timestamp_ms)
    except Exception:
        timestamp_ms = None
    try:
        result = luxriot_manager.send_bookmark_event(
            channel_id=channel_id,
            title=title,
            description=description,
            severity=severity,
            state=state,
            timestamp_ms=timestamp_ms,
        )
        return jsonify(result)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/probes/query', methods=['POST'])
def probes_query():
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    try:
        channel_id = int(data.get('channel_id') or data.get('channel') or config.LUXRIOT_DEFAULT_CHANNEL_ID)
    except Exception:
        return jsonify({'error': 'Provide a valid channel_id'}), 400
    positives = data.get('positives') or []
    negatives = data.get('negatives') or []
    try:
        pos_floor = float(data.get('pos_floor', 0.2))
    except Exception:
        pos_floor = 0.2
    try:
        margin_thr = float(data.get('margin', 0.05))
    except Exception:
        margin_thr = 0.05
    try:
        top_k = int(data.get('top_k', 5))
    except Exception:
        top_k = 5
    try:
        window_sec = float(data.get('window_sec', 0))
    except Exception:
        window_sec = 0
    result = probe_manager.query(channel_id, positives, negatives, pos_floor, margin_thr, top_k, window_sec=window_sec, image_probe=data.get('image_probe'))
    status_code = 200 if 'error' not in result else 400
    hits = result.get('results') or []
    if hits and data.get('bookmark'):
        try:
            luxriot_manager.send_bookmark_event(
                channel_id=channel_id,
                title=f"Probe hit: {data.get('name') or 'probe'}",
                description=f"pos {hits[0].get('pos_score'):.3f} / neg {hits[0].get('neg_score'):.3f} · margin {hits[0].get('margin'):.3f}",
                severity=(data.get('severity') or 'critical'),
                state='new',
                timestamp_ms=hits[0].get('timestamp_ms'),
            )
        except Exception:
            pass
    if hits:
        # trim recent hits (kept in request payload only; not persisted unless saved)
        recent_hits = data.get('recent_hits') or []
        recent_hits = (recent_hits + hits)[:PROBE_MAX_STORED_HITS]
        result['recent_hits'] = recent_hits
    return jsonify(result), status_code


@app.route('/probes/status', methods=['GET'])
def probes_status():
    channel_id = request.args.get('channel_id', default=config.LUXRIOT_DEFAULT_CHANNEL_ID, type=int)
    try:
        return jsonify(probe_manager.status(channel_id))
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/probes/start_capture', methods=['POST'])
def probes_start_capture():
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    try:
        channel_id = int(data.get('channel_id') or config.LUXRIOT_DEFAULT_CHANNEL_ID)
    except Exception:
        return jsonify({'error': 'Provide a valid channel_id'}), 400
    try:
        fps = data.get('fps')
        fps_val = None
        try:
            if fps is not None:
                fps_val = float(fps)
        except Exception:
            fps_val = None
        clear_pause = _coerce_bool(data.get('clear_pause'), True)
        state = luxriot_manager.start_probe_capture(channel_id, fps=fps_val, clear_pause=clear_pause)
        return jsonify({'success': True, 'state': state})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/probes/stop_capture', methods=['POST'])
def probes_stop_capture():
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    try:
        channel_id = int(data.get('channel_id') or config.LUXRIOT_DEFAULT_CHANNEL_ID)
    except Exception:
        return jsonify({'error': 'Provide a valid channel_id'}), 400
    try:
        pause = _coerce_bool(data.get('pause'), True)
        state = luxriot_manager.stop_probe_capture(channel_id, pause=pause)
        return jsonify({'success': True, 'state': state})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/probes/save', methods=['POST'])
def probes_save():
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    try:
        channel_id = int(data.get('channel_id') or config.LUXRIOT_DEFAULT_CHANNEL_ID)
    except Exception:
        return jsonify({'error': 'Provide a valid channel_id'}), 400
    positives = [str(x).strip() for x in (data.get('positives') or []) if str(x).strip()]
    negatives = [str(x).strip() for x in (data.get('negatives') or []) if str(x).strip()]
    image_probe = data.get('image_probe') or {}
    if not positives and not (image_probe.get('data') and image_probe.get('enabled', True) is not False):
        return jsonify({'error': 'Provide at least one positive (text or image).'}), 400

    def _float(val, default):
        try:
            return float(val)
        except Exception:
            return default

    def _int(val, default):
        try:
            return int(val)
        except Exception:
            return default

    probe = {
        "id": data.get('id') or None,
        "name": (data.get('name') or '').strip() or f"probe-{int(time.time())}",
        "channel_id": channel_id,
        "positives": positives,
        "negatives": negatives,
        "pos_floor": _float(data.get('pos_floor'), 0.2),
        "margin": _float(data.get('margin'), 0.05),
        "top_k": _int(data.get('top_k'), 6),
        "window_sec": _float(data.get('window_sec'), 300.0),
        "severity": (data.get('severity') or 'critical').lower(),
        "bookmark": bool(data.get('bookmark', True)),
        "enabled": bool(data.get('enabled', True)),
        "image_probe": image_probe,
        "pairs": data.get('pairs') or [],
        "last_hit": data.get('last_hit'),
        "recent_hits": (data.get('recent_hits') or [])[:PROBE_MAX_STORED_HITS],
    }
    saved = probes_store.upsert_probe(probe)
    return jsonify({'success': True, 'probe': saved})


@app.route('/probes/list', methods=['GET'])
def probes_list():
    probes = probes_store.list_probes()
    return jsonify({'probes': probes})


@app.route('/probes/delete', methods=['POST'])
def probes_delete():
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    probe_id = data.get('id')
    if not probe_id:
        return jsonify({'error': 'Provide probe id'}), 400
    ok = probes_store.delete_probe(probe_id)
    if not ok:
        return jsonify({'error': 'Probe not found'}), 404
    return jsonify({'success': True})


@app.route('/probes/run', methods=['POST'])
def probes_run():
    guard = _mutation_guard_error()
    if guard is not None:
        return guard
    data = _json_body()
    probe_id = data.get('id')
    if not probe_id:
        return jsonify({'error': 'Provide probe id'}), 400
    probes = {p.get('id'): p for p in probes_store.list_probes()}
    probe = probes.get(probe_id)
    if not probe:
        return jsonify({'error': 'Probe not found'}), 404
    result = probe_manager.query(
        probe.get('channel_id', config.LUXRIOT_DEFAULT_CHANNEL_ID),
        probe.get('positives', []),
        probe.get('negatives', []),
        probe.get('pos_floor', 0.2),
        probe.get('margin', 0.05),
        probe.get('top_k', 6),
        window_sec=probe.get('window_sec', 300.0),
        image_probe=probe.get('image_probe'),
    )
    if 'error' in result:
        return jsonify(result), 400
    hits = result.get('results') or []
    if hits:
        probe['last_hit'] = hits[0]
        # keep a short rolling history of hits for UI while capping thumbnails
        recent = probe.get('recent_hits') or []
        recent = (hits + recent)[:PROBE_MAX_STORED_HITS]
        probe['recent_hits'] = recent
        probes_store.upsert_probe(probe)
        if probe.get('bookmark'):
            try:
                luxriot_manager.send_bookmark_event(
                    channel_id=probe.get('channel_id', config.LUXRIOT_DEFAULT_CHANNEL_ID),
                    title=f"Probe hit: {probe.get('name', 'probe')}",
                    description=f"pos {hits[0].get('pos_score'):.3f} / neg {hits[0].get('neg_score'):.3f} · margin {hits[0].get('margin'):.3f}",
                    severity=probe.get('severity', 'critical'),
                    state='new',
                    timestamp_ms=hits[0].get('timestamp_ms'),
                )
            except Exception:
                pass
    return jsonify({'results': hits, 'status': result.get('status'), 'probe': probe})


@app.route('/probes/bench', methods=['GET'])
def probes_bench():
    """Lightweight throughput estimate (image embedding)."""
    try:
        import torch  # type: ignore
        init_clip()
    except Exception:
        return jsonify({
            "error": "PyTorch/CLIP not available; install torch+clip to run benchmark."
        }), 400
    batch = int(request.args.get('batch', PROBE_BENCH_BATCH))
    batch = max(4, min(64, batch))
    try:
        # build random batch at 224x224
        rnd = torch.randint(0, 255, (batch, 3, 224, 224), device=device, dtype=torch.uint8)
        images = rnd.float() / 255.0
        started = time.time()
        with torch.no_grad():
            feats = clip_model.encode_image(images)  # type: ignore
            _ = feats.cpu()
        elapsed = time.time() - started
        fps = batch / elapsed if elapsed > 0 else 0
        return jsonify({
            "batch": batch,
            "elapsed_sec": round(elapsed, 3),
            "approx_fps": round(fps, 1),
            "device": device,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route('/settings', methods=['GET'])
def get_settings():
    """Get current configuration settings"""
    guard = _settings_guard(write=False)
    if guard is not None:
        return guard
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
            'luxriotBaseUrl': config.LUXRIOT_BASE_URL,
            'luxriotUsername': config.LUXRIOT_USERNAME,
            'luxriotPassword': '',
            'luxriotPasswordSet': bool(config.LUXRIOT_PASSWORD),
            'luxriotSnapshotInterval': config.LUXRIOT_SNAPSHOT_INTERVAL,
            'luxriotSnapshotMaxEdge': config.LUXRIOT_SNAPSHOT_MAX_EDGE,
            'luxriotDefaultChannelId': config.LUXRIOT_DEFAULT_CHANNEL_ID,
            'luxriotMaxBufferFrames': config.LUXRIOT_MAX_BUFFER_FRAMES,
            'luxriotAutoBookmarks': config.LUXRIOT_AUTO_BOOKMARKS,
            'luxriotSeverityMap': config.LUXRIOT_SEVERITY_MAP,
            'luxriotBatchSizes': list(config.LUXRIOT_BATCH_SIZES),
            'minResults': config.MIN_RESULTS,
            'maxResults': config.MAX_RESULTS,
            'defaultResults': config.DEFAULT_RESULTS,
            'batchSize': config.BATCH_SIZE,
            'thumbnailQuality': config.THUMBNAIL_QUALITY,
            'maxCommentLength': config.MAX_COMMENT_LENGTH,
            'maxFileSize': config.MAX_FILE_SIZE_MB,
            'indexFolderName': config.INDEX_FOLDER_NAME,
            'settingsLocalOnly': config.SETTINGS_LOCAL_ONLY,
            'adminTokenSet': bool(config.ADMIN_TOKEN),
            'corsAllowedOrigins': list(config.CORS_ALLOWED_ORIGINS),
            'allowedRoots': list(config.ALLOWED_ROOTS),
        }
        return jsonify({'success': True, 'settings': settings})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/settings', methods=['POST'])
def save_settings():
    """Save configuration settings to .env file"""
    guard = _settings_guard(write=True)
    if guard is not None:
        return guard
    try:
        data = _json_body()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        global active_embedder, clip_model, clip_preprocess, dino_encoder

        required_fields = ['host', 'port', 'debug', 'clipModel', 'minResults', 'maxResults', 'defaultResults']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400
        debug_enabled = _coerce_bool(data.get('debug', config.DEBUG), config.DEBUG)

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

        fusion_enabled = _coerce_bool(data.get('fusionEnabled', config.FUSION_ENABLED), config.FUSION_ENABLED)

        try:
            fusion_alpha = float(data.get('fusionAlpha', config.FUSION_ALPHA))
        except (TypeError, ValueError):
            fusion_alpha = config.FUSION_ALPHA
        fusion_alpha = min(1.0, max(0.0, fusion_alpha))

        rerank_enabled = _coerce_bool(data.get('rerankEnabled', config.RERANK_ENABLED), config.RERANK_ENABLED)

        try:
            rerank_top_k = int(data.get('rerankTopK', config.RERANK_TOP_K))
        except (TypeError, ValueError):
            rerank_top_k = config.RERANK_TOP_K
        if rerank_top_k < 1:
            rerank_top_k = 1

        segments_enabled = _coerce_bool(data.get('segmentsEnabled', config.DINO_SEGMENTS_ENABLED), config.DINO_SEGMENTS_ENABLED)

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

        luxriot_base_url = str(data.get('luxriotBaseUrl', config.LUXRIOT_BASE_URL)).strip().rstrip('/')
        luxriot_username = str(data.get('luxriotUsername', config.LUXRIOT_USERNAME)).strip()
        luxriot_password_raw = data.get('luxriotPassword', None)
        if luxriot_password_raw is None:
            luxriot_password = config.LUXRIOT_PASSWORD
        else:
            luxriot_password = str(luxriot_password_raw).strip() or config.LUXRIOT_PASSWORD
        try:
            luxriot_snapshot_interval = int(data.get('luxriotSnapshotInterval', config.LUXRIOT_SNAPSHOT_INTERVAL))
        except (TypeError, ValueError):
            luxriot_snapshot_interval = config.LUXRIOT_SNAPSHOT_INTERVAL
        if luxriot_snapshot_interval < 1:
            luxriot_snapshot_interval = 5
        try:
            luxriot_snapshot_max_edge = int(data.get('luxriotSnapshotMaxEdge', config.LUXRIOT_SNAPSHOT_MAX_EDGE))
        except (TypeError, ValueError):
            luxriot_snapshot_max_edge = config.LUXRIOT_SNAPSHOT_MAX_EDGE
        if luxriot_snapshot_max_edge < 640:
            luxriot_snapshot_max_edge = 640
        try:
            luxriot_default_channel_id = int(data.get('luxriotDefaultChannelId', config.LUXRIOT_DEFAULT_CHANNEL_ID))
        except (TypeError, ValueError):
            luxriot_default_channel_id = config.LUXRIOT_DEFAULT_CHANNEL_ID
        try:
            luxriot_max_buffer_frames = int(data.get('luxriotMaxBufferFrames', config.LUXRIOT_MAX_BUFFER_FRAMES))
        except (TypeError, ValueError):
            luxriot_max_buffer_frames = config.LUXRIOT_MAX_BUFFER_FRAMES
        if luxriot_max_buffer_frames < 12:
            luxriot_max_buffer_frames = 12
        luxriot_auto_bookmarks = _coerce_bool(
            data.get('luxriotAutoBookmarks', config.LUXRIOT_AUTO_BOOKMARKS),
            config.LUXRIOT_AUTO_BOOKMARKS,
        )
        severity_map = data.get('luxriotSeverityMap', {}) or {}
        merged_sev = dict(config.LUXRIOT_SEVERITY_MAP)
        for key in ['info', 'low', 'normal', 'high', 'critical']:
            if key in severity_map:
                merged_sev[key] = str(severity_map[key] or merged_sev.get(key, key)).lower()

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
EVOSSEARCH_DEBUG={str(debug_enabled).lower()}

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

# Luxriot Evo integration
EVOSSEARCH_LUXRIOT_BASE_URL={luxriot_base_url}
EVOSSEARCH_LUXRIOT_USERNAME={luxriot_username}
EVOSSEARCH_LUXRIOT_PASSWORD={luxriot_password}
EVOSSEARCH_LUXRIOT_DEFAULT_CHANNEL_ID={luxriot_default_channel_id}
EVOSSEARCH_LUXRIOT_SNAPSHOT_INTERVAL={luxriot_snapshot_interval}
EVOSSEARCH_LUXRIOT_SNAPSHOT_MAX_EDGE={luxriot_snapshot_max_edge}
EVOSSEARCH_LUXRIOT_MAX_BUFFER_FRAMES={luxriot_max_buffer_frames}
EVOSSEARCH_LUXRIOT_AUTO_BOOKMARKS={str(luxriot_auto_bookmarks).lower()}
EVOSSEARCH_LUXRIOT_SEV_INFO={merged_sev['info']}
EVOSSEARCH_LUXRIOT_SEV_LOW={merged_sev['low']}
EVOSSEARCH_LUXRIOT_SEV_NORMAL={merged_sev['normal']}
EVOSSEARCH_LUXRIOT_SEV_HIGH={merged_sev['high']}
EVOSSEARCH_LUXRIOT_SEV_CRITICAL={merged_sev['critical']}

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
EVOSSEARCH_ADMIN_TOKEN={config.ADMIN_TOKEN}
EVOSSEARCH_SETTINGS_LOCAL_ONLY={str(config.SETTINGS_LOCAL_ONLY).lower()}
EVOSSEARCH_CORS_ALLOWED_ORIGINS={','.join(config.CORS_ALLOWED_ORIGINS)}
EVOSSEARCH_ALLOWED_ROOTS={os.pathsep.join(config.ALLOWED_ROOTS)}
"""

        with open('.env', 'w', encoding='utf-8') as f:
            f.write(env_content)

        config.HOST = data['host']
        config.PORT = port
        config.DEBUG = debug_enabled
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
        app.config["MAX_CONTENT_LENGTH"] = max(1, int(config.MAX_FILE_SIZE_MB)) * 1024 * 1024
        config.FUSION_ENABLED = fusion_enabled
        config.FUSION_ALPHA = fusion_alpha
        config.RERANK_ENABLED = rerank_enabled
        config.RERANK_TOP_K = rerank_top_k
        config.DINO_SEGMENTS_ENABLED = segments_enabled
        config.DINO_SEGMENT_MIN_PATCHES = segment_min_patches
        config.DINO_HEATMAP_THRESHOLD = segment_threshold
        config.LUXRIOT_BASE_URL = luxriot_base_url
        config.LUXRIOT_USERNAME = luxriot_username
        config.LUXRIOT_PASSWORD = luxriot_password
        config.LUXRIOT_DEFAULT_CHANNEL_ID = luxriot_default_channel_id
        config.LUXRIOT_SNAPSHOT_INTERVAL = luxriot_snapshot_interval
        config.LUXRIOT_SNAPSHOT_MAX_EDGE = luxriot_snapshot_max_edge
        config.LUXRIOT_MAX_BUFFER_FRAMES = luxriot_max_buffer_frames
        config.LUXRIOT_AUTO_BOOKMARKS = luxriot_auto_bookmarks
        config.LUXRIOT_SEVERITY_MAP = merged_sev

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


def _stop_probe_daemon_thread() -> None:
    global probe_daemon_thread
    probe_daemon_stop.set()
    if probe_daemon_thread is not None and probe_daemon_thread.is_alive():
        probe_daemon_thread.join(timeout=1.5)


@atexit.register
def _shutdown_background_workers() -> None:
    try:
        luxriot_manager.stop_all_streams(stop_video=True, stop_analytics=True, pause_analytics=False)
    except Exception:
        pass
    try:
        _stop_probe_daemon_thread()
    except Exception:
        pass


if __name__ == '__main__':
    ensure_embedder_loaded()
    config.print_startup_info()
    if probe_daemon_thread is None:
        probe_daemon_thread = threading.Thread(target=_probe_daemon, daemon=True)
        probe_daemon_thread.start()
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
