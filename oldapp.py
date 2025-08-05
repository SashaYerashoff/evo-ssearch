import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pickle
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template_string, make_response
from flask_cors import CORS
import torch
import clip
import faiss
from PIL import Image
from pathlib import Path
import base64
from io import BytesIO
from config import config
import time

app = Flask(__name__)
CORS(app)

# Global variables
model = None
preprocess = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def init_clip():
    """Initialize CLIP model"""
    global model, preprocess
    model, preprocess = clip.load(config.CLIP_MODEL, device=device)
    
def get_image_embedding(image_path):
    """Extract CLIP embedding from image"""
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy().flatten()

def get_image_embedding_from_pil(pil_image):
    """Extract CLIP embedding from PIL Image"""
    image = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy().flatten()

def get_text_embedding(text):
    """Extract CLIP embedding from text"""
    text_tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy().flatten()

def create_index(folder_path):
    """Create FAISS index for folder"""
    folder_path = Path(folder_path)
    image_paths = []
    embeddings = []
    image_metadata = []
    
    # Supported image formats
    extensions = config.SUPPORTED_EXTENSIONS
    
    for ext in extensions:
        for img_path in folder_path.glob(f'*{ext}'):
            try:
                embedding = get_image_embedding(img_path)
                embeddings.append(embedding)
                image_paths.append(str(img_path))
                
                # Get file metadata
                stat = img_path.stat()
                metadata = {
                    'path': str(img_path),
                    'mtime': stat.st_mtime,
                    'size': stat.st_size
                }
                image_metadata.append(metadata)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    if not embeddings:
        return None, None, None
    
    # Create FAISS index
    embeddings_array = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatIP(embeddings_array.shape[1])  # Inner product for cosine similarity
    index.add(embeddings_array)
    
    return index, image_paths, image_metadata

def save_index(index, image_paths, image_metadata, folder_path):
    """Save FAISS index and metadata"""
    index_path = Path(folder_path) / config.INDEX_FOLDER_NAME
    index_path.mkdir(exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, str(index_path / 'index.faiss'))
    
    # Save image paths
    with open(index_path / 'paths.pkl', 'wb') as f:
        pickle.dump(image_paths, f)
    
    # Save image metadata
    with open(index_path / 'metadata.pkl', 'wb') as f:
        pickle.dump(image_metadata, f)

def load_index(folder_path):
    """Load FAISS index and metadata"""
    index_path = Path(folder_path) / config.INDEX_FOLDER_NAME
    
    if not index_path.exists():
        return None, None, None
    
    try:
        # Load FAISS index
        index = faiss.read_index(str(index_path / 'index.faiss'))
        
        # Load image paths
        with open(index_path / 'paths.pkl', 'rb') as f:
            image_paths = pickle.load(f)
        
        # Load image metadata (backwards compatible)
        image_metadata = None
        metadata_file = index_path / 'metadata.pkl'
        if metadata_file.exists():
            try:
                with open(metadata_file, 'rb') as f:
                    image_metadata = pickle.load(f)
            except:
                image_metadata = None
        
        return index, image_paths, image_metadata
    except:
        return None, None, None

def load_comments(folder_path):
    """Load comments from JSON file"""
    index_path = Path(folder_path) / config.INDEX_FOLDER_NAME
    comments_file = index_path / 'comments.json'
    
    if not comments_file.exists():
        return {}
    
    try:
        import json
        with open(comments_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}

def save_comments(folder_path, comments_data):
    """Save comments to JSON file"""
    index_path = Path(folder_path) / config.INDEX_FOLDER_NAME
    index_path.mkdir(exist_ok=True)
    comments_file = index_path / 'comments.json'
    
    try:
        import json
        with open(comments_file, 'w', encoding='utf-8') as f:
            json.dump(comments_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving comments: {e}")
        return False

def get_image_comments(folder_path, image_path):
    """Get comments for specific image"""
    comments_data = load_comments(folder_path)
    return comments_data.get(image_path, [])

def add_image_comment(folder_path, image_path, comment):
    """Add new comment to image"""
    comments_data = load_comments(folder_path)
    
    if image_path not in comments_data:
        comments_data[image_path] = []
    
    # Add timestamp to comment
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    comment_with_timestamp = f"[{timestamp}] {comment}"
    
    comments_data[image_path].append(comment_with_timestamp)
    
    return save_comments(folder_path, comments_data)

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
            color: #888;
            font-size: 0.8rem;
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
        const textSearchBox = document.getElementById('textSearchBox');
        const imageSearchBox = document.getElementById('imageSearchBox');
        const resultLimitSelect = document.getElementById('resultLimit');
        const sortBySelect = document.getElementById('sortBy');
        const showCommentedBtn = document.getElementById('showCommentedBtn');
        const resultsContainer = document.getElementById('results');
        
        let currentFolder = '';
        let currentMode = 'text';
        
        // Settings modal elements
        const settingsBtn = document.getElementById('settingsBtn');
        const settingsModal = document.getElementById('settingsModal');
        const closeSettingsBtn = document.getElementById('closeSettings');
        const saveSettingsBtn = document.getElementById('saveSettings');
        const resetSettingsBtn = document.getElementById('resetSettings');
        const settingsStatus = document.getElementById('settingsStatus');
        const thumbnailQualitySlider = document.getElementById('thumbnailQuality');
        const qualityValue = document.getElementById('qualityValue');
        
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
                    clipModel: document.getElementById('clipModel').value,
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
        
        // Mode switching
        textModeBtn.addEventListener('click', () => {
            currentMode = 'text';
            textModeBtn.classList.add('active');
            imageModeBtn.classList.remove('active');
            textSearchBox.style.display = 'flex';
            imageSearchBox.style.display = 'none';
        });
        
        imageModeBtn.addEventListener('click', () => {
            currentMode = 'image';
            imageModeBtn.classList.add('active');
            textModeBtn.classList.remove('active');
            imageSearchBox.style.display = 'flex';
            textSearchBox.style.display = 'none';
        });
        
        // Check index status
        async function checkIndexStatus(folder) {
            try {
                const response = await fetch('/check_index', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ folder })
                });
                const data = await response.json();
                return data.indexed;
            } catch (error) {
                return false;
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
                    indexStatus.textContent = `Indexed ${data.count} images successfully`;
                    indexStatus.className = 'status success';
                    currentFolder = folder;
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
        
        // Show commented images
        showCommentedBtn.addEventListener('click', async () => {
            const folder = folderInput.value.trim();
            
            if (!folder) {
                alert('Please enter a folder path first');
                return;
            }
            
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
            const similarityText = isCommented 
                ? `Comments: ${result.comment_count} | Latest: ${result.latest_comment.substring(0, 50)}${result.latest_comment.length > 50 ? '...' : ''}`
                : `Similarity: ${(result.similarity * 100).toFixed(1)}%`;
                
            return `
                <div class="image-container">
                    <img src="data:image/jpeg;base64,${result.thumbnail}" class="thumbnail" alt="" />
                    <div class="image-overlay">
                        <div class="expand-collapse-icon" data-index="${index}">
                            <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="#e3e3e3">
                                <path d="M240-240v-240h72v168h168v72H240Zm408-240v-168H480v-72h240v240h-72Z"/>
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
                    <div class="similarity">${similarityText}</div>
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
            copyIcon.addEventListener('click', (e) => {
                e.stopPropagation();
                copyImagePath(result.path);
            });
            
            
            // Handle find similar button
            const findSimilarIcon = item.querySelector('.find-similar-icon');
            findSimilarIcon.addEventListener('click', (e) => {
                e.stopPropagation();
                findSimilarImages(result.path);
            });
            
            // Add save comment functionality
            const saveBtn = item.querySelector(`#save-btn-${index}`);
            const commentInput = item.querySelector(`#comment-input-${index}`);
            
            saveBtn.addEventListener('click', () => {
                saveComment(index, result.path, folderInput.value.trim(), commentInput.value.trim());
            });
        }

        // Display results
        function displayResults(results) {
            resultsContainer.innerHTML = '';
            
            results.forEach((result, index) => {
                const item = document.createElement('div');
                item.className = 'result-item';
                item.innerHTML = generateResultItemHTML(result, index, false);
                
                setupResultItemEventHandlers(item, result, index);
                resultsContainer.appendChild(item);
            });
        }
        
        // Display commented results (similar to displayResults but with comment info)
        function displayCommentedResults(results) {
            resultsContainer.innerHTML = '';
            
            results.forEach((result, index) => {
                const item = document.createElement('div');
                item.className = 'result-item';
                item.innerHTML = generateResultItemHTML(result, index, true);
                
                setupResultItemEventHandlers(item, result, index);
                resultsContainer.appendChild(item);
            });
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
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        function toggleImageExpansion(item, result, index) {
            const img = item.querySelector('.thumbnail');
            const expandCollapseIcon = item.querySelector('.expand-collapse-icon');
            const isExpanded = item.classList.contains('expanded');
            
            if (isExpanded) {
                // Collapse: switch back to thumbnail
                img.src = `data:image/jpeg;base64,${result.thumbnail}`;
                item.classList.remove('expanded');
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
                // Update icon to collapse
                expandCollapseIcon.innerHTML = `
                    <svg xmlns="http://www.w3.org/2000/svg" height="20px" viewBox="0 -960 960 960" width="20px" fill="#e3e3e3">
                        <path d="M432-432v240h-72v-168H192v-72h240Zm168-336v168h168v72H528v-240h72Z"/>
                    </svg>
                `;
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
                const indexed = await checkIndexStatus(folder);
                if (indexed) {
                    indexStatus.textContent = 'Folder is indexed';
                    indexStatus.className = 'status success';
                } else {
                    indexStatus.textContent = 'Folder not indexed';
                    indexStatus.className = 'status';
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
    """Serve original images"""
    try:
        # Security check - prevent directory traversal
        if '..' in filepath or filepath.startswith('/'):
            return "Access denied", 403
        
        # Convert to absolute path and check if file exists
        abs_path = os.path.abspath(filepath)
        if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
            return "Image not found", 404
            
        return send_file(abs_path)
    except Exception as e:
        return f"Error serving image: {str(e)}", 500

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
        index, image_paths, image_metadata = load_index(folder)
        if index is None:
            return jsonify({'error': 'Folder not indexed'}), 400
        
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
    
    index, _, _ = load_index(folder)
    return jsonify({'indexed': index is not None})

@app.route('/index', methods=['POST'])
def index_folder():
    """Index a folder"""
    folder = request.json.get('folder')
    if not folder or not os.path.exists(folder):
        return jsonify({'error': 'Invalid folder path'}), 400
    
    try:
        index, image_paths, image_metadata = create_index(folder)
        if index is None:
            return jsonify({'error': 'No images found in folder'}), 400
        
        save_index(index, image_paths, image_metadata, folder)
        return jsonify({'success': True, 'count': len(image_paths)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    """Search for images"""
    folder = request.json.get('folder')
    query = request.json.get('query')
    limit = request.json.get('limit', 10)
    sort_by = request.json.get('sort_by', 'similarity')  # 'similarity' or 'time'
    print(f"Search request: folder={folder}, query={query}, limit={limit}, sort_by={sort_by}")
    
    if not folder or not query:
        return jsonify({'error': 'Missing folder or query'}), 400
    
    # Validate limit
    try:
        limit = int(limit)
        if limit < config.MIN_RESULTS or limit > config.MAX_RESULTS:
            limit = config.DEFAULT_RESULTS
    except (ValueError, TypeError):
        limit = config.DEFAULT_RESULTS
    
    # Load index
    index, image_paths, image_metadata = load_index(folder)
    if index is None:
        return jsonify({'error': 'Folder not indexed'}), 400
    
    try:
        # Get text embedding
        text_embedding = get_text_embedding(query)
        
        # Search
        k = min(limit, len(image_paths))
        if k == 0:
            return jsonify({'results': []})
        similarities, indices = index.search(text_embedding.reshape(1, -1), k)
        
        results = []
        for i, (idx, sim) in enumerate(zip(indices[0], similarities[0])):
            if idx >= 0 and idx < len(image_paths):
                try:
                    img_path = image_paths[idx]
                    
                    # Create thumbnail
                    img = Image.open(img_path)
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
                        'path': img_path,
                        'filename': os.path.basename(img_path),
                        'similarity': float(sim),
                        'thumbnail': img_base64,
                        'metadata': metadata_info
                    })
                except Exception as img_error:
                    print(f"Error processing image {img_path}: {img_error}")
                    continue
        
        # Sort results based on sort_by parameter
        if sort_by == 'time' and image_metadata:
            # Sort by modification time (newest first)
            results.sort(key=lambda x: x['metadata'].get('mtime', 0), reverse=True)
        # Otherwise keep similarity sort (default FAISS order)
        
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
    
    # Validate limit
    try:
        limit = int(limit)
        if limit < config.MIN_RESULTS or limit > config.MAX_RESULTS:
            limit = config.DEFAULT_RESULTS
    except (ValueError, TypeError):
        limit = config.DEFAULT_RESULTS
    
    # Check for either uploaded file or image path
    file = request.files.get('image')
    image_path = request.form.get('image_path')
    
    if not file and not image_path:
        return jsonify({'error': 'No image uploaded or path provided'}), 400
    
    if file and file.filename == '':
        file = None
    
    # Load index
    index, image_paths, image_metadata = load_index(folder)
    if index is None:
        return jsonify({'error': 'Folder not indexed'}), 400
    
    try:
        # Process image from either file upload or path
        if file:
            # Process uploaded image
            uploaded_image = Image.open(file.stream)
            if uploaded_image.mode != 'RGB':
                uploaded_image = uploaded_image.convert('RGB')
            # Get image embedding
            image_embedding = get_image_embedding_from_pil(uploaded_image)
        else:
            # Process image from path
            if not os.path.exists(image_path):
                return jsonify({'error': f'Image file not found: {image_path}'}), 400
            
            try:
                # Get image embedding directly from path
                image_embedding = get_image_embedding(image_path)
            except Exception as path_error:
                return jsonify({'error': f'Error processing image from path: {str(path_error)}'}), 400
        
        # Search
        k = min(limit, len(image_paths))
        if k == 0:
            return jsonify({'results': []})
        similarities, indices = index.search(image_embedding.reshape(1, -1), k)
        
        results = []
        for i, (idx, sim) in enumerate(zip(indices[0], similarities[0])):
            if idx >= 0 and idx < len(image_paths):
                try:
                    img_path = image_paths[idx]
                    
                    # Create thumbnail
                    img = Image.open(img_path)
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
                        'path': img_path,
                        'filename': os.path.basename(img_path),
                        'similarity': float(sim),
                        'thumbnail': img_base64,
                        'metadata': metadata_info
                    })
                except Exception as img_error:
                    print(f"Error processing image {img_path}: {img_error}")
                    continue
        
        # Sort results based on sort_by parameter
        if sort_by == 'time' and image_metadata:
            # Sort by modification time (newest first)
            results.sort(key=lambda x: x['metadata'].get('mtime', 0), reverse=True)
        # Otherwise keep similarity sort (default FAISS order)
        
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/settings', methods=['GET'])
def get_settings():
    """Get current configuration settings"""
    try:
        settings = {
            'host': config.HOST,
            'port': config.PORT,
            'debug': config.DEBUG,
            'clipModel': config.CLIP_MODEL,
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
        
        # Validate required fields
        required_fields = ['host', 'port', 'debug', 'clipModel', 'minResults', 'maxResults', 'defaultResults']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400
        
        # Validate data types and ranges
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
        
        # Create .env file content
        env_content = f"""# evo-ssearch Configuration
# Generated by settings panel

# Server Configuration
EVOSSEARCH_HOST={data['host']}
EVOSSEARCH_PORT={data['port']}
EVOSSEARCH_DEBUG={str(data['debug']).lower()}

# CLIP model configuration
EVOSSEARCH_CLIP_MODEL={data['clipModel']}

# Search result limits
EVOSSEARCH_MIN_RESULTS={data['minResults']}
EVOSSEARCH_MAX_RESULTS={data['maxResults']}
EVOSSEARCH_DEFAULT_RESULTS={data['defaultResults']}

# Processing configuration
EVOSSEARCH_BATCH_SIZE={data.get('batchSize', 32)}
EVOSSEARCH_THUMBNAIL_QUALITY={data.get('thumbnailQuality', 85)}

# File system configuration
EVOSSEARCH_INDEX_FOLDER={data.get('indexFolderName', '.clip_index')}

# Comment system configuration
EVOSSEARCH_MAX_COMMENT_LENGTH={data.get('maxCommentLength', 100)}

# Security configuration
EVOSSEARCH_MAX_FILE_SIZE_MB={data.get('maxFileSize', 50)}
"""
        
        # Write to .env file
        with open('.env', 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        return jsonify({'success': True, 'message': 'Settings saved successfully. Restart the server to apply changes.'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    init_clip()
    config.print_startup_info()
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
