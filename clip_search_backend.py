import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
import torch
import clip
import faiss
from PIL import Image
from pathlib import Path
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Global variables
model = None
preprocess = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def init_clip():
    """Initialize CLIP model"""
    global model, preprocess
    model, preprocess = clip.load("ViT-B/32", device=device)
    
def get_image_embedding(image_path):
    """Extract CLIP embedding from image"""
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
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
    
    # Supported image formats
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    for ext in extensions:
        for img_path in folder_path.glob(f'*{ext}'):
            try:
                embedding = get_image_embedding(img_path)
                embeddings.append(embedding)
                image_paths.append(str(img_path))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    if not embeddings:
        return None, None
    
    # Create FAISS index
    embeddings_array = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatIP(embeddings_array.shape[1])  # Inner product for cosine similarity
    index.add(embeddings_array)
    
    return index, image_paths

def save_index(index, image_paths, folder_path):
    """Save FAISS index and metadata"""
    index_path = Path(folder_path) / '.clip_index'
    index_path.mkdir(exist_ok=True)
    
    # Save FAISS index
    faiss.write_index(index, str(index_path / 'index.faiss'))
    
    # Save image paths
    with open(index_path / 'paths.pkl', 'wb') as f:
        pickle.dump(image_paths, f)

def load_index(folder_path):
    """Load FAISS index and metadata"""
    index_path = Path(folder_path) / '.clip_index'
    
    if not index_path.exists():
        return None, None
    
    try:
        # Load FAISS index
        index = faiss.read_index(str(index_path / 'index.faiss'))
        
        # Load image paths
        with open(index_path / 'paths.pkl', 'rb') as f:
            image_paths = pickle.load(f)
        
        return index, image_paths
    except:
        return None, None

@app.route('/')
def home():
    """Serve the frontend"""
    return render_template_string('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Natural Language Image Search</title>
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
            margin: 0 auto;
            padding: 2rem;
            flex: 1;
        }
        
        h1 {
            font-size: 2rem;
            font-weight: 300;
            margin-bottom: 2rem;
            letter-spacing: -0.02em;
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
        
        .search-box {
            display: flex;
            gap: 1rem;
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
            grid-column: span 2;
        }
        
        .thumbnail {
            width: 100%;
            height: 150px;
            object-fit: cover;
            display: block;
        }
        
        .result-item.expanded .thumbnail {
            height: auto;
            max-height: 400px;
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
    </style>
</head>
<body>
    <div class="container">
        <h1>Natural Language Image Search</h1>
        
        <div class="control-panel">
            <div class="folder-select">
                <input type="text" id="folderPath" placeholder="Enter folder path..." />
                <button id="indexBtn">Index Folder</button>
            </div>
            <div class="status" id="indexStatus"></div>
        </div>
        
        <div class="search-panel">
            <div class="search-box">
                <input type="text" id="searchQuery" placeholder="Describe what you're looking for..." />
                <button id="searchBtn">Search</button>
            </div>
        </div>
        
        <div id="results" class="results-grid"></div>
    </div>
    
    <script>
        const folderInput = document.getElementById('folderPath');
        const indexBtn = document.getElementById('indexBtn');
        const indexStatus = document.getElementById('indexStatus');
        const searchInput = document.getElementById('searchQuery');
        const searchBtn = document.getElementById('searchBtn');
        const resultsContainer = document.getElementById('results');
        
        let currentFolder = '';
        
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
        
        // Search
        searchBtn.addEventListener('click', async () => {
            const query = searchInput.value.trim();
            const folder = folderInput.value.trim();
            
            if (!query || !folder) return;
            
            resultsContainer.innerHTML = '<div class="loading"><div class="spinner"></div> Searching...</div>';
            
            try {
                const response = await fetch('/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ folder, query })
                });
                
                const data = await response.json();
                
                if (data.results) {
                    displayResults(data.results);
                } else {
                    resultsContainer.innerHTML = '<div class="loading">No results found</div>';
                }
            } catch (error) {
                resultsContainer.innerHTML = '<div class="loading">Error: ' + error.message + '</div>';
            }
        });
        
        // Display results
        function displayResults(results) {
            resultsContainer.innerHTML = '';
            
            results.forEach((result, index) => {
                const item = document.createElement('div');
                item.className = 'result-item';
                item.innerHTML = `
                    <img src="data:image/jpeg;base64,${result.thumbnail}" class="thumbnail" alt="" />
                    <div class="result-info">
                        <div class="filename">${result.filename}</div>
                        <div class="similarity">Similarity: ${(result.similarity * 100).toFixed(1)}%</div>
                    </div>
                `;
                
                item.addEventListener('click', () => {
                    item.classList.toggle('expanded');
                });
                
                resultsContainer.appendChild(item);
            });
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
    ''')

@app.route('/check_index', methods=['POST'])
def check_index():
    """Check if folder is indexed"""
    folder = request.json.get('folder')
    if not folder:
        return jsonify({'error': 'No folder specified'}), 400
    
    index, _ = load_index(folder)
    return jsonify({'indexed': index is not None})

@app.route('/index', methods=['POST'])
def index_folder():
    """Index a folder"""
    folder = request.json.get('folder')
    if not folder or not os.path.exists(folder):
        return jsonify({'error': 'Invalid folder path'}), 400
    
    try:
        index, image_paths = create_index(folder)
        if index is None:
            return jsonify({'error': 'No images found in folder'}), 400
        
        save_index(index, image_paths, folder)
        return jsonify({'success': True, 'count': len(image_paths)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    """Search for images"""
    folder = request.json.get('folder')
    query = request.json.get('query')
    
    if not folder or not query:
        return jsonify({'error': 'Missing folder or query'}), 400
    
    # Load index
    index, image_paths = load_index(folder)
    if index is None:
        return jsonify({'error': 'Folder not indexed'}), 400
    
    try:
        # Get text embedding
        text_embedding = get_text_embedding(query)
        
        # Search
        k = min(10, len(image_paths))
        similarities, indices = index.search(text_embedding.reshape(1, -1), k)
        
        results = []
        for i, (idx, sim) in enumerate(zip(indices[0], similarities[0])):
            if idx < len(image_paths):
                img_path = image_paths[idx]
                
                # Create thumbnail
                img = Image.open(img_path)
                img.thumbnail((400, 400), Image.Resampling.LANCZOS)
                
                # Convert to base64
                buffer = BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                
                results.append({
                    'path': img_path,
                    'filename': os.path.basename(img_path),
                    'similarity': float(sim),
                    'thumbnail': img_base64
                })
        
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_clip()
    app.run(debug=True, port=5000)
