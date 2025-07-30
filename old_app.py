"""
Refactored CLIP‑powered image search server — v2 with enhanced UI
————————————————————————————————————————————————————————
• Responsive grid now truly fills ultra‑wide monitors (container 100%,
  grid auto‑fit 240 px cells).
• Clicking a thumbnail sets it as a blurred, full‑page background.
• Control + search panels sport a subtle glass‑morphic sheen (backdrop‑filter).
"""

from __future__ import annotations

import base64
import os
import pickle
import threading
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import faiss  # type: ignore
import numpy as np
import torch
from PIL import Image
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS
import clip  # type: ignore

###############################################################################
# Configuration
###############################################################################
BATCH_SIZE = 32
THUMBNAIL_SIZE = (400, 400)
SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    _supported_arches = {50, 60, 70, 75, 80, 86, 90}
    DEVICE = "cuda" if major * 10 + minor in _supported_arches else "cpu"
else:
    DEVICE = "cpu"

###############################################################################
# Flask app
###############################################################################
app = Flask(__name__)
CORS(app)
model = preprocess = None  # type: ignore

###############################################################################
# Model helpers
###############################################################################

def init_clip():
    global model, preprocess
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model.eval()


def _encode_batch(t: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        feat = model.encode_image(t)  # type: ignore[attr-defined]
        feat /= feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype("float32")


def embed_text(txt: str) -> np.ndarray:
    tok = clip.tokenize([txt]).to(DEVICE)
    with torch.no_grad():
        feat = model.encode_text(tok)  # type: ignore[attr-defined]
        feat /= feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype("float32").flatten()

###############################################################################
# Index helpers (unchanged)
###############################################################################

def _imgs(folder: Path):
    return [p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXT]


def _load_paths(d: Path):
    try:
        with open(d / "paths.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return []


def create_or_update_index(folder: Path):
    d = folder / ".clip_index"
    d.mkdir(exist_ok=True)
    idx = None
    prev = _load_paths(d)
    if (d / "index.faiss").exists():
        idx = faiss.read_index(str(d / "index.faiss"))
    new_paths = [str(p) for p in _imgs(folder) if str(p) not in prev]
    if not new_paths and idx is not None:
        return idx, prev
    embs = []
    for i in range(0, len(new_paths), BATCH_SIZE):
        bt = torch.cat([preprocess(Image.open(p)).unsqueeze(0) for p in new_paths[i:i+BATCH_SIZE]]).to(DEVICE)
        embs.append(_encode_batch(bt))
    arr = np.vstack(embs) if embs else np.empty((0,512),dtype='float32')
    if idx is None:
        idx = faiss.IndexFlatIP(arr.shape[1])
    if arr.size:
        idx.add(arr)
        prev.extend(new_paths)
    faiss.write_index(idx, str(d / "index.faiss"))
    with open(d / "paths.pkl", "wb") as f:
        pickle.dump(prev, f)
    return idx, prev


def load_index(folder: Path):
    d = folder / ".clip_index"
    if not (d / "index.faiss").exists():
        return None, []
    return faiss.read_index(str(d / "index.faiss")), _load_paths(d)

###############################################################################
# Utils
###############################################################################

def _safe_path(p: str) -> Path:
    if ".." in p:
        raise ValueError("Parent refs not allowed")
    q = Path(p).expanduser().resolve()
    if not q.is_dir():
        raise ValueError("Not a directory")
    return q

###############################################################################
# Frontend (responsive grid + glassmorph & dynamic bg)
###############################################################################

FRONT_HTML = r"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='UTF-8'>
<meta name='viewport' content='width=device-width,initial-scale=1.0'>
<title>CLIP Image Search</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0a0a0a;color:#e0e0e0;min-height:100vh;display:flex;flex-direction:column;background-size:cover;background-position:center;transition:background-image .6s ease-in-out}
body::before{content:"";position:fixed;inset:0;background:rgba(0,0,0,.5);backdrop-filter:blur(6px);pointer-events:none;transition:opacity .3s}
.container{width:100%;max-width:1800px;margin:0 auto;padding:2rem;flex:1}
h1{font-size:2rem;font-weight:300;margin-bottom:2rem;letter-spacing:-.02em}
.control-panel,.search-panel{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.15);border-radius:16px;padding:1.5rem;margin-bottom:2rem;backdrop-filter:blur(14px) saturate(160%)}
.folder-select,.search-box{display:flex;gap:1rem;margin-bottom:1rem}
input[type=text]{flex:1;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.2);padding:.75rem 1rem;border-radius:12px;color:#e0e0e0;font-size:.95rem;transition:border .2s,background .2s}
input[type=text]:focus{outline:none;border-color:#fff;background:rgba(255,255,255,.15)}
button{background:rgba(0,0,0,.4);border:1px solid rgba(255,255,255,.25);color:#e0e0e0;padding:.75rem 1.5rem;border-radius:12px;cursor:pointer;font-size:.95rem;transition:background .2s,border .2s}
button:hover{background:rgba(0,0,0,.55);border-color:#fff}
button:active{transform:translateY(1px)}
.status{font-size:.875rem;color:#888;margin-top:.5rem}
.status.success{color:#4ade80}.status.error{color:#f87171}
.results-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:1.5rem;transition:grid-template-columns .2s}
.result-item{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);border-radius:12px;overflow:hidden;cursor:pointer;transition:transform .3s,border .3s}
.result-item:hover{transform:translateY(-2px);border-color:#fff}
.result-item.expanded{grid-column:span 2}
.thumbnail{width:100%;height:170px;object-fit:cover;display:block}
.result-item.expanded .thumbnail{height:auto;max-height:500px}
.result-info{padding:.75rem;font-size:.875rem}
.filename{color:#e0e0e0;margin-bottom:.25rem;word-break:break-all}
.similarity{color:#aaa;font-size:.8rem}
.loading{text-align:center;padding:2rem;color:#aaa}
.spinner{display:inline-block;width:22px;height:22px;border:3px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div class='container'>
<h1>Natural Language Image Search</h1>
<div class='control-panel'>
  <div class='folder-select'>
    <input type='text' id='folderPath' placeholder='Enter folder path...' />
    <button id='indexBtn'>Index Folder</button>
  </div>
  <div class='status' id='indexStatus'></div>
</div>
<div class='search-panel'>
  <div class='search-box'>
    <input type='text' id='searchQuery' placeholder="Describe what you're looking for..." />
    <button id='searchBtn'>Search</button>
  </div>
</div>
<div id='results' class='results-grid'></div>
</div>
<script>
const $=q=>document.querySelector(q);
const folderInput=$('#folderPath');
const indexBtn=$('#indexBtn');
const indexStatus=$('#indexStatus');
const searchInput=$('#searchQuery');
const searchBtn=$('#searchBtn');
const results=$('#results');

async function check(folder){try{const r=await fetch('/check_index',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({folder})});return (await r.json()).indexed}catch{ return false;}}

indexBtn.onclick=async()=>{const f=folderInput.value.trim();if(!f)return;indexStatus.textContent='Indexing...';indexStatus.className='status';indexBtn.disabled=true;try{const r=await fetch('/index',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({folder:f})});const d=await r.json();if(d.started){indexStatus.textContent='Indexing started — refresh soon';}else if(d.error){indexStatus.textContent=d.error;indexStatus.className='status error';}}catch(e){indexStatus.textContent='Error '+e;indexStatus.className='status error';}finally{indexBtn.disabled=false;}
};

searchBtn.onclick=runSearch;
async function runSearch(){const q=searchInput.value.trim();const f=folderInput.value.trim();if(!q||!f)return;results.innerHTML='<div class="loading"><div class="spinner"></div> Searching...</div>';
try{const r=await fetch('/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({folder:f,query:q})});const d=await r.json();
if(d.results)render(d.results);else results.textContent='No results';}catch(e){results.textContent='Error '+e;}}

function render(arr){results.innerHTML='';arr.forEach(o=>{const div=document.createElement('div');div.className='result-item';div.innerHTML=`<img class='thumbnail' src="data:image/jpeg;base64,${o.thumbnail}" alt=''><div class='result-info'><div class='filename'>${o.filename}</div><div class='similarity'>${(o.similarity*100).toFixed(1)}%</div></div>`;
  div.onclick=()=>{document.body.style.backgroundImage=`url(data:image/jpeg;base64,${o.thumbnail})`;div.classList.toggle('expanded');};
  results.appendChild(div);});}

searchInput.addEventListener('keypress',e=>e.key==='Enter'&&runSearch());folderInput.addEventListener('keypress',e=>e.key==='Enter'&&indexBtn.click());folderInput.addEventListener('blur',async()=>{const f=folderInput.value.trim();if(f){indexStatus.textContent=(await check(f))?'Folder indexed':'Folder not indexed';indexStatus.className='status';}});
</script>
</body>
</html>"""

###############################################################################
# Routes
###############################################################################

@app.route('/')
def home():
    return render_template_string(FRONT_HTML)

@app.route('/check_index', methods=['POST'])
def check_index():
    try:
        folder = _safe_path(request.json['folder'])
    except (KeyError, ValueError) as e:
        return jsonify({'error': str(e)}), 400
    idx,_ = load_index(folder)
    return jsonify({'indexed': idx is not None})

@app.route('/index', methods=['POST'])
def index_folder():
    try:
        folder = _safe_path(request.json['folder'])
    except (KeyError, ValueError) as e:
        return jsonify({'error': str(e)}), 400
    threading.Thread(target=lambda: create_or_update_index(folder),daemon=True).start()
    return jsonify({'started': True})

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    try:
        folder = _safe_path(data['folder'])
    except (KeyError, ValueError) as e:
        return jsonify({'error': str(e)}), 400
    query = data.get('query','').strip()
    if not query:
        return jsonify({'error': 'Query empty'}), 400
    idx, paths = load_index(folder)
    if idx is None:
        return jsonify({'error': 'Folder not indexed'}), 400
    emb = embed_text(query)
    k=min(10,len(paths))
    sims,ids = idx.search(emb.reshape(1,-1),k)
    res=[]
    for i,sim in zip(ids[0],sims[0]):
        if i>=len(paths): continue
        p=paths[i]
        img=Image.open(p);img.thumbnail(THUMBNAIL_SIZE,Image.Resampling.LANCZOS)
        buf=BytesIO();img.save(buf,'JPEG',quality=85)
        res.append({'path':p,'filename':os.path.basename(p),'similarity':float(sim),'thumbnail':base64.b64encode(buf.getvalue()).decode()})
    return jsonify({'results':res})

###############################################################################
if __name__=='__main__':
    init_clip()
    app.run(host='0.0.0.0',port=5000,debug=True)
