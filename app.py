"""
Refactored CLIP‑powered image search server — v3 UI tweaks
———————————————————————————————————————————————————————————————————
• Container given higher z‑index so heading & results always sit above overlay.
• Full‑view background now fills screen (background‑size:cover already, z‑fix).
• Overlay blur dialled back (4 px) and given a top‑to‑bottom gradient so top
  half stays sharper, improving legibility.
"""

from __future__ import annotations

import base64, os, pickle, threading
from io import BytesIO
from pathlib import Path

import faiss, numpy as np, torch, clip  # type: ignore
from PIL import Image
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS

BATCH_SIZE, THUMBNAIL_SIZE = 32, (400, 400)
SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
DEVICE = "cuda" if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0]*10+torch.cuda.get_device_capability(0)[1] in {50,60,70,75,80,86,90} else "cpu"

app = Flask(__name__); CORS(app)
model = preprocess = None  # type: ignore

def init_clip():
    global model, preprocess
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model.eval()

def _encode(t: torch.Tensor):
    with torch.no_grad():
        f = model.encode_image(t); f /= f.norm(dim=-1, keepdim=True)
    return f.cpu().numpy().astype('float32')

def embed_text(txt:str):
    tok = clip.tokenize([txt]).to(DEVICE)
    with torch.no_grad():
        f = model.encode_text(tok); f /= f.norm(dim=-1, keepdim=True)
    return f.cpu().numpy().astype('float32').flatten()

def _imgs(p:Path):
    return [x for x in p.iterdir() if x.suffix.lower() in SUPPORTED_EXT]

def _load_paths(d:Path):
    try: return pickle.load(open(d/'paths.pkl','rb'))
    except: return []

def create_or_update_index(folder:Path):
    d = folder/'.clip_index'; d.mkdir(exist_ok=True)
    idx = faiss.read_index(str(d/'index.faiss')) if (d/'index.faiss').exists() else None
    prev = _load_paths(d)
    new = [str(p) for p in _imgs(folder) if str(p) not in prev]
    embs=[]
    for i in range(0,len(new),BATCH_SIZE):
        bt = torch.cat([preprocess(Image.open(p)).unsqueeze(0) for p in new[i:i+BATCH_SIZE]]).to(DEVICE)
        embs.append(_encode(bt))
    arr = np.vstack(embs) if embs else np.empty((0,512),dtype='float32')
    if idx is None: idx = faiss.IndexFlatIP(arr.shape[1])
    if arr.size: idx.add(arr); prev.extend(new)
    faiss.write_index(idx,str(d/'index.faiss'))
    pickle.dump(prev,open(d/'paths.pkl','wb'))
    return idx, prev

def load_index(folder:Path):
    d = folder/'.clip_index'
    if not (d/'index.faiss').exists(): return None,[]
    return faiss.read_index(str(d/'index.faiss')), _load_paths(d)

def _safe(p:str):
    if '..' in p: raise ValueError('Parent refs not allowed')
    q = Path(p).expanduser().resolve()
    if not q.is_dir(): raise ValueError('Not a directory')
    return q

FRONT_HTML=r"""<!DOCTYPE html><html lang='en'><head>
<meta charset='UTF-8'><meta name='viewport' content='width=device-width,initial-scale=1.0'>
<title>CLIP Image Search</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;min-height:100vh;display:flex;flex-direction:column;background:#000;background-size:cover;background-position:center;transition:background-image .6s ease-in-out}
body::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:1;background:linear-gradient(to bottom,rgba(0,0,0,.35) 0%,rgba(0,0,0,.5) 50%,rgb a(0,0,0,.65) 100%);backdrop-filter:blur(4px);}
.container{position:relative;z-index:2;width:100%;max-width:1800px;margin:auto;padding:2rem;flex:1}
h1{font-size:2rem;font-weight:300;margin-bottom:2rem;letter-spacing:-.02em}
.control-panel,.search-panel{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.15);border-radius:16px;padding:1.5rem;margin-bottom:2rem;backdrop-filter:blur(3px) saturate(160%)}
.folder-select,.search-box{display:flex;gap:1rem;margin-bottom:1rem}
input[type=text]{flex:1;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.25);padding:.75rem 1rem;border-radius:12px;color:#e0e0e0;font-size:.95rem;transition:all .2s}
input[type=text]:focus{outline:none;border-color:#fff;background:rgba(255,255,255,.18)}
button{background:rgba(0,0,0,.45);border:1px solid rgba(255,255,255,.3);color:#eee;padding:.75rem 1.5rem;border-radius:12px;font-size:.95rem;cursor:pointer;transition:all .2s}
button:hover{background:rgba(0,0,0,.6);border-color:#fff}
button:active{transform:translateY(1px)}
.status{font-size:.875rem;margin-top:.5rem;color:#888}
.status.success{color:#4ade80}.status.error{color:#f87171}
.results-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:1.5rem}
.result-item{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);border-radius:12px;overflow:hidden;cursor:pointer;transition:transform .3s,border .3s}
.result-item:hover{transform:translateY(-2px);border-color:#fff}
.result-item.expanded{grid-column:span 2}
.thumbnail{width:100%;height:170px;object-fit:cover}
.result-item.expanded .thumbnail{height:auto;max-height:520px}
.result-info{padding:.75rem;font-size:.875rem}
.filename{word-break:break-all;color:#eee;margin-bottom:.25rem}
.similarity{color:#bbb;font-size:.8rem}
.loading{text-align:center;padding:2rem;color:#aaa}
.spinner{display:inline-block;width:22px;height:22px;border:3px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
</style></head><body>
<div class='container'>
<h1>Natural Language Image Search</h1>
<div class='control-panel'><div class='folder-select'><input type='text' id='folder' placeholder='Enter folder path...'><button id='index'>Index</button></div><div id='idxStatus' class='status'></div></div>
<div class='search-panel'><div class='search-box'><input type='text' id='query' placeholder="Describe what you're looking for..."><button id='go'>Search</button></div></div>
<div id='grid' class='results-grid'></div></div>
<script>
const g=q=>document.querySelector(q), folder=g('#folder'), indexBtn=g('#index'), stat=g('#idxStatus'), query=g('#query'), go=g('#go'), grid=g('#grid');
async function post(u,d){return fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)}).then(r=>r.json())}
indexBtn.onclick=async()=>{if(!folder.value.trim())return;stat.textContent='Indexing...';stat.className='status';post('/index',{folder:folder.value.trim()}).then(r=>{stat.textContent=r.started?'Indexing started':'Error';if(r.error)stat.className='status error'})};
go.onclick=async()=>{if(!query.value.trim()||!folder.value.trim())return;grid.innerHTML=`<div class='loading'><div class='spinner'></div> Searching...</div>`;const d=await post('/search',{folder:folder.value.trim(),query:query.value.trim()});if(d.results)render(d.results);else grid.textContent=d.error||'No results'};
function render(arr){grid.innerHTML='';arr.forEach(o=>{const div=document.createElement('div');div.className='result-item';div.innerHTML=`<img class='thumbnail' src="data:image/jpeg;base64,${o.thumbnail}"><div class='result-info'><div class='filename'>${o.filename}</div><div class='similarity'>${(o.similarity*100).toFixed(1)}%</div></div>`;div.onclick=()=>{document.body.style.backgroundImage=`url(data:image/jpeg;base64,${o.thumbnail})`;div.classList.toggle('expanded');};grid.appendChild(div);})}
query.addEventListener('keypress',e=>e.key==='Enter'&&go.click());folder.addEventListener('keypress',e=>e.key==='Enter'&&indexBtn.click());</script></body></html>"""

@app.route('/')
def home():
    return render_template_string(FRONT_HTML)

@app.route('/check_index', methods=['POST'])
def check_index():
    try: fol=_safe(request.json['folder'])
    except Exception as e: return jsonify({'error':str(e)}),400
    idx,_=load_index(fol)
    return jsonify({'indexed':idx is not None})

@app.route('/index', methods=['POST'])
def index_folder():
    try: fol=_safe(request.json['folder'])
    except Exception as e: return jsonify({'error':str(e)}),400
    threading.Thread(target=lambda:create_or_update_index(fol),daemon=True).start()
    return jsonify({'started':True})

@app.route('/search', methods=['POST'])
def search():
    d=request.json
    try: fol=_safe(d['folder'])
    except Exception as e: return jsonify({'error':str(e)}),400
    q=d.get('query','').strip();
    if not q: return jsonify({'error':'Query empty'}),400
    idx,paths=load_index(fol)
    if idx is None: return jsonify({'error':'Folder not indexed'}),400
    k=min(10,len(paths)); sims,ids=idx.search(embed_text(q).reshape(1,-1),k)
    res=[]
    for i,sim in zip(ids[0],sims[0]):
        if i>=len(paths): continue
        p=paths[i]; img=Image.open(p); img.thumbnail(THUMBNAIL_SIZE,Image.Resampling.LANCZOS)
        buf=BytesIO(); img.save(buf,'JPEG',quality=85)
        res.append({'path':p,'filename':os.path.basename(p),'similarity':float(sim),'thumbnail':base64.b64encode(buf.getvalue()).decode()})
    return jsonify({'results':res})

if __name__=='__main__':
    init_clip(); app.run(host='0.0.0.0',port=5000,debug=True)
