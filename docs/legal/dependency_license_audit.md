# Dependency & Model License Audit → PM Task

Purpose: surface the licensing obligations of everything EVA AI ships or depends
on, and turn the gaps into a concrete list for the PM/legal to resolve before a
commercial government deployment / GA.

**Disclaimer:** license identifications below are **indicative**, from common
knowledge of these projects. They **must be confirmed** against the exact pinned
versions and the exact model checkpoints in use. This document is the starting
checklist, not legal advice. `[NEEDS LEGAL]` / `[VERIFY]` mark open items.

## 1. Python dependencies (`requirements.txt`, `requirements-db.txt`)

| Package | License (indicative) | Note |
|---|---|---|
| flask, flask-cors | BSD-3 / MIT | permissive |
| setuptools | MIT | pinned `<81` for CLIP's `pkg_resources` |
| torch, torchvision | BSD-3 (+ bundled components) | `[VERIFY]` bundled CUDA/cuDNN/3rd-party notices |
| torchmetrics, timm, transformers, ftfy | Apache-2.0 | permissive |
| numpy, python-dotenv, psutil | BSD-3 | permissive |
| pillow | HPND (MIT-like) | permissive |
| **opencv-python-headless** | Apache-2.0 (OpenCV) | **`[VERIFY]` codec/FFmpeg linkage** (see §3) |
| **faiss-cpu** | MIT | permissive |
| **clip-anytorch** / OpenAI CLIP code | MIT | code MIT; **weights separate** (see §2) |
| einops, regex, tqdm | MIT / Apache / **MPL-2.0 (tqdm)** | tqdm is weak-copyleft (file-level); fine to use, include notice |
| requests | Apache-2.0 | permissive |
| gunicorn | MIT | permissive |
| **psycopg (v3)** | **LGPL-3.0** | weak copyleft; on-prem/dynamic use generally OK, **must be disclosed**; do not statically bundle without review `[NEEDS LEGAL]` |
| SQLAlchemy, alembic | MIT | permissive |

**Action:** generate a `THIRD_PARTY_NOTICES` / NOTICE file from the resolved
lockfile; flag LGPL (psycopg) and MPL (tqdm) explicitly.

## 2. Model weights — the real legal risk

Code licenses above do **not** cover the model weights. These are the items most
likely to constrain commercial/government use:

| Model | Used for | Concern → `[NEEDS LEGAL]` |
|---|---|---|
| **Qwen3-VL-4B** (VLM) | video-descriptions | Confirm the **exact Qwen license** for the checkpoint (Apache-2.0 vs Tongyi Qianwen license); check **acceptable-use** clauses and any restrictions on government/surveillance use. **Primary item.** |
| **Qwen3.5-9B** (agent) | the agent | Same as above for the agent checkpoint. |
| **CLIP ViT-B/32** | embedding/search | OpenAI CLIP weights are MIT (indicative) — confirm. Generally permissive. |
| **DINOv3** (`dinov3_vith16plus`) | experimental, **disabled in prod** | DINOv3 has its **own license with potential commercial restrictions** — ensure it is **excluded from the production path**, or clear the license before enabling. |
| **Mask2Former** (`facebook/mask2former-swin-base-ade-semantic`) | experimental | Code permissive, but **ADE20K training data has research-use terms** and Swin backbone has its own license — ensure off in prod or cleared. |
| **SigLIP2** (optional CLIP alt) | optional embedder | Google model license — confirm if ever used. |

**Action:** for each model actually in the production path (Qwen VLM, Qwen agent,
CLIP), attach the model card + license + acceptable-use, and a written statement
that government public-space monitoring is a permitted use. Confirm DINOv3 /
Mask2Former / SigLIP2 are **not** in the production deployment.

## 3. Media / codecs

- If any video decode path uses **FFmpeg** (directly or via OpenCV builds),
  FFmpeg components are **LGPL or GPL** depending on build flags, with patent
  considerations for some codecs. `[VERIFY]` how frames are decoded (EVA mostly
  consumes JPEG snapshots from Luxriot, which limits exposure) and which OpenCV
  build is shipped.

## 4. Third-party integration

- **Luxriot Evo** — commercial third-party product. The EVA↔Luxriot integration
  (API use, redistribution, co-marketing) needs a **commercial/partnership
  agreement and API license** confirmation. `[NEEDS LEGAL]` / commercial.

## 5. PM task list (what to gather / decide)

1. Confirm the **Qwen VLM and agent checkpoint licenses** + acceptable-use, and
   obtain a written OK for commercial **government surveillance** use. *(highest)*
2. Confirm **CLIP weights** license; confirm **DINOv3/Mask2Former/SigLIP2 are
   excluded** from prod (or obtain clearance).
3. Decide **psycopg (LGPL)** disclosure approach; confirm no static bundling issue.
4. Verify the **OpenCV/FFmpeg** build and codec/patent exposure for the deployment.
5. Secure the **Luxriot commercial/integration agreement** terms.
6. Produce ship artifacts: `THIRD_PARTY_NOTICES`/NOTICE file from the lockfile,
   model notices/cards, customer **EULA**, and tie-in to the DPIA / EU AI Act work
   (see [data_retention_privacy](../architecture/data_retention_privacy.md)).
7. Decide a **dependency lockfile + offline wheelhouse** so the shipped license set
   is deterministic (also a reproducibility requirement — `[VERIFY]`).

Owner: PM/legal, with engineering providing the resolved dependency lockfile and
the list of model checkpoints actually deployed.
