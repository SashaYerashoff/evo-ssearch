#!/usr/bin/env python3
"""Index pre-generated segmentation masks into the DINO segment FAISS index."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse server utilities
from config import config
from oldapp import (
    dino_encoder,
    ensure_embedder_loaded,
    load_segment_index,
    save_segment_index,
)


def _load_masks_manifest(path: Path) -> Dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Mask manifest not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse mask manifest {path}: {exc}") from exc


def _load_mask_image(mask_entry: dict) -> Image.Image:
    mask_png = mask_entry.get("mask_png")
    if mask_png:
        png_path = Path(mask_png)
        if png_path.exists():
            return Image.open(png_path).convert("L")
    mask_npy = mask_entry.get("mask_npy")
    if mask_npy:
        mask_array = np.load(mask_npy)
        if mask_array.ndim == 3:
            mask_array = mask_array[..., 0]
        img = Image.fromarray(mask_array.astype("uint8"), mode="L")
        return img
    raise FileNotFoundError("Neither mask_png nor mask_npy found for entry")


def _normalise_mask(mask_image: Image.Image) -> Image.Image:
    # Binarise mask to make segment label stable (0 background, 1 foreground)
    return mask_image.point(lambda v: 255 if v > 0 else 0, mode="L")


def _collect_existing_ids(folder: Path) -> set:
    _, segment_meta, _ = load_segment_index(folder)
    return {entry.get("segment_id") for entry in segment_meta if entry.get("segment_id")}


def index_masks(folder: Path, manifest: Path, overwrite: bool = False) -> Tuple[int, int]:
    if not config.DINO_SEGMENTS_ENABLED:
        raise RuntimeError(
            "Segment indexing is disabled. Enable EVOSSEARCH_DINO_SEGMENTS_ENABLED before running this tool."
        )

    data = _load_masks_manifest(manifest)

    ensure_embedder_loaded("dino")
    from oldapp import dino_encoder as active_encoder

    if active_encoder is None:
        raise RuntimeError("Failed to initialise DINO encoder")

    encoder = active_encoder

    embeddings: List[np.ndarray] = []
    meta: List[dict] = []

    existing = set()
    if not overwrite:
        existing = _collect_existing_ids(folder)

    processed_images = 0
    added_segments = 0

    for image_path, info in data.items():
        image_path = Path(image_path)
        if not image_path.exists():
            continue
        if folder not in image_path.parents:
            continue

        variants = info.get("variants", [])
        if not variants and info.get("mask_png"):
            variants = [
                {
                    "variant_id": info.get("selected_variant_id") or info.get("best_variant_id"),
                    "mask_png": info.get("mask_png"),
                    "mask_npy": info.get("mask_npy"),
                    "predicted_iou": info.get("score"),
                    "area": None,
                    "clip_score": info.get("clip_score"),
                }
            ]

        for variant in variants:
            variant_id = variant.get("variant_id")
            if variant_id is None:
                continue
            if not overwrite and variant_id in existing:
                continue

            try:
                mask_img = _normalise_mask(_load_mask_image(variant))
            except FileNotFoundError:
                continue

            segments = encoder.encode_masked(
                image_path,
                mask_img,
                segment_ids=None,
                min_patches=config.DINO_SEGMENT_MIN_PATCHES,
            )

            for seg_key, seg_info in segments.items():
                if seg_key == "full":
                    continue
                embedding = seg_info.get("embedding")
                if embedding is None:
                    continue
                embedding_np = np.asarray(embedding, dtype=np.float32)
                if embedding_np.ndim != 1:
                    continue

                embeddings.append(embedding_np)
                meta.append(
                    {
                        "image_path": str(image_path),
                        "segment_id": str(variant_id),
                        "label": info.get("label") or variant.get("label"),
                        "area": int(seg_info.get("patch_count", 0)),
                        "patch_fraction": float(seg_info.get("patch_fraction", 0.0)),
                        "predicted_iou": variant.get("predicted_iou"),
                        "clip_score": variant.get("clip_score"),
                        "source": variant.get("model") or info.get("model") or "sam",
                        "created_at": info.get("last_segmented_at"),
                    }
                )
                added_segments += 1
        processed_images += 1

    if not embeddings:
        return processed_images, 0

    embedding_matrix = np.stack(embeddings, axis=0)
    save_segment_index(folder, embedding_matrix, meta)
    return processed_images, added_segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Index SAM masks into DINO segment index")
    parser.add_argument("folder", type=Path, help="Image folder that was indexed")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to masks.json (defaults to <folder>/.clip_index/masks.json)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-index even if segment IDs already exist",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Allow GPU usage. By default CUDA devices are disabled to avoid OOM",
    )
    args = parser.parse_args()

    if not args.use_gpu:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    folder = args.folder.resolve()
    manifest = args.manifest or (folder / config.INDEX_FOLDER_NAME / "masks.json")

    try:
        processed, added = index_masks(folder, manifest, overwrite=args.overwrite)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Processed {processed} images; added {added} segment embeddings to index")


if __name__ == "__main__":
    main()
