#!/usr/bin/env python3
"""
Standalone harness for evaluating segmentation heads (Mask2Former) in isolation.

Usage
-----
python tools/head_harness.py --model facebook/mask2former-swin-base-ade-semantic --image path/to/sample.jpg

The script reports memory consumption, load latencies, and forward-pass timings
to help validate that the head can run within the available hardware envelope.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import psutil

try:
    import torch
except ImportError as exc:  # pragma: no cover - defensive guard
        raise SystemExit("PyTorch is required for the head harness. Install torch>=2.1.") from exc

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - defensive guard
        raise SystemExit("Pillow is required for image loading. Install pillow>=10.0.") from exc

try:
    from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
except ImportError as exc:  # pragma: no cover - defensive guard
        raise SystemExit("Transformers is required. Install transformers>=4.40.") from exc


def _bytes_to_mb(value: int) -> float:
    return round(value / (1024 * 1024), 2)


def _current_cpu_mem_mb() -> float:
    process = psutil.Process(os.getpid())
    return _bytes_to_mb(process.memory_info().rss)


def _current_gpu_mem_mb(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    try:
        torch.cuda.synchronize(device)
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        stats = torch.cuda.memory_stats(device)
        return _bytes_to_mb(stats.get("allocated_bytes.all.current", 0))
    except Exception:  # pragma: no cover - defensive
        return None


def load_model(model_name: str, device: torch.device) -> Dict[str, Any]:
    t0 = time.time()
    processor = Mask2FormerImageProcessor.from_pretrained(model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
    load_time = time.time() - t0

    model.eval()
    runtime_model = cast(torch.nn.Module, model)
    runtime_model.to(device)
    if device.type == "cuda":
        runtime_model = runtime_model.half()
    model = cast(Mask2FormerForUniversalSegmentation, runtime_model)

    return {
        "processor": processor,
        "model": model,
        "load_time": load_time,
    }


def run_inference(
    bundle: Dict[str, Any],
    image_paths: List[Path],
    device: torch.device,
    warmup: int,
) -> Dict[str, Any]:
    processor: Mask2FormerImageProcessor = bundle["processor"]
    model: Mask2FormerForUniversalSegmentation = bundle["model"]

    def load_images() -> List[Image.Image]:
        frames: List[Image.Image] = []
        for p in image_paths:
            with Image.open(p) as img:
                frames.append(img.convert("RGB"))
        return frames

    images = load_images()
    if not images:
        raise ValueError("No images provided for inference.")

    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    if device.type == "cuda":
        pixel_values = pixel_values.half()

    summary: Dict[str, Any] = {}

    with torch.inference_mode():
        for _ in range(max(0, warmup)):
            _ = model(pixel_values)

        t0 = time.time()
        outputs = model(pixel_values)
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        infer_time = time.time() - t0

    summary["forward_time_s"] = round(infer_time, 4)
    summary["batch_size"] = len(images)
    summary["class_logits_shape"] = list(outputs.class_queries_logits.shape)
    summary["mask_logits_shape"] = list(outputs.masks_queries_logits.shape)
    summary["num_queries"] = outputs.class_queries_logits.shape[1]
    return summary


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mask2Former head harness for evo-ssearch.")
    parser.add_argument("--model", default="facebook/mask2former-swin-base-ade-semantic", help="Hugging Face model id.")
    parser.add_argument("--image", action="append", type=Path, help="Path to an image. Can be repeated.")
    parser.add_argument("--device", default=None, help="Torch device to run on, e.g. cuda:0 or cpu. Defaults to CUDA if available.")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup passes before timed inference.")
    parser.add_argument("--summary", type=Path, default=None, help="Optional path to write JSON summary.")
    parser.add_argument("--skip-run", action="store_true", help="Load the head but skip the forward pass.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    device_str = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    cpu_before = _current_cpu_mem_mb()
    gpu_before = _current_gpu_mem_mb(device) if device.type == "cuda" else None

    bundle = load_model(args.model, device)

    cpu_after_load = _current_cpu_mem_mb()
    gpu_after_load = _current_gpu_mem_mb(device) if device.type == "cuda" else None

    summary = {
        "model": args.model,
        "device": device_str,
        "load_time_s": round(bundle["load_time"], 4),
        "cpu_mem_before_mb": cpu_before,
        "cpu_mem_after_mb": cpu_after_load,
        "gpu_mem_before_mb": gpu_before,
        "gpu_mem_after_mb": gpu_after_load,
    }

    if not args.skip_run:
        if not args.image:
            raise SystemExit("Provide at least one --image when running inference.")
        inference_summary = run_inference(bundle, args.image, device, args.warmup)
        summary.update(inference_summary)
        summary["cpu_mem_post_run_mb"] = _current_cpu_mem_mb()
        if device.type == "cuda":
            summary["gpu_mem_post_run_mb"] = _current_gpu_mem_mb(device)

    output = json.dumps(summary, indent=2)
    print(output)

    if args.summary:
        args.summary.write_text(output, encoding="utf-8")

    return 0


if __name__ == "__main__":
    sys.exit(main())
