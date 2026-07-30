from __future__ import annotations

from typing import Dict, Optional, Tuple, cast

import numpy as np
import torch
from PIL import Image
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

if hasattr(Image, "Resampling"):
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
else:  # pragma: no cover - Pillow compatibility fallback
    RESAMPLE_BILINEAR = Image.BILINEAR  # type: ignore[attr-defined]
    RESAMPLE_NEAREST = Image.NEAREST  # type: ignore[attr-defined]


class Mask2FormerHead:
    """Thin wrapper around Mask2Former for single-image region refinement."""

    def __init__(
        self,
        model_name: str,
        device: str,
        max_size: int = 1024,
        *,
        local_files_only: bool = True,
        cache_dir: str | None = None,
    ) -> None:
        target_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        if target_device.startswith("cuda") and not torch.cuda.is_available():
            target_device = "cpu"
        self.device = torch.device(target_device)
        self.model_name = model_name
        self.max_size = max(256, int(max_size)) if max_size else 0

        self.processor = Mask2FormerImageProcessor.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
        cast(torch.nn.Module, self.model).to(self.device)
        self.model.eval()
        self.use_autocast = self.device.type == "cuda"

        id2label: Dict[int, str] = {}
        try:
            cfg_map = getattr(self.model.config, "id2label", None)
            if isinstance(cfg_map, dict):
                id2label.update({int(k): str(v) for k, v in cfg_map.items()})
        except Exception:
            pass
        try:
            proc_map = getattr(self.processor, "id2label", None)
            if isinstance(proc_map, dict):
                id2label.update({int(k): str(v) for k, v in proc_map.items()})
        except Exception:
            pass
        self.id2label = id2label

    def _prepare_image(self, image: Image.Image) -> Tuple[Image.Image, Tuple[int, int]]:
        original_size = image.size
        if self.max_size and max(original_size) > self.max_size:
            scale = self.max_size / float(max(original_size))
            new_size = (
                max(32, int(round(original_size[0] * scale))),
                max(32, int(round(original_size[1] * scale))),
            )
            resized = image.resize(new_size, RESAMPLE_BILINEAR)
            return resized, original_size
        return image, original_size

    def _postprocess_segmentation(self, segmentation: torch.Tensor, target_size: Tuple[int, int]) -> np.ndarray:
        seg_np = segmentation.cpu().numpy().astype(np.int32)
        if segmentation.shape[-2:] != (target_size[1], target_size[0]):
            seg_img = Image.fromarray(seg_np.astype(np.uint8), mode="L")
            seg_img = seg_img.resize(target_size, resample=RESAMPLE_NEAREST)
            seg_np = np.asarray(seg_img, dtype=np.int32)
        return seg_np

    def refine(
        self,
        image: Image.Image,
        coarse_mask: np.ndarray,
        click_xy: Tuple[float, float],
    ) -> Optional[Dict[str, object]]:
        if image.mode != "RGB":
            pil_image = image.convert("RGB")
        else:
            pil_image = image

        prepared, original_size = self._prepare_image(pil_image)
        inputs = self.processor(images=prepared, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            if self.use_autocast:
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

        segmentation = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(prepared.height, prepared.width)]
        )[0]
        seg_np = self._postprocess_segmentation(segmentation, original_size)

        height, width = seg_np.shape
        if height == 0 or width == 0:
            return None

        cx = min(max(int(round(click_xy[0] * (width - 1))), 0), width - 1)
        cy = min(max(int(round(click_xy[1] * (height - 1))), 0), height - 1)
        class_id = int(seg_np[cy, cx])
        region = seg_np == class_id

        if coarse_mask is not None:
            try:
                coarse_array = np.asarray(coarse_mask)
                coarse_img = Image.fromarray(coarse_array.astype(np.uint8), mode="L")
                coarse_resized = coarse_img.resize((width, height), resample=RESAMPLE_NEAREST)
                coarse_bool = np.asarray(coarse_resized) > 0
                region = np.logical_and(region, coarse_bool)
                if not region.any():
                    region = coarse_bool
            except Exception:
                pass

        if not region.any():
            region = seg_np == class_id

        mask = np.zeros((height, width), dtype=np.uint8)
        mask[region] = 255

        mask_fraction = float(region.sum()) / float(region.size) if region.size else 0.0
        label = self.id2label.get(class_id, f"class_{class_id}")

        class_labels = {int(cid): self.id2label.get(int(cid), f"class_{int(cid)}") for cid in np.unique(seg_np)}

        return {
            "mask": mask,
            "class_id": class_id,
            "label": label,
            "mask_fraction": mask_fraction,
            "segment_value": 255,
            "segmentation": seg_np,
            "class_labels": class_labels,
        }
