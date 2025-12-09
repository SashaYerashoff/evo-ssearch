from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast
from urllib.error import HTTPError

import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F

try:  # torchvision>=0.15 preferred for v2 API
    from torchvision.transforms import v2 as transforms_v2  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover - fallback for older torchvision
    transforms_v2 = None  # type: ignore
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from transformers import AutoImageProcessor, AutoModel


ImageInput = Union[str, Path, Image.Image, np.ndarray]
TransformType = Callable[[Image.Image], torch.Tensor]
MaskInput = Union[np.ndarray, Image.Image]


@dataclass(frozen=True)
class ModelMetadata:
    name: str
    embed_dim: int
    weights_url: Optional[str] = None


class DINOEncoder:
    """Wrapper around the official DINOv3 models for embedding extraction."""

    _MODEL_ALIASES = {
        "dinov3_vitb": "dinov3_vitb16",
        "dinov3_vits": "dinov3_vits16",
        "dinov3_vitl": "dinov3_vitl16",
        "vitb": "dinov3_vitb16",
        "vits": "dinov3_vits16",
        "vitl": "dinov3_vitl16",
    }

    def __init__(
        self,
        model_name: str = "dinov3_vitb16",
        device: Optional[str] = None,
        batch_size: int = 16,
        weights_path: Optional[str] = None,
    ) -> None:
        self.model_name = self._resolve_model_name(model_name)
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "")
        self.device = torch.device(resolved_device)
        if self.device.type != "cuda":
            raise RuntimeError(
                "DINOEncoder requires a CUDA device. Set EVOSSEARCH_DINO_DEVICE to an available cuda device."
            )
        self.batch_size = max(1, int(batch_size))
        self.weights_path = self._validate_weights_path(weights_path)
        if self.weights_path is None:
            raise RuntimeError(
                "EVOSSEARCH_DINO_WEIGHTS_PATH must point to a local DINO checkpoint to avoid CPU fallback."
            )

        self.transform: Optional[TransformType] = None
        self.model: Optional[torch.nn.Module] = None
        self.processor: Optional[AutoImageProcessor] = None
        self.hf_model: Optional[torch.nn.Module] = None
        self.crop_size = 224

        self._register_token_count: int = 0

        if self._try_init_huggingface():
            return

        self._init_torch_hub()

    model: Optional[torch.nn.Module]
    transform: Optional[TransformType]
    processor: Optional[AutoImageProcessor]
    hf_model: Optional[torch.nn.Module]

    @staticmethod
    def _validate_weights_path(weights: Optional[str]) -> Optional[Path]:
        if not weights:
            return None
        candidate = Path(weights).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"DINO weights file not found: {candidate}")
        return candidate

    @staticmethod
    def _resolve_model_name(model_name: str) -> str:
        key = model_name.strip().lower()
        base = DINOEncoder._MODEL_ALIASES.get(key, model_name)
        if not base.startswith("dinov3"):
            base = f"dinov3_{base}" if not base.startswith("dinov3_") else base
        return base

    @staticmethod
    def _build_transform(model_name: str) -> "TransformType":
        """Create preprocessing pipeline matching DINOv3 eval transforms."""
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        if "sat493m" in model_name.lower():
            mean = (0.430, 0.411, 0.296)
            std = (0.213, 0.156, 0.143)

        resize_size = 256
        crop_size = 224

        if transforms_v2 is not None:
            transform = transforms_v2.Compose(
                [
                    transforms_v2.ToImage(),
                    transforms_v2.Resize(resize_size, antialias=True),
                    transforms_v2.CenterCrop(crop_size),
                    transforms_v2.ToDtype(torch.float32, scale=True),
                    transforms_v2.Normalize(mean=mean, std=std),
                ]
            )
            return cast("TransformType", transform)

        transform = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        return cast("TransformType", transform)

    def _try_init_huggingface(self) -> bool:
        if self.weights_path is None:
            return False

        repo_dir = self.weights_path if self.weights_path.is_dir() else self.weights_path.parent
        config_path = repo_dir / "config.json"
        if not config_path.exists():
            return False

        self.processor = AutoImageProcessor.from_pretrained(str(repo_dir))
        self.hf_model = cast(torch.nn.Module, AutoModel.from_pretrained(str(repo_dir)))
        self.hf_model.to(self.device)
        self.hf_model.eval()

        embed_dim = getattr(self.hf_model.config, "hidden_size", None)
        if embed_dim is None:
            raise ValueError("Unable to determine embedding dimension from Hugging Face model configuration.")
        self.metadata = ModelMetadata(name=self.model_name, embed_dim=int(embed_dim))
        self._register_token_count = int(getattr(self.hf_model.config, "num_register_tokens", 0) or 0)
        return True

    def _init_torch_hub(self) -> None:
        load_kwargs = {
            'pretrained': False,
            'trust_repo': True,
            'map_location': self.device,
        }

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=FutureWarning)
                self.model = cast(
                    torch.nn.Module,
                    torch.hub.load(
                        "facebookresearch/dinov3",
                        self.model_name,
                        **load_kwargs,
                    ),
                )
        except HTTPError as exc:
            if getattr(exc, 'code', None) == 403:
                raise RuntimeError(
                    "DINO architecture download blocked (403). Ensure the Dinov3 repo is available offline."
                ) from exc
            raise
        except ModuleNotFoundError as exc:
            missing = (getattr(exc, 'name', '') or str(exc) or '').lower()
            install_hints = {
                'torchmetrics': "DINO models require the torchmetrics package. Install it via 'pip install torchmetrics'.",
                'termcolor': "DINO models require the termcolor package. Install it via 'pip install termcolor'.",
            }
            for key, hint in install_hints.items():
                if key in missing:
                    raise ModuleNotFoundError(hint) from exc
            raise

        assert self.model is not None

        state_dict = torch.load(str(self.weights_path), map_location=self.device)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Loaded DINO weights mismatch. Missing keys: {missing}; unexpected keys: {unexpected}"
            )
        del state_dict

        self.model.to(self.device)
        self.model.eval()
        first_param = next(self.model.parameters(), None)
        if first_param is None or first_param.device.type != "cuda":
            raise RuntimeError("DINO model failed to remain on CUDA device; aborting to protect session.")

        embed_dim = getattr(self.model, "embed_dim", None) or getattr(self.model, "num_features", None)
        if embed_dim is None:  # pragma: no cover - defensive check
            raise ValueError(f"Unable to determine embedding dimension for {self.model_name}")
        self.metadata = ModelMetadata(name=self.model_name, embed_dim=int(embed_dim))
        self.transform = self._build_transform(self.model_name)
        register_tokens = getattr(self.model, "num_register_tokens", None)
        if register_tokens is None:
            tokens_param = getattr(self.model, "register_tokens", None)
            if tokens_param is not None and hasattr(tokens_param, "shape"):
                register_tokens = tokens_param.shape[-2] if tokens_param.ndim >= 2 else tokens_param.shape[0]
        register_tokens = self._coerce_register_token_count(register_tokens)
        if not register_tokens:
            register_tokens = getattr(self.model, "n_storage_tokens", None)
        self._register_token_count = self._coerce_register_token_count(register_tokens)

    def _encode_images_hf(self, batch: List[ImageInput]) -> np.ndarray:
        assert self.processor is not None and self.hf_model is not None

        embeddings: List[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(batch), self.batch_size):
                chunk = batch[start : start + self.batch_size]
                pil_images = [self._to_pil(item) for item in chunk]
                inputs = self.processor(images=pil_images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.hf_model(**inputs)
                pooled = getattr(outputs, "pooler_output", None)
                if pooled is None:
                    pooled = outputs.last_hidden_state[:, 0]
                pooled = F.normalize(pooled, dim=-1)
                embeddings.append(pooled.detach().cpu())

        result = torch.cat(embeddings, dim=0).numpy().astype(np.float32, copy=False)
        return result

    def encode_images(self, images: Union[ImageInput, Sequence[ImageInput]]) -> np.ndarray:
        """Encode a single image or a sequence into L2-normalized DINO embeddings."""
        batch = self._iter_inputs(images)
        if not batch:
            return np.empty((0, self.metadata.embed_dim), dtype=np.float32)

        if self.hf_model is not None and self.processor is not None:
            return self._encode_images_hf(batch)

        assert self.model is not None and self.transform is not None

        tensors: List[torch.Tensor] = []
        for item in batch:
            pil = self._to_pil(item)
            tensor = self.transform(pil)
            tensors.append(tensor)

        stack = torch.stack(tensors, dim=0).to(self.device)
        embeddings: List[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, stack.size(0), self.batch_size):
                chunk = stack[start : start + self.batch_size]
                feats = self.model(chunk)
                if isinstance(feats, dict):  # pragma: no cover - defensive
                    token = feats.get("x_norm_clstoken")
                    if token is None:
                        token = feats.get("x_norm_cls")
                    if token is None:
                        raise RuntimeError("Unexpected DINO forward output structure")
                    feats = token
                feats = F.normalize(feats, dim=-1)
                embeddings.append(feats.detach().cpu())
        result = torch.cat(embeddings, dim=0).numpy().astype(np.float32, copy=False)
        return result

    def encode_masked(
        self,
        image: ImageInput,
        mask: MaskInput,
        segment_ids: Optional[Sequence[Union[int, str]]] = None,
        min_patches: int = 1,
    ) -> Dict[str, Dict[str, Union[np.ndarray, float, int]]]:
        """Compute embeddings for one or more masked segments using patch tokens."""

        pil_image = self._to_pil(image)
        mask_array = self._prepare_mask_array(mask, pil_image.size)

        if self.hf_model is not None and self.processor is not None:
            return self._encode_masked_hf(pil_image, mask_array, segment_ids, min_patches)

        assert self.model is not None and self.transform is not None

        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            forward_features = getattr(self.model, "forward_features", None)
            if callable(forward_features):
                outputs = forward_features(tensor)
            else:
                outputs = self.model(tensor)

        if not isinstance(outputs, dict):  # pragma: no cover - defensive
            raise RuntimeError("DINO torch hub model did not return patch tokens")

        cls_token = outputs.get("x_norm_clstoken")
        if cls_token is None:
            cls_token = outputs.get("x_norm_cls")
        patch_tokens = outputs.get("x_norm_patchtokens")
        if patch_tokens is None:
            patch_tokens = outputs.get("x_norm_patch")
        if cls_token is None or patch_tokens is None:
            raise RuntimeError("DINO model outputs do not expose normalized patch tokens")

        cls_token = F.normalize(cls_token, dim=-1).detach().cpu()
        patch_tokens = patch_tokens.detach().cpu()

        return self._aggregate_masked_embeddings(
            cls_token.squeeze(0),
            patch_tokens.squeeze(0),
            mask_array,
            segment_ids,
            min_patches,
            self._register_token_count,
        )

    def _encode_masked_hf(
        self,
        image: Image.Image,
        mask_array: np.ndarray,
        segment_ids: Optional[Sequence[Union[int, str]]],
        min_patches: int,
    ) -> Dict[str, Dict[str, Union[np.ndarray, float, int]]]:
        assert self.processor is not None and self.hf_model is not None

        with torch.inference_mode():
            inputs = self.processor(images=[image], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.hf_model(**inputs)
            hidden = outputs.last_hidden_state.detach().cpu()
            cls_token = hidden[:, 0]
            patch_tokens = hidden[:, 1:]
            cls_token = F.normalize(cls_token, dim=-1)
            patch_tokens = F.normalize(patch_tokens, dim=-1)

        register_tokens = int(getattr(self.hf_model.config, "num_register_tokens", 0) or 0)

        return self._aggregate_masked_embeddings(
            cls_token.squeeze(0),
            patch_tokens.squeeze(0),
            mask_array,
            segment_ids,
            min_patches,
            register_tokens,
        )

    @staticmethod
    def _prepare_mask_array(mask: MaskInput, image_size: Tuple[int, int]) -> np.ndarray:
        if isinstance(mask, Image.Image):
            mask_img = mask
        else:
            mask_array = np.asarray(mask)
            if mask_array.ndim == 3:
                mask_array = mask_array[..., 0]
            mask_img = Image.fromarray(mask_array.astype(np.int32))
        if mask_img.size != image_size:
            mask_img = mask_img.resize(image_size, resample=Image.NEAREST)
        mask_data = np.asarray(mask_img)
        if mask_data.ndim != 2:
            raise ValueError("Mask must be a single-channel array")
        return mask_data.astype(np.int64)

    def _aggregate_masked_embeddings(
        self,
        cls_token: torch.Tensor,
        patch_tokens: torch.Tensor,
        mask_array: np.ndarray,
        segment_ids: Optional[Sequence[Union[int, str]]],
        min_patches: int,
        register_tokens: int,
    ) -> Dict[str, Dict[str, Union[np.ndarray, float, int]]]:
        if patch_tokens.ndim != 2:
            raise ValueError("Patch tokens must have shape (num_patches, embed_dim)")

        num_patches = patch_tokens.size(0)
        embed_dim = patch_tokens.size(1)
        grid_size = int(math.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            candidates = []
            if register_tokens and num_patches > register_tokens:
                if num_patches - register_tokens > 0:
                    candidates.append(patch_tokens[register_tokens:])  # drop leading tokens
                    candidates.append(patch_tokens[:-register_tokens])  # drop trailing tokens
            candidates.append(patch_tokens)
            chosen = None
            for candidate in candidates:
                n = candidate.size(0)
                root = int(math.sqrt(n))
                if root * root == n:
                    patch_tokens = candidate
                    num_patches = n
                    grid_size = root
                    chosen = candidate
                    break
            if chosen is None:
                raise ValueError("Unexpected patch token count; unable to form square grid for masking")

        normalized_tokens, grid_size = self._square_patch_tokens(patch_tokens, register_tokens)

        mask_tensor = torch.from_numpy(mask_array.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        mask_resized = F.interpolate(mask_tensor, size=(grid_size, grid_size), mode="nearest")
        mask_vector = mask_resized.reshape(-1).to(dtype=torch.int64)

        unique_segments = torch.unique(mask_vector)
        if segment_ids is not None:
            requested = set(str(seg) for seg in segment_ids)
            unique_segments = torch.tensor(
                [seg for seg in unique_segments.tolist() if str(seg) in requested],
                dtype=torch.int64,
            )
        result: Dict[str, Dict[str, Union[np.ndarray, float, int]]] = {}

        patch_np = normalized_tokens.numpy()
        num_patches = patch_np.shape[0]

        mask_flat_np = mask_vector.numpy()
        total_patches = num_patches

        for seg_value in unique_segments.tolist():
            if seg_value <= 0:
                continue
            mask_idx = mask_flat_np == seg_value
            patch_count = int(mask_idx.sum())
            if patch_count < max(1, min_patches):
                continue
            weights = mask_idx.astype(np.float32)
            weighted = (patch_np * weights[:, None]).sum(axis=0)
            if np.linalg.norm(weighted) == 0:
                continue
            embedding = weighted / np.linalg.norm(weighted)
            seg_key = str(seg_value)
            result[seg_key] = {
                "embedding": embedding.astype(np.float32, copy=False),
                "patch_count": patch_count,
                "patch_fraction": float(patch_count) / float(total_patches),
            }

        if "full" not in result:
            full_vec = cls_token.numpy()
            if np.linalg.norm(full_vec) > 0:
                result["full"] = {
                    "embedding": (full_vec / np.linalg.norm(full_vec)).astype(np.float32, copy=False),
                    "patch_count": num_patches,
                    "patch_fraction": 1.0,
                }

        return result

    @staticmethod
    def _square_patch_tokens(patch_tokens: torch.Tensor, register_tokens: int) -> Tuple[torch.Tensor, int]:
        if patch_tokens.ndim != 2:
            raise ValueError("Patch tokens must be 2D")

        candidates: List[torch.Tensor] = [patch_tokens]
        if register_tokens:
            if patch_tokens.size(0) > register_tokens:
                candidates.append(patch_tokens[register_tokens:])
                candidates.append(patch_tokens[:-register_tokens])
            if patch_tokens.size(0) > 2 * register_tokens:
                candidates.append(patch_tokens[register_tokens:-register_tokens])

        for candidate in candidates:
            count = candidate.size(0)
            grid = int(math.isqrt(count))
            if grid * grid == count:
                normalized = F.normalize(candidate, dim=-1)
                return normalized, grid

        raise ValueError("Unexpected patch token count; unable to form square grid for masking")

    def _forward_patch_features(
        self, image: ImageInput
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        pil_image = self._to_pil(image)

        if self.hf_model is not None and self.processor is not None:
            with torch.inference_mode():
                inputs = self.processor(images=[pil_image], return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.hf_model(**inputs)
                hidden = outputs.last_hidden_state.detach().cpu()
                cls_token = F.normalize(hidden[:, 0], dim=-1)
                patch_tokens = hidden[:, 1:].detach().cpu()
                register_tokens = int(getattr(self.hf_model.config, "num_register_tokens", 0) or 0)
        else:
            assert self.model is not None and self.transform is not None
            tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                forward_features = getattr(self.model, "forward_features", None)
                if callable(forward_features):
                    outputs = forward_features(tensor)
                else:
                    outputs = self.model(tensor)
            if not isinstance(outputs, dict):  # pragma: no cover - defensive
                raise RuntimeError("DINO torch hub model did not return patch tokens")
            cls_token = outputs.get("x_norm_clstoken")
            if cls_token is None:
                cls_token = outputs.get("x_norm_cls")
            patch_tokens = outputs.get("x_norm_patchtokens")
            if patch_tokens is None:
                patch_tokens = outputs.get("x_norm_patch")
            if cls_token is None or patch_tokens is None:
                raise RuntimeError("DINO model outputs do not expose normalized patch tokens")
            cls_token = F.normalize(cls_token, dim=-1).detach().cpu()
            patch_tokens = patch_tokens.detach().cpu()
            register_tokens = self._register_token_count

        normalized_patches, grid = self._square_patch_tokens(patch_tokens.squeeze(0), register_tokens)
        return cls_token.squeeze(0), normalized_patches, grid

    def patch_similarity_map(
        self,
        image: ImageInput,
        x_norm: float,
        y_norm: float,
    ) -> Tuple[np.ndarray, np.ndarray, int, Tuple[int, int]]:
        cls_token, patch_tokens, grid = self._forward_patch_features(image)
        x_norm = float(np.clip(x_norm, 0.0, 0.999999))
        y_norm = float(np.clip(y_norm, 0.0, 0.999999))
        px = int(round(x_norm * (grid - 1)))
        py = int(round(y_norm * (grid - 1)))
        px = max(0, min(grid - 1, px))
        py = max(0, min(grid - 1, py))
        index = py * grid + px
        tokens_np = patch_tokens.numpy()
        anchor = tokens_np[index]
        sims = tokens_np @ anchor
        heatmap = sims.reshape(grid, grid).astype(np.float32)
        return heatmap, tokens_np, grid, (px, py)

    @staticmethod
    def _iter_inputs(images: Union[ImageInput, Sequence[ImageInput]]) -> List[ImageInput]:
        if isinstance(images, Sequence) and not isinstance(images, (str, bytes, bytearray)):
            return list(images)
        return [cast(ImageInput, images)]

    @staticmethod
    def _to_pil(image: ImageInput) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.ndim == 3 and arr.shape[0] in {1, 3} and arr.shape[0] != arr.shape[-1]:
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 1)
                arr = (arr * 255).astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")
        raise TypeError(f"Unsupported image input type: {type(image)!r}")
    @staticmethod
    def _coerce_register_token_count(value: Optional[Union[int, float, torch.Tensor]]) -> int:
        if value is None:
            return 0
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return int(value.item())
            return int(value.view(-1)[0].item())
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
