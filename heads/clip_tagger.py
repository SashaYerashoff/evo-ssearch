from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image


DEFAULT_PROMPTS: Tuple[str, ...] = (
    # People & roles
    "person",
    "man",
    "woman",
    "child",
    "group of people",
    "portrait",
    "selfie",
    "smiling person",
    "profile view",
    "close-up face",
    # Accessories / apparel
    "wearing sunglasses",
    "wearing glasses",
    "wearing a hat",
    "wearing headphones",
    "wearing jewelry",
    "wearing a backpack",
    "wearing a dress",
    "wearing a suit",
    "wearing a jacket",
    "wearing a t-shirt",
    "wearing gloves",
    "wearing a scarf",
    # Actions / interactions
    "holding a drink",
    "holding a phone",
    "holding a book",
    "taking a photo",
    "playing an instrument",
    "sitting",
    "standing",
    "dancing",
    "running",
    "jumping",
    # Objects / props
    "cat",
    "dog",
    "pet",
    "plush toy",
    "camera",
    "microphone",
    "laptop",
    "bicycle",
    "car",
    "food",
    "dessert",
    "cup of coffee",
    "glass of wine",
    "musical instrument",
    # Scene context
    "outdoors",
    "indoors",
    "beach",
    "mountains",
    "forest",
    "city street",
    "night",
    "sunset",
    "party",
    "concert stage",
    "studio lighting",
    # Background elements
    "text",
    "sign",
    "logo",
    "graffiti",
    "plant",
    "flower",
    "tree",
    "building",
    "sky",
    "water",
    "snow",
    # Colors
    "bright colors",
    "pastel colors",
    "black clothing",
    "white clothing",
    "red clothing",
    "blue clothing",
    "green clothing",
    "yellow clothing",
)


class CLIPAutoTagger:
    """Zero-shot classifier for region crops based on CLIP prompts."""

    def __init__(
        self,
        clip_model: "torch.nn.Module",
        preprocess,
        device: torch.device,
        prompts: Sequence[str] = DEFAULT_PROMPTS,
    ) -> None:
        self.model = clip_model
        self.preprocess = preprocess
        self.device = device
        merged: List[str] = []
        seen = set()
        for item in list(prompts) + list(DEFAULT_PROMPTS):
            cleaned = (item or "").strip()
            key = cleaned.lower()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            merged.append(cleaned)
        self.prompts = tuple(merged) if merged else DEFAULT_PROMPTS
        if not self.prompts:
            raise ValueError("CLIPAutoTagger requires at least one text prompt")
        self._text_features = self._encode_prompts(self.prompts)

    def _encode_prompts(self, prompts: Iterable[str]) -> torch.Tensor:
        import clip  # delayed import to reuse loaded package

        with torch.no_grad():
            tokens = clip.tokenize(list(prompts)).to(self.device)
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def tag_image(
        self,
        image: Image.Image,
        top_k: int,
        threshold: float,
        extra_prompts: Optional[Sequence[str]] = None,
        extra_weight: float = 1.15,
    ) -> List[dict]:
        image_rgb = image.convert("RGB")
        tensor = self.preprocess(image_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        features = self._text_features
        labels = list(self.prompts)
        sources: List[str] = ["base"] * len(labels)
        if extra_prompts:
            cleaned = [p.strip() for p in extra_prompts if isinstance(p, str) and p.strip()]
            if cleaned:
                extra_feats = self._encode_prompts(cleaned)
                if extra_weight != 1.0:
                    extra_feats = extra_feats * extra_weight
                features = torch.cat([features, extra_feats], dim=0)
                labels.extend(cleaned)
                sources.extend(["extra"] * len(cleaned))

        logits = image_features @ features.T
        scores = logits.squeeze(0).float().cpu().numpy()

        paired = sorted(
            ((float(score), label, src) for score, label, src in zip(scores, labels, sources)),
            key=lambda item: item[0],
            reverse=True,
        )

        tags: List[dict] = []
        for score, label, src in paired[: max(1, top_k)]:
            if score < threshold:
                continue
            tags.append({
                "label": label,
                "score": round(score, 4),
                "source": src,
            })
        return tags
