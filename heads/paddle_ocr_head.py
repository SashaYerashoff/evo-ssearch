from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

try:  # PaddleOCR relies on PaddlePaddle; keep import localized
    from paddleocr import PaddleOCR  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    PaddleOCR = None  # type: ignore


class PaddleOCRHead:
    """Thin wrapper over PaddleOCR for text recognition."""

    def __init__(
        self,
        lang: str = 'en',
        det: bool = False,
        rec: bool = True,
        use_angle_cls: bool = True,
    ) -> None:
        if PaddleOCR is None:  # pragma: no cover - safety
            raise RuntimeError("PaddleOCR package is not available")
        self.ocr = PaddleOCR(
            lang=lang,
            det=det,
            rec=rec,
            use_angle_cls=use_angle_cls,
            show_log=False,
        )

    def recognize(self, image: Image.Image) -> str:
        arr = np.array(image.convert('RGB'))
        result = self.ocr.ocr(arr, cls=True, det=False, rec=True)
        if not result:
            return ''
        # result is list per image
        lines: Sequence[Sequence[Sequence[float]]] = result[0]  # type: ignore
        best_text = ''
        best_score = -1.0
        for item in lines:
            if len(item) < 2:
                continue
            text, score = item[0], float(item[1])
            if score > best_score:
                best_text = str(text)
                best_score = score
        return best_text.strip()
