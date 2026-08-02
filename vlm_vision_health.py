"""Content-aware health checks for OpenAI-compatible vision models.

HTTP liveness cannot detect a multimodal encoder that keeps returning stale or
corrupted visual features.  The helpers in this module generate a fresh,
deterministic control image for every check and validate facts which are not
present in the text prompt.
"""

from __future__ import annotations

import base64
import json
import random
import re
import struct
import time
import urllib.request
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


_COLORS: dict[str, tuple[int, int, int]] = {
    "RED": (220, 35, 45),
    "GREEN": (30, 170, 75),
    "BLUE": (35, 90, 220),
}

_DIGITS = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00010", "00010", "01110", "00010", "00010", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "11100"),
}


@dataclass(frozen=True)
class VisionCanaryResult:
    ok: bool
    expected: str
    observed: str
    latency_ms: float
    error: str = ""


def build_control_png(
    code: str,
    colors: Sequence[str],
    *,
    width: int = 640,
    height: int = 360,
) -> bytes:
    """Build a dependency-free RGB PNG with coloured blocks and four digits."""

    normalized_code = str(code).strip()
    normalized_colors = tuple(str(color).strip().upper() for color in colors)
    if len(normalized_code) != 4 or any(digit not in _DIGITS for digit in normalized_code):
        raise ValueError("control code must contain exactly four digits")
    if len(normalized_colors) != 3 or set(normalized_colors) != set(_COLORS):
        raise ValueError("control colors must be a permutation of RED, GREEN, BLUE")

    pixels = bytearray([255, 255, 255] * width * height)

    def rectangle(
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: tuple[int, int, int],
    ) -> None:
        for y in range(max(0, y0), min(height, y1)):
            row = y * width * 3
            for x in range(max(0, x0), min(width, x1)):
                offset = row + x * 3
                pixels[offset : offset + 3] = bytes(color)

    for index, color_name in enumerate(normalized_colors):
        x0 = 70 + index * 180
        rectangle(x0, 35, x0 + 140, 145, _COLORS[color_name])

    scale = 18
    digit_width = 5 * scale
    gap = 2 * scale
    total_width = len(normalized_code) * digit_width + (len(normalized_code) - 1) * gap
    cursor_x = (width - total_width) // 2
    for digit in normalized_code:
        for row_index, row in enumerate(_DIGITS[digit]):
            for column_index, enabled in enumerate(row):
                if enabled == "1":
                    rectangle(
                        cursor_x + column_index * scale,
                        190 + row_index * scale,
                        cursor_x + (column_index + 1) * scale,
                        190 + (row_index + 1) * scale,
                        (10, 10, 10),
                    )
        cursor_x += digit_width + gap

    raw = b"".join(
        b"\x00" + bytes(pixels[y * width * 3 : (y + 1) * width * 3])
        for y in range(height)
    )

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=9))
        + chunk(b"IEND", b"")
    )


def _normalized_answer(value: str) -> str:
    return " ".join(re.findall(r"[A-Z0-9]+", str(value).upper()))


def probe_vision(
    base_url: str,
    model: str,
    *,
    api_key: str = "",
    timeout_sec: float = 30.0,
    seed: int | None = None,
) -> VisionCanaryResult:
    """Send a fresh control frame and verify ordered visual facts."""

    rng = random.Random(seed if seed is not None else time.time_ns())
    # Avoid a leading zero: small VLMs occasionally omit it even when the
    # visual encoder is healthy, which would create a false recovery cycle.
    code = str(rng.randrange(1, 10)) + "".join(
        str(rng.randrange(10)) for _ in range(3)
    )
    colors = list(_COLORS)
    rng.shuffle(colors)
    facts = f"{code} {' '.join(colors)}"
    expected = f"VISION_OK {facts}"
    image = base64.b64encode(build_control_png(code, colors)).decode("ascii")
    payload = {
        "model": str(model).strip(),
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Read the four black digits and name the three coloured "
                            "blocks from left to right. Reply on one line exactly as "
                            "VISION_OK <digits> <COLOR> <COLOR> <COLOR>."
                        ),
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 64,
    }
    headers = {"Content-Type": "application/json"}
    if str(api_key).strip():
        headers["Authorization"] = f"Bearer {str(api_key).strip()}"
    request = urllib.request.Request(
        str(base_url).rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=max(1.0, float(timeout_sec))) as response:
            result = json.loads(response.read().decode("utf-8"))
        observed = str(result["choices"][0]["message"]["content"])
        normalized_tokens = _normalized_answer(observed).split()
        if normalized_tokens[:2] == ["VISION", "OK"]:
            normalized_tokens = normalized_tokens[2:]
        observed_code = normalized_tokens[0] if normalized_tokens else ""
        observed_colors = normalized_tokens[1:4]
        matching_digits = sum(
            left == right for left, right in zip(code, observed_code)
        ) if len(observed_code) == len(code) and observed_code.isdigit() else 0
        # OCR is not the workload under test. A healthy 4B may confuse one
        # block digit while still proving that it received a fresh image.
        # Strict random colour order plus >=3 matching nonce digits keeps the
        # probability of a stale prior canary passing below 0.1%.
        ok = observed_colors == colors and matching_digits >= 3
        return VisionCanaryResult(
            ok=ok,
            expected=expected,
            observed=observed[:500],
            latency_ms=round((time.monotonic() - started) * 1000.0, 3),
            error="" if ok else "visual_control_mismatch",
        )
    except Exception as exc:
        return VisionCanaryResult(
            ok=False,
            expected=expected,
            observed="",
            latency_ms=round((time.monotonic() - started) * 1000.0, 3),
            error=f"{type(exc).__name__}: {exc}"[:500],
        )


def read_health_state(path: str | Path) -> dict[str, Any]:
    state_path = Path(path).expanduser()
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"status": "missing", "ok": False}
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}"[:300],
        }
    if not isinstance(payload, Mapping):
        return {"status": "invalid", "ok": False}
    return dict(payload)


def write_health_state(path: str | Path, payload: Mapping[str, Any]) -> None:
    state_path = Path(path).expanduser()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_path.with_name(f".{state_path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o640)
    temporary.replace(state_path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "VisionCanaryResult",
    "build_control_png",
    "probe_vision",
    "read_health_state",
    "utc_now_iso",
    "write_health_state",
]
