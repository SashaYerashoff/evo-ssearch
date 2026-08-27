"""End-to-end prompt smoke: real decider -> real batch -> real qwen3-vl-4b.

Builds synthetic 'scene' frames (walking figure, then a fast blurred dash),
runs them through the real LuxriotCaptureSession decider, creates a real
summary batch (capture_attention + homeostasis + companion), renders the real
message payload and sends it to the local LM Studio qwen/qwen3-vl-4b.
No EVA services are touched.
"""

import base64
import json
import sys
import tempfile
import time
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import requests
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, "/home/sasha/Projects/evo-ssearch")

from luxriot_connector import LuxriotCaptureSession, LuxriotManager
import oldapp  # noqa: E402  (for the real message builder)

LM_URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "qwen/qwen3-vl-4b"


def jpeg_encoder(image, max_edge=640, quality=85):
    img = image
    if max(img.size) > max_edge:
        scale = max_edge / float(max(img.size))
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def scene_frame(figure_x, blur=0.0, size=(640, 360)):
    img = Image.new("RGB", size, (98, 105, 112))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 300, 640, 360], fill=(70, 74, 80))          # floor
    draw.rectangle([40, 60, 180, 300], fill=(60, 90, 130))         # doorway
    draw.rectangle([500, 80, 620, 300], fill=(90, 70, 60))         # shelf
    x = int(figure_x)
    draw.ellipse([x - 12, 120, x + 12, 148], fill=(210, 190, 170))  # head
    draw.rectangle([x - 18, 148, x + 18, 240], fill=(150, 40, 40))  # torso
    draw.rectangle([x - 16, 240, x - 4, 300], fill=(40, 40, 90))    # legs
    draw.rectangle([x + 4, 240, x + 16, 300], fill=(40, 40, 90))
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    return img


def fake_lm(messages, model_hint=None, **_kwargs):
    raise RuntimeError("smoke drives LM manually")


def build_manager(directory: Path) -> LuxriotManager:
    config = SimpleNamespace(
        LUXRIOT_SYSTEM_PROMPT_DEFAULT=(
            "You are EVA AI, a video surveillance describer. Be concise and factual."
        ),
        LUXRIOT_ALERTS_JSON_PROMPT="",
        LUXRIOT_SUMMARY_HISTORY_LIMIT=100,
        LUXRIOT_SUMMARY_RETENTION_DAYS=0,
        LUXRIOT_AUTO_BOOKMARKS=False,
        LUXRIOT_BOOKMARK_COOLDOWN_SEC=5.0,
        LUXRIOT_ALERTS_MAX_PER_BATCH=8,
        LUXRIOT_SUMMARY_STATE_FILE=str(directory / "summaries.json"),
        LUXRIOT_ROLLUP_CACHE_FILE=str(directory / "rollups.json"),
        LUXRIOT_ROLLUP_L1_LLM_ENABLED=False,
        LUXRIOT_ROLLUP_LLM_LEVELS="",
        LUXRIOT_ROLLUP_MIN_SOURCE_TOKENS=8000,
        LUXRIOT_ROLLUP_LLM_CHAR_BUDGET=12000,
        LUXRIOT_ROLLUP_LLM_MAX_NEW_PER_CALL=1,
        LUXRIOT_ROLLUP_SUMMARY_CACHE_LIMIT=100,
        LUXRIOT_ROLLUP_TIME_ONLY=True,
        LUXRIOT_SNAPSHOT_INTERVAL=1,
        LUXRIOT_SNAPSHOT_MAX_EDGE=640,
        LUXRIOT_CAPTURE_SOURCE="auto",
        LUXRIOT_LIVE_SEGMENT_SECONDS=15.0,
        LUXRIOT_LIVE_SEGMENT_MB=8.0,
        LUXRIOT_LIVE_SEGMENT_EVERY_N=1,
        LUXRIOT_MAX_BUFFER_FRAMES=180,
        LUXRIOT_SUMMARY_ARCHIVE_FRAMES_PER_BATCH=4,
        LUXRIOT_BASE_URL="http://luxriot.invalid",
        LUXRIOT_USERNAME="",
        LUXRIOT_PASSWORD="",
        LUXRIOT_VECTOR_SIGNALS_ENABLED=True,
        LUXRIOT_VECTOR_SIGNAL_PROBE_LIMIT=0,
        LUXRIOT_ROAD_CV_BATCH_SIGNALS=False,
        LUXRIOT_CAPTURE_ACTIVITY_NOISE_FLOOR=0.001,
    )
    return LuxriotManager(
        config=config,
        lm_callback=fake_lm,
        jpeg_encoder=jpeg_encoder,
        message_builder=oldapp._build_luxriot_messages,
    )


def main():
    with tempfile.TemporaryDirectory() as temp:
        manager = build_manager(Path(temp))
        # Persisted channel pulse: calm office channel, past warmup.
        manager.note_capture_baseline(112, {"level": 0.001, "dev": 0.0002, "buckets": 720})

        session = LuxriotCaptureSession(
            manager,
            channel_id=112,
            batch_size=12,
            prompt="Describe activity.",
            run_id="smoke-run",
            interval_override=0.25,
        )
        session.capture_activity_baseline_level = 0.001
        session.capture_activity_baseline_dev = 0.0002
        session.capture_activity_baseline_buckets = 720

        # Second 1: figure standing near the doorway (calm).
        session._accept_captured_frame(scene_frame(140), 1_000, summarize=False)
        # Second 2: slow step (normal motion).
        session._accept_captured_frame(scene_frame(180), 2_000, summarize=False)
        session._accept_captured_frame(scene_frame(210), 2_400, summarize=False)
        # Second 3: burst - fast dash across the room, motion peak blurred,
        # then a sharper frame of the same second.
        session._accept_captured_frame(scene_frame(420, blur=7.0), 3_100, summarize=False)
        session._accept_captured_frame(scene_frame(480, blur=0.6), 3_500, summarize=False)
        session._flush_capture_apex_bucket()

        frames = list(session.frames)
        print(f"frames selected: {len(frames)}")
        for frame in frames:
            sel = frame.get("capture_selection") or {}
            print(
                "  mode={mode} src={src} ax={ax} companion={comp}".format(
                    mode=sel.get("selection_mode"),
                    src=sel.get("selection_source"),
                    ax=sel.get("activity_x"),
                    comp=bool(frame.get("burst_companion")),
                )
            )

        batch = manager.create_summary_batch(
            channel_id=112,
            run_id="smoke-run",
            batch_size=12,
            prompt=(
                "Describe notable activity. Watch for fast or aggressive movement."
            ),
            model_hint=None,
            interval_sec=1.0,
            frames=frames,
        )
        attention = (batch.get("vector_signal") or {}).get("capture_attention")
        print("capture_attention:", json.dumps(attention, ensure_ascii=False))
        stats = batch.get("llm_input_stats") or {}

        messages = oldapp._build_luxriot_messages(
            "#112",
            batch["frames"],
            batch["prompt"],
            batch["system_prompt"],
        )
        message_stats = manager._estimate_message_payload_chars(messages)
        print("estimated_context_tokens:", message_stats.get("estimated_context_tokens"))
        print("image_parts:", message_stats.get("image_parts"))

        started = time.time()
        response = requests.post(
            LM_URL,
            json={
                "model": MODEL,
                "messages": messages,
                "max_tokens": 700,
                "temperature": 0.2,
            },
            timeout=240,
        )
        elapsed = time.time() - started
        if response.status_code >= 400:
            print("LM error body:", response.text[:800])
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage") or {}
        text = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        print(f"\nLM latency: {elapsed:.1f}s")
        print("usage:", json.dumps(usage))
        estimate = int(message_stats.get("estimated_context_tokens") or 0)
        actual = int(usage.get("prompt_tokens") or 0)
        if actual:
            print(f"estimate vs actual prompt tokens: {estimate} vs {actual} ({estimate / actual:.2f}x)")
        print("\n----- MODEL RESPONSE -----\n")
        print(text)


if __name__ == "__main__":
    main()
