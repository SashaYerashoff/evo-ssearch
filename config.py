"""
Configuration file for evo-ssearch oldapp.py
Contains all configurable settings with environment variable support
"""
import os
from pathlib import Path

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path('.') / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, fall back to environment variables only
    pass


def _get_bool_env(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() in ('true', '1', 'yes', 'on')


class Config:
    # Server configuration
    HOST = os.getenv('EVOSSEARCH_HOST', '0.0.0.0')  # 0.0.0.0 allows network access
    PORT = int(os.getenv('EVOSSEARCH_PORT', '5000'))
    DEBUG = _get_bool_env('EVOSSEARCH_DEBUG', 'False')

    # Embedder configuration
    EMBEDDER = os.getenv('EVOSSEARCH_EMBEDDER', 'clip').strip().lower()
    CLIP_MODEL = os.getenv('EVOSSEARCH_CLIP_MODEL', 'ViT-B/32')
    DINO_MODEL = os.getenv('EVOSSEARCH_DINO_MODEL', 'dinov3_vith16plus')
    try:
        EMB_DIM_DINO = int(os.getenv('EVOSSEARCH_EMB_DIM_DINO', '1280'))
    except ValueError:
        EMB_DIM_DINO = 1280
    DINO_WEIGHTS_PATH = os.getenv(
        'EVOSSEARCH_DINO_WEIGHTS_PATH',
        '/home/sasha/Downloads/dinoweigths/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth',
    )
    DINO_DEVICE = os.getenv('EVOSSEARCH_DINO_DEVICE', 'cuda:0').strip()

    MASK2FORMER_ENABLED = _get_bool_env('EVOSSEARCH_M2F_ENABLED', 'True')
    MASK2FORMER_MODEL = os.getenv('EVOSSEARCH_M2F_MODEL', 'facebook/mask2former-swin-base-ade-semantic')
    MASK2FORMER_DEVICE = os.getenv('EVOSSEARCH_M2F_DEVICE', DINO_DEVICE or 'cuda:0').strip()
    try:
        MASK2FORMER_MAX_SIZE = int(os.getenv('EVOSSEARCH_M2F_MAX_SIZE', '1024'))
    except (TypeError, ValueError):
        MASK2FORMER_MAX_SIZE = 1024
    if MASK2FORMER_MAX_SIZE < 256:
        MASK2FORMER_MAX_SIZE = 256

    INDEX_MODE = os.getenv('EVOSSEARCH_INDEX_MODE', 'clip').strip().lower()
    if INDEX_MODE not in {'clip', 'dino', 'dual'}:
        INDEX_MODE = 'clip'

    FUSION_ENABLED = _get_bool_env('EVOSSEARCH_FUSION_ENABLED', 'False')
    try:
        FUSION_ALPHA = float(os.getenv('EVOSSEARCH_FUSION_ALPHA', '0.7'))
    except (TypeError, ValueError):
        FUSION_ALPHA = 0.7
    FUSION_ALPHA = min(1.0, max(0.0, FUSION_ALPHA))

    RERANK_ENABLED = _get_bool_env('EVOSSEARCH_RERANK_ENABLED', 'False')
    try:
        RERANK_TOP_K = int(os.getenv('EVOSSEARCH_RERANK_TOP_K', '50'))
    except (TypeError, ValueError):
        RERANK_TOP_K = 50
    if RERANK_TOP_K < 1:
        RERANK_TOP_K = 1

    DINO_SEGMENTS_ENABLED = _get_bool_env('EVOSSEARCH_DINO_SEGMENTS_ENABLED', 'False')
    try:
        DINO_SEGMENT_MIN_PATCHES = int(os.getenv('EVOSSEARCH_DINO_SEGMENT_MIN_PATCHES', '3'))
    except (TypeError, ValueError):
        DINO_SEGMENT_MIN_PATCHES = 3
    if DINO_SEGMENT_MIN_PATCHES < 1:
        DINO_SEGMENT_MIN_PATCHES = 1
    try:
        DINO_HEATMAP_THRESHOLD = float(os.getenv('EVOSSEARCH_DINO_HEATMAP_THRESHOLD', '0.7'))
    except (TypeError, ValueError):
        DINO_HEATMAP_THRESHOLD = 0.7
    DINO_HEATMAP_THRESHOLD = min(0.99, max(0.0, DINO_HEATMAP_THRESHOLD))

    # Search result limits
    MIN_RESULTS = int(os.getenv('EVOSSEARCH_MIN_RESULTS', '3'))
    MAX_RESULTS = int(os.getenv('EVOSSEARCH_MAX_RESULTS', '48'))
    DEFAULT_RESULTS = int(os.getenv('EVOSSEARCH_DEFAULT_RESULTS', '12'))

    # Processing configuration
    BATCH_SIZE = int(os.getenv('EVOSSEARCH_BATCH_SIZE', '32'))
    THUMBNAIL_SIZE = (400, 400)
    THUMBNAIL_QUALITY = int(os.getenv('EVOSSEARCH_THUMBNAIL_QUALITY', '85'))

    # File system configuration
    INDEX_FOLDER_NAME = os.getenv('EVOSSEARCH_INDEX_FOLDER', '.clip_index')
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    # Comment system configuration
    MAX_COMMENT_LENGTH = int(os.getenv('EVOSSEARCH_MAX_COMMENT_LENGTH', '100'))

    # Security configuration
    MAX_FILE_SIZE_MB = int(os.getenv('EVOSSEARCH_MAX_FILE_SIZE_MB', '50'))

    # LM Studio / Qwen video understanding
    LM_BASE_URL = os.getenv('EVOSSEARCH_LM_BASE_URL', 'http://192.168.1.104:1234/v1').strip().rstrip('/')
    LM_MODEL = os.getenv('EVOSSEARCH_LM_MODEL', 'qwen/qwen3-vl-4b').strip()
    LM_API_KEY = os.getenv('EVOSSEARCH_LM_API_KEY', '').strip()
    try:
        LM_TIMEOUT = int(os.getenv('EVOSSEARCH_LM_TIMEOUT', '120'))
    except (TypeError, ValueError):
        LM_TIMEOUT = 120
    try:
        LM_VIDEO_DEFAULT_FRAMES = int(os.getenv('EVOSSEARCH_LM_VIDEO_DEFAULT_FRAMES', '16'))
    except (TypeError, ValueError):
        LM_VIDEO_DEFAULT_FRAMES = 16
    if LM_VIDEO_DEFAULT_FRAMES < 1:
        LM_VIDEO_DEFAULT_FRAMES = 1
    try:
        LM_VIDEO_MAX_FRAMES = int(os.getenv('EVOSSEARCH_LM_VIDEO_MAX_FRAMES', '64'))
    except (TypeError, ValueError):
        LM_VIDEO_MAX_FRAMES = 64
    if LM_VIDEO_MAX_FRAMES < 1:
        LM_VIDEO_MAX_FRAMES = 1
    LM_VIDEO_FRAME_OPTIONS = (16, 32, 64)
    try:
        LM_VIDEO_MAX_EDGE = int(os.getenv('EVOSSEARCH_LM_VIDEO_MAX_EDGE', '960'))
    except (TypeError, ValueError):
        LM_VIDEO_MAX_EDGE = 960
    try:
        LM_VIDEO_MAX_TOKENS = int(os.getenv('EVOSSEARCH_LM_VIDEO_MAX_TOKENS', '1536'))
    except (TypeError, ValueError):
        LM_VIDEO_MAX_TOKENS = 1536
    try:
        LM_VIDEO_TEMPERATURE = float(os.getenv('EVOSSEARCH_LM_VIDEO_TEMPERATURE', '0.2'))
    except (TypeError, ValueError):
        LM_VIDEO_TEMPERATURE = 0.2
    LM_VIDEO_TEMPERATURE = min(1.5, max(0.0, LM_VIDEO_TEMPERATURE))
    # Luxriot Evo integration
    LUXRIOT_BASE_URL = os.getenv('EVOSSEARCH_LUXRIOT_BASE_URL', 'http://192.168.1.102:8080').strip().rstrip('/')
    LUXRIOT_USERNAME = os.getenv('EVOSSEARCH_LUXRIOT_USERNAME', 'admin').strip()
    LUXRIOT_PASSWORD = os.getenv('EVOSSEARCH_LUXRIOT_PASSWORD', '123').strip()
    try:
        LUXRIOT_SNAPSHOT_INTERVAL = int(os.getenv('EVOSSEARCH_LUXRIOT_SNAPSHOT_INTERVAL', '5'))
    except (TypeError, ValueError):
        LUXRIOT_SNAPSHOT_INTERVAL = 5
    try:
        LUXRIOT_SNAPSHOT_MAX_EDGE = int(os.getenv('EVOSSEARCH_LUXRIOT_SNAPSHOT_MAX_EDGE', '800'))
    except (TypeError, ValueError):
        LUXRIOT_SNAPSHOT_MAX_EDGE = 800
    if LUXRIOT_SNAPSHOT_MAX_EDGE < 640:
        LUXRIOT_SNAPSHOT_MAX_EDGE = 640
    LUXRIOT_BATCH_SIZES = (12, 24, 36)
    try:
        LUXRIOT_DEFAULT_CHANNEL_ID = int(os.getenv('EVOSSEARCH_LUXRIOT_DEFAULT_CHANNEL_ID', '103'))
    except (TypeError, ValueError):
        LUXRIOT_DEFAULT_CHANNEL_ID = 103
    try:
        LUXRIOT_MAX_BUFFER_FRAMES = int(os.getenv('EVOSSEARCH_LUXRIOT_MAX_BUFFER_FRAMES', '180'))
    except (TypeError, ValueError):
        LUXRIOT_MAX_BUFFER_FRAMES = 180
    if LUXRIOT_MAX_BUFFER_FRAMES < 12:
        LUXRIOT_MAX_BUFFER_FRAMES = 12

    @classmethod
    def get_server_urls(cls):
        """Get list of server URLs for display"""
        import socket

        urls = []

        # Always include localhost
        urls.append(f"http://localhost:{cls.PORT}")

        # Add network IPs if binding to all interfaces
        if cls.HOST == '0.0.0.0':
            try:
                # Get local IP address
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.connect(("8.8.8.8", 80))
                    local_ip = s.getsockname()[0]
                    urls.append(f"http://{local_ip}:{cls.PORT}")
            except Exception:
                pass

            # Try to get all network interfaces
            try:
                hostname = socket.gethostname()
                for addr_info in socket.getaddrinfo(hostname, None):
                    ip = addr_info[4][0]
                    if ip not in ['127.0.0.1', '::1'] and not ip.startswith('169.254'):
                        url = f"http://{ip}:{cls.PORT}"
                        if url not in urls:
                            urls.append(url)
            except Exception:
                pass

        return urls

    @classmethod
    def print_startup_info(cls):
        """Print configuration info on startup"""
        print("=" * 60)
        print("evo-ssearch (oldapp.py) - Visual Search Server")
        print("=" * 60)
        print(f"Host: {cls.HOST}")
        print(f"Port: {cls.PORT}")
        print(f"Debug: {cls.DEBUG}")
        print(f"Embedder: {cls.EMBEDDER}")
        if cls.EMBEDDER == 'clip':
            print(f"  CLIP Model: {cls.CLIP_MODEL}")
        elif cls.EMBEDDER == 'dino':
            print(f"  DINO Model: {cls.DINO_MODEL}")
            print(f"  DINO Embedding Dim: {cls.EMB_DIM_DINO}")
            if cls.DINO_DEVICE:
                print(f"  DINO Device: {cls.DINO_DEVICE}")
        elif cls.EMBEDDER == 'fusion':
            print(f"  CLIP Model: {cls.CLIP_MODEL}")
            print(f"  DINO Model: {cls.DINO_MODEL}")
            print(f"  Fusion Alpha: {cls.FUSION_ALPHA:.2f}")
        print(f"Index Mode: {cls.INDEX_MODE}")
        print(f"Fusion: {'enabled' if cls.FUSION_ENABLED else 'disabled'} (alpha={cls.FUSION_ALPHA:.2f})")
        print(f"Rerank: {'enabled' if cls.RERANK_ENABLED else 'disabled'} (top_k={cls.RERANK_TOP_K})")
        print(
            "Segments: {} (min patches={})".format(
                'enabled' if cls.DINO_SEGMENTS_ENABLED else 'disabled', cls.DINO_SEGMENT_MIN_PATCHES
            )
        )
        print(
            f"Video LM: {cls.LM_MODEL} @ {cls.LM_BASE_URL or 'unset'} "
            f"(frames: default {cls.LM_VIDEO_DEFAULT_FRAMES}, max {cls.LM_VIDEO_MAX_FRAMES}, max_edge={cls.LM_VIDEO_MAX_EDGE})"
        )
        if cls.MASK2FORMER_ENABLED:
            print(
                f"Mask2Former: enabled ({cls.MASK2FORMER_MODEL}, device={cls.MASK2FORMER_DEVICE}, max_edge={cls.MASK2FORMER_MAX_SIZE})"
            )
        else:
            print("Mask2Former: disabled")
        print(f"Result Limits: {cls.MIN_RESULTS}-{cls.MAX_RESULTS} (default: {cls.DEFAULT_RESULTS})")
        print(
            f"Luxriot Evo: {cls.LUXRIOT_BASE_URL or 'unset'} "
            f"(default channel: {cls.LUXRIOT_DEFAULT_CHANNEL_ID}, "
            f"snapshot every {cls.LUXRIOT_SNAPSHOT_INTERVAL}s @ <= {cls.LUXRIOT_SNAPSHOT_MAX_EDGE}px, "
            f"buffer cap {cls.LUXRIOT_MAX_BUFFER_FRAMES} frames)"
        )
        print()
        print("Server available at:")
        for url in cls.get_server_urls():
            print(f"  {url}")
        print()
        print("Use Ctrl+C to stop the server")
        print("=" * 60)


# Create a default config instance
config = Config()
