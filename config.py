"""
Configuration file for evo-ssearch oldapp.py
Contains all configurable settings with environment variable support
"""
import os
import re
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


def _get_list_env(name: str, separator: str = ',') -> tuple[str, ...]:
    raw = os.getenv(name, '').strip()
    if not raw:
        return ()
    return tuple(item.strip() for item in raw.split(separator) if item.strip())


def _get_path_list_env(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, '').strip()
    if not raw:
        return ()
    resolved: list[str] = []
    for part in raw.split(os.pathsep):
        token = part.strip()
        if not token:
            continue
        try:
            resolved.append(str(Path(token).expanduser().resolve()))
        except Exception:
            continue
    return tuple(resolved)


def _profile_env_key(profile_id: str, suffix: str) -> str:
    normalized = re.sub(r'[^A-Za-z0-9]+', '_', profile_id).strip('_').upper()
    if not normalized:
        normalized = 'DEFAULT'
    return f'EVOSSEARCH_LM_PROFILE_{normalized}_{suffix}'


def _get_lm_profiles(
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout: int,
) -> dict[str, dict[str, object]]:
    profiles: dict[str, dict[str, object]] = {
        'default': {
            'id': 'default',
            'kind': 'general',
            'base_url': base_url,
            'model': model,
            'api_key': api_key,
            'timeout': timeout,
            'enabled': True,
            'gpu': '',
        }
    }
    raw_ids = os.getenv('EVOSSEARCH_LM_PROFILES', '').strip()
    if not raw_ids:
        return profiles
    for raw_profile_id in raw_ids.split(','):
        profile_id = raw_profile_id.strip()
        if not profile_id:
            continue
        profile_base_url = os.getenv(
            _profile_env_key(profile_id, 'BASE_URL'),
            base_url,
        ).strip().rstrip('/')
        profile_model = os.getenv(
            _profile_env_key(profile_id, 'MODEL'),
            model,
        ).strip()
        profile_api_key = os.getenv(
            _profile_env_key(profile_id, 'API_KEY'),
            api_key,
        ).strip()
        profile_kind = os.getenv(
            _profile_env_key(profile_id, 'KIND'),
            'general',
        ).strip().lower() or 'general'
        try:
            profile_timeout = int(
                os.getenv(
                    _profile_env_key(profile_id, 'TIMEOUT'),
                    str(timeout),
                )
            )
        except (TypeError, ValueError):
            profile_timeout = timeout
        profile_enabled = os.getenv(
            _profile_env_key(profile_id, 'ENABLED'),
            'true',
        ).strip().lower() in ('true', '1', 'yes', 'on')
        profile_gpu = os.getenv(
            _profile_env_key(profile_id, 'GPU'),
            '',
        ).strip()
        profiles[profile_id] = {
            'id': profile_id,
            'kind': profile_kind,
            'base_url': profile_base_url,
            'model': profile_model,
            'api_key': profile_api_key,
            'timeout': min(3600, max(1, profile_timeout)),
            'enabled': profile_enabled,
            'gpu': profile_gpu,
        }
    return profiles


def _get_app_version(default: str = "β 0.8.1") -> str:
    env_value = os.getenv("EVOSSEARCH_APP_VERSION", "").strip()
    if env_value:
        return env_value
    version_path = Path(__file__).resolve().parent / "VERSION"
    try:
        text = version_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    except Exception:
        pass
    return default


class Config:
    # Server configuration
    HOST = os.getenv('EVOSSEARCH_HOST', '0.0.0.0')  # 0.0.0.0 allows network access
    PORT = int(os.getenv('EVOSSEARCH_PORT', '5000'))
    DEBUG = _get_bool_env('EVOSSEARCH_DEBUG', 'False')
    APP_VERSION = _get_app_version()

    # Embedder configuration
    EXPERIMENTAL_EMBEDDERS_ENABLED = _get_bool_env(
        'EVOSSEARCH_EXPERIMENTAL_EMBEDDERS_ENABLED',
        'False',
    )
    PRODUCTION_CLIP_MODEL = (
        os.getenv('EVOSSEARCH_PRODUCTION_CLIP_MODEL', 'ViT-B/32').strip()
        or 'ViT-B/32'
    )
    EMBEDDER = os.getenv('EVOSSEARCH_EMBEDDER', 'clip').strip().lower()
    CLIP_MODEL = os.getenv('EVOSSEARCH_CLIP_MODEL', PRODUCTION_CLIP_MODEL).strip() or PRODUCTION_CLIP_MODEL
    if not EXPERIMENTAL_EMBEDDERS_ENABLED:
        EMBEDDER = 'clip'
        CLIP_MODEL = PRODUCTION_CLIP_MODEL
    DINO_MODEL = os.getenv('EVOSSEARCH_DINO_MODEL', 'dinov3_vith16plus')
    try:
        EMB_DIM_DINO = int(os.getenv('EVOSSEARCH_EMB_DIM_DINO', '1280'))
    except ValueError:
        EMB_DIM_DINO = 1280
    DINO_WEIGHTS_PATH = os.getenv(
        'EVOSSEARCH_DINO_WEIGHTS_PATH',
        '',
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
    if not EXPERIMENTAL_EMBEDDERS_ENABLED:
        INDEX_MODE = 'clip'
        FUSION_ENABLED = False

    RERANK_ENABLED = _get_bool_env('EVOSSEARCH_RERANK_ENABLED', 'False')
    try:
        RERANK_TOP_K = int(os.getenv('EVOSSEARCH_RERANK_TOP_K', '50'))
    except (TypeError, ValueError):
        RERANK_TOP_K = 50
    if RERANK_TOP_K < 1:
        RERANK_TOP_K = 1

    DINO_SEGMENTS_ENABLED = _get_bool_env('EVOSSEARCH_DINO_SEGMENTS_ENABLED', 'False')
    if not EXPERIMENTAL_EMBEDDERS_ENABLED:
        DINO_SEGMENTS_ENABLED = False
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
    ADMIN_TOKEN = os.getenv('EVOSSEARCH_ADMIN_TOKEN', '').strip()
    SETTINGS_LOCAL_ONLY = _get_bool_env('EVOSSEARCH_SETTINGS_LOCAL_ONLY', 'True')
    CORS_ALLOWED_ORIGINS = _get_list_env('EVOSSEARCH_CORS_ALLOWED_ORIGINS')
    ALLOWED_ROOTS = _get_path_list_env('EVOSSEARCH_ALLOWED_ROOTS')
    AUTH_ENABLED = _get_bool_env('EVOSSEARCH_AUTH_ENABLED', 'False')
    AUTH_TENANT_ID = os.getenv('EVOSSEARCH_AUTH_TENANT_ID', '').strip()
    AUTH_SESSION_COOKIE = (
        os.getenv('EVOSSEARCH_AUTH_SESSION_COOKIE', 'eva_session').strip()
        or 'eva_session'
    )
    AUTH_CSRF_COOKIE = (
        os.getenv('EVOSSEARCH_AUTH_CSRF_COOKIE', 'eva_csrf').strip()
        or 'eva_csrf'
    )
    AUTH_COOKIE_SECURE = _get_bool_env('EVOSSEARCH_AUTH_COOKIE_SECURE', 'True')
    try:
        AUTH_SESSION_TTL_HOURS = int(
            os.getenv('EVOSSEARCH_AUTH_SESSION_TTL_HOURS', '12')
        )
    except (TypeError, ValueError):
        AUTH_SESSION_TTL_HOURS = 12
    AUTH_SESSION_TTL_HOURS = min(24 * 30, max(1, AUTH_SESSION_TTL_HOURS))
    DB_STRICT_RUNTIME_ROLES = _get_bool_env(
        'EVOSSEARCH_DB_STRICT_RUNTIME_ROLES',
        os.getenv('EVA_DB_STRICT_RUNTIME_ROLES', 'False'),
    )
    SECURE_DEPLOYMENT_REQUIRED = _get_bool_env(
        'EVOSSEARCH_SECURE_DEPLOYMENT_REQUIRED',
        'False',
    )
    ARCHIVE_STORE = os.getenv('EVOSSEARCH_ARCHIVE_STORE', 'postgres').strip().lower() or 'postgres'
    if ARCHIVE_STORE != 'postgres':
        ARCHIVE_STORE = 'postgres'
    ARCHIVE_TENANT_ID = (
        os.getenv('EVOSSEARCH_ARCHIVE_TENANT_ID', AUTH_TENANT_ID).strip()
    )
    ARCHIVE_RETENTION_ENABLED = _get_bool_env(
        'EVOSSEARCH_ARCHIVE_RETENTION_ENABLED',
        'True',
    )
    try:
        ARCHIVE_MAX_RECORDS = int(os.getenv('EVOSSEARCH_ARCHIVE_MAX_RECORDS', '5000000'))
    except (TypeError, ValueError):
        ARCHIVE_MAX_RECORDS = 5000000
    ARCHIVE_MAX_RECORDS = max(1000, ARCHIVE_MAX_RECORDS)
    try:
        ARCHIVE_ROW_RETENTION_DAYS = float(
            os.getenv('EVOSSEARCH_ARCHIVE_ROW_RETENTION_DAYS', '90')
        )
    except (TypeError, ValueError):
        ARCHIVE_ROW_RETENTION_DAYS = 90.0
    ARCHIVE_ROW_RETENTION_DAYS = max(0.0, ARCHIVE_ROW_RETENTION_DAYS)
    try:
        ARCHIVE_THUMBNAIL_RETENTION_DAYS = float(
            os.getenv('EVOSSEARCH_ARCHIVE_THUMBNAIL_RETENTION_DAYS', '14')
        )
    except (TypeError, ValueError):
        ARCHIVE_THUMBNAIL_RETENTION_DAYS = 14.0
    ARCHIVE_THUMBNAIL_RETENTION_DAYS = max(0.0, ARCHIVE_THUMBNAIL_RETENTION_DAYS)
    try:
        ARCHIVE_RETENTION_PRUNE_INTERVAL_SEC = float(
            os.getenv('EVOSSEARCH_ARCHIVE_RETENTION_PRUNE_INTERVAL_SEC', '3600')
        )
    except (TypeError, ValueError):
        ARCHIVE_RETENTION_PRUNE_INTERVAL_SEC = 3600.0
    ARCHIVE_RETENTION_PRUNE_INTERVAL_SEC = max(60.0, ARCHIVE_RETENTION_PRUNE_INTERVAL_SEC)
    try:
        ARCHIVE_RETENTION_BATCH_SIZE = int(
            os.getenv('EVOSSEARCH_ARCHIVE_RETENTION_BATCH_SIZE', '5000')
        )
    except (TypeError, ValueError):
        ARCHIVE_RETENTION_BATCH_SIZE = 5000
    ARCHIVE_RETENTION_BATCH_SIZE = max(100, min(50000, ARCHIVE_RETENTION_BATCH_SIZE))

    # LM Studio / Qwen video understanding
    LM_BASE_URL = os.getenv('EVOSSEARCH_LM_BASE_URL', 'http://127.0.0.1:8088/v1').strip().rstrip('/')
    LM_MODEL = os.getenv('EVOSSEARCH_LM_MODEL', 'qwen/qwen3-vl-4b').strip()
    LM_API_KEY = os.getenv('EVOSSEARCH_LM_API_KEY', '').strip()
    try:
        LM_TIMEOUT = int(os.getenv('EVOSSEARCH_LM_TIMEOUT', '120'))
    except (TypeError, ValueError):
        LM_TIMEOUT = 120
    LM_PROFILES = _get_lm_profiles(
        base_url=LM_BASE_URL,
        model=LM_MODEL,
        api_key=LM_API_KEY,
        timeout=LM_TIMEOUT,
    )
    LM_AGENT_PROFILE_ID = (
        os.getenv(
            'EVOSSEARCH_LM_AGENT_PROFILE_ID',
            'agent' if 'agent' in LM_PROFILES else 'default',
        ).strip()
        or 'default'
    )
    LM_VLM_PROFILE_ID = (
        os.getenv(
            'EVOSSEARCH_LM_VLM_PROFILE_ID',
            'vlm' if 'vlm' in LM_PROFILES else 'default',
        ).strip()
        or 'default'
    )
    LM_VLM_BALANCER_ENABLED = _get_bool_env(
        'EVOSSEARCH_LM_VLM_BALANCER_ENABLED',
        'False',
    )
    LM_VLM_BALANCER_PROFILES = _get_list_env(
        'EVOSSEARCH_LM_VLM_BALANCER_PROFILES',
    )
    LM_MAX_TIMEOUT = max(
        int(profile.get('timeout') or LM_TIMEOUT)
        for profile in LM_PROFILES.values()
    )
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

    # Durable L0 video-summary queue. Keep disabled until PostgreSQL roles and
    # the shared spool directory are configured.
    INFERENCE_QUEUE_ENABLED = _get_bool_env(
        'EVOSSEARCH_INFERENCE_QUEUE_ENABLED',
        'False',
    )
    INFERENCE_QUEUE_TENANT_ID = os.getenv(
        'EVOSSEARCH_INFERENCE_QUEUE_TENANT_ID',
        AUTH_TENANT_ID,
    ).strip()
    INFERENCE_QUEUE_SPOOL_DIR = (
        os.getenv(
            'EVOSSEARCH_INFERENCE_QUEUE_SPOOL_DIR',
            'inference_spool',
        ).strip()
        or 'inference_spool'
    )
    try:
        INFERENCE_QUEUE_CAPACITY = int(
            os.getenv('EVOSSEARCH_INFERENCE_QUEUE_CAPACITY', '200')
        )
    except (TypeError, ValueError):
        INFERENCE_QUEUE_CAPACITY = 200
    INFERENCE_QUEUE_CAPACITY = min(100_000, max(1, INFERENCE_QUEUE_CAPACITY))
    try:
        INFERENCE_WORKER_COUNT = int(
            os.getenv('EVOSSEARCH_INFERENCE_WORKER_COUNT', '0')
        )
    except (TypeError, ValueError):
        INFERENCE_WORKER_COUNT = 0
    INFERENCE_WORKER_COUNT = min(64, max(0, INFERENCE_WORKER_COUNT))
    try:
        INFERENCE_WORKER_POLL_INTERVAL_SEC = float(
            os.getenv('EVOSSEARCH_INFERENCE_WORKER_POLL_INTERVAL_SEC', '0.25')
        )
    except (TypeError, ValueError):
        INFERENCE_WORKER_POLL_INTERVAL_SEC = 0.25
    INFERENCE_WORKER_POLL_INTERVAL_SEC = min(
        30.0,
        max(0.05, INFERENCE_WORKER_POLL_INTERVAL_SEC),
    )
    try:
        INFERENCE_WORKER_LEASE_SEC = float(
            os.getenv(
                'EVOSSEARCH_INFERENCE_WORKER_LEASE_SEC',
                str(max(180, LM_MAX_TIMEOUT * 2)),
            )
        )
    except (TypeError, ValueError):
        INFERENCE_WORKER_LEASE_SEC = float(max(180, LM_MAX_TIMEOUT * 2))
    INFERENCE_WORKER_LEASE_SEC = min(
        3600.0,
        max(10.0, INFERENCE_WORKER_LEASE_SEC),
    )
    try:
        INFERENCE_SPOOL_RETENTION_HOURS = float(
            os.getenv('EVOSSEARCH_INFERENCE_SPOOL_RETENTION_HOURS', '24')
        )
    except (TypeError, ValueError):
        INFERENCE_SPOOL_RETENTION_HOURS = 24.0
    INFERENCE_SPOOL_RETENTION_HOURS = min(
        24.0 * 30.0,
        max(1.0, INFERENCE_SPOOL_RETENTION_HOURS),
    )

    # Luxriot Evo integration
    LUXRIOT_BASE_URL = os.getenv('EVOSSEARCH_LUXRIOT_BASE_URL', 'http://luxriot-host:8080').strip().rstrip('/')
    LUXRIOT_USERNAME = os.getenv('EVOSSEARCH_LUXRIOT_USERNAME', '').strip()
    LUXRIOT_PASSWORD = os.getenv('EVOSSEARCH_LUXRIOT_PASSWORD', '').strip()
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
        LUXRIOT_SUMMARY_RETENTION_DAYS = float(
            os.getenv('EVOSSEARCH_LUXRIOT_SUMMARY_RETENTION_DAYS', '7')
        )
    except (TypeError, ValueError):
        LUXRIOT_SUMMARY_RETENTION_DAYS = 7.0
    LUXRIOT_SUMMARY_RETENTION_DAYS = max(0.0, LUXRIOT_SUMMARY_RETENTION_DAYS)
    _SUMMARY_DEFAULT_BATCH = LUXRIOT_BATCH_SIZES[0] if LUXRIOT_BATCH_SIZES else 12
    _SUMMARY_DEFAULT_LIMIT = int(
        max(
            600,
            (LUXRIOT_SUMMARY_RETENTION_DAYS * 86400.0)
            / max(1.0, float(LUXRIOT_SNAPSHOT_INTERVAL * _SUMMARY_DEFAULT_BATCH)),
        )
    )
    try:
        LUXRIOT_SUMMARY_HISTORY_LIMIT = int(
            os.getenv(
                'EVOSSEARCH_LUXRIOT_SUMMARY_HISTORY_LIMIT',
                str(_SUMMARY_DEFAULT_LIMIT),
            )
        )
    except (TypeError, ValueError):
        LUXRIOT_SUMMARY_HISTORY_LIMIT = _SUMMARY_DEFAULT_LIMIT
    LUXRIOT_SUMMARY_HISTORY_LIMIT = max(40, LUXRIOT_SUMMARY_HISTORY_LIMIT)
    try:
        LUXRIOT_SUMMARY_ARCHIVE_FRAMES_PER_BATCH = int(
            os.getenv('EVOSSEARCH_LUXRIOT_SUMMARY_ARCHIVE_FRAMES_PER_BATCH', '4')
        )
    except (TypeError, ValueError):
        LUXRIOT_SUMMARY_ARCHIVE_FRAMES_PER_BATCH = 4
    LUXRIOT_SUMMARY_ARCHIVE_FRAMES_PER_BATCH = max(
        1,
        min(16, LUXRIOT_SUMMARY_ARCHIVE_FRAMES_PER_BATCH),
    )
    try:
        ARCHIVE_ESTIMATE_CHANNELS = int(os.getenv('EVOSSEARCH_ARCHIVE_ESTIMATE_CHANNELS', '50'))
    except (TypeError, ValueError):
        ARCHIVE_ESTIMATE_CHANNELS = 50
    ARCHIVE_ESTIMATE_CHANNELS = max(1, min(10000, ARCHIVE_ESTIMATE_CHANNELS))
    try:
        ARCHIVE_ESTIMATE_FRAMES_PER_BATCH = float(
            os.getenv(
                'EVOSSEARCH_ARCHIVE_ESTIMATE_FRAMES_PER_BATCH',
                str(float(LUXRIOT_SUMMARY_ARCHIVE_FRAMES_PER_BATCH)),
            )
        )
    except (TypeError, ValueError):
        ARCHIVE_ESTIMATE_FRAMES_PER_BATCH = float(LUXRIOT_SUMMARY_ARCHIVE_FRAMES_PER_BATCH)
    ARCHIVE_ESTIMATE_FRAMES_PER_BATCH = max(0.0, min(32.0, ARCHIVE_ESTIMATE_FRAMES_PER_BATCH))
    try:
        ARCHIVE_ESTIMATE_AVG_JPEG_KB = float(
            os.getenv('EVOSSEARCH_ARCHIVE_ESTIMATE_AVG_JPEG_KB', '100')
        )
    except (TypeError, ValueError):
        ARCHIVE_ESTIMATE_AVG_JPEG_KB = 100.0
    ARCHIVE_ESTIMATE_AVG_JPEG_KB = max(1.0, min(5000.0, ARCHIVE_ESTIMATE_AVG_JPEG_KB))
    try:
        ARCHIVE_ESTIMATE_PROBE_RECORDS_PER_CHANNEL_DAY = float(
            os.getenv('EVOSSEARCH_ARCHIVE_ESTIMATE_PROBE_RECORDS_PER_CHANNEL_DAY', '250')
        )
    except (TypeError, ValueError):
        ARCHIVE_ESTIMATE_PROBE_RECORDS_PER_CHANNEL_DAY = 250.0
    ARCHIVE_ESTIMATE_PROBE_RECORDS_PER_CHANNEL_DAY = max(
        0.0,
        min(100000.0, ARCHIVE_ESTIMATE_PROBE_RECORDS_PER_CHANNEL_DAY),
    )
    try:
        LUXRIOT_DEFAULT_CHANNEL_ID = int(os.getenv('EVOSSEARCH_LUXRIOT_DEFAULT_CHANNEL_ID', '1'))
    except (TypeError, ValueError):
        LUXRIOT_DEFAULT_CHANNEL_ID = 1
    try:
        LUXRIOT_MAX_BUFFER_FRAMES = int(os.getenv('EVOSSEARCH_LUXRIOT_MAX_BUFFER_FRAMES', '180'))
    except (TypeError, ValueError):
        LUXRIOT_MAX_BUFFER_FRAMES = 180
    if LUXRIOT_MAX_BUFFER_FRAMES < 12:
        LUXRIOT_MAX_BUFFER_FRAMES = 12
    LUXRIOT_AUTO_BOOKMARKS = _get_bool_env('EVOSSEARCH_LUXRIOT_AUTO_BOOKMARKS', 'False')
    OFFLINE_VIDEO_ENABLED = _get_bool_env('EVOSSEARCH_OFFLINE_VIDEO_ENABLED', 'False')
    PROBE_SNAP_ENABLED = _get_bool_env('EVOSSEARCH_PROBE_SNAP_ENABLED', 'False')
    INDEXED_FOLDER_ENABLED = _get_bool_env('EVOSSEARCH_INDEXED_FOLDER_ENABLED', 'False')
    try:
        LUXRIOT_BOOKMARK_COOLDOWN_SEC = float(os.getenv('EVOSSEARCH_LUXRIOT_BOOKMARK_COOLDOWN_SEC', '60.0'))
    except (TypeError, ValueError):
        LUXRIOT_BOOKMARK_COOLDOWN_SEC = 60.0
    LUXRIOT_BOOKMARK_COOLDOWN_SEC = max(0.0, LUXRIOT_BOOKMARK_COOLDOWN_SEC)
    try:
        LUXRIOT_ALERTS_MAX_PER_BATCH = int(os.getenv('EVOSSEARCH_LUXRIOT_ALERTS_MAX_PER_BATCH', '8'))
    except (TypeError, ValueError):
        LUXRIOT_ALERTS_MAX_PER_BATCH = 8
    LUXRIOT_ALERTS_MAX_PER_BATCH = max(1, min(32, LUXRIOT_ALERTS_MAX_PER_BATCH))
    LUXRIOT_SYSTEM_PROMPT_DEFAULT = os.getenv('EVOSSEARCH_LUXRIOT_SYSTEM_PROMPT_DEFAULT', '').strip()
    LUXRIOT_ALERTS_JSON_PROMPT = os.getenv('EVOSSEARCH_LUXRIOT_ALERTS_JSON_PROMPT', '').strip()
    LUXRIOT_SEVERITY_MAP = {
        'info': os.getenv('EVOSSEARCH_LUXRIOT_SEV_INFO', 'info').lower(),
        'low': os.getenv('EVOSSEARCH_LUXRIOT_SEV_LOW', 'low').lower(),
        'normal': os.getenv('EVOSSEARCH_LUXRIOT_SEV_NORMAL', 'normal').lower(),
        'high': os.getenv('EVOSSEARCH_LUXRIOT_SEV_HIGH', 'high').lower(),
        'critical': os.getenv('EVOSSEARCH_LUXRIOT_SEV_CRITICAL', 'critical').lower(),
    }
    LUXRIOT_ROLLUP_L1_LLM_ENABLED = _get_bool_env('EVOSSEARCH_LUXRIOT_ROLLUP_L1_LLM_ENABLED', 'True')
    try:
        LUXRIOT_ROLLUP_L1_CHAR_BUDGET = int(os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_L1_CHAR_BUDGET', '12000'))
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_L1_CHAR_BUDGET = 12000
    if LUXRIOT_ROLLUP_L1_CHAR_BUDGET < 2000:
        LUXRIOT_ROLLUP_L1_CHAR_BUDGET = 2000
    try:
        LUXRIOT_ROLLUP_L1_MAX_NEW_PER_CALL = int(os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_L1_MAX_NEW_PER_CALL', '2'))
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_L1_MAX_NEW_PER_CALL = 2
    if LUXRIOT_ROLLUP_L1_MAX_NEW_PER_CALL < 1:
        LUXRIOT_ROLLUP_L1_MAX_NEW_PER_CALL = 1
    try:
        LUXRIOT_ROLLUP_SUMMARY_CACHE_LIMIT = int(os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_SUMMARY_CACHE_LIMIT', '800'))
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_SUMMARY_CACHE_LIMIT = 800
    if LUXRIOT_ROLLUP_SUMMARY_CACHE_LIMIT < 100:
        LUXRIOT_ROLLUP_SUMMARY_CACHE_LIMIT = 100
    LUXRIOT_ROLLUP_CACHE_FILE = (
        os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_CACHE_FILE', 'luxriot_rollups_cache.json').strip()
        or 'luxriot_rollups_cache.json'
    )
    LUXRIOT_SUMMARY_STATE_FILE = (
        os.getenv('EVOSSEARCH_LUXRIOT_SUMMARY_STATE_FILE', 'luxriot_summary_state.json').strip()
        or 'luxriot_summary_state.json'
    )
    LUXRIOT_ROLLUP_TIME_ONLY = _get_bool_env('EVOSSEARCH_LUXRIOT_ROLLUP_TIME_ONLY', 'True')
    LUXRIOT_ROLLUP_L1_MODEL = os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_L1_MODEL', '').strip()
    LUXRIOT_ROLLUP_L1_SYSTEM_PROMPT = os.getenv(
        'EVOSSEARCH_LUXRIOT_ROLLUP_L1_SYSTEM_PROMPT',
        (
            "You are a CCTV operations analyst. Summarize multiple L0 batch notes for one short time window.\n"
            "Return Markdown using exactly these sections:\n"
            "### Window Snapshot\n"
            "### Routine Baseline\n"
            "### Preserved Deviations\n"
            "### Alert Ledger\n"
            "### Alert Tuning Notes\n"
            "### Alerts/Signals\n"
            "### Operator Notes\n"
            "Append MEMORY_UPDATE_JSON with routine_baseline, active_watchlist, preserved_deviations, "
            "alert_tuning_notes, and ignore_as_routine. Rules: keep factual language; preserve every grounded "
            "alert/deviation even when the rest of the window is routine; do not classify behavior as illegal; "
            "describe observable security/safety facts requiring operator review; avoid phrases like 'L1 rollup from L0'."
        ),
    ).strip()
    LUXRIOT_ROLLUP_L2_SYSTEM_PROMPT = os.getenv(
        'EVOSSEARCH_LUXRIOT_ROLLUP_L2_SYSTEM_PROMPT',
        (
            "You are a CCTV operations analyst. Summarize multiple L1 summaries into one hour-scale view.\n"
            "Return Markdown using exactly these sections:\n"
            "### Window Snapshot\n"
            "### Routine Baseline\n"
            "### Preserved Deviations\n"
            "### Alert Ledger\n"
            "### Alert Tuning Notes\n"
            "### Alerts/Signals\n"
            "### Operator Notes\n"
            "Append MEMORY_UPDATE_JSON with routine_baseline, active_watchlist, preserved_deviations, "
            "alert_tuning_notes, and ignore_as_routine. Rules: preserve meaningful deviations from routine; "
            "never compress alerts or operator-review incidents into routine; avoid repeating unchanged background details; "
            "keep concise, operator-facing language."
        ),
    ).strip()
    LUXRIOT_ROLLUP_L3_SYSTEM_PROMPT = os.getenv(
        'EVOSSEARCH_LUXRIOT_ROLLUP_L3_SYSTEM_PROMPT',
        (
            "You are a CCTV operations analyst. Summarize multiple L2 summaries into a longer period narrative.\n"
            "Return Markdown using exactly these sections:\n"
            "### Window Snapshot\n"
            "### Routine Baseline\n"
            "### Preserved Deviations\n"
            "### Alert Ledger\n"
            "### Alert Tuning Notes\n"
            "### Alerts/Signals\n"
            "### Operator Notes\n"
            "Append MEMORY_UPDATE_JSON with durable routine_baseline, active_watchlist, preserved_deviations, "
            "alert_tuning_notes, and ignore_as_routine. Rules: emphasize trend shifts and durable signals; "
            "preserve real security/safety deviations; remove duplicate wording; focus on actionable context."
        ),
    ).strip()
    LUXRIOT_ROLLUP_LLM_LEVELS = os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_LLM_LEVELS', 'L1,L2,L3').strip() or 'L1,L2,L3'
    try:
        LUXRIOT_ROLLUP_MIN_SOURCE_TOKENS = int(os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_MIN_SOURCE_TOKENS', '8000'))
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_MIN_SOURCE_TOKENS = 8000
    if LUXRIOT_ROLLUP_MIN_SOURCE_TOKENS < 512:
        LUXRIOT_ROLLUP_MIN_SOURCE_TOKENS = 512
    try:
        LUXRIOT_ROLLUP_LLM_CHAR_BUDGET = int(
            os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_LLM_CHAR_BUDGET', str(LUXRIOT_ROLLUP_L1_CHAR_BUDGET))
        )
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_LLM_CHAR_BUDGET = LUXRIOT_ROLLUP_L1_CHAR_BUDGET
    if LUXRIOT_ROLLUP_LLM_CHAR_BUDGET < 2000:
        LUXRIOT_ROLLUP_LLM_CHAR_BUDGET = 2000
    try:
        LUXRIOT_ROLLUP_LLM_MAX_NEW_PER_CALL = int(
            os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_LLM_MAX_NEW_PER_CALL', str(LUXRIOT_ROLLUP_L1_MAX_NEW_PER_CALL))
        )
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_LLM_MAX_NEW_PER_CALL = LUXRIOT_ROLLUP_L1_MAX_NEW_PER_CALL
    if LUXRIOT_ROLLUP_LLM_MAX_NEW_PER_CALL < 1:
        LUXRIOT_ROLLUP_LLM_MAX_NEW_PER_CALL = 1
    LUXRIOT_ROLLUP_LLM_MODEL = os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_LLM_MODEL', LUXRIOT_ROLLUP_L1_MODEL).strip()
    LUXRIOT_ROLLUP_LLM_SYSTEM_PROMPT = os.getenv(
        'EVOSSEARCH_LUXRIOT_ROLLUP_LLM_SYSTEM_PROMPT',
        LUXRIOT_ROLLUP_L1_SYSTEM_PROMPT,
    ).strip()

    # Probe / CLIP monitoring
    try:
        PROBE_MAX_FRAMES = int(os.getenv('EVOSSEARCH_PROBE_MAX_FRAMES', '2000'))
    except (TypeError, ValueError):
        PROBE_MAX_FRAMES = 2000
    if PROBE_MAX_FRAMES < 100:
        PROBE_MAX_FRAMES = 100
    try:
        PROBE_THUMB_MAX_EDGE = int(os.getenv('EVOSSEARCH_PROBE_THUMB_MAX_EDGE', '256'))
    except (TypeError, ValueError):
        PROBE_THUMB_MAX_EDGE = 256
    if PROBE_THUMB_MAX_EDGE < 64:
        PROBE_THUMB_MAX_EDGE = 64
    try:
        PROBE_BOOKMARK_COOLDOWN_SEC = float(os.getenv('EVOSSEARCH_PROBE_BOOKMARK_COOLDOWN_SEC', '8.0'))
    except (TypeError, ValueError):
        PROBE_BOOKMARK_COOLDOWN_SEC = 8.0
    PROBE_BOOKMARK_COOLDOWN_SEC = max(0.0, PROBE_BOOKMARK_COOLDOWN_SEC)
    try:
        PROBE_BOOKMARK_DEDUPE_WINDOW_SEC = float(os.getenv('EVOSSEARCH_PROBE_BOOKMARK_DEDUPE_WINDOW_SEC', '20.0'))
    except (TypeError, ValueError):
        PROBE_BOOKMARK_DEDUPE_WINDOW_SEC = 20.0
    PROBE_BOOKMARK_DEDUPE_WINDOW_SEC = max(0.5, PROBE_BOOKMARK_DEDUPE_WINDOW_SEC)
    try:
        PROBE_BOOKMARK_SIM_HIGH = float(os.getenv('EVOSSEARCH_PROBE_BOOKMARK_SIM_HIGH', '0.985'))
    except (TypeError, ValueError):
        PROBE_BOOKMARK_SIM_HIGH = 0.985
    PROBE_BOOKMARK_SIM_HIGH = min(0.9999, max(0.5, PROBE_BOOKMARK_SIM_HIGH))
    try:
        PROBE_BOOKMARK_MARGIN_DELTA = float(os.getenv('EVOSSEARCH_PROBE_BOOKMARK_MARGIN_DELTA', '0.08'))
    except (TypeError, ValueError):
        PROBE_BOOKMARK_MARGIN_DELTA = 0.08
    PROBE_BOOKMARK_MARGIN_DELTA = max(0.0, PROBE_BOOKMARK_MARGIN_DELTA)
    try:
        PROBE_BOOKMARK_SCORE_DELTA = float(os.getenv('EVOSSEARCH_PROBE_BOOKMARK_SCORE_DELTA', '0.08'))
    except (TypeError, ValueError):
        PROBE_BOOKMARK_SCORE_DELTA = 0.08
    PROBE_BOOKMARK_SCORE_DELTA = max(0.0, PROBE_BOOKMARK_SCORE_DELTA)
    try:
        PROBE_BOOKMARK_MAX_FRAME_GAP = int(os.getenv('EVOSSEARCH_PROBE_BOOKMARK_MAX_FRAME_GAP', '8'))
    except (TypeError, ValueError):
        PROBE_BOOKMARK_MAX_FRAME_GAP = 8
    if PROBE_BOOKMARK_MAX_FRAME_GAP < 1:
        PROBE_BOOKMARK_MAX_FRAME_GAP = 1

    # Detection archive + adaptive retention
    DETECTIONS_ARCHIVE_ENABLED = _get_bool_env('EVOSSEARCH_DETECTIONS_ARCHIVE_ENABLED', 'True')
    DETECTIONS_ARCHIVE_DIR = os.getenv('EVOSSEARCH_DETECTIONS_ARCHIVE_DIR', 'detections_archive').strip() or 'detections_archive'
    try:
        DETECTIONS_ARCHIVE_JPEG_QUALITY = int(os.getenv('EVOSSEARCH_DETECTIONS_ARCHIVE_JPEG_QUALITY', '88'))
    except (TypeError, ValueError):
        DETECTIONS_ARCHIVE_JPEG_QUALITY = 88
    DETECTIONS_ARCHIVE_JPEG_QUALITY = max(60, min(95, DETECTIONS_ARCHIVE_JPEG_QUALITY))

    DETECTIONS_RETENTION_ENABLED = _get_bool_env('EVOSSEARCH_DETECTIONS_RETENTION_ENABLED', 'True')
    DETECTIONS_RETENTION_DROP_SKIPPED = _get_bool_env('EVOSSEARCH_DETECTIONS_RETENTION_DROP_SKIPPED', 'True')
    try:
        DETECTIONS_RETENTION_WINDOW_SEC = float(os.getenv('EVOSSEARCH_DETECTIONS_RETENTION_WINDOW_SEC', '6'))
    except (TypeError, ValueError):
        DETECTIONS_RETENTION_WINDOW_SEC = 6.0
    DETECTIONS_RETENTION_WINDOW_SEC = max(0.5, DETECTIONS_RETENTION_WINDOW_SEC)
    try:
        DETECTIONS_RETENTION_FORCE_KEEP_SEC = float(os.getenv('EVOSSEARCH_DETECTIONS_RETENTION_FORCE_KEEP_SEC', '20'))
    except (TypeError, ValueError):
        DETECTIONS_RETENTION_FORCE_KEEP_SEC = 20.0
    DETECTIONS_RETENTION_FORCE_KEEP_SEC = max(1.0, DETECTIONS_RETENTION_FORCE_KEEP_SEC)
    try:
        DETECTIONS_RETENTION_SIMILARITY_HIGH = float(os.getenv('EVOSSEARCH_DETECTIONS_RETENTION_SIMILARITY_HIGH', '0.985'))
    except (TypeError, ValueError):
        DETECTIONS_RETENTION_SIMILARITY_HIGH = 0.985
    DETECTIONS_RETENTION_SIMILARITY_HIGH = min(0.9999, max(0.5, DETECTIONS_RETENTION_SIMILARITY_HIGH))
    try:
        DETECTIONS_RETENTION_SIMILARITY_LOW = float(os.getenv('EVOSSEARCH_DETECTIONS_RETENTION_SIMILARITY_LOW', '0.94'))
    except (TypeError, ValueError):
        DETECTIONS_RETENTION_SIMILARITY_LOW = 0.94
    DETECTIONS_RETENTION_SIMILARITY_LOW = min(
        DETECTIONS_RETENTION_SIMILARITY_HIGH - 0.001,
        max(0.3, DETECTIONS_RETENTION_SIMILARITY_LOW),
    )
    try:
        DETECTIONS_RETENTION_MARGIN_DELTA = float(os.getenv('EVOSSEARCH_DETECTIONS_RETENTION_MARGIN_DELTA', '0.08'))
    except (TypeError, ValueError):
        DETECTIONS_RETENTION_MARGIN_DELTA = 0.08
    DETECTIONS_RETENTION_MARGIN_DELTA = max(0.0, DETECTIONS_RETENTION_MARGIN_DELTA)
    try:
        DETECTIONS_RETENTION_SCORE_DELTA = float(os.getenv('EVOSSEARCH_DETECTIONS_RETENTION_SCORE_DELTA', '0.08'))
    except (TypeError, ValueError):
        DETECTIONS_RETENTION_SCORE_DELTA = 0.08
    DETECTIONS_RETENTION_SCORE_DELTA = max(0.0, DETECTIONS_RETENTION_SCORE_DELTA)

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
                        if not isinstance(ip, str):
                            continue
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
        print(
            "Inference queue: {} (capacity={}, local_workers={}, spool={})".format(
                'enabled' if cls.INFERENCE_QUEUE_ENABLED else 'disabled',
                cls.INFERENCE_QUEUE_CAPACITY,
                cls.INFERENCE_WORKER_COUNT,
                cls.INFERENCE_QUEUE_SPOOL_DIR,
            )
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
            f"buffer cap {cls.LUXRIOT_MAX_BUFFER_FRAMES} frames, "
            f"auto-bookmarks {'on' if cls.LUXRIOT_AUTO_BOOKMARKS else 'off'})"
        )
        print(
            "Rollups: levels={} mode={} min_tokens={} char_budget={} max_new={} cache_limit={} cache_file={} state_file={} model={}".format(
                cls.LUXRIOT_ROLLUP_LLM_LEVELS,
                'time-only' if cls.LUXRIOT_ROLLUP_TIME_ONLY else 'token-gated',
                cls.LUXRIOT_ROLLUP_MIN_SOURCE_TOKENS,
                cls.LUXRIOT_ROLLUP_LLM_CHAR_BUDGET,
                cls.LUXRIOT_ROLLUP_LLM_MAX_NEW_PER_CALL,
                cls.LUXRIOT_ROLLUP_SUMMARY_CACHE_LIMIT,
                cls.LUXRIOT_ROLLUP_CACHE_FILE,
                cls.LUXRIOT_SUMMARY_STATE_FILE,
                cls.LUXRIOT_ROLLUP_LLM_MODEL or cls.LM_MODEL,
            )
        )
        print(
            "Detections archive: {} ({}, adaptive retention {}, keep_all_rows {}, window {}s, force {}s)".format(
                'enabled' if cls.DETECTIONS_ARCHIVE_ENABLED else 'disabled',
                cls.DETECTIONS_ARCHIVE_DIR,
                'on' if cls.DETECTIONS_RETENTION_ENABLED else 'off',
                'on' if not cls.DETECTIONS_RETENTION_DROP_SKIPPED else 'off',
                cls.DETECTIONS_RETENTION_WINDOW_SEC,
                cls.DETECTIONS_RETENTION_FORCE_KEEP_SEC,
            )
        )
        print(
            "Security: upload_limit={}MB, settings_local_only={}, admin_token={}, cors_origins={}, allowed_roots={}".format(
                cls.MAX_FILE_SIZE_MB,
                'on' if cls.SETTINGS_LOCAL_ONLY else 'off',
                'set' if cls.ADMIN_TOKEN else 'unset',
                len(cls.CORS_ALLOWED_ORIGINS),
                len(cls.ALLOWED_ROOTS),
            )
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
