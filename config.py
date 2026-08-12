"""
Configuration file for evo-ssearch oldapp.py
Contains all configurable settings with environment variable support
"""
import os
import re
import hashlib
from pathlib import Path

from local_video_source import parse_local_video_sources

# Keep only names, never values.  After python-dotenv runs, ``os.environ`` no
# longer reveals whether a setting came from systemd/process environment or the
# project file.  Settings diagnostics use this frozen set to explain precedence
# without exposing secrets.
ENV_KEYS_BEFORE_DOTENV = frozenset(os.environ.keys())
ENV_VALUE_HASHES_BEFORE_DOTENV = {
    key: hashlib.sha256(str(value).encode("utf-8", errors="replace")).hexdigest()
    for key, value in os.environ.items()
    if key.startswith("EVOSSEARCH_")
}
CONFIG_ENV_FILE_BEFORE_DOTENV = str(
    os.environ.get("EVOSSEARCH_CONFIG_ENV_FILE") or ""
).strip()

# Load the same env file declared by the service when one is explicit.  A
# systemd EnvironmentFile still wins because python-dotenv does not override
# existing process variables, but direct/admin launches now inspect the same
# file that Settings will update instead of an unrelated cwd ``.env``.
try:
    from dotenv import load_dotenv
    env_path = (
        Path(CONFIG_ENV_FILE_BEFORE_DOTENV).expanduser()
        if CONFIG_ENV_FILE_BEFORE_DOTENV
        else Path('.') / '.env'
    )
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


def _get_lm_profile_max_inflight(profile_id: str, default: int = 1) -> int:
    """Return the endpoint concurrency advertised for one named profile."""

    raw = os.getenv(_profile_env_key(profile_id, 'MAX_INFLIGHT'))
    if raw is None or not str(raw).strip():
        raw = os.getenv('EVOSSEARCH_LM_MAX_INFLIGHT', str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = int(default)
    return max(1, min(64, value))


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


def _get_app_version(default: str = "β 0.8.7") -> str:
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
    # Configuration provenance is frozen before python-dotenv mutates the
    # process environment.  Expose it on the runtime config object because the
    # Settings API receives ``config = Config()`` rather than this module.
    ENV_KEYS_BEFORE_DOTENV = ENV_KEYS_BEFORE_DOTENV
    ENV_VALUE_HASHES_BEFORE_DOTENV = ENV_VALUE_HASHES_BEFORE_DOTENV
    CONFIG_ENV_FILE_BEFORE_DOTENV = CONFIG_ENV_FILE_BEFORE_DOTENV

    # Server configuration
    HOST = os.getenv('EVOSSEARCH_HOST', '0.0.0.0')  # 0.0.0.0 allows network access
    PORT = int(os.getenv('EVOSSEARCH_PORT', '5000'))
    DEBUG = _get_bool_env('EVOSSEARCH_DEBUG', 'False')
    APP_VERSION = _get_app_version()
    UI_MODE = os.getenv('EVOSSEARCH_UI_MODE', 'legacy').strip().lower()
    if UI_MODE not in {'legacy', 'react'}:
        UI_MODE = 'legacy'
    OFFLINE_MODE = _get_bool_env('EVOSSEARCH_OFFLINE_MODE', 'True')
    MODEL_CACHE_DIR = Path(
        os.getenv(
            'EVOSSEARCH_MODEL_CACHE_DIR',
            str(Path.home() / '.cache' / 'eva-ai' / 'models'),
        )
    ).expanduser()
    OPENAI_CLIP_CACHE_DIR = Path(
        os.getenv(
            'EVOSSEARCH_OPENAI_CLIP_CACHE_DIR',
            str(Path.home() / '.cache' / 'clip'),
        )
    ).expanduser()

    # Embedder configuration
    EXPERIMENTAL_EMBEDDERS_ENABLED = _get_bool_env(
        'EVOSSEARCH_EXPERIMENTAL_EMBEDDERS_ENABLED',
        'True',
    )
    PRODUCTION_CLIP_MODEL = (
        os.getenv(
            'EVOSSEARCH_PRODUCTION_CLIP_MODEL',
            'google/siglip2-base-patch16-224',
        ).strip()
        or 'google/siglip2-base-patch16-224'
    )
    EMBEDDER = os.getenv('EVOSSEARCH_EMBEDDER', 'clip').strip().lower()
    EMBEDDER_EAGER_LOAD = _get_bool_env(
        'EVOSSEARCH_EMBEDDER_EAGER_LOAD',
        'false',
    )
    # A fallback changes the embedding space, invalidating archive vectors and
    # calibrated probe thresholds. Keep it opt-in so production never silently
    # swaps SigLIP2 for OpenAI CLIP after a load failure.
    EMBEDDER_FALLBACK_ENABLED = _get_bool_env(
        'EVOSSEARCH_EMBEDDER_FALLBACK_ENABLED',
        'False',
    )
    CLIP_DEVICE = (
        os.getenv('EVOSSEARCH_CLIP_DEVICE', 'auto').strip().lower()
        or 'auto'
    )
    CLIP_MODEL = os.getenv('EVOSSEARCH_CLIP_MODEL', PRODUCTION_CLIP_MODEL).strip() or PRODUCTION_CLIP_MODEL
    CLIP_MODEL_REVISION = os.getenv(
        'EVOSSEARCH_CLIP_MODEL_REVISION',
        '75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2',
    ).strip()
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
    try:
        TRUSTED_PROXY_HOPS = int(
            os.getenv('EVOSSEARCH_TRUSTED_PROXY_HOPS', '0')
        )
    except (TypeError, ValueError):
        TRUSTED_PROXY_HOPS = 0
    TRUSTED_PROXY_HOPS = min(4, max(0, TRUSTED_PROXY_HOPS))
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
    # "Remember me" session lifetime — used when the login request opts in.
    try:
        AUTH_SESSION_REMEMBER_TTL_HOURS = int(
            os.getenv('EVOSSEARCH_AUTH_SESSION_REMEMBER_TTL_HOURS', str(24 * 30))
        )
    except (TypeError, ValueError):
        AUTH_SESSION_REMEMBER_TTL_HOURS = 24 * 30
    AUTH_SESSION_REMEMBER_TTL_HOURS = min(
        24 * 30, max(AUTH_SESSION_TTL_HOURS, AUTH_SESSION_REMEMBER_TTL_HOURS)
    )
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
    LM_VISION_HEALTH_STATE_FILE = os.getenv(
        'EVOSSEARCH_LM_VISION_HEALTH_STATE_FILE',
        '',
    ).strip()
    try:
        LM_VISION_HEALTH_MAX_AGE_SEC = float(
            os.getenv('EVOSSEARCH_LM_VISION_HEALTH_MAX_AGE_SEC', '180')
        )
    except (TypeError, ValueError):
        LM_VISION_HEALTH_MAX_AGE_SEC = 180.0
    LM_VISION_HEALTH_MAX_AGE_SEC = min(
        3600.0,
        max(30.0, LM_VISION_HEALTH_MAX_AGE_SEC),
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
        LM_VIDEO_INPUT_WARNING_CHARS = int(os.getenv('EVOSSEARCH_LM_VIDEO_INPUT_WARNING_CHARS', '24000'))
    except (TypeError, ValueError):
        LM_VIDEO_INPUT_WARNING_CHARS = 24000
    LM_VIDEO_INPUT_WARNING_CHARS = max(4000, LM_VIDEO_INPUT_WARNING_CHARS)
    try:
        LM_VIDEO_IMAGE_PAYLOAD_WARNING_CHARS = int(os.getenv('EVOSSEARCH_LM_VIDEO_IMAGE_PAYLOAD_WARNING_CHARS', '2500000'))
    except (TypeError, ValueError):
        LM_VIDEO_IMAGE_PAYLOAD_WARNING_CHARS = 2500000
    LM_VIDEO_IMAGE_PAYLOAD_WARNING_CHARS = max(100000, LM_VIDEO_IMAGE_PAYLOAD_WARNING_CHARS)
    # Rough VLM context ceiling used only for llm_input_stats warnings
    # (~4 chars/token + ~300 visual tokens per <=640px image).
    try:
        LM_VIDEO_CONTEXT_TOKENS_WARN = int(os.getenv('EVOSSEARCH_LM_VIDEO_CONTEXT_TOKENS_WARN', '7000'))
    except (TypeError, ValueError):
        LM_VIDEO_CONTEXT_TOKENS_WARN = 7000
    LM_VIDEO_CONTEXT_TOKENS_WARN = max(1000, LM_VIDEO_CONTEXT_TOKENS_WARN)
    try:
        LM_VIDEO_TEMPERATURE = float(os.getenv('EVOSSEARCH_LM_VIDEO_TEMPERATURE', '0.2'))
    except (TypeError, ValueError):
        LM_VIDEO_TEMPERATURE = 0.2
    LM_VIDEO_TEMPERATURE = min(1.5, max(0.0, LM_VIDEO_TEMPERATURE))
    try:
        LM_VIDEO_REPETITION_PENALTY = float(
            os.getenv('EVOSSEARCH_LM_VIDEO_REPETITION_PENALTY', '1.08')
        )
    except (TypeError, ValueError):
        LM_VIDEO_REPETITION_PENALTY = 1.08
    LM_VIDEO_REPETITION_PENALTY = min(
        1.3,
        max(1.0, LM_VIDEO_REPETITION_PENALTY),
    )

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
    LOCAL_VIDEO_SOURCES = parse_local_video_sources(
        os.getenv('EVOSSEARCH_LOCAL_VIDEO_SOURCES_JSON', '')
    )
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
    LUXRIOT_CAPTURE_SOURCE = os.getenv('EVOSSEARCH_LUXRIOT_CAPTURE_SOURCE', 'auto').strip().lower() or 'auto'
    if LUXRIOT_CAPTURE_SOURCE not in {'auto', 'snapshot', 'live_segment'}:
        LUXRIOT_CAPTURE_SOURCE = 'auto'
    try:
        LUXRIOT_LIVE_SEGMENT_SECONDS = float(os.getenv('EVOSSEARCH_LUXRIOT_LIVE_SEGMENT_SECONDS', '60.0'))
    except (TypeError, ValueError):
        LUXRIOT_LIVE_SEGMENT_SECONDS = 60.0
    LUXRIOT_LIVE_SEGMENT_SECONDS = max(2.0, min(60.0, LUXRIOT_LIVE_SEGMENT_SECONDS))
    try:
        LUXRIOT_LIVE_SEGMENT_MB = float(os.getenv('EVOSSEARCH_LUXRIOT_LIVE_SEGMENT_MB', '8.0'))
    except (TypeError, ValueError):
        LUXRIOT_LIVE_SEGMENT_MB = 8.0
    LUXRIOT_LIVE_SEGMENT_MB = max(0.5, min(128.0, LUXRIOT_LIVE_SEGMENT_MB))
    try:
        LUXRIOT_LIVE_SEGMENT_EVERY_N = int(os.getenv('EVOSSEARCH_LUXRIOT_LIVE_SEGMENT_EVERY_N', '25'))
    except (TypeError, ValueError):
        LUXRIOT_LIVE_SEGMENT_EVERY_N = 25
    LUXRIOT_LIVE_SEGMENT_EVERY_N = max(1, min(240, LUXRIOT_LIVE_SEGMENT_EVERY_N))
    try:
        LUXRIOT_LIVE_SEGMENT_FPS = float(os.getenv('EVOSSEARCH_LUXRIOT_LIVE_SEGMENT_FPS', '2.0'))
    except (TypeError, ValueError):
        LUXRIOT_LIVE_SEGMENT_FPS = 2.0
    LUXRIOT_LIVE_SEGMENT_FPS = max(0.2, min(10.0, LUXRIOT_LIVE_SEGMENT_FPS))
    try:
        LUXRIOT_SUMMARY_MAX_BATCH_FRAMES = int(
            os.getenv('EVOSSEARCH_LUXRIOT_SUMMARY_MAX_BATCH_FRAMES', '16')
        )
    except (TypeError, ValueError):
        LUXRIOT_SUMMARY_MAX_BATCH_FRAMES = 16
    LUXRIOT_SUMMARY_MAX_BATCH_FRAMES = max(
        1,
        min(16, LUXRIOT_SUMMARY_MAX_BATCH_FRAMES),
    )
    try:
        LUXRIOT_SUMMARY_MAX_WINDOW_SEC = float(
            os.getenv('EVOSSEARCH_LUXRIOT_SUMMARY_MAX_WINDOW_SEC', '60')
        )
    except (TypeError, ValueError):
        LUXRIOT_SUMMARY_MAX_WINDOW_SEC = 60.0
    LUXRIOT_SUMMARY_MAX_WINDOW_SEC = max(
        5.0,
        min(300.0, LUXRIOT_SUMMARY_MAX_WINDOW_SEC),
    )
    try:
        LUXRIOT_SUMMARY_QUIET_CADENCE_SEC = float(
            os.getenv('EVOSSEARCH_LUXRIOT_SUMMARY_QUIET_CADENCE_SEC', '5')
        )
    except (TypeError, ValueError):
        LUXRIOT_SUMMARY_QUIET_CADENCE_SEC = 5.0
    LUXRIOT_SUMMARY_QUIET_CADENCE_SEC = max(
        1.0,
        min(60.0, LUXRIOT_SUMMARY_QUIET_CADENCE_SEC),
    )
    try:
        LUXRIOT_SUMMARY_NORMAL_CADENCE_SEC = float(
            os.getenv('EVOSSEARCH_LUXRIOT_SUMMARY_NORMAL_CADENCE_SEC', '2')
        )
    except (TypeError, ValueError):
        LUXRIOT_SUMMARY_NORMAL_CADENCE_SEC = 2.0
    LUXRIOT_SUMMARY_NORMAL_CADENCE_SEC = max(
        0.5,
        min(LUXRIOT_SUMMARY_QUIET_CADENCE_SEC, LUXRIOT_SUMMARY_NORMAL_CADENCE_SEC),
    )
    try:
        LUXRIOT_SUMMARY_BURST_CADENCE_SEC = float(
            os.getenv('EVOSSEARCH_LUXRIOT_SUMMARY_BURST_CADENCE_SEC', '1')
        )
    except (TypeError, ValueError):
        LUXRIOT_SUMMARY_BURST_CADENCE_SEC = 1.0
    LUXRIOT_SUMMARY_BURST_CADENCE_SEC = max(
        0.2,
        min(LUXRIOT_SUMMARY_NORMAL_CADENCE_SEC, LUXRIOT_SUMMARY_BURST_CADENCE_SEC),
    )
    LUXRIOT_ATTENTION_SCHEDULER_ENABLED = _get_bool_env(
        'EVOSSEARCH_LUXRIOT_ATTENTION_SCHEDULER_ENABLED',
        'False',
    )
    LUXRIOT_ATTENTION_EPISODE_DISPATCH_ENABLED = _get_bool_env(
        'EVOSSEARCH_LUXRIOT_ATTENTION_EPISODE_DISPATCH_ENABLED',
        'False',
    )
    LUXRIOT_ATTENTION_EMBED_ALL_CHANNELS = _get_bool_env(
        'EVOSSEARCH_LUXRIOT_ATTENTION_EMBED_ALL_CHANNELS',
        'False',
    )
    try:
        LUXRIOT_ATTENTION_EMBEDDING_CADENCE_MS = int(
            os.getenv(
                'EVOSSEARCH_LUXRIOT_ATTENTION_EMBEDDING_CADENCE_MS',
                '1000',
            )
        )
    except (TypeError, ValueError):
        LUXRIOT_ATTENTION_EMBEDDING_CADENCE_MS = 1000
    LUXRIOT_ATTENTION_EMBEDDING_CADENCE_MS = max(
        200,
        min(60000, LUXRIOT_ATTENTION_EMBEDDING_CADENCE_MS),
    )
    LUXRIOT_CLIP_ASYNC_ENABLED = _get_bool_env(
        'EVOSSEARCH_LUXRIOT_CLIP_ASYNC_ENABLED',
        'True',
    )
    try:
        LUXRIOT_CLIP_ASYNC_WORKERS = int(
            os.getenv('EVOSSEARCH_LUXRIOT_CLIP_ASYNC_WORKERS', '8')
        )
    except (TypeError, ValueError):
        LUXRIOT_CLIP_ASYNC_WORKERS = 8
    LUXRIOT_CLIP_ASYNC_WORKERS = max(
        1,
        min(16, LUXRIOT_CLIP_ASYNC_WORKERS),
    )
    try:
        LUXRIOT_CLIP_ASYNC_QUEUE_CAPACITY = int(
            os.getenv(
                'EVOSSEARCH_LUXRIOT_CLIP_ASYNC_QUEUE_CAPACITY',
                '64',
            )
        )
    except (TypeError, ValueError):
        LUXRIOT_CLIP_ASYNC_QUEUE_CAPACITY = 64
    LUXRIOT_CLIP_ASYNC_QUEUE_CAPACITY = max(
        LUXRIOT_CLIP_ASYNC_WORKERS,
        min(512, LUXRIOT_CLIP_ASYNC_QUEUE_CAPACITY),
    )
    LUXRIOT_ATTENTION_STORAGE_ENABLED = _get_bool_env(
        'EVOSSEARCH_LUXRIOT_ATTENTION_STORAGE_ENABLED',
        'False',
    )
    try:
        LUXRIOT_ATTENTION_RING_SECONDS = float(
            os.getenv('EVOSSEARCH_LUXRIOT_ATTENTION_RING_SECONDS', '90')
        )
    except (TypeError, ValueError):
        LUXRIOT_ATTENTION_RING_SECONDS = 90.0
    LUXRIOT_ATTENTION_RING_SECONDS = max(
        30.0,
        min(600.0, LUXRIOT_ATTENTION_RING_SECONDS),
    )
    try:
        LUXRIOT_ATTENTION_REQUESTS_PER_MINUTE = float(
            os.getenv('EVOSSEARCH_LUXRIOT_ATTENTION_REQUESTS_PER_MINUTE', '6')
        )
    except (TypeError, ValueError):
        LUXRIOT_ATTENTION_REQUESTS_PER_MINUTE = 6.0
    LUXRIOT_ATTENTION_REQUESTS_PER_MINUTE = max(
        0.2,
        min(120.0, LUXRIOT_ATTENTION_REQUESTS_PER_MINUTE),
    )
    # Slot-seconds measure wall-clock occupancy. A batching server can execute
    # several live L0 requests concurrently, so charging every request its full
    # wall time would silently turn a six-request/minute budget into roughly two
    # requests/minute once inference takes ~30 seconds. Keep one protected lane
    # for agent/alert work and let the operator override the derived live width.
    _LUXRIOT_ATTENTION_L0_SLOT_PARALLELISM_DEFAULT = max(
        1,
        min(
            3,
            _get_lm_profile_max_inflight(LM_VLM_PROFILE_ID, 1) - 1,
        ),
    )
    try:
        LUXRIOT_ATTENTION_L0_SLOT_PARALLELISM = float(
            os.getenv(
                'EVOSSEARCH_LUXRIOT_ATTENTION_L0_SLOT_PARALLELISM',
                str(_LUXRIOT_ATTENTION_L0_SLOT_PARALLELISM_DEFAULT),
            )
        )
    except (TypeError, ValueError):
        LUXRIOT_ATTENTION_L0_SLOT_PARALLELISM = float(
            _LUXRIOT_ATTENTION_L0_SLOT_PARALLELISM_DEFAULT
        )
    LUXRIOT_ATTENTION_L0_SLOT_PARALLELISM = max(
        1.0,
        min(16.0, LUXRIOT_ATTENTION_L0_SLOT_PARALLELISM),
    )
    try:
        LUXRIOT_ATTENTION_MAX_OUTSTANDING = int(
            os.getenv('EVOSSEARCH_LUXRIOT_ATTENTION_MAX_OUTSTANDING', '1')
        )
    except (TypeError, ValueError):
        LUXRIOT_ATTENTION_MAX_OUTSTANDING = 1
    LUXRIOT_ATTENTION_MAX_OUTSTANDING = max(
        1,
        min(16, LUXRIOT_ATTENTION_MAX_OUTSTANDING),
    )
    try:
        LIVE_CLIP_BATCH_SIZE = int(
            os.getenv('EVOSSEARCH_LIVE_CLIP_BATCH_SIZE', '8')
        )
    except (TypeError, ValueError):
        LIVE_CLIP_BATCH_SIZE = 8
    LIVE_CLIP_BATCH_SIZE = max(1, min(32, LIVE_CLIP_BATCH_SIZE))
    try:
        LIVE_CLIP_BATCH_WAIT_MS = float(
            os.getenv('EVOSSEARCH_LIVE_CLIP_BATCH_WAIT_MS', '75')
        )
    except (TypeError, ValueError):
        LIVE_CLIP_BATCH_WAIT_MS = 75.0
    LIVE_CLIP_BATCH_WAIT_MS = max(0.0, min(500.0, LIVE_CLIP_BATCH_WAIT_MS))
    try:
        LIVE_CLIP_BATCH_QUEUE_CAPACITY = int(
            os.getenv('EVOSSEARCH_LIVE_CLIP_BATCH_QUEUE_CAPACITY', '128')
        )
    except (TypeError, ValueError):
        LIVE_CLIP_BATCH_QUEUE_CAPACITY = 128
    LIVE_CLIP_BATCH_QUEUE_CAPACITY = max(
        LIVE_CLIP_BATCH_SIZE,
        min(4096, LIVE_CLIP_BATCH_QUEUE_CAPACITY),
    )
    try:
        LIVE_CLIP_BATCH_TIMEOUT_SEC = float(
            os.getenv('EVOSSEARCH_LIVE_CLIP_BATCH_TIMEOUT_SEC', '45')
        )
    except (TypeError, ValueError):
        LIVE_CLIP_BATCH_TIMEOUT_SEC = 45.0
    LIVE_CLIP_BATCH_TIMEOUT_SEC = max(
        1.0,
        min(120.0, LIVE_CLIP_BATCH_TIMEOUT_SEC),
    )
    SEMANTIC_SNAPSHOT_ARCHIVE_ENABLED = _get_bool_env(
        'EVOSSEARCH_SEMANTIC_SNAPSHOT_ARCHIVE_ENABLED',
        'True',
    )
    try:
        SEMANTIC_SNAPSHOT_ARCHIVE_QUEUE = int(
            os.getenv('EVOSSEARCH_SEMANTIC_SNAPSHOT_ARCHIVE_QUEUE', '512')
        )
    except (TypeError, ValueError):
        SEMANTIC_SNAPSHOT_ARCHIVE_QUEUE = 512
    SEMANTIC_SNAPSHOT_ARCHIVE_QUEUE = max(
        32,
        min(16384, SEMANTIC_SNAPSHOT_ARCHIVE_QUEUE),
    )
    try:
        SEMANTIC_SNAPSHOT_ARCHIVE_BATCH_SIZE = int(
            os.getenv('EVOSSEARCH_SEMANTIC_SNAPSHOT_ARCHIVE_BATCH_SIZE', '32')
        )
    except (TypeError, ValueError):
        SEMANTIC_SNAPSHOT_ARCHIVE_BATCH_SIZE = 32
    SEMANTIC_SNAPSHOT_ARCHIVE_BATCH_SIZE = max(
        1,
        min(
            SEMANTIC_SNAPSHOT_ARCHIVE_QUEUE,
            SEMANTIC_SNAPSHOT_ARCHIVE_BATCH_SIZE,
        ),
    )
    try:
        LUXRIOT_ATTENTION_POSTROLL_SEC = float(
            os.getenv('EVOSSEARCH_LUXRIOT_ATTENTION_POSTROLL_SEC', '3')
        )
    except (TypeError, ValueError):
        LUXRIOT_ATTENTION_POSTROLL_SEC = 3.0
    LUXRIOT_ATTENTION_POSTROLL_SEC = max(
        0.0,
        min(30.0, LUXRIOT_ATTENTION_POSTROLL_SEC),
    )
    try:
        LUXRIOT_ATTENTION_MAX_VLM_FRAMES = int(
            os.getenv('EVOSSEARCH_LUXRIOT_ATTENTION_MAX_VLM_FRAMES', '8')
        )
    except (TypeError, ValueError):
        LUXRIOT_ATTENTION_MAX_VLM_FRAMES = 8
    LUXRIOT_ATTENTION_MAX_VLM_FRAMES = max(
        2,
        min(24, LUXRIOT_ATTENTION_MAX_VLM_FRAMES),
    )
    # Incident attention is bounded independently from durable lifecycle state:
    # two foreground incidents in the ordinary case, up to four in the model
    # envelope when critical/operator-selected work competes, and eight hot
    # unresolved incidents retained by the scheduler.
    _INCIDENT_ATTENTION_DEFAULTS = {
        'LUXRIOT_INCIDENT_FOREGROUND_LIMIT': (
            'EVOSSEARCH_LUXRIOT_INCIDENT_FOREGROUND_LIMIT', 2, 1, 4
        ),
        'LUXRIOT_INCIDENT_FOREGROUND_HARD_LIMIT': (
            'EVOSSEARCH_LUXRIOT_INCIDENT_FOREGROUND_HARD_LIMIT', 4, 2, 8
        ),
        'LUXRIOT_INCIDENT_HOT_LIMIT': (
            'EVOSSEARCH_LUXRIOT_INCIDENT_HOT_LIMIT', 8, 4, 32
        ),
        'LUXRIOT_INCIDENT_TRACKED_LIMIT': (
            'EVOSSEARCH_LUXRIOT_INCIDENT_TRACKED_LIMIT', 64, 8, 256
        ),
    }
    for _name, (_env_name, _default, _minimum, _maximum) in _INCIDENT_ATTENTION_DEFAULTS.items():
        try:
            _value = int(os.getenv(_env_name, str(_default)))
        except (TypeError, ValueError):
            _value = _default
        locals()[_name] = max(_minimum, min(_maximum, _value))
    del _name, _env_name, _default, _minimum, _maximum, _value, _INCIDENT_ATTENTION_DEFAULTS
    LUXRIOT_INCIDENT_FOREGROUND_HARD_LIMIT = max(
        LUXRIOT_INCIDENT_FOREGROUND_LIMIT,
        LUXRIOT_INCIDENT_FOREGROUND_HARD_LIMIT,
    )
    LUXRIOT_INCIDENT_HOT_LIMIT = max(
        LUXRIOT_INCIDENT_FOREGROUND_HARD_LIMIT,
        LUXRIOT_INCIDENT_HOT_LIMIT,
    )
    LUXRIOT_INCIDENT_TRACKED_LIMIT = max(
        LUXRIOT_INCIDENT_HOT_LIMIT,
        LUXRIOT_INCIDENT_TRACKED_LIMIT,
    )
    INCIDENT_MAINTENANCE_ENABLED = _get_bool_env(
        'EVOSSEARCH_INCIDENT_MAINTENANCE_ENABLED',
        'true',
    )
    try:
        INCIDENT_MAINTENANCE_INTERVAL_SEC = float(
            os.getenv('EVOSSEARCH_INCIDENT_MAINTENANCE_INTERVAL_SEC', '15')
        )
    except (TypeError, ValueError):
        INCIDENT_MAINTENANCE_INTERVAL_SEC = 15.0
    INCIDENT_MAINTENANCE_INTERVAL_SEC = max(
        1.0,
        min(300.0, INCIDENT_MAINTENANCE_INTERVAL_SEC),
    )
    _L0_PROMPT_BUDGET_DEFAULTS = {
        'LUXRIOT_L0_CONTEXT_WINDOW_TOKENS': (
            'EVOSSEARCH_LUXRIOT_L0_CONTEXT_WINDOW_TOKENS', 16384, 8192, 262144
        ),
        'LUXRIOT_L0_TEXT_BUDGET_TOKENS': (
            'EVOSSEARCH_LUXRIOT_L0_TEXT_BUDGET_TOKENS', 5000, 1024, 65536
        ),
        'LUXRIOT_L0_VISION_BUDGET_TOKENS': (
            'EVOSSEARCH_LUXRIOT_L0_VISION_BUDGET_TOKENS', 5500, 512, 65536
        ),
        'LUXRIOT_L0_OUTPUT_BUDGET_TOKENS': (
            'EVOSSEARCH_LUXRIOT_L0_OUTPUT_BUDGET_TOKENS', 512, 256, 32768
        ),
        'LUXRIOT_L0_HEARTBEAT_OUTPUT_TOKENS': (
            'EVOSSEARCH_LUXRIOT_L0_HEARTBEAT_OUTPUT_TOKENS', 384, 256, 32768
        ),
        'LUXRIOT_L0_EVENT_OUTPUT_TOKENS': (
            'EVOSSEARCH_LUXRIOT_L0_EVENT_OUTPUT_TOKENS', 512, 256, 32768
        ),
        'LUXRIOT_L0_INCIDENT_BUDGET_TOKENS': (
            'EVOSSEARCH_LUXRIOT_L0_INCIDENT_BUDGET_TOKENS', 900, 128, 16384
        ),
        'LUXRIOT_L0_VISION_TOKENS_PER_IMAGE_ESTIMATE': (
            'EVOSSEARCH_LUXRIOT_L0_VISION_TOKENS_PER_IMAGE_ESTIMATE', 300, 64, 2048
        ),
    }
    for _name, (_env_name, _default, _minimum, _maximum) in _L0_PROMPT_BUDGET_DEFAULTS.items():
        try:
            _value = int(os.getenv(_env_name, str(_default)))
        except (TypeError, ValueError):
            _value = _default
        locals()[_name] = max(_minimum, min(_maximum, _value))
    del _name, _env_name, _default, _minimum, _maximum, _value, _L0_PROMPT_BUDGET_DEFAULTS
    LUXRIOT_ALERT_DERIVED_PROBES_ENABLED = _get_bool_env(
        'EVOSSEARCH_LUXRIOT_ALERT_DERIVED_PROBES_ENABLED',
        'False',
    )
    try:
        LUXRIOT_ALERT_DERIVED_PROBE_TTL_SEC = float(
            os.getenv('EVOSSEARCH_LUXRIOT_ALERT_DERIVED_PROBE_TTL_SEC', '300')
        )
    except (TypeError, ValueError):
        LUXRIOT_ALERT_DERIVED_PROBE_TTL_SEC = 300.0
    LUXRIOT_ALERT_DERIVED_PROBE_TTL_SEC = max(
        30.0,
        min(3600.0, LUXRIOT_ALERT_DERIVED_PROBE_TTL_SEC),
    )
    try:
        LUXRIOT_CAPTURE_REQUEST_TIMEOUT_SEC = float(
            os.getenv('EVOSSEARCH_LUXRIOT_CAPTURE_REQUEST_TIMEOUT_SEC', '5.0')
        )
    except (TypeError, ValueError):
        LUXRIOT_CAPTURE_REQUEST_TIMEOUT_SEC = 5.0
    LUXRIOT_CAPTURE_REQUEST_TIMEOUT_SEC = max(1.0, min(30.0, LUXRIOT_CAPTURE_REQUEST_TIMEOUT_SEC))
    try:
        LUXRIOT_LIVE_SEGMENT_READ_TIMEOUT_SEC = float(
            os.getenv('EVOSSEARCH_LUXRIOT_LIVE_SEGMENT_READ_TIMEOUT_SEC', '5.0')
        )
    except (TypeError, ValueError):
        LUXRIOT_LIVE_SEGMENT_READ_TIMEOUT_SEC = 5.0
    LUXRIOT_LIVE_SEGMENT_READ_TIMEOUT_SEC = max(1.0, min(30.0, LUXRIOT_LIVE_SEGMENT_READ_TIMEOUT_SEC))
    LUXRIOT_FFMPEG_HWACCEL = str(
        os.getenv('EVOSSEARCH_LUXRIOT_FFMPEG_HWACCEL', 'auto') or 'auto'
    ).strip().lower()
    if LUXRIOT_FFMPEG_HWACCEL not in {'auto', 'qsv', 'vaapi', 'software', 'none', 'off'}:
        LUXRIOT_FFMPEG_HWACCEL = 'auto'
    LUXRIOT_FFMPEG_INTEL_DEVICE = str(
        os.getenv('EVOSSEARCH_LUXRIOT_FFMPEG_INTEL_DEVICE', '') or ''
    ).strip()
    LUXRIOT_FFMPEG_QSV_DEVICE = str(
        os.getenv('EVOSSEARCH_LUXRIOT_FFMPEG_QSV_DEVICE', '') or ''
    ).strip()
    try:
        LUXRIOT_MEDIA_CONNECT_TIMEOUT_SEC = float(
            os.getenv('EVOSSEARCH_LUXRIOT_MEDIA_CONNECT_TIMEOUT_SEC', '3.0')
        )
    except (TypeError, ValueError):
        LUXRIOT_MEDIA_CONNECT_TIMEOUT_SEC = 3.0
    LUXRIOT_MEDIA_CONNECT_TIMEOUT_SEC = max(0.25, min(30.0, LUXRIOT_MEDIA_CONNECT_TIMEOUT_SEC))
    try:
        LUXRIOT_MEDIA_READ_TIMEOUT_SEC = float(
            os.getenv('EVOSSEARCH_LUXRIOT_MEDIA_READ_TIMEOUT_SEC', '8.0')
        )
    except (TypeError, ValueError):
        LUXRIOT_MEDIA_READ_TIMEOUT_SEC = 8.0
    LUXRIOT_MEDIA_READ_TIMEOUT_SEC = max(0.5, min(60.0, LUXRIOT_MEDIA_READ_TIMEOUT_SEC))
    try:
        LUXRIOT_ARCHIVE_PREPARE_TIMEOUT_SEC = float(
            os.getenv('EVOSSEARCH_LUXRIOT_ARCHIVE_PREPARE_TIMEOUT_SEC', '90.0')
        )
    except (TypeError, ValueError):
        LUXRIOT_ARCHIVE_PREPARE_TIMEOUT_SEC = 90.0
    LUXRIOT_ARCHIVE_PREPARE_TIMEOUT_SEC = max(
        15.0,
        min(180.0, LUXRIOT_ARCHIVE_PREPARE_TIMEOUT_SEC),
    )
    try:
        LUXRIOT_LIVE_MEDIA_MAX_SECONDS = float(
            os.getenv('EVOSSEARCH_LUXRIOT_LIVE_MEDIA_MAX_SECONDS', '120.0')
        )
    except (TypeError, ValueError):
        LUXRIOT_LIVE_MEDIA_MAX_SECONDS = 120.0
    LUXRIOT_LIVE_MEDIA_MAX_SECONDS = max(1.0, min(120.0, LUXRIOT_LIVE_MEDIA_MAX_SECONDS))
    try:
        LUXRIOT_LIVE_MEDIA_MAX_BYTES = int(
            os.getenv('EVOSSEARCH_LUXRIOT_LIVE_MEDIA_MAX_BYTES', str(256 * 1024 * 1024))
        )
    except (TypeError, ValueError):
        LUXRIOT_LIVE_MEDIA_MAX_BYTES = 256 * 1024 * 1024
    LUXRIOT_LIVE_MEDIA_MAX_BYTES = max(1024, min(256 * 1024 * 1024, LUXRIOT_LIVE_MEDIA_MAX_BYTES))
    try:
        LUXRIOT_ARCHIVE_MEDIA_MAX_SECONDS = float(
            os.getenv('EVOSSEARCH_LUXRIOT_ARCHIVE_MEDIA_MAX_SECONDS', '45.0')
        )
    except (TypeError, ValueError):
        LUXRIOT_ARCHIVE_MEDIA_MAX_SECONDS = 45.0
    LUXRIOT_ARCHIVE_MEDIA_MAX_SECONDS = max(1.0, min(300.0, LUXRIOT_ARCHIVE_MEDIA_MAX_SECONDS))
    try:
        LUXRIOT_ARCHIVE_MEDIA_MAX_BYTES = int(
            os.getenv('EVOSSEARCH_LUXRIOT_ARCHIVE_MEDIA_MAX_BYTES', str(128 * 1024 * 1024))
        )
    except (TypeError, ValueError):
        LUXRIOT_ARCHIVE_MEDIA_MAX_BYTES = 128 * 1024 * 1024
    LUXRIOT_ARCHIVE_MEDIA_MAX_BYTES = max(
        1024,
        min(512 * 1024 * 1024, LUXRIOT_ARCHIVE_MEDIA_MAX_BYTES),
    )
    # Per-second CV apex decider (capture_per_second_cv_apex_v2).
    try:
        LUXRIOT_CAPTURE_BURST_ZSCORE = float(os.getenv('EVOSSEARCH_LUXRIOT_CAPTURE_BURST_ZSCORE', '6.0'))
    except (TypeError, ValueError):
        LUXRIOT_CAPTURE_BURST_ZSCORE = 6.0
    LUXRIOT_CAPTURE_BURST_ZSCORE = max(1.0, min(50.0, LUXRIOT_CAPTURE_BURST_ZSCORE))
    try:
        LUXRIOT_CAPTURE_ACTIVITY_NOISE_FLOOR = float(
            os.getenv('EVOSSEARCH_LUXRIOT_CAPTURE_ACTIVITY_NOISE_FLOOR', '0.004')
        )
    except (TypeError, ValueError):
        LUXRIOT_CAPTURE_ACTIVITY_NOISE_FLOOR = 0.004
    LUXRIOT_CAPTURE_ACTIVITY_NOISE_FLOOR = max(0.0, min(0.25, LUXRIOT_CAPTURE_ACTIVITY_NOISE_FLOOR))
    LUXRIOT_CAPTURE_SELECTOR_ENABLED = _get_bool_env(
        'EVOSSEARCH_LUXRIOT_CAPTURE_SELECTOR_ENABLED',
        'True',
    )
    LUXRIOT_CAPTURE_SELECTOR_BIAS = os.getenv('EVOSSEARCH_LUXRIOT_CAPTURE_SELECTOR_BIAS', 'auto').strip().lower()
    if LUXRIOT_CAPTURE_SELECTOR_BIAS not in {'auto', 'action', 'clarity'}:
        LUXRIOT_CAPTURE_SELECTOR_BIAS = 'auto'
    LUXRIOT_VECTOR_SIGNALS_ENABLED = os.getenv('EVOSSEARCH_LUXRIOT_VECTOR_SIGNALS_ENABLED', 'true').strip().lower() not in {'0', 'false', 'no', 'off'}
    try:
        LUXRIOT_VECTOR_SIGNAL_PROBE_LIMIT = int(os.getenv('EVOSSEARCH_LUXRIOT_VECTOR_SIGNAL_PROBE_LIMIT', '6'))
    except (TypeError, ValueError):
        LUXRIOT_VECTOR_SIGNAL_PROBE_LIMIT = 6
    LUXRIOT_VECTOR_SIGNAL_PROBE_LIMIT = max(0, min(16, LUXRIOT_VECTOR_SIGNAL_PROBE_LIMIT))
    try:
        LUXRIOT_VECTOR_SIGNAL_TOP_HITS = int(os.getenv('EVOSSEARCH_LUXRIOT_VECTOR_SIGNAL_TOP_HITS', '2'))
    except (TypeError, ValueError):
        LUXRIOT_VECTOR_SIGNAL_TOP_HITS = 2
    LUXRIOT_VECTOR_SIGNAL_TOP_HITS = max(1, min(5, LUXRIOT_VECTOR_SIGNAL_TOP_HITS))
    # Road geometry is an opt-in domain ray. Running it for every ordinary
    # room/coastline channel adds seconds of CPU work to every L0 batch and can
    # delay the visual model without adding meaningful evidence.
    LUXRIOT_ROAD_CV_BATCH_SIGNALS = os.getenv('EVOSSEARCH_LUXRIOT_ROAD_CV_BATCH_SIGNALS', 'false').strip().lower() not in {'0', 'false', 'no', 'off'}
    try:
        LUXRIOT_ROAD_CV_BATCH_MAX_FRAMES = int(os.getenv('EVOSSEARCH_LUXRIOT_ROAD_CV_BATCH_MAX_FRAMES', '24'))
    except (TypeError, ValueError):
        LUXRIOT_ROAD_CV_BATCH_MAX_FRAMES = 24
    LUXRIOT_ROAD_CV_BATCH_MAX_FRAMES = max(4, min(48, LUXRIOT_ROAD_CV_BATCH_MAX_FRAMES))
    try:
        LUXRIOT_ROAD_CV_BATCH_MAX_EDGE = int(os.getenv('EVOSSEARCH_LUXRIOT_ROAD_CV_BATCH_MAX_EDGE', '240'))
    except (TypeError, ValueError):
        LUXRIOT_ROAD_CV_BATCH_MAX_EDGE = 240
    LUXRIOT_ROAD_CV_BATCH_MAX_EDGE = max(96, min(480, LUXRIOT_ROAD_CV_BATCH_MAX_EDGE))
    try:
        LUXRIOT_ROAD_SCENE_CALIBRATION_SAMPLES = int(
            os.getenv('EVOSSEARCH_LUXRIOT_ROAD_SCENE_CALIBRATION_SAMPLES', '8')
        )
    except (TypeError, ValueError):
        LUXRIOT_ROAD_SCENE_CALIBRATION_SAMPLES = 8
    LUXRIOT_ROAD_SCENE_CALIBRATION_SAMPLES = max(4, min(64, LUXRIOT_ROAD_SCENE_CALIBRATION_SAMPLES))
    # One VLM request must never exceed the bounded L0 delivery contract.
    # Keep this list in sync with the React stream-settings selector.  The
    # explicit default must not depend on ordering: operators may deliberately
    # choose a smaller batch to trade visual coverage for lower alert latency.
    LUXRIOT_BATCH_SIZES = (4, 8, 12, 16)
    LUXRIOT_DEFAULT_BATCH_SIZE = 12
    try:
        LUXRIOT_SUMMARY_RETENTION_DAYS = float(
            os.getenv('EVOSSEARCH_LUXRIOT_SUMMARY_RETENTION_DAYS', '7')
        )
    except (TypeError, ValueError):
        LUXRIOT_SUMMARY_RETENTION_DAYS = 7.0
    LUXRIOT_SUMMARY_RETENTION_DAYS = max(0.0, LUXRIOT_SUMMARY_RETENTION_DAYS)
    try:
        LUXRIOT_ROLLUP_RETENTION_DAYS = float(
            os.getenv(
                'EVOSSEARCH_LUXRIOT_ROLLUP_RETENTION_DAYS',
                str(ARCHIVE_ROW_RETENTION_DAYS),
            )
        )
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_RETENTION_DAYS = float(ARCHIVE_ROW_RETENTION_DAYS)
    LUXRIOT_ROLLUP_RETENTION_DAYS = max(0.0, LUXRIOT_ROLLUP_RETENTION_DAYS)
    _SUMMARY_DEFAULT_BATCH = LUXRIOT_DEFAULT_BATCH_SIZE
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
        LUXRIOT_SUMMARY_STATE_HOT_LIMIT = int(
            os.getenv('EVOSSEARCH_LUXRIOT_SUMMARY_STATE_HOT_LIMIT', '2160')
        )
    except (TypeError, ValueError):
        LUXRIOT_SUMMARY_STATE_HOT_LIMIT = 2160
    LUXRIOT_SUMMARY_STATE_HOT_LIMIT = max(240, min(10000, LUXRIOT_SUMMARY_STATE_HOT_LIMIT))
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
        LUXRIOT_SUMMARY_QUEUE_MAX_BATCHES = int(
            os.getenv('EVOSSEARCH_LUXRIOT_SUMMARY_QUEUE_MAX_BATCHES', '2')
        )
    except (TypeError, ValueError):
        LUXRIOT_SUMMARY_QUEUE_MAX_BATCHES = 2
    LUXRIOT_SUMMARY_QUEUE_MAX_BATCHES = max(1, min(12, LUXRIOT_SUMMARY_QUEUE_MAX_BATCHES))
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
    try:
        LUXRIOT_RECENT_FRAME_MAX_AGE_SEC = float(os.getenv('EVOSSEARCH_LUXRIOT_RECENT_FRAME_MAX_AGE_SEC', '45'))
    except (TypeError, ValueError):
        LUXRIOT_RECENT_FRAME_MAX_AGE_SEC = 45.0
    LUXRIOT_RECENT_FRAME_MAX_AGE_SEC = max(3.0, min(300.0, LUXRIOT_RECENT_FRAME_MAX_AGE_SEC))
    try:
        LUXRIOT_FROZEN_FRAME_MAX_SEC = float(os.getenv('EVOSSEARCH_LUXRIOT_FROZEN_FRAME_MAX_SEC', '20'))
    except (TypeError, ValueError):
        LUXRIOT_FROZEN_FRAME_MAX_SEC = 20.0
    LUXRIOT_FROZEN_FRAME_MAX_SEC = max(5.0, min(300.0, LUXRIOT_FROZEN_FRAME_MAX_SEC))
    try:
        LUXRIOT_FROZEN_FRAME_MIN_COUNT = int(os.getenv('EVOSSEARCH_LUXRIOT_FROZEN_FRAME_MIN_COUNT', '3'))
    except (TypeError, ValueError):
        LUXRIOT_FROZEN_FRAME_MIN_COUNT = 3
    LUXRIOT_FROZEN_FRAME_MIN_COUNT = max(2, min(120, LUXRIOT_FROZEN_FRAME_MIN_COUNT))
    LUXRIOT_AUTO_BOOKMARKS = _get_bool_env('EVOSSEARCH_LUXRIOT_AUTO_BOOKMARKS', 'False')
    OFFLINE_VIDEO_ENABLED = _get_bool_env('EVOSSEARCH_OFFLINE_VIDEO_ENABLED', 'False')
    PROBE_SNAP_ENABLED = _get_bool_env('EVOSSEARCH_PROBE_SNAP_ENABLED', 'False')
    INDEXED_FOLDER_ENABLED = _get_bool_env('EVOSSEARCH_INDEXED_FOLDER_ENABLED', 'False')
    ROAD_CV_ENABLED = _get_bool_env('EVOSSEARCH_ROAD_CV_ENABLED', 'False')
    ROAD_CV_SCENE_CARDS = os.getenv('EVOSSEARCH_ROAD_CV_SCENE_CARDS', '').strip()
    try:
        ROAD_CV_MAX_EDGE = int(os.getenv('EVOSSEARCH_ROAD_CV_MAX_EDGE', '360'))
    except (TypeError, ValueError):
        ROAD_CV_MAX_EDGE = 360
    ROAD_CV_MAX_EDGE = max(96, min(1280, ROAD_CV_MAX_EDGE))
    try:
        ROAD_CV_MIN_MOTION_PX = float(os.getenv('EVOSSEARCH_ROAD_CV_MIN_MOTION_PX', '0.7'))
    except (TypeError, ValueError):
        ROAD_CV_MIN_MOTION_PX = 0.7
    ROAD_CV_MIN_MOTION_PX = max(0.05, min(20.0, ROAD_CV_MIN_MOTION_PX))
    try:
        ROAD_CV_ACTIVE_RATIO_FLOOR = float(os.getenv('EVOSSEARCH_ROAD_CV_ACTIVE_RATIO_FLOOR', '0.012'))
    except (TypeError, ValueError):
        ROAD_CV_ACTIVE_RATIO_FLOOR = 0.012
    ROAD_CV_ACTIVE_RATIO_FLOOR = max(0.001, min(0.5, ROAD_CV_ACTIVE_RATIO_FLOOR))
    try:
        ROAD_CV_WRONG_WAY_ALIGNMENT = float(os.getenv('EVOSSEARCH_ROAD_CV_WRONG_WAY_ALIGNMENT', '-0.45'))
    except (TypeError, ValueError):
        ROAD_CV_WRONG_WAY_ALIGNMENT = -0.45
    ROAD_CV_WRONG_WAY_ALIGNMENT = max(-1.0, min(0.0, ROAD_CV_WRONG_WAY_ALIGNMENT))
    try:
        LUXRIOT_BOOKMARK_COOLDOWN_SEC = float(os.getenv('EVOSSEARCH_LUXRIOT_BOOKMARK_COOLDOWN_SEC', '60.0'))
    except (TypeError, ValueError):
        LUXRIOT_BOOKMARK_COOLDOWN_SEC = 60.0
    LUXRIOT_BOOKMARK_COOLDOWN_SEC = max(0.0, LUXRIOT_BOOKMARK_COOLDOWN_SEC)
    try:
        LUXRIOT_ALERT_DEDUPE_WINDOW_SEC = float(
            os.getenv('EVOSSEARCH_LUXRIOT_ALERT_DEDUPE_WINDOW_SEC', '600')
        )
    except (TypeError, ValueError):
        LUXRIOT_ALERT_DEDUPE_WINDOW_SEC = 600.0
    LUXRIOT_ALERT_DEDUPE_WINDOW_SEC = max(0.0, min(86400.0, LUXRIOT_ALERT_DEDUPE_WINDOW_SEC))
    try:
        LUXRIOT_ALERTS_MAX_PER_BATCH = int(os.getenv('EVOSSEARCH_LUXRIOT_ALERTS_MAX_PER_BATCH', '8'))
    except (TypeError, ValueError):
        LUXRIOT_ALERTS_MAX_PER_BATCH = 8
    LUXRIOT_ALERTS_MAX_PER_BATCH = max(1, min(32, LUXRIOT_ALERTS_MAX_PER_BATCH))
    LUXRIOT_STATE_TRANSITIONS_ENABLED = _get_bool_env('EVOSSEARCH_LUXRIOT_STATE_TRANSITIONS_ENABLED', 'True')
    try:
        LUXRIOT_STATE_TRANSITION_CONFIRM_BATCHES = int(
            os.getenv('EVOSSEARCH_LUXRIOT_STATE_TRANSITION_CONFIRM_BATCHES', '2')
        )
    except (TypeError, ValueError):
        LUXRIOT_STATE_TRANSITION_CONFIRM_BATCHES = 2
    LUXRIOT_STATE_TRANSITION_CONFIRM_BATCHES = max(1, min(6, LUXRIOT_STATE_TRANSITION_CONFIRM_BATCHES))
    LUXRIOT_STATE_TRANSITION_ALERT_EVENTS = _get_bool_env(
        'EVOSSEARCH_LUXRIOT_STATE_TRANSITION_ALERT_EVENTS',
        'True',
    )
    LUXRIOT_SYSTEM_PROMPT_DEFAULT = os.getenv('EVOSSEARCH_LUXRIOT_SYSTEM_PROMPT_DEFAULT', '').strip()
    LUXRIOT_ALERT_POLICY_PROMPT = os.getenv('EVOSSEARCH_LUXRIOT_ALERT_POLICY_PROMPT', '').strip()
    LUXRIOT_ALERTS_JSON_PROMPT = os.getenv('EVOSSEARCH_LUXRIOT_ALERTS_JSON_PROMPT', '').strip()
    LUXRIOT_SEVERITY_MAP = {
        'info': os.getenv('EVOSSEARCH_LUXRIOT_SEV_INFO', 'info').lower(),
        'low': os.getenv('EVOSSEARCH_LUXRIOT_SEV_LOW', 'low').lower(),
        'normal': os.getenv('EVOSSEARCH_LUXRIOT_SEV_NORMAL', 'normal').lower(),
        'high': os.getenv('EVOSSEARCH_LUXRIOT_SEV_HIGH', 'high').lower(),
        'critical': os.getenv('EVOSSEARCH_LUXRIOT_SEV_CRITICAL', 'critical').lower(),
    }
    LUXRIOT_ROLLUP_L1_LLM_ENABLED = _get_bool_env('EVOSSEARCH_LUXRIOT_ROLLUP_L1_LLM_ENABLED', 'True')
    LUXRIOT_ROLLUP_SCHEDULER_ENABLED = _get_bool_env(
        'EVOSSEARCH_LUXRIOT_ROLLUP_SCHEDULER_ENABLED',
        'True',
    )
    try:
        LUXRIOT_ROLLUP_SCHEDULER_INITIAL_DELAY_SEC = float(
            os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_SCHEDULER_INITIAL_DELAY_SEC', '30')
        )
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_SCHEDULER_INITIAL_DELAY_SEC = 30.0
    LUXRIOT_ROLLUP_SCHEDULER_INITIAL_DELAY_SEC = max(
        1.0,
        min(600.0, LUXRIOT_ROLLUP_SCHEDULER_INITIAL_DELAY_SEC),
    )
    try:
        LUXRIOT_ROLLUP_SCHEDULER_SPACING_SEC = float(
            os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_SCHEDULER_SPACING_SEC', '5')
        )
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_SCHEDULER_SPACING_SEC = 5.0
    LUXRIOT_ROLLUP_SCHEDULER_SPACING_SEC = max(
        1.0,
        min(300.0, LUXRIOT_ROLLUP_SCHEDULER_SPACING_SEC),
    )
    try:
        LUXRIOT_ROLLUP_L1_SETTLE_DELAY_SEC = float(
            os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_L1_SETTLE_DELAY_SEC', '30')
        )
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_L1_SETTLE_DELAY_SEC = 30.0
    LUXRIOT_ROLLUP_L1_SETTLE_DELAY_SEC = max(
        0.0,
        min(1800.0, LUXRIOT_ROLLUP_L1_SETTLE_DELAY_SEC),
    )
    try:
        LUXRIOT_ROLLUP_L2_SETTLE_DELAY_SEC = float(
            os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_L2_SETTLE_DELAY_SEC', '120')
        )
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_L2_SETTLE_DELAY_SEC = 120.0
    LUXRIOT_ROLLUP_L2_SETTLE_DELAY_SEC = max(
        0.0,
        min(1800.0, LUXRIOT_ROLLUP_L2_SETTLE_DELAY_SEC),
    )
    try:
        LUXRIOT_ROLLUP_L3_SETTLE_DELAY_SEC = float(
            os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_L3_SETTLE_DELAY_SEC', '300')
        )
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_L3_SETTLE_DELAY_SEC = 300.0
    LUXRIOT_ROLLUP_L3_SETTLE_DELAY_SEC = max(
        0.0,
        min(1800.0, LUXRIOT_ROLLUP_L3_SETTLE_DELAY_SEC),
    )
    try:
        LUXRIOT_ROLLUP_SCHEDULER_BACKFILL_WINDOWS = int(
            os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_SCHEDULER_BACKFILL_WINDOWS', '2')
        )
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_SCHEDULER_BACKFILL_WINDOWS = 2
    LUXRIOT_ROLLUP_SCHEDULER_BACKFILL_WINDOWS = max(
        1,
        min(12, LUXRIOT_ROLLUP_SCHEDULER_BACKFILL_WINDOWS),
    )
    try:
        LUXRIOT_ROLLUP_SCHEDULER_MAX_DEFERRAL_WINDOWS = float(
            os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_SCHEDULER_MAX_DEFERRAL_WINDOWS', '2')
        )
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_SCHEDULER_MAX_DEFERRAL_WINDOWS = 2.0
    LUXRIOT_ROLLUP_SCHEDULER_MAX_DEFERRAL_WINDOWS = max(
        0.0,
        min(10.0, LUXRIOT_ROLLUP_SCHEDULER_MAX_DEFERRAL_WINDOWS),
    )
    try:
        LUXRIOT_ROLLUP_SCHEDULER_MAX_DEFERRAL_SEC = float(
            os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_SCHEDULER_MAX_DEFERRAL_SEC', '180')
        )
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_SCHEDULER_MAX_DEFERRAL_SEC = 180.0
    LUXRIOT_ROLLUP_SCHEDULER_MAX_DEFERRAL_SEC = max(
        0.0,
        min(3600.0, LUXRIOT_ROLLUP_SCHEDULER_MAX_DEFERRAL_SEC),
    )
    try:
        LUXRIOT_ROLLUP_BACKFILL_SPACING_SEC = float(
            os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_BACKFILL_SPACING_SEC', '10')
        )
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_BACKFILL_SPACING_SEC = 10.0
    LUXRIOT_ROLLUP_BACKFILL_SPACING_SEC = max(
        1.0,
        min(300.0, LUXRIOT_ROLLUP_BACKFILL_SPACING_SEC),
    )
    try:
        LUXRIOT_ROLLUP_BACKFILL_MAX_ATTEMPTS = int(
            os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_BACKFILL_MAX_ATTEMPTS', '3')
        )
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_BACKFILL_MAX_ATTEMPTS = 3
    LUXRIOT_ROLLUP_BACKFILL_MAX_ATTEMPTS = max(
        1,
        min(10, LUXRIOT_ROLLUP_BACKFILL_MAX_ATTEMPTS),
    )
    try:
        LUXRIOT_ROLLUP_BACKFILL_ESTIMATE_SEC = float(
            os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_BACKFILL_ESTIMATE_SEC', '45')
        )
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_BACKFILL_ESTIMATE_SEC = 45.0
    LUXRIOT_ROLLUP_BACKFILL_ESTIMATE_SEC = max(
        1.0,
        min(900.0, LUXRIOT_ROLLUP_BACKFILL_ESTIMATE_SEC),
    )
    try:
        LUXRIOT_ROLLUP_L1_WINDOW_SEC = int(os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_L1_WINDOW_SEC', '900'))
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_L1_WINDOW_SEC = 900
    LUXRIOT_ROLLUP_L1_WINDOW_SEC = max(300, LUXRIOT_ROLLUP_L1_WINDOW_SEC)
    try:
        LUXRIOT_ROLLUP_L2_WINDOW_SEC = int(os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_L2_WINDOW_SEC', '3600'))
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_L2_WINDOW_SEC = 3600
    LUXRIOT_ROLLUP_L2_WINDOW_SEC = max(900, LUXRIOT_ROLLUP_L2_WINDOW_SEC)
    try:
        LUXRIOT_ROLLUP_L3_WINDOW_SEC = int(os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_L3_WINDOW_SEC', '28800'))
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_L3_WINDOW_SEC = 28800
    LUXRIOT_ROLLUP_L3_WINDOW_SEC = max(1800, LUXRIOT_ROLLUP_L3_WINDOW_SEC)
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
    # Operator-defined channel groups for the Probes board.  Luxriot exposes no
    # group concept, so this is EVA-side organisation state.
    PROBE_CHANNEL_GROUPS_FILE = (
        os.getenv('EVOSSEARCH_PROBE_CHANNEL_GROUPS_FILE', 'probe_channel_groups.json').strip()
        or 'probe_channel_groups.json'
    )
    LUXRIOT_ROLLUP_TIME_ONLY = _get_bool_env('EVOSSEARCH_LUXRIOT_ROLLUP_TIME_ONLY', 'True')
    LUXRIOT_ROLLUP_L1_MODEL = os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_L1_MODEL', '').strip()
    LUXRIOT_ROLLUP_L1_SYSTEM_PROMPT = os.getenv(
        'EVOSSEARCH_LUXRIOT_ROLLUP_L1_SYSTEM_PROMPT',
        (
            "You execute short-horizon episodic consolidation (L1) within EVA's visual-semantic intellectual core. "
            "EVA may operate from a home installation to city-scale infrastructure; deployment rules and alert criteria "
            "are supplied separately. Your output becomes short-lived system memory and may affect continuity and future "
            "attention, so preserve evidence, uncertainty, provenance, rare deviations, and coverage gaps. Turn grounded "
            "L0 observations into a readable 15-minute episode account: what persisted, changed, resolved, or remains "
            "uncertain. Maintain unresolved episodes and a short-lived watchlist, but do not establish durable routine, "
            "suppress alerts, or invent new visual facts. "
            "Use the mandatory EVA operator rollup contract appended by the backend."
        ),
    ).strip()
    LUXRIOT_ROLLUP_L2_SYSTEM_PROMPT = os.getenv(
        'EVOSSEARCH_LUXRIOT_ROLLUP_L2_SYSTEM_PROMPT',
        (
            "You execute medium-horizon routine and recurrence consolidation (L2) within EVA's visual-semantic "
            "intellectual core. Convert L1 episodes into a grounded hour-scale account of recurrence, routine shifts, "
            "alerts and outcomes, unresolved exceptions, and coverage interruptions. Your output may regulate the future "
            "channel baseline, so distinguish repeated observation from certainty, preserve isolated important events, "
            "and express baseline changes as grounded candidates rather than visual facts. Do not concatenate lower-level "
            "summaries, invent new observations, or let routine suppress visible hazards and operator criteria. "
            "Use the mandatory EVA operator rollup contract appended by the backend."
        ),
    ).strip()
    LUXRIOT_ROLLUP_L3_SYSTEM_PROMPT = os.getenv(
        'EVOSSEARCH_LUXRIOT_ROLLUP_L3_SYSTEM_PROMPT',
        (
            "You execute slow-horizon audit and regulatory memory consolidation (L3) within EVA's visual-semantic "
            "intellectual core. Build a grounded eight-hour account of durable routine, repeated or changing behavior, "
            "unresolved incidents, alert meaning, exceptions, and coverage quality. Reconcile the hierarchy audit and "
            "bounded L0 drills without treating samples as complete coverage. Operator false-positive annotations are "
            "privileged review feedback: analyze recurring failure modes separately, while leaving unreviewed alerts "
            "unclassified. Propose durable memory and alert tuning conservatively; never silently suppress general hazards "
            "or operator criteria, and never invent new visual facts. This pass is review/proposal-only: it must not "
            "change probes, thresholds, alert policy, or live sampling automatically. "
            "Use the mandatory EVA operator rollup contract appended by the backend."
        ),
    ).strip()
    LUXRIOT_ROLLUP_LLM_LEVELS = os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_LLM_LEVELS', 'L1,L2,L3').strip() or 'L1,L2,L3'
    # Rollups are text-only and need a different completion budget from live
    # visual L0.  In particular, a complete eight-hour L3 report plus its
    # machine-memory block does not reliably fit in the small L0 budget used on
    # edge GPUs.  Keep the levels independently configurable so increasing L3
    # quality does not increase every live VLM request.
    _ROLLUP_MAX_TOKEN_DEFAULTS = {
        'L1': 768,
        'L2': 1024,
        'L3': 2048,
    }
    for _rollup_level, _rollup_default in _ROLLUP_MAX_TOKEN_DEFAULTS.items():
        try:
            _rollup_value = int(
                os.getenv(
                    f'EVOSSEARCH_LUXRIOT_ROLLUP_{_rollup_level}_MAX_TOKENS',
                    str(_rollup_default),
                )
            )
        except (TypeError, ValueError):
            _rollup_value = _rollup_default
        locals()[f'LUXRIOT_ROLLUP_{_rollup_level}_MAX_TOKENS'] = min(
            32768,
            max(256, _rollup_value),
        )
    del _rollup_level, _rollup_default, _rollup_value, _ROLLUP_MAX_TOKEN_DEFAULTS
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
        LUXRIOT_ROLLUP_CONTEXT_LIMIT_TOKENS = int(
            os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_CONTEXT_LIMIT_TOKENS', '32768')
        )
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_CONTEXT_LIMIT_TOKENS = 32768
    LUXRIOT_ROLLUP_CONTEXT_LIMIT_TOKENS = min(
        262144,
        max(8192, LUXRIOT_ROLLUP_CONTEXT_LIMIT_TOKENS),
    )
    try:
        LUXRIOT_ROLLUP_LLM_MAX_NEW_PER_CALL = int(
            os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_LLM_MAX_NEW_PER_CALL', str(LUXRIOT_ROLLUP_L1_MAX_NEW_PER_CALL))
        )
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_LLM_MAX_NEW_PER_CALL = LUXRIOT_ROLLUP_L1_MAX_NEW_PER_CALL
    if LUXRIOT_ROLLUP_LLM_MAX_NEW_PER_CALL < 1:
        LUXRIOT_ROLLUP_LLM_MAX_NEW_PER_CALL = 1
    LUXRIOT_ROLLUP_LLM_MODEL = os.getenv(
        'EVOSSEARCH_LUXRIOT_ROLLUP_LLM_MODEL',
        LUXRIOT_ROLLUP_L1_MODEL or LM_AGENT_PROFILE_ID,
    ).strip()
    LUXRIOT_ROLLUP_LLM_SYSTEM_PROMPT = os.getenv(
        'EVOSSEARCH_LUXRIOT_ROLLUP_LLM_SYSTEM_PROMPT',
        LUXRIOT_ROLLUP_L1_SYSTEM_PROMPT,
    ).strip()
    LUXRIOT_ROLLUP_L3_DEEP_ENABLED = _get_bool_env(
        'EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_ENABLED',
        'False',
    )
    LUXRIOT_ROLLUP_L3_DEEP_BASE_URL = os.getenv(
        'EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_BASE_URL',
        '',
    ).strip().rstrip('/')
    LUXRIOT_ROLLUP_L3_DEEP_MODEL = os.getenv(
        'EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_MODEL',
        '',
    ).strip()
    LUXRIOT_ROLLUP_L3_DEEP_API_KEY = os.getenv(
        'EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_API_KEY',
        '',
    ).strip()
    try:
        LUXRIOT_ROLLUP_L3_DEEP_CONNECT_TIMEOUT_SEC = float(
            os.getenv(
                'EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_CONNECT_TIMEOUT_SEC',
                '5',
            )
        )
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_L3_DEEP_CONNECT_TIMEOUT_SEC = 5.0
    LUXRIOT_ROLLUP_L3_DEEP_CONNECT_TIMEOUT_SEC = min(
        60.0,
        max(0.25, LUXRIOT_ROLLUP_L3_DEEP_CONNECT_TIMEOUT_SEC),
    )
    try:
        LUXRIOT_ROLLUP_L3_DEEP_READ_TIMEOUT_SEC = float(
            os.getenv(
                'EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_READ_TIMEOUT_SEC',
                '600',
            )
        )
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_L3_DEEP_READ_TIMEOUT_SEC = 600.0
    LUXRIOT_ROLLUP_L3_DEEP_READ_TIMEOUT_SEC = min(
        3600.0,
        max(1.0, LUXRIOT_ROLLUP_L3_DEEP_READ_TIMEOUT_SEC),
    )
    try:
        LUXRIOT_ROLLUP_L3_DEEP_MAX_TOKENS = int(
            os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_MAX_TOKENS', '3072')
        )
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_L3_DEEP_MAX_TOKENS = 3072
    LUXRIOT_ROLLUP_L3_DEEP_MAX_TOKENS = min(
        32768,
        max(128, LUXRIOT_ROLLUP_L3_DEEP_MAX_TOKENS),
    )
    try:
        LUXRIOT_ROLLUP_L3_DEEP_TEMPERATURE = float(
            os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_TEMPERATURE', '0.1')
        )
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_L3_DEEP_TEMPERATURE = 0.1
    LUXRIOT_ROLLUP_L3_DEEP_TEMPERATURE = min(
        2.0,
        max(0.0, LUXRIOT_ROLLUP_L3_DEEP_TEMPERATURE),
    )
    try:
        LUXRIOT_ROLLUP_L3_DEEP_QUEUE_CAPACITY = int(
            os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_QUEUE_CAPACITY', '64')
        )
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_L3_DEEP_QUEUE_CAPACITY = 64
    LUXRIOT_ROLLUP_L3_DEEP_QUEUE_CAPACITY = min(
        256,
        max(1, LUXRIOT_ROLLUP_L3_DEEP_QUEUE_CAPACITY),
    )
    try:
        LUXRIOT_ROLLUP_L3_DEEP_MAX_ATTEMPTS = int(
            os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_MAX_ATTEMPTS', '3')
        )
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_L3_DEEP_MAX_ATTEMPTS = 3
    LUXRIOT_ROLLUP_L3_DEEP_MAX_ATTEMPTS = min(
        10,
        max(1, LUXRIOT_ROLLUP_L3_DEEP_MAX_ATTEMPTS),
    )
    try:
        LUXRIOT_ROLLUP_L3_DEEP_BACKOFF_INITIAL_SEC = float(
            os.getenv(
                'EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_BACKOFF_INITIAL_SEC',
                '30',
            )
        )
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_L3_DEEP_BACKOFF_INITIAL_SEC = 30.0
    LUXRIOT_ROLLUP_L3_DEEP_BACKOFF_INITIAL_SEC = min(
        3600.0,
        max(1.0, LUXRIOT_ROLLUP_L3_DEEP_BACKOFF_INITIAL_SEC),
    )
    try:
        LUXRIOT_ROLLUP_L3_DEEP_BACKOFF_MAX_SEC = float(
            os.getenv(
                'EVOSSEARCH_LUXRIOT_ROLLUP_L3_DEEP_BACKOFF_MAX_SEC',
                '900',
            )
        )
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_L3_DEEP_BACKOFF_MAX_SEC = 900.0
    LUXRIOT_ROLLUP_L3_DEEP_BACKOFF_MAX_SEC = max(
        LUXRIOT_ROLLUP_L3_DEEP_BACKOFF_INITIAL_SEC,
        min(3600.0, LUXRIOT_ROLLUP_L3_DEEP_BACKOFF_MAX_SEC),
    )
    LUXRIOT_ROLLUP_L3_QUIET_WINDOW_ENABLED = _get_bool_env(
        'EVOSSEARCH_LUXRIOT_ROLLUP_L3_QUIET_WINDOW_ENABLED',
        'False',
    )
    LUXRIOT_ROLLUP_L3_QUIET_WINDOW_TIMEZONE = os.getenv(
        'EVOSSEARCH_LUXRIOT_ROLLUP_L3_QUIET_WINDOW_TIMEZONE',
        os.getenv('EVOSSEARCH_SITE_TIMEZONE', 'UTC'),
    ).strip() or 'UTC'
    LUXRIOT_ROLLUP_L3_QUIET_WINDOW_START = os.getenv(
        'EVOSSEARCH_LUXRIOT_ROLLUP_L3_QUIET_WINDOW_START',
        '01:00',
    ).strip() or '01:00'
    LUXRIOT_ROLLUP_L3_QUIET_WINDOW_END = os.getenv(
        'EVOSSEARCH_LUXRIOT_ROLLUP_L3_QUIET_WINDOW_END',
        '05:00',
    ).strip() or '05:00'
    LUXRIOT_ROLLUP_L3_QUIET_WINDOW_DAYS = os.getenv(
        'EVOSSEARCH_LUXRIOT_ROLLUP_L3_QUIET_WINDOW_DAYS',
        'mon,tue,wed,thu,fri,sat,sun',
    ).strip() or 'mon,tue,wed,thu,fri,sat,sun'
    try:
        LUXRIOT_ROLLUP_L3_QUIET_MAX_DEFERRAL_SEC = float(
            os.getenv(
                'EVOSSEARCH_LUXRIOT_ROLLUP_L3_QUIET_MAX_DEFERRAL_SEC',
                '86400',
            )
        )
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_L3_QUIET_MAX_DEFERRAL_SEC = 86400.0
    LUXRIOT_ROLLUP_L3_QUIET_MAX_DEFERRAL_SEC = min(
        604800.0,
        max(60.0, LUXRIOT_ROLLUP_L3_QUIET_MAX_DEFERRAL_SEC),
    )
    try:
        LUXRIOT_ROLLUP_L3_QUIET_POLL_SEC = float(
            os.getenv('EVOSSEARCH_LUXRIOT_ROLLUP_L3_QUIET_POLL_SEC', '60')
        )
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_L3_QUIET_POLL_SEC = 60.0
    LUXRIOT_ROLLUP_L3_QUIET_POLL_SEC = min(
        3600.0,
        max(5.0, LUXRIOT_ROLLUP_L3_QUIET_POLL_SEC),
    )
    try:
        LUXRIOT_ROLLUP_L3_QUIET_MAX_ACTIVITY_X = float(
            os.getenv(
                'EVOSSEARCH_LUXRIOT_ROLLUP_L3_QUIET_MAX_ACTIVITY_X',
                '1.5',
            )
        )
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_L3_QUIET_MAX_ACTIVITY_X = 1.5
    LUXRIOT_ROLLUP_L3_QUIET_MAX_ACTIVITY_X = min(
        1000.0,
        max(0.0, LUXRIOT_ROLLUP_L3_QUIET_MAX_ACTIVITY_X),
    )
    try:
        LUXRIOT_ROLLUP_L3_QUIET_ALERT_LOOKBACK_SEC = float(
            os.getenv(
                'EVOSSEARCH_LUXRIOT_ROLLUP_L3_QUIET_ALERT_LOOKBACK_SEC',
                '900',
            )
        )
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_L3_QUIET_ALERT_LOOKBACK_SEC = 900.0
    LUXRIOT_ROLLUP_L3_QUIET_ALERT_LOOKBACK_SEC = min(
        86400.0,
        max(0.0, LUXRIOT_ROLLUP_L3_QUIET_ALERT_LOOKBACK_SEC),
    )
    try:
        LUXRIOT_ROLLUP_L3_QUIET_MAX_L0_DEBT = float(
            os.getenv(
                'EVOSSEARCH_LUXRIOT_ROLLUP_L3_QUIET_MAX_L0_DEBT',
                '0.75',
            )
        )
    except (TypeError, ValueError):
        LUXRIOT_ROLLUP_L3_QUIET_MAX_L0_DEBT = 0.75
    LUXRIOT_ROLLUP_L3_QUIET_MAX_L0_DEBT = min(
        10.0,
        max(0.0, LUXRIOT_ROLLUP_L3_QUIET_MAX_L0_DEBT),
    )

    # Probe / CLIP monitoring
    _PROBE_SIGLIP2_DEFAULTS = "siglip2" in CLIP_MODEL.lower()
    try:
        PROBE_POS_FLOOR_DEFAULT = float(
            os.getenv(
                'EVOSSEARCH_PROBE_POS_FLOOR_DEFAULT',
                '0.05' if _PROBE_SIGLIP2_DEFAULTS else '0.28',
            )
        )
    except (TypeError, ValueError):
        PROBE_POS_FLOOR_DEFAULT = 0.05 if _PROBE_SIGLIP2_DEFAULTS else 0.28
    PROBE_POS_FLOOR_DEFAULT = min(1.0, max(-1.0, PROBE_POS_FLOOR_DEFAULT))
    try:
        PROBE_MARGIN_DEFAULT = float(
            os.getenv(
                'EVOSSEARCH_PROBE_MARGIN_DEFAULT',
                '0.02' if _PROBE_SIGLIP2_DEFAULTS else '0.08',
            )
        )
    except (TypeError, ValueError):
        PROBE_MARGIN_DEFAULT = 0.02 if _PROBE_SIGLIP2_DEFAULTS else 0.08
    PROBE_MARGIN_DEFAULT = min(2.0, max(0.0, PROBE_MARGIN_DEFAULT))
    try:
        PROBE_CAPTURE_WARMUP_SEC = float(
            os.getenv('EVOSSEARCH_PROBE_CAPTURE_WARMUP_SEC', '2.5')
        )
    except (TypeError, ValueError):
        PROBE_CAPTURE_WARMUP_SEC = 2.5
    PROBE_CAPTURE_WARMUP_SEC = min(10.0, max(0.0, PROBE_CAPTURE_WARMUP_SEC))
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
        PROBE_ROI_QUERY_EMBED_BUDGET = int(
            os.getenv('EVOSSEARCH_PROBE_ROI_QUERY_EMBED_BUDGET', '2')
        )
    except (TypeError, ValueError):
        PROBE_ROI_QUERY_EMBED_BUDGET = 2
    PROBE_ROI_QUERY_EMBED_BUDGET = min(
        16,
        max(0, PROBE_ROI_QUERY_EMBED_BUDGET),
    )
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
    PROBE_REALTIME_BOOKMARK_ENABLED = _get_bool_env(
        'EVOSSEARCH_PROBE_REALTIME_BOOKMARK_ENABLED',
        'True',
    )
    try:
        PROBE_REALTIME_CONFIRM_HITS = int(
            os.getenv('EVOSSEARCH_PROBE_REALTIME_CONFIRM_HITS', '2')
        )
    except (TypeError, ValueError):
        PROBE_REALTIME_CONFIRM_HITS = 2
    PROBE_REALTIME_CONFIRM_HITS = min(3, max(1, PROBE_REALTIME_CONFIRM_HITS))
    try:
        PROBE_REALTIME_CONFIRM_WINDOW_SEC = float(
            os.getenv('EVOSSEARCH_PROBE_REALTIME_CONFIRM_WINDOW_SEC', '3.2')
        )
    except (TypeError, ValueError):
        PROBE_REALTIME_CONFIRM_WINDOW_SEC = 3.2
    PROBE_REALTIME_CONFIRM_WINDOW_SEC = min(
        10.0,
        max(1.0, PROBE_REALTIME_CONFIRM_WINDOW_SEC),
    )
    try:
        PROBE_REALTIME_MAX_EVENT_AGE_SEC = float(
            os.getenv('EVOSSEARCH_PROBE_REALTIME_MAX_EVENT_AGE_SEC', '5.0')
        )
    except (TypeError, ValueError):
        PROBE_REALTIME_MAX_EVENT_AGE_SEC = 5.0
    PROBE_REALTIME_MAX_EVENT_AGE_SEC = min(
        30.0,
        max(1.0, PROBE_REALTIME_MAX_EVENT_AGE_SEC),
    )
    try:
        PROBE_REALTIME_STRONG_SCORE_BOOST = float(
            os.getenv('EVOSSEARCH_PROBE_REALTIME_STRONG_SCORE_BOOST', '0.06')
        )
    except (TypeError, ValueError):
        PROBE_REALTIME_STRONG_SCORE_BOOST = 0.06
    PROBE_REALTIME_STRONG_SCORE_BOOST = min(
        1.0,
        max(0.0, PROBE_REALTIME_STRONG_SCORE_BOOST),
    )
    try:
        PROBE_REALTIME_WORKERS = int(
            os.getenv('EVOSSEARCH_PROBE_REALTIME_WORKERS', '2')
        )
    except (TypeError, ValueError):
        PROBE_REALTIME_WORKERS = 2
    PROBE_REALTIME_WORKERS = min(8, max(1, PROBE_REALTIME_WORKERS))
    try:
        PROBE_REALTIME_QUEUE_CAPACITY = int(
            os.getenv('EVOSSEARCH_PROBE_REALTIME_QUEUE_CAPACITY', '32')
        )
    except (TypeError, ValueError):
        PROBE_REALTIME_QUEUE_CAPACITY = 32
    PROBE_REALTIME_QUEUE_CAPACITY = min(
        256,
        max(PROBE_REALTIME_WORKERS, PROBE_REALTIME_QUEUE_CAPACITY),
    )
    VLM_FAST_ALERT_ENABLED = _get_bool_env(
        'EVOSSEARCH_VLM_FAST_ALERT_ENABLED',
        'True',
    )
    try:
        VLM_FAST_ALERT_POST_ROLL_SEC = float(
            os.getenv('EVOSSEARCH_VLM_FAST_ALERT_POST_ROLL_SEC', '2.5')
        )
    except (TypeError, ValueError):
        VLM_FAST_ALERT_POST_ROLL_SEC = 2.5
    VLM_FAST_ALERT_POST_ROLL_SEC = min(3.0, max(0.0, VLM_FAST_ALERT_POST_ROLL_SEC))
    try:
        VLM_FAST_ALERT_COOLDOWN_SEC = float(
            os.getenv('EVOSSEARCH_VLM_FAST_ALERT_COOLDOWN_SEC', '12.0')
        )
    except (TypeError, ValueError):
        VLM_FAST_ALERT_COOLDOWN_SEC = 12.0
    VLM_FAST_ALERT_COOLDOWN_SEC = min(120.0, max(1.0, VLM_FAST_ALERT_COOLDOWN_SEC))
    try:
        VLM_FAST_ALERT_MAX_FRAMES = int(
            os.getenv('EVOSSEARCH_VLM_FAST_ALERT_MAX_FRAMES', '6')
        )
    except (TypeError, ValueError):
        VLM_FAST_ALERT_MAX_FRAMES = 6
    VLM_FAST_ALERT_MAX_FRAMES = min(8, max(4, VLM_FAST_ALERT_MAX_FRAMES))
    try:
        VLM_FAST_ALERT_MAX_TOKENS = int(
            os.getenv('EVOSSEARCH_VLM_FAST_ALERT_MAX_TOKENS', '128')
        )
    except (TypeError, ValueError):
        VLM_FAST_ALERT_MAX_TOKENS = 128
    VLM_FAST_ALERT_MAX_TOKENS = min(512, max(128, VLM_FAST_ALERT_MAX_TOKENS))
    try:
        VLM_FAST_ALERT_WORKERS = int(
            os.getenv('EVOSSEARCH_VLM_FAST_ALERT_WORKERS', '2')
        )
    except (TypeError, ValueError):
        VLM_FAST_ALERT_WORKERS = 2
    VLM_FAST_ALERT_WORKERS = min(4, max(1, VLM_FAST_ALERT_WORKERS))
    try:
        VLM_FAST_ALERT_SEMANTIC_DELTA = float(
            os.getenv('EVOSSEARCH_VLM_FAST_ALERT_SEMANTIC_DELTA', '0.22')
        )
    except (TypeError, ValueError):
        VLM_FAST_ALERT_SEMANTIC_DELTA = 0.22
    VLM_FAST_ALERT_SEMANTIC_DELTA = min(
        1.0,
        max(0.0, VLM_FAST_ALERT_SEMANTIC_DELTA),
    )
    try:
        VLM_FAST_ALERT_MIN_MOVING_FRACTION = float(
            os.getenv('EVOSSEARCH_VLM_FAST_ALERT_MIN_MOVING_FRACTION', '0.15')
        )
    except (TypeError, ValueError):
        VLM_FAST_ALERT_MIN_MOVING_FRACTION = 0.15
    VLM_FAST_ALERT_MIN_MOVING_FRACTION = min(
        1.0,
        max(0.0, VLM_FAST_ALERT_MIN_MOVING_FRACTION),
    )
    try:
        VLM_FAST_ALERT_DEDUPE_WINDOW_SEC = float(
            os.getenv('EVOSSEARCH_VLM_FAST_ALERT_DEDUPE_WINDOW_SEC', '12.0')
        )
    except (TypeError, ValueError):
        VLM_FAST_ALERT_DEDUPE_WINDOW_SEC = 12.0
    VLM_FAST_ALERT_DEDUPE_WINDOW_SEC = min(
        120.0,
        max(1.0, VLM_FAST_ALERT_DEDUPE_WINDOW_SEC),
    )

    # Detection archive + adaptive retention
    DETECTIONS_ARCHIVE_ENABLED = _get_bool_env('EVOSSEARCH_DETECTIONS_ARCHIVE_ENABLED', 'True')
    DETECTIONS_ARCHIVE_DIR = os.getenv('EVOSSEARCH_DETECTIONS_ARCHIVE_DIR', 'detections_archive').strip() or 'detections_archive'
    try:
        ARCHIVE_DISK_MIN_FREE_GB = float(
            os.getenv('EVOSSEARCH_ARCHIVE_DISK_MIN_FREE_GB', '2.0')
        )
    except (TypeError, ValueError):
        ARCHIVE_DISK_MIN_FREE_GB = 2.0
    ARCHIVE_DISK_MIN_FREE_GB = max(0.0, ARCHIVE_DISK_MIN_FREE_GB)
    try:
        ARCHIVE_DISK_MIN_FREE_PERCENT = float(
            os.getenv('EVOSSEARCH_ARCHIVE_DISK_MIN_FREE_PERCENT', '5.0')
        )
    except (TypeError, ValueError):
        ARCHIVE_DISK_MIN_FREE_PERCENT = 5.0
    ARCHIVE_DISK_MIN_FREE_PERCENT = min(
        50.0,
        max(0.0, ARCHIVE_DISK_MIN_FREE_PERCENT),
    )
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
            f"(frames: default {cls.LM_VIDEO_DEFAULT_FRAMES}, max {cls.LM_VIDEO_MAX_FRAMES}, "
            f"max_edge={cls.LM_VIDEO_MAX_EDGE}, max_tokens={cls.LM_VIDEO_MAX_TOKENS}, "
            f"repetition_penalty={cls.LM_VIDEO_REPETITION_PENALTY}, "
            f"input_warn_chars={cls.LM_VIDEO_INPUT_WARNING_CHARS})"
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
            f"capture_source={cls.LUXRIOT_CAPTURE_SOURCE}, "
            f"buffer cap {cls.LUXRIOT_MAX_BUFFER_FRAMES} frames, "
            f"vector_signals {'on' if cls.LUXRIOT_VECTOR_SIGNALS_ENABLED else 'off'}, "
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
        print(f"Probe channel groups: {cls.PROBE_CHANNEL_GROUPS_FILE}")
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
