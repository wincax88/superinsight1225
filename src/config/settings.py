"""
SuperInsight Platform Configuration Settings
"""
import os
from typing import Optional
from dataclasses import dataclass, field


def get_env(key: str, default: str = "") -> str:
    """Get environment variable with default value"""
    return os.getenv(key, default)


def get_env_int(key: str, default: int = 0) -> int:
    """Get environment variable as integer"""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def get_env_float(key: str, default: float = 0.0) -> float:
    """Get environment variable as float"""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get environment variable as boolean"""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


@dataclass
class DatabaseSettings:
    """Database configuration settings"""
    
    # PostgreSQL Configuration
    database_url: str = field(default_factory=lambda: get_env("DATABASE_URL", "postgresql://superinsight:password@localhost:5432/superinsight"))
    database_host: str = field(default_factory=lambda: get_env("DATABASE_HOST", "localhost"))
    database_port: int = field(default_factory=lambda: get_env_int("DATABASE_PORT", 5432))
    database_name: str = field(default_factory=lambda: get_env("DATABASE_NAME", "superinsight"))
    database_user: str = field(default_factory=lambda: get_env("DATABASE_USER", "superinsight"))
    database_password: str = field(default_factory=lambda: get_env("DATABASE_PASSWORD", "password"))
    
    # Connection pool settings
    database_pool_size: int = field(default_factory=lambda: get_env_int("DATABASE_POOL_SIZE", 10))
    database_max_overflow: int = field(default_factory=lambda: get_env_int("DATABASE_MAX_OVERFLOW", 20))
    database_pool_timeout: int = field(default_factory=lambda: get_env_int("DATABASE_POOL_TIMEOUT", 30))


@dataclass
class LabelStudioSettings:
    """Label Studio configuration settings"""
    
    label_studio_url: str = field(default_factory=lambda: get_env("LABEL_STUDIO_URL", "http://localhost:8080"))
    label_studio_api_token: Optional[str] = field(default_factory=lambda: get_env("LABEL_STUDIO_API_TOKEN") or None)
    label_studio_project_id: int = field(default_factory=lambda: get_env_int("LABEL_STUDIO_PROJECT_ID", 1))


@dataclass
class AISettings:
    """AI services configuration settings"""
    
    # Ollama Configuration
    ollama_base_url: str = field(default_factory=lambda: get_env("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: get_env("OLLAMA_MODEL", "llama2"))
    
    # HuggingFace Configuration
    huggingface_api_token: Optional[str] = field(default_factory=lambda: get_env("HUGGINGFACE_API_TOKEN") or None)
    huggingface_model: str = field(default_factory=lambda: get_env("HUGGINGFACE_MODEL", "bert-base-chinese"))
    
    # Chinese LLM APIs
    zhipu_api_key: Optional[str] = field(default_factory=lambda: get_env("ZHIPU_API_KEY") or None)
    baidu_api_key: Optional[str] = field(default_factory=lambda: get_env("BAIDU_API_KEY") or None)
    baidu_secret_key: Optional[str] = field(default_factory=lambda: get_env("BAIDU_SECRET_KEY") or None)
    alibaba_api_key: Optional[str] = field(default_factory=lambda: get_env("ALIBABA_API_KEY") or None)
    tencent_api_key: Optional[str] = field(default_factory=lambda: get_env("TENCENT_API_KEY") or None)


@dataclass
class SecuritySettings:
    """Security configuration settings"""
    
    secret_key: str = field(default_factory=lambda: get_env("SECRET_KEY", "dev-secret-key-change-in-production"))
    jwt_secret_key: str = field(default_factory=lambda: get_env("JWT_SECRET_KEY", "dev-jwt-secret-change-in-production"))
    encryption_key: str = field(default_factory=lambda: get_env("ENCRYPTION_KEY", "dev-encryption-key-change-in-production"))
    
    # JWT settings
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30


@dataclass
class TCBSettings:
    """Tencent Cloud TCB configuration settings"""
    
    tcb_env_id: Optional[str] = field(default_factory=lambda: get_env("TCB_ENV_ID") or None)
    tcb_secret_id: Optional[str] = field(default_factory=lambda: get_env("TCB_SECRET_ID") or None)
    tcb_secret_key: Optional[str] = field(default_factory=lambda: get_env("TCB_SECRET_KEY") or None)


@dataclass
class RedisSettings:
    """Redis configuration settings"""
    
    redis_url: str = field(default_factory=lambda: get_env("REDIS_URL", "redis://localhost:6379/0"))
    redis_host: str = field(default_factory=lambda: get_env("REDIS_HOST", "localhost"))
    redis_port: int = field(default_factory=lambda: get_env_int("REDIS_PORT", 6379))
    redis_db: int = field(default_factory=lambda: get_env_int("REDIS_DB", 0))


@dataclass
class HealthCheckSettings:
    """Health check configuration settings"""

    # General health check settings
    health_check_enabled: bool = field(default_factory=lambda: get_env_bool("HEALTH_CHECK_ENABLED", True))
    health_check_timeout: int = field(default_factory=lambda: get_env_int("HEALTH_CHECK_TIMEOUT", 30))
    health_check_retry_attempts: int = field(default_factory=lambda: get_env_int("HEALTH_CHECK_RETRY_ATTEMPTS", 3))
    health_check_retry_delay: float = field(default_factory=lambda: get_env_float("HEALTH_CHECK_RETRY_DELAY", 1.0))

    # Individual health check toggles
    database_check_enabled: bool = field(default_factory=lambda: get_env_bool("DATABASE_CHECK_ENABLED", True))
    label_studio_check_enabled: bool = field(default_factory=lambda: get_env_bool("LABEL_STUDIO_CHECK_ENABLED", True))
    ai_services_check_enabled: bool = field(default_factory=lambda: get_env_bool("AI_SERVICES_CHECK_ENABLED", True))
    storage_check_enabled: bool = field(default_factory=lambda: get_env_bool("STORAGE_CHECK_ENABLED", True))
    security_check_enabled: bool = field(default_factory=lambda: get_env_bool("SECURITY_CHECK_ENABLED", True))
    external_deps_check_enabled: bool = field(default_factory=lambda: get_env_bool("EXTERNAL_DEPS_CHECK_ENABLED", True))

    # Kubernetes probe settings
    liveness_probe_path: str = field(default_factory=lambda: get_env("LIVENESS_PROBE_PATH", "/health/live"))
    readiness_probe_path: str = field(default_factory=lambda: get_env("READINESS_PROBE_PATH", "/health/ready"))

    # Thresholds
    min_disk_space_gb: float = field(default_factory=lambda: get_env_float("MIN_DISK_SPACE_GB", 1.0))
    max_response_time_ms: int = field(default_factory=lambda: get_env_int("MAX_RESPONSE_TIME_MS", 5000))


@dataclass
class AppSettings:
    """Application configuration settings"""

    app_name: str = field(default_factory=lambda: get_env("APP_NAME", "SuperInsight Platform"))
    app_version: str = field(default_factory=lambda: get_env("APP_VERSION", "1.0.0"))
    debug: bool = field(default_factory=lambda: get_env_bool("DEBUG", True))
    log_level: str = field(default_factory=lambda: get_env("LOG_LEVEL", "INFO"))

    # File storage settings
    upload_dir: str = field(default_factory=lambda: get_env("UPLOAD_DIR", "./uploads"))
    max_file_size: str = field(default_factory=lambda: get_env("MAX_FILE_SIZE", "100MB"))

    # Quality management settings
    ragas_api_key: Optional[str] = field(default_factory=lambda: get_env("RAGAS_API_KEY") or None)
    quality_threshold: float = field(default_factory=lambda: get_env_float("QUALITY_THRESHOLD", 0.8))

    # Billing settings
    billing_currency: str = field(default_factory=lambda: get_env("BILLING_CURRENCY", "CNY"))
    billing_rate_per_hour: float = field(default_factory=lambda: get_env_float("BILLING_RATE_PER_HOUR", 100.0))
    billing_rate_per_annotation: float = field(default_factory=lambda: get_env_float("BILLING_RATE_PER_ANNOTATION", 1.0))


@dataclass
class Settings:
    """Main settings class that combines all configuration sections"""

    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    label_studio: LabelStudioSettings = field(default_factory=LabelStudioSettings)
    ai: AISettings = field(default_factory=AISettings)
    security: SecuritySettings = field(default_factory=SecuritySettings)
    tcb: TCBSettings = field(default_factory=TCBSettings)
    redis: RedisSettings = field(default_factory=RedisSettings)
    app: AppSettings = field(default_factory=AppSettings)
    health_check: HealthCheckSettings = field(default_factory=HealthCheckSettings)


# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Global settings instance
settings = Settings()