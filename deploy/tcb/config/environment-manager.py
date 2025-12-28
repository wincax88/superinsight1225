#!/usr/bin/env python3
"""
SuperInsight TCB Environment Configuration Manager

This module provides a centralized configuration management system for
multi-environment deployments on Tencent CloudBase (TCB).

Features:
- Multi-environment support (development, staging, production)
- Secure secret management
- Configuration validation
- Hot reload capability
- Audit logging
"""

import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ConfigValidationError:
    """Represents a configuration validation error."""
    field: str
    message: str
    severity: str = "error"  # error, warning, info


@dataclass
class EnvironmentConfig:
    """Environment-specific configuration."""
    name: str
    debug: bool = False
    log_level: str = "INFO"

    # Database configuration
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "superinsight"
    postgres_user: str = "superinsight"
    postgres_pool_size: int = 10
    postgres_max_overflow: int = 20

    # Redis configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # Application configuration
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    workers: int = 4

    # Label Studio configuration
    label_studio_host: str = "0.0.0.0"
    label_studio_port: int = 8080

    # Monitoring
    prometheus_enabled: bool = True
    prometheus_port: int = 9090

    # Security
    allowed_hosts: List[str] = field(default_factory=list)
    csrf_trusted_origins: List[str] = field(default_factory=list)

    # Feature flags
    features: Dict[str, bool] = field(default_factory=dict)


class ConfigManager:
    """
    Centralized configuration manager for TCB deployments.

    Handles environment-specific configurations, secret management,
    and configuration validation.
    """

    ENVIRONMENTS = ["development", "staging", "production"]

    REQUIRED_SECRETS = [
        "POSTGRES_PASSWORD",
        "SECRET_KEY",
        "JWT_SECRET_KEY",
    ]

    OPTIONAL_SECRETS = [
        "HUNYUAN_API_KEY",
        "HUNYUAN_SECRET_KEY",
        "COS_SECRET_ID",
        "COS_SECRET_KEY",
        "LABEL_STUDIO_TOKEN",
        "ENCRYPTION_KEY",
    ]

    def __init__(self, environment: Optional[str] = None):
        """Initialize the configuration manager."""
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        self._validate_environment()
        self._config: Optional[EnvironmentConfig] = None
        self._config_hash: Optional[str] = None
        self._last_loaded: Optional[datetime] = None

    def _validate_environment(self) -> None:
        """Validate that the environment is supported."""
        if self.environment not in self.ENVIRONMENTS:
            raise ValueError(
                f"Invalid environment: {self.environment}. "
                f"Must be one of: {', '.join(self.ENVIRONMENTS)}"
            )

    @lru_cache(maxsize=3)
    def _load_env_file(self, env_name: str) -> Dict[str, str]:
        """Load environment variables from a .env file."""
        env_file = Path(__file__).parent.parent / "env" / f"{env_name}.env"

        if not env_file.exists():
            logger.warning(f"Environment file not found: {env_file}")
            return {}

        config = {}
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Handle variable substitution
                    if '${' in value:
                        value = self._substitute_variables(value, config)
                    config[key.strip()] = value.strip()

        return config

    def _substitute_variables(self, value: str, config: Dict[str, str]) -> str:
        """Substitute ${VAR} patterns with actual values."""
        import re
        pattern = r'\$\{([^}]+)\}'

        def replace(match):
            var_name = match.group(1)
            # Check environment first, then config, then return placeholder
            return os.getenv(var_name, config.get(var_name, f'${{{var_name}}}'))

        return re.sub(pattern, replace, value)

    def load_config(self, force_reload: bool = False) -> EnvironmentConfig:
        """Load configuration for the current environment."""
        if self._config and not force_reload:
            return self._config

        logger.info(f"Loading configuration for environment: {self.environment}")

        # Load from environment file
        file_config = self._load_env_file(self.environment)

        # Merge with environment variables (env vars take precedence)
        merged_config = {**file_config}
        for key in file_config:
            env_value = os.getenv(key)
            if env_value is not None:
                merged_config[key] = env_value

        # Create configuration object
        self._config = EnvironmentConfig(
            name=self.environment,
            debug=merged_config.get("DEBUG", "false").lower() == "true",
            log_level=merged_config.get("LOG_LEVEL", "INFO"),
            postgres_host=merged_config.get("POSTGRES_HOST", "localhost"),
            postgres_port=int(merged_config.get("POSTGRES_PORT", 5432)),
            postgres_db=merged_config.get("POSTGRES_DB", "superinsight"),
            postgres_user=merged_config.get("POSTGRES_USER", "superinsight"),
            postgres_pool_size=int(merged_config.get("DATABASE_POOL_SIZE", 10)),
            postgres_max_overflow=int(merged_config.get("DATABASE_MAX_OVERFLOW", 20)),
            redis_host=merged_config.get("REDIS_HOST", "localhost"),
            redis_port=int(merged_config.get("REDIS_PORT", 6379)),
            redis_db=int(merged_config.get("REDIS_DB", 0)),
            app_host=merged_config.get("APP_HOST", "0.0.0.0"),
            app_port=int(merged_config.get("APP_PORT", 8000)),
            workers=int(merged_config.get("WORKERS", 4)),
            label_studio_host=merged_config.get("LABEL_STUDIO_HOST", "0.0.0.0"),
            label_studio_port=int(merged_config.get("LABEL_STUDIO_PORT", 8080)),
            prometheus_enabled=merged_config.get("PROMETHEUS_ENABLED", "true").lower() == "true",
            prometheus_port=int(merged_config.get("PROMETHEUS_PORT", 9090)),
            allowed_hosts=merged_config.get("ALLOWED_HOSTS", "").split(","),
            csrf_trusted_origins=merged_config.get("CSRF_TRUSTED_ORIGINS", "").split(","),
        )

        # Compute config hash for change detection
        self._config_hash = self._compute_hash(merged_config)
        self._last_loaded = datetime.now()

        logger.info(f"Configuration loaded successfully (hash: {self._config_hash[:8]})")
        return self._config

    def _compute_hash(self, config: Dict[str, str]) -> str:
        """Compute a hash of the configuration for change detection."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def validate_config(self) -> List[ConfigValidationError]:
        """Validate the current configuration."""
        errors: List[ConfigValidationError] = []

        if not self._config:
            self.load_config()

        # Validate required secrets
        for secret in self.REQUIRED_SECRETS:
            if not os.getenv(secret):
                errors.append(ConfigValidationError(
                    field=secret,
                    message=f"Required secret '{secret}' is not set",
                    severity="error"
                ))

        # Validate optional secrets
        for secret in self.OPTIONAL_SECRETS:
            if not os.getenv(secret):
                errors.append(ConfigValidationError(
                    field=secret,
                    message=f"Optional secret '{secret}' is not set",
                    severity="warning"
                ))

        # Validate port ranges
        if not (1 <= self._config.app_port <= 65535):
            errors.append(ConfigValidationError(
                field="APP_PORT",
                message=f"Invalid port: {self._config.app_port}",
                severity="error"
            ))

        # Validate production-specific settings
        if self.environment == "production":
            if self._config.debug:
                errors.append(ConfigValidationError(
                    field="DEBUG",
                    message="Debug mode should be disabled in production",
                    severity="error"
                ))

            if self._config.log_level == "DEBUG":
                errors.append(ConfigValidationError(
                    field="LOG_LEVEL",
                    message="Debug log level not recommended in production",
                    severity="warning"
                ))

            if not self._config.allowed_hosts or self._config.allowed_hosts == ['']:
                errors.append(ConfigValidationError(
                    field="ALLOWED_HOSTS",
                    message="ALLOWED_HOSTS should be configured in production",
                    severity="warning"
                ))

        return errors

    def get_database_url(self) -> str:
        """Get the database connection URL."""
        if not self._config:
            self.load_config()

        password = os.getenv("POSTGRES_PASSWORD", "")
        return (
            f"postgresql://{self._config.postgres_user}:{password}"
            f"@{self._config.postgres_host}:{self._config.postgres_port}"
            f"/{self._config.postgres_db}"
        )

    def get_redis_url(self) -> str:
        """Get the Redis connection URL."""
        if not self._config:
            self.load_config()

        return f"redis://{self._config.redis_host}:{self._config.redis_port}/{self._config.redis_db}"

    def export_env_vars(self) -> Dict[str, str]:
        """Export configuration as environment variables."""
        if not self._config:
            self.load_config()

        return {
            "ENVIRONMENT": self._config.name,
            "DEBUG": str(self._config.debug).lower(),
            "LOG_LEVEL": self._config.log_level,
            "POSTGRES_HOST": self._config.postgres_host,
            "POSTGRES_PORT": str(self._config.postgres_port),
            "POSTGRES_DB": self._config.postgres_db,
            "POSTGRES_USER": self._config.postgres_user,
            "DATABASE_URL": self.get_database_url(),
            "REDIS_HOST": self._config.redis_host,
            "REDIS_PORT": str(self._config.redis_port),
            "REDIS_URL": self.get_redis_url(),
            "APP_HOST": self._config.app_host,
            "APP_PORT": str(self._config.app_port),
            "PROMETHEUS_ENABLED": str(self._config.prometheus_enabled).lower(),
        }

    def print_config_summary(self) -> None:
        """Print a summary of the current configuration."""
        if not self._config:
            self.load_config()

        print(f"\n{'=' * 60}")
        print(f"SuperInsight Configuration Summary")
        print(f"{'=' * 60}")
        print(f"Environment:     {self._config.name}")
        print(f"Debug Mode:      {self._config.debug}")
        print(f"Log Level:       {self._config.log_level}")
        print(f"{'=' * 60}")
        print(f"Database:        {self._config.postgres_host}:{self._config.postgres_port}/{self._config.postgres_db}")
        print(f"Redis:           {self._config.redis_host}:{self._config.redis_port}")
        print(f"API Server:      {self._config.app_host}:{self._config.app_port}")
        print(f"Label Studio:    {self._config.label_studio_host}:{self._config.label_studio_port}")
        print(f"{'=' * 60}")
        print(f"Prometheus:      {'Enabled' if self._config.prometheus_enabled else 'Disabled'}")
        print(f"Config Hash:     {self._config_hash[:16] if self._config_hash else 'N/A'}")
        print(f"Last Loaded:     {self._last_loaded}")
        print(f"{'=' * 60}\n")


def main():
    """CLI entry point for configuration management."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SuperInsight TCB Configuration Manager"
    )
    parser.add_argument(
        "--env", "-e",
        choices=ConfigManager.ENVIRONMENTS,
        default=os.getenv("ENVIRONMENT", "development"),
        help="Target environment"
    )
    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Validate configuration"
    )
    parser.add_argument(
        "--export", "-x",
        action="store_true",
        help="Export configuration as environment variables"
    )
    parser.add_argument(
        "--summary", "-s",
        action="store_true",
        help="Print configuration summary"
    )

    args = parser.parse_args()

    manager = ConfigManager(environment=args.env)
    config = manager.load_config()

    if args.validate:
        errors = manager.validate_config()
        if errors:
            print("\nConfiguration Validation Results:")
            print("-" * 40)
            for error in errors:
                icon = "ERROR" if error.severity == "error" else "WARN"
                print(f"[{icon}] {error.field}: {error.message}")

            error_count = sum(1 for e in errors if e.severity == "error")
            if error_count > 0:
                print(f"\n{error_count} error(s) found. Please fix before deployment.")
                sys.exit(1)
        else:
            print("\nConfiguration validation passed!")

    if args.export:
        env_vars = manager.export_env_vars()
        print("\n# Exported Environment Variables")
        for key, value in env_vars.items():
            # Mask sensitive values
            if "PASSWORD" in key or "SECRET" in key or "KEY" in key:
                value = "********"
            print(f"export {key}=\"{value}\"")

    if args.summary:
        manager.print_config_summary()


if __name__ == "__main__":
    main()
