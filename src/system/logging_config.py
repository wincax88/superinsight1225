"""
Centralized Logging Configuration for SuperInsight Platform.

Provides structured logging with different handlers for different environments.
"""

import logging
import logging.handlers
import os
import sys
from typing import Dict, Any
from datetime import datetime

from src.config.settings import settings


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def format(self, record):
        # Add custom fields to log record
        record.service_name = getattr(record, 'service_name', 'superinsight')
        record.request_id = getattr(record, 'request_id', None)
        record.user_id = getattr(record, 'user_id', None)
        
        # Format timestamp
        record.timestamp = datetime.fromtimestamp(record.created).isoformat()
        
        return super().format(record)


def setup_logging() -> None:
    """Setup comprehensive logging configuration."""
    
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.app.log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler for development
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    console_format = (
        "%(asctime)s - %(name)s - %(levelname)s - "
        "%(message)s"
    )
    
    if settings.app.debug:
        console_format = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[%(filename)s:%(lineno)d] - %(message)s"
        )
    
    console_formatter = logging.Formatter(console_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for general application logs
    app_file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, "app.log"),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    app_file_handler.setLevel(logging.INFO)
    
    app_file_format = (
        "%(timestamp)s - %(name)s - %(levelname)s - "
        "%(service_name)s - %(message)s"
    )
    
    app_file_formatter = StructuredFormatter(app_file_format)
    app_file_handler.setFormatter(app_file_formatter)
    root_logger.addHandler(app_file_handler)
    
    # Error file handler for errors and above
    error_file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, "errors.log"),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=10
    )
    error_file_handler.setLevel(logging.ERROR)
    
    error_file_format = (
        "%(timestamp)s - %(name)s - %(levelname)s - "
        "%(service_name)s - %(request_id)s - %(user_id)s - "
        "%(message)s"
    )
    
    error_file_formatter = StructuredFormatter(error_file_format)
    error_file_handler.setFormatter(error_file_formatter)
    root_logger.addHandler(error_file_handler)
    
    # Access log handler for API requests
    access_logger = logging.getLogger("access")
    access_logger.setLevel(logging.INFO)
    access_logger.propagate = False  # Don't propagate to root logger
    
    access_file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, "access.log"),
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=10
    )
    
    access_format = (
        "%(timestamp)s - %(request_id)s - %(user_id)s - "
        "%(method)s %(path)s - %(status_code)s - "
        "%(duration_ms)sms - %(ip_address)s - %(user_agent)s"
    )
    
    access_formatter = StructuredFormatter(access_format)
    access_file_handler.setFormatter(access_formatter)
    access_logger.addHandler(access_file_handler)
    
    # Security log handler for security events
    security_logger = logging.getLogger("security")
    security_logger.setLevel(logging.INFO)
    security_logger.propagate = False
    
    security_file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, "security.log"),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=20
    )
    
    security_format = (
        "%(timestamp)s - %(levelname)s - %(event_type)s - "
        "%(user_id)s - %(ip_address)s - %(message)s"
    )
    
    security_formatter = StructuredFormatter(security_format)
    security_file_handler.setFormatter(security_formatter)
    security_logger.addHandler(security_file_handler)
    
    # Audit log handler for audit events
    audit_logger = logging.getLogger("audit")
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False
    
    audit_file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, "audit.log"),
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=50  # Keep more audit logs
    )
    
    audit_format = (
        "%(timestamp)s - %(user_id)s - %(action)s - "
        "%(resource_type)s - %(resource_id)s - "
        "%(ip_address)s - %(details)s"
    )
    
    audit_formatter = StructuredFormatter(audit_format)
    audit_file_handler.setFormatter(audit_formatter)
    audit_logger.addHandler(audit_file_handler)
    
    # Set specific logger levels
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    
    logging.info("Logging configuration initialized")


class LoggerAdapter(logging.LoggerAdapter):
    """Custom logger adapter for adding context to log messages."""
    
    def __init__(self, logger, extra=None):
        super().__init__(logger, extra or {})
    
    def process(self, msg, kwargs):
        # Add extra context to log record
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        kwargs['extra'].update(self.extra)
        return msg, kwargs


def get_logger(name: str, **context) -> LoggerAdapter:
    """Get a logger with optional context."""
    logger = logging.getLogger(name)
    return LoggerAdapter(logger, context)


def log_access(
    request_id: str,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    ip_address: str,
    user_agent: str = "",
    user_id: str = None
) -> None:
    """Log an access event."""
    access_logger = logging.getLogger("access")
    
    access_logger.info(
        f"{method} {path} - {status_code}",
        extra={
            "request_id": request_id,
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "user_id": user_id
        }
    )


def log_security_event(
    event_type: str,
    message: str,
    user_id: str = None,
    ip_address: str = None,
    level: str = "INFO"
) -> None:
    """Log a security event."""
    security_logger = logging.getLogger("security")
    
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    security_logger.log(
        log_level,
        message,
        extra={
            "event_type": event_type,
            "user_id": user_id,
            "ip_address": ip_address
        }
    )


def log_audit_event(
    user_id: str,
    action: str,
    resource_type: str,
    resource_id: str = None,
    ip_address: str = None,
    details: str = None
) -> None:
    """Log an audit event."""
    audit_logger = logging.getLogger("audit")
    
    audit_logger.info(
        f"User {user_id} performed {action} on {resource_type}",
        extra={
            "user_id": user_id,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "ip_address": ip_address,
            "details": details or ""
        }
    )


# Initialize logging on module import
setup_logging()