"""
Logging configuration and management for SuperInsight Platform.

Provides structured logging, log rotation, and log level management.
"""

import logging
import logging.handlers
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class SecurityLogFormatter(logging.Formatter):
    """Custom formatter for security-related logs."""
    
    def format(self, record):
        """Format log record with security context."""
        # Create structured log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add security context if available
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'tenant_id'):
            log_entry['tenant_id'] = record.tenant_id
        if hasattr(record, 'ip_address'):
            log_entry['ip_address'] = record.ip_address
        if hasattr(record, 'action'):
            log_entry['action'] = record.action
        if hasattr(record, 'resource_type'):
            log_entry['resource_type'] = record.resource_type
        if hasattr(record, 'resource_id'):
            log_entry['resource_id'] = record.resource_id
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


class LogManager:
    """
    Centralized log management for security and audit purposes.
    
    Handles log configuration, rotation, and structured logging.
    """
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.loggers = {}
        self._setup_loggers()
    
    def _setup_loggers(self):
        """Set up different loggers for different purposes."""
        
        # Security logger for authentication, authorization, etc.
        security_logger = logging.getLogger("security")
        security_logger.setLevel(logging.INFO)
        
        security_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "security.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10
        )
        security_handler.setFormatter(SecurityLogFormatter())
        security_logger.addHandler(security_handler)
        
        # Audit logger for user actions
        audit_logger = logging.getLogger("audit")
        audit_logger.setLevel(logging.INFO)
        
        audit_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "audit.log",
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=20
        )
        audit_handler.setFormatter(SecurityLogFormatter())
        audit_logger.addHandler(audit_handler)
        
        # Error logger for system errors
        error_logger = logging.getLogger("errors")
        error_logger.setLevel(logging.ERROR)
        
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "errors.log",
            maxBytes=20 * 1024 * 1024,  # 20MB
            backupCount=15
        )
        error_handler.setFormatter(SecurityLogFormatter())
        error_logger.addHandler(error_handler)
        
        # Access logger for API access
        access_logger = logging.getLogger("access")
        access_logger.setLevel(logging.INFO)
        
        access_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "access.log",
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=30
        )
        access_handler.setFormatter(SecurityLogFormatter())
        access_logger.addHandler(access_handler)
        
        self.loggers = {
            "security": security_logger,
            "audit": audit_logger,
            "errors": error_logger,
            "access": access_logger
        }
    
    def log_security_event(
        self,
        message: str,
        level: str = "INFO",
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None
    ):
        """Log a security event with structured data."""
        logger = self.loggers["security"]
        
        # Create log record with extra attributes
        extra = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "ip_address": ip_address,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id
        }
        
        if extra_data:
            extra.update(extra_data)
        
        # Remove None values
        extra = {k: v for k, v in extra.items() if v is not None}
        
        log_level = getattr(logging, level.upper(), logging.INFO)
        logger.log(log_level, message, extra=extra)
    
    def log_audit_event(
        self,
        action: str,
        resource_type: str,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log an audit event."""
        logger = self.loggers["audit"]
        
        message = f"User {user_id} performed {action} on {resource_type}"
        if resource_id:
            message += f" (ID: {resource_id})"
        
        extra = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "ip_address": ip_address
        }
        
        if details:
            extra.update(details)
        
        # Remove None values
        extra = {k: v for k, v in extra.items() if v is not None}
        
        logger.info(message, extra=extra)
    
    def log_error(
        self,
        message: str,
        exception: Optional[Exception] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None
    ):
        """Log an error event."""
        logger = self.loggers["errors"]
        
        extra = {
            "user_id": user_id,
            "tenant_id": tenant_id
        }
        
        if extra_data:
            extra.update(extra_data)
        
        # Remove None values
        extra = {k: v for k, v in extra.items() if v is not None}
        
        if exception:
            logger.error(message, exc_info=exception, extra=extra)
        else:
            logger.error(message, extra=extra)
    
    def log_access(
        self,
        method: str,
        path: str,
        status_code: int,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        response_time_ms: Optional[float] = None
    ):
        """Log an API access event."""
        logger = self.loggers["access"]
        
        message = f"{method} {path} - {status_code}"
        
        extra = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "ip_address": ip_address,
            "method": method,
            "path": path,
            "status_code": status_code,
            "user_agent": user_agent,
            "response_time_ms": response_time_ms
        }
        
        # Remove None values
        extra = {k: v for k, v in extra.items() if v is not None}
        
        logger.info(message, extra=extra)
    
    def get_log_files(self) -> Dict[str, Dict[str, Any]]:
        """Get information about log files."""
        log_files = {}
        
        for log_type in ["security", "audit", "errors", "access"]:
            log_file = self.log_dir / f"{log_type}.log"
            if log_file.exists():
                stat = log_file.stat()
                log_files[log_type] = {
                    "path": str(log_file),
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
        
        return log_files
    
    def rotate_logs(self) -> Dict[str, Any]:
        """Manually trigger log rotation for all loggers."""
        rotated = []
        
        for logger_name, logger in self.loggers.items():
            for handler in logger.handlers:
                if isinstance(handler, logging.handlers.RotatingFileHandler):
                    try:
                        handler.doRollover()
                        rotated.append(logger_name)
                    except Exception as e:
                        self.log_error(f"Failed to rotate {logger_name} log", e)
        
        return {
            "rotated_loggers": rotated,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def set_log_level(self, logger_name: str, level: str) -> bool:
        """Set log level for a specific logger."""
        if logger_name not in self.loggers:
            return False
        
        try:
            log_level = getattr(logging, level.upper())
            self.loggers[logger_name].setLevel(log_level)
            return True
        except AttributeError:
            return False
    
    def get_log_levels(self) -> Dict[str, str]:
        """Get current log levels for all loggers."""
        return {
            name: logging.getLevelName(logger.level)
            for name, logger in self.loggers.items()
        }


# Global log manager instance
log_manager = LogManager()


# Convenience functions for common logging operations

def log_security_event(message: str, **kwargs):
    """Convenience function to log security events."""
    log_manager.log_security_event(message, **kwargs)


def log_audit_event(action: str, resource_type: str, **kwargs):
    """Convenience function to log audit events."""
    log_manager.log_audit_event(action, resource_type, **kwargs)


def log_error(message: str, exception: Optional[Exception] = None, **kwargs):
    """Convenience function to log errors."""
    log_manager.log_error(message, exception, **kwargs)


def log_access(method: str, path: str, status_code: int, **kwargs):
    """Convenience function to log API access."""
    log_manager.log_access(method, path, status_code, **kwargs)