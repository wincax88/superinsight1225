"""
Unified Error Handling System for SuperInsight Platform.

Provides centralized error handling, logging, and recovery mechanisms
for all platform components.
"""

import logging
import traceback
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

from src.config.settings import settings


logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    DATABASE = "database"
    EXTRACTION = "extraction"
    ANNOTATION = "annotation"
    QUALITY = "quality"
    BILLING = "billing"
    SECURITY = "security"
    INTEGRATION = "integration"
    EXTERNAL_API = "external_api"
    VALIDATION = "validation"
    SYSTEM = "system"


@dataclass
class ErrorContext:
    """Error context information."""
    error_id: str
    timestamp: float
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    exception: Optional[Exception] = None
    traceback_str: Optional[str] = None
    service_name: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryAction:
    """Recovery action definition."""
    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    backoff_multiplier: float = 2.0


class ErrorHandler:
    """
    Centralized error handling system.
    
    Provides:
    - Error classification and logging
    - Recovery action execution
    - Error metrics and reporting
    - Notification system integration
    """
    
    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.recovery_handlers: Dict[ErrorCategory, Callable] = {}
        self.notification_handlers: List[Callable] = []
        self.max_history_size = 1000
        
    def register_recovery_handler(self, category: ErrorCategory, handler: Callable):
        """Register a recovery handler for a specific error category."""
        self.recovery_handlers[category] = handler
        logger.info(f"Registered recovery handler for category: {category.value}")
    
    def register_notification_handler(self, handler: Callable):
        """Register a notification handler for error alerts."""
        self.notification_handlers.append(handler)
        logger.info("Registered notification handler")
    
    def handle_error(
        self,
        exception: Exception,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        service_name: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        **metadata
    ) -> ErrorContext:
        """Handle an error with full context and recovery."""
        
        # Create error context
        error_id = f"{category.value}_{int(time.time() * 1000)}"
        error_context = ErrorContext(
            error_id=error_id,
            timestamp=time.time(),
            category=category,
            severity=severity,
            message=str(exception),
            exception=exception,
            traceback_str=traceback.format_exc(),
            service_name=service_name,
            user_id=user_id,
            request_id=request_id,
            metadata=metadata
        )
        
        # Log the error
        self._log_error(error_context)
        
        # Store in history
        self._store_error(error_context)
        
        # Send notifications for high severity errors
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self._send_notifications(error_context)
        
        # Attempt recovery
        self._attempt_recovery(error_context)
        
        return error_context
    
    def _log_error(self, error_context: ErrorContext):
        """Log error with appropriate level based on severity."""
        log_message = (
            f"[{error_context.error_id}] {error_context.category.value.upper()} ERROR: "
            f"{error_context.message}"
        )
        
        if error_context.service_name:
            log_message += f" (Service: {error_context.service_name})"
        
        if error_context.user_id:
            log_message += f" (User: {error_context.user_id})"
        
        if error_context.request_id:
            log_message += f" (Request: {error_context.request_id})"
        
        # Log based on severity
        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
            if error_context.traceback_str:
                logger.critical(f"Traceback:\n{error_context.traceback_str}")
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
            if error_context.traceback_str:
                logger.error(f"Traceback:\n{error_context.traceback_str}")
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:  # LOW
            logger.info(log_message)
    
    def _store_error(self, error_context: ErrorContext):
        """Store error in history with size limit."""
        self.error_history.append(error_context)
        
        # Maintain history size limit
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
    
    def _send_notifications(self, error_context: ErrorContext):
        """Send notifications for high severity errors."""
        for handler in self.notification_handlers:
            try:
                handler(error_context)
            except Exception as e:
                logger.error(f"Notification handler failed: {e}")
    
    def _attempt_recovery(self, error_context: ErrorContext):
        """Attempt automatic recovery based on error category."""
        if error_context.category in self.recovery_handlers:
            try:
                recovery_handler = self.recovery_handlers[error_context.category]
                recovery_action = recovery_handler(error_context)
                
                if recovery_action:
                    self._execute_recovery_action(recovery_action, error_context)
                    
            except Exception as e:
                logger.error(f"Recovery handler failed for {error_context.category.value}: {e}")
    
    def _execute_recovery_action(self, action: RecoveryAction, error_context: ErrorContext):
        """Execute a recovery action with retry logic."""
        for attempt in range(action.max_retries + 1):
            try:
                logger.info(f"Executing recovery action: {action.action_type} (attempt {attempt + 1})")
                
                # Execute based on action type
                if action.action_type == "retry":
                    self._retry_operation(action, error_context)
                elif action.action_type == "fallback":
                    self._execute_fallback(action, error_context)
                elif action.action_type == "restart_service":
                    self._restart_service(action, error_context)
                elif action.action_type == "clear_cache":
                    self._clear_cache(action, error_context)
                else:
                    logger.warning(f"Unknown recovery action type: {action.action_type}")
                
                logger.info(f"Recovery action {action.action_type} completed successfully")
                return
                
            except Exception as e:
                logger.error(f"Recovery action {action.action_type} failed (attempt {attempt + 1}): {e}")
                
                if attempt < action.max_retries:
                    # Calculate backoff delay
                    delay = (action.backoff_multiplier ** attempt)
                    logger.info(f"Retrying recovery action in {delay} seconds...")
                    time.sleep(delay)
        
        logger.error(f"All recovery attempts failed for action: {action.action_type}")
    
    def _retry_operation(self, action: RecoveryAction, error_context: ErrorContext):
        """Retry the original operation."""
        # This would need to be implemented based on the specific operation
        logger.info("Retry operation recovery action executed")
    
    def _execute_fallback(self, action: RecoveryAction, error_context: ErrorContext):
        """Execute fallback operation."""
        # This would need to be implemented based on the specific fallback
        logger.info("Fallback recovery action executed")
    
    def _restart_service(self, action: RecoveryAction, error_context: ErrorContext):
        """Restart a service."""
        service_name = action.parameters.get("service_name", error_context.service_name)
        if service_name:
            logger.info(f"Restarting service: {service_name}")
            # This would integrate with the system integration manager
    
    def _clear_cache(self, action: RecoveryAction, error_context: ErrorContext):
        """Clear cache."""
        cache_type = action.parameters.get("cache_type", "all")
        logger.info(f"Clearing cache: {cache_type}")
        # This would integrate with cache management
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and metrics."""
        if not self.error_history:
            return {
                "total_errors": 0,
                "by_category": {},
                "by_severity": {},
                "recent_errors": []
            }
        
        # Count by category
        by_category = {}
        for error in self.error_history:
            category = error.category.value
            by_category[category] = by_category.get(category, 0) + 1
        
        # Count by severity
        by_severity = {}
        for error in self.error_history:
            severity = error.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        # Recent errors (last 10)
        recent_errors = []
        for error in self.error_history[-10:]:
            recent_errors.append({
                "error_id": error.error_id,
                "timestamp": error.timestamp,
                "category": error.category.value,
                "severity": error.severity.value,
                "message": error.message,
                "service_name": error.service_name
            })
        
        return {
            "total_errors": len(self.error_history),
            "by_category": by_category,
            "by_severity": by_severity,
            "recent_errors": recent_errors
        }
    
    def clear_error_history(self):
        """Clear error history."""
        self.error_history.clear()
        logger.info("Error history cleared")


# Global error handler instance
error_handler = ErrorHandler()


def handle_exceptions(
    category: ErrorCategory,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    service_name: Optional[str] = None
):
    """Decorator for automatic exception handling."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler.handle_error(
                    exception=e,
                    category=category,
                    severity=severity,
                    service_name=service_name,
                    function_name=func.__name__
                )
                raise
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_handler.handle_error(
                    exception=e,
                    category=category,
                    severity=severity,
                    service_name=service_name,
                    function_name=func.__name__
                )
                raise
        
        return async_wrapper if hasattr(func, '__code__') and func.__code__.co_flags & 0x80 else wrapper
    
    return decorator


# Recovery handlers for different error categories
def database_recovery_handler(error_context: ErrorContext) -> Optional[RecoveryAction]:
    """Recovery handler for database errors."""
    if "connection" in error_context.message.lower():
        return RecoveryAction(
            action_type="retry",
            parameters={"delay": 5},
            max_retries=3
        )
    return None


def extraction_recovery_handler(error_context: ErrorContext) -> Optional[RecoveryAction]:
    """Recovery handler for data extraction errors."""
    if "timeout" in error_context.message.lower():
        return RecoveryAction(
            action_type="retry",
            parameters={"timeout": 60},
            max_retries=2
        )
    return None


def external_api_recovery_handler(error_context: ErrorContext) -> Optional[RecoveryAction]:
    """Recovery handler for external API errors."""
    if "rate limit" in error_context.message.lower():
        return RecoveryAction(
            action_type="retry",
            parameters={"delay": 60},
            max_retries=3,
            backoff_multiplier=2.0
        )
    return None


# Register default recovery handlers
error_handler.register_recovery_handler(ErrorCategory.DATABASE, database_recovery_handler)
error_handler.register_recovery_handler(ErrorCategory.EXTRACTION, extraction_recovery_handler)
error_handler.register_recovery_handler(ErrorCategory.EXTERNAL_API, external_api_recovery_handler)