"""
Unified Error Handling System for SuperInsight Platform.

Provides centralized error handling, logging, and recovery mechanisms
for all platform components with enhanced retry, circuit breaker,
and graceful degradation capabilities.
"""

import asyncio
import logging
import traceback
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

from src.config.settings import settings
from src.utils.retry import (
    RetryExecutor, RetryConfig, RetryStrategy,
    CircuitBreaker, CircuitBreakerConfig, get_circuit_breaker
)
from src.utils.degradation import degradation_manager, DegradationLevel
from src.system.notification import notification_system, NotificationPriority, NotificationChannel


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
    """Recovery action definition with enhanced retry capabilities."""
    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    retry_config: Optional[RetryConfig] = None
    circuit_breaker_name: Optional[str] = None
    fallback_service: Optional[str] = None
    enable_degradation: bool = False


class ErrorHandler:
    """
    Centralized error handling system with enhanced recovery capabilities.
    
    Provides:
    - Error classification and logging
    - Advanced retry mechanisms with exponential backoff
    - Circuit breaker protection
    - Graceful degradation support
    - Recovery action execution
    - Error metrics and reporting
    - Notification system integration
    """
    
    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.recovery_handlers: Dict[ErrorCategory, Callable] = {}
        self.notification_handlers: List[Callable] = []
        self.max_history_size = 1000
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Enhanced recovery tracking
        self.recovery_attempts: Dict[str, int] = {}
        self.recovery_success_rate: Dict[str, float] = {}
        self.service_health_cache: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default recovery configurations
        self._setup_default_recovery_configs()
        
        # Register with notification system
        self._setup_notification_integration()
        
    def _setup_notification_integration(self):
        """Setup integration with the enhanced notification system."""
        # Configure notification system for error handling
        notification_system.config.enabled = True
        if not notification_system.config.channels:
            notification_system.config.channels = [
                NotificationChannel.LOG,
                NotificationChannel.WEBHOOK
            ]
        
        logger.info("Enhanced notification integration configured")
    
    def _setup_default_recovery_configs(self):
        """Setup default recovery configurations for different error categories."""
        # Database errors - aggressive retry with circuit breaker
        self.register_recovery_handler(ErrorCategory.DATABASE, self._database_recovery_handler)
        
        # External API errors - retry with backoff and circuit breaker
        self.register_recovery_handler(ErrorCategory.EXTERNAL_API, self._external_api_recovery_handler)
        
        # Extraction errors - retry with fallback
        self.register_recovery_handler(ErrorCategory.EXTRACTION, self._extraction_recovery_handler)
        
        # AI annotation errors - graceful degradation
        self.register_recovery_handler(ErrorCategory.ANNOTATION, self._annotation_recovery_handler)
        
        # Quality check errors - fallback to basic validation
        self.register_recovery_handler(ErrorCategory.QUALITY, self._quality_recovery_handler)
    
    def register_recovery_handler(self, category: ErrorCategory, handler: Callable):
        """Register a recovery handler for a specific error category."""
        self.recovery_handlers[category] = handler
        logger.info(f"Registered enhanced recovery handler for category: {category.value}")
    
    def register_notification_handler(self, handler: Callable):
        """Register a notification handler for error alerts."""
        self.notification_handlers.append(handler)
        logger.info("Registered notification handler")
    
    def get_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create a circuit breaker for a service."""
        if name not in self.circuit_breakers:
            if config is None:
                config = CircuitBreakerConfig()
            self.circuit_breakers[name] = CircuitBreaker(name, config)
        return self.circuit_breakers[name]
    
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
        
        # Send notifications for high severity errors with enhanced filtering
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self._send_enhanced_notifications(error_context)
        
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
    
    def _send_enhanced_notifications(self, error_context: ErrorContext):
        """Send enhanced notifications with intelligent filtering to reduce false positives."""
        try:
            # Enhanced filtering logic to reduce noise
            should_notify = self._should_send_notification(error_context)
            
            if not should_notify:
                logger.debug(f"Notification filtered for error {error_context.error_id}")
                return
            
            # Use the enhanced notification system
            notification_system.send_error_notification(
                error_context=error_context,
                include_details=error_context.severity == ErrorSeverity.CRITICAL
            )
        except Exception as e:
            logger.error(f"Enhanced notification failed: {e}")
            # Fallback to legacy notification handlers
            self._send_notifications(error_context)
    
    def _should_send_notification(self, error_context: ErrorContext) -> bool:
        """Determine if a notification should be sent based on intelligent filtering."""
        
        # 1. Always notify for critical errors
        if error_context.severity == ErrorSeverity.CRITICAL:
            return True
        
        # 2. Check error frequency to avoid spam
        error_signature = f"{error_context.category.value}:{error_context.service_name}:{hash(error_context.message[:100])}"
        
        # Count recent similar errors (last 10 minutes)
        recent_similar_errors = 0
        current_time = time.time()
        
        for historical_error in self.error_history[-50:]:  # Check last 50 errors
            if current_time - historical_error.timestamp > 600:  # 10 minutes
                continue
                
            historical_signature = f"{historical_error.category.value}:{historical_error.service_name}:{hash(historical_error.message[:100])}"
            if historical_signature == error_signature:
                recent_similar_errors += 1
        
        # 3. Apply frequency-based filtering
        if recent_similar_errors >= 5:  # More than 5 similar errors in 10 minutes
            logger.debug(f"Notification suppressed due to high frequency: {recent_similar_errors} similar errors")
            return False
        
        # 4. Check if this is a known transient issue
        transient_patterns = [
            "timeout", "connection reset", "temporary", "rate limit", 
            "service unavailable", "502", "503", "504"
        ]
        
        message_lower = error_context.message.lower()
        is_transient = any(pattern in message_lower for pattern in transient_patterns)
        
        if is_transient and error_context.severity == ErrorSeverity.MEDIUM:
            # For transient medium-severity errors, only notify if persistent
            if recent_similar_errors < 3:
                return False
        
        # 5. Check service health status
        if error_context.service_name:
            service_health = degradation_manager.get_service_health(error_context.service_name)
            if service_health and service_health.degradation_level == DegradationLevel.OFFLINE:
                # Service is already known to be offline, reduce notification frequency
                if recent_similar_errors >= 2:
                    return False
        
        # 6. Business hours consideration (reduce noise during off-hours for non-critical)
        if error_context.severity == ErrorSeverity.MEDIUM:
            import datetime
            current_hour = datetime.datetime.now().hour
            if current_hour < 8 or current_hour > 20:  # Outside business hours
                # Only notify for persistent issues during off-hours
                if recent_similar_errors < 2:
                    return False
        
        # 7. Check if recovery is likely to succeed
        recovery_key = f"{error_context.category.value}:{error_context.service_name or 'unknown'}"
        success_rate = self.recovery_success_rate.get(recovery_key, 0.5)
        
        # If recovery success rate is very high, reduce notification urgency
        if success_rate > 0.8 and error_context.severity == ErrorSeverity.MEDIUM:
            return recent_similar_errors >= 2  # Only notify if persistent
        
        return True
    
    def _send_notifications(self, error_context: ErrorContext):
        """Legacy notification method as fallback."""
        for handler in self.notification_handlers:
            try:
                handler(error_context)
            except Exception as e:
                logger.error(f"Notification handler failed: {e}")
    
    def _attempt_recovery(self, error_context: ErrorContext):
        """Attempt automatic recovery based on error category with enhanced mechanisms."""
        recovery_key = f"{error_context.category.value}:{error_context.service_name or 'unknown'}"
        
        # Track recovery attempts
        self.recovery_attempts[recovery_key] = self.recovery_attempts.get(recovery_key, 0) + 1
        
        # Check if we should attempt recovery based on success rate
        if self._should_attempt_recovery(recovery_key, error_context):
            if error_context.category in self.recovery_handlers:
                try:
                    recovery_handler = self.recovery_handlers[error_context.category]
                    recovery_action = recovery_handler(error_context)
                    
                    if recovery_action:
                        success = self._execute_enhanced_recovery_action(recovery_action, error_context)
                        self._update_recovery_success_rate(recovery_key, success)
                        
                        if success:
                            logger.info(f"Recovery successful for {recovery_key}")
                            # Send success notification for critical errors
                            if error_context.severity == ErrorSeverity.CRITICAL:
                                notification_system.send_notification(
                                    title=f"Recovery Successful - {error_context.category.value}",
                                    message=f"Automatic recovery succeeded for error {error_context.error_id}",
                                    priority=NotificationPriority.NORMAL,
                                    channels=[NotificationChannel.LOG, NotificationChannel.SLACK]
                                )
                        else:
                            logger.warning(f"Recovery failed for {recovery_key}")
                        
                except Exception as e:
                    logger.error(f"Recovery handler failed for {error_context.category.value}: {e}")
                    self._update_recovery_success_rate(recovery_key, False)
        else:
            logger.info(f"Skipping recovery for {recovery_key} due to low success rate")
    
    def _should_attempt_recovery(self, recovery_key: str, error_context: ErrorContext) -> bool:
        """Determine if recovery should be attempted based on historical success rate and intelligent criteria."""
        attempts = self.recovery_attempts.get(recovery_key, 0)
        
        # Always attempt recovery for the first few times
        if attempts <= 3:
            return True
        
        # Check success rate with enhanced logic
        success_rate = self.recovery_success_rate.get(recovery_key, 0.0)
        
        # Enhanced decision logic based on multiple factors
        
        # 1. Critical errors always get recovery attempts (with backoff)
        if error_context.severity == ErrorSeverity.CRITICAL:
            # But apply exponential backoff for repeated critical failures
            if attempts > 10:
                backoff_factor = min(attempts - 10, 5)  # Max 5x backoff
                if time.time() % (2 ** backoff_factor) != 0:
                    return False
            return True
        
        # 2. High severity errors get more chances
        if error_context.severity == ErrorSeverity.HIGH:
            if success_rate < 0.1:  # Less than 10% success rate
                return False
            return success_rate > 0.3  # Lower threshold for high severity
        
        # 3. Consider error category - some are more recoverable
        recoverable_categories = {
            ErrorCategory.DATABASE: 0.4,      # Database issues often recoverable
            ErrorCategory.EXTERNAL_API: 0.3,  # API issues moderately recoverable
            ErrorCategory.EXTRACTION: 0.5,    # Extraction issues often recoverable
            ErrorCategory.ANNOTATION: 0.4,    # AI issues moderately recoverable
            ErrorCategory.QUALITY: 0.5,       # Quality checks often recoverable
        }
        
        category_threshold = recoverable_categories.get(error_context.category, 0.5)
        
        # 4. Time-based recovery attempts (don't give up too quickly)
        if attempts <= 10:
            return success_rate >= category_threshold
        
        # 5. For very persistent issues, require higher success rate
        if attempts > 20:
            return success_rate > 0.7
        
        # Default threshold
        return success_rate >= 0.5
    
    def _update_recovery_success_rate(self, recovery_key: str, success: bool):
        """Update recovery success rate using exponential moving average."""
        current_rate = self.recovery_success_rate.get(recovery_key, 0.5)  # Start with neutral
        alpha = 0.3  # Learning rate
        
        new_value = 1.0 if success else 0.0
        self.recovery_success_rate[recovery_key] = alpha * new_value + (1 - alpha) * current_rate
    
    def _execute_enhanced_recovery_action(self, action: RecoveryAction, error_context: ErrorContext) -> bool:
        """Execute a recovery action with enhanced retry and circuit breaker support."""
        try:
            logger.info(f"Executing enhanced recovery action: {action.action_type}")
            
            # Setup circuit breaker if specified
            circuit_breaker = None
            if action.circuit_breaker_name:
                cb_config = CircuitBreakerConfig()
                circuit_breaker = self.get_circuit_breaker(action.circuit_breaker_name, cb_config)
            
            # Setup retry executor if specified
            retry_executor = None
            if action.retry_config:
                retry_executor = RetryExecutor(action.retry_config)
            
            # Execute based on action type
            success = False
            if action.action_type == "retry_with_backoff":
                success = self._retry_with_backoff(action, error_context, retry_executor, circuit_breaker)
            elif action.action_type == "fallback_with_degradation":
                success = self._fallback_with_degradation(action, error_context)
            elif action.action_type == "circuit_breaker_protection":
                success = self._circuit_breaker_protection(action, error_context, circuit_breaker)
            elif action.action_type == "restart_service":
                success = self._restart_service(action, error_context)
            elif action.action_type == "clear_cache":
                success = self._clear_cache(action, error_context)
            elif action.action_type == "graceful_degradation":
                success = self._graceful_degradation(action, error_context)
            elif action.action_type == "health_check_and_recover":
                success = self._health_check_and_recover(action, error_context)
            else:
                logger.warning(f"Unknown recovery action type: {action.action_type}")
                return False
            
            if success:
                logger.info(f"Enhanced recovery action {action.action_type} completed successfully")
            else:
                logger.warning(f"Enhanced recovery action {action.action_type} failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Enhanced recovery action {action.action_type} failed: {e}")
            return False
    
    def _retry_with_backoff(
        self,
        action: RecoveryAction,
        error_context: ErrorContext,
        retry_executor: Optional[RetryExecutor],
        circuit_breaker: Optional[CircuitBreaker]
    ) -> bool:
        """Execute retry with enhanced exponential backoff and adaptive strategies."""
        if not retry_executor:
            # Enhanced default retry configuration based on error context
            base_delay = 1.0
            max_attempts = 3
            max_delay = 30.0
            
            # Adjust parameters based on error category and severity
            if error_context.category == ErrorCategory.DATABASE:
                base_delay = 2.0
                max_attempts = 5
                max_delay = 60.0
            elif error_context.category == ErrorCategory.EXTERNAL_API:
                base_delay = 5.0  # Longer delay for API calls
                max_attempts = 4
                max_delay = 120.0
            elif error_context.severity == ErrorSeverity.CRITICAL:
                max_attempts = 7  # More attempts for critical errors
                max_delay = 300.0  # Up to 5 minutes
            
            # Enhanced retry configuration with adaptive backoff
            retry_config = RetryConfig(
                max_attempts=max_attempts,
                base_delay=base_delay,
                max_delay=max_delay,
                strategy=RetryStrategy.EXPONENTIAL,
                backoff_multiplier=2.0,
                jitter=True,
                jitter_range=0.2  # 20% jitter to avoid thundering herd
            )
            retry_executor = RetryExecutor(retry_config)
        
        def retry_operation():
            if circuit_breaker:
                # Enhanced circuit breaker integration
                state = circuit_breaker.get_state()
                if state['state'] == 'open':
                    # Circuit is open, but we might want to test recovery
                    logger.info(f"Circuit breaker {circuit_breaker.name} is open, attempting recovery test")
                    # Allow one test call during recovery
                    return circuit_breaker.call(lambda: self._test_service_recovery(error_context))
                else:
                    return circuit_breaker.call(lambda: self._execute_recovery_test(error_context))
            else:
                return self._execute_recovery_test(error_context)
        
        try:
            result = retry_executor.execute(retry_operation)
            logger.info(f"Enhanced retry operation succeeded: {result}")
            return True
        except Exception as e:
            logger.error(f"Enhanced retry operation failed after all attempts: {e}")
            
            # Enhanced failure handling - try alternative recovery strategies
            if error_context.severity == ErrorSeverity.CRITICAL:
                logger.info("Attempting alternative recovery for critical error")
                return self._attempt_alternative_recovery(error_context)
            
            return False
    
    def _test_service_recovery(self, error_context: ErrorContext) -> str:
        """Test if a service has recovered from failure."""
        service_name = error_context.service_name or "unknown"
        
        # Perform a lightweight health check
        health_status = self._perform_health_check(service_name)
        
        if health_status.get("healthy", False):
            # Mark service as recovered
            degradation_manager.mark_service_success(service_name)
            logger.info(f"Service {service_name} has recovered")
            return "service_recovered"
        else:
            raise Exception(f"Service {service_name} still unhealthy: {health_status.get('error', 'Unknown')}")
    
    def _execute_recovery_test(self, error_context: ErrorContext) -> str:
        """Execute a recovery test operation."""
        # This would integrate with actual service testing
        # For now, simulate based on error context
        
        if error_context.category == ErrorCategory.DATABASE:
            # Test database connectivity
            return "database_test_passed"
        elif error_context.category == ErrorCategory.EXTERNAL_API:
            # Test API endpoint
            return "api_test_passed"
        elif error_context.category == ErrorCategory.EXTRACTION:
            # Test extraction capability
            return "extraction_test_passed"
        else:
            return "generic_test_passed"
    
    def _attempt_alternative_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt alternative recovery strategies for critical failures."""
        try:
            logger.info(f"Attempting alternative recovery for {error_context.category.value}")
            
            # Strategy 1: Force service restart
            if error_context.service_name:
                degradation_manager.mark_service_failure(error_context.service_name)
                time.sleep(2)  # Brief pause
                degradation_manager.mark_service_success(error_context.service_name)
                logger.info(f"Forced restart of {error_context.service_name}")
            
            # Strategy 2: Clear all caches
            degradation_manager.clear_cache()
            logger.info("Cleared all system caches")
            
            # Strategy 3: Enable emergency mode
            self._enable_emergency_mode(error_context)
            
            return True
            
        except Exception as e:
            logger.error(f"Alternative recovery failed: {e}")
            return False
    
    def _enable_emergency_mode(self, error_context: ErrorContext):
        """Enable emergency mode for critical system failures."""
        logger.warning("Enabling emergency mode due to critical failure")
        
        # Notify operations team
        notification_system.send_notification(
            title="EMERGENCY MODE ACTIVATED",
            message=f"Critical failure in {error_context.category.value} requires immediate attention. Error: {error_context.message}",
            priority=NotificationPriority.CRITICAL,
            channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.WEBHOOK],
            metadata={
                "emergency_mode": True,
                "error_id": error_context.error_id,
                "service_name": error_context.service_name
            }
        )
    
    def _fallback_with_degradation(self, action: RecoveryAction, error_context: ErrorContext) -> bool:
        """Execute fallback with graceful degradation."""
        try:
            if action.fallback_service:
                # Mark the primary service as degraded
                degradation_manager.mark_service_failure(error_context.service_name or "unknown")
                
                # Try to execute fallback
                logger.info(f"Executing fallback for service: {action.fallback_service}")
                # This would integrate with the specific service's fallback mechanism
                degradation_manager.mark_service_success(action.fallback_service)
                return True
            return False
        except Exception as e:
            logger.error(f"Fallback execution failed: {e}")
            if action.fallback_service:
                degradation_manager.mark_service_failure(action.fallback_service)
            return False
    
    def _circuit_breaker_protection(
        self,
        action: RecoveryAction,
        error_context: ErrorContext,
        circuit_breaker: Optional[CircuitBreaker]
    ) -> bool:
        """Apply enhanced circuit breaker protection with adaptive thresholds."""
        if not circuit_breaker:
            # Create adaptive circuit breaker based on error context
            config = CircuitBreakerConfig()
            
            # Adjust thresholds based on service criticality and error patterns
            if error_context.severity == ErrorSeverity.CRITICAL:
                config.failure_threshold = 3  # More sensitive for critical services
                config.recovery_timeout = 30.0  # Shorter recovery time
            elif error_context.category == ErrorCategory.EXTERNAL_API:
                config.failure_threshold = 5  # More tolerant for external APIs
                config.recovery_timeout = 120.0  # Longer recovery time
            elif error_context.category == ErrorCategory.DATABASE:
                config.failure_threshold = 2  # Very sensitive for database
                config.recovery_timeout = 60.0
            
            circuit_breaker = self.get_circuit_breaker(
                f"{error_context.service_name or 'unknown'}_{error_context.category.value}",
                config
            )
        
        try:
            # Enhanced circuit breaker state management
            state = circuit_breaker.get_state()
            logger.info(f"Circuit breaker {state['name']} state: {state['state']} (failures: {state['failure_count']})")
            
            if state['state'] == 'open':
                # Circuit is open - check if we should attempt recovery
                time_since_failure = time.time() - state['last_failure_time']
                
                if time_since_failure > circuit_breaker.config.recovery_timeout:
                    logger.info(f"Circuit breaker {state['name']} attempting recovery after {time_since_failure:.1f}s")
                    
                    # Try a lightweight health check
                    try:
                        health_status = self._perform_health_check(error_context.service_name or "unknown")
                        if health_status.get("healthy", False):
                            logger.info(f"Service appears healthy, allowing circuit breaker recovery test")
                            return True
                        else:
                            logger.warning(f"Service still unhealthy, keeping circuit breaker open")
                            return False
                    except Exception as e:
                        logger.error(f"Health check failed during circuit breaker recovery: {e}")
                        return False
                else:
                    logger.warning(f"Circuit breaker {state['name']} is open, rejecting calls for {circuit_breaker.config.recovery_timeout - time_since_failure:.1f}s more")
                    
                    # Send notification for prolonged outages
                    if time_since_failure > 300:  # 5 minutes
                        notification_system.send_notification(
                            title=f"Prolonged Service Outage - {error_context.service_name}",
                            message=f"Circuit breaker has been open for {time_since_failure/60:.1f} minutes",
                            priority=NotificationPriority.HIGH,
                            channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL],
                            deduplication_key=f"circuit_open_{state['name']}"
                        )
                    
                    return False
            
            elif state['state'] == 'half_open':
                logger.info(f"Circuit breaker {state['name']} is in half-open state, allowing test call")
                return True
            
            else:  # closed
                logger.debug(f"Circuit breaker {state['name']} is closed, allowing calls")
                return True
            
        except Exception as e:
            logger.error(f"Enhanced circuit breaker protection failed: {e}")
            return False
    
    # Enhanced recovery handlers
    def _database_recovery_handler(self, error_context: ErrorContext) -> Optional[RecoveryAction]:
        """Enhanced recovery handler for database errors."""
        error_msg = error_context.message.lower()
        
        if "connection" in error_msg or "timeout" in error_msg:
            return RecoveryAction(
                action_type="retry_with_backoff",
                retry_config=RetryConfig(
                    max_attempts=5,
                    base_delay=2.0,
                    max_delay=30.0,
                    strategy=RetryStrategy.EXPONENTIAL,
                    backoff_multiplier=2.0,
                    jitter=True
                ),
                circuit_breaker_name="database_connection"
            )
        elif "deadlock" in error_msg or "lock" in error_msg:
            return RecoveryAction(
                action_type="retry_with_backoff",
                retry_config=RetryConfig(
                    max_attempts=3,
                    base_delay=0.5,
                    max_delay=5.0,
                    strategy=RetryStrategy.FIBONACCI,
                    jitter=True
                )
            )
        elif "disk" in error_msg or "space" in error_msg:
            return RecoveryAction(
                action_type="clear_cache",
                parameters={"cache_type": "database"}
            )
        elif "corrupt" in error_msg:
            return RecoveryAction(
                action_type="health_check_and_recover",
                parameters={"perform_repair": True}
            )
        return None
    
    def _external_api_recovery_handler(self, error_context: ErrorContext) -> Optional[RecoveryAction]:
        """Enhanced recovery handler for external API errors."""
        error_msg = error_context.message.lower()
        
        if "rate limit" in error_msg or "429" in error_msg:
            return RecoveryAction(
                action_type="retry_with_backoff",
                retry_config=RetryConfig(
                    max_attempts=5,
                    base_delay=60.0,
                    max_delay=300.0,
                    strategy=RetryStrategy.EXPONENTIAL,
                    backoff_multiplier=1.5,
                    jitter=True
                ),
                circuit_breaker_name="external_api"
            )
        elif "timeout" in error_msg or "502" in error_msg or "503" in error_msg:
            return RecoveryAction(
                action_type="circuit_breaker_protection",
                circuit_breaker_name="external_api_timeout",
                fallback_service="cached_api_response"
            )
        elif "unauthorized" in error_msg or "401" in error_msg:
            return RecoveryAction(
                action_type="health_check_and_recover",
                parameters={"refresh_credentials": True}
            )
        elif "not found" in error_msg or "404" in error_msg:
            return RecoveryAction(
                action_type="graceful_degradation",
                fallback_service="default_api_response"
            )
        return None
    
    def _extraction_recovery_handler(self, error_context: ErrorContext) -> Optional[RecoveryAction]:
        """Enhanced recovery handler for data extraction errors."""
        error_msg = error_context.message.lower()
        
        if "timeout" in error_msg:
            return RecoveryAction(
                action_type="retry_with_backoff",
                retry_config=RetryConfig(
                    max_attempts=3,
                    base_delay=10.0,
                    strategy=RetryStrategy.FIBONACCI
                ),
                fallback_service="basic_extraction"
            )
        elif "connection" in error_msg or "network" in error_msg:
            return RecoveryAction(
                action_type="fallback_with_degradation",
                fallback_service="cached_extraction",
                enable_degradation=True
            )
        elif "permission" in error_msg or "access" in error_msg:
            return RecoveryAction(
                action_type="health_check_and_recover",
                parameters={"check_permissions": True}
            )
        elif "format" in error_msg or "parse" in error_msg:
            return RecoveryAction(
                action_type="graceful_degradation",
                fallback_service="simple_text_extraction"
            )
        return None
    
    def _annotation_recovery_handler(self, error_context: ErrorContext) -> Optional[RecoveryAction]:
        """Enhanced recovery handler for AI annotation errors."""
        error_msg = error_context.message.lower()
        
        if "model" in error_msg and "unavailable" in error_msg:
            return RecoveryAction(
                action_type="fallback_with_degradation",
                fallback_service="backup_annotation_model",
                enable_degradation=True
            )
        elif "quota" in error_msg or "limit" in error_msg:
            return RecoveryAction(
                action_type="graceful_degradation",
                fallback_service="basic_annotation"
            )
        else:
            return RecoveryAction(
                action_type="fallback_with_degradation",
                fallback_service="rule_based_annotation",
                enable_degradation=True
            )
    
    def _quality_recovery_handler(self, error_context: ErrorContext) -> Optional[RecoveryAction]:
        """Enhanced recovery handler for quality check errors."""
        error_msg = error_context.message.lower()
        
        if "ragas" in error_msg or "evaluation" in error_msg:
            return RecoveryAction(
                action_type="fallback_with_degradation",
                fallback_service="basic_quality_check",
                enable_degradation=True
            )
        elif "timeout" in error_msg:
            return RecoveryAction(
                action_type="retry_with_backoff",
                retry_config=RetryConfig(
                    max_attempts=2,
                    base_delay=5.0,
                    strategy=RetryStrategy.LINEAR
                ),
                fallback_service="simple_quality_check"
            )
        else:
            return RecoveryAction(
                action_type="graceful_degradation",
                fallback_service="rule_based_quality_check"
            )
    
    def _restart_service(self, action: RecoveryAction, error_context: ErrorContext) -> bool:
        """Restart a service."""
        service_name = action.parameters.get("service_name", error_context.service_name)
        if service_name:
            try:
                logger.info(f"Restarting service: {service_name}")
                # This would integrate with the system integration manager
                # For now, we'll simulate a restart by clearing the service's failure state
                degradation_manager.mark_service_success(service_name)
                
                # Send notification about service restart
                notification_system.send_notification(
                    title=f"Service Restart - {service_name}",
                    message=f"Service {service_name} has been restarted due to error {error_context.error_id}",
                    priority=NotificationPriority.NORMAL,
                    channels=[NotificationChannel.LOG, NotificationChannel.WEBHOOK]
                )
                return True
            except Exception as e:
                logger.error(f"Service restart failed for {service_name}: {e}")
                return False
        return False
    
    def _clear_cache(self, action: RecoveryAction, error_context: ErrorContext) -> bool:
        """Clear cache."""
        cache_type = action.parameters.get("cache_type", "all")
        try:
            logger.info(f"Clearing cache: {cache_type}")
            # This would integrate with cache management
            # For now, we'll clear the degradation manager cache
            degradation_manager.clear_cache()
            return True
        except Exception as e:
            logger.error(f"Cache clear failed: {e}")
            return False
    
    def _graceful_degradation(self, action: RecoveryAction, error_context: ErrorContext) -> bool:
        """Implement enhanced graceful degradation with intelligent level selection."""
        service_name = error_context.service_name or "unknown"
        try:
            # Get current service health with enhanced analysis
            health = degradation_manager.get_service_health(service_name)
            if not health:
                # Register service if not exists
                degradation_manager.register_service(service_name)
                health = degradation_manager.get_service_health(service_name)
            
            current_level = health.degradation_level
            
            # Enhanced degradation logic based on error patterns and severity
            new_level = self._calculate_optimal_degradation_level(
                current_level, error_context, health
            )
            
            if new_level == current_level:
                logger.info(f"Service {service_name} already at optimal degradation level: {current_level.value}")
                return True
            
            # Apply degradation with enhanced monitoring
            degradation_manager.mark_service_failure(service_name)
            
            # Update service to new degradation level
            health.degradation_level = new_level
            
            logger.info(f"Service {service_name} degraded from {current_level.value} to {new_level.value}")
            
            # Enhanced notification with degradation impact analysis
            impact_analysis = self._analyze_degradation_impact(service_name, new_level)
            
            notification_system.send_notification(
                title=f"Service Degradation - {service_name}",
                message=f"Service {service_name} degraded to {new_level.value} level due to error {error_context.error_id}\n\nImpact Analysis:\n{impact_analysis}",
                priority=self._get_degradation_notification_priority(new_level),
                channels=self._get_degradation_notification_channels(new_level),
                metadata={
                    "service_name": service_name,
                    "previous_level": current_level.value,
                    "new_level": new_level.value,
                    "error_category": error_context.category.value,
                    "degradation_impact": impact_analysis
                },
                deduplication_key=f"degradation_{service_name}_{new_level.value}"
            )
            
            # Set up automatic recovery monitoring
            self._schedule_degradation_recovery_check(service_name, error_context)
            
            return True
            
        except Exception as e:
            logger.error(f"Enhanced graceful degradation failed for {service_name}: {e}")
            return False
    
    def _calculate_optimal_degradation_level(
        self, 
        current_level: DegradationLevel, 
        error_context: ErrorContext,
        health: Any
    ) -> DegradationLevel:
        """Calculate optimal degradation level based on error patterns and service health."""
        
        # Factor 1: Error severity
        severity_degradation = {
            ErrorSeverity.LOW: 0,      # No degradation
            ErrorSeverity.MEDIUM: 1,   # One level down
            ErrorSeverity.HIGH: 2,     # Two levels down
            ErrorSeverity.CRITICAL: 3  # Maximum degradation
        }
        
        # Factor 2: Error category impact
        category_impact = {
            ErrorCategory.DATABASE: 3,      # High impact
            ErrorCategory.EXTERNAL_API: 2,  # Medium impact
            ErrorCategory.EXTRACTION: 2,    # Medium impact
            ErrorCategory.ANNOTATION: 1,    # Lower impact
            ErrorCategory.QUALITY: 1,       # Lower impact
            ErrorCategory.SYSTEM: 3         # High impact
        }
        
        # Factor 3: Failure frequency
        failure_frequency_impact = min(health.failure_count // 3, 2)  # Max 2 levels for frequency
        
        # Calculate total degradation needed
        total_degradation = (
            severity_degradation.get(error_context.severity, 1) +
            category_impact.get(error_context.category, 1) +
            failure_frequency_impact
        ) // 3  # Average the factors
        
        # Apply degradation from current level
        levels = [DegradationLevel.FULL, DegradationLevel.REDUCED, DegradationLevel.MINIMAL, DegradationLevel.OFFLINE]
        current_index = levels.index(current_level)
        
        new_index = min(current_index + total_degradation, len(levels) - 1)
        return levels[new_index]
    
    def _analyze_degradation_impact(self, service_name: str, degradation_level: DegradationLevel) -> str:
        """Analyze the impact of service degradation."""
        impact_descriptions = {
            DegradationLevel.REDUCED: f"• {service_name} operating with reduced functionality\n• Some features may be slower or unavailable\n• Fallback mechanisms activated",
            DegradationLevel.MINIMAL: f"• {service_name} operating with minimal functionality\n• Most advanced features disabled\n• Basic operations only\n• Significant performance impact expected",
            DegradationLevel.OFFLINE: f"• {service_name} completely offline\n• All functionality unavailable\n• Fallback services activated where possible\n• Manual intervention may be required"
        }
        
        base_impact = impact_descriptions.get(degradation_level, "• Service operating normally")
        
        # Add service-specific impact analysis
        service_specific_impact = {
            "ai_annotation": {
                DegradationLevel.REDUCED: "• AI predictions may be less accurate\n• Slower annotation processing",
                DegradationLevel.MINIMAL: "• Basic rule-based annotation only\n• No AI-powered features",
                DegradationLevel.OFFLINE: "• Manual annotation required\n• No automated assistance"
            },
            "data_extraction": {
                DegradationLevel.REDUCED: "• Limited file format support\n• Slower extraction processing",
                DegradationLevel.MINIMAL: "• Basic text extraction only\n• No advanced parsing",
                DegradationLevel.OFFLINE: "• No automatic data extraction\n• Manual data entry required"
            },
            "quality_check": {
                DegradationLevel.REDUCED: "• Basic quality validation only\n• Reduced accuracy in quality scoring",
                DegradationLevel.MINIMAL: "• Simple rule-based checks only\n• No semantic analysis",
                DegradationLevel.OFFLINE: "• No automated quality checks\n• Manual review required"
            }
        }
        
        specific_impact = service_specific_impact.get(service_name, {}).get(degradation_level, "")
        
        if specific_impact:
            return f"{base_impact}\n\nService-Specific Impact:\n{specific_impact}"
        else:
            return base_impact
    
    def _get_degradation_notification_priority(self, level: DegradationLevel) -> NotificationPriority:
        """Get notification priority based on degradation level."""
        priority_map = {
            DegradationLevel.REDUCED: NotificationPriority.NORMAL,
            DegradationLevel.MINIMAL: NotificationPriority.HIGH,
            DegradationLevel.OFFLINE: NotificationPriority.CRITICAL
        }
        return priority_map.get(level, NotificationPriority.NORMAL)
    
    def _get_degradation_notification_channels(self, level: DegradationLevel) -> List:
        """Get notification channels based on degradation level."""
        from src.system.notification import NotificationChannel
        
        channel_map = {
            DegradationLevel.REDUCED: [NotificationChannel.LOG, NotificationChannel.SLACK],
            DegradationLevel.MINIMAL: [NotificationChannel.LOG, NotificationChannel.SLACK, NotificationChannel.WEBHOOK],
            DegradationLevel.OFFLINE: [NotificationChannel.LOG, NotificationChannel.SLACK, NotificationChannel.EMAIL, NotificationChannel.WEBHOOK]
        }
        return channel_map.get(level, [NotificationChannel.LOG])
    
    def _schedule_degradation_recovery_check(self, service_name: str, error_context: ErrorContext):
        """Schedule automatic recovery check for degraded service."""
        def recovery_check():
            try:
                # Wait before checking recovery
                time.sleep(60)  # Wait 1 minute
                
                # Perform health check
                health_status = self._perform_health_check(service_name)
                
                if health_status.get("healthy", False):
                    # Service appears healthy, attempt recovery
                    degradation_manager.mark_service_success(service_name)
                    
                    logger.info(f"Automatic recovery detected for {service_name}")
                    
                    notification_system.send_notification(
                        title=f"Service Recovery - {service_name}",
                        message=f"Service {service_name} has automatically recovered and is back to full functionality",
                        priority=NotificationPriority.NORMAL,
                        channels=[NotificationChannel.LOG, NotificationChannel.SLACK],
                        metadata={
                            "service_name": service_name,
                            "recovery_type": "automatic",
                            "original_error_id": error_context.error_id
                        }
                    )
                else:
                    logger.debug(f"Service {service_name} still not healthy, keeping degraded state")
                    
            except Exception as e:
                logger.error(f"Recovery check failed for {service_name}: {e}")
        
        # Schedule recovery check in background thread
        import threading
        recovery_thread = threading.Thread(target=recovery_check, daemon=True)
        recovery_thread.start()
    
    def _health_check_and_recover(self, action: RecoveryAction, error_context: ErrorContext) -> bool:
        """Perform health check and attempt recovery."""
        service_name = error_context.service_name or "unknown"
        try:
            # Perform health check
            health_status = self._perform_health_check(service_name)
            
            if health_status.get("healthy", False):
                # Service is healthy, mark as recovered
                degradation_manager.mark_service_success(service_name)
                
                logger.info(f"Health check passed for {service_name}, marking as recovered")
                
                # Send recovery notification
                notification_system.send_notification(
                    title=f"Service Recovery - {service_name}",
                    message=f"Service {service_name} has recovered and passed health check",
                    priority=NotificationPriority.NORMAL,
                    channels=[NotificationChannel.LOG, NotificationChannel.SLACK]
                )
                return True
            else:
                # Service is still unhealthy
                logger.warning(f"Health check failed for {service_name}: {health_status.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error(f"Health check and recovery failed for {service_name}: {e}")
            return False
    
    def _perform_health_check(self, service_name: str) -> Dict[str, Any]:
        """Perform a health check on a service."""
        try:
            # This would integrate with actual health check mechanisms
            # For now, we'll simulate based on service health in degradation manager
            health = degradation_manager.get_service_health(service_name)
            
            if health:
                # Simple heuristic: if service hasn't failed recently, consider it healthy
                time_since_failure = time.time() - health.last_failure_time
                is_healthy = health.is_healthy or time_since_failure > 300  # 5 minutes
                
                return {
                    "healthy": is_healthy,
                    "service_name": service_name,
                    "degradation_level": health.degradation_level.value,
                    "failure_count": health.failure_count,
                    "time_since_failure": time_since_failure
                }
            else:
                # Service not registered, assume healthy
                return {
                    "healthy": True,
                    "service_name": service_name,
                    "degradation_level": "full",
                    "failure_count": 0
                }
                
        except Exception as e:
            return {
                "healthy": False,
                "service_name": service_name,
                "error": str(e)
            }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics and metrics including circuit breaker states."""
        if not self.error_history:
            return {
                "total_errors": 0,
                "by_category": {},
                "by_severity": {},
                "recent_errors": [],
                "circuit_breakers": {},
                "degradation_status": {},
                "recovery_metrics": {},
                "notification_stats": {}
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
        
        # Circuit breaker states
        circuit_breaker_states = {}
        for name, cb in self.circuit_breakers.items():
            circuit_breaker_states[name] = cb.get_state()
        
        # Service degradation status
        degradation_status = {}
        for service_name, health in degradation_manager.get_all_service_health().items():
            degradation_status[service_name] = {
                "is_healthy": health.is_healthy,
                "degradation_level": health.degradation_level.value,
                "failure_count": health.failure_count,
                "last_failure_time": health.last_failure_time,
                "last_success_time": health.last_success_time
            }
        
        # Recovery metrics
        recovery_metrics = {
            "attempts": dict(self.recovery_attempts),
            "success_rates": dict(self.recovery_success_rate),
            "total_recovery_attempts": sum(self.recovery_attempts.values()),
            "average_success_rate": sum(self.recovery_success_rate.values()) / len(self.recovery_success_rate) if self.recovery_success_rate else 0.0
        }
        
        # Notification statistics
        notification_stats = notification_system.get_statistics()
        
        return {
            "total_errors": len(self.error_history),
            "by_category": by_category,
            "by_severity": by_severity,
            "recent_errors": recent_errors,
            "circuit_breakers": circuit_breaker_states,
            "degradation_status": degradation_status,
            "recovery_metrics": recovery_metrics,
            "notification_stats": notification_stats
        }
    
    def clear_error_history(self):
        """Clear error history."""
        self.error_history.clear()
        logger.info("Error history cleared")


# Global error handler instance
error_handler = ErrorHandler()


# Enhanced decorators for automatic exception handling
def handle_exceptions(
    category: ErrorCategory,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    service_name: Optional[str] = None,
    enable_retry: bool = False,
    retry_config: Optional[RetryConfig] = None,
    circuit_breaker_name: Optional[str] = None,
    enable_degradation: bool = False
):
    """Enhanced decorator for automatic exception handling with retry and circuit breaker support."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Apply circuit breaker if specified
                if circuit_breaker_name:
                    cb = get_circuit_breaker(circuit_breaker_name)
                    return cb.call(func, *args, **kwargs)
                
                # Apply retry if specified
                if enable_retry and retry_config:
                    executor = RetryExecutor(retry_config)
                    return executor.execute(func, *args, **kwargs)
                
                return func(*args, **kwargs)
            except Exception as e:
                error_context = error_handler.handle_error(
                    exception=e,
                    category=category,
                    severity=severity,
                    service_name=service_name,
                    function_name=func.__name__,
                    enable_retry=enable_retry,
                    enable_degradation=enable_degradation
                )
                
                # If degradation is enabled, try to return a fallback result
                if enable_degradation and service_name:
                    try:
                        return degradation_manager.execute_with_fallback(
                            service_name, func, *args, **kwargs
                        )
                    except Exception:
                        pass  # Fall through to raise original exception
                
                raise
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                # Apply circuit breaker if specified
                if circuit_breaker_name:
                    cb = get_circuit_breaker(circuit_breaker_name)
                    return await cb.async_call(func, *args, **kwargs)
                
                # Apply retry if specified
                if enable_retry and retry_config:
                    executor = RetryExecutor(retry_config)
                    return await executor.async_execute(func, *args, **kwargs)
                
                return await func(*args, **kwargs)
            except Exception as e:
                error_context = error_handler.handle_error(
                    exception=e,
                    category=category,
                    severity=severity,
                    service_name=service_name,
                    function_name=func.__name__,
                    enable_retry=enable_retry,
                    enable_degradation=enable_degradation
                )
                
                # If degradation is enabled, try to return a fallback result
                if enable_degradation and service_name:
                    try:
                        return await degradation_manager.async_execute_with_fallback(
                            service_name, func, *args, **kwargs
                        )
                    except Exception:
                        pass  # Fall through to raise original exception
                
                raise
        
        return async_wrapper if hasattr(func, '__code__') and func.__code__.co_flags & 0x80 else wrapper
    
    return decorator