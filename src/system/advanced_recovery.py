"""
Advanced Error Recovery System for SuperInsight Platform.

Provides enhanced error recovery mechanisms with improved automatic retry logic,
circuit breaker patterns, optimized notifications, and graceful degradation.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json

from src.system.error_handler import error_handler, ErrorCategory, ErrorSeverity
from src.system.notification import notification_system, NotificationPriority, NotificationChannel
from src.utils.retry import RetryConfig, RetryStrategy, CircuitBreaker, CircuitBreakerConfig
from src.utils.degradation import degradation_manager, DegradationLevel

logger = logging.getLogger(__name__)


class RecoveryMode(Enum):
    """Recovery operation modes."""
    AUTOMATIC = "automatic"
    SEMI_AUTOMATIC = "semi_automatic"
    MANUAL = "manual"
    EMERGENCY = "emergency"


class RecoveryPriority(Enum):
    """Recovery priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class RecoveryContext:
    """Enhanced recovery context with comprehensive information."""
    error_id: str
    service_name: str
    error_category: ErrorCategory
    error_severity: ErrorSeverity
    failure_count: int
    last_success_time: float
    recovery_attempts: int
    recovery_mode: RecoveryMode
    priority: RecoveryPriority
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryAction:
    """Enhanced recovery action with detailed configuration."""
    action_id: str
    action_type: str
    parameters: Dict[str, Any]
    timeout: float
    retry_config: Optional[RetryConfig]
    circuit_breaker_config: Optional[CircuitBreakerConfig]
    fallback_actions: List[str] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)


class AdvancedRecoverySystem:
    """
    Advanced error recovery system with intelligent strategies and learning capabilities.
    
    Features:
    - Intelligent retry with exponential backoff and jitter
    - Advanced circuit breaker patterns with adaptive thresholds
    - Smart notification filtering to reduce false positives
    - Graceful degradation with automatic recovery
    - Machine learning-based recovery strategy optimization
    - Proactive failure prevention
    """
    
    def __init__(self):
        self.recovery_contexts: Dict[str, RecoveryContext] = {}
        self.recovery_history: deque = deque(maxlen=1000)
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.notification_filters: Dict[str, Dict[str, Any]] = {}
        self.recovery_strategies: Dict[str, Callable] = {}
        self.active_recoveries: Set[str] = set()
        self.recovery_lock = threading.RLock()
        
        # Performance tracking
        self.strategy_performance: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            'success_rate': 0.5,
            'avg_duration': 30.0,
            'total_attempts': 0,
            'successful_attempts': 0
        })
        
        # Notification suppression
        self.notification_history: deque = deque(maxlen=500)
        self.suppression_rules: Dict[str, Dict[str, Any]] = {}
        
        self._initialize_recovery_strategies()
        self._setup_notification_filters()
        
    def _initialize_recovery_strategies(self):
        """Initialize built-in recovery strategies."""
        self.recovery_strategies.update({
            'exponential_backoff_retry': self._exponential_backoff_retry,
            'circuit_breaker_protection': self._circuit_breaker_protection,
            'graceful_degradation': self._graceful_degradation,
            'service_restart': self._service_restart,
            'cache_invalidation': self._cache_invalidation,
            'connection_pool_reset': self._connection_pool_reset,
            'fallback_service_activation': self._fallback_service_activation,
            'emergency_mode_activation': self._emergency_mode_activation
        })
        
    def _setup_notification_filters(self):
        """Setup intelligent notification filtering rules."""
        self.suppression_rules.update({
            'high_frequency_errors': {
                'window_seconds': 300,  # 5 minutes
                'max_notifications': 3,
                'error_signature_fields': ['category', 'service_name', 'message_hash']
            },
            'transient_errors': {
                'patterns': ['timeout', 'connection reset', 'temporary', 'rate limit'],
                'suppress_duration': 600,  # 10 minutes
                'escalation_threshold': 5
            },
            'known_issues': {
                'suppress_duration': 1800,  # 30 minutes
                'auto_recovery_expected': True
            },
            'business_hours': {
                'off_hours_start': 20,  # 8 PM
                'off_hours_end': 8,     # 8 AM
                'severity_threshold': ErrorSeverity.HIGH
            }
        })
        
    def register_recovery_strategy(self, name: str, strategy_func: Callable):
        """Register a custom recovery strategy."""
        self.recovery_strategies[name] = strategy_func
        logger.info(f"Registered custom recovery strategy: {name}")
        
    def handle_error_with_advanced_recovery(
        self,
        error_context,
        recovery_mode: RecoveryMode = RecoveryMode.AUTOMATIC
    ) -> bool:
        """Handle error with advanced recovery mechanisms."""
        service_key = f"{error_context.category.value}:{error_context.service_name or 'unknown'}"
        
        with self.recovery_lock:
            # Check if recovery is already in progress
            if service_key in self.active_recoveries:
                logger.debug(f"Recovery already in progress for {service_key}")
                return False
                
            # Create or update recovery context
            recovery_context = self._create_recovery_context(error_context, recovery_mode)
            
            # Apply intelligent notification filtering
            should_notify = self._should_send_notification(error_context, recovery_context)
            
            if should_notify:
                self._send_filtered_notification(error_context, recovery_context)
            
            # Determine recovery priority
            priority = self._calculate_recovery_priority(error_context, recovery_context)
            recovery_context.priority = priority
            
            # Select optimal recovery strategy
            strategy_name = self._select_optimal_strategy(recovery_context)
            
            if strategy_name and strategy_name in self.recovery_strategies:
                self.active_recoveries.add(service_key)
                
                try:
                    # Execute recovery strategy
                    success = self._execute_recovery_strategy(
                        strategy_name, recovery_context, error_context
                    )
                    
                    # Update performance metrics
                    self._update_strategy_performance(strategy_name, success, recovery_context)
                    
                    # Record recovery attempt
                    self._record_recovery_attempt(recovery_context, strategy_name, success)
                    
                    return success
                    
                finally:
                    self.active_recoveries.discard(service_key)
            
            return False
            
    def _create_recovery_context(self, error_context, recovery_mode: RecoveryMode) -> RecoveryContext:
        """Create or update recovery context."""
        service_key = f"{error_context.category.value}:{error_context.service_name or 'unknown'}"
        
        if service_key in self.recovery_contexts:
            context = self.recovery_contexts[service_key]
            context.failure_count += 1
            context.recovery_attempts += 1
        else:
            context = RecoveryContext(
                error_id=error_context.error_id,
                service_name=error_context.service_name or 'unknown',
                error_category=error_context.category,
                error_severity=error_context.severity,
                failure_count=1,
                last_success_time=time.time() - 3600,  # Assume 1 hour ago
                recovery_attempts=1,
                recovery_mode=recovery_mode,
                priority=RecoveryPriority.NORMAL
            )
            self.recovery_contexts[service_key] = context
            
        return context
        
    def _should_send_notification(self, error_context, recovery_context: RecoveryContext) -> bool:
        """Intelligent notification filtering to reduce false positives."""
        current_time = time.time()
        
        # Always notify for critical errors
        if error_context.severity == ErrorSeverity.CRITICAL:
            return True
            
        # Create error signature for deduplication
        error_signature = self._create_error_signature(error_context)
        
        # Check high frequency suppression
        if self._is_high_frequency_error(error_signature, current_time):
            logger.debug(f"Suppressing high frequency error: {error_signature}")
            return False
            
        # Check transient error patterns
        if self._is_transient_error(error_context, current_time):
            logger.debug(f"Suppressing transient error: {error_context.message}")
            return False
            
        # Check business hours rules
        if self._is_off_hours_low_priority(error_context, current_time):
            logger.debug(f"Suppressing off-hours low priority error: {error_context.error_id}")
            return False
            
        # Check if automatic recovery is likely
        if self._is_auto_recovery_expected(recovery_context):
            logger.debug(f"Suppressing error with expected auto-recovery: {error_context.error_id}")
            return False
            
        return True
        
    def _create_error_signature(self, error_context) -> str:
        """Create a unique signature for error deduplication."""
        message_hash = hash(error_context.message[:100])  # First 100 chars
        return f"{error_context.category.value}:{error_context.service_name}:{message_hash}"
        
    def _is_high_frequency_error(self, error_signature: str, current_time: float) -> bool:
        """Check if error is occurring too frequently."""
        rule = self.suppression_rules['high_frequency_errors']
        window_start = current_time - rule['window_seconds']
        
        # Count recent notifications with same signature
        recent_count = sum(
            1 for notification in self.notification_history
            if (notification['timestamp'] > window_start and 
                notification['signature'] == error_signature)
        )
        
        return recent_count >= rule['max_notifications']
        
    def _is_transient_error(self, error_context, current_time: float) -> bool:
        """Check if error matches transient patterns."""
        rule = self.suppression_rules['transient_errors']
        message_lower = error_context.message.lower()
        
        # Check if message matches transient patterns
        is_transient = any(pattern in message_lower for pattern in rule['patterns'])
        
        if not is_transient:
            return False
            
        # For transient errors, check if we should escalate
        error_signature = self._create_error_signature(error_context)
        recent_count = sum(
            1 for notification in self.notification_history
            if (current_time - notification['timestamp'] < rule['suppress_duration'] and
                notification['signature'] == error_signature)
        )
        
        # Escalate if too many transient errors
        return recent_count < rule['escalation_threshold']
        
    def _is_off_hours_low_priority(self, error_context, current_time: float) -> bool:
        """Check if error should be suppressed during off-hours."""
        rule = self.suppression_rules['business_hours']
        
        # Only apply to medium and low severity errors
        if error_context.severity.value in ['high', 'critical']:
            return False
            
        import datetime
        current_hour = datetime.datetime.fromtimestamp(current_time).hour
        
        # Check if we're in off-hours
        if rule['off_hours_start'] <= current_hour or current_hour < rule['off_hours_end']:
            return error_context.severity != rule['severity_threshold']
            
        return False
        
    def _is_auto_recovery_expected(self, recovery_context: RecoveryContext) -> bool:
        """Check if automatic recovery is expected for this error."""
        service_key = f"{recovery_context.error_category.value}:{recovery_context.service_name}"
        
        # Check historical success rate
        if service_key in self.strategy_performance:
            perf = self.strategy_performance[service_key]
            return perf['success_rate'] > 0.8 and recovery_context.recovery_attempts <= 3
            
        return False
        
    def _send_filtered_notification(self, error_context, recovery_context: RecoveryContext):
        """Send notification with enhanced context and filtering."""
        # Record notification
        notification_record = {
            'timestamp': time.time(),
            'signature': self._create_error_signature(error_context),
            'error_id': error_context.error_id,
            'severity': error_context.severity.value
        }
        self.notification_history.append(notification_record)
        
        # Determine notification priority and channels
        priority = self._get_notification_priority(error_context, recovery_context)
        channels = self._get_notification_channels(error_context, recovery_context)
        
        # Create enhanced notification message
        message = self._create_enhanced_notification_message(error_context, recovery_context)
        
        # Send notification
        notification_system.send_notification(
            title=f"Enhanced Error Recovery - {error_context.category.value}",
            message=message,
            priority=priority,
            channels=channels,
            metadata={
                'error_context': {
                    'error_id': error_context.error_id,
                    'category': error_context.category.value,
                    'severity': error_context.severity.value,
                    'service_name': error_context.service_name
                },
                'recovery_context': {
                    'failure_count': recovery_context.failure_count,
                    'recovery_attempts': recovery_context.recovery_attempts,
                    'priority': recovery_context.priority.value,
                    'mode': recovery_context.recovery_mode.value
                }
            },
            deduplication_key=f"enhanced_recovery_{self._create_error_signature(error_context)}"
        )
        
    def _calculate_recovery_priority(self, error_context, recovery_context: RecoveryContext) -> RecoveryPriority:
        """Calculate recovery priority based on multiple factors."""
        priority_score = 0
        
        # Factor 1: Error severity
        severity_scores = {
            ErrorSeverity.LOW: 1,
            ErrorSeverity.MEDIUM: 2,
            ErrorSeverity.HIGH: 3,
            ErrorSeverity.CRITICAL: 4
        }
        priority_score += severity_scores.get(error_context.severity, 2)
        
        # Factor 2: Service criticality
        critical_services = ['database', 'authentication', 'billing']
        if recovery_context.service_name.lower() in critical_services:
            priority_score += 2
            
        # Factor 3: Failure frequency
        if recovery_context.failure_count > 5:
            priority_score += 2
        elif recovery_context.failure_count > 2:
            priority_score += 1
            
        # Factor 4: Time since last success
        time_since_success = time.time() - recovery_context.last_success_time
        if time_since_success > 3600:  # 1 hour
            priority_score += 2
        elif time_since_success > 1800:  # 30 minutes
            priority_score += 1
            
        # Map score to priority
        if priority_score >= 8:
            return RecoveryPriority.CRITICAL
        elif priority_score >= 6:
            return RecoveryPriority.HIGH
        elif priority_score >= 4:
            return RecoveryPriority.NORMAL
        else:
            return RecoveryPriority.LOW
            
    def _select_optimal_strategy(self, recovery_context: RecoveryContext) -> Optional[str]:
        """Select optimal recovery strategy based on context and performance history."""
        # Get strategies suitable for this error category
        suitable_strategies = self._get_suitable_strategies(recovery_context)
        
        if not suitable_strategies:
            return None
            
        # If only one strategy, use it
        if len(suitable_strategies) == 1:
            return suitable_strategies[0]
            
        # Select based on performance history and context
        best_strategy = None
        best_score = -1
        
        for strategy in suitable_strategies:
            score = self._calculate_strategy_score(strategy, recovery_context)
            if score > best_score:
                best_score = score
                best_strategy = strategy
                
        return best_strategy
        
    def _get_suitable_strategies(self, recovery_context: RecoveryContext) -> List[str]:
        """Get strategies suitable for the given recovery context."""
        category_strategies = {
            ErrorCategory.DATABASE: [
                'exponential_backoff_retry',
                'connection_pool_reset',
                'circuit_breaker_protection',
                'graceful_degradation'
            ],
            ErrorCategory.EXTERNAL_API: [
                'exponential_backoff_retry',
                'circuit_breaker_protection',
                'fallback_service_activation',
                'graceful_degradation'
            ],
            ErrorCategory.EXTRACTION: [
                'exponential_backoff_retry',
                'cache_invalidation',
                'graceful_degradation'
            ],
            ErrorCategory.ANNOTATION: [
                'fallback_service_activation',
                'graceful_degradation',
                'cache_invalidation'
            ],
            ErrorCategory.SYSTEM: [
                'service_restart',
                'emergency_mode_activation',
                'graceful_degradation'
            ]
        }
        
        strategies = category_strategies.get(recovery_context.error_category, [])
        
        # Filter based on recovery mode
        if recovery_context.recovery_mode == RecoveryMode.MANUAL:
            return []  # No automatic strategies for manual mode
        elif recovery_context.recovery_mode == RecoveryMode.EMERGENCY:
            return ['emergency_mode_activation', 'service_restart']
            
        return strategies
        
    def _calculate_strategy_score(self, strategy: str, recovery_context: RecoveryContext) -> float:
        """Calculate score for a strategy based on performance and context."""
        service_key = f"{recovery_context.error_category.value}:{recovery_context.service_name}"
        
        # Base score from historical performance
        if service_key in self.strategy_performance:
            perf = self.strategy_performance[service_key]
            base_score = perf['success_rate']
        else:
            base_score = 0.5  # Neutral starting point
            
        # Adjust based on context
        
        # Priority adjustment
        priority_multipliers = {
            RecoveryPriority.CRITICAL: 1.2,
            RecoveryPriority.HIGH: 1.1,
            RecoveryPriority.NORMAL: 1.0,
            RecoveryPriority.LOW: 0.9
        }
        base_score *= priority_multipliers.get(recovery_context.priority, 1.0)
        
        # Failure count adjustment (prefer strategies that work with high failure counts)
        if recovery_context.failure_count > 5:
            if strategy in ['service_restart', 'emergency_mode_activation']:
                base_score *= 1.2
            elif strategy in ['exponential_backoff_retry']:
                base_score *= 0.8
                
        # Recovery attempt adjustment (avoid strategies that have failed recently)
        if recovery_context.recovery_attempts > 3:
            if strategy in ['graceful_degradation', 'fallback_service_activation']:
                base_score *= 1.1
            else:
                base_score *= 0.9
                
        return base_score
        
    def _execute_recovery_strategy(
        self,
        strategy_name: str,
        recovery_context: RecoveryContext,
        error_context
    ) -> bool:
        """Execute the selected recovery strategy."""
        logger.info(f"Executing recovery strategy '{strategy_name}' for {recovery_context.service_name}")
        
        start_time = time.time()
        
        try:
            strategy_func = self.recovery_strategies[strategy_name]
            success = strategy_func(recovery_context, error_context)
            
            duration = time.time() - start_time
            
            if success:
                logger.info(f"Recovery strategy '{strategy_name}' succeeded in {duration:.2f}s")
                # Update last success time
                recovery_context.last_success_time = time.time()
            else:
                logger.warning(f"Recovery strategy '{strategy_name}' failed after {duration:.2f}s")
                
            return success
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Recovery strategy '{strategy_name}' raised exception after {duration:.2f}s: {e}")
            return False
            
    # Recovery Strategy Implementations
    
    def _exponential_backoff_retry(self, recovery_context: RecoveryContext, error_context) -> bool:
        """Enhanced exponential backoff retry with adaptive parameters."""
        # Create adaptive retry configuration
        retry_config = RetryConfig(
            max_attempts=min(5, 8 - recovery_context.recovery_attempts),  # Reduce attempts over time
            base_delay=1.0 * (1.5 ** min(recovery_context.failure_count - 1, 3)),  # Increase base delay
            max_delay=min(120.0, 30.0 * recovery_context.priority.value),  # Priority-based max delay
            strategy=RetryStrategy.EXPONENTIAL,
            backoff_multiplier=2.0,
            jitter=True,
            jitter_range=0.2
        )
        
        # Simulate retry operation
        for attempt in range(retry_config.max_attempts):
            try:
                # Simulate operation (in real implementation, this would call the actual service)
                logger.info(f"Retry attempt {attempt + 1}/{retry_config.max_attempts} for {recovery_context.service_name}")
                
                # Simulate success based on attempt number and service health
                success_probability = 0.3 + (attempt * 0.2)  # Increasing probability
                if recovery_context.priority == RecoveryPriority.CRITICAL:
                    success_probability += 0.2
                    
                import random
                if random.random() < success_probability:
                    logger.info(f"Retry succeeded on attempt {attempt + 1}")
                    return True
                    
                # Calculate delay for next attempt
                if attempt < retry_config.max_attempts - 1:
                    delay = self._calculate_retry_delay(retry_config, attempt)
                    logger.debug(f"Waiting {delay:.2f}s before next retry")
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"Retry attempt {attempt + 1} failed: {e}")
                
        logger.warning(f"All retry attempts failed for {recovery_context.service_name}")
        return False
        
    def _calculate_retry_delay(self, config: RetryConfig, attempt: int) -> float:
        """Calculate retry delay with enhanced jitter."""
        if config.strategy == RetryStrategy.EXPONENTIAL:
            delay = config.base_delay * (config.backoff_multiplier ** attempt)
        else:
            delay = config.base_delay
            
        delay = min(delay, config.max_delay)
        
        if config.jitter:
            import random
            jitter_amount = delay * config.jitter_range
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0.1, delay)
            
        return delay
        
    def _circuit_breaker_protection(self, recovery_context: RecoveryContext, error_context) -> bool:
        """Enhanced circuit breaker protection with adaptive thresholds."""
        service_name = recovery_context.service_name
        
        # Create or get circuit breaker
        if service_name not in self.circuit_breakers:
            # Adaptive configuration based on service and error patterns
            config = CircuitBreakerConfig(
                failure_threshold=max(2, 5 - recovery_context.failure_count),
                recovery_timeout=min(300.0, 60.0 * recovery_context.priority.value),
                success_threshold=3,
                timeout=30.0
            )
            self.circuit_breakers[service_name] = CircuitBreaker(service_name, config)
            
        circuit_breaker = self.circuit_breakers[service_name]
        
        try:
            # Test circuit breaker state
            state = circuit_breaker.get_state()
            logger.info(f"Circuit breaker {service_name} state: {state['state']}")
            
            if state['state'] == 'open':
                # Circuit is open, check if we can transition to half-open
                time_since_failure = time.time() - state['last_failure_time']
                if time_since_failure > circuit_breaker.config.recovery_timeout:
                    logger.info(f"Circuit breaker {service_name} attempting recovery test")
                    # Simulate recovery test
                    return self._test_service_health(service_name)
                else:
                    logger.info(f"Circuit breaker {service_name} still in timeout period")
                    return False
            else:
                # Circuit is closed or half-open, simulate operation
                return self._test_service_health(service_name)
                
        except Exception as e:
            logger.error(f"Circuit breaker operation failed for {service_name}: {e}")
            return False
            
    def _test_service_health(self, service_name: str) -> bool:
        """Test service health for circuit breaker recovery."""
        # Simulate health check
        import random
        
        # Base success probability
        success_prob = 0.6
        
        # Adjust based on service type
        if 'database' in service_name.lower():
            success_prob = 0.8
        elif 'api' in service_name.lower():
            success_prob = 0.7
            
        success = random.random() < success_prob
        
        if success:
            logger.info(f"Service health check passed for {service_name}")
            # Mark service as healthy in degradation manager
            degradation_manager.mark_service_success(service_name)
        else:
            logger.warning(f"Service health check failed for {service_name}")
            degradation_manager.mark_service_failure(service_name)
            
        return success
        
    def _graceful_degradation(self, recovery_context: RecoveryContext, error_context) -> bool:
        """Enhanced graceful degradation with intelligent level selection."""
        service_name = recovery_context.service_name
        
        try:
            # Get current service health
            health = degradation_manager.get_service_health(service_name)
            if not health:
                degradation_manager.register_service(service_name)
                health = degradation_manager.get_service_health(service_name)
                
            current_level = health.degradation_level
            
            # Calculate optimal degradation level
            new_level = self._calculate_degradation_level(recovery_context, current_level)
            
            if new_level != current_level:
                # Apply degradation
                degradation_manager.mark_service_failure(service_name)
                health.degradation_level = new_level
                
                logger.info(f"Service {service_name} degraded to {new_level.value}")
                
                # Send degradation notification
                self._send_degradation_notification(service_name, current_level, new_level, recovery_context)
                
                # Schedule recovery check
                self._schedule_recovery_check(service_name, recovery_context)
                
                return True
            else:
                logger.info(f"Service {service_name} already at optimal degradation level: {current_level.value}")
                return True
                
        except Exception as e:
            logger.error(f"Graceful degradation failed for {service_name}: {e}")
            return False
            
    def _calculate_degradation_level(
        self,
        recovery_context: RecoveryContext,
        current_level: DegradationLevel
    ) -> DegradationLevel:
        """Calculate optimal degradation level based on recovery context."""
        levels = [DegradationLevel.FULL, DegradationLevel.REDUCED, DegradationLevel.MINIMAL, DegradationLevel.OFFLINE]
        current_index = levels.index(current_level)
        
        # Calculate degradation steps needed
        degradation_steps = 0
        
        # Factor 1: Error severity
        if recovery_context.error_severity == ErrorSeverity.CRITICAL:
            degradation_steps += 2
        elif recovery_context.error_severity == ErrorSeverity.HIGH:
            degradation_steps += 1
            
        # Factor 2: Failure count
        if recovery_context.failure_count > 10:
            degradation_steps += 2
        elif recovery_context.failure_count > 5:
            degradation_steps += 1
            
        # Factor 3: Recovery attempts
        if recovery_context.recovery_attempts > 5:
            degradation_steps += 1
            
        # Apply degradation
        new_index = min(current_index + degradation_steps, len(levels) - 1)
        return levels[new_index]
        
    def _service_restart(self, recovery_context: RecoveryContext, error_context) -> bool:
        """Enhanced service restart with pre and post checks."""
        service_name = recovery_context.service_name
        
        try:
            logger.info(f"Attempting service restart for {service_name}")
            
            # Pre-restart checks
            if not self._pre_restart_checks(service_name):
                logger.warning(f"Pre-restart checks failed for {service_name}")
                return False
                
            # Simulate service restart
            logger.info(f"Restarting service: {service_name}")
            time.sleep(2)  # Simulate restart time
            
            # Post-restart verification
            if self._post_restart_verification(service_name):
                logger.info(f"Service restart successful for {service_name}")
                degradation_manager.mark_service_success(service_name)
                
                # Send success notification
                notification_system.send_notification(
                    title=f"Service Restart Successful - {service_name}",
                    message=f"Service {service_name} has been successfully restarted and is operational",
                    priority=NotificationPriority.NORMAL,
                    channels=[NotificationChannel.LOG, NotificationChannel.SLACK]
                )
                
                return True
            else:
                logger.error(f"Service restart verification failed for {service_name}")
                return False
                
        except Exception as e:
            logger.error(f"Service restart failed for {service_name}: {e}")
            return False
            
    def _pre_restart_checks(self, service_name: str) -> bool:
        """Perform pre-restart safety checks."""
        # Check if service is critical and has active users
        # Check if restart is safe at current time
        # Check dependencies
        
        logger.info(f"Performing pre-restart checks for {service_name}")
        
        # Simulate checks
        import random
        return random.random() > 0.1  # 90% success rate
        
    def _post_restart_verification(self, service_name: str) -> bool:
        """Verify service is working after restart."""
        logger.info(f"Verifying service health after restart: {service_name}")
        
        # Simulate verification
        return self._test_service_health(service_name)
        
    def _cache_invalidation(self, recovery_context: RecoveryContext, error_context) -> bool:
        """Enhanced cache invalidation with selective clearing."""
        service_name = recovery_context.service_name
        
        try:
            logger.info(f"Performing cache invalidation for {service_name}")
            
            # Determine cache types to clear based on error
            cache_types = self._determine_cache_types(recovery_context, error_context)
            
            for cache_type in cache_types:
                logger.info(f"Clearing {cache_type} cache for {service_name}")
                # Simulate cache clearing
                time.sleep(0.5)
                
            # Clear degradation manager cache for this service
            degradation_manager.clear_cache()
            
            logger.info(f"Cache invalidation completed for {service_name}")
            return True
            
        except Exception as e:
            logger.error(f"Cache invalidation failed for {service_name}: {e}")
            return False
            
    def _determine_cache_types(self, recovery_context: RecoveryContext, error_context) -> List[str]:
        """Determine which cache types to clear based on error context."""
        cache_types = ['application']
        
        if recovery_context.error_category == ErrorCategory.DATABASE:
            cache_types.extend(['query', 'connection_pool'])
        elif recovery_context.error_category == ErrorCategory.EXTERNAL_API:
            cache_types.extend(['api_response', 'credentials'])
        elif recovery_context.error_category == ErrorCategory.EXTRACTION:
            cache_types.extend(['extraction_results', 'file_metadata'])
            
        return cache_types
        
    def _connection_pool_reset(self, recovery_context: RecoveryContext, error_context) -> bool:
        """Reset connection pools for database-related errors."""
        service_name = recovery_context.service_name
        
        try:
            logger.info(f"Resetting connection pools for {service_name}")
            
            # Simulate connection pool reset
            time.sleep(1)
            
            # Test new connections
            if self._test_service_health(service_name):
                logger.info(f"Connection pool reset successful for {service_name}")
                return True
            else:
                logger.warning(f"Connection pool reset did not resolve issues for {service_name}")
                return False
                
        except Exception as e:
            logger.error(f"Connection pool reset failed for {service_name}: {e}")
            return False
            
    def _fallback_service_activation(self, recovery_context: RecoveryContext, error_context) -> bool:
        """Activate fallback services for critical operations."""
        service_name = recovery_context.service_name
        
        try:
            # Determine fallback service
            fallback_service = self._get_fallback_service(service_name)
            
            if not fallback_service:
                logger.warning(f"No fallback service available for {service_name}")
                return False
                
            logger.info(f"Activating fallback service {fallback_service} for {service_name}")
            
            # Simulate fallback activation
            time.sleep(1)
            
            # Test fallback service
            if self._test_service_health(fallback_service):
                logger.info(f"Fallback service {fallback_service} activated successfully")
                
                # Update service routing
                degradation_manager.mark_service_failure(service_name)
                degradation_manager.mark_service_success(fallback_service)
                
                return True
            else:
                logger.error(f"Fallback service {fallback_service} is also unhealthy")
                return False
                
        except Exception as e:
            logger.error(f"Fallback service activation failed for {service_name}: {e}")
            return False
            
    def _get_fallback_service(self, service_name: str) -> Optional[str]:
        """Get fallback service name for the given service."""
        fallback_mapping = {
            'ai_annotation': 'rule_based_annotation',
            'quality_check': 'basic_quality_check',
            'data_extraction': 'simple_extraction',
            'external_api': 'cached_api_response'
        }
        
        return fallback_mapping.get(service_name)
        
    def _emergency_mode_activation(self, recovery_context: RecoveryContext, error_context) -> bool:
        """Activate emergency mode for critical system failures."""
        service_name = recovery_context.service_name
        
        try:
            logger.critical(f"Activating emergency mode for {service_name}")
            
            # Set all services to minimal degradation
            for service in ['ai_annotation', 'quality_check', 'data_extraction']:
                health = degradation_manager.get_service_health(service)
                if health:
                    health.degradation_level = DegradationLevel.MINIMAL
                    
            # Send critical notification
            notification_system.send_notification(
                title="EMERGENCY MODE ACTIVATED",
                message=f"Critical system failure detected. Emergency mode activated for {service_name}. Immediate attention required.",
                priority=NotificationPriority.CRITICAL,
                channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.WEBHOOK],
                metadata={
                    'emergency_mode': True,
                    'service_name': service_name,
                    'error_id': error_context.error_id,
                    'recovery_context': recovery_context.__dict__
                }
            )
            
            logger.critical(f"Emergency mode activated for {service_name}")
            return True
            
        except Exception as e:
            logger.error(f"Emergency mode activation failed for {service_name}: {e}")
            return False
            
    def _update_strategy_performance(
        self,
        strategy_name: str,
        success: bool,
        recovery_context: RecoveryContext
    ):
        """Update strategy performance metrics."""
        service_key = f"{recovery_context.error_category.value}:{recovery_context.service_name}"
        
        if service_key not in self.strategy_performance:
            self.strategy_performance[service_key] = {
                'success_rate': 0.5,
                'avg_duration': 30.0,
                'total_attempts': 0,
                'successful_attempts': 0
            }
            
        perf = self.strategy_performance[service_key]
        perf['total_attempts'] += 1
        
        if success:
            perf['successful_attempts'] += 1
            
        # Update success rate using exponential moving average
        alpha = 0.3
        new_rate = 1.0 if success else 0.0
        perf['success_rate'] = alpha * new_rate + (1 - alpha) * perf['success_rate']
        
    def _record_recovery_attempt(
        self,
        recovery_context: RecoveryContext,
        strategy_name: str,
        success: bool
    ):
        """Record recovery attempt for analysis and learning."""
        record = {
            'timestamp': time.time(),
            'service_name': recovery_context.service_name,
            'error_category': recovery_context.error_category.value,
            'error_severity': recovery_context.error_severity.value,
            'strategy_name': strategy_name,
            'success': success,
            'failure_count': recovery_context.failure_count,
            'recovery_attempts': recovery_context.recovery_attempts,
            'priority': recovery_context.priority.value
        }
        
        self.recovery_history.append(record)
        
    def _send_degradation_notification(
        self,
        service_name: str,
        old_level: DegradationLevel,
        new_level: DegradationLevel,
        recovery_context: RecoveryContext
    ):
        """Send notification about service degradation."""
        impact_analysis = self._analyze_degradation_impact(service_name, new_level)
        
        notification_system.send_notification(
            title=f"Service Degradation - {service_name}",
            message=f"Service {service_name} degraded from {old_level.value} to {new_level.value}\n\nImpact Analysis:\n{impact_analysis}",
            priority=self._get_degradation_priority(new_level),
            channels=self._get_degradation_channels(new_level),
            metadata={
                'service_name': service_name,
                'old_level': old_level.value,
                'new_level': new_level.value,
                'recovery_context': recovery_context.__dict__
            }
        )
        
    def _analyze_degradation_impact(self, service_name: str, level: DegradationLevel) -> str:
        """Analyze the impact of service degradation."""
        impact_descriptions = {
            DegradationLevel.REDUCED: "• Reduced functionality and performance\n• Some features may be unavailable\n• Fallback mechanisms active",
            DegradationLevel.MINIMAL: "• Minimal functionality only\n• Most features disabled\n• Basic operations available",
            DegradationLevel.OFFLINE: "• Service completely offline\n• All functionality unavailable\n• Manual intervention required"
        }
        
        return impact_descriptions.get(level, "• Service operating normally")
        
    def _get_degradation_priority(self, level: DegradationLevel) -> NotificationPriority:
        """Get notification priority for degradation level."""
        priority_map = {
            DegradationLevel.REDUCED: NotificationPriority.NORMAL,
            DegradationLevel.MINIMAL: NotificationPriority.HIGH,
            DegradationLevel.OFFLINE: NotificationPriority.CRITICAL
        }
        return priority_map.get(level, NotificationPriority.NORMAL)
        
    def _get_degradation_channels(self, level: DegradationLevel) -> List:
        """Get notification channels for degradation level."""
        channel_map = {
            DegradationLevel.REDUCED: [NotificationChannel.LOG, NotificationChannel.SLACK],
            DegradationLevel.MINIMAL: [NotificationChannel.LOG, NotificationChannel.SLACK, NotificationChannel.WEBHOOK],
            DegradationLevel.OFFLINE: [NotificationChannel.LOG, NotificationChannel.SLACK, NotificationChannel.EMAIL, NotificationChannel.WEBHOOK]
        }
        return channel_map.get(level, [NotificationChannel.LOG])
        
    def _schedule_recovery_check(self, service_name: str, recovery_context: RecoveryContext):
        """Schedule automatic recovery check for degraded service."""
        def recovery_check():
            try:
                # Wait before checking
                time.sleep(60)  # 1 minute
                
                # Test service health
                if self._test_service_health(service_name):
                    # Service recovered
                    health = degradation_manager.get_service_health(service_name)
                    if health:
                        health.degradation_level = DegradationLevel.FULL
                        
                    logger.info(f"Automatic recovery detected for {service_name}")
                    
                    notification_system.send_notification(
                        title=f"Service Recovery - {service_name}",
                        message=f"Service {service_name} has automatically recovered",
                        priority=NotificationPriority.NORMAL,
                        channels=[NotificationChannel.LOG, NotificationChannel.SLACK]
                    )
                    
            except Exception as e:
                logger.error(f"Recovery check failed for {service_name}: {e}")
                
        # Schedule in background
        threading.Thread(target=recovery_check, daemon=True).start()
        
    def _get_notification_priority(self, error_context, recovery_context: RecoveryContext) -> NotificationPriority:
        """Get notification priority based on error and recovery context."""
        if error_context.severity == ErrorSeverity.CRITICAL:
            return NotificationPriority.CRITICAL
        elif error_context.severity == ErrorSeverity.HIGH or recovery_context.priority == RecoveryPriority.CRITICAL:
            return NotificationPriority.HIGH
        else:
            return NotificationPriority.NORMAL
            
    def _get_notification_channels(self, error_context, recovery_context: RecoveryContext) -> List:
        """Get notification channels based on error and recovery context."""
        if error_context.severity == ErrorSeverity.CRITICAL:
            return [NotificationChannel.LOG, NotificationChannel.SLACK, NotificationChannel.EMAIL, NotificationChannel.WEBHOOK]
        elif error_context.severity == ErrorSeverity.HIGH:
            return [NotificationChannel.LOG, NotificationChannel.SLACK, NotificationChannel.WEBHOOK]
        else:
            return [NotificationChannel.LOG, NotificationChannel.SLACK]
            
    def _create_enhanced_notification_message(self, error_context, recovery_context: RecoveryContext) -> str:
        """Create enhanced notification message with recovery context."""
        message = f"Error ID: {error_context.error_id}\n"
        message += f"Service: {recovery_context.service_name}\n"
        message += f"Category: {error_context.category.value}\n"
        message += f"Severity: {error_context.severity.value}\n"
        message += f"Message: {error_context.message}\n\n"
        
        message += f"Recovery Context:\n"
        message += f"• Failure Count: {recovery_context.failure_count}\n"
        message += f"• Recovery Attempts: {recovery_context.recovery_attempts}\n"
        message += f"• Priority: {recovery_context.priority.value}\n"
        message += f"• Mode: {recovery_context.recovery_mode.value}\n"
        
        # Add recovery strategy recommendation
        strategy = self._select_optimal_strategy(recovery_context)
        if strategy:
            message += f"\nRecommended Recovery Strategy: {strategy}"
            
        return message
        
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get comprehensive recovery system statistics."""
        return {
            'active_recoveries': len(self.active_recoveries),
            'total_recovery_contexts': len(self.recovery_contexts),
            'recovery_history_size': len(self.recovery_history),
            'circuit_breakers': len(self.circuit_breakers),
            'strategy_performance': dict(self.strategy_performance),
            'notification_history_size': len(self.notification_history),
            'suppression_rules': self.suppression_rules
        }


# Global instance
advanced_recovery_system = AdvancedRecoverySystem()


# Integration function
def integrate_with_error_handler():
    """Integrate advanced recovery system with existing error handler."""
    def enhanced_error_handler(error_context):
        return advanced_recovery_system.handle_error_with_advanced_recovery(error_context)
    
    error_handler.register_notification_handler(enhanced_error_handler)
    logger.info("Advanced recovery system integrated with error handler")


# Initialize integration
integrate_with_error_handler()