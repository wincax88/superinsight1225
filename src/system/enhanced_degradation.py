"""
Enhanced Graceful Degradation System for SuperInsight Platform.

Provides intelligent service degradation with automatic recovery,
performance monitoring, and adaptive fallback strategies.
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

from src.utils.degradation import degradation_manager, DegradationLevel
from src.system.notification import notification_system, NotificationPriority, NotificationChannel
from src.system.error_handler import ErrorCategory, ErrorSeverity

logger = logging.getLogger(__name__)


class DegradationTrigger(Enum):
    """Triggers for service degradation."""
    ERROR_RATE = "error_rate"
    RESPONSE_TIME = "response_time"
    RESOURCE_USAGE = "resource_usage"
    DEPENDENCY_FAILURE = "dependency_failure"
    MANUAL = "manual"
    SCHEDULED = "scheduled"


class RecoveryStrategy(Enum):
    """Recovery strategies for degraded services."""
    IMMEDIATE = "immediate"
    GRADUAL = "gradual"
    SCHEDULED = "scheduled"
    MANUAL = "manual"


@dataclass
class DegradationRule:
    """Rule for automatic service degradation."""
    name: str
    service_pattern: str  # Regex pattern for service names
    trigger: DegradationTrigger
    threshold: Dict[str, Any]
    target_level: DegradationLevel
    enabled: bool = True
    priority: int = 1


@dataclass
class RecoveryRule:
    """Rule for automatic service recovery."""
    name: str
    service_pattern: str
    strategy: RecoveryStrategy
    conditions: Dict[str, Any]
    target_level: DegradationLevel
    enabled: bool = True


@dataclass
class ServiceHealth:
    """Enhanced service health information."""
    service_name: str
    current_level: DegradationLevel
    target_level: Optional[DegradationLevel]
    last_degradation_time: float
    last_recovery_time: float
    degradation_count: int
    recovery_count: int
    error_rate: float
    avg_response_time: float
    resource_usage: Dict[str, float]
    dependencies: List[str]
    fallback_services: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DegradationEvent:
    """Event record for degradation/recovery actions."""
    timestamp: float
    service_name: str
    event_type: str  # 'degradation', 'recovery', 'fallback'
    from_level: DegradationLevel
    to_level: DegradationLevel
    trigger: str
    reason: str
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnhancedDegradationManager:
    """
    Enhanced graceful degradation manager with intelligent strategies.
    
    Features:
    - Automatic degradation based on multiple triggers
    - Intelligent recovery with gradual restoration
    - Fallback service management
    - Performance-based degradation decisions
    - Dependency-aware degradation cascading
    - Adaptive thresholds based on historical data
    """
    
    def __init__(self):
        self.service_health: Dict[str, ServiceHealth] = {}
        self.degradation_rules: List[DegradationRule] = []
        self.recovery_rules: List[RecoveryRule] = []
        self.event_history: deque = deque(maxlen=1000)
        self.fallback_mappings: Dict[str, List[str]] = {}
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.lock = threading.RLock()
        
        # Monitoring and recovery threads
        self.monitoring_active = False
        self.recovery_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.recovery_thread: Optional[threading.Thread] = None
        
        self._initialize_default_rules()
        
    def _initialize_default_rules(self):
        """Initialize default degradation and recovery rules."""
        # Error rate based degradation
        self.degradation_rules.extend([
            DegradationRule(
                name="high_error_rate_reduced",
                service_pattern=".*",
                trigger=DegradationTrigger.ERROR_RATE,
                threshold={'error_rate': 0.1, 'window_seconds': 300},  # 10% error rate in 5 minutes
                target_level=DegradationLevel.REDUCED,
                priority=5
            ),
            DegradationRule(
                name="critical_error_rate_minimal",
                service_pattern=".*",
                trigger=DegradationTrigger.ERROR_RATE,
                threshold={'error_rate': 0.25, 'window_seconds': 180},  # 25% error rate in 3 minutes
                target_level=DegradationLevel.MINIMAL,
                priority=8
            ),
            DegradationRule(
                name="extreme_error_rate_offline",
                service_pattern=".*",
                trigger=DegradationTrigger.ERROR_RATE,
                threshold={'error_rate': 0.5, 'window_seconds': 120},  # 50% error rate in 2 minutes
                target_level=DegradationLevel.OFFLINE,
                priority=10
            )
        ])
        
        # Response time based degradation
        self.degradation_rules.extend([
            DegradationRule(
                name="slow_response_reduced",
                service_pattern=".*",
                trigger=DegradationTrigger.RESPONSE_TIME,
                threshold={'avg_response_time': 5.0, 'window_seconds': 300},  # 5s average in 5 minutes
                target_level=DegradationLevel.REDUCED,
                priority=4
            ),
            DegradationRule(
                name="very_slow_response_minimal",
                service_pattern=".*",
                trigger=DegradationTrigger.RESPONSE_TIME,
                threshold={'avg_response_time': 15.0, 'window_seconds': 180},  # 15s average in 3 minutes
                target_level=DegradationLevel.MINIMAL,
                priority=7
            )
        ])
        
        # Resource usage based degradation
        self.degradation_rules.extend([
            DegradationRule(
                name="high_cpu_reduced",
                service_pattern=".*",
                trigger=DegradationTrigger.RESOURCE_USAGE,
                threshold={'cpu_usage': 0.8, 'window_seconds': 600},  # 80% CPU for 10 minutes
                target_level=DegradationLevel.REDUCED,
                priority=3
            ),
            DegradationRule(
                name="high_memory_reduced",
                service_pattern=".*",
                trigger=DegradationTrigger.RESOURCE_USAGE,
                threshold={'memory_usage': 0.9, 'window_seconds': 300},  # 90% memory for 5 minutes
                target_level=DegradationLevel.REDUCED,
                priority=6
            )
        ])
        
        # Recovery rules
        self.recovery_rules.extend([
            RecoveryRule(
                name="gradual_recovery_to_reduced",
                service_pattern=".*",
                strategy=RecoveryStrategy.GRADUAL,
                conditions={
                    'error_rate_below': 0.05,
                    'response_time_below': 2.0,
                    'stable_duration': 300  # 5 minutes of stability
                },
                target_level=DegradationLevel.REDUCED
            ),
            RecoveryRule(
                name="gradual_recovery_to_full",
                service_pattern=".*",
                strategy=RecoveryStrategy.GRADUAL,
                conditions={
                    'error_rate_below': 0.02,
                    'response_time_below': 1.0,
                    'stable_duration': 600  # 10 minutes of stability
                },
                target_level=DegradationLevel.FULL
            )
        ])
        
        # Default fallback mappings
        self.fallback_mappings.update({
            'ai_annotation': ['rule_based_annotation', 'basic_annotation'],
            'quality_check': ['basic_quality_check', 'rule_based_quality'],
            'data_extraction': ['simple_extraction', 'manual_extraction'],
            'external_api': ['cached_response', 'default_response']
        })
        
        # Default dependencies
        self.dependency_graph.update({
            'ai_annotation': {'database', 'external_api'},
            'quality_check': {'database', 'ai_annotation'},
            'data_extraction': {'database', 'external_api'},
            'billing': {'database'},
            'export': {'database', 'quality_check'}
        })
        
    def register_service(self, service_name: str, dependencies: Optional[List[str]] = None,
                        fallback_services: Optional[List[str]] = None):
        """Register a service with the degradation manager."""
        with self.lock:
            if service_name not in self.service_health:
                self.service_health[service_name] = ServiceHealth(
                    service_name=service_name,
                    current_level=DegradationLevel.FULL,
                    target_level=None,
                    last_degradation_time=0,
                    last_recovery_time=time.time(),
                    degradation_count=0,
                    recovery_count=0,
                    error_rate=0.0,
                    avg_response_time=0.0,
                    resource_usage={},
                    dependencies=dependencies or [],
                    fallback_services=fallback_services or []
                )
                
                # Update dependency graph
                if dependencies:
                    self.dependency_graph[service_name].update(dependencies)
                    
                # Update fallback mappings
                if fallback_services:
                    self.fallback_mappings[service_name] = fallback_services
                    
                logger.info(f"Registered service: {service_name}")
                
    def update_service_metrics(self, service_name: str, metrics: Dict[str, Any]):
        """Update service performance metrics."""
        with self.lock:
            if service_name not in self.service_health:
                self.register_service(service_name)
                
            health = self.service_health[service_name]
            
            # Update metrics with exponential moving average
            alpha = 0.3  # Learning rate
            
            if 'error_rate' in metrics:
                health.error_rate = alpha * metrics['error_rate'] + (1 - alpha) * health.error_rate
                
            if 'response_time' in metrics:
                health.avg_response_time = alpha * metrics['response_time'] + (1 - alpha) * health.avg_response_time
                
            if 'resource_usage' in metrics:
                for resource, value in metrics['resource_usage'].items():
                    current = health.resource_usage.get(resource, 0.0)
                    health.resource_usage[resource] = alpha * value + (1 - alpha) * current
                    
            # Store in performance history
            self.performance_history[service_name].append({
                'timestamp': time.time(),
                'metrics': metrics.copy()
            })
            
            # Check for degradation triggers
            self._check_degradation_triggers(service_name)
            
    def degrade_service(self, service_name: str, target_level: DegradationLevel,
                       reason: str, trigger: str = "manual") -> bool:
        """Degrade a service to the specified level."""
        with self.lock:
            if service_name not in self.service_health:
                self.register_service(service_name)
                
            health = self.service_health[service_name]
            from_level = health.current_level
            
            # Check if degradation is needed
            if self._should_degrade(health, target_level):
                # Apply degradation
                success = self._apply_degradation(service_name, target_level, reason, trigger)
                
                if success:
                    # Update health status
                    health.current_level = target_level
                    health.last_degradation_time = time.time()
                    health.degradation_count += 1
                    
                    # Record event
                    self._record_event(DegradationEvent(
                        timestamp=time.time(),
                        service_name=service_name,
                        event_type='degradation',
                        from_level=from_level,
                        to_level=target_level,
                        trigger=trigger,
                        reason=reason,
                        success=True
                    ))
                    
                    # Check for cascade effects
                    self._check_cascade_degradation(service_name, target_level)
                    
                    logger.info(f"Service {service_name} degraded from {from_level.value} to {target_level.value}: {reason}")
                    return True
                else:
                    logger.error(f"Failed to degrade service {service_name} to {target_level.value}")
                    return False
            else:
                logger.debug(f"Degradation not needed for {service_name}: already at {health.current_level.value}")
                return True
                
    def recover_service(self, service_name: str, target_level: DegradationLevel,
                       strategy: RecoveryStrategy = RecoveryStrategy.GRADUAL) -> bool:
        """Recover a service to the specified level."""
        with self.lock:
            if service_name not in self.service_health:
                logger.warning(f"Cannot recover unregistered service: {service_name}")
                return False
                
            health = self.service_health[service_name]
            from_level = health.current_level
            
            # Check if recovery is appropriate
            if self._should_recover(health, target_level):
                # Apply recovery based on strategy
                success = self._apply_recovery(service_name, target_level, strategy)
                
                if success:
                    # Update health status
                    health.current_level = target_level
                    health.last_recovery_time = time.time()
                    health.recovery_count += 1
                    health.target_level = None
                    
                    # Record event
                    self._record_event(DegradationEvent(
                        timestamp=time.time(),
                        service_name=service_name,
                        event_type='recovery',
                        from_level=from_level,
                        to_level=target_level,
                        trigger=strategy.value,
                        reason=f"Recovery using {strategy.value} strategy",
                        success=True
                    ))
                    
                    logger.info(f"Service {service_name} recovered from {from_level.value} to {target_level.value}")
                    return True
                else:
                    logger.error(f"Failed to recover service {service_name} to {target_level.value}")
                    return False
            else:
                logger.debug(f"Recovery not appropriate for {service_name}")
                return False
                
    def activate_fallback(self, service_name: str, fallback_service: Optional[str] = None) -> bool:
        """Activate fallback service for a degraded service."""
        with self.lock:
            # Determine fallback service
            if not fallback_service:
                fallback_services = self.fallback_mappings.get(service_name, [])
                if not fallback_services:
                    logger.warning(f"No fallback services available for {service_name}")
                    return False
                    
                # Select the first available fallback
                for fb_service in fallback_services:
                    if self._is_service_available(fb_service):
                        fallback_service = fb_service
                        break
                        
                if not fallback_service:
                    logger.warning(f"No available fallback services for {service_name}")
                    return False
                    
            # Activate fallback
            success = self._activate_fallback_service(service_name, fallback_service)
            
            if success:
                # Record event
                self._record_event(DegradationEvent(
                    timestamp=time.time(),
                    service_name=service_name,
                    event_type='fallback',
                    from_level=DegradationLevel.OFFLINE,
                    to_level=DegradationLevel.MINIMAL,
                    trigger='fallback_activation',
                    reason=f"Activated fallback service: {fallback_service}",
                    success=True,
                    metadata={'fallback_service': fallback_service}
                ))
                
                logger.info(f"Activated fallback service {fallback_service} for {service_name}")
                return True
            else:
                logger.error(f"Failed to activate fallback service {fallback_service} for {service_name}")
                return False
                
    def start_monitoring(self):
        """Start automatic monitoring and recovery."""
        with self.lock:
            if not self.monitoring_active:
                self.monitoring_active = True
                self.recovery_active = True
                
                # Start monitoring thread
                self.monitoring_thread = threading.Thread(
                    target=self._monitoring_loop,
                    daemon=True,
                    name="DegradationMonitoring"
                )
                self.monitoring_thread.start()
                
                # Start recovery thread
                self.recovery_thread = threading.Thread(
                    target=self._recovery_loop,
                    daemon=True,
                    name="DegradationRecovery"
                )
                self.recovery_thread.start()
                
                logger.info("Started degradation monitoring and recovery")
                
    def stop_monitoring(self):
        """Stop automatic monitoring and recovery."""
        with self.lock:
            self.monitoring_active = False
            self.recovery_active = False
            logger.info("Stopped degradation monitoring and recovery")
            
    def _check_degradation_triggers(self, service_name: str):
        """Check if any degradation rules are triggered for a service."""
        health = self.service_health[service_name]
        
        for rule in sorted(self.degradation_rules, key=lambda r: r.priority, reverse=True):
            if not rule.enabled:
                continue
                
            # Check if service matches pattern
            import re
            if not re.match(rule.service_pattern, service_name):
                continue
                
            # Check trigger conditions
            if self._evaluate_degradation_rule(rule, health):
                logger.info(f"Degradation rule '{rule.name}' triggered for {service_name}")
                
                # Apply degradation
                self.degrade_service(
                    service_name,
                    rule.target_level,
                    f"Automatic degradation: {rule.name}",
                    rule.trigger.value
                )
                break  # Only apply the highest priority rule
                
    def _evaluate_degradation_rule(self, rule: DegradationRule, health: ServiceHealth) -> bool:
        """Evaluate if a degradation rule should trigger."""
        threshold = rule.threshold
        
        if rule.trigger == DegradationTrigger.ERROR_RATE:
            return health.error_rate >= threshold.get('error_rate', 0.1)
            
        elif rule.trigger == DegradationTrigger.RESPONSE_TIME:
            return health.avg_response_time >= threshold.get('avg_response_time', 5.0)
            
        elif rule.trigger == DegradationTrigger.RESOURCE_USAGE:
            cpu_threshold = threshold.get('cpu_usage')
            memory_threshold = threshold.get('memory_usage')
            
            if cpu_threshold and health.resource_usage.get('cpu', 0) >= cpu_threshold:
                return True
            if memory_threshold and health.resource_usage.get('memory', 0) >= memory_threshold:
                return True
                
        return False
        
    def _should_degrade(self, health: ServiceHealth, target_level: DegradationLevel) -> bool:
        """Determine if service should be degraded to target level."""
        current_index = self._get_level_index(health.current_level)
        target_index = self._get_level_index(target_level)
        
        # Only degrade if target is more degraded than current
        return target_index > current_index
        
    def _should_recover(self, health: ServiceHealth, target_level: DegradationLevel) -> bool:
        """Determine if service should be recovered to target level."""
        current_index = self._get_level_index(health.current_level)
        target_index = self._get_level_index(target_level)
        
        # Only recover if target is less degraded than current
        return target_index < current_index
        
    def _get_level_index(self, level: DegradationLevel) -> int:
        """Get numeric index for degradation level."""
        level_order = [DegradationLevel.FULL, DegradationLevel.REDUCED, DegradationLevel.MINIMAL, DegradationLevel.OFFLINE]
        return level_order.index(level)
        
    def _apply_degradation(self, service_name: str, target_level: DegradationLevel,
                          reason: str, trigger: str) -> bool:
        """Apply degradation to a service."""
        try:
            # Use existing degradation manager
            degradation_manager.mark_service_failure(service_name)
            
            # Update service level in degradation manager
            health = degradation_manager.get_service_health(service_name)
            if health:
                health.degradation_level = target_level
                
            # Send notification
            self._send_degradation_notification(service_name, target_level, reason, trigger)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply degradation for {service_name}: {e}")
            return False
            
    def _apply_recovery(self, service_name: str, target_level: DegradationLevel,
                       strategy: RecoveryStrategy) -> bool:
        """Apply recovery to a service."""
        try:
            if strategy == RecoveryStrategy.IMMEDIATE:
                return self._immediate_recovery(service_name, target_level)
            elif strategy == RecoveryStrategy.GRADUAL:
                return self._gradual_recovery(service_name, target_level)
            elif strategy == RecoveryStrategy.SCHEDULED:
                return self._scheduled_recovery(service_name, target_level)
            else:
                logger.warning(f"Manual recovery strategy not implemented for {service_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to apply recovery for {service_name}: {e}")
            return False
            
    def _immediate_recovery(self, service_name: str, target_level: DegradationLevel) -> bool:
        """Apply immediate recovery."""
        # Mark service as successful
        degradation_manager.mark_service_success(service_name)
        
        # Update service level
        health = degradation_manager.get_service_health(service_name)
        if health:
            health.degradation_level = target_level
            
        # Send notification
        self._send_recovery_notification(service_name, target_level, "immediate")
        
        return True
        
    def _gradual_recovery(self, service_name: str, target_level: DegradationLevel) -> bool:
        """Apply gradual recovery."""
        health = self.service_health[service_name]
        current_level = health.current_level
        
        # Calculate intermediate level
        current_index = self._get_level_index(current_level)
        target_index = self._get_level_index(target_level)
        
        if current_index - target_index > 1:
            # Recover one level at a time
            intermediate_index = current_index - 1
            level_order = [DegradationLevel.FULL, DegradationLevel.REDUCED, DegradationLevel.MINIMAL, DegradationLevel.OFFLINE]
            intermediate_level = level_order[intermediate_index]
            
            # Set target for next recovery
            health.target_level = target_level
            
            # Apply intermediate recovery
            return self._immediate_recovery(service_name, intermediate_level)
        else:
            # Direct recovery
            return self._immediate_recovery(service_name, target_level)
            
    def _scheduled_recovery(self, service_name: str, target_level: DegradationLevel) -> bool:
        """Apply scheduled recovery."""
        # Schedule recovery for later (simplified implementation)
        def delayed_recovery():
            time.sleep(60)  # Wait 1 minute
            self._immediate_recovery(service_name, target_level)
            
        threading.Thread(target=delayed_recovery, daemon=True).start()
        return True
        
    def _check_cascade_degradation(self, service_name: str, degradation_level: DegradationLevel):
        """Check if degradation should cascade to dependent services."""
        # Find services that depend on this service
        dependent_services = [
            svc for svc, deps in self.dependency_graph.items()
            if service_name in deps
        ]
        
        for dependent_service in dependent_services:
            if dependent_service in self.service_health:
                dep_health = self.service_health[dependent_service]
                
                # Determine if dependent service should be degraded
                if degradation_level in [DegradationLevel.OFFLINE, DegradationLevel.MINIMAL]:
                    # Critical dependency failure - degrade dependent service
                    target_level = DegradationLevel.REDUCED if degradation_level == DegradationLevel.MINIMAL else DegradationLevel.MINIMAL
                    
                    self.degrade_service(
                        dependent_service,
                        target_level,
                        f"Cascade degradation due to {service_name} failure",
                        "dependency_failure"
                    )
                    
    def _activate_fallback_service(self, service_name: str, fallback_service: str) -> bool:
        """Activate a fallback service."""
        try:
            # Register fallback service if not exists
            if fallback_service not in self.service_health:
                self.register_service(fallback_service)
                
            # Mark fallback as healthy
            degradation_manager.mark_service_success(fallback_service)
            
            # Update routing (this would integrate with actual service routing)
            logger.info(f"Routing {service_name} traffic to {fallback_service}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to activate fallback service {fallback_service}: {e}")
            return False
            
    def _is_service_available(self, service_name: str) -> bool:
        """Check if a service is available for use as fallback."""
        if service_name in self.service_health:
            health = self.service_health[service_name]
            return health.current_level in [DegradationLevel.FULL, DegradationLevel.REDUCED]
        return True  # Assume available if not tracked
        
    def _monitoring_loop(self):
        """Main monitoring loop for automatic degradation."""
        while self.monitoring_active:
            try:
                with self.lock:
                    # Check all services for degradation triggers
                    for service_name in list(self.service_health.keys()):
                        self._check_degradation_triggers(service_name)
                        
                # Sleep between checks
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error
                
    def _recovery_loop(self):
        """Main recovery loop for automatic service recovery."""
        while self.recovery_active:
            try:
                with self.lock:
                    # Check all services for recovery opportunities
                    for service_name, health in self.service_health.items():
                        if health.current_level != DegradationLevel.FULL:
                            self._check_recovery_conditions(service_name, health)
                            
                # Sleep between checks
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in recovery loop: {e}")
                time.sleep(120)  # Wait longer on error
                
    def _check_recovery_conditions(self, service_name: str, health: ServiceHealth):
        """Check if service meets recovery conditions."""
        for rule in self.recovery_rules:
            if not rule.enabled:
                continue
                
            # Check if service matches pattern
            import re
            if not re.match(rule.service_pattern, service_name):
                continue
                
            # Check recovery conditions
            if self._evaluate_recovery_rule(rule, health):
                logger.info(f"Recovery rule '{rule.name}' triggered for {service_name}")
                
                # Apply recovery
                self.recover_service(service_name, rule.target_level, rule.strategy)
                break
                
    def _evaluate_recovery_rule(self, rule: RecoveryRule, health: ServiceHealth) -> bool:
        """Evaluate if a recovery rule should trigger."""
        conditions = rule.conditions
        
        # Check error rate condition
        error_rate_threshold = conditions.get('error_rate_below')
        if error_rate_threshold and health.error_rate >= error_rate_threshold:
            return False
            
        # Check response time condition
        response_time_threshold = conditions.get('response_time_below')
        if response_time_threshold and health.avg_response_time >= response_time_threshold:
            return False
            
        # Check stability duration
        stable_duration = conditions.get('stable_duration', 0)
        if stable_duration > 0:
            time_since_degradation = time.time() - health.last_degradation_time
            if time_since_degradation < stable_duration:
                return False
                
        return True
        
    def _send_degradation_notification(self, service_name: str, level: DegradationLevel,
                                     reason: str, trigger: str):
        """Send notification about service degradation."""
        priority = self._get_notification_priority(level)
        channels = self._get_notification_channels(level)
        
        notification_system.send_notification(
            title=f"Service Degradation - {service_name}",
            message=f"Service {service_name} has been degraded to {level.value}\n\nReason: {reason}\nTrigger: {trigger}",
            priority=priority,
            channels=channels,
            metadata={
                'service_name': service_name,
                'degradation_level': level.value,
                'reason': reason,
                'trigger': trigger
            }
        )
        
    def _send_recovery_notification(self, service_name: str, level: DegradationLevel, strategy: str):
        """Send notification about service recovery."""
        notification_system.send_notification(
            title=f"Service Recovery - {service_name}",
            message=f"Service {service_name} has been recovered to {level.value}\n\nStrategy: {strategy}",
            priority=NotificationPriority.NORMAL,
            channels=[NotificationChannel.LOG, NotificationChannel.SLACK],
            metadata={
                'service_name': service_name,
                'recovery_level': level.value,
                'strategy': strategy
            }
        )
        
    def _get_notification_priority(self, level: DegradationLevel) -> NotificationPriority:
        """Get notification priority for degradation level."""
        priority_map = {
            DegradationLevel.REDUCED: NotificationPriority.NORMAL,
            DegradationLevel.MINIMAL: NotificationPriority.HIGH,
            DegradationLevel.OFFLINE: NotificationPriority.CRITICAL
        }
        return priority_map.get(level, NotificationPriority.NORMAL)
        
    def _get_notification_channels(self, level: DegradationLevel) -> List:
        """Get notification channels for degradation level."""
        channel_map = {
            DegradationLevel.REDUCED: [NotificationChannel.LOG, NotificationChannel.SLACK],
            DegradationLevel.MINIMAL: [NotificationChannel.LOG, NotificationChannel.SLACK, NotificationChannel.WEBHOOK],
            DegradationLevel.OFFLINE: [NotificationChannel.LOG, NotificationChannel.SLACK, NotificationChannel.EMAIL, NotificationChannel.WEBHOOK]
        }
        return channel_map.get(level, [NotificationChannel.LOG])
        
    def _record_event(self, event: DegradationEvent):
        """Record a degradation/recovery event."""
        self.event_history.append(event)
        
    def get_service_status(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive status for a service."""
        with self.lock:
            if service_name not in self.service_health:
                return None
                
            health = self.service_health[service_name]
            
            return {
                'service_name': service_name,
                'current_level': health.current_level.value,
                'target_level': health.target_level.value if health.target_level else None,
                'error_rate': health.error_rate,
                'avg_response_time': health.avg_response_time,
                'resource_usage': health.resource_usage,
                'degradation_count': health.degradation_count,
                'recovery_count': health.recovery_count,
                'last_degradation_time': health.last_degradation_time,
                'last_recovery_time': health.last_recovery_time,
                'dependencies': health.dependencies,
                'fallback_services': health.fallback_services
            }
            
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system degradation status."""
        with self.lock:
            service_statuses = {}
            for service_name in self.service_health:
                service_statuses[service_name] = self.get_service_status(service_name)
                
            # Calculate overall system health
            total_services = len(self.service_health)
            if total_services == 0:
                overall_health = "unknown"
            else:
                degraded_services = sum(
                    1 for health in self.service_health.values()
                    if health.current_level != DegradationLevel.FULL
                )
                
                if degraded_services == 0:
                    overall_health = "healthy"
                elif degraded_services / total_services < 0.3:
                    overall_health = "warning"
                else:
                    overall_health = "critical"
                    
            return {
                'overall_health': overall_health,
                'total_services': total_services,
                'degraded_services': sum(
                    1 for health in self.service_health.values()
                    if health.current_level != DegradationLevel.FULL
                ),
                'offline_services': sum(
                    1 for health in self.service_health.values()
                    if health.current_level == DegradationLevel.OFFLINE
                ),
                'monitoring_active': self.monitoring_active,
                'recovery_active': self.recovery_active,
                'services': service_statuses,
                'recent_events': [
                    {
                        'timestamp': event.timestamp,
                        'service_name': event.service_name,
                        'event_type': event.event_type,
                        'from_level': event.from_level.value,
                        'to_level': event.to_level.value,
                        'reason': event.reason
                    }
                    for event in list(self.event_history)[-10:]  # Last 10 events
                ]
            }


# Global instance
enhanced_degradation_manager = EnhancedDegradationManager()


# Integration functions
def integrate_with_existing_systems():
    """Integrate enhanced degradation manager with existing systems."""
    # Start monitoring
    enhanced_degradation_manager.start_monitoring()
    
    # Register default services
    services_to_register = [
        ('ai_annotation', ['database', 'external_api'], ['rule_based_annotation']),
        ('quality_check', ['database'], ['basic_quality_check']),
        ('data_extraction', ['database'], ['simple_extraction']),
        ('billing', ['database'], []),
        ('export', ['database', 'quality_check'], [])
    ]
    
    for service_name, dependencies, fallbacks in services_to_register:
        enhanced_degradation_manager.register_service(service_name, dependencies, fallbacks)
        
    logger.info("Enhanced degradation manager integrated with existing systems")


# Initialize integration
integrate_with_existing_systems()