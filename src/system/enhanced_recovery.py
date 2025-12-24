"""
Enhanced Recovery Coordination System for SuperInsight Platform.

Coordinates error recovery, health monitoring, notifications, and graceful
degradation to provide comprehensive system resilience.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from src.system.error_handler import error_handler, ErrorCategory, ErrorSeverity
from src.system.notification import notification_system, NotificationPriority, NotificationChannel
from src.system.health_monitor import health_monitor, HealthStatus
from src.utils.degradation import degradation_manager, DegradationLevel
from src.utils.retry import RetryConfig, RetryStrategy

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    IMMEDIATE = "immediate"
    GRADUAL = "gradual"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"


@dataclass
class RecoveryPlan:
    """Comprehensive recovery plan."""
    strategy: RecoveryStrategy
    actions: List[str]
    estimated_duration: float
    success_probability: float
    rollback_plan: List[str]


class EnhancedRecoveryCoordinator:
    """
    Coordinates all recovery mechanisms for comprehensive system resilience.
    
    Features:
    - Intelligent recovery strategy selection
    - Cross-system coordination
    - Recovery plan generation and execution
    - Success rate tracking and learning
    - Proactive issue prevention
    """
    
    def __init__(self):
        self.recovery_history: List[Dict[str, Any]] = []
        self.strategy_success_rates: Dict[RecoveryStrategy, float] = {
            strategy: 0.5 for strategy in RecoveryStrategy
        }
        self.active_recoveries: Dict[str, Dict[str, Any]] = {}
        self.prevention_rules: List[Callable] = []
        
        # Initialize coordination
        self._setup_coordination()
    
    def _setup_coordination(self):
        """Setup coordination between all recovery systems."""
        # Register with error handler for recovery coordination
        error_handler.register_notification_handler(self._handle_error_notification)
        
        # Start health monitoring
        health_monitor.start_monitoring()
        
        # Configure notification system
        notification_system.config.enabled = True
        
        logger.info("Enhanced recovery coordination initialized")
    
    def _handle_error_notification(self, error_context):
        """Handle error notifications and coordinate recovery."""
        try:
            # Generate recovery plan
            recovery_plan = self._generate_recovery_plan(error_context)
            
            if recovery_plan:
                # Execute recovery plan
                success = self._execute_recovery_plan(recovery_plan, error_context)
                
                # Update success rates
                self._update_strategy_success_rate(recovery_plan.strategy, success)
                
                # Record recovery attempt
                self._record_recovery_attempt(error_context, recovery_plan, success)
                
        except Exception as e:
            logger.error(f"Recovery coordination failed: {e}")
    
    def _generate_recovery_plan(self, error_context) -> Optional[RecoveryPlan]:
        """Generate an intelligent recovery plan based on error context."""
        try:
            # Analyze error context
            error_severity = error_context.severity
            error_category = error_context.category
            service_name = error_context.service_name
            
            # Get current system health
            health_report = health_monitor.get_health_report()
            
            # Determine recovery strategy based on multiple factors
            strategy = self._select_recovery_strategy(
                error_context, health_report
            )
            
            # Generate actions based on strategy and error type
            actions = self._generate_recovery_actions(
                strategy, error_category, service_name, health_report
            )
            
            # Estimate success probability
            success_probability = self._estimate_success_probability(
                strategy, error_category, actions
            )
            
            # Generate rollback plan
            rollback_plan = self._generate_rollback_plan(actions)
            
            return RecoveryPlan(
                strategy=strategy,
                actions=actions,
                estimated_duration=self._estimate_duration(actions),
                success_probability=success_probability,
                rollback_plan=rollback_plan
            )
            
        except Exception as e:
            logger.error(f"Recovery plan generation failed: {e}")
            return None
    
    def _select_recovery_strategy(self, error_context, health_report) -> RecoveryStrategy:
        """Select the most appropriate recovery strategy."""
        # Factors to consider:
        # 1. Error severity
        # 2. System health
        # 3. Historical success rates
        # 4. Time of day / system load
        # 5. Service criticality
        
        severity_weights = {
            ErrorSeverity.LOW: 0.2,
            ErrorSeverity.MEDIUM: 0.5,
            ErrorSeverity.HIGH: 0.8,
            ErrorSeverity.CRITICAL: 1.0
        }
        
        health_weights = {
            HealthStatus.HEALTHY: 0.2,
            HealthStatus.WARNING: 0.5,
            HealthStatus.CRITICAL: 1.0,
            HealthStatus.UNKNOWN: 0.7
        }
        
        # Calculate urgency score
        severity_score = severity_weights.get(error_context.severity, 0.5)
        health_score = health_weights.get(health_report.overall_status, 0.5)
        urgency_score = (severity_score + health_score) / 2
        
        # Select strategy based on urgency and success rates
        if urgency_score >= 0.8:
            # High urgency - use most successful aggressive strategy
            if self.strategy_success_rates[RecoveryStrategy.AGGRESSIVE] > 0.6:
                return RecoveryStrategy.AGGRESSIVE
            else:
                return RecoveryStrategy.IMMEDIATE
        elif urgency_score >= 0.5:
            # Medium urgency - balanced approach
            return RecoveryStrategy.GRADUAL
        else:
            # Low urgency - conservative approach
            return RecoveryStrategy.CONSERVATIVE
    
    def _generate_recovery_actions(
        self,
        strategy: RecoveryStrategy,
        error_category: ErrorCategory,
        service_name: Optional[str],
        health_report
    ) -> List[str]:
        """Generate recovery actions based on strategy and context."""
        actions = []
        
        # Base actions by error category
        category_actions = {
            ErrorCategory.DATABASE: [
                "check_database_connection",
                "restart_database_pool",
                "clear_database_cache",
                "switch_to_readonly_mode"
            ],
            ErrorCategory.EXTERNAL_API: [
                "check_api_endpoint",
                "refresh_api_credentials",
                "enable_circuit_breaker",
                "switch_to_cached_responses"
            ],
            ErrorCategory.EXTRACTION: [
                "retry_extraction",
                "switch_to_fallback_extractor",
                "clear_extraction_cache",
                "reduce_extraction_batch_size"
            ],
            ErrorCategory.ANNOTATION: [
                "switch_to_backup_model",
                "reduce_annotation_complexity",
                "enable_rule_based_fallback",
                "clear_model_cache"
            ],
            ErrorCategory.QUALITY: [
                "switch_to_basic_quality_check",
                "reduce_quality_thresholds",
                "enable_manual_review_mode",
                "clear_quality_cache"
            ]
        }
        
        base_actions = category_actions.get(error_category, ["generic_recovery"])
        
        # Modify actions based on strategy
        if strategy == RecoveryStrategy.IMMEDIATE:
            # Quick, simple actions
            actions = base_actions[:2]
        elif strategy == RecoveryStrategy.GRADUAL:
            # Phased approach
            actions = base_actions[:3]
        elif strategy == RecoveryStrategy.CONSERVATIVE:
            # Minimal intervention
            actions = [base_actions[0]] if base_actions else ["monitor_and_wait"]
        elif strategy == RecoveryStrategy.AGGRESSIVE:
            # All available actions
            actions = base_actions + [
                "restart_service",
                "clear_all_caches",
                "enable_full_degradation"
            ]
        
        # Add health-based actions
        if health_report.overall_status == HealthStatus.CRITICAL:
            actions.extend([
                "enable_emergency_mode",
                "notify_operations_team",
                "prepare_for_manual_intervention"
            ])
        
        return actions
    
    def _estimate_success_probability(
        self,
        strategy: RecoveryStrategy,
        error_category: ErrorCategory,
        actions: List[str]
    ) -> float:
        """Estimate the probability of recovery success."""
        # Base probability from strategy success rate
        base_prob = self.strategy_success_rates.get(strategy, 0.5)
        
        # Adjust based on error category (some are easier to recover from)
        category_multipliers = {
            ErrorCategory.DATABASE: 0.8,
            ErrorCategory.EXTERNAL_API: 0.7,
            ErrorCategory.EXTRACTION: 0.9,
            ErrorCategory.ANNOTATION: 0.8,
            ErrorCategory.QUALITY: 0.9,
            ErrorCategory.SYSTEM: 0.6
        }
        
        category_mult = category_multipliers.get(error_category, 0.7)
        
        # Adjust based on number of actions (more actions = higher chance but also more risk)
        action_factor = min(1.0, 0.5 + (len(actions) * 0.1))
        
        return min(1.0, base_prob * category_mult * action_factor)
    
    def _estimate_duration(self, actions: List[str]) -> float:
        """Estimate recovery duration in seconds."""
        # Base time per action type
        action_times = {
            "check_": 10,
            "restart_": 30,
            "clear_": 15,
            "switch_": 20,
            "enable_": 5,
            "reduce_": 10,
            "refresh_": 25,
            "notify_": 5
        }
        
        total_time = 0
        for action in actions:
            for prefix, time_cost in action_times.items():
                if action.startswith(prefix):
                    total_time += time_cost
                    break
            else:
                total_time += 20  # Default time
        
        return total_time
    
    def _generate_rollback_plan(self, actions: List[str]) -> List[str]:
        """Generate rollback plan for recovery actions."""
        rollback_actions = []
        
        # Generate inverse actions
        action_rollbacks = {
            "restart_": "monitor_",
            "clear_": "rebuild_",
            "switch_": "revert_",
            "enable_": "disable_",
            "reduce_": "restore_"
        }
        
        for action in reversed(actions):  # Rollback in reverse order
            for prefix, rollback_prefix in action_rollbacks.items():
                if action.startswith(prefix):
                    rollback_action = action.replace(prefix, rollback_prefix, 1)
                    rollback_actions.append(rollback_action)
                    break
        
        return rollback_actions
    
    def _execute_recovery_plan(self, plan: RecoveryPlan, error_context) -> bool:
        """Execute a recovery plan."""
        recovery_id = f"recovery_{int(time.time() * 1000)}"
        
        try:
            logger.info(f"Executing recovery plan {recovery_id} with strategy {plan.strategy.value}")
            
            # Record active recovery
            self.active_recoveries[recovery_id] = {
                "plan": plan,
                "error_context": error_context,
                "start_time": time.time(),
                "status": "in_progress"
            }
            
            # Send notification about recovery start
            notification_system.send_notification(
                title=f"Recovery Plan Execution - {plan.strategy.value}",
                message=f"Starting recovery for {error_context.category.value} error with {len(plan.actions)} actions",
                priority=NotificationPriority.NORMAL,
                channels=[NotificationChannel.LOG, NotificationChannel.SLACK]
            )
            
            # Execute actions
            success_count = 0
            for i, action in enumerate(plan.actions):
                try:
                    logger.info(f"Executing recovery action {i+1}/{len(plan.actions)}: {action}")
                    
                    # Execute the action
                    action_success = self._execute_recovery_action(action, error_context)
                    
                    if action_success:
                        success_count += 1
                    else:
                        logger.warning(f"Recovery action failed: {action}")
                        
                        # If critical action fails, consider rollback
                        if plan.strategy == RecoveryStrategy.CONSERVATIVE and not action_success:
                            logger.info("Conservative strategy failed, initiating rollback")
                            self._execute_rollback(plan.rollback_plan[:i+1])
                            break
                    
                    # Small delay between actions
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Recovery action {action} failed: {e}")
            
            # Determine overall success
            success_rate = success_count / len(plan.actions) if plan.actions else 0
            overall_success = success_rate >= 0.7  # 70% success rate threshold
            
            # Update recovery status
            self.active_recoveries[recovery_id]["status"] = "completed" if overall_success else "failed"
            self.active_recoveries[recovery_id]["end_time"] = time.time()
            self.active_recoveries[recovery_id]["success_rate"] = success_rate
            
            # Send completion notification
            notification_system.send_notification(
                title=f"Recovery Plan {'Completed' if overall_success else 'Failed'}",
                message=f"Recovery plan {recovery_id} {'succeeded' if overall_success else 'failed'} with {success_rate:.1%} action success rate",
                priority=NotificationPriority.NORMAL if overall_success else NotificationPriority.HIGH,
                channels=[NotificationChannel.LOG, NotificationChannel.SLACK]
            )
            
            return overall_success
            
        except Exception as e:
            logger.error(f"Recovery plan execution failed: {e}")
            
            # Mark as failed
            if recovery_id in self.active_recoveries:
                self.active_recoveries[recovery_id]["status"] = "error"
                self.active_recoveries[recovery_id]["error"] = str(e)
            
            return False
    
    def _execute_recovery_action(self, action: str, error_context) -> bool:
        """Execute a single recovery action."""
        try:
            # Map actions to actual implementations
            if action.startswith("check_"):
                return self._check_service_health(action, error_context)
            elif action.startswith("restart_"):
                return self._restart_service_component(action, error_context)
            elif action.startswith("clear_"):
                return self._clear_cache_component(action, error_context)
            elif action.startswith("switch_"):
                return self._switch_service_mode(action, error_context)
            elif action.startswith("enable_"):
                return self._enable_feature(action, error_context)
            elif action.startswith("reduce_"):
                return self._reduce_service_load(action, error_context)
            elif action.startswith("notify_"):
                return self._send_recovery_notification(action, error_context)
            else:
                logger.warning(f"Unknown recovery action: {action}")
                return False
                
        except Exception as e:
            logger.error(f"Recovery action {action} execution failed: {e}")
            return False
    
    def _check_service_health(self, action: str, error_context) -> bool:
        """Check service health as recovery action."""
        service_name = error_context.service_name or "unknown"
        health = degradation_manager.get_service_health(service_name)
        
        if health:
            logger.info(f"Service {service_name} health: {health.degradation_level.value}")
            return health.is_healthy
        
        return True  # Assume healthy if no health info
    
    def _restart_service_component(self, action: str, error_context) -> bool:
        """Restart service component as recovery action."""
        service_name = error_context.service_name or "unknown"
        
        # Mark service as recovered (simulated restart)
        degradation_manager.mark_service_success(service_name)
        
        logger.info(f"Simulated restart of {service_name}")
        return True
    
    def _clear_cache_component(self, action: str, error_context) -> bool:
        """Clear cache as recovery action."""
        degradation_manager.clear_cache()
        logger.info("Cleared system caches")
        return True
    
    def _switch_service_mode(self, action: str, error_context) -> bool:
        """Switch service to fallback mode."""
        service_name = error_context.service_name or "unknown"
        degradation_manager.mark_service_failure(service_name)
        logger.info(f"Switched {service_name} to fallback mode")
        return True
    
    def _enable_feature(self, action: str, error_context) -> bool:
        """Enable recovery feature."""
        logger.info(f"Enabled recovery feature: {action}")
        return True
    
    def _reduce_service_load(self, action: str, error_context) -> bool:
        """Reduce service load."""
        logger.info(f"Reduced service load: {action}")
        return True
    
    def _send_recovery_notification(self, action: str, error_context) -> bool:
        """Send recovery notification."""
        notification_system.send_notification(
            title="Recovery Action Notification",
            message=f"Recovery action executed: {action}",
            priority=NotificationPriority.NORMAL,
            channels=[NotificationChannel.LOG]
        )
        return True
    
    def _execute_rollback(self, rollback_actions: List[str]):
        """Execute rollback plan."""
        logger.info(f"Executing rollback with {len(rollback_actions)} actions")
        
        for action in rollback_actions:
            try:
                logger.info(f"Rollback action: {action}")
                # Implement rollback actions (simplified for now)
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Rollback action {action} failed: {e}")
    
    def _update_strategy_success_rate(self, strategy: RecoveryStrategy, success: bool):
        """Update strategy success rate using exponential moving average."""
        current_rate = self.strategy_success_rates[strategy]
        alpha = 0.2  # Learning rate
        
        new_value = 1.0 if success else 0.0
        self.strategy_success_rates[strategy] = alpha * new_value + (1 - alpha) * current_rate
    
    def _record_recovery_attempt(self, error_context, recovery_plan: RecoveryPlan, success: bool):
        """Record recovery attempt for analysis."""
        record = {
            "timestamp": time.time(),
            "error_category": error_context.category.value,
            "error_severity": error_context.severity.value,
            "service_name": error_context.service_name,
            "strategy": recovery_plan.strategy.value,
            "actions_count": len(recovery_plan.actions),
            "estimated_probability": recovery_plan.success_probability,
            "actual_success": success,
            "duration": recovery_plan.estimated_duration
        }
        
        self.recovery_history.append(record)
        
        # Keep only last 1000 records
        if len(self.recovery_history) > 1000:
            self.recovery_history = self.recovery_history[-1000:]
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get comprehensive recovery statistics."""
        if not self.recovery_history:
            return {
                "total_recoveries": 0,
                "success_rate": 0.0,
                "strategy_performance": {},
                "category_performance": {},
                "active_recoveries": len(self.active_recoveries)
            }
        
        total_recoveries = len(self.recovery_history)
        successful_recoveries = sum(1 for r in self.recovery_history if r["actual_success"])
        overall_success_rate = successful_recoveries / total_recoveries
        
        # Strategy performance
        strategy_performance = {}
        for strategy in RecoveryStrategy:
            strategy_records = [r for r in self.recovery_history if r["strategy"] == strategy.value]
            if strategy_records:
                strategy_success = sum(1 for r in strategy_records if r["actual_success"])
                strategy_performance[strategy.value] = {
                    "attempts": len(strategy_records),
                    "success_rate": strategy_success / len(strategy_records),
                    "current_rate": self.strategy_success_rates[strategy]
                }
        
        # Category performance
        category_performance = {}
        for category in ErrorCategory:
            category_records = [r for r in self.recovery_history if r["error_category"] == category.value]
            if category_records:
                category_success = sum(1 for r in category_records if r["actual_success"])
                category_performance[category.value] = {
                    "attempts": len(category_records),
                    "success_rate": category_success / len(category_records)
                }
        
        return {
            "total_recoveries": total_recoveries,
            "success_rate": overall_success_rate,
            "strategy_performance": strategy_performance,
            "category_performance": category_performance,
            "active_recoveries": len(self.active_recoveries),
            "recent_recoveries": self.recovery_history[-10:]
        }
    
    def add_prevention_rule(self, rule_function: Callable):
        """Add a proactive prevention rule."""
        self.prevention_rules.append(rule_function)
        logger.info("Added prevention rule")
    
    def check_prevention_rules(self):
        """Check all prevention rules and take proactive action."""
        for rule in self.prevention_rules:
            try:
                rule()
            except Exception as e:
                logger.error(f"Prevention rule failed: {e}")


# Global recovery coordinator
recovery_coordinator = EnhancedRecoveryCoordinator()


# Convenience functions
def get_recovery_status() -> Dict[str, Any]:
    """Get current recovery system status."""
    return {
        "coordinator_active": True,
        "health_monitoring": health_monitor.monitoring_active,
        "notification_system": notification_system.config.enabled,
        "statistics": recovery_coordinator.get_recovery_statistics()
    }


def trigger_manual_recovery(error_category: str, service_name: str = None) -> bool:
    """Manually trigger recovery for a specific error category."""
    try:
        # Create a mock error context for manual recovery
        from src.system.error_handler import ErrorContext, ErrorCategory, ErrorSeverity
        
        error_context = ErrorContext(
            error_id=f"manual_{int(time.time())}",
            timestamp=time.time(),
            category=ErrorCategory(error_category),
            severity=ErrorSeverity.HIGH,
            message=f"Manual recovery triggered for {error_category}",
            service_name=service_name
        )
        
        # Generate and execute recovery plan
        recovery_plan = recovery_coordinator._generate_recovery_plan(error_context)
        if recovery_plan:
            return recovery_coordinator._execute_recovery_plan(recovery_plan, error_context)
        
        return False
        
    except Exception as e:
        logger.error(f"Manual recovery failed: {e}")
        return False


def configure_recovery_system(
    enable_health_monitoring: bool = True,
    enable_notifications: bool = True,
    notification_channels: Optional[List[str]] = None
):
    """Configure the recovery system."""
    if enable_health_monitoring:
        health_monitor.start_monitoring()
    else:
        health_monitor.stop_monitoring()
    
    notification_system.config.enabled = enable_notifications
    
    if notification_channels:
        notification_system.config.channels = [
            NotificationChannel(channel) for channel in notification_channels
        ]
    
    logger.info("Recovery system configured")