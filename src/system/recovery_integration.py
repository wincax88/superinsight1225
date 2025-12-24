"""
Recovery System Integration for SuperInsight Platform.

Integrates all enhanced recovery mechanisms into a unified system
with coordinated error handling, intelligent notifications, and
graceful degradation.
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from src.system.error_handler import error_handler, ErrorCategory, ErrorSeverity
from src.system.advanced_recovery import advanced_recovery_system, RecoveryMode, RecoveryPriority
from src.system.intelligent_notifications import intelligent_notification_filter, filter_notification
from src.system.enhanced_degradation import enhanced_degradation_manager
from src.system.notification import notification_system, NotificationPriority, NotificationChannel

logger = logging.getLogger(__name__)


class SystemHealthStatus(Enum):
    """Overall system health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class SystemHealthReport:
    """Comprehensive system health report."""
    status: SystemHealthStatus
    timestamp: float
    error_rate: float
    recovery_rate: float
    degraded_services: int
    total_services: int
    active_recoveries: int
    notification_filter_rate: float
    recommendations: List[str]
    metadata: Dict[str, Any]


class IntegratedRecoverySystem:
    """
    Integrated recovery system that coordinates all recovery mechanisms.
    
    Features:
    - Unified error handling with intelligent recovery
    - Smart notification filtering with escalation
    - Coordinated graceful degradation
    - System-wide health monitoring
    - Adaptive recovery strategies
    - Performance optimization
    """
    
    def __init__(self):
        self.system_health_history: List[SystemHealthReport] = []
        self.integration_active = False
        self.health_monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
        
        # Performance metrics
        self.metrics = {
            'total_errors_handled': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'notifications_sent': 0,
            'notifications_filtered': 0,
            'services_degraded': 0,
            'services_recovered': 0
        }
        
        self._setup_integration()
        
    def _setup_integration(self):
        """Setup integration between all recovery systems."""
        # Register integrated error handler
        error_handler.register_notification_handler(self._handle_integrated_error)
        
        # Configure notification system integration
        self._setup_notification_integration()
        
        # Setup degradation monitoring
        self._setup_degradation_integration()
        
        logger.info("Integrated recovery system initialized")
        
    def _setup_notification_integration(self):
        """Setup notification system integration."""
        # Override default notification sending with intelligent filtering
        original_send = notification_system.send_notification
        
        def filtered_send_notification(*args, **kwargs):
            # Apply intelligent filtering if this is an error notification
            if 'error_context' in kwargs.get('metadata', {}):
                error_context = kwargs['metadata']['error_context']
                should_send, notification_context = filter_notification(error_context)
                
                if should_send:
                    self.metrics['notifications_sent'] += 1
                    return original_send(*args, **kwargs)
                else:
                    self.metrics['notifications_filtered'] += 1
                    logger.debug(f"Notification filtered: {notification_context.suppression_reason}")
                    return True
            else:
                # Non-error notifications go through normally
                return original_send(*args, **kwargs)
                
        # Replace the method (this is a simplified approach)
        notification_system._original_send_notification = original_send
        notification_system.send_notification = filtered_send_notification
        
    def _setup_degradation_integration(self):
        """Setup degradation system integration."""
        # Register services with enhanced degradation manager
        services = [
            'ai_annotation', 'quality_check', 'data_extraction',
            'billing', 'export', 'database', 'external_api'
        ]
        
        for service in services:
            if service not in enhanced_degradation_manager.service_health:
                enhanced_degradation_manager.register_service(service)
                
    def _handle_integrated_error(self, error_context):
        """Handle errors with integrated recovery approach."""
        with self.lock:
            self.metrics['total_errors_handled'] += 1
            
            try:
                # Step 1: Apply intelligent notification filtering
                should_notify, notification_context = filter_notification(error_context)
                
                # Step 2: Determine recovery mode based on error severity and system state
                recovery_mode = self._determine_recovery_mode(error_context)
                
                # Step 3: Execute advanced recovery
                recovery_success = advanced_recovery_system.handle_error_with_advanced_recovery(
                    error_context, recovery_mode
                )
                
                # Step 4: Update degradation status if needed
                self._update_degradation_status(error_context, recovery_success)
                
                # Step 5: Send notification if appropriate
                if should_notify:
                    self._send_integrated_notification(error_context, notification_context, recovery_success)
                    
                # Step 6: Update metrics
                if recovery_success:
                    self.metrics['successful_recoveries'] += 1
                else:
                    self.metrics['failed_recoveries'] += 1
                    
                # Step 7: Check system health and trigger system-wide actions if needed
                self._check_system_health()
                
                return recovery_success
                
            except Exception as e:
                logger.error(f"Integrated error handling failed: {e}")
                self.metrics['failed_recoveries'] += 1
                return False
                
    def _determine_recovery_mode(self, error_context) -> RecoveryMode:
        """Determine appropriate recovery mode based on error and system context."""
        # Get current system health
        system_status = self.get_system_health_status()
        
        # Critical errors or emergency system state
        if (error_context.severity == ErrorSeverity.CRITICAL or 
            system_status.status == SystemHealthStatus.EMERGENCY):
            return RecoveryMode.EMERGENCY
            
        # High severity errors or critical system state
        elif (error_context.severity == ErrorSeverity.HIGH or 
              system_status.status == SystemHealthStatus.CRITICAL):
            return RecoveryMode.AUTOMATIC
            
        # Medium severity errors or degraded system state
        elif (error_context.severity == ErrorSeverity.MEDIUM or 
              system_status.status == SystemHealthStatus.DEGRADED):
            return RecoveryMode.SEMI_AUTOMATIC
            
        # Low severity errors or healthy system state
        else:
            return RecoveryMode.AUTOMATIC
            
    def _update_degradation_status(self, error_context, recovery_success: bool):
        """Update service degradation status based on error and recovery outcome."""
        service_name = error_context.service_name
        
        if not service_name:
            return
            
        # Update service metrics
        metrics = {
            'error_rate': 1.0 if not recovery_success else 0.0,
            'response_time': 10.0 if not recovery_success else 1.0,  # Simulate response time
            'resource_usage': {'cpu': 0.5, 'memory': 0.6}
        }
        
        enhanced_degradation_manager.update_service_metrics(service_name, metrics)
        
        # If recovery failed and error is severe, consider degradation
        if not recovery_success and error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            from src.utils.degradation import DegradationLevel
            
            # Determine degradation level based on error severity
            if error_context.severity == ErrorSeverity.CRITICAL:
                target_level = DegradationLevel.MINIMAL
            else:
                target_level = DegradationLevel.REDUCED
                
            enhanced_degradation_manager.degrade_service(
                service_name,
                target_level,
                f"Recovery failed for {error_context.category.value} error",
                "recovery_failure"
            )
            
            self.metrics['services_degraded'] += 1
            
    def _send_integrated_notification(self, error_context, notification_context, recovery_success: bool):
        """Send integrated notification with recovery context."""
        # Enhance notification with recovery information
        recovery_status = "successful" if recovery_success else "failed"
        
        enhanced_message = f"""
Error Details:
• ID: {error_context.error_id}
• Service: {error_context.service_name or 'unknown'}
• Category: {error_context.category.value}
• Severity: {error_context.severity.value}
• Message: {error_context.message}

Recovery Status: {recovery_status}

Notification Context:
• Similar errors in last hour: {notification_context.similar_count}
• Escalation level: {notification_context.escalation_level.value}
• Correlation group: {notification_context.correlation_group or 'none'}

System Health: {self.get_system_health_status().status.value}
"""
        
        # Determine priority and channels based on recovery success and escalation
        if not recovery_success and error_context.severity == ErrorSeverity.CRITICAL:
            priority = NotificationPriority.CRITICAL
            channels = [NotificationChannel.LOG, NotificationChannel.SLACK, NotificationChannel.EMAIL, NotificationChannel.WEBHOOK]
        elif notification_context.escalation_level.value >= 2:
            priority = NotificationPriority.HIGH
            channels = [NotificationChannel.LOG, NotificationChannel.SLACK, NotificationChannel.WEBHOOK]
        else:
            priority = NotificationPriority.NORMAL
            channels = [NotificationChannel.LOG, NotificationChannel.SLACK]
            
        # Send notification using original method to avoid recursion
        notification_system._original_send_notification(
            title=f"Integrated Recovery Report - {error_context.category.value}",
            message=enhanced_message,
            priority=priority,
            channels=channels,
            metadata={
                'error_context': {
                    'error_id': error_context.error_id,
                    'category': error_context.category.value,
                    'severity': error_context.severity.value,
                    'service_name': error_context.service_name
                },
                'recovery_success': recovery_success,
                'notification_context': {
                    'similar_count': notification_context.similar_count,
                    'escalation_level': notification_context.escalation_level.value,
                    'correlation_group': notification_context.correlation_group
                },
                'integrated_recovery': True
            }
        )
        
        self.metrics['notifications_sent'] += 1
        
    def _check_system_health(self):
        """Check overall system health and trigger system-wide actions if needed."""
        health_status = self.get_system_health_status()
        
        # Store health report
        self.system_health_history.append(health_status)
        
        # Keep only last 100 reports
        if len(self.system_health_history) > 100:
            self.system_health_history = self.system_health_history[-100:]
            
        # Trigger system-wide actions based on health status
        if health_status.status == SystemHealthStatus.EMERGENCY:
            self._trigger_emergency_mode()
        elif health_status.status == SystemHealthStatus.CRITICAL:
            self._trigger_critical_mode()
        elif health_status.status == SystemHealthStatus.DEGRADED:
            self._trigger_degraded_mode()
            
    def _trigger_emergency_mode(self):
        """Trigger emergency mode for system-wide critical failures."""
        logger.critical("EMERGENCY MODE ACTIVATED - System-wide critical failures detected")
        
        # Degrade all non-critical services to minimal
        from src.utils.degradation import DegradationLevel
        
        non_critical_services = ['export', 'quality_check', 'ai_annotation']
        for service in non_critical_services:
            enhanced_degradation_manager.degrade_service(
                service,
                DegradationLevel.MINIMAL,
                "Emergency mode activation",
                "system_emergency"
            )
            
        # Send critical notification
        notification_system._original_send_notification(
            title="SYSTEM EMERGENCY MODE ACTIVATED",
            message="Critical system-wide failures detected. Emergency mode activated. Immediate attention required.",
            priority=NotificationPriority.CRITICAL,
            channels=[NotificationChannel.LOG, NotificationChannel.SLACK, NotificationChannel.EMAIL, NotificationChannel.WEBHOOK],
            metadata={'emergency_mode': True, 'system_health': 'emergency'}
        )
        
    def _trigger_critical_mode(self):
        """Trigger critical mode for severe system degradation."""
        logger.error("CRITICAL MODE ACTIVATED - Severe system degradation detected")
        
        # Activate fallback services for critical components
        critical_services = ['database', 'billing']
        for service in critical_services:
            enhanced_degradation_manager.activate_fallback(service)
            
        # Send high priority notification
        notification_system._original_send_notification(
            title="SYSTEM CRITICAL MODE ACTIVATED",
            message="Severe system degradation detected. Critical mode activated. Fallback services enabled.",
            priority=NotificationPriority.HIGH,
            channels=[NotificationChannel.LOG, NotificationChannel.SLACK, NotificationChannel.WEBHOOK],
            metadata={'critical_mode': True, 'system_health': 'critical'}
        )
        
    def _trigger_degraded_mode(self):
        """Trigger degraded mode for moderate system issues."""
        logger.warning("DEGRADED MODE ACTIVATED - Moderate system issues detected")
        
        # Optimize notification filtering to reduce noise
        intelligent_notification_filter.optimize_filters()
        
        # Send normal priority notification
        notification_system._original_send_notification(
            title="System Degraded Mode Activated",
            message="Moderate system issues detected. Degraded mode activated. Enhanced monitoring enabled.",
            priority=NotificationPriority.NORMAL,
            channels=[NotificationChannel.LOG, NotificationChannel.SLACK],
            metadata={'degraded_mode': True, 'system_health': 'degraded'}
        )
        
    def get_system_health_status(self) -> SystemHealthReport:
        """Get comprehensive system health status."""
        with self.lock:
            current_time = time.time()
            
            # Get degradation system status
            degradation_status = enhanced_degradation_manager.get_system_status()
            
            # Get recovery system statistics
            recovery_stats = advanced_recovery_system.get_recovery_statistics()
            
            # Get notification filter statistics
            filter_stats = intelligent_notification_filter.get_filter_statistics()
            
            # Calculate error rate (errors per minute over last hour)
            recent_errors = self.metrics['total_errors_handled']  # Simplified
            error_rate = recent_errors / 60.0  # Errors per minute
            
            # Calculate recovery rate
            total_recoveries = self.metrics['successful_recoveries'] + self.metrics['failed_recoveries']
            recovery_rate = (self.metrics['successful_recoveries'] / max(total_recoveries, 1)) * 100
            
            # Calculate notification filter rate
            total_notifications = self.metrics['notifications_sent'] + self.metrics['notifications_filtered']
            notification_filter_rate = (self.metrics['notifications_filtered'] / max(total_notifications, 1)) * 100
            
            # Determine overall system health status
            status = self._calculate_system_health_status(
                degradation_status, error_rate, recovery_rate
            )
            
            # Generate recommendations
            recommendations = self._generate_health_recommendations(
                status, degradation_status, error_rate, recovery_rate
            )
            
            return SystemHealthReport(
                status=status,
                timestamp=current_time,
                error_rate=error_rate,
                recovery_rate=recovery_rate,
                degraded_services=degradation_status['degraded_services'],
                total_services=degradation_status['total_services'],
                active_recoveries=recovery_stats['active_recoveries'],
                notification_filter_rate=notification_filter_rate,
                recommendations=recommendations,
                metadata={
                    'degradation_status': degradation_status,
                    'recovery_stats': recovery_stats,
                    'filter_stats': filter_stats,
                    'metrics': self.metrics.copy()
                }
            )
            
    def _calculate_system_health_status(
        self, degradation_status: Dict[str, Any], error_rate: float, recovery_rate: float
    ) -> SystemHealthStatus:
        """Calculate overall system health status."""
        
        # Check for emergency conditions
        if (degradation_status['offline_services'] > 0 or 
            error_rate > 10.0 or 
            recovery_rate < 20.0):
            return SystemHealthStatus.EMERGENCY
            
        # Check for critical conditions
        if (degradation_status['degraded_services'] > degradation_status['total_services'] * 0.5 or
            error_rate > 5.0 or
            recovery_rate < 50.0):
            return SystemHealthStatus.CRITICAL
            
        # Check for degraded conditions
        if (degradation_status['degraded_services'] > 0 or
            error_rate > 2.0 or
            recovery_rate < 80.0):
            return SystemHealthStatus.DEGRADED
            
        # Check for warning conditions
        if (error_rate > 1.0 or recovery_rate < 90.0):
            return SystemHealthStatus.WARNING
            
        # System is healthy
        return SystemHealthStatus.HEALTHY
        
    def _generate_health_recommendations(
        self, status: SystemHealthStatus, degradation_status: Dict[str, Any],
        error_rate: float, recovery_rate: float
    ) -> List[str]:
        """Generate health recommendations based on system status."""
        recommendations = []
        
        if status == SystemHealthStatus.EMERGENCY:
            recommendations.extend([
                "IMMEDIATE ACTION REQUIRED: System in emergency state",
                "Check offline services and restore critical functionality",
                "Consider manual intervention for failed automatic recoveries",
                "Review system logs for root cause analysis"
            ])
        elif status == SystemHealthStatus.CRITICAL:
            recommendations.extend([
                "Urgent attention required: Multiple services degraded",
                "Investigate high error rate and recovery failures",
                "Consider scaling up resources or enabling additional fallbacks",
                "Review and optimize recovery strategies"
            ])
        elif status == SystemHealthStatus.DEGRADED:
            recommendations.extend([
                "Monitor degraded services closely",
                "Investigate causes of service degradation",
                "Consider proactive measures to prevent further degradation",
                "Review notification filtering to reduce noise"
            ])
        elif status == SystemHealthStatus.WARNING:
            recommendations.extend([
                "Monitor system performance trends",
                "Review recent error patterns for potential issues",
                "Consider optimizing recovery thresholds",
                "Ensure monitoring coverage is adequate"
            ])
        else:  # HEALTHY
            recommendations.extend([
                "System operating normally",
                "Continue regular monitoring",
                "Consider optimizing notification filters based on recent feedback",
                "Review recovery performance for potential improvements"
            ])
            
        # Add specific recommendations based on metrics
        if error_rate > 5.0:
            recommendations.append(f"High error rate detected ({error_rate:.1f}/min) - investigate error sources")
            
        if recovery_rate < 70.0:
            recommendations.append(f"Low recovery rate ({recovery_rate:.1f}%) - review recovery strategies")
            
        if degradation_status['degraded_services'] > 3:
            recommendations.append("Multiple services degraded - check for systemic issues")
            
        return recommendations
        
    def start_health_monitoring(self):
        """Start continuous system health monitoring."""
        with self.lock:
            if not self.integration_active:
                self.integration_active = True
                
                self.health_monitor_thread = threading.Thread(
                    target=self._health_monitoring_loop,
                    daemon=True,
                    name="IntegratedHealthMonitoring"
                )
                self.health_monitor_thread.start()
                
                logger.info("Started integrated health monitoring")
                
    def stop_health_monitoring(self):
        """Stop continuous system health monitoring."""
        with self.lock:
            self.integration_active = False
            logger.info("Stopped integrated health monitoring")
            
    def _health_monitoring_loop(self):
        """Main health monitoring loop."""
        while self.integration_active:
            try:
                # Check system health
                health_status = self.get_system_health_status()
                
                # Log health status periodically
                if len(self.system_health_history) % 10 == 0:  # Every 10 checks
                    logger.info(f"System health: {health_status.status.value} "
                              f"(Error rate: {health_status.error_rate:.1f}/min, "
                              f"Recovery rate: {health_status.recovery_rate:.1f}%)")
                    
                # Sleep between checks
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(120)  # Wait longer on error
                
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all integrated systems."""
        with self.lock:
            health_status = self.get_system_health_status()
            
            return {
                'system_health': {
                    'status': health_status.status.value,
                    'error_rate': health_status.error_rate,
                    'recovery_rate': health_status.recovery_rate,
                    'degraded_services': health_status.degraded_services,
                    'total_services': health_status.total_services,
                    'recommendations': health_status.recommendations
                },
                'recovery_system': advanced_recovery_system.get_recovery_statistics(),
                'notification_filter': intelligent_notification_filter.get_filter_statistics(),
                'degradation_manager': enhanced_degradation_manager.get_system_status(),
                'integration_metrics': self.metrics.copy(),
                'monitoring_active': self.integration_active
            }
            
    def reset_metrics(self):
        """Reset performance metrics."""
        with self.lock:
            self.metrics = {
                'total_errors_handled': 0,
                'successful_recoveries': 0,
                'failed_recoveries': 0,
                'notifications_sent': 0,
                'notifications_filtered': 0,
                'services_degraded': 0,
                'services_recovered': 0
            }
            logger.info("Integration metrics reset")


# Global instance
integrated_recovery_system = IntegratedRecoverySystem()


def start_integrated_recovery():
    """Start the integrated recovery system."""
    integrated_recovery_system.start_health_monitoring()
    logger.info("Integrated recovery system started")


def stop_integrated_recovery():
    """Stop the integrated recovery system."""
    integrated_recovery_system.stop_health_monitoring()
    logger.info("Integrated recovery system stopped")


def get_system_health() -> SystemHealthReport:
    """Get current system health status."""
    return integrated_recovery_system.get_system_health_status()


def get_comprehensive_status() -> Dict[str, Any]:
    """Get comprehensive status of all systems."""
    return integrated_recovery_system.get_comprehensive_status()


# Auto-start integration
start_integrated_recovery()