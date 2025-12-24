"""
Enhanced Health Monitoring System for SuperInsight Platform.

Provides comprehensive health monitoring, proactive issue detection,
and automatic recovery coordination with the error handling system.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import requests
from collections import defaultdict, deque

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

from src.system.error_handler import error_handler, ErrorCategory, ErrorSeverity
from src.system.notification import notification_system, NotificationPriority, NotificationChannel
from src.utils.degradation import degradation_manager, DegradationLevel

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class MetricType(Enum):
    """Types of health metrics."""
    SYSTEM = "system"
    APPLICATION = "application"
    EXTERNAL = "external"
    BUSINESS = "business"


@dataclass
class HealthMetric:
    """Health metric definition."""
    name: str
    value: float
    threshold_warning: float
    threshold_critical: float
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    unit: str = ""
    description: str = ""


@dataclass
class HealthCheck:
    """Health check configuration."""
    name: str
    check_function: Callable[[], HealthMetric]
    interval: float = 60.0  # seconds
    timeout: float = 30.0
    enabled: bool = True
    dependencies: List[str] = field(default_factory=list)
    retry_attempts: int = 3
    retry_delay: float = 5.0


@dataclass
class HealthReport:
    """Comprehensive health report."""
    timestamp: float
    overall_status: HealthStatus
    metrics: Dict[str, HealthMetric]
    issues: List[str]
    recommendations: List[str]
    service_health: Dict[str, Dict[str, Any]]


class SystemHealthMonitor:
    """
    Comprehensive system health monitoring with proactive issue detection.
    
    Features:
    - System resource monitoring (CPU, memory, disk)
    - Application health checks
    - External dependency monitoring
    - Business metric tracking
    - Proactive issue detection
    - Automatic recovery coordination
    """
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.last_check_results: Dict[str, HealthMetric] = {}
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.check_lock = threading.Lock()
        
        # Issue detection
        self.detected_issues: Set[str] = set()
        self.issue_cooldown: Dict[str, float] = {}
        self.cooldown_period = 300  # 5 minutes
        
        # Setup default health checks
        self._setup_default_health_checks()
    
    def _setup_default_health_checks(self):
        """Setup default system health checks."""
        # System resource checks
        self.register_health_check(HealthCheck(
            name="cpu_usage",
            check_function=self._check_cpu_usage,
            interval=30.0
        ))
        
        self.register_health_check(HealthCheck(
            name="memory_usage",
            check_function=self._check_memory_usage,
            interval=30.0
        ))
        
        self.register_health_check(HealthCheck(
            name="disk_usage",
            check_function=self._check_disk_usage,
            interval=60.0
        ))
        
        # Application health checks
        self.register_health_check(HealthCheck(
            name="database_connection",
            check_function=self._check_database_connection,
            interval=60.0
        ))
        
        self.register_health_check(HealthCheck(
            name="error_rate",
            check_function=self._check_error_rate,
            interval=60.0
        ))
        
        # External dependency checks
        self.register_health_check(HealthCheck(
            name="label_studio_health",
            check_function=self._check_label_studio_health,
            interval=120.0
        ))
    
    def register_health_check(self, health_check: HealthCheck):
        """Register a new health check."""
        self.health_checks[health_check.name] = health_check
        logger.info(f"Registered health check: {health_check.name}")
    
    def start_monitoring(self):
        """Start the health monitoring system."""
        if self.monitoring_active:
            logger.warning("Health monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop the health monitoring system."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        next_check_times = {name: time.time() for name in self.health_checks.keys()}
        
        while self.monitoring_active:
            try:
                current_time = time.time()
                
                # Check which health checks need to run
                for check_name, health_check in self.health_checks.items():
                    if not health_check.enabled:
                        continue
                    
                    if current_time >= next_check_times[check_name]:
                        self._run_health_check(health_check)
                        next_check_times[check_name] = current_time + health_check.interval
                
                # Analyze trends and detect issues
                self._analyze_health_trends()
                
                # Sleep for a short interval
                time.sleep(5.0)
                
            except Exception as e:
                logger.error(f"Health monitoring loop error: {e}")
                time.sleep(10.0)
    
    def _run_health_check(self, health_check: HealthCheck):
        """Run a single health check with retry logic."""
        for attempt in range(health_check.retry_attempts):
            try:
                with self.check_lock:
                    # Check dependencies first
                    if not self._check_dependencies(health_check.dependencies):
                        logger.debug(f"Skipping {health_check.name} due to failed dependencies")
                        return
                    
                    # Run the health check
                    start_time = time.time()
                    metric = health_check.check_function()
                    duration = time.time() - start_time
                    
                    if duration > health_check.timeout:
                        logger.warning(f"Health check {health_check.name} took {duration:.2f}s (timeout: {health_check.timeout}s)")
                    
                    # Store results
                    self.last_check_results[health_check.name] = metric
                    self.metric_history[health_check.name].append(metric)
                    
                    # Evaluate metric and take action if needed
                    self._evaluate_metric(metric, health_check.name)
                    
                    logger.debug(f"Health check {health_check.name} completed: {metric.value} {metric.unit}")
                    return
                    
            except Exception as e:
                logger.error(f"Health check {health_check.name} failed (attempt {attempt + 1}): {e}")
                if attempt < health_check.retry_attempts - 1:
                    time.sleep(health_check.retry_delay)
                else:
                    # All attempts failed, create error metric
                    error_metric = HealthMetric(
                        name=health_check.name,
                        value=-1,
                        threshold_warning=0,
                        threshold_critical=0,
                        metric_type=MetricType.SYSTEM,
                        description=f"Health check failed: {str(e)}"
                    )
                    self.last_check_results[health_check.name] = error_metric
                    
                    # Report to error handler
                    error_handler.handle_error(
                        exception=Exception(f"Health check {health_check.name} failed: {e}"),
                        category=ErrorCategory.SYSTEM,
                        severity=ErrorSeverity.HIGH,
                        service_name="health_monitor"
                    )
    
    def _check_dependencies(self, dependencies: List[str]) -> bool:
        """Check if all dependencies are healthy."""
        for dep in dependencies:
            if dep in self.last_check_results:
                metric = self.last_check_results[dep]
                if metric.value >= metric.threshold_critical:
                    return False
        return True
    
    def _evaluate_metric(self, metric: HealthMetric, check_name: str):
        """Evaluate a metric and take appropriate action."""
        status = self._get_metric_status(metric)
        
        if status == HealthStatus.CRITICAL:
            issue_key = f"{check_name}_critical"
            if self._should_report_issue(issue_key):
                logger.critical(f"Critical health issue detected: {check_name} = {metric.value} {metric.unit}")
                
                # Send critical notification
                notification_system.send_notification(
                    title=f"Critical Health Alert - {check_name}",
                    message=f"{metric.description or check_name} is in critical state: {metric.value} {metric.unit} (threshold: {metric.threshold_critical})",
                    priority=NotificationPriority.CRITICAL,
                    channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.WEBHOOK]
                )
                
                # Trigger automatic recovery if possible
                self._trigger_automatic_recovery(check_name, metric)
                
        elif status == HealthStatus.WARNING:
            issue_key = f"{check_name}_warning"
            if self._should_report_issue(issue_key):
                logger.warning(f"Health warning detected: {check_name} = {metric.value} {metric.unit}")
                
                # Send warning notification
                notification_system.send_notification(
                    title=f"Health Warning - {check_name}",
                    message=f"{metric.description or check_name} is in warning state: {metric.value} {metric.unit} (threshold: {metric.threshold_warning})",
                    priority=NotificationPriority.HIGH,
                    channels=[NotificationChannel.SLACK, NotificationChannel.LOG]
                )
    
    def _get_metric_status(self, metric: HealthMetric) -> HealthStatus:
        """Determine the status of a metric based on thresholds."""
        if metric.value < 0:  # Error condition
            return HealthStatus.UNKNOWN
        elif metric.value >= metric.threshold_critical:
            return HealthStatus.CRITICAL
        elif metric.value >= metric.threshold_warning:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def _should_report_issue(self, issue_key: str) -> bool:
        """Check if an issue should be reported based on cooldown."""
        current_time = time.time()
        
        if issue_key in self.issue_cooldown:
            if current_time - self.issue_cooldown[issue_key] < self.cooldown_period:
                return False
        
        self.issue_cooldown[issue_key] = current_time
        self.detected_issues.add(issue_key)
        return True
    
    def _trigger_automatic_recovery(self, check_name: str, metric: HealthMetric):
        """Trigger automatic recovery actions based on the health check."""
        logger.info(f"Triggering automatic recovery for {check_name}")
        
        # Map health checks to recovery actions
        recovery_actions = {
            "cpu_usage": self._recover_high_cpu,
            "memory_usage": self._recover_high_memory,
            "disk_usage": self._recover_disk_space,
            "database_connection": self._recover_database_connection,
            "error_rate": self._recover_high_error_rate
        }
        
        recovery_func = recovery_actions.get(check_name)
        if recovery_func:
            try:
                recovery_func(metric)
            except Exception as e:
                logger.error(f"Automatic recovery failed for {check_name}: {e}")
    
    def _analyze_health_trends(self):
        """Analyze health trends to detect patterns and predict issues."""
        for check_name, history in self.metric_history.items():
            if len(history) < 5:  # Need at least 5 data points
                continue
            
            try:
                # Calculate trend
                recent_values = [m.value for m in list(history)[-5:]]
                if len(set(recent_values)) == 1:  # All values are the same
                    continue
                
                # Simple trend analysis
                trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
                
                # Check if trend is concerning
                latest_metric = history[-1]
                if trend > 0 and latest_metric.value > latest_metric.threshold_warning * 0.8:
                    # Upward trend approaching warning threshold
                    issue_key = f"{check_name}_trend_warning"
                    if self._should_report_issue(issue_key):
                        notification_system.send_notification(
                            title=f"Health Trend Alert - {check_name}",
                            message=f"{check_name} is trending upward and may reach warning threshold soon. Current: {latest_metric.value} {latest_metric.unit}, Trend: +{trend:.2f}",
                            priority=NotificationPriority.NORMAL,
                            channels=[NotificationChannel.LOG, NotificationChannel.SLACK]
                        )
                        
            except Exception as e:
                logger.debug(f"Trend analysis failed for {check_name}: {e}")
    
    # Default health check implementations
    def _check_cpu_usage(self) -> HealthMetric:
        """Check CPU usage percentage."""
        if not PSUTIL_AVAILABLE:
            return HealthMetric(
                name="cpu_usage",
                value=0.0,  # Assume healthy if can't check
                threshold_warning=80.0,
                threshold_critical=95.0,
                metric_type=MetricType.SYSTEM,
                unit="%",
                description="CPU usage check unavailable (psutil not installed)"
            )
        
        cpu_percent = psutil.cpu_percent(interval=1)
        return HealthMetric(
            name="cpu_usage",
            value=cpu_percent,
            threshold_warning=80.0,
            threshold_critical=95.0,
            metric_type=MetricType.SYSTEM,
            unit="%",
            description="CPU usage percentage"
        )
    
    def _check_memory_usage(self) -> HealthMetric:
        """Check memory usage percentage."""
        if not PSUTIL_AVAILABLE:
            return HealthMetric(
                name="memory_usage",
                value=0.0,  # Assume healthy if can't check
                threshold_warning=85.0,
                threshold_critical=95.0,
                metric_type=MetricType.SYSTEM,
                unit="%",
                description="Memory usage check unavailable (psutil not installed)"
            )
        
        memory = psutil.virtual_memory()
        return HealthMetric(
            name="memory_usage",
            value=memory.percent,
            threshold_warning=85.0,
            threshold_critical=95.0,
            metric_type=MetricType.SYSTEM,
            unit="%",
            description="Memory usage percentage"
        )
    
    def _check_disk_usage(self) -> HealthMetric:
        """Check disk usage percentage."""
        if not PSUTIL_AVAILABLE:
            return HealthMetric(
                name="disk_usage",
                value=0.0,  # Assume healthy if can't check
                threshold_warning=85.0,
                threshold_critical=95.0,
                metric_type=MetricType.SYSTEM,
                unit="%",
                description="Disk usage check unavailable (psutil not installed)"
            )
        
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        return HealthMetric(
            name="disk_usage",
            value=disk_percent,
            threshold_warning=85.0,
            threshold_critical=95.0,
            metric_type=MetricType.SYSTEM,
            unit="%",
            description="Disk usage percentage"
        )
    
    def _check_database_connection(self) -> HealthMetric:
        """Check database connection health."""
        try:
            # This would integrate with actual database health check
            # For now, simulate based on degradation manager
            db_health = degradation_manager.get_service_health("database")
            if db_health and db_health.is_healthy:
                return HealthMetric(
                    name="database_connection",
                    value=0.0,  # 0 = healthy
                    threshold_warning=1.0,
                    threshold_critical=2.0,
                    metric_type=MetricType.APPLICATION,
                    description="Database connection status"
                )
            else:
                return HealthMetric(
                    name="database_connection",
                    value=2.0,  # 2 = critical
                    threshold_warning=1.0,
                    threshold_critical=2.0,
                    metric_type=MetricType.APPLICATION,
                    description="Database connection failed"
                )
        except Exception as e:
            return HealthMetric(
                name="database_connection",
                value=2.0,
                threshold_warning=1.0,
                threshold_critical=2.0,
                metric_type=MetricType.APPLICATION,
                description=f"Database check error: {str(e)}"
            )
    
    def _check_error_rate(self) -> HealthMetric:
        """Check application error rate."""
        try:
            # Get error statistics from error handler
            stats = error_handler.get_error_statistics()
            
            # Calculate error rate (errors per minute in last 10 minutes)
            recent_errors = [e for e in error_handler.error_history 
                           if time.time() - e.timestamp < 600]  # Last 10 minutes
            error_rate = len(recent_errors) / 10.0  # errors per minute
            
            return HealthMetric(
                name="error_rate",
                value=error_rate,
                threshold_warning=5.0,  # 5 errors per minute
                threshold_critical=20.0,  # 20 errors per minute
                metric_type=MetricType.APPLICATION,
                unit="errors/min",
                description="Application error rate"
            )
        except Exception as e:
            return HealthMetric(
                name="error_rate",
                value=-1,
                threshold_warning=5.0,
                threshold_critical=20.0,
                metric_type=MetricType.APPLICATION,
                description=f"Error rate check failed: {str(e)}"
            )
    
    def _check_label_studio_health(self) -> HealthMetric:
        """Check Label Studio health."""
        try:
            # This would make an actual HTTP request to Label Studio health endpoint
            # For now, simulate based on service health
            ls_health = degradation_manager.get_service_health("label_studio")
            if ls_health and ls_health.is_healthy:
                return HealthMetric(
                    name="label_studio_health",
                    value=0.0,
                    threshold_warning=1.0,
                    threshold_critical=2.0,
                    metric_type=MetricType.EXTERNAL,
                    description="Label Studio is healthy"
                )
            else:
                return HealthMetric(
                    name="label_studio_health",
                    value=2.0,
                    threshold_warning=1.0,
                    threshold_critical=2.0,
                    metric_type=MetricType.EXTERNAL,
                    description="Label Studio is unhealthy"
                )
        except Exception as e:
            return HealthMetric(
                name="label_studio_health",
                value=2.0,
                threshold_warning=1.0,
                threshold_critical=2.0,
                metric_type=MetricType.EXTERNAL,
                description=f"Label Studio check failed: {str(e)}"
            )
    
    # Recovery action implementations
    def _recover_high_cpu(self, metric: HealthMetric):
        """Recover from high CPU usage."""
        logger.info("Attempting to recover from high CPU usage")
        
        # Send notification about recovery attempt
        notification_system.send_notification(
            title="CPU Recovery Action",
            message=f"Attempting to recover from high CPU usage: {metric.value}%",
            priority=NotificationPriority.NORMAL,
            channels=[NotificationChannel.LOG]
        )
        
        # Could implement actual recovery actions like:
        # - Reducing worker processes
        # - Throttling requests
        # - Clearing caches
    
    def _recover_high_memory(self, metric: HealthMetric):
        """Recover from high memory usage."""
        logger.info("Attempting to recover from high memory usage")
        
        # Clear caches
        degradation_manager.clear_cache()
        
        notification_system.send_notification(
            title="Memory Recovery Action",
            message=f"Cleared caches due to high memory usage: {metric.value}%",
            priority=NotificationPriority.NORMAL,
            channels=[NotificationChannel.LOG]
        )
    
    def _recover_disk_space(self, metric: HealthMetric):
        """Recover from low disk space."""
        logger.info("Attempting to recover from low disk space")
        
        notification_system.send_notification(
            title="Disk Space Recovery Action",
            message=f"High disk usage detected: {metric.value}%. Manual intervention may be required.",
            priority=NotificationPriority.HIGH,
            channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK]
        )
    
    def _recover_database_connection(self, metric: HealthMetric):
        """Recover from database connection issues."""
        logger.info("Attempting to recover database connection")
        
        # Mark database service for recovery
        degradation_manager.mark_service_failure("database")
        
        notification_system.send_notification(
            title="Database Recovery Action",
            message="Database connection issue detected. Attempting recovery.",
            priority=NotificationPriority.HIGH,
            channels=[NotificationChannel.SLACK, NotificationChannel.LOG]
        )
    
    def _recover_high_error_rate(self, metric: HealthMetric):
        """Recover from high error rate."""
        logger.info("Attempting to recover from high error rate")
        
        # Enable degradation for all services
        for service_name in ["ai_annotation", "data_extraction", "quality_check"]:
            degradation_manager.mark_service_failure(service_name)
        
        notification_system.send_notification(
            title="Error Rate Recovery Action",
            message=f"High error rate detected: {metric.value} errors/min. Enabling graceful degradation.",
            priority=NotificationPriority.HIGH,
            channels=[NotificationChannel.SLACK, NotificationChannel.LOG]
        )
    
    def get_health_report(self) -> HealthReport:
        """Generate a comprehensive health report."""
        current_time = time.time()
        
        # Determine overall status
        overall_status = HealthStatus.HEALTHY
        issues = []
        recommendations = []
        
        for check_name, metric in self.last_check_results.items():
            status = self._get_metric_status(metric)
            
            if status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
                issues.append(f"{check_name}: {metric.description}")
                recommendations.append(f"Address critical issue in {check_name}")
            elif status == HealthStatus.WARNING and overall_status != HealthStatus.CRITICAL:
                if overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.WARNING
                issues.append(f"{check_name}: {metric.description}")
                recommendations.append(f"Monitor {check_name} closely")
        
        # Get service health from degradation manager
        service_health = {}
        for service_name, health in degradation_manager.get_all_service_health().items():
            service_health[service_name] = {
                "is_healthy": health.is_healthy,
                "degradation_level": health.degradation_level.value,
                "failure_count": health.failure_count,
                "last_failure_time": health.last_failure_time
            }
        
        return HealthReport(
            timestamp=current_time,
            overall_status=overall_status,
            metrics=dict(self.last_check_results),
            issues=issues,
            recommendations=recommendations,
            service_health=service_health
        )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all health metrics."""
        summary = {
            "timestamp": time.time(),
            "monitoring_active": self.monitoring_active,
            "total_checks": len(self.health_checks),
            "enabled_checks": sum(1 for hc in self.health_checks.values() if hc.enabled),
            "metrics": {},
            "trends": {}
        }
        
        for check_name, metric in self.last_check_results.items():
            summary["metrics"][check_name] = {
                "value": metric.value,
                "unit": metric.unit,
                "status": self._get_metric_status(metric).value,
                "timestamp": metric.timestamp
            }
            
            # Add trend information if available
            if check_name in self.metric_history and len(self.metric_history[check_name]) >= 2:
                history = list(self.metric_history[check_name])
                recent_values = [m.value for m in history[-5:]]
                if len(recent_values) >= 2:
                    trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
                    summary["trends"][check_name] = {
                        "direction": "up" if trend > 0 else "down" if trend < 0 else "stable",
                        "rate": trend
                    }
        
        return summary


# Global health monitor instance
health_monitor = SystemHealthMonitor()


# Convenience functions
def start_health_monitoring():
    """Start the global health monitoring system."""
    health_monitor.start_monitoring()


def stop_health_monitoring():
    """Stop the global health monitoring system."""
    health_monitor.stop_monitoring()


def get_health_status() -> Dict[str, Any]:
    """Get current health status."""
    report = health_monitor.get_health_report()
    return {
        "status": report.overall_status.value,
        "timestamp": report.timestamp,
        "issues": report.issues,
        "recommendations": report.recommendations
    }


def register_custom_health_check(name: str, check_function: Callable[[], HealthMetric], **kwargs):
    """Register a custom health check."""
    health_check = HealthCheck(
        name=name,
        check_function=check_function,
        **kwargs
    )
    health_monitor.register_health_check(health_check)