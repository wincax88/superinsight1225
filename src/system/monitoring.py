"""
System Monitoring and Metrics Collection for SuperInsight Platform.

Provides comprehensive monitoring capabilities including:
- Performance metrics collection
- Resource usage monitoring
- Service health tracking
- Custom metrics and alerts
"""

import asyncio
import logging
import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta

from src.config.settings import settings


logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: float
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """A time series of metric points."""
    name: str
    points: deque = field(default_factory=lambda: deque(maxlen=1000))
    unit: str = ""
    description: str = ""


class MetricsCollector:
    """
    Metrics collection and aggregation system.
    
    Collects various system and application metrics for monitoring
    and performance analysis.
    """
    
    def __init__(self, max_points_per_metric: int = 1000):
        self.metrics: Dict[str, MetricSeries] = {}
        self.max_points_per_metric = max_points_per_metric
        self.collection_interval = 10  # seconds
        self.is_collecting = False
        self._collection_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        
    def register_metric(self, name: str, unit: str = "", description: str = ""):
        """Register a new metric for collection."""
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = MetricSeries(
                    name=name,
                    points=deque(maxlen=self.max_points_per_metric),
                    unit=unit,
                    description=description
                )
                logger.debug(f"Registered metric: {name}")
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        with self._lock:
            if name not in self.metrics:
                # Register metric directly without calling register_metric to avoid deadlock
                self.metrics[name] = MetricSeries(
                    name=name,
                    points=deque(maxlen=self.max_points_per_metric),
                    unit="",
                    description=""
                )
                logger.debug(f"Auto-registered metric: {name}")
            
            point = MetricPoint(
                timestamp=time.time(),
                value=value,
                tags=tags or {}
            )
            
            self.metrics[name].points.append(point)
    
    def increment_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        # For counters, we'll track the cumulative value
        with self._lock:
            if name not in self.metrics:
                # Register metric directly without calling register_metric to avoid deadlock
                self.metrics[name] = MetricSeries(
                    name=name,
                    points=deque(maxlen=self.max_points_per_metric),
                    unit="count",
                    description=""
                )
                logger.debug(f"Auto-registered counter metric: {name}")
            
            # Get the last value and add to it
            last_value = 0.0
            if self.metrics[name].points:
                last_value = self.metrics[name].points[-1].value
            
            # Record directly without calling record_metric to avoid potential recursion
            point = MetricPoint(
                timestamp=time.time(),
                value=last_value + value,
                tags=tags or {}
            )
            
            self.metrics[name].points.append(point)
    
    def record_timing(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a timing metric."""
        self.record_metric(f"{name}.duration", duration, tags)
    
    def get_metric_values(self, name: str, since: Optional[float] = None) -> List[MetricPoint]:
        """Get metric values, optionally filtered by time."""
        with self._lock:
            if name not in self.metrics:
                return []
            
            points = list(self.metrics[name].points)
            
            if since is not None:
                points = [p for p in points if p.timestamp >= since]
            
            return points
    
    def get_metric_summary(self, name: str, since: Optional[float] = None) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        points = self.get_metric_values(name, since)
        
        if not points:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "avg": None,
                "latest": None
            }
        
        values = [p.value for p in points]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1] if values else None,
            "latest_timestamp": points[-1].timestamp if points else None
        }
    
    def get_all_metrics_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary for all metrics."""
        summary = {}
        
        with self._lock:
            for name in self.metrics:
                summary[name] = self.get_metric_summary(name)
        
        return summary
    
    async def start_collection(self):
        """Start automatic metrics collection."""
        if self.is_collecting:
            logger.warning("Metrics collection is already running")
            return
        
        self.is_collecting = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Started metrics collection")
    
    async def stop_collection(self):
        """Stop automatic metrics collection."""
        self.is_collecting = False
        
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped metrics collection")
    
    async def _collection_loop(self):
        """Main metrics collection loop."""
        while self.is_collecting:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(5)  # Short delay before retrying
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric("system.cpu.usage_percent", cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.record_metric("system.memory.usage_percent", memory.percent)
            self.record_metric("system.memory.available_bytes", memory.available)
            self.record_metric("system.memory.used_bytes", memory.used)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.record_metric("system.disk.usage_percent", (disk.used / disk.total) * 100)
            self.record_metric("system.disk.free_bytes", disk.free)
            
            # Network I/O
            network = psutil.net_io_counters()
            self.record_metric("system.network.bytes_sent", network.bytes_sent)
            self.record_metric("system.network.bytes_recv", network.bytes_recv)
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")


class PerformanceMonitor:
    """
    Performance monitoring and profiling system.
    
    Tracks application performance metrics and provides
    insights into bottlenecks and optimization opportunities.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.active_requests: Dict[str, float] = {}
        self.request_counts = defaultdict(int)
        
    def start_request(self, request_id: str, endpoint: str):
        """Start tracking a request."""
        self.active_requests[request_id] = time.time()
        self.request_counts[endpoint] += 1
        self.metrics.increment_counter("requests.total", tags={"endpoint": endpoint})
    
    def end_request(self, request_id: str, endpoint: str, status_code: int):
        """End tracking a request."""
        if request_id in self.active_requests:
            duration = time.time() - self.active_requests[request_id]
            del self.active_requests[request_id]
            
            self.metrics.record_timing(
                "requests.duration",
                duration,
                tags={"endpoint": endpoint, "status": str(status_code)}
            )
            
            # Track status codes
            self.metrics.increment_counter(
                "requests.status",
                tags={"endpoint": endpoint, "status": str(status_code)}
            )
    
    def record_database_query(self, query_type: str, duration: float, success: bool):
        """Record database query metrics."""
        self.metrics.record_metric(
            "database.query.duration",
            duration,
            tags={"type": query_type, "success": str(success)}
        )
        
        self.metrics.increment_counter(
            "database.queries.total",
            tags={"type": query_type, "success": str(success)}
        )
    
    def record_ai_inference(self, model_name: str, duration: float, success: bool):
        """Record AI inference metrics."""
        self.metrics.record_timing(
            "ai.inference.duration",
            duration,
            tags={"model": model_name, "success": str(success)}
        )
        
        self.metrics.increment_counter(
            "ai.inferences.total",
            tags={"model": model_name, "success": str(success)}
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            "active_requests": len(self.active_requests),
            "request_counts": dict(self.request_counts),
            "avg_request_duration": self.metrics.get_metric_summary("requests.duration"),
            "database_performance": self.metrics.get_metric_summary("database.query.duration"),
            "ai_performance": self.metrics.get_metric_summary("ai.inference.duration")
        }


class HealthMonitor:
    """
    System health monitoring and alerting.
    
    Monitors system health indicators and triggers alerts
    when thresholds are exceeded.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.health_checks: Dict[str, Callable] = {}
        self.thresholds: Dict[str, Dict[str, float]] = {}
        self.alert_handlers: List[Callable] = []
        
    def register_health_check(self, name: str, check_func: Callable):
        """Register a health check function."""
        self.health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    def set_threshold(self, metric_name: str, warning: float, critical: float):
        """Set alert thresholds for a metric."""
        self.thresholds[metric_name] = {
            "warning": warning,
            "critical": critical
        }
    
    def register_alert_handler(self, handler: Callable):
        """Register an alert handler."""
        self.alert_handlers.append(handler)
    
    async def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        health_status = {
            "overall_status": "healthy",
            "checks": {},
            "alerts": []
        }
        
        # Run registered health checks
        for name, check_func in self.health_checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                health_status["checks"][name] = {
                    "status": "healthy" if result else "unhealthy",
                    "result": result
                }
                
                if not result:
                    health_status["overall_status"] = "unhealthy"
                    
            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                health_status["checks"][name] = {
                    "status": "error",
                    "error": str(e)
                }
                health_status["overall_status"] = "unhealthy"
        
        # Check metric thresholds
        for metric_name, thresholds in self.thresholds.items():
            summary = self.metrics.get_metric_summary(metric_name)
            
            if summary["latest"] is not None:
                latest_value = summary["latest"]
                
                if latest_value >= thresholds["critical"]:
                    alert = {
                        "level": "critical",
                        "metric": metric_name,
                        "value": latest_value,
                        "threshold": thresholds["critical"]
                    }
                    health_status["alerts"].append(alert)
                    health_status["overall_status"] = "critical"
                    
                    # Send alert
                    await self._send_alert(alert)
                    
                elif latest_value >= thresholds["warning"]:
                    alert = {
                        "level": "warning",
                        "metric": metric_name,
                        "value": latest_value,
                        "threshold": thresholds["warning"]
                    }
                    health_status["alerts"].append(alert)
                    
                    if health_status["overall_status"] == "healthy":
                        health_status["overall_status"] = "warning"
        
        return health_status
    
    async def _send_alert(self, alert: Dict[str, Any]):
        """Send alert to registered handlers."""
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")


# Global monitoring instances
metrics_collector = MetricsCollector()
performance_monitor = PerformanceMonitor(metrics_collector)
health_monitor = HealthMonitor(metrics_collector)


# Context manager for request tracking
class RequestTracker:
    """Context manager for tracking request performance."""
    
    def __init__(self, endpoint: str, request_id: Optional[str] = None):
        self.endpoint = endpoint
        self.request_id = request_id or f"req_{int(time.time() * 1000)}"
        self.status_code = 200
    
    def __enter__(self):
        performance_monitor.start_request(self.request_id, self.endpoint)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.status_code = 500
        
        performance_monitor.end_request(self.request_id, self.endpoint, self.status_code)
    
    def set_status_code(self, status_code: int):
        """Set the response status code."""
        self.status_code = status_code


# Timing decorator
def monitor_performance(endpoint: str):
    """Decorator for monitoring function performance."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                with RequestTracker(endpoint):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def wrapper(*args, **kwargs):
                with RequestTracker(endpoint):
                    return func(*args, **kwargs)
            return wrapper
    return decorator