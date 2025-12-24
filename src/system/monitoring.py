"""
System Monitoring and Metrics Collection for SuperInsight Platform.

Provides comprehensive monitoring capabilities including:
- Performance metrics collection with detailed analysis
- Resource usage monitoring with trend analysis
- Service health tracking with predictive alerts
- Custom metrics and automated bottleneck detection
- Business metrics integration
- Performance optimization recommendations
"""

import asyncio
import logging
import time
import psutil
import threading
import statistics
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json

from src.config.settings import settings


logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """A single metric data point with enhanced metadata."""
    timestamp: float
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """A time series of metric points with statistical analysis."""
    name: str
    points: deque = field(default_factory=lambda: deque(maxlen=1000))
    unit: str = ""
    description: str = ""
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    trend_window: int = 100  # Number of points for trend analysis


@dataclass
class PerformanceAlert:
    """Performance alert definition."""
    metric_name: str
    alert_type: str  # threshold, trend, anomaly
    severity: str    # low, medium, high, critical
    message: str
    timestamp: float
    value: float
    threshold: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BottleneckAnalysis:
    """Bottleneck analysis result."""
    component: str
    severity: str
    description: str
    metrics: Dict[str, float]
    recommendations: List[str]
    timestamp: float


class MetricsCollector:
    """
    Enhanced metrics collection and aggregation system.
    
    Collects various system and application metrics for monitoring,
    performance analysis, bottleneck detection, and trend analysis.
    """
    
    def __init__(self, max_points_per_metric: int = 1000):
        self.metrics: Dict[str, MetricSeries] = {}
        self.max_points_per_metric = max_points_per_metric
        self.collection_interval = 10  # seconds
        self.is_collecting = False
        self._collection_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        self.alerts: List[PerformanceAlert] = []
        self.bottleneck_analyses: List[BottleneckAnalysis] = []
        self.alert_handlers: List[Callable] = []
        
        # Performance baselines for anomaly detection
        self.baselines: Dict[str, Dict[str, float]] = {}
        
        # Trend analysis configuration
        self.trend_analysis_enabled = True
        self.anomaly_detection_enabled = True
        
    def register_metric(
        self,
        name: str,
        unit: str = "",
        description: str = "",
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        """Register a new metric for collection with alert thresholds."""
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = MetricSeries(
                    name=name,
                    points=deque(maxlen=self.max_points_per_metric),
                    unit=unit,
                    description=description,
                    alert_thresholds=alert_thresholds or {}
                )
                logger.debug(f"Registered metric: {name}")
    
    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a metric value with enhanced metadata."""
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
                tags=tags or {},
                metadata=metadata or {}
            )
            
            self.metrics[name].points.append(point)
            
            # Check for alerts
            self._check_metric_alerts(name, value)
    
    def _check_metric_alerts(self, metric_name: str, value: float):
        """Check if metric value triggers any alerts."""
        if metric_name not in self.metrics:
            return
        
        metric = self.metrics[metric_name]
        current_time = time.time()
        
        # Check threshold alerts
        for alert_type, threshold in metric.alert_thresholds.items():
            if alert_type == "high" and value > threshold:
                alert = PerformanceAlert(
                    metric_name=metric_name,
                    alert_type="threshold",
                    severity="high",
                    message=f"Metric {metric_name} exceeded high threshold: {value} > {threshold}",
                    timestamp=current_time,
                    value=value,
                    threshold=threshold
                )
                self._trigger_alert(alert)
            elif alert_type == "critical" and value > threshold:
                alert = PerformanceAlert(
                    metric_name=metric_name,
                    alert_type="threshold",
                    severity="critical",
                    message=f"Metric {metric_name} exceeded critical threshold: {value} > {threshold}",
                    timestamp=current_time,
                    value=value,
                    threshold=threshold
                )
                self._trigger_alert(alert)
        
        # Check for anomalies if enabled
        if self.anomaly_detection_enabled:
            self._check_anomaly(metric_name, value)
    
    def _check_anomaly(self, metric_name: str, value: float):
        """Check for anomalous values using statistical analysis."""
        if metric_name not in self.metrics:
            return
        
        metric = self.metrics[metric_name]
        if len(metric.points) < 20:  # Need sufficient data for anomaly detection
            return
        
        # Get recent values for baseline
        recent_values = [p.value for p in list(metric.points)[-20:]]
        mean_value = statistics.mean(recent_values)
        std_dev = statistics.stdev(recent_values) if len(recent_values) > 1 else 0
        
        # Check if current value is anomalous (more than 2 standard deviations from mean)
        if std_dev > 0 and abs(value - mean_value) > 2 * std_dev:
            alert = PerformanceAlert(
                metric_name=metric_name,
                alert_type="anomaly",
                severity="medium",
                message=f"Anomalous value detected for {metric_name}: {value} (mean: {mean_value:.2f}, std: {std_dev:.2f})",
                timestamp=time.time(),
                value=value,
                metadata={"mean": mean_value, "std_dev": std_dev}
            )
            self._trigger_alert(alert)
    
    def _trigger_alert(self, alert: PerformanceAlert):
        """Trigger a performance alert."""
        self.alerts.append(alert)
        
        # Keep only recent alerts (last 1000)
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
        
        logger.warning(f"Performance alert: {alert.message}")
        
        # Notify alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    def register_alert_handler(self, handler: Callable):
        """Register an alert handler."""
        self.alert_handlers.append(handler)
        logger.info("Registered performance alert handler")
    
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
        """Get enhanced summary statistics for a metric."""
        points = self.get_metric_values(name, since)
        
        if not points:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "avg": None,
                "latest": None,
                "trend": "unknown",
                "percentiles": {},
                "anomaly_score": 0.0
            }
        
        values = [p.value for p in points]
        
        # Calculate percentiles
        percentiles = {}
        if len(values) >= 5:
            percentiles = {
                "p50": statistics.median(values),
                "p90": self._percentile(values, 90),
                "p95": self._percentile(values, 95),
                "p99": self._percentile(values, 99)
            }
        
        # Calculate trend
        trend = self._calculate_trend(values)
        
        # Calculate anomaly score
        anomaly_score = self._calculate_anomaly_score(values)
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1] if values else None,
            "latest_timestamp": points[-1].timestamp if points else None,
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "trend": trend,
            "percentiles": percentiles,
            "anomaly_score": anomaly_score
        }
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100.0) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for values."""
        if len(values) < 10:
            return "insufficient_data"
        
        # Use linear regression to determine trend
        n = len(values)
        x_values = list(range(n))
        
        # Calculate slope
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n
        
        numerator = sum((x_values[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        # Classify trend based on slope
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_anomaly_score(self, values: List[float]) -> float:
        """Calculate anomaly score for recent values."""
        if len(values) < 10:
            return 0.0
        
        # Use recent values to calculate baseline
        baseline_values = values[:-5] if len(values) > 10 else values[:-2]
        recent_values = values[-5:] if len(values) > 10 else values[-2:]
        
        if not baseline_values or not recent_values:
            return 0.0
        
        baseline_mean = statistics.mean(baseline_values)
        baseline_std = statistics.stdev(baseline_values) if len(baseline_values) > 1 else 0
        
        if baseline_std == 0:
            return 0.0
        
        # Calculate z-scores for recent values
        z_scores = [abs(value - baseline_mean) / baseline_std for value in recent_values]
        
        # Return maximum z-score as anomaly score
        return max(z_scores)
    
    def detect_bottlenecks(self) -> List[BottleneckAnalysis]:
        """Detect performance bottlenecks across all metrics."""
        bottlenecks = []
        current_time = time.time()
        
        # Analyze system metrics for bottlenecks
        cpu_summary = self.get_metric_summary("system.cpu.usage_percent")
        memory_summary = self.get_metric_summary("system.memory.usage_percent")
        disk_summary = self.get_metric_summary("system.disk.usage_percent")
        
        # CPU bottleneck detection
        if cpu_summary["latest"] and cpu_summary["latest"] > 80:
            bottleneck = BottleneckAnalysis(
                component="CPU",
                severity="high" if cpu_summary["latest"] > 90 else "medium",
                description=f"High CPU usage detected: {cpu_summary['latest']:.1f}%",
                metrics={"cpu_usage": cpu_summary["latest"], "trend": cpu_summary["trend"]},
                recommendations=[
                    "Consider scaling up CPU resources",
                    "Optimize CPU-intensive operations",
                    "Review process priorities"
                ],
                timestamp=current_time
            )
            bottlenecks.append(bottleneck)
        
        # Memory bottleneck detection
        if memory_summary["latest"] and memory_summary["latest"] > 85:
            bottleneck = BottleneckAnalysis(
                component="Memory",
                severity="high" if memory_summary["latest"] > 95 else "medium",
                description=f"High memory usage detected: {memory_summary['latest']:.1f}%",
                metrics={"memory_usage": memory_summary["latest"], "trend": memory_summary["trend"]},
                recommendations=[
                    "Consider increasing memory allocation",
                    "Review memory leaks in applications",
                    "Optimize data structures and caching"
                ],
                timestamp=current_time
            )
            bottlenecks.append(bottleneck)
        
        # Database performance bottleneck detection
        db_duration_summary = self.get_metric_summary("database.query.duration")
        if db_duration_summary["latest"] and db_duration_summary["latest"] > 5.0:
            bottleneck = BottleneckAnalysis(
                component="Database",
                severity="high" if db_duration_summary["latest"] > 10.0 else "medium",
                description=f"Slow database queries detected: {db_duration_summary['latest']:.2f}s average",
                metrics={"query_duration": db_duration_summary["latest"], "trend": db_duration_summary["trend"]},
                recommendations=[
                    "Optimize slow queries",
                    "Add database indexes",
                    "Consider database connection pooling",
                    "Review query patterns"
                ],
                timestamp=current_time
            )
            bottlenecks.append(bottleneck)
        
        # Request processing bottleneck detection
        request_duration_summary = self.get_metric_summary("requests.duration")
        if request_duration_summary["latest"] and request_duration_summary["latest"] > 2.0:
            bottleneck = BottleneckAnalysis(
                component="Request Processing",
                severity="medium",
                description=f"Slow request processing detected: {request_duration_summary['latest']:.2f}s average",
                metrics={"request_duration": request_duration_summary["latest"], "trend": request_duration_summary["trend"]},
                recommendations=[
                    "Optimize request handlers",
                    "Implement caching strategies",
                    "Review middleware performance",
                    "Consider load balancing"
                ],
                timestamp=current_time
            )
            bottlenecks.append(bottleneck)
        
        # Store bottleneck analyses
        self.bottleneck_analyses.extend(bottlenecks)
        
        # Keep only recent analyses (last 100)
        if len(self.bottleneck_analyses) > 100:
            self.bottleneck_analyses = self.bottleneck_analyses[-100:]
        
        return bottlenecks
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Get comprehensive performance insights and recommendations."""
        insights = {
            "timestamp": time.time(),
            "bottlenecks": self.detect_bottlenecks(),
            "alerts": self.alerts[-10:],  # Recent alerts
            "trends": {},
            "recommendations": []
        }
        
        # Analyze trends for key metrics
        key_metrics = [
            "system.cpu.usage_percent",
            "system.memory.usage_percent",
            "requests.duration",
            "database.query.duration"
        ]
        
        for metric_name in key_metrics:
            summary = self.get_metric_summary(metric_name)
            if summary["count"] > 0:
                insights["trends"][metric_name] = {
                    "trend": summary["trend"],
                    "latest": summary["latest"],
                    "avg": summary["avg"],
                    "anomaly_score": summary["anomaly_score"]
                }
        
        # Generate recommendations based on analysis
        recommendations = []
        
        # CPU recommendations
        cpu_trend = insights["trends"].get("system.cpu.usage_percent", {})
        if cpu_trend.get("trend") == "increasing" and cpu_trend.get("latest", 0) > 70:
            recommendations.append("CPU usage is trending upward. Consider optimizing CPU-intensive operations.")
        
        # Memory recommendations
        memory_trend = insights["trends"].get("system.memory.usage_percent", {})
        if memory_trend.get("trend") == "increasing" and memory_trend.get("latest", 0) > 80:
            recommendations.append("Memory usage is trending upward. Review for memory leaks and optimize data structures.")
        
        # Request performance recommendations
        request_trend = insights["trends"].get("requests.duration", {})
        if request_trend.get("anomaly_score", 0) > 2.0:
            recommendations.append("Request processing times show anomalous behavior. Investigate recent changes.")
        
        insights["recommendations"] = recommendations
        
        return insights
    
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
        """Collect enhanced system-level metrics."""
        try:
            # CPU usage with per-core breakdown
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
            
            self.record_metric("system.cpu.usage_percent", cpu_percent)
            for i, core_usage in enumerate(cpu_per_core):
                self.record_metric(f"system.cpu.core_{i}.usage_percent", core_usage)
            
            # CPU load averages (if available)
            try:
                load_avg = psutil.getloadavg()
                self.record_metric("system.cpu.load_1min", load_avg[0])
                self.record_metric("system.cpu.load_5min", load_avg[1])
                self.record_metric("system.cpu.load_15min", load_avg[2])
            except AttributeError:
                pass  # Not available on all platforms
            
            # Memory usage with detailed breakdown
            memory = psutil.virtual_memory()
            self.record_metric("system.memory.usage_percent", memory.percent)
            self.record_metric("system.memory.available_bytes", memory.available)
            self.record_metric("system.memory.used_bytes", memory.used)
            self.record_metric("system.memory.cached_bytes", getattr(memory, 'cached', 0))
            self.record_metric("system.memory.buffers_bytes", getattr(memory, 'buffers', 0))
            
            # Swap usage
            swap = psutil.swap_memory()
            self.record_metric("system.swap.usage_percent", swap.percent)
            self.record_metric("system.swap.used_bytes", swap.used)
            
            # Disk usage and I/O
            disk = psutil.disk_usage('/')
            self.record_metric("system.disk.usage_percent", (disk.used / disk.total) * 100)
            self.record_metric("system.disk.free_bytes", disk.free)
            self.record_metric("system.disk.total_bytes", disk.total)
            
            # Disk I/O statistics
            disk_io = psutil.disk_io_counters()
            if disk_io:
                self.record_metric("system.disk.read_bytes", disk_io.read_bytes)
                self.record_metric("system.disk.write_bytes", disk_io.write_bytes)
                self.record_metric("system.disk.read_ops", disk_io.read_count)
                self.record_metric("system.disk.write_ops", disk_io.write_count)
            
            # Network I/O with detailed statistics
            network = psutil.net_io_counters()
            if network:
                self.record_metric("system.network.bytes_sent", network.bytes_sent)
                self.record_metric("system.network.bytes_recv", network.bytes_recv)
                self.record_metric("system.network.packets_sent", network.packets_sent)
                self.record_metric("system.network.packets_recv", network.packets_recv)
                self.record_metric("system.network.errors_in", network.errin)
                self.record_metric("system.network.errors_out", network.errout)
                self.record_metric("system.network.drops_in", network.dropin)
                self.record_metric("system.network.drops_out", network.dropout)
            
            # Process-specific metrics
            current_process = psutil.Process()
            process_memory = current_process.memory_info()
            
            self.record_metric("process.memory.rss_bytes", process_memory.rss)
            self.record_metric("process.memory.vms_bytes", process_memory.vms)
            self.record_metric("process.cpu.usage_percent", current_process.cpu_percent())
            self.record_metric("process.threads.count", current_process.num_threads())
            self.record_metric("process.fds.count", current_process.num_fds() if hasattr(current_process, 'num_fds') else 0)
            
            # System-wide process statistics
            self.record_metric("system.processes.total", len(psutil.pids()))
            self.record_metric("system.processes.running", len([p for p in psutil.process_iter(['status']) if p.info['status'] == 'running']))
            
        except Exception as e:
            logger.error(f"Failed to collect enhanced system metrics: {e}")


class PerformanceMonitor:
    """
    Enhanced performance monitoring and profiling system.
    
    Tracks application performance metrics, provides insights into
    bottlenecks, optimization opportunities, and predictive analysis.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self.request_counts = defaultdict(int)
        self.slow_queries: List[Dict[str, Any]] = []
        self.performance_baselines: Dict[str, float] = {}
        
        # Performance thresholds
        self.slow_request_threshold = 2.0  # seconds
        self.slow_query_threshold = 1.0    # seconds
        
    def start_request(self, request_id: str, endpoint: str, metadata: Optional[Dict[str, Any]] = None):
        """Start tracking a request with enhanced metadata."""
        self.active_requests[request_id] = {
            "endpoint": endpoint,
            "start_time": time.time(),
            "metadata": metadata or {}
        }
        self.request_counts[endpoint] += 1
        self.metrics.increment_counter("requests.total", tags={"endpoint": endpoint})
    
    def end_request(self, request_id: str, status_code: int, metadata: Optional[Dict[str, Any]] = None):
        """End tracking a request with enhanced analysis."""
        if request_id not in self.active_requests:
            logger.warning(f"Request {request_id} not found in active requests")
            return
        
        request_info = self.active_requests[request_id]
        endpoint = request_info["endpoint"]
        duration = time.time() - request_info["start_time"]
        
        del self.active_requests[request_id]
        
        # Record timing metrics
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
        
        # Detect slow requests
        if duration > self.slow_request_threshold:
            self._record_slow_request(endpoint, duration, status_code, metadata)
        
        # Update performance baselines
        self._update_baseline(f"request.{endpoint}", duration)
    
    def _record_slow_request(self, endpoint: str, duration: float, status_code: int, metadata: Optional[Dict[str, Any]]):
        """Record slow request for analysis."""
        slow_request = {
            "endpoint": endpoint,
            "duration": duration,
            "status_code": status_code,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        # Keep only recent slow requests (last 100)
        self.slow_queries.append(slow_request)
        if len(self.slow_queries) > 100:
            self.slow_queries = self.slow_queries[-100:]
        
        logger.warning(f"Slow request detected: {endpoint} took {duration:.2f}s")
    
    def record_database_query(
        self,
        query_type: str,
        duration: float,
        success: bool,
        query_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record database query metrics with enhanced analysis."""
        self.metrics.record_metric(
            "database.query.duration",
            duration,
            tags={"type": query_type, "success": str(success)},
            metadata=metadata
        )
        
        self.metrics.increment_counter(
            "database.queries.total",
            tags={"type": query_type, "success": str(success)}
        )
        
        # Track slow queries
        if duration > self.slow_query_threshold:
            self._record_slow_query(query_type, duration, success, query_hash, metadata)
        
        # Update performance baselines
        self._update_baseline(f"database.{query_type}", duration)
    
    def _record_slow_query(
        self,
        query_type: str,
        duration: float,
        success: bool,
        query_hash: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ):
        """Record slow database query for analysis."""
        slow_query = {
            "query_type": query_type,
            "duration": duration,
            "success": success,
            "query_hash": query_hash,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        # Keep only recent slow queries (last 100)
        self.slow_queries.append(slow_query)
        if len(self.slow_queries) > 100:
            self.slow_queries = self.slow_queries[-100:]
        
        logger.warning(f"Slow database query detected: {query_type} took {duration:.2f}s")
    
    def _update_baseline(self, operation: str, duration: float):
        """Update performance baseline for an operation."""
        if operation not in self.performance_baselines:
            self.performance_baselines[operation] = duration
        else:
            # Use exponential moving average
            alpha = 0.1  # Smoothing factor
            self.performance_baselines[operation] = (
                alpha * duration + (1 - alpha) * self.performance_baselines[operation]
            )
    
    def record_ai_inference(
        self,
        model_name: str,
        duration: float,
        success: bool,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record AI inference metrics with enhanced tracking."""
        self.metrics.record_timing(
            "ai.inference.duration",
            duration,
            tags={"model": model_name, "success": str(success)}
        )
        
        self.metrics.increment_counter(
            "ai.inferences.total",
            tags={"model": model_name, "success": str(success)}
        )
        
        # Track token usage if available
        if input_tokens is not None:
            self.metrics.record_metric(
                "ai.tokens.input",
                input_tokens,
                tags={"model": model_name}
            )
        
        if output_tokens is not None:
            self.metrics.record_metric(
                "ai.tokens.output",
                output_tokens,
                tags={"model": model_name}
            )
        
        # Calculate tokens per second if available
        if success and duration > 0 and output_tokens is not None:
            tokens_per_second = output_tokens / duration
            self.metrics.record_metric(
                "ai.tokens_per_second",
                tokens_per_second,
                tags={"model": model_name}
            )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get enhanced performance summary with insights."""
        summary = {
            "active_requests": len(self.active_requests),
            "request_counts": dict(self.request_counts),
            "avg_request_duration": self.metrics.get_metric_summary("requests.duration"),
            "database_performance": self.metrics.get_metric_summary("database.query.duration"),
            "ai_performance": self.metrics.get_metric_summary("ai.inference.duration"),
            "slow_requests": len([r for r in self.slow_queries if r.get("endpoint")]),
            "slow_queries": len([q for q in self.slow_queries if q.get("query_type")]),
            "performance_baselines": self.performance_baselines.copy(),
            "bottlenecks": self.metrics.detect_bottlenecks(),
            "recommendations": self._generate_performance_recommendations()
        }
        
        return summary
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Analyze request performance
        request_summary = self.metrics.get_metric_summary("requests.duration")
        if request_summary["latest"] and request_summary["latest"] > 1.0:
            recommendations.append("Request processing time is high. Consider optimizing request handlers.")
        
        # Analyze database performance
        db_summary = self.metrics.get_metric_summary("database.query.duration")
        if db_summary["latest"] and db_summary["latest"] > 0.5:
            recommendations.append("Database queries are slow. Consider adding indexes or optimizing queries.")
        
        # Analyze AI inference performance
        ai_summary = self.metrics.get_metric_summary("ai.inference.duration")
        if ai_summary["latest"] and ai_summary["latest"] > 5.0:
            recommendations.append("AI inference is slow. Consider model optimization or caching strategies.")
        
        # Check for high error rates
        error_rate = self._calculate_error_rate()
        if error_rate > 0.05:  # 5% error rate
            recommendations.append(f"High error rate detected ({error_rate:.1%}). Investigate error causes.")
        
        return recommendations
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate."""
        try:
            # Get recent status code metrics
            total_requests = 0
            error_requests = 0
            
            # This is a simplified calculation - in practice, you'd want to
            # aggregate status code metrics properly
            for endpoint, count in self.request_counts.items():
                total_requests += count
                # Assume some percentage are errors for demonstration
                # In real implementation, track actual error counts
            
            return error_requests / total_requests if total_requests > 0 else 0.0
        except Exception:
            return 0.0


class HealthMonitor:
    """
    Enhanced system health monitoring and alerting with predictive capabilities.
    
    Monitors system health indicators, triggers alerts when thresholds
    are exceeded, and provides predictive health analysis.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.health_checks: Dict[str, Callable] = {}
        self.thresholds: Dict[str, Dict[str, float]] = {}
        self.alert_handlers: List[Callable] = []
        self.health_history: List[Dict[str, Any]] = []
        
        # Predictive health analysis
        self.prediction_window = 300  # 5 minutes
        self.health_trends: Dict[str, List[float]] = defaultdict(list)
        
        # Setup default thresholds
        self._setup_default_thresholds()
    
    def _setup_default_thresholds(self):
        """Setup default alert thresholds for key metrics."""
        self.set_threshold("system.cpu.usage_percent", warning=70.0, critical=90.0)
        self.set_threshold("system.memory.usage_percent", warning=80.0, critical=95.0)
        self.set_threshold("system.disk.usage_percent", warning=85.0, critical=95.0)
        self.set_threshold("requests.duration", warning=2.0, critical=5.0)
        self.set_threshold("database.query.duration", warning=1.0, critical=3.0)
        self.set_threshold("ai.inference.duration", warning=10.0, critical=30.0)
        
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
        """Perform comprehensive system health check with predictive analysis."""
        health_status = {
            "overall_status": "healthy",
            "checks": {},
            "alerts": [],
            "predictions": {},
            "trends": {},
            "recommendations": []
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
        
        # Check metric thresholds with trend analysis
        for metric_name, thresholds in self.thresholds.items():
            summary = self.metrics.get_metric_summary(metric_name)
            
            if summary["latest"] is not None:
                latest_value = summary["latest"]
                
                # Store trend data
                self.health_trends[metric_name].append(latest_value)
                if len(self.health_trends[metric_name]) > 100:
                    self.health_trends[metric_name] = self.health_trends[metric_name][-100:]
                
                # Add trend information
                health_status["trends"][metric_name] = {
                    "current": latest_value,
                    "trend": summary["trend"],
                    "anomaly_score": summary["anomaly_score"]
                }
                
                # Check thresholds
                if latest_value >= thresholds["critical"]:
                    alert = {
                        "level": "critical",
                        "metric": metric_name,
                        "value": latest_value,
                        "threshold": thresholds["critical"],
                        "trend": summary["trend"]
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
                        "threshold": thresholds["warning"],
                        "trend": summary["trend"]
                    }
                    health_status["alerts"].append(alert)
                    
                    if health_status["overall_status"] == "healthy":
                        health_status["overall_status"] = "warning"
                
                # Predictive analysis
                prediction = self._predict_metric_health(metric_name, thresholds)
                if prediction:
                    health_status["predictions"][metric_name] = prediction
        
        # Generate health recommendations
        health_status["recommendations"] = self._generate_health_recommendations(health_status)
        
        # Store health history
        self.health_history.append({
            "timestamp": time.time(),
            "overall_status": health_status["overall_status"],
            "alert_count": len(health_status["alerts"]),
            "failed_checks": len([c for c in health_status["checks"].values() if c["status"] != "healthy"])
        })
        
        # Keep only recent history (last 1000 entries)
        if len(self.health_history) > 1000:
            self.health_history = self.health_history[-1000:]
        
        return health_status
    
    def _predict_metric_health(self, metric_name: str, thresholds: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Predict future health issues based on current trends."""
        if metric_name not in self.health_trends:
            return None
        
        trend_data = self.health_trends[metric_name]
        if len(trend_data) < 10:
            return None
        
        # Simple linear prediction
        recent_data = trend_data[-10:]
        if len(recent_data) < 2:
            return None
        
        # Calculate trend slope
        x_values = list(range(len(recent_data)))
        y_values = recent_data
        
        n = len(recent_data)
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n
        
        numerator = sum((x_values[i] - x_mean) * (y_values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return None
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Predict value in next 10 time periods
        future_periods = 10
        predicted_value = slope * (len(recent_data) + future_periods) + intercept
        
        # Check if prediction exceeds thresholds
        prediction = {
            "current_value": recent_data[-1],
            "predicted_value": predicted_value,
            "slope": slope,
            "periods_ahead": future_periods
        }
        
        if predicted_value >= thresholds["critical"]:
            prediction["alert_level"] = "critical"
            prediction["message"] = f"Metric {metric_name} predicted to reach critical threshold in {future_periods} periods"
        elif predicted_value >= thresholds["warning"]:
            prediction["alert_level"] = "warning"
            prediction["message"] = f"Metric {metric_name} predicted to reach warning threshold in {future_periods} periods"
        else:
            return None  # No predicted issues
        
        return prediction
    
    def _generate_health_recommendations(self, health_status: Dict[str, Any]) -> List[str]:
        """Generate health recommendations based on current status."""
        recommendations = []
        
        # CPU recommendations
        cpu_trend = health_status["trends"].get("system.cpu.usage_percent", {})
        if cpu_trend.get("trend") == "increasing" and cpu_trend.get("current", 0) > 60:
            recommendations.append("CPU usage is trending upward. Monitor for potential bottlenecks.")
        
        # Memory recommendations
        memory_trend = health_status["trends"].get("system.memory.usage_percent", {})
        if memory_trend.get("trend") == "increasing" and memory_trend.get("current", 0) > 70:
            recommendations.append("Memory usage is increasing. Check for memory leaks.")
        
        # Request performance recommendations
        request_trend = health_status["trends"].get("requests.duration", {})
        if request_trend.get("anomaly_score", 0) > 2.0:
            recommendations.append("Request processing shows anomalous behavior. Investigate recent changes.")
        
        # Database performance recommendations
        db_trend = health_status["trends"].get("database.query.duration", {})
        if db_trend.get("trend") == "increasing":
            recommendations.append("Database query performance is degrading. Consider query optimization.")
        
        # Predictive recommendations
        for metric_name, prediction in health_status.get("predictions", {}).items():
            if prediction["alert_level"] == "critical":
                recommendations.append(f"Take immediate action for {metric_name} - critical threshold predicted soon.")
            elif prediction["alert_level"] == "warning":
                recommendations.append(f"Monitor {metric_name} closely - warning threshold predicted.")
        
        return recommendations
    
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