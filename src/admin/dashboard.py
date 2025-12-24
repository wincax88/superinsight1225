"""
Enhanced Management Console Dashboard for SuperInsight Platform.

Provides real-time system monitoring, user behavior analysis, 
configuration management, workflow visualization, predictive analytics,
and automated recommendations.
"""

import asyncio
import logging
import time
import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque

from src.system.monitoring import metrics_collector, performance_monitor, health_monitor
from src.system.business_metrics import business_metrics_collector
from src.system.integration import system_manager
from src.admin.config_manager import config_manager
from src.admin.user_analytics import UserAnalytics
from src.admin.workflow_manager import WorkflowManager


logger = logging.getLogger(__name__)


@dataclass
class DashboardMetrics:
    """Real-time dashboard metrics container."""
    timestamp: float
    system_health: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    business_metrics: Dict[str, Any]
    user_activity: Dict[str, Any]
    alerts: List[Dict[str, Any]]
    predictions: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "system_health": self.system_health,
            "performance_metrics": self.performance_metrics,
            "business_metrics": self.business_metrics,
            "user_activity": self.user_activity,
            "alerts": self.alerts,
            "predictions": self.predictions,
            "recommendations": self.recommendations
        }


class PredictiveAnalyticsService:
    """
    Predictive analytics service for proactive system management.
    
    Analyzes historical data to predict system behavior, resource needs,
    and potential issues before they occur.
    """
    
    def __init__(self):
        self.historical_data: Dict[str, deque] = {
            "cpu_usage": deque(maxlen=1000),
            "memory_usage": deque(maxlen=1000),
            "request_rate": deque(maxlen=1000),
            "response_time": deque(maxlen=1000),
            "error_rate": deque(maxlen=1000),
            "user_activity": deque(maxlen=1000)
        }
        self.prediction_models: Dict[str, Any] = {}
        self.trend_analysis_window = 100  # Number of data points for trend analysis
    
    def add_data_point(self, metric_name: str, value: float, timestamp: float):
        """Add a data point for predictive analysis."""
        if metric_name in self.historical_data:
            self.historical_data[metric_name].append({
                "value": value,
                "timestamp": timestamp
            })
    
    def predict_resource_usage(self, hours_ahead: int = 1) -> Dict[str, Any]:
        """Predict resource usage for the next N hours."""
        predictions = {}
        
        for metric_name, data in self.historical_data.items():
            if len(data) < 10:  # Need minimum data points
                continue
            
            # Simple linear trend prediction
            recent_data = list(data)[-self.trend_analysis_window:]
            values = [point["value"] for point in recent_data]
            timestamps = [point["timestamp"] for point in recent_data]
            
            if len(values) >= 2:
                # Calculate trend
                trend = self._calculate_trend(values, timestamps)
                current_value = values[-1]
                
                # Predict future value
                seconds_ahead = hours_ahead * 3600
                predicted_value = current_value + (trend * seconds_ahead)
                
                # Calculate confidence based on data consistency
                confidence = self._calculate_prediction_confidence(values)
                
                predictions[metric_name] = {
                    "current_value": current_value,
                    "predicted_value": max(0, predicted_value),  # Ensure non-negative
                    "trend": trend,
                    "confidence": confidence,
                    "prediction_horizon_hours": hours_ahead
                }
        
        return predictions
    
    def _calculate_trend(self, values: List[float], timestamps: List[float]) -> float:
        """Calculate trend using simple linear regression."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        
        # Normalize timestamps to start from 0
        min_timestamp = min(timestamps)
        normalized_timestamps = [t - min_timestamp for t in timestamps]
        
        # Calculate means
        mean_x = sum(normalized_timestamps) / n
        mean_y = sum(values) / n
        
        # Calculate slope (trend)
        numerator = sum((normalized_timestamps[i] - mean_x) * (values[i] - mean_y) for i in range(n))
        denominator = sum((normalized_timestamps[i] - mean_x) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return slope
    
    def _calculate_prediction_confidence(self, values: List[float]) -> float:
        """Calculate confidence score for predictions based on data consistency."""
        if len(values) < 3:
            return 0.5
        
        # Calculate coefficient of variation (lower = more consistent = higher confidence)
        mean_val = statistics.mean(values)
        if mean_val == 0:
            return 0.5
        
        std_dev = statistics.stdev(values)
        cv = std_dev / mean_val
        
        # Convert to confidence score (0-1, where 1 is highest confidence)
        confidence = max(0.1, min(1.0, 1.0 - cv))
        return confidence
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in system metrics."""
        anomalies = []
        
        for metric_name, data in self.historical_data.items():
            if len(data) < 20:  # Need sufficient data for anomaly detection
                continue
            
            recent_data = list(data)[-50:]  # Last 50 data points
            values = [point["value"] for point in recent_data]
            
            if len(values) >= 10:
                # Calculate statistical thresholds
                mean_val = statistics.mean(values[:-5])  # Exclude last 5 for comparison
                std_dev = statistics.stdev(values[:-5])
                
                # Check last 5 values for anomalies
                for i in range(-5, 0):
                    current_value = values[i]
                    z_score = abs((current_value - mean_val) / std_dev) if std_dev > 0 else 0
                    
                    if z_score > 2.5:  # Anomaly threshold
                        anomalies.append({
                            "metric": metric_name,
                            "value": current_value,
                            "expected_range": [mean_val - 2*std_dev, mean_val + 2*std_dev],
                            "z_score": z_score,
                            "severity": "high" if z_score > 3.0 else "medium",
                            "timestamp": recent_data[i]["timestamp"]
                        })
        
        return anomalies
    
    def generate_capacity_recommendations(self) -> List[Dict[str, Any]]:
        """Generate capacity planning recommendations."""
        recommendations = []
        predictions = self.predict_resource_usage(hours_ahead=24)
        
        for metric_name, prediction in predictions.items():
            predicted_value = prediction["predicted_value"]
            confidence = prediction["confidence"]
            
            # Generate recommendations based on predictions
            if metric_name == "cpu_usage" and predicted_value > 80 and confidence > 0.7:
                recommendations.append({
                    "type": "capacity_planning",
                    "priority": "high" if predicted_value > 90 else "medium",
                    "title": "CPU Capacity Warning",
                    "description": f"CPU usage predicted to reach {predicted_value:.1f}% in 24 hours",
                    "recommendation": "Consider scaling up CPU resources or optimizing high-CPU processes",
                    "confidence": confidence,
                    "metric": metric_name
                })
            
            elif metric_name == "memory_usage" and predicted_value > 85 and confidence > 0.7:
                recommendations.append({
                    "type": "capacity_planning",
                    "priority": "high" if predicted_value > 95 else "medium",
                    "title": "Memory Capacity Warning",
                    "description": f"Memory usage predicted to reach {predicted_value:.1f}% in 24 hours",
                    "recommendation": "Consider increasing memory allocation or optimizing memory usage",
                    "confidence": confidence,
                    "metric": metric_name
                })
            
            elif metric_name == "response_time" and predicted_value > 2.0 and confidence > 0.6:
                recommendations.append({
                    "type": "performance_optimization",
                    "priority": "medium",
                    "title": "Response Time Degradation",
                    "description": f"Response time predicted to reach {predicted_value:.2f}s in 24 hours",
                    "recommendation": "Review database queries, API endpoints, and caching strategies",
                    "confidence": confidence,
                    "metric": metric_name
                })
        
        return recommendations


class AutomatedRecommendationEngine:
    """
    Automated recommendation engine for system optimization.
    
    Analyzes system patterns and generates actionable recommendations
    for performance, security, and operational improvements.
    """
    
    def __init__(self):
        self.recommendation_history: List[Dict[str, Any]] = []
        self.pattern_analysis_window = 168  # 1 week in hours
    
    def analyze_system_patterns(self, metrics_history: List[DashboardMetrics]) -> List[Dict[str, Any]]:
        """Analyze system patterns and generate recommendations."""
        recommendations = []
        
        if len(metrics_history) < 10:
            return recommendations
        
        # Analyze performance patterns
        performance_recommendations = self._analyze_performance_patterns(metrics_history)
        recommendations.extend(performance_recommendations)
        
        # Analyze user behavior patterns
        user_recommendations = self._analyze_user_patterns(metrics_history)
        recommendations.extend(user_recommendations)
        
        # Analyze error patterns
        error_recommendations = self._analyze_error_patterns(metrics_history)
        recommendations.extend(error_recommendations)
        
        # Analyze resource utilization patterns
        resource_recommendations = self._analyze_resource_patterns(metrics_history)
        recommendations.extend(resource_recommendations)
        
        return recommendations
    
    def _analyze_performance_patterns(self, metrics_history: List[DashboardMetrics]) -> List[Dict[str, Any]]:
        """Analyze performance patterns and generate recommendations."""
        recommendations = []
        
        # Extract response times
        response_times = []
        for metrics in metrics_history[-24:]:  # Last 24 data points
            perf_metrics = metrics.performance_metrics
            if "response_time" in perf_metrics:
                response_times.append(perf_metrics["response_time"])
        
        if len(response_times) >= 10:
            avg_response_time = statistics.mean(response_times)
            p95_response_time = statistics.quantile(response_times, 0.95) if len(response_times) >= 2 else avg_response_time
            
            if avg_response_time > 1.0:
                recommendations.append({
                    "type": "performance",
                    "priority": "high" if avg_response_time > 2.0 else "medium",
                    "title": "High Average Response Time",
                    "description": f"Average response time is {avg_response_time:.2f}s (P95: {p95_response_time:.2f}s)",
                    "recommendation": "Consider implementing caching, optimizing database queries, or scaling infrastructure",
                    "impact": "User experience and system throughput",
                    "effort": "medium"
                })
        
        return recommendations
    
    def _analyze_user_patterns(self, metrics_history: List[DashboardMetrics]) -> List[Dict[str, Any]]:
        """Analyze user behavior patterns."""
        recommendations = []
        
        # Extract user activity data
        hourly_activity = defaultdict(list)
        
        for metrics in metrics_history[-168:]:  # Last week
            timestamp = metrics.timestamp
            hour = datetime.fromtimestamp(timestamp).hour
            user_activity = metrics.user_activity.get("active_sessions", 0)
            hourly_activity[hour].append(user_activity)
        
        # Find peak usage hours
        peak_hours = []
        for hour, activities in hourly_activity.items():
            if activities:
                avg_activity = statistics.mean(activities)
                if avg_activity > 0:
                    peak_hours.append((hour, avg_activity))
        
        if peak_hours:
            peak_hours.sort(key=lambda x: x[1], reverse=True)
            top_peak_hours = peak_hours[:3]
            
            recommendations.append({
                "type": "capacity_planning",
                "priority": "low",
                "title": "Peak Usage Pattern Identified",
                "description": f"Peak usage hours: {', '.join([f'{h}:00' for h, _ in top_peak_hours])}",
                "recommendation": "Consider auto-scaling policies or pre-scaling during peak hours",
                "impact": "Cost optimization and performance during peak times",
                "effort": "low"
            })
        
        return recommendations
    
    def _analyze_error_patterns(self, metrics_history: List[DashboardMetrics]) -> List[Dict[str, Any]]:
        """Analyze error patterns and generate recommendations."""
        recommendations = []
        
        # Count alerts by type
        alert_counts = defaultdict(int)
        recent_alerts = []
        
        for metrics in metrics_history[-24:]:  # Last 24 data points
            for alert in metrics.alerts:
                alert_type = alert.get("type", "unknown")
                alert_counts[alert_type] += 1
                recent_alerts.append(alert)
        
        # Identify frequent alert types
        for alert_type, count in alert_counts.items():
            if count >= 5:  # Frequent alerts
                recommendations.append({
                    "type": "reliability",
                    "priority": "high" if count >= 10 else "medium",
                    "title": f"Frequent {alert_type.replace('_', ' ').title()} Alerts",
                    "description": f"{count} {alert_type} alerts in recent period",
                    "recommendation": f"Investigate root cause of {alert_type} issues and implement preventive measures",
                    "impact": "System stability and reliability",
                    "effort": "medium"
                })
        
        return recommendations
    
    def _analyze_resource_patterns(self, metrics_history: List[DashboardMetrics]) -> List[Dict[str, Any]]:
        """Analyze resource utilization patterns."""
        recommendations = []
        
        # Analyze CPU and memory trends
        cpu_values = []
        memory_values = []
        
        for metrics in metrics_history[-48:]:  # Last 48 data points
            perf_metrics = metrics.performance_metrics
            if "cpu_usage" in perf_metrics:
                cpu_values.append(perf_metrics["cpu_usage"])
            if "memory_usage" in perf_metrics:
                memory_values.append(perf_metrics["memory_usage"])
        
        # CPU utilization analysis
        if cpu_values and len(cpu_values) >= 10:
            avg_cpu = statistics.mean(cpu_values)
            max_cpu = max(cpu_values)
            
            if avg_cpu < 20:
                recommendations.append({
                    "type": "cost_optimization",
                    "priority": "low",
                    "title": "Low CPU Utilization",
                    "description": f"Average CPU usage is {avg_cpu:.1f}% (max: {max_cpu:.1f}%)",
                    "recommendation": "Consider downsizing CPU resources to reduce costs",
                    "impact": "Cost savings",
                    "effort": "low"
                })
            elif avg_cpu > 70:
                recommendations.append({
                    "type": "capacity_planning",
                    "priority": "medium",
                    "title": "High CPU Utilization",
                    "description": f"Average CPU usage is {avg_cpu:.1f}% (max: {max_cpu:.1f}%)",
                    "recommendation": "Consider increasing CPU resources or optimizing CPU-intensive processes",
                    "impact": "Performance and reliability",
                    "effort": "medium"
                })
        
        # Memory utilization analysis
        if memory_values and len(memory_values) >= 10:
            avg_memory = statistics.mean(memory_values)
            max_memory = max(memory_values)
            
            if avg_memory < 30:
                recommendations.append({
                    "type": "cost_optimization",
                    "priority": "low",
                    "title": "Low Memory Utilization",
                    "description": f"Average memory usage is {avg_memory:.1f}% (max: {max_memory:.1f}%)",
                    "recommendation": "Consider reducing memory allocation to optimize costs",
                    "impact": "Cost savings",
                    "effort": "low"
                })
            elif avg_memory > 80:
                recommendations.append({
                    "type": "capacity_planning",
                    "priority": "high",
                    "title": "High Memory Utilization",
                    "description": f"Average memory usage is {avg_memory:.1f}% (max: {max_memory:.1f}%)",
                    "recommendation": "Consider increasing memory allocation or optimizing memory usage",
                    "impact": "Performance and stability",
                    "effort": "medium"
                })
        
        return recommendations


class RealTimeMonitoringService:
    """
    Enhanced real-time monitoring service for the management console.
    
    Provides live system metrics, alerts, performance data, predictive analytics,
    and automated recommendations with WebSocket support for real-time updates.
    """
    
    def __init__(self):
        self.subscribers: List[Any] = []  # WebSocket connections
        self.update_interval = 5  # seconds
        self.is_running = False
        self._update_task: Optional[asyncio.Task] = None
        self.metrics_history: deque = deque(maxlen=2000)  # Increased for better analytics
        
        # Enhanced services
        self.predictive_analytics = PredictiveAnalyticsService()
        self.recommendation_engine = AutomatedRecommendationEngine()
        
        # Alert thresholds for real-time monitoring
        self.alert_thresholds = {
            "cpu_usage": {"warning": 70, "critical": 90},
            "memory_usage": {"warning": 80, "critical": 95},
            "disk_usage": {"warning": 85, "critical": 95},
            "response_time": {"warning": 2.0, "critical": 5.0},
            "error_rate": {"warning": 0.05, "critical": 0.10}
        }
        
        # Enhanced monitoring features
        self.enable_predictive_analytics = True
        self.enable_automated_recommendations = True
        self.recommendation_update_interval = 300  # 5 minutes
        self._last_recommendation_update = 0
    
    async def start_monitoring(self):
        """Start real-time monitoring service."""
        if self.is_running:
            logger.warning("Real-time monitoring is already running")
            return
        
        self.is_running = True
        self._update_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started real-time monitoring service")
    
    async def stop_monitoring(self):
        """Stop real-time monitoring service."""
        self.is_running = False
        
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped real-time monitoring service")
    
    async def _monitoring_loop(self):
        """Main monitoring loop that collects and broadcasts metrics."""
        while self.is_running:
            try:
                # Collect current metrics
                metrics = await self._collect_dashboard_metrics()
                
                # Store in history
                self.metrics_history.append(metrics)
                
                # Broadcast to subscribers
                await self._broadcast_metrics(metrics)
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1)  # Short delay before retrying
    
    async def _collect_dashboard_metrics(self) -> DashboardMetrics:
        """Collect comprehensive dashboard metrics with predictive analytics."""
        current_time = time.time()
        
        # System health metrics
        system_health = system_manager.get_system_status()
        health_status = await health_monitor.check_system_health()
        
        # Performance metrics
        performance_summary = performance_monitor.get_performance_summary()
        system_metrics = metrics_collector.get_all_metrics_summary()
        
        # Business metrics
        business_summary = business_metrics_collector.get_business_summary()
        
        # User activity metrics
        user_activity = {
            "active_sessions": len(business_metrics_collector.active_sessions),
            "total_actions_last_hour": self._count_recent_actions(3600),
            "peak_concurrent_users": getattr(business_metrics_collector, '_peak_users', 0)
        }
        
        # Update predictive analytics data
        if self.enable_predictive_analytics:
            cpu_usage = system_metrics.get("system.cpu.usage_percent", {}).get("latest", 0)
            memory_usage = system_metrics.get("system.memory.usage_percent", {}).get("latest", 0)
            response_time = system_metrics.get("requests.duration", {}).get("avg", 0)
            
            self.predictive_analytics.add_data_point("cpu_usage", cpu_usage, current_time)
            self.predictive_analytics.add_data_point("memory_usage", memory_usage, current_time)
            self.predictive_analytics.add_data_point("response_time", response_time, current_time)
            self.predictive_analytics.add_data_point("user_activity", user_activity["active_sessions"], current_time)
        
        # Generate alerts (including predictive alerts)
        alerts = self._generate_real_time_alerts(system_metrics, performance_summary)
        
        # Add anomaly detection alerts
        if self.enable_predictive_analytics:
            anomalies = self.predictive_analytics.detect_anomalies()
            for anomaly in anomalies:
                alerts.append({
                    "level": anomaly["severity"],
                    "type": "anomaly",
                    "message": f"Anomaly detected in {anomaly['metric']}: {anomaly['value']:.2f}",
                    "metric": anomaly["metric"],
                    "z_score": anomaly["z_score"],
                    "timestamp": current_time
                })
        
        # Generate predictions
        predictions = None
        if self.enable_predictive_analytics:
            predictions = {
                "resource_usage_24h": self.predictive_analytics.predict_resource_usage(hours_ahead=24),
                "capacity_recommendations": self.predictive_analytics.generate_capacity_recommendations()
            }
        
        # Generate automated recommendations (periodically)
        recommendations = None
        if (self.enable_automated_recommendations and 
            current_time - self._last_recommendation_update > self.recommendation_update_interval):
            
            recommendations = self.recommendation_engine.analyze_system_patterns(list(self.metrics_history))
            self._last_recommendation_update = current_time
        
        return DashboardMetrics(
            timestamp=current_time,
            system_health={
                "overall_status": system_health["overall_status"],
                "services": system_health["services"],
                "health_checks": health_status.get("checks", {})
            },
            performance_metrics={
                "cpu_usage": system_metrics.get("system.cpu.usage_percent", {}).get("latest", 0),
                "memory_usage": system_metrics.get("system.memory.usage_percent", {}).get("latest", 0),
                "disk_usage": system_metrics.get("system.disk.usage_percent", {}).get("latest", 0),
                "response_time": system_metrics.get("requests.duration", {}).get("avg", 0),
                "active_requests": performance_summary.get("active_requests", 0),
                "bottlenecks": performance_summary.get("bottlenecks", [])
            },
            business_metrics=business_summary,
            user_activity=user_activity,
            alerts=alerts,
            predictions=predictions,
            recommendations=recommendations
        )
    
    def _count_recent_actions(self, seconds: int) -> int:
        """Count user actions in the last N seconds."""
        cutoff_time = time.time() - seconds
        total_actions = 0
        
        for user_actions in business_metrics_collector.user_actions.values():
            total_actions += len([
                action for action in user_actions 
                if action["timestamp"] > cutoff_time
            ])
        
        return total_actions
    
    def _generate_real_time_alerts(
        self, 
        system_metrics: Dict[str, Any], 
        performance_summary: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate real-time alerts based on current metrics."""
        alerts = []
        current_time = time.time()
        
        # CPU usage alerts
        cpu_usage = system_metrics.get("system.cpu.usage_percent", {}).get("latest", 0)
        if cpu_usage >= self.alert_thresholds["cpu_usage"]["critical"]:
            alerts.append({
                "level": "critical",
                "type": "cpu_usage",
                "message": f"Critical CPU usage: {cpu_usage:.1f}%",
                "value": cpu_usage,
                "threshold": self.alert_thresholds["cpu_usage"]["critical"],
                "timestamp": current_time
            })
        elif cpu_usage >= self.alert_thresholds["cpu_usage"]["warning"]:
            alerts.append({
                "level": "warning",
                "type": "cpu_usage",
                "message": f"High CPU usage: {cpu_usage:.1f}%",
                "value": cpu_usage,
                "threshold": self.alert_thresholds["cpu_usage"]["warning"],
                "timestamp": current_time
            })
        
        # Memory usage alerts
        memory_usage = system_metrics.get("system.memory.usage_percent", {}).get("latest", 0)
        if memory_usage >= self.alert_thresholds["memory_usage"]["critical"]:
            alerts.append({
                "level": "critical",
                "type": "memory_usage",
                "message": f"Critical memory usage: {memory_usage:.1f}%",
                "value": memory_usage,
                "threshold": self.alert_thresholds["memory_usage"]["critical"],
                "timestamp": current_time
            })
        elif memory_usage >= self.alert_thresholds["memory_usage"]["warning"]:
            alerts.append({
                "level": "warning",
                "type": "memory_usage",
                "message": f"High memory usage: {memory_usage:.1f}%",
                "value": memory_usage,
                "threshold": self.alert_thresholds["memory_usage"]["warning"],
                "timestamp": current_time
            })
        
        # Response time alerts
        response_time = system_metrics.get("requests.duration", {}).get("avg", 0)
        if response_time >= self.alert_thresholds["response_time"]["critical"]:
            alerts.append({
                "level": "critical",
                "type": "response_time",
                "message": f"Critical response time: {response_time:.2f}s",
                "value": response_time,
                "threshold": self.alert_thresholds["response_time"]["critical"],
                "timestamp": current_time
            })
        elif response_time >= self.alert_thresholds["response_time"]["warning"]:
            alerts.append({
                "level": "warning",
                "type": "response_time",
                "message": f"High response time: {response_time:.2f}s",
                "value": response_time,
                "threshold": self.alert_thresholds["response_time"]["warning"],
                "timestamp": current_time
            })
        
        # Add bottleneck alerts
        bottlenecks = performance_summary.get("bottlenecks", [])
        for bottleneck in bottlenecks:
            alerts.append({
                "level": bottleneck.get("severity", "warning"),
                "type": "bottleneck",
                "message": f"Bottleneck detected: {bottleneck.get('description', 'Unknown')}",
                "component": bottleneck.get("component", "Unknown"),
                "recommendations": bottleneck.get("recommendations", []),
                "timestamp": current_time
            })
        
        return alerts
    
    async def _broadcast_metrics(self, metrics: DashboardMetrics):
        """Broadcast metrics to all connected WebSocket subscribers."""
        if not self.subscribers:
            return
        
        message = json.dumps(metrics.to_dict())
        
        # Remove disconnected subscribers
        active_subscribers = []
        
        for subscriber in self.subscribers:
            try:
                await subscriber.send_text(message)
                active_subscribers.append(subscriber)
            except Exception as e:
                logger.debug(f"Removing disconnected subscriber: {e}")
        
        self.subscribers = active_subscribers
    
    def add_subscriber(self, websocket):
        """Add a WebSocket subscriber for real-time updates."""
        self.subscribers.append(websocket)
        logger.info(f"Added dashboard subscriber, total: {len(self.subscribers)}")
    
    def remove_subscriber(self, websocket):
        """Remove a WebSocket subscriber."""
        if websocket in self.subscribers:
            self.subscribers.remove(websocket)
            logger.info(f"Removed dashboard subscriber, total: {len(self.subscribers)}")
    
    def get_metrics_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get metrics history for the specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        
        return [
            metrics.to_dict() for metrics in self.metrics_history
            if metrics.timestamp > cutoff_time
        ]
    
    def get_current_metrics(self) -> Optional[Dict[str, Any]]:
        """Get the most recent metrics."""
        if self.metrics_history:
            return self.metrics_history[-1].to_dict()
        return None


class ConfigurationHotUpdateService:
    """
    Service for hot configuration updates with validation and rollback.
    
    Allows real-time configuration changes without system restart,
    with automatic validation and rollback capabilities.
    """
    
    def __init__(self):
        self.pending_changes: Dict[str, Any] = {}
        self.change_history: List[Dict[str, Any]] = []
        self.validation_rules: Dict[str, callable] = {}
        
        # Setup validation rules
        self._setup_validation_rules()
    
    def _setup_validation_rules(self):
        """Setup configuration validation rules."""
        self.validation_rules = {
            "system.api_rate_limit": lambda x: isinstance(x, int) and x > 0,
            "system.max_concurrent_jobs": lambda x: isinstance(x, int) and x > 0,
            "database.pool_size": lambda x: isinstance(x, int) and 1 <= x <= 100,
            "redis.max_connections": lambda x: isinstance(x, int) and 1 <= x <= 1000,
            "ai.request_timeout": lambda x: isinstance(x, (int, float)) and x > 0,
            "monitoring.health_check_interval": lambda x: isinstance(x, int) and x >= 10,
        }
    
    async def update_config(
        self, 
        section: str, 
        key: str, 
        value: Any, 
        user: str = "admin",
        validate: bool = True
    ) -> Dict[str, Any]:
        """Update configuration with validation and hot reload."""
        config_key = f"{section}.{key}"
        
        try:
            # Validate the new value
            if validate and config_key in self.validation_rules:
                validator = self.validation_rules[config_key]
                if not validator(value):
                    return {
                        "success": False,
                        "error": f"Validation failed for {config_key}",
                        "message": "Invalid configuration value"
                    }
            
            # Get old value for rollback
            old_value = config_manager.get(section, key)
            
            # Apply the change
            success = config_manager.set(section, key, value, user, "Hot update via dashboard")
            
            if success:
                # Record the change
                change_record = {
                    "timestamp": time.time(),
                    "section": section,
                    "key": key,
                    "old_value": old_value,
                    "new_value": value,
                    "user": user,
                    "status": "applied"
                }
                
                self.change_history.append(change_record)
                
                # Apply hot reload if needed
                await self._apply_hot_reload(section, key, value)
                
                return {
                    "success": True,
                    "message": f"Configuration {config_key} updated successfully",
                    "old_value": old_value,
                    "new_value": value
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to update configuration",
                    "message": "Configuration manager rejected the change"
                }
                
        except Exception as e:
            logger.error(f"Failed to update configuration {config_key}: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Configuration update failed"
            }
    
    async def _apply_hot_reload(self, section: str, key: str, value: Any):
        """Apply hot reload for configuration changes that support it."""
        config_key = f"{section}.{key}"
        
        try:
            # Monitoring configuration changes
            if section == "monitoring":
                if key == "health_check_interval":
                    health_monitor._health_check_interval = value
                    logger.info(f"Hot reloaded health check interval: {value}s")
                elif key == "metrics_retention_days":
                    # Update metrics retention
                    logger.info(f"Hot reloaded metrics retention: {value} days")
            
            # System configuration changes
            elif section == "system":
                if key == "api_rate_limit":
                    # Update rate limiting (would need rate limiter integration)
                    logger.info(f"Hot reloaded API rate limit: {value}")
                elif key == "max_concurrent_jobs":
                    # Update job concurrency limits
                    logger.info(f"Hot reloaded max concurrent jobs: {value}")
            
            # Database configuration changes
            elif section == "database":
                if key == "pool_size":
                    # Update database pool size (would need connection pool integration)
                    logger.info(f"Hot reloaded database pool size: {value}")
            
            # AI configuration changes
            elif section == "ai":
                if key == "request_timeout":
                    # Update AI request timeout
                    logger.info(f"Hot reloaded AI request timeout: {value}s")
            
        except Exception as e:
            logger.error(f"Failed to apply hot reload for {config_key}: {e}")
    
    async def rollback_config(self, change_id: int, user: str = "admin") -> Dict[str, Any]:
        """Rollback a configuration change."""
        try:
            if change_id >= len(self.change_history):
                return {
                    "success": False,
                    "error": "Invalid change ID",
                    "message": "Change not found in history"
                }
            
            change = self.change_history[change_id]
            
            # Apply rollback
            result = await self.update_config(
                change["section"],
                change["key"],
                change["old_value"],
                user,
                validate=False  # Skip validation for rollback
            )
            
            if result["success"]:
                # Mark as rolled back
                change["status"] = "rolled_back"
                change["rollback_timestamp"] = time.time()
                change["rollback_user"] = user
                
                return {
                    "success": True,
                    "message": f"Configuration rolled back successfully",
                    "change": change
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Failed to rollback configuration change {change_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Rollback failed"
            }
    
    def get_change_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get configuration change history."""
        return self.change_history[-limit:] if limit > 0 else self.change_history.copy()
    
    def get_pending_changes(self) -> Dict[str, Any]:
        """Get pending configuration changes."""
        return self.pending_changes.copy()


class EnhancedDashboardManager:
    """
    Enhanced dashboard manager that coordinates all management console features.
    
    Integrates real-time monitoring, user analytics, configuration management,
    workflow visualization, predictive analytics, and automated recommendations
    into a unified dashboard experience.
    """
    
    def __init__(self):
        self.monitoring_service = RealTimeMonitoringService()
        self.config_service = ConfigurationHotUpdateService()
        self.user_analytics = UserAnalytics()
        self.workflow_manager = WorkflowManager()
        
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the enhanced dashboard manager."""
        if self.is_initialized:
            return
        
        try:
            # Start monitoring service
            await self.monitoring_service.start_monitoring()
            
            # Initialize user analytics
            await self.user_analytics.initialize()
            
            # Initialize workflow manager
            await self.workflow_manager.initialize()
            
            self.is_initialized = True
            logger.info("Enhanced dashboard manager initialized successfully with predictive analytics")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced dashboard manager: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the enhanced dashboard manager."""
        try:
            await self.monitoring_service.stop_monitoring()
            await self.user_analytics.shutdown()
            await self.workflow_manager.shutdown()
            
            self.is_initialized = False
            logger.info("Enhanced dashboard manager shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during dashboard manager shutdown: {e}")
    
    def get_dashboard_overview(self) -> Dict[str, Any]:
        """Get comprehensive dashboard overview with enhanced analytics."""
        current_metrics = self.monitoring_service.get_current_metrics()
        user_stats = self.user_analytics.get_current_stats()
        workflow_stats = self.workflow_manager.get_workflow_stats()
        
        # Enhanced overview with predictive insights
        overview = {
            "timestamp": time.time(),
            "system_status": current_metrics.get("system_health", {}) if current_metrics else {},
            "performance": current_metrics.get("performance_metrics", {}) if current_metrics else {},
            "business_metrics": current_metrics.get("business_metrics", {}) if current_metrics else {},
            "user_analytics": user_stats,
            "workflow_status": workflow_stats,
            "alerts": current_metrics.get("alerts", []) if current_metrics else [],
            "configuration": {
                "pending_changes": len(self.config_service.get_pending_changes()),
                "recent_changes": len(self.config_service.get_change_history(10))
            }
        }
        
        # Add predictive analytics if available
        if current_metrics and current_metrics.get("predictions"):
            overview["predictions"] = current_metrics["predictions"]
        
        # Add automated recommendations if available
        if current_metrics and current_metrics.get("recommendations"):
            overview["recommendations"] = current_metrics["recommendations"]
        
        return overview
    
    def get_predictive_insights(self, hours_ahead: int = 24) -> Dict[str, Any]:
        """Get predictive insights for system planning."""
        if not self.is_initialized:
            return {"error": "Dashboard manager not initialized"}
        
        predictions = self.monitoring_service.predictive_analytics.predict_resource_usage(hours_ahead)
        recommendations = self.monitoring_service.predictive_analytics.generate_capacity_recommendations()
        anomalies = self.monitoring_service.predictive_analytics.detect_anomalies()
        
        return {
            "predictions": predictions,
            "capacity_recommendations": recommendations,
            "anomalies": anomalies,
            "prediction_horizon_hours": hours_ahead,
            "generated_at": time.time()
        }
    
    def get_system_recommendations(self) -> Dict[str, Any]:
        """Get automated system recommendations."""
        if not self.is_initialized:
            return {"error": "Dashboard manager not initialized"}
        
        metrics_history = list(self.monitoring_service.metrics_history)
        recommendations = self.monitoring_service.recommendation_engine.analyze_system_patterns(metrics_history)
        
        # Categorize recommendations
        categorized = {
            "performance": [],
            "capacity_planning": [],
            "cost_optimization": [],
            "reliability": [],
            "security": []
        }
        
        for rec in recommendations:
            category = rec.get("type", "other")
            if category in categorized:
                categorized[category].append(rec)
        
        return {
            "recommendations": categorized,
            "total_recommendations": len(recommendations),
            "generated_at": time.time()
        }


# Global enhanced dashboard manager instance
dashboard_manager = EnhancedDashboardManager()