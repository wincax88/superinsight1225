"""
Prometheus Metrics Exporter for SuperInsight Platform.

Exports system and business metrics in Prometheus format for monitoring
and alerting with Prometheus and Grafana.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from prometheus_client import (
    Counter, Gauge, Histogram, Summary, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)
from fastapi import Response


logger = logging.getLogger(__name__)


class PrometheusExporter:
    """
    Prometheus metrics exporter for SuperInsight platform.
    
    Converts internal metrics to Prometheus format and provides
    HTTP endpoint for Prometheus scraping.
    """
    
    def __init__(self):
        # Create custom registry to avoid conflicts
        self.registry = CollectorRegistry()
        
        # System metrics
        self.system_cpu_usage = Gauge(
            'superinsight_system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'superinsight_system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.system_disk_usage = Gauge(
            'superinsight_system_disk_usage_percent',
            'System disk usage percentage',
            registry=self.registry
        )
        
        # Application metrics
        self.http_requests_total = Counter(
            'superinsight_http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.http_request_duration = Histogram(
            'superinsight_http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.database_queries_total = Counter(
            'superinsight_database_queries_total',
            'Total database queries',
            ['type', 'success'],
            registry=self.registry
        )
        
        self.database_query_duration = Histogram(
            'superinsight_database_query_duration_seconds',
            'Database query duration in seconds',
            ['type'],
            registry=self.registry
        )
        
        # Business metrics
        self.annotation_efficiency = Gauge(
            'superinsight_annotation_efficiency_per_hour',
            'Annotations completed per hour',
            registry=self.registry
        )
        
        self.annotation_quality_score = Gauge(
            'superinsight_annotation_quality_score',
            'Average annotation quality score',
            registry=self.registry
        )
        
        self.active_users = Gauge(
            'superinsight_active_users_count',
            'Number of currently active users',
            registry=self.registry
        )
        
        self.user_sessions_total = Counter(
            'superinsight_user_sessions_total',
            'Total user sessions',
            registry=self.registry
        )
        
        self.ai_inferences_total = Counter(
            'superinsight_ai_inferences_total',
            'Total AI inferences',
            ['model', 'success'],
            registry=self.registry
        )
        
        self.ai_inference_duration = Histogram(
            'superinsight_ai_inference_duration_seconds',
            'AI inference duration in seconds',
            ['model'],
            registry=self.registry
        )
        
        self.ai_confidence_score = Gauge(
            'superinsight_ai_confidence_score',
            'Average AI confidence score',
            ['model'],
            registry=self.registry
        )
        
        self.project_completion_percentage = Gauge(
            'superinsight_project_completion_percentage',
            'Project completion percentage',
            ['project_id'],
            registry=self.registry
        )
        
        self.project_tasks_total = Gauge(
            'superinsight_project_tasks_total',
            'Total tasks in project',
            ['project_id', 'status'],
            registry=self.registry
        )
        
        # Billing metrics
        self.billing_cost_total = Counter(
            'superinsight_billing_cost_total',
            'Total billing cost',
            ['tenant_id'],
            registry=self.registry
        )
        
        self.billing_annotations_total = Counter(
            'superinsight_billing_annotations_total',
            'Total billable annotations',
            ['tenant_id'],
            registry=self.registry
        )
        
        # Quality metrics
        self.quality_issues_total = Counter(
            'superinsight_quality_issues_total',
            'Total quality issues',
            ['severity', 'status'],
            registry=self.registry
        )
        
        # Application info
        self.app_info = Info(
            'superinsight_app_info',
            'Application information',
            registry=self.registry
        )
        
        # Set application info
        self.app_info.info({
            'version': '1.0.0',
            'name': 'SuperInsight Platform',
            'description': 'AI Data Governance and Annotation Platform'
        })
        
        # Last update timestamp
        self.last_update = 0
        self.update_interval = 30  # seconds
    
    def update_metrics(self):
        """Update all Prometheus metrics from internal collectors."""
        current_time = time.time()
        
        # Skip update if too recent
        if current_time - self.last_update < self.update_interval:
            return
        
        try:
            self._update_system_metrics()
            self._update_application_metrics()
            self._update_business_metrics()
            self._update_ai_metrics()
            self._update_project_metrics()
            
            self.last_update = current_time
            
        except Exception as e:
            logger.error(f"Failed to update Prometheus metrics: {e}")
    
    def _update_system_metrics(self):
        """Update system-level metrics."""
        try:
            # Import here to avoid circular imports
            from src.system.monitoring import metrics_collector
            
            system_metrics = metrics_collector.get_all_metrics_summary()
            
            # CPU usage
            cpu_metric = system_metrics.get("system.cpu.usage_percent", {})
            if cpu_metric.get("latest") is not None:
                self.system_cpu_usage.set(cpu_metric["latest"])
            
            # Memory usage
            memory_metric = system_metrics.get("system.memory.usage_percent", {})
            if memory_metric.get("latest") is not None:
                self.system_memory_usage.set(memory_metric["latest"])
            
            # Disk usage
            disk_metric = system_metrics.get("system.disk.usage_percent", {})
            if disk_metric.get("latest") is not None:
                self.system_disk_usage.set(disk_metric["latest"])
                
        except Exception as e:
            logger.warning(f"Failed to update system metrics: {e}")
    
    def _update_application_metrics(self):
        """Update application-level metrics."""
        try:
            # Import here to avoid circular imports
            from src.system.monitoring import performance_monitor
            
            performance_summary = performance_monitor.get_performance_summary()
            
            # Active requests
            active_requests = performance_summary.get("active_requests", 0)
            # Note: This would be better as a gauge, but we'll track it differently
            
            # Request counts and durations would be updated via the track_* methods
            
        except Exception as e:
            logger.warning(f"Failed to update application metrics: {e}")
    
    def _update_business_metrics(self):
        """Update business-level metrics."""
        try:
            # Import here to avoid circular imports
            from src.system.business_metrics import business_metrics_collector
            
            business_summary = business_metrics_collector.get_business_summary()
            
            # Annotation efficiency
            annotation_efficiency = business_summary.get("annotation_efficiency", {})
            if annotation_efficiency.get("annotations_per_hour"):
                self.annotation_efficiency.set(annotation_efficiency["annotations_per_hour"])
            
            if annotation_efficiency.get("quality_score"):
                self.annotation_quality_score.set(annotation_efficiency["quality_score"])
            
            # User activity
            user_activity = business_summary.get("user_activity", {})
            if user_activity.get("active_users"):
                self.active_users.set(user_activity["active_users"])
                
        except Exception as e:
            logger.warning(f"Failed to update business metrics: {e}")
    
    def _update_ai_metrics(self):
        """Update AI model metrics."""
        try:
            # Import here to avoid circular imports
            from src.system.business_metrics import business_metrics_collector
            
            business_summary = business_metrics_collector.get_business_summary()
            ai_models = business_summary.get("ai_models", {})
            
            for model_name, model_data in ai_models.items():
                if model_data.get("confidence_avg") is not None:
                    self.ai_confidence_score.labels(model=model_name).set(model_data["confidence_avg"])
                
        except Exception as e:
            logger.warning(f"Failed to update AI metrics: {e}")
    
    def _update_project_metrics(self):
        """Update project metrics."""
        try:
            # Import here to avoid circular imports
            from src.system.business_metrics import business_metrics_collector
            
            business_summary = business_metrics_collector.get_business_summary()
            projects = business_summary.get("projects", {})
            
            for project_id, project_data in projects.items():
                if project_data.get("completion_percentage") is not None:
                    self.project_completion_percentage.labels(project_id=project_id).set(
                        project_data["completion_percentage"]
                    )
                
                if project_data.get("total_tasks") is not None:
                    self.project_tasks_total.labels(project_id=project_id, status="total").set(
                        project_data["total_tasks"]
                    )
                
                if project_data.get("completed_tasks") is not None:
                    self.project_tasks_total.labels(project_id=project_id, status="completed").set(
                        project_data["completed_tasks"]
                    )
                
        except Exception as e:
            logger.warning(f"Failed to update project metrics: {e}")
    
    def track_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Track HTTP request metrics."""
        self.http_requests_total.labels(method=method, endpoint=endpoint, status=str(status_code)).inc()
        self.http_request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def track_database_query(self, query_type: str, success: bool, duration: float):
        """Track database query metrics."""
        self.database_queries_total.labels(type=query_type, success=str(success)).inc()
        self.database_query_duration.labels(type=query_type).observe(duration)
    
    def track_ai_inference(self, model_name: str, success: bool, duration: float):
        """Track AI inference metrics."""
        self.ai_inferences_total.labels(model=model_name, success=str(success)).inc()
        self.ai_inference_duration.labels(model=model_name).observe(duration)
    
    def track_user_session(self):
        """Track user session start."""
        self.user_sessions_total.inc()
    
    def track_quality_issue(self, severity: str, status: str):
        """Track quality issue."""
        self.quality_issues_total.labels(severity=severity, status=status).inc()
    
    def track_billing(self, tenant_id: str, cost: float, annotations: int):
        """Track billing metrics."""
        self.billing_cost_total.labels(tenant_id=tenant_id).inc(cost)
        self.billing_annotations_total.labels(tenant_id=tenant_id).inc(annotations)
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        self.update_metrics()
        return generate_latest(self.registry).decode('utf-8')
    
    def get_metrics_response(self) -> Response:
        """Get metrics as FastAPI Response."""
        metrics_data = self.get_metrics()
        return Response(
            content=metrics_data,
            media_type=CONTENT_TYPE_LATEST
        )


# Global Prometheus exporter instance
prometheus_exporter = PrometheusExporter()