"""
Business Metrics Collection and Analysis for SuperInsight Platform.

Provides comprehensive business intelligence metrics including:
- Annotation efficiency and quality trends
- User activity and engagement statistics
- AI model performance monitoring
- Project progress and completion metrics
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta, date
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from decimal import Decimal
import statistics

from src.system.monitoring import MetricsCollector, metrics_collector
from src.billing.service import BillingSystem
from src.billing.analytics import BillingAnalytics


logger = logging.getLogger(__name__)


@dataclass
class AnnotationEfficiencyMetrics:
    """Annotation efficiency metrics container."""
    annotations_per_hour: float
    average_annotation_time: float
    quality_score: float
    completion_rate: float
    revision_rate: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class UserActivityMetrics:
    """User activity metrics container."""
    active_users_count: int
    new_users_count: int
    returning_users_count: int
    session_duration_avg: float
    actions_per_session: float
    peak_concurrent_users: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class AIModelMetrics:
    """AI model performance metrics container."""
    model_name: str
    inference_count: int
    average_inference_time: float
    success_rate: float
    confidence_score_avg: float
    accuracy_score: float
    error_rate: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class ProjectProgressMetrics:
    """Project progress metrics container."""
    project_id: str
    total_tasks: int
    completed_tasks: int
    in_progress_tasks: int
    pending_tasks: int
    completion_percentage: float
    estimated_completion_date: Optional[str]
    average_task_duration: float
    timestamp: float = field(default_factory=time.time)


class BusinessMetricsCollector:
    """
    Business metrics collection and analysis system.
    
    Collects and analyzes business-specific metrics for operational insights.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.billing_system = BillingSystem()
        self.billing_analytics = BillingAnalytics(self.billing_system)
        
        # Metric storage
        self.annotation_metrics: deque = deque(maxlen=1000)
        self.user_activity_metrics: deque = deque(maxlen=1000)
        self.ai_model_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.project_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Active sessions tracking
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.user_actions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Collection intervals
        self.collection_interval = 60  # seconds
        self.is_collecting = False
        self._collection_task: Optional[asyncio.Task] = None
    
    async def start_collection(self):
        """Start automatic business metrics collection."""
        if self.is_collecting:
            logger.warning("Business metrics collection is already running")
            return
        
        self.is_collecting = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Started business metrics collection")
    
    async def stop_collection(self):
        """Stop automatic business metrics collection."""
        self.is_collecting = False
        
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped business metrics collection")
    
    async def _collection_loop(self):
        """Main business metrics collection loop."""
        while self.is_collecting:
            try:
                await self._collect_annotation_metrics()
                await self._collect_user_activity_metrics()
                await self._collect_ai_model_metrics()
                await self._collect_project_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in business metrics collection: {e}")
                await asyncio.sleep(10)  # Short delay before retrying
    
    async def _collect_annotation_metrics(self):
        """Collect annotation efficiency and quality metrics."""
        try:
            # Get recent billing data for efficiency calculation
            end_date = date.today()
            start_date = end_date - timedelta(days=1)  # Last 24 hours
            
            # Calculate metrics for all active tenants
            tenants = await self._get_active_tenants()
            
            total_annotations = 0
            total_time_spent = 0
            total_quality_score = 0
            total_completed_tasks = 0
            total_revised_tasks = 0
            
            for tenant_id in tenants:
                try:
                    report = self.billing_system.generate_report(tenant_id, start_date, end_date)
                    
                    total_annotations += report.total_annotations
                    total_time_spent += report.total_time_spent
                    
                    # Get quality scores from recent tasks
                    quality_scores = await self._get_recent_quality_scores(tenant_id)
                    if quality_scores:
                        total_quality_score += sum(quality_scores)
                        total_completed_tasks += len(quality_scores)
                    
                    # Get revision counts
                    revision_count = await self._get_revision_count(tenant_id, start_date, end_date)
                    total_revised_tasks += revision_count
                    
                except Exception as e:
                    logger.warning(f"Failed to collect metrics for tenant {tenant_id}: {e}")
                    continue
            
            # Calculate efficiency metrics
            if total_time_spent > 0:
                annotations_per_hour = (total_annotations * 3600) / total_time_spent
                average_annotation_time = total_time_spent / total_annotations if total_annotations > 0 else 0
            else:
                annotations_per_hour = 0
                average_annotation_time = 0
            
            quality_score = (total_quality_score / total_completed_tasks) if total_completed_tasks > 0 else 0
            completion_rate = total_completed_tasks / (total_completed_tasks + total_revised_tasks) if (total_completed_tasks + total_revised_tasks) > 0 else 0
            revision_rate = total_revised_tasks / total_completed_tasks if total_completed_tasks > 0 else 0
            
            # Store metrics
            efficiency_metrics = AnnotationEfficiencyMetrics(
                annotations_per_hour=annotations_per_hour,
                average_annotation_time=average_annotation_time,
                quality_score=quality_score,
                completion_rate=completion_rate,
                revision_rate=revision_rate
            )
            
            self.annotation_metrics.append(efficiency_metrics)
            
            # Record to metrics collector
            self.metrics.record_metric("business.annotation.efficiency.annotations_per_hour", annotations_per_hour)
            self.metrics.record_metric("business.annotation.efficiency.average_time", average_annotation_time)
            self.metrics.record_metric("business.annotation.quality.score", quality_score)
            self.metrics.record_metric("business.annotation.completion_rate", completion_rate)
            self.metrics.record_metric("business.annotation.revision_rate", revision_rate)
            
        except Exception as e:
            logger.error(f"Failed to collect annotation metrics: {e}")
    
    async def _collect_user_activity_metrics(self):
        """Collect user activity and engagement metrics."""
        try:
            # Calculate user activity from active sessions
            current_time = time.time()
            active_users = 0
            session_durations = []
            actions_counts = []
            
            # Clean up old sessions and calculate metrics
            expired_sessions = []
            for session_id, session_data in self.active_sessions.items():
                session_age = current_time - session_data.get("start_time", current_time)
                
                if session_age > 3600:  # 1 hour timeout
                    expired_sessions.append(session_id)
                    session_durations.append(session_age)
                    
                    user_id = session_data.get("user_id")
                    if user_id and user_id in self.user_actions:
                        actions_counts.append(len(self.user_actions[user_id]))
                        # Clean up old actions
                        del self.user_actions[user_id]
                else:
                    active_users += 1
            
            # Remove expired sessions
            for session_id in expired_sessions:
                del self.active_sessions[session_id]
            
            # Calculate metrics
            avg_session_duration = statistics.mean(session_durations) if session_durations else 0
            avg_actions_per_session = statistics.mean(actions_counts) if actions_counts else 0
            
            # Get new and returning users from last 24 hours
            new_users, returning_users = await self._get_user_activity_stats()
            
            # Store metrics
            activity_metrics = UserActivityMetrics(
                active_users_count=active_users,
                new_users_count=new_users,
                returning_users_count=returning_users,
                session_duration_avg=avg_session_duration,
                actions_per_session=avg_actions_per_session,
                peak_concurrent_users=max(active_users, getattr(self, '_peak_users', 0))
            )
            
            self.user_activity_metrics.append(activity_metrics)
            self._peak_users = max(active_users, getattr(self, '_peak_users', 0))
            
            # Record to metrics collector
            self.metrics.record_metric("business.users.active_count", active_users)
            self.metrics.record_metric("business.users.new_count", new_users)
            self.metrics.record_metric("business.users.returning_count", returning_users)
            self.metrics.record_metric("business.users.session_duration_avg", avg_session_duration)
            self.metrics.record_metric("business.users.actions_per_session", avg_actions_per_session)
            self.metrics.record_metric("business.users.peak_concurrent", self._peak_users)
            
        except Exception as e:
            logger.error(f"Failed to collect user activity metrics: {e}")
    
    async def _collect_ai_model_metrics(self):
        """Collect AI model performance metrics."""
        try:
            # Get AI inference metrics from the metrics collector
            ai_metrics = self.metrics.get_all_metrics_summary()
            
            for metric_name, summary in ai_metrics.items():
                if metric_name.startswith("ai.inference"):
                    # Extract model name from tags or metric name
                    model_name = "unknown"
                    if "model" in metric_name:
                        parts = metric_name.split(".")
                        for part in parts:
                            if part not in ["ai", "inference", "duration", "total"]:
                                model_name = part
                                break
                    
                    # Get related metrics
                    inference_count = ai_metrics.get(f"ai.inferences.total.{model_name}", {}).get("latest", 0) or 0
                    avg_inference_time = summary.get("avg", 0)
                    success_rate = self._calculate_ai_success_rate(model_name, ai_metrics)
                    confidence_avg = self._get_ai_confidence_average(model_name)
                    accuracy_score = self._get_ai_accuracy_score(model_name)
                    error_rate = 1.0 - success_rate
                    
                    # Store metrics
                    model_metrics = AIModelMetrics(
                        model_name=model_name,
                        inference_count=int(inference_count),
                        average_inference_time=avg_inference_time,
                        success_rate=success_rate,
                        confidence_score_avg=confidence_avg,
                        accuracy_score=accuracy_score,
                        error_rate=error_rate
                    )
                    
                    self.ai_model_metrics[model_name].append(model_metrics)
                    
                    # Record to metrics collector
                    self.metrics.record_metric(f"business.ai.{model_name}.inference_count", inference_count)
                    self.metrics.record_metric(f"business.ai.{model_name}.avg_inference_time", avg_inference_time)
                    self.metrics.record_metric(f"business.ai.{model_name}.success_rate", success_rate)
                    self.metrics.record_metric(f"business.ai.{model_name}.confidence_avg", confidence_avg)
                    self.metrics.record_metric(f"business.ai.{model_name}.accuracy_score", accuracy_score)
                    self.metrics.record_metric(f"business.ai.{model_name}.error_rate", error_rate)
            
        except Exception as e:
            logger.error(f"Failed to collect AI model metrics: {e}")
    
    async def _collect_project_metrics(self):
        """Collect project progress and completion metrics."""
        try:
            # Get active projects
            projects = await self._get_active_projects()
            
            for project_id in projects:
                try:
                    # Get project task statistics
                    task_stats = await self._get_project_task_stats(project_id)
                    
                    total_tasks = task_stats.get("total", 0)
                    completed_tasks = task_stats.get("completed", 0)
                    in_progress_tasks = task_stats.get("in_progress", 0)
                    pending_tasks = task_stats.get("pending", 0)
                    
                    completion_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
                    
                    # Estimate completion date
                    avg_task_duration = await self._get_average_task_duration(project_id)
                    estimated_completion = None
                    if avg_task_duration > 0 and (in_progress_tasks + pending_tasks) > 0:
                        remaining_time = avg_task_duration * (in_progress_tasks + pending_tasks)
                        completion_date = datetime.now() + timedelta(seconds=remaining_time)
                        estimated_completion = completion_date.isoformat()
                    
                    # Store metrics
                    project_metrics = ProjectProgressMetrics(
                        project_id=project_id,
                        total_tasks=total_tasks,
                        completed_tasks=completed_tasks,
                        in_progress_tasks=in_progress_tasks,
                        pending_tasks=pending_tasks,
                        completion_percentage=completion_percentage,
                        estimated_completion_date=estimated_completion,
                        average_task_duration=avg_task_duration
                    )
                    
                    self.project_metrics[project_id].append(project_metrics)
                    
                    # Record to metrics collector
                    self.metrics.record_metric(f"business.project.{project_id}.completion_percentage", completion_percentage)
                    self.metrics.record_metric(f"business.project.{project_id}.total_tasks", total_tasks)
                    self.metrics.record_metric(f"business.project.{project_id}.completed_tasks", completed_tasks)
                    self.metrics.record_metric(f"business.project.{project_id}.avg_task_duration", avg_task_duration)
                    
                except Exception as e:
                    logger.warning(f"Failed to collect metrics for project {project_id}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Failed to collect project metrics: {e}")
    
    # User session tracking methods
    def track_user_session_start(self, user_id: str, session_id: str):
        """Track the start of a user session."""
        self.active_sessions[session_id] = {
            "user_id": user_id,
            "start_time": time.time()
        }
        
        # Initialize user actions tracking
        if user_id not in self.user_actions:
            self.user_actions[user_id] = []
    
    def track_user_action(self, user_id: str, action_type: str, details: Optional[Dict[str, Any]] = None):
        """Track a user action."""
        action = {
            "type": action_type,
            "timestamp": time.time(),
            "details": details or {}
        }
        
        self.user_actions[user_id].append(action)
        
        # Keep only recent actions (last 24 hours)
        cutoff_time = time.time() - 86400  # 24 hours
        self.user_actions[user_id] = [
            a for a in self.user_actions[user_id] 
            if a["timestamp"] > cutoff_time
        ]
    
    def track_user_session_end(self, session_id: str):
        """Track the end of a user session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
    
    # AI model tracking methods
    def track_ai_inference(self, model_name: str, duration: float, success: bool, confidence: Optional[float] = None):
        """Track an AI inference operation."""
        self.metrics.record_timing(f"ai.inference.{model_name}", duration)
        self.metrics.increment_counter(f"ai.inferences.total.{model_name}")
        
        if success:
            self.metrics.increment_counter(f"ai.inferences.success.{model_name}")
        else:
            self.metrics.increment_counter(f"ai.inferences.error.{model_name}")
        
        if confidence is not None:
            self.metrics.record_metric(f"ai.confidence.{model_name}", confidence)
    
    def track_annotation_quality(self, project_id: str, task_id: str, quality_score: float):
        """Track annotation quality score."""
        self.metrics.record_metric(f"annotation.quality.{project_id}", quality_score)
        self.metrics.record_metric("annotation.quality.overall", quality_score)
    
    # Getter methods for metrics
    def get_annotation_efficiency_trends(self, hours: int = 24) -> List[AnnotationEfficiencyMetrics]:
        """Get annotation efficiency trends for the specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        return [m for m in self.annotation_metrics if m.timestamp > cutoff_time]
    
    def get_user_activity_trends(self, hours: int = 24) -> List[UserActivityMetrics]:
        """Get user activity trends for the specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        return [m for m in self.user_activity_metrics if m.timestamp > cutoff_time]
    
    def get_ai_model_performance(self, model_name: str, hours: int = 24) -> List[AIModelMetrics]:
        """Get AI model performance metrics for the specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        if model_name in self.ai_model_metrics:
            return [m for m in self.ai_model_metrics[model_name] if m.timestamp > cutoff_time]
        return []
    
    def get_project_progress(self, project_id: str, hours: int = 24) -> List[ProjectProgressMetrics]:
        """Get project progress metrics for the specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        if project_id in self.project_metrics:
            return [m for m in self.project_metrics[project_id] if m.timestamp > cutoff_time]
        return []
    
    def get_business_summary(self) -> Dict[str, Any]:
        """Get comprehensive business metrics summary."""
        current_time = time.time()
        
        # Get latest metrics
        latest_annotation = self.annotation_metrics[-1] if self.annotation_metrics else None
        latest_activity = self.user_activity_metrics[-1] if self.user_activity_metrics else None
        
        # Calculate AI model summary
        ai_summary = {}
        for model_name, metrics_list in self.ai_model_metrics.items():
            if metrics_list:
                latest_ai = metrics_list[-1]
                ai_summary[model_name] = {
                    "inference_count": latest_ai.inference_count,
                    "success_rate": latest_ai.success_rate,
                    "avg_inference_time": latest_ai.average_inference_time,
                    "confidence_avg": latest_ai.confidence_score_avg
                }
        
        # Calculate project summary
        project_summary = {}
        for project_id, metrics_list in self.project_metrics.items():
            if metrics_list:
                latest_project = metrics_list[-1]
                project_summary[project_id] = {
                    "completion_percentage": latest_project.completion_percentage,
                    "total_tasks": latest_project.total_tasks,
                    "completed_tasks": latest_project.completed_tasks,
                    "estimated_completion": latest_project.estimated_completion_date
                }
        
        return {
            "timestamp": current_time,
            "annotation_efficiency": {
                "annotations_per_hour": latest_annotation.annotations_per_hour if latest_annotation else 0,
                "quality_score": latest_annotation.quality_score if latest_annotation else 0,
                "completion_rate": latest_annotation.completion_rate if latest_annotation else 0
            },
            "user_activity": {
                "active_users": latest_activity.active_users_count if latest_activity else 0,
                "new_users": latest_activity.new_users_count if latest_activity else 0,
                "peak_concurrent": latest_activity.peak_concurrent_users if latest_activity else 0
            },
            "ai_models": ai_summary,
            "projects": project_summary
        }
    
    # Helper methods (these would need to be implemented based on your database schema)
    async def _get_active_tenants(self) -> List[str]:
        """Get list of active tenant IDs."""
        # This would query your database for active tenants
        # For now, return a placeholder
        return ["default_tenant"]
    
    async def _get_recent_quality_scores(self, tenant_id: str) -> List[float]:
        """Get recent quality scores for a tenant."""
        # This would query your database for recent quality scores
        # For now, return placeholder data
        return [0.85, 0.92, 0.78, 0.88, 0.91]
    
    async def _get_revision_count(self, tenant_id: str, start_date: date, end_date: date) -> int:
        """Get count of revised tasks for a tenant in the date range."""
        # This would query your database for revision counts
        # For now, return a placeholder
        return 5
    
    async def _get_user_activity_stats(self) -> Tuple[int, int]:
        """Get new and returning user counts for the last 24 hours."""
        # This would query your database for user activity
        # For now, return placeholder data
        return (3, 12)  # (new_users, returning_users)
    
    def _calculate_ai_success_rate(self, model_name: str, ai_metrics: Dict[str, Any]) -> float:
        """Calculate AI model success rate."""
        success_count = ai_metrics.get(f"ai.inferences.success.{model_name}", {}).get("latest", 0) or 0
        total_count = ai_metrics.get(f"ai.inferences.total.{model_name}", {}).get("latest", 0) or 0
        
        return (success_count / total_count) if total_count > 0 else 0.0
    
    def _get_ai_confidence_average(self, model_name: str) -> float:
        """Get average confidence score for an AI model."""
        confidence_summary = self.metrics.get_metric_summary(f"ai.confidence.{model_name}")
        return confidence_summary.get("avg", 0.0) or 0.0
    
    def _get_ai_accuracy_score(self, model_name: str) -> float:
        """Get accuracy score for an AI model."""
        # This would be calculated based on ground truth comparisons
        # For now, return a placeholder based on success rate
        return self._calculate_ai_success_rate(model_name, self.metrics.get_all_metrics_summary())
    
    async def _get_active_projects(self) -> List[str]:
        """Get list of active project IDs."""
        # This would query your database for active projects
        # For now, return placeholder data
        return ["project_1", "project_2", "project_3"]
    
    async def _get_project_task_stats(self, project_id: str) -> Dict[str, int]:
        """Get task statistics for a project."""
        # This would query your database for project task stats
        # For now, return placeholder data
        return {
            "total": 100,
            "completed": 65,
            "in_progress": 20,
            "pending": 15
        }
    
    async def _get_average_task_duration(self, project_id: str) -> float:
        """Get average task duration for a project in seconds."""
        # This would calculate based on historical task completion times
        # For now, return placeholder data (2 hours average)
        return 7200.0


# Global business metrics collector instance
business_metrics_collector = BusinessMetricsCollector(metrics_collector)