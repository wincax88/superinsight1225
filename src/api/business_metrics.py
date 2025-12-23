"""
Business Metrics API for SuperInsight Platform.

Provides endpoints for accessing business intelligence metrics including:
- Annotation efficiency and quality trends
- User activity and engagement statistics  
- AI model performance monitoring
- Project progress and completion metrics
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse

from src.system.business_metrics import business_metrics_collector
from src.system.monitoring import metrics_collector, performance_monitor
from src.security.middleware import require_permission, Permission, User


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/business-metrics", tags=["business-metrics"])


@router.get("/summary")
async def get_business_summary(
    current_user: User = Depends(require_permission(Permission.VIEW_ANALYTICS))
) -> Dict[str, Any]:
    """
    Get comprehensive business metrics summary.
    
    Provides high-level overview of all business metrics including
    annotation efficiency, user activity, AI performance, and project progress.
    """
    try:
        summary = business_metrics_collector.get_business_summary()
        
        # Add system metrics
        system_summary = metrics_collector.get_all_metrics_summary()
        performance_summary = performance_monitor.get_performance_summary()
        
        return {
            "business_metrics": summary,
            "system_performance": {
                "active_requests": performance_summary.get("active_requests", 0),
                "avg_request_duration": performance_summary.get("avg_request_duration", {}),
                "database_performance": performance_summary.get("database_performance", {}),
                "ai_performance": performance_summary.get("ai_performance", {})
            },
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get business summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve business metrics summary: {str(e)}"
        )


@router.get("/annotation-efficiency")
async def get_annotation_efficiency_trends(
    hours: int = Query(24, ge=1, le=168, description="Number of hours to analyze (1-168)"),
    current_user: User = Depends(require_permission(Permission.VIEW_ANALYTICS))
) -> Dict[str, Any]:
    """
    Get annotation efficiency and quality trends.
    
    Provides detailed metrics on annotation productivity, quality scores,
    completion rates, and revision patterns over the specified time period.
    """
    try:
        trends = business_metrics_collector.get_annotation_efficiency_trends(hours)
        
        if not trends:
            return {
                "period_hours": hours,
                "data_points": 0,
                "trends": [],
                "summary": {
                    "avg_annotations_per_hour": 0,
                    "avg_quality_score": 0,
                    "avg_completion_rate": 0,
                    "avg_revision_rate": 0
                },
                "message": "No annotation efficiency data available for the specified period"
            }
        
        # Calculate summary statistics
        annotations_per_hour = [t.annotations_per_hour for t in trends]
        quality_scores = [t.quality_score for t in trends]
        completion_rates = [t.completion_rate for t in trends]
        revision_rates = [t.revision_rate for t in trends]
        
        return {
            "period_hours": hours,
            "data_points": len(trends),
            "trends": [
                {
                    "timestamp": t.timestamp,
                    "datetime": datetime.fromtimestamp(t.timestamp).isoformat(),
                    "annotations_per_hour": t.annotations_per_hour,
                    "average_annotation_time": t.average_annotation_time,
                    "quality_score": t.quality_score,
                    "completion_rate": t.completion_rate,
                    "revision_rate": t.revision_rate
                }
                for t in trends
            ],
            "summary": {
                "avg_annotations_per_hour": sum(annotations_per_hour) / len(annotations_per_hour),
                "avg_quality_score": sum(quality_scores) / len(quality_scores),
                "avg_completion_rate": sum(completion_rates) / len(completion_rates),
                "avg_revision_rate": sum(revision_rates) / len(revision_rates),
                "peak_annotations_per_hour": max(annotations_per_hour),
                "best_quality_score": max(quality_scores),
                "trend_direction": _calculate_trend_direction(annotations_per_hour)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get annotation efficiency trends: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve annotation efficiency trends: {str(e)}"
        )


@router.get("/user-activity")
async def get_user_activity_trends(
    hours: int = Query(24, ge=1, le=168, description="Number of hours to analyze (1-168)"),
    current_user: User = Depends(require_permission(Permission.VIEW_ANALYTICS))
) -> Dict[str, Any]:
    """
    Get user activity and engagement trends.
    
    Provides metrics on active users, session patterns, user engagement,
    and activity trends over the specified time period.
    """
    try:
        trends = business_metrics_collector.get_user_activity_trends(hours)
        
        if not trends:
            return {
                "period_hours": hours,
                "data_points": 0,
                "trends": [],
                "summary": {
                    "avg_active_users": 0,
                    "total_new_users": 0,
                    "avg_session_duration": 0,
                    "peak_concurrent_users": 0
                },
                "message": "No user activity data available for the specified period"
            }
        
        # Calculate summary statistics
        active_users = [t.active_users_count for t in trends]
        new_users = [t.new_users_count for t in trends]
        session_durations = [t.session_duration_avg for t in trends]
        actions_per_session = [t.actions_per_session for t in trends]
        
        return {
            "period_hours": hours,
            "data_points": len(trends),
            "trends": [
                {
                    "timestamp": t.timestamp,
                    "datetime": datetime.fromtimestamp(t.timestamp).isoformat(),
                    "active_users_count": t.active_users_count,
                    "new_users_count": t.new_users_count,
                    "returning_users_count": t.returning_users_count,
                    "session_duration_avg": t.session_duration_avg,
                    "actions_per_session": t.actions_per_session,
                    "peak_concurrent_users": t.peak_concurrent_users
                }
                for t in trends
            ],
            "summary": {
                "avg_active_users": sum(active_users) / len(active_users),
                "total_new_users": sum(new_users),
                "avg_session_duration": sum(session_durations) / len(session_durations),
                "avg_actions_per_session": sum(actions_per_session) / len(actions_per_session),
                "peak_concurrent_users": max([t.peak_concurrent_users for t in trends]),
                "user_growth_trend": _calculate_trend_direction(new_users),
                "engagement_trend": _calculate_trend_direction(actions_per_session)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get user activity trends: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve user activity trends: {str(e)}"
        )


@router.get("/ai-models")
async def get_ai_model_performance(
    model_name: Optional[str] = Query(None, description="Specific model name to analyze"),
    hours: int = Query(24, ge=1, le=168, description="Number of hours to analyze (1-168)"),
    current_user: User = Depends(require_permission(Permission.VIEW_ANALYTICS))
) -> Dict[str, Any]:
    """
    Get AI model performance metrics.
    
    Provides detailed performance metrics for AI models including
    inference times, success rates, confidence scores, and accuracy metrics.
    """
    try:
        if model_name:
            # Get metrics for specific model
            model_metrics = business_metrics_collector.get_ai_model_performance(model_name, hours)
            
            if not model_metrics:
                return {
                    "model_name": model_name,
                    "period_hours": hours,
                    "data_points": 0,
                    "metrics": [],
                    "summary": {},
                    "message": f"No performance data available for model '{model_name}' in the specified period"
                }
            
            # Calculate summary statistics
            inference_counts = [m.inference_count for m in model_metrics]
            inference_times = [m.average_inference_time for m in model_metrics]
            success_rates = [m.success_rate for m in model_metrics]
            confidence_scores = [m.confidence_score_avg for m in model_metrics]
            
            return {
                "model_name": model_name,
                "period_hours": hours,
                "data_points": len(model_metrics),
                "metrics": [
                    {
                        "timestamp": m.timestamp,
                        "datetime": datetime.fromtimestamp(m.timestamp).isoformat(),
                        "inference_count": m.inference_count,
                        "average_inference_time": m.average_inference_time,
                        "success_rate": m.success_rate,
                        "confidence_score_avg": m.confidence_score_avg,
                        "accuracy_score": m.accuracy_score,
                        "error_rate": m.error_rate
                    }
                    for m in model_metrics
                ],
                "summary": {
                    "total_inferences": sum(inference_counts),
                    "avg_inference_time": sum(inference_times) / len(inference_times),
                    "avg_success_rate": sum(success_rates) / len(success_rates),
                    "avg_confidence_score": sum(confidence_scores) / len(confidence_scores),
                    "performance_trend": _calculate_trend_direction([1/t for t in inference_times if t > 0])  # Inverse for performance
                }
            }
        else:
            # Get metrics for all models
            all_models = {}
            
            for model in business_metrics_collector.ai_model_metrics.keys():
                model_metrics = business_metrics_collector.get_ai_model_performance(model, hours)
                
                if model_metrics:
                    latest_metric = model_metrics[-1]
                    all_models[model] = {
                        "inference_count": latest_metric.inference_count,
                        "average_inference_time": latest_metric.average_inference_time,
                        "success_rate": latest_metric.success_rate,
                        "confidence_score_avg": latest_metric.confidence_score_avg,
                        "accuracy_score": latest_metric.accuracy_score,
                        "error_rate": latest_metric.error_rate,
                        "last_updated": datetime.fromtimestamp(latest_metric.timestamp).isoformat()
                    }
            
            return {
                "period_hours": hours,
                "models": all_models,
                "model_count": len(all_models),
                "summary": {
                    "total_models": len(all_models),
                    "active_models": len([m for m in all_models.values() if m["inference_count"] > 0]),
                    "avg_success_rate": sum([m["success_rate"] for m in all_models.values()]) / len(all_models) if all_models else 0
                }
            }
        
    except Exception as e:
        logger.error(f"Failed to get AI model performance: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve AI model performance: {str(e)}"
        )


@router.get("/projects")
async def get_project_progress(
    project_id: Optional[str] = Query(None, description="Specific project ID to analyze"),
    hours: int = Query(24, ge=1, le=168, description="Number of hours to analyze (1-168)"),
    current_user: User = Depends(require_permission(Permission.VIEW_ANALYTICS))
) -> Dict[str, Any]:
    """
    Get project progress and completion metrics.
    
    Provides detailed progress tracking for projects including
    task completion rates, timeline estimates, and productivity metrics.
    """
    try:
        if project_id:
            # Get metrics for specific project
            project_metrics = business_metrics_collector.get_project_progress(project_id, hours)
            
            if not project_metrics:
                return {
                    "project_id": project_id,
                    "period_hours": hours,
                    "data_points": 0,
                    "progress": [],
                    "summary": {},
                    "message": f"No progress data available for project '{project_id}' in the specified period"
                }
            
            # Get latest metrics
            latest_metric = project_metrics[-1]
            
            # Calculate progress trend
            completion_percentages = [m.completion_percentage for m in project_metrics]
            
            return {
                "project_id": project_id,
                "period_hours": hours,
                "data_points": len(project_metrics),
                "progress": [
                    {
                        "timestamp": m.timestamp,
                        "datetime": datetime.fromtimestamp(m.timestamp).isoformat(),
                        "total_tasks": m.total_tasks,
                        "completed_tasks": m.completed_tasks,
                        "in_progress_tasks": m.in_progress_tasks,
                        "pending_tasks": m.pending_tasks,
                        "completion_percentage": m.completion_percentage,
                        "estimated_completion_date": m.estimated_completion_date,
                        "average_task_duration": m.average_task_duration
                    }
                    for m in project_metrics
                ],
                "current_status": {
                    "completion_percentage": latest_metric.completion_percentage,
                    "total_tasks": latest_metric.total_tasks,
                    "completed_tasks": latest_metric.completed_tasks,
                    "remaining_tasks": latest_metric.in_progress_tasks + latest_metric.pending_tasks,
                    "estimated_completion": latest_metric.estimated_completion_date,
                    "average_task_duration_hours": latest_metric.average_task_duration / 3600
                },
                "summary": {
                    "progress_trend": _calculate_trend_direction(completion_percentages),
                    "completion_velocity": _calculate_completion_velocity(project_metrics),
                    "on_track": latest_metric.completion_percentage >= _expected_completion_percentage(project_metrics)
                }
            }
        else:
            # Get metrics for all projects
            all_projects = {}
            
            for project in business_metrics_collector.project_metrics.keys():
                project_metrics = business_metrics_collector.get_project_progress(project, hours)
                
                if project_metrics:
                    latest_metric = project_metrics[-1]
                    all_projects[project] = {
                        "completion_percentage": latest_metric.completion_percentage,
                        "total_tasks": latest_metric.total_tasks,
                        "completed_tasks": latest_metric.completed_tasks,
                        "remaining_tasks": latest_metric.in_progress_tasks + latest_metric.pending_tasks,
                        "estimated_completion": latest_metric.estimated_completion_date,
                        "last_updated": datetime.fromtimestamp(latest_metric.timestamp).isoformat()
                    }
            
            return {
                "period_hours": hours,
                "projects": all_projects,
                "project_count": len(all_projects),
                "summary": {
                    "total_projects": len(all_projects),
                    "completed_projects": len([p for p in all_projects.values() if p["completion_percentage"] >= 100]),
                    "avg_completion_percentage": sum([p["completion_percentage"] for p in all_projects.values()]) / len(all_projects) if all_projects else 0,
                    "projects_on_track": len([p for p in all_projects.values() if p["completion_percentage"] >= 50])  # Simple heuristic
                }
            }
        
    except Exception as e:
        logger.error(f"Failed to get project progress: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve project progress: {str(e)}"
        )


@router.post("/track/user-session")
async def track_user_session(
    action: str = Query(..., description="Action: 'start' or 'end'"),
    user_id: str = Query(..., description="User ID"),
    session_id: str = Query(..., description="Session ID"),
    current_user: User = Depends(require_permission(Permission.ANNOTATE))
) -> JSONResponse:
    """
    Track user session events for activity monitoring.
    
    Records session start/end events to calculate user engagement metrics.
    """
    try:
        if action == "start":
            business_metrics_collector.track_user_session_start(user_id, session_id)
            message = f"Started tracking session {session_id} for user {user_id}"
        elif action == "end":
            business_metrics_collector.track_user_session_end(session_id)
            message = f"Ended tracking session {session_id}"
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid action. Must be 'start' or 'end'"
            )
        
        return JSONResponse(
            status_code=200,
            content={"message": message, "timestamp": datetime.now().isoformat()}
        )
        
    except Exception as e:
        logger.error(f"Failed to track user session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to track user session: {str(e)}"
        )


@router.post("/track/user-action")
async def track_user_action(
    user_id: str = Query(..., description="User ID"),
    action_type: str = Query(..., description="Type of action performed"),
    details: Optional[Dict[str, Any]] = None,
    current_user: User = Depends(require_permission(Permission.ANNOTATE))
) -> JSONResponse:
    """
    Track user actions for engagement monitoring.
    
    Records user actions to calculate engagement and productivity metrics.
    """
    try:
        business_metrics_collector.track_user_action(user_id, action_type, details)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Tracked action '{action_type}' for user {user_id}",
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to track user action: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to track user action: {str(e)}"
        )


@router.post("/track/ai-inference")
async def track_ai_inference(
    model_name: str = Query(..., description="AI model name"),
    duration: float = Query(..., description="Inference duration in seconds"),
    success: bool = Query(..., description="Whether inference was successful"),
    confidence: Optional[float] = Query(None, description="Confidence score (0-1)"),
    current_user: User = Depends(require_permission(Permission.USE_AI))
) -> JSONResponse:
    """
    Track AI inference operations for performance monitoring.
    
    Records AI model usage and performance metrics.
    """
    try:
        business_metrics_collector.track_ai_inference(model_name, duration, success, confidence)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Tracked AI inference for model '{model_name}'",
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to track AI inference: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to track AI inference: {str(e)}"
        )


@router.post("/track/annotation-quality")
async def track_annotation_quality(
    project_id: str = Query(..., description="Project ID"),
    task_id: str = Query(..., description="Task ID"),
    quality_score: float = Query(..., ge=0.0, le=1.0, description="Quality score (0-1)"),
    current_user: User = Depends(require_permission(Permission.QUALITY_CONTROL))
) -> JSONResponse:
    """
    Track annotation quality scores for quality monitoring.
    
    Records quality assessments to calculate quality trends and metrics.
    """
    try:
        business_metrics_collector.track_annotation_quality(project_id, task_id, quality_score)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Tracked quality score {quality_score} for task {task_id}",
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to track annotation quality: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to track annotation quality: {str(e)}"
        )


@router.post("/start-collection")
async def start_metrics_collection(
    current_user: User = Depends(require_permission(Permission.ADMIN))
) -> JSONResponse:
    """
    Start automatic business metrics collection.
    
    Begins background collection of business metrics. Admin permission required.
    """
    try:
        await business_metrics_collector.start_collection()
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Business metrics collection started",
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to start metrics collection: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start metrics collection: {str(e)}"
        )


@router.post("/stop-collection")
async def stop_metrics_collection(
    current_user: User = Depends(require_permission(Permission.ADMIN))
) -> JSONResponse:
    """
    Stop automatic business metrics collection.
    
    Stops background collection of business metrics. Admin permission required.
    """
    try:
        await business_metrics_collector.stop_collection()
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Business metrics collection stopped",
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to stop metrics collection: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stop metrics collection: {str(e)}"
        )


# Helper functions
def _calculate_trend_direction(values: List[float]) -> str:
    """Calculate trend direction from a list of values."""
    if len(values) < 2:
        return "stable"
    
    # Simple linear trend calculation
    first_half = values[:len(values)//2]
    second_half = values[len(values)//2:]
    
    if not first_half or not second_half:
        return "stable"
    
    first_avg = sum(first_half) / len(first_half)
    second_avg = sum(second_half) / len(second_half)
    
    if second_avg > first_avg * 1.05:  # 5% threshold
        return "increasing"
    elif second_avg < first_avg * 0.95:  # 5% threshold
        return "decreasing"
    else:
        return "stable"


def _calculate_completion_velocity(project_metrics: List) -> float:
    """Calculate project completion velocity (percentage per day)."""
    if len(project_metrics) < 2:
        return 0.0
    
    # Calculate velocity based on first and last metrics
    first_metric = project_metrics[0]
    last_metric = project_metrics[-1]
    
    time_diff_days = (last_metric.timestamp - first_metric.timestamp) / 86400  # Convert to days
    completion_diff = last_metric.completion_percentage - first_metric.completion_percentage
    
    return completion_diff / time_diff_days if time_diff_days > 0 else 0.0


def _expected_completion_percentage(project_metrics: List) -> float:
    """Calculate expected completion percentage based on project timeline."""
    if not project_metrics:
        return 0.0
    
    # Simple heuristic: assume linear progress
    # In a real implementation, this would consider project start date and deadline
    return 50.0  # Placeholder: expect 50% completion at this point