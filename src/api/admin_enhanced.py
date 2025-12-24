"""
Enhanced Admin API endpoints for SuperInsight Platform Management Console.

Provides advanced management console features including:
- Real-time system monitoring with WebSocket support
- User behavior analysis and reporting
- Hot configuration updates with validation
- Visual workflow management and monitoring
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict
from fastapi import APIRouter, HTTPException, Depends, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config.settings import settings
from src.admin.dashboard import dashboard_manager
from src.admin.user_analytics import ActionType
from src.admin.workflow_manager import WorkflowStatus
from src.api.admin import get_admin_user


logger = logging.getLogger(__name__)

# Create enhanced admin router
router = APIRouter(prefix="/admin/enhanced", tags=["Enhanced Admin"])


# Pydantic models for enhanced admin API
class ConfigUpdateRequest(BaseModel):
    """Configuration update request."""
    section: str = Field(..., description="配置节")
    key: str = Field(..., description="配置键")
    value: Any = Field(..., description="配置值")
    validate: bool = Field(True, description="是否验证配置值")


class WorkflowCreateRequest(BaseModel):
    """Workflow creation request."""
    template_id: Optional[str] = Field(None, description="模板ID")
    name: str = Field(..., description="工作流名称")
    description: Optional[str] = Field("", description="工作流描述")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="参数")
    priority: int = Field(0, description="优先级")
    task_definitions: Optional[List[Dict[str, Any]]] = Field(None, description="自定义任务定义")


class UserActionTrackingRequest(BaseModel):
    """User action tracking request."""
    user_id: str = Field(..., description="用户ID")
    session_id: str = Field(..., description="会话ID")
    action_type: str = Field(..., description="操作类型")
    details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="操作详情")
    duration_ms: Optional[float] = Field(None, description="操作耗时（毫秒）")
    success: bool = Field(True, description="操作是否成功")


# Real-time Monitoring Endpoints
@router.websocket("/monitoring/realtime")
async def realtime_monitoring_websocket(websocket: WebSocket):
    """Real-time monitoring WebSocket endpoint."""
    await websocket.accept()
    
    # Initialize dashboard manager if needed
    if not dashboard_manager.is_initialized:
        await dashboard_manager.initialize()
    
    # Add subscriber
    dashboard_manager.monitoring_service.add_subscriber(websocket)
    
    try:
        # Send initial data
        current_metrics = dashboard_manager.monitoring_service.get_current_metrics()
        if current_metrics:
            await websocket.send_text(json.dumps(current_metrics))
        
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client messages (ping/pong, configuration changes, etc.)
                message = await websocket.receive_text()
                
                # Handle client requests
                try:
                    request = json.loads(message)
                    if request.get("type") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                    elif request.get("type") == "get_history":
                        hours = request.get("hours", 1)
                        history = dashboard_manager.monitoring_service.get_metrics_history(hours)
                        await websocket.send_text(json.dumps({
                            "type": "history",
                            "data": history
                        }))
                except json.JSONDecodeError:
                    # Ignore invalid JSON
                    pass
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
    
    finally:
        # Remove subscriber
        dashboard_manager.monitoring_service.remove_subscriber(websocket)


@router.get("/monitoring/current")
async def get_current_monitoring_data(admin_user=Depends(get_admin_user)):
    """Get current monitoring data."""
    try:
        if not dashboard_manager.is_initialized:
            await dashboard_manager.initialize()
        
        current_metrics = dashboard_manager.monitoring_service.get_current_metrics()
        
        if current_metrics:
            return current_metrics
        else:
            return {
                "message": "No current metrics available",
                "timestamp": datetime.now().timestamp()
            }
            
    except Exception as e:
        logger.error(f"Failed to get current monitoring data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get monitoring data: {str(e)}"
        )


@router.get("/monitoring/history")
async def get_monitoring_history(
    hours: int = Query(1, ge=1, le=24, description="历史数据时长（小时）"),
    admin_user=Depends(get_admin_user)
):
    """Get monitoring history data."""
    try:
        if not dashboard_manager.is_initialized:
            await dashboard_manager.initialize()
        
        history = dashboard_manager.monitoring_service.get_metrics_history(hours)
        
        return {
            "history": history,
            "period_hours": hours,
            "total_points": len(history)
        }
        
    except Exception as e:
        logger.error(f"Failed to get monitoring history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get monitoring history: {str(e)}"
        )


# Configuration Management Endpoints
@router.post("/config/update")
async def update_configuration(
    request: ConfigUpdateRequest,
    admin_user=Depends(get_admin_user)
):
    """Update system configuration with hot reload."""
    try:
        if not dashboard_manager.is_initialized:
            await dashboard_manager.initialize()
        
        user_id = admin_user.get('user_id', 'unknown')
        
        result = await dashboard_manager.config_service.update_config(
            section=request.section,
            key=request.key,
            value=request.value,
            user=user_id,
            validate=request.validate
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration update failed: {str(e)}"
        )


@router.post("/config/rollback/{change_id}")
async def rollback_configuration(
    change_id: int,
    admin_user=Depends(get_admin_user)
):
    """Rollback a configuration change."""
    try:
        if not dashboard_manager.is_initialized:
            await dashboard_manager.initialize()
        
        user_id = admin_user.get('user_id', 'unknown')
        
        result = await dashboard_manager.config_service.rollback_config(
            change_id=change_id,
            user=user_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to rollback configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration rollback failed: {str(e)}"
        )


@router.get("/config/changes")
async def get_configuration_changes(
    limit: int = Query(50, ge=1, le=500, description="返回记录数量"),
    admin_user=Depends(get_admin_user)
):
    """Get configuration change history."""
    try:
        if not dashboard_manager.is_initialized:
            await dashboard_manager.initialize()
        
        changes = dashboard_manager.config_service.get_change_history(limit)
        
        return {
            "changes": changes,
            "total": len(changes)
        }
        
    except Exception as e:
        logger.error(f"Failed to get configuration changes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get configuration changes: {str(e)}"
        )


@router.get("/config/pending")
async def get_pending_configuration_changes(admin_user=Depends(get_admin_user)):
    """Get pending configuration changes."""
    try:
        if not dashboard_manager.is_initialized:
            await dashboard_manager.initialize()
        
        pending = dashboard_manager.config_service.get_pending_changes()
        
        return {
            "pending_changes": pending,
            "count": len(pending)
        }
        
    except Exception as e:
        logger.error(f"Failed to get pending changes: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get pending changes: {str(e)}"
        )


# User Analytics Endpoints
@router.post("/analytics/track-action")
async def track_user_action(
    request: UserActionTrackingRequest,
    admin_user=Depends(get_admin_user)
):
    """Track a user action for analytics."""
    try:
        if not dashboard_manager.is_initialized:
            await dashboard_manager.initialize()
        
        # Convert string action type to enum
        try:
            action_type = ActionType(request.action_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid action type: {request.action_type}"
            )
        
        dashboard_manager.user_analytics.track_action(
            user_id=request.user_id,
            session_id=request.session_id,
            action_type=action_type,
            details=request.details,
            duration_ms=request.duration_ms,
            success=request.success
        )
        
        return {
            "message": "Action tracked successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to track user action: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to track action: {str(e)}"
        )


@router.get("/analytics/current-stats")
async def get_current_analytics_stats(admin_user=Depends(get_admin_user)):
    """Get current user analytics statistics."""
    try:
        if not dashboard_manager.is_initialized:
            await dashboard_manager.initialize()
        
        stats = dashboard_manager.user_analytics.get_current_stats()
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get analytics stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analytics stats: {str(e)}"
        )


@router.get("/analytics/user/{user_id}")
async def get_user_analytics_profile(
    user_id: str,
    admin_user=Depends(get_admin_user)
):
    """Get analytics profile for a specific user."""
    try:
        if not dashboard_manager.is_initialized:
            await dashboard_manager.initialize()
        
        profile = dashboard_manager.user_analytics.get_user_profile(user_id)
        
        if profile:
            return profile
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User profile not found: {user_id}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user profile: {str(e)}"
        )


@router.get("/analytics/user/{user_id}/report")
async def get_user_activity_report(
    user_id: str,
    days: int = Query(7, ge=1, le=30, description="报告天数"),
    admin_user=Depends(get_admin_user)
):
    """Get detailed activity report for a user."""
    try:
        if not dashboard_manager.is_initialized:
            await dashboard_manager.initialize()
        
        report = dashboard_manager.user_analytics.get_user_activity_report(user_id, days)
        
        return report
        
    except Exception as e:
        logger.error(f"Failed to get user activity report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user activity report: {str(e)}"
        )


@router.get("/analytics/system-report")
async def get_system_activity_report(
    days: int = Query(7, ge=1, le=30, description="报告天数"),
    admin_user=Depends(get_admin_user)
):
    """Get system-wide activity report."""
    try:
        if not dashboard_manager.is_initialized:
            await dashboard_manager.initialize()
        
        report = dashboard_manager.user_analytics.get_system_activity_report(days)
        
        return report
        
    except Exception as e:
        logger.error(f"Failed to get system activity report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system activity report: {str(e)}"
        )


@router.get("/analytics/realtime-activity")
async def get_realtime_activity(admin_user=Depends(get_admin_user)):
    """Get real-time activity data."""
    try:
        if not dashboard_manager.is_initialized:
            await dashboard_manager.initialize()
        
        activity = dashboard_manager.user_analytics.get_real_time_activity()
        
        return activity
        
    except Exception as e:
        logger.error(f"Failed to get real-time activity: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get real-time activity: {str(e)}"
        )


# Workflow Management Endpoints
@router.post("/workflows")
async def create_workflow(
    request: WorkflowCreateRequest,
    admin_user=Depends(get_admin_user)
):
    """Create a new workflow."""
    try:
        if not dashboard_manager.is_initialized:
            await dashboard_manager.initialize()
        
        user_id = admin_user.get('user_id', 'unknown')
        
        if request.template_id:
            # Create from template
            workflow_id = dashboard_manager.workflow_manager.create_workflow_from_template(
                template_id=request.template_id,
                name=request.name,
                parameters=request.parameters,
                created_by=user_id,
                priority=request.priority
            )
        else:
            # Create custom workflow
            if not request.task_definitions:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Task definitions required for custom workflow"
                )
            
            workflow_id = dashboard_manager.workflow_manager.create_custom_workflow(
                name=request.name,
                description=request.description,
                task_definitions=request.task_definitions,
                created_by=user_id,
                priority=request.priority
            )
        
        return {
            "workflow_id": workflow_id,
            "message": "Workflow created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create workflow: {str(e)}"
        )


@router.get("/workflows")
async def get_workflows(
    status: Optional[str] = Query(None, description="工作流状态过滤"),
    created_by: Optional[str] = Query(None, description="创建者过滤"),
    limit: int = Query(50, ge=1, le=200, description="返回数量限制"),
    admin_user=Depends(get_admin_user)
):
    """Get workflows with optional filtering."""
    try:
        if not dashboard_manager.is_initialized:
            await dashboard_manager.initialize()
        
        # Convert status string to enum if provided
        status_filter = None
        if status:
            try:
                status_filter = WorkflowStatus(status)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid workflow status: {status}"
                )
        
        workflows = dashboard_manager.workflow_manager.get_workflows(
            status=status_filter,
            created_by=created_by,
            limit=limit
        )
        
        return {
            "workflows": workflows,
            "total": len(workflows)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflows: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflows: {str(e)}"
        )


@router.get("/workflows/{workflow_id}")
async def get_workflow(
    workflow_id: str,
    admin_user=Depends(get_admin_user)
):
    """Get workflow details."""
    try:
        if not dashboard_manager.is_initialized:
            await dashboard_manager.initialize()
        
        workflow = dashboard_manager.workflow_manager.get_workflow(workflow_id)
        
        if workflow:
            return workflow
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow not found: {workflow_id}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow: {str(e)}"
        )


@router.post("/workflows/{workflow_id}/start")
async def start_workflow(
    workflow_id: str,
    admin_user=Depends(get_admin_user)
):
    """Start a workflow execution."""
    try:
        if not dashboard_manager.is_initialized:
            await dashboard_manager.initialize()
        
        success = dashboard_manager.workflow_manager.start_workflow(workflow_id)
        
        if success:
            return {
                "message": "Workflow started successfully",
                "workflow_id": workflow_id
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to start workflow"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start workflow: {str(e)}"
        )


@router.post("/workflows/{workflow_id}/pause")
async def pause_workflow(
    workflow_id: str,
    admin_user=Depends(get_admin_user)
):
    """Pause a running workflow."""
    try:
        if not dashboard_manager.is_initialized:
            await dashboard_manager.initialize()
        
        success = dashboard_manager.workflow_manager.pause_workflow(workflow_id)
        
        if success:
            return {
                "message": "Workflow paused successfully",
                "workflow_id": workflow_id
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to pause workflow"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to pause workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to pause workflow: {str(e)}"
        )


@router.post("/workflows/{workflow_id}/resume")
async def resume_workflow(
    workflow_id: str,
    admin_user=Depends(get_admin_user)
):
    """Resume a paused workflow."""
    try:
        if not dashboard_manager.is_initialized:
            await dashboard_manager.initialize()
        
        success = dashboard_manager.workflow_manager.resume_workflow(workflow_id)
        
        if success:
            return {
                "message": "Workflow resumed successfully",
                "workflow_id": workflow_id
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to resume workflow"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resume workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resume workflow: {str(e)}"
        )


@router.post("/workflows/{workflow_id}/cancel")
async def cancel_workflow(
    workflow_id: str,
    admin_user=Depends(get_admin_user)
):
    """Cancel a workflow."""
    try:
        if not dashboard_manager.is_initialized:
            await dashboard_manager.initialize()
        
        success = dashboard_manager.workflow_manager.cancel_workflow(workflow_id)
        
        if success:
            return {
                "message": "Workflow cancelled successfully",
                "workflow_id": workflow_id
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to cancel workflow"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel workflow: {str(e)}"
        )


@router.get("/workflows/{workflow_id}/visualization")
async def get_workflow_visualization(
    workflow_id: str,
    admin_user=Depends(get_admin_user)
):
    """Get workflow visualization data."""
    try:
        if not dashboard_manager.is_initialized:
            await dashboard_manager.initialize()
        
        visualization = dashboard_manager.workflow_manager.get_workflow_visualization_data(workflow_id)
        
        if visualization:
            return visualization
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow not found: {workflow_id}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow visualization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow visualization: {str(e)}"
        )


@router.get("/workflows/templates")
async def get_workflow_templates(admin_user=Depends(get_admin_user)):
    """Get all workflow templates."""
    try:
        if not dashboard_manager.is_initialized:
            await dashboard_manager.initialize()
        
        templates = dashboard_manager.workflow_manager.get_templates()
        
        return {
            "templates": templates,
            "total": len(templates)
        }
        
    except Exception as e:
        logger.error(f"Failed to get workflow templates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow templates: {str(e)}"
        )


@router.get("/workflows/stats")
async def get_workflow_stats(admin_user=Depends(get_admin_user)):
    """Get workflow statistics."""
    try:
        if not dashboard_manager.is_initialized:
            await dashboard_manager.initialize()
        
        stats = dashboard_manager.workflow_manager.get_workflow_stats()
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get workflow stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow stats: {str(e)}"
        )


# Enhanced Analytics and Predictions Endpoints
@router.get("/analytics/predictive-insights")
async def get_predictive_insights(
    hours_ahead: int = Query(24, ge=1, le=168, description="预测时长（小时）"),
    admin_user=Depends(get_admin_user)
):
    """Get predictive insights for system planning."""
    try:
        if not dashboard_manager.is_initialized:
            await dashboard_manager.initialize()
        
        insights = dashboard_manager.get_predictive_insights(hours_ahead)
        
        return insights
        
    except Exception as e:
        logger.error(f"Failed to get predictive insights: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get predictive insights: {str(e)}"
        )


@router.get("/analytics/system-recommendations")
async def get_system_recommendations(admin_user=Depends(get_admin_user)):
    """Get automated system recommendations."""
    try:
        if not dashboard_manager.is_initialized:
            await dashboard_manager.initialize()
        
        recommendations = dashboard_manager.get_system_recommendations()
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Failed to get system recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system recommendations: {str(e)}"
        )


@router.get("/analytics/anomaly-detection")
async def get_anomaly_detection(admin_user=Depends(get_admin_user)):
    """Get current anomaly detection results."""
    try:
        if not dashboard_manager.is_initialized:
            await dashboard_manager.initialize()
        
        anomalies = dashboard_manager.monitoring_service.predictive_analytics.detect_anomalies()
        
        return {
            "anomalies": anomalies,
            "total_anomalies": len(anomalies),
            "detection_timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get anomaly detection: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get anomaly detection: {str(e)}"
        )


@router.get("/analytics/capacity-forecast")
async def get_capacity_forecast(
    resource_type: str = Query("all", description="资源类型 (cpu, memory, disk, all)"),
    forecast_days: int = Query(7, ge=1, le=30, description="预测天数"),
    admin_user=Depends(get_admin_user)
):
    """Get capacity forecasting for specific resources."""
    try:
        if not dashboard_manager.is_initialized:
            await dashboard_manager.initialize()
        
        # Get predictions for multiple time horizons
        forecasts = {}
        
        for days in range(1, forecast_days + 1):
            hours_ahead = days * 24
            predictions = dashboard_manager.monitoring_service.predictive_analytics.predict_resource_usage(hours_ahead)
            
            if resource_type == "all":
                forecasts[f"day_{days}"] = predictions
            else:
                if resource_type in predictions:
                    forecasts[f"day_{days}"] = {resource_type: predictions[resource_type]}
        
        return {
            "forecasts": forecasts,
            "resource_type": resource_type,
            "forecast_days": forecast_days,
            "generated_at": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get capacity forecast: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get capacity forecast: {str(e)}"
        )


# Enhanced Visualization Endpoints
@router.get("/visualization/performance-trends")
async def get_performance_trends(
    hours: int = Query(24, ge=1, le=168, description="时间范围（小时）"),
    metrics: List[str] = Query(["cpu_usage", "memory_usage", "response_time"], description="指标列表"),
    admin_user=Depends(get_admin_user)
):
    """Get performance trends for visualization."""
    try:
        if not dashboard_manager.is_initialized:
            await dashboard_manager.initialize()
        
        history = dashboard_manager.monitoring_service.get_metrics_history(hours)
        
        # Extract trend data for requested metrics
        trends = {metric: [] for metric in metrics}
        timestamps = []
        
        for data_point in history:
            timestamps.append(data_point["timestamp"])
            perf_metrics = data_point.get("performance_metrics", {})
            
            for metric in metrics:
                value = perf_metrics.get(metric, 0)
                trends[metric].append(value)
        
        return {
            "trends": trends,
            "timestamps": timestamps,
            "metrics": metrics,
            "data_points": len(timestamps),
            "time_range_hours": hours
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance trends: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance trends: {str(e)}"
        )


@router.get("/visualization/alert-heatmap")
async def get_alert_heatmap(
    days: int = Query(7, ge=1, le=30, description="天数"),
    admin_user=Depends(get_admin_user)
):
    """Get alert heatmap data for visualization."""
    try:
        if not dashboard_manager.is_initialized:
            await dashboard_manager.initialize()
        
        hours = days * 24
        history = dashboard_manager.monitoring_service.get_metrics_history(hours)
        
        # Create heatmap data structure
        heatmap_data = defaultdict(lambda: defaultdict(int))
        
        for data_point in history:
            timestamp = data_point["timestamp"]
            dt = datetime.fromtimestamp(timestamp)
            day_key = dt.strftime("%Y-%m-%d")
            hour_key = dt.hour
            
            alerts = data_point.get("alerts", [])
            alert_count = len(alerts)
            
            heatmap_data[day_key][hour_key] += alert_count
        
        # Convert to list format for easier visualization
        heatmap_list = []
        for day, hours_data in heatmap_data.items():
            for hour, count in hours_data.items():
                heatmap_list.append({
                    "day": day,
                    "hour": hour,
                    "alert_count": count
                })
        
        return {
            "heatmap_data": heatmap_list,
            "days": days,
            "generated_at": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get alert heatmap: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get alert heatmap: {str(e)}"
        )


# Enhanced Dashboard Overview Endpoint
@router.get("/dashboard/enhanced-overview")
async def get_enhanced_dashboard_overview(admin_user=Depends(get_admin_user)):
    """Get enhanced dashboard overview with predictive analytics."""
    try:
        if not dashboard_manager.is_initialized:
            await dashboard_manager.initialize()
        
        # Get standard overview
        overview = dashboard_manager.get_dashboard_overview()
        
        # Add enhanced features
        predictive_insights = dashboard_manager.get_predictive_insights(hours_ahead=24)
        system_recommendations = dashboard_manager.get_system_recommendations()
        
        enhanced_overview = {
            **overview,
            "predictive_insights": predictive_insights,
            "system_recommendations": system_recommendations,
            "enhanced_features": {
                "predictive_analytics_enabled": dashboard_manager.monitoring_service.enable_predictive_analytics,
                "automated_recommendations_enabled": dashboard_manager.monitoring_service.enable_automated_recommendations,
                "anomaly_detection_enabled": True,
                "capacity_forecasting_enabled": True
            }
        }
        
        return enhanced_overview
        
    except Exception as e:
        logger.error(f"Failed to get enhanced dashboard overview: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get enhanced dashboard overview: {str(e)}"
        )


# Dashboard Overview Endpoint (Original)
@router.get("/dashboard/overview")
async def get_dashboard_overview(admin_user=Depends(get_admin_user)):
    """Get comprehensive dashboard overview."""
    try:
        if not dashboard_manager.is_initialized:
            await dashboard_manager.initialize()
        
        overview = dashboard_manager.get_dashboard_overview()
        
        return overview
        
    except Exception as e:
        logger.error(f"Failed to get dashboard overview: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get dashboard overview: {str(e)}"
        )


# Health check for enhanced admin features
@router.get("/health")
async def enhanced_admin_health_check():
    """Enhanced admin features health check."""
    try:
        health_status = {
            "status": "healthy",
            "service": "enhanced_admin",
            "timestamp": datetime.now().isoformat(),
            "features": {
                "dashboard_manager": dashboard_manager.is_initialized,
                "real_time_monitoring": dashboard_manager.monitoring_service.is_running if dashboard_manager.is_initialized else False,
                "user_analytics": dashboard_manager.user_analytics.is_running if dashboard_manager.is_initialized else False,
                "workflow_manager": dashboard_manager.workflow_manager.is_running if dashboard_manager.is_initialized else False
            }
        }
        
        # Check if all features are healthy
        all_healthy = all(health_status["features"].values())
        if not all_healthy:
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Enhanced admin health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "enhanced_admin",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }