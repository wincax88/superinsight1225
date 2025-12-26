"""
Monitoring API for SuperInsight Platform.

Provides REST API endpoints for quality monitoring, alerts,
and dashboard data.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.monitoring.quality_monitor import QualityMonitor
from src.monitoring.alert_manager import AlertManager, AlertSeverity, AlertStatus, NotificationChannel


router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])

# Global instances - use lazy initialization
_quality_monitor: Optional[QualityMonitor] = None
_alert_manager: Optional[AlertManager] = None


def get_quality_monitor() -> QualityMonitor:
    """Get or create quality monitor instance."""
    global _quality_monitor
    if _quality_monitor is None:
        _quality_monitor = QualityMonitor()
    return _quality_monitor


def get_alert_manager() -> AlertManager:
    """Get or create alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


# ==================== Request/Response Models ====================

class CreateAlertRequest(BaseModel):
    """Request model for creating an alert."""
    alert_type: str = Field(..., description="Type of alert")
    severity: str = Field(..., description="Alert severity")
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Alert message")
    source: str = Field(..., description="Source component")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")
    user_id: Optional[str] = Field(None, description="User identifier")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class AcknowledgeAlertRequest(BaseModel):
    """Request model for acknowledging an alert."""
    acknowledged_by: str = Field(..., description="User acknowledging")
    notes: Optional[str] = Field(None, description="Acknowledgment notes")


class ResolveAlertRequest(BaseModel):
    """Request model for resolving an alert."""
    resolved_by: str = Field(..., description="User resolving")
    resolution_notes: Optional[str] = Field(None, description="Resolution notes")


class SilenceAlertRequest(BaseModel):
    """Request model for silencing alerts."""
    alert_type: str = Field(..., description="Alert type to silence")
    duration_minutes: int = Field(..., ge=1, le=1440, description="Duration in minutes")
    source: Optional[str] = Field(None, description="Source filter")
    tenant_id: Optional[str] = Field(None, description="Tenant filter")


class ConfigureNotificationRequest(BaseModel):
    """Request model for configuring notifications."""
    name: str = Field(..., description="Configuration name")
    channel: str = Field(..., description="Notification channel")
    enabled: bool = Field(True, description="Whether enabled")
    min_severity: str = Field("warning", description="Minimum severity")
    recipients: Optional[List[str]] = Field(None, description="Recipients")
    config: Optional[Dict[str, Any]] = Field(None, description="Channel config")


class UpdateThresholdsRequest(BaseModel):
    """Request model for updating thresholds."""
    thresholds: Dict[str, Dict[str, float]] = Field(..., description="Threshold configuration")


# ==================== Dashboard Endpoints ====================

@router.get("/dashboard", response_model=Dict[str, Any])
async def get_quality_dashboard(
    tenant_id: Optional[str] = Query(None, description="Tenant filter")
) -> Dict[str, Any]:
    """
    Get comprehensive quality monitoring dashboard.

    Returns real-time metrics, historical data, anomalies, and health score.
    """
    try:
        monitor = get_quality_monitor()

        dashboard = await monitor.get_quality_dashboard(tenant_id)

        return {
            "status": "success",
            "dashboard": dashboard
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard: {str(e)}")


@router.get("/realtime", response_model=Dict[str, Any])
async def get_realtime_metrics(
    tenant_id: Optional[str] = Query(None, description="Tenant filter")
) -> Dict[str, Any]:
    """
    Get current real-time quality metrics.
    """
    try:
        monitor = get_quality_monitor()

        metrics = await monitor.get_realtime_metrics(tenant_id)

        return {
            "status": "success",
            "metrics": metrics
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get("/anomalies", response_model=Dict[str, Any])
async def get_anomalies(
    tenant_id: Optional[str] = Query(None, description="Tenant filter")
) -> Dict[str, Any]:
    """
    Get detected quality anomalies.
    """
    try:
        monitor = get_quality_monitor()

        anomalies = await monitor.check_anomalies(tenant_id)

        return {
            "status": "success",
            "anomalies": anomalies,
            "count": len(anomalies)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get anomalies: {str(e)}")


@router.get("/metrics/summary", response_model=Dict[str, Any])
async def get_metrics_summary() -> Dict[str, Any]:
    """
    Get summary of all collected metrics.
    """
    try:
        monitor = get_quality_monitor()

        summary = monitor.get_metrics_summary()

        return {
            "status": "success",
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")


@router.post("/thresholds", response_model=Dict[str, Any])
async def update_alert_thresholds(request: UpdateThresholdsRequest) -> Dict[str, Any]:
    """
    Update alert thresholds.
    """
    try:
        monitor = get_quality_monitor()

        monitor.update_thresholds(request.thresholds)

        return {
            "status": "success",
            "message": "Thresholds updated"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update thresholds: {str(e)}")


# ==================== Monitoring Control Endpoints ====================

@router.post("/start", response_model=Dict[str, Any])
async def start_monitoring(
    interval: int = Query(60, ge=10, le=300, description="Collection interval in seconds")
) -> Dict[str, Any]:
    """
    Start real-time quality monitoring.
    """
    try:
        monitor = get_quality_monitor()

        await monitor.start_monitoring(interval)

        return {
            "status": "success",
            "message": f"Monitoring started with {interval}s interval"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {str(e)}")


@router.post("/stop", response_model=Dict[str, Any])
async def stop_monitoring() -> Dict[str, Any]:
    """
    Stop real-time quality monitoring.
    """
    try:
        monitor = get_quality_monitor()

        await monitor.stop_monitoring()

        return {
            "status": "success",
            "message": "Monitoring stopped"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop monitoring: {str(e)}")


# ==================== Alert Endpoints ====================

@router.post("/alerts", response_model=Dict[str, Any])
async def create_alert(request: CreateAlertRequest) -> Dict[str, Any]:
    """
    Create a new alert.
    """
    try:
        manager = get_alert_manager()

        severity = AlertSeverity(request.severity)

        alert = await manager.create_alert(
            alert_type=request.alert_type,
            severity=severity,
            title=request.title,
            message=request.message,
            source=request.source,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            context=request.context
        )

        if not alert:
            return {
                "status": "silenced",
                "message": "Alert was silenced"
            }

        return {
            "status": "success",
            "alert": alert.to_dict()
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create alert: {str(e)}")


@router.get("/alerts", response_model=Dict[str, Any])
async def get_alerts(
    tenant_id: Optional[str] = Query(None, description="Tenant filter"),
    severity: Optional[str] = Query(None, description="Severity filter"),
    source: Optional[str] = Query(None, description="Source filter"),
    limit: int = Query(100, ge=1, le=500, description="Max results")
) -> Dict[str, Any]:
    """
    Get active alerts.
    """
    try:
        manager = get_alert_manager()

        severity_filter = AlertSeverity(severity) if severity else None

        alerts = await manager.get_active_alerts(
            tenant_id=tenant_id,
            severity=severity_filter,
            source=source,
            limit=limit
        )

        return {
            "status": "success",
            "alerts": alerts,
            "count": len(alerts)
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")


@router.put("/alerts/{alert_id}/acknowledge", response_model=Dict[str, Any])
async def acknowledge_alert(
    alert_id: UUID,
    request: AcknowledgeAlertRequest
) -> Dict[str, Any]:
    """
    Acknowledge an alert.
    """
    try:
        manager = get_alert_manager()

        success = await manager.acknowledge_alert(
            alert_id=alert_id,
            acknowledged_by=request.acknowledged_by,
            notes=request.notes
        )

        if not success:
            raise HTTPException(status_code=400, detail="Cannot acknowledge alert")

        return {
            "status": "success",
            "message": "Alert acknowledged"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to acknowledge alert: {str(e)}")


@router.put("/alerts/{alert_id}/resolve", response_model=Dict[str, Any])
async def resolve_alert(
    alert_id: UUID,
    request: ResolveAlertRequest
) -> Dict[str, Any]:
    """
    Resolve an alert.
    """
    try:
        manager = get_alert_manager()

        success = await manager.resolve_alert(
            alert_id=alert_id,
            resolved_by=request.resolved_by,
            resolution_notes=request.resolution_notes
        )

        if not success:
            raise HTTPException(status_code=400, detail="Cannot resolve alert")

        return {
            "status": "success",
            "message": "Alert resolved"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resolve alert: {str(e)}")


@router.post("/alerts/silence", response_model=Dict[str, Any])
async def silence_alerts(request: SilenceAlertRequest) -> Dict[str, Any]:
    """
    Silence an alert type for a duration.
    """
    try:
        manager = get_alert_manager()

        rule_id = manager.silence_alert_type(
            alert_type=request.alert_type,
            duration_minutes=request.duration_minutes,
            source=request.source,
            tenant_id=request.tenant_id
        )

        return {
            "status": "success",
            "rule_id": rule_id,
            "message": f"Alerts silenced for {request.duration_minutes} minutes"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to silence alerts: {str(e)}")


@router.get("/alerts/statistics", response_model=Dict[str, Any])
async def get_alert_statistics(
    days: int = Query(7, ge=1, le=30, description="Analysis period"),
    tenant_id: Optional[str] = Query(None, description="Tenant filter")
) -> Dict[str, Any]:
    """
    Get alert statistics.
    """
    try:
        manager = get_alert_manager()

        stats = await manager.get_alert_statistics(days, tenant_id)

        return {
            "status": "success",
            "statistics": stats
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


# ==================== Notification Configuration Endpoints ====================

@router.post("/notifications/configure", response_model=Dict[str, Any])
async def configure_notification(request: ConfigureNotificationRequest) -> Dict[str, Any]:
    """
    Configure a notification channel.
    """
    try:
        manager = get_alert_manager()

        channel = NotificationChannel(request.channel)
        severity = AlertSeverity(request.min_severity)

        manager.configure_notification(
            name=request.name,
            channel=channel,
            enabled=request.enabled,
            min_severity=severity,
            recipients=request.recipients,
            config=request.config
        )

        return {
            "status": "success",
            "message": f"Notification '{request.name}' configured"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to configure notification: {str(e)}")


@router.get("/notifications/config", response_model=Dict[str, Any])
async def get_notification_configs() -> Dict[str, Any]:
    """
    Get current notification configurations.
    """
    try:
        manager = get_alert_manager()

        configs = manager.get_notification_configs()

        return {
            "status": "success",
            "configurations": configs
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get configs: {str(e)}")


# ==================== Health Check ====================

@router.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """
    Health check for monitoring service.
    """
    monitor = get_quality_monitor()
    manager = get_alert_manager()

    return {
        "status": "healthy",
        "service": "quality-monitoring",
        "components": {
            "quality_monitor": "available",
            "alert_manager": "available"
        },
        "monitoring_active": monitor._is_monitoring,
        "timestamp": datetime.now().isoformat()
    }
