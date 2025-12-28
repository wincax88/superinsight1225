"""
Comprehensive Dashboard API for SuperInsight Platform.

Provides REST API endpoints for:
- System overview dashboard
- Real-time metrics visualization
- Advanced anomaly detection
- SLA compliance monitoring
- Capacity planning
- Report generation and scheduling
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from src.monitoring.advanced_anomaly_detection import (
    AdvancedAnomalyDetector,
    AnomalyType,
    AnomalySeverity,
    advanced_anomaly_detector
)
from src.monitoring.report_service import (
    MonitoringReportService,
    ReportType,
    ReportFormat,
    ReportFrequency,
    SLADefinition,
    monitoring_report_service
)
from src.system.monitoring import metrics_collector, performance_monitor, health_monitor
from src.system.business_metrics import business_metrics_collector


router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


# ==================== Request/Response Models ====================

class AnalyzeMetricRequest(BaseModel):
    """Request model for analyzing a metric."""
    metric_name: str = Field(..., description="Name of the metric")
    value: float = Field(..., description="Current metric value")
    timestamp: Optional[float] = Field(None, description="Unix timestamp")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class AddResponseRuleRequest(BaseModel):
    """Request model for adding automated response rule."""
    metric_pattern: str = Field(..., description="Metric name pattern (supports *)")
    anomaly_type: str = Field(..., description="Type of anomaly to respond to")
    min_severity: str = Field(..., description="Minimum severity to trigger response")
    action_type: str = Field(..., description="Type of action to take")
    action_params: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")
    cooldown_seconds: int = Field(300, ge=60, le=3600, description="Cooldown between actions")


class GenerateReportRequest(BaseModel):
    """Request model for generating a report."""
    report_type: str = Field(..., description="Type of report")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Report parameters")
    output_format: str = Field("json", description="Output format")


class AddScheduleRequest(BaseModel):
    """Request model for adding a report schedule."""
    report_type: str = Field(..., description="Type of report")
    frequency: str = Field(..., description="Report frequency")
    recipients: List[str] = Field(..., description="Email recipients")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Report parameters")


class AddSLARequest(BaseModel):
    """Request model for adding an SLA definition."""
    name: str = Field(..., description="SLA name")
    metric_name: str = Field(..., description="Metric to monitor")
    target_value: float = Field(..., description="Target value")
    comparison: str = Field(..., description="Comparison type: gte, lte, eq")
    measurement_period_hours: int = Field(24, description="Measurement period in hours")
    description: str = Field("", description="SLA description")
    critical: bool = Field(False, description="Whether this is a critical SLA")


# ==================== System Overview Endpoints ====================

@router.get("/overview", response_model=Dict[str, Any])
async def get_system_overview() -> Dict[str, Any]:
    """
    Get comprehensive system overview dashboard.

    Returns system health, performance metrics, active alerts,
    and business KPIs in a single response.
    """
    try:
        # Get system health
        system_health = await health_monitor.check_system_health()

        # Get performance insights
        performance_insights = metrics_collector.get_performance_insights()

        # Get business summary
        business_summary = business_metrics_collector.get_business_summary()

        # Get active anomalies
        active_anomalies = advanced_anomaly_detector.get_active_anomalies()

        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "system_health": {
                "overall_status": system_health.get("overall_status", "unknown"),
                "checks": system_health.get("checks", {}),
                "alerts": system_health.get("alerts", [])[:5],
                "predictions": system_health.get("predictions", {})
            },
            "performance": {
                "bottlenecks": [
                    {
                        "component": b.component,
                        "severity": b.severity,
                        "description": b.description
                    }
                    for b in performance_insights.get("bottlenecks", [])
                ],
                "trends": performance_insights.get("trends", {}),
                "recommendations": performance_insights.get("recommendations", [])
            },
            "business_metrics": business_summary,
            "anomalies": {
                "active_count": len(active_anomalies),
                "recent": active_anomalies[:5]
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get overview: {str(e)}")


@router.get("/metrics/realtime", response_model=Dict[str, Any])
async def get_realtime_metrics(
    metrics: Optional[str] = Query(None, description="Comma-separated metric names")
) -> Dict[str, Any]:
    """
    Get real-time metrics data.

    Returns current values and recent history for specified metrics.
    """
    try:
        if metrics:
            metric_names = [m.strip() for m in metrics.split(",")]
        else:
            # Default key metrics
            metric_names = [
                "system.cpu.usage_percent",
                "system.memory.usage_percent",
                "system.disk.usage_percent",
                "requests.duration",
                "database.query.duration"
            ]

        metric_data = {}
        for name in metric_names:
            summary = metrics_collector.get_metric_summary(name)
            metric_data[name] = summary

        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "metrics": metric_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get("/metrics/history", response_model=Dict[str, Any])
async def get_metrics_history(
    metric_name: str = Query(..., description="Metric name"),
    hours: int = Query(24, ge=1, le=168, description="Hours of history")
) -> Dict[str, Any]:
    """
    Get historical data for a specific metric.
    """
    try:
        since = datetime.now().timestamp() - (hours * 3600)
        points = metrics_collector.get_metric_values(metric_name, since)

        return {
            "status": "success",
            "metric_name": metric_name,
            "period_hours": hours,
            "data_points": [
                {
                    "timestamp": p.timestamp,
                    "value": p.value,
                    "tags": p.tags
                }
                for p in points
            ],
            "summary": metrics_collector.get_metric_summary(metric_name, since)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


# ==================== Advanced Anomaly Detection Endpoints ====================

@router.post("/anomaly/analyze", response_model=Dict[str, Any])
async def analyze_metric(request: AnalyzeMetricRequest) -> Dict[str, Any]:
    """
    Analyze a metric value for anomalies using ML-based detection.

    Uses multiple detection methods:
    - Isolation Forest for multivariate anomaly detection
    - EWMA for trend detection
    - Seasonal decomposition for periodic patterns
    """
    try:
        anomalies = await advanced_anomaly_detector.analyze_metric(
            metric_name=request.metric_name,
            value=request.value,
            timestamp=request.timestamp,
            context=request.context
        )

        return {
            "status": "success",
            "metric_name": request.metric_name,
            "value": request.value,
            "anomalies_detected": len(anomalies),
            "anomalies": [a.to_dict() for a in anomalies]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze metric: {str(e)}")


@router.get("/anomaly/active", response_model=Dict[str, Any])
async def get_active_anomalies() -> Dict[str, Any]:
    """
    Get all currently active anomalies.
    """
    try:
        anomalies = advanced_anomaly_detector.get_active_anomalies()
        stats = advanced_anomaly_detector.get_detection_statistics()

        return {
            "status": "success",
            "active_anomalies": anomalies,
            "count": len(anomalies),
            "statistics": stats
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get anomalies: {str(e)}")


@router.get("/anomaly/statistics", response_model=Dict[str, Any])
async def get_anomaly_statistics() -> Dict[str, Any]:
    """
    Get anomaly detection statistics.
    """
    try:
        stats = advanced_anomaly_detector.get_detection_statistics()

        return {
            "status": "success",
            "statistics": stats
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@router.post("/anomaly/response-rule", response_model=Dict[str, Any])
async def add_response_rule(request: AddResponseRuleRequest) -> Dict[str, Any]:
    """
    Add an automated response rule for anomaly handling.
    """
    try:
        anomaly_type = AnomalyType(request.anomaly_type)
        severity = AnomalySeverity(request.min_severity)

        advanced_anomaly_detector.add_response_rule(
            metric_pattern=request.metric_pattern,
            anomaly_type=anomaly_type,
            min_severity=severity,
            action_type=request.action_type,
            action_params=request.action_params,
            cooldown_seconds=request.cooldown_seconds
        )

        return {
            "status": "success",
            "message": f"Response rule added for {request.metric_pattern}"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add rule: {str(e)}")


@router.get("/anomaly/response-history", response_model=Dict[str, Any])
async def get_response_history(
    limit: int = Query(50, ge=1, le=200, description="Maximum results")
) -> Dict[str, Any]:
    """
    Get automated response history.
    """
    try:
        history = advanced_anomaly_detector.get_response_history(limit)

        return {
            "status": "success",
            "history": history,
            "count": len(history)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


# ==================== SLA Compliance Endpoints ====================

@router.get("/sla/compliance", response_model=Dict[str, Any])
async def get_sla_compliance() -> Dict[str, Any]:
    """
    Get current SLA compliance status.
    """
    try:
        report = await monitoring_report_service.sla_monitor.generate_sla_report()

        return {
            "status": "success",
            "compliance_report": report
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get compliance: {str(e)}")


@router.get("/sla/definitions", response_model=Dict[str, Any])
async def get_sla_definitions() -> Dict[str, Any]:
    """
    Get all SLA definitions.
    """
    try:
        slas = monitoring_report_service.get_slas()

        return {
            "status": "success",
            "slas": slas,
            "count": len(slas)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get SLAs: {str(e)}")


@router.post("/sla/definitions", response_model=Dict[str, Any])
async def add_sla_definition(request: AddSLARequest) -> Dict[str, Any]:
    """
    Add a new SLA definition.
    """
    try:
        sla = SLADefinition(
            name=request.name,
            metric_name=request.metric_name,
            target_value=request.target_value,
            comparison=request.comparison,
            measurement_period_hours=request.measurement_period_hours,
            description=request.description,
            critical=request.critical
        )

        monitoring_report_service.add_sla(sla)

        return {
            "status": "success",
            "message": f"SLA '{request.name}' added successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add SLA: {str(e)}")


@router.delete("/sla/definitions/{sla_name}", response_model=Dict[str, Any])
async def delete_sla_definition(sla_name: str) -> Dict[str, Any]:
    """
    Delete an SLA definition.
    """
    try:
        success = monitoring_report_service.remove_sla(sla_name)

        if not success:
            raise HTTPException(status_code=404, detail="SLA not found")

        return {
            "status": "success",
            "message": f"SLA '{sla_name}' deleted"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete SLA: {str(e)}")


# ==================== Capacity Planning Endpoints ====================

@router.get("/capacity/prediction", response_model=Dict[str, Any])
async def get_capacity_prediction(
    history_days: int = Query(30, ge=7, le=90, description="Days of history for prediction")
) -> Dict[str, Any]:
    """
    Get capacity planning predictions.
    """
    try:
        report = await monitoring_report_service.capacity_planner.generate_capacity_report(
            history_days
        )

        return {
            "status": "success",
            "capacity_report": report
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get prediction: {str(e)}")


@router.get("/capacity/resource/{resource_name}", response_model=Dict[str, Any])
async def get_resource_prediction(
    resource_name: str,
    history_days: int = Query(30, ge=7, le=90, description="Days of history")
) -> Dict[str, Any]:
    """
    Get capacity prediction for a specific resource.
    """
    try:
        prediction = await monitoring_report_service.capacity_planner.predict_capacity(
            resource_name, history_days
        )

        if not prediction:
            raise HTTPException(status_code=404, detail="Resource not found or insufficient data")

        return {
            "status": "success",
            "prediction": {
                "resource": prediction.resource_name,
                "current_usage": prediction.current_usage,
                "current_capacity": prediction.current_capacity,
                "predicted_7d": prediction.predicted_usage_7d,
                "predicted_30d": prediction.predicted_usage_30d,
                "predicted_90d": prediction.predicted_usage_90d,
                "days_until_threshold": prediction.days_until_threshold,
                "confidence": prediction.confidence,
                "recommendation": prediction.recommendation
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get prediction: {str(e)}")


# ==================== Report Generation Endpoints ====================

@router.post("/reports/generate", response_model=Dict[str, Any])
async def generate_report(
    request: GenerateReportRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Generate a monitoring report on demand.
    """
    try:
        report_type = ReportType(request.report_type)
        output_format = ReportFormat(request.output_format)

        report = await monitoring_report_service.generate_report(
            report_type=report_type,
            parameters=request.parameters,
            output_format=output_format
        )

        return {
            "status": "success",
            "report": {
                "report_id": report.report_id,
                "report_type": report.report_type.value,
                "title": report.title,
                "generated_at": report.generated_at.isoformat(),
                "period_start": report.period_start.isoformat(),
                "period_end": report.period_end.isoformat(),
                "format": report.format.value,
                "content": report.content
            }
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


@router.get("/reports/history", response_model=Dict[str, Any])
async def get_report_history(
    limit: int = Query(20, ge=1, le=100, description="Maximum results")
) -> Dict[str, Any]:
    """
    Get report generation history.
    """
    try:
        history = monitoring_report_service.get_report_history(limit)

        return {
            "status": "success",
            "reports": history,
            "count": len(history)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@router.get("/reports/types", response_model=Dict[str, Any])
async def get_report_types() -> Dict[str, Any]:
    """
    Get available report types.
    """
    return {
        "status": "success",
        "report_types": [
            {"value": rt.value, "name": rt.value.replace("_", " ").title()}
            for rt in ReportType
        ],
        "formats": [
            {"value": rf.value, "name": rf.value.upper()}
            for rf in ReportFormat
        ],
        "frequencies": [
            {"value": freq.value, "name": freq.value.title()}
            for freq in ReportFrequency
        ]
    }


# ==================== Report Scheduling Endpoints ====================

@router.get("/reports/schedules", response_model=Dict[str, Any])
async def get_report_schedules() -> Dict[str, Any]:
    """
    Get all report schedules.
    """
    try:
        schedules = monitoring_report_service.get_schedules()

        return {
            "status": "success",
            "schedules": schedules,
            "count": len(schedules)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get schedules: {str(e)}")


@router.post("/reports/schedules", response_model=Dict[str, Any])
async def add_report_schedule(request: AddScheduleRequest) -> Dict[str, Any]:
    """
    Add a new report schedule.
    """
    try:
        report_type = ReportType(request.report_type)
        frequency = ReportFrequency(request.frequency)

        schedule_id = monitoring_report_service.add_schedule(
            report_type=report_type,
            frequency=frequency,
            recipients=request.recipients,
            parameters=request.parameters
        )

        return {
            "status": "success",
            "schedule_id": schedule_id,
            "message": f"Schedule created for {request.report_type} reports"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add schedule: {str(e)}")


@router.delete("/reports/schedules/{schedule_id}", response_model=Dict[str, Any])
async def delete_report_schedule(schedule_id: str) -> Dict[str, Any]:
    """
    Delete a report schedule.
    """
    try:
        success = monitoring_report_service.remove_schedule(schedule_id)

        if not success:
            raise HTTPException(status_code=404, detail="Schedule not found")

        return {
            "status": "success",
            "message": f"Schedule '{schedule_id}' deleted"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete schedule: {str(e)}")


# ==================== Trend Analysis Endpoints ====================

@router.get("/trends/analysis", response_model=Dict[str, Any])
async def get_trend_analysis(
    metrics: Optional[str] = Query(None, description="Comma-separated metric names"),
    period_hours: int = Query(24, ge=1, le=168, description="Analysis period in hours")
) -> Dict[str, Any]:
    """
    Get trend analysis for metrics.
    """
    try:
        metric_list = None
        if metrics:
            metric_list = [m.strip() for m in metrics.split(",")]

        report = await monitoring_report_service.trend_analyzer.generate_trend_report(
            metric_list, period_hours
        )

        return {
            "status": "success",
            "trend_analysis": report
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trends: {str(e)}")


@router.get("/trends/metric/{metric_name}", response_model=Dict[str, Any])
async def get_metric_trend(
    metric_name: str,
    period_hours: int = Query(24, ge=1, le=168, description="Analysis period")
) -> Dict[str, Any]:
    """
    Get trend analysis for a specific metric.
    """
    try:
        analysis = await monitoring_report_service.trend_analyzer.analyze_trend(
            metric_name, period_hours
        )

        if not analysis:
            raise HTTPException(status_code=404, detail="Metric not found or insufficient data")

        return {
            "status": "success",
            "trend": {
                "metric_name": analysis.metric_name,
                "period_start": analysis.period_start.isoformat(),
                "period_end": analysis.period_end.isoformat(),
                "direction": analysis.direction,
                "change_percentage": analysis.change_percentage,
                "average_value": analysis.average_value,
                "min_value": analysis.min_value,
                "max_value": analysis.max_value,
                "volatility": analysis.volatility,
                "anomaly_count": analysis.anomaly_count,
                "forecast_next_24h": analysis.forecast_next_24h
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trend: {str(e)}")


# ==================== Health Check ====================

@router.get("/health", response_model=Dict[str, Any])
async def dashboard_health() -> Dict[str, Any]:
    """
    Health check for dashboard service.
    """
    return {
        "status": "healthy",
        "service": "dashboard",
        "components": {
            "anomaly_detector": "available",
            "report_service": "available",
            "metrics_collector": "available",
            "health_monitor": "available"
        },
        "timestamp": datetime.now().isoformat()
    }
