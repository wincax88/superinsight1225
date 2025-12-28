"""
Billing API endpoints for SuperInsight platform.

Provides REST API for billing operations, reports, and analysis.
"""

from datetime import date, datetime
from typing import Dict, Any, List, Optional
from uuid import UUID
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from src.billing.service import BillingSystem
from src.billing.analytics import BillingAnalytics
from src.billing.models import BillingRule, BillingMode, BillingReport


# Initialize router
router = APIRouter(prefix="/api/billing", tags=["billing"])

# Initialize billing system and analytics
billing_system = BillingSystem()
billing_analytics = BillingAnalytics(billing_system)


class TrackAnnotationRequest(BaseModel):
    """Request model for tracking annotation work."""
    user_id: str = Field(..., description="User who performed the annotation")
    task_id: UUID = Field(..., description="Task being annotated")
    duration: int = Field(..., description="Time spent in seconds")
    tenant_id: str = Field(..., description="Tenant identifier")
    annotation_count: int = Field(default=1, description="Number of annotations completed")


class BillingRuleRequest(BaseModel):
    """Request model for setting billing rules."""
    tenant_id: str = Field(..., description="Tenant identifier")
    billing_mode: BillingMode = Field(default=BillingMode.BY_COUNT, description="Billing mode")
    rate_per_annotation: float = Field(default=0.10, description="Cost per annotation")
    rate_per_hour: float = Field(default=50.00, description="Cost per hour")
    project_annual_fee: float = Field(default=10000.00, description="Annual project fee")


class BillingReportRequest(BaseModel):
    """Request model for generating billing reports."""
    tenant_id: str = Field(..., description="Tenant identifier")
    start_date: date = Field(..., description="Report start date")
    end_date: date = Field(..., description="Report end date")


class BillingAnalysisResponse(BaseModel):
    """Response model for billing analysis."""
    tenant_id: str
    period: str
    total_cost: float
    total_annotations: int
    total_time_spent: int
    average_cost_per_annotation: float
    average_cost_per_hour: float
    top_users: List[Dict[str, Any]]
    cost_trend: List[Dict[str, Any]]


@router.post("/track")
async def track_annotation_work(request: TrackAnnotationRequest) -> Dict[str, Any]:
    """
    Track annotation work for billing.
    
    Records time spent and annotations completed for billing calculation.
    """
    try:
        success = billing_system.track_annotation_time(
            user_id=request.user_id,
            task_id=request.task_id,
            duration=request.duration,
            tenant_id=request.tenant_id,
            annotation_count=request.annotation_count
        )
        
        if success:
            return {
                "status": "success",
                "message": "Annotation work tracked successfully",
                "tracked_at": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to track annotation work")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error tracking annotation work: {str(e)}")


@router.post("/rules")
async def set_billing_rule(request: BillingRuleRequest) -> Dict[str, Any]:
    """
    Set billing rule for a tenant.
    
    Configures how billing is calculated for the tenant.
    """
    try:
        from decimal import Decimal
        
        rule = BillingRule(
            tenant_id=request.tenant_id,
            billing_mode=request.billing_mode,
            rate_per_annotation=Decimal(str(request.rate_per_annotation)),
            rate_per_hour=Decimal(str(request.rate_per_hour)),
            project_annual_fee=Decimal(str(request.project_annual_fee))
        )
        
        billing_system.set_billing_rule(request.tenant_id, rule)
        
        return {
            "status": "success",
            "message": "Billing rule set successfully",
            "tenant_id": request.tenant_id,
            "billing_mode": request.billing_mode.value
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting billing rule: {str(e)}")


@router.get("/rules/{tenant_id}")
async def get_billing_rule(tenant_id: str) -> Dict[str, Any]:
    """
    Get billing rule for a tenant.
    
    Returns the current billing configuration for the tenant.
    """
    try:
        rule = billing_system.get_billing_rule(tenant_id)
        
        return {
            "tenant_id": rule.tenant_id,
            "billing_mode": rule.billing_mode.value,
            "rate_per_annotation": float(rule.rate_per_annotation),
            "rate_per_hour": float(rule.rate_per_hour),
            "project_annual_fee": float(rule.project_annual_fee)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting billing rule: {str(e)}")


@router.get("/bill/{tenant_id}/{month}")
async def get_monthly_bill(tenant_id: str, month: str) -> Dict[str, Any]:
    """
    Get monthly bill for a tenant.
    
    Calculates and returns the bill for the specified month (YYYY-MM format).
    """
    try:
        bill = billing_system.calculate_monthly_bill(tenant_id, month)
        return bill.to_dict()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating monthly bill: {str(e)}")


@router.post("/report")
async def generate_billing_report(request: BillingReportRequest) -> Dict[str, Any]:
    """
    Generate billing report for a date range.
    
    Provides detailed billing analysis and breakdown.
    """
    try:
        report = billing_system.generate_report(
            tenant_id=request.tenant_id,
            start_date=request.start_date,
            end_date=request.end_date
        )
        return report.to_dict()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating billing report: {str(e)}")


@router.get("/records/{tenant_id}")
async def get_billing_records(
    tenant_id: str,
    start_date: Optional[date] = Query(None, description="Start date filter"),
    end_date: Optional[date] = Query(None, description="End date filter")
) -> Dict[str, Any]:
    """
    Get billing records for a tenant.
    
    Returns list of billing records with optional date filtering.
    """
    try:
        records = billing_system.get_tenant_billing_records(
            tenant_id=tenant_id,
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            "tenant_id": tenant_id,
            "record_count": len(records),
            "records": [record.to_dict() for record in records]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting billing records: {str(e)}")


@router.get("/summary/{tenant_id}/{user_id}")
async def get_user_billing_summary(
    tenant_id: str,
    user_id: str,
    start_date: Optional[date] = Query(None, description="Start date filter"),
    end_date: Optional[date] = Query(None, description="End date filter")
) -> Dict[str, Any]:
    """
    Get billing summary for a specific user.
    
    Returns aggregated billing statistics for the user.
    """
    try:
        summary = billing_system.get_user_billing_summary(
            tenant_id=tenant_id,
            user_id=user_id,
            start_date=start_date,
            end_date=end_date
        )
        return summary
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting user billing summary: {str(e)}")


@router.get("/export/{tenant_id}")
async def export_billing_data(
    tenant_id: str,
    format_type: str = Query("json", description="Export format (json, csv)"),
    start_date: Optional[date] = Query(None, description="Start date filter"),
    end_date: Optional[date] = Query(None, description="End date filter")
) -> Dict[str, Any]:
    """
    Export billing data in specified format.
    
    Exports billing records for analysis or external processing.
    """
    try:
        export_result = billing_system.export_billing_data(
            tenant_id=tenant_id,
            format_type=format_type,
            start_date=start_date,
            end_date=end_date
        )
        return export_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting billing data: {str(e)}")


@router.get("/analysis/{tenant_id}")
async def get_billing_analysis(
    tenant_id: str,
    start_date: Optional[date] = Query(None, description="Analysis start date"),
    end_date: Optional[date] = Query(None, description="Analysis end date")
) -> BillingAnalysisResponse:
    """
    Get advanced billing analysis and trends.
    
    Provides comprehensive billing analytics including trends and top users.
    """
    try:
        # Default to last 30 days if no dates provided
        if not end_date:
            end_date = date.today()
        if not start_date:
            from datetime import timedelta
            start_date = end_date - timedelta(days=30)
        
        # Generate report
        report = billing_system.generate_report(tenant_id, start_date, end_date)
        
        # Calculate additional analytics
        total_cost = float(report.total_cost)
        total_annotations = report.total_annotations
        total_time_hours = report.total_time_spent / 3600.0 if report.total_time_spent > 0 else 0
        
        avg_cost_per_annotation = total_cost / total_annotations if total_annotations > 0 else 0
        avg_cost_per_hour = total_cost / total_time_hours if total_time_hours > 0 else 0
        
        # Get top users by cost
        top_users = []
        for user_id, stats in report.user_breakdown.items():
            top_users.append({
                "user_id": user_id,
                "cost": stats["cost"],
                "annotations": stats["annotations"],
                "time_spent": stats["time_spent"]
            })
        top_users.sort(key=lambda x: x["cost"], reverse=True)
        top_users = top_users[:10]  # Top 10 users
        
        # Generate cost trend (daily)
        cost_trend = []
        for day_str, stats in report.daily_breakdown.items():
            cost_trend.append({
                "date": day_str,
                "cost": stats["cost"],
                "annotations": stats["annotations"]
            })
        cost_trend.sort(key=lambda x: x["date"])
        
        return BillingAnalysisResponse(
            tenant_id=tenant_id,
            period=f"{start_date.isoformat()} to {end_date.isoformat()}",
            total_cost=total_cost,
            total_annotations=total_annotations,
            total_time_spent=report.total_time_spent,
            average_cost_per_annotation=avg_cost_per_annotation,
            average_cost_per_hour=avg_cost_per_hour,
            top_users=top_users,
            cost_trend=cost_trend
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating billing analysis: {str(e)}")


@router.get("/health")
async def billing_health_check() -> Dict[str, Any]:
    """
    Health check endpoint for billing service.
    
    Returns service status and basic statistics.
    """
    return {
        "status": "healthy",
        "service": "billing",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


# Analytics endpoints
@router.get("/analytics/trends/{tenant_id}")
async def get_cost_trends(
    tenant_id: str,
    days: int = Query(30, description="Number of days to analyze")
) -> Dict[str, Any]:
    """
    Get cost trends analysis for a tenant.
    
    Provides trend analysis over specified number of days.
    """
    try:
        trends = billing_analytics.calculate_cost_trends(tenant_id, days)
        return trends
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating cost trends: {str(e)}")


@router.get("/analytics/productivity/{tenant_id}")
async def get_user_productivity(
    tenant_id: str,
    days: int = Query(30, description="Number of days to analyze")
) -> Dict[str, Any]:
    """
    Get user productivity analysis for a tenant.
    
    Provides productivity metrics and efficiency rankings.
    """
    try:
        productivity = billing_analytics.analyze_user_productivity(tenant_id, days)
        return productivity
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing user productivity: {str(e)}")


@router.get("/analytics/forecast/{tenant_id}/{target_month}")
async def get_cost_forecast(tenant_id: str, target_month: str) -> Dict[str, Any]:
    """
    Get cost forecast for a target month.
    
    Provides cost prediction based on historical trends.
    """
    try:
        forecast = billing_analytics.forecast_monthly_cost(tenant_id, target_month)
        return forecast
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating cost forecast: {str(e)}")


@router.post("/analytics/compare/{tenant_id}")
async def compare_billing_periods(
    tenant_id: str,
    period1_start: date = Query(..., description="First period start date"),
    period1_end: date = Query(..., description="First period end date"),
    period2_start: date = Query(..., description="Second period start date"),
    period2_end: date = Query(..., description="Second period end date")
) -> Dict[str, Any]:
    """
    Compare billing metrics between two periods.
    
    Provides detailed comparison analysis.
    """
    try:
        comparison = billing_analytics.compare_periods(
            tenant_id, period1_start, period1_end, period2_start, period2_end
        )
        return comparison
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing billing periods: {str(e)}")


@router.get("/analytics/recommendations/{tenant_id}")
async def get_optimization_recommendations(
    tenant_id: str,
    days: int = Query(30, description="Number of days to analyze")
) -> Dict[str, Any]:
    """
    Get cost optimization recommendations.

    Provides actionable recommendations for cost reduction.
    """
    try:
        recommendations = billing_analytics.generate_cost_optimization_recommendations(tenant_id, days)
        return recommendations

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


# ============================================================================
# Enhanced Report Service Endpoints
# ============================================================================

from src.billing.report_service import (
    BillingReportService,
    ReportType,
    get_billing_report_service
)
from decimal import Decimal

# Initialize enhanced report service
report_service = get_billing_report_service()


class EnhancedReportRequest(BaseModel):
    """Request model for enhanced billing reports."""
    tenant_id: str = Field(..., description="Tenant identifier")
    start_date: date = Field(..., description="Report start date")
    end_date: date = Field(..., description="Report end date")
    report_type: str = Field(default="detailed", description="Report type")
    user_names: Optional[Dict[str, str]] = Field(None, description="User name mappings")
    project_names: Optional[Dict[str, str]] = Field(None, description="Project name mappings")
    department_names: Optional[Dict[str, str]] = Field(None, description="Department name mappings")


class BillingRuleVersionRequest(BaseModel):
    """Request model for creating billing rule versions."""
    tenant_id: str = Field(..., description="Tenant identifier")
    billing_mode: BillingMode = Field(default=BillingMode.BY_COUNT, description="Billing mode")
    rate_per_annotation: float = Field(default=0.10, description="Cost per annotation")
    rate_per_hour: float = Field(default=50.00, description="Cost per hour")
    project_annual_fee: float = Field(default=10000.00, description="Annual project fee")
    created_by: str = Field(..., description="User creating the rule")


class ApproveRuleRequest(BaseModel):
    """Request model for approving billing rules."""
    approved_by: str = Field(..., description="User approving the rule")


class ProjectMappingRequest(BaseModel):
    """Request model for configuring project mappings."""
    tenant_id: str = Field(..., description="Tenant identifier")
    mappings: Dict[str, str] = Field(..., description="Task ID to Project ID mappings")


class DepartmentMappingRequest(BaseModel):
    """Request model for configuring department mappings."""
    tenant_id: str = Field(..., description="Tenant identifier")
    project_mappings: Dict[str, str] = Field(..., description="Project ID to Department ID mappings")
    user_mappings: Dict[str, str] = Field(..., description="User ID to Department ID mappings")


@router.post("/enhanced-report")
async def generate_enhanced_report(request: EnhancedReportRequest) -> Dict[str, Any]:
    """
    Generate enhanced billing report with advanced analytics.

    Includes project breakdown, department allocation, and work hours statistics.
    """
    try:
        report_type = ReportType(request.report_type) if request.report_type else ReportType.DETAILED

        report = report_service.generate_enhanced_report(
            tenant_id=request.tenant_id,
            start_date=request.start_date,
            end_date=request.end_date,
            report_type=report_type,
            user_names=request.user_names,
            project_names=request.project_names,
            department_names=request.department_names
        )

        return report.to_dict()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating enhanced report: {str(e)}")


@router.get("/work-hours/{tenant_id}")
async def get_work_hours_statistics(
    tenant_id: str,
    start_date: date = Query(..., description="Report start date"),
    end_date: date = Query(..., description="Report end date")
) -> Dict[str, Any]:
    """
    Get work hours statistics for all users.

    Provides detailed work hours breakdown per user with efficiency scores.
    """
    try:
        statistics = report_service.generate_work_hours_report(
            tenant_id=tenant_id,
            start_date=start_date,
            end_date=end_date
        )

        return {
            "tenant_id": tenant_id,
            "period": f"{start_date.isoformat()} to {end_date.isoformat()}",
            "user_count": len(statistics),
            "statistics": [
                {
                    "user_id": s.user_id,
                    "user_name": s.user_name,
                    "total_hours": s.total_hours,
                    "billable_hours": s.billable_hours,
                    "total_annotations": s.total_annotations,
                    "annotations_per_hour": s.annotations_per_hour,
                    "total_cost": float(s.total_cost),
                    "efficiency_score": s.efficiency_score
                }
                for s in statistics
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting work hours statistics: {str(e)}")


@router.get("/project-breakdown/{tenant_id}")
async def get_project_cost_breakdown(
    tenant_id: str,
    start_date: date = Query(..., description="Report start date"),
    end_date: date = Query(..., description="Report end date")
) -> Dict[str, Any]:
    """
    Get cost breakdown by project.

    Provides detailed cost allocation per project.
    """
    try:
        breakdowns = report_service.generate_project_cost_breakdown(
            tenant_id=tenant_id,
            start_date=start_date,
            end_date=end_date
        )

        total_cost = sum(float(b.total_cost) for b in breakdowns)

        return {
            "tenant_id": tenant_id,
            "period": f"{start_date.isoformat()} to {end_date.isoformat()}",
            "total_cost": total_cost,
            "project_count": len(breakdowns),
            "breakdowns": [
                {
                    "project_id": b.project_id,
                    "project_name": b.project_name,
                    "total_cost": float(b.total_cost),
                    "total_annotations": b.total_annotations,
                    "total_time_spent": b.total_time_spent,
                    "user_count": b.user_count,
                    "avg_cost_per_annotation": float(b.avg_cost_per_annotation),
                    "percentage_of_total": b.percentage_of_total
                }
                for b in breakdowns
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting project breakdown: {str(e)}")


@router.get("/department-allocation/{tenant_id}")
async def get_department_cost_allocation(
    tenant_id: str,
    start_date: date = Query(..., description="Report start date"),
    end_date: date = Query(..., description="Report end date")
) -> Dict[str, Any]:
    """
    Get cost allocation by department.

    Provides cost distribution across departments.
    """
    try:
        allocations = report_service.generate_department_cost_allocation(
            tenant_id=tenant_id,
            start_date=start_date,
            end_date=end_date
        )

        total_cost = sum(float(a.total_cost) for a in allocations)

        return {
            "tenant_id": tenant_id,
            "period": f"{start_date.isoformat()} to {end_date.isoformat()}",
            "total_cost": total_cost,
            "department_count": len(allocations),
            "allocations": [
                {
                    "department_id": a.department_id,
                    "department_name": a.department_name,
                    "total_cost": float(a.total_cost),
                    "projects": a.projects,
                    "user_count": a.user_count,
                    "percentage_of_total": a.percentage_of_total
                }
                for a in allocations
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting department allocation: {str(e)}")


@router.post("/rules/versions")
async def create_billing_rule_version(request: BillingRuleVersionRequest) -> Dict[str, Any]:
    """
    Create a new versioned billing rule.

    Creates a new version of billing rule with audit trail.
    """
    try:
        rule = report_service.create_billing_rule_version(
            tenant_id=request.tenant_id,
            billing_mode=request.billing_mode,
            rate_per_annotation=Decimal(str(request.rate_per_annotation)),
            rate_per_hour=Decimal(str(request.rate_per_hour)),
            created_by=request.created_by,
            project_annual_fee=Decimal(str(request.project_annual_fee))
        )

        return {
            "status": "success",
            "message": f"Created billing rule version {rule.version}",
            "rule": rule.to_dict()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating billing rule version: {str(e)}")


@router.post("/rules/versions/{tenant_id}/{version}/approve")
async def approve_billing_rule_version(
    tenant_id: str,
    version: int,
    request: ApproveRuleRequest
) -> Dict[str, Any]:
    """
    Approve a billing rule version.

    Approves a pending billing rule version.
    """
    try:
        rule = report_service.approve_billing_rule(
            tenant_id=tenant_id,
            version=version,
            approved_by=request.approved_by
        )

        if rule:
            return {
                "status": "success",
                "message": f"Approved billing rule version {version}",
                "rule": rule.to_dict()
            }
        else:
            raise HTTPException(status_code=404, detail=f"Billing rule version {version} not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error approving billing rule: {str(e)}")


@router.get("/rules/versions/{tenant_id}")
async def get_billing_rule_history(tenant_id: str) -> Dict[str, Any]:
    """
    Get billing rule version history.

    Returns all versions of billing rules for the tenant.
    """
    try:
        versions = report_service.get_billing_rule_history(tenant_id)
        active_rule = report_service.get_active_billing_rule(tenant_id)

        return {
            "tenant_id": tenant_id,
            "active_version": active_rule.version if active_rule else None,
            "version_count": len(versions),
            "versions": [v.to_dict() for v in versions]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting billing rule history: {str(e)}")


@router.post("/mappings/projects")
async def configure_project_mappings(request: ProjectMappingRequest) -> Dict[str, Any]:
    """
    Configure task-to-project mappings.

    Sets up mappings for project-based cost breakdown.
    """
    try:
        report_service.configure_project_mapping(
            tenant_id=request.tenant_id,
            mappings=request.mappings
        )

        return {
            "status": "success",
            "message": f"Configured {len(request.mappings)} project mappings",
            "tenant_id": request.tenant_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error configuring project mappings: {str(e)}")


@router.post("/mappings/departments")
async def configure_department_mappings(request: DepartmentMappingRequest) -> Dict[str, Any]:
    """
    Configure department mappings.

    Sets up mappings for department-based cost allocation.
    """
    try:
        report_service.configure_department_mapping(
            tenant_id=request.tenant_id,
            project_mappings=request.project_mappings,
            user_mappings=request.user_mappings
        )

        return {
            "status": "success",
            "message": "Configured department mappings",
            "tenant_id": request.tenant_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error configuring department mappings: {str(e)}")


@router.get("/export-excel/{tenant_id}")
async def export_billing_to_excel(
    tenant_id: str,
    start_date: date = Query(..., description="Report start date"),
    end_date: date = Query(..., description="Report end date"),
    report_type: str = Query("detailed", description="Report type")
) -> Dict[str, Any]:
    """
    Prepare billing data for Excel export.

    Returns structured data ready for Excel file generation.
    """
    try:
        rtype = ReportType(report_type) if report_type else ReportType.DETAILED

        report = report_service.generate_enhanced_report(
            tenant_id=tenant_id,
            start_date=start_date,
            end_date=end_date,
            report_type=rtype
        )

        excel_data = report_service.export_to_excel_data(report)

        return {
            "tenant_id": tenant_id,
            "period": f"{start_date.isoformat()} to {end_date.isoformat()}",
            "sheets": excel_data,
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing Excel export: {str(e)}")