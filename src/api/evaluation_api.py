"""
Performance Evaluation API for SuperInsight Platform.

Provides REST API endpoints for performance assessment, appeals,
and reporting operations.
"""

from datetime import datetime, date
from typing import Dict, Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.evaluation.models import (
    PerformanceStatus,
    AppealStatus,
    PerformancePeriod,
)
from src.evaluation.performance import PerformanceEngine
from src.evaluation.appeal import AppealManager
from src.evaluation.report_generator import ReportGenerator


router = APIRouter(prefix="/api/v1/evaluation", tags=["evaluation"])

# Global instances - use lazy initialization
_performance_engine: Optional[PerformanceEngine] = None
_appeal_manager: Optional[AppealManager] = None
_report_generator: Optional[ReportGenerator] = None


def get_performance_engine() -> PerformanceEngine:
    """Get or create performance engine instance."""
    global _performance_engine
    if _performance_engine is None:
        _performance_engine = PerformanceEngine()
    return _performance_engine


def get_appeal_manager() -> AppealManager:
    """Get or create appeal manager instance."""
    global _appeal_manager
    if _appeal_manager is None:
        _appeal_manager = AppealManager()
    return _appeal_manager


def get_report_generator() -> ReportGenerator:
    """Get or create report generator instance."""
    global _report_generator
    if _report_generator is None:
        _report_generator = ReportGenerator()
    return _report_generator


# ==================== Request/Response Models ====================

class CalculatePerformanceRequest(BaseModel):
    """Request model for calculating performance."""
    user_id: str = Field(..., description="User identifier")
    period_start: date = Field(..., description="Period start date")
    period_end: date = Field(..., description="Period end date")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")
    period_type: str = Field("monthly", description="Period type (daily, weekly, monthly, quarterly)")


class SubmitAppealRequest(BaseModel):
    """Request model for submitting an appeal."""
    performance_record_id: UUID = Field(..., description="Performance record ID")
    user_id: str = Field(..., description="User submitting the appeal")
    appeal_type: str = Field(..., description="Type of appeal")
    reason: str = Field(..., min_length=10, description="Detailed reason for appeal")
    disputed_fields: Optional[List[str]] = Field(None, description="List of disputed fields")
    supporting_evidence: Optional[Dict[str, Any]] = Field(None, description="Supporting evidence")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")


class ReviewAppealRequest(BaseModel):
    """Request model for reviewing an appeal."""
    reviewer_id: str = Field(..., description="User reviewing the appeal")
    decision: str = Field(..., description="Decision: 'approved' or 'rejected'")
    review_notes: Optional[str] = Field(None, description="Review notes")
    adjusted_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Adjusted score if approved")
    adjustment_reason: Optional[str] = Field(None, description="Reason for adjustment")


class GenerateReportRequest(BaseModel):
    """Request model for generating reports."""
    user_id: Optional[str] = Field(None, description="User ID for individual report")
    tenant_id: Optional[str] = Field(None, description="Tenant ID for team report")
    period_start: date = Field(..., description="Report period start")
    period_end: date = Field(..., description="Report period end")
    report_type: str = Field("individual", description="Report type: individual, team, comparison, trend")
    user_ids: Optional[List[str]] = Field(None, description="User IDs for comparison report")


# ==================== Performance Endpoints ====================

@router.post("/performance/calculate", response_model=Dict[str, Any])
async def calculate_performance(request: CalculatePerformanceRequest) -> Dict[str, Any]:
    """
    Calculate performance for a user over a period.

    Computes multi-dimensional performance score based on quality,
    efficiency, compliance, and improvement metrics.
    """
    try:
        engine = get_performance_engine()

        period_type = PerformancePeriod(request.period_type)

        record = await engine.calculate_performance(
            user_id=request.user_id,
            period_start=request.period_start,
            period_end=request.period_end,
            tenant_id=request.tenant_id,
            period_type=period_type,
        )

        return {
            "status": "success",
            "performance": record.to_dict(),
            "message": f"Performance calculated: {record.overall_score:.3f}"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate performance: {str(e)}")


@router.get("/performance/{user_id}", response_model=Dict[str, Any])
async def get_user_performance(
    user_id: str,
    period_start: Optional[date] = Query(None, description="Period start filter"),
    period_end: Optional[date] = Query(None, description="Period end filter"),
    tenant_id: Optional[str] = Query(None, description="Tenant filter")
) -> Dict[str, Any]:
    """
    Get performance records for a user.

    Returns performance history with optional period filtering.
    """
    try:
        engine = get_performance_engine()

        history = await engine.get_user_performance_history(user_id)

        return {
            "status": "success",
            "user_id": user_id,
            "records": history,
            "count": len(history)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance: {str(e)}")


@router.get("/performance/{user_id}/regression", response_model=Dict[str, Any])
async def detect_performance_regression(
    user_id: str,
    periods: int = Query(3, ge=2, le=12, description="Periods to analyze"),
    threshold: float = Query(0.1, ge=0.01, le=0.5, description="Decline threshold")
) -> Dict[str, Any]:
    """
    Detect performance regression for a user.

    Analyzes recent periods for declining performance trends.
    """
    try:
        engine = get_performance_engine()

        regressions = await engine.detect_regression(user_id, periods, threshold)

        return {
            "status": "success",
            "user_id": user_id,
            "regressions_detected": len(regressions) > 0,
            "regressions": regressions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to detect regression: {str(e)}")


@router.get("/ranking", response_model=Dict[str, Any])
async def get_performance_ranking(
    tenant_id: Optional[str] = Query(None, description="Tenant filter"),
    period_start: Optional[date] = Query(None, description="Period start"),
    period_end: Optional[date] = Query(None, description="Period end"),
    limit: int = Query(10, ge=1, le=100, description="Max results")
) -> Dict[str, Any]:
    """
    Get performance ranking.

    Returns ranked list of users by overall performance score.
    """
    try:
        engine = get_performance_engine()

        ranking = await engine.get_performance_ranking(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
            limit=limit
        )

        return {
            "status": "success",
            "ranking": ranking,
            "count": len(ranking)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get ranking: {str(e)}")


# ==================== Appeal Endpoints ====================

@router.post("/appeals", response_model=Dict[str, Any])
async def submit_appeal(request: SubmitAppealRequest) -> Dict[str, Any]:
    """
    Submit a performance evaluation appeal.

    Creates an appeal for review with supporting evidence.
    """
    try:
        manager = get_appeal_manager()

        appeal = await manager.submit_appeal(
            performance_record_id=request.performance_record_id,
            user_id=request.user_id,
            appeal_type=request.appeal_type,
            reason=request.reason,
            disputed_fields=request.disputed_fields,
            supporting_evidence=request.supporting_evidence,
            tenant_id=request.tenant_id,
        )

        return {
            "status": "success",
            "appeal": appeal.to_dict(),
            "message": f"Appeal submitted: {appeal.id}"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit appeal: {str(e)}")


@router.get("/appeals", response_model=Dict[str, Any])
async def list_appeals(
    user_id: Optional[str] = Query(None, description="Filter by user"),
    tenant_id: Optional[str] = Query(None, description="Filter by tenant"),
    status: Optional[str] = Query(None, description="Filter by status"),
    appeal_type: Optional[str] = Query(None, description="Filter by type"),
    limit: int = Query(50, ge=1, le=200, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset")
) -> Dict[str, Any]:
    """
    List appeals with optional filters.
    """
    try:
        manager = get_appeal_manager()

        status_enum = AppealStatus(status) if status else None

        appeals, total = await manager.list_appeals(
            user_id=user_id,
            tenant_id=tenant_id,
            status=status_enum,
            appeal_type=appeal_type,
            limit=limit,
            offset=offset
        )

        return {
            "status": "success",
            "appeals": [a.to_dict() for a in appeals],
            "total": total,
            "limit": limit,
            "offset": offset
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list appeals: {str(e)}")


@router.get("/appeals/{appeal_id}", response_model=Dict[str, Any])
async def get_appeal(appeal_id: UUID) -> Dict[str, Any]:
    """
    Get an appeal by ID.
    """
    try:
        manager = get_appeal_manager()

        appeal = await manager.get_appeal(appeal_id)

        if not appeal:
            raise HTTPException(status_code=404, detail="Appeal not found")

        return {
            "status": "success",
            "appeal": appeal.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get appeal: {str(e)}")


@router.put("/appeals/{appeal_id}/review", response_model=Dict[str, Any])
async def review_appeal(
    appeal_id: UUID,
    request: ReviewAppealRequest
) -> Dict[str, Any]:
    """
    Review an appeal and make a decision.

    Approves or rejects the appeal with optional score adjustment.
    """
    try:
        manager = get_appeal_manager()

        success = await manager.review_appeal(
            appeal_id=appeal_id,
            reviewer_id=request.reviewer_id,
            decision=request.decision,
            review_notes=request.review_notes,
            adjusted_score=request.adjusted_score,
            adjustment_reason=request.adjustment_reason
        )

        if not success:
            raise HTTPException(status_code=400, detail="Failed to review appeal")

        return {
            "status": "success",
            "message": f"Appeal {request.decision}"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to review appeal: {str(e)}")


@router.post("/appeals/{appeal_id}/withdraw", response_model=Dict[str, Any])
async def withdraw_appeal(
    appeal_id: UUID,
    user_id: str = Query(..., description="User withdrawing the appeal")
) -> Dict[str, Any]:
    """
    Withdraw an appeal.
    """
    try:
        manager = get_appeal_manager()

        success = await manager.withdraw_appeal(appeal_id, user_id)

        if not success:
            raise HTTPException(status_code=400, detail="Failed to withdraw appeal")

        return {
            "status": "success",
            "message": "Appeal withdrawn"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to withdraw appeal: {str(e)}")


@router.get("/appeals/pending", response_model=Dict[str, Any])
async def get_pending_appeals(
    reviewer_id: Optional[str] = Query(None, description="Filter by reviewer"),
    tenant_id: Optional[str] = Query(None, description="Filter by tenant")
) -> Dict[str, Any]:
    """
    Get pending appeals for review.
    """
    try:
        manager = get_appeal_manager()

        pending = await manager.get_pending_appeals(reviewer_id, tenant_id)

        return {
            "status": "success",
            "pending_appeals": pending,
            "count": len(pending)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pending appeals: {str(e)}")


@router.get("/appeals/statistics", response_model=Dict[str, Any])
async def get_appeal_statistics(
    tenant_id: Optional[str] = Query(None, description="Filter by tenant"),
    days: int = Query(30, ge=1, le=365, description="Analysis period")
) -> Dict[str, Any]:
    """
    Get appeal statistics.
    """
    try:
        manager = get_appeal_manager()

        stats = await manager.get_appeal_statistics(tenant_id, days)

        return {
            "status": "success",
            "statistics": stats
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


# ==================== Report Endpoints ====================

@router.post("/reports/generate", response_model=Dict[str, Any])
async def generate_report(request: GenerateReportRequest) -> Dict[str, Any]:
    """
    Generate a performance report.

    Supports individual, team, comparison, and trend reports.
    """
    try:
        generator = get_report_generator()

        if request.report_type == "individual":
            if not request.user_id:
                raise HTTPException(status_code=400, detail="user_id required for individual report")
            report = await generator.generate_individual_report(
                request.user_id,
                request.period_start,
                request.period_end,
                request.tenant_id
            )
        elif request.report_type == "team":
            if not request.tenant_id:
                raise HTTPException(status_code=400, detail="tenant_id required for team report")
            report = await generator.generate_team_report(
                request.tenant_id,
                request.period_start,
                request.period_end
            )
        elif request.report_type == "comparison":
            if not request.user_ids or len(request.user_ids) < 2:
                raise HTTPException(status_code=400, detail="At least 2 user_ids required for comparison")
            report = await generator.generate_comparison_report(
                request.user_ids,
                request.period_start,
                request.period_end,
                request.tenant_id
            )
        elif request.report_type == "trend":
            if not request.user_id:
                raise HTTPException(status_code=400, detail="user_id required for trend report")
            report = await generator.generate_trend_report(
                request.user_id,
                tenant_id=request.tenant_id
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown report type: {request.report_type}")

        if "error" in report:
            raise HTTPException(status_code=400, detail=report["error"])

        return {
            "status": "success",
            "report": report
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


@router.get("/reports/individual/{user_id}", response_model=Dict[str, Any])
async def get_individual_report(
    user_id: str,
    period_start: date = Query(..., description="Period start"),
    period_end: date = Query(..., description="Period end"),
    tenant_id: Optional[str] = Query(None, description="Tenant filter")
) -> Dict[str, Any]:
    """
    Generate individual performance report.
    """
    try:
        generator = get_report_generator()

        report = await generator.generate_individual_report(
            user_id, period_start, period_end, tenant_id
        )

        if "error" in report:
            raise HTTPException(status_code=400, detail=report["error"])

        return {
            "status": "success",
            "report": report
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


@router.get("/reports/team/{tenant_id}", response_model=Dict[str, Any])
async def get_team_report(
    tenant_id: str,
    period_start: date = Query(..., description="Period start"),
    period_end: date = Query(..., description="Period end")
) -> Dict[str, Any]:
    """
    Generate team performance report.
    """
    try:
        generator = get_report_generator()

        report = await generator.generate_team_report(
            tenant_id, period_start, period_end
        )

        if "error" in report:
            raise HTTPException(status_code=400, detail=report["error"])

        return {
            "status": "success",
            "report": report
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


# ==================== Health Check ====================

@router.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """
    Health check for evaluation service.
    """
    return {
        "status": "healthy",
        "service": "performance-evaluation",
        "timestamp": datetime.now().isoformat()
    }
