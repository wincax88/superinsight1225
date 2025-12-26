"""
Quality Analysis API for SuperInsight Platform.

Provides REST API endpoints for quality trend analysis,
auto-retraining, and quality-driven billing.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.quality.trend_analyzer import QualityTrendAnalyzer
from src.quality.auto_retrain import AutoRetrainTrigger, RetrainTriggerType, RetrainJobStatus
from src.billing.quality_pricing import QualityPricingEngine, DifficultyLevel
from src.billing.incentive_manager import IncentiveManager


router = APIRouter(prefix="/api/v1/quality", tags=["quality-analysis"])

# Global instances - use lazy initialization
_trend_analyzer: Optional[QualityTrendAnalyzer] = None
_retrain_trigger: Optional[AutoRetrainTrigger] = None
_pricing_engine: Optional[QualityPricingEngine] = None
_incentive_manager: Optional[IncentiveManager] = None


def get_trend_analyzer() -> QualityTrendAnalyzer:
    """Get or create trend analyzer instance."""
    global _trend_analyzer
    if _trend_analyzer is None:
        _trend_analyzer = QualityTrendAnalyzer()
    return _trend_analyzer


def get_retrain_trigger() -> AutoRetrainTrigger:
    """Get or create retrain trigger instance."""
    global _retrain_trigger
    if _retrain_trigger is None:
        _retrain_trigger = AutoRetrainTrigger()
    return _retrain_trigger


def get_pricing_engine() -> QualityPricingEngine:
    """Get or create pricing engine instance."""
    global _pricing_engine
    if _pricing_engine is None:
        _pricing_engine = QualityPricingEngine()
    return _pricing_engine


def get_incentive_manager() -> IncentiveManager:
    """Get or create incentive manager instance."""
    global _incentive_manager
    if _incentive_manager is None:
        _incentive_manager = IncentiveManager()
    return _incentive_manager


# ==================== Request/Response Models ====================

class TrendAnalysisRequest(BaseModel):
    """Request model for trend analysis."""
    tenant_id: Optional[str] = Field(None, description="Tenant filter")
    days: int = Field(30, ge=1, le=365, description="Analysis period in days")
    granularity: str = Field("daily", description="Time granularity (daily, weekly)")


class QualityDeclineRequest(BaseModel):
    """Request model for quality decline detection."""
    tenant_id: Optional[str] = Field(None, description="Tenant filter")
    threshold: float = Field(0.1, ge=0.01, le=0.5, description="Decline threshold")
    window_days: int = Field(7, ge=1, le=30, description="Comparison window")


class QualityPredictionRequest(BaseModel):
    """Request model for quality prediction."""
    tenant_id: Optional[str] = Field(None, description="Tenant filter")
    days_ahead: int = Field(7, ge=1, le=30, description="Days to predict")
    history_days: int = Field(30, ge=7, le=90, description="Historical data period")


class RetrainConditionRequest(BaseModel):
    """Request model for checking retrain conditions."""
    model_id: str = Field(..., description="Model identifier")
    current_metrics: Dict[str, float] = Field(..., description="Current quality metrics")


class TriggerRetrainRequest(BaseModel):
    """Request model for triggering retraining."""
    model_id: str = Field(..., description="Model to retrain")
    reason: str = Field(..., min_length=5, description="Reason for retraining")
    trigger_type: str = Field("manual", description="Trigger type")
    current_metrics: Optional[Dict[str, float]] = Field(None, description="Current metrics")


class CompleteRetrainRequest(BaseModel):
    """Request model for completing retraining."""
    result_metrics: Dict[str, float] = Field(..., description="Post-training metrics")
    success: bool = Field(True, description="Training success status")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class QualityCostRequest(BaseModel):
    """Request model for quality-adjusted cost calculation."""
    base_cost: float = Field(..., gt=0, description="Base cost")
    quality_score: float = Field(..., ge=0, le=1, description="Quality score")
    difficulty_level: str = Field("standard", description="Difficulty level")


class BatchCostRequest(BaseModel):
    """Request model for batch cost calculation."""
    items: List[Dict[str, Any]] = Field(..., description="List of items to calculate")


class IncentiveCalculationRequest(BaseModel):
    """Request model for incentive calculation."""
    user_id: str = Field(..., description="User identifier")
    period: str = Field(..., description="Billing period (YYYY-MM)")
    performance_metrics: Dict[str, Any] = Field(..., description="Performance metrics")
    base_earnings: float = Field(..., gt=0, description="Base earnings")


class PenaltyRequest(BaseModel):
    """Request model for applying penalties."""
    user_id: str = Field(..., description="User identifier")
    violations: List[Dict[str, Any]] = Field(..., description="List of violations")


# ==================== Trend Analysis Endpoints ====================

@router.post("/trends/analyze", response_model=Dict[str, Any])
async def analyze_quality_trends(request: TrendAnalysisRequest) -> Dict[str, Any]:
    """
    Analyze quality trends over a specified period.

    Returns trend direction, daily/weekly metrics, and severity distribution.
    """
    try:
        analyzer = get_trend_analyzer()

        result = await analyzer.analyze_trends(
            tenant_id=request.tenant_id,
            days=request.days,
            granularity=request.granularity
        )

        return {
            "status": "success",
            "analysis": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze trends: {str(e)}")


@router.post("/trends/decline", response_model=Dict[str, Any])
async def detect_quality_decline(request: QualityDeclineRequest) -> Dict[str, Any]:
    """
    Detect quality decline alerts.

    Compares recent period with previous period to identify declining trends.
    """
    try:
        analyzer = get_trend_analyzer()

        alerts = await analyzer.detect_quality_decline(
            tenant_id=request.tenant_id,
            threshold=request.threshold,
            window_days=request.window_days
        )

        return {
            "status": "success",
            "alerts": alerts,
            "alert_count": len(alerts),
            "has_critical": any(a.get("severity") == "critical" for a in alerts)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to detect decline: {str(e)}")


@router.post("/trends/predict", response_model=Dict[str, Any])
async def predict_quality(request: QualityPredictionRequest) -> Dict[str, Any]:
    """
    Predict future quality metrics.

    Uses historical data to forecast quality trends.
    """
    try:
        analyzer = get_trend_analyzer()

        prediction = await analyzer.predict_quality(
            tenant_id=request.tenant_id,
            days_ahead=request.days_ahead,
            history_days=request.history_days
        )

        return {
            "status": "success",
            "prediction": prediction
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to predict quality: {str(e)}")


@router.get("/trends/root-causes", response_model=Dict[str, Any])
async def get_root_causes(
    tenant_id: Optional[str] = Query(None, description="Tenant filter"),
    days: int = Query(30, ge=1, le=365, description="Analysis period"),
    min_occurrences: int = Query(3, ge=1, description="Minimum occurrences")
) -> Dict[str, Any]:
    """
    Identify root causes of quality issues.

    Analyzes issue patterns to determine primary causes.
    """
    try:
        analyzer = get_trend_analyzer()

        root_causes = await analyzer.identify_root_causes(
            tenant_id=tenant_id,
            days=days,
            min_occurrences=min_occurrences
        )

        return {
            "status": "success",
            "root_causes": root_causes,
            "cause_count": len(root_causes)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to identify root causes: {str(e)}")


# ==================== Auto-Retrain Endpoints ====================

@router.post("/retrain/check", response_model=Dict[str, Any])
async def check_retrain_conditions(request: RetrainConditionRequest) -> Dict[str, Any]:
    """
    Check if retraining conditions are met for a model.

    Evaluates current metrics against thresholds.
    """
    try:
        trigger = get_retrain_trigger()

        result = await trigger.check_retrain_conditions(
            model_id=request.model_id,
            current_metrics=request.current_metrics
        )

        return {
            "status": "success",
            "result": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check conditions: {str(e)}")


@router.post("/retrain/trigger", response_model=Dict[str, Any])
async def trigger_retraining(request: TriggerRetrainRequest) -> Dict[str, Any]:
    """
    Trigger a model retraining job.

    Creates a new retraining job and queues it for processing.
    """
    try:
        trigger = get_retrain_trigger()

        trigger_type = RetrainTriggerType(request.trigger_type)

        job = await trigger.trigger_retrain(
            model_id=request.model_id,
            reason=request.reason,
            trigger_type=trigger_type,
            current_metrics=request.current_metrics
        )

        return {
            "status": "success",
            "job_id": str(job.id),
            "model_id": job.model_id,
            "message": f"Retraining triggered: {job.trigger_reason}"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger retraining: {str(e)}")


@router.get("/retrain/jobs", response_model=Dict[str, Any])
async def list_retrain_jobs(
    model_id: Optional[str] = Query(None, description="Filter by model"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200, description="Max results")
) -> Dict[str, Any]:
    """
    List retraining jobs.
    """
    try:
        trigger = get_retrain_trigger()

        status_filter = RetrainJobStatus(status) if status else None

        jobs = await trigger.list_jobs(
            model_id=model_id,
            status=status_filter,
            limit=limit
        )

        return {
            "status": "success",
            "jobs": jobs,
            "count": len(jobs)
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")


@router.get("/retrain/jobs/{job_id}", response_model=Dict[str, Any])
async def get_retrain_job(job_id: UUID) -> Dict[str, Any]:
    """
    Get retraining job details.
    """
    try:
        trigger = get_retrain_trigger()

        job = await trigger.get_job_status(job_id)

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        return {
            "status": "success",
            "job": job
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job: {str(e)}")


@router.put("/retrain/jobs/{job_id}/complete", response_model=Dict[str, Any])
async def complete_retrain_job(
    job_id: UUID,
    request: CompleteRetrainRequest
) -> Dict[str, Any]:
    """
    Mark a retraining job as complete and validate results.
    """
    try:
        trigger = get_retrain_trigger()

        success = await trigger.complete_retraining(
            job_id=job_id,
            result_metrics=request.result_metrics,
            success=request.success,
            error_message=request.error_message
        )

        if not success:
            raise HTTPException(status_code=400, detail="Failed to complete job")

        # Validate effect
        validation = await trigger.validate_retrain_effect(job_id)

        return {
            "status": "success",
            "validation": validation
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to complete job: {str(e)}")


@router.post("/retrain/jobs/{job_id}/cancel", response_model=Dict[str, Any])
async def cancel_retrain_job(
    job_id: UUID,
    reason: str = Query(..., description="Cancellation reason")
) -> Dict[str, Any]:
    """
    Cancel a pending or running retraining job.
    """
    try:
        trigger = get_retrain_trigger()

        success = await trigger.cancel_job(job_id, reason)

        if not success:
            raise HTTPException(status_code=400, detail="Cannot cancel job")

        return {
            "status": "success",
            "message": "Job cancelled"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")


@router.get("/retrain/statistics", response_model=Dict[str, Any])
async def get_retrain_statistics(
    days: int = Query(30, ge=1, le=365, description="Analysis period")
) -> Dict[str, Any]:
    """
    Get retraining statistics.
    """
    try:
        trigger = get_retrain_trigger()

        stats = await trigger.get_retrain_statistics(days=days)

        return {
            "status": "success",
            "statistics": stats
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


# ==================== Quality Pricing Endpoints ====================

@router.post("/pricing/calculate", response_model=Dict[str, Any])
async def calculate_quality_cost(request: QualityCostRequest) -> Dict[str, Any]:
    """
    Calculate quality-adjusted cost.

    Applies quality and difficulty multipliers to base cost.
    """
    try:
        engine = get_pricing_engine()
        from decimal import Decimal

        difficulty = DifficultyLevel(request.difficulty_level)

        result = engine.calculate_quality_adjusted_cost(
            base_cost=Decimal(str(request.base_cost)),
            quality_score=request.quality_score,
            difficulty_level=difficulty
        )

        return {
            "status": "success",
            "calculation": result
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate cost: {str(e)}")


@router.post("/pricing/batch", response_model=Dict[str, Any])
async def calculate_batch_cost(request: BatchCostRequest) -> Dict[str, Any]:
    """
    Calculate costs for a batch of items.
    """
    try:
        engine = get_pricing_engine()

        result = engine.calculate_batch_cost(request.items)

        return {
            "status": "success",
            "result": result
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate batch: {str(e)}")


@router.get("/pricing/config", response_model=Dict[str, Any])
async def get_pricing_config() -> Dict[str, Any]:
    """
    Get current pricing configuration.
    """
    try:
        engine = get_pricing_engine()

        config = engine.get_pricing_config()

        return {
            "status": "success",
            "config": config
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get config: {str(e)}")


# ==================== Incentive Endpoints ====================

@router.post("/incentives/calculate", response_model=Dict[str, Any])
async def calculate_incentives(request: IncentiveCalculationRequest) -> Dict[str, Any]:
    """
    Calculate incentives for a user.
    """
    try:
        manager = get_incentive_manager()
        from decimal import Decimal

        incentives = await manager.calculate_incentives(
            user_id=request.user_id,
            period=request.period,
            performance_metrics=request.performance_metrics,
            base_earnings=Decimal(str(request.base_earnings))
        )

        return {
            "status": "success",
            "incentives": [i.to_dict() for i in incentives],
            "total_count": len(incentives),
            "total_amount": sum(float(i.amount) for i in incentives)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate incentives: {str(e)}")


@router.post("/penalties/apply", response_model=Dict[str, Any])
async def apply_penalties(request: PenaltyRequest) -> Dict[str, Any]:
    """
    Apply penalties based on violations.
    """
    try:
        manager = get_incentive_manager()

        penalties = await manager.apply_penalties(
            user_id=request.user_id,
            violations=request.violations
        )

        return {
            "status": "success",
            "penalties": [p.to_dict() for p in penalties],
            "total_count": len(penalties),
            "total_amount": sum(float(p.amount) for p in penalties)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply penalties: {str(e)}")


@router.get("/incentives/user/{user_id}", response_model=Dict[str, Any])
async def get_user_incentives(
    user_id: str,
    period: Optional[str] = Query(None, description="Billing period (YYYY-MM)")
) -> Dict[str, Any]:
    """
    Get incentive summary for a user.
    """
    try:
        manager = get_incentive_manager()

        summary = await manager.get_user_incentive_summary(user_id, period)

        return {
            "status": "success",
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get incentives: {str(e)}")


@router.get("/incentives/tenant/{tenant_id}", response_model=Dict[str, Any])
async def get_tenant_incentives(
    tenant_id: str,
    period: str = Query(..., description="Billing period (YYYY-MM)")
) -> Dict[str, Any]:
    """
    Get incentive summary for a tenant.
    """
    try:
        manager = get_incentive_manager()

        summary = await manager.get_tenant_incentive_summary(tenant_id, period)

        return {
            "status": "success",
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get incentives: {str(e)}")


@router.get("/incentives/rules", response_model=Dict[str, Any])
async def get_incentive_rules() -> Dict[str, Any]:
    """
    Get current incentive and penalty rules.
    """
    try:
        manager = get_incentive_manager()

        config = manager.get_rules_config()

        return {
            "status": "success",
            "rules": config
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get rules: {str(e)}")


# ==================== Health Check ====================

@router.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """
    Health check for quality analysis service.
    """
    return {
        "status": "healthy",
        "service": "quality-analysis",
        "components": {
            "trend_analyzer": "available",
            "retrain_trigger": "available",
            "pricing_engine": "available",
            "incentive_manager": "available"
        },
        "timestamp": datetime.now().isoformat()
    }
