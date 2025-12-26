"""
Auto-retraining trigger for SuperInsight Platform.

Provides:
- Automatic model retraining detection
- Retraining job management
- Effect validation
- Threshold-based triggers
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RetrainTriggerType(str, Enum):
    """Types of retraining triggers."""
    QUALITY_DECLINE = "quality_decline"
    ERROR_RATE_SPIKE = "error_rate_spike"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    DRIFT_DETECTED = "drift_detected"


class RetrainJobStatus(str, Enum):
    """Retraining job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    VALIDATING = "validating"


class RetrainJob(BaseModel):
    """Retraining job model."""
    id: UUID = Field(default_factory=uuid4)
    model_id: str
    trigger_type: RetrainTriggerType
    trigger_reason: str
    status: RetrainJobStatus = RetrainJobStatus.PENDING

    # Quality metrics at trigger time
    trigger_metrics: Dict[str, float] = Field(default_factory=dict)

    # Results after training
    result_metrics: Optional[Dict[str, float]] = None
    improvement: Optional[float] = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Error handling
    error_message: Optional[str] = None


class AutoRetrainTrigger:
    """
    Automatic retraining trigger engine.

    Monitors quality metrics and triggers model retraining
    when thresholds are exceeded.
    """

    # Threshold configuration
    THRESHOLDS = {
        "quality_score": 0.70,      # Retrain if quality drops below 70%
        "decline_rate": 0.15,       # Retrain if 15% decline detected
        "error_rate": 0.20,         # Retrain if error rate exceeds 20%
        "consistency_drop": 0.10,   # Retrain if consistency drops 10%
    }

    # Cooldown to prevent excessive retraining
    COOLDOWN_HOURS = 24  # Minimum hours between retraining triggers

    # Validation thresholds
    MIN_IMPROVEMENT = 0.05  # Minimum 5% improvement to consider success

    def __init__(self):
        """Initialize the auto-retrain trigger."""
        self._active_jobs: Dict[str, RetrainJob] = {}
        self._last_trigger_times: Dict[str, datetime] = {}

    async def check_retrain_conditions(
        self,
        model_id: str,
        current_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Check if retraining conditions are met.

        Args:
            model_id: Model identifier
            current_metrics: Current quality metrics

        Returns:
            Condition check results with recommendations
        """
        triggers = []
        should_retrain = False

        # Check quality score threshold
        quality_score = current_metrics.get("quality_score", 1.0)
        if quality_score < self.THRESHOLDS["quality_score"]:
            triggers.append({
                "type": RetrainTriggerType.QUALITY_DECLINE.value,
                "reason": f"Quality score {quality_score:.2f} below threshold {self.THRESHOLDS['quality_score']}",
                "severity": "high" if quality_score < 0.5 else "medium"
            })
            should_retrain = True

        # Check error rate threshold
        error_rate = current_metrics.get("error_rate", 0.0)
        if error_rate > self.THRESHOLDS["error_rate"]:
            triggers.append({
                "type": RetrainTriggerType.ERROR_RATE_SPIKE.value,
                "reason": f"Error rate {error_rate:.2f} exceeds threshold {self.THRESHOLDS['error_rate']}",
                "severity": "high" if error_rate > 0.3 else "medium"
            })
            should_retrain = True

        # Check decline rate
        decline_rate = current_metrics.get("decline_rate", 0.0)
        if decline_rate > self.THRESHOLDS["decline_rate"]:
            triggers.append({
                "type": RetrainTriggerType.DRIFT_DETECTED.value,
                "reason": f"Quality decline rate {decline_rate:.2f} exceeds threshold {self.THRESHOLDS['decline_rate']}",
                "severity": "medium"
            })
            should_retrain = True

        # Check cooldown
        in_cooldown = self._is_in_cooldown(model_id)
        if should_retrain and in_cooldown:
            should_retrain = False

        return {
            "model_id": model_id,
            "should_retrain": should_retrain,
            "triggers": triggers,
            "in_cooldown": in_cooldown,
            "cooldown_remaining": self._get_cooldown_remaining(model_id),
            "current_metrics": current_metrics,
            "thresholds": self.THRESHOLDS,
            "checked_at": datetime.now().isoformat()
        }

    async def trigger_retrain(
        self,
        model_id: str,
        reason: str,
        trigger_type: RetrainTriggerType = RetrainTriggerType.MANUAL,
        current_metrics: Optional[Dict[str, float]] = None
    ) -> RetrainJob:
        """
        Trigger a retraining job.

        Args:
            model_id: Model to retrain
            reason: Reason for retraining
            trigger_type: Type of trigger
            current_metrics: Current metrics at trigger time

        Returns:
            Created retraining job
        """
        # Check cooldown for non-manual triggers
        if trigger_type != RetrainTriggerType.MANUAL:
            if self._is_in_cooldown(model_id):
                raise ValueError(f"Model {model_id} is in cooldown period")

        # Check for existing active job
        if model_id in self._active_jobs:
            existing = self._active_jobs[model_id]
            if existing.status in [RetrainJobStatus.PENDING, RetrainJobStatus.RUNNING]:
                raise ValueError(f"Active retraining job exists: {existing.id}")

        # Create job
        job = RetrainJob(
            model_id=model_id,
            trigger_type=trigger_type,
            trigger_reason=reason,
            trigger_metrics=current_metrics or {}
        )

        self._active_jobs[model_id] = job
        self._last_trigger_times[model_id] = datetime.now()

        logger.info(f"Retraining triggered for model {model_id}: {reason}")

        # In a real implementation, this would:
        # 1. Submit job to training pipeline
        # 2. Queue for async processing
        # 3. Notify relevant stakeholders

        return job

    async def start_retraining(self, job_id: UUID) -> bool:
        """
        Start a pending retraining job.

        Args:
            job_id: Job UUID

        Returns:
            True if started successfully
        """
        # Find job
        job = None
        for j in self._active_jobs.values():
            if j.id == job_id:
                job = j
                break

        if not job:
            logger.error(f"Job not found: {job_id}")
            return False

        if job.status != RetrainJobStatus.PENDING:
            logger.error(f"Job not in pending state: {job.status}")
            return False

        job.status = RetrainJobStatus.RUNNING
        job.started_at = datetime.now()

        logger.info(f"Retraining started: {job_id}")

        # In production, this would trigger actual model training
        return True

    async def complete_retraining(
        self,
        job_id: UUID,
        result_metrics: Dict[str, float],
        success: bool = True,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Mark retraining as complete.

        Args:
            job_id: Job UUID
            result_metrics: Post-training metrics
            success: Whether training succeeded
            error_message: Error message if failed

        Returns:
            True if marked successfully
        """
        # Find job
        job = None
        for j in self._active_jobs.values():
            if j.id == job_id:
                job = j
                break

        if not job:
            return False

        job.result_metrics = result_metrics
        job.completed_at = datetime.now()

        if success:
            job.status = RetrainJobStatus.VALIDATING

            # Calculate improvement
            if job.trigger_metrics and result_metrics:
                before = job.trigger_metrics.get("quality_score", 0)
                after = result_metrics.get("quality_score", 0)
                job.improvement = after - before
        else:
            job.status = RetrainJobStatus.FAILED
            job.error_message = error_message

        return True

    async def validate_retrain_effect(self, job_id: UUID) -> Dict[str, Any]:
        """
        Validate the effect of retraining.

        Args:
            job_id: Job UUID

        Returns:
            Validation results
        """
        # Find job
        job = None
        for j in self._active_jobs.values():
            if j.id == job_id:
                job = j
                break

        if not job:
            return {"error": "Job not found"}

        if job.status != RetrainJobStatus.VALIDATING:
            return {"error": f"Job not in validating state: {job.status}"}

        validation = {
            "job_id": str(job_id),
            "model_id": job.model_id,
            "before_metrics": job.trigger_metrics,
            "after_metrics": job.result_metrics,
            "improvement": job.improvement,
            "success": False,
            "recommendations": []
        }

        # Evaluate improvement
        if job.improvement is not None:
            if job.improvement >= self.MIN_IMPROVEMENT:
                validation["success"] = True
                validation["message"] = f"Retraining improved quality by {job.improvement * 100:.1f}%"
                job.status = RetrainJobStatus.COMPLETED
            elif job.improvement >= 0:
                validation["success"] = True
                validation["message"] = f"Marginal improvement of {job.improvement * 100:.1f}%"
                validation["recommendations"].append(
                    "Consider additional training data or hyperparameter tuning"
                )
                job.status = RetrainJobStatus.COMPLETED
            else:
                validation["success"] = False
                validation["message"] = f"Quality regressed by {abs(job.improvement) * 100:.1f}%"
                validation["recommendations"].extend([
                    "Roll back to previous model version",
                    "Investigate training data quality",
                    "Review model architecture"
                ])
                job.status = RetrainJobStatus.FAILED
                job.error_message = "Quality regression detected"

        validation["validated_at"] = datetime.now().isoformat()
        return validation

    async def get_job_status(self, job_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get status of a retraining job.

        Args:
            job_id: Job UUID

        Returns:
            Job status details
        """
        for job in self._active_jobs.values():
            if job.id == job_id:
                return {
                    "id": str(job.id),
                    "model_id": job.model_id,
                    "status": job.status.value,
                    "trigger_type": job.trigger_type.value,
                    "trigger_reason": job.trigger_reason,
                    "trigger_metrics": job.trigger_metrics,
                    "result_metrics": job.result_metrics,
                    "improvement": job.improvement,
                    "created_at": job.created_at.isoformat(),
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    "error_message": job.error_message
                }
        return None

    async def list_jobs(
        self,
        model_id: Optional[str] = None,
        status: Optional[RetrainJobStatus] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        List retraining jobs.

        Args:
            model_id: Filter by model
            status: Filter by status
            limit: Max results

        Returns:
            List of jobs
        """
        jobs = list(self._active_jobs.values())

        if model_id:
            jobs = [j for j in jobs if j.model_id == model_id]

        if status:
            jobs = [j for j in jobs if j.status == status]

        # Sort by created_at descending
        jobs.sort(key=lambda x: x.created_at, reverse=True)

        return [
            {
                "id": str(j.id),
                "model_id": j.model_id,
                "status": j.status.value,
                "trigger_type": j.trigger_type.value,
                "improvement": j.improvement,
                "created_at": j.created_at.isoformat()
            }
            for j in jobs[:limit]
        ]

    async def cancel_job(self, job_id: UUID, reason: str) -> bool:
        """
        Cancel a pending or running job.

        Args:
            job_id: Job UUID
            reason: Cancellation reason

        Returns:
            True if cancelled
        """
        for job in self._active_jobs.values():
            if job.id == job_id:
                if job.status in [RetrainJobStatus.PENDING, RetrainJobStatus.RUNNING]:
                    job.status = RetrainJobStatus.CANCELLED
                    job.error_message = f"Cancelled: {reason}"
                    job.completed_at = datetime.now()
                    logger.info(f"Job cancelled: {job_id} - {reason}")
                    return True
                return False
        return False

    async def get_retrain_statistics(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get retraining statistics.

        Args:
            days: Analysis period

        Returns:
            Statistics dictionary
        """
        cutoff = datetime.now() - timedelta(days=days)

        jobs = [
            j for j in self._active_jobs.values()
            if j.created_at >= cutoff
        ]

        total = len(jobs)
        by_status = {}
        by_trigger = {}
        improvements = []

        for job in jobs:
            # By status
            status_key = job.status.value
            by_status[status_key] = by_status.get(status_key, 0) + 1

            # By trigger type
            trigger_key = job.trigger_type.value
            by_trigger[trigger_key] = by_trigger.get(trigger_key, 0) + 1

            # Collect improvements
            if job.improvement is not None:
                improvements.append(job.improvement)

        avg_improvement = sum(improvements) / len(improvements) if improvements else 0
        success_rate = by_status.get("completed", 0) / total if total > 0 else 0

        return {
            "period_days": days,
            "total_jobs": total,
            "by_status": by_status,
            "by_trigger_type": by_trigger,
            "avg_improvement": round(avg_improvement, 4),
            "success_rate": round(success_rate, 4),
            "failed_count": by_status.get("failed", 0),
            "generated_at": datetime.now().isoformat()
        }

    def update_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """
        Update retraining thresholds.

        Args:
            new_thresholds: New threshold values
        """
        for key, value in new_thresholds.items():
            if key in self.THRESHOLDS:
                self.THRESHOLDS[key] = value
                logger.info(f"Updated threshold {key}: {value}")

    def _is_in_cooldown(self, model_id: str) -> bool:
        """Check if model is in cooldown period."""
        if model_id not in self._last_trigger_times:
            return False

        elapsed = datetime.now() - self._last_trigger_times[model_id]
        return elapsed.total_seconds() < self.COOLDOWN_HOURS * 3600

    def _get_cooldown_remaining(self, model_id: str) -> Optional[int]:
        """Get remaining cooldown time in seconds."""
        if model_id not in self._last_trigger_times:
            return None

        elapsed = datetime.now() - self._last_trigger_times[model_id]
        remaining = self.COOLDOWN_HOURS * 3600 - elapsed.total_seconds()

        return max(0, int(remaining)) if remaining > 0 else None
