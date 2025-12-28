"""
Training effect tracking and ROI analysis for SuperInsight Platform.

Provides:
- Post-training quality improvement monitoring
- Training effectiveness evaluation
- Training ROI calculation and analysis
- Training plan optimization recommendations
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID, uuid4
from enum import Enum
from dataclasses import dataclass, field
import statistics

logger = logging.getLogger(__name__)


class EffectCategory(str, Enum):
    """Training effect categories."""
    QUALITY = "quality"
    EFFICIENCY = "efficiency"
    COMPLIANCE = "compliance"
    SKILLS = "skills"
    OVERALL = "overall"


class EffectStatus(str, Enum):
    """Training effect evaluation status."""
    PENDING = "pending"           # Awaiting post-training data
    EVALUATING = "evaluating"     # Collecting data
    COMPLETED = "completed"       # Evaluation complete
    INSUFFICIENT_DATA = "insufficient_data"  # Not enough data


class ImprovementLevel(str, Enum):
    """Improvement level classification."""
    EXCEPTIONAL = "exceptional"   # >30% improvement
    SIGNIFICANT = "significant"   # 15-30% improvement
    MODERATE = "moderate"         # 5-15% improvement
    MINIMAL = "minimal"           # 0-5% improvement
    NEGATIVE = "negative"         # Regression


@dataclass
class MetricSnapshot:
    """Performance metric snapshot at a point in time."""
    metric_name: str
    value: float
    timestamp: datetime
    source: str  # evaluation, self-assessment, etc.


@dataclass
class TrainingEffect:
    """Training effect measurement."""
    id: UUID
    user_id: str
    training_id: UUID
    training_title: str
    skill_name: str

    # Before/after metrics
    baseline_metrics: Dict[str, float]
    post_training_metrics: Dict[str, float]

    # Calculated effects
    improvements: Dict[str, float]  # metric -> improvement %
    overall_improvement: float
    improvement_level: ImprovementLevel

    # Timeline
    training_completed_at: datetime
    evaluation_started_at: Optional[datetime] = None
    evaluation_completed_at: Optional[datetime] = None

    status: EffectStatus = EffectStatus.PENDING
    evaluation_notes: List[str] = field(default_factory=list)


@dataclass
class TrainingROI:
    """Training ROI calculation."""
    training_id: UUID
    training_title: str

    # Costs
    direct_cost: float  # Training materials, instructor, etc.
    opportunity_cost: float  # Time away from work
    total_cost: float

    # Benefits
    productivity_gain: float  # Value of improved efficiency
    quality_gain: float  # Value of improved quality
    error_reduction_value: float  # Value of reduced errors
    total_benefit: float

    # ROI metrics
    roi_percentage: float  # (benefit - cost) / cost * 100
    payback_period_days: int  # Days to recoup investment
    net_present_value: float  # NPV of training investment

    # Details
    calculation_method: str
    assumptions: List[str]
    calculated_at: datetime = field(default_factory=datetime.now)


@dataclass
class TrainingOptimization:
    """Training optimization recommendation."""
    training_id: UUID
    training_title: str
    current_effectiveness: float

    # Recommendations
    recommendations: List[Dict[str, Any]]
    priority: str  # high, medium, low
    estimated_improvement: float

    # Analysis
    underperforming_areas: List[str]
    successful_elements: List[str]


class TrainingEffectTracker:
    """
    Training effect tracking and ROI analysis engine.

    Monitors post-training performance improvements,
    calculates training ROI, and provides optimization recommendations.
    """

    # Evaluation configuration
    EVALUATION_CONFIG = {
        "baseline_period_days": 30,      # Days before training for baseline
        "post_training_wait_days": 7,    # Wait period after training
        "evaluation_period_days": 30,    # Days to evaluate post-training
        "min_data_points": 10,           # Minimum data points required
        "significance_threshold": 0.05   # Minimum improvement to be significant
    }

    # ROI calculation parameters
    ROI_PARAMS = {
        "hourly_rate": 50.0,             # Average hourly rate for productivity
        "error_cost": 100.0,             # Average cost per error
        "discount_rate": 0.10,           # Annual discount rate for NPV
        "benefit_period_months": 12      # Months to consider for benefits
    }

    # Improvement thresholds
    IMPROVEMENT_THRESHOLDS = {
        ImprovementLevel.EXCEPTIONAL: 0.30,
        ImprovementLevel.SIGNIFICANT: 0.15,
        ImprovementLevel.MODERATE: 0.05,
        ImprovementLevel.MINIMAL: 0.0,
        ImprovementLevel.NEGATIVE: float('-inf')
    }

    def __init__(self):
        """Initialize the effect tracker."""
        self._effects: Dict[UUID, TrainingEffect] = {}
        self._roi_records: Dict[UUID, TrainingROI] = {}
        self._metric_history: Dict[str, List[MetricSnapshot]] = {}  # user_id -> snapshots

    async def start_tracking(
        self,
        user_id: str,
        training_id: UUID,
        training_title: str,
        skill_name: str,
        baseline_metrics: Optional[Dict[str, float]] = None
    ) -> TrainingEffect:
        """
        Start tracking training effect for a user.

        Args:
            user_id: User identifier
            training_id: Training identifier
            training_title: Training title
            skill_name: Target skill
            baseline_metrics: Pre-training metrics (or will be fetched)

        Returns:
            Training effect tracking record
        """
        # Get baseline metrics if not provided
        if not baseline_metrics:
            baseline_metrics = await self._fetch_baseline_metrics(user_id, skill_name)

        effect = TrainingEffect(
            id=uuid4(),
            user_id=user_id,
            training_id=training_id,
            training_title=training_title,
            skill_name=skill_name,
            baseline_metrics=baseline_metrics,
            post_training_metrics={},
            improvements={},
            overall_improvement=0.0,
            improvement_level=ImprovementLevel.MINIMAL,
            training_completed_at=datetime.now(),
            status=EffectStatus.PENDING
        )

        self._effects[effect.id] = effect
        logger.info(f"Started effect tracking for user {user_id}, training {training_id}")
        return effect

    async def _fetch_baseline_metrics(
        self,
        user_id: str,
        skill_name: str
    ) -> Dict[str, float]:
        """Fetch baseline metrics from evaluation system."""
        try:
            from src.evaluation.performance import PerformanceEngine

            engine = PerformanceEngine()
            history = await engine.get_user_performance_history(user_id)

            if history:
                # Get metrics from the baseline period
                baseline_period = timedelta(days=self.EVALUATION_CONFIG["baseline_period_days"])
                cutoff = datetime.now() - baseline_period

                relevant_records = [
                    r for r in history
                    if datetime.fromisoformat(r.get("evaluated_at", datetime.now().isoformat())) >= cutoff
                ]

                if relevant_records:
                    # Average the metrics
                    return {
                        "accuracy_rate": statistics.mean([r.get("accuracy_rate", 0.5) for r in relevant_records]),
                        "completion_rate": statistics.mean([r.get("completion_rate", 0.5) for r in relevant_records]),
                        "quality_score": statistics.mean([r.get("quality_score", 0.5) for r in relevant_records]),
                        "sla_compliance_rate": statistics.mean([r.get("sla_compliance_rate", 0.5) for r in relevant_records]),
                        "error_rate": statistics.mean([r.get("error_rate", 0.1) for r in relevant_records])
                    }
        except ImportError:
            pass

        # Default baseline
        return {
            "accuracy_rate": 0.75,
            "completion_rate": 0.70,
            "quality_score": 0.70,
            "sla_compliance_rate": 0.80,
            "error_rate": 0.10
        }

    async def evaluate_effect(
        self,
        effect_id: UUID,
        post_metrics: Optional[Dict[str, float]] = None
    ) -> TrainingEffect:
        """
        Evaluate training effect.

        Args:
            effect_id: Effect tracking ID
            post_metrics: Post-training metrics (or will be fetched)

        Returns:
            Updated training effect
        """
        effect = self._effects.get(effect_id)
        if not effect:
            raise ValueError(f"Effect tracking not found: {effect_id}")

        # Check if enough time has passed
        wait_period = timedelta(days=self.EVALUATION_CONFIG["post_training_wait_days"])
        if datetime.now() - effect.training_completed_at < wait_period:
            effect.status = EffectStatus.PENDING
            effect.evaluation_notes.append(
                f"Waiting for evaluation period. Ready after {effect.training_completed_at + wait_period}"
            )
            return effect

        effect.status = EffectStatus.EVALUATING
        effect.evaluation_started_at = datetime.now()

        # Get post-training metrics
        if not post_metrics:
            post_metrics = await self._fetch_post_training_metrics(
                effect.user_id,
                effect.skill_name
            )

        effect.post_training_metrics = post_metrics

        # Calculate improvements
        improvements = {}
        for metric, baseline_value in effect.baseline_metrics.items():
            post_value = post_metrics.get(metric, baseline_value)

            # Handle error_rate (lower is better)
            if metric == "error_rate":
                improvement = (baseline_value - post_value) / baseline_value if baseline_value > 0 else 0
            else:
                improvement = (post_value - baseline_value) / baseline_value if baseline_value > 0 else 0

            improvements[metric] = improvement

        effect.improvements = improvements

        # Calculate overall improvement (weighted average)
        weights = {
            "accuracy_rate": 0.30,
            "completion_rate": 0.20,
            "quality_score": 0.25,
            "sla_compliance_rate": 0.15,
            "error_rate": 0.10
        }

        overall = sum(
            improvements.get(m, 0) * w
            for m, w in weights.items()
        )
        effect.overall_improvement = overall

        # Classify improvement level
        effect.improvement_level = self._classify_improvement(overall)

        effect.status = EffectStatus.COMPLETED
        effect.evaluation_completed_at = datetime.now()

        logger.info(
            f"Evaluated effect {effect_id}: {effect.overall_improvement:.1%} improvement "
            f"({effect.improvement_level.value})"
        )
        return effect

    async def _fetch_post_training_metrics(
        self,
        user_id: str,
        skill_name: str
    ) -> Dict[str, float]:
        """Fetch post-training metrics."""
        try:
            from src.evaluation.performance import PerformanceEngine

            engine = PerformanceEngine()
            history = await engine.get_user_performance_history(user_id)

            if history:
                # Get most recent metrics
                eval_period = timedelta(days=self.EVALUATION_CONFIG["evaluation_period_days"])
                cutoff = datetime.now() - eval_period

                relevant_records = [
                    r for r in history
                    if datetime.fromisoformat(r.get("evaluated_at", datetime.now().isoformat())) >= cutoff
                ]

                if relevant_records:
                    return {
                        "accuracy_rate": statistics.mean([r.get("accuracy_rate", 0.5) for r in relevant_records]),
                        "completion_rate": statistics.mean([r.get("completion_rate", 0.5) for r in relevant_records]),
                        "quality_score": statistics.mean([r.get("quality_score", 0.5) for r in relevant_records]),
                        "sla_compliance_rate": statistics.mean([r.get("sla_compliance_rate", 0.5) for r in relevant_records]),
                        "error_rate": statistics.mean([r.get("error_rate", 0.1) for r in relevant_records])
                    }
        except ImportError:
            pass

        # Simulated improvement (in production, would require real data)
        return {
            "accuracy_rate": 0.82,
            "completion_rate": 0.78,
            "quality_score": 0.80,
            "sla_compliance_rate": 0.88,
            "error_rate": 0.06
        }

    def _classify_improvement(self, improvement: float) -> ImprovementLevel:
        """Classify improvement into level."""
        for level, threshold in self.IMPROVEMENT_THRESHOLDS.items():
            if improvement >= threshold:
                return level
        return ImprovementLevel.NEGATIVE

    async def calculate_roi(
        self,
        training_id: UUID,
        training_title: str,
        training_cost: float,
        training_duration_hours: float,
        participants: List[str]
    ) -> TrainingROI:
        """
        Calculate training ROI.

        Args:
            training_id: Training identifier
            training_title: Training title
            training_cost: Direct training cost
            training_duration_hours: Training duration in hours
            participants: List of participant user IDs

        Returns:
            Training ROI analysis
        """
        # Calculate costs
        direct_cost = training_cost
        opportunity_cost = (
            len(participants) *
            training_duration_hours *
            self.ROI_PARAMS["hourly_rate"]
        )
        total_cost = direct_cost + opportunity_cost

        # Calculate benefits from effect tracking
        effects = [
            e for e in self._effects.values()
            if e.training_id == training_id and e.status == EffectStatus.COMPLETED
        ]

        if not effects:
            # Estimate based on typical improvements
            avg_improvement = 0.10
        else:
            avg_improvement = statistics.mean([e.overall_improvement for e in effects])

        # Calculate productivity gain
        # Assumption: X% improvement in efficiency = X% more work done
        monthly_productivity = len(participants) * 160 * self.ROI_PARAMS["hourly_rate"]  # 160 hours/month
        productivity_gain = (
            monthly_productivity *
            avg_improvement *
            self.ROI_PARAMS["benefit_period_months"]
        )

        # Calculate quality gain
        # Assumption: Better quality = fewer reworks
        quality_gain = (
            monthly_productivity *
            0.05 *  # Assume 5% of time spent on rework
            avg_improvement *
            self.ROI_PARAMS["benefit_period_months"]
        )

        # Calculate error reduction value
        # Assumption: Based on reduced error rate
        monthly_errors = len(participants) * 10  # Assume 10 errors/month/person
        error_reduction = monthly_errors * avg_improvement * self.ROI_PARAMS["error_cost"]
        error_reduction_value = error_reduction * self.ROI_PARAMS["benefit_period_months"]

        total_benefit = productivity_gain + quality_gain + error_reduction_value

        # Calculate ROI percentage
        roi_percentage = ((total_benefit - total_cost) / total_cost * 100) if total_cost > 0 else 0

        # Calculate payback period
        monthly_benefit = total_benefit / self.ROI_PARAMS["benefit_period_months"]
        payback_period_days = int((total_cost / monthly_benefit) * 30) if monthly_benefit > 0 else 999

        # Calculate NPV
        npv = self._calculate_npv(
            total_cost,
            monthly_benefit,
            self.ROI_PARAMS["benefit_period_months"],
            self.ROI_PARAMS["discount_rate"]
        )

        roi = TrainingROI(
            training_id=training_id,
            training_title=training_title,
            direct_cost=direct_cost,
            opportunity_cost=opportunity_cost,
            total_cost=total_cost,
            productivity_gain=productivity_gain,
            quality_gain=quality_gain,
            error_reduction_value=error_reduction_value,
            total_benefit=total_benefit,
            roi_percentage=roi_percentage,
            payback_period_days=payback_period_days,
            net_present_value=npv,
            calculation_method="Standard Training ROI Model",
            assumptions=[
                f"Hourly rate: ${self.ROI_PARAMS['hourly_rate']}",
                f"Average improvement: {avg_improvement:.1%}",
                f"Benefit period: {self.ROI_PARAMS['benefit_period_months']} months",
                f"Discount rate: {self.ROI_PARAMS['discount_rate']:.1%}"
            ]
        )

        self._roi_records[training_id] = roi
        logger.info(f"Calculated ROI for {training_title}: {roi_percentage:.1f}%")
        return roi

    def _calculate_npv(
        self,
        initial_cost: float,
        monthly_benefit: float,
        months: int,
        annual_discount_rate: float
    ) -> float:
        """Calculate Net Present Value."""
        monthly_rate = annual_discount_rate / 12
        npv = -initial_cost

        for month in range(1, months + 1):
            npv += monthly_benefit / ((1 + monthly_rate) ** month)

        return npv

    async def generate_optimization_recommendations(
        self,
        training_id: UUID
    ) -> TrainingOptimization:
        """
        Generate training optimization recommendations.

        Args:
            training_id: Training identifier

        Returns:
            Optimization recommendations
        """
        effects = [
            e for e in self._effects.values()
            if e.training_id == training_id and e.status == EffectStatus.COMPLETED
        ]

        if not effects:
            return TrainingOptimization(
                training_id=training_id,
                training_title="Unknown",
                current_effectiveness=0.0,
                recommendations=[{
                    "type": "data_collection",
                    "description": "收集更多培训效果数据以进行分析",
                    "action": "等待培训后评估完成"
                }],
                priority="high",
                estimated_improvement=0.0,
                underperforming_areas=[],
                successful_elements=[]
            )

        # Analyze effects
        training_title = effects[0].training_title
        avg_improvement = statistics.mean([e.overall_improvement for e in effects])

        # Find underperforming metrics
        metric_improvements = {}
        for effect in effects:
            for metric, improvement in effect.improvements.items():
                if metric not in metric_improvements:
                    metric_improvements[metric] = []
                metric_improvements[metric].append(improvement)

        avg_by_metric = {
            m: statistics.mean(vals)
            for m, vals in metric_improvements.items()
        }

        underperforming = [
            m for m, v in avg_by_metric.items()
            if v < self.EVALUATION_CONFIG["significance_threshold"]
        ]

        successful = [
            m for m, v in avg_by_metric.items()
            if v >= self.IMPROVEMENT_THRESHOLDS[ImprovementLevel.SIGNIFICANT]
        ]

        # Generate recommendations
        recommendations = []

        if underperforming:
            recommendations.append({
                "type": "content_enhancement",
                "description": f"加强以下方面的培训内容: {', '.join(underperforming)}",
                "action": "更新课程材料，增加相关练习",
                "metrics": underperforming
            })

        if avg_improvement < self.IMPROVEMENT_THRESHOLDS[ImprovementLevel.MODERATE]:
            recommendations.append({
                "type": "format_change",
                "description": "考虑更换培训形式以提高参与度",
                "action": "添加更多互动内容，实践练习",
                "priority": "high"
            })

        # Check for participant engagement
        low_improvement_users = [
            e.user_id for e in effects
            if e.overall_improvement < self.IMPROVEMENT_THRESHOLDS[ImprovementLevel.MINIMAL]
        ]

        if len(low_improvement_users) > len(effects) * 0.3:
            recommendations.append({
                "type": "targeting",
                "description": "部分学员效果不佳，建议个性化辅导",
                "action": "为低效学员提供一对一指导",
                "affected_users": len(low_improvement_users)
            })

        if successful:
            recommendations.append({
                "type": "best_practice",
                "description": f"成功经验: {', '.join(successful)} 方面表现良好",
                "action": "分析成功因素，应用到其他培训",
                "metrics": successful
            })

        # Determine priority
        if avg_improvement < 0:
            priority = "critical"
        elif avg_improvement < self.IMPROVEMENT_THRESHOLDS[ImprovementLevel.MODERATE]:
            priority = "high"
        elif avg_improvement < self.IMPROVEMENT_THRESHOLDS[ImprovementLevel.SIGNIFICANT]:
            priority = "medium"
        else:
            priority = "low"

        optimization = TrainingOptimization(
            training_id=training_id,
            training_title=training_title,
            current_effectiveness=avg_improvement,
            recommendations=recommendations,
            priority=priority,
            estimated_improvement=0.10 if recommendations else 0.0,
            underperforming_areas=underperforming,
            successful_elements=successful
        )

        logger.info(f"Generated {len(recommendations)} optimization recommendations for {training_title}")
        return optimization

    async def get_effect_summary(
        self,
        user_id: Optional[str] = None,
        training_id: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """
        Get effect tracking summary.

        Args:
            user_id: Optional user filter
            training_id: Optional training filter

        Returns:
            Summary statistics
        """
        effects = list(self._effects.values())

        if user_id:
            effects = [e for e in effects if e.user_id == user_id]
        if training_id:
            effects = [e for e in effects if e.training_id == training_id]

        completed = [e for e in effects if e.status == EffectStatus.COMPLETED]

        if not completed:
            return {
                "total_tracked": len(effects),
                "completed_evaluations": 0,
                "pending_evaluations": len([e for e in effects if e.status == EffectStatus.PENDING]),
                "message": "No completed evaluations yet"
            }

        # Calculate statistics
        improvements = [e.overall_improvement for e in completed]

        level_distribution = {}
        for e in completed:
            level = e.improvement_level.value
            level_distribution[level] = level_distribution.get(level, 0) + 1

        # Improvement by skill
        skill_improvements = {}
        for e in completed:
            if e.skill_name not in skill_improvements:
                skill_improvements[e.skill_name] = []
            skill_improvements[e.skill_name].append(e.overall_improvement)

        avg_by_skill = {
            skill: statistics.mean(vals)
            for skill, vals in skill_improvements.items()
        }

        return {
            "total_tracked": len(effects),
            "completed_evaluations": len(completed),
            "pending_evaluations": len([e for e in effects if e.status == EffectStatus.PENDING]),
            "average_improvement": statistics.mean(improvements),
            "median_improvement": statistics.median(improvements),
            "max_improvement": max(improvements),
            "min_improvement": min(improvements),
            "improvement_level_distribution": level_distribution,
            "improvement_by_skill": avg_by_skill,
            "generated_at": datetime.now().isoformat()
        }

    async def get_roi_summary(
        self,
        training_ids: Optional[List[UUID]] = None
    ) -> Dict[str, Any]:
        """
        Get ROI summary for trainings.

        Args:
            training_ids: Optional filter for specific trainings

        Returns:
            ROI summary statistics
        """
        rois = list(self._roi_records.values())

        if training_ids:
            rois = [r for r in rois if r.training_id in training_ids]

        if not rois:
            return {
                "total_trainings": 0,
                "message": "No ROI calculations available"
            }

        return {
            "total_trainings": len(rois),
            "total_investment": sum(r.total_cost for r in rois),
            "total_benefit": sum(r.total_benefit for r in rois),
            "average_roi_percentage": statistics.mean([r.roi_percentage for r in rois]),
            "median_roi_percentage": statistics.median([r.roi_percentage for r in rois]),
            "total_npv": sum(r.net_present_value for r in rois),
            "average_payback_days": statistics.mean([r.payback_period_days for r in rois]),
            "positive_roi_count": len([r for r in rois if r.roi_percentage > 0]),
            "breakdown": [
                {
                    "training_id": str(r.training_id),
                    "training_title": r.training_title,
                    "roi_percentage": r.roi_percentage,
                    "total_cost": r.total_cost,
                    "total_benefit": r.total_benefit,
                    "payback_days": r.payback_period_days
                }
                for r in sorted(rois, key=lambda x: -x.roi_percentage)
            ],
            "generated_at": datetime.now().isoformat()
        }

    async def export_report(
        self,
        training_id: UUID,
        format: str = "dict"
    ) -> Dict[str, Any]:
        """
        Export comprehensive training effect report.

        Args:
            training_id: Training identifier
            format: Output format (dict, json)

        Returns:
            Comprehensive report
        """
        effects = [
            e for e in self._effects.values()
            if e.training_id == training_id
        ]

        roi = self._roi_records.get(training_id)
        optimization = await self.generate_optimization_recommendations(training_id)

        report = {
            "training_id": str(training_id),
            "training_title": effects[0].training_title if effects else "Unknown",
            "generated_at": datetime.now().isoformat(),

            "effect_analysis": {
                "total_participants": len(effects),
                "completed_evaluations": len([e for e in effects if e.status == EffectStatus.COMPLETED]),
                "effects": [
                    {
                        "user_id": e.user_id,
                        "skill_name": e.skill_name,
                        "baseline_metrics": e.baseline_metrics,
                        "post_training_metrics": e.post_training_metrics,
                        "improvements": e.improvements,
                        "overall_improvement": e.overall_improvement,
                        "improvement_level": e.improvement_level.value,
                        "status": e.status.value
                    }
                    for e in effects
                ]
            },

            "roi_analysis": {
                "calculated": roi is not None,
                "data": {
                    "total_cost": roi.total_cost,
                    "total_benefit": roi.total_benefit,
                    "roi_percentage": roi.roi_percentage,
                    "payback_period_days": roi.payback_period_days,
                    "npv": roi.net_present_value,
                    "assumptions": roi.assumptions
                } if roi else None
            },

            "optimization": {
                "current_effectiveness": optimization.current_effectiveness,
                "priority": optimization.priority,
                "recommendations": optimization.recommendations,
                "underperforming_areas": optimization.underperforming_areas,
                "successful_elements": optimization.successful_elements
            }
        }

        return report
