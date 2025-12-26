"""
Incentive management for SuperInsight Platform.

Provides:
- Quality-based bonuses
- Consistency rewards
- Improvement incentives
- Penalty management
"""

import logging
from datetime import datetime, timedelta, date
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class IncentiveType(str, Enum):
    """Types of incentives."""
    QUALITY_BONUS = "quality_bonus"
    CONSISTENCY_BONUS = "consistency_bonus"
    IMPROVEMENT_BONUS = "improvement_bonus"
    VOLUME_BONUS = "volume_bonus"
    PERFECT_SCORE = "perfect_score"
    SLA_COMPLIANCE = "sla_compliance"


class PenaltyType(str, Enum):
    """Types of penalties."""
    QUALITY_VIOLATION = "quality_violation"
    SLA_BREACH = "sla_breach"
    RULE_VIOLATION = "rule_violation"
    CONSISTENCY_FAILURE = "consistency_failure"


class Incentive(BaseModel):
    """Incentive record."""
    id: UUID = Field(default_factory=uuid4)
    user_id: str
    tenant_id: Optional[str] = None
    incentive_type: IncentiveType
    amount: Decimal
    reason: str
    period: str  # YYYY-MM
    metrics: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    paid_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "incentive_type": self.incentive_type.value,
            "amount": float(self.amount),
            "reason": self.reason,
            "period": self.period,
            "metrics": self.metrics,
            "created_at": self.created_at.isoformat(),
            "paid_at": self.paid_at.isoformat() if self.paid_at else None
        }


class Penalty(BaseModel):
    """Penalty record."""
    id: UUID = Field(default_factory=uuid4)
    user_id: str
    tenant_id: Optional[str] = None
    penalty_type: PenaltyType
    amount: Decimal
    reason: str
    violation_details: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    applied_at: Optional[datetime] = None
    waived: bool = False
    waived_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "penalty_type": self.penalty_type.value,
            "amount": float(self.amount),
            "reason": self.reason,
            "violation_details": self.violation_details,
            "created_at": self.created_at.isoformat(),
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            "waived": self.waived,
            "waived_reason": self.waived_reason
        }


class IncentiveManager:
    """
    Incentive and penalty management engine.

    Handles calculation and tracking of incentives and penalties
    based on performance metrics.
    """

    # Incentive rules configuration
    INCENTIVE_RULES = {
        IncentiveType.QUALITY_BONUS: {
            "threshold": 0.95,  # Quality score >= 95%
            "bonus_rate": 0.15,  # 15% bonus
            "description": "Quality excellence bonus"
        },
        IncentiveType.CONSISTENCY_BONUS: {
            "consecutive_days": 30,  # 30 days consistent
            "min_quality": 0.85,  # Minimum quality during period
            "bonus_rate": 0.10,  # 10% bonus
            "description": "Consistency reward"
        },
        IncentiveType.IMPROVEMENT_BONUS: {
            "improvement_rate": 0.20,  # 20% improvement
            "min_tasks": 100,  # Minimum tasks
            "bonus_rate": 0.10,  # 10% bonus
            "description": "Quality improvement bonus"
        },
        IncentiveType.VOLUME_BONUS: {
            "volume_threshold": 5000,  # Annotations per month
            "bonus_rate": 0.05,  # 5% bonus
            "description": "High volume bonus"
        },
        IncentiveType.PERFECT_SCORE: {
            "perfect_count": 10,  # Number of perfect tasks
            "fixed_bonus": Decimal("50.00"),  # Fixed bonus amount
            "description": "Perfect score achievement"
        },
        IncentiveType.SLA_COMPLIANCE: {
            "compliance_rate": 0.98,  # 98% SLA compliance
            "bonus_rate": 0.08,  # 8% bonus
            "description": "SLA compliance bonus"
        }
    }

    # Penalty rules configuration
    PENALTY_RULES = {
        PenaltyType.QUALITY_VIOLATION: {
            "threshold": 0.60,  # Quality below 60%
            "penalty_rate": 0.20,  # 20% penalty
            "description": "Quality standard violation"
        },
        PenaltyType.SLA_BREACH: {
            "max_breaches": 3,  # Max allowed breaches
            "penalty_per_breach": Decimal("10.00"),
            "description": "SLA breach penalty"
        },
        PenaltyType.RULE_VIOLATION: {
            "penalty_amounts": {
                "minor": Decimal("5.00"),
                "major": Decimal("25.00"),
                "critical": Decimal("100.00")
            },
            "description": "Rule violation penalty"
        },
        PenaltyType.CONSISTENCY_FAILURE: {
            "variance_threshold": 0.30,  # 30% variance
            "penalty_rate": 0.10,  # 10% penalty
            "description": "Inconsistent performance penalty"
        }
    }

    def __init__(self):
        """Initialize the incentive manager."""
        self._incentives: List[Incentive] = []
        self._penalties: List[Penalty] = []

    async def calculate_incentives(
        self,
        user_id: str,
        period: str,
        performance_metrics: Dict[str, Any],
        base_earnings: Decimal
    ) -> List[Incentive]:
        """
        Calculate incentives for a user in a period.

        Args:
            user_id: User identifier
            period: Billing period (YYYY-MM)
            performance_metrics: User performance metrics
            base_earnings: Base earnings for the period

        Returns:
            List of earned incentives
        """
        earned_incentives = []

        # Quality bonus
        quality_score = performance_metrics.get("quality_score", 0)
        if quality_score >= self.INCENTIVE_RULES[IncentiveType.QUALITY_BONUS]["threshold"]:
            bonus_rate = self.INCENTIVE_RULES[IncentiveType.QUALITY_BONUS]["bonus_rate"]
            amount = (base_earnings * Decimal(str(bonus_rate))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            incentive = Incentive(
                user_id=user_id,
                incentive_type=IncentiveType.QUALITY_BONUS,
                amount=amount,
                reason=f"Quality score {quality_score:.2%} exceeded threshold",
                period=period,
                metrics={"quality_score": quality_score}
            )
            earned_incentives.append(incentive)
            self._incentives.append(incentive)

        # Consistency bonus
        consecutive_days = performance_metrics.get("consecutive_good_days", 0)
        avg_quality = performance_metrics.get("avg_quality", 0)
        rule = self.INCENTIVE_RULES[IncentiveType.CONSISTENCY_BONUS]
        if (consecutive_days >= rule["consecutive_days"] and
                avg_quality >= rule["min_quality"]):
            bonus_rate = rule["bonus_rate"]
            amount = (base_earnings * Decimal(str(bonus_rate))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            incentive = Incentive(
                user_id=user_id,
                incentive_type=IncentiveType.CONSISTENCY_BONUS,
                amount=amount,
                reason=f"{consecutive_days} consecutive days of good performance",
                period=period,
                metrics={
                    "consecutive_days": consecutive_days,
                    "avg_quality": avg_quality
                }
            )
            earned_incentives.append(incentive)
            self._incentives.append(incentive)

        # Improvement bonus
        improvement_rate = performance_metrics.get("improvement_rate", 0)
        tasks_completed = performance_metrics.get("tasks_completed", 0)
        rule = self.INCENTIVE_RULES[IncentiveType.IMPROVEMENT_BONUS]
        if (improvement_rate >= rule["improvement_rate"] and
                tasks_completed >= rule["min_tasks"]):
            bonus_rate = rule["bonus_rate"]
            amount = (base_earnings * Decimal(str(bonus_rate))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            incentive = Incentive(
                user_id=user_id,
                incentive_type=IncentiveType.IMPROVEMENT_BONUS,
                amount=amount,
                reason=f"Quality improved by {improvement_rate:.1%}",
                period=period,
                metrics={
                    "improvement_rate": improvement_rate,
                    "tasks_completed": tasks_completed
                }
            )
            earned_incentives.append(incentive)
            self._incentives.append(incentive)

        # Volume bonus
        annotation_count = performance_metrics.get("annotation_count", 0)
        rule = self.INCENTIVE_RULES[IncentiveType.VOLUME_BONUS]
        if annotation_count >= rule["volume_threshold"]:
            bonus_rate = rule["bonus_rate"]
            amount = (base_earnings * Decimal(str(bonus_rate))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            incentive = Incentive(
                user_id=user_id,
                incentive_type=IncentiveType.VOLUME_BONUS,
                amount=amount,
                reason=f"Completed {annotation_count} annotations",
                period=period,
                metrics={"annotation_count": annotation_count}
            )
            earned_incentives.append(incentive)
            self._incentives.append(incentive)

        # Perfect score bonus
        perfect_tasks = performance_metrics.get("perfect_score_tasks", 0)
        rule = self.INCENTIVE_RULES[IncentiveType.PERFECT_SCORE]
        if perfect_tasks >= rule["perfect_count"]:
            amount = rule["fixed_bonus"]
            incentive = Incentive(
                user_id=user_id,
                incentive_type=IncentiveType.PERFECT_SCORE,
                amount=amount,
                reason=f"Achieved {perfect_tasks} perfect score tasks",
                period=period,
                metrics={"perfect_tasks": perfect_tasks}
            )
            earned_incentives.append(incentive)
            self._incentives.append(incentive)

        # SLA compliance bonus
        sla_compliance = performance_metrics.get("sla_compliance_rate", 0)
        rule = self.INCENTIVE_RULES[IncentiveType.SLA_COMPLIANCE]
        if sla_compliance >= rule["compliance_rate"]:
            bonus_rate = rule["bonus_rate"]
            amount = (base_earnings * Decimal(str(bonus_rate))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            incentive = Incentive(
                user_id=user_id,
                incentive_type=IncentiveType.SLA_COMPLIANCE,
                amount=amount,
                reason=f"SLA compliance rate {sla_compliance:.1%}",
                period=period,
                metrics={"sla_compliance_rate": sla_compliance}
            )
            earned_incentives.append(incentive)
            self._incentives.append(incentive)

        logger.info(f"Calculated {len(earned_incentives)} incentives for user {user_id}")
        return earned_incentives

    async def apply_penalties(
        self,
        user_id: str,
        violations: List[Dict[str, Any]]
    ) -> List[Penalty]:
        """
        Apply penalties based on violations.

        Args:
            user_id: User identifier
            violations: List of violations

        Returns:
            List of applied penalties
        """
        applied_penalties = []

        for violation in violations:
            violation_type = violation.get("type")
            severity = violation.get("severity", "minor")
            details = violation.get("details", {})

            penalty = None

            if violation_type == "quality":
                quality_score = details.get("quality_score", 0)
                base_amount = details.get("base_amount", Decimal("0"))
                rule = self.PENALTY_RULES[PenaltyType.QUALITY_VIOLATION]

                if quality_score < rule["threshold"]:
                    amount = (base_amount * Decimal(str(rule["penalty_rate"]))).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                    penalty = Penalty(
                        user_id=user_id,
                        penalty_type=PenaltyType.QUALITY_VIOLATION,
                        amount=amount,
                        reason=f"Quality score {quality_score:.2%} below minimum standard",
                        violation_details=details
                    )

            elif violation_type == "sla_breach":
                breach_count = details.get("breach_count", 0)
                rule = self.PENALTY_RULES[PenaltyType.SLA_BREACH]

                if breach_count > rule["max_breaches"]:
                    excess = breach_count - rule["max_breaches"]
                    amount = rule["penalty_per_breach"] * excess
                    penalty = Penalty(
                        user_id=user_id,
                        penalty_type=PenaltyType.SLA_BREACH,
                        amount=amount,
                        reason=f"{excess} SLA breaches beyond allowed limit",
                        violation_details=details
                    )

            elif violation_type == "rule_violation":
                rule = self.PENALTY_RULES[PenaltyType.RULE_VIOLATION]
                amount = rule["penalty_amounts"].get(severity, rule["penalty_amounts"]["minor"])
                penalty = Penalty(
                    user_id=user_id,
                    penalty_type=PenaltyType.RULE_VIOLATION,
                    amount=amount,
                    reason=details.get("reason", "Rule violation"),
                    violation_details=details
                )

            elif violation_type == "consistency":
                variance = details.get("variance", 0)
                base_amount = details.get("base_amount", Decimal("0"))
                rule = self.PENALTY_RULES[PenaltyType.CONSISTENCY_FAILURE]

                if variance > rule["variance_threshold"]:
                    amount = (base_amount * Decimal(str(rule["penalty_rate"]))).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                    penalty = Penalty(
                        user_id=user_id,
                        penalty_type=PenaltyType.CONSISTENCY_FAILURE,
                        amount=amount,
                        reason=f"Performance variance {variance:.1%} exceeds threshold",
                        violation_details=details
                    )

            if penalty:
                penalty.applied_at = datetime.now()
                applied_penalties.append(penalty)
                self._penalties.append(penalty)

        logger.info(f"Applied {len(applied_penalties)} penalties for user {user_id}")
        return applied_penalties

    async def waive_penalty(
        self,
        penalty_id: UUID,
        reason: str,
        waived_by: str
    ) -> bool:
        """
        Waive a penalty.

        Args:
            penalty_id: Penalty UUID
            reason: Reason for waiving
            waived_by: User who waived

        Returns:
            True if waived successfully
        """
        for penalty in self._penalties:
            if penalty.id == penalty_id:
                if penalty.waived:
                    return False
                penalty.waived = True
                penalty.waived_reason = f"{reason} (by {waived_by})"
                logger.info(f"Penalty {penalty_id} waived: {reason}")
                return True
        return False

    async def get_user_incentive_summary(
        self,
        user_id: str,
        period: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get incentive summary for a user.

        Args:
            user_id: User identifier
            period: Optional period filter (YYYY-MM)

        Returns:
            Incentive summary
        """
        user_incentives = [
            i for i in self._incentives
            if i.user_id == user_id and (not period or i.period == period)
        ]

        user_penalties = [
            p for p in self._penalties
            if p.user_id == user_id and not p.waived
        ]

        total_incentives = sum(i.amount for i in user_incentives)
        total_penalties = sum(p.amount for p in user_penalties)

        by_type = {}
        for incentive in user_incentives:
            type_key = incentive.incentive_type.value
            if type_key not in by_type:
                by_type[type_key] = {"count": 0, "amount": Decimal("0")}
            by_type[type_key]["count"] += 1
            by_type[type_key]["amount"] += incentive.amount

        return {
            "user_id": user_id,
            "period": period,
            "incentives": {
                "count": len(user_incentives),
                "total_amount": float(total_incentives),
                "by_type": {
                    k: {"count": v["count"], "amount": float(v["amount"])}
                    for k, v in by_type.items()
                }
            },
            "penalties": {
                "count": len(user_penalties),
                "total_amount": float(total_penalties)
            },
            "net_amount": float(total_incentives - total_penalties),
            "incentive_details": [i.to_dict() for i in user_incentives],
            "penalty_details": [p.to_dict() for p in user_penalties]
        }

    async def get_tenant_incentive_summary(
        self,
        tenant_id: str,
        period: str
    ) -> Dict[str, Any]:
        """
        Get incentive summary for a tenant.

        Args:
            tenant_id: Tenant identifier
            period: Billing period (YYYY-MM)

        Returns:
            Tenant incentive summary
        """
        tenant_incentives = [
            i for i in self._incentives
            if i.tenant_id == tenant_id and i.period == period
        ]

        tenant_penalties = [
            p for p in self._penalties
            if p.tenant_id == tenant_id and not p.waived
        ]

        # Group by user
        user_summaries = {}
        for incentive in tenant_incentives:
            if incentive.user_id not in user_summaries:
                user_summaries[incentive.user_id] = {
                    "incentives": Decimal("0"),
                    "penalties": Decimal("0")
                }
            user_summaries[incentive.user_id]["incentives"] += incentive.amount

        for penalty in tenant_penalties:
            if penalty.user_id not in user_summaries:
                user_summaries[penalty.user_id] = {
                    "incentives": Decimal("0"),
                    "penalties": Decimal("0")
                }
            user_summaries[penalty.user_id]["penalties"] += penalty.amount

        total_incentives = sum(i.amount for i in tenant_incentives)
        total_penalties = sum(p.amount for p in tenant_penalties)

        return {
            "tenant_id": tenant_id,
            "period": period,
            "total_incentives": float(total_incentives),
            "total_penalties": float(total_penalties),
            "net_amount": float(total_incentives - total_penalties),
            "user_count": len(user_summaries),
            "user_breakdown": {
                user_id: {
                    "incentives": float(data["incentives"]),
                    "penalties": float(data["penalties"]),
                    "net": float(data["incentives"] - data["penalties"])
                }
                for user_id, data in user_summaries.items()
            },
            "generated_at": datetime.now().isoformat()
        }

    def update_incentive_rules(
        self,
        incentive_type: IncentiveType,
        new_rules: Dict[str, Any]
    ) -> None:
        """
        Update incentive rules.

        Args:
            incentive_type: Type of incentive
            new_rules: New rule configuration
        """
        if incentive_type in self.INCENTIVE_RULES:
            self.INCENTIVE_RULES[incentive_type].update(new_rules)
            logger.info(f"Updated incentive rules for {incentive_type.value}")

    def update_penalty_rules(
        self,
        penalty_type: PenaltyType,
        new_rules: Dict[str, Any]
    ) -> None:
        """
        Update penalty rules.

        Args:
            penalty_type: Type of penalty
            new_rules: New rule configuration
        """
        if penalty_type in self.PENALTY_RULES:
            self.PENALTY_RULES[penalty_type].update(new_rules)
            logger.info(f"Updated penalty rules for {penalty_type.value}")

    def get_rules_config(self) -> Dict[str, Any]:
        """Get current rules configuration."""
        return {
            "incentive_rules": {
                k.value: {
                    key: float(val) if isinstance(val, Decimal) else val
                    for key, val in v.items()
                }
                for k, v in self.INCENTIVE_RULES.items()
            },
            "penalty_rules": {
                k.value: {
                    key: (
                        float(val) if isinstance(val, Decimal)
                        else {kk: float(vv) for kk, vv in val.items()} if isinstance(val, dict) and all(isinstance(vv, Decimal) for vv in val.values())
                        else val
                    )
                    for key, val in v.items()
                }
                for k, v in self.PENALTY_RULES.items()
            }
        }
