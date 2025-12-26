"""
Quality-driven pricing engine for SuperInsight Platform.

Provides:
- Quality-based cost adjustments
- Difficulty-weighted pricing
- Quality certificates for invoices
- Detailed billing with quality breakdown
"""

import logging
from datetime import datetime, date
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class QualityLevel(str, Enum):
    """Quality level classification."""
    EXCELLENT = "excellent"     # >= 0.90
    GOOD = "good"               # >= 0.80
    ACCEPTABLE = "acceptable"   # >= 0.70
    POOR = "poor"               # < 0.70


class DifficultyLevel(str, Enum):
    """Task difficulty level."""
    SIMPLE = "simple"           # Basic tasks
    STANDARD = "standard"       # Normal complexity
    COMPLEX = "complex"         # Advanced tasks
    EXPERT = "expert"           # Specialist level


class QualityPriceAdjustment(BaseModel):
    """Quality price adjustment record."""
    id: UUID = Field(default_factory=uuid4)
    billing_record_id: UUID
    base_cost: Decimal
    quality_score: float
    quality_level: QualityLevel
    quality_multiplier: float
    difficulty_level: DifficultyLevel
    difficulty_multiplier: float
    adjusted_cost: Decimal
    adjustment_details: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class QualityCertificate(BaseModel):
    """Quality certificate for billing."""
    id: UUID = Field(default_factory=uuid4)
    tenant_id: str
    billing_period: str
    total_tasks: int = 0
    quality_distribution: Dict[str, int] = Field(default_factory=dict)
    avg_quality_score: float = 0.0
    quality_compliance_rate: float = 0.0
    certified_at: datetime = Field(default_factory=datetime.now)


class QualityPricingEngine:
    """
    Quality-driven pricing engine.

    Calculates costs adjusted for quality performance
    and task difficulty.
    """

    # Quality multipliers (adjust final cost)
    QUALITY_MULTIPLIERS = {
        QualityLevel.EXCELLENT: Decimal("1.20"),   # 20% bonus
        QualityLevel.GOOD: Decimal("1.00"),        # Standard rate
        QualityLevel.ACCEPTABLE: Decimal("0.90"),  # 10% reduction
        QualityLevel.POOR: Decimal("0.70"),        # 30% reduction
    }

    # Quality score thresholds
    QUALITY_THRESHOLDS = {
        QualityLevel.EXCELLENT: 0.90,
        QualityLevel.GOOD: 0.80,
        QualityLevel.ACCEPTABLE: 0.70,
        # Below 0.70 is POOR
    }

    # Difficulty multipliers (adjust base rate)
    DIFFICULTY_MULTIPLIERS = {
        DifficultyLevel.SIMPLE: Decimal("0.80"),   # 20% discount
        DifficultyLevel.STANDARD: Decimal("1.00"), # Standard rate
        DifficultyLevel.COMPLEX: Decimal("1.30"),  # 30% premium
        DifficultyLevel.EXPERT: Decimal("1.60"),   # 60% premium
    }

    # Minimum quality threshold for payment
    MIN_QUALITY_FOR_PAYMENT = 0.50

    def __init__(self):
        """Initialize the quality pricing engine."""
        self._adjustments: List[QualityPriceAdjustment] = []

    def get_quality_level(self, quality_score: float) -> QualityLevel:
        """
        Determine quality level from score.

        Args:
            quality_score: Quality score (0-1)

        Returns:
            Quality level classification
        """
        if quality_score >= self.QUALITY_THRESHOLDS[QualityLevel.EXCELLENT]:
            return QualityLevel.EXCELLENT
        elif quality_score >= self.QUALITY_THRESHOLDS[QualityLevel.GOOD]:
            return QualityLevel.GOOD
        elif quality_score >= self.QUALITY_THRESHOLDS[QualityLevel.ACCEPTABLE]:
            return QualityLevel.ACCEPTABLE
        else:
            return QualityLevel.POOR

    def calculate_quality_adjusted_cost(
        self,
        base_cost: Decimal,
        quality_score: float,
        difficulty_level: DifficultyLevel = DifficultyLevel.STANDARD,
        min_cost: Optional[Decimal] = None,
        max_cost: Optional[Decimal] = None
    ) -> Dict[str, Any]:
        """
        Calculate quality-adjusted cost.

        Args:
            base_cost: Base cost before adjustment
            quality_score: Quality score (0-1)
            difficulty_level: Task difficulty level
            min_cost: Minimum cost floor
            max_cost: Maximum cost ceiling

        Returns:
            Adjusted cost with breakdown
        """
        # Validate quality score
        if quality_score < 0 or quality_score > 1:
            raise ValueError("Quality score must be between 0 and 1")

        # Check minimum quality threshold
        if quality_score < self.MIN_QUALITY_FOR_PAYMENT:
            return {
                "base_cost": float(base_cost),
                "adjusted_cost": 0.0,
                "quality_score": quality_score,
                "quality_level": "rejected",
                "quality_multiplier": 0.0,
                "difficulty_multiplier": 1.0,
                "rejected": True,
                "rejection_reason": f"Quality score {quality_score:.2f} below minimum threshold {self.MIN_QUALITY_FOR_PAYMENT}"
            }

        # Get quality level and multiplier
        quality_level = self.get_quality_level(quality_score)
        quality_multiplier = self.QUALITY_MULTIPLIERS[quality_level]

        # Get difficulty multiplier
        difficulty_multiplier = self.DIFFICULTY_MULTIPLIERS[difficulty_level]

        # Calculate adjusted cost
        adjusted_cost = base_cost * quality_multiplier * difficulty_multiplier
        adjusted_cost = adjusted_cost.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Apply bounds
        if min_cost is not None:
            adjusted_cost = max(adjusted_cost, min_cost)
        if max_cost is not None:
            adjusted_cost = min(adjusted_cost, max_cost)

        return {
            "base_cost": float(base_cost),
            "adjusted_cost": float(adjusted_cost),
            "quality_score": quality_score,
            "quality_level": quality_level.value,
            "quality_multiplier": float(quality_multiplier),
            "difficulty_level": difficulty_level.value,
            "difficulty_multiplier": float(difficulty_multiplier),
            "adjustment_amount": float(adjusted_cost - base_cost),
            "adjustment_percentage": float((adjusted_cost - base_cost) / base_cost * 100) if base_cost > 0 else 0,
            "rejected": False
        }

    def calculate_batch_cost(
        self,
        items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate costs for a batch of items.

        Args:
            items: List of items with base_cost, quality_score, difficulty_level

        Returns:
            Batch calculation results
        """
        results = []
        total_base = Decimal("0")
        total_adjusted = Decimal("0")
        rejected_count = 0

        for item in items:
            base_cost = Decimal(str(item.get("base_cost", 0)))
            quality_score = item.get("quality_score", 0)
            difficulty = DifficultyLevel(item.get("difficulty_level", "standard"))

            result = self.calculate_quality_adjusted_cost(
                base_cost=base_cost,
                quality_score=quality_score,
                difficulty_level=difficulty
            )

            results.append(result)
            total_base += base_cost

            if not result.get("rejected"):
                total_adjusted += Decimal(str(result["adjusted_cost"]))
            else:
                rejected_count += 1

        return {
            "items": results,
            "summary": {
                "total_items": len(items),
                "accepted_items": len(items) - rejected_count,
                "rejected_items": rejected_count,
                "total_base_cost": float(total_base),
                "total_adjusted_cost": float(total_adjusted),
                "overall_adjustment": float(total_adjusted - total_base),
                "adjustment_percentage": float((total_adjusted - total_base) / total_base * 100) if total_base > 0 else 0
            }
        }

    async def generate_detailed_invoice(
        self,
        tenant_id: str,
        billing_period: str,
        billing_records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate detailed invoice with quality breakdown.

        Args:
            tenant_id: Tenant identifier
            billing_period: Period (YYYY-MM)
            billing_records: List of billing records with quality data

        Returns:
            Detailed invoice
        """
        invoice_id = uuid4()
        line_items = []
        quality_summary = {
            "excellent": {"count": 0, "amount": Decimal("0")},
            "good": {"count": 0, "amount": Decimal("0")},
            "acceptable": {"count": 0, "amount": Decimal("0")},
            "poor": {"count": 0, "amount": Decimal("0")},
            "rejected": {"count": 0, "amount": Decimal("0")}
        }

        total_base = Decimal("0")
        total_adjusted = Decimal("0")
        total_annotations = 0
        total_quality_bonus = Decimal("0")
        total_quality_penalty = Decimal("0")

        for record in billing_records:
            base_cost = Decimal(str(record.get("cost", 0)))
            quality_score = record.get("quality_score", 0.8)
            difficulty = record.get("difficulty_level", "standard")
            annotations = record.get("annotation_count", 0)

            # Calculate quality-adjusted cost
            adjustment = self.calculate_quality_adjusted_cost(
                base_cost=base_cost,
                quality_score=quality_score,
                difficulty_level=DifficultyLevel(difficulty)
            )

            adjusted_cost = Decimal(str(adjustment["adjusted_cost"]))
            quality_level = adjustment.get("quality_level", "rejected")

            # Track summary
            if quality_level in quality_summary:
                quality_summary[quality_level]["count"] += 1
                quality_summary[quality_level]["amount"] += adjusted_cost

            total_base += base_cost
            total_adjusted += adjusted_cost
            total_annotations += annotations

            # Track bonuses/penalties
            diff = adjusted_cost - base_cost
            if diff > 0:
                total_quality_bonus += diff
            else:
                total_quality_penalty += abs(diff)

            line_items.append({
                "record_id": str(record.get("id", uuid4())),
                "user_id": record.get("user_id"),
                "annotations": annotations,
                "base_cost": float(base_cost),
                "quality_score": quality_score,
                "quality_level": quality_level,
                "adjusted_cost": float(adjusted_cost),
                "adjustment": float(diff)
            })

        # Calculate quality metrics
        valid_scores = [r.get("quality_score", 0) for r in billing_records
                        if r.get("quality_score", 0) >= self.MIN_QUALITY_FOR_PAYMENT]
        avg_quality = sum(valid_scores) / len(valid_scores) if valid_scores else 0

        return {
            "invoice_id": str(invoice_id),
            "tenant_id": tenant_id,
            "billing_period": billing_period,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_annotations": total_annotations,
                "total_records": len(billing_records),
                "base_total": float(total_base),
                "adjusted_total": float(total_adjusted),
                "quality_bonus": float(total_quality_bonus),
                "quality_penalty": float(total_quality_penalty),
                "net_adjustment": float(total_adjusted - total_base),
                "avg_quality_score": round(avg_quality, 4)
            },
            "quality_breakdown": {
                level: {
                    "count": data["count"],
                    "amount": float(data["amount"])
                }
                for level, data in quality_summary.items()
            },
            "line_items": line_items,
            "quality_certificate": self._generate_quality_certificate(
                tenant_id, billing_period, quality_summary, avg_quality, len(billing_records)
            )
        }

    def _generate_quality_certificate(
        self,
        tenant_id: str,
        billing_period: str,
        quality_summary: Dict,
        avg_quality: float,
        total_records: int
    ) -> Dict[str, Any]:
        """Generate quality certificate for invoice."""
        excellent_good = (
            quality_summary["excellent"]["count"] +
            quality_summary["good"]["count"]
        )
        compliance_rate = excellent_good / total_records if total_records > 0 else 0

        return {
            "certificate_id": str(uuid4()),
            "tenant_id": tenant_id,
            "billing_period": billing_period,
            "total_tasks": total_records,
            "quality_distribution": {
                level: data["count"]
                for level, data in quality_summary.items()
            },
            "avg_quality_score": round(avg_quality, 4),
            "quality_compliance_rate": round(compliance_rate, 4),
            "certified_at": datetime.now().isoformat(),
            "certification_statement": (
                f"This certifies that {compliance_rate * 100:.1f}% of deliverables "
                f"met or exceeded quality standards (Good or Excellent) during {billing_period}."
            )
        }

    async def get_pricing_recommendations(
        self,
        tenant_id: str,
        historical_quality: float,
        volume_per_month: int
    ) -> Dict[str, Any]:
        """
        Get pricing recommendations based on historical performance.

        Args:
            tenant_id: Tenant identifier
            historical_quality: Historical average quality score
            volume_per_month: Expected monthly volume

        Returns:
            Pricing recommendations
        """
        quality_level = self.get_quality_level(historical_quality)

        # Base recommendations
        recommendations = {
            "tenant_id": tenant_id,
            "historical_quality": historical_quality,
            "quality_level": quality_level.value,
            "monthly_volume": volume_per_month,
            "recommendations": []
        }

        if quality_level == QualityLevel.EXCELLENT:
            recommendations["recommendations"].extend([
                {
                    "type": "premium_pricing",
                    "description": "Qualify for premium rate tier with quality bonus",
                    "potential_bonus": "Up to 20% quality bonus"
                },
                {
                    "type": "volume_discount",
                    "description": "Consider volume-based discounts for consistent quality",
                    "threshold": 10000
                }
            ])
        elif quality_level == QualityLevel.GOOD:
            recommendations["recommendations"].extend([
                {
                    "type": "maintain_quality",
                    "description": "Maintain current quality to keep standard rates",
                    "potential_bonus": "Achieve 5% more for excellent tier"
                }
            ])
        elif quality_level == QualityLevel.ACCEPTABLE:
            recommendations["recommendations"].extend([
                {
                    "type": "quality_improvement",
                    "description": "Improve quality to avoid rate reduction",
                    "current_penalty": "10% reduction applied"
                },
                {
                    "type": "training",
                    "description": "Consider additional training to improve scores"
                }
            ])
        else:
            recommendations["recommendations"].extend([
                {
                    "type": "urgent_improvement",
                    "description": "Quality below acceptable threshold",
                    "current_penalty": "30% reduction applied"
                },
                {
                    "type": "review_process",
                    "description": "Review annotation processes and guidelines"
                }
            ])

        # Volume-based recommendations
        if volume_per_month > 50000:
            recommendations["recommendations"].append({
                "type": "enterprise_pricing",
                "description": "Consider enterprise pricing model for high volume",
                "volume_tier": "enterprise"
            })
        elif volume_per_month > 10000:
            recommendations["recommendations"].append({
                "type": "volume_tier",
                "description": "Qualify for volume discount tier",
                "discount": "5-10%"
            })

        return recommendations

    def update_multipliers(
        self,
        quality_multipliers: Optional[Dict[str, float]] = None,
        difficulty_multipliers: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Update pricing multipliers.

        Args:
            quality_multipliers: New quality multipliers
            difficulty_multipliers: New difficulty multipliers
        """
        if quality_multipliers:
            for level, value in quality_multipliers.items():
                try:
                    quality_level = QualityLevel(level)
                    self.QUALITY_MULTIPLIERS[quality_level] = Decimal(str(value))
                    logger.info(f"Updated quality multiplier {level}: {value}")
                except ValueError:
                    logger.warning(f"Unknown quality level: {level}")

        if difficulty_multipliers:
            for level, value in difficulty_multipliers.items():
                try:
                    difficulty_level = DifficultyLevel(level)
                    self.DIFFICULTY_MULTIPLIERS[difficulty_level] = Decimal(str(value))
                    logger.info(f"Updated difficulty multiplier {level}: {value}")
                except ValueError:
                    logger.warning(f"Unknown difficulty level: {level}")

    def get_pricing_config(self) -> Dict[str, Any]:
        """Get current pricing configuration."""
        return {
            "quality_multipliers": {
                level.value: float(mult)
                for level, mult in self.QUALITY_MULTIPLIERS.items()
            },
            "quality_thresholds": {
                level.value: thresh
                for level, thresh in self.QUALITY_THRESHOLDS.items()
            },
            "difficulty_multipliers": {
                level.value: float(mult)
                for level, mult in self.DIFFICULTY_MULTIPLIERS.items()
            },
            "min_quality_for_payment": self.MIN_QUALITY_FOR_PAYMENT
        }
