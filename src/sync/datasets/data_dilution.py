"""
Data Dilution Engine for Noise Reduction.

This module provides capabilities for diluting low-quality or noisy data
by mixing it with high-quality industry datasets.
"""

import asyncio
import hashlib
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, TypeVar

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DilutionStrategy(str, Enum):
    """Strategies for data dilution."""

    UNIFORM = "uniform"  # Uniform mixing ratio
    QUALITY_WEIGHTED = "quality_weighted"  # Weight by quality scores
    DOMAIN_BALANCED = "domain_balanced"  # Balance across domains
    NOISE_TARGETED = "noise_targeted"  # Target noisy samples
    ADAPTIVE = "adaptive"  # Adaptive based on evaluation


class SampleQuality(str, Enum):
    """Quality levels for data samples."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NOISE = "noise"


@dataclass
class DataSample:
    """A single data sample for processing."""

    id: str
    content: Dict[str, Any]
    source: str
    domain: str = "general"
    quality_score: float = 0.5
    quality_level: SampleQuality = SampleQuality.MEDIUM
    labels: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "domain": self.domain,
            "quality_score": self.quality_score,
            "quality_level": self.quality_level.value,
            "labels": self.labels,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


class DilutionConfig(BaseModel):
    """Configuration for data dilution."""

    strategy: DilutionStrategy = DilutionStrategy.QUALITY_WEIGHTED
    target_noise_ratio: float = Field(
        default=0.1, ge=0.0, le=0.5, description="Target ratio of noisy samples"
    )
    min_quality_score: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum quality score threshold"
    )
    max_dilution_ratio: float = Field(
        default=0.5, ge=0.0, le=0.9, description="Maximum ratio of added samples"
    )
    preserve_distribution: bool = Field(
        default=True, description="Preserve domain distribution"
    )
    seed: Optional[int] = None

    class Config:
        use_enum_values = True


@dataclass
class DilutionMetrics:
    """Metrics from a dilution operation."""

    original_count: int = 0
    added_count: int = 0
    removed_count: int = 0
    final_count: int = 0
    original_noise_ratio: float = 0.0
    final_noise_ratio: float = 0.0
    original_avg_quality: float = 0.0
    final_avg_quality: float = 0.0
    domain_distribution: Dict[str, int] = field(default_factory=dict)
    execution_time_ms: float = 0.0

    @property
    def quality_improvement(self) -> float:
        return self.final_avg_quality - self.original_avg_quality

    @property
    def noise_reduction(self) -> float:
        return self.original_noise_ratio - self.final_noise_ratio

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_count": self.original_count,
            "added_count": self.added_count,
            "removed_count": self.removed_count,
            "final_count": self.final_count,
            "original_noise_ratio": round(self.original_noise_ratio, 4),
            "final_noise_ratio": round(self.final_noise_ratio, 4),
            "original_avg_quality": round(self.original_avg_quality, 4),
            "final_avg_quality": round(self.final_avg_quality, 4),
            "quality_improvement": round(self.quality_improvement, 4),
            "noise_reduction": round(self.noise_reduction, 4),
            "domain_distribution": self.domain_distribution,
            "execution_time_ms": round(self.execution_time_ms, 2),
        }


class QualityAssessor:
    """Assesses quality of data samples."""

    def __init__(
        self,
        noise_threshold: float = 0.3,
        low_quality_threshold: float = 0.5,
        high_quality_threshold: float = 0.8,
    ):
        self.noise_threshold = noise_threshold
        self.low_quality_threshold = low_quality_threshold
        self.high_quality_threshold = high_quality_threshold

    def assess(self, sample: DataSample) -> Tuple[float, SampleQuality]:
        """Assess quality of a sample."""
        # Calculate quality score based on multiple factors
        scores = []

        # Content completeness
        content = sample.content
        if isinstance(content, dict):
            non_empty = sum(1 for v in content.values() if v)
            total = len(content)
            completeness = non_empty / total if total > 0 else 0
            scores.append(completeness)

            # Text length quality (for text fields)
            text_fields = [v for v in content.values() if isinstance(v, str)]
            if text_fields:
                avg_length = sum(len(t) for t in text_fields) / len(text_fields)
                # Prefer moderate length (not too short, not too long)
                length_score = min(avg_length / 100, 1.0) * (1 - max(0, (avg_length - 500) / 1000))
                scores.append(max(0, length_score))

        # Label quality
        if sample.labels:
            label_score = min(len(sample.labels) / 3, 1.0)
            scores.append(label_score)

        # Calculate final score
        quality_score = sum(scores) / len(scores) if scores else 0.5

        # Override with provided score if available
        if sample.quality_score > 0:
            quality_score = (quality_score + sample.quality_score) / 2

        # Determine quality level
        if quality_score < self.noise_threshold:
            quality_level = SampleQuality.NOISE
        elif quality_score < self.low_quality_threshold:
            quality_level = SampleQuality.LOW
        elif quality_score < self.high_quality_threshold:
            quality_level = SampleQuality.MEDIUM
        else:
            quality_level = SampleQuality.HIGH

        return quality_score, quality_level

    def assess_batch(
        self, samples: List[DataSample]
    ) -> List[Tuple[DataSample, float, SampleQuality]]:
        """Assess quality of multiple samples."""
        results = []
        for sample in samples:
            score, level = self.assess(sample)
            sample.quality_score = score
            sample.quality_level = level
            results.append((sample, score, level))
        return results


class SampleSelector:
    """Selects samples for dilution based on various criteria."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def select_for_removal(
        self,
        samples: List[DataSample],
        count: int,
        strategy: DilutionStrategy,
    ) -> List[DataSample]:
        """Select samples to remove (typically low quality ones)."""
        if strategy == DilutionStrategy.NOISE_TARGETED:
            # Sort by quality, remove lowest
            sorted_samples = sorted(samples, key=lambda s: s.quality_score)
            return sorted_samples[:count]

        elif strategy == DilutionStrategy.QUALITY_WEIGHTED:
            # Weighted selection favoring low quality
            weights = [1 - s.quality_score for s in samples]
            total_weight = sum(weights)
            if total_weight == 0:
                return self.rng.sample(samples, min(count, len(samples)))

            selected = []
            remaining = list(zip(samples, weights))

            for _ in range(min(count, len(samples))):
                if not remaining:
                    break
                total = sum(w for _, w in remaining)
                if total == 0:
                    break
                r = self.rng.random() * total
                cumulative = 0
                for i, (sample, weight) in enumerate(remaining):
                    cumulative += weight
                    if cumulative >= r:
                        selected.append(sample)
                        remaining.pop(i)
                        break

            return selected

        else:
            # Random selection
            return self.rng.sample(samples, min(count, len(samples)))

    def select_for_addition(
        self,
        samples: List[DataSample],
        count: int,
        strategy: DilutionStrategy,
        target_domains: Optional[Dict[str, float]] = None,
    ) -> List[DataSample]:
        """Select samples to add from a pool (typically high quality ones)."""
        if strategy == DilutionStrategy.DOMAIN_BALANCED and target_domains:
            # Balance across domains
            selected = []
            for domain, ratio in target_domains.items():
                domain_samples = [s for s in samples if s.domain == domain]
                domain_count = int(count * ratio)
                selected.extend(
                    self.rng.sample(
                        domain_samples, min(domain_count, len(domain_samples))
                    )
                )
            return selected

        elif strategy == DilutionStrategy.QUALITY_WEIGHTED:
            # Weighted selection favoring high quality
            weights = [s.quality_score for s in samples]
            total_weight = sum(weights)
            if total_weight == 0:
                return self.rng.sample(samples, min(count, len(samples)))

            selected = []
            remaining = list(zip(samples, weights))

            for _ in range(min(count, len(samples))):
                if not remaining:
                    break
                total = sum(w for _, w in remaining)
                if total == 0:
                    break
                r = self.rng.random() * total
                cumulative = 0
                for i, (sample, weight) in enumerate(remaining):
                    cumulative += weight
                    if cumulative >= r:
                        selected.append(sample)
                        remaining.pop(i)
                        break

            return selected

        else:
            # Random selection
            return self.rng.sample(samples, min(count, len(samples)))


class DataDilutionEngine:
    """Engine for diluting noisy data with high-quality samples."""

    def __init__(self, config: Optional[DilutionConfig] = None):
        self.config = config or DilutionConfig()
        self.assessor = QualityAssessor()
        self.selector = SampleSelector(seed=self.config.seed)

    async def dilute(
        self,
        customer_data: List[DataSample],
        industry_data: List[DataSample],
        config: Optional[DilutionConfig] = None,
    ) -> Tuple[List[DataSample], DilutionMetrics]:
        """
        Dilute customer data with high-quality industry data.

        Args:
            customer_data: Customer's original data samples
            industry_data: High-quality industry dataset samples
            config: Optional override configuration

        Returns:
            Tuple of (diluted_samples, metrics)
        """
        import time

        start_time = time.time()
        config = config or self.config

        metrics = DilutionMetrics(original_count=len(customer_data))

        # Assess quality of customer data
        assessed_customer = self.assessor.assess_batch(customer_data)
        quality_scores = [score for _, score, _ in assessed_customer]

        if quality_scores:
            metrics.original_avg_quality = sum(quality_scores) / len(quality_scores)

        # Calculate noise ratio
        noise_count = sum(
            1 for _, _, level in assessed_customer if level == SampleQuality.NOISE
        )
        metrics.original_noise_ratio = noise_count / len(customer_data) if customer_data else 0

        # Calculate domain distribution
        domain_counts: Dict[str, int] = {}
        for sample in customer_data:
            domain_counts[sample.domain] = domain_counts.get(sample.domain, 0) + 1

        # Determine how many samples to add/remove
        current_noise_ratio = metrics.original_noise_ratio
        target_noise_ratio = config.target_noise_ratio

        samples_to_remove = 0
        samples_to_add = 0

        if current_noise_ratio > target_noise_ratio:
            # Calculate required dilution
            total_samples = len(customer_data)
            current_noise = int(total_samples * current_noise_ratio)
            target_noise = int(total_samples * target_noise_ratio)

            if config.strategy == DilutionStrategy.NOISE_TARGETED:
                # Remove noisy samples directly
                samples_to_remove = min(
                    current_noise - target_noise,
                    int(total_samples * 0.3),  # Don't remove more than 30%
                )
            else:
                # Add high-quality samples to dilute noise ratio
                # new_noise_ratio = current_noise / (total + added)
                # target = current_noise / (total + added)
                # added = current_noise / target - total
                if target_noise_ratio > 0:
                    samples_to_add = int(
                        current_noise / target_noise_ratio - total_samples
                    )
                    samples_to_add = min(
                        samples_to_add,
                        int(total_samples * config.max_dilution_ratio),
                    )
                    samples_to_add = max(0, samples_to_add)

        # Assess quality of industry data
        assessed_industry = self.assessor.assess_batch(industry_data)

        # Filter to high-quality samples only
        high_quality_industry = [
            sample
            for sample, score, _ in assessed_industry
            if score >= config.min_quality_score
        ]

        # Select samples to remove from customer data
        if samples_to_remove > 0:
            to_remove = self.selector.select_for_removal(
                [sample for sample, _, _ in assessed_customer],
                samples_to_remove,
                DilutionStrategy(config.strategy),
            )
            remove_ids = {s.id for s in to_remove}
            customer_data = [s for s in customer_data if s.id not in remove_ids]
            metrics.removed_count = len(to_remove)

        # Select samples to add from industry data
        result_samples = list(customer_data)

        if samples_to_add > 0 and high_quality_industry:
            # Calculate target domain distribution
            target_domains = None
            if config.preserve_distribution:
                total = sum(domain_counts.values())
                target_domains = {
                    domain: count / total for domain, count in domain_counts.items()
                }

            to_add = self.selector.select_for_addition(
                high_quality_industry,
                samples_to_add,
                DilutionStrategy(config.strategy),
                target_domains,
            )

            # Mark added samples with industry source
            for sample in to_add:
                sample.metadata["dilution_source"] = "industry"
                sample.metadata["diluted_at"] = datetime.utcnow().isoformat()

            result_samples.extend(to_add)
            metrics.added_count = len(to_add)

        # Calculate final metrics
        metrics.final_count = len(result_samples)

        final_assessed = self.assessor.assess_batch(result_samples)
        final_scores = [score for _, score, _ in final_assessed]
        final_noise = sum(
            1 for _, _, level in final_assessed if level == SampleQuality.NOISE
        )

        if final_scores:
            metrics.final_avg_quality = sum(final_scores) / len(final_scores)
        metrics.final_noise_ratio = final_noise / len(result_samples) if result_samples else 0

        # Update domain distribution
        for sample in result_samples:
            metrics.domain_distribution[sample.domain] = (
                metrics.domain_distribution.get(sample.domain, 0) + 1
            )

        metrics.execution_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Dilution complete: {metrics.original_count} -> {metrics.final_count} samples, "
            f"quality: {metrics.original_avg_quality:.3f} -> {metrics.final_avg_quality:.3f}, "
            f"noise: {metrics.original_noise_ratio:.3f} -> {metrics.final_noise_ratio:.3f}"
        )

        return result_samples, metrics

    async def optimize_dilution_ratio(
        self,
        customer_data: List[DataSample],
        industry_data: List[DataSample],
        target_quality: float = 0.85,
        max_iterations: int = 5,
    ) -> DilutionConfig:
        """
        Automatically find optimal dilution configuration.

        Args:
            customer_data: Customer's data
            industry_data: Industry data pool
            target_quality: Target average quality score
            max_iterations: Maximum optimization iterations

        Returns:
            Optimized DilutionConfig
        """
        best_config = self.config.copy()
        best_quality = 0.0

        for i in range(max_iterations):
            # Try different configurations
            test_config = best_config.copy()
            test_config.max_dilution_ratio = 0.1 + (i * 0.1)

            _, metrics = await self.dilute(
                list(customer_data), list(industry_data), test_config
            )

            if metrics.final_avg_quality > best_quality:
                best_quality = metrics.final_avg_quality
                best_config = test_config

            if best_quality >= target_quality:
                break

        logger.info(
            f"Optimized dilution config: ratio={best_config.max_dilution_ratio}, "
            f"achieved quality={best_quality:.3f}"
        )

        return best_config


class DataMerger:
    """Merges multiple datasets while handling conflicts and deduplication."""

    def __init__(self, dedup_threshold: float = 0.9):
        self.dedup_threshold = dedup_threshold

    def _compute_hash(self, sample: DataSample) -> str:
        """Compute content hash for deduplication."""
        content_str = str(sorted(sample.content.items()))
        return hashlib.md5(content_str.encode()).hexdigest()

    async def merge(
        self,
        datasets: List[List[DataSample]],
        deduplicate: bool = True,
        conflict_resolution: str = "quality",  # "quality", "first", "last"
    ) -> List[DataSample]:
        """
        Merge multiple datasets into one.

        Args:
            datasets: List of datasets to merge
            deduplicate: Whether to remove duplicates
            conflict_resolution: How to handle conflicts

        Returns:
            Merged dataset
        """
        all_samples: Dict[str, DataSample] = {}
        hash_to_id: Dict[str, str] = {}

        for dataset in datasets:
            for sample in dataset:
                content_hash = self._compute_hash(sample)

                if deduplicate and content_hash in hash_to_id:
                    # Handle duplicate
                    existing_id = hash_to_id[content_hash]
                    existing = all_samples[existing_id]

                    if conflict_resolution == "quality":
                        if sample.quality_score > existing.quality_score:
                            all_samples[sample.id] = sample
                            del all_samples[existing_id]
                            hash_to_id[content_hash] = sample.id
                    elif conflict_resolution == "last":
                        all_samples[sample.id] = sample
                        del all_samples[existing_id]
                        hash_to_id[content_hash] = sample.id
                    # "first" keeps existing
                else:
                    all_samples[sample.id] = sample
                    hash_to_id[content_hash] = sample.id

        return list(all_samples.values())


# Global instances
_dilution_engine: Optional[DataDilutionEngine] = None
_data_merger: Optional[DataMerger] = None


def get_dilution_engine(config: Optional[DilutionConfig] = None) -> DataDilutionEngine:
    """Get or create global dilution engine."""
    global _dilution_engine
    if _dilution_engine is None or config:
        _dilution_engine = DataDilutionEngine(config)
    return _dilution_engine


def get_data_merger() -> DataMerger:
    """Get or create global data merger."""
    global _data_merger
    if _data_merger is None:
        _data_merger = DataMerger()
    return _data_merger
