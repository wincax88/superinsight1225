"""
Text Augmentation Engine for AI-Friendly Data Generation.

This module provides multi-strategy text augmentation capabilities
for enhancing data quality and increasing sample diversity.
"""

import asyncio
import logging
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AugmentationStrategy(str, Enum):
    """Text augmentation strategies."""

    SYNONYM_REPLACEMENT = "synonym_replacement"
    RANDOM_INSERTION = "random_insertion"
    RANDOM_SWAP = "random_swap"
    RANDOM_DELETION = "random_deletion"
    BACK_TRANSLATION = "back_translation"
    PARAPHRASE = "paraphrase"
    CONTEXTUAL_EMBEDDING = "contextual_embedding"
    NOISE_INJECTION = "noise_injection"


class AugmentationConfig(BaseModel):
    """Configuration for text augmentation."""

    strategies: List[AugmentationStrategy] = Field(
        default_factory=lambda: [
            AugmentationStrategy.SYNONYM_REPLACEMENT,
            AugmentationStrategy.RANDOM_SWAP,
        ]
    )
    augmentation_factor: int = Field(
        default=3, ge=1, le=10, description="Number of augmented samples per original"
    )
    min_text_length: int = Field(
        default=10, description="Minimum text length to augment"
    )
    max_modifications_per_sample: float = Field(
        default=0.3, ge=0.0, le=0.5, description="Max ratio of words to modify"
    )
    preserve_semantic_meaning: bool = Field(
        default=True, description="Try to preserve original meaning"
    )
    language: str = Field(default="en", description="Primary language")
    seed: Optional[int] = None

    class Config:
        use_enum_values = True


@dataclass
class AugmentedSample:
    """An augmented text sample."""

    original_id: str
    original_text: str
    augmented_text: str
    strategy: AugmentationStrategy
    modifications: List[Dict[str, str]] = field(default_factory=list)
    similarity_score: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_id": self.original_id,
            "original_text": self.original_text,
            "augmented_text": self.augmented_text,
            "strategy": self.strategy.value,
            "modifications_count": len(self.modifications),
            "similarity_score": round(self.similarity_score, 3),
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class AugmentationMetrics:
    """Metrics from augmentation operations."""

    original_count: int = 0
    augmented_count: int = 0
    total_count: int = 0
    avg_similarity: float = 0.0
    strategy_distribution: Dict[str, int] = field(default_factory=dict)
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_count": self.original_count,
            "augmented_count": self.augmented_count,
            "total_count": self.total_count,
            "augmentation_factor": (
                self.augmented_count / self.original_count
                if self.original_count > 0
                else 0
            ),
            "avg_similarity": round(self.avg_similarity, 3),
            "strategy_distribution": self.strategy_distribution,
            "execution_time_ms": round(self.execution_time_ms, 2),
        }


class TextAugmenter(ABC):
    """Abstract base class for text augmenters."""

    @abstractmethod
    def augment(self, text: str, **kwargs) -> str:
        """Augment a text string."""
        pass

    @property
    @abstractmethod
    def strategy(self) -> AugmentationStrategy:
        """Get the augmentation strategy."""
        pass


class SynonymReplacer(TextAugmenter):
    """Replace words with synonyms."""

    # Simple synonym dictionary (in production, use WordNet or similar)
    SYNONYMS = {
        # Common verbs
        "good": ["excellent", "great", "fine", "wonderful", "superior"],
        "bad": ["poor", "terrible", "awful", "inferior", "unsatisfactory"],
        "big": ["large", "huge", "enormous", "massive", "substantial"],
        "small": ["little", "tiny", "minor", "minimal", "compact"],
        "fast": ["quick", "rapid", "swift", "speedy", "prompt"],
        "slow": ["gradual", "leisurely", "unhurried", "sluggish"],
        "important": ["significant", "crucial", "vital", "essential", "key"],
        "show": ["demonstrate", "display", "present", "reveal", "indicate"],
        "make": ["create", "produce", "build", "construct", "generate"],
        "get": ["obtain", "acquire", "receive", "gain", "achieve"],
        "use": ["utilize", "employ", "apply", "leverage"],
        "help": ["assist", "support", "aid", "facilitate"],
        "need": ["require", "demand", "necessitate"],
        "want": ["desire", "wish", "seek"],
        # Common adjectives
        "new": ["novel", "recent", "fresh", "modern", "latest"],
        "old": ["ancient", "aged", "former", "previous"],
        "different": ["distinct", "various", "diverse", "unique"],
        "same": ["identical", "similar", "equivalent"],
        "easy": ["simple", "straightforward", "effortless"],
        "hard": ["difficult", "challenging", "tough", "complex"],
        # Common nouns
        "problem": ["issue", "challenge", "difficulty", "concern"],
        "solution": ["answer", "resolution", "remedy", "fix"],
        "result": ["outcome", "consequence", "effect", "finding"],
        "change": ["modification", "alteration", "adjustment", "shift"],
        "increase": ["growth", "rise", "gain", "boost"],
        "decrease": ["reduction", "decline", "drop", "fall"],
        "data": ["information", "details", "facts", "records"],
        "system": ["framework", "structure", "platform", "mechanism"],
    }

    def __init__(self, replacement_ratio: float = 0.2, seed: Optional[int] = None):
        self.replacement_ratio = replacement_ratio
        self.rng = random.Random(seed)

    @property
    def strategy(self) -> AugmentationStrategy:
        return AugmentationStrategy.SYNONYM_REPLACEMENT

    def augment(self, text: str, **kwargs) -> str:
        words = text.split()
        if not words:
            return text

        num_to_replace = max(1, int(len(words) * self.replacement_ratio))
        replaceable_indices = [
            i for i, w in enumerate(words) if w.lower() in self.SYNONYMS
        ]

        if not replaceable_indices:
            return text

        indices_to_replace = self.rng.sample(
            replaceable_indices, min(num_to_replace, len(replaceable_indices))
        )

        result = words.copy()
        for idx in indices_to_replace:
            word = words[idx].lower()
            synonyms = self.SYNONYMS.get(word, [])
            if synonyms:
                replacement = self.rng.choice(synonyms)
                # Preserve capitalization
                if words[idx][0].isupper():
                    replacement = replacement.capitalize()
                result[idx] = replacement

        return " ".join(result)


class RandomSwapper(TextAugmenter):
    """Randomly swap words in a sentence."""

    def __init__(self, swap_ratio: float = 0.1, seed: Optional[int] = None):
        self.swap_ratio = swap_ratio
        self.rng = random.Random(seed)

    @property
    def strategy(self) -> AugmentationStrategy:
        return AugmentationStrategy.RANDOM_SWAP

    def augment(self, text: str, **kwargs) -> str:
        words = text.split()
        if len(words) < 2:
            return text

        num_swaps = max(1, int(len(words) * self.swap_ratio))

        for _ in range(num_swaps):
            idx1, idx2 = self.rng.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]

        return " ".join(words)


class RandomDeleter(TextAugmenter):
    """Randomly delete words from text."""

    def __init__(self, deletion_ratio: float = 0.1, seed: Optional[int] = None):
        self.deletion_ratio = deletion_ratio
        self.rng = random.Random(seed)

    @property
    def strategy(self) -> AugmentationStrategy:
        return AugmentationStrategy.RANDOM_DELETION

    def augment(self, text: str, **kwargs) -> str:
        words = text.split()
        if len(words) < 3:
            return text

        # Don't delete too many words
        num_to_keep = max(
            len(words) // 2, int(len(words) * (1 - self.deletion_ratio))
        )
        indices_to_keep = sorted(self.rng.sample(range(len(words)), num_to_keep))

        return " ".join(words[i] for i in indices_to_keep)


class RandomInserter(TextAugmenter):
    """Randomly insert words into text."""

    FILLER_WORDS = [
        "also", "actually", "basically", "certainly", "clearly",
        "definitely", "especially", "extremely", "generally",
        "indeed", "likely", "mainly", "mostly", "nearly",
        "obviously", "particularly", "possibly", "primarily",
        "probably", "quite", "rather", "really", "simply",
        "somewhat", "specifically", "strongly", "surely", "truly",
        "typically", "usually", "very",
    ]

    def __init__(self, insertion_ratio: float = 0.1, seed: Optional[int] = None):
        self.insertion_ratio = insertion_ratio
        self.rng = random.Random(seed)

    @property
    def strategy(self) -> AugmentationStrategy:
        return AugmentationStrategy.RANDOM_INSERTION

    def augment(self, text: str, **kwargs) -> str:
        words = text.split()
        if not words:
            return text

        num_insertions = max(1, int(len(words) * self.insertion_ratio))
        result = words.copy()

        for _ in range(num_insertions):
            insert_pos = self.rng.randint(0, len(result))
            insert_word = self.rng.choice(self.FILLER_WORDS)
            result.insert(insert_pos, insert_word)

        return " ".join(result)


class NoiseInjector(TextAugmenter):
    """Inject typos and noise into text."""

    def __init__(self, noise_ratio: float = 0.05, seed: Optional[int] = None):
        self.noise_ratio = noise_ratio
        self.rng = random.Random(seed)

    @property
    def strategy(self) -> AugmentationStrategy:
        return AugmentationStrategy.NOISE_INJECTION

    def augment(self, text: str, **kwargs) -> str:
        if not text:
            return text

        chars = list(text)
        num_changes = max(1, int(len(chars) * self.noise_ratio))

        for _ in range(num_changes):
            if not chars:
                break
            idx = self.rng.randint(0, len(chars) - 1)
            if chars[idx].isalpha():
                operation = self.rng.choice(["swap", "delete", "duplicate"])
                if operation == "swap" and idx < len(chars) - 1:
                    chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
                elif operation == "delete":
                    chars.pop(idx)
                elif operation == "duplicate":
                    chars.insert(idx, chars[idx])

        return "".join(chars)


class BackTranslator(TextAugmenter):
    """Simulate back-translation augmentation."""

    # Simulated paraphrases (in production, use actual translation APIs)
    PARAPHRASE_PATTERNS = [
        (r"\bthe\b", "a"),
        (r"\ba\b", "the"),
        (r"\bvery\b", "extremely"),
        (r"\bextremely\b", "very"),
        (r"\bis\b", "remains"),
        (r"\bare\b", "remain"),
        (r"\bwas\b", "had been"),
        (r"\bwere\b", "had been"),
        (r"\bhas\b", "possesses"),
        (r"\bhave\b", "possess"),
        (r"\bcan\b", "is able to"),
        (r"\bwill\b", "shall"),
        (r"\bbut\b", "however"),
        (r"\band\b", "as well as"),
        (r"\bbecause\b", "since"),
        (r"\bif\b", "in case"),
    ]

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    @property
    def strategy(self) -> AugmentationStrategy:
        return AugmentationStrategy.BACK_TRANSLATION

    def augment(self, text: str, **kwargs) -> str:
        result = text

        # Apply random subset of paraphrase patterns
        patterns_to_apply = self.rng.sample(
            self.PARAPHRASE_PATTERNS,
            min(3, len(self.PARAPHRASE_PATTERNS)),
        )

        for pattern, replacement in patterns_to_apply:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        return result


class TextAugmentationEngine:
    """
    Main engine for text augmentation.

    Supports multiple augmentation strategies and can generate
    diverse augmented samples while preserving semantic meaning.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        seed = self.config.seed

        self.augmenters: Dict[AugmentationStrategy, TextAugmenter] = {
            AugmentationStrategy.SYNONYM_REPLACEMENT: SynonymReplacer(seed=seed),
            AugmentationStrategy.RANDOM_SWAP: RandomSwapper(seed=seed),
            AugmentationStrategy.RANDOM_DELETION: RandomDeleter(seed=seed),
            AugmentationStrategy.RANDOM_INSERTION: RandomInserter(seed=seed),
            AugmentationStrategy.NOISE_INJECTION: NoiseInjector(seed=seed),
            AugmentationStrategy.BACK_TRANSLATION: BackTranslator(seed=seed),
        }

        self.rng = random.Random(seed)

    def _calculate_similarity(self, original: str, augmented: str) -> float:
        """Calculate text similarity score."""
        if not original or not augmented:
            return 0.0

        # Simple word overlap similarity
        original_words = set(original.lower().split())
        augmented_words = set(augmented.lower().split())

        if not original_words:
            return 0.0

        intersection = original_words & augmented_words
        union = original_words | augmented_words

        # Jaccard similarity
        return len(intersection) / len(union) if union else 0.0

    async def augment_text(
        self,
        text: str,
        sample_id: str,
        num_augmentations: Optional[int] = None,
        strategies: Optional[List[AugmentationStrategy]] = None,
    ) -> List[AugmentedSample]:
        """
        Augment a single text with multiple strategies.

        Args:
            text: Original text to augment
            sample_id: ID of the original sample
            num_augmentations: Number of augmented samples to generate
            strategies: Specific strategies to use

        Returns:
            List of augmented samples
        """
        if len(text) < self.config.min_text_length:
            return []

        num_augmentations = num_augmentations or self.config.augmentation_factor
        strategies = strategies or [
            AugmentationStrategy(s) for s in self.config.strategies
        ]

        results = []

        for i in range(num_augmentations):
            # Select strategy for this augmentation
            strategy = self.rng.choice(strategies)
            augmenter = self.augmenters.get(strategy)

            if not augmenter:
                continue

            try:
                augmented_text = augmenter.augment(text)

                # Calculate similarity
                similarity = self._calculate_similarity(text, augmented_text)

                # Only keep if sufficiently different but not too different
                if 0.4 <= similarity <= 0.95:
                    results.append(
                        AugmentedSample(
                            original_id=sample_id,
                            original_text=text,
                            augmented_text=augmented_text,
                            strategy=strategy,
                            similarity_score=similarity,
                        )
                    )
            except Exception as e:
                logger.warning(f"Augmentation failed for strategy {strategy}: {e}")

        return results

    async def augment_batch(
        self,
        samples: List[Dict[str, Any]],
        text_field: str = "text",
        id_field: str = "id",
    ) -> Tuple[List[AugmentedSample], AugmentationMetrics]:
        """
        Augment a batch of samples.

        Args:
            samples: List of sample dictionaries
            text_field: Name of the text field
            id_field: Name of the ID field

        Returns:
            Tuple of (augmented_samples, metrics)
        """
        import time

        start_time = time.time()
        metrics = AugmentationMetrics(original_count=len(samples))

        all_augmented = []

        for sample in samples:
            text = sample.get(text_field, "")
            sample_id = sample.get(id_field, str(id(sample)))

            augmented = await self.augment_text(text, sample_id)
            all_augmented.extend(augmented)

            # Update strategy distribution
            for aug in augmented:
                strategy_name = aug.strategy.value
                metrics.strategy_distribution[strategy_name] = (
                    metrics.strategy_distribution.get(strategy_name, 0) + 1
                )

        metrics.augmented_count = len(all_augmented)
        metrics.total_count = metrics.original_count + metrics.augmented_count

        if all_augmented:
            metrics.avg_similarity = sum(a.similarity_score for a in all_augmented) / len(
                all_augmented
            )

        metrics.execution_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Augmented {metrics.original_count} samples into {metrics.total_count} total "
            f"(factor: {metrics.augmented_count / metrics.original_count:.1f}x)"
        )

        return all_augmented, metrics

    def get_available_strategies(self) -> List[AugmentationStrategy]:
        """Get list of available augmentation strategies."""
        return list(self.augmenters.keys())


class QualitySampleAmplifier:
    """
    Amplifies high-quality samples through targeted augmentation.

    Identifies quality samples and applies augmentation strategies
    to increase their representation in the dataset.
    """

    def __init__(
        self,
        augmentation_engine: Optional[TextAugmentationEngine] = None,
        quality_threshold: float = 0.8,
        target_amplification: int = 5,
    ):
        self.engine = augmentation_engine or TextAugmentationEngine()
        self.quality_threshold = quality_threshold
        self.target_amplification = target_amplification

    async def amplify(
        self,
        samples: List[Dict[str, Any]],
        quality_scores: Dict[str, float],
        text_field: str = "text",
        id_field: str = "id",
    ) -> Tuple[List[AugmentedSample], Dict[str, Any]]:
        """
        Amplify high-quality samples.

        Args:
            samples: List of sample dictionaries
            quality_scores: Quality scores by sample ID
            text_field: Name of the text field
            id_field: Name of the ID field

        Returns:
            Tuple of (augmented_samples, amplification_report)
        """
        # Identify high-quality samples
        high_quality = [
            sample
            for sample in samples
            if quality_scores.get(sample.get(id_field, ""), 0) >= self.quality_threshold
        ]

        if not high_quality:
            return [], {"message": "No high-quality samples found for amplification"}

        # Augment high-quality samples more aggressively
        all_augmented = []
        for sample in high_quality:
            text = sample.get(text_field, "")
            sample_id = sample.get(id_field, str(id(sample)))

            augmented = await self.engine.augment_text(
                text, sample_id, num_augmentations=self.target_amplification
            )
            all_augmented.extend(augmented)

        report = {
            "total_samples": len(samples),
            "high_quality_samples": len(high_quality),
            "augmented_samples": len(all_augmented),
            "effective_amplification": (
                len(all_augmented) / len(high_quality) if high_quality else 0
            ),
            "quality_threshold": self.quality_threshold,
        }

        return all_augmented, report


# Global instances
_augmentation_engine: Optional[TextAugmentationEngine] = None
_sample_amplifier: Optional[QualitySampleAmplifier] = None


def get_augmentation_engine(
    config: Optional[AugmentationConfig] = None,
) -> TextAugmentationEngine:
    """Get or create global augmentation engine."""
    global _augmentation_engine
    if _augmentation_engine is None or config:
        _augmentation_engine = TextAugmentationEngine(config)
    return _augmentation_engine


def get_sample_amplifier(
    quality_threshold: float = 0.8,
    target_amplification: int = 5,
) -> QualitySampleAmplifier:
    """Get or create global sample amplifier."""
    global _sample_amplifier
    if _sample_amplifier is None:
        _sample_amplifier = QualitySampleAmplifier(
            quality_threshold=quality_threshold,
            target_amplification=target_amplification,
        )
    return _sample_amplifier
