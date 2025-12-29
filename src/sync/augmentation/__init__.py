"""
Data Augmentation Module.

Provides data augmentation capabilities for AI-friendly dataset generation.
"""

from src.sync.augmentation.text_augmentation import (
    TextAugmentationEngine,
    QualitySampleAmplifier,
    AugmentationConfig,
    AugmentationStrategy,
    AugmentedSample,
    AugmentationMetrics,
    TextAugmenter,
    SynonymReplacer,
    RandomSwapper,
    RandomDeleter,
    RandomInserter,
    NoiseInjector,
    BackTranslator,
    get_augmentation_engine,
    get_sample_amplifier,
)

__all__ = [
    # Augmentation Engine
    "TextAugmentationEngine",
    "QualitySampleAmplifier",
    "AugmentationConfig",
    "AugmentationStrategy",
    "AugmentedSample",
    "AugmentationMetrics",
    # Augmenters
    "TextAugmenter",
    "SynonymReplacer",
    "RandomSwapper",
    "RandomDeleter",
    "RandomInserter",
    "NoiseInjector",
    "BackTranslator",
    # Global accessors
    "get_augmentation_engine",
    "get_sample_amplifier",
]
