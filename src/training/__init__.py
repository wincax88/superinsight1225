"""
Training support module for SuperInsight Platform.

Provides:
- Training needs analysis and skill gap identification
- Content recommendations and learning path optimization
- Training effect tracking and ROI analysis
"""


def get_training_needs_analyzer():
    """Get TrainingNeedsAnalyzer instance with lazy import."""
    from .needs_analyzer import TrainingNeedsAnalyzer
    return TrainingNeedsAnalyzer()


def get_content_recommender():
    """Get TrainingContentRecommender instance with lazy import."""
    from .content_recommender import TrainingContentRecommender
    return TrainingContentRecommender()


def get_effect_tracker():
    """Get TrainingEffectTracker instance with lazy import."""
    from .effect_tracker import TrainingEffectTracker
    return TrainingEffectTracker()


__all__ = [
    "get_training_needs_analyzer",
    "get_content_recommender",
    "get_effect_tracker",
]
