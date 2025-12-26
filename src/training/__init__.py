"""
Training support module for SuperInsight Platform.

Provides training needs analysis and content recommendations.
"""

def get_training_needs_analyzer():
    """Get TrainingNeedsAnalyzer instance with lazy import."""
    from .needs_analyzer import TrainingNeedsAnalyzer
    return TrainingNeedsAnalyzer()


__all__ = [
    "get_training_needs_analyzer",
]
