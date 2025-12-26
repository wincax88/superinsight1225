"""
Feedback management module for SuperInsight Platform.

Provides feedback collection and processing.
"""

def get_feedback_collector():
    """Get FeedbackCollector instance with lazy import."""
    from .collector import FeedbackCollector
    return FeedbackCollector()


__all__ = [
    "get_feedback_collector",
]
