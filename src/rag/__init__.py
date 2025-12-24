"""
RAG (Retrieval-Augmented Generation) module for SuperInsight Platform.

Provides RAG testing interfaces for AI applications.
"""

from .service import RAGService
from .models import (
    RAGRequest, RAGResponse, DocumentChunk, RAGMetrics,
    RAGEvaluationRequest, RAGEvaluationResult
)

__all__ = [
    "RAGService",
    "RAGRequest", 
    "RAGResponse",
    "DocumentChunk",
    "RAGMetrics",
    "RAGEvaluationRequest",
    "RAGEvaluationResult"
]