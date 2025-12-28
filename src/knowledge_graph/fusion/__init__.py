"""
Fusion module for Knowledge Graph.

Provides entity alignment and knowledge merging.
"""

from .entity_alignment import (
    AlignmentMethod,
    SimilarityMetric,
    AlignmentStatus,
    EntityCandidate,
    AlignmentPair,
    AlignmentConfig,
    AlignmentResult,
    BlockingConfig,
    EntityAligner,
    get_entity_aligner,
)

from .knowledge_merger import (
    MergeStrategy,
    QualityMetric,
    ExternalKBType,
    KnowledgeSource,
    KnowledgeFact,
    MergedFact,
    QualityAssessment,
    MergeConfig,
    MergeResult,
    KnowledgeMerger,
    get_knowledge_merger,
)

from .conflict_resolver import (
    ConflictType,
    ResolutionStrategy,
    ConflictSeverity,
    ConflictingValue,
    Conflict,
    ResolutionRule,
    ResolutionResult,
    ValidationResult,
    ConflictStats,
    ConflictResolver,
    get_conflict_resolver,
)

__all__ = [
    # Entity Alignment
    "AlignmentMethod",
    "SimilarityMetric",
    "AlignmentStatus",
    "EntityCandidate",
    "AlignmentPair",
    "AlignmentConfig",
    "AlignmentResult",
    "BlockingConfig",
    "EntityAligner",
    "get_entity_aligner",
    # Knowledge Merger
    "MergeStrategy",
    "QualityMetric",
    "ExternalKBType",
    "KnowledgeSource",
    "KnowledgeFact",
    "MergedFact",
    "QualityAssessment",
    "MergeConfig",
    "MergeResult",
    "KnowledgeMerger",
    "get_knowledge_merger",
    # Conflict Resolver
    "ConflictType",
    "ResolutionStrategy",
    "ConflictSeverity",
    "ConflictingValue",
    "Conflict",
    "ResolutionRule",
    "ResolutionResult",
    "ValidationResult",
    "ConflictStats",
    "ConflictResolver",
    "get_conflict_resolver",
]
