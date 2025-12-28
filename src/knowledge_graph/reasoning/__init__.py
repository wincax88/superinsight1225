"""
Reasoning module for Knowledge Graph.

Provides rule-based and ML-based inference.
"""

from .rule_engine import (
    RuleType,
    RulePriority,
    ConditionOperator,
    Condition,
    Action,
    Rule,
    InferenceStep,
    InferenceChain,
    InferredFact,
    RuleEngine,
    get_rule_engine,
)

from .ml_inference import (
    InferenceMethod,
    EmbeddingModel,
    EntityEmbedding,
    RelationEmbedding,
    LinkPredictionResult,
    EntityAlignmentResult,
    InferenceEvaluation,
    TrainingConfig,
    MLInference,
    get_ml_inference,
)

from .explanation import (
    ExplanationType,
    ConfidenceLevel,
    ValidationStatus,
    EvidenceItem,
    ReasoningPath,
    Explanation,
    ValidationResult,
    ConfidenceBreakdown,
    ReasoningExplainer,
    get_reasoning_explainer,
)

__all__ = [
    # Rule Engine
    "RuleType",
    "RulePriority",
    "ConditionOperator",
    "Condition",
    "Action",
    "Rule",
    "InferenceStep",
    "InferenceChain",
    "InferredFact",
    "RuleEngine",
    "get_rule_engine",
    # ML Inference
    "InferenceMethod",
    "EmbeddingModel",
    "EntityEmbedding",
    "RelationEmbedding",
    "LinkPredictionResult",
    "EntityAlignmentResult",
    "InferenceEvaluation",
    "TrainingConfig",
    "MLInference",
    "get_ml_inference",
    # Explanation
    "ExplanationType",
    "ConfidenceLevel",
    "ValidationStatus",
    "EvidenceItem",
    "ReasoningPath",
    "Explanation",
    "ValidationResult",
    "ConfidenceBreakdown",
    "ReasoningExplainer",
    "get_reasoning_explainer",
]
