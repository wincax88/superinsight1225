"""
AI module for SuperInsight platform.

Provides AI pre-annotation services using various models and APIs.
"""

from .base import (
    AIAnnotator,
    ModelConfig,
    ModelType,
    Prediction,
    AIAnnotationError,
    ModelUpdateResult
)

from .factory import (
    AnnotatorFactory,
    ConfidenceScorer,
    ModelManager
)

from .batch_processor import (
    BatchAnnotationProcessor,
    BatchJobConfig,
    BatchResult,
    BatchStatus
)

from .model_manager import (
    ModelVersionManager,
    ModelVersion,
    ModelStatus
)

from .cache_service import (
    PredictionCacheService,
    CacheConfig,
    CacheStrategy
)

from .model_performance import (
    ModelPerformanceAnalyzer,
    ModelAutoSelector,
    PerformanceMetric,
    ModelPerformanceData
)

from .enhanced_model_manager import (
    EnhancedModelManager
)

from .ollama_annotator import OllamaAnnotator
from .huggingface_annotator import HuggingFaceAnnotator
from .zhipu_annotator import ZhipuAnnotator
from .baidu_annotator import BaiduAnnotator
from .alibaba_annotator import AlibabaAnnotator
from .chatglm_annotator import ChatGLMAnnotator

__all__ = [
    # Base classes
    "AIAnnotator",
    "ModelConfig", 
    "ModelType",
    "Prediction",
    "AIAnnotationError",
    "ModelUpdateResult",
    
    # Factory and utilities
    "AnnotatorFactory",
    "ConfidenceScorer",
    "ModelManager",
    
    # Batch processing
    "BatchAnnotationProcessor",
    "BatchJobConfig",
    "BatchResult",
    "BatchStatus",
    
    # Model version management
    "ModelVersionManager",
    "ModelVersion",
    "ModelStatus",
    
    # Caching
    "PredictionCacheService",
    "CacheConfig",
    "CacheStrategy",
    
    # Performance analysis and auto-selection
    "ModelPerformanceAnalyzer",
    "ModelAutoSelector",
    "PerformanceMetric",
    "ModelPerformanceData",
    "EnhancedModelManager",
    
    # Specific annotators
    "OllamaAnnotator",
    "HuggingFaceAnnotator", 
    "ZhipuAnnotator",
    "BaiduAnnotator",
    "AlibabaAnnotator",
    "ChatGLMAnnotator",
]