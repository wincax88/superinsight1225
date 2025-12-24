"""
AI Annotator Factory for SuperInsight platform.

Provides factory methods to create and manage different AI annotators.
"""

from typing import Dict, Any, List, Optional, Type, Union
from enum import Enum

from .base import AIAnnotator, ModelConfig, ModelType, AIAnnotationError
from .ollama_annotator import OllamaAnnotator
from .huggingface_annotator import HuggingFaceAnnotator
from .zhipu_annotator import ZhipuAnnotator
from .baidu_annotator import BaiduAnnotator
from .alibaba_annotator import AlibabaAnnotator
from .hunyuan_annotator import HunyuanAnnotator
from .chatglm_annotator import ChatGLMAnnotator


class AnnotatorFactory:
    """Factory class for creating AI annotators."""
    
    # Registry of available annotator classes
    _annotators: Dict[ModelType, Type[AIAnnotator]] = {
        ModelType.OLLAMA: OllamaAnnotator,
        ModelType.HUGGINGFACE: HuggingFaceAnnotator,
        ModelType.ZHIPU_GLM: ZhipuAnnotator,
        ModelType.BAIDU_WENXIN: BaiduAnnotator,
        ModelType.ALIBABA_TONGYI: AlibabaAnnotator,
        ModelType.TENCENT_HUNYUAN: HunyuanAnnotator,
    }
    
    # Special registry for open source models that use generic types
    _special_models: Dict[str, Type[AIAnnotator]] = {
        "chatglm": ChatGLMAnnotator,
        "chatglm3": ChatGLMAnnotator,
        "chatglm2": ChatGLMAnnotator,
    }
    
    @classmethod
    def create_annotator(cls, config: ModelConfig) -> AIAnnotator:
        """
        Create an AI annotator based on the model configuration.
        
        Args:
            config: Model configuration specifying the type and parameters
            
        Returns:
            Configured AI annotator instance
            
        Raises:
            AIAnnotationError: If the model type is not supported
        """
        # Check for special models first (like ChatGLM)
        for model_prefix, annotator_class in cls._special_models.items():
            if model_prefix.lower() in config.model_name.lower():
                try:
                    return annotator_class(config)
                except Exception as e:
                    raise AIAnnotationError(
                        f"Failed to create {model_prefix} annotator: {str(e)}",
                        model_type=str(config.model_type)
                    )
        
        # Use standard model type registry
        annotator_class = cls._annotators.get(config.model_type)
        
        if not annotator_class:
            available_types = list(cls._annotators.keys())
            raise AIAnnotationError(
                f"Unsupported model type: {config.model_type}. "
                f"Available types: {available_types}",
                model_type=str(config.model_type)
            )
        
        try:
            return annotator_class(config)
        except Exception as e:
            raise AIAnnotationError(
                f"Failed to create {config.model_type} annotator: {str(e)}",
                model_type=str(config.model_type)
            )
    
    @classmethod
    def get_supported_model_types(cls) -> List[ModelType]:
        """Get list of supported model types."""
        return list(cls._annotators.keys())
    
    @classmethod
    def is_model_type_supported(cls, model_type: ModelType) -> bool:
        """Check if a model type is supported."""
        return model_type in cls._annotators
    
    @classmethod
    def register_annotator(cls, model_type: ModelType, annotator_class: Type[AIAnnotator]) -> None:
        """
        Register a new annotator class.
        
        Args:
            model_type: The model type to register
            annotator_class: The annotator class to register
        """
        cls._annotators[model_type] = annotator_class
    
    @classmethod
    def create_from_dict(cls, config_dict: Dict[str, Any]) -> AIAnnotator:
        """
        Create an annotator from a dictionary configuration.
        
        Args:
            config_dict: Dictionary containing model configuration
            
        Returns:
            Configured AI annotator instance
        """
        config = ModelConfig(**config_dict)
        return cls.create_annotator(config)
    
    @classmethod
    def create_multiple(cls, configs: List[ModelConfig]) -> List[AIAnnotator]:
        """
        Create multiple annotators from a list of configurations.
        
        Args:
            configs: List of model configurations
            
        Returns:
            List of configured AI annotator instances
        """
        annotators = []
        for config in configs:
            try:
                annotator = cls.create_annotator(config)
                annotators.append(annotator)
            except AIAnnotationError as e:
                # Log error but continue with other annotators
                print(f"Failed to create annotator for {config.model_type}: {e}")
                continue
        
        return annotators


class ConfidenceScorer:
    """Utility class for calculating and managing confidence scores."""
    
    @staticmethod
    def calculate_ensemble_confidence(confidences: List[float], method: str = "average") -> float:
        """
        Calculate ensemble confidence from multiple model predictions.
        
        Args:
            confidences: List of confidence scores from different models
            method: Method to use for ensemble calculation
            
        Returns:
            Ensemble confidence score between 0.0 and 1.0
        """
        if not confidences:
            return 0.0
        
        if method == "average":
            return sum(confidences) / len(confidences)
        elif method == "max":
            return max(confidences)
        elif method == "min":
            return min(confidences)
        elif method == "weighted_average":
            # Simple weighted average (can be enhanced with model-specific weights)
            weights = [1.0] * len(confidences)  # Equal weights for now
            weighted_sum = sum(c * w for c, w in zip(confidences, weights))
            total_weight = sum(weights)
            return weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
    
    @staticmethod
    def adjust_confidence_by_model_type(confidence: float, model_type: ModelType) -> float:
        """
        Adjust confidence score based on model type characteristics.
        
        Args:
            confidence: Original confidence score
            model_type: Type of model that generated the prediction
            
        Returns:
            Adjusted confidence score
        """
        # Model-specific confidence adjustments based on empirical performance
        adjustments = {
            ModelType.OLLAMA: 0.9,  # Local models might be less reliable
            ModelType.HUGGINGFACE: 1.0,  # Standard baseline
            ModelType.ZHIPU_GLM: 1.1,  # Chinese-optimized models
            ModelType.BAIDU_WENXIN: 1.1,  # Chinese-optimized models
            ModelType.ALIBABA_TONGYI: 1.1,  # Chinese-optimized models
            ModelType.TENCENT_HUNYUAN: 1.1,  # Chinese-optimized models
        }
        
        adjustment_factor = adjustments.get(model_type, 1.0)
        adjusted_confidence = confidence * adjustment_factor
        
        # Ensure confidence stays within valid range
        return max(0.0, min(1.0, adjusted_confidence))
    
    @staticmethod
    def calculate_confidence_from_agreement(predictions: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence based on agreement between multiple predictions.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Confidence score based on prediction agreement
        """
        if len(predictions) <= 1:
            return predictions[0].get("confidence", 0.5) if predictions else 0.0
        
        # Simple agreement calculation for sentiment analysis
        if all("sentiment" in pred for pred in predictions):
            sentiments = [pred["sentiment"] for pred in predictions]
            most_common = max(set(sentiments), key=sentiments.count)
            agreement_ratio = sentiments.count(most_common) / len(sentiments)
            return agreement_ratio
        
        # For other types, use average confidence
        confidences = [pred.get("confidence", 0.5) for pred in predictions]
        return sum(confidences) / len(confidences)
    
    @staticmethod
    def validate_confidence_range(confidence: float) -> float:
        """
        Validate and clamp confidence to valid range [0.0, 1.0].
        
        Args:
            confidence: Confidence score to validate
            
        Returns:
            Validated confidence score
        """
        if confidence < 0.0:
            return 0.0
        elif confidence > 1.0:
            return 1.0
        else:
            return confidence


class AIAnnotatorFactory:
    """
    Alias for AnnotatorFactory with additional health check capabilities.

    Provides a simpler interface for health checks and service availability testing.
    """

    # Service name to ModelType mapping
    _service_mapping: Dict[str, ModelType] = {
        "ollama": ModelType.OLLAMA,
        "huggingface": ModelType.HUGGINGFACE,
        "zhipu": ModelType.ZHIPU_GLM,
        "baidu": ModelType.BAIDU_WENXIN,
        "alibaba": ModelType.ALIBABA_TONGYI,
        "tencent": ModelType.TENCENT_HUNYUAN,
    }

    @classmethod
    def create_annotator(cls, service_name: str) -> Optional[AIAnnotator]:
        """
        Create an annotator by service name for health checking.

        Args:
            service_name: Name of the service (ollama, huggingface, zhipu, baidu, etc.)

        Returns:
            AI annotator instance or None if service is not available
        """
        model_type = cls._service_mapping.get(service_name.lower())
        if not model_type:
            return None

        try:
            # Create a minimal config for health checking
            config = ModelConfig(
                model_type=model_type,
                model_name=f"{service_name}_health_check",
                api_key="",  # Will be loaded from settings
                base_url=""  # Will use default
            )
            return AnnotatorFactory.create_annotator(config)
        except Exception:
            return None

    @classmethod
    def get_supported_services(cls) -> List[str]:
        """Get list of supported service names."""
        return list(cls._service_mapping.keys())

    @classmethod
    async def check_service_health(cls, service_name: str) -> Dict[str, Any]:
        """
        Check health of a specific AI service.

        Args:
            service_name: Name of the service to check

        Returns:
            Dictionary with health status and details
        """
        try:
            annotator = cls.create_annotator(service_name)
            if not annotator:
                return {
                    "service": service_name,
                    "available": False,
                    "error": "Service not supported"
                }

            # Check if annotator has health check method
            if hasattr(annotator, 'check_model_availability'):
                is_available = await annotator.check_model_availability()
            elif hasattr(annotator, 'test_connection'):
                is_available = await annotator.test_connection()
            else:
                # Assume available if no check method
                is_available = True

            return {
                "service": service_name,
                "available": is_available,
                "model_type": str(cls._service_mapping.get(service_name.lower()))
            }
        except Exception as e:
            return {
                "service": service_name,
                "available": False,
                "error": str(e)
            }


class ModelManager:
    """Manager class for handling multiple AI models and their configurations."""

    def __init__(self):
        """Initialize the model manager."""
        self.annotators: Dict[str, AIAnnotator] = {}
        self.default_configs: Dict[ModelType, ModelConfig] = {}
    
    def add_annotator(self, name: str, config: ModelConfig) -> None:
        """
        Add an annotator to the manager.
        
        Args:
            name: Unique name for the annotator
            config: Model configuration
        """
        annotator = AnnotatorFactory.create_annotator(config)
        self.annotators[name] = annotator
    
    def get_annotator(self, name: str) -> Optional[AIAnnotator]:
        """
        Get an annotator by name.
        
        Args:
            name: Name of the annotator
            
        Returns:
            AI annotator instance or None if not found
        """
        return self.annotators.get(name)
    
    def list_annotators(self) -> List[str]:
        """Get list of available annotator names."""
        return list(self.annotators.keys())
    
    def remove_annotator(self, name: str) -> bool:
        """
        Remove an annotator from the manager.
        
        Args:
            name: Name of the annotator to remove
            
        Returns:
            True if removed, False if not found
        """
        if name in self.annotators:
            del self.annotators[name]
            return True
        return False
    
    def set_default_config(self, model_type: ModelType, config: ModelConfig) -> None:
        """
        Set default configuration for a model type.
        
        Args:
            model_type: Type of model
            config: Default configuration
        """
        self.default_configs[model_type] = config
    
    def get_default_config(self, model_type: ModelType) -> Optional[ModelConfig]:
        """
        Get default configuration for a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Default configuration or None if not set
        """
        return self.default_configs.get(model_type)
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Check health status of all annotators.
        
        Returns:
            Dictionary mapping annotator names to health status
        """
        health_status = {}
        
        for name, annotator in self.annotators.items():
            try:
                if hasattr(annotator, 'check_model_availability'):
                    health_status[name] = await annotator.check_model_availability()
                else:
                    health_status[name] = True  # Assume healthy if no check method
            except Exception:
                health_status[name] = False
        
        return health_status