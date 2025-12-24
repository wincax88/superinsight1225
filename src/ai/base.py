"""
Base AI Annotator classes for SuperInsight platform.

Provides abstract base classes and common functionality for AI pre-annotation services.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from uuid import UUID
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ConfigDict

# Use absolute imports to avoid relative import issues
try:
    from models.task import Task
    from models.annotation import Annotation
except ImportError:
    # Fallback for when running as part of larger application
    from src.models.task import Task
    from src.models.annotation import Annotation


class ModelType(str, Enum):
    """Enumeration of supported AI model types."""
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    ZHIPU_GLM = "zhipu_glm"
    BAIDU_WENXIN = "baidu_wenxin"
    ALIBABA_TONGYI = "alibaba_tongyi"
    TENCENT_HUNYUAN = "tencent_hunyuan"


class ModelConfig(BaseModel):
    """Configuration for AI models."""
    
    model_type: ModelType = Field(..., description="Type of AI model")
    model_name: str = Field(..., description="Name/identifier of the specific model")
    api_key: Optional[str] = Field(None, description="API key for external services")
    base_url: Optional[str] = Field(None, description="Base URL for API endpoints")
    max_tokens: int = Field(default=1000, description="Maximum tokens for generation")
    temperature: float = Field(default=0.7, description="Temperature for generation")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    
    # Additional fields for specific providers
    secret_key: Optional[str] = Field(None, description="Secret key for Tencent Cloud")
    region: Optional[str] = Field(None, description="Region for cloud services")
    
    model_config = ConfigDict(extra='allow')  # Allow additional fields
    
    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        """Validate temperature is between 0.0 and 2.0."""
        if not 0.0 <= v <= 2.0:
            raise ValueError('temperature must be between 0.0 and 2.0')
        return v
    
    @field_validator('max_tokens')
    @classmethod
    def validate_max_tokens(cls, v):
        """Validate max_tokens is positive."""
        if v <= 0:
            raise ValueError('max_tokens must be positive')
        return v
    
    @field_validator('timeout')
    @classmethod
    def validate_timeout(cls, v):
        """Validate timeout is positive."""
        if v <= 0:
            raise ValueError('timeout must be positive')
        return v


class Prediction(BaseModel):
    """AI prediction result."""
    
    id: UUID = Field(..., description="Unique prediction identifier")
    task_id: UUID = Field(..., description="Reference to annotation task")
    ai_model_config: ModelConfig = Field(..., description="Configuration used for prediction")
    prediction_data: Dict[str, Any] = Field(..., description="The actual prediction content")
    confidence: float = Field(..., description="Confidence score for this prediction")
    processing_time: float = Field(..., description="Time taken to generate prediction in seconds")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v):
        """Validate that confidence is between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('confidence must be between 0.0 and 1.0')
        return v
    
    @field_validator('processing_time')
    @classmethod
    def validate_processing_time(cls, v):
        """Validate that processing_time is non-negative."""
        if v < 0:
            raise ValueError('processing_time must be non-negative')
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "task_id": str(self.task_id),
            "model_config": self.ai_model_config.dict(),
            "prediction_data": self.prediction_data,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "created_at": self.created_at.isoformat()
        }
    
    model_config = ConfigDict(
        json_encoders={
            UUID: str,
            datetime: lambda v: v.isoformat()
        }
    )


class AIAnnotator(ABC):
    """
    Abstract base class for AI annotation services.
    
    Defines the interface that all AI annotators must implement.
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize the AI annotator with configuration."""
        self.config = config
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate the model configuration. Implemented by subclasses."""
        pass
    
    @abstractmethod
    async def predict(self, task: Task) -> Prediction:
        """
        Generate a prediction for a single task.
        
        Args:
            task: The annotation task to predict
            
        Returns:
            Prediction object with results and confidence score
            
        Raises:
            AIAnnotationError: If prediction fails
        """
        pass
    
    async def batch_predict(self, tasks: List[Task]) -> List[Prediction]:
        """
        Generate predictions for multiple tasks.
        
        Default implementation processes tasks sequentially.
        Subclasses can override for parallel processing.
        
        Args:
            tasks: List of annotation tasks to predict
            
        Returns:
            List of prediction objects
        """
        predictions = []
        for task in tasks:
            try:
                prediction = await self.predict(task)
                predictions.append(prediction)
            except Exception as e:
                # Log error but continue with other tasks
                print(f"Failed to predict task {task.id}: {e}")
                continue
        return predictions
    
    def get_confidence_score(self, prediction: Prediction) -> float:
        """
        Get confidence score for a prediction.
        
        Args:
            prediction: The prediction object
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        return prediction.confidence
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        pass
    
    def update_config(self, new_config: ModelConfig) -> None:
        """
        Update the model configuration.
        
        Args:
            new_config: New configuration to use
        """
        self.config = new_config
        self._validate_config()


class AIAnnotationError(Exception):
    """Exception raised when AI annotation fails."""
    
    def __init__(self, message: str, model_type: str, task_id: Optional[UUID] = None):
        self.message = message
        self.model_type = model_type
        self.task_id = task_id
        super().__init__(self.message)
    
    def __str__(self):
        base_msg = f"AI Annotation Error ({self.model_type}): {self.message}"
        if self.task_id:
            base_msg += f" (Task ID: {self.task_id})"
        return base_msg


class ModelUpdateResult(BaseModel):
    """Result of model update operation."""
    
    success: bool = Field(..., description="Whether the update was successful")
    message: str = Field(..., description="Status message")
    model_version: Optional[str] = Field(None, description="New model version if applicable")
    updated_at: datetime = Field(default_factory=datetime.now, description="Update timestamp")