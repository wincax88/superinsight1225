"""
Data models for data enhancement functionality.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class EnhancementType(str, Enum):
    """Types of data enhancement operations."""
    QUALITY_SAMPLE_FILL = "quality_sample_fill"
    POSITIVE_AMPLIFICATION = "positive_amplification"
    BATCH_ENHANCEMENT = "batch_enhancement"


class QualitySample(BaseModel):
    """Model representing a high-quality sample for enhancement."""
    
    id: UUID = Field(default_factory=uuid4, description="Unique sample identifier")
    content: str = Field(..., description="Sample content")
    quality_score: float = Field(..., description="Quality score (0.0-1.0)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    @field_validator('quality_score')
    @classmethod
    def validate_quality_score(cls, v):
        """Validate that quality_score is between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('quality_score must be between 0.0 and 1.0')
        return v
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        """Validate that content is not empty."""
        if not v or not v.strip():
            raise ValueError('content cannot be empty')
        return v


class EnhancementConfig(BaseModel):
    """Configuration for data enhancement operations."""
    
    enhancement_type: EnhancementType = Field(..., description="Type of enhancement to perform")
    target_quality_threshold: float = Field(default=0.8, description="Minimum quality threshold")
    amplification_factor: float = Field(default=2.0, description="Factor for positive data amplification")
    batch_size: int = Field(default=100, description="Batch size for processing")
    preserve_original: bool = Field(default=True, description="Whether to preserve original data")
    
    @field_validator('target_quality_threshold')
    @classmethod
    def validate_quality_threshold(cls, v):
        """Validate that quality threshold is between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('target_quality_threshold must be between 0.0 and 1.0')
        return v
    
    @field_validator('amplification_factor')
    @classmethod
    def validate_amplification_factor(cls, v):
        """Validate that amplification factor is positive."""
        if v <= 0:
            raise ValueError('amplification_factor must be positive')
        return v
    
    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v):
        """Validate that batch size is positive."""
        if v <= 0:
            raise ValueError('batch_size must be positive')
        return v


class EnhancementResult(BaseModel):
    """Result of a data enhancement operation."""
    
    id: UUID = Field(default_factory=uuid4, description="Unique result identifier")
    enhancement_type: EnhancementType = Field(..., description="Type of enhancement performed")
    original_count: int = Field(..., description="Number of original samples")
    enhanced_count: int = Field(..., description="Number of enhanced samples")
    quality_improvement: float = Field(..., description="Quality score improvement")
    processing_time: float = Field(..., description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional result metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    @field_validator('original_count', 'enhanced_count')
    @classmethod
    def validate_counts(cls, v):
        """Validate that counts are non-negative."""
        if v < 0:
            raise ValueError('counts must be non-negative')
        return v
    
    @field_validator('processing_time')
    @classmethod
    def validate_processing_time(cls, v):
        """Validate that processing time is non-negative."""
        if v < 0:
            raise ValueError('processing_time must be non-negative')
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "enhancement_type": self.enhancement_type.value,
            "original_count": self.original_count,
            "enhanced_count": self.enhanced_count,
            "quality_improvement": self.quality_improvement,
            "processing_time": self.processing_time,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancementResult':
        """Create result from dictionary (JSON deserialization)."""
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data.get('id'), str):
            data['id'] = UUID(data['id'])
        return cls(**data)