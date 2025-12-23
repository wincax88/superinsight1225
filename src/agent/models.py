"""
Agent data models for SuperInsight Platform.

Defines data structures for Agent operations.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, field_validator


class AgentRequest(BaseModel):
    """Request model for Agent testing."""
    
    task_type: str = Field(..., description="Type of agent task")
    input_data: Dict[str, Any] = Field(..., description="Input data for the agent")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    project_id: Optional[str] = Field(None, description="Project context")
    max_iterations: int = Field(10, ge=1, le=100, description="Maximum agent iterations")
    timeout: int = Field(30, ge=5, le=300, description="Timeout in seconds")
    
    @field_validator('task_type')
    @classmethod
    def validate_task_type(cls, v):
        """Validate task type is supported."""
        allowed_types = {
            'classification', 'extraction', 'summarization', 
            'question_answering', 'text_generation', 'analysis'
        }
        if v not in allowed_types:
            raise ValueError(f'task_type must be one of {allowed_types}')
        return v


class AgentStep(BaseModel):
    """Model for individual agent execution steps."""
    
    step_number: int = Field(..., description="Step number in execution")
    action: str = Field(..., description="Action taken")
    input_data: Dict[str, Any] = Field(..., description="Input for this step")
    output_data: Dict[str, Any] = Field(..., description="Output from this step")
    confidence: float = Field(..., description="Confidence score for this step")
    execution_time: float = Field(..., description="Execution time for this step")
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v):
        """Validate confidence is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('confidence must be between 0.0 and 1.0')
        return v


class AgentResponse(BaseModel):
    """Response model for Agent operations."""
    
    task_type: str = Field(..., description="Type of agent task")
    status: str = Field(..., description="Execution status")
    result: Dict[str, Any] = Field(..., description="Final result")
    steps: List[AgentStep] = Field(..., description="Execution steps")
    total_steps: int = Field(..., description="Total number of steps")
    execution_time: float = Field(..., description="Total execution time")
    confidence: float = Field(..., description="Overall confidence score")
    error: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AgentMetrics(BaseModel):
    """Model for Agent performance metrics."""
    
    task_count: int = Field(..., description="Number of tasks processed")
    success_rate: float = Field(..., description="Success rate percentage")
    avg_execution_time: float = Field(..., description="Average execution time")
    avg_steps: float = Field(..., description="Average steps per task")
    avg_confidence: float = Field(..., description="Average confidence score")
    
    @field_validator('success_rate')
    @classmethod
    def validate_success_rate(cls, v):
        """Validate success rate is between 0 and 100."""
        if not 0.0 <= v <= 100.0:
            raise ValueError('success_rate must be between 0.0 and 100.0')
        return v