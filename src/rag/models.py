"""
RAG data models for SuperInsight Platform.

Defines data structures for RAG operations.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID
from pydantic import BaseModel, Field, field_validator


class DocumentChunk(BaseModel):
    """Model for document chunks used in RAG."""
    
    id: str = Field(..., description="Chunk identifier")
    document_id: str = Field(..., description="Source document ID")
    content: str = Field(..., description="Chunk content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    similarity_score: Optional[float] = Field(None, description="Similarity score to query")
    
    @field_validator('similarity_score')
    @classmethod
    def validate_similarity_score(cls, v):
        """Validate similarity score is between 0 and 1."""
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError('similarity_score must be between 0.0 and 1.0')
        return v


class RAGRequest(BaseModel):
    """Request model for RAG operations."""
    
    query: str = Field(..., description="Search query")
    project_id: Optional[str] = Field(None, description="Filter by project ID")
    document_ids: Optional[List[str]] = Field(None, description="Filter by document IDs")
    top_k: int = Field(5, ge=1, le=50, description="Number of top results to return")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity threshold")
    include_metadata: bool = Field(True, description="Include document metadata")
    chunk_size: int = Field(512, ge=100, le=2000, description="Text chunk size for processing")
    chunk_overlap: int = Field(50, ge=0, le=500, description="Overlap between chunks")
    
    @field_validator('document_ids')
    @classmethod
    def validate_document_ids(cls, v):
        """Validate document IDs are valid UUIDs if provided."""
        if v is not None:
            for doc_id in v:
                try:
                    UUID(doc_id)
                except ValueError:
                    raise ValueError(f'Invalid UUID format: {doc_id}')
        return v


class RAGResponse(BaseModel):
    """Response model for RAG operations."""
    
    query: str = Field(..., description="Original query")
    chunks: List[DocumentChunk] = Field(..., description="Retrieved document chunks")
    total_chunks: int = Field(..., description="Total number of chunks found")
    processing_time: float = Field(..., description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


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


class RAGMetrics(BaseModel):
    """Model for RAG performance metrics."""
    
    query_count: int = Field(..., description="Number of queries processed")
    avg_response_time: float = Field(..., description="Average response time")
    avg_chunks_returned: float = Field(..., description="Average chunks per query")
    avg_similarity_score: float = Field(..., description="Average similarity score")
    cache_hit_rate: float = Field(..., description="Cache hit rate percentage")
    
    @field_validator('cache_hit_rate')
    @classmethod
    def validate_cache_hit_rate(cls, v):
        """Validate cache hit rate is between 0 and 100."""
        if not 0.0 <= v <= 100.0:
            raise ValueError('cache_hit_rate must be between 0.0 and 100.0')
        return v


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