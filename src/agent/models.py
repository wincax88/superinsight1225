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
            'question_answering', 'text_generation', 'analysis',
            'conversation', 'chat', 'follow_up', 'clarification', 'context_aware'
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
    """Model for enhanced Agent performance metrics."""
    
    task_count: int = Field(..., description="Number of tasks processed")
    success_rate: float = Field(..., description="Success rate percentage")
    avg_execution_time: float = Field(..., description="Average execution time")
    avg_steps: float = Field(..., description="Average steps per task")
    avg_confidence: float = Field(..., description="Average confidence score")
    conversation_count: int = Field(0, description="Total conversations created")
    multi_turn_sessions: int = Field(0, description="Multi-turn conversation sessions")
    avg_conversation_length: float = Field(0.0, description="Average messages per conversation")
    avg_response_quality: float = Field(0.0, description="Average response quality score")
    context_utilization_rate: float = Field(0.0, description="Context utilization rate percentage")
    
    @field_validator('success_rate', 'context_utilization_rate')
    @classmethod
    def validate_percentage(cls, v):
        """Validate percentage is between 0 and 100."""
        if not 0.0 <= v <= 100.0:
            raise ValueError('percentage must be between 0.0 and 100.0')
        return v


class ConversationMessage(BaseModel):
    """Model for conversation messages."""
    
    id: str = Field(..., description="Message ID")
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Message metadata")
    
    @field_validator('role')
    @classmethod
    def validate_role(cls, v):
        """Validate message role."""
        allowed_roles = {'user', 'assistant', 'system'}
        if v not in allowed_roles:
            raise ValueError(f'role must be one of {allowed_roles}')
        return v


class ConversationHistory(BaseModel):
    """Model for conversation history management."""
    
    conversation_id: str = Field(..., description="Conversation ID")
    messages: List[ConversationMessage] = Field(default_factory=list, description="Conversation messages")
    context: Dict[str, Any] = Field(default_factory=dict, description="Conversation context")
    created_at: datetime = Field(default_factory=datetime.now, description="Conversation creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update time")
    project_id: Optional[str] = Field(None, description="Associated project ID")
    user_id: Optional[str] = Field(None, description="User ID")
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> ConversationMessage:
        """Add a message to the conversation."""
        import uuid
        message = ConversationMessage(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message
    
    def get_recent_messages(self, count: int = 10) -> List[ConversationMessage]:
        """Get recent messages from the conversation."""
        return self.messages[-count:] if count > 0 else self.messages
    
    def get_context_window(self, max_tokens: int = 4000) -> List[ConversationMessage]:
        """Get messages that fit within token limit (simplified)."""
        # Simplified token counting - in production use proper tokenizer
        total_tokens = 0
        context_messages = []
        
        for message in reversed(self.messages):
            message_tokens = len(message.content.split()) * 1.3  # Rough estimate
            if total_tokens + message_tokens > max_tokens:
                break
            context_messages.insert(0, message)
            total_tokens += message_tokens
        
        return context_messages


class MultiTurnAgentRequest(BaseModel):
    """Request model for multi-turn Agent conversations."""
    
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID")
    message: str = Field(..., description="User message")
    task_type: str = Field(..., description="Type of agent task")
    context_window: int = Field(10, ge=1, le=50, description="Number of previous messages to include")
    max_context_tokens: int = Field(4000, ge=500, le=16000, description="Maximum context tokens")
    preserve_context: bool = Field(True, description="Whether to preserve conversation context")
    project_id: Optional[str] = Field(None, description="Project context")
    user_id: Optional[str] = Field(None, description="User ID")
    
    @field_validator('task_type')
    @classmethod
    def validate_task_type(cls, v):
        """Validate task type is supported."""
        allowed_types = {
            'classification', 'extraction', 'summarization', 
            'question_answering', 'text_generation', 'analysis',
            'conversation', 'chat', 'follow_up', 'clarification', 'context_aware'
        }
        if v not in allowed_types:
            raise ValueError(f'task_type must be one of {allowed_types}')
        return v


class MultiTurnAgentResponse(BaseModel):
    """Response model for multi-turn Agent conversations."""
    
    conversation_id: str = Field(..., description="Conversation ID")
    message_id: str = Field(..., description="Response message ID")
    response: str = Field(..., description="Agent response")
    task_type: str = Field(..., description="Task type executed")
    execution_time: float = Field(..., description="Execution time")
    confidence: float = Field(..., description="Response confidence")
    context_used: int = Field(..., description="Number of context messages used")
    total_messages: int = Field(..., description="Total messages in conversation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")