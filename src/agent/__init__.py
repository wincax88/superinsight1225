"""
Agent module for SuperInsight Platform.

Provides Agent testing interfaces for AI applications.
"""

from .service import AgentService
from .models import (
    AgentRequest, AgentResponse, AgentStep, AgentMetrics,
    ConversationHistory, ConversationMessage,
    MultiTurnAgentRequest, MultiTurnAgentResponse
)

__all__ = [
    "AgentService",
    "AgentRequest", 
    "AgentResponse",
    "AgentStep",
    "AgentMetrics",
    "ConversationHistory",
    "ConversationMessage",
    "MultiTurnAgentRequest",
    "MultiTurnAgentResponse"
]