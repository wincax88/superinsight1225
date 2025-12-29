"""
Agent module for SuperInsight Platform.

Provides Agent testing interfaces for AI applications with advanced reasoning,
knowledge graph integration, tool calling, decision support, and risk assessment.
"""

from .service import AgentService
from .models import (
    AgentRequest, AgentResponse, AgentStep, AgentMetrics,
    ConversationHistory, ConversationMessage,
    MultiTurnAgentRequest, MultiTurnAgentResponse
)

# Phase 5: Advanced Reasoning and Tool Integration
from .reasoning_chain import (
    ReasoningChain, ReasoningStep, ReasoningChainBuilder, ReasoningEngine,
    ReasoningStepType, ReasoningStatus, Hypothesis,
    get_reasoning_engine, create_analysis_reasoning_chain
)

from .graph_reasoning import (
    GraphReasoningEngine, GraphQuestion, GraphAnswer,
    InferenceResult, GraphReasoningType,
    get_graph_reasoning_engine, initialize_graph_reasoning
)

from .tool_framework import (
    ToolFramework, BaseTool, FunctionTool, ToolRegistry,
    ToolSelector, ToolExecutor, ToolDefinition, ToolParameter,
    ToolExecutionRequest, ToolExecutionResult, ToolChain,
    ToolCategory, ExecutionStatus,
    get_tool_framework, create_function_tool
)

from .decision_tree import (
    DecisionTree, DecisionNode, DecisionOption, DecisionPath,
    DecisionTreeBuilder, DecisionAnalyzer, DecisionResult,
    MultiObjectiveOptimizer, OutcomePredictor,
    DecisionNodeType, DecisionStatus, OptimizationObjective,
    get_decision_analyzer, get_outcome_predictor, create_simple_decision_tree
)

from .risk_assessment import (
    RiskAssessmentEngine, Risk, RiskFactor, RiskIndicator,
    RiskAssessment, RiskAlert, MitigationStrategy,
    RiskIdentifier, RiskCalculator, MitigationAdvisor, RiskMonitor,
    RiskCategory, RiskSeverity, RiskStatus, AlertLevel,
    get_risk_engine, quick_risk_assessment
)

# Phase 6: Performance Optimization and Production Deployment
from .performance import (
    # Caching
    InMemoryCache, ResponseCache, CacheStrategy, CacheEntry, CacheMetrics,
    cached_response,
    # Concurrent Processing
    ConcurrentExecutor, TaskResult, ConcurrencyMode,
    # Performance Monitoring
    PerformanceMonitor, PerformanceMetric, LatencyStats, MetricType,
    measure_latency,
    # Health Check
    HealthChecker, HealthStatus, ComponentHealth, SystemHealth,
    # Service Discovery
    ServiceRegistry, ServiceInstance, LoadBalancer, LoadBalanceStrategy,
    # Global accessors
    get_response_cache, get_performance_monitor, get_concurrent_executor,
    get_health_checker, get_service_registry,
    create_default_health_checks,
)

__all__ = [
    # Core Agent
    "AgentService",
    "AgentRequest",
    "AgentResponse",
    "AgentStep",
    "AgentMetrics",
    "ConversationHistory",
    "ConversationMessage",
    "MultiTurnAgentRequest",
    "MultiTurnAgentResponse",

    # Reasoning Chain
    "ReasoningChain",
    "ReasoningStep",
    "ReasoningChainBuilder",
    "ReasoningEngine",
    "ReasoningStepType",
    "ReasoningStatus",
    "Hypothesis",
    "get_reasoning_engine",
    "create_analysis_reasoning_chain",

    # Graph Reasoning
    "GraphReasoningEngine",
    "GraphQuestion",
    "GraphAnswer",
    "InferenceResult",
    "GraphReasoningType",
    "get_graph_reasoning_engine",
    "initialize_graph_reasoning",

    # Tool Framework
    "ToolFramework",
    "BaseTool",
    "FunctionTool",
    "ToolRegistry",
    "ToolSelector",
    "ToolExecutor",
    "ToolDefinition",
    "ToolParameter",
    "ToolExecutionRequest",
    "ToolExecutionResult",
    "ToolChain",
    "ToolCategory",
    "ExecutionStatus",
    "get_tool_framework",
    "create_function_tool",

    # Decision Tree
    "DecisionTree",
    "DecisionNode",
    "DecisionOption",
    "DecisionPath",
    "DecisionTreeBuilder",
    "DecisionAnalyzer",
    "DecisionResult",
    "MultiObjectiveOptimizer",
    "OutcomePredictor",
    "DecisionNodeType",
    "DecisionStatus",
    "OptimizationObjective",
    "get_decision_analyzer",
    "get_outcome_predictor",
    "create_simple_decision_tree",

    # Risk Assessment
    "RiskAssessmentEngine",
    "Risk",
    "RiskFactor",
    "RiskIndicator",
    "RiskAssessment",
    "RiskAlert",
    "MitigationStrategy",
    "RiskIdentifier",
    "RiskCalculator",
    "MitigationAdvisor",
    "RiskMonitor",
    "RiskCategory",
    "RiskSeverity",
    "RiskStatus",
    "AlertLevel",
    "get_risk_engine",
    "quick_risk_assessment",

    # Performance Optimization
    "InMemoryCache",
    "ResponseCache",
    "CacheStrategy",
    "CacheEntry",
    "CacheMetrics",
    "cached_response",
    "ConcurrentExecutor",
    "TaskResult",
    "ConcurrencyMode",
    "PerformanceMonitor",
    "PerformanceMetric",
    "LatencyStats",
    "MetricType",
    "measure_latency",
    "HealthChecker",
    "HealthStatus",
    "ComponentHealth",
    "SystemHealth",
    "ServiceRegistry",
    "ServiceInstance",
    "LoadBalancer",
    "LoadBalanceStrategy",
    "get_response_cache",
    "get_performance_monitor",
    "get_concurrent_executor",
    "get_health_checker",
    "get_service_registry",
    "create_default_health_checks",
]