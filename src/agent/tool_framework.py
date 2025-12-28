"""
Tool Calling Framework for AI Agent System.

Provides external tool call interface, tool selection and composition logic,
tool execution result validation, and tool call chain management.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Dict, Any, List, Optional, Callable, Awaitable,
    Union, TypeVar, Generic
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ToolCategory(str, Enum):
    """Categories of tools."""
    DATA_RETRIEVAL = "data_retrieval"     # Fetch data from sources
    DATA_PROCESSING = "data_processing"   # Process and transform data
    ANALYSIS = "analysis"                 # Perform analysis
    COMMUNICATION = "communication"       # Send notifications, emails
    EXTERNAL_API = "external_api"         # Call external APIs
    DATABASE = "database"                 # Database operations
    FILE_SYSTEM = "file_system"           # File operations
    COMPUTATION = "computation"           # Mathematical computations
    VISUALIZATION = "visualization"       # Generate charts, graphs


class ToolStatus(str, Enum):
    """Status of a tool."""
    AVAILABLE = "available"
    BUSY = "busy"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    DEPRECATED = "deprecated"


class ExecutionStatus(str, Enum):
    """Status of tool execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    param_type: str  # str, int, float, bool, list, dict
    description: str = ""
    required: bool = True
    default: Any = None
    validators: List[Callable[[Any], bool]] = field(default_factory=list)
    examples: List[Any] = field(default_factory=list)


@dataclass
class ToolDefinition:
    """Definition of a tool."""
    name: str
    description: str
    category: ToolCategory
    parameters: List[ToolParameter] = field(default_factory=list)
    return_type: str = "dict"
    examples: List[Dict[str, Any]] = field(default_factory=list)
    version: str = "1.0.0"
    author: str = ""
    tags: List[str] = field(default_factory=list)
    timeout: float = 30.0
    max_retries: int = 3
    requires_auth: bool = False
    rate_limit: Optional[int] = None  # Calls per minute


@dataclass
class ToolExecutionRequest:
    """Request to execute a tool."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher = more important
    timeout: Optional[float] = None
    callback: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ToolExecutionResult:
    """Result of tool execution."""
    request_id: str
    tool_name: str
    status: ExecutionStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    retries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ExecutionStatus.COMPLETED and self.error is None


@dataclass
class ToolChainStep:
    """A step in a tool chain."""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_mappings: Dict[str, str] = field(default_factory=dict)  # Map from previous step outputs
    output_key: str = ""  # Key to store output for next steps
    condition: Optional[str] = None  # Condition to execute this step
    on_failure: str = "abort"  # abort, skip, retry


@dataclass
class ToolChain:
    """A chain of tool calls."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    steps: List[ToolChainStep] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, ToolExecutionResult] = field(default_factory=dict)
    current_step: int = 0
    status: ExecutionStatus = ExecutionStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)


class BaseTool(ABC):
    """Abstract base class for tools."""

    def __init__(self, definition: ToolDefinition):
        """Initialize the tool with its definition."""
        self.definition = definition
        self.status = ToolStatus.AVAILABLE
        self.call_count = 0
        self.error_count = 0
        self.total_execution_time = 0.0

    @abstractmethod
    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute the tool with given parameters."""
        pass

    def validate_parameters(self, parameters: Dict[str, Any]) -> List[str]:
        """Validate parameters against definition."""
        errors = []

        for param in self.definition.parameters:
            if param.required and param.name not in parameters:
                errors.append(f"Missing required parameter: {param.name}")
                continue

            if param.name in parameters:
                value = parameters[param.name]

                # Type validation
                type_mapping = {
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "list": list,
                    "dict": dict
                }

                expected_type = type_mapping.get(param.param_type)
                if expected_type and not isinstance(value, expected_type):
                    errors.append(
                        f"Parameter '{param.name}' expected {param.param_type}, got {type(value).__name__}"
                    )

                # Custom validators
                for validator in param.validators:
                    try:
                        if not validator(value):
                            errors.append(f"Validation failed for parameter: {param.name}")
                    except Exception as e:
                        errors.append(f"Validator error for {param.name}: {e}")

        return errors

    def get_metrics(self) -> Dict[str, Any]:
        """Get tool metrics."""
        return {
            "name": self.definition.name,
            "call_count": self.call_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / self.call_count if self.call_count > 0 else 0,
            "avg_execution_time": self.total_execution_time / self.call_count if self.call_count > 0 else 0,
            "status": self.status.value
        }


class FunctionTool(BaseTool):
    """Tool that wraps a Python function."""

    def __init__(
        self,
        definition: ToolDefinition,
        func: Union[Callable, Callable[..., Awaitable]]
    ):
        """Initialize with function."""
        super().__init__(definition)
        self.func = func
        self.is_async = asyncio.iscoroutinefunction(func)

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute the wrapped function."""
        if self.is_async:
            return await self.func(**parameters)
        else:
            return self.func(**parameters)


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        """Initialize the tool registry."""
        self.tools: Dict[str, BaseTool] = {}
        self.tool_aliases: Dict[str, str] = {}

    def register(self, tool: BaseTool, aliases: Optional[List[str]] = None) -> None:
        """Register a tool."""
        self.tools[tool.definition.name] = tool

        if aliases:
            for alias in aliases:
                self.tool_aliases[alias] = tool.definition.name

        logger.info(f"Registered tool: {tool.definition.name}")

    def unregister(self, tool_name: str) -> bool:
        """Unregister a tool."""
        if tool_name in self.tools:
            del self.tools[tool_name]
            # Remove aliases
            aliases_to_remove = [
                alias for alias, name in self.tool_aliases.items()
                if name == tool_name
            ]
            for alias in aliases_to_remove:
                del self.tool_aliases[alias]
            return True
        return False

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name or alias."""
        # Check direct name
        if name in self.tools:
            return self.tools[name]

        # Check aliases
        if name in self.tool_aliases:
            return self.tools.get(self.tool_aliases[name])

        return None

    def list_tools(
        self,
        category: Optional[ToolCategory] = None,
        status: Optional[ToolStatus] = None
    ) -> List[ToolDefinition]:
        """List available tools."""
        tools = list(self.tools.values())

        if category:
            tools = [t for t in tools if t.definition.category == category]

        if status:
            tools = [t for t in tools if t.status == status]

        return [t.definition for t in tools]

    def search_tools(self, query: str) -> List[ToolDefinition]:
        """Search tools by name, description, or tags."""
        query_lower = query.lower()
        results = []

        for tool in self.tools.values():
            # Check name
            if query_lower in tool.definition.name.lower():
                results.append(tool.definition)
                continue

            # Check description
            if query_lower in tool.definition.description.lower():
                results.append(tool.definition)
                continue

            # Check tags
            if any(query_lower in tag.lower() for tag in tool.definition.tags):
                results.append(tool.definition)

        return results


class ToolSelector:
    """Selector for choosing appropriate tools based on task."""

    def __init__(self, registry: ToolRegistry):
        """Initialize with tool registry."""
        self.registry = registry
        self.selection_history: List[Dict[str, Any]] = []

    def select_tool(
        self,
        task_description: str,
        required_capabilities: Optional[List[str]] = None,
        preferred_categories: Optional[List[ToolCategory]] = None
    ) -> Optional[BaseTool]:
        """Select the best tool for a task."""
        candidates = []

        for tool in self.registry.tools.values():
            if tool.status != ToolStatus.AVAILABLE:
                continue

            score = self._calculate_tool_score(
                tool,
                task_description,
                required_capabilities,
                preferred_categories
            )

            if score > 0:
                candidates.append((tool, score))

        if not candidates:
            return None

        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)

        selected = candidates[0][0]

        # Record selection
        self.selection_history.append({
            "task": task_description,
            "selected": selected.definition.name,
            "score": candidates[0][1],
            "alternatives": [c[0].definition.name for c in candidates[1:4]],
            "timestamp": datetime.now().isoformat()
        })

        return selected

    def _calculate_tool_score(
        self,
        tool: BaseTool,
        task_description: str,
        required_capabilities: Optional[List[str]],
        preferred_categories: Optional[List[ToolCategory]]
    ) -> float:
        """Calculate relevance score for a tool."""
        score = 0.0
        task_lower = task_description.lower()

        # Name match
        if tool.definition.name.lower() in task_lower:
            score += 0.5

        # Description match
        desc_words = tool.definition.description.lower().split()
        task_words = task_lower.split()
        overlap = len(set(desc_words) & set(task_words))
        score += min(0.3, overlap * 0.05)

        # Tag match
        for tag in tool.definition.tags:
            if tag.lower() in task_lower:
                score += 0.2

        # Category preference
        if preferred_categories and tool.definition.category in preferred_categories:
            score += 0.2

        # Required capabilities (check parameter names)
        if required_capabilities:
            param_names = [p.name for p in tool.definition.parameters]
            capability_match = len(set(required_capabilities) & set(param_names))
            score += capability_match * 0.1

        # Tool reliability bonus
        metrics = tool.get_metrics()
        if metrics["call_count"] > 10 and metrics["error_rate"] < 0.1:
            score += 0.1

        return score

    def select_tool_chain(
        self,
        task_description: str,
        max_tools: int = 5
    ) -> List[BaseTool]:
        """Select a chain of tools for a complex task."""
        selected_tools = []
        remaining_task = task_description

        for _ in range(max_tools):
            tool = self.select_tool(remaining_task)
            if not tool or tool in selected_tools:
                break

            selected_tools.append(tool)

            # Update remaining task (simplified - in production, use NLP)
            tool_name = tool.definition.name
            remaining_task = remaining_task.replace(tool_name, "")

        return selected_tools


class ToolExecutor:
    """Executor for running tools with proper error handling."""

    def __init__(self, registry: ToolRegistry):
        """Initialize with tool registry."""
        self.registry = registry
        self.execution_history: List[ToolExecutionResult] = []
        self.max_history = 1000

    async def execute(
        self,
        request: ToolExecutionRequest
    ) -> ToolExecutionResult:
        """Execute a tool request."""
        result = ToolExecutionResult(
            request_id=request.id,
            tool_name=request.tool_name,
            status=ExecutionStatus.PENDING
        )

        tool = self.registry.get_tool(request.tool_name)
        if not tool:
            result.status = ExecutionStatus.FAILED
            result.error = f"Tool not found: {request.tool_name}"
            self._record_result(result)
            return result

        # Validate parameters
        validation_errors = tool.validate_parameters(request.parameters)
        if validation_errors:
            result.status = ExecutionStatus.FAILED
            result.error = f"Validation errors: {'; '.join(validation_errors)}"
            self._record_result(result)
            return result

        # Execute with timeout and retry
        timeout = request.timeout or tool.definition.timeout
        max_retries = tool.definition.max_retries

        result.started_at = datetime.now()
        result.status = ExecutionStatus.RUNNING

        for retry in range(max_retries):
            try:
                start_time = time.time()

                # Execute with timeout
                execution_result = await asyncio.wait_for(
                    tool.execute(request.parameters, request.context),
                    timeout=timeout
                )

                result.execution_time = time.time() - start_time
                result.result = execution_result
                result.status = ExecutionStatus.COMPLETED
                result.retries = retry

                # Update tool metrics
                tool.call_count += 1
                tool.total_execution_time += result.execution_time

                break

            except asyncio.TimeoutError:
                result.error = f"Execution timeout after {timeout}s"
                result.status = ExecutionStatus.TIMEOUT
                result.retries = retry

            except Exception as e:
                result.error = str(e)
                result.status = ExecutionStatus.FAILED
                result.retries = retry

                logger.warning(f"Tool execution failed (retry {retry + 1}): {e}")

                if retry < max_retries - 1:
                    await asyncio.sleep(0.5 * (retry + 1))  # Exponential backoff
                else:
                    tool.error_count += 1

        result.completed_at = datetime.now()
        self._record_result(result)

        return result

    async def execute_chain(
        self,
        chain: ToolChain
    ) -> ToolChain:
        """Execute a tool chain."""
        chain.status = ExecutionStatus.RUNNING
        accumulated_context = dict(chain.context)

        for i, step in enumerate(chain.steps):
            chain.current_step = i

            # Check condition
            if step.condition:
                # Simplified condition evaluation
                try:
                    if not eval(step.condition, {"context": accumulated_context}):
                        logger.info(f"Skipping step {step.step_id}: condition not met")
                        continue
                except Exception as e:
                    logger.warning(f"Condition evaluation failed: {e}")

            # Prepare parameters with input mappings
            parameters = dict(step.parameters)
            for param_name, source_key in step.input_mappings.items():
                if source_key in accumulated_context:
                    parameters[param_name] = accumulated_context[source_key]

            # Create and execute request
            request = ToolExecutionRequest(
                tool_name=step.tool_name,
                parameters=parameters,
                context=accumulated_context
            )

            result = await self.execute(request)
            chain.results[step.step_id] = result

            # Handle failure
            if not result.is_success():
                if step.on_failure == "abort":
                    chain.status = ExecutionStatus.FAILED
                    return chain
                elif step.on_failure == "skip":
                    continue
                elif step.on_failure == "retry":
                    # Already handled in execute
                    pass

            # Store output for next steps
            if step.output_key and result.result is not None:
                accumulated_context[step.output_key] = result.result

        chain.status = ExecutionStatus.COMPLETED
        chain.context = accumulated_context
        return chain

    def _record_result(self, result: ToolExecutionResult) -> None:
        """Record execution result in history."""
        self.execution_history.append(result)

        # Trim history if too large
        if len(self.execution_history) > self.max_history:
            self.execution_history = self.execution_history[-self.max_history:]

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self.execution_history:
            return {"total_executions": 0}

        total = len(self.execution_history)
        successful = sum(1 for r in self.execution_history if r.is_success())
        failed = total - successful
        avg_time = sum(r.execution_time for r in self.execution_history) / total

        return {
            "total_executions": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0,
            "average_execution_time": avg_time,
            "tools_used": list(set(r.tool_name for r in self.execution_history))
        }


class ResultValidator:
    """Validator for tool execution results."""

    def __init__(self):
        """Initialize the validator."""
        self.validators: Dict[str, List[Callable[[Any], bool]]] = {}
        self.validation_history: List[Dict[str, Any]] = []

    def register_validator(
        self,
        tool_name: str,
        validator: Callable[[Any], bool],
        description: str = ""
    ) -> None:
        """Register a result validator for a tool."""
        if tool_name not in self.validators:
            self.validators[tool_name] = []
        self.validators[tool_name].append(validator)

    def validate(
        self,
        result: ToolExecutionResult
    ) -> Tuple[bool, List[str]]:
        """Validate a tool execution result."""
        if result.status != ExecutionStatus.COMPLETED:
            return False, [f"Execution not completed: {result.status.value}"]

        if result.result is None:
            return False, ["Result is None"]

        errors = []
        validators = self.validators.get(result.tool_name, [])

        for validator in validators:
            try:
                if not validator(result.result):
                    errors.append("Validation check failed")
            except Exception as e:
                errors.append(f"Validator error: {e}")

        is_valid = len(errors) == 0

        # Record validation
        self.validation_history.append({
            "request_id": result.request_id,
            "tool_name": result.tool_name,
            "is_valid": is_valid,
            "errors": errors,
            "timestamp": datetime.now().isoformat()
        })

        return is_valid, errors


class ToolFramework:
    """Main framework for tool management and execution."""

    def __init__(self):
        """Initialize the tool framework."""
        self.registry = ToolRegistry()
        self.selector = ToolSelector(self.registry)
        self.executor = ToolExecutor(self.registry)
        self.validator = ResultValidator()

        # Register default tools
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register default utility tools."""
        # Data processing tool
        data_tool = FunctionTool(
            definition=ToolDefinition(
                name="data_transform",
                description="Transform and process data",
                category=ToolCategory.DATA_PROCESSING,
                parameters=[
                    ToolParameter(name="data", param_type="dict", description="Data to transform"),
                    ToolParameter(name="operation", param_type="str", description="Transform operation")
                ],
                tags=["data", "transform", "process"]
            ),
            func=self._data_transform
        )
        self.registry.register(data_tool)

        # Computation tool
        calc_tool = FunctionTool(
            definition=ToolDefinition(
                name="calculate",
                description="Perform mathematical calculations",
                category=ToolCategory.COMPUTATION,
                parameters=[
                    ToolParameter(name="expression", param_type="str", description="Math expression"),
                    ToolParameter(name="variables", param_type="dict", required=False, description="Variables")
                ],
                tags=["math", "calculate", "compute"]
            ),
            func=self._calculate
        )
        self.registry.register(calc_tool, aliases=["calc", "compute"])

        # Text analysis tool
        text_tool = FunctionTool(
            definition=ToolDefinition(
                name="text_analyze",
                description="Analyze text for patterns and insights",
                category=ToolCategory.ANALYSIS,
                parameters=[
                    ToolParameter(name="text", param_type="str", description="Text to analyze"),
                    ToolParameter(name="analysis_type", param_type="str", description="Type of analysis")
                ],
                tags=["text", "nlp", "analysis"]
            ),
            func=self._text_analyze
        )
        self.registry.register(text_tool)

    async def _data_transform(
        self,
        data: Dict[str, Any],
        operation: str
    ) -> Dict[str, Any]:
        """Default data transform implementation."""
        result = {"original": data, "operation": operation}

        if operation == "flatten":
            # Flatten nested dict
            def flatten(d, parent_key=''):
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}.{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten(v, new_key).items())
                    else:
                        items.append((new_key, v))
                return dict(items)
            result["transformed"] = flatten(data)

        elif operation == "keys":
            result["transformed"] = list(data.keys())

        elif operation == "values":
            result["transformed"] = list(data.values())

        else:
            result["transformed"] = data

        return result

    async def _calculate(
        self,
        expression: str,
        variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Default calculation implementation."""
        try:
            # Safe evaluation (simplified - in production use proper math parser)
            allowed_names = {"abs": abs, "max": max, "min": min, "sum": sum, "len": len}
            if variables:
                allowed_names.update(variables)

            # Only allow safe operations
            result = eval(expression, {"__builtins__": {}}, allowed_names)

            return {
                "expression": expression,
                "result": result,
                "variables": variables or {}
            }
        except Exception as e:
            return {
                "expression": expression,
                "error": str(e),
                "variables": variables or {}
            }

    async def _text_analyze(
        self,
        text: str,
        analysis_type: str
    ) -> Dict[str, Any]:
        """Default text analysis implementation."""
        result = {
            "text_length": len(text),
            "word_count": len(text.split()),
            "analysis_type": analysis_type
        }

        if analysis_type == "basic":
            result["sentences"] = text.count('.') + text.count('!') + text.count('?')
            result["paragraphs"] = text.count('\n\n') + 1

        elif analysis_type == "keywords":
            words = text.lower().split()
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            result["top_keywords"] = sorted_words[:10]

        return result

    def register_tool(
        self,
        tool: BaseTool,
        aliases: Optional[List[str]] = None
    ) -> None:
        """Register a tool with the framework."""
        self.registry.register(tool, aliases)

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolExecutionResult:
        """Execute a tool by name."""
        request = ToolExecutionRequest(
            tool_name=tool_name,
            parameters=parameters,
            context=context or {}
        )
        return await self.executor.execute(request)

    async def execute_for_task(
        self,
        task_description: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolExecutionResult:
        """Select and execute a tool for a task."""
        tool = self.selector.select_tool(task_description)

        if not tool:
            return ToolExecutionResult(
                request_id=str(uuid.uuid4()),
                tool_name="unknown",
                status=ExecutionStatus.FAILED,
                error="No suitable tool found for task"
            )

        return await self.execute_tool(
            tool.definition.name,
            parameters,
            context
        )

    async def execute_chain(
        self,
        chain: ToolChain
    ) -> ToolChain:
        """Execute a tool chain."""
        return await self.executor.execute_chain(chain)

    def get_framework_stats(self) -> Dict[str, Any]:
        """Get framework statistics."""
        return {
            "registered_tools": len(self.registry.tools),
            "tool_categories": list(set(
                t.definition.category.value for t in self.registry.tools.values()
            )),
            "execution_stats": self.executor.get_execution_stats(),
            "selection_history_count": len(self.selector.selection_history)
        }


# Global instance
_tool_framework: Optional[ToolFramework] = None


def get_tool_framework() -> ToolFramework:
    """Get or create global tool framework instance."""
    global _tool_framework
    if _tool_framework is None:
        _tool_framework = ToolFramework()
    return _tool_framework


def create_function_tool(
    name: str,
    description: str,
    category: ToolCategory,
    func: Union[Callable, Callable[..., Awaitable]],
    parameters: Optional[List[ToolParameter]] = None,
    tags: Optional[List[str]] = None
) -> FunctionTool:
    """Helper to create a function-based tool."""
    definition = ToolDefinition(
        name=name,
        description=description,
        category=category,
        parameters=parameters or [],
        tags=tags or []
    )
    return FunctionTool(definition, func)
