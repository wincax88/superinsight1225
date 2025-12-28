"""
Sync Orchestrator Module.

Provides comprehensive orchestration for multi-source data synchronization,
including dependency management, parallel execution, and workflow coordination.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"  # Some steps succeeded, some failed


class StepStatus(str, Enum):
    """Individual step execution status."""
    PENDING = "pending"
    WAITING = "waiting"    # Waiting for dependencies
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class StepType(str, Enum):
    """Types of workflow steps."""
    EXTRACT = "extract"     # Extract data from source
    TRANSFORM = "transform" # Transform data
    LOAD = "load"           # Load data to destination
    VALIDATE = "validate"   # Validate data quality
    NOTIFY = "notify"       # Send notifications
    CUSTOM = "custom"       # Custom action


class RetryStrategy(str, Enum):
    """Retry strategies for failed steps."""
    NONE = "none"
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"


@dataclass
class RetryConfig:
    """Retry configuration for steps."""
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0


@dataclass
class StepResult:
    """Result of a workflow step execution."""
    step_id: str
    status: StepStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    records_processed: int = 0
    error: Optional[str] = None
    output: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""
    workflow_id: str
    status: WorkflowStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    steps_completed: int = 0
    steps_failed: int = 0
    steps_skipped: int = 0
    total_records: int = 0
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)


class WorkflowStep(BaseModel):
    """Definition of a workflow step."""
    id: str
    name: str
    type: StepType
    description: Optional[str] = None

    # Execution settings
    connector_id: Optional[str] = None
    action: str = ""
    parameters: Dict[str, Any] = Field(default_factory=dict)

    # Dependencies
    depends_on: List[str] = Field(default_factory=list)

    # Error handling
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
    continue_on_error: bool = False
    timeout_seconds: int = 3600

    # Conditional execution
    condition: Optional[str] = None  # Python expression to evaluate

    # Resource requirements
    priority: int = 0
    parallel_group: Optional[str] = None  # Steps in same group run in parallel


class WorkflowDefinition(BaseModel):
    """Definition of a sync workflow."""
    id: str
    name: str
    description: Optional[str] = None
    version: str = "1.0"

    # Steps
    steps: List[WorkflowStep] = Field(default_factory=list)

    # Global settings
    max_parallel_steps: int = 5
    fail_fast: bool = True  # Stop on first failure
    timeout_seconds: int = 7200

    # Retry settings
    default_retry_config: RetryConfig = Field(default_factory=RetryConfig)

    # Notifications
    notify_on_complete: bool = False
    notify_on_failure: bool = True


class SyncOrchestrator:
    """
    Sync Orchestrator for coordinating multi-source synchronization.

    Features:
    - DAG-based workflow execution
    - Parallel step execution with dependency management
    - Retry logic with configurable strategies
    - Pause/resume/cancel support
    - Progress tracking and monitoring
    - Event hooks for customization
    """

    def __init__(self, max_workers: int = 10):
        """
        Initialize orchestrator.

        Args:
            max_workers: Maximum concurrent step executions
        """
        self.max_workers = max_workers
        self._workflows: Dict[str, WorkflowDefinition] = {}
        self._running_workflows: Dict[str, WorkflowResult] = {}
        self._step_handlers: Dict[StepType, Callable] = {}
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._semaphore = asyncio.Semaphore(max_workers)
        self._cancel_events: Dict[str, asyncio.Event] = {}
        self._pause_events: Dict[str, asyncio.Event] = {}

    def register_workflow(self, workflow: WorkflowDefinition) -> None:
        """Register a workflow definition."""
        self._validate_workflow(workflow)
        self._workflows[workflow.id] = workflow
        logger.info(f"Registered workflow: {workflow.id}")

    def register_step_handler(
        self,
        step_type: StepType,
        handler: Callable
    ) -> None:
        """
        Register a handler for a step type.

        Args:
            step_type: Type of step
            handler: Async function to handle the step
        """
        self._step_handlers[step_type] = handler
        logger.info(f"Registered handler for step type: {step_type.value}")

    def on_event(self, event: str, handler: Callable) -> None:
        """
        Register an event handler.

        Events:
        - workflow_started
        - workflow_completed
        - workflow_failed
        - step_started
        - step_completed
        - step_failed
        - step_retrying
        """
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    async def execute_workflow(
        self,
        workflow_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> WorkflowResult:
        """
        Execute a workflow.

        Args:
            workflow_id: ID of registered workflow
            context: Initial context data

        Returns:
            WorkflowResult with execution details
        """
        if workflow_id not in self._workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")

        workflow = self._workflows[workflow_id]
        execution_id = f"{workflow_id}_{uuid.uuid4().hex[:8]}"

        # Initialize result
        result = WorkflowResult(
            workflow_id=execution_id,
            status=WorkflowStatus.RUNNING,
            started_at=datetime.utcnow()
        )
        self._running_workflows[execution_id] = result

        # Initialize control events
        self._cancel_events[execution_id] = asyncio.Event()
        self._pause_events[execution_id] = asyncio.Event()
        self._pause_events[execution_id].set()  # Not paused initially

        context = context or {}
        context["_workflow_id"] = execution_id
        context["_started_at"] = result.started_at.isoformat()

        await self._emit_event("workflow_started", {
            "workflow_id": execution_id,
            "definition_id": workflow_id
        })

        try:
            # Build execution plan
            execution_plan = self._build_execution_plan(workflow)

            # Execute steps according to plan
            await self._execute_plan(
                execution_id,
                workflow,
                execution_plan,
                context,
                result
            )

            # Determine final status
            if result.steps_failed > 0:
                if result.steps_completed > 0:
                    result.status = WorkflowStatus.PARTIAL
                else:
                    result.status = WorkflowStatus.FAILED
            else:
                result.status = WorkflowStatus.COMPLETED

            await self._emit_event("workflow_completed", {
                "workflow_id": execution_id,
                "status": result.status.value
            })

        except asyncio.CancelledError:
            result.status = WorkflowStatus.CANCELLED
            await self._emit_event("workflow_cancelled", {
                "workflow_id": execution_id
            })

        except Exception as e:
            result.status = WorkflowStatus.FAILED
            result.errors.append({
                "error": str(e),
                "type": type(e).__name__
            })
            await self._emit_event("workflow_failed", {
                "workflow_id": execution_id,
                "error": str(e)
            })
            logger.exception(f"Workflow failed: {execution_id}")

        finally:
            result.completed_at = datetime.utcnow()
            result.duration_seconds = (
                result.completed_at - result.started_at
            ).total_seconds()

            # Cleanup
            self._cancel_events.pop(execution_id, None)
            self._pause_events.pop(execution_id, None)

        return result

    async def pause_workflow(self, execution_id: str) -> bool:
        """Pause a running workflow."""
        if execution_id in self._pause_events:
            self._pause_events[execution_id].clear()
            if execution_id in self._running_workflows:
                self._running_workflows[execution_id].status = WorkflowStatus.PAUSED
            logger.info(f"Paused workflow: {execution_id}")
            return True
        return False

    async def resume_workflow(self, execution_id: str) -> bool:
        """Resume a paused workflow."""
        if execution_id in self._pause_events:
            self._pause_events[execution_id].set()
            if execution_id in self._running_workflows:
                self._running_workflows[execution_id].status = WorkflowStatus.RUNNING
            logger.info(f"Resumed workflow: {execution_id}")
            return True
        return False

    async def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel a running workflow."""
        if execution_id in self._cancel_events:
            self._cancel_events[execution_id].set()
            logger.info(f"Cancelling workflow: {execution_id}")
            return True
        return False

    def get_workflow_status(self, execution_id: str) -> Optional[WorkflowResult]:
        """Get current status of a workflow."""
        return self._running_workflows.get(execution_id)

    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all registered workflows."""
        return [
            {
                "id": w.id,
                "name": w.name,
                "version": w.version,
                "steps": len(w.steps)
            }
            for w in self._workflows.values()
        ]

    def _validate_workflow(self, workflow: WorkflowDefinition) -> None:
        """Validate workflow definition."""
        step_ids = {step.id for step in workflow.steps}

        for step in workflow.steps:
            # Check dependencies exist
            for dep in step.depends_on:
                if dep not in step_ids:
                    raise ValueError(
                        f"Step '{step.id}' depends on unknown step '{dep}'"
                    )

            # Check for self-dependency
            if step.id in step.depends_on:
                raise ValueError(f"Step '{step.id}' cannot depend on itself")

        # Check for circular dependencies
        if self._has_circular_dependency(workflow):
            raise ValueError("Workflow contains circular dependencies")

    def _has_circular_dependency(self, workflow: WorkflowDefinition) -> bool:
        """Check for circular dependencies using DFS."""
        graph = {step.id: set(step.depends_on) for step in workflow.steps}

        def has_cycle(node: str, visited: Set[str], rec_stack: Set[str]) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.discard(node)
            return False

        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        for node in graph:
            if node not in visited:
                if has_cycle(node, visited, rec_stack):
                    return True

        return False

    def _build_execution_plan(
        self,
        workflow: WorkflowDefinition
    ) -> List[List[str]]:
        """
        Build execution plan using topological sort.

        Returns list of "levels" where each level can be executed in parallel.
        """
        graph = {step.id: set(step.depends_on) for step in workflow.steps}
        in_degree = {node: len(deps) for node, deps in graph.items()}
        levels = []

        remaining = set(graph.keys())

        while remaining:
            # Find all nodes with no dependencies
            ready = [n for n in remaining if in_degree[n] == 0]

            if not ready:
                raise ValueError("Circular dependency detected")

            levels.append(ready)

            # Remove processed nodes and update in-degrees
            for node in ready:
                remaining.remove(node)
                for other in remaining:
                    if node in graph.get(other, set()):
                        # This shouldn't affect in_degree calculation
                        pass
                    # Update in_degree for nodes that depend on this one
                    step = next(s for s in workflow.steps if s.id == other)
                    if node in step.depends_on:
                        in_degree[other] -= 1

        return levels

    async def _execute_plan(
        self,
        execution_id: str,
        workflow: WorkflowDefinition,
        plan: List[List[str]],
        context: Dict[str, Any],
        result: WorkflowResult
    ) -> None:
        """Execute the workflow plan."""
        step_map = {step.id: step for step in workflow.steps}

        for level_idx, level in enumerate(plan):
            # Check for cancellation
            if self._cancel_events[execution_id].is_set():
                # Mark remaining steps as cancelled
                for step_id in level:
                    result.step_results[step_id] = StepResult(
                        step_id=step_id,
                        status=StepStatus.CANCELLED
                    )
                break

            # Wait if paused
            await self._pause_events[execution_id].wait()

            logger.info(f"Executing level {level_idx + 1}/{len(plan)}: {level}")

            # Execute steps in this level in parallel
            # But respect max_parallel_steps
            tasks = []
            for step_id in level:
                step = step_map[step_id]

                # Check condition
                if step.condition and not self._evaluate_condition(
                    step.condition, context
                ):
                    result.step_results[step_id] = StepResult(
                        step_id=step_id,
                        status=StepStatus.SKIPPED
                    )
                    result.steps_skipped += 1
                    continue

                task = asyncio.create_task(
                    self._execute_step(
                        execution_id,
                        step,
                        context,
                        workflow.default_retry_config
                    )
                )
                tasks.append((step_id, task))

            # Wait for all tasks in this level
            for step_id, task in tasks:
                try:
                    step_result = await task
                    result.step_results[step_id] = step_result

                    if step_result.status == StepStatus.COMPLETED:
                        result.steps_completed += 1
                        result.total_records += step_result.records_processed
                        # Update context with step output
                        context[f"_step_{step_id}"] = step_result.output
                    else:
                        result.steps_failed += 1

                        if workflow.fail_fast and not step_map[step_id].continue_on_error:
                            # Cancel remaining steps
                            self._cancel_events[execution_id].set()
                            break

                except Exception as e:
                    result.step_results[step_id] = StepResult(
                        step_id=step_id,
                        status=StepStatus.FAILED,
                        error=str(e)
                    )
                    result.steps_failed += 1
                    result.errors.append({
                        "step_id": step_id,
                        "error": str(e)
                    })

    async def _execute_step(
        self,
        execution_id: str,
        step: WorkflowStep,
        context: Dict[str, Any],
        default_retry: RetryConfig
    ) -> StepResult:
        """Execute a single workflow step with retry logic."""
        result = StepResult(
            step_id=step.id,
            status=StepStatus.RUNNING,
            started_at=datetime.utcnow()
        )

        retry_config = step.retry_config or default_retry
        retry_count = 0
        last_error = None

        await self._emit_event("step_started", {
            "workflow_id": execution_id,
            "step_id": step.id
        })

        while retry_count <= retry_config.max_retries:
            try:
                async with self._semaphore:
                    # Get handler for step type
                    handler = self._step_handlers.get(step.type)

                    if handler:
                        # Execute with timeout
                        output = await asyncio.wait_for(
                            handler(step, context),
                            timeout=step.timeout_seconds
                        )
                    else:
                        # Default behavior based on step type
                        output = await self._default_step_handler(step, context)

                    result.status = StepStatus.COMPLETED
                    result.output = output or {}
                    result.records_processed = output.get("records_processed", 0) \
                        if isinstance(output, dict) else 0

                    await self._emit_event("step_completed", {
                        "workflow_id": execution_id,
                        "step_id": step.id,
                        "records": result.records_processed
                    })

                    break

            except asyncio.TimeoutError:
                last_error = f"Step timed out after {step.timeout_seconds}s"
                result.status = StepStatus.FAILED

            except Exception as e:
                last_error = str(e)
                retry_count += 1
                result.retry_count = retry_count

                if retry_count <= retry_config.max_retries:
                    delay = self._calculate_retry_delay(retry_config, retry_count)
                    logger.warning(
                        f"Step {step.id} failed, retrying in {delay}s "
                        f"({retry_count}/{retry_config.max_retries})"
                    )

                    await self._emit_event("step_retrying", {
                        "workflow_id": execution_id,
                        "step_id": step.id,
                        "retry": retry_count,
                        "delay": delay
                    })

                    await asyncio.sleep(delay)
                else:
                    result.status = StepStatus.FAILED
                    result.error = last_error

                    await self._emit_event("step_failed", {
                        "workflow_id": execution_id,
                        "step_id": step.id,
                        "error": last_error
                    })

        result.completed_at = datetime.utcnow()
        result.duration_seconds = (
            result.completed_at - result.started_at
        ).total_seconds()

        return result

    async def _default_step_handler(
        self,
        step: WorkflowStep,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Default handler for steps without registered handlers."""
        logger.info(f"Executing step: {step.name} ({step.type.value})")

        # Simulate some work
        await asyncio.sleep(0.1)

        return {
            "step_id": step.id,
            "action": step.action,
            "parameters": step.parameters,
            "executed_at": datetime.utcnow().isoformat()
        }

    def _calculate_retry_delay(
        self,
        config: RetryConfig,
        retry_count: int
    ) -> float:
        """Calculate delay before retry."""
        if config.strategy == RetryStrategy.NONE:
            return 0

        if config.strategy == RetryStrategy.FIXED:
            return config.initial_delay

        if config.strategy == RetryStrategy.LINEAR:
            delay = config.initial_delay * retry_count
        else:  # EXPONENTIAL
            delay = config.initial_delay * (config.multiplier ** (retry_count - 1))

        return min(delay, config.max_delay)

    def _evaluate_condition(
        self,
        condition: str,
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate step condition."""
        try:
            # Simple evaluation - in production, use a proper expression evaluator
            return eval(condition, {"__builtins__": {}}, context)
        except Exception as e:
            logger.warning(f"Condition evaluation failed: {e}")
            return True  # Default to executing the step

    async def _emit_event(self, event: str, data: Dict[str, Any]) -> None:
        """Emit event to registered handlers."""
        handlers = self._event_handlers.get(event, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")


class WorkflowBuilder:
    """Builder for creating workflow definitions."""

    def __init__(self, workflow_id: str, name: str):
        self._workflow = WorkflowDefinition(id=workflow_id, name=name)
        self._step_counter = 0

    def description(self, desc: str) -> "WorkflowBuilder":
        self._workflow.description = desc
        return self

    def max_parallel(self, count: int) -> "WorkflowBuilder":
        self._workflow.max_parallel_steps = count
        return self

    def fail_fast(self, enabled: bool = True) -> "WorkflowBuilder":
        self._workflow.fail_fast = enabled
        return self

    def timeout(self, seconds: int) -> "WorkflowBuilder":
        self._workflow.timeout_seconds = seconds
        return self

    def add_extract_step(
        self,
        name: str,
        connector_id: str,
        parameters: Dict[str, Any] = None,
        depends_on: List[str] = None,
        **kwargs
    ) -> "WorkflowBuilder":
        """Add an extract step."""
        return self._add_step(
            StepType.EXTRACT,
            name,
            connector_id=connector_id,
            parameters=parameters or {},
            depends_on=depends_on or [],
            **kwargs
        )

    def add_transform_step(
        self,
        name: str,
        action: str,
        parameters: Dict[str, Any] = None,
        depends_on: List[str] = None,
        **kwargs
    ) -> "WorkflowBuilder":
        """Add a transform step."""
        return self._add_step(
            StepType.TRANSFORM,
            name,
            action=action,
            parameters=parameters or {},
            depends_on=depends_on or [],
            **kwargs
        )

    def add_load_step(
        self,
        name: str,
        connector_id: str,
        parameters: Dict[str, Any] = None,
        depends_on: List[str] = None,
        **kwargs
    ) -> "WorkflowBuilder":
        """Add a load step."""
        return self._add_step(
            StepType.LOAD,
            name,
            connector_id=connector_id,
            parameters=parameters or {},
            depends_on=depends_on or [],
            **kwargs
        )

    def add_validate_step(
        self,
        name: str,
        action: str,
        parameters: Dict[str, Any] = None,
        depends_on: List[str] = None,
        **kwargs
    ) -> "WorkflowBuilder":
        """Add a validation step."""
        return self._add_step(
            StepType.VALIDATE,
            name,
            action=action,
            parameters=parameters or {},
            depends_on=depends_on or [],
            **kwargs
        )

    def _add_step(
        self,
        step_type: StepType,
        name: str,
        **kwargs
    ) -> "WorkflowBuilder":
        """Add a step to the workflow."""
        self._step_counter += 1
        step_id = kwargs.pop("id", f"step_{self._step_counter}")

        step = WorkflowStep(
            id=step_id,
            name=name,
            type=step_type,
            **kwargs
        )
        self._workflow.steps.append(step)
        return self

    def build(self) -> WorkflowDefinition:
        """Build and return the workflow definition."""
        return self._workflow


# Export main classes
__all__ = [
    "SyncOrchestrator",
    "WorkflowBuilder",
    "WorkflowDefinition",
    "WorkflowStep",
    "WorkflowResult",
    "StepResult",
    "WorkflowStatus",
    "StepStatus",
    "StepType",
    "RetryStrategy",
    "RetryConfig",
]
