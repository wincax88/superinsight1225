"""
Visual Workflow Management for SuperInsight Platform.

Provides workflow visualization, management, and optimization capabilities
for the management console.
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

from src.system.business_metrics import business_metrics_collector


logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(Enum):
    """Individual task status within a workflow."""
    WAITING = "waiting"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowTask:
    """Individual task within a workflow."""
    task_id: str
    name: str
    task_type: str
    status: TaskStatus = TaskStatus.WAITING
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    progress: float = 0.0
    assigned_to: Optional[str] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Get task execution duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def is_ready(self) -> bool:
        """Check if task is ready to execute (all dependencies completed)."""
        return self.status == TaskStatus.READY
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "task_type": self.task_type,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "parameters": self.parameters,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "error_message": self.error_message,
            "progress": self.progress,
            "assigned_to": self.assigned_to
        }


@dataclass
class Workflow:
    """Workflow definition and execution state."""
    workflow_id: str
    name: str
    description: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    tasks: Dict[str, WorkflowTask] = field(default_factory=dict)
    created_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    created_by: Optional[str] = None
    tenant_id: Optional[str] = None
    priority: int = 0
    tags: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> Optional[float]:
        """Get workflow execution duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def progress(self) -> float:
        """Calculate overall workflow progress."""
        if not self.tasks:
            return 0.0
        
        total_tasks = len(self.tasks)
        completed_tasks = sum(
            1 for task in self.tasks.values() 
            if task.status == TaskStatus.COMPLETED
        )
        
        return (completed_tasks / total_tasks) * 100
    
    @property
    def ready_tasks(self) -> List[WorkflowTask]:
        """Get tasks that are ready to execute."""
        ready = []
        
        for task in self.tasks.values():
            if task.status == TaskStatus.WAITING:
                # Check if all dependencies are completed
                deps_completed = all(
                    self.tasks.get(dep_id, WorkflowTask("", "", "")).status == TaskStatus.COMPLETED
                    for dep_id in task.dependencies
                )
                
                if deps_completed:
                    task.status = TaskStatus.READY
                    ready.append(task)
        
        return ready
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()},
            "created_time": self.created_time,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "created_by": self.created_by,
            "tenant_id": self.tenant_id,
            "priority": self.priority,
            "tags": self.tags,
            "progress": self.progress
        }


@dataclass
class WorkflowTemplate:
    """Reusable workflow template."""
    template_id: str
    name: str
    description: str
    category: str
    task_definitions: List[Dict[str, Any]]
    default_parameters: Dict[str, Any] = field(default_factory=dict)
    created_by: Optional[str] = None
    created_time: float = field(default_factory=time.time)
    usage_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "task_definitions": self.task_definitions,
            "default_parameters": self.default_parameters,
            "created_by": self.created_by,
            "created_time": self.created_time,
            "usage_count": self.usage_count
        }


class WorkflowManager:
    """
    Visual workflow management system.
    
    Provides workflow creation, execution, monitoring, and optimization
    capabilities with visual representation support.
    """
    
    def __init__(self):
        self.workflows: Dict[str, Workflow] = {}
        self.templates: Dict[str, WorkflowTemplate] = {}
        self.execution_history: deque = deque(maxlen=1000)
        
        # Execution configuration
        self.max_concurrent_workflows = 10
        self.max_concurrent_tasks_per_workflow = 5
        self.task_timeout = 3600  # 1 hour
        
        # Monitoring
        self.is_running = False
        self._execution_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.workflow_stats = {
            "total_created": 0,
            "total_completed": 0,
            "total_failed": 0,
            "avg_execution_time": 0.0,
            "most_used_templates": []
        }
        
        # Initialize built-in templates
        self._initialize_builtin_templates()
    
    def _initialize_builtin_templates(self):
        """Initialize built-in workflow templates."""
        # Data Processing Workflow
        data_processing_template = WorkflowTemplate(
            template_id="data_processing",
            name="Data Processing Pipeline",
            description="Complete data processing workflow from extraction to annotation",
            category="data_processing",
            task_definitions=[
                {
                    "task_id": "extract_data",
                    "name": "Extract Data",
                    "task_type": "data_extraction",
                    "dependencies": [],
                    "parameters": {"source_type": "database", "batch_size": 100}
                },
                {
                    "task_id": "validate_data",
                    "name": "Validate Data",
                    "task_type": "data_validation",
                    "dependencies": ["extract_data"],
                    "parameters": {"validation_rules": ["required_fields", "data_types"]}
                },
                {
                    "task_id": "ai_preannotation",
                    "name": "AI Pre-annotation",
                    "task_type": "ai_annotation",
                    "dependencies": ["validate_data"],
                    "parameters": {"model": "auto", "confidence_threshold": 0.7}
                },
                {
                    "task_id": "human_review",
                    "name": "Human Review",
                    "task_type": "human_annotation",
                    "dependencies": ["ai_preannotation"],
                    "parameters": {"review_percentage": 20}
                },
                {
                    "task_id": "quality_check",
                    "name": "Quality Check",
                    "task_type": "quality_assessment",
                    "dependencies": ["human_review"],
                    "parameters": {"quality_threshold": 0.85}
                }
            ]
        )
        
        # Model Training Workflow
        model_training_template = WorkflowTemplate(
            template_id="model_training",
            name="AI Model Training Pipeline",
            description="End-to-end AI model training and evaluation workflow",
            category="ai_training",
            task_definitions=[
                {
                    "task_id": "prepare_dataset",
                    "name": "Prepare Training Dataset",
                    "task_type": "dataset_preparation",
                    "dependencies": [],
                    "parameters": {"train_split": 0.8, "validation_split": 0.2}
                },
                {
                    "task_id": "train_model",
                    "name": "Train Model",
                    "task_type": "model_training",
                    "dependencies": ["prepare_dataset"],
                    "parameters": {"epochs": 10, "batch_size": 32}
                },
                {
                    "task_id": "evaluate_model",
                    "name": "Evaluate Model",
                    "task_type": "model_evaluation",
                    "dependencies": ["train_model"],
                    "parameters": {"metrics": ["accuracy", "f1_score", "precision", "recall"]}
                },
                {
                    "task_id": "deploy_model",
                    "name": "Deploy Model",
                    "task_type": "model_deployment",
                    "dependencies": ["evaluate_model"],
                    "parameters": {"deployment_target": "production"}
                }
            ]
        )
        
        # Quality Assurance Workflow
        qa_workflow_template = WorkflowTemplate(
            template_id="quality_assurance",
            name="Quality Assurance Pipeline",
            description="Comprehensive quality assurance and improvement workflow",
            category="quality_management",
            task_definitions=[
                {
                    "task_id": "quality_scan",
                    "name": "Quality Scan",
                    "task_type": "quality_assessment",
                    "dependencies": [],
                    "parameters": {"scan_type": "comprehensive"}
                },
                {
                    "task_id": "identify_issues",
                    "name": "Identify Issues",
                    "task_type": "issue_detection",
                    "dependencies": ["quality_scan"],
                    "parameters": {"severity_threshold": "medium"}
                },
                {
                    "task_id": "create_work_orders",
                    "name": "Create Work Orders",
                    "task_type": "work_order_creation",
                    "dependencies": ["identify_issues"],
                    "parameters": {"auto_assign": True}
                },
                {
                    "task_id": "track_resolution",
                    "name": "Track Resolution",
                    "task_type": "resolution_tracking",
                    "dependencies": ["create_work_orders"],
                    "parameters": {"follow_up_interval": 24}
                }
            ]
        )
        
        self.templates = {
            "data_processing": data_processing_template,
            "model_training": model_training_template,
            "quality_assurance": qa_workflow_template
        }
    
    async def initialize(self):
        """Initialize workflow manager."""
        if self.is_running:
            return
        
        self.is_running = True
        self._execution_task = asyncio.create_task(self._execution_loop())
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Workflow manager initialized")
    
    async def shutdown(self):
        """Shutdown workflow manager."""
        self.is_running = False
        
        # Cancel running tasks
        if self._execution_task:
            self._execution_task.cancel()
            try:
                await self._execution_task
            except asyncio.CancelledError:
                pass
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Workflow manager shutdown")
    
    async def _execution_loop(self):
        """Main workflow execution loop."""
        while self.is_running:
            try:
                await self._process_workflows()
                await asyncio.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Error in workflow execution loop: {e}")
                await asyncio.sleep(10)
    
    async def _monitoring_loop(self):
        """Workflow monitoring and statistics update loop."""
        while self.is_running:
            try:
                await self._update_statistics()
                await self._cleanup_completed_workflows()
                await asyncio.sleep(60)  # Update every minute
            except Exception as e:
                logger.error(f"Error in workflow monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def _process_workflows(self):
        """Process all active workflows."""
        active_workflows = [
            wf for wf in self.workflows.values()
            if wf.status in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING]
        ]
        
        # Limit concurrent workflows
        running_count = sum(
            1 for wf in active_workflows 
            if wf.status == WorkflowStatus.RUNNING
        )
        
        # Start pending workflows if under limit
        if running_count < self.max_concurrent_workflows:
            pending_workflows = [
                wf for wf in active_workflows 
                if wf.status == WorkflowStatus.PENDING
            ]
            
            # Sort by priority
            pending_workflows.sort(key=lambda x: x.priority, reverse=True)
            
            for workflow in pending_workflows[:self.max_concurrent_workflows - running_count]:
                await self._start_workflow(workflow)
        
        # Process running workflows
        for workflow in active_workflows:
            if workflow.status == WorkflowStatus.RUNNING:
                await self._process_workflow_tasks(workflow)
    
    async def _start_workflow(self, workflow: Workflow):
        """Start executing a workflow."""
        workflow.status = WorkflowStatus.RUNNING
        workflow.start_time = time.time()
        
        logger.info(f"Started workflow: {workflow.name} ({workflow.workflow_id})")
        
        # Mark initial tasks as ready
        for task in workflow.tasks.values():
            if not task.dependencies:
                task.status = TaskStatus.READY
    
    async def _process_workflow_tasks(self, workflow: Workflow):
        """Process tasks within a running workflow."""
        # Get ready tasks
        ready_tasks = workflow.ready_tasks
        
        # Get currently running tasks
        running_tasks = [
            task for task in workflow.tasks.values()
            if task.status == TaskStatus.RUNNING
        ]
        
        # Start new tasks if under concurrency limit
        available_slots = self.max_concurrent_tasks_per_workflow - len(running_tasks)
        
        for task in ready_tasks[:available_slots]:
            await self._start_task(workflow, task)
        
        # Check for workflow completion
        if self._is_workflow_complete(workflow):
            await self._complete_workflow(workflow)
        elif self._is_workflow_failed(workflow):
            await self._fail_workflow(workflow)
    
    async def _start_task(self, workflow: Workflow, task: WorkflowTask):
        """Start executing a task."""
        task.status = TaskStatus.RUNNING
        task.start_time = time.time()
        
        logger.info(f"Started task: {task.name} in workflow {workflow.name}")
        
        # Simulate task execution (in real implementation, this would dispatch to actual task handlers)
        asyncio.create_task(self._simulate_task_execution(workflow, task))
    
    async def _simulate_task_execution(self, workflow: Workflow, task: WorkflowTask):
        """Simulate task execution (placeholder for real task execution)."""
        try:
            # Simulate work with progress updates
            for progress in range(0, 101, 20):
                task.progress = progress
                await asyncio.sleep(1)  # Simulate work
                
                if not self.is_running:
                    break
            
            # Complete the task
            task.status = TaskStatus.COMPLETED
            task.end_time = time.time()
            task.progress = 100.0
            
            logger.info(f"Completed task: {task.name} in {task.duration:.2f}s")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.end_time = time.time()
            task.error_message = str(e)
            
            logger.error(f"Task failed: {task.name} - {e}")
    
    def _is_workflow_complete(self, workflow: Workflow) -> bool:
        """Check if workflow is complete."""
        return all(
            task.status in [TaskStatus.COMPLETED, TaskStatus.SKIPPED]
            for task in workflow.tasks.values()
        )
    
    def _is_workflow_failed(self, workflow: Workflow) -> bool:
        """Check if workflow has failed."""
        # Workflow fails if any critical task fails
        return any(
            task.status == TaskStatus.FAILED
            for task in workflow.tasks.values()
        )
    
    async def _complete_workflow(self, workflow: Workflow):
        """Complete a workflow."""
        workflow.status = WorkflowStatus.COMPLETED
        workflow.end_time = time.time()
        
        # Add to execution history
        self.execution_history.append({
            "workflow_id": workflow.workflow_id,
            "name": workflow.name,
            "status": "completed",
            "duration": workflow.duration,
            "completed_time": workflow.end_time
        })
        
        self.workflow_stats["total_completed"] += 1
        
        logger.info(f"Completed workflow: {workflow.name} in {workflow.duration:.2f}s")
    
    async def _fail_workflow(self, workflow: Workflow):
        """Fail a workflow."""
        workflow.status = WorkflowStatus.FAILED
        workflow.end_time = time.time()
        
        # Add to execution history
        self.execution_history.append({
            "workflow_id": workflow.workflow_id,
            "name": workflow.name,
            "status": "failed",
            "duration": workflow.duration,
            "completed_time": workflow.end_time
        })
        
        self.workflow_stats["total_failed"] += 1
        
        logger.warning(f"Failed workflow: {workflow.name}")
    
    async def _update_statistics(self):
        """Update workflow statistics."""
        completed_workflows = [
            wf for wf in self.workflows.values()
            if wf.status == WorkflowStatus.COMPLETED and wf.duration is not None
        ]
        
        if completed_workflows:
            durations = [wf.duration for wf in completed_workflows]
            self.workflow_stats["avg_execution_time"] = sum(durations) / len(durations)
        
        # Update template usage statistics
        template_usage = defaultdict(int)
        for wf in self.workflows.values():
            if hasattr(wf, 'template_id'):
                template_usage[wf.template_id] += 1
        
        self.workflow_stats["most_used_templates"] = sorted(
            template_usage.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
    
    async def _cleanup_completed_workflows(self):
        """Clean up old completed workflows."""
        cutoff_time = time.time() - (7 * 24 * 3600)  # 7 days
        
        to_remove = []
        for workflow_id, workflow in self.workflows.items():
            if (workflow.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED] and
                workflow.end_time and workflow.end_time < cutoff_time):
                to_remove.append(workflow_id)
        
        for workflow_id in to_remove:
            del self.workflows[workflow_id]
    
    # Public API methods
    def create_workflow_from_template(
        self,
        template_id: str,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None,
        tenant_id: Optional[str] = None,
        priority: int = 0
    ) -> str:
        """Create a new workflow from a template."""
        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")
        
        template = self.templates[template_id]
        workflow_id = f"wf_{int(time.time() * 1000)}"
        
        # Create workflow
        workflow = Workflow(
            workflow_id=workflow_id,
            name=name,
            description=f"Created from template: {template.name}",
            created_by=created_by,
            tenant_id=tenant_id,
            priority=priority
        )
        
        # Add template reference
        workflow.template_id = template_id
        
        # Create tasks from template
        merged_params = {**template.default_parameters, **(parameters or {})}
        
        for task_def in template.task_definitions:
            task_params = {**merged_params, **task_def.get("parameters", {})}
            
            task = WorkflowTask(
                task_id=task_def["task_id"],
                name=task_def["name"],
                task_type=task_def["task_type"],
                dependencies=task_def.get("dependencies", []),
                parameters=task_params
            )
            
            workflow.tasks[task.task_id] = task
        
        self.workflows[workflow_id] = workflow
        self.workflow_stats["total_created"] += 1
        
        # Update template usage
        template.usage_count += 1
        
        logger.info(f"Created workflow {name} from template {template_id}")
        
        return workflow_id
    
    def create_custom_workflow(
        self,
        name: str,
        description: str,
        task_definitions: List[Dict[str, Any]],
        created_by: Optional[str] = None,
        tenant_id: Optional[str] = None,
        priority: int = 0
    ) -> str:
        """Create a custom workflow."""
        workflow_id = f"wf_{int(time.time() * 1000)}"
        
        workflow = Workflow(
            workflow_id=workflow_id,
            name=name,
            description=description,
            created_by=created_by,
            tenant_id=tenant_id,
            priority=priority
        )
        
        # Create tasks
        for task_def in task_definitions:
            task = WorkflowTask(
                task_id=task_def["task_id"],
                name=task_def["name"],
                task_type=task_def["task_type"],
                dependencies=task_def.get("dependencies", []),
                parameters=task_def.get("parameters", {})
            )
            
            workflow.tasks[task.task_id] = task
        
        self.workflows[workflow_id] = workflow
        self.workflow_stats["total_created"] += 1
        
        logger.info(f"Created custom workflow: {name}")
        
        return workflow_id
    
    def start_workflow(self, workflow_id: str) -> bool:
        """Start a workflow execution."""
        if workflow_id not in self.workflows:
            return False
        
        workflow = self.workflows[workflow_id]
        if workflow.status != WorkflowStatus.PENDING:
            return False
        
        # Workflow will be picked up by the execution loop
        logger.info(f"Queued workflow for execution: {workflow.name}")
        return True
    
    def pause_workflow(self, workflow_id: str) -> bool:
        """Pause a running workflow."""
        if workflow_id not in self.workflows:
            return False
        
        workflow = self.workflows[workflow_id]
        if workflow.status == WorkflowStatus.RUNNING:
            workflow.status = WorkflowStatus.PAUSED
            logger.info(f"Paused workflow: {workflow.name}")
            return True
        
        return False
    
    def resume_workflow(self, workflow_id: str) -> bool:
        """Resume a paused workflow."""
        if workflow_id not in self.workflows:
            return False
        
        workflow = self.workflows[workflow_id]
        if workflow.status == WorkflowStatus.PAUSED:
            workflow.status = WorkflowStatus.RUNNING
            logger.info(f"Resumed workflow: {workflow.name}")
            return True
        
        return False
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a workflow."""
        if workflow_id not in self.workflows:
            return False
        
        workflow = self.workflows[workflow_id]
        if workflow.status in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING, WorkflowStatus.PAUSED]:
            workflow.status = WorkflowStatus.CANCELLED
            workflow.end_time = time.time()
            logger.info(f"Cancelled workflow: {workflow.name}")
            return True
        
        return False
    
    def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow details."""
        if workflow_id in self.workflows:
            return self.workflows[workflow_id].to_dict()
        return None
    
    def get_workflows(
        self,
        status: Optional[WorkflowStatus] = None,
        created_by: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get workflows with optional filtering."""
        workflows = list(self.workflows.values())
        
        # Apply filters
        if status:
            workflows = [wf for wf in workflows if wf.status == status]
        
        if created_by:
            workflows = [wf for wf in workflows if wf.created_by == created_by]
        
        if tenant_id:
            workflows = [wf for wf in workflows if wf.tenant_id == tenant_id]
        
        # Sort by creation time (newest first)
        workflows.sort(key=lambda x: x.created_time, reverse=True)
        
        return [wf.to_dict() for wf in workflows[:limit]]
    
    def get_templates(self) -> List[Dict[str, Any]]:
        """Get all workflow templates."""
        return [template.to_dict() for template in self.templates.values()]
    
    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get workflow statistics."""
        active_count = len([
            wf for wf in self.workflows.values()
            if wf.status in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING, WorkflowStatus.PAUSED]
        ])
        
        return {
            **self.workflow_stats,
            "active_workflows": active_count,
            "total_workflows": len(self.workflows),
            "templates_count": len(self.templates)
        }
    
    def get_workflow_visualization_data(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow data for visualization."""
        if workflow_id not in self.workflows:
            return None
        
        workflow = self.workflows[workflow_id]
        
        # Create nodes and edges for visualization
        nodes = []
        edges = []
        
        for task in workflow.tasks.values():
            # Create node
            node = {
                "id": task.task_id,
                "label": task.name,
                "type": task.task_type,
                "status": task.status.value,
                "progress": task.progress,
                "duration": task.duration,
                "assigned_to": task.assigned_to
            }
            nodes.append(node)
            
            # Create edges for dependencies
            for dep_id in task.dependencies:
                edge = {
                    "from": dep_id,
                    "to": task.task_id,
                    "type": "dependency"
                }
                edges.append(edge)
        
        return {
            "workflow": workflow.to_dict(),
            "visualization": {
                "nodes": nodes,
                "edges": edges
            }
        }