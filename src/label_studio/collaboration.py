"""
Multi-user Collaboration Module for Label Studio

Provides role-based access control, task assignment, and progress tracking
for business experts, technical experts, and outsourced annotators.
"""

import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum
from dataclasses import dataclass, field
from sqlalchemy.orm import Session
from sqlalchemy import select
import httpx

from src.database.connection import get_db_session
from src.database.models import TaskModel, TaskStatus
from src.config.settings import settings

logger = logging.getLogger(__name__)


class UserRole(str, Enum):
    """User roles for annotation collaboration"""
    BUSINESS_EXPERT = "business_expert"  # 业务专家
    TECHNICAL_EXPERT = "technical_expert"  # 技术专家
    OUTSOURCED_ANNOTATOR = "outsourced_annotator"  # 外包人员
    ADMIN = "admin"  # 管理员
    REVIEWER = "reviewer"  # 审核员


class Permission(str, Enum):
    """Permissions for different operations"""
    VIEW_TASKS = "view_tasks"
    CREATE_TASKS = "create_tasks"
    EDIT_TASKS = "edit_tasks"
    DELETE_TASKS = "delete_tasks"
    ANNOTATE = "annotate"
    REVIEW = "review"
    APPROVE = "approve"
    MANAGE_USERS = "manage_users"
    MANAGE_PROJECT = "manage_project"
    VIEW_ANALYTICS = "view_analytics"
    EXPORT_DATA = "export_data"


# Role-based permission mapping
ROLE_PERMISSIONS: Dict[UserRole, Set[Permission]] = {
    UserRole.ADMIN: {
        Permission.VIEW_TASKS,
        Permission.CREATE_TASKS,
        Permission.EDIT_TASKS,
        Permission.DELETE_TASKS,
        Permission.ANNOTATE,
        Permission.REVIEW,
        Permission.APPROVE,
        Permission.MANAGE_USERS,
        Permission.MANAGE_PROJECT,
        Permission.VIEW_ANALYTICS,
        Permission.EXPORT_DATA
    },
    UserRole.BUSINESS_EXPERT: {
        Permission.VIEW_TASKS,
        Permission.CREATE_TASKS,
        Permission.ANNOTATE,
        Permission.REVIEW,
        Permission.VIEW_ANALYTICS
    },
    UserRole.TECHNICAL_EXPERT: {
        Permission.VIEW_TASKS,
        Permission.CREATE_TASKS,
        Permission.EDIT_TASKS,
        Permission.ANNOTATE,
        Permission.REVIEW,
        Permission.VIEW_ANALYTICS,
        Permission.EXPORT_DATA
    },
    UserRole.OUTSOURCED_ANNOTATOR: {
        Permission.VIEW_TASKS,
        Permission.ANNOTATE
    },
    UserRole.REVIEWER: {
        Permission.VIEW_TASKS,
        Permission.REVIEW,
        Permission.APPROVE,
        Permission.VIEW_ANALYTICS
    }
}


@dataclass
class User:
    """User model for collaboration"""
    id: str
    username: str
    email: str
    role: UserRole
    tenant_id: str
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission"""
        return permission in ROLE_PERMISSIONS.get(self.role, set())
    
    def get_permissions(self) -> Set[Permission]:
        """Get all permissions for this user's role"""
        return ROLE_PERMISSIONS.get(self.role, set())


@dataclass
class TaskAssignment:
    """Task assignment to a user"""
    id: UUID = field(default_factory=uuid4)
    task_id: UUID = field(default=None)
    user_id: str = field(default="")
    assigned_by: str = field(default="")
    assigned_at: datetime = field(default_factory=datetime.now)
    due_date: Optional[datetime] = None
    status: str = "assigned"  # assigned, in_progress, completed, reviewed
    notes: str = ""


@dataclass
class ProgressStats:
    """Progress tracking statistics"""
    total_tasks: int = 0
    pending_tasks: int = 0
    in_progress_tasks: int = 0
    completed_tasks: int = 0
    reviewed_tasks: int = 0
    completion_rate: float = 0.0
    average_time_per_task: float = 0.0
    user_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class CollaborationManager:
    """
    Manages multi-user collaboration for annotation projects.
    
    Handles role-based access control, task assignment, and progress tracking.
    """
    
    def __init__(self, label_studio_url: str = None, api_token: str = None):
        """Initialize collaboration manager"""
        self.label_studio_url = (label_studio_url or 
                                settings.label_studio.label_studio_url).rstrip('/')
        self.api_token = api_token or settings.label_studio.label_studio_api_token
        self.headers = {
            'Authorization': f'Token {self.api_token}' if self.api_token else '',
            'Content-Type': 'application/json'
        }
        
        # In-memory storage for users and assignments (should be moved to database)
        self._users: Dict[str, User] = {}
        self._assignments: Dict[UUID, TaskAssignment] = {}
    
    def create_user(self, 
                   username: str,
                   email: str,
                   role: UserRole,
                   tenant_id: str,
                   metadata: Dict[str, Any] = None) -> User:
        """
        Create a new user with specified role.
        
        Args:
            username: User's username
            email: User's email address
            role: User's role (business expert, technical expert, etc.)
            tenant_id: Tenant identifier for multi-tenancy
            metadata: Additional user metadata
            
        Returns:
            User: Created user object
        """
        user_id = str(uuid4())
        user = User(
            id=user_id,
            username=username,
            email=email,
            role=role,
            tenant_id=tenant_id,
            metadata=metadata or {}
        )
        
        self._users[user_id] = user
        logger.info(f"Created user {username} with role {role.value}")
        
        return user
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self._users.get(user_id)
    
    def list_users(self, tenant_id: Optional[str] = None, role: Optional[UserRole] = None) -> List[User]:
        """
        List users with optional filtering.
        
        Args:
            tenant_id: Filter by tenant ID
            role: Filter by user role
            
        Returns:
            List[User]: Filtered list of users
        """
        users = list(self._users.values())
        
        if tenant_id:
            users = [u for u in users if u.tenant_id == tenant_id]
        
        if role:
            users = [u for u in users if u.role == role]
        
        return users
    
    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """
        Check if a user has a specific permission.
        
        Args:
            user_id: User identifier
            permission: Permission to check
            
        Returns:
            bool: True if user has permission
        """
        user = self.get_user(user_id)
        if not user:
            logger.warning(f"User {user_id} not found")
            return False
        
        if not user.is_active:
            logger.warning(f"User {user_id} is not active")
            return False
        
        return user.has_permission(permission)
    
    def assign_task(self,
                   task_id: UUID,
                   user_id: str,
                   assigned_by: str,
                   due_date: Optional[datetime] = None,
                   notes: str = "") -> TaskAssignment:
        """
        Assign a task to a user.
        
        Args:
            task_id: Task identifier
            user_id: User to assign task to
            assigned_by: User who is making the assignment
            due_date: Optional due date for the task
            notes: Optional notes about the assignment
            
        Returns:
            TaskAssignment: Created assignment
            
        Raises:
            ValueError: If user doesn't have permission to annotate
        """
        # Check if user exists and has permission
        user = self.get_user(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        if not user.has_permission(Permission.ANNOTATE):
            raise ValueError(f"User {user_id} does not have permission to annotate")
        
        # Create assignment
        assignment = TaskAssignment(
            task_id=task_id,
            user_id=user_id,
            assigned_by=assigned_by,
            assigned_at=datetime.now(),
            due_date=due_date,
            status="assigned",
            notes=notes
        )
        
        self._assignments[assignment.id] = assignment
        logger.info(f"Assigned task {task_id} to user {user_id}")
        
        return assignment
    
    def bulk_assign_tasks(self,
                         task_ids: List[UUID],
                         user_ids: List[str],
                         assigned_by: str,
                         strategy: str = "round_robin") -> List[TaskAssignment]:
        """
        Assign multiple tasks to multiple users.
        
        Args:
            task_ids: List of task identifiers
            user_ids: List of user identifiers
            assigned_by: User making the assignments
            strategy: Assignment strategy (round_robin, random, load_balanced)
            
        Returns:
            List[TaskAssignment]: Created assignments
        """
        assignments = []
        
        if strategy == "round_robin":
            for i, task_id in enumerate(task_ids):
                user_id = user_ids[i % len(user_ids)]
                try:
                    assignment = self.assign_task(task_id, user_id, assigned_by)
                    assignments.append(assignment)
                except ValueError as e:
                    logger.error(f"Failed to assign task {task_id}: {str(e)}")
        
        elif strategy == "load_balanced":
            # Get current workload for each user
            user_workloads = {uid: 0 for uid in user_ids}
            for assignment in self._assignments.values():
                if assignment.user_id in user_workloads and assignment.status in ["assigned", "in_progress"]:
                    user_workloads[assignment.user_id] += 1
            
            # Assign to user with least workload
            for task_id in task_ids:
                user_id = min(user_workloads, key=user_workloads.get)
                try:
                    assignment = self.assign_task(task_id, user_id, assigned_by)
                    assignments.append(assignment)
                    user_workloads[user_id] += 1
                except ValueError as e:
                    logger.error(f"Failed to assign task {task_id}: {str(e)}")
        
        logger.info(f"Bulk assigned {len(assignments)} tasks using {strategy} strategy")
        return assignments
    
    def get_user_tasks(self, user_id: str, status: Optional[str] = None) -> List[TaskAssignment]:
        """
        Get all tasks assigned to a user.
        
        Args:
            user_id: User identifier
            status: Optional status filter
            
        Returns:
            List[TaskAssignment]: User's task assignments
        """
        assignments = [a for a in self._assignments.values() if a.user_id == user_id]
        
        if status:
            assignments = [a for a in assignments if a.status == status]
        
        return assignments
    
    def update_assignment_status(self, assignment_id: UUID, new_status: str) -> bool:
        """
        Update the status of a task assignment.
        
        Args:
            assignment_id: Assignment identifier
            new_status: New status value
            
        Returns:
            bool: True if update was successful
        """
        assignment = self._assignments.get(assignment_id)
        if not assignment:
            logger.warning(f"Assignment {assignment_id} not found")
            return False
        
        assignment.status = new_status
        logger.info(f"Updated assignment {assignment_id} status to {new_status}")
        
        return True
    
    def get_progress_stats(self, project_id: str, tenant_id: Optional[str] = None) -> ProgressStats:
        """
        Get real-time progress statistics for a project.
        
        Args:
            project_id: Project identifier
            tenant_id: Optional tenant filter
            
        Returns:
            ProgressStats: Progress statistics
        """
        stats = ProgressStats()
        
        try:
            with get_db_session() as db:
                # Query tasks for the project
                stmt = select(TaskModel).where(TaskModel.project_id == project_id)
                
                tasks = list(db.execute(stmt).scalars().all())
                stats.total_tasks = len(tasks)
                
                # Count by status
                for task in tasks:
                    if task.status == TaskStatus.PENDING:
                        stats.pending_tasks += 1
                    elif task.status == TaskStatus.IN_PROGRESS:
                        stats.in_progress_tasks += 1
                    elif task.status == TaskStatus.COMPLETED:
                        stats.completed_tasks += 1
                    elif task.status == TaskStatus.REVIEWED:
                        stats.reviewed_tasks += 1
                
                # Calculate completion rate
                if stats.total_tasks > 0:
                    stats.completion_rate = (stats.completed_tasks + stats.reviewed_tasks) / stats.total_tasks
                
                # Calculate user-specific stats
                user_task_counts: Dict[str, Dict[str, int]] = {}
                for assignment in self._assignments.values():
                    user_id = assignment.user_id
                    if user_id not in user_task_counts:
                        user_task_counts[user_id] = {
                            "assigned": 0,
                            "in_progress": 0,
                            "completed": 0
                        }
                    
                    if assignment.status == "assigned":
                        user_task_counts[user_id]["assigned"] += 1
                    elif assignment.status == "in_progress":
                        user_task_counts[user_id]["in_progress"] += 1
                    elif assignment.status in ["completed", "reviewed"]:
                        user_task_counts[user_id]["completed"] += 1
                
                # Add user information to stats
                for user_id, counts in user_task_counts.items():
                    user = self.get_user(user_id)
                    if user:
                        stats.user_stats[user_id] = {
                            "username": user.username,
                            "role": user.role.value,
                            **counts
                        }
                
                logger.info(f"Generated progress stats for project {project_id}")
                
        except Exception as e:
            logger.error(f"Error generating progress stats: {str(e)}")
        
        return stats
    
    async def sync_with_label_studio(self, project_id: str) -> bool:
        """
        Synchronize user roles and permissions with Label Studio.
        
        Args:
            project_id: Label Studio project ID
            
        Returns:
            bool: True if sync was successful
        """
        try:
            # Get project members from Label Studio
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.label_studio_url}/api/projects/{project_id}/members/",
                    headers=self.headers
                )
                
                if response.status_code == 200:
                    members = response.json()
                    logger.info(f"Synced {len(members)} members from Label Studio project {project_id}")
                    return True
                else:
                    logger.error(f"Failed to sync with Label Studio: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error syncing with Label Studio: {str(e)}")
            return False
    
    async def add_user_to_project(self, project_id: str, user_id: str) -> bool:
        """
        Add a user to a Label Studio project.
        
        Args:
            project_id: Label Studio project ID
            user_id: User identifier
            
        Returns:
            bool: True if user was added successfully
        """
        user = self.get_user(user_id)
        if not user:
            logger.error(f"User {user_id} not found")
            return False
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.label_studio_url}/api/projects/{project_id}/members/",
                    headers=self.headers,
                    json={
                        "user": user.id,
                        "email": user.email
                    }
                )
                
                if response.status_code in [200, 201]:
                    logger.info(f"Added user {user_id} to project {project_id}")
                    return True
                else:
                    logger.error(f"Failed to add user to project: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error adding user to project: {str(e)}")
            return False


# Global collaboration manager instance
collaboration_manager = CollaborationManager()
