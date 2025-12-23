"""
Collaboration API endpoints for multi-user annotation management.

Provides REST API endpoints for user management, task assignment,
and progress tracking in Label Studio projects.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from src.label_studio.collaboration import (
    collaboration_manager,
    User,
    UserRole,
    Permission,
    TaskAssignment,
    ProgressStats
)
from src.label_studio.auth import auth_manager, AuthenticationError

logger = logging.getLogger(__name__)

# FastAPI router for collaboration endpoints
router = APIRouter(prefix="/api/collaboration", tags=["collaboration"])

# Security scheme for JWT authentication
security = HTTPBearer()


# Pydantic models for API requests/responses
class UserCreateRequest(BaseModel):
    """Request model for creating a new user"""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    role: UserRole
    tenant_id: str = Field(..., min_length=1, max_length=100)
    metadata: Optional[Dict[str, Any]] = None


class UserResponse(BaseModel):
    """Response model for user information"""
    id: str
    username: str
    email: str
    role: UserRole
    tenant_id: str
    is_active: bool
    created_at: datetime
    metadata: Dict[str, Any]


class LoginRequest(BaseModel):
    """Request model for user login"""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Response model for login"""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class TaskAssignmentRequest(BaseModel):
    """Request model for task assignment"""
    task_id: UUID
    user_id: str
    due_date: Optional[datetime] = None
    notes: Optional[str] = ""


class BulkAssignmentRequest(BaseModel):
    """Request model for bulk task assignment"""
    task_ids: List[UUID]
    user_ids: List[str]
    strategy: str = Field(default="round_robin", pattern="^(round_robin|load_balanced|random)$")


class TaskAssignmentResponse(BaseModel):
    """Response model for task assignment"""
    id: UUID
    task_id: UUID
    user_id: str
    assigned_by: str
    assigned_at: datetime
    due_date: Optional[datetime]
    status: str
    notes: str


class ProgressStatsResponse(BaseModel):
    """Response model for progress statistics"""
    total_tasks: int
    pending_tasks: int
    in_progress_tasks: int
    completed_tasks: int
    reviewed_tasks: int
    completion_rate: float
    average_time_per_task: float
    user_stats: Dict[str, Dict[str, Any]]


# Dependency to get current user from JWT token
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current user from JWT token"""
    try:
        user = auth_manager.get_current_user(credentials.credentials)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Dependency to check user permissions
def require_permission(permission: Permission):
    """Dependency factory to require specific permission"""
    def permission_checker(current_user: User = Depends(get_current_user)) -> User:
        if not current_user.has_permission(permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {permission.value}"
            )
        return current_user
    return permission_checker


# Authentication endpoints
@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Authenticate user and return access token"""
    try:
        user = auth_manager.authenticate_user(request.username, request.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
        
        # Create access token
        access_token = auth_manager.create_access_token(user.id)
        
        return LoginResponse(
            access_token=access_token,
            user=UserResponse(
                id=user.id,
                username=user.username,
                email=user.email,
                role=user.role,
                tenant_id=user.tenant_id,
                is_active=user.is_active,
                created_at=user.created_at,
                metadata=user.metadata
            )
        )
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        role=current_user.role,
        tenant_id=current_user.tenant_id,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
        metadata=current_user.metadata
    )


# User management endpoints
@router.post("/users", response_model=UserResponse)
async def create_user(
    request: UserCreateRequest,
    current_user: User = Depends(require_permission(Permission.MANAGE_USERS))
):
    """Create a new user (admin only)"""
    try:
        user = collaboration_manager.create_user(
            username=request.username,
            email=request.email,
            role=request.role,
            tenant_id=request.tenant_id,
            metadata=request.metadata or {}
        )
        
        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            role=user.role,
            tenant_id=user.tenant_id,
            is_active=user.is_active,
            created_at=user.created_at,
            metadata=user.metadata
        )
        
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )


@router.get("/users", response_model=List[UserResponse])
async def list_users(
    tenant_id: Optional[str] = None,
    role: Optional[UserRole] = None,
    current_user: User = Depends(require_permission(Permission.MANAGE_USERS))
):
    """List users with optional filtering (admin only)"""
    try:
        users = collaboration_manager.list_users(tenant_id=tenant_id, role=role)
        
        return [
            UserResponse(
                id=user.id,
                username=user.username,
                email=user.email,
                role=user.role,
                tenant_id=user.tenant_id,
                is_active=user.is_active,
                created_at=user.created_at,
                metadata=user.metadata
            )
            for user in users
        ]
        
    except Exception as e:
        logger.error(f"Error listing users: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list users"
        )


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    current_user: User = Depends(require_permission(Permission.MANAGE_USERS))
):
    """Get user by ID (admin only)"""
    user = collaboration_manager.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        role=user.role,
        tenant_id=user.tenant_id,
        is_active=user.is_active,
        created_at=user.created_at,
        metadata=user.metadata
    )


# Task assignment endpoints
@router.post("/assignments", response_model=TaskAssignmentResponse)
async def assign_task(
    request: TaskAssignmentRequest,
    current_user: User = Depends(require_permission(Permission.CREATE_TASKS))
):
    """Assign a task to a user"""
    try:
        assignment = collaboration_manager.assign_task(
            task_id=request.task_id,
            user_id=request.user_id,
            assigned_by=current_user.id,
            due_date=request.due_date,
            notes=request.notes or ""
        )
        
        return TaskAssignmentResponse(
            id=assignment.id,
            task_id=assignment.task_id,
            user_id=assignment.user_id,
            assigned_by=assignment.assigned_by,
            assigned_at=assignment.assigned_at,
            due_date=assignment.due_date,
            status=assignment.status,
            notes=assignment.notes
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error assigning task: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to assign task"
        )


@router.post("/assignments/bulk", response_model=List[TaskAssignmentResponse])
async def bulk_assign_tasks(
    request: BulkAssignmentRequest,
    current_user: User = Depends(require_permission(Permission.CREATE_TASKS))
):
    """Assign multiple tasks to multiple users"""
    try:
        assignments = collaboration_manager.bulk_assign_tasks(
            task_ids=request.task_ids,
            user_ids=request.user_ids,
            assigned_by=current_user.id,
            strategy=request.strategy
        )
        
        return [
            TaskAssignmentResponse(
                id=assignment.id,
                task_id=assignment.task_id,
                user_id=assignment.user_id,
                assigned_by=assignment.assigned_by,
                assigned_at=assignment.assigned_at,
                due_date=assignment.due_date,
                status=assignment.status,
                notes=assignment.notes
            )
            for assignment in assignments
        ]
        
    except Exception as e:
        logger.error(f"Error bulk assigning tasks: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to bulk assign tasks"
        )


@router.get("/assignments/user/{user_id}", response_model=List[TaskAssignmentResponse])
async def get_user_assignments(
    user_id: str,
    status: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Get task assignments for a user"""
    # Users can only see their own assignments unless they're admin
    if user_id != current_user.id and not current_user.has_permission(Permission.MANAGE_USERS):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Can only view your own assignments"
        )
    
    try:
        assignments = collaboration_manager.get_user_tasks(user_id, status)
        
        return [
            TaskAssignmentResponse(
                id=assignment.id,
                task_id=assignment.task_id,
                user_id=assignment.user_id,
                assigned_by=assignment.assigned_by,
                assigned_at=assignment.assigned_at,
                due_date=assignment.due_date,
                status=assignment.status,
                notes=assignment.notes
            )
            for assignment in assignments
        ]
        
    except Exception as e:
        logger.error(f"Error getting user assignments: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user assignments"
        )


@router.put("/assignments/{assignment_id}/status")
async def update_assignment_status(
    assignment_id: UUID,
    new_status: str,
    current_user: User = Depends(get_current_user)
):
    """Update assignment status"""
    try:
        success = collaboration_manager.update_assignment_status(assignment_id, new_status)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Assignment not found"
            )
        
        return {"message": "Assignment status updated successfully"}
        
    except Exception as e:
        logger.error(f"Error updating assignment status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update assignment status"
        )


# Progress tracking endpoints
@router.get("/progress/{project_id}", response_model=ProgressStatsResponse)
async def get_progress_stats(
    project_id: str,
    tenant_id: Optional[str] = None,
    current_user: User = Depends(require_permission(Permission.VIEW_ANALYTICS))
):
    """Get real-time progress statistics for a project"""
    try:
        stats = collaboration_manager.get_progress_stats(project_id, tenant_id)
        
        return ProgressStatsResponse(
            total_tasks=stats.total_tasks,
            pending_tasks=stats.pending_tasks,
            in_progress_tasks=stats.in_progress_tasks,
            completed_tasks=stats.completed_tasks,
            reviewed_tasks=stats.reviewed_tasks,
            completion_rate=stats.completion_rate,
            average_time_per_task=stats.average_time_per_task,
            user_stats=stats.user_stats
        )
        
    except Exception as e:
        logger.error(f"Error getting progress stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get progress statistics"
        )


# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}