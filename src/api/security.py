"""
Security API endpoints for SuperInsight Platform.

Provides authentication, user management, and security configuration endpoints.
"""

from datetime import datetime
from typing import List, Optional, Dict
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from src.security.controller import SecurityController
from src.security.models import UserRole, PermissionType, AuditAction
from src.security.middleware import get_current_active_user, require_role, audit_action
from src.database.connection import get_db_session


router = APIRouter(prefix="/api/security", tags=["security"])
security_controller = SecurityController()


# Request/Response Models

class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    username: str
    role: str
    tenant_id: str


class CreateUserRequest(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: str
    role: UserRole
    tenant_id: str


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    full_name: Optional[str]
    role: str
    tenant_id: str
    is_active: bool
    last_login: Optional[datetime]
    created_at: datetime


class PermissionRequest(BaseModel):
    user_id: str
    project_id: str
    permission_type: PermissionType


class IPWhitelistRequest(BaseModel):
    ip_address: str
    ip_range: Optional[str] = None
    description: Optional[str] = None


class IPWhitelistResponse(BaseModel):
    id: str
    ip_address: str
    ip_range: Optional[str]
    description: Optional[str]
    is_active: bool
    created_at: datetime


class MaskingRuleRequest(BaseModel):
    field_name: str
    field_pattern: Optional[str] = None
    masking_type: str
    masking_config: Optional[dict] = None


class MaskingRuleResponse(BaseModel):
    id: str
    field_name: str
    field_pattern: Optional[str]
    masking_type: str
    masking_config: dict
    is_active: bool
    created_at: datetime


class AuditLogResponse(BaseModel):
    id: str
    user_id: Optional[str]
    action: str
    resource_type: str
    resource_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    details: dict
    timestamp: datetime


# Authentication Endpoints

@router.post("/login", response_model=LoginResponse)
async def login(
    request: Request,
    login_data: LoginRequest,
    db: Session = Depends(get_db_session)
):
    """Authenticate user and return access token."""
    user = security_controller.authenticate_user(
        login_data.username, login_data.password, db
    )
    
    if not user:
        # Log failed login attempt
        security_controller.log_user_action(
            user_id=None,
            tenant_id="unknown",
            action=AuditAction.LOGIN,
            resource_type="authentication",
            ip_address=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent"),
            details={"status": "failed", "username": login_data.username},
            db=db
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    # Create access token
    access_token = security_controller.create_access_token(
        str(user.id), user.tenant_id
    )
    
    # Log successful login
    security_controller.log_user_action(
        user_id=user.id,
        tenant_id=user.tenant_id,
        action=AuditAction.LOGIN,
        resource_type="authentication",
        ip_address=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent"),
        details={"status": "success"},
        db=db
    )
    
    return LoginResponse(
        access_token=access_token,
        user_id=str(user.id),
        username=user.username,
        role=user.role.value,
        tenant_id=user.tenant_id
    )


@router.post("/logout")
@audit_action(AuditAction.LOGOUT, "authentication")
async def logout(
    request: Request,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Logout user (mainly for audit logging)."""
    return {"message": "Logged out successfully"}


# User Management Endpoints

@router.post("/users", response_model=UserResponse)
@require_role(["admin"])
@audit_action(AuditAction.CREATE, "user")
async def create_user(
    user_data: CreateUserRequest,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Create a new user (admin only)."""
    user = security_controller.create_user(
        username=user_data.username,
        email=user_data.email,
        password=user_data.password,
        full_name=user_data.full_name,
        role=user_data.role,
        tenant_id=user_data.tenant_id,
        db=db
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already exists"
        )
    
    return UserResponse(
        id=str(user.id),
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        role=user.role.value,
        tenant_id=user.tenant_id,
        is_active=user.is_active,
        last_login=user.last_login,
        created_at=user.created_at
    )


@router.get("/users/me", response_model=UserResponse)
async def get_current_user_info(
    current_user = Depends(get_current_active_user)
):
    """Get current user information."""
    return UserResponse(
        id=str(current_user.id),
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role.value,
        tenant_id=current_user.tenant_id,
        is_active=current_user.is_active,
        last_login=current_user.last_login,
        created_at=current_user.created_at
    )


@router.put("/users/{user_id}/role")
@require_role(["admin"])
@audit_action(AuditAction.UPDATE, "user", "user_id")
async def update_user_role(
    user_id: UUID,
    new_role: UserRole,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Update a user's role (admin only)."""
    success = security_controller.update_user_role(user_id, new_role, db)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return {"message": "User role updated successfully"}


@router.delete("/users/{user_id}")
@require_role(["admin"])
@audit_action(AuditAction.DELETE, "user", "user_id")
async def deactivate_user(
    user_id: UUID,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Deactivate a user account (admin only)."""
    success = security_controller.deactivate_user(user_id, db)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return {"message": "User deactivated successfully"}


# Permission Management Endpoints

@router.post("/permissions")
@require_role(["admin"])
@audit_action(AuditAction.CREATE, "permission")
async def grant_permission(
    permission_data: PermissionRequest,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Grant a project permission to a user (admin only)."""
    success = security_controller.grant_project_permission(
        user_id=UUID(permission_data.user_id),
        project_id=permission_data.project_id,
        permission_type=permission_data.permission_type,
        granted_by=current_user.id,
        db=db
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to grant permission"
        )
    
    return {"message": "Permission granted successfully"}


@router.delete("/permissions")
@require_role(["admin"])
@audit_action(AuditAction.DELETE, "permission")
async def revoke_permission(
    permission_data: PermissionRequest,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Revoke a project permission from a user (admin only)."""
    success = security_controller.revoke_project_permission(
        user_id=UUID(permission_data.user_id),
        project_id=permission_data.project_id,
        permission_type=permission_data.permission_type,
        db=db
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Permission not found"
        )
    
    return {"message": "Permission revoked successfully"}


@router.get("/permissions/projects")
async def get_user_projects(
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Get projects accessible to current user."""
    projects = security_controller.get_user_projects(current_user.id, db)
    return {"projects": projects}


# IP Whitelist Management

@router.post("/ip-whitelist", response_model=IPWhitelistResponse)
@require_role(["admin"])
@audit_action(AuditAction.CREATE, "ip_whitelist")
async def add_ip_whitelist(
    ip_data: IPWhitelistRequest,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Add IP address to whitelist (admin only)."""
    success = security_controller.add_ip_to_whitelist(
        ip_address=ip_data.ip_address,
        tenant_id=current_user.tenant_id,
        created_by=current_user.id,
        description=ip_data.description,
        ip_range=ip_data.ip_range,
        db=db
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to add IP to whitelist"
        )
    
    return {"message": "IP added to whitelist successfully"}


# Data Masking Rules

@router.post("/masking-rules", response_model=MaskingRuleResponse)
@require_role(["admin"])
@audit_action(AuditAction.CREATE, "masking_rule")
async def add_masking_rule(
    rule_data: MaskingRuleRequest,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Add data masking rule (admin only)."""
    success = security_controller.add_masking_rule(
        tenant_id=current_user.tenant_id,
        field_name=rule_data.field_name,
        masking_type=rule_data.masking_type,
        created_by=current_user.id,
        field_pattern=rule_data.field_pattern,
        masking_config=rule_data.masking_config,
        db=db
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to add masking rule"
        )
    
    return {"message": "Masking rule added successfully"}


# Audit Log Endpoints

@router.get("/audit-logs", response_model=List[AuditLogResponse])
@require_role(["admin"])
async def get_audit_logs(
    user_id: Optional[UUID] = None,
    action: Optional[AuditAction] = None,
    resource_type: Optional[str] = None,
    limit: int = 100,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Get audit logs (admin only)."""
    logs = security_controller.get_audit_logs(
        tenant_id=current_user.tenant_id,
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        limit=limit,
        db=db
    )
    
    return [
        AuditLogResponse(
            id=str(log.id),
            user_id=str(log.user_id) if log.user_id else None,
            action=log.action.value,
            resource_type=log.resource_type,
            resource_id=log.resource_id,
            ip_address=str(log.ip_address) if log.ip_address else None,
            user_agent=log.user_agent,
            details=log.details,
            timestamp=log.timestamp
        )
        for log in logs
    ]

# Additional imports for audit service
from src.security.audit_service import AuditService

# Initialize audit service
audit_service = AuditService()


# Additional Response Models for Audit Features

class SecuritySummaryResponse(BaseModel):
    period_days: int
    total_events: int
    failed_logins: int
    sensitive_operations: int
    active_users: int
    unique_ip_addresses: int
    recent_failed_logins: List[dict]


class UserActivityResponse(BaseModel):
    total_actions: int
    actions_by_type: Dict[str, int]
    resources_accessed: Dict[str, int]
    daily_activity: Dict[str, int]
    suspicious_patterns: List[dict]
    analysis_period_days: int


class SecurityAlertResponse(BaseModel):
    type: str
    severity: str
    message: str
    timestamp: str
    action_required: str


class LogSearchRequest(BaseModel):
    user_id: Optional[str] = None
    action: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    ip_address: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    search_text: Optional[str] = None
    page: int = 1
    page_size: int = 50


class LogSearchResponse(BaseModel):
    logs: List[AuditLogResponse]
    total_count: int
    page: int
    page_size: int
    total_pages: int


class LogStatisticsResponse(BaseModel):
    total_logs: int
    oldest_log: Optional[str]
    newest_log: Optional[str]
    storage_size_estimate: str


# Enhanced Audit Endpoints

@router.get("/audit/summary", response_model=SecuritySummaryResponse)
@require_role(["admin"])
async def get_security_summary(
    days: int = 7,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Get security summary for the tenant (admin only)."""
    summary = audit_service.get_security_summary(
        tenant_id=current_user.tenant_id,
        days=days,
        db=db
    )
    return SecuritySummaryResponse(**summary)


@router.get("/audit/user-activity/{user_id}", response_model=UserActivityResponse)
@require_role(["admin"])
async def get_user_activity_analysis(
    user_id: UUID,
    days: int = 30,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Get detailed user activity analysis (admin only)."""
    activity = audit_service.analyze_user_activity(
        user_id=user_id,
        tenant_id=current_user.tenant_id,
        days=days,
        db=db
    )
    return UserActivityResponse(**activity)


@router.post("/audit/search", response_model=LogSearchResponse)
@require_role(["admin"])
async def search_audit_logs(
    search_params: LogSearchRequest,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Advanced search in audit logs (admin only)."""
    query_params = search_params.dict(exclude_none=True)
    
    logs, total_count = audit_service.search_logs(
        tenant_id=current_user.tenant_id,
        query_params=query_params,
        db=db
    )
    
    total_pages = (total_count + search_params.page_size - 1) // search_params.page_size
    
    log_responses = [
        AuditLogResponse(
            id=str(log.id),
            user_id=str(log.user_id) if log.user_id else None,
            action=log.action.value,
            resource_type=log.resource_type,
            resource_id=log.resource_id,
            ip_address=str(log.ip_address) if log.ip_address else None,
            user_agent=log.user_agent,
            details=log.details,
            timestamp=log.timestamp
        )
        for log in logs
    ]
    
    return LogSearchResponse(
        logs=log_responses,
        total_count=total_count,
        page=search_params.page,
        page_size=search_params.page_size,
        total_pages=total_pages
    )


@router.get("/audit/alerts", response_model=List[SecurityAlertResponse])
@require_role(["admin"])
async def get_security_alerts(
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Get current security alerts (admin only)."""
    alerts = audit_service.check_security_alerts(
        tenant_id=current_user.tenant_id,
        db=db
    )
    
    return [SecurityAlertResponse(**alert) for alert in alerts]


@router.get("/audit/statistics", response_model=LogStatisticsResponse)
@require_role(["admin"])
async def get_log_statistics(
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Get audit log storage statistics (admin only)."""
    stats = audit_service.get_log_statistics(
        tenant_id=current_user.tenant_id,
        db=db
    )
    return LogStatisticsResponse(**stats)


@router.post("/audit/rotate-logs")
@require_role(["admin"])
@audit_action(AuditAction.DELETE, "audit_logs")
async def rotate_audit_logs(
    retention_days: int = 365,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Rotate (archive) old audit logs (admin only)."""
    result = audit_service.rotate_logs(
        tenant_id=current_user.tenant_id,
        retention_days=retention_days,
        db=db
    )
    return result


@router.post("/audit/system-event")
@require_role(["admin"])
async def log_system_event(
    event_type: str,
    description: str,
    details: Optional[dict] = None,
    current_user = Depends(get_current_active_user),
    db: Session = Depends(get_db_session)
):
    """Log a system event (admin only)."""
    success = audit_service.log_system_event(
        event_type=event_type,
        description=description,
        tenant_id=current_user.tenant_id,
        details=details,
        db=db
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to log system event"
        )
    
    return {"message": "System event logged successfully"}

# Additional imports for log management
from src.security.log_config import log_manager


# Log Management Response Models

class LogFileInfo(BaseModel):
    path: str
    size_bytes: int
    size_mb: float
    modified: str


class LogFilesResponse(BaseModel):
    log_files: Dict[str, LogFileInfo]


class LogLevelsResponse(BaseModel):
    log_levels: Dict[str, str]


# Log Management Endpoints

@router.get("/logs/files", response_model=LogFilesResponse)
@require_role(["admin"])
async def get_log_files(
    current_user = Depends(get_current_active_user)
):
    """Get information about log files (admin only)."""
    files_info = log_manager.get_log_files()
    
    log_files = {
        name: LogFileInfo(**info)
        for name, info in files_info.items()
    }
    
    return LogFilesResponse(log_files=log_files)


@router.get("/logs/levels", response_model=LogLevelsResponse)
@require_role(["admin"])
async def get_log_levels(
    current_user = Depends(get_current_active_user)
):
    """Get current log levels (admin only)."""
    levels = log_manager.get_log_levels()
    return LogLevelsResponse(log_levels=levels)


@router.put("/logs/levels/{logger_name}")
@require_role(["admin"])
@audit_action(AuditAction.UPDATE, "log_config")
async def set_log_level(
    logger_name: str,
    level: str,
    current_user = Depends(get_current_active_user)
):
    """Set log level for a specific logger (admin only)."""
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    
    if level.upper() not in valid_levels:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid log level. Must be one of: {valid_levels}"
        )
    
    success = log_manager.set_log_level(logger_name, level)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Logger not found"
        )
    
    return {"message": f"Log level for {logger_name} set to {level.upper()}"}


@router.post("/logs/rotate")
@require_role(["admin"])
@audit_action(AuditAction.UPDATE, "log_files")
async def rotate_logs(
    current_user = Depends(get_current_active_user)
):
    """Manually rotate log files (admin only)."""
    result = log_manager.rotate_logs()
    return result