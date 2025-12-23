"""
Security middleware for SuperInsight Platform.

Provides authentication and authorization middleware for FastAPI applications.
"""

from functools import wraps
from typing import Optional, Callable, List
from fastapi import HTTPException, Request, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from src.security.controller import SecurityController
from src.security.models import UserModel, PermissionType, AuditAction
from src.database.connection import get_db_session


security_scheme = HTTPBearer()
security_controller = SecurityController()


class SecurityMiddleware:
    """Security middleware for request authentication and authorization."""
    
    def __init__(self):
        self.security_controller = SecurityController()
    
    async def authenticate_request(
        self,
        request: Request,
        credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
        db: Session = Depends(get_db_session)
    ) -> UserModel:
        """Authenticate a request using JWT token."""
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication credentials required"
            )
        
        # Verify JWT token
        payload = self.security_controller.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        
        # Get user from database
        user = self.security_controller.get_user_by_id(payload["user_id"], db)
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Check IP whitelist if configured
        client_ip = self._get_client_ip(request)
        if not self.security_controller.is_ip_whitelisted(client_ip, user.tenant_id, db):
            # Log unauthorized access attempt
            self.security_controller.log_user_action(
                user_id=user.id,
                tenant_id=user.tenant_id,
                action=AuditAction.LOGIN,
                resource_type="authentication",
                ip_address=client_ip,
                user_agent=request.headers.get("user-agent"),
                details={"status": "ip_not_whitelisted"},
                db=db
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: IP not whitelisted"
            )
        
        return user
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers first (for proxy/load balancer scenarios)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        return request.client.host if request.client else "unknown"


# Global middleware instance
security_middleware = SecurityMiddleware()


def require_auth(func: Callable) -> Callable:
    """Decorator to require authentication for a function."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # This decorator is used with FastAPI dependency injection
        # The actual authentication is handled by the authenticate_request dependency
        return await func(*args, **kwargs)
    return wrapper


def require_permission(
    project_id_param: str = "project_id",
    permission_type: PermissionType = PermissionType.READ
):
    """Decorator to require specific project permission."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract dependencies from kwargs
            user = kwargs.get("current_user")
            db = kwargs.get("db")
            
            if not user or not db:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Missing authentication dependencies"
                )
            
            # Get project_id from function parameters
            project_id = kwargs.get(project_id_param)
            if not project_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing required parameter: {project_id_param}"
                )
            
            # Check permission
            if not security_controller.check_project_permission(
                user.id, project_id, permission_type, db
            ):
                # Log unauthorized access attempt
                security_controller.log_user_action(
                    user_id=user.id,
                    tenant_id=user.tenant_id,
                    action=AuditAction.READ,
                    resource_type="project",
                    resource_id=project_id,
                    details={"permission_denied": permission_type.value},
                    db=db
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions for this project"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def require_role(allowed_roles: List[str]):
    """Decorator to require specific user roles."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user = kwargs.get("current_user")
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Missing authentication dependencies"
                )
            
            if user.role.value not in allowed_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient role permissions"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def audit_action(
    action: AuditAction,
    resource_type: str,
    resource_id_param: Optional[str] = None
):
    """Decorator to automatically log user actions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user = kwargs.get("current_user")
            db = kwargs.get("db")
            request = kwargs.get("request")
            
            if user and db:
                resource_id = None
                if resource_id_param:
                    resource_id = kwargs.get(resource_id_param)
                
                client_ip = None
                user_agent = None
                if request:
                    client_ip = security_middleware._get_client_ip(request)
                    user_agent = request.headers.get("user-agent")
                
                # Execute the function first
                result = await func(*args, **kwargs)
                
                # Log the action after successful execution
                security_controller.log_user_action(
                    user_id=user.id,
                    tenant_id=user.tenant_id,
                    action=action,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    ip_address=client_ip,
                    user_agent=user_agent,
                    db=db
                )
                
                return result
            else:
                return await func(*args, **kwargs)
        return wrapper
    return decorator


# FastAPI Dependencies

async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
    db: Session = Depends(get_db_session)
) -> UserModel:
    """FastAPI dependency to get the current authenticated user."""
    return await security_middleware.authenticate_request(request, credentials, db)


async def get_current_active_user(
    current_user: UserModel = Depends(get_current_user)
) -> UserModel:
    """FastAPI dependency to get the current active user."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


def require_project_permission(
    permission_type: PermissionType = PermissionType.READ
):
    """FastAPI dependency factory for project permissions."""
    def check_permission(
        project_id: str,
        current_user: UserModel = Depends(get_current_active_user),
        db: Session = Depends(get_db_session)
    ) -> bool:
        if not security_controller.check_project_permission(
            current_user.id, project_id, permission_type, db
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions for this project"
            )
        return True
    
    return check_permission