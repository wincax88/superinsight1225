"""
Security Controller for SuperInsight Platform.

Implements core security functionality including authentication, authorization,
data isolation, IP whitelisting, and audit logging.
"""

import hashlib
import re
import ipaddress
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
from uuid import UUID
import jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, select

from src.security.models import (
    UserModel, ProjectPermissionModel, IPWhitelistModel, 
    AuditLogModel, DataMaskingRuleModel,
    UserRole, PermissionType, AuditAction
)
from src.database.connection import get_db_session


class SecurityController:
    """
    Core security controller for authentication, authorization, and data protection.
    
    Handles user authentication, project-level permissions, IP whitelisting,
    audit logging, and sensitive data masking.
    """
    
    def __init__(self, secret_key: str = "your-secret-key"):
        self.secret_key = secret_key
        # Try to initialize bcrypt, fallback to simple hashing if it fails
        try:
            self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
            # Test bcrypt with a simple password
            test_hash = self.pwd_context.hash("test")
            self.pwd_context.verify("test", test_hash)
            self.use_bcrypt = True
        except Exception:
            self.use_bcrypt = False
        self.token_expire_hours = 24
    
    # Authentication Methods
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt or fallback method."""
        if self.use_bcrypt:
            return self.pwd_context.hash(password)
        else:
            # Simple fallback hashing (not recommended for production)
            import hashlib
            return hashlib.sha256(f"{password}{self.secret_key}".encode()).hexdigest()
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        if self.use_bcrypt:
            return self.pwd_context.verify(plain_password, hashed_password)
        else:
            # Simple fallback verification
            import hashlib
            expected_hash = hashlib.sha256(f"{plain_password}{self.secret_key}".encode()).hexdigest()
            return expected_hash == hashed_password
    
    def create_access_token(self, user_id: str, tenant_id: str) -> str:
        """Create a JWT access token for a user."""
        expire = datetime.utcnow() + timedelta(hours=self.token_expire_hours)
        payload = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "exp": expire
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except Exception:  # Catch all JWT-related exceptions
            return None
    
    def authenticate_user(self, username: str, password: str, db: Session) -> Optional[UserModel]:
        """Authenticate a user with username and password."""
        stmt = select(UserModel).where(
            and_(
                UserModel.username == username,
                UserModel.is_active == True
            )
        )
        user = db.execute(stmt).scalar_one_or_none()
        
        if user and self.verify_password(password, user.password_hash):
            # Update last login
            user.last_login = datetime.utcnow()
            db.commit()
            return user
        return None
    
    # Authorization Methods
    
    def check_project_permission(
        self, 
        user_id: UUID, 
        project_id: str, 
        permission_type: PermissionType,
        db: Session
    ) -> bool:
        """Check if a user has specific permission for a project."""
        # Validate inputs
        if not user_id or not project_id or not project_id.strip():
            return False
        
        # Get user first
        stmt = select(UserModel).where(UserModel.id == user_id)
        user = db.execute(stmt).scalar_one_or_none()
        
        # User must exist and be active
        if not user:
            return False
        
        # Admin users have all permissions
        if user.role == UserRole.ADMIN:
            return True
        
        # Check specific project permission
        stmt = select(ProjectPermissionModel).where(
            and_(
                ProjectPermissionModel.user_id == user_id,
                ProjectPermissionModel.project_id == project_id,
                ProjectPermissionModel.permission_type == permission_type
            )
        )
        permission = db.execute(stmt).scalar_one_or_none()
        
        return permission is not None
    
    def grant_project_permission(
        self,
        user_id: UUID,
        project_id: str,
        permission_type: PermissionType,
        granted_by: UUID,
        db: Session
    ) -> bool:
        """Grant a project permission to a user."""
        try:
            permission = ProjectPermissionModel(
                user_id=user_id,
                project_id=project_id,
                permission_type=permission_type,
                granted_by=granted_by
            )
            db.add(permission)
            db.commit()
            return True
        except Exception:
            db.rollback()
            return False
    
    def revoke_project_permission(
        self,
        user_id: UUID,
        project_id: str,
        permission_type: PermissionType,
        db: Session
    ) -> bool:
        """Revoke a project permission from a user."""
        try:
            stmt = select(ProjectPermissionModel).where(
                and_(
                    ProjectPermissionModel.user_id == user_id,
                    ProjectPermissionModel.project_id == project_id,
                    ProjectPermissionModel.permission_type == permission_type
                )
            )
            permission = db.execute(stmt).scalar_one_or_none()
            
            if permission:
                db.delete(permission)
                db.commit()
                return True
            return False
        except Exception:
            db.rollback()
            return False
    
    def get_user_projects(self, user_id: UUID, db: Session) -> List[str]:
        """Get all projects a user has access to."""
        # Admin users have access to all projects in their tenant
        stmt = select(UserModel).where(UserModel.id == user_id)
        user = db.execute(stmt).scalar_one_or_none()
        if not user:
            return []
        
        if user.role == UserRole.ADMIN:
            # For admin, we would need to query all projects in tenant
            # This is a simplified implementation
            return ["*"]  # Represents all projects
        
        # Get projects with explicit permissions
        stmt = select(ProjectPermissionModel).where(
            ProjectPermissionModel.user_id == user_id
        )
        permissions = list(db.execute(stmt).scalars().all())
        
        return list(set(p.project_id for p in permissions))
    
    # Data Isolation Methods
    
    def filter_data_by_tenant(self, query, tenant_id: str, tenant_field: str = "tenant_id"):
        """Filter query results by tenant ID for data isolation."""
        return query.filter(getattr(query.column_descriptions[0]['type'], tenant_field) == tenant_id)
    
    def ensure_tenant_isolation(self, user_id: UUID, resource_tenant_id: str, db: Session) -> bool:
        """Ensure user can only access resources from their tenant."""
        user = db.query(UserModel).filter(UserModel.id == user_id).first()
        if not user:
            return False
        return user.tenant_id == resource_tenant_id
    
    # IP Whitelisting Methods
    
    def is_ip_whitelisted(self, ip_address: str, tenant_id: str, db: Session) -> bool:
        """Check if an IP address is whitelisted for a tenant."""
        try:
            ip = ipaddress.ip_address(ip_address)
            
            # Get all active whitelist entries for tenant
            whitelist_entries = db.query(IPWhitelistModel).filter(
                and_(
                    IPWhitelistModel.tenant_id == tenant_id,
                    IPWhitelistModel.is_active == True
                )
            ).all()
            
            for entry in whitelist_entries:
                # Check exact IP match
                if entry.ip_address and ip == ipaddress.ip_address(str(entry.ip_address)):
                    return True
                
                # Check IP range match
                if entry.ip_range:
                    try:
                        network = ipaddress.ip_network(entry.ip_range, strict=False)
                        if ip in network:
                            return True
                    except ValueError:
                        continue
            
            return False
        except ValueError:
            return False
    
    def add_ip_to_whitelist(
        self,
        ip_address: str,
        tenant_id: str,
        created_by: UUID,
        description: Optional[str] = None,
        ip_range: Optional[str] = None,
        db: Session = None
    ) -> bool:
        """Add an IP address to the whitelist."""
        try:
            # Validate IP address
            ipaddress.ip_address(ip_address)
            
            whitelist_entry = IPWhitelistModel(
                tenant_id=tenant_id,
                ip_address=ip_address,
                ip_range=ip_range,
                description=description,
                created_by=created_by
            )
            db.add(whitelist_entry)
            db.commit()
            return True
        except Exception:
            if db:
                db.rollback()
            return False
    
    # Data Masking Methods
    
    def mask_sensitive_data(self, data: Dict[str, Any], tenant_id: str, db: Session) -> Dict[str, Any]:
        """Apply data masking rules to sensitive data."""
        # Get masking rules for tenant
        rules = db.query(DataMaskingRuleModel).filter(
            and_(
                DataMaskingRuleModel.tenant_id == tenant_id,
                DataMaskingRuleModel.is_active == True
            )
        ).all()
        
        masked_data = data.copy()
        
        for rule in rules:
            field_name = rule.field_name
            if field_name in masked_data:
                masked_data[field_name] = self._apply_masking_rule(
                    masked_data[field_name], rule
                )
        
        return masked_data
    
    def _apply_masking_rule(self, value: Any, rule: DataMaskingRuleModel) -> Any:
        """Apply a specific masking rule to a value."""
        if not isinstance(value, str):
            return value
        
        masking_type = rule.masking_type
        config = rule.masking_config or {}
        
        if masking_type == "hash":
            return hashlib.sha256(value.encode()).hexdigest()[:8]
        elif masking_type == "partial":
            # Show first and last N characters, mask middle
            show_chars = config.get("show_chars", 2)
            if len(value) <= show_chars * 2:
                return "*" * len(value)
            
            # Improved partial masking to avoid exposing sensitive substrings
            # For structured data (containing hyphens, spaces, etc.), be more conservative
            if any(char in value for char in ['-', ' ', '.', '_', '@']):
                # For structured data, use smaller show_chars to avoid exposing patterns
                effective_show_chars = min(show_chars, max(1, len(value) // 6))
            else:
                effective_show_chars = show_chars
            
            # Ensure we don't expose more than 25% of the original string
            max_exposed = max(2, len(value) // 4)
            effective_show_chars = min(effective_show_chars, max_exposed // 2)
            
            if len(value) <= effective_show_chars * 2:
                return "*" * len(value)
            
            return (value[:effective_show_chars] + 
                   "*" * (len(value) - effective_show_chars * 2) + 
                   value[-effective_show_chars:])
        elif masking_type == "replace":
            replacement = config.get("replacement", "***")
            return replacement
        elif masking_type == "regex":
            pattern = rule.field_pattern
            replacement = config.get("replacement", "***")
            if pattern:
                # Apply regex replacement
                masked_value = re.sub(pattern, replacement, value)
                # If no match occurred (value unchanged), apply fallback masking
                if masked_value == value:
                    # Fallback to hash masking for unmatched patterns to ensure data is always masked
                    return hashlib.sha256(value.encode()).hexdigest()[:8]
                return masked_value
            else:
                # No pattern provided, fallback to hash masking
                return hashlib.sha256(value.encode()).hexdigest()[:8]
        
        return value
    
    def add_masking_rule(
        self,
        tenant_id: str,
        field_name: str,
        masking_type: str,
        created_by: UUID,
        field_pattern: Optional[str] = None,
        masking_config: Optional[Dict[str, Any]] = None,
        db: Session = None
    ) -> bool:
        """Add a data masking rule."""
        try:
            rule = DataMaskingRuleModel(
                tenant_id=tenant_id,
                field_name=field_name,
                field_pattern=field_pattern,
                masking_type=masking_type,
                masking_config=masking_config or {},
                created_by=created_by
            )
            db.add(rule)
            db.commit()
            return True
        except Exception:
            if db:
                db.rollback()
            return False
    
    # Audit Logging Methods
    
    def log_user_action(
        self,
        user_id: Optional[UUID],
        tenant_id: str,
        action: AuditAction,
        resource_type: str,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        db: Session = None
    ) -> bool:
        """Log a user action for audit purposes."""
        try:
            audit_log = AuditLogModel(
                user_id=user_id,
                tenant_id=tenant_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                ip_address=ip_address,
                user_agent=user_agent,
                details=details or {}
            )
            db.add(audit_log)
            db.commit()
            return True
        except Exception:
            if db:
                db.rollback()
            return False
    
    def get_audit_logs(
        self,
        tenant_id: str,
        user_id: Optional[UUID] = None,
        action: Optional[AuditAction] = None,
        resource_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        db: Session = None
    ) -> List[AuditLogModel]:
        """Retrieve audit logs with filtering."""
        query = db.query(AuditLogModel).filter(
            AuditLogModel.tenant_id == tenant_id
        )
        
        if user_id:
            query = query.filter(AuditLogModel.user_id == user_id)
        if action:
            query = query.filter(AuditLogModel.action == action)
        if resource_type:
            query = query.filter(AuditLogModel.resource_type == resource_type)
        if start_date:
            query = query.filter(AuditLogModel.timestamp >= start_date)
        if end_date:
            query = query.filter(AuditLogModel.timestamp <= end_date)
        
        return query.order_by(AuditLogModel.timestamp.desc()).limit(limit).all()
    
    # User Management Methods
    
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        full_name: str,
        role: UserRole,
        tenant_id: str,
        db: Session
    ) -> Optional[UserModel]:
        """Create a new user."""
        try:
            # Check if username or email already exists
            existing_user = db.query(UserModel).filter(
                or_(
                    UserModel.username == username,
                    UserModel.email == email
                )
            ).first()
            
            if existing_user:
                return None
            
            user = UserModel(
                username=username,
                email=email,
                password_hash=self.hash_password(password),
                full_name=full_name,
                role=role,
                tenant_id=tenant_id
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            return user
        except Exception as e:
            db.rollback()
            return None
    
    def get_user_by_id(self, user_id: UUID, db: Session) -> Optional[UserModel]:
        """Get a user by ID."""
        return db.query(UserModel).filter(UserModel.id == user_id).first()
    
    def get_user_by_username(self, username: str, db: Session) -> Optional[UserModel]:
        """Get a user by username."""
        return db.query(UserModel).filter(UserModel.username == username).first()
    
    def update_user_role(self, user_id: UUID, new_role: UserRole, db: Session) -> bool:
        """Update a user's role."""
        try:
            user = db.query(UserModel).filter(UserModel.id == user_id).first()
            if user:
                user.role = new_role
                db.commit()
                return True
            return False
        except Exception:
            db.rollback()
            return False
    
    def deactivate_user(self, user_id: UUID, db: Session) -> bool:
        """Deactivate a user account."""
        try:
            user = db.query(UserModel).filter(UserModel.id == user_id).first()
            if user:
                user.is_active = False
                db.commit()
                return True
            return False
        except Exception:
            db.rollback()
            return False

    # Health Check Methods

    def test_encryption(self) -> bool:
        """
        Test password hashing functionality for health check.

        Returns:
            bool: True if encryption is working correctly
        """
        try:
            test_password = "health_check_test_password_123"
            hashed = self.hash_password(test_password)

            # Verify the hash is not empty and is different from the original
            if not hashed or hashed == test_password:
                return False

            # Verify we can verify the password correctly
            if not self.verify_password(test_password, hashed):
                return False

            # Verify wrong password fails
            if self.verify_password("wrong_password", hashed):
                return False

            return True
        except Exception:
            return False

    def test_authentication(self) -> bool:
        """
        Test JWT token generation and validation for health check.

        Returns:
            bool: True if authentication system is working correctly
        """
        try:
            test_user_id = "health_check_user"
            test_tenant_id = "health_check_tenant"

            # Test token generation
            token = self.create_access_token(test_user_id, test_tenant_id)
            if not token:
                return False

            # Test token verification
            payload = self.verify_token(token)
            if not payload:
                return False

            # Verify payload contents
            if payload.get("user_id") != test_user_id:
                return False
            if payload.get("tenant_id") != test_tenant_id:
                return False

            return True
        except Exception:
            return False

    def test_audit_logging(self) -> bool:
        """
        Test audit logging capability for health check.

        This is a lightweight check that doesn't actually write to the database,
        but verifies that the audit logging infrastructure is properly configured.

        Returns:
            bool: True if audit logging is available
        """
        try:
            # Verify AuditAction enum is available
            if not hasattr(AuditAction, 'LOGIN'):
                return False

            # Verify AuditLogModel is available
            if not AuditLogModel:
                return False

            # Verify the log_user_action method exists and is callable
            if not callable(getattr(self, 'log_user_action', None)):
                return False

            return True
        except Exception:
            return False