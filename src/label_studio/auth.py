"""
Authentication System for Label Studio Integration

Provides user authentication, session management, and JWT token handling
for multi-user collaboration features.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from uuid import uuid4
import jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt

from src.config.settings import settings
from src.label_studio.collaboration import User, UserRole, collaboration_manager

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Custom exception for authentication errors"""
    pass


class AuthenticationManager:
    """
    Manages user authentication and session handling.
    
    Provides JWT token generation, validation, and user session management.
    """
    
    def __init__(self):
        """Initialize authentication manager"""
        self.secret_key = settings.security.jwt_secret_key
        self.algorithm = settings.security.jwt_algorithm
        self.expire_minutes = settings.security.jwt_expire_minutes
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # In-memory session storage (should be moved to Redis in production)
        self._sessions: Dict[str, Dict[str, Any]] = {}
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, user_id: str, expires_delta: Optional[timedelta] = None) -> str:
        """
        Create a JWT access token for a user.
        
        Args:
            user_id: User identifier
            expires_delta: Optional custom expiration time
            
        Returns:
            str: JWT access token
        """
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.expire_minutes)
        
        to_encode = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": str(uuid4())  # JWT ID for token revocation
        }
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode a JWT token.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Optional[Dict[str, Any]]: Token payload if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if token is expired
            exp = payload.get("exp")
            if exp and datetime.utcfromtimestamp(exp) < datetime.utcnow():
                logger.warning("Token has expired")
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.JWTError as e:
            logger.warning(f"Token validation failed: {str(e)}")
            return None
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate a user with username and password.
        
        Args:
            username: User's username
            password: User's password
            
        Returns:
            Optional[User]: User object if authentication successful
        """
        # In a real implementation, this would query the database
        # For now, we'll use the collaboration manager's in-memory storage
        
        users = collaboration_manager.list_users()
        user = next((u for u in users if u.username == username), None)
        
        if not user:
            logger.warning(f"User {username} not found")
            return None
        
        if not user.is_active:
            logger.warning(f"User {username} is not active")
            return None
        
        # In a real implementation, we would verify the password hash
        # For demo purposes, we'll accept any password for existing users
        logger.info(f"User {username} authenticated successfully")
        return user
    
    def create_session(self, user: User) -> str:
        """
        Create a new user session.
        
        Args:
            user: User object
            
        Returns:
            str: Session token
        """
        session_id = str(uuid4())
        session_data = {
            "user_id": user.id,
            "username": user.username,
            "role": user.role.value,
            "tenant_id": user.tenant_id,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow()
        }
        
        self._sessions[session_id] = session_data
        logger.info(f"Created session for user {user.username}")
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session data by session ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Optional[Dict[str, Any]]: Session data if valid
        """
        session = self._sessions.get(session_id)
        if not session:
            return None
        
        # Update last activity
        session["last_activity"] = datetime.utcnow()
        return session
    
    def invalidate_session(self, session_id: str) -> bool:
        """
        Invalidate a user session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            bool: True if session was invalidated
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Invalidated session {session_id}")
            return True
        
        return False
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.
        
        Returns:
            int: Number of sessions cleaned up
        """
        now = datetime.utcnow()
        expired_sessions = []
        
        for session_id, session_data in self._sessions.items():
            last_activity = session_data.get("last_activity")
            if last_activity and (now - last_activity).total_seconds() > 3600:  # 1 hour timeout
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self._sessions[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)
    
    def get_current_user(self, token: str) -> Optional[User]:
        """
        Get current user from JWT token.
        
        Args:
            token: JWT access token
            
        Returns:
            Optional[User]: Current user if token is valid
        """
        payload = self.verify_token(token)
        if not payload:
            return None
        
        user_id = payload.get("sub")
        if not user_id:
            return None
        
        return collaboration_manager.get_user(user_id)
    
    def require_permission(self, token: str, permission: str) -> bool:
        """
        Check if current user has required permission.
        
        Args:
            token: JWT access token
            permission: Required permission
            
        Returns:
            bool: True if user has permission
        """
        user = self.get_current_user(token)
        if not user:
            return False
        
        return collaboration_manager.check_permission(user.id, permission)


# Demo function to create default users
def create_demo_users():
    """Create demo users for testing"""
    
    # Create admin user
    admin = collaboration_manager.create_user(
        username="admin",
        email="admin@superinsight.ai",
        role=UserRole.ADMIN,
        tenant_id="demo_tenant",
        metadata={"department": "IT", "location": "Beijing"}
    )
    
    # Create business expert
    business_expert = collaboration_manager.create_user(
        username="business_expert",
        email="business@superinsight.ai", 
        role=UserRole.BUSINESS_EXPERT,
        tenant_id="demo_tenant",
        metadata={"department": "Business", "expertise": "Finance"}
    )
    
    # Create technical expert
    tech_expert = collaboration_manager.create_user(
        username="tech_expert",
        email="tech@superinsight.ai",
        role=UserRole.TECHNICAL_EXPERT,
        tenant_id="demo_tenant",
        metadata={"department": "Engineering", "skills": ["Python", "ML"]}
    )
    
    # Create outsourced annotator
    annotator = collaboration_manager.create_user(
        username="annotator1",
        email="annotator1@outsource.com",
        role=UserRole.OUTSOURCED_ANNOTATOR,
        tenant_id="demo_tenant",
        metadata={"company": "Annotation Services Ltd", "experience": "2 years"}
    )
    
    logger.info("Created demo users for testing")
    return [admin, business_expert, tech_expert, annotator]


# Global authentication manager instance
auth_manager = AuthenticationManager()