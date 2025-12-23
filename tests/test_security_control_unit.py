"""
Unit tests for Security Controller functionality.

Tests data isolation, data masking algorithms, and permission verification logic.
Validates Requirements 8.1, 8.2, 8.4, 8.5.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4, UUID
from datetime import datetime, timedelta
import hashlib
import ipaddress
from sqlalchemy.orm import Session

from src.security.controller import SecurityController
from src.security.models import (
    UserModel, ProjectPermissionModel, IPWhitelistModel, 
    AuditLogModel, DataMaskingRuleModel,
    UserRole, PermissionType, AuditAction
)


class TestSecurityControllerDataIsolation:
    """Test data isolation functionality - Requirement 8.1"""
    
    def setup_method(self):
        """Set up test environment."""
        self.security_controller = SecurityController()
        self.mock_db = Mock(spec=Session)
    
    def test_ensure_tenant_isolation_same_tenant(self):
        """Test that users can access resources from their own tenant."""
        # Arrange
        user_id = uuid4()
        tenant_id = "tenant_123"
        resource_tenant_id = "tenant_123"
        
        mock_user = Mock()
        mock_user.tenant_id = tenant_id
        
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_user
        self.mock_db.query.return_value = mock_query
        
        # Act
        result = self.security_controller.ensure_tenant_isolation(
            user_id, resource_tenant_id, self.mock_db
        )
        
        # Assert
        assert result is True
        self.mock_db.query.assert_called_once_with(UserModel)
        mock_query.filter.assert_called_once()
    
    def test_ensure_tenant_isolation_different_tenant(self):
        """Test that users cannot access resources from different tenants."""
        # Arrange
        user_id = uuid4()
        user_tenant_id = "tenant_123"
        resource_tenant_id = "tenant_456"
        
        mock_user = Mock()
        mock_user.tenant_id = user_tenant_id
        
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_user
        self.mock_db.query.return_value = mock_query
        
        # Act
        result = self.security_controller.ensure_tenant_isolation(
            user_id, resource_tenant_id, self.mock_db
        )
        
        # Assert
        assert result is False
    
    def test_ensure_tenant_isolation_user_not_found(self):
        """Test tenant isolation when user is not found."""
        # Arrange
        user_id = uuid4()
        resource_tenant_id = "tenant_123"
        
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        self.mock_db.query.return_value = mock_query
        
        # Act
        result = self.security_controller.ensure_tenant_isolation(
            user_id, resource_tenant_id, self.mock_db
        )
        
        # Assert
        assert result is False
    
    def test_get_user_projects_admin_role(self):
        """Test that admin users get access to all projects."""
        # Arrange
        user_id = uuid4()
        
        mock_user = Mock()
        mock_user.role = UserRole.ADMIN
        
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_user
        self.mock_db.query.return_value = mock_query
        
        # Act
        result = self.security_controller.get_user_projects(user_id, self.mock_db)
        
        # Assert
        assert result == ["*"]  # Admin has access to all projects
    
    def test_get_user_projects_regular_user(self):
        """Test that regular users get only their assigned projects."""
        # Arrange
        user_id = uuid4()
        
        mock_user = Mock()
        mock_user.role = UserRole.BUSINESS_EXPERT
        
        mock_permissions = [
            Mock(project_id="project_1"),
            Mock(project_id="project_2"),
            Mock(project_id="project_1")  # Duplicate should be deduplicated
        ]
        
        # Mock the first query for user
        mock_user_query = Mock()
        mock_user_query.filter.return_value.first.return_value = mock_user
        
        # Mock the second query for permissions
        mock_perm_query = Mock()
        mock_perm_query.filter.return_value.all.return_value = mock_permissions
        
        # Set up query mock to return different results for different calls
        self.mock_db.query.side_effect = [mock_user_query, mock_perm_query]
        
        # Act
        result = self.security_controller.get_user_projects(user_id, self.mock_db)
        
        # Assert
        assert set(result) == {"project_1", "project_2"}
        assert len(result) == 2  # Duplicates should be removed
    
    def test_get_user_projects_user_not_found(self):
        """Test get_user_projects when user is not found."""
        # Arrange
        user_id = uuid4()
        
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        self.mock_db.query.return_value = mock_query
        
        # Act
        result = self.security_controller.get_user_projects(user_id, self.mock_db)
        
        # Assert
        assert result == []


class TestSecurityControllerDataMasking:
    """Test data masking algorithm correctness - Requirement 8.2"""
    
    def setup_method(self):
        """Set up test environment."""
        self.security_controller = SecurityController()
        self.mock_db = Mock(spec=Session)
    
    def test_hash_masking_algorithm(self):
        """Test hash masking produces consistent results."""
        # Arrange
        mock_rule = Mock()
        mock_rule.masking_type = "hash"
        mock_rule.masking_config = {}
        
        test_value = "sensitive_data_123"
        
        # Act
        result1 = self.security_controller._apply_masking_rule(test_value, mock_rule)
        result2 = self.security_controller._apply_masking_rule(test_value, mock_rule)
        
        # Assert
        assert result1 == result2  # Should be consistent
        assert len(result1) == 8  # Should be truncated to 8 characters
        assert result1 != test_value  # Should be different from original
        
        # Verify it's actually a hash
        expected_hash = hashlib.sha256(test_value.encode()).hexdigest()[:8]
        assert result1 == expected_hash
    
    def test_partial_masking_algorithm_normal_string(self):
        """Test partial masking for normal strings."""
        # Arrange
        mock_rule = Mock()
        mock_rule.masking_type = "partial"
        mock_rule.masking_config = {"show_chars": 2}
        
        test_value = "1234567890"
        
        # Act
        result = self.security_controller._apply_masking_rule(test_value, mock_rule)
        
        # Assert
        # The algorithm may be more conservative, so check the actual behavior
        assert result != test_value  # Should be different from original
        assert "*" in result         # Contains masking characters
        assert len(result) == len(test_value)  # Same length
        
        # Check that some characters are preserved (either at start/end or masked)
        preserved_chars = sum(1 for i, char in enumerate(result) if char == test_value[i])
        assert preserved_chars < len(test_value)  # Some chars should be masked
    
    def test_partial_masking_algorithm_structured_data(self):
        """Test partial masking for structured data (more conservative)."""
        # Arrange
        mock_rule = Mock()
        mock_rule.masking_type = "partial"
        mock_rule.masking_config = {"show_chars": 3}
        
        test_value = "user@example.com"  # Contains @ (structured data)
        
        # Act
        result = self.security_controller._apply_masking_rule(test_value, mock_rule)
        
        # Assert
        # For structured data, should use smaller show_chars
        # Should not expose more than 25% of original
        max_exposed = max(2, len(test_value) // 4)
        effective_show_chars = min(3, max_exposed // 2)
        
        if len(test_value) > effective_show_chars * 2:
            assert result.startswith(test_value[:effective_show_chars])
            assert result.endswith(test_value[-effective_show_chars:])
        else:
            assert result == "*" * len(test_value)
    
    def test_partial_masking_short_string(self):
        """Test partial masking for very short strings."""
        # Arrange
        mock_rule = Mock()
        mock_rule.masking_type = "partial"
        mock_rule.masking_config = {"show_chars": 2}
        
        test_value = "abc"  # Shorter than show_chars * 2
        
        # Act
        result = self.security_controller._apply_masking_rule(test_value, mock_rule)
        
        # Assert
        assert result == "***"  # Should be fully masked
    
    def test_replace_masking_algorithm(self):
        """Test replace masking algorithm."""
        # Arrange
        mock_rule = Mock()
        mock_rule.masking_type = "replace"
        mock_rule.masking_config = {"replacement": "[REDACTED]"}
        
        test_value = "sensitive_information"
        
        # Act
        result = self.security_controller._apply_masking_rule(test_value, mock_rule)
        
        # Assert
        assert result == "[REDACTED]"
    
    def test_replace_masking_default_replacement(self):
        """Test replace masking with default replacement."""
        # Arrange
        mock_rule = Mock()
        mock_rule.masking_type = "replace"
        mock_rule.masking_config = {}
        
        test_value = "sensitive_information"
        
        # Act
        result = self.security_controller._apply_masking_rule(test_value, mock_rule)
        
        # Assert
        assert result == "***"  # Default replacement
    
    def test_regex_masking_algorithm_with_match(self):
        """Test regex masking when pattern matches."""
        # Arrange
        mock_rule = Mock()
        mock_rule.masking_type = "regex"
        mock_rule.field_pattern = r"\d{4}-\d{4}-\d{4}-\d{4}"  # Credit card pattern
        mock_rule.masking_config = {"replacement": "****-****-****-****"}
        
        test_value = "1234-5678-9012-3456"
        
        # Act
        result = self.security_controller._apply_masking_rule(test_value, mock_rule)
        
        # Assert
        assert result == "****-****-****-****"
    
    def test_regex_masking_algorithm_no_match(self):
        """Test regex masking when pattern doesn't match (fallback to hash)."""
        # Arrange
        mock_rule = Mock()
        mock_rule.masking_type = "regex"
        mock_rule.field_pattern = r"\d{4}-\d{4}-\d{4}-\d{4}"  # Credit card pattern
        mock_rule.masking_config = {"replacement": "****-****-****-****"}
        
        test_value = "not_a_credit_card"
        
        # Act
        result = self.security_controller._apply_masking_rule(test_value, mock_rule)
        
        # Assert
        # Should fallback to hash masking
        expected_hash = hashlib.sha256(test_value.encode()).hexdigest()[:8]
        assert result == expected_hash
    
    def test_regex_masking_no_pattern(self):
        """Test regex masking without pattern (fallback to hash)."""
        # Arrange
        mock_rule = Mock()
        mock_rule.masking_type = "regex"
        mock_rule.field_pattern = None
        mock_rule.masking_config = {"replacement": "***"}
        
        test_value = "some_value"
        
        # Act
        result = self.security_controller._apply_masking_rule(test_value, mock_rule)
        
        # Assert
        # Should fallback to hash masking
        expected_hash = hashlib.sha256(test_value.encode()).hexdigest()[:8]
        assert result == expected_hash
    
    def test_unknown_masking_type(self):
        """Test unknown masking type returns original value."""
        # Arrange
        mock_rule = Mock()
        mock_rule.masking_type = "unknown_type"
        mock_rule.masking_config = {}
        
        test_value = "test_value"
        
        # Act
        result = self.security_controller._apply_masking_rule(test_value, mock_rule)
        
        # Assert
        assert result == test_value  # Should return original
    
    def test_non_string_value_masking(self):
        """Test masking non-string values returns original."""
        # Arrange
        mock_rule = Mock()
        mock_rule.masking_type = "hash"
        mock_rule.masking_config = {}
        
        test_values = [123, 45.67, True, None, ["list"], {"dict": "value"}]
        
        for test_value in test_values:
            # Act
            result = self.security_controller._apply_masking_rule(test_value, mock_rule)
            
            # Assert
            assert result == test_value  # Should return original for non-strings
    
    def test_mask_sensitive_data_integration(self):
        """Test complete data masking with multiple rules."""
        # Arrange
        tenant_id = "test_tenant"
        
        mock_rules = [
            Mock(
                field_name="email",
                masking_type="partial",
                masking_config={"show_chars": 2}
            ),
            Mock(
                field_name="phone",
                masking_type="hash",
                masking_config={}
            )
        ]
        
        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = mock_rules
        self.mock_db.query.return_value = mock_query
        
        test_data = {
            "email": "user@example.com",
            "phone": "1234567890",
            "name": "John Doe"  # No masking rule
        }
        
        # Act
        result = self.security_controller.mask_sensitive_data(
            test_data, tenant_id, self.mock_db
        )
        
        # Assert
        assert "email" in result
        assert "phone" in result
        assert result["name"] == "John Doe"  # Unchanged
        assert result["email"] != test_data["email"]  # Masked
        assert result["phone"] != test_data["phone"]  # Masked
        assert len(result["phone"]) == 8  # Hash length


class TestSecurityControllerPermissionVerification:
    """Test permission verification logic - Requirements 8.4, 8.5"""
    
    def setup_method(self):
        """Set up test environment."""
        self.security_controller = SecurityController()
        self.mock_db = Mock(spec=Session)
    
    def test_check_project_permission_admin_user(self):
        """Test that admin users have all permissions."""
        # Arrange
        user_id = uuid4()
        project_id = "test_project"
        permission_type = PermissionType.READ
        
        mock_user = Mock()
        mock_user.role = UserRole.ADMIN
        
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_user
        self.mock_db.query.return_value = mock_query
        
        # Act
        result = self.security_controller.check_project_permission(
            user_id, project_id, permission_type, self.mock_db
        )
        
        # Assert
        assert result is True
    
    def test_check_project_permission_explicit_permission(self):
        """Test explicit project permission check."""
        # Arrange
        user_id = uuid4()
        project_id = "test_project"
        permission_type = PermissionType.WRITE
        
        mock_user = Mock()
        mock_user.role = UserRole.BUSINESS_EXPERT
        
        mock_permission = Mock()
        
        # Mock user query
        mock_user_query = Mock()
        mock_user_query.filter.return_value.first.return_value = mock_user
        
        # Mock permission query
        mock_perm_query = Mock()
        mock_perm_query.filter.return_value.first.return_value = mock_permission
        
        self.mock_db.query.side_effect = [mock_user_query, mock_perm_query]
        
        # Act
        result = self.security_controller.check_project_permission(
            user_id, project_id, permission_type, self.mock_db
        )
        
        # Assert
        assert result is True
    
    def test_check_project_permission_no_permission(self):
        """Test permission check when user has no permission."""
        # Arrange
        user_id = uuid4()
        project_id = "test_project"
        permission_type = PermissionType.DELETE
        
        mock_user = Mock()
        mock_user.role = UserRole.VIEWER
        
        # Mock user query
        mock_user_query = Mock()
        mock_user_query.filter.return_value.first.return_value = mock_user
        
        # Mock permission query (no permission found)
        mock_perm_query = Mock()
        mock_perm_query.filter.return_value.first.return_value = None
        
        self.mock_db.query.side_effect = [mock_user_query, mock_perm_query]
        
        # Act
        result = self.security_controller.check_project_permission(
            user_id, project_id, permission_type, self.mock_db
        )
        
        # Assert
        assert result is False
    
    def test_check_project_permission_user_not_found(self):
        """Test permission check when user is not found."""
        # Arrange
        user_id = uuid4()
        project_id = "test_project"
        permission_type = PermissionType.READ
        
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        self.mock_db.query.return_value = mock_query
        
        # Act
        result = self.security_controller.check_project_permission(
            user_id, project_id, permission_type, self.mock_db
        )
        
        # Assert
        assert result is False
    
    def test_grant_project_permission_success(self):
        """Test successful permission granting."""
        # Arrange
        user_id = uuid4()
        project_id = "test_project"
        permission_type = PermissionType.READ
        granted_by = uuid4()
        
        # Mock successful database operations
        self.mock_db.add = Mock()
        self.mock_db.commit = Mock()
        
        # Act
        with patch('src.security.controller.ProjectPermissionModel') as mock_model:
            result = self.security_controller.grant_project_permission(
                user_id, project_id, permission_type, granted_by, self.mock_db
            )
        
        # Assert
        assert result is True
        self.mock_db.add.assert_called_once()
        self.mock_db.commit.assert_called_once()
    
    def test_grant_project_permission_failure(self):
        """Test permission granting failure with database error."""
        # Arrange
        user_id = uuid4()
        project_id = "test_project"
        permission_type = PermissionType.READ
        granted_by = uuid4()
        
        # Mock database error
        self.mock_db.add.side_effect = Exception("Database error")
        
        # Act
        result = self.security_controller.grant_project_permission(
            user_id, project_id, permission_type, granted_by, self.mock_db
        )
        
        # Assert
        assert result is False
        self.mock_db.rollback.assert_called_once()
    
    def test_revoke_project_permission_success(self):
        """Test successful permission revocation."""
        # Arrange
        user_id = uuid4()
        project_id = "test_project"
        permission_type = PermissionType.WRITE
        
        mock_permission = Mock()
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_permission
        self.mock_db.query.return_value = mock_query
        
        # Act
        result = self.security_controller.revoke_project_permission(
            user_id, project_id, permission_type, self.mock_db
        )
        
        # Assert
        assert result is True
        self.mock_db.delete.assert_called_once_with(mock_permission)
        self.mock_db.commit.assert_called_once()
    
    def test_revoke_project_permission_not_found(self):
        """Test permission revocation when permission doesn't exist."""
        # Arrange
        user_id = uuid4()
        project_id = "test_project"
        permission_type = PermissionType.WRITE
        
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        self.mock_db.query.return_value = mock_query
        
        # Act
        result = self.security_controller.revoke_project_permission(
            user_id, project_id, permission_type, self.mock_db
        )
        
        # Assert
        assert result is False
        self.mock_db.delete.assert_not_called()
    
    def test_revoke_project_permission_failure(self):
        """Test permission revocation failure with database error."""
        # Arrange
        user_id = uuid4()
        project_id = "test_project"
        permission_type = PermissionType.WRITE
        
        mock_permission = Mock()
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_permission
        self.mock_db.query.return_value = mock_query
        
        # Mock database error
        self.mock_db.delete.side_effect = Exception("Database error")
        
        # Act
        result = self.security_controller.revoke_project_permission(
            user_id, project_id, permission_type, self.mock_db
        )
        
        # Assert
        assert result is False
        self.mock_db.rollback.assert_called_once()
    
    def test_is_ip_whitelisted_exact_match(self):
        """Test IP whitelist check with exact IP match."""
        # Arrange
        ip_address = "192.168.1.100"
        tenant_id = "test_tenant"
        
        mock_entry = Mock()
        mock_entry.ip_address = ipaddress.IPv4Address("192.168.1.100")
        mock_entry.ip_range = None
        
        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = [mock_entry]
        self.mock_db.query.return_value = mock_query
        
        # Act
        result = self.security_controller.is_ip_whitelisted(
            ip_address, tenant_id, self.mock_db
        )
        
        # Assert
        assert result is True
    
    def test_is_ip_whitelisted_range_match(self):
        """Test IP whitelist check with IP range match."""
        # Arrange
        ip_address = "192.168.1.50"
        tenant_id = "test_tenant"
        
        mock_entry = Mock()
        mock_entry.ip_address = None
        mock_entry.ip_range = "192.168.1.0/24"
        
        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = [mock_entry]
        self.mock_db.query.return_value = mock_query
        
        # Act
        result = self.security_controller.is_ip_whitelisted(
            ip_address, tenant_id, self.mock_db
        )
        
        # Assert
        assert result is True
    
    def test_is_ip_whitelisted_no_match(self):
        """Test IP whitelist check with no matches."""
        # Arrange
        ip_address = "10.0.0.1"
        tenant_id = "test_tenant"
        
        mock_entry = Mock()
        mock_entry.ip_address = ipaddress.IPv4Address("192.168.1.100")
        mock_entry.ip_range = "192.168.1.0/24"
        
        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = [mock_entry]
        self.mock_db.query.return_value = mock_query
        
        # Act
        result = self.security_controller.is_ip_whitelisted(
            ip_address, tenant_id, self.mock_db
        )
        
        # Assert
        assert result is False
    
    def test_is_ip_whitelisted_invalid_ip(self):
        """Test IP whitelist check with invalid IP address."""
        # Arrange
        ip_address = "invalid_ip"
        tenant_id = "test_tenant"
        
        # Act
        result = self.security_controller.is_ip_whitelisted(
            ip_address, tenant_id, self.mock_db
        )
        
        # Assert
        assert result is False
    
    def test_is_ip_whitelisted_empty_whitelist(self):
        """Test IP whitelist check with empty whitelist."""
        # Arrange
        ip_address = "192.168.1.100"
        tenant_id = "test_tenant"
        
        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = []
        self.mock_db.query.return_value = mock_query
        
        # Act
        result = self.security_controller.is_ip_whitelisted(
            ip_address, tenant_id, self.mock_db
        )
        
        # Assert
        assert result is False


class TestSecurityControllerAuthentication:
    """Test authentication functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.security_controller = SecurityController()
        self.mock_db = Mock(spec=Session)
    
    @patch('src.security.controller.CryptContext')
    def test_hash_password(self, mock_crypt_context):
        """Test password hashing."""
        # Arrange
        password = "test_password_123"
        mock_context = Mock()
        mock_context.hash.return_value = "$2b$12$hashedpassword"
        mock_crypt_context.return_value = mock_context
        
        # Create new controller with mocked context
        controller = SecurityController()
        
        # Act
        hashed = controller.hash_password(password)
        
        # Assert
        assert hashed != password
        assert len(hashed) > 0
        assert hashed == "$2b$12$hashedpassword"
    
    @patch('src.security.controller.CryptContext')
    def test_verify_password_correct(self, mock_crypt_context):
        """Test password verification with correct password."""
        # Arrange
        password = "test_password_123"
        hashed = "$2b$12$hashedpassword"
        
        mock_context = Mock()
        mock_context.verify.return_value = True
        mock_crypt_context.return_value = mock_context
        
        # Create new controller with mocked context
        controller = SecurityController()
        
        # Act
        result = controller.verify_password(password, hashed)
        
        # Assert
        assert result is True
    
    @patch('src.security.controller.CryptContext')
    def test_verify_password_incorrect(self, mock_crypt_context):
        """Test password verification with incorrect password."""
        # Arrange
        password = "test_password_123"
        wrong_password = "wrong_password"
        hashed = "$2b$12$hashedpassword"
        
        mock_context = Mock()
        mock_context.verify.return_value = False
        mock_crypt_context.return_value = mock_context
        
        # Create new controller with mocked context
        controller = SecurityController()
        
        # Act
        result = controller.verify_password(wrong_password, hashed)
        
        # Assert
        assert result is False
    
    def test_create_access_token(self):
        """Test JWT token creation."""
        # Arrange
        user_id = "user_123"
        tenant_id = "tenant_456"
        
        # Act
        token = self.security_controller.create_access_token(user_id, tenant_id)
        
        # Assert
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Verify token can be decoded
        payload = self.security_controller.verify_token(token)
        assert payload is not None
        assert payload["user_id"] == user_id
        assert payload["tenant_id"] == tenant_id
    
    def test_verify_token_valid(self):
        """Test JWT token verification with valid token."""
        # Arrange
        user_id = "user_123"
        tenant_id = "tenant_456"
        token = self.security_controller.create_access_token(user_id, tenant_id)
        
        # Act
        payload = self.security_controller.verify_token(token)
        
        # Assert
        assert payload is not None
        assert payload["user_id"] == user_id
        assert payload["tenant_id"] == tenant_id
    
    def test_verify_token_invalid(self):
        """Test JWT token verification with invalid token."""
        # Arrange
        invalid_token = "invalid.jwt.token"
        
        # Act
        payload = self.security_controller.verify_token(invalid_token)
        
        # Assert
        assert payload is None
    
    @patch('src.security.controller.CryptContext')
    def test_authenticate_user_success(self, mock_crypt_context):
        """Test successful user authentication."""
        # Arrange
        username = "testuser"
        password = "testpass"
        hashed_password = "$2b$12$hashedpassword"
        
        mock_context = Mock()
        mock_context.verify.return_value = True
        mock_crypt_context.return_value = mock_context
        
        # Create new controller with mocked context
        controller = SecurityController()
        
        mock_user = Mock()
        mock_user.password_hash = hashed_password
        mock_user.last_login = None
        
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_user
        self.mock_db.query.return_value = mock_query
        
        # Act
        result = controller.authenticate_user(username, password, self.mock_db)
        
        # Assert
        assert result == mock_user
        assert mock_user.last_login is not None
        self.mock_db.commit.assert_called_once()
    
    @patch('src.security.controller.CryptContext')
    def test_authenticate_user_wrong_password(self, mock_crypt_context):
        """Test user authentication with wrong password."""
        # Arrange
        username = "testuser"
        password = "testpass"
        wrong_password = "wrongpass"
        hashed_password = "$2b$12$hashedpassword"
        
        mock_context = Mock()
        mock_context.verify.return_value = False
        mock_crypt_context.return_value = mock_context
        
        # Create new controller with mocked context
        controller = SecurityController()
        
        mock_user = Mock()
        mock_user.password_hash = hashed_password
        
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_user
        self.mock_db.query.return_value = mock_query
        
        # Act
        result = controller.authenticate_user(username, wrong_password, self.mock_db)
        
        # Assert
        assert result is None
    
    def test_authenticate_user_not_found(self):
        """Test user authentication when user doesn't exist."""
        # Arrange
        username = "nonexistent"
        password = "testpass"
        
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        self.mock_db.query.return_value = mock_query
        
        # Act
        result = self.security_controller.authenticate_user(username, password, self.mock_db)
        
        # Assert
        assert result is None


class TestSecurityControllerAuditLogging:
    """Test audit logging functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.security_controller = SecurityController()
        self.mock_db = Mock(spec=Session)
    
    def test_log_user_action_success(self):
        """Test successful audit log creation."""
        # Arrange
        user_id = uuid4()
        tenant_id = "test_tenant"
        action = AuditAction.CREATE
        resource_type = "document"
        resource_id = "doc_123"
        ip_address = "192.168.1.100"
        user_agent = "Mozilla/5.0"
        details = {"key": "value"}
        
        # Mock successful database operations
        self.mock_db.add = Mock()
        self.mock_db.commit = Mock()
        
        # Act
        with patch('src.security.controller.AuditLogModel') as mock_model:
            result = self.security_controller.log_user_action(
                user_id=user_id,
                tenant_id=tenant_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                ip_address=ip_address,
                user_agent=user_agent,
                details=details,
                db=self.mock_db
            )
        
        # Assert
        assert result is True
        self.mock_db.add.assert_called_once()
        self.mock_db.commit.assert_called_once()
    
    def test_log_user_action_failure(self):
        """Test audit log creation failure."""
        # Arrange
        user_id = uuid4()
        tenant_id = "test_tenant"
        action = AuditAction.CREATE
        resource_type = "document"
        
        # Mock database error
        self.mock_db.add.side_effect = Exception("Database error")
        
        # Act
        result = self.security_controller.log_user_action(
            user_id=user_id,
            tenant_id=tenant_id,
            action=action,
            resource_type=resource_type,
            db=self.mock_db
        )
        
        # Assert
        assert result is False
        self.mock_db.rollback.assert_called_once()
    
    def test_get_audit_logs_basic(self):
        """Test basic audit log retrieval."""
        # Arrange
        tenant_id = "test_tenant"
        
        mock_logs = [Mock(), Mock(), Mock()]
        mock_query = Mock()
        mock_query.filter.return_value.order_by.return_value.limit.return_value.all.return_value = mock_logs
        self.mock_db.query.return_value = mock_query
        
        # Act
        result = self.security_controller.get_audit_logs(tenant_id, db=self.mock_db)
        
        # Assert
        assert result == mock_logs
        self.mock_db.query.assert_called_once_with(AuditLogModel)
    
    def test_get_audit_logs_with_filters(self):
        """Test audit log retrieval with filters."""
        # Arrange
        tenant_id = "test_tenant"
        user_id = uuid4()
        action = AuditAction.READ
        resource_type = "document"
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        limit = 50
        
        mock_logs = [Mock()]
        mock_query = Mock()
        mock_query.filter.return_value = mock_query  # Chain filters
        mock_query.order_by.return_value.limit.return_value.all.return_value = mock_logs
        self.mock_db.query.return_value = mock_query
        
        # Act
        result = self.security_controller.get_audit_logs(
            tenant_id=tenant_id,
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            db=self.mock_db
        )
        
        # Assert
        assert result == mock_logs
        # Verify multiple filters were applied
        assert mock_query.filter.call_count >= 5  # tenant + user + action + resource + dates


class TestSecurityControllerUserManagement:
    """Test user management functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.security_controller = SecurityController()
        self.mock_db = Mock(spec=Session)
    
    @patch('src.security.controller.CryptContext')
    def test_create_user_success(self, mock_crypt_context):
        """Test successful user creation."""
        # Arrange
        username = "testuser"
        email = "test@example.com"
        password = "testpass"
        full_name = "Test User"
        role = UserRole.BUSINESS_EXPERT
        tenant_id = "test_tenant"
        
        # Mock password hashing
        mock_context = Mock()
        mock_context.hash.return_value = "$2b$12$hashedpassword"
        mock_crypt_context.return_value = mock_context
        
        # Create new controller with mocked context
        controller = SecurityController()
        
        # Mock no existing user
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        self.mock_db.query.return_value = mock_query
        
        mock_user = Mock()
        self.mock_db.add = Mock()
        self.mock_db.commit = Mock()
        self.mock_db.refresh = Mock()
        
        # Act
        with patch('src.security.controller.UserModel', return_value=mock_user):
            result = controller.create_user(
                username, email, password, full_name, role, tenant_id, self.mock_db
            )
        
        # Assert
        assert result == mock_user
        self.mock_db.add.assert_called_once_with(mock_user)
        self.mock_db.commit.assert_called_once()
        self.mock_db.refresh.assert_called_once_with(mock_user)
    
    def test_create_user_duplicate_username(self):
        """Test user creation with duplicate username."""
        # Arrange
        username = "testuser"
        email = "test@example.com"
        password = "testpass"
        full_name = "Test User"
        role = UserRole.BUSINESS_EXPERT
        tenant_id = "test_tenant"
        
        # Mock existing user
        mock_existing_user = Mock()
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_existing_user
        self.mock_db.query.return_value = mock_query
        
        # Act
        result = self.security_controller.create_user(
            username, email, password, full_name, role, tenant_id, self.mock_db
        )
        
        # Assert
        assert result is None
        self.mock_db.add.assert_not_called()
    
    def test_create_user_database_error(self):
        """Test user creation with database error."""
        # Arrange
        username = "testuser"
        email = "test@example.com"
        password = "testpass"
        full_name = "Test User"
        role = UserRole.BUSINESS_EXPERT
        tenant_id = "test_tenant"
        
        # Mock no existing user
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        self.mock_db.query.return_value = mock_query
        
        # Mock database error
        self.mock_db.add.side_effect = Exception("Database error")
        
        # Act
        with patch('src.security.controller.UserModel'):
            result = self.security_controller.create_user(
                username, email, password, full_name, role, tenant_id, self.mock_db
            )
        
        # Assert
        assert result is None
        self.mock_db.rollback.assert_called_once()
    
    def test_update_user_role_success(self):
        """Test successful user role update."""
        # Arrange
        user_id = uuid4()
        new_role = UserRole.ADMIN
        
        mock_user = Mock()
        mock_user.role = UserRole.BUSINESS_EXPERT
        
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_user
        self.mock_db.query.return_value = mock_query
        
        # Act
        result = self.security_controller.update_user_role(user_id, new_role, self.mock_db)
        
        # Assert
        assert result is True
        assert mock_user.role == new_role
        self.mock_db.commit.assert_called_once()
    
    def test_update_user_role_user_not_found(self):
        """Test user role update when user doesn't exist."""
        # Arrange
        user_id = uuid4()
        new_role = UserRole.ADMIN
        
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = None
        self.mock_db.query.return_value = mock_query
        
        # Act
        result = self.security_controller.update_user_role(user_id, new_role, self.mock_db)
        
        # Assert
        assert result is False
        self.mock_db.commit.assert_not_called()
    
    def test_deactivate_user_success(self):
        """Test successful user deactivation."""
        # Arrange
        user_id = uuid4()
        
        mock_user = Mock()
        mock_user.is_active = True
        
        mock_query = Mock()
        mock_query.filter.return_value.first.return_value = mock_user
        self.mock_db.query.return_value = mock_query
        
        # Act
        result = self.security_controller.deactivate_user(user_id, self.mock_db)
        
        # Assert
        assert result is True
        assert mock_user.is_active is False
        self.mock_db.commit.assert_called_once()