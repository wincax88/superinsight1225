"""
Property-based tests for IP whitelist access control in SuperInsight Platform.

Tests the IP whitelist access control property to ensure that non-whitelisted IPs
cannot access the system.
"""

import pytest
import ipaddress
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from uuid import uuid4, UUID
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings, assume
from hypothesis.strategies import composite

from src.security.controller import SecurityController
from src.security.models import IPWhitelistModel, UserRole
from src.database.connection import get_db_session


# Test data generators
@composite
def ip_address_strategy(draw):
    """Generate valid IP addresses for testing."""
    # Generate IPv4 addresses
    octets = [draw(st.integers(min_value=0, max_value=255)) for _ in range(4)]
    
    # Avoid reserved ranges for more realistic testing
    # Skip 0.x.x.x, 127.x.x.x, 224.x.x.x-255.x.x.x
    assume(octets[0] not in [0, 127] and octets[0] < 224)
    
    # Skip 10.x.x.x, 172.16-31.x.x, 192.168.x.x for some variety
    if octets[0] == 10:
        assume(draw(st.booleans()))  # Sometimes allow, sometimes skip
    elif octets[0] == 172 and 16 <= octets[1] <= 31:
        assume(draw(st.booleans()))
    elif octets[0] == 192 and octets[1] == 168:
        assume(draw(st.booleans()))
    
    ip_str = ".".join(map(str, octets))
    
    return {
        "ip": ip_str,
        "octets": octets,
        "is_private": ipaddress.ip_address(ip_str).is_private
    }


@composite
def ip_range_strategy(draw):
    """Generate IP ranges in CIDR notation."""
    base_ip = draw(ip_address_strategy())
    
    # Generate CIDR prefix length (8-30 for meaningful ranges)
    prefix_length = draw(st.integers(min_value=8, max_value=30))
    
    cidr = f"{base_ip['ip']}/{prefix_length}"
    
    try:
        network = ipaddress.ip_network(cidr, strict=False)
        return {
            "cidr": str(network),
            "network": network,
            "base_ip": base_ip["ip"],
            "prefix_length": prefix_length
        }
    except ValueError:
        # Fallback to a simple /24 network
        base_octets = base_ip["octets"]
        network_ip = f"{base_octets[0]}.{base_octets[1]}.{base_octets[2]}.0/24"
        network = ipaddress.ip_network(network_ip)
        return {
            "cidr": str(network),
            "network": network,
            "base_ip": base_ip["ip"],
            "prefix_length": 24
        }


@composite
def whitelist_entry_strategy(draw):
    """Generate IP whitelist entries."""
    tenant_id = f"tenant_{draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))}"
    
    # Choose between single IP or IP range
    entry_type = draw(st.sampled_from(["single_ip", "ip_range", "both"]))
    
    if entry_type == "single_ip":
        ip_data = draw(ip_address_strategy())
        return {
            "tenant_id": tenant_id,
            "ip_address": ip_data["ip"],
            "ip_range": None,
            "description": f"Single IP for {tenant_id}",
            "entry_type": "single_ip"
        }
    elif entry_type == "ip_range":
        range_data = draw(ip_range_strategy())
        # For range-only entries, use the network address as the IP
        network_ip = str(range_data["network"].network_address)
        return {
            "tenant_id": tenant_id,
            "ip_address": network_ip,
            "ip_range": range_data["cidr"],
            "description": f"IP range for {tenant_id}",
            "entry_type": "ip_range"
        }
    else:  # both
        ip_data = draw(ip_address_strategy())
        range_data = draw(ip_range_strategy())
        return {
            "tenant_id": tenant_id,
            "ip_address": ip_data["ip"],
            "ip_range": range_data["cidr"],
            "description": f"IP and range for {tenant_id}",
            "entry_type": "both"
        }


@composite
def multi_tenant_whitelist_strategy(draw):
    """Generate whitelist data for multiple tenants."""
    num_tenants = draw(st.integers(min_value=1, max_value=5))
    tenants_data = {}
    
    for i in range(num_tenants):
        tenant_id = f"tenant_{i}_{draw(st.text(min_size=1, max_size=15, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))}"
        
        # Generate 1-5 whitelist entries per tenant
        num_entries = draw(st.integers(min_value=1, max_value=5))
        whitelist_entries = []
        
        for j in range(num_entries):
            entry = draw(whitelist_entry_strategy())
            entry["tenant_id"] = tenant_id  # Override to ensure consistency
            whitelist_entries.append(entry)
        
        tenants_data[tenant_id] = whitelist_entries
    
    return tenants_data


@composite
def access_attempt_strategy(draw):
    """Generate access attempts from various IP addresses."""
    ip_data = draw(ip_address_strategy())
    
    return {
        "source_ip": ip_data["ip"],
        "user_agent": draw(st.text(min_size=10, max_size=100)),
        "timestamp": datetime.now(),
        "request_path": draw(st.sampled_from(["/api/tasks", "/api/annotations", "/api/export", "/login", "/dashboard"])),
        "is_private": ip_data["is_private"]
    }


class TestIPWhitelistAccessControl:
    """
    Property-based tests for IP whitelist access control.
    
    Validates Requirement 8.4:
    - THE Security_Controller SHALL 支持 IP 白名单访问控制
    """
    
    def setup_method(self):
        """Set up test environment."""
        self.security_controller = SecurityController()
        self.mock_db_whitelist = {}  # In-memory storage for whitelist entries
        self.mock_db_session = Mock()
    
    def teardown_method(self):
        """Clean up test environment."""
        self.mock_db_whitelist.clear()
    
    def _create_mock_whitelist_entry(self, entry_data: Dict[str, Any]) -> IPWhitelistModel:
        """Create a mock whitelist entry."""
        entry_id = str(uuid4())
        
        # Create a mock whitelist entry object
        mock_entry = Mock(spec=IPWhitelistModel)
        mock_entry.id = entry_id
        mock_entry.tenant_id = entry_data["tenant_id"]
        mock_entry.ip_address = entry_data["ip_address"]
        mock_entry.ip_range = entry_data.get("ip_range")
        mock_entry.description = entry_data.get("description", "")
        mock_entry.is_active = True
        mock_entry.created_by = uuid4()
        mock_entry.created_at = datetime.now()
        
        # Store in mock database
        tenant_id = entry_data["tenant_id"]
        if tenant_id not in self.mock_db_whitelist:
            self.mock_db_whitelist[tenant_id] = []
        self.mock_db_whitelist[tenant_id].append(mock_entry)
        
        return mock_entry
    
    def _mock_query_whitelist_entries(self, tenant_id: str) -> List[IPWhitelistModel]:
        """Mock database query for whitelist entries."""
        return self.mock_db_whitelist.get(tenant_id, [])
    
    def _generate_non_whitelisted_ip(self, whitelisted_entries: List[Dict[str, Any]]) -> str:
        """Generate an IP that is definitely not in the whitelist."""
        # Use a simple approach: generate IPs from a different network
        # Use 203.0.113.x (TEST-NET-3) which is unlikely to be in whitelists
        for i in range(1, 255):
            candidate_ip = f"203.0.113.{i}"
            
            # Check if this IP would be whitelisted
            is_whitelisted = False
            for entry in whitelisted_entries:
                try:
                    # Check exact IP match
                    if entry.get("ip_address") == candidate_ip:
                        is_whitelisted = True
                        break
                    
                    # Check IP range match
                    if entry.get("ip_range"):
                        network = ipaddress.ip_network(entry["ip_range"], strict=False)
                        if ipaddress.ip_address(candidate_ip) in network:
                            is_whitelisted = True
                            break
                except (ValueError, TypeError):
                    continue
            
            if not is_whitelisted:
                return candidate_ip
        
        # Fallback: use a definitely non-whitelisted IP
        return "198.51.100.1"  # TEST-NET-2
    
    @given(whitelist_entry_strategy(), access_attempt_strategy())
    @settings(max_examples=100, deadline=30000)
    def test_single_ip_whitelist_access_control_property(self, whitelist_entry, access_attempt):
        """
        **Feature: superinsight-platform, Property 14: IP 白名单访问控制**
        **Validates: Requirements 8.4**
        
        For any IP whitelist configuration, only whitelisted IPs should be allowed access,
        and non-whitelisted IPs should be denied access.
        """
        # Clear mock database
        self.mock_db_whitelist.clear()
        
        tenant_id = whitelist_entry["tenant_id"]
        source_ip = access_attempt["source_ip"]
        
        # Create whitelist entry
        mock_entry = self._create_mock_whitelist_entry(whitelist_entry)
        
        # Mock the database query
        with patch.object(self.mock_db_session, 'query') as mock_query:
            mock_query.return_value.filter.return_value.all.return_value = [mock_entry]
            
            # Test IP whitelist check
            is_allowed = self.security_controller.is_ip_whitelisted(
                ip_address=source_ip,
                tenant_id=tenant_id,
                db=self.mock_db_session
            )
            
            # Determine if IP should be whitelisted based on entry
            should_be_allowed = False
            
            try:
                source_ip_obj = ipaddress.ip_address(source_ip)
                
                # Check exact IP match
                if whitelist_entry["ip_address"] == source_ip:
                    should_be_allowed = True
                
                # Check IP range match
                if whitelist_entry.get("ip_range") and not should_be_allowed:
                    try:
                        network = ipaddress.ip_network(whitelist_entry["ip_range"], strict=False)
                        if source_ip_obj in network:
                            should_be_allowed = True
                    except ValueError:
                        pass
                
            except ValueError:
                # Invalid IP address should not be allowed
                should_be_allowed = False
            
            # Assert the access control property
            assert is_allowed == should_be_allowed, (
                f"IP whitelist access control failed for IP {source_ip} in tenant {tenant_id}. "
                f"Expected: {should_be_allowed}, Got: {is_allowed}. "
                f"Whitelist entry: IP={whitelist_entry['ip_address']}, Range={whitelist_entry.get('ip_range')}"
            )
    
    @given(multi_tenant_whitelist_strategy())
    @settings(max_examples=50, deadline=30000)
    def test_multi_tenant_ip_whitelist_isolation_property(self, tenants_data):
        """
        **Feature: superinsight-platform, Property 14: IP 白名单访问控制 (Multi-Tenant)**
        **Validates: Requirements 8.4**
        
        For any multi-tenant environment, each tenant's IP whitelist should be
        isolated and not affect other tenants' access control.
        """
        # Clear mock database
        self.mock_db_whitelist.clear()
        
        # Skip if no tenants
        assume(len(tenants_data) > 0)
        
        # Create whitelist entries for all tenants
        all_entries = {}
        for tenant_id, entries in tenants_data.items():
            all_entries[tenant_id] = []
            for entry_data in entries:
                mock_entry = self._create_mock_whitelist_entry(entry_data)
                all_entries[tenant_id].append(mock_entry)
        
        # Test each tenant's whitelist isolation
        for tenant_id, tenant_entries in all_entries.items():
            # Mock the database query for this tenant
            with patch.object(self.mock_db_session, 'query') as mock_query:
                mock_query.return_value.filter.return_value.all.return_value = tenant_entries
                
                # Test with IPs from this tenant's whitelist
                for entry in tenant_entries:
                    test_ip = entry.ip_address
                    
                    is_allowed = self.security_controller.is_ip_whitelisted(
                        ip_address=test_ip,
                        tenant_id=tenant_id,
                        db=self.mock_db_session
                    )
                    
                    # Should be allowed for this tenant
                    assert is_allowed, (
                        f"Tenant {tenant_id} should allow its whitelisted IP {test_ip}"
                    )
                
                # Test with IPs from other tenants' whitelists
                for other_tenant_id, other_entries in all_entries.items():
                    if other_tenant_id != tenant_id:
                        for other_entry in other_entries:
                            other_ip = other_entry.ip_address
                            
                            # Skip if the IP happens to be the same (unlikely but possible)
                            if any(entry.ip_address == other_ip for entry in tenant_entries):
                                continue
                            
                            is_allowed = self.security_controller.is_ip_whitelisted(
                                ip_address=other_ip,
                                tenant_id=tenant_id,
                                db=self.mock_db_session
                            )
                            
                            # Should NOT be allowed for this tenant (unless coincidentally in range)
                            # We need to check if it's actually in this tenant's ranges
                            should_be_allowed = False
                            try:
                                other_ip_obj = ipaddress.ip_address(other_ip)
                                for entry in tenant_entries:
                                    if entry.ip_range:
                                        try:
                                            network = ipaddress.ip_network(entry.ip_range, strict=False)
                                            if other_ip_obj in network:
                                                should_be_allowed = True
                                                break
                                        except ValueError:
                                            continue
                            except ValueError:
                                pass
                            
                            if not should_be_allowed:
                                assert not is_allowed, (
                                    f"Tenant {tenant_id} should NOT allow IP {other_ip} "
                                    f"from tenant {other_tenant_id}'s whitelist"
                                )
    
    @given(whitelist_entry_strategy())
    @settings(max_examples=50, deadline=30000)
    def test_non_whitelisted_ip_denial_property(self, whitelist_entry):
        """
        **Feature: superinsight-platform, Property 14: IP 白名单访问控制 (Denial)**
        **Validates: Requirements 8.4**
        
        For any IP whitelist configuration, IPs that are definitely not in the
        whitelist should be denied access.
        """
        # Clear mock database
        self.mock_db_whitelist.clear()
        
        tenant_id = whitelist_entry["tenant_id"]
        
        # Create whitelist entry
        mock_entry = self._create_mock_whitelist_entry(whitelist_entry)
        
        # Generate a non-whitelisted IP
        non_whitelisted_ip = self._generate_non_whitelisted_ip([whitelist_entry])
        
        # Mock the database query
        with patch.object(self.mock_db_session, 'query') as mock_query:
            mock_query.return_value.filter.return_value.all.return_value = [mock_entry]
            
            # Test that non-whitelisted IP is denied
            is_allowed = self.security_controller.is_ip_whitelisted(
                ip_address=non_whitelisted_ip,
                tenant_id=tenant_id,
                db=self.mock_db_session
            )
            
            # Should be denied
            assert not is_allowed, (
                f"Non-whitelisted IP {non_whitelisted_ip} should be denied access "
                f"for tenant {tenant_id} with whitelist entry: "
                f"IP={whitelist_entry['ip_address']}, Range={whitelist_entry.get('ip_range')}"
            )
    
    @given(st.lists(whitelist_entry_strategy(), min_size=1, max_size=5))
    @settings(max_examples=30, deadline=30000)
    def test_multiple_whitelist_entries_property(self, whitelist_entries):
        """
        **Feature: superinsight-platform, Property 14: IP 白名单访问控制 (Multiple Entries)**
        **Validates: Requirements 8.4**
        
        For any tenant with multiple whitelist entries, an IP should be allowed
        if it matches ANY of the entries, and denied if it matches NONE.
        """
        # Clear mock database
        self.mock_db_whitelist.clear()
        
        # Use the same tenant for all entries
        tenant_id = "multi_entry_tenant"
        
        # Update all entries to use the same tenant
        for entry in whitelist_entries:
            entry["tenant_id"] = tenant_id
        
        # Create mock entries
        mock_entries = []
        for entry_data in whitelist_entries:
            mock_entry = self._create_mock_whitelist_entry(entry_data)
            mock_entries.append(mock_entry)
        
        # Mock the database query
        with patch.object(self.mock_db_session, 'query') as mock_query:
            mock_query.return_value.filter.return_value.all.return_value = mock_entries
            
            # Test IPs from each whitelist entry
            for entry_data in whitelist_entries:
                test_ip = entry_data["ip_address"]
                
                is_allowed = self.security_controller.is_ip_whitelisted(
                    ip_address=test_ip,
                    tenant_id=tenant_id,
                    db=self.mock_db_session
                )
                
                # Should be allowed since it's in at least one entry
                assert is_allowed, (
                    f"IP {test_ip} should be allowed as it's in the whitelist entries"
                )
            
            # Test a definitely non-whitelisted IP
            non_whitelisted_ip = self._generate_non_whitelisted_ip(whitelist_entries)
            
            is_allowed = self.security_controller.is_ip_whitelisted(
                ip_address=non_whitelisted_ip,
                tenant_id=tenant_id,
                db=self.mock_db_session
            )
            
            # Should be denied
            assert not is_allowed, (
                f"Non-whitelisted IP {non_whitelisted_ip} should be denied access "
                f"even with multiple whitelist entries"
            )
    
    @given(ip_range_strategy(), st.integers(min_value=1, max_value=10))
    @settings(max_examples=30, deadline=30000)
    def test_ip_range_whitelist_property(self, ip_range_data, num_test_ips):
        """
        **Feature: superinsight-platform, Property 14: IP 白名单访问控制 (IP Ranges)**
        **Validates: Requirements 8.4**
        
        For any IP range whitelist entry, IPs within the range should be allowed
        and IPs outside the range should be denied.
        """
        # Clear mock database
        self.mock_db_whitelist.clear()
        
        tenant_id = "range_test_tenant"
        network = ip_range_data["network"]
        
        # Create whitelist entry with IP range
        whitelist_entry = {
            "tenant_id": tenant_id,
            "ip_address": str(network.network_address),
            "ip_range": ip_range_data["cidr"],
            "description": "Range test entry"
        }
        
        mock_entry = self._create_mock_whitelist_entry(whitelist_entry)
        
        # Mock the database query
        with patch.object(self.mock_db_session, 'query') as mock_query:
            mock_query.return_value.filter.return_value.all.return_value = [mock_entry]
            
            # Test IPs within the range
            tested_ips = set()
            for i, ip in enumerate(network.hosts()):
                if i >= num_test_ips:
                    break
                
                ip_str = str(ip)
                tested_ips.add(ip_str)
                
                is_allowed = self.security_controller.is_ip_whitelisted(
                    ip_address=ip_str,
                    tenant_id=tenant_id,
                    db=self.mock_db_session
                )
                
                # Should be allowed since it's in the range
                assert is_allowed, (
                    f"IP {ip_str} should be allowed as it's within range {ip_range_data['cidr']}"
                )
            
            # Test IPs outside the range
            outside_ips = [
                "203.0.113.1",  # TEST-NET-3
                "198.51.100.1",  # TEST-NET-2
                "192.0.2.1"     # TEST-NET-1
            ]
            
            for outside_ip in outside_ips:
                # Skip if this IP happens to be in the range
                try:
                    if ipaddress.ip_address(outside_ip) in network:
                        continue
                except ValueError:
                    continue
                
                is_allowed = self.security_controller.is_ip_whitelisted(
                    ip_address=outside_ip,
                    tenant_id=tenant_id,
                    db=self.mock_db_session
                )
                
                # Should be denied since it's outside the range
                assert not is_allowed, (
                    f"IP {outside_ip} should be denied as it's outside range {ip_range_data['cidr']}"
                )


# Additional edge case tests for IP whitelist access control
class TestIPWhitelistAccessControlEdgeCases:
    """Test edge cases and boundary conditions for IP whitelist access control."""
    
    def setup_method(self):
        """Set up test environment."""
        self.security_controller = SecurityController()
        self.mock_db_session = Mock()
    
    def test_empty_whitelist_denial_property(self):
        """
        **Feature: superinsight-platform, Property 14: IP 白名单访问控制 (Empty Whitelist)**
        **Validates: Requirements 8.4**
        
        For any tenant with no whitelist entries, all IP addresses should be denied access.
        """
        tenant_id = "empty_whitelist_tenant"
        test_ips = ["192.168.1.1", "10.0.0.1", "203.0.113.1", "8.8.8.8"]
        
        # Mock empty whitelist
        with patch.object(self.mock_db_session, 'query') as mock_query:
            mock_query.return_value.filter.return_value.all.return_value = []
            
            for test_ip in test_ips:
                is_allowed = self.security_controller.is_ip_whitelisted(
                    ip_address=test_ip,
                    tenant_id=tenant_id,
                    db=self.mock_db_session
                )
                
                # Should be denied
                assert not is_allowed, (
                    f"IP {test_ip} should be denied when whitelist is empty"
                )
    
    def test_invalid_ip_address_denial_property(self):
        """
        **Feature: superinsight-platform, Property 14: IP 白名单访问控制 (Invalid IPs)**
        **Validates: Requirements 8.4**
        
        For any invalid IP address format, access should be denied regardless of whitelist.
        """
        tenant_id = "invalid_ip_test_tenant"
        invalid_ips = [
            "not.an.ip.address",
            "256.256.256.256",
            "192.168.1",
            "192.168.1.1.1",
            "",
            "localhost",
            "192.168.1.999"
        ]
        
        # Create a permissive whitelist entry
        mock_entry = Mock()
        mock_entry.tenant_id = tenant_id
        mock_entry.ip_address = "192.168.1.1"
        mock_entry.ip_range = "0.0.0.0/0"  # Allow all IPs
        mock_entry.is_active = True
        
        with patch.object(self.mock_db_session, 'query') as mock_query:
            mock_query.return_value.filter.return_value.all.return_value = [mock_entry]
            
            for invalid_ip in invalid_ips:
                is_allowed = self.security_controller.is_ip_whitelisted(
                    ip_address=invalid_ip,
                    tenant_id=tenant_id,
                    db=self.mock_db_session
                )
                
                # Should be denied due to invalid format
                assert not is_allowed, (
                    f"Invalid IP '{invalid_ip}' should be denied regardless of whitelist"
                )
    
    def test_inactive_whitelist_entry_denial_property(self):
        """
        **Feature: superinsight-platform, Property 14: IP 白名单访问控制 (Inactive Entries)**
        **Validates: Requirements 8.4**
        
        For any inactive whitelist entry, the IP should be denied access.
        """
        tenant_id = "inactive_entry_tenant"
        test_ip = "192.168.1.100"
        
        # Create inactive whitelist entry
        mock_entry = Mock()
        mock_entry.tenant_id = tenant_id
        mock_entry.ip_address = test_ip
        mock_entry.ip_range = None
        mock_entry.is_active = False  # Inactive entry
        
        with patch.object(self.mock_db_session, 'query') as mock_query:
            # Mock query should filter out inactive entries
            mock_query.return_value.filter.return_value.all.return_value = []
            
            is_allowed = self.security_controller.is_ip_whitelisted(
                ip_address=test_ip,
                tenant_id=tenant_id,
                db=self.mock_db_session
            )
            
            # Should be denied since entry is inactive
            assert not is_allowed, (
                f"IP {test_ip} should be denied when whitelist entry is inactive"
            )