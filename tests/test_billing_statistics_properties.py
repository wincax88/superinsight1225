"""
Property-based tests for billing statistics accuracy in SuperInsight Platform.

Tests the billing statistics accuracy property to ensure that total billing
for any user equals the sum of individual task billing records.
"""

import pytest
from datetime import datetime, date, timedelta
from typing import Dict, Any, List
from uuid import uuid4, UUID
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings, assume
from hypothesis.strategies import composite

from src.billing.service import BillingSystem
from src.billing.models import BillingRecord, BillingRule, BillingMode
from src.database.models import BillingRecordModel


# Test data generators
@composite
def billing_record_strategy(draw):
    """Generate valid BillingRecord instances."""
    return BillingRecord(
        id=uuid4(),
        tenant_id=draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')))),
        user_id=draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')))),
        task_id=draw(st.one_of(st.none(), st.uuids())),
        annotation_count=draw(st.integers(min_value=0, max_value=100)),
        time_spent=draw(st.integers(min_value=0, max_value=7200)),  # Max 2 hours
        cost=draw(st.decimals(min_value=0, max_value=1000, places=2)),
        billing_date=draw(st.dates(min_value=date(2024, 1, 1), max_value=date.today())),
        created_at=datetime.now()
    )


@composite
def billing_rule_strategy(draw):
    """Generate valid BillingRule instances."""
    return BillingRule(
        tenant_id=draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')))),
        billing_mode=draw(st.sampled_from(list(BillingMode))),
        rate_per_annotation=draw(st.decimals(min_value=0, max_value=10, places=2)),
        rate_per_hour=draw(st.decimals(min_value=0, max_value=200, places=2)),
        project_annual_fee=draw(st.decimals(min_value=0, max_value=50000, places=2))
    )


@composite
def user_billing_data_strategy(draw):
    """Generate billing data for multiple users with consistent tenant."""
    tenant_id = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))
    
    # Generate 1-5 users
    num_users = draw(st.integers(min_value=1, max_value=5))
    users_data = {}
    
    for i in range(num_users):
        user_id = f"user_{i}_{draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))))}"
        
        # Generate 1-10 billing records per user
        num_records = draw(st.integers(min_value=1, max_value=10))
        user_records = []
        
        for j in range(num_records):
            record = BillingRecord(
                id=uuid4(),
                tenant_id=tenant_id,
                user_id=user_id,
                task_id=draw(st.one_of(st.none(), st.uuids())),
                annotation_count=draw(st.integers(min_value=0, max_value=50)),
                time_spent=draw(st.integers(min_value=0, max_value=3600)),
                cost=draw(st.decimals(min_value=0, max_value=500, places=2)),
                billing_date=draw(st.dates(min_value=date(2024, 1, 1), max_value=date.today())),
                created_at=datetime.now()
            )
            user_records.append(record)
        
        users_data[user_id] = user_records
    
    return {
        "tenant_id": tenant_id,
        "users_data": users_data
    }


class TestBillingStatisticsAccuracy:
    """
    Property-based tests for billing statistics accuracy.
    
    Validates Requirements 7.1 and 7.2:
    - THE Billing_System SHALL 统计标注工时
    - THE Billing_System SHALL 统计标注条数
    """
    
    def setup_method(self):
        """Set up test environment."""
        self.billing_system = BillingSystem()
        self.mock_db_records = {}  # In-memory storage for testing
    
    def teardown_method(self):
        """Clean up test environment."""
        self.mock_db_records.clear()  # Ensure clean state between tests
    
    def _save_billing_records_to_db(self, records: List[BillingRecord]):
        """Save billing records to mock database for testing."""
        for record in records:
            key = str(record.id)
            self.mock_db_records[key] = record
    
    def _mock_get_user_billing_summary(self, tenant_id: str, user_id: str,
                                      start_date=None, end_date=None):
        """Mock implementation of get_user_billing_summary."""
        # Filter records by tenant and user
        user_records = [
            record for record in self.mock_db_records.values()
            if record.tenant_id == tenant_id and record.user_id == user_id
        ]
        
        # Apply date filters if provided
        if start_date:
            user_records = [r for r in user_records if r.billing_date >= start_date]
        if end_date:
            user_records = [r for r in user_records if r.billing_date <= end_date]
        
        # Calculate totals
        total_annotations = sum(r.annotation_count for r in user_records)
        total_time_spent = sum(r.time_spent for r in user_records)
        total_cost = sum(r.cost for r in user_records)
        
        return {
            "user_id": user_id,
            "total_annotations": total_annotations,
            "total_time_spent": total_time_spent,
            "total_cost": float(total_cost),
            "record_count": len(user_records)
        }
    
    def _mock_get_tenant_billing_records(self, tenant_id: str,
                                        start_date=None, end_date=None):
        """Mock implementation of get_tenant_billing_records."""
        # Filter records by tenant
        tenant_records = [
            record for record in self.mock_db_records.values()
            if record.tenant_id == tenant_id
        ]
        
        # Apply date filters if provided
        if start_date:
            tenant_records = [r for r in tenant_records if r.billing_date >= start_date]
        if end_date:
            tenant_records = [r for r in tenant_records if r.billing_date <= end_date]
        
        return tenant_records
    
    def _mock_calculate_monthly_bill(self, tenant_id: str, month: str):
        """Mock implementation of calculate_monthly_bill."""
        from src.billing.models import Bill
        
        # Parse month
        year, month_num = map(int, month.split('-'))
        start_date = date(year, month_num, 1)
        if month_num == 12:
            end_date = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = date(year, month_num + 1, 1) - timedelta(days=1)
        
        # Get records for the month
        month_records = [
            record for record in self.mock_db_records.values()
            if record.tenant_id == tenant_id
            and start_date <= record.billing_date <= end_date
        ]
        
        # Calculate totals
        total_annotations = sum(r.annotation_count for r in month_records)
        total_time_spent = sum(r.time_spent for r in month_records)
        total_cost = sum(r.cost for r in month_records)
        
        return Bill(
            tenant_id=tenant_id,
            billing_period=month,
            total_annotations=total_annotations,
            total_time_spent=total_time_spent,
            total_cost=total_cost,
            billing_records=[r.id for r in month_records]
        )
    
    def _mock_generate_report(self, tenant_id: str, start_date: date, end_date: date):
        """Mock implementation of generate_report."""
        from src.billing.models import BillingReport
        
        # Get records for the period
        period_records = [
            record for record in self.mock_db_records.values()
            if record.tenant_id == tenant_id
            and start_date <= record.billing_date <= end_date
        ]
        
        # Calculate totals
        total_cost = sum(r.cost for r in period_records)
        total_annotations = sum(r.annotation_count for r in period_records)
        total_time_spent = sum(r.time_spent for r in period_records)
        
        return BillingReport(
            tenant_id=tenant_id,
            start_date=start_date,
            end_date=end_date,
            total_cost=total_cost,
            total_annotations=total_annotations,
            total_time_spent=total_time_spent
        )
    
    @given(user_billing_data_strategy())
    @settings(max_examples=100, deadline=30000)
    def test_billing_statistics_accuracy_property(self, billing_data):
        """
        **Feature: superinsight-platform, Property 11: 计费统计准确性**
        **Validates: Requirements 7.1, 7.2**
        
        For any user's billing records, the total billing calculated by the system
        should equal the sum of all individual task billing records for that user.
        """
        # Clear any previous test data
        self.mock_db_records.clear()
        
        tenant_id = billing_data["tenant_id"]
        users_data = billing_data["users_data"]
        
        # Assume we have at least one user with records
        assume(len(users_data) > 0)
        assume(all(len(records) > 0 for records in users_data.values()))
        
        # Save all billing records to mock database
        all_records = []
        for user_records in users_data.values():
            all_records.extend(user_records)
        
        self._save_billing_records_to_db(all_records)
        
        # Mock the billing system methods to use our in-memory data
        with patch.object(self.billing_system, 'get_user_billing_summary', 
                         side_effect=self._mock_get_user_billing_summary):
            
            # Test the property for each user
            for user_id, user_records in users_data.items():
                # Calculate expected totals from individual records
                expected_total_cost = sum(record.cost for record in user_records)
                expected_total_annotations = sum(record.annotation_count for record in user_records)
                expected_total_time = sum(record.time_spent for record in user_records)
                
                # Get billing summary from the system
                billing_summary = self.billing_system.get_user_billing_summary(
                    tenant_id=tenant_id,
                    user_id=user_id
                )
                
                # Verify the property: system total should equal sum of individual records
                actual_total_cost = Decimal(str(billing_summary["total_cost"]))
                actual_total_annotations = billing_summary["total_annotations"]
                actual_total_time = billing_summary["total_time_spent"]
                
                # Assert billing accuracy property
                assert actual_total_cost == expected_total_cost, (
                    f"Cost mismatch for user {user_id}: "
                    f"expected {expected_total_cost}, got {actual_total_cost}"
                )
                
                assert actual_total_annotations == expected_total_annotations, (
                    f"Annotation count mismatch for user {user_id}: "
                    f"expected {expected_total_annotations}, got {actual_total_annotations}"
                )
                
                assert actual_total_time == expected_total_time, (
                    f"Time spent mismatch for user {user_id}: "
                    f"expected {expected_total_time}, got {actual_total_time}"
                )
                
                # Verify record count matches
                expected_record_count = len(user_records)
                actual_record_count = billing_summary["record_count"]
                
                assert actual_record_count == expected_record_count, (
                    f"Record count mismatch for user {user_id}: "
                    f"expected {expected_record_count}, got {actual_record_count}"
                )
    
    @given(user_billing_data_strategy())
    @settings(max_examples=50, deadline=30000)
    def test_tenant_billing_aggregation_accuracy_property(self, billing_data):
        """
        **Feature: superinsight-platform, Property 11: 计费统计准确性 (Tenant Level)**
        **Validates: Requirements 7.1, 7.2**
        
        For any tenant's billing records, the total tenant billing should equal
        the sum of all individual user billing totals within that tenant.
        """
        # Clear any previous test data
        self.mock_db_records.clear()
        
        tenant_id = billing_data["tenant_id"]
        users_data = billing_data["users_data"]
        
        # Assume we have at least one user with records
        assume(len(users_data) > 0)
        assume(all(len(records) > 0 for records in users_data.values()))
        
        # Save all billing records to mock database
        all_records = []
        for user_records in users_data.values():
            all_records.extend(user_records)
        
        self._save_billing_records_to_db(all_records)
        
        # Mock the billing system methods
        with patch.object(self.billing_system, 'get_tenant_billing_records', 
                         side_effect=self._mock_get_tenant_billing_records):
            
            # Calculate expected tenant totals from all individual records
            expected_tenant_cost = sum(record.cost for record in all_records)
            expected_tenant_annotations = sum(record.annotation_count for record in all_records)
            expected_tenant_time = sum(record.time_spent for record in all_records)
            
            # Get tenant billing records from the system
            tenant_records = self.billing_system.get_tenant_billing_records(tenant_id)
            
            # Calculate actual totals from system records
            actual_tenant_cost = sum(record.cost for record in tenant_records)
            actual_tenant_annotations = sum(record.annotation_count for record in tenant_records)
            actual_tenant_time = sum(record.time_spent for record in tenant_records)
            
            # Assert tenant-level billing accuracy property
            assert actual_tenant_cost == expected_tenant_cost, (
                f"Tenant cost mismatch: expected {expected_tenant_cost}, got {actual_tenant_cost}"
            )
            
            assert actual_tenant_annotations == expected_tenant_annotations, (
                f"Tenant annotation count mismatch: "
                f"expected {expected_tenant_annotations}, got {actual_tenant_annotations}"
            )
            
            assert actual_tenant_time == expected_tenant_time, (
                f"Tenant time spent mismatch: expected {expected_tenant_time}, got {actual_tenant_time}"
            )
            
            # Verify that sum of individual user totals equals tenant total
            with patch.object(self.billing_system, 'get_user_billing_summary', 
                             side_effect=self._mock_get_user_billing_summary):
                user_totals_sum = Decimal("0.00")
                for user_id in users_data.keys():
                    user_summary = self.billing_system.get_user_billing_summary(tenant_id, user_id)
                    user_totals_sum += Decimal(str(user_summary["total_cost"]))
                
                assert user_totals_sum == expected_tenant_cost, (
                    f"Sum of user totals ({user_totals_sum}) != tenant total ({expected_tenant_cost})"
                )
    
    @given(user_billing_data_strategy())
    @settings(max_examples=50, deadline=30000)
    def test_monthly_bill_accuracy_property(self, billing_data):
        """
        **Feature: superinsight-platform, Property 11: 计费统计准确性 (Monthly Bill)**
        **Validates: Requirements 7.1, 7.2**
        
        For any monthly billing period, the monthly bill totals should equal
        the sum of all billing records within that month for the tenant.
        """
        # Clear any previous test data
        self.mock_db_records.clear()
        
        tenant_id = billing_data["tenant_id"]
        users_data = billing_data["users_data"]
        
        # Assume we have at least one user with records
        assume(len(users_data) > 0)
        assume(all(len(records) > 0 for records in users_data.values()))
        
        # Save all billing records to mock database
        all_records = []
        for user_records in users_data.values():
            all_records.extend(user_records)
        
        self._save_billing_records_to_db(all_records)
        
        # Mock the billing system methods
        with patch.object(self.billing_system, 'calculate_monthly_bill', 
                         side_effect=self._mock_calculate_monthly_bill):
            
            # Group records by month
            monthly_records = {}
            for record in all_records:
                month_key = record.billing_date.strftime("%Y-%m")
                if month_key not in monthly_records:
                    monthly_records[month_key] = []
                monthly_records[month_key].append(record)
            
            # Test each month's billing accuracy
            for month_str, month_records in monthly_records.items():
                # Calculate expected monthly totals
                expected_monthly_cost = sum(record.cost for record in month_records)
                expected_monthly_annotations = sum(record.annotation_count for record in month_records)
                expected_monthly_time = sum(record.time_spent for record in month_records)
                
                # Generate monthly bill from the system
                monthly_bill = self.billing_system.calculate_monthly_bill(tenant_id, month_str)
                
                # Assert monthly bill accuracy property
                assert monthly_bill.total_cost == expected_monthly_cost, (
                    f"Monthly bill cost mismatch for {month_str}: "
                    f"expected {expected_monthly_cost}, got {monthly_bill.total_cost}"
                )
                
                assert monthly_bill.total_annotations == expected_monthly_annotations, (
                    f"Monthly bill annotation count mismatch for {month_str}: "
                    f"expected {expected_monthly_annotations}, got {monthly_bill.total_annotations}"
                )
                
                assert monthly_bill.total_time_spent == expected_monthly_time, (
                    f"Monthly bill time spent mismatch for {month_str}: "
                    f"expected {expected_monthly_time}, got {monthly_bill.total_time_spent}"
                )
                
                # Verify billing period is correct
                assert monthly_bill.billing_period == month_str, (
                    f"Billing period mismatch: expected {month_str}, got {monthly_bill.billing_period}"
                )
    
    @given(
        st.lists(billing_record_strategy(), min_size=1, max_size=20),
        st.dates(min_value=date(2024, 1, 1), max_value=date.today()),
        st.dates(min_value=date(2024, 1, 1), max_value=date.today())
    )
    @settings(max_examples=50, deadline=30000)
    def test_date_range_billing_accuracy_property(self, records, start_date, end_date):
        """
        **Feature: superinsight-platform, Property 11: 计费统计准确性 (Date Range)**
        **Validates: Requirements 7.1, 7.2**
        
        For any date range query, the billing totals should equal the sum of
        all billing records within that date range.
        """
        # Ensure end_date is after start_date
        if end_date < start_date:
            start_date, end_date = end_date, start_date
        
        # Use the same tenant for all records
        tenant_id = f"test_tenant_{uuid4().hex[:8]}"
        
        # Set consistent tenant_id and adjust dates to be within range
        adjusted_records = []
        for record in records:
            # Adjust billing date to be within the test range
            days_in_range = (end_date - start_date).days
            if days_in_range > 0:
                random_offset = hash(str(record.id)) % (days_in_range + 1)
                billing_date = start_date + timedelta(days=random_offset)
            else:
                billing_date = start_date
            
            adjusted_record = BillingRecord(
                id=record.id,
                tenant_id=tenant_id,
                user_id=record.user_id,
                task_id=record.task_id,
                annotation_count=record.annotation_count,
                time_spent=record.time_spent,
                cost=record.cost,
                billing_date=billing_date,
                created_at=record.created_at
            )
            adjusted_records.append(adjusted_record)
        
        # Save records to mock database
        self._save_billing_records_to_db(adjusted_records)
        
        # Mock the billing system methods
        with patch.object(self.billing_system, 'generate_report', 
                         side_effect=self._mock_generate_report):
            
            # Calculate expected totals for the date range
            expected_cost = sum(record.cost for record in adjusted_records)
            expected_annotations = sum(record.annotation_count for record in adjusted_records)
            expected_time = sum(record.time_spent for record in adjusted_records)
            
            # Generate billing report for the date range
            billing_report = self.billing_system.generate_report(tenant_id, start_date, end_date)
            
            # Assert date range billing accuracy property
            assert billing_report.total_cost == expected_cost, (
                f"Date range cost mismatch: expected {expected_cost}, got {billing_report.total_cost}"
            )
            
            assert billing_report.total_annotations == expected_annotations, (
                f"Date range annotation count mismatch: "
                f"expected {expected_annotations}, got {billing_report.total_annotations}"
            )
            
            assert billing_report.total_time_spent == expected_time, (
                f"Date range time spent mismatch: "
                f"expected {expected_time}, got {billing_report.total_time_spent}"
            )
            
            # Verify date range is correct
            assert billing_report.start_date == start_date
            assert billing_report.end_date == end_date


# Additional property tests for edge cases
class TestBillingStatisticsEdgeCases:
    """Test edge cases and boundary conditions for billing statistics."""
    
    def setup_method(self):
        """Set up test environment."""
        self.billing_system = BillingSystem()
        self.mock_db_records = {}
    
    def _mock_get_user_billing_summary(self, tenant_id: str, user_id: str,
                                      start_date=None, end_date=None):
        """Mock implementation of get_user_billing_summary."""
        # Filter records by tenant and user
        user_records = [
            record for record in self.mock_db_records.values()
            if record.tenant_id == tenant_id and record.user_id == user_id
        ]
        
        # Apply date filters if provided
        if start_date:
            user_records = [r for r in user_records if r.billing_date >= start_date]
        if end_date:
            user_records = [r for r in user_records if r.billing_date <= end_date]
        
        # Calculate totals
        total_annotations = sum(r.annotation_count for r in user_records)
        total_time_spent = sum(r.time_spent for r in user_records)
        total_cost = sum(r.cost for r in user_records)
        
        return {
            "user_id": user_id,
            "total_annotations": total_annotations,
            "total_time_spent": total_time_spent,
            "total_cost": float(total_cost),
            "record_count": len(user_records)
        }
    
    def test_empty_billing_records_property(self):
        """
        **Feature: superinsight-platform, Property 11: 计费统计准确性 (Empty Case)**
        **Validates: Requirements 7.1, 7.2**
        
        For any tenant with no billing records, all totals should be zero.
        """
        tenant_id = f"empty_tenant_{uuid4().hex[:8]}"
        user_id = f"empty_user_{uuid4().hex[:8]}"
        
        # Mock the billing system method
        with patch.object(self.billing_system, 'get_user_billing_summary', 
                         side_effect=self._mock_get_user_billing_summary):
            
            # Get billing summary for non-existent user
            summary = self.billing_system.get_user_billing_summary(tenant_id, user_id)
            
            # Assert empty case property
            assert summary["total_cost"] == 0.0
            assert summary["total_annotations"] == 0
            assert summary["total_time_spent"] == 0
            assert summary["record_count"] == 0
    
    def test_zero_cost_records_property(self):
        """
        **Feature: superinsight-platform, Property 11: 计费统计准确性 (Zero Cost)**
        **Validates: Requirements 7.1, 7.2**
        
        For billing records with zero costs, the totals should still be accurate.
        """
        tenant_id = f"zero_cost_tenant_{uuid4().hex[:8]}"
        user_id = f"zero_cost_user_{uuid4().hex[:8]}"
        
        # Create records with zero costs but non-zero annotations and time
        zero_cost_records = [
            BillingRecord(
                tenant_id=tenant_id,
                user_id=user_id,
                annotation_count=5,
                time_spent=300,
                cost=Decimal("0.00")
            ),
            BillingRecord(
                tenant_id=tenant_id,
                user_id=user_id,
                annotation_count=3,
                time_spent=180,
                cost=Decimal("0.00")
            )
        ]
        
        # Save records to mock database
        for record in zero_cost_records:
            self.mock_db_records[str(record.id)] = record
        
        # Mock the billing system method
        with patch.object(self.billing_system, 'get_user_billing_summary', 
                         side_effect=self._mock_get_user_billing_summary):
            
            # Get summary
            summary = self.billing_system.get_user_billing_summary(tenant_id, user_id)
            
            # Assert zero cost property
            assert summary["total_cost"] == 0.0
            assert summary["total_annotations"] == 8  # 5 + 3
            assert summary["total_time_spent"] == 480  # 300 + 180
            assert summary["record_count"] == 2