"""
Unit tests for billing system in SuperInsight Platform.

Tests core billing functionality including time tracking accuracy,
bill generation logic, and multi-tenant data isolation.
"""

import pytest
from datetime import datetime, date, timedelta
from typing import Dict, Any, List
from uuid import uuid4, UUID
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock

from src.billing.service import BillingSystem
from src.billing.models import BillingRecord, BillingRule, BillingMode, Bill, BillingReport
from src.database.models import BillingRecordModel


class TestBillingSystemCore:
    """
    Unit tests for core billing system functionality.
    
    Tests Requirements 7.1, 7.2, 7.3, 7.4, 7.5:
    - THE Billing_System SHALL 统计标注工时
    - THE Billing_System SHALL 统计标注条数
    - WHEN 月度结束时，THE Billing_System SHALL 生成月度账单
    - THE Billing_System SHALL 支持多租户隔离计费
    - THE Billing_System SHALL 提供计费报表和分析
    """
    
    def setup_method(self):
        """Set up test environment."""
        self.billing_system = BillingSystem()
        self.test_tenant_id = "test_tenant_123"
        self.test_user_id = "test_user_456"
        self.test_task_id = uuid4()
    
    def test_billing_rule_configuration(self):
        """Test billing rule configuration and retrieval."""
        # Test default billing rule
        default_rule = self.billing_system.get_billing_rule(self.test_tenant_id)
        assert default_rule.tenant_id == self.test_tenant_id
        assert default_rule.billing_mode == BillingMode.BY_COUNT
        
        # Test custom billing rule
        custom_rule = BillingRule(
            tenant_id=self.test_tenant_id,
            billing_mode=BillingMode.BY_TIME,
            rate_per_annotation=Decimal("0.25"),
            rate_per_hour=Decimal("75.00")
        )
        
        self.billing_system.set_billing_rule(self.test_tenant_id, custom_rule)
        retrieved_rule = self.billing_system.get_billing_rule(self.test_tenant_id)
        
        assert retrieved_rule.tenant_id == self.test_tenant_id
        assert retrieved_rule.billing_mode == BillingMode.BY_TIME
        assert retrieved_rule.rate_per_annotation == Decimal("0.25")
        assert retrieved_rule.rate_per_hour == Decimal("75.00")
    
    def test_billing_rule_cost_calculation(self):
        """Test cost calculation for different billing modes."""
        # Test BY_COUNT mode
        count_rule = BillingRule(
            tenant_id=self.test_tenant_id,
            billing_mode=BillingMode.BY_COUNT,
            rate_per_annotation=Decimal("0.50")
        )
        
        cost = count_rule.calculate_cost(annotation_count=10, time_spent=1800)
        assert cost == Decimal("5.00")  # 10 * 0.50
        
        # Test BY_TIME mode
        time_rule = BillingRule(
            tenant_id=self.test_tenant_id,
            billing_mode=BillingMode.BY_TIME,
            rate_per_hour=Decimal("60.00")
        )
        
        cost = time_rule.calculate_cost(annotation_count=10, time_spent=1800)  # 0.5 hours
        assert cost == Decimal("30.00")  # 0.5 * 60.00
        
        # Test HYBRID mode
        hybrid_rule = BillingRule(
            tenant_id=self.test_tenant_id,
            billing_mode=BillingMode.HYBRID,
            rate_per_annotation=Decimal("0.20"),
            rate_per_hour=Decimal("40.00")
        )
        
        cost = hybrid_rule.calculate_cost(annotation_count=5, time_spent=1800)  # 0.5 hours
        assert cost == Decimal("21.00")  # (5 * 0.20) + (0.5 * 40.00) = 1.00 + 20.00
        
        # Test BY_PROJECT mode
        project_rule = BillingRule(
            tenant_id=self.test_tenant_id,
            billing_mode=BillingMode.BY_PROJECT
        )
        
        cost = project_rule.calculate_cost(annotation_count=100, time_spent=7200)
        assert cost == Decimal("0.00")  # Project billing handled separately
    
    @patch('src.billing.service.get_db_session')
    def test_track_annotation_time_success(self, mock_get_db_session):
        """Test successful annotation time tracking."""
        # Mock database session
        mock_session = MagicMock()
        mock_get_db_session.return_value.__enter__.return_value = mock_session
        mock_get_db_session.return_value.__exit__.return_value = None
        
        # Set up billing rule
        rule = BillingRule(
            tenant_id=self.test_tenant_id,
            billing_mode=BillingMode.BY_COUNT,
            rate_per_annotation=Decimal("0.30")
        )
        self.billing_system.set_billing_rule(self.test_tenant_id, rule)
        
        # Track annotation time
        result = self.billing_system.track_annotation_time(
            user_id=self.test_user_id,
            task_id=self.test_task_id,
            duration=600,  # 10 minutes
            tenant_id=self.test_tenant_id,
            annotation_count=3
        )
        
        # Verify success
        assert result is True
        
        # Verify database interaction
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        
        # Verify billing record creation
        added_record = mock_session.add.call_args[0][0]
        assert isinstance(added_record, BillingRecordModel)
        assert added_record.tenant_id == self.test_tenant_id
        assert added_record.user_id == self.test_user_id
        assert added_record.task_id == self.test_task_id
        assert added_record.annotation_count == 3
        assert added_record.time_spent == 600
        assert added_record.cost == 0.90  # 3 * 0.30
    
    @patch('src.billing.service.get_db_session')
    def test_track_annotation_time_failure(self, mock_get_db_session):
        """Test annotation time tracking failure handling."""
        # Mock database session to raise exception
        mock_session = MagicMock()
        mock_session.commit.side_effect = Exception("Database error")
        mock_get_db_session.return_value.__enter__.return_value = mock_session
        
        # Track annotation time
        result = self.billing_system.track_annotation_time(
            user_id=self.test_user_id,
            task_id=self.test_task_id,
            duration=600,
            tenant_id=self.test_tenant_id,
            annotation_count=3
        )
        
        # Verify failure
        assert result is False
    
    @patch('src.billing.service.get_db_session')
    def test_calculate_monthly_bill_success(self, mock_get_db_session):
        """Test successful monthly bill calculation."""
        # Mock database session and query results
        mock_session = MagicMock()
        mock_get_db_session.return_value.__enter__.return_value = mock_session
        
        # Create mock billing records for January 2024
        mock_records = [
            MagicMock(
                id=uuid4(),
                tenant_id=self.test_tenant_id,
                user_id="user1",
                annotation_count=5,
                time_spent=300,
                cost=2.50,
                billing_date=date(2024, 1, 15)
            ),
            MagicMock(
                id=uuid4(),
                tenant_id=self.test_tenant_id,
                user_id="user2",
                annotation_count=8,
                time_spent=480,
                cost=4.00,
                billing_date=date(2024, 1, 20)
            )
        ]
        
        # Mock SQLAlchemy 2.0 style execute/scalars
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = mock_records
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        # Calculate monthly bill
        bill = self.billing_system.calculate_monthly_bill(self.test_tenant_id, "2024-01")
        
        # Verify bill contents
        assert isinstance(bill, Bill)
        assert bill.tenant_id == self.test_tenant_id
        assert bill.billing_period == "2024-01"
        assert bill.total_annotations == 13  # 5 + 8
        assert bill.total_time_spent == 780  # 300 + 480
        assert bill.total_cost == Decimal("6.50")  # 2.50 + 4.00
        assert len(bill.billing_records) == 2
    
    @patch('src.billing.service.get_db_session')
    def test_calculate_monthly_bill_empty_month(self, mock_get_db_session):
        """Test monthly bill calculation for month with no records."""
        # Mock database session with empty results
        mock_session = MagicMock()
        mock_get_db_session.return_value.__enter__.return_value = mock_session
        
        # Mock SQLAlchemy 2.0 style execute/scalars for empty results
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        # Calculate monthly bill for empty month
        bill = self.billing_system.calculate_monthly_bill(self.test_tenant_id, "2024-02")
        
        # Verify empty bill
        assert isinstance(bill, Bill)
        assert bill.tenant_id == self.test_tenant_id
        assert bill.billing_period == "2024-02"
        assert bill.total_annotations == 0
        assert bill.total_time_spent == 0
        assert bill.total_cost == Decimal("0.00")
        assert len(bill.billing_records) == 0
    
    @patch('src.billing.service.get_db_session')
    def test_calculate_monthly_bill_december(self, mock_get_db_session):
        """Test monthly bill calculation for December (year boundary)."""
        # Mock database session
        mock_session = MagicMock()
        mock_get_db_session.return_value.__enter__.return_value = mock_session
        
        mock_records = [
            MagicMock(
                id=uuid4(),
                tenant_id=self.test_tenant_id,
                user_id="user1",
                annotation_count=10,
                time_spent=600,
                cost=5.00,
                billing_date=date(2024, 12, 25)
            )
        ]
        
        # Mock SQLAlchemy 2.0 style execute/scalars
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = mock_records
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        # Calculate December bill
        bill = self.billing_system.calculate_monthly_bill(self.test_tenant_id, "2024-12")
        
        # Verify bill
        assert bill.billing_period == "2024-12"
        assert bill.total_annotations == 10
        assert bill.total_cost == Decimal("5.00")
    
    @patch('src.billing.service.get_db_session')
    def test_generate_report_success(self, mock_get_db_session):
        """Test successful billing report generation."""
        # Mock database session
        mock_session = MagicMock()
        mock_get_db_session.return_value.__enter__.return_value = mock_session
        
        # Create mock billing records
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        
        mock_records = [
            MagicMock(
                tenant_id=self.test_tenant_id,
                user_id="user1",
                annotation_count=5,
                time_spent=300,
                cost=2.50,
                billing_date=date(2024, 1, 10)
            ),
            MagicMock(
                tenant_id=self.test_tenant_id,
                user_id="user1",
                annotation_count=3,
                time_spent=180,
                cost=1.50,
                billing_date=date(2024, 1, 15)
            ),
            MagicMock(
                tenant_id=self.test_tenant_id,
                user_id="user2",
                annotation_count=7,
                time_spent=420,
                cost=3.50,
                billing_date=date(2024, 1, 20)
            )
        ]
        
        # Mock SQLAlchemy 2.0 style execute/scalars
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = mock_records
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        # Generate report
        report = self.billing_system.generate_report(self.test_tenant_id, start_date, end_date)
        
        # Verify report structure
        assert isinstance(report, BillingReport)
        assert report.tenant_id == self.test_tenant_id
        assert report.start_date == start_date
        assert report.end_date == end_date
        assert report.total_cost == Decimal("7.50")  # 2.50 + 1.50 + 3.50
        assert report.total_annotations == 15  # 5 + 3 + 7
        assert report.total_time_spent == 900  # 300 + 180 + 420
        
        # Verify user breakdown
        assert "user1" in report.user_breakdown
        assert "user2" in report.user_breakdown
        
        user1_stats = report.user_breakdown["user1"]
        assert user1_stats["annotations"] == 8  # 5 + 3
        assert user1_stats["time_spent"] == 480  # 300 + 180
        assert user1_stats["cost"] == 4.00  # 2.50 + 1.50
        
        user2_stats = report.user_breakdown["user2"]
        assert user2_stats["annotations"] == 7
        assert user2_stats["time_spent"] == 420
        assert user2_stats["cost"] == 3.50
        
        # Verify daily breakdown
        assert "2024-01-10" in report.daily_breakdown
        assert "2024-01-15" in report.daily_breakdown
        assert "2024-01-20" in report.daily_breakdown
    
    @patch('src.billing.service.get_db_session')
    def test_get_tenant_billing_records_with_date_filters(self, mock_get_db_session):
        """Test getting tenant billing records with date filters."""
        # Mock database session
        mock_session = MagicMock()
        mock_get_db_session.return_value.__enter__.return_value = mock_session
        
        # Create mock database records
        mock_db_records = [
            MagicMock(
                id=uuid4(),
                tenant_id=self.test_tenant_id,
                user_id="user1",
                task_id=uuid4(),
                annotation_count=5,
                time_spent=300,
                cost=2.50,
                billing_date=date(2024, 1, 15),
                created_at=datetime.now()
            )
        ]
        
        # Mock SQLAlchemy 2.0 style execute/scalars
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = mock_db_records
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        # Get records with date filters
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        
        records = self.billing_system.get_tenant_billing_records(
            self.test_tenant_id, start_date, end_date
        )
        
        # Verify results
        assert len(records) == 1
        assert isinstance(records[0], BillingRecord)
        assert records[0].tenant_id == self.test_tenant_id
        assert records[0].annotation_count == 5
        assert records[0].cost == Decimal("2.50")
    
    @patch('src.billing.service.get_db_session')
    def test_get_user_billing_summary(self, mock_get_db_session):
        """Test getting user billing summary."""
        # Mock database session
        mock_session = MagicMock()
        mock_get_db_session.return_value.__enter__.return_value = mock_session
        
        # Create mock database records for specific user
        mock_db_records = [
            MagicMock(
                tenant_id=self.test_tenant_id,
                user_id=self.test_user_id,
                annotation_count=5,
                time_spent=300,
                cost=2.50,
                billing_date=date(2024, 1, 15)
            ),
            MagicMock(
                tenant_id=self.test_tenant_id,
                user_id=self.test_user_id,
                annotation_count=3,
                time_spent=180,
                cost=1.50,
                billing_date=date(2024, 1, 20)
            )
        ]
        
        # Mock SQLAlchemy 2.0 style execute/scalars
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = mock_db_records
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        # Get user billing summary
        summary = self.billing_system.get_user_billing_summary(
            self.test_tenant_id, self.test_user_id
        )
        
        # Verify summary
        assert summary["user_id"] == self.test_user_id
        assert summary["total_annotations"] == 8  # 5 + 3
        assert summary["total_time_spent"] == 480  # 300 + 180
        assert summary["total_cost"] == 4.00  # 2.50 + 1.50
        assert summary["record_count"] == 2
    
    def test_export_billing_data_json_format(self):
        """Test exporting billing data in JSON format."""
        # Mock get_tenant_billing_records method
        mock_records = [
            BillingRecord(
                id=uuid4(),
                tenant_id=self.test_tenant_id,
                user_id="user1",
                annotation_count=5,
                time_spent=300,
                cost=Decimal("2.50"),
                billing_date=date(2024, 1, 15)
            )
        ]
        
        with patch.object(self.billing_system, 'get_tenant_billing_records', 
                         return_value=mock_records):
            
            # Export data in JSON format
            export_result = self.billing_system.export_billing_data(
                self.test_tenant_id, "json"
            )
            
            # Verify export result
            assert export_result["format"] == "json"
            assert export_result["record_count"] == 1
            assert len(export_result["data"]) == 1
            
            # Verify record data
            record_data = export_result["data"][0]
            assert record_data["tenant_id"] == self.test_tenant_id
            assert record_data["user_id"] == "user1"
            assert record_data["annotation_count"] == 5
            assert record_data["cost"] == 2.50
    
    def test_export_billing_data_csv_format(self):
        """Test exporting billing data in CSV format."""
        # Mock get_tenant_billing_records method
        mock_records = [
            BillingRecord(
                id=uuid4(),
                tenant_id=self.test_tenant_id,
                user_id="user1",
                annotation_count=5,
                time_spent=300,
                cost=Decimal("2.50"),
                billing_date=date(2024, 1, 15)
            )
        ]
        
        with patch.object(self.billing_system, 'get_tenant_billing_records', 
                         return_value=mock_records):
            
            # Export data in CSV format
            export_result = self.billing_system.export_billing_data(
                self.test_tenant_id, "csv"
            )
            
            # Verify export result
            assert export_result["format"] == "csv"
            assert export_result["record_count"] == 1
            assert len(export_result["data"]) == 1
            
            # Verify CSV record format
            record_data = export_result["data"][0]
            assert "id" in record_data
            assert "tenant_id" in record_data
            assert "billing_date" in record_data
            assert record_data["annotation_count"] == 5
    
    def test_export_billing_data_unsupported_format(self):
        """Test exporting billing data with unsupported format."""
        # Export data with unsupported format
        export_result = self.billing_system.export_billing_data(
            self.test_tenant_id, "xml"
        )
        
        # Verify error handling
        assert export_result["format"] == "xml"
        assert export_result["record_count"] == 0
        assert "error" in export_result
        assert "Unsupported format" in export_result["error"]


class TestBillingSystemMultiTenantIsolation:
    """
    Unit tests for multi-tenant data isolation in billing system.
    
    Tests Requirement 7.4:
    - THE Billing_System SHALL 支持多租户隔离计费
    """
    
    def setup_method(self):
        """Set up test environment."""
        self.billing_system = BillingSystem()
        self.tenant_a = "tenant_a_123"
        self.tenant_b = "tenant_b_456"
        self.user_a = "user_a_789"
        self.user_b = "user_b_012"
    
    @patch('src.billing.service.get_db_session')
    def test_tenant_isolation_in_billing_records(self, mock_get_db_session):
        """Test that billing records are properly isolated by tenant."""
        # Mock database session
        mock_session = MagicMock()
        mock_get_db_session.return_value.__enter__.return_value = mock_session
        
        # Create mock records for different tenants with proper UUIDs
        record_id_a = uuid4()
        task_id_a = uuid4()
        
        tenant_a_records = [
            MagicMock(
                id=record_id_a,
                tenant_id=self.tenant_a,
                user_id=self.user_a,
                task_id=task_id_a,
                annotation_count=5,
                time_spent=300,
                cost=2.50,
                billing_date=date(2024, 1, 15),
                created_at=datetime.now()
            )
        ]
        
        # Mock SQLAlchemy 2.0 style execute/scalars
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = tenant_a_records
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        # Get records for tenant A
        records_a = self.billing_system.get_tenant_billing_records(self.tenant_a)
        
        # Verify only tenant A records are returned
        assert len(records_a) == 1
        assert records_a[0].tenant_id == self.tenant_a
        assert records_a[0].user_id == self.user_a
        
        # Verify database execute was called
        mock_session.execute.assert_called()
    
    @patch('src.billing.service.get_db_session')
    def test_tenant_isolation_in_user_summary(self, mock_get_db_session):
        """Test that user billing summaries are isolated by tenant."""
        # Mock database session
        mock_session = MagicMock()
        mock_get_db_session.return_value.__enter__.return_value = mock_session
        
        # Create mock records for same user in different tenants
        tenant_a_user_records = [
            MagicMock(
                tenant_id=self.tenant_a,
                user_id="shared_user",
                annotation_count=5,
                time_spent=300,
                cost=2.50,
                billing_date=date(2024, 1, 15)
            )
        ]
        
        # Mock SQLAlchemy 2.0 style execute/scalars
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = tenant_a_user_records
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        # Get user summary for tenant A
        summary_a = self.billing_system.get_user_billing_summary(
            self.tenant_a, "shared_user"
        )
        
        # Verify summary contains only tenant A data
        assert summary_a["user_id"] == "shared_user"
        assert summary_a["total_annotations"] == 5
        assert summary_a["total_cost"] == 2.50
        assert summary_a["record_count"] == 1
        
        # Verify database execute was called
        mock_session.execute.assert_called()
    
    @patch('src.billing.service.get_db_session')
    def test_tenant_isolation_in_monthly_bills(self, mock_get_db_session):
        """Test that monthly bills are isolated by tenant."""
        # Mock database session
        mock_session = MagicMock()
        mock_get_db_session.return_value.__enter__.return_value = mock_session
        
        # Create mock records for tenant A only
        tenant_a_records = [
            MagicMock(
                id=uuid4(),
                tenant_id=self.tenant_a,
                user_id=self.user_a,
                annotation_count=10,
                time_spent=600,
                cost=5.00,
                billing_date=date(2024, 1, 15)
            )
        ]
        
        # Mock SQLAlchemy 2.0 style execute/scalars
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = tenant_a_records
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        # Calculate monthly bill for tenant A
        bill_a = self.billing_system.calculate_monthly_bill(self.tenant_a, "2024-01")
        
        # Verify bill contains only tenant A data
        assert bill_a.tenant_id == self.tenant_a
        assert bill_a.total_annotations == 10
        assert bill_a.total_cost == Decimal("5.00")
        assert len(bill_a.billing_records) == 1
        
        # Verify database execute was called
        mock_session.execute.assert_called()
    
    @patch('src.billing.service.get_db_session')
    def test_tenant_isolation_in_reports(self, mock_get_db_session):
        """Test that billing reports are isolated by tenant."""
        # Mock database session
        mock_session = MagicMock()
        mock_get_db_session.return_value.__enter__.return_value = mock_session
        
        # Create mock records for tenant B only
        tenant_b_records = [
            MagicMock(
                tenant_id=self.tenant_b,
                user_id=self.user_b,
                annotation_count=7,
                time_spent=420,
                cost=3.50,
                billing_date=date(2024, 1, 10)
            )
        ]
        
        # Mock SQLAlchemy 2.0 style execute/scalars
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = tenant_b_records
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        # Generate report for tenant B
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        report_b = self.billing_system.generate_report(self.tenant_b, start_date, end_date)
        
        # Verify report contains only tenant B data
        assert report_b.tenant_id == self.tenant_b
        assert report_b.total_annotations == 7
        assert report_b.total_cost == Decimal("3.50")
        
        # Verify user breakdown contains only tenant B users
        assert self.user_b in report_b.user_breakdown
        assert len(report_b.user_breakdown) == 1
    
    def test_billing_rule_isolation_by_tenant(self):
        """Test that billing rules are isolated by tenant."""
        # Set different billing rules for different tenants
        rule_a = BillingRule(
            tenant_id=self.tenant_a,
            billing_mode=BillingMode.BY_COUNT,
            rate_per_annotation=Decimal("0.25"),
            rate_per_hour=Decimal("30.00")  # Different from default
        )
        
        rule_b = BillingRule(
            tenant_id=self.tenant_b,
            billing_mode=BillingMode.BY_TIME,
            rate_per_hour=Decimal("75.00"),  # Different from rule_a
            rate_per_annotation=Decimal("0.15")  # Different from rule_a
        )
        
        self.billing_system.set_billing_rule(self.tenant_a, rule_a)
        self.billing_system.set_billing_rule(self.tenant_b, rule_b)
        
        # Verify each tenant gets their own rule
        retrieved_rule_a = self.billing_system.get_billing_rule(self.tenant_a)
        retrieved_rule_b = self.billing_system.get_billing_rule(self.tenant_b)
        
        assert retrieved_rule_a.tenant_id == self.tenant_a
        assert retrieved_rule_a.billing_mode == BillingMode.BY_COUNT
        assert retrieved_rule_a.rate_per_annotation == Decimal("0.25")
        assert retrieved_rule_a.rate_per_hour == Decimal("30.00")
        
        assert retrieved_rule_b.tenant_id == self.tenant_b
        assert retrieved_rule_b.billing_mode == BillingMode.BY_TIME
        assert retrieved_rule_b.rate_per_hour == Decimal("75.00")
        assert retrieved_rule_b.rate_per_annotation == Decimal("0.15")
        
        # Verify rules don't interfere with each other
        assert retrieved_rule_a.rate_per_hour != retrieved_rule_b.rate_per_hour
        assert retrieved_rule_a.billing_mode != retrieved_rule_b.billing_mode
        assert retrieved_rule_a.rate_per_annotation != retrieved_rule_b.rate_per_annotation
    
    def test_cross_tenant_data_access_prevention(self):
        """Test that tenants cannot access each other's billing data."""
        # This test verifies the design principle that all billing methods
        # require tenant_id parameter and filter by it
        
        # Verify all key methods require tenant_id parameter
        import inspect
        
        # Check track_annotation_time method signature
        track_sig = inspect.signature(self.billing_system.track_annotation_time)
        assert 'tenant_id' in track_sig.parameters
        
        # Check calculate_monthly_bill method signature
        bill_sig = inspect.signature(self.billing_system.calculate_monthly_bill)
        assert 'tenant_id' in bill_sig.parameters
        
        # Check generate_report method signature
        report_sig = inspect.signature(self.billing_system.generate_report)
        assert 'tenant_id' in report_sig.parameters
        
        # Check get_tenant_billing_records method signature
        records_sig = inspect.signature(self.billing_system.get_tenant_billing_records)
        assert 'tenant_id' in records_sig.parameters
        
        # Check get_user_billing_summary method signature
        summary_sig = inspect.signature(self.billing_system.get_user_billing_summary)
        assert 'tenant_id' in summary_sig.parameters
        
        # Check export_billing_data method signature
        export_sig = inspect.signature(self.billing_system.export_billing_data)
        assert 'tenant_id' in export_sig.parameters


class TestBillingSystemTimeTracking:
    """
    Unit tests for time tracking accuracy in billing system.
    
    Tests Requirement 7.1:
    - THE Billing_System SHALL 统计标注工时
    """
    
    def setup_method(self):
        """Set up test environment."""
        self.billing_system = BillingSystem()
        self.test_tenant_id = "time_test_tenant"
        self.test_user_id = "time_test_user"
    
    def test_time_tracking_precision(self):
        """Test that time tracking maintains precision in seconds."""
        # Test various time durations
        test_durations = [1, 30, 60, 300, 1800, 3600, 7200]  # 1s to 2h
        
        for duration in test_durations:
            with patch('src.billing.service.get_db_session') as mock_get_db_session:
                # Mock database session
                mock_session = MagicMock()
                mock_get_db_session.return_value.__enter__.return_value = mock_session
                mock_get_db_session.return_value.__exit__.return_value = None
                
                # Track annotation time
                result = self.billing_system.track_annotation_time(
                    user_id=self.test_user_id,
                    task_id=uuid4(),
                    duration=duration,
                    tenant_id=self.test_tenant_id,
                    annotation_count=1
                )
                
                # Verify success
                assert result is True
                
                # Verify time precision is maintained
                added_record = mock_session.add.call_args[0][0]
                assert added_record.time_spent == duration
    
    def test_time_tracking_with_zero_duration(self):
        """Test time tracking with zero duration."""
        with patch('src.billing.service.get_db_session') as mock_get_db_session:
            # Mock database session
            mock_session = MagicMock()
            mock_get_db_session.return_value.__enter__.return_value = mock_session
            mock_get_db_session.return_value.__exit__.return_value = None
            
            # Track annotation with zero time
            result = self.billing_system.track_annotation_time(
                user_id=self.test_user_id,
                task_id=uuid4(),
                duration=0,
                tenant_id=self.test_tenant_id,
                annotation_count=1
            )
            
            # Verify success and zero time is recorded
            assert result is True
            added_record = mock_session.add.call_args[0][0]
            assert added_record.time_spent == 0
    
    def test_time_aggregation_accuracy(self):
        """Test that time aggregation across multiple records is accurate."""
        with patch('src.billing.service.get_db_session') as mock_get_db_session:
            # Mock database session
            mock_session = MagicMock()
            mock_get_db_session.return_value.__enter__.return_value = mock_session
            
            # Create mock records with different time values
            mock_records = [
                MagicMock(
                    tenant_id=self.test_tenant_id,
                    user_id=self.test_user_id,
                    annotation_count=1,
                    time_spent=300,  # 5 minutes
                    cost=1.00,
                    billing_date=date(2024, 1, 10)
                ),
                MagicMock(
                    tenant_id=self.test_tenant_id,
                    user_id=self.test_user_id,
                    annotation_count=2,
                    time_spent=450,  # 7.5 minutes
                    cost=2.00,
                    billing_date=date(2024, 1, 15)
                ),
                MagicMock(
                    tenant_id=self.test_tenant_id,
                    user_id=self.test_user_id,
                    annotation_count=1,
                    time_spent=150,  # 2.5 minutes
                    cost=1.00,
                    billing_date=date(2024, 1, 20)
                )
            ]
            
            # Mock SQLAlchemy 2.0 style execute/scalars
            mock_result = MagicMock()
            mock_scalars = MagicMock()
            mock_scalars.all.return_value = mock_records
            mock_result.scalars.return_value = mock_scalars
            mock_session.execute.return_value = mock_result
            
            # Get user billing summary
            summary = self.billing_system.get_user_billing_summary(
                self.test_tenant_id, self.test_user_id
            )
            
            # Verify time aggregation accuracy
            expected_total_time = 300 + 450 + 150  # 900 seconds = 15 minutes
            assert summary["total_time_spent"] == expected_total_time
            assert summary["total_annotations"] == 4  # 1 + 2 + 1
    
    def test_time_tracking_with_billing_rule_calculation(self):
        """Test time tracking accuracy with different billing rule calculations."""
        # Set up time-based billing rule
        time_rule = BillingRule(
            tenant_id=self.test_tenant_id,
            billing_mode=BillingMode.BY_TIME,
            rate_per_hour=Decimal("60.00")  # $60 per hour
        )
        self.billing_system.set_billing_rule(self.test_tenant_id, time_rule)
        
        with patch('src.billing.service.get_db_session') as mock_get_db_session:
            # Mock database session
            mock_session = MagicMock()
            mock_get_db_session.return_value.__enter__.return_value = mock_session
            mock_get_db_session.return_value.__exit__.return_value = None
            
            # Track 30 minutes of work (1800 seconds)
            result = self.billing_system.track_annotation_time(
                user_id=self.test_user_id,
                task_id=uuid4(),
                duration=1800,  # 30 minutes
                tenant_id=self.test_tenant_id,
                annotation_count=5
            )
            
            # Verify success
            assert result is True
            
            # Verify time-based cost calculation
            added_record = mock_session.add.call_args[0][0]
            assert added_record.time_spent == 1800
            assert added_record.cost == 30.00  # 0.5 hours * $60/hour
    
    def test_time_tracking_boundary_values(self):
        """Test time tracking with boundary values."""
        boundary_values = [
            0,      # Minimum time
            1,      # One second
            59,     # Just under a minute
            60,     # Exactly one minute
            3599,   # Just under an hour
            3600,   # Exactly one hour
            86400   # One day (24 hours)
        ]
        
        for duration in boundary_values:
            with patch('src.billing.service.get_db_session') as mock_get_db_session:
                # Mock database session
                mock_session = MagicMock()
                mock_get_db_session.return_value.__enter__.return_value = mock_session
                mock_get_db_session.return_value.__exit__.return_value = None
                
                # Track annotation time
                result = self.billing_system.track_annotation_time(
                    user_id=self.test_user_id,
                    task_id=uuid4(),
                    duration=duration,
                    tenant_id=self.test_tenant_id,
                    annotation_count=1
                )
                
                # Verify success and exact duration recording
                assert result is True
                added_record = mock_session.add.call_args[0][0]
                assert added_record.time_spent == duration
    
    @patch('src.billing.service.get_db_session')
    def test_time_tracking_in_monthly_bill(self, mock_get_db_session):
        """Test that time tracking is accurately reflected in monthly bills."""
        # Mock database session
        mock_session = MagicMock()
        mock_get_db_session.return_value.__enter__.return_value = mock_session
        
        # Create mock records with specific time values
        mock_records = [
            MagicMock(
                id=uuid4(),
                tenant_id=self.test_tenant_id,
                user_id="user1",
                annotation_count=3,
                time_spent=1800,  # 30 minutes
                cost=15.00,
                billing_date=date(2024, 1, 10)
            ),
            MagicMock(
                id=uuid4(),
                tenant_id=self.test_tenant_id,
                user_id="user2",
                annotation_count=5,
                time_spent=2700,  # 45 minutes
                cost=22.50,
                billing_date=date(2024, 1, 15)
            )
        ]
        
        # Mock SQLAlchemy 2.0 style execute/scalars
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = mock_records
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result
        
        # Calculate monthly bill
        bill = self.billing_system.calculate_monthly_bill(self.test_tenant_id, "2024-01")
        
        # Verify time aggregation in monthly bill
        expected_total_time = 1800 + 2700  # 4500 seconds = 75 minutes
        assert bill.total_time_spent == expected_total_time
        assert bill.total_annotations == 8  # 3 + 5
        assert bill.total_cost == Decimal("37.50")  # 15.00 + 22.50