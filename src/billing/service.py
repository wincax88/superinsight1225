"""
Billing system service for SuperInsight platform.

Handles billing calculations, record tracking, and bill generation.
"""

from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional
from uuid import UUID
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, extract, select

from src.database.connection import get_db_session
from src.database.models import BillingRecordModel, TaskModel
from src.billing.models import BillingRecord, BillingRule, Bill, BillingReport, BillingMode


class BillingSystem:
    """
    Core billing system for tracking annotation work and generating bills.
    
    Supports multiple billing modes and multi-tenant isolation.
    """
    
    def __init__(self):
        """Initialize billing system."""
        self._billing_rules: Dict[str, BillingRule] = {}
    
    def set_billing_rule(self, tenant_id: str, rule: BillingRule) -> None:
        """
        Set billing rule for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            rule: Billing rule configuration
        """
        self._billing_rules[tenant_id] = rule
    
    def get_billing_rule(self, tenant_id: str) -> BillingRule:
        """
        Get billing rule for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            Billing rule for the tenant
        """
        return self._billing_rules.get(tenant_id, BillingRule(tenant_id=tenant_id))
    
    def track_annotation_time(self, user_id: str, task_id: UUID, duration: int, 
                            tenant_id: str, annotation_count: int = 1) -> bool:
        """
        Track annotation time and create billing record.
        
        Args:
            user_id: User who performed the annotation
            task_id: Task being annotated
            duration: Time spent in seconds
            tenant_id: Tenant identifier
            annotation_count: Number of annotations completed
            
        Returns:
            True if tracking was successful
        """
        try:
            # Get billing rule for tenant
            rule = self.get_billing_rule(tenant_id)
            
            # Calculate cost
            cost = rule.calculate_cost(annotation_count, duration)
            
            # Create billing record
            billing_record = BillingRecord(
                tenant_id=tenant_id,
                user_id=user_id,
                task_id=task_id,
                annotation_count=annotation_count,
                time_spent=duration,
                cost=cost,
                billing_date=date.today()
            )
            
            # Save to database
            with get_db_session() as db:
                db_record = BillingRecordModel(
                    id=billing_record.id,
                    tenant_id=billing_record.tenant_id,
                    user_id=billing_record.user_id,
                    task_id=billing_record.task_id,
                    annotation_count=billing_record.annotation_count,
                    time_spent=billing_record.time_spent,
                    cost=float(billing_record.cost),
                    billing_date=billing_record.billing_date,
                    created_at=billing_record.created_at
                )
                db.add(db_record)
                db.commit()
            
            return True
            
        except Exception as e:
            print(f"Error tracking annotation time: {e}")
            return False
    
    def calculate_monthly_bill(self, tenant_id: str, month: str) -> Bill:
        """
        Calculate monthly bill for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            month: Month in YYYY-MM format
            
        Returns:
            Generated bill
        """
        try:
            # Parse month
            year, month_num = map(int, month.split('-'))
            
            # Get start and end dates for the month
            start_date = date(year, month_num, 1)
            if month_num == 12:
                end_date = date(year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = date(year, month_num + 1, 1) - timedelta(days=1)
            
            # Query billing records for the month
            with get_db_session() as db:
                stmt = select(BillingRecordModel).where(
                    and_(
                        BillingRecordModel.tenant_id == tenant_id,
                        BillingRecordModel.billing_date >= start_date,
                        BillingRecordModel.billing_date <= end_date
                    )
                )
                records = db.execute(stmt).scalars().all()
                
                # Calculate totals
                total_annotations = sum(r.annotation_count for r in records)
                total_time_spent = sum(r.time_spent for r in records)
                total_cost = sum(Decimal(str(r.cost)) for r in records)
                billing_record_ids = [r.id for r in records]
                
                # Create bill
                bill = Bill(
                    tenant_id=tenant_id,
                    billing_period=month,
                    total_annotations=total_annotations,
                    total_time_spent=total_time_spent,
                    total_cost=total_cost,
                    billing_records=billing_record_ids
                )
                
                return bill
                
        except Exception as e:
            print(f"Error calculating monthly bill: {e}")
            # Return empty bill on error
            return Bill(
                tenant_id=tenant_id,
                billing_period=month
            )
    
    def generate_report(self, tenant_id: str, start_date: date, end_date: date) -> BillingReport:
        """
        Generate billing report for a date range.
        
        Args:
            tenant_id: Tenant identifier
            start_date: Report start date
            end_date: Report end date
            
        Returns:
            Generated billing report
        """
        try:
            with get_db_session() as db:
                # Query billing records for the period
                stmt = select(BillingRecordModel).where(
                    and_(
                        BillingRecordModel.tenant_id == tenant_id,
                        BillingRecordModel.billing_date >= start_date,
                        BillingRecordModel.billing_date <= end_date
                    )
                )
                records = db.execute(stmt).scalars().all()
                
                # Calculate totals
                total_cost = sum(Decimal(str(r.cost)) for r in records)
                total_annotations = sum(r.annotation_count for r in records)
                total_time_spent = sum(r.time_spent for r in records)
                
                # Calculate user breakdown
                user_breakdown = {}
                for record in records:
                    user_id = record.user_id
                    if user_id not in user_breakdown:
                        user_breakdown[user_id] = {
                            "annotations": 0,
                            "time_spent": 0,
                            "cost": 0.0
                        }
                    
                    user_breakdown[user_id]["annotations"] += record.annotation_count
                    user_breakdown[user_id]["time_spent"] += record.time_spent
                    user_breakdown[user_id]["cost"] += float(record.cost)
                
                # Calculate daily breakdown
                daily_breakdown = {}
                for record in records:
                    day_str = record.billing_date.isoformat()
                    if day_str not in daily_breakdown:
                        daily_breakdown[day_str] = {
                            "annotations": 0,
                            "time_spent": 0,
                            "cost": 0.0
                        }
                    
                    daily_breakdown[day_str]["annotations"] += record.annotation_count
                    daily_breakdown[day_str]["time_spent"] += record.time_spent
                    daily_breakdown[day_str]["cost"] += float(record.cost)
                
                # Create report
                report = BillingReport(
                    tenant_id=tenant_id,
                    start_date=start_date,
                    end_date=end_date,
                    total_cost=total_cost,
                    total_annotations=total_annotations,
                    total_time_spent=total_time_spent,
                    user_breakdown=user_breakdown,
                    daily_breakdown=daily_breakdown
                )
                
                return report
                
        except Exception as e:
            print(f"Error generating billing report: {e}")
            # Return empty report on error
            return BillingReport(
                tenant_id=tenant_id,
                start_date=start_date,
                end_date=end_date
            )
    
    def get_tenant_billing_records(self, tenant_id: str, 
                                 start_date: Optional[date] = None,
                                 end_date: Optional[date] = None) -> List[BillingRecord]:
        """
        Get billing records for a tenant within a date range.
        
        Args:
            tenant_id: Tenant identifier
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            List of billing records
        """
        try:
            with get_db_session() as db:
                stmt = select(BillingRecordModel).where(
                    BillingRecordModel.tenant_id == tenant_id
                )
                
                if start_date:
                    stmt = stmt.where(BillingRecordModel.billing_date >= start_date)
                if end_date:
                    stmt = stmt.where(BillingRecordModel.billing_date <= end_date)
                
                records = db.execute(stmt).scalars().all()
                
                # Convert to Pydantic models
                billing_records = []
                for record in records:
                    billing_record = BillingRecord(
                        id=record.id,
                        tenant_id=record.tenant_id,
                        user_id=record.user_id,
                        task_id=record.task_id,
                        annotation_count=record.annotation_count,
                        time_spent=record.time_spent,
                        cost=Decimal(str(record.cost)),
                        billing_date=record.billing_date,
                        created_at=record.created_at
                    )
                    billing_records.append(billing_record)
                
                return billing_records
                
        except Exception as e:
            print(f"Error getting tenant billing records: {e}")
            return []
    
    def get_user_billing_summary(self, tenant_id: str, user_id: str, 
                               start_date: Optional[date] = None,
                               end_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Get billing summary for a specific user.
        
        Args:
            tenant_id: Tenant identifier
            user_id: User identifier
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            User billing summary
        """
        try:
            with get_db_session() as db:
                stmt = select(BillingRecordModel).where(
                    and_(
                        BillingRecordModel.tenant_id == tenant_id,
                        BillingRecordModel.user_id == user_id
                    )
                )
                
                if start_date:
                    stmt = stmt.where(BillingRecordModel.billing_date >= start_date)
                if end_date:
                    stmt = stmt.where(BillingRecordModel.billing_date <= end_date)
                
                records = db.execute(stmt).scalars().all()
                
                # Calculate summary
                total_annotations = sum(r.annotation_count for r in records)
                total_time_spent = sum(r.time_spent for r in records)
                total_cost = sum(Decimal(str(r.cost)) for r in records)
                
                return {
                    "user_id": user_id,
                    "total_annotations": total_annotations,
                    "total_time_spent": total_time_spent,
                    "total_cost": float(total_cost),
                    "record_count": len(records)
                }
                
        except Exception as e:
            print(f"Error getting user billing summary: {e}")
            return {
                "user_id": user_id,
                "total_annotations": 0,
                "total_time_spent": 0,
                "total_cost": 0.0,
                "record_count": 0
            }
    
    def export_billing_data(self, tenant_id: str, format_type: str = "json",
                          start_date: Optional[date] = None,
                          end_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Export billing data in specified format.
        
        Args:
            tenant_id: Tenant identifier
            format_type: Export format (json, csv)
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Export result with data and metadata
        """
        try:
            records = self.get_tenant_billing_records(tenant_id, start_date, end_date)
            
            if format_type.lower() == "json":
                data = [record.to_dict() for record in records]
            elif format_type.lower() == "csv":
                # Convert to CSV-friendly format
                data = []
                for record in records:
                    row = {
                        "id": str(record.id),
                        "tenant_id": record.tenant_id,
                        "user_id": record.user_id,
                        "task_id": str(record.task_id) if record.task_id else "",
                        "annotation_count": record.annotation_count,
                        "time_spent": record.time_spent,
                        "cost": float(record.cost),
                        "billing_date": record.billing_date.isoformat(),
                        "created_at": record.created_at.isoformat()
                    }
                    data.append(row)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            return {
                "format": format_type,
                "record_count": len(records),
                "data": data,
                "exported_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error exporting billing data: {e}")
            return {
                "format": format_type,
                "record_count": 0,
                "data": [],
                "error": str(e),
                "exported_at": datetime.now().isoformat()
            }