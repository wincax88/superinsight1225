"""
Billing data models for SuperInsight platform.

Represents billing records, bills, and billing reports.
"""

from datetime import datetime, date
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from decimal import Decimal


class BillingMode(str, Enum):
    """Enumeration of billing modes."""
    BY_COUNT = "by_count"  # 按标注条数计费
    BY_TIME = "by_time"  # 按工时计费
    BY_PROJECT = "by_project"  # 按项目包年计费
    HYBRID = "hybrid"  # 混合计费模式


class BillingRecord(BaseModel):
    """
    Billing record model representing a single billing entry.
    
    Tracks annotation work for billing purposes.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Unique billing record identifier")
    tenant_id: str = Field(..., description="Tenant identifier for multi-tenancy")
    user_id: str = Field(..., description="User who performed the work")
    task_id: Optional[UUID] = Field(None, description="Reference to annotation task")
    annotation_count: int = Field(default=0, description="Number of annotations completed")
    time_spent: int = Field(default=0, description="Time spent in seconds")
    cost: Decimal = Field(default=Decimal("0.00"), description="Calculated cost")
    billing_date: date = Field(default_factory=date.today, description="Date of billing")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    @field_validator('tenant_id', 'user_id')
    @classmethod
    def validate_not_empty(cls, v):
        """Validate that tenant_id and user_id are not empty."""
        if not v or not v.strip():
            raise ValueError('Field cannot be empty')
        return v
    
    @field_validator('annotation_count', 'time_spent')
    @classmethod
    def validate_non_negative(cls, v):
        """Validate that counts and time are non-negative."""
        if v < 0:
            raise ValueError('Value must be non-negative')
        return v
    
    @field_validator('cost')
    @classmethod
    def validate_cost(cls, v):
        """Validate that cost is non-negative."""
        if v < 0:
            raise ValueError('Cost must be non-negative')
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert billing record to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "task_id": str(self.task_id) if self.task_id else None,
            "annotation_count": self.annotation_count,
            "time_spent": self.time_spent,
            "cost": float(self.cost),
            "billing_date": self.billing_date.isoformat(),
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BillingRecord':
        """Create billing record from dictionary (JSON deserialization)."""
        # Convert string timestamp back to datetime object
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        # Convert string date back to date object
        if isinstance(data.get('billing_date'), str):
            data['billing_date'] = date.fromisoformat(data['billing_date'])
        
        # Convert string UUIDs back to UUID objects
        if isinstance(data.get('id'), str):
            data['id'] = UUID(data['id'])
        if data.get('task_id') and isinstance(data['task_id'], str):
            data['task_id'] = UUID(data['task_id'])
        
        # Convert cost to Decimal
        if isinstance(data.get('cost'), (int, float)):
            data['cost'] = Decimal(str(data['cost']))
            
        return cls(**data)
    
    model_config = ConfigDict(
        json_encoders={
            UUID: str,
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
            Decimal: float
        }
    )


class BillingRule(BaseModel):
    """
    Billing rule configuration.
    
    Defines how to calculate costs for different billing modes.
    """
    
    tenant_id: str = Field(..., description="Tenant identifier")
    billing_mode: BillingMode = Field(default=BillingMode.BY_COUNT, description="Billing mode")
    rate_per_annotation: Decimal = Field(default=Decimal("0.10"), description="Cost per annotation")
    rate_per_hour: Decimal = Field(default=Decimal("50.00"), description="Cost per hour")
    project_annual_fee: Decimal = Field(default=Decimal("10000.00"), description="Annual project fee")
    
    @field_validator('rate_per_annotation', 'rate_per_hour', 'project_annual_fee')
    @classmethod
    def validate_rates(cls, v):
        """Validate that rates are non-negative."""
        if v < 0:
            raise ValueError('Rate must be non-negative')
        return v
    
    def calculate_cost(self, annotation_count: int, time_spent: int) -> Decimal:
        """
        Calculate cost based on billing mode.
        
        Args:
            annotation_count: Number of annotations
            time_spent: Time spent in seconds
            
        Returns:
            Calculated cost as Decimal
        """
        if self.billing_mode == BillingMode.BY_COUNT:
            return self.rate_per_annotation * annotation_count
        elif self.billing_mode == BillingMode.BY_TIME:
            hours = Decimal(str(time_spent)) / Decimal("3600")
            return self.rate_per_hour * hours
        elif self.billing_mode == BillingMode.HYBRID:
            count_cost = self.rate_per_annotation * annotation_count
            hours = Decimal(str(time_spent)) / Decimal("3600")
            time_cost = self.rate_per_hour * hours
            return count_cost + time_cost
        else:  # BY_PROJECT
            return Decimal("0.00")  # Project billing is handled separately
    
    model_config = ConfigDict(
        json_encoders={
            Decimal: float
        }
    )


class Bill(BaseModel):
    """
    Monthly bill model.
    
    Represents a monthly billing statement for a tenant.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Unique bill identifier")
    tenant_id: str = Field(..., description="Tenant identifier")
    billing_period: str = Field(..., description="Billing period (YYYY-MM)")
    total_annotations: int = Field(default=0, description="Total annotations in period")
    total_time_spent: int = Field(default=0, description="Total time spent in seconds")
    total_cost: Decimal = Field(default=Decimal("0.00"), description="Total cost")
    billing_records: List[UUID] = Field(default_factory=list, description="Associated billing record IDs")
    generated_at: datetime = Field(default_factory=datetime.now, description="Bill generation timestamp")
    
    @field_validator('billing_period')
    @classmethod
    def validate_billing_period(cls, v):
        """Validate billing period format (YYYY-MM)."""
        try:
            datetime.strptime(v, "%Y-%m")
        except ValueError:
            raise ValueError('Billing period must be in YYYY-MM format')
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert bill to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "tenant_id": self.tenant_id,
            "billing_period": self.billing_period,
            "total_annotations": self.total_annotations,
            "total_time_spent": self.total_time_spent,
            "total_cost": float(self.total_cost),
            "billing_records": [str(rid) for rid in self.billing_records],
            "generated_at": self.generated_at.isoformat()
        }
    
    model_config = ConfigDict(
        json_encoders={
            UUID: str,
            datetime: lambda v: v.isoformat(),
            Decimal: float
        }
    )


class BillingReport(BaseModel):
    """
    Billing report model for analysis and trends.
    
    Provides aggregated billing data for a specific period.
    """
    
    tenant_id: str = Field(..., description="Tenant identifier")
    start_date: date = Field(..., description="Report start date")
    end_date: date = Field(..., description="Report end date")
    total_cost: Decimal = Field(default=Decimal("0.00"), description="Total cost in period")
    total_annotations: int = Field(default=0, description="Total annotations")
    total_time_spent: int = Field(default=0, description="Total time in seconds")
    user_breakdown: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Per-user statistics")
    daily_breakdown: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Daily statistics")
    generated_at: datetime = Field(default_factory=datetime.now, description="Report generation timestamp")
    
    @model_validator(mode='after')
    def validate_date_range(self):
        """Validate that end_date is after start_date."""
        if self.end_date < self.start_date:
            raise ValueError('end_date must be after start_date')
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "tenant_id": self.tenant_id,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "total_cost": float(self.total_cost),
            "total_annotations": self.total_annotations,
            "total_time_spent": self.total_time_spent,
            "user_breakdown": self.user_breakdown,
            "daily_breakdown": self.daily_breakdown,
            "generated_at": self.generated_at.isoformat()
        }
    
    model_config = ConfigDict(
        json_encoders={
            date: lambda v: v.isoformat(),
            datetime: lambda v: v.isoformat(),
            Decimal: float
        }
    )
