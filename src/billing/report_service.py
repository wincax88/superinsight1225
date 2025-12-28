"""
Enhanced billing report service for SuperInsight Platform.

Provides:
- Project-based cost breakdown
- Department-based cost allocation
- Work hours statistics reports
- Billing rule versioning
- Excel export capabilities
"""

import logging
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID, uuid4
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict
from enum import Enum

from pydantic import BaseModel, Field

from src.billing.models import BillingRecord, Bill, BillingReport, BillingMode
from src.billing.service import BillingSystem

logger = logging.getLogger(__name__)


class ReportType(str, Enum):
    """Types of billing reports."""
    SUMMARY = "summary"
    DETAILED = "detailed"
    USER_BREAKDOWN = "user_breakdown"
    PROJECT_BREAKDOWN = "project_breakdown"
    DEPARTMENT_BREAKDOWN = "department_breakdown"
    WORK_HOURS = "work_hours"
    TREND_ANALYSIS = "trend_analysis"


class BillingRuleVersion(BaseModel):
    """Versioned billing rule for audit trail."""
    id: UUID = Field(default_factory=uuid4)
    tenant_id: str
    version: int = 1
    billing_mode: BillingMode = BillingMode.BY_COUNT
    rate_per_annotation: Decimal = Decimal("0.10")
    rate_per_hour: Decimal = Decimal("50.00")
    project_annual_fee: Decimal = Decimal("10000.00")
    effective_from: datetime
    effective_to: Optional[datetime] = None
    created_by: str = "system"
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "tenant_id": self.tenant_id,
            "version": self.version,
            "billing_mode": self.billing_mode.value,
            "rate_per_annotation": float(self.rate_per_annotation),
            "rate_per_hour": float(self.rate_per_hour),
            "project_annual_fee": float(self.project_annual_fee),
            "effective_from": self.effective_from.isoformat(),
            "effective_to": self.effective_to.isoformat() if self.effective_to else None,
            "created_by": self.created_by,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "is_active": self.is_active,
            "metadata": self.metadata
        }


class ProjectCostBreakdown(BaseModel):
    """Cost breakdown by project."""
    project_id: str
    project_name: str
    total_cost: Decimal = Decimal("0.00")
    total_annotations: int = 0
    total_time_spent: int = 0
    user_count: int = 0
    avg_cost_per_annotation: Decimal = Decimal("0.00")
    percentage_of_total: float = 0.0


class DepartmentCostAllocation(BaseModel):
    """Cost allocation by department."""
    department_id: str
    department_name: str
    total_cost: Decimal = Decimal("0.00")
    projects: List[str] = Field(default_factory=list)
    user_count: int = 0
    percentage_of_total: float = 0.0


class WorkHoursStatistics(BaseModel):
    """Work hours statistics for reporting."""
    user_id: str
    user_name: Optional[str] = None
    total_hours: float = 0.0
    billable_hours: float = 0.0
    total_annotations: int = 0
    annotations_per_hour: float = 0.0
    total_cost: Decimal = Decimal("0.00")
    daily_breakdown: Dict[str, float] = Field(default_factory=dict)
    efficiency_score: float = 0.0


class EnhancedBillingReport(BaseModel):
    """Enhanced billing report with advanced analytics."""
    id: UUID = Field(default_factory=uuid4)
    tenant_id: str
    report_type: ReportType
    start_date: date
    end_date: date

    # Summary metrics
    total_cost: Decimal = Decimal("0.00")
    total_annotations: int = 0
    total_time_spent: int = 0

    # Breakdowns
    user_breakdown: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    project_breakdown: List[Dict[str, Any]] = Field(default_factory=list)
    department_breakdown: List[Dict[str, Any]] = Field(default_factory=list)
    daily_breakdown: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    work_hours_statistics: List[Dict[str, Any]] = Field(default_factory=list)

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.now)
    generated_by: str = "system"
    billing_rule_version: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "tenant_id": self.tenant_id,
            "report_type": self.report_type.value,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "total_cost": float(self.total_cost),
            "total_annotations": self.total_annotations,
            "total_time_spent": self.total_time_spent,
            "user_breakdown": self.user_breakdown,
            "project_breakdown": self.project_breakdown,
            "department_breakdown": self.department_breakdown,
            "daily_breakdown": self.daily_breakdown,
            "work_hours_statistics": self.work_hours_statistics,
            "generated_at": self.generated_at.isoformat(),
            "generated_by": self.generated_by,
            "billing_rule_version": self.billing_rule_version
        }


class BillingReportService:
    """
    Enhanced billing report service.

    Provides comprehensive billing reports with:
    - Project-based cost breakdown
    - Department-based cost allocation
    - Work hours statistics
    - Billing rule versioning
    """

    def __init__(self, billing_system: Optional[BillingSystem] = None):
        """Initialize report service."""
        self.billing_system = billing_system or BillingSystem()
        self._rule_versions: Dict[str, List[BillingRuleVersion]] = defaultdict(list)
        self._project_mappings: Dict[str, Dict[str, str]] = {}  # tenant_id -> {task_id: project_id}
        self._department_mappings: Dict[str, Dict[str, str]] = {}  # tenant_id -> {project_id: department_id}
        self._user_departments: Dict[str, Dict[str, str]] = {}  # tenant_id -> {user_id: department_id}

    def configure_project_mapping(
        self,
        tenant_id: str,
        mappings: Dict[str, str]
    ) -> None:
        """
        Configure task-to-project mappings for cost breakdown.

        Args:
            tenant_id: Tenant identifier
            mappings: Dictionary mapping task_id to project_id
        """
        self._project_mappings[tenant_id] = mappings
        logger.info(f"Configured project mappings for tenant {tenant_id}: {len(mappings)} mappings")

    def configure_department_mapping(
        self,
        tenant_id: str,
        project_mappings: Dict[str, str],
        user_mappings: Dict[str, str]
    ) -> None:
        """
        Configure department mappings for cost allocation.

        Args:
            tenant_id: Tenant identifier
            project_mappings: Dictionary mapping project_id to department_id
            user_mappings: Dictionary mapping user_id to department_id
        """
        self._department_mappings[tenant_id] = project_mappings
        self._user_departments[tenant_id] = user_mappings
        logger.info(f"Configured department mappings for tenant {tenant_id}")

    def create_billing_rule_version(
        self,
        tenant_id: str,
        billing_mode: BillingMode,
        rate_per_annotation: Decimal,
        rate_per_hour: Decimal,
        created_by: str,
        effective_from: Optional[datetime] = None,
        project_annual_fee: Optional[Decimal] = None
    ) -> BillingRuleVersion:
        """
        Create a new versioned billing rule.

        Args:
            tenant_id: Tenant identifier
            billing_mode: Billing mode
            rate_per_annotation: Rate per annotation
            rate_per_hour: Rate per hour
            created_by: User who created the rule
            effective_from: When the rule becomes effective
            project_annual_fee: Annual project fee (optional)

        Returns:
            Created billing rule version
        """
        # Deactivate previous versions
        existing_versions = self._rule_versions.get(tenant_id, [])
        new_version = len(existing_versions) + 1

        if existing_versions:
            for v in existing_versions:
                if v.is_active:
                    v.is_active = False
                    v.effective_to = effective_from or datetime.now()

        rule = BillingRuleVersion(
            tenant_id=tenant_id,
            version=new_version,
            billing_mode=billing_mode,
            rate_per_annotation=rate_per_annotation,
            rate_per_hour=rate_per_hour,
            project_annual_fee=project_annual_fee or Decimal("10000.00"),
            effective_from=effective_from or datetime.now(),
            created_by=created_by,
            is_active=True
        )

        self._rule_versions[tenant_id].append(rule)
        logger.info(f"Created billing rule version {new_version} for tenant {tenant_id}")

        return rule

    def approve_billing_rule(
        self,
        tenant_id: str,
        version: int,
        approved_by: str
    ) -> Optional[BillingRuleVersion]:
        """
        Approve a billing rule version.

        Args:
            tenant_id: Tenant identifier
            version: Rule version to approve
            approved_by: User approving the rule

        Returns:
            Approved billing rule or None if not found
        """
        versions = self._rule_versions.get(tenant_id, [])
        for rule in versions:
            if rule.version == version:
                rule.approved_by = approved_by
                rule.approved_at = datetime.now()
                logger.info(f"Approved billing rule version {version} for tenant {tenant_id}")
                return rule
        return None

    def get_billing_rule_history(
        self,
        tenant_id: str
    ) -> List[BillingRuleVersion]:
        """
        Get billing rule version history for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            List of billing rule versions
        """
        return self._rule_versions.get(tenant_id, [])

    def get_active_billing_rule(
        self,
        tenant_id: str
    ) -> Optional[BillingRuleVersion]:
        """
        Get the currently active billing rule for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Active billing rule or None
        """
        versions = self._rule_versions.get(tenant_id, [])
        for rule in reversed(versions):
            if rule.is_active:
                return rule
        return None

    def generate_work_hours_report(
        self,
        tenant_id: str,
        start_date: date,
        end_date: date,
        user_names: Optional[Dict[str, str]] = None
    ) -> List[WorkHoursStatistics]:
        """
        Generate work hours statistics for all users.

        Args:
            tenant_id: Tenant identifier
            start_date: Report start date
            end_date: Report end date
            user_names: Optional mapping of user_id to display name

        Returns:
            List of work hours statistics per user
        """
        report = self.billing_system.generate_report(tenant_id, start_date, end_date)
        user_names = user_names or {}

        statistics = []

        for user_id, stats in report.user_breakdown.items():
            total_seconds = stats.get("time_spent", 0)
            total_hours = total_seconds / 3600.0
            total_annotations = stats.get("annotations", 0)
            total_cost = Decimal(str(stats.get("cost", 0)))

            annotations_per_hour = (
                total_annotations / total_hours
                if total_hours > 0 else 0
            )

            # Calculate efficiency score (annotations per hour normalized to 0-100)
            efficiency_score = min(100.0, annotations_per_hour * 2)

            user_stats = WorkHoursStatistics(
                user_id=user_id,
                user_name=user_names.get(user_id),
                total_hours=round(total_hours, 2),
                billable_hours=round(total_hours, 2),  # All hours are billable
                total_annotations=total_annotations,
                annotations_per_hour=round(annotations_per_hour, 2),
                total_cost=total_cost,
                efficiency_score=round(efficiency_score, 1)
            )

            statistics.append(user_stats)

        # Sort by efficiency score descending
        statistics.sort(key=lambda x: x.efficiency_score, reverse=True)

        return statistics

    def generate_project_cost_breakdown(
        self,
        tenant_id: str,
        start_date: date,
        end_date: date,
        project_names: Optional[Dict[str, str]] = None
    ) -> List[ProjectCostBreakdown]:
        """
        Generate cost breakdown by project.

        Args:
            tenant_id: Tenant identifier
            start_date: Report start date
            end_date: Report end date
            project_names: Optional mapping of project_id to display name

        Returns:
            List of project cost breakdowns
        """
        records = self.billing_system.get_tenant_billing_records(
            tenant_id, start_date, end_date
        )

        project_mappings = self._project_mappings.get(tenant_id, {})
        project_names = project_names or {}

        # Aggregate by project
        project_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "cost": Decimal("0.00"),
            "annotations": 0,
            "time_spent": 0,
            "users": set()
        })

        total_cost = Decimal("0.00")

        for record in records:
            task_id = str(record.task_id) if record.task_id else "unassigned"
            project_id = project_mappings.get(task_id, "default")

            project_data[project_id]["cost"] += record.cost
            project_data[project_id]["annotations"] += record.annotation_count
            project_data[project_id]["time_spent"] += record.time_spent
            project_data[project_id]["users"].add(record.user_id)

            total_cost += record.cost

        # Build breakdown list
        breakdowns = []
        for project_id, data in project_data.items():
            avg_cost = (
                data["cost"] / data["annotations"]
                if data["annotations"] > 0 else Decimal("0.00")
            )

            percentage = (
                float(data["cost"] / total_cost * 100)
                if total_cost > 0 else 0.0
            )

            breakdown = ProjectCostBreakdown(
                project_id=project_id,
                project_name=project_names.get(project_id, project_id),
                total_cost=data["cost"],
                total_annotations=data["annotations"],
                total_time_spent=data["time_spent"],
                user_count=len(data["users"]),
                avg_cost_per_annotation=avg_cost.quantize(Decimal("0.01"), ROUND_HALF_UP),
                percentage_of_total=round(percentage, 2)
            )
            breakdowns.append(breakdown)

        # Sort by cost descending
        breakdowns.sort(key=lambda x: x.total_cost, reverse=True)

        return breakdowns

    def generate_department_cost_allocation(
        self,
        tenant_id: str,
        start_date: date,
        end_date: date,
        department_names: Optional[Dict[str, str]] = None
    ) -> List[DepartmentCostAllocation]:
        """
        Generate cost allocation by department.

        Args:
            tenant_id: Tenant identifier
            start_date: Report start date
            end_date: Report end date
            department_names: Optional mapping of department_id to display name

        Returns:
            List of department cost allocations
        """
        records = self.billing_system.get_tenant_billing_records(
            tenant_id, start_date, end_date
        )

        user_depts = self._user_departments.get(tenant_id, {})
        department_names = department_names or {}

        # Aggregate by department
        dept_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "cost": Decimal("0.00"),
            "projects": set(),
            "users": set()
        })

        total_cost = Decimal("0.00")
        project_mappings = self._project_mappings.get(tenant_id, {})

        for record in records:
            dept_id = user_depts.get(record.user_id, "default")
            task_id = str(record.task_id) if record.task_id else "unassigned"
            project_id = project_mappings.get(task_id, "default")

            dept_data[dept_id]["cost"] += record.cost
            dept_data[dept_id]["projects"].add(project_id)
            dept_data[dept_id]["users"].add(record.user_id)

            total_cost += record.cost

        # Build allocation list
        allocations = []
        for dept_id, data in dept_data.items():
            percentage = (
                float(data["cost"] / total_cost * 100)
                if total_cost > 0 else 0.0
            )

            allocation = DepartmentCostAllocation(
                department_id=dept_id,
                department_name=department_names.get(dept_id, dept_id),
                total_cost=data["cost"],
                projects=list(data["projects"]),
                user_count=len(data["users"]),
                percentage_of_total=round(percentage, 2)
            )
            allocations.append(allocation)

        # Sort by cost descending
        allocations.sort(key=lambda x: x.total_cost, reverse=True)

        return allocations

    def generate_enhanced_report(
        self,
        tenant_id: str,
        start_date: date,
        end_date: date,
        report_type: ReportType = ReportType.DETAILED,
        user_names: Optional[Dict[str, str]] = None,
        project_names: Optional[Dict[str, str]] = None,
        department_names: Optional[Dict[str, str]] = None,
        generated_by: str = "system"
    ) -> EnhancedBillingReport:
        """
        Generate comprehensive enhanced billing report.

        Args:
            tenant_id: Tenant identifier
            start_date: Report start date
            end_date: Report end date
            report_type: Type of report to generate
            user_names: Optional user name mappings
            project_names: Optional project name mappings
            department_names: Optional department name mappings
            generated_by: User generating the report

        Returns:
            Enhanced billing report
        """
        # Get base report
        base_report = self.billing_system.generate_report(
            tenant_id, start_date, end_date
        )

        # Get active rule version
        active_rule = self.get_active_billing_rule(tenant_id)
        rule_version = active_rule.version if active_rule else None

        # Generate breakdowns based on report type
        project_breakdown = []
        department_breakdown = []
        work_hours_stats = []

        if report_type in [ReportType.DETAILED, ReportType.PROJECT_BREAKDOWN]:
            project_breakdowns = self.generate_project_cost_breakdown(
                tenant_id, start_date, end_date, project_names
            )
            project_breakdown = [
                {
                    "project_id": pb.project_id,
                    "project_name": pb.project_name,
                    "total_cost": float(pb.total_cost),
                    "total_annotations": pb.total_annotations,
                    "total_time_spent": pb.total_time_spent,
                    "user_count": pb.user_count,
                    "avg_cost_per_annotation": float(pb.avg_cost_per_annotation),
                    "percentage_of_total": pb.percentage_of_total
                }
                for pb in project_breakdowns
            ]

        if report_type in [ReportType.DETAILED, ReportType.DEPARTMENT_BREAKDOWN]:
            dept_allocations = self.generate_department_cost_allocation(
                tenant_id, start_date, end_date, department_names
            )
            department_breakdown = [
                {
                    "department_id": da.department_id,
                    "department_name": da.department_name,
                    "total_cost": float(da.total_cost),
                    "projects": da.projects,
                    "user_count": da.user_count,
                    "percentage_of_total": da.percentage_of_total
                }
                for da in dept_allocations
            ]

        if report_type in [ReportType.DETAILED, ReportType.WORK_HOURS]:
            work_hours = self.generate_work_hours_report(
                tenant_id, start_date, end_date, user_names
            )
            work_hours_stats = [
                {
                    "user_id": wh.user_id,
                    "user_name": wh.user_name,
                    "total_hours": wh.total_hours,
                    "billable_hours": wh.billable_hours,
                    "total_annotations": wh.total_annotations,
                    "annotations_per_hour": wh.annotations_per_hour,
                    "total_cost": float(wh.total_cost),
                    "efficiency_score": wh.efficiency_score
                }
                for wh in work_hours
            ]

        # Create enhanced report
        report = EnhancedBillingReport(
            tenant_id=tenant_id,
            report_type=report_type,
            start_date=start_date,
            end_date=end_date,
            total_cost=base_report.total_cost,
            total_annotations=base_report.total_annotations,
            total_time_spent=base_report.total_time_spent,
            user_breakdown=base_report.user_breakdown,
            project_breakdown=project_breakdown,
            department_breakdown=department_breakdown,
            daily_breakdown=base_report.daily_breakdown,
            work_hours_statistics=work_hours_stats,
            generated_by=generated_by,
            billing_rule_version=rule_version
        )

        logger.info(
            f"Generated {report_type.value} report for tenant {tenant_id}: "
            f"{start_date} to {end_date}"
        )

        return report

    def export_to_excel_data(
        self,
        report: EnhancedBillingReport
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Prepare report data for Excel export.

        Args:
            report: Enhanced billing report

        Returns:
            Dictionary with sheet names as keys and data rows as values
        """
        sheets = {}

        # Summary sheet
        sheets["Summary"] = [{
            "Metric": "Total Cost",
            "Value": float(report.total_cost)
        }, {
            "Metric": "Total Annotations",
            "Value": report.total_annotations
        }, {
            "Metric": "Total Time (hours)",
            "Value": round(report.total_time_spent / 3600, 2)
        }, {
            "Metric": "Report Period",
            "Value": f"{report.start_date} to {report.end_date}"
        }, {
            "Metric": "Generated At",
            "Value": report.generated_at.isoformat()
        }]

        # User breakdown sheet
        if report.user_breakdown:
            sheets["User Breakdown"] = [
                {
                    "User ID": user_id,
                    "Annotations": stats["annotations"],
                    "Time (hours)": round(stats["time_spent"] / 3600, 2),
                    "Cost": stats["cost"]
                }
                for user_id, stats in report.user_breakdown.items()
            ]

        # Project breakdown sheet
        if report.project_breakdown:
            sheets["Project Breakdown"] = report.project_breakdown

        # Department breakdown sheet
        if report.department_breakdown:
            sheets["Department Breakdown"] = [
                {
                    "Department": d["department_name"],
                    "Total Cost": d["total_cost"],
                    "User Count": d["user_count"],
                    "Percentage": f"{d['percentage_of_total']}%"
                }
                for d in report.department_breakdown
            ]

        # Work hours statistics sheet
        if report.work_hours_statistics:
            sheets["Work Hours"] = [
                {
                    "User": wh.get("user_name") or wh["user_id"],
                    "Total Hours": wh["total_hours"],
                    "Billable Hours": wh["billable_hours"],
                    "Annotations": wh["total_annotations"],
                    "Rate (ann/hr)": wh["annotations_per_hour"],
                    "Cost": wh["total_cost"],
                    "Efficiency Score": wh["efficiency_score"]
                }
                for wh in report.work_hours_statistics
            ]

        # Daily breakdown sheet
        if report.daily_breakdown:
            sheets["Daily Breakdown"] = [
                {
                    "Date": day,
                    "Annotations": stats["annotations"],
                    "Time (hours)": round(stats["time_spent"] / 3600, 2),
                    "Cost": stats["cost"]
                }
                for day, stats in sorted(report.daily_breakdown.items())
            ]

        return sheets


# Singleton instance for easy access
_report_service: Optional[BillingReportService] = None


def get_billing_report_service() -> BillingReportService:
    """Get or create the billing report service singleton."""
    global _report_service
    if _report_service is None:
        _report_service = BillingReportService()
    return _report_service
