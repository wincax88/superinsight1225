"""
SLA monitoring and alerting for ticket management.

Provides:
- SLA violation detection
- Automatic escalation
- Alert notifications
- SLA compliance reporting
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from uuid import UUID
from enum import Enum

from sqlalchemy import select, and_, or_
from sqlalchemy.orm import Session

from src.database.connection import db_manager
from src.ticket.models import (
    TicketModel,
    TicketHistoryModel,
    TicketStatus,
    TicketPriority,
    SLAConfig,
)

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of SLA alerts."""
    SLA_WARNING = "sla_warning"          # Approaching SLA deadline
    SLA_BREACH = "sla_breach"            # SLA deadline exceeded
    ESCALATION = "escalation"            # Ticket escalated
    WORKLOAD_HIGH = "workload_high"      # High workload detected
    UNASSIGNED = "unassigned"            # Ticket unassigned too long


class SLAAlert:
    """Represents an SLA alert."""

    def __init__(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        ticket_id: UUID,
        message: str,
        details: Dict[str, Any] = None
    ):
        self.id = UUID(int=0)  # Will be assigned by storage
        self.alert_type = alert_type
        self.severity = severity
        self.ticket_id = ticket_id
        self.message = message
        self.details = details or {}
        self.created_at = datetime.now()
        self.acknowledged = False
        self.acknowledged_by: Optional[str] = None
        self.acknowledged_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "id": str(self.id),
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "ticket_id": str(self.ticket_id),
            "message": self.message,
            "details": self.details,
            "created_at": self.created_at.isoformat(),
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
        }


class SLAMonitor:
    """
    SLA monitoring and alerting system.

    Monitors ticket SLA compliance and triggers alerts/escalations.
    """

    # SLA configuration (standard mode - user confirmed)
    SLA_CONFIG = {
        TicketPriority.CRITICAL: 4 * 3600,   # 4 hours
        TicketPriority.HIGH: 8 * 3600,       # 8 hours
        TicketPriority.MEDIUM: 24 * 3600,    # 24 hours
        TicketPriority.LOW: 72 * 3600,       # 72 hours
    }

    # Warning threshold (percentage of SLA time remaining)
    WARNING_THRESHOLD = 0.25  # 25%

    # Unassigned timeout (seconds before alerting unassigned tickets)
    UNASSIGNED_TIMEOUT = {
        TicketPriority.CRITICAL: 15 * 60,    # 15 minutes
        TicketPriority.HIGH: 30 * 60,        # 30 minutes
        TicketPriority.MEDIUM: 60 * 60,      # 1 hour
        TicketPriority.LOW: 2 * 60 * 60,     # 2 hours
    }

    def __init__(self):
        """Initialize the SLA monitor."""
        self._alerts: List[SLAAlert] = []
        self._last_check: Optional[datetime] = None

    async def check_sla_violations(
        self,
        tenant_id: Optional[str] = None
    ) -> List[TicketModel]:
        """
        Check for SLA violations across all active tickets.

        Args:
            tenant_id: Optional tenant filter

        Returns:
            List of tickets with SLA violations
        """
        violations = []

        try:
            with db_manager.get_session() as session:
                # Query active tickets with SLA deadlines
                query = select(TicketModel).where(
                    and_(
                        TicketModel.status.in_([
                            TicketStatus.OPEN,
                            TicketStatus.ASSIGNED,
                            TicketStatus.IN_PROGRESS,
                            TicketStatus.ESCALATED,
                        ]),
                        TicketModel.sla_deadline.isnot(None)
                    )
                )

                if tenant_id:
                    query = query.where(TicketModel.tenant_id == tenant_id)

                tickets = session.execute(query).scalars().all()
                now = datetime.now()

                for ticket in tickets:
                    if ticket.sla_deadline < now and not ticket.sla_breached:
                        # Mark as breached
                        ticket.sla_breached = True
                        ticket.updated_at = now

                        # Record history
                        history = TicketHistoryModel(
                            ticket_id=ticket.id,
                            action="sla_breached",
                            old_value="false",
                            new_value="true",
                            performed_by="system",
                            notes=f"SLA deadline exceeded at {now.isoformat()}"
                        )
                        session.add(history)

                        violations.append(ticket)

                        # Create alert
                        alert = SLAAlert(
                            alert_type=AlertType.SLA_BREACH,
                            severity=AlertSeverity.CRITICAL,
                            ticket_id=ticket.id,
                            message=f"SLA breached for ticket: {ticket.title}",
                            details={
                                "priority": ticket.priority.value,
                                "sla_deadline": ticket.sla_deadline.isoformat(),
                                "assigned_to": ticket.assigned_to,
                                "overdue_seconds": int((now - ticket.sla_deadline).total_seconds()),
                            }
                        )
                        self._alerts.append(alert)

                        logger.warning(f"SLA breached for ticket {ticket.id}")

                session.commit()
                self._last_check = now

        except Exception as e:
            logger.error(f"Error checking SLA violations: {e}")

        return violations

    async def check_sla_warnings(
        self,
        tenant_id: Optional[str] = None
    ) -> List[TicketModel]:
        """
        Check for tickets approaching SLA deadline.

        Args:
            tenant_id: Optional tenant filter

        Returns:
            List of tickets needing warnings
        """
        warnings = []

        try:
            with db_manager.get_session() as session:
                query = select(TicketModel).where(
                    and_(
                        TicketModel.status.in_([
                            TicketStatus.OPEN,
                            TicketStatus.ASSIGNED,
                            TicketStatus.IN_PROGRESS,
                        ]),
                        TicketModel.sla_deadline.isnot(None),
                        TicketModel.sla_breached == False
                    )
                )

                if tenant_id:
                    query = query.where(TicketModel.tenant_id == tenant_id)

                tickets = session.execute(query).scalars().all()
                now = datetime.now()

                for ticket in tickets:
                    if SLAConfig.needs_warning(ticket.sla_deadline, ticket.priority):
                        remaining = SLAConfig.get_time_remaining(ticket.sla_deadline)
                        warnings.append(ticket)

                        # Create warning alert
                        alert = SLAAlert(
                            alert_type=AlertType.SLA_WARNING,
                            severity=AlertSeverity.WARNING,
                            ticket_id=ticket.id,
                            message=f"SLA warning: {remaining // 60} minutes remaining",
                            details={
                                "priority": ticket.priority.value,
                                "sla_deadline": ticket.sla_deadline.isoformat(),
                                "remaining_seconds": remaining,
                                "assigned_to": ticket.assigned_to,
                            }
                        )
                        self._alerts.append(alert)

        except Exception as e:
            logger.error(f"Error checking SLA warnings: {e}")

        return warnings

    async def check_unassigned_tickets(
        self,
        tenant_id: Optional[str] = None
    ) -> List[TicketModel]:
        """
        Check for tickets that have been unassigned too long.

        Args:
            tenant_id: Optional tenant filter

        Returns:
            List of unassigned tickets needing attention
        """
        unassigned = []

        try:
            with db_manager.get_session() as session:
                query = select(TicketModel).where(
                    and_(
                        TicketModel.status == TicketStatus.OPEN,
                        TicketModel.assigned_to.is_(None)
                    )
                )

                if tenant_id:
                    query = query.where(TicketModel.tenant_id == tenant_id)

                tickets = session.execute(query).scalars().all()
                now = datetime.now()

                for ticket in tickets:
                    timeout = self.UNASSIGNED_TIMEOUT.get(
                        ticket.priority,
                        self.UNASSIGNED_TIMEOUT[TicketPriority.MEDIUM]
                    )
                    age = (now - ticket.created_at).total_seconds()

                    if age > timeout:
                        unassigned.append(ticket)

                        alert = SLAAlert(
                            alert_type=AlertType.UNASSIGNED,
                            severity=AlertSeverity.WARNING,
                            ticket_id=ticket.id,
                            message=f"Ticket unassigned for {int(age // 60)} minutes",
                            details={
                                "priority": ticket.priority.value,
                                "age_seconds": int(age),
                                "timeout_seconds": timeout,
                            }
                        )
                        self._alerts.append(alert)

        except Exception as e:
            logger.error(f"Error checking unassigned tickets: {e}")

        return unassigned

    async def escalate_ticket(
        self,
        ticket_id: UUID,
        reason: str = "SLA escalation",
        escalate_by: str = "system"
    ) -> bool:
        """
        Escalate a ticket to higher priority/attention.

        Args:
            ticket_id: UUID of ticket to escalate
            reason: Reason for escalation
            escalate_by: User performing escalation

        Returns:
            True if escalation successful
        """
        try:
            with db_manager.get_session() as session:
                ticket = session.execute(
                    select(TicketModel).where(TicketModel.id == ticket_id)
                ).scalar_one_or_none()

                if not ticket:
                    logger.error(f"Ticket not found: {ticket_id}")
                    return False

                old_level = ticket.escalation_level
                old_status = ticket.status

                # Increment escalation level
                ticket.escalation_level += 1
                ticket.status = TicketStatus.ESCALATED
                ticket.updated_at = datetime.now()

                # Record history
                history = TicketHistoryModel(
                    ticket_id=ticket.id,
                    action="escalated",
                    old_value=str(old_level),
                    new_value=str(ticket.escalation_level),
                    performed_by=escalate_by,
                    notes=reason
                )
                session.add(history)

                if old_status != TicketStatus.ESCALATED:
                    status_history = TicketHistoryModel(
                        ticket_id=ticket.id,
                        action="status_changed",
                        old_value=old_status.value,
                        new_value=TicketStatus.ESCALATED.value,
                        performed_by=escalate_by,
                        notes="Status changed due to escalation"
                    )
                    session.add(status_history)

                session.commit()

                # Create escalation alert
                alert = SLAAlert(
                    alert_type=AlertType.ESCALATION,
                    severity=AlertSeverity.CRITICAL,
                    ticket_id=ticket.id,
                    message=f"Ticket escalated to level {ticket.escalation_level}",
                    details={
                        "reason": reason,
                        "escalated_by": escalate_by,
                        "previous_level": old_level,
                        "new_level": ticket.escalation_level,
                    }
                )
                self._alerts.append(alert)

                logger.info(f"Ticket {ticket_id} escalated to level {ticket.escalation_level}")
                return True

        except Exception as e:
            logger.error(f"Error escalating ticket {ticket_id}: {e}")
            return False

    async def auto_escalate_breached(
        self,
        tenant_id: Optional[str] = None
    ) -> List[UUID]:
        """
        Automatically escalate all breached tickets.

        Args:
            tenant_id: Optional tenant filter

        Returns:
            List of escalated ticket IDs
        """
        escalated = []

        try:
            with db_manager.get_session() as session:
                query = select(TicketModel).where(
                    and_(
                        TicketModel.sla_breached == True,
                        TicketModel.status != TicketStatus.ESCALATED,
                        TicketModel.status.in_([
                            TicketStatus.OPEN,
                            TicketStatus.ASSIGNED,
                            TicketStatus.IN_PROGRESS,
                        ])
                    )
                )

                if tenant_id:
                    query = query.where(TicketModel.tenant_id == tenant_id)

                tickets = session.execute(query).scalars().all()

                for ticket in tickets:
                    success = await self.escalate_ticket(
                        ticket.id,
                        reason="Auto-escalation due to SLA breach",
                        escalate_by="system"
                    )
                    if success:
                        escalated.append(ticket.id)

        except Exception as e:
            logger.error(f"Error auto-escalating breached tickets: {e}")

        return escalated

    async def send_sla_alert(
        self,
        ticket: TicketModel,
        alert_type: AlertType,
        channels: List[str] = None
    ) -> bool:
        """
        Send SLA alert notification.

        Args:
            ticket: Ticket that triggered alert
            alert_type: Type of alert
            channels: Notification channels (e.g., ["email", "wechat"])

        Returns:
            True if alert sent successfully
        """
        channels = channels or ["email"]

        try:
            # Build alert message
            if alert_type == AlertType.SLA_BREACH:
                subject = f"[URGENT] SLA Breach: {ticket.title}"
                message = f"Ticket {ticket.id} has breached its SLA deadline."
            elif alert_type == AlertType.SLA_WARNING:
                remaining = SLAConfig.get_time_remaining(ticket.sla_deadline)
                subject = f"[WARNING] SLA Warning: {ticket.title}"
                message = f"Ticket {ticket.id} has {remaining // 60} minutes until SLA deadline."
            elif alert_type == AlertType.ESCALATION:
                subject = f"[ESCALATED] Ticket Escalated: {ticket.title}"
                message = f"Ticket {ticket.id} has been escalated to level {ticket.escalation_level}."
            else:
                subject = f"SLA Alert: {ticket.title}"
                message = f"Alert for ticket {ticket.id}"

            # Send to each channel
            for channel in channels:
                if channel == "email":
                    await self._send_email_alert(ticket, subject, message)
                elif channel == "wechat":
                    await self._send_wechat_alert(ticket, subject, message)
                else:
                    logger.warning(f"Unknown notification channel: {channel}")

            logger.info(f"SLA alert sent for ticket {ticket.id} via {channels}")
            return True

        except Exception as e:
            logger.error(f"Error sending SLA alert: {e}")
            return False

    async def _send_email_alert(
        self,
        ticket: TicketModel,
        subject: str,
        message: str
    ) -> None:
        """Send email notification (placeholder)."""
        # TODO: Integrate with email service
        logger.info(f"[Email] {subject}: {message}")

    async def _send_wechat_alert(
        self,
        ticket: TicketModel,
        subject: str,
        message: str
    ) -> None:
        """Send WeChat Work notification (placeholder)."""
        # TODO: Integrate with WeChat Work API
        logger.info(f"[WeChat] {subject}: {message}")

    async def get_sla_compliance_report(
        self,
        tenant_id: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Generate SLA compliance report.

        Args:
            tenant_id: Optional tenant filter
            days: Number of days to analyze

        Returns:
            SLA compliance statistics
        """
        try:
            with db_manager.get_session() as session:
                cutoff = datetime.now() - timedelta(days=days)

                # Query resolved tickets in period
                query = select(TicketModel).where(
                    and_(
                        TicketModel.status.in_([
                            TicketStatus.RESOLVED,
                            TicketStatus.CLOSED
                        ]),
                        TicketModel.created_at >= cutoff
                    )
                )

                if tenant_id:
                    query = query.where(TicketModel.tenant_id == tenant_id)

                tickets = session.execute(query).scalars().all()

                total = len(tickets)
                breached = sum(1 for t in tickets if t.sla_breached)
                compliant = total - breached

                # Calculate by priority
                by_priority = {}
                for priority in TicketPriority:
                    priority_tickets = [t for t in tickets if t.priority == priority]
                    priority_total = len(priority_tickets)
                    priority_breached = sum(1 for t in priority_tickets if t.sla_breached)

                    by_priority[priority.value] = {
                        "total": priority_total,
                        "compliant": priority_total - priority_breached,
                        "breached": priority_breached,
                        "compliance_rate": (priority_total - priority_breached) / priority_total if priority_total > 0 else 1.0
                    }

                # Calculate average resolution times
                resolution_times = []
                for t in tickets:
                    if t.resolved_at and t.created_at:
                        resolution_times.append((t.resolved_at - t.created_at).total_seconds())

                avg_resolution_time = sum(resolution_times) / len(resolution_times) if resolution_times else 0

                return {
                    "period_days": days,
                    "total_tickets": total,
                    "compliant_tickets": compliant,
                    "breached_tickets": breached,
                    "compliance_rate": compliant / total if total > 0 else 1.0,
                    "avg_resolution_time_seconds": avg_resolution_time,
                    "by_priority": by_priority,
                    "generated_at": datetime.now().isoformat(),
                }

        except Exception as e:
            logger.error(f"Error generating SLA compliance report: {e}")
            return {}

    async def get_active_alerts(
        self,
        acknowledged: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Get active alerts.

        Args:
            acknowledged: Filter by acknowledgement status

        Returns:
            List of alert dictionaries
        """
        alerts = self._alerts

        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]

        return [a.to_dict() for a in alerts]

    async def acknowledge_alert(
        self,
        alert_id: UUID,
        acknowledged_by: str
    ) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert ID to acknowledge
            acknowledged_by: User acknowledging

        Returns:
            True if acknowledged successfully
        """
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()
                return True

        return False

    async def run_monitoring_cycle(
        self,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a complete monitoring cycle.

        Checks violations, warnings, unassigned tickets, and auto-escalates.

        Args:
            tenant_id: Optional tenant filter

        Returns:
            Summary of monitoring cycle results
        """
        logger.info("Starting SLA monitoring cycle...")

        violations = await self.check_sla_violations(tenant_id)
        warnings = await self.check_sla_warnings(tenant_id)
        unassigned = await self.check_unassigned_tickets(tenant_id)
        escalated = await self.auto_escalate_breached(tenant_id)

        summary = {
            "cycle_time": datetime.now().isoformat(),
            "violations_found": len(violations),
            "warnings_found": len(warnings),
            "unassigned_found": len(unassigned),
            "tickets_escalated": len(escalated),
            "total_active_alerts": len(self._alerts),
        }

        logger.info(f"SLA monitoring cycle complete: {summary}")
        return summary
