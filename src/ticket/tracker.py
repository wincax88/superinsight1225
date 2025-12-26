"""
Ticket tracking and lifecycle management for SuperInsight platform.

Provides:
- Ticket CRUD operations
- Status lifecycle management
- History tracking
- Statistics and reporting
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import select, and_, or_, func
from sqlalchemy.orm import Session

from src.database.connection import db_manager
from src.ticket.models import (
    TicketModel,
    TicketHistoryModel,
    AnnotatorSkillModel,
    Ticket,
    TicketHistory,
    TicketStatus,
    TicketPriority,
    TicketType,
    SLAConfig,
)

logger = logging.getLogger(__name__)


class TicketTracker:
    """
    Ticket tracking and lifecycle management.

    Handles ticket creation, updates, status changes, and reporting.
    """

    # Valid status transitions
    STATUS_TRANSITIONS = {
        TicketStatus.OPEN: [TicketStatus.ASSIGNED, TicketStatus.CLOSED],
        TicketStatus.ASSIGNED: [TicketStatus.IN_PROGRESS, TicketStatus.OPEN, TicketStatus.ESCALATED],
        TicketStatus.IN_PROGRESS: [TicketStatus.PENDING_REVIEW, TicketStatus.ASSIGNED, TicketStatus.ESCALATED],
        TicketStatus.PENDING_REVIEW: [TicketStatus.RESOLVED, TicketStatus.IN_PROGRESS],
        TicketStatus.ESCALATED: [TicketStatus.ASSIGNED, TicketStatus.IN_PROGRESS, TicketStatus.RESOLVED],
        TicketStatus.RESOLVED: [TicketStatus.CLOSED, TicketStatus.IN_PROGRESS],
        TicketStatus.CLOSED: [],  # Terminal state
    }

    def __init__(self):
        """Initialize the ticket tracker."""
        pass

    async def create_ticket(
        self,
        ticket_type: TicketType,
        title: str,
        description: Optional[str] = None,
        priority: TicketPriority = TicketPriority.MEDIUM,
        tenant_id: Optional[str] = None,
        created_by: Optional[str] = None,
        quality_issue_id: Optional[UUID] = None,
        task_id: Optional[UUID] = None,
        skill_requirements: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Ticket:
        """
        Create a new ticket.

        Args:
            ticket_type: Type of ticket
            title: Ticket title
            description: Detailed description
            priority: Priority level
            tenant_id: Tenant identifier
            created_by: User creating the ticket
            quality_issue_id: Related quality issue
            task_id: Related task
            skill_requirements: Required skills for dispatch
            metadata: Additional metadata

        Returns:
            Created Ticket object
        """
        try:
            with db_manager.get_session() as session:
                now = datetime.now()

                # Create ticket model
                ticket_model = TicketModel(
                    ticket_type=ticket_type,
                    title=title,
                    description=description,
                    priority=priority,
                    status=TicketStatus.OPEN,
                    tenant_id=tenant_id,
                    created_by=created_by,
                    quality_issue_id=quality_issue_id,
                    task_id=task_id,
                    skill_requirements=skill_requirements or {},
                    metadata=metadata or {},
                    sla_deadline=SLAConfig.get_sla_deadline(priority, now),
                    created_at=now,
                    updated_at=now,
                )

                session.add(ticket_model)

                # Record creation history
                history = TicketHistoryModel(
                    ticket_id=ticket_model.id,
                    action="created",
                    new_value=TicketStatus.OPEN.value,
                    performed_by=created_by or "system",
                    notes=f"Ticket created with priority {priority.value}"
                )
                session.add(history)

                session.commit()

                logger.info(f"Created ticket {ticket_model.id}: {title}")

                # Convert to Pydantic model
                return self._to_pydantic(ticket_model)

        except Exception as e:
            logger.error(f"Error creating ticket: {e}")
            raise

    async def get_ticket(self, ticket_id: UUID) -> Optional[Ticket]:
        """
        Get a ticket by ID.

        Args:
            ticket_id: Ticket UUID

        Returns:
            Ticket if found, None otherwise
        """
        try:
            with db_manager.get_session() as session:
                ticket_model = session.execute(
                    select(TicketModel).where(TicketModel.id == ticket_id)
                ).scalar_one_or_none()

                if ticket_model:
                    return self._to_pydantic(ticket_model)
                return None

        except Exception as e:
            logger.error(f"Error getting ticket {ticket_id}: {e}")
            return None

    async def update_ticket(
        self,
        ticket_id: UUID,
        updated_by: str,
        **updates
    ) -> Optional[Ticket]:
        """
        Update ticket fields.

        Args:
            ticket_id: Ticket UUID
            updated_by: User making the update
            **updates: Fields to update

        Returns:
            Updated Ticket if successful
        """
        try:
            with db_manager.get_session() as session:
                ticket = session.execute(
                    select(TicketModel).where(TicketModel.id == ticket_id)
                ).scalar_one_or_none()

                if not ticket:
                    logger.error(f"Ticket not found: {ticket_id}")
                    return None

                # Track changes for history
                changes = []
                now = datetime.now()

                for field, new_value in updates.items():
                    if hasattr(ticket, field):
                        old_value = getattr(ticket, field)
                        if old_value != new_value:
                            setattr(ticket, field, new_value)
                            changes.append({
                                "field": field,
                                "old_value": str(old_value) if old_value else None,
                                "new_value": str(new_value) if new_value else None,
                            })

                if changes:
                    ticket.updated_at = now

                    # Record history for each change
                    for change in changes:
                        history = TicketHistoryModel(
                            ticket_id=ticket.id,
                            action=f"updated_{change['field']}",
                            old_value=change['old_value'],
                            new_value=change['new_value'],
                            performed_by=updated_by,
                        )
                        session.add(history)

                    session.commit()
                    logger.info(f"Updated ticket {ticket_id}: {len(changes)} field(s) changed")

                return self._to_pydantic(ticket)

        except Exception as e:
            logger.error(f"Error updating ticket {ticket_id}: {e}")
            return None

    async def change_status(
        self,
        ticket_id: UUID,
        new_status: TicketStatus,
        changed_by: str,
        notes: Optional[str] = None
    ) -> bool:
        """
        Change ticket status with validation.

        Args:
            ticket_id: Ticket UUID
            new_status: New status
            changed_by: User making the change
            notes: Optional notes

        Returns:
            True if status changed successfully
        """
        try:
            with db_manager.get_session() as session:
                ticket = session.execute(
                    select(TicketModel).where(TicketModel.id == ticket_id)
                ).scalar_one_or_none()

                if not ticket:
                    logger.error(f"Ticket not found: {ticket_id}")
                    return False

                # Validate transition
                allowed_transitions = self.STATUS_TRANSITIONS.get(ticket.status, [])
                if new_status not in allowed_transitions:
                    logger.warning(
                        f"Invalid status transition: {ticket.status} -> {new_status}"
                    )
                    return False

                old_status = ticket.status
                now = datetime.now()

                ticket.status = new_status
                ticket.updated_at = now

                # Set resolved_at for terminal states
                if new_status in [TicketStatus.RESOLVED, TicketStatus.CLOSED]:
                    ticket.resolved_at = now

                    # Update annotator metrics if assigned
                    if ticket.assigned_to:
                        await self._update_annotator_completion(
                            session,
                            ticket.assigned_to,
                            ticket
                        )

                # Record history
                history = TicketHistoryModel(
                    ticket_id=ticket.id,
                    action="status_changed",
                    old_value=old_status.value,
                    new_value=new_status.value,
                    performed_by=changed_by,
                    notes=notes
                )
                session.add(history)

                session.commit()
                logger.info(f"Ticket {ticket_id} status: {old_status} -> {new_status}")
                return True

        except Exception as e:
            logger.error(f"Error changing ticket status: {e}")
            return False

    async def _update_annotator_completion(
        self,
        session: Session,
        user_id: str,
        ticket: TicketModel
    ) -> None:
        """Update annotator metrics after ticket completion."""
        annotator = session.execute(
            select(AnnotatorSkillModel).where(
                AnnotatorSkillModel.user_id == user_id
            )
        ).scalar_one_or_none()

        if annotator:
            # Decrease workload
            annotator.current_workload = max(0, annotator.current_workload - 1)
            annotator.completed_tasks += 1

            # Update success rate
            if annotator.total_tasks > 0:
                annotator.success_rate = annotator.completed_tasks / annotator.total_tasks

            # Update average resolution time
            if ticket.created_at and ticket.resolved_at:
                resolution_time = int((ticket.resolved_at - ticket.created_at).total_seconds())
                if annotator.avg_resolution_time == 0:
                    annotator.avg_resolution_time = resolution_time
                else:
                    # Running average
                    annotator.avg_resolution_time = int(
                        (annotator.avg_resolution_time * (annotator.completed_tasks - 1) + resolution_time)
                        / annotator.completed_tasks
                    )

            annotator.last_active_at = datetime.now()

    async def assign_ticket(
        self,
        ticket_id: UUID,
        assignee_id: str,
        assigned_by: str
    ) -> bool:
        """
        Manually assign a ticket to a user.

        Args:
            ticket_id: Ticket UUID
            assignee_id: User to assign to
            assigned_by: User making the assignment

        Returns:
            True if assigned successfully
        """
        try:
            with db_manager.get_session() as session:
                ticket = session.execute(
                    select(TicketModel).where(TicketModel.id == ticket_id)
                ).scalar_one_or_none()

                if not ticket:
                    return False

                old_assignee = ticket.assigned_to
                now = datetime.now()

                ticket.assigned_to = assignee_id
                ticket.assigned_by = assigned_by
                ticket.assigned_at = now
                ticket.updated_at = now

                if ticket.status == TicketStatus.OPEN:
                    ticket.status = TicketStatus.ASSIGNED

                # Update old annotator workload
                if old_assignee and old_assignee != assignee_id:
                    old_annotator = session.execute(
                        select(AnnotatorSkillModel).where(
                            AnnotatorSkillModel.user_id == old_assignee
                        )
                    ).scalar_one_or_none()
                    if old_annotator:
                        old_annotator.current_workload = max(0, old_annotator.current_workload - 1)

                # Update new annotator workload
                new_annotator = session.execute(
                    select(AnnotatorSkillModel).where(
                        AnnotatorSkillModel.user_id == assignee_id
                    )
                ).scalar_one_or_none()
                if new_annotator:
                    new_annotator.current_workload += 1
                    new_annotator.total_tasks += 1
                    new_annotator.last_active_at = now

                # Record history
                history = TicketHistoryModel(
                    ticket_id=ticket.id,
                    action="assigned",
                    old_value=old_assignee,
                    new_value=assignee_id,
                    performed_by=assigned_by,
                )
                session.add(history)

                session.commit()
                return True

        except Exception as e:
            logger.error(f"Error assigning ticket: {e}")
            return False

    async def resolve_ticket(
        self,
        ticket_id: UUID,
        resolved_by: str,
        resolution_notes: Optional[str] = None
    ) -> bool:
        """
        Resolve a ticket.

        Args:
            ticket_id: Ticket UUID
            resolved_by: User resolving the ticket
            resolution_notes: Resolution notes

        Returns:
            True if resolved successfully
        """
        try:
            with db_manager.get_session() as session:
                ticket = session.execute(
                    select(TicketModel).where(TicketModel.id == ticket_id)
                ).scalar_one_or_none()

                if not ticket:
                    return False

                ticket.resolution_notes = resolution_notes

                return await self.change_status(
                    ticket_id,
                    TicketStatus.RESOLVED,
                    resolved_by,
                    notes=resolution_notes
                )

        except Exception as e:
            logger.error(f"Error resolving ticket: {e}")
            return False

    async def list_tickets(
        self,
        tenant_id: Optional[str] = None,
        status: Optional[TicketStatus] = None,
        priority: Optional[TicketPriority] = None,
        assigned_to: Optional[str] = None,
        ticket_type: Optional[TicketType] = None,
        sla_breached: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[Ticket], int]:
        """
        List tickets with filters.

        Args:
            tenant_id: Filter by tenant
            status: Filter by status
            priority: Filter by priority
            assigned_to: Filter by assignee
            ticket_type: Filter by type
            sla_breached: Filter by SLA breach status
            limit: Max results
            offset: Pagination offset

        Returns:
            Tuple of (tickets, total_count)
        """
        try:
            with db_manager.get_session() as session:
                query = select(TicketModel)
                count_query = select(func.count(TicketModel.id))

                # Apply filters
                filters = []
                if tenant_id:
                    filters.append(TicketModel.tenant_id == tenant_id)
                if status:
                    filters.append(TicketModel.status == status)
                if priority:
                    filters.append(TicketModel.priority == priority)
                if assigned_to:
                    filters.append(TicketModel.assigned_to == assigned_to)
                if ticket_type:
                    filters.append(TicketModel.ticket_type == ticket_type)
                if sla_breached is not None:
                    filters.append(TicketModel.sla_breached == sla_breached)

                if filters:
                    query = query.where(and_(*filters))
                    count_query = count_query.where(and_(*filters))

                # Get total count
                total = session.execute(count_query).scalar()

                # Get paginated results
                query = query.order_by(
                    TicketModel.priority.desc(),
                    TicketModel.created_at.desc()
                ).limit(limit).offset(offset)

                tickets = session.execute(query).scalars().all()

                return [self._to_pydantic(t) for t in tickets], total

        except Exception as e:
            logger.error(f"Error listing tickets: {e}")
            return [], 0

    async def get_ticket_history(
        self,
        ticket_id: UUID
    ) -> List[TicketHistory]:
        """
        Get history for a ticket.

        Args:
            ticket_id: Ticket UUID

        Returns:
            List of history records
        """
        try:
            with db_manager.get_session() as session:
                history_records = session.execute(
                    select(TicketHistoryModel)
                    .where(TicketHistoryModel.ticket_id == ticket_id)
                    .order_by(TicketHistoryModel.created_at.desc())
                ).scalars().all()

                return [
                    TicketHistory(
                        id=h.id,
                        ticket_id=h.ticket_id,
                        action=h.action,
                        old_value=h.old_value,
                        new_value=h.new_value,
                        performed_by=h.performed_by,
                        notes=h.notes,
                        created_at=h.created_at,
                    )
                    for h in history_records
                ]

        except Exception as e:
            logger.error(f"Error getting ticket history: {e}")
            return []

    async def get_statistics(
        self,
        tenant_id: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get ticket statistics.

        Args:
            tenant_id: Optional tenant filter
            days: Number of days to analyze

        Returns:
            Statistics dictionary
        """
        try:
            with db_manager.get_session() as session:
                cutoff = datetime.now() - timedelta(days=days)

                base_filter = [TicketModel.created_at >= cutoff]
                if tenant_id:
                    base_filter.append(TicketModel.tenant_id == tenant_id)

                # Total tickets
                total = session.execute(
                    select(func.count(TicketModel.id)).where(and_(*base_filter))
                ).scalar()

                # By status
                by_status = {}
                for status in TicketStatus:
                    count = session.execute(
                        select(func.count(TicketModel.id)).where(
                            and_(*base_filter, TicketModel.status == status)
                        )
                    ).scalar()
                    by_status[status.value] = count

                # By priority
                by_priority = {}
                for priority in TicketPriority:
                    count = session.execute(
                        select(func.count(TicketModel.id)).where(
                            and_(*base_filter, TicketModel.priority == priority)
                        )
                    ).scalar()
                    by_priority[priority.value] = count

                # SLA metrics
                sla_breached = session.execute(
                    select(func.count(TicketModel.id)).where(
                        and_(*base_filter, TicketModel.sla_breached == True)
                    )
                ).scalar()

                # Average resolution time
                resolved_tickets = session.execute(
                    select(TicketModel).where(
                        and_(
                            *base_filter,
                            TicketModel.resolved_at.isnot(None)
                        )
                    )
                ).scalars().all()

                resolution_times = []
                for t in resolved_tickets:
                    if t.resolved_at and t.created_at:
                        resolution_times.append(
                            (t.resolved_at - t.created_at).total_seconds()
                        )

                avg_resolution = sum(resolution_times) / len(resolution_times) if resolution_times else 0

                return {
                    "period_days": days,
                    "total_tickets": total,
                    "by_status": by_status,
                    "by_priority": by_priority,
                    "sla_breached": sla_breached,
                    "sla_compliance_rate": (total - sla_breached) / total if total > 0 else 1.0,
                    "avg_resolution_time_seconds": avg_resolution,
                    "open_tickets": by_status.get("open", 0) + by_status.get("assigned", 0) + by_status.get("in_progress", 0),
                    "resolved_tickets": by_status.get("resolved", 0) + by_status.get("closed", 0),
                    "generated_at": datetime.now().isoformat(),
                }

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

    def _to_pydantic(self, model: TicketModel) -> Ticket:
        """Convert SQLAlchemy model to Pydantic model."""
        return Ticket(
            id=model.id,
            ticket_type=model.ticket_type,
            title=model.title,
            description=model.description,
            priority=model.priority,
            status=model.status,
            quality_issue_id=model.quality_issue_id,
            task_id=model.task_id,
            assigned_to=model.assigned_to,
            assigned_by=model.assigned_by,
            assigned_at=model.assigned_at,
            skill_requirements=model.skill_requirements,
            workload_weight=model.workload_weight,
            sla_deadline=model.sla_deadline,
            sla_breached=model.sla_breached,
            escalation_level=model.escalation_level,
            tenant_id=model.tenant_id,
            created_by=model.created_by,
            created_at=model.created_at,
            updated_at=model.updated_at,
            resolved_at=model.resolved_at,
            metadata=model.metadata,
            resolution_notes=model.resolution_notes,
        )
