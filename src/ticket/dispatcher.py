"""
Intelligent ticket dispatch engine for SuperInsight platform.

Implements smart ticket assignment based on:
- Skill matching
- Workload balancing
- Historical performance
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import select, and_, func
from sqlalchemy.orm import Session

from src.database.connection import db_manager
from src.ticket.models import (
    TicketModel,
    AnnotatorSkillModel,
    TicketHistoryModel,
    Ticket,
    AnnotatorSkill,
    TicketStatus,
    TicketPriority,
    TicketType,
    SLAConfig,
)

logger = logging.getLogger(__name__)


class TicketDispatcher:
    """
    Intelligent ticket dispatch engine.

    Assigns tickets to annotators based on:
    - Skill matching (40% weight)
    - Available capacity (30% weight)
    - Historical performance (30% weight)
    """

    # Scoring weights
    SKILL_WEIGHT = 0.4
    CAPACITY_WEIGHT = 0.3
    PERFORMANCE_WEIGHT = 0.3

    # Minimum thresholds
    MIN_SKILL_LEVEL = 0.3  # Minimum skill level required
    MIN_CAPACITY = 1  # At least 1 available slot

    def __init__(self):
        """Initialize the ticket dispatcher."""
        self._assignment_cache: Dict[str, List[str]] = {}

    async def dispatch_ticket(
        self,
        ticket_id: UUID,
        auto_assign: bool = True,
        preferred_user: Optional[str] = None
    ) -> Optional[str]:
        """
        Dispatch a ticket to the best available annotator.

        Args:
            ticket_id: UUID of the ticket to dispatch
            auto_assign: Whether to automatically assign the ticket
            preferred_user: Optional preferred user to assign to

        Returns:
            User ID of the assigned annotator, or None if no suitable annotator found
        """
        try:
            with db_manager.get_session() as session:
                # Get ticket
                ticket = session.execute(
                    select(TicketModel).where(TicketModel.id == ticket_id)
                ).scalar_one_or_none()

                if not ticket:
                    logger.error(f"Ticket not found: {ticket_id}")
                    return None

                if ticket.status not in [TicketStatus.OPEN, TicketStatus.ESCALATED]:
                    logger.warning(f"Ticket {ticket_id} is not in dispatchable state: {ticket.status}")
                    return None

                # If preferred user is specified, try to assign to them first
                if preferred_user:
                    if await self._can_assign_to_user(session, preferred_user, ticket):
                        if auto_assign:
                            await self._assign_ticket(session, ticket, preferred_user, "manual_preferred")
                        return preferred_user

                # Get available annotators
                candidates = await self.get_available_annotators(
                    session,
                    ticket.skill_requirements,
                    ticket.tenant_id
                )

                if not candidates:
                    logger.warning(f"No available annotators for ticket {ticket_id}")
                    return None

                # Score and rank candidates
                scored_candidates = await self._score_candidates(session, candidates, ticket)

                if not scored_candidates:
                    return None

                # Get best candidate
                best_candidate = scored_candidates[0]
                best_user_id = best_candidate["user_id"]

                if auto_assign:
                    await self._assign_ticket(session, ticket, best_user_id, "auto_dispatch")
                    logger.info(f"Auto-assigned ticket {ticket_id} to {best_user_id} (score: {best_candidate['score']:.3f})")

                return best_user_id

        except Exception as e:
            logger.error(f"Error dispatching ticket {ticket_id}: {e}")
            raise

    async def get_available_annotators(
        self,
        session: Session,
        skill_requirements: Dict[str, Any],
        tenant_id: Optional[str] = None
    ) -> List[AnnotatorSkillModel]:
        """
        Get annotators that meet skill requirements and have available capacity.

        Args:
            session: Database session
            skill_requirements: Required skills and levels
            tenant_id: Optional tenant filter

        Returns:
            List of available annotators matching requirements
        """
        try:
            # Build base query
            query = select(AnnotatorSkillModel).where(
                and_(
                    AnnotatorSkillModel.is_available == True,
                    AnnotatorSkillModel.current_workload < AnnotatorSkillModel.max_workload
                )
            )

            # Apply tenant filter if specified
            if tenant_id:
                query = query.where(AnnotatorSkillModel.tenant_id == tenant_id)

            # Execute query
            result = session.execute(query)
            all_annotators = result.scalars().all()

            # Filter by skill requirements
            if not skill_requirements:
                return list(all_annotators)

            qualified_annotators = []
            required_skills = skill_requirements.get("skills", [])
            min_level = skill_requirements.get("min_level", self.MIN_SKILL_LEVEL)

            for annotator in all_annotators:
                # Check if annotator has required skill type
                if required_skills and annotator.skill_type not in required_skills:
                    continue

                # Check skill level
                if annotator.skill_level < min_level:
                    continue

                qualified_annotators.append(annotator)

            return qualified_annotators

        except Exception as e:
            logger.error(f"Error getting available annotators: {e}")
            return []

    def calculate_assignment_score(
        self,
        annotator: AnnotatorSkillModel,
        ticket: TicketModel
    ) -> float:
        """
        Calculate assignment score for an annotator-ticket pair.

        Score = skill_match * 0.4 + capacity_score * 0.3 + performance_score * 0.3

        Args:
            annotator: Annotator skill record
            ticket: Ticket to assign

        Returns:
            Assignment score (0-1)
        """
        # Skill matching score
        skill_score = self._calculate_skill_score(annotator, ticket)

        # Capacity score (inverse of utilization)
        capacity_score = self._calculate_capacity_score(annotator)

        # Performance score
        performance_score = self._calculate_performance_score(annotator)

        # Calculate weighted score
        total_score = (
            skill_score * self.SKILL_WEIGHT +
            capacity_score * self.CAPACITY_WEIGHT +
            performance_score * self.PERFORMANCE_WEIGHT
        )

        return min(1.0, max(0.0, total_score))

    def _calculate_skill_score(
        self,
        annotator: AnnotatorSkillModel,
        ticket: TicketModel
    ) -> float:
        """Calculate skill matching score."""
        base_score = annotator.skill_level

        # Bonus for exact skill match
        skill_requirements = ticket.skill_requirements or {}
        required_skills = skill_requirements.get("skills", [])

        if required_skills and annotator.skill_type in required_skills:
            base_score = min(1.0, base_score * 1.2)

        # Priority boost for high priority tickets
        if ticket.priority in [TicketPriority.CRITICAL, TicketPriority.HIGH]:
            # Prefer higher skilled annotators for critical tickets
            base_score = min(1.0, base_score * 1.1)

        return base_score

    def _calculate_capacity_score(self, annotator: AnnotatorSkillModel) -> float:
        """Calculate capacity score (higher is better)."""
        if annotator.max_workload == 0:
            return 0.0

        utilization = annotator.current_workload / annotator.max_workload
        # Inverse utilization: lower workload = higher score
        return 1.0 - utilization

    def _calculate_performance_score(self, annotator: AnnotatorSkillModel) -> float:
        """Calculate historical performance score."""
        # Weight success rate and quality score equally
        success_score = annotator.success_rate
        quality_score = annotator.avg_quality_score

        # Consider resolution time (lower is better)
        # Normalize to 0-1 (assume 1 hour = 3600s is baseline)
        time_score = 1.0
        if annotator.avg_resolution_time > 0:
            time_score = min(1.0, 3600 / annotator.avg_resolution_time)

        return (success_score * 0.4 + quality_score * 0.4 + time_score * 0.2)

    async def _score_candidates(
        self,
        session: Session,
        candidates: List[AnnotatorSkillModel],
        ticket: TicketModel
    ) -> List[Dict[str, Any]]:
        """Score and rank candidate annotators."""
        scored = []

        for annotator in candidates:
            score = self.calculate_assignment_score(annotator, ticket)
            scored.append({
                "user_id": annotator.user_id,
                "score": score,
                "skill_level": annotator.skill_level,
                "current_workload": annotator.current_workload,
                "success_rate": annotator.success_rate,
            })

        # Sort by score (descending)
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    async def _can_assign_to_user(
        self,
        session: Session,
        user_id: str,
        ticket: TicketModel
    ) -> bool:
        """Check if ticket can be assigned to specific user."""
        annotator = session.execute(
            select(AnnotatorSkillModel).where(
                and_(
                    AnnotatorSkillModel.user_id == user_id,
                    AnnotatorSkillModel.is_available == True,
                    AnnotatorSkillModel.current_workload < AnnotatorSkillModel.max_workload
                )
            )
        ).scalar_one_or_none()

        return annotator is not None

    async def _assign_ticket(
        self,
        session: Session,
        ticket: TicketModel,
        user_id: str,
        assignment_type: str
    ) -> None:
        """Assign ticket to user and update related records."""
        now = datetime.now()

        # Update ticket
        old_status = ticket.status
        old_assignee = ticket.assigned_to

        ticket.assigned_to = user_id
        ticket.assigned_at = now
        ticket.status = TicketStatus.ASSIGNED
        ticket.updated_at = now

        # Set SLA deadline if not set
        if not ticket.sla_deadline:
            ticket.sla_deadline = SLAConfig.get_sla_deadline(ticket.priority, now)

        # Update annotator workload
        annotator = session.execute(
            select(AnnotatorSkillModel).where(AnnotatorSkillModel.user_id == user_id)
        ).scalar_one_or_none()

        if annotator:
            annotator.current_workload += 1
            annotator.total_tasks += 1
            annotator.last_active_at = now

        # Record history
        history = TicketHistoryModel(
            ticket_id=ticket.id,
            action="assigned",
            old_value=old_assignee,
            new_value=user_id,
            performed_by="system",
            notes=f"Auto-assigned via {assignment_type}"
        )
        session.add(history)

        # Record status change
        if old_status != TicketStatus.ASSIGNED:
            status_history = TicketHistoryModel(
                ticket_id=ticket.id,
                action="status_changed",
                old_value=old_status.value,
                new_value=TicketStatus.ASSIGNED.value,
                performed_by="system",
                notes="Status changed due to assignment"
            )
            session.add(status_history)

        session.commit()
        logger.info(f"Ticket {ticket.id} assigned to {user_id}")

    async def rebalance_workload(
        self,
        tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Rebalance workload across annotators.

        Reassigns tickets from overloaded annotators to underutilized ones.

        Args:
            tenant_id: Optional tenant filter

        Returns:
            List of rebalancing actions taken
        """
        actions = []

        try:
            with db_manager.get_session() as session:
                # Find overloaded annotators (> 80% utilization)
                overloaded_query = select(AnnotatorSkillModel).where(
                    and_(
                        AnnotatorSkillModel.is_available == True,
                        AnnotatorSkillModel.current_workload > AnnotatorSkillModel.max_workload * 0.8
                    )
                )
                if tenant_id:
                    overloaded_query = overloaded_query.where(
                        AnnotatorSkillModel.tenant_id == tenant_id
                    )

                overloaded = session.execute(overloaded_query).scalars().all()

                for annotator in overloaded:
                    # Find tickets that can be reassigned
                    tickets = session.execute(
                        select(TicketModel).where(
                            and_(
                                TicketModel.assigned_to == annotator.user_id,
                                TicketModel.status == TicketStatus.ASSIGNED,
                                TicketModel.sla_breached == False
                            )
                        ).order_by(TicketModel.priority.desc())
                    ).scalars().all()

                    # Try to reassign lowest priority tickets
                    for ticket in reversed(tickets):
                        if annotator.current_workload <= annotator.max_workload * 0.6:
                            break  # Balanced enough

                        new_assignee = await self.dispatch_ticket(
                            ticket.id,
                            auto_assign=False
                        )

                        if new_assignee and new_assignee != annotator.user_id:
                            await self._reassign_ticket(session, ticket, new_assignee)
                            actions.append({
                                "ticket_id": str(ticket.id),
                                "from_user": annotator.user_id,
                                "to_user": new_assignee,
                                "reason": "workload_rebalancing"
                            })

                session.commit()

        except Exception as e:
            logger.error(f"Error rebalancing workload: {e}")

        return actions

    async def _reassign_ticket(
        self,
        session: Session,
        ticket: TicketModel,
        new_user_id: str
    ) -> None:
        """Reassign ticket from one user to another."""
        old_user_id = ticket.assigned_to

        # Update old annotator workload
        if old_user_id:
            old_annotator = session.execute(
                select(AnnotatorSkillModel).where(
                    AnnotatorSkillModel.user_id == old_user_id
                )
            ).scalar_one_or_none()

            if old_annotator:
                old_annotator.current_workload = max(0, old_annotator.current_workload - 1)

        # Assign to new user
        await self._assign_ticket(session, ticket, new_user_id, "workload_rebalancing")

    async def get_workload_distribution(
        self,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get current workload distribution statistics.

        Args:
            tenant_id: Optional tenant filter

        Returns:
            Workload distribution summary
        """
        try:
            with db_manager.get_session() as session:
                query = select(AnnotatorSkillModel)
                if tenant_id:
                    query = query.where(AnnotatorSkillModel.tenant_id == tenant_id)

                annotators = session.execute(query).scalars().all()

                total_capacity = 0
                total_workload = 0
                by_user = []

                for a in annotators:
                    total_capacity += a.max_workload
                    total_workload += a.current_workload
                    by_user.append({
                        "user_id": a.user_id,
                        "current_workload": a.current_workload,
                        "max_workload": a.max_workload,
                        "utilization": a.current_workload / a.max_workload if a.max_workload > 0 else 0,
                        "is_available": a.is_available,
                    })

                return {
                    "total_capacity": total_capacity,
                    "total_workload": total_workload,
                    "overall_utilization": total_workload / total_capacity if total_capacity > 0 else 0,
                    "annotator_count": len(annotators),
                    "available_count": sum(1 for a in annotators if a.is_available),
                    "by_user": by_user,
                }

        except Exception as e:
            logger.error(f"Error getting workload distribution: {e}")
            return {}

    async def get_dispatch_recommendations(
        self,
        ticket_id: UUID
    ) -> List[Dict[str, Any]]:
        """
        Get dispatch recommendations for a ticket without auto-assigning.

        Args:
            ticket_id: UUID of the ticket

        Returns:
            List of recommended annotators with scores
        """
        try:
            with db_manager.get_session() as session:
                ticket = session.execute(
                    select(TicketModel).where(TicketModel.id == ticket_id)
                ).scalar_one_or_none()

                if not ticket:
                    return []

                candidates = await self.get_available_annotators(
                    session,
                    ticket.skill_requirements,
                    ticket.tenant_id
                )

                scored = await self._score_candidates(session, candidates, ticket)
                return scored[:5]  # Return top 5 recommendations

        except Exception as e:
            logger.error(f"Error getting dispatch recommendations: {e}")
            return []
