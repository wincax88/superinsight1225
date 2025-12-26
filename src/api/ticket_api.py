"""
Ticket Management API for SuperInsight Platform.

Provides REST API endpoints for ticket creation, dispatch, tracking,
and SLA monitoring operations.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.ticket.models import (
    TicketStatus,
    TicketPriority,
    TicketType,
)
from src.ticket.tracker import TicketTracker
from src.ticket.dispatcher import TicketDispatcher
from src.ticket.sla_monitor import SLAMonitor


router = APIRouter(prefix="/api/v1/tickets", tags=["tickets"])

# Global instances - use lazy initialization
_ticket_tracker: Optional[TicketTracker] = None
_ticket_dispatcher: Optional[TicketDispatcher] = None
_sla_monitor: Optional[SLAMonitor] = None


def get_ticket_tracker() -> TicketTracker:
    """Get or create ticket tracker instance."""
    global _ticket_tracker
    if _ticket_tracker is None:
        _ticket_tracker = TicketTracker()
    return _ticket_tracker


def get_ticket_dispatcher() -> TicketDispatcher:
    """Get or create ticket dispatcher instance."""
    global _ticket_dispatcher
    if _ticket_dispatcher is None:
        _ticket_dispatcher = TicketDispatcher()
    return _ticket_dispatcher


def get_sla_monitor() -> SLAMonitor:
    """Get or create SLA monitor instance."""
    global _sla_monitor
    if _sla_monitor is None:
        _sla_monitor = SLAMonitor()
    return _sla_monitor


# ==================== Request/Response Models ====================

class CreateTicketRequest(BaseModel):
    """Request model for creating a ticket."""
    ticket_type: str = Field(..., description="Type of ticket (quality_issue, annotation_error, etc.)")
    title: str = Field(..., min_length=1, max_length=200, description="Ticket title")
    description: Optional[str] = Field(None, description="Detailed description")
    priority: str = Field("medium", description="Priority level (critical, high, medium, low)")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")
    created_by: Optional[str] = Field(None, description="User creating the ticket")
    quality_issue_id: Optional[UUID] = Field(None, description="Related quality issue ID")
    task_id: Optional[UUID] = Field(None, description="Related task ID")
    skill_requirements: Optional[Dict[str, Any]] = Field(None, description="Required skills for dispatch")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class UpdateTicketRequest(BaseModel):
    """Request model for updating a ticket."""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    priority: Optional[str] = None
    skill_requirements: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    updated_by: str = Field(..., description="User making the update")


class AssignTicketRequest(BaseModel):
    """Request model for assigning a ticket."""
    assignee_id: str = Field(..., description="User to assign to")
    assigned_by: str = Field(..., description="User making the assignment")


class ChangeStatusRequest(BaseModel):
    """Request model for changing ticket status."""
    status: str = Field(..., description="New status")
    changed_by: str = Field(..., description="User making the change")
    notes: Optional[str] = Field(None, description="Optional notes")


class ResolveTicketRequest(BaseModel):
    """Request model for resolving a ticket."""
    resolved_by: str = Field(..., description="User resolving the ticket")
    resolution_notes: Optional[str] = Field(None, description="Resolution notes")


class DispatchTicketRequest(BaseModel):
    """Request model for dispatching a ticket."""
    auto_assign: bool = Field(True, description="Whether to auto-assign")
    preferred_user: Optional[str] = Field(None, description="Preferred user to assign")


class EscalateTicketRequest(BaseModel):
    """Request model for escalating a ticket."""
    reason: str = Field("Manual escalation", description="Reason for escalation")
    escalated_by: str = Field(..., description="User escalating the ticket")


class TicketResponse(BaseModel):
    """Response model for ticket data."""
    id: str
    ticket_type: str
    title: str
    description: Optional[str]
    priority: str
    status: str
    assigned_to: Optional[str]
    sla_deadline: Optional[str]
    sla_breached: bool
    created_at: str
    updated_at: str


class TicketListResponse(BaseModel):
    """Response model for ticket list."""
    tickets: List[Dict[str, Any]]
    total: int
    limit: int
    offset: int


# ==================== Ticket CRUD Endpoints ====================

@router.post("", response_model=Dict[str, Any])
async def create_ticket(request: CreateTicketRequest) -> Dict[str, Any]:
    """
    Create a new ticket.

    Creates a ticket with automatic SLA deadline calculation.
    """
    try:
        tracker = get_ticket_tracker()

        # Convert string enums
        ticket_type = TicketType(request.ticket_type)
        priority = TicketPriority(request.priority)

        ticket = await tracker.create_ticket(
            ticket_type=ticket_type,
            title=request.title,
            description=request.description,
            priority=priority,
            tenant_id=request.tenant_id,
            created_by=request.created_by,
            quality_issue_id=request.quality_issue_id,
            task_id=request.task_id,
            skill_requirements=request.skill_requirements,
            metadata=request.metadata,
        )

        return {
            "status": "success",
            "ticket": ticket.to_dict(),
            "message": f"Ticket created: {ticket.id}"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create ticket: {str(e)}")


@router.get("", response_model=TicketListResponse)
async def list_tickets(
    tenant_id: Optional[str] = Query(None, description="Filter by tenant"),
    status: Optional[str] = Query(None, description="Filter by status"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    assigned_to: Optional[str] = Query(None, description="Filter by assignee"),
    ticket_type: Optional[str] = Query(None, description="Filter by type"),
    sla_breached: Optional[bool] = Query(None, description="Filter by SLA breach"),
    limit: int = Query(50, ge=1, le=200, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
) -> TicketListResponse:
    """
    List tickets with optional filters.

    Supports filtering by tenant, status, priority, assignee, type, and SLA breach.
    """
    try:
        tracker = get_ticket_tracker()

        # Convert string enums if provided
        status_enum = TicketStatus(status) if status else None
        priority_enum = TicketPriority(priority) if priority else None
        type_enum = TicketType(ticket_type) if ticket_type else None

        tickets, total = await tracker.list_tickets(
            tenant_id=tenant_id,
            status=status_enum,
            priority=priority_enum,
            assigned_to=assigned_to,
            ticket_type=type_enum,
            sla_breached=sla_breached,
            limit=limit,
            offset=offset,
        )

        return TicketListResponse(
            tickets=[t.to_dict() for t in tickets],
            total=total,
            limit=limit,
            offset=offset,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid filter: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list tickets: {str(e)}")


@router.get("/{ticket_id}", response_model=Dict[str, Any])
async def get_ticket(ticket_id: UUID) -> Dict[str, Any]:
    """
    Get a ticket by ID.

    Returns full ticket details including SLA status.
    """
    try:
        tracker = get_ticket_tracker()
        ticket = await tracker.get_ticket(ticket_id)

        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")

        return {
            "status": "success",
            "ticket": ticket.to_dict()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get ticket: {str(e)}")


@router.put("/{ticket_id}", response_model=Dict[str, Any])
async def update_ticket(
    ticket_id: UUID,
    request: UpdateTicketRequest
) -> Dict[str, Any]:
    """
    Update ticket fields.

    Updates the specified fields and records history.
    """
    try:
        tracker = get_ticket_tracker()

        updates = {}
        if request.title is not None:
            updates["title"] = request.title
        if request.description is not None:
            updates["description"] = request.description
        if request.priority is not None:
            updates["priority"] = TicketPriority(request.priority)
        if request.skill_requirements is not None:
            updates["skill_requirements"] = request.skill_requirements
        if request.metadata is not None:
            updates["metadata"] = request.metadata

        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")

        ticket = await tracker.update_ticket(
            ticket_id,
            request.updated_by,
            **updates
        )

        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")

        return {
            "status": "success",
            "ticket": ticket.to_dict(),
            "message": "Ticket updated"
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update ticket: {str(e)}")


# ==================== Assignment Endpoints ====================

@router.put("/{ticket_id}/assign", response_model=Dict[str, Any])
async def assign_ticket(
    ticket_id: UUID,
    request: AssignTicketRequest
) -> Dict[str, Any]:
    """
    Manually assign a ticket to a user.

    Updates the ticket assignment and annotator workload.
    """
    try:
        tracker = get_ticket_tracker()

        success = await tracker.assign_ticket(
            ticket_id,
            request.assignee_id,
            request.assigned_by
        )

        if not success:
            raise HTTPException(status_code=404, detail="Ticket not found or assignment failed")

        return {
            "status": "success",
            "message": f"Ticket {ticket_id} assigned to {request.assignee_id}"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to assign ticket: {str(e)}")


@router.post("/{ticket_id}/dispatch", response_model=Dict[str, Any])
async def dispatch_ticket(
    ticket_id: UUID,
    request: DispatchTicketRequest
) -> Dict[str, Any]:
    """
    Dispatch a ticket using intelligent assignment.

    Uses skill matching, workload balancing, and performance history.
    """
    try:
        dispatcher = get_ticket_dispatcher()

        assigned_to = await dispatcher.dispatch_ticket(
            ticket_id,
            auto_assign=request.auto_assign,
            preferred_user=request.preferred_user
        )

        if not assigned_to:
            return {
                "status": "no_match",
                "message": "No suitable annotator found for this ticket"
            }

        return {
            "status": "success",
            "assigned_to": assigned_to,
            "message": f"Ticket dispatched to {assigned_to}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to dispatch ticket: {str(e)}")


@router.get("/{ticket_id}/recommendations", response_model=Dict[str, Any])
async def get_dispatch_recommendations(ticket_id: UUID) -> Dict[str, Any]:
    """
    Get dispatch recommendations for a ticket.

    Returns ranked list of suitable annotators without assigning.
    """
    try:
        dispatcher = get_ticket_dispatcher()

        recommendations = await dispatcher.get_dispatch_recommendations(ticket_id)

        return {
            "status": "success",
            "ticket_id": str(ticket_id),
            "recommendations": recommendations
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")


# ==================== Status Management Endpoints ====================

@router.put("/{ticket_id}/status", response_model=Dict[str, Any])
async def change_ticket_status(
    ticket_id: UUID,
    request: ChangeStatusRequest
) -> Dict[str, Any]:
    """
    Change ticket status.

    Validates status transition and records history.
    """
    try:
        tracker = get_ticket_tracker()

        new_status = TicketStatus(request.status)

        success = await tracker.change_status(
            ticket_id,
            new_status,
            request.changed_by,
            request.notes
        )

        if not success:
            raise HTTPException(
                status_code=400,
                detail="Invalid status transition or ticket not found"
            )

        return {
            "status": "success",
            "message": f"Ticket status changed to {request.status}"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid status: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to change status: {str(e)}")


@router.post("/{ticket_id}/resolve", response_model=Dict[str, Any])
async def resolve_ticket(
    ticket_id: UUID,
    request: ResolveTicketRequest
) -> Dict[str, Any]:
    """
    Resolve a ticket.

    Marks the ticket as resolved with optional resolution notes.
    """
    try:
        tracker = get_ticket_tracker()

        success = await tracker.resolve_ticket(
            ticket_id,
            request.resolved_by,
            request.resolution_notes
        )

        if not success:
            raise HTTPException(status_code=404, detail="Ticket not found or cannot be resolved")

        return {
            "status": "success",
            "message": f"Ticket {ticket_id} resolved"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to resolve ticket: {str(e)}")


@router.post("/{ticket_id}/escalate", response_model=Dict[str, Any])
async def escalate_ticket(
    ticket_id: UUID,
    request: EscalateTicketRequest
) -> Dict[str, Any]:
    """
    Escalate a ticket.

    Increases escalation level and marks ticket as escalated.
    """
    try:
        monitor = get_sla_monitor()

        success = await monitor.escalate_ticket(
            ticket_id,
            request.reason,
            request.escalated_by
        )

        if not success:
            raise HTTPException(status_code=404, detail="Ticket not found or escalation failed")

        return {
            "status": "success",
            "message": f"Ticket {ticket_id} escalated"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to escalate ticket: {str(e)}")


# ==================== History Endpoints ====================

@router.get("/{ticket_id}/history", response_model=Dict[str, Any])
async def get_ticket_history(ticket_id: UUID) -> Dict[str, Any]:
    """
    Get ticket history.

    Returns all status changes and actions on the ticket.
    """
    try:
        tracker = get_ticket_tracker()

        history = await tracker.get_ticket_history(ticket_id)

        return {
            "status": "success",
            "ticket_id": str(ticket_id),
            "history": [h.to_dict() for h in history]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


# ==================== SLA Monitoring Endpoints ====================

@router.get("/sla/violations", response_model=Dict[str, Any])
async def get_sla_violations(
    tenant_id: Optional[str] = Query(None, description="Filter by tenant")
) -> Dict[str, Any]:
    """
    Get tickets with SLA violations.

    Checks all active tickets for SLA breaches.
    """
    try:
        monitor = get_sla_monitor()

        violations = await monitor.check_sla_violations(tenant_id)

        return {
            "status": "success",
            "violations_count": len(violations),
            "violations": [
                {
                    "ticket_id": str(v.id),
                    "title": v.title,
                    "priority": v.priority.value,
                    "sla_deadline": v.sla_deadline.isoformat() if v.sla_deadline else None,
                    "assigned_to": v.assigned_to,
                }
                for v in violations
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check SLA violations: {str(e)}")


@router.get("/sla/warnings", response_model=Dict[str, Any])
async def get_sla_warnings(
    tenant_id: Optional[str] = Query(None, description="Filter by tenant")
) -> Dict[str, Any]:
    """
    Get tickets approaching SLA deadline.

    Returns tickets within warning threshold of their SLA.
    """
    try:
        monitor = get_sla_monitor()

        warnings = await monitor.check_sla_warnings(tenant_id)

        return {
            "status": "success",
            "warnings_count": len(warnings),
            "warnings": [
                {
                    "ticket_id": str(w.id),
                    "title": w.title,
                    "priority": w.priority.value,
                    "sla_deadline": w.sla_deadline.isoformat() if w.sla_deadline else None,
                    "assigned_to": w.assigned_to,
                }
                for w in warnings
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check SLA warnings: {str(e)}")


@router.get("/sla/compliance", response_model=Dict[str, Any])
async def get_sla_compliance_report(
    tenant_id: Optional[str] = Query(None, description="Filter by tenant"),
    days: int = Query(30, ge=1, le=365, description="Analysis period in days")
) -> Dict[str, Any]:
    """
    Get SLA compliance report.

    Returns SLA compliance statistics for the specified period.
    """
    try:
        monitor = get_sla_monitor()

        report = await monitor.get_sla_compliance_report(tenant_id, days)

        return {
            "status": "success",
            "report": report
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


# ==================== Workload Endpoints ====================

@router.get("/workload", response_model=Dict[str, Any])
async def get_workload_distribution(
    tenant_id: Optional[str] = Query(None, description="Filter by tenant")
) -> Dict[str, Any]:
    """
    Get workload distribution across annotators.

    Shows current workload and capacity for each annotator.
    """
    try:
        dispatcher = get_ticket_dispatcher()

        distribution = await dispatcher.get_workload_distribution(tenant_id)

        return {
            "status": "success",
            "workload": distribution
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get workload: {str(e)}")


@router.post("/workload/rebalance", response_model=Dict[str, Any])
async def rebalance_workload(
    tenant_id: Optional[str] = Query(None, description="Filter by tenant")
) -> Dict[str, Any]:
    """
    Rebalance workload across annotators.

    Reassigns tickets from overloaded annotators to underutilized ones.
    """
    try:
        dispatcher = get_ticket_dispatcher()

        actions = await dispatcher.rebalance_workload(tenant_id)

        return {
            "status": "success",
            "actions_taken": len(actions),
            "actions": actions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rebalance: {str(e)}")


# ==================== Statistics Endpoints ====================

@router.get("/statistics", response_model=Dict[str, Any])
async def get_ticket_statistics(
    tenant_id: Optional[str] = Query(None, description="Filter by tenant"),
    days: int = Query(30, ge=1, le=365, description="Analysis period in days")
) -> Dict[str, Any]:
    """
    Get ticket statistics.

    Returns aggregate statistics for tickets in the specified period.
    """
    try:
        tracker = get_ticket_tracker()

        stats = await tracker.get_statistics(tenant_id, days)

        return {
            "status": "success",
            "statistics": stats
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


# ==================== Alerts Endpoints ====================

@router.get("/alerts", response_model=Dict[str, Any])
async def get_active_alerts(
    acknowledged: Optional[bool] = Query(None, description="Filter by acknowledgement status")
) -> Dict[str, Any]:
    """
    Get active SLA alerts.

    Returns alerts for SLA violations, warnings, and escalations.
    """
    try:
        monitor = get_sla_monitor()

        alerts = await monitor.get_active_alerts(acknowledged)

        return {
            "status": "success",
            "alerts_count": len(alerts),
            "alerts": alerts
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")


@router.post("/monitoring/cycle", response_model=Dict[str, Any])
async def run_monitoring_cycle(
    tenant_id: Optional[str] = Query(None, description="Filter by tenant")
) -> Dict[str, Any]:
    """
    Run a complete SLA monitoring cycle.

    Checks violations, warnings, unassigned tickets, and auto-escalates.
    """
    try:
        monitor = get_sla_monitor()

        summary = await monitor.run_monitoring_cycle(tenant_id)

        return {
            "status": "success",
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run monitoring: {str(e)}")


# ==================== Health Check ====================

@router.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """
    Health check for ticket management service.
    """
    return {
        "status": "healthy",
        "service": "ticket-management",
        "timestamp": datetime.now().isoformat()
    }
