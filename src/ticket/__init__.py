"""
Ticket management module for SuperInsight Platform.

Provides intelligent ticket dispatch, SLA monitoring, workload balancing,
and ticket lifecycle management functionality.
"""

from .models import (
    TicketModel,
    AnnotatorSkillModel,
    TicketStatus,
    TicketPriority,
    TicketType,
    Ticket,
    AnnotatorSkill,
)

# Lazy imports for services to avoid circular dependency issues
def get_ticket_dispatcher():
    """Get TicketDispatcher instance with lazy import."""
    from .dispatcher import TicketDispatcher
    return TicketDispatcher()

def get_sla_monitor():
    """Get SLAMonitor instance with lazy import."""
    from .sla_monitor import SLAMonitor
    return SLAMonitor()

def get_ticket_tracker():
    """Get TicketTracker instance with lazy import."""
    from .tracker import TicketTracker
    return TicketTracker()

__all__ = [
    # Models
    "TicketModel",
    "AnnotatorSkillModel",
    "TicketStatus",
    "TicketPriority",
    "TicketType",
    "Ticket",
    "AnnotatorSkill",
    # Service getters
    "get_ticket_dispatcher",
    "get_sla_monitor",
    "get_ticket_tracker",
]
