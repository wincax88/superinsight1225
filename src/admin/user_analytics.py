"""
User Behavior Analysis and Reporting for SuperInsight Platform.

Provides comprehensive user activity tracking, behavior analysis,
and detailed reporting capabilities for the management console.
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import statistics

from src.system.business_metrics import business_metrics_collector


logger = logging.getLogger(__name__)


class ActionType(Enum):
    """User action types for analytics."""
    LOGIN = "login"
    LOGOUT = "logout"
    ANNOTATION = "annotation"
    DOCUMENT_UPLOAD = "document_upload"
    PROJECT_CREATE = "project_create"
    PROJECT_VIEW = "project_view"
    TASK_COMPLETE = "task_complete"
    QUALITY_REVIEW = "quality_review"
    EXPORT_DATA = "export_data"
    CONFIG_CHANGE = "config_change"
    AI_INFERENCE = "ai_inference"
    SEARCH = "search"
    NAVIGATION = "navigation"


@dataclass
class UserSession:
    """User session information."""
    session_id: str
    user_id: str
    start_time: float
    last_activity: float
    actions_count: int = 0
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    tenant_id: Optional[str] = None
    
    @property
    def duration(self) -> float:
        """Get session duration in seconds."""
        return self.last_activity - self.start_time
    
    @property
    def is_active(self) -> bool:
        """Check if session is still active (within last 30 minutes)."""
        return (time.time() - self.last_activity) < 1800  # 30 minutes


@dataclass
class UserAction:
    """User action record."""
    action_id: str
    user_id: str
    session_id: str
    action_type: ActionType
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    success: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "action_id": self.action_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "action_type": self.action_type.value,
            "timestamp": self.timestamp,
            "details": self.details,
            "duration_ms": self.duration_ms,
            "success": self.success
        }


@dataclass
class UserBehaviorProfile:
    """User behavior profile for analytics."""
    user_id: str
    total_sessions: int
    total_actions: int
    avg_session_duration: float
    most_common_actions: List[Tuple[str, int]]
    peak_activity_hours: List[int]
    productivity_score: float
    last_activity: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "total_sessions": self.total_sessions,
            "total_actions": self.total_actions,
            "avg_session_duration": self.avg_session_duration,
            "most_common_actions": self.most_common_actions,
            "peak_activity_hours": self.peak_activity_hours,
            "productivity_score": self.productivity_score,
            "last_activity": self.last_activity
        }


class UserAnalytics:
    """
    Comprehensive user behavior analysis and reporting system.
    
    Tracks user activities, analyzes behavior patterns, and generates
    detailed reports for management insights.
    """
    
    def __init__(self):
        self.sessions: Dict[str, UserSession] = {}
        self.actions: deque = deque(maxlen=10000)  # Keep last 10k actions
        self.user_profiles: Dict[str, UserBehaviorProfile] = {}
        
        # Analytics configuration
        self.session_timeout = 1800  # 30 minutes
        self.analytics_interval = 300  # 5 minutes
        self.is_running = False
        self._analytics_task: Optional[asyncio.Task] = None
        
        # Behavior analysis data
        self.hourly_activity: Dict[int, int] = defaultdict(int)
        self.daily_activity: Dict[str, int] = defaultdict(int)
        self.action_patterns: Dict[str, List[str]] = defaultdict(list)
        
    async def initialize(self):
        """Initialize user analytics service."""
        if self.is_running:
            return
        
        self.is_running = True
        self._analytics_task = asyncio.create_task(self._analytics_loop())
        logger.info("User analytics service initialized")
    
    async def shutdown(self):
        """Shutdown user analytics service."""
        self.is_running = False
        
        if self._analytics_task:
            self._analytics_task.cancel()
            try:
                await self._analytics_task
            except asyncio.CancelledError:
                pass
        
        logger.info("User analytics service shutdown")
    
    async def _analytics_loop(self):
        """Main analytics processing loop."""
        while self.is_running:
            try:
                await self._process_analytics()
                await self._cleanup_old_data()
                await asyncio.sleep(self.analytics_interval)
            except Exception as e:
                logger.error(f"Error in analytics loop: {e}")
                await asyncio.sleep(30)  # Wait before retrying
    
    async def _process_analytics(self):
        """Process user analytics and update profiles."""
        current_time = time.time()
        
        # Update user behavior profiles
        for user_id in set(session.user_id for session in self.sessions.values()):
            await self._update_user_profile(user_id, current_time)
        
        # Update activity patterns
        self._update_activity_patterns()
        
        # Clean up expired sessions
        self._cleanup_expired_sessions(current_time)
    
    async def _update_user_profile(self, user_id: str, current_time: float):
        """Update behavior profile for a specific user."""
        user_sessions = [s for s in self.sessions.values() if s.user_id == user_id]
        user_actions = [a for a in self.actions if a.user_id == user_id]
        
        if not user_sessions and not user_actions:
            return
        
        # Calculate session statistics
        total_sessions = len(user_sessions)
        avg_session_duration = 0
        if user_sessions:
            durations = [s.duration for s in user_sessions if s.duration > 0]
            avg_session_duration = statistics.mean(durations) if durations else 0
        
        # Calculate action statistics
        total_actions = len(user_actions)
        action_counts = defaultdict(int)
        for action in user_actions:
            action_counts[action.action_type.value] += 1
        
        most_common_actions = sorted(
            action_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        # Calculate peak activity hours
        hourly_counts = defaultdict(int)
        for action in user_actions:
            hour = datetime.fromtimestamp(action.timestamp).hour
            hourly_counts[hour] += 1
        
        peak_hours = sorted(
            hourly_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        peak_activity_hours = [hour for hour, _ in peak_hours]
        
        # Calculate productivity score (simplified)
        productivity_score = self._calculate_productivity_score(user_actions, user_sessions)
        
        # Get last activity time
        last_activity = max(
            (max(s.last_activity for s in user_sessions) if user_sessions else 0),
            (max(a.timestamp for a in user_actions) if user_actions else 0)
        )
        
        # Update profile
        self.user_profiles[user_id] = UserBehaviorProfile(
            user_id=user_id,
            total_sessions=total_sessions,
            total_actions=total_actions,
            avg_session_duration=avg_session_duration,
            most_common_actions=most_common_actions,
            peak_activity_hours=peak_activity_hours,
            productivity_score=productivity_score,
            last_activity=last_activity
        )
    
    def _calculate_productivity_score(
        self, 
        user_actions: List[UserAction], 
        user_sessions: List[UserSession]
    ) -> float:
        """Calculate user productivity score (0-100)."""
        if not user_actions or not user_sessions:
            return 0.0
        
        # Base score from action frequency
        total_time = sum(s.duration for s in user_sessions if s.duration > 0)
        if total_time == 0:
            return 0.0
        
        actions_per_hour = (len(user_actions) / total_time) * 3600
        
        # Weight different action types
        action_weights = {
            ActionType.ANNOTATION: 10,
            ActionType.TASK_COMPLETE: 15,
            ActionType.QUALITY_REVIEW: 12,
            ActionType.DOCUMENT_UPLOAD: 8,
            ActionType.PROJECT_CREATE: 20,
            ActionType.AI_INFERENCE: 5,
            ActionType.EXPORT_DATA: 8,
            ActionType.SEARCH: 2,
            ActionType.NAVIGATION: 1
        }
        
        weighted_score = 0
        for action in user_actions:
            weight = action_weights.get(action.action_type, 1)
            weighted_score += weight * (1 if action.success else 0.5)
        
        # Normalize to 0-100 scale
        base_score = min(100, (weighted_score / len(user_actions)) * 5)
        
        # Apply time-based adjustments
        time_efficiency = min(1.0, actions_per_hour / 10)  # Optimal: 10 actions/hour
        
        return base_score * time_efficiency
    
    def _update_activity_patterns(self):
        """Update global activity patterns."""
        current_time = time.time()
        cutoff_time = current_time - 86400  # Last 24 hours
        
        # Reset counters
        self.hourly_activity.clear()
        self.daily_activity.clear()
        
        # Process recent actions
        for action in self.actions:
            if action.timestamp > cutoff_time:
                dt = datetime.fromtimestamp(action.timestamp)
                hour = dt.hour
                day = dt.strftime("%Y-%m-%d")
                
                self.hourly_activity[hour] += 1
                self.daily_activity[day] += 1
    
    def _cleanup_expired_sessions(self, current_time: float):
        """Clean up expired sessions."""
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if (current_time - session.last_activity) > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
    
    async def _cleanup_old_data(self):
        """Clean up old analytics data."""
        cutoff_time = time.time() - (7 * 24 * 3600)  # 7 days
        
        # Remove old actions (deque handles this automatically with maxlen)
        # Remove old user profiles that haven't been active
        inactive_users = []
        for user_id, profile in self.user_profiles.items():
            if profile.last_activity < cutoff_time:
                inactive_users.append(user_id)
        
        for user_id in inactive_users:
            del self.user_profiles[user_id]
    
    # Public API methods
    def start_session(
        self, 
        session_id: str, 
        user_id: str, 
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        tenant_id: Optional[str] = None
    ):
        """Start tracking a user session."""
        current_time = time.time()
        
        self.sessions[session_id] = UserSession(
            session_id=session_id,
            user_id=user_id,
            start_time=current_time,
            last_activity=current_time,
            ip_address=ip_address,
            user_agent=user_agent,
            tenant_id=tenant_id
        )
        
        # Track login action
        self.track_action(
            user_id=user_id,
            session_id=session_id,
            action_type=ActionType.LOGIN,
            details={
                "ip_address": ip_address,
                "user_agent": user_agent,
                "tenant_id": tenant_id
            }
        )
        
        logger.debug(f"Started session tracking for user {user_id}")
    
    def end_session(self, session_id: str):
        """End tracking a user session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Track logout action
            self.track_action(
                user_id=session.user_id,
                session_id=session_id,
                action_type=ActionType.LOGOUT,
                details={"session_duration": session.duration}
            )
            
            del self.sessions[session_id]
            logger.debug(f"Ended session tracking for session {session_id}")
    
    def track_action(
        self,
        user_id: str,
        session_id: str,
        action_type: ActionType,
        details: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        success: bool = True
    ):
        """Track a user action."""
        current_time = time.time()
        
        action = UserAction(
            action_id=f"{user_id}_{session_id}_{int(current_time * 1000)}",
            user_id=user_id,
            session_id=session_id,
            action_type=action_type,
            timestamp=current_time,
            details=details or {},
            duration_ms=duration_ms,
            success=success
        )
        
        self.actions.append(action)
        
        # Update session activity
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.last_activity = current_time
            session.actions_count += 1
        
        # Also track with business metrics collector
        business_metrics_collector.track_user_action(
            user_id, action_type.value, details
        )
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current user analytics statistics."""
        current_time = time.time()
        
        # Active sessions
        active_sessions = [s for s in self.sessions.values() if s.is_active]
        
        # Recent actions (last hour)
        recent_actions = [
            a for a in self.actions 
            if (current_time - a.timestamp) < 3600
        ]
        
        # User activity summary
        unique_users_today = len(set(
            a.user_id for a in self.actions
            if (current_time - a.timestamp) < 86400
        ))
        
        return {
            "active_sessions": len(active_sessions),
            "total_sessions": len(self.sessions),
            "recent_actions": len(recent_actions),
            "unique_users_today": unique_users_today,
            "total_tracked_users": len(self.user_profiles),
            "hourly_activity": dict(self.hourly_activity),
            "peak_hour": max(self.hourly_activity.items(), key=lambda x: x[1])[0] if self.hourly_activity else 0
        }
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get behavior profile for a specific user."""
        if user_id in self.user_profiles:
            return self.user_profiles[user_id].to_dict()
        return None
    
    def get_user_activity_report(
        self, 
        user_id: str, 
        days: int = 7
    ) -> Dict[str, Any]:
        """Generate detailed activity report for a user."""
        cutoff_time = time.time() - (days * 24 * 3600)
        
        user_actions = [
            a for a in self.actions 
            if a.user_id == user_id and a.timestamp > cutoff_time
        ]
        
        user_sessions = [
            s for s in self.sessions.values() 
            if s.user_id == user_id and s.start_time > cutoff_time
        ]
        
        # Daily activity breakdown
        daily_breakdown = defaultdict(lambda: {"actions": 0, "session_time": 0})
        
        for action in user_actions:
            day = datetime.fromtimestamp(action.timestamp).strftime("%Y-%m-%d")
            daily_breakdown[day]["actions"] += 1
        
        for session in user_sessions:
            day = datetime.fromtimestamp(session.start_time).strftime("%Y-%m-%d")
            daily_breakdown[day]["session_time"] += session.duration
        
        # Action type distribution
        action_distribution = defaultdict(int)
        for action in user_actions:
            action_distribution[action.action_type.value] += 1
        
        # Performance metrics
        successful_actions = sum(1 for a in user_actions if a.success)
        success_rate = (successful_actions / len(user_actions)) if user_actions else 0
        
        avg_action_duration = 0
        if user_actions:
            durations = [a.duration_ms for a in user_actions if a.duration_ms is not None]
            avg_action_duration = statistics.mean(durations) if durations else 0
        
        return {
            "user_id": user_id,
            "period_days": days,
            "total_actions": len(user_actions),
            "total_sessions": len(user_sessions),
            "success_rate": success_rate,
            "avg_action_duration_ms": avg_action_duration,
            "daily_breakdown": dict(daily_breakdown),
            "action_distribution": dict(action_distribution),
            "profile": self.get_user_profile(user_id)
        }
    
    def get_system_activity_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate system-wide activity report."""
        cutoff_time = time.time() - (days * 24 * 3600)
        
        recent_actions = [
            a for a in self.actions 
            if a.timestamp > cutoff_time
        ]
        
        recent_sessions = [
            s for s in self.sessions.values() 
            if s.start_time > cutoff_time
        ]
        
        # User engagement metrics
        unique_users = len(set(a.user_id for a in recent_actions))
        total_actions = len(recent_actions)
        total_sessions = len(recent_sessions)
        
        # Activity trends
        daily_users = defaultdict(set)
        daily_actions = defaultdict(int)
        
        for action in recent_actions:
            day = datetime.fromtimestamp(action.timestamp).strftime("%Y-%m-%d")
            daily_users[day].add(action.user_id)
            daily_actions[day] += 1
        
        # Convert sets to counts
        daily_user_counts = {day: len(users) for day, users in daily_users.items()}
        
        # Top users by activity
        user_action_counts = defaultdict(int)
        for action in recent_actions:
            user_action_counts[action.user_id] += 1
        
        top_users = sorted(
            user_action_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Most common actions
        action_counts = defaultdict(int)
        for action in recent_actions:
            action_counts[action.action_type.value] += 1
        
        top_actions = sorted(
            action_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            "period_days": days,
            "summary": {
                "unique_users": unique_users,
                "total_actions": total_actions,
                "total_sessions": total_sessions,
                "avg_actions_per_user": total_actions / unique_users if unique_users > 0 else 0,
                "avg_session_duration": statistics.mean([s.duration for s in recent_sessions]) if recent_sessions else 0
            },
            "trends": {
                "daily_users": daily_user_counts,
                "daily_actions": dict(daily_actions),
                "hourly_activity": dict(self.hourly_activity)
            },
            "top_users": top_users,
            "top_actions": top_actions,
            "user_profiles_count": len(self.user_profiles)
        }
    
    def get_real_time_activity(self) -> Dict[str, Any]:
        """Get real-time activity dashboard data."""
        current_time = time.time()
        
        # Last 5 minutes activity
        recent_cutoff = current_time - 300
        recent_actions = [
            a for a in self.actions 
            if a.timestamp > recent_cutoff
        ]
        
        # Active sessions
        active_sessions = [s for s in self.sessions.values() if s.is_active]
        
        # Recent action types
        recent_action_types = defaultdict(int)
        for action in recent_actions:
            recent_action_types[action.action_type.value] += 1
        
        return {
            "timestamp": current_time,
            "active_users": len(active_sessions),
            "recent_actions": len(recent_actions),
            "actions_per_minute": len(recent_actions) / 5,
            "recent_action_types": dict(recent_action_types),
            "active_sessions_details": [
                {
                    "user_id": s.user_id,
                    "duration": s.duration,
                    "actions_count": s.actions_count,
                    "tenant_id": s.tenant_id
                }
                for s in active_sessions
            ]
        }