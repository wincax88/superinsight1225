"""
Behavior Analyzer for Knowledge Graph.

Provides behavior analysis capabilities including:
- User annotation behavior analysis
- User capability and preference modeling
- Collaboration pattern analysis
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

from .process_miner import Event, Trace, EventLog

logger = logging.getLogger(__name__)


class UserExpertiseLevel(str, Enum):
    """User expertise levels."""
    NOVICE = "novice"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class CollaborationType(str, Enum):
    """Types of collaboration patterns."""
    SEQUENTIAL = "sequential"  # Users work on same item in sequence
    PARALLEL = "parallel"  # Users work on same item simultaneously
    REVIEW = "review"  # One user reviews another's work
    HANDOFF = "handoff"  # Work is handed off between users
    MENTORING = "mentoring"  # Expert helps novice


class WorkloadDistribution(str, Enum):
    """Workload distribution patterns."""
    BALANCED = "balanced"
    CONCENTRATED = "concentrated"
    SPARSE = "sparse"


@dataclass
class UserProfile:
    """Profile of a user's behavior and capabilities."""

    user_id: str = ""
    total_events: int = 0
    total_cases: int = 0
    activity_distribution: Dict[str, int] = field(default_factory=dict)
    avg_task_duration_seconds: float = 0.0
    expertise_level: UserExpertiseLevel = UserExpertiseLevel.BEGINNER
    expertise_score: float = 0.0  # 0-1 score
    preferred_activities: List[str] = field(default_factory=list)
    active_hours: List[int] = field(default_factory=list)  # Hours of day when most active
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    collaboration_score: float = 0.0  # 0-1 score indicating collaboration tendency
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "total_events": self.total_events,
            "total_cases": self.total_cases,
            "activity_distribution": self.activity_distribution,
            "avg_task_duration_seconds": self.avg_task_duration_seconds,
            "expertise_level": self.expertise_level.value,
            "expertise_score": self.expertise_score,
            "preferred_activities": self.preferred_activities,
            "active_hours": self.active_hours,
            "quality_metrics": self.quality_metrics,
            "collaboration_score": self.collaboration_score,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class CollaborationPattern:
    """Collaboration pattern between users."""

    pattern_id: str = field(default_factory=lambda: str(uuid4()))
    collaboration_type: CollaborationType = CollaborationType.SEQUENTIAL
    user_ids: List[str] = field(default_factory=list)
    frequency: int = 0
    avg_handoff_time_seconds: float = 0.0
    case_ids: List[str] = field(default_factory=list)
    effectiveness_score: float = 0.0  # 0-1 score
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "collaboration_type": self.collaboration_type.value,
            "user_ids": self.user_ids,
            "frequency": self.frequency,
            "avg_handoff_time_seconds": self.avg_handoff_time_seconds,
            "case_ids": self.case_ids[:10],  # Limit for readability
            "effectiveness_score": self.effectiveness_score,
            "description": self.description,
        }


@dataclass
class TeamMetrics:
    """Metrics for a team of users."""

    team_id: str = ""
    user_ids: List[str] = field(default_factory=list)
    total_events: int = 0
    total_cases: int = 0
    workload_distribution: WorkloadDistribution = WorkloadDistribution.BALANCED
    gini_coefficient: float = 0.0  # Inequality measure (0 = equal, 1 = unequal)
    avg_collaboration_score: float = 0.0
    top_performers: List[str] = field(default_factory=list)
    bottleneck_users: List[str] = field(default_factory=list)
    expertise_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "team_id": self.team_id,
            "user_ids": self.user_ids,
            "total_events": self.total_events,
            "total_cases": self.total_cases,
            "workload_distribution": self.workload_distribution.value,
            "gini_coefficient": self.gini_coefficient,
            "avg_collaboration_score": self.avg_collaboration_score,
            "top_performers": self.top_performers,
            "bottleneck_users": self.bottleneck_users,
            "expertise_distribution": self.expertise_distribution,
        }


class BehaviorAnalysisResult(BaseModel):
    """Result of behavior analysis."""

    user_profiles: List[Dict[str, Any]] = Field(default_factory=list, description="User profiles")
    collaboration_patterns: List[Dict[str, Any]] = Field(default_factory=list, description="Collaboration patterns")
    team_metrics: Dict[str, Any] = Field(default_factory=dict, description="Team-level metrics")
    behavior_insights: List[Dict[str, Any]] = Field(default_factory=list, description="Behavioral insights")
    recommendations: List[Dict[str, Any]] = Field(default_factory=list, description="Recommendations")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Analysis statistics")
    processing_time_ms: float = Field(default=0.0, description="Processing time in milliseconds")


class BehaviorAnalyzer:
    """
    Behavior analyzer for user annotation patterns.

    Analyzes user behavior, builds capability models, and detects collaboration patterns.
    """

    def __init__(
        self,
        min_events_for_profile: int = 5,
        expertise_thresholds: Optional[Dict[UserExpertiseLevel, float]] = None,
        collaboration_time_threshold_seconds: float = 3600.0,  # 1 hour
    ):
        """
        Initialize BehaviorAnalyzer.

        Args:
            min_events_for_profile: Minimum events required to build user profile
            expertise_thresholds: Score thresholds for expertise levels
            collaboration_time_threshold_seconds: Max time between events for collaboration
        """
        self.min_events_for_profile = min_events_for_profile
        self.expertise_thresholds = expertise_thresholds or {
            UserExpertiseLevel.EXPERT: 0.9,
            UserExpertiseLevel.ADVANCED: 0.7,
            UserExpertiseLevel.INTERMEDIATE: 0.5,
            UserExpertiseLevel.BEGINNER: 0.3,
            UserExpertiseLevel.NOVICE: 0.0,
        }
        self.collaboration_time_threshold = collaboration_time_threshold_seconds
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the analyzer."""
        if self._initialized:
            return
        self._initialized = True
        logger.info("BehaviorAnalyzer initialized")

    def analyze(self, event_log: EventLog) -> BehaviorAnalysisResult:
        """
        Perform comprehensive behavior analysis.

        Args:
            event_log: Event log to analyze

        Returns:
            BehaviorAnalysisResult with profiles, patterns, and insights
        """
        if not self._initialized:
            self.initialize()

        start_time = datetime.now()

        # Build user profiles
        user_profiles = self._build_user_profiles(event_log)

        # Detect collaboration patterns
        collaboration_patterns = self._detect_collaboration_patterns(event_log, user_profiles)

        # Calculate team metrics
        team_metrics = self._calculate_team_metrics(event_log, user_profiles)

        # Generate insights
        insights = self._generate_insights(user_profiles, collaboration_patterns, team_metrics)

        # Generate recommendations
        recommendations = self._generate_recommendations(user_profiles, collaboration_patterns, team_metrics)

        # Calculate statistics
        statistics = self._calculate_statistics(event_log, user_profiles, collaboration_patterns)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return BehaviorAnalysisResult(
            user_profiles=[p.to_dict() for p in user_profiles],
            collaboration_patterns=[c.to_dict() for c in collaboration_patterns],
            team_metrics=team_metrics.to_dict(),
            behavior_insights=insights,
            recommendations=recommendations,
            statistics=statistics,
            processing_time_ms=processing_time,
        )

    def _build_user_profiles(self, event_log: EventLog) -> List[UserProfile]:
        """
        Build profiles for each user based on their behavior.

        Args:
            event_log: Event log to analyze

        Returns:
            List of user profiles
        """
        # Collect user statistics
        user_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "events": [],
            "cases": set(),
            "activities": defaultdict(int),
            "durations": [],
            "hours": [],
        })

        for trace in event_log.traces:
            sorted_events = sorted(trace.events, key=lambda e: e.timestamp)

            for i, event in enumerate(sorted_events):
                if not event.resource:
                    continue

                user_id = event.resource
                stats = user_stats[user_id]

                stats["events"].append(event)
                stats["cases"].add(trace.case_id)
                stats["activities"][event.activity] += 1
                stats["hours"].append(event.timestamp.hour)

                # Calculate duration to next event
                if i < len(sorted_events) - 1:
                    duration = (sorted_events[i + 1].timestamp - event.timestamp).total_seconds()
                    if duration > 0 and duration < 86400:  # Ignore > 1 day
                        stats["durations"].append(duration)

        # Build profiles
        profiles = []

        for user_id, stats in user_stats.items():
            if len(stats["events"]) < self.min_events_for_profile:
                continue

            # Calculate metrics
            total_events = len(stats["events"])
            total_cases = len(stats["cases"])
            activity_dist = dict(stats["activities"])

            avg_duration = sum(stats["durations"]) / len(stats["durations"]) if stats["durations"] else 0

            # Determine preferred activities (top 3)
            sorted_activities = sorted(activity_dist.items(), key=lambda x: -x[1])
            preferred = [a[0] for a in sorted_activities[:3]]

            # Find active hours (peak hours)
            hour_counts = defaultdict(int)
            for hour in stats["hours"]:
                hour_counts[hour] += 1
            sorted_hours = sorted(hour_counts.items(), key=lambda x: -x[1])
            active_hours = [h[0] for h in sorted_hours[:3]]

            # Calculate expertise score
            expertise_score = self._calculate_expertise_score(stats)

            # Determine expertise level
            expertise_level = self._determine_expertise_level(expertise_score)

            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(stats)

            profiles.append(UserProfile(
                user_id=user_id,
                total_events=total_events,
                total_cases=total_cases,
                activity_distribution=activity_dist,
                avg_task_duration_seconds=avg_duration,
                expertise_level=expertise_level,
                expertise_score=expertise_score,
                preferred_activities=preferred,
                active_hours=active_hours,
                quality_metrics=quality_metrics,
            ))

        # Sort by expertise score
        profiles.sort(key=lambda p: -p.expertise_score)

        return profiles

    def _calculate_expertise_score(self, stats: Dict[str, Any]) -> float:
        """
        Calculate expertise score for a user.

        Factors:
        - Volume of work (events and cases)
        - Activity diversity
        - Consistency (low variance in duration)
        """
        scores = []

        # Volume score (normalized)
        events = len(stats["events"])
        volume_score = min(1.0, events / 100)  # Cap at 100 events
        scores.append(volume_score * 0.3)

        # Diversity score
        unique_activities = len(stats["activities"])
        diversity_score = min(1.0, unique_activities / 10)  # Cap at 10 activities
        scores.append(diversity_score * 0.3)

        # Consistency score (lower variance is better)
        if stats["durations"]:
            durations = stats["durations"]
            avg = sum(durations) / len(durations)
            if avg > 0:
                cv = (sum((d - avg) ** 2 for d in durations) / len(durations)) ** 0.5 / avg
                consistency_score = max(0, 1 - cv)
                scores.append(consistency_score * 0.2)
            else:
                scores.append(0.2)
        else:
            scores.append(0.0)

        # Experience score (based on cases)
        cases = len(stats["cases"])
        experience_score = min(1.0, cases / 20)  # Cap at 20 cases
        scores.append(experience_score * 0.2)

        return sum(scores)

    def _determine_expertise_level(self, score: float) -> UserExpertiseLevel:
        """Determine expertise level from score."""
        for level, threshold in sorted(self.expertise_thresholds.items(),
                                       key=lambda x: -x[1]):
            if score >= threshold:
                return level
        return UserExpertiseLevel.NOVICE

    def _calculate_quality_metrics(self, stats: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality metrics for a user."""
        metrics = {}

        # Throughput (events per day)
        events = stats["events"]
        if events:
            first_event = min(e.timestamp for e in events)
            last_event = max(e.timestamp for e in events)
            days = max(1, (last_event - first_event).days)
            metrics["throughput_per_day"] = len(events) / days
        else:
            metrics["throughput_per_day"] = 0

        # Average task time
        if stats["durations"]:
            metrics["avg_task_time_seconds"] = sum(stats["durations"]) / len(stats["durations"])
        else:
            metrics["avg_task_time_seconds"] = 0

        # Activity coverage (proportion of unique activities)
        all_activities = len(stats["activities"])
        metrics["activity_coverage"] = all_activities / max(1, sum(stats["activities"].values()))

        return metrics

    def _detect_collaboration_patterns(
        self,
        event_log: EventLog,
        user_profiles: List[UserProfile],
    ) -> List[CollaborationPattern]:
        """
        Detect collaboration patterns between users.

        Args:
            event_log: Event log to analyze
            user_profiles: User profiles

        Returns:
            List of collaboration patterns
        """
        patterns = []

        # Track user interactions per case
        case_interactions: Dict[str, List[Tuple[str, str, datetime]]] = defaultdict(list)

        for trace in event_log.traces:
            sorted_events = sorted(trace.events, key=lambda e: e.timestamp)

            for i in range(len(sorted_events) - 1):
                event_a = sorted_events[i]
                event_b = sorted_events[i + 1]

                if event_a.resource and event_b.resource and event_a.resource != event_b.resource:
                    case_interactions[trace.case_id].append((
                        event_a.resource,
                        event_b.resource,
                        event_b.timestamp,
                    ))

        # Analyze handoff patterns
        handoff_counts: Dict[Tuple[str, str], List[Tuple[str, float]]] = defaultdict(list)

        for case_id, interactions in case_interactions.items():
            for user_a, user_b, timestamp in interactions:
                # Look for handoff time (time between user switches)
                handoff_counts[(user_a, user_b)].append((case_id, 0))  # Simplified

        # Create collaboration patterns
        for (user_a, user_b), cases in handoff_counts.items():
            if len(cases) < 2:
                continue

            # Determine collaboration type
            collab_type = CollaborationType.HANDOFF

            # Check for review pattern (if one user is expert and other is novice)
            user_a_profile = next((p for p in user_profiles if p.user_id == user_a), None)
            user_b_profile = next((p for p in user_profiles if p.user_id == user_b), None)

            if user_a_profile and user_b_profile:
                if (user_a_profile.expertise_level in [UserExpertiseLevel.EXPERT, UserExpertiseLevel.ADVANCED] and
                    user_b_profile.expertise_level in [UserExpertiseLevel.NOVICE, UserExpertiseLevel.BEGINNER]):
                    collab_type = CollaborationType.MENTORING
                elif "review" in [a.lower() for a in user_b_profile.preferred_activities]:
                    collab_type = CollaborationType.REVIEW

            patterns.append(CollaborationPattern(
                collaboration_type=collab_type,
                user_ids=[user_a, user_b],
                frequency=len(cases),
                case_ids=[c[0] for c in cases],
                description=f"{user_a} -> {user_b} ({collab_type.value})",
            ))

        # Sort by frequency
        patterns.sort(key=lambda p: -p.frequency)

        return patterns

    def _calculate_team_metrics(
        self,
        event_log: EventLog,
        user_profiles: List[UserProfile],
    ) -> TeamMetrics:
        """
        Calculate team-level metrics.

        Args:
            event_log: Event log to analyze
            user_profiles: User profiles

        Returns:
            TeamMetrics
        """
        user_ids = [p.user_id for p in user_profiles]
        total_events = sum(p.total_events for p in user_profiles)
        total_cases = len(set(c for p in user_profiles for c in event_log.traces if any(
            e.resource == p.user_id for e in c.events
        )))

        # Calculate Gini coefficient for workload distribution
        workloads = [p.total_events for p in user_profiles]
        gini = self._calculate_gini(workloads) if workloads else 0

        # Determine workload distribution type
        if gini < 0.2:
            distribution = WorkloadDistribution.BALANCED
        elif gini > 0.5:
            distribution = WorkloadDistribution.CONCENTRATED
        else:
            distribution = WorkloadDistribution.SPARSE

        # Calculate average collaboration score
        avg_collab = sum(p.collaboration_score for p in user_profiles) / len(user_profiles) if user_profiles else 0

        # Find top performers (top 20%)
        sorted_by_score = sorted(user_profiles, key=lambda p: -p.expertise_score)
        top_count = max(1, len(sorted_by_score) // 5)
        top_performers = [p.user_id for p in sorted_by_score[:top_count]]

        # Find bottlenecks (high workload, slow average time)
        bottleneck_users = []
        if user_profiles:
            avg_throughput = sum(p.quality_metrics.get("throughput_per_day", 0)
                                for p in user_profiles) / len(user_profiles)
            for profile in user_profiles:
                if (profile.total_events > avg_throughput * 1.5 and
                    profile.avg_task_duration_seconds > sum(p.avg_task_duration_seconds
                                                            for p in user_profiles) / len(user_profiles) * 1.5):
                    bottleneck_users.append(profile.user_id)

        # Expertise distribution
        expertise_dist = defaultdict(int)
        for profile in user_profiles:
            expertise_dist[profile.expertise_level.value] += 1

        return TeamMetrics(
            team_id="default_team",
            user_ids=user_ids,
            total_events=total_events,
            total_cases=total_cases,
            workload_distribution=distribution,
            gini_coefficient=gini,
            avg_collaboration_score=avg_collab,
            top_performers=top_performers,
            bottleneck_users=bottleneck_users,
            expertise_distribution=dict(expertise_dist),
        )

    def _calculate_gini(self, values: List[float]) -> float:
        """Calculate Gini coefficient for inequality measurement."""
        if not values or sum(values) == 0:
            return 0

        n = len(values)
        sorted_values = sorted(values)
        cumulative = 0
        total = sum(sorted_values)

        for i, value in enumerate(sorted_values):
            cumulative += (n - i) * value

        return (n + 1 - 2 * cumulative / total) / n

    def _generate_insights(
        self,
        user_profiles: List[UserProfile],
        collaboration_patterns: List[CollaborationPattern],
        team_metrics: TeamMetrics,
    ) -> List[Dict[str, Any]]:
        """Generate behavioral insights."""
        insights = []

        # Workload distribution insight
        if team_metrics.gini_coefficient > 0.4:
            insights.append({
                "type": "workload_imbalance",
                "severity": "high" if team_metrics.gini_coefficient > 0.6 else "medium",
                "message": f"Workload is unevenly distributed (Gini: {team_metrics.gini_coefficient:.2f}). "
                          f"Consider redistributing tasks.",
                "affected_users": team_metrics.top_performers[:3],
            })

        # Expertise gap insight
        if team_metrics.expertise_distribution:
            novice_count = team_metrics.expertise_distribution.get("novice", 0)
            expert_count = team_metrics.expertise_distribution.get("expert", 0)

            if novice_count > expert_count * 2:
                insights.append({
                    "type": "expertise_gap",
                    "severity": "medium",
                    "message": f"Team has {novice_count} novice users but only {expert_count} experts. "
                              f"Consider training programs or mentoring.",
                    "recommendation": "Implement structured mentoring program",
                })

        # Collaboration pattern insights
        mentoring_patterns = [p for p in collaboration_patterns
                            if p.collaboration_type == CollaborationType.MENTORING]
        if mentoring_patterns:
            insights.append({
                "type": "active_mentoring",
                "severity": "positive",
                "message": f"Found {len(mentoring_patterns)} active mentoring relationships.",
                "patterns": [p.to_dict() for p in mentoring_patterns[:3]],
            })

        # Bottleneck insight
        if team_metrics.bottleneck_users:
            insights.append({
                "type": "bottleneck_detected",
                "severity": "high",
                "message": f"Detected {len(team_metrics.bottleneck_users)} potential bottleneck users: "
                          f"{', '.join(team_metrics.bottleneck_users)}",
                "recommendation": "Review workload allocation and consider task reassignment",
            })

        # Peak hours insight
        if user_profiles:
            all_hours = []
            for p in user_profiles:
                all_hours.extend(p.active_hours)
            if all_hours:
                peak_hour = max(set(all_hours), key=all_hours.count)
                insights.append({
                    "type": "peak_activity_time",
                    "severity": "info",
                    "message": f"Peak activity time is around {peak_hour}:00",
                    "hour": peak_hour,
                })

        return insights

    def _generate_recommendations(
        self,
        user_profiles: List[UserProfile],
        collaboration_patterns: List[CollaborationPattern],
        team_metrics: TeamMetrics,
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Workload balancing
        if team_metrics.gini_coefficient > 0.4:
            recommendations.append({
                "type": "workload_balance",
                "priority": "high",
                "title": "Balance Workload Distribution",
                "description": "Redistribute tasks from overloaded users to underutilized team members.",
                "actions": [
                    f"Review assignments for {', '.join(team_metrics.top_performers[:2])}",
                    "Implement automated task distribution",
                    "Set workload caps per user",
                ],
            })

        # Skill development
        novice_users = [p for p in user_profiles
                       if p.expertise_level in [UserExpertiseLevel.NOVICE, UserExpertiseLevel.BEGINNER]]
        if novice_users:
            recommendations.append({
                "type": "skill_development",
                "priority": "medium",
                "title": "Develop Team Skills",
                "description": f"Support skill development for {len(novice_users)} newer team members.",
                "actions": [
                    "Create training documentation for common tasks",
                    "Pair novice users with experts for mentoring",
                    "Implement graduated task complexity",
                ],
                "affected_users": [u.user_id for u in novice_users[:5]],
            })

        # Collaboration enhancement
        if len(collaboration_patterns) < len(user_profiles) // 2:
            recommendations.append({
                "type": "collaboration",
                "priority": "medium",
                "title": "Enhance Team Collaboration",
                "description": "Limited collaboration patterns detected. Consider fostering more teamwork.",
                "actions": [
                    "Implement pair annotations for complex tasks",
                    "Create shared task queues",
                    "Schedule regular team sync meetings",
                ],
            })

        # Process optimization
        if user_profiles:
            slow_users = [p for p in user_profiles
                        if p.avg_task_duration_seconds > sum(u.avg_task_duration_seconds
                                                             for u in user_profiles) / len(user_profiles) * 2]
            if slow_users:
                recommendations.append({
                    "type": "process_optimization",
                    "priority": "medium",
                    "title": "Optimize Task Completion Time",
                    "description": f"{len(slow_users)} users have above-average task completion times.",
                    "actions": [
                        "Review task complexity assignments",
                        "Provide additional training or tools",
                        "Analyze workflow for optimization opportunities",
                    ],
                })

        return recommendations

    def _calculate_statistics(
        self,
        event_log: EventLog,
        user_profiles: List[UserProfile],
        collaboration_patterns: List[CollaborationPattern],
    ) -> Dict[str, Any]:
        """Calculate analysis statistics."""
        return {
            "total_users_analyzed": len(user_profiles),
            "total_collaboration_patterns": len(collaboration_patterns),
            "avg_expertise_score": sum(p.expertise_score for p in user_profiles) / len(user_profiles) if user_profiles else 0,
            "expertise_distribution": {
                level.value: len([p for p in user_profiles if p.expertise_level == level])
                for level in UserExpertiseLevel
            },
            "collaboration_type_distribution": {
                ctype.value: len([c for c in collaboration_patterns if c.collaboration_type == ctype])
                for ctype in CollaborationType
            },
            "most_active_users": [p.user_id for p in sorted(user_profiles, key=lambda x: -x.total_events)[:5]],
            "highest_expertise_users": [p.user_id for p in sorted(user_profiles, key=lambda x: -x.expertise_score)[:5]],
        }

    def build_user_profile(self, user_id: str, event_log: EventLog) -> Optional[UserProfile]:
        """
        Build profile for a specific user.

        Args:
            user_id: User ID to analyze
            event_log: Event log containing user's events

        Returns:
            UserProfile or None if insufficient data
        """
        if not self._initialized:
            self.initialize()

        # Filter events for this user
        user_events = []
        user_cases = set()

        for trace in event_log.traces:
            for event in trace.events:
                if event.resource == user_id:
                    user_events.append(event)
                    user_cases.add(trace.case_id)

        if len(user_events) < self.min_events_for_profile:
            return None

        # Build profile from filtered data
        stats = {
            "events": user_events,
            "cases": user_cases,
            "activities": defaultdict(int),
            "durations": [],
            "hours": [],
        }

        for event in user_events:
            stats["activities"][event.activity] += 1
            stats["hours"].append(event.timestamp.hour)

        expertise_score = self._calculate_expertise_score(stats)
        expertise_level = self._determine_expertise_level(expertise_score)
        quality_metrics = self._calculate_quality_metrics(stats)

        sorted_activities = sorted(stats["activities"].items(), key=lambda x: -x[1])
        preferred = [a[0] for a in sorted_activities[:3]]

        hour_counts = defaultdict(int)
        for hour in stats["hours"]:
            hour_counts[hour] += 1
        sorted_hours = sorted(hour_counts.items(), key=lambda x: -x[1])
        active_hours = [h[0] for h in sorted_hours[:3]]

        return UserProfile(
            user_id=user_id,
            total_events=len(user_events),
            total_cases=len(user_cases),
            activity_distribution=dict(stats["activities"]),
            expertise_level=expertise_level,
            expertise_score=expertise_score,
            preferred_activities=preferred,
            active_hours=active_hours,
            quality_metrics=quality_metrics,
        )


# Global instance
_behavior_analyzer: Optional[BehaviorAnalyzer] = None


def get_behavior_analyzer() -> BehaviorAnalyzer:
    """Get or create global BehaviorAnalyzer instance."""
    global _behavior_analyzer

    if _behavior_analyzer is None:
        _behavior_analyzer = BehaviorAnalyzer()
        _behavior_analyzer.initialize()

    return _behavior_analyzer
