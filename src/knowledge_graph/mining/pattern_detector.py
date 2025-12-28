"""
Pattern Detector for Knowledge Graph.

Provides pattern detection capabilities including:
- Temporal anomaly detection
- Sequence anomaly identification
- Pattern visualization data generation
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


class AnomalyType(str, Enum):
    """Types of anomalies detected."""
    TEMPORAL = "temporal"
    SEQUENCE = "sequence"
    FREQUENCY = "frequency"
    RESOURCE = "resource"
    DURATION = "duration"
    MISSING_ACTIVITY = "missing_activity"
    UNEXPECTED_ACTIVITY = "unexpected_activity"
    LOOP = "loop"
    DEADLOCK = "deadlock"


class AnomalySeverity(str, Enum):
    """Severity levels for anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Pattern:
    """Detected pattern in event data."""

    pattern_id: str = field(default_factory=lambda: str(uuid4()))
    pattern_type: str = ""
    activities: List[str] = field(default_factory=list)
    frequency: int = 0
    support: float = 0.0  # Support ratio (0-1)
    confidence: float = 0.0  # Pattern confidence (0-1)
    avg_duration: Optional[float] = None  # Average duration in seconds
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "activities": self.activities,
            "frequency": self.frequency,
            "support": self.support,
            "confidence": self.confidence,
            "avg_duration": self.avg_duration,
            "metadata": self.metadata,
        }


@dataclass
class Anomaly:
    """Detected anomaly in event data."""

    anomaly_id: str = field(default_factory=lambda: str(uuid4()))
    anomaly_type: AnomalyType = AnomalyType.TEMPORAL
    severity: AnomalySeverity = AnomalySeverity.LOW
    description: str = ""
    case_id: Optional[str] = None
    event_ids: List[str] = field(default_factory=list)
    detected_at: datetime = field(default_factory=datetime.now)
    score: float = 0.0  # Anomaly score (higher = more anomalous)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "anomaly_id": self.anomaly_id,
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "case_id": self.case_id,
            "event_ids": self.event_ids,
            "detected_at": self.detected_at.isoformat(),
            "score": self.score,
            "context": self.context,
        }


@dataclass
class TemporalPattern:
    """Temporal pattern between activities."""

    source_activity: str = ""
    target_activity: str = ""
    avg_interval_seconds: float = 0.0
    std_interval_seconds: float = 0.0
    min_interval_seconds: float = 0.0
    max_interval_seconds: float = 0.0
    occurrence_count: int = 0

    def is_anomalous(self, interval: float, threshold: float = 2.0) -> bool:
        """Check if an interval is anomalous based on z-score."""
        if self.std_interval_seconds == 0:
            return False
        z_score = abs(interval - self.avg_interval_seconds) / self.std_interval_seconds
        return z_score > threshold


class PatternDetectionResult(BaseModel):
    """Result of pattern detection analysis."""

    patterns: List[Dict[str, Any]] = Field(default_factory=list, description="Detected patterns")
    anomalies: List[Dict[str, Any]] = Field(default_factory=list, description="Detected anomalies")
    temporal_patterns: List[Dict[str, Any]] = Field(default_factory=list, description="Temporal patterns")
    sequence_patterns: List[Dict[str, Any]] = Field(default_factory=list, description="Sequence patterns")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Detection statistics")
    visualization_data: Dict[str, Any] = Field(default_factory=dict, description="Data for visualization")
    processing_time_ms: float = Field(default=0.0, description="Processing time in milliseconds")


class PatternDetector:
    """
    Pattern detector for temporal and sequence anomalies.

    Detects patterns and anomalies in process event data.
    """

    def __init__(
        self,
        temporal_threshold: float = 2.0,  # Z-score threshold for temporal anomalies
        sequence_min_support: float = 0.1,  # Minimum support for sequence patterns
        sequence_min_length: int = 2,  # Minimum sequence length
        sequence_max_length: int = 5,  # Maximum sequence length
        anomaly_score_threshold: float = 0.5,  # Threshold for reporting anomalies
    ):
        """
        Initialize PatternDetector.

        Args:
            temporal_threshold: Z-score threshold for temporal anomaly detection
            sequence_min_support: Minimum support ratio for sequence patterns
            sequence_min_length: Minimum length for sequence patterns
            sequence_max_length: Maximum length for sequence patterns
            anomaly_score_threshold: Score threshold for reporting anomalies
        """
        self.temporal_threshold = temporal_threshold
        self.sequence_min_support = sequence_min_support
        self.sequence_min_length = sequence_min_length
        self.sequence_max_length = sequence_max_length
        self.anomaly_score_threshold = anomaly_score_threshold
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the detector."""
        if self._initialized:
            return
        self._initialized = True
        logger.info("PatternDetector initialized")

    def detect_patterns(self, event_log: EventLog) -> PatternDetectionResult:
        """
        Detect patterns and anomalies in event log.

        Args:
            event_log: Event log to analyze

        Returns:
            PatternDetectionResult with detected patterns and anomalies
        """
        if not self._initialized:
            self.initialize()

        start_time = datetime.now()

        # Detect temporal patterns
        temporal_patterns = self._detect_temporal_patterns(event_log)

        # Detect sequence patterns
        sequence_patterns = self._detect_sequence_patterns(event_log)

        # Detect temporal anomalies
        temporal_anomalies = self._detect_temporal_anomalies(event_log, temporal_patterns)

        # Detect sequence anomalies
        sequence_anomalies = self._detect_sequence_anomalies(event_log, sequence_patterns)

        # Detect frequency anomalies
        frequency_anomalies = self._detect_frequency_anomalies(event_log)

        # Detect loop patterns
        loop_patterns = self._detect_loop_patterns(event_log)

        # Combine all anomalies
        all_anomalies = temporal_anomalies + sequence_anomalies + frequency_anomalies

        # Filter by threshold
        all_anomalies = [a for a in all_anomalies if a.score >= self.anomaly_score_threshold]

        # Sort by severity and score
        severity_order = {
            AnomalySeverity.CRITICAL: 0,
            AnomalySeverity.HIGH: 1,
            AnomalySeverity.MEDIUM: 2,
            AnomalySeverity.LOW: 3,
        }
        all_anomalies.sort(key=lambda a: (severity_order[a.severity], -a.score))

        # Generate visualization data
        visualization_data = self._generate_visualization_data(
            event_log, temporal_patterns, sequence_patterns, all_anomalies
        )

        # Calculate statistics
        statistics = self._calculate_detection_statistics(
            event_log, temporal_patterns, sequence_patterns, all_anomalies
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return PatternDetectionResult(
            patterns=[p.to_dict() for p in loop_patterns],
            anomalies=[a.to_dict() for a in all_anomalies],
            temporal_patterns=[self._temporal_pattern_to_dict(tp) for tp in temporal_patterns],
            sequence_patterns=[p.to_dict() for p in sequence_patterns],
            statistics=statistics,
            visualization_data=visualization_data,
            processing_time_ms=processing_time,
        )

    def _temporal_pattern_to_dict(self, tp: TemporalPattern) -> Dict[str, Any]:
        """Convert temporal pattern to dictionary."""
        return {
            "source_activity": tp.source_activity,
            "target_activity": tp.target_activity,
            "avg_interval_seconds": tp.avg_interval_seconds,
            "std_interval_seconds": tp.std_interval_seconds,
            "min_interval_seconds": tp.min_interval_seconds,
            "max_interval_seconds": tp.max_interval_seconds,
            "occurrence_count": tp.occurrence_count,
        }

    def _detect_temporal_patterns(self, event_log: EventLog) -> List[TemporalPattern]:
        """
        Detect temporal patterns (time intervals between activities).

        Args:
            event_log: Event log to analyze

        Returns:
            List of temporal patterns
        """
        # Collect intervals between activity pairs
        intervals: Dict[Tuple[str, str], List[float]] = defaultdict(list)

        for trace in event_log.traces:
            sorted_events = sorted(trace.events, key=lambda e: e.timestamp)

            for i in range(len(sorted_events) - 1):
                event_a = sorted_events[i]
                event_b = sorted_events[i + 1]

                interval = (event_b.timestamp - event_a.timestamp).total_seconds()
                intervals[(event_a.activity, event_b.activity)].append(interval)

        # Build temporal patterns
        patterns = []
        for (source, target), interval_list in intervals.items():
            if len(interval_list) < 2:
                continue

            avg = sum(interval_list) / len(interval_list)
            variance = sum((x - avg) ** 2 for x in interval_list) / len(interval_list)
            std = math.sqrt(variance)

            patterns.append(TemporalPattern(
                source_activity=source,
                target_activity=target,
                avg_interval_seconds=avg,
                std_interval_seconds=std,
                min_interval_seconds=min(interval_list),
                max_interval_seconds=max(interval_list),
                occurrence_count=len(interval_list),
            ))

        return patterns

    def _detect_sequence_patterns(self, event_log: EventLog) -> List[Pattern]:
        """
        Detect frequent sequence patterns (subsequences of activities).

        Uses a simplified version of sequential pattern mining.

        Args:
            event_log: Event log to analyze

        Returns:
            List of sequence patterns
        """
        total_traces = len(event_log.traces)
        if total_traces == 0:
            return []

        patterns = []

        # Generate candidate sequences of different lengths
        for length in range(self.sequence_min_length, self.sequence_max_length + 1):
            sequence_counts: Dict[Tuple[str, ...], int] = defaultdict(int)

            for trace in event_log.traces:
                activities = trace.activities
                # Generate all subsequences of this length
                for i in range(len(activities) - length + 1):
                    subsequence = tuple(activities[i:i + length])
                    sequence_counts[subsequence] += 1

            # Filter by minimum support
            for sequence, count in sequence_counts.items():
                support = count / total_traces
                if support >= self.sequence_min_support:
                    patterns.append(Pattern(
                        pattern_type="sequence",
                        activities=list(sequence),
                        frequency=count,
                        support=support,
                        confidence=support,  # Simplified confidence
                        metadata={"length": length},
                    ))

        # Sort by support descending
        patterns.sort(key=lambda p: -p.support)

        return patterns

    def _detect_temporal_anomalies(
        self,
        event_log: EventLog,
        temporal_patterns: List[TemporalPattern],
    ) -> List[Anomaly]:
        """
        Detect temporal anomalies based on unusual time intervals.

        Args:
            event_log: Event log to analyze
            temporal_patterns: Known temporal patterns

        Returns:
            List of temporal anomalies
        """
        anomalies = []

        # Build pattern lookup
        pattern_lookup: Dict[Tuple[str, str], TemporalPattern] = {
            (p.source_activity, p.target_activity): p for p in temporal_patterns
        }

        for trace in event_log.traces:
            sorted_events = sorted(trace.events, key=lambda e: e.timestamp)

            for i in range(len(sorted_events) - 1):
                event_a = sorted_events[i]
                event_b = sorted_events[i + 1]

                pattern = pattern_lookup.get((event_a.activity, event_b.activity))
                if not pattern or pattern.std_interval_seconds == 0:
                    continue

                interval = (event_b.timestamp - event_a.timestamp).total_seconds()

                if pattern.is_anomalous(interval, self.temporal_threshold):
                    z_score = abs(interval - pattern.avg_interval_seconds) / pattern.std_interval_seconds

                    # Determine severity based on z-score
                    if z_score > 4:
                        severity = AnomalySeverity.CRITICAL
                    elif z_score > 3:
                        severity = AnomalySeverity.HIGH
                    elif z_score > 2.5:
                        severity = AnomalySeverity.MEDIUM
                    else:
                        severity = AnomalySeverity.LOW

                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.TEMPORAL,
                        severity=severity,
                        description=f"Unusual time interval ({interval:.1f}s) between "
                                  f"'{event_a.activity}' and '{event_b.activity}' "
                                  f"(expected: {pattern.avg_interval_seconds:.1f}s ± {pattern.std_interval_seconds:.1f}s)",
                        case_id=trace.case_id,
                        event_ids=[event_a.event_id, event_b.event_id],
                        score=min(1.0, z_score / 5),  # Normalize to 0-1
                        context={
                            "actual_interval": interval,
                            "expected_interval": pattern.avg_interval_seconds,
                            "std_interval": pattern.std_interval_seconds,
                            "z_score": z_score,
                            "source_activity": event_a.activity,
                            "target_activity": event_b.activity,
                        },
                    ))

        return anomalies

    def _detect_sequence_anomalies(
        self,
        event_log: EventLog,
        sequence_patterns: List[Pattern],
    ) -> List[Anomaly]:
        """
        Detect sequence anomalies (unexpected activity orders).

        Args:
            event_log: Event log to analyze
            sequence_patterns: Known sequence patterns

        Returns:
            List of sequence anomalies
        """
        anomalies = []

        # Build set of expected 2-grams from patterns
        expected_bigrams: Set[Tuple[str, str]] = set()
        for pattern in sequence_patterns:
            activities = pattern.activities
            for i in range(len(activities) - 1):
                expected_bigrams.add((activities[i], activities[i + 1]))

        if not expected_bigrams:
            return anomalies

        # Check each trace for unexpected sequences
        for trace in event_log.traces:
            activities = trace.activities
            unexpected_sequences = []

            for i in range(len(activities) - 1):
                bigram = (activities[i], activities[i + 1])
                if bigram not in expected_bigrams:
                    unexpected_sequences.append(bigram)

            if unexpected_sequences:
                # Calculate anomaly score based on proportion of unexpected sequences
                score = len(unexpected_sequences) / max(len(activities) - 1, 1)

                if score > 0.5:
                    severity = AnomalySeverity.HIGH
                elif score > 0.3:
                    severity = AnomalySeverity.MEDIUM
                else:
                    severity = AnomalySeverity.LOW

                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.SEQUENCE,
                    severity=severity,
                    description=f"Unexpected activity sequences in case {trace.case_id}: "
                              f"{', '.join([f'{a}->{b}' for a, b in unexpected_sequences[:3]])}",
                    case_id=trace.case_id,
                    score=score,
                    context={
                        "unexpected_sequences": [list(seq) for seq in unexpected_sequences],
                        "trace_activities": activities,
                    },
                ))

        return anomalies

    def _detect_frequency_anomalies(self, event_log: EventLog) -> List[Anomaly]:
        """
        Detect frequency anomalies (unusual activity counts in traces).

        Args:
            event_log: Event log to analyze

        Returns:
            List of frequency anomalies
        """
        anomalies = []

        # Calculate activity frequency statistics per trace
        activity_counts: Dict[str, List[int]] = defaultdict(list)

        for trace in event_log.traces:
            trace_counts: Dict[str, int] = defaultdict(int)
            for event in trace.events:
                trace_counts[event.activity] += 1

            for activity, count in trace_counts.items():
                activity_counts[activity].append(count)

        # Calculate mean and std for each activity
        activity_stats: Dict[str, Tuple[float, float]] = {}
        for activity, counts in activity_counts.items():
            if len(counts) < 2:
                continue
            avg = sum(counts) / len(counts)
            variance = sum((c - avg) ** 2 for c in counts) / len(counts)
            std = math.sqrt(variance)
            activity_stats[activity] = (avg, std)

        # Check each trace for frequency anomalies
        for trace in event_log.traces:
            trace_counts: Dict[str, int] = defaultdict(int)
            for event in trace.events:
                trace_counts[event.activity] += 1

            for activity, count in trace_counts.items():
                if activity not in activity_stats:
                    continue

                avg, std = activity_stats[activity]
                if std == 0:
                    continue

                z_score = abs(count - avg) / std

                if z_score > self.temporal_threshold:
                    if z_score > 4:
                        severity = AnomalySeverity.HIGH
                    elif z_score > 3:
                        severity = AnomalySeverity.MEDIUM
                    else:
                        severity = AnomalySeverity.LOW

                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.FREQUENCY,
                        severity=severity,
                        description=f"Unusual frequency of '{activity}' in case {trace.case_id}: "
                                  f"{count} times (expected: {avg:.1f} ± {std:.1f})",
                        case_id=trace.case_id,
                        score=min(1.0, z_score / 5),
                        context={
                            "activity": activity,
                            "actual_count": count,
                            "expected_count": avg,
                            "std_count": std,
                            "z_score": z_score,
                        },
                    ))

        return anomalies

    def _detect_loop_patterns(self, event_log: EventLog) -> List[Pattern]:
        """
        Detect loop patterns (repeated sequences of activities).

        Args:
            event_log: Event log to analyze

        Returns:
            List of loop patterns
        """
        patterns = []
        total_traces = len(event_log.traces)

        if total_traces == 0:
            return patterns

        # Count self-loops and small loops
        self_loops: Dict[str, int] = defaultdict(int)
        two_loops: Dict[Tuple[str, str], int] = defaultdict(int)

        for trace in event_log.traces:
            activities = trace.activities

            # Detect self-loops (a -> a)
            for i in range(len(activities) - 1):
                if activities[i] == activities[i + 1]:
                    self_loops[activities[i]] += 1

            # Detect 2-loops (a -> b -> a)
            for i in range(len(activities) - 2):
                if activities[i] == activities[i + 2] and activities[i] != activities[i + 1]:
                    two_loops[(activities[i], activities[i + 1])] += 1

        # Create patterns for significant loops
        for activity, count in self_loops.items():
            support = count / total_traces
            if support >= self.sequence_min_support:
                patterns.append(Pattern(
                    pattern_type="self_loop",
                    activities=[activity],
                    frequency=count,
                    support=support,
                    confidence=support,
                    metadata={"loop_type": "self_loop"},
                ))

        for (a, b), count in two_loops.items():
            support = count / total_traces
            if support >= self.sequence_min_support:
                patterns.append(Pattern(
                    pattern_type="two_loop",
                    activities=[a, b, a],
                    frequency=count,
                    support=support,
                    confidence=support,
                    metadata={"loop_type": "two_loop"},
                ))

        return patterns

    def _generate_visualization_data(
        self,
        event_log: EventLog,
        temporal_patterns: List[TemporalPattern],
        sequence_patterns: List[Pattern],
        anomalies: List[Anomaly],
    ) -> Dict[str, Any]:
        """
        Generate data for visualization components.

        Returns data suitable for:
        - Process flow diagrams
        - Timeline charts
        - Anomaly heatmaps
        """
        # Activity nodes
        activities = list(event_log.all_activities)
        activity_freq = event_log.get_activity_frequency()

        nodes = [
            {
                "id": activity,
                "label": activity,
                "frequency": activity_freq.get(activity, 0),
                "type": "activity",
            }
            for activity in activities
        ]

        # Edges from temporal patterns
        edges = [
            {
                "source": tp.source_activity,
                "target": tp.target_activity,
                "weight": tp.occurrence_count,
                "avg_time": tp.avg_interval_seconds,
                "type": "transition",
            }
            for tp in temporal_patterns
        ]

        # Timeline data
        timeline = []
        for trace in event_log.traces[:100]:  # Limit for performance
            for event in trace.events:
                timeline.append({
                    "case_id": trace.case_id,
                    "activity": event.activity,
                    "timestamp": event.timestamp.isoformat(),
                    "resource": event.resource,
                })

        # Anomaly heatmap data
        anomaly_by_activity: Dict[str, int] = defaultdict(int)
        anomaly_by_type: Dict[str, int] = defaultdict(int)

        for anomaly in anomalies:
            anomaly_by_type[anomaly.anomaly_type.value] += 1
            if "activity" in anomaly.context:
                anomaly_by_activity[anomaly.context["activity"]] += 1
            elif "source_activity" in anomaly.context:
                anomaly_by_activity[anomaly.context["source_activity"]] += 1

        return {
            "nodes": nodes,
            "edges": edges,
            "timeline": timeline,
            "anomaly_heatmap": {
                "by_activity": dict(anomaly_by_activity),
                "by_type": dict(anomaly_by_type),
            },
            "pattern_summary": {
                "total_sequence_patterns": len(sequence_patterns),
                "total_temporal_patterns": len(temporal_patterns),
                "top_patterns": [p.to_dict() for p in sequence_patterns[:5]],
            },
        }

    def _calculate_detection_statistics(
        self,
        event_log: EventLog,
        temporal_patterns: List[TemporalPattern],
        sequence_patterns: List[Pattern],
        anomalies: List[Anomaly],
    ) -> Dict[str, Any]:
        """Calculate statistics about detection results."""
        # Anomaly distribution
        anomaly_distribution = defaultdict(int)
        severity_distribution = defaultdict(int)

        for anomaly in anomalies:
            anomaly_distribution[anomaly.anomaly_type.value] += 1
            severity_distribution[anomaly.severity.value] += 1

        # Coverage statistics
        traces_with_anomalies = len(set(a.case_id for a in anomalies if a.case_id))

        return {
            "total_temporal_patterns": len(temporal_patterns),
            "total_sequence_patterns": len(sequence_patterns),
            "total_anomalies": len(anomalies),
            "anomaly_distribution": dict(anomaly_distribution),
            "severity_distribution": dict(severity_distribution),
            "traces_analyzed": len(event_log.traces),
            "traces_with_anomalies": traces_with_anomalies,
            "anomaly_rate": traces_with_anomalies / len(event_log.traces) if event_log.traces else 0,
            "avg_anomaly_score": sum(a.score for a in anomalies) / len(anomalies) if anomalies else 0,
        }

    def detect_temporal_anomalies(self, event_log: EventLog) -> List[Anomaly]:
        """
        Public method to detect only temporal anomalies.

        Args:
            event_log: Event log to analyze

        Returns:
            List of temporal anomalies
        """
        if not self._initialized:
            self.initialize()

        temporal_patterns = self._detect_temporal_patterns(event_log)
        return self._detect_temporal_anomalies(event_log, temporal_patterns)

    def detect_sequence_anomalies(self, event_log: EventLog) -> List[Anomaly]:
        """
        Public method to detect only sequence anomalies.

        Args:
            event_log: Event log to analyze

        Returns:
            List of sequence anomalies
        """
        if not self._initialized:
            self.initialize()

        sequence_patterns = self._detect_sequence_patterns(event_log)
        return self._detect_sequence_anomalies(event_log, sequence_patterns)


# Global instance
_pattern_detector: Optional[PatternDetector] = None


def get_pattern_detector() -> PatternDetector:
    """Get or create global PatternDetector instance."""
    global _pattern_detector

    if _pattern_detector is None:
        _pattern_detector = PatternDetector()
        _pattern_detector.initialize()

    return _pattern_detector
