"""
Process Miner for Knowledge Graph.

Provides process mining capabilities including:
- Event log construction from annotation data
- Process discovery using Alpha algorithm
- Conformance checking and anomaly detection
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ActivityType(str, Enum):
    """Types of activities in process mining."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    ANNOTATE = "annotate"
    REVIEW = "review"
    APPROVE = "approve"
    REJECT = "reject"
    ASSIGN = "assign"
    COMPLETE = "complete"
    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    CUSTOM = "custom"


class RelationshipType(str, Enum):
    """Process relationship types for Petri nets."""
    DIRECT_SUCCESSION = "direct_succession"  # a > b: a is directly followed by b
    CAUSALITY = "causality"  # a -> b: a causes b
    PARALLEL = "parallel"  # a || b: a and b can happen in parallel
    CHOICE = "choice"  # a # b: a and b are in exclusive choice


@dataclass
class Event:
    """Single event in an event log."""

    event_id: str = field(default_factory=lambda: str(uuid4()))
    case_id: str = ""  # Process instance ID
    activity: str = ""  # Activity name
    activity_type: ActivityType = ActivityType.CUSTOM
    timestamp: datetime = field(default_factory=datetime.now)
    resource: Optional[str] = None  # User/resource performing the activity
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.case_id:
            self.case_id = str(uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "case_id": self.case_id,
            "activity": self.activity,
            "activity_type": self.activity_type.value,
            "timestamp": self.timestamp.isoformat(),
            "resource": self.resource,
            "attributes": self.attributes,
        }


@dataclass
class Trace:
    """Sequence of events for a single process instance."""

    case_id: str = ""
    events: List[Event] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.case_id and self.events:
            self.case_id = self.events[0].case_id

    @property
    def activities(self) -> List[str]:
        """Get ordered list of activity names."""
        return [e.activity for e in sorted(self.events, key=lambda x: x.timestamp)]

    @property
    def duration(self) -> Optional[float]:
        """Get trace duration in seconds."""
        if len(self.events) < 2:
            return None
        sorted_events = sorted(self.events, key=lambda x: x.timestamp)
        return (sorted_events[-1].timestamp - sorted_events[0].timestamp).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "case_id": self.case_id,
            "events": [e.to_dict() for e in self.events],
            "activities": self.activities,
            "duration": self.duration,
            "attributes": self.attributes,
        }


@dataclass
class EventLog:
    """Collection of traces forming an event log."""

    name: str = "event_log"
    traces: List[Trace] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_activities(self) -> Set[str]:
        """Get all unique activities in the log."""
        activities = set()
        for trace in self.traces:
            activities.update(trace.activities)
        return activities

    @property
    def total_events(self) -> int:
        """Get total number of events."""
        return sum(len(t.events) for t in self.traces)

    def get_activity_frequency(self) -> Dict[str, int]:
        """Get frequency of each activity."""
        freq = defaultdict(int)
        for trace in self.traces:
            for activity in trace.activities:
                freq[activity] += 1
        return dict(freq)

    def get_trace_variants(self) -> Dict[str, int]:
        """Get frequency of trace variants (unique activity sequences)."""
        variants = defaultdict(int)
        for trace in self.traces:
            variant = tuple(trace.activities)
            variants[variant] += 1
        return {",".join(k): v for k, v in variants.items()}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "traces": [t.to_dict() for t in self.traces],
            "total_traces": len(self.traces),
            "total_events": self.total_events,
            "unique_activities": list(self.all_activities),
            "attributes": self.attributes,
        }


@dataclass
class Place:
    """Place in a Petri net."""

    place_id: str = field(default_factory=lambda: f"p_{uuid4().hex[:8]}")
    name: str = ""
    tokens: int = 0

    def __hash__(self):
        return hash(self.place_id)

    def __eq__(self, other):
        if isinstance(other, Place):
            return self.place_id == other.place_id
        return False


@dataclass
class Transition:
    """Transition in a Petri net (represents an activity)."""

    transition_id: str = field(default_factory=lambda: f"t_{uuid4().hex[:8]}")
    name: str = ""  # Activity name
    is_silent: bool = False  # Silent/tau transition

    def __hash__(self):
        return hash(self.transition_id)

    def __eq__(self, other):
        if isinstance(other, Transition):
            return self.transition_id == other.transition_id
        return False


@dataclass
class Arc:
    """Arc connecting places and transitions."""

    source: str = ""  # Place or Transition ID
    target: str = ""  # Place or Transition ID
    weight: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
        }


@dataclass
class PetriNet:
    """Petri net model discovered from event log."""

    name: str = "process_model"
    places: List[Place] = field(default_factory=list)
    transitions: List[Transition] = field(default_factory=list)
    arcs: List[Arc] = field(default_factory=list)
    initial_marking: Dict[str, int] = field(default_factory=dict)
    final_marking: Dict[str, int] = field(default_factory=dict)

    def get_input_places(self, transition: Transition) -> List[Place]:
        """Get input places for a transition."""
        input_place_ids = {arc.source for arc in self.arcs if arc.target == transition.transition_id}
        return [p for p in self.places if p.place_id in input_place_ids]

    def get_output_places(self, transition: Transition) -> List[Place]:
        """Get output places for a transition."""
        output_place_ids = {arc.target for arc in self.arcs if arc.source == transition.transition_id}
        return [p for p in self.places if p.place_id in output_place_ids]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "places": [{"id": p.place_id, "name": p.name, "tokens": p.tokens} for p in self.places],
            "transitions": [{"id": t.transition_id, "name": t.name, "is_silent": t.is_silent} for t in self.transitions],
            "arcs": [a.to_dict() for a in self.arcs],
            "initial_marking": self.initial_marking,
            "final_marking": self.final_marking,
        }


class ProcessMiningResult(BaseModel):
    """Result of process mining analysis."""

    petri_net: Dict[str, Any] = Field(default_factory=dict, description="Discovered Petri net")
    footprint_matrix: Dict[str, Dict[str, str]] = Field(default_factory=dict, description="Footprint matrix")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Process statistics")
    anomalies: List[Dict[str, Any]] = Field(default_factory=list, description="Detected anomalies")
    conformance_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Conformance score")
    processing_time_ms: float = Field(default=0.0, description="Processing time in milliseconds")


class ProcessMiner:
    """
    Process miner using Alpha algorithm and anomaly detection.

    Discovers process models from event logs and identifies anomalies.
    """

    def __init__(
        self,
        min_trace_frequency: int = 1,
        max_loop_depth: int = 3,
        anomaly_threshold: float = 0.1,
    ):
        """
        Initialize ProcessMiner.

        Args:
            min_trace_frequency: Minimum frequency for trace variants to consider
            max_loop_depth: Maximum depth for loop detection
            anomaly_threshold: Threshold for anomaly detection (0-1)
        """
        self.min_trace_frequency = min_trace_frequency
        self.max_loop_depth = max_loop_depth
        self.anomaly_threshold = anomaly_threshold
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the miner."""
        if self._initialized:
            return
        self._initialized = True
        logger.info("ProcessMiner initialized")

    def build_event_log(
        self,
        annotations: List[Dict[str, Any]],
        case_id_field: str = "document_id",
        activity_field: str = "action",
        timestamp_field: str = "timestamp",
        resource_field: str = "user_id",
    ) -> EventLog:
        """
        Build event log from annotation data.

        Args:
            annotations: List of annotation records
            case_id_field: Field name for case ID
            activity_field: Field name for activity
            timestamp_field: Field name for timestamp
            resource_field: Field name for resource

        Returns:
            Constructed EventLog
        """
        if not self._initialized:
            self.initialize()

        # Group events by case ID
        cases: Dict[str, List[Event]] = defaultdict(list)

        for annotation in annotations:
            case_id = str(annotation.get(case_id_field, uuid4()))
            activity = str(annotation.get(activity_field, "unknown"))

            # Parse timestamp
            ts_value = annotation.get(timestamp_field)
            if isinstance(ts_value, datetime):
                timestamp = ts_value
            elif isinstance(ts_value, str):
                try:
                    timestamp = datetime.fromisoformat(ts_value.replace("Z", "+00:00"))
                except ValueError:
                    timestamp = datetime.now()
            else:
                timestamp = datetime.now()

            # Infer activity type
            activity_type = self._infer_activity_type(activity)

            event = Event(
                case_id=case_id,
                activity=activity,
                activity_type=activity_type,
                timestamp=timestamp,
                resource=annotation.get(resource_field),
                attributes={k: v for k, v in annotation.items()
                          if k not in {case_id_field, activity_field, timestamp_field, resource_field}},
            )
            cases[case_id].append(event)

        # Create traces
        traces = []
        for case_id, events in cases.items():
            sorted_events = sorted(events, key=lambda e: e.timestamp)
            trace = Trace(case_id=case_id, events=sorted_events)
            traces.append(trace)

        event_log = EventLog(name="annotation_log", traces=traces)
        logger.info(f"Built event log with {len(traces)} traces and {event_log.total_events} events")

        return event_log

    def _infer_activity_type(self, activity: str) -> ActivityType:
        """Infer activity type from activity name."""
        activity_lower = activity.lower()

        type_mapping = {
            ActivityType.CREATE: ["create", "创建", "新建", "add", "添加"],
            ActivityType.UPDATE: ["update", "更新", "修改", "edit", "编辑"],
            ActivityType.DELETE: ["delete", "删除", "remove", "移除"],
            ActivityType.ANNOTATE: ["annotate", "标注", "label", "标签"],
            ActivityType.REVIEW: ["review", "审核", "check", "检查"],
            ActivityType.APPROVE: ["approve", "批准", "通过", "accept", "接受"],
            ActivityType.REJECT: ["reject", "拒绝", "驳回", "deny"],
            ActivityType.ASSIGN: ["assign", "分配", "指派"],
            ActivityType.COMPLETE: ["complete", "完成", "finish", "结束"],
            ActivityType.START: ["start", "开始", "begin", "启动"],
            ActivityType.PAUSE: ["pause", "暂停", "suspend"],
            ActivityType.RESUME: ["resume", "恢复", "continue"],
        }

        for activity_type, keywords in type_mapping.items():
            if any(kw in activity_lower for kw in keywords):
                return activity_type

        return ActivityType.CUSTOM

    def discover_process(self, event_log: EventLog) -> ProcessMiningResult:
        """
        Discover process model using Alpha algorithm.

        Args:
            event_log: Event log to analyze

        Returns:
            ProcessMiningResult with discovered model and statistics
        """
        if not self._initialized:
            self.initialize()

        start_time = datetime.now()

        # Step 1: Build footprint matrix
        footprint = self._build_footprint_matrix(event_log)

        # Step 2: Find start and end activities
        start_activities = self._find_start_activities(event_log)
        end_activities = self._find_end_activities(event_log)

        # Step 3: Discover places using Alpha algorithm
        petri_net = self._alpha_algorithm(event_log, footprint, start_activities, end_activities)

        # Step 4: Calculate statistics
        statistics = self._calculate_statistics(event_log)

        # Step 5: Detect anomalies
        anomalies = self._detect_anomalies(event_log, petri_net)

        # Step 6: Calculate conformance
        conformance_score = self._calculate_conformance(event_log, petri_net)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return ProcessMiningResult(
            petri_net=petri_net.to_dict(),
            footprint_matrix=footprint,
            statistics=statistics,
            anomalies=anomalies,
            conformance_score=conformance_score,
            processing_time_ms=processing_time,
        )

    def _build_footprint_matrix(self, event_log: EventLog) -> Dict[str, Dict[str, str]]:
        """
        Build footprint matrix showing relationships between activities.

        Relations:
        - '>': Direct succession (a directly followed by b)
        - '->': Causality (a causes b)
        - '<-': Reverse causality
        - '||': Parallel
        - '#': No relation / exclusive choice
        """
        activities = list(event_log.all_activities)

        # Count direct successions
        direct_successions: Dict[Tuple[str, str], int] = defaultdict(int)

        for trace in event_log.traces:
            trace_activities = trace.activities
            for i in range(len(trace_activities) - 1):
                a, b = trace_activities[i], trace_activities[i + 1]
                direct_successions[(a, b)] += 1

        # Build footprint matrix
        footprint: Dict[str, Dict[str, str]] = {}

        for a in activities:
            footprint[a] = {}
            for b in activities:
                if a == b:
                    # Self-loop check
                    if direct_successions.get((a, a), 0) > 0:
                        footprint[a][b] = "||"  # Activity can repeat
                    else:
                        footprint[a][b] = "#"
                else:
                    a_to_b = direct_successions.get((a, b), 0) > 0
                    b_to_a = direct_successions.get((b, a), 0) > 0

                    if a_to_b and b_to_a:
                        footprint[a][b] = "||"  # Parallel
                    elif a_to_b and not b_to_a:
                        footprint[a][b] = "->"  # Causality
                    elif not a_to_b and b_to_a:
                        footprint[a][b] = "<-"  # Reverse causality
                    else:
                        footprint[a][b] = "#"  # No relation

        return footprint

    def _find_start_activities(self, event_log: EventLog) -> Set[str]:
        """Find activities that can start a trace."""
        start_activities = set()
        for trace in event_log.traces:
            if trace.activities:
                start_activities.add(trace.activities[0])
        return start_activities

    def _find_end_activities(self, event_log: EventLog) -> Set[str]:
        """Find activities that can end a trace."""
        end_activities = set()
        for trace in event_log.traces:
            if trace.activities:
                end_activities.add(trace.activities[-1])
        return end_activities

    def _alpha_algorithm(
        self,
        event_log: EventLog,
        footprint: Dict[str, Dict[str, str]],
        start_activities: Set[str],
        end_activities: Set[str],
    ) -> PetriNet:
        """
        Implement Alpha algorithm for process discovery.

        Args:
            event_log: Event log
            footprint: Footprint matrix
            start_activities: Start activities
            end_activities: End activities

        Returns:
            Discovered PetriNet
        """
        activities = list(event_log.all_activities)

        # Create places and transitions
        places = []
        transitions = []
        arcs = []

        # Create start place
        start_place = Place(place_id="p_start", name="start", tokens=1)
        places.append(start_place)

        # Create end place
        end_place = Place(place_id="p_end", name="end", tokens=0)
        places.append(end_place)

        # Create transitions for each activity
        activity_transitions = {}
        for activity in activities:
            transition = Transition(
                transition_id=f"t_{activity}",
                name=activity,
                is_silent=False,
            )
            transitions.append(transition)
            activity_transitions[activity] = transition

        # Connect start place to start activities
        for activity in start_activities:
            arcs.append(Arc(source=start_place.place_id, target=activity_transitions[activity].transition_id))

        # Connect end activities to end place
        for activity in end_activities:
            arcs.append(Arc(source=activity_transitions[activity].transition_id, target=end_place.place_id))

        # Find causality pairs and create intermediate places
        place_counter = 0
        for a in activities:
            for b in activities:
                if a != b and footprint.get(a, {}).get(b) == "->":
                    # Create intermediate place
                    place_counter += 1
                    intermediate_place = Place(
                        place_id=f"p_{place_counter}",
                        name=f"{a}_to_{b}",
                        tokens=0,
                    )
                    places.append(intermediate_place)

                    # Connect a -> place -> b
                    arcs.append(Arc(source=activity_transitions[a].transition_id, target=intermediate_place.place_id))
                    arcs.append(Arc(source=intermediate_place.place_id, target=activity_transitions[b].transition_id))

        # Set initial and final markings
        initial_marking = {start_place.place_id: 1}
        final_marking = {end_place.place_id: 1}

        return PetriNet(
            name="discovered_process",
            places=places,
            transitions=transitions,
            arcs=arcs,
            initial_marking=initial_marking,
            final_marking=final_marking,
        )

    def _calculate_statistics(self, event_log: EventLog) -> Dict[str, Any]:
        """Calculate process statistics from event log."""
        # Trace durations
        durations = [t.duration for t in event_log.traces if t.duration is not None]

        # Activity frequencies
        activity_freq = event_log.get_activity_frequency()

        # Trace variants
        variants = event_log.get_trace_variants()

        # Resource analysis
        resources = defaultdict(int)
        for trace in event_log.traces:
            for event in trace.events:
                if event.resource:
                    resources[event.resource] += 1

        return {
            "total_traces": len(event_log.traces),
            "total_events": event_log.total_events,
            "unique_activities": len(event_log.all_activities),
            "activity_frequency": activity_freq,
            "trace_variants": len(variants),
            "variant_distribution": variants,
            "avg_trace_duration_seconds": sum(durations) / len(durations) if durations else 0,
            "min_trace_duration_seconds": min(durations) if durations else 0,
            "max_trace_duration_seconds": max(durations) if durations else 0,
            "avg_trace_length": event_log.total_events / len(event_log.traces) if event_log.traces else 0,
            "resource_workload": dict(resources),
        }

    def _detect_anomalies(self, event_log: EventLog, petri_net: PetriNet) -> List[Dict[str, Any]]:
        """
        Detect anomalies in the event log.

        Detects:
        - Unusual trace durations
        - Rare trace variants
        - Missing activities
        - Unexpected activity sequences
        """
        anomalies = []

        # Get variant frequencies
        variants = event_log.get_trace_variants()
        total_traces = len(event_log.traces)

        if total_traces == 0:
            return anomalies

        # Find rare variants (below threshold)
        for variant, count in variants.items():
            frequency = count / total_traces
            if frequency < self.anomaly_threshold and count < self.min_trace_frequency:
                anomalies.append({
                    "type": "rare_variant",
                    "description": f"Rare trace variant: {variant}",
                    "frequency": frequency,
                    "count": count,
                    "severity": "low",
                })

        # Detect duration anomalies
        durations = [t.duration for t in event_log.traces if t.duration is not None]
        if durations:
            avg_duration = sum(durations) / len(durations)
            std_duration = (sum((d - avg_duration) ** 2 for d in durations) / len(durations)) ** 0.5

            for trace in event_log.traces:
                if trace.duration is not None:
                    z_score = (trace.duration - avg_duration) / std_duration if std_duration > 0 else 0
                    if abs(z_score) > 2:  # More than 2 standard deviations
                        anomalies.append({
                            "type": "duration_anomaly",
                            "description": f"Unusual duration for case {trace.case_id}",
                            "case_id": trace.case_id,
                            "duration": trace.duration,
                            "expected_avg": avg_duration,
                            "z_score": z_score,
                            "severity": "medium" if abs(z_score) < 3 else "high",
                        })

        # Detect missing start/end activities
        expected_transitions = {t.name for t in petri_net.transitions if not t.is_silent}
        for trace in event_log.traces:
            trace_activities = set(trace.activities)
            missing = expected_transitions - trace_activities
            if missing and len(missing) > len(expected_transitions) * 0.5:
                anomalies.append({
                    "type": "incomplete_trace",
                    "description": f"Trace {trace.case_id} missing many activities",
                    "case_id": trace.case_id,
                    "missing_activities": list(missing),
                    "severity": "medium",
                })

        return anomalies

    def _calculate_conformance(self, event_log: EventLog, petri_net: PetriNet) -> float:
        """
        Calculate conformance score between event log and discovered model.

        Uses simplified token replay for conformance checking.
        """
        if not event_log.traces:
            return 1.0

        # Build activity connections from Petri net
        valid_sequences: Set[Tuple[str, str]] = set()

        transition_map = {t.transition_id: t.name for t in petri_net.transitions}

        # Find valid transitions through places
        for place in petri_net.places:
            # Get incoming transitions
            incoming = [arc.source for arc in petri_net.arcs if arc.target == place.place_id]
            # Get outgoing transitions
            outgoing = [arc.target for arc in petri_net.arcs if arc.source == place.place_id]

            for in_t in incoming:
                for out_t in outgoing:
                    if in_t in transition_map and out_t in transition_map:
                        valid_sequences.add((transition_map[in_t], transition_map[out_t]))

        # Check each trace for conformance
        conformant_transitions = 0
        total_transitions = 0

        for trace in event_log.traces:
            activities = trace.activities
            for i in range(len(activities) - 1):
                total_transitions += 1
                if (activities[i], activities[i + 1]) in valid_sequences:
                    conformant_transitions += 1

        return conformant_transitions / total_transitions if total_transitions > 0 else 1.0

    def get_process_statistics(self, event_log: EventLog) -> Dict[str, Any]:
        """Get detailed process statistics."""
        return self._calculate_statistics(event_log)


# Global instance
_process_miner: Optional[ProcessMiner] = None


def get_process_miner() -> ProcessMiner:
    """Get or create global ProcessMiner instance."""
    global _process_miner

    if _process_miner is None:
        _process_miner = ProcessMiner()
        _process_miner.initialize()

    return _process_miner
