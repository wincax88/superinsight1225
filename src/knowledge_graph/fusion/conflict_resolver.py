"""
Conflict resolution module for Knowledge Graph fusion.

Provides conflict detection, resolution strategies, and validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ConflictType(str, Enum):
    """Types of knowledge conflicts."""

    VALUE_CONFLICT = "value_conflict"  # Different values for same fact
    TYPE_CONFLICT = "type_conflict"  # Conflicting entity types
    RELATION_CONFLICT = "relation_conflict"  # Conflicting relations
    CARDINALITY_CONFLICT = "cardinality_conflict"  # Single vs multiple values
    TEMPORAL_CONFLICT = "temporal_conflict"  # Time-based conflicts
    SEMANTIC_CONFLICT = "semantic_conflict"  # Meaning contradictions


class ResolutionStrategy(str, Enum):
    """Strategies for resolving conflicts."""

    PREFER_SOURCE = "prefer_source"  # Prefer specific source
    PREFER_RECENT = "prefer_recent"  # Prefer most recent
    PREFER_MAJORITY = "prefer_majority"  # Majority voting
    PREFER_CONFIDENT = "prefer_confident"  # Highest confidence
    MERGE_VALUES = "merge_values"  # Combine all values
    HUMAN_REVIEW = "human_review"  # Flag for manual review
    KEEP_ALL = "keep_all"  # Keep all conflicting values
    DISCARD_ALL = "discard_all"  # Remove conflicting facts


class ConflictSeverity(str, Enum):
    """Severity levels for conflicts."""

    CRITICAL = "critical"  # Must be resolved
    HIGH = "high"  # Should be resolved soon
    MEDIUM = "medium"  # Should be reviewed
    LOW = "low"  # Minor inconsistency
    INFO = "info"  # Informational only


class ConflictingValue(BaseModel):
    """A value involved in a conflict."""

    value: Any = Field(..., description="The conflicting value")
    source_id: str = Field(..., description="Source of this value")
    confidence: float = Field(default=1.0, description="Confidence in value")
    timestamp: datetime = Field(default_factory=datetime.now)
    evidence: list[str] = Field(default_factory=list, description="Supporting evidence")


class Conflict(BaseModel):
    """A detected conflict."""

    conflict_id: str = Field(..., description="Unique conflict identifier")
    conflict_type: ConflictType = Field(..., description="Type of conflict")
    severity: ConflictSeverity = Field(default=ConflictSeverity.MEDIUM)
    subject: str = Field(..., description="Subject entity")
    predicate: str = Field(default="", description="Predicate/relation")
    conflicting_values: list[ConflictingValue] = Field(default_factory=list)
    description: str = Field(default="", description="Conflict description")
    detected_at: datetime = Field(default_factory=datetime.now)
    resolved: bool = Field(default=False)
    resolution: Optional[str] = Field(default=None)
    resolved_value: Optional[Any] = Field(default=None)
    resolved_at: Optional[datetime] = Field(default=None)
    resolved_by: Optional[str] = Field(default=None)


class ResolutionRule(BaseModel):
    """A rule for automatic conflict resolution."""

    rule_id: str = Field(..., description="Rule identifier")
    name: str = Field(..., description="Rule name")
    conflict_type: ConflictType = Field(..., description="Applicable conflict type")
    strategy: ResolutionStrategy = Field(..., description="Resolution strategy")
    priority: int = Field(default=0, description="Rule priority")
    conditions: dict[str, Any] = Field(
        default_factory=dict, description="Conditions for rule application"
    )
    preferred_source: Optional[str] = Field(default=None)
    enabled: bool = Field(default=True)


class ResolutionResult(BaseModel):
    """Result of conflict resolution."""

    conflict_id: str = Field(..., description="Conflict identifier")
    resolved: bool = Field(default=False)
    strategy_used: ResolutionStrategy = Field(default=ResolutionStrategy.HUMAN_REVIEW)
    resolved_value: Optional[Any] = Field(default=None)
    confidence: float = Field(default=0.0)
    explanation: str = Field(default="")
    rule_applied: Optional[str] = Field(default=None)
    requires_review: bool = Field(default=False)
    timestamp: datetime = Field(default_factory=datetime.now)


class ValidationResult(BaseModel):
    """Result of alignment/resolution validation."""

    validation_id: str = Field(..., description="Validation identifier")
    is_valid: bool = Field(default=False)
    checks_passed: list[str] = Field(default_factory=list)
    checks_failed: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    score: float = Field(default=0.0, description="Validation score (0-1)")
    details: dict[str, Any] = Field(default_factory=dict)


class ConflictStats(BaseModel):
    """Statistics about conflicts."""

    total_conflicts: int = Field(default=0)
    resolved_conflicts: int = Field(default=0)
    pending_conflicts: int = Field(default=0)
    by_type: dict[str, int] = Field(default_factory=dict)
    by_severity: dict[str, int] = Field(default_factory=dict)
    by_source: dict[str, int] = Field(default_factory=dict)
    resolution_rate: float = Field(default=0.0)
    avg_resolution_time: float = Field(default=0.0, description="Average time in hours")


@dataclass
class ConflictResolver:
    """Conflict detection and resolution engine."""

    conflicts: dict[str, Conflict] = field(default_factory=dict)
    rules: list[ResolutionRule] = field(default_factory=list)
    source_priorities: dict[str, int] = field(default_factory=dict)
    custom_resolvers: dict[ConflictType, Callable] = field(default_factory=dict)
    history: list[ResolutionResult] = field(default_factory=list)

    def __post_init__(self):
        # Add default rules
        self._add_default_rules()

    def _add_default_rules(self) -> None:
        """Add default resolution rules."""
        self.rules = [
            ResolutionRule(
                rule_id="rule_confidence",
                name="Prefer High Confidence",
                conflict_type=ConflictType.VALUE_CONFLICT,
                strategy=ResolutionStrategy.PREFER_CONFIDENT,
                priority=1,
                conditions={"min_confidence_diff": 0.2},
            ),
            ResolutionRule(
                rule_id="rule_majority",
                name="Majority Voting",
                conflict_type=ConflictType.VALUE_CONFLICT,
                strategy=ResolutionStrategy.PREFER_MAJORITY,
                priority=2,
                conditions={"min_sources": 3},
            ),
            ResolutionRule(
                rule_id="rule_recent",
                name="Prefer Recent for Temporal",
                conflict_type=ConflictType.TEMPORAL_CONFLICT,
                strategy=ResolutionStrategy.PREFER_RECENT,
                priority=1,
            ),
            ResolutionRule(
                rule_id="rule_type_merge",
                name="Merge Type Conflicts",
                conflict_type=ConflictType.TYPE_CONFLICT,
                strategy=ResolutionStrategy.MERGE_VALUES,
                priority=1,
            ),
        ]

    def detect_conflicts(
        self,
        facts: list[dict[str, Any]],
    ) -> list[Conflict]:
        """Detect conflicts in a set of facts."""
        import uuid

        conflicts = []

        # Group facts by subject-predicate
        fact_groups: dict[str, list[dict[str, Any]]] = {}
        for fact in facts:
            key = f"{fact.get('subject', '')}::{fact.get('predicate', '')}"
            if key not in fact_groups:
                fact_groups[key] = []
            fact_groups[key].append(fact)

        # Check each group for conflicts
        for key, group in fact_groups.items():
            if len(group) < 2:
                continue

            # Collect unique values
            values: dict[Any, list[dict[str, Any]]] = {}
            for fact in group:
                obj = fact.get("object", fact.get("value", ""))
                if obj not in values:
                    values[obj] = []
                values[obj].append(fact)

            if len(values) > 1:
                # Conflict detected
                parts = key.split("::")
                subject = parts[0] if parts else ""
                predicate = parts[1] if len(parts) > 1 else ""

                conflicting = [
                    ConflictingValue(
                        value=val,
                        source_id=facts_list[0].get("source_id", "unknown"),
                        confidence=facts_list[0].get("confidence", 1.0),
                        timestamp=facts_list[0].get(
                            "timestamp", datetime.now()
                        ),
                    )
                    for val, facts_list in values.items()
                ]

                conflict = Conflict(
                    conflict_id=f"conflict_{uuid.uuid4().hex[:8]}",
                    conflict_type=self._determine_conflict_type(group),
                    severity=self._determine_severity(conflicting),
                    subject=subject,
                    predicate=predicate,
                    conflicting_values=conflicting,
                    description=self._generate_description(subject, predicate, conflicting),
                )

                conflicts.append(conflict)
                self.conflicts[conflict.conflict_id] = conflict

        return conflicts

    def resolve_conflict(
        self,
        conflict: Conflict,
        strategy: Optional[ResolutionStrategy] = None,
    ) -> ResolutionResult:
        """Resolve a single conflict."""
        if conflict.resolved:
            return ResolutionResult(
                conflict_id=conflict.conflict_id,
                resolved=True,
                resolved_value=conflict.resolved_value,
                explanation="Already resolved",
            )

        # Find applicable rule
        rule = self._find_applicable_rule(conflict)

        if strategy:
            used_strategy = strategy
        elif rule:
            used_strategy = rule.strategy
        else:
            used_strategy = ResolutionStrategy.HUMAN_REVIEW

        # Apply resolution strategy
        result = self._apply_strategy(conflict, used_strategy, rule)

        # Update conflict if resolved
        if result.resolved:
            conflict.resolved = True
            conflict.resolution = used_strategy.value
            conflict.resolved_value = result.resolved_value
            conflict.resolved_at = datetime.now()
            conflict.resolved_by = "auto" if rule else "manual"

        self.history.append(result)
        return result

    def resolve_all(
        self,
        strategy: Optional[ResolutionStrategy] = None,
    ) -> list[ResolutionResult]:
        """Resolve all pending conflicts."""
        results = []
        for conflict in self.conflicts.values():
            if not conflict.resolved:
                result = self.resolve_conflict(conflict, strategy)
                results.append(result)
        return results

    def _determine_conflict_type(
        self,
        facts: list[dict[str, Any]],
    ) -> ConflictType:
        """Determine the type of conflict."""
        predicates = {f.get("predicate", "") for f in facts}

        if any("type" in p.lower() for p in predicates):
            return ConflictType.TYPE_CONFLICT
        elif any("date" in p.lower() or "time" in p.lower() for p in predicates):
            return ConflictType.TEMPORAL_CONFLICT
        else:
            return ConflictType.VALUE_CONFLICT

    def _determine_severity(
        self,
        conflicting: list[ConflictingValue],
    ) -> ConflictSeverity:
        """Determine conflict severity."""
        if len(conflicting) > 3:
            return ConflictSeverity.HIGH

        confidences = [v.confidence for v in conflicting]
        max_conf = max(confidences)
        min_conf = min(confidences)

        if max_conf - min_conf < 0.2:
            # All sources are similarly confident
            return ConflictSeverity.HIGH
        elif max_conf > 0.9:
            # One source is very confident
            return ConflictSeverity.MEDIUM
        else:
            return ConflictSeverity.LOW

    def _generate_description(
        self,
        subject: str,
        predicate: str,
        conflicting: list[ConflictingValue],
    ) -> str:
        """Generate human-readable conflict description."""
        values_str = ", ".join(f"'{v.value}' (from {v.source_id})" for v in conflicting)
        return (
            f"Conflict for {subject}.{predicate}: "
            f"Found {len(conflicting)} different values: {values_str}"
        )

    def _find_applicable_rule(
        self,
        conflict: Conflict,
    ) -> Optional[ResolutionRule]:
        """Find the best applicable rule for a conflict."""
        applicable = [
            r
            for r in self.rules
            if r.enabled and r.conflict_type == conflict.conflict_type
        ]

        if not applicable:
            return None

        # Check conditions
        valid_rules = []
        for rule in applicable:
            if self._check_rule_conditions(rule, conflict):
                valid_rules.append(rule)

        if not valid_rules:
            return None

        # Return highest priority
        return max(valid_rules, key=lambda r: r.priority)

    def _check_rule_conditions(
        self,
        rule: ResolutionRule,
        conflict: Conflict,
    ) -> bool:
        """Check if rule conditions are met."""
        conditions = rule.conditions

        if "min_sources" in conditions:
            if len(conflict.conflicting_values) < conditions["min_sources"]:
                return False

        if "min_confidence_diff" in conditions:
            confs = [v.confidence for v in conflict.conflicting_values]
            if max(confs) - min(confs) < conditions["min_confidence_diff"]:
                return False

        return True

    def _apply_strategy(
        self,
        conflict: Conflict,
        strategy: ResolutionStrategy,
        rule: Optional[ResolutionRule],
    ) -> ResolutionResult:
        """Apply resolution strategy to conflict."""
        values = conflict.conflicting_values

        if strategy == ResolutionStrategy.PREFER_CONFIDENT:
            best = max(values, key=lambda v: v.confidence)
            return ResolutionResult(
                conflict_id=conflict.conflict_id,
                resolved=True,
                strategy_used=strategy,
                resolved_value=best.value,
                confidence=best.confidence,
                explanation=f"Selected value with highest confidence ({best.confidence:.2f})",
                rule_applied=rule.rule_id if rule else None,
            )

        elif strategy == ResolutionStrategy.PREFER_MAJORITY:
            # Count occurrences
            value_counts: dict[Any, int] = {}
            for v in values:
                value_counts[v.value] = value_counts.get(v.value, 0) + 1

            majority_value = max(value_counts.keys(), key=lambda k: value_counts[k])
            majority_count = value_counts[majority_value]

            return ResolutionResult(
                conflict_id=conflict.conflict_id,
                resolved=True,
                strategy_used=strategy,
                resolved_value=majority_value,
                confidence=majority_count / len(values),
                explanation=f"Selected value supported by {majority_count}/{len(values)} sources",
                rule_applied=rule.rule_id if rule else None,
            )

        elif strategy == ResolutionStrategy.PREFER_RECENT:
            most_recent = max(values, key=lambda v: v.timestamp)
            return ResolutionResult(
                conflict_id=conflict.conflict_id,
                resolved=True,
                strategy_used=strategy,
                resolved_value=most_recent.value,
                confidence=most_recent.confidence,
                explanation=f"Selected most recent value ({most_recent.timestamp})",
                rule_applied=rule.rule_id if rule else None,
            )

        elif strategy == ResolutionStrategy.PREFER_SOURCE:
            preferred_source = rule.preferred_source if rule else None
            if preferred_source:
                for v in values:
                    if v.source_id == preferred_source:
                        return ResolutionResult(
                            conflict_id=conflict.conflict_id,
                            resolved=True,
                            strategy_used=strategy,
                            resolved_value=v.value,
                            confidence=v.confidence,
                            explanation=f"Selected value from preferred source: {preferred_source}",
                            rule_applied=rule.rule_id if rule else None,
                        )

            # Fallback to source priority
            if self.source_priorities:
                best = max(
                    values,
                    key=lambda v: self.source_priorities.get(v.source_id, 0),
                )
                return ResolutionResult(
                    conflict_id=conflict.conflict_id,
                    resolved=True,
                    strategy_used=strategy,
                    resolved_value=best.value,
                    confidence=best.confidence,
                    explanation=f"Selected value from highest priority source: {best.source_id}",
                    rule_applied=rule.rule_id if rule else None,
                )

        elif strategy == ResolutionStrategy.MERGE_VALUES:
            merged = "; ".join(str(v.value) for v in values)
            avg_conf = sum(v.confidence for v in values) / len(values)
            return ResolutionResult(
                conflict_id=conflict.conflict_id,
                resolved=True,
                strategy_used=strategy,
                resolved_value=merged,
                confidence=avg_conf,
                explanation=f"Merged {len(values)} values",
                rule_applied=rule.rule_id if rule else None,
            )

        elif strategy == ResolutionStrategy.KEEP_ALL:
            all_values = [v.value for v in values]
            return ResolutionResult(
                conflict_id=conflict.conflict_id,
                resolved=True,
                strategy_used=strategy,
                resolved_value=all_values,
                confidence=0.5,
                explanation="Kept all conflicting values",
                rule_applied=rule.rule_id if rule else None,
            )

        elif strategy == ResolutionStrategy.DISCARD_ALL:
            return ResolutionResult(
                conflict_id=conflict.conflict_id,
                resolved=True,
                strategy_used=strategy,
                resolved_value=None,
                confidence=0.0,
                explanation="Discarded all conflicting values",
                rule_applied=rule.rule_id if rule else None,
            )

        # Default: human review
        return ResolutionResult(
            conflict_id=conflict.conflict_id,
            resolved=False,
            strategy_used=ResolutionStrategy.HUMAN_REVIEW,
            requires_review=True,
            explanation="Requires manual review",
        )

    def validate_resolution(
        self,
        conflict: Conflict,
    ) -> ValidationResult:
        """Validate a conflict resolution."""
        import uuid

        checks_passed = []
        checks_failed = []
        warnings = []

        # Check if resolved
        if conflict.resolved:
            checks_passed.append("Conflict is marked as resolved")
        else:
            checks_failed.append("Conflict is not resolved")

        # Check resolved value
        if conflict.resolved_value is not None:
            checks_passed.append("Resolved value is set")

            # Check if resolved value matches one of the original values
            original_values = [v.value for v in conflict.conflicting_values]
            if conflict.resolved_value in original_values:
                checks_passed.append("Resolved value matches an original value")
            else:
                warnings.append("Resolved value differs from all original values")
        else:
            if conflict.resolution != ResolutionStrategy.DISCARD_ALL.value:
                checks_failed.append("Resolved value is missing")

        # Check resolution method
        if conflict.resolution:
            checks_passed.append(f"Resolution method recorded: {conflict.resolution}")
        else:
            warnings.append("Resolution method not recorded")

        # Calculate score
        total_checks = len(checks_passed) + len(checks_failed)
        score = len(checks_passed) / total_checks if total_checks > 0 else 0.0

        return ValidationResult(
            validation_id=f"val_{uuid.uuid4().hex[:8]}",
            is_valid=len(checks_failed) == 0,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            warnings=warnings,
            score=score,
            details={
                "conflict_id": conflict.conflict_id,
                "conflict_type": conflict.conflict_type.value,
                "resolution": conflict.resolution,
            },
        )

    def add_rule(self, rule: ResolutionRule) -> None:
        """Add a custom resolution rule."""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def set_source_priority(self, source_id: str, priority: int) -> None:
        """Set priority for a source."""
        self.source_priorities[source_id] = priority

    def get_stats(self) -> ConflictStats:
        """Get conflict statistics."""
        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        by_source: dict[str, int] = {}
        resolved = 0
        pending = 0

        for conflict in self.conflicts.values():
            # By type
            type_key = conflict.conflict_type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1

            # By severity
            sev_key = conflict.severity.value
            by_severity[sev_key] = by_severity.get(sev_key, 0) + 1

            # By source
            for cv in conflict.conflicting_values:
                by_source[cv.source_id] = by_source.get(cv.source_id, 0) + 1

            if conflict.resolved:
                resolved += 1
            else:
                pending += 1

        total = len(self.conflicts)
        resolution_rate = resolved / total if total > 0 else 0.0

        # Calculate avg resolution time
        resolution_times = []
        for conflict in self.conflicts.values():
            if conflict.resolved and conflict.resolved_at:
                delta = conflict.resolved_at - conflict.detected_at
                resolution_times.append(delta.total_seconds() / 3600)  # hours

        avg_time = (
            sum(resolution_times) / len(resolution_times) if resolution_times else 0.0
        )

        return ConflictStats(
            total_conflicts=total,
            resolved_conflicts=resolved,
            pending_conflicts=pending,
            by_type=by_type,
            by_severity=by_severity,
            by_source=by_source,
            resolution_rate=resolution_rate,
            avg_resolution_time=avg_time,
        )

    def get_pending_conflicts(
        self,
        severity: Optional[ConflictSeverity] = None,
    ) -> list[Conflict]:
        """Get pending conflicts, optionally filtered by severity."""
        pending = [c for c in self.conflicts.values() if not c.resolved]

        if severity:
            pending = [c for c in pending if c.severity == severity]

        # Sort by severity
        severity_order = {
            ConflictSeverity.CRITICAL: 0,
            ConflictSeverity.HIGH: 1,
            ConflictSeverity.MEDIUM: 2,
            ConflictSeverity.LOW: 3,
            ConflictSeverity.INFO: 4,
        }
        pending.sort(key=lambda c: severity_order.get(c.severity, 5))

        return pending


# Global instance
_conflict_resolver: Optional[ConflictResolver] = None


def get_conflict_resolver() -> ConflictResolver:
    """Get or create the global conflict resolver instance."""
    global _conflict_resolver
    if _conflict_resolver is None:
        _conflict_resolver = ConflictResolver()
    return _conflict_resolver
