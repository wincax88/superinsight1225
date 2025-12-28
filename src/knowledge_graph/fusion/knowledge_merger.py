"""
Knowledge merger module for Knowledge Graph fusion.

Provides multi-source knowledge fusion, quality assessment, and external KB integration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MergeStrategy(str, Enum):
    """Strategies for merging knowledge."""

    UNION = "union"  # Keep all unique facts
    INTERSECTION = "intersection"  # Keep only common facts
    SOURCE_PRIORITY = "source_priority"  # Prefer specific source
    CONFIDENCE_BASED = "confidence_based"  # Prefer higher confidence
    RECENCY_BASED = "recency_based"  # Prefer more recent
    VOTING = "voting"  # Majority voting


class QualityMetric(str, Enum):
    """Quality metrics for knowledge."""

    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    PROVENANCE = "provenance"


class ExternalKBType(str, Enum):
    """Types of external knowledge bases."""

    DBPEDIA = "dbpedia"
    WIKIDATA = "wikidata"
    FREEBASE = "freebase"
    YAGO = "yago"
    CONCEPTNET = "conceptnet"
    CUSTOM = "custom"


class KnowledgeSource(BaseModel):
    """A source of knowledge."""

    source_id: str = Field(..., description="Source identifier")
    source_name: str = Field(..., description="Human-readable name")
    source_type: str = Field(default="internal", description="Type of source")
    priority: int = Field(default=1, description="Priority level (higher = better)")
    reliability_score: float = Field(default=0.8, description="Source reliability (0-1)")
    last_updated: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeFact(BaseModel):
    """A single fact/triple in the knowledge graph."""

    fact_id: str = Field(..., description="Unique fact identifier")
    subject: str = Field(..., description="Subject entity")
    predicate: str = Field(..., description="Predicate/relation")
    object: str = Field(..., description="Object entity/value")
    source_id: str = Field(..., description="Source of this fact")
    confidence: float = Field(default=1.0, description="Confidence score")
    timestamp: datetime = Field(default_factory=datetime.now)
    properties: dict[str, Any] = Field(default_factory=dict)


class MergedFact(BaseModel):
    """A fact merged from multiple sources."""

    fact_id: str = Field(..., description="Merged fact identifier")
    subject: str = Field(..., description="Subject entity")
    predicate: str = Field(..., description="Predicate/relation")
    object: str = Field(..., description="Object value (merged)")
    sources: list[str] = Field(default_factory=list, description="Contributing sources")
    merged_confidence: float = Field(default=1.0, description="Merged confidence")
    original_values: dict[str, Any] = Field(
        default_factory=dict, description="Values from each source"
    )
    merge_strategy: MergeStrategy = Field(default=MergeStrategy.CONFIDENCE_BASED)
    conflict_detected: bool = Field(default=False)
    timestamp: datetime = Field(default_factory=datetime.now)


class QualityAssessment(BaseModel):
    """Quality assessment of merged knowledge."""

    assessment_id: str = Field(..., description="Assessment identifier")
    source_id: Optional[str] = Field(default=None, description="Source being assessed")
    completeness: float = Field(default=0.0, description="Completeness score")
    accuracy: float = Field(default=0.0, description="Accuracy score")
    consistency: float = Field(default=0.0, description="Consistency score")
    timeliness: float = Field(default=0.0, description="Timeliness score")
    overall_quality: float = Field(default=0.0, description="Overall quality score")
    issues: list[str] = Field(default_factory=list, description="Identified issues")
    recommendations: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class MergeConfig(BaseModel):
    """Configuration for knowledge merging."""

    strategy: MergeStrategy = Field(default=MergeStrategy.CONFIDENCE_BASED)
    source_priorities: dict[str, int] = Field(default_factory=dict)
    confidence_threshold: float = Field(default=0.5)
    conflict_threshold: float = Field(default=0.3)
    merge_attributes: bool = Field(default=True)
    track_provenance: bool = Field(default=True)
    enable_quality_assessment: bool = Field(default=True)


class MergeResult(BaseModel):
    """Result of knowledge merging operation."""

    merged_facts: list[MergedFact] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    quality_assessment: Optional[QualityAssessment] = None
    total_facts_processed: int = Field(default=0)
    facts_merged: int = Field(default=0)
    conflicts_detected: int = Field(default=0)
    conflicts_resolved: int = Field(default=0)
    sources_used: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


@dataclass
class KnowledgeMerger:
    """Knowledge merger for multi-source fusion."""

    config: MergeConfig = field(default_factory=MergeConfig)
    sources: dict[str, KnowledgeSource] = field(default_factory=dict)
    facts_by_key: dict[str, list[KnowledgeFact]] = field(default_factory=dict)

    def register_source(self, source: KnowledgeSource) -> None:
        """Register a knowledge source."""
        self.sources[source.source_id] = source
        logger.info(f"Registered source: {source.source_name}")

    def add_facts(
        self,
        facts: list[dict[str, Any]],
        source_id: str,
    ) -> int:
        """Add facts from a source."""
        count = 0
        for fact_data in facts:
            fact = KnowledgeFact(
                fact_id=fact_data.get("id", f"{source_id}_{count}"),
                subject=str(fact_data.get("subject", fact_data.get("head", ""))),
                predicate=str(fact_data.get("predicate", fact_data.get("relation", ""))),
                object=str(fact_data.get("object", fact_data.get("tail", ""))),
                source_id=source_id,
                confidence=fact_data.get("confidence", 1.0),
                properties=fact_data.get("properties", {}),
            )

            # Group by subject-predicate key
            key = f"{fact.subject}::{fact.predicate}"
            if key not in self.facts_by_key:
                self.facts_by_key[key] = []
            self.facts_by_key[key].append(fact)
            count += 1

        logger.info(f"Added {count} facts from source {source_id}")
        return count

    def merge(
        self,
        config: Optional[MergeConfig] = None,
    ) -> MergeResult:
        """Merge all facts according to strategy."""
        if config:
            self.config = config

        merged_facts = []
        conflicts = []
        facts_processed = 0
        conflicts_detected = 0
        conflicts_resolved = 0

        for key, fact_group in self.facts_by_key.items():
            facts_processed += len(fact_group)

            if len(fact_group) == 1:
                # Single source, no merging needed
                fact = fact_group[0]
                merged = MergedFact(
                    fact_id=f"merged_{fact.fact_id}",
                    subject=fact.subject,
                    predicate=fact.predicate,
                    object=fact.object,
                    sources=[fact.source_id],
                    merged_confidence=fact.confidence,
                    merge_strategy=self.config.strategy,
                )
                merged_facts.append(merged)
            else:
                # Multiple sources, need to merge
                merged, conflict = self._merge_facts(fact_group)
                merged_facts.append(merged)

                if conflict:
                    conflicts.append(conflict)
                    conflicts_detected += 1
                    if merged.object:  # Conflict was resolved
                        conflicts_resolved += 1

        # Quality assessment
        quality_assessment = None
        if self.config.enable_quality_assessment:
            quality_assessment = self._assess_quality(merged_facts)

        return MergeResult(
            merged_facts=merged_facts,
            conflicts=conflicts,
            quality_assessment=quality_assessment,
            total_facts_processed=facts_processed,
            facts_merged=len(merged_facts),
            conflicts_detected=conflicts_detected,
            conflicts_resolved=conflicts_resolved,
            sources_used=list(self.sources.keys()),
        )

    def _merge_facts(
        self,
        facts: list[KnowledgeFact],
    ) -> tuple[MergedFact, Optional[dict[str, Any]]]:
        """Merge a group of facts for the same subject-predicate."""
        # Collect all unique values
        values: dict[str, list[KnowledgeFact]] = {}
        for fact in facts:
            if fact.object not in values:
                values[fact.object] = []
            values[fact.object].append(fact)

        # Check for conflict
        conflict = None
        if len(values) > 1:
            conflict = {
                "subject": facts[0].subject,
                "predicate": facts[0].predicate,
                "values": {
                    v: [f.source_id for f in fs] for v, fs in values.items()
                },
            }

        # Select value based on strategy
        selected_value, confidence = self._select_value(values, facts)

        merged = MergedFact(
            fact_id=f"merged_{facts[0].subject}_{facts[0].predicate}",
            subject=facts[0].subject,
            predicate=facts[0].predicate,
            object=selected_value,
            sources=[f.source_id for f in facts],
            merged_confidence=confidence,
            original_values={f.source_id: f.object for f in facts},
            merge_strategy=self.config.strategy,
            conflict_detected=len(values) > 1,
        )

        return merged, conflict

    def _select_value(
        self,
        values: dict[str, list[KnowledgeFact]],
        all_facts: list[KnowledgeFact],
    ) -> tuple[str, float]:
        """Select the best value based on merge strategy."""
        if self.config.strategy == MergeStrategy.VOTING:
            # Select value with most sources
            best_value = max(values.keys(), key=lambda v: len(values[v]))
            confidence = len(values[best_value]) / len(all_facts)
            return best_value, confidence

        elif self.config.strategy == MergeStrategy.CONFIDENCE_BASED:
            # Select value with highest average confidence
            best_value = ""
            best_confidence = 0.0

            for value, facts in values.items():
                avg_conf = sum(f.confidence for f in facts) / len(facts)
                if avg_conf > best_confidence:
                    best_confidence = avg_conf
                    best_value = value

            return best_value, best_confidence

        elif self.config.strategy == MergeStrategy.SOURCE_PRIORITY:
            # Select value from highest priority source
            priorities = self.config.source_priorities
            best_value = ""
            best_priority = -1
            best_confidence = 0.0

            for fact in all_facts:
                priority = priorities.get(fact.source_id, 0)
                if priority > best_priority:
                    best_priority = priority
                    best_value = fact.object
                    best_confidence = fact.confidence

            return best_value, best_confidence

        elif self.config.strategy == MergeStrategy.RECENCY_BASED:
            # Select most recent value
            most_recent = max(all_facts, key=lambda f: f.timestamp)
            return most_recent.object, most_recent.confidence

        elif self.config.strategy == MergeStrategy.UNION:
            # Combine all values (return first for simplicity)
            all_values = list(values.keys())
            combined = "; ".join(all_values)
            return combined, 1.0

        else:
            # Default: highest confidence
            best_fact = max(all_facts, key=lambda f: f.confidence)
            return best_fact.object, best_fact.confidence

    def _assess_quality(
        self,
        merged_facts: list[MergedFact],
    ) -> QualityAssessment:
        """Assess quality of merged knowledge."""
        import uuid

        issues = []
        recommendations = []

        # Completeness: ratio of facts with values
        non_empty = sum(1 for f in merged_facts if f.object)
        completeness = non_empty / len(merged_facts) if merged_facts else 0.0

        if completeness < 0.9:
            issues.append(f"Completeness is low: {completeness:.1%}")
            recommendations.append("Review and fill in missing values")

        # Consistency: ratio of non-conflicting facts
        conflicting = sum(1 for f in merged_facts if f.conflict_detected)
        consistency = 1.0 - (conflicting / len(merged_facts)) if merged_facts else 1.0

        if consistency < 0.8:
            issues.append(f"High conflict rate: {1-consistency:.1%}")
            recommendations.append("Investigate and resolve conflicts")

        # Accuracy: based on confidence scores
        if merged_facts:
            accuracy = sum(f.merged_confidence for f in merged_facts) / len(merged_facts)
        else:
            accuracy = 0.0

        if accuracy < 0.7:
            issues.append(f"Low average confidence: {accuracy:.1%}")
            recommendations.append("Verify low-confidence facts")

        # Timeliness: how recent the facts are
        if merged_facts:
            now = datetime.now()
            ages = [(now - f.timestamp).days for f in merged_facts]
            avg_age = sum(ages) / len(ages)
            timeliness = max(0, 1.0 - avg_age / 365)  # 1 year = 0 timeliness
        else:
            timeliness = 0.0

        if timeliness < 0.5:
            issues.append("Knowledge may be outdated")
            recommendations.append("Update facts from recent sources")

        overall = (completeness + accuracy + consistency + timeliness) / 4

        return QualityAssessment(
            assessment_id=f"qa_{uuid.uuid4().hex[:8]}",
            completeness=completeness,
            accuracy=accuracy,
            consistency=consistency,
            timeliness=timeliness,
            overall_quality=overall,
            issues=issues,
            recommendations=recommendations,
        )

    def integrate_external_kb(
        self,
        kb_type: ExternalKBType,
        entities: list[str],
        predicates: Optional[list[str]] = None,
    ) -> list[KnowledgeFact]:
        """Integrate knowledge from external knowledge base."""
        facts = []

        # Simulate external KB integration
        # In real implementation, this would query SPARQL endpoints or APIs

        for entity in entities:
            # Generate mock facts from external KB
            if kb_type == ExternalKBType.DBPEDIA:
                facts.extend(self._query_dbpedia(entity, predicates))
            elif kb_type == ExternalKBType.WIKIDATA:
                facts.extend(self._query_wikidata(entity, predicates))
            elif kb_type == ExternalKBType.CONCEPTNET:
                facts.extend(self._query_conceptnet(entity, predicates))

        # Register external source if not already
        source_id = f"external_{kb_type.value}"
        if source_id not in self.sources:
            self.register_source(
                KnowledgeSource(
                    source_id=source_id,
                    source_name=kb_type.value.title(),
                    source_type="external",
                    priority=2,
                    reliability_score=0.85,
                )
            )

        # Add facts
        if facts:
            self.add_facts(
                [f.model_dump() for f in facts],
                source_id,
            )

        return facts

    def _query_dbpedia(
        self,
        entity: str,
        predicates: Optional[list[str]],
    ) -> list[KnowledgeFact]:
        """Query DBpedia for entity facts (mock implementation)."""
        # Mock data for demonstration
        facts = []
        import uuid

        if entity:
            facts.append(
                KnowledgeFact(
                    fact_id=f"dbpedia_{uuid.uuid4().hex[:8]}",
                    subject=entity,
                    predicate="rdf:type",
                    object="owl:Thing",
                    source_id="external_dbpedia",
                    confidence=0.9,
                )
            )

        return facts

    def _query_wikidata(
        self,
        entity: str,
        predicates: Optional[list[str]],
    ) -> list[KnowledgeFact]:
        """Query Wikidata for entity facts (mock implementation)."""
        facts = []
        import uuid

        if entity:
            facts.append(
                KnowledgeFact(
                    fact_id=f"wikidata_{uuid.uuid4().hex[:8]}",
                    subject=entity,
                    predicate="wdt:P31",  # instance of
                    object="Q35120",  # entity
                    source_id="external_wikidata",
                    confidence=0.95,
                )
            )

        return facts

    def _query_conceptnet(
        self,
        entity: str,
        predicates: Optional[list[str]],
    ) -> list[KnowledgeFact]:
        """Query ConceptNet for entity facts (mock implementation)."""
        facts = []
        import uuid

        if entity:
            facts.append(
                KnowledgeFact(
                    fact_id=f"conceptnet_{uuid.uuid4().hex[:8]}",
                    subject=entity,
                    predicate="IsA",
                    object="concept",
                    source_id="external_conceptnet",
                    confidence=0.85,
                )
            )

        return facts

    def export_merged_knowledge(
        self,
        format: str = "json",
    ) -> dict[str, Any]:
        """Export merged knowledge in specified format."""
        result = self.merge()

        if format == "json":
            return {
                "facts": [f.model_dump() for f in result.merged_facts],
                "metadata": {
                    "total_facts": result.facts_merged,
                    "conflicts": result.conflicts_detected,
                    "sources": result.sources_used,
                    "timestamp": result.timestamp.isoformat(),
                },
            }
        elif format == "triples":
            return {
                "triples": [
                    (f.subject, f.predicate, f.object)
                    for f in result.merged_facts
                ],
            }
        else:
            return {"merged_facts": [f.model_dump() for f in result.merged_facts]}

    def clear(self) -> None:
        """Clear all facts and reset merger."""
        self.facts_by_key.clear()
        logger.info("Cleared all facts")


# Global instance
_knowledge_merger: Optional[KnowledgeMerger] = None


def get_knowledge_merger() -> KnowledgeMerger:
    """Get or create the global knowledge merger instance."""
    global _knowledge_merger
    if _knowledge_merger is None:
        _knowledge_merger = KnowledgeMerger()
    return _knowledge_merger
