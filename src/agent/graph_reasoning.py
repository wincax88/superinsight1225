"""
Knowledge Graph Integrated Reasoning Module.

Provides graph-based reasoning capabilities including entity relation inference,
path-based reasoning, and graph-driven Q&A.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Set
from uuid import UUID
import asyncio

logger = logging.getLogger(__name__)


class GraphReasoningType(str, Enum):
    """Types of graph reasoning."""
    ENTITY_RELATION = "entity_relation"       # Infer relations between entities
    PATH_BASED = "path_based"                 # Reason based on graph paths
    PATTERN_MATCHING = "pattern_matching"     # Match graph patterns
    SUBGRAPH_ANALYSIS = "subgraph_analysis"   # Analyze subgraph structures
    TEMPORAL = "temporal"                     # Time-based reasoning
    SIMILARITY = "similarity"                 # Entity similarity reasoning


class InferenceConfidence(str, Enum):
    """Confidence levels for graph inferences."""
    HIGH = "high"           # >= 0.8
    MEDIUM = "medium"       # 0.5 - 0.8
    LOW = "low"             # 0.3 - 0.5
    UNCERTAIN = "uncertain" # < 0.3


@dataclass
class GraphEntity:
    """Simplified entity for graph reasoning."""
    id: str
    entity_type: str
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class GraphRelation:
    """Simplified relation for graph reasoning."""
    id: str
    source_id: str
    target_id: str
    relation_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    confidence: float = 1.0


@dataclass
class GraphPath:
    """A path in the knowledge graph."""
    nodes: List[GraphEntity] = field(default_factory=list)
    edges: List[GraphRelation] = field(default_factory=list)
    total_weight: float = 0.0
    average_confidence: float = 0.0

    def calculate_metrics(self) -> None:
        """Calculate path metrics."""
        if self.edges:
            self.total_weight = sum(e.weight for e in self.edges)
            self.average_confidence = sum(e.confidence for e in self.edges) / len(self.edges)


@dataclass
class GraphPattern:
    """A pattern to match in the graph."""
    entity_types: List[str] = field(default_factory=list)
    relation_types: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    """Result of a graph inference."""
    inference_type: GraphReasoningType
    source_entities: List[str] = field(default_factory=list)
    target_entities: List[str] = field(default_factory=list)
    inferred_relations: List[Dict[str, Any]] = field(default_factory=list)
    supporting_paths: List[GraphPath] = field(default_factory=list)
    confidence: float = 0.0
    confidence_level: InferenceConfidence = InferenceConfidence.UNCERTAIN
    explanation: str = ""
    evidence: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def set_confidence(self, confidence: float) -> None:
        """Set confidence and determine level."""
        self.confidence = confidence
        if confidence >= 0.8:
            self.confidence_level = InferenceConfidence.HIGH
        elif confidence >= 0.5:
            self.confidence_level = InferenceConfidence.MEDIUM
        elif confidence >= 0.3:
            self.confidence_level = InferenceConfidence.LOW
        else:
            self.confidence_level = InferenceConfidence.UNCERTAIN


@dataclass
class GraphQuestion:
    """A question to be answered using the knowledge graph."""
    question: str
    entities_mentioned: List[str] = field(default_factory=list)
    relation_types_mentioned: List[str] = field(default_factory=list)
    question_type: str = "factual"  # factual, comparative, exploratory, causal


@dataclass
class GraphAnswer:
    """Answer derived from knowledge graph reasoning."""
    question: str
    answer: str
    entities_involved: List[GraphEntity] = field(default_factory=list)
    relations_used: List[GraphRelation] = field(default_factory=list)
    reasoning_paths: List[GraphPath] = field(default_factory=list)
    confidence: float = 0.0
    supporting_evidence: List[str] = field(default_factory=list)
    alternative_answers: List[str] = field(default_factory=list)


class GraphReasoningEngine:
    """Engine for knowledge graph-based reasoning."""

    def __init__(self, graph_db=None):
        """
        Initialize the graph reasoning engine.

        Args:
            graph_db: Optional graph database connection
        """
        self.graph_db = graph_db
        self.entity_cache: Dict[str, GraphEntity] = {}
        self.relation_cache: Dict[str, GraphRelation] = {}
        self.inference_rules: List[Dict[str, Any]] = []
        self.max_path_depth = 5
        self.min_confidence_threshold = 0.3

        # Initialize default inference rules
        self._load_default_inference_rules()

    def _load_default_inference_rules(self) -> None:
        """Load default inference rules."""
        self.inference_rules = [
            {
                "name": "transitive_works_for",
                "description": "If A works_for B and B subsidiary_of C, then A works_for C",
                "antecedent": [
                    {"relation": "works_for", "from": "A", "to": "B"},
                    {"relation": "subsidiary_of", "from": "B", "to": "C"}
                ],
                "consequent": {"relation": "works_for", "from": "A", "to": "C"},
                "confidence_factor": 0.8
            },
            {
                "name": "symmetric_related_to",
                "description": "If A related_to B, then B related_to A",
                "antecedent": [
                    {"relation": "related_to", "from": "A", "to": "B"}
                ],
                "consequent": {"relation": "related_to", "from": "B", "to": "A"},
                "confidence_factor": 1.0
            },
            {
                "name": "co_occurrence_similarity",
                "description": "Entities appearing in same context are related",
                "antecedent": [
                    {"relation": "appears_with", "from": "A", "to": "B"},
                    {"count_threshold": 3}
                ],
                "consequent": {"relation": "similar_to", "from": "A", "to": "B"},
                "confidence_factor": 0.6
            },
            {
                "name": "hierarchical_membership",
                "description": "If A member_of B and B part_of C, then A belongs_to C",
                "antecedent": [
                    {"relation": "member_of", "from": "A", "to": "B"},
                    {"relation": "part_of", "from": "B", "to": "C"}
                ],
                "consequent": {"relation": "belongs_to", "from": "A", "to": "C"},
                "confidence_factor": 0.75
            }
        ]

    async def set_graph_db(self, graph_db) -> None:
        """Set the graph database connection."""
        self.graph_db = graph_db

    async def infer_entity_relations(
        self,
        entity_ids: List[str],
        max_depth: int = 2
    ) -> List[InferenceResult]:
        """
        Infer relations between specified entities.

        Args:
            entity_ids: List of entity IDs to analyze
            max_depth: Maximum path depth to search

        Returns:
            List of inference results
        """
        results = []

        # Find all paths between entities
        paths = await self._find_connecting_paths(entity_ids, max_depth)

        # Apply inference rules
        for rule in self.inference_rules:
            inferred = self._apply_inference_rule(rule, paths)
            if inferred:
                result = InferenceResult(
                    inference_type=GraphReasoningType.ENTITY_RELATION,
                    source_entities=entity_ids,
                    inferred_relations=[inferred],
                    explanation=rule["description"],
                    evidence=[f"Applied rule: {rule['name']}"]
                )
                result.set_confidence(inferred.get("confidence", 0.5))
                results.append(result)

        # Analyze path patterns for implicit relations
        pattern_results = await self._analyze_path_patterns(paths)
        results.extend(pattern_results)

        return results

    async def _find_connecting_paths(
        self,
        entity_ids: List[str],
        max_depth: int
    ) -> List[GraphPath]:
        """Find paths connecting the specified entities."""
        paths = []

        if self.graph_db:
            # Use actual graph database queries
            for i, source_id in enumerate(entity_ids):
                for target_id in entity_ids[i + 1:]:
                    try:
                        result = await self.graph_db.find_path(
                            UUID(source_id),
                            UUID(target_id),
                            max_depth=max_depth
                        )
                        if result.paths:
                            for path_data in result.paths:
                                path = self._convert_path_data(path_data)
                                paths.append(path)
                    except Exception as e:
                        logger.warning(f"Error finding path: {e}")
        else:
            # Return mock paths for testing
            paths = self._generate_mock_paths(entity_ids)

        return paths

    def _convert_path_data(self, path_data: List[Dict]) -> GraphPath:
        """Convert path data from database to GraphPath."""
        path = GraphPath()

        for item in path_data:
            if item.get("type") == "node":
                entity = GraphEntity(
                    id=item.get("id", ""),
                    entity_type=item.get("entity_type", "unknown"),
                    name=item.get("name", "")
                )
                path.nodes.append(entity)
            elif item.get("type") == "relation":
                relation = GraphRelation(
                    id=item.get("id", ""),
                    source_id="",
                    target_id="",
                    relation_type=item.get("relation_type", "related_to")
                )
                path.edges.append(relation)

        path.calculate_metrics()
        return path

    def _generate_mock_paths(self, entity_ids: List[str]) -> List[GraphPath]:
        """Generate mock paths for testing."""
        paths = []

        if len(entity_ids) >= 2:
            path = GraphPath(
                nodes=[
                    GraphEntity(id=entity_ids[0], entity_type="entity", name=f"Entity_{entity_ids[0][:8]}"),
                    GraphEntity(id="intermediate", entity_type="entity", name="Intermediate"),
                    GraphEntity(id=entity_ids[1], entity_type="entity", name=f"Entity_{entity_ids[1][:8]}")
                ],
                edges=[
                    GraphRelation(id="rel1", source_id=entity_ids[0], target_id="intermediate", relation_type="related_to"),
                    GraphRelation(id="rel2", source_id="intermediate", target_id=entity_ids[1], relation_type="connected_to")
                ]
            )
            path.calculate_metrics()
            paths.append(path)

        return paths

    def _apply_inference_rule(
        self,
        rule: Dict[str, Any],
        paths: List[GraphPath]
    ) -> Optional[Dict[str, Any]]:
        """Apply an inference rule to the paths."""
        antecedent = rule.get("antecedent", [])
        consequent = rule.get("consequent", {})
        confidence_factor = rule.get("confidence_factor", 1.0)

        # Check if antecedent conditions are met
        bindings = {}
        antecedent_satisfied = True

        for condition in antecedent:
            if "relation" in condition:
                # Look for this relation in paths
                found = False
                for path in paths:
                    for edge in path.edges:
                        if edge.relation_type == condition["relation"]:
                            found = True
                            # Bind variables
                            if condition.get("from"):
                                bindings[condition["from"]] = edge.source_id
                            if condition.get("to"):
                                bindings[condition["to"]] = edge.target_id
                            break
                    if found:
                        break

                if not found:
                    antecedent_satisfied = False
                    break

        if antecedent_satisfied and consequent:
            # Generate inferred relation
            inferred = {
                "relation_type": consequent.get("relation"),
                "source_id": bindings.get(consequent.get("from"), ""),
                "target_id": bindings.get(consequent.get("to"), ""),
                "confidence": confidence_factor,
                "inferred_by": rule.get("name")
            }
            return inferred

        return None

    async def _analyze_path_patterns(
        self,
        paths: List[GraphPath]
    ) -> List[InferenceResult]:
        """Analyze path patterns for implicit relations."""
        results = []

        # Analyze common patterns
        relation_counts: Dict[str, int] = {}
        entity_type_counts: Dict[str, int] = {}

        for path in paths:
            for edge in path.edges:
                relation_counts[edge.relation_type] = relation_counts.get(edge.relation_type, 0) + 1
            for node in path.nodes:
                entity_type_counts[node.entity_type] = entity_type_counts.get(node.entity_type, 0) + 1

        # Identify dominant patterns
        if relation_counts:
            dominant_relation = max(relation_counts.items(), key=lambda x: x[1])
            if dominant_relation[1] >= 2:  # At least 2 occurrences
                result = InferenceResult(
                    inference_type=GraphReasoningType.PATTERN_MATCHING,
                    inferred_relations=[{
                        "pattern": f"Dominant relation: {dominant_relation[0]}",
                        "count": dominant_relation[1]
                    }],
                    explanation=f"Pattern detected: '{dominant_relation[0]}' appears {dominant_relation[1]} times",
                    supporting_paths=paths
                )
                result.set_confidence(min(0.9, 0.5 + dominant_relation[1] * 0.1))
                results.append(result)

        return results

    async def path_based_reasoning(
        self,
        source_entity: str,
        target_entity: str,
        max_depth: int = 3
    ) -> InferenceResult:
        """
        Perform path-based reasoning between two entities.

        Args:
            source_entity: Source entity ID
            target_entity: Target entity ID
            max_depth: Maximum path depth

        Returns:
            Inference result with path analysis
        """
        paths = await self._find_connecting_paths([source_entity, target_entity], max_depth)

        if not paths:
            return InferenceResult(
                inference_type=GraphReasoningType.PATH_BASED,
                source_entities=[source_entity],
                target_entities=[target_entity],
                explanation="No connecting paths found between entities",
                confidence=0.0
            )

        # Analyze paths
        best_path = min(paths, key=lambda p: len(p.edges))
        all_relation_types = set()
        for path in paths:
            for edge in path.edges:
                all_relation_types.add(edge.relation_type)

        result = InferenceResult(
            inference_type=GraphReasoningType.PATH_BASED,
            source_entities=[source_entity],
            target_entities=[target_entity],
            supporting_paths=paths,
            explanation=f"Found {len(paths)} paths connecting entities",
            evidence=[
                f"Shortest path length: {len(best_path.edges)}",
                f"Relation types involved: {', '.join(all_relation_types)}"
            ]
        )

        # Calculate confidence based on path characteristics
        path_confidence = 1.0 / (1.0 + len(best_path.edges) * 0.1)  # Shorter paths = higher confidence
        result.set_confidence(path_confidence)

        return result

    async def pattern_matching_reasoning(
        self,
        pattern: GraphPattern,
        limit: int = 10
    ) -> List[InferenceResult]:
        """
        Find and analyze instances of a graph pattern.

        Args:
            pattern: Pattern to match
            limit: Maximum results

        Returns:
            List of matching pattern instances
        """
        results = []

        # Build pattern query
        if self.graph_db:
            try:
                # Construct Cypher pattern query
                entity_conditions = []
                for i, etype in enumerate(pattern.entity_types):
                    entity_conditions.append(f"e{i}.entity_type = '{etype}'")

                # This is a simplified implementation - production would use proper Cypher
                cypher = "MATCH (e:Entity) WHERE e.is_active = true"
                if entity_conditions:
                    cypher += " AND (" + " OR ".join(entity_conditions) + ")"
                cypher += f" RETURN e LIMIT {limit}"

                query_results = await self.graph_db.execute_cypher(cypher)

                for record in query_results:
                    result = InferenceResult(
                        inference_type=GraphReasoningType.PATTERN_MATCHING,
                        inferred_relations=[{
                            "matched_pattern": pattern.entity_types,
                            "entity_data": record
                        }],
                        explanation="Pattern match found"
                    )
                    result.set_confidence(0.85)
                    results.append(result)

            except Exception as e:
                logger.warning(f"Pattern matching query failed: {e}")

        else:
            # Mock results for testing
            result = InferenceResult(
                inference_type=GraphReasoningType.PATTERN_MATCHING,
                inferred_relations=[{
                    "matched_pattern": pattern.entity_types,
                    "mock": True
                }],
                explanation="Mock pattern match"
            )
            result.set_confidence(0.7)
            results.append(result)

        return results

    async def similarity_reasoning(
        self,
        entity_id: str,
        top_k: int = 5
    ) -> InferenceResult:
        """
        Find and reason about similar entities.

        Args:
            entity_id: Source entity ID
            top_k: Number of similar entities to find

        Returns:
            Inference result with similarity analysis
        """
        similar_entities = []
        similarity_evidence = []

        if self.graph_db:
            try:
                # Get entity neighbors and their properties
                neighbors = await self.graph_db.get_neighbors(UUID(entity_id), depth=2)

                # Calculate similarity based on shared neighbors
                neighbor_counts: Dict[str, int] = {}
                for entity in neighbors.entities:
                    neighbor_counts[entity.id] = neighbor_counts.get(entity.id, 0) + 1

                # Sort by neighbor overlap
                sorted_similar = sorted(neighbor_counts.items(), key=lambda x: x[1], reverse=True)

                for eid, count in sorted_similar[:top_k]:
                    similar_entities.append({
                        "entity_id": str(eid),
                        "shared_neighbors": count,
                        "similarity_score": count / (count + 1)  # Normalized
                    })
                    similarity_evidence.append(f"Entity {eid} shares {count} neighbors")

            except Exception as e:
                logger.warning(f"Similarity reasoning failed: {e}")

        if not similar_entities:
            # Mock data for testing
            similar_entities = [
                {"entity_id": "similar_1", "shared_neighbors": 3, "similarity_score": 0.75},
                {"entity_id": "similar_2", "shared_neighbors": 2, "similarity_score": 0.67}
            ]
            similarity_evidence = ["Mock similarity data"]

        result = InferenceResult(
            inference_type=GraphReasoningType.SIMILARITY,
            source_entities=[entity_id],
            target_entities=[e["entity_id"] for e in similar_entities],
            inferred_relations=similar_entities,
            explanation=f"Found {len(similar_entities)} similar entities",
            evidence=similarity_evidence
        )

        avg_similarity = sum(e["similarity_score"] for e in similar_entities) / len(similar_entities) if similar_entities else 0
        result.set_confidence(avg_similarity)

        return result

    async def answer_question(
        self,
        question: GraphQuestion
    ) -> GraphAnswer:
        """
        Answer a question using knowledge graph reasoning.

        Args:
            question: Question to answer

        Returns:
            Answer derived from graph reasoning
        """
        answer = GraphAnswer(
            question=question.question,
            answer="",
            supporting_evidence=[]
        )

        entities_involved = []
        relations_used = []
        reasoning_paths = []

        # Step 1: Find mentioned entities in graph
        mentioned_entities = question.entities_mentioned
        if self.graph_db and mentioned_entities:
            for entity_name in mentioned_entities:
                try:
                    entities = await self.graph_db.search_entities(query_text=entity_name, limit=5)
                    entities_involved.extend([
                        GraphEntity(
                            id=str(e.id),
                            entity_type=e.entity_type.value if hasattr(e.entity_type, 'value') else str(e.entity_type),
                            name=e.name,
                            properties=e.properties
                        )
                        for e in entities
                    ])
                except Exception as e:
                    logger.warning(f"Entity search failed: {e}")

        # Step 2: Find relations based on question type
        if question.question_type == "factual":
            # Look for direct relations
            if len(entities_involved) >= 1:
                for entity in entities_involved[:3]:  # Limit to first 3
                    inference = await self.path_based_reasoning(
                        entity.id,
                        entities_involved[1].id if len(entities_involved) > 1 else entity.id
                    )
                    reasoning_paths.extend(inference.supporting_paths)

        elif question.question_type == "comparative":
            # Compare entities
            if len(entities_involved) >= 2:
                for i, e1 in enumerate(entities_involved[:-1]):
                    for e2 in entities_involved[i + 1:]:
                        inference = await self.similarity_reasoning(e1.id)
                        answer.supporting_evidence.append(
                            f"Similarity analysis for {e1.name}: {inference.explanation}"
                        )

        elif question.question_type == "exploratory":
            # Explore entity neighborhoods
            if entities_involved:
                entity = entities_involved[0]
                if self.graph_db:
                    try:
                        neighbors = await self.graph_db.get_neighbors(UUID(entity.id))
                        answer.supporting_evidence.append(
                            f"Found {len(neighbors.entities)} related entities"
                        )
                        for rel in neighbors.relations[:5]:
                            relations_used.append(GraphRelation(
                                id=str(rel.id),
                                source_id=str(rel.source_id),
                                target_id=str(rel.target_id),
                                relation_type=rel.relation_type.value if hasattr(rel.relation_type, 'value') else str(rel.relation_type)
                            ))
                    except Exception as e:
                        logger.warning(f"Exploration failed: {e}")

        # Step 3: Generate answer
        if entities_involved or reasoning_paths:
            answer_parts = []

            if entities_involved:
                entity_names = [e.name for e in entities_involved[:5]]
                answer_parts.append(f"Found relevant entities: {', '.join(entity_names)}")

            if reasoning_paths:
                answer_parts.append(f"Discovered {len(reasoning_paths)} reasoning paths")

            if relations_used:
                relation_types = list(set(r.relation_type for r in relations_used))
                answer_parts.append(f"Identified relations: {', '.join(relation_types)}")

            answer.answer = ". ".join(answer_parts)
            answer.confidence = 0.7

        else:
            answer.answer = "Unable to find sufficient information in the knowledge graph to answer this question."
            answer.confidence = 0.2

        answer.entities_involved = entities_involved
        answer.relations_used = relations_used
        answer.reasoning_paths = reasoning_paths

        return answer

    def add_inference_rule(self, rule: Dict[str, Any]) -> None:
        """Add a custom inference rule."""
        self.inference_rules.append(rule)

    def remove_inference_rule(self, rule_name: str) -> bool:
        """Remove an inference rule by name."""
        for i, rule in enumerate(self.inference_rules):
            if rule.get("name") == rule_name:
                self.inference_rules.pop(i)
                return True
        return False


# Global instance
_graph_reasoning_engine: Optional[GraphReasoningEngine] = None


def get_graph_reasoning_engine(graph_db=None) -> GraphReasoningEngine:
    """Get or create global graph reasoning engine instance."""
    global _graph_reasoning_engine
    if _graph_reasoning_engine is None:
        _graph_reasoning_engine = GraphReasoningEngine(graph_db)
    return _graph_reasoning_engine


async def initialize_graph_reasoning(graph_db) -> GraphReasoningEngine:
    """Initialize graph reasoning with database connection."""
    engine = get_graph_reasoning_engine()
    await engine.set_graph_db(graph_db)
    return engine
