"""
Cypher Query Generator for Knowledge Graph.

Generates Cypher queries from parsed natural language queries:
- Query template selection
- Parameter binding
- Query optimization
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from .nl_query_engine import QueryIntent, ParsedQuery, EntityType

logger = logging.getLogger(__name__)


class CypherTemplate(str, Enum):
    """Pre-defined Cypher query templates."""
    FIND_ENTITY_BY_NAME = "find_entity_by_name"
    FIND_ENTITY_BY_TYPE = "find_entity_by_type"
    GET_ENTITY_BY_ID = "get_entity_by_id"
    LIST_ENTITIES = "list_entities"
    COUNT_ENTITIES = "count_entities"
    FIND_RELATIONS = "find_relations"
    GET_NEIGHBORS = "get_neighbors"
    FIND_PATH = "find_path"
    SHORTEST_PATH = "shortest_path"
    GET_STATISTICS = "get_statistics"
    FIND_SIMILAR = "find_similar"
    CUSTOM = "custom"


@dataclass
class CypherQuery:
    """Generated Cypher query with parameters."""

    query: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    template: CypherTemplate = CypherTemplate.CUSTOM
    is_read_only: bool = True
    estimated_complexity: str = "low"  # low, medium, high
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "parameters": self.parameters,
            "template": self.template.value,
            "is_read_only": self.is_read_only,
            "estimated_complexity": self.estimated_complexity,
            "description": self.description,
        }


class CypherGenerationResult(BaseModel):
    """Result of Cypher query generation."""

    success: bool = Field(default=True, description="Whether generation succeeded")
    query: str = Field(default="", description="Generated Cypher query")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Query parameters")
    template_used: str = Field(default="", description="Template used")
    alternatives: List[Dict[str, Any]] = Field(default_factory=list, description="Alternative queries")
    explanation: str = Field(default="", description="Explanation of the query")
    warnings: List[str] = Field(default_factory=list, description="Generation warnings")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class CypherGenerator:
    """
    Cypher query generator from parsed natural language queries.

    Translates structured query information into executable Cypher queries.
    """

    # Cypher query templates
    TEMPLATES = {
        CypherTemplate.FIND_ENTITY_BY_NAME: """
MATCH (n)
WHERE toLower(n.name) CONTAINS toLower($name)
RETURN n
ORDER BY n.name
LIMIT $limit
""",

        CypherTemplate.FIND_ENTITY_BY_TYPE: """
MATCH (n:$entity_type)
RETURN n
ORDER BY n.name
LIMIT $limit
""",

        CypherTemplate.GET_ENTITY_BY_ID: """
MATCH (n)
WHERE n.id = $entity_id
RETURN n
""",

        CypherTemplate.LIST_ENTITIES: """
MATCH (n:$entity_type)
RETURN n
ORDER BY n.name
SKIP $offset
LIMIT $limit
""",

        CypherTemplate.COUNT_ENTITIES: """
MATCH (n:$entity_type)
RETURN count(n) as count
""",

        CypherTemplate.FIND_RELATIONS: """
MATCH (a)-[r]-(b)
WHERE a.id = $entity_id OR toLower(a.name) CONTAINS toLower($name)
RETURN a, r, b
LIMIT $limit
""",

        CypherTemplate.GET_NEIGHBORS: """
MATCH (n)-[r*1..$depth]-(neighbor)
WHERE n.id = $entity_id OR toLower(n.name) CONTAINS toLower($name)
RETURN DISTINCT neighbor, r
LIMIT $limit
""",

        CypherTemplate.FIND_PATH: """
MATCH path = shortestPath((a)-[*1..$depth]-(b))
WHERE (a.id = $source_id OR toLower(a.name) CONTAINS toLower($source_name))
  AND (b.id = $target_id OR toLower(b.name) CONTAINS toLower($target_name))
RETURN path
LIMIT $limit
""",

        CypherTemplate.SHORTEST_PATH: """
MATCH path = shortestPath((a)-[*]-(b))
WHERE (a.id = $source_id OR toLower(a.name) CONTAINS toLower($source_name))
  AND (b.id = $target_id OR toLower(b.name) CONTAINS toLower($target_name))
RETURN path, length(path) as pathLength
ORDER BY pathLength
LIMIT 1
""",

        CypherTemplate.GET_STATISTICS: """
MATCH (n)
WITH labels(n) as nodeLabels, count(n) as nodeCount
UNWIND nodeLabels as label
WITH label, sum(nodeCount) as count
RETURN label, count
ORDER BY count DESC
""",

        CypherTemplate.FIND_SIMILAR: """
MATCH (n)-[r]-(common)-[r2]-(similar)
WHERE n.id = $entity_id OR toLower(n.name) CONTAINS toLower($name)
  AND n <> similar
WITH similar, count(DISTINCT common) as commonConnections
ORDER BY commonConnections DESC
RETURN similar, commonConnections
LIMIT $limit
""",
    }

    # Entity type to Neo4j label mapping
    ENTITY_TYPE_LABELS = {
        EntityType.PERSON: "Person",
        EntityType.ORGANIZATION: "Organization",
        EntityType.LOCATION: "Location",
        EntityType.DATE: "Date",
        EntityType.PRODUCT: "Product",
        EntityType.EVENT: "Event",
        EntityType.CONCEPT: "Concept",
        EntityType.DOCUMENT: "Document",
        EntityType.UNKNOWN: "Entity",
    }

    def __init__(
        self,
        default_limit: int = 10,
        max_depth: int = 5,
        use_parameterized_queries: bool = True,
    ):
        """
        Initialize CypherGenerator.

        Args:
            default_limit: Default result limit
            max_depth: Maximum path depth
            use_parameterized_queries: Whether to use parameterized queries
        """
        self.default_limit = default_limit
        self.max_depth = max_depth
        self.use_parameterized_queries = use_parameterized_queries
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the generator."""
        if self._initialized:
            return
        self._initialized = True
        logger.info("CypherGenerator initialized")

    def generate(self, parsed_query: ParsedQuery) -> CypherGenerationResult:
        """
        Generate Cypher query from parsed natural language query.

        Args:
            parsed_query: Parsed query from NLQueryEngine

        Returns:
            CypherGenerationResult with generated query
        """
        if not self._initialized:
            self.initialize()

        if not parsed_query.is_valid:
            return CypherGenerationResult(
                success=False,
                error_message=parsed_query.error_message or "Invalid parsed query",
            )

        try:
            # Select template based on intent
            template = self._select_template(parsed_query.intent)

            # Generate query from template
            cypher_query = self._generate_from_template(template, parsed_query)

            # Generate alternative queries
            alternatives = self._generate_alternatives(parsed_query, template)

            # Create explanation
            explanation = self._create_explanation(parsed_query, template)

            # Check for warnings
            warnings = self._check_warnings(cypher_query, parsed_query)

            return CypherGenerationResult(
                success=True,
                query=cypher_query.query.strip(),
                parameters=cypher_query.parameters,
                template_used=template.value,
                alternatives=[alt.to_dict() for alt in alternatives],
                explanation=explanation,
                warnings=warnings,
            )

        except Exception as e:
            logger.error(f"Failed to generate Cypher query: {e}")
            return CypherGenerationResult(
                success=False,
                error_message=str(e),
            )

    def _select_template(self, intent: QueryIntent) -> CypherTemplate:
        """Select appropriate template based on query intent."""
        intent_to_template = {
            QueryIntent.FIND_ENTITY: CypherTemplate.FIND_ENTITY_BY_NAME,
            QueryIntent.GET_ENTITY: CypherTemplate.GET_ENTITY_BY_ID,
            QueryIntent.LIST_ENTITIES: CypherTemplate.LIST_ENTITIES,
            QueryIntent.COUNT_ENTITIES: CypherTemplate.COUNT_ENTITIES,
            QueryIntent.FIND_RELATIONS: CypherTemplate.FIND_RELATIONS,
            QueryIntent.GET_RELATION: CypherTemplate.FIND_RELATIONS,
            QueryIntent.FIND_CONNECTED: CypherTemplate.GET_NEIGHBORS,
            QueryIntent.FIND_PATH: CypherTemplate.FIND_PATH,
            QueryIntent.SHORTEST_PATH: CypherTemplate.SHORTEST_PATH,
            QueryIntent.GET_NEIGHBORS: CypherTemplate.GET_NEIGHBORS,
            QueryIntent.GET_STATISTICS: CypherTemplate.GET_STATISTICS,
            QueryIntent.FIND_SIMILAR: CypherTemplate.FIND_SIMILAR,
            QueryIntent.UNKNOWN: CypherTemplate.FIND_ENTITY_BY_NAME,
        }
        return intent_to_template.get(intent, CypherTemplate.FIND_ENTITY_BY_NAME)

    def _generate_from_template(self, template: CypherTemplate, parsed_query: ParsedQuery) -> CypherQuery:
        """Generate Cypher query from template and parsed query."""
        # Extract parameters from parsed query
        params = self._extract_parameters(parsed_query)

        # Get base template
        base_query = self.TEMPLATES.get(template, self.TEMPLATES[CypherTemplate.FIND_ENTITY_BY_NAME])

        # Build the query with proper parameters
        query = base_query

        # Replace entity type placeholder if needed
        entity_type = params.get("entity_type", "Entity")
        if "$entity_type" in query:
            # Note: Neo4j doesn't support parameterized labels, so we substitute directly
            query = query.replace("$entity_type", entity_type)
            if "entity_type" in params:
                del params["entity_type"]

        # Set complexity
        complexity = "low"
        if template in [CypherTemplate.FIND_PATH, CypherTemplate.SHORTEST_PATH]:
            complexity = "high"
        elif template in [CypherTemplate.FIND_SIMILAR, CypherTemplate.GET_NEIGHBORS]:
            complexity = "medium"

        return CypherQuery(
            query=query,
            parameters=params,
            template=template,
            is_read_only=True,
            estimated_complexity=complexity,
            description=self._get_template_description(template),
        )

    def _extract_parameters(self, parsed_query: ParsedQuery) -> Dict[str, Any]:
        """Extract Cypher parameters from parsed query."""
        params = {}

        # Get limit from parsed parameters
        query_params = parsed_query.parameters
        params["limit"] = query_params.get("limit", self.default_limit)
        params["offset"] = query_params.get("offset", 0)

        # Get depth
        depth = query_params.get("depth", self.max_depth)
        params["depth"] = min(depth if depth else self.max_depth, self.max_depth)

        # Extract entity information
        entities = parsed_query.entities
        if entities:
            first_entity = entities[0]
            params["name"] = first_entity.get("text", "")
            params["entity_id"] = first_entity.get("normalized", first_entity.get("text", ""))

            # Get entity type label
            entity_type = first_entity.get("entity_type", "unknown")
            params["entity_type"] = self.ENTITY_TYPE_LABELS.get(
                EntityType(entity_type) if entity_type in [e.value for e in EntityType] else EntityType.UNKNOWN,
                "Entity"
            )

            # For path queries, get source and target
            if len(entities) >= 2:
                second_entity = entities[1]
                params["source_name"] = first_entity.get("text", "")
                params["source_id"] = first_entity.get("normalized", first_entity.get("text", ""))
                params["target_name"] = second_entity.get("text", "")
                params["target_id"] = second_entity.get("normalized", second_entity.get("text", ""))
        else:
            # Use keywords as search terms
            if parsed_query.keywords:
                params["name"] = parsed_query.keywords[0]
                params["entity_id"] = parsed_query.keywords[0]
            else:
                params["name"] = ""
                params["entity_id"] = ""
            params["entity_type"] = "Entity"

        # Set default path parameters if not set
        if "source_name" not in params:
            params["source_name"] = params.get("name", "")
            params["source_id"] = params.get("entity_id", "")
        if "target_name" not in params:
            params["target_name"] = ""
            params["target_id"] = ""

        return params

    def _generate_alternatives(self, parsed_query: ParsedQuery, primary_template: CypherTemplate) -> List[CypherQuery]:
        """Generate alternative queries."""
        alternatives = []

        # Generate a more specific query if entities have types
        if parsed_query.entities and parsed_query.entities[0].get("entity_type") != "unknown":
            alt_template = CypherTemplate.FIND_ENTITY_BY_TYPE
            if alt_template != primary_template:
                alt_query = self._generate_from_template(alt_template, parsed_query)
                alt_query.description = "Search by entity type"
                alternatives.append(alt_query)

        # Generate a broader search if primary is specific
        if primary_template == CypherTemplate.GET_ENTITY_BY_ID:
            alt_template = CypherTemplate.FIND_ENTITY_BY_NAME
            alt_query = self._generate_from_template(alt_template, parsed_query)
            alt_query.description = "Broader name-based search"
            alternatives.append(alt_query)

        return alternatives[:3]  # Limit to 3 alternatives

    def _create_explanation(self, parsed_query: ParsedQuery, template: CypherTemplate) -> str:
        """Create human-readable explanation of the query."""
        intent = parsed_query.intent
        entities = parsed_query.entities

        explanations = {
            QueryIntent.FIND_ENTITY: "Search for entities matching the specified name",
            QueryIntent.GET_ENTITY: "Retrieve details for a specific entity",
            QueryIntent.LIST_ENTITIES: "List all entities of the specified type",
            QueryIntent.COUNT_ENTITIES: "Count the number of entities",
            QueryIntent.FIND_RELATIONS: "Find relationships connected to the entity",
            QueryIntent.GET_NEIGHBORS: "Get entities directly connected to the specified entity",
            QueryIntent.FIND_PATH: "Find a path between two entities",
            QueryIntent.SHORTEST_PATH: "Find the shortest path between two entities",
            QueryIntent.GET_STATISTICS: "Get statistics about the knowledge graph",
            QueryIntent.FIND_SIMILAR: "Find entities similar to the specified entity",
        }

        base_explanation = explanations.get(intent, "Execute a graph query")

        if entities:
            entity_names = [e.get("text", "") for e in entities[:2]]
            if entity_names:
                base_explanation += f" (entities: {', '.join(entity_names)})"

        return base_explanation

    def _get_template_description(self, template: CypherTemplate) -> str:
        """Get description for a template."""
        descriptions = {
            CypherTemplate.FIND_ENTITY_BY_NAME: "Find entities by name search",
            CypherTemplate.FIND_ENTITY_BY_TYPE: "Find entities by type",
            CypherTemplate.GET_ENTITY_BY_ID: "Get entity by ID",
            CypherTemplate.LIST_ENTITIES: "List all entities of a type",
            CypherTemplate.COUNT_ENTITIES: "Count entities",
            CypherTemplate.FIND_RELATIONS: "Find entity relationships",
            CypherTemplate.GET_NEIGHBORS: "Get neighboring entities",
            CypherTemplate.FIND_PATH: "Find path between entities",
            CypherTemplate.SHORTEST_PATH: "Find shortest path",
            CypherTemplate.GET_STATISTICS: "Get graph statistics",
            CypherTemplate.FIND_SIMILAR: "Find similar entities",
        }
        return descriptions.get(template, "Custom query")

    def _check_warnings(self, query: CypherQuery, parsed_query: ParsedQuery) -> List[str]:
        """Check for potential issues and generate warnings."""
        warnings = []

        # Check for potentially expensive queries
        if query.estimated_complexity == "high":
            warnings.append("This query may take longer to execute due to path traversal")

        # Check for missing parameters
        if not parsed_query.entities and not parsed_query.keywords:
            warnings.append("No specific entity or keyword found - results may be broad")

        # Check depth for path queries
        params = parsed_query.parameters
        depth = params.get("depth", 0)
        if depth and depth > 3:
            warnings.append(f"Path depth of {depth} may return many results")

        return warnings

    def generate_custom(
        self,
        query_type: str,
        entity_name: Optional[str] = None,
        entity_type: Optional[str] = None,
        relation_type: Optional[str] = None,
        target_entity: Optional[str] = None,
        limit: Optional[int] = None,
        depth: Optional[int] = None,
    ) -> CypherQuery:
        """
        Generate a custom Cypher query with explicit parameters.

        Args:
            query_type: Type of query (find, list, path, neighbors, etc.)
            entity_name: Name of the entity to search
            entity_type: Type of entity
            relation_type: Type of relation to filter
            target_entity: Target entity for path queries
            limit: Result limit
            depth: Traversal depth

        Returns:
            CypherQuery with generated query
        """
        if not self._initialized:
            self.initialize()

        params = {
            "name": entity_name or "",
            "entity_id": entity_name or "",
            "entity_type": entity_type or "Entity",
            "limit": limit or self.default_limit,
            "offset": 0,
            "depth": min(depth or 2, self.max_depth),
        }

        if target_entity:
            params["target_name"] = target_entity
            params["target_id"] = target_entity
            params["source_name"] = entity_name or ""
            params["source_id"] = entity_name or ""

        # Select template based on query type
        template_map = {
            "find": CypherTemplate.FIND_ENTITY_BY_NAME,
            "list": CypherTemplate.LIST_ENTITIES,
            "count": CypherTemplate.COUNT_ENTITIES,
            "relations": CypherTemplate.FIND_RELATIONS,
            "neighbors": CypherTemplate.GET_NEIGHBORS,
            "path": CypherTemplate.FIND_PATH,
            "shortest_path": CypherTemplate.SHORTEST_PATH,
            "similar": CypherTemplate.FIND_SIMILAR,
            "statistics": CypherTemplate.GET_STATISTICS,
        }

        template = template_map.get(query_type.lower(), CypherTemplate.FIND_ENTITY_BY_NAME)

        # Get query and substitute entity_type (since Neo4j doesn't support parameterized labels)
        query = self.TEMPLATES.get(template, self.TEMPLATES[CypherTemplate.FIND_ENTITY_BY_NAME])
        query = query.replace("$entity_type", params["entity_type"])
        del params["entity_type"]

        # Add relation type filter if specified
        if relation_type and template in [CypherTemplate.FIND_RELATIONS, CypherTemplate.GET_NEIGHBORS]:
            query = query.replace("[r]", f"[r:{relation_type}]")

        return CypherQuery(
            query=query.strip(),
            parameters=params,
            template=template,
            is_read_only=True,
            estimated_complexity="medium" if template in [CypherTemplate.FIND_PATH, CypherTemplate.FIND_SIMILAR] else "low",
            description=self._get_template_description(template),
        )


# Global instance
_cypher_generator: Optional[CypherGenerator] = None


def get_cypher_generator() -> CypherGenerator:
    """Get or create global CypherGenerator instance."""
    global _cypher_generator

    if _cypher_generator is None:
        _cypher_generator = CypherGenerator()
        _cypher_generator.initialize()

    return _cypher_generator
