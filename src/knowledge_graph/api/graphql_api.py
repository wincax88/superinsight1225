"""
Knowledge Graph GraphQL API.

Task 13: GraphQL API implementation for Knowledge Graph system.
Provides GraphQL schema and resolvers for entity and relation queries.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID
from enum import Enum

try:
    import strawberry
    from strawberry.fastapi import GraphQLRouter
    from strawberry.types import Info
    STRAWBERRY_AVAILABLE = True
except ImportError:
    STRAWBERRY_AVAILABLE = False
    strawberry = None

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# =============================================================================
# GraphQL Types (if strawberry is available)
# =============================================================================

if STRAWBERRY_AVAILABLE:

    @strawberry.enum
    class EntityTypeGQL(Enum):
        """Entity types in GraphQL."""
        PERSON = "PERSON"
        ORGANIZATION = "ORGANIZATION"
        LOCATION = "LOCATION"
        EVENT = "EVENT"
        PRODUCT = "PRODUCT"
        CONCEPT = "CONCEPT"
        DOCUMENT = "DOCUMENT"
        DATE = "DATE"
        QUANTITY = "QUANTITY"
        TECHNOLOGY = "TECHNOLOGY"
        PROCESS = "PROCESS"
        TASK = "TASK"
        PROJECT = "PROJECT"
        SKILL = "SKILL"
        DOMAIN = "DOMAIN"
        OTHER = "OTHER"

    @strawberry.enum
    class RelationTypeGQL(Enum):
        """Relation types in GraphQL."""
        WORKS_FOR = "WORKS_FOR"
        LOCATED_IN = "LOCATED_IN"
        PART_OF = "PART_OF"
        RELATED_TO = "RELATED_TO"
        CREATED_BY = "CREATED_BY"
        OWNS = "OWNS"
        MANAGES = "MANAGES"
        REPORTS_TO = "REPORTS_TO"
        COLLABORATES_WITH = "COLLABORATES_WITH"
        DEPENDS_ON = "DEPENDS_ON"
        PRECEDES = "PRECEDES"
        FOLLOWS = "FOLLOWS"
        SIMILAR_TO = "SIMILAR_TO"
        DERIVED_FROM = "DERIVED_FROM"
        INSTANCE_OF = "INSTANCE_OF"
        SUBCLASS_OF = "SUBCLASS_OF"
        HAS_PROPERTY = "HAS_PROPERTY"
        MENTIONS = "MENTIONS"
        AUTHORED_BY = "AUTHORED_BY"
        PARTICIPATED_IN = "PARTICIPATED_IN"
        USES = "USES"
        PRODUCES = "PRODUCES"
        CONSUMES = "CONSUMES"

    @strawberry.type
    class PropertyValue:
        """Key-value property."""
        key: str
        value: str

    @strawberry.type
    class Entity:
        """Entity in the knowledge graph."""
        id: strawberry.ID
        entity_type: EntityTypeGQL
        name: str
        description: Optional[str] = None
        properties: List[PropertyValue]
        aliases: List[str]
        confidence: float
        source: Optional[str] = None
        tenant_id: Optional[str] = None
        created_at: datetime
        updated_at: datetime
        is_active: bool

        @strawberry.field
        async def relations(
            self,
            info: Info,
            direction: Optional[str] = "both",
            relation_types: Optional[List[RelationTypeGQL]] = None,
            limit: int = 100,
        ) -> List["Relation"]:
            """Get relations for this entity."""
            context = info.context
            db = context.get("db")
            if not db:
                return []

            try:
                types = [rt.value for rt in relation_types] if relation_types else None
                relations = await db.get_entity_relations(
                    entity_id=UUID(self.id),
                    direction=direction,
                    relation_types=types,
                    limit=limit,
                )
                return [_relation_to_gql(r) for r in relations]
            except Exception as e:
                logger.error(f"Failed to get relations: {e}")
                return []

        @strawberry.field
        async def neighbors(
            self,
            info: Info,
            depth: int = 1,
            limit: int = 50,
        ) -> List["Entity"]:
            """Get neighboring entities."""
            context = info.context
            db = context.get("db")
            if not db:
                return []

            try:
                result = await db.get_neighbors(
                    entity_id=UUID(self.id),
                    depth=depth,
                    limit=limit,
                )
                return [_entity_to_gql(e) for e in result.get("entities", [])]
            except Exception as e:
                logger.error(f"Failed to get neighbors: {e}")
                return []

    @strawberry.type
    class Relation:
        """Relation between entities in the knowledge graph."""
        id: strawberry.ID
        source_id: strawberry.ID
        target_id: strawberry.ID
        relation_type: RelationTypeGQL
        properties: List[PropertyValue]
        weight: float
        confidence: float
        evidence: Optional[str] = None
        tenant_id: Optional[str] = None
        created_at: datetime
        is_active: bool

        @strawberry.field
        async def source(self, info: Info) -> Optional[Entity]:
            """Get source entity."""
            context = info.context
            db = context.get("db")
            if not db:
                return None

            try:
                entity = await db.get_entity(UUID(self.source_id))
                return _entity_to_gql(entity) if entity else None
            except Exception as e:
                logger.error(f"Failed to get source entity: {e}")
                return None

        @strawberry.field
        async def target(self, info: Info) -> Optional[Entity]:
            """Get target entity."""
            context = info.context
            db = context.get("db")
            if not db:
                return None

            try:
                entity = await db.get_entity(UUID(self.target_id))
                return _entity_to_gql(entity) if entity else None
            except Exception as e:
                logger.error(f"Failed to get target entity: {e}")
                return None

    @strawberry.type
    class Path:
        """Path between two entities."""
        found: bool
        length: int
        entities: List[Entity]
        relations: List[Relation]

    @strawberry.type
    class GraphStatistics:
        """Knowledge graph statistics."""
        total_entities: int
        total_relations: int
        entities_by_type: List[PropertyValue]
        relations_by_type: List[PropertyValue]
        graph_density: float
        average_degree: float
        last_updated: datetime

    @strawberry.type
    class ExtractionResult:
        """Result of text extraction."""
        entities: List[Entity]
        relations: List[Relation]
        processing_time_ms: float

    @strawberry.type
    class EntityConnection:
        """Paginated entity list."""
        entities: List[Entity]
        total_count: int
        has_next_page: bool
        end_cursor: Optional[str] = None

    @strawberry.type
    class RelationConnection:
        """Paginated relation list."""
        relations: List[Relation]
        total_count: int
        has_next_page: bool
        end_cursor: Optional[str] = None

    # =========================================================================
    # Input Types
    # =========================================================================

    @strawberry.input
    class EntityInput:
        """Input for creating an entity."""
        name: str
        entity_type: EntityTypeGQL
        description: Optional[str] = None
        properties: Optional[List[PropertyValue]] = None
        aliases: Optional[List[str]] = None
        confidence: float = 1.0
        source: Optional[str] = None

    @strawberry.input
    class EntityUpdateInput:
        """Input for updating an entity."""
        name: Optional[str] = None
        description: Optional[str] = None
        properties: Optional[List[PropertyValue]] = None
        aliases: Optional[List[str]] = None

    @strawberry.input
    class RelationInput:
        """Input for creating a relation."""
        source_id: strawberry.ID
        target_id: strawberry.ID
        relation_type: RelationTypeGQL
        properties: Optional[List[PropertyValue]] = None
        weight: float = 1.0
        confidence: float = 1.0
        evidence: Optional[str] = None

    @strawberry.input
    class TextExtractionInput:
        """Input for text extraction."""
        text: str
        extract_entities: bool = True
        extract_relations: bool = True
        entity_types: Optional[List[EntityTypeGQL]] = None
        min_confidence: float = 0.5
        save_to_graph: bool = False

    # =========================================================================
    # Query Type
    # =========================================================================

    @strawberry.type
    class Query:
        """Root query type."""

        @strawberry.field
        async def entity(
            self,
            info: Info,
            id: strawberry.ID,
        ) -> Optional[Entity]:
            """Get an entity by ID."""
            db = info.context.get("db")
            if not db:
                return None

            try:
                entity = await db.get_entity(UUID(id))
                return _entity_to_gql(entity) if entity else None
            except Exception as e:
                logger.error(f"Failed to get entity: {e}")
                return None

        @strawberry.field
        async def entities(
            self,
            info: Info,
            query: Optional[str] = None,
            entity_type: Optional[EntityTypeGQL] = None,
            tenant_id: Optional[str] = None,
            limit: int = 100,
            offset: int = 0,
        ) -> EntityConnection:
            """Search for entities."""
            db = info.context.get("db")
            if not db:
                return EntityConnection(
                    entities=[],
                    total_count=0,
                    has_next_page=False,
                )

            try:
                type_filter = entity_type.value if entity_type else None
                entities = await db.search_entities(
                    query_text=query,
                    entity_type=type_filter,
                    tenant_id=tenant_id,
                    limit=limit + 1,  # Fetch one extra to check for next page
                    offset=offset,
                )

                has_next = len(entities) > limit
                if has_next:
                    entities = entities[:limit]

                return EntityConnection(
                    entities=[_entity_to_gql(e) for e in entities],
                    total_count=len(entities),
                    has_next_page=has_next,
                    end_cursor=str(offset + len(entities)) if entities else None,
                )
            except Exception as e:
                logger.error(f"Failed to search entities: {e}")
                return EntityConnection(
                    entities=[],
                    total_count=0,
                    has_next_page=False,
                )

        @strawberry.field
        async def relation(
            self,
            info: Info,
            id: strawberry.ID,
        ) -> Optional[Relation]:
            """Get a relation by ID."""
            db = info.context.get("db")
            if not db:
                return None

            try:
                relation = await db.get_relation(UUID(id))
                return _relation_to_gql(relation) if relation else None
            except Exception as e:
                logger.error(f"Failed to get relation: {e}")
                return None

        @strawberry.field
        async def path(
            self,
            info: Info,
            source_id: strawberry.ID,
            target_id: strawberry.ID,
            max_depth: int = 5,
        ) -> Path:
            """Find shortest path between two entities."""
            db = info.context.get("db")
            if not db:
                return Path(
                    found=False,
                    length=0,
                    entities=[],
                    relations=[],
                )

            try:
                result = await db.find_path(
                    source_id=UUID(source_id),
                    target_id=UUID(target_id),
                    max_depth=max_depth,
                )

                return Path(
                    found=result.get("found", False),
                    length=result.get("length", 0),
                    entities=[_entity_to_gql(e) for e in result.get("entities", [])],
                    relations=[_relation_to_gql(r) for r in result.get("relations", [])],
                )
            except Exception as e:
                logger.error(f"Failed to find path: {e}")
                return Path(
                    found=False,
                    length=0,
                    entities=[],
                    relations=[],
                )

        @strawberry.field
        async def statistics(
            self,
            info: Info,
            tenant_id: Optional[str] = None,
        ) -> GraphStatistics:
            """Get graph statistics."""
            db = info.context.get("db")
            if not db:
                return GraphStatistics(
                    total_entities=0,
                    total_relations=0,
                    entities_by_type=[],
                    relations_by_type=[],
                    graph_density=0.0,
                    average_degree=0.0,
                    last_updated=datetime.utcnow(),
                )

            try:
                stats = await db.get_statistics(tenant_id)
                return GraphStatistics(
                    total_entities=stats.entity_count,
                    total_relations=stats.relation_count,
                    entities_by_type=[
                        PropertyValue(key=k, value=str(v))
                        for k, v in stats.entity_type_counts.items()
                    ],
                    relations_by_type=[
                        PropertyValue(key=k, value=str(v))
                        for k, v in stats.relation_type_counts.items()
                    ],
                    graph_density=stats.density,
                    average_degree=stats.average_degree,
                    last_updated=stats.last_updated,
                )
            except Exception as e:
                logger.error(f"Failed to get statistics: {e}")
                return GraphStatistics(
                    total_entities=0,
                    total_relations=0,
                    entities_by_type=[],
                    relations_by_type=[],
                    graph_density=0.0,
                    average_degree=0.0,
                    last_updated=datetime.utcnow(),
                )

    # =========================================================================
    # Mutation Type
    # =========================================================================

    @strawberry.type
    class Mutation:
        """Root mutation type."""

        @strawberry.mutation
        async def create_entity(
            self,
            info: Info,
            input: EntityInput,
        ) -> Entity:
            """Create a new entity."""
            db = info.context.get("db")
            if not db:
                raise ValueError("Database not available")

            from ..core.models import Entity as EntityModel, EntityType

            entity = EntityModel(
                entity_type=EntityType(input.entity_type.value),
                name=input.name,
                description=input.description,
                properties={p.key: p.value for p in input.properties} if input.properties else {},
                aliases=input.aliases or [],
                confidence=input.confidence,
                source=input.source,
            )

            created = await db.create_entity(entity)
            return _entity_to_gql(created)

        @strawberry.mutation
        async def update_entity(
            self,
            info: Info,
            id: strawberry.ID,
            input: EntityUpdateInput,
        ) -> Optional[Entity]:
            """Update an entity."""
            db = info.context.get("db")
            if not db:
                raise ValueError("Database not available")

            updates = {}
            if input.name is not None:
                updates["name"] = input.name
            if input.description is not None:
                updates["description"] = input.description
            if input.properties is not None:
                updates["properties"] = {p.key: p.value for p in input.properties}
            if input.aliases is not None:
                updates["aliases"] = input.aliases

            updated = await db.update_entity(UUID(id), updates)
            return _entity_to_gql(updated) if updated else None

        @strawberry.mutation
        async def delete_entity(
            self,
            info: Info,
            id: strawberry.ID,
            hard_delete: bool = False,
        ) -> bool:
            """Delete an entity."""
            db = info.context.get("db")
            if not db:
                raise ValueError("Database not available")

            return await db.delete_entity(UUID(id), hard_delete)

        @strawberry.mutation
        async def create_relation(
            self,
            info: Info,
            input: RelationInput,
        ) -> Relation:
            """Create a new relation."""
            db = info.context.get("db")
            if not db:
                raise ValueError("Database not available")

            from ..core.models import Relation as RelationModel, RelationType

            relation = RelationModel(
                source_id=UUID(input.source_id),
                target_id=UUID(input.target_id),
                relation_type=RelationType(input.relation_type.value),
                properties={p.key: p.value for p in input.properties} if input.properties else {},
                weight=input.weight,
                confidence=input.confidence,
                evidence=input.evidence,
            )

            created = await db.create_relation(relation)
            return _relation_to_gql(created)

        @strawberry.mutation
        async def delete_relation(
            self,
            info: Info,
            id: strawberry.ID,
            hard_delete: bool = False,
        ) -> bool:
            """Delete a relation."""
            db = info.context.get("db")
            if not db:
                raise ValueError("Database not available")

            return await db.delete_relation(UUID(id), hard_delete)

        @strawberry.mutation
        async def extract_from_text(
            self,
            info: Info,
            input: TextExtractionInput,
        ) -> ExtractionResult:
            """Extract entities and relations from text."""
            from ..nlp.entity_extractor import get_entity_extractor
            from ..nlp.relation_extractor import get_relation_extractor

            start_time = datetime.utcnow()

            entities = []
            relations = []

            if input.extract_entities:
                entity_extractor = get_entity_extractor()
                entity_types = [et.value for et in input.entity_types] if input.entity_types else None
                extracted_entities = entity_extractor.extract(
                    input.text,
                    entity_types=entity_types,
                )
                entities = [
                    e for e in extracted_entities
                    if e.confidence >= input.min_confidence
                ]

            if input.extract_relations and entities:
                relation_extractor = get_relation_extractor()
                extracted_relations = relation_extractor.extract(
                    input.text,
                    entities=entities,
                )
                relations = [
                    r for r in extracted_relations
                    if r.confidence >= input.min_confidence
                ]

            # Save to graph if requested
            if input.save_to_graph:
                db = info.context.get("db")
                if db:
                    entity_map = {}
                    for extracted in entities:
                        entity = extracted.to_entity()
                        try:
                            created = await db.create_entity(entity)
                            entity_map[extracted.text] = created.id
                        except Exception as e:
                            logger.warning(f"Failed to save entity: {e}")

                    for extracted in relations:
                        source_id = entity_map.get(extracted.source_entity.text)
                        target_id = entity_map.get(extracted.target_entity.text)
                        if source_id and target_id:
                            relation = extracted.to_relation(source_id, target_id)
                            try:
                                await db.create_relation(relation)
                            except Exception as e:
                                logger.warning(f"Failed to save relation: {e}")

            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            return ExtractionResult(
                entities=[_extracted_entity_to_gql(e) for e in entities],
                relations=[_extracted_relation_to_gql(r) for r in relations],
                processing_time_ms=processing_time,
            )

    # =========================================================================
    # Subscription Type
    # =========================================================================

    @strawberry.type
    class Subscription:
        """Root subscription type for real-time updates."""

        @strawberry.subscription
        async def entity_created(
            self,
            info: Info,
            entity_type: Optional[EntityTypeGQL] = None,
        ) -> Entity:
            """Subscribe to entity creation events."""
            from .websocket_api import get_connection_manager
            import asyncio

            manager = get_connection_manager()
            queue = asyncio.Queue()

            # This is a simplified implementation
            # In production, you'd integrate with the WebSocket manager
            while True:
                await asyncio.sleep(1)
                # Placeholder - would receive from event queue

        @strawberry.subscription
        async def entity_updated(
            self,
            info: Info,
            entity_id: Optional[strawberry.ID] = None,
        ) -> Entity:
            """Subscribe to entity update events."""
            import asyncio
            while True:
                await asyncio.sleep(1)
                # Placeholder - would receive from event queue

    # =========================================================================
    # Helper Functions
    # =========================================================================

    def _entity_to_gql(entity) -> Entity:
        """Convert domain entity to GraphQL entity."""
        return Entity(
            id=strawberry.ID(str(entity.id)),
            entity_type=EntityTypeGQL(entity.entity_type.value),
            name=entity.name,
            description=entity.description,
            properties=[
                PropertyValue(key=k, value=str(v))
                for k, v in entity.properties.items()
            ],
            aliases=entity.aliases or [],
            confidence=entity.confidence,
            source=entity.source,
            tenant_id=entity.tenant_id,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            is_active=entity.is_active,
        )

    def _relation_to_gql(relation) -> Relation:
        """Convert domain relation to GraphQL relation."""
        return Relation(
            id=strawberry.ID(str(relation.id)),
            source_id=strawberry.ID(str(relation.source_id)),
            target_id=strawberry.ID(str(relation.target_id)),
            relation_type=RelationTypeGQL(relation.relation_type.value),
            properties=[
                PropertyValue(key=k, value=str(v))
                for k, v in relation.properties.items()
            ],
            weight=relation.weight,
            confidence=relation.confidence,
            evidence=relation.evidence,
            tenant_id=relation.tenant_id,
            created_at=relation.created_at,
            is_active=relation.is_active,
        )

    def _extracted_entity_to_gql(extracted) -> Entity:
        """Convert extracted entity to GraphQL entity."""
        return Entity(
            id=strawberry.ID(str(extracted.id) if hasattr(extracted, 'id') else "temp"),
            entity_type=EntityTypeGQL(extracted.entity_type.value if hasattr(extracted.entity_type, 'value') else str(extracted.entity_type)),
            name=extracted.text,
            description=None,
            properties=[],
            aliases=[],
            confidence=extracted.confidence,
            source="extraction",
            tenant_id=None,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            is_active=True,
        )

    def _extracted_relation_to_gql(extracted) -> Relation:
        """Convert extracted relation to GraphQL relation."""
        return Relation(
            id=strawberry.ID("temp"),
            source_id=strawberry.ID("temp"),
            target_id=strawberry.ID("temp"),
            relation_type=RelationTypeGQL(extracted.relation_type.value if hasattr(extracted.relation_type, 'value') else str(extracted.relation_type)),
            properties=[],
            weight=1.0,
            confidence=extracted.confidence,
            evidence=extracted.evidence if hasattr(extracted, 'evidence') else None,
            tenant_id=None,
            created_at=datetime.utcnow(),
            is_active=True,
        )

    # =========================================================================
    # Schema Creation
    # =========================================================================

    schema = strawberry.Schema(
        query=Query,
        mutation=Mutation,
        subscription=Subscription,
    )

    def create_graphql_router(get_db_func) -> GraphQLRouter:
        """Create a GraphQL router with context."""

        async def get_context(request=None):
            """Build context for GraphQL resolvers."""
            db = None
            if get_db_func:
                try:
                    db = await get_db_func()
                except Exception as e:
                    logger.error(f"Failed to get database: {e}")

            return {
                "db": db,
                "request": request,
            }

        return GraphQLRouter(
            schema,
            context_getter=get_context,
            path="/graphql",
        )

else:
    # Strawberry not available - provide stub
    schema = None

    def create_graphql_router(get_db_func):
        """GraphQL is not available."""
        raise ImportError(
            "strawberry-graphql is not installed. "
            "Install it with: pip install strawberry-graphql"
        )


# =============================================================================
# Public API
# =============================================================================

def is_graphql_available() -> bool:
    """Check if GraphQL is available."""
    return STRAWBERRY_AVAILABLE


def get_graphql_router(get_db_func=None):
    """
    Get the GraphQL router for integration with FastAPI.

    Usage:
        from fastapi import FastAPI
        from src.knowledge_graph.api.graphql_api import get_graphql_router

        app = FastAPI()
        graphql_router = get_graphql_router(get_db)
        app.include_router(graphql_router)
    """
    if not STRAWBERRY_AVAILABLE:
        raise ImportError(
            "strawberry-graphql is not installed. "
            "Install it with: pip install strawberry-graphql"
        )

    return create_graphql_router(get_db_func)
