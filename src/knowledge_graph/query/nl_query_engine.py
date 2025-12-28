"""
Natural Language Query Engine for Knowledge Graph.

Provides natural language understanding for graph queries:
- Query intent classification
- Entity and relation extraction from queries
- Query parameter extraction
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    """Intent types for natural language queries."""
    # Entity queries
    FIND_ENTITY = "find_entity"  # Find entities by name/type
    GET_ENTITY = "get_entity"  # Get specific entity details
    LIST_ENTITIES = "list_entities"  # List all entities of a type
    COUNT_ENTITIES = "count_entities"  # Count entities

    # Relation queries
    FIND_RELATIONS = "find_relations"  # Find relations of an entity
    GET_RELATION = "get_relation"  # Get specific relation
    FIND_CONNECTED = "find_connected"  # Find connected entities

    # Path queries
    FIND_PATH = "find_path"  # Find path between entities
    SHORTEST_PATH = "shortest_path"  # Find shortest path

    # Analytics queries
    GET_NEIGHBORS = "get_neighbors"  # Get neighboring entities
    GET_STATISTICS = "get_statistics"  # Get graph statistics
    FIND_SIMILAR = "find_similar"  # Find similar entities

    # Aggregate queries
    AGGREGATE = "aggregate"  # Aggregation queries
    GROUP_BY = "group_by"  # Group by queries

    # Unknown
    UNKNOWN = "unknown"


class EntityType(str, Enum):
    """Entity types referenced in queries."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    PRODUCT = "product"
    EVENT = "event"
    CONCEPT = "concept"
    DOCUMENT = "document"
    UNKNOWN = "unknown"


@dataclass
class QueryEntity:
    """Entity extracted from a query."""

    text: str = ""
    entity_type: EntityType = EntityType.UNKNOWN
    start_pos: int = 0
    end_pos: int = 0
    confidence: float = 0.0
    normalized: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "entity_type": self.entity_type.value,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "confidence": self.confidence,
            "normalized": self.normalized,
        }


@dataclass
class QueryRelation:
    """Relation extracted from a query."""

    text: str = ""
    relation_type: str = ""
    source_entity: Optional[str] = None
    target_entity: Optional[str] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "relation_type": self.relation_type,
            "source_entity": self.source_entity,
            "target_entity": self.target_entity,
            "confidence": self.confidence,
        }


@dataclass
class QueryParameters:
    """Extracted parameters from a query."""

    limit: Optional[int] = None
    offset: Optional[int] = None
    order_by: Optional[str] = None
    ascending: bool = True
    filters: Dict[str, Any] = field(default_factory=dict)
    depth: Optional[int] = None
    include_properties: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "limit": self.limit,
            "offset": self.offset,
            "order_by": self.order_by,
            "ascending": self.ascending,
            "filters": self.filters,
            "depth": self.depth,
            "include_properties": self.include_properties,
        }


class ParsedQuery(BaseModel):
    """Result of parsing a natural language query."""

    original_query: str = Field(default="", description="Original query text")
    intent: QueryIntent = Field(default=QueryIntent.UNKNOWN, description="Query intent")
    intent_confidence: float = Field(default=0.0, description="Intent confidence")
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted entities")
    relations: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted relations")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Query parameters")
    keywords: List[str] = Field(default_factory=list, description="Important keywords")
    is_valid: bool = Field(default=True, description="Whether query is valid")
    error_message: Optional[str] = Field(None, description="Error message if invalid")


class NLQueryEngine:
    """
    Natural language query engine for knowledge graphs.

    Parses natural language queries and extracts structured information.
    """

    # Intent patterns (Chinese and English)
    INTENT_PATTERNS = {
        QueryIntent.FIND_ENTITY: [
            r"找[到]?(.+)",
            r"查找(.+)",
            r"搜索(.+)",
            r"find\s+(.+)",
            r"search\s+(.+)",
            r"look\s+for\s+(.+)",
        ],
        QueryIntent.GET_ENTITY: [
            r"(.+)是什么",
            r"(.+)的信息",
            r"显示(.+)",
            r"what\s+is\s+(.+)",
            r"show\s+(.+)",
            r"get\s+(.+)",
        ],
        QueryIntent.LIST_ENTITIES: [
            r"列出所有(.+)",
            r"显示所有(.+)",
            r"有哪些(.+)",
            r"list\s+all\s+(.+)",
            r"show\s+all\s+(.+)",
            r"what\s+(.+)\s+are\s+there",
        ],
        QueryIntent.COUNT_ENTITIES: [
            r"有多少(.+)",
            r"(.+)的数量",
            r"统计(.+)",
            r"how\s+many\s+(.+)",
            r"count\s+(.+)",
        ],
        QueryIntent.FIND_RELATIONS: [
            r"(.+)和(.+)的关系",
            r"(.+)与(.+)有什么关系",
            r"(.+)关联的(.+)",
            r"relationship\s+between\s+(.+)\s+and\s+(.+)",
            r"how\s+is\s+(.+)\s+related\s+to\s+(.+)",
        ],
        QueryIntent.FIND_PATH: [
            r"从(.+)到(.+)的路径",
            r"(.+)到(.+)怎么走",
            r"path\s+from\s+(.+)\s+to\s+(.+)",
            r"route\s+from\s+(.+)\s+to\s+(.+)",
        ],
        QueryIntent.SHORTEST_PATH: [
            r"(.+)到(.+)的最短路径",
            r"shortest\s+path\s+from\s+(.+)\s+to\s+(.+)",
        ],
        QueryIntent.GET_NEIGHBORS: [
            r"(.+)的邻居",
            r"(.+)周围的",
            r"(.+)相关的",
            r"neighbors\s+of\s+(.+)",
            r"connected\s+to\s+(.+)",
        ],
        QueryIntent.GET_STATISTICS: [
            r"统计信息",
            r"图谱概况",
            r"数据统计",
            r"statistics",
            r"overview",
            r"summary",
        ],
        QueryIntent.FIND_SIMILAR: [
            r"和(.+)相似的",
            r"类似(.+)的",
            r"similar\s+to\s+(.+)",
            r"like\s+(.+)",
        ],
    }

    # Entity type indicators (Chinese and English)
    ENTITY_TYPE_INDICATORS = {
        EntityType.PERSON: ["人", "员工", "用户", "person", "people", "user", "employee"],
        EntityType.ORGANIZATION: ["公司", "组织", "机构", "企业", "organization", "company", "org"],
        EntityType.LOCATION: ["地点", "位置", "地址", "城市", "location", "place", "city", "address"],
        EntityType.DATE: ["日期", "时间", "date", "time"],
        EntityType.PRODUCT: ["产品", "商品", "服务", "product", "item", "service"],
        EntityType.EVENT: ["事件", "活动", "event", "activity"],
        EntityType.DOCUMENT: ["文档", "文件", "document", "file"],
    }

    # Relation type indicators
    RELATION_INDICATORS = {
        "works_for": ["工作于", "就职于", "属于", "works for", "employed by"],
        "located_in": ["位于", "在", "located in", "in"],
        "part_of": ["属于", "包含于", "part of", "belongs to"],
        "knows": ["认识", "知道", "knows", "know"],
        "created_by": ["创建", "创建者", "created by", "made by"],
        "related_to": ["相关", "关联", "related to", "associated with"],
    }

    # Limit/count patterns
    LIMIT_PATTERNS = [
        r"前(\d+)个",
        r"(\d+)个",
        r"top\s+(\d+)",
        r"first\s+(\d+)",
        r"limit\s+(\d+)",
    ]

    def __init__(
        self,
        default_limit: int = 10,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize NLQueryEngine.

        Args:
            default_limit: Default limit for query results
            confidence_threshold: Minimum confidence for accepting matches
        """
        self.default_limit = default_limit
        self.confidence_threshold = confidence_threshold
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the engine."""
        if self._initialized:
            return
        self._initialized = True
        logger.info("NLQueryEngine initialized")

    def parse(self, query: str) -> ParsedQuery:
        """
        Parse a natural language query.

        Args:
            query: Natural language query string

        Returns:
            ParsedQuery with extracted information
        """
        if not self._initialized:
            self.initialize()

        if not query or not query.strip():
            return ParsedQuery(
                original_query=query,
                is_valid=False,
                error_message="Empty query",
            )

        query = query.strip()

        # Classify intent
        intent, intent_confidence, intent_matches = self._classify_intent(query)

        # Extract entities
        entities = self._extract_entities(query, intent_matches)

        # Extract relations
        relations = self._extract_relations(query)

        # Extract parameters
        parameters = self._extract_parameters(query)

        # Extract keywords
        keywords = self._extract_keywords(query)

        return ParsedQuery(
            original_query=query,
            intent=intent,
            intent_confidence=intent_confidence,
            entities=[e.to_dict() for e in entities],
            relations=[r.to_dict() for r in relations],
            parameters=parameters.to_dict(),
            keywords=keywords,
            is_valid=True,
        )

    def _classify_intent(self, query: str) -> Tuple[QueryIntent, float, List[str]]:
        """Classify the query intent."""
        query_lower = query.lower()

        best_intent = QueryIntent.UNKNOWN
        best_confidence = 0.0
        best_matches = []

        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                try:
                    match = re.search(pattern, query_lower, re.IGNORECASE)
                    if match:
                        # Calculate confidence based on match quality
                        match_len = len(match.group(0))
                        query_len = len(query)
                        confidence = min(1.0, match_len / query_len + 0.3)

                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_intent = intent
                            best_matches = list(match.groups())
                except re.error:
                    continue

        # Default to FIND_ENTITY if no strong match
        if best_confidence < self.confidence_threshold:
            best_intent = QueryIntent.FIND_ENTITY
            best_confidence = 0.5
            best_matches = [query]

        return best_intent, best_confidence, best_matches

    def _extract_entities(self, query: str, intent_matches: List[str]) -> List[QueryEntity]:
        """Extract entities from the query."""
        entities = []

        # Extract from intent matches
        for match in intent_matches:
            if match and len(match) > 1:
                entity_type = self._infer_entity_type(match)
                start = query.find(match)
                entities.append(QueryEntity(
                    text=match,
                    entity_type=entity_type,
                    start_pos=start,
                    end_pos=start + len(match) if start >= 0 else 0,
                    confidence=0.8,
                    normalized=match.strip(),
                ))

        # Extract quoted strings as entities
        quoted_pattern = r'["\']([^"\']+)["\']'
        for match in re.finditer(quoted_pattern, query):
            text = match.group(1)
            if text and len(text) > 1:
                entity_type = self._infer_entity_type(text)
                entities.append(QueryEntity(
                    text=text,
                    entity_type=entity_type,
                    start_pos=match.start(1),
                    end_pos=match.end(1),
                    confidence=0.9,
                    normalized=text.strip(),
                ))

        # Extract proper nouns (capitalized words in English, or Chinese names)
        proper_noun_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        for match in re.finditer(proper_noun_pattern, query):
            text = match.group(1)
            # Avoid common words
            if text.lower() not in {"the", "a", "an", "is", "are", "what", "who", "where", "when"}:
                entity_type = self._infer_entity_type(text)
                entities.append(QueryEntity(
                    text=text,
                    entity_type=entity_type,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.7,
                    normalized=text.strip(),
                ))

        # Deduplicate
        seen = set()
        unique_entities = []
        for entity in entities:
            key = (entity.text.lower(), entity.entity_type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)

        return unique_entities

    def _infer_entity_type(self, text: str) -> EntityType:
        """Infer entity type from text."""
        text_lower = text.lower()

        for entity_type, indicators in self.ENTITY_TYPE_INDICATORS.items():
            for indicator in indicators:
                if indicator in text_lower:
                    return entity_type

        # Check for patterns
        if re.match(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', text):
            return EntityType.DATE

        return EntityType.UNKNOWN

    def _extract_relations(self, query: str) -> List[QueryRelation]:
        """Extract relations from the query."""
        relations = []
        query_lower = query.lower()

        for relation_type, indicators in self.RELATION_INDICATORS.items():
            for indicator in indicators:
                if indicator in query_lower:
                    # Find surrounding context
                    idx = query_lower.find(indicator)
                    relations.append(QueryRelation(
                        text=indicator,
                        relation_type=relation_type,
                        confidence=0.7,
                    ))
                    break

        return relations

    def _extract_parameters(self, query: str) -> QueryParameters:
        """Extract query parameters."""
        params = QueryParameters()

        # Extract limit
        for pattern in self.LIMIT_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    params.limit = int(match.group(1))
                except (ValueError, IndexError):
                    pass
                break

        if params.limit is None:
            params.limit = self.default_limit

        # Extract depth for path queries
        depth_patterns = [r"深度(\d+)", r"(\d+)层", r"depth\s+(\d+)", r"(\d+)\s+hops?"]
        for pattern in depth_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    params.depth = int(match.group(1))
                except (ValueError, IndexError):
                    pass
                break

        # Extract ordering
        if any(kw in query.lower() for kw in ["降序", "从大到小", "descending", "desc"]):
            params.ascending = False
        elif any(kw in query.lower() for kw in ["升序", "从小到大", "ascending", "asc"]):
            params.ascending = True

        return params

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query."""
        # Remove common stop words
        stop_words = {
            "的", "是", "在", "有", "和", "与", "或", "从", "到", "了", "吗", "呢",
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once", "here",
            "there", "when", "where", "why", "how", "all", "each", "every",
            "both", "few", "more", "most", "other", "some", "such", "no",
            "nor", "not", "only", "own", "same", "so", "than", "too", "very",
            "what", "which", "who", "whom", "this", "that", "these", "those",
            "am", "it", "its", "itself", "they", "them", "their", "theirs",
        }

        # Tokenize
        words = re.findall(r'\b[\w\u4e00-\u9fff]+\b', query.lower())

        # Filter stop words and short words
        keywords = [w for w in words if w not in stop_words and len(w) > 1]

        return keywords[:10]  # Limit to top 10 keywords

    def get_query_suggestions(self, partial_query: str) -> List[str]:
        """
        Get query suggestions based on partial input.

        Args:
            partial_query: Partial query string

        Returns:
            List of suggested completions
        """
        if not self._initialized:
            self.initialize()

        suggestions = []

        # Common query templates
        templates = [
            "找到所有{entity_type}",
            "查找与{entity}相关的内容",
            "{entity}是什么",
            "{entity}和{entity}的关系",
            "列出所有{entity_type}",
            "有多少{entity_type}",
            "从{entity}到{entity}的路径",
        ]

        partial_lower = partial_query.lower()

        for template in templates:
            if template.startswith(partial_lower[:2]) or any(
                kw in partial_lower for kw in ["找", "查", "列", "有", "从", "find", "search", "list"]
            ):
                suggestions.append(template)

        return suggestions[:5]


# Global instance
_nl_query_engine: Optional[NLQueryEngine] = None


def get_nl_query_engine() -> NLQueryEngine:
    """Get or create global NLQueryEngine instance."""
    global _nl_query_engine

    if _nl_query_engine is None:
        _nl_query_engine = NLQueryEngine()
        _nl_query_engine.initialize()

    return _nl_query_engine
