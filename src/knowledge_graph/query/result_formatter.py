"""
Result Formatter for Knowledge Graph Queries.

Formats and presents query results:
- Result formatting for different output types
- Natural language explanation generation
- Related content recommendations
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class OutputFormat(str, Enum):
    """Output format types."""
    JSON = "json"
    TABLE = "table"
    GRAPH = "graph"
    TEXT = "text"
    MARKDOWN = "markdown"
    CSV = "csv"


class ResultType(str, Enum):
    """Types of query results."""
    ENTITIES = "entities"
    RELATIONS = "relations"
    PATHS = "paths"
    STATISTICS = "statistics"
    COUNT = "count"
    MIXED = "mixed"
    ERROR = "error"


@dataclass
class FormattedEntity:
    """Formatted entity for display."""

    id: str = ""
    name: str = ""
    entity_type: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    display_name: str = ""
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "properties": self.properties,
            "display_name": self.display_name,
            "summary": self.summary,
        }


@dataclass
class FormattedRelation:
    """Formatted relation for display."""

    id: str = ""
    source_id: str = ""
    source_name: str = ""
    target_id: str = ""
    target_name: str = ""
    relation_type: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    display_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "target_id": self.target_id,
            "target_name": self.target_name,
            "relation_type": self.relation_type,
            "properties": self.properties,
            "display_text": self.display_text,
        }


@dataclass
class FormattedPath:
    """Formatted path for display."""

    path_id: str = ""
    nodes: List[FormattedEntity] = field(default_factory=list)
    relationships: List[FormattedRelation] = field(default_factory=list)
    length: int = 0
    display_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path_id": self.path_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "relationships": [r.to_dict() for r in self.relationships],
            "length": self.length,
            "display_text": self.display_text,
        }


@dataclass
class Recommendation:
    """Related content recommendation."""

    item_id: str = ""
    item_type: str = ""
    title: str = ""
    reason: str = ""
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "item_type": self.item_type,
            "title": self.title,
            "reason": self.reason,
            "score": self.score,
        }


class FormattedResult(BaseModel):
    """Formatted query result."""

    result_type: ResultType = Field(default=ResultType.MIXED, description="Type of result")
    output_format: OutputFormat = Field(default=OutputFormat.JSON, description="Output format")
    total_count: int = Field(default=0, description="Total number of results")
    returned_count: int = Field(default=0, description="Number of returned results")
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="Formatted entities")
    relations: List[Dict[str, Any]] = Field(default_factory=list, description="Formatted relations")
    paths: List[Dict[str, Any]] = Field(default_factory=list, description="Formatted paths")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Statistics")
    explanation: str = Field(default="", description="Natural language explanation")
    recommendations: List[Dict[str, Any]] = Field(default_factory=list, description="Recommendations")
    formatted_output: str = Field(default="", description="Formatted string output")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Result metadata")
    query_time_ms: float = Field(default=0.0, description="Query execution time")


class ResultFormatter:
    """
    Result formatter for knowledge graph queries.

    Formats raw query results into human-readable output.
    """

    # Relation type display names (Chinese and English)
    RELATION_DISPLAY_NAMES = {
        "works_for": ("就职于", "works for"),
        "located_in": ("位于", "located in"),
        "part_of": ("属于", "part of"),
        "knows": ("认识", "knows"),
        "created_by": ("创建者", "created by"),
        "related_to": ("相关于", "related to"),
        "has_part": ("包含", "has part"),
        "instance_of": ("实例类型", "instance of"),
        "subclass_of": ("子类", "subclass of"),
        "before": ("早于", "before"),
        "after": ("晚于", "after"),
        "during": ("期间", "during"),
    }

    # Entity type display names
    ENTITY_TYPE_DISPLAY_NAMES = {
        "person": ("人物", "Person"),
        "organization": ("组织", "Organization"),
        "location": ("地点", "Location"),
        "date": ("日期", "Date"),
        "product": ("产品", "Product"),
        "event": ("事件", "Event"),
        "concept": ("概念", "Concept"),
        "document": ("文档", "Document"),
        "task": ("任务", "Task"),
        "annotation": ("标注", "Annotation"),
    }

    def __init__(
        self,
        default_format: OutputFormat = OutputFormat.JSON,
        language: str = "zh",  # "zh" or "en"
        max_text_length: int = 200,
        include_recommendations: bool = True,
    ):
        """
        Initialize ResultFormatter.

        Args:
            default_format: Default output format
            language: Language for display names ("zh" or "en")
            max_text_length: Maximum text length for summaries
            include_recommendations: Whether to include recommendations
        """
        self.default_format = default_format
        self.language = language
        self.max_text_length = max_text_length
        self.include_recommendations = include_recommendations
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the formatter."""
        if self._initialized:
            return
        self._initialized = True
        logger.info("ResultFormatter initialized")

    def format(
        self,
        raw_results: List[Dict[str, Any]],
        output_format: Optional[OutputFormat] = None,
        query_context: Optional[Dict[str, Any]] = None,
    ) -> FormattedResult:
        """
        Format raw query results.

        Args:
            raw_results: Raw results from Neo4j query
            output_format: Desired output format
            query_context: Context from the original query

        Returns:
            FormattedResult with formatted output
        """
        if not self._initialized:
            self.initialize()

        output_format = output_format or self.default_format
        query_context = query_context or {}

        # Detect result type
        result_type = self._detect_result_type(raw_results)

        # Format based on result type
        formatted_entities = []
        formatted_relations = []
        formatted_paths = []
        statistics = {}

        if result_type == ResultType.ENTITIES:
            formatted_entities = self._format_entities(raw_results)
        elif result_type == ResultType.RELATIONS:
            formatted_relations = self._format_relations(raw_results)
        elif result_type == ResultType.PATHS:
            formatted_paths = self._format_paths(raw_results)
        elif result_type == ResultType.STATISTICS:
            statistics = self._format_statistics(raw_results)
        elif result_type == ResultType.COUNT:
            statistics = {"count": raw_results[0].get("count", 0) if raw_results else 0}
        else:
            # Mixed results
            formatted_entities, formatted_relations = self._format_mixed(raw_results)

        # Generate explanation
        explanation = self._generate_explanation(
            result_type, formatted_entities, formatted_relations, formatted_paths, statistics, query_context
        )

        # Generate recommendations
        recommendations = []
        if self.include_recommendations:
            recommendations = self._generate_recommendations(
                formatted_entities, formatted_relations, query_context
            )

        # Generate formatted string output
        formatted_output = self._generate_formatted_output(
            output_format, formatted_entities, formatted_relations, formatted_paths, statistics
        )

        return FormattedResult(
            result_type=result_type,
            output_format=output_format,
            total_count=len(raw_results),
            returned_count=len(formatted_entities) + len(formatted_relations) + len(formatted_paths),
            entities=[e.to_dict() for e in formatted_entities],
            relations=[r.to_dict() for r in formatted_relations],
            paths=[p.to_dict() for p in formatted_paths],
            statistics=statistics,
            explanation=explanation,
            recommendations=[r.to_dict() for r in recommendations],
            formatted_output=formatted_output,
            metadata={
                "language": self.language,
                "format": output_format.value,
            },
        )

    def _detect_result_type(self, raw_results: List[Dict[str, Any]]) -> ResultType:
        """Detect the type of results."""
        if not raw_results:
            return ResultType.ENTITIES

        first_result = raw_results[0]

        # Check for count
        if "count" in first_result and len(first_result) == 1:
            return ResultType.COUNT

        # Check for statistics (label, count pairs)
        if "label" in first_result and "count" in first_result:
            return ResultType.STATISTICS

        # Check for paths
        if "path" in first_result or "pathLength" in first_result:
            return ResultType.PATHS

        # Check for relations
        has_relation = any("r" in r or "relationship" in str(r).lower() for r in raw_results)
        has_nodes = any("n" in r or "a" in r or "b" in r for r in raw_results)

        if has_relation and has_nodes:
            return ResultType.MIXED

        if has_relation:
            return ResultType.RELATIONS

        return ResultType.ENTITIES

    def _format_entities(self, raw_results: List[Dict[str, Any]]) -> List[FormattedEntity]:
        """Format entity results."""
        entities = []

        for result in raw_results:
            # Extract node data
            node_data = result.get("n") or result.get("neighbor") or result

            if isinstance(node_data, dict):
                entity_type = node_data.get("entity_type", node_data.get("type", "unknown"))
                name = node_data.get("name", "Unknown")

                # Create display name
                type_display = self._get_entity_type_display(entity_type)
                display_name = f"{name} ({type_display})"

                # Create summary
                description = node_data.get("description", "")
                summary = description[:self.max_text_length] + "..." if len(description) > self.max_text_length else description

                entities.append(FormattedEntity(
                    id=str(node_data.get("id", "")),
                    name=name,
                    entity_type=entity_type,
                    properties={k: v for k, v in node_data.items()
                               if k not in {"id", "name", "entity_type", "type"}},
                    display_name=display_name,
                    summary=summary or f"{type_display}: {name}",
                ))

        return entities

    def _format_relations(self, raw_results: List[Dict[str, Any]]) -> List[FormattedRelation]:
        """Format relation results."""
        relations = []

        for result in raw_results:
            rel_data = result.get("r") or result

            if isinstance(rel_data, dict):
                relation_type = rel_data.get("relation_type", rel_data.get("type", "related_to"))

                # Get display text
                type_display = self._get_relation_display(relation_type)

                source_name = result.get("a", {}).get("name", "") if isinstance(result.get("a"), dict) else ""
                target_name = result.get("b", {}).get("name", "") if isinstance(result.get("b"), dict) else ""

                display_text = f"{source_name} {type_display} {target_name}" if source_name and target_name else type_display

                relations.append(FormattedRelation(
                    id=str(rel_data.get("id", "")),
                    source_id=str(result.get("a", {}).get("id", "")) if isinstance(result.get("a"), dict) else "",
                    source_name=source_name,
                    target_id=str(result.get("b", {}).get("id", "")) if isinstance(result.get("b"), dict) else "",
                    target_name=target_name,
                    relation_type=relation_type,
                    properties={k: v for k, v in rel_data.items()
                               if k not in {"id", "relation_type", "type"}},
                    display_text=display_text,
                ))

        return relations

    def _format_paths(self, raw_results: List[Dict[str, Any]]) -> List[FormattedPath]:
        """Format path results."""
        paths = []

        for i, result in enumerate(raw_results):
            path_data = result.get("path", result)
            path_length = result.get("pathLength", 0)

            nodes = []
            relationships = []

            # Extract nodes and relationships from path
            if isinstance(path_data, dict):
                path_nodes = path_data.get("nodes", [])
                path_rels = path_data.get("relationships", [])

                for node in path_nodes:
                    if isinstance(node, dict):
                        nodes.append(FormattedEntity(
                            id=str(node.get("id", "")),
                            name=node.get("name", ""),
                            entity_type=node.get("entity_type", ""),
                        ))

                for rel in path_rels:
                    if isinstance(rel, dict):
                        relationships.append(FormattedRelation(
                            id=str(rel.get("id", "")),
                            relation_type=rel.get("relation_type", "related_to"),
                        ))

            # Create display text
            if nodes:
                node_names = [n.name for n in nodes if n.name]
                display_text = " → ".join(node_names)
            else:
                display_text = f"Path {i + 1}"

            paths.append(FormattedPath(
                path_id=f"path_{i}",
                nodes=nodes,
                relationships=relationships,
                length=path_length if path_length else len(relationships),
                display_text=display_text,
            ))

        return paths

    def _format_statistics(self, raw_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format statistics results."""
        stats = {
            "by_type": {},
            "total": 0,
        }

        for result in raw_results:
            label = result.get("label", "Unknown")
            count = result.get("count", 0)
            stats["by_type"][label] = count
            stats["total"] += count

        return stats

    def _format_mixed(self, raw_results: List[Dict[str, Any]]) -> Tuple[List[FormattedEntity], List[FormattedRelation]]:
        """Format mixed results (entities and relations)."""
        entities = []
        relations = []

        for result in raw_results:
            # Extract entities
            for key in ["a", "b", "n", "neighbor", "similar"]:
                if key in result and isinstance(result[key], dict):
                    node = result[key]
                    entity_type = node.get("entity_type", node.get("type", "unknown"))
                    entities.append(FormattedEntity(
                        id=str(node.get("id", "")),
                        name=node.get("name", ""),
                        entity_type=entity_type,
                        display_name=f"{node.get('name', '')} ({self._get_entity_type_display(entity_type)})",
                    ))

            # Extract relations
            if "r" in result and isinstance(result["r"], dict):
                rel = result["r"]
                relation_type = rel.get("relation_type", rel.get("type", "related_to"))
                relations.append(FormattedRelation(
                    id=str(rel.get("id", "")),
                    relation_type=relation_type,
                    display_text=self._get_relation_display(relation_type),
                ))

        # Deduplicate entities
        seen_ids = set()
        unique_entities = []
        for e in entities:
            if e.id and e.id not in seen_ids:
                seen_ids.add(e.id)
                unique_entities.append(e)

        return unique_entities, relations

    def _get_entity_type_display(self, entity_type: str) -> str:
        """Get display name for entity type."""
        entity_type_lower = entity_type.lower()
        display_names = self.ENTITY_TYPE_DISPLAY_NAMES.get(entity_type_lower, (entity_type, entity_type))
        return display_names[0] if self.language == "zh" else display_names[1]

    def _get_relation_display(self, relation_type: str) -> str:
        """Get display name for relation type."""
        relation_type_lower = relation_type.lower()
        display_names = self.RELATION_DISPLAY_NAMES.get(relation_type_lower, (relation_type, relation_type))
        return display_names[0] if self.language == "zh" else display_names[1]

    def _generate_explanation(
        self,
        result_type: ResultType,
        entities: List[FormattedEntity],
        relations: List[FormattedRelation],
        paths: List[FormattedPath],
        statistics: Dict[str, Any],
        query_context: Dict[str, Any],
    ) -> str:
        """Generate natural language explanation of results."""
        if self.language == "zh":
            return self._generate_explanation_zh(result_type, entities, relations, paths, statistics, query_context)
        else:
            return self._generate_explanation_en(result_type, entities, relations, paths, statistics, query_context)

    def _generate_explanation_zh(
        self,
        result_type: ResultType,
        entities: List[FormattedEntity],
        relations: List[FormattedRelation],
        paths: List[FormattedPath],
        statistics: Dict[str, Any],
        query_context: Dict[str, Any],
    ) -> str:
        """Generate Chinese explanation."""
        if result_type == ResultType.ENTITIES:
            if not entities:
                return "未找到匹配的实体。"
            return f"找到 {len(entities)} 个相关实体。"

        elif result_type == ResultType.RELATIONS:
            if not relations:
                return "未找到相关关系。"
            return f"找到 {len(relations)} 个关系。"

        elif result_type == ResultType.PATHS:
            if not paths:
                return "未找到连接路径。"
            shortest = min(paths, key=lambda p: p.length) if paths else None
            if shortest:
                return f"找到 {len(paths)} 条路径，最短路径长度为 {shortest.length}。"
            return f"找到 {len(paths)} 条路径。"

        elif result_type == ResultType.STATISTICS:
            total = statistics.get("total", 0)
            return f"图谱统计：共有 {total} 个节点。"

        elif result_type == ResultType.COUNT:
            count = statistics.get("count", 0)
            return f"统计结果：共 {count} 个。"

        else:
            entity_count = len(entities)
            relation_count = len(relations)
            return f"找到 {entity_count} 个实体和 {relation_count} 个关系。"

    def _generate_explanation_en(
        self,
        result_type: ResultType,
        entities: List[FormattedEntity],
        relations: List[FormattedRelation],
        paths: List[FormattedPath],
        statistics: Dict[str, Any],
        query_context: Dict[str, Any],
    ) -> str:
        """Generate English explanation."""
        if result_type == ResultType.ENTITIES:
            if not entities:
                return "No matching entities found."
            return f"Found {len(entities)} related entities."

        elif result_type == ResultType.RELATIONS:
            if not relations:
                return "No related relationships found."
            return f"Found {len(relations)} relationships."

        elif result_type == ResultType.PATHS:
            if not paths:
                return "No connecting paths found."
            shortest = min(paths, key=lambda p: p.length) if paths else None
            if shortest:
                return f"Found {len(paths)} paths. Shortest path length: {shortest.length}."
            return f"Found {len(paths)} paths."

        elif result_type == ResultType.STATISTICS:
            total = statistics.get("total", 0)
            return f"Graph statistics: {total} total nodes."

        elif result_type == ResultType.COUNT:
            count = statistics.get("count", 0)
            return f"Count result: {count} items."

        else:
            entity_count = len(entities)
            relation_count = len(relations)
            return f"Found {entity_count} entities and {relation_count} relationships."

    def _generate_recommendations(
        self,
        entities: List[FormattedEntity],
        relations: List[FormattedRelation],
        query_context: Dict[str, Any],
    ) -> List[Recommendation]:
        """Generate related content recommendations."""
        recommendations = []

        # Recommend based on entity types found
        entity_types = set(e.entity_type for e in entities)

        if "person" in entity_types:
            recommendations.append(Recommendation(
                item_type="query",
                title="查看相关组织" if self.language == "zh" else "View related organizations",
                reason="基于人物查找其所属组织" if self.language == "zh" else "Find organizations related to these people",
                score=0.8,
            ))

        if "organization" in entity_types:
            recommendations.append(Recommendation(
                item_type="query",
                title="查看组织成员" if self.language == "zh" else "View organization members",
                reason="查找组织中的人员" if self.language == "zh" else "Find people in these organizations",
                score=0.8,
            ))

        # General recommendations
        if entities:
            recommendations.append(Recommendation(
                item_type="query",
                title="探索关系网络" if self.language == "zh" else "Explore relationship network",
                reason="查看实体之间的更多关联" if self.language == "zh" else "View more connections between entities",
                score=0.7,
            ))

        return recommendations[:5]

    def _generate_formatted_output(
        self,
        output_format: OutputFormat,
        entities: List[FormattedEntity],
        relations: List[FormattedRelation],
        paths: List[FormattedPath],
        statistics: Dict[str, Any],
    ) -> str:
        """Generate formatted string output."""
        if output_format == OutputFormat.TEXT:
            return self._format_as_text(entities, relations, paths, statistics)
        elif output_format == OutputFormat.MARKDOWN:
            return self._format_as_markdown(entities, relations, paths, statistics)
        elif output_format == OutputFormat.TABLE:
            return self._format_as_table(entities, relations, paths)
        elif output_format == OutputFormat.CSV:
            return self._format_as_csv(entities, relations)
        else:
            return ""  # JSON format is handled by the FormattedResult itself

    def _format_as_text(
        self,
        entities: List[FormattedEntity],
        relations: List[FormattedRelation],
        paths: List[FormattedPath],
        statistics: Dict[str, Any],
    ) -> str:
        """Format as plain text."""
        lines = []

        if entities:
            lines.append("Entities:")
            for e in entities[:10]:
                lines.append(f"  - {e.display_name}")

        if relations:
            lines.append("\nRelations:")
            for r in relations[:10]:
                lines.append(f"  - {r.display_text}")

        if paths:
            lines.append("\nPaths:")
            for p in paths[:5]:
                lines.append(f"  - {p.display_text} (length: {p.length})")

        if statistics:
            lines.append("\nStatistics:")
            for key, value in statistics.items():
                lines.append(f"  - {key}: {value}")

        return "\n".join(lines)

    def _format_as_markdown(
        self,
        entities: List[FormattedEntity],
        relations: List[FormattedRelation],
        paths: List[FormattedPath],
        statistics: Dict[str, Any],
    ) -> str:
        """Format as Markdown."""
        lines = []

        if entities:
            lines.append("## Entities\n")
            for e in entities[:10]:
                lines.append(f"- **{e.name}** ({e.entity_type})")
                if e.summary:
                    lines.append(f"  - {e.summary}")
            lines.append("")

        if relations:
            lines.append("## Relations\n")
            for r in relations[:10]:
                lines.append(f"- {r.display_text}")
            lines.append("")

        if paths:
            lines.append("## Paths\n")
            for p in paths[:5]:
                lines.append(f"- `{p.display_text}` (length: {p.length})")
            lines.append("")

        if statistics:
            lines.append("## Statistics\n")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for key, value in statistics.items():
                lines.append(f"| {key} | {value} |")

        return "\n".join(lines)

    def _format_as_table(
        self,
        entities: List[FormattedEntity],
        relations: List[FormattedRelation],
        paths: List[FormattedPath],
    ) -> str:
        """Format as ASCII table."""
        lines = []

        if entities:
            lines.append("+" + "-" * 30 + "+" + "-" * 15 + "+")
            lines.append("| {:28} | {:13} |".format("Name", "Type"))
            lines.append("+" + "-" * 30 + "+" + "-" * 15 + "+")
            for e in entities[:10]:
                name = e.name[:28] if len(e.name) > 28 else e.name
                etype = e.entity_type[:13] if len(e.entity_type) > 13 else e.entity_type
                lines.append("| {:28} | {:13} |".format(name, etype))
            lines.append("+" + "-" * 30 + "+" + "-" * 15 + "+")

        return "\n".join(lines)

    def _format_as_csv(
        self,
        entities: List[FormattedEntity],
        relations: List[FormattedRelation],
    ) -> str:
        """Format as CSV."""
        lines = []

        if entities:
            lines.append("id,name,type")
            for e in entities:
                name = e.name.replace(",", ";").replace('"', '""')
                lines.append(f'"{e.id}","{name}","{e.entity_type}"')

        return "\n".join(lines)


# Global instance
_result_formatter: Optional[ResultFormatter] = None


def get_result_formatter() -> ResultFormatter:
    """Get or create global ResultFormatter instance."""
    global _result_formatter

    if _result_formatter is None:
        _result_formatter = ResultFormatter()
        _result_formatter.initialize()

    return _result_formatter
