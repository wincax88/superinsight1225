"""
Interactive UI data layer for Knowledge Graph visualization.

Provides node expansion, search/filter, and configuration management.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class InteractionType(str, Enum):
    """Types of user interactions."""

    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    HOVER = "hover"
    DRAG = "drag"
    SELECT = "select"
    EXPAND = "expand"
    COLLAPSE = "collapse"
    ZOOM = "zoom"
    PAN = "pan"
    CONTEXT_MENU = "context_menu"


class FilterOperator(str, Enum):
    """Operators for filtering."""

    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"
    REGEX = "regex"


class SelectionMode(str, Enum):
    """Selection modes."""

    SINGLE = "single"
    MULTIPLE = "multiple"
    LASSO = "lasso"
    RECTANGLE = "rectangle"


class ExpandDirection(str, Enum):
    """Direction for node expansion."""

    OUTGOING = "outgoing"
    INCOMING = "incoming"
    BOTH = "both"


class NodeExpansionData(BaseModel):
    """Data for node expansion result."""

    source_node_id: str = Field(..., description="ID of expanded node")
    new_nodes: list[dict[str, Any]] = Field(default_factory=list, description="Newly revealed nodes")
    new_edges: list[dict[str, Any]] = Field(default_factory=list, description="Newly revealed edges")
    direction: ExpandDirection = Field(default=ExpandDirection.BOTH)
    depth: int = Field(default=1, description="Expansion depth")
    total_neighbors: int = Field(default=0, description="Total available neighbors")
    has_more: bool = Field(default=False, description="More neighbors available")


class SearchQuery(BaseModel):
    """Search query configuration."""

    query: str = Field(..., description="Search query text")
    search_fields: list[str] = Field(
        default_factory=lambda: ["label", "type"],
        description="Fields to search in",
    )
    case_sensitive: bool = Field(default=False)
    fuzzy: bool = Field(default=False)
    fuzzy_threshold: float = Field(default=0.7, description="Fuzzy match threshold")
    max_results: int = Field(default=50)


class SearchResult(BaseModel):
    """Search result item."""

    node_id: str = Field(..., description="Matched node ID")
    label: str = Field(..., description="Node label")
    node_type: str = Field(default="", description="Node type")
    match_field: str = Field(default="", description="Field that matched")
    match_score: float = Field(default=1.0, description="Match score (0-1)")
    highlight: str = Field(default="", description="Highlighted match text")


class FilterCondition(BaseModel):
    """A single filter condition."""

    field: str = Field(..., description="Field to filter on")
    operator: FilterOperator = Field(..., description="Filter operator")
    value: Any = Field(..., description="Value to filter against")
    negate: bool = Field(default=False, description="Negate the condition")


class FilterConfig(BaseModel):
    """Configuration for graph filtering."""

    conditions: list[FilterCondition] = Field(default_factory=list)
    node_types: list[str] = Field(default_factory=list, description="Include only these node types")
    edge_types: list[str] = Field(default_factory=list, description="Include only these edge types")
    min_degree: Optional[int] = Field(default=None, description="Minimum node degree")
    max_degree: Optional[int] = Field(default=None, description="Maximum node degree")
    date_range: Optional[tuple[datetime, datetime]] = Field(default=None)
    combine_mode: str = Field(default="and", description="'and' or 'or' for combining conditions")


class FilterResult(BaseModel):
    """Result of applying filters."""

    visible_node_ids: list[str] = Field(default_factory=list)
    visible_edge_ids: list[str] = Field(default_factory=list)
    hidden_node_count: int = Field(default=0)
    hidden_edge_count: int = Field(default=0)
    filters_applied: list[str] = Field(default_factory=list)


class SelectionState(BaseModel):
    """Current selection state."""

    selected_nodes: list[str] = Field(default_factory=list)
    selected_edges: list[str] = Field(default_factory=list)
    mode: SelectionMode = Field(default=SelectionMode.SINGLE)
    anchor_node: Optional[str] = Field(default=None, description="Anchor for range selection")


class ViewportState(BaseModel):
    """Current viewport state."""

    center_x: float = Field(default=0.0)
    center_y: float = Field(default=0.0)
    zoom: float = Field(default=1.0)
    min_zoom: float = Field(default=0.1)
    max_zoom: float = Field(default=5.0)
    width: int = Field(default=1200)
    height: int = Field(default=800)


class TooltipData(BaseModel):
    """Data for node/edge tooltip."""

    target_id: str = Field(..., description="ID of hovered element")
    target_type: str = Field(..., description="'node' or 'edge'")
    title: str = Field(..., description="Tooltip title")
    content: dict[str, Any] = Field(default_factory=dict, description="Key-value pairs to display")
    position: dict[str, float] = Field(default_factory=dict, description="Tooltip position")


class ContextMenuData(BaseModel):
    """Data for context menu."""

    target_id: str = Field(..., description="ID of right-clicked element")
    target_type: str = Field(..., description="'node', 'edge', or 'canvas'")
    position: dict[str, float] = Field(default_factory=dict)
    menu_items: list[dict[str, Any]] = Field(default_factory=list)


class VisualizationConfig(BaseModel):
    """Complete visualization configuration."""

    # Display options
    show_labels: bool = Field(default=True)
    show_edge_labels: bool = Field(default=True)
    show_legend: bool = Field(default=True)
    show_minimap: bool = Field(default=True)
    show_toolbar: bool = Field(default=True)
    show_stats: bool = Field(default=True)

    # Interaction options
    enable_zoom: bool = Field(default=True)
    enable_pan: bool = Field(default=True)
    enable_drag: bool = Field(default=True)
    enable_selection: bool = Field(default=True)
    enable_expansion: bool = Field(default=True)
    enable_tooltips: bool = Field(default=True)
    enable_context_menu: bool = Field(default=True)

    # Selection options
    selection_mode: SelectionMode = Field(default=SelectionMode.MULTIPLE)
    highlight_neighbors_on_select: bool = Field(default=True)
    dim_unselected: bool = Field(default=True)
    dim_opacity: float = Field(default=0.3)

    # Performance options
    max_visible_nodes: int = Field(default=500)
    max_visible_edges: int = Field(default=1000)
    enable_clustering: bool = Field(default=True)
    cluster_threshold: int = Field(default=100)

    # Animation options
    animate_layout: bool = Field(default=True)
    animation_duration: int = Field(default=500, description="Animation duration in ms")

    # Theme options
    theme: str = Field(default="light", description="'light' or 'dark'")
    custom_colors: dict[str, str] = Field(default_factory=dict)


class InteractionEvent(BaseModel):
    """An interaction event."""

    event_id: str = Field(..., description="Unique event identifier")
    event_type: InteractionType = Field(..., description="Type of interaction")
    target_id: Optional[str] = Field(default=None, description="Target element ID")
    target_type: Optional[str] = Field(default=None, description="Target element type")
    position: dict[str, float] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict, description="Additional event data")
    timestamp: datetime = Field(default_factory=datetime.now)


@dataclass
class InteractiveUI:
    """Interactive UI data layer for graph visualization."""

    config: VisualizationConfig = field(default_factory=VisualizationConfig)
    selection: SelectionState = field(default_factory=SelectionState)
    viewport: ViewportState = field(default_factory=ViewportState)
    filters: FilterConfig = field(default_factory=FilterConfig)
    event_handlers: dict[InteractionType, list[Callable]] = field(default_factory=dict)
    expansion_cache: dict[str, NodeExpansionData] = field(default_factory=dict)

    def expand_node(
        self,
        node_id: str,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        direction: ExpandDirection = ExpandDirection.BOTH,
        depth: int = 1,
        max_neighbors: int = 20,
    ) -> NodeExpansionData:
        """Generate expansion data for a node."""
        # Find connected nodes
        connected_nodes = []
        connected_edges = []

        existing_node_ids = {n.get("id", n.get("node_id", "")) for n in nodes}

        for edge in edges:
            source = edge.get("source", edge.get("source_id", edge.get("from", "")))
            target = edge.get("target", edge.get("target_id", edge.get("to", "")))

            is_source = str(source) == str(node_id)
            is_target = str(target) == str(node_id)

            include = False
            neighbor_id = None

            if direction == ExpandDirection.OUTGOING and is_source:
                include = True
                neighbor_id = target
            elif direction == ExpandDirection.INCOMING and is_target:
                include = True
                neighbor_id = source
            elif direction == ExpandDirection.BOTH and (is_source or is_target):
                include = True
                neighbor_id = target if is_source else source

            if include and neighbor_id:
                # Check if neighbor is not already visible
                if str(neighbor_id) not in existing_node_ids:
                    connected_edges.append(edge)
                    # Find neighbor node data (mock for now)
                    connected_nodes.append({
                        "id": str(neighbor_id),
                        "label": str(neighbor_id),
                        "type": "default",
                    })

        # Limit results
        total = len(connected_nodes)
        has_more = total > max_neighbors

        if has_more:
            connected_nodes = connected_nodes[:max_neighbors]
            connected_edges = connected_edges[:max_neighbors]

        expansion = NodeExpansionData(
            source_node_id=node_id,
            new_nodes=connected_nodes,
            new_edges=connected_edges,
            direction=direction,
            depth=depth,
            total_neighbors=total,
            has_more=has_more,
        )

        # Cache result
        self.expansion_cache[node_id] = expansion

        return expansion

    def search(
        self,
        query: SearchQuery,
        nodes: list[dict[str, Any]],
    ) -> list[SearchResult]:
        """Search nodes based on query."""
        results = []
        search_text = query.query if query.case_sensitive else query.query.lower()

        for node in nodes:
            node_id = str(node.get("id", node.get("node_id", "")))

            for field in query.search_fields:
                field_value = node.get(field, "")
                if field_value is None:
                    continue

                field_str = str(field_value)
                compare_str = field_str if query.case_sensitive else field_str.lower()

                match_score = 0.0
                if query.fuzzy:
                    match_score = self._fuzzy_match(search_text, compare_str)
                    if match_score < query.fuzzy_threshold:
                        continue
                else:
                    if search_text in compare_str:
                        match_score = 1.0
                    else:
                        continue

                # Generate highlight
                highlight = self._highlight_match(field_str, query.query, query.case_sensitive)

                results.append(
                    SearchResult(
                        node_id=node_id,
                        label=str(node.get("label", node.get("name", node_id))),
                        node_type=str(node.get("type", node.get("node_type", ""))),
                        match_field=field,
                        match_score=match_score,
                        highlight=highlight,
                    )
                )
                break  # Only one result per node

        # Sort by score and limit
        results.sort(key=lambda x: x.match_score, reverse=True)
        return results[: query.max_results]

    def apply_filters(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        filter_config: Optional[FilterConfig] = None,
    ) -> FilterResult:
        """Apply filters to nodes and edges."""
        if filter_config:
            self.filters = filter_config

        visible_nodes = []
        hidden_node_count = 0

        # Calculate degrees for degree filtering
        degrees: dict[str, int] = {}
        for edge in edges:
            source = str(edge.get("source", edge.get("source_id", "")))
            target = str(edge.get("target", edge.get("target_id", "")))
            degrees[source] = degrees.get(source, 0) + 1
            degrees[target] = degrees.get(target, 0) + 1

        # Filter nodes
        for node in nodes:
            node_id = str(node.get("id", node.get("node_id", "")))
            node_type = str(node.get("type", node.get("node_type", ""))).lower()

            # Check type filter
            if self.filters.node_types and node_type not in [
                t.lower() for t in self.filters.node_types
            ]:
                hidden_node_count += 1
                continue

            # Check degree filters
            degree = degrees.get(node_id, 0)
            if self.filters.min_degree is not None and degree < self.filters.min_degree:
                hidden_node_count += 1
                continue
            if self.filters.max_degree is not None and degree > self.filters.max_degree:
                hidden_node_count += 1
                continue

            # Check custom conditions
            if not self._check_conditions(node, self.filters.conditions, self.filters.combine_mode):
                hidden_node_count += 1
                continue

            visible_nodes.append(node_id)

        # Filter edges (only those connecting visible nodes)
        visible_edges = []
        hidden_edge_count = 0
        visible_set = set(visible_nodes)

        for edge in edges:
            source = str(edge.get("source", edge.get("source_id", "")))
            target = str(edge.get("target", edge.get("target_id", "")))
            edge_type = str(edge.get("type", edge.get("edge_type", ""))).lower()

            # Check if both ends are visible
            if source not in visible_set or target not in visible_set:
                hidden_edge_count += 1
                continue

            # Check edge type filter
            if self.filters.edge_types and edge_type not in [
                t.lower() for t in self.filters.edge_types
            ]:
                hidden_edge_count += 1
                continue

            edge_id = str(edge.get("id", edge.get("edge_id", f"{source}_{target}")))
            visible_edges.append(edge_id)

        # Build filter description
        filters_applied = []
        if self.filters.node_types:
            filters_applied.append(f"Node types: {', '.join(self.filters.node_types)}")
        if self.filters.edge_types:
            filters_applied.append(f"Edge types: {', '.join(self.filters.edge_types)}")
        if self.filters.min_degree:
            filters_applied.append(f"Min degree: {self.filters.min_degree}")
        if self.filters.max_degree:
            filters_applied.append(f"Max degree: {self.filters.max_degree}")
        if self.filters.conditions:
            filters_applied.append(f"{len(self.filters.conditions)} custom conditions")

        return FilterResult(
            visible_node_ids=visible_nodes,
            visible_edge_ids=visible_edges,
            hidden_node_count=hidden_node_count,
            hidden_edge_count=hidden_edge_count,
            filters_applied=filters_applied,
        )

    def update_selection(
        self,
        node_ids: Optional[list[str]] = None,
        edge_ids: Optional[list[str]] = None,
        add: bool = False,
        clear: bool = False,
    ) -> SelectionState:
        """Update selection state."""
        if clear:
            self.selection.selected_nodes = []
            self.selection.selected_edges = []
        elif add:
            if node_ids:
                self.selection.selected_nodes.extend(
                    [n for n in node_ids if n not in self.selection.selected_nodes]
                )
            if edge_ids:
                self.selection.selected_edges.extend(
                    [e for e in edge_ids if e not in self.selection.selected_edges]
                )
        else:
            if node_ids is not None:
                self.selection.selected_nodes = node_ids
            if edge_ids is not None:
                self.selection.selected_edges = edge_ids

        return self.selection

    def get_tooltip_data(
        self,
        element_id: str,
        element_type: str,
        element_data: dict[str, Any],
        position: tuple[float, float],
    ) -> TooltipData:
        """Generate tooltip data for an element."""
        if element_type == "node":
            title = str(element_data.get("label", element_data.get("name", element_id)))
            content = {
                "Type": element_data.get("type", element_data.get("node_type", "Unknown")),
            }
            # Add properties
            properties = element_data.get("properties", {})
            for key, value in list(properties.items())[:5]:  # Limit to 5 properties
                content[key] = str(value)
        else:
            title = str(element_data.get("label", element_data.get("type", "Relationship")))
            content = {
                "Source": str(element_data.get("source", element_data.get("source_id", ""))),
                "Target": str(element_data.get("target", element_data.get("target_id", ""))),
                "Type": element_data.get("type", element_data.get("edge_type", "")),
            }
            if "weight" in element_data:
                content["Weight"] = str(element_data["weight"])

        return TooltipData(
            target_id=element_id,
            target_type=element_type,
            title=title,
            content=content,
            position={"x": position[0], "y": position[1]},
        )

    def get_context_menu(
        self,
        element_id: str,
        element_type: str,
        position: tuple[float, float],
    ) -> ContextMenuData:
        """Generate context menu data."""
        menu_items = []

        if element_type == "node":
            menu_items = [
                {"id": "expand", "label": "Expand Node", "icon": "expand"},
                {"id": "collapse", "label": "Collapse Node", "icon": "collapse"},
                {"id": "separator"},
                {"id": "select_neighbors", "label": "Select Neighbors", "icon": "select"},
                {"id": "hide", "label": "Hide Node", "icon": "hide"},
                {"id": "separator"},
                {"id": "details", "label": "View Details", "icon": "info"},
                {"id": "edit", "label": "Edit Node", "icon": "edit"},
            ]
        elif element_type == "edge":
            menu_items = [
                {"id": "details", "label": "View Details", "icon": "info"},
                {"id": "hide", "label": "Hide Edge", "icon": "hide"},
                {"id": "highlight_path", "label": "Highlight Path", "icon": "path"},
            ]
        else:  # canvas
            menu_items = [
                {"id": "select_all", "label": "Select All", "icon": "select"},
                {"id": "clear_selection", "label": "Clear Selection", "icon": "clear"},
                {"id": "separator"},
                {"id": "fit_view", "label": "Fit to View", "icon": "fit"},
                {"id": "reset_view", "label": "Reset View", "icon": "reset"},
                {"id": "separator"},
                {"id": "export_image", "label": "Export as Image", "icon": "image"},
                {"id": "export_data", "label": "Export Data", "icon": "export"},
            ]

        return ContextMenuData(
            target_id=element_id,
            target_type=element_type,
            position={"x": position[0], "y": position[1]},
            menu_items=menu_items,
        )

    def update_viewport(
        self,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
        zoom: Optional[float] = None,
    ) -> ViewportState:
        """Update viewport state."""
        if center_x is not None:
            self.viewport.center_x = center_x
        if center_y is not None:
            self.viewport.center_y = center_y
        if zoom is not None:
            self.viewport.zoom = max(
                self.viewport.min_zoom,
                min(self.viewport.max_zoom, zoom),
            )
        return self.viewport

    def fit_to_content(
        self,
        bounds: dict[str, float],
        padding: float = 50.0,
    ) -> ViewportState:
        """Fit viewport to content bounds."""
        content_width = bounds.get("max_x", 0) - bounds.get("min_x", 0)
        content_height = bounds.get("max_y", 0) - bounds.get("min_y", 0)

        if content_width <= 0 or content_height <= 0:
            return self.viewport

        # Calculate zoom to fit
        zoom_x = (self.viewport.width - 2 * padding) / content_width
        zoom_y = (self.viewport.height - 2 * padding) / content_height
        zoom = min(zoom_x, zoom_y, self.viewport.max_zoom)
        zoom = max(zoom, self.viewport.min_zoom)

        # Calculate center
        center_x = (bounds.get("min_x", 0) + bounds.get("max_x", 0)) / 2
        center_y = (bounds.get("min_y", 0) + bounds.get("max_y", 0)) / 2

        self.viewport.center_x = center_x
        self.viewport.center_y = center_y
        self.viewport.zoom = zoom

        return self.viewport

    def register_event_handler(
        self,
        event_type: InteractionType,
        handler: Callable,
    ) -> None:
        """Register an event handler."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    def _check_conditions(
        self,
        item: dict[str, Any],
        conditions: list[FilterCondition],
        combine_mode: str,
    ) -> bool:
        """Check if item passes all/any conditions."""
        if not conditions:
            return True

        results = []
        for condition in conditions:
            value = item.get(condition.field)
            result = self._evaluate_condition(value, condition)
            if condition.negate:
                result = not result
            results.append(result)

        if combine_mode == "or":
            return any(results)
        else:
            return all(results)

    def _evaluate_condition(
        self,
        value: Any,
        condition: FilterCondition,
    ) -> bool:
        """Evaluate a single condition."""
        if value is None:
            return False

        value_str = str(value).lower()
        cond_value = condition.value

        if condition.operator == FilterOperator.EQUALS:
            return str(value).lower() == str(cond_value).lower()
        elif condition.operator == FilterOperator.NOT_EQUALS:
            return str(value).lower() != str(cond_value).lower()
        elif condition.operator == FilterOperator.CONTAINS:
            return str(cond_value).lower() in value_str
        elif condition.operator == FilterOperator.STARTS_WITH:
            return value_str.startswith(str(cond_value).lower())
        elif condition.operator == FilterOperator.ENDS_WITH:
            return value_str.endswith(str(cond_value).lower())
        elif condition.operator == FilterOperator.GREATER_THAN:
            try:
                return float(value) > float(cond_value)
            except (ValueError, TypeError):
                return False
        elif condition.operator == FilterOperator.LESS_THAN:
            try:
                return float(value) < float(cond_value)
            except (ValueError, TypeError):
                return False
        elif condition.operator == FilterOperator.IN:
            if isinstance(cond_value, list):
                return value in cond_value
            return False
        elif condition.operator == FilterOperator.NOT_IN:
            if isinstance(cond_value, list):
                return value not in cond_value
            return True

        return False

    def _fuzzy_match(self, query: str, text: str) -> float:
        """Simple fuzzy matching using character overlap."""
        if not query or not text:
            return 0.0

        query_chars = set(query.lower())
        text_chars = set(text.lower())

        overlap = len(query_chars & text_chars)
        total = len(query_chars | text_chars)

        return overlap / total if total > 0 else 0.0

    def _highlight_match(
        self,
        text: str,
        query: str,
        case_sensitive: bool,
    ) -> str:
        """Generate highlighted text with match markers."""
        if not query:
            return text

        search_text = text if case_sensitive else text.lower()
        search_query = query if case_sensitive else query.lower()

        idx = search_text.find(search_query)
        if idx == -1:
            return text

        return (
            text[:idx]
            + "**"
            + text[idx : idx + len(query)]
            + "**"
            + text[idx + len(query) :]
        )


# Global instance
_interactive_ui: Optional[InteractiveUI] = None


def get_interactive_ui() -> InteractiveUI:
    """Get or create the global interactive UI instance."""
    global _interactive_ui
    if _interactive_ui is None:
        _interactive_ui = InteractiveUI()
    return _interactive_ui
