"""
Graph rendering engine for Knowledge Graph visualization.

Provides graph data processing, node/edge formatting, and large-scale optimization.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class NodeShape(str, Enum):
    """Shapes for rendering nodes."""

    CIRCLE = "circle"
    RECTANGLE = "rectangle"
    DIAMOND = "diamond"
    TRIANGLE = "triangle"
    HEXAGON = "hexagon"
    STAR = "star"
    ELLIPSE = "ellipse"


class EdgeStyle(str, Enum):
    """Styles for rendering edges."""

    SOLID = "solid"
    DASHED = "dashed"
    DOTTED = "dotted"
    DOUBLE = "double"


class ColorScheme(str, Enum):
    """Color schemes for visualization."""

    DEFAULT = "default"
    CATEGORICAL = "categorical"
    SEQUENTIAL = "sequential"
    DIVERGING = "diverging"
    CUSTOM = "custom"


class RenderingMode(str, Enum):
    """Rendering modes for different use cases."""

    FULL = "full"  # All details
    SIMPLIFIED = "simplified"  # Reduced details for performance
    PREVIEW = "preview"  # Minimal for quick preview
    THUMBNAIL = "thumbnail"  # Very small overview


class NodeStyle(BaseModel):
    """Visual style for a node."""

    shape: NodeShape = Field(default=NodeShape.CIRCLE, description="Node shape")
    size: float = Field(default=30.0, description="Node size in pixels")
    fill_color: str = Field(default="#4A90D9", description="Fill color (hex)")
    border_color: str = Field(default="#2C5282", description="Border color")
    border_width: float = Field(default=2.0, description="Border width")
    opacity: float = Field(default=1.0, description="Opacity (0-1)")
    label_visible: bool = Field(default=True, description="Show label")
    label_font_size: int = Field(default=12, description="Label font size")
    label_color: str = Field(default="#1A202C", description="Label color")
    icon: Optional[str] = Field(default=None, description="Icon identifier")


class EdgeStyleConfig(BaseModel):
    """Visual style for an edge."""

    style: EdgeStyle = Field(default=EdgeStyle.SOLID, description="Line style")
    width: float = Field(default=2.0, description="Line width")
    color: str = Field(default="#718096", description="Edge color")
    opacity: float = Field(default=0.8, description="Opacity")
    arrow_size: float = Field(default=8.0, description="Arrow size")
    show_arrow: bool = Field(default=True, description="Show direction arrow")
    label_visible: bool = Field(default=True, description="Show edge label")
    label_font_size: int = Field(default=10, description="Label font size")
    curved: bool = Field(default=False, description="Use curved lines")
    curve_strength: float = Field(default=0.3, description="Curve strength")


class RenderedNode(BaseModel):
    """A node prepared for rendering."""

    node_id: str = Field(..., description="Unique node identifier")
    label: str = Field(..., description="Display label")
    node_type: str = Field(default="default", description="Node type/category")
    x: float = Field(default=0.0, description="X coordinate")
    y: float = Field(default=0.0, description="Y coordinate")
    style: NodeStyle = Field(default_factory=NodeStyle, description="Visual style")
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Additional properties"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata for interaction"
    )


class RenderedEdge(BaseModel):
    """An edge prepared for rendering."""

    edge_id: str = Field(..., description="Unique edge identifier")
    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    label: str = Field(default="", description="Edge label")
    edge_type: str = Field(default="default", description="Edge type")
    style: EdgeStyleConfig = Field(
        default_factory=EdgeStyleConfig, description="Visual style"
    )
    weight: float = Field(default=1.0, description="Edge weight")
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Additional properties"
    )


class GraphBounds(BaseModel):
    """Bounding box for the graph."""

    min_x: float = Field(default=0.0)
    max_x: float = Field(default=100.0)
    min_y: float = Field(default=0.0)
    max_y: float = Field(default=100.0)
    width: float = Field(default=100.0)
    height: float = Field(default=100.0)
    padding: float = Field(default=20.0)


class GraphStatistics(BaseModel):
    """Statistics about the graph."""

    node_count: int = Field(default=0)
    edge_count: int = Field(default=0)
    density: float = Field(default=0.0)
    avg_degree: float = Field(default=0.0)
    connected_components: int = Field(default=1)
    node_types: dict[str, int] = Field(default_factory=dict)
    edge_types: dict[str, int] = Field(default_factory=dict)


class RenderConfig(BaseModel):
    """Configuration for rendering."""

    mode: RenderingMode = Field(default=RenderingMode.FULL)
    color_scheme: ColorScheme = Field(default=ColorScheme.CATEGORICAL)
    width: int = Field(default=1200, description="Canvas width")
    height: int = Field(default=800, description="Canvas height")
    background_color: str = Field(default="#FFFFFF")
    show_legend: bool = Field(default=True)
    show_minimap: bool = Field(default=True)
    enable_zoom: bool = Field(default=True)
    enable_pan: bool = Field(default=True)
    max_visible_nodes: int = Field(default=500)
    max_visible_edges: int = Field(default=1000)
    cluster_threshold: int = Field(default=100)


class RenderedGraph(BaseModel):
    """Complete rendered graph ready for visualization."""

    graph_id: str = Field(..., description="Unique graph identifier")
    nodes: list[RenderedNode] = Field(default_factory=list)
    edges: list[RenderedEdge] = Field(default_factory=list)
    bounds: GraphBounds = Field(default_factory=GraphBounds)
    statistics: GraphStatistics = Field(default_factory=GraphStatistics)
    config: RenderConfig = Field(default_factory=RenderConfig)
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class GraphRenderer:
    """Graph rendering engine."""

    default_node_style: NodeStyle = field(default_factory=NodeStyle)
    default_edge_style: EdgeStyleConfig = field(default_factory=EdgeStyleConfig)
    type_colors: dict[str, str] = field(default_factory=dict)
    type_shapes: dict[str, NodeShape] = field(default_factory=dict)

    def __post_init__(self):
        # Default color palette for node types
        self.type_colors = {
            "person": "#E53E3E",
            "organization": "#3182CE",
            "location": "#38A169",
            "event": "#D69E2E",
            "concept": "#805AD5",
            "document": "#DD6B20",
            "product": "#00B5D8",
            "default": "#4A5568",
        }

        # Default shapes for node types
        self.type_shapes = {
            "person": NodeShape.CIRCLE,
            "organization": NodeShape.RECTANGLE,
            "location": NodeShape.HEXAGON,
            "event": NodeShape.DIAMOND,
            "concept": NodeShape.ELLIPSE,
            "document": NodeShape.RECTANGLE,
            "default": NodeShape.CIRCLE,
        }

    def render_graph(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        config: Optional[RenderConfig] = None,
    ) -> RenderedGraph:
        """Render a graph from raw node and edge data."""
        import uuid

        if config is None:
            config = RenderConfig()

        graph_id = f"graph_{uuid.uuid4().hex[:8]}"

        # Process nodes
        rendered_nodes = self._process_nodes(nodes, config)

        # Process edges
        rendered_edges = self._process_edges(edges, config)

        # Apply optimization for large graphs
        if len(rendered_nodes) > config.max_visible_nodes:
            rendered_nodes = self._optimize_nodes(rendered_nodes, config)

        if len(rendered_edges) > config.max_visible_edges:
            rendered_edges = self._optimize_edges(rendered_edges, config)

        # Calculate bounds
        bounds = self._calculate_bounds(rendered_nodes, config)

        # Calculate statistics
        statistics = self._calculate_statistics(rendered_nodes, rendered_edges)

        return RenderedGraph(
            graph_id=graph_id,
            nodes=rendered_nodes,
            edges=rendered_edges,
            bounds=bounds,
            statistics=statistics,
            config=config,
        )

    def _process_nodes(
        self,
        nodes: list[dict[str, Any]],
        config: RenderConfig,
    ) -> list[RenderedNode]:
        """Process raw nodes into rendered nodes."""
        rendered = []

        for node in nodes:
            node_id = node.get("id", node.get("node_id", str(len(rendered))))
            label = node.get("label", node.get("name", node_id))
            node_type = node.get("type", node.get("node_type", "default")).lower()

            # Get style based on type
            style = self._get_node_style(node_type, config)

            # Get position if available
            x = node.get("x", node.get("position", {}).get("x", 0.0))
            y = node.get("y", node.get("position", {}).get("y", 0.0))

            rendered.append(
                RenderedNode(
                    node_id=str(node_id),
                    label=str(label),
                    node_type=node_type,
                    x=float(x),
                    y=float(y),
                    style=style,
                    properties=node.get("properties", {}),
                    metadata=node.get("metadata", {}),
                )
            )

        return rendered

    def _process_edges(
        self,
        edges: list[dict[str, Any]],
        config: RenderConfig,
    ) -> list[RenderedEdge]:
        """Process raw edges into rendered edges."""
        rendered = []

        for edge in edges:
            edge_id = edge.get("id", edge.get("edge_id", str(len(rendered))))
            source = edge.get("source", edge.get("source_id", edge.get("from", "")))
            target = edge.get("target", edge.get("target_id", edge.get("to", "")))
            label = edge.get("label", edge.get("type", edge.get("relation", "")))
            edge_type = edge.get("type", edge.get("edge_type", "default"))

            # Get style based on type
            style = self._get_edge_style(edge_type, config)

            # Get weight
            weight = edge.get("weight", 1.0)

            rendered.append(
                RenderedEdge(
                    edge_id=str(edge_id),
                    source_id=str(source),
                    target_id=str(target),
                    label=str(label),
                    edge_type=edge_type,
                    style=style,
                    weight=float(weight),
                    properties=edge.get("properties", {}),
                )
            )

        return rendered

    def _get_node_style(
        self,
        node_type: str,
        config: RenderConfig,
    ) -> NodeStyle:
        """Get node style based on type."""
        color = self.type_colors.get(node_type, self.type_colors["default"])
        shape = self.type_shapes.get(node_type, self.type_shapes["default"])

        # Adjust style based on rendering mode
        if config.mode == RenderingMode.SIMPLIFIED:
            return NodeStyle(
                shape=shape,
                fill_color=color,
                size=20.0,
                label_visible=False,
            )
        elif config.mode == RenderingMode.PREVIEW:
            return NodeStyle(
                shape=NodeShape.CIRCLE,
                fill_color=color,
                size=10.0,
                label_visible=False,
                border_width=0,
            )
        elif config.mode == RenderingMode.THUMBNAIL:
            return NodeStyle(
                shape=NodeShape.CIRCLE,
                fill_color=color,
                size=5.0,
                label_visible=False,
                border_width=0,
            )
        else:
            return NodeStyle(
                shape=shape,
                fill_color=color,
                border_color=self._darken_color(color),
            )

    def _get_edge_style(
        self,
        edge_type: str,
        config: RenderConfig,
    ) -> EdgeStyleConfig:
        """Get edge style based on type."""
        # Default style with adjustments for edge type
        if config.mode in [RenderingMode.PREVIEW, RenderingMode.THUMBNAIL]:
            return EdgeStyleConfig(
                width=1.0,
                label_visible=False,
                show_arrow=False,
            )
        elif config.mode == RenderingMode.SIMPLIFIED:
            return EdgeStyleConfig(
                width=1.5,
                label_visible=False,
            )
        else:
            return EdgeStyleConfig()

    def _optimize_nodes(
        self,
        nodes: list[RenderedNode],
        config: RenderConfig,
    ) -> list[RenderedNode]:
        """Optimize nodes for large graphs."""
        # Keep most connected or important nodes
        if len(nodes) <= config.max_visible_nodes:
            return nodes

        # Simple strategy: sample nodes
        step = len(nodes) // config.max_visible_nodes
        return nodes[::step][: config.max_visible_nodes]

    def _optimize_edges(
        self,
        edges: list[RenderedEdge],
        config: RenderConfig,
    ) -> list[RenderedEdge]:
        """Optimize edges for large graphs."""
        if len(edges) <= config.max_visible_edges:
            return edges

        # Keep edges with highest weights
        sorted_edges = sorted(edges, key=lambda e: e.weight, reverse=True)
        return sorted_edges[: config.max_visible_edges]

    def _calculate_bounds(
        self,
        nodes: list[RenderedNode],
        config: RenderConfig,
    ) -> GraphBounds:
        """Calculate bounding box for the graph."""
        if not nodes:
            return GraphBounds()

        x_coords = [n.x for n in nodes]
        y_coords = [n.y for n in nodes]

        min_x = min(x_coords) if x_coords else 0
        max_x = max(x_coords) if x_coords else config.width
        min_y = min(y_coords) if y_coords else 0
        max_y = max(y_coords) if y_coords else config.height

        padding = 50.0

        return GraphBounds(
            min_x=min_x - padding,
            max_x=max_x + padding,
            min_y=min_y - padding,
            max_y=max_y + padding,
            width=max_x - min_x + 2 * padding,
            height=max_y - min_y + 2 * padding,
            padding=padding,
        )

    def _calculate_statistics(
        self,
        nodes: list[RenderedNode],
        edges: list[RenderedEdge],
    ) -> GraphStatistics:
        """Calculate graph statistics."""
        node_count = len(nodes)
        edge_count = len(edges)

        # Calculate density
        max_edges = node_count * (node_count - 1) if node_count > 1 else 1
        density = edge_count / max_edges if max_edges > 0 else 0

        # Calculate average degree
        avg_degree = (2 * edge_count) / node_count if node_count > 0 else 0

        # Count types
        node_types: dict[str, int] = {}
        for node in nodes:
            node_types[node.node_type] = node_types.get(node.node_type, 0) + 1

        edge_types: dict[str, int] = {}
        for edge in edges:
            edge_types[edge.edge_type] = edge_types.get(edge.edge_type, 0) + 1

        return GraphStatistics(
            node_count=node_count,
            edge_count=edge_count,
            density=density,
            avg_degree=avg_degree,
            node_types=node_types,
            edge_types=edge_types,
        )

    def _darken_color(self, hex_color: str, factor: float = 0.7) -> str:
        """Darken a hex color."""
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)

        return f"#{r:02x}{g:02x}{b:02x}"

    def export_to_json(self, graph: RenderedGraph) -> dict[str, Any]:
        """Export rendered graph to JSON format."""
        return graph.model_dump()

    def export_to_cytoscape(self, graph: RenderedGraph) -> dict[str, Any]:
        """Export to Cytoscape.js format."""
        elements = []

        # Add nodes
        for node in graph.nodes:
            elements.append({
                "data": {
                    "id": node.node_id,
                    "label": node.label,
                    "type": node.node_type,
                    **node.properties,
                },
                "position": {"x": node.x, "y": node.y},
                "style": {
                    "background-color": node.style.fill_color,
                    "border-color": node.style.border_color,
                    "border-width": node.style.border_width,
                    "width": node.style.size,
                    "height": node.style.size,
                    "shape": node.style.shape.value,
                },
            })

        # Add edges
        for edge in graph.edges:
            elements.append({
                "data": {
                    "id": edge.edge_id,
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "label": edge.label,
                    "type": edge.edge_type,
                    "weight": edge.weight,
                },
                "style": {
                    "line-color": edge.style.color,
                    "width": edge.style.width,
                    "line-style": edge.style.style.value,
                    "target-arrow-shape": "triangle" if edge.style.show_arrow else "none",
                },
            })

        return {"elements": elements}

    def export_to_d3(self, graph: RenderedGraph) -> dict[str, Any]:
        """Export to D3.js format."""
        nodes = []
        for node in graph.nodes:
            nodes.append({
                "id": node.node_id,
                "label": node.label,
                "group": node.node_type,
                "x": node.x,
                "y": node.y,
                "size": node.style.size,
                "color": node.style.fill_color,
            })

        links = []
        for edge in graph.edges:
            links.append({
                "source": edge.source_id,
                "target": edge.target_id,
                "label": edge.label,
                "value": edge.weight,
            })

        return {"nodes": nodes, "links": links}

    def generate_svg(
        self,
        graph: RenderedGraph,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> str:
        """Generate SVG representation of the graph."""
        w = width or graph.config.width
        h = height or graph.config.height

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{w}" height="{h}" '
            f'viewBox="{graph.bounds.min_x} {graph.bounds.min_y} '
            f'{graph.bounds.width} {graph.bounds.height}">',
            f'<rect width="100%" height="100%" fill="{graph.config.background_color}"/>',
        ]

        # Draw edges
        svg_parts.append('<g class="edges">')
        for edge in graph.edges:
            source_node = next(
                (n for n in graph.nodes if n.node_id == edge.source_id), None
            )
            target_node = next(
                (n for n in graph.nodes if n.node_id == edge.target_id), None
            )

            if source_node and target_node:
                svg_parts.append(
                    f'<line x1="{source_node.x}" y1="{source_node.y}" '
                    f'x2="{target_node.x}" y2="{target_node.y}" '
                    f'stroke="{edge.style.color}" '
                    f'stroke-width="{edge.style.width}" '
                    f'opacity="{edge.style.opacity}"/>'
                )
        svg_parts.append("</g>")

        # Draw nodes
        svg_parts.append('<g class="nodes">')
        for node in graph.nodes:
            if node.style.shape == NodeShape.CIRCLE:
                svg_parts.append(
                    f'<circle cx="{node.x}" cy="{node.y}" '
                    f'r="{node.style.size / 2}" '
                    f'fill="{node.style.fill_color}" '
                    f'stroke="{node.style.border_color}" '
                    f'stroke-width="{node.style.border_width}"/>'
                )
            else:
                half = node.style.size / 2
                svg_parts.append(
                    f'<rect x="{node.x - half}" y="{node.y - half}" '
                    f'width="{node.style.size}" height="{node.style.size}" '
                    f'fill="{node.style.fill_color}" '
                    f'stroke="{node.style.border_color}" '
                    f'stroke-width="{node.style.border_width}"/>'
                )

            if node.style.label_visible:
                svg_parts.append(
                    f'<text x="{node.x}" y="{node.y + node.style.size / 2 + 15}" '
                    f'text-anchor="middle" '
                    f'font-size="{node.style.label_font_size}" '
                    f'fill="{node.style.label_color}">{node.label}</text>'
                )
        svg_parts.append("</g>")

        svg_parts.append("</svg>")

        return "\n".join(svg_parts)


# Global instance
_graph_renderer: Optional[GraphRenderer] = None


def get_graph_renderer() -> GraphRenderer:
    """Get or create the global graph renderer instance."""
    global _graph_renderer
    if _graph_renderer is None:
        _graph_renderer = GraphRenderer()
    return _graph_renderer
