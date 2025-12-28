"""
Layout engine for Knowledge Graph visualization.

Provides force-directed, hierarchical, circular, and other layout algorithms.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LayoutType(str, Enum):
    """Types of layout algorithms."""

    FORCE_DIRECTED = "force_directed"
    HIERARCHICAL = "hierarchical"
    CIRCULAR = "circular"
    GRID = "grid"
    RADIAL = "radial"
    TREE = "tree"
    RANDOM = "random"
    SPECTRAL = "spectral"
    KAMADA_KAWAI = "kamada_kawai"


class LayoutDirection(str, Enum):
    """Direction for hierarchical layouts."""

    TOP_DOWN = "top_down"
    BOTTOM_UP = "bottom_up"
    LEFT_RIGHT = "left_right"
    RIGHT_LEFT = "right_left"


class LayoutConfig(BaseModel):
    """Configuration for layout algorithms."""

    layout_type: LayoutType = Field(
        default=LayoutType.FORCE_DIRECTED, description="Layout algorithm"
    )
    width: float = Field(default=1000.0, description="Canvas width")
    height: float = Field(default=800.0, description="Canvas height")
    padding: float = Field(default=50.0, description="Padding from edges")
    iterations: int = Field(default=100, description="Iteration count for iterative layouts")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")

    # Force-directed specific
    repulsion_force: float = Field(default=1000.0, description="Node repulsion strength")
    attraction_force: float = Field(default=0.01, description="Edge attraction strength")
    damping: float = Field(default=0.85, description="Velocity damping factor")
    min_distance: float = Field(default=30.0, description="Minimum node distance")

    # Hierarchical specific
    direction: LayoutDirection = Field(default=LayoutDirection.TOP_DOWN)
    level_separation: float = Field(default=100.0, description="Separation between levels")
    node_separation: float = Field(default=50.0, description="Separation between nodes")

    # Circular specific
    radius: Optional[float] = Field(default=None, description="Circle radius (auto if None)")
    start_angle: float = Field(default=0.0, description="Starting angle in radians")

    # Grid specific
    columns: Optional[int] = Field(default=None, description="Number of columns (auto if None)")
    cell_width: float = Field(default=100.0, description="Grid cell width")
    cell_height: float = Field(default=100.0, description="Grid cell height")


class NodePosition(BaseModel):
    """Position of a node."""

    node_id: str = Field(..., description="Node identifier")
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    fixed: bool = Field(default=False, description="Whether position is fixed")


class LayoutResult(BaseModel):
    """Result of layout computation."""

    positions: list[NodePosition] = Field(default_factory=list)
    bounds: dict[str, float] = Field(default_factory=dict)
    iterations_used: int = Field(default=0)
    converged: bool = Field(default=False)
    layout_type: LayoutType = Field(default=LayoutType.FORCE_DIRECTED)
    config: LayoutConfig = Field(default_factory=LayoutConfig)


@dataclass
class LayoutEngine:
    """Layout engine for graph visualization."""

    config: LayoutConfig = field(default_factory=LayoutConfig)
    positions: dict[str, tuple[float, float]] = field(default_factory=dict)
    velocities: dict[str, tuple[float, float]] = field(default_factory=dict)

    def compute_layout(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        config: Optional[LayoutConfig] = None,
    ) -> LayoutResult:
        """Compute layout for nodes and edges."""
        if config:
            self.config = config

        # Set random seed if specified
        if self.config.seed is not None:
            random.seed(self.config.seed)

        # Dispatch to appropriate algorithm
        if self.config.layout_type == LayoutType.FORCE_DIRECTED:
            return self._force_directed_layout(nodes, edges)
        elif self.config.layout_type == LayoutType.HIERARCHICAL:
            return self._hierarchical_layout(nodes, edges)
        elif self.config.layout_type == LayoutType.CIRCULAR:
            return self._circular_layout(nodes, edges)
        elif self.config.layout_type == LayoutType.GRID:
            return self._grid_layout(nodes, edges)
        elif self.config.layout_type == LayoutType.RADIAL:
            return self._radial_layout(nodes, edges)
        elif self.config.layout_type == LayoutType.RANDOM:
            return self._random_layout(nodes, edges)
        else:
            return self._force_directed_layout(nodes, edges)

    def _force_directed_layout(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> LayoutResult:
        """Compute force-directed layout using Fruchterman-Reingold algorithm."""
        if not nodes:
            return LayoutResult(layout_type=LayoutType.FORCE_DIRECTED, config=self.config)

        # Initialize positions randomly
        for node in nodes:
            node_id = self._get_node_id(node)
            if node_id not in self.positions:
                x = random.uniform(self.config.padding, self.config.width - self.config.padding)
                y = random.uniform(self.config.padding, self.config.height - self.config.padding)
                self.positions[node_id] = (x, y)
                self.velocities[node_id] = (0.0, 0.0)

        # Build adjacency
        adjacency: dict[str, set[str]] = {self._get_node_id(n): set() for n in nodes}
        for edge in edges:
            source = str(edge.get("source", edge.get("source_id", edge.get("from", ""))))
            target = str(edge.get("target", edge.get("target_id", edge.get("to", ""))))
            if source in adjacency and target in adjacency:
                adjacency[source].add(target)
                adjacency[target].add(source)

        # Optimal distance
        area = self.config.width * self.config.height
        k = math.sqrt(area / len(nodes))

        converged = False
        for iteration in range(self.config.iterations):
            forces: dict[str, tuple[float, float]] = {}

            # Calculate repulsive forces
            for node in nodes:
                node_id = self._get_node_id(node)
                fx, fy = 0.0, 0.0
                x1, y1 = self.positions[node_id]

                for other_node in nodes:
                    other_id = self._get_node_id(other_node)
                    if node_id == other_id:
                        continue

                    x2, y2 = self.positions[other_id]
                    dx = x1 - x2
                    dy = y1 - y2
                    dist = max(math.sqrt(dx * dx + dy * dy), 0.01)

                    # Repulsion force
                    repulsion = self.config.repulsion_force / (dist * dist)
                    fx += (dx / dist) * repulsion
                    fy += (dy / dist) * repulsion

                forces[node_id] = (fx, fy)

            # Calculate attractive forces
            for edge in edges:
                source = str(edge.get("source", edge.get("source_id", edge.get("from", ""))))
                target = str(edge.get("target", edge.get("target_id", edge.get("to", ""))))

                if source not in self.positions or target not in self.positions:
                    continue

                x1, y1 = self.positions[source]
                x2, y2 = self.positions[target]
                dx = x2 - x1
                dy = y2 - y1
                dist = max(math.sqrt(dx * dx + dy * dy), 0.01)

                # Attraction force
                attraction = self.config.attraction_force * dist

                fx1, fy1 = forces.get(source, (0, 0))
                fx2, fy2 = forces.get(target, (0, 0))

                forces[source] = (fx1 + (dx / dist) * attraction, fy1 + (dy / dist) * attraction)
                forces[target] = (fx2 - (dx / dist) * attraction, fy2 - (dy / dist) * attraction)

            # Apply forces with damping
            max_displacement = 0.0
            for node_id, (fx, fy) in forces.items():
                vx, vy = self.velocities[node_id]
                vx = (vx + fx) * self.config.damping
                vy = (vy + fy) * self.config.damping

                # Limit velocity
                speed = math.sqrt(vx * vx + vy * vy)
                max_speed = 50.0
                if speed > max_speed:
                    vx = (vx / speed) * max_speed
                    vy = (vy / speed) * max_speed

                self.velocities[node_id] = (vx, vy)

                x, y = self.positions[node_id]
                new_x = x + vx
                new_y = y + vy

                # Keep within bounds
                new_x = max(self.config.padding, min(self.config.width - self.config.padding, new_x))
                new_y = max(self.config.padding, min(self.config.height - self.config.padding, new_y))

                displacement = math.sqrt((new_x - x) ** 2 + (new_y - y) ** 2)
                max_displacement = max(max_displacement, displacement)

                self.positions[node_id] = (new_x, new_y)

            # Check convergence
            if max_displacement < 0.1:
                converged = True
                break

        return self._create_result(nodes, LayoutType.FORCE_DIRECTED, iteration + 1, converged)

    def _hierarchical_layout(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> LayoutResult:
        """Compute hierarchical layout."""
        if not nodes:
            return LayoutResult(layout_type=LayoutType.HIERARCHICAL, config=self.config)

        # Build graph and find levels
        node_ids = [self._get_node_id(n) for n in nodes]
        children: dict[str, list[str]] = {nid: [] for nid in node_ids}
        parents: dict[str, Optional[str]] = {nid: None for nid in node_ids}

        for edge in edges:
            source = str(edge.get("source", edge.get("source_id", edge.get("from", ""))))
            target = str(edge.get("target", edge.get("target_id", edge.get("to", ""))))

            if source in children and target in node_ids:
                children[source].append(target)
                parents[target] = source

        # Find roots (nodes without parents)
        roots = [nid for nid, parent in parents.items() if parent is None]
        if not roots:
            roots = [node_ids[0]] if node_ids else []

        # Assign levels
        levels: dict[str, int] = {}
        visited = set()

        def assign_level(node_id: str, level: int) -> None:
            if node_id in visited:
                return
            visited.add(node_id)
            levels[node_id] = level
            for child in children.get(node_id, []):
                assign_level(child, level + 1)

        for root in roots:
            assign_level(root, 0)

        # Assign remaining nodes
        for nid in node_ids:
            if nid not in levels:
                levels[nid] = 0

        # Group by level
        level_nodes: dict[int, list[str]] = {}
        for nid, level in levels.items():
            if level not in level_nodes:
                level_nodes[level] = []
            level_nodes[level].append(nid)

        # Calculate positions
        max_level = max(level_nodes.keys()) if level_nodes else 0

        for level, nodes_at_level in level_nodes.items():
            n = len(nodes_at_level)
            total_width = (n - 1) * self.config.node_separation

            start_x = (self.config.width - total_width) / 2

            if self.config.direction == LayoutDirection.TOP_DOWN:
                y = self.config.padding + level * self.config.level_separation
            elif self.config.direction == LayoutDirection.BOTTOM_UP:
                y = self.config.height - self.config.padding - level * self.config.level_separation
            else:
                y = self.config.height / 2

            for i, node_id in enumerate(nodes_at_level):
                if self.config.direction in [LayoutDirection.LEFT_RIGHT, LayoutDirection.RIGHT_LEFT]:
                    x = self.config.padding + level * self.config.level_separation
                    y = start_x + i * self.config.node_separation
                    if self.config.direction == LayoutDirection.RIGHT_LEFT:
                        x = self.config.width - x
                else:
                    x = start_x + i * self.config.node_separation

                self.positions[node_id] = (x, y)

        return self._create_result(nodes, LayoutType.HIERARCHICAL, 1, True)

    def _circular_layout(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> LayoutResult:
        """Compute circular layout."""
        if not nodes:
            return LayoutResult(layout_type=LayoutType.CIRCULAR, config=self.config)

        n = len(nodes)
        center_x = self.config.width / 2
        center_y = self.config.height / 2

        # Calculate radius
        if self.config.radius:
            radius = self.config.radius
        else:
            radius = min(self.config.width, self.config.height) / 2 - self.config.padding

        # Place nodes in circle
        angle_step = 2 * math.pi / n

        for i, node in enumerate(nodes):
            node_id = self._get_node_id(node)
            angle = self.config.start_angle + i * angle_step
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            self.positions[node_id] = (x, y)

        return self._create_result(nodes, LayoutType.CIRCULAR, 1, True)

    def _grid_layout(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> LayoutResult:
        """Compute grid layout."""
        if not nodes:
            return LayoutResult(layout_type=LayoutType.GRID, config=self.config)

        n = len(nodes)

        # Calculate columns
        if self.config.columns:
            cols = self.config.columns
        else:
            cols = max(1, int(math.ceil(math.sqrt(n))))

        rows = max(1, int(math.ceil(n / cols)))

        # Calculate starting position
        total_width = (cols - 1) * self.config.cell_width
        total_height = (rows - 1) * self.config.cell_height
        start_x = (self.config.width - total_width) / 2
        start_y = (self.config.height - total_height) / 2

        # Place nodes
        for i, node in enumerate(nodes):
            node_id = self._get_node_id(node)
            row = i // cols
            col = i % cols
            x = start_x + col * self.config.cell_width
            y = start_y + row * self.config.cell_height
            self.positions[node_id] = (x, y)

        return self._create_result(nodes, LayoutType.GRID, 1, True)

    def _radial_layout(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> LayoutResult:
        """Compute radial layout with center node."""
        if not nodes:
            return LayoutResult(layout_type=LayoutType.RADIAL, config=self.config)

        center_x = self.config.width / 2
        center_y = self.config.height / 2

        # Build adjacency and find degrees
        node_ids = [self._get_node_id(n) for n in nodes]
        degrees: dict[str, int] = {nid: 0 for nid in node_ids}

        for edge in edges:
            source = str(edge.get("source", edge.get("source_id", edge.get("from", ""))))
            target = str(edge.get("target", edge.get("target_id", edge.get("to", ""))))
            if source in degrees:
                degrees[source] += 1
            if target in degrees:
                degrees[target] += 1

        # Sort by degree (most connected at center)
        sorted_nodes = sorted(node_ids, key=lambda x: degrees.get(x, 0), reverse=True)

        # Place central node
        if sorted_nodes:
            self.positions[sorted_nodes[0]] = (center_x, center_y)

        # Place remaining nodes in concentric circles
        remaining = sorted_nodes[1:]
        radius = self.config.min_distance * 2
        placed = 0

        while placed < len(remaining):
            circumference = 2 * math.pi * radius
            nodes_in_ring = max(1, int(circumference / self.config.min_distance))
            nodes_in_ring = min(nodes_in_ring, len(remaining) - placed)

            angle_step = 2 * math.pi / nodes_in_ring

            for i in range(nodes_in_ring):
                if placed >= len(remaining):
                    break

                angle = i * angle_step
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                self.positions[remaining[placed]] = (x, y)
                placed += 1

            radius += self.config.min_distance * 1.5

        return self._create_result(nodes, LayoutType.RADIAL, 1, True)

    def _random_layout(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> LayoutResult:
        """Compute random layout."""
        for node in nodes:
            node_id = self._get_node_id(node)
            x = random.uniform(self.config.padding, self.config.width - self.config.padding)
            y = random.uniform(self.config.padding, self.config.height - self.config.padding)
            self.positions[node_id] = (x, y)

        return self._create_result(nodes, LayoutType.RANDOM, 1, True)

    def _get_node_id(self, node: dict[str, Any]) -> str:
        """Extract node ID from node dict."""
        return str(node.get("id", node.get("node_id", "")))

    def _create_result(
        self,
        nodes: list[dict[str, Any]],
        layout_type: LayoutType,
        iterations: int,
        converged: bool,
    ) -> LayoutResult:
        """Create layout result from current positions."""
        positions = []
        for node in nodes:
            node_id = self._get_node_id(node)
            x, y = self.positions.get(node_id, (0, 0))
            positions.append(
                NodePosition(
                    node_id=node_id,
                    x=x,
                    y=y,
                )
            )

        # Calculate bounds
        if positions:
            xs = [p.x for p in positions]
            ys = [p.y for p in positions]
            bounds = {
                "min_x": min(xs),
                "max_x": max(xs),
                "min_y": min(ys),
                "max_y": max(ys),
            }
        else:
            bounds = {"min_x": 0, "max_x": self.config.width, "min_y": 0, "max_y": self.config.height}

        return LayoutResult(
            positions=positions,
            bounds=bounds,
            iterations_used=iterations,
            converged=converged,
            layout_type=layout_type,
            config=self.config,
        )

    def apply_layout(
        self,
        layout_result: LayoutResult,
        nodes: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply layout positions to nodes."""
        position_map = {p.node_id: (p.x, p.y) for p in layout_result.positions}

        for node in nodes:
            node_id = self._get_node_id(node)
            if node_id in position_map:
                x, y = position_map[node_id]
                node["x"] = x
                node["y"] = y

        return nodes

    def reset(self) -> None:
        """Reset all positions."""
        self.positions.clear()
        self.velocities.clear()


# Global instance
_layout_engine: Optional[LayoutEngine] = None


def get_layout_engine() -> LayoutEngine:
    """Get or create the global layout engine instance."""
    global _layout_engine
    if _layout_engine is None:
        _layout_engine = LayoutEngine()
    return _layout_engine
