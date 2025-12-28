"""
Centrality Algorithms for Knowledge Graph.

Provides centrality analysis including:
- Degree centrality
- Betweenness centrality
- Closeness centrality
- PageRank algorithm
- Eigenvector centrality
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Tuple
from uuid import UUID

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class Node:
    """Graph node for algorithm computation."""

    node_id: str = ""
    label: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.node_id)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.node_id == other.node_id
        return False


@dataclass
class Edge:
    """Graph edge for algorithm computation."""

    source_id: str = ""
    target_id: str = ""
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Graph:
    """In-memory graph structure for algorithm computation."""

    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    directed: bool = False

    def add_node(self, node_id: str, label: str = "", properties: Optional[Dict] = None) -> Node:
        """Add a node to the graph."""
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(
                node_id=node_id,
                label=label,
                properties=properties or {},
            )
        return self.nodes[node_id]

    def add_edge(self, source_id: str, target_id: str, weight: float = 1.0, properties: Optional[Dict] = None) -> Edge:
        """Add an edge to the graph."""
        edge = Edge(
            source_id=source_id,
            target_id=target_id,
            weight=weight,
            properties=properties or {},
        )
        self.edges.append(edge)
        return edge

    def get_neighbors(self, node_id: str, direction: str = "both") -> List[str]:
        """Get neighboring node IDs."""
        neighbors = []
        for edge in self.edges:
            if direction in ("out", "both") and edge.source_id == node_id:
                neighbors.append(edge.target_id)
            if direction in ("in", "both") and edge.target_id == node_id:
                neighbors.append(edge.source_id)
        return list(set(neighbors))

    def get_adjacency_list(self) -> Dict[str, List[Tuple[str, float]]]:
        """Get adjacency list representation."""
        adj = defaultdict(list)
        for edge in self.edges:
            adj[edge.source_id].append((edge.target_id, edge.weight))
            if not self.directed:
                adj[edge.target_id].append((edge.source_id, edge.weight))
        return dict(adj)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)


@dataclass
class CentralityScore:
    """Centrality score for a node."""

    node_id: str = ""
    score: float = 0.0
    rank: int = 0
    percentile: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "score": self.score,
            "rank": self.rank,
            "percentile": self.percentile,
        }


class CentralityResult(BaseModel):
    """Result of centrality analysis."""

    algorithm: str = Field(default="", description="Algorithm name")
    scores: List[Dict[str, Any]] = Field(default_factory=list, description="Node scores")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Statistics")
    top_nodes: List[Dict[str, Any]] = Field(default_factory=list, description="Top ranked nodes")
    processing_time_ms: float = Field(default=0.0, description="Processing time")


class CentralityAnalyzer:
    """
    Centrality analyzer for graph nodes.

    Implements various centrality algorithms for identifying important nodes.
    """

    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        damping_factor: float = 0.85,
    ):
        """
        Initialize CentralityAnalyzer.

        Args:
            max_iterations: Maximum iterations for iterative algorithms
            tolerance: Convergence tolerance
            damping_factor: Damping factor for PageRank
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.damping_factor = damping_factor
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the analyzer."""
        if self._initialized:
            return
        self._initialized = True
        logger.info("CentralityAnalyzer initialized")

    def degree_centrality(self, graph: Graph, normalized: bool = True) -> CentralityResult:
        """
        Calculate degree centrality for all nodes.

        Degree centrality is the number of edges connected to a node.

        Args:
            graph: Input graph
            normalized: Whether to normalize scores

        Returns:
            CentralityResult with scores
        """
        if not self._initialized:
            self.initialize()

        import time
        start_time = time.time()

        scores = {}
        n = graph.node_count

        # Count degrees
        for node_id in graph.nodes:
            if graph.directed:
                out_degree = sum(1 for e in graph.edges if e.source_id == node_id)
                in_degree = sum(1 for e in graph.edges if e.target_id == node_id)
                degree = out_degree + in_degree
            else:
                degree = len(graph.get_neighbors(node_id))

            if normalized and n > 1:
                scores[node_id] = degree / (n - 1)
            else:
                scores[node_id] = float(degree)

        result = self._build_result("degree_centrality", scores, graph, start_time)
        return result

    def betweenness_centrality(self, graph: Graph, normalized: bool = True) -> CentralityResult:
        """
        Calculate betweenness centrality for all nodes.

        Betweenness centrality measures how often a node lies on shortest paths.

        Args:
            graph: Input graph
            normalized: Whether to normalize scores

        Returns:
            CentralityResult with scores
        """
        if not self._initialized:
            self.initialize()

        import time
        start_time = time.time()

        n = graph.node_count
        scores = {node_id: 0.0 for node_id in graph.nodes}

        # Use Brandes algorithm for efficiency
        for source in graph.nodes:
            # BFS from source
            stack = []
            predecessors = {node_id: [] for node_id in graph.nodes}
            sigma = {node_id: 0.0 for node_id in graph.nodes}
            sigma[source] = 1.0
            distances = {node_id: -1 for node_id in graph.nodes}
            distances[source] = 0

            queue = [source]

            while queue:
                v = queue.pop(0)
                stack.append(v)

                for neighbor in graph.get_neighbors(v):
                    # First visit
                    if distances[neighbor] < 0:
                        queue.append(neighbor)
                        distances[neighbor] = distances[v] + 1

                    # Shortest path to neighbor via v
                    if distances[neighbor] == distances[v] + 1:
                        sigma[neighbor] += sigma[v]
                        predecessors[neighbor].append(v)

            # Accumulation
            delta = {node_id: 0.0 for node_id in graph.nodes}

            while stack:
                w = stack.pop()
                for v in predecessors[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                if w != source:
                    scores[w] += delta[w]

        # Normalize
        if normalized and n > 2:
            if graph.directed:
                scale = 1.0 / ((n - 1) * (n - 2))
            else:
                scale = 2.0 / ((n - 1) * (n - 2))

            for node_id in scores:
                scores[node_id] *= scale

        result = self._build_result("betweenness_centrality", scores, graph, start_time)
        return result

    def closeness_centrality(self, graph: Graph, normalized: bool = True) -> CentralityResult:
        """
        Calculate closeness centrality for all nodes.

        Closeness centrality is the reciprocal of the sum of shortest path distances.

        Args:
            graph: Input graph
            normalized: Whether to normalize scores

        Returns:
            CentralityResult with scores
        """
        if not self._initialized:
            self.initialize()

        import time
        start_time = time.time()

        n = graph.node_count
        scores = {}

        for source in graph.nodes:
            # BFS to find shortest paths
            distances = {source: 0}
            queue = [source]

            while queue:
                v = queue.pop(0)
                for neighbor in graph.get_neighbors(v):
                    if neighbor not in distances:
                        distances[neighbor] = distances[v] + 1
                        queue.append(neighbor)

            # Calculate closeness
            total_distance = sum(distances.values())
            reachable = len(distances)

            if total_distance > 0 and reachable > 1:
                closeness = (reachable - 1) / total_distance
                if normalized:
                    closeness *= (reachable - 1) / (n - 1)
            else:
                closeness = 0.0

            scores[source] = closeness

        result = self._build_result("closeness_centrality", scores, graph, start_time)
        return result

    def pagerank(self, graph: Graph, personalization: Optional[Dict[str, float]] = None) -> CentralityResult:
        """
        Calculate PageRank for all nodes.

        PageRank measures node importance based on link structure.

        Args:
            graph: Input graph
            personalization: Optional personalization vector

        Returns:
            CentralityResult with scores
        """
        if not self._initialized:
            self.initialize()

        import time
        start_time = time.time()

        n = graph.node_count
        if n == 0:
            return CentralityResult(algorithm="pagerank", scores=[], statistics={})

        # Initialize scores
        if personalization:
            total = sum(personalization.values())
            scores = {node_id: personalization.get(node_id, 0) / total for node_id in graph.nodes}
        else:
            scores = {node_id: 1.0 / n for node_id in graph.nodes}

        # Build adjacency list
        adj = graph.get_adjacency_list()

        # Calculate out-degrees
        out_degree = defaultdict(int)
        for edge in graph.edges:
            out_degree[edge.source_id] += 1

        # Iterative PageRank
        for iteration in range(self.max_iterations):
            prev_scores = scores.copy()

            # Calculate new scores
            for node_id in graph.nodes:
                incoming_score = 0.0

                # Sum contributions from incoming edges
                for edge in graph.edges:
                    if edge.target_id == node_id:
                        source_out_degree = out_degree[edge.source_id]
                        if source_out_degree > 0:
                            incoming_score += prev_scores[edge.source_id] / source_out_degree

                # Apply damping factor
                if personalization:
                    personal_weight = personalization.get(node_id, 0) / sum(personalization.values())
                else:
                    personal_weight = 1.0 / n

                scores[node_id] = (1 - self.damping_factor) * personal_weight + self.damping_factor * incoming_score

            # Check convergence
            diff = sum(abs(scores[n] - prev_scores[n]) for n in graph.nodes)
            if diff < self.tolerance:
                logger.info(f"PageRank converged after {iteration + 1} iterations")
                break

        result = self._build_result("pagerank", scores, graph, start_time)
        return result

    def eigenvector_centrality(self, graph: Graph) -> CentralityResult:
        """
        Calculate eigenvector centrality for all nodes.

        Eigenvector centrality measures influence based on connections to high-scoring nodes.

        Args:
            graph: Input graph

        Returns:
            CentralityResult with scores
        """
        if not self._initialized:
            self.initialize()

        import time
        start_time = time.time()

        n = graph.node_count
        if n == 0:
            return CentralityResult(algorithm="eigenvector_centrality", scores=[], statistics={})

        # Initialize scores
        scores = {node_id: 1.0 / n for node_id in graph.nodes}

        # Power iteration
        for iteration in range(self.max_iterations):
            prev_scores = scores.copy()

            # Calculate new scores
            for node_id in graph.nodes:
                neighbor_sum = 0.0
                for neighbor in graph.get_neighbors(node_id):
                    neighbor_sum += prev_scores[neighbor]
                scores[node_id] = neighbor_sum

            # Normalize
            norm = math.sqrt(sum(s * s for s in scores.values()))
            if norm > 0:
                scores = {n: s / norm for n, s in scores.items()}

            # Check convergence
            diff = sum(abs(scores[n] - prev_scores[n]) for n in graph.nodes)
            if diff < self.tolerance:
                logger.info(f"Eigenvector centrality converged after {iteration + 1} iterations")
                break

        result = self._build_result("eigenvector_centrality", scores, graph, start_time)
        return result

    def katz_centrality(self, graph: Graph, alpha: float = 0.1, beta: float = 1.0) -> CentralityResult:
        """
        Calculate Katz centrality for all nodes.

        Katz centrality measures influence with attenuation for path length.

        Args:
            graph: Input graph
            alpha: Attenuation factor
            beta: Weight for immediate connections

        Returns:
            CentralityResult with scores
        """
        if not self._initialized:
            self.initialize()

        import time
        start_time = time.time()

        n = graph.node_count
        if n == 0:
            return CentralityResult(algorithm="katz_centrality", scores=[], statistics={})

        # Initialize scores
        scores = {node_id: 0.0 for node_id in graph.nodes}

        # Power iteration
        for iteration in range(self.max_iterations):
            prev_scores = scores.copy()

            # Calculate new scores
            for node_id in graph.nodes:
                neighbor_sum = 0.0
                for neighbor in graph.get_neighbors(node_id, direction="in"):
                    neighbor_sum += prev_scores[neighbor]
                scores[node_id] = alpha * neighbor_sum + beta

            # Check convergence
            diff = sum(abs(scores[n] - prev_scores[n]) for n in graph.nodes)
            if diff < self.tolerance:
                logger.info(f"Katz centrality converged after {iteration + 1} iterations")
                break

        result = self._build_result("katz_centrality", scores, graph, start_time)
        return result

    def analyze_all(self, graph: Graph) -> Dict[str, CentralityResult]:
        """
        Run all centrality algorithms on the graph.

        Args:
            graph: Input graph

        Returns:
            Dictionary of algorithm name to CentralityResult
        """
        return {
            "degree": self.degree_centrality(graph),
            "betweenness": self.betweenness_centrality(graph),
            "closeness": self.closeness_centrality(graph),
            "pagerank": self.pagerank(graph),
            "eigenvector": self.eigenvector_centrality(graph),
        }

    def _build_result(
        self,
        algorithm: str,
        scores: Dict[str, float],
        graph: Graph,
        start_time: float,
    ) -> CentralityResult:
        """Build CentralityResult from scores."""
        import time

        # Sort by score descending
        sorted_items = sorted(scores.items(), key=lambda x: -x[1])

        # Build score objects with ranks
        score_objects = []
        for rank, (node_id, score) in enumerate(sorted_items, 1):
            percentile = (len(sorted_items) - rank) / len(sorted_items) * 100 if sorted_items else 0
            score_objects.append(CentralityScore(
                node_id=node_id,
                score=score,
                rank=rank,
                percentile=percentile,
            ))

        # Calculate statistics
        score_values = list(scores.values())
        statistics = {}
        if score_values:
            statistics = {
                "min": min(score_values),
                "max": max(score_values),
                "mean": sum(score_values) / len(score_values),
                "std": self._calculate_std(score_values),
                "node_count": len(score_values),
            }

        # Top nodes
        top_nodes = [s.to_dict() for s in score_objects[:10]]

        processing_time = (time.time() - start_time) * 1000

        return CentralityResult(
            algorithm=algorithm,
            scores=[s.to_dict() for s in score_objects],
            statistics=statistics,
            top_nodes=top_nodes,
            processing_time_ms=processing_time,
        )

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return math.sqrt(variance)


# Global instance
_centrality_analyzer: Optional[CentralityAnalyzer] = None


def get_centrality_analyzer() -> CentralityAnalyzer:
    """Get or create global CentralityAnalyzer instance."""
    global _centrality_analyzer

    if _centrality_analyzer is None:
        _centrality_analyzer = CentralityAnalyzer()
        _centrality_analyzer.initialize()

    return _centrality_analyzer
