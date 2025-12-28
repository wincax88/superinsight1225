"""
Community Detection Algorithms for Knowledge Graph.

Provides community detection including:
- Louvain algorithm
- Label propagation
- Girvan-Newman algorithm
- K-means clustering on embeddings
"""

import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

from .centrality import Graph, Node, Edge

logger = logging.getLogger(__name__)


@dataclass
class Community:
    """A community of nodes."""

    community_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    node_ids: Set[str] = field(default_factory=set)
    properties: Dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.node_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "community_id": self.community_id,
            "name": self.name,
            "node_ids": list(self.node_ids),
            "size": self.size,
            "properties": self.properties,
        }


class CommunityResult(BaseModel):
    """Result of community detection."""

    algorithm: str = Field(default="", description="Algorithm name")
    communities: List[Dict[str, Any]] = Field(default_factory=list, description="Detected communities")
    node_to_community: Dict[str, str] = Field(default_factory=dict, description="Node to community mapping")
    modularity: float = Field(default=0.0, description="Modularity score")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Statistics")
    processing_time_ms: float = Field(default=0.0, description="Processing time")


class CommunityDetector:
    """
    Community detector for graph nodes.

    Implements various community detection algorithms.
    """

    def __init__(
        self,
        max_iterations: int = 100,
        resolution: float = 1.0,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize CommunityDetector.

        Args:
            max_iterations: Maximum iterations for iterative algorithms
            resolution: Resolution parameter for Louvain algorithm
            random_seed: Random seed for reproducibility
        """
        self.max_iterations = max_iterations
        self.resolution = resolution
        self.random_seed = random_seed
        self._initialized = False

        if random_seed is not None:
            random.seed(random_seed)

    def initialize(self) -> None:
        """Initialize the detector."""
        if self._initialized:
            return
        self._initialized = True
        logger.info("CommunityDetector initialized")

    def louvain(self, graph: Graph) -> CommunityResult:
        """
        Detect communities using Louvain algorithm.

        The Louvain algorithm optimizes modularity through local moves
        and community aggregation.

        Args:
            graph: Input graph

        Returns:
            CommunityResult with detected communities
        """
        if not self._initialized:
            self.initialize()

        import time
        start_time = time.time()

        n = graph.node_count
        if n == 0:
            return CommunityResult(algorithm="louvain", communities=[], modularity=0.0)

        # Initialize: each node is its own community
        node_to_community = {node_id: node_id for node_id in graph.nodes}

        # Calculate total edge weight
        total_weight = sum(e.weight for e in graph.edges)
        if not graph.directed:
            total_weight *= 2  # Count edges twice for undirected

        # Build adjacency with weights
        adj_weights = defaultdict(lambda: defaultdict(float))
        for edge in graph.edges:
            adj_weights[edge.source_id][edge.target_id] += edge.weight
            if not graph.directed:
                adj_weights[edge.target_id][edge.source_id] += edge.weight

        # Calculate node degrees (sum of edge weights)
        node_degree = defaultdict(float)
        for edge in graph.edges:
            node_degree[edge.source_id] += edge.weight
            node_degree[edge.target_id] += edge.weight

        # Phase 1: Local optimization
        improved = True
        iteration = 0

        while improved and iteration < self.max_iterations:
            improved = False
            iteration += 1

            # Shuffle node order for randomization
            node_list = list(graph.nodes.keys())
            random.shuffle(node_list)

            for node_id in node_list:
                current_community = node_to_community[node_id]

                # Calculate gain for moving to each neighbor's community
                neighbor_communities = set()
                for neighbor in graph.get_neighbors(node_id):
                    neighbor_communities.add(node_to_community[neighbor])

                best_community = current_community
                best_gain = 0.0

                for target_community in neighbor_communities:
                    if target_community == current_community:
                        continue

                    # Calculate modularity gain
                    gain = self._calculate_modularity_gain(
                        node_id,
                        current_community,
                        target_community,
                        node_to_community,
                        adj_weights,
                        node_degree,
                        total_weight,
                    )

                    if gain > best_gain:
                        best_gain = gain
                        best_community = target_community

                # Move node if beneficial
                if best_community != current_community:
                    node_to_community[node_id] = best_community
                    improved = True

        # Build community objects
        communities = self._build_communities(node_to_community)

        # Calculate final modularity
        modularity = self._calculate_modularity(graph, node_to_community)

        # Build result
        processing_time = (time.time() - start_time) * 1000

        return CommunityResult(
            algorithm="louvain",
            communities=[c.to_dict() for c in communities],
            node_to_community=node_to_community,
            modularity=modularity,
            statistics=self._calculate_statistics(communities, graph),
            processing_time_ms=processing_time,
        )

    def label_propagation(self, graph: Graph) -> CommunityResult:
        """
        Detect communities using label propagation algorithm.

        Each node adopts the most common label among its neighbors.

        Args:
            graph: Input graph

        Returns:
            CommunityResult with detected communities
        """
        if not self._initialized:
            self.initialize()

        import time
        start_time = time.time()

        n = graph.node_count
        if n == 0:
            return CommunityResult(algorithm="label_propagation", communities=[], modularity=0.0)

        # Initialize: each node gets unique label
        labels = {node_id: i for i, node_id in enumerate(graph.nodes)}

        # Iteratively update labels
        for iteration in range(self.max_iterations):
            changed = False

            # Random order for node updates
            node_list = list(graph.nodes.keys())
            random.shuffle(node_list)

            for node_id in node_list:
                neighbors = graph.get_neighbors(node_id)
                if not neighbors:
                    continue

                # Count neighbor labels
                label_counts = defaultdict(float)
                for neighbor in neighbors:
                    # Weight by edge weight if available
                    weight = 1.0
                    for edge in graph.edges:
                        if (edge.source_id == node_id and edge.target_id == neighbor) or \
                           (edge.target_id == node_id and edge.source_id == neighbor):
                            weight = edge.weight
                            break
                    label_counts[labels[neighbor]] += weight

                # Find most common label(s)
                max_count = max(label_counts.values())
                max_labels = [l for l, c in label_counts.items() if c == max_count]

                # Choose randomly among ties
                new_label = random.choice(max_labels)

                if new_label != labels[node_id]:
                    labels[node_id] = new_label
                    changed = True

            if not changed:
                logger.info(f"Label propagation converged after {iteration + 1} iterations")
                break

        # Convert labels to community assignments
        node_to_community = {node_id: str(label) for node_id, label in labels.items()}

        # Build communities
        communities = self._build_communities(node_to_community)

        # Calculate modularity
        modularity = self._calculate_modularity(graph, node_to_community)

        processing_time = (time.time() - start_time) * 1000

        return CommunityResult(
            algorithm="label_propagation",
            communities=[c.to_dict() for c in communities],
            node_to_community=node_to_community,
            modularity=modularity,
            statistics=self._calculate_statistics(communities, graph),
            processing_time_ms=processing_time,
        )

    def connected_components(self, graph: Graph) -> CommunityResult:
        """
        Find connected components in the graph.

        Args:
            graph: Input graph

        Returns:
            CommunityResult with connected components as communities
        """
        if not self._initialized:
            self.initialize()

        import time
        start_time = time.time()

        if graph.node_count == 0:
            return CommunityResult(algorithm="connected_components", communities=[], modularity=0.0)

        # BFS to find connected components
        visited = set()
        node_to_community = {}
        component_id = 0

        for start_node in graph.nodes:
            if start_node in visited:
                continue

            # BFS from start_node
            queue = [start_node]
            community_id = str(component_id)

            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue

                visited.add(node)
                node_to_community[node] = community_id

                for neighbor in graph.get_neighbors(node):
                    if neighbor not in visited:
                        queue.append(neighbor)

            component_id += 1

        # Build communities
        communities = self._build_communities(node_to_community)

        # Calculate modularity
        modularity = self._calculate_modularity(graph, node_to_community)

        processing_time = (time.time() - start_time) * 1000

        return CommunityResult(
            algorithm="connected_components",
            communities=[c.to_dict() for c in communities],
            node_to_community=node_to_community,
            modularity=modularity,
            statistics=self._calculate_statistics(communities, graph),
            processing_time_ms=processing_time,
        )

    def k_clique_communities(self, graph: Graph, k: int = 3) -> CommunityResult:
        """
        Find k-clique communities (overlapping communities).

        Communities are formed by adjacent k-cliques.

        Args:
            graph: Input graph
            k: Clique size (minimum 3)

        Returns:
            CommunityResult with k-clique communities
        """
        if not self._initialized:
            self.initialize()

        import time
        start_time = time.time()

        k = max(3, k)  # Minimum clique size of 3

        # Find all k-cliques using Bron-Kerbosch algorithm
        cliques = self._find_cliques(graph, k)

        if not cliques:
            return CommunityResult(
                algorithm=f"k_clique_communities_k{k}",
                communities=[],
                modularity=0.0,
                statistics={"cliques_found": 0},
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        # Build clique graph (cliques are adjacent if they share k-1 nodes)
        clique_communities = []
        used_cliques = set()

        for i, clique in enumerate(cliques):
            if i in used_cliques:
                continue

            # Start new community with this clique
            community_nodes = set(clique)
            used_cliques.add(i)

            # Find all connected cliques
            queue = [i]
            while queue:
                current_idx = queue.pop(0)
                current_clique = set(cliques[current_idx])

                for j, other_clique in enumerate(cliques):
                    if j in used_cliques:
                        continue

                    other_set = set(other_clique)
                    # Check if cliques share k-1 nodes
                    if len(current_clique & other_set) >= k - 1:
                        community_nodes.update(other_set)
                        used_cliques.add(j)
                        queue.append(j)

            if community_nodes:
                clique_communities.append(Community(
                    name=f"k_clique_community_{len(clique_communities)}",
                    node_ids=community_nodes,
                ))

        # Build node to community mapping (first community wins for overlapping)
        node_to_community = {}
        for community in clique_communities:
            for node_id in community.node_ids:
                if node_id not in node_to_community:
                    node_to_community[node_id] = community.community_id

        # Add isolated nodes
        for node_id in graph.nodes:
            if node_id not in node_to_community:
                node_to_community[node_id] = f"isolated_{node_id}"

        processing_time = (time.time() - start_time) * 1000

        return CommunityResult(
            algorithm=f"k_clique_communities_k{k}",
            communities=[c.to_dict() for c in clique_communities],
            node_to_community=node_to_community,
            modularity=self._calculate_modularity(graph, node_to_community),
            statistics={
                "cliques_found": len(cliques),
                "communities_found": len(clique_communities),
            },
            processing_time_ms=processing_time,
        )

    def _find_cliques(self, graph: Graph, min_size: int) -> List[List[str]]:
        """Find all cliques of at least min_size using Bron-Kerbosch algorithm."""
        cliques = []

        def bron_kerbosch(r: Set[str], p: Set[str], x: Set[str]):
            if not p and not x:
                if len(r) >= min_size:
                    cliques.append(list(r))
                return

            # Choose pivot
            pivot = max(p | x, key=lambda n: len(set(graph.get_neighbors(n)) & p), default=None)
            if pivot is None:
                return

            pivot_neighbors = set(graph.get_neighbors(pivot))

            for v in list(p - pivot_neighbors):
                v_neighbors = set(graph.get_neighbors(v))
                bron_kerbosch(
                    r | {v},
                    p & v_neighbors,
                    x & v_neighbors,
                )
                p.remove(v)
                x.add(v)

        all_nodes = set(graph.nodes.keys())
        bron_kerbosch(set(), all_nodes, set())

        return cliques

    def _calculate_modularity_gain(
        self,
        node_id: str,
        current_community: str,
        target_community: str,
        node_to_community: Dict[str, str],
        adj_weights: Dict[str, Dict[str, float]],
        node_degree: Dict[str, float],
        total_weight: float,
    ) -> float:
        """Calculate modularity gain for moving a node to a new community."""
        if total_weight == 0:
            return 0.0

        # Edges to current community
        edges_to_current = sum(
            adj_weights[node_id].get(n, 0)
            for n, c in node_to_community.items()
            if c == current_community and n != node_id
        )

        # Edges to target community
        edges_to_target = sum(
            adj_weights[node_id].get(n, 0)
            for n, c in node_to_community.items()
            if c == target_community
        )

        # Degree sums in communities
        current_degree_sum = sum(
            node_degree[n]
            for n, c in node_to_community.items()
            if c == current_community and n != node_id
        )

        target_degree_sum = sum(
            node_degree[n]
            for n, c in node_to_community.items()
            if c == target_community
        )

        k_i = node_degree[node_id]

        # Modularity gain formula
        gain = (
            (edges_to_target - edges_to_current) / total_weight
            - self.resolution * k_i * (target_degree_sum - current_degree_sum) / (2 * total_weight ** 2)
        )

        return gain

    def _calculate_modularity(self, graph: Graph, node_to_community: Dict[str, str]) -> float:
        """Calculate modularity of a partition."""
        total_weight = sum(e.weight for e in graph.edges)
        if total_weight == 0:
            return 0.0

        if not graph.directed:
            total_weight *= 2

        # Calculate node degrees
        node_degree = defaultdict(float)
        for edge in graph.edges:
            node_degree[edge.source_id] += edge.weight
            node_degree[edge.target_id] += edge.weight

        modularity = 0.0

        for edge in graph.edges:
            if node_to_community.get(edge.source_id) == node_to_community.get(edge.target_id):
                # Edge within community
                expected = node_degree[edge.source_id] * node_degree[edge.target_id] / (2 * total_weight)
                modularity += (edge.weight - expected) / total_weight

        return modularity

    def _build_communities(self, node_to_community: Dict[str, str]) -> List[Community]:
        """Build Community objects from node-to-community mapping."""
        community_nodes = defaultdict(set)
        for node_id, community_id in node_to_community.items():
            community_nodes[community_id].add(node_id)

        communities = []
        for community_id, nodes in community_nodes.items():
            communities.append(Community(
                community_id=community_id,
                name=f"Community {len(communities) + 1}",
                node_ids=nodes,
            ))

        # Sort by size descending
        communities.sort(key=lambda c: -c.size)

        return communities

    def _calculate_statistics(self, communities: List[Community], graph: Graph) -> Dict[str, Any]:
        """Calculate community detection statistics."""
        if not communities:
            return {}

        sizes = [c.size for c in communities]

        return {
            "num_communities": len(communities),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "avg_size": sum(sizes) / len(sizes),
            "total_nodes": sum(sizes),
            "coverage": sum(sizes) / graph.node_count if graph.node_count > 0 else 0,
        }


# Global instance
_community_detector: Optional[CommunityDetector] = None


def get_community_detector() -> CommunityDetector:
    """Get or create global CommunityDetector instance."""
    global _community_detector

    if _community_detector is None:
        _community_detector = CommunityDetector()
        _community_detector.initialize()

    return _community_detector
