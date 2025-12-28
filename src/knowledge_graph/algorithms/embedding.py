"""
Graph Embedding Algorithms for Knowledge Graph.

Provides graph embedding including:
- Node2Vec algorithm
- DeepWalk algorithm
- Graph factorization
- Spectral embedding
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
class NodeEmbedding:
    """Embedding vector for a node."""

    node_id: str = ""
    vector: List[float] = field(default_factory=list)
    dimension: int = 0

    def __post_init__(self):
        self.dimension = len(self.vector)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "vector": self.vector,
            "dimension": self.dimension,
        }

    def similarity(self, other: "NodeEmbedding") -> float:
        """Calculate cosine similarity with another embedding."""
        if len(self.vector) != len(other.vector):
            return 0.0

        dot_product = sum(a * b for a, b in zip(self.vector, other.vector))
        norm_a = math.sqrt(sum(a * a for a in self.vector))
        norm_b = math.sqrt(sum(b * b for b in other.vector))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)


class EmbeddingResult(BaseModel):
    """Result of graph embedding."""

    algorithm: str = Field(default="", description="Algorithm name")
    embeddings: Dict[str, List[float]] = Field(default_factory=dict, description="Node embeddings")
    dimension: int = Field(default=0, description="Embedding dimension")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Statistics")
    processing_time_ms: float = Field(default=0.0, description="Processing time")

    def get_embedding(self, node_id: str) -> Optional[NodeEmbedding]:
        """Get embedding for a node."""
        if node_id in self.embeddings:
            return NodeEmbedding(node_id=node_id, vector=self.embeddings[node_id])
        return None

    def find_similar(self, node_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find top-k most similar nodes."""
        if node_id not in self.embeddings:
            return []

        target_embedding = NodeEmbedding(node_id=node_id, vector=self.embeddings[node_id])

        similarities = []
        for other_id, vector in self.embeddings.items():
            if other_id == node_id:
                continue
            other_embedding = NodeEmbedding(node_id=other_id, vector=vector)
            sim = target_embedding.similarity(other_embedding)
            similarities.append((other_id, sim))

        # Sort by similarity descending
        similarities.sort(key=lambda x: -x[1])

        return similarities[:top_k]


class GraphEmbedder:
    """
    Graph embedder for learning node representations.

    Implements various graph embedding algorithms.
    """

    def __init__(
        self,
        dimension: int = 64,
        walk_length: int = 80,
        num_walks: int = 10,
        window_size: int = 10,
        p: float = 1.0,  # Return parameter for Node2Vec
        q: float = 1.0,  # In-out parameter for Node2Vec
        learning_rate: float = 0.025,
        epochs: int = 5,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize GraphEmbedder.

        Args:
            dimension: Embedding dimension
            walk_length: Length of random walks
            num_walks: Number of walks per node
            window_size: Context window size for skip-gram
            p: Return parameter (Node2Vec)
            q: In-out parameter (Node2Vec)
            learning_rate: Learning rate for SGD
            epochs: Number of training epochs
            random_seed: Random seed for reproducibility
        """
        self.dimension = dimension
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.p = p
        self.q = q
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_seed = random_seed
        self._initialized = False

        if random_seed is not None:
            random.seed(random_seed)

    def initialize(self) -> None:
        """Initialize the embedder."""
        if self._initialized:
            return
        self._initialized = True
        logger.info("GraphEmbedder initialized")

    def node2vec(self, graph: Graph) -> EmbeddingResult:
        """
        Learn node embeddings using Node2Vec algorithm.

        Node2Vec uses biased random walks to explore neighborhoods.

        Args:
            graph: Input graph

        Returns:
            EmbeddingResult with node embeddings
        """
        if not self._initialized:
            self.initialize()

        import time
        start_time = time.time()

        if graph.node_count == 0:
            return EmbeddingResult(algorithm="node2vec", dimension=self.dimension)

        # Precompute transition probabilities
        transition_probs = self._precompute_transition_probs(graph)

        # Generate random walks
        walks = []
        node_list = list(graph.nodes.keys())

        for _ in range(self.num_walks):
            random.shuffle(node_list)
            for node in node_list:
                walk = self._node2vec_walk(graph, node, transition_probs)
                walks.append(walk)

        # Train skip-gram model
        embeddings = self._train_skip_gram(walks, graph)

        processing_time = (time.time() - start_time) * 1000

        return EmbeddingResult(
            algorithm="node2vec",
            embeddings=embeddings,
            dimension=self.dimension,
            statistics={
                "num_walks": len(walks),
                "walk_length": self.walk_length,
                "p": self.p,
                "q": self.q,
            },
            processing_time_ms=processing_time,
        )

    def deepwalk(self, graph: Graph) -> EmbeddingResult:
        """
        Learn node embeddings using DeepWalk algorithm.

        DeepWalk uses uniform random walks (equivalent to Node2Vec with p=q=1).

        Args:
            graph: Input graph

        Returns:
            EmbeddingResult with node embeddings
        """
        if not self._initialized:
            self.initialize()

        import time
        start_time = time.time()

        if graph.node_count == 0:
            return EmbeddingResult(algorithm="deepwalk", dimension=self.dimension)

        # Generate uniform random walks
        walks = []
        node_list = list(graph.nodes.keys())

        for _ in range(self.num_walks):
            random.shuffle(node_list)
            for node in node_list:
                walk = self._random_walk(graph, node)
                walks.append(walk)

        # Train skip-gram model
        embeddings = self._train_skip_gram(walks, graph)

        processing_time = (time.time() - start_time) * 1000

        return EmbeddingResult(
            algorithm="deepwalk",
            embeddings=embeddings,
            dimension=self.dimension,
            statistics={
                "num_walks": len(walks),
                "walk_length": self.walk_length,
            },
            processing_time_ms=processing_time,
        )

    def spectral_embedding(self, graph: Graph, num_components: Optional[int] = None) -> EmbeddingResult:
        """
        Learn node embeddings using spectral decomposition.

        Uses eigendecomposition of the normalized Laplacian.

        Args:
            graph: Input graph
            num_components: Number of components (defaults to self.dimension)

        Returns:
            EmbeddingResult with node embeddings
        """
        if not self._initialized:
            self.initialize()

        import time
        start_time = time.time()

        n = graph.node_count
        if n == 0:
            return EmbeddingResult(algorithm="spectral", dimension=self.dimension)

        num_components = num_components or min(self.dimension, n - 1)

        node_ids = list(graph.nodes.keys())
        node_idx = {node_id: i for i, node_id in enumerate(node_ids)}

        # Build adjacency matrix
        adj_matrix = [[0.0] * n for _ in range(n)]
        for edge in graph.edges:
            i = node_idx.get(edge.source_id, -1)
            j = node_idx.get(edge.target_id, -1)
            if i >= 0 and j >= 0:
                adj_matrix[i][j] = edge.weight
                if not graph.directed:
                    adj_matrix[j][i] = edge.weight

        # Build degree matrix
        degrees = [sum(row) for row in adj_matrix]

        # Build normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
        laplacian = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    laplacian[i][j] = 1.0 if degrees[i] > 0 else 0.0
                elif adj_matrix[i][j] > 0:
                    laplacian[i][j] = -adj_matrix[i][j] / math.sqrt(degrees[i] * degrees[j])

        # Power iteration for eigendecomposition (simplified)
        eigenvectors = self._power_iteration(laplacian, num_components)

        # Build embeddings
        embeddings = {}
        for i, node_id in enumerate(node_ids):
            vector = [eigenvectors[k][i] for k in range(num_components)]
            # Pad to full dimension if needed
            while len(vector) < self.dimension:
                vector.append(0.0)
            embeddings[node_id] = vector[:self.dimension]

        processing_time = (time.time() - start_time) * 1000

        return EmbeddingResult(
            algorithm="spectral",
            embeddings=embeddings,
            dimension=self.dimension,
            statistics={
                "num_components": num_components,
            },
            processing_time_ms=processing_time,
        )

    def graph_factorization(self, graph: Graph, regularization: float = 0.01) -> EmbeddingResult:
        """
        Learn node embeddings using matrix factorization.

        Factorizes the adjacency matrix using gradient descent.

        Args:
            graph: Input graph
            regularization: L2 regularization weight

        Returns:
            EmbeddingResult with node embeddings
        """
        if not self._initialized:
            self.initialize()

        import time
        start_time = time.time()

        n = graph.node_count
        if n == 0:
            return EmbeddingResult(algorithm="graph_factorization", dimension=self.dimension)

        node_ids = list(graph.nodes.keys())
        node_idx = {node_id: i for i, node_id in enumerate(node_ids)}

        # Initialize embeddings randomly
        embeddings_matrix = [
            [random.gauss(0, 0.1) for _ in range(self.dimension)]
            for _ in range(n)
        ]

        # Build edge list with indices
        edge_list = []
        for edge in graph.edges:
            i = node_idx.get(edge.source_id, -1)
            j = node_idx.get(edge.target_id, -1)
            if i >= 0 and j >= 0:
                edge_list.append((i, j, edge.weight))
                if not graph.directed:
                    edge_list.append((j, i, edge.weight))

        # SGD optimization
        for epoch in range(self.epochs):
            random.shuffle(edge_list)
            total_loss = 0.0

            for i, j, weight in edge_list:
                # Compute prediction
                pred = sum(embeddings_matrix[i][k] * embeddings_matrix[j][k]
                          for k in range(self.dimension))

                # Compute error
                error = weight - pred
                total_loss += error ** 2

                # Update embeddings
                for k in range(self.dimension):
                    grad_i = -2 * error * embeddings_matrix[j][k] + 2 * regularization * embeddings_matrix[i][k]
                    grad_j = -2 * error * embeddings_matrix[i][k] + 2 * regularization * embeddings_matrix[j][k]

                    embeddings_matrix[i][k] -= self.learning_rate * grad_i
                    embeddings_matrix[j][k] -= self.learning_rate * grad_j

            logger.debug(f"Epoch {epoch + 1}: loss = {total_loss}")

        # Build embeddings dict
        embeddings = {
            node_ids[i]: embeddings_matrix[i]
            for i in range(n)
        }

        processing_time = (time.time() - start_time) * 1000

        return EmbeddingResult(
            algorithm="graph_factorization",
            embeddings=embeddings,
            dimension=self.dimension,
            statistics={
                "epochs": self.epochs,
                "regularization": regularization,
            },
            processing_time_ms=processing_time,
        )

    def _precompute_transition_probs(self, graph: Graph) -> Dict[Tuple[str, str], Dict[str, float]]:
        """Precompute Node2Vec transition probabilities."""
        probs = {}

        for edge in graph.edges:
            src, dst = edge.source_id, edge.target_id

            # For each edge (src, dst), compute probabilities for next step from dst
            dst_neighbors = graph.get_neighbors(dst)
            unnormalized = {}

            for neighbor in dst_neighbors:
                if neighbor == src:
                    # Return to previous node
                    unnormalized[neighbor] = 1.0 / self.p
                elif neighbor in graph.get_neighbors(src):
                    # Common neighbor
                    unnormalized[neighbor] = 1.0
                else:
                    # Move away
                    unnormalized[neighbor] = 1.0 / self.q

            # Normalize
            total = sum(unnormalized.values())
            if total > 0:
                probs[(src, dst)] = {k: v / total for k, v in unnormalized.items()}

        return probs

    def _node2vec_walk(
        self,
        graph: Graph,
        start: str,
        transition_probs: Dict[Tuple[str, str], Dict[str, float]],
    ) -> List[str]:
        """Generate a single Node2Vec random walk."""
        walk = [start]

        while len(walk) < self.walk_length:
            current = walk[-1]
            neighbors = graph.get_neighbors(current)

            if not neighbors:
                break

            if len(walk) == 1:
                # First step: uniform random
                walk.append(random.choice(neighbors))
            else:
                prev = walk[-2]
                probs = transition_probs.get((prev, current), {})

                if probs:
                    # Weighted random choice
                    choices = list(probs.keys())
                    weights = list(probs.values())
                    walk.append(random.choices(choices, weights=weights)[0])
                else:
                    # Fallback to uniform
                    walk.append(random.choice(neighbors))

        return walk

    def _random_walk(self, graph: Graph, start: str) -> List[str]:
        """Generate a uniform random walk (DeepWalk)."""
        walk = [start]

        while len(walk) < self.walk_length:
            current = walk[-1]
            neighbors = graph.get_neighbors(current)

            if not neighbors:
                break

            walk.append(random.choice(neighbors))

        return walk

    def _train_skip_gram(self, walks: List[List[str]], graph: Graph) -> Dict[str, List[float]]:
        """Train skip-gram model on random walks."""
        # Initialize embeddings
        embeddings = {
            node_id: [random.gauss(0, 0.1) for _ in range(self.dimension)]
            for node_id in graph.nodes
        }

        # Context embeddings
        context_embeddings = {
            node_id: [random.gauss(0, 0.1) for _ in range(self.dimension)]
            for node_id in graph.nodes
        }

        # Build vocabulary
        node_list = list(graph.nodes.keys())

        # Training with negative sampling
        for epoch in range(self.epochs):
            for walk in walks:
                for i, target in enumerate(walk):
                    # Context window
                    start = max(0, i - self.window_size)
                    end = min(len(walk), i + self.window_size + 1)

                    for j in range(start, end):
                        if i == j:
                            continue

                        context = walk[j]

                        # Positive sample
                        self._update_embeddings(
                            embeddings[target],
                            context_embeddings[context],
                            1.0,
                        )

                        # Negative samples
                        for _ in range(5):  # 5 negative samples
                            neg = random.choice(node_list)
                            if neg != context:
                                self._update_embeddings(
                                    embeddings[target],
                                    context_embeddings[neg],
                                    0.0,
                                )

        return embeddings

    def _update_embeddings(self, target: List[float], context: List[float], label: float):
        """Update embeddings using SGD."""
        # Sigmoid activation
        dot = sum(t * c for t, c in zip(target, context))
        dot = max(-10, min(10, dot))  # Clip for numerical stability
        pred = 1.0 / (1.0 + math.exp(-dot))

        # Gradient
        error = label - pred

        for i in range(len(target)):
            grad = error * context[i]
            target[i] += self.learning_rate * grad
            context[i] += self.learning_rate * error * target[i]

    def _power_iteration(self, matrix: List[List[float]], num_vectors: int) -> List[List[float]]:
        """Power iteration for finding top eigenvectors."""
        n = len(matrix)
        eigenvectors = []

        for k in range(num_vectors):
            # Initialize random vector
            v = [random.gauss(0, 1) for _ in range(n)]
            norm = math.sqrt(sum(x * x for x in v))
            v = [x / norm for x in v]

            # Power iteration
            for _ in range(100):
                # Matrix-vector multiplication
                new_v = [sum(matrix[i][j] * v[j] for j in range(n)) for i in range(n)]

                # Orthogonalize against previous eigenvectors
                for prev_v in eigenvectors:
                    dot = sum(new_v[i] * prev_v[i] for i in range(n))
                    new_v = [new_v[i] - dot * prev_v[i] for i in range(n)]

                # Normalize
                norm = math.sqrt(sum(x * x for x in new_v))
                if norm > 0:
                    new_v = [x / norm for x in new_v]

                v = new_v

            eigenvectors.append(v)

        return eigenvectors


# Global instance
_graph_embedder: Optional[GraphEmbedder] = None


def get_graph_embedder() -> GraphEmbedder:
    """Get or create global GraphEmbedder instance."""
    global _graph_embedder

    if _graph_embedder is None:
        _graph_embedder = GraphEmbedder()
        _graph_embedder.initialize()

    return _graph_embedder
