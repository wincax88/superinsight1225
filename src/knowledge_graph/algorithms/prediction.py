"""
Prediction Algorithms for Knowledge Graph.

Provides prediction capabilities including:
- Link prediction
- Node classification
- Similarity-based recommendations
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
from .embedding import GraphEmbedder, EmbeddingResult, NodeEmbedding

logger = logging.getLogger(__name__)


@dataclass
class LinkPrediction:
    """Predicted link between two nodes."""

    source_id: str = ""
    target_id: str = ""
    score: float = 0.0
    method: str = ""
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "score": self.score,
            "method": self.method,
            "rank": self.rank,
        }


@dataclass
class NodeClassification:
    """Predicted class for a node."""

    node_id: str = ""
    predicted_class: str = ""
    confidence: float = 0.0
    class_probabilities: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "class_probabilities": self.class_probabilities,
        }


@dataclass
class Recommendation:
    """Recommended node for a target node."""

    target_node_id: str = ""
    recommended_node_id: str = ""
    score: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_node_id": self.target_node_id,
            "recommended_node_id": self.recommended_node_id,
            "score": self.score,
            "reason": self.reason,
        }


class LinkPredictionResult(BaseModel):
    """Result of link prediction."""

    method: str = Field(default="", description="Prediction method")
    predictions: List[Dict[str, Any]] = Field(default_factory=list, description="Predicted links")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Statistics")
    processing_time_ms: float = Field(default=0.0, description="Processing time")


class NodeClassificationResult(BaseModel):
    """Result of node classification."""

    method: str = Field(default="", description="Classification method")
    classifications: List[Dict[str, Any]] = Field(default_factory=list, description="Node classifications")
    accuracy: Optional[float] = Field(None, description="Accuracy if ground truth available")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Statistics")
    processing_time_ms: float = Field(default=0.0, description="Processing time")


class RecommendationResult(BaseModel):
    """Result of recommendation."""

    method: str = Field(default="", description="Recommendation method")
    recommendations: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict, description="Recommendations per target node"
    )
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Statistics")
    processing_time_ms: float = Field(default=0.0, description="Processing time")


class GraphPredictor:
    """
    Graph predictor for link prediction and node classification.

    Implements various prediction algorithms.
    """

    def __init__(
        self,
        embedding_dimension: int = 64,
        top_k: int = 10,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize GraphPredictor.

        Args:
            embedding_dimension: Dimension for embedding-based methods
            top_k: Number of top predictions to return
            random_seed: Random seed for reproducibility
        """
        self.embedding_dimension = embedding_dimension
        self.top_k = top_k
        self.random_seed = random_seed
        self._initialized = False
        self._embedder: Optional[GraphEmbedder] = None

        if random_seed is not None:
            random.seed(random_seed)

    def initialize(self) -> None:
        """Initialize the predictor."""
        if self._initialized:
            return

        self._embedder = GraphEmbedder(dimension=self.embedding_dimension)
        self._embedder.initialize()

        self._initialized = True
        logger.info("GraphPredictor initialized")

    def predict_links_common_neighbors(self, graph: Graph, candidate_pairs: Optional[List[Tuple[str, str]]] = None) -> LinkPredictionResult:
        """
        Predict links using common neighbors method.

        Score = |N(u) ∩ N(v)|

        Args:
            graph: Input graph
            candidate_pairs: Optional list of node pairs to score

        Returns:
            LinkPredictionResult with predictions
        """
        if not self._initialized:
            self.initialize()

        import time
        start_time = time.time()

        # Get existing edges
        existing_edges = {(e.source_id, e.target_id) for e in graph.edges}
        if not graph.directed:
            existing_edges.update((e.target_id, e.source_id) for e in graph.edges)

        # Generate candidate pairs if not provided
        if candidate_pairs is None:
            candidate_pairs = self._generate_candidate_pairs(graph, existing_edges)

        predictions = []
        for source, target in candidate_pairs:
            if (source, target) in existing_edges:
                continue

            # Count common neighbors
            source_neighbors = set(graph.get_neighbors(source))
            target_neighbors = set(graph.get_neighbors(target))
            common = len(source_neighbors & target_neighbors)

            if common > 0:
                predictions.append(LinkPrediction(
                    source_id=source,
                    target_id=target,
                    score=float(common),
                    method="common_neighbors",
                ))

        # Sort and rank
        predictions.sort(key=lambda p: -p.score)
        for i, pred in enumerate(predictions[:self.top_k]):
            pred.rank = i + 1

        processing_time = (time.time() - start_time) * 1000

        return LinkPredictionResult(
            method="common_neighbors",
            predictions=[p.to_dict() for p in predictions[:self.top_k]],
            statistics={"total_candidates": len(candidate_pairs)},
            processing_time_ms=processing_time,
        )

    def predict_links_jaccard(self, graph: Graph, candidate_pairs: Optional[List[Tuple[str, str]]] = None) -> LinkPredictionResult:
        """
        Predict links using Jaccard coefficient.

        Score = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|

        Args:
            graph: Input graph
            candidate_pairs: Optional list of node pairs to score

        Returns:
            LinkPredictionResult with predictions
        """
        if not self._initialized:
            self.initialize()

        import time
        start_time = time.time()

        existing_edges = {(e.source_id, e.target_id) for e in graph.edges}
        if not graph.directed:
            existing_edges.update((e.target_id, e.source_id) for e in graph.edges)

        if candidate_pairs is None:
            candidate_pairs = self._generate_candidate_pairs(graph, existing_edges)

        predictions = []
        for source, target in candidate_pairs:
            if (source, target) in existing_edges:
                continue

            source_neighbors = set(graph.get_neighbors(source))
            target_neighbors = set(graph.get_neighbors(target))

            intersection = len(source_neighbors & target_neighbors)
            union = len(source_neighbors | target_neighbors)

            if union > 0:
                score = intersection / union
                if score > 0:
                    predictions.append(LinkPrediction(
                        source_id=source,
                        target_id=target,
                        score=score,
                        method="jaccard",
                    ))

        predictions.sort(key=lambda p: -p.score)
        for i, pred in enumerate(predictions[:self.top_k]):
            pred.rank = i + 1

        processing_time = (time.time() - start_time) * 1000

        return LinkPredictionResult(
            method="jaccard",
            predictions=[p.to_dict() for p in predictions[:self.top_k]],
            statistics={"total_candidates": len(candidate_pairs)},
            processing_time_ms=processing_time,
        )

    def predict_links_adamic_adar(self, graph: Graph, candidate_pairs: Optional[List[Tuple[str, str]]] = None) -> LinkPredictionResult:
        """
        Predict links using Adamic-Adar index.

        Score = Σ 1/log(|N(z)|) for z ∈ N(u) ∩ N(v)

        Args:
            graph: Input graph
            candidate_pairs: Optional list of node pairs to score

        Returns:
            LinkPredictionResult with predictions
        """
        if not self._initialized:
            self.initialize()

        import time
        start_time = time.time()

        existing_edges = {(e.source_id, e.target_id) for e in graph.edges}
        if not graph.directed:
            existing_edges.update((e.target_id, e.source_id) for e in graph.edges)

        if candidate_pairs is None:
            candidate_pairs = self._generate_candidate_pairs(graph, existing_edges)

        # Precompute degrees
        degrees = {node_id: len(graph.get_neighbors(node_id)) for node_id in graph.nodes}

        predictions = []
        for source, target in candidate_pairs:
            if (source, target) in existing_edges:
                continue

            source_neighbors = set(graph.get_neighbors(source))
            target_neighbors = set(graph.get_neighbors(target))
            common = source_neighbors & target_neighbors

            if common:
                score = sum(1.0 / math.log(degrees[z] + 1) for z in common if degrees[z] > 1)
                if score > 0:
                    predictions.append(LinkPrediction(
                        source_id=source,
                        target_id=target,
                        score=score,
                        method="adamic_adar",
                    ))

        predictions.sort(key=lambda p: -p.score)
        for i, pred in enumerate(predictions[:self.top_k]):
            pred.rank = i + 1

        processing_time = (time.time() - start_time) * 1000

        return LinkPredictionResult(
            method="adamic_adar",
            predictions=[p.to_dict() for p in predictions[:self.top_k]],
            statistics={"total_candidates": len(candidate_pairs)},
            processing_time_ms=processing_time,
        )

    def predict_links_preferential_attachment(self, graph: Graph, candidate_pairs: Optional[List[Tuple[str, str]]] = None) -> LinkPredictionResult:
        """
        Predict links using preferential attachment.

        Score = |N(u)| × |N(v)|

        Args:
            graph: Input graph
            candidate_pairs: Optional list of node pairs to score

        Returns:
            LinkPredictionResult with predictions
        """
        if not self._initialized:
            self.initialize()

        import time
        start_time = time.time()

        existing_edges = {(e.source_id, e.target_id) for e in graph.edges}
        if not graph.directed:
            existing_edges.update((e.target_id, e.source_id) for e in graph.edges)

        if candidate_pairs is None:
            candidate_pairs = self._generate_candidate_pairs(graph, existing_edges)

        degrees = {node_id: len(graph.get_neighbors(node_id)) for node_id in graph.nodes}

        predictions = []
        for source, target in candidate_pairs:
            if (source, target) in existing_edges:
                continue

            score = degrees[source] * degrees[target]
            if score > 0:
                predictions.append(LinkPrediction(
                    source_id=source,
                    target_id=target,
                    score=float(score),
                    method="preferential_attachment",
                ))

        predictions.sort(key=lambda p: -p.score)
        for i, pred in enumerate(predictions[:self.top_k]):
            pred.rank = i + 1

        processing_time = (time.time() - start_time) * 1000

        return LinkPredictionResult(
            method="preferential_attachment",
            predictions=[p.to_dict() for p in predictions[:self.top_k]],
            statistics={"total_candidates": len(candidate_pairs)},
            processing_time_ms=processing_time,
        )

    def predict_links_embedding(self, graph: Graph, candidate_pairs: Optional[List[Tuple[str, str]]] = None) -> LinkPredictionResult:
        """
        Predict links using node embeddings.

        Score = cosine similarity of node embeddings

        Args:
            graph: Input graph
            candidate_pairs: Optional list of node pairs to score

        Returns:
            LinkPredictionResult with predictions
        """
        if not self._initialized:
            self.initialize()

        import time
        start_time = time.time()

        # Generate embeddings
        embedding_result = self._embedder.node2vec(graph)

        existing_edges = {(e.source_id, e.target_id) for e in graph.edges}
        if not graph.directed:
            existing_edges.update((e.target_id, e.source_id) for e in graph.edges)

        if candidate_pairs is None:
            candidate_pairs = self._generate_candidate_pairs(graph, existing_edges)

        predictions = []
        for source, target in candidate_pairs:
            if (source, target) in existing_edges:
                continue

            source_emb = embedding_result.get_embedding(source)
            target_emb = embedding_result.get_embedding(target)

            if source_emb and target_emb:
                score = source_emb.similarity(target_emb)
                if score > 0:
                    predictions.append(LinkPrediction(
                        source_id=source,
                        target_id=target,
                        score=score,
                        method="embedding_similarity",
                    ))

        predictions.sort(key=lambda p: -p.score)
        for i, pred in enumerate(predictions[:self.top_k]):
            pred.rank = i + 1

        processing_time = (time.time() - start_time) * 1000

        return LinkPredictionResult(
            method="embedding_similarity",
            predictions=[p.to_dict() for p in predictions[:self.top_k]],
            statistics={
                "total_candidates": len(candidate_pairs),
                "embedding_dimension": self.embedding_dimension,
            },
            processing_time_ms=processing_time,
        )

    def classify_nodes(
        self,
        graph: Graph,
        labeled_nodes: Dict[str, str],
        unlabeled_nodes: Optional[List[str]] = None,
    ) -> NodeClassificationResult:
        """
        Classify nodes using label propagation.

        Args:
            graph: Input graph
            labeled_nodes: Dictionary of node_id -> class label
            unlabeled_nodes: List of nodes to classify (defaults to all unlabeled)

        Returns:
            NodeClassificationResult with classifications
        """
        if not self._initialized:
            self.initialize()

        import time
        start_time = time.time()

        if unlabeled_nodes is None:
            unlabeled_nodes = [n for n in graph.nodes if n not in labeled_nodes]

        if not unlabeled_nodes:
            return NodeClassificationResult(
                method="label_propagation",
                classifications=[],
                statistics={"labeled_count": len(labeled_nodes)},
            )

        # Get all classes
        classes = list(set(labeled_nodes.values()))

        # Initialize label distributions
        label_dist = {}
        for node_id in graph.nodes:
            if node_id in labeled_nodes:
                # Labeled node: one-hot distribution
                label_dist[node_id] = {c: 1.0 if c == labeled_nodes[node_id] else 0.0 for c in classes}
            else:
                # Unlabeled: uniform distribution
                label_dist[node_id] = {c: 1.0 / len(classes) for c in classes}

        # Iterate label propagation
        for _ in range(100):
            new_dist = {}

            for node_id in graph.nodes:
                if node_id in labeled_nodes:
                    # Keep labeled nodes fixed
                    new_dist[node_id] = label_dist[node_id]
                else:
                    # Average neighbor distributions
                    neighbors = graph.get_neighbors(node_id)
                    if neighbors:
                        avg_dist = {c: 0.0 for c in classes}
                        for neighbor in neighbors:
                            for c in classes:
                                avg_dist[c] += label_dist[neighbor][c]
                        total = sum(avg_dist.values())
                        if total > 0:
                            new_dist[node_id] = {c: v / total for c, v in avg_dist.items()}
                        else:
                            new_dist[node_id] = label_dist[node_id]
                    else:
                        new_dist[node_id] = label_dist[node_id]

            label_dist = new_dist

        # Build classifications
        classifications = []
        for node_id in unlabeled_nodes:
            dist = label_dist[node_id]
            predicted_class = max(dist, key=dist.get)
            confidence = dist[predicted_class]

            classifications.append(NodeClassification(
                node_id=node_id,
                predicted_class=predicted_class,
                confidence=confidence,
                class_probabilities=dist,
            ))

        # Sort by confidence
        classifications.sort(key=lambda c: -c.confidence)

        processing_time = (time.time() - start_time) * 1000

        return NodeClassificationResult(
            method="label_propagation",
            classifications=[c.to_dict() for c in classifications],
            statistics={
                "labeled_count": len(labeled_nodes),
                "unlabeled_count": len(unlabeled_nodes),
                "num_classes": len(classes),
            },
            processing_time_ms=processing_time,
        )

    def recommend_nodes(
        self,
        graph: Graph,
        target_nodes: List[str],
        exclude_neighbors: bool = True,
    ) -> RecommendationResult:
        """
        Recommend similar nodes for target nodes.

        Args:
            graph: Input graph
            target_nodes: List of nodes to get recommendations for
            exclude_neighbors: Whether to exclude existing neighbors

        Returns:
            RecommendationResult with recommendations
        """
        if not self._initialized:
            self.initialize()

        import time
        start_time = time.time()

        # Generate embeddings
        embedding_result = self._embedder.node2vec(graph)

        recommendations = {}

        for target_id in target_nodes:
            if target_id not in graph.nodes:
                continue

            # Get similar nodes
            similar = embedding_result.find_similar(target_id, top_k=self.top_k * 2)

            # Filter neighbors if requested
            if exclude_neighbors:
                neighbors = set(graph.get_neighbors(target_id))
                similar = [(n, s) for n, s in similar if n not in neighbors]

            # Build recommendations
            recs = []
            for node_id, score in similar[:self.top_k]:
                recs.append(Recommendation(
                    target_node_id=target_id,
                    recommended_node_id=node_id,
                    score=score,
                    reason="embedding_similarity",
                ))

            recommendations[target_id] = [r.to_dict() for r in recs]

        processing_time = (time.time() - start_time) * 1000

        return RecommendationResult(
            method="embedding_similarity",
            recommendations=recommendations,
            statistics={
                "target_count": len(target_nodes),
                "avg_recommendations": sum(len(r) for r in recommendations.values()) / len(target_nodes) if target_nodes else 0,
            },
            processing_time_ms=processing_time,
        )

    def predict_all_methods(self, graph: Graph) -> Dict[str, LinkPredictionResult]:
        """
        Run all link prediction methods.

        Args:
            graph: Input graph

        Returns:
            Dictionary of method name to LinkPredictionResult
        """
        return {
            "common_neighbors": self.predict_links_common_neighbors(graph),
            "jaccard": self.predict_links_jaccard(graph),
            "adamic_adar": self.predict_links_adamic_adar(graph),
            "preferential_attachment": self.predict_links_preferential_attachment(graph),
            "embedding": self.predict_links_embedding(graph),
        }

    def _generate_candidate_pairs(
        self,
        graph: Graph,
        existing_edges: Set[Tuple[str, str]],
        max_pairs: int = 10000,
    ) -> List[Tuple[str, str]]:
        """Generate candidate node pairs for link prediction."""
        candidates = []
        node_list = list(graph.nodes.keys())

        # Focus on 2-hop neighbors (more likely to form links)
        for node in node_list:
            neighbors = set(graph.get_neighbors(node))
            two_hop = set()

            for neighbor in neighbors:
                for n2 in graph.get_neighbors(neighbor):
                    if n2 != node and n2 not in neighbors:
                        two_hop.add(n2)

            for target in two_hop:
                if (node, target) not in existing_edges and (target, node) not in existing_edges:
                    candidates.append((node, target))

                if len(candidates) >= max_pairs:
                    return candidates

        return candidates


# Global instance
_graph_predictor: Optional[GraphPredictor] = None


def get_graph_predictor() -> GraphPredictor:
    """Get or create global GraphPredictor instance."""
    global _graph_predictor

    if _graph_predictor is None:
        _graph_predictor = GraphPredictor()
        _graph_predictor.initialize()

    return _graph_predictor
