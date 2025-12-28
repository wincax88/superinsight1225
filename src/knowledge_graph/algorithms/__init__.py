"""
Algorithms module for Knowledge Graph.

Provides centrality, community detection, embedding, and prediction algorithms.
"""

from .centrality import (
    Node,
    Edge,
    Graph,
    CentralityScore,
    CentralityResult,
    CentralityAnalyzer,
    get_centrality_analyzer,
)

from .community import (
    Community,
    CommunityResult,
    CommunityDetector,
    get_community_detector,
)

from .embedding import (
    NodeEmbedding,
    EmbeddingResult,
    GraphEmbedder,
    get_graph_embedder,
)

from .prediction import (
    LinkPrediction,
    NodeClassification,
    Recommendation,
    LinkPredictionResult,
    NodeClassificationResult,
    RecommendationResult,
    GraphPredictor,
    get_graph_predictor,
)

__all__ = [
    # Centrality
    "Node",
    "Edge",
    "Graph",
    "CentralityScore",
    "CentralityResult",
    "CentralityAnalyzer",
    "get_centrality_analyzer",
    # Community
    "Community",
    "CommunityResult",
    "CommunityDetector",
    "get_community_detector",
    # Embedding
    "NodeEmbedding",
    "EmbeddingResult",
    "GraphEmbedder",
    "get_graph_embedder",
    # Prediction
    "LinkPrediction",
    "NodeClassification",
    "Recommendation",
    "LinkPredictionResult",
    "NodeClassificationResult",
    "RecommendationResult",
    "GraphPredictor",
    "get_graph_predictor",
]
