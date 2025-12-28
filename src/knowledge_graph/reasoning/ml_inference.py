"""
Machine learning based inference for Knowledge Graph.

Provides link prediction, entity alignment, and inference evaluation.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class InferenceMethod(str, Enum):
    """ML inference methods."""

    # Link prediction methods
    EMBEDDING_SIMILARITY = "embedding_similarity"
    KNOWLEDGE_GRAPH_EMBEDDING = "kg_embedding"
    GRAPH_NEURAL_NETWORK = "gnn"

    # Entity alignment methods
    NAME_SIMILARITY = "name_similarity"
    ATTRIBUTE_SIMILARITY = "attribute_similarity"
    EMBEDDING_ALIGNMENT = "embedding_alignment"

    # Classification methods
    LOGISTIC_REGRESSION = "logistic_regression"
    NEURAL_NETWORK = "neural_network"
    RANDOM_FOREST = "random_forest"


class EmbeddingModel(str, Enum):
    """Knowledge graph embedding models."""

    TRANSE = "transe"  # Translation-based
    DISTMULT = "distmult"  # Bilinear
    COMPLEX = "complex"  # Complex embeddings
    ROTATE = "rotate"  # Rotation-based
    CONVE = "conve"  # Convolutional


class EntityEmbedding(BaseModel):
    """Entity embedding representation."""

    entity_id: str = Field(..., description="Entity identifier")
    entity_name: str = Field(..., description="Entity name")
    embedding: list[float] = Field(..., description="Embedding vector")
    model: str = Field(default="default", description="Embedding model used")
    timestamp: datetime = Field(default_factory=datetime.now)


class RelationEmbedding(BaseModel):
    """Relation embedding representation."""

    relation_id: str = Field(..., description="Relation identifier")
    relation_name: str = Field(..., description="Relation name")
    embedding: list[float] = Field(..., description="Embedding vector")
    model: str = Field(default="default", description="Embedding model used")


class LinkPredictionResult(BaseModel):
    """Result of link prediction inference."""

    head_entity: str = Field(..., description="Head entity")
    relation: str = Field(..., description="Predicted relation")
    tail_entity: str = Field(..., description="Tail entity")
    score: float = Field(..., description="Prediction confidence score")
    method: InferenceMethod = Field(..., description="Method used")
    rank: int = Field(default=0, description="Ranking among predictions")
    supporting_evidence: list[str] = Field(
        default_factory=list, description="Supporting evidence"
    )


class EntityAlignmentResult(BaseModel):
    """Result of entity alignment inference."""

    entity1_id: str = Field(..., description="First entity ID")
    entity1_name: str = Field(..., description="First entity name")
    entity2_id: str = Field(..., description="Second entity ID")
    entity2_name: str = Field(..., description="Second entity name")
    similarity_score: float = Field(..., description="Alignment similarity")
    method: InferenceMethod = Field(..., description="Method used")
    aligned: bool = Field(default=False, description="Whether entities are aligned")
    alignment_features: dict[str, float] = Field(
        default_factory=dict, description="Feature breakdown"
    )


class InferenceEvaluation(BaseModel):
    """Evaluation metrics for inference results."""

    method: InferenceMethod = Field(..., description="Method evaluated")
    precision: float = Field(default=0.0, description="Precision score")
    recall: float = Field(default=0.0, description="Recall score")
    f1_score: float = Field(default=0.0, description="F1 score")
    mean_rank: float = Field(default=0.0, description="Mean rank (for ranking)")
    mrr: float = Field(default=0.0, description="Mean reciprocal rank")
    hits_at_1: float = Field(default=0.0, description="Hits@1")
    hits_at_3: float = Field(default=0.0, description="Hits@3")
    hits_at_10: float = Field(default=0.0, description="Hits@10")
    samples_evaluated: int = Field(default=0, description="Number of samples")
    timestamp: datetime = Field(default_factory=datetime.now)


class TrainingConfig(BaseModel):
    """Configuration for ML model training."""

    embedding_dim: int = Field(default=128, description="Embedding dimension")
    learning_rate: float = Field(default=0.01, description="Learning rate")
    batch_size: int = Field(default=64, description="Batch size")
    epochs: int = Field(default=100, description="Training epochs")
    negative_samples: int = Field(default=10, description="Negative samples per positive")
    margin: float = Field(default=1.0, description="Margin for margin-based loss")
    regularization: float = Field(default=0.01, description="L2 regularization")


@dataclass
class MLInference:
    """Machine learning based inference engine."""

    entity_embeddings: dict[str, EntityEmbedding] = field(default_factory=dict)
    relation_embeddings: dict[str, RelationEmbedding] = field(default_factory=dict)
    embedding_dim: int = 128
    model_type: EmbeddingModel = EmbeddingModel.TRANSE
    training_config: Optional[TrainingConfig] = None

    def __post_init__(self):
        if self.training_config is None:
            self.training_config = TrainingConfig(embedding_dim=self.embedding_dim)

    def initialize_embeddings(
        self,
        entities: list[dict[str, str]],
        relations: list[dict[str, str]],
    ) -> None:
        """Initialize random embeddings for entities and relations."""
        for entity in entities:
            entity_id = entity.get("id", entity.get("entity_id", ""))
            entity_name = entity.get("name", entity.get("entity_name", entity_id))

            # Initialize with random normalized vector
            embedding = self._random_normalized_vector()

            self.entity_embeddings[entity_id] = EntityEmbedding(
                entity_id=entity_id,
                entity_name=entity_name,
                embedding=embedding,
                model=self.model_type.value,
            )

        for relation in relations:
            relation_id = relation.get("id", relation.get("relation_id", ""))
            relation_name = relation.get("name", relation.get("relation_name", relation_id))

            embedding = self._random_normalized_vector()

            self.relation_embeddings[relation_id] = RelationEmbedding(
                relation_id=relation_id,
                relation_name=relation_name,
                embedding=embedding,
                model=self.model_type.value,
            )

        logger.info(
            f"Initialized {len(self.entity_embeddings)} entity embeddings "
            f"and {len(self.relation_embeddings)} relation embeddings"
        )

    def _random_normalized_vector(self) -> list[float]:
        """Generate a random normalized vector."""
        vec = [random.gauss(0, 1) for _ in range(self.embedding_dim)]
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec]

    def predict_links(
        self,
        head_entity: str,
        relation: Optional[str] = None,
        top_k: int = 10,
        method: InferenceMethod = InferenceMethod.EMBEDDING_SIMILARITY,
    ) -> list[LinkPredictionResult]:
        """Predict potential links for a given entity."""
        results = []

        if head_entity not in self.entity_embeddings:
            logger.warning(f"Entity not found: {head_entity}")
            return results

        head_emb = self.entity_embeddings[head_entity].embedding

        # Get candidate tail entities
        candidates = []
        for entity_id, entity_emb in self.entity_embeddings.items():
            if entity_id == head_entity:
                continue

            # Calculate score based on method
            if method == InferenceMethod.EMBEDDING_SIMILARITY:
                score = self._cosine_similarity(head_emb, entity_emb.embedding)
            elif method == InferenceMethod.KNOWLEDGE_GRAPH_EMBEDDING:
                if relation and relation in self.relation_embeddings:
                    rel_emb = self.relation_embeddings[relation].embedding
                    score = self._transe_score(head_emb, rel_emb, entity_emb.embedding)
                else:
                    score = self._cosine_similarity(head_emb, entity_emb.embedding)
            else:
                score = self._cosine_similarity(head_emb, entity_emb.embedding)

            candidates.append((entity_id, score))

        # Sort by score (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Generate results
        for rank, (tail_entity, score) in enumerate(candidates[:top_k], 1):
            results.append(
                LinkPredictionResult(
                    head_entity=head_entity,
                    relation=relation or "related_to",
                    tail_entity=tail_entity,
                    score=score,
                    method=method,
                    rank=rank,
                )
            )

        return results

    def predict_head(
        self,
        relation: str,
        tail_entity: str,
        top_k: int = 10,
    ) -> list[LinkPredictionResult]:
        """Predict potential head entities for (?, relation, tail)."""
        results = []

        if tail_entity not in self.entity_embeddings:
            return results

        if relation not in self.relation_embeddings:
            return results

        tail_emb = self.entity_embeddings[tail_entity].embedding
        rel_emb = self.relation_embeddings[relation].embedding

        candidates = []
        for entity_id, entity_emb in self.entity_embeddings.items():
            if entity_id == tail_entity:
                continue

            score = self._transe_score(entity_emb.embedding, rel_emb, tail_emb)
            candidates.append((entity_id, score))

        candidates.sort(key=lambda x: x[1], reverse=True)

        for rank, (head_entity, score) in enumerate(candidates[:top_k], 1):
            results.append(
                LinkPredictionResult(
                    head_entity=head_entity,
                    relation=relation,
                    tail_entity=tail_entity,
                    score=score,
                    method=InferenceMethod.KNOWLEDGE_GRAPH_EMBEDDING,
                    rank=rank,
                )
            )

        return results

    def align_entities(
        self,
        source_entities: list[str],
        target_entities: list[str],
        threshold: float = 0.7,
        method: InferenceMethod = InferenceMethod.EMBEDDING_ALIGNMENT,
    ) -> list[EntityAlignmentResult]:
        """Find aligned entities between two sets."""
        results = []

        for source_id in source_entities:
            if source_id not in self.entity_embeddings:
                continue

            source_emb = self.entity_embeddings[source_id]
            best_match = None
            best_score = 0.0

            for target_id in target_entities:
                if target_id not in self.entity_embeddings:
                    continue

                target_emb = self.entity_embeddings[target_id]

                # Calculate similarity based on method
                if method == InferenceMethod.EMBEDDING_ALIGNMENT:
                    score = self._cosine_similarity(
                        source_emb.embedding, target_emb.embedding
                    )
                elif method == InferenceMethod.NAME_SIMILARITY:
                    score = self._name_similarity(
                        source_emb.entity_name, target_emb.entity_name
                    )
                else:
                    score = self._cosine_similarity(
                        source_emb.embedding, target_emb.embedding
                    )

                if score > best_score:
                    best_score = score
                    best_match = target_id

            if best_match and best_score >= threshold:
                target_emb = self.entity_embeddings[best_match]
                results.append(
                    EntityAlignmentResult(
                        entity1_id=source_id,
                        entity1_name=source_emb.entity_name,
                        entity2_id=best_match,
                        entity2_name=target_emb.entity_name,
                        similarity_score=best_score,
                        method=method,
                        aligned=True,
                        alignment_features={
                            "embedding_similarity": best_score,
                        },
                    )
                )

        return results

    def train_embeddings(
        self,
        triples: list[tuple[str, str, str]],
        config: Optional[TrainingConfig] = None,
    ) -> dict[str, float]:
        """Train embeddings on knowledge graph triples."""
        if config:
            self.training_config = config

        metrics = {
            "initial_loss": 0.0,
            "final_loss": 0.0,
            "epochs_trained": 0,
        }

        if not triples:
            return metrics

        # Simple training simulation
        logger.info(f"Training on {len(triples)} triples")

        for epoch in range(self.training_config.epochs):
            epoch_loss = 0.0

            for head, relation, tail in triples:
                if head not in self.entity_embeddings:
                    continue
                if relation not in self.relation_embeddings:
                    continue
                if tail not in self.entity_embeddings:
                    continue

                # Get embeddings
                h = np.array(self.entity_embeddings[head].embedding)
                r = np.array(self.relation_embeddings[relation].embedding)
                t = np.array(self.entity_embeddings[tail].embedding)

                # Calculate loss (TransE style)
                score = np.sum(np.abs(h + r - t))
                epoch_loss += score

                # Generate negative sample
                neg_tail = random.choice(list(self.entity_embeddings.keys()))
                if neg_tail != tail:
                    t_neg = np.array(self.entity_embeddings[neg_tail].embedding)
                    neg_score = np.sum(np.abs(h + r - t_neg))

                    # Margin-based update
                    if score + self.training_config.margin > neg_score:
                        # Gradient update (simplified)
                        lr = self.training_config.learning_rate
                        gradient = np.sign(h + r - t)

                        h_new = h - lr * gradient
                        t_new = t + lr * gradient
                        r_new = r - lr * gradient * 0.1

                        # Normalize and update
                        self.entity_embeddings[head].embedding = (
                            h_new / np.linalg.norm(h_new)
                        ).tolist()
                        self.entity_embeddings[tail].embedding = (
                            t_new / np.linalg.norm(t_new)
                        ).tolist()
                        self.relation_embeddings[relation].embedding = (
                            r_new / np.linalg.norm(r_new)
                        ).tolist()

            avg_loss = epoch_loss / len(triples)

            if epoch == 0:
                metrics["initial_loss"] = avg_loss
            metrics["final_loss"] = avg_loss
            metrics["epochs_trained"] = epoch + 1

            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}: avg_loss = {avg_loss:.4f}")

        logger.info(
            f"Training complete: {metrics['epochs_trained']} epochs, "
            f"final_loss = {metrics['final_loss']:.4f}"
        )

        return metrics

    def evaluate(
        self,
        test_triples: list[tuple[str, str, str]],
        method: InferenceMethod = InferenceMethod.KNOWLEDGE_GRAPH_EMBEDDING,
    ) -> InferenceEvaluation:
        """Evaluate inference performance on test data."""
        ranks = []
        hits_1 = 0
        hits_3 = 0
        hits_10 = 0

        for head, relation, tail in test_triples:
            # Predict tail entities
            predictions = self.predict_links(
                head_entity=head,
                relation=relation,
                top_k=len(self.entity_embeddings),
                method=method,
            )

            # Find rank of true tail
            for pred in predictions:
                if pred.tail_entity == tail:
                    rank = pred.rank
                    ranks.append(rank)

                    if rank <= 1:
                        hits_1 += 1
                    if rank <= 3:
                        hits_3 += 1
                    if rank <= 10:
                        hits_10 += 1
                    break

        n = len(test_triples)
        if n == 0:
            return InferenceEvaluation(
                method=method,
                samples_evaluated=0,
            )

        mean_rank = sum(ranks) / len(ranks) if ranks else 0
        mrr = sum(1.0 / r for r in ranks) / len(ranks) if ranks else 0

        return InferenceEvaluation(
            method=method,
            mean_rank=mean_rank,
            mrr=mrr,
            hits_at_1=hits_1 / n,
            hits_at_3=hits_3 / n,
            hits_at_10=hits_10 / n,
            samples_evaluated=n,
        )

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    def _transe_score(
        self,
        head: list[float],
        relation: list[float],
        tail: list[float],
    ) -> float:
        """Calculate TransE score: -||h + r - t||."""
        distance = sum(
            (h + r - t) ** 2 for h, r, t in zip(head, relation, tail)
        )
        return -math.sqrt(distance)

    def _name_similarity(self, name1: str, name2: str) -> float:
        """Calculate name similarity using Jaccard on character n-grams."""
        def get_ngrams(s: str, n: int = 3) -> set[str]:
            s = s.lower()
            return {s[i : i + n] for i in range(max(len(s) - n + 1, 1))}

        ngrams1 = get_ngrams(name1)
        ngrams2 = get_ngrams(name2)

        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)

        return intersection / union if union > 0 else 0.0

    def get_entity_embedding(self, entity_id: str) -> Optional[list[float]]:
        """Get embedding for an entity."""
        if entity_id in self.entity_embeddings:
            return self.entity_embeddings[entity_id].embedding
        return None

    def get_relation_embedding(self, relation_id: str) -> Optional[list[float]]:
        """Get embedding for a relation."""
        if relation_id in self.relation_embeddings:
            return self.relation_embeddings[relation_id].embedding
        return None

    def find_similar_entities(
        self,
        entity_id: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Find most similar entities to a given entity."""
        if entity_id not in self.entity_embeddings:
            return []

        source_emb = self.entity_embeddings[entity_id].embedding
        similarities = []

        for eid, emb in self.entity_embeddings.items():
            if eid == entity_id:
                continue

            sim = self._cosine_similarity(source_emb, emb.embedding)
            similarities.append((eid, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


# Global instance
_ml_inference: Optional[MLInference] = None


def get_ml_inference() -> MLInference:
    """Get or create the global ML inference instance."""
    global _ml_inference
    if _ml_inference is None:
        _ml_inference = MLInference()
    return _ml_inference
