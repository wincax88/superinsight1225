"""
Entity alignment module for Knowledge Graph fusion.

Provides name-based, attribute-based, and ML-based entity matching.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AlignmentMethod(str, Enum):
    """Methods for entity alignment."""

    NAME_SIMILARITY = "name_similarity"
    ATTRIBUTE_SIMILARITY = "attribute_similarity"
    EMBEDDING_ALIGNMENT = "embedding_alignment"
    HYBRID = "hybrid"
    RULE_BASED = "rule_based"


class SimilarityMetric(str, Enum):
    """Similarity metrics for string comparison."""

    JACCARD = "jaccard"
    LEVENSHTEIN = "levenshtein"
    JARO_WINKLER = "jaro_winkler"
    COSINE = "cosine"
    NGRAM = "ngram"


class AlignmentStatus(str, Enum):
    """Status of alignment decision."""

    ALIGNED = "aligned"
    NOT_ALIGNED = "not_aligned"
    UNCERTAIN = "uncertain"
    NEEDS_REVIEW = "needs_review"


class EntityCandidate(BaseModel):
    """An entity candidate for alignment."""

    entity_id: str = Field(..., description="Entity identifier")
    entity_name: str = Field(..., description="Entity name")
    entity_type: str = Field(default="", description="Entity type")
    source: str = Field(default="", description="Source knowledge base")
    attributes: dict[str, Any] = Field(
        default_factory=dict, description="Entity attributes"
    )
    embedding: Optional[list[float]] = Field(
        default=None, description="Entity embedding"
    )


class AlignmentPair(BaseModel):
    """A pair of aligned entities."""

    entity1: EntityCandidate = Field(..., description="First entity")
    entity2: EntityCandidate = Field(..., description="Second entity")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    confidence: float = Field(..., description="Alignment confidence (0-1)")
    method: AlignmentMethod = Field(..., description="Method used")
    status: AlignmentStatus = Field(default=AlignmentStatus.UNCERTAIN)
    feature_scores: dict[str, float] = Field(
        default_factory=dict, description="Individual feature scores"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class AlignmentConfig(BaseModel):
    """Configuration for entity alignment."""

    method: AlignmentMethod = Field(default=AlignmentMethod.HYBRID)
    similarity_metric: SimilarityMetric = Field(default=SimilarityMetric.JARO_WINKLER)
    alignment_threshold: float = Field(default=0.8, description="Threshold for alignment")
    uncertain_threshold: float = Field(default=0.6, description="Threshold for uncertainty")
    use_type_matching: bool = Field(default=True)
    type_mismatch_penalty: float = Field(default=0.3)
    attribute_weights: dict[str, float] = Field(
        default_factory=lambda: {"name": 0.5, "type": 0.2, "attributes": 0.3}
    )
    ngram_size: int = Field(default=3, description="N-gram size for similarity")
    max_candidates: int = Field(default=10, description="Max candidates per entity")


class AlignmentResult(BaseModel):
    """Result of entity alignment process."""

    aligned_pairs: list[AlignmentPair] = Field(default_factory=list)
    uncertain_pairs: list[AlignmentPair] = Field(default_factory=list)
    unmatched_source: list[str] = Field(default_factory=list)
    unmatched_target: list[str] = Field(default_factory=list)
    total_source: int = Field(default=0)
    total_target: int = Field(default=0)
    precision: float = Field(default=0.0)
    recall: float = Field(default=0.0)
    f1_score: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=datetime.now)
    config: AlignmentConfig = Field(default_factory=AlignmentConfig)


class BlockingConfig(BaseModel):
    """Configuration for blocking (candidate generation)."""

    blocking_keys: list[str] = Field(
        default_factory=lambda: ["type", "first_letter"],
        description="Keys for blocking",
    )
    max_block_size: int = Field(default=100)
    min_similarity_for_block: float = Field(default=0.3)


@dataclass
class EntityAligner:
    """Entity alignment engine."""

    config: AlignmentConfig = field(default_factory=AlignmentConfig)
    blocking_config: BlockingConfig = field(default_factory=BlockingConfig)
    alignment_cache: dict[str, AlignmentPair] = field(default_factory=dict)

    def align_entities(
        self,
        source_entities: list[dict[str, Any]],
        target_entities: list[dict[str, Any]],
        config: Optional[AlignmentConfig] = None,
    ) -> AlignmentResult:
        """Align entities from source to target knowledge base."""
        if config:
            self.config = config

        # Convert to EntityCandidate objects
        source_candidates = [self._to_candidate(e, "source") for e in source_entities]
        target_candidates = [self._to_candidate(e, "target") for e in target_entities]

        # Generate candidate pairs using blocking
        candidate_pairs = self._generate_candidates(source_candidates, target_candidates)

        # Score each pair
        scored_pairs = []
        for source, target in candidate_pairs:
            score = self._calculate_similarity(source, target)
            pair = AlignmentPair(
                entity1=source,
                entity2=target,
                similarity_score=score["overall"],
                confidence=self._calculate_confidence(score),
                method=self.config.method,
                feature_scores=score,
            )
            scored_pairs.append(pair)

        # Select best alignments
        aligned, uncertain = self._select_alignments(scored_pairs, source_candidates)

        # Find unmatched entities
        aligned_source_ids = {p.entity1.entity_id for p in aligned}
        aligned_target_ids = {p.entity2.entity_id for p in aligned}

        unmatched_source = [
            e.entity_id for e in source_candidates if e.entity_id not in aligned_source_ids
        ]
        unmatched_target = [
            e.entity_id for e in target_candidates if e.entity_id not in aligned_target_ids
        ]

        # Calculate metrics
        precision = len(aligned) / len(source_candidates) if source_candidates else 0
        recall = len(aligned) / len(target_candidates) if target_candidates else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return AlignmentResult(
            aligned_pairs=aligned,
            uncertain_pairs=uncertain,
            unmatched_source=unmatched_source,
            unmatched_target=unmatched_target,
            total_source=len(source_candidates),
            total_target=len(target_candidates),
            precision=precision,
            recall=recall,
            f1_score=f1,
            config=self.config,
        )

    def align_single(
        self,
        entity: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> list[AlignmentPair]:
        """Find alignment candidates for a single entity."""
        source = self._to_candidate(entity, "source")
        target_candidates = [self._to_candidate(e, "target") for e in candidates]

        results = []
        for target in target_candidates:
            score = self._calculate_similarity(source, target)
            if score["overall"] >= self.config.uncertain_threshold:
                pair = AlignmentPair(
                    entity1=source,
                    entity2=target,
                    similarity_score=score["overall"],
                    confidence=self._calculate_confidence(score),
                    method=self.config.method,
                    feature_scores=score,
                    status=self._determine_status(score["overall"]),
                )
                results.append(pair)

        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[: self.config.max_candidates]

    def _to_candidate(
        self,
        entity: dict[str, Any],
        source: str,
    ) -> EntityCandidate:
        """Convert entity dict to EntityCandidate."""
        return EntityCandidate(
            entity_id=str(entity.get("id", entity.get("entity_id", ""))),
            entity_name=str(entity.get("name", entity.get("label", ""))),
            entity_type=str(entity.get("type", entity.get("entity_type", ""))),
            source=source,
            attributes=entity.get("attributes", entity.get("properties", {})),
            embedding=entity.get("embedding"),
        )

    def _generate_candidates(
        self,
        source: list[EntityCandidate],
        target: list[EntityCandidate],
    ) -> list[tuple[EntityCandidate, EntityCandidate]]:
        """Generate candidate pairs using blocking."""
        candidates = []

        # Create blocks for target entities
        blocks: dict[str, list[EntityCandidate]] = {}

        for entity in target:
            keys = self._get_blocking_keys(entity)
            for key in keys:
                if key not in blocks:
                    blocks[key] = []
                if len(blocks[key]) < self.blocking_config.max_block_size:
                    blocks[key].append(entity)

        # Match source entities to blocks
        for src_entity in source:
            src_keys = self._get_blocking_keys(src_entity)
            matched_targets = set()

            for key in src_keys:
                if key in blocks:
                    for tgt_entity in blocks[key]:
                        if tgt_entity.entity_id not in matched_targets:
                            candidates.append((src_entity, tgt_entity))
                            matched_targets.add(tgt_entity.entity_id)

            # If no blocking matches, compare with all (fallback)
            if not matched_targets and len(target) <= 100:
                for tgt_entity in target:
                    candidates.append((src_entity, tgt_entity))

        return candidates

    def _get_blocking_keys(self, entity: EntityCandidate) -> list[str]:
        """Generate blocking keys for an entity."""
        keys = []

        for key_type in self.blocking_config.blocking_keys:
            if key_type == "type" and entity.entity_type:
                keys.append(f"type:{entity.entity_type.lower()}")
            elif key_type == "first_letter" and entity.entity_name:
                keys.append(f"fl:{entity.entity_name[0].lower()}")
            elif key_type == "first_word" and entity.entity_name:
                first_word = entity.entity_name.split()[0].lower() if entity.entity_name else ""
                keys.append(f"fw:{first_word}")

        return keys if keys else ["default"]

    def _calculate_similarity(
        self,
        entity1: EntityCandidate,
        entity2: EntityCandidate,
    ) -> dict[str, float]:
        """Calculate similarity between two entities."""
        scores = {}

        # Name similarity
        name_sim = self._string_similarity(
            entity1.entity_name.lower(),
            entity2.entity_name.lower(),
        )
        scores["name"] = name_sim

        # Type similarity
        if self.config.use_type_matching:
            if entity1.entity_type and entity2.entity_type:
                if entity1.entity_type.lower() == entity2.entity_type.lower():
                    scores["type"] = 1.0
                else:
                    scores["type"] = 0.0
            else:
                scores["type"] = 0.5  # Unknown type
        else:
            scores["type"] = 1.0

        # Attribute similarity
        attr_sim = self._attribute_similarity(entity1.attributes, entity2.attributes)
        scores["attributes"] = attr_sim

        # Embedding similarity (if available)
        if entity1.embedding and entity2.embedding:
            emb_sim = self._cosine_similarity(entity1.embedding, entity2.embedding)
            scores["embedding"] = emb_sim
        else:
            scores["embedding"] = scores["name"]  # Fallback to name

        # Calculate weighted overall score
        weights = self.config.attribute_weights
        overall = (
            weights.get("name", 0.5) * scores["name"]
            + weights.get("type", 0.2) * scores["type"]
            + weights.get("attributes", 0.3) * scores["attributes"]
        )

        # Apply type mismatch penalty
        if scores["type"] == 0.0:
            overall -= self.config.type_mismatch_penalty

        scores["overall"] = max(0.0, min(1.0, overall))

        return scores

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity based on configured metric."""
        if not s1 or not s2:
            return 0.0

        if s1 == s2:
            return 1.0

        if self.config.similarity_metric == SimilarityMetric.JACCARD:
            return self._jaccard_similarity(s1, s2)
        elif self.config.similarity_metric == SimilarityMetric.LEVENSHTEIN:
            return self._levenshtein_similarity(s1, s2)
        elif self.config.similarity_metric == SimilarityMetric.JARO_WINKLER:
            return self._jaro_winkler_similarity(s1, s2)
        elif self.config.similarity_metric == SimilarityMetric.NGRAM:
            return self._ngram_similarity(s1, s2)
        else:
            return self._jaro_winkler_similarity(s1, s2)

    def _jaccard_similarity(self, s1: str, s2: str) -> float:
        """Calculate Jaccard similarity on character sets."""
        set1 = set(s1.lower())
        set2 = set(s2.lower())
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def _levenshtein_similarity(self, s1: str, s2: str) -> float:
        """Calculate normalized Levenshtein similarity."""
        if len(s1) == 0 and len(s2) == 0:
            return 1.0

        # Simple Levenshtein distance
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        distance = dp[m][n]
        max_len = max(m, n)
        return 1.0 - (distance / max_len) if max_len > 0 else 1.0

    def _jaro_winkler_similarity(self, s1: str, s2: str) -> float:
        """Calculate Jaro-Winkler similarity."""
        if s1 == s2:
            return 1.0

        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0

        match_distance = max(len1, len2) // 2 - 1
        match_distance = max(0, match_distance)

        s1_matches = [False] * len1
        s2_matches = [False] * len2

        matches = 0
        transpositions = 0

        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)

            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

        if matches == 0:
            return 0.0

        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

        jaro = (
            matches / len1 + matches / len2 + (matches - transpositions / 2) / matches
        ) / 3

        # Winkler modification
        prefix = 0
        for i in range(min(len1, len2, 4)):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break

        return jaro + prefix * 0.1 * (1 - jaro)

    def _ngram_similarity(self, s1: str, s2: str) -> float:
        """Calculate n-gram similarity."""
        n = self.config.ngram_size

        def get_ngrams(s: str) -> set[str]:
            return {s[i : i + n] for i in range(max(len(s) - n + 1, 1))}

        ngrams1 = get_ngrams(s1.lower())
        ngrams2 = get_ngrams(s2.lower())

        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)

        return intersection / union if union > 0 else 0.0

    def _attribute_similarity(
        self,
        attrs1: dict[str, Any],
        attrs2: dict[str, Any],
    ) -> float:
        """Calculate similarity between attribute dictionaries."""
        if not attrs1 or not attrs2:
            return 0.5  # Neutral if no attributes

        common_keys = set(attrs1.keys()) & set(attrs2.keys())
        if not common_keys:
            return 0.3  # Low score for no common keys

        scores = []
        for key in common_keys:
            v1, v2 = str(attrs1[key]).lower(), str(attrs2[key]).lower()
            if v1 == v2:
                scores.append(1.0)
            else:
                scores.append(self._string_similarity(v1, v2))

        return sum(scores) / len(scores) if scores else 0.5

    def _cosine_similarity(
        self,
        vec1: list[float],
        vec2: list[float],
    ) -> float:
        """Calculate cosine similarity between vectors."""
        if len(vec1) != len(vec2):
            return 0.0

        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    def _calculate_confidence(self, scores: dict[str, float]) -> float:
        """Calculate confidence in alignment decision."""
        overall = scores.get("overall", 0.0)

        # Higher confidence if scores are consistent
        feature_scores = [
            scores.get("name", 0),
            scores.get("type", 0),
            scores.get("attributes", 0),
        ]

        variance = sum((s - overall) ** 2 for s in feature_scores) / len(feature_scores)
        consistency_boost = max(0, 0.1 - variance)

        confidence = overall + consistency_boost
        return min(1.0, max(0.0, confidence))

    def _determine_status(self, score: float) -> AlignmentStatus:
        """Determine alignment status based on score."""
        if score >= self.config.alignment_threshold:
            return AlignmentStatus.ALIGNED
        elif score >= self.config.uncertain_threshold:
            return AlignmentStatus.UNCERTAIN
        else:
            return AlignmentStatus.NOT_ALIGNED

    def _select_alignments(
        self,
        pairs: list[AlignmentPair],
        source_entities: list[EntityCandidate],
    ) -> tuple[list[AlignmentPair], list[AlignmentPair]]:
        """Select best alignments using Hungarian-like algorithm."""
        aligned = []
        uncertain = []

        # Sort pairs by score
        pairs.sort(key=lambda x: x.similarity_score, reverse=True)

        used_source = set()
        used_target = set()

        for pair in pairs:
            src_id = pair.entity1.entity_id
            tgt_id = pair.entity2.entity_id

            if src_id in used_source or tgt_id in used_target:
                continue

            if pair.similarity_score >= self.config.alignment_threshold:
                pair.status = AlignmentStatus.ALIGNED
                aligned.append(pair)
                used_source.add(src_id)
                used_target.add(tgt_id)
            elif pair.similarity_score >= self.config.uncertain_threshold:
                pair.status = AlignmentStatus.UNCERTAIN
                uncertain.append(pair)

        return aligned, uncertain


# Global instance
_entity_aligner: Optional[EntityAligner] = None


def get_entity_aligner() -> EntityAligner:
    """Get or create the global entity aligner instance."""
    global _entity_aligner
    if _entity_aligner is None:
        _entity_aligner = EntityAligner()
    return _entity_aligner
