"""
Knowledge Base for SuperInsight Platform.

Core knowledge storage and retrieval system.
"""

import logging
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from uuid import UUID
from pathlib import Path

from .models import (
    KnowledgeEntry,
    KnowledgeCategory,
    QualityScore,
    KnowledgeSearchResult,
    FeedbackEntry
)

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    Core knowledge base for storing and retrieving knowledge entries.

    Features:
    - CRUD operations for knowledge entries
    - Search and retrieval
    - Version management
    - Quality scoring
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize KnowledgeBase.

        Args:
            storage_path: Path for persistent storage
        """
        self.storage_path = Path(storage_path) if storage_path else None
        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory storage
        self._entries: Dict[UUID, KnowledgeEntry] = {}
        self._index: Dict[str, Set[UUID]] = {}  # tag -> entry IDs
        self._category_index: Dict[KnowledgeCategory, Set[UUID]] = {}
        self._feedback: List[FeedbackEntry] = []

        # Load from storage if available
        if self.storage_path:
            self._load_from_storage()

    def add_entry(self, entry: KnowledgeEntry) -> UUID:
        """
        Add a knowledge entry.

        Args:
            entry: Knowledge entry to add

        Returns:
            Entry ID
        """
        # Check for duplicates
        if entry.id in self._entries:
            logger.warning(f"Entry {entry.id} already exists, updating instead")
            return self.update_entry(entry.id, entry.model_dump())

        self._entries[entry.id] = entry
        self._update_indices(entry)

        logger.info(f"Added knowledge entry: {entry.id} - {entry.title}")

        if self.storage_path:
            self._save_entry(entry)

        return entry.id

    def update_entry(self, entry_id: UUID,
                    updates: Dict[str, Any]) -> Optional[UUID]:
        """
        Update a knowledge entry.

        Args:
            entry_id: Entry ID to update
            updates: Field updates

        Returns:
            Entry ID if updated, None if not found
        """
        if entry_id not in self._entries:
            logger.warning(f"Entry {entry_id} not found")
            return None

        entry = self._entries[entry_id]

        # Remove from indices before update
        self._remove_from_indices(entry)

        # Apply updates
        entry.update(**updates)

        # Re-add to indices
        self._update_indices(entry)

        logger.info(f"Updated knowledge entry: {entry_id}")

        if self.storage_path:
            self._save_entry(entry)

        return entry_id

    def get_entry(self, entry_id: UUID) -> Optional[KnowledgeEntry]:
        """Get a knowledge entry by ID."""
        return self._entries.get(entry_id)

    def delete_entry(self, entry_id: UUID) -> bool:
        """
        Delete a knowledge entry.

        Args:
            entry_id: Entry ID to delete

        Returns:
            True if deleted, False if not found
        """
        if entry_id not in self._entries:
            return False

        entry = self._entries[entry_id]
        self._remove_from_indices(entry)
        del self._entries[entry_id]

        logger.info(f"Deleted knowledge entry: {entry_id}")

        if self.storage_path:
            self._delete_entry_file(entry_id)

        return True

    def search(self, query: str,
              category: Optional[KnowledgeCategory] = None,
              tags: Optional[List[str]] = None,
              limit: int = 10) -> KnowledgeSearchResult:
        """
        Search knowledge entries.

        Args:
            query: Search query
            category: Filter by category
            tags: Filter by tags
            limit: Maximum results

        Returns:
            Search result
        """
        import time
        start_time = time.time()

        candidates = set(self._entries.keys())

        # Filter by category
        if category:
            if category in self._category_index:
                candidates &= self._category_index[category]
            else:
                candidates = set()

        # Filter by tags
        if tags:
            for tag in tags:
                tag_lower = tag.lower()
                if tag_lower in self._index:
                    candidates &= self._index[tag_lower]

        # Search in content
        query_lower = query.lower()
        results = []

        for entry_id in candidates:
            entry = self._entries[entry_id]
            score = self._calculate_relevance(entry, query_lower)
            if score > 0:
                results.append((entry, score))

        # Sort by relevance
        results.sort(key=lambda x: x[1], reverse=True)

        # Limit results
        entries = [r[0] for r in results[:limit]]

        search_time = time.time() - start_time

        return KnowledgeSearchResult(
            entries=entries,
            total_count=len(results),
            query=query,
            search_time=search_time
        )

    def get_related(self, entry_id: UUID, limit: int = 5) -> List[KnowledgeEntry]:
        """
        Get entries related to the given entry.

        Args:
            entry_id: Entry ID
            limit: Maximum results

        Returns:
            List of related entries
        """
        if entry_id not in self._entries:
            return []

        entry = self._entries[entry_id]
        related = []

        # Find entries with same category
        if entry.category in self._category_index:
            for other_id in self._category_index[entry.category]:
                if other_id != entry_id:
                    other = self._entries[other_id]
                    # Calculate similarity
                    similarity = self._calculate_similarity(entry, other)
                    if similarity > 0.3:
                        related.append((other, similarity))

        # Sort by similarity
        related.sort(key=lambda x: x[1], reverse=True)

        return [r[0] for r in related[:limit]]

    def validate_entry(self, entry: KnowledgeEntry) -> QualityScore:
        """
        Validate and score a knowledge entry.

        Args:
            entry: Entry to validate

        Returns:
            Quality score
        """
        score = QualityScore()

        # Accuracy: based on confidence and source
        score.accuracy = entry.confidence
        if entry.source:
            score.accuracy = min(1.0, score.accuracy + 0.1)

        # Completeness: based on filled fields
        filled_fields = 0
        total_fields = 5  # title, content, tags, metadata, source
        if entry.title:
            filled_fields += 1
        if entry.content:
            filled_fields += 1
        if entry.tags:
            filled_fields += 1
        if entry.metadata:
            filled_fields += 1
        if entry.source:
            filled_fields += 1
        score.completeness = filled_fields / total_fields

        # Consistency: check for duplicates
        duplicates = self._find_duplicates(entry)
        score.consistency = 1.0 if not duplicates else 0.5

        # Relevance: based on category matching
        score.relevance = 0.8 if entry.category else 0.5

        # Timeliness: based on update time
        age_days = (datetime.now() - entry.updated_at).days
        if age_days < 7:
            score.timeliness = 1.0
        elif age_days < 30:
            score.timeliness = 0.8
        elif age_days < 90:
            score.timeliness = 0.6
        else:
            score.timeliness = 0.4

        return score

    def add_feedback(self, feedback: FeedbackEntry) -> None:
        """Add feedback for an entry."""
        self._feedback.append(feedback)
        logger.info(f"Added feedback for {feedback.target_id}")

    def get_pending_feedback(self) -> List[FeedbackEntry]:
        """Get unprocessed feedback."""
        return [f for f in self._feedback if not f.is_processed]

    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        category_counts = {}
        for cat in KnowledgeCategory:
            if cat in self._category_index:
                category_counts[cat.value] = len(self._category_index[cat])
            else:
                category_counts[cat.value] = 0

        return {
            "total_entries": len(self._entries),
            "by_category": category_counts,
            "total_tags": len(self._index),
            "pending_feedback": len(self.get_pending_feedback()),
            "average_quality": self._calculate_average_quality()
        }

    def _update_indices(self, entry: KnowledgeEntry) -> None:
        """Update search indices for an entry."""
        # Tag index
        for tag in entry.tags:
            tag_lower = tag.lower()
            if tag_lower not in self._index:
                self._index[tag_lower] = set()
            self._index[tag_lower].add(entry.id)

        # Category index
        if entry.category not in self._category_index:
            self._category_index[entry.category] = set()
        self._category_index[entry.category].add(entry.id)

    def _remove_from_indices(self, entry: KnowledgeEntry) -> None:
        """Remove entry from indices."""
        for tag in entry.tags:
            tag_lower = tag.lower()
            if tag_lower in self._index:
                self._index[tag_lower].discard(entry.id)

        if entry.category in self._category_index:
            self._category_index[entry.category].discard(entry.id)

    def _calculate_relevance(self, entry: KnowledgeEntry, query: str) -> float:
        """Calculate relevance score for search."""
        score = 0.0

        # Title match
        if query in entry.title.lower():
            score += 0.5

        # Content match
        if query in entry.content.lower():
            score += 0.3

        # Tag match
        for tag in entry.tags:
            if query in tag.lower():
                score += 0.2

        return min(1.0, score)

    def _calculate_similarity(self, entry1: KnowledgeEntry,
                             entry2: KnowledgeEntry) -> float:
        """Calculate similarity between two entries."""
        score = 0.0

        # Same category
        if entry1.category == entry2.category:
            score += 0.3

        # Shared tags
        tags1 = set(t.lower() for t in entry1.tags)
        tags2 = set(t.lower() for t in entry2.tags)
        if tags1 and tags2:
            jaccard = len(tags1 & tags2) / len(tags1 | tags2)
            score += 0.4 * jaccard

        # Content similarity (simple word overlap)
        words1 = set(entry1.content.lower().split())
        words2 = set(entry2.content.lower().split())
        if words1 and words2:
            overlap = len(words1 & words2) / len(words1 | words2)
            score += 0.3 * overlap

        return score

    def _find_duplicates(self, entry: KnowledgeEntry) -> List[UUID]:
        """Find potential duplicate entries."""
        duplicates = []
        entry_hash = self._content_hash(entry)

        for other_id, other in self._entries.items():
            if other_id != entry.id:
                if self._content_hash(other) == entry_hash:
                    duplicates.append(other_id)

        return duplicates

    def _content_hash(self, entry: KnowledgeEntry) -> str:
        """Generate content hash for duplicate detection."""
        content = f"{entry.title.lower()}{entry.content.lower()}"
        return hashlib.md5(content.encode()).hexdigest()

    def _calculate_average_quality(self) -> float:
        """Calculate average quality score across all entries."""
        if not self._entries:
            return 0.0

        total = sum(e.quality.overall for e in self._entries.values())
        return total / len(self._entries)

    def _save_entry(self, entry: KnowledgeEntry) -> None:
        """Save entry to storage."""
        if not self.storage_path:
            return

        file_path = self.storage_path / f"{entry.id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(entry.model_dump(mode='json'), f, ensure_ascii=False, indent=2, default=str)

    def _delete_entry_file(self, entry_id: UUID) -> None:
        """Delete entry file from storage."""
        if not self.storage_path:
            return

        file_path = self.storage_path / f"{entry_id}.json"
        if file_path.exists():
            file_path.unlink()

    def _load_from_storage(self) -> None:
        """Load entries from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return

        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                entry = KnowledgeEntry(**data)
                self._entries[entry.id] = entry
                self._update_indices(entry)
            except Exception as e:
                logger.error(f"Failed to load entry from {file_path}: {e}")

        logger.info(f"Loaded {len(self._entries)} entries from storage")

    def export_all(self) -> List[Dict[str, Any]]:
        """Export all entries as dictionaries."""
        return [e.model_dump(mode='json') for e in self._entries.values()]

    def import_entries(self, entries: List[Dict[str, Any]]) -> int:
        """Import entries from dictionaries."""
        count = 0
        for data in entries:
            try:
                entry = KnowledgeEntry(**data)
                self.add_entry(entry)
                count += 1
            except Exception as e:
                logger.error(f"Failed to import entry: {e}")
        return count


# Global instance
_knowledge_base: Optional[KnowledgeBase] = None


def get_knowledge_base(storage_path: Optional[str] = None) -> KnowledgeBase:
    """Get or create global KnowledgeBase instance."""
    global _knowledge_base

    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase(storage_path)

    return _knowledge_base
