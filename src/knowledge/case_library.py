"""
Case Library for SuperInsight Platform.

Manages case entries for knowledge learning and retrieval.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from uuid import UUID
from pathlib import Path
from collections import defaultdict

from .models import (
    CaseEntry,
    CaseStatus,
    FeedbackEntry
)

logger = logging.getLogger(__name__)


class CaseLibrary:
    """
    Case library for storing and retrieving cases.

    Features:
    - Case CRUD operations
    - Similar case retrieval
    - Case outcome tracking
    - Case archival
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize CaseLibrary.

        Args:
            storage_path: Path for persistent storage
        """
        self.storage_path = Path(storage_path) if storage_path else None
        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory storage
        self._cases: Dict[UUID, CaseEntry] = {}
        self._status_index: Dict[CaseStatus, Set[UUID]] = defaultdict(set)
        self._tag_index: Dict[str, Set[UUID]] = defaultdict(set)

        # Load from storage
        if self.storage_path:
            self._load_from_storage()

    def add_case(self, case: CaseEntry) -> UUID:
        """
        Add a case.

        Args:
            case: Case to add

        Returns:
            Case ID
        """
        self._cases[case.id] = case
        self._update_indices(case)

        logger.info(f"Added case: {case.id} - {case.title}")

        if self.storage_path:
            self._save_case(case)

        return case.id

    def update_case(self, case_id: UUID,
                   updates: Dict[str, Any]) -> Optional[UUID]:
        """
        Update a case.

        Args:
            case_id: Case ID
            updates: Field updates

        Returns:
            Case ID if updated, None if not found
        """
        if case_id not in self._cases:
            return None

        case = self._cases[case_id]

        # Remove from indices
        self._remove_from_indices(case)

        # Apply updates
        for key, value in updates.items():
            if hasattr(case, key):
                setattr(case, key, value)

        case.updated_at = datetime.now()

        # Re-add to indices
        self._update_indices(case)

        logger.info(f"Updated case: {case_id}")

        if self.storage_path:
            self._save_case(case)

        return case_id

    def get_case(self, case_id: UUID) -> Optional[CaseEntry]:
        """Get a case by ID."""
        return self._cases.get(case_id)

    def delete_case(self, case_id: UUID) -> bool:
        """Delete a case."""
        if case_id not in self._cases:
            return False

        case = self._cases[case_id]
        self._remove_from_indices(case)
        del self._cases[case_id]

        if self.storage_path:
            self._delete_case_file(case_id)

        logger.info(f"Deleted case: {case_id}")
        return True

    def find_similar_cases(self, problem: str,
                          context: Optional[Dict[str, Any]] = None,
                          limit: int = 5) -> List[CaseEntry]:
        """
        Find cases similar to the given problem.

        Args:
            problem: Problem description
            context: Optional context
            limit: Maximum results

        Returns:
            List of similar cases
        """
        candidates = []
        problem_lower = problem.lower()
        problem_words = set(problem_lower.split())

        for case in self._cases.values():
            # Only consider resolved cases
            if case.status != CaseStatus.RESOLVED:
                continue

            similarity = self._calculate_similarity(problem_lower, problem_words, case, context)

            if similarity > 0.1:
                candidates.append((case, similarity))

        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)

        return [c[0] for c in candidates[:limit]]

    def update_case_outcome(self, case_id: UUID,
                           solution: str,
                           outcome: str,
                           feedback: Optional[str] = None,
                           rating: Optional[int] = None) -> bool:
        """
        Update case outcome.

        Args:
            case_id: Case ID
            solution: Solution applied
            outcome: Outcome description
            feedback: Optional feedback
            rating: Optional rating (1-5)

        Returns:
            True if updated
        """
        case = self.get_case(case_id)
        if not case:
            return False

        case.resolve(solution, outcome)
        if feedback:
            case.feedback = feedback
        if rating:
            case.rating = rating

        self._update_indices(case)

        if self.storage_path:
            self._save_case(case)

        logger.info(f"Updated case outcome: {case_id}")
        return True

    def archive_case(self, case_id: UUID) -> bool:
        """Archive a case."""
        case = self.get_case(case_id)
        if not case:
            return False

        self._remove_from_indices(case)
        case.archive()
        self._update_indices(case)

        if self.storage_path:
            self._save_case(case)

        logger.info(f"Archived case: {case_id}")
        return True

    def get_cases_by_status(self, status: CaseStatus) -> List[CaseEntry]:
        """Get cases by status."""
        case_ids = self._status_index.get(status, set())
        return [self._cases[cid] for cid in case_ids if cid in self._cases]

    def get_cases_by_tag(self, tag: str) -> List[CaseEntry]:
        """Get cases by tag."""
        tag_lower = tag.lower()
        case_ids = self._tag_index.get(tag_lower, set())
        return [self._cases[cid] for cid in case_ids if cid in self._cases]

    def get_recent_cases(self, days: int = 7, limit: int = 20) -> List[CaseEntry]:
        """Get recent cases."""
        cutoff = datetime.now() - timedelta(days=days)
        recent = [c for c in self._cases.values() if c.created_at >= cutoff]
        recent.sort(key=lambda c: c.created_at, reverse=True)
        return recent[:limit]

    def get_high_rated_cases(self, min_rating: int = 4, limit: int = 20) -> List[CaseEntry]:
        """Get high-rated resolved cases."""
        high_rated = [
            c for c in self._cases.values()
            if c.status == CaseStatus.RESOLVED and c.rating and c.rating >= min_rating
        ]
        high_rated.sort(key=lambda c: (c.rating or 0, c.resolved_at or c.created_at), reverse=True)
        return high_rated[:limit]

    def get_pending_cases(self, limit: int = 50) -> List[CaseEntry]:
        """Get pending cases awaiting resolution."""
        pending = self.get_cases_by_status(CaseStatus.PENDING)
        pending.sort(key=lambda c: c.created_at)
        return pending[:limit]

    def link_cases(self, case_id1: UUID, case_id2: UUID) -> bool:
        """Link two related cases."""
        case1 = self.get_case(case_id1)
        case2 = self.get_case(case_id2)

        if not case1 or not case2:
            return False

        if case_id2 not in case1.related_cases:
            case1.related_cases.append(case_id2)
        if case_id1 not in case2.related_cases:
            case2.related_cases.append(case_id1)

        case1.updated_at = datetime.now()
        case2.updated_at = datetime.now()

        if self.storage_path:
            self._save_case(case1)
            self._save_case(case2)

        return True

    def _update_indices(self, case: CaseEntry) -> None:
        """Update indices for a case."""
        self._status_index[case.status].add(case.id)

        for tag in case.tags:
            self._tag_index[tag.lower()].add(case.id)

    def _remove_from_indices(self, case: CaseEntry) -> None:
        """Remove case from indices."""
        for status_set in self._status_index.values():
            status_set.discard(case.id)

        for tag in case.tags:
            self._tag_index[tag.lower()].discard(case.id)

    def _calculate_similarity(self, problem_lower: str,
                             problem_words: Set[str],
                             case: CaseEntry,
                             context: Optional[Dict[str, Any]]) -> float:
        """Calculate similarity between problem and case."""
        score = 0.0

        # Problem text similarity
        case_problem_lower = case.problem.lower()
        case_words = set(case_problem_lower.split())

        if problem_words and case_words:
            jaccard = len(problem_words & case_words) / len(problem_words | case_words)
            score += 0.4 * jaccard

        # Substring matching
        if problem_lower in case_problem_lower or case_problem_lower in problem_lower:
            score += 0.3

        # Tag matching with context
        if context:
            context_tags = set(str(v).lower() for v in context.values())
            case_tags = set(t.lower() for t in case.tags)

            if context_tags and case_tags:
                tag_overlap = len(context_tags & case_tags) / len(context_tags | case_tags)
                score += 0.3 * tag_overlap

        # Boost for high-rated cases
        if case.rating and case.rating >= 4:
            score *= 1.1

        return min(1.0, score)

    def _save_case(self, case: CaseEntry) -> None:
        """Save case to storage."""
        if not self.storage_path:
            return

        file_path = self.storage_path / f"{case.id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(case.model_dump(mode='json'), f, ensure_ascii=False, indent=2, default=str)

    def _delete_case_file(self, case_id: UUID) -> None:
        """Delete case file from storage."""
        if not self.storage_path:
            return

        file_path = self.storage_path / f"{case_id}.json"
        if file_path.exists():
            file_path.unlink()

    def _load_from_storage(self) -> None:
        """Load cases from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return

        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                case = CaseEntry(**data)
                self._cases[case.id] = case
                self._update_indices(case)
            except Exception as e:
                logger.error(f"Failed to load case from {file_path}: {e}")

        logger.info(f"Loaded {len(self._cases)} cases from storage")

    def get_statistics(self) -> Dict[str, Any]:
        """Get case library statistics."""
        status_counts = {}
        for status in CaseStatus:
            status_counts[status.value] = len(self._status_index.get(status, set()))

        resolved_cases = self.get_cases_by_status(CaseStatus.RESOLVED)
        avg_rating = 0.0
        if resolved_cases:
            rated = [c for c in resolved_cases if c.rating]
            if rated:
                avg_rating = sum(c.rating for c in rated) / len(rated)

        return {
            "total_cases": len(self._cases),
            "by_status": status_counts,
            "total_tags": len(self._tag_index),
            "average_rating": avg_rating,
            "recent_7_days": len(self.get_recent_cases(days=7, limit=1000))
        }

    def export_for_learning(self, min_rating: int = 3) -> List[CaseEntry]:
        """Export cases suitable for rule learning."""
        return [
            c for c in self._cases.values()
            if c.status == CaseStatus.RESOLVED and (not c.rating or c.rating >= min_rating)
        ]


# Global instance
_case_library: Optional[CaseLibrary] = None


def get_case_library(storage_path: Optional[str] = None) -> CaseLibrary:
    """Get or create global CaseLibrary instance."""
    global _case_library

    if _case_library is None:
        _case_library = CaseLibrary(storage_path)

    return _case_library
