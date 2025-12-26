"""
Knowledge Auto Updater for SuperInsight Platform.

Automatically updates knowledge base from feedback and cases.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from uuid import UUID

from .models import (
    KnowledgeEntry,
    KnowledgeRule,
    KnowledgeCategory,
    KnowledgeUpdateResult,
    FeedbackEntry,
    QualityScore
)
from .knowledge_base import KnowledgeBase, get_knowledge_base
from .rule_engine import RuleEngine, get_rule_engine
from .case_library import CaseLibrary, get_case_library

logger = logging.getLogger(__name__)


class KnowledgeAutoUpdater:
    """
    Automatic knowledge updater.

    Features:
    - Scheduled updates
    - Feedback processing
    - Knowledge merging
    - Quality control
    """

    def __init__(self,
                 knowledge_base: Optional[KnowledgeBase] = None,
                 rule_engine: Optional[RuleEngine] = None,
                 case_library: Optional[CaseLibrary] = None):
        """
        Initialize KnowledgeAutoUpdater.

        Args:
            knowledge_base: Knowledge base instance
            rule_engine: Rule engine instance
            case_library: Case library instance
        """
        self._kb = knowledge_base or get_knowledge_base()
        self._re = rule_engine or get_rule_engine()
        self._cl = case_library or get_case_library()

        self._is_running = False
        self._update_task: Optional[asyncio.Task] = None
        self._update_interval = 3600  # 1 hour
        self._last_update: Optional[datetime] = None

        # Quality thresholds
        self._min_quality_score = 0.5
        self._min_confidence = 0.6
        self._min_support = 0.1

    async def start(self, interval: int = 3600) -> None:
        """
        Start automatic updates.

        Args:
            interval: Update interval in seconds
        """
        if self._is_running:
            return

        self._update_interval = interval
        self._is_running = True
        self._update_task = asyncio.create_task(self._update_loop())

        logger.info(f"Knowledge auto-updater started with {interval}s interval")

    async def stop(self) -> None:
        """Stop automatic updates."""
        self._is_running = False

        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

        logger.info("Knowledge auto-updater stopped")

    async def _update_loop(self) -> None:
        """Main update loop."""
        while self._is_running:
            try:
                await self.run_update()
            except Exception as e:
                logger.error(f"Update failed: {e}")

            await asyncio.sleep(self._update_interval)

    async def run_update(self) -> KnowledgeUpdateResult:
        """
        Run a full update cycle.

        Returns:
            Update result
        """
        import time
        start_time = time.time()

        result = KnowledgeUpdateResult(success=True)

        try:
            # Process pending feedback
            feedback_result = await self.process_feedback()
            result.entries_updated += feedback_result.get("entries_updated", 0)

            # Learn rules from cases
            learning_result = await self.learn_from_cases()
            result.rules_learned = learning_result.get("rules_learned", 0)
            result.cases_processed = learning_result.get("cases_processed", 0)

            # Quality check
            quality_result = await self.quality_check()
            result.quality_scores = quality_result.get("scores", {})
            result.entries_removed = quality_result.get("removed", 0)

            # Optimize rules
            optimize_result = self._re.optimize_rules()
            result.warnings.extend([
                f"Disabled {optimize_result['disabled']} ineffective rules",
                f"Removed {optimize_result['removed']} very low performance rules"
            ])

            self._last_update = datetime.now()

        except Exception as e:
            result.success = False
            result.errors.append(str(e))
            logger.error(f"Update cycle failed: {e}")

        result.processing_time = time.time() - start_time

        logger.info(f"Update completed: {result.entries_updated} updated, "
                   f"{result.rules_learned} rules learned")

        return result

    async def process_feedback(self) -> Dict[str, Any]:
        """
        Process pending feedback.

        Returns:
            Processing result
        """
        pending = self._kb.get_pending_feedback()
        entries_updated = 0

        for feedback in pending:
            try:
                if await self._apply_feedback(feedback):
                    entries_updated += 1
                feedback.is_processed = True
            except Exception as e:
                logger.warning(f"Failed to process feedback {feedback.id}: {e}")

        return {"entries_updated": entries_updated, "total_feedback": len(pending)}

    async def _apply_feedback(self, feedback: FeedbackEntry) -> bool:
        """Apply a single feedback entry."""
        if feedback.target_type == "entry":
            entry = self._kb.get_entry(feedback.target_id)
            if not entry:
                return False

            if feedback.feedback_type == "incorrect":
                # Lower confidence
                new_confidence = max(0.1, entry.confidence - 0.2)
                self._kb.update_entry(feedback.target_id, {"confidence": new_confidence})

            elif feedback.feedback_type == "improve" and feedback.suggested_change:
                # Apply suggestion
                self._kb.update_entry(feedback.target_id, {"content": feedback.suggested_change})

            elif feedback.feedback_type == "correct":
                # Boost confidence
                new_confidence = min(1.0, entry.confidence + 0.1)
                self._kb.update_entry(feedback.target_id, {"confidence": new_confidence})

            return True

        elif feedback.target_type == "rule":
            rule = self._re.get_rule(feedback.target_id)
            if not rule:
                return False

            if feedback.feedback_type == "incorrect":
                rule.is_enabled = False
                self._re.update_rule(feedback.target_id, {"is_enabled": False})

            elif feedback.feedback_type == "correct":
                rule.record_success()

            return True

        return False

    async def learn_from_cases(self) -> Dict[str, Any]:
        """
        Learn rules from cases.

        Returns:
            Learning result
        """
        # Get resolved cases suitable for learning
        cases = self._cl.export_for_learning(min_rating=3)

        if len(cases) < 5:
            return {"rules_learned": 0, "cases_processed": 0}

        # Learn association rules
        association_rules = self._re.learn_rule(
            cases,
            min_support=self._min_support,
            min_confidence=self._min_confidence
        )

        # Learn classification rules
        classification_rules = self._re.learn_rule(
            cases,
            rule_type=self._re._rules.get(list(self._re._rules.keys())[0]).rule_type if self._re._rules else None,
            min_support=self._min_support,
            min_confidence=self._min_confidence
        ) if self._re._rules else []

        total_rules = len(association_rules) + len(classification_rules)

        # Create knowledge entries from learned patterns
        entries_created = 0
        for rule in association_rules[:10]:  # Limit to top 10
            entry = KnowledgeEntry(
                title=f"Learned Pattern: {rule.name}",
                content=rule.description,
                category=KnowledgeCategory.PATTERN,
                tags=["learned", "association"],
                confidence=rule.confidence,
                source="auto_learning"
            )
            self._kb.add_entry(entry)
            entries_created += 1

        return {
            "rules_learned": total_rules,
            "cases_processed": len(cases),
            "entries_created": entries_created
        }

    async def quality_check(self) -> Dict[str, Any]:
        """
        Perform quality check on knowledge base.

        Returns:
            Quality check result
        """
        entries = list(self._kb._entries.values())
        removed = 0
        scores = {"average": 0.0, "low_quality": 0, "high_quality": 0}

        total_score = 0.0

        for entry in entries:
            # Validate and score
            quality = self._kb.validate_entry(entry)
            entry.quality = quality
            total_score += quality.overall

            if quality.overall < self._min_quality_score:
                scores["low_quality"] += 1

                # Remove very low quality entries
                if quality.overall < 0.3 and entry.confidence < 0.5:
                    self._kb.delete_entry(entry.id)
                    removed += 1

            elif quality.overall >= 0.8:
                scores["high_quality"] += 1

        if entries:
            scores["average"] = total_score / len(entries)

        return {"scores": scores, "removed": removed, "total_checked": len(entries)}

    def merge_knowledge(self, source_entries: List[KnowledgeEntry]) -> KnowledgeUpdateResult:
        """
        Merge knowledge from external source.

        Args:
            source_entries: Entries to merge

        Returns:
            Merge result
        """
        result = KnowledgeUpdateResult(success=True)

        for entry in source_entries:
            try:
                # Check for existing similar entries
                search_result = self._kb.search(entry.title, category=entry.category, limit=5)

                if search_result.entries:
                    # Check for duplicates
                    best_match = search_result.entries[0]
                    similarity = self._calculate_entry_similarity(entry, best_match)

                    if similarity > 0.9:
                        # Merge into existing
                        if entry.confidence > best_match.confidence:
                            self._kb.update_entry(best_match.id, entry.model_dump())
                            result.entries_updated += 1
                        continue

                # Add new entry
                self._kb.add_entry(entry)
                result.entries_added += 1

            except Exception as e:
                result.errors.append(f"Failed to merge entry {entry.id}: {e}")

        return result

    def _calculate_entry_similarity(self, entry1: KnowledgeEntry,
                                   entry2: KnowledgeEntry) -> float:
        """Calculate similarity between two entries."""
        score = 0.0

        # Title similarity
        if entry1.title.lower() == entry2.title.lower():
            score += 0.4
        elif entry1.title.lower() in entry2.title.lower() or entry2.title.lower() in entry1.title.lower():
            score += 0.2

        # Content similarity
        words1 = set(entry1.content.lower().split())
        words2 = set(entry2.content.lower().split())
        if words1 and words2:
            jaccard = len(words1 & words2) / len(words1 | words2)
            score += 0.4 * jaccard

        # Category match
        if entry1.category == entry2.category:
            score += 0.2

        return score

    def schedule_update(self, delay_seconds: int = 0) -> None:
        """
        Schedule an update.

        Args:
            delay_seconds: Delay before update
        """
        asyncio.create_task(self._delayed_update(delay_seconds))

    async def _delayed_update(self, delay: int) -> None:
        """Run update after delay."""
        if delay > 0:
            await asyncio.sleep(delay)
        await self.run_update()

    def get_status(self) -> Dict[str, Any]:
        """Get updater status."""
        return {
            "is_running": self._is_running,
            "update_interval": self._update_interval,
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "knowledge_base_stats": self._kb.get_statistics(),
            "rule_engine_stats": self._re.get_statistics(),
            "case_library_stats": self._cl.get_statistics()
        }

    def set_thresholds(self,
                      min_quality: Optional[float] = None,
                      min_confidence: Optional[float] = None,
                      min_support: Optional[float] = None) -> None:
        """
        Set quality thresholds.

        Args:
            min_quality: Minimum quality score
            min_confidence: Minimum confidence for rules
            min_support: Minimum support for rules
        """
        if min_quality is not None:
            self._min_quality_score = min_quality
        if min_confidence is not None:
            self._min_confidence = min_confidence
        if min_support is not None:
            self._min_support = min_support


# Global instance
_auto_updater: Optional[KnowledgeAutoUpdater] = None


def get_auto_updater() -> KnowledgeAutoUpdater:
    """Get or create global KnowledgeAutoUpdater instance."""
    global _auto_updater

    if _auto_updater is None:
        _auto_updater = KnowledgeAutoUpdater()

    return _auto_updater
