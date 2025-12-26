"""
Feedback collection for SuperInsight Platform.

Provides:
- Customer feedback collection
- Sentiment analysis
- Feedback categorization
- Feedback statistics
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4
from enum import Enum
from dataclasses import dataclass, field
import re

logger = logging.getLogger(__name__)


class FeedbackSource(str, Enum):
    """Feedback sources."""
    CUSTOMER = "customer"       # 客户反馈
    ANNOTATOR = "annotator"     # 标注员反馈
    REVIEWER = "reviewer"       # 审核员反馈
    SYSTEM = "system"           # 系统生成
    SURVEY = "survey"           # 问卷调查


class FeedbackCategory(str, Enum):
    """Feedback categories."""
    QUALITY = "quality"         # 质量相关
    EFFICIENCY = "efficiency"   # 效率相关
    TOOL = "tool"               # 工具问题
    GUIDELINE = "guideline"     # 规范问题
    COMMUNICATION = "communication"  # 沟通问题
    SUGGESTION = "suggestion"   # 建议
    PRAISE = "praise"           # 表扬
    COMPLAINT = "complaint"     # 投诉
    OTHER = "other"             # 其他


class SentimentType(str, Enum):
    """Sentiment types."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class FeedbackStatus(str, Enum):
    """Feedback status."""
    NEW = "new"
    REVIEWED = "reviewed"
    ACTIONED = "actioned"
    CLOSED = "closed"


@dataclass
class Feedback:
    """Feedback record."""
    id: UUID
    source: FeedbackSource
    category: FeedbackCategory
    content: str
    sentiment: SentimentType = SentimentType.NEUTRAL
    sentiment_score: float = 0.0  # -1 to 1
    status: FeedbackStatus = FeedbackStatus.NEW
    submitter_id: Optional[str] = None
    submitter_type: Optional[str] = None
    tenant_id: Optional[str] = None
    related_task_id: Optional[UUID] = None
    related_user_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    reviewed_at: Optional[datetime] = None
    reviewed_by: Optional[str] = None
    action_taken: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "source": self.source.value,
            "category": self.category.value,
            "content": self.content,
            "sentiment": self.sentiment.value,
            "sentiment_score": self.sentiment_score,
            "status": self.status.value,
            "submitter_id": self.submitter_id,
            "submitter_type": self.submitter_type,
            "tenant_id": self.tenant_id,
            "related_task_id": str(self.related_task_id) if self.related_task_id else None,
            "related_user_id": self.related_user_id,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "reviewed_by": self.reviewed_by,
            "action_taken": self.action_taken
        }


@dataclass
class SentimentResult:
    """Sentiment analysis result."""
    sentiment: SentimentType
    score: float  # -1 to 1
    confidence: float  # 0 to 1
    keywords: List[str]
    analysis_method: str


class FeedbackCollector:
    """
    Feedback collection and analysis engine.

    Collects feedback from various sources, performs sentiment analysis,
    and categorizes for action.
    """

    # Keyword-based sentiment indicators
    POSITIVE_KEYWORDS = [
        "好", "优秀", "满意", "感谢", "高效", "准确", "专业",
        "great", "excellent", "good", "thank", "helpful", "accurate"
    ]

    NEGATIVE_KEYWORDS = [
        "差", "错误", "问题", "慢", "不满", "投诉", "失望",
        "bad", "error", "wrong", "slow", "complaint", "disappointed"
    ]

    # Category detection patterns
    CATEGORY_PATTERNS = {
        FeedbackCategory.QUALITY: ["质量", "准确", "错误", "accuracy", "quality", "error"],
        FeedbackCategory.EFFICIENCY: ["速度", "效率", "慢", "快", "speed", "efficiency", "slow"],
        FeedbackCategory.TOOL: ["工具", "系统", "界面", "bug", "tool", "system", "interface"],
        FeedbackCategory.GUIDELINE: ["规范", "标准", "指南", "guideline", "standard", "rule"],
        FeedbackCategory.SUGGESTION: ["建议", "希望", "改进", "suggest", "improve", "recommend"],
        FeedbackCategory.PRAISE: ["表扬", "感谢", "满意", "praise", "thank", "satisfied"],
        FeedbackCategory.COMPLAINT: ["投诉", "不满", "抱怨", "complaint", "unsatisfied"]
    }

    def __init__(self):
        """Initialize the feedback collector."""
        self._feedbacks: Dict[UUID, Feedback] = {}

    async def collect_feedback(
        self,
        source: FeedbackSource,
        content: str,
        submitter_id: Optional[str] = None,
        submitter_type: Optional[str] = None,
        tenant_id: Optional[str] = None,
        related_task_id: Optional[UUID] = None,
        related_user_id: Optional[str] = None,
        category: Optional[FeedbackCategory] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Feedback:
        """
        Collect a feedback entry.

        Args:
            source: Feedback source
            content: Feedback content
            submitter_id: Who submitted
            submitter_type: Submitter type
            tenant_id: Tenant identifier
            related_task_id: Related task
            related_user_id: Related user
            category: Pre-set category
            tags: Tags
            metadata: Additional metadata

        Returns:
            Created feedback
        """
        # Analyze sentiment
        sentiment_result = await self.analyze_sentiment(content)

        # Auto-detect category if not provided
        if not category:
            category = self._detect_category(content)

        # Extract additional tags
        auto_tags = self._extract_tags(content)
        all_tags = list(set((tags or []) + auto_tags))

        feedback = Feedback(
            id=uuid4(),
            source=source,
            category=category,
            content=content,
            sentiment=sentiment_result.sentiment,
            sentiment_score=sentiment_result.score,
            submitter_id=submitter_id,
            submitter_type=submitter_type,
            tenant_id=tenant_id,
            related_task_id=related_task_id,
            related_user_id=related_user_id,
            tags=all_tags,
            metadata=metadata or {}
        )

        self._feedbacks[feedback.id] = feedback

        logger.info(
            f"Feedback collected: {feedback.id} - {source.value} - "
            f"{category.value} - {sentiment_result.sentiment.value}"
        )

        return feedback

    async def analyze_sentiment(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            Sentiment analysis result
        """
        text_lower = text.lower()

        # Count positive/negative indicators
        positive_count = sum(1 for kw in self.POSITIVE_KEYWORDS if kw in text_lower)
        negative_count = sum(1 for kw in self.NEGATIVE_KEYWORDS if kw in text_lower)

        # Calculate score
        total = positive_count + negative_count
        if total == 0:
            score = 0.0
            sentiment = SentimentType.NEUTRAL
            confidence = 0.5
        else:
            score = (positive_count - negative_count) / total
            if score > 0.2:
                sentiment = SentimentType.POSITIVE
            elif score < -0.2:
                sentiment = SentimentType.NEGATIVE
            else:
                sentiment = SentimentType.NEUTRAL
            confidence = min(0.9, 0.5 + total * 0.05)

        # Extract matched keywords
        keywords = []
        for kw in self.POSITIVE_KEYWORDS + self.NEGATIVE_KEYWORDS:
            if kw in text_lower:
                keywords.append(kw)

        return SentimentResult(
            sentiment=sentiment,
            score=score,
            confidence=confidence,
            keywords=keywords,
            analysis_method="keyword_based"
        )

    def _detect_category(self, text: str) -> FeedbackCategory:
        """Detect feedback category from content."""
        text_lower = text.lower()

        max_matches = 0
        detected_category = FeedbackCategory.OTHER

        for category, patterns in self.CATEGORY_PATTERNS.items():
            matches = sum(1 for p in patterns if p in text_lower)
            if matches > max_matches:
                max_matches = matches
                detected_category = category

        return detected_category

    def _extract_tags(self, text: str) -> List[str]:
        """Extract relevant tags from content."""
        tags = []
        text_lower = text.lower()

        # Priority indicators
        if any(kw in text_lower for kw in ["紧急", "urgent", "asap", "立即"]):
            tags.append("urgent")

        # Quality indicators
        if any(kw in text_lower for kw in ["质量", "quality", "准确", "accuracy"]):
            tags.append("quality_related")

        # Tool indicators
        if any(kw in text_lower for kw in ["系统", "工具", "system", "tool", "bug"]):
            tags.append("tool_related")

        return tags

    async def get_feedback(self, feedback_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get a feedback entry.

        Args:
            feedback_id: Feedback UUID

        Returns:
            Feedback data or None
        """
        feedback = self._feedbacks.get(feedback_id)
        if feedback:
            return feedback.to_dict()
        return None

    async def list_feedbacks(
        self,
        source: Optional[FeedbackSource] = None,
        category: Optional[FeedbackCategory] = None,
        sentiment: Optional[SentimentType] = None,
        status: Optional[FeedbackStatus] = None,
        tenant_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        List feedbacks with filters.

        Args:
            source: Filter by source
            category: Filter by category
            sentiment: Filter by sentiment
            status: Filter by status
            tenant_id: Filter by tenant
            limit: Max results
            offset: Pagination offset

        Returns:
            Tuple of (feedbacks, total_count)
        """
        feedbacks = list(self._feedbacks.values())

        if source:
            feedbacks = [f for f in feedbacks if f.source == source]
        if category:
            feedbacks = [f for f in feedbacks if f.category == category]
        if sentiment:
            feedbacks = [f for f in feedbacks if f.sentiment == sentiment]
        if status:
            feedbacks = [f for f in feedbacks if f.status == status]
        if tenant_id:
            feedbacks = [f for f in feedbacks if f.tenant_id == tenant_id]

        # Sort by created_at descending
        feedbacks.sort(key=lambda x: x.created_at, reverse=True)

        total = len(feedbacks)
        feedbacks = feedbacks[offset:offset + limit]

        return [f.to_dict() for f in feedbacks], total

    async def update_feedback_status(
        self,
        feedback_id: UUID,
        status: FeedbackStatus,
        reviewed_by: Optional[str] = None,
        action_taken: Optional[str] = None
    ) -> bool:
        """
        Update feedback status.

        Args:
            feedback_id: Feedback UUID
            status: New status
            reviewed_by: Reviewer
            action_taken: Action description

        Returns:
            True if updated
        """
        feedback = self._feedbacks.get(feedback_id)
        if not feedback:
            return False

        feedback.status = status
        if reviewed_by:
            feedback.reviewed_by = reviewed_by
            feedback.reviewed_at = datetime.now()
        if action_taken:
            feedback.action_taken = action_taken

        logger.info(f"Feedback status updated: {feedback_id} -> {status.value}")
        return True

    async def get_feedback_statistics(
        self,
        tenant_id: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get feedback statistics.

        Args:
            tenant_id: Filter by tenant
            days: Analysis period

        Returns:
            Statistics dictionary
        """
        cutoff = datetime.now() - timedelta(days=days)

        feedbacks = [
            f for f in self._feedbacks.values()
            if f.created_at >= cutoff
        ]

        if tenant_id:
            feedbacks = [f for f in feedbacks if f.tenant_id == tenant_id]

        total = len(feedbacks)

        # By source
        by_source = {}
        for source in FeedbackSource:
            by_source[source.value] = len([f for f in feedbacks if f.source == source])

        # By category
        by_category = {}
        for category in FeedbackCategory:
            by_category[category.value] = len([f for f in feedbacks if f.category == category])

        # By sentiment
        by_sentiment = {}
        for sentiment in SentimentType:
            by_sentiment[sentiment.value] = len([f for f in feedbacks if f.sentiment == sentiment])

        # By status
        by_status = {}
        for status in FeedbackStatus:
            by_status[status.value] = len([f for f in feedbacks if f.status == status])

        # Average sentiment score
        avg_sentiment = sum(f.sentiment_score for f in feedbacks) / total if total > 0 else 0

        # Response rate
        actioned = by_status.get("actioned", 0) + by_status.get("closed", 0)
        response_rate = actioned / total if total > 0 else 0

        return {
            "period_days": days,
            "total_feedbacks": total,
            "by_source": by_source,
            "by_category": by_category,
            "by_sentiment": by_sentiment,
            "by_status": by_status,
            "avg_sentiment_score": round(avg_sentiment, 4),
            "response_rate": round(response_rate, 4),
            "positive_rate": round(by_sentiment.get("positive", 0) / total, 4) if total > 0 else 0,
            "negative_rate": round(by_sentiment.get("negative", 0) / total, 4) if total > 0 else 0,
            "generated_at": datetime.now().isoformat()
        }

    async def get_common_issues(
        self,
        tenant_id: Optional[str] = None,
        days: int = 30,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get most common feedback issues.

        Args:
            tenant_id: Filter by tenant
            days: Analysis period
            limit: Max results

        Returns:
            List of common issues
        """
        cutoff = datetime.now() - timedelta(days=days)

        feedbacks = [
            f for f in self._feedbacks.values()
            if f.created_at >= cutoff and f.sentiment == SentimentType.NEGATIVE
        ]

        if tenant_id:
            feedbacks = [f for f in feedbacks if f.tenant_id == tenant_id]

        # Group by category
        category_counts = {}
        for feedback in feedbacks:
            key = feedback.category.value
            if key not in category_counts:
                category_counts[key] = {
                    "count": 0,
                    "examples": []
                }
            category_counts[key]["count"] += 1
            if len(category_counts[key]["examples"]) < 3:
                category_counts[key]["examples"].append(feedback.content[:100])

        # Sort by count
        issues = [
            {
                "category": category,
                "count": data["count"],
                "examples": data["examples"]
            }
            for category, data in category_counts.items()
        ]
        issues.sort(key=lambda x: x["count"], reverse=True)

        return issues[:limit]
