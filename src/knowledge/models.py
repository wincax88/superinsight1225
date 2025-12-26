"""
Data models for Knowledge Management module.

Defines knowledge entries, rules, cases and related structures.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator


class KnowledgeCategory(str, Enum):
    """Knowledge categories."""
    DOMAIN = "domain"           # Domain-specific knowledge
    RULE = "rule"               # Business rules
    PATTERN = "pattern"         # Data patterns
    MAPPING = "mapping"         # Term/entity mappings
    TEMPLATE = "template"       # Query/response templates
    FAQ = "faq"                 # Frequently asked questions
    GLOSSARY = "glossary"       # Term definitions


class RuleType(str, Enum):
    """Rule types."""
    VALIDATION = "validation"   # Data validation rules
    TRANSFORMATION = "transformation"  # Data transformation rules
    INFERENCE = "inference"     # Inference rules
    CONSTRAINT = "constraint"   # Constraint rules
    ASSOCIATION = "association" # Association rules
    CLASSIFICATION = "classification"  # Classification rules


class CaseStatus(str, Enum):
    """Case status."""
    ACTIVE = "active"           # Active case
    RESOLVED = "resolved"       # Resolved case
    ARCHIVED = "archived"       # Archived case
    PENDING = "pending"         # Pending review


class QualityScore(BaseModel):
    """Quality score for knowledge entries."""

    accuracy: float = Field(default=0.0, ge=0.0, le=1.0, description="Accuracy score")
    completeness: float = Field(default=0.0, ge=0.0, le=1.0, description="Completeness score")
    consistency: float = Field(default=0.0, ge=0.0, le=1.0, description="Consistency score")
    relevance: float = Field(default=0.0, ge=0.0, le=1.0, description="Relevance score")
    timeliness: float = Field(default=0.0, ge=0.0, le=1.0, description="Timeliness score")

    @property
    def overall(self) -> float:
        """Calculate overall quality score."""
        weights = {
            "accuracy": 0.3,
            "completeness": 0.2,
            "consistency": 0.2,
            "relevance": 0.2,
            "timeliness": 0.1
        }
        return (
            self.accuracy * weights["accuracy"] +
            self.completeness * weights["completeness"] +
            self.consistency * weights["consistency"] +
            self.relevance * weights["relevance"] +
            self.timeliness * weights["timeliness"]
        )


class KnowledgeEntry(BaseModel):
    """Knowledge entry model."""

    id: UUID = Field(default_factory=uuid4, description="Entry ID")
    title: str = Field(..., min_length=1, description="Entry title")
    content: str = Field(..., description="Entry content")
    category: KnowledgeCategory = Field(..., description="Entry category")
    tags: List[str] = Field(default_factory=list, description="Tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    source: Optional[str] = Field(None, description="Knowledge source")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    quality: QualityScore = Field(default_factory=QualityScore, description="Quality score")
    version: int = Field(default=1, ge=1, description="Version number")
    is_active: bool = Field(default=True, description="Whether entry is active")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Update time")
    created_by: Optional[str] = Field(None, description="Creator ID")
    updated_by: Optional[str] = Field(None, description="Updater ID")

    def update(self, **kwargs) -> None:
        """Update entry fields."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.now()
        self.version += 1


class KnowledgeRule(BaseModel):
    """Knowledge rule model."""

    id: UUID = Field(default_factory=uuid4, description="Rule ID")
    name: str = Field(..., min_length=1, description="Rule name")
    description: str = Field(default="", description="Rule description")
    rule_type: RuleType = Field(..., description="Rule type")
    condition: str = Field(..., description="Rule condition expression")
    action: str = Field(..., description="Rule action expression")
    priority: int = Field(default=0, description="Rule priority (higher = more important)")
    is_enabled: bool = Field(default=True, description="Whether rule is enabled")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Rule confidence")
    support: float = Field(default=0.0, ge=0.0, le=1.0, description="Rule support (from learning)")
    hit_count: int = Field(default=0, ge=0, description="Number of times rule was triggered")
    success_count: int = Field(default=0, ge=0, description="Number of successful applications")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    source_cases: List[UUID] = Field(default_factory=list, description="Cases this rule was learned from")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Update time")

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.hit_count == 0:
            return 0.0
        return self.success_count / self.hit_count

    def apply(self) -> None:
        """Record rule application."""
        self.hit_count += 1
        self.updated_at = datetime.now()

    def record_success(self) -> None:
        """Record successful rule application."""
        self.success_count += 1
        self.updated_at = datetime.now()


class CaseEntry(BaseModel):
    """Case entry model."""

    id: UUID = Field(default_factory=uuid4, description="Case ID")
    title: str = Field(..., min_length=1, description="Case title")
    description: str = Field(default="", description="Case description")
    problem: str = Field(..., description="Problem description")
    solution: str = Field(default="", description="Solution description")
    context: Dict[str, Any] = Field(default_factory=dict, description="Case context")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data")
    output_data: Dict[str, Any] = Field(default_factory=dict, description="Output data")
    status: CaseStatus = Field(default=CaseStatus.PENDING, description="Case status")
    outcome: Optional[str] = Field(None, description="Case outcome")
    feedback: Optional[str] = Field(None, description="User feedback")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Case rating (1-5)")
    tags: List[str] = Field(default_factory=list, description="Case tags")
    related_cases: List[UUID] = Field(default_factory=list, description="Related case IDs")
    derived_rules: List[UUID] = Field(default_factory=list, description="Rules derived from this case")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="Update time")
    resolved_at: Optional[datetime] = Field(None, description="Resolution time")

    def resolve(self, solution: str, outcome: str) -> None:
        """Mark case as resolved."""
        self.solution = solution
        self.outcome = outcome
        self.status = CaseStatus.RESOLVED
        self.resolved_at = datetime.now()
        self.updated_at = datetime.now()

    def archive(self) -> None:
        """Archive the case."""
        self.status = CaseStatus.ARCHIVED
        self.updated_at = datetime.now()


class KnowledgeUpdateResult(BaseModel):
    """Result of knowledge update operation."""

    success: bool = Field(..., description="Whether update succeeded")
    message: str = Field(default="", description="Result message")
    entries_added: int = Field(default=0, ge=0, description="Entries added")
    entries_updated: int = Field(default=0, ge=0, description="Entries updated")
    entries_removed: int = Field(default=0, ge=0, description="Entries removed")
    rules_learned: int = Field(default=0, ge=0, description="Rules learned")
    cases_processed: int = Field(default=0, ge=0, description="Cases processed")
    quality_scores: Dict[str, float] = Field(default_factory=dict, description="Quality scores")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    processing_time: float = Field(default=0.0, description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Update timestamp")


class KnowledgeSearchResult(BaseModel):
    """Search result for knowledge queries."""

    entries: List[KnowledgeEntry] = Field(default_factory=list, description="Matching entries")
    rules: List[KnowledgeRule] = Field(default_factory=list, description="Matching rules")
    cases: List[CaseEntry] = Field(default_factory=list, description="Matching cases")
    total_count: int = Field(default=0, description="Total matching count")
    query: str = Field(default="", description="Original query")
    search_time: float = Field(default=0.0, description="Search time in seconds")


class FeedbackEntry(BaseModel):
    """Feedback entry for knowledge improvement."""

    id: UUID = Field(default_factory=uuid4, description="Feedback ID")
    target_id: UUID = Field(..., description="Target entry/rule/case ID")
    target_type: str = Field(..., description="Target type (entry/rule/case)")
    feedback_type: str = Field(..., description="Feedback type (correct/incorrect/improve)")
    content: str = Field(default="", description="Feedback content")
    suggested_change: Optional[str] = Field(None, description="Suggested change")
    user_id: Optional[str] = Field(None, description="User ID")
    is_processed: bool = Field(default=False, description="Whether feedback is processed")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation time")
