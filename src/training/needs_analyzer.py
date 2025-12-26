"""
Training needs analysis for SuperInsight Platform.

Provides:
- Skill gap identification
- Training recommendations
- Learning path generation
- Progress tracking
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class SkillCategory(str, Enum):
    """Skill categories."""
    ANNOTATION = "annotation"
    QUALITY = "quality"
    EFFICIENCY = "efficiency"
    COMPLIANCE = "compliance"
    DOMAIN = "domain"
    TOOL = "tool"


class TrainingPriority(str, Enum):
    """Training priority levels."""
    CRITICAL = "critical"  # Immediate attention
    HIGH = "high"          # Within 1 week
    MEDIUM = "medium"      # Within 1 month
    LOW = "low"            # Optional enhancement


class TrainingStatus(str, Enum):
    """Training status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    EXPIRED = "expired"


@dataclass
class SkillGap:
    """Identified skill gap."""
    skill_name: str
    category: SkillCategory
    current_level: float  # 0-1
    required_level: float  # 0-1
    gap: float  # required - current
    priority: TrainingPriority
    evidence: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TrainingRecommendation:
    """Training recommendation."""
    id: UUID
    skill_name: str
    category: SkillCategory
    title: str
    description: str
    priority: TrainingPriority
    estimated_duration_hours: float
    content_type: str  # video, document, practice, quiz
    resources: List[Dict[str, Any]] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)


@dataclass
class LearningPath:
    """Personalized learning path."""
    id: UUID
    user_id: str
    title: str
    description: str
    skill_gaps: List[SkillGap]
    recommendations: List[TrainingRecommendation]
    estimated_total_hours: float
    created_at: datetime = field(default_factory=datetime.now)
    status: TrainingStatus = TrainingStatus.NOT_STARTED
    progress: float = 0.0  # 0-1


class TrainingNeedsAnalyzer:
    """
    Training needs analysis engine.

    Analyzes performance data to identify skill gaps and
    recommend appropriate training.
    """

    # Skill requirements by role/level
    SKILL_REQUIREMENTS = {
        "annotator_basic": {
            "annotation_accuracy": 0.85,
            "annotation_speed": 0.70,
            "guideline_compliance": 0.90,
            "tool_proficiency": 0.80
        },
        "annotator_senior": {
            "annotation_accuracy": 0.95,
            "annotation_speed": 0.85,
            "guideline_compliance": 0.95,
            "tool_proficiency": 0.90,
            "quality_review": 0.85
        },
        "reviewer": {
            "annotation_accuracy": 0.98,
            "quality_review": 0.95,
            "feedback_quality": 0.90,
            "calibration": 0.90
        }
    }

    # Training content library
    TRAINING_LIBRARY = {
        "annotation_accuracy": [
            {
                "title": "标注准确性提升课程",
                "description": "学习如何提高标注准确率，减少错误",
                "content_type": "video",
                "duration_hours": 2.0,
                "resources": [
                    {"type": "video", "url": "/training/accuracy-basics"},
                    {"type": "quiz", "url": "/training/accuracy-quiz"}
                ]
            },
            {
                "title": "边界案例处理指南",
                "description": "学习如何处理复杂和边界情况",
                "content_type": "document",
                "duration_hours": 1.5,
                "resources": [
                    {"type": "document", "url": "/training/edge-cases"}
                ]
            }
        ],
        "annotation_speed": [
            {
                "title": "高效标注技巧",
                "description": "提升标注速度的实用技巧",
                "content_type": "video",
                "duration_hours": 1.5,
                "resources": [
                    {"type": "video", "url": "/training/speed-tips"}
                ]
            }
        ],
        "guideline_compliance": [
            {
                "title": "标注规范详解",
                "description": "深入理解标注规范和要求",
                "content_type": "document",
                "duration_hours": 2.0,
                "resources": [
                    {"type": "document", "url": "/training/guidelines"}
                ]
            }
        ],
        "tool_proficiency": [
            {
                "title": "标注工具使用教程",
                "description": "掌握标注工具的高级功能",
                "content_type": "practice",
                "duration_hours": 3.0,
                "resources": [
                    {"type": "video", "url": "/training/tool-tutorial"},
                    {"type": "practice", "url": "/training/tool-practice"}
                ]
            }
        ],
        "quality_review": [
            {
                "title": "质量审核标准培训",
                "description": "学习如何进行有效的质量审核",
                "content_type": "video",
                "duration_hours": 2.5,
                "resources": [
                    {"type": "video", "url": "/training/qa-standards"}
                ]
            }
        ]
    }

    def __init__(self):
        """Initialize the training needs analyzer."""
        self._learning_paths: Dict[str, LearningPath] = {}

    async def identify_skill_gaps(
        self,
        user_id: str,
        role: str = "annotator_basic",
        performance_data: Optional[Dict[str, float]] = None
    ) -> List[SkillGap]:
        """
        Identify skill gaps for a user.

        Args:
            user_id: User identifier
            role: User role for requirements
            performance_data: Current performance metrics

        Returns:
            List of identified skill gaps
        """
        gaps = []

        # Get requirements for role
        requirements = self.SKILL_REQUIREMENTS.get(role, self.SKILL_REQUIREMENTS["annotator_basic"])

        # If no performance data, fetch from evaluation system
        if not performance_data:
            performance_data = await self._fetch_performance_data(user_id)

        for skill_name, required_level in requirements.items():
            current_level = performance_data.get(skill_name, 0.5)
            gap = required_level - current_level

            if gap > 0.05:  # Only consider gaps > 5%
                # Determine priority based on gap size
                if gap >= 0.30:
                    priority = TrainingPriority.CRITICAL
                elif gap >= 0.20:
                    priority = TrainingPriority.HIGH
                elif gap >= 0.10:
                    priority = TrainingPriority.MEDIUM
                else:
                    priority = TrainingPriority.LOW

                skill_gap = SkillGap(
                    skill_name=skill_name,
                    category=self._get_skill_category(skill_name),
                    current_level=current_level,
                    required_level=required_level,
                    gap=gap,
                    priority=priority,
                    evidence=[
                        {
                            "metric": skill_name,
                            "current": current_level,
                            "required": required_level,
                            "measured_at": datetime.now().isoformat()
                        }
                    ]
                )
                gaps.append(skill_gap)

        # Sort by priority and gap size
        priority_order = {
            TrainingPriority.CRITICAL: 0,
            TrainingPriority.HIGH: 1,
            TrainingPriority.MEDIUM: 2,
            TrainingPriority.LOW: 3
        }
        gaps.sort(key=lambda x: (priority_order[x.priority], -x.gap))

        logger.info(f"Identified {len(gaps)} skill gaps for user {user_id}")
        return gaps

    async def _fetch_performance_data(self, user_id: str) -> Dict[str, float]:
        """Fetch performance data from evaluation system."""
        try:
            from src.evaluation.performance import PerformanceEngine

            engine = PerformanceEngine()
            history = await engine.get_user_performance_history(user_id)

            if history:
                latest = history[-1]
                return {
                    "annotation_accuracy": latest.get("accuracy_rate", 0.5),
                    "annotation_speed": latest.get("completion_rate", 0.5),
                    "guideline_compliance": latest.get("sla_compliance_rate", 0.5),
                    "tool_proficiency": 0.7,  # Default, would need tool-specific metrics
                    "quality_review": latest.get("quality_score", 0.5)
                }
        except ImportError:
            pass

        # Return defaults if no data
        return {
            "annotation_accuracy": 0.7,
            "annotation_speed": 0.6,
            "guideline_compliance": 0.8,
            "tool_proficiency": 0.7
        }

    def _get_skill_category(self, skill_name: str) -> SkillCategory:
        """Map skill name to category."""
        category_map = {
            "annotation_accuracy": SkillCategory.QUALITY,
            "annotation_speed": SkillCategory.EFFICIENCY,
            "guideline_compliance": SkillCategory.COMPLIANCE,
            "tool_proficiency": SkillCategory.TOOL,
            "quality_review": SkillCategory.QUALITY,
            "feedback_quality": SkillCategory.QUALITY,
            "calibration": SkillCategory.QUALITY
        }
        return category_map.get(skill_name, SkillCategory.ANNOTATION)

    async def recommend_training(
        self,
        user_id: str,
        skill_gaps: Optional[List[SkillGap]] = None,
        max_recommendations: int = 5
    ) -> List[TrainingRecommendation]:
        """
        Recommend training based on skill gaps.

        Args:
            user_id: User identifier
            skill_gaps: Pre-identified gaps (or will be analyzed)
            max_recommendations: Maximum recommendations

        Returns:
            List of training recommendations
        """
        if not skill_gaps:
            skill_gaps = await self.identify_skill_gaps(user_id)

        recommendations = []

        for gap in skill_gaps[:max_recommendations]:
            # Get training content for this skill
            content_options = self.TRAINING_LIBRARY.get(gap.skill_name, [])

            for content in content_options:
                recommendation = TrainingRecommendation(
                    id=uuid4(),
                    skill_name=gap.skill_name,
                    category=gap.category,
                    title=content["title"],
                    description=content["description"],
                    priority=gap.priority,
                    estimated_duration_hours=content["duration_hours"],
                    content_type=content["content_type"],
                    resources=content.get("resources", [])
                )
                recommendations.append(recommendation)

        logger.info(f"Generated {len(recommendations)} training recommendations for user {user_id}")
        return recommendations

    async def generate_learning_path(
        self,
        user_id: str,
        role: str = "annotator_basic"
    ) -> LearningPath:
        """
        Generate a personalized learning path.

        Args:
            user_id: User identifier
            role: Target role

        Returns:
            Learning path
        """
        # Identify gaps
        gaps = await self.identify_skill_gaps(user_id, role)

        # Get recommendations
        recommendations = await self.recommend_training(user_id, gaps)

        # Calculate total hours
        total_hours = sum(r.estimated_duration_hours for r in recommendations)

        # Create learning path
        path = LearningPath(
            id=uuid4(),
            user_id=user_id,
            title=f"个性化学习路径 - {role}",
            description=f"基于您的技能评估，为您定制的学习路径。共 {len(recommendations)} 个课程，预计 {total_hours:.1f} 小时。",
            skill_gaps=gaps,
            recommendations=recommendations,
            estimated_total_hours=total_hours
        )

        self._learning_paths[user_id] = path

        logger.info(f"Generated learning path for user {user_id}: {path.id}")
        return path

    async def get_learning_path(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user's current learning path.

        Args:
            user_id: User identifier

        Returns:
            Learning path data or None
        """
        path = self._learning_paths.get(user_id)
        if not path:
            return None

        return {
            "id": str(path.id),
            "user_id": path.user_id,
            "title": path.title,
            "description": path.description,
            "skill_gaps": [
                {
                    "skill_name": g.skill_name,
                    "category": g.category.value,
                    "current_level": g.current_level,
                    "required_level": g.required_level,
                    "gap": g.gap,
                    "priority": g.priority.value
                }
                for g in path.skill_gaps
            ],
            "recommendations": [
                {
                    "id": str(r.id),
                    "skill_name": r.skill_name,
                    "title": r.title,
                    "description": r.description,
                    "priority": r.priority.value,
                    "duration_hours": r.estimated_duration_hours,
                    "content_type": r.content_type,
                    "resources": r.resources
                }
                for r in path.recommendations
            ],
            "estimated_total_hours": path.estimated_total_hours,
            "status": path.status.value,
            "progress": path.progress,
            "created_at": path.created_at.isoformat()
        }

    async def update_progress(
        self,
        user_id: str,
        recommendation_id: UUID,
        completed: bool = True
    ) -> bool:
        """
        Update training progress.

        Args:
            user_id: User identifier
            recommendation_id: Completed recommendation
            completed: Whether completed

        Returns:
            True if updated
        """
        path = self._learning_paths.get(user_id)
        if not path:
            return False

        # Find and mark recommendation
        total = len(path.recommendations)
        completed_count = 0

        for rec in path.recommendations:
            if rec.id == recommendation_id:
                # Mark as completed (in production, would store status per recommendation)
                completed_count += 1

        # Update progress
        path.progress = completed_count / total if total > 0 else 0
        path.status = TrainingStatus.IN_PROGRESS if 0 < path.progress < 1 else (
            TrainingStatus.COMPLETED if path.progress >= 1 else TrainingStatus.NOT_STARTED
        )

        return True

    async def get_training_statistics(
        self,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get training statistics.

        Args:
            tenant_id: Optional tenant filter

        Returns:
            Statistics dictionary
        """
        paths = list(self._learning_paths.values())

        total_paths = len(paths)
        completed = len([p for p in paths if p.status == TrainingStatus.COMPLETED])
        in_progress = len([p for p in paths if p.status == TrainingStatus.IN_PROGRESS])

        # Average progress
        avg_progress = sum(p.progress for p in paths) / total_paths if total_paths > 0 else 0

        # Skill gap distribution
        all_gaps = []
        for path in paths:
            all_gaps.extend(path.skill_gaps)

        gap_by_skill = {}
        for gap in all_gaps:
            if gap.skill_name not in gap_by_skill:
                gap_by_skill[gap.skill_name] = []
            gap_by_skill[gap.skill_name].append(gap.gap)

        common_gaps = {
            skill: sum(gaps) / len(gaps)
            for skill, gaps in gap_by_skill.items()
        }

        return {
            "total_learning_paths": total_paths,
            "completed": completed,
            "in_progress": in_progress,
            "not_started": total_paths - completed - in_progress,
            "avg_progress": round(avg_progress, 4),
            "common_skill_gaps": common_gaps,
            "generated_at": datetime.now().isoformat()
        }
