"""
Training content recommendation engine for SuperInsight Platform.

Provides:
- Intelligent training content matching
- Personalized learning path optimization
- Training resource library management
- Training effectiveness prediction
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID, uuid4
from enum import Enum
from dataclasses import dataclass, field
import math

from .needs_analyzer import (
    SkillGap,
    TrainingRecommendation,
    SkillCategory,
    TrainingPriority,
    TrainingStatus
)

logger = logging.getLogger(__name__)


class ContentType(str, Enum):
    """Training content types."""
    VIDEO = "video"
    DOCUMENT = "document"
    INTERACTIVE = "interactive"
    QUIZ = "quiz"
    PRACTICE = "practice"
    WORKSHOP = "workshop"
    MENTORING = "mentoring"


class DifficultyLevel(str, Enum):
    """Content difficulty levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class LearningStyle(str, Enum):
    """Learning style preferences."""
    VISUAL = "visual"
    READING = "reading"
    KINESTHETIC = "kinesthetic"
    MIXED = "mixed"


@dataclass
class TrainingContent:
    """Training content item."""
    id: UUID
    title: str
    description: str
    skill_name: str
    category: SkillCategory
    content_type: ContentType
    difficulty: DifficultyLevel
    duration_minutes: int
    resources: List[Dict[str, Any]]
    prerequisites: List[str] = field(default_factory=list)
    learning_objectives: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    effectiveness_score: float = 0.0  # 0-1 based on historical data
    completion_rate: float = 0.0  # Historical completion rate
    avg_rating: float = 0.0  # 0-5 rating
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ContentMatch:
    """Matched content with relevance score."""
    content: TrainingContent
    relevance_score: float  # 0-1
    skill_coverage: float  # 0-1, how much of the skill gap it covers
    predicted_effectiveness: float  # 0-1
    reasons: List[str] = field(default_factory=list)


@dataclass
class LearningPathOptimization:
    """Optimized learning path."""
    original_path: List[TrainingContent]
    optimized_path: List[TrainingContent]
    estimated_time_saved: int  # minutes
    coverage_improvement: float
    optimization_reasons: List[str]


@dataclass
class UserLearningProfile:
    """User's learning profile."""
    user_id: str
    preferred_style: LearningStyle
    preferred_content_types: List[ContentType]
    available_hours_per_week: float
    skill_levels: Dict[str, float]  # skill_name -> level
    completed_trainings: List[UUID]
    training_history: List[Dict[str, Any]]
    learning_pace: float = 1.0  # Multiplier for estimated duration
    last_updated: datetime = field(default_factory=datetime.now)


class TrainingContentRecommender:
    """
    Training content recommendation engine.

    Matches training content to user needs based on skill gaps,
    learning preferences, and historical effectiveness data.
    """

    def __init__(self):
        """Initialize the content recommender."""
        self._content_library: Dict[UUID, TrainingContent] = {}
        self._user_profiles: Dict[str, UserLearningProfile] = {}
        self._initialize_content_library()

    def _initialize_content_library(self):
        """Initialize default training content library."""
        default_contents = [
            # Annotation accuracy content
            {
                "title": "标注准确性基础课程",
                "description": "学习标注的基本原则和技巧，提高准确率",
                "skill_name": "annotation_accuracy",
                "category": SkillCategory.QUALITY,
                "content_type": ContentType.VIDEO,
                "difficulty": DifficultyLevel.BEGINNER,
                "duration_minutes": 60,
                "resources": [
                    {"type": "video", "url": "/training/accuracy-basics"},
                    {"type": "slides", "url": "/training/accuracy-slides"}
                ],
                "learning_objectives": ["理解标注准确性的定义", "掌握常见错误类型", "学会自检方法"],
                "effectiveness_score": 0.85,
                "avg_rating": 4.5
            },
            {
                "title": "高级标注技巧进阶",
                "description": "处理复杂标注场景，边界案例和模糊情况",
                "skill_name": "annotation_accuracy",
                "category": SkillCategory.QUALITY,
                "content_type": ContentType.INTERACTIVE,
                "difficulty": DifficultyLevel.ADVANCED,
                "duration_minutes": 120,
                "resources": [
                    {"type": "interactive", "url": "/training/advanced-annotation"},
                    {"type": "practice", "url": "/training/edge-cases-practice"}
                ],
                "prerequisites": ["标注准确性基础课程"],
                "learning_objectives": ["处理边界案例", "解决模糊标注问题", "提高复杂场景准确率"],
                "effectiveness_score": 0.90,
                "avg_rating": 4.7
            },
            # Annotation speed content
            {
                "title": "高效标注工作流程",
                "description": "优化标注流程，提升工作效率",
                "skill_name": "annotation_speed",
                "category": SkillCategory.EFFICIENCY,
                "content_type": ContentType.VIDEO,
                "difficulty": DifficultyLevel.INTERMEDIATE,
                "duration_minutes": 45,
                "resources": [
                    {"type": "video", "url": "/training/efficient-workflow"},
                    {"type": "checklist", "url": "/training/speed-checklist"}
                ],
                "learning_objectives": ["优化工作流程", "减少无效操作", "批量处理技巧"],
                "effectiveness_score": 0.82,
                "avg_rating": 4.3
            },
            {
                "title": "快捷键和自动化技巧",
                "description": "掌握快捷键和自动化工具提升速度",
                "skill_name": "annotation_speed",
                "category": SkillCategory.EFFICIENCY,
                "content_type": ContentType.PRACTICE,
                "difficulty": DifficultyLevel.BEGINNER,
                "duration_minutes": 30,
                "resources": [
                    {"type": "practice", "url": "/training/shortcuts-practice"},
                    {"type": "cheatsheet", "url": "/training/shortcuts-ref"}
                ],
                "learning_objectives": ["掌握常用快捷键", "配置个性化快捷方式", "使用自动化脚本"],
                "effectiveness_score": 0.78,
                "avg_rating": 4.6
            },
            # Guideline compliance content
            {
                "title": "标注规范详解与实践",
                "description": "深入理解并应用标注规范",
                "skill_name": "guideline_compliance",
                "category": SkillCategory.COMPLIANCE,
                "content_type": ContentType.DOCUMENT,
                "difficulty": DifficultyLevel.BEGINNER,
                "duration_minutes": 90,
                "resources": [
                    {"type": "document", "url": "/training/guidelines-doc"},
                    {"type": "examples", "url": "/training/guidelines-examples"}
                ],
                "learning_objectives": ["理解标注规范", "正确应用规范", "识别规范违规"],
                "effectiveness_score": 0.88,
                "avg_rating": 4.4
            },
            {
                "title": "规范更新与变更适应",
                "description": "快速适应规范变更和更新",
                "skill_name": "guideline_compliance",
                "category": SkillCategory.COMPLIANCE,
                "content_type": ContentType.WORKSHOP,
                "difficulty": DifficultyLevel.INTERMEDIATE,
                "duration_minutes": 60,
                "resources": [
                    {"type": "workshop", "url": "/training/guideline-changes"},
                    {"type": "quiz", "url": "/training/guideline-quiz"}
                ],
                "learning_objectives": ["跟踪规范变更", "快速适应新规范", "处理规范冲突"],
                "effectiveness_score": 0.85,
                "avg_rating": 4.2
            },
            # Tool proficiency content
            {
                "title": "标注工具完全指南",
                "description": "全面掌握标注工具的所有功能",
                "skill_name": "tool_proficiency",
                "category": SkillCategory.TOOL,
                "content_type": ContentType.INTERACTIVE,
                "difficulty": DifficultyLevel.BEGINNER,
                "duration_minutes": 120,
                "resources": [
                    {"type": "interactive", "url": "/training/tool-guide"},
                    {"type": "sandbox", "url": "/training/tool-sandbox"}
                ],
                "learning_objectives": ["掌握基本功能", "使用高级特性", "解决常见问题"],
                "effectiveness_score": 0.92,
                "avg_rating": 4.8
            },
            # Quality review content
            {
                "title": "质量审核标准与方法",
                "description": "学习专业的质量审核技能",
                "skill_name": "quality_review",
                "category": SkillCategory.QUALITY,
                "content_type": ContentType.VIDEO,
                "difficulty": DifficultyLevel.ADVANCED,
                "duration_minutes": 90,
                "resources": [
                    {"type": "video", "url": "/training/qa-methods"},
                    {"type": "template", "url": "/training/qa-templates"}
                ],
                "prerequisites": ["标注准确性基础课程", "高级标注技巧进阶"],
                "learning_objectives": ["审核标准理解", "问题识别技巧", "反馈撰写方法"],
                "effectiveness_score": 0.87,
                "avg_rating": 4.5
            },
            {
                "title": "审核一致性校准",
                "description": "确保审核标准的一致性",
                "skill_name": "calibration",
                "category": SkillCategory.QUALITY,
                "content_type": ContentType.WORKSHOP,
                "difficulty": DifficultyLevel.EXPERT,
                "duration_minutes": 150,
                "resources": [
                    {"type": "workshop", "url": "/training/calibration-workshop"},
                    {"type": "cases", "url": "/training/calibration-cases"}
                ],
                "learning_objectives": ["校准方法论", "一致性维护", "偏差检测与纠正"],
                "effectiveness_score": 0.89,
                "avg_rating": 4.6
            }
        ]

        for content_data in default_contents:
            content = TrainingContent(
                id=uuid4(),
                title=content_data["title"],
                description=content_data["description"],
                skill_name=content_data["skill_name"],
                category=content_data["category"],
                content_type=content_data["content_type"],
                difficulty=content_data["difficulty"],
                duration_minutes=content_data["duration_minutes"],
                resources=content_data["resources"],
                prerequisites=content_data.get("prerequisites", []),
                learning_objectives=content_data.get("learning_objectives", []),
                effectiveness_score=content_data.get("effectiveness_score", 0.0),
                avg_rating=content_data.get("avg_rating", 0.0)
            )
            self._content_library[content.id] = content

    async def get_user_profile(self, user_id: str) -> UserLearningProfile:
        """
        Get or create user learning profile.

        Args:
            user_id: User identifier

        Returns:
            User's learning profile
        """
        if user_id not in self._user_profiles:
            # Create default profile
            self._user_profiles[user_id] = UserLearningProfile(
                user_id=user_id,
                preferred_style=LearningStyle.MIXED,
                preferred_content_types=[ContentType.VIDEO, ContentType.PRACTICE],
                available_hours_per_week=5.0,
                skill_levels={},
                completed_trainings=[],
                training_history=[]
            )
        return self._user_profiles[user_id]

    async def update_user_profile(
        self,
        user_id: str,
        updates: Dict[str, Any]
    ) -> UserLearningProfile:
        """
        Update user learning profile.

        Args:
            user_id: User identifier
            updates: Profile updates

        Returns:
            Updated profile
        """
        profile = await self.get_user_profile(user_id)

        if "preferred_style" in updates:
            profile.preferred_style = LearningStyle(updates["preferred_style"])
        if "preferred_content_types" in updates:
            profile.preferred_content_types = [
                ContentType(ct) for ct in updates["preferred_content_types"]
            ]
        if "available_hours_per_week" in updates:
            profile.available_hours_per_week = updates["available_hours_per_week"]
        if "skill_levels" in updates:
            profile.skill_levels.update(updates["skill_levels"])
        if "learning_pace" in updates:
            profile.learning_pace = updates["learning_pace"]

        profile.last_updated = datetime.now()
        return profile

    async def recommend_content(
        self,
        user_id: str,
        skill_gaps: List[SkillGap],
        max_results: int = 10,
        include_prerequisites: bool = True
    ) -> List[ContentMatch]:
        """
        Recommend training content based on skill gaps.

        Args:
            user_id: User identifier
            skill_gaps: Identified skill gaps
            max_results: Maximum number of recommendations
            include_prerequisites: Include prerequisite courses

        Returns:
            List of matched content with relevance scores
        """
        profile = await self.get_user_profile(user_id)
        matches: List[ContentMatch] = []

        for gap in skill_gaps:
            # Find content for this skill
            matching_content = [
                c for c in self._content_library.values()
                if c.skill_name == gap.skill_name
            ]

            for content in matching_content:
                # Skip already completed
                if content.id in profile.completed_trainings:
                    continue

                # Calculate relevance score
                relevance = self._calculate_relevance(content, gap, profile)

                # Calculate skill coverage
                coverage = self._calculate_skill_coverage(content, gap)

                # Predict effectiveness for this user
                effectiveness = self._predict_effectiveness(content, profile)

                # Generate reasons
                reasons = self._generate_match_reasons(content, gap, profile)

                match = ContentMatch(
                    content=content,
                    relevance_score=relevance,
                    skill_coverage=coverage,
                    predicted_effectiveness=effectiveness,
                    reasons=reasons
                )
                matches.append(match)

        # Sort by combined score
        matches.sort(
            key=lambda m: (m.relevance_score * 0.4 + m.skill_coverage * 0.3 + m.predicted_effectiveness * 0.3),
            reverse=True
        )

        # Add prerequisites if needed
        if include_prerequisites:
            matches = await self._add_prerequisites(matches, profile)

        logger.info(f"Recommended {len(matches[:max_results])} content items for user {user_id}")
        return matches[:max_results]

    def _calculate_relevance(
        self,
        content: TrainingContent,
        gap: SkillGap,
        profile: UserLearningProfile
    ) -> float:
        """Calculate content relevance score."""
        score = 0.0

        # Skill match (base score)
        if content.skill_name == gap.skill_name:
            score += 0.4

        # Difficulty appropriateness
        difficulty_match = self._match_difficulty(content.difficulty, gap)
        score += difficulty_match * 0.2

        # Content type preference
        if content.content_type in profile.preferred_content_types:
            score += 0.2

        # Learning style alignment
        style_match = self._match_learning_style(content.content_type, profile.preferred_style)
        score += style_match * 0.1

        # Historical effectiveness
        score += content.effectiveness_score * 0.1

        return min(score, 1.0)

    def _match_difficulty(self, difficulty: DifficultyLevel, gap: SkillGap) -> float:
        """Match content difficulty to skill gap."""
        difficulty_map = {
            DifficultyLevel.BEGINNER: 0.3,
            DifficultyLevel.INTERMEDIATE: 0.6,
            DifficultyLevel.ADVANCED: 0.8,
            DifficultyLevel.EXPERT: 0.95
        }

        target_level = 1 - gap.current_level  # Inverse of current level
        content_level = difficulty_map[difficulty]

        # Best match when difficulty aligns with gap
        difference = abs(target_level - content_level)
        return 1 - difference

    def _match_learning_style(
        self,
        content_type: ContentType,
        style: LearningStyle
    ) -> float:
        """Match content type to learning style."""
        style_content_map = {
            LearningStyle.VISUAL: [ContentType.VIDEO, ContentType.INTERACTIVE],
            LearningStyle.READING: [ContentType.DOCUMENT],
            LearningStyle.KINESTHETIC: [ContentType.PRACTICE, ContentType.WORKSHOP],
            LearningStyle.MIXED: list(ContentType)
        }

        preferred = style_content_map.get(style, [])
        return 1.0 if content_type in preferred else 0.5

    def _calculate_skill_coverage(
        self,
        content: TrainingContent,
        gap: SkillGap
    ) -> float:
        """Calculate how much of the skill gap the content covers."""
        # Estimate based on difficulty and duration
        base_coverage = {
            DifficultyLevel.BEGINNER: 0.15,
            DifficultyLevel.INTERMEDIATE: 0.25,
            DifficultyLevel.ADVANCED: 0.35,
            DifficultyLevel.EXPERT: 0.25  # Expert is specialized, less broad coverage
        }

        coverage = base_coverage.get(content.difficulty, 0.2)

        # Adjust by duration (longer = more coverage, with diminishing returns)
        duration_factor = min(content.duration_minutes / 60, 2.0)  # Cap at 2 hours
        coverage *= (0.8 + 0.2 * duration_factor)

        # Adjust by effectiveness score
        coverage *= content.effectiveness_score

        return min(coverage / gap.gap, 1.0)

    def _predict_effectiveness(
        self,
        content: TrainingContent,
        profile: UserLearningProfile
    ) -> float:
        """Predict content effectiveness for specific user."""
        base_effectiveness = content.effectiveness_score

        # Adjust by completion rate history
        if profile.training_history:
            completed = [t for t in profile.training_history if t.get("completed", False)]
            completion_rate = len(completed) / len(profile.training_history)
            base_effectiveness *= (0.7 + 0.3 * completion_rate)

        # Adjust by content type preference
        if content.content_type in profile.preferred_content_types:
            base_effectiveness *= 1.1

        # Adjust by learning pace
        if profile.learning_pace > 1.0:
            # Fast learners may get less from beginner content
            if content.difficulty == DifficultyLevel.BEGINNER:
                base_effectiveness *= 0.9
        elif profile.learning_pace < 1.0:
            # Slower learners may struggle with advanced content
            if content.difficulty == DifficultyLevel.ADVANCED:
                base_effectiveness *= 0.9

        return min(base_effectiveness, 1.0)

    def _generate_match_reasons(
        self,
        content: TrainingContent,
        gap: SkillGap,
        profile: UserLearningProfile
    ) -> List[str]:
        """Generate human-readable match reasons."""
        reasons = []

        if content.skill_name == gap.skill_name:
            reasons.append(f"直接针对 {gap.skill_name} 技能差距")

        if content.content_type in profile.preferred_content_types:
            reasons.append(f"符合您偏好的 {content.content_type.value} 学习方式")

        if content.effectiveness_score >= 0.85:
            reasons.append(f"历史有效性评分高: {content.effectiveness_score:.0%}")

        if content.avg_rating >= 4.5:
            reasons.append(f"学员评分: {content.avg_rating}/5")

        if gap.priority in [TrainingPriority.CRITICAL, TrainingPriority.HIGH]:
            reasons.append("解决高优先级技能差距")

        return reasons

    async def _add_prerequisites(
        self,
        matches: List[ContentMatch],
        profile: UserLearningProfile
    ) -> List[ContentMatch]:
        """Add prerequisite courses to recommendations."""
        all_matches = list(matches)
        seen_titles = {m.content.title for m in matches}

        for match in matches:
            for prereq_title in match.content.prerequisites:
                if prereq_title in seen_titles:
                    continue

                # Find prerequisite content
                prereq_content = next(
                    (c for c in self._content_library.values() if c.title == prereq_title),
                    None
                )

                if prereq_content and prereq_content.id not in profile.completed_trainings:
                    prereq_match = ContentMatch(
                        content=prereq_content,
                        relevance_score=0.8,  # Prerequisites are important
                        skill_coverage=0.2,
                        predicted_effectiveness=prereq_content.effectiveness_score,
                        reasons=["先修课程", f"为 '{match.content.title}' 的前置要求"]
                    )
                    all_matches.insert(0, prereq_match)
                    seen_titles.add(prereq_title)

        return all_matches

    async def optimize_learning_path(
        self,
        user_id: str,
        content_list: List[TrainingContent]
    ) -> LearningPathOptimization:
        """
        Optimize learning path order and composition.

        Args:
            user_id: User identifier
            content_list: Initial content list

        Returns:
            Optimized learning path
        """
        profile = await self.get_user_profile(user_id)

        # Sort by prerequisites and difficulty
        optimized = sorted(
            content_list,
            key=lambda c: (
                len([p for p in c.prerequisites if p not in [x.title for x in content_list[:content_list.index(c)]]]),
                list(DifficultyLevel).index(c.difficulty),
                -c.effectiveness_score
            )
        )

        # Calculate time saved (by removing redundant content)
        original_time = sum(c.duration_minutes for c in content_list)

        # Remove redundant content (same skill, lower effectiveness)
        skill_best = {}
        for content in optimized:
            if content.skill_name not in skill_best:
                skill_best[content.skill_name] = content
            elif content.effectiveness_score > skill_best[content.skill_name].effectiveness_score:
                skill_best[content.skill_name] = content

        # Keep only best per skill if significant overlap
        unique_optimized = list(skill_best.values())
        optimized_time = sum(c.duration_minutes for c in unique_optimized)

        optimization_reasons = []
        if len(unique_optimized) < len(content_list):
            optimization_reasons.append(f"移除了 {len(content_list) - len(unique_optimized)} 个重复课程")
        if original_time > optimized_time:
            optimization_reasons.append(f"减少学习时间 {original_time - optimized_time} 分钟")

        optimization_reasons.append("按难度递进排序")
        optimization_reasons.append("确保先修课程在前")

        return LearningPathOptimization(
            original_path=content_list,
            optimized_path=unique_optimized,
            estimated_time_saved=original_time - optimized_time,
            coverage_improvement=0.0,  # Would calculate based on gap coverage
            optimization_reasons=optimization_reasons
        )

    async def add_content(self, content_data: Dict[str, Any]) -> TrainingContent:
        """
        Add new content to the library.

        Args:
            content_data: Content definition

        Returns:
            Created content
        """
        content = TrainingContent(
            id=uuid4(),
            title=content_data["title"],
            description=content_data["description"],
            skill_name=content_data["skill_name"],
            category=SkillCategory(content_data["category"]),
            content_type=ContentType(content_data["content_type"]),
            difficulty=DifficultyLevel(content_data["difficulty"]),
            duration_minutes=content_data["duration_minutes"],
            resources=content_data.get("resources", []),
            prerequisites=content_data.get("prerequisites", []),
            learning_objectives=content_data.get("learning_objectives", []),
            tags=content_data.get("tags", []),
            effectiveness_score=content_data.get("effectiveness_score", 0.5)
        )

        self._content_library[content.id] = content
        logger.info(f"Added content: {content.title} ({content.id})")
        return content

    async def update_content_effectiveness(
        self,
        content_id: UUID,
        new_effectiveness: float,
        completion_rate: Optional[float] = None,
        avg_rating: Optional[float] = None
    ) -> bool:
        """
        Update content effectiveness metrics.

        Args:
            content_id: Content identifier
            new_effectiveness: New effectiveness score
            completion_rate: New completion rate
            avg_rating: New average rating

        Returns:
            True if updated
        """
        content = self._content_library.get(content_id)
        if not content:
            return False

        content.effectiveness_score = new_effectiveness
        if completion_rate is not None:
            content.completion_rate = completion_rate
        if avg_rating is not None:
            content.avg_rating = avg_rating
        content.updated_at = datetime.now()

        return True

    async def get_content_by_skill(
        self,
        skill_name: str,
        difficulty: Optional[DifficultyLevel] = None
    ) -> List[TrainingContent]:
        """
        Get content for a specific skill.

        Args:
            skill_name: Skill name to filter
            difficulty: Optional difficulty filter

        Returns:
            Matching content list
        """
        results = [
            c for c in self._content_library.values()
            if c.skill_name == skill_name
        ]

        if difficulty:
            results = [c for c in results if c.difficulty == difficulty]

        return sorted(results, key=lambda c: -c.effectiveness_score)

    async def get_library_statistics(self) -> Dict[str, Any]:
        """
        Get training library statistics.

        Returns:
            Library statistics
        """
        contents = list(self._content_library.values())

        by_category = {}
        by_difficulty = {}
        by_type = {}

        for content in contents:
            cat = content.category.value
            by_category[cat] = by_category.get(cat, 0) + 1

            diff = content.difficulty.value
            by_difficulty[diff] = by_difficulty.get(diff, 0) + 1

            ct = content.content_type.value
            by_type[ct] = by_type.get(ct, 0) + 1

        total_duration = sum(c.duration_minutes for c in contents)
        avg_effectiveness = sum(c.effectiveness_score for c in contents) / len(contents) if contents else 0
        avg_rating = sum(c.avg_rating for c in contents) / len(contents) if contents else 0

        return {
            "total_content": len(contents),
            "by_category": by_category,
            "by_difficulty": by_difficulty,
            "by_content_type": by_type,
            "total_duration_hours": total_duration / 60,
            "avg_effectiveness_score": round(avg_effectiveness, 3),
            "avg_rating": round(avg_rating, 2),
            "generated_at": datetime.now().isoformat()
        }

    async def record_training_completion(
        self,
        user_id: str,
        content_id: UUID,
        completion_data: Dict[str, Any]
    ) -> bool:
        """
        Record training completion for a user.

        Args:
            user_id: User identifier
            content_id: Completed content
            completion_data: Completion details

        Returns:
            True if recorded
        """
        profile = await self.get_user_profile(user_id)

        if content_id not in profile.completed_trainings:
            profile.completed_trainings.append(content_id)

        profile.training_history.append({
            "content_id": str(content_id),
            "completed": True,
            "completed_at": datetime.now().isoformat(),
            "score": completion_data.get("score"),
            "time_spent_minutes": completion_data.get("time_spent"),
            "rating": completion_data.get("rating")
        })

        # Update learning pace based on completion time
        content = self._content_library.get(content_id)
        if content and completion_data.get("time_spent"):
            actual_time = completion_data["time_spent"]
            expected_time = content.duration_minutes
            profile.learning_pace = (profile.learning_pace + (expected_time / actual_time)) / 2

        profile.last_updated = datetime.now()
        return True
