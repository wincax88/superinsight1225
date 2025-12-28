"""
Tests for Ragas Model Optimizer functionality.

Tests the model comparison, optimization, and selection capabilities
of the Ragas integration module.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from uuid import uuid4

from src.ragas_integration.model_optimizer import (
    ModelOptimizer,
    ModelComparisonEngine,
    OptimizationRecommendation,
    ModelPerformanceProfile,
    ModelComparisonReport
)
from src.ai.base import ModelConfig, ModelType


class TestModelComparisonEngine:
    """Test cases for ModelComparisonEngine."""
    
    @pytest.fixture
    def comparison_engine(self, tmp_path):
        """Create comparison engine with temporary storage."""
        return ModelComparisonEngine(storage_path=str(tmp_path))
    
    @pytest.fixture
    def sample_model_configs(self):
        """Create sample model configurations."""
        return [
            ModelConfig(
                model_name="test_model_1",
                model_type=ModelType.HUGGINGFACE,
                api_key="test_key_1"
            ),
            ModelConfig(
                model_name="test_model_2", 
                model_type=ModelType.OLLAMA,
                api_key="test_key_2"
            )
        ]
    
    @pytest.fixture
    def sample_evaluation_dataset(self):
        """Create sample evaluation dataset."""
        return [
            {
                "question": "什么是人工智能？",
                "contexts": ["人工智能是计算机科学的一个分支"],
                "ground_truth": "人工智能是让机器模拟人类智能的技术"
            },
            {
                "question": "机器学习的主要类型有哪些？",
                "contexts": ["机器学习包括监督学习、无监督学习和强化学习"],
                "ground_truth": "主要包括监督学习、无监督学习和强化学习"
            }
        ]
    
    def test_initialization(self, comparison_engine):
        """Test comparison engine initialization."""
        assert comparison_engine is not None
        assert comparison_engine.ragas_metrics is not None
        assert comparison_engine.metric_weights is not None
        assert len(comparison_engine.comparison_history) == 0
    
    @patch('src.ragas_integration.model_optimizer.RAGAS_AVAILABLE', False)
    async def test_basic_model_comparison_without_ragas(self, comparison_engine, sample_model_configs, sample_evaluation_dataset):
        """Test basic model comparison when Ragas is not available."""
        with patch.object(comparison_engine, '_generate_model_predictions', new_callable=AsyncMock) as mock_predictions:
            mock_predictions.return_value = [
                {
                    "question": "test question",
                    "answer": "test answer",
                    "confidence": 0.8,
                    "contexts": [],
                    "ground_truth": "",
                    "metadata": {}
                }
            ]
            
            result = await comparison_engine._basic_model_comparison(
                sample_model_configs, sample_evaluation_dataset, "test_task"
            )
            
            assert isinstance(result, ModelComparisonReport)
            assert result.comparison_id.startswith("basic_comparison_")
            assert len(result.models_compared) == 2
            assert result.selection_recommendation == "建议安装Ragas进行详细的质量评估和模型对比"
    
    def test_analyze_model_strengths_weaknesses(self, comparison_engine):
        """Test model strengths and weaknesses analysis."""
        ragas_results = {
            "ragas_faithfulness": 0.9,
            "ragas_answer_relevancy": 0.5,
            "ragas_context_precision": 0.8
        }
        
        performance_metrics = {
            "avg_confidence": 0.9,
            "confidence_std": 0.05
        }
        
        strengths, weaknesses = comparison_engine._analyze_model_strengths_weaknesses(
            ragas_results, performance_metrics
        )
        
        assert len(strengths) > 0
        assert len(weaknesses) > 0
        assert any("高忠实度" in s for s in strengths)
        assert any("相关性不足" in w for w in weaknesses)
    
    def test_calculate_optimization_potential(self, comparison_engine):
        """Test optimization potential calculation."""
        ragas_results = {"ragas_faithfulness": 0.6, "ragas_answer_relevancy": 0.7}
        performance_metrics = {"avg_confidence": 0.8, "confidence_std": 0.2}
        
        potential = comparison_engine._calculate_optimization_potential(
            ragas_results, performance_metrics
        )
        
        assert 0.0 <= potential <= 1.0
        assert potential > 0  # Should have some optimization potential
    
    def test_generate_model_specific_recommendations(self, comparison_engine):
        """Test generation of model-specific recommendations."""
        profile = ModelPerformanceProfile(
            model_name="test_model",
            model_type="test_type",
            ragas_metrics={
                "ragas_faithfulness": 0.5,  # Low score should trigger recommendation
                "ragas_answer_relevancy": 0.6
            },
            performance_metrics={
                "avg_confidence": 0.6,
                "confidence_std": 0.4  # High std should trigger consistency recommendation
            },
            optimization_potential=0.8  # High potential should trigger comprehensive optimization
        )
        
        recommendations = comparison_engine._generate_model_specific_recommendations(profile)
        
        assert len(recommendations) > 0
        assert any(rec.recommendation_type == "faithfulness_improvement" for rec in recommendations)
        assert any(rec.recommendation_type == "consistency_improvement" for rec in recommendations)
        assert any(rec.recommendation_type == "comprehensive_optimization" for rec in recommendations)
    
    def test_calculate_quality_rankings(self, comparison_engine):
        """Test quality rankings calculation."""
        ragas_comparison = {
            "model_1": {"ragas_faithfulness": 0.8, "ragas_answer_relevancy": 0.7},
            "model_2": {"ragas_faithfulness": 0.6, "ragas_answer_relevancy": 0.9}
        }
        
        rankings = comparison_engine._calculate_quality_rankings(ragas_comparison)
        
        assert "ragas_faithfulness" in rankings
        assert "ragas_answer_relevancy" in rankings
        
        # Check faithfulness ranking (model_1 should be first)
        faithfulness_ranking = rankings["ragas_faithfulness"]
        assert faithfulness_ranking[0][0] == "model_1"
        assert faithfulness_ranking[0][1] == 0.8
        
        # Check relevancy ranking (model_2 should be first)
        relevancy_ranking = rankings["ragas_answer_relevancy"]
        assert relevancy_ranking[0][0] == "model_2"
        assert relevancy_ranking[0][1] == 0.9
    
    def test_calculate_overall_ranking(self, comparison_engine):
        """Test overall ranking calculation."""
        ragas_comparison = {
            "model_1": {"ragas_faithfulness": 0.8, "ragas_answer_relevancy": 0.7},
            "model_2": {"ragas_faithfulness": 0.6, "ragas_answer_relevancy": 0.9}
        }
        
        overall_ranking = comparison_engine._calculate_overall_ranking(ragas_comparison)
        
        assert len(overall_ranking) == 2
        assert all(isinstance(item, tuple) and len(item) == 2 for item in overall_ranking)
        
        # Rankings should be sorted by score (descending)
        scores = [score for _, score in overall_ranking]
        assert scores == sorted(scores, reverse=True)


class TestModelOptimizer:
    """Test cases for ModelOptimizer."""
    
    @pytest.fixture
    def model_optimizer(self, tmp_path):
        """Create model optimizer with temporary storage."""
        return ModelOptimizer(storage_path=str(tmp_path))
    
    @pytest.fixture
    def sample_optimization_goals(self):
        """Create sample optimization goals."""
        return {
            "ragas_faithfulness": 0.8,
            "ragas_answer_relevancy": 0.75,
            "avg_confidence": 0.8
        }
    
    def test_initialization(self, model_optimizer):
        """Test model optimizer initialization."""
        assert model_optimizer is not None
        assert model_optimizer.comparison_engine is not None
        assert model_optimizer.benchmark_suite is not None
        assert len(model_optimizer.optimization_history) == 0
    
    def test_apply_optimization_goals(self, model_optimizer):
        """Test application of optimization goals to recommendations."""
        # Create mock comparison report
        report = ModelComparisonReport(
            comparison_id="test_comparison",
            models_compared=["test_model"]
        )
        
        # Add model profile
        profile = ModelPerformanceProfile(
            model_name="test_model",
            model_type="test_type",
            ragas_metrics={"ragas_faithfulness": 0.6},  # Below goal of 0.8
            performance_metrics={"avg_confidence": 0.7}  # Below goal of 0.8
        )
        report.model_profiles["test_model"] = profile
        
        # Add recommendation
        recommendation = OptimizationRecommendation(
            model_name="test_model",
            recommendation_type="test_improvement",
            priority="low",
            description="Test recommendation",
            expected_improvement=0.1,
            implementation_effort="easy",
            metrics_to_improve=["ragas_faithfulness"]
        )
        report.optimization_recommendations = [recommendation]
        
        # Apply goals
        goals = {"ragas_faithfulness": 0.8, "avg_confidence": 0.8}
        updated_report = model_optimizer._apply_optimization_goals(report, goals)
        
        # Check that priority was updated
        updated_rec = updated_report.optimization_recommendations[0]
        assert updated_rec.priority == "high"  # Should be upgraded due to large gap
        assert updated_rec.expected_improvement >= 0.2  # Gap to goal
    
    def test_filter_models_by_criteria(self, model_optimizer):
        """Test filtering models by selection criteria."""
        # Create mock comparison report
        report = ModelComparisonReport(
            comparison_id="test_comparison",
            models_compared=["model_1", "model_2"],
            overall_ranking=[("model_1", 0.8), ("model_2", 0.6)]
        )
        
        # Add model profiles
        profile_1 = ModelPerformanceProfile(
            model_name="model_1",
            model_type="test_type",
            ragas_metrics={"ragas_faithfulness": 0.9},
            performance_metrics={"response_time": 1.0}
        )
        
        profile_2 = ModelPerformanceProfile(
            model_name="model_2", 
            model_type="test_type",
            ragas_metrics={"ragas_faithfulness": 0.5},
            performance_metrics={"response_time": 0.5}
        )
        
        report.model_profiles["model_1"] = profile_1
        report.model_profiles["model_2"] = profile_2
        
        # Test filtering
        criteria = {"min_faithfulness": 0.8, "max_response_time": 1.5}
        filtered = model_optimizer._filter_models_by_criteria(report, criteria)
        
        # Only model_1 should pass (faithfulness >= 0.8 and response_time <= 1.5)
        assert len(filtered) == 1
        assert filtered[0][0] == "model_1"
    
    def test_generate_optimization_report(self, model_optimizer):
        """Test optimization report generation."""
        # Create mock comparison report
        report = ModelComparisonReport(
            comparison_id="test_comparison",
            models_compared=["test_model"],
            overall_ranking=[("test_model", 0.75)],
            selection_recommendation="Test recommendation"
        )
        
        # Add optimization recommendation
        recommendation = OptimizationRecommendation(
            model_name="test_model",
            recommendation_type="test_improvement",
            priority="high",
            description="Test optimization",
            expected_improvement=0.2,
            implementation_effort="medium",
            specific_actions=["Action 1", "Action 2"]
        )
        report.optimization_recommendations = [recommendation]
        
        # Add model profile
        profile = ModelPerformanceProfile(
            model_name="test_model",
            model_type="test_type",
            ragas_metrics={"ragas_faithfulness": 0.7},
            strengths=["Good accuracy"],
            weaknesses=["Slow response"],
            optimization_potential=0.6
        )
        report.model_profiles["test_model"] = profile
        
        # Generate report
        report_text = model_optimizer.generate_optimization_report(report, include_detailed_analysis=True)
        
        assert "模型优化分析报告" in report_text
        assert "test_comparison" in report_text
        assert "Test recommendation" in report_text
        assert "高优先级建议" in report_text
        assert "Test optimization" in report_text
        assert "详细模型分析" in report_text
        assert "Good accuracy" in report_text
        assert "Slow response" in report_text


class TestOptimizationRecommendation:
    """Test cases for OptimizationRecommendation."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        recommendation = OptimizationRecommendation(
            model_name="test_model",
            recommendation_type="test_type",
            priority="high",
            description="Test description",
            expected_improvement=0.2,
            implementation_effort="medium",
            specific_actions=["Action 1", "Action 2"],
            metrics_to_improve=["metric1", "metric2"],
            estimated_timeline="2-4 weeks"
        )
        
        result_dict = recommendation.to_dict()
        
        assert result_dict["model_name"] == "test_model"
        assert result_dict["recommendation_type"] == "test_type"
        assert result_dict["priority"] == "high"
        assert result_dict["description"] == "Test description"
        assert result_dict["expected_improvement"] == 0.2
        assert result_dict["implementation_effort"] == "medium"
        assert result_dict["specific_actions"] == ["Action 1", "Action 2"]
        assert result_dict["metrics_to_improve"] == ["metric1", "metric2"]
        assert result_dict["estimated_timeline"] == "2-4 weeks"


class TestModelPerformanceProfile:
    """Test cases for ModelPerformanceProfile."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        profile = ModelPerformanceProfile(
            model_name="test_model",
            model_type="test_type",
            ragas_metrics={"faithfulness": 0.8},
            performance_metrics={"confidence": 0.9},
            quality_trends=[{"date": "2024-01-01", "score": 0.8}],
            strengths=["High accuracy"],
            weaknesses=["Slow response"],
            optimization_potential=0.6
        )
        
        result_dict = profile.to_dict()
        
        assert result_dict["model_name"] == "test_model"
        assert result_dict["model_type"] == "test_type"
        assert result_dict["ragas_metrics"] == {"faithfulness": 0.8}
        assert result_dict["performance_metrics"] == {"confidence": 0.9}
        assert result_dict["strengths"] == ["High accuracy"]
        assert result_dict["weaknesses"] == ["Slow response"]
        assert result_dict["optimization_potential"] == 0.6
        assert "benchmark_date" in result_dict


if __name__ == "__main__":
    pytest.main([__file__])