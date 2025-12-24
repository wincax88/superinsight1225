"""
Unit tests for AI pre-annotation functionality.

Tests LLM interface integration, confidence score calculation, and batch processing logic
as specified in Requirements 10.1, 10.2, 10.5 of the SuperInsight Platform requirements.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4
from datetime import datetime, timedelta
import httpx
from httpx import HTTPError, TimeoutException, ConnectError

from src.ai.base import (
    AIAnnotator, ModelConfig, ModelType, Prediction, AIAnnotationError, ModelUpdateResult
)
from src.ai.factory import AnnotatorFactory, ConfidenceScorer, ModelManager
from src.ai.batch_processor import BatchAnnotationProcessor, BatchJobConfig, BatchStatus, BatchResult
from src.ai.model_manager import ModelVersionManager, ModelVersion, ModelStatus
from src.ai.ollama_annotator import OllamaAnnotator
from src.ai.huggingface_annotator import HuggingFaceAnnotator
from src.ai.zhipu_annotator import ZhipuAnnotator
from src.ai.baidu_annotator import BaiduAnnotator
from src.models.task import Task, TaskStatus


class TestOllamaAnnotatorIntegration:
    """Unit tests for Ollama annotator LLM interface integration."""
    
    def test_ollama_annotator_initialization_success(self):
        """Test successful Ollama annotator initialization."""
        config = ModelConfig(
            model_type=ModelType.OLLAMA,
            model_name="llama2",
            base_url="http://localhost:11434",
            max_tokens=1000,
            temperature=0.7,
            timeout=30
        )
        
        annotator = OllamaAnnotator(config)
        
        assert annotator.config == config
        assert annotator.client is not None
        # Check that timeout is configured (httpx.Timeout object)
        assert annotator.client.timeout is not None
    
    def test_ollama_annotator_initialization_failure_missing_base_url(self):
        """Test Ollama annotator initialization failure with missing base_url."""
        config = ModelConfig(
            model_type=ModelType.OLLAMA,
            model_name="llama2",
            # base_url is missing
            max_tokens=1000,
            temperature=0.7,
            timeout=30
        )
        
        with pytest.raises(ValueError, match="base_url is required for Ollama"):
            OllamaAnnotator(config)
    
    def test_ollama_annotator_initialization_failure_wrong_model_type(self):
        """Test Ollama annotator initialization failure with wrong model type."""
        config = ModelConfig(
            model_type=ModelType.HUGGINGFACE,  # Wrong type
            model_name="llama2",
            base_url="http://localhost:11434",
            max_tokens=1000,
            temperature=0.7,
            timeout=30
        )
        
        with pytest.raises(ValueError, match="ModelConfig must have model_type=OLLAMA"):
            OllamaAnnotator(config)
    
    @pytest.mark.asyncio
    async def test_ollama_predict_success(self):
        """Test successful Ollama prediction."""
        config = ModelConfig(
            model_type=ModelType.OLLAMA,
            model_name="llama2",
            base_url="http://localhost:11434",
            max_tokens=1000,
            temperature=0.7,
            timeout=30
        )
        
        annotator = OllamaAnnotator(config)
        
        task = Task(
            id=uuid4(),
            document_id=uuid4(),
            project_id="sentiment_analysis",
            status=TaskStatus.PENDING
        )
        
        # Mock successful Ollama API response
        mock_response = {
            "response": '{"sentiment": "positive", "confidence": 0.85, "entities": [], "categories": ["review"]}'
        }
        
        with patch.object(annotator, '_make_ollama_request', return_value=mock_response) as mock_request:
            prediction = await annotator.predict(task)
            
            # Verify request was made
            mock_request.assert_called_once()
            
            # Verify prediction structure
            assert isinstance(prediction, Prediction)
            assert prediction.task_id == task.id
            assert prediction.ai_model_config == config
            assert 0.0 <= prediction.confidence <= 1.0
            assert prediction.processing_time > 0
            
            # Verify prediction data
            assert "sentiment" in prediction.prediction_data
            assert prediction.prediction_data["sentiment"] == "positive"
            assert prediction.confidence == 0.85
    
    @pytest.mark.asyncio
    async def test_ollama_predict_api_failure(self):
        """Test Ollama prediction with API failure."""
        config = ModelConfig(
            model_type=ModelType.OLLAMA,
            model_name="llama2",
            base_url="http://localhost:11434",
            max_tokens=1000,
            temperature=0.7,
            timeout=30
        )
        
        annotator = OllamaAnnotator(config)
        
        task = Task(
            id=uuid4(),
            document_id=uuid4(),
            project_id="sentiment_analysis",
            status=TaskStatus.PENDING
        )
        
        # Mock API failure
        with patch.object(annotator, '_make_ollama_request', side_effect=HTTPError("Connection failed")):
            with pytest.raises(AIAnnotationError) as exc_info:
                await annotator.predict(task)
            
            assert "Ollama prediction failed" in str(exc_info.value)
            assert exc_info.value.model_type == "ollama"
            assert exc_info.value.task_id == task.id
    
    @pytest.mark.asyncio
    async def test_ollama_predict_malformed_response(self):
        """Test Ollama prediction with malformed JSON response."""
        config = ModelConfig(
            model_type=ModelType.OLLAMA,
            model_name="llama2",
            base_url="http://localhost:11434",
            max_tokens=1000,
            temperature=0.7,
            timeout=30
        )
        
        annotator = OllamaAnnotator(config)
        
        task = Task(
            id=uuid4(),
            document_id=uuid4(),
            project_id="sentiment_analysis",
            status=TaskStatus.PENDING
        )
        
        # Mock malformed response
        mock_response = {
            "response": "This is not valid JSON content"
        }
        
        with patch.object(annotator, '_make_ollama_request', return_value=mock_response):
            prediction = await annotator.predict(task)
            
            # Should handle gracefully with fallback
            assert isinstance(prediction, Prediction)
            assert prediction.task_id == task.id
            assert "raw_response" in prediction.prediction_data
            assert prediction.confidence == 0.5  # Default fallback confidence
    
    @pytest.mark.asyncio
    async def test_ollama_check_model_availability_success(self):
        """Test successful Ollama model availability check."""
        config = ModelConfig(
            model_type=ModelType.OLLAMA,
            model_name="llama2",
            base_url="http://localhost:11434",
            max_tokens=1000,
            temperature=0.7,
            timeout=30
        )
        
        annotator = OllamaAnnotator(config)
        
        # Mock successful model list response
        mock_models_response = {
            "models": [
                {"name": "llama2"},
                {"name": "codellama"},
                {"name": "mistral"}
            ]
        }
        
        with patch.object(annotator.client, 'get') as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = mock_models_response
            mock_get.return_value = mock_response
            
            is_available = await annotator.check_model_availability()
            
            assert is_available is True
            mock_get.assert_called_once_with(f"{config.base_url}/api/tags")
    
    @pytest.mark.asyncio
    async def test_ollama_check_model_availability_failure(self):
        """Test Ollama model availability check failure."""
        config = ModelConfig(
            model_type=ModelType.OLLAMA,
            model_name="nonexistent_model",
            base_url="http://localhost:11434",
            max_tokens=1000,
            temperature=0.7,
            timeout=30
        )
        
        annotator = OllamaAnnotator(config)
        
        # Mock model list response without the requested model
        mock_models_response = {
            "models": [
                {"name": "llama2"},
                {"name": "codellama"}
            ]
        }
        
        with patch.object(annotator.client, 'get') as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = mock_models_response
            mock_get.return_value = mock_response
            
            is_available = await annotator.check_model_availability()
            
            assert is_available is False


class TestHuggingFaceAnnotatorIntegration:
    """Unit tests for HuggingFace annotator LLM interface integration."""
    
    def test_huggingface_annotator_initialization_success(self):
        """Test successful HuggingFace annotator initialization."""
        config = ModelConfig(
            model_type=ModelType.HUGGINGFACE,
            model_name="bert-base-uncased",
            api_key="hf_test_key",
            max_tokens=512,
            temperature=0.5,
            timeout=60
        )
        
        annotator = HuggingFaceAnnotator(config)
        
        assert annotator.config == config
        assert annotator.client is not None
        assert "Authorization" in annotator.client.headers
        assert annotator.client.headers["Authorization"] == "Bearer hf_test_key"
    
    def test_huggingface_annotator_initialization_without_api_key(self):
        """Test HuggingFace annotator initialization without API key (public models)."""
        config = ModelConfig(
            model_type=ModelType.HUGGINGFACE,
            model_name="bert-base-uncased",
            # No API key for public models
            max_tokens=512,
            temperature=0.5,
            timeout=60
        )
        
        annotator = HuggingFaceAnnotator(config)
        
        assert annotator.config == config
        assert annotator.client is not None
        # Should not have Authorization header for public models
        assert "Authorization" not in annotator.client.headers
    
    @pytest.mark.asyncio
    async def test_huggingface_sentiment_analysis_success(self):
        """Test successful HuggingFace sentiment analysis."""
        config = ModelConfig(
            model_type=ModelType.HUGGINGFACE,
            model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
            max_tokens=512,
            temperature=0.5,
            timeout=60
        )
        
        annotator = HuggingFaceAnnotator(config)
        
        task = Task(
            id=uuid4(),
            document_id=uuid4(),
            project_id="sentiment_analysis",
            status=TaskStatus.PENDING
        )
        
        # Mock successful HuggingFace API response for sentiment analysis
        mock_response = [
            [
                {"label": "POSITIVE", "score": 0.8234},
                {"label": "NEGATIVE", "score": 0.1234},
                {"label": "NEUTRAL", "score": 0.0532}
            ]
        ]
        
        with patch.object(annotator.client, 'post') as mock_post:
            mock_http_response = Mock()
            mock_http_response.raise_for_status.return_value = None
            mock_http_response.json.return_value = mock_response
            mock_post.return_value = mock_http_response
            
            prediction = await annotator.predict(task)
            
            # Verify request was made
            mock_post.assert_called_once()
            
            # Verify prediction structure
            assert isinstance(prediction, Prediction)
            assert prediction.task_id == task.id
            assert prediction.ai_model_config == config
            assert 0.0 <= prediction.confidence <= 1.0
            
            # Verify sentiment analysis results
            assert prediction.prediction_data["sentiment"] == "POSITIVE"
            assert prediction.prediction_data["task_type"] == "sentiment_analysis"
            assert prediction.confidence == 0.8234
    
    @pytest.mark.asyncio
    async def test_huggingface_ner_success(self):
        """Test successful HuggingFace named entity recognition."""
        config = ModelConfig(
            model_type=ModelType.HUGGINGFACE,
            model_name="dbmdz/bert-large-cased-finetuned-conll03-english",
            max_tokens=512,
            temperature=0.5,
            timeout=60
        )
        
        annotator = HuggingFaceAnnotator(config)
        
        task = Task(
            id=uuid4(),
            document_id=uuid4(),
            project_id="ner_extraction",
            status=TaskStatus.PENDING
        )
        
        # Mock successful HuggingFace API response for NER
        mock_response = [
            {"word": "John", "entity": "B-PER", "score": 0.9998, "start": 0, "end": 4},
            {"word": "Doe", "entity": "I-PER", "score": 0.9995, "start": 5, "end": 8},
            {"word": "Apple", "entity": "B-ORG", "score": 0.9987, "start": 20, "end": 25}
        ]
        
        with patch.object(annotator.client, 'post') as mock_post:
            mock_http_response = Mock()
            mock_http_response.raise_for_status.return_value = None
            mock_http_response.json.return_value = mock_response
            mock_post.return_value = mock_http_response
            
            prediction = await annotator.predict(task)
            
            # Verify prediction structure
            assert isinstance(prediction, Prediction)
            assert prediction.task_id == task.id
            
            # Verify NER results
            assert prediction.prediction_data["task_type"] == "named_entity_recognition"
            assert "entities" in prediction.prediction_data
            entities = prediction.prediction_data["entities"]
            assert len(entities) == 3
            assert entities[0]["text"] == "John"
            assert entities[0]["label"] == "B-PER"
            assert entities[0]["confidence"] == 0.9998
    
    @pytest.mark.asyncio
    async def test_huggingface_api_error_handling(self):
        """Test HuggingFace API error handling."""
        config = ModelConfig(
            model_type=ModelType.HUGGINGFACE,
            model_name="invalid-model",
            max_tokens=512,
            temperature=0.5,
            timeout=60
        )
        
        annotator = HuggingFaceAnnotator(config)
        
        task = Task(
            id=uuid4(),
            document_id=uuid4(),
            project_id="sentiment_analysis",
            status=TaskStatus.PENDING
        )
        
        # Mock API error
        with patch.object(annotator.client, 'post', side_effect=HTTPError("Model not found")):
            with pytest.raises(AIAnnotationError) as exc_info:
                await annotator.predict(task)
            
            assert "HuggingFace prediction failed" in str(exc_info.value)
            assert exc_info.value.model_type == "huggingface"
            assert exc_info.value.task_id == task.id


class TestZhipuAnnotatorIntegration:
    """Unit tests for Zhipu GLM annotator LLM interface integration."""
    
    def test_zhipu_annotator_initialization_success(self):
        """Test successful Zhipu annotator initialization."""
        config = ModelConfig(
            model_type=ModelType.ZHIPU_GLM,
            model_name="glm-4",
            api_key="zhipu_test_key",
            max_tokens=1000,
            temperature=0.7,
            timeout=30
        )
        
        annotator = ZhipuAnnotator(config)
        
        assert annotator.config == config
        assert annotator.client is not None
        assert "Authorization" in annotator.client.headers
        assert annotator.client.headers["Authorization"] == "Bearer zhipu_test_key"
    
    def test_zhipu_annotator_initialization_failure_missing_api_key(self):
        """Test Zhipu annotator initialization failure with missing API key."""
        config = ModelConfig(
            model_type=ModelType.ZHIPU_GLM,
            model_name="glm-4",
            # api_key is missing
            max_tokens=1000,
            temperature=0.7,
            timeout=30
        )
        
        with pytest.raises(ValueError, match="api_key is required for Zhipu GLM"):
            ZhipuAnnotator(config)
    
    @pytest.mark.asyncio
    async def test_zhipu_predict_success(self):
        """Test successful Zhipu GLM prediction."""
        config = ModelConfig(
            model_type=ModelType.ZHIPU_GLM,
            model_name="glm-4",
            api_key="zhipu_test_key",
            max_tokens=1000,
            temperature=0.7,
            timeout=30
        )
        
        annotator = ZhipuAnnotator(config)
        
        task = Task(
            id=uuid4(),
            document_id=uuid4(),
            project_id="sentiment_analysis",
            status=TaskStatus.PENDING
        )
        
        # Mock successful Zhipu API response
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": '{"sentiment": "positive", "confidence": 0.92, "reasoning": "积极的情感表达"}'
                    }
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
            "model": "glm-4"
        }
        
        with patch.object(annotator, '_make_chat_completion_request', return_value=mock_response):
            prediction = await annotator.predict(task)
            
            # Verify prediction structure
            assert isinstance(prediction, Prediction)
            assert prediction.task_id == task.id
            assert prediction.ai_model_config == config
            assert 0.0 <= prediction.confidence <= 1.0
            
            # Verify prediction data
            assert prediction.prediction_data["sentiment"] == "positive"
            assert prediction.confidence == 0.92
            assert "model_response" in prediction.prediction_data
    
    @pytest.mark.asyncio
    async def test_zhipu_check_model_availability_success(self):
        """Test successful Zhipu model availability check."""
        config = ModelConfig(
            model_type=ModelType.ZHIPU_GLM,
            model_name="glm-4",
            api_key="zhipu_test_key",
            max_tokens=1000,
            temperature=0.7,
            timeout=30
        )
        
        annotator = ZhipuAnnotator(config)
        
        # Mock successful test request
        with patch.object(annotator.client, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            
            is_available = await annotator.check_model_availability()
            
            assert is_available is True
            mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_zhipu_check_model_availability_failure(self):
        """Test Zhipu model availability check failure."""
        config = ModelConfig(
            model_type=ModelType.ZHIPU_GLM,
            model_name="glm-4",
            api_key="invalid_key",
            max_tokens=1000,
            temperature=0.7,
            timeout=30
        )
        
        annotator = ZhipuAnnotator(config)
        
        # Mock failed test request
        with patch.object(annotator.client, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 401  # Unauthorized
            mock_post.return_value = mock_response
            
            is_available = await annotator.check_model_availability()
            
            assert is_available is False


class TestBaiduAnnotatorIntegration:
    """Unit tests for Baidu Wenxin annotator LLM interface integration."""
    
    def test_baidu_annotator_initialization_success(self):
        """Test successful Baidu annotator initialization."""
        config = ModelConfig(
            model_type=ModelType.BAIDU_WENXIN,
            model_name="ernie-bot",
            api_key="baidu_api_key",
            base_url="baidu_secret_key",  # Secret key stored in base_url
            max_tokens=1000,
            temperature=0.7,
            timeout=30
        )
        
        annotator = BaiduAnnotator(config)
        
        assert annotator.config == config
        assert annotator.client is not None
        assert annotator.access_token is None  # Should be None initially
    
    def test_baidu_annotator_initialization_failure_missing_credentials(self):
        """Test Baidu annotator initialization failure with missing credentials."""
        # Missing API key
        config = ModelConfig(
            model_type=ModelType.BAIDU_WENXIN,
            model_name="ernie-bot",
            # api_key is missing
            base_url="baidu_secret_key",
            max_tokens=1000,
            temperature=0.7,
            timeout=30
        )
        
        with pytest.raises(ValueError, match="api_key \\(API Key\\) is required for Baidu Wenxin"):
            BaiduAnnotator(config)
        
        # Missing secret key
        config = ModelConfig(
            model_type=ModelType.BAIDU_WENXIN,
            model_name="ernie-bot",
            api_key="baidu_api_key",
            # base_url (secret key) is missing
            max_tokens=1000,
            temperature=0.7,
            timeout=30
        )
        
        with pytest.raises(ValueError, match="base_url \\(Secret Key\\) is required for Baidu Wenxin"):
            BaiduAnnotator(config)
    
    @pytest.mark.asyncio
    async def test_baidu_get_access_token_success(self):
        """Test successful Baidu access token retrieval."""
        config = ModelConfig(
            model_type=ModelType.BAIDU_WENXIN,
            model_name="ernie-bot",
            api_key="baidu_api_key",
            base_url="baidu_secret_key",
            max_tokens=1000,
            temperature=0.7,
            timeout=30
        )
        
        annotator = BaiduAnnotator(config)
        
        # Mock successful token response
        mock_token_response = {
            "access_token": "test_access_token_123",
            "expires_in": 3600
        }
        
        with patch.object(annotator.client, 'post') as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = mock_token_response
            mock_post.return_value = mock_response
            
            token = await annotator._get_access_token()
            
            assert token == "test_access_token_123"
            assert annotator.access_token == "test_access_token_123"
            mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_baidu_predict_success(self):
        """Test successful Baidu Wenxin prediction."""
        config = ModelConfig(
            model_type=ModelType.BAIDU_WENXIN,
            model_name="ernie-bot",
            api_key="baidu_api_key",
            base_url="baidu_secret_key",
            max_tokens=1000,
            temperature=0.7,
            timeout=30
        )
        
        annotator = BaiduAnnotator(config)
        
        task = Task(
            id=uuid4(),
            document_id=uuid4(),
            project_id="sentiment_analysis",
            status=TaskStatus.PENDING
        )
        
        # Mock successful Baidu API response
        mock_response = {
            "result": '{"sentiment": "positive", "confidence": 0.88, "reasoning": "正面情感"}',
            "usage": {"prompt_tokens": 45, "completion_tokens": 25, "total_tokens": 70},
            "id": "baidu_response_id",
            "created": 1640995200
        }
        
        with patch.object(annotator, '_make_chat_completion_request', return_value=mock_response):
            prediction = await annotator.predict(task)
            
            # Verify prediction structure
            assert isinstance(prediction, Prediction)
            assert prediction.task_id == task.id
            assert prediction.ai_model_config == config
            assert 0.0 <= prediction.confidence <= 1.0
            
            # Verify prediction data
            assert prediction.prediction_data["sentiment"] == "positive"
            assert prediction.confidence == 0.88
            assert "model_response" in prediction.prediction_data


class TestConfidenceScoreCalculation:
    """Unit tests for confidence score calculation functionality."""
    
    def test_confidence_scorer_calculate_ensemble_average(self):
        """Test ensemble confidence calculation using average method."""
        confidences = [0.8, 0.9, 0.7, 0.85]
        
        result = ConfidenceScorer.calculate_ensemble_confidence(confidences, "average")
        
        expected = sum(confidences) / len(confidences)
        assert abs(result - expected) < 1e-10
        assert 0.0 <= result <= 1.0
    
    def test_confidence_scorer_calculate_ensemble_max(self):
        """Test ensemble confidence calculation using max method."""
        confidences = [0.8, 0.9, 0.7, 0.85]
        
        result = ConfidenceScorer.calculate_ensemble_confidence(confidences, "max")
        
        assert result == max(confidences)
        assert result == 0.9
    
    def test_confidence_scorer_calculate_ensemble_min(self):
        """Test ensemble confidence calculation using min method."""
        confidences = [0.8, 0.9, 0.7, 0.85]
        
        result = ConfidenceScorer.calculate_ensemble_confidence(confidences, "min")
        
        assert result == min(confidences)
        assert result == 0.7
    
    def test_confidence_scorer_calculate_ensemble_weighted_average(self):
        """Test ensemble confidence calculation using weighted average method."""
        confidences = [0.8, 0.9, 0.7]
        
        result = ConfidenceScorer.calculate_ensemble_confidence(confidences, "weighted_average")
        
        # With equal weights, should be same as average
        expected = sum(confidences) / len(confidences)
        assert abs(result - expected) < 1e-10
    
    def test_confidence_scorer_empty_list(self):
        """Test ensemble confidence calculation with empty confidence list."""
        confidences = []
        
        result = ConfidenceScorer.calculate_ensemble_confidence(confidences, "average")
        
        assert result == 0.0
    
    def test_confidence_scorer_invalid_method(self):
        """Test ensemble confidence calculation with invalid method."""
        confidences = [0.8, 0.9, 0.7]
        
        with pytest.raises(ValueError, match="Unknown ensemble method"):
            ConfidenceScorer.calculate_ensemble_confidence(confidences, "invalid_method")
    
    def test_confidence_scorer_adjust_by_model_type(self):
        """Test confidence adjustment based on model type."""
        base_confidence = 0.8
        
        # Test different model types
        ollama_adjusted = ConfidenceScorer.adjust_confidence_by_model_type(
            base_confidence, ModelType.OLLAMA
        )
        assert ollama_adjusted == 0.8 * 0.9  # Ollama has 0.9 adjustment factor
        
        huggingface_adjusted = ConfidenceScorer.adjust_confidence_by_model_type(
            base_confidence, ModelType.HUGGINGFACE
        )
        assert huggingface_adjusted == 0.8 * 1.0  # HuggingFace has 1.0 adjustment factor
        
        zhipu_adjusted = ConfidenceScorer.adjust_confidence_by_model_type(
            base_confidence, ModelType.ZHIPU_GLM
        )
        assert zhipu_adjusted == 0.8 * 1.1  # Zhipu has 1.1 adjustment factor
    
    def test_confidence_scorer_adjust_clamps_to_valid_range(self):
        """Test confidence adjustment clamps values to valid range [0.0, 1.0]."""
        # Test upper bound clamping
        high_confidence = 0.95
        adjusted_high = ConfidenceScorer.adjust_confidence_by_model_type(
            high_confidence, ModelType.ZHIPU_GLM  # 1.1 factor
        )
        assert adjusted_high == 1.0  # Should be clamped to 1.0
        
        # Test lower bound clamping (hypothetical negative adjustment)
        low_confidence = 0.1
        # Simulate a very low adjustment factor
        with patch.dict('src.ai.factory.ConfidenceScorer.adjust_confidence_by_model_type.__globals__', 
                       {'adjustments': {ModelType.OLLAMA: 0.1}}):
            adjusted_low = ConfidenceScorer.adjust_confidence_by_model_type(
                low_confidence, ModelType.OLLAMA
            )
            assert adjusted_low >= 0.0  # Should not go below 0.0
    
    def test_confidence_scorer_validate_confidence_range(self):
        """Test confidence range validation."""
        # Valid confidence values
        assert ConfidenceScorer.validate_confidence_range(0.0) == 0.0
        assert ConfidenceScorer.validate_confidence_range(0.5) == 0.5
        assert ConfidenceScorer.validate_confidence_range(1.0) == 1.0
        
        # Invalid confidence values (should be clamped)
        assert ConfidenceScorer.validate_confidence_range(-0.5) == 0.0
        assert ConfidenceScorer.validate_confidence_range(1.5) == 1.0
        assert ConfidenceScorer.validate_confidence_range(100.0) == 1.0
    
    def test_confidence_scorer_calculate_from_agreement(self):
        """Test confidence calculation based on prediction agreement."""
        # Test sentiment agreement
        predictions = [
            {"sentiment": "positive", "confidence": 0.8},
            {"sentiment": "positive", "confidence": 0.9},
            {"sentiment": "negative", "confidence": 0.7}
        ]
        
        agreement_confidence = ConfidenceScorer.calculate_confidence_from_agreement(predictions)
        
        # 2 out of 3 predictions agree on "positive"
        assert agreement_confidence == 2/3
        
        # Test single prediction
        single_prediction = [{"sentiment": "positive", "confidence": 0.8}]
        single_confidence = ConfidenceScorer.calculate_confidence_from_agreement(single_prediction)
        assert single_confidence == 0.8
        
        # Test empty predictions
        empty_confidence = ConfidenceScorer.calculate_confidence_from_agreement([])
        assert empty_confidence == 0.0


class TestBatchProcessingLogic:
    """Unit tests for batch processing logic."""
    
    def test_batch_job_config_initialization(self):
        """Test batch job configuration initialization."""
        model_configs = [
            ModelConfig(
                model_type=ModelType.OLLAMA,
                model_name="llama2",
                base_url="http://localhost:11434"
            ),
            ModelConfig(
                model_type=ModelType.HUGGINGFACE,
                model_name="bert-base-uncased"
            )
        ]
        
        config = BatchJobConfig(
            model_configs=model_configs,
            max_concurrent_tasks=5,
            retry_attempts=2,
            timeout_seconds=60,
            enable_caching=True,
            confidence_threshold=0.6
        )
        
        assert len(config.model_configs) == 2
        assert config.max_concurrent_tasks == 5
        assert config.retry_attempts == 2
        assert config.timeout_seconds == 60
        assert config.enable_caching is True
        assert config.confidence_threshold == 0.6
        assert config.ensemble_method == "average"  # Default value
    
    def test_batch_result_initialization(self):
        """Test batch result initialization and properties."""
        result = BatchResult(
            job_id="test_job_123",
            status=BatchStatus.PENDING,
            total_tasks=10,
            completed_tasks=0,
            failed_tasks=0
        )
        
        assert result.job_id == "test_job_123"
        assert result.status == BatchStatus.PENDING
        assert result.total_tasks == 10
        assert result.success_rate == 0.0  # No completed tasks yet
        
        # Update with some completed tasks
        result.completed_tasks = 7
        result.failed_tasks = 2
        
        assert result.success_rate == 70.0  # 7/10 * 100
    
    def test_batch_result_to_dict(self):
        """Test batch result serialization to dictionary."""
        result = BatchResult(
            job_id="test_job_123",
            status=BatchStatus.COMPLETED,
            total_tasks=5,
            completed_tasks=4,
            failed_tasks=1,
            started_at=datetime(2024, 1, 1, 12, 0, 0),
            completed_at=datetime(2024, 1, 1, 12, 5, 0),
            processing_time=300.0
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["job_id"] == "test_job_123"
        assert result_dict["status"] == "completed"
        assert result_dict["total_tasks"] == 5
        assert result_dict["completed_tasks"] == 4
        assert result_dict["failed_tasks"] == 1
        assert result_dict["success_rate"] == 80.0
        assert result_dict["processing_time"] == 300.0
        assert "2024-01-01T12:00:00" in result_dict["started_at"]
        assert "2024-01-01T12:05:00" in result_dict["completed_at"]
    
    @pytest.mark.asyncio
    async def test_batch_processor_initialization(self):
        """Test batch processor initialization."""
        # Without Redis
        processor = BatchAnnotationProcessor()
        
        assert processor.redis_client is None
        assert processor.active_jobs == {}
        assert processor.executor is not None
        
        # With Redis (mocked)
        mock_redis = Mock()
        processor_with_redis = BatchAnnotationProcessor(redis_client=mock_redis)
        
        assert processor_with_redis.redis_client == mock_redis
    
    @pytest.mark.asyncio
    async def test_batch_processor_submit_job(self):
        """Test batch job submission."""
        processor = BatchAnnotationProcessor()
        
        # Create test tasks
        tasks = [
            Task(
                id=uuid4(),
                document_id=uuid4(),
                project_id="test_project",
                status=TaskStatus.PENDING
            )
            for _ in range(3)
        ]
        
        # Create batch config
        model_configs = [
            ModelConfig(
                model_type=ModelType.OLLAMA,
                model_name="llama2",
                base_url="http://localhost:11434"
            )
        ]
        
        config = BatchJobConfig(
            model_configs=model_configs,
            max_concurrent_tasks=2,
            retry_attempts=1,
            timeout_seconds=30
        )
        
        # Submit job
        job_id = await processor.submit_batch_job(tasks, config)
        
        assert job_id is not None
        assert job_id in processor.active_jobs
        
        # Check initial job status
        job_status = processor.active_jobs[job_id]
        assert job_status.total_tasks == 3
        assert job_status.status in [BatchStatus.PENDING, BatchStatus.PROCESSING]
    
    @pytest.mark.asyncio
    async def test_batch_processor_get_job_status(self):
        """Test getting batch job status."""
        processor = BatchAnnotationProcessor()
        
        # Create a test job result
        job_id = "test_job_456"
        test_result = BatchResult(
            job_id=job_id,
            status=BatchStatus.COMPLETED,
            total_tasks=2,
            completed_tasks=2,
            failed_tasks=0
        )
        
        processor.active_jobs[job_id] = test_result
        
        # Get job status
        status = await processor.get_job_status(job_id)
        
        assert status is not None
        assert status.job_id == job_id
        assert status.status == BatchStatus.COMPLETED
        assert status.total_tasks == 2
        assert status.completed_tasks == 2
    
    @pytest.mark.asyncio
    async def test_batch_processor_cancel_job(self):
        """Test cancelling a batch job."""
        processor = BatchAnnotationProcessor()
        
        # Create a test job in progress
        job_id = "test_job_789"
        test_result = BatchResult(
            job_id=job_id,
            status=BatchStatus.PROCESSING,
            total_tasks=5,
            completed_tasks=2,
            failed_tasks=0
        )
        
        processor.active_jobs[job_id] = test_result
        
        # Cancel job
        cancelled = await processor.cancel_job(job_id)
        
        assert cancelled is True
        assert processor.active_jobs[job_id].status == BatchStatus.CANCELLED
        assert processor.active_jobs[job_id].completed_at is not None
    
    @pytest.mark.asyncio
    async def test_batch_processor_cleanup_completed_jobs(self):
        """Test cleanup of old completed jobs."""
        processor = BatchAnnotationProcessor()
        
        # Create old completed jobs
        old_time = datetime.now() - timedelta(hours=25)  # 25 hours ago
        recent_time = datetime.now() - timedelta(hours=1)  # 1 hour ago
        
        old_job = BatchResult(
            job_id="old_job",
            status=BatchStatus.COMPLETED,
            total_tasks=1,
            completed_tasks=1,
            failed_tasks=0,
            completed_at=old_time
        )
        
        recent_job = BatchResult(
            job_id="recent_job",
            status=BatchStatus.COMPLETED,
            total_tasks=1,
            completed_tasks=1,
            failed_tasks=0,
            completed_at=recent_time
        )
        
        active_job = BatchResult(
            job_id="active_job",
            status=BatchStatus.PROCESSING,
            total_tasks=1,
            completed_tasks=0,
            failed_tasks=0
        )
        
        processor.active_jobs["old_job"] = old_job
        processor.active_jobs["recent_job"] = recent_job
        processor.active_jobs["active_job"] = active_job
        
        # Cleanup jobs older than 24 hours
        cleaned_count = await processor.cleanup_completed_jobs(max_age_hours=24)
        
        assert cleaned_count == 1  # Only old_job should be cleaned
        assert "old_job" not in processor.active_jobs
        assert "recent_job" in processor.active_jobs  # Should remain
        assert "active_job" in processor.active_jobs  # Should remain (still active)
    
    @pytest.mark.asyncio
    async def test_batch_processor_get_statistics(self):
        """Test getting batch processor statistics."""
        processor = BatchAnnotationProcessor()
        
        # Add some test jobs
        processor.active_jobs["job1"] = BatchResult(
            job_id="job1",
            status=BatchStatus.PROCESSING,
            total_tasks=5,
            completed_tasks=3,
            failed_tasks=0
        )
        
        processor.active_jobs["job2"] = BatchResult(
            job_id="job2",
            status=BatchStatus.COMPLETED,
            total_tasks=2,
            completed_tasks=2,
            failed_tasks=0
        )
        
        processor.active_jobs["job3"] = BatchResult(
            job_id="job3",
            status=BatchStatus.FAILED,
            total_tasks=1,
            completed_tasks=0,
            failed_tasks=1
        )
        
        # Get statistics
        stats = await processor.get_job_statistics()
        
        assert stats["total_jobs"] == 3
        assert stats["status_counts"]["processing"] == 1
        assert stats["status_counts"]["completed"] == 1
        assert stats["status_counts"]["failed"] == 1
        assert len(stats["active_jobs"]) == 3


class TestAnnotatorFactory:
    """Unit tests for annotator factory functionality."""
    
    def test_factory_create_ollama_annotator(self):
        """Test factory creation of Ollama annotator."""
        config = ModelConfig(
            model_type=ModelType.OLLAMA,
            model_name="llama2",
            base_url="http://localhost:11434"
        )
        
        annotator = AnnotatorFactory.create_annotator(config)
        
        assert isinstance(annotator, OllamaAnnotator)
        assert annotator.config == config
    
    def test_factory_create_huggingface_annotator(self):
        """Test factory creation of HuggingFace annotator."""
        config = ModelConfig(
            model_type=ModelType.HUGGINGFACE,
            model_name="bert-base-uncased"
        )
        
        annotator = AnnotatorFactory.create_annotator(config)
        
        assert isinstance(annotator, HuggingFaceAnnotator)
        assert annotator.config == config
    
    def test_factory_create_zhipu_annotator(self):
        """Test factory creation of Zhipu annotator."""
        config = ModelConfig(
            model_type=ModelType.ZHIPU_GLM,
            model_name="glm-4",
            api_key="test_key"
        )
        
        annotator = AnnotatorFactory.create_annotator(config)
        
        assert isinstance(annotator, ZhipuAnnotator)
        assert annotator.config == config
    
    def test_factory_create_baidu_annotator(self):
        """Test factory creation of Baidu annotator."""
        config = ModelConfig(
            model_type=ModelType.BAIDU_WENXIN,
            model_name="ernie-bot",
            api_key="api_key",
            base_url="secret_key"
        )
        
        annotator = AnnotatorFactory.create_annotator(config)
        
        assert isinstance(annotator, BaiduAnnotator)
        assert annotator.config == config
    
    def test_factory_unsupported_model_type(self):
        """Test factory with unsupported model type."""
        # Create a fake model type that doesn't exist
        from enum import Enum
        
        class FakeModelType(str, Enum):
            FAKE_MODEL = "fake_model"
        
        # This should fail because we're using a non-existent model type
        config = ModelConfig(
            model_type=ModelType.TENCENT_HUNYUAN,  # This one is not fully implemented yet
            model_name="hunyuan-test"
        )
        
        with pytest.raises(AIAnnotationError):
            AnnotatorFactory.create_annotator(config)
    
    def test_factory_get_supported_model_types(self):
        """Test getting supported model types from factory."""
        supported_types = AnnotatorFactory.get_supported_model_types()
        
        assert ModelType.OLLAMA in supported_types
        assert ModelType.HUGGINGFACE in supported_types
        assert ModelType.ZHIPU_GLM in supported_types
        assert ModelType.BAIDU_WENXIN in supported_types
        assert ModelType.ALIBABA_TONGYI in supported_types  # Now supported
        # TENCENT_HUNYUAN should not be in supported types yet
    
    def test_factory_is_model_type_supported(self):
        """Test checking if model type is supported."""
        assert AnnotatorFactory.is_model_type_supported(ModelType.OLLAMA) is True
        assert AnnotatorFactory.is_model_type_supported(ModelType.HUGGINGFACE) is True
        assert AnnotatorFactory.is_model_type_supported(ModelType.ALIBABA_TONGYI) is True  # Now supported
        assert AnnotatorFactory.is_model_type_supported(ModelType.TENCENT_HUNYUAN) is True  # Now supported
    
    def test_factory_create_from_dict(self):
        """Test factory creation from dictionary configuration."""
        config_dict = {
            "model_type": "ollama",
            "model_name": "llama2",
            "base_url": "http://localhost:11434",
            "max_tokens": 1000,
            "temperature": 0.7,
            "timeout": 30
        }
        
        annotator = AnnotatorFactory.create_from_dict(config_dict)
        
        assert isinstance(annotator, OllamaAnnotator)
        assert annotator.config.model_type == ModelType.OLLAMA
        assert annotator.config.model_name == "llama2"
    
    def test_factory_create_multiple(self):
        """Test factory creation of multiple annotators."""
        configs = [
            ModelConfig(
                model_type=ModelType.OLLAMA,
                model_name="llama2",
                base_url="http://localhost:11434"
            ),
            ModelConfig(
                model_type=ModelType.HUGGINGFACE,
                model_name="bert-base-uncased"
            ),
            ModelConfig(
                model_type=ModelType.ZHIPU_GLM,
                model_name="glm-4",
                api_key="test_key"
            )
        ]
        
        annotators = AnnotatorFactory.create_multiple(configs)
        
        assert len(annotators) == 3
        assert isinstance(annotators[0], OllamaAnnotator)
        assert isinstance(annotators[1], HuggingFaceAnnotator)
        assert isinstance(annotators[2], ZhipuAnnotator)


class TestModelManager:
    """Unit tests for model manager functionality."""
    
    def test_model_manager_initialization(self):
        """Test model manager initialization."""
        manager = ModelManager()
        
        assert manager.annotators == {}
        assert manager.default_configs == {}
    
    def test_model_manager_add_annotator(self):
        """Test adding annotator to model manager."""
        manager = ModelManager()
        
        config = ModelConfig(
            model_type=ModelType.OLLAMA,
            model_name="llama2",
            base_url="http://localhost:11434"
        )
        
        with patch.object(AnnotatorFactory, 'create_annotator') as mock_factory:
            mock_annotator = Mock(spec=OllamaAnnotator)
            mock_factory.return_value = mock_annotator
            
            manager.add_annotator("test_ollama", config)
            
            assert "test_ollama" in manager.annotators
            assert manager.annotators["test_ollama"] == mock_annotator
            mock_factory.assert_called_once_with(config)
    
    def test_model_manager_get_annotator(self):
        """Test getting annotator from model manager."""
        manager = ModelManager()
        
        mock_annotator = Mock(spec=OllamaAnnotator)
        manager.annotators["test_annotator"] = mock_annotator
        
        retrieved = manager.get_annotator("test_annotator")
        assert retrieved == mock_annotator
        
        # Test non-existent annotator
        not_found = manager.get_annotator("nonexistent")
        assert not_found is None
    
    def test_model_manager_list_annotators(self):
        """Test listing annotators in model manager."""
        manager = ModelManager()
        
        manager.annotators["annotator1"] = Mock()
        manager.annotators["annotator2"] = Mock()
        
        annotator_names = manager.list_annotators()
        
        assert len(annotator_names) == 2
        assert "annotator1" in annotator_names
        assert "annotator2" in annotator_names
    
    def test_model_manager_remove_annotator(self):
        """Test removing annotator from model manager."""
        manager = ModelManager()
        
        manager.annotators["test_annotator"] = Mock()
        
        # Remove existing annotator
        removed = manager.remove_annotator("test_annotator")
        assert removed is True
        assert "test_annotator" not in manager.annotators
        
        # Try to remove non-existent annotator
        not_removed = manager.remove_annotator("nonexistent")
        assert not_removed is False
    
    def test_model_manager_default_configs(self):
        """Test default configuration management."""
        manager = ModelManager()
        
        config = ModelConfig(
            model_type=ModelType.OLLAMA,
            model_name="llama2",
            base_url="http://localhost:11434"
        )
        
        # Set default config
        manager.set_default_config(ModelType.OLLAMA, config)
        
        # Get default config
        retrieved_config = manager.get_default_config(ModelType.OLLAMA)
        assert retrieved_config == config
        
        # Get non-existent default config
        not_found_config = manager.get_default_config(ModelType.HUGGINGFACE)
        assert not_found_config is None
    
    @pytest.mark.asyncio
    async def test_model_manager_health_check(self):
        """Test model manager health check."""
        manager = ModelManager()
        
        # Create mock annotators with health check methods
        mock_annotator1 = Mock()
        mock_annotator1.check_model_availability = AsyncMock(return_value=True)
        
        mock_annotator2 = Mock()
        mock_annotator2.check_model_availability = AsyncMock(return_value=False)
        
        mock_annotator3 = Mock()
        # No check_model_availability method (should default to True)
        
        manager.annotators["healthy"] = mock_annotator1
        manager.annotators["unhealthy"] = mock_annotator2
        manager.annotators["no_check"] = mock_annotator3
        
        health_status = await manager.health_check()
        
        assert health_status["healthy"] is True
        assert health_status["unhealthy"] is False
        assert health_status["no_check"] is False  # Should be False when check method fails


if __name__ == "__main__":
    pytest.main([__file__, "-v"])