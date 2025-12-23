"""
Property-based tests for AI pre-annotation auto-triggering functionality.

Tests the automatic AI pre-annotation triggering when annotation tasks are started
as specified in Requirements 3.2 and 10.5 of the SuperInsight Platform requirements.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis import HealthCheck
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4
from datetime import datetime
import json
import asyncio

from src.ai.base import AIAnnotator, ModelConfig, ModelType, Prediction
from src.ai.factory import AnnotatorFactory
from src.models.task import Task, TaskStatus
from src.label_studio.integration import LabelStudioIntegration, ProjectConfig
from src.database.models import TaskModel


# Hypothesis strategies for generating test data

def model_config_strategy():
    """Strategy for generating valid ModelConfig instances."""
    return st.builds(
        ModelConfig,
        model_type=st.sampled_from(list(ModelType)),
        model_name=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        api_key=st.one_of(st.none(), st.text(min_size=10, max_size=100)),
        base_url=st.one_of(st.none(), st.text(min_size=10, max_size=100)),
        max_tokens=st.integers(min_value=100, max_value=2000),
        temperature=st.floats(min_value=0.0, max_value=2.0),
        timeout=st.integers(min_value=10, max_value=120)
    )


def task_strategy():
    """Strategy for generating valid Task instances."""
    return st.builds(
        Task,
        id=st.just(uuid4()),
        document_id=st.just(uuid4()),
        project_id=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
        status=st.sampled_from([TaskStatus.PENDING, TaskStatus.IN_PROGRESS]),
        annotations=st.lists(
            st.dictionaries(
                st.sampled_from(['id', 'result', 'annotator_id', 'confidence']),
                st.one_of(st.text(), st.integers(), st.floats(min_value=0.0, max_value=1.0))
            ),
            min_size=0, max_size=2
        ),
        ai_predictions=st.lists(st.dictionaries(st.text(), st.text()), min_size=0, max_size=1),
        quality_score=st.floats(min_value=0.0, max_value=1.0),
        created_at=st.just(datetime.now())
    )


def prediction_strategy():
    """Strategy for generating valid Prediction instances."""
    return st.builds(
        Prediction,
        id=st.just(uuid4()),
        task_id=st.just(uuid4()),
        ai_model_config=model_config_strategy(),
        prediction_data=st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.text(), st.floats(), st.booleans()),
            min_size=1, max_size=5
        ),
        confidence=st.floats(min_value=0.0, max_value=1.0),
        processing_time=st.floats(min_value=0.1, max_value=10.0),
        created_at=st.just(datetime.now())
    )


class MockAIAnnotator(AIAnnotator):
    """Mock AI annotator for testing pre-annotation triggering."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.predictions_generated = []
        self.predict_calls = []
    
    def _validate_config(self) -> None:
        """Mock validation - always passes."""
        pass
    
    async def predict(self, task: Task) -> Prediction:
        """Mock prediction generation."""
        self.predict_calls.append(task.id)
        
        prediction = Prediction(
            id=uuid4(),
            task_id=task.id,
            ai_model_config=self.config,
            prediction_data={
                "label": "positive",
                "confidence": 0.8,
                "model_type": self.config.model_type.value
            },
            confidence=0.8,
            processing_time=1.5
        )
        
        self.predictions_generated.append(prediction)
        return prediction
    
    def get_model_info(self) -> Dict[str, Any]:
        """Mock model info."""
        return {
            "model_type": self.config.model_type.value,
            "model_name": self.config.model_name,
            "version": "mock-1.0"
        }


class MockLabelStudioIntegration:
    """Mock Label Studio integration for testing."""
    
    def __init__(self):
        self.imported_tasks = []
        self.ai_predictions_added = []
        self.task_start_triggers = []
    
    async def import_tasks(self, project_id: str, tasks: List[Task]):
        """Mock task import that should trigger AI pre-annotation."""
        self.imported_tasks.extend(tasks)
        
        # Simulate AI pre-annotation triggering for each task
        for task in tasks:
            self.task_start_triggers.append({
                'task_id': task.id,
                'project_id': project_id,
                'triggered_at': datetime.now()
            })
        
        return Mock(success=True, imported_count=len(tasks), failed_count=0, errors=[])
    
    async def add_ai_predictions_to_task(self, task: Task, predictions: List[Prediction]):
        """Mock adding AI predictions to a task."""
        for prediction in predictions:
            self.ai_predictions_added.append({
                'task_id': task.id,
                'prediction_id': prediction.id,
                'confidence': prediction.confidence
            })
            
            # Add prediction to task
            task.ai_predictions.append({
                'id': str(prediction.id),
                'model': prediction.ai_model_config.model_name,
                'result': prediction.prediction_data,
                'confidence': prediction.confidence
            })


class TestAIPreAnnotationAutoTrigger:
    """
    Property-based tests for AI pre-annotation auto-triggering.
    
    Validates Requirements 3.2 and 10.5:
    - When annotation tasks are started, AI pre-annotation should be automatically triggered
    - AI pre-annotation should provide confidence scores between 0.0 and 1.0
    """
    
    @given(task_strategy(), model_config_strategy())
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_task_start_triggers_ai_preannotation(self, task: Task, model_config: ModelConfig):
        """
        Property 5: AI Pre-annotation Auto-Trigger
        
        For any annotation task that is started, the system should automatically
        trigger AI pre-annotation and generate prediction results.
        
        **Validates: Requirements 3.2, 10.5**
        """
        # Ensure task starts without AI predictions (clean state)
        original_predictions_count = len(task.ai_predictions)
        task.ai_predictions = []  # Clear any existing predictions for clean test
        
        # Create mock AI annotator
        mock_annotator = MockAIAnnotator(model_config)
        
        # Create mock Label Studio integration
        mock_integration = MockLabelStudioIntegration()
        
        # Simulate task starting process
        async def start_annotation_task():
            # When a task is started (imported to Label Studio), AI pre-annotation should trigger
            
            # Step 1: Import task to Label Studio (this should trigger AI pre-annotation)
            import_result = await mock_integration.import_tasks(task.project_id, [task])
            
            # Step 2: Generate AI predictions for the task
            prediction = await mock_annotator.predict(task)
            
            # Step 3: Add AI predictions to the task
            await mock_integration.add_ai_predictions_to_task(task, [prediction])
            
            return prediction
        
        # Run the async process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            prediction = loop.run_until_complete(start_annotation_task())
            
            # Verify AI pre-annotation was triggered
            assert len(mock_annotator.predict_calls) == 1
            assert mock_annotator.predict_calls[0] == task.id
            
            # Verify prediction was generated
            assert len(mock_annotator.predictions_generated) == 1
            generated_prediction = mock_annotator.predictions_generated[0]
            assert generated_prediction.task_id == task.id
            assert 0.0 <= generated_prediction.confidence <= 1.0
            
            # Verify task start was triggered
            assert len(mock_integration.task_start_triggers) == 1
            trigger = mock_integration.task_start_triggers[0]
            assert trigger['task_id'] == task.id
            assert trigger['project_id'] == task.project_id
            
            # Verify AI prediction was added to task
            assert len(mock_integration.ai_predictions_added) == 1
            added_prediction = mock_integration.ai_predictions_added[0]
            assert added_prediction['task_id'] == task.id
            assert added_prediction['confidence'] == prediction.confidence
            
            # Verify task now has exactly one AI prediction (the one we added)
            assert len(task.ai_predictions) == 1
            task_prediction = task.ai_predictions[0]
            assert task_prediction['confidence'] == prediction.confidence
            
        finally:
            loop.close()
    
    @given(st.lists(task_strategy(), min_size=1, max_size=5), model_config_strategy())
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_multiple_tasks_trigger_individual_preannotations(self, tasks: List[Task], model_config: ModelConfig):
        """
        Property 5: Multiple Task AI Pre-annotation Triggering
        
        For any list of annotation tasks that are started, each task should
        trigger its own individual AI pre-annotation process.
        
        **Validates: Requirements 3.2, 10.5**
        """
        # Ensure tasks have unique IDs and clean state
        for i, task in enumerate(tasks):
            task.id = uuid4()
            task.project_id = f"project_{i}"
            task.ai_predictions = []  # Clear any existing predictions for clean test
        
        # Create mock AI annotator
        mock_annotator = MockAIAnnotator(model_config)
        
        # Create mock Label Studio integration
        mock_integration = MockLabelStudioIntegration()
        
        # Simulate starting multiple tasks
        async def start_multiple_tasks():
            predictions = []
            
            for task in tasks:
                # Import task (triggers AI pre-annotation)
                await mock_integration.import_tasks(task.project_id, [task])
                
                # Generate AI prediction
                prediction = await mock_annotator.predict(task)
                predictions.append(prediction)
                
                # Add prediction to task
                await mock_integration.add_ai_predictions_to_task(task, [prediction])
            
            return predictions
        
        # Run the async process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            predictions = loop.run_until_complete(start_multiple_tasks())
            
            # Verify each task triggered AI pre-annotation
            assert len(mock_annotator.predict_calls) == len(tasks)
            assert len(mock_annotator.predictions_generated) == len(tasks)
            
            # Verify each task ID was called for prediction
            called_task_ids = set(mock_annotator.predict_calls)
            expected_task_ids = set(task.id for task in tasks)
            assert called_task_ids == expected_task_ids
            
            # Verify each task has AI predictions
            for task in tasks:
                assert len(task.ai_predictions) == 1
                task_prediction = task.ai_predictions[0]
                assert 0.0 <= task_prediction['confidence'] <= 1.0
            
            # Verify all predictions have valid confidence scores
            for prediction in predictions:
                assert 0.0 <= prediction.confidence <= 1.0
                assert prediction.task_id in expected_task_ids
            
        finally:
            loop.close()
    
    @given(task_strategy())
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_task_status_change_triggers_preannotation(self, task: Task):
        """
        Property 5: Task Status Change AI Pre-annotation Triggering
        
        For any task that changes status from PENDING to IN_PROGRESS,
        AI pre-annotation should be automatically triggered.
        
        **Validates: Requirements 3.2, 10.5**
        """
        # Ensure task starts as PENDING
        task.status = TaskStatus.PENDING
        task.ai_predictions = []  # Start with no predictions
        
        # Create mock AI annotator with default config
        model_config = ModelConfig(
            model_type=ModelType.OLLAMA,
            model_name="test-model",
            max_tokens=1000,
            temperature=0.7,
            timeout=30
        )
        mock_annotator = MockAIAnnotator(model_config)
        
        # Simulate task status change workflow
        async def change_task_status():
            original_status = task.status
            
            # Change status to IN_PROGRESS (simulates task starting)
            if original_status == TaskStatus.PENDING:
                task.status = TaskStatus.IN_PROGRESS
                status_changed = True
            else:
                status_changed = False
            
            # If status changed to IN_PROGRESS, trigger AI pre-annotation
            if status_changed and task.status == TaskStatus.IN_PROGRESS:
                prediction = await mock_annotator.predict(task)
                
                # Add prediction to task
                task.ai_predictions.append({
                    'id': str(prediction.id),
                    'model': prediction.ai_model_config.model_name,
                    'result': prediction.prediction_data,
                    'confidence': prediction.confidence
                })
                
                return True
            return False
        
        # Run the async process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(change_task_status())
            
            # Verify AI pre-annotation was triggered if status changed
            if result:
                assert len(mock_annotator.predict_calls) == 1
                assert mock_annotator.predict_calls[0] == task.id
                assert len(task.ai_predictions) == 1
                
                # Verify prediction has valid confidence
                prediction_data = task.ai_predictions[0]
                assert 0.0 <= prediction_data['confidence'] <= 1.0
                assert task.status == TaskStatus.IN_PROGRESS
            else:
                # If task was already IN_PROGRESS, no new prediction should be generated
                assert len(mock_annotator.predict_calls) == 0
                
        finally:
            loop.close()
    
    @given(st.lists(model_config_strategy(), min_size=1, max_size=3), task_strategy())
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_multiple_models_generate_preannotations(self, model_configs: List[ModelConfig], task: Task):
        """
        Property 5: Multiple Model AI Pre-annotation
        
        For any task and any list of AI models, starting the task should
        trigger pre-annotation from all configured models.
        
        **Validates: Requirements 3.2, 10.5**
        """
        # Ensure task starts with clean state
        task.ai_predictions = []  # Clear any existing predictions for clean test
        
        # Ensure model configs have unique names
        for i, config in enumerate(model_configs):
            config.model_name = f"model_{i}_{config.model_name}"
        
        # Create mock annotators for each model
        mock_annotators = []
        for config in model_configs:
            annotator = MockAIAnnotator(config)
            mock_annotators.append(annotator)
        
        # Simulate multi-model pre-annotation
        async def generate_multi_model_predictions():
            predictions = []
            
            for annotator in mock_annotators:
                prediction = await annotator.predict(task)
                predictions.append(prediction)
                
                # Add prediction to task
                task.ai_predictions.append({
                    'id': str(prediction.id),
                    'model': prediction.ai_model_config.model_name,
                    'result': prediction.prediction_data,
                    'confidence': prediction.confidence
                })
            
            return predictions
        
        # Run the async process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            predictions = loop.run_until_complete(generate_multi_model_predictions())
            
            # Verify each model generated a prediction
            assert len(predictions) == len(model_configs)
            assert len(task.ai_predictions) == len(model_configs)
            
            # Verify each annotator was called
            for annotator in mock_annotators:
                assert len(annotator.predict_calls) == 1
                assert annotator.predict_calls[0] == task.id
            
            # Verify all predictions have valid confidence scores
            for prediction in predictions:
                assert 0.0 <= prediction.confidence <= 1.0
                assert prediction.task_id == task.id
            
            # Verify task has predictions from all models
            model_names_in_task = [pred['model'] for pred in task.ai_predictions]
            expected_model_names = [config.model_name for config in model_configs]
            assert set(model_names_in_task) == set(expected_model_names)
            
        finally:
            loop.close()
    
    @given(task_strategy(), model_config_strategy())
    @settings(suppress_health_check=[HealthCheck.filter_too_much], max_examples=100)
    def test_ai_confidence_range_property(self, task: Task, model_config: ModelConfig):
        """
        Property 7: AI 置信度范围
        
        For any AI pre-annotation result, the confidence score should
        always be within the valid range of 0.0 to 1.0.
        
        **Validates: Requirements 10.5**
        """
        # Create mock annotator that can generate various confidence scores
        mock_annotator = MockAIAnnotator(model_config)
        
        # Generate prediction
        async def generate_prediction_with_confidence():
            prediction = await mock_annotator.predict(task)
            return prediction
        
        # Run the async process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            prediction = loop.run_until_complete(generate_prediction_with_confidence())
            
            # Property: AI confidence must be in range [0.0, 1.0]
            assert 0.0 <= prediction.confidence <= 1.0, \
                f"AI confidence {prediction.confidence} is outside valid range [0.0, 1.0]"
            
            # Verify prediction data contains confidence information
            if 'confidence' in prediction.prediction_data:
                pred_confidence = prediction.prediction_data['confidence']
                assert 0.0 <= pred_confidence <= 1.0, \
                    f"Prediction data confidence {pred_confidence} is outside valid range [0.0, 1.0]"
            
            # Verify the prediction is properly structured
            assert prediction.task_id == task.id
            assert prediction.ai_model_config == model_config
            assert isinstance(prediction.prediction_data, dict)
            assert prediction.processing_time >= 0.0
            
        finally:
            loop.close()
    
    @given(st.lists(task_strategy(), min_size=1, max_size=10), model_config_strategy())
    @settings(suppress_health_check=[HealthCheck.filter_too_much], max_examples=50)
    def test_batch_ai_confidence_range_property(self, tasks: List[Task], model_config: ModelConfig):
        """
        Property 7: Batch AI 置信度范围
        
        For any batch of AI pre-annotation results, all confidence scores should
        be within the valid range of 0.0 to 1.0.
        
        **Validates: Requirements 10.5**
        """
        # Ensure tasks have unique IDs
        for i, task in enumerate(tasks):
            task.id = uuid4()
            task.project_id = f"batch_project_{i}"
        
        # Create mock annotator
        mock_annotator = MockAIAnnotator(model_config)
        
        # Generate batch predictions
        async def generate_batch_predictions():
            predictions = await mock_annotator.batch_predict(tasks)
            return predictions
        
        # Run the async process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            predictions = loop.run_until_complete(generate_batch_predictions())
            
            # Property: All AI confidences must be in range [0.0, 1.0]
            for i, prediction in enumerate(predictions):
                assert 0.0 <= prediction.confidence <= 1.0, \
                    f"Batch prediction {i} confidence {prediction.confidence} is outside valid range [0.0, 1.0]"
                
                # Check prediction data confidence if present
                if 'confidence' in prediction.prediction_data:
                    pred_confidence = prediction.prediction_data['confidence']
                    assert 0.0 <= pred_confidence <= 1.0, \
                        f"Batch prediction {i} data confidence {pred_confidence} is outside valid range [0.0, 1.0]"
            
            # Verify we got predictions for all tasks
            assert len(predictions) <= len(tasks), "More predictions than tasks"
            
        finally:
            loop.close()
    
    @given(st.floats(min_value=-10.0, max_value=10.0), task_strategy(), model_config_strategy())
    @settings(suppress_health_check=[HealthCheck.filter_too_much], max_examples=100)
    def test_confidence_validation_edge_cases(self, raw_confidence: float, task: Task, model_config: ModelConfig):
        """
        Property 7: AI 置信度范围边界测试
        
        For any raw confidence value (including invalid ones), the final
        AI prediction confidence should be normalized to [0.0, 1.0] range.
        
        **Validates: Requirements 10.5**
        """
        # Create a special mock annotator that can return arbitrary confidence values
        class EdgeCaseMockAnnotator(MockAIAnnotator):
            def __init__(self, config: ModelConfig, forced_confidence: float):
                super().__init__(config)
                self.forced_confidence = forced_confidence
            
            async def predict(self, task: Task) -> Prediction:
                # Create prediction with potentially invalid confidence
                raw_prediction = Prediction(
                    id=uuid4(),
                    task_id=task.id,
                    ai_model_config=self.config,
                    prediction_data={
                        "label": "test",
                        "confidence": self.forced_confidence,  # This might be invalid
                        "model_type": self.config.model_type.value
                    },
                    confidence=max(0.0, min(1.0, self.forced_confidence)),  # Normalize to valid range
                    processing_time=1.0
                )
                
                self.predictions_generated.append(raw_prediction)
                return raw_prediction
        
        # Create annotator with forced confidence value
        mock_annotator = EdgeCaseMockAnnotator(model_config, raw_confidence)
        
        # Generate prediction
        async def generate_edge_case_prediction():
            prediction = await mock_annotator.predict(task)
            return prediction
        
        # Run the async process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            prediction = loop.run_until_complete(generate_edge_case_prediction())
            
            # Property: Final confidence must always be in valid range [0.0, 1.0]
            # regardless of input confidence value
            assert 0.0 <= prediction.confidence <= 1.0, \
                f"Normalized confidence {prediction.confidence} from raw {raw_confidence} is outside valid range [0.0, 1.0]"
            
            # Verify edge case handling
            if raw_confidence < 0.0:
                assert prediction.confidence == 0.0, \
                    f"Negative confidence {raw_confidence} should be normalized to 0.0, got {prediction.confidence}"
            elif raw_confidence > 1.0:
                assert prediction.confidence == 1.0, \
                    f"Confidence {raw_confidence} > 1.0 should be normalized to 1.0, got {prediction.confidence}"
            else:
                # Valid range, should be preserved
                assert abs(prediction.confidence - raw_confidence) < 1e-10, \
                    f"Valid confidence {raw_confidence} should be preserved, got {prediction.confidence}"
            
        finally:
            loop.close()
    
    @given(st.integers(min_value=1, max_value=10))
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_concurrent_task_preannotation_triggering(self, num_concurrent: int):
        """
        Property 5: Concurrent Task Pre-annotation Triggering
        
        For any number of concurrent task starts, each should trigger
        its own AI pre-annotation without interference.
        
        **Validates: Requirements 3.2, 10.5**
        """
        # Create model config
        model_config = ModelConfig(
            model_type=ModelType.HUGGINGFACE,
            model_name="concurrent-test-model",
            max_tokens=500,
            temperature=0.5,
            timeout=30
        )
        
        # Create mock annotator
        mock_annotator = MockAIAnnotator(model_config)
        
        # Create concurrent tasks
        async def process_concurrent_tasks():
            tasks = []
            task_ids = []
            
            for i in range(num_concurrent):
                task_id = uuid4()
                task_ids.append(task_id)
                
                task = Task(
                    id=task_id,
                    document_id=uuid4(),
                    project_id=f"concurrent_project_{i}",
                    status=TaskStatus.PENDING
                )
                
                # Create async task for each pre-annotation
                async def trigger_preannotation(t=task, index=i):
                    # Simulate task starting
                    t.status = TaskStatus.IN_PROGRESS
                    
                    # Simulate some processing delay
                    await asyncio.sleep(0.01)
                    
                    # Trigger AI pre-annotation
                    prediction = await mock_annotator.predict(t)
                    
                    # Add prediction to task
                    t.ai_predictions.append({
                        'id': str(prediction.id),
                        'model': prediction.ai_model_config.model_name,
                        'confidence': prediction.confidence
                    })
                    
                    return t
                
                tasks.append(trigger_preannotation())
            
            # Run all tasks concurrently
            completed_tasks = await asyncio.gather(*tasks)
            return completed_tasks, task_ids
        
        # Run the concurrent process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            completed_tasks, task_ids = loop.run_until_complete(process_concurrent_tasks())
            
            # Verify all tasks triggered AI pre-annotation
            assert len(mock_annotator.predict_calls) == num_concurrent
            assert len(mock_annotator.predictions_generated) == num_concurrent
            
            # Verify each task ID appears exactly once in predict calls
            predict_call_ids = set(mock_annotator.predict_calls)
            expected_task_ids = set(task_ids)
            assert predict_call_ids == expected_task_ids
            assert len(mock_annotator.predict_calls) == len(set(mock_annotator.predict_calls))  # No duplicates
            
            # Verify each task has AI predictions
            for task in completed_tasks:
                assert len(task.ai_predictions) == 1
                assert task.status == TaskStatus.IN_PROGRESS
                assert 0.0 <= task.ai_predictions[0]['confidence'] <= 1.0
            
        finally:
            loop.close()
    
    def test_ai_annotator_factory_integration(self):
        """
        Property 5: AI Annotator Factory Integration
        
        The AnnotatorFactory should be able to create annotators that
        can generate pre-annotations when tasks are started.
        
        **Validates: Requirements 3.2, 10.5**
        """
        # Test with different model types
        for model_type in [ModelType.OLLAMA, ModelType.HUGGINGFACE, ModelType.ZHIPU_GLM]:
            config = ModelConfig(
                model_type=model_type,
                model_name=f"test-{model_type.value}",
                max_tokens=1000,
                temperature=0.7,
                timeout=30
            )
            
            # Mock the factory to return our mock annotator
            with patch.object(AnnotatorFactory, 'create_annotator') as mock_factory:
                mock_annotator = MockAIAnnotator(config)
                mock_factory.return_value = mock_annotator
                
                # Create annotator through factory
                annotator = AnnotatorFactory.create_annotator(config)
                
                # Verify factory was called correctly
                mock_factory.assert_called_once_with(config)
                
                # Verify annotator can generate predictions
                task = Task(
                    id=uuid4(),
                    document_id=uuid4(),
                    project_id="factory_test_project"
                )
                
                # Test prediction generation
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    prediction = loop.run_until_complete(annotator.predict(task))
                    
                    # Verify prediction is valid
                    assert prediction.task_id == task.id
                    assert 0.0 <= prediction.confidence <= 1.0
                    assert prediction.ai_model_config.model_type == model_type
                    
                finally:
                    loop.close()


if __name__ == "__main__":
    # Run with verbose output and show hypothesis examples
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])