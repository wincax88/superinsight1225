#!/usr/bin/env python3
"""
Test script for AI Model Extensions (Task 23.3).

Tests the new AI model integrations, performance analysis, and auto-selection features.
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from uuid import uuid4

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai import (
    ModelConfig, ModelType, EnhancedModelManager,
    ModelPerformanceAnalyzer, ModelAutoSelector, PerformanceMetric
)


async def test_alibaba_annotator():
    """Test Alibaba Tongyi annotator."""
    print("Testing Alibaba Tongyi Annotator...")
    
    try:
        from ai.alibaba_annotator import AlibabaAnnotator
        
        config = ModelConfig(
            model_type=ModelType.ALIBABA_TONGYI,
            model_name="qwen-turbo",
            api_key="test_key",  # Mock key for testing
            max_tokens=100,
            temperature=0.7
        )
        
        annotator = AlibabaAnnotator(config)
        model_info = annotator.get_model_info()
        
        print(f"✓ Alibaba annotator created successfully")
        print(f"  Model info: {model_info}")
        
        # Test available models
        models = await annotator.list_available_models()
        print(f"  Available models: {models}")
        
        return True
        
    except Exception as e:
        print(f"✗ Alibaba annotator test failed: {e}")
        return False


async def test_chatglm_annotator():
    """Test ChatGLM annotator."""
    print("Testing ChatGLM Annotator...")
    
    try:
        from ai.chatglm_annotator import ChatGLMAnnotator
        
        config = ModelConfig(
            model_type=ModelType.HUGGINGFACE,  # ChatGLM uses HuggingFace type
            model_name="chatglm3-6b",
            base_url="http://localhost:8000",
            max_tokens=100,
            temperature=0.7
        )
        
        annotator = ChatGLMAnnotator(config)
        model_info = annotator.get_model_info()
        
        print(f"✓ ChatGLM annotator created successfully")
        print(f"  Model info: {model_info}")
        
        # Test available models
        models = await annotator.list_available_models()
        print(f"  Available models: {models}")
        
        return True
        
    except Exception as e:
        print(f"✗ ChatGLM annotator test failed: {e}")
        return False


async def test_performance_analyzer():
    """Test model performance analyzer."""
    print("Testing Model Performance Analyzer...")
    
    try:
        analyzer = ModelPerformanceAnalyzer("./test_performance")
        
        # Create mock prediction data
        from ai.base import Prediction
        from datetime import datetime
        
        mock_prediction = Prediction(
            id=uuid4(),
            task_id=uuid4(),
            ai_model_config=ModelConfig(
                model_type=ModelType.ZHIPU_GLM,
                model_name="glm-4"
            ),
            prediction_data={"sentiment": "positive", "confidence": 0.85},
            confidence=0.85,
            processing_time=1.2,
            created_at=datetime.now()
        )
        
        # Record prediction
        analyzer.record_prediction(mock_prediction, "sentiment", is_correct=True)
        
        # Get performance data
        perf_data = analyzer.get_model_performance("glm-4", ModelType.ZHIPU_GLM, "sentiment")
        
        if perf_data:
            print(f"✓ Performance data recorded successfully")
            print(f"  Sample count: {perf_data.sample_count}")
            print(f"  Metrics: {perf_data.metrics}")
            print(f"  Overall score: {perf_data.get_overall_score():.3f}")
        else:
            print("✗ No performance data found")
            return False
        
        # Test model comparison
        comparison = analyzer.compare_models("sentiment", PerformanceMetric.ACCURACY)
        print(f"  Model comparison: {comparison}")
        
        # Test performance summary
        summary = analyzer.get_performance_summary("sentiment")
        print(f"  Performance summary: {summary['total_models']} models, {summary['total_predictions']} predictions")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance analyzer test failed: {e}")
        return False


async def test_auto_selector():
    """Test model auto-selector."""
    print("Testing Model Auto-Selector...")
    
    try:
        analyzer = ModelPerformanceAnalyzer("./test_performance")
        auto_selector = ModelAutoSelector(analyzer)
        
        # Add some mock performance data first
        from ai.model_performance import ModelPerformanceData
        
        # Mock performance data for different models
        perf_data_1 = ModelPerformanceData(
            model_name="glm-4",
            model_type=ModelType.ZHIPU_GLM,
            task_type="sentiment",
            sample_count=50
        )
        perf_data_1.metrics[PerformanceMetric.ACCURACY] = 0.85
        perf_data_1.metrics[PerformanceMetric.RESPONSE_TIME] = 1.2
        perf_data_1.metrics[PerformanceMetric.CONFIDENCE] = 0.82
        
        perf_data_2 = ModelPerformanceData(
            model_name="qwen-turbo",
            model_type=ModelType.ALIBABA_TONGYI,
            task_type="sentiment",
            sample_count=30
        )
        perf_data_2.metrics[PerformanceMetric.ACCURACY] = 0.88
        perf_data_2.metrics[PerformanceMetric.RESPONSE_TIME] = 0.8
        perf_data_2.metrics[PerformanceMetric.CONFIDENCE] = 0.86
        
        # Store in analyzer
        analyzer.performance_data["zhipu_glm:glm-4:sentiment"] = perf_data_1
        analyzer.performance_data["alibaba_tongyi:qwen-turbo:sentiment"] = perf_data_2
        
        # Test auto selection
        selected_config = auto_selector.select_best_model("sentiment")
        
        if selected_config:
            print(f"✓ Auto-selected model: {selected_config.model_type.value}:{selected_config.model_name}")
        else:
            print("✗ No model selected")
            return False
        
        # Test recommendations
        recommendations = auto_selector.get_model_recommendations("sentiment", top_n=2)
        print(f"  Recommendations: {len(recommendations)} models")
        for i, rec in enumerate(recommendations, 1):
            print(f"    {i}. {rec['model_type']}:{rec['model_name']} (score: {rec['overall_score']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"✗ Auto-selector test failed: {e}")
        return False


async def test_integration_service():
    """Test AI model integration service."""
    print("Testing AI Model Integration Service...")
    
    try:
        from ai.integration_service import get_integration_service
        
        service = get_integration_service("./test_integration_service")
        
        # Test service status
        status = await service.get_integration_status()
        print(f"✓ Integration service status: {status['status']}")
        print(f"  Service components: {list(status['service_info']['components'].keys())}")
        
        # Test model registration with credentials
        try:
            version_id = await service.register_model_with_credentials(
                ModelType.ALIBABA_TONGYI,
                "qwen-test",
                "mock_api_key",
                max_tokens=500,
                temperature=0.8
            )
            print(f"✓ Model registered with version ID: {version_id}")
        except Exception as e:
            print(f"  Model registration failed (expected with mock credentials): {e}")
        
        # Test health check
        health_status = await service.health_check_all_models()
        print(f"✓ Health check completed")
        print(f"  Total models: {health_status['summary']['total_models']}")
        print(f"  Health percentage: {health_status['summary']['health_percentage']:.1f}%")
        
        # Test performance report
        report = await service.get_model_performance_report()
        print(f"✓ Performance report generated ({len(report)} characters)")
        
        # Test model comparison
        comparison_result = await service.compare_models_for_task("sentiment")
        if comparison_result.get("success"):
            print(f"✓ Model comparison completed")
        else:
            print(f"  Model comparison: {comparison_result.get('error', 'No models to compare')}")
        
        # Test optimization
        optimization_result = await service.optimize_model_selection("sentiment")
        print(f"✓ Model optimization completed: {optimization_result.get('success', False)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration service test failed: {e}")
        return False


async def test_model_comparison_suite():
    """Test model comparison and benchmarking suite."""
    print("Testing Model Comparison Suite...")
    
    try:
        from ai.model_comparison import ModelBenchmarkSuite, ModelAutoSelector
        
        benchmark_suite = ModelBenchmarkSuite("./test_benchmark_suite")
        
        # Test benchmark tasks
        print(f"✓ Benchmark suite initialized")
        print(f"  Available task types: {list(benchmark_suite.benchmark_tasks.keys())}")
        
        for task_type, tasks in benchmark_suite.benchmark_tasks.items():
            print(f"    {task_type}: {len(tasks)} benchmark tasks")
        
        # Test comparison history
        history = benchmark_suite.get_comparison_history()
        print(f"  Comparison history: {len(history)} entries")
        
        # Test report export
        report = benchmark_suite.export_comparison_report()
        print(f"✓ Comparison report generated ({len(report)} characters)")
        
        # Test auto selector with benchmark suite
        analyzer = ModelPerformanceAnalyzer("./test_benchmark_performance")
        auto_selector = ModelAutoSelector(benchmark_suite, analyzer)
        
        # Test with empty model list (should return None)
        selected = await auto_selector.select_optimal_model("sentiment", [])
        print(f"✓ Auto selector handled empty model list: {selected is None}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model comparison suite test failed: {e}")
        return False


async def test_hunyuan_annotator():
    """Test Tencent Hunyuan annotator."""
    print("Testing Tencent Hunyuan Annotator...")
    
    try:
        from ai.hunyuan_annotator import HunyuanAnnotator
        
        config = ModelConfig(
            model_type=ModelType.TENCENT_HUNYUAN,
            model_name="hunyuan-lite",
            api_key="test_secret_id",
            max_tokens=100,
            temperature=0.7
        )
        
        # Add secret key for Tencent
        config.secret_key = "test_secret_key"
        
        annotator = HunyuanAnnotator(config)
        model_info = annotator.get_model_info()
        
        print(f"✓ Hunyuan annotator created successfully")
        print(f"  Model info: {model_info}")
        
        # Test available models
        models = await annotator.list_available_models()
        print(f"  Available models: {models}")
        
        return True
        
    except Exception as e:
        print(f"✗ Hunyuan annotator test failed: {e}")
        return False


async def test_factory_integration():
    """Test factory integration with new annotators."""
    print("Testing Factory Integration...")
    
    try:
        from ai.factory import AnnotatorFactory
        
        # Test Alibaba annotator creation
        alibaba_config = ModelConfig(
            model_type=ModelType.ALIBABA_TONGYI,
            model_name="qwen-turbo",
            api_key="test_key"
        )
        
        alibaba_annotator = AnnotatorFactory.create_annotator(alibaba_config)
        print(f"✓ Alibaba annotator created via factory")
        
        # Test ChatGLM annotator creation (special model detection)
        chatglm_config = ModelConfig(
            model_type=ModelType.HUGGINGFACE,
            model_name="chatglm3-6b",
            base_url="http://localhost:8000"
        )
        
        chatglm_annotator = AnnotatorFactory.create_annotator(chatglm_config)
        print(f"✓ ChatGLM annotator created via factory (special model detection)")
        
        # Test supported model types
        supported_types = AnnotatorFactory.get_supported_model_types()
        print(f"  Supported model types: {[t.value for t in supported_types]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Factory integration test failed: {e}")
        return False


async def run_all_tests():
    """Run all tests for AI model extensions."""
    print("=" * 60)
    print("AI Model Extensions Test Suite (Task 23.3)")
    print("=" * 60)
    
    tests = [
        ("Alibaba Annotator", test_alibaba_annotator),
        ("Hunyuan Annotator", test_hunyuan_annotator),
        ("ChatGLM Annotator", test_chatglm_annotator),
        ("Performance Analyzer", test_performance_analyzer),
        ("Auto Selector", test_auto_selector),
        ("Model Comparison Suite", test_model_comparison_suite),
        ("Integration Service", test_integration_service),
        ("Factory Integration", test_factory_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! AI model extensions are working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    # Create test directories
    os.makedirs("./test_performance", exist_ok=True)
    os.makedirs("./test_enhanced_manager", exist_ok=True)
    os.makedirs("./test_integration_service", exist_ok=True)
    os.makedirs("./test_benchmark_suite", exist_ok=True)
    os.makedirs("./test_benchmark_performance", exist_ok=True)
    
    # Run tests
    success = asyncio.run(run_all_tests())
    
    # Cleanup test directories
    import shutil
    try:
        shutil.rmtree("./test_performance")
        shutil.rmtree("./test_enhanced_manager")
        shutil.rmtree("./test_integration_service")
        shutil.rmtree("./test_benchmark_suite")
        shutil.rmtree("./test_benchmark_performance")
    except:
        pass
    
    sys.exit(0 if success else 1)