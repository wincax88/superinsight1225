# AI Model Extensions Guide (Task 23.3)

## Overview

This document describes the AI model integration extensions implemented for the SuperInsight platform, including new model integrations, performance analysis, and automatic model selection capabilities.

## New Features Implemented

### 1. 阿里云通义千问 API 集成

**File**: `src/ai/alibaba_annotator.py`

**Features**:
- Full integration with Alibaba Cloud Tongyi Qianwen API
- Support for multiple Qwen models (qwen-turbo, qwen-plus, qwen-max, etc.)
- Automatic JSON response parsing
- Error handling and retry mechanisms
- Health check capabilities

**Supported Models**:
- qwen-turbo
- qwen-plus  
- qwen-max
- qwen-max-1201
- qwen-max-longcontext
- qwen1.5-72b-chat
- qwen1.5-14b-chat
- qwen1.5-7b-chat

**Usage Example**:
```python
from ai import ModelConfig, ModelType
from ai.alibaba_annotator import AlibabaAnnotator

config = ModelConfig(
    model_type=ModelType.ALIBABA_TONGYI,
    model_name="qwen-turbo",
    api_key="your_api_key",
    max_tokens=1000,
    temperature=0.7
)

annotator = AlibabaAnnotator(config)
prediction = await annotator.predict(task)
```

### 2. 开源模型集成扩展

**File**: `src/ai/chatglm_annotator.py`

**Features**:
- ChatGLM model family support (ChatGLM3, ChatGLM2, etc.)
- Local deployment and API endpoint support
- OpenAI-compatible API format support
- Flexible endpoint configuration

**Supported Models**:
- chatglm3-6b
- chatglm3-6b-32k
- chatglm3-6b-base
- chatglm2-6b
- chatglm2-6b-32k
- chatglm2-6b-base
- chatglm-6b

**Usage Example**:
```python
config = ModelConfig(
    model_type=ModelType.HUGGINGFACE,  # Uses HuggingFace type
    model_name="chatglm3-6b",
    base_url="http://localhost:8000",  # Local deployment
    max_tokens=1000,
    temperature=0.7
)

annotator = ChatGLMAnnotator(config)
```

### 3. 模型性能对比分析

**File**: `src/ai/model_performance.py`

**Features**:
- Comprehensive performance metrics tracking
- Model comparison and ranking
- Performance trend analysis
- Benchmark testing capabilities

**Performance Metrics**:
- Accuracy
- Precision/Recall/F1-Score
- Confidence scores
- Response time
- Throughput
- Error rate
- Cost efficiency

**Key Classes**:

#### ModelPerformanceAnalyzer
```python
analyzer = ModelPerformanceAnalyzer("./performance_data")

# Record prediction results
analyzer.record_prediction(prediction, task_type, is_correct=True)

# Compare models
comparison = analyzer.compare_models("sentiment", PerformanceMetric.ACCURACY)

# Get performance summary
summary = analyzer.get_performance_summary("sentiment")

# Export report
report = analyzer.export_performance_report("sentiment")
```

#### ModelPerformanceData
- Tracks individual model performance
- Calculates overall performance scores
- Maintains historical performance data
- Supports weighted metric calculations

### 4. 模型自动选择策略

**File**: `src/ai/model_performance.py` (ModelAutoSelector class)

**Features**:
- Intelligent model selection based on requirements
- Multi-criteria decision making
- Performance-based recommendations
- Requirement constraint handling

**Selection Criteria**:
- Task type compatibility
- Performance requirements (accuracy, speed)
- Resource constraints (cost, latency)
- Model availability and health
- Historical performance data

**Usage Example**:
```python
auto_selector = ModelAutoSelector(performance_analyzer)

# Define requirements
requirements = {
    "max_response_time": 2.0,
    "min_accuracy": 0.85,
    "preferred_models": [ModelType.ALIBABA_TONGYI, ModelType.ZHIPU_GLM],
    "min_samples": 10
}

# Auto-select best model
selected_config = auto_selector.select_best_model("sentiment", requirements)

# Get recommendations
recommendations = auto_selector.get_model_recommendations("sentiment", top_n=3)
```

### 5. 增强模型管理器

**File**: `src/ai/enhanced_model_manager.py`

**Features**:
- Unified model lifecycle management
- Performance tracking integration
- Auto-selection capabilities
- Health monitoring and statistics
- Model version management
- Cleanup and optimization tools

**Key Capabilities**:

#### Model Registration and Management
```python
manager = EnhancedModelManager("./model_data")

# Register new model
version_id = await manager.register_model(
    name="qwen_plus",
    config=model_config,
    description="Alibaba Qwen Plus model",
    tags={"alibaba", "production"},
    auto_activate=True
)

# Get annotator with auto-creation
annotator = await manager.get_annotator("qwen_plus")
```

#### Performance Tracking
```python
# Make prediction with performance tracking
prediction = await manager.predict_with_performance_tracking(
    annotator_name="qwen_plus",
    task=annotation_task,
    task_type="sentiment",
    ground_truth=expected_result
)
```

#### Auto-Selection and Optimization
```python
# Auto-select best model
best_model = await manager.auto_select_model("classification", requirements)

# Optimize model selection
optimization = await manager.optimize_model_selection(
    "sentiment", 
    PerformanceMetric.ACCURACY
)

# Get recommendations
recommendations = await manager.get_model_recommendations("sentiment")
```

#### Health Monitoring
```python
# Health check all models
health_status = await manager.health_check_all_models()

# Get comprehensive statistics
stats = await manager.get_model_statistics()

# Cleanup inactive models
cleanup_results = await manager.cleanup_inactive_models(days_threshold=30)
```

### 6. REST API 接口

**File**: `src/api/ai_models.py`

**Endpoints**:

#### Model Management
- `POST /api/ai-models/register` - Register new model
- `GET /api/ai-models/list` - List all models
- `GET /api/ai-models/health` - Health check
- `GET /api/ai-models/statistics` - Model statistics
- `POST /api/ai-models/cleanup` - Cleanup inactive models

#### Prediction and Performance
- `POST /api/ai-models/predict` - Make prediction with tracking
- `GET /api/ai-models/performance/compare/{task_type}` - Compare models
- `GET /api/ai-models/performance/report` - Performance report
- `POST /api/ai-models/benchmark` - Benchmark models

#### Auto-Selection
- `POST /api/ai-models/auto-select` - Auto-select model
- `GET /api/ai-models/recommendations/{task_type}` - Get recommendations
- `POST /api/ai-models/optimize/{task_type}` - Optimize selection

#### Utility
- `GET /api/ai-models/supported-types` - Supported model types

**API Usage Examples**:

```bash
# Register new model
curl -X POST "http://localhost:8000/api/ai-models/register" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "alibaba_tongyi",
    "model_name": "qwen-turbo",
    "api_key": "your_api_key",
    "description": "Alibaba Qwen Turbo model",
    "tags": ["alibaba", "fast"],
    "auto_activate": true
  }'

# Auto-select model
curl -X POST "http://localhost:8000/api/ai-models/auto-select" \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "sentiment",
    "max_response_time": 2.0,
    "min_accuracy": 0.85
  }'

# Get model recommendations
curl "http://localhost:8000/api/ai-models/recommendations/sentiment?top_n=3"

# Compare models
curl "http://localhost:8000/api/ai-models/performance/compare/sentiment?metric=accuracy"
```

## Integration with Existing System

### Factory Integration

The new annotators are integrated into the existing `AnnotatorFactory`:

```python
# Updated factory supports new model types
from ai.factory import AnnotatorFactory

# Alibaba Tongyi
config = ModelConfig(model_type=ModelType.ALIBABA_TONGYI, ...)
annotator = AnnotatorFactory.create_annotator(config)

# ChatGLM (auto-detected by model name)
config = ModelConfig(model_type=ModelType.HUGGINGFACE, model_name="chatglm3-6b", ...)
annotator = AnnotatorFactory.create_annotator(config)  # Returns ChatGLMAnnotator
```

### Model Type Enumeration

Updated `ModelType` enum in `src/ai/base.py`:
```python
class ModelType(str, Enum):
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    ZHIPU_GLM = "zhipu_glm"
    BAIDU_WENXIN = "baidu_wenxin"
    ALIBABA_TONGYI = "alibaba_tongyi"  # New
    TENCENT_HUNYUAN = "tencent_hunyuan"
```

### Module Exports

Updated `src/ai/__init__.py` to export new components:
```python
from .model_performance import (
    ModelPerformanceAnalyzer,
    ModelAutoSelector,
    PerformanceMetric,
    ModelPerformanceData
)

from .enhanced_model_manager import EnhancedModelManager
from .alibaba_annotator import AlibabaAnnotator
from .chatglm_annotator import ChatGLMAnnotator
```

## Configuration Examples

### Environment Variables

```bash
# Alibaba Tongyi
export ALIBABA_API_KEY="your_tongyi_api_key"
export ALIBABA_BASE_URL="https://dashscope.aliyuncs.com/api/v1"

# ChatGLM Local Deployment
export CHATGLM_BASE_URL="http://localhost:8000"
export CHATGLM_MODEL="chatglm3-6b"
```

### Configuration Files

**Model Configuration JSON**:
```json
{
  "alibaba_qwen_turbo": {
    "model_type": "alibaba_tongyi",
    "model_name": "qwen-turbo",
    "api_key": "${ALIBABA_API_KEY}",
    "max_tokens": 1000,
    "temperature": 0.7,
    "timeout": 30
  },
  "local_chatglm3": {
    "model_type": "huggingface",
    "model_name": "chatglm3-6b",
    "base_url": "http://localhost:8000",
    "max_tokens": 2000,
    "temperature": 0.8,
    "timeout": 60
  }
}
```

## Performance Optimization

### Caching Strategy

The system includes intelligent caching for:
- Model performance data
- Prediction results
- Model health status
- Configuration data

### Batch Processing

Support for batch predictions with performance tracking:
```python
# Batch prediction with performance tracking
predictions = []
for task in tasks:
    prediction = await manager.predict_with_performance_tracking(
        annotator_name="best_model",
        task=task,
        task_type="classification"
    )
    predictions.append(prediction)
```

### Auto-Optimization

The system can automatically optimize model selection:
```python
# Periodic optimization
optimization_results = await manager.optimize_model_selection(
    "sentiment",
    PerformanceMetric.ACCURACY
)

if optimization_results["status"] == "success":
    best_model = optimization_results["best_model"]
    # Switch to best performing model
```

## Monitoring and Observability

### Performance Metrics

The system tracks comprehensive metrics:
- **Accuracy**: Prediction correctness rate
- **Response Time**: Average prediction latency
- **Throughput**: Predictions per second
- **Error Rate**: Failure percentage
- **Confidence**: Average confidence scores
- **Cost Efficiency**: Performance per unit cost

### Health Monitoring

Continuous health monitoring includes:
- Model availability checks
- API endpoint health
- Performance degradation detection
- Resource usage monitoring

### Reporting

Automated reporting capabilities:
- Performance trend reports
- Model comparison reports
- Optimization recommendations
- Health status summaries

## Best Practices

### Model Selection

1. **Start with Auto-Selection**: Use the auto-selector for initial model choice
2. **Monitor Performance**: Continuously track model performance metrics
3. **Regular Optimization**: Periodically optimize model selection
4. **A/B Testing**: Compare models on real workloads

### Performance Tuning

1. **Baseline Establishment**: Establish performance baselines for each task type
2. **Threshold Setting**: Set appropriate performance thresholds
3. **Regular Benchmarking**: Run regular benchmark tests
4. **Gradual Rollout**: Gradually roll out new models

### Maintenance

1. **Regular Cleanup**: Clean up inactive models periodically
2. **Health Monitoring**: Monitor model health continuously
3. **Performance Review**: Review performance metrics regularly
4. **Configuration Updates**: Keep model configurations up to date

## Troubleshooting

### Common Issues

1. **API Key Issues**:
   - Verify API keys are correctly configured
   - Check API key permissions and quotas
   - Ensure proper environment variable setup

2. **Model Availability**:
   - Check model endpoint health
   - Verify network connectivity
   - Confirm model deployment status

3. **Performance Issues**:
   - Monitor response times and throughput
   - Check for resource constraints
   - Review model configuration parameters

4. **Selection Issues**:
   - Ensure sufficient performance data
   - Check selection criteria and requirements
   - Verify model compatibility with task types

### Debugging Tools

1. **Health Check API**: Use health check endpoints to verify model status
2. **Performance Reports**: Generate detailed performance reports
3. **Statistics API**: Get comprehensive system statistics
4. **Logging**: Enable detailed logging for troubleshooting

## Future Enhancements

### Planned Features

1. **More Model Integrations**:
   - Additional Chinese LLM providers
   - More open-source model support
   - Custom model deployment support

2. **Advanced Analytics**:
   - Predictive performance modeling
   - Cost optimization algorithms
   - Advanced benchmarking suites

3. **Enhanced Auto-Selection**:
   - Machine learning-based selection
   - Multi-objective optimization
   - Dynamic model switching

4. **Improved Monitoring**:
   - Real-time performance dashboards
   - Alerting and notification systems
   - Advanced anomaly detection

## Conclusion

The AI Model Extensions (Task 23.3) successfully implement:

✅ **阿里云通义千问 API 集成** - Complete integration with Alibaba Tongyi Qianwen
✅ **更多开源模型集成** - ChatGLM and other open-source model support  
✅ **模型性能对比分析** - Comprehensive performance analysis and comparison
✅ **模型自动选择策略** - Intelligent auto-selection based on requirements

The implementation provides a robust, scalable foundation for AI model management with advanced performance tracking and intelligent selection capabilities. The system is production-ready and integrates seamlessly with the existing SuperInsight platform architecture.