# AI 预标注 API 详细文档

## 概述

AI 预标注 API 提供智能化的数据预标注服务，支持多种大语言模型（LLM）和机器学习模型，帮助提高标注效率和质量。

**基础路径**: `/api/v1/ai`

## 支持的模型

### 大语言模型 (LLM)
- **智谱 GLM**: GLM-4, GLM-3-turbo, ChatGLM3-6B
- **百度文心一言**: ERNIE-Bot, ERNIE-Bot-turbo, ERNIE-4.0
- **腾讯混元**: Hunyuan-lite, Hunyuan-standard, Hunyuan-pro
- **OpenAI**: GPT-3.5-turbo, GPT-4 (需要 API Key)

### 本地模型
- **Ollama**: Llama2, CodeLlama, Mistral, Qwen 等
- **HuggingFace**: BERT, DistilBERT, RoBERTa 等预训练模型

### 自定义模型
- 支持自定义模型接口集成
- 模型版本管理和切换

## 端点详情

### 1. 单个预测

#### POST `/api/v1/ai/predict`

为单个任务生成 AI 预标注结果。

**请求体**:
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "ai_model_config": {
    "model_type": "zhipu_glm",
    "model_name": "glm-4",
    "api_key": "your_api_key",
    "base_url": "https://open.bigmodel.cn/api/paas/v4/",
    "temperature": 0.7,
    "max_tokens": 2048,
    "prompt_template": "请对以下文本进行情感分析：{content}"
  },
  "content": "今天天气真好，心情很愉快！"
}
```

**参数说明**:
- `task_id` (必需): 任务唯一标识符
- `ai_model_config` (必需): 模型配置对象
  - `model_type`: 模型类型 (`zhipu_glm`, `baidu_wenxin`, `hunyuan`, `ollama`, `huggingface`)
  - `model_name`: 具体模型名称
  - `api_key`: API 密钥（云端模型需要）
  - `base_url`: API 基础 URL
  - `temperature`: 生成温度 (0.0-1.0)
  - `max_tokens`: 最大生成长度
  - `prompt_template`: 提示词模板
- `content` (可选): 待标注内容，如果不提供则从任务中获取

**响应示例**:
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "prediction_data": {
    "sentiment": "positive",
    "confidence": 0.95,
    "labels": ["积极", "愉快"],
    "reasoning": "文本表达了对天气的满意和愉快的心情"
  },
  "confidence": 0.95,
  "processing_time": 1.23,
  "model_info": {
    "model_type": "zhipu_glm",
    "model_name": "glm-4",
    "version": "1.0"
  },
  "cached": false
}
```

**使用示例**:

```bash
# 智谱 GLM 情感分析
curl -X POST "http://localhost:8000/api/v1/ai/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "ai_model_config": {
      "model_type": "zhipu_glm",
      "model_name": "glm-4",
      "api_key": "your_zhipu_api_key",
      "temperature": 0.3,
      "prompt_template": "请分析以下文本的情感倾向（积极/消极/中性）：{content}"
    },
    "content": "这个产品质量很差，完全不值这个价格。"
  }'

# Ollama 本地模型文本分类
curl -X POST "http://localhost:8000/api/v1/ai/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "550e8400-e29b-41d4-a716-446655440001",
    "ai_model_config": {
      "model_type": "ollama",
      "model_name": "llama2",
      "base_url": "http://localhost:11434",
      "temperature": 0.1,
      "prompt_template": "Classify the following text into categories: {content}"
    },
    "content": "The new iPhone has amazing camera quality and battery life."
  }'
```

```python
# Python 示例 - 多模型预测比较
import requests
import json

def compare_model_predictions(content, models_config):
    """比较多个模型的预测结果"""
    results = []
    
    for i, config in enumerate(models_config):
        request_data = {
            "task_id": f"compare_task_{i}",
            "ai_model_config": config,
            "content": content
        }
        
        try:
            response = requests.post(
                "http://localhost:8000/api/v1/ai/predict",
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                results.append({
                    "model": f"{config['model_type']}/{config['model_name']}",
                    "prediction": result["prediction_data"],
                    "confidence": result["confidence"],
                    "processing_time": result["processing_time"],
                    "cached": result["cached"]
                })
            else:
                results.append({
                    "model": f"{config['model_type']}/{config['model_name']}",
                    "error": response.text
                })
                
        except Exception as e:
            results.append({
                "model": f"{config['model_type']}/{config['model_name']}",
                "error": str(e)
            })
    
    return results

# 配置多个模型
models_config = [
    {
        "model_type": "zhipu_glm",
        "model_name": "glm-4",
        "api_key": "your_zhipu_key",
        "temperature": 0.3,
        "prompt_template": "情感分析：{content}"
    },
    {
        "model_type": "ollama",
        "model_name": "llama2",
        "base_url": "http://localhost:11434",
        "temperature": 0.3,
        "prompt_template": "Sentiment analysis: {content}"
    }
]

# 执行比较
content = "这个电影真的很棒，演员表演很自然，剧情也很吸引人。"
results = compare_model_predictions(content, models_config)

for result in results:
    print(f"模型: {result['model']}")
    if 'error' in result:
        print(f"  错误: {result['error']}")
    else:
        print(f"  预测: {result['prediction']}")
        print(f"  置信度: {result['confidence']}")
        print(f"  处理时间: {result['processing_time']}s")
        print(f"  缓存命中: {result['cached']}")
    print()
```

### 2. 批量预测

#### POST `/api/v1/ai/predict/batch`

提交批量预测任务，适合大规模数据处理。

**请求体**:
```json
{
  "task_ids": [
    "550e8400-e29b-41d4-a716-446655440000",
    "550e8400-e29b-41d4-a716-446655440001",
    "550e8400-e29b-41d4-a716-446655440002"
  ],
  "ai_model_configs": [
    {
      "model_type": "zhipu_glm",
      "model_name": "glm-4",
      "api_key": "your_api_key",
      "temperature": 0.7,
      "prompt_template": "文本分类：{content}"
    }
  ],
  "batch_config": {
    "max_concurrent_tasks": 5,
    "retry_attempts": 2,
    "timeout_seconds": 30
  }
}
```

**参数说明**:
- `task_ids` (必需): 任务 ID 列表
- `ai_model_configs` (必需): 模型配置列表
- `batch_config` (可选): 批量处理配置
  - `max_concurrent_tasks`: 最大并发任务数，默认 5
  - `retry_attempts`: 重试次数，默认 2
  - `timeout_seconds`: 单个任务超时时间，默认 30

**响应示例**:
```json
{
  "job_id": "batch_550e8400-e29b-41d4-a716-446655440000",
  "status": "submitted",
  "message": "Batch job submitted with 3 tasks"
}
```

**使用示例**:

```python
# Python 示例 - 大规模批量标注
import requests
import time
import json

class BatchAnnotationManager:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def submit_batch_job(self, task_ids, model_config, batch_size=100):
        """提交批量标注任务"""
        # 分批处理大量任务
        job_ids = []
        
        for i in range(0, len(task_ids), batch_size):
            batch_task_ids = task_ids[i:i + batch_size]
            
            request_data = {
                "task_ids": batch_task_ids,
                "ai_model_configs": [model_config],
                "batch_config": {
                    "max_concurrent_tasks": 10,
                    "retry_attempts": 3,
                    "timeout_seconds": 60
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/v1/ai/predict/batch",
                json=request_data
            )
            
            if response.status_code == 200:
                job_id = response.json()["job_id"]
                job_ids.append(job_id)
                print(f"批次 {i//batch_size + 1} 已提交，任务ID: {job_id}")
            else:
                print(f"批次 {i//batch_size + 1} 提交失败: {response.text}")
        
        return job_ids
    
    def monitor_batch_jobs(self, job_ids, check_interval=10):
        """监控批量任务进度"""
        completed_jobs = []
        
        while len(completed_jobs) < len(job_ids):
            for job_id in job_ids:
                if job_id in completed_jobs:
                    continue
                
                response = requests.get(
                    f"{self.base_url}/api/v1/ai/batch/{job_id}"
                )
                
                if response.status_code == 200:
                    job_data = response.json()
                    status = job_data["status"]
                    
                    print(f"任务 {job_id}: {status}")
                    
                    if status in ["completed", "failed", "cancelled"]:
                        completed_jobs.append(job_id)
                        
                        if status == "completed":
                            success_count = job_data.get("successful_tasks", 0)
                            total_count = job_data.get("total_tasks", 0)
                            print(f"  成功: {success_count}/{total_count}")
                        elif status == "failed":
                            print(f"  失败原因: {job_data.get('error', 'Unknown')}")
            
            if len(completed_jobs) < len(job_ids):
                time.sleep(check_interval)
        
        print("所有批量任务已完成")
        return completed_jobs

# 使用示例
manager = BatchAnnotationManager()

# 准备任务列表（实际应用中从数据库获取）
task_ids = [f"task_{i:06d}" for i in range(1000)]  # 1000个任务

# 模型配置
model_config = {
    "model_type": "zhipu_glm",
    "model_name": "glm-4",
    "api_key": "your_api_key",
    "temperature": 0.3,
    "prompt_template": "请对以下文本进行主题分类：{content}"
}

# 提交批量任务
job_ids = manager.submit_batch_job(task_ids, model_config, batch_size=50)

# 监控任务进度
completed_jobs = manager.monitor_batch_jobs(job_ids)
```

### 3. 批量任务管理

#### GET `/api/v1/ai/batch/{job_id}`

查询批量任务状态和结果。

**响应示例**:
```json
{
  "job_id": "batch_550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "total_tasks": 100,
  "successful_tasks": 95,
  "failed_tasks": 5,
  "progress": 100.0,
  "started_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:45:00Z",
  "results": [
    {
      "task_id": "task_001",
      "status": "completed",
      "prediction": {...},
      "processing_time": 1.2
    }
  ],
  "failed_tasks_details": [
    {
      "task_id": "task_099",
      "error": "Model timeout"
    }
  ]
}
```

#### DELETE `/api/v1/ai/batch/{job_id}`

取消正在执行的批量任务。

### 4. 模型管理

#### GET `/api/v1/ai/models`

获取可用模型列表。

**响应示例**:
```json
{
  "models": [
    {
      "model_type": "zhipu_glm",
      "description": "智谱 GLM 系列模型",
      "supported": true,
      "examples": ["glm-4", "glm-3-turbo", "chatglm3-6b"]
    },
    {
      "model_type": "ollama",
      "description": "Ollama 本地模型",
      "supported": true,
      "examples": ["llama2", "codellama", "mistral"]
    }
  ]
}
```

#### GET `/api/v1/ai/models/{model_type}/health`

检查特定模型类型的健康状态。

**使用示例**:

```bash
# 检查智谱 GLM 模型状态
curl -X GET "http://localhost:8000/api/v1/ai/models/zhipu_glm/health"

# 检查 Ollama 模型状态
curl -X GET "http://localhost:8000/api/v1/ai/models/ollama/health"
```

**响应示例**:
```json
{
  "model_type": "zhipu_glm",
  "status": "healthy",
  "available": true,
  "message": "Model is accessible",
  "response_time": 0.85
}
```

### 5. 缓存管理

#### GET `/api/v1/ai/cache/stats`

获取预测缓存统计信息。

**响应示例**:
```json
{
  "total_entries": 1500,
  "cache_hits": 850,
  "cache_misses": 650,
  "hit_rate": 0.567,
  "memory_usage": "45.2MB",
  "oldest_entry": "2024-01-14T10:30:00Z",
  "newest_entry": "2024-01-15T15:45:00Z"
}
```

#### DELETE `/api/v1/ai/cache`

清空所有预测缓存。

**使用示例**:

```python
# Python 示例 - 缓存管理
import requests

def manage_prediction_cache():
    """管理预测缓存"""
    base_url = "http://localhost:8000/api/v1/ai"
    
    # 获取缓存统计
    stats_response = requests.get(f"{base_url}/cache/stats")
    if stats_response.status_code == 200:
        stats = stats_response.json()
        print(f"缓存统计:")
        print(f"  总条目: {stats['total_entries']}")
        print(f"  命中率: {stats['hit_rate']:.2%}")
        print(f"  内存使用: {stats['memory_usage']}")
        
        # 如果命中率太低，清空缓存
        if stats['hit_rate'] < 0.3:
            print("命中率过低，清空缓存...")
            clear_response = requests.delete(f"{base_url}/cache")
            if clear_response.status_code == 200:
                result = clear_response.json()
                print(f"已清空 {result['message']}")
    
    return stats

# 使用示例
cache_stats = manage_prediction_cache()
```

### 6. 模型版本管理

#### POST `/api/v1/ai/models/register`

注册新的模型版本。

**请求体**:
```json
{
  "model_type": "zhipu_glm",
  "model_name": "glm-4-plus",
  "version": "1.0.0",
  "config": {
    "model_type": "zhipu_glm",
    "model_name": "glm-4-plus",
    "api_key": "your_api_key",
    "base_url": "https://open.bigmodel.cn/api/paas/v4/",
    "max_tokens": 4096,
    "temperature": 0.7
  },
  "description": "GLM-4 增强版本，支持更长上下文"
}
```

#### GET `/api/v1/ai/models/versions`

列出已注册的模型版本。

**查询参数**:
- `model_type`: 模型类型过滤
- `model_name`: 模型名称过滤

## 高级功能

### 1. 自定义提示词模板

```python
# Python 示例 - 动态提示词生成
def create_dynamic_prompt(task_type, content, context=None):
    """根据任务类型创建动态提示词"""
    
    templates = {
        "sentiment": "请分析以下文本的情感倾向（积极/消极/中性），并给出置信度：\n文本：{content}",
        "classification": "请将以下文本分类到合适的类别中：\n文本：{content}\n类别：{categories}",
        "extraction": "请从以下文本中提取关键信息：\n文本：{content}\n需要提取：{extract_fields}",
        "summarization": "请对以下文本进行摘要，控制在{max_length}字以内：\n文本：{content}",
        "qa": "基于以下上下文回答问题：\n上下文：{context}\n问题：{content}"
    }
    
    template = templates.get(task_type, "请处理以下文本：{content}")
    
    # 根据上下文填充模板
    if context:
        return template.format(content=content, context=context, **context)
    else:
        return template.format(content=content)

# 使用示例
def batch_annotate_with_dynamic_prompts(tasks, model_config):
    """使用动态提示词进行批量标注"""
    results = []
    
    for task in tasks:
        # 根据任务类型生成提示词
        prompt = create_dynamic_prompt(
            task_type=task["type"],
            content=task["content"],
            context=task.get("context")
        )
        
        # 更新模型配置
        config = model_config.copy()
        config["prompt_template"] = prompt
        
        # 发送预测请求
        request_data = {
            "task_id": task["id"],
            "ai_model_config": config,
            "content": task["content"]
        }
        
        response = requests.post(
            "http://localhost:8000/api/v1/ai/predict",
            json=request_data
        )
        
        if response.status_code == 200:
            result = response.json()
            results.append({
                "task_id": task["id"],
                "task_type": task["type"],
                "prediction": result["prediction_data"],
                "confidence": result["confidence"]
            })
        else:
            results.append({
                "task_id": task["id"],
                "error": response.text
            })
    
    return results
```

### 2. 模型性能监控

```python
# Python 示例 - 模型性能监控
import time
import statistics
from collections import defaultdict

class ModelPerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def track_prediction(self, model_info, processing_time, confidence, success):
        """跟踪预测性能指标"""
        model_key = f"{model_info['model_type']}/{model_info['model_name']}"
        
        self.metrics[model_key].append({
            "timestamp": time.time(),
            "processing_time": processing_time,
            "confidence": confidence,
            "success": success
        })
    
    def get_model_stats(self, model_key, time_window=3600):
        """获取模型统计信息"""
        current_time = time.time()
        recent_metrics = [
            m for m in self.metrics[model_key]
            if current_time - m["timestamp"] <= time_window
        ]
        
        if not recent_metrics:
            return None
        
        processing_times = [m["processing_time"] for m in recent_metrics]
        confidences = [m["confidence"] for m in recent_metrics if m["success"]]
        success_count = sum(1 for m in recent_metrics if m["success"])
        
        return {
            "model": model_key,
            "total_requests": len(recent_metrics),
            "success_rate": success_count / len(recent_metrics),
            "avg_processing_time": statistics.mean(processing_times),
            "median_processing_time": statistics.median(processing_times),
            "avg_confidence": statistics.mean(confidences) if confidences else 0,
            "min_confidence": min(confidences) if confidences else 0,
            "max_confidence": max(confidences) if confidences else 0
        }
    
    def compare_models(self, time_window=3600):
        """比较不同模型的性能"""
        comparison = []
        
        for model_key in self.metrics:
            stats = self.get_model_stats(model_key, time_window)
            if stats:
                comparison.append(stats)
        
        # 按成功率和平均置信度排序
        comparison.sort(
            key=lambda x: (x["success_rate"], x["avg_confidence"]),
            reverse=True
        )
        
        return comparison

# 使用示例
monitor = ModelPerformanceMonitor()

def monitored_prediction(task_id, model_config, content):
    """带监控的预测请求"""
    start_time = time.time()
    
    request_data = {
        "task_id": task_id,
        "ai_model_config": model_config,
        "content": content
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/api/v1/ai/predict",
            json=request_data
        )
        
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            # 记录性能指标
            monitor.track_prediction(
                model_info=result["model_info"],
                processing_time=processing_time,
                confidence=result["confidence"],
                success=True
            )
            
            return result
        else:
            # 记录失败
            monitor.track_prediction(
                model_info={"model_type": model_config["model_type"], "model_name": model_config["model_name"]},
                processing_time=processing_time,
                confidence=0,
                success=False
            )
            return None
            
    except Exception as e:
        processing_time = time.time() - start_time
        monitor.track_prediction(
            model_info={"model_type": model_config["model_type"], "model_name": model_config["model_name"]},
            processing_time=processing_time,
            confidence=0,
            success=False
        )
        raise e

# 定期生成性能报告
def generate_performance_report():
    """生成性能报告"""
    comparison = monitor.compare_models()
    
    print("模型性能报告 (最近1小时)")
    print("=" * 50)
    
    for i, stats in enumerate(comparison, 1):
        print(f"{i}. {stats['model']}")
        print(f"   请求数: {stats['total_requests']}")
        print(f"   成功率: {stats['success_rate']:.2%}")
        print(f"   平均处理时间: {stats['avg_processing_time']:.2f}s")
        print(f"   平均置信度: {stats['avg_confidence']:.2f}")
        print()

# 每小时生成一次报告
import threading
def periodic_report():
    while True:
        time.sleep(3600)  # 1小时
        generate_performance_report()

# 启动后台监控
threading.Thread(target=periodic_report, daemon=True).start()
```

### 3. 智能模型选择

```python
# Python 示例 - 智能模型选择
class IntelligentModelSelector:
    def __init__(self, performance_monitor):
        self.monitor = performance_monitor
        self.model_configs = {
            "fast": {
                "model_type": "ollama",
                "model_name": "llama2",
                "temperature": 0.3
            },
            "accurate": {
                "model_type": "zhipu_glm",
                "model_name": "glm-4",
                "temperature": 0.1
            },
            "balanced": {
                "model_type": "zhipu_glm",
                "model_name": "glm-3-turbo",
                "temperature": 0.5
            }
        }
    
    def select_best_model(self, task_requirements):
        """根据任务需求选择最佳模型"""
        priority = task_requirements.get("priority", "balanced")  # fast, accurate, balanced
        max_processing_time = task_requirements.get("max_processing_time", 10)
        min_confidence = task_requirements.get("min_confidence", 0.7)
        
        # 获取模型性能统计
        model_stats = self.monitor.compare_models()
        
        # 根据需求筛选模型
        suitable_models = []
        for stats in model_stats:
            if (stats["avg_processing_time"] <= max_processing_time and
                stats["avg_confidence"] >= min_confidence and
                stats["success_rate"] >= 0.9):
                suitable_models.append(stats)
        
        if not suitable_models:
            # 如果没有合适的模型，使用默认配置
            return self.model_configs[priority]
        
        # 根据优先级选择
        if priority == "fast":
            best_model = min(suitable_models, key=lambda x: x["avg_processing_time"])
        elif priority == "accurate":
            best_model = max(suitable_models, key=lambda x: x["avg_confidence"])
        else:  # balanced
            # 综合评分：成功率 * 置信度 / 处理时间
            for model in suitable_models:
                model["score"] = (model["success_rate"] * model["avg_confidence"]) / model["avg_processing_time"]
            best_model = max(suitable_models, key=lambda x: x["score"])
        
        # 返回对应的模型配置
        model_key = best_model["model"]
        model_type, model_name = model_key.split("/")
        
        return {
            "model_type": model_type,
            "model_name": model_name,
            "temperature": 0.3,
            "selected_reason": f"基于{priority}优先级选择，成功率{best_model['success_rate']:.2%}"
        }

# 使用示例
selector = IntelligentModelSelector(monitor)

def smart_prediction(task_id, content, requirements=None):
    """智能预测，自动选择最佳模型"""
    if requirements is None:
        requirements = {"priority": "balanced"}
    
    # 选择最佳模型
    model_config = selector.select_best_model(requirements)
    print(f"选择模型: {model_config['model_type']}/{model_config['model_name']}")
    print(f"选择原因: {model_config.get('selected_reason', 'Default selection')}")
    
    # 执行预测
    return monitored_prediction(task_id, model_config, content)

# 不同场景的使用示例
# 快速处理场景
fast_result = smart_prediction(
    "task_001", 
    "简单文本分类任务",
    {"priority": "fast", "max_processing_time": 2}
)

# 高精度场景
accurate_result = smart_prediction(
    "task_002",
    "复杂情感分析任务",
    {"priority": "accurate", "min_confidence": 0.9}
)
```

## 错误处理和故障排除

### 常见错误

1. **模型不可用**
```json
{
  "error": "Model Unavailable",
  "message": "Model zhipu_glm/glm-4 is not accessible",
  "details": {
    "model_type": "zhipu_glm",
    "model_name": "glm-4",
    "error_code": "API_KEY_INVALID"
  }
}
```

2. **请求超时**
```json
{
  "error": "Request Timeout",
  "message": "Model request timed out after 30 seconds",
  "details": {
    "timeout": 30,
    "model_info": {...}
  }
}
```

3. **配额超限**
```json
{
  "error": "Quota Exceeded",
  "message": "API quota exceeded for model",
  "details": {
    "quota_limit": 1000,
    "quota_used": 1000,
    "reset_time": "2024-01-16T00:00:00Z"
  }
}
```

### 故障排除指南

```python
# Python 示例 - 健壮的错误处理
import requests
from requests.exceptions import Timeout, ConnectionError
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def robust_ai_prediction(task_id, model_config, content, max_retries=3):
    """健壮的 AI 预测请求"""
    
    for attempt in range(max_retries):
        try:
            request_data = {
                "task_id": task_id,
                "ai_model_config": model_config,
                "content": content
            }
            
            response = requests.post(
                "http://localhost:8000/api/v1/ai/predict",
                json=request_data,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # 频率限制
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"频率限制，等待 {retry_after} 秒后重试")
                time.sleep(retry_after)
                continue
            elif response.status_code == 503:  # 服务不可用
                wait_time = 2 ** attempt
                logger.warning(f"服务不可用，等待 {wait_time} 秒后重试")
                time.sleep(wait_time)
                continue
            else:
                error_data = response.json() if response.headers.get('content-type') == 'application/json' else {}
                logger.error(f"预测失败: HTTP {response.status_code}, {error_data}")
                
                # 对于客户端错误，不重试
                if 400 <= response.status_code < 500:
                    return {"error": error_data, "status_code": response.status_code}
                
        except Timeout:
            logger.warning(f"请求超时 (尝试 {attempt + 1}/{max_retries})")
            if attempt == max_retries - 1:
                return {"error": "Request timeout", "attempts": max_retries}
            time.sleep(2 ** attempt)
            
        except ConnectionError:
            logger.warning(f"连接错误 (尝试 {attempt + 1}/{max_retries})")
            if attempt == max_retries - 1:
                return {"error": "Connection error", "attempts": max_retries}
            time.sleep(2 ** attempt)
            
        except Exception as e:
            logger.error(f"未知错误: {e}")
            return {"error": str(e)}
    
    return {"error": "Max retries exceeded"}

# 使用示例
result = robust_ai_prediction(
    task_id="test_task",
    model_config={
        "model_type": "zhipu_glm",
        "model_name": "glm-4",
        "api_key": "your_api_key"
    },
    content="测试文本"
)

if "error" in result:
    print(f"预测失败: {result['error']}")
else:
    print(f"预测成功: {result['prediction_data']}")
```

---

*更多详细信息请参考 [API 总览文档](README.md) 或联系技术支持。*