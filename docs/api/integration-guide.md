# SuperInsight 平台集成指南

## 概述

本指南提供了将 SuperInsight 平台集成到现有系统中的详细说明，包括最佳实践、示例代码和常见集成模式。

## 集成架构模式

### 1. 直接 API 集成

最简单的集成方式，直接调用 SuperInsight API。

```mermaid
graph LR
    A[客户端应用] --> B[SuperInsight API]
    B --> C[数据源]
    B --> D[AI 模型]
    B --> E[数据库]
```

**适用场景**:
- 小型应用
- 简单的数据处理需求
- 快速原型开发

### 2. 微服务集成

通过微服务架构集成，提供更好的可扩展性。

```mermaid
graph TB
    A[前端应用] --> B[API 网关]
    B --> C[业务服务]
    C --> D[SuperInsight API]
    C --> E[其他微服务]
    D --> F[数据处理]
    D --> G[AI 标注]
```

**适用场景**:
- 大型企业应用
- 复杂的业务逻辑
- 需要高可用性和扩展性

### 3. 事件驱动集成

基于消息队列的异步集成模式。

```mermaid
graph TB
    A[数据源] --> B[消息队列]
    B --> C[SuperInsight 处理器]
    C --> D[SuperInsight API]
    D --> E[结果队列]
    E --> F[下游系统]
```

**适用场景**:
- 大数据量处理
- 实时数据流
- 解耦系统依赖

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install requests aiohttp python-dotenv

# 创建环境配置文件
cat > .env << EOF
SUPERINSIGHT_API_URL=http://localhost:8000
SUPERINSIGHT_API_TOKEN=your_api_token_here
ZHIPU_API_KEY=your_zhipu_api_key
BAIDU_API_KEY=your_baidu_api_key
EOF
```

### 2. 基础客户端实现

```python
# superinsight_client.py
import os
import requests
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtractionConfig:
    """数据提取配置"""
    source_type: str  # database, file, web, api
    config: Dict[str, Any]

@dataclass
class ModelConfig:
    """AI 模型配置"""
    model_type: str
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    prompt_template: str = "请处理以下内容：{content}"

class SuperInsightClient:
    """SuperInsight 平台客户端"""
    
    def __init__(self, base_url: str, api_token: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        # 设置认证头
        if api_token:
            self.session.headers.update({
                'Authorization': f'Bearer {api_token}'
            })
        
        # 设置默认头
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'SuperInsight-Client/1.0'
        })
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            raise
    
    def extract_data(self, config: ExtractionConfig) -> str:
        """提取数据"""
        endpoint_map = {
            'database': '/api/v1/extraction/database',
            'file': '/api/v1/extraction/file',
            'web': '/api/v1/extraction/web',
            'api': '/api/v1/extraction/api'
        }
        
        endpoint = endpoint_map.get(config.source_type)
        if not endpoint:
            raise ValueError(f"不支持的数据源类型: {config.source_type}")
        
        try:
            response = self.session.post(
                f"{self.base_url}{endpoint}",
                json=config.config,
                timeout=30
            )
            response.raise_for_status()
            return response.json()['job_id']
        except Exception as e:
            logger.error(f"数据提取失败: {e}")
            raise
    
    def get_extraction_status(self, job_id: str) -> Dict[str, Any]:
        """获取提取任务状态"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/extraction/jobs/{job_id}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"获取任务状态失败: {e}")
            raise
    
    def wait_for_extraction(self, job_id: str, timeout: int = 300, 
                          check_interval: int = 5) -> Dict[str, Any]:
        """等待提取任务完成"""
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_extraction_status(job_id)
            
            if status['status'] == 'completed':
                logger.info(f"任务 {job_id} 完成，提取了 {status['documents_count']} 个文档")
                return status
            elif status['status'] == 'failed':
                logger.error(f"任务 {job_id} 失败: {status.get('error', 'Unknown error')}")
                raise Exception(f"提取任务失败: {status.get('error')}")
            
            logger.info(f"任务 {job_id} 状态: {status['status']}")
            time.sleep(check_interval)
        
        raise TimeoutError(f"任务 {job_id} 超时")
    
    def ai_predict(self, task_id: str, content: str, model_config: ModelConfig) -> Dict[str, Any]:
        """AI 预测"""
        request_data = {
            "task_id": task_id,
            "ai_model_config": {
                "model_type": model_config.model_type,
                "model_name": model_config.model_name,
                "api_key": model_config.api_key,
                "base_url": model_config.base_url,
                "temperature": model_config.temperature,
                "max_tokens": model_config.max_tokens,
                "prompt_template": model_config.prompt_template
            },
            "content": content
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/ai/predict",
                json=request_data,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"AI 预测失败: {e}")
            raise
    
    def ai_predict_batch(self, task_ids: List[str], model_configs: List[ModelConfig]) -> str:
        """批量 AI 预测"""
        request_data = {
            "task_ids": task_ids,
            "ai_model_configs": [
                {
                    "model_type": config.model_type,
                    "model_name": config.model_name,
                    "api_key": config.api_key,
                    "base_url": config.base_url,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "prompt_template": config.prompt_template
                }
                for config in model_configs
            ]
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/ai/predict/batch",
                json=request_data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()['job_id']
        except Exception as e:
            logger.error(f"批量 AI 预测失败: {e}")
            raise
    
    def get_batch_status(self, job_id: str) -> Dict[str, Any]:
        """获取批量任务状态"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v1/ai/batch/{job_id}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"获取批量任务状态失败: {e}")
            raise

# 使用示例
def main():
    # 从环境变量加载配置
    from dotenv import load_dotenv
    load_dotenv()
    
    client = SuperInsightClient(
        base_url=os.getenv('SUPERINSIGHT_API_URL', 'http://localhost:8000'),
        api_token=os.getenv('SUPERINSIGHT_API_TOKEN')
    )
    
    # 健康检查
    try:
        health = client.health_check()
        print(f"服务状态: {health['overall_status']}")
    except Exception as e:
        print(f"服务不可用: {e}")
        return
    
    # 数据提取示例
    extraction_config = ExtractionConfig(
        source_type='database',
        config={
            "host": "localhost",
            "port": 5432,
            "database": "mydb",
            "username": "user",
            "password": "password",
            "database_type": "postgresql",
            "table_name": "articles",
            "limit": 100
        }
    )
    
    try:
        job_id = client.extract_data(extraction_config)
        print(f"数据提取任务已提交: {job_id}")
        
        # 等待任务完成
        result = client.wait_for_extraction(job_id)
        print(f"提取完成，共 {result['documents_count']} 个文档")
        
        # AI 预测示例
        if result['documents']:
            model_config = ModelConfig(
                model_type='zhipu_glm',
                model_name='glm-4',
                api_key=os.getenv('ZHIPU_API_KEY'),
                prompt_template='请对以下文本进行情感分析：{content}'
            )
            
            # 对第一个文档进行预测
            doc = result['documents'][0]
            prediction = client.ai_predict(
                task_id=f"predict_{doc['id']}",
                content=doc['content'][:1000],  # 限制长度
                model_config=model_config
            )
            
            print(f"AI 预测结果: {prediction['prediction_data']}")
            print(f"置信度: {prediction['confidence']}")
    
    except Exception as e:
        print(f"处理失败: {e}")

if __name__ == "__main__":
    main()
```

## 高级集成模式

### 1. 异步客户端

```python
# async_client.py
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class AsyncSuperInsightClient:
    """异步 SuperInsight 客户端"""
    
    def __init__(self, base_url: str, api_token: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.headers = {'Content-Type': 'application/json'}
        
        if api_token:
            self.headers['Authorization'] = f'Bearer {api_token}'
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
    
    async def extract_data_async(self, configs: List[ExtractionConfig]) -> List[str]:
        """异步批量数据提取"""
        tasks = []
        
        for config in configs:
            endpoint_map = {
                'database': '/api/v1/extraction/database',
                'file': '/api/v1/extraction/file',
                'web': '/api/v1/extraction/web',
                'api': '/api/v1/extraction/api'
            }
            
            endpoint = endpoint_map.get(config.source_type)
            if endpoint:
                task = self._submit_extraction_task(endpoint, config.config)
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        job_ids = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"任务 {i} 提交失败: {result}")
            else:
                job_ids.append(result)
        
        return job_ids
    
    async def _submit_extraction_task(self, endpoint: str, config: Dict[str, Any]) -> str:
        """提交单个提取任务"""
        async with self.session.post(
            f"{self.base_url}{endpoint}",
            json=config,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            response.raise_for_status()
            data = await response.json()
            return data['job_id']
    
    async def monitor_jobs(self, job_ids: List[str], 
                          check_interval: int = 5) -> List[Dict[str, Any]]:
        """监控多个任务状态"""
        completed_jobs = []
        pending_jobs = job_ids.copy()
        
        while pending_jobs:
            # 并发检查所有待完成任务
            tasks = [
                self._get_job_status(job_id) 
                for job_id in pending_jobs
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            still_pending = []
            for job_id, result in zip(pending_jobs, results):
                if isinstance(result, Exception):
                    logger.error(f"检查任务 {job_id} 状态失败: {result}")
                    still_pending.append(job_id)
                    continue
                
                status = result['status']
                if status in ['completed', 'failed']:
                    completed_jobs.append(result)
                    logger.info(f"任务 {job_id} 完成，状态: {status}")
                else:
                    still_pending.append(job_id)
            
            pending_jobs = still_pending
            
            if pending_jobs:
                await asyncio.sleep(check_interval)
        
        return completed_jobs
    
    async def _get_job_status(self, job_id: str) -> Dict[str, Any]:
        """获取单个任务状态"""
        async with self.session.get(
            f"{self.base_url}/api/v1/extraction/jobs/{job_id}",
            timeout=aiohttp.ClientTimeout(total=10)
        ) as response:
            response.raise_for_status()
            return await response.json()

# 使用示例
async def async_processing_example():
    """异步处理示例"""
    configs = [
        ExtractionConfig('database', {
            "host": "db1.example.com",
            "port": 5432,
            "database": "db1",
            "username": "user",
            "password": "pass",
            "database_type": "postgresql",
            "table_name": "articles"
        }),
        ExtractionConfig('database', {
            "host": "db2.example.com",
            "port": 3306,
            "database": "db2",
            "username": "user",
            "password": "pass",
            "database_type": "mysql",
            "table_name": "posts"
        }),
        ExtractionConfig('web', {
            "base_url": "https://news.example.com",
            "max_pages": 50
        })
    ]
    
    async with AsyncSuperInsightClient("http://localhost:8000") as client:
        # 并发提交所有任务
        job_ids = await client.extract_data_async(configs)
        print(f"提交了 {len(job_ids)} 个任务")
        
        # 监控任务完成
        completed_jobs = await client.monitor_jobs(job_ids)
        
        total_documents = sum(job['documents_count'] for job in completed_jobs if job['success'])
        print(f"所有任务完成，共提取 {total_documents} 个文档")

# 运行异步示例
# asyncio.run(async_processing_example())
```

### 2. 企业级集成包装器

```python
# enterprise_wrapper.py
import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue, Empty

logger = logging.getLogger(__name__)

@dataclass
class ProcessingJob:
    """处理任务"""
    job_id: str
    job_type: str  # extraction, ai_prediction
    config: Dict[str, Any]
    status: str = 'pending'
    created_at: datetime = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class SuperInsightEnterprise:
    """企业级 SuperInsight 集成包装器"""
    
    def __init__(self, base_url: str, api_token: Optional[str] = None,
                 max_workers: int = 10, retry_attempts: int = 3):
        self.client = SuperInsightClient(base_url, api_token)
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts
        
        # 任务管理
        self.jobs: Dict[str, ProcessingJob] = {}
        self.job_queue = Queue()
        self.result_callbacks: Dict[str, Callable] = {}
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.monitor_thread = None
        self.running = False
        
        # 统计信息
        self.stats = {
            'total_jobs': 0,
            'completed_jobs': 0,
            'failed_jobs': 0,
            'processing_time': []
        }
    
    def start(self):
        """启动后台处理"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_jobs, daemon=True)
        self.monitor_thread.start()
        logger.info("企业级处理器已启动")
    
    def stop(self):
        """停止后台处理"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.executor.shutdown(wait=True)
        logger.info("企业级处理器已停止")
    
    def submit_extraction_job(self, config: ExtractionConfig, 
                            callback: Optional[Callable] = None) -> str:
        """提交数据提取任务"""
        job_id = f"ext_{int(time.time() * 1000)}"
        job = ProcessingJob(
            job_id=job_id,
            job_type='extraction',
            config=asdict(config)
        )
        
        self.jobs[job_id] = job
        self.job_queue.put(job_id)
        
        if callback:
            self.result_callbacks[job_id] = callback
        
        self.stats['total_jobs'] += 1
        logger.info(f"提交提取任务: {job_id}")
        return job_id
    
    def submit_ai_job(self, task_id: str, content: str, model_config: ModelConfig,
                     callback: Optional[Callable] = None) -> str:
        """提交 AI 预测任务"""
        job_id = f"ai_{int(time.time() * 1000)}"
        job = ProcessingJob(
            job_id=job_id,
            job_type='ai_prediction',
            config={
                'task_id': task_id,
                'content': content,
                'model_config': asdict(model_config)
            }
        )
        
        self.jobs[job_id] = job
        self.job_queue.put(job_id)
        
        if callback:
            self.result_callbacks[job_id] = callback
        
        self.stats['total_jobs'] += 1
        logger.info(f"提交 AI 任务: {job_id}")
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[ProcessingJob]:
        """获取任务状态"""
        return self.jobs.get(job_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        stats['active_jobs'] = len([j for j in self.jobs.values() if j.status in ['pending', 'running']])
        stats['average_processing_time'] = (
            sum(stats['processing_time']) / len(stats['processing_time'])
            if stats['processing_time'] else 0
        )
        return stats
    
    def _monitor_jobs(self):
        """监控任务执行"""
        while self.running:
            try:
                # 从队列获取任务
                try:
                    job_id = self.job_queue.get(timeout=1)
                except Empty:
                    continue
                
                job = self.jobs.get(job_id)
                if not job:
                    continue
                
                # 提交到线程池执行
                future = self.executor.submit(self._process_job, job)
                
                # 不阻塞，继续处理下一个任务
                
            except Exception as e:
                logger.error(f"任务监控异常: {e}")
    
    def _process_job(self, job: ProcessingJob):
        """处理单个任务"""
        job.status = 'running'
        start_time = time.time()
        
        try:
            if job.job_type == 'extraction':
                result = self._process_extraction_job(job)
            elif job.job_type == 'ai_prediction':
                result = self._process_ai_job(job)
            else:
                raise ValueError(f"未知任务类型: {job.job_type}")
            
            job.status = 'completed'
            job.result = result
            job.completed_at = datetime.now()
            
            processing_time = time.time() - start_time
            self.stats['processing_time'].append(processing_time)
            self.stats['completed_jobs'] += 1
            
            logger.info(f"任务 {job.job_id} 完成，耗时 {processing_time:.2f}s")
            
        except Exception as e:
            job.status = 'failed'
            job.error = str(e)
            job.completed_at = datetime.now()
            self.stats['failed_jobs'] += 1
            
            logger.error(f"任务 {job.job_id} 失败: {e}")
        
        # 执行回调
        if job.job_id in self.result_callbacks:
            try:
                callback = self.result_callbacks[job.job_id]
                callback(job)
                del self.result_callbacks[job.job_id]
            except Exception as e:
                logger.error(f"回调执行失败: {e}")
    
    def _process_extraction_job(self, job: ProcessingJob) -> Dict[str, Any]:
        """处理数据提取任务"""
        config_dict = job.config
        config = ExtractionConfig(
            source_type=config_dict['source_type'],
            config=config_dict['config']
        )
        
        # 重试机制
        for attempt in range(self.retry_attempts):
            try:
                api_job_id = self.client.extract_data(config)
                result = self.client.wait_for_extraction(api_job_id, timeout=600)
                return result
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    raise e
                logger.warning(f"任务 {job.job_id} 第 {attempt + 1} 次尝试失败: {e}")
                time.sleep(2 ** attempt)  # 指数退避
    
    def _process_ai_job(self, job: ProcessingJob) -> Dict[str, Any]:
        """处理 AI 预测任务"""
        config_dict = job.config
        model_config = ModelConfig(**config_dict['model_config'])
        
        # 重试机制
        for attempt in range(self.retry_attempts):
            try:
                result = self.client.ai_predict(
                    task_id=config_dict['task_id'],
                    content=config_dict['content'],
                    model_config=model_config
                )
                return result
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    raise e
                logger.warning(f"任务 {job.job_id} 第 {attempt + 1} 次尝试失败: {e}")
                time.sleep(2 ** attempt)

# 使用示例
def enterprise_integration_example():
    """企业级集成示例"""
    
    # 创建企业级处理器
    processor = SuperInsightEnterprise(
        base_url="http://localhost:8000",
        max_workers=5,
        retry_attempts=3
    )
    
    # 启动后台处理
    processor.start()
    
    # 定义结果回调
    def extraction_callback(job: ProcessingJob):
        if job.status == 'completed':
            doc_count = job.result.get('documents_count', 0)
            print(f"提取任务 {job.job_id} 完成，获得 {doc_count} 个文档")
        else:
            print(f"提取任务 {job.job_id} 失败: {job.error}")
    
    def ai_callback(job: ProcessingJob):
        if job.status == 'completed':
            confidence = job.result.get('confidence', 0)
            print(f"AI 任务 {job.job_id} 完成，置信度: {confidence}")
        else:
            print(f"AI 任务 {job.job_id} 失败: {job.error}")
    
    # 提交多个任务
    extraction_configs = [
        ExtractionConfig('database', {
            "host": "localhost",
            "port": 5432,
            "database": "db1",
            "username": "user",
            "password": "pass",
            "database_type": "postgresql",
            "table_name": "articles",
            "limit": 100
        }),
        ExtractionConfig('web', {
            "base_url": "https://example.com",
            "max_pages": 20
        })
    ]
    
    # 提交提取任务
    extraction_jobs = []
    for config in extraction_configs:
        job_id = processor.submit_extraction_job(config, extraction_callback)
        extraction_jobs.append(job_id)
    
    # 提交 AI 任务
    model_config = ModelConfig(
        model_type='zhipu_glm',
        model_name='glm-4',
        api_key=os.getenv('ZHIPU_API_KEY'),
        prompt_template='情感分析：{content}'
    )
    
    ai_jobs = []
    test_contents = [
        "这个产品真的很棒！",
        "服务态度很差，不推荐。",
        "价格合理，质量一般。"
    ]
    
    for i, content in enumerate(test_contents):
        job_id = processor.submit_ai_job(
            task_id=f"sentiment_{i}",
            content=content,
            model_config=model_config,
            callback=ai_callback
        )
        ai_jobs.append(job_id)
    
    # 等待所有任务完成
    all_jobs = extraction_jobs + ai_jobs
    while True:
        completed = 0
        for job_id in all_jobs:
            job = processor.get_job_status(job_id)
            if job and job.status in ['completed', 'failed']:
                completed += 1
        
        if completed == len(all_jobs):
            break
        
        print(f"进度: {completed}/{len(all_jobs)} 任务完成")
        time.sleep(2)
    
    # 显示统计信息
    stats = processor.get_statistics()
    print(f"\n处理统计:")
    print(f"  总任务数: {stats['total_jobs']}")
    print(f"  完成任务: {stats['completed_jobs']}")
    print(f"  失败任务: {stats['failed_jobs']}")
    print(f"  平均处理时间: {stats['average_processing_time']:.2f}s")
    
    # 停止处理器
    processor.stop()

if __name__ == "__main__":
    enterprise_integration_example()
```

## 部署集成

### 1. Docker 容器集成

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 复制应用代码
COPY . .

# 设置环境变量
ENV SUPERINSIGHT_API_URL=http://superinsight-api:8000
ENV PYTHONPATH=/app

# 启动应用
CMD ["python", "main.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  superinsight-api:
    image: superinsight/platform:latest
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/superinsight
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
  
  my-app:
    build: .
    environment:
      - SUPERINSIGHT_API_URL=http://superinsight-api:8000
      - SUPERINSIGHT_API_TOKEN=${API_TOKEN}
    depends_on:
      - superinsight-api
  
  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=superinsight
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### 2. Kubernetes 集成

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: my-app
        image: my-app:latest
        env:
        - name: SUPERINSIGHT_API_URL
          value: "http://superinsight-service:8000"
        - name: SUPERINSIGHT_API_TOKEN
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: superinsight-token
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: my-app-service
spec:
  selector:
    app: my-app
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer

---
apiVersion: v1
kind: Secret
metadata:
  name: api-secrets
type: Opaque
data:
  superinsight-token: <base64-encoded-token>
```

## 监控和日志

### 1. 集成监控

```python
# monitoring.py
import time
import logging
from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading

@dataclass
class MetricPoint:
    timestamp: datetime
    value: float
    labels: Dict[str, str] = None

class IntegrationMonitor:
    """集成监控器"""
    
    def __init__(self):
        self.metrics: Dict[str, list] = {}
        self.alerts: list = []
        self.thresholds = {
            'api_response_time': 5.0,  # 秒
            'error_rate': 0.1,  # 10%
            'queue_size': 100
        }
    
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """记录指标"""
        if name not in self.metrics:
            self.metrics[name] = []
        
        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            labels=labels or {}
        )
        
        self.metrics[name].append(point)
        
        # 保留最近1小时的数据
        cutoff = datetime.now() - timedelta(hours=1)
        self.metrics[name] = [
            p for p in self.metrics[name] 
            if p.timestamp > cutoff
        ]
        
        # 检查阈值
        self._check_threshold(name, value)
    
    def _check_threshold(self, metric_name: str, value: float):
        """检查阈值告警"""
        threshold = self.thresholds.get(metric_name)
        if threshold and value > threshold:
            alert = {
                'metric': metric_name,
                'value': value,
                'threshold': threshold,
                'timestamp': datetime.now(),
                'message': f'{metric_name} 超过阈值: {value} > {threshold}'
            }
            self.alerts.append(alert)
            logging.warning(alert['message'])
    
    def get_metric_summary(self, name: str, minutes: int = 5) -> Dict[str, Any]:
        """获取指标摘要"""
        if name not in self.metrics:
            return {}
        
        cutoff = datetime.now() - timedelta(minutes=minutes)
        recent_points = [
            p for p in self.metrics[name]
            if p.timestamp > cutoff
        ]
        
        if not recent_points:
            return {}
        
        values = [p.value for p in recent_points]
        return {
            'count': len(values),
            'avg': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'latest': values[-1]
        }

# 监控装饰器
def monitor_api_call(monitor: IntegrationMonitor):
    """API 调用监控装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                monitor.record_metric('api_response_time', duration, {
                    'function': func.__name__,
                    'status': 'success'
                })
                return result
            except Exception as e:
                duration = time.time() - start_time
                monitor.record_metric('api_response_time', duration, {
                    'function': func.__name__,
                    'status': 'error'
                })
                monitor.record_metric('api_error_count', 1, {
                    'function': func.__name__,
                    'error_type': type(e).__name__
                })
                raise
        return wrapper
    return decorator

# 使用示例
monitor = IntegrationMonitor()

class MonitoredSuperInsightClient(SuperInsightClient):
    """带监控的客户端"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.monitor = monitor
    
    @monitor_api_call(monitor)
    def extract_data(self, config: ExtractionConfig) -> str:
        return super().extract_data(config)
    
    @monitor_api_call(monitor)
    def ai_predict(self, task_id: str, content: str, model_config: ModelConfig) -> Dict[str, Any]:
        return super().ai_predict(task_id, content, model_config)
```

### 2. 结构化日志

```python
# structured_logging.py
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

class StructuredLogger:
    """结构化日志记录器"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 添加控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def log_api_call(self, method: str, endpoint: str, 
                    duration: float, status_code: int,
                    request_id: Optional[str] = None,
                    extra_data: Optional[Dict[str, Any]] = None):
        """记录 API 调用日志"""
        log_data = {
            'event_type': 'api_call',
            'method': method,
            'endpoint': endpoint,
            'duration_ms': round(duration * 1000, 2),
            'status_code': status_code,
            'timestamp': datetime.now().isoformat(),
            'request_id': request_id
        }
        
        if extra_data:
            log_data.update(extra_data)
        
        level = logging.INFO if status_code < 400 else logging.ERROR
        self.logger.log(level, json.dumps(log_data, ensure_ascii=False))
    
    def log_job_event(self, job_id: str, job_type: str, 
                     event: str, status: str,
                     extra_data: Optional[Dict[str, Any]] = None):
        """记录任务事件日志"""
        log_data = {
            'event_type': 'job_event',
            'job_id': job_id,
            'job_type': job_type,
            'event': event,
            'status': status,
            'timestamp': datetime.now().isoformat()
        }
        
        if extra_data:
            log_data.update(extra_data)
        
        self.logger.info(json.dumps(log_data, ensure_ascii=False))
    
    def log_error(self, error: Exception, context: Dict[str, Any]):
        """记录错误日志"""
        log_data = {
            'event_type': 'error',
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.error(json.dumps(log_data, ensure_ascii=False))

# 使用示例
structured_logger = StructuredLogger('superinsight_integration')

class LoggedSuperInsightClient(SuperInsightClient):
    """带结构化日志的客户端"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = structured_logger
    
    def extract_data(self, config: ExtractionConfig) -> str:
        start_time = time.time()
        try:
            result = super().extract_data(config)
            duration = time.time() - start_time
            
            self.logger.log_api_call(
                method='POST',
                endpoint=f'/api/v1/extraction/{config.source_type}',
                duration=duration,
                status_code=200,
                extra_data={'job_id': result}
            )
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            self.logger.log_api_call(
                method='POST',
                endpoint=f'/api/v1/extraction/{config.source_type}',
                duration=duration,
                status_code=500
            )
            self.logger.log_error(e, {'config': asdict(config)})
            raise
```

## 最佳实践总结

### 1. 性能优化
- 使用连接池减少连接开销
- 实现智能重试机制
- 合理设置超时时间
- 使用批量 API 提高效率
- 实现客户端缓存

### 2. 错误处理
- 实现全面的异常捕获
- 区分可重试和不可重试错误
- 记录详细的错误上下文
- 提供用户友好的错误信息

### 3. 监控和日志
- 记录关键性能指标
- 实现结构化日志
- 设置合理的告警阈值
- 定期分析性能趋势

### 4. 安全考虑
- 安全存储 API 密钥
- 实现访问控制
- 验证输入数据
- 使用 HTTPS 传输

### 5. 可扩展性
- 设计模块化架构
- 支持水平扩展
- 实现负载均衡
- 考虑容器化部署

---

*更多集成示例和最佳实践请参考 [GitHub 示例仓库](https://github.com/superinsight/integration-examples) 或联系技术支持团队。*