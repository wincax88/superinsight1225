# SuperInsight AI 数据治理与标注平台 API 文档

## 概述

SuperInsight 平台提供完整的 RESTful API，支持企业级 AI 语料治理与智能标注的全流程操作。本文档提供详细的 API 使用指南、示例代码和最佳实践。

## 快速开始

### 基础信息

- **API 基础 URL**: `http://localhost:8000` (开发环境)
- **API 版本**: v1
- **认证方式**: Bearer Token (生产环境)
- **数据格式**: JSON
- **字符编码**: UTF-8

### 健康检查

在开始使用 API 之前，建议先检查服务状态：

```bash
curl -X GET "http://localhost:8000/health"
```

**响应示例**:
```json
{
  "overall_status": "healthy",
  "services": {
    "database": "healthy",
    "metrics": "healthy",
    "ai_service": "healthy"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## API 模块概览

### 1. 数据提取 API (`/api/v1/extraction`)
- 支持数据库、文件、网页、API 数据源
- 异步任务处理
- 进度跟踪和结果管理

### 2. AI 预标注 API (`/api/v1/ai`)
- 多模型支持 (智谱GLM、百度文心、腾讯混元等)
- 批量处理和缓存机制
- 模型版本管理

### 3. 质量管理 API (`/api/quality`)
- Ragas 语义质量评估
- 质量工单管理
- 自动修复建议

### 4. 计费结算 API (`/api/billing`)
- 工时和条数统计
- 多租户计费
- 成本分析和预测

### 5. 数据导出 API (`/api/export`)
- 多格式支持 (JSON、CSV、COCO)
- 大数据量分批导出
- RAG/Agent 测试接口

### 6. 安全控制 API (`/api/security`)
- 权限管理
- 数据脱敏
- 审计日志

## 错误处理

### 标准错误格式

所有 API 错误都遵循统一格式：

```json
{
  "error": "错误类型",
  "error_id": "唯一错误标识",
  "message": "详细错误描述",
  "details": {
    "field": "具体字段错误信息"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### 常见错误码

| HTTP 状态码 | 错误类型 | 描述 |
|------------|---------|------|
| 400 | Bad Request | 请求参数错误 |
| 401 | Unauthorized | 认证失败 |
| 403 | Forbidden | 权限不足 |
| 404 | Not Found | 资源不存在 |
| 429 | Too Many Requests | 请求频率超限 |
| 500 | Internal Server Error | 服务器内部错误 |

## 认证和授权

### 开发环境
开发环境暂不需要认证，可直接访问所有 API。

### 生产环境
生产环境需要在请求头中包含认证令牌：

```bash
curl -X GET "https://api.superinsight.com/health" \
  -H "Authorization: Bearer YOUR_API_TOKEN"
```

## 分页和限制

### 分页参数
- `page`: 页码 (从 1 开始)
- `page_size`: 每页大小 (默认 20，最大 100)

### 示例
```bash
curl -X GET "http://localhost:8000/api/v1/extraction/jobs?page=1&page_size=20"
```

### 响应格式
```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "page_size": 20,
    "total": 150,
    "total_pages": 8,
    "has_next": true,
    "has_prev": false
  }
}
```

## 异步任务处理

许多操作（如数据提取、批量标注）采用异步处理模式：

1. **提交任务**: 返回任务 ID
2. **查询状态**: 使用任务 ID 查询进度
3. **获取结果**: 任务完成后获取结果

### 任务状态
- `pending`: 等待处理
- `running`: 正在执行
- `completed`: 执行成功
- `failed`: 执行失败
- `cancelled`: 已取消

## 最佳实践

### 1. 错误重试
对于网络错误或临时服务不可用，建议实现指数退避重试：

```python
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_session_with_retries():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
```

### 2. 批量操作
对于大量数据处理，优先使用批量 API：

```python
# 推荐：批量提交
batch_request = {
    "task_ids": [task1_id, task2_id, task3_id],
    "ai_model_configs": [config1, config2, config3]
}
response = requests.post("/api/v1/ai/predict/batch", json=batch_request)

# 不推荐：逐个提交
for task_id in task_ids:
    response = requests.post("/api/v1/ai/predict", json={"task_id": task_id})
```

### 3. 缓存利用
AI 预标注支持智能缓存，相同内容的重复请求会直接返回缓存结果。

### 4. 监控和日志
生产环境建议监控以下指标：
- API 响应时间
- 错误率
- 任务成功率
- 资源使用情况

## 集成示例

### Python SDK 示例

```python
import requests
from typing import Dict, Any, List

class SuperInsightClient:
    def __init__(self, base_url: str, api_token: str = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        if api_token:
            self.session.headers.update({
                'Authorization': f'Bearer {api_token}'
            })
    
    def extract_from_database(self, config: Dict[str, Any]) -> str:
        """提交数据库提取任务"""
        response = self.session.post(
            f"{self.base_url}/api/v1/extraction/database",
            json=config
        )
        response.raise_for_status()
        return response.json()['job_id']
    
    def get_extraction_status(self, job_id: str) -> Dict[str, Any]:
        """查询提取任务状态"""
        response = self.session.get(
            f"{self.base_url}/api/v1/extraction/jobs/{job_id}"
        )
        response.raise_for_status()
        return response.json()
    
    def ai_predict_batch(self, task_ids: List[str], model_config: Dict[str, Any]) -> str:
        """批量 AI 预标注"""
        response = self.session.post(
            f"{self.base_url}/api/v1/ai/predict/batch",
            json={
                "task_ids": task_ids,
                "ai_model_configs": [model_config] * len(task_ids)
            }
        )
        response.raise_for_status()
        return response.json()['job_id']

# 使用示例
client = SuperInsightClient("http://localhost:8000")

# 数据库提取
db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "mydb",
    "username": "user",
    "password": "pass",
    "database_type": "postgresql",
    "limit": 1000
}
job_id = client.extract_from_database(db_config)

# 查询状态
status = client.get_extraction_status(job_id)
print(f"任务状态: {status['status']}")
```

### JavaScript/Node.js 示例

```javascript
class SuperInsightClient {
    constructor(baseUrl, apiToken = null) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.headers = {
            'Content-Type': 'application/json'
        };
        if (apiToken) {
            this.headers['Authorization'] = `Bearer ${apiToken}`;
        }
    }
    
    async extractFromDatabase(config) {
        const response = await fetch(`${this.baseUrl}/api/v1/extraction/database`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(config)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        return result.job_id;
    }
    
    async getExtractionStatus(jobId) {
        const response = await fetch(`${this.baseUrl}/api/v1/extraction/jobs/${jobId}`, {
            headers: this.headers
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
}

// 使用示例
const client = new SuperInsightClient('http://localhost:8000');

async function example() {
    try {
        // 提交数据库提取任务
        const jobId = await client.extractFromDatabase({
            host: 'localhost',
            port: 5432,
            database: 'mydb',
            username: 'user',
            password: 'pass',
            database_type: 'postgresql',
            limit: 1000
        });
        
        console.log(`任务已提交，ID: ${jobId}`);
        
        // 轮询任务状态
        let status;
        do {
            await new Promise(resolve => setTimeout(resolve, 2000)); // 等待2秒
            status = await client.getExtractionStatus(jobId);
            console.log(`任务状态: ${status.status}`);
        } while (status.status === 'pending' || status.status === 'running');
        
        if (status.status === 'completed') {
            console.log(`提取完成，共 ${status.documents_count} 个文档`);
        } else {
            console.error(`任务失败: ${status.error}`);
        }
        
    } catch (error) {
        console.error('操作失败:', error);
    }
}

example();
```

## 性能优化建议

### 1. 连接池
使用连接池减少连接开销：

```python
import requests
from requests.adapters import HTTPAdapter

session = requests.Session()
adapter = HTTPAdapter(pool_connections=10, pool_maxsize=20)
session.mount('http://', adapter)
session.mount('https://', adapter)
```

### 2. 并发控制
合理控制并发请求数量，避免服务器过载：

```python
import asyncio
import aiohttp

async def process_batch(session, items, max_concurrent=5):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_item(item):
        async with semaphore:
            async with session.post('/api/endpoint', json=item) as response:
                return await response.json()
    
    tasks = [process_item(item) for item in items]
    return await asyncio.gather(*tasks)
```

### 3. 缓存策略
对于不经常变化的数据，实现客户端缓存：

```python
import time
from functools import wraps

def cache_result(ttl_seconds=300):
    def decorator(func):
        cache = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            now = time.time()
            
            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < ttl_seconds:
                    return result
            
            result = func(*args, **kwargs)
            cache[key] = (result, now)
            return result
        
        return wrapper
    return decorator

@cache_result(ttl_seconds=600)  # 缓存10分钟
def get_model_list():
    response = requests.get('/api/v1/ai/models')
    return response.json()
```

## 故障排除

### 常见问题

1. **连接超时**
   - 检查网络连接
   - 增加超时时间
   - 使用重试机制

2. **认证失败**
   - 检查 API Token 是否正确
   - 确认 Token 未过期
   - 验证请求头格式

3. **任务一直处于 pending 状态**
   - 检查系统负载
   - 查看错误日志
   - 联系技术支持

4. **数据提取失败**
   - 验证数据源连接配置
   - 检查网络防火墙设置
   - 确认数据源权限

### 调试技巧

1. **启用详细日志**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **使用 curl 测试**
```bash
curl -v -X POST "http://localhost:8000/api/v1/extraction/database" \
  -H "Content-Type: application/json" \
  -d '{"host":"localhost","port":5432,"database":"test"}'
```

3. **检查系统状态**
```bash
curl "http://localhost:8000/system/status"
```

## 更新日志

### v1.0.0 (2024-01-15)
- 初始版本发布
- 支持数据提取、AI 标注、质量管理等核心功能
- 提供完整的 REST API

## 技术支持

- **文档**: [https://docs.superinsight.com](https://docs.superinsight.com)
- **GitHub**: [https://github.com/superinsight/platform](https://github.com/superinsight/platform)
- **邮箱**: support@superinsight.com
- **微信群**: 扫描二维码加入技术交流群

---

*本文档持续更新中，如有疑问请联系技术支持团队。*