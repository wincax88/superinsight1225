# 数据提取 API 详细文档

## 概述

数据提取 API 提供安全、高效的数据源连接和内容提取功能，支持数据库、文件、网页和 API 等多种数据源。

**基础路径**: `/api/v1/extraction`

## 支持的数据源

### 数据库
- MySQL 5.7+
- PostgreSQL 10+
- Oracle 11g+

### 文件格式
- PDF 文档
- Word 文档 (.docx)
- 纯文本 (.txt)
- HTML 文件

### 网页内容
- 静态网页爬取
- 深度爬取支持
- 智能内容提取

### API 接口
- RESTful API
- 分页数据获取
- 认证支持

## 端点详情

### 1. 数据库提取

#### POST `/api/v1/extraction/database`

从数据库中提取数据，支持表级别和自定义 SQL 查询。

**请求体**:
```json
{
  "host": "localhost",
  "port": 5432,
  "database": "mydb",
  "username": "user",
  "password": "password",
  "database_type": "postgresql",
  "table_name": "documents",
  "query": "SELECT * FROM documents WHERE created_at > '2024-01-01'",
  "limit": 1000,
  "use_ssl": true
}
```

**参数说明**:
- `host` (必需): 数据库主机地址
- `port` (必需): 数据库端口
- `database` (必需): 数据库名称
- `username` (必需): 用户名
- `password` (必需): 密码
- `database_type` (必需): 数据库类型 (`mysql`, `postgresql`, `oracle`)
- `table_name` (可选): 指定表名
- `query` (可选): 自定义 SQL 查询
- `limit` (可选): 最大记录数，默认 100，最大 10000
- `use_ssl` (可选): 是否使用 SSL，默认 true

**响应示例**:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "created_at": "2024-01-15T10:30:00Z",
  "message": "Database extraction job started"
}
```

**使用示例**:

```bash
# 基础表提取
curl -X POST "http://localhost:8000/api/v1/extraction/database" \
  -H "Content-Type: application/json" \
  -d '{
    "host": "localhost",
    "port": 5432,
    "database": "mydb",
    "username": "user",
    "password": "pass",
    "database_type": "postgresql",
    "table_name": "articles",
    "limit": 500
  }'

# 自定义查询
curl -X POST "http://localhost:8000/api/v1/extraction/database" \
  -H "Content-Type: application/json" \
  -d '{
    "host": "localhost",
    "port": 3306,
    "database": "content_db",
    "username": "reader",
    "password": "readonly123",
    "database_type": "mysql",
    "query": "SELECT title, content, created_at FROM articles WHERE status = \"published\" ORDER BY created_at DESC",
    "limit": 1000
  }'
```

```python
# Python 示例
import requests

config = {
    "host": "localhost",
    "port": 5432,
    "database": "mydb",
    "username": "user",
    "password": "password",
    "database_type": "postgresql",
    "table_name": "documents",
    "limit": 1000
}

response = requests.post(
    "http://localhost:8000/api/v1/extraction/database",
    json=config
)

if response.status_code == 200:
    job_id = response.json()["job_id"]
    print(f"提取任务已启动，任务ID: {job_id}")
else:
    print(f"请求失败: {response.text}")
```

### 2. 文件提取

#### POST `/api/v1/extraction/file`

从文件中提取文本内容，支持本地文件和远程 URL。

**请求体**:
```json
{
  "file_path": "/path/to/document.pdf",
  "file_type": "pdf",
  "encoding": "utf-8",
  "use_ssl": true
}
```

**参数说明**:
- `file_path` (必需): 文件路径或 URL
- `file_type` (可选): 文件类型 (`pdf`, `docx`, `txt`, `html`)
- `encoding` (可选): 文件编码，默认 `utf-8`
- `use_ssl` (可选): URL 是否使用 SSL，默认 true

**使用示例**:

```bash
# PDF 文件提取
curl -X POST "http://localhost:8000/api/v1/extraction/file" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/data/reports/annual_report_2023.pdf",
    "file_type": "pdf"
  }'

# 远程文档提取
curl -X POST "http://localhost:8000/api/v1/extraction/file" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "https://example.com/documents/whitepaper.pdf",
    "file_type": "pdf",
    "use_ssl": true
  }'
```

```python
# Python 批量文件处理示例
import requests
import os

def extract_files_from_directory(directory_path, file_extensions=['.pdf', '.docx']):
    results = []
    
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if any(file.endswith(ext) for ext in file_extensions):
                file_path = os.path.join(root, file)
                file_type = file.split('.')[-1].lower()
                
                config = {
                    "file_path": file_path,
                    "file_type": file_type
                }
                
                response = requests.post(
                    "http://localhost:8000/api/v1/extraction/file",
                    json=config
                )
                
                if response.status_code == 200:
                    job_id = response.json()["job_id"]
                    results.append({"file": file_path, "job_id": job_id})
                    print(f"已提交文件: {file_path}, 任务ID: {job_id}")
                else:
                    print(f"文件提取失败: {file_path}, 错误: {response.text}")
    
    return results

# 使用示例
results = extract_files_from_directory("/data/documents")
```

### 3. 网页提取

#### POST `/api/v1/extraction/web`

从网站爬取内容，支持深度爬取和智能内容提取。

**请求体**:
```json
{
  "base_url": "https://example.com",
  "max_pages": 50,
  "max_depth": 3
}
```

**参数说明**:
- `base_url` (必需): 起始 URL
- `max_pages` (可选): 最大页面数，默认 10，最大 100
- `max_depth` (可选): 最大爬取深度，默认 2，最大 5

**使用示例**:

```bash
# 基础网页爬取
curl -X POST "http://localhost:8000/api/v1/extraction/web" \
  -H "Content-Type: application/json" \
  -d '{
    "base_url": "https://blog.example.com",
    "max_pages": 20,
    "max_depth": 2
  }'
```

```python
# Python 示例 - 新闻网站内容提取
import requests
import time

def crawl_news_site(base_url, max_articles=100):
    config = {
        "base_url": base_url,
        "max_pages": max_articles,
        "max_depth": 2
    }
    
    # 提交爬取任务
    response = requests.post(
        "http://localhost:8000/api/v1/extraction/web",
        json=config
    )
    
    if response.status_code != 200:
        print(f"任务提交失败: {response.text}")
        return None
    
    job_id = response.json()["job_id"]
    print(f"网页爬取任务已启动，任务ID: {job_id}")
    
    # 轮询任务状态
    while True:
        status_response = requests.get(
            f"http://localhost:8000/api/v1/extraction/jobs/{job_id}"
        )
        
        if status_response.status_code != 200:
            print("状态查询失败")
            break
        
        status_data = status_response.json()
        print(f"任务状态: {status_data['status']}")
        
        if status_data["status"] == "completed":
            print(f"爬取完成，共获取 {status_data['documents_count']} 个页面")
            return status_data["documents"]
        elif status_data["status"] == "failed":
            print(f"爬取失败: {status_data['error']}")
            break
        
        time.sleep(5)  # 等待5秒后再次查询
    
    return None

# 使用示例
articles = crawl_news_site("https://news.example.com", max_articles=50)
```

### 4. API 提取

#### POST `/api/v1/extraction/api`

从 REST API 获取数据，支持认证和分页。

**请求体**:
```json
{
  "base_url": "https://api.example.com",
  "endpoint": "/v1/articles",
  "method": "GET",
  "headers": {
    "Content-Type": "application/json",
    "User-Agent": "SuperInsight/1.0"
  },
  "auth_token": "your_api_token",
  "params": {
    "category": "technology",
    "limit": 100
  },
  "paginate": true,
  "use_ssl": true
}
```

**参数说明**:
- `base_url` (必需): API 基础 URL
- `endpoint` (可选): API 端点，默认为空
- `method` (可选): HTTP 方法，支持 GET/POST，默认 GET
- `headers` (可选): HTTP 请求头
- `auth_token` (可选): 认证令牌
- `params` (可选): 查询参数
- `data` (可选): POST 请求体数据
- `paginate` (可选): 是否启用分页，默认 true
- `use_ssl` (可选): 是否使用 SSL，默认 true

**使用示例**:

```bash
# GET 请求示例
curl -X POST "http://localhost:8000/api/v1/extraction/api" \
  -H "Content-Type: application/json" \
  -d '{
    "base_url": "https://jsonplaceholder.typicode.com",
    "endpoint": "/posts",
    "method": "GET",
    "params": {
      "userId": 1
    }
  }'

# 带认证的 API 调用
curl -X POST "http://localhost:8000/api/v1/extraction/api" \
  -H "Content-Type: application/json" \
  -d '{
    "base_url": "https://api.github.com",
    "endpoint": "/user/repos",
    "method": "GET",
    "headers": {
      "Accept": "application/vnd.github.v3+json"
    },
    "auth_token": "ghp_your_token_here",
    "params": {
      "type": "owner",
      "sort": "updated"
    }
  }'
```

```python
# Python 示例 - 从多个 API 源提取数据
import requests
import concurrent.futures

def extract_from_api(api_config):
    """从单个 API 提取数据"""
    response = requests.post(
        "http://localhost:8000/api/v1/extraction/api",
        json=api_config
    )
    
    if response.status_code == 200:
        return response.json()["job_id"]
    else:
        print(f"API 提取失败: {response.text}")
        return None

def extract_from_multiple_apis(api_configs):
    """并行从多个 API 提取数据"""
    job_ids = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_config = {
            executor.submit(extract_from_api, config): config 
            for config in api_configs
        }
        
        for future in concurrent.futures.as_completed(future_to_config):
            config = future_to_config[future]
            try:
                job_id = future.result()
                if job_id:
                    job_ids.append(job_id)
                    print(f"API 任务已提交: {config['base_url']}, 任务ID: {job_id}")
            except Exception as exc:
                print(f"API 提取异常: {config['base_url']}, 错误: {exc}")
    
    return job_ids

# 配置多个 API 源
api_configs = [
    {
        "base_url": "https://api.github.com",
        "endpoint": "/search/repositories",
        "params": {"q": "machine learning", "sort": "stars"}
    },
    {
        "base_url": "https://hacker-news.firebaseio.com",
        "endpoint": "/v0/topstories.json"
    },
    {
        "base_url": "https://jsonplaceholder.typicode.com",
        "endpoint": "/posts"
    }
]

# 执行并行提取
job_ids = extract_from_multiple_apis(api_configs)
print(f"共提交了 {len(job_ids)} 个 API 提取任务")
```

### 5. 任务管理

#### GET `/api/v1/extraction/jobs/{job_id}`

查询特定提取任务的状态和结果。

**路径参数**:
- `job_id`: 任务唯一标识符

**响应示例**:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "success": true,
  "documents_count": 150,
  "error": null,
  "documents": [
    {
      "id": "doc_001",
      "content": "文档内容...",
      "metadata": {
        "source": "database",
        "table": "articles",
        "extracted_at": "2024-01-15T10:35:00Z"
      }
    }
  ],
  "created_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:35:00Z"
}
```

#### GET `/api/v1/extraction/jobs`

列出所有提取任务。

**查询参数**:
- `page`: 页码，默认 1
- `page_size`: 每页大小，默认 20

**使用示例**:

```python
# Python 示例 - 任务状态监控
import requests
import time

def monitor_extraction_job(job_id, timeout=300):
    """监控提取任务直到完成"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        response = requests.get(
            f"http://localhost:8000/api/v1/extraction/jobs/{job_id}"
        )
        
        if response.status_code != 200:
            print(f"状态查询失败: {response.text}")
            return None
        
        job_data = response.json()
        status = job_data["status"]
        
        print(f"任务 {job_id} 状态: {status}")
        
        if status == "completed":
            print(f"任务完成，提取了 {job_data['documents_count']} 个文档")
            return job_data
        elif status == "failed":
            print(f"任务失败: {job_data['error']}")
            return None
        
        time.sleep(10)  # 每10秒检查一次
    
    print("任务监控超时")
    return None

# 使用示例
job_data = monitor_extraction_job("your_job_id_here")
if job_data and job_data["success"]:
    documents = job_data["documents"]
    print(f"成功获取 {len(documents)} 个文档")
```

#### POST `/api/v1/extraction/jobs/{job_id}/save`

将提取结果保存到数据库。

**使用示例**:

```bash
curl -X POST "http://localhost:8000/api/v1/extraction/jobs/550e8400-e29b-41d4-a716-446655440000/save"
```

#### DELETE `/api/v1/extraction/jobs/{job_id}`

删除提取任务及其结果。

### 6. 连接测试

#### POST `/api/v1/extraction/test-connection`

测试数据源连接，不执行实际提取。

**请求体**: 与对应提取端点相同的配置

**响应示例**:
```json
{
  "success": true,
  "message": "Connection successful"
}
```

**使用示例**:

```python
# Python 示例 - 批量连接测试
import requests

def test_database_connections(db_configs):
    """测试多个数据库连接"""
    results = []
    
    for config in db_configs:
        response = requests.post(
            "http://localhost:8000/api/v1/extraction/test-connection",
            json=config
        )
        
        result = {
            "host": config["host"],
            "database": config["database"],
            "success": False,
            "message": ""
        }
        
        if response.status_code == 200:
            data = response.json()
            result["success"] = data["success"]
            result["message"] = data["message"]
        else:
            result["message"] = f"HTTP {response.status_code}: {response.text}"
        
        results.append(result)
        print(f"数据库 {config['host']}/{config['database']}: {'✓' if result['success'] else '✗'} {result['message']}")
    
    return results

# 测试配置
db_configs = [
    {
        "host": "localhost",
        "port": 5432,
        "database": "prod_db",
        "username": "reader",
        "password": "readonly",
        "database_type": "postgresql"
    },
    {
        "host": "mysql.example.com",
        "port": 3306,
        "database": "analytics",
        "username": "analyst",
        "password": "secret",
        "database_type": "mysql"
    }
]

# 执行测试
test_results = test_database_connections(db_configs)
```

## 错误处理

### 常见错误场景

1. **数据库连接失败**
```json
{
  "error": "Database Connection Error",
  "message": "Could not connect to database: Connection refused",
  "details": {
    "host": "localhost",
    "port": 5432,
    "database": "mydb"
  }
}
```

2. **文件访问失败**
```json
{
  "error": "File Access Error",
  "message": "File not found or permission denied",
  "details": {
    "file_path": "/path/to/file.pdf"
  }
}
```

3. **网页爬取失败**
```json
{
  "error": "Web Crawling Error",
  "message": "HTTP 403: Access forbidden",
  "details": {
    "url": "https://example.com/page"
  }
}
```

### 错误处理最佳实践

```python
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

def robust_extraction_request(config, max_retries=3):
    """健壮的提取请求处理"""
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:8000/api/v1/extraction/database",
                json=config,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 400:
                # 客户端错误，不重试
                print(f"请求参数错误: {response.text}")
                return None
            elif response.status_code >= 500:
                # 服务器错误，可以重试
                print(f"服务器错误 (尝试 {attempt + 1}/{max_retries}): {response.text}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2 ** attempt)  # 指数退避
            else:
                print(f"未知错误: HTTP {response.status_code}")
                return None
                
        except Timeout:
            print(f"请求超时 (尝试 {attempt + 1}/{max_retries})")
            if attempt == max_retries - 1:
                return None
            time.sleep(2 ** attempt)
            
        except ConnectionError:
            print(f"连接错误 (尝试 {attempt + 1}/{max_retries})")
            if attempt == max_retries - 1:
                return None
            time.sleep(2 ** attempt)
            
        except RequestException as e:
            print(f"请求异常: {e}")
            return None
    
    return None
```

## 性能优化

### 1. 批量处理策略

```python
import asyncio
import aiohttp

async def batch_extract_files(file_paths, max_concurrent=5):
    """异步批量文件提取"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def extract_single_file(session, file_path):
        async with semaphore:
            config = {
                "file_path": file_path,
                "file_type": file_path.split('.')[-1].lower()
            }
            
            async with session.post(
                "http://localhost:8000/api/v1/extraction/file",
                json=config
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {"file": file_path, "job_id": data["job_id"]}
                else:
                    return {"file": file_path, "error": await response.text()}
    
    async with aiohttp.ClientSession() as session:
        tasks = [extract_single_file(session, fp) for fp in file_paths]
        results = await asyncio.gather(*tasks)
    
    return results

# 使用示例
file_paths = ["/data/doc1.pdf", "/data/doc2.pdf", "/data/doc3.pdf"]
results = asyncio.run(batch_extract_files(file_paths))
```

### 2. 智能重试机制

```python
import time
import random

def exponential_backoff_retry(func, max_retries=3, base_delay=1):
    """指数退避重试装饰器"""
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"重试 {attempt + 1}/{max_retries}，等待 {delay:.2f} 秒")
                time.sleep(delay)
        
        return None
    return wrapper

@exponential_backoff_retry(max_retries=3)
def submit_extraction_job(config):
    response = requests.post(
        "http://localhost:8000/api/v1/extraction/database",
        json=config
    )
    response.raise_for_status()
    return response.json()
```

## 安全注意事项

### 1. 敏感信息保护

```python
import os
from cryptography.fernet import Fernet

class SecureConfig:
    def __init__(self):
        # 从环境变量获取加密密钥
        key = os.environ.get('ENCRYPTION_KEY')
        if not key:
            key = Fernet.generate_key()
            print(f"生成新的加密密钥: {key.decode()}")
        
        self.cipher = Fernet(key.encode() if isinstance(key, str) else key)
    
    def encrypt_password(self, password):
        """加密密码"""
        return self.cipher.encrypt(password.encode()).decode()
    
    def decrypt_password(self, encrypted_password):
        """解密密码"""
        return self.cipher.decrypt(encrypted_password.encode()).decode()
    
    def create_secure_config(self, host, username, password, **kwargs):
        """创建安全配置"""
        config = {
            "host": host,
            "username": username,
            "password": self.encrypt_password(password),
            **kwargs
        }
        return config

# 使用示例
secure_config = SecureConfig()
config = secure_config.create_secure_config(
    host="localhost",
    username="user",
    password="sensitive_password",
    database="mydb",
    database_type="postgresql"
)
```

### 2. 访问控制

```python
def validate_extraction_request(config):
    """验证提取请求的安全性"""
    
    # 检查主机白名单
    allowed_hosts = ["localhost", "127.0.0.1", "internal.company.com"]
    if config.get("host") not in allowed_hosts:
        raise ValueError(f"主机 {config['host']} 不在允许列表中")
    
    # 检查端口范围
    port = config.get("port", 0)
    if not (1024 <= port <= 65535):
        raise ValueError(f"端口 {port} 不在允许范围内")
    
    # 检查文件路径
    if "file_path" in config:
        file_path = config["file_path"]
        if ".." in file_path or file_path.startswith("/"):
            raise ValueError("不允许的文件路径")
    
    return True

# 使用示例
try:
    validate_extraction_request(config)
    # 继续处理请求
except ValueError as e:
    print(f"安全验证失败: {e}")
```

---

*更多详细信息请参考 [API 总览文档](README.md) 或联系技术支持。*