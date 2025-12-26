# TCB 全栈部署系统 - 设计文档

## 概述

TCB 全栈部署系统为 SuperInsight 平台提供基于腾讯云 CloudBase 的 Serverless 部署解决方案。通过单一 Docker 镜像集成所有服务组件，实现一键部署、自动扩缩容和企业级运维能力。

## 架构设计

### 系统架构图

```mermaid
graph TB
    subgraph "TCB 云托管环境"
        subgraph "Docker 容器实例"
            SV[Supervisor 进程管理器]
            API[FastAPI 服务:8000]
            LS[Label Studio:8080]
            PG[PostgreSQL:5432]
            RD[Redis:6379]
            
            SV --> API
            SV --> LS
            SV --> PG
            SV --> RD
        end
        
        subgraph "TCB 基础设施"
            LB[负载均衡器]
            AS[自动扩缩容]
            MON[监控告警]
            LOG[日志收集]
        end
        
        subgraph "持久化存储"
            CBS[云硬盘存储]
            BK[自动备份]
        end
    end
    
    subgraph "外部访问"
        CDN[CDN 加速]
        DNS[自定义域名]
        SSL[SSL 证书]
    end
    
    subgraph "CI/CD 流水线"
        GH[GitHub Actions]
        TCR[容器镜像仓库]
        DEP[自动部署]
    end
    
    DNS --> CDN
    CDN --> LB
    LB --> API
    LB --> LS
    
    PG --> CBS
    RD --> CBS
    
    GH --> TCR
    TCR --> DEP
    DEP --> AS
    
    AS --> MON
    AS --> LOG
    CBS --> BK
```

### 核心组件设计

#### 1. 全栈 Docker 镜像

**多阶段构建策略:**
```dockerfile
# 阶段1: 基础环境
FROM ubuntu:22.04 as base
RUN apt-get update && apt-get install -y \
    python3.11 python3-pip \
    postgresql-14 redis-server \
    supervisor nginx

# 阶段2: 应用构建
FROM base as builder
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
WORKDIR /app

# 阶段3: 生产镜像
FROM base as production
COPY --from=builder /app /app
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY supervisord.conf /etc/supervisor/conf.d/
COPY nginx.conf /etc/nginx/
EXPOSE 8000 8080
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]
```

**进程管理配置:**
```ini
[supervisord]
nodaemon=true
user=root

[program:postgresql]
command=/usr/lib/postgresql/14/bin/postgres -D /var/lib/postgresql/14/main
user=postgres
autostart=true
autorestart=true
priority=100

[program:redis]
command=/usr/bin/redis-server /etc/redis/redis.conf
autostart=true
autorestart=true
priority=200

[program:fastapi]
command=python -m uvicorn main:app --host 0.0.0.0 --port 8000
directory=/app
autostart=true
autorestart=true
priority=300

[program:labelstudio]
command=label-studio start --host 0.0.0.0 --port 8080
directory=/app
autostart=true
autorestart=true
priority=400

[program:nginx]
command=/usr/sbin/nginx -g "daemon off;"
autostart=true
autorestart=true
priority=500
```

#### 2. TCB 配置管理

**cloudbaserc.json 配置:**
```json
{
  "version": "2.0",
  "envId": "superinsight-prod",
  "$schema": "https://framework-1258016615.tcloudbaseapp.com/schema/latest.json",
  "framework": {
    "name": "superinsight",
    "plugins": {
      "container": {
        "use": "@cloudbase/framework-plugin-container",
        "inputs": {
          "serviceName": "superinsight-app",
          "servicePath": "/superinsight",
          "dockerfilePath": "./Dockerfile",
          "buildDir": ".",
          "containerPort": 8000,
          "minNum": 1,
          "maxNum": 10,
          "cpu": 1,
          "mem": 2,
          "policyType": "cpu",
          "policyThreshold": 70,
          "envVariables": {
            "DATABASE_URL": "postgresql://postgres:password@localhost:5432/superinsight",
            "REDIS_URL": "redis://localhost:6379/0",
            "LABEL_STUDIO_URL": "http://localhost:8080",
            "SECRET_KEY": "${env.SECRET_KEY}",
            "TCB_ENV_ID": "${env.TCB_ENV_ID}"
          },
          "customLogs": "stdout",
          "dataBaseName": "superinsight",
          "vpc": {
            "vpcId": "${env.VPC_ID}",
            "subnetId": "${env.SUBNET_ID}"
          }
        }
      }
    }
  }
}
```

#### 3. 持久化存储设计

**存储卷挂载策略:**
```yaml
# TCB 存储配置
storage:
  - name: postgres-data
    type: cbs
    size: 50Gi
    mountPath: /var/lib/postgresql/14/main
    
  - name: redis-data
    type: cbs
    size: 10Gi
    mountPath: /var/lib/redis
    
  - name: app-logs
    type: cbs
    size: 20Gi
    mountPath: /app/logs
    
  - name: label-studio-data
    type: cbs
    size: 100Gi
    mountPath: /app/label-studio-data
```

**数据备份策略:**
```python
# 自动备份脚本
import os
import subprocess
from datetime import datetime

class TCBBackupManager:
    def __init__(self):
        self.backup_schedule = {
            'postgres': '0 2 * * *',  # 每日凌晨2点
            'redis': '0 */6 * * *',   # 每6小时
            'files': '0 1 * * 0'      # 每周日凌晨1点
        }
    
    def create_postgres_backup(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = f"/backup/postgres_{timestamp}.sql"
        
        cmd = [
            'pg_dump',
            '-h', 'localhost',
            '-U', 'postgres',
            '-d', 'superinsight',
            '-f', backup_file
        ]
        
        subprocess.run(cmd, check=True)
        self.upload_to_cos(backup_file)
    
    def create_redis_backup(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = f"/backup/redis_{timestamp}.rdb"
        
        subprocess.run(['redis-cli', 'BGSAVE'], check=True)
        subprocess.run(['cp', '/var/lib/redis/dump.rdb', backup_file], check=True)
        self.upload_to_cos(backup_file)
    
    def upload_to_cos(self, file_path):
        # 上传到腾讯云对象存储
        pass
```

#### 4. 监控和告警系统

**监控指标收集:**
```python
from prometheus_client import Counter, Histogram, Gauge
import psutil
import redis
import psycopg2

class TCBMonitoring:
    def __init__(self):
        # 业务指标
        self.request_count = Counter('http_requests_total', 'Total HTTP requests')
        self.request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration')
        
        # 系统指标
        self.cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
        self.memory_usage = Gauge('system_memory_usage_percent', 'Memory usage percentage')
        self.disk_usage = Gauge('system_disk_usage_percent', 'Disk usage percentage')
        
        # 服务指标
        self.postgres_connections = Gauge('postgres_active_connections', 'Active PostgreSQL connections')
        self.redis_memory = Gauge('redis_memory_usage_bytes', 'Redis memory usage')
    
    def collect_system_metrics(self):
        self.cpu_usage.set(psutil.cpu_percent())
        self.memory_usage.set(psutil.virtual_memory().percent)
        self.disk_usage.set(psutil.disk_usage('/').percent)
    
    def collect_service_metrics(self):
        # PostgreSQL 指标
        try:
            conn = psycopg2.connect("postgresql://postgres:password@localhost:5432/superinsight")
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")
            active_connections = cur.fetchone()[0]
            self.postgres_connections.set(active_connections)
            conn.close()
        except Exception as e:
            print(f"Failed to collect PostgreSQL metrics: {e}")
        
        # Redis 指标
        try:
            r = redis.Redis(host='localhost', port=6379, db=0)
            info = r.info('memory')
            self.redis_memory.set(info['used_memory'])
        except Exception as e:
            print(f"Failed to collect Redis metrics: {e}")
```

**告警规则配置:**
```yaml
# TCB 告警配置
alerts:
  - name: high_cpu_usage
    condition: cpu_usage > 80
    duration: 5m
    action: scale_up
    
  - name: high_memory_usage
    condition: memory_usage > 85
    duration: 3m
    action: scale_up
    
  - name: database_connection_limit
    condition: postgres_connections > 90
    duration: 1m
    action: alert_admin
    
  - name: disk_space_low
    condition: disk_usage > 90
    duration: 1m
    action: alert_admin
    
  - name: service_down
    condition: service_health == 0
    duration: 30s
    action: restart_service
```

#### 5. CI/CD 流水线设计

**GitHub Actions 工作流:**
```yaml
name: TCB Deployment Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest --cov=src tests/
      
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: |
          docker build -t superinsight:${{ github.sha }} .
          docker tag superinsight:${{ github.sha }} superinsight:latest
      - name: Push to TCB Registry
        run: |
          echo ${{ secrets.TCB_REGISTRY_PASSWORD }} | docker login -u ${{ secrets.TCB_REGISTRY_USERNAME }} --password-stdin
          docker push superinsight:${{ github.sha }}
          docker push superinsight:latest
  
  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - name: Install CloudBase CLI
        run: npm install -g @cloudbase/cli
      - name: Deploy to TCB
        run: |
          tcb login --apiKeyId ${{ secrets.TCB_API_KEY_ID }} --apiKey ${{ secrets.TCB_API_KEY }}
          tcb framework deploy --envId ${{ secrets.TCB_ENV_ID }}
      - name: Health Check
        run: |
          sleep 60
          curl -f https://superinsight.example.com/health || exit 1
```

## 数据模型

### TCB 配置模型

```python
from pydantic import BaseModel
from typing import Dict, List, Optional

class TCBResourceConfig(BaseModel):
    cpu: float = 1.0
    memory: int = 2  # GB
    min_instances: int = 1
    max_instances: int = 10
    scale_threshold: int = 70  # CPU 使用率阈值

class TCBStorageConfig(BaseModel):
    name: str
    type: str = "cbs"
    size: str = "50Gi"
    mount_path: str

class TCBNetworkConfig(BaseModel):
    vpc_id: Optional[str] = None
    subnet_id: Optional[str] = None
    security_group_id: Optional[str] = None

class TCBServiceConfig(BaseModel):
    service_name: str
    service_path: str
    container_port: int = 8000
    resource: TCBResourceConfig
    storage: List[TCBStorageConfig]
    network: TCBNetworkConfig
    env_variables: Dict[str, str]
    
class TCBDeploymentConfig(BaseModel):
    env_id: str
    region: str = "ap-shanghai"
    services: List[TCBServiceConfig]
    domain: Optional[str] = None
    ssl_cert_id: Optional[str] = None
```

### 监控数据模型

```python
class SystemMetrics(BaseModel):
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_in: int
    network_out: int

class ServiceMetrics(BaseModel):
    timestamp: datetime
    service_name: str
    response_time: float
    request_count: int
    error_rate: float
    active_connections: int

class AlertRule(BaseModel):
    name: str
    metric: str
    condition: str  # e.g., "> 80"
    duration: str   # e.g., "5m"
    action: str     # e.g., "scale_up"
    enabled: bool = True
```

## 正确性属性

*正确性属性是系统必须满足的形式化规范，通过属性测试验证系统行为的正确性。*

### 属性 1: 容器启动顺序正确性
*对于任何* TCB 部署实例，PostgreSQL 和 Redis 服务必须在 FastAPI 和 Label Studio 服务之前完全启动并就绪
**验证: 需求 1.2, 1.3**

### 属性 2: 自动扩缩容一致性
*对于任何* CPU 使用率超过阈值的情况，系统应在指定时间内启动新实例，且新实例应与现有实例具有相同的配置
**验证: 需求 2.3, 2.4**

### 属性 3: 数据持久化完整性
*对于任何* 容器重启操作，持久化存储中的数据应保持完整且可访问
**验证: 需求 3.1, 3.2**

### 属性 4: 部署回滚安全性
*对于任何* 部署失败的情况，系统应能够安全回滚到上一个稳定版本，且不丢失数据
**验证: 需求 4.4**

### 属性 5: 监控指标准确性
*对于任何* 系统运行状态，监控系统收集的指标应准确反映实际的资源使用情况
**验证: 需求 5.1, 5.4**

### 属性 6: 成本计算正确性
*对于任何* 资源使用情况，成本计算应准确反映实际的 TCB 服务费用
**验证: 需求 6.1, 6.2**

### 属性 7: 安全访问控制
*对于任何* 外部访问请求，系统应正确验证身份并执行访问控制策略
**验证: 需求 7.1, 7.2**

### 属性 8: 高可用故障转移
*对于任何* 单点故障情况，系统应在指定时间内完成故障转移且服务不中断
**验证: 需求 8.1, 8.2**

## 错误处理

### 部署错误处理
- **镜像构建失败**: 自动重试3次，失败后通知开发团队
- **服务启动失败**: 检查依赖服务状态，按优先级重启
- **健康检查失败**: 自动重启容器，连续失败则标记为不健康

### 运行时错误处理
- **资源不足**: 自动扩容或降级服务
- **网络异常**: 重试机制和熔断保护
- **存储故障**: 自动切换到备用存储

### 数据错误处理
- **数据损坏**: 从最近备份恢复
- **同步失败**: 重新同步并记录冲突
- **备份失败**: 告警并尝试手动备份

## 测试策略

### 单元测试
- TCB 配置解析和验证
- Docker 镜像构建脚本
- 监控指标收集逻辑
- 自动扩缩容算法

### 集成测试
- 完整部署流程测试
- 服务间通信测试
- 持久化存储测试
- 监控告警测试

### 性能测试
- 容器启动时间测试
- 自动扩缩容响应时间
- 并发访问压力测试
- 资源使用效率测试

### 属性测试
- 使用 Hypothesis 生成随机配置测试部署一致性
- 模拟各种故障场景测试恢复能力
- 验证监控数据的准确性和完整性
- 测试成本计算的正确性

每个属性测试运行最少 100 次迭代，确保在各种输入条件下系统行为的正确性。