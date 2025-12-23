# API 错误码和故障排除指南

## 概述

本文档详细说明了 SuperInsight 平台 API 的错误码、错误处理机制和常见问题的解决方案。

## 错误响应格式

所有 API 错误都遵循统一的响应格式：

```json
{
  "error": "错误类型标识",
  "error_id": "唯一错误标识符",
  "message": "人类可读的错误描述",
  "details": {
    "field": "具体字段错误信息",
    "code": "内部错误码",
    "context": "错误上下文信息"
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_550e8400-e29b-41d4-a716-446655440000"
}
```

## HTTP 状态码

### 2xx 成功状态码

| 状态码 | 含义 | 说明 |
|--------|------|------|
| 200 | OK | 请求成功 |
| 201 | Created | 资源创建成功 |
| 202 | Accepted | 请求已接受，异步处理中 |
| 204 | No Content | 请求成功，无返回内容 |

### 4xx 客户端错误

| 状态码 | 错误类型 | 含义 | 常见原因 |
|--------|----------|------|----------|
| 400 | Bad Request | 请求参数错误 | 参数格式错误、必需参数缺失 |
| 401 | Unauthorized | 认证失败 | API Token 无效或过期 |
| 403 | Forbidden | 权限不足 | 用户无权限访问资源 |
| 404 | Not Found | 资源不存在 | 请求的资源不存在 |
| 409 | Conflict | 资源冲突 | 资源已存在或状态冲突 |
| 422 | Unprocessable Entity | 实体无法处理 | 数据验证失败 |
| 429 | Too Many Requests | 请求频率超限 | 超过 API 调用限制 |

### 5xx 服务器错误

| 状态码 | 错误类型 | 含义 | 常见原因 |
|--------|----------|------|----------|
| 500 | Internal Server Error | 服务器内部错误 | 系统异常、代码错误 |
| 502 | Bad Gateway | 网关错误 | 上游服务不可用 |
| 503 | Service Unavailable | 服务不可用 | 系统维护、过载 |
| 504 | Gateway Timeout | 网关超时 | 上游服务响应超时 |

## 详细错误码

### 数据提取错误 (EXT_xxx)

#### EXT_001 - 数据库连接失败
```json
{
  "error": "Database Connection Error",
  "error_id": "EXT_001",
  "message": "无法连接到数据库",
  "details": {
    "host": "localhost",
    "port": 5432,
    "database": "mydb",
    "error_code": "CONNECTION_REFUSED"
  }
}
```

**解决方案**:
1. 检查数据库服务是否运行
2. 验证连接参数（主机、端口、数据库名）
3. 检查网络连接和防火墙设置
4. 确认用户名和密码正确

#### EXT_002 - 认证失败
```json
{
  "error": "Database Authentication Error",
  "error_id": "EXT_002",
  "message": "数据库认证失败",
  "details": {
    "username": "user",
    "error_code": "INVALID_CREDENTIALS"
  }
}
```

**解决方案**:
1. 验证用户名和密码
2. 检查用户是否有数据库访问权限
3. 确认用户账户未被锁定

#### EXT_003 - 权限不足
```json
{
  "error": "Database Permission Error",
  "error_id": "EXT_003",
  "message": "用户无权限访问指定表或执行查询",
  "details": {
    "table_name": "sensitive_data",
    "required_permission": "SELECT"
  }
}
```

**解决方案**:
1. 为用户授予必要的数据库权限
2. 使用具有适当权限的用户账户
3. 检查表级别的权限设置

#### EXT_004 - 文件不存在
```json
{
  "error": "File Not Found Error",
  "error_id": "EXT_004",
  "message": "指定的文件不存在或无法访问",
  "details": {
    "file_path": "/path/to/document.pdf",
    "error_code": "FILE_NOT_FOUND"
  }
}
```

**解决方案**:
1. 检查文件路径是否正确
2. 确认文件存在且可读
3. 检查文件权限设置

#### EXT_005 - 文件格式不支持
```json
{
  "error": "Unsupported File Format",
  "error_id": "EXT_005",
  "message": "不支持的文件格式",
  "details": {
    "file_path": "/path/to/document.xyz",
    "detected_format": "xyz",
    "supported_formats": ["pdf", "docx", "txt", "html"]
  }
}
```

**解决方案**:
1. 使用支持的文件格式
2. 转换文件到支持的格式
3. 联系技术支持添加新格式支持

### AI 预标注错误 (AI_xxx)

#### AI_001 - 模型不可用
```json
{
  "error": "Model Unavailable",
  "error_id": "AI_001",
  "message": "指定的 AI 模型不可用",
  "details": {
    "model_type": "zhipu_glm",
    "model_name": "glm-4",
    "error_code": "MODEL_NOT_ACCESSIBLE"
  }
}
```

**解决方案**:
1. 检查模型配置是否正确
2. 验证 API Key 是否有效
3. 确认模型服务是否正常运行
4. 检查网络连接

#### AI_002 - API 密钥无效
```json
{
  "error": "Invalid API Key",
  "error_id": "AI_002",
  "message": "API 密钥无效或已过期",
  "details": {
    "model_type": "zhipu_glm",
    "error_code": "INVALID_API_KEY"
  }
}
```

**解决方案**:
1. 检查 API Key 是否正确
2. 确认 API Key 未过期
3. 重新生成 API Key
4. 检查 API Key 权限范围

#### AI_003 - 配额超限
```json
{
  "error": "Quota Exceeded",
  "error_id": "AI_003",
  "message": "API 调用配额已超限",
  "details": {
    "model_type": "zhipu_glm",
    "quota_limit": 1000,
    "quota_used": 1000,
    "reset_time": "2024-01-16T00:00:00Z"
  }
}
```

**解决方案**:
1. 等待配额重置时间
2. 升级 API 套餐
3. 优化请求频率
4. 使用其他可用模型

#### AI_004 - 请求超时
```json
{
  "error": "Request Timeout",
  "error_id": "AI_004",
  "message": "模型请求超时",
  "details": {
    "timeout_seconds": 30,
    "model_info": {
      "model_type": "zhipu_glm",
      "model_name": "glm-4"
    }
  }
}
```

**解决方案**:
1. 增加请求超时时间
2. 简化输入内容
3. 使用更快的模型
4. 检查网络连接稳定性

### 质量管理错误 (QM_xxx)

#### QM_001 - 质量规则不存在
```json
{
  "error": "Quality Rule Not Found",
  "error_id": "QM_001",
  "message": "指定的质量规则不存在",
  "details": {
    "rule_id": "rule_123",
    "project_id": "project_456"
  }
}
```

#### QM_002 - 质量评估失败
```json
{
  "error": "Quality Assessment Failed",
  "error_id": "QM_002",
  "message": "质量评估过程中发生错误",
  "details": {
    "task_id": "task_789",
    "assessment_type": "semantic",
    "error_code": "RAGAS_SERVICE_ERROR"
  }
}
```

### 计费系统错误 (BILL_xxx)

#### BILL_001 - 计费规则未设置
```json
{
  "error": "Billing Rule Not Set",
  "error_id": "BILL_001",
  "message": "租户未设置计费规则",
  "details": {
    "tenant_id": "tenant_123"
  }
}
```

#### BILL_002 - 计费数据不完整
```json
{
  "error": "Incomplete Billing Data",
  "error_id": "BILL_002",
  "message": "计费数据不完整，无法生成账单",
  "details": {
    "tenant_id": "tenant_123",
    "missing_fields": ["annotation_count", "time_spent"]
  }
}
```

### 安全控制错误 (SEC_xxx)

#### SEC_001 - 权限验证失败
```json
{
  "error": "Permission Denied",
  "error_id": "SEC_001",
  "message": "用户无权限执行此操作",
  "details": {
    "user_id": "user_123",
    "required_permission": "project:read",
    "resource_id": "project_456"
  }
}
```

#### SEC_002 - IP 地址被拒绝
```json
{
  "error": "IP Address Blocked",
  "error_id": "SEC_002",
  "message": "请求 IP 地址不在白名单中",
  "details": {
    "client_ip": "192.168.1.100",
    "allowed_ips": ["192.168.1.1", "192.168.1.2"]
  }
}
```

## 故障排除工具

### 1. 健康检查脚本

```python
#!/usr/bin/env python3
"""
SuperInsight 平台健康检查脚本
"""

import requests
import json
import sys
from datetime import datetime

class HealthChecker:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.results = []
    
    def check_service_health(self):
        """检查服务整体健康状态"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.results.append({
                    "service": "overall",
                    "status": "healthy" if data["overall_status"] == "healthy" else "unhealthy",
                    "details": data
                })
                return True
            else:
                self.results.append({
                    "service": "overall",
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}"
                })
                return False
        except Exception as e:
            self.results.append({
                "service": "overall",
                "status": "error",
                "error": str(e)
            })
            return False
    
    def check_database_connection(self):
        """检查数据库连接"""
        try:
            # 尝试提交一个测试连接请求
            test_config = {
                "host": "localhost",
                "port": 5432,
                "database": "test",
                "username": "test",
                "password": "test",
                "database_type": "postgresql"
            }
            
            response = requests.post(
                f"{self.base_url}/api/v1/extraction/test-connection",
                json=test_config,
                timeout=10
            )
            
            # 即使连接失败，只要 API 响应正常就说明服务可用
            self.results.append({
                "service": "database_api",
                "status": "healthy" if response.status_code in [200, 400] else "unhealthy",
                "details": response.json() if response.headers.get('content-type') == 'application/json' else {}
            })
            
        except Exception as e:
            self.results.append({
                "service": "database_api",
                "status": "error",
                "error": str(e)
            })
    
    def check_ai_models(self):
        """检查 AI 模型可用性"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/ai/models", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.results.append({
                    "service": "ai_models",
                    "status": "healthy",
                    "details": f"发现 {len(data['models'])} 个模型类型"
                })
                
                # 检查具体模型健康状态
                for model in data["models"]:
                    if model["supported"]:
                        model_type = model["model_type"]
                        try:
                            health_response = requests.get(
                                f"{self.base_url}/api/v1/ai/models/{model_type}/health",
                                timeout=5
                            )
                            if health_response.status_code == 200:
                                health_data = health_response.json()
                                self.results.append({
                                    "service": f"ai_model_{model_type}",
                                    "status": "healthy" if health_data["available"] else "unhealthy",
                                    "details": health_data["message"]
                                })
                        except:
                            self.results.append({
                                "service": f"ai_model_{model_type}",
                                "status": "unknown",
                                "details": "健康检查超时"
                            })
            else:
                self.results.append({
                    "service": "ai_models",
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}"
                })
        except Exception as e:
            self.results.append({
                "service": "ai_models",
                "status": "error",
                "error": str(e)
            })
    
    def check_system_metrics(self):
        """检查系统指标"""
        try:
            response = requests.get(f"{self.base_url}/system/metrics", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.results.append({
                    "service": "system_metrics",
                    "status": "healthy",
                    "details": f"收集了 {len(data)} 个指标类别"
                })
            else:
                self.results.append({
                    "service": "system_metrics",
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}"
                })
        except Exception as e:
            self.results.append({
                "service": "system_metrics",
                "status": "error",
                "error": str(e)
            })
    
    def run_all_checks(self):
        """运行所有健康检查"""
        print(f"开始健康检查 - {datetime.now().isoformat()}")
        print(f"目标服务: {self.base_url}")
        print("=" * 50)
        
        checks = [
            ("服务整体状态", self.check_service_health),
            ("数据库 API", self.check_database_connection),
            ("AI 模型", self.check_ai_models),
            ("系统指标", self.check_system_metrics)
        ]
        
        for check_name, check_func in checks:
            print(f"检查 {check_name}...", end=" ")
            try:
                check_func()
                print("完成")
            except Exception as e:
                print(f"异常: {e}")
        
        print("\n" + "=" * 50)
        print("健康检查结果:")
        
        healthy_count = 0
        total_count = len(self.results)
        
        for result in self.results:
            status_icon = {
                "healthy": "✅",
                "unhealthy": "❌",
                "error": "⚠️",
                "unknown": "❓"
            }.get(result["status"], "❓")
            
            print(f"{status_icon} {result['service']}: {result['status']}")
            
            if result["status"] == "healthy":
                healthy_count += 1
            
            if "error" in result:
                print(f"   错误: {result['error']}")
            elif "details" in result:
                print(f"   详情: {result['details']}")
        
        print(f"\n总体状态: {healthy_count}/{total_count} 服务正常")
        
        if healthy_count == total_count:
            print("🎉 所有服务运行正常！")
            return 0
        else:
            print("⚠️ 部分服务存在问题，请检查上述错误信息")
            return 1

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SuperInsight 平台健康检查")
    parser.add_argument("--url", default="http://localhost:8000", help="API 基础 URL")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出结果")
    
    args = parser.parse_args()
    
    checker = HealthChecker(args.url)
    exit_code = checker.run_all_checks()
    
    if args.json:
        print("\nJSON 格式结果:")
        print(json.dumps(checker.results, indent=2, ensure_ascii=False))
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
```

### 2. 错误诊断脚本

```python
#!/usr/bin/env python3
"""
SuperInsight 平台错误诊断脚本
"""

import requests
import json
import time
from datetime import datetime, timedelta

class ErrorDiagnostic:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def diagnose_extraction_error(self, job_id):
        """诊断数据提取错误"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/extraction/jobs/{job_id}")
            
            if response.status_code == 404:
                return {
                    "diagnosis": "任务不存在",
                    "suggestions": [
                        "检查任务 ID 是否正确",
                        "任务可能已被删除",
                        "查看任务列表确认任务状态"
                    ]
                }
            
            if response.status_code == 200:
                job_data = response.json()
                
                if job_data["status"] == "failed":
                    error_msg = job_data.get("error", "")
                    
                    # 根据错误信息提供诊断
                    if "connection" in error_msg.lower():
                        return {
                            "diagnosis": "数据库连接问题",
                            "error": error_msg,
                            "suggestions": [
                                "检查数据库服务是否运行",
                                "验证连接参数（主机、端口、数据库名）",
                                "检查网络连接和防火墙设置",
                                "确认用户名和密码正确"
                            ]
                        }
                    elif "permission" in error_msg.lower() or "access" in error_msg.lower():
                        return {
                            "diagnosis": "权限问题",
                            "error": error_msg,
                            "suggestions": [
                                "检查数据库用户权限",
                                "确认用户有表的 SELECT 权限",
                                "检查文件读取权限"
                            ]
                        }
                    elif "timeout" in error_msg.lower():
                        return {
                            "diagnosis": "超时问题",
                            "error": error_msg,
                            "suggestions": [
                                "减少查询数据量",
                                "优化 SQL 查询",
                                "增加超时时间",
                                "检查网络延迟"
                            ]
                        }
                    else:
                        return {
                            "diagnosis": "未知错误",
                            "error": error_msg,
                            "suggestions": [
                                "查看详细错误日志",
                                "联系技术支持",
                                "尝试重新提交任务"
                            ]
                        }
                else:
                    return {
                        "diagnosis": f"任务状态: {job_data['status']}",
                        "details": job_data
                    }
            
        except Exception as e:
            return {
                "diagnosis": "诊断过程中发生错误",
                "error": str(e),
                "suggestions": [
                    "检查网络连接",
                    "确认 API 服务正常运行",
                    "验证 API 地址是否正确"
                ]
            }
    
    def diagnose_ai_prediction_error(self, task_id, model_config):
        """诊断 AI 预测错误"""
        try:
            # 首先检查模型健康状态
            model_type = model_config.get("model_type")
            if model_type:
                health_response = requests.get(
                    f"{self.base_url}/api/v1/ai/models/{model_type}/health"
                )
                
                if health_response.status_code == 200:
                    health_data = health_response.json()
                    if not health_data["available"]:
                        return {
                            "diagnosis": "模型不可用",
                            "error": health_data["message"],
                            "suggestions": [
                                "检查模型配置",
                                "验证 API Key",
                                "确认模型服务运行状态",
                                "尝试使用其他模型"
                            ]
                        }
            
            # 尝试简单预测测试
            test_request = {
                "task_id": f"test_{int(time.time())}",
                "ai_model_config": model_config,
                "content": "测试文本"
            }
            
            response = requests.post(
                f"{self.base_url}/api/v1/ai/predict",
                json=test_request,
                timeout=30
            )
            
            if response.status_code == 401:
                return {
                    "diagnosis": "认证失败",
                    "suggestions": [
                        "检查 API Key 是否正确",
                        "确认 API Key 未过期",
                        "验证 API Key 权限范围"
                    ]
                }
            elif response.status_code == 429:
                return {
                    "diagnosis": "请求频率超限",
                    "suggestions": [
                        "降低请求频率",
                        "等待配额重置",
                        "升级 API 套餐"
                    ]
                }
            elif response.status_code == 500:
                error_data = response.json() if response.headers.get('content-type') == 'application/json' else {}
                return {
                    "diagnosis": "服务器内部错误",
                    "error": error_data.get("message", ""),
                    "suggestions": [
                        "稍后重试",
                        "检查输入数据格式",
                        "联系技术支持"
                    ]
                }
            elif response.status_code == 200:
                return {
                    "diagnosis": "模型工作正常",
                    "details": "测试预测成功完成"
                }
            else:
                return {
                    "diagnosis": f"未知错误 (HTTP {response.status_code})",
                    "suggestions": [
                        "查看 API 文档",
                        "检查请求格式",
                        "联系技术支持"
                    ]
                }
                
        except requests.exceptions.Timeout:
            return {
                "diagnosis": "请求超时",
                "suggestions": [
                    "增加超时时间",
                    "简化输入内容",
                    "使用更快的模型",
                    "检查网络连接"
                ]
            }
        except Exception as e:
            return {
                "diagnosis": "诊断过程中发生错误",
                "error": str(e),
                "suggestions": [
                    "检查网络连接",
                    "确认 API 服务正常运行"
                ]
            }
    
    def generate_diagnostic_report(self, error_type, **kwargs):
        """生成诊断报告"""
        print(f"错误诊断报告 - {datetime.now().isoformat()}")
        print("=" * 50)
        
        if error_type == "extraction":
            job_id = kwargs.get("job_id")
            if not job_id:
                print("错误: 需要提供任务 ID")
                return
            
            print(f"诊断数据提取任务: {job_id}")
            result = self.diagnose_extraction_error(job_id)
            
        elif error_type == "ai_prediction":
            task_id = kwargs.get("task_id")
            model_config = kwargs.get("model_config")
            if not model_config:
                print("错误: 需要提供模型配置")
                return
            
            print(f"诊断 AI 预测任务: {task_id}")
            result = self.diagnose_ai_prediction_error(task_id, model_config)
            
        else:
            print(f"不支持的错误类型: {error_type}")
            return
        
        print(f"\n诊断结果: {result['diagnosis']}")
        
        if "error" in result:
            print(f"错误信息: {result['error']}")
        
        if "suggestions" in result:
            print("\n建议解决方案:")
            for i, suggestion in enumerate(result["suggestions"], 1):
                print(f"  {i}. {suggestion}")
        
        if "details" in result:
            print(f"\n详细信息: {result['details']}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="SuperInsight 平台错误诊断")
    parser.add_argument("--url", default="http://localhost:8000", help="API 基础 URL")
    parser.add_argument("--type", choices=["extraction", "ai_prediction"], required=True, help="错误类型")
    parser.add_argument("--job-id", help="数据提取任务 ID")
    parser.add_argument("--task-id", help="AI 预测任务 ID")
    parser.add_argument("--model-config", help="模型配置 JSON 文件路径")
    
    args = parser.parse_args()
    
    diagnostic = ErrorDiagnostic(args.url)
    
    kwargs = {}
    if args.job_id:
        kwargs["job_id"] = args.job_id
    if args.task_id:
        kwargs["task_id"] = args.task_id
    if args.model_config:
        try:
            with open(args.model_config, 'r') as f:
                kwargs["model_config"] = json.load(f)
        except Exception as e:
            print(f"无法读取模型配置文件: {e}")
            return
    
    diagnostic.generate_diagnostic_report(args.type, **kwargs)

if __name__ == "__main__":
    main()
```

### 3. 性能监控脚本

```bash
#!/bin/bash
# SuperInsight 平台性能监控脚本

BASE_URL="${1:-http://localhost:8000}"
INTERVAL="${2:-60}"  # 监控间隔（秒）

echo "SuperInsight 平台性能监控"
echo "API 地址: $BASE_URL"
echo "监控间隔: ${INTERVAL}秒"
echo "按 Ctrl+C 停止监控"
echo "=========================="

while true; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] 检查中..."
    
    # 检查 API 响应时间
    response_time=$(curl -o /dev/null -s -w '%{time_total}' "$BASE_URL/health")
    if [ $? -eq 0 ]; then
        echo "  API 响应时间: ${response_time}s"
    else
        echo "  API 响应: 失败"
    fi
    
    # 检查系统指标
    metrics_response=$(curl -s "$BASE_URL/system/metrics" 2>/dev/null)
    if [ $? -eq 0 ]; then
        # 提取关键指标（需要 jq 工具）
        if command -v jq >/dev/null 2>&1; then
            cpu_usage=$(echo "$metrics_response" | jq -r '.system.cpu.usage_percent // "N/A"')
            memory_usage=$(echo "$metrics_response" | jq -r '.system.memory.usage_percent // "N/A"')
            echo "  CPU 使用率: ${cpu_usage}%"
            echo "  内存使用率: ${memory_usage}%"
        fi
    fi
    
    # 检查活跃任务数
    jobs_response=$(curl -s "$BASE_URL/api/v1/extraction/jobs" 2>/dev/null)
    if [ $? -eq 0 ] && command -v jq >/dev/null 2>&1; then
        running_jobs=$(echo "$jobs_response" | jq '[.[] | select(.status == "running")] | length')
        pending_jobs=$(echo "$jobs_response" | jq '[.[] | select(.status == "pending")] | length')
        echo "  运行中任务: $running_jobs"
        echo "  等待中任务: $pending_jobs"
    fi
    
    echo "  ----"
    sleep $INTERVAL
done
```

## 常见问题解答 (FAQ)

### Q1: 为什么数据提取任务一直处于 pending 状态？

**A**: 可能的原因包括：
1. 系统负载过高，任务队列积压
2. 数据源连接问题
3. 系统资源不足

**解决方案**:
1. 检查系统资源使用情况
2. 查看错误日志
3. 尝试重新提交任务
4. 联系技术支持

### Q2: AI 预标注返回置信度很低怎么办？

**A**: 置信度低可能表示：
1. 输入数据质量问题
2. 模型不适合当前任务
3. 提示词需要优化

**解决方案**:
1. 检查输入数据质量
2. 尝试不同的模型
3. 优化提示词模板
4. 增加训练数据

### Q3: 如何处理 API 频率限制？

**A**: 当遇到 429 错误时：
1. 检查 `Retry-After` 响应头
2. 实现指数退避重试
3. 降低请求频率
4. 考虑升级 API 套餐

### Q4: 批量任务部分失败怎么处理？

**A**: 批量任务失败处理：
1. 查看失败任务的具体错误信息
2. 对失败的任务单独重试
3. 调整批量配置参数
4. 检查输入数据质量

### Q5: 如何优化 API 性能？

**A**: 性能优化建议：
1. 使用批量 API 而非单个请求
2. 实现客户端缓存
3. 合理设置并发数
4. 使用连接池
5. 启用 gzip 压缩

---

*如果以上信息无法解决您的问题，请联系技术支持团队获取进一步帮助。*