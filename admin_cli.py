#!/usr/bin/env python3
"""
SuperInsight 平台管理命令行工具

提供系统管理、监控和维护功能的命令行接口。
"""

import os
import sys
import json
import time
import argparse
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SuperInsightAdmin:
    """SuperInsight 管理客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_token: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        if api_token:
            self.session.headers.update({'Authorization': f'Bearer {api_token}'})
        
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'SuperInsight-Admin-CLI/1.0'
        })
    
    def _api_call(self, endpoint: str, method: str = 'GET', data: Optional[Dict] = None) -> Dict[str, Any]:
        """执行 API 调用"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url, timeout=30)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data, timeout=30)
            elif method.upper() == 'PUT':
                response = self.session.put(url, json=data, timeout=30)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url, timeout=30)
            else:
                raise ValueError(f"不支持的 HTTP 方法: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API 调用失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    logger.error(f"错误详情: {error_data}")
                except:
                    logger.error(f"HTTP {e.response.status_code}: {e.response.text}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """系统健康检查"""
        return self._api_call('/health')
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return self._api_call('/admin/system/status')
    
    def get_overview_stats(self) -> Dict[str, Any]:
        """获取概览统计"""
        return self._api_call('/admin/stats/overview')
    
    def get_recent_activities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近活动"""
        return self._api_call(f'/admin/activities/recent?limit={limit}')
    
    def get_users(self) -> List[Dict[str, Any]]:
        """获取用户列表"""
        return self._api_call('/admin/users')
    
    def create_user(self, username: str, email: str, role: str, tenant_id: str) -> Dict[str, Any]:
        """创建用户"""
        data = {
            "username": username,
            "email": email,
            "role": role,
            "tenant_id": tenant_id,
            "is_active": True
        }
        return self._api_call('/admin/users', method='POST', data=data)
    
    def get_error_stats(self) -> Dict[str, Any]:
        """获取错误统计"""
        return self._api_call('/admin/errors/stats')
    
    def get_recent_logs(self, limit: int = 50, level: str = 'all') -> List[Dict[str, Any]]:
        """获取最近日志"""
        return self._api_call(f'/admin/logs/recent?limit={limit}&level={level}')
    
    def get_settings(self) -> Dict[str, Any]:
        """获取系统设置"""
        return self._api_call('/admin/settings')
    
    def update_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """更新系统设置"""
        return self._api_call('/admin/settings', method='POST', data=settings)
    
    def toggle_maintenance_mode(self, action: str, duration_minutes: int = 30, message: str = "") -> Dict[str, Any]:
        """切换维护模式"""
        data = {
            "action": action,
            "duration_minutes": duration_minutes,
            "message": message
        }
        return self._api_call('/admin/maintenance', method='POST', data=data)
    
    def get_billing_overview(self, days: int = 30) -> Dict[str, Any]:
        """获取计费概览"""
        return self._api_call(f'/admin/billing/overview?days={days}')
    
    def restart_system(self) -> Dict[str, Any]:
        """重启系统"""
        return self._api_call('/admin/system/restart', method='POST')


def format_table(data: List[Dict], headers: List[str]) -> str:
    """格式化表格输出"""
    if not data:
        return "无数据"
    
    # 计算列宽
    col_widths = {}
    for header in headers:
        col_widths[header] = len(header)
        for row in data:
            value = str(row.get(header, ''))
            col_widths[header] = max(col_widths[header], len(value))
    
    # 生成表格
    lines = []
    
    # 表头
    header_line = " | ".join(header.ljust(col_widths[header]) for header in headers)
    lines.append(header_line)
    lines.append("-" * len(header_line))
    
    # 数据行
    for row in data:
        data_line = " | ".join(str(row.get(header, '')).ljust(col_widths[header]) for header in headers)
        lines.append(data_line)
    
    return "\n".join(lines)


def format_json(data: Any, indent: int = 2) -> str:
    """格式化 JSON 输出"""
    return json.dumps(data, indent=indent, ensure_ascii=False)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SuperInsight 平台管理工具")
    parser.add_argument("--url", default="http://localhost:8000", help="API 基础 URL")
    parser.add_argument("--token", help="API 认证令牌")
    parser.add_argument("--format", choices=["table", "json"], default="table", help="输出格式")
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 健康检查命令
    health_parser = subparsers.add_parser("health", help="系统健康检查")
    
    # 状态命令
    status_parser = subparsers.add_parser("status", help="获取系统状态")
    
    # 统计命令
    stats_parser = subparsers.add_parser("stats", help="获取系统统计")
    
    # 活动命令
    activities_parser = subparsers.add_parser("activities", help="获取最近活动")
    activities_parser.add_argument("--limit", type=int, default=10, help="显示数量")
    
    # 用户管理命令
    users_parser = subparsers.add_parser("users", help="用户管理")
    users_subparsers = users_parser.add_subparsers(dest="users_action")
    
    users_list_parser = users_subparsers.add_parser("list", help="列出用户")
    
    users_create_parser = users_subparsers.add_parser("create", help="创建用户")
    users_create_parser.add_argument("--username", required=True, help="用户名")
    users_create_parser.add_argument("--email", required=True, help="邮箱")
    users_create_parser.add_argument("--role", required=True, help="角色")
    users_create_parser.add_argument("--tenant", required=True, help="租户ID")
    
    # 错误统计命令
    errors_parser = subparsers.add_parser("errors", help="获取错误统计")
    
    # 日志命令
    logs_parser = subparsers.add_parser("logs", help="获取系统日志")
    logs_parser.add_argument("--limit", type=int, default=50, help="显示数量")
    logs_parser.add_argument("--level", choices=["all", "error", "warning", "info", "debug"], 
                           default="all", help="日志级别")
    
    # 设置命令
    settings_parser = subparsers.add_parser("settings", help="系统设置管理")
    settings_subparsers = settings_parser.add_subparsers(dest="settings_action")
    
    settings_get_parser = settings_subparsers.add_parser("get", help="获取设置")
    
    settings_set_parser = settings_subparsers.add_parser("set", help="更新设置")
    settings_set_parser.add_argument("--api-rate-limit", type=int, help="API 请求限制")
    settings_set_parser.add_argument("--max-concurrent-jobs", type=int, help="最大并发任务数")
    settings_set_parser.add_argument("--log-retention-days", type=int, help="日志保留天数")
    settings_set_parser.add_argument("--maintenance-mode", action="store_true", help="启用维护模式")
    
    # 维护命令
    maintenance_parser = subparsers.add_parser("maintenance", help="维护模式管理")
    maintenance_parser.add_argument("action", choices=["enable", "disable"], help="维护操作")
    maintenance_parser.add_argument("--duration", type=int, default=30, help="维护时长（分钟）")
    maintenance_parser.add_argument("--message", default="", help="维护消息")
    
    # 计费命令
    billing_parser = subparsers.add_parser("billing", help="获取计费概览")
    billing_parser.add_argument("--days", type=int, default=30, help="统计天数")
    
    # 重启命令
    restart_parser = subparsers.add_parser("restart", help="重启系统")
    restart_parser.add_argument("--confirm", action="store_true", help="确认重启")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 创建管理客户端
    admin = SuperInsightAdmin(args.url, args.token)
    
    try:
        # 执行命令
        if args.command == "health":
            result = admin.health_check()
            if args.format == "json":
                print(format_json(result))
            else:
                status = result.get("overall_status", "unknown")
                print(f"系统状态: {status}")
                if "services" in result:
                    print("\n服务状态:")
                    for service, info in result["services"].items():
                        print(f"  {service}: {info.get('status', 'unknown')}")
        
        elif args.command == "status":
            result = admin.get_system_status()
            if args.format == "json":
                print(format_json(result))
            else:
                print(f"系统状态: {result['overall_status']}")
                print(f"版本: {result['version']}")
                print(f"运行时间: {result['uptime']}")
                print("\n服务状态:")
                for service, info in result["services"].items():
                    print(f"  {service}: {info.get('status', 'unknown')}")
        
        elif args.command == "stats":
            result = admin.get_overview_stats()
            if args.format == "json":
                print(format_json(result))
            else:
                print("系统统计:")
                print(f"  总用户数: {result.get('total_users', 0)}")
                print(f"  活跃任务: {result.get('active_jobs', 0)}")
                print(f"  文档总数: {result.get('total_documents', 0)}")
        
        elif args.command == "activities":
            result = admin.get_recent_activities(args.limit)
            if args.format == "json":
                print(format_json(result))
            else:
                if result:
                    headers = ["timestamp", "user", "action", "status"]
                    # 格式化时间戳
                    for activity in result:
                        if "timestamp" in activity:
                            dt = datetime.fromisoformat(activity["timestamp"].replace('Z', '+00:00'))
                            activity["timestamp"] = dt.strftime("%Y-%m-%d %H:%M:%S")
                    print(format_table(result, headers))
                else:
                    print("暂无活动记录")
        
        elif args.command == "users":
            if args.users_action == "list":
                result = admin.get_users()
                if args.format == "json":
                    print(format_json(result))
                else:
                    if result:
                        headers = ["username", "email", "role", "tenant_id", "is_active"]
                        print(format_table(result, headers))
                    else:
                        print("暂无用户")
            
            elif args.users_action == "create":
                result = admin.create_user(args.username, args.email, args.role, args.tenant)
                if args.format == "json":
                    print(format_json(result))
                else:
                    print(f"用户创建成功: {args.username}")
        
        elif args.command == "errors":
            result = admin.get_error_stats()
            if args.format == "json":
                print(format_json(result))
            else:
                print("错误统计:")
                print(f"  总错误数: {result.get('total_errors', 0)}")
                print(f"  最近1小时: {result.get('errors_last_hour', 0)}")
                print(f"  错误率: {result.get('error_rate', 0):.2f}%")
        
        elif args.command == "logs":
            result = admin.get_recent_logs(args.limit, args.level)
            if args.format == "json":
                print(format_json(result))
            else:
                if result:
                    for log in result:
                        timestamp = log.get("timestamp", "")
                        if timestamp:
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
                        level = log.get("level", "").upper()
                        message = log.get("message", "")
                        print(f"[{timestamp}] [{level}] {message}")
                else:
                    print("暂无日志记录")
        
        elif args.command == "settings":
            if args.settings_action == "get":
                result = admin.get_settings()
                if args.format == "json":
                    print(format_json(result))
                else:
                    print("系统设置:")
                    for key, value in result.items():
                        print(f"  {key}: {value}")
            
            elif args.settings_action == "set":
                settings = {}
                if args.api_rate_limit is not None:
                    settings["api_rate_limit"] = args.api_rate_limit
                if args.max_concurrent_jobs is not None:
                    settings["max_concurrent_jobs"] = args.max_concurrent_jobs
                if args.log_retention_days is not None:
                    settings["log_retention_days"] = args.log_retention_days
                if args.maintenance_mode:
                    settings["maintenance_mode"] = True
                
                if settings:
                    result = admin.update_settings(settings)
                    if args.format == "json":
                        print(format_json(result))
                    else:
                        print("设置更新成功")
                else:
                    print("未指定要更新的设置")
        
        elif args.command == "maintenance":
            result = admin.toggle_maintenance_mode(args.action, args.duration, args.message)
            if args.format == "json":
                print(format_json(result))
            else:
                print(result.get("message", "操作完成"))
        
        elif args.command == "billing":
            result = admin.get_billing_overview(args.days)
            if args.format == "json":
                print(format_json(result))
            else:
                print(f"计费概览 (最近 {args.days} 天):")
                print(f"  总收入: ¥{result.get('total_revenue', 0):,.2f}")
                print(f"  总标注数: {result.get('total_annotations', 0):,}")
                print(f"  总工时: {result.get('total_hours', 0):,} 小时")
                print(f"  活跃租户: {result.get('active_tenants', 0)}")
                
                if "top_tenants" in result:
                    print("\n主要租户:")
                    for tenant in result["top_tenants"]:
                        print(f"  {tenant['tenant_id']}: ¥{tenant['revenue']:,.2f} ({tenant['annotations']:,} 标注)")
        
        elif args.command == "restart":
            if not args.confirm:
                print("警告: 此操作将重启系统，可能导致短暂的服务中断。")
                print("如果确认要重启，请添加 --confirm 参数。")
                return
            
            result = admin.restart_system()
            if args.format == "json":
                print(format_json(result))
            else:
                print(result.get("message", "重启请求已提交"))
                if "estimated_downtime_minutes" in result:
                    print(f"预计停机时间: {result['estimated_downtime_minutes']} 分钟")
    
    except Exception as e:
        logger.error(f"命令执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()