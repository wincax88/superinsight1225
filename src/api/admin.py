"""
Admin API endpoints for SuperInsight platform.

Provides administrative functions for system management, monitoring, and configuration.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, Query, status
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field

from src.config.settings import settings
from src.database.connection import get_db_session
from src.database.models import DocumentModel, TaskModel, BillingRecordModel
from src.system.integration import system_manager
from src.system.monitoring import metrics_collector, health_monitor
from src.system.error_handler import error_handler
from src.security.models import User, UserRole, AccessLog
from src.billing.service import BillingSystem
from src.admin.config_manager import config_manager
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/admin", tags=["Admin"])

# Initialize services
billing_system = BillingSystem()


# Pydantic models for admin API
class SystemStatusResponse(BaseModel):
    """系统状态响应"""
    overall_status: str
    services: Dict[str, Any]
    metrics: Dict[str, Any]
    uptime: str
    version: str


class UserManagementRequest(BaseModel):
    """用户管理请求"""
    username: str = Field(..., description="用户名")
    email: str = Field(..., description="邮箱")
    role: str = Field(..., description="角色")
    tenant_id: str = Field(..., description="租户ID")
    is_active: bool = Field(True, description="是否激活")


class ConfigUpdateRequest(BaseModel):
    """配置更新请求"""
    section: str = Field(..., description="配置节")
    key: str = Field(..., description="配置键")
    value: Any = Field(..., description="配置值")


class MaintenanceRequest(BaseModel):
    """维护请求"""
    action: str = Field(..., description="维护动作")
    duration_minutes: int = Field(30, description="维护时长（分钟）")
    message: str = Field("", description="维护消息")


# Admin authentication dependency (simplified)
def get_admin_user():
    """获取管理员用户（简化版本）"""
    # 在实际实现中，这里应该验证管理员权限
    return {"user_id": "admin", "role": "admin"}


# System Management Endpoints
@router.get("/enhanced", response_class=HTMLResponse)
async def enhanced_admin_dashboard():
    """Enhanced management console dashboard page."""
    from src.admin.enhanced_dashboard_template import get_enhanced_dashboard_html
    return HTMLResponse(content=get_enhanced_dashboard_html())


@router.get("/", response_class=HTMLResponse)
async def admin_dashboard():
    """管理员仪表板页面"""
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SuperInsight 管理控制台</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; }
            .header { background: #2c3e50; color: white; padding: 1rem 2rem; }
            .header h1 { font-size: 1.5rem; }
            .nav { background: #34495e; padding: 0.5rem 2rem; }
            .nav a { color: white; text-decoration: none; margin-right: 2rem; padding: 0.5rem 1rem; border-radius: 4px; }
            .nav a:hover { background: #4a6741; }
            .nav a.active { background: #3498db; }
            .container { max-width: 1200px; margin: 2rem auto; padding: 0 2rem; }
            .card { background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 2rem; }
            .card-header { padding: 1rem 1.5rem; border-bottom: 1px solid #eee; font-weight: 600; }
            .card-body { padding: 1.5rem; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem; }
            .metric { text-align: center; padding: 1rem; }
            .metric-value { font-size: 2rem; font-weight: bold; color: #3498db; }
            .metric-label { color: #666; margin-top: 0.5rem; }
            .status-healthy { color: #27ae60; }
            .status-warning { color: #f39c12; }
            .status-error { color: #e74c3c; }
            .btn { padding: 0.5rem 1rem; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block; }
            .btn-primary { background: #3498db; color: white; }
            .btn-danger { background: #e74c3c; color: white; }
            .btn-success { background: #27ae60; color: white; }
            .table { width: 100%; border-collapse: collapse; }
            .table th, .table td { padding: 0.75rem; text-align: left; border-bottom: 1px solid #eee; }
            .table th { background: #f8f9fa; font-weight: 600; }
            .loading { text-align: center; padding: 2rem; color: #666; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>SuperInsight 管理控制台</h1>
        </div>
        <div class="nav">
            <a href="#dashboard" class="active" onclick="showSection('dashboard')">仪表板</a>
            <a href="#system" onclick="showSection('system')">系统状态</a>
            <a href="#users" onclick="showSection('users')">用户管理</a>
            <a href="#monitoring" onclick="showSection('monitoring')">监控</a>
            <a href="#logs" onclick="showSection('logs')">日志</a>
            <a href="#settings" onclick="showSection('settings')">设置</a>
        </div>
        
        <div class="container">
            <!-- Dashboard Section -->
            <div id="dashboard" class="section">
                <div class="card">
                    <div class="card-header">系统概览</div>
                    <div class="card-body">
                        <div class="grid">
                            <div class="metric">
                                <div class="metric-value" id="total-users">-</div>
                                <div class="metric-label">总用户数</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value" id="active-jobs">-</div>
                                <div class="metric-label">活跃任务</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value" id="total-documents">-</div>
                                <div class="metric-label">文档总数</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value" id="system-status">-</div>
                                <div class="metric-label">系统状态</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">最近活动</div>
                    <div class="card-body">
                        <table class="table">
                            <thead>
                                <tr>
                                    <th>时间</th>
                                    <th>用户</th>
                                    <th>操作</th>
                                    <th>状态</th>
                                </tr>
                            </thead>
                            <tbody id="recent-activities">
                                <tr><td colspan="4" class="loading">加载中...</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            
            <!-- System Status Section -->
            <div id="system" class="section" style="display: none;">
                <div class="card">
                    <div class="card-header">系统服务状态</div>
                    <div class="card-body">
                        <div id="services-status" class="loading">加载中...</div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">系统指标</div>
                    <div class="card-body">
                        <div id="system-metrics" class="loading">加载中...</div>
                    </div>
                </div>
            </div>
            
            <!-- User Management Section -->
            <div id="users" class="section" style="display: none;">
                <div class="card">
                    <div class="card-header">
                        用户管理
                        <button class="btn btn-primary" style="float: right;" onclick="showAddUserModal()">添加用户</button>
                    </div>
                    <div class="card-body">
                        <table class="table">
                            <thead>
                                <tr>
                                    <th>用户名</th>
                                    <th>邮箱</th>
                                    <th>角色</th>
                                    <th>租户</th>
                                    <th>状态</th>
                                    <th>操作</th>
                                </tr>
                            </thead>
                            <tbody id="users-table">
                                <tr><td colspan="6" class="loading">加载中...</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            
            <!-- Monitoring Section -->
            <div id="monitoring" class="section" style="display: none;">
                <div class="card">
                    <div class="card-header">性能监控</div>
                    <div class="card-body">
                        <div class="grid">
                            <div class="metric">
                                <div class="metric-value" id="cpu-usage">-</div>
                                <div class="metric-label">CPU 使用率</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value" id="memory-usage">-</div>
                                <div class="metric-label">内存使用率</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value" id="disk-usage">-</div>
                                <div class="metric-label">磁盘使用率</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value" id="api-requests">-</div>
                                <div class="metric-label">API 请求/分钟</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">错误统计</div>
                    <div class="card-body">
                        <div id="error-stats" class="loading">加载中...</div>
                    </div>
                </div>
            </div>
            
            <!-- Logs Section -->
            <div id="logs" class="section" style="display: none;">
                <div class="card">
                    <div class="card-header">
                        系统日志
                        <select id="log-level" style="float: right; margin-left: 1rem;">
                            <option value="all">所有级别</option>
                            <option value="error">错误</option>
                            <option value="warning">警告</option>
                            <option value="info">信息</option>
                        </select>
                        <button class="btn btn-primary" style="float: right;" onclick="refreshLogs()">刷新</button>
                    </div>
                    <div class="card-body">
                        <div id="logs-content" style="background: #f8f9fa; padding: 1rem; border-radius: 4px; font-family: monospace; max-height: 400px; overflow-y: auto;">
                            加载中...
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Settings Section -->
            <div id="settings" class="section" style="display: none;">
                <div class="card">
                    <div class="card-header">系统设置</div>
                    <div class="card-body">
                        <form id="settings-form">
                            <div style="margin-bottom: 1rem;">
                                <label>API 请求限制（每分钟）:</label>
                                <input type="number" id="api-rate-limit" value="1000" style="margin-left: 1rem; padding: 0.5rem;">
                            </div>
                            <div style="margin-bottom: 1rem;">
                                <label>最大并发任务数:</label>
                                <input type="number" id="max-concurrent-jobs" value="10" style="margin-left: 1rem; padding: 0.5rem;">
                            </div>
                            <div style="margin-bottom: 1rem;">
                                <label>日志保留天数:</label>
                                <input type="number" id="log-retention-days" value="30" style="margin-left: 1rem; padding: 0.5rem;">
                            </div>
                            <div style="margin-bottom: 1rem;">
                                <label>
                                    <input type="checkbox" id="maintenance-mode"> 维护模式
                                </label>
                            </div>
                            <button type="button" class="btn btn-success" onclick="saveSettings()">保存设置</button>
                        </form>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Global variables
            let currentSection = 'dashboard';
            let refreshInterval;
            
            // Initialize dashboard
            document.addEventListener('DOMContentLoaded', function() {
                loadDashboardData();
                startAutoRefresh();
            });
            
            // Section navigation
            function showSection(sectionName) {
                // Hide all sections
                document.querySelectorAll('.section').forEach(section => {
                    section.style.display = 'none';
                });
                
                // Show selected section
                document.getElementById(sectionName).style.display = 'block';
                
                // Update navigation
                document.querySelectorAll('.nav a').forEach(link => {
                    link.classList.remove('active');
                });
                document.querySelector(`[onclick="showSection('${sectionName}')"]`).classList.add('active');
                
                currentSection = sectionName;
                
                // Load section-specific data
                switch(sectionName) {
                    case 'system':
                        loadSystemStatus();
                        break;
                    case 'users':
                        loadUsers();
                        break;
                    case 'monitoring':
                        loadMonitoring();
                        break;
                    case 'logs':
                        loadLogs();
                        break;
                    case 'settings':
                        loadSettings();
                        break;
                }
            }
            
            // API helper function
            async function apiCall(endpoint, options = {}) {
                try {
                    const response = await fetch(endpoint, {
                        headers: {
                            'Content-Type': 'application/json',
                            ...options.headers
                        },
                        ...options
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    
                    return await response.json();
                } catch (error) {
                    console.error('API call failed:', error);
                    throw error;
                }
            }
            
            // Dashboard functions
            async function loadDashboardData() {
                try {
                    const [systemStatus, stats] = await Promise.all([
                        apiCall('/admin/system/status'),
                        apiCall('/admin/stats/overview')
                    ]);
                    
                    // Update metrics
                    document.getElementById('total-users').textContent = stats.total_users || 0;
                    document.getElementById('active-jobs').textContent = stats.active_jobs || 0;
                    document.getElementById('total-documents').textContent = stats.total_documents || 0;
                    
                    const statusElement = document.getElementById('system-status');
                    statusElement.textContent = systemStatus.overall_status;
                    statusElement.className = `metric-value status-${systemStatus.overall_status}`;
                    
                    // Load recent activities
                    loadRecentActivities();
                    
                } catch (error) {
                    console.error('Failed to load dashboard data:', error);
                }
            }
            
            async function loadRecentActivities() {
                try {
                    const activities = await apiCall('/admin/activities/recent');
                    const tbody = document.getElementById('recent-activities');
                    
                    if (activities.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="4">暂无活动记录</td></tr>';
                        return;
                    }
                    
                    tbody.innerHTML = activities.map(activity => `
                        <tr>
                            <td>${new Date(activity.timestamp).toLocaleString()}</td>
                            <td>${activity.user}</td>
                            <td>${activity.action}</td>
                            <td><span class="status-${activity.status}">${activity.status}</span></td>
                        </tr>
                    `).join('');
                    
                } catch (error) {
                    document.getElementById('recent-activities').innerHTML = 
                        '<tr><td colspan="4">加载失败</td></tr>';
                }
            }
            
            // System status functions
            async function loadSystemStatus() {
                try {
                    const [status, metrics] = await Promise.all([
                        apiCall('/system/status'),
                        apiCall('/system/metrics')
                    ]);
                    
                    // Display services status
                    const servicesHtml = Object.entries(status.services).map(([name, info]) => `
                        <div style="margin-bottom: 1rem; padding: 1rem; border: 1px solid #ddd; border-radius: 4px;">
                            <strong>${name}</strong>: 
                            <span class="status-${info.status}">${info.status}</span>
                            ${info.message ? `<br><small>${info.message}</small>` : ''}
                        </div>
                    `).join('');
                    
                    document.getElementById('services-status').innerHTML = servicesHtml;
                    
                    // Display system metrics
                    const metricsHtml = `
                        <div class="grid">
                            <div class="metric">
                                <div class="metric-value">${metrics.system?.cpu?.usage_percent?.toFixed(1) || 'N/A'}%</div>
                                <div class="metric-label">CPU 使用率</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value">${metrics.system?.memory?.usage_percent?.toFixed(1) || 'N/A'}%</div>
                                <div class="metric-label">内存使用率</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value">${metrics.system?.disk?.usage_percent?.toFixed(1) || 'N/A'}%</div>
                                <div class="metric-label">磁盘使用率</div>
                            </div>
                        </div>
                    `;
                    
                    document.getElementById('system-metrics').innerHTML = metricsHtml;
                    
                } catch (error) {
                    document.getElementById('services-status').innerHTML = '加载失败';
                    document.getElementById('system-metrics').innerHTML = '加载失败';
                }
            }
            
            // User management functions
            async function loadUsers() {
                try {
                    const users = await apiCall('/admin/users');
                    const tbody = document.getElementById('users-table');
                    
                    if (users.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="6">暂无用户</td></tr>';
                        return;
                    }
                    
                    tbody.innerHTML = users.map(user => `
                        <tr>
                            <td>${user.username}</td>
                            <td>${user.email}</td>
                            <td>${user.role}</td>
                            <td>${user.tenant_id}</td>
                            <td><span class="status-${user.is_active ? 'healthy' : 'error'}">${user.is_active ? '激活' : '禁用'}</span></td>
                            <td>
                                <button class="btn btn-primary" onclick="editUser('${user.id}')">编辑</button>
                                <button class="btn btn-danger" onclick="deleteUser('${user.id}')">删除</button>
                            </td>
                        </tr>
                    `).join('');
                    
                } catch (error) {
                    document.getElementById('users-table').innerHTML = 
                        '<tr><td colspan="6">加载失败</td></tr>';
                }
            }
            
            // Monitoring functions
            async function loadMonitoring() {
                try {
                    const [metrics, errors] = await Promise.all([
                        apiCall('/system/metrics'),
                        apiCall('/admin/errors/stats')
                    ]);
                    
                    // Update performance metrics
                    document.getElementById('cpu-usage').textContent = 
                        `${metrics.system?.cpu?.usage_percent?.toFixed(1) || 'N/A'}%`;
                    document.getElementById('memory-usage').textContent = 
                        `${metrics.system?.memory?.usage_percent?.toFixed(1) || 'N/A'}%`;
                    document.getElementById('disk-usage').textContent = 
                        `${metrics.system?.disk?.usage_percent?.toFixed(1) || 'N/A'}%`;
                    document.getElementById('api-requests').textContent = 
                        metrics.api?.requests_per_minute || 'N/A';
                    
                    // Display error statistics
                    const errorHtml = `
                        <div class="grid">
                            <div class="metric">
                                <div class="metric-value status-error">${errors.total_errors || 0}</div>
                                <div class="metric-label">总错误数</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value status-warning">${errors.errors_last_hour || 0}</div>
                                <div class="metric-label">最近1小时</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value">${errors.error_rate?.toFixed(2) || 0}%</div>
                                <div class="metric-label">错误率</div>
                            </div>
                        </div>
                    `;
                    
                    document.getElementById('error-stats').innerHTML = errorHtml;
                    
                } catch (error) {
                    console.error('Failed to load monitoring data:', error);
                }
            }
            
            // Logs functions
            async function loadLogs() {
                try {
                    const logs = await apiCall('/admin/logs/recent');
                    const content = document.getElementById('logs-content');
                    
                    if (logs.length === 0) {
                        content.textContent = '暂无日志记录';
                        return;
                    }
                    
                    content.innerHTML = logs.map(log => `
                        <div style="margin-bottom: 0.5rem; padding: 0.25rem; border-left: 3px solid ${getLogLevelColor(log.level)};">
                            <span style="color: #666;">[${new Date(log.timestamp).toLocaleString()}]</span>
                            <span style="font-weight: bold;">[${log.level.toUpperCase()}]</span>
                            ${log.message}
                        </div>
                    `).join('');
                    
                } catch (error) {
                    document.getElementById('logs-content').textContent = '加载日志失败';
                }
            }
            
            function getLogLevelColor(level) {
                switch(level.toLowerCase()) {
                    case 'error': return '#e74c3c';
                    case 'warning': return '#f39c12';
                    case 'info': return '#3498db';
                    default: return '#95a5a6';
                }
            }
            
            function refreshLogs() {
                loadLogs();
            }
            
            // Settings functions
            async function loadSettings() {
                try {
                    const settings = await apiCall('/admin/settings');
                    
                    document.getElementById('api-rate-limit').value = settings.api_rate_limit || 1000;
                    document.getElementById('max-concurrent-jobs').value = settings.max_concurrent_jobs || 10;
                    document.getElementById('log-retention-days').value = settings.log_retention_days || 30;
                    document.getElementById('maintenance-mode').checked = settings.maintenance_mode || false;
                    
                } catch (error) {
                    console.error('Failed to load settings:', error);
                }
            }
            
            async function saveSettings() {
                try {
                    const settings = {
                        api_rate_limit: parseInt(document.getElementById('api-rate-limit').value),
                        max_concurrent_jobs: parseInt(document.getElementById('max-concurrent-jobs').value),
                        log_retention_days: parseInt(document.getElementById('log-retention-days').value),
                        maintenance_mode: document.getElementById('maintenance-mode').checked
                    };
                    
                    await apiCall('/admin/settings', {
                        method: 'POST',
                        body: JSON.stringify(settings)
                    });
                    
                    alert('设置已保存');
                    
                } catch (error) {
                    alert('保存设置失败: ' + error.message);
                }
            }
            
            // Auto refresh
            function startAutoRefresh() {
                refreshInterval = setInterval(() => {
                    if (currentSection === 'dashboard') {
                        loadDashboardData();
                    } else if (currentSection === 'monitoring') {
                        loadMonitoring();
                    }
                }, 30000); // Refresh every 30 seconds
            }
            
            // Utility functions
            function showAddUserModal() {
                // In a real implementation, this would show a modal dialog
                alert('添加用户功能需要实现模态对话框');
            }
            
            function editUser(userId) {
                alert(`编辑用户 ${userId} 功能需要实现`);
            }
            
            function deleteUser(userId) {
                if (confirm('确定要删除此用户吗？')) {
                    // Implement user deletion
                    alert(`删除用户 ${userId} 功能需要实现`);
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@router.get("/system/status", response_model=SystemStatusResponse)
async def get_system_status(admin_user=Depends(get_admin_user)):
    """获取系统状态"""
    try:
        # Get system status from system manager
        system_status = system_manager.get_system_status()
        
        # Get metrics summary
        metrics = metrics_collector.get_all_metrics_summary()
        
        # Calculate uptime
        import time
        uptime_seconds = time.time() - system_manager.start_time if hasattr(system_manager, 'start_time') else 0
        uptime = str(timedelta(seconds=int(uptime_seconds)))
        
        return SystemStatusResponse(
            overall_status=system_status["overall_status"],
            services=system_status["services"],
            metrics=metrics,
            uptime=uptime,
            version=settings.app.app_version
        )
        
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取系统状态失败: {str(e)}"
        )


@router.get("/stats/overview")
async def get_overview_stats(
    admin_user=Depends(get_admin_user),
    db: Session = Depends(get_db_session)
):
    """获取概览统计"""
    try:
        # Get database statistics
        total_documents = db.query(func.count(DocumentModel.id)).scalar() or 0
        
        # Get active jobs count (simplified)
        active_jobs = 0  # In real implementation, query from job tracking system
        
        # Get user count (simplified)
        total_users = 10  # In real implementation, query from user management system
        
        return {
            "total_users": total_users,
            "active_jobs": active_jobs,
            "total_documents": total_documents,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取概览统计失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取统计数据失败: {str(e)}"
        )


@router.get("/activities/recent")
async def get_recent_activities(
    limit: int = Query(10, ge=1, le=100),
    admin_user=Depends(get_admin_user)
):
    """获取最近活动"""
    try:
        # In a real implementation, this would query from an activity log table
        activities = [
            {
                "timestamp": (datetime.now() - timedelta(minutes=5)).isoformat(),
                "user": "user1",
                "action": "数据提取",
                "status": "completed"
            },
            {
                "timestamp": (datetime.now() - timedelta(minutes=15)).isoformat(),
                "user": "user2",
                "action": "AI 预标注",
                "status": "running"
            },
            {
                "timestamp": (datetime.now() - timedelta(minutes=30)).isoformat(),
                "user": "admin",
                "action": "系统配置更新",
                "status": "completed"
            }
        ]
        
        return activities[:limit]
        
    except Exception as e:
        logger.error(f"获取最近活动失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取活动记录失败: {str(e)}"
        )


@router.get("/users")
async def get_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    admin_user=Depends(get_admin_user)
):
    """获取用户列表"""
    try:
        # In a real implementation, this would query from user management system
        users = [
            {
                "id": "user1",
                "username": "admin",
                "email": "admin@example.com",
                "role": "admin",
                "tenant_id": "default",
                "is_active": True,
                "created_at": "2024-01-01T00:00:00Z"
            },
            {
                "id": "user2",
                "username": "annotator1",
                "email": "annotator1@example.com",
                "role": "annotator",
                "tenant_id": "tenant1",
                "is_active": True,
                "created_at": "2024-01-02T00:00:00Z"
            }
        ]
        
        # Apply pagination
        start = (page - 1) * page_size
        end = start + page_size
        
        return users[start:end]
        
    except Exception as e:
        logger.error(f"获取用户列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取用户列表失败: {str(e)}"
        )


@router.post("/users")
async def create_user(
    user_data: UserManagementRequest,
    admin_user=Depends(get_admin_user)
):
    """创建用户"""
    try:
        # In a real implementation, this would create user in user management system
        new_user = {
            "id": f"user_{int(datetime.now().timestamp())}",
            "username": user_data.username,
            "email": user_data.email,
            "role": user_data.role,
            "tenant_id": user_data.tenant_id,
            "is_active": user_data.is_active,
            "created_at": datetime.now().isoformat()
        }
        
        logger.info(f"管理员 {admin_user['user_id']} 创建了用户 {user_data.username}")
        
        return {
            "message": "用户创建成功",
            "user": new_user
        }
        
    except Exception as e:
        logger.error(f"创建用户失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建用户失败: {str(e)}"
        )


@router.get("/errors/stats")
async def get_error_stats(admin_user=Depends(get_admin_user)):
    """获取错误统计"""
    try:
        # Get error statistics from error handler
        error_stats = error_handler.get_error_statistics()
        
        return {
            "total_errors": error_stats.get("total_errors", 0),
            "errors_last_hour": error_stats.get("errors_last_hour", 0),
            "error_rate": error_stats.get("error_rate", 0.0),
            "top_error_types": error_stats.get("top_error_types", [])
        }
        
    except Exception as e:
        logger.error(f"获取错误统计失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取错误统计失败: {str(e)}"
        )


@router.get("/logs/recent")
async def get_recent_logs(
    limit: int = Query(100, ge=1, le=1000),
    level: str = Query("all", regex="^(all|error|warning|info|debug)$"),
    admin_user=Depends(get_admin_user)
):
    """获取最近日志"""
    try:
        # In a real implementation, this would read from log files or log database
        logs = [
            {
                "timestamp": (datetime.now() - timedelta(minutes=1)).isoformat(),
                "level": "info",
                "message": "数据提取任务完成，提取了 150 个文档"
            },
            {
                "timestamp": (datetime.now() - timedelta(minutes=3)).isoformat(),
                "level": "warning",
                "message": "AI 模型响应时间较长: 5.2 秒"
            },
            {
                "timestamp": (datetime.now() - timedelta(minutes=5)).isoformat(),
                "level": "error",
                "message": "数据库连接失败，正在重试"
            },
            {
                "timestamp": (datetime.now() - timedelta(minutes=8)).isoformat(),
                "level": "info",
                "message": "系统启动完成，所有服务正常运行"
            }
        ]
        
        # Filter by level if specified
        if level != "all":
            logs = [log for log in logs if log["level"] == level]
        
        return logs[:limit]
        
    except Exception as e:
        logger.error(f"获取日志失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取日志失败: {str(e)}"
        )


@router.get("/settings")
async def get_settings(admin_user=Depends(get_admin_user)):
    """获取系统设置"""
    try:
        # 从配置管理器获取所有配置
        all_config = config_manager.get_all()
        
        # 返回扁平化的配置用于前端显示
        flattened_config = {}
        for section, section_config in all_config.items():
            if isinstance(section_config, dict):
                for key, value in section_config.items():
                    flattened_config[f"{section}_{key}"] = value
        
        return flattened_config
        
    except Exception as e:
        logger.error(f"获取系统设置失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取系统设置失败: {str(e)}"
        )


@router.post("/settings")
async def update_settings(
    settings: Dict[str, Any],
    admin_user=Depends(get_admin_user)
):
    """更新系统设置"""
    try:
        user_id = admin_user.get('user_id', 'unknown')
        success_count = 0
        error_count = 0
        errors = []
        
        # 解析扁平化的配置并更新
        for flat_key, value in settings.items():
            if '_' in flat_key:
                section, key = flat_key.split('_', 1)
                
                if config_manager.set(section, key, value, user=user_id, reason="管理员界面更新"):
                    success_count += 1
                else:
                    error_count += 1
                    errors.append(f"{flat_key}: 更新失败")
            else:
                error_count += 1
                errors.append(f"{flat_key}: 无效的配置键格式")
        
        logger.info(f"管理员 {user_id} 更新了系统设置: 成功 {success_count}, 失败 {error_count}")
        
        response_data = {
            "message": f"设置更新完成: 成功 {success_count}, 失败 {error_count}",
            "success_count": success_count,
            "error_count": error_count,
            "timestamp": datetime.now().isoformat()
        }
        
        if errors:
            response_data["errors"] = errors
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新系统设置失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新系统设置失败: {str(e)}"
        )


@router.post("/maintenance")
async def toggle_maintenance_mode(
    request: MaintenanceRequest,
    admin_user=Depends(get_admin_user)
):
    """切换维护模式"""
    try:
        if request.action == "enable":
            # Enable maintenance mode
            logger.info(f"管理员 {admin_user['user_id']} 启用了维护模式，持续 {request.duration_minutes} 分钟")
            
            return {
                "message": "维护模式已启用",
                "duration_minutes": request.duration_minutes,
                "maintenance_message": request.message,
                "enabled_at": datetime.now().isoformat()
            }
            
        elif request.action == "disable":
            # Disable maintenance mode
            logger.info(f"管理员 {admin_user['user_id']} 禁用了维护模式")
            
            return {
                "message": "维护模式已禁用",
                "disabled_at": datetime.now().isoformat()
            }
            
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="无效的维护操作，支持的操作: enable, disable"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"切换维护模式失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"切换维护模式失败: {str(e)}"
        )


@router.get("/billing/overview")
async def get_billing_overview(
    days: int = Query(30, ge=1, le=365),
    admin_user=Depends(get_admin_user)
):
    """获取计费概览"""
    try:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        # Get billing statistics for all tenants
        # In a real implementation, this would aggregate billing data
        overview = {
            "total_revenue": 15000.00,
            "total_annotations": 50000,
            "total_hours": 1200,
            "active_tenants": 5,
            "top_tenants": [
                {"tenant_id": "tenant1", "revenue": 8000.00, "annotations": 25000},
                {"tenant_id": "tenant2", "revenue": 4000.00, "annotations": 15000},
                {"tenant_id": "tenant3", "revenue": 2000.00, "annotations": 8000}
            ],
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days
            }
        }
        
        return overview
        
    except Exception as e:
        logger.error(f"获取计费概览失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取计费概览失败: {str(e)}"
        )


@router.post("/system/restart")
async def restart_system(admin_user=Depends(get_admin_user)):
    """重启系统服务"""
    try:
        logger.warning(f"管理员 {admin_user['user_id']} 请求重启系统")
        
        # In a real implementation, this would trigger a graceful system restart
        # For now, just return a success message
        
        return {
            "message": "系统重启请求已提交",
            "restart_initiated_at": datetime.now().isoformat(),
            "estimated_downtime_minutes": 2
        }
        
    except Exception as e:
        logger.error(f"系统重启失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"系统重启失败: {str(e)}"
        )


@router.get("/config/changes")
async def get_config_changes(
    limit: int = Query(50, ge=1, le=500),
    admin_user=Depends(get_admin_user)
):
    """获取配置变更历史"""
    try:
        changes = config_manager.get_changes(limit)
        
        return [
            {
                "timestamp": change.timestamp.isoformat(),
                "section": change.section,
                "key": change.key,
                "old_value": change.old_value,
                "new_value": change.new_value,
                "user": change.user,
                "reason": change.reason
            }
            for change in changes
        ]
        
    except Exception as e:
        logger.error(f"获取配置变更历史失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取配置变更历史失败: {str(e)}"
        )


@router.post("/config/reset")
async def reset_config(
    section: Optional[str] = None,
    admin_user=Depends(get_admin_user)
):
    """重置配置为默认值"""
    try:
        user_id = admin_user.get('user_id', 'unknown')
        
        success = config_manager.reset_to_default(
            section=section,
            user=user_id,
            reason="管理员重置配置"
        )
        
        if success:
            message = f"配置已重置为默认值: {section or 'all'}"
            logger.info(f"管理员 {user_id} {message}")
            return {"message": message, "timestamp": datetime.now().isoformat()}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="重置配置失败"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"重置配置失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"重置配置失败: {str(e)}"
        )


@router.post("/config/export")
async def export_config(
    format: str = Query("yaml", regex="^(yaml|json)$"),
    admin_user=Depends(get_admin_user)
):
    """导出配置"""
    try:
        user_id = admin_user.get('user_id', 'unknown')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"superinsight_config_{timestamp}.{format}"
        filepath = f"/tmp/{filename}"
        
        success = config_manager.export_config(filepath, format)
        
        if success:
            logger.info(f"管理员 {user_id} 导出了配置到 {filepath}")
            return {
                "message": "配置导出成功",
                "filename": filename,
                "filepath": filepath,
                "format": format,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="导出配置失败"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"导出配置失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"导出配置失败: {str(e)}"
        )


@router.post("/config/validate")
async def validate_config(admin_user=Depends(get_admin_user)):
    """验证所有配置"""
    try:
        errors = config_manager.validate_all()
        
        if errors:
            return {
                "valid": False,
                "errors": errors,
                "message": f"发现 {sum(len(errs) for errs in errors.values())} 个配置错误"
            }
        else:
            return {
                "valid": True,
                "message": "所有配置验证通过"
            }
            
    except Exception as e:
        logger.error(f"验证配置失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"验证配置失败: {str(e)}"
        )


@router.get("/health")
async def admin_health_check():
    """管理员接口健康检查"""
    return {
        "status": "healthy",
        "service": "admin",
        "timestamp": datetime.now().isoformat(),
        "version": settings.app.app_version
    }