"""
Enhanced Dashboard HTML Template for SuperInsight Platform.

Provides the HTML template for the enhanced management console
with real-time monitoring, user analytics, and workflow management.
"""

def get_enhanced_dashboard_html() -> str:
    """Get the enhanced dashboard HTML template."""
    return """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SuperInsight 增强管理控制台</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                background: #f5f7fa; 
                color: #2d3748;
            }
            
            /* Header Styles */
            .header { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; 
                padding: 1rem 2rem; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .header h1 { font-size: 1.8rem; font-weight: 600; }
            .header .subtitle { opacity: 0.9; margin-top: 0.25rem; }
            
            /* Navigation Styles */
            .nav { 
                background: white; 
                padding: 0 2rem; 
                border-bottom: 1px solid #e2e8f0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .nav-tabs {
                display: flex;
                list-style: none;
            }
            .nav-tab {
                padding: 1rem 1.5rem;
                cursor: pointer;
                border-bottom: 3px solid transparent;
                transition: all 0.3s ease;
                font-weight: 500;
            }
            .nav-tab:hover {
                background: #f7fafc;
                color: #667eea;
            }
            .nav-tab.active {
                color: #667eea;
                border-bottom-color: #667eea;
                background: #f7fafc;
            }
            
            /* Container Styles */
            .container { 
                max-width: 1400px; 
                margin: 2rem auto; 
                padding: 0 2rem; 
            }
            
            /* Card Styles */
            .card { 
                background: white; 
                border-radius: 12px; 
                box-shadow: 0 4px 6px rgba(0,0,0,0.05); 
                margin-bottom: 2rem;
                overflow: hidden;
            }
            .card-header { 
                padding: 1.5rem; 
                border-bottom: 1px solid #e2e8f0; 
                font-weight: 600; 
                font-size: 1.1rem;
                background: #f8fafc;
            }
            .card-body { padding: 1.5rem; }
            
            /* Grid Styles */
            .grid { 
                display: grid; 
                gap: 2rem; 
            }
            .grid-2 { grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }
            .grid-3 { grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); }
            .grid-4 { grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }
            
            /* Metric Styles */
            .metric { 
                text-align: center; 
                padding: 1.5rem; 
                border-radius: 8px;
                background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            }
            .metric-value { 
                font-size: 2.5rem; 
                font-weight: bold; 
                margin-bottom: 0.5rem;
            }
            .metric-label { 
                color: #718096; 
                font-size: 0.9rem;
                font-weight: 500;
            }
            .metric-trend {
                font-size: 0.8rem;
                margin-top: 0.25rem;
            }
            
            /* Status Colors */
            .status-healthy { color: #38a169; }
            .status-warning { color: #d69e2e; }
            .status-error { color: #e53e3e; }
            .status-critical { color: #c53030; }
            
            /* Button Styles */
            .btn { 
                padding: 0.75rem 1.5rem; 
                border: none; 
                border-radius: 8px; 
                cursor: pointer; 
                font-weight: 500;
                transition: all 0.3s ease;
                text-decoration: none; 
                display: inline-block;
            }
            .btn-primary { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; 
            }
            .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4); }
            .btn-success { background: #38a169; color: white; }
            .btn-warning { background: #d69e2e; color: white; }
            .btn-danger { background: #e53e3e; color: white; }
            .btn-secondary { background: #718096; color: white; }
            
            /* Table Styles */
            .table { 
                width: 100%; 
                border-collapse: collapse; 
            }
            .table th, .table td { 
                padding: 1rem; 
                text-align: left; 
                border-bottom: 1px solid #e2e8f0; 
            }
            .table th { 
                background: #f8fafc; 
                font-weight: 600; 
                color: #4a5568;
            }
            .table tr:hover { background: #f7fafc; }
            
            /* Alert Styles */
            .alert {
                padding: 1rem;
                border-radius: 8px;
                margin-bottom: 1rem;
                border-left: 4px solid;
            }
            .alert-info { background: #ebf8ff; border-color: #3182ce; color: #2c5282; }
            .alert-success { background: #f0fff4; border-color: #38a169; color: #276749; }
            .alert-warning { background: #fffbeb; border-color: #d69e2e; color: #975a16; }
            .alert-error { background: #fed7d7; border-color: #e53e3e; color: #c53030; }
            
            /* Loading Styles */
            .loading { 
                text-align: center; 
                padding: 3rem; 
                color: #718096; 
            }
            .spinner {
                border: 3px solid #e2e8f0;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                width: 30px;
                height: 30px;
                animation: spin 1s linear infinite;
                margin: 0 auto 1rem;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            /* Section Styles */
            .section { display: none; }
            .section.active { display: block; }
            
            /* Real-time indicator */
            .realtime-indicator {
                display: inline-block;
                width: 8px;
                height: 8px;
                background: #38a169;
                border-radius: 50%;
                margin-right: 0.5rem;
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            
            /* Chart container */
            .chart-container {
                height: 300px;
                position: relative;
            }
            
            /* Workflow visualization */
            .workflow-canvas {
                width: 100%;
                height: 400px;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
            }
            
            /* Configuration form */
            .config-form {
                display: grid;
                gap: 1rem;
            }
            .form-group {
                display: flex;
                flex-direction: column;
            }
            .form-label {
                font-weight: 500;
                margin-bottom: 0.5rem;
                color: #4a5568;
            }
            .form-input {
                padding: 0.75rem;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                font-size: 0.9rem;
            }
            .form-input:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>SuperInsight 增强管理控制台</h1>
            <div class="subtitle">实时监控 • 用户分析 • 工作流管理 • 配置热更新</div>
        </div>
        
        <nav class="nav">
            <ul class="nav-tabs">
                <li class="nav-tab active" onclick="showSection('realtime-monitoring')">
                    <span class="realtime-indicator"></span>实时监控
                </li>
                <li class="nav-tab" onclick="showSection('user-analytics')">用户分析</li>
                <li class="nav-tab" onclick="showSection('workflow-management')">工作流管理</li>
                <li class="nav-tab" onclick="showSection('config-management')">配置管理</li>
                <li class="nav-tab" onclick="showSection('system-overview')">系统概览</li>
            </ul>
        </nav>
        
        <div class="container">
            <!-- Real-time Monitoring Section -->
            <div id="realtime-monitoring" class="section active">
                <div class="grid grid-4">
                    <div class="metric">
                        <div class="metric-value status-healthy" id="cpu-usage">-</div>
                        <div class="metric-label">CPU 使用率</div>
                        <div class="metric-trend" id="cpu-trend">-</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value status-healthy" id="memory-usage">-</div>
                        <div class="metric-label">内存使用率</div>
                        <div class="metric-trend" id="memory-trend">-</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="active-users">-</div>
                        <div class="metric-label">活跃用户</div>
                        <div class="metric-trend" id="users-trend">-</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="response-time">-</div>
                        <div class="metric-label">响应时间</div>
                        <div class="metric-trend" id="response-trend">-</div>
                    </div>
                </div>
                
                <div class="grid grid-2">
                    <div class="card">
                        <div class="card-header">系统性能趋势</div>
                        <div class="card-body">
                            <div class="chart-container" id="performance-chart">
                                <div class="loading">
                                    <div class="spinner"></div>
                                    加载性能数据中...
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">实时告警</div>
                        <div class="card-body">
                            <div id="alerts-container">
                                <div class="loading">加载告警数据中...</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">服务状态</div>
                    <div class="card-body">
                        <div id="services-status" class="loading">加载服务状态中...</div>
                    </div>
                </div>
            </div>
            
            <!-- User Analytics Section -->
            <div id="user-analytics" class="section">
                <div class="grid grid-4">
                    <div class="metric">
                        <div class="metric-value" id="total-users">-</div>
                        <div class="metric-label">总用户数</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="active-sessions">-</div>
                        <div class="metric-label">活跃会话</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="actions-per-minute">-</div>
                        <div class="metric-label">每分钟操作数</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="avg-session-duration">-</div>
                        <div class="metric-label">平均会话时长</div>
                    </div>
                </div>
                
                <div class="grid grid-2">
                    <div class="card">
                        <div class="card-header">用户活动热力图</div>
                        <div class="card-body">
                            <div class="chart-container" id="activity-heatmap">
                                <div class="loading">加载活动数据中...</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">操作类型分布</div>
                        <div class="card-body">
                            <div class="chart-container" id="action-distribution">
                                <div class="loading">加载操作数据中...</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        用户行为分析报告
                        <button class="btn btn-primary" style="float: right;" onclick="generateUserReport()">
                            生成报告
                        </button>
                    </div>
                    <div class="card-body">
                        <div id="user-report-container">
                            <p>点击"生成报告"按钮获取详细的用户行为分析报告</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Workflow Management Section -->
            <div id="workflow-management" class="section">
                <div class="card">
                    <div class="card-header">
                        工作流管理
                        <button class="btn btn-primary" style="float: right;" onclick="showCreateWorkflowModal()">
                            创建工作流
                        </button>
                    </div>
                    <div class="card-body">
                        <div class="grid grid-4" style="margin-bottom: 2rem;">
                            <div class="metric">
                                <div class="metric-value" id="total-workflows">-</div>
                                <div class="metric-label">总工作流数</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value status-healthy" id="running-workflows">-</div>
                                <div class="metric-label">运行中</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value status-warning" id="pending-workflows">-</div>
                                <div class="metric-label">等待中</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value" id="completed-workflows">-</div>
                                <div class="metric-label">已完成</div>
                            </div>
                        </div>
                        
                        <table class="table">
                            <thead>
                                <tr>
                                    <th>工作流名称</th>
                                    <th>状态</th>
                                    <th>进度</th>
                                    <th>创建时间</th>
                                    <th>操作</th>
                                </tr>
                            </thead>
                            <tbody id="workflows-table">
                                <tr><td colspan="5" class="loading">加载工作流数据中...</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">工作流可视化</div>
                    <div class="card-body">
                        <div id="workflow-visualization">
                            <p>选择一个工作流以查看其可视化图表</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Configuration Management Section -->
            <div id="config-management" class="section">
                <div class="grid grid-2">
                    <div class="card">
                        <div class="card-header">配置热更新</div>
                        <div class="card-body">
                            <form class="config-form" onsubmit="updateConfiguration(event)">
                                <div class="form-group">
                                    <label class="form-label">配置节</label>
                                    <select class="form-input" id="config-section" required>
                                        <option value="">选择配置节</option>
                                        <option value="system">系统配置</option>
                                        <option value="database">数据库配置</option>
                                        <option value="ai">AI 配置</option>
                                        <option value="monitoring">监控配置</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label class="form-label">配置键</label>
                                    <input type="text" class="form-input" id="config-key" 
                                           placeholder="例如: api_rate_limit" required>
                                </div>
                                <div class="form-group">
                                    <label class="form-label">配置值</label>
                                    <input type="text" class="form-input" id="config-value" 
                                           placeholder="输入新的配置值" required>
                                </div>
                                <button type="submit" class="btn btn-primary">应用配置</button>
                            </form>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">配置变更历史</div>
                        <div class="card-body">
                            <div id="config-changes" class="loading">加载配置变更历史中...</div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">当前配置状态</div>
                    <div class="card-body">
                        <div id="current-config" class="loading">加载当前配置中...</div>
                    </div>
                </div>
            </div>
            
            <!-- System Overview Section -->
            <div id="system-overview" class="section">
                <div class="card">
                    <div class="card-header">系统概览</div>
                    <div class="card-body">
                        <div id="system-overview-content" class="loading">
                            <div class="spinner"></div>
                            加载系统概览数据中...
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Global variables
            let currentSection = 'realtime-monitoring';
            let websocket = null;
            let refreshInterval = null;
            
            // Initialize dashboard
            document.addEventListener('DOMContentLoaded', function() {
                initializeWebSocket();
                loadInitialData();
                startPeriodicUpdates();
            });
            
            // WebSocket connection for real-time updates
            function initializeWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/admin/enhanced/monitoring/realtime`;
                
                websocket = new WebSocket(wsUrl);
                
                websocket.onopen = function(event) {
                    console.log('WebSocket connected');
                    showAlert('实时监控连接已建立', 'success');
                };
                
                websocket.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        updateRealTimeMetrics(data);
                    } catch (error) {
                        console.error('Failed to parse WebSocket message:', error);
                    }
                };
                
                websocket.onclose = function(event) {
                    console.log('WebSocket disconnected');
                    showAlert('实时监控连接已断开，尝试重连中...', 'warning');
                    
                    // Attempt to reconnect after 5 seconds
                    setTimeout(initializeWebSocket, 5000);
                };
                
                websocket.onerror = function(error) {
                    console.error('WebSocket error:', error);
                    showAlert('实时监控连接出错', 'error');
                };
            }
            
            // Update real-time metrics from WebSocket data
            function updateRealTimeMetrics(data) {
                if (data.performance_metrics) {
                    const perf = data.performance_metrics;
                    
                    updateMetric('cpu-usage', perf.cpu_usage, '%');
                    updateMetric('memory-usage', perf.memory_usage, '%');
                    updateMetric('response-time', perf.response_time, 's');
                }
                
                if (data.user_activity) {
                    const activity = data.user_activity;
                    updateMetric('active-users', activity.active_sessions);
                }
                
                if (data.alerts) {
                    updateAlerts(data.alerts);
                }
                
                if (data.system_health && data.system_health.services) {
                    updateServicesStatus(data.system_health.services);
                }
            }
            
            // Update individual metric display
            function updateMetric(elementId, value, unit = '') {
                const element = document.getElementById(elementId);
                if (element && value !== undefined && value !== null) {
                    let displayValue = value;
                    
                    if (typeof value === 'number') {
                        if (unit === '%') {
                            displayValue = value.toFixed(1) + '%';
                        } else if (unit === 's') {
                            displayValue = value.toFixed(2) + 's';
                        } else {
                            displayValue = Math.round(value);
                        }
                    }
                    
                    element.textContent = displayValue;
                    
                    // Update status color based on value
                    if (unit === '%' && typeof value === 'number') {
                        element.className = 'metric-value ' + getStatusClass(value);
                    }
                }
            }
            
            // Get status class based on percentage value
            function getStatusClass(percentage) {
                if (percentage >= 90) return 'status-critical';
                if (percentage >= 75) return 'status-error';
                if (percentage >= 60) return 'status-warning';
                return 'status-healthy';
            }
            
            // Update alerts display
            function updateAlerts(alerts) {
                const container = document.getElementById('alerts-container');
                if (!container) return;
                
                if (!alerts || alerts.length === 0) {
                    container.innerHTML = '<p class="text-muted">暂无告警</p>';
                    return;
                }
                
                const alertsHtml = alerts.map(alert => `
                    <div class="alert alert-${alert.level === 'critical' ? 'error' : alert.level}">
                        <strong>${alert.level.toUpperCase()}</strong>: ${alert.message}
                        <small style="float: right;">${new Date(alert.timestamp * 1000).toLocaleTimeString()}</small>
                    </div>
                `).join('');
                
                container.innerHTML = alertsHtml;
            }
            
            // Update services status
            function updateServicesStatus(services) {
                const container = document.getElementById('services-status');
                if (!container) return;
                
                const servicesHtml = Object.entries(services).map(([name, info]) => `
                    <div style="margin-bottom: 1rem; padding: 1rem; border: 1px solid #e2e8f0; border-radius: 8px;">
                        <strong>${name}</strong>: 
                        <span class="status-${info.status === 'healthy' ? 'healthy' : 'error'}">${info.status}</span>
                        ${info.message ? `<br><small>${info.message}</small>` : ''}
                    </div>
                `).join('');
                
                container.innerHTML = servicesHtml;
            }
            
            // Section navigation
            function showSection(sectionName) {
                // Hide all sections
                document.querySelectorAll('.section').forEach(section => {
                    section.classList.remove('active');
                });
                
                // Show selected section
                document.getElementById(sectionName).classList.add('active');
                
                // Update navigation
                document.querySelectorAll('.nav-tab').forEach(tab => {
                    tab.classList.remove('active');
                });
                event.target.classList.add('active');
                
                currentSection = sectionName;
                
                // Load section-specific data
                loadSectionData(sectionName);
            }
            
            // Load section-specific data
            function loadSectionData(sectionName) {
                switch(sectionName) {
                    case 'user-analytics':
                        loadUserAnalytics();
                        break;
                    case 'workflow-management':
                        loadWorkflowManagement();
                        break;
                    case 'config-management':
                        loadConfigManagement();
                        break;
                    case 'system-overview':
                        loadSystemOverview();
                        break;
                }
            }
            
            // Load initial data
            function loadInitialData() {
                loadSectionData(currentSection);
            }
            
            // Start periodic updates for non-real-time data
            function startPeriodicUpdates() {
                refreshInterval = setInterval(() => {
                    if (currentSection !== 'realtime-monitoring') {
                        loadSectionData(currentSection);
                    }
                }, 30000); // Update every 30 seconds
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
            
            // Show alert message
            function showAlert(message, type = 'info') {
                // Create alert element
                const alert = document.createElement('div');
                alert.className = `alert alert-${type}`;
                alert.innerHTML = `
                    ${message}
                    <button onclick="this.parentElement.remove()" style="float: right; background: none; border: none; font-size: 1.2rem; cursor: pointer;">&times;</button>
                `;
                
                // Insert at top of container
                const container = document.querySelector('.container');
                container.insertBefore(alert, container.firstChild);
                
                // Auto-remove after 5 seconds
                setTimeout(() => {
                    if (alert.parentElement) {
                        alert.remove();
                    }
                }, 5000);
            }
            
            // Placeholder functions for features to be implemented
            function loadUserAnalytics() {
                console.log('Loading user analytics...');
            }
            
            function loadWorkflowManagement() {
                console.log('Loading workflow management...');
            }
            
            function loadConfigManagement() {
                console.log('Loading config management...');
            }
            
            function loadSystemOverview() {
                console.log('Loading system overview...');
            }
            
            function generateUserReport() {
                showAlert('用户报告生成功能开发中...', 'info');
            }
            
            function showCreateWorkflowModal() {
                showAlert('工作流创建功能开发中...', 'info');
            }
            
            function updateConfiguration(event) {
                event.preventDefault();
                showAlert('配置更新功能开发中...', 'info');
            }
        </script>
    </body>
    </html>
    """