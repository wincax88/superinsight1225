"""
Tests for Enhanced Admin Console Features.

Tests the enhanced management console functionality including:
- Real-time monitoring
- User analytics
- Configuration hot updates
- Workflow management
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from src.admin.dashboard import (
    RealTimeMonitoringService, 
    ConfigurationHotUpdateService, 
    EnhancedDashboardManager
)
from src.admin.user_analytics import UserAnalytics, ActionType
from src.admin.workflow_manager import WorkflowManager, WorkflowStatus


class TestRealTimeMonitoringService:
    """Test real-time monitoring service."""
    
    @pytest.fixture
    def monitoring_service(self):
        """Create monitoring service instance."""
        return RealTimeMonitoringService()
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, monitoring_service):
        """Test starting and stopping monitoring service."""
        assert not monitoring_service.is_running
        
        await monitoring_service.start_monitoring()
        assert monitoring_service.is_running
        
        await monitoring_service.stop_monitoring()
        assert not monitoring_service.is_running
    
    def test_add_remove_subscribers(self, monitoring_service):
        """Test WebSocket subscriber management."""
        mock_websocket = Mock()
        
        # Add subscriber
        monitoring_service.add_subscriber(mock_websocket)
        assert len(monitoring_service.subscribers) == 1
        
        # Remove subscriber
        monitoring_service.remove_subscriber(mock_websocket)
        assert len(monitoring_service.subscribers) == 0
    
    def test_generate_real_time_alerts(self, monitoring_service):
        """Test real-time alert generation."""
        system_metrics = {
            "system.cpu.usage_percent": {"latest": 95.0},
            "system.memory.usage_percent": {"latest": 85.0},
            "requests.duration": {"avg": 3.0}
        }
        
        performance_summary = {"bottlenecks": []}
        
        alerts = monitoring_service._generate_real_time_alerts(
            system_metrics, performance_summary
        )
        
        # Should generate CPU critical alert
        cpu_alerts = [a for a in alerts if a["type"] == "cpu_usage"]
        assert len(cpu_alerts) == 1
        assert cpu_alerts[0]["level"] == "critical"
        
        # Should generate memory warning alert
        memory_alerts = [a for a in alerts if a["type"] == "memory_usage"]
        assert len(memory_alerts) == 1
        assert memory_alerts[0]["level"] == "warning"
        
        # Should generate response time critical alert
        response_alerts = [a for a in alerts if a["type"] == "response_time"]
        assert len(response_alerts) == 1
        assert response_alerts[0]["level"] == "critical"


class TestConfigurationHotUpdateService:
    """Test configuration hot update service."""
    
    @pytest.fixture
    def config_service(self):
        """Create configuration service instance."""
        return ConfigurationHotUpdateService()
    
    @pytest.mark.asyncio
    async def test_update_config_success(self, config_service):
        """Test successful configuration update."""
        with patch('src.admin.config_manager.config_manager') as mock_config:
            mock_config.get.return_value = 100
            mock_config.set.return_value = True
            
            result = await config_service.update_config(
                section="system",
                key="api_rate_limit",
                value=200,
                user="test_user"
            )
            
            assert result["success"] is True
            assert result["old_value"] == 100
            assert result["new_value"] == 200
    
    @pytest.mark.asyncio
    async def test_update_config_validation_failure(self, config_service):
        """Test configuration update with validation failure."""
        result = await config_service.update_config(
            section="system",
            key="api_rate_limit",
            value=-1,  # Invalid value
            user="test_user"
        )
        
        assert result["success"] is False
        assert "Validation failed" in result["error"]
    
    @pytest.mark.asyncio
    async def test_rollback_config(self, config_service):
        """Test configuration rollback."""
        # First, add a change to history
        config_service.change_history.append({
            "timestamp": datetime.now().timestamp(),
            "section": "system",
            "key": "api_rate_limit",
            "old_value": 100,
            "new_value": 200,
            "user": "test_user",
            "status": "applied"
        })
        
        with patch.object(config_service, 'update_config') as mock_update:
            mock_update.return_value = {"success": True}
            
            result = await config_service.rollback_config(0, "admin")
            
            assert result["success"] is True
            mock_update.assert_called_once()


class TestUserAnalytics:
    """Test user analytics service."""
    
    @pytest.fixture
    def user_analytics(self):
        """Create user analytics instance."""
        return UserAnalytics()
    
    @pytest.mark.asyncio
    async def test_initialize_shutdown(self, user_analytics):
        """Test service initialization and shutdown."""
        await user_analytics.initialize()
        assert user_analytics.is_running
        
        await user_analytics.shutdown()
        assert not user_analytics.is_running
    
    def test_start_end_session(self, user_analytics):
        """Test session tracking."""
        session_id = "test_session_123"
        user_id = "test_user"
        
        # Start session
        user_analytics.start_session(session_id, user_id)
        assert session_id in user_analytics.sessions
        
        session = user_analytics.sessions[session_id]
        assert session.user_id == user_id
        assert session.is_active
        
        # End session
        user_analytics.end_session(session_id)
        assert session_id not in user_analytics.sessions
    
    def test_track_action(self, user_analytics):
        """Test action tracking."""
        user_id = "test_user"
        session_id = "test_session"
        
        # Start session first
        user_analytics.start_session(session_id, user_id)
        
        # Track action
        user_analytics.track_action(
            user_id=user_id,
            session_id=session_id,
            action_type=ActionType.ANNOTATION,
            details={"task_id": "task_123"},
            duration_ms=1500.0,
            success=True
        )
        
        # Check action was recorded
        assert len(user_analytics.actions) > 0
        
        action = user_analytics.actions[-1]
        assert action.user_id == user_id
        assert action.action_type == ActionType.ANNOTATION
        assert action.success is True
    
    def test_get_current_stats(self, user_analytics):
        """Test getting current analytics statistics."""
        # Add some test data
        user_analytics.start_session("session1", "user1")
        user_analytics.track_action("user1", "session1", ActionType.LOGIN)
        
        stats = user_analytics.get_current_stats()
        
        assert "active_sessions" in stats
        assert "recent_actions" in stats
        assert "unique_users_today" in stats
        assert stats["active_sessions"] >= 1


class TestWorkflowManager:
    """Test workflow management service."""
    
    @pytest.fixture
    def workflow_manager(self):
        """Create workflow manager instance."""
        return WorkflowManager()
    
    @pytest.mark.asyncio
    async def test_initialize_shutdown(self, workflow_manager):
        """Test service initialization and shutdown."""
        await workflow_manager.initialize()
        assert workflow_manager.is_running
        
        await workflow_manager.shutdown()
        assert not workflow_manager.is_running
    
    def test_create_workflow_from_template(self, workflow_manager):
        """Test creating workflow from template."""
        workflow_id = workflow_manager.create_workflow_from_template(
            template_id="data_processing",
            name="Test Data Processing",
            parameters={"batch_size": 50},
            created_by="test_user"
        )
        
        assert workflow_id in workflow_manager.workflows
        
        workflow = workflow_manager.workflows[workflow_id]
        assert workflow.name == "Test Data Processing"
        assert workflow.status == WorkflowStatus.PENDING
        assert len(workflow.tasks) > 0
    
    def test_create_custom_workflow(self, workflow_manager):
        """Test creating custom workflow."""
        task_definitions = [
            {
                "task_id": "task1",
                "name": "First Task",
                "task_type": "test_task",
                "dependencies": [],
                "parameters": {}
            },
            {
                "task_id": "task2",
                "name": "Second Task",
                "task_type": "test_task",
                "dependencies": ["task1"],
                "parameters": {}
            }
        ]
        
        workflow_id = workflow_manager.create_custom_workflow(
            name="Custom Test Workflow",
            description="Test workflow",
            task_definitions=task_definitions,
            created_by="test_user"
        )
        
        assert workflow_id in workflow_manager.workflows
        
        workflow = workflow_manager.workflows[workflow_id]
        assert len(workflow.tasks) == 2
        assert "task1" in workflow.tasks
        assert "task2" in workflow.tasks
        assert workflow.tasks["task2"].dependencies == ["task1"]
    
    def test_workflow_control_operations(self, workflow_manager):
        """Test workflow control operations (start, pause, resume, cancel)."""
        # Create a test workflow
        workflow_id = workflow_manager.create_workflow_from_template(
            template_id="data_processing",
            name="Control Test Workflow"
        )
        
        # Test start
        success = workflow_manager.start_workflow(workflow_id)
        assert success is True
        
        # Test pause
        workflow = workflow_manager.workflows[workflow_id]
        workflow.status = WorkflowStatus.RUNNING  # Simulate running state
        
        success = workflow_manager.pause_workflow(workflow_id)
        assert success is True
        assert workflow.status == WorkflowStatus.PAUSED
        
        # Test resume
        success = workflow_manager.resume_workflow(workflow_id)
        assert success is True
        assert workflow.status == WorkflowStatus.RUNNING
        
        # Test cancel
        success = workflow_manager.cancel_workflow(workflow_id)
        assert success is True
        assert workflow.status == WorkflowStatus.CANCELLED
    
    def test_get_workflow_visualization_data(self, workflow_manager):
        """Test workflow visualization data generation."""
        workflow_id = workflow_manager.create_workflow_from_template(
            template_id="data_processing",
            name="Visualization Test"
        )
        
        viz_data = workflow_manager.get_workflow_visualization_data(workflow_id)
        
        assert viz_data is not None
        assert "workflow" in viz_data
        assert "visualization" in viz_data
        assert "nodes" in viz_data["visualization"]
        assert "edges" in viz_data["visualization"]
        
        # Check nodes
        nodes = viz_data["visualization"]["nodes"]
        assert len(nodes) > 0
        
        # Check edges (dependencies)
        edges = viz_data["visualization"]["edges"]
        assert isinstance(edges, list)


class TestEnhancedDashboardManager:
    """Test enhanced dashboard manager."""
    
    @pytest.fixture
    def dashboard_manager(self):
        """Create dashboard manager instance."""
        return EnhancedDashboardManager()
    
    @pytest.mark.asyncio
    async def test_initialize_shutdown(self, dashboard_manager):
        """Test dashboard manager initialization and shutdown."""
        with patch.object(dashboard_manager.monitoring_service, 'start_monitoring') as mock_start_monitoring, \
             patch.object(dashboard_manager.user_analytics, 'initialize') as mock_user_init, \
             patch.object(dashboard_manager.workflow_manager, 'initialize') as mock_workflow_init:
            
            await dashboard_manager.initialize()
            assert dashboard_manager.is_initialized
            
            mock_start_monitoring.assert_called_once()
            mock_user_init.assert_called_once()
            mock_workflow_init.assert_called_once()
        
        with patch.object(dashboard_manager.monitoring_service, 'stop_monitoring') as mock_stop_monitoring, \
             patch.object(dashboard_manager.user_analytics, 'shutdown') as mock_user_shutdown, \
             patch.object(dashboard_manager.workflow_manager, 'shutdown') as mock_workflow_shutdown:
            
            await dashboard_manager.shutdown()
            assert not dashboard_manager.is_initialized
            
            mock_stop_monitoring.assert_called_once()
            mock_user_shutdown.assert_called_once()
            mock_workflow_shutdown.assert_called_once()
    
    def test_get_dashboard_overview(self, dashboard_manager):
        """Test getting dashboard overview."""
        with patch.object(dashboard_manager.monitoring_service, 'get_current_metrics') as mock_metrics, \
             patch.object(dashboard_manager.user_analytics, 'get_current_stats') as mock_user_stats, \
             patch.object(dashboard_manager.workflow_manager, 'get_workflow_stats') as mock_workflow_stats:
            
            mock_metrics.return_value = {
                "system_health": {"overall_status": "healthy"},
                "performance_metrics": {"cpu_usage": 45.0},
                "alerts": []
            }
            mock_user_stats.return_value = {"active_sessions": 5}
            mock_workflow_stats.return_value = {"active_workflows": 3}
            
            overview = dashboard_manager.get_dashboard_overview()
            
            assert "timestamp" in overview
            assert "system_status" in overview
            assert "user_analytics" in overview
            assert "workflow_status" in overview
            assert overview["user_analytics"]["active_sessions"] == 5
            assert overview["workflow_status"]["active_workflows"] == 3


@pytest.mark.integration
class TestEnhancedAdminIntegration:
    """Integration tests for enhanced admin features."""
    
    @pytest.mark.asyncio
    async def test_full_dashboard_workflow(self):
        """Test complete dashboard workflow integration."""
        dashboard_manager = EnhancedDashboardManager()
        
        try:
            # Initialize dashboard
            await dashboard_manager.initialize()
            
            # Test user session tracking
            dashboard_manager.user_analytics.start_session(
                session_id="integration_test_session",
                user_id="integration_test_user"
            )
            
            # Test action tracking
            dashboard_manager.user_analytics.track_action(
                user_id="integration_test_user",
                session_id="integration_test_session",
                action_type=ActionType.ANNOTATION,
                details={"test": "integration"}
            )
            
            # Test workflow creation
            workflow_id = dashboard_manager.workflow_manager.create_workflow_from_template(
                template_id="data_processing",
                name="Integration Test Workflow"
            )
            
            # Test configuration update
            config_result = await dashboard_manager.config_service.update_config(
                section="system",
                key="test_config",
                value="integration_test_value",
                user="integration_test_user"
            )
            
            # Get dashboard overview
            overview = dashboard_manager.get_dashboard_overview()
            
            # Verify integration
            assert overview is not None
            assert "timestamp" in overview
            assert workflow_id in dashboard_manager.workflow_manager.workflows
            
        finally:
            # Cleanup
            await dashboard_manager.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])