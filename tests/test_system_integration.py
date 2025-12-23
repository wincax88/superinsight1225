"""
Tests for system integration functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch

from src.system.integration import SystemIntegrationManager, ServiceStatus
from src.system.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
from src.system.monitoring import MetricsCollector, PerformanceMonitor
from src.system.health import HealthChecker, HealthStatus
from src.system.service_registry import ServiceRegistry, ServiceType


class TestSystemIntegration:
    """Test system integration manager."""
    
    def test_service_registration(self):
        """Test service registration."""
        manager = SystemIntegrationManager()
        
        # Register a test service
        def test_startup():
            return True
        
        def test_health():
            return True
        
        manager.register_service(
            name="test_service",
            startup_func=test_startup,
            health_check=test_health
        )
        
        assert "test_service" in manager.services
        assert manager.services["test_service"].name == "test_service"
        assert manager.services["test_service"].status == ServiceStatus.STOPPED
    
    @pytest.mark.asyncio
    async def test_service_startup(self):
        """Test service startup."""
        manager = SystemIntegrationManager()
        
        startup_called = False
        
        def test_startup():
            nonlocal startup_called
            startup_called = True
        
        def test_health():
            return True
        
        manager.register_service(
            name="test_service",
            startup_func=test_startup,
            health_check=test_health
        )
        
        success = await manager.start_service("test_service")
        
        assert success
        assert startup_called
        assert manager.services["test_service"].status == ServiceStatus.HEALTHY
    
    @pytest.mark.asyncio
    async def test_service_dependencies(self):
        """Test service dependency handling."""
        manager = SystemIntegrationManager()
        
        # Register dependency service
        manager.register_service(
            name="dependency",
            startup_func=lambda: None,
            health_check=lambda: True
        )
        
        # Register dependent service
        manager.register_service(
            name="dependent",
            startup_func=lambda: None,
            health_check=lambda: True,
            dependencies=["dependency"]
        )
        
        # Start dependency first
        await manager.start_service("dependency")
        
        # Start dependent service
        success = await manager.start_service("dependent")
        assert success
    
    def test_system_status(self):
        """Test system status reporting."""
        manager = SystemIntegrationManager()
        
        manager.register_service(
            name="test_service",
            health_check=lambda: True
        )
        
        status = manager.get_system_status()
        
        assert "overall_status" in status
        assert "services" in status
        assert "test_service" in status["services"]


class TestErrorHandler:
    """Test error handling system."""
    
    def test_error_handling(self):
        """Test basic error handling."""
        handler = ErrorHandler()
        
        test_exception = Exception("Test error")
        
        error_context = handler.handle_error(
            exception=test_exception,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.MEDIUM,
            service_name="test_service"
        )
        
        assert error_context.category == ErrorCategory.SYSTEM
        assert error_context.severity == ErrorSeverity.MEDIUM
        assert error_context.service_name == "test_service"
        assert "Test error" in error_context.message
    
    def test_error_statistics(self):
        """Test error statistics collection."""
        handler = ErrorHandler()
        
        # Generate some test errors
        for i in range(5):
            handler.handle_error(
                exception=Exception(f"Error {i}"),
                category=ErrorCategory.DATABASE,
                severity=ErrorSeverity.LOW
            )
        
        stats = handler.get_error_statistics()
        
        assert stats["total_errors"] == 5
        assert "database" in stats["by_category"]
        assert stats["by_category"]["database"] == 5


class TestMetricsCollector:
    """Test metrics collection system."""
    
    def test_metric_registration(self):
        """Test metric registration."""
        collector = MetricsCollector()
        
        collector.register_metric("test_metric", unit="count", description="Test metric")
        
        assert "test_metric" in collector.metrics
        assert collector.metrics["test_metric"].unit == "count"
        assert collector.metrics["test_metric"].description == "Test metric"
    
    def test_metric_recording(self):
        """Test metric value recording."""
        collector = MetricsCollector()
        
        collector.record_metric("test_metric", 42.0)
        collector.record_metric("test_metric", 24.0)
        
        values = collector.get_metric_values("test_metric")
        assert len(values) == 2
        assert values[0].value == 42.0
        assert values[1].value == 24.0
    
    def test_metric_summary(self):
        """Test metric summary calculation."""
        collector = MetricsCollector()
        
        collector.record_metric("test_metric", 10.0)
        collector.record_metric("test_metric", 20.0)
        collector.record_metric("test_metric", 30.0)
        
        summary = collector.get_metric_summary("test_metric")
        
        assert summary["count"] == 3
        assert summary["min"] == 10.0
        assert summary["max"] == 30.0
        assert summary["avg"] == 20.0
        assert summary["latest"] == 30.0


class TestHealthChecker:
    """Test health checking system."""
    
    @pytest.mark.asyncio
    async def test_health_check_registration(self):
        """Test health check registration."""
        checker = HealthChecker()
        
        def test_check():
            return True
        
        checker.register_check("test_check", test_check)
        
        assert "test_check" in checker.checks
    
    @pytest.mark.asyncio
    async def test_health_check_execution(self):
        """Test health check execution."""
        checker = HealthChecker()
        
        def test_check():
            return {
                "status": "healthy",
                "message": "All good"
            }
        
        checker.register_check("test_check", test_check)
        
        result = await checker.run_check("test_check")
        
        assert result.name == "test_check"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "All good"
    
    @pytest.mark.asyncio
    async def test_system_health(self):
        """Test system health aggregation."""
        checker = HealthChecker()
        
        def healthy_check():
            return True
        
        def unhealthy_check():
            return False
        
        checker.register_check("healthy", healthy_check)
        checker.register_check("unhealthy", unhealthy_check)
        
        health = await checker.get_system_health()
        
        assert health["overall_status"] == "unhealthy"  # One unhealthy check
        assert health["summary"]["total_checks"] == 2
        assert health["summary"]["healthy"] == 1
        assert health["summary"]["unhealthy"] == 1


class TestServiceRegistry:
    """Test service registry functionality."""
    
    def test_service_registration(self):
        """Test service registration in registry."""
        registry = ServiceRegistry()
        
        registry.register_service(
            name="test_service",
            service_type=ServiceType.FEATURE,
            version="1.0.0"
        )
        
        assert "test_service" in registry.services
        service = registry.get_service("test_service")
        assert service.name == "test_service"
        assert service.service_type == ServiceType.FEATURE
        assert service.version == "1.0.0"
    
    def test_service_dependencies(self):
        """Test service dependency validation."""
        registry = ServiceRegistry()
        
        registry.register_service(
            name="service_a",
            service_type=ServiceType.CORE,
            dependencies=["service_b"]  # Missing dependency
        )
        
        missing_deps = registry.validate_dependencies()
        
        assert "service_a" in missing_deps
        assert "service_b" in missing_deps["service_a"]
    
    def test_startup_order(self):
        """Test startup order calculation."""
        registry = ServiceRegistry()
        
        registry.register_service(
            name="database",
            service_type=ServiceType.CORE
        )
        
        registry.register_service(
            name="api",
            service_type=ServiceType.FEATURE,
            dependencies=["database"]
        )
        
        order = registry.get_startup_order()
        
        # Database should come before API
        db_index = order.index("database")
        api_index = order.index("api")
        assert db_index < api_index


class TestPerformanceMonitor:
    """Test performance monitoring."""
    
    def test_request_tracking(self):
        """Test request performance tracking."""
        collector = MetricsCollector()
        monitor = PerformanceMonitor(collector)
        
        # Simulate request tracking
        monitor.start_request("req_123", "/api/test")
        monitor.end_request("req_123", "/api/test", 200)
        
        # Check metrics were recorded
        summary = collector.get_metric_summary("requests.total")
        assert summary["count"] > 0
    
    def test_database_query_tracking(self):
        """Test database query performance tracking."""
        collector = MetricsCollector()
        monitor = PerformanceMonitor(collector)
        
        monitor.record_database_query("SELECT", 0.05, True)
        
        summary = collector.get_metric_summary("database.query.duration")
        assert summary["count"] == 1
        assert summary["latest"] == 0.05


if __name__ == "__main__":
    pytest.main([__file__])