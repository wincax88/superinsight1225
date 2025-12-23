#!/usr/bin/env python3
"""
Simple integration test for SuperInsight Platform system integration.
"""

import sys
import asyncio
import logging

# Add src to path
sys.path.insert(0, 'src')

from src.system.integration import SystemIntegrationManager
from src.system.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
from src.system.monitoring import MetricsCollector
from src.system.health import HealthChecker
from src.system.service_registry import ServiceRegistry, ServiceType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_system_integration():
    """Test the complete system integration."""
    
    logger.info("Starting system integration test...")
    
    # Test 1: System Integration Manager
    logger.info("Testing SystemIntegrationManager...")
    manager = SystemIntegrationManager()
    
    # Register a test service
    def test_startup():
        logger.info("Test service starting up")
        return True
    
    def test_health():
        return True
    
    manager.register_service(
        name="test_service",
        startup_func=test_startup,
        health_check=test_health
    )
    
    # Start the service
    success = await manager.start_service("test_service")
    assert success, "Failed to start test service"
    logger.info("✓ Service startup successful")
    
    # Check system status
    status = manager.get_system_status()
    assert status["overall_status"] in ["healthy", "warning"], f"System status: {status['overall_status']}"
    logger.info("✓ System status check successful")
    
    # Test 2: Error Handler
    logger.info("Testing ErrorHandler...")
    error_handler = ErrorHandler()
    
    test_exception = Exception("Test error for integration")
    error_context = error_handler.handle_error(
        exception=test_exception,
        category=ErrorCategory.SYSTEM,
        severity=ErrorSeverity.LOW,
        service_name="test_service"
    )
    
    assert error_context.category == ErrorCategory.SYSTEM
    assert "Test error for integration" in error_context.message
    logger.info("✓ Error handling successful")
    
    # Test 3: Metrics Collector
    logger.info("Testing MetricsCollector...")
    metrics = MetricsCollector()
    
    metrics.record_metric("test_metric", 42.0)
    metrics.increment_counter("test_counter", 1.0)
    
    summary = metrics.get_metric_summary("test_metric")
    assert summary["latest"] == 42.0
    logger.info("✓ Metrics collection successful")
    
    # Test 4: Health Checker
    logger.info("Testing HealthChecker...")
    health_checker = HealthChecker()
    
    def test_health_check():
        return {"status": "healthy", "message": "Test check OK"}
    
    health_checker.register_check("test_check", test_health_check)
    
    health_result = await health_checker.run_check("test_check")
    assert health_result.message == "Test check OK"
    logger.info("✓ Health checking successful")
    
    # Test 5: Service Registry
    logger.info("Testing ServiceRegistry...")
    registry = ServiceRegistry()
    
    registry.register_service(
        name="test_registry_service",
        service_type=ServiceType.FEATURE,
        version="1.0.0"
    )
    
    service = registry.get_service("test_registry_service")
    assert service is not None
    assert service.name == "test_registry_service"
    logger.info("✓ Service registry successful")
    
    # Cleanup
    await manager.stop_service("test_service")
    logger.info("✓ Service cleanup successful")
    
    logger.info("All system integration tests passed! ✓")
    return True


async def main():
    """Main test function."""
    try:
        success = await test_system_integration()
        if success:
            logger.info("🎉 System integration test completed successfully!")
            return 0
        else:
            logger.error("❌ System integration test failed!")
            return 1
    except Exception as e:
        logger.error(f"❌ System integration test failed with exception: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)