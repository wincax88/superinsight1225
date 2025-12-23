#!/usr/bin/env python3
"""
Simple System Integration Demo for SuperInsight Platform
"""

import sys
import asyncio
import logging

# Add src to path
sys.path.insert(0, 'src')

from src.system.integration import SystemIntegrationManager
from src.system.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
from src.system.monitoring import MetricsCollector
from src.system.service_registry import ServiceRegistry, ServiceType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


async def simple_demo():
    """Simple system integration demonstration."""
    
    print("🚀 SuperInsight Platform - Simple Integration Demo")
    print("=" * 55)
    
    # 1. Service Management
    print("\n1️⃣ Service Management")
    manager = SystemIntegrationManager()
    
    def test_startup():
        print("   📦 Service starting...")
        return True
    
    def test_health():
        return True
    
    manager.register_service(
        name="test_service",
        startup_func=test_startup,
        health_check=test_health
    )
    
    success = await manager.start_service("test_service")
    print(f"   ✅ Service started: {success}")
    
    status = manager.get_system_status()
    print(f"   📊 System status: {status['overall_status']}")
    
    # 2. Error Handling
    print("\n2️⃣ Error Handling")
    error_handler = ErrorHandler()
    
    test_error = Exception("Demo error")
    error_context = error_handler.handle_error(
        exception=test_error,
        category=ErrorCategory.SYSTEM,
        severity=ErrorSeverity.LOW,
        service_name="test_service"
    )
    
    print(f"   🚨 Error handled: {error_context.error_id}")
    
    stats = error_handler.get_error_statistics()
    print(f"   📈 Total errors: {stats['total_errors']}")
    
    # 3. Metrics Collection
    print("\n3️⃣ Metrics Collection")
    metrics = MetricsCollector()
    
    metrics.record_metric("demo_metric", 42.0)
    metrics.increment_counter("demo_counter", 5.0)
    
    summary = metrics.get_metric_summary("demo_metric")
    print(f"   📊 Metric recorded: {summary['latest']}")
    
    counter_summary = metrics.get_metric_summary("demo_counter")
    print(f"   🔢 Counter value: {counter_summary['latest']}")
    
    # 4. Service Registry
    print("\n4️⃣ Service Registry")
    registry = ServiceRegistry()
    
    registry.register_service(
        name="demo_registry_service",
        service_type=ServiceType.FEATURE,
        version="1.0.0"
    )
    
    service = registry.get_service("demo_registry_service")
    print(f"   📋 Service registered: {service.name} v{service.version}")
    
    all_services = registry.get_all_services()
    print(f"   📚 Total services: {len(all_services)}")
    
    # 5. Cleanup
    print("\n5️⃣ Cleanup")
    await manager.stop_service("test_service")
    print("   🧹 Service stopped")
    
    print("\n🎉 Demo completed successfully!")
    print("=" * 55)
    
    return True


async def main():
    """Main function."""
    try:
        await simple_demo()
        return 0
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)