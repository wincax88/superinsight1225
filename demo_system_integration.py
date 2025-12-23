#!/usr/bin/env python3
"""
SuperInsight Platform System Integration Demo

This script demonstrates the comprehensive system integration capabilities
including service management, error handling, monitoring, and health checks.
"""

import sys
import asyncio
import logging
import time

# Add src to path
sys.path.insert(0, 'src')

from src.system.integration import system_manager
from src.system.error_handler import error_handler, ErrorCategory, ErrorSeverity
from src.system.monitoring import metrics_collector, performance_monitor
from src.system.health import health_checker
from src.system.service_registry import service_registry, ServiceType
from src.system.logging_config import get_logger

# Configure logging
logger = get_logger(__name__, service_name="demo")


async def demo_system_integration():
    """Demonstrate system integration capabilities."""
    
    print("🚀 SuperInsight Platform System Integration Demo")
    print("=" * 60)
    
    # 1. Service Registration and Management
    print("\n📋 1. Service Registration and Management")
    print("-" * 40)
    
    # Register demo services
    def demo_service_startup():
        logger.info("Demo service starting up...")
        time.sleep(0.1)  # Simulate startup time
        return True
    
    def demo_service_health():
        return True
    
    def demo_service_shutdown():
        logger.info("Demo service shutting down...")
    
    system_manager.register_service(
        name="demo_service",
        startup_func=demo_service_startup,
        shutdown_func=demo_service_shutdown,
        health_check=demo_service_health,
        description="Demo service for integration testing"
    )
    
    # Register in service registry
    service_registry.register_service(
        name="demo_service",
        service_type=ServiceType.FEATURE,
        version="1.0.0",
        description="Demo service for system integration"
    )
    
    print(f"✓ Registered demo service")
    
    # Start services
    print("\n🔄 Starting services...")
    success = await system_manager.start_service("demo_service")
    print(f"✓ Service startup: {'Success' if success else 'Failed'}")
    
    # Check system status
    status = system_manager.get_system_status()
    print(f"✓ System status: {status['overall_status']}")
    print(f"✓ Services running: {status['healthy_services']}/{status['total_services']}")
    
    # 2. Error Handling Demo
    print("\n🚨 2. Error Handling and Recovery")
    print("-" * 40)
    
    # Simulate different types of errors
    errors_to_demo = [
        (Exception("Database connection timeout"), ErrorCategory.DATABASE, ErrorSeverity.HIGH),
        (ValueError("Invalid input format"), ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM),
        (ConnectionError("External API unavailable"), ErrorCategory.EXTERNAL_API, ErrorSeverity.LOW)
    ]
    
    for exc, category, severity in errors_to_demo:
        error_context = error_handler.handle_error(
            exception=exc,
            category=category,
            severity=severity,
            service_name="demo_service"
        )
        print(f"✓ Handled {category.value} error: {error_context.error_id}")
    
    # Show error statistics
    stats = error_handler.get_error_statistics()
    print(f"✓ Total errors handled: {stats['total_errors']}")
    print(f"✓ Error categories: {list(stats['by_category'].keys())}")
    
    # 3. Metrics Collection Demo
    print("\n📊 3. Metrics Collection and Monitoring")
    print("-" * 40)
    
    # Record various metrics
    metrics_collector.record_metric("demo.requests", 100.0)
    metrics_collector.record_metric("demo.requests", 150.0)
    metrics_collector.record_metric("demo.requests", 120.0)
    
    metrics_collector.increment_counter("demo.api_calls", 5.0)
    metrics_collector.increment_counter("demo.api_calls", 3.0)
    
    metrics_collector.record_timing("demo.response_time", 0.25)
    metrics_collector.record_timing("demo.response_time", 0.18)
    
    # Show metrics summary
    request_summary = metrics_collector.get_metric_summary("demo.requests")
    print(f"✓ Request metrics - Count: {request_summary['count']}, Avg: {request_summary['avg']:.1f}")
    
    counter_summary = metrics_collector.get_metric_summary("demo.api_calls")
    print(f"✓ API calls counter: {counter_summary['latest']}")
    
    timing_summary = metrics_collector.get_metric_summary("demo.response_time.duration")
    print(f"✓ Response time - Avg: {timing_summary['avg']:.3f}s")
    
    # 4. Performance Monitoring Demo
    print("\n⚡ 4. Performance Monitoring")
    print("-" * 40)
    
    # Simulate request tracking
    performance_monitor.start_request("demo_req_1", "/api/demo")
    await asyncio.sleep(0.1)  # Simulate processing time
    performance_monitor.end_request("demo_req_1", "/api/demo", 200)
    
    # Simulate database query tracking
    performance_monitor.record_database_query("SELECT", 0.05, True)
    performance_monitor.record_database_query("INSERT", 0.12, True)
    
    # Simulate AI inference tracking
    performance_monitor.record_ai_inference("demo_model", 0.8, True)
    
    perf_summary = performance_monitor.get_performance_summary()
    print(f"✓ Active requests: {perf_summary['active_requests']}")
    print(f"✓ Database queries tracked: {len(perf_summary.get('request_counts', {}))}")
    
    # 5. Health Checking Demo
    print("\n🏥 5. Health Checking System")
    print("-" * 40)
    
    # Register demo health checks
    def demo_healthy_check():
        return {"status": "healthy", "message": "Demo service is running well"}
    
    def demo_warning_check():
        return {"status": "warning", "message": "Demo service has minor issues"}
    
    health_checker.register_check("demo_healthy", demo_healthy_check)
    health_checker.register_check("demo_warning", demo_warning_check)
    
    # Run health checks
    health_result = await health_checker.run_check("demo_healthy")
    print(f"✓ Health check result: {health_result.status.value} - {health_result.message}")
    
    # Get system health
    system_health = await health_checker.get_system_health()
    print(f"✓ Overall system health: {system_health['overall_status']}")
    print(f"✓ Health checks: {system_health['summary']['total_checks']} total")
    
    # 6. Service Registry Demo
    print("\n📚 6. Service Registry and Discovery")
    print("-" * 40)
    
    # Show registered services
    all_services = service_registry.get_all_services()
    print(f"✓ Total registered services: {len(all_services)}")
    
    for name, service in all_services.items():
        print(f"  - {name}: {service.service_type.value} v{service.version}")
    
    # Show service dependencies
    startup_order = service_registry.get_startup_order()
    print(f"✓ Service startup order: {startup_order}")
    
    # Validate dependencies
    missing_deps = service_registry.validate_dependencies()
    if missing_deps:
        print(f"⚠️  Missing dependencies: {missing_deps}")
    else:
        print("✓ All service dependencies satisfied")
    
    # 7. System Status Summary
    print("\n📈 7. Complete System Status")
    print("-" * 40)
    
    # Get comprehensive status
    system_status = system_manager.get_system_status()
    all_metrics = metrics_collector.get_all_metrics_summary()
    error_stats = error_handler.get_error_statistics()
    
    print(f"✓ System Status: {system_status['overall_status']}")
    print(f"✓ Services: {system_status['healthy_services']}/{system_status['total_services']} healthy")
    print(f"✓ Metrics tracked: {len(all_metrics)} different metrics")
    print(f"✓ Errors handled: {error_stats['total_errors']} total")
    
    # Cleanup
    print("\n🧹 8. System Cleanup")
    print("-" * 40)
    
    await system_manager.stop_service("demo_service")
    print("✓ Demo service stopped")
    
    print("\n🎉 System Integration Demo Completed Successfully!")
    print("=" * 60)
    
    return True


async def main():
    """Main demo function."""
    try:
        await demo_system_integration()
        return 0
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)