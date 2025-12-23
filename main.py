"""
SuperInsight Platform Main Application Entry Point
"""
import logging
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Initialize logging first
from src.system.logging_config import setup_logging, get_logger

# Setup logging before importing other modules
setup_logging()

from src.config.settings import settings
from src.system.integration import system_manager
from src.system.service_registry import service_registry, ServiceType

logger = get_logger(__name__, service_name="main")


async def initialize_system():
    """Initialize the complete system."""
    logger.info(f"Initializing {settings.app.app_name} v{settings.app.app_version}")
    
    try:
        # Register core services in the service registry
        from src.system.service_registry import register_core_service
        
        register_core_service(
            name="database",
            description="PostgreSQL database service",
            version="1.0.0"
        )
        
        register_core_service(
            name="metrics",
            description="Metrics collection service",
            dependencies=["database"],
            version="1.0.0"
        )
        
        register_core_service(
            name="health_monitor",
            description="Health monitoring service",
            dependencies=["metrics"],
            version="1.0.0"
        )
        
        # Register feature services
        service_registry.register_service(
            name="extraction",
            service_type=ServiceType.FEATURE,
            version="1.0.0",
            dependencies=["database"],
            description="Data extraction service"
        )
        
        service_registry.register_service(
            name="quality",
            service_type=ServiceType.FEATURE,
            version="1.0.0",
            dependencies=["database"],
            description="Quality management service"
        )
        
        service_registry.register_service(
            name="ai_annotation",
            service_type=ServiceType.FEATURE,
            version="1.0.0",
            dependencies=["database"],
            description="AI annotation service"
        )
        
        service_registry.register_service(
            name="billing",
            service_type=ServiceType.FEATURE,
            version="1.0.0",
            dependencies=["database"],
            description="Billing and analytics service"
        )
        
        service_registry.register_service(
            name="security",
            service_type=ServiceType.SECURITY,
            version="1.0.0",
            dependencies=["database"],
            description="Security and access control service"
        )
        
        # Validate service dependencies
        missing_deps = service_registry.validate_dependencies()
        if missing_deps:
            logger.warning(f"Missing service dependencies: {missing_deps}")
        
        # Get startup order
        startup_order = service_registry.get_startup_order()
        logger.info(f"Service startup order: {startup_order}")
        
        logger.info("System initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        return False


def main():
    """Main application entry point"""
    import asyncio
    
    async def async_main():
        success = await initialize_system()
        if not success:
            return False
        
        logger.info("SuperInsight Platform initialized successfully")
        logger.info("Use 'python -m uvicorn src.app:app --reload' to start the web server")
        return True
    
    # Run async initialization
    success = asyncio.run(async_main())
    return success


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)