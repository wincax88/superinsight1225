"""
System Integration Manager for SuperInsight Platform.

Manages the integration of all system components including:
- Service discovery and registration
- Inter-service communication
- Unified error handling
- System monitoring and health checks
- Graceful startup and shutdown
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

from src.config.settings import settings


logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status enumeration."""
    STARTING = "starting"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class ServiceInfo:
    """Service information container."""
    name: str
    status: ServiceStatus = ServiceStatus.STOPPED
    health_check: Optional[Callable] = None
    startup_func: Optional[Callable] = None
    shutdown_func: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)
    last_health_check: Optional[float] = None
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SystemIntegrationManager:
    """
    Central system integration manager that coordinates all platform services.
    
    Responsibilities:
    - Service lifecycle management
    - Health monitoring
    - Error handling and recovery
    - Inter-service communication
    - System metrics collection
    """
    
    def __init__(self):
        self.services: Dict[str, ServiceInfo] = {}
        self.is_running = False
        self.startup_order: List[str] = []
        self.shutdown_order: List[str] = []
        self._health_check_interval = 30  # seconds
        self._health_check_task: Optional[asyncio.Task] = None
        
    def register_service(
        self,
        name: str,
        health_check: Optional[Callable] = None,
        startup_func: Optional[Callable] = None,
        shutdown_func: Optional[Callable] = None,
        dependencies: Optional[List[str]] = None,
        **metadata
    ) -> None:
        """Register a service with the integration manager."""
        if name in self.services:
            logger.warning(f"Service {name} is already registered, updating...")
        
        self.services[name] = ServiceInfo(
            name=name,
            health_check=health_check,
            startup_func=startup_func,
            shutdown_func=shutdown_func,
            dependencies=dependencies or [],
            metadata=metadata
        )
        
        logger.info(f"Registered service: {name}")
    
    def _calculate_startup_order(self) -> List[str]:
        """Calculate the correct startup order based on dependencies."""
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(service_name: str):
            if service_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {service_name}")
            if service_name in visited:
                return
            
            temp_visited.add(service_name)
            
            service = self.services.get(service_name)
            if service:
                for dep in service.dependencies:
                    if dep in self.services:
                        visit(dep)
                    else:
                        logger.warning(f"Dependency {dep} for service {service_name} not found")
            
            temp_visited.remove(service_name)
            visited.add(service_name)
            order.append(service_name)
        
        for service_name in self.services:
            visit(service_name)
        
        return order
    
    async def start_service(self, name: str) -> bool:
        """Start a specific service."""
        if name not in self.services:
            logger.error(f"Service {name} not found")
            return False
        
        service = self.services[name]
        
        if service.status == ServiceStatus.HEALTHY:
            logger.info(f"Service {name} is already running")
            return True
        
        logger.info(f"Starting service: {name}")
        service.status = ServiceStatus.STARTING
        
        try:
            # Check dependencies first
            for dep in service.dependencies:
                if dep in self.services:
                    dep_service = self.services[dep]
                    if dep_service.status != ServiceStatus.HEALTHY:
                        logger.error(f"Dependency {dep} for service {name} is not healthy")
                        service.status = ServiceStatus.UNHEALTHY
                        return False
            
            # Start the service
            if service.startup_func:
                if asyncio.iscoroutinefunction(service.startup_func):
                    await service.startup_func()
                else:
                    service.startup_func()
            
            # Perform initial health check
            if await self._check_service_health(name):
                service.status = ServiceStatus.HEALTHY
                logger.info(f"Service {name} started successfully")
                return True
            else:
                service.status = ServiceStatus.UNHEALTHY
                logger.error(f"Service {name} failed health check after startup")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start service {name}: {e}")
            service.status = ServiceStatus.UNHEALTHY
            service.error_count += 1
            return False
    
    async def stop_service(self, name: str) -> bool:
        """Stop a specific service."""
        if name not in self.services:
            logger.error(f"Service {name} not found")
            return False
        
        service = self.services[name]
        
        if service.status == ServiceStatus.STOPPED:
            logger.info(f"Service {name} is already stopped")
            return True
        
        logger.info(f"Stopping service: {name}")
        service.status = ServiceStatus.STOPPING
        
        try:
            if service.shutdown_func:
                if asyncio.iscoroutinefunction(service.shutdown_func):
                    await service.shutdown_func()
                else:
                    service.shutdown_func()
            
            service.status = ServiceStatus.STOPPED
            logger.info(f"Service {name} stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop service {name}: {e}")
            service.status = ServiceStatus.UNHEALTHY
            return False
    
    async def _check_service_health(self, name: str) -> bool:
        """Check the health of a specific service."""
        if name not in self.services:
            return False
        
        service = self.services[name]
        
        if not service.health_check:
            # If no health check is defined, assume healthy if not stopped
            return service.status != ServiceStatus.STOPPED
        
        try:
            if asyncio.iscoroutinefunction(service.health_check):
                result = await service.health_check()
            else:
                result = service.health_check()
            
            service.last_health_check = time.time()
            
            if result:
                if service.status == ServiceStatus.UNHEALTHY:
                    logger.info(f"Service {name} recovered")
                    service.error_count = 0
                return True
            else:
                if service.status == ServiceStatus.HEALTHY:
                    logger.warning(f"Service {name} became unhealthy")
                service.error_count += 1
                return False
                
        except Exception as e:
            logger.error(f"Health check failed for service {name}: {e}")
            service.error_count += 1
            return False
    
    async def _health_check_loop(self):
        """Continuous health check loop for all services."""
        while self.is_running:
            try:
                for name, service in self.services.items():
                    if service.status in [ServiceStatus.HEALTHY, ServiceStatus.UNHEALTHY]:
                        is_healthy = await self._check_service_health(name)
                        
                        if is_healthy and service.status != ServiceStatus.HEALTHY:
                            service.status = ServiceStatus.HEALTHY
                        elif not is_healthy and service.status != ServiceStatus.UNHEALTHY:
                            service.status = ServiceStatus.UNHEALTHY
                
                await asyncio.sleep(self._health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5)  # Short delay before retrying
    
    async def start_all_services(self) -> bool:
        """Start all registered services in dependency order."""
        logger.info("Starting all services...")
        
        try:
            self.startup_order = self._calculate_startup_order()
            logger.info(f"Service startup order: {self.startup_order}")
            
            for service_name in self.startup_order:
                success = await self.start_service(service_name)
                if not success:
                    logger.error(f"Failed to start service {service_name}, aborting startup")
                    return False
            
            self.is_running = True
            
            # Start health check loop
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            logger.info("All services started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start services: {e}")
            return False
    
    async def stop_all_services(self) -> bool:
        """Stop all services in reverse dependency order."""
        logger.info("Stopping all services...")
        
        self.is_running = False
        
        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Stop services in reverse order
        self.shutdown_order = list(reversed(self.startup_order))
        
        success = True
        for service_name in self.shutdown_order:
            if not await self.stop_service(service_name):
                success = False
        
        if success:
            logger.info("All services stopped successfully")
        else:
            logger.warning("Some services failed to stop cleanly")
        
        return success
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        service_statuses = {}
        healthy_count = 0
        total_count = len(self.services)
        
        for name, service in self.services.items():
            service_statuses[name] = {
                "status": service.status.value,
                "last_health_check": service.last_health_check,
                "error_count": service.error_count,
                "dependencies": service.dependencies,
                "metadata": service.metadata
            }
            
            if service.status == ServiceStatus.HEALTHY:
                healthy_count += 1
        
        overall_status = "healthy" if healthy_count == total_count and total_count > 0 else "unhealthy"
        
        return {
            "overall_status": overall_status,
            "healthy_services": healthy_count,
            "total_services": total_count,
            "services": service_statuses,
            "startup_order": self.startup_order,
            "is_running": self.is_running
        }
    
    def get_service_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific service."""
        if name not in self.services:
            return None
        
        service = self.services[name]
        return {
            "name": name,
            "status": service.status.value,
            "last_health_check": service.last_health_check,
            "error_count": service.error_count,
            "dependencies": service.dependencies,
            "metadata": service.metadata
        }


# Global system integration manager instance
system_manager = SystemIntegrationManager()


@asynccontextmanager
async def system_lifespan():
    """Context manager for system lifecycle management."""
    try:
        # Startup
        success = await system_manager.start_all_services()
        if not success:
            raise RuntimeError("Failed to start all services")
        
        yield system_manager
        
    finally:
        # Shutdown
        await system_manager.stop_all_services()