"""
Service Registry for SuperInsight Platform.

Provides service discovery and inter-service communication capabilities.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ServiceType(Enum):
    """Service type enumeration."""
    CORE = "core"
    FEATURE = "feature"
    MONITORING = "monitoring"
    SECURITY = "security"
    EXTERNAL = "external"


@dataclass
class ServiceEndpoint:
    """Service endpoint information."""
    name: str
    url: str
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    retry_count: int = 3


@dataclass
class RegisteredService:
    """Registered service information."""
    name: str
    service_type: ServiceType
    version: str
    endpoints: Dict[str, ServiceEndpoint] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    health_check_url: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


class ServiceRegistry:
    """
    Service registry for managing service discovery and communication.
    
    Provides:
    - Service registration and discovery
    - Endpoint management
    - Service dependency tracking
    - Health status monitoring
    """
    
    def __init__(self):
        self.services: Dict[str, RegisteredService] = {}
        self.event_handlers: Dict[str, List[Callable]] = {
            "service_registered": [],
            "service_unregistered": [],
            "service_health_changed": []
        }
    
    def register_service(
        self,
        name: str,
        service_type: ServiceType,
        version: str = "1.0.0",
        endpoints: Optional[Dict[str, ServiceEndpoint]] = None,
        health_check_url: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        **metadata
    ) -> None:
        """Register a service in the registry."""
        
        service = RegisteredService(
            name=name,
            service_type=service_type,
            version=version,
            endpoints=endpoints or {},
            metadata=metadata,
            health_check_url=health_check_url,
            dependencies=dependencies or []
        )
        
        self.services[name] = service
        
        logger.info(f"Registered service: {name} (type: {service_type.value}, version: {version})")
        
        # Trigger event handlers
        self._trigger_event("service_registered", service)
    
    def unregister_service(self, name: str) -> bool:
        """Unregister a service from the registry."""
        if name in self.services:
            service = self.services.pop(name)
            logger.info(f"Unregistered service: {name}")
            
            # Trigger event handlers
            self._trigger_event("service_unregistered", service)
            return True
        
        return False
    
    def get_service(self, name: str) -> Optional[RegisteredService]:
        """Get service information by name."""
        return self.services.get(name)
    
    def get_services_by_type(self, service_type: ServiceType) -> List[RegisteredService]:
        """Get all services of a specific type."""
        return [
            service for service in self.services.values()
            if service.service_type == service_type
        ]
    
    def get_all_services(self) -> Dict[str, RegisteredService]:
        """Get all registered services."""
        return self.services.copy()
    
    def add_endpoint(self, service_name: str, endpoint_name: str, endpoint: ServiceEndpoint) -> bool:
        """Add an endpoint to a registered service."""
        if service_name in self.services:
            self.services[service_name].endpoints[endpoint_name] = endpoint
            logger.info(f"Added endpoint {endpoint_name} to service {service_name}")
            return True
        
        logger.warning(f"Cannot add endpoint to unregistered service: {service_name}")
        return False
    
    def get_endpoint(self, service_name: str, endpoint_name: str) -> Optional[ServiceEndpoint]:
        """Get a specific endpoint from a service."""
        service = self.get_service(service_name)
        if service:
            return service.endpoints.get(endpoint_name)
        return None
    
    def get_service_dependencies(self, service_name: str) -> List[str]:
        """Get dependencies for a service."""
        service = self.get_service(service_name)
        if service:
            return service.dependencies.copy()
        return []
    
    def get_dependent_services(self, service_name: str) -> List[str]:
        """Get services that depend on the given service."""
        dependents = []
        for name, service in self.services.items():
            if service_name in service.dependencies:
                dependents.append(name)
        return dependents
    
    def validate_dependencies(self) -> Dict[str, List[str]]:
        """Validate service dependencies and return missing dependencies."""
        missing_deps = {}
        
        for service_name, service in self.services.items():
            missing = []
            for dep in service.dependencies:
                if dep not in self.services:
                    missing.append(dep)
            
            if missing:
                missing_deps[service_name] = missing
        
        return missing_deps
    
    def get_startup_order(self) -> List[str]:
        """Calculate service startup order based on dependencies."""
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
            
            temp_visited.remove(service_name)
            visited.add(service_name)
            order.append(service_name)
        
        for service_name in self.services:
            visit(service_name)
        
        return order
    
    def register_event_handler(self, event_type: str, handler: Callable) -> None:
        """Register an event handler."""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
            logger.info(f"Registered event handler for: {event_type}")
        else:
            logger.warning(f"Unknown event type: {event_type}")
    
    def _trigger_event(self, event_type: str, service: RegisteredService) -> None:
        """Trigger event handlers for a specific event."""
        for handler in self.event_handlers.get(event_type, []):
            try:
                handler(service)
            except Exception as e:
                logger.error(f"Event handler failed for {event_type}: {e}")
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get registry status and statistics."""
        service_counts = {}
        for service in self.services.values():
            service_type = service.service_type.value
            service_counts[service_type] = service_counts.get(service_type, 0) + 1
        
        missing_deps = self.validate_dependencies()
        
        return {
            "total_services": len(self.services),
            "services_by_type": service_counts,
            "missing_dependencies": missing_deps,
            "has_circular_dependencies": self._check_circular_dependencies(),
            "services": {
                name: {
                    "type": service.service_type.value,
                    "version": service.version,
                    "endpoints": list(service.endpoints.keys()),
                    "dependencies": service.dependencies,
                    "metadata": service.metadata
                }
                for name, service in self.services.items()
            }
        }
    
    def _check_circular_dependencies(self) -> bool:
        """Check if there are circular dependencies."""
        try:
            self.get_startup_order()
            return False
        except ValueError:
            return True


# Global service registry instance
service_registry = ServiceRegistry()


# Helper functions for common service operations
def register_api_service(
    name: str,
    base_url: str,
    endpoints: Dict[str, str],
    version: str = "1.0.0",
    dependencies: Optional[List[str]] = None
) -> None:
    """Helper function to register an API service."""
    
    service_endpoints = {}
    for endpoint_name, path in endpoints.items():
        service_endpoints[endpoint_name] = ServiceEndpoint(
            name=endpoint_name,
            url=f"{base_url}{path}"
        )
    
    service_registry.register_service(
        name=name,
        service_type=ServiceType.FEATURE,
        version=version,
        endpoints=service_endpoints,
        health_check_url=f"{base_url}/health",
        dependencies=dependencies or []
    )


def register_core_service(
    name: str,
    health_check_func: Optional[Callable] = None,
    dependencies: Optional[List[str]] = None,
    **metadata
) -> None:
    """Helper function to register a core service."""
    
    service_registry.register_service(
        name=name,
        service_type=ServiceType.CORE,
        dependencies=dependencies or [],
        **metadata
    )


def get_service_endpoint_url(service_name: str, endpoint_name: str) -> Optional[str]:
    """Get the URL for a specific service endpoint."""
    endpoint = service_registry.get_endpoint(service_name, endpoint_name)
    if endpoint:
        return endpoint.url
    return None


def is_service_available(service_name: str) -> bool:
    """Check if a service is registered and available."""
    return service_name in service_registry.services