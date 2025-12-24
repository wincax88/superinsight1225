"""
Graceful Degradation System for SuperInsight Platform.

Provides mechanisms for graceful service degradation when components
fail or become unavailable, ensuring system resilience.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import threading
import time

logger = logging.getLogger(__name__)


class DegradationLevel(Enum):
    """Levels of service degradation."""
    FULL = "full"              # Full functionality
    REDUCED = "reduced"        # Reduced functionality
    MINIMAL = "minimal"        # Minimal functionality
    OFFLINE = "offline"        # Service offline


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior."""
    primary_service: str
    fallback_services: List[str] = field(default_factory=list)
    timeout: float = 30.0
    max_failures: int = 3
    recovery_check_interval: float = 60.0
    enable_caching: bool = True
    cache_ttl: float = 300.0


@dataclass
class ServiceHealth:
    """Health status of a service."""
    name: str
    is_healthy: bool = True
    failure_count: int = 0
    last_failure_time: float = 0
    last_success_time: float = 0
    degradation_level: DegradationLevel = DegradationLevel.FULL


class GracefulDegradationManager:
    """
    Manages graceful degradation of services.
    
    Provides fallback mechanisms, service health tracking,
    and automatic recovery detection.
    """
    
    def __init__(self):
        self.services: Dict[str, ServiceHealth] = {}
        self.fallback_configs: Dict[str, FallbackConfig] = {}
        self.fallback_handlers: Dict[str, Dict[str, Callable]] = {}
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        
    def register_service(self, name: str, degradation_level: DegradationLevel = DegradationLevel.FULL):
        """Register a service for degradation management."""
        with self.lock:
            self.services[name] = ServiceHealth(
                name=name,
                degradation_level=degradation_level
            )
            self.fallback_handlers[name] = {}
            logger.info(f"Registered service for degradation management: {name}")
    
    def configure_fallback(self, config: FallbackConfig):
        """Configure fallback behavior for a service."""
        with self.lock:
            self.fallback_configs[config.primary_service] = config
            
            # Ensure all services are registered
            if config.primary_service not in self.services:
                self.register_service(config.primary_service)
            
            for fallback_service in config.fallback_services:
                if fallback_service not in self.services:
                    self.register_service(fallback_service)
            
            logger.info(f"Configured fallback for {config.primary_service}: {config.fallback_services}")
    
    def register_fallback_handler(
        self,
        service_name: str,
        degradation_level: DegradationLevel,
        handler: Callable
    ):
        """Register a fallback handler for a specific degradation level."""
        with self.lock:
            if service_name not in self.fallback_handlers:
                self.fallback_handlers[service_name] = {}
            
            self.fallback_handlers[service_name][degradation_level.value] = handler
            logger.info(f"Registered fallback handler for {service_name} at {degradation_level.value} level")
    
    def mark_service_failure(self, service_name: str, exception: Optional[Exception] = None):
        """Mark a service as failed."""
        with self.lock:
            if service_name not in self.services:
                self.register_service(service_name)
            
            service = self.services[service_name]
            service.failure_count += 1
            service.last_failure_time = time.time()
            service.is_healthy = False
            
            # Determine degradation level based on failure count
            config = self.fallback_configs.get(service_name)
            if config and service.failure_count >= config.max_failures:
                if service.degradation_level == DegradationLevel.FULL:
                    service.degradation_level = DegradationLevel.REDUCED
                elif service.degradation_level == DegradationLevel.REDUCED:
                    service.degradation_level = DegradationLevel.MINIMAL
                elif service.degradation_level == DegradationLevel.MINIMAL:
                    service.degradation_level = DegradationLevel.OFFLINE
            
            logger.warning(
                f"Service {service_name} marked as failed "
                f"(failures: {service.failure_count}, level: {service.degradation_level.value})"
            )
    
    def mark_service_success(self, service_name: str):
        """Mark a service as successful."""
        with self.lock:
            if service_name not in self.services:
                self.register_service(service_name)
            
            service = self.services[service_name]
            service.last_success_time = time.time()
            service.is_healthy = True
            service.failure_count = 0
            service.degradation_level = DegradationLevel.FULL
            
            logger.info(f"Service {service_name} marked as healthy")
    
    def get_service_health(self, service_name: str) -> Optional[ServiceHealth]:
        """Get health status of a service."""
        with self.lock:
            return self.services.get(service_name)
    
    def get_all_service_health(self) -> Dict[str, ServiceHealth]:
        """Get health status of all services."""
        with self.lock:
            return self.services.copy()
    
    def execute_with_fallback(self, service_name: str, primary_func: Callable, *args, **kwargs) -> Any:
        """Execute function with fallback support."""
        service = self.get_service_health(service_name)
        
        if not service:
            # Service not registered, execute directly
            return primary_func(*args, **kwargs)
        
        # Try primary function first if service is healthy enough
        if service.degradation_level in [DegradationLevel.FULL, DegradationLevel.REDUCED]:
            try:
                result = primary_func(*args, **kwargs)
                self.mark_service_success(service_name)
                return result
            except Exception as e:
                logger.warning(f"Primary function failed for {service_name}: {str(e)}")
                self.mark_service_failure(service_name, e)
        
        # Try fallback handlers
        return self._execute_fallback(service_name, service.degradation_level, *args, **kwargs)
    
    async def async_execute_with_fallback(
        self,
        service_name: str,
        primary_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute async function with fallback support."""
        service = self.get_service_health(service_name)
        
        if not service:
            # Service not registered, execute directly
            if asyncio.iscoroutinefunction(primary_func):
                return await primary_func(*args, **kwargs)
            else:
                return primary_func(*args, **kwargs)
        
        # Try primary function first if service is healthy enough
        if service.degradation_level in [DegradationLevel.FULL, DegradationLevel.REDUCED]:
            try:
                if asyncio.iscoroutinefunction(primary_func):
                    result = await primary_func(*args, **kwargs)
                else:
                    result = primary_func(*args, **kwargs)
                
                self.mark_service_success(service_name)
                return result
            except Exception as e:
                logger.warning(f"Primary function failed for {service_name}: {str(e)}")
                self.mark_service_failure(service_name, e)
        
        # Try fallback handlers
        return await self._async_execute_fallback(service_name, service.degradation_level, *args, **kwargs)
    
    def _execute_fallback(self, service_name: str, degradation_level: DegradationLevel, *args, **kwargs) -> Any:
        """Execute fallback handler for the given degradation level."""
        handlers = self.fallback_handlers.get(service_name, {})
        
        # Try handlers in order of degradation severity
        levels_to_try = [degradation_level]
        
        if degradation_level == DegradationLevel.REDUCED:
            levels_to_try.append(DegradationLevel.MINIMAL)
        elif degradation_level == DegradationLevel.FULL:
            levels_to_try.extend([DegradationLevel.REDUCED, DegradationLevel.MINIMAL])
        
        for level in levels_to_try:
            handler = handlers.get(level.value)
            if handler:
                try:
                    logger.info(f"Executing {level.value} fallback for {service_name}")
                    return handler(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Fallback handler {level.value} failed for {service_name}: {str(e)}")
                    continue
        
        # No fallback handlers available
        logger.error(f"No fallback handlers available for {service_name}")
        raise RuntimeError(f"Service {service_name} is unavailable and no fallbacks are configured")
    
    async def _async_execute_fallback(
        self,
        service_name: str,
        degradation_level: DegradationLevel,
        *args,
        **kwargs
    ) -> Any:
        """Execute async fallback handler for the given degradation level."""
        handlers = self.fallback_handlers.get(service_name, {})
        
        # Try handlers in order of degradation severity
        levels_to_try = [degradation_level]
        
        if degradation_level == DegradationLevel.REDUCED:
            levels_to_try.append(DegradationLevel.MINIMAL)
        elif degradation_level == DegradationLevel.FULL:
            levels_to_try.extend([DegradationLevel.REDUCED, DegradationLevel.MINIMAL])
        
        for level in levels_to_try:
            handler = handlers.get(level.value)
            if handler:
                try:
                    logger.info(f"Executing {level.value} fallback for {service_name}")
                    if asyncio.iscoroutinefunction(handler):
                        return await handler(*args, **kwargs)
                    else:
                        return handler(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Fallback handler {level.value} failed for {service_name}: {str(e)}")
                    continue
        
        # No fallback handlers available
        logger.error(f"No fallback handlers available for {service_name}")
        raise RuntimeError(f"Service {service_name} is unavailable and no fallbacks are configured")
    
    def cache_result(self, key: str, result: Any, ttl: float = 300.0):
        """Cache a result for fallback use."""
        with self.lock:
            if "cache" not in self.cache:
                self.cache["cache"] = {}
            
            self.cache["cache"][key] = {
                "result": result,
                "timestamp": time.time(),
                "ttl": ttl
            }
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """Get a cached result if still valid."""
        with self.lock:
            if "cache" not in self.cache:
                return None
            
            cached = self.cache["cache"].get(key)
            if not cached:
                return None
            
            # Check if cache is still valid
            if time.time() - cached["timestamp"] > cached["ttl"]:
                del self.cache["cache"][key]
                return None
            
            return cached["result"]
    
    def clear_cache(self, key: Optional[str] = None):
        """Clear cache entries."""
        with self.lock:
            if key:
                if "cache" in self.cache and key in self.cache["cache"]:
                    del self.cache["cache"][key]
            else:
                self.cache.clear()


# Global degradation manager
degradation_manager = GracefulDegradationManager()


# Decorators for easy usage
def with_fallback(
    service_name: str,
    fallback_services: Optional[List[str]] = None,
    timeout: float = 30.0,
    enable_caching: bool = False,
    cache_ttl: float = 300.0
):
    """Decorator for adding fallback support to functions."""
    def decorator(func):
        # Register service if not already registered
        if service_name not in degradation_manager.services:
            degradation_manager.register_service(service_name)
        
        # Configure fallback if specified
        if fallback_services:
            config = FallbackConfig(
                primary_service=service_name,
                fallback_services=fallback_services,
                timeout=timeout,
                enable_caching=enable_caching,
                cache_ttl=cache_ttl
            )
            degradation_manager.configure_fallback(config)
        
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Check cache first if enabled
                if enable_caching:
                    cache_key = f"{service_name}:{func.__name__}:{hash(str(args) + str(kwargs))}"
                    cached_result = degradation_manager.get_cached_result(cache_key)
                    if cached_result is not None:
                        logger.debug(f"Returning cached result for {service_name}")
                        return cached_result
                
                try:
                    result = await degradation_manager.async_execute_with_fallback(
                        service_name, func, *args, **kwargs
                    )
                    
                    # Cache successful result if enabled
                    if enable_caching:
                        degradation_manager.cache_result(cache_key, result, cache_ttl)
                    
                    return result
                except Exception as e:
                    # Try to return cached result as last resort
                    if enable_caching:
                        cached_result = degradation_manager.get_cached_result(cache_key)
                        if cached_result is not None:
                            logger.warning(f"Returning stale cached result for {service_name} due to error: {str(e)}")
                            return cached_result
                    raise
            
            return async_wrapper
        else:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Check cache first if enabled
                if enable_caching:
                    cache_key = f"{service_name}:{func.__name__}:{hash(str(args) + str(kwargs))}"
                    cached_result = degradation_manager.get_cached_result(cache_key)
                    if cached_result is not None:
                        logger.debug(f"Returning cached result for {service_name}")
                        return cached_result
                
                try:
                    result = degradation_manager.execute_with_fallback(
                        service_name, func, *args, **kwargs
                    )
                    
                    # Cache successful result if enabled
                    if enable_caching:
                        degradation_manager.cache_result(cache_key, result, cache_ttl)
                    
                    return result
                except Exception as e:
                    # Try to return cached result as last resort
                    if enable_caching:
                        cached_result = degradation_manager.get_cached_result(cache_key)
                        if cached_result is not None:
                            logger.warning(f"Returning stale cached result for {service_name} due to error: {str(e)}")
                            return cached_result
                    raise
            
            return wrapper
    
    return decorator


def fallback_handler(service_name: str, degradation_level: DegradationLevel):
    """Decorator for registering fallback handlers."""
    def decorator(func):
        degradation_manager.register_fallback_handler(service_name, degradation_level, func)
        return func
    
    return decorator


# Common fallback handlers
@fallback_handler("ai_annotation", DegradationLevel.REDUCED)
def ai_annotation_reduced_fallback(*args, **kwargs):
    """Reduced functionality fallback for AI annotation."""
    logger.info("Using reduced AI annotation fallback - returning empty predictions")
    return {
        "predictions": [],
        "confidence": 0.0,
        "fallback_used": True,
        "fallback_level": "reduced"
    }


@fallback_handler("ai_annotation", DegradationLevel.MINIMAL)
def ai_annotation_minimal_fallback(*args, **kwargs):
    """Minimal functionality fallback for AI annotation."""
    logger.info("Using minimal AI annotation fallback - returning basic structure")
    return {
        "predictions": [],
        "confidence": 0.0,
        "fallback_used": True,
        "fallback_level": "minimal",
        "message": "AI annotation service is temporarily unavailable"
    }


@fallback_handler("data_extraction", DegradationLevel.REDUCED)
def data_extraction_reduced_fallback(*args, **kwargs):
    """Reduced functionality fallback for data extraction."""
    logger.info("Using reduced data extraction fallback - basic extraction only")
    return {
        "documents": [],
        "success": False,
        "fallback_used": True,
        "fallback_level": "reduced",
        "message": "Using reduced extraction capabilities"
    }


@fallback_handler("quality_check", DegradationLevel.REDUCED)
def quality_check_reduced_fallback(*args, **kwargs):
    """Reduced functionality fallback for quality checks."""
    logger.info("Using reduced quality check fallback - basic validation only")
    return {
        "quality_score": 0.5,  # Neutral score
        "issues": [],
        "fallback_used": True,
        "fallback_level": "reduced",
        "message": "Using basic quality validation"
    }