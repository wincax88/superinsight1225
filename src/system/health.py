"""
Comprehensive Health Check System for SuperInsight Platform.

Provides detailed health checks for all system components and dependencies.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from src.config.settings import settings


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Health check result container."""
    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    timestamp: float
    details: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "details": self.details or {}
        }


class HealthChecker:
    """
    Comprehensive health checking system.

    Performs health checks on all system components and provides
    detailed status information.
    """

    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self._load_config()

    def _load_config(self):
        """Load health check configuration from settings."""
        try:
            self.check_timeout = settings.health_check.health_check_timeout
            self.retry_attempts = settings.health_check.health_check_retry_attempts
            self.retry_delay = settings.health_check.health_check_retry_delay
            self.enabled = settings.health_check.health_check_enabled

            # Individual check toggles
            self.check_toggles = {
                "database": settings.health_check.database_check_enabled,
                "label_studio": settings.health_check.label_studio_check_enabled,
                "ai_services": settings.health_check.ai_services_check_enabled,
                "storage": settings.health_check.storage_check_enabled,
                "security": settings.health_check.security_check_enabled,
                "external_dependencies": settings.health_check.external_deps_check_enabled,
            }
        except AttributeError:
            # Fallback to defaults if settings not available
            self.check_timeout = 30
            self.retry_attempts = 3
            self.retry_delay = 1.0
            self.enabled = True
            self.check_toggles = {}

    def is_check_enabled(self, name: str) -> bool:
        """Check if a specific health check is enabled."""
        if not self.enabled:
            return False
        return self.check_toggles.get(name, True)
        
    def register_check(self, name: str, check_func: Callable):
        """Register a health check function."""
        self.checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    async def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check with retry support."""
        if name not in self.checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check '{name}' not found",
                duration_ms=0,
                timestamp=time.time()
            )

        # Check if this health check is enabled
        if not self.is_check_enabled(name):
            return HealthCheckResult(
                name=name,
                status=HealthStatus.HEALTHY,
                message=f"Health check '{name}' is disabled",
                duration_ms=0,
                timestamp=time.time(),
                details={"disabled": True}
            )

        start_time = time.time()
        last_error = None

        # Retry logic
        for attempt in range(self.retry_attempts):
            try:
                check_func = self.checks[name]

                # Run with timeout
                if asyncio.iscoroutinefunction(check_func):
                    result = await asyncio.wait_for(check_func(), timeout=self.check_timeout)
                else:
                    result = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, check_func),
                        timeout=self.check_timeout
                    )

                duration_ms = (time.time() - start_time) * 1000

                # Parse result
                if isinstance(result, dict):
                    return HealthCheckResult(
                        name=name,
                        status=HealthStatus(result.get("status", "unknown")),
                        message=result.get("message", "OK"),
                        duration_ms=duration_ms,
                        timestamp=time.time(),
                        details=result.get("details")
                    )
                elif isinstance(result, bool):
                    return HealthCheckResult(
                        name=name,
                        status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                        message="OK" if result else "Check failed",
                        duration_ms=duration_ms,
                        timestamp=time.time()
                    )
                else:
                    return HealthCheckResult(
                        name=name,
                        status=HealthStatus.HEALTHY,
                        message=str(result),
                        duration_ms=duration_ms,
                        timestamp=time.time()
                    )

            except asyncio.TimeoutError:
                last_error = f"Health check timed out after {self.check_timeout}s"
            except Exception as e:
                last_error = f"Health check failed: {str(e)}"

            # Wait before retry (except on last attempt)
            if attempt < self.retry_attempts - 1:
                await asyncio.sleep(self.retry_delay)

        # All retries failed
        duration_ms = (time.time() - start_time) * 1000
        return HealthCheckResult(
            name=name,
            status=HealthStatus.UNHEALTHY,
            message=last_error or "Health check failed after retries",
            duration_ms=duration_ms,
            timestamp=time.time(),
            details={"attempts": self.retry_attempts}
        )
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        
        # Run all checks concurrently
        tasks = []
        for name in self.checks:
            task = asyncio.create_task(self.run_check(name))
            tasks.append((name, task))
        
        # Wait for all checks to complete
        for name, task in tasks:
            try:
                result = await task
                results[name] = result
            except Exception as e:
                logger.error(f"Failed to run health check {name}: {e}")
                results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check execution failed: {str(e)}",
                    duration_ms=0,
                    timestamp=time.time()
                )
        
        return results
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        check_results = await self.run_all_checks()
        
        # Calculate overall status
        overall_status = HealthStatus.HEALTHY
        healthy_count = 0
        warning_count = 0
        unhealthy_count = 0
        
        for result in check_results.values():
            if result.status == HealthStatus.HEALTHY:
                healthy_count += 1
            elif result.status == HealthStatus.WARNING:
                warning_count += 1
                if overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.WARNING
            elif result.status == HealthStatus.UNHEALTHY:
                unhealthy_count += 1
                overall_status = HealthStatus.UNHEALTHY
        
        return {
            "overall_status": overall_status.value,
            "summary": {
                "total_checks": len(check_results),
                "healthy": healthy_count,
                "warning": warning_count,
                "unhealthy": unhealthy_count
            },
            "checks": {name: result.to_dict() for name, result in check_results.items()},
            "timestamp": time.time()
        }


# Global health checker instance
health_checker = HealthChecker()


# Built-in health checks
async def database_health_check() -> Dict[str, Any]:
    """Check database connectivity and performance."""
    try:
        from src.database.connection import test_database_connection, get_database_stats
        
        # Test basic connectivity
        is_connected = test_database_connection()
        
        if not is_connected:
            return {
                "status": "unhealthy",
                "message": "Database connection failed"
            }
        
        # Get database statistics
        try:
            stats = get_database_stats()
            return {
                "status": "healthy",
                "message": "Database is accessible",
                "details": stats
            }
        except Exception as e:
            return {
                "status": "warning",
                "message": f"Database connected but stats unavailable: {str(e)}"
            }
            
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Database health check failed: {str(e)}"
        }


async def label_studio_health_check() -> Dict[str, Any]:
    """Check Label Studio service availability."""
    try:
        from src.label_studio.integration import LabelStudioIntegration
        
        integration = LabelStudioIntegration()
        
        # Test Label Studio connectivity
        is_available = await integration.test_connection()
        
        if is_available:
            return {
                "status": "healthy",
                "message": "Label Studio is accessible"
            }
        else:
            return {
                "status": "unhealthy",
                "message": "Label Studio is not accessible"
            }
            
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Label Studio health check failed: {str(e)}"
        }


async def ai_services_health_check() -> Dict[str, Any]:
    """Check AI services availability."""
    try:
        from src.ai.factory import AIAnnotatorFactory

        available_services = []
        unavailable_services = []

        # Test each AI service using the factory's health check method
        for service_name in ["ollama", "huggingface", "zhipu", "baidu"]:
            try:
                health_result = await AIAnnotatorFactory.check_service_health(service_name)
                if health_result.get("available", False):
                    available_services.append(service_name)
                else:
                    unavailable_services.append(service_name)
            except Exception:
                unavailable_services.append(service_name)

        if available_services:
            status = "healthy" if not unavailable_services else "warning"
            message = f"AI services available: {', '.join(available_services)}"
            if unavailable_services:
                message += f", unavailable: {', '.join(unavailable_services)}"
        else:
            # If no AI services are configured, treat as warning not unhealthy
            status = "warning"
            message = "No AI services are currently available"

        return {
            "status": status,
            "message": message,
            "details": {
                "available": available_services,
                "unavailable": unavailable_services
            }
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"AI services health check failed: {str(e)}"
        }


async def storage_health_check() -> Dict[str, Any]:
    """Check storage system health."""
    try:
        import os
        import shutil

        upload_dir = settings.app.upload_dir

        # Get minimum disk space threshold from config
        try:
            min_disk_space_gb = settings.health_check.min_disk_space_gb
        except AttributeError:
            min_disk_space_gb = 1.0

        # Check if upload directory exists and is writable
        if not os.path.exists(upload_dir):
            try:
                os.makedirs(upload_dir, exist_ok=True)
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "message": f"Cannot create upload directory: {str(e)}"
                }

        # Check disk space
        disk_usage = shutil.disk_usage(upload_dir)
        free_space_gb = disk_usage.free / (1024**3)

        if free_space_gb < min_disk_space_gb:
            status = "warning"
            message = f"Low disk space: {free_space_gb:.2f}GB free (minimum: {min_disk_space_gb}GB)"
        else:
            status = "healthy"
            message = f"Storage OK: {free_space_gb:.2f}GB free"

        return {
            "status": status,
            "message": message,
            "details": {
                "upload_dir": upload_dir,
                "free_space_gb": round(free_space_gb, 2),
                "total_space_gb": round(disk_usage.total / (1024**3), 2),
                "min_required_gb": min_disk_space_gb
            }
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Storage health check failed: {str(e)}"
        }


async def security_health_check() -> Dict[str, Any]:
    """Check security system health."""
    try:
        from src.security.controller import SecurityController
        
        controller = SecurityController()
        
        # Test security components
        checks = {
            "encryption": controller.test_encryption(),
            "authentication": controller.test_authentication(),
            "audit_logging": controller.test_audit_logging()
        }
        
        failed_checks = [name for name, result in checks.items() if not result]
        
        if not failed_checks:
            return {
                "status": "healthy",
                "message": "All security checks passed",
                "details": checks
            }
        else:
            return {
                "status": "unhealthy",
                "message": f"Security checks failed: {', '.join(failed_checks)}",
                "details": checks
            }
            
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Security health check failed: {str(e)}"
        }


async def external_dependencies_health_check() -> Dict[str, Any]:
    """Check external dependencies health."""
    try:
        import aiohttp
        
        dependencies = []
        
        # Check internet connectivity (optional)
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get("https://httpbin.org/status/200") as response:
                    if response.status == 200:
                        dependencies.append({"name": "internet", "status": "available"})
                    else:
                        dependencies.append({"name": "internet", "status": "limited"})
        except Exception:
            dependencies.append({"name": "internet", "status": "unavailable"})
        
        # Check if any dependencies are critical
        critical_unavailable = []
        
        if critical_unavailable:
            return {
                "status": "unhealthy",
                "message": f"Critical dependencies unavailable: {', '.join(critical_unavailable)}",
                "details": {"dependencies": dependencies}
            }
        else:
            return {
                "status": "healthy",
                "message": "External dependencies OK",
                "details": {"dependencies": dependencies}
            }
            
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"External dependencies health check failed: {str(e)}"
        }


# Register built-in health checks
health_checker.register_check("database", database_health_check)
health_checker.register_check("label_studio", label_studio_health_check)
health_checker.register_check("ai_services", ai_services_health_check)
health_checker.register_check("storage", storage_health_check)
health_checker.register_check("security", security_health_check)
health_checker.register_check("external_dependencies", external_dependencies_health_check)