"""
Integration tests for Health Check System.

Tests:
- Task 4.2: Integration Tests for Health Checker
  - Overall health status aggregation
  - Graceful degradation scenarios
  - Kubernetes probe compatibility
"""

import pytest
import asyncio
import time
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from src.system.health import (
    HealthChecker, HealthStatus, HealthCheckResult,
    database_health_check, label_studio_health_check,
    ai_services_health_check, storage_health_check,
    security_health_check, external_dependencies_health_check
)


class TestHealthStatus:
    """Tests for HealthStatus enumeration."""

    def test_status_values(self):
        """Test health status enumeration values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.WARNING.value == "warning"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"

    def test_status_comparison(self):
        """Test health status comparison."""
        assert HealthStatus.HEALTHY == HealthStatus.HEALTHY
        assert HealthStatus.HEALTHY != HealthStatus.WARNING


class TestHealthCheckResult:
    """Tests for HealthCheckResult data class."""

    def test_create_result(self):
        """Test creating health check result."""
        result = HealthCheckResult(
            name="test_check",
            status=HealthStatus.HEALTHY,
            message="Test passed",
            duration_ms=50.5,
            timestamp=time.time()
        )

        assert result.name == "test_check"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "Test passed"
        assert result.duration_ms == 50.5

    def test_result_with_details(self):
        """Test result with details."""
        result = HealthCheckResult(
            name="detailed_check",
            status=HealthStatus.WARNING,
            message="Partial success",
            duration_ms=100.0,
            timestamp=time.time(),
            details={"items_checked": 10, "items_failed": 2}
        )

        assert result.details is not None
        assert result.details["items_checked"] == 10

    def test_result_to_dict(self):
        """Test result serialization."""
        timestamp = time.time()
        result = HealthCheckResult(
            name="serialization_test",
            status=HealthStatus.HEALTHY,
            message="OK",
            duration_ms=25.0,
            timestamp=timestamp,
            details={"key": "value"}
        )

        data = result.to_dict()

        assert data["name"] == "serialization_test"
        assert data["status"] == "healthy"
        assert data["message"] == "OK"
        assert data["duration_ms"] == 25.0
        assert data["timestamp"] == timestamp
        assert data["details"]["key"] == "value"


class TestHealthChecker:
    """Tests for HealthChecker functionality."""

    @pytest.fixture
    def checker(self):
        """Create health checker instance."""
        with patch.object(HealthChecker, '_load_config'):
            checker = HealthChecker()
            checker.check_timeout = 30
            checker.retry_attempts = 3
            checker.retry_delay = 0.1
            checker.enabled = True
            checker.check_toggles = {}
            return checker

    def test_initialization(self, checker):
        """Test checker initialization."""
        assert checker.checks == {}
        assert checker.enabled is True

    def test_register_check(self, checker):
        """Test registering health check."""
        async def test_check():
            return True

        checker.register_check("test", test_check)

        assert "test" in checker.checks
        assert checker.checks["test"] == test_check

    def test_is_check_enabled(self, checker):
        """Test check enabled status."""
        checker.check_toggles = {"enabled_check": True, "disabled_check": False}

        assert checker.is_check_enabled("enabled_check") is True
        assert checker.is_check_enabled("disabled_check") is False
        assert checker.is_check_enabled("unknown_check") is True  # Default enabled

    def test_is_check_enabled_global_disable(self, checker):
        """Test global health check disable."""
        checker.enabled = False

        assert checker.is_check_enabled("any_check") is False

    @pytest.mark.asyncio
    async def test_run_nonexistent_check(self, checker):
        """Test running non-existent health check."""
        result = await checker.run_check("nonexistent")

        assert result.status == HealthStatus.UNKNOWN
        assert "not found" in result.message

    @pytest.mark.asyncio
    async def test_run_disabled_check(self, checker):
        """Test running disabled health check."""
        checker.check_toggles["disabled_test"] = False

        async def test_check():
            return True

        checker.register_check("disabled_test", test_check)

        result = await checker.run_check("disabled_test")

        assert result.status == HealthStatus.HEALTHY
        assert "disabled" in result.message.lower()

    @pytest.mark.asyncio
    async def test_run_async_check_success(self, checker):
        """Test running successful async health check."""
        async def passing_check():
            return {"status": "healthy", "message": "All good"}

        checker.register_check("passing", passing_check)

        result = await checker.run_check("passing")

        assert result.status == HealthStatus.HEALTHY
        assert result.message == "All good"
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_run_sync_check_success(self, checker):
        """Test running successful sync health check."""
        def sync_check():
            return True

        checker.register_check("sync_check", sync_check)

        result = await checker.run_check("sync_check")

        assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_run_check_boolean_result(self, checker):
        """Test health check with boolean result."""
        async def bool_check():
            return True

        checker.register_check("bool_check", bool_check)

        result = await checker.run_check("bool_check")
        assert result.status == HealthStatus.HEALTHY

        async def failing_bool_check():
            return False

        checker.register_check("failing_bool", failing_bool_check)

        result = await checker.run_check("failing_bool")
        assert result.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_run_check_with_timeout(self, checker):
        """Test health check timeout handling."""
        checker.check_timeout = 0.1
        checker.retry_attempts = 1

        async def slow_check():
            await asyncio.sleep(1)
            return True

        checker.register_check("slow_check", slow_check)

        result = await checker.run_check("slow_check")

        assert result.status == HealthStatus.UNHEALTHY
        assert "timed out" in result.message.lower()

    @pytest.mark.asyncio
    async def test_run_check_with_exception(self, checker):
        """Test health check exception handling."""
        checker.retry_attempts = 1

        async def failing_check():
            raise Exception("Something went wrong")

        checker.register_check("failing", failing_check)

        result = await checker.run_check("failing")

        assert result.status == HealthStatus.UNHEALTHY
        assert "failed" in result.message.lower()

    @pytest.mark.asyncio
    async def test_run_check_with_retries(self, checker):
        """Test health check retry logic."""
        checker.retry_attempts = 3
        checker.retry_delay = 0.01

        attempt_count = 0

        async def flaky_check():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Temporary failure")
            return True

        checker.register_check("flaky", flaky_check)

        result = await checker.run_check("flaky")

        assert result.status == HealthStatus.HEALTHY
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_run_all_checks(self, checker):
        """Test running all registered checks."""
        async def check1():
            return {"status": "healthy", "message": "Check 1 OK"}

        async def check2():
            return {"status": "healthy", "message": "Check 2 OK"}

        async def check3():
            return {"status": "warning", "message": "Check 3 warning"}

        checker.register_check("check1", check1)
        checker.register_check("check2", check2)
        checker.register_check("check3", check3)

        results = await checker.run_all_checks()

        assert len(results) == 3
        assert "check1" in results
        assert "check2" in results
        assert "check3" in results
        assert results["check1"].status == HealthStatus.HEALTHY
        assert results["check3"].status == HealthStatus.WARNING

    @pytest.mark.asyncio
    async def test_run_all_checks_concurrent(self, checker):
        """Test that checks run concurrently."""
        start_time = time.time()

        async def slow_check1():
            await asyncio.sleep(0.1)
            return True

        async def slow_check2():
            await asyncio.sleep(0.1)
            return True

        checker.register_check("slow1", slow_check1)
        checker.register_check("slow2", slow_check2)

        await checker.run_all_checks()

        duration = time.time() - start_time
        # Should take ~0.1s, not ~0.2s if run sequentially
        assert duration < 0.2

    @pytest.mark.asyncio
    async def test_get_system_health_all_healthy(self, checker):
        """Test system health with all healthy checks."""
        async def healthy_check1():
            return {"status": "healthy", "message": "OK"}

        async def healthy_check2():
            return {"status": "healthy", "message": "OK"}

        checker.register_check("healthy1", healthy_check1)
        checker.register_check("healthy2", healthy_check2)

        health = await checker.get_system_health()

        assert health["overall_status"] == "healthy"
        assert health["summary"]["total_checks"] == 2
        assert health["summary"]["healthy"] == 2
        assert health["summary"]["warning"] == 0
        assert health["summary"]["unhealthy"] == 0

    @pytest.mark.asyncio
    async def test_get_system_health_with_warning(self, checker):
        """Test system health with warning status."""
        async def healthy_check():
            return {"status": "healthy", "message": "OK"}

        async def warning_check():
            return {"status": "warning", "message": "Low disk space"}

        checker.register_check("healthy", healthy_check)
        checker.register_check("warning", warning_check)

        health = await checker.get_system_health()

        assert health["overall_status"] == "warning"
        assert health["summary"]["healthy"] == 1
        assert health["summary"]["warning"] == 1

    @pytest.mark.asyncio
    async def test_get_system_health_with_unhealthy(self, checker):
        """Test system health with unhealthy status."""
        async def healthy_check():
            return {"status": "healthy", "message": "OK"}

        async def unhealthy_check():
            return {"status": "unhealthy", "message": "Service down"}

        checker.register_check("healthy", healthy_check)
        checker.register_check("unhealthy", unhealthy_check)

        health = await checker.get_system_health()

        assert health["overall_status"] == "unhealthy"
        assert health["summary"]["unhealthy"] == 1

    @pytest.mark.asyncio
    async def test_get_system_health_includes_checks(self, checker):
        """Test that system health includes check details."""
        async def test_check():
            return {
                "status": "healthy",
                "message": "OK",
                "details": {"version": "1.0"}
            }

        checker.register_check("test", test_check)

        health = await checker.get_system_health()

        assert "checks" in health
        assert "test" in health["checks"]
        assert health["checks"]["test"]["details"]["version"] == "1.0"


class TestOverallHealthStatusAggregation:
    """Tests for overall health status aggregation (Task 4.2 - Status Aggregation)."""

    @pytest.fixture
    def checker(self):
        """Create health checker instance."""
        with patch.object(HealthChecker, '_load_config'):
            checker = HealthChecker()
            checker.check_timeout = 30
            checker.retry_attempts = 1
            checker.retry_delay = 0.1
            checker.enabled = True
            checker.check_toggles = {}
            return checker

    @pytest.mark.asyncio
    async def test_aggregation_all_healthy(self, checker):
        """Test: HEALTHY = all checks HEALTHY."""
        for i in range(5):
            async def healthy():
                return {"status": "healthy", "message": "OK"}
            checker.register_check(f"check_{i}", healthy)

        health = await checker.get_system_health()

        assert health["overall_status"] == "healthy"
        assert health["summary"]["healthy"] == 5

    @pytest.mark.asyncio
    async def test_aggregation_warning_priority(self, checker):
        """Test: WARNING = at least one WARNING, no UNHEALTHY."""
        async def healthy():
            return {"status": "healthy", "message": "OK"}

        async def warning():
            return {"status": "warning", "message": "Warning"}

        checker.register_check("healthy1", healthy)
        checker.register_check("healthy2", healthy)
        checker.register_check("warning", warning)

        health = await checker.get_system_health()

        assert health["overall_status"] == "warning"

    @pytest.mark.asyncio
    async def test_aggregation_unhealthy_priority(self, checker):
        """Test: UNHEALTHY = at least one UNHEALTHY."""
        async def healthy():
            return {"status": "healthy", "message": "OK"}

        async def warning():
            return {"status": "warning", "message": "Warning"}

        async def unhealthy():
            return {"status": "unhealthy", "message": "Error"}

        checker.register_check("healthy", healthy)
        checker.register_check("warning", warning)
        checker.register_check("unhealthy", unhealthy)

        health = await checker.get_system_health()

        # Unhealthy takes priority over warning
        assert health["overall_status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_aggregation_counts(self, checker):
        """Test correct counting of check statuses."""
        for i in range(3):
            async def healthy():
                return {"status": "healthy", "message": "OK"}
            checker.register_check(f"healthy_{i}", healthy)

        for i in range(2):
            async def warning():
                return {"status": "warning", "message": "Warn"}
            checker.register_check(f"warning_{i}", warning)

        async def unhealthy():
            return {"status": "unhealthy", "message": "Error"}
        checker.register_check("unhealthy_0", unhealthy)

        health = await checker.get_system_health()

        assert health["summary"]["healthy"] == 3
        assert health["summary"]["warning"] == 2
        assert health["summary"]["unhealthy"] == 1
        assert health["summary"]["total_checks"] == 6


class TestGracefulDegradation:
    """Tests for graceful degradation scenarios (Task 4.2 - Graceful Degradation)."""

    @pytest.fixture
    def checker(self):
        """Create health checker instance."""
        with patch.object(HealthChecker, '_load_config'):
            checker = HealthChecker()
            checker.check_timeout = 0.5
            checker.retry_attempts = 1
            checker.retry_delay = 0.01
            checker.enabled = True
            checker.check_toggles = {}
            return checker

    @pytest.mark.asyncio
    async def test_partial_check_failure(self, checker):
        """Test system continues when some checks fail."""
        async def passing():
            return {"status": "healthy", "message": "OK"}

        async def failing():
            raise Exception("Service unavailable")

        checker.register_check("passing", passing)
        checker.register_check("failing", failing)

        # Should not raise exception
        health = await checker.get_system_health()

        # Should have results for both
        assert len(health["checks"]) == 2
        assert health["checks"]["passing"]["status"] == "healthy"
        assert health["checks"]["failing"]["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_timeout_handling(self, checker):
        """Test graceful handling of slow checks."""
        async def fast():
            return {"status": "healthy", "message": "Fast"}

        async def slow():
            await asyncio.sleep(10)  # Will timeout
            return {"status": "healthy", "message": "Slow"}

        checker.register_check("fast", fast)
        checker.register_check("slow", slow)

        health = await checker.get_system_health()

        # Fast check should succeed
        assert health["checks"]["fast"]["status"] == "healthy"
        # Slow check should timeout but not crash
        assert health["checks"]["slow"]["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_disabled_checks_graceful(self, checker):
        """Test disabled checks don't affect health."""
        checker.check_toggles["disabled"] = False

        async def enabled():
            return {"status": "healthy", "message": "OK"}

        async def disabled():
            raise Exception("Should not run")

        checker.register_check("enabled", enabled)
        checker.register_check("disabled", disabled)

        health = await checker.get_system_health()

        assert health["checks"]["enabled"]["status"] == "healthy"
        assert health["checks"]["disabled"]["status"] == "healthy"  # Returns healthy when disabled

    @pytest.mark.asyncio
    async def test_empty_checks_list(self, checker):
        """Test health with no registered checks."""
        health = await checker.get_system_health()

        assert health["overall_status"] == "healthy"
        assert health["summary"]["total_checks"] == 0

    @pytest.mark.asyncio
    async def test_mixed_sync_async_checks(self, checker):
        """Test mixing synchronous and async checks."""
        async def async_check():
            return {"status": "healthy", "message": "Async OK"}

        def sync_check():
            return {"status": "healthy", "message": "Sync OK"}

        checker.register_check("async", async_check)
        checker.register_check("sync", sync_check)

        health = await checker.get_system_health()

        assert health["checks"]["async"]["status"] == "healthy"
        assert health["checks"]["sync"]["status"] == "healthy"


class TestKubernetesProbeCompatibility:
    """Tests for Kubernetes probe compatibility (Task 4.2 - K8s Probes)."""

    @pytest.fixture
    def checker(self):
        """Create health checker instance."""
        with patch.object(HealthChecker, '_load_config'):
            checker = HealthChecker()
            checker.check_timeout = 5
            checker.retry_attempts = 1
            checker.retry_delay = 0.1
            checker.enabled = True
            checker.check_toggles = {}
            return checker

    @pytest.mark.asyncio
    async def test_liveness_probe_format(self, checker):
        """Test health response is suitable for liveness probe."""
        async def basic_check():
            return {"status": "healthy", "message": "OK"}

        checker.register_check("basic", basic_check)

        health = await checker.get_system_health()

        # K8s expects simple status
        assert health["overall_status"] in ["healthy", "warning", "unhealthy", "unknown"]
        assert "timestamp" in health

    @pytest.mark.asyncio
    async def test_readiness_probe_format(self, checker):
        """Test health response is suitable for readiness probe."""
        async def db_check():
            return {"status": "healthy", "message": "Database connected"}

        async def cache_check():
            return {"status": "healthy", "message": "Cache available"}

        checker.register_check("database", db_check)
        checker.register_check("cache", cache_check)

        health = await checker.get_system_health()

        # For readiness, we need detailed check info
        assert "checks" in health
        assert "summary" in health

    @pytest.mark.asyncio
    async def test_response_time_acceptable(self, checker):
        """Test health check completes in acceptable time for K8s."""
        async def quick_check():
            return {"status": "healthy", "message": "OK"}

        for i in range(10):
            checker.register_check(f"check_{i}", quick_check)

        start = time.time()
        health = await checker.get_system_health()
        duration = time.time() - start

        # K8s default timeout is often 1s
        assert duration < 1.0

    @pytest.mark.asyncio
    async def test_healthy_returns_true_semantics(self, checker):
        """Test healthy status can be used as boolean."""
        async def passing():
            return {"status": "healthy", "message": "OK"}

        checker.register_check("test", passing)

        health = await checker.get_system_health()

        # Can determine pass/fail for probe
        is_healthy = health["overall_status"] == "healthy"
        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_unhealthy_returns_false_semantics(self, checker):
        """Test unhealthy status can be used as boolean."""
        async def failing():
            return {"status": "unhealthy", "message": "Down"}

        checker.register_check("test", failing)

        health = await checker.get_system_health()

        is_healthy = health["overall_status"] == "healthy"
        assert is_healthy is False


class TestBuiltInHealthChecks:
    """Tests for built-in health check functions."""

    @pytest.mark.asyncio
    async def test_database_health_check_success(self):
        """Test database health check with successful connection."""
        with patch("src.system.health.test_database_connection", return_value=True):
            with patch("src.system.health.get_database_stats", return_value={"tables": 10}):
                result = await database_health_check()

                assert result["status"] == "healthy"
                assert "Database is accessible" in result["message"]

    @pytest.mark.asyncio
    async def test_database_health_check_failure(self):
        """Test database health check with failed connection."""
        with patch("src.system.health.test_database_connection", return_value=False):
            result = await database_health_check()

            assert result["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_database_health_check_stats_warning(self):
        """Test database health check with stats failure."""
        with patch("src.system.health.test_database_connection", return_value=True):
            with patch("src.system.health.get_database_stats", side_effect=Exception("Stats error")):
                result = await database_health_check()

                assert result["status"] == "warning"

    @pytest.mark.asyncio
    async def test_storage_health_check_success(self):
        """Test storage health check with sufficient space."""
        with patch("src.system.health.settings") as mock_settings:
            mock_settings.app.upload_dir = "/tmp"
            mock_settings.health_check.min_disk_space_gb = 0.1

            with patch("os.path.exists", return_value=True):
                with patch("shutil.disk_usage") as mock_disk:
                    mock_disk.return_value = MagicMock(
                        free=10 * 1024**3,  # 10 GB
                        total=100 * 1024**3
                    )

                    result = await storage_health_check()

                    assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_storage_health_check_low_space(self):
        """Test storage health check with low disk space."""
        with patch("src.system.health.settings") as mock_settings:
            mock_settings.app.upload_dir = "/tmp"
            mock_settings.health_check.min_disk_space_gb = 5.0

            with patch("os.path.exists", return_value=True):
                with patch("shutil.disk_usage") as mock_disk:
                    mock_disk.return_value = MagicMock(
                        free=1 * 1024**3,  # 1 GB (below threshold)
                        total=100 * 1024**3
                    )

                    result = await storage_health_check()

                    assert result["status"] == "warning"
                    assert "Low disk space" in result["message"]

    @pytest.mark.asyncio
    async def test_external_dependencies_health_check(self):
        """Test external dependencies health check."""
        # Mock aiohttp session
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_session.__aenter__.return_value = mock_session
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_session_class.return_value = mock_session

            result = await external_dependencies_health_check()

            assert result["status"] == "healthy"


class TestHealthCheckConfiguration:
    """Tests for health check configuration."""

    def test_default_configuration(self):
        """Test default health check configuration."""
        with patch("src.system.health.settings") as mock_settings:
            mock_settings.health_check.health_check_timeout = 30
            mock_settings.health_check.health_check_retry_attempts = 3
            mock_settings.health_check.health_check_retry_delay = 1.0
            mock_settings.health_check.health_check_enabled = True
            mock_settings.health_check.database_check_enabled = True
            mock_settings.health_check.label_studio_check_enabled = True
            mock_settings.health_check.ai_services_check_enabled = True
            mock_settings.health_check.storage_check_enabled = True
            mock_settings.health_check.security_check_enabled = True
            mock_settings.health_check.external_deps_check_enabled = True

            checker = HealthChecker()

            assert checker.check_timeout == 30
            assert checker.retry_attempts == 3
            assert checker.enabled is True

    def test_configuration_fallback(self):
        """Test fallback to defaults when settings unavailable."""
        with patch("src.system.health.settings") as mock_settings:
            # Simulate missing attribute
            del mock_settings.health_check

            checker = HealthChecker()

            # Should use default values
            assert checker.check_timeout == 30
            assert checker.retry_attempts == 3
            assert checker.enabled is True


class TestHealthCheckIntegration:
    """Integration tests for full health check scenarios."""

    @pytest.fixture
    def checker(self):
        """Create health checker instance."""
        with patch.object(HealthChecker, '_load_config'):
            checker = HealthChecker()
            checker.check_timeout = 30
            checker.retry_attempts = 3
            checker.retry_delay = 0.1
            checker.enabled = True
            checker.check_toggles = {}
            return checker

    @pytest.mark.asyncio
    async def test_full_health_check_flow(self, checker):
        """Test complete health check workflow."""
        # Register multiple checks
        async def db_check():
            return {"status": "healthy", "message": "DB OK", "details": {"latency_ms": 5}}

        async def cache_check():
            return {"status": "healthy", "message": "Cache OK"}

        async def service_check():
            return {"status": "warning", "message": "High latency"}

        checker.register_check("database", db_check)
        checker.register_check("cache", cache_check)
        checker.register_check("service", service_check)

        # Run all checks
        health = await checker.get_system_health()

        # Verify structure
        assert "overall_status" in health
        assert "summary" in health
        assert "checks" in health
        assert "timestamp" in health

        # Verify status aggregation
        assert health["overall_status"] == "warning"

        # Verify individual check results
        assert health["checks"]["database"]["status"] == "healthy"
        assert health["checks"]["database"]["details"]["latency_ms"] == 5

    @pytest.mark.asyncio
    async def test_health_check_with_retries(self, checker):
        """Test health checks with retry behavior."""
        call_count = 0

        async def flaky_check():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            return {"status": "healthy", "message": "Recovered"}

        checker.register_check("flaky", flaky_check)

        result = await checker.run_check("flaky")

        assert result.status == HealthStatus.HEALTHY
        assert call_count >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
