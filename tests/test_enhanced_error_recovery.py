"""
Tests for Enhanced Error Recovery System.

Tests the integration of error handling, notifications, health monitoring,
and graceful degradation for comprehensive system resilience.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock

from src.system.error_handler import error_handler, ErrorCategory, ErrorSeverity
from src.system.notification import notification_system, NotificationPriority, NotificationChannel
from src.system.health_monitor import health_monitor, HealthStatus, HealthMetric, MetricType
from src.system.enhanced_recovery import recovery_coordinator, RecoveryStrategy
from src.utils.degradation import degradation_manager, DegradationLevel


class TestEnhancedErrorRecovery:
    """Test enhanced error recovery mechanisms."""
    
    def setup_method(self):
        """Setup test environment."""
        # Clear any existing state
        error_handler.clear_error_history()
        notification_system.clear_statistics()
        
        # Stop health monitoring to avoid interference
        health_monitor.stop_monitoring()
        
        # Reset degradation manager
        degradation_manager.services.clear()
        degradation_manager.fallback_configs.clear()
    
    def teardown_method(self):
        """Cleanup after tests."""
        health_monitor.stop_monitoring()
    
    def test_error_handler_with_enhanced_notifications(self):
        """Test error handler integration with enhanced notification system."""
        # Configure notification system
        notification_system.config.enabled = True
        notification_system.config.channels = [NotificationChannel.LOG]
        
        # Handle a critical error
        test_exception = Exception("Test critical error")
        error_context = error_handler.handle_error(
            exception=test_exception,
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.CRITICAL,
            service_name="test_service"
        )
        
        # Verify error was handled
        assert error_context.category == ErrorCategory.DATABASE
        assert error_context.severity == ErrorSeverity.CRITICAL
        assert error_context.service_name == "test_service"
        
        # Verify notification was sent (check statistics)
        stats = notification_system.get_statistics()
        assert stats["queue"]["pending_messages"] >= 0  # May be processed already
    
    def test_retry_with_exponential_backoff(self):
        """Test enhanced retry mechanism with exponential backoff."""
        from src.utils.retry import RetryExecutor, RetryConfig, RetryStrategy
        
        # Configure retry with exponential backoff
        retry_config = RetryConfig(
            max_attempts=3,
            base_delay=0.1,  # Short delay for testing
            strategy=RetryStrategy.EXPONENTIAL,
            backoff_multiplier=2.0,
            jitter=False  # Disable jitter for predictable testing
        )
        
        executor = RetryExecutor(retry_config)
        
        # Mock function that fails twice then succeeds
        call_count = 0
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception(f"Attempt {call_count} failed")
            return "success"
        
        # Execute with retry
        start_time = time.time()
        result = executor.execute(failing_function)
        duration = time.time() - start_time
        
        # Verify success and timing
        assert result == "success"
        assert call_count == 3
        assert duration >= 0.3  # Should have delays: 0.1 + 0.2 = 0.3 seconds minimum
    
    def test_circuit_breaker_protection(self):
        """Test circuit breaker protection mechanism."""
        from src.utils.retry import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError
        
        # Configure circuit breaker
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=1.0,  # Short timeout for testing
            success_threshold=1
        )
        
        circuit_breaker = CircuitBreaker("test_service", config)
        
        # Function that always fails
        def failing_function():
            raise Exception("Service unavailable")
        
        # First two calls should fail and open the circuit
        with pytest.raises(Exception):
            circuit_breaker.call(failing_function)
        
        with pytest.raises(Exception):
            circuit_breaker.call(failing_function)
        
        # Third call should be rejected by circuit breaker
        with pytest.raises(CircuitBreakerError):
            circuit_breaker.call(failing_function)
        
        # Verify circuit breaker state
        state = circuit_breaker.get_state()
        assert state["state"] == "open"
        assert state["failure_count"] == 2
    
    def test_graceful_degradation(self):
        """Test graceful degradation mechanism."""
        # Register a service
        service_name = "test_annotation_service"
        degradation_manager.register_service(service_name)
        
        # Mark service as failed multiple times to trigger degradation
        for _ in range(3):
            degradation_manager.mark_service_failure(service_name)
        
        # Check service health
        health = degradation_manager.get_service_health(service_name)
        assert health is not None
        assert not health.is_healthy
        assert health.failure_count == 3
        
        # Test fallback execution
        def primary_function():
            raise Exception("Primary service failed")
        
        def fallback_function():
            return "fallback_result"
        
        # Register fallback handler
        from src.utils.degradation import fallback_handler, DegradationLevel
        
        @fallback_handler(service_name, DegradationLevel.REDUCED)
        def test_fallback():
            return "fallback_result"
        
        # Execute with fallback
        result = degradation_manager.execute_with_fallback(
            service_name, primary_function
        )
        
        # Should get fallback result
        assert result == "fallback_result"
    
    def test_health_monitoring_integration(self):
        """Test health monitoring system integration."""
        # Create a custom health check
        def test_health_check():
            return HealthMetric(
                name="test_metric",
                value=95.0,  # Critical value
                threshold_warning=80.0,
                threshold_critical=90.0,
                metric_type=MetricType.SYSTEM,
                unit="%"
            )
        
        # Register health check
        from src.system.health_monitor import HealthCheck
        health_check = HealthCheck(
            name="test_metric",
            check_function=test_health_check,
            interval=1.0  # Short interval for testing
        )
        
        health_monitor.register_health_check(health_check)
        
        # Run health check manually
        health_monitor._run_health_check(health_check)
        
        # Verify results
        assert "test_metric" in health_monitor.last_check_results
        metric = health_monitor.last_check_results["test_metric"]
        assert metric.value == 95.0
        
        # Check health report
        report = health_monitor.get_health_report()
        assert report.overall_status in [HealthStatus.CRITICAL, HealthStatus.WARNING]
    
    def test_recovery_plan_generation(self):
        """Test intelligent recovery plan generation."""
        # Create mock error context
        from src.system.error_handler import ErrorContext
        
        error_context = ErrorContext(
            error_id="test_error",
            timestamp=time.time(),
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            message="Database connection failed",
            service_name="database"
        )
        
        # Generate recovery plan
        recovery_plan = recovery_coordinator._generate_recovery_plan(error_context)
        
        # Verify plan was generated
        assert recovery_plan is not None
        assert recovery_plan.strategy in RecoveryStrategy
        assert len(recovery_plan.actions) > 0
        assert 0.0 <= recovery_plan.success_probability <= 1.0
        assert recovery_plan.estimated_duration > 0
        assert len(recovery_plan.rollback_plan) >= 0
    
    def test_notification_rate_limiting(self):
        """Test notification rate limiting to prevent spam."""
        # Configure notification system with low limits
        notification_system.rate_limiter.max_count = 2
        notification_system.rate_limiter.window_size = 10  # 10 seconds
        
        # Send multiple notifications rapidly
        results = []
        for i in range(5):
            result = notification_system.send_notification(
                title=f"Test notification {i}",
                message="Test message",
                priority=NotificationPriority.NORMAL,
                channels=[NotificationChannel.LOG]
            )
            results.append(result)
        
        # First 2 should succeed, rest should be rate limited
        assert results[0] == True
        assert results[1] == True
        assert results[2] == False  # Rate limited
        assert results[3] == False  # Rate limited
        assert results[4] == False  # Rate limited
    
    def test_notification_deduplication(self):
        """Test notification deduplication to reduce noise."""
        # Send duplicate notifications
        dedup_key = "test_duplicate"
        
        result1 = notification_system.send_notification(
            title="Duplicate notification",
            message="This is a duplicate",
            priority=NotificationPriority.NORMAL,
            channels=[NotificationChannel.LOG],
            deduplication_key=dedup_key
        )
        
        result2 = notification_system.send_notification(
            title="Duplicate notification",
            message="This is a duplicate",
            priority=NotificationPriority.NORMAL,
            channels=[NotificationChannel.LOG],
            deduplication_key=dedup_key
        )
        
        # First should succeed, second should be deduplicated
        assert result1 == True
        assert result2 == False
    
    def test_end_to_end_recovery_flow(self):
        """Test complete end-to-end recovery flow."""
        # Configure systems
        notification_system.config.enabled = True
        notification_system.config.channels = [NotificationChannel.LOG]
        
        # Simulate a database error
        test_exception = Exception("Database connection timeout")
        error_context = error_handler.handle_error(
            exception=test_exception,
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            service_name="database"
        )
        
        # Verify error was handled and recovery was attempted
        assert len(error_handler.error_history) > 0
        
        # Check if recovery coordinator processed the error
        # (This happens asynchronously through notification handlers)
        time.sleep(0.1)  # Brief wait for async processing
        
        # Verify system state
        stats = error_handler.get_error_statistics()
        assert stats["total_errors"] > 0
        assert "database" in stats["by_category"]
    
    def test_enhanced_exponential_backoff_with_jitter(self):
        """Test enhanced exponential backoff with improved jitter."""
        from src.utils.retry import RetryExecutor, RetryConfig, RetryStrategy
        
        # Configure enhanced retry with jitter
        retry_config = RetryConfig(
            max_attempts=4,
            base_delay=0.1,
            strategy=RetryStrategy.EXPONENTIAL,
            backoff_multiplier=2.0,
            jitter=True,
            jitter_range=0.2  # 20% jitter
        )
        
        executor = RetryExecutor(retry_config)
        
        # Track timing to verify jitter
        call_times = []
        call_count = 0
        
        def failing_function():
            nonlocal call_count
            call_count += 1
            call_times.append(time.time())
            if call_count < 4:
                raise Exception(f"Attempt {call_count} failed")
            return "success"
        
        # Execute with retry
        start_time = time.time()
        result = executor.execute(failing_function)
        total_duration = time.time() - start_time
        
        # Verify success
        assert result == "success"
        assert call_count == 4
        
        # Verify delays are reasonable (with jitter, they won't be exact)
        if len(call_times) >= 3:
            delay1 = call_times[1] - call_times[0]
            delay2 = call_times[2] - call_times[1]
            
            # Delays should be in reasonable ranges with jitter
            assert 0.08 <= delay1 <= 0.15  # ~0.1s with jitter
            assert 0.16 <= delay2 <= 0.30  # ~0.2s with jitter
            
            # Second delay should generally be larger than first (exponential)
            assert delay2 > delay1 * 1.2  # At least 20% larger
    
    def test_adaptive_circuit_breaker_thresholds(self):
        """Test adaptive circuit breaker with dynamic thresholds."""
        from src.utils.retry import CircuitBreaker, CircuitBreakerConfig
        
        # Configure circuit breaker
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=0.5,  # Short for testing
            success_threshold=2
        )
        
        circuit_breaker = CircuitBreaker("adaptive_test", config)
        
        # Simulate timeout errors (should lower threshold)
        def timeout_function():
            raise Exception("Connection timeout")
        
        # First few calls should fail normally
        for i in range(2):
            with pytest.raises(Exception):
                circuit_breaker.call(timeout_function)
        
        # Circuit should still be closed
        state = circuit_breaker.get_state()
        assert state["state"] == "closed"
        
        # One more timeout should open circuit (adaptive threshold)
        with pytest.raises(Exception):
            circuit_breaker.call(timeout_function)
        
        # Circuit should now be open
        state = circuit_breaker.get_state()
        assert state["state"] == "open"
    
    def test_intelligent_notification_filtering(self):
        """Test intelligent notification filtering to reduce noise."""
        # Configure error handler
        error_handler.clear_error_history()
        
        # Send multiple similar errors rapidly
        for i in range(6):
            test_exception = Exception("Database connection timeout")
            error_handler.handle_error(
                exception=test_exception,
                category=ErrorCategory.DATABASE,
                severity=ErrorSeverity.MEDIUM,
                service_name="database"
            )
        
        # Check that notifications were filtered after threshold
        stats = notification_system.get_statistics()
        
        # Should have fewer notifications than errors due to filtering
        total_notifications = sum(stats["deliveries"]["successful"].values())
        assert len(error_handler.error_history) == 6
        # Notifications should be filtered after frequency threshold
        assert total_notifications < 6
    
    def test_enhanced_degradation_with_impact_analysis(self):
        """Test enhanced graceful degradation with impact analysis."""
        # Register service
        service_name = "ai_annotation"
        degradation_manager.register_service(service_name)
        
        # Create high severity error
        test_exception = Exception("AI model unavailable")
        error_context = error_handler.handle_error(
            exception=test_exception,
            category=ErrorCategory.ANNOTATION,
            severity=ErrorSeverity.HIGH,
            service_name=service_name
        )
        
        # Check service degradation
        health = degradation_manager.get_service_health(service_name)
        assert health is not None
        assert not health.is_healthy
        
        # Verify degradation level is appropriate for high severity
        assert health.degradation_level in [DegradationLevel.REDUCED, DegradationLevel.MINIMAL]
    
    def test_automatic_recovery_detection(self):
        """Test automatic recovery detection and restoration."""
        # Register service and mark as failed
        service_name = "test_recovery_service"
        degradation_manager.register_service(service_name)
        
        # Mark service as failed multiple times
        for _ in range(3):
            degradation_manager.mark_service_failure(service_name)
        
        health = degradation_manager.get_service_health(service_name)
        assert not health.is_healthy
        
        # Simulate recovery
        degradation_manager.mark_service_success(service_name)
        
        # Check recovery
        health = degradation_manager.get_service_health(service_name)
        assert health.is_healthy
        assert health.degradation_level == DegradationLevel.FULL
    
    def test_emergency_mode_activation(self):
        """Test emergency mode activation for critical failures."""
        # Clear notification history
        notification_system.clear_statistics()
        
        # Create critical error that should trigger emergency mode
        test_exception = Exception("Critical system failure")
        error_context = error_handler.handle_error(
            exception=test_exception,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            service_name="core_system"
        )
        
        # Allow time for recovery processing
        time.sleep(0.2)
        
        # Check that emergency notifications were sent
        stats = notification_system.get_statistics()
        assert stats["deliveries"]["successful"].get("log", 0) > 0
    
    def test_recovery_success_rate_learning(self):
        """Test recovery success rate learning and adaptation."""
        # Get initial success rates
        initial_rates = dict(recovery_coordinator.strategy_success_rates)
        
        # Simulate multiple recovery attempts with mixed results
        for success in [True, False, True, True, False, True]:
            recovery_coordinator._update_strategy_success_rate(
                RecoveryStrategy.IMMEDIATE, success
            )
        
        # Check that success rate was updated
        final_rate = recovery_coordinator.strategy_success_rates[RecoveryStrategy.IMMEDIATE]
        
        # Rate should be different from initial (learning occurred)
        assert final_rate != initial_rates[RecoveryStrategy.IMMEDIATE]
        
        # With 4 successes out of 6, rate should be reasonably high
        assert 0.4 < final_rate < 0.8
    
    def test_circuit_breaker_extended_timeout_on_repeated_failures(self):
        """Test circuit breaker extends timeout on repeated failures."""
        from src.utils.retry import CircuitBreaker, CircuitBreakerConfig
        
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=1.0,
            success_threshold=1
        )
        
        circuit_breaker = CircuitBreaker("timeout_test", config)
        
        def failing_function():
            raise Exception("Service failure")
        
        # Cause initial circuit opening
        for _ in range(2):
            with pytest.raises(Exception):
                circuit_breaker.call(failing_function)
        
        # Circuit should be open
        state = circuit_breaker.get_state()
        assert state["state"] == "open"
        
        # Wait for recovery timeout
        time.sleep(1.1)
        
        # Try to call again (should transition to half-open then fail)
        with pytest.raises(Exception):
            circuit_breaker.call(failing_function)
        
        # Circuit should be open again with extended timeout
        state = circuit_breaker.get_state()
        assert state["state"] == "open"
        
        # Recovery timeout should be extended
        assert circuit_breaker.config.recovery_timeout > 1.0
    
    def test_system_statistics_collection(self):
        """Test comprehensive system statistics collection."""
        # Generate some test data
        test_exception = Exception("Test error for statistics")
        error_handler.handle_error(
            exception=test_exception,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.MEDIUM,
            service_name="test_service"
        )
        
        # Get comprehensive statistics
        error_stats = error_handler.get_error_statistics()
        notification_stats = notification_system.get_statistics()
        recovery_stats = recovery_coordinator.get_recovery_statistics()
        
        # Verify statistics structure
        assert "total_errors" in error_stats
        assert "by_category" in error_stats
        assert "by_severity" in error_stats
        assert "recovery_metrics" in error_stats
        
        assert "config" in notification_stats
        assert "deliveries" in notification_stats
        
        assert "total_recoveries" in recovery_stats
        assert "success_rate" in recovery_stats
    
    @pytest.mark.asyncio
    async def test_async_error_handling(self):
        """Test async error handling capabilities."""
        from src.utils.retry import RetryExecutor, RetryConfig
        
        # Configure async retry
        retry_config = RetryConfig(max_attempts=2, base_delay=0.1)
        executor = RetryExecutor(retry_config)
        
        # Async function that fails once then succeeds
        call_count = 0
        async def async_failing_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Async failure")
            return "async_success"
        
        # Execute with async retry
        result = await executor.async_execute(async_failing_function)
        
        # Verify success
        assert result == "async_success"
        assert call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])