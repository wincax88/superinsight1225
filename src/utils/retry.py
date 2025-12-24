"""
Enhanced Retry and Circuit Breaker Utilities for SuperInsight Platform.

Provides advanced retry mechanisms with exponential backoff, jitter,
circuit breaker patterns, and graceful degradation strategies.
"""

import asyncio
import logging
import random
import time
from typing import Any, Callable, Dict, Optional, Type, Union, List
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import threading

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategy types."""
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIBONACCI = "fibonacci"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    backoff_multiplier: float = 2.0
    jitter: bool = True
    jitter_range: float = 0.1
    retryable_exceptions: List[Type[Exception]] = field(default_factory=list)
    non_retryable_exceptions: List[Type[Exception]] = field(default_factory=list)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    timeout: float = 30.0


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.
    
    Prevents cascading failures by temporarily stopping calls to
    a failing service and allowing it time to recover.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.lock = threading.Lock()
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through enhanced circuit breaker with adaptive behavior."""
        with self.lock:
            current_time = time.time()
            
            if self.state == CircuitState.OPEN:
                # Enhanced recovery logic
                time_since_failure = current_time - self.last_failure_time
                
                if time_since_failure < self.config.recovery_timeout:
                    # Still in timeout period
                    raise CircuitBreakerError(
                        f"Circuit breaker {self.name} is OPEN "
                        f"(will retry in {self.config.recovery_timeout - time_since_failure:.1f}s)"
                    )
                else:
                    # Transition to half-open with enhanced logic
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN after {time_since_failure:.1f}s")
        
        try:
            # Execute the function with enhanced monitoring
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            with self.lock:
                if self.state == CircuitState.HALF_OPEN:
                    self.success_count += 1
                    
                    # Enhanced recovery criteria
                    if self.success_count >= self.config.success_threshold:
                        self.state = CircuitState.CLOSED
                        self.failure_count = 0
                        logger.info(f"Circuit breaker {self.name} transitioning to CLOSED after {self.success_count} successful calls")
                    else:
                        logger.debug(f"Circuit breaker {self.name} half-open success {self.success_count}/{self.config.success_threshold}")
                        
                elif self.state == CircuitState.CLOSED:
                    # Reset failure count on success and track performance
                    self.failure_count = 0
                    
                    # Adaptive threshold adjustment based on performance
                    if hasattr(self, 'avg_execution_time'):
                        self.avg_execution_time = (self.avg_execution_time * 0.9) + (execution_time * 0.1)
                    else:
                        self.avg_execution_time = execution_time
                    
                    # If execution time is consistently high, lower failure threshold
                    if execution_time > self.avg_execution_time * 3:
                        logger.warning(f"Circuit breaker {self.name} detected slow execution: {execution_time:.2f}s vs avg {self.avg_execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = current_time
                
                # Enhanced failure analysis
                is_timeout = "timeout" in str(e).lower()
                is_connection_error = any(term in str(e).lower() for term in ["connection", "network", "unreachable"])
                
                # Adjust failure threshold based on error type
                effective_threshold = self.config.failure_threshold
                if is_timeout or is_connection_error:
                    effective_threshold = max(2, self.config.failure_threshold - 1)  # More sensitive to network issues
                
                if self.state == CircuitState.HALF_OPEN:
                    # Failed during half-open, go back to open with extended timeout
                    self.state = CircuitState.OPEN
                    # Increase recovery timeout for repeated failures
                    self.config.recovery_timeout = min(self.config.recovery_timeout * 1.5, 300.0)  # Max 5 minutes
                    logger.warning(f"Circuit breaker {self.name} failed during HALF_OPEN, returning to OPEN with extended timeout {self.config.recovery_timeout:.1f}s")
                    
                elif self.state == CircuitState.CLOSED and self.failure_count >= effective_threshold:
                    # Too many failures, open the circuit
                    self.state = CircuitState.OPEN
                    logger.warning(f"Circuit breaker {self.name} opening due to {self.failure_count} failures (threshold: {effective_threshold})")
                else:
                    logger.debug(f"Circuit breaker {self.name} failure {self.failure_count}/{effective_threshold}")
            
            raise
    
    async def async_call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function through circuit breaker."""
        with self.lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time < self.config.recovery_timeout:
                    raise CircuitBreakerError(f"Circuit breaker {self.name} is OPEN")
                else:
                    # Transition to half-open
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
        
        try:
            # Execute the async function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            with self.lock:
                if self.state == CircuitState.HALF_OPEN:
                    self.success_count += 1
                    if self.success_count >= self.config.success_threshold:
                        self.state = CircuitState.CLOSED
                        self.failure_count = 0
                        logger.info(f"Circuit breaker {self.name} transitioning to CLOSED")
                elif self.state == CircuitState.CLOSED:
                    self.failure_count = 0  # Reset failure count on success
            
            return result
            
        except Exception as e:
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.state == CircuitState.HALF_OPEN:
                    # Failed during half-open, go back to open
                    self.state = CircuitState.OPEN
                    logger.warning(f"Circuit breaker {self.name} failed during HALF_OPEN, returning to OPEN")
                elif self.state == CircuitState.CLOSED and self.failure_count >= self.config.failure_threshold:
                    # Too many failures, open the circuit
                    self.state = CircuitState.OPEN
                    logger.warning(f"Circuit breaker {self.name} opening due to {self.failure_count} failures")
            
            raise
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        with self.lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": self.last_failure_time
            }


class RetryExecutor:
    """
    Advanced retry executor with multiple strategies and circuit breaker integration.
    """
    
    def __init__(self, config: RetryConfig):
        self.config = config
        
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt based on strategy with enhanced jitter."""
        if self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.base_delay
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** attempt)
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * (attempt + 1)
        elif self.config.strategy == RetryStrategy.FIBONACCI:
            # Enhanced Fibonacci sequence for delays
            if attempt <= 1:
                delay = self.config.base_delay
            else:
                fib_a, fib_b = 1, 1
                for _ in range(attempt - 1):
                    fib_a, fib_b = fib_b, fib_a + fib_b
                delay = self.config.base_delay * fib_b
        else:
            delay = self.config.base_delay
        
        # Apply max delay limit
        delay = min(delay, self.config.max_delay)
        
        # Enhanced jitter implementation
        if self.config.jitter:
            if self.config.jitter_range > 0:
                # Full jitter: random value between 0 and calculated delay
                jitter_amount = delay * self.config.jitter_range
                # Use decorrelated jitter for better distribution
                delay = random.uniform(delay - jitter_amount, delay + jitter_amount)
            else:
                # Default jitter (10% of delay)
                jitter_amount = delay * 0.1
                delay += random.uniform(-jitter_amount, jitter_amount)
            
            delay = max(0.1, delay)  # Ensure minimum delay of 100ms
        
        return delay
    
    def _should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if the exception should trigger a retry."""
        # Check if we've exceeded max attempts
        if attempt >= self.config.max_attempts:
            return False
        
        # Check non-retryable exceptions first
        if self.config.non_retryable_exceptions:
            for exc_type in self.config.non_retryable_exceptions:
                if isinstance(exception, exc_type):
                    return False
        
        # Check retryable exceptions
        if self.config.retryable_exceptions:
            for exc_type in self.config.retryable_exceptions:
                if isinstance(exception, exc_type):
                    return True
            return False  # If retryable list is specified, only retry those
        
        # Default: retry most exceptions except some critical ones
        non_retryable_defaults = [
            KeyboardInterrupt,
            SystemExit,
            MemoryError,
            SyntaxError,
            TypeError,
            ValueError
        ]
        
        for exc_type in non_retryable_defaults:
            if isinstance(exception, exc_type):
                return False
        
        return True
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if not self._should_retry(e, attempt):
                    logger.debug(f"Not retrying after attempt {attempt + 1}: {str(e)}")
                    raise
                
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {str(e)}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    time.sleep(delay)
        
        # If we get here, all attempts failed
        logger.error(f"All {self.config.max_attempts} attempts failed")
        raise last_exception
    
    async def async_execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if not self._should_retry(e, attempt):
                    logger.debug(f"Not retrying after attempt {attempt + 1}: {str(e)}")
                    raise
                
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {str(e)}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    await asyncio.sleep(delay)
        
        # If we get here, all attempts failed
        logger.error(f"All {self.config.max_attempts} attempts failed")
        raise last_exception


# Global circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_circuit_breaker_lock = threading.Lock()


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get or create a circuit breaker by name."""
    with _circuit_breaker_lock:
        if name not in _circuit_breakers:
            if config is None:
                config = CircuitBreakerConfig()
            _circuit_breakers[name] = CircuitBreaker(name, config)
        return _circuit_breakers[name]


def list_circuit_breakers() -> Dict[str, Dict[str, Any]]:
    """List all circuit breakers and their states."""
    with _circuit_breaker_lock:
        return {name: cb.get_state() for name, cb in _circuit_breakers.items()}


# Decorators for easy usage
def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    backoff_multiplier: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[List[Type[Exception]]] = None,
    non_retryable_exceptions: Optional[List[Type[Exception]]] = None
):
    """Decorator for adding retry logic to functions."""
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        strategy=strategy,
        backoff_multiplier=backoff_multiplier,
        jitter=jitter,
        retryable_exceptions=retryable_exceptions or [],
        non_retryable_exceptions=non_retryable_exceptions or []
    )
    
    executor = RetryExecutor(config)
    
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await executor.async_execute(func, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return executor.execute(func, *args, **kwargs)
            return wrapper
    
    return decorator


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    success_threshold: int = 3,
    timeout: float = 30.0
):
    """Decorator for adding circuit breaker protection to functions."""
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        success_threshold=success_threshold,
        timeout=timeout
    )
    
    cb = get_circuit_breaker(name, config)
    
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await cb.async_call(func, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return cb.call(func, *args, **kwargs)
            return wrapper
    
    return decorator


def retry_with_circuit_breaker(
    circuit_name: str,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0
):
    """Decorator combining retry logic with circuit breaker protection."""
    # Configure retry
    retry_config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        strategy=strategy
    )
    
    # Configure circuit breaker
    cb_config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout
    )
    
    executor = RetryExecutor(retry_config)
    cb = get_circuit_breaker(circuit_name, cb_config)
    
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                async def protected_func():
                    return await cb.async_call(func, *args, **kwargs)
                return await executor.async_execute(protected_func)
            return async_wrapper
        else:
            @wraps(func)
            def wrapper(*args, **kwargs):
                def protected_func():
                    return cb.call(func, *args, **kwargs)
                return executor.execute(protected_func)
            return wrapper
    
    return decorator