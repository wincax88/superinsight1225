# System Health Check Fixes - Design Document

## Overview

This design addresses the health check implementation gaps in the SuperInsight platform. The current system has three failing health checks due to missing methods in service integrations. This design provides a comprehensive solution to implement proper health checking while maintaining system stability and performance, with support for graceful degradation, configurable parameters, and Kubernetes probe compatibility.

## Architecture

### Health Check Flow
```
Health Checker → Service Integration → External Service → Status Response
     ↓                    ↓                   ↓              ↓
  Aggregate           Test Method         API Call        Success/Failure
   Results             Implementation      Validation      Error Details
     ↓                                                          ↓
Configuration    ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←    Retry Logic
Parameters                                                   & Backoff
```

### Service Classification

**Core Services** (Must be healthy for system operation):
- Database connectivity
- File storage access
- Basic authentication

**Optional Services** (Can fail without affecting core operations):
- Label Studio integration
- AI annotation services
- Advanced security features

### Service Integration Points

1. **Label Studio Integration**
   - Add `test_connection()` method to `LabelStudioIntegration` class
   - Implement API connectivity check with configurable timeout handling
   - Validate authentication and project access
   - Support graceful degradation when unavailable

2. **AI Services Integration**
   - Fix `AIAnnotatorFactory` import in `src/ai/factory.py`
   - Implement health check for all configured AI providers
   - Add graceful degradation for unavailable services
   - Support API key validation for each provider

3. **Security Controller**
   - Add `test_encryption()` method to `SecurityController` class
   - Verify password hashing and JWT token functionality
   - Test database connectivity for user authentication
   - Support degraded mode for non-critical security features

## Components and Interfaces

### Health Check Configuration
```python
@dataclass
class HealthCheckConfig:
    """Configuration for health check parameters."""
    timeout_seconds: int = 5
    retry_attempts: int = 3
    retry_backoff_factor: float = 2.0
    enabled_checks: List[str] = field(default_factory=list)
    check_intervals: Dict[str, int] = field(default_factory=dict)
    
    @classmethod
    def from_environment(cls) -> 'HealthCheckConfig':
        """Load configuration from environment variables."""
        # Implementation details in tasks
```

### Label Studio Health Check
```python
class LabelStudioIntegration:
    def __init__(self, config: HealthCheckConfig):
        self.config = config
        
    def test_connection(self) -> Dict[str, Any]:
        """Test Label Studio connectivity and authentication.
        
        Returns:
            Health status with connection details, response time,
            and authentication validation results.
        """
        # Implementation details in tasks
```

### AI Services Health Check
```python
class AIAnnotatorFactory:
    def __init__(self, config: HealthCheckConfig):
        self.config = config
        
    @staticmethod
    def get_health_status() -> Dict[str, Any]:
        """Check AI services availability across all configured providers.
        
        Returns:
            Aggregated health status for all AI providers with
            individual provider status and authentication validation.
        """
        # Implementation details in tasks
        
    def _check_provider_health(self, provider: str) -> Dict[str, Any]:
        """Check individual AI provider health."""
        # Implementation details in tasks
```

### Security Health Check
```python
class SecurityController:
    def __init__(self, config: HealthCheckConfig):
        self.config = config
        
    def test_encryption(self) -> Dict[str, Any]:
        """Test encryption and authentication functionality.
        
        Returns:
            Health status including password hashing verification,
            JWT token operations, and database connectivity.
        """
        # Implementation details in tasks
```

### Health Status Aggregator
```python
class HealthStatusAggregator:
    """Aggregates individual service health checks into overall system status."""
    
    def __init__(self, config: HealthCheckConfig):
        self.config = config
        
    def aggregate_status(self, service_statuses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate individual service statuses into overall system health.
        
        Args:
            service_statuses: Dictionary of service names to their health status
            
        Returns:
            Overall system health status with detailed breakdown
        """
        # Implementation details in tasks
        
    def determine_overall_status(self, core_services: List[str], 
                               optional_services: List[str],
                               statuses: Dict[str, str]) -> str:
        """Determine overall system status based on core and optional service health."""
        # Implementation details in tasks
```

## Data Models

### Health Check Response Format
```python
{
    "status": "healthy|warning|unhealthy",
    "message": "Descriptive status message",
    "details": {
        "service_specific": "information",
        "response_time_ms": 123.45,
        "last_check": "2025-12-23T13:00:00Z",
        "authentication_valid": true,
        "retry_count": 0,
        "error_details": "Optional error information"
    }
}
```

### Kubernetes Probe Response Format
```python
{
    "status": "UP|DOWN",
    "checks": {
        "database": {"status": "UP"},
        "labelstudio": {"status": "UP", "details": {"response_time": "45ms"}},
        "ai_services": {"status": "DOWN", "details": {"error": "Connection timeout"}},
        "security": {"status": "UP"}
    }
}
```

### Service Status Levels
- **healthy**: All core services operational, optional services may have warnings
- **warning**: Core services operational, some optional services degraded
- **unhealthy**: One or more core services failing, system may not function properly

### Performance Metrics
```python
{
    "response_times": {
        "labelstudio": 45.2,
        "ai_services": 120.8,
        "security": 12.1
    },
    "success_rates": {
        "labelstudio": 0.98,
        "ai_services": 0.85,
        "security": 1.0
    },
    "last_failure_times": {
        "ai_services": "2025-12-23T12:45:00Z"
    }
}
```
## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Health Check Method Implementation
*For any* required service integration class, the health check method should exist and be callable when requested
**Validates: Requirements 1.1, 2.1, 3.1**

### Property 2: Health Check Response Format Consistency
*For any* health check method, the response should always include status, message, and details fields in the correct format
**Validates: Requirements 4.3, 4.4**

### Property 3: Service Connection Status Accuracy
*For any* service health check, when the service is accessible, the method should return success status, and when unreachable, it should return failure with error details
**Validates: Requirements 1.2, 1.3, 2.2, 2.3, 3.2**

### Property 4: Timeout Handling Compliance
*For any* health check with configured timeout, the check should complete within the timeout period or return a timeout error
**Validates: Requirements 1.4, 6.1**

### Property 5: Authentication Validation
*For any* service requiring authentication, the health check should validate API keys, tokens, or credentials and report authentication status
**Validates: Requirements 1.5, 2.5, 3.3, 3.4, 3.5**

### Property 6: Graceful Service Degradation
*For any* failed optional service, the overall system health should remain operational (healthy or warning) and not become unhealthy
**Validates: Requirements 5.1, 5.2, 5.3**

### Property 7: Status Aggregation Logic
*For any* combination of core and optional service statuses, the overall system status should be healthy when all core services pass, warning when optional services fail, and unhealthy only when core services fail
**Validates: Requirements 4.1, 4.2**

### Property 8: Configuration Parameter Loading
*For any* health check configuration parameter, the system should load the value from environment variables or use sensible defaults
**Validates: Requirements 6.5**

### Property 9: Retry Mechanism Behavior
*For any* failed service connection, the system should automatically retry with exponential backoff according to configured parameters
**Validates: Requirements 5.5, 6.2**

### Property 10: Kubernetes Probe Compatibility
*For any* health check response, the format should be compatible with Kubernetes readiness and liveness probe expectations
**Validates: Requirements 4.5**

## Error Handling

### Service Unavailability Scenarios
- **Label Studio unreachable**: Log warning, continue with manual annotation only, set status to "warning"
- **AI services unavailable**: Log warning, disable AI annotation features, set status to "warning"  
- **Security issues**: Log error, may affect authentication but not core data access, evaluate criticality
- **Database connectivity**: Log error, set status to "unhealthy" as this affects core operations

### Timeout Management Strategy
- **Default timeout**: 5 seconds for external service calls (configurable via `HEALTH_CHECK_TIMEOUT`)
- **Configurable per service**: Different timeouts for different service types via environment variables
- **Exponential backoff**: Retry failed connections with configurable backoff factor (default 2.0)
- **Maximum retry attempts**: Configurable retry limit (default 3 attempts)

### Graceful Degradation Implementation
- **Core services** (database, storage): Must be healthy for system operation
- **Optional services** (Label Studio, AI): Can fail without affecting overall system health
- **Clear status indicators**: Provide detailed status for each service mode
- **Feature disabling**: Automatically disable features when their dependencies are unavailable
- **Recovery detection**: Automatically re-enable features when services recover

### Configuration Error Handling
- **Missing environment variables**: Use sensible defaults with warning logs
- **Invalid configuration values**: Validate and fall back to defaults
- **Malformed configuration**: Log errors and use default configuration
- **Runtime configuration changes**: Support dynamic reconfiguration without restart

## Testing Strategy

### Unit Tests
- Test each health check method individually with mocked dependencies
- Mock external service responses (success, failure, timeout scenarios)
- Verify error handling and timeout behavior across all services
- Test configuration loading and validation with various environment setups
- Test authentication validation for all service types
- Verify retry logic and exponential backoff implementation

### Property-Based Tests
- Generate random service states and verify health check responses follow correct format
- Test timeout handling with various delay scenarios and configuration values
- Verify graceful degradation across different failure combinations
- Test configuration parameter validation with random valid/invalid values
- Verify status aggregation logic with all possible service state combinations
- Test retry mechanism behavior with various failure patterns
- Validate Kubernetes probe response format compliance

### Integration Tests
- Test actual service connectivity with real external services
- Verify end-to-end health check flow from request to aggregated response
- Test system behavior during planned service outages
- Validate Kubernetes readiness and liveness probe compatibility
- Test configuration changes and dynamic reconfiguration
- Verify performance metrics collection and reporting
- Test service recovery detection and feature re-enabling

### Performance Tests
- Measure health check response times under various load conditions
- Test concurrent health check execution
- Verify timeout accuracy and consistency
- Test memory usage during extended health check operations