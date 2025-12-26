#!/bin/bash
# Multi-Service Health Check Script
# Checks PostgreSQL, Redis, FastAPI, and Label Studio

set -e

FASTAPI_PORT="${APP_PORT:-8000}"
LABEL_STUDIO_PORT="${LABEL_STUDIO_PORT:-8080}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
REDIS_PORT="${REDIS_PORT:-6379}"

# Exit codes
EXIT_HEALTHY=0
EXIT_UNHEALTHY=1

log() {
    echo "[$(date -Iseconds)] [health-check] $1"
}

check_postgres() {
    if pg_isready -h localhost -p "$POSTGRES_PORT" -q; then
        log "PostgreSQL: HEALTHY"
        return 0
    else
        log "PostgreSQL: UNHEALTHY"
        return 1
    fi
}

check_redis() {
    if nc -z localhost "$REDIS_PORT" 2>/dev/null; then
        log "Redis: HEALTHY"
        return 0
    else
        log "Redis: UNHEALTHY"
        return 1
    fi
}

check_fastapi() {
    if curl -sf "http://localhost:$FASTAPI_PORT/health" > /dev/null 2>&1; then
        log "FastAPI: HEALTHY"
        return 0
    else
        log "FastAPI: UNHEALTHY"
        return 1
    fi
}

check_label_studio() {
    if curl -sf "http://localhost:$LABEL_STUDIO_PORT/health" > /dev/null 2>&1; then
        log "Label Studio: HEALTHY"
        return 0
    else
        # Label Studio might not have /health endpoint, try root
        if curl -sf "http://localhost:$LABEL_STUDIO_PORT/" > /dev/null 2>&1; then
            log "Label Studio: HEALTHY"
            return 0
        fi
        log "Label Studio: UNHEALTHY"
        return 1
    fi
}

# Main health check
main() {
    local healthy=true

    log "Starting health check..."

    # Check all services
    check_postgres || healthy=false
    check_redis || healthy=false
    check_fastapi || healthy=false
    check_label_studio || healthy=false

    if [ "$healthy" = true ]; then
        log "Overall status: HEALTHY"
        exit $EXIT_HEALTHY
    else
        log "Overall status: UNHEALTHY"
        exit $EXIT_UNHEALTHY
    fi
}

# Allow checking individual services
case "${1:-all}" in
    postgres)
        check_postgres
        ;;
    redis)
        check_redis
        ;;
    fastapi)
        check_fastapi
        ;;
    label-studio)
        check_label_studio
        ;;
    all|"")
        main
        ;;
    *)
        log "Usage: $0 [postgres|redis|fastapi|label-studio|all]"
        exit 1
        ;;
esac
