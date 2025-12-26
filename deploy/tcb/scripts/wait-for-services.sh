#!/bin/bash
# Service Dependency Checker
# Waits for PostgreSQL and Redis to be ready before starting dependent services

set -e

POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
TIMEOUT="${WAIT_TIMEOUT:-60}"

log() {
    echo "[$(date -Iseconds)] [wait-for-services] $1"
}

wait_for_postgres() {
    log "Waiting for PostgreSQL at $POSTGRES_HOST:$POSTGRES_PORT..."
    local count=0
    while [ $count -lt $TIMEOUT ]; do
        if pg_isready -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -q; then
            log "PostgreSQL is ready"
            return 0
        fi
        count=$((count + 1))
        sleep 1
    done
    log "ERROR: PostgreSQL did not become ready within $TIMEOUT seconds"
    return 1
}

wait_for_redis() {
    log "Waiting for Redis at $REDIS_HOST:$REDIS_PORT..."
    local count=0
    while [ $count -lt $TIMEOUT ]; do
        if nc -z "$REDIS_HOST" "$REDIS_PORT" 2>/dev/null; then
            log "Redis is ready"
            return 0
        fi
        count=$((count + 1))
        sleep 1
    done
    log "ERROR: Redis did not become ready within $TIMEOUT seconds"
    return 1
}

# Main execution
case "${1:-all}" in
    postgres)
        wait_for_postgres
        ;;
    redis)
        wait_for_redis
        ;;
    all)
        wait_for_postgres
        wait_for_redis
        ;;
    *)
        log "Usage: $0 [postgres|redis|all]"
        exit 1
        ;;
esac

log "All requested services are ready"
