#!/bin/bash
# Graceful Shutdown Handler
# Stops services in reverse order: Label Studio -> FastAPI -> Redis -> PostgreSQL

log() {
    echo "[$(date -Iseconds)] [graceful-shutdown] $1"
}

shutdown_services() {
    log "Initiating graceful shutdown..."

    # Stop Label Studio first
    log "Stopping Label Studio..."
    supervisorctl stop label-studio 2>/dev/null || true
    sleep 2

    # Stop FastAPI
    log "Stopping FastAPI..."
    supervisorctl stop fastapi 2>/dev/null || true
    sleep 2

    # Stop Redis
    log "Stopping Redis..."
    supervisorctl stop redis 2>/dev/null || true
    sleep 2

    # Stop PostgreSQL last
    log "Stopping PostgreSQL..."
    supervisorctl stop postgres 2>/dev/null || true
    sleep 2

    log "Graceful shutdown complete"
}

# Handle signals
trap shutdown_services SIGTERM SIGINT

# If called as event listener
if [ "$1" = "READY" ]; then
    echo "READY"
    while read -r line; do
        case "$line" in
            *PROCESS_STATE_STOPPING*|*PROCESS_STATE_EXITED*)
                shutdown_services
                echo "RESULT 2"
                echo "OK"
                ;;
            *)
                echo "RESULT 2"
                echo "OK"
                ;;
        esac
    done
else
    # Direct invocation
    shutdown_services
fi
