#!/bin/bash
# Database Migration Script
# Runs Alembic migrations for SuperInsight

set -e

log() {
    echo "[$(date -Iseconds)] [init-db] $1"
}

# Set database URL
export DATABASE_URL="${DATABASE_URL:-postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}}"

log "Running database migrations..."
log "Database URL: postgresql://${POSTGRES_USER}:****@localhost:5432/${POSTGRES_DB}"

cd /app

# Check if alembic.ini exists
if [ ! -f "alembic.ini" ]; then
    log "Warning: alembic.ini not found, skipping migrations"
    exit 0
fi

# Run migrations
if /opt/venv/bin/python -m alembic upgrade head; then
    log "Database migrations completed successfully"
else
    log "Warning: Database migrations failed"
    exit 1
fi
