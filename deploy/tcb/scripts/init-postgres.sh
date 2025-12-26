#!/bin/bash
# PostgreSQL Initialization Script
# Creates the database cluster if it doesn't exist

set -e

PGDATA="/var/lib/postgresql/14/main"
POSTGRES_USER="${POSTGRES_USER:-superinsight}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-superinsight_secret}"
POSTGRES_DB="${POSTGRES_DB:-superinsight}"

log() {
    echo "[$(date -Iseconds)] [init-postgres] $1"
}

log "Initializing PostgreSQL database cluster..."

# Check if already initialized
if [ -f "$PGDATA/PG_VERSION" ]; then
    log "PostgreSQL data directory already exists, skipping initialization"
    exit 0
fi

# Ensure directory exists and has correct permissions
mkdir -p "$PGDATA"
chown postgres:postgres "$PGDATA"
chmod 700 "$PGDATA"

# Initialize the database cluster
log "Running initdb..."
su - postgres -c "/usr/lib/postgresql/14/bin/initdb -D $PGDATA --encoding=UTF8 --locale=C.UTF-8"

# Copy custom configuration
if [ -f "/etc/postgresql/14/main/postgresql.conf" ]; then
    log "Copying custom PostgreSQL configuration..."
    cp /etc/postgresql/14/main/postgresql.conf "$PGDATA/postgresql.conf"
    chown postgres:postgres "$PGDATA/postgresql.conf"
fi

if [ -f "/etc/postgresql/14/main/pg_hba.conf" ]; then
    log "Copying custom pg_hba.conf..."
    cp /etc/postgresql/14/main/pg_hba.conf "$PGDATA/pg_hba.conf"
    chown postgres:postgres "$PGDATA/pg_hba.conf"
fi

log "PostgreSQL initialization complete"
