#!/bin/bash
# SuperInsight TCB Container Entrypoint
# Initializes services and starts Supervisor

set -e

echo "=== SuperInsight TCB Container Starting ==="
echo "Timestamp: $(date -Iseconds)"

# Function to log messages
log() {
    echo "[$(date -Iseconds)] $1"
}

# Create required directories if they don't exist
log "Creating required directories..."
mkdir -p /var/lib/postgresql/14/main
mkdir -p /var/lib/redis
mkdir -p /var/log/supervisor
mkdir -p /var/run/supervisor
mkdir -p /run/postgresql
mkdir -p /var/run/redis
mkdir -p /app/uploads
mkdir -p /app/logs
mkdir -p /app/label-studio-data

# Set correct ownership
log "Setting directory permissions..."
chown -R postgres:postgres /var/lib/postgresql /run/postgresql
chown -R redis:redis /var/lib/redis /var/run/redis
chown -R superinsight:superinsight /app/uploads /app/logs /app/label-studio-data

# Initialize PostgreSQL if not already initialized
if [ ! -f "/var/lib/postgresql/14/main/PG_VERSION" ]; then
    log "Initializing PostgreSQL database cluster..."
    /app/scripts/init-postgres.sh
fi

# Configure PostgreSQL authentication
log "Configuring PostgreSQL authentication..."
cp /etc/postgresql/14/main/pg_hba.conf /var/lib/postgresql/14/main/pg_hba.conf 2>/dev/null || true
chown postgres:postgres /var/lib/postgresql/14/main/pg_hba.conf 2>/dev/null || true

# Start PostgreSQL temporarily to create databases
log "Starting PostgreSQL for initialization..."
su - postgres -c "/usr/lib/postgresql/14/bin/pg_ctl -D /var/lib/postgresql/14/main -l /var/log/supervisor/postgres_init.log start" || true

# Wait for PostgreSQL to be ready
log "Waiting for PostgreSQL to be ready..."
for i in {1..30}; do
    if su - postgres -c "pg_isready -q"; then
        log "PostgreSQL is ready"
        break
    fi
    sleep 1
done

# Create databases and users
log "Creating databases and users..."
su - postgres -c "psql -c \"SELECT 1 FROM pg_roles WHERE rolname='${POSTGRES_USER}'\" | grep -q 1 || psql -c \"CREATE USER ${POSTGRES_USER} WITH PASSWORD '${POSTGRES_PASSWORD}' SUPERUSER;\""
su - postgres -c "psql -c \"SELECT 1 FROM pg_database WHERE datname='${POSTGRES_DB}'\" | grep -q 1 || psql -c \"CREATE DATABASE ${POSTGRES_DB} OWNER ${POSTGRES_USER};\""
su - postgres -c "psql -c \"SELECT 1 FROM pg_database WHERE datname='label_studio'\" | grep -q 1 || psql -c \"CREATE DATABASE label_studio OWNER ${POSTGRES_USER};\""

# Run database migrations
log "Running database migrations..."
/app/scripts/init-db.sh || log "Warning: Database migrations failed or already applied"

# Stop temporary PostgreSQL
log "Stopping temporary PostgreSQL..."
su - postgres -c "/usr/lib/postgresql/14/bin/pg_ctl -D /var/lib/postgresql/14/main stop -m fast" || true
sleep 2

log "=== Initialization Complete ==="
log "Starting Supervisor..."

# Execute the main command (Supervisor)
exec "$@"
