#!/bin/bash
# =============================================================================
# SuperInsight TCB Rollback and Disaster Recovery Manager
# =============================================================================
# This script provides automated rollback capabilities and disaster recovery
# procedures for TCB deployments.
#
# Usage:
#   ./rollback-manager.sh <command> [options]
#
# Commands:
#   rollback        - Roll back to a previous version
#   restore         - Restore from backup
#   health-check    - Verify deployment health
#   list-versions   - List available versions for rollback
#   backup          - Create a backup before operations
#   validate        - Validate rollback readiness
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/rollback"
BACKUP_DIR="${PROJECT_ROOT}/backups"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Log file for this session
SESSION_LOG="${LOG_DIR}/rollback_$(date '+%Y%m%d_%H%M%S').log"

# Function to log to both console and file
log_both() {
    echo "$1" | tee -a "${SESSION_LOG}"
}

# =============================================================================
# Validation Functions
# =============================================================================

validate_environment() {
    log_info "Validating environment..."

    # Check required environment variables
    local required_vars=(
        "TCB_ENV_ID"
        "TCB_SECRET_ID"
        "TCB_SECRET_KEY"
    )

    local missing=0
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            log_error "Required environment variable not set: ${var}"
            missing=$((missing + 1))
        fi
    done

    if [ $missing -gt 0 ]; then
        log_error "Missing ${missing} required environment variable(s)"
        return 1
    fi

    # Check CloudBase CLI
    if ! command -v tcb &> /dev/null; then
        log_error "CloudBase CLI (tcb) is not installed"
        return 1
    fi

    log_success "Environment validation passed"
    return 0
}

validate_rollback_readiness() {
    log_info "Validating rollback readiness..."

    # Check if we have a previous version to roll back to
    local versions
    versions=$(tcb framework:list --envId "${TCB_ENV_ID}" 2>/dev/null || echo "")

    if [ -z "$versions" ]; then
        log_warning "No previous versions found for rollback"
        return 1
    fi

    # Check current deployment status
    local status
    status=$(tcb framework:status --envId "${TCB_ENV_ID}" 2>/dev/null || echo "unknown")

    log_info "Current deployment status: ${status}"
    log_success "Rollback readiness validated"
    return 0
}

# =============================================================================
# Backup Functions
# =============================================================================

create_backup() {
    local backup_type="${1:-full}"
    local backup_timestamp
    backup_timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_path="${BACKUP_DIR}/${backup_timestamp}"

    log_info "Creating ${backup_type} backup at ${backup_path}..."
    mkdir -p "${backup_path}"

    case $backup_type in
        "full")
            create_database_backup "${backup_path}"
            create_config_backup "${backup_path}"
            create_redis_backup "${backup_path}"
            ;;
        "database")
            create_database_backup "${backup_path}"
            ;;
        "config")
            create_config_backup "${backup_path}"
            ;;
        *)
            log_error "Unknown backup type: ${backup_type}"
            return 1
            ;;
    esac

    # Create backup manifest
    cat > "${backup_path}/manifest.json" << EOF
{
    "timestamp": "${backup_timestamp}",
    "type": "${backup_type}",
    "environment": "${TCB_ENV_ID}",
    "created_at": "$(date -Iseconds)"
}
EOF

    log_success "Backup created successfully at ${backup_path}"
    echo "${backup_path}"
}

create_database_backup() {
    local backup_path="$1"
    log_info "Creating database backup..."

    # Using pg_dump if available in container
    if command -v pg_dump &> /dev/null; then
        pg_dump -h localhost -U "${POSTGRES_USER:-superinsight}" \
            "${POSTGRES_DB:-superinsight}" > "${backup_path}/database.sql" 2>/dev/null || {
            log_warning "pg_dump not available or failed, skipping database backup"
        }
    else
        log_warning "pg_dump not available, skipping database backup"
    fi
}

create_config_backup() {
    local backup_path="$1"
    log_info "Creating configuration backup..."

    # Backup environment files
    cp -r "${PROJECT_ROOT}/deploy/tcb/env" "${backup_path}/env" 2>/dev/null || true

    # Backup CloudBase configuration
    cp "${PROJECT_ROOT}/deploy/tcb/cloudbaserc.json" "${backup_path}/" 2>/dev/null || true
    cp "${PROJECT_ROOT}/deploy/tcb/tcb-config.yaml" "${backup_path}/" 2>/dev/null || true
}

create_redis_backup() {
    local backup_path="$1"
    log_info "Creating Redis backup..."

    # Save Redis snapshot if redis-cli available
    if command -v redis-cli &> /dev/null; then
        redis-cli BGSAVE 2>/dev/null || log_warning "Redis BGSAVE failed"
    fi
}

# =============================================================================
# Rollback Functions
# =============================================================================

list_versions() {
    log_info "Listing available versions for rollback..."

    # Get version list from TCB
    tcb framework:list --envId "${TCB_ENV_ID}" 2>/dev/null || {
        log_error "Failed to list versions"
        return 1
    }
}

rollback_to_version() {
    local target_version="$1"

    if [ -z "$target_version" ]; then
        log_error "Target version not specified"
        echo "Usage: $0 rollback <version>"
        return 1
    fi

    log_info "Rolling back to version: ${target_version}"

    # Create backup before rollback
    log_info "Creating pre-rollback backup..."
    create_backup "full" || log_warning "Pre-rollback backup failed, continuing..."

    # Perform rollback
    log_info "Executing rollback..."
    tcb framework:rollback --envId "${TCB_ENV_ID}" --version "${target_version}" || {
        log_error "Rollback failed"
        return 1
    }

    # Wait for rollback to complete
    log_info "Waiting for rollback to complete..."
    sleep 30

    # Verify rollback
    if perform_health_check; then
        log_success "Rollback completed successfully"
        return 0
    else
        log_error "Rollback completed but health check failed"
        return 1
    fi
}

rollback_to_previous() {
    log_info "Rolling back to previous version..."

    # Create backup before rollback
    create_backup "full" || log_warning "Pre-rollback backup failed, continuing..."

    # Perform rollback to previous version
    tcb framework:rollback --envId "${TCB_ENV_ID}" || {
        log_error "Rollback failed"
        return 1
    }

    # Wait and verify
    sleep 30
    perform_health_check
}

# =============================================================================
# Health Check Functions
# =============================================================================

perform_health_check() {
    log_info "Performing health check..."

    local max_attempts=5
    local attempt=1
    local wait_time=15

    # Get service URL
    local service_url
    service_url=$(tcb hosting:detail --envId "${TCB_ENV_ID}" 2>/dev/null | grep -o 'https://[^"]*' | head -1)

    if [ -z "$service_url" ]; then
        log_warning "Could not determine service URL, using default health endpoint"
        service_url="http://localhost:8000"
    fi

    local health_endpoint="${service_url}/health"
    log_info "Health endpoint: ${health_endpoint}"

    while [ $attempt -le $max_attempts ]; do
        log_info "Health check attempt ${attempt}/${max_attempts}..."

        if curl -sf "${health_endpoint}" -o /dev/null --connect-timeout 10; then
            log_success "Health check passed!"
            return 0
        fi

        log_warning "Health check failed, waiting ${wait_time}s before retry..."
        sleep $wait_time
        attempt=$((attempt + 1))
    done

    log_error "Health check failed after ${max_attempts} attempts"
    return 1
}

perform_deep_health_check() {
    log_info "Performing deep health check..."

    local all_healthy=true

    # Check main API
    log_info "Checking API health..."
    if ! curl -sf "http://localhost:8000/health" -o /dev/null 2>/dev/null; then
        log_error "API health check failed"
        all_healthy=false
    else
        log_success "API is healthy"
    fi

    # Check PostgreSQL
    log_info "Checking PostgreSQL health..."
    if command -v pg_isready &> /dev/null; then
        if pg_isready -h localhost -p 5432 -U "${POSTGRES_USER:-superinsight}" 2>/dev/null; then
            log_success "PostgreSQL is healthy"
        else
            log_error "PostgreSQL health check failed"
            all_healthy=false
        fi
    fi

    # Check Redis
    log_info "Checking Redis health..."
    if command -v redis-cli &> /dev/null; then
        if redis-cli ping 2>/dev/null | grep -q "PONG"; then
            log_success "Redis is healthy"
        else
            log_error "Redis health check failed"
            all_healthy=false
        fi
    fi

    # Check Label Studio
    log_info "Checking Label Studio health..."
    if curl -sf "http://localhost:8080/health" -o /dev/null 2>/dev/null; then
        log_success "Label Studio is healthy"
    else
        log_warning "Label Studio health check failed"
    fi

    if $all_healthy; then
        log_success "All services are healthy"
        return 0
    else
        log_error "Some services are not healthy"
        return 1
    fi
}

# =============================================================================
# Restore Functions
# =============================================================================

restore_from_backup() {
    local backup_path="$1"

    if [ -z "$backup_path" ] || [ ! -d "$backup_path" ]; then
        log_error "Invalid or missing backup path: ${backup_path}"
        echo "Usage: $0 restore <backup_path>"
        return 1
    fi

    log_info "Restoring from backup: ${backup_path}"

    # Verify backup manifest
    if [ ! -f "${backup_path}/manifest.json" ]; then
        log_error "Invalid backup: manifest.json not found"
        return 1
    fi

    # Restore database if available
    if [ -f "${backup_path}/database.sql" ]; then
        log_info "Restoring database..."
        restore_database "${backup_path}/database.sql" || {
            log_error "Database restore failed"
            return 1
        }
    fi

    # Restore configuration
    if [ -d "${backup_path}/env" ]; then
        log_info "Restoring configuration..."
        cp -r "${backup_path}/env"/* "${PROJECT_ROOT}/deploy/tcb/env/" 2>/dev/null || true
    fi

    log_success "Restore completed successfully"
    return 0
}

restore_database() {
    local sql_file="$1"

    if command -v psql &> /dev/null; then
        psql -h localhost -U "${POSTGRES_USER:-superinsight}" \
            "${POSTGRES_DB:-superinsight}" < "${sql_file}" 2>/dev/null || {
            log_error "Database restore failed"
            return 1
        }
        log_success "Database restored successfully"
    else
        log_warning "psql not available, cannot restore database"
        return 1
    fi
}

# =============================================================================
# Disaster Recovery
# =============================================================================

disaster_recovery() {
    log_info "Initiating disaster recovery procedure..."

    echo "=============================================="
    echo "       DISASTER RECOVERY PROCEDURE"
    echo "=============================================="
    echo ""
    echo "This will attempt to recover the system from a failure."
    echo ""
    read -p "Are you sure you want to proceed? (yes/no): " confirm

    if [ "$confirm" != "yes" ]; then
        log_info "Disaster recovery cancelled"
        return 0
    fi

    # Step 1: Find latest backup
    log_info "Step 1: Finding latest backup..."
    local latest_backup
    latest_backup=$(ls -td "${BACKUP_DIR}"/*/ 2>/dev/null | head -1)

    if [ -z "$latest_backup" ]; then
        log_warning "No backups found, attempting rollback to previous version..."
        rollback_to_previous
        return $?
    fi

    # Step 2: Attempt rollback first
    log_info "Step 2: Attempting rollback to previous version..."
    if rollback_to_previous; then
        log_success "Rollback successful, system recovered"
        return 0
    fi

    # Step 3: Restore from backup
    log_info "Step 3: Rollback failed, restoring from backup..."
    if restore_from_backup "$latest_backup"; then
        log_success "Restore successful, verifying health..."
        perform_deep_health_check
        return $?
    fi

    log_error "Disaster recovery failed, manual intervention required"
    return 1
}

# =============================================================================
# Main
# =============================================================================

print_usage() {
    cat << EOF
SuperInsight TCB Rollback Manager

Usage: $0 <command> [options]

Commands:
    rollback [version]   Roll back to a specific version (or previous if not specified)
    restore <path>       Restore from a backup
    health               Perform health check
    deep-health          Perform deep health check
    list                 List available versions for rollback
    backup [type]        Create backup (full, database, config)
    validate             Validate rollback readiness
    dr                   Initiate disaster recovery

Options:
    -h, --help          Show this help message
    -v, --verbose       Enable verbose output

Examples:
    $0 rollback                     # Roll back to previous version
    $0 rollback v1.2.3              # Roll back to specific version
    $0 restore /backups/20231215    # Restore from backup
    $0 backup full                  # Create full backup
    $0 health                       # Check deployment health

EOF
}

main() {
    local command="${1:-}"
    shift || true

    case "$command" in
        rollback)
            validate_environment || exit 1
            if [ -n "$1" ]; then
                rollback_to_version "$1"
            else
                rollback_to_previous
            fi
            ;;
        restore)
            validate_environment || exit 1
            restore_from_backup "$1"
            ;;
        health)
            perform_health_check
            ;;
        deep-health)
            perform_deep_health_check
            ;;
        list)
            validate_environment || exit 1
            list_versions
            ;;
        backup)
            create_backup "${1:-full}"
            ;;
        validate)
            validate_environment || exit 1
            validate_rollback_readiness
            ;;
        dr|disaster-recovery)
            validate_environment || exit 1
            disaster_recovery
            ;;
        -h|--help|help)
            print_usage
            ;;
        *)
            log_error "Unknown command: ${command}"
            print_usage
            exit 1
            ;;
    esac
}

main "$@"
