#!/bin/bash

# SuperInsight TCB Full-Stack Deployment Script
# Deploys a single container with PostgreSQL, Redis, Label Studio, and FastAPI

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Image configuration
REGISTRY="ccr.ccs.tencentyun.com"
IMAGE_NAME="superinsight/fullstack"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

# Print usage
usage() {
    cat << EOF
Usage: $0 <command> [options]

Commands:
    deploy          Full deployment (build, push, deploy)
    build           Build Docker image only
    push            Push image to registry
    deploy-only     Deploy without building (use existing image)
    rollback        Rollback to previous version
    health          Check service health
    logs            View service logs
    status          Show deployment status
    backup          Backup data to COS
    restore         Restore data from COS

Options:
    -e, --env       Environment (development|staging|production)
    -t, --tag       Docker image tag
    -h, --help      Show this help message

Environment Variables:
    TCB_ENV_ID          TCB environment ID (required)
    TCB_SECRET_ID       TCB secret ID (required)
    TCB_SECRET_KEY      TCB secret key (required)
    POSTGRES_USER       PostgreSQL user (default: superinsight)
    POSTGRES_PASSWORD   PostgreSQL password (required)
    POSTGRES_DB         PostgreSQL database (default: superinsight)

Examples:
    $0 deploy -e production
    $0 build -t v1.0.0
    $0 rollback -e production
EOF
}

# Check required environment variables
check_env_vars() {
    log_step "Checking environment variables..."

    local required_vars=(
        "TCB_ENV_ID"
        "TCB_SECRET_ID"
        "TCB_SECRET_KEY"
        "POSTGRES_PASSWORD"
    )

    local missing=()
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            missing+=("$var")
        fi
    done

    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing required environment variables: ${missing[*]}"
        exit 1
    fi

    # Set defaults
    export POSTGRES_USER="${POSTGRES_USER:-superinsight}"
    export POSTGRES_DB="${POSTGRES_DB:-superinsight}"
    export ENVIRONMENT="${ENVIRONMENT:-production}"

    log_info "Environment variables verified"
}

# Check dependencies
check_dependencies() {
    log_step "Checking dependencies..."

    local deps=("docker" "npm")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            log_error "$dep is not installed"
            exit 1
        fi
    done

    # Check CloudBase CLI
    if ! command -v tcb &> /dev/null; then
        log_info "Installing CloudBase CLI..."
        npm install -g @cloudbase/cli
    fi

    log_info "All dependencies available"
}

# Build fullstack Docker image
build_image() {
    log_step "Building fullstack Docker image..."

    cd "$PROJECT_ROOT"

    local full_image="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

    docker build \
        -t "${full_image}" \
        -f deploy/tcb/Dockerfile.fullstack \
        --build-arg BUILD_DATE="$(date -Iseconds)" \
        --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
        .

    # Also tag as latest if not already
    if [ "$IMAGE_TAG" != "latest" ]; then
        docker tag "${full_image}" "${REGISTRY}/${IMAGE_NAME}:latest"
    fi

    log_info "Image built: ${full_image}"
}

# Push image to Tencent Container Registry
push_image() {
    log_step "Pushing image to Tencent Container Registry..."

    # Login to registry
    echo "${TCB_SECRET_KEY}" | docker login "${REGISTRY}" -u "${TCB_SECRET_ID}" --password-stdin

    local full_image="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

    docker push "${full_image}"

    if [ "$IMAGE_TAG" != "latest" ]; then
        docker push "${REGISTRY}/${IMAGE_NAME}:latest"
    fi

    log_info "Image pushed: ${full_image}"
}

# Deploy to CloudBase
deploy_cloudbase() {
    log_step "Deploying to CloudBase..."

    cd "$PROJECT_ROOT"

    # Login to CloudBase
    if [ -f "deploy/tcb/tcb-key.json" ]; then
        tcb login --key-file deploy/tcb/tcb-key.json
    else
        log_warn "No key file found, using environment credentials"
        tcb login
    fi

    # Deploy using cloudbaserc.json
    tcb framework:deploy --envId "${TCB_ENV_ID}" --mode "${ENVIRONMENT}"

    log_info "CloudBase deployment completed"
}

# Deploy cloud functions
deploy_functions() {
    log_step "Deploying cloud functions..."

    tcb functions:deploy data-extractor --envId "${TCB_ENV_ID}"
    tcb functions:deploy ai-annotator --envId "${TCB_ENV_ID}"
    tcb functions:deploy quality-manager --envId "${TCB_ENV_ID}"

    log_info "Cloud functions deployed"
}

# Health check
health_check() {
    log_step "Performing health check..."

    local max_attempts=10
    local wait_time=15

    # Get service URL
    local service_url
    service_url=$(tcb hosting:detail --envId "${TCB_ENV_ID}" 2>/dev/null | grep -o 'https://[^"]*' | head -1)

    if [ -z "$service_url" ]; then
        log_warn "Could not determine service URL"
        return 1
    fi

    log_info "Checking health at: ${service_url}/health"

    for ((i=1; i<=max_attempts; i++)); do
        if curl -sf "${service_url}/health" -o /dev/null; then
            log_info "Health check passed!"
            echo "Service URL: ${service_url}"
            return 0
        fi
        log_warn "Health check attempt $i/$max_attempts failed, waiting ${wait_time}s..."
        sleep "$wait_time"
    done

    log_error "Health check failed after $max_attempts attempts"
    return 1
}

# Rollback to previous version
rollback() {
    log_step "Rolling back to previous version..."

    tcb framework:rollback --envId "${TCB_ENV_ID}"

    log_info "Rollback initiated, verifying..."
    sleep 30
    health_check
}

# Show deployment status
show_status() {
    log_step "Checking deployment status..."

    tcb hosting:detail --envId "${TCB_ENV_ID}"
}

# View logs
view_logs() {
    log_step "Fetching service logs..."

    tcb logs:detail --envId "${TCB_ENV_ID}" --limit 100
}

# Backup data
backup_data() {
    log_step "Backing up data to COS..."

    local backup_date
    backup_date=$(date '+%Y%m%d_%H%M%S')

    log_info "Backup functionality requires COS configuration"
    log_warn "Please ensure COS_BUCKET and COS credentials are configured"

    # This would typically:
    # 1. Connect to the running container
    # 2. Dump PostgreSQL database
    # 3. Backup Redis data
    # 4. Upload to COS

    log_info "Backup placeholder - implement COS upload logic"
}

# Restore data
restore_data() {
    log_step "Restoring data from COS..."

    log_info "Restore functionality requires COS configuration"
    log_warn "Please specify backup file to restore"

    # This would typically:
    # 1. Download backup from COS
    # 2. Restore PostgreSQL database
    # 3. Restore Redis data

    log_info "Restore placeholder - implement COS download logic"
}

# Full deployment
full_deploy() {
    log_info "Starting full deployment..."

    check_env_vars
    check_dependencies
    build_image
    push_image
    deploy_cloudbase
    deploy_functions

    log_info "Waiting for service to start..."
    sleep 60

    health_check

    log_info "Full deployment completed successfully!"
}

# Parse arguments
COMMAND=""
ENVIRONMENT="production"

while [[ $# -gt 0 ]]; do
    case $1 in
        deploy|build|push|deploy-only|rollback|health|logs|status|backup|restore)
            COMMAND="$1"
            shift
            ;;
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Execute command
case "$COMMAND" in
    deploy)
        full_deploy
        ;;
    build)
        check_dependencies
        build_image
        ;;
    push)
        check_env_vars
        check_dependencies
        push_image
        ;;
    deploy-only)
        check_env_vars
        check_dependencies
        deploy_cloudbase
        deploy_functions
        sleep 60
        health_check
        ;;
    rollback)
        check_env_vars
        check_dependencies
        rollback
        ;;
    health)
        check_env_vars
        check_dependencies
        health_check
        ;;
    logs)
        check_env_vars
        check_dependencies
        view_logs
        ;;
    status)
        check_env_vars
        check_dependencies
        show_status
        ;;
    backup)
        check_env_vars
        backup_data
        ;;
    restore)
        check_env_vars
        restore_data
        ;;
    "")
        log_error "No command specified"
        usage
        exit 1
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        usage
        exit 1
        ;;
esac
