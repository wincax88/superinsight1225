#!/bin/bash
# Knowledge Graph Environment Setup Script
# Task 18: Environment setup scripts for Knowledge Graph system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_ROOT/config"
VENV_DIR="$PROJECT_ROOT/.venv"

# Default settings
NEO4J_VERSION="5.15.0"
NEO4J_PORT="7687"
NEO4J_HTTP_PORT="7474"
NEO4J_PASSWORD="superinsight_kg"
SPACY_MODEL="en_core_web_lg"

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# =============================================================================
# Prerequisite Checks
# =============================================================================

check_prerequisites() {
    log_info "Checking prerequisites..."

    local missing_deps=()

    # Check Python
    if check_command python3; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        log_success "Python 3 found: $PYTHON_VERSION"
    else
        missing_deps+=("python3")
    fi

    # Check pip
    if check_command pip3; then
        log_success "pip3 found"
    else
        missing_deps+=("pip3")
    fi

    # Check Docker (optional, for Neo4j)
    if check_command docker; then
        log_success "Docker found"
        DOCKER_AVAILABLE=true
    else
        log_warning "Docker not found - Neo4j must be installed manually"
        DOCKER_AVAILABLE=false
    fi

    # Check Docker Compose (optional)
    if check_command docker-compose || docker compose version &> /dev/null 2>&1; then
        log_success "Docker Compose found"
        COMPOSE_AVAILABLE=true
    else
        log_warning "Docker Compose not found"
        COMPOSE_AVAILABLE=false
    fi

    if [ ${#missing_deps[@]} -gt 0 ]; then
        log_error "Missing required dependencies: ${missing_deps[*]}"
        log_info "Please install them before continuing."
        exit 1
    fi

    log_success "All required prerequisites met"
}

# =============================================================================
# Python Environment Setup
# =============================================================================

setup_python_environment() {
    log_info "Setting up Python virtual environment..."

    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
        log_success "Created virtual environment at $VENV_DIR"
    else
        log_info "Virtual environment already exists"
    fi

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"

    # Upgrade pip
    pip install --upgrade pip

    log_success "Python environment ready"
}

install_python_dependencies() {
    log_info "Installing Python dependencies..."

    source "$VENV_DIR/bin/activate"

    # Core dependencies
    pip install neo4j>=5.0.0
    pip install spacy>=3.5.0
    pip install pydantic>=2.0.0
    pip install httpx>=0.24.0
    pip install fastapi>=0.100.0
    pip install uvicorn>=0.22.0
    pip install python-multipart>=0.0.6
    pip install pyyaml>=6.0

    # Optional: GraphQL support
    pip install strawberry-graphql>=0.200.0 || log_warning "GraphQL package installation failed"

    # Optional: WebSocket support
    pip install websockets>=11.0 || log_warning "WebSocket package installation failed"

    # Testing dependencies
    pip install pytest>=7.0.0
    pip install pytest-asyncio>=0.21.0
    pip install pytest-benchmark>=4.0.0

    log_success "Python dependencies installed"
}

install_spacy_model() {
    log_info "Installing spaCy language model: $SPACY_MODEL..."

    source "$VENV_DIR/bin/activate"

    # Download spaCy model
    python3 -m spacy download "$SPACY_MODEL" || {
        log_warning "Could not download $SPACY_MODEL, trying smaller model..."
        python3 -m spacy download en_core_web_sm
    }

    log_success "spaCy model installed"
}

# =============================================================================
# Neo4j Setup
# =============================================================================

setup_neo4j_docker() {
    log_info "Setting up Neo4j using Docker..."

    if [ "$DOCKER_AVAILABLE" != "true" ]; then
        log_error "Docker is not available. Please install Docker or set up Neo4j manually."
        return 1
    fi

    # Check if Neo4j container already exists
    if docker ps -a --format '{{.Names}}' | grep -q "^superinsight-neo4j$"; then
        log_info "Neo4j container already exists"

        # Start if not running
        if ! docker ps --format '{{.Names}}' | grep -q "^superinsight-neo4j$"; then
            docker start superinsight-neo4j
            log_success "Started existing Neo4j container"
        else
            log_info "Neo4j container is already running"
        fi
    else
        # Create and start new container
        docker run -d \
            --name superinsight-neo4j \
            -p "$NEO4J_HTTP_PORT:7474" \
            -p "$NEO4J_PORT:7687" \
            -e NEO4J_AUTH="neo4j/$NEO4J_PASSWORD" \
            -e NEO4J_PLUGINS='["apoc"]' \
            -v neo4j-data:/data \
            -v neo4j-logs:/logs \
            neo4j:$NEO4J_VERSION

        log_success "Neo4j container created and started"
        log_info "Waiting for Neo4j to be ready..."

        # Wait for Neo4j to be ready
        local max_attempts=30
        local attempt=0
        while [ $attempt -lt $max_attempts ]; do
            if curl -s "http://localhost:$NEO4J_HTTP_PORT" > /dev/null 2>&1; then
                break
            fi
            sleep 2
            attempt=$((attempt + 1))
        done

        if [ $attempt -eq $max_attempts ]; then
            log_warning "Neo4j may not be fully ready yet. Please wait a moment."
        else
            log_success "Neo4j is ready!"
        fi
    fi

    log_info "Neo4j Browser: http://localhost:$NEO4J_HTTP_PORT"
    log_info "Neo4j Bolt: bolt://localhost:$NEO4J_PORT"
    log_info "Default credentials: neo4j / $NEO4J_PASSWORD"
}

create_docker_compose() {
    log_info "Creating Docker Compose configuration..."

    cat > "$PROJECT_ROOT/docker-compose.kg.yml" << 'EOF'
version: '3.8'

services:
  neo4j:
    image: neo4j:5.15.0
    container_name: superinsight-neo4j
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    environment:
      - NEO4J_AUTH=neo4j/superinsight_kg
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_dbms_memory_heap_initial__size=512m
      - NEO4J_dbms_memory_heap_max__size=2G
      - NEO4J_dbms_memory_pagecache_size=512m
    volumes:
      - neo4j-data:/data
      - neo4j-logs:/logs
      - neo4j-import:/var/lib/neo4j/import
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7474"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - kg-network

  redis:
    image: redis:7-alpine
    container_name: superinsight-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - kg-network

volumes:
  neo4j-data:
  neo4j-logs:
  neo4j-import:
  redis-data:

networks:
  kg-network:
    driver: bridge
EOF

    log_success "Docker Compose file created: docker-compose.kg.yml"
}

# =============================================================================
# Configuration Setup
# =============================================================================

setup_configuration() {
    log_info "Setting up configuration files..."

    # Create config directory if needed
    mkdir -p "$CONFIG_DIR"

    # Copy example config if main config doesn't exist
    if [ -f "$CONFIG_DIR/knowledge_graph.example.yaml" ] && [ ! -f "$CONFIG_DIR/knowledge_graph.yaml" ]; then
        cp "$CONFIG_DIR/knowledge_graph.example.yaml" "$CONFIG_DIR/knowledge_graph.yaml"

        # Update password in config
        sed -i "s/your_password_here/$NEO4J_PASSWORD/g" "$CONFIG_DIR/knowledge_graph.yaml"

        log_success "Configuration file created: $CONFIG_DIR/knowledge_graph.yaml"
    else
        log_info "Configuration file already exists or example not found"
    fi

    # Create .env file for environment variables
    if [ ! -f "$PROJECT_ROOT/.env.kg" ]; then
        cat > "$PROJECT_ROOT/.env.kg" << EOF
# Knowledge Graph Environment Variables
# Generated by setup_knowledge_graph.sh

# Neo4j Configuration
NEO4J_URI=bolt://localhost:$NEO4J_PORT
NEO4J_USER=neo4j
NEO4J_PASSWORD=$NEO4J_PASSWORD
NEO4J_DATABASE=neo4j

# Redis Configuration (for caching)
REDIS_URL=redis://localhost:6379/0

# API Configuration
KG_API_HOST=0.0.0.0
KG_API_PORT=8080

# Security
JWT_SECRET_KEY=$(openssl rand -hex 32 2>/dev/null || echo "change-this-secret-key-in-production")

# Logging
LOG_LEVEL=INFO

# Feature Flags
ENABLE_GRAPHQL=true
ENABLE_WEBSOCKET=true
EOF
        log_success "Environment file created: .env.kg"
    else
        log_info "Environment file already exists"
    fi
}

# =============================================================================
# Database Initialization
# =============================================================================

initialize_database() {
    log_info "Initializing Neo4j database schema..."

    source "$VENV_DIR/bin/activate"

    # Create initialization script
    python3 << 'PYTHON_SCRIPT'
import os
import sys

try:
    from neo4j import GraphDatabase
except ImportError:
    print("Neo4j driver not installed. Skipping database initialization.")
    sys.exit(0)

uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
user = os.getenv("NEO4J_USER", "neo4j")
password = os.getenv("NEO4J_PASSWORD", "superinsight_kg")

try:
    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        # Create constraints for entity uniqueness
        constraints = [
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT organization_id IF NOT EXISTS FOR (o:Organization) REQUIRE o.id IS UNIQUE",
            "CREATE CONSTRAINT location_id IF NOT EXISTS FOR (l:Location) REQUIRE l.id IS UNIQUE",
        ]

        for constraint in constraints:
            try:
                session.run(constraint)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    print(f"Warning: {e}")

        # Create indexes for common queries
        indexes = [
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name)",
            "CREATE INDEX organization_name IF NOT EXISTS FOR (o:Organization) ON (o.name)",
        ]

        for index in indexes:
            try:
                session.run(index)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    print(f"Warning: {e}")

        print("Database schema initialized successfully!")

    driver.close()

except Exception as e:
    print(f"Could not connect to Neo4j: {e}")
    print("Make sure Neo4j is running and credentials are correct.")
    sys.exit(1)
PYTHON_SCRIPT

    log_success "Database initialization complete"
}

# =============================================================================
# Verification
# =============================================================================

verify_installation() {
    log_info "Verifying installation..."

    source "$VENV_DIR/bin/activate"

    local all_good=true

    # Check Python packages
    python3 -c "import neo4j" 2>/dev/null && log_success "neo4j package: OK" || { log_error "neo4j package: MISSING"; all_good=false; }
    python3 -c "import spacy" 2>/dev/null && log_success "spacy package: OK" || { log_error "spacy package: MISSING"; all_good=false; }
    python3 -c "import fastapi" 2>/dev/null && log_success "fastapi package: OK" || { log_error "fastapi package: MISSING"; all_good=false; }
    python3 -c "import pydantic" 2>/dev/null && log_success "pydantic package: OK" || { log_error "pydantic package: MISSING"; all_good=false; }

    # Check Neo4j connection
    if [ "$DOCKER_AVAILABLE" = "true" ]; then
        if docker ps --format '{{.Names}}' | grep -q "^superinsight-neo4j$"; then
            log_success "Neo4j container: RUNNING"
        else
            log_warning "Neo4j container: NOT RUNNING"
        fi
    fi

    # Check spaCy model
    python3 -c "import spacy; spacy.load('en_core_web_lg')" 2>/dev/null && log_success "spaCy model (lg): OK" || {
        python3 -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null && log_success "spaCy model (sm): OK" || log_warning "spaCy model: NOT LOADED"
    }

    if [ "$all_good" = true ]; then
        log_success "All verifications passed!"
    else
        log_warning "Some components may need attention"
    fi
}

# =============================================================================
# Usage Information
# =============================================================================

print_usage() {
    echo ""
    echo "Knowledge Graph Setup Script"
    echo "============================"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  full          Complete setup (default)"
    echo "  python        Setup Python environment only"
    echo "  neo4j         Setup Neo4j only"
    echo "  config        Setup configuration only"
    echo "  verify        Verify installation"
    echo "  clean         Remove Docker containers and volumes"
    echo "  help          Show this help message"
    echo ""
}

print_next_steps() {
    echo ""
    echo "==========================================="
    echo "  Knowledge Graph Setup Complete!"
    echo "==========================================="
    echo ""
    echo "Next Steps:"
    echo ""
    echo "1. Activate the virtual environment:"
    echo "   source $VENV_DIR/bin/activate"
    echo ""
    echo "2. Load environment variables:"
    echo "   source $PROJECT_ROOT/.env.kg"
    echo ""
    echo "3. Start the Knowledge Graph API:"
    echo "   python -m uvicorn src.knowledge_graph.api.main:app --reload"
    echo ""
    echo "4. Access Neo4j Browser:"
    echo "   http://localhost:$NEO4J_HTTP_PORT"
    echo ""
    echo "5. Run tests:"
    echo "   pytest tests/knowledge_graph/ -v"
    echo ""
    echo "Documentation:"
    echo "  - API Reference: docs/knowledge-graph/API_REFERENCE.md"
    echo "  - Configuration: config/knowledge_graph.example.yaml"
    echo ""
}

clean_docker() {
    log_info "Cleaning up Docker resources..."

    # Stop and remove container
    docker stop superinsight-neo4j 2>/dev/null || true
    docker rm superinsight-neo4j 2>/dev/null || true

    # Remove volumes (optional, ask first)
    read -p "Remove Neo4j data volumes? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker volume rm neo4j-data neo4j-logs 2>/dev/null || true
        log_success "Volumes removed"
    fi

    log_success "Docker cleanup complete"
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    local command="${1:-full}"

    echo ""
    echo "==========================================="
    echo "  SuperInsight Knowledge Graph Setup"
    echo "==========================================="
    echo ""

    case "$command" in
        full)
            check_prerequisites
            setup_python_environment
            install_python_dependencies
            install_spacy_model
            setup_neo4j_docker
            create_docker_compose
            setup_configuration
            initialize_database
            verify_installation
            print_next_steps
            ;;
        python)
            check_prerequisites
            setup_python_environment
            install_python_dependencies
            install_spacy_model
            ;;
        neo4j)
            check_prerequisites
            setup_neo4j_docker
            create_docker_compose
            ;;
        config)
            setup_configuration
            ;;
        verify)
            check_prerequisites
            verify_installation
            ;;
        clean)
            clean_docker
            ;;
        help|--help|-h)
            print_usage
            ;;
        *)
            log_error "Unknown command: $command"
            print_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
