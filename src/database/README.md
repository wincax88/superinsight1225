# SuperInsight Platform Database Setup

This directory contains the database configuration, models, and migration tools for the SuperInsight Platform.

## Overview

The SuperInsight Platform uses PostgreSQL as its primary database with the following key features:

- **JSONB Support**: Extensive use of JSONB columns for flexible data storage
- **GIN Indexes**: Optimized JSONB queries using GIN indexes
- **UUID Primary Keys**: All tables use UUID primary keys for better scalability
- **Alembic Migrations**: Database schema versioning and migration management
- **SQLAlchemy ORM**: Object-relational mapping for Python

## Database Schema

### Core Tables

1. **documents** - Stores source documents from various data sources
   - `id` (UUID): Primary key
   - `source_type` (VARCHAR): Type of data source (database, file, api)
   - `source_config` (JSONB): Configuration for data source connection
   - `content` (TEXT): Document content
   - `metadata` (JSONB): Additional metadata
   - `created_at`, `updated_at` (TIMESTAMP): Audit timestamps

2. **tasks** - Manages annotation tasks
   - `id` (UUID): Primary key
   - `document_id` (UUID): Foreign key to documents table
   - `project_id` (VARCHAR): Label Studio project identifier
   - `status` (ENUM): Task status (pending, in_progress, completed, reviewed)
   - `annotations` (JSONB): Annotation results
   - `ai_predictions` (JSONB): AI pre-annotation results
   - `quality_score` (FLOAT): Quality assessment score
   - `created_at` (TIMESTAMP): Creation timestamp

3. **billing_records** - Tracks annotation costs and usage
   - `id` (UUID): Primary key
   - `tenant_id` (VARCHAR): Tenant identifier for multi-tenancy
   - `user_id` (VARCHAR): User identifier
   - `task_id` (UUID): Optional foreign key to tasks table
   - `annotation_count` (INTEGER): Number of annotations
   - `time_spent` (INTEGER): Time spent in seconds
   - `cost` (FLOAT): Calculated cost
   - `billing_date` (DATE): Billing date
   - `created_at` (TIMESTAMP): Creation timestamp

4. **quality_issues** - Manages quality problems and work orders
   - `id` (UUID): Primary key
   - `task_id` (UUID): Foreign key to tasks table
   - `issue_type` (VARCHAR): Type of quality issue
   - `description` (TEXT): Detailed description
   - `severity` (ENUM): Issue severity (low, medium, high, critical)
   - `status` (ENUM): Issue status (open, in_progress, resolved, closed)
   - `assignee_id` (VARCHAR): Assigned user ID
   - `created_at`, `resolved_at` (TIMESTAMP): Timestamps

### Indexes

The database includes optimized indexes for performance:

#### GIN Indexes (for JSONB columns)
- `idx_documents_metadata_gin` - Documents metadata queries
- `idx_documents_source_config_gin` - Documents source config queries
- `idx_tasks_annotations_gin` - Task annotations queries
- `idx_tasks_ai_predictions_gin` - AI predictions queries

#### Regular Indexes
- `idx_tasks_document_id` - Task-document relationships
- `idx_tasks_project_id` - Project-based task queries
- `idx_tasks_status` - Status-based filtering
- `idx_billing_records_tenant_id` - Tenant-based billing queries
- `idx_billing_records_user_id` - User-based billing queries
- `idx_billing_records_billing_date` - Date-based billing queries
- `idx_quality_issues_task_id` - Task-issue relationships
- `idx_quality_issues_status` - Status-based issue filtering
- `idx_quality_issues_assignee_id` - Assignee-based queries

## Setup Instructions

### 1. Database Installation

Install PostgreSQL 12+ with JSONB and GIN index support:

```bash
# macOS (using Homebrew)
brew install postgresql
brew services start postgresql

# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# CentOS/RHEL
sudo yum install postgresql-server postgresql-contrib
```

### 2. Database Creation

Create the database and user:

```bash
# Connect to PostgreSQL as superuser
psql -U postgres

# Create database and user
CREATE DATABASE superinsight;
CREATE USER superinsight WITH PASSWORD 'your_password_here';
GRANT ALL PRIVILEGES ON DATABASE superinsight TO superinsight;
\q
```

### 3. Environment Configuration

Configure your database connection in `.env`:

```bash
DATABASE_URL=postgresql://superinsight:your_password_here@localhost:5432/superinsight
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=superinsight
DATABASE_USER=superinsight
DATABASE_PASSWORD=your_password_here
```

### 4. Database Initialization

Choose one of the following methods:

#### Method A: Using Alembic Migrations (Recommended)

```bash
# Run the initial migration
python3 -m alembic upgrade head

# Or use the migration script
python3 scripts/run_migrations.py upgrade
```

#### Method B: Using Python Initialization Script

```bash
# Run the database initialization script
python3 src/database/init_db.py
```

#### Method C: Using SQL Script (Development Only)

```bash
# Run the SQL initialization script
psql -U superinsight -d superinsight -f scripts/init-db.sql
```

### 5. Verification

Test the database setup:

```bash
# Run database tests
python3 tests/test_database_setup.py

# Or use pytest
pytest tests/test_database_setup.py -v
```

## Usage Examples

### Using the Database Manager

```python
from src.database.manager import database_manager

# Create a document
document = database_manager.create_document(
    source_type="database",
    source_config={"host": "localhost", "database": "customer_db"},
    content="Sample document content",
    metadata={"language": "zh-CN", "category": "feedback"}
)

# Create a task
task = database_manager.create_task(
    document_id=document.id,
    project_id="sentiment_analysis"
)

# Search documents by metadata
results = database_manager.search_documents_by_metadata({
    "category": "feedback",
    "language": "zh-CN"
})

# Calculate billing costs
costs = database_manager.calculate_tenant_costs("tenant_123")
print(f"Total cost: {costs['total_cost']}")
```

### Using SQLAlchemy ORM Directly

```python
from src.database.connection import db_manager
from src.database.models import DocumentModel, TaskModel

with db_manager.get_session() as session:
    # Query documents
    documents = session.query(DocumentModel).filter(
        DocumentModel.source_type == "database"
    ).all()
    
    # Complex JSONB query
    chinese_docs = session.query(DocumentModel).filter(
        DocumentModel.document_metadata['language'].astext == 'zh-CN'
    ).all()
```

## Migration Management

### Creating New Migrations

```bash
# Auto-generate migration from model changes
python3 -m alembic revision --autogenerate -m "Add new column to tasks table"

# Create empty migration for manual changes
python3 -m alembic revision -m "Add custom indexes"
```

### Running Migrations

```bash
# Upgrade to latest
python3 -m alembic upgrade head

# Upgrade to specific revision
python3 -m alembic upgrade abc123

# Downgrade to previous revision
python3 -m alembic downgrade -1

# Show current revision
python3 -m alembic current

# Show migration history
python3 -m alembic history
```

### Using the Migration Script

```bash
# Upgrade database
python3 scripts/run_migrations.py upgrade

# Show current status
python3 scripts/run_migrations.py current

# Create new migration
python3 scripts/run_migrations.py create "Add new feature" --autogenerate
```

## Performance Optimization

### JSONB Query Optimization

The database uses GIN indexes for JSONB columns. Use these query patterns for optimal performance:

```python
# Good: Uses GIN index
documents = session.query(DocumentModel).filter(
    DocumentModel.document_metadata.contains({"language": "zh-CN"})
).all()

# Good: Uses GIN index with operator
documents = session.query(DocumentModel).filter(
    DocumentModel.document_metadata['category'].astext == 'feedback'
).all()

# Avoid: Full table scan
documents = session.query(DocumentModel).filter(
    DocumentModel.document_metadata.cast(String).like('%feedback%')
).all()
```

### Connection Pooling

The database connection uses SQLAlchemy's connection pooling:

```python
# Configuration in settings.py
DATABASE_POOL_SIZE=10          # Number of connections to maintain
DATABASE_MAX_OVERFLOW=20       # Additional connections when needed
DATABASE_POOL_TIMEOUT=30       # Timeout for getting connection
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   ```bash
   # Check if PostgreSQL is running
   brew services list | grep postgresql  # macOS
   sudo systemctl status postgresql      # Linux
   ```

2. **Permission Denied**
   ```bash
   # Grant proper permissions
   psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE superinsight TO superinsight;"
   ```

3. **Migration Conflicts**
   ```bash
   # Reset to clean state (CAUTION: This will lose data)
   python3 -m alembic downgrade base
   python3 -m alembic upgrade head
   ```

4. **JSONB Index Not Used**
   ```sql
   -- Check if GIN indexes exist
   SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'documents';
   
   -- Analyze query performance
   EXPLAIN ANALYZE SELECT * FROM documents WHERE metadata @> '{"language": "zh-CN"}';
   ```

### Health Checks

```python
from src.database.manager import database_manager

# Check database health
health = database_manager.check_database_health()
print(f"Connection: {health['connection']}")
print(f"Tables exist: {health['tables_exist']}")
print(f"Indexes exist: {health['indexes_exist']}")

# Get database statistics
stats = database_manager.get_database_stats()
print(f"Documents: {stats['documents_count']}")
print(f"Tasks: {stats['tasks_count']}")
```

## Development vs Production

### Development Setup
- Use local PostgreSQL instance
- Enable SQL query logging (`echo=True` in engine config)
- Use simple passwords
- Run initialization scripts

### Production Setup
- Use managed PostgreSQL service (AWS RDS, Google Cloud SQL, etc.)
- Disable SQL query logging
- Use strong passwords and connection encryption
- Use environment variables for all configuration
- Set up proper backup and monitoring
- Configure connection pooling appropriately

## Files Reference

- `connection.py` - Database connection management
- `models.py` - SQLAlchemy ORM models
- `manager.py` - High-level database operations
- `init_db.py` - Database initialization script
- `../scripts/run_migrations.py` - Migration management script
- `../scripts/init-db.sql` - SQL initialization script
- `../../alembic/` - Alembic migration files
- `../../alembic.ini` - Alembic configuration