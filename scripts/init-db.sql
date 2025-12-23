-- SuperInsight Platform Database Initialization Script
-- This script creates the basic database structure and is compatible with Alembic migrations

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable JSONB GIN indexing extension
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create database user if not exists (for development)
DO $
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'superinsight') THEN
        CREATE ROLE superinsight WITH LOGIN PASSWORD 'password';
    END IF;
END
$;

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE superinsight TO superinsight;
GRANT ALL ON SCHEMA public TO superinsight;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO superinsight;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO superinsight;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO superinsight;

-- Create Alembic version table if it doesn't exist
CREATE TABLE IF NOT EXISTS alembic_version (
    version_num VARCHAR(32) NOT NULL,
    CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);

-- Note: The actual table creation is handled by Alembic migrations
-- This script only sets up the database environment and permissions

-- Grant permissions on all existing tables
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO superinsight;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO superinsight;