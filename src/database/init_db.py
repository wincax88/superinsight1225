"""
Database initialization script for SuperInsight Platform.

This script initializes the database connection and creates all tables.
"""

import logging
from sqlalchemy import text, select, func

from src.database.connection import db_manager, Base
from src.database.models import DocumentModel, TaskModel, BillingRecordModel, QualityIssueModel

logger = logging.getLogger(__name__)


def create_database_tables():
    """
    Create all database tables using SQLAlchemy.
    
    This is an alternative to using Alembic migrations for development/testing.
    """
    try:
        # Initialize database connection
        db_manager.initialize()
        engine = db_manager.get_engine()
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        logger.info("Database tables created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        return False


def test_database_setup():
    """
    Test the database setup by performing basic operations.
    """
    try:
        # Test database connection
        if not db_manager.test_connection():
            logger.error("Database connection test failed")
            return False
        
        # Test table creation
        with db_manager.get_session() as session:
            # Test that we can query each table (they should be empty initially)
            doc_count = session.execute(select(func.count(DocumentModel.id))).scalar()
            task_count = session.execute(select(func.count(TaskModel.id))).scalar()
            billing_count = session.execute(select(func.count(BillingRecordModel.id))).scalar()
            quality_count = session.execute(select(func.count(QualityIssueModel.id))).scalar()
            
            logger.info(f"Database test successful - Tables exist with counts: "
                       f"documents={doc_count}, tasks={task_count}, "
                       f"billing_records={billing_count}, quality_issues={quality_count}")
            return True
            
    except Exception as e:
        logger.error(f"Database setup test failed: {e}")
        return False


def create_gin_indexes():
    """
    Create GIN indexes for JSONB columns to optimize query performance.
    
    This function creates the indexes manually if they don't exist.
    """
    try:
        engine = db_manager.get_engine()
        
        with engine.connect() as connection:
            # Create GIN indexes for JSONB columns
            gin_indexes = [
                "CREATE INDEX IF NOT EXISTS idx_documents_metadata_gin ON documents USING GIN (metadata)",
                "CREATE INDEX IF NOT EXISTS idx_documents_source_config_gin ON documents USING GIN (source_config)",
                "CREATE INDEX IF NOT EXISTS idx_tasks_annotations_gin ON tasks USING GIN (annotations)",
                "CREATE INDEX IF NOT EXISTS idx_tasks_ai_predictions_gin ON tasks USING GIN (ai_predictions)",
            ]
            
            # Create additional performance indexes
            performance_indexes = [
                "CREATE INDEX IF NOT EXISTS idx_tasks_document_id ON tasks (document_id)",
                "CREATE INDEX IF NOT EXISTS idx_tasks_project_id ON tasks (project_id)",
                "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks (status)",
                "CREATE INDEX IF NOT EXISTS idx_billing_records_tenant_id ON billing_records (tenant_id)",
                "CREATE INDEX IF NOT EXISTS idx_billing_records_user_id ON billing_records (user_id)",
                "CREATE INDEX IF NOT EXISTS idx_billing_records_billing_date ON billing_records (billing_date)",
                "CREATE INDEX IF NOT EXISTS idx_quality_issues_task_id ON quality_issues (task_id)",
                "CREATE INDEX IF NOT EXISTS idx_quality_issues_status ON quality_issues (status)",
                "CREATE INDEX IF NOT EXISTS idx_quality_issues_assignee_id ON quality_issues (assignee_id)",
            ]
            
            all_indexes = gin_indexes + performance_indexes
            
            for index_sql in all_indexes:
                try:
                    connection.execute(text(index_sql))
                    logger.info(f"Created index: {index_sql.split('idx_')[1].split(' ')[0]}")
                except Exception as e:
                    logger.warning(f"Failed to create index {index_sql}: {e}")
            
            connection.commit()
            
        logger.info("Database indexes created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create database indexes: {e}")
        return False


def initialize_database():
    """
    Complete database initialization including tables and indexes.
    
    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    logger.info("Starting database initialization...")
    
    # Step 1: Create tables
    if not create_database_tables():
        return False
    
    # Step 2: Create indexes
    if not create_gin_indexes():
        return False
    
    # Step 3: Test setup
    if not test_database_setup():
        return False
    
    logger.info("Database initialization completed successfully")
    return True


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize database
    success = initialize_database()
    
    if success:
        print("✅ Database initialization successful!")
    else:
        print("❌ Database initialization failed!")
        exit(1)