"""
Database connection management for SuperInsight Platform
"""
import logging
from typing import Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager

from src.config.settings import settings

logger = logging.getLogger(__name__)

# SQLAlchemy 2.0 base class for models
class Base(DeclarativeBase):
    pass


class DatabaseManager:
    """Database connection and session management"""
    
    def __init__(self):
        self._engine: Optional[object] = None
        self._session_factory: Optional[sessionmaker] = None
    
    def initialize(self) -> None:
        """Initialize database connection and session factory"""
        try:
            # Check if using SQLite
            is_sqlite = settings.database.database_url.startswith('sqlite')
            
            if is_sqlite:
                # SQLite configuration
                self._engine = create_engine(
                    settings.database.database_url,
                    echo=settings.app.debug,  # Log SQL queries in debug mode
                    connect_args={"check_same_thread": False}  # Allow SQLite to be used with multiple threads
                )
            else:
                # PostgreSQL/MySQL configuration with connection pooling
                self._engine = create_engine(
                    settings.database.database_url,
                    poolclass=QueuePool,
                    pool_size=settings.database.database_pool_size,
                    max_overflow=settings.database.database_max_overflow,
                    pool_timeout=settings.database.database_pool_timeout,
                    pool_pre_ping=True,  # Validate connections before use
                    echo=settings.app.debug,  # Log SQL queries in debug mode
                )
            
            # Create session factory
            self._session_factory = sessionmaker(
                bind=self._engine,
                expire_on_commit=False
            )
            
            logger.info("Database connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            raise
    
    def get_engine(self):
        """Get the database engine"""
        if self._engine is None:
            self.initialize()
        return self._engine
    
    def get_session_factory(self) -> sessionmaker:
        """Get the session factory"""
        if self._session_factory is None:
            self.initialize()
        return self._session_factory
    
    @contextmanager
    def get_session(self) -> Session:
        """Get a database session with automatic cleanup"""
        if self._session_factory is None:
            self.initialize()
        
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.get_session() as session:
                # Execute a simple query to test connection
                result = session.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def close(self) -> None:
        """Close database connections"""
        if self._engine:
            self._engine.dispose()
            logger.info("Database connections closed")


# Global database manager instance
db_manager = DatabaseManager()


def get_db_session() -> Session:
    """Dependency function to get database session for FastAPI"""
    with db_manager.get_session() as session:
        yield session


def init_database() -> None:
    """Initialize database connection"""
    db_manager.initialize()


def test_database_connection() -> bool:
    """Test database connection"""
    return db_manager.test_connection()


def get_database_stats() -> dict:
    """Get database statistics and connection info"""
    try:
        with db_manager.get_session() as session:
            # Get basic database info
            stats = {}
            
            # Check if using SQLite
            is_sqlite = settings.database.database_url.startswith('sqlite')
            
            if is_sqlite:
                # SQLite specific queries
                try:
                    version_result = session.execute(text("SELECT sqlite_version()"))
                    stats["version"] = f"SQLite {version_result.scalar()}"
                except Exception:
                    stats["version"] = "SQLite unknown"
                
                try:
                    table_result = session.execute(text(
                        "SELECT count(*) FROM sqlite_master WHERE type='table'"
                    ))
                    stats["table_count"] = table_result.scalar()
                except Exception:
                    stats["table_count"] = "unknown"
                
                stats["active_connections"] = "N/A (SQLite)"
                stats["database_size"] = "N/A (SQLite)"
            else:
                # PostgreSQL specific queries
                try:
                    version_result = session.execute(text("SELECT version()"))
                    stats["version"] = version_result.scalar()
                except Exception:
                    stats["version"] = "unknown"
                
                # Connection count (PostgreSQL specific)
                try:
                    conn_result = session.execute(text(
                        "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
                    ))
                    stats["active_connections"] = conn_result.scalar()
                except Exception:
                    stats["active_connections"] = "unknown"
                
                # Database size (PostgreSQL specific)
                try:
                    size_result = session.execute(text(
                        f"SELECT pg_size_pretty(pg_database_size('{settings.database.database_name}'))"
                    ))
                    stats["database_size"] = size_result.scalar()
                except Exception:
                    stats["database_size"] = "unknown"
                
                # Table count
                try:
                    table_result = session.execute(text(
                        "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public'"
                    ))
                    stats["table_count"] = table_result.scalar()
                except Exception:
                    stats["table_count"] = "unknown"
            
            return stats
            
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {"error": str(e)}


def close_database() -> None:
    """Close database connections"""
    db_manager.close()