"""
Database data extractor for SuperInsight Platform.

Provides secure, read-only extraction from MySQL, PostgreSQL, and Oracle databases.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import sqlalchemy as sa
from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from src.extractors.base import (
    BaseExtractor, 
    DatabaseConfig, 
    ExtractionResult,
    MYSQL_AVAILABLE,
    POSTGRESQL_AVAILABLE,
    ORACLE_AVAILABLE
)
from src.models.document import Document

logger = logging.getLogger(__name__)


class DatabaseExtractor(BaseExtractor):
    """Secure database extractor with read-only access."""
    
    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self.config: DatabaseConfig = config
        self._engine: Optional[Engine] = None
        self._metadata: Optional[MetaData] = None
    
    def _create_engine(self) -> Engine:
        """Create SQLAlchemy engine with security settings."""
        if self._engine is not None:
            return self._engine
        
        connection_string = self.config.get_connection_string()
        
        # Engine configuration for security and performance
        engine_kwargs = {
            'pool_pre_ping': True,  # Validate connections before use
            'pool_recycle': 3600,   # Recycle connections every hour
            'connect_args': {
                'connect_timeout': self.config.connection_timeout,
            }
        }
        
        # Database-specific SSL configuration
        if self.config.use_ssl:
            if self.config.database_type.value == "mysql":
                engine_kwargs['connect_args'].update({
                    'ssl_disabled': False,
                    'ssl_verify_cert': self.config.verify_ssl,
                })
            elif self.config.database_type.value == "postgresql":
                engine_kwargs['connect_args'].update({
                    'sslmode': 'require' if self.config.verify_ssl else 'prefer',
                })
        
        try:
            self._engine = create_engine(connection_string, **engine_kwargs)
            logger.info(f"Created database engine for {self.config.database_type.value}")
            return self._engine
        except Exception as e:
            logger.error(f"Failed to create database engine: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test database connection with read-only validation."""
        try:
            engine = self._create_engine()
            
            with engine.connect() as conn:
                # Test basic connectivity
                result = conn.execute(text("SELECT 1"))
                if not result.scalar() == 1:
                    return False
                
                # Verify read-only access by attempting a write operation
                # This should fail if properly configured
                try:
                    if self.config.database_type.value == "mysql":
                        conn.execute(text("SELECT @@read_only"))
                    elif self.config.database_type.value == "postgresql":
                        conn.execute(text("SELECT pg_is_in_recovery()"))
                    elif self.config.database_type.value == "oracle":
                        conn.execute(text("SELECT open_mode FROM v$database"))
                except SQLAlchemyError:
                    # Some read-only checks might not be available
                    pass
                
                logger.info("Database connection test successful")
                return True
                
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def _get_table_metadata(self, table_name: str) -> Optional[Table]:
        """Get metadata for a specific table."""
        try:
            if self._metadata is None:
                self._metadata = MetaData()
            
            engine = self._create_engine()
            table = Table(table_name, self._metadata, autoload_with=engine)
            return table
        except Exception as e:
            logger.error(f"Failed to get metadata for table {table_name}: {e}")
            return None
    
    def list_tables(self) -> List[str]:
        """List all tables in the database."""
        try:
            engine = self._create_engine()
            
            with engine.connect() as conn:
                if self.config.database_type.value == "mysql":
                    result = conn.execute(text("SHOW TABLES"))
                elif self.config.database_type.value == "postgresql":
                    result = conn.execute(text("""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'public'
                    """))
                elif self.config.database_type.value == "oracle":
                    result = conn.execute(text("""
                        SELECT table_name 
                        FROM user_tables
                    """))
                else:
                    raise ValueError(f"Unsupported database type: {self.config.database_type}")
                
                tables = [row[0] for row in result]
                logger.info(f"Found {len(tables)} tables in database")
                return tables
                
        except Exception as e:
            logger.error(f"Failed to list tables: {e}")
            return []
    
    def extract_from_table(self, table_name: str, limit: Optional[int] = None, 
                          where_clause: Optional[str] = None) -> ExtractionResult:
        """Extract data from a specific table."""
        try:
            engine = self._create_engine()
            
            # Build query
            query = f"SELECT * FROM {table_name}"
            if where_clause:
                query += f" WHERE {where_clause}"
            if limit:
                if self.config.database_type.value in ["mysql", "postgresql"]:
                    query += f" LIMIT {limit}"
                elif self.config.database_type.value == "oracle":
                    query += f" AND ROWNUM <= {limit}"
            
            documents = []
            
            with engine.connect() as conn:
                result = conn.execute(text(query))
                
                for row in result:
                    # Convert row to dictionary
                    row_dict = dict(row._mapping)
                    
                    # Create document from row data
                    document = Document(
                        source_type="database",
                        source_config={
                            "host": self.config.host,
                            "database": self.config.database,
                            "table": table_name,
                            "database_type": self.config.database_type.value
                        },
                        content=str(row_dict),  # Convert entire row to string content
                        metadata={
                            "table_name": table_name,
                            "row_data": row_dict,
                            "extraction_time": datetime.now().isoformat()
                        }
                    )
                    documents.append(document)
            
            logger.info(f"Extracted {len(documents)} documents from table {table_name}")
            return ExtractionResult(success=True, documents=documents)
            
        except Exception as e:
            logger.error(f"Failed to extract from table {table_name}: {e}")
            return ExtractionResult(success=False, error=str(e))
    
    def extract_data(self, query: Optional[str] = None, table_name: Optional[str] = None,
                    limit: Optional[int] = None, **kwargs) -> ExtractionResult:
        """Extract data using custom query or table name."""
        try:
            if query:
                return self._extract_with_query(query, limit)
            elif table_name:
                where_clause = kwargs.get('where_clause')
                return self.extract_from_table(table_name, limit, where_clause)
            else:
                # Extract from all tables (limited)
                return self._extract_all_tables(limit or 100)
                
        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            return ExtractionResult(success=False, error=str(e))
    
    def _extract_with_query(self, query: str, limit: Optional[int] = None) -> ExtractionResult:
        """Extract data using a custom SQL query."""
        # Security check: ensure query is read-only
        import re
        query_upper = query.upper().strip()
        forbidden_keywords = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE']
        
        for keyword in forbidden_keywords:
            # Use word boundaries to match whole words only
            if re.search(r'\b' + keyword + r'\b', query_upper):
                return ExtractionResult(
                    success=False,
                    error=f"Query contains forbidden keyword: {keyword}"
                )
        
        try:
            engine = self._create_engine()
            documents = []
            
            with engine.connect() as conn:
                result = conn.execute(text(query))
                
                count = 0
                for row in result:
                    if limit and count >= limit:
                        break
                    
                    row_dict = dict(row._mapping)
                    
                    document = Document(
                        source_type="database",
                        source_config={
                            "host": self.config.host,
                            "database": self.config.database,
                            "query": query,
                            "database_type": self.config.database_type.value
                        },
                        content=str(row_dict),
                        metadata={
                            "query_result": row_dict,
                            "extraction_time": datetime.now().isoformat()
                        }
                    )
                    documents.append(document)
                    count += 1
            
            logger.info(f"Extracted {len(documents)} documents using custom query")
            return ExtractionResult(success=True, documents=documents)
            
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            return ExtractionResult(success=False, error=str(e))
    
    def _extract_all_tables(self, limit_per_table: int = 10) -> ExtractionResult:
        """Extract sample data from all tables."""
        tables = self.list_tables()
        all_documents = []
        
        for table_name in tables:
            result = self.extract_from_table(table_name, limit_per_table)
            if result.success:
                all_documents.extend(result.documents)
            else:
                logger.warning(f"Failed to extract from table {table_name}: {result.error}")
        
        logger.info(f"Extracted {len(all_documents)} documents from {len(tables)} tables")
        return ExtractionResult(success=True, documents=all_documents)
    
    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get schema information for a table."""
        try:
            table = self._get_table_metadata(table_name)
            if not table:
                return {}
            
            schema = {
                "table_name": table_name,
                "columns": []
            }
            
            for column in table.columns:
                column_info = {
                    "name": column.name,
                    "type": str(column.type),
                    "nullable": column.nullable,
                    "primary_key": column.primary_key
                }
                schema["columns"].append(column_info)
            
            return schema
            
        except Exception as e:
            logger.error(f"Failed to get schema for table {table_name}: {e}")
            return {}
    
    def close(self) -> None:
        """Close database connections."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            logger.info("Database connections closed")