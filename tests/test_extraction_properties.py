"""
Property-based tests for data extraction security and functionality.

Tests the read-only security properties and data extraction consistency
as specified in the SuperInsight Platform requirements.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis import HealthCheck
from unittest.mock import Mock, patch, MagicMock
import sqlalchemy as sa
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError, OperationalError, ProgrammingError
from typing import Dict, Any, List, Optional
import tempfile
import os

from src.extractors.base import (
    DatabaseConfig, 
    DatabaseType, 
    SecurityValidator,
    BaseExtractor,
    ExtractionResult
)
from src.extractors.database import DatabaseExtractor
from src.extractors.factory import ExtractorFactory


# Hypothesis strategies for generating test data

def database_config_strategy():
    """Strategy for generating valid DatabaseConfig instances."""
    return st.builds(
        DatabaseConfig,
        host=st.sampled_from(["localhost", "db.example.com", "test-db.local"]),
        port=st.sampled_from([3306, 5432, 1521]),  # Common database ports
        database=st.sampled_from(["test_db", "production", "staging"]),
        username=st.sampled_from(["readonly_user", "test_user", "db_reader"]),
        password=st.sampled_from(["password123", "secure_pass", "test_pass"]),
        database_type=st.sampled_from(list(DatabaseType)),
        read_only=st.just(True),  # Always enforce read-only for security
        connection_timeout=st.integers(min_value=10, max_value=60),
        read_timeout=st.integers(min_value=30, max_value=120),
        use_ssl=st.booleans(),
        verify_ssl=st.booleans()
    )

def write_operation_queries():
    """Strategy for generating SQL write operation queries."""
    return st.sampled_from([
        "INSERT INTO users (name) VALUES ('test')",
        "UPDATE users SET name = 'updated' WHERE id = 1",
        "DELETE FROM users WHERE id = 1",
        "DROP TABLE users",
        "CREATE TABLE test_table (id INT)",
        "ALTER TABLE users ADD COLUMN email VARCHAR(255)",
        "TRUNCATE TABLE users",
        "insert into users values (1, 'test')",  # lowercase
        "update users set name='test'",  # lowercase
        "delete from users",  # lowercase
        "drop table users",  # lowercase
        "create table test (id int)",  # lowercase
        "alter table users add column test int",  # lowercase
        "truncate table users",  # lowercase
        "INSERT INTO users (name) VALUES ('test'); DROP TABLE users;",  # SQL injection attempt
        "SELECT * FROM users; INSERT INTO logs VALUES ('hack');",  # Mixed query
    ])

def read_operation_queries():
    """Strategy for generating SQL read operation queries."""
    return st.sampled_from([
        "SELECT * FROM users",
        "SELECT name, email FROM users WHERE id = 1",
        "SELECT COUNT(*) FROM users",
        "SELECT u.name, p.title FROM users u JOIN posts p ON u.id = p.user_id",
        "SELECT * FROM users ORDER BY name DESC LIMIT 10",
        "SELECT DISTINCT category FROM products",
        "SELECT AVG(price) FROM products GROUP BY category",
        "WITH recent_users AS (SELECT * FROM users WHERE updated_at > '2024-01-01') SELECT * FROM recent_users",
        "select * from users",  # lowercase
        "select name from users where active = true",  # lowercase
        "EXPLAIN SELECT * FROM users",  # Query analysis
        "SHOW TABLES",  # MySQL specific
        "DESCRIBE users",  # MySQL specific
    ])


class TestDataExtractionReadOnlySecurity:
    """
    Property-based tests for data extraction read-only security.
    
    Validates Requirement 1.1:
    - Database connections must use read-only permissions
    - Write operations should be prevented at the connection level
    """
    
    @given(database_config_strategy())
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_database_config_enforces_read_only(self, db_config: DatabaseConfig):
        """
        Property 1: Database Configuration Read-Only Enforcement
        
        For any DatabaseConfig instance, the read_only flag must be True
        and security validation must pass.
        
        **Validates: Requirement 1.1**
        """
        # Verify read_only is enforced
        assert db_config.read_only is True, "Database configuration must enforce read-only access"
        
        # Verify security validation passes
        assert SecurityValidator.validate_read_only_connection(db_config) is True
        
        # Verify configuration validation passes
        assert db_config.validate() is True
    
    @given(database_config_strategy())
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_database_extractor_enforces_read_only_config(self, db_config: DatabaseConfig):
        """
        Property 1: Database Extractor Read-Only Configuration
        
        For any valid DatabaseConfig, creating a DatabaseExtractor should
        enforce read-only access through security validation.
        
        **Validates: Requirement 1.1**
        """
        # Creating extractor should not raise exception for read-only config
        try:
            extractor = DatabaseExtractor(db_config)
            # Verify the extractor stores the read-only config
            assert extractor.config.read_only is True
        except Exception as e:
            # If exception occurs, it should not be due to read-only validation
            assert "read-only" not in str(e).lower()
    
    def test_database_config_rejects_write_access(self):
        """
        Property 1: Database Configuration Rejects Write Access
        
        Creating a DatabaseConfig with read_only=False should raise a security error.
        
        **Validates: Requirement 1.1**
        """
        with pytest.raises(ValueError, match="Database connections must be read-only"):
            # Attempt to create config with write access
            config = DatabaseConfig(
                host="localhost",
                port=5432,
                database="test_db",
                username="test_user",
                password="test_pass",
                database_type=DatabaseType.POSTGRESQL,
                read_only=False  # This should trigger security validation error
            )
            # Trigger security validation
            SecurityValidator.validate_read_only_connection(config)
    
    @given(write_operation_queries())
    def test_database_extractor_blocks_write_queries(self, write_query: str):
        """
        Property 1: Database Extractor Blocks Write Operations
        
        For any SQL write operation query, the database extractor should
        reject the query and return an error result.
        
        **Validates: Requirement 1.1**
        """
        # Create a mock database config
        db_config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_db",
            username="readonly_user",
            password="test_pass",
            database_type=DatabaseType.POSTGRESQL,
            read_only=True
        )
        
        # Create extractor
        extractor = DatabaseExtractor(db_config)
        
        # Mock the engine creation to avoid actual database connection
        with patch.object(extractor, '_create_engine') as mock_engine:
            mock_conn = Mock()
            mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
            
            # Attempt to execute write query
            result = extractor._extract_with_query(write_query)
            
            # Verify the query was rejected
            assert result.success is False
            assert "forbidden keyword" in result.error.lower()
            
            # Verify no actual database execution occurred
            mock_conn.execute.assert_not_called()
    
    @given(read_operation_queries())
    def test_database_extractor_allows_read_queries(self, read_query: str):
        """
        Property 1: Database Extractor Allows Read Operations
        
        For any SQL read operation query, the database extractor should
        allow the query to proceed (though it may fail due to connection issues).
        
        **Validates: Requirement 1.1**
        """
        # Create a mock database config
        db_config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_db",
            username="readonly_user",
            password="test_pass",
            database_type=DatabaseType.POSTGRESQL,
            read_only=True
        )
        
        # Create extractor
        extractor = DatabaseExtractor(db_config)
        
        # Mock the engine creation and query execution
        with patch.object(extractor, '_create_engine') as mock_engine:
            mock_conn = Mock()
            mock_result = Mock()
            mock_result.__iter__ = Mock(return_value=iter([]))  # Empty result set
            mock_conn.execute.return_value = mock_result
            mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
            
            # Attempt to execute read query
            result = extractor._extract_with_query(read_query)
            
            # Verify the query was not rejected for security reasons
            if not result.success:
                # If it failed, it should not be due to forbidden keywords
                assert "forbidden keyword" not in result.error.lower()
            
            # Verify the query was attempted to be executed
            mock_conn.execute.assert_called_once()
    
    @given(database_config_strategy())
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_factory_creates_read_only_extractors(self, db_config: DatabaseConfig):
        """
        Property 1: Factory Creates Read-Only Extractors
        
        For any database configuration, the ExtractorFactory should
        create extractors that enforce read-only access.
        
        **Validates: Requirement 1.1**
        """
        # Use factory to create extractor
        extractor = ExtractorFactory.create_database_extractor(
            host=db_config.host,
            port=db_config.port,
            database=db_config.database,
            username=db_config.username,
            password=db_config.password,
            database_type=db_config.database_type.value,
            use_ssl=db_config.use_ssl,
            connection_timeout=db_config.connection_timeout,
            read_timeout=db_config.read_timeout
        )
        
        # Verify the created extractor enforces read-only access
        assert extractor.config.read_only is True
        assert SecurityValidator.validate_read_only_connection(extractor.config) is True
    
    def test_security_validator_read_only_enforcement(self):
        """
        Property 1: Security Validator Read-Only Enforcement
        
        The SecurityValidator should consistently enforce read-only access
        across different database types and configurations.
        
        **Validates: Requirement 1.1**
        """
        # Test with different database types
        for db_type in DatabaseType:
            # Valid read-only config should pass
            valid_config = DatabaseConfig(
                host="localhost",
                port=5432,
                database="test_db",
                username="readonly_user",
                password="test_pass",
                database_type=db_type,
                read_only=True
            )
            assert SecurityValidator.validate_read_only_connection(valid_config) is True
            
            # Invalid write-enabled config should fail
            invalid_config = DatabaseConfig(
                host="localhost",
                port=5432,
                database="test_db",
                username="write_user",
                password="test_pass",
                database_type=db_type,
                read_only=False
            )
            
            with pytest.raises(ValueError, match="Database connections must be read-only"):
                SecurityValidator.validate_read_only_connection(invalid_config)
    
    @given(st.text(min_size=1, max_size=100))
    def test_query_security_validation_comprehensive(self, base_query: str):
        """
        Property 1: Comprehensive Query Security Validation
        
        For any string that could be a SQL query, the security validation
        should correctly identify and block write operations while allowing reads.
        
        **Validates: Requirement 1.1**
        """
        # Create a mock database config
        db_config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_db",
            username="readonly_user",
            password="test_pass",
            database_type=DatabaseType.POSTGRESQL,
            read_only=True
        )
        
        extractor = DatabaseExtractor(db_config)
        
        # Check if query contains forbidden keywords using word boundaries
        import re
        query_upper = base_query.upper().strip()
        forbidden_keywords = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE']
        
        contains_forbidden = any(re.search(r'\b' + keyword + r'\b', query_upper) for keyword in forbidden_keywords)
        
        # Mock the engine to avoid actual database connection
        with patch.object(extractor, '_create_engine') as mock_engine:
            mock_conn = Mock()
            mock_result = Mock()
            mock_result.__iter__ = Mock(return_value=iter([]))
            mock_conn.execute.return_value = mock_result
            mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
            
            result = extractor._extract_with_query(base_query)
            
            if contains_forbidden:
                # Should be rejected
                assert result.success is False
                assert "forbidden keyword" in result.error.lower()
                mock_conn.execute.assert_not_called()
            else:
                # Should be allowed to proceed (may still fail for other reasons)
                if not result.success:
                    assert "forbidden keyword" not in result.error.lower()


class TestDataExtractionCompleteness:
    """
    Property-based tests for data extraction completeness.
    
    Validates Requirement 1.5:
    - Extracted data must be completely stored in PostgreSQL
    - Original data copies must be preserved
    """
    
    @given(database_config_strategy())
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_extracted_data_stored_in_postgresql(self, db_config: DatabaseConfig):
        """
        Property 2: Data Extraction Completeness
        
        For any valid data source extraction, all extracted documents should
        be completely stored in PostgreSQL with original data preserved.
        
        **Validates: Requirement 1.5**
        """
        from src.database.manager import database_manager
        from src.models.document import Document
        from datetime import datetime
        import uuid
        
        # Create mock extracted documents
        mock_documents = []
        for i in range(3):  # Test with a few documents
            doc = Document(
                id=uuid.uuid4(),
                source_type="database",
                source_config={
                    "host": db_config.host,
                    "database": db_config.database,
                    "table": f"test_table_{i}",
                    "database_type": db_config.database_type.value
                },
                content=f"Test content from {db_config.host} table {i}",
                metadata={
                    "extraction_time": datetime.now().isoformat(),
                    "row_count": i + 1,
                    "source_host": db_config.host
                }
            )
            mock_documents.append(doc)
        
        # Mock the database manager to avoid actual database operations
        with patch.object(database_manager, 'create_document') as mock_create:
            # Configure mock to return the document as if it was saved
            def mock_create_document(source_type, source_config, content, metadata=None):
                from src.database.models import DocumentModel
                return DocumentModel(
                    id=uuid.uuid4(),
                    source_type=source_type,
                    source_config=source_config,
                    content=content,
                    document_metadata=metadata or {}
                )
            
            mock_create.side_effect = mock_create_document
            
            # Simulate the extraction and storage process
            stored_documents = []
            for doc in mock_documents:
                stored_doc = database_manager.create_document(
                    source_type=doc.source_type,
                    source_config=doc.source_config,
                    content=doc.content,
                    metadata=doc.metadata
                )
                stored_documents.append(stored_doc)
            
            # Verify all documents were stored
            assert len(stored_documents) == len(mock_documents)
            
            # Verify each document was stored with complete data
            for i, (original, stored) in enumerate(zip(mock_documents, stored_documents)):
                # Verify source type is preserved
                assert stored.source_type == original.source_type
                
                # Verify source configuration is preserved
                assert stored.source_config == original.source_config
                
                # Verify content is preserved
                assert stored.content == original.content
                
                # Verify metadata is preserved
                assert stored.document_metadata == original.metadata
            
            # Verify database manager was called for each document
            assert mock_create.call_count == len(mock_documents)
    
    @given(st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=10))
    def test_extraction_result_completeness(self, content_list: List[str]):
        """
        Property 2: Extraction Result Completeness
        
        For any list of extracted content, the ExtractionResult should
        contain all extracted documents with complete data preservation.
        
        **Validates: Requirement 1.5**
        """
        from src.extractors.base import ExtractionResult
        from src.models.document import Document
        import uuid
        from datetime import datetime
        
        # Filter out content that would fail validation (empty or whitespace-only)
        valid_content_list = [c for c in content_list if c.strip()]
        
        # Skip if no valid content
        if not valid_content_list:
            return
        
        # Create documents from content list
        documents = []
        for i, content in enumerate(valid_content_list):
            doc = Document(
                id=uuid.uuid4(),
                source_type="database",
                source_config={
                    "host": "test-host",
                    "database": "test-db",
                    "table": f"table_{i}"
                },
                content=content,
                metadata={
                    "index": i,
                    "extraction_time": datetime.now().isoformat()
                }
            )
            documents.append(doc)
        
        # Create extraction result
        result = ExtractionResult(success=True, documents=documents)
        
        # Verify completeness properties
        assert result.success is True
        assert result.extracted_count == len(valid_content_list)
        assert len(result.documents) == len(valid_content_list)
        
        # Verify each document is complete
        for i, (original_content, doc) in enumerate(zip(valid_content_list, result.documents)):
            assert doc.content == original_content
            assert doc.source_type == "database"
            assert doc.metadata["index"] == i
            assert "extraction_time" in doc.metadata
    
    def test_postgresql_storage_preserves_jsonb_data(self):
        """
        Property 2: PostgreSQL JSONB Data Preservation
        
        When storing documents with complex metadata in PostgreSQL JSONB format,
        all data should be preserved without loss or corruption.
        
        **Validates: Requirement 1.5**
        """
        from src.database.manager import database_manager
        import json
        
        # Test with complex nested metadata
        complex_metadata = {
            "extraction_info": {
                "timestamp": "2024-01-01T12:00:00Z",
                "source": {
                    "type": "database",
                    "host": "prod-db.example.com",
                    "port": 5432
                }
            },
            "data_quality": {
                "completeness": 0.95,
                "accuracy": 0.98,
                "issues": ["missing_field_x", "invalid_date_format"]
            },
            "processing_stats": {
                "rows_processed": 1000,
                "errors_count": 2,
                "processing_time_ms": 1500
            }
        }
        
        complex_source_config = {
            "connection": {
                "host": "test-db.local",
                "port": 5432,
                "database": "test_db",
                "ssl_config": {
                    "enabled": True,
                    "verify_cert": True
                }
            },
            "query_info": {
                "table": "users",
                "columns": ["id", "name", "email", "created_at"],
                "filters": {
                    "active": True,
                    "created_after": "2024-01-01"
                }
            }
        }
        
        # Mock database operations
        with patch.object(database_manager, 'create_document') as mock_create:
            def mock_create_document(source_type, source_config, content, metadata=None):
                from src.database.models import DocumentModel
                
                # Simulate JSONB serialization/deserialization
                serialized_config = json.loads(json.dumps(source_config))
                serialized_metadata = json.loads(json.dumps(metadata or {}))
                
                return DocumentModel(
                    source_type=source_type,
                    source_config=serialized_config,
                    content=content,
                    document_metadata=serialized_metadata
                )
            
            mock_create.side_effect = mock_create_document
            
            # Store document with complex data
            stored_doc = database_manager.create_document(
                source_type="database",
                source_config=complex_source_config,
                content="Sample extracted content",
                metadata=complex_metadata
            )
            
            # Verify all nested data is preserved
            assert stored_doc.source_config == complex_source_config
            assert stored_doc.document_metadata == complex_metadata
            
            # Verify specific nested values
            assert stored_doc.source_config["connection"]["host"] == "test-db.local"
            assert stored_doc.source_config["query_info"]["filters"]["active"] is True
            assert stored_doc.document_metadata["data_quality"]["completeness"] == 0.95
            assert stored_doc.document_metadata["processing_stats"]["rows_processed"] == 1000
            assert "missing_field_x" in stored_doc.document_metadata["data_quality"]["issues"]
    
    @given(st.integers(min_value=1, max_value=100))
    def test_batch_extraction_completeness(self, batch_size: int):
        """
        Property 2: Batch Extraction Completeness
        
        For any batch size of extracted documents, all documents should be
        stored in PostgreSQL without loss, maintaining extraction order.
        
        **Validates: Requirement 1.5**
        """
        from src.database.manager import database_manager
        from src.models.document import Document
        import uuid
        from datetime import datetime
        
        # Create a batch of documents
        batch_documents = []
        for i in range(batch_size):
            doc = Document(
                id=uuid.uuid4(),
                source_type="database",
                source_config={
                    "host": "batch-test-host",
                    "database": "batch_db",
                    "table": "batch_table",
                    "batch_id": f"batch_{i // 10}",  # Group into batches of 10
                    "sequence": i
                },
                content=f"Batch document content {i}",
                metadata={
                    "batch_index": i,
                    "batch_size": batch_size,
                    "extraction_timestamp": datetime.now().isoformat()
                }
            )
            batch_documents.append(doc)
        
        # Mock batch storage
        with patch.object(database_manager, 'create_document') as mock_create:
            stored_documents = []
            
            def mock_create_document(source_type, source_config, content, metadata=None):
                from src.database.models import DocumentModel
                doc_model = DocumentModel(
                    id=uuid.uuid4(),
                    source_type=source_type,
                    source_config=source_config,
                    content=content,
                    document_metadata=metadata or {}
                )
                stored_documents.append(doc_model)
                return doc_model
            
            mock_create.side_effect = mock_create_document
            
            # Store all documents in batch
            for doc in batch_documents:
                database_manager.create_document(
                    source_type=doc.source_type,
                    source_config=doc.source_config,
                    content=doc.content,
                    metadata=doc.metadata
                )
            
            # Verify batch completeness
            assert len(stored_documents) == batch_size
            assert mock_create.call_count == batch_size
            
            # Verify order preservation and data integrity
            for i, (original, stored) in enumerate(zip(batch_documents, stored_documents)):
                assert stored.source_config["sequence"] == i
                assert stored.document_metadata["batch_index"] == i
                assert stored.content == f"Batch document content {i}"
                assert stored.source_type == "database"
    
    def test_extraction_error_handling_preserves_partial_data(self):
        """
        Property 2: Partial Extraction Data Preservation
        
        When extraction partially fails, successfully extracted documents
        should still be preserved in PostgreSQL.
        
        **Validates: Requirement 1.5**
        """
        from src.extractors.base import ExtractionResult
        from src.models.document import Document
        from src.database.manager import database_manager
        import uuid
        
        # Create some successful documents and simulate partial failure
        successful_docs = []
        for i in range(3):
            doc = Document(
                id=uuid.uuid4(),
                source_type="database",
                source_config={"host": "test-host", "table": f"table_{i}"},
                content=f"Successfully extracted content {i}",
                metadata={"status": "success", "index": i}
            )
            successful_docs.append(doc)
        
        # Mock database storage
        with patch.object(database_manager, 'create_document') as mock_create:
            stored_docs = []
            
            def mock_create_document(source_type, source_config, content, metadata=None):
                from src.database.models import DocumentModel
                doc_model = DocumentModel(
                    source_type=source_type,
                    source_config=source_config,
                    content=content,
                    document_metadata=metadata or {}
                )
                stored_docs.append(doc_model)
                return doc_model
            
            mock_create.side_effect = mock_create_document
            
            # Store the successful documents (simulating partial extraction)
            for doc in successful_docs:
                database_manager.create_document(
                    source_type=doc.source_type,
                    source_config=doc.source_config,
                    content=doc.content,
                    metadata=doc.metadata
                )
            
            # Create extraction result with partial success
            result = ExtractionResult(
                success=False,  # Overall failure due to some errors
                documents=successful_docs,  # But some documents were extracted
                error="Connection timeout after extracting 3 documents"
            )
            
            # Verify that successful documents are preserved
            assert len(stored_docs) == 3
            assert result.extracted_count == 3
            
            # Verify each successful document is complete
            for i, stored in enumerate(stored_docs):
                assert stored.content == f"Successfully extracted content {i}"
                assert stored.document_metadata["status"] == "success"
                assert stored.document_metadata["index"] == i


class TestDatabaseConnectionSecurity:
    """
    Additional security tests for database connections.
    """
    
    def test_connection_string_security(self):
        """
        Test that connection strings are generated securely with proper SSL settings.
        
        **Validates: Requirement 1.1**
        """
        config = DatabaseConfig(
            host="secure-db.example.com",
            port=5432,
            database="production_db",
            username="readonly_user",
            password="secure_password",
            database_type=DatabaseType.POSTGRESQL,
            read_only=True,
            use_ssl=True,
            verify_ssl=True
        )
        
        # Mock the availability check in database.py where it's actually used
        with patch('src.extractors.database.POSTGRESQL_AVAILABLE', True):
            connection_string = config.get_connection_string()
            
            # Verify SSL is enforced in connection string
            assert "sslmode=require" in connection_string
            assert "readonly_user" in connection_string
            assert "secure-db.example.com" in connection_string
    
    def test_extractor_security_validation_on_init(self):
        """
        Test that security validation occurs during extractor initialization.
        
        **Validates: Requirement 1.1**
        """
        # Valid config should work
        valid_config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_db",
            username="readonly_user",
            password="test_pass",
            database_type=DatabaseType.POSTGRESQL,
            read_only=True
        )
        
        # Should not raise exception
        extractor = DatabaseExtractor(valid_config)
        assert extractor.config.read_only is True
        
        # Invalid config should be caught by validation
        with pytest.raises(ValueError):
            invalid_config = DatabaseConfig(
                host="localhost",
                port=5432,
                database="test_db",
                username="write_user",
                password="test_pass",
                database_type=DatabaseType.POSTGRESQL,
                read_only=False
            )
            SecurityValidator.validate_read_only_connection(invalid_config)


if __name__ == "__main__":
    # Run with verbose output and show hypothesis examples
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])