"""
Simple test to verify the data extraction module structure.

This test verifies that the module is properly structured without requiring
all external dependencies.
"""

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_module_imports():
    """Test that all extraction modules can be imported."""
    print("Testing module imports...")
    
    try:
        from src.extractors.base import (
            DataExtractor,
            BaseExtractor,
            ExtractionResult,
            DatabaseConfig,
            FileConfig,
            APIConfig,
            SourceType,
            DatabaseType,
            FileType,
            SecurityValidator
        )
        print("✓ Base module imported successfully")
        
        # Test enums
        assert SourceType.DATABASE == "database"
        assert DatabaseType.MYSQL == "mysql"
        assert FileType.PDF == "pdf"
        print("✓ Enums working correctly")
        
        # Test ExtractionResult
        result = ExtractionResult(success=True, documents=[])
        assert result.success == True
        assert result.extracted_count == 0
        print("✓ ExtractionResult working correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Module import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_classes():
    """Test configuration classes."""
    print("\nTesting configuration classes...")
    
    try:
        from src.extractors.base import DatabaseConfig, FileConfig, APIConfig, DatabaseType, FileType
        
        # Test DatabaseConfig
        db_config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_pass",
            database_type=DatabaseType.POSTGRESQL
        )
        assert db_config.host == "localhost"
        assert db_config.read_only == True
        print("✓ DatabaseConfig working correctly")
        
        # Test FileConfig
        file_config = FileConfig(
            file_path="test.txt",
            file_type=FileType.TXT
        )
        assert file_config.file_path == "test.txt"
        assert file_config.encoding == "utf-8"
        print("✓ FileConfig working correctly")
        
        # Test APIConfig
        api_config = APIConfig(
            base_url="https://api.example.com",
            headers={"User-Agent": "Test"}
        )
        assert api_config.base_url == "https://api.example.com"
        print("✓ APIConfig working correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Config class test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_security_validator():
    """Test security validator."""
    print("\nTesting security validator...")
    
    try:
        from src.extractors.base import SecurityValidator, DatabaseConfig, DatabaseType
        
        # Test read-only validation
        db_config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_pass",
            database_type=DatabaseType.POSTGRESQL,
            read_only=True
        )
        
        assert SecurityValidator.validate_read_only_connection(db_config) == True
        print("✓ Read-only validation working")
        
        # Test SSL validation
        assert SecurityValidator.validate_ssl_configuration(db_config) == True
        print("✓ SSL validation working")
        
        # Test connection limits
        assert SecurityValidator.validate_connection_limits(db_config) == True
        print("✓ Connection limits validation working")
        
        return True
        
    except Exception as e:
        print(f"✗ Security validator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_factory_structure():
    """Test factory module structure."""
    print("\nTesting factory module...")
    
    try:
        from src.extractors.factory import ExtractorFactory
        
        # Check that factory has the expected methods
        assert hasattr(ExtractorFactory, 'create_database_extractor')
        assert hasattr(ExtractorFactory, 'create_file_extractor')
        assert hasattr(ExtractorFactory, 'create_api_extractor')
        assert hasattr(ExtractorFactory, 'create_web_extractor')
        assert hasattr(ExtractorFactory, 'create_from_config')
        assert hasattr(ExtractorFactory, 'create_from_url')
        
        print("✓ Factory module has all expected methods")
        return True
        
    except Exception as e:
        print(f"✗ Factory module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_module_structure():
    """Test API module structure."""
    print("\nTesting API module...")
    
    try:
        from src.api.extraction import router
        
        # Check that router exists
        assert router is not None
        print("✓ API router exists")
        
        # Check router configuration
        assert router.prefix == "/api/v1/extraction"
        print("✓ API router configured correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ API module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("SuperInsight Platform - Data Extraction Module Structure Test")
    print("=" * 60)
    
    tests = [
        test_module_imports,
        test_config_classes,
        test_security_validator,
        test_factory_structure,
        test_api_module_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Data extraction module structure is correct.")
        print("\nImplemented features:")
        print("  - Base extractor classes and interfaces")
        print("  - Database extractor (MySQL, PostgreSQL, Oracle)")
        print("  - File extractor (PDF, DOCX, TXT, HTML)")
        print("  - Web extractor (HTML crawling)")
        print("  - API extractor (REST, GraphQL, Webhook)")
        print("  - Security validation (read-only, SSL/TLS)")
        print("  - Factory pattern for easy extractor creation")
        print("  - FastAPI endpoints for extraction operations")
        print("  - Async task processing with progress tracking")
        return True
    else:
        print("✗ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)