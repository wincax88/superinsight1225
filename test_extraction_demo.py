"""
Demo script to test the data extraction functionality.

This script demonstrates the basic usage of the data extraction module.
"""

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.extractors.factory import ExtractorFactory
from src.extractors.base import DatabaseConfig, FileConfig, APIConfig, DatabaseType, FileType


def test_file_extraction():
    """Test file extraction with a simple text file."""
    print("Testing file extraction...")
    
    try:
        # Create a test text file
        test_file = "test_document.txt"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("这是一个测试文档。\n这个文档用于测试SuperInsight平台的数据提取功能。\n包含中文内容。")
        
        # Create file extractor
        extractor = ExtractorFactory.create_file_extractor(
            file_path=test_file,
            file_type="txt"
        )
        
        # Test connection
        if extractor.test_connection():
            print("✓ File connection test passed")
        else:
            print("✗ File connection test failed")
            return False
        
        # Extract data
        result = extractor.extract_data()
        
        if result.success:
            print(f"✓ File extraction successful: {result.extracted_count} documents")
            for i, doc in enumerate(result.documents):
                print(f"  Document {i+1}: {len(doc.content)} characters")
                print(f"  Content preview: {doc.content[:100]}...")
        else:
            print(f"✗ File extraction failed: {result.error}")
            return False
        
        # Clean up
        Path(test_file).unlink()
        return True
        
    except Exception as e:
        print(f"✗ File extraction test failed: {e}")
        return False


def test_api_extraction():
    """Test API extraction with a public API."""
    print("\nTesting API extraction...")
    
    try:
        # Create API extractor for a public API
        extractor = ExtractorFactory.create_api_extractor(
            base_url="https://jsonplaceholder.typicode.com",
            headers={"User-Agent": "SuperInsight-Test/1.0"}
        )
        
        # Test connection
        if extractor.test_connection():
            print("✓ API connection test passed")
        else:
            print("✗ API connection test failed")
            return False
        
        # Extract data from posts endpoint
        result = extractor.extract_data(
            endpoint="posts",
            method="GET",
            paginate=False
        )
        
        if result.success:
            print(f"✓ API extraction successful: {result.extracted_count} documents")
            for i, doc in enumerate(result.documents[:2]):  # Show first 2
                print(f"  Document {i+1}: {len(doc.content)} characters")
        else:
            print(f"✗ API extraction failed: {result.error}")
            return False
        
        extractor.close()
        return True
        
    except Exception as e:
        print(f"✗ API extraction test failed: {e}")
        return False


def test_web_extraction():
    """Test web extraction with a simple webpage."""
    print("\nTesting web extraction...")
    
    try:
        # Create web extractor
        extractor = ExtractorFactory.create_web_extractor(
            base_url="https://httpbin.org/html",
            max_pages=1
        )
        
        # Test connection
        if extractor.test_connection():
            print("✓ Web connection test passed")
        else:
            print("✗ Web connection test failed")
            return False
        
        # Extract data
        result = extractor.extract_data(max_depth=1)
        
        if result.success:
            print(f"✓ Web extraction successful: {result.extracted_count} documents")
            for i, doc in enumerate(result.documents):
                print(f"  Document {i+1}: {len(doc.content)} characters")
                print(f"  Title: {doc.metadata.get('title', 'N/A')}")
        else:
            print(f"✗ Web extraction failed: {result.error}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Web extraction test failed: {e}")
        return False


def main():
    """Run all extraction tests."""
    print("SuperInsight Platform - Data Extraction Demo")
    print("=" * 50)
    
    tests = [
        test_file_extraction,
        test_api_extraction,
        test_web_extraction
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Data extraction module is working correctly.")
        return True
    else:
        print("✗ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)