#!/usr/bin/env python3
"""
SuperInsight Platform Setup Test Script
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_project_structure():
    """Test that all required directories and files exist"""
    print("🔍 Testing project structure...")
    
    required_dirs = [
        "src",
        "src/models",
        "src/config", 
        "src/database",
        "src/extractors",
        "src/label_studio",
        "src/ai",
        "src/quality",
        "src/billing",
        "src/security",
        "src/api",
        "src/utils",
        "tests",
        "scripts",
        ".kiro/specs/superinsight-platform"
    ]
    
    required_files = [
        "requirements.txt",
        ".env",
        ".env.example", 
        ".gitignore",
        "README.md",
        "main.py",
        "docker-compose.yml",
        "Dockerfile.dev",
        "scripts/init-db.sql",
        "src/config/settings.py",
        "src/database/connection.py",
        "src/label_studio/config.py"
    ]
    
    missing_dirs = []
    missing_files = []
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required directories and files exist")
    return True


def test_configuration():
    """Test configuration loading"""
    print("🔍 Testing configuration...")
    
    try:
        from config.settings import settings
        
        # Test basic settings
        assert settings.app.app_name == "SuperInsight Platform"
        assert settings.app.app_version == "1.0.0"
        assert settings.database.database_name == "superinsight"
        assert settings.label_studio.label_studio_url == "http://localhost:8080"
        
        print("✅ Configuration loaded and validated successfully")
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def test_label_studio_config():
    """Test Label Studio configuration"""
    print("🔍 Testing Label Studio configuration...")
    
    try:
        from label_studio.config import label_studio_config
        
        # Test validation
        is_valid = label_studio_config.validate_config()
        assert is_valid, "Label Studio config validation failed"
        
        # Test default configs
        text_config = label_studio_config.get_default_label_config("text_classification")
        assert len(text_config) > 0, "Text classification config is empty"
        
        ner_config = label_studio_config.get_default_label_config("named_entity_recognition")
        assert len(ner_config) > 0, "NER config is empty"
        
        # Test project config
        project_config = label_studio_config.get_project_config()
        required_keys = ["title", "description", "label_config"]
        for key in required_keys:
            assert key in project_config, f"Missing key: {key}"
        
        print("✅ Label Studio configuration validated successfully")
        return True
        
    except Exception as e:
        print(f"❌ Label Studio configuration error: {e}")
        return False


def test_environment_files():
    """Test environment configuration files"""
    print("🔍 Testing environment files...")
    
    try:
        # Check .env.example exists and has required keys
        with open(".env.example", "r") as f:
            env_example = f.read()
        
        required_env_vars = [
            "DATABASE_URL",
            "LABEL_STUDIO_URL", 
            "SECRET_KEY",
            "OLLAMA_BASE_URL",
            "APP_NAME"
        ]
        
        for var in required_env_vars:
            assert var in env_example, f"Missing environment variable: {var}"
        
        # Check .env exists
        assert os.path.exists(".env"), ".env file missing"
        
        print("✅ Environment files validated successfully")
        return True
        
    except Exception as e:
        print(f"❌ Environment files error: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 SuperInsight Platform Setup Test")
    print("=" * 50)
    
    tests = [
        test_project_structure,
        test_configuration,
        test_label_studio_config,
        test_environment_files
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Project setup is complete.")
        print("\n📋 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Start services: docker-compose up -d")
        print("3. Run application: python main.py")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)