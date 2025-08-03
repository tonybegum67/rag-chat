#!/usr/bin/env python3
"""
Test script to validate the chat app improvements.

This script tests:
1. Import compatibility
2. Configuration usage
3. Function integration
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_chat_imports():
    """Test that chat app imports work correctly."""
    print("Testing chat app imports...")
    
    try:
        # Add the streamlit chat directory to the path
        chat_dir = project_root / "streamlit" / "chat"
        sys.path.insert(0, str(chat_dir))
        
        # Test key imports from the chat module
        from config.settings import ChatConfig, AppConfig
        from utils.document_processor import document_processor
        from utils.vector_store import chat_vector_store
        
        print("✅ Chat imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Chat imports failed: {e}")
        return False

def test_chat_config():
    """Test chat configuration values."""
    print("Testing chat configuration...")
    
    try:
        from config.settings import ChatConfig
        
        # Test configuration values
        assert ChatConfig.PAGE_TITLE, "PAGE_TITLE must be set"
        assert ChatConfig.PAGE_ICON, "PAGE_ICON must be set"
        assert ChatConfig.DEFAULT_CHUNK_SIZE > 0, "DEFAULT_CHUNK_SIZE must be positive"
        assert ChatConfig.DEFAULT_OVERLAP >= 0, "DEFAULT_OVERLAP must be non-negative"
        assert ChatConfig.DEFAULT_N_RESULTS > 0, "DEFAULT_N_RESULTS must be positive"
        assert 0 <= ChatConfig.DEFAULT_TEMPERATURE <= 1, "DEFAULT_TEMPERATURE must be between 0 and 1"
        
        # Test constraint values
        assert ChatConfig.MIN_CHUNK_SIZE < ChatConfig.MAX_CHUNK_SIZE, "MIN_CHUNK_SIZE must be less than MAX_CHUNK_SIZE"
        assert ChatConfig.MIN_N_RESULTS < ChatConfig.MAX_N_RESULTS, "MIN_N_RESULTS must be less than MAX_N_RESULTS"
        assert ChatConfig.MIN_TEMPERATURE < ChatConfig.MAX_TEMPERATURE, "MIN_TEMPERATURE must be less than MAX_TEMPERATURE"
        
        print("✅ Chat configuration tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Chat configuration tests failed: {e}")
        return False

def test_integration():
    """Test integration between chat components."""
    print("Testing chat integration...")
    
    try:
        from config.settings import ChatConfig
        from utils.document_processor import document_processor
        
        # Test that document processor can use chat config values
        test_text = "This is a test document for the chat app."
        chunks = document_processor.chunk_text(
            test_text, 
            chunk_size=ChatConfig.DEFAULT_CHUNK_SIZE,
            overlap=ChatConfig.DEFAULT_OVERLAP
        )
        assert isinstance(chunks, list), "Chunking should return a list"
        assert len(chunks) > 0, "Should create at least one chunk"
        
        print("✅ Chat integration tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Chat integration tests failed: {e}")
        return False

def test_file_structure():
    """Test that the chat file has been properly updated."""
    print("Testing chat file structure...")
    
    try:
        chat_file = project_root / "streamlit" / "chat" / "chat.py"
        
        if not chat_file.exists():
            print("❌ Chat file not found")
            return False
        
        # Read and check for improved patterns
        with open(chat_file, 'r') as f:
            content = f.read()
        
        # Check for shared utilities usage
        if "from config.settings import ChatConfig" not in content:
            print("❌ Missing ChatConfig import")
            return False
        
        if "from utils.document_processor import document_processor" not in content:
            print("❌ Missing document_processor import")
            return False
        
        if "from utils.vector_store import chat_vector_store" not in content:
            print("❌ Missing chat_vector_store import")
            return False
        
        # Check for configuration usage
        if "ChatConfig.DEFAULT_CHUNK_SIZE" not in content:
            print("❌ Missing ChatConfig usage")
            return False
        
        print("✅ Chat file structure tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Chat file structure tests failed: {e}")
        return False

def main():
    """Run all chat improvement tests."""
    print("🚀 Running Chat App improvement validation tests...")
    print("=" * 60)
    
    tests = [
        test_chat_imports,
        test_chat_config,
        test_integration,
        test_file_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All chat tests passed! Chat app improvements are working correctly.")
        return 0
    else:
        print("⚠️  Some chat tests failed. Please review the improvements.")
        return 1

if __name__ == "__main__":
    sys.exit(main())