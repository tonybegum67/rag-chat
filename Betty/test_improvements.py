#!/usr/bin/env python3
"""
Simple test script to validate the improvements made to Betty AI Assistant.

This script tests:
1. Configuration loading and validation
2. Document processor functionality
3. Vector store initialization
4. Import compatibility
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_configuration():
    """Test configuration loading and validation."""
    print("Testing configuration...")
    
    try:
        from config.settings import AppConfig, ChatConfig
        
        # Test configuration initialization
        AppConfig.init_environment()
        
        # Test basic configuration values
        assert AppConfig.CHUNK_SIZE > 0, "CHUNK_SIZE must be positive"
        assert AppConfig.CHUNK_OVERLAP >= 0, "CHUNK_OVERLAP must be non-negative"
        assert AppConfig.CHUNK_OVERLAP < AppConfig.CHUNK_SIZE, "CHUNK_OVERLAP must be less than CHUNK_SIZE"
        assert AppConfig.PAGE_TITLE, "PAGE_TITLE must be set"
        assert AppConfig.EMBEDDING_MODEL, "EMBEDDING_MODEL must be set"
        
        print("✅ Configuration tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration tests failed: {e}")
        return False

def test_document_processor():
    """Test document processor functionality."""
    print("Testing document processor...")
    
    try:
        from utils.document_processor import DocumentProcessor, document_processor
        
        # Test initialization
        processor = DocumentProcessor()
        assert processor.tokenizer is not None, "Tokenizer must be initialized"
        
        # Test text cleaning
        test_text = "  This   is\n\na\n  test.\n\n  "
        cleaned = processor.clean_text(test_text)
        assert cleaned == "This is\na\ntest.", f"Text cleaning failed: '{cleaned}'"
        
        # Test chunking
        test_text = "This is a test document. " * 100  # Create a longer text
        chunks = processor.chunk_text(test_text, chunk_size=50, overlap=10)
        assert len(chunks) > 1, "Should create multiple chunks"
        assert all(chunk.strip() for chunk in chunks), "All chunks should have content"
        
        # Test file type detection
        assert processor.get_file_type("test.pdf") == "pdf"
        assert processor.get_file_type("test.docx") == "docx"
        assert processor.get_file_type("test.txt") == "txt"
        assert processor.get_file_type("test.xyz") is None
        
        print("✅ Document processor tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Document processor tests failed: {e}")
        return False

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        # Test configuration imports
        from config.settings import AppConfig, ChatConfig
        
        # Test utilities imports
        from utils.document_processor import document_processor
        from utils.vector_store import betty_vector_store, chat_vector_store
        
        print("✅ Import tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Import tests failed: {e}")
        return False

def test_integration():
    """Test integration between components."""
    print("Testing component integration...")
    
    try:
        from config.settings import AppConfig
        from utils.document_processor import document_processor
        
        # Test that document processor uses configuration
        chunks = document_processor.chunk_text("Test text", chunk_size=AppConfig.CHUNK_SIZE)
        assert isinstance(chunks, list), "Chunking should return a list"
        
        print("✅ Integration tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Integration tests failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Running Betty AI Assistant improvement validation tests...")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_configuration,
        test_document_processor,
        test_integration
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
        print("🎉 All tests passed! Improvements are working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the improvements.")
        return 1

if __name__ == "__main__":
    sys.exit(main())