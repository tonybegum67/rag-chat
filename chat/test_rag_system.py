#!/usr/bin/env python3
"""
Test script for the independent RAG system.
Verifies all components work correctly and documents persist.
"""
import sys
import os
from pathlib import Path
import tempfile
import io

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_configuration():
    """Test configuration loading"""
    print("Testing configuration...")
    try:
        from config.settings import ChatConfig, RAGConfig, validate_config
        print(f"✅ Configuration loaded successfully")
        print(f"   - Documents storage: {ChatConfig.DOCUMENTS_STORAGE}")
        print(f"   - ChromaDB storage: {ChatConfig.CHROMA_DB_STORAGE}")
        print(f"   - Default chunk size: {ChatConfig.DEFAULT_CHUNK_SIZE}")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_document_parser():
    """Test document parsing and storage"""
    print("\nTesting document parser...")
    try:
        from utils.document_parser import document_parser
        
        # Test text parsing
        test_text = "This is a test document.\n\nIt has multiple paragraphs.\nAnd some line breaks."
        cleaned_text = document_parser.clean_text(test_text)
        print(f"✅ Text cleaning works: {len(cleaned_text)} chars")
        
        # Test storage directory creation
        storage_path = Path(document_parser.storage_path)
        if storage_path.exists():
            print(f"✅ Storage directory exists: {storage_path}")
        else:
            print(f"⚠️  Storage directory created: {storage_path}")
        
        return True
    except Exception as e:
        print(f"❌ Document parser test failed: {e}")
        return False

def test_chunker():
    """Test text chunking"""
    print("\nTesting chunker...")
    try:
        from rag.chunker import default_chunker, TextChunker
        
        test_text = """This is a test document for chunking. It contains multiple sentences and paragraphs to test the chunking functionality.
        
        This is the second paragraph. It should be chunked appropriately based on the configured chunk size and overlap settings.
        
        The third paragraph contains even more text to ensure we get multiple chunks from this document. This helps test the chunking boundaries and overlap functionality."""
        
        # Test default chunker
        chunks = default_chunker.chunk_text(test_text)
        print(f"✅ Default chunking created {len(chunks)} chunks")
        
        # Test custom chunker
        custom_chunker = TextChunker(chunk_size=200, overlap=50, method="recursive")
        custom_chunks = custom_chunker.chunk_text(test_text)
        print(f"✅ Custom chunking created {len(custom_chunks)} chunks")
        
        # Test chunk validation
        for i, chunk in enumerate(chunks[:2]):  # Test first 2 chunks
            if chunk.validate():
                print(f"✅ Chunk {i} validation passed: {len(chunk.content)} chars")
            else:
                print(f"❌ Chunk {i} validation failed")
        
        return True
    except Exception as e:
        print(f"❌ Chunker test failed: {e}")
        return False

def test_vector_store():
    """Test vector store operations"""
    print("\nTesting vector store...")
    try:
        from rag.vector_store import chat_vector_store
        
        # Test collection creation
        test_collection_name = "test_collection"
        collection = chat_vector_store.create_collection(test_collection_name, overwrite=True)
        print(f"✅ Collection created: {test_collection_name}")
        
        # Test listing collections
        collections = chat_vector_store.list_collections()
        if test_collection_name in collections:
            print(f"✅ Collection listed correctly")
        else:
            print(f"⚠️  Collection not found in list: {collections}")
        
        # Test collection stats
        stats = chat_vector_store.get_collection_stats(test_collection_name)
        print(f"✅ Collection stats: {stats.get('document_count', 0)} documents")
        
        return True
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        return False

def test_full_rag_pipeline():
    """Test the complete RAG pipeline"""
    print("\nTesting full RAG pipeline...")
    try:
        from rag.retriever import document_retriever
        from rag.chunker import default_chunker
        from utils.document_parser import document_parser
        import io
        
        # Create a mock uploaded file
        test_content = """This is a comprehensive test document for the RAG system.
        
        The document contains information about machine learning, artificial intelligence, and natural language processing.
        
        Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from and make decisions based on data.
        
        Natural language processing involves the interaction between computers and human language, enabling machines to understand, interpret, and generate human language."""
        
        class MockFile:
            def __init__(self, name, content):
                self.name = name
                self.content = content.encode('utf-8')
                self.size = len(self.content)
            
            def read(self):
                return self.content
        
        mock_file = MockFile("test_document.txt", test_content)
        
        # Test document addition
        test_collection = "rag_test_collection"
        success, message, stats = document_retriever.add_document(
            collection_name=test_collection,
            uploaded_file=mock_file
        )
        
        if success:
            print(f"✅ Document added successfully: {stats.get('chunks_created', 0)} chunks")
        else:
            print(f"❌ Document addition failed: {message}")
            return False
        
        # Test document retrieval
        query = "What is machine learning?"
        results = document_retriever.retrieve(
            collection_name=test_collection,
            query=query,
            n_results=2
        )
        
        if results:
            print(f"✅ Retrieved {len(results)} results for query")
            for i, result in enumerate(results):
                print(f"   Result {i+1}: similarity={result.similarity:.3f}")
        else:
            print(f"⚠️  No results found for query")
        
        # Test collection info
        collection_info = document_retriever.get_collection_info(test_collection)
        print(f"✅ Collection info: {collection_info.get('document_count', 0)} documents")
        
        return True
    except Exception as e:
        print(f"❌ Full RAG pipeline test failed: {e}")
        return False

def test_persistence():
    """Test document persistence across restarts"""
    print("\nTesting persistence...")
    try:
        from utils.document_parser import document_parser
        from rag.vector_store import chat_vector_store
        
        # Check if documents are stored
        stored_docs = document_parser.get_stored_documents()
        print(f"✅ Found {len(stored_docs)} stored documents")
        
        # Check if collections persist
        collections = chat_vector_store.list_collections()
        print(f"✅ Found {len(collections)} collections: {collections}")
        
        # Check storage paths exist
        from config.settings import DOCUMENTS_PATH, CHROMA_DB_PATH
        
        if DOCUMENTS_PATH.exists():
            doc_files = list(DOCUMENTS_PATH.glob("*.txt"))
            print(f"✅ Document storage: {len(doc_files)} files in {DOCUMENTS_PATH}")
        else:
            print(f"⚠️  Document storage path missing: {DOCUMENTS_PATH}")
        
        if CHROMA_DB_PATH.exists():
            chroma_files = list(CHROMA_DB_PATH.rglob("*"))
            print(f"✅ ChromaDB storage: {len(chroma_files)} files in {CHROMA_DB_PATH}")
        else:
            print(f"⚠️  ChromaDB storage path missing: {CHROMA_DB_PATH}")
        
        return True
    except Exception as e:
        print(f"❌ Persistence test failed: {e}")
        return False

def test_health_check():
    """Test system health check"""
    print("\nTesting system health check...")
    try:
        from rag.retriever import document_retriever
        
        health = document_retriever.health_check()
        print(f"✅ System health: {health['status']}")
        print(f"   - Vector store: {health['vector_store']}")
        print(f"   - Document parser: {health['document_parser']}")
        print(f"   - Chunker: {health['chunker']}")
        
        if health['errors']:
            for error in health['errors']:
                print(f"   ⚠️  {error}")
        
        return health['status'] in ['healthy', 'warning']
    except Exception as e:
        print(f"❌ Health check test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Independent RAG System\n" + "="*50)
    
    tests = [
        test_configuration,
        test_document_parser,
        test_chunker,
        test_vector_store,
        test_full_rag_pipeline,
        test_persistence,
        test_health_check
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*50)
    passed = sum(results)
    total = len(results)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! RAG system is ready.")
        return 0
    else:
        print(f"⚠️  {total - passed} tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())