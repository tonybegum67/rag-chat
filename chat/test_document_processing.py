#!/usr/bin/env python3
"""
Test script to verify document processing works correctly
"""
import os
import io
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_environment_detection():
    """Test environment detection"""
    from config.settings import IS_STREAMLIT_CLOUD, STORAGE_ROOT, DOCUMENTS_PATH, CHROMA_DB_PATH
    
    logger.info("=" * 60)
    logger.info("ENVIRONMENT DETECTION TEST")
    logger.info("=" * 60)
    logger.info(f"IS_STREAMLIT_CLOUD: {IS_STREAMLIT_CLOUD}")
    logger.info(f"STORAGE_ROOT: {STORAGE_ROOT}")
    logger.info(f"DOCUMENTS_PATH: {DOCUMENTS_PATH}")
    logger.info(f"CHROMA_DB_PATH: {CHROMA_DB_PATH}")
    logger.info(f"Storage root exists: {STORAGE_ROOT.exists()}")
    logger.info(f"Documents path exists: {DOCUMENTS_PATH.exists()}")
    logger.info(f"ChromaDB path exists: {CHROMA_DB_PATH.exists()}")
    logger.info("")

def test_vector_store_init():
    """Test vector store initialization"""
    logger.info("=" * 60)
    logger.info("VECTOR STORE INITIALIZATION TEST")
    logger.info("=" * 60)
    
    try:
        from rag.vector_store import chat_vector_store
        logger.info(f"Vector store initialized successfully")
        logger.info(f"Persist directory: {chat_vector_store.persist_directory}")
        logger.info(f"Embedding model: {chat_vector_store.embedding_model_name}")
        logger.info("")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        return False

def test_document_parser_init():
    """Test document parser initialization"""
    logger.info("=" * 60)
    logger.info("DOCUMENT PARSER INITIALIZATION TEST")
    logger.info("=" * 60)
    
    try:
        from utils.document_parser import document_parser
        logger.info(f"Document parser initialized successfully")
        logger.info(f"Storage path: {document_parser.storage_path}")
        logger.info(f"Metadata file: {document_parser.metadata_file}")
        logger.info("")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize document parser: {e}")
        return False

def test_document_processing():
    """Test document processing with a simple text file"""
    logger.info("=" * 60)
    logger.info("DOCUMENT PROCESSING TEST")
    logger.info("=" * 60)
    
    try:
        from utils.document_parser import document_parser
        from rag.retriever import document_retriever
        
        # Create a mock uploaded file
        class MockUploadedFile:
            def __init__(self, name, content):
                self.name = name
                self._content = content
                
            def read(self):
                return self._content.encode('utf-8')
            
            def getvalue(self):
                return self._content.encode('utf-8')
        
        # Create test content
        test_content = "This is a test document for RAG processing. It contains sample text that should be chunked and stored in the vector database."
        mock_file = MockUploadedFile("test_document.txt", test_content)
        
        # Process the document
        logger.info("Processing test document...")
        success, message, stats = document_retriever.add_document(
            collection_name="test_collection",
            uploaded_file=mock_file,
            chunk_size=100,
            overlap=20
        )
        
        if success:
            logger.info(f"Document processed successfully: {message}")
            logger.info(f"Stats: {stats}")
        else:
            logger.error(f"Document processing failed: {message}")
            
        return success
        
    except Exception as e:
        logger.error(f"Failed to process document: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_document_retrieval():
    """Test document retrieval"""
    logger.info("=" * 60)
    logger.info("DOCUMENT RETRIEVAL TEST")
    logger.info("=" * 60)
    
    try:
        from rag.retriever import document_retriever
        
        # Try to retrieve from test collection
        results = document_retriever.retrieve(
            collection_name="test_collection",
            query="test document RAG",
            n_results=3
        )
        
        if results:
            logger.info(f"Retrieved {len(results)} results")
            for i, result in enumerate(results, 1):
                logger.info(f"Result {i}: Similarity={result.similarity:.3f}, Content preview: {result.content[:100]}...")
        else:
            logger.warning("No results retrieved (collection might be empty)")
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to retrieve documents: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    logger.info("Starting RAG Document Processing Tests")
    logger.info("")
    
    # Run tests
    test_environment_detection()
    
    if test_vector_store_init():
        logger.info("✓ Vector store initialization successful")
    else:
        logger.error("✗ Vector store initialization failed")
        
    if test_document_parser_init():
        logger.info("✓ Document parser initialization successful")
    else:
        logger.error("✗ Document parser initialization failed")
        
    if test_document_processing():
        logger.info("✓ Document processing successful")
    else:
        logger.error("✗ Document processing failed")
        
    if test_document_retrieval():
        logger.info("✓ Document retrieval successful")
    else:
        logger.error("✗ Document retrieval failed")
    
    logger.info("")
    logger.info("Tests completed!")

if __name__ == "__main__":
    main()