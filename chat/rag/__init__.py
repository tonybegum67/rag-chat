"""
RAG (Retrieval-Augmented Generation) system for Chat application.
Independent implementation with persistent storage and comprehensive error handling.
"""
from .chunker import TextChunker, Chunk, default_chunker, ChunkingError
from .vector_store import ChatVectorStore, chat_vector_store, VectorStoreError
from .retriever import DocumentRetriever, RetrievalResult, document_retriever, RetrievalError

__all__ = [
    # Chunking
    'TextChunker', 'Chunk', 'default_chunker', 'ChunkingError',
    
    # Vector Store
    'ChatVectorStore', 'chat_vector_store', 'VectorStoreError',
    
    # Retrieval
    'DocumentRetriever', 'RetrievalResult', 'document_retriever', 'RetrievalError'
]

# Version info
__version__ = "1.0.0"
__author__ = "Chat RAG System"
__description__ = "Independent RAG system with persistent storage"