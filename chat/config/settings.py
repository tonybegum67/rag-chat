"""
Configuration settings for the Chat RAG system.
Independent of Betty configuration for clean separation.
"""
import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
STORAGE_ROOT = PROJECT_ROOT / "storage"
DOCUMENTS_PATH = STORAGE_ROOT / "documents"
CHROMA_DB_PATH = STORAGE_ROOT / "chroma_db"

# Ensure storage directories exist
DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)
CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)

class ChatConfig:
    """Chat application configuration"""
    PAGE_TITLE = "Chat with Documents"
    PAGE_ICON = "💬"
    
    # OpenAI settings
    OPENAI_MODEL = "gpt-4o"
    DEFAULT_TEMPERATURE = 0.7
    MIN_TEMPERATURE = 0.0
    MAX_TEMPERATURE = 2.0
    MAX_TOKENS = 1000
    
    # Document processing
    MAX_FILE_SIZE_MB = 50
    SUPPORTED_EXTENSIONS = [".txt", ".pdf", ".docx", ".md"]
    
    # Chunking settings
    DEFAULT_CHUNK_SIZE = 1000
    MIN_CHUNK_SIZE = 100
    MAX_CHUNK_SIZE = 5000
    DEFAULT_OVERLAP = 200
    MAX_OVERLAP = 500
    
    # Retrieval settings
    DEFAULT_N_RESULTS = 5
    MIN_N_RESULTS = 1
    MAX_N_RESULTS = 20
    SIMILARITY_THRESHOLD = 0.1  # Lowered for better retrieval
    
    # ChromaDB settings
    COLLECTION_PREFIX = "chat_docs"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384
    
    # Storage paths
    DOCUMENTS_STORAGE = str(DOCUMENTS_PATH)
    CHROMA_DB_STORAGE = str(CHROMA_DB_PATH)

class RAGConfig:
    """RAG system specific configuration"""
    
    # Chunking strategy
    CHUNK_METHOD = "recursive"  # recursive, fixed, semantic
    SENTENCE_SPLITTERS = ['. ', '! ', '? ', '\n\n', '\n']
    MIN_CHUNK_CHARS = 20  # Reduced for better compatibility
    MAX_CHUNK_CHARS = 2000
    
    # Document metadata
    METADATA_FIELDS = [
        "filename", "file_type", "upload_date", 
        "chunk_index", "total_chunks", "file_size"
    ]
    
    # Error handling
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    CHUNK_VALIDATION = True
    
    # Performance
    BATCH_SIZE = 100
    PARALLEL_PROCESSING = True
    MAX_WORKERS = 4

def get_api_key():
    """Get OpenAI API key from environment or secrets"""
    return os.getenv("OPENAI_API_KEY")

def validate_config():
    """Validate configuration settings"""
    if not get_api_key():
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    if not DOCUMENTS_PATH.exists():
        raise FileNotFoundError(f"Documents storage path does not exist: {DOCUMENTS_PATH}")
    
    if not CHROMA_DB_PATH.exists():
        raise FileNotFoundError(f"ChromaDB storage path does not exist: {CHROMA_DB_PATH}")
    
    return True