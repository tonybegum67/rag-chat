"""
Configuration settings for the RAG Chat application.
Standalone configuration for the chat system with cloud compatibility.
"""
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Detect if running on Streamlit Cloud
IS_STREAMLIT_CLOUD = (
    os.environ.get('STREAMLIT_RUNTIME_ENV') == 'cloud' or
    os.environ.get('STREAMLIT_SERVER_ADDRESS') is not None or
    os.environ.get('STREAMLIT_SHARING_MODE') == 'true'
)

def get_storage_root():
    """Get appropriate storage root based on environment"""
    if IS_STREAMLIT_CLOUD:
        # Use temp directory on Streamlit Cloud
        # Note: This is ephemeral and will be cleared on restart
        base_temp = Path(tempfile.gettempdir())
        storage_dir = base_temp / "rag_chat_storage"
        storage_dir.mkdir(parents=True, exist_ok=True)
        return storage_dir
    else:
        # Use local project directory for development
        return Path(__file__).parent.parent / "storage"

# Base paths with environment detection
PROJECT_ROOT = Path(__file__).parent.parent
STORAGE_ROOT = get_storage_root()
DOCUMENTS_PATH = STORAGE_ROOT / "documents"
CHROMA_DB_PATH = STORAGE_ROOT / "chroma_db"

# Ensure storage directories exist with proper error handling
try:
    DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)
    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
except PermissionError as e:
    # Fallback to temp directory if permission denied
    import logging
    logging.warning(f"Permission denied creating directories: {e}")
    temp_root = Path(tempfile.mkdtemp(prefix="rag_chat_"))
    STORAGE_ROOT = temp_root
    DOCUMENTS_PATH = STORAGE_ROOT / "documents"
    CHROMA_DB_PATH = STORAGE_ROOT / "chroma_db"
    DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)
    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)

class ChatConfig:
    """Chat application configuration with cloud awareness"""
    PAGE_TITLE = "Chat with Documents"
    PAGE_ICON = "💬"
    
    # OpenAI settings
    OPENAI_MODEL = "gpt-4o"
    DEFAULT_TEMPERATURE = 0.7
    MIN_TEMPERATURE = 0.0
    MAX_TEMPERATURE = 2.0
    MAX_TOKENS = 1000
    
    # Document processing - Adjust for cloud limitations
    MAX_FILE_SIZE_MB = 10 if IS_STREAMLIT_CLOUD else 50  # Smaller on cloud
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
    SIMILARITY_THRESHOLD = 0.0  # Very permissive threshold to allow more results
    
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
    
    # Performance - Optimize for cloud constraints
    BATCH_SIZE = 50 if IS_STREAMLIT_CLOUD else 100
    PARALLEL_PROCESSING = not IS_STREAMLIT_CLOUD  # Disable parallel on cloud to avoid resource issues
    MAX_WORKERS = 2 if IS_STREAMLIT_CLOUD else 4

def get_api_key():
    """Get OpenAI API key from environment or Streamlit secrets"""
    # Try Streamlit secrets first (for cloud deployment)
    if IS_STREAMLIT_CLOUD:
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
                return st.secrets['OPENAI_API_KEY']
        except:
            pass
    
    # Fall back to environment variable (for local development)
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