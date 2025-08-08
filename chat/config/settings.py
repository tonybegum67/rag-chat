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

# Detect if running on Streamlit Cloud or in container environment
IS_STREAMLIT_CLOUD = (
    # Official Streamlit Cloud environment variables
    os.environ.get('STREAMLIT_RUNTIME_ENV') == 'cloud' or
    os.environ.get('STREAMLIT_SERVER_ADDRESS') is not None or
    os.environ.get('STREAMLIT_SHARING_MODE') == 'true' or
    # Check for Streamlit-specific environment
    'STREAMLIT' in os.environ or
    os.environ.get('STREAMLIT_SERVER_HEADLESS') == 'true' or
    # Common Streamlit Cloud paths and users
    os.environ.get('HOME') == '/home/appuser' or
    os.environ.get('USER') == 'appuser' or
    # Check if running in a container (common for Streamlit Cloud)
    os.path.exists('/.dockerenv') or
    os.environ.get('KUBERNETES_SERVICE_HOST') is not None or
    # Check for cloud platform indicators
    os.environ.get('DYNO') is not None or  # Heroku
    os.environ.get('AWS_LAMBDA_FUNCTION_NAME') is not None or  # AWS Lambda
    os.environ.get('GOOGLE_CLOUD_PROJECT') is not None or  # Google Cloud
    # Check if not running locally (no .git directory in parent)
    (not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.git')) and
     os.environ.get('USER') != os.environ.get('LOGNAME'))  # Different user context
)

def is_streamlit_running():
    """Detect if we're running in a Streamlit environment (local or cloud)"""
    try:
        import streamlit as st
        # Check if we can access Streamlit's runtime context
        try:
            # This will only work if we're actually running in Streamlit
            st.get_option("server.headless")
            return True
        except:
            # Fallback - if streamlit is imported, we're likely in Streamlit
            return True
    except ImportError:
        return False

def is_cloud_environment():
    """Enhanced cloud detection that works at runtime"""
    # First check the static IS_STREAMLIT_CLOUD
    if IS_STREAMLIT_CLOUD:
        return True
    
    # Runtime checks for Streamlit Cloud
    if is_streamlit_running():
        try:
            import streamlit as st
            # Check if secrets are available (typical in Streamlit Cloud)
            if hasattr(st, 'secrets') and len(st.secrets) > 0:
                return True
        except:
            pass
    
    # Additional runtime checks
    cloud_indicators = [
        # Check for temp directory usage (common in cloud)
        '/tmp' in str(get_storage_root()) if 'get_storage_root' in globals() else False,
        # Check if we're running on a different user than locally expected
        os.environ.get('USER') in ['appuser', 'runner', 'app'],
        # Check for common cloud environment variables
        any(key in os.environ for key in ['DYNO', 'AWS_EXECUTION_ENV', 'GOOGLE_CLOUD_PROJECT']),
        # Check if we're in a containerized environment
        os.path.exists('/.dockerenv'),
    ]
    
    return any(cloud_indicators)

def get_storage_root():
    """Get appropriate storage root based on environment"""
    import logging
    logger = logging.getLogger(__name__)
    
    # Use both static and dynamic detection for maximum reliability
    is_cloud = IS_STREAMLIT_CLOUD or is_cloud_environment()
    
    if is_cloud:
        # Use temp directory on Streamlit Cloud
        # Note: This is ephemeral and will be cleared on restart
        base_temp = Path(tempfile.gettempdir())
        storage_dir = base_temp / "rag_chat_storage"
        try:
            storage_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cloud environment detected. Using temp storage: {storage_dir}")
        except Exception as e:
            logger.error(f"Failed to create temp storage directory: {e}")
            # Create a unique temp directory as fallback
            import uuid
            storage_dir = base_temp / f"rag_chat_{uuid.uuid4().hex[:8]}"
            storage_dir.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Using fallback temp storage: {storage_dir}")
        return storage_dir
    else:
        # Use local project directory for development
        storage_dir = Path(__file__).parent.parent / "storage"
        logger.info(f"Local environment detected. Using project storage: {storage_dir}")
        return storage_dir

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