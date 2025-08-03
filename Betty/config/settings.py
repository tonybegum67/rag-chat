"""
Configuration settings for Betty AI Assistant.

This module centralizes all configuration parameters for the application,
making it easier to manage different environments and deployment scenarios.
"""

import os
from typing import Optional


class AppConfig:
    """Main application configuration class."""
    
    # API Configuration
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o"
    
    # Claude API Configuration
    ANTHROPIC_API_KEY: Optional[str] = None
    CLAUDE_MODEL: str = "claude-3-5-sonnet-20241022"
    
    # AI Provider Selection
    AI_PROVIDER: str = os.getenv("AI_PROVIDER", "claude")  # "openai" or "claude"
    
    # Database Configuration
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./data/betty_chroma_db")
    
    # Text Processing Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    TOKENIZER_MODEL: str = os.getenv("TOKENIZER_MODEL", "cl100k_base")
    
    # File Processing Configuration
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    SUPPORTED_FILE_TYPES: tuple = (".pdf", ".docx", ".txt")
    
    # UI Configuration
    PAGE_TITLE: str = "Betty - Your AI Assistant"
    PAGE_ICON: str = "💁‍♀️"
    
    # Knowledge Base Configuration
    KNOWLEDGE_COLLECTION_NAME: str = "betty_knowledge"
    DEFAULT_KNOWLEDGE_FILES: tuple = (
        "docs/Betty for Molex GPS.docx",
        "docs/Molex Manufacturing BA Reference Architecture.docx"
    )
    
    # RAG Enhancement Configuration
    USE_RERANKING: bool = bool(os.getenv("USE_RERANKING", "True"))
    USE_SEMANTIC_CHUNKING: bool = bool(os.getenv("USE_SEMANTIC_CHUNKING", "True"))
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Environment Configuration
    DISABLE_TOKENIZER_PARALLELISM: bool = True
    
    @classmethod
    def init_environment(cls):
        """Initialize environment variables and settings."""
        if cls.DISABLE_TOKENIZER_PARALLELISM:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings."""
        # Check API key based on selected provider
        if cls.AI_PROVIDER == "claude":
            if not cls.ANTHROPIC_API_KEY:
                return False
        elif cls.AI_PROVIDER == "openai":
            if not cls.OPENAI_API_KEY:
                return False
        else:
            return False  # Invalid provider
        
        if cls.CHUNK_SIZE <= 0 or cls.CHUNK_OVERLAP < 0:
            return False
            
        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            return False
            
        return True


class ChatConfig:
    """Configuration for the generic chat interface."""
    
    CHROMA_DB_PATH: str = os.getenv("CHAT_CHROMA_DB_PATH", "./data/chroma_db")
    PAGE_TITLE: str = "GPT-4o RAG Chat App"
    PAGE_ICON: str = "🤖"
    
    # Advanced Settings Defaults
    DEFAULT_CHUNK_SIZE: int = 500
    DEFAULT_OVERLAP: int = 50
    DEFAULT_N_RESULTS: int = 3
    DEFAULT_TEMPERATURE: float = 0.7
    
    # Constraints
    MIN_CHUNK_SIZE: int = 200
    MAX_CHUNK_SIZE: int = 1000
    MAX_OVERLAP: int = 200
    MIN_N_RESULTS: int = 1
    MAX_N_RESULTS: int = 10
    MIN_TEMPERATURE: float = 0.0
    MAX_TEMPERATURE: float = 1.0


# Initialize configuration on import
AppConfig.init_environment()