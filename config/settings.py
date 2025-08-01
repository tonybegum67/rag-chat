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
    
    # Database Configuration
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./betty_chroma_db")
    
    # Text Processing Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "3"))
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
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
        "Betty for Molex GPS.docx",
        "Molex Manufacturing BA Reference Architecture.docx"
    )
    
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
        if not cls.OPENAI_API_KEY:
            return False
        
        if cls.CHUNK_SIZE <= 0 or cls.CHUNK_OVERLAP < 0:
            return False
            
        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            return False
            
        return True


class ChatConfig:
    """Configuration for the generic chat interface."""
    
    CHROMA_DB_PATH: str = os.getenv("CHAT_CHROMA_DB_PATH", "./chroma_db")
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