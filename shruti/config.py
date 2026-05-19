"""
Configuration module for ŚRUTI package.
Handles environment variables and default settings.
"""

import os
from pathlib import Path
from typing import Optional


class Config:
    """Configuration class with environment variable defaults."""
    
    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    # Vector Database
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./knowledge_base")
    
    # Models
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", 
        "paraphrase-multilingual-MiniLM-L12-v2"
    )
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    
    # Text Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "75"))
    
    # Audio Processing
    AUDIO_OUTPUT_DIR: str = os.getenv("AUDIO_OUTPUT_DIR", "/tmp")
    
    # Groq Whisper Settings
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "whisper-large-v3")
    
    # V2.0 Multi-Modal Settings
    FRAME_INTERVAL: float = float(os.getenv("FRAME_INTERVAL", "5.0"))
    MAX_FRAMES: int = int(os.getenv("MAX_FRAMES", "200"))
    OCR_LANGUAGES: str = os.getenv("OCR_LANGUAGES", "eng+hin+san")
    MAX_SPEAKERS: int = int(os.getenv("MAX_SPEAKERS", "10"))
    KNOWLEDGE_GRAPH_DIR: str = os.getenv("KNOWLEDGE_GRAPH_DIR", "")
    HUGGINGFACE_TOKEN: str = os.getenv("HUGGINGFACE_TOKEN", "")
    
    @classmethod
    def validate(cls) -> None:
        """Validate required configuration."""
        if not cls.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY environment variable is required. "
                "Get your API key from https://console.groq.com/keys"
            )
    
    @classmethod
    def get_chroma_persist_path(cls) -> Path:
        """Get the Chroma persistence directory as a Path object."""
        return Path(cls.CHROMA_PERSIST_DIR).resolve()
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure required directories exist."""
        cls.get_chroma_persist_path().mkdir(parents=True, exist_ok=True)
        Path(cls.AUDIO_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def validate_config() -> None:
    """Validate the current configuration."""
    config.validate()


def setup_directories() -> None:
    """Setup required directories."""
    config.ensure_directories()