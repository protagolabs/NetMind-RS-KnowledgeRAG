"""
Configuration settings for Knowledge RAG system
"""

import os
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class FileManagerConfig:
    """Configuration for the file management system"""
    root_folder: str = os.getenv("KNOWLEDGE_RAG_ROOT_FOLDER", "./documents")
    db_path: str = os.getenv("KNOWLEDGE_RAG_DB_PATH", "knowledge_rag.db")
    supported_extensions: Optional[List[str]] = None
    max_file_size_mb: int = int(os.getenv("KNOWLEDGE_RAG_MAX_FILE_SIZE_MB", "100"))
    recursive_scan: bool = os.getenv("KNOWLEDGE_RAG_RECURSIVE_SCAN", "true").lower() == "true"
    
    def __post_init__(self):
        # Ensure root folder exists
        Path(self.root_folder).mkdir(parents=True, exist_ok=True)


@dataclass
class ParsingConfig:
    """Configuration for document parsing"""
    max_chunk_size: int = int(os.getenv("KNOWLEDGE_RAG_MAX_CHUNK_SIZE", "5000"))
    csv_rows_per_episode: int = int(os.getenv("KNOWLEDGE_RAG_CSV_ROWS_PER_EPISODE", "10"))
    paragraphs_per_episode: int = int(os.getenv("KNOWLEDGE_RAG_PARAGRAPHS_PER_EPISODE", "8"))
    min_paragraph_length: int = int(os.getenv("KNOWLEDGE_RAG_MIN_PARAGRAPH_LENGTH", "50"))
    detect_language: bool = os.getenv("KNOWLEDGE_RAG_DETECT_LANGUAGE", "true").lower() == "true"
    
    # PDF specific settings
    extract_images: bool = os.getenv("KNOWLEDGE_RAG_EXTRACT_IMAGES", "true").lower() == "true"
    extract_tables: bool = os.getenv("KNOWLEDGE_RAG_EXTRACT_TABLES", "true").lower() == "true"
    detect_cross_page: bool = os.getenv("KNOWLEDGE_RAG_DETECT_CROSS_PAGE", "true").lower() == "true"


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    level: str = os.getenv("KNOWLEDGE_RAG_LOG_LEVEL", "INFO")
    log_file: Optional[str] = os.getenv("KNOWLEDGE_RAG_LOG_FILE", None)
    log_format: str = os.getenv(
        "KNOWLEDGE_RAG_LOG_FORMAT", 
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


@dataclass
class APIConfig:
    """Configuration for API server"""
    host: str = os.getenv("KNOWLEDGE_RAG_HOST", "127.0.0.1")
    port: int = int(os.getenv("KNOWLEDGE_RAG_PORT", "8000"))
    reload: bool = os.getenv("KNOWLEDGE_RAG_RELOAD", "false").lower() == "true"
    cors_origins: List[str] = os.getenv("KNOWLEDGE_RAG_CORS_ORIGINS", "*").split(",")


class Config:
    """Main configuration class"""
    
    def __init__(self):
        self.file_manager = FileManagerConfig()
        self.parsing = ParsingConfig()
        self.logging = LoggingConfig()
        self.api = APIConfig()
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables"""
        return cls()
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Create configuration from file (future implementation)"""
        # This could load from YAML, JSON, etc.
        # For now, just return default config
        return cls()


# Global configuration instance
config = Config.from_env() 