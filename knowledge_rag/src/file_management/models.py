"""
Data models for file management and tracking
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib


class FileStatus(Enum):
    """File processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class FileFormat(Enum):
    """Supported file formats"""
    # Text formats
    TXT = "txt"
    MD = "md"
    CSV = "csv"
    
    # Document formats
    DOC = "doc"
    DOCX = "docx"
    PDF = "pdf"
    RTF = "rtf"
    
    # Presentation formats
    PPT = "ppt"
    PPTX = "pptx"
    
    # Spreadsheet formats
    XLS = "xls"
    XLSX = "xlsx"
    
    # Structured data formats
    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    YML = "yml"
    
    # Web formats
    HTML = "html"
    HTM = "htm"
    
    # Unknown format
    UNKNOWN = "unknown"
    
    @classmethod
    def from_extension(cls, extension: str) -> 'FileFormat':
        """Get file format from extension"""
        ext = extension.lower().lstrip('.')
        try:
            return cls(ext)
        except ValueError:
            return cls.UNKNOWN


@dataclass
class FileMetadata:
    """File metadata container"""
    file_path: Path
    file_name: str
    file_size: int
    file_format: FileFormat
    created_at: datetime
    modified_at: datetime
    file_hash: str
    encoding: Optional[str] = None
    language: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)


@dataclass
class ProcessedFile:
    """Represents a file that has been processed by the system"""
    id: str
    metadata: FileMetadata
    status: FileStatus
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
    episodes_created: int = 0
    entities_extracted: int = 0
    relationships_extracted: int = 0
    error_message: Optional[str] = None
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = self.generate_id()
    
    def generate_id(self) -> str:
        """Generate unique ID based on file path and hash"""
        content = f"{self.metadata.file_path}:{self.metadata.file_hash}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def mark_processing_started(self):
        """Mark file as started processing"""
        self.status = FileStatus.PROCESSING
        self.processing_started_at = datetime.now()
    
    def mark_processing_completed(self, episodes: int = 0, entities: int = 0, relationships: int = 0):
        """Mark file as successfully processed"""
        self.status = FileStatus.COMPLETED
        self.processing_completed_at = datetime.now()
        self.episodes_created = episodes
        self.entities_extracted = entities
        self.relationships_extracted = relationships
    
    def mark_processing_failed(self, error: str):
        """Mark file as failed processing"""
        self.status = FileStatus.FAILED
        self.processing_completed_at = datetime.now()
        self.error_message = error


@dataclass
class Episode:
    """Represents a parsed content episode from a document"""
    id: str
    content: str
    episode_type: str  # 'page', 'section', 'chunk', etc.
    sequence_number: int
    source_file_id: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = self.generate_id()
    
    def generate_id(self) -> str:
        """Generate unique episode ID"""
        content = f"{self.source_file_id}:{self.sequence_number}:{self.episode_type}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class ParsingResult:
    """Result of parsing a document"""
    file_id: str
    episodes: List[Episode]
    parsing_metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None
    
    @property
    def episode_count(self) -> int:
        return len(self.episodes) 