"""
Base Parser - Abstract base class for all document parsers
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any
import hashlib
import logging

from ..models import FileMetadata, Episode, ParsingResult, FileFormat


class BaseParser(ABC):
    """Abstract base class for all document parsers"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def can_parse(self, file_format: FileFormat) -> bool:
        """Check if this parser can handle the given file format"""
        pass
    
    @abstractmethod
    def parse(self, file_path: Path, metadata: FileMetadata) -> ParsingResult:
        """Parse the document and return episodes"""
        pass
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file content"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating file hash: {e}")
            return ""
    
    def detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding"""
        try:
            import chardet
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                return result.get('encoding', 'utf-8')
        except ImportError:
            self.logger.warning("chardet not available, defaulting to utf-8")
            return 'utf-8'
        except Exception as e:
            self.logger.error(f"Error detecting encoding: {e}")
            return 'utf-8'
    
    def detect_language(self, text: str) -> str:
        """Detect document language"""
        try:
            from langdetect import detect
            if len(text) > 100:  # Need sufficient text for detection
                return detect(text)
            return 'unknown'
        except ImportError:
            self.logger.warning("langdetect not available")
            return 'unknown'
        except Exception as e:
            self.logger.debug(f"Could not detect language: {e}")
            return 'unknown'
    
    def create_episode(self, 
                      content: str, 
                      episode_type: str, 
                      sequence_number: int, 
                      source_file_id: str, 
                      metadata: Dict[str, Any] = None) -> Episode:
        """Create an episode from parsed content"""
        from datetime import datetime
        
        return Episode(
            id="",  # Will be generated in __post_init__
            content=content,
            episode_type=episode_type,
            sequence_number=sequence_number,
            source_file_id=source_file_id,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove special characters that might cause issues
        text = text.replace('\x00', '')  # Remove null bytes
        text = text.replace('\ufeff', '')  # Remove BOM
        
        return text.strip()
    
    def chunk_large_content(self, content: str, max_chunk_size: int = 5000) -> List[str]:
        """Split large content into smaller chunks"""
        if len(content) <= max_chunk_size:
            return [content]
        
        chunks = []
        sentences = content.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 2 <= max_chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks 