"""
Document loader module for handling various document input formats.

This module provides functionality to load documents from different sources
and extract chunk-level data in a standardized format.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

from pydantic import BaseModel

# Import from parent modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from parsing.enhanced_document_parser import EnhancedParsedDocument, ContentChunk

logger = logging.getLogger(__name__)


class ChunkData(BaseModel):
    """Standardized chunk data structure."""
    
    content: str
    chunk_id: str
    section_title: Optional[str] = None
    section_level: Optional[int] = None
    references: List[str] = []
    metadata: Dict[str, Any] = {}


class DocumentData(BaseModel):
    """Standardized document data structure."""
    
    content: str
    chunks: List[ChunkData]
    doc_name: str
    parsed_doc: Optional[Any] = None
    metadata: Dict[str, Any] = {}


class DocumentLoader:
    """
    Handles loading documents from various sources.
    
    Supports:
    - EnhancedParsedDocument objects
    - JSON files with parsed content
    - Raw text files
    - Direct string content
    """
    
    def __init__(self):
        """Initialize the document loader."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_document(
        self,
        document_path: Union[str, Path, EnhancedParsedDocument, Dict]
    ) -> DocumentData:
        """
        Load a document from various sources.
        
        Args:
            document_path: Path to document, parsed object, or dict
            
        Returns:
            DocumentData with standardized structure
            
        Raises:
            TypeError: If document type is not supported
            FileNotFoundError: If file path doesn't exist
        """
        if isinstance(document_path, EnhancedParsedDocument):
            return self._load_from_parsed_document(document_path)
        elif isinstance(document_path, dict):
            return self._load_from_dict(document_path)
        elif isinstance(document_path, (str, Path)):
            return self._load_from_path(Path(document_path))
        else:
            raise TypeError(f"Unsupported document type: {type(document_path)}")
    
    def _load_from_parsed_document(
        self,
        parsed_doc: EnhancedParsedDocument
    ) -> DocumentData:
        """
        Load from an EnhancedParsedDocument object.
        
        Args:
            parsed_doc: EnhancedParsedDocument instance
            
        Returns:
            DocumentData with extracted information
        """
        chunks = []
        for chunk in parsed_doc.text_chunks:
            chunks.append(self._extract_chunk_data(chunk))
        
        return DocumentData(
            content=parsed_doc.content,
            chunks=chunks,
            doc_name=getattr(parsed_doc.metadata, "file_name", "document"),
            parsed_doc=parsed_doc,
            metadata={
                "file_path": getattr(parsed_doc.metadata, "file_path", None),
                "page_count": getattr(parsed_doc.metadata, "page_count", 0),
                "word_count": getattr(parsed_doc.metadata, "word_count", 0)
            }
        )
    
    def _load_from_dict(self, data: Dict) -> DocumentData:
        """
        Load from a dictionary (typically from JSON).
        
        Args:
            data: Dictionary with document data
            
        Returns:
            DocumentData with extracted information
        """
        chunks = []
        for i, chunk in enumerate(data.get("text_chunks", [])):
            if isinstance(chunk, dict):
                chunks.append(ChunkData(
                    content=chunk.get("content", ""),
                    chunk_id=chunk.get("id", f"chunk_{i}"),
                    section_title=chunk.get("section_title"),
                    section_level=chunk.get("section_level"),
                    references=chunk.get("references", []),
                    metadata=chunk.get("metadata", {})
                ))
        
        return DocumentData(
            content=data.get("content", ""),
            chunks=chunks,
            doc_name=data.get("metadata", {}).get("file_name", "document"),
            metadata=data.get("metadata", {})
        )
    
    def _load_from_path(self, file_path: Path) -> DocumentData:
        """
        Load from a file path.
        
        Args:
            file_path: Path to the file
            
        Returns:
            DocumentData with extracted information
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix == ".json":
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            doc_data = self._load_from_dict(data)
            doc_data.doc_name = file_path.name
            return doc_data
        else:
            # Load as raw text
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return DocumentData(
                content=content,
                chunks=[],  # No chunks for raw text
                doc_name=file_path.name,
                metadata={"file_path": str(file_path)}
            )
    
    def _extract_chunk_data(
        self,
        chunk: Union[ContentChunk, Dict, Any]
    ) -> ChunkData:
        """
        Extract standardized chunk data from various formats.
        
        Args:
            chunk: Chunk in various formats
            
        Returns:
            ChunkData with standardized structure
        """
        if hasattr(chunk, 'content'):
            # ContentChunk object
            return ChunkData(
                content=chunk.content,
                chunk_id=chunk.id if hasattr(chunk, 'id') else "unknown",
                section_title=chunk.section_title if hasattr(chunk, 'section_title') else None,
                section_level=chunk.section_level if hasattr(chunk, 'section_level') else None,
                references=chunk.references if hasattr(chunk, 'references') else [],
                metadata={
                    "word_count": chunk.word_count if hasattr(chunk, 'word_count') else 0,
                    "char_count": chunk.char_count if hasattr(chunk, 'char_count') else 0,
                    "chunk_number_in_section": chunk.chunk_number_in_section if hasattr(chunk, 'chunk_number_in_section') else None
                }
            )
        elif isinstance(chunk, dict):
            # Dictionary format
            return ChunkData(
                content=chunk.get("content", ""),
                chunk_id=chunk.get("id", chunk.get("chunk_id", "unknown")),
                section_title=chunk.get("section_title"),
                section_level=chunk.get("section_level"),
                references=chunk.get("references", []),
                metadata=chunk.get("metadata", {})
            )
        else:
            # Fallback for unknown formats
            return ChunkData(
                content=str(chunk),
                chunk_id="unknown",
                metadata={}
            )
    
    def split_content_into_sections(
        self,
        content: str,
        max_section_length: int = 3000
    ) -> List[ChunkData]:
        """
        Split raw content into sections for processing.
        
        Args:
            content: Raw text content
            max_section_length: Maximum characters per section
            
        Returns:
            List of ChunkData representing sections
        """
        sections = content.split('\n\n')
        chunks = []
        
        for i, section in enumerate(sections):
            if len(section) > 100:  # Skip very short sections
                chunks.append(ChunkData(
                    content=section[:max_section_length],
                    chunk_id=f"section_{i}",
                    metadata={"section_index": i}
                ))
        
        return chunks