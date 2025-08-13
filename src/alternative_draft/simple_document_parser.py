"""
Simple document parser using Marker library for local PDF conversion.

This module runs entirely on your local hardware (GPU/CPU) without requiring external APIs.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import marker
try:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    MARKER_AVAILABLE = True
    logger.info("Marker library loaded successfully - will run on local hardware")
except ImportError:
    MARKER_AVAILABLE = False
    logger.error("Marker library not installed. Install with: pip install marker-pdf")


class DocumentMetadata(BaseModel):
    """Document metadata extracted during parsing."""
    
    file_name: str = Field(description="Name of the file")
    file_path: str = Field(description="Full path to the file")
    file_size: int = Field(description="File size in bytes")
    page_count: int = Field(default=0, description="Number of pages")
    word_count: int = Field(default=0, description="Total word count")
    char_count: int = Field(default=0, description="Total character count")
    parse_time: str = Field(description="Timestamp of parsing")
    parser_type: str = Field(default="marker", description="Parser used")


class ParsedChunk(BaseModel):
    """A chunk of parsed document."""
    
    id: int = Field(description="Chunk ID")
    content: str = Field(description="Chunk text content")
    word_count: int = Field(description="Words in chunk")
    char_count: int = Field(description="Characters in chunk")
    page_start: Optional[int] = Field(default=None, description="Starting page")
    page_end: Optional[int] = Field(default=None, description="Ending page")


class ParsedDocument(BaseModel):
    """Complete parsed document with all extracted information."""
    
    content: str = Field(description="Full markdown content")
    chunks: List[ParsedChunk] = Field(default_factory=list, description="Document chunks")
    metadata: DocumentMetadata = Field(description="Document metadata")
    images: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted images")
    tables: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted tables")
    sections: List[Dict[str, Any]] = Field(default_factory=list, description="Document sections")


class SimpleDocumentParser:
    """
    Local document parser using Marker - runs entirely on your hardware.
    
    No external API calls required - models are downloaded and run locally.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Initialize the parser.
        
        Args:
            chunk_size: Maximum words per chunk
            chunk_overlap: Word overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if not MARKER_AVAILABLE:
            raise RuntimeError("Marker not installed. Run: pip install marker-pdf")
        
        # Initialize Marker models (downloaded and run locally)
        logger.info("Initializing Marker models on local hardware...")
        self.model_dict = create_model_dict()
        self.converter = PdfConverter(artifact_dict=self.model_dict)
        logger.info("Marker models loaded - ready for local processing")
    
    def parse_pdf(self, file_path: Union[str, Path]) -> ParsedDocument:
        """
        Parse a PDF file using local Marker models.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            ParsedDocument with all extracted content
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() != '.pdf':
            raise ValueError(f"Expected PDF file, got: {file_path.suffix}")
        
        logger.info(f"Processing {file_path.name} locally...")
        
        try:
            # Convert PDF using local models
            rendered = self.converter(str(file_path))
            
            # Extract content
            text, _, images = text_from_rendered(rendered)
            
            # Create metadata
            import os
            from datetime import datetime
            
            metadata = DocumentMetadata(
                file_name=file_path.name,
                file_path=str(file_path.absolute()),
                file_size=os.path.getsize(file_path),
                word_count=len(text.split()),
                char_count=len(text),
                parse_time=datetime.now().isoformat(),
                parser_type="marker_local"
            )
            
            # Extract page count if available
            if hasattr(rendered, 'metadata') and rendered.metadata:
                metadata.page_count = rendered.metadata.get('page_count', 0)
            
            # Create chunks
            chunks = self._create_chunks(text)
            
            # Extract sections
            sections = self._extract_sections(text)
            
            # Extract tables
            tables = self._extract_tables(text)
            
            # Process images
            processed_images = []
            if images:
                for i, img in enumerate(images):
                    processed_images.append({
                        'id': i,
                        'type': 'image',
                        'data': img if isinstance(img, dict) else {'raw': str(img)}
                    })
            
            return ParsedDocument(
                content=text,
                chunks=chunks,
                metadata=metadata,
                images=processed_images,
                tables=tables,
                sections=sections
            )
            
        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            raise RuntimeError(f"Failed to parse PDF: {e}")
    
    def _create_chunks(self, text: str) -> List[ParsedChunk]:
        """
        Create text chunks with overlap.
        
        Args:
            text: Full document text
            
        Returns:
            List of ParsedChunk objects
        """
        words = text.split()
        chunks = []
        
        start = 0
        chunk_id = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            
            chunk = ParsedChunk(
                id=chunk_id,
                content=chunk_text,
                word_count=len(chunk_words),
                char_count=len(chunk_text)
            )
            chunks.append(chunk)
            
            # Move forward with overlap
            start = end - self.chunk_overlap if end < len(words) else end
            chunk_id += 1
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def _extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract sections from markdown text.
        
        Args:
            text: Markdown text
            
        Returns:
            List of section dictionaries
        """
        import re
        
        sections = []
        
        # Find all headers
        header_pattern = r'^(#{1,6})\s+(.+)$'
        
        for match in re.finditer(header_pattern, text, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            position = match.start()
            
            sections.append({
                'level': level,
                'title': title,
                'position': position
            })
        
        logger.info(f"Found {len(sections)} sections")
        return sections
    
    def _extract_tables(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract markdown tables from text.
        
        Args:
            text: Markdown text
            
        Returns:
            List of table dictionaries
        """
        import re
        
        tables = []
        
        # Simple markdown table pattern
        table_pattern = r'(\|.*?\|(?:\n\|.*?\|)+)'
        
        for i, match in enumerate(re.finditer(table_pattern, text, re.MULTILINE)):
            table_text = match.group(1)
            lines = table_text.strip().split('\n')
            
            # Count rows and estimate columns
            rows = len(lines)
            cols = len(lines[0].split('|')) - 2 if lines else 0
            
            tables.append({
                'id': i,
                'content': table_text,
                'rows': rows,
                'columns': cols,
                'position': match.start()
            })
        
        logger.info(f"Found {len(tables)} tables")
        return tables
    
    def save_to_json(self, parsed_doc: ParsedDocument, output_path: Union[str, Path]) -> None:
        """
        Save parsed document to JSON file.
        
        Args:
            parsed_doc: ParsedDocument object
            output_path: Path for output JSON file
        """
        output_path = Path(output_path)
        
        # Convert to dictionary
        doc_dict = parsed_doc.model_dump()
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(doc_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved parsed document to {output_path}")


def parse_pdf_locally(
    pdf_path: Union[str, Path],
    chunk_size: int = 1000,
    save_json: bool = True
) -> ParsedDocument:
    """
    Convenience function to parse a PDF locally.
    
    Args:
        pdf_path: Path to PDF file
        chunk_size: Maximum words per chunk
        save_json: Whether to save output as JSON
        
    Returns:
        ParsedDocument object
    """
    parser = SimpleDocumentParser(chunk_size=chunk_size)
    parsed = parser.parse_pdf(pdf_path)
    
    if save_json:
        pdf_path = Path(pdf_path)
        json_path = pdf_path.with_suffix('.parsed.json')
        parser.save_to_json(parsed, json_path)
        logger.info(f"JSON output saved to: {json_path}")
    
    return parsed