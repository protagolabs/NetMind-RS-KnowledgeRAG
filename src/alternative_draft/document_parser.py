"""
Document parser using Marker library for high-accuracy file conversion.

This module provides a simple interface to parse various document formats
(PDF, DOCX, PPTX, XLSX, HTML, EPUB, images) and convert them to structured markdown.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from pydantic import BaseModel, Field

# Import marker components
try:
    from marker.converters.pdf import PdfConverter
    from marker.converters.docx import DocxConverter
    from marker.converters.pptx import PptxConverter  
    from marker.converters.xlsx import XlsxConverter
    from marker.converters.html import HtmlConverter
    from marker.converters.epub import EpubConverter
    from marker.converters.image import ImageConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    MARKER_AVAILABLE = True
except ImportError:
    MARKER_AVAILABLE = False
    logging.warning("Marker library not installed. Install with: pip install marker-pdf[full]")


class ChunkConfiguration(BaseModel):
    """Configuration for document chunking."""
    
    max_chunk_size: int = Field(default=1000, description="Maximum chunk size in words")
    chunk_overlap: int = Field(default=100, description="Overlap between chunks in words")
    preserve_sections: bool = Field(default=True, description="Preserve section boundaries when chunking")
    min_chunk_size: int = Field(default=100, description="Minimum chunk size in words")


class ParsedDocument(BaseModel):
    """Structured output from document parsing."""
    
    content: str = Field(description="Full markdown content")
    chunks: List[Dict[str, Any]] = Field(default_factory=list, description="Document chunks")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    tables: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted tables")
    images: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted images")
    references: List[str] = Field(default_factory=list, description="Extracted references")
    equations: List[str] = Field(default_factory=list, description="Extracted equations")


class DocumentParser:
    """
    High-level document parser using Marker library.
    
    Supports parsing of PDF, DOCX, PPTX, XLSX, HTML, EPUB, and image files.
    """
    
    def __init__(
        self,
        chunk_config: Optional[ChunkConfiguration] = None,
        use_llm: bool = False,
        force_ocr: bool = False,
        output_format: str = "markdown"
    ):
        """
        Initialize the document parser.
        
        Args:
            chunk_config: Configuration for chunking documents
            use_llm: Whether to use LLM for enhanced accuracy (requires API key)
            force_ocr: Force OCR for all documents
            output_format: Output format (markdown, json, html, chunks)
        """
        self.chunk_config = chunk_config or ChunkConfiguration()
        self.use_llm = use_llm
        self.force_ocr = force_ocr
        self.output_format = output_format
        self.logger = logging.getLogger(__name__)
        
        # Initialize marker models if available
        if MARKER_AVAILABLE:
            self._initialize_marker()
        else:
            self.converters = {}
            self.logger.error("Marker library not available. Please install marker-pdf[full]")
    
    def _initialize_marker(self) -> None:
        """Initialize Marker converters and models."""
        try:
            # Create model dictionary (shared across converters)
            self.model_dict = create_model_dict()
            
            # Initialize converters for different file types
            self.converters = {
                '.pdf': PdfConverter(artifact_dict=self.model_dict),
                '.docx': DocxConverter(artifact_dict=self.model_dict),
                '.pptx': PptxConverter(artifact_dict=self.model_dict),
                '.xlsx': XlsxConverter(artifact_dict=self.model_dict),
                '.html': HtmlConverter(artifact_dict=self.model_dict),
                '.htm': HtmlConverter(artifact_dict=self.model_dict),
                '.epub': EpubConverter(artifact_dict=self.model_dict),
                '.png': ImageConverter(artifact_dict=self.model_dict),
                '.jpg': ImageConverter(artifact_dict=self.model_dict),
                '.jpeg': ImageConverter(artifact_dict=self.model_dict),
                '.webp': ImageConverter(artifact_dict=self.model_dict),
            }
            
            self.logger.info("Marker converters initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Marker: {e}")
            self.converters = {}
    
    def parse(self, file_path: Union[str, Path]) -> ParsedDocument:
        """
        Parse a document file and return structured content.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            ParsedDocument containing content, chunks, metadata, etc.
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type is not supported
            RuntimeError: If parsing fails
        """
        file_path = Path(file_path)
        
        # Validate file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check if Marker is available
        if not MARKER_AVAILABLE:
            raise RuntimeError("Marker library not installed. Install with: pip install marker-pdf[full]")
        
        # Get file extension
        file_ext = file_path.suffix.lower()
        
        # Check if file type is supported
        if file_ext not in self.converters:
            supported = ", ".join(self.converters.keys())
            raise ValueError(f"Unsupported file type: {file_ext}. Supported types: {supported}")
        
        try:
            # Parse with Marker
            self.logger.info(f"Parsing {file_path} with Marker")
            converter = self.converters[file_ext]
            
            # Configure converter options
            config = {
                "force_ocr": self.force_ocr,
                "output_format": self.output_format
            }
            
            # Add LLM configuration if enabled
            if self.use_llm:
                config["use_llm"] = True
                self.logger.info("Using LLM for enhanced accuracy")
            
            # Convert the document
            rendered = converter(str(file_path), config=config)
            
            # Extract text and metadata
            text, metadata, images = text_from_rendered(rendered)
            
            # Create parsed document
            parsed_doc = ParsedDocument(
                content=text,
                metadata=self._extract_metadata(file_path, metadata),
                images=self._process_images(images),
                tables=self._extract_tables(rendered),
                references=self._extract_references(text),
                equations=self._extract_equations(rendered)
            )
            
            # Generate chunks if needed
            if self.chunk_config.preserve_sections:
                parsed_doc.chunks = self._chunk_by_sections(text)
            else:
                parsed_doc.chunks = self._chunk_text(text)
            
            self.logger.info(f"Successfully parsed {file_path}")
            return parsed_doc
            
        except Exception as e:
            self.logger.error(f"Failed to parse {file_path}: {e}")
            raise RuntimeError(f"Failed to parse document: {e}")
    
    def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunk text into smaller segments.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        words = text.split()
        
        chunk_size = self.chunk_config.max_chunk_size
        overlap = self.chunk_config.chunk_overlap
        min_size = self.chunk_config.min_chunk_size
        
        start = 0
        chunk_id = 0
        
        while start < len(words):
            end = min(start + chunk_size, len(words))
            
            # Ensure minimum chunk size
            if end - start < min_size and start > 0:
                break
            
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'id': chunk_id,
                'content': chunk_text,
                'word_count': len(chunk_words),
                'char_count': len(chunk_text),
                'start_word': start,
                'end_word': end
            })
            
            start = end - overlap if end < len(words) else end
            chunk_id += 1
        
        return chunks
    
    def _chunk_by_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunk text while preserving section boundaries.
        
        Args:
            text: Markdown text with sections
            
        Returns:
            List of chunk dictionaries
        """
        import re
        
        chunks = []
        chunk_id = 0
        
        # Split by markdown headers
        sections = re.split(r'(^#{1,6}\s+.*$)', text, flags=re.MULTILINE)
        
        current_section = ""
        current_header = ""
        
        for i, part in enumerate(sections):
            if re.match(r'^#{1,6}\s+', part):
                # This is a header
                if current_section:
                    # Process previous section
                    section_chunks = self._chunk_text(current_section)
                    for chunk in section_chunks:
                        chunk['id'] = chunk_id
                        chunk['section'] = current_header
                        chunks.append(chunk)
                        chunk_id += 1
                
                current_header = part.strip()
                current_section = part + "\n"
            else:
                current_section += part
        
        # Process last section
        if current_section:
            section_chunks = self._chunk_text(current_section)
            for chunk in section_chunks:
                chunk['id'] = chunk_id
                chunk['section'] = current_header if current_header else "Main"
                chunks.append(chunk)
                chunk_id += 1
        
        return chunks
    
    def _extract_metadata(self, file_path: Path, marker_metadata: Dict) -> Dict[str, Any]:
        """
        Extract and enhance document metadata.
        
        Args:
            file_path: Path to the document
            marker_metadata: Metadata from Marker
            
        Returns:
            Enhanced metadata dictionary
        """
        import os
        from datetime import datetime
        
        metadata = {
            'file_name': file_path.name,
            'file_path': str(file_path.absolute()),
            'file_size': os.path.getsize(file_path),
            'file_type': file_path.suffix.lower(),
            'parsed_at': datetime.now().isoformat(),
            'parser': 'marker',
            'parser_version': marker_metadata.get('version', 'unknown')
        }
        
        # Add Marker-specific metadata
        if marker_metadata:
            metadata.update({
                'page_count': marker_metadata.get('page_count', 0),
                'word_count': marker_metadata.get('word_count', 0),
                'language': marker_metadata.get('language', 'unknown'),
                'title': marker_metadata.get('title', ''),
                'author': marker_metadata.get('author', ''),
                'creation_date': marker_metadata.get('creation_date', ''),
                'modification_date': marker_metadata.get('modification_date', '')
            })
        
        return metadata
    
    def _process_images(self, images: List) -> List[Dict[str, Any]]:
        """
        Process extracted images.
        
        Args:
            images: List of image data from Marker
            
        Returns:
            List of processed image dictionaries
        """
        processed_images = []
        
        for i, img in enumerate(images):
            processed_images.append({
                'id': i,
                'type': 'image',
                'caption': img.get('caption', ''),
                'alt_text': img.get('alt_text', ''),
                'page': img.get('page', 0),
                'bbox': img.get('bbox', []),
                'file_path': img.get('path', '')
            })
        
        return processed_images
    
    def _extract_tables(self, rendered_data: Any) -> List[Dict[str, Any]]:
        """
        Extract tables from rendered document.
        
        Args:
            rendered_data: Rendered document data from Marker
            
        Returns:
            List of table dictionaries
        """
        tables = []
        
        # Extract tables from Marker's rendered output
        if hasattr(rendered_data, 'tables'):
            for i, table in enumerate(rendered_data.tables):
                tables.append({
                    'id': i,
                    'type': 'table',
                    'content': table.get('content', ''),
                    'caption': table.get('caption', ''),
                    'rows': table.get('rows', 0),
                    'columns': table.get('columns', 0),
                    'page': table.get('page', 0)
                })
        
        return tables
    
    def _extract_references(self, text: str) -> List[str]:
        """
        Extract references from document text.
        
        Args:
            text: Document text
            
        Returns:
            List of references
        """
        import re
        
        references = []
        
        # Look for reference section
        ref_pattern = r'(?:References|Bibliography|Works Cited)\s*\n+(.*?)(?:\n\n|\Z)'
        ref_match = re.search(ref_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if ref_match:
            ref_section = ref_match.group(1)
            # Split by newlines and filter
            refs = [r.strip() for r in ref_section.split('\n') if r.strip()]
            references.extend(refs)
        
        # Also look for inline citations
        citation_pattern = r'\[(\d+)\]|\(([^)]+,\s*\d{4})\)'
        citations = re.findall(citation_pattern, text)
        
        return references
    
    def _extract_equations(self, rendered_data: Any) -> List[str]:
        """
        Extract mathematical equations from document.
        
        Args:
            rendered_data: Rendered document data from Marker
            
        Returns:
            List of equations
        """
        equations = []
        
        # Extract equations from Marker's rendered output
        if hasattr(rendered_data, 'equations'):
            for eq in rendered_data.equations:
                equations.append(eq.get('latex', ''))
        
        return equations
    
    def parse_batch(self, file_paths: List[Union[str, Path]]) -> List[ParsedDocument]:
        """
        Parse multiple documents in batch.
        
        Args:
            file_paths: List of file paths to parse
            
        Returns:
            List of ParsedDocument objects
        """
        results = []
        
        for file_path in file_paths:
            try:
                parsed = self.parse(file_path)
                results.append(parsed)
            except Exception as e:
                self.logger.error(f"Failed to parse {file_path}: {e}")
                # Create error document
                error_doc = ParsedDocument(
                    content="",
                    metadata={
                        'file_path': str(file_path),
                        'error': str(e),
                        'status': 'failed'
                    }
                )
                results.append(error_doc)
        
        return results


def parse_document(
    file_path: Union[str, Path],
    chunk_size: int = 1000,
    use_llm: bool = False,
    output_format: str = "markdown"
) -> ParsedDocument:
    """
    Convenience function to parse a single document.
    
    Args:
        file_path: Path to the document
        chunk_size: Maximum chunk size in words
        use_llm: Whether to use LLM for enhanced accuracy
        output_format: Output format (markdown, json, html, chunks)
        
    Returns:
        ParsedDocument with content, chunks, metadata, etc.
        
    Example:
        >>> doc = parse_document("research_paper.pdf", chunk_size=500)
        >>> print(f"Document has {len(doc.chunks)} chunks")
        >>> print(f"Found {len(doc.tables)} tables and {len(doc.images)} images")
    """
    config = ChunkConfiguration(max_chunk_size=chunk_size)
    parser = DocumentParser(chunk_config=config, use_llm=use_llm, output_format=output_format)
    return parser.parse(file_path)


if __name__ == "__main__":
    """Example usage of the document parser."""
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) < 2:
        print("Usage: python document_parser.py <file_path>")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    
    try:
        # Parse the document
        doc = parse_document(file_path, chunk_size=500)
        
        # Display results
        print(f"\nDocument: {file_path.name}")
        print(f"Content length: {len(doc.content)} characters")
        print(f"Chunks: {len(doc.chunks)}")
        print(f"Tables: {len(doc.tables)}")
        print(f"Images: {len(doc.images)}")
        print(f"References: {len(doc.references)}")
        print(f"Equations: {len(doc.equations)}")
        
        # Show first chunk
        if doc.chunks:
            print(f"\nFirst chunk ({doc.chunks[0]['word_count']} words):")
            print(doc.chunks[0]['content'][:200] + "...")
        
        # Show metadata
        print(f"\nMetadata:")
        for key, value in doc.metadata.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)