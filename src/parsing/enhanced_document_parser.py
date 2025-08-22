"""
Enhanced document parser with special handling for tables and images.

This module extracts tables and images as separate chunks and replaces them
with references in the main text to maintain content integrity.
"""

import logging
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
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
    logger.info("Marker library loaded successfully")
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
    parser_type: str = Field(default="enhanced_marker", description="Parser used")
    table_count: int = Field(default=0, description="Number of tables extracted")
    image_count: int = Field(default=0, description="Number of images extracted")


class ContentChunk(BaseModel):
    """A chunk of parsed document content."""
    
    id: str = Field(description="Unique chunk ID")
    chunk_type: str = Field(description="Type: text, table, or image")
    content: str = Field(description="Chunk content or description")
    word_count: int = Field(description="Words in chunk")
    char_count: int = Field(description="Characters in chunk")
    page_start: Optional[int] = Field(default=None, description="Starting page")
    page_end: Optional[int] = Field(default=None, description="Ending page")
    references: List[str] = Field(default_factory=list, description="Referenced table/image IDs")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    section_title: Optional[str] = Field(default=None, description="Title of the section this chunk belongs to")
    section_level: Optional[int] = Field(default=None, description="Hierarchical level of the section (1-6)")
    chunk_number_in_section: Optional[int] = Field(default=None, description="Chunk number within the section")


class TableChunk(ContentChunk):
    """Special chunk for table content."""
    
    chunk_type: str = Field(default="table", description="Type is always table")
    table_format: str = Field(default="markdown", description="Format of table content")
    rows: int = Field(default=0, description="Number of rows")
    columns: int = Field(default=0, description="Number of columns")
    caption: Optional[str] = Field(default=None, description="Table caption if available")


class ImageChunk(ContentChunk):
    """Special chunk for image content."""
    
    chunk_type: str = Field(default="image", description="Type is always image")
    image_path: Optional[str] = Field(default=None, description="Path to extracted image")
    caption: Optional[str] = Field(default=None, description="Image caption if available")
    alt_text: Optional[str] = Field(default=None, description="Alternative text")


class EnhancedParsedDocument(BaseModel):
    """Document with separated content, table, and image chunks."""
    
    content: str = Field(description="Full text with table/image placeholders")
    text_chunks: List[ContentChunk] = Field(default_factory=list, description="Text chunks")
    table_chunks: List[TableChunk] = Field(default_factory=list, description="Table chunks")
    image_chunks: List[ImageChunk] = Field(default_factory=list, description="Image chunks")
    metadata: DocumentMetadata = Field(description="Document metadata")
    sections: List[Dict[str, Any]] = Field(default_factory=list, description="Document sections")
    chunk_index: Dict[str, str] = Field(default_factory=dict, description="ID to chunk type mapping")


class EnhancedDocumentParser:
    """
    Enhanced parser that extracts tables and images as separate chunks.
    
    This parser:
    1. Extracts tables and images as complete, separate chunks
    2. Replaces them in text with reference placeholders
    3. Creates text chunks that reference table/image chunks
    4. Maintains relationships between chunks
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        extract_images: bool = True,
        extract_tables: bool = True,
        use_section_chunking: bool = False,
        section_chunk_size: int = 2000,
        section_chunk_overlap: int = 200
    ):
        """
        Initialize the enhanced parser.
        
        Args:
            chunk_size: Maximum words per text chunk (used when use_section_chunking=False)
            chunk_overlap: Word overlap between text chunks (used when use_section_chunking=False)
            extract_images: Whether to extract images separately
            extract_tables: Whether to extract tables separately
            use_section_chunking: Whether to use section-based chunking
            section_chunk_size: Maximum words per chunk within a section (used when use_section_chunking=True)
            section_chunk_overlap: Word overlap between chunks within a section (used when use_section_chunking=True)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.use_section_chunking = use_section_chunking
        self.section_chunk_size = section_chunk_size
        self.section_chunk_overlap = section_chunk_overlap
        
        if not MARKER_AVAILABLE:
            raise RuntimeError("Marker not installed. Run: pip install marker-pdf")
        
        # Initialize Marker models
        logger.info("Initializing Marker models...")
        self.model_dict = create_model_dict()
        self.converter = PdfConverter(artifact_dict=self.model_dict)
        logger.info("Marker models loaded")
    
    def parse_pdf(self, file_path: Union[str, Path]) -> EnhancedParsedDocument:
        """
        Parse a PDF with enhanced table/image extraction.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            EnhancedParsedDocument with separated chunks
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() != '.pdf':
            raise ValueError(f"Expected PDF file, got: {file_path.suffix}")
        
        logger.info(f"Processing {file_path.name}...")
        
        try:
            # Convert PDF using Marker
            rendered = self.converter(str(file_path))
            
            # Extract content
            text, _, images = text_from_rendered(rendered)
            
            # Extract tables and images, get modified text
            table_chunks, image_chunks, modified_text = self._extract_special_content(
                text, images, rendered
            )
            
            # Create metadata
            import os
            metadata = DocumentMetadata(
                file_name=file_path.name,
                file_path=str(file_path.absolute()),
                file_size=os.path.getsize(file_path),
                word_count=len(modified_text.split()),
                char_count=len(modified_text),
                parse_time=datetime.now().isoformat(),
                table_count=len(table_chunks),
                image_count=len(image_chunks)
            )
            
            # Extract page count if available
            if hasattr(rendered, 'metadata') and rendered.metadata:
                metadata.page_count = rendered.metadata.get('page_count', 0)
            
            # Extract sections
            sections = self._extract_sections(modified_text)
            
            # Create text chunks with references
            if self.use_section_chunking:
                text_chunks = self._create_section_based_chunks(
                    modified_text, sections, table_chunks, image_chunks,
                    self.section_chunk_size, self.section_chunk_overlap
                )
            else:
                text_chunks = self._create_text_chunks(modified_text, table_chunks, image_chunks)
            
            # Create chunk index
            chunk_index = {}
            for chunk in text_chunks:
                chunk_index[chunk.id] = "text"
            for chunk in table_chunks:
                chunk_index[chunk.id] = "table"
            for chunk in image_chunks:
                chunk_index[chunk.id] = "image"
            
            return EnhancedParsedDocument(
                content=modified_text,
                text_chunks=text_chunks,
                table_chunks=table_chunks,
                image_chunks=image_chunks,
                metadata=metadata,
                sections=sections,
                chunk_index=chunk_index
            )
            
        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            raise RuntimeError(f"Failed to parse PDF: {e}")
    
    def _extract_special_content(
        self,
        text: str,
        images: List,
        rendered_data: Any
    ) -> Tuple[List[TableChunk], List[ImageChunk], str]:
        """
        Extract tables and images, replacing them with placeholders.
        
        Args:
            text: Original document text
            images: Extracted images from Marker
            rendered_data: Rendered data from Marker
            
        Returns:
            Tuple of (table_chunks, image_chunks, modified_text)
        """
        table_chunks = []
        image_chunks = []
        modified_text = text
        
        # Extract and replace tables
        if self.extract_tables:
            table_chunks, modified_text = self._extract_and_replace_tables(modified_text)
        
        # Extract and replace images
        if self.extract_images and images:
            image_chunks = self._process_images(images)
            # Note: Image replacement in text would require position info from Marker
            # For now, we just extract them without modifying text
        
        return table_chunks, image_chunks, modified_text
    
    def _extract_and_replace_tables(self, text: str) -> Tuple[List[TableChunk], str]:
        """
        Extract tables and replace with reference placeholders.
        
        Note: Marker converts tables to markdown format during PDF processing.
        For more structured table extraction, use converter.build_document()
        and access table blocks directly.
        
        Args:
            text: Document text with markdown-formatted tables from Marker
            
        Returns:
            Tuple of (table_chunks, modified_text)
        """
        table_chunks = []
        modified_text = text
        
        # Pattern to match markdown tables that Marker produces
        # Marker converts PDF tables to markdown with header separator row
        table_pattern = r'(\n\|[^\n]+\|\n\|[-:\s|]+\|\n(?:\|[^\n]+\|\n)+)'
        
        # Find all tables
        matches = list(re.finditer(table_pattern, text))
        
        # Process tables in reverse to maintain positions
        for i, match in enumerate(reversed(matches)):
            table_idx = len(matches) - i - 1
            table_id = f"table_{table_idx:03d}"
            table_content = match.group(1).strip()
            
            # Parse table structure
            lines = table_content.split('\n')
            rows = len([l for l in lines if l.strip() and '|---' not in l])
            cols = len(lines[0].split('|')) - 2 if lines else 0
            
            # Extract caption if present (look for text before table)
            caption = None
            pre_context = text[max(0, match.start() - 200):match.start()]
            caption_match = re.search(r'Table\s+\d+[:\.]?\s*([^\n]+)', pre_context)
            if caption_match:
                caption = caption_match.group(1).strip()
            
            # Create table chunk
            table_chunk = TableChunk(
                id=table_id,
                content=table_content,
                word_count=len(table_content.split()),
                char_count=len(table_content),
                rows=rows,
                columns=cols,
                caption=caption,
                metadata={
                    "position": match.start(),
                    "original_length": len(table_content),
                    "format": "markdown"  # Marker converts to markdown
                }
            )
            table_chunks.append(table_chunk)
            
            # Create placeholder with reference
            placeholder = f"\n[TABLE_REF:{table_id}]"
            if caption:
                placeholder += f" {caption}"
            placeholder += "\n"
            
            # Replace table with placeholder
            modified_text = (
                modified_text[:match.start()] +
                placeholder +
                modified_text[match.end():]
            )
        
        # Reverse table_chunks to maintain original order
        table_chunks.reverse()
        
        logger.info(f"Extracted {len(table_chunks)} markdown tables from Marker output")
        return table_chunks, modified_text
    
    def _process_images(self, images: List) -> List[ImageChunk]:
        """
        Process extracted images into chunks.
        
        Args:
            images: List of image data from Marker
            
        Returns:
            List of ImageChunk objects
        """
        image_chunks = []
        
        for i, img in enumerate(images):
            image_id = f"image_{i:03d}"
            
            # Extract image metadata
            caption = img.get('caption', '') if isinstance(img, dict) else ''
            alt_text = img.get('alt_text', '') if isinstance(img, dict) else ''
            
            # Create description
            description = f"Image {i}"
            if caption:
                description = caption
            elif alt_text:
                description = alt_text
            
            image_chunk = ImageChunk(
                id=image_id,
                content=description,
                word_count=len(description.split()),
                char_count=len(description),
                caption=caption,
                alt_text=alt_text,
                metadata={
                    "index": i,
                    "raw_data": img if isinstance(img, dict) else {"raw": str(img)}
                }
            )
            image_chunks.append(image_chunk)
        
        logger.info(f"Extracted {len(image_chunks)} images")
        return image_chunks
    
    def _create_section_based_chunks(
        self,
        text: str,
        sections: List[Dict[str, Any]],
        table_chunks: List[TableChunk],
        image_chunks: List[ImageChunk],
        section_chunk_size: int = 2000,
        section_chunk_overlap: int = 200
    ) -> List[ContentChunk]:
        """
        Create text chunks based on document sections.
        
        Each section is split into chunks of section_chunk_size words.
        Maintains section boundaries and metadata.
        
        Args:
            text: Modified text with placeholders
            sections: List of section dictionaries with position, title, level
            table_chunks: List of table chunks
            image_chunks: List of image chunks
            section_chunk_size: Maximum words per chunk within a section
            section_chunk_overlap: Word overlap between chunks within a section
            
        Returns:
            List of text ContentChunk objects with section metadata
        """
        chunks = []
        table_refs = {t.id for t in table_chunks}
        image_refs = {i.id for i in image_chunks}
        
        # Sort sections by position
        sorted_sections = sorted(sections, key=lambda x: x['position'])
        
        # Add an ending position for the last section
        for i in range(len(sorted_sections)):
            if i < len(sorted_sections) - 1:
                sorted_sections[i]['end_position'] = sorted_sections[i + 1]['position']
            else:
                sorted_sections[i]['end_position'] = len(text)
        
        # If no sections, treat entire document as one section
        if not sorted_sections:
            sorted_sections = [{
                'title': 'Document',
                'level': 1,
                'position': 0,
                'end_position': len(text)
            }]
        
        global_chunk_idx = 0
        
        # Process each section
        for section in sorted_sections:
            section_text = text[section['position']:section['end_position']]
            section_words = section_text.split()
            
            # Skip empty sections
            if not section_words:
                continue
            
            # Split section into chunks
            section_chunk_num = 0
            start = 0
            
            while start < len(section_words):
                end = min(start + section_chunk_size, len(section_words))
                chunk_words = section_words[start:end]
                chunk_text = ' '.join(chunk_words)
                
                # Find references in this chunk
                references = []
                for table_id in table_refs:
                    if f"TABLE_REF:{table_id}" in chunk_text:
                        references.append(table_id)
                for image_id in image_refs:
                    if f"IMAGE_REF:{image_id}" in chunk_text:
                        references.append(image_id)
                
                chunk = ContentChunk(
                    id=f"text_{global_chunk_idx:03d}",
                    chunk_type="text",
                    content=chunk_text,
                    word_count=len(chunk_words),
                    char_count=len(chunk_text),
                    references=references,
                    section_title=section.get('title', 'Unknown'),
                    section_level=section.get('level', 1),
                    chunk_number_in_section=section_chunk_num,
                    metadata={
                        "start_word_in_section": start,
                        "end_word_in_section": end,
                        "section_position": section['position'],
                        "is_last_chunk_in_section": end >= len(section_words)
                    }
                )
                chunks.append(chunk)
                
                # Move forward with overlap
                if end < len(section_words):
                    start = end - section_chunk_overlap
                else:
                    start = end
                    
                section_chunk_num += 1
                global_chunk_idx += 1
        
        logger.info(f"Created {len(chunks)} section-based text chunks from {len(sorted_sections)} sections")
        return chunks
    
    def _create_text_chunks(
        self,
        text: str,
        table_chunks: List[TableChunk],
        image_chunks: List[ImageChunk]
    ) -> List[ContentChunk]:
        """
        Create text chunks with references to tables/images.
        
        Args:
            text: Modified text with placeholders
            table_chunks: List of table chunks
            image_chunks: List of image chunks
            
        Returns:
            List of text ContentChunk objects
        """
        chunks = []
        words = text.split()
        
        # Create mapping of references
        table_refs = {t.id for t in table_chunks}
        image_refs = {i.id for i in image_chunks}
        
        start = 0
        chunk_idx = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            
            # Find references in this chunk
            references = []
            for table_id in table_refs:
                if f"TABLE_REF:{table_id}" in chunk_text:
                    references.append(table_id)
            for image_id in image_refs:
                if f"IMAGE_REF:{image_id}" in chunk_text:
                    references.append(image_id)
            
            chunk = ContentChunk(
                id=f"text_{chunk_idx:03d}",
                chunk_type="text",
                content=chunk_text,
                word_count=len(chunk_words),
                char_count=len(chunk_text),
                references=references,
                metadata={
                    "start_word": start,
                    "end_word": end
                }
            )
            chunks.append(chunk)
            
            # Move forward with overlap
            start = end - self.chunk_overlap if end < len(words) else end
            chunk_idx += 1
        
        logger.info(f"Created {len(chunks)} text chunks")
        return chunks
    
    def _extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract sections from markdown text.
        
        Args:
            text: Markdown text
            
        Returns:
            List of section dictionaries
        """
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
    
    def save_to_json(
        self,
        parsed_doc: EnhancedParsedDocument,
        output_path: Union[str, Path]
    ) -> None:
        """
        Save parsed document to JSON file.
        
        Args:
            parsed_doc: EnhancedParsedDocument object
            output_path: Path for output JSON file
        """
        output_path = Path(output_path)
        
        # Convert to dictionary
        doc_dict = parsed_doc.model_dump()
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(doc_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved parsed document to {output_path}")


def parse_pdf_enhanced(
    pdf_path: Union[str, Path],
    chunk_size: int = 1000,
    save_json: bool = True,
    extract_tables: bool = True,
    extract_images: bool = True,
    use_section_chunking: bool = False,
    section_chunk_size: int = 2000,
    section_chunk_overlap: int = 200
) -> EnhancedParsedDocument:
    """
    Parse a PDF with enhanced table/image extraction.
    
    Args:
        pdf_path: Path to PDF file
        chunk_size: Maximum words per text chunk (used when use_section_chunking=False)
        save_json: Whether to save output as JSON
        extract_tables: Whether to extract tables separately
        extract_images: Whether to extract images separately
        use_section_chunking: Whether to use section-based chunking
        section_chunk_size: Maximum words per chunk within a section (used when use_section_chunking=True)
        section_chunk_overlap: Word overlap between chunks within a section (used when use_section_chunking=True)
        
    Returns:
        EnhancedParsedDocument object
    """
    parser = EnhancedDocumentParser(
        chunk_size=chunk_size,
        extract_tables=extract_tables,
        extract_images=extract_images,
        use_section_chunking=use_section_chunking,
        section_chunk_size=section_chunk_size,
        section_chunk_overlap=section_chunk_overlap
    )
    parsed = parser.parse_pdf(pdf_path)
    
    if save_json:
        pdf_path = Path(pdf_path)
        json_path = pdf_path.with_suffix('.enhanced.json')
        parser.save_to_json(parsed, json_path)
        logger.info(f"Enhanced JSON output saved to: {json_path}")
    
    return parsed


if __name__ == "__main__":
    """Example usage of the enhanced parser."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python enhanced_document_parser.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    
    try:
        # Parse with enhanced extraction
        doc = parse_pdf_enhanced(pdf_path, chunk_size=500)
        
        # Display results
        print(f"\nDocument: {pdf_path.name}")
        print(f"Text chunks: {len(doc.text_chunks)}")
        print(f"Table chunks: {len(doc.table_chunks)}")
        print(f"Image chunks: {len(doc.image_chunks)}")
        
        # Show first text chunk with references
        if doc.text_chunks:
            first_chunk = doc.text_chunks[0]
            print(f"\nFirst text chunk ({first_chunk.word_count} words):")
            print(first_chunk.content[:200] + "...")
            if first_chunk.references:
                print(f"References: {first_chunk.references}")
        
        # Show first table chunk
        if doc.table_chunks:
            first_table = doc.table_chunks[0]
            print(f"\nFirst table ({first_table.id}):")
            print(f"  Rows: {first_table.rows}, Columns: {first_table.columns}")
            if first_table.caption:
                print(f"  Caption: {first_table.caption}")
            print(f"  Content preview: {first_table.content[:100]}...")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)