"""High-accuracy file parsing using marker library for document conversion."""

import re
import json
import subprocess
import tempfile
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

# Import existing dependencies for fallback processing
try:
    import pymupdf
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    from pptx import Presentation
    HAS_PPTX = True
except ImportError:
    HAS_PPTX = False

try:
    import openpyxl
    import pandas as pd
    HAS_EXCEL = True
except ImportError:
    HAS_EXCEL = False

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for intelligent chunking based on document structure."""
    level_1_max: int = 4000      # Document sections (based on research: full sections)
    level_2_max: int = 1100      # Major sections (Methods/Results: ~1,126 words)  
    level_3_max: int = 550       # Subsections (Introduction: ~553 words)
    level_4_max: int = 275       # Minor sections (paragraph clusters)
    table_threshold: int = 500   # Words - extract tables larger than this
    overlap: int = 100           # Word overlap between chunks


class FileParser:
    """High-accuracy file parsing using marker library with intelligent fallbacks."""
    
    def __init__(self, chunk_config: Optional[ChunkConfig] = None):
        """Initialize the file parser with optional chunk configuration."""
        self.chunk_config = chunk_config or ChunkConfig()
        self.marker_available = self._check_marker_installation()
        self.markdown_converter = MarkdownConverter(self.chunk_config)
        self.chunk_processor = ChunkProcessor(self.chunk_config.level_2_max)  # Default to level 2 for backward compatibility
    
    def _check_marker_installation(self) -> bool:
        """Check if marker is installed and available."""
        try:
            result = subprocess.run(['marker', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("Marker library is available for high-accuracy PDF processing")
                return True
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        logger.warning("Marker not found. Install with: pip install marker-pdf")
        logger.info("Falling back to alternative parsers for document processing")
        return False
    
    def parse(self, file_path: Path) -> Dict[str, Any]:
        """Parse a file and return structured content.
        
        Args:
            file_path: Path to the file to parse.
            
        Returns:
            Dictionary containing parsed content and metadata.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_type = file_path.suffix.lower()
        
        # Route to appropriate parser
        if file_type == '.pdf':
            return self._parse_pdf(file_path)
        elif file_type in ['.docx', '.doc']:
            return self._parse_docx(file_path)
        elif file_type in ['.pptx', '.ppt']:
            return self._parse_pptx(file_path)
        elif file_type == '.txt':
            return self._parse_txt(file_path)
        elif file_type in ['.xlsx', '.xls']:
            return self._parse_excel(file_path)
        elif file_type == '.md':
            return self._parse_markdown(file_path)
        else:
            # Try to process as text
            logger.warning(f"Unsupported file type {file_type}, attempting text processing")
            return self._parse_txt(file_path)
    
    def _parse_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Parse PDF using marker with fallback to PyMuPDF."""
        if self.marker_available:
            try:
                return self._parse_pdf_with_marker(file_path)
            except Exception as e:
                logger.warning(f"Marker parsing failed: {e}. Falling back to PyMuPDF.")
        
        if HAS_PYMUPDF:
            return self._parse_pdf_with_pymupdf(file_path)
        else:
            raise RuntimeError("No PDF parsing library available. Install marker-pdf or PyMuPDF.")
    
    def _parse_pdf_with_marker(self, file_path: Path) -> Dict[str, Any]:
        """Parse PDF using marker library for best accuracy."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_path = temp_path / "output"
            output_path.mkdir()
            
            # Build marker command with optimizations
            cmd = [
                'marker', str(file_path),
                '--output_dir', str(output_path),
                '--debug'  # Get detailed structure info
            ]
            
            # Add LLM flag if GOOGLE_API_KEY is available
            import os
            if os.getenv('GOOGLE_API_KEY'):
                cmd.append('--use_llm')
                logger.info("Using LLM enhancement for better table and structure detection")
            
            # Execute marker
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                raise RuntimeError(f"Marker failed: {result.stderr}")
            
            # Read generated markdown
            md_files = list(output_path.glob('*.md'))
            if not md_files:
                raise RuntimeError("No markdown output from marker")
            
            markdown = md_files[0].read_text(encoding='utf-8')
            
            # Read JSON metadata if available
            json_metadata = {}
            json_files = list(output_path.glob('*.json'))
            if json_files:
                with open(json_files[0], 'r', encoding='utf-8') as f:
                    json_metadata = json.load(f)
            
            # Process with intelligent chunking
            chunks = self.chunk_processor.chunk_by_section(markdown)
            
            # Extract tables and images using marker's metadata
            tables = self.markdown_converter.extract_tables(markdown, json_metadata)
            images = self.markdown_converter.extract_images(markdown, json_metadata)
            
            return {
                'content': markdown,
                'chunks': chunks,
                'tables': tables,
                'images': images,
                'metadata': {
                    'file_path': str(file_path),
                    'file_type': 'pdf',
                    'parser': 'marker',
                    'total_chunks': len(chunks),
                    'total_tables': len(tables),
                    'total_images': len(images),
                    'word_count': len(markdown.split()),
                    'marker_metadata': json_metadata
                }
            }
    
    def _parse_pdf_with_pymupdf(self, file_path: Path) -> Dict[str, Any]:
        """Fallback PDF parsing using PyMuPDF."""
        doc = pymupdf.open(file_path)
        full_text = ""
        tables = []
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            full_text += f"\n\n## Page {page_num + 1}\n\n{text}"
            
            # Extract images
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                images.append({
                    'page': page_num + 1,
                    'index': img_index,
                    'type': 'image',
                    'description': f"Image {img_index + 1} on page {page_num + 1}"
                })
            
            # Basic table detection
            tables.extend(self._detect_tables_heuristic(text, page_num + 1))
        
        doc.close()
        
        chunks = self.chunk_processor.chunk_by_section(full_text)
        
        return {
            'content': full_text,
            'chunks': chunks,
            'tables': tables,
            'images': images,
            'metadata': {
                'file_path': str(file_path),
                'file_type': 'pdf',
                'parser': 'pymupdf',
                'total_chunks': len(chunks),
                'total_tables': len(tables),
                'total_images': len(images),
                'word_count': len(full_text.split())
            }
        }
    
    def _parse_docx(self, file_path: Path) -> Dict[str, Any]:
        """Parse DOCX file with table extraction."""
        if not HAS_DOCX:
            raise RuntimeError("python-docx not available. Install with: pip install python-docx")
        
        doc = Document(file_path)
        markdown = ""
        tables = []
        
        # Convert paragraphs to markdown
        for paragraph in doc.paragraphs:
            if paragraph.style.name.startswith('Heading'):
                level = int(paragraph.style.name.split()[-1])
                markdown += f"\n{'#' * level} {paragraph.text}\n\n"
            else:
                markdown += f"{paragraph.text}\n\n"
        
        # Extract tables
        for i, table in enumerate(doc.tables):
            table_data = []
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                table_data.append(row_data)
            
            if table_data:
                # Convert to markdown table
                table_md = self._create_markdown_table(table_data)
                tables.append({
                    'index': i,
                    'type': 'table',
                    'content': table_md,
                    'word_count': len(table_md.split()),
                    'rows': len(table_data),
                    'columns': len(table_data[0]) if table_data else 0
                })
        
        chunks = self.chunk_processor.chunk_by_section(markdown)
        
        return {
            'content': markdown,
            'chunks': chunks,
            'tables': tables,
            'images': [],
            'metadata': {
                'file_path': str(file_path),
                'file_type': 'docx',
                'parser': 'python-docx',
                'total_chunks': len(chunks),
                'total_tables': len(tables),
                'total_images': 0,
                'word_count': len(markdown.split())
            }
        }
    
    def _parse_pptx(self, file_path: Path) -> Dict[str, Any]:
        """Parse PPTX file."""
        if not HAS_PPTX:
            raise RuntimeError("python-pptx not available. Install with: pip install python-pptx")
        
        prs = Presentation(file_path)
        markdown = f"# {file_path.stem}\n\n"
        tables = []
        
        for i, slide in enumerate(prs.slides):
            markdown += f"\n## Slide {i + 1}\n\n"
            
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    markdown += f"{shape.text}\n\n"
                
                # Handle tables in slides
                if shape.shape_type == 19:  # Table
                    table_data = []
                    for row in shape.table.rows:
                        row_data = [cell.text.strip() for cell in row.cells]
                        table_data.append(row_data)
                    
                    if table_data:
                        table_md = self._create_markdown_table(table_data)
                        tables.append({
                            'slide': i + 1,
                            'type': 'table',
                            'content': table_md,
                            'word_count': len(table_md.split())
                        })
                        markdown += f"{table_md}\n\n"
        
        chunks = self.chunk_processor.chunk_by_section(markdown)
        
        return {
            'content': markdown,
            'chunks': chunks,
            'tables': tables,
            'images': [],
            'metadata': {
                'file_path': str(file_path),
                'file_type': 'pptx',
                'parser': 'python-pptx',
                'total_chunks': len(chunks),
                'total_tables': len(tables),
                'total_images': 0,
                'word_count': len(markdown.split())
            }
        }
    
    def _parse_excel(self, file_path: Path) -> Dict[str, Any]:
        """Parse Excel file."""
        if not HAS_EXCEL:
            raise RuntimeError("openpyxl and pandas not available. Install with: pip install openpyxl pandas")
        
        workbook = openpyxl.load_workbook(file_path, data_only=True)
        markdown = f"# {file_path.stem}\n\n"
        tables = []
        
        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            markdown += f"\n## {sheet_name}\n\n"
            
            # Convert sheet to data
            data = []
            for row in sheet.iter_rows(values_only=True):
                if any(cell is not None for cell in row):
                    data.append([str(cell) if cell is not None else "" for cell in row])
            
            if data and len(data) > 1:  # Has header and data
                table_md = self._create_markdown_table(data)
                tables.append({
                    'sheet': sheet_name,
                    'type': 'table',
                    'content': table_md,
                    'word_count': len(table_md.split()),
                    'rows': len(data),
                    'columns': len(data[0]) if data else 0
                })
                markdown += f"{table_md}\n\n"
        
        chunks = self.chunk_processor.chunk_by_section(markdown)
        
        return {
            'content': markdown,
            'chunks': chunks,
            'tables': tables,
            'images': [],
            'metadata': {
                'file_path': str(file_path),
                'file_type': 'xlsx',
                'parser': 'openpyxl',
                'total_chunks': len(chunks),
                'total_tables': len(tables),
                'total_images': 0,
                'word_count': len(markdown.split())
            }
        }
    
    def _parse_txt(self, file_path: Path) -> Dict[str, Any]:
        """Parse plain text file with structure detection."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        # Convert to markdown with structure detection
        markdown = self._text_to_markdown(text)
        chunks = self.chunk_processor.chunk_by_section(markdown)
        
        return {
            'content': markdown,
            'chunks': chunks,
            'tables': [],
            'images': [],
            'metadata': {
                'file_path': str(file_path),
                'file_type': 'txt',
                'parser': 'builtin',
                'total_chunks': len(chunks),
                'total_tables': 0,
                'total_images': 0,
                'word_count': len(text.split())
            }
        }
    
    def _parse_markdown(self, file_path: Path) -> Dict[str, Any]:
        """Parse existing markdown file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            markdown = f.read()
        
        chunks = self.chunk_processor.chunk_by_section(markdown)
        tables = self.markdown_converter.extract_tables(markdown, {})
        images = self.markdown_converter.extract_images(markdown, {})
        
        return {
            'content': markdown,
            'chunks': chunks,
            'tables': tables,
            'images': images,
            'metadata': {
                'file_path': str(file_path),
                'file_type': 'markdown',
                'parser': 'builtin',
                'total_chunks': len(chunks),
                'total_tables': len(tables),
                'total_images': len(images),
                'word_count': len(markdown.split())
            }
        }
    
    def _create_markdown_table(self, data: List[List[str]]) -> str:
        """Convert table data to markdown format."""
        if not data:
            return ""
        
        lines = []
        # Header row
        lines.append('| ' + ' | '.join(str(cell) for cell in data[0]) + ' |')
        # Separator
        lines.append('| ' + ' | '.join(['---'] * len(data[0])) + ' |')
        # Data rows
        for row in data[1:]:
            lines.append('| ' + ' | '.join(str(cell) for cell in row) + ' |')
        
        return '\n'.join(lines)
    
    def _detect_tables_heuristic(self, text: str, page_num: int) -> List[Dict[str, Any]]:
        """Basic table detection for fallback processing."""
        tables = []
        lines = text.split('\n')
        
        table_lines = []
        for line in lines:
            # Heuristic: lines with multiple spaces or tabs might be table rows
            if re.search(r'\s{3,}|\t{2,}', line) and len(line.split()) > 2:
                table_lines.append(line)
            elif table_lines and len(table_lines) > 2:
                # End of potential table
                table_content = '\n'.join(table_lines)
                tables.append({
                    'page': page_num,
                    'type': 'table',
                    'content': table_content,
                    'word_count': len(table_content.split()),
                    'detection': 'heuristic'
                })
                table_lines = []
        
        return tables
    
    def _text_to_markdown(self, text: str) -> str:
        """Convert plain text to markdown with structure detection."""
        lines = text.split('\n')
        markdown_lines = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                markdown_lines.append('')
                continue
            
            # Detect potential headings
            if stripped.isupper() and len(stripped.split()) <= 8:
                markdown_lines.append(f'## {stripped}')
            elif re.match(r'^\d+\.?\s+[A-Z]', stripped):
                markdown_lines.append(f'### {stripped}')
            elif re.match(r'^[A-Z][^.!?]*[:.]\s*$', stripped):
                markdown_lines.append(f'#### {stripped}')
            else:
                markdown_lines.append(stripped)
        
        return '\n'.join(markdown_lines)


class MarkdownConverter:
    """Enhanced markdown conversion and processing."""
    
    def __init__(self, chunk_config: ChunkConfig):
        """Initialize with chunk configuration."""
        self.chunk_config = chunk_config
    
    def convert_to_markdown(self, file_path: Path, file_type: str) -> str:
        """Convert a file to markdown format using the appropriate parser."""
        parser = FileParser(self.chunk_config)
        result = parser.parse(file_path)
        return result['content']
    
    def extract_images(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract images from markdown and metadata."""
        images = []
        
        # Extract markdown image references
        img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        for match in re.finditer(img_pattern, content):
            alt_text, url = match.groups()
            images.append({
                'type': 'image',
                'alt_text': alt_text,
                'url': url,
                'source': 'markdown'
            })
        
        # Add images from marker metadata
        if 'images' in metadata:
            for img in metadata['images']:
                images.append({
                    'type': 'image',
                    'source': 'marker',
                    **img
                })
        
        return images
    
    def extract_tables(self, content: str, metadata: Dict[str, Any]) -> List[str]:
        """Extract tables from markdown with intelligent size-based handling."""
        tables = []
        
        # Extract markdown tables
        table_pattern = r'(\|.*?\|(?:\n\|.*?\|)*)'
        for i, match in enumerate(re.finditer(table_pattern, content, re.MULTILINE)):
            table_text = match.group(1)
            word_count = len(table_text.split())
            
            tables.append({
                'index': i,
                'type': 'table',
                'content': table_text,
                'word_count': word_count,
                'source': 'markdown',
                'should_extract': word_count > self.chunk_config.table_threshold,
                'size_category': 'large' if word_count > self.chunk_config.table_threshold else 'small'
            })
        
        # Add tables from marker metadata
        if 'tables' in metadata:
            for table in metadata['tables']:
                tables.append({
                    'type': 'table',
                    'source': 'marker',
                    **table
                })
        
        return tables


class ChunkProcessor:
    """Intelligent document chunking with research-based size limits."""
    
    def __init__(self, max_chunk_size: int = 1000):
        """Initialize with maximum chunk size (for backward compatibility)."""
        self.max_chunk_size = max_chunk_size
        # Use research-based defaults if not configured
        self.config = ChunkConfig()
    
    def chunk_by_section(self, markdown_text: str) -> List[Dict[str, Any]]:
        """Chunk markdown text by sections with intelligent length limits."""
        chunks = []
        sections = self._parse_markdown_sections(markdown_text)
        
        for section in sections:
            section_chunks = self._chunk_section_intelligently(section)
            chunks.extend(section_chunks)
        
        return chunks
    
    def create_universal_chunks(self, text: str, doc_length: int) -> List[Dict[str, Any]]:
        """Create chunks with universal range related to document length."""
        chunks = []
        words = text.split()
        
        # Dynamic chunk size based on document length
        chunk_size = min(self.max_chunk_size, max(200, doc_length // 20))
        overlap = min(100, chunk_size // 10)
        
        start = 0
        position = 0
        
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_content = ' '.join(chunk_words)
            
            chunk = self.add_chunk_metadata(chunk_content, position, None)
            chunks.append(chunk)
            
            start = end - overlap
            position += 1
        
        return chunks
    
    def add_chunk_metadata(self, chunk: str, position: int, section: Optional[str]) -> Dict[str, Any]:
        """Add comprehensive metadata to chunks."""
        return {
            'content': chunk,
            'position': position,
            'section': section or f"chunk_{position}",
            'word_count': len(chunk.split()),
            'char_count': len(chunk),
            'type': 'text_chunk'
        }
    
    def _parse_markdown_sections(self, markdown: str) -> List[Dict[str, Any]]:
        """Parse markdown into hierarchical sections."""
        lines = markdown.split('\n')
        sections = []
        current_section = {'level': 0, 'title': '', 'content': '', 'children': []}
        section_stack = [current_section]
        
        for line in lines:
            heading_match = re.match(r'^(#{1,6})\s+(.+)', line)
            
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                
                # Close sections at this level and above
                while len(section_stack) > level:
                    section_stack.pop()
                
                # Create new section
                new_section = {
                    'level': level,
                    'title': title,
                    'content': '',
                    'children': []
                }
                
                if section_stack:
                    section_stack[-1]['children'].append(new_section)
                else:
                    sections.append(new_section)
                
                section_stack.append(new_section)
            else:
                # Add content to current section
                if section_stack:
                    section_stack[-1]['content'] += line + '\n'
        
        return sections
    
    def _chunk_section_intelligently(self, section: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk section based on heading level using research-based limits."""
        chunks = []
        level = section['level']
        content = section['content'].strip()
        
        # Determine max size based on research data
        if level <= 1:
            max_size = self.config.level_1_max      # 4000 words
        elif level == 2:
            max_size = self.config.level_2_max      # 1100 words (Methods/Results)
        elif level == 3:
            max_size = self.config.level_3_max      # 550 words (Introduction)
        else:
            max_size = self.config.level_4_max      # 275 words (paragraphs)
        
        words = content.split()
        
        if len(words) <= max_size:
            # Section fits in one chunk
            chunk = self.add_chunk_metadata(content, 0, section['title'])
            chunk['heading_level'] = level
            chunks.append(chunk)
        else:
            # Split with overlap
            overlap_words = self.config.overlap
            start = 0
            chunk_index = 0
            
            while start < len(words):
                end = min(start + max_size, len(words))
                chunk_words = words[start:end]
                chunk_content = ' '.join(chunk_words)
                
                chunk = self.add_chunk_metadata(
                    chunk_content, 
                    chunk_index, 
                    f"{section['title']} (Part {chunk_index + 1})"
                )
                chunk['heading_level'] = level
                chunks.append(chunk)
                
                start = end - overlap_words
                chunk_index += 1
        
        # Process child sections
        for child in section['children']:
            child_chunks = self._chunk_section_intelligently(child)
            chunks.extend(child_chunks)
        
        return chunks


# Convenience function for easy usage
def parse_file_with_marker(file_path: Path, chunk_config: Optional[ChunkConfig] = None) -> Dict[str, Any]:
    """
    High-level function to parse any file using marker and intelligent processing.
    
    Args:
        file_path: Path to the file to parse
        chunk_config: Optional configuration for chunking behavior
        
    Returns:
        Dictionary with parsed content, chunks, tables, images, and metadata
        
    Example:
        >>> result = parse_file_with_marker(Path("research_paper.pdf"))
        >>> print(f"Parsed into {len(result['chunks'])} chunks with {len(result['tables'])} tables")
    """
    parser = FileParser(chunk_config)
    return parser.parse(file_path) 