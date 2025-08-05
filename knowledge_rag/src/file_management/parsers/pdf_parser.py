"""
PDF Parser - Handles PDF documents using PyMuPDF
"""

from pathlib import Path
from typing import List, Dict, Any
import logging

from .base_parser import BaseParser
from ..models import FileMetadata, Episode, ParsingResult, FileFormat


class PDFParser(BaseParser):
    """Parser for PDF documents"""
    
    SUPPORTED_FORMATS = {FileFormat.PDF}
    
    def __init__(self):
        super().__init__()
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required dependencies are available"""
        try:
            import fitz  # PyMuPDF
            self.fitz = fitz
        except ImportError:
            self.logger.error("PyMuPDF (fitz) not available. Install with: pip install PyMuPDF")
            self.fitz = None
    
    def can_parse(self, file_format: FileFormat) -> bool:
        """Check if this parser can handle the given file format"""
        return file_format in self.SUPPORTED_FORMATS and self.fitz is not None
    
    def parse(self, file_path: Path, metadata: FileMetadata) -> ParsingResult:
        """Parse PDF file and create episodes"""
        if not self.fitz:
            return ParsingResult(
                file_id=metadata.file_hash,
                episodes=[],
                success=False,
                error_message="PyMuPDF not available"
            )
        
        try:
            episodes = self._parse_pdf(file_path, metadata)
            
            return ParsingResult(
                file_id=metadata.file_hash,
                episodes=episodes,
                parsing_metadata={
                    'parser': 'PDFParser',
                    'format': metadata.file_format.value,
                    'total_pages': len(episodes)
                },
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing PDF file {file_path}: {e}")
            return ParsingResult(
                file_id=metadata.file_hash,
                episodes=[],
                success=False,
                error_message=str(e)
            )
    
    def _parse_pdf(self, file_path: Path, metadata: FileMetadata) -> List[Episode]:
        """Parse PDF document page by page"""
        episodes = []
        
        try:
            doc = self.fitz.open(str(file_path))
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text content
                text_content = page.get_text()
                
                # Extract images info
                images_info = self._extract_images_info(page)
                
                # Extract tables info
                tables_info = self._extract_tables_info(page)
                
                # Get layout information
                layout_info = self._analyze_page_layout(page)
                
                # Clean text content
                text_content = self.clean_text(text_content)
                
                # Skip empty pages
                if not text_content.strip() and not images_info and not tables_info:
                    continue
                
                # Check for cross-page elements
                continuation_marker = self._check_cross_page_elements(page, doc, page_num)
                
                # Create episode for this page
                episode_content = self._format_page_content(
                    text_content, images_info, tables_info, page_num + 1
                )
                
                episode = self.create_episode(
                    content=episode_content,
                    episode_type='page',
                    sequence_number=page_num + 1,
                    source_file_id=metadata.file_hash,
                    metadata={
                        'page_number': page_num + 1,
                        'has_images': len(images_info) > 0,
                        'has_tables': len(tables_info) > 0,
                        'images_count': len(images_info),
                        'tables_count': len(tables_info),
                        'continuation_marker': continuation_marker,
                        'layout_info': layout_info,
                        'text_length': len(text_content)
                    }
                )
                episodes.append(episode)
            
            doc.close()
            
        except Exception as e:
            self.logger.error(f"Error processing PDF pages: {e}")
            raise
        
        return episodes
    
    def _extract_images_info(self, page) -> List[Dict[str, Any]]:
        """Extract information about images on the page"""
        images_info = []
        
        try:
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                # Get image details
                xref, smask, width, height, bpc, colorspace, alt, name, filter = img[:9]
                
                images_info.append({
                    'index': img_index,
                    'width': width,
                    'height': height,
                    'colorspace': colorspace,
                    'name': name or f"image_{img_index}",
                    'filter': filter
                })
                
        except Exception as e:
            self.logger.debug(f"Error extracting image info: {e}")
        
        return images_info
    
    def _extract_tables_info(self, page) -> List[Dict[str, Any]]:
        """Extract information about tables on the page"""
        tables_info = []
        
        try:
            # Try to detect tables using text blocks
            blocks = page.get_text("blocks")
            
            # Simple heuristic: look for blocks with multiple lines and tab/space patterns
            for block_idx, block in enumerate(blocks):
                if len(block) >= 4:  # block format: (x0, y0, x1, y1, text, block_no, block_type)
                    text = block[4]
                    lines = text.split('\n')
                    
                    # Check if it looks like a table (multiple lines with consistent structure)
                    if len(lines) >= 3:
                        tab_count = sum(1 for line in lines if '\t' in line or '  ' in line)
                        if tab_count >= len(lines) * 0.7:  # 70% of lines have tab/space patterns
                            tables_info.append({
                                'index': block_idx,
                                'rows_estimate': len(lines),
                                'position': (block[0], block[1], block[2], block[3]),
                                'text_preview': text[:200] + "..." if len(text) > 200 else text
                            })
                            
        except Exception as e:
            self.logger.debug(f"Error extracting table info: {e}")
        
        return tables_info
    
    def _analyze_page_layout(self, page) -> Dict[str, Any]:
        """Analyze page layout and structure"""
        layout_info = {}
        
        try:
            # Get page dimensions
            rect = page.rect
            layout_info['width'] = rect.width
            layout_info['height'] = rect.height
            
            # Get text blocks
            blocks = page.get_text("blocks")
            layout_info['text_blocks_count'] = len(blocks)
            
            # Analyze text distribution
            if blocks:
                y_positions = [block[1] for block in blocks if len(block) >= 4]  # y0 positions
                if y_positions:
                    layout_info['text_start_y'] = min(y_positions)
                    layout_info['text_end_y'] = max(y_positions)
                    layout_info['text_height_ratio'] = (max(y_positions) - min(y_positions)) / rect.height
            
            # Check for multi-column layout
            x_positions = [block[0] for block in blocks if len(block) >= 4]  # x0 positions
            if len(set([round(x, -1) for x in x_positions])) > 2:  # More than 2 distinct x positions
                layout_info['multi_column'] = True
            else:
                layout_info['multi_column'] = False
                
        except Exception as e:
            self.logger.debug(f"Error analyzing page layout: {e}")
        
        return layout_info
    
    def _check_cross_page_elements(self, page, doc, page_num) -> bool:
        """Check if content spans multiple pages"""
        try:
            # Simple heuristic: check if text ends mid-sentence and next page starts continuing
            if page_num < len(doc) - 1:
                current_text = page.get_text().strip()
                next_page = doc[page_num + 1]
                next_text = next_page.get_text().strip()
                
                if (current_text and next_text and 
                    not current_text.endswith('.') and 
                    not current_text.endswith('!') and 
                    not current_text.endswith('?') and
                    next_text[0].islower()):
                    return True
                    
        except Exception as e:
            self.logger.debug(f"Error checking cross-page elements: {e}")
        
        return False
    
    def _format_page_content(self, text_content: str, images_info: List[Dict], 
                           tables_info: List[Dict], page_num: int) -> str:
        """Format page content including text, images, and tables info"""
        content = f"Page {page_num}\n\n"
        
        # Add text content
        if text_content:
            content += text_content + "\n\n"
        
        # Add images information
        if images_info:
            content += f"Images on this page ({len(images_info)}):\n"
            for img in images_info:
                content += f"- {img['name']}: {img['width']}x{img['height']} pixels\n"
            content += "\n"
        
        # Add tables information
        if tables_info:
            content += f"Tables on this page ({len(tables_info)}):\n"
            for table in tables_info:
                content += f"- Table {table['index']}: ~{table['rows_estimate']} rows\n"
                content += f"  Preview: {table['text_preview'][:100]}...\n"
            content += "\n"
        
        return content 