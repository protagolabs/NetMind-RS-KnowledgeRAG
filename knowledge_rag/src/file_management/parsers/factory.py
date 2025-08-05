"""
Parser Factory - Factory class to manage and create appropriate parsers
"""

from typing import Optional, List, Dict, Type
import logging

from .base_parser import BaseParser
from .text_parser import TextParser
from .pdf_parser import PDFParser
from .docx_parser import DocxParser
from ..models import FileFormat


class ParserFactory:
    """Factory for creating and managing document parsers"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._parsers: List[BaseParser] = []
        self._parser_map: Dict[FileFormat, BaseParser] = {}
        self._initialize_parsers()
    
    def _initialize_parsers(self):
        """Initialize all available parsers"""
        parser_classes = [
            TextParser,
            PDFParser,
            DocxParser
        ]
        
        for parser_class in parser_classes:
            try:
                parser = parser_class()
                self._parsers.append(parser)
                self.logger.info(f"Initialized {parser_class.__name__}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize {parser_class.__name__}: {e}")
        
        # Build parser mapping
        self._build_parser_map()
    
    def _build_parser_map(self):
        """Build mapping of file formats to parsers"""
        for parser in self._parsers:
            for file_format in FileFormat:
                if parser.can_parse(file_format):
                    if file_format not in self._parser_map:
                        self._parser_map[file_format] = parser
                        self.logger.debug(f"Mapped {file_format.value} to {parser.__class__.__name__}")
    
    def get_parser(self, file_format: FileFormat) -> Optional[BaseParser]:
        """Get appropriate parser for the given file format"""
        return self._parser_map.get(file_format)
    
    def get_supported_formats(self) -> List[FileFormat]:
        """Get list of all supported file formats"""
        return list(self._parser_map.keys())
    
    def is_format_supported(self, file_format: FileFormat) -> bool:
        """Check if a file format is supported"""
        return file_format in self._parser_map
    
    def get_parser_info(self) -> Dict[str, List[str]]:
        """Get information about available parsers and supported formats"""
        info = {}
        
        for parser in self._parsers:
            parser_name = parser.__class__.__name__
            supported_formats = []
            
            for file_format in FileFormat:
                if parser.can_parse(file_format):
                    supported_formats.append(file_format.value)
            
            info[parser_name] = supported_formats
        
        return info
    
    def add_custom_parser(self, parser: BaseParser):
        """Add a custom parser to the factory"""
        try:
            self._parsers.append(parser)
            
            # Update parser mapping
            for file_format in FileFormat:
                if parser.can_parse(file_format):
                    # Only replace if format not already supported or if this parser is better
                    if file_format not in self._parser_map:
                        self._parser_map[file_format] = parser
                        self.logger.info(f"Added custom parser {parser.__class__.__name__} for {file_format.value}")
            
        except Exception as e:
            self.logger.error(f"Error adding custom parser: {e}")
    
    def get_parser_stats(self) -> Dict[str, int]:
        """Get statistics about parser usage"""
        stats = {
            'total_parsers': len(self._parsers),
            'supported_formats': len(self._parser_map),
            'unsupported_formats': len(FileFormat) - len(self._parser_map)
        }
        
        # Count formats per parser
        parser_format_count = {}
        for parser in self._parsers:
            parser_name = parser.__class__.__name__
            count = sum(1 for fmt in FileFormat if parser.can_parse(fmt))
            parser_format_count[parser_name] = count
        
        stats['formats_per_parser'] = parser_format_count
        
        return stats 