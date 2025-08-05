"""
Text Parser - Handles plain text files (.txt, .md, .csv)
"""

import csv
from pathlib import Path
from typing import List
import logging

from .base_parser import BaseParser
from ..models import FileMetadata, Episode, ParsingResult, FileFormat


class TextParser(BaseParser):
    """Parser for plain text files"""
    
    SUPPORTED_FORMATS = {FileFormat.TXT, FileFormat.MD, FileFormat.CSV}
    
    def can_parse(self, file_format: FileFormat) -> bool:
        """Check if this parser can handle the given file format"""
        return file_format in self.SUPPORTED_FORMATS
    
    def parse(self, file_path: Path, metadata: FileMetadata) -> ParsingResult:
        """Parse text file and create episodes"""
        try:
            episodes = []
            
            if metadata.file_format == FileFormat.CSV:
                episodes = self._parse_csv(file_path, metadata)
            else:
                episodes = self._parse_text(file_path, metadata)
            
            return ParsingResult(
                file_id=metadata.file_hash,
                episodes=episodes,
                parsing_metadata={
                    'parser': 'TextParser',
                    'format': metadata.file_format.value,
                    'encoding': metadata.encoding
                },
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing text file {file_path}: {e}")
            return ParsingResult(
                file_id=metadata.file_hash,
                episodes=[],
                success=False,
                error_message=str(e)
            )
    
    def _parse_text(self, file_path: Path, metadata: FileMetadata) -> List[Episode]:
        """Parse regular text files (txt, md)"""
        episodes = []
        
        try:
            with open(file_path, 'r', encoding=metadata.encoding or 'utf-8') as f:
                content = f.read()
            
            # Clean the content
            content = self.clean_text(content)
            
            if not content:
                return episodes
            
            # For Markdown files, try to split by headers
            if metadata.file_format == FileFormat.MD:
                episodes = self._parse_markdown(content, metadata.file_hash)
            else:
                # For plain text, split by paragraphs or chunks
                episodes = self._parse_plain_text(content, metadata.file_hash)
                
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                content = self.clean_text(content)
                episodes = self._parse_plain_text(content, metadata.file_hash)
            except Exception as e:
                self.logger.error(f"Failed to read file with fallback encoding: {e}")
                
        return episodes
    
    def _parse_markdown(self, content: str, file_id: str) -> List[Episode]:
        """Parse Markdown content by hierarchical sections with smart chunking"""
        episodes = []
        
        # Parse document into hierarchical structure
        sections = self._parse_markdown_hierarchy(content)
        
        # If no sections found (no headers), treat as plain text
        if not sections:
            self.logger.info("No markdown headers found, treating as plain text")
            return self._parse_plain_text(content, file_id)
        
        # Convert hierarchical sections to episodes
        sequence_num = 1
        for section in sections:
            new_episodes = self._create_episodes_from_section(section, file_id, sequence_num)
            episodes.extend(new_episodes)
            sequence_num += len(new_episodes)
        
        return episodes
    
    def _parse_markdown_hierarchy(self, content: str) -> List[dict]:
        """Parse markdown into hierarchical sections"""
        lines = content.split('\n')
        sections = []
        current_sections = {}  # Track sections at each level
        
        for line_num, line in enumerate(lines):
            stripped_line = line.strip()
            
            # Check for header
            if stripped_line.startswith('#'):
                header_level = self._get_header_level(stripped_line)
                header_text = stripped_line.lstrip('#').strip()
                
                # Close sections deeper than current level
                levels_to_close = [level for level in current_sections.keys() if level >= header_level]
                for level in levels_to_close:
                    if current_sections[level]['content'].strip():
                        sections.append(current_sections[level])
                    del current_sections[level]
                
                # Start new section at current level
                current_sections[header_level] = {
                    'header_level': header_level,
                    'header_text': header_text,
                    'full_header': stripped_line,
                    'content': stripped_line + '\n',
                    'start_line': line_num,
                    'subsections': []
                }
            else:
                # Add content to all open sections
                for section in current_sections.values():
                    section['content'] += line + '\n'
        
        # Close any remaining open sections
        for section in current_sections.values():
            if section['content'].strip():
                sections.append(section)
        
        return sections
    
    def _get_header_level(self, line: str) -> int:
        """Get the header level (number of # characters)"""
        count = 0
        for char in line:
            if char == '#':
                count += 1
            else:
                break
        return min(count, 6)  # Cap at level 6
    
    def _create_episodes_from_section(self, section: dict, file_id: str, start_sequence: int) -> List[Episode]:
        """Create episodes from a markdown section, chunking if too long"""
        episodes = []
        content = section['content'].strip()
        
        # Skip very short sections
        if len(content) < 50:
            return episodes
        
        # Determine target chunk size based on header level
        max_chunk_size = self._get_chunk_size_for_level(section['header_level'])
        
        # If content is short enough, create single episode
        if len(content) <= max_chunk_size:
            episode = self.create_episode(
                content=content,
                episode_type=self._get_episode_type_for_level(section['header_level']),
                sequence_number=start_sequence,
                source_file_id=file_id,
                metadata={
                    'header_level': section['header_level'],
                    'header_text': section['header_text'],
                    'full_header': section['full_header'],
                    'section_type': 'markdown_section',
                    'is_chunked': False,
                    'start_line': section.get('start_line', 0)
                }
            )
            episodes.append(episode)
        else:
            # Content is too long, need to chunk it intelligently
            chunks = self._smart_chunk_markdown_section(content, max_chunk_size)
            
            for i, chunk in enumerate(chunks):
                episode = self.create_episode(
                    content=chunk,
                    episode_type=self._get_episode_type_for_level(section['header_level']),
                    sequence_number=start_sequence + i,
                    source_file_id=file_id,
                    metadata={
                        'header_level': section['header_level'],
                        'header_text': section['header_text'],
                        'full_header': section['full_header'],
                        'section_type': 'markdown_section',
                        'is_chunked': True,
                        'chunk_number': i + 1,
                        'total_chunks': len(chunks),
                        'start_line': section.get('start_line', 0)
                    }
                )
                episodes.append(episode)
        
        return episodes
    
    def _get_chunk_size_for_level(self, header_level: int) -> int:
        """Get appropriate chunk size based on header level"""
        # Smaller chunks for deeper sections
        size_map = {
            1: 3000,  # Main sections can be longer
            2: 2000,  # Subsections 
            3: 1500,  # Sub-subsections
            4: 1200,  # Detailed sections
            5: 1000,  # Very specific sections
            6: 800    # Deepest level sections
        }
        return size_map.get(header_level, 1000)
    
    def _get_episode_type_for_level(self, header_level: int) -> str:
        """Get episode type based on header level"""
        type_map = {
            1: 'main_section',
            2: 'section', 
            3: 'subsection',
            4: 'subsubsection',
            5: 'detailed_section',
            6: 'minor_section'
        }
        return type_map.get(header_level, 'section')
    
    def _smart_chunk_markdown_section(self, content: str, max_size: int) -> List[str]:
        """Intelligently chunk a markdown section while preserving structure"""
        chunks = []
        lines = content.split('\n')
        
        current_chunk = ""
        current_size = 0
        
        # Try to preserve paragraph boundaries
        current_paragraph = ""
        
        for line in lines:
            line_with_newline = line + '\n'
            
            # If adding this line would exceed limit
            if current_size + len(line_with_newline) > max_size:
                # If we have content in current chunk, save it
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_size = 0
                
                # If single line is too long, force break
                if len(line_with_newline) > max_size:
                    # Split the line at word boundaries
                    words = line.split()
                    temp_line = ""
                    for word in words:
                        if len(temp_line + word + " ") <= max_size:
                            temp_line += word + " "
                        else:
                            if temp_line.strip():
                                chunks.append(temp_line.strip())
                            temp_line = word + " "
                    if temp_line.strip():
                        current_chunk = temp_line
                        current_size = len(temp_line)
                else:
                    current_chunk = line_with_newline
                    current_size = len(line_with_newline)
            else:
                current_chunk += line_with_newline
                current_size += len(line_with_newline)
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [content]
    
    def _parse_plain_text(self, content: str, file_id: str) -> List[Episode]:
        """Parse plain text content"""
        episodes = []
        
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        if not paragraphs:
            # Fallback: split by single newlines
            paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
        
        # If still too few paragraphs, chunk the content
        if len(paragraphs) < 3:
            chunks = self.chunk_large_content(content, max_chunk_size=3000)
            for i, chunk in enumerate(chunks):
                episode = self.create_episode(
                    content=chunk,
                    episode_type='chunk',
                    sequence_number=i + 1,
                    source_file_id=file_id,
                    metadata={'chunk_type': 'text_chunk'}
                )
                episodes.append(episode)
        else:
            # Create episodes from paragraphs
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph) > 50:  # Skip very short paragraphs
                    episode = self.create_episode(
                        content=paragraph,
                        episode_type='paragraph',
                        sequence_number=i + 1,
                        source_file_id=file_id,
                        metadata={'paragraph_type': 'text_paragraph'}
                    )
                    episodes.append(episode)
        
        return episodes
    
    def _parse_csv(self, file_path: Path, metadata: FileMetadata) -> List[Episode]:
        """Parse CSV files"""
        episodes = []
        
        try:
            with open(file_path, 'r', encoding=metadata.encoding or 'utf-8') as f:
                # Try to detect CSV dialect
                sample = f.read(1024)
                f.seek(0)
                sniffer = csv.Sniffer()
                
                try:
                    dialect = sniffer.sniff(sample)
                except csv.Error:
                    dialect = csv.excel
                
                reader = csv.DictReader(f, dialect=dialect)
                
                # Group rows for episodes (e.g., every 10 rows)
                rows_per_episode = 10
                current_rows = []
                episode_num = 1
                
                for row_num, row in enumerate(reader, 1):
                    current_rows.append(row)
                    
                    if len(current_rows) >= rows_per_episode:
                        episode_content = self._format_csv_rows(current_rows, list(row.keys()))
                        episode = self.create_episode(
                            content=episode_content,
                            episode_type='csv_batch',
                            sequence_number=episode_num,
                            source_file_id=metadata.file_hash,
                            metadata={
                                'row_count': len(current_rows),
                                'columns': list(row.keys()),
                                'batch_start_row': row_num - len(current_rows) + 1,
                                'batch_end_row': row_num
                            }
                        )
                        episodes.append(episode)
                        
                        current_rows = []
                        episode_num += 1
                
                # Handle remaining rows
                if current_rows:
                    episode_content = self._format_csv_rows(current_rows, list(current_rows[0].keys()))
                    episode = self.create_episode(
                        content=episode_content,
                        episode_type='csv_batch',
                        sequence_number=episode_num,
                        source_file_id=metadata.file_hash,
                        metadata={
                            'row_count': len(current_rows),
                            'columns': list(current_rows[0].keys())
                        }
                    )
                    episodes.append(episode)
                    
        except Exception as e:
            self.logger.error(f"Error parsing CSV file: {e}")
            # Fallback: treat as plain text
            episodes = self._parse_text(file_path, metadata)
        
        return episodes
    
    def _format_csv_rows(self, rows: List[dict], columns: List[str]) -> str:
        """Format CSV rows into readable text"""
        content = f"CSV Data ({len(rows)} rows):\n\n"
        
        for i, row in enumerate(rows, 1):
            content += f"Row {i}:\n"
            for col in columns:
                value = row.get(col, '')
                content += f"  {col}: {value}\n"
            content += "\n"
        
        return content 