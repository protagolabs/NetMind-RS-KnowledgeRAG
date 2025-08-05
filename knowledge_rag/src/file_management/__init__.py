"""
File Management System for Knowledge RAG

This package handles file ingestion, parsing, and tracking across multiple formats.
"""

from .file_manager import FileManager
from .file_tracker import FileTracker
from .parsers import ParserFactory

__all__ = ['FileManager', 'FileTracker', 'ParserFactory'] 