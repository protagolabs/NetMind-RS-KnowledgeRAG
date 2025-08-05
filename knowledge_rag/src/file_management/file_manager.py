"""
File Manager - Main coordinator for file management operations
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
import logging
from datetime import datetime
import hashlib

from .models import FileMetadata, ProcessedFile, FileStatus, FileFormat, Episode, ParsingResult
from .file_tracker import FileTracker
from .parsers.factory import ParserFactory


class FileManager:
    """Main file management system coordinator"""
    
    def __init__(self, 
                 root_folder: str,
                 db_path: str = "knowledge_rag.db",
                 supported_extensions: Optional[List[str]] = None):
        """
        Initialize FileManager
        
        Args:
            root_folder: Root directory to monitor for files
            db_path: Path to SQLite database for tracking
            supported_extensions: List of file extensions to process (if None, uses all supported)
        """
        self.root_folder = Path(root_folder)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.tracker = FileTracker(db_path)
        self.parser_factory = ParserFactory()
        
        # Set supported extensions
        if supported_extensions:
            self.supported_extensions = set(ext.lower().lstrip('.') for ext in supported_extensions)
        else:
            self.supported_extensions = set(fmt.value for fmt in self.parser_factory.get_supported_formats())
        
        # Callbacks for processing events
        self.on_file_processed: Optional[Callable[[ProcessedFile, ParsingResult], None]] = None
        self.on_processing_error: Optional[Callable[[ProcessedFile, str], None]] = None
        
        self.logger.info(f"FileManager initialized with root folder: {self.root_folder}")
        self.logger.info(f"Supported extensions: {sorted(self.supported_extensions)}")
    
    def scan_files(self, recursive: bool = True) -> List[Path]:
        """
        Scan root folder for supported files
        
        Args:
            recursive: Whether to scan subdirectories recursively
            
        Returns:
            List of file paths found
        """
        files_found = []
        
        try:
            if not self.root_folder.exists():
                self.logger.error(f"Root folder does not exist: {self.root_folder}")
                return files_found
            
            pattern = "**/*" if recursive else "*"
            
            for file_path in self.root_folder.glob(pattern):
                if file_path.is_file():
                    extension = file_path.suffix.lower().lstrip('.')
                    if extension in self.supported_extensions:
                        files_found.append(file_path)
            
            self.logger.info(f"Found {len(files_found)} supported files")
            
        except Exception as e:
            self.logger.error(f"Error scanning files: {e}")
        
        return files_found
    
    def get_unprocessed_files(self, recursive: bool = True) -> List[Path]:
        """
        Get list of files that haven't been processed yet
        
        Args:
            recursive: Whether to scan subdirectories recursively
            
        Returns:
            List of unprocessed file paths
        """
        all_files = self.scan_files(recursive)
        unprocessed_files = []
        
        for file_path in all_files:
            try:
                # Calculate file hash to check if it's been processed
                file_hash = self._calculate_file_hash(file_path)
                
                if not self.tracker.is_file_processed(str(file_path), file_hash):
                    unprocessed_files.append(file_path)
                else:
                    self.logger.debug(f"File already processed: {file_path}")
                    
            except Exception as e:
                self.logger.error(f"Error checking file {file_path}: {e}")
        
        self.logger.info(f"Found {len(unprocessed_files)} unprocessed files")
        return unprocessed_files
    
    def process_file(self, file_path: Path) -> Optional[ProcessedFile]:
        """
        Process a single file
        
        Args:
            file_path: Path to file to process
            
        Returns:
            ProcessedFile object if successful, None otherwise
        """
        try:
            # Create file metadata
            metadata = self._create_file_metadata(file_path)
            if not metadata:
                return None
            
            # Create ProcessedFile record
            processed_file = ProcessedFile(
                id="",  # Will be generated
                metadata=metadata,
                status=FileStatus.PENDING
            )
            
            # Add to tracker
            if not self.tracker.add_file(processed_file):
                self.logger.error(f"Failed to add file to tracker: {file_path}")
                return None
            
            # Mark as processing started
            processed_file.mark_processing_started()
            self.tracker.update_file_status(processed_file.id, FileStatus.PROCESSING)
            
            # Get appropriate parser
            parser = self.parser_factory.get_parser(metadata.file_format)
            if not parser:
                error_msg = f"No parser available for format: {metadata.file_format.value}"
                self.logger.error(error_msg)
                processed_file.mark_processing_failed(error_msg)
                self.tracker.update_file_status(processed_file.id, FileStatus.FAILED, error=error_msg)
                
                if self.on_processing_error:
                    self.on_processing_error(processed_file, error_msg)
                
                return processed_file
            
            # Parse the file
            self.logger.info(f"Processing file: {file_path}")
            parsing_result = parser.parse(file_path, metadata)
            
            if parsing_result.success:
                # Mark as completed
                processed_file.mark_processing_completed(
                    episodes=len(parsing_result.episodes),
                    entities=0,  # Will be filled by knowledge extraction
                    relationships=0  # Will be filled by knowledge extraction
                )
                processed_file.processing_metadata = parsing_result.parsing_metadata
                
                self.tracker.update_file_status(
                    processed_file.id, 
                    FileStatus.COMPLETED,
                    episodes=len(parsing_result.episodes)
                )
                
                self.logger.info(f"Successfully processed {file_path}: {len(parsing_result.episodes)} episodes created")
                
                # Trigger callback
                if self.on_file_processed:
                    self.on_file_processed(processed_file, parsing_result)
                
            else:
                # Mark as failed
                error_msg = parsing_result.error_message or "Unknown parsing error"
                processed_file.mark_processing_failed(error_msg)
                self.tracker.update_file_status(processed_file.id, FileStatus.FAILED, error=error_msg)
                
                self.logger.error(f"Failed to process {file_path}: {error_msg}")
                
                if self.on_processing_error:
                    self.on_processing_error(processed_file, error_msg)
            
            return processed_file
            
        except Exception as e:
            error_msg = f"Unexpected error processing file: {str(e)}"
            self.logger.error(f"Error processing {file_path}: {e}")
            
            # Try to update status if we have a processed_file object
            try:
                if 'processed_file' in locals():
                    processed_file.mark_processing_failed(error_msg)
                    self.tracker.update_file_status(processed_file.id, FileStatus.FAILED, error=error_msg)
                    return processed_file
            except:
                pass
            
            return None
    
    def process_all_unprocessed(self, recursive: bool = True) -> Dict[str, int]:
        """
        Process all unprocessed files
        
        Args:
            recursive: Whether to scan subdirectories recursively
            
        Returns:
            Dictionary with processing statistics
        """
        unprocessed_files = self.get_unprocessed_files(recursive)
        
        stats = {
            'total_files': len(unprocessed_files),
            'processed_successfully': 0,
            'processing_failed': 0,
            'skipped': 0
        }
        
        for file_path in unprocessed_files:
            try:
                result = self.process_file(file_path)
                
                if result:
                    if result.status == FileStatus.COMPLETED:
                        stats['processed_successfully'] += 1
                    elif result.status == FileStatus.FAILED:
                        stats['processing_failed'] += 1
                    else:
                        stats['skipped'] += 1
                else:
                    stats['skipped'] += 1
                    
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                stats['processing_failed'] += 1
        
        self.logger.info(f"Batch processing completed: {stats}")
        return stats
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status and statistics"""
        tracker_stats = self.tracker.get_processing_stats()
        parser_stats = self.parser_factory.get_parser_stats()
        
        # Count files in root folder
        total_files_in_folder = len(self.scan_files())
        
        status = {
            'root_folder': str(self.root_folder),
            'total_files_in_folder': total_files_in_folder,
            'supported_extensions': sorted(self.supported_extensions),
            'tracker_stats': tracker_stats,
            'parser_stats': parser_stats,
            'last_updated': datetime.now().isoformat()
        }
        
        return status
    
    def reprocess_failed_files(self) -> Dict[str, int]:
        """Reprocess files that previously failed"""
        failed_files = self.tracker.get_files_by_status(FileStatus.FAILED)
        
        stats = {
            'total_failed': len(failed_files),
            'reprocessed_successfully': 0,
            'still_failing': 0
        }
        
        for failed_file in failed_files:
            try:
                file_path = failed_file.metadata.file_path
                
                if file_path.exists():
                    result = self.process_file(file_path)
                    
                    if result and result.status == FileStatus.COMPLETED:
                        stats['reprocessed_successfully'] += 1
                    else:
                        stats['still_failing'] += 1
                else:
                    self.logger.warning(f"Failed file no longer exists: {file_path}")
                    stats['still_failing'] += 1
                    
            except Exception as e:
                self.logger.error(f"Error reprocessing failed file: {e}")
                stats['still_failing'] += 1
        
        return stats
    
    def _create_file_metadata(self, file_path: Path) -> Optional[FileMetadata]:
        """Create FileMetadata object for a file"""
        try:
            if not file_path.exists():
                self.logger.error(f"File does not exist: {file_path}")
                return None
            
            stat = file_path.stat()
            file_format = FileFormat.from_extension(file_path.suffix)
            file_hash = self._calculate_file_hash(file_path)
            
            # Detect encoding for text files
            encoding = None
            if file_format in {FileFormat.TXT, FileFormat.MD, FileFormat.CSV}:
                encoding = self._detect_encoding(file_path)
            
            metadata = FileMetadata(
                file_path=file_path,
                file_name=file_path.name,
                file_size=stat.st_size,
                file_format=file_format,
                created_at=datetime.fromtimestamp(stat.st_ctime),
                modified_at=datetime.fromtimestamp(stat.st_mtime),
                file_hash=file_hash,
                encoding=encoding
            )
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error creating file metadata for {file_path}: {e}")
            return None
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding"""
        try:
            import chardet
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)
                result = chardet.detect(raw_data)
                return result.get('encoding', 'utf-8')
        except ImportError:
            return 'utf-8'
        except Exception as e:
            self.logger.debug(f"Could not detect encoding for {file_path}: {e}")
            return 'utf-8'
    
    def cleanup_tracker(self, max_age_days: int = 7) -> int:
        """Clean up old failed records from tracker"""
        return self.tracker.cleanup_failed_files(max_age_days)
    
    def set_callbacks(self, 
                     on_file_processed: Optional[Callable] = None,
                     on_processing_error: Optional[Callable] = None):
        """Set callback functions for processing events"""
        self.on_file_processed = on_file_processed
        self.on_processing_error = on_processing_error 