"""
File Tracker - Manages tracking of processed files and their status
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from .models import ProcessedFile, FileMetadata, FileStatus, FileFormat


class FileTracker:
    """Tracks processed files and their status using SQLite database"""
    
    def __init__(self, db_path: str = "knowledge_rag.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_files (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    file_format TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    modified_at TEXT NOT NULL,
                    encoding TEXT,
                    language TEXT,
                    status TEXT NOT NULL,
                    processing_started_at TEXT,
                    processing_completed_at TEXT,
                    episodes_created INTEGER DEFAULT 0,
                    entities_extracted INTEGER DEFAULT 0,
                    relationships_extracted INTEGER DEFAULT 0,
                    error_message TEXT,
                    processing_metadata TEXT,
                    tracked_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_path ON processed_files(file_path)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_hash ON processed_files(file_hash)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON processed_files(status)
            """)
            
            conn.commit()
    
    def add_file(self, processed_file: ProcessedFile) -> bool:
        """Add a new processed file to tracking"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO processed_files (
                        id, file_path, file_name, file_size, file_format, file_hash,
                        created_at, modified_at, encoding, language, status,
                        processing_started_at, processing_completed_at,
                        episodes_created, entities_extracted, relationships_extracted,
                        error_message, processing_metadata, tracked_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    processed_file.id,
                    str(processed_file.metadata.file_path),
                    processed_file.metadata.file_name,
                    processed_file.metadata.file_size,
                    processed_file.metadata.file_format.value,
                    processed_file.metadata.file_hash,
                    processed_file.metadata.created_at.isoformat(),
                    processed_file.metadata.modified_at.isoformat(),
                    processed_file.metadata.encoding,
                    processed_file.metadata.language,
                    processed_file.status.value,
                    processed_file.processing_started_at.isoformat() if processed_file.processing_started_at else None,
                    processed_file.processing_completed_at.isoformat() if processed_file.processing_completed_at else None,
                    processed_file.episodes_created,
                    processed_file.entities_extracted,
                    processed_file.relationships_extracted,
                    processed_file.error_message,
                    json.dumps(processed_file.processing_metadata),
                    datetime.now().isoformat()
                ))
                conn.commit()
                return True
        except Exception as e:
            self.logger.error(f"Error adding file to tracker: {e}")
            return False
    
    def get_file(self, file_id: str) -> Optional[ProcessedFile]:
        """Get a processed file by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM processed_files WHERE id = ?", (file_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_processed_file(row)
                return None
        except Exception as e:
            self.logger.error(f"Error getting file from tracker: {e}")
            return None
    
    def get_file_by_path(self, file_path: str) -> Optional[ProcessedFile]:
        """Get a processed file by file path"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM processed_files WHERE file_path = ?", (str(file_path),)
                )
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_processed_file(row)
                return None
        except Exception as e:
            self.logger.error(f"Error getting file by path from tracker: {e}")
            return None
    
    def is_file_processed(self, file_path: str, file_hash: str) -> bool:
        """Check if a file has already been processed"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM processed_files WHERE file_path = ? AND file_hash = ? AND status = ?",
                    (str(file_path), file_hash, FileStatus.COMPLETED.value)
                )
                count = cursor.fetchone()[0]
                return count > 0
        except Exception as e:
            self.logger.error(f"Error checking if file is processed: {e}")
            return False
    
    def get_files_by_status(self, status: FileStatus) -> List[ProcessedFile]:
        """Get all files with specific status"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM processed_files WHERE status = ? ORDER BY tracked_at",
                    (status.value,)
                )
                rows = cursor.fetchall()
                
                return [self._row_to_processed_file(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Error getting files by status: {e}")
            return []
    
    def update_file_status(self, file_id: str, status: FileStatus, **kwargs):
        """Update file processing status and related fields"""
        try:
            update_fields = ["status = ?"]
            update_values = [status.value]
            
            if status == FileStatus.PROCESSING:
                update_fields.append("processing_started_at = ?")
                update_values.append(datetime.now().isoformat())
            elif status == FileStatus.COMPLETED:
                update_fields.append("processing_completed_at = ?")
                update_values.append(datetime.now().isoformat())
                
                if 'episodes' in kwargs:
                    update_fields.append("episodes_created = ?")
                    update_values.append(kwargs['episodes'])
                
                if 'entities' in kwargs:
                    update_fields.append("entities_extracted = ?")
                    update_values.append(kwargs['entities'])
                
                if 'relationships' in kwargs:
                    update_fields.append("relationships_extracted = ?")
                    update_values.append(kwargs['relationships'])
            
            elif status == FileStatus.FAILED:
                update_fields.append("processing_completed_at = ?")
                update_values.append(datetime.now().isoformat())
                
                if 'error' in kwargs:
                    update_fields.append("error_message = ?")
                    update_values.append(kwargs['error'])
            
            update_values.append(file_id)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    f"UPDATE processed_files SET {', '.join(update_fields)} WHERE id = ?",
                    update_values
                )
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error updating file status: {e}")
    
    def get_processing_stats(self) -> Dict[str, int]:
        """Get statistics about processed files"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        status,
                        COUNT(*) as count,
                        SUM(episodes_created) as total_episodes,
                        SUM(entities_extracted) as total_entities,
                        SUM(relationships_extracted) as total_relationships
                    FROM processed_files 
                    GROUP BY status
                """)
                
                stats = {}
                total_episodes = 0
                total_entities = 0
                total_relationships = 0
                
                for row in cursor.fetchall():
                    status, count, episodes, entities, relationships = row
                    stats[status] = count
                    if episodes:
                        total_episodes += episodes
                    if entities:
                        total_entities += entities
                    if relationships:
                        total_relationships += relationships
                
                stats['total_episodes'] = total_episodes
                stats['total_entities'] = total_entities
                stats['total_relationships'] = total_relationships
                
                return stats
        except Exception as e:
            self.logger.error(f"Error getting processing stats: {e}")
            return {}
    
    def _row_to_processed_file(self, row: sqlite3.Row) -> ProcessedFile:
        """Convert database row to ProcessedFile object"""
        metadata = FileMetadata(
            file_path=Path(row['file_path']),
            file_name=row['file_name'],
            file_size=row['file_size'],
            file_format=FileFormat(row['file_format']),
            created_at=datetime.fromisoformat(row['created_at']),
            modified_at=datetime.fromisoformat(row['modified_at']),
            file_hash=row['file_hash'],
            encoding=row['encoding'],
            language=row['language']
        )
        
        return ProcessedFile(
            id=row['id'],
            metadata=metadata,
            status=FileStatus(row['status']),
            processing_started_at=datetime.fromisoformat(row['processing_started_at']) if row['processing_started_at'] else None,
            processing_completed_at=datetime.fromisoformat(row['processing_completed_at']) if row['processing_completed_at'] else None,
            episodes_created=row['episodes_created'],
            entities_extracted=row['entities_extracted'],
            relationships_extracted=row['relationships_extracted'],
            error_message=row['error_message'],
            processing_metadata=json.loads(row['processing_metadata']) if row['processing_metadata'] else {}
        )
    
    def cleanup_failed_files(self, max_age_days: int = 7) -> int:
        """Clean up failed file records older than specified days"""
        try:
            cutoff_date = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
            cutoff_iso = datetime.fromtimestamp(cutoff_date).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM processed_files WHERE status = ? AND tracked_at < ?",
                    (FileStatus.FAILED.value, cutoff_iso)
                )
                deleted_count = cursor.rowcount
                conn.commit()
                
                self.logger.info(f"Cleaned up {deleted_count} failed file records")
                return deleted_count
        except Exception as e:
            self.logger.error(f"Error cleaning up failed files: {e}")
            return 0 