#!/usr/bin/env python3
"""
Example usage of the Knowledge RAG File Management System

This script demonstrates how to use the file management system
to process documents and track their status.
"""

import logging
from pathlib import Path
from src.file_management import FileManager
from src.file_management.models import FileStatus

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_sample_documents():
    """Create some sample documents for testing"""
    docs_folder = Path("./test_documents")
    docs_folder.mkdir(exist_ok=True)
    
    # Create a sample text file
    (docs_folder / "sample.txt").write_text("""
This is a sample text document for testing the Knowledge RAG system.

It contains multiple paragraphs with different types of content.
This paragraph discusses the importance of document processing in modern AI systems.

Knowledge graphs are powerful structures for representing relationships between entities.
They enable more sophisticated querying and reasoning capabilities.

The temporal aspect of knowledge is crucial for understanding how facts change over time.
This system tracks when information was extracted and when it becomes invalid.
""")
    
    # Create a sample markdown file
    (docs_folder / "sample.md").write_text("""
# Knowledge RAG System Documentation

## Overview

The Knowledge RAG system is designed to process various document formats
and extract meaningful information for knowledge graph construction.

### Features

- Multi-format document parsing
- Temporal knowledge tracking
- Entity and relationship extraction
- Community detection

### Supported Formats

- Text files (.txt, .md, .csv)
- PDF documents
- Microsoft Word documents (.docx)
- And more...

## Implementation

The system is built with a modular architecture that allows for easy
extension and customization of parsing capabilities.
""")
    
    # Create a sample CSV file
    (docs_folder / "sample.csv").write_text("""
name,age,city,profession
John Smith,30,New York,Engineer
Jane Doe,25,San Francisco,Designer
Bob Johnson,35,Chicago,Manager
Alice Brown,28,Seattle,Developer
Charlie Wilson,32,Boston,Analyst
""")
    
    logger.info(f"Created sample documents in {docs_folder}")
    return docs_folder


def on_file_processed(processed_file, parsing_result):
    """Callback function called when a file is successfully processed"""
    logger.info(f"Successfully processed: {processed_file.metadata.file_name}")
    logger.info(f"Episodes created: {len(parsing_result.episodes)}")
    
    # Print first episode content (truncated)
    if parsing_result.episodes:
        first_episode = parsing_result.episodes[0]
        content_preview = first_episode.content[:200] + "..." if len(first_episode.content) > 200 else first_episode.content
        logger.info(f"First episode preview: {content_preview}")


def on_processing_error(processed_file, error_message):
    """Callback function called when file processing fails"""
    logger.error(f"Failed to process: {processed_file.metadata.file_name}")
    logger.error(f"Error: {error_message}")


def main():
    """Main example function"""
    logger.info("=== Knowledge RAG File Management System Demo ===")
    
    # Create sample documents
    docs_folder = create_sample_documents()
    
    # Initialize FileManager
    file_manager = FileManager(
        root_folder=str(docs_folder),
        db_path="example_knowledge_rag.db"
    )
    
    # Set up callbacks for processing events
    file_manager.set_callbacks(
        on_file_processed=on_file_processed,
        on_processing_error=on_processing_error
    )
    
    logger.info(f"Initialized FileManager with root folder: {docs_folder}")
    
    # Scan for files
    logger.info("\n--- Scanning for files ---")
    files = file_manager.scan_files()
    logger.info(f"Found {len(files)} supported files:")
    for file_path in files:
        logger.info(f"  - {file_path.name} ({file_path.suffix})")
    
    # Check for unprocessed files
    logger.info("\n--- Checking for unprocessed files ---")
    unprocessed_files = file_manager.get_unprocessed_files()
    logger.info(f"Found {len(unprocessed_files)} unprocessed files")
    
    # Process all unprocessed files
    if unprocessed_files:
        logger.info("\n--- Processing files ---")
        stats = file_manager.process_all_unprocessed()
        logger.info(f"Processing complete: {stats}")
    
    # Get processing status
    logger.info("\n--- Processing Status ---")
    status = file_manager.get_processing_status()
    logger.info(f"Total files in folder: {status['total_files_in_folder']}")
    logger.info(f"Supported extensions: {status['supported_extensions']}")
    
    if 'tracker_stats' in status:
        tracker_stats = status['tracker_stats']
        logger.info("Processing statistics:")
        for stat_name, count in tracker_stats.items():
            if isinstance(count, int):
                logger.info(f"  - {stat_name}: {count}")
    
    # Demonstrate individual file processing
    logger.info("\n--- Individual File Processing Demo ---")
    if files:
        sample_file = files[0]
        logger.info(f"Processing individual file: {sample_file.name}")
        
        result = file_manager.process_file(sample_file)
        if result:
            logger.info(f"Result: {result.status.value}")
            if result.status == FileStatus.COMPLETED:
                logger.info(f"Episodes created: {result.episodes_created}")
            elif result.status == FileStatus.FAILED:
                logger.info(f"Error message: {result.error_message}")
    
    # Show parser information
    logger.info("\n--- Parser Information ---")
    parser_info = file_manager.parser_factory.get_parser_info()
    for parser_name, supported_formats in parser_info.items():
        logger.info(f"{parser_name}: {', '.join(supported_formats)}")
    
    # Final status check
    logger.info("\n--- Final Status ---")
    final_status = file_manager.get_processing_status()
    if 'tracker_stats' in final_status:
        completed_count = final_status['tracker_stats'].get('completed', 0)
        failed_count = final_status['tracker_stats'].get('failed', 0)
        total_episodes = final_status['tracker_stats'].get('total_episodes', 0)
        
        logger.info(f"Successfully processed: {completed_count} files")
        logger.info(f"Failed processing: {failed_count} files")
        logger.info(f"Total episodes created: {total_episodes}")
    
    logger.info("\n=== Demo Complete ===")


if __name__ == "__main__":
    main() 