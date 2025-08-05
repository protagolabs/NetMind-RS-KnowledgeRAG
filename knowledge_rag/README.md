# Knowledge RAG File Management System

A comprehensive file management system for the Knowledge RAG project that handles document ingestion, parsing, and tracking across multiple formats with temporal knowledge graph construction capabilities.

## Features

### 🔧 Core Capabilities
- **Multi-format Document Processing**: Support for txt, doc, pdf, pptx, xlsx, markdown, json, and more
- **Intelligent File Tracking**: SQLite-based tracking system that prevents duplicate processing
- **Modular Parser Architecture**: Extensible parser system with format-specific handlers
- **Episode-based Content Organization**: Documents are parsed into structured episodes for knowledge extraction
- **Comprehensive Metadata Extraction**: File properties, encoding detection, and format analysis

### 📁 Supported File Formats

| Category | Formats | Parser | Status |
|----------|---------|---------|---------|
| **Text** | .txt, .md, .csv | TextParser | ✅ Implemented |
| **Documents** | .pdf | PDFParser | ✅ Implemented |
| **Documents** | .docx | DocxParser | ✅ Implemented |
| **Documents** | .doc | DocxParser | ⚠️ Limited (convert to .docx) |
| **Presentations** | .ppt, .pptx | - | 🔄 Planned |
| **Spreadsheets** | .xls, .xlsx | - | 🔄 Planned |
| **Structured** | .json, .xml, .yaml | - | 🔄 Planned |
| **Web** | .html, .htm | - | 🔄 Planned |

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd knowledge_rag
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install optional dependencies** (for advanced parsing):
```bash
# For PDF parsing
pip install PyMuPDF

# For DOCX parsing
pip install python-docx

# For encoding detection
pip install chardet

# For language detection
pip install langdetect
```

## Quick Start

### Using the CLI

1. **Set up your document folder**:
```bash
mkdir documents
# Add your documents to the documents folder
```

2. **Scan for files**:
```bash
python cli.py scan --root-folder ./documents
```

3. **Process all files**:
```bash
python cli.py process --root-folder ./documents
```

4. **Check status**:
```bash
python cli.py status --root-folder ./documents
```

### Using the API

```python
from src.file_management import FileManager

# Initialize file manager
file_manager = FileManager(
    root_folder="./documents",
    db_path="knowledge_rag.db"
)

# Scan for files
files = file_manager.scan_files()
print(f"Found {len(files)} files")

# Process all unprocessed files
stats = file_manager.process_all_unprocessed()
print(f"Processing stats: {stats}")

# Get processing status
status = file_manager.get_processing_status()
print(f"Status: {status}")
```

### Example with Callbacks

```python
def on_file_processed(processed_file, parsing_result):
    print(f"✓ Processed: {processed_file.metadata.file_name}")
    print(f"  Episodes: {len(parsing_result.episodes)}")

def on_processing_error(processed_file, error_message):
    print(f"✗ Failed: {processed_file.metadata.file_name}")
    print(f"  Error: {error_message}")

file_manager.set_callbacks(
    on_file_processed=on_file_processed,
    on_processing_error=on_processing_error
)

# Process files with callbacks
file_manager.process_all_unprocessed()
```

## Configuration

The system can be configured using environment variables or the `config.py` file:

```python
# Environment variables
KNOWLEDGE_RAG_ROOT_FOLDER=./documents
KNOWLEDGE_RAG_DB_PATH=knowledge_rag.db
KNOWLEDGE_RAG_MAX_FILE_SIZE_MB=100
KNOWLEDGE_RAG_LOG_LEVEL=INFO

# File-specific settings
KNOWLEDGE_RAG_MAX_CHUNK_SIZE=5000
KNOWLEDGE_RAG_CSV_ROWS_PER_EPISODE=10
KNOWLEDGE_RAG_PARAGRAPHS_PER_EPISODE=8
```

## Architecture

### Core Components

```
knowledge_rag/
├── src/
│   └── file_management/
│       ├── __init__.py          # Package initialization
│       ├── models.py            # Data models and enums
│       ├── file_manager.py      # Main coordinator
│       ├── file_tracker.py      # SQLite-based tracking
│       └── parsers/
│           ├── __init__.py      # Parser package
│           ├── base_parser.py   # Abstract base parser
│           ├── text_parser.py   # Text file parser
│           ├── pdf_parser.py    # PDF parser
│           ├── docx_parser.py   # DOCX parser
│           └── factory.py       # Parser factory
├── config.py                    # Configuration management
├── cli.py                      # Command-line interface
├── example_usage.py            # Example usage script
└── requirements.txt            # Dependencies
```

### Data Models

#### ProcessedFile
Represents a file that has been processed by the system:
- **id**: Unique identifier
- **metadata**: File metadata (FileMetadata)
- **status**: Processing status (PENDING, PROCESSING, COMPLETED, FAILED, SKIPPED)
- **episodes_created**: Number of episodes extracted
- **processing_metadata**: Parser-specific metadata

#### Episode
Represents a parsed content episode from a document:
- **id**: Unique identifier
- **content**: Parsed text content
- **episode_type**: Type (page, section, chunk, table, etc.)
- **sequence_number**: Order within the document
- **metadata**: Episode-specific metadata

#### FileMetadata
Contains file information:
- **file_path**: Path to the file
- **file_format**: Detected format (FileFormat enum)
- **file_size**: Size in bytes
- **file_hash**: SHA-256 hash for duplicate detection
- **encoding**: Text encoding (for text files)
- **created_at/modified_at**: File timestamps

## CLI Commands

### Basic Commands

```bash
# Scan for files
python cli.py scan -r ./documents

# Process all unprocessed files
python cli.py process -r ./documents

# Show processing status
python cli.py status -r ./documents

# Process a single file
python cli.py process-single ./documents/sample.pdf

# Reprocess failed files
python cli.py reprocess-failed -r ./documents

# Clean up old failed records
python cli.py cleanup --max-age-days 7
```

### Command Options

- `--root-folder, -r`: Root folder to scan for documents
- `--db-path, -d`: Path to SQLite database
- `--recursive/--no-recursive`: Enable/disable recursive scanning
- `--verbose, -v`: Enable verbose logging

## Extending the System

### Adding a Custom Parser

```python
from src.file_management.parsers.base_parser import BaseParser
from src.file_management.models import FileFormat, ParsingResult

class CustomParser(BaseParser):
    SUPPORTED_FORMATS = {FileFormat.JSON}  # Example
    
    def can_parse(self, file_format: FileFormat) -> bool:
        return file_format in self.SUPPORTED_FORMATS
    
    def parse(self, file_path: Path, metadata: FileMetadata) -> ParsingResult:
        # Your parsing logic here
        episodes = []  # Create episodes from the file
        
        return ParsingResult(
            file_id=metadata.file_hash,
            episodes=episodes,
            success=True
        )

# Add to parser factory
file_manager.parser_factory.add_custom_parser(CustomParser())
```

### Custom File Processing Callbacks

```python
def my_processing_callback(processed_file, parsing_result):
    """Custom callback for successful processing"""
    # Log to external system
    # Send notifications
    # Trigger downstream processing
    pass

def my_error_callback(processed_file, error_message):
    """Custom callback for processing errors"""
    # Error reporting
    # Retry logic
    # Alerting
    pass

file_manager.set_callbacks(
    on_file_processed=my_processing_callback,
    on_processing_error=my_error_callback
)
```

## File Processing Details

### Text Files (.txt, .md, .csv)
- **Encoding Detection**: Automatic encoding detection using chardet
- **Markdown Parsing**: Section-based parsing using headers
- **CSV Parsing**: Row-based episodes with configurable batch sizes
- **Paragraph Splitting**: Intelligent paragraph detection and chunking

### PDF Files (.pdf)
- **Page-based Episodes**: Each page becomes an episode
- **Image Detection**: Metadata about images on each page
- **Table Detection**: Heuristic-based table identification
- **Cross-page Elements**: Detection of content spanning multiple pages
- **Layout Analysis**: Page structure and multi-column detection

### DOCX Files (.docx)
- **Section-based Parsing**: Uses document headings for organization
- **Document Properties**: Extracts metadata (author, title, etc.)
- **Table Extraction**: Separate episodes for tables
- **Image Detection**: Identifies inline images
- **Fallback Parsing**: Paragraph-based grouping when no sections

## Database Schema

The system uses SQLite for tracking processed files:

```sql
CREATE TABLE processed_files (
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
);
```

## Performance Considerations

- **File Hashing**: SHA-256 hashing prevents duplicate processing
- **Incremental Processing**: Only processes new or modified files
- **Memory Management**: Streaming processing for large files
- **Database Indexing**: Optimized queries for file lookup
- **Error Recovery**: Graceful handling of parsing failures

## Error Handling

The system provides comprehensive error handling:

- **Parser Availability**: Checks for required dependencies
- **File Access**: Handles permission and I/O errors
- **Encoding Issues**: Fallback encoding strategies
- **Memory Limits**: Chunking for large files
- **Format Validation**: Graceful handling of unsupported formats

## Monitoring and Logging

- **Rich Logging**: Detailed logging with configurable levels
- **Progress Tracking**: Visual progress bars for CLI operations
- **Statistics**: Comprehensive processing statistics
- **Error Tracking**: Detailed error reporting and recovery

## Future Enhancements

- [ ] **Additional Parsers**: PPT, XLS, JSON, XML, HTML parsers
- [ ] **Advanced Text Processing**: Language-specific text processing
- [ ] **Image OCR**: Extract text from images in documents
- [ ] **Table Structure**: Preserve table structure in episodes
- [ ] **Async Processing**: Parallel file processing
- [ ] **REST API**: HTTP API for remote file processing
- [ ] **File Watching**: Real-time monitoring for new files
- [ ] **Cloud Storage**: Support for S3, GCS, Azure Blob
- [ ] **Metadata Enrichment**: External metadata sources
- [ ] **Quality Scoring**: Content quality assessment

## Troubleshooting

### Common Issues

1. **"No parser available for format"**
   - Install required dependencies (PyMuPDF, python-docx)
   - Check file extension is supported

2. **"Permission denied"**
   - Ensure read permissions on files and write permissions on database

3. **"Encoding detection failed"**
   - Install chardet: `pip install chardet`
   - Manually specify encoding in configuration

4. **"Database locked"**
   - Ensure no other processes are using the database
   - Check file permissions on database file

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
python cli.py --verbose status
```

Or in Python:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 