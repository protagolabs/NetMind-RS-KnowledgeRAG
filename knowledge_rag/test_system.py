#!/usr/bin/env python3
"""
Simple test script to verify the Knowledge RAG file management system works
"""

import sys
import os
from pathlib import Path
import tempfile
import shutil

# Add the current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

def test_basic_functionality():
    """Test basic file management functionality"""
    print("=== Testing Knowledge RAG File Management System ===\n")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files
        print("1. Creating test files...")
        test_files = {
            "test.txt": "This is a test text file.\n\nIt has multiple paragraphs.\n\nEach paragraph should be parsed separately.",
            "test.md": "# Test Markdown\n\n## Section 1\n\nThis is section 1 content.\n\n## Section 2\n\nThis is section 2 content.",
            "test.csv": "name,age,city\nJohn,30,NYC\nJane,25,LA\nBob,35,Chicago"
        }
        
        for filename, content in test_files.items():
            (temp_path / filename).write_text(content)
        
        print(f"Created {len(test_files)} test files in {temp_path}")
        
        # Test file management system
        try:
            from src.file_management import FileManager
            from src.file_management.models import FileStatus
            
            print("\n2. Initializing FileManager...")
            file_manager = FileManager(
                root_folder=str(temp_path),
                db_path=str(temp_path / "test.db")
            )
            
            print("✓ FileManager initialized successfully")
            
            # Test scanning
            print("\n3. Scanning for files...")
            files = file_manager.scan_files()
            print(f"✓ Found {len(files)} files:")
            for file_path in files:
                print(f"  - {file_path.name}")
            
            # Test processing
            print("\n4. Processing files...")
            stats = file_manager.process_all_unprocessed()
            print(f"✓ Processing complete: {stats}")
            
            # Test status
            print("\n5. Getting status...")
            status = file_manager.get_processing_status()
            print(f"✓ Total files processed: {status.get('tracker_stats', {}).get('completed', 0)}")
            
            # Test individual parsers
            print("\n6. Testing parser factory...")
            parser_info = file_manager.parser_factory.get_parser_info()
            print("✓ Available parsers:")
            for parser_name, formats in parser_info.items():
                print(f"  - {parser_name}: {', '.join(formats)}")
            
            print("\n=== All tests passed! ===")
            return True
            
        except ImportError as e:
            print(f"✗ Import error: {e}")
            print("Make sure all required modules are available")
            return False
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_parser_dependencies():
    """Test which parsers are available based on dependencies"""
    print("\n=== Testing Parser Dependencies ===")
    
    dependencies = {
        "chardet": "Text encoding detection",
        "fitz": "PDF parsing (PyMuPDF)", 
        "docx": "DOCX parsing (python-docx)",
        "langdetect": "Language detection"
    }
    
    available = []
    missing = []
    
    for dep, description in dependencies.items():
        try:
            __import__(dep)
            available.append(f"✓ {dep}: {description}")
        except ImportError:
            missing.append(f"✗ {dep}: {description}")
    
    print("Available dependencies:")
    for item in available:
        print(f"  {item}")
    
    if missing:
        print("\nMissing dependencies (install for full functionality):")
        for item in missing:
            print(f"  {item}")
        
        print("\nTo install missing dependencies:")
        if "fitz" in [item.split()[1].rstrip(':') for item in missing]:
            print("  pip install PyMuPDF")
        if "docx" in [item.split()[1].rstrip(':') for item in missing]:
            print("  pip install python-docx")
        if "chardet" in [item.split()[1].rstrip(':') for item in missing]:
            print("  pip install chardet")
        if "langdetect" in [item.split()[1].rstrip(':') for item in missing]:
            print("  pip install langdetect")
    
    return len(missing) == 0

if __name__ == "__main__":
    print("Knowledge RAG File Management System Test")
    print("=" * 50)
    
    # Test dependencies first
    deps_ok = test_parser_dependencies()
    
    # Test basic functionality
    basic_ok = test_basic_functionality()
    
    print("\n" + "=" * 50)
    if basic_ok:
        print("✓ Basic functionality test: PASSED")
    else:
        print("✗ Basic functionality test: FAILED")
    
    if deps_ok:
        print("✓ All dependencies available")
    else:
        print("⚠ Some dependencies missing (system will work with reduced functionality)")
    
    if basic_ok:
        print("\n🎉 The Knowledge RAG File Management System is working correctly!")
        print("\nNext steps:")
        print("1. Try the CLI: python cli.py scan --root-folder ./documents")
        print("2. Run the example: python example_usage.py")
        print("3. Read the README.md for detailed usage instructions")
    else:
        print("\n❌ There are issues with the system. Check the error messages above.")
        
    sys.exit(0 if basic_ok else 1) 