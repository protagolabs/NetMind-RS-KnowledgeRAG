#!/usr/bin/env python3
"""
Simple test script to test markdown parsing with the Knowledge_RAG_Design_Document.md file
"""

import sys
import logging
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_markdown_parsing():
    """Test parsing the Knowledge_RAG_Design_Document.md file"""
    
    # Path to the markdown file
    markdown_file = Path("/home/administrator/projects/XYZ_memory/knowledge_rag/Knowledge_RAG_Design_Document.md")
    
    if not markdown_file.exists():
        print(f"❌ File not found: {markdown_file}")
        return False
    
    print(f"🔍 Testing markdown parsing with: {markdown_file.name}")
    print(f"📁 File size: {markdown_file.stat().st_size:,} bytes")
    print()
    
    try:
        # Import the file management system
        from src.file_management import FileManager
        from src.file_management.models import FileStatus
        
        # Create a temporary directory for the database
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_parsing.db"
            
            # Initialize FileManager with the parent directory
            file_manager = FileManager(
                root_folder=str(markdown_file.parent),
                db_path=str(db_path)
            )
            
            print("✅ FileManager initialized")
            
            # Process the specific markdown file
            print("🔄 Processing markdown file...")
            result = file_manager.process_file(markdown_file)
            
            if result and result.status == FileStatus.COMPLETED:
                print("✅ File processed successfully!")
                print(f"📄 Episodes created: {result.episodes_created}")
                print(f"⏱️  Processing time: {result.processing_completed_at - result.processing_started_at}")
                print()
                
                # Get the parsing result to show episode details
                parser = file_manager.parser_factory.get_parser(result.metadata.file_format)
                parsing_result = parser.parse(markdown_file, result.metadata)
                
                print("📝 Episode breakdown by sections:")
                print("=" * 80)
                
                # Group episodes by header level and show structure
                episode_structure = {}
                for episode in parsing_result.episodes:
                    header_level = episode.metadata.get('header_level', 0)
                    if header_level not in episode_structure:
                        episode_structure[header_level] = []
                    episode_structure[header_level].append(episode)
                
                # Show episodes organized by header level
                for level in sorted(episode_structure.keys()):
                    episodes_at_level = episode_structure[level]
                    level_name = {
                        1: "Main Sections (#)",
                        2: "Sections (##)", 
                        3: "Subsections (###)",
                        4: "Sub-subsections (####)",
                        5: "Detailed sections (#####)",
                        6: "Minor sections (######)"
                    }.get(level, f"Level {level}")
                    
                    print(f"\n🏷️  {level_name}: {len(episodes_at_level)} episodes")
                    print("-" * 60)
                    
                    for i, episode in enumerate(episodes_at_level):
                        header_text = episode.metadata.get('header_text', 'No header')
                        content_length = len(episode.content)
                        is_chunked = episode.metadata.get('is_chunked', False)
                        
                        chunk_info = ""
                        if is_chunked:
                            chunk_num = episode.metadata.get('chunk_number', 1)
                            total_chunks = episode.metadata.get('total_chunks', 1)
                            chunk_info = f" [Chunk {chunk_num}/{total_chunks}]"
                        
                        # Show content preview (first 100 chars)
                        content_preview = episode.content[:100].replace('\n', ' ').strip()
                        if len(episode.content) > 100:
                            content_preview += "..."
                        
                        print(f"  📄 Episode {episode.sequence_number}: '{header_text}'{chunk_info}")
                        print(f"      Type: {episode.episode_type} | Length: {content_length:,} chars")
                        print(f"      Preview: {content_preview}")
                        print()
                
                # Show statistics
                print("📊 Parsing Statistics:")
                print("-" * 40)
                total_episodes = len(parsing_result.episodes)
                chunked_episodes = sum(1 for ep in parsing_result.episodes if ep.metadata.get('is_chunked', False))
                avg_length = sum(len(ep.content) for ep in parsing_result.episodes) / total_episodes if total_episodes > 0 else 0
                
                print(f"Total episodes: {total_episodes}")
                print(f"Chunked episodes: {chunked_episodes}")
                print(f"Average episode length: {avg_length:.0f} characters")
                
                # Show length distribution
                print(f"\n📏 Episode length distribution:")
                length_ranges = {
                    "0-500": 0, "501-1000": 0, "1001-1500": 0, 
                    "1501-2000": 0, "2001-3000": 0, "3000+": 0
                }
                
                for episode in parsing_result.episodes:
                    length = len(episode.content)
                    if length <= 500:
                        length_ranges["0-500"] += 1
                    elif length <= 1000:
                        length_ranges["501-1000"] += 1
                    elif length <= 1500:
                        length_ranges["1001-1500"] += 1
                    elif length <= 2000:
                        length_ranges["1501-2000"] += 1
                    elif length <= 3000:
                        length_ranges["2001-3000"] += 1
                    else:
                        length_ranges["3000+"] += 1
                
                for range_name, count in length_ranges.items():
                    if count > 0:
                        print(f"  {range_name} chars: {count} episodes")
                
                return True
                
            elif result and result.status == FileStatus.FAILED:
                print(f"❌ Processing failed: {result.error_message}")
                return False
            else:
                print("❌ Processing returned no result")
                return False
                
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you've installed the required dependencies:")
        print("pip install click rich python-dotenv")
        return False
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_parser_info():
    """Show information about available parsers"""
    try:
        from src.file_management.parsers.factory import ParserFactory
        
        factory = ParserFactory()
        parser_info = factory.get_parser_info()
        
        print("🔧 Available parsers:")
        for parser_name, formats in parser_info.items():
            print(f"  • {parser_name}: {', '.join(formats)}")
        
        # Check if markdown is supported
        from src.file_management.models import FileFormat
        md_format = FileFormat.MD
        if factory.is_format_supported(md_format):
            parser = factory.get_parser(md_format)
            print(f"✅ Markdown parsing supported by: {parser.__class__.__name__}")
        else:
            print("❌ Markdown parsing not supported")
        
        print()
        
    except Exception as e:
        print(f"❌ Error getting parser info: {e}")

if __name__ == "__main__":
    print("Knowledge RAG Markdown Parsing Test")
    print("=" * 50)
    
    # Show parser information
    show_parser_info()
    
    # Test markdown parsing
    success = test_markdown_parsing()
    
    print("=" * 50)
    if success:
        print("🎉 Markdown parsing test completed successfully!")
        print("\nThe file was parsed into episodes. Each episode represents a section")
        print("of the markdown document, organized by headers (# ## ### etc.).")
        print("\nYou can now use these episodes for knowledge extraction!")
    else:
        print("❌ Markdown parsing test failed.")
        print("Check the error messages above for troubleshooting.")
    
    sys.exit(0 if success else 1) 