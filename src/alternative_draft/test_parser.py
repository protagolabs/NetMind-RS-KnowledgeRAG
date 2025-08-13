"""
Test script for parsing PDF documents using Marker (local execution).

This script demonstrates parsing the "Attention Is All You Need" paper
and saving the results to JSON format.
"""

import json
import logging
from pathlib import Path
from alternative_draft.simple_document_parser import SimpleDocumentParser, parse_pdf_locally

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_parse_attention_paper():
    """
    Test parsing the Attention Is All You Need paper.
    """
    # Path to the PDF
    pdf_path = Path("/home/administrator/projects/NetMind-RS-KnowledgeRAG/paper_sets/paper_set_1/docs/Attention Is All You Need.pdf")
    
    # Check if file exists
    if not pdf_path.exists():
        logger.error(f"PDF not found at: {pdf_path}")
        return
    
    logger.info(f"Starting to parse: {pdf_path.name}")
    logger.info(f"File size: {pdf_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    try:
        # Parse the PDF (runs locally on your hardware)
        logger.info("Initializing local Marker models...")
        parser = SimpleDocumentParser(chunk_size=500, chunk_overlap=50)
        
        logger.info("Parsing PDF with local models...")
        parsed_doc = parser.parse_pdf(pdf_path)
        
        # Display results
        print("\n" + "="*60)
        print("PARSING RESULTS")
        print("="*60)
        
        # Metadata
        print("\n📊 METADATA:")
        print(f"  - File: {parsed_doc.metadata.file_name}")
        print(f"  - Size: {parsed_doc.metadata.file_size / 1024 / 1024:.2f} MB")
        print(f"  - Pages: {parsed_doc.metadata.page_count}")
        print(f"  - Words: {parsed_doc.metadata.word_count:,}")
        print(f"  - Characters: {parsed_doc.metadata.char_count:,}")
        print(f"  - Parse time: {parsed_doc.metadata.parse_time}")
        print(f"  - Parser: {parsed_doc.metadata.parser_type} (local)")
        
        # Content statistics
        print("\n📄 CONTENT STATISTICS:")
        print(f"  - Chunks created: {len(parsed_doc.chunks)}")
        print(f"  - Sections found: {len(parsed_doc.sections)}")
        print(f"  - Tables extracted: {len(parsed_doc.tables)}")
        print(f"  - Images found: {len(parsed_doc.images)}")
        
        # Show first few sections
        if parsed_doc.sections:
            print("\n📑 DOCUMENT SECTIONS (first 10):")
            for section in parsed_doc.sections[:10]:
                indent = "  " * section['level']
                print(f"{indent}{'#' * section['level']} {section['title']}")
        
        # Show sample of first chunk
        if parsed_doc.chunks:
            first_chunk = parsed_doc.chunks[0]
            print(f"\n📝 FIRST CHUNK (ID: {first_chunk.id}, {first_chunk.word_count} words):")
            print("-" * 40)
            preview = first_chunk.content[:500] + "..." if len(first_chunk.content) > 500 else first_chunk.content
            print(preview)
            print("-" * 40)
        
        # Show table info
        if parsed_doc.tables:
            print(f"\n📊 TABLES FOUND ({len(parsed_doc.tables)} total):")
            for i, table in enumerate(parsed_doc.tables[:3]):
                print(f"  Table {i+1}: {table['rows']} rows × {table['columns']} columns")
        
        # Save to JSON
        output_path = pdf_path.parent / f"{pdf_path.stem}_parsed.json"
        parser.save_to_json(parsed_doc, output_path)
        print(f"\n✅ Results saved to: {output_path}")
        
        # Also save a summary JSON
        summary_path = pdf_path.parent / f"{pdf_path.stem}_summary.json"
        summary = {
            "file_name": parsed_doc.metadata.file_name,
            "file_path": str(pdf_path.absolute()),
            "metadata": parsed_doc.metadata.model_dump(),
            "statistics": {
                "total_chunks": len(parsed_doc.chunks),
                "total_sections": len(parsed_doc.sections),
                "total_tables": len(parsed_doc.tables),
                "total_images": len(parsed_doc.images),
                "word_count": parsed_doc.metadata.word_count,
                "char_count": parsed_doc.metadata.char_count
            },
            "sections": [s['title'] for s in parsed_doc.sections[:20]],  # First 20 sections
            "first_500_chars": parsed_doc.content[:500] if parsed_doc.content else "",
            "chunk_samples": [
                {
                    "id": c.id,
                    "word_count": c.word_count,
                    "preview": c.content[:100] + "..."
                } for c in parsed_doc.chunks[:3]
            ]
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📋 Summary saved to: {summary_path}")
        
        return parsed_doc
        
    except Exception as e:
        logger.error(f"Failed to parse PDF: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_quick_parse():
    """
    Quick test using the convenience function.
    """
    pdf_path = "/home/administrator/projects/NetMind-RS-KnowledgeRAG/paper_sets/paper_set_1/docs/Attention Is All You Need.pdf"
    
    print("\n" + "="*60)
    print("QUICK PARSE TEST")
    print("="*60)
    
    try:
        # This will automatically save to JSON
        parsed = parse_pdf_locally(pdf_path, chunk_size=500)
        
        print(f"\n✅ Successfully parsed!")
        print(f"  - Content length: {len(parsed.content)} characters")
        print(f"  - Chunks: {len(parsed.chunks)}")
        print(f"  - Sections: {len(parsed.sections)}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    print("🚀 Testing Marker PDF Parser (Local Execution)")
    print("=" * 60)
    print("This runs entirely on your local hardware - no API calls needed!")
    print("Models will be downloaded on first run and cached locally.")
    print("=" * 60)
    
    # Run the main test
    test_parse_attention_paper()
    
    # Uncomment to also run quick test
    # test_quick_parse()