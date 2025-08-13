"""
Test script to understand what Marker provides for table extraction.
"""

from pathlib import Path
import json

try:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    from marker.schema import BlockTypes
    MARKER_AVAILABLE = True
except ImportError:
    MARKER_AVAILABLE = False
    print("Marker not available")


def explore_marker_output():
    """Explore what Marker provides in its rendered output."""
    
    if not MARKER_AVAILABLE:
        print("Marker library not installed")
        return
    
    # Use the already parsed simple JSON to avoid long processing
    simple_json = Path("paper_sets/paper_set_1/docs/Attention Is All You Need_parsed.json")
    
    if simple_json.exists():
        print("Using existing parsed content to demonstrate table extraction\n")
        
        # Load the content
        with open(simple_json, 'r') as f:
            data = json.load(f)
        
        content = data['content']
        
        # Let's analyze what we have
        print("=== Analyzing Markdown Content for Tables ===\n")
        
        # Look for markdown tables
        import re
        
        # Pattern for markdown tables (with header separator)
        table_pattern = r'(\|[^\n]+\|\n\|[-:\s|]+\|\n(?:\|[^\n]+\|\n)+)'
        
        tables = list(re.finditer(table_pattern, content))
        
        print(f"Found {len(tables)} markdown tables in the content\n")
        
        # Show first table
        if tables:
            first_table = tables[0]
            table_text = first_table.group(0)
            
            print("First table found:")
            print("-" * 50)
            print(table_text[:500])  # Show first 500 chars
            print("-" * 50)
            
            # Get context around the table
            start = max(0, first_table.start() - 200)
            end = min(len(content), first_table.end() + 200)
            context = content[start:first_table.start()]
            
            # Look for Table caption
            caption_match = re.search(r'Table\s+\d+[:\.]?\s*([^\n]+)', context)
            if caption_match:
                print(f"\nTable caption found: {caption_match.group(0)}")
            
        print("\n=== Marker Output Structure ===")
        print("When Marker processes a PDF, it:")
        print("1. Converts tables to markdown format (as shown above)")
        print("2. Preserves table structure with | separators")
        print("3. Tables are embedded in the text flow")
        print("4. For JSON output with full structure, use:")
        print("   - converter.build_document() to get structured blocks")
        print("   - document.contained_blocks((BlockTypes.Table,)) for tables")
        
    else:
        print(f"Parsed JSON not found: {simple_json}")
        
    # Show how to properly use Marker for table extraction
    print("\n=== Proper Marker Table Extraction ===")
    print("""
To extract tables as structured data with Marker:

1. Use the converter directly:
   ```python
   from marker.converters.pdf import PdfConverter
   from marker.schema import BlockTypes
   
   converter = PdfConverter(artifact_dict=model_dict)
   document = converter.build_document("path/to/pdf")
   
   # Get all table blocks
   tables = document.contained_blocks((BlockTypes.Table,))
   
   for table in tables:
       # Access table HTML representation
       table_html = table.html
       # Access table bounding box/polygon
       table_bbox = table.polygon
   ```

2. Or use JSON output format:
   ```python
   rendered = converter("path/to/pdf", {"output_format": "json"})
   # This gives structured blocks with table metadata
   ```
""")


if __name__ == "__main__":
    explore_marker_output()