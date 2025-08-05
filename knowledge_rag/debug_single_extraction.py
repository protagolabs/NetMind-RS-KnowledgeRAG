#!/usr/bin/env python3
"""
Debug Script for Single Entity Extraction
Processes just one episode to see detailed logging and debug information
"""

import logging
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Add project root to Python path
import sys
sys.path.append(str(Path(__file__).parent))

from src.knowledge_extraction.extractor import KnowledgeExtractor
from src.file_management.models import Episode
from src.knowledge_extraction.models import Document
from src.cost_tracking.cost_tracker import CostTracker

def test_single_extraction():
    """Test entity extraction on a single episode with full debug logging"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Starting single extraction debug test...")
    
    # Initialize cost tracker
    cost_tracker = CostTracker()
    
    # Initialize extractor with gpt-4o instead of o4-mini
    extractor = KnowledgeExtractor(
        model_name="gpt-4o-2024-08-06",
        temperature=0.1,
        max_retries=3,
        cost_tracker=cost_tracker,
        reasoning_effort="high"  # Will be ignored for non-o-series models
    )
    
    # Create a simple test episode
    test_content = """
    The Transformer architecture introduced by Vaswani et al. in 2017 revolutionized natural language processing.
    The paper, titled "Attention Is All You Need", proposed a novel attention mechanism that eliminated the need
    for recurrent neural networks (RNNs) and convolutional neural networks (CNNs) in sequence-to-sequence tasks.
    
    The key innovation was the self-attention mechanism, which allows the model to weigh the importance of different
    words in a sequence when processing each word. This enables parallel processing and better capture of long-range
    dependencies compared to RNNs.
    
    Google Brain and University of Toronto collaborated on this research, with Ashish Vaswani as the lead author.
    The architecture consists of an encoder-decoder structure with multi-head attention layers and feed-forward networks.
    """
    
    episode = Episode(
        id="test_episode_001",
        content=test_content,
        source_file_id="test_document",
        episode_type="content",
        sequence_number=1,
        timestamp=datetime.now(),
        metadata={"test": True}
    )
    
    document = Document(
        title="Test Document",
        file_path="test.txt",
        file_name="test.txt",
        document_type="text",
        file_hash="test_hash"
    )
    
    print(f"📄 Test episode created: {len(test_content)} characters")
    print(f"🤖 Using model: {extractor.model_name}")
    print(f"📁 Debug directory: {extractor.debug_dir.absolute()}")
    
    # Extract entities
    try:
        print("\n🧠 Starting entity extraction...")
        result = extractor.extract_from_episode(episode, document)
        
        print(f"\n✅ Extraction completed!")
        print(f"   - Success: {result.success}")
        print(f"   - Entities: {len(result.entities)}")
        print(f"   - Relationships: {len(result.relationships)}")
        print(f"   - Processing time: {result.processing_time:.2f}s")
        
        if result.entities:
            print("\n📝 Extracted entities:")
            for i, entity in enumerate(result.entities, 1):
                print(f"   {i}. {entity.name} ({entity.entity_type.value}) - Confidence: {entity.confidence}")
        
        if result.relationships:
            print("\n🔗 Extracted relationships:")
            for i, rel in enumerate(result.relationships, 1):
                print(f"   {i}. {rel.fact} ({rel.relationship_type.value}) - Confidence: {rel.confidence}")
        
        # Show debug files created
        print(f"\n💾 Debug files created in: {extractor.debug_dir.absolute()}")
        debug_files = list(extractor.debug_dir.glob("*"))
        for file in sorted(debug_files):
            print(f"   - {file.name}")
        
        # Show cost tracking
        cost_stats = cost_tracker.get_session_stats()
        if cost_stats:
            print(f"\n💰 Cost tracking:")
            print(f"   - Total cost: ${cost_stats.get('total_cost', 0):.4f}")
            print(f"   - API calls: {cost_stats.get('total_api_calls', 0)}")
            print(f"   - Total tokens: {cost_stats.get('total_tokens', 0)}")
        
        if result.error_message:
            print(f"\n❌ Error: {result.error_message}")
            
    except Exception as e:
        print(f"\n❌ Extraction failed: {e}")
        print(f"💾 Check debug files in: {extractor.debug_dir.absolute()}")
        raise

if __name__ == "__main__":
    test_single_extraction() 