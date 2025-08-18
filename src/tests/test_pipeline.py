"""
Test script for the PDF to Knowledge Pipeline.
Processes all PDFs in paper_set_1 directory.
Modified to ensure proper storage of llm_extraction to database.
"""

import asyncio
import sys
import json
from pathlib import Path
import glob
from datetime import datetime

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.pdf_to_knowledge_pipeline import PDFToKnowledgePipeline, PipelineConfig
from storage.storage_manager import PaperStorageManager
from storage.models import ExtractedPaper, ExtractedEntity, ExtractedRelationship


async def store_extraction_properly(extraction_json_path: str, storage_manager: PaperStorageManager) -> dict:
    """
    Store extraction using the proven format that creates proper chunk connections.
    
    Args:
        extraction_json_path: Path to the .llm_extraction.json file
        storage_manager: Initialized storage manager
        
    Returns:
        Storage result dictionary
    """
    # Load the extraction data
    with open(extraction_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract paper title from filename
    paper_title = Path(extraction_json_path).stem.replace('.llm_extraction', '')
    
    # Create ExtractedPaper object with proper format
    paper = ExtractedPaper(
        file_path=extraction_json_path.replace('.llm_extraction.json', '.pdf'),
        title=paper_title,
        document_type=data.get("document_type", "academic_paper"),
        schema_used=data.get("schema_used", "academic_paper"),
        entities=[],
        relationships=[],
        chunks=[],
        metadata={
            "paper_id": f"paper_{paper_title.replace(' ', '_').lower()}",
            "extraction_metadata": data.get("extraction_metadata", {})
        }
    )
    
    # Process chunks - create chunk list with proper format
    chunk_ids = set()
    for entity_data in data.get('entities', []):
        chunk_id = entity_data.get('chunk_id', 'unknown')
        chunk_ids.add(chunk_id)
    
    # Create chunks with proper content
    for chunk_id in sorted(chunk_ids):
        # Get all entities in this chunk to build context
        chunk_entities = [e for e in data.get('entities', []) 
                         if e.get('chunk_id') == chunk_id]
        
        # Aggregate unique contexts from entities in this chunk
        contexts = set(e.get('context', '') for e in chunk_entities if e.get('context'))
        chunk_content = " ".join(contexts) if contexts else f"Content for {chunk_id}"
        
        chunk_index = int(chunk_id.split('_')[-1]) if '_' in chunk_id else 0
        
        paper.chunks.append({
            "chunk_id": chunk_id,
            "content": chunk_content,
            "chunk_type": "text",
            "word_count": len(chunk_content.split()),
            "sequence": chunk_index,
            "metadata": {
                "entity_count": len(chunk_entities)
            }
        })
    
    # Process entities with correct field names
    for entity_data in data.get('entities', []):
        entity = ExtractedEntity(
            name=entity_data.get('name', ''),
            type=entity_data.get('type', 'UNKNOWN'),
            description=entity_data.get('description', ''),
            confidence=entity_data.get('confidence', 0.5),
            attributes=entity_data.get('attributes', {}),
            context=entity_data.get('context', ''),
            chunk_id=entity_data.get('chunk_id', 'unknown'),
            source_file=extraction_json_path
        )
        paper.entities.append(entity)
    
    # Process relationships with correct field names
    for rel_data in data.get('relationships', []):
        relationship = ExtractedRelationship(
            source_entity=rel_data.get('source', ''),
            target_entity=rel_data.get('target', ''),
            relationship_type=rel_data.get('type', 'RELATED_TO'),
            confidence=rel_data.get('confidence', 0.5),
            properties={
                'description': rel_data.get('description', ''),
                'evidence': rel_data.get('evidence', '')
            },
            context=rel_data.get('evidence', ''),
            chunk_id=rel_data.get('chunk_id', 'unknown'),
            source_file=extraction_json_path
        )
        paper.relationships.append(relationship)
    
    # Store the paper
    result = await storage_manager.store_paper(paper)
    return result


async def main():
    """Process all PDFs in paper_set_1."""
    
    # Configure pipeline
    config = PipelineConfig(
        llm_model="gpt-3.5-turbo",
        temperature=0.1,
        chunk_size=500,
        chunk_overlap=50,
        max_entities_per_chunk=15,
        max_relationships_per_chunk=10,
        track_costs=True,
        cost_file="api_costs.json",
        llm_max_tokens=2048,
        parse_output_dir="/home/administrator/projects/NetMind-RS-KnowledgeRAG/paper_sets/paper_set_1/parsed_docs"
    )
    
    # Create pipeline (we'll handle storage separately)
    pipeline = PDFToKnowledgePipeline(config)
    
    # Initialize storage manager separately for custom storage
    custom_storage = PaperStorageManager()
    await custom_storage.initialize()
    
    # Get all PDFs in paper_set_1
    #pdf_dir = "/home/administrator/projects/NetMind-RS-KnowledgeRAG/paper_sets/paper_set_1/docs"
    pdf_dir = "/home/administrator/projects/NetMind-RS-KnowledgeRAG/paper_sets/paper_set_1/docs"
    pdf_files = glob.glob(f"{pdf_dir}/*.pdf", recursive=True)
    
    # Option to skip already processed files
    skip_existing = True  # Set to False to reprocess all files
    
    if skip_existing:
        filtered_pdfs = []
        for pdf in pdf_files:
            # Check if extraction file already exists
            extraction_file = Path(pdf).parent / f"{Path(pdf).stem}.enhanced_llm_extraction.json"
            if not extraction_file.exists():
                filtered_pdfs.append(pdf)
            else:
                print(f"⏭️  Skipping (already processed): {Path(pdf).name}")
        pdf_files = filtered_pdfs
    
    print(f"\nFound {len(pdf_files)} PDF files to process")
    print("="*50)
    
    # Track overall results
    successful = 0
    failed = 0
    total_cost = 0.0
    
    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{idx}/{len(pdf_files)}] Processing: {Path(pdf_path).name}")
        print("-"*40)
        
        try:
            # Process the PDF (only parsing and extraction, not storage)
            # We'll handle storage separately with our custom method
            
            # Temporarily disable automatic storage in pipeline
            original_storage = pipeline.storage_manager
            pipeline.storage_manager = None  # Disable automatic storage
            
            results = await pipeline.process_pdf(pdf_path, pipeline.config.parse_output_dir)
            
            # Re-enable storage manager for potential cleanup
            pipeline.storage_manager = original_storage
            
            # Display results
            if results["status"] == "completed" or "extraction" in results:
                print("✅ Extraction: SUCCESS")
                
                # Quick summary
                if "extraction" in results:
                    print(f"   📊 Entities: {results['extraction']['entities']}, Relationships: {results['extraction']['relationships']}")
                    
                    # Now use our custom storage method
                    extraction_file = results['extraction']['output_file']
                    print(f"   💾 Storing with custom method...")
                    
                    try:
                        storage_result = await store_extraction_properly(extraction_file, custom_storage)
                        if storage_result and storage_result.get('graph_stored'):
                            print(f"   ✅ Storage: SUCCESS")
                            print(f"      - Entities stored: {storage_result.get('entities_count', 0)}")
                            print(f"      - Relationships stored: {storage_result.get('relationships_count', 0)}")
                            print(f"      - Chunks created: {storage_result.get('chunks_count', 0)}")
                            successful += 1
                        else:
                            print(f"   ❌ Storage: FAILED")
                            failed += 1
                    except Exception as storage_error:
                        print(f"   ❌ Storage error: {storage_error}")
                        failed += 1
                else:
                    print(f"   ⚠️  No extraction results found")
                    failed += 1
                
                # Cost tracking
                if "cost_tracking" in results:
                    cost = results['cost_tracking']['total_cost']
                    total_cost += cost
                    print(f"   💰 Cost: ${cost:.4f}")
                
                if "processing_time" in results:
                    print(f"   ⏱️  Time: {results['processing_time']:.2f}s")
                
            else:
                print(f"❌ Status: FAILED")
                failed += 1
                if "error" in results:
                    print(f"   Error: {results['error']}")
        
        except Exception as e:
            print(f"❌ Exception: {e}")
            failed += 1
    
    # Final summary
    print("\n" + "="*50)
    print("📊 FINAL SUMMARY")
    print("="*50)
    print(f"Total PDFs processed: {len(pdf_files)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"💰 Total Cost: ${total_cost:.4f}")
    
    if successful > 0:
        print(f"💵 Average Cost per PDF: ${total_cost/successful:.4f}")
    
    # Show detailed cost summary
    print("\n" + "="*50)
    print("📊 Detailed Cost Summary")
    print("-"*40)
    
    summary = pipeline.get_cost_summary()
    if "error" not in summary:
        print(f"Total Runs in History: {summary['total_runs']}")
        print(f"Total Historical Cost: ${summary['total_cost_usd']:.4f}")
        print(f"Average Cost per Run: ${summary['average_cost_per_run']:.4f}")
        print(f"Total Tokens Used: {summary['total_tokens']:,}")
    else:
        print(summary["error"])
    
    # Close connections
    await custom_storage.close()
    if pipeline.storage_manager:
        await pipeline.storage_manager.close()
    print("\n✅ Storage connections closed")
    
    # Verify final storage results
    print("\n" + "="*50)
    print("🔍 VERIFYING FINAL STORAGE")
    print("-"*40)
    
    # Re-open connection for verification
    verify_storage = PaperStorageManager()
    await verify_storage.initialize()
    
    if verify_storage.graph_store and verify_storage.graph_store.driver:
        async with verify_storage.graph_store.driver.session() as session:
            # Check overall stats
            result = await session.run("MATCH (p:Paper) RETURN count(p) as count")
            paper_count = (await result.single())['count']
            
            result = await session.run("MATCH (c:Chunk) RETURN count(c) as count")
            chunk_count = (await result.single())['count']
            
            result = await session.run("MATCH (e:Entity) RETURN count(e) as count")
            entity_count = (await result.single())['count']
            
            result = await session.run("MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity) RETURN count(*) as count")
            connections = (await result.single())['count']
            
            print(f"Total Papers in Neo4j: {paper_count}")
            print(f"Total Chunks: {chunk_count}")
            print(f"Total Entities: {entity_count}")
            print(f"Chunk->Entity connections: {connections}")
            
            # Check for orphaned entities
            result = await session.run("""
                MATCH (e:Entity)
                WHERE NOT (c:Chunk)-[:CONTAINS_ENTITY]->(e)
                RETURN count(e) as count
            """)
            orphaned = (await result.single())['count']
            
            if orphaned > 0:
                print(f"⚠️  Orphaned entities (no chunk connection): {orphaned}")
            else:
                print("✅ All entities properly connected to chunks!")
    
    await verify_storage.close()


if __name__ == "__main__":
    print("🚀 Starting PDF to Knowledge Pipeline Test")
    print("="*50)
    asyncio.run(main())