"""
Test script for the Paper Extraction Storage System.

This script demonstrates how to use the storage system to store and retrieve
extracted paper knowledge from JSON files.
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

import sys
sys.path.append('..')
from storage import PaperStorageManager

# Load environment variables
load_dotenv()


async def main():
    """Main test function."""
    
    # Get OpenAI API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not found in environment. Vector search may not work.")
    
    # Initialize storage manager
    storage_manager = PaperStorageManager(
        # Neo4j configuration (adjust as needed)
        neo4j_uri="bolt://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="gfll9999",
        neo4j_database="neo4j",
        
        # ChromaDB configuration
        chroma_persist_dir="./chroma_db_papers",
        openai_api_key=openai_api_key,
        embedding_model="text-embedding-3-small"
    )
    
    try:
        # Initialize connections
        print("Initializing storage system...")
        await storage_manager.initialize()
        print("Storage system initialized successfully!")
        
        # Test JSON file path
        json_path = "/home/administrator/projects/NetMind-RS-KnowledgeRAG/paper_sets/paper_set_1/docs/Attention Is All You Need.enhanced_llm_extraction.json"
        
        # Check if file exists
        if not Path(json_path).exists():
            print(f"Error: Test file not found at {json_path}")
            return
        
        # Store the paper
        print(f"\nStoring paper from: {json_path}")
        storage_result = await storage_manager.store_paper(json_path=json_path)
        
        print("\nStorage Results:")
        print(f"  - Graph stored: {storage_result['graph_stored']}")
        print(f"  - Embeddings stored: {storage_result['embeddings_stored']}")
        print(f"  - Entities count: {storage_result['entities_count']}")
        print(f"  - Relationships count: {storage_result['relationships_count']}")
        print(f"  - Chunks count: {storage_result['chunks_count']}")
        
        # Test semantic search
        print("\n" + "="*50)
        print("Testing Semantic Search")
        print("="*50)
        
        test_queries = [
            "Transformer model",
            "attention mechanism",
            "BLEU score",
            "machine translation"
        ]
        
        for query in test_queries:
            print(f"\nSearching for: '{query}'")
            search_results = await storage_manager.search_hybrid(
                query=query,
                search_types=['semantic_entities', 'graph_entities'],
                limit=3
            )
            
            # Display semantic entity results
            if 'semantic_entities' in search_results:
                print(f"\n  Semantic Entity Results (top 3):")
                for i, result in enumerate(search_results['semantic_entities'][:3], 1):
                    print(f"    {i}. {result.entity_name} ({result.entity_type}) - Score: {result.score:.3f}")
                    print(f"       {result.content[:100]}...")
            
            # Display graph entity results
            if 'graph_entities' in search_results:
                print(f"\n  Graph Entity Results (top 3):")
                for i, result in enumerate(search_results['graph_entities'][:3], 1):
                    print(f"    {i}. {result.entity_name} ({result.entity_type}) - Score: {result.score:.3f}")
                    print(f"       {result.content[:100]}...")
        
        # Test relationship search
        print("\n" + "="*50)
        print("Testing Relationship Search")
        print("="*50)
        
        entity_name = "Transformer"
        print(f"\nFinding relationships for entity: '{entity_name}'")
        relationships = await storage_manager.graph_store.find_relationships(
            entity_name=entity_name,
            depth=2
        )
        
        if relationships:
            print(f"Found {len(relationships)} relationship paths:")
            for i, rel in enumerate(relationships[:5], 1):
                path = rel['path']
                rel_types = rel['relationship_types']
                print(f"\n  Path {i} (depth {rel['depth']}):")
                for node, rel_type in zip(path[:-1], rel_types):
                    print(f"    {node['name']} --[{rel_type}]--> ", end="")
                print(f"{path[-1]['name']}")
        else:
            print("No relationships found")
        
        # Get storage statistics
        print("\n" + "="*50)
        print("Storage Statistics")
        print("="*50)
        
        stats = await storage_manager.get_storage_statistics()
        
        print("\nGraph Database:")
        print(f"  - Total papers: {stats['graph_database']['total_papers']}")
        print(f"  - Total entities: {stats['graph_database']['total_entities']}")
        print(f"  - Total relationships: {stats['graph_database']['total_relationships']}")
        print(f"  - Total chunks: {stats['graph_database']['total_chunks']}")
        
        if stats['graph_database']['entities_by_type']:
            print("\n  Entities by type:")
            for entity_type, count in list(stats['graph_database']['entities_by_type'].items())[:5]:
                print(f"    - {entity_type}: {count}")
        
        if stats['graph_database']['relationships_by_type']:
            print("\n  Relationships by type:")
            for rel_type, count in list(stats['graph_database']['relationships_by_type'].items())[:5]:
                print(f"    - {rel_type}: {count}")
        
        print("\nVector Database:")
        for collection, info in stats['vector_database'].items():
            if isinstance(info, dict) and 'count' in info:
                print(f"  - {collection}: {info['count']} embeddings")
        
        # Get paper knowledge graph
        print("\n" + "="*50)
        print("Paper Knowledge Graph")
        print("="*50)
        
        paper_path = storage_result['file_path']
        graph_data = await storage_manager.get_paper_knowledge_graph(paper_path)
        
        if graph_data:
            print(f"\nPaper: {paper_path}")
            print(f"  - Entities: {graph_data['entity_count']}")
            print(f"  - Relationships: {graph_data['relationship_count']}")
            
            print("\n  Sample entities:")
            for entity in graph_data['entities'][:5]:
                print(f"    - {entity['name']} ({entity['type']})")
            
            if graph_data['relationships']:
                print("\n  Sample relationships:")
                for rel in graph_data['relationships'][:5]:
                    print(f"    - {rel['source']} --[{rel['type']}]--> {rel['target']}")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Close connections
        print("\n\nClosing storage system...")
        await storage_manager.close()
        print("Storage system closed.")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())