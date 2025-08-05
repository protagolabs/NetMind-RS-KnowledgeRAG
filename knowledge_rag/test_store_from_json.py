#!/usr/bin/env python3
"""
Store Extracted Knowledge from JSON Test Results

Loads the knowledge extraction test results from JSON and stores them
in Neo4j graph database and ChromaDB vector database.
"""

import sys
import os
import asyncio
import json
from pathlib import Path
from datetime import datetime, timezone

# Add the current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed, trying system environment variables")


def load_json_data(json_file_path: str):
    """Load and parse the JSON test results"""
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📄 Loaded test data:")
    print(f"   Session: {data['metadata']['session_id']}")
    print(f"   Episodes tested: {data['metadata']['episodes_tested']}")
    print(f"   Entities: {len(data['entities'])}")
    print(f"   Relationships: {len(data.get('relationships', []))}")
    
    return data


def convert_json_to_storage_objects(json_data):
    """Convert JSON data back to storage-compatible objects"""
    
    from src.knowledge_extraction.models import (
        Entity, Relationship, Document, EntityType, RelationshipType
    )
    from src.file_management.models import Episode
    
    # Convert document
    doc_info = json_data['document_info']
    document = Document(
        uuid=doc_info['uuid'],
        file_path=doc_info['file_path'],
        file_name=doc_info['file_name'],
        file_hash="test_hash",  # Not in JSON, using placeholder
        title=doc_info['title'],
        document_type="markdown",
        created_at=datetime.now(timezone.utc),
        modified_at=datetime.now(timezone.utc)
    )
    
    # Convert entities
    entities = []
    for entity_data in json_data['entities']:
        try:
            entity_type = EntityType(entity_data['entity_type'])
        except ValueError:
            entity_type = EntityType.CONCEPT  # Default fallback
        
        entity = Entity(
            uuid=entity_data['uuid'],
            name=entity_data['name'],
            entity_type=entity_type,
            summary=entity_data['summary'],
            aliases=entity_data['aliases'],
            attributes=entity_data['attributes'],
            confidence=entity_data['confidence'],
            created_at=datetime.fromisoformat(entity_data['created_at'].replace('Z', '+00:00')),
            first_mentioned_at=datetime.fromisoformat(entity_data['first_mentioned_at'].replace('Z', '+00:00')) if entity_data.get('first_mentioned_at') else None,
            last_updated_at=datetime.fromisoformat(entity_data['created_at'].replace('Z', '+00:00')),
            source_episodes=entity_data['source_episodes'],
            source_documents=entity_data['source_documents']
        )
        entities.append(entity)
    
    # Convert relationships
    relationships = []
    for rel_data in json_data.get('relationships', []):
        try:
            rel_type = RelationshipType(rel_data['relationship_type'])
        except ValueError:
            rel_type = RelationshipType.RELATED_TO  # Default fallback
        
        relationship = Relationship(
            uuid=rel_data['uuid'],
            source_entity_id=rel_data['source_entity_id'],
            target_entity_id=rel_data['target_entity_id'],
            relationship_type=rel_type,
            fact=rel_data['fact'],
            confidence=rel_data['confidence'],
            created_at=datetime.fromisoformat(rel_data['created_at'].replace('Z', '+00:00')),
            valid_at=datetime.fromisoformat(rel_data['valid_at'].replace('Z', '+00:00')) if rel_data.get('valid_at') else None,
            invalid_at=datetime.fromisoformat(rel_data['invalid_at'].replace('Z', '+00:00')) if rel_data.get('invalid_at') else None,
            source_episodes=rel_data['source_episodes'],
            source_documents=rel_data['source_documents'],
            attributes=rel_data.get('attributes', {})
        )
        relationships.append(relationship)
    
    # Convert test episodes to full episodes
    episodes = []
    for i, ep_data in enumerate(json_data.get('test_episodes', [])):
        episode = Episode(
            id=ep_data['id'],
            content=ep_data.get('content_preview', '').replace('...', ''),  # Expand if needed
            episode_type=ep_data['episode_type'],
            sequence_number=ep_data['sequence_number'],
            source_file_id=document.uuid,  # Use document UUID as source file ID
            timestamp=datetime.fromisoformat(ep_data['timestamp'].replace('Z', '+00:00')),
            metadata=ep_data.get('metadata', {})
        )
        episodes.append(episode)
        # Add episode to document using the proper API
        document.add_episode(episode.id)
    
    # Update document knowledge statistics using the proper API
    document.update_knowledge_stats(entities, relationships)
    
    return document, entities, relationships, episodes


async def store_knowledge_in_databases(document, entities, relationships, episodes):
    """Store the knowledge in Neo4j and ChromaDB"""
    
    from src.storage import StorageManager
    
    # Initialize storage manager
    storage_manager = StorageManager(
        # Neo4j configuration
        neo4j_uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        neo4j_username=os.getenv('NEO4J_USERNAME', 'neo4j'),
        neo4j_password=os.getenv('NEO4J_PASSWORD', 'password'),
        
        # ChromaDB configuration
        chroma_persist_dir="./test_chroma_db",
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )
    
    try:
        # Initialize connections
        print("\n🔌 Connecting to databases...")
        await storage_manager.initialize()
        print("✅ Connected to Neo4j and ChromaDB")
        
        # Store the knowledge base
        print("\n💾 Storing knowledge in databases...")
        storage_results = await storage_manager.store_knowledge_base(
            document=document,
            episodes=episodes,
            entities=entities,
            relationships=relationships
        )
        
        print("✅ Knowledge stored successfully:")
        for key, value in storage_results.items():
            print(f"   {key}: {value}")
        
        # Get storage statistics
        print("\n📊 Storage Statistics:")
        stats = await storage_manager.get_storage_statistics()
        
        print("   Graph Database (Neo4j):")
        graph_stats = stats.get('graph_database', {})
        print(f"     • Documents: {graph_stats.get('total_documents', 0)}")
        print(f"     • Episodes: {graph_stats.get('total_episodes', 0)}")
        print(f"     • Entities: {graph_stats.get('total_entities', 0)}")
        print(f"     • Relationships: {graph_stats.get('total_relationships', 0)}")
        
        if graph_stats.get('entities_by_type'):
            print("     • Entity types:")
            for entity_type, count in graph_stats['entities_by_type'].items():
                print(f"       - {entity_type}: {count}")
        
        print("   Vector Database (ChromaDB):")
        vector_stats = stats.get('vector_database', {})
        entities_count = vector_stats.get('entities', {}).get('count', 0)
        episodes_count = vector_stats.get('episodes', {}).get('count', 0)
        print(f"     • Entity embeddings: {entities_count}")
        print(f"     • Episode embeddings: {episodes_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to store knowledge: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up connections
        try:
            await storage_manager.close()
            print("\n✅ Database connections closed")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")


async def main():
    """Main function to store knowledge from JSON"""
    
    print("🗄️  Knowledge RAG - Store from JSON Test Results")
    print("=" * 55)
    
    # Check for required environment variables
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("Please set it in your .env file for embedding generation")
        return False
    
    # Find the latest JSON test file
    test_output_dir = Path("test_extraction_output")
    if not test_output_dir.exists():
        print("❌ test_extraction_output directory not found")
        print("Please run test_knowledge_extraction.py first to generate test data")
        return False
    
    json_files = list(test_output_dir.glob("test_extraction_*.json"))
    if not json_files:
        print("❌ No JSON test files found")
        print("Please run test_knowledge_extraction.py first to generate test data")
        return False
    
    # Use the most recent JSON file
    latest_json = max(json_files, key=lambda f: f.stat().st_mtime)
    print(f"📁 Using test file: {latest_json.name}")
    
    try:
        # Load JSON data
        json_data = load_json_data(str(latest_json))
        
        # Convert to storage objects
        print("\n🔄 Converting JSON data to storage objects...")
        document, entities, relationships, episodes = convert_json_to_storage_objects(json_data)
        
        print(f"✅ Converted objects:")
        print(f"   Document: {document.file_name}")
        print(f"   Entities: {len(entities)}")
        print(f"   Relationships: {len(relationships)}")
        print(f"   Episodes: {len(episodes)}")
        
        # Store in databases
        success = await store_knowledge_in_databases(document, entities, relationships, episodes)
        
        return success
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Knowledge RAG - Store Test Results")
    print("=" * 40)
    
    success = asyncio.run(main())
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 Knowledge successfully stored in databases!")
        print("\nNext steps:")
        print("  1. Open Neo4j Browser at http://localhost:7474")
        print("  2. Run visualization queries")
        print("  3. Test search functionality")
        print("\nTo visualize: python test_visualize_neo4j.py")
        print("To search: python test_search_knowledge.py")
    else:
        print("❌ Failed to store knowledge in databases")
        print("Check the error messages above for troubleshooting")
    
    sys.exit(0 if success else 1) 