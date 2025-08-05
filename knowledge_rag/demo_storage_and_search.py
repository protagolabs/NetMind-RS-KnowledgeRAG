#!/usr/bin/env python3
"""
Storage and Search System Demo

Demonstrates the complete Knowledge RAG storage and search pipeline:
1. Load extracted knowledge from previous extraction
2. Store in Neo4j and ChromaDB
3. Perform various types of searches with full document traceability
"""

import sys
import os
import asyncio
import json
from pathlib import Path
from datetime import datetime

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


async def demo_storage_and_search():
    """Demonstrate the storage and search system"""
    
    print("🚀 Knowledge RAG Storage and Search System Demo")
    print("=" * 60)
    
    # Check required environment variables
    required_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Please set these in your .env file to run the demo")
        return False
    
    try:
        # Import modules
        from src.storage import StorageManager
        from src.search import SearchEngine, SearchQuery, SearchType
        
        print("✅ Successfully imported storage and search modules")
        
        # Check if we have extracted knowledge to work with
        kb_output_dir = Path("knowledge_base_output")
        pickle_file = None
        
        if kb_output_dir.exists():
            pickle_files = list(kb_output_dir.glob("knowledge_base.pkl"))
            if pickle_files:
                pickle_file = pickle_files[0]
                print(f"📁 Found knowledge base: {pickle_file}")
            else:
                print("❌ No knowledge base found. Please run build_knowledge_base.py first")
                return False
        else:
            print("❌ Knowledge base output directory not found. Please run build_knowledge_base.py first")
            return False
        
        # Load the knowledge base
        print("\n💾 Loading extracted knowledge base...")
        
        import pickle
        with open(pickle_file, 'rb') as f:
            kb_data = pickle.load(f)
        
        document = kb_data['document']
        entities = kb_data['entities']
        relationships = kb_data['relationships']
        episodes = kb_data['episodes']
        
        print(f"✅ Loaded knowledge base:")
        print(f"   📄 Document: {document.file_name}")
        print(f"   📝 Episodes: {len(episodes)}")
        print(f"   🏷️  Entities: {len(entities)}")
        print(f"   🔗 Relationships: {len(relationships)}")
        
        # Initialize storage manager
        print("\n🗄️  Initializing Storage Manager...")
        
        storage_manager = StorageManager(
            # Neo4j configuration
            neo4j_uri="bolt://localhost:7687",
            neo4j_username="neo4j",
            neo4j_password="password",
            
            # ChromaDB configuration  
            chroma_persist_dir="./demo_chroma_db",
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        try:
            await storage_manager.initialize()
            print("✅ Storage manager initialized successfully")
            
            # Store the knowledge base
            print("\n💾 Storing knowledge base in Neo4j and ChromaDB...")
            
            storage_results = await storage_manager.store_knowledge_base(
                document=document,
                episodes=episodes,
                entities=entities,
                relationships=relationships
            )
            
            print("✅ Knowledge base stored successfully:")
            for key, value in storage_results.items():
                print(f"   {key}: {value}")
            
            # Get storage statistics
            print("\n📊 Storage Statistics:")
            stats = await storage_manager.get_storage_statistics()
            
            print(f"   Graph Database:")
            graph_stats = stats.get('graph_database', {})
            print(f"     • Documents: {graph_stats.get('total_documents', 0)}")
            print(f"     • Episodes: {graph_stats.get('total_episodes', 0)}")
            print(f"     • Entities: {graph_stats.get('total_entities', 0)}")
            print(f"     • Relationships: {graph_stats.get('total_relationships', 0)}")
            
            print(f"   Vector Database:")
            vector_stats = stats.get('vector_database', {})
            entities_count = vector_stats.get('entities', {}).get('count', 0)
            episodes_count = vector_stats.get('episodes', {}).get('count', 0)
            print(f"     • Entity embeddings: {entities_count}")
            print(f"     • Episode embeddings: {episodes_count}")
            
            # Initialize search engine
            print("\n🔍 Initializing Search Engine...")
            
            search_engine = SearchEngine(storage_manager)
            await search_engine.warm_up()
            
            print("✅ Search engine initialized and warmed up")
            
            # Demonstrate different search types
            await demo_searches(search_engine)
            
            # Show search analytics
            print("\n📈 Search Analytics:")
            analytics = search_engine.get_search_analytics()
            print(f"   Total queries: {analytics.get('total_queries', 0)}")
            print(f"   Success rate: {analytics.get('success_rate', 0):.1%}")
            print(f"   Average search time: {analytics.get('average_search_time_ms', 0):.1f}ms")
            print(f"   Average results per query: {analytics.get('average_results_per_query', 0):.1f}")
            
            # Health check
            health = await search_engine.health_check()
            print(f"\n🏥 System Health: {health['status']}")
            if health.get('warnings'):
                for warning in health['warnings']:
                    print(f"   ⚠️  {warning}")
            
        except Exception as e:
            print(f"❌ Storage/Search demo failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            # Clean up
            print("\n🧹 Cleaning up...")
            try:
                await storage_manager.close()
                print("✅ Storage connections closed")
            except Exception as e:
                print(f"⚠️  Cleanup warning: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please make sure you have installed the required dependencies:")
        print("pip install neo4j chromadb")
        return False
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def demo_searches(search_engine):
    """Demonstrate different types of searches"""
    
    print("\n🔍 Search System Demonstrations:")
    print("-" * 40)
    
    # Search test cases
    search_tests = [
        {
            "name": "Hybrid Search",
            "description": "General search combining all methods",
            "query": "knowledge graph construction",
            "search_type": SearchType.HYBRID
        },
        {
            "name": "Entity Search", 
            "description": "Search specifically for entities",
            "query": "document processing system",
            "search_type": SearchType.SEMANTIC_ENTITIES
        },
        {
            "name": "Episode Search",
            "description": "Search through document chunks",
            "query": "temporal knowledge graphs",
            "search_type": SearchType.SEMANTIC_EPISODES
        },
        {
            "name": "Graph Traversal",
            "description": "Find related entities via graph connections",
            "query": "Neo4j ChromaDB",
            "search_type": SearchType.GRAPH_TRAVERSAL
        }
    ]
    
    for i, test in enumerate(search_tests, 1):
        print(f"\n{i}. {test['name']}")
        print(f"   Description: {test['description']}")
        print(f"   Query: '{test['query']}'")
        
        try:
            # Create search query
            query = SearchQuery(
                query_text=test['query'],
                search_types=[test['search_type']],
                limit=5  # Limit results for demo
            )
            
            # Execute search
            results = await search_engine.search(query)
            
            print(f"   ✅ Found {results.total_results} results in {results.search_time_ms:.1f}ms")
            
            if results.results:
                print(f"   📋 Top Results:")
                
                for j, result in enumerate(results.results[:3], 1):
                    score_bar = "█" * int(result.score * 10)
                    print(f"      {j}. [{score_bar:<10}] {result.score:.3f}")
                    print(f"         Content: {result.get_display_content()[:100]}...")
                    print(f"         Source: {result.get_source_context()}")
                    print(f"         Type: {result.result_type}")
                    
                    if result.entity_name:
                        print(f"         Entity: {result.entity_name} ({result.entity_type})")
                    
                    print()
            else:
                print("   ℹ️  No results found")
            
            # Show search strategy breakdown
            if results.results_by_type:
                print(f"   📊 Results by type: {dict(results.results_by_type)}")
            
        except Exception as e:
            print(f"   ❌ Search failed: {e}")
    
    # Demonstrate convenience methods
    print(f"\n🎯 Convenience Search Methods:")
    print("-" * 30)
    
    try:
        # Simple text search
        print("1. Simple text search:")
        simple_results = await search_engine.search_text("RAG system architecture", limit=3)
        print(f"   Found {simple_results.total_results} results")
        
        # Entity-specific search
        print("2. Entity-specific search:")
        entity_results = await search_engine.search_entities("knowledge", limit=3)
        print(f"   Found {entity_results.total_results} entities")
        
        # Episode-specific search
        print("3. Episode-specific search:")
        episode_results = await search_engine.search_episodes("processing pipeline", limit=3)
        print(f"   Found {episode_results.total_results} episodes")
        
    except Exception as e:
        print(f"   ❌ Convenience methods failed: {e}")


async def check_dependencies():
    """Check if required services are available"""
    
    print("🔧 Checking Dependencies:")
    print("-" * 25)
    
    # Check Neo4j
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
        with driver.session() as session:
            result = session.run("RETURN 1")
            result.single()
        driver.close()
        print("✅ Neo4j: Connected")
    except Exception as e:
        print(f"❌ Neo4j: Not available ({e})")
        print("   Please start Neo4j with:")
        print("   docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest")
        return False
    
    # Check OpenAI API key
    if os.getenv('OPENAI_API_KEY'):
        print("✅ OpenAI API key: Found")
    else:
        print("❌ OpenAI API key: Missing")
        print("   Please set OPENAI_API_KEY in your .env file")
        return False
    
    # Check ChromaDB
    try:
        import chromadb
        print("✅ ChromaDB: Available")
    except ImportError:
        print("❌ ChromaDB: Not installed")
        print("   Please install with: pip install chromadb")
        return False
    
    return True


if __name__ == "__main__":
    print("Knowledge RAG - Storage and Search System Demo")
    print("=" * 50)
    
    # Check dependencies first
    if not asyncio.run(check_dependencies()):
        print("\n❌ Dependency check failed. Please resolve the issues above.")
        sys.exit(1)
    
    # Run the demo
    success = asyncio.run(demo_storage_and_search())
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Storage and Search Demo completed successfully!")
        print("\nThis demonstrated:")
        print("  ✅ Neo4j graph database storage")
        print("  ✅ ChromaDB vector embeddings")
        print("  ✅ Hybrid search capabilities")
        print("  ✅ Semantic similarity search")
        print("  ✅ Graph traversal search")
        print("  ✅ Full document traceability")
        print("  ✅ Result fusion and ranking")
        print("  ✅ Search analytics and monitoring")
        print("\nYour Knowledge RAG storage and search system is ready!")
        print("\nNext steps:")
        print("  • Scale up with more documents")
        print("  • Add community detection")
        print("  • Implement temporal search")
        print("  • Build a web interface")
    else:
        print("❌ Storage and Search Demo failed.")
        print("Check the error messages above for troubleshooting.")
    
    sys.exit(0 if success else 1) 