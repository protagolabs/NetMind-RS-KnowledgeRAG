#!/usr/bin/env python3
"""
Test Search Functionality on Stored Knowledge

Demonstrates comprehensive search capabilities across the stored knowledge graph
including semantic search, graph traversal, and hybrid search with full traceability.
"""

import sys
import os
import asyncio
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


class SearchTester:
    """Comprehensive search testing class"""
    
    def __init__(self):
        self.storage_manager = None
        self.search_engine = None
    
    async def initialize(self):
        """Initialize storage and search systems"""
        
        from src.storage import StorageManager
        from src.search import SearchEngine
        
        # Initialize storage manager
        self.storage_manager = StorageManager(
            # Neo4j configuration
            neo4j_uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            neo4j_username=os.getenv('NEO4J_USERNAME', 'neo4j'),
            neo4j_password=os.getenv('NEO4J_PASSWORD', 'password'),
            
            # ChromaDB configuration
            chroma_persist_dir="./test_chroma_db",
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        await self.storage_manager.initialize()
        
        # Initialize search engine
        self.search_engine = SearchEngine(self.storage_manager)
        await self.search_engine.warm_up()
        
        print("✅ Search system initialized")
    
    async def close(self):
        """Close connections"""
        if self.storage_manager:
            await self.storage_manager.close()
    
    async def test_basic_searches(self):
        """Test basic search functionality"""
        
        print("\n🔍 Basic Search Tests")
        print("-" * 30)
        
        # Test queries based on the JSON data we know exists
        test_queries = [
            "RAG system",
            "knowledge graph", 
            "Neo4j database",
            "document processing",
            "entity extraction",
            "temporal tracking",
            "OpenAI models",
            "ChromaDB vector",
            "search architecture"
        ]
        
        for query in test_queries:
            try:
                print(f"\n🔎 Query: '{query}'")
                
                # Simple text search
                results = await self.search_engine.search_text(query, limit=5)
                
                print(f"   ✅ Found {results.total_results} results in {results.search_time_ms:.1f}ms")
                print(f"   📊 Search strategies: {results.search_strategy_used}")
                print(f"   📈 Average score: {results.average_score:.3f}")
                
                if results.results:
                    print("   🎯 Top results:")
                    for i, result in enumerate(results.results[:2], 1):
                        print(f"      {i}. [{result.score:.3f}] {result.result_type}")
                        print(f"         {result.get_display_content()[:80]}...")
                        print(f"         📄 {result.get_source_context()}")
                
            except Exception as e:
                print(f"   ❌ Search failed: {e}")
    
    async def test_entity_searches(self):
        """Test entity-specific searches"""
        
        print("\n🏷️  Entity Search Tests")
        print("-" * 25)
        
        entity_queries = [
            ("technology entities", ["TECHNOLOGY"]),
            ("all entities", None),
            ("system concepts", ["CONCEPT", "TECHNOLOGY"])
        ]
        
        for query_desc, entity_types in entity_queries:
            try:
                print(f"\n🔎 {query_desc.title()}:")
                
                results = await self.search_engine.search_entities(
                    query_text="knowledge processing system",
                    entity_types=entity_types,
                    limit=5
                )
                
                print(f"   ✅ Found {results.total_results} entities")
                
                if results.results:
                    print("   🏷️  Entities found:")
                    for result in results.results[:3]:
                        print(f"      • {result.entity_name} ({result.entity_type})")
                        print(f"        Score: {result.score:.3f}")
                        print(f"        Content: {result.content[:60]}...")
                        print()
                
            except Exception as e:
                print(f"   ❌ Entity search failed: {e}")
    
    async def test_episode_searches(self):
        """Test episode/chunk searches"""
        
        print("\n📝 Episode Search Tests")
        print("-" * 25)
        
        episode_queries = [
            "system architecture design",
            "database storage methods", 
            "search algorithms implementation",
            "temporal knowledge processing"
        ]
        
        for query in episode_queries:
            try:
                print(f"\n🔎 Query: '{query}'")
                
                results = await self.search_engine.search_episodes(query, limit=3)
                
                print(f"   ✅ Found {results.total_results} episodes")
                
                if results.results:
                    print("   📝 Episodes found:")
                    for result in results.results:
                        print(f"      📄 {result.document_name}")
                        print(f"         Section {result.episode_sequence}: {result.score:.3f}")
                        print(f"         {result.content[:100]}...")
                        print()
                
            except Exception as e:
                print(f"   ❌ Episode search failed: {e}")
    
    async def test_graph_traversal(self):
        """Test graph traversal searches"""
        
        print("\n🕸️  Graph Traversal Tests")
        print("-" * 27)
        
        # Test traversal from known entities
        traversal_tests = [
            {
                "name": "From RAG System",
                "entities": ["RAG", "RAG system"],
                "hops": 2
            },
            {
                "name": "From Neo4j",
                "entities": ["Neo4j"],
                "hops": 1
            },
            {
                "name": "From Technology Stack",
                "entities": ["spaCy", "Transformers", "OpenAI"],
                "hops": 2
            }
        ]
        
        for test in traversal_tests:
            try:
                print(f"\n🔎 {test['name']}:")
                print(f"   Starting entities: {test['entities']}")
                print(f"   Max hops: {test['hops']}")
                
                results = await self.search_engine.graph_traversal_search(
                    entity_names=test['entities'],
                    hops=test['hops'],
                    limit=5
                )
                
                print(f"   ✅ Found {results.total_results} connected entities")
                
                if results.results:
                    print("   🕸️  Connected entities:")
                    for result in results.results:
                        distance = result.metadata.get('distance', 'unknown')
                        print(f"      • {result.entity_name} (distance: {distance})")
                        print(f"        Type: {result.entity_type}")
                        print(f"        Score: {result.score:.3f}")
                        print()
                
            except Exception as e:
                print(f"   ❌ Graph traversal failed: {e}")
    
    async def test_hybrid_search(self):
        """Test hybrid search combining all methods"""
        
        print("\n🔄 Hybrid Search Tests")
        print("-" * 23)
        
        from src.search import SearchQuery, SearchType, RankingStrategy
        
        hybrid_tests = [
            {
                "name": "Knowledge Graph Construction",
                "query": "knowledge graph construction temporal tracking",
                "strategy": RankingStrategy.RRF
            },
            {
                "name": "Document Processing Pipeline", 
                "query": "document processing entity extraction pipeline",
                "strategy": RankingStrategy.WEIGHTED_COMBINATION
            },
            {
                "name": "Search System Architecture",
                "query": "search system vector database Neo4j ChromaDB",
                "strategy": RankingStrategy.GRAPH_CENTRALITY
            }
        ]
        
        for test in hybrid_tests:
            try:
                print(f"\n🔎 {test['name']}:")
                print(f"   Query: '{test['query']}'")
                print(f"   Strategy: {test['strategy'].value}")
                
                # Create comprehensive search query
                search_query = SearchQuery(
                    query_text=test['query'],
                    search_types=[SearchType.HYBRID],
                    ranking_strategy=test['strategy'],
                    limit=8,
                    include_context=True,
                    deduplicate_results=True
                )
                
                results = await self.search_engine.search(search_query)
                
                print(f"   ✅ Found {results.total_results} results in {results.search_time_ms:.1f}ms")
                print(f"   📊 Fusion method: {results.fusion_method}")
                print(f"   📈 Result types: {dict(results.results_by_type)}")
                print(f"   📄 Documents covered: {len(results.source_documents)}")
                
                if results.results:
                    print("   🎯 Top hybrid results:")
                    for i, result in enumerate(results.results[:3], 1):
                        print(f"      {i}. [{result.score:.3f}] {result.result_type} - {result.source_type}")
                        print(f"         {result.get_display_content()[:80]}...")
                        print(f"         📄 {result.get_source_context()}")
                        print()
                
            except Exception as e:
                print(f"   ❌ Hybrid search failed: {e}")
    
    async def test_advanced_search_features(self):
        """Test advanced search features"""
        
        print("\n⚡ Advanced Search Features")
        print("-" * 30)
        
        from src.search import SearchQuery, SearchType
        
        # Test score filtering
        try:
            print("\n🎯 Score Filtering Test:")
            query = SearchQuery(
                query_text="RAG system knowledge graph",
                search_types=[SearchType.HYBRID],
                min_score=0.5,  # Only high-relevance results
                limit=10
            )
            
            results = await self.search_engine.search(query)
            print(f"   ✅ High-relevance results: {results.total_results}")
            
            if results.results:
                scores = [r.score for r in results.results]
                print(f"   📈 Score range: {min(scores):.3f} - {max(scores):.3f}")
                print(f"   📊 Average score: {sum(scores)/len(scores):.3f}")
            
        except Exception as e:
            print(f"   ❌ Score filtering failed: {e}")
        
        # Test entity type filtering
        try:
            print("\n🏷️  Entity Type Filtering:")
            results = await self.search_engine.search_entities(
                query_text="database processing system",
                entity_types=["TECHNOLOGY"],
                limit=5
            )
            
            print(f"   ✅ Technology entities: {results.total_results}")
            if results.results:
                types = [r.entity_type for r in results.results]
                print(f"   📊 Types found: {set(types)}")
            
        except Exception as e:
            print(f"   ❌ Entity filtering failed: {e}")
    
    async def test_search_analytics(self):
        """Test search analytics and performance"""
        
        print("\n📊 Search Analytics")
        print("-" * 20)
        
        try:
            # Get current analytics
            analytics = self.search_engine.get_search_analytics()
            
            print("   📈 Performance Metrics:")
            print(f"      • Total queries: {analytics.get('total_queries', 0)}")
            print(f"      • Success rate: {analytics.get('success_rate', 0):.1%}")
            print(f"      • Average search time: {analytics.get('average_search_time_ms', 0):.1f}ms")
            print(f"      • Average results per query: {analytics.get('average_results_per_query', 0):.1f}")
            
            # Show most used search types
            search_types = analytics.get('most_used_search_types', {})
            if search_types:
                print("   🔍 Most used search types:")
                for search_type, count in search_types.items():
                    print(f"      • {search_type}: {count}")
            
            # Health check
            health = await self.search_engine.health_check()
            print(f"   🏥 System health: {health['status']}")
            
            if health.get('warnings'):
                for warning in health['warnings']:
                    print(f"      ⚠️  {warning}")
            
        except Exception as e:
            print(f"   ❌ Analytics failed: {e}")
    
    async def test_document_traceability(self):
        """Test full document traceability in search results"""
        
        print("\n📄 Document Traceability Test")
        print("-" * 32)
        
        try:
            query = "knowledge graph temporal processing"
            results = await self.search_engine.search_text(query, limit=5)
            
            print(f"   🔎 Query: '{query}'")
            print(f"   ✅ Found {results.total_results} results")
            
            if results.results:
                print("\n   📋 Traceability Details:")
                
                for i, result in enumerate(results.results, 1):
                    print(f"\n      Result {i}:")
                    print(f"         Content: {result.get_display_content()[:60]}...")
                    print(f"         Score: {result.score:.3f}")
                    print(f"         Type: {result.result_type}")
                    print(f"         Source Type: {result.source_type}")
                    
                    # Document info
                    print(f"         📄 Document: {result.document_name}")
                    print(f"         📁 Path: {result.document_path}")
                    print(f"         🔗 UUID: {result.document_uuid}")
                    
                    # Episode info (if available)
                    if result.episode_uuid:
                        print(f"         📝 Episode: Section {result.episode_sequence}")
                        print(f"         🔗 Episode UUID: {result.episode_uuid}")
                    
                    # Entity info (if available)
                    if result.entity_name:
                        print(f"         🏷️  Entity: {result.entity_name} ({result.entity_type})")
                    
                    # Metadata
                    if result.metadata:
                        print(f"         📊 Metadata: {list(result.metadata.keys())}")
                
                # Document coverage analysis
                doc_coverage = results.get_results_by_document()
                print(f"\n   📊 Document Coverage:")
                print(f"      • Total documents accessed: {len(doc_coverage)}")
                
                for doc_uuid, doc_results in list(doc_coverage.items())[:3]:
                    doc_name = doc_results[0].document_name
                    print(f"      • {doc_name}: {len(doc_results)} results")
            
        except Exception as e:
            print(f"   ❌ Traceability test failed: {e}")


async def run_comprehensive_search_tests():
    """Run all search tests"""
    
    print("🔍 Knowledge RAG - Comprehensive Search Tests")
    print("=" * 50)
    
    # Check prerequisites
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("Please set it in your .env file")
        return False
    
    tester = SearchTester()
    
    try:
        # Initialize
        print("🔧 Initializing search system...")
        await tester.initialize()
        
        # Run test suite
        await tester.test_basic_searches()
        await tester.test_entity_searches()
        await tester.test_episode_searches()
        await tester.test_graph_traversal()
        await tester.test_hybrid_search()
        await tester.test_advanced_search_features()
        await tester.test_document_traceability()
        await tester.test_search_analytics()
        
        print("\n🎉 All search tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Search tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up
        try:
            await tester.close()
            print("\n✅ Search system connections closed")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")


async def interactive_search_demo():
    """Interactive search demonstration"""
    
    print("\n🎮 Interactive Search Demo")
    print("-" * 27)
    print("Enter queries to search the knowledge base (or 'quit' to exit)")
    
    tester = SearchTester()
    
    try:
        await tester.initialize()
        
        while True:
            try:
                query = input("\n🔎 Enter search query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                print(f"   Searching for: '{query}'...")
                
                # Perform search
                results = await tester.search_engine.search_text(query, limit=3)
                
                print(f"   ✅ Found {results.total_results} results in {results.search_time_ms:.1f}ms")
                
                if results.results:
                    for i, result in enumerate(results.results, 1):
                        print(f"\n   {i}. [{result.score:.3f}] {result.result_type}")
                        print(f"      {result.get_display_content()[:120]}...")
                        print(f"      📄 {result.get_source_context()}")
                else:
                    print("   ℹ️  No results found")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"   ❌ Search error: {e}")
        
    except Exception as e:
        print(f"❌ Interactive demo failed: {e}")
        
    finally:
        await tester.close()
        print("\n👋 Interactive search demo ended")


def main():
    """Main function with options"""
    
    print("🔍 Knowledge RAG - Search Testing")
    print("=" * 35)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        # Run interactive demo
        asyncio.run(interactive_search_demo())
    else:
        # Run comprehensive tests
        success = asyncio.run(run_comprehensive_search_tests())
        
        print("\n" + "=" * 35)
        if success:
            print("🎉 Search system testing completed!")
            print("\nThis demonstrated:")
            print("  ✅ Basic text search")
            print("  ✅ Entity-specific search")
            print("  ✅ Episode/chunk search")
            print("  ✅ Graph traversal search")
            print("  ✅ Hybrid search with result fusion")
            print("  ✅ Advanced filtering and analytics")
            print("  ✅ Full document traceability")
            print("\nTo try interactive search:")
            print("  python test_search_knowledge.py --interactive")
        else:
            print("❌ Search testing failed")
            print("Make sure you've run test_store_from_json.py first")
        
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 